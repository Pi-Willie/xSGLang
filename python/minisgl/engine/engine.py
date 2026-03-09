from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, clear_global_ctx, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, reset_tp_info, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.models.lora import LoRAManager
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


def _ensure_runtime_cache_dirs() -> None:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".minisgl_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "minisgl-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(fallback)


@dataclass
class ForwardOutput:
    next_tokens_gpu: torch.Tensor | None
    next_tokens_cpu: torch.Tensor | None
    copy_done_event: torch.cuda.Event
    sample_outputs_cpu: Dict[str, torch.Tensor] = field(default_factory=dict)
    topk_ids_cpu: torch.Tensor | None = None
    topk_logprobs_cpu: torch.Tensor | None = None


class Engine:
    def __init__(self, config: EngineConfig):
        _ensure_runtime_cache_dirs()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available to PyTorch in this environment. "
                "XSGLang currently requires a working CUDA runtime for engine execution."
            )
        if torch.cuda.is_initialized():
            logger.debug("CUDA already initialized before Engine startup.")
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        output_search_paths = []
        for path in (config.resolved_lora_path, config.model_path, config.resolved_model_path):
            if path is None or path in output_search_paths:
                continue
            output_search_paths.append(path)
        self.model.configure_output_manager(
            hidden_size=config.model_config.hidden_size,
            hidden_dtype=config.dtype,
            search_paths=tuple(output_search_paths),
            device=self.device,
        )
        self.ctx.state_cache = self.state_cache = self.model.create_state_cache(
            num_tables=config.max_running_req + 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.lora_manager = LoRAManager(
            model=self.model,
            model_config=config.model_config,
            base_model_path=config.resolved_model_path,
        )
        if config.resolved_lora_path is not None:
            self.lora_manager.load(config.resolved_lora_path)

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=post_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            sample_output_specs=self.model.sample_output_specs(),
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_weight(config.resolved_model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_kv_cache_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            reserve_bytes = min(
                max(1 << 30, int(0.2 * old_free_memory)),
                max(256 << 20, new_free_memory // 2),
            )
            available_memory = max(0, int(config.memory_ratio * new_free_memory) - reserve_bytes)
            logger.info_rank0(
                "Reserving %s of free GPU memory as runtime headroom before KV allocation.",
                mem_GB(reserve_bytes),
            )
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if batch.has_hooked_requests:
                model_output = self.model.forward_with_hooks()
            elif self.graph_runner.can_use_cuda_graph(batch):
                model_output = self.graph_runner.replay(batch)
            else:
                model_output = self.model.forward()
        logits = self._apply_logit_processors(batch, model_output.logits)
        sample_outputs_cpu = {
            name: model_output.sample_outputs[name][: batch.size].to("cpu", non_blocking=True)
            for name in batch.requested_sample_outputs
            if name in model_output.sample_outputs
        }
        topk_ids_cpu = None
        topk_logprobs_cpu = None
        if batch.requested_topk_k > 0:
            topk_source = logits[: batch.size]
            if batch.requested_topk_logprobs:
                topk_logprobs, topk_ids = torch.topk(
                    torch.log_softmax(topk_source.float(), dim=-1),
                    k=batch.requested_topk_k,
                    dim=-1,
                )
                topk_ids_cpu = topk_ids.to("cpu", non_blocking=True)
                topk_logprobs_cpu = topk_logprobs.to("cpu", non_blocking=True)
            elif batch.requested_topk_ids:
                topk_ids = torch.topk(topk_source, k=batch.requested_topk_k, dim=-1).indices
                topk_ids_cpu = topk_ids.to("cpu", non_blocking=True)

        next_tokens_gpu = None
        next_tokens_cpu = None
        if batch.sample_next_token:
            for req in batch.reqs:
                req.complete_one()
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        elif batch.is_prefill:
            # Prefill-only blocks warm the KV cache without consuming the first decode slot.
            for req in batch.reqs:
                req.cached_len = req.device_len

        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(
            next_tokens_gpu=next_tokens_gpu,
            next_tokens_cpu=next_tokens_cpu,
            sample_outputs_cpu=sample_outputs_cpu,
            topk_ids_cpu=topk_ids_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            copy_done_event=copy_done_event,
        )

    @staticmethod
    def _apply_logit_processors(batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        if not batch.has_logit_processors:
            return logits

        processors = [req.logit_processor for req in batch.reqs]
        active = [p for p in processors if p is not None]
        if len(active) == 0:
            return logits

        # Fast path when all requests use the same processor.
        if len(active) == len(processors) and len({id(p) for p in active}) == 1:
            result = active[0](logits)  # type: ignore[misc]
            return logits if result is logits else result

        for i, proc in enumerate(processors):
            if proc is None:
                continue
            src = logits[i : i + 1]
            dst = proc(src)
            if dst is not src:
                logits[i : i + 1].copy_(dst)
        return logits

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        clear_global_ctx(self.ctx)
        torch.distributed.destroy_process_group()
        destroy_distributed()
        reset_tp_info()
        torch.cuda.empty_cache()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
