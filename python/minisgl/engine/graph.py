from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.models.outputs import ModelForwardOutput, SampleOutputSpec
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    sample_outputs: Dict[str, torch.Tensor]

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        sample_output_specs: Tuple[SampleOutputSpec, ...],
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            sample_outputs={
                spec.name: torch.empty((bs,) + spec.shape, dtype=spec.dtype, device=device)
                for spec in sample_output_specs
            },
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb < 2:
            logger.info_rank0(
                "Disabling CUDA graphs because only %.2f GiB is free after initialization.",
                free_memory_gb,
            )
            return []
        if free_memory_gb < 4:
            cuda_graph_max_bs = 4
        elif free_memory_gb < 8:
            cuda_graph_max_bs = 16
        elif free_memory_gb < 16:
            cuda_graph_max_bs = 32
        elif free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    base = [1, 2, 4]
    return [bs for bs in base if bs <= cuda_graph_max_bs] + list(
        range(8, cuda_graph_max_bs + 1, 8)
    )


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        sample_output_specs: Tuple[SampleOutputSpec, ...],
        dummy_req: Req,
    ) -> None:
        if not getattr(model, "supports_cuda_graph", True):
            cuda_graph_bs = []
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.sample_output_specs = sample_output_specs
        self._all_sample_output_names = tuple(spec.name for spec in sample_output_specs)
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[Tuple[bool, int], torch.cuda.CUDAGraph] = {}
        self.buffer_map: Dict[bool, GraphCaptureBuffer] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        modes = [False]
        if self._all_sample_output_names and free_memory >= (2 << 30):
            modes.append(True)
        elif self._all_sample_output_names:
            logger.info_rank0(
                "Skipping CUDA graph capture for sample-output batches due to low free memory: %s",
                mem_GB(free_memory),
            )
        for wants_outputs in modes:
            self.buffer_map[wants_outputs] = GraphCaptureBuffer.init(
                self.max_graph_bs,
                vocab_size,
                self.device,
                self.sample_output_specs,
            )

        pbar = tqdm(
            [(wants_outputs, bs) for wants_outputs in modes for bs in sorted(self.graph_bs_list, reverse=True)],
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for wants_outputs, bs in pbar:
            free_memory = get_free_memory(self.device)
            mode_name = "outputs" if wants_outputs else "logits"
            pbar.desc = (
                f"Capturing graphs: mode={mode_name:<7} bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            )
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            batch.requested_sample_outputs = self._all_sample_output_names if wants_outputs else ()
            self.attn_backend.prepare_for_capture(batch)
            buffer = self.buffer_map[wants_outputs]
            buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                warmup_output = model.forward()
                buffer.logits[:bs] = warmup_output.logits
                for name, tensor in warmup_output.sample_outputs.items():
                    buffer.sample_outputs[name][:bs] = tensor
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    capture_output = model.forward()
                    buffer.logits[:bs] = capture_output.logits
                    for name, tensor in capture_output.sample_outputs.items():
                        buffer.sample_outputs[name][:bs] = tensor
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[(wants_outputs, bs)] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if (
            not batch.allow_cuda_graph
            or batch.has_hooked_requests
            or not batch.is_decode
            or batch.size > self.max_graph_bs
        ):
            return False
        if not batch.sample_next_token or batch.requested_topk_k > 0:
            return False
        wants_outputs = bool(batch.requested_sample_outputs)
        if wants_outputs and not self._all_sample_output_names:
            return False
        if (wants_outputs, batch.padded_size) not in self.graph_map:
            return False
        return True

    def replay(self, batch: Batch) -> ModelForwardOutput:
        assert self.can_use_cuda_graph(batch)
        wants_outputs = bool(batch.requested_sample_outputs)
        buffer = self.buffer_map[wants_outputs]
        buffer.copy_from(batch)
        g = self.graph_map[(wants_outputs, batch.padded_size)]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return ModelForwardOutput(
            logits=buffer.logits[: batch.size],
            sample_outputs={
                name: tensor[: batch.size]
                for name, tensor in buffer.sample_outputs.items()
                if name in batch.requested_sample_outputs
            },
        )

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        del self.buffer_map
        gc.collect()
