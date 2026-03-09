from __future__ import annotations

from collections import OrderedDict
import glob
import json
import os
import re
from dataclasses import dataclass
import time
from typing import Dict, Iterable, Tuple

import torch
from minisgl.distributed import get_tp_info
from minisgl.models.config import ModelConfig
from minisgl.utils import cached_load_hf_config, div_even, init_logger, resolve_model_paths
from tqdm.asyncio import tqdm

logger = init_logger(__name__)

_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_PREFIXES = ("base_model.model.",)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


@dataclass(frozen=True)
class LoRABundle:
    source_path: str
    resolved_path: str
    config: Dict
    tensors: Dict[str, torch.Tensor]


def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    prepared = tensor.contiguous()
    if not prepared.is_cpu or prepared.is_pinned():
        return prepared
    try:
        pinned = torch.empty_like(prepared, device="cpu", pin_memory=True)
        pinned.copy_(prepared, non_blocking=False)
        return pinned
    except RuntimeError:
        return prepared


def _snapshot_download_adapter(adapter_path: str) -> str:
    if os.path.isdir(adapter_path):
        return adapter_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        adapter_path,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model*.safetensors",
            "adapter_model.bin",
        ],
        tqdm_class=DisabledTqdm,
    )


def load_lora_bundle(adapter_path: str) -> LoRABundle:
    folder = _snapshot_download_adapter(adapter_path)
    config_path = os.path.join(folder, "adapter_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"LoRA adapter config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tensors: Dict[str, torch.Tensor] = {}
    safetensor_files = sorted(glob.glob(os.path.join(folder, "adapter_model*.safetensors")))
    if safetensor_files:
        import safetensors

        for file in safetensor_files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensors[name] = _prepare_tensor(handle.get_tensor(name))
    else:
        bin_path = os.path.join(folder, "adapter_model.bin")
        if not os.path.exists(bin_path):
            raise ValueError(f"No adapter weights found under {folder}")
        loaded = torch.load(bin_path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise ValueError(f"Unexpected LoRA checkpoint payload type: {type(loaded)}")
        tensors = {name: _prepare_tensor(tensor) for name, tensor in loaded.items()}

    return LoRABundle(
        source_path=folder if os.path.isdir(adapter_path) else adapter_path,
        resolved_path=folder,
        config=config,
        tensors=tensors,
    )


def _strip_prefix(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _pattern_lookup(pattern: Dict[str, int | float], module_name: str, default: int | float) -> int | float:
    if module_name in pattern:
        return pattern[module_name]
    for key, value in pattern.items():
        if module_name.endswith(key):
            return value
    return default


def iter_lora_pairs(bundle: LoRABundle) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor, float]]:
    config = bundle.config
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Unsupported PEFT type: {config.get('peft_type')}")

    rank_default = int(config.get("r", 0))
    alpha_default = float(config.get("lora_alpha", 0.0))
    rank_pattern = {str(k): int(v) for k, v in config.get("rank_pattern", {}).items()}
    alpha_pattern = {str(k): float(v) for k, v in config.get("alpha_pattern", {}).items()}
    use_rslora = bool(config.get("use_rslora", False))

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in bundle.tensors.items():
        if name.endswith(_LORA_A_SUFFIX):
            pairs.setdefault(_strip_prefix(name[: -len(_LORA_A_SUFFIX)]), {})["A"] = tensor
        elif name.endswith(_LORA_B_SUFFIX):
            pairs.setdefault(_strip_prefix(name[: -len(_LORA_B_SUFFIX)]), {})["B"] = tensor
        elif "lora_" in name:
            raise ValueError(f"Unsupported LoRA tensor entry: {name}")

    for module_name, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA pair for module: {module_name}")
        rank = int(_pattern_lookup(rank_pattern, module_name, rank_default))
        alpha = float(_pattern_lookup(alpha_pattern, module_name, alpha_default))
        if rank <= 0:
            raise ValueError(f"Invalid LoRA rank for module {module_name}: {rank}")
        scale = alpha / (rank**0.5 if use_rslora else rank)
        yield module_name, pair["A"], pair["B"], scale


class LoRAManager:
    def __init__(
        self,
        *,
        model,
        model_config: ModelConfig,
        base_model_path: str,
        cached_adapter_limit: int = 2,
    ) -> None:
        self._params = model.state_dict()
        self._config = model_config
        self._base_model_path = resolve_model_paths(base_model_path).model_path
        self._active_bundle: LoRABundle | None = None
        self._bundle_cache: OrderedDict[str, LoRABundle] = OrderedDict()
        self._cached_adapter_limit = max(0, cached_adapter_limit)

    @property
    def active_adapter(self) -> str | None:
        if self._active_bundle is None:
            return None
        return self._active_bundle.source_path

    def load(self, adapter_path: str) -> str:
        bundle_fetch_start = time.perf_counter()
        bundle = self._get_bundle(adapter_path)
        bundle_fetch_time = time.perf_counter() - bundle_fetch_start

        if self._active_bundle is not None and self._active_bundle.resolved_path == bundle.resolved_path:
            logger.info(
                "LoRA adapter %s is already active; skipping reload (lookup %.3fs).",
                bundle.source_path,
                bundle_fetch_time,
            )
            return bundle.source_path

        base_hint = bundle.config.get("base_model_name_or_path")
        if self._base_hint_mismatches(base_hint):
            logger.warning(
                "Loading LoRA adapter trained for %s on top of base model %s",
                base_hint,
                self._base_model_path,
            )
        if self._active_bundle is not None:
            self.unload()
        apply_start = time.perf_counter()
        self._apply_bundle(bundle, sign=1.0)
        apply_time = time.perf_counter() - apply_start
        self._active_bundle = bundle
        logger.info(
            "Loaded LoRA adapter %s (resolve/cache %.3fs, apply %.3fs).",
            bundle.source_path,
            bundle_fetch_time,
            apply_time,
        )
        return bundle.source_path

    def unload(self) -> None:
        if self._active_bundle is None:
            return
        apply_start = time.perf_counter()
        self._apply_bundle(self._active_bundle, sign=-1.0)
        apply_time = time.perf_counter() - apply_start
        self._remember_bundle(self._active_bundle)
        logger.info(
            "Unloaded LoRA adapter %s (apply %.3fs).",
            self._active_bundle.source_path,
            apply_time,
        )
        self._active_bundle = None

    def _get_bundle(self, adapter_path: str) -> LoRABundle:
        resolved_path = _snapshot_download_adapter(adapter_path)
        if self._active_bundle is not None and self._active_bundle.resolved_path == resolved_path:
            return self._active_bundle

        bundle = self._bundle_cache.pop(resolved_path, None)
        if bundle is None:
            return load_lora_bundle(adapter_path)

        # Mark as most recently used.
        self._bundle_cache[resolved_path] = bundle
        return bundle

    def _remember_bundle(self, bundle: LoRABundle) -> None:
        if self._cached_adapter_limit <= 0:
            self._bundle_cache.clear()
            return
        self._bundle_cache.pop(bundle.resolved_path, None)
        self._bundle_cache[bundle.resolved_path] = bundle
        while len(self._bundle_cache) > self._cached_adapter_limit:
            self._bundle_cache.popitem(last=False)

    def _base_hint_mismatches(self, base_hint: object) -> bool:
        if not isinstance(base_hint, str) or len(base_hint) == 0:
            return False
        try:
            resolved_hint = resolve_model_paths(base_hint).model_path
            if os.path.abspath(resolved_hint) == os.path.abspath(self._base_model_path):
                return False
            current_cfg = cached_load_hf_config(self._base_model_path).to_dict()
            hint_cfg = cached_load_hf_config(resolved_hint).to_dict()
            keys = (
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "vocab_size",
                "tie_word_embeddings",
            )
            return any(current_cfg.get(key) != hint_cfg.get(key) for key in keys)
        except Exception:
            return os.path.basename(base_hint) != os.path.basename(self._base_model_path)

    def _apply_bundle(self, bundle: LoRABundle, *, sign: float) -> None:
        with torch.no_grad():
            for module_name, a_cpu, b_cpu, scale in iter_lora_pairs(bundle):
                self._apply_module_delta(
                    module_name=module_name,
                    a_cpu=a_cpu,
                    b_cpu=b_cpu,
                    scale=sign * scale,
                )

    def _apply_module_delta(
        self,
        *,
        module_name: str,
        a_cpu: torch.Tensor,
        b_cpu: torch.Tensor,
        scale: float,
    ) -> None:
        tp = get_tp_info()
        cfg = self._config

        def _to_target(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return tensor.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=tensor.is_pinned(),
            )

        if module_name == "lm_head":
            target = self._params["lm_head.weight"]
            rows = target.shape[0]
            start = tp.rank * rows
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + rows], target)
            target.add_(b @ a, alpha=scale)
            return
        if module_name == "model.embed_tokens":
            target = self._params["model.embed_tokens.weight"]
            rows = target.shape[0]
            start = tp.rank * rows
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + rows], target)
            target.add_(b @ a, alpha=scale)
            return

        layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", module_name)
        if not layer_match:
            raise ValueError(f"Unsupported LoRA module target: {module_name}")
        layer_idx = int(layer_match.group(1))
        suffix = layer_match.group(2)

        q_local = div_even(cfg.num_qo_heads, tp.size) * cfg.head_dim
        kv_local = div_even(cfg.num_kv_heads, tp.size) * cfg.head_dim
        mlp_local = div_even(cfg.intermediate_size, tp.size)
        is_qwen3_5 = cfg.model_type == "qwen3_5_text"
        q_proj_local = q_local * 2 if is_qwen3_5 else q_local

        if suffix == "self_attn.q_proj":
            if is_qwen3_5:
                full_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                target = self._params[full_key][:q_proj_local]
            else:
                target = self._params[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"][:q_local]
            start = tp.rank * q_proj_local
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + target.shape[0]], target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "self_attn.k_proj":
            if is_qwen3_5:
                full_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                target = self._params[full_key][q_proj_local : q_proj_local + kv_local]
            else:
                full_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                target = self._params[full_key][q_local : q_local + kv_local]
            start = tp.rank * kv_local
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + kv_local], target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "self_attn.v_proj":
            if is_qwen3_5:
                full_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                target = self._params[full_key][
                    q_proj_local + kv_local : q_proj_local + 2 * kv_local
                ]
            else:
                full_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                target = self._params[full_key][q_local + kv_local : q_local + 2 * kv_local]
            start = tp.rank * kv_local
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + kv_local], target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "self_attn.o_proj":
            target = self._params[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            cols = target.shape[1]
            start = tp.rank * cols
            a = _to_target(a_cpu[:, start : start + cols], target)
            b = _to_target(b_cpu, target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "mlp.gate_proj":
            full_key = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
            target = self._params[full_key][:mlp_local]
            start = tp.rank * mlp_local
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + mlp_local], target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "mlp.up_proj":
            full_key = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
            target = self._params[full_key][mlp_local : 2 * mlp_local]
            start = tp.rank * mlp_local
            a = _to_target(a_cpu, target)
            b = _to_target(b_cpu[start : start + mlp_local], target)
            target.add_(b @ a, alpha=scale)
            return
        if suffix == "mlp.down_proj":
            target = self._params[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            cols = target.shape[1]
            start = tp.rank * cols
            a = _to_target(a_cpu[:, start : start + cols], target)
            b = _to_target(b_cpu, target)
            target.add_(b @ a, alpha=scale)
            return

        raise ValueError(f"Unsupported LoRA target module: {module_name}")
