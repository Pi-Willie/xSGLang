from __future__ import annotations

import functools
import glob
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from .logger import init_logger

logger = init_logger(__name__)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
_QUANTIZED_SUFFIXES = (
    "-unsloth-bnb-4bit",
    "-unsloth-bnb-8bit",
    "-bnb-4bit",
    "-bnb-8bit",
    "-4bit",
    "-8bit",
    "-unsloth",
)
_MODEL_PREFETCH_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


@dataclass(frozen=True)
class ResolvedModelPaths:
    model_path: str
    lora_path: str | None
    tokenizer_path: str


@dataclass(frozen=True)
class LocalModelCandidate:
    repo_id: str
    path: str
    config: dict[str, Any]


def _hf_snapshot_download(path: str, *, allow_patterns: Iterable[str] | None = None) -> str:
    # If hf_transfer is installed, the hub client can use its faster transfer path.
    # We enable it opportunistically instead of making it mandatory.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        try:
            import hf_transfer  # noqa: F401
        except Exception:
            pass
        else:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    kwargs: dict[str, Any] = {
        "tqdm_class": DisabledTqdm,
        "max_workers": max(8, min(32, os.cpu_count() or 8)),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    return snapshot_download(path, **kwargs)


def _repo_name(path: str) -> str:
    if os.path.isdir(path):
        return os.path.basename(os.path.abspath(path))
    return path.rsplit("/", 1)[-1]


def _path_has_required_files(path: str, required_patterns: Iterable[str] | None) -> bool:
    if not os.path.isdir(path):
        return False
    if required_patterns is None:
        return True
    return all(glob.glob(os.path.join(path, pattern)) for pattern in required_patterns)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _strip_quantized_suffix(name: str) -> str:
    lowered = name.lower()
    while True:
        for suffix in _QUANTIZED_SUFFIXES:
            if lowered.endswith(suffix):
                name = name[: -len(suffix)]
                lowered = name.lower()
                break
        else:
            return name


def _resolve_metadata_dir(path: str, *, allow_patterns: Iterable[str]) -> str | None:
    if os.path.isdir(path):
        return path
    local_snapshot = _resolve_local_cached_repo_snapshot(path)
    if local_snapshot is not None:
        return local_snapshot
    try:
        return _hf_snapshot_download(path, allow_patterns=allow_patterns)
    except Exception:
        return None


def _load_optional_json(path: str, filename: str) -> tuple[str, dict[str, Any]] | None:
    folder = _resolve_metadata_dir(path, allow_patterns=[filename])
    if folder is None:
        return None
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return folder, json.load(f)


def _load_raw_config_dict(model_path: str) -> dict[str, Any] | None:
    loaded = _load_optional_json(model_path, "config.json")
    return None if loaded is None else loaded[1]


def _config_signature(config: PretrainedConfig | dict[str, Any]) -> tuple[Any, ...]:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return (
        data.get("model_type"),
        tuple(data.get("architectures", [])),
        data.get("hidden_size"),
        data.get("num_hidden_layers"),
        data.get("num_attention_heads"),
        data.get("num_key_value_heads", data.get("num_attention_heads")),
        data.get("head_dim"),
        data.get("intermediate_size"),
        data.get("vocab_size"),
        bool(data.get("tie_word_embeddings", False)),
        data.get("rope_theta"),
    )


def _is_bitsandbytes_quantized(config: PretrainedConfig | dict[str, Any]) -> bool:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    quant = data.get("quantization_config") or {}
    method = str(quant.get("quant_method", "")).lower()
    return (
        method == "bitsandbytes"
        or bool(quant.get("load_in_4bit"))
        or bool(quant.get("_load_in_4bit"))
        or bool(quant.get("load_in_8bit"))
        or bool(quant.get("_load_in_8bit"))
    )


def _candidate_score(source_path: str, candidate: LocalModelCandidate) -> float:
    source_name = _normalize_name(_strip_quantized_suffix(_repo_name(source_path)))
    candidate_name = _normalize_name(candidate.repo_id.rsplit("/", 1)[-1])
    if not source_name or not candidate_name:
        return 0.0
    score = SequenceMatcher(None, source_name, candidate_name).ratio()
    if candidate_name == source_name:
        score += 1.0
    elif source_name in candidate_name:
        score += 0.5
    return score


def _resolve_local_cached_repo_snapshot(
    path: str, *, required_patterns: Iterable[str] | None = None
) -> str | None:
    normalized_path = path.strip()
    if not normalized_path or os.path.isdir(normalized_path):
        return None

    for candidate in _iter_local_model_candidates():
        if candidate.repo_id == normalized_path and _path_has_required_files(
            candidate.path, required_patterns
        ):
            return candidate.path
    return None


def _resolve_local_or_cached_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isdir(path):
        return path
    return _resolve_local_cached_repo_snapshot(path) or path


@functools.cache
def _iter_local_model_candidates() -> tuple[LocalModelCandidate, ...]:
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )
    if not cache_root.is_dir():
        return ()

    candidates: list[LocalModelCandidate] = []
    for model_dir in sorted(cache_root.glob("models--*")):
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if not snapshots:
            continue
        # Newest snapshot tends to reflect the latest local state for a cached model.
        snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        config_path = snapshot / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue

        repo_id = model_dir.name.removeprefix("models--").replace("--", "/")
        candidates.append(LocalModelCandidate(repo_id=repo_id, path=str(snapshot), config=config))
    return tuple(candidates)


def _resolve_dense_base_path(model_path: str) -> str:
    config = _load_hf_config(model_path)
    if not _is_bitsandbytes_quantized(config):
        return model_path

    signature = _config_signature(config)
    ranked: list[tuple[float, LocalModelCandidate]] = []
    for candidate in _iter_local_model_candidates():
        if candidate.path == model_path:
            continue
        if _is_bitsandbytes_quantized(candidate.config):
            continue
        if _config_signature(candidate.config) != signature:
            continue
        ranked.append((_candidate_score(model_path, candidate), candidate))

    if not ranked:
        raise ValueError(
            f"Model path '{model_path}' points to a bitsandbytes-quantized checkpoint, "
            "but no dense local sibling with a matching architecture was found. "
            "Provide a dense base model path explicitly."
        )

    score, best = max(ranked, key=lambda item: (item[0], item[1].repo_id))
    logger.warning(
        "Resolved quantized model %s to dense local sibling %s (%s, score=%.3f)",
        model_path,
        best.repo_id,
        best.path,
        score,
    )
    return best.path


def _adapter_tokenizer_is_compatible(adapter_path: str, base_model_path: str) -> bool:
    folder = _resolve_metadata_dir(adapter_path, allow_patterns=_TOKENIZER_FILES)
    if folder is None:
        return False

    if not any(
        os.path.exists(os.path.join(folder, filename))
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    ):
        return False

    added_tokens_path = os.path.join(folder, "added_tokens.json")
    if not os.path.exists(added_tokens_path):
        return True

    try:
        with open(added_tokens_path, "r", encoding="utf-8") as f:
            added_tokens = json.load(f)
        if len(added_tokens) == 0:
            return True
        max_token_id = max(int(v) for v in added_tokens.values())
        vocab_size = int(_load_hf_config(base_model_path).vocab_size)
    except Exception as exc:
        logger.warning("Failed to validate adapter tokenizer compatibility for %s: %s", adapter_path, exc)
        return False

    if max_token_id >= vocab_size:
        logger.warning(
            "Ignoring tokenizer files from %s because added token id %d exceeds base vocab size %d.",
            adapter_path,
            max_token_id,
            vocab_size,
        )
        return False
    return True


@functools.lru_cache(maxsize=128)
def resolve_model_paths(model_path: str, lora_path: str | None = None) -> ResolvedModelPaths:
    resolved_model_path = _resolve_local_or_cached_path(model_path) or model_path
    resolved_lora_path = _resolve_local_or_cached_path(lora_path)

    if resolved_lora_path is None:
        adapter_meta = _load_optional_json(model_path, "adapter_config.json")
        if adapter_meta is not None:
            _, adapter_config = adapter_meta
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not isinstance(base_model_path, str) or len(base_model_path) == 0:
                raise ValueError(
                    f"Adapter path '{model_path}' is missing base_model_name_or_path in adapter_config.json"
                )
            logger.info(
                "Treating %s as an adapter bundle and resolving base model %s",
                model_path,
                base_model_path,
            )
            resolved_model_path = base_model_path
            resolved_lora_path = model_path

    resolved_model_path = _resolve_dense_base_path(resolved_model_path)
    tokenizer_path = resolved_model_path
    if resolved_lora_path is not None and _adapter_tokenizer_is_compatible(
        resolved_lora_path, resolved_model_path
    ):
        tokenizer_path = resolved_lora_path

    return ResolvedModelPaths(
        model_path=resolved_model_path,
        lora_path=resolved_lora_path,
        tokenizer_path=tokenizer_path,
    )


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    resolved_model_path = _resolve_local_cached_repo_snapshot(model_path) or model_path
    try:
        return AutoTokenizer.from_pretrained(resolved_model_path)
    except Exception as exc:
        adapter_meta = _load_optional_json(resolved_model_path, "adapter_config.json")
        if adapter_meta is None:
            raise

        _, adapter_config = adapter_meta
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not isinstance(base_model_path, str) or len(base_model_path) == 0:
            raise

        logger.warning(
            "Falling back to base tokenizer %s because tokenizer loading failed for adapter path %s: %s",
            base_model_path,
            model_path,
            exc,
        )
        return AutoTokenizer.from_pretrained(base_model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    resolved_model_path = _resolve_local_cached_repo_snapshot(model_path) or model_path
    try:
        return AutoConfig.from_pretrained(resolved_model_path)
    except ValueError as exc:
        raw_config = _load_raw_config_dict(resolved_model_path) or {}
        model_type = raw_config.get("model_type")
        architectures = raw_config.get("architectures", [])
        if model_type == "qwen3_5":
            raise ValueError(
                f"Model path '{resolved_model_path}' uses the unsupported qwen3_5 architecture "
                f"({architectures or ['unknown']}), which MiniSGL does not implement yet."
            ) from exc
        raise


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        if not _path_has_required_files(model_path, ["*.safetensors"]):
            raise ValueError(
                f"Local model path '{model_path}' does not contain any .safetensors weight files."
            )
        return model_path
    local_snapshot = _resolve_local_cached_repo_snapshot(
        model_path,
        required_patterns=["*.safetensors"],
    )
    if local_snapshot is not None:
        return local_snapshot
    try:
        return _hf_snapshot_download(model_path, allow_patterns=["*.safetensors"])
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )


def ensure_local_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        if not _path_has_required_files(model_path, ["*.safetensors"]):
            raise ValueError(
                f"Local model path '{model_path}' does not contain any .safetensors weight files."
            )
        return model_path
    local_snapshot = _resolve_local_cached_repo_snapshot(
        model_path,
        required_patterns=["*.safetensors"],
    )
    if local_snapshot is not None:
        return local_snapshot
    return _hf_snapshot_download(model_path, allow_patterns=_MODEL_PREFETCH_ALLOW_PATTERNS)
