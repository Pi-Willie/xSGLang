"""Stub HF-compat backend.

The upstream commit references an HF-compatibility backend that was not
included in this trimmed repo. Qwen3 (and the other listed architectures)
are supported by the native runtime, so we always route to NativeLLM.
"""
from __future__ import annotations
from typing import Any


def should_use_hf_compat_backend(model_path: str, lora_path: Any = None) -> bool:
    return False


class HFCompatLLM:  # pragma: no cover - never instantiated; native backend is used
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "HFCompatLLM backend is not bundled in this build; the native runtime is used instead."
        )
