from __future__ import annotations

from typing import Any

from .hf_compat import HFCompatLLM, should_use_hf_compat_backend
from .llm import LLM as NativeLLM


def LLM(*args: Any, **kwargs: Any):
    model_path = kwargs.get("model_path")
    if model_path is None and args:
        model_path = args[0]
    lora_path = kwargs.get("lora_path")
    if model_path is not None and should_use_hf_compat_backend(str(model_path), lora_path):
        return HFCompatLLM(*args, **kwargs)
    return NativeLLM(*args, **kwargs)


__all__ = ["HFCompatLLM", "LLM", "NativeLLM"]
