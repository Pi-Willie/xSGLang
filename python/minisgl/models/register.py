from __future__ import annotations

import importlib

from .architectures import architecture_metadata, resolve_decoder_architecture, supported_architecture_names
from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3_5ForCausalLM": (".qwen3_5", "Qwen3_5ForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "MinistralForCausalLM": (".ministral", "MinistralForCausalLM"),
    "GemmaForCausalLM": (".gemma", "GemmaForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    resolved = resolve_decoder_architecture(model_architecture, model_config)
    entry = _MODEL_REGISTRY.get(resolved.supported_hf_architectures[0])
    if entry is None:
        supported = ", ".join(supported_architecture_names())
        raise ValueError(
            f"Model architecture {model_architecture} not supported. "
            f"Supported Hugging Face architectures: {supported}"
        )
    module_path, class_name = entry
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = [
    "architecture_metadata",
    "get_model_class",
    "supported_architecture_names",
]
