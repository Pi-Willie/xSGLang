from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from .config import ModelConfig


@dataclass(frozen=True)
class DecoderArchitectureSpec:
    """Declarative description of a decoder family.

    The point of this object is to make architecture support readable and easy to extend.
    Adding a compatible Hugging Face family should be as small as registering one new spec,
    not copy-pasting an entire model implementation.
    """

    key: str
    supported_hf_architectures: tuple[str, ...]
    supported_model_types: tuple[str, ...]
    attention_bias: bool = False
    qk_norm: bool = False
    mlp_variant: str = "dense"
    experimental: bool = False
    notes: str = ""

    @property
    def is_moe(self) -> bool:
        return self.mlp_variant == "moe"


_DECODER_SPECS = (
    DecoderArchitectureSpec(
        key="llama",
        supported_hf_architectures=("LlamaForCausalLM",),
        supported_model_types=("llama",),
        attention_bias=False,
        qk_norm=False,
        mlp_variant="dense",
        notes="Baseline Llama-style RoPE + RMSNorm + SwiGLU stack.",
    ),
    DecoderArchitectureSpec(
        key="qwen2",
        supported_hf_architectures=("Qwen2ForCausalLM",),
        supported_model_types=("qwen2",),
        attention_bias=True,
        qk_norm=False,
        mlp_variant="dense",
        notes="Qwen2 / Qwen2.5 style stack with attention bias.",
    ),
    DecoderArchitectureSpec(
        key="qwen3",
        supported_hf_architectures=("Qwen3ForCausalLM",),
        supported_model_types=("qwen3",),
        attention_bias=False,
        qk_norm=True,
        mlp_variant="dense",
        notes="Qwen3 dense stack with Q/K RMSNorm.",
    ),
    DecoderArchitectureSpec(
        key="qwen3_moe",
        supported_hf_architectures=("Qwen3MoeForCausalLM",),
        supported_model_types=("qwen3_moe",),
        attention_bias=False,
        qk_norm=True,
        mlp_variant="moe",
        notes="Qwen3 MoE stack with routed MLP blocks.",
    ),
    DecoderArchitectureSpec(
        key="qwen3_5",
        supported_hf_architectures=("Qwen3_5ForCausalLM",),
        supported_model_types=("qwen3_5_text",),
        attention_bias=False,
        qk_norm=True,
        mlp_variant="dense",
        experimental=True,
        notes=(
            "Qwen 3.5 hybrid stack with gated full-attention blocks and linear-attention "
            "recurrent state."
        ),
    ),
    DecoderArchitectureSpec(
        key="mistral",
        supported_hf_architectures=("MistralForCausalLM",),
        supported_model_types=("mistral",),
        attention_bias=False,
        qk_norm=False,
        mlp_variant="dense",
        experimental=True,
        notes=(
            "Runs on the same dense decoder substrate as Llama. "
            "The current runtime does not yet specialize sliding-window masking."
        ),
    ),
    DecoderArchitectureSpec(
        key="ministral",
        supported_hf_architectures=("MinistralForCausalLM",),
        supported_model_types=("ministral",),
        attention_bias=True,
        qk_norm=False,
        mlp_variant="dense",
        experimental=True,
        notes=(
            "Close to the Qwen2-style dense stack with attention bias. "
            "Alternating local/global attention is not yet specialized."
        ),
    ),
    DecoderArchitectureSpec(
        key="gemma",
        supported_hf_architectures=("GemmaForCausalLM",),
        supported_model_types=("gemma",),
        attention_bias=False,
        qk_norm=False,
        mlp_variant="dense",
        experimental=True,
        notes=(
            "Gemma-style dense decoder using RoPE, RMSNorm, and GeGLU-compatible activation "
            "aliases. This path is intended for research use and extension work."
        ),
    ),
)

_ARCH_BY_NAME: Dict[str, DecoderArchitectureSpec] = {
    name: spec
    for spec in _DECODER_SPECS
    for name in spec.supported_hf_architectures
}
_MODEL_TYPE_TO_ARCH: Dict[str, DecoderArchitectureSpec] = {
    model_type: spec
    for spec in _DECODER_SPECS
    for model_type in spec.supported_model_types
}
_SPEC_BY_KEY: Dict[str, DecoderArchitectureSpec] = {spec.key: spec for spec in _DECODER_SPECS}


def supported_architecture_names() -> tuple[str, ...]:
    return tuple(sorted(_ARCH_BY_NAME))


def iter_decoder_specs() -> Iterable[DecoderArchitectureSpec]:
    return iter(_DECODER_SPECS)


def get_decoder_architecture(name_or_key: str) -> DecoderArchitectureSpec:
    spec = _ARCH_BY_NAME.get(name_or_key) or _SPEC_BY_KEY.get(name_or_key)
    if spec is None:
        raise KeyError(f"Unknown decoder architecture '{name_or_key}'")
    return spec


def architecture_metadata() -> Mapping[str, Dict[str, object]]:
    return {
        spec.key: {
            "hf_architectures": spec.supported_hf_architectures,
            "model_types": spec.supported_model_types,
            "attention_bias": spec.attention_bias,
            "qk_norm": spec.qk_norm,
            "mlp_variant": spec.mlp_variant,
            "experimental": spec.experimental,
            "notes": spec.notes,
        }
        for spec in _DECODER_SPECS
    }


def resolve_decoder_architecture(
    architecture_name: str | None,
    model_config: ModelConfig,
) -> DecoderArchitectureSpec:
    """Resolve the best decoder implementation for a HF config.

    Resolution prefers the explicit HF `architectures` entry, then falls back to
    `model_type`. This keeps the engine deterministic while still being robust to
    lightly customized configs.
    """

    if architecture_name:
        spec = _ARCH_BY_NAME.get(architecture_name)
        if spec is not None:
            return spec

    for candidate in model_config.architectures:
        spec = _ARCH_BY_NAME.get(candidate)
        if spec is not None:
            return spec

    spec = _MODEL_TYPE_TO_ARCH.get(model_config.model_type)
    if spec is not None:
        return spec

    supported = ", ".join(supported_architecture_names())
    raise ValueError(
        f"Model architecture {architecture_name or model_config.architectures or model_config.model_type} "
        f"is not supported. Supported Hugging Face architectures: {supported}."
    )


__all__ = [
    "DecoderArchitectureSpec",
    "architecture_metadata",
    "get_decoder_architecture",
    "iter_decoder_specs",
    "resolve_decoder_architecture",
    "supported_architecture_names",
]
