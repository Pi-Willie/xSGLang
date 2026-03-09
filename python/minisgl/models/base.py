from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP
from .outputs import (
    AttachedOutputManager,
    ModelForwardOutput,
    SampleOutputSpec,
    discover_output_manager,
    select_sample_hidden_states,
)

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> ModelForwardOutput: ...

    def __init__(self) -> None:
        self._output_manager: AttachedOutputManager | None = None
        self.supports_cuda_graph = True

    def forward_with_hooks(self) -> ModelForwardOutput:
        return self.forward()

    def create_state_cache(
        self,
        *,
        num_tables: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        _ = num_tables, device, dtype
        return None

    @property
    def num_layers(self) -> int:
        model = getattr(self, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            raise AttributeError(f"{type(self).__name__} does not expose a decoder layer stack")
        return len(layers.op_list)

    def configure_output_manager(
        self,
        *,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        search_paths: Sequence[str | None],
        device: torch.device,
    ) -> None:
        self._output_manager = discover_output_manager(
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
            search_paths=search_paths,
            device=device,
        )

    def sample_output_specs(self) -> tuple[SampleOutputSpec, ...]:
        if self._output_manager is None:
            return ()
        return self._output_manager.output_specs()

    def available_sample_output_names(self) -> tuple[str, ...]:
        if self._output_manager is None:
            return ()
        return self._output_manager.available_output_names()

    def _build_forward_output(
        self,
        *,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> ModelForwardOutput:
        if self._output_manager is None:
            return ModelForwardOutput(logits=logits, sample_outputs={})
        batch = get_global_ctx().batch
        sample_hidden = select_sample_hidden_states(hidden_states)
        return ModelForwardOutput(
            logits=logits,
            sample_outputs=self._output_manager.forward(
                sample_hidden,
                requested_outputs=batch.requested_sample_outputs,
            ),
        )
