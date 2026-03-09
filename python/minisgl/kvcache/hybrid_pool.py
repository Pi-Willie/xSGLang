from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class HybridKVCache(BaseKVCachePool):
    """KV cache that only materializes full-attention layers."""

    def __init__(
        self,
        *,
        num_layers: int,
        full_attention_layer_ids: tuple[int, ...],
        num_kv_heads: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size)
        self._num_layers = num_layers
        self._cached_layer_ids = tuple(full_attention_layer_ids)
        self._layer_map = {
            layer_id: idx for idx, layer_id in enumerate(self._cached_layer_ids)
        }
        self._device = device
        if len(self._cached_layer_ids) == 0:
            raise ValueError("HybridKVCache requires at least one full-attention layer")
        self._kv_buffer = torch.empty(
            (2, len(self._cached_layer_ids), num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def _physical_layer_id(self, layer_id: int) -> int:
        physical = self._layer_map.get(layer_id)
        if physical is None:
            raise ValueError(
                f"Layer {layer_id} does not use paged KV cache. "
                f"Cached layers: {self._cached_layer_ids}"
            )
        return physical

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._physical_layer_id(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._physical_layer_id(index)]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        physical = self._physical_layer_id(layer_id)
        store_cache(
            k_cache=self._k_buffer[physical].view(self._storage_shape),
            v_cache=self._v_buffer[physical].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def cached_layer_ids(self) -> tuple[int, ...]:
        return self._cached_layer_ids

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._layer_map
