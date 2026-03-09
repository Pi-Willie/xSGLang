from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.models.hook_utils import (
    dispatch_layer_entries,
    dispatch_special_point,
    prepare_hook_runtime,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .qwen3_5_kernels import (
    TRITON_GDN_AVAILABLE,
    depthwise_conv1d_decode_ring_update,
    fused_sigmoid_gating_delta_rule_decode,
)

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.hooks import HookContext, HookDispatchEntry

    from .config import ModelConfig


QWEN3_5_TRACK_INTERVAL = 64


class DepthwiseConv1d(BaseOP):
    def __init__(self, channels: int, kernel_size: int) -> None:
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        raise NotImplementedError("DepthwiseConv1d is a parameter holder in the native Qwen 3.5 path")


class Qwen3_5RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self._runtime_weight: torch.Tensor | None = None
        self._weight_ptr: int | None = None
        self._weight_version: int | None = None

    def _get_runtime_weight(self) -> torch.Tensor:
        ptr = self.weight.untyped_storage().data_ptr()
        version = getattr(self.weight, "_version", None)
        runtime_weight = self._runtime_weight
        if (
            runtime_weight is None
            or runtime_weight.device != self.weight.device
            or self._weight_ptr != ptr
            or self._weight_version != version
        ):
            runtime_weight = (self.weight + 1.0).contiguous()
            self._runtime_weight = runtime_weight
            self._weight_ptr = ptr
            self._weight_version = version
        return runtime_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self._get_runtime_weight(), self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self._get_runtime_weight(), self.eps, out=x)


class Qwen3_5RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        self._runtime_weight: torch.Tensor | None = None
        self._weight_ptr: int | None = None
        self._weight_version: int | None = None

    def _get_runtime_weight(self) -> torch.Tensor:
        ptr = self.weight.untyped_storage().data_ptr()
        version = getattr(self.weight, "_version", None)
        runtime_weight = self._runtime_weight
        if (
            runtime_weight is None
            or runtime_weight.device != self.weight.device
            or self._weight_ptr != ptr
            or self._weight_version != version
        ):
            runtime_weight = (self.weight + 1.0).contiguous()
            self._runtime_weight = runtime_weight
            self._weight_ptr = ptr
            self._weight_version = version
        return runtime_weight

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        runtime_weight = self._get_runtime_weight()
        if residual is None:
            return self.rmsnorm(x, runtime_weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, runtime_weight, self.eps)
        return x, residual


class Qwen3_5RMSNormGated(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normed = self.rmsnorm(hidden_states, self.weight, self.eps)
        return normed.mul_(F.silu(gate))


class Qwen3_5PartialRotaryEmbedding(BaseOP):
    def __init__(
        self,
        *,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        base: float,
    ) -> None:
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.base = base
        self._cos_sin_cache: torch.Tensor | None = None
        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cos_sin_cache is not None and self._cos_sin_cache.device == device:
            return
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / float(self.rotary_dim))
        )
        positions = torch.arange(self.max_position, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self._cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return query, key

        self._ensure_cache(query.device)
        assert self._cos_sin_cache is not None
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query[..., : self.rotary_dim],
            key=key[..., : self.rotary_dim],
            head_size=self.rotary_dim,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    track_positions: Tuple[int, ...] = (),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    tracked_states: List[torch.Tensor] = []
    if track_positions:
        for pos in track_positions:
            if 0 < pos <= seq_len:
                tracked_states.append(hidden_states_new[:, :, pos : pos + state_len].clone())
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight, bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype), tracked_states


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    track_positions: Tuple[int, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor | None, List[torch.Tensor]]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    lower_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(lower_mask, 0)
    for idx in range(1, chunk_size):
        row = attn[..., idx, :idx].clone()
        sub = attn[..., :idx, :idx].clone()
        attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(device=value.device, dtype=value.dtype)
    )
    core_attn_out = torch.zeros_like(value)
    tracked_states: Dict[int, torch.Tensor] = {}
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for chunk_idx in range(total_sequence_length // chunk_size):
        q_chunk = query[:, :, chunk_idx]
        k_chunk = key[:, :, chunk_idx]
        v_chunk = value[:, :, chunk_idx]
        attn = (q_chunk @ k_chunk.transpose(-1, -2) * decay_mask[:, :, chunk_idx]).masked_fill_(
            upper_mask,
            0,
        )
        v_prime = k_cumdecay[:, :, chunk_idx] @ last_recurrent_state
        v_new = v_chunk - v_prime
        attn_inter = (q_chunk * g[:, :, chunk_idx, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, chunk_idx] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, chunk_idx, -1, None, None].exp()
            + (
                k_chunk
                * (g[:, :, chunk_idx, -1, None] - g[:, :, chunk_idx]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
        chunk_end = min(sequence_length, (chunk_idx + 1) * chunk_size)
        if chunk_end in track_positions:
            tracked_states[chunk_end] = last_recurrent_state.clone()

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return (
        core_attn_out,
        last_recurrent_state,
        [tracked_states[pos] for pos in track_positions if pos in tracked_states],
    )


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    track_positions: Tuple[int, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor | None, List[torch.Tensor]]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    query = query * (query.shape[-1] ** -0.5)

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device)
    tracked_states: Dict[int, torch.Tensor] = {}
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(device=value.device, dtype=value.dtype)
    )

    for step_idx in range(sequence_length):
        q_t = query[:, :, step_idx]
        k_t = key[:, :, step_idx]
        v_t = value[:, :, step_idx]
        g_t = g[:, :, step_idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, step_idx].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, step_idx] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        step_end = step_idx + 1
        if step_end in track_positions:
            tracked_states[step_end] = last_recurrent_state.clone()

    if not output_final_state:
        last_recurrent_state = None
    return (
        core_attn_out.transpose(1, 2).contiguous().to(initial_dtype),
        last_recurrent_state,
        [tracked_states[pos] for pos in track_positions if pos in tracked_states],
    )


@dataclass(frozen=True)
class Qwen3_5TrackedBoundary:
    rel_offset: int
    total_len: int
    snapshot_slot: int


@dataclass
class Qwen3_5TrackingPlan:
    boundaries: Tuple[Qwen3_5TrackedBoundary, ...]

    @property
    def rel_offsets(self) -> Tuple[int, ...]:
        return tuple(boundary.rel_offset for boundary in self.boundaries)


class Qwen3_5StateCache:
    def __init__(
        self,
        config: "ModelConfig",
        *,
        num_tables: int,
        device: torch.device,
        dtype: torch.dtype,
        snapshot_rows: int | None = None,
    ) -> None:
        tp = get_tp_info()
        self.track_interval = max(QWEN3_5_TRACK_INTERVAL, get_global_ctx().page_size)
        self.linear_layers = tuple(
            layer_idx for layer_idx, layer_type in enumerate(config.layer_types) if layer_type == "linear_attention"
        )
        self._tracking: Dict[int, Qwen3_5TrackingPlan] = {}
        self.has_previous = torch.zeros(num_tables, dtype=torch.bool, device=device)
        if len(self.linear_layers) == 0:
            self.conv_states: List[torch.Tensor | None] = [None] * config.num_layers
            self.recurrent_states: List[torch.Tensor | None] = [None] * config.num_layers
            self.snapshot_conv_states: List[torch.Tensor | None] = [None] * config.num_layers
            self.snapshot_recurrent_states: List[torch.Tensor | None] = [None] * config.num_layers
            self.conv_write_positions: List[torch.Tensor | None] = [None] * config.num_layers
            self.snapshot_conv_write_positions: List[torch.Tensor | None] = [None] * config.num_layers
            self._snapshot_free_slots: List[int] = []
            return

        local_num_k_heads = div_even(config.linear_num_key_heads, tp.size)
        local_num_v_heads = div_even(config.linear_num_value_heads, tp.size)
        local_key_dim = local_num_k_heads * config.linear_key_head_dim
        local_value_dim = local_num_v_heads * config.linear_value_head_dim
        conv_dim = local_key_dim * 2 + local_value_dim
        kernel = config.linear_conv_kernel_dim
        snapshot_rows = snapshot_rows if snapshot_rows is not None else max(32, num_tables * 4)

        self.conv_states = [None] * config.num_layers
        self.recurrent_states = [None] * config.num_layers
        self.snapshot_conv_states = [None] * config.num_layers
        self.snapshot_recurrent_states = [None] * config.num_layers
        self.conv_write_positions = [None] * config.num_layers
        self.snapshot_conv_write_positions = [None] * config.num_layers
        self._snapshot_free_slots = list(range(snapshot_rows))
        for layer_idx in self.linear_layers:
            self.conv_states[layer_idx] = torch.zeros(
                num_tables,
                conv_dim,
                kernel,
                device=device,
                dtype=dtype,
            )
            self.recurrent_states[layer_idx] = torch.zeros(
                num_tables,
                local_num_v_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                device=device,
                dtype=torch.float32,
            )
            self.snapshot_conv_states[layer_idx] = torch.zeros(
                snapshot_rows,
                conv_dim,
                kernel,
                device=device,
                dtype=dtype,
            )
            self.snapshot_recurrent_states[layer_idx] = torch.zeros(
                snapshot_rows,
                local_num_v_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                device=device,
                dtype=torch.float32,
            )
            self.conv_write_positions[layer_idx] = torch.zeros(num_tables, device=device, dtype=torch.int32)
            self.snapshot_conv_write_positions[layer_idx] = torch.zeros(
                snapshot_rows,
                device=device,
                dtype=torch.int32,
            )

    def reset_row(self, table_idx: int) -> None:
        self.discard_tracked_prefixes(table_idx)
        self.has_previous[table_idx] = False
        for layer_idx in self.linear_layers:
            conv = self.conv_states[layer_idx]
            recurrent = self.recurrent_states[layer_idx]
            write_pos = self.conv_write_positions[layer_idx]
            assert conv is not None and recurrent is not None and write_pos is not None
            conv[table_idx].zero_()
            recurrent[table_idx].zero_()
            write_pos[table_idx] = 0

    def copy_row(self, src_table_idx: int, dst_table_idx: int) -> None:
        self.has_previous[dst_table_idx] = self.has_previous[src_table_idx]
        for layer_idx in self.linear_layers:
            conv = self.conv_states[layer_idx]
            recurrent = self.recurrent_states[layer_idx]
            write_pos = self.conv_write_positions[layer_idx]
            assert conv is not None and recurrent is not None and write_pos is not None
            conv[dst_table_idx].copy_(conv[src_table_idx])
            recurrent[dst_table_idx].copy_(recurrent[src_table_idx])
            write_pos[dst_table_idx] = write_pos[src_table_idx]

    def _allocate_snapshot_slot(self) -> int | None:
        if len(self._snapshot_free_slots) == 0:
            return None
        slot = self._snapshot_free_slots.pop()
        for layer_idx in self.linear_layers:
            conv = self.snapshot_conv_states[layer_idx]
            recurrent = self.snapshot_recurrent_states[layer_idx]
            write_pos = self.snapshot_conv_write_positions[layer_idx]
            assert conv is not None and recurrent is not None and write_pos is not None
            conv[slot].zero_()
            recurrent[slot].zero_()
            write_pos[slot] = 0
        return slot

    def release_snapshot_slot(self, slot: int) -> None:
        if slot not in self._snapshot_free_slots:
            self._snapshot_free_slots.append(slot)

    def restore_snapshot(self, snapshot_slot: int, table_idx: int) -> None:
        self.has_previous[table_idx] = True
        for layer_idx in self.linear_layers:
            live_conv = self.conv_states[layer_idx]
            live_recurrent = self.recurrent_states[layer_idx]
            snap_conv = self.snapshot_conv_states[layer_idx]
            snap_recurrent = self.snapshot_recurrent_states[layer_idx]
            live_write_pos = self.conv_write_positions[layer_idx]
            snap_write_pos = self.snapshot_conv_write_positions[layer_idx]
            assert (
                live_conv is not None
                and live_recurrent is not None
                and snap_conv is not None
                and snap_recurrent is not None
                and live_write_pos is not None
                and snap_write_pos is not None
            )
            live_conv[table_idx].copy_(snap_conv[snapshot_slot])
            live_recurrent[table_idx].copy_(snap_recurrent[snapshot_slot])
            live_write_pos[table_idx] = snap_write_pos[snapshot_slot]

    def begin_batch_tracking(self, batch: "Batch") -> None:
        stale = set(self._tracking) - {req.table_idx for req in batch.reqs}
        for table_idx in stale:
            self.discard_tracked_prefixes(table_idx)
        if not batch.is_prefill:
            return
        for req in batch.reqs:
            self.discard_tracked_prefixes(req.table_idx)
            if req.extend_len <= 0:
                continue
            first_boundary = ((req.cached_len // self.track_interval) + 1) * self.track_interval
            boundaries: List[Qwen3_5TrackedBoundary] = []
            for total_len in range(first_boundary, req.device_len + 1, self.track_interval):
                slot = self._allocate_snapshot_slot()
                if slot is None:
                    break
                boundaries.append(
                    Qwen3_5TrackedBoundary(
                        rel_offset=total_len - req.cached_len,
                        total_len=total_len,
                        snapshot_slot=slot,
                    )
                )
            if boundaries:
                self._tracking[req.table_idx] = Qwen3_5TrackingPlan(tuple(boundaries))

    def tracking_offsets(self, table_idx: int) -> Tuple[int, ...]:
        plan = self._tracking.get(table_idx)
        return () if plan is None else plan.rel_offsets

    def write_tracked_boundary(
        self,
        *,
        layer_idx: int,
        table_idx: int,
        rel_offset: int,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        plan = self._tracking.get(table_idx)
        if plan is None:
            return
        boundary = next((item for item in plan.boundaries if item.rel_offset == rel_offset), None)
        if boundary is None:
            return
        snap_conv = self.snapshot_conv_states[layer_idx]
        snap_recurrent = self.snapshot_recurrent_states[layer_idx]
        snap_write_pos = self.snapshot_conv_write_positions[layer_idx]
        assert snap_conv is not None and snap_recurrent is not None and snap_write_pos is not None
        snap_conv[boundary.snapshot_slot].copy_(conv_state[0])
        snap_recurrent[boundary.snapshot_slot].copy_(recurrent_state[0])
        snap_write_pos[boundary.snapshot_slot] = 0

    def consume_tracked_prefixes(self, table_idx: int, *, upto: int) -> List[Tuple[int, int]]:
        plan = self._tracking.pop(table_idx, None)
        if plan is None:
            return []
        consumed: List[Tuple[int, int]] = []
        leftovers: List[Qwen3_5TrackedBoundary] = []
        for boundary in plan.boundaries:
            if boundary.total_len <= upto:
                consumed.append((boundary.total_len, boundary.snapshot_slot))
            else:
                leftovers.append(boundary)
        if leftovers:
            self._tracking[table_idx] = Qwen3_5TrackingPlan(tuple(leftovers))
        return consumed

    def discard_tracked_prefixes(self, table_idx: int) -> None:
        plan = self._tracking.pop(table_idx, None)
        if plan is None:
            return
        for boundary in plan.boundaries:
            self.release_snapshot_slot(boundary.snapshot_slot)

    def has_previous_state(self, table_idx: int) -> bool:
        return bool(self.has_previous[table_idx].item())

    def mark_row_active(self, table_idx: int) -> None:
        self.has_previous[table_idx] = True

    def recurrent_state_source(self, layer_idx: int) -> torch.Tensor:
        recurrent = self.recurrent_states[layer_idx]
        assert recurrent is not None
        return recurrent

    def conv_state_pool(self, layer_idx: int) -> torch.Tensor:
        conv = self.conv_states[layer_idx]
        assert conv is not None
        return conv

    def conv_write_pos_pool(self, layer_idx: int) -> torch.Tensor:
        write_pos = self.conv_write_positions[layer_idx]
        assert write_pos is not None
        return write_pos

    def prepare_conv_state(self, layer_idx: int, table_idx: int) -> torch.Tensor:
        conv = self.conv_states[layer_idx]
        write_pos = self.conv_write_positions[layer_idx]
        assert conv is not None and write_pos is not None
        row = conv[table_idx : table_idx + 1]
        pos = int(write_pos[table_idx].item())
        if pos != 0:
            row.copy_(torch.roll(row, shifts=-pos, dims=-1))
            write_pos[table_idx] = 0
        return row

    def conv_state(self, layer_idx: int, table_idx: int) -> torch.Tensor:
        conv = self.conv_states[layer_idx]
        assert conv is not None
        return conv[table_idx : table_idx + 1]

    def recurrent_state(self, layer_idx: int, table_idx: int) -> torch.Tensor:
        recurrent = self.recurrent_states[layer_idx]
        assert recurrent is not None
        return recurrent[table_idx : table_idx + 1]


class Qwen3_5FullAttention(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        tp = get_tp_info()
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.num_q_heads = div_even(config.num_qo_heads, tp.size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp.size)
        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size,
            [
                config.num_qo_heads * self.head_dim,
                config.num_qo_heads * self.head_dim,
                config.num_kv_heads * self.head_dim,
                config.num_kv_heads * self.head_dim,
            ],
            has_bias=config.attention_bias,
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * self.head_dim,
            config.hidden_size,
            has_bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(round(self.head_dim * config.rotary_partial_factor))
        rotary_dim -= rotary_dim % 2
        self.rotary = Qwen3_5PartialRotaryEmbedding(
            head_dim=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
        )

    @nvtx_annotate("Qwen3_5FullAttention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        q, gate, k, v = qkv.split((self.q_dim, self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(-1, self.num_q_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(ctx.batch.positions, q, k)

        o = ctx.attn_backend.forward(
            q,
            k.view(-1, self.kv_dim),
            v,
            self.layer_id,
            ctx.batch,
        )
        o = o.view(-1, self.q_dim)
        o.mul_(gate.sigmoid_())
        return self.o_proj.forward(o)


class Qwen3_5LinearAttention(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        tp = get_tp_info()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_k_heads = div_even(config.linear_num_key_heads, tp.size)
        self.num_v_heads = div_even(config.linear_num_value_heads, tp.size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.in_proj = LinearColParallelMerged(
            config.hidden_size,
            [
                self.conv_dim * tp.size,
                self.value_dim * tp.size,
                self.num_v_heads * tp.size,
                self.num_v_heads * tp.size,
            ],
            has_bias=False,
        )
        self.conv1d = DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.A_log = torch.empty(self.num_v_heads)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(self.value_dim * tp.size, config.hidden_size, has_bias=False)
        self._A_fp32: torch.Tensor | None = None
        self._A_ptr: int | None = None
        self._A_version: int | None = None
        self._dt_bias_fp32: torch.Tensor | None = None
        self._dt_bias_ptr: int | None = None
        self._dt_bias_version: int | None = None
        self._conv_weight: torch.Tensor | None = None
        self._conv_weight_ptr: int | None = None
        self._conv_weight_version: int | None = None

    def _decay_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a_ptr = self.A_log.untyped_storage().data_ptr()
        a_version = getattr(self.A_log, "_version", None)
        if (
            self._A_fp32 is None
            or self._A_fp32.device != self.A_log.device
            or self._A_ptr != a_ptr
            or self._A_version != a_version
        ):
            self._A_fp32 = self.A_log.to(dtype=torch.float32).exp().contiguous()
            self._A_ptr = a_ptr
            self._A_version = a_version

        dt_ptr = self.dt_bias.untyped_storage().data_ptr()
        dt_version = getattr(self.dt_bias, "_version", None)
        if (
            self._dt_bias_fp32 is None
            or self._dt_bias_fp32.device != self.dt_bias.device
            or self._dt_bias_ptr != dt_ptr
            or self._dt_bias_version != dt_version
        ):
            self._dt_bias_fp32 = self.dt_bias.to(dtype=torch.float32).contiguous()
            self._dt_bias_ptr = dt_ptr
            self._dt_bias_version = dt_version
        return self._A_fp32, self._dt_bias_fp32

    def _conv_kernel(self) -> torch.Tensor:
        ptr = self.conv1d.weight.untyped_storage().data_ptr()
        version = getattr(self.conv1d.weight, "_version", None)
        if (
            self._conv_weight is None
            or self._conv_weight.device != self.conv1d.weight.device
            or self._conv_weight_ptr != ptr
            or self._conv_weight_version != version
        ):
            self._conv_weight = self.conv1d.weight[:, 0].contiguous()
            self._conv_weight_ptr = ptr
            self._conv_weight_version = version
        return self._conv_weight

    def _project_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.view(-1, self.hidden_size)
        fused = self.in_proj.forward(flat)
        mixed_qkv, z, a, b = torch.split(
            fused,
            [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads],
            dim=-1,
        )
        mixed_qkv = mixed_qkv.view(-1, self.conv_dim)
        z = z.view(-1, self.num_v_heads, self.head_v_dim)
        a = a.view(-1, self.num_v_heads)
        b = b.view(-1, self.num_v_heads)
        return mixed_qkv, z, a, b

    def _apply_conv_fast_path(
        self,
        *,
        batch: "Batch",
        mixed_qkv: torch.Tensor,
        state_cache: Qwen3_5StateCache,
    ) -> torch.Tensor:
        if batch.phase == "decode" and mixed_qkv.shape[0] == len(batch.reqs):
            return depthwise_conv1d_decode_ring_update(
                hidden=mixed_qkv.view(len(batch.reqs), self.conv_dim),
                state_pool=state_cache.conv_state_pool(self.layer_id),
                write_positions=state_cache.conv_write_pos_pool(self.layer_id),
                table_indices=batch.req_table_indices_i32,
                weight=self._conv_kernel(),
            ).reshape(-1, self.conv_dim)

        out = torch.empty_like(mixed_qkv)
        for req, (start, end) in zip(batch.reqs, batch.req_slices):
            if end <= start:
                continue
            conv_state = state_cache.prepare_conv_state(self.layer_id, req.table_idx)
            req_hidden = mixed_qkv[start:end].transpose(0, 1).unsqueeze(0)
            req_out, _ = torch_causal_conv1d_update(req_hidden, conv_state, self.conv1d.weight)
            out[start:end].copy_(req_out[0].transpose(0, 1))
        return out

    def _can_use_fused_recurrent(
        self,
        batch: "Batch",
        state_cache: Qwen3_5StateCache,
        *,
        total_tokens: int,
    ) -> bool:
        if batch.phase != "decode":
            return False
        if not TRITON_GDN_AVAILABLE or self.head_k_dim > 128:
            return False
        if total_tokens == 0:
            return False
        return all(len(state_cache.tracking_offsets(req.table_idx)) == 0 for req in batch.reqs)

    def _forward_fused_recurrent(
        self,
        *,
        batch: "Batch",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state_cache: Qwen3_5StateCache,
    ) -> torch.Tensor:
        A_fp32, dt_bias_fp32 = self._decay_params()
        attn_out = fused_sigmoid_gating_delta_rule_decode(
            A=A_fp32,
            a=a,
            dt_bias=dt_bias_fp32,
            q=query,
            k=key,
            v=value,
            b=b,
            state_source=state_cache.recurrent_state_source(self.layer_id),
            state_indices=batch.req_table_indices_i32,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = self.norm.forward(
            attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        return self.out_proj.forward(core_attn_out.reshape(-1, self.value_dim)).view(-1, self.hidden_size)

    def _forward_one(
        self,
        hidden_states: torch.Tensor,
        *,
        state_cache: Qwen3_5StateCache,
        table_idx: int,
        had_previous_state: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size)
        track_positions = tuple(
            pos for pos in state_cache.tracking_offsets(table_idx) if 0 < pos <= seq_len
        )

        fused = self.in_proj.forward(flat)
        mixed_qkv, z, a, b = torch.split(
            fused,
            [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads],
            dim=-1,
        )
        mixed_qkv = mixed_qkv.view(batch_size, seq_len, self.conv_dim).transpose(1, 2)
        z = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        a = a.view(batch_size, seq_len, self.num_v_heads)
        b = b.view(batch_size, seq_len, self.num_v_heads)

        conv_state = state_cache.prepare_conv_state(self.layer_id, table_idx)
        mixed_qkv, tracked_conv_states = torch_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight,
            track_positions=track_positions,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        A_fp32, dt_bias_fp32 = self._decay_params()
        beta = b.sigmoid()
        g = -A_fp32.to(device=a.device) * F.softplus(a.to(torch.float32) + dt_bias_fp32.to(device=a.device))
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        recurrent_state = state_cache.recurrent_state(self.layer_id, table_idx)
        initial_state = recurrent_state if had_previous_state else None
        if seq_len > 128 and not had_previous_state:
            core_attn_out, last_recurrent_state, tracked_recurrent_states = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                track_positions=track_positions,
            )
        else:
            core_attn_out, last_recurrent_state, tracked_recurrent_states = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                track_positions=track_positions,
            )
        assert last_recurrent_state is not None
        recurrent_state.copy_(last_recurrent_state.to(recurrent_state.dtype))
        if track_positions:
            for rel_offset, conv_snapshot, recurrent_snapshot in zip(
                track_positions,
                tracked_conv_states,
                tracked_recurrent_states,
            ):
                state_cache.write_tracked_boundary(
                    layer_idx=self.layer_id,
                    table_idx=table_idx,
                    rel_offset=rel_offset,
                    conv_state=conv_snapshot,
                    recurrent_state=recurrent_snapshot,
                )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj.forward(core_attn_out.view(-1, self.value_dim)).view(batch_size, seq_len, -1)

    @nvtx_annotate("Qwen3_5LinearAttention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        state_cache = ctx.state_cache
        if not isinstance(state_cache, Qwen3_5StateCache):
            raise RuntimeError("Qwen 3.5 linear attention requires a bound Qwen3_5StateCache")

        batch = ctx.batch
        if self._can_use_fused_recurrent(batch, state_cache, total_tokens=x.shape[0]):
            mixed_qkv, z, a, b = self._project_inputs(x)
            mixed_qkv = self._apply_conv_fast_path(batch=batch, mixed_qkv=mixed_qkv, state_cache=state_cache)
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim, self.key_dim, self.value_dim],
                dim=-1,
            )
            query = query.view(-1, self.num_k_heads, self.head_k_dim)
            key = key.view(-1, self.num_k_heads, self.head_k_dim)
            value = value.view(-1, self.num_v_heads, self.head_v_dim)
            return self._forward_fused_recurrent(
                batch=batch,
                query=query,
                key=key,
                value=value,
                z=z,
                a=a,
                b=b,
                state_cache=state_cache,
            )

        out = torch.empty_like(x)
        for req, (start, end) in zip(batch.reqs, batch.req_slices):
            if end <= start:
                continue
            req_hidden = x[start:end].unsqueeze(0)
            req_out = self._forward_one(
                req_hidden,
                state_cache=state_cache,
                table_idx=req.table_idx,
                had_previous_state=state_cache.has_previous_state(req.table_idx),
            )
            out[start:end].copy_(req_out[0])
        return out


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id)
        else:
            raise ValueError(f"Unsupported Qwen 3.5 layer type: {self.layer_type}")
        self.mlp = GatedMLP(config)
        self.input_layernorm = Qwen3_5RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Qwen3_5Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        intra_entries: Tuple["HookDispatchEntry", ...] = (),
        contexts: List["HookContext | None"] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if self.layer_type == "linear_attention":
            x = self.linear_attn.forward(x)
        else:
            x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        if contexts is not None and len(intra_entries) > 0:
            x = dispatch_layer_entries(x, self._layer_id, intra_entries, contexts)
        x = self.mlp.forward(x)
        return x, residual


class GatedMLP(BaseOP):
    def __init__(self, config: "ModelConfig") -> None:
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("Qwen3_5MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5Model(BaseOP):
    def __init__(self, config: "ModelConfig") -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList([Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)])
        self.norm = Qwen3_5RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.architecture = "qwen3_5"

    @property
    def num_layers(self) -> int:
        return len(self.layers.op_list)

    @staticmethod
    def _mark_state_cache_rows() -> None:
        ctx = get_global_ctx()
        state_cache = ctx.state_cache
        if not isinstance(state_cache, Qwen3_5StateCache):
            return
        if len(ctx.batch.reqs) > 0:
            state_cache.has_previous.index_fill_(0, ctx.batch.req_table_indices, True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        self._mark_state_cache_rows()
        return self.norm.forward(x, residual)[0]

    def forward_with_hooks(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        layer_dispatch, intra_dispatch, contexts = prepare_hook_runtime(batch, num_layers=self.num_layers)

        x = self.embed_tokens.forward(input_ids)
        x = dispatch_special_point(x, batch, contexts, point="post_embedding")
        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers.op_list):
            intra_entries = intra_dispatch[layer_idx] if intra_dispatch else ()
            x, residual = layer.forward(x, residual, intra_entries=intra_entries, contexts=contexts)
            entries = layer_dispatch[layer_idx] if layer_dispatch else ()
            if len(entries) > 0:
                x = dispatch_layer_entries(x, layer_idx, entries, contexts, residual=residual)

        self._mark_state_cache_rows()
        x = self.norm.forward(x, residual)[0]
        return dispatch_special_point(x, batch, contexts, point="pre_lm_head")


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig") -> None:
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()
        self.supports_cuda_graph = False

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    def create_state_cache(
        self,
        *,
        num_tables: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen3_5StateCache:
        return Qwen3_5StateCache(self.config, num_tables=num_tables, device=device, dtype=dtype)

    def forward(self):
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden_states)
        return self._build_forward_output(hidden_states=hidden_states, logits=logits)

    def forward_with_hooks(self):
        hidden_states = self.model.forward_with_hooks(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden_states)
        return self._build_forward_output(hidden_states=hidden_states, logits=logits)


__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5StateCache"]
