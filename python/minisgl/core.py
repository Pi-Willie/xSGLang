from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.hooks import HookContext, HookSpec, LogitProcessor
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


_STATE_COUNTER = count()
_CONTINUATION_COUNTER = count()
_BLOCK_COUNTER = count()
_SESSION_COUNTER = count()


def _next_state_id() -> str:
    return f"state-{next(_STATE_COUNTER)}"


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


class ContinuationStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ABORTED = "aborted"
    FREED = "freed"


class StepPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ExecutionLane(str, Enum):
    PLAIN = "plain"
    EXTENDED_READ = "extended_read"
    INTERVENTION = "intervention"


class ContinuationCapability(IntFlag):
    NONE = 0
    PLAIN_DECODE = 1 << 0
    READ_HOOKS = 1 << 1
    WRITE_HOOKS = 1 << 2
    EXTRA_OUTPUTS = 1 << 3
    ADAPTER_SWAP = 1 << 4
    CONTINUATION_CONTROL = 1 << 5
    DEBUG_INTROSPECTION = 1 << 6


DEFAULT_ENGINE_CAP_MASK = int(
    ContinuationCapability.PLAIN_DECODE
    | ContinuationCapability.READ_HOOKS
    | ContinuationCapability.WRITE_HOOKS
    | ContinuationCapability.EXTRA_OUTPUTS
    | ContinuationCapability.ADAPTER_SWAP
    | ContinuationCapability.CONTINUATION_CONTROL
    | ContinuationCapability.DEBUG_INTROSPECTION
)

OUTPUT_TOKENS = "tokens"
OUTPUT_TEXT = "text"
OUTPUT_TOPK_IDS = "topk_ids"
OUTPUT_TOPK_LOGPROBS = "topk_logprobs"
OUTPUT_HOOK_OUTPUTS = "hook_outputs"
DEFAULT_TOPK_K = 4
_RUNTIME_OUTPUT_PREFIXES = (
    OUTPUT_TOKENS,
    OUTPUT_TEXT,
    OUTPUT_TOPK_IDS,
    OUTPUT_TOPK_LOGPROBS,
    OUTPUT_HOOK_OUTPUTS,
)


def next_block_id() -> int:
    return next(_BLOCK_COUNTER)


def next_session_id() -> int:
    return next(_SESSION_COUNTER)


def normalize_output_names(requested_outputs: Sequence[str] | str | None) -> Tuple[str, ...]:
    if requested_outputs is None:
        return ()
    if isinstance(requested_outputs, str):
        names = [requested_outputs]
    else:
        names = [str(name) for name in requested_outputs]
    deduped: Dict[str, None] = {}
    for name in names:
        stripped = name.strip()
        if stripped:
            deduped.setdefault(stripped, None)
    return tuple(deduped)


def parse_topk_output(name: str) -> Tuple[str, int] | None:
    if ":" in name:
        prefix, suffix = name.split(":", 1)
        if prefix not in {OUTPUT_TOPK_IDS, OUTPUT_TOPK_LOGPROBS}:
            return None
        k = int(suffix)
        if k <= 0:
            raise ValueError(f"Top-k output '{name}' must use k > 0")
        return prefix, k
    if name in {OUTPUT_TOPK_IDS, OUTPUT_TOPK_LOGPROBS}:
        return name, DEFAULT_TOPK_K
    return None


def split_requested_outputs(
    requested_outputs: Sequence[str] | str | None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], int]:
    runtime_outputs: List[str] = []
    model_outputs: List[str] = []
    max_topk = 0
    for name in normalize_output_names(requested_outputs):
        topk = parse_topk_output(name)
        if topk is not None:
            runtime_outputs.append(name)
            max_topk = max(max_topk, topk[1])
            continue
        if name in _RUNTIME_OUTPUT_PREFIXES:
            runtime_outputs.append(name)
            continue
        model_outputs.append(name)
    return tuple(runtime_outputs), tuple(model_outputs), max_topk


@dataclass(frozen=True)
class ContinuationSpec:
    continuation_id: int
    parent_id: int | None
    session_id: int | None
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    adapter_id: str | None
    engine_cap_mask: int
    active_cap_mask: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ContinuationState:
    output_ids: torch.Tensor
    total_len: int
    decode_len: int
    status: ContinuationStatus
    last_token_id: int | None
    eos_reached: bool
    stop_reason: str | None


@dataclass(frozen=True)
class ContinuationMemory:
    table_row: int
    page_indices: torch.Tensor
    prefix_ref: int | None
    cow_epoch: int
    kv_len: int


@dataclass(frozen=True)
class ContinuationInspection:
    spec: ContinuationSpec
    state: ContinuationState
    memory: ContinuationMemory


@dataclass(frozen=True)
class Session:
    session_id: int
    continuation_ids: Tuple[int, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookProgram:
    hook_spec: HookSpec | None = None
    logit_processor: LogitProcessor | None = None
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    label: str | None = None

    @property
    def has_hooks(self) -> bool:
        return self.hook_spec is not None and self.hook_spec.has_any_hook

    @property
    def has_writes(self) -> bool:
        return bool(self.hook_spec is not None and self.hook_spec.has_writes)


@dataclass(frozen=True)
class ChildContinuationSpec:
    forced_first_token: int | None = None
    sampling_override: SamplingParams | None = None
    hook_program: HookProgram | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    adapter_id: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class BlockSpec:
    continuation_ids: Tuple[int, ...]
    max_new_tokens: int
    min_new_tokens: int = 0
    stop_on_eos: bool = True
    stop_strings: Tuple[str, ...] = ()
    pause_on_finish: bool = True
    request_outputs: Tuple[str, ...] = ()
    hook_program: HookProgram | None = None
    forced_next_tokens: torch.Tensor | None = None
    sampler_override: SamplingParams | Tuple[SamplingParams, ...] | None = None
    block_cap_mask: int = 0
    adapter_id: str | None = None
    block_id: int = field(default_factory=next_block_id)

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("BlockSpec.max_new_tokens must be > 0")
        if self.min_new_tokens < 0:
            raise ValueError("BlockSpec.min_new_tokens must be >= 0")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("BlockSpec.min_new_tokens cannot exceed max_new_tokens")
        if len(self.continuation_ids) == 0:
            raise ValueError("BlockSpec requires at least one continuation")
        object.__setattr__(self, "continuation_ids", tuple(int(v) for v in self.continuation_ids))
        object.__setattr__(self, "request_outputs", normalize_output_names(self.request_outputs))
        object.__setattr__(self, "stop_strings", tuple(str(v) for v in self.stop_strings if v))
        if self.forced_next_tokens is not None and self.forced_next_tokens.numel() != len(
            self.continuation_ids
        ):
            raise ValueError(
                "forced_next_tokens must provide exactly one token per continuation"
            )


@dataclass(frozen=True)
class ContinuationBlockResult:
    continuation_id: int
    emitted_token_ids: torch.Tensor
    text: str | None
    final_status: ContinuationStatus
    stop_reason: str | None
    topk_ids: torch.Tensor | None = None
    topk_logprobs: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    hook_outputs: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BlockResult:
    block_id: int
    lane: ExecutionLane
    continuation_results: Tuple[ContinuationBlockResult, ...]
    steps: int
    elapsed_ms: float


@dataclass(frozen=True)
class StepPlan:
    phase: StepPhase
    lane: ExecutionLane
    continuation_ids: Tuple[int, ...]
    request_outputs: Tuple[str, ...]
    model_outputs: Tuple[str, ...]
    runtime_outputs: Tuple[str, ...]
    topk_k: int
    allow_cuda_graph: bool
    allow_jit_sampler: bool
    sample_next_token: bool = True
    forced_next_tokens: torch.Tensor | None = None
    hook_program: HookProgram | None = None
    block_id: int | None = None


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    hook_spec: HookSpec | None = None
    hook_user_state: Dict[str, Any] = field(default_factory=dict)
    logit_processor: LogitProcessor | None = None
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    adapter_id: str | None = None
    continuation_id: int = field(default_factory=lambda: next(_CONTINUATION_COUNTER))
    parent_id: int | None = None
    session_id: int | None = None
    engine_cap_mask: int = DEFAULT_ENGINE_CAP_MASK
    active_cap_mask: int = DEFAULT_ENGINE_CAP_MASK
    requested_outputs: Tuple[str, ...] = ()
    capture_output_history: bool = False
    auto_free_on_finish: bool = True
    generation_step: int = 0
    status: ContinuationStatus = ContinuationStatus.READY
    eos_reached: bool = False
    stop_reason: str | None = None
    last_token_id: int | None = None
    hook_context: HookContext | None = None
    sampling_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latest_sample_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_output_history: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    state_id: str = field(default_factory=_next_state_id)
    cow_active: bool = False
    _runtime: Any = field(default=None, repr=False, compare=False)
    _kv_view: KVView | None = field(default=None, init=False, repr=False, compare=False)
    _generated_token_ids: List[int] = field(default_factory=list, init=False)
    _input_ids_seed: List[int] | None = field(default=None, repr=False, compare=False)
    _input_ids_list: List[int] = field(default_factory=list, init=False, repr=False)
    _runtime_outputs: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _model_outputs: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _requested_topk_k: int = field(default=0, init=False, repr=False, compare=False)
    _requested_topk_ids: bool = field(default=False, init=False, repr=False, compare=False)
    _requested_topk_logprobs: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.prompt_len = len(self.input_ids)
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        if self._input_ids_seed is None:
            self._input_ids_list = self.input_ids.tolist()
        else:
            self._input_ids_list = list(self._input_ids_seed)
            self._input_ids_seed = None
        self.set_requested_outputs(self.requested_outputs)
        if self.active_cap_mask == DEFAULT_ENGINE_CAP_MASK:
            self.active_cap_mask = self.engine_cap_mask
        self.sampling_state.setdefault("params", self.sampling_params)
        if len(self.metadata) == 0:
            self.metadata = {
                "created_at": time.time(),
                "parent_state_id": None,
                "fork_step": None,
                "lineage": [self.state_id],
            }
        else:
            self.metadata.setdefault("created_at", time.time())
            self.metadata.setdefault("parent_state_id", None)
            self.metadata.setdefault("fork_step", None)
            self.metadata.setdefault("lineage", [self.state_id])
        if self.parent_id is None:
            parent_state_id = self.metadata.get("parent_state_id")
            if parent_state_id is not None:
                parent_cont_id = self.metadata.get("parent_continuation_id")
                self.parent_id = None if parent_cont_id is None else int(parent_cont_id)
        self.metadata.setdefault("continuation_id", self.continuation_id)
        self.metadata.setdefault("session_id", self.session_id)

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1
        self.generation_step += 1
        if self.status is not ContinuationStatus.FINISHED:
            self.status = ContinuationStatus.RUNNING

    def append_token_id(self, token_id: int) -> None:
        self._generated_token_ids.append(token_id)
        self._input_ids_list.append(token_id)
        self.last_token_id = token_id

    def append_host(self, next_token: torch.Tensor) -> None:
        self.append_token_id(int(next_token.item()))

    def materialize_input_ids(self) -> torch.Tensor:
        if len(self.input_ids) != len(self._input_ids_list):
            self.input_ids = torch.tensor(self._input_ids_list, dtype=torch.int32, device="cpu")
        return self.input_ids

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    @property
    def generated_token_ids(self) -> List[int]:
        return self._generated_token_ids

    @property
    def tokens(self) -> List[int]:
        return list(self._input_ids_list)

    @property
    def decode_len(self) -> int:
        return len(self._generated_token_ids)

    @property
    def total_len(self) -> int:
        return self.device_len

    def set_requested_outputs(self, requested_outputs: Sequence[str] | str | None) -> None:
        normalized = normalize_output_names(requested_outputs)
        runtime_outputs, model_outputs, topk_k = split_requested_outputs(normalized)
        self.requested_outputs = normalized
        self._runtime_outputs = runtime_outputs
        self._model_outputs = model_outputs
        self._requested_topk_k = topk_k
        self._requested_topk_ids = any(
            name == OUTPUT_TOPK_IDS or name.startswith(f"{OUTPUT_TOPK_IDS}:")
            for name in runtime_outputs
        )
        self._requested_topk_logprobs = any(
            name == OUTPUT_TOPK_LOGPROBS or name.startswith(f"{OUTPUT_TOPK_LOGPROBS}:")
            for name in runtime_outputs
        )

    @property
    def runtime_outputs(self) -> Tuple[str, ...]:
        return self._runtime_outputs

    @property
    def model_outputs(self) -> Tuple[str, ...]:
        return self._model_outputs

    @property
    def requested_topk_k(self) -> int:
        return self._requested_topk_k

    @property
    def requested_topk_ids(self) -> bool:
        return self._requested_topk_ids

    @property
    def requested_topk_logprobs(self) -> bool:
        return self._requested_topk_logprobs

    @property
    def continuation_spec(self) -> ContinuationSpec:
        prompt_ids = self.materialize_input_ids()[: self.prompt_len].clone()
        return ContinuationSpec(
            continuation_id=self.continuation_id,
            parent_id=self.parent_id,
            session_id=self.session_id,
            input_ids=prompt_ids,
            sampling_params=self.sampling_params,
            adapter_id=self.adapter_id,
            engine_cap_mask=self.engine_cap_mask,
            active_cap_mask=self.active_cap_mask,
            metadata=dict(self.metadata),
        )

    @property
    def continuation_state(self) -> ContinuationState:
        output_ids = torch.tensor(self._generated_token_ids, dtype=torch.int32, device="cpu")
        return ContinuationState(
            output_ids=output_ids,
            total_len=self.total_len,
            decode_len=self.decode_len,
            status=self.status,
            last_token_id=self.last_token_id,
            eos_reached=self.eos_reached,
            stop_reason=self.stop_reason,
        )

    @property
    def continuation_memory(self) -> ContinuationMemory:
        if self._runtime is None:
            page_indices = torch.empty(0, dtype=torch.int32, device="cpu")
        else:
            page_indices = (
                self._runtime.engine.page_table[self.table_idx, : self.device_len]
                .detach()
                .cpu()
                .clone()
            )
        prefix_ref = None if self.cache_handle is None else int(self.cache_handle.cached_len)
        return ContinuationMemory(
            table_row=self.table_idx,
            page_indices=page_indices,
            prefix_ref=prefix_ref,
            cow_epoch=int(self.metadata.get("cow_epoch", 0)),
            kv_len=self.cached_len,
        )

    @property
    def continuation(self) -> ContinuationInspection:
        return ContinuationInspection(
            spec=self.continuation_spec,
            state=self.continuation_state,
            memory=self.continuation_memory,
        )

    @property
    def kv_handle(self) -> KVHandle:
        return KVHandle(
            table_idx=self.table_idx,
            length=self.device_len,
            state_id=self.state_id,
        )

    @property
    def position(self) -> int:
        return self.device_len

    @property
    def user_state(self) -> Dict[str, Any]:
        return self.hook_user_state

    @property
    def hook_spec_or_none(self) -> HookSpec | None:
        return self.hook_spec

    @property
    def available_sample_outputs(self) -> Tuple[str, ...]:
        return self.requested_outputs

    def bind_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def kv_view(self) -> KVView:
        if self._kv_view is None:
            self._kv_view = KVView(self)
        return self._kv_view

    def fork(self) -> Req:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot fork")
        return runtime.fork_state(self, add_to_scheduler=True, snapshot=False)

    def spawn_children(self, child_specs: Sequence[ChildContinuationSpec] | None = None) -> List[Req]:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot spawn children")
        specs = list(child_specs or [ChildContinuationSpec()])
        return runtime.spawn_children(self, specs)

    def snapshot(self) -> Req:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot snapshot")
        return runtime.fork_state(self, add_to_scheduler=False, snapshot=True)

    def restore(self, snapshot: Req) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot restore")
        runtime.restore_state(self, snapshot)

    def step(self) -> int:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot step")
        return runtime.step_state(self)

    def run_block(
        self,
        *,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        stop_on_eos: bool = True,
        stop_strings: Sequence[str] | None = None,
        request_outputs: Sequence[str] | str | None = None,
        hook_program: HookProgram | None = None,
        forced_next_token: int | None = None,
        sampler_override: SamplingParams | None = None,
        adapter_id: str | None = None,
    ) -> BlockResult:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot run block")
        forced = None
        if forced_next_token is not None:
            forced = torch.tensor([forced_next_token], dtype=torch.int32)
        return runtime.run_block(
            BlockSpec(
                continuation_ids=(self.continuation_id,),
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                stop_on_eos=stop_on_eos,
                stop_strings=tuple(stop_strings or ()),
                request_outputs=normalize_output_names(request_outputs),
                hook_program=hook_program,
                forced_next_tokens=forced,
                sampler_override=sampler_override,
                adapter_id=adapter_id,
            )
        )

    def trace_hidden_states(
        self,
        *,
        layers: Sequence[int] | str,
        max_new_tokens: int,
        capture: str = "last_token",
        to_cpu: bool = True,
        request_outputs: Sequence[str] | str | None = None,
        hook_program: HookProgram | None = None,
        min_new_tokens: int = 0,
        stop_on_eos: bool = True,
        stop_strings: Sequence[str] | None = None,
        sampler_override: SamplingParams | None = None,
        adapter_id: str | None = None,
    ):
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot trace hidden states")
        if not hasattr(runtime, "trace_hidden_states"):
            raise RuntimeError("Bound runtime does not implement trace_hidden_states")
        return runtime.trace_hidden_states(
            self,
            layers=layers,
            max_new_tokens=max_new_tokens,
            capture=capture,
            to_cpu=to_cpu,
            request_outputs=request_outputs,
            hook_program=hook_program,
            min_new_tokens=min_new_tokens,
            stop_on_eos=stop_on_eos,
            stop_strings=stop_strings,
            sampler_override=sampler_override,
            adapter_id=adapter_id,
        )

    def inspect(self) -> ContinuationInspection:
        return self.continuation

    @staticmethod
    def step_many(states: Sequence[Req]) -> List[int]:
        if len(states) == 0:
            return []
        runtime = states[0]._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot step_many")
        for state in states:
            if state._runtime is None:
                raise RuntimeError("State runtime is not bound; cannot step_many")
            if state._runtime is not runtime:
                raise RuntimeError("All states in step_many must share the same runtime")
        if hasattr(runtime, "step_states"):
            return runtime.step_states(states)
        return [runtime.step_state(state) for state in states]

    def close(self) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound; cannot close")
        runtime.dispose_state(self)
        self.status = ContinuationStatus.FREED

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


Continuation = Req
GenerationState = Req


@dataclass(frozen=True)
class KVHandle:
    table_idx: int
    length: int
    state_id: str


class KVView:
    def __init__(self, state: Req):
        self._state = state

    def _runtime(self) -> Any:
        runtime = self._state._runtime
        if runtime is None:
            raise RuntimeError("State runtime is not bound")
        return runtime

    def _ctx(self) -> Context:
        return get_global_ctx()

    def _kvcache(self):
        return self._ctx().kv_cache

    def _page_table(self) -> torch.Tensor:
        return self._ctx().page_table

    def _layer_cache(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self._kvcache()
        if layer < 0 or layer >= cache.num_layers:
            raise ValueError(f"Layer index out of range: {layer}")
        if not cache.has_layer(layer):
            raise ValueError(f"Layer {layer} does not expose paged KV cache")
        k_cache = cache.k_cache(layer)
        v_cache = cache.v_cache(layer)
        k_flat = k_cache.view((-1,) + tuple(k_cache.shape[2:]))
        v_flat = v_cache.view((-1,) + tuple(v_cache.shape[2:]))
        return k_flat, v_flat

    @staticmethod
    def _normalize_positions(
        positions: int | slice | Sequence[int],
        *,
        upper_bound: int,
    ) -> List[int]:
        if isinstance(positions, int):
            if positions < 0:
                raise ValueError(f"Negative position is not allowed: {positions}")
            return [positions]
        if isinstance(positions, slice):
            start = 0 if positions.start is None else positions.start
            stop = upper_bound if positions.stop is None else positions.stop
            step = 1 if positions.step is None else positions.step
            if step == 0:
                raise ValueError("Slice step cannot be zero")
            return list(range(start, stop, step))
        out = [int(p) for p in positions]
        if any(p < 0 for p in out):
            raise ValueError("Negative positions are not allowed")
        return out

    def _read_indices(self, positions: List[int]) -> torch.Tensor:
        if len(positions) == 0:
            return torch.empty(0, device=self._ctx().page_table.device, dtype=torch.int64)
        max_pos = max(positions)
        if max_pos >= self._state.device_len:
            raise ValueError(
                f"Position {max_pos} is out of bounds for state length {self._state.device_len}"
            )
        pos_tensor = torch.tensor(
            positions,
            device=self._ctx().page_table.device,
            dtype=torch.int64,
        )
        return self._page_table()[self._state.table_idx, pos_tensor].to(torch.int64)

    def read(
        self,
        layer: int,
        position: int | slice | Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = self._normalize_positions(position, upper_bound=self._state.device_len)
        indices = self._read_indices(positions)
        k_flat, v_flat = self._layer_cache(layer)
        k_block = k_flat[indices]
        v_block = v_flat[indices]
        if isinstance(position, int):
            return k_block[0], v_block[0]
        return k_block, v_block

    def read_all_layers(self, position: int) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        values: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in self._kvcache().iter_layer_ids():
            values[layer_idx] = self.read(layer_idx, position)
        return values

    def _project_query(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        runtime = self._runtime()
        model = runtime.engine.model.model
        layer_mod = model.layers.op_list[layer]
        self_attn = layer_mod.self_attn
        qkv_proj = self_attn.qkv_proj
        x = hidden.to(device=runtime.device, dtype=qkv_proj.weight.dtype).view(1, -1)
        qkv = qkv_proj.forward(x)
        attn = self_attn.attn
        q = qkv[0, : attn.qo_attn_dim].view(attn.num_qo_heads, attn.head_dim)
        if self_attn.q_norm is not None:
            self_attn.q_norm.forward_inplace(q)
        return q

    def attention_scores(
        self,
        layer: int,
        query: torch.Tensor,
        positions: int | slice | Sequence[int],
    ) -> torch.Tensor:
        pos = self._normalize_positions(positions, upper_bound=self._state.device_len)
        k_block, _ = self.read(layer, pos)
        assert k_block.dim() == 3

        if query.dim() == 1:
            q = self._project_query(layer, query)
        elif query.dim() == 2:
            q = query
        else:
            raise ValueError(
                f"Query must be 1D hidden state or 2D projected Q, got shape {tuple(query.shape)}"
            )

        q = q.to(device=k_block.device, dtype=torch.float32)
        k_block = k_block.to(torch.float32)
        num_q_heads = q.shape[0]
        num_kv_heads = k_block.shape[1]
        if num_q_heads != num_kv_heads:
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    "Projected query head count must be divisible by KV head count: "
                    f"{num_q_heads} vs {num_kv_heads}"
                )
            k_block = k_block.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        return torch.einsum("hd,phd->hp", q, k_block) / math.sqrt(float(q.shape[1]))

    def _coerce_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_flat, _ = self._layer_cache(layer)
        expected = k_flat.shape[1:]
        if tuple(k.shape) != tuple(expected):
            raise ValueError(f"K shape mismatch: expected {tuple(expected)}, got {tuple(k.shape)}")
        if tuple(v.shape) != tuple(expected):
            raise ValueError(f"V shape mismatch: expected {tuple(expected)}, got {tuple(v.shape)}")
        return (
            k.to(device=k_flat.device, dtype=k_flat.dtype),
            v.to(device=k_flat.device, dtype=k_flat.dtype),
        )

    def write(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if position < 0:
            raise ValueError(f"Negative position is not allowed: {position}")
        if position >= self._state.device_len:
            raise ValueError(
                f"Position {position} out of bounds for write with state length {self._state.device_len}"
            )
        k, v = self._coerce_kv(layer, k, v)
        token_index = self._runtime()._ensure_state_writable_token_index(self._state, position)
        k_flat, v_flat = self._layer_cache(layer)
        k_flat[token_index].copy_(k)
        v_flat[token_index].copy_(v)

    def zero(self, layer: int, positions: int | slice | Sequence[int]) -> None:
        for pos in self._normalize_positions(positions, upper_bound=self._state.device_len):
            k, v = self.read(layer, pos)
            self.write(layer, pos, torch.zeros_like(k), torch.zeros_like(v))

    def inject(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if position < 0:
            raise ValueError(f"Negative position is not allowed: {position}")
        runtime = self._runtime()
        runtime._ensure_state_position(self._state, position + 1)
        self.write(layer, position, k, v)
        if self._state.cached_len < position + 1:
            self._state.cached_len = position + 1

    def transplant_from(
        self,
        source_view: KVView,
        layer: int,
        source_positions: int | slice | Sequence[int],
        target_positions: int | slice | Sequence[int],
    ) -> None:
        src = source_view._normalize_positions(
            source_positions,
            upper_bound=source_view._state.device_len,
        )
        dst = self._normalize_positions(target_positions, upper_bound=self._state.device_len)
        if len(src) != len(dst):
            raise ValueError(
                f"Source/target position length mismatch: {len(src)} vs {len(dst)}"
            )
        k_block, v_block = source_view.read(layer, src)
        assert k_block.dim() == 3 and v_block.dim() == 3
        for i, pos in enumerate(dst):
            self.write(layer, pos, k_block[i], v_block[i])

    def blend_from(
        self,
        source_view: KVView,
        layer: int,
        positions: int | slice | Sequence[int],
        alpha: float,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        pos = self._normalize_positions(positions, upper_bound=self._state.device_len)
        own_k, own_v = self.read(layer, pos)
        src_k, src_v = source_view.read(layer, pos)
        assert own_k.dim() == 3 and own_v.dim() == 3
        blended_k = own_k * (1.0 - alpha) + src_k * alpha
        blended_v = own_v * (1.0 - alpha) + src_v * alpha
        for i, p in enumerate(pos):
            self.write(layer, p, blended_k[i], blended_v[i])

    def diff(self, other_view: KVView, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        count = min(self._state.device_len, other_view._state.device_len)
        positions = list(range(count))
        k_self, v_self = self.read(layer, positions)
        k_other, v_other = other_view.read(layer, positions)
        return k_self - k_other, v_self - v_other

    def scale(
        self,
        layer: int,
        positions: int | slice | Sequence[int],
        k_scale: float,
        v_scale: float,
    ) -> None:
        pos = self._normalize_positions(positions, upper_bound=self._state.device_len)
        k_block, v_block = self.read(layer, pos)
        assert k_block.dim() == 3 and v_block.dim() == 3
        k_scaled = k_block * float(k_scale)
        v_scaled = v_block * float(v_scale)
        for i, p in enumerate(pos):
            self.write(layer, p, k_scaled[i], v_scaled[i])


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    plan: StepPlan | None = None
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    req_table_indices: torch.Tensor = field(init=False)
    req_table_indices_i32: torch.Tensor = field(init=False)
    req_cu_seqlens: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    has_hooked_requests: bool = field(init=False)
    has_logit_processors: bool = field(init=False)
    requested_sample_outputs: Tuple[str, ...] = field(init=False)
    requested_runtime_outputs: Tuple[str, ...] = field(init=False)
    requested_topk_k: int = field(init=False)
    requested_topk_ids: bool = field(init=False)
    requested_topk_logprobs: bool = field(init=False)
    lane: ExecutionLane = field(init=False)
    sample_next_token: bool = field(init=False)
    allow_cuda_graph: bool = field(init=False)
    forced_next_tokens: torch.Tensor | None = field(init=False)
    req_slices: List[Tuple[int, int]] = field(init=False)
    hook_contexts: List[HookContext | None] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.padded_reqs = self.reqs
        self._refresh_runtime_metadata()
        self.req_slices = []
        offset = 0
        for req in self.reqs:
            end = offset + req.extend_len
            self.req_slices.append((offset, end))
            offset = end

    def _refresh_runtime_metadata(self) -> None:
        runtime_outputs: Dict[str, None] = {}
        model_outputs: Dict[str, None] = {}
        requested_topk = 0
        for req in self.reqs:
            for name in req.runtime_outputs:
                runtime_outputs.setdefault(name, None)
            for name in req.model_outputs:
                model_outputs.setdefault(name, None)
            requested_topk = max(requested_topk, req.requested_topk_k)
        if self.plan is not None:
            for name in self.plan.runtime_outputs:
                runtime_outputs.setdefault(name, None)
            for name in self.plan.model_outputs:
                model_outputs.setdefault(name, None)
            requested_topk = max(requested_topk, self.plan.topk_k)
        runtime_output_names = tuple(runtime_outputs)
        self.requested_runtime_outputs = runtime_output_names
        self.requested_sample_outputs = tuple(model_outputs)
        self.requested_topk_k = requested_topk
        self.requested_topk_ids = any(
            name == OUTPUT_TOPK_IDS or name.startswith(f"{OUTPUT_TOPK_IDS}:")
            for name in runtime_output_names
        )
        self.requested_topk_logprobs = any(
            name == OUTPUT_TOPK_LOGPROBS or name.startswith(f"{OUTPUT_TOPK_LOGPROBS}:")
            for name in runtime_output_names
        )
        self.lane = self.plan.lane if self.plan is not None else ExecutionLane.PLAIN
        self.sample_next_token = self.plan.sample_next_token if self.plan is not None else True
        self.allow_cuda_graph = self.plan.allow_cuda_graph if self.plan is not None else True
        self.forced_next_tokens = self.plan.forced_next_tokens if self.plan is not None else None
        self.has_hooked_requests = any(
            req.hook_spec is not None and req.hook_spec.has_any_hook for req in self.reqs
        )
        if self.plan is not None and self.plan.hook_program is not None:
            self.has_hooked_requests = self.has_hooked_requests or self.plan.hook_program.has_hooks
        self.has_logit_processors = any(req.logit_processor is not None for req in self.reqs)
        if self.plan is not None and self.plan.hook_program is not None:
            self.has_logit_processors = self.has_logit_processors or (
                self.plan.hook_program.logit_processor is not None
            )

    def bind_plan(self, plan: StepPlan) -> None:
        self.plan = plan
        self.phase = plan.phase.value
        self._refresh_runtime_metadata()

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    state_cache: Any | None = field(default=None, init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def clear_global_ctx(ctx: Context | None = None):
    global _GLOBAL_CTX
    if ctx is not None and _GLOBAL_CTX is not None and _GLOBAL_CTX is not ctx:
        raise AssertionError("Attempted to clear a different global context")
    _GLOBAL_CTX = None


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
