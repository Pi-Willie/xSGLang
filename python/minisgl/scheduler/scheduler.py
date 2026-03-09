from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, NoReturn, Sequence, Set, Tuple, TypeAlias

import torch
from minisgl.core import (
    Batch,
    BlockResult,
    BlockSpec,
    ChildContinuationSpec,
    ContinuationCapability,
    ContinuationBlockResult,
    ContinuationInspection,
    ContinuationStatus,
    DEFAULT_ENGINE_CAP_MASK,
    ExecutionLane,
    HookProgram,
    OUTPUT_HOOK_OUTPUTS,
    OUTPUT_TEXT,
    OUTPUT_TOPK_IDS,
    OUTPUT_TOPK_LOGPROBS,
    OUTPUT_TOKENS,
    Req,
    Session,
    StepPhase,
    StepPlan,
    next_session_id,
    normalize_output_names,
    split_requested_outputs,
)
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.message import (
    AdapterBackendControlMsg,
    AdapterResultMsg,
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import PinnedRingBuffer, init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillAdder, PrefillManager
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class _BlockBindingSnapshot(NamedTuple):
    sampling_params: Any
    hook_spec: Any
    logit_processor: Any
    hook_config: Dict[str, Any] | None
    hook_preset_name: str | None
    requested_outputs: tuple[str, ...]
    status: ContinuationStatus
    stop_reason: str | None


class _ResolvedOutputRequest(NamedTuple):
    effective_outputs: Tuple[str, ...]
    runtime_outputs: Tuple[str, ...]
    model_outputs: Tuple[str, ...]
    topk_k: int


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
            model_config=config.model_config,
            state_cache=self.engine.state_cache,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            state_cache=self.engine.state_cache,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.tokenizer_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        self._state_registry: Dict[str, Req] = {}
        self._continuation_registry: Dict[int, Req] = {}
        self._session_members: Dict[int, Set[int]] = {}
        self._session_metadata: Dict[int, Dict[str, Any]] = {}
        self.engine_cap_mask = getattr(config, "engine_cap_mask", 0) or DEFAULT_ENGINE_CAP_MASK
        self._next_internal_uid = -1
        self._pending_adapter_control: List[AdapterBackendControlMsg] = []
        self._available_sample_outputs = self.engine.model.available_sample_output_names()
        self._available_sample_output_set = frozenset(self._available_sample_outputs)
        self._positions_buffer = PinnedRingBuffer(self.device, torch.int32, slots=2, min_capacity=256)
        self._positions_index_buffer = PinnedRingBuffer(
            self.device, torch.int64, slots=2, min_capacity=256
        )
        self._input_mapping_buffer = PinnedRingBuffer(
            self.device, torch.int64, slots=2, min_capacity=256
        )
        self._req_table_buffer = PinnedRingBuffer(self.device, torch.int64, slots=2, min_capacity=256)
        self._req_table_i32_buffer = PinnedRingBuffer(
            self.device, torch.int32, slots=2, min_capacity=256
        )
        self._req_cu_seqlens_buffer = PinnedRingBuffer(
            self.device, torch.int32, slots=2, min_capacity=256
        )
        self._write_mapping_buffer = PinnedRingBuffer(
            self.device, torch.int64, slots=2, min_capacity=256
        )
        self._write_index_buffer = PinnedRingBuffer(
            self.device, torch.int64, slots=2, min_capacity=256
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self._pending_adapter_control:
            self._process_last_data(last_data)
            self._handle_pending_adapter_control()
            return None

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self._pending_adapter_control:
            self._handle_pending_adapter_control()
            return

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _next_uid(self) -> int:
        uid = self._next_internal_uid
        self._next_internal_uid -= 1
        return uid

    def _normalize_requested_outputs(self, requested_outputs: Any) -> tuple[str, ...]:
        normalized = normalize_output_names(requested_outputs)
        _, model_outputs, _ = split_requested_outputs(normalized)
        available_outputs = getattr(self, "_available_sample_outputs", None)
        available_output_set = getattr(self, "_available_sample_output_set", None)
        if available_outputs is None or available_output_set is None:
            available_outputs = self.engine.model.available_sample_output_names()
            available_output_set = frozenset(available_outputs)
            self._available_sample_outputs = available_outputs
            self._available_sample_output_set = available_output_set
        unknown = sorted(set(model_outputs) - available_output_set)
        if unknown:
            raise ValueError(
                f"Unknown requested outputs {unknown}. "
                f"Available outputs: {list(available_outputs)}"
            )
        return normalized

    def _register_state(self, req: Req) -> None:
        existing_state = self._state_registry.get(req.state_id)
        if existing_state is req:
            return
        if existing_state is not None and existing_state is not req:
            raise RuntimeError(f"State registry collision for {req.state_id}")
        existing_continuation = self._continuation_registry.get(req.continuation_id)
        if existing_continuation is not None and existing_continuation is not req:
            raise RuntimeError(
                f"Continuation registry collision for {req.continuation_id}"
            )
        req.bind_runtime(self)
        self._state_registry[req.state_id] = req
        self._continuation_registry[req.continuation_id] = req
        if req.session_id is not None:
            self._session_members.setdefault(req.session_id, set()).add(req.continuation_id)

    def _unregister_state(self, req: Req) -> None:
        self._state_registry.pop(req.state_id, None)
        self._continuation_registry.pop(req.continuation_id, None)
        if req.session_id is not None:
            members = self._session_members.get(req.session_id)
            if members is not None:
                members.discard(req.continuation_id)
                if len(members) == 0:
                    self._session_members.pop(req.session_id, None)
                    self._session_metadata.pop(req.session_id, None)

    def _tracked_pages(self, req: Req, *, upto: int | None = None) -> Set[int]:
        return set(self.cache_manager.state_page_starts(req, upto=upto))

    def open_session(self, metadata: Dict[str, Any] | None = None) -> Session:
        session_id = next_session_id()
        self._session_members.setdefault(session_id, set())
        self._session_metadata[session_id] = dict(metadata or {})
        return Session(
            session_id=session_id,
            continuation_ids=(),
            metadata=dict(self._session_metadata[session_id]),
        )

    def inspect_session(self, session_id: int) -> Session:
        if session_id not in self._session_members:
            raise KeyError(f"Unknown session_id: {session_id}")
        members = tuple(sorted(self._session_members[session_id]))
        return Session(
            session_id=session_id,
            continuation_ids=members,
            metadata=dict(self._session_metadata.get(session_id, {})),
        )

    def resolve_continuation(self, continuation: Req | int) -> Req:
        if isinstance(continuation, Req):
            if continuation._runtime is not self:
                raise RuntimeError("Continuation does not belong to this scheduler")
            return continuation
        req = self._continuation_registry.get(int(continuation))
        if req is None:
            raise KeyError(f"Unknown continuation_id: {continuation}")
        return req

    def inspect_continuation(self, continuation: Req | int) -> ContinuationInspection:
        return self.resolve_continuation(continuation).continuation

    def _required_cap_mask(
        self,
        *,
        hook_program: HookProgram | None,
        reqs: Sequence[Req],
        requested_outputs: Sequence[str],
        forced_next_tokens: torch.Tensor | None,
        adapter_id: str | None,
    ) -> int:
        outputs = self._resolve_output_request(reqs=reqs, requested_outputs=requested_outputs)
        required = int(ContinuationCapability.PLAIN_DECODE | ContinuationCapability.CONTINUATION_CONTROL)
        has_hooks, has_writes, has_logit_processors = self._resolve_hook_activity(
            reqs=reqs,
            hook_program=hook_program,
        )
        if has_hooks:
            required |= int(ContinuationCapability.READ_HOOKS)
        if has_writes or has_logit_processors or forced_next_tokens is not None:
            required |= int(ContinuationCapability.WRITE_HOOKS)
        if outputs.runtime_outputs or outputs.model_outputs:
            required |= int(
                ContinuationCapability.EXTRA_OUTPUTS | ContinuationCapability.DEBUG_INTROSPECTION
            )
        if adapter_id is not None or any(req.adapter_id is not None for req in reqs):
            required |= int(ContinuationCapability.ADAPTER_SWAP)
        return required

    def _assert_capabilities(
        self,
        reqs: Sequence[Req],
        *,
        requested_outputs: Sequence[str],
        hook_program: HookProgram | None,
        forced_next_tokens: torch.Tensor | None,
        adapter_id: str | None,
    ) -> None:
        required = self._required_cap_mask(
            hook_program=hook_program,
            reqs=reqs,
            requested_outputs=requested_outputs,
            forced_next_tokens=forced_next_tokens,
            adapter_id=adapter_id,
        )
        missing_engine = required & ~self.engine_cap_mask
        if missing_engine:
            raise RuntimeError(
                f"Block requires capabilities 0x{required:x}, but engine only exposes 0x{self.engine_cap_mask:x}"
            )
        for req in reqs:
            missing_req = required & ~req.active_cap_mask
            if missing_req:
                raise RuntimeError(
                    f"Continuation {req.continuation_id} forbids capabilities 0x{missing_req:x}"
                )

    def _assert_active_cap_mask(
        self,
        *,
        active_cap_mask: int | None,
        requested_outputs: Sequence[str],
        hook_program: HookProgram | None,
        forced_next_tokens: torch.Tensor | None,
        adapter_id: str | None,
    ) -> None:
        if active_cap_mask is None:
            return
        required = self._required_cap_mask(
            hook_program=hook_program,
            reqs=(),
            requested_outputs=requested_outputs,
            forced_next_tokens=forced_next_tokens,
            adapter_id=adapter_id,
        )
        missing = required & ~int(active_cap_mask)
        if missing:
            raise RuntimeError(f"Continuation active_cap_mask forbids capabilities 0x{missing:x}")

    def _assert_block_cap_mask(self, block: BlockSpec, reqs: Sequence[Req]) -> None:
        if block.block_cap_mask == 0:
            return
        required = self._required_cap_mask(
            hook_program=block.hook_program,
            reqs=reqs,
            requested_outputs=block.request_outputs,
            forced_next_tokens=block.forced_next_tokens,
            adapter_id=block.adapter_id,
        )
        missing = required & ~int(block.block_cap_mask)
        if missing:
            raise RuntimeError(f"Block {block.block_id} block_cap_mask forbids capabilities 0x{missing:x}")

    def _compile_step_plan(
        self,
        *,
        reqs: Sequence[Req],
        phase: StepPhase,
        requested_outputs: Sequence[str],
        hook_program: HookProgram | None = None,
        forced_next_tokens: torch.Tensor | None = None,
        sample_next_token: bool = True,
        block_id: int | None = None,
    ) -> StepPlan:
        outputs = self._resolve_output_request(reqs=reqs, requested_outputs=requested_outputs)
        has_hooks, has_writes, has_logit_processors = self._resolve_hook_activity(
            reqs=reqs,
            hook_program=hook_program,
        )

        if forced_next_tokens is not None or has_writes or has_logit_processors:
            lane = ExecutionLane.INTERVENTION
        elif has_hooks or outputs.runtime_outputs or outputs.model_outputs:
            lane = ExecutionLane.EXTENDED_READ
        else:
            lane = ExecutionLane.PLAIN

        allow_cuda_graph = (
            phase is StepPhase.DECODE
            and lane is ExecutionLane.PLAIN
            and sample_next_token
            and outputs.topk_k == 0
        )
        allow_jit_sampler = forced_next_tokens is None and not has_logit_processors
        return StepPlan(
            phase=phase,
            lane=lane,
            continuation_ids=tuple(req.continuation_id for req in reqs),
            request_outputs=outputs.effective_outputs,
            model_outputs=outputs.model_outputs,
            runtime_outputs=outputs.runtime_outputs,
            topk_k=outputs.topk_k,
            allow_cuda_graph=allow_cuda_graph,
            allow_jit_sampler=allow_jit_sampler,
            sample_next_token=sample_next_token,
            forced_next_tokens=forced_next_tokens,
            hook_program=hook_program,
            block_id=block_id,
        )

    @staticmethod
    def _merge_requested_outputs(
        reqs: Sequence[Req],
        requested_outputs: Sequence[str],
    ) -> Tuple[str, ...]:
        merged: Dict[str, None] = dict.fromkeys(normalize_output_names(requested_outputs))
        for req in reqs:
            for name in req.requested_outputs:
                merged.setdefault(name, None)
        return tuple(merged)

    @classmethod
    def _resolve_output_request(
        cls,
        *,
        reqs: Sequence[Req],
        requested_outputs: Sequence[str],
    ) -> _ResolvedOutputRequest:
        effective_outputs = cls._merge_requested_outputs(reqs, requested_outputs)
        runtime_outputs, model_outputs, topk_k = split_requested_outputs(effective_outputs)
        return _ResolvedOutputRequest(
            effective_outputs=effective_outputs,
            runtime_outputs=runtime_outputs,
            model_outputs=model_outputs,
            topk_k=topk_k,
        )

    @staticmethod
    def _resolve_hook_activity(
        *,
        reqs: Sequence[Req],
        hook_program: HookProgram | None,
    ) -> Tuple[bool, bool, bool]:
        has_hooks = any(req.hook_spec is not None and req.hook_spec.has_any_hook for req in reqs)
        has_writes = any(req.hook_spec is not None and req.hook_spec.has_writes for req in reqs)
        has_logit_processors = any(req.logit_processor is not None for req in reqs)
        if hook_program is not None:
            has_hooks = has_hooks or hook_program.has_hooks
            has_writes = has_writes or hook_program.has_writes
            has_logit_processors = has_logit_processors or (hook_program.logit_processor is not None)
        return has_hooks, has_writes, has_logit_processors

    def _build_root_req(
        self,
        *,
        input_ids: torch.Tensor,
        sampling_params: Any,
        hook_spec: Any,
        logit_processor: Any,
        hook_config: Dict[str, Any] | None,
        hook_preset_name: str | None,
        adapter_id: str | None,
        requested_outputs: tuple[str, ...],
        capture_output_history: bool,
        session_id: int | None,
        active_cap_mask: int | None,
        metadata: Dict[str, Any] | None,
    ) -> Req:
        pending = PendingReq(
            uid=self._next_uid(),
            input_ids=input_ids.to(torch.int32),
            sampling_params=sampling_params,
            hook_spec=hook_spec,
            logit_processor=logit_processor,
            hook_config=hook_config,
            hook_preset_name=hook_preset_name,
            adapter_id=adapter_id,
            requested_outputs=requested_outputs,
            capture_output_history=bool(capture_output_history),
        )
        adder = PrefillAdder(
            token_budget=max(self.prefill_budget, pending.input_len + 1),
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            state_cache=self.engine.state_cache,
        )
        req = adder.try_add_one(pending)
        if req is None:
            raise RuntimeError("Unable to allocate resources for new continuation")
        if isinstance(req, ChunkedReq):
            raise RuntimeError(
                "Chunked prefill is not supported in open_continuation; increase prefill budget"
            )
        req.session_id = session_id
        req.engine_cap_mask = self.engine_cap_mask
        req.active_cap_mask = self.engine_cap_mask if active_cap_mask is None else int(active_cap_mask)
        req.auto_free_on_finish = False
        if metadata:
            req.metadata.update(copy.deepcopy(metadata))
        req.metadata["continuation_id"] = req.continuation_id
        req.metadata["session_id"] = session_id
        return req

    def open_continuation_from_ids(
        self,
        input_ids: torch.Tensor,
        sampling_params: Any,
        *,
        hook_spec: Any = None,
        logit_processor: Any = None,
        hook_config: Dict[str, Any] | None = None,
        hook_preset_name: str | None = None,
        adapter_id: str | None = None,
        requested_outputs: Any = None,
        capture_output_history: bool = False,
        session_id: int | None = None,
        active_cap_mask: int | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Req:
        if not input_ids.is_cpu:
            raise ValueError("input_ids must be a CPU tensor")
        if input_ids.dim() != 1:
            raise ValueError("input_ids must be a 1D tensor")
        normalized_outputs = self._normalize_requested_outputs(requested_outputs)
        if session_id is not None:
            self._session_members.setdefault(session_id, set())
            self._session_metadata.setdefault(session_id, {})
        self._assert_capabilities(
            (),
            requested_outputs=normalized_outputs,
            hook_program=HookProgram(
                hook_spec=hook_spec,
                logit_processor=logit_processor,
                hook_config=hook_config,
                hook_preset_name=hook_preset_name,
            ),
            forced_next_tokens=None,
            adapter_id=adapter_id,
        )
        self._assert_active_cap_mask(
            active_cap_mask=active_cap_mask,
            requested_outputs=normalized_outputs,
            hook_program=HookProgram(
                hook_spec=hook_spec,
                logit_processor=logit_processor,
                hook_config=hook_config,
                hook_preset_name=hook_preset_name,
            ),
            forced_next_tokens=None,
            adapter_id=adapter_id,
        )
        req = self._build_root_req(
            input_ids=input_ids,
            sampling_params=sampling_params,
            hook_spec=hook_spec,
            logit_processor=logit_processor,
            hook_config=hook_config,
            hook_preset_name=hook_preset_name,
            adapter_id=adapter_id,
            requested_outputs=normalized_outputs,
            capture_output_history=capture_output_history,
            session_id=session_id,
            active_cap_mask=active_cap_mask,
            metadata=metadata,
        )
        # Opening a continuation is a KV residency operation, not an implicit decode.
        plan = self._compile_step_plan(
            reqs=[req],
            phase=StepPhase.PREFILL,
            requested_outputs=req.requested_outputs,
            sample_next_token=False,
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.bind_plan(plan)
        with self.engine_stream_ctx:
            forward_input = self._prepare_batch(batch)
            forward_output = self._forward(forward_input)
        self._finalize_batch(
            batch=forward_input.batch,
            forward_output=forward_output,
            emit_reply=False,
        )
        self.decode_manager.remove_req(req)
        self._prime_prefilled_continuation_for_decode(req)
        return req

    def create_state_from_ids(
        self,
        input_ids: torch.Tensor,
        sampling_params: Any,
        *,
        hook_spec: Any = None,
        logit_processor: Any = None,
        hook_config: Dict[str, Any] | None = None,
        hook_preset_name: str | None = None,
        adapter_id: str | None = None,
        requested_outputs: Any = None,
        capture_output_history: bool = False,
    ) -> Req:
        return self.open_continuation_from_ids(
            input_ids=input_ids,
            sampling_params=sampling_params,
            hook_spec=hook_spec,
            logit_processor=logit_processor,
            hook_config=hook_config,
            hook_preset_name=hook_preset_name,
            adapter_id=adapter_id,
            requested_outputs=requested_outputs,
            capture_output_history=capture_output_history,
        )

    def step_state(self, state: Req) -> int:
        return self.step_states([state])[0]

    def step_states(self, states: Sequence[Req]) -> List[int]:
        if len(states) == 0:
            return []
        seen_state_ids: Set[str] = set()
        for state in states:
            if state._runtime is not self:
                raise RuntimeError("State does not belong to this scheduler")
            if state.state_id not in self._state_registry:
                raise RuntimeError("State is no longer active")
            if not state.can_decode:
                raise RuntimeError("State cannot decode: no remaining tokens")
            if state.state_id in seen_state_ids:
                raise ValueError(f"Duplicate state passed to step_states: {state.state_id}")
            seen_state_ids.add(state.state_id)
        block = BlockSpec(
            continuation_ids=tuple(state.continuation_id for state in states),
            max_new_tokens=1,
            request_outputs=(OUTPUT_TOKENS,),
        )
        result = self.run_block(block)
        outputs: List[int] = []
        for continuation_result in result.continuation_results:
            if continuation_result.emitted_token_ids.numel() != 1:
                raise RuntimeError(
                    "step_states expected exactly one emitted token per continuation"
                )
            outputs.append(int(continuation_result.emitted_token_ids[0].item()))
        return outputs

    @staticmethod
    def _expand_sampler_override(
        sampler_override: Any,
        n: int,
    ) -> List[Any]:
        if sampler_override is None:
            return [None] * n
        if isinstance(sampler_override, Sequence) and not isinstance(
            sampler_override, (str, bytes, dict)
        ):
            if len(sampler_override) != n:
                raise ValueError(
                    f"sampler_override length {len(sampler_override)} does not match block size {n}"
                )
            return list(sampler_override)
        return [sampler_override] * n

    @contextmanager
    def _temporary_block_bindings(self, reqs: Sequence[Req], block: BlockSpec):
        # Blocks are the only place where hook programs and sampler overrides become active.
        sampler_overrides = self._expand_sampler_override(block.sampler_override, len(reqs))
        target_ignore_eos = not block.stop_on_eos
        block_outputs = block.request_outputs if block.request_outputs else None
        saved: List[Tuple[Req, _BlockBindingSnapshot]] = []
        try:
            for req, sampler_override in zip(reqs, sampler_overrides):
                saved.append(
                    (
                        req,
                        _BlockBindingSnapshot(
                            sampling_params=req.sampling_params,
                            hook_spec=req.hook_spec,
                            logit_processor=req.logit_processor,
                            hook_config=req.hook_config,
                            hook_preset_name=req.hook_preset_name,
                            requested_outputs=req.requested_outputs,
                            status=req.status,
                            stop_reason=req.stop_reason,
                        ),
                    )
                )
                if sampler_override is not None:
                    params = copy.copy(sampler_override)
                    params.ignore_eos = target_ignore_eos
                    req.sampling_params = params
                elif req.sampling_params.ignore_eos != target_ignore_eos:
                    params = copy.copy(req.sampling_params)
                    params.ignore_eos = target_ignore_eos
                    req.sampling_params = params
                if block.hook_program is not None:
                    req.hook_spec = block.hook_program.hook_spec
                    req.logit_processor = block.hook_program.logit_processor
                    req.hook_config = (
                        None
                        if block.hook_program.hook_config is None
                        else dict(block.hook_program.hook_config)
                    )
                    req.hook_preset_name = block.hook_program.hook_preset_name
                if block_outputs is not None:
                    req.set_requested_outputs(block_outputs)
                req.status = ContinuationStatus.RUNNING
                req.stop_reason = None
            yield
        finally:
            for req, snapshot in saved:
                req.sampling_params = snapshot.sampling_params
                req.hook_spec = snapshot.hook_spec
                req.logit_processor = snapshot.logit_processor
                req.hook_config = snapshot.hook_config
                req.hook_preset_name = snapshot.hook_preset_name
                req.set_requested_outputs(snapshot.requested_outputs)
                if req.status is ContinuationStatus.RUNNING:
                    req.status = snapshot.status
                if req.stop_reason is None:
                    req.stop_reason = snapshot.stop_reason

    def _activate_block_adapter(self, reqs: Sequence[Req], block: BlockSpec) -> None:
        desired_ids = set()
        if block.adapter_id is not None:
            desired_ids.add(block.adapter_id)
        else:
            desired_ids.update(req.adapter_id for req in reqs if req.adapter_id is not None)
        if len(desired_ids) > 1:
            raise RuntimeError("A block may only batch continuations with the same adapter_id")
        desired = next(iter(desired_ids), None)
        if desired == self.engine.lora_manager.active_adapter:
            return
        self._assert_capabilities(
            reqs,
            requested_outputs=block.request_outputs,
            hook_program=block.hook_program,
            forced_next_tokens=block.forced_next_tokens,
            adapter_id=desired,
        )
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        if desired is None:
            self.engine.lora_manager.unload()
        else:
            self.engine.lora_manager.load(desired)

    def _resolve_forced_tokens(
        self,
        block: BlockSpec,
        reqs: Sequence[Req],
        runnable: Sequence[Req],
        step_idx: int,
    ) -> torch.Tensor | None:
        if block.forced_next_tokens is None or step_idx > 0:
            return None
        forced_map = {
            req.continuation_id: int(block.forced_next_tokens[idx].item())
            for idx, req in enumerate(reqs)
        }
        forced = [forced_map.get(req.continuation_id, -1) for req in runnable]
        return torch.tensor(forced, dtype=torch.int32)

    def _build_block_text(self, token_ids: Sequence[int]) -> str:
        if len(token_ids) == 0:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def _prime_prefilled_continuation_for_decode(req: Req) -> None:
        # A prefill-only open must still leave one logical token to drive the
        # first decode step. Reuse the prompt's final token as the next decode
        # input, mirroring the regular prefill+sample path without consuming a
        # token during `open_continuation`.
        req.cached_len = max(0, req.device_len - 1)
        req.status = ContinuationStatus.PAUSED
        req.stop_reason = None

    def run_block(self, block: BlockSpec) -> BlockResult:
        reqs = [self.resolve_continuation(continuation_id) for continuation_id in block.continuation_ids]
        seen: Set[int] = set()
        for req in reqs:
            if req.continuation_id in seen:
                raise ValueError(f"Duplicate continuation in block: {req.continuation_id}")
            if req.state_id not in self._state_registry:
                raise RuntimeError(f"Continuation {req.continuation_id} is no longer active")
            if req.status is ContinuationStatus.FREED:
                raise RuntimeError(f"Continuation {req.continuation_id} has been freed")
            if not req.can_decode:
                raise RuntimeError(f"Continuation {req.continuation_id} cannot decode")
            seen.add(req.continuation_id)

        self._assert_capabilities(
            reqs,
            requested_outputs=block.request_outputs,
            hook_program=block.hook_program,
            forced_next_tokens=block.forced_next_tokens,
            adapter_id=block.adapter_id,
        )
        self._assert_block_cap_mask(block, reqs)
        self._activate_block_adapter(reqs, block)
        base_plan = self._compile_step_plan(
            reqs=reqs,
            phase=StepPhase.DECODE,
            requested_outputs=block.request_outputs,
            hook_program=block.hook_program,
            forced_next_tokens=block.forced_next_tokens,
            block_id=block.block_id,
        )

        emitted: Dict[int, List[int]] = {req.continuation_id: [] for req in reqs}
        wants_topk_logprobs = any(
            name == OUTPUT_TOPK_LOGPROBS or name.startswith(f"{OUTPUT_TOPK_LOGPROBS}:")
            for name in base_plan.runtime_outputs
        )
        wants_topk_ids = wants_topk_logprobs or any(
            name == OUTPUT_TOPK_IDS or name.startswith(f"{OUTPUT_TOPK_IDS}:")
            for name in base_plan.runtime_outputs
        )
        topk_ids_hist = (
            {req.continuation_id: [] for req in reqs}
            if wants_topk_ids
            else None
        )
        topk_logprobs_hist = (
            {req.continuation_id: [] for req in reqs}
            if wants_topk_logprobs
            else None
        )
        stop_reasons: Dict[int, str | None] = {req.continuation_id: None for req in reqs}
        live_text: Dict[int, str] = {req.continuation_id: "" for req in reqs} if block.stop_strings else {}

        tic = time.perf_counter()
        steps = 0
        with self._temporary_block_bindings(reqs, block):
            active = list(reqs)
            for step_idx in range(block.max_new_tokens):
                # The scheduler loop stays deliberately small: choose the active frontier,
                # compile one step, execute it, then commit the continuation state.
                runnable = [req for req in active if req.can_decode and req.status is not ContinuationStatus.ABORTED]
                if len(runnable) == 0:
                    break

                plan = self._compile_step_plan(
                    reqs=runnable,
                    phase=StepPhase.DECODE,
                    requested_outputs=block.request_outputs,
                    hook_program=block.hook_program,
                    forced_next_tokens=self._resolve_forced_tokens(block, reqs, runnable, step_idx),
                    sample_next_token=True,
                    block_id=block.block_id,
                )
                batch = Batch(reqs=list(runnable), phase="decode")
                batch.bind_plan(plan)
                with self.engine_stream_ctx:
                    forward_input = self._prepare_batch(batch)
                    forward_output = self._forward(forward_input)
                replies = self._finalize_batch(
                    batch=forward_input.batch,
                    forward_output=forward_output,
                    emit_reply=False,
                )
                steps += 1
                for batch_idx, (req, reply) in enumerate(zip(runnable, replies)):
                    emitted_token = int(reply.next_token)
                    emitted[req.continuation_id].append(emitted_token)
                    if topk_ids_hist is not None and forward_output.topk_ids_cpu is not None:
                        topk_ids_hist[req.continuation_id].append(
                            forward_output.topk_ids_cpu[batch_idx].clone()
                        )
                    if (
                        topk_logprobs_hist is not None
                        and forward_output.topk_logprobs_cpu is not None
                    ):
                        topk_logprobs_hist[req.continuation_id].append(
                            forward_output.topk_logprobs_cpu[batch_idx].clone()
                        )
                    if block.stop_strings:
                        live_text[req.continuation_id] = self._build_block_text(
                            emitted[req.continuation_id]
                        )

                    if reply.finished:
                        if emitted_token == self.eos_token_id and block.stop_on_eos:
                            req.eos_reached = True
                            req.stop_reason = "eos"
                        elif not req.can_decode:
                            req.stop_reason = "max_tokens"
                        req.status = ContinuationStatus.FINISHED
                        stop_reasons[req.continuation_id] = req.stop_reason
                        self.decode_manager.remove_req(req)
                        continue

                    if (
                        step_idx + 1 >= block.min_new_tokens
                        and block.stop_strings
                        and any(stop in live_text[req.continuation_id] for stop in block.stop_strings)
                    ):
                        req.status = ContinuationStatus.PAUSED
                        req.stop_reason = "stop_string"
                        stop_reasons[req.continuation_id] = req.stop_reason
                        self.decode_manager.remove_req(req)

                active = [
                    req
                    for req in active
                    if req.status not in {ContinuationStatus.FINISHED, ContinuationStatus.ABORTED}
                    and stop_reasons[req.continuation_id] is None
                ]
                if len(active) == 0:
                    break

        for req in reqs:
            self.decode_manager.remove_req(req)
            if req.status not in {ContinuationStatus.FINISHED, ContinuationStatus.ABORTED}:
                req.status = ContinuationStatus.PAUSED
                if stop_reasons[req.continuation_id] is None:
                    stop_reasons[req.continuation_id] = "block_limit"
                req.stop_reason = stop_reasons[req.continuation_id]

        requested_outputs = set(block.request_outputs)
        _, requested_model_outputs, _ = split_requested_outputs(block.request_outputs)
        wants_text = OUTPUT_TEXT in requested_outputs or bool(block.stop_strings)
        wants_hook_outputs = OUTPUT_HOOK_OUTPUTS in requested_outputs
        results = []
        for req in reqs:
            continuation_id = req.continuation_id
            emitted_token_ids = torch.tensor(emitted[continuation_id], dtype=torch.int32)
            hidden = None
            if (
                "last_hidden_state" in requested_model_outputs
                and "last_hidden_state" in req.latest_sample_outputs
            ):
                hidden = req.latest_sample_outputs["last_hidden_state"].clone()
            hook_outputs = copy.deepcopy(req.hook_user_state) if wants_hook_outputs else None
            topk_ids = (
                torch.stack(topk_ids_hist[continuation_id])
                if topk_ids_hist is not None and topk_ids_hist[continuation_id]
                else None
            )
            topk_logprobs = (
                torch.stack(topk_logprobs_hist[continuation_id])
                if topk_logprobs_hist is not None and topk_logprobs_hist[continuation_id]
                else None
            )
            results.append(
                ContinuationBlockResult(
                    continuation_id=continuation_id,
                    emitted_token_ids=emitted_token_ids,
                    text=(
                        live_text[continuation_id]
                        if block.stop_strings
                        else self._build_block_text(emitted[continuation_id])
                    )
                    if wants_text
                    else None,
                    final_status=req.status,
                    stop_reason=stop_reasons[continuation_id],
                    topk_ids=topk_ids,
                    topk_logprobs=topk_logprobs,
                    hidden=hidden,
                    hook_outputs=hook_outputs,
                )
            )

        toc = time.perf_counter()
        return BlockResult(
            block_id=block.block_id,
            lane=base_plan.lane,
            continuation_results=tuple(results),
            steps=steps,
            elapsed_ms=(toc - tic) * 1000.0,
        )

    def spawn_children(
        self,
        parent: Req | int,
        child_specs: Sequence[ChildContinuationSpec],
    ) -> List[Req]:
        parent_req = self.resolve_continuation(parent)
        children: List[Req] = []
        forced_groups: Dict[str | None, List[Tuple[Req, int]]] = {}
        for child_spec in child_specs:
            child = self.fork_state(parent_req, add_to_scheduler=False, snapshot=False)
            child.status = ContinuationStatus.PAUSED
            child.parent_id = parent_req.continuation_id
            child.session_id = parent_req.session_id
            child.metadata.update(copy.deepcopy(child_spec.metadata))
            if child_spec.label is not None:
                child.metadata["label"] = child_spec.label
            if child_spec.sampling_override is not None:
                child.sampling_params = copy.deepcopy(child_spec.sampling_override)
            if child_spec.adapter_id is not None:
                child.adapter_id = child_spec.adapter_id
            if child_spec.hook_program is not None:
                child.hook_spec = child_spec.hook_program.hook_spec
                child.logit_processor = child_spec.hook_program.logit_processor
                child.hook_config = copy.deepcopy(child_spec.hook_program.hook_config)
                child.hook_preset_name = child_spec.hook_program.hook_preset_name
            if child_spec.forced_first_token is not None:
                forced_groups.setdefault(child.adapter_id, []).append(
                    (child, int(child_spec.forced_first_token))
                )
            children.append(child)
        for adapter_id, grouped in forced_groups.items():
            continuation_ids = tuple(child.continuation_id for child, _ in grouped)
            forced_tokens = torch.tensor(
                [token for _, token in grouped],
                dtype=torch.int32,
            )
            self.run_block(
                BlockSpec(
                    continuation_ids=continuation_ids,
                    max_new_tokens=1,
                    request_outputs=(OUTPUT_TOKENS,),
                    forced_next_tokens=forced_tokens,
                    adapter_id=adapter_id,
                )
            )
        return children

    def free_continuation(self, continuation: Req | int) -> None:
        self.dispose_state(self.resolve_continuation(continuation))

    def fork_state(self, state: Req, *, add_to_scheduler: bool, snapshot: bool) -> Req:
        if state._runtime is not self:
            raise RuntimeError("State does not belong to this scheduler")
        if state.state_id not in self._state_registry:
            raise RuntimeError("State is no longer active")
        if self.table_manager.available_size == 0:
            raise RuntimeError("No free table slots left for fork")

        table_idx = self.table_manager.allocate()
        limit = state.device_len
        self.token_pool[table_idx, :limit].copy_(self.token_pool[state.table_idx, :limit])
        # Page table entries are allocated per-page. Copying only `:limit` can leave
        # future positions in the current page unmapped in the child row, which then
        # corrupts subsequent decode reads/writes after fork.
        page_size = self.cache_manager.page_size
        map_limit = ((limit + page_size - 1) // page_size) * page_size
        map_limit = min(map_limit, self.engine.page_table.shape[1])
        if map_limit > 0:
            self.engine.page_table[table_idx, :map_limit].copy_(
                self.engine.page_table[state.table_idx, :map_limit]
            )
        if self.engine.state_cache is not None:
            self.engine.state_cache.copy_row(state.table_idx, table_idx)

        metadata = {
            "created_at": time.time(),
            "parent_state_id": state.state_id,
            "parent_continuation_id": state.continuation_id,
            "fork_step": state.generation_step,
            "lineage": list(state.metadata.get("lineage", [])),
            "snapshot": snapshot,
            "session_id": state.session_id,
        }
        parent_input_ids = state.materialize_input_ids()
        child = Req(
            input_ids=parent_input_ids,
            table_idx=table_idx,
            cached_len=state.cached_len,
            output_len=max(0, state.max_device_len - state.device_len),
            uid=self._next_uid(),
            sampling_params=copy.deepcopy(state.sampling_params),
            cache_handle=state.cache_handle,
            hook_spec=state.hook_spec,
            hook_user_state=copy.deepcopy(state.hook_user_state),
            logit_processor=state.logit_processor,
            hook_config=copy.deepcopy(state.hook_config),
            hook_preset_name=state.hook_preset_name,
            adapter_id=state.adapter_id,
            parent_id=state.continuation_id,
            session_id=state.session_id,
            engine_cap_mask=state.engine_cap_mask,
            active_cap_mask=state.active_cap_mask,
            requested_outputs=state.requested_outputs,
            capture_output_history=state.capture_output_history,
            auto_free_on_finish=state.auto_free_on_finish,
            generation_step=state.generation_step,
            status=ContinuationStatus.PAUSED,
            eos_reached=state.eos_reached,
            stop_reason=state.stop_reason,
            last_token_id=state.last_token_id,
            sampling_state=copy.deepcopy(state.sampling_state),
            metadata=metadata,
            _input_ids_seed=list(state._input_ids_list),
            latest_sample_outputs={
                name: tensor.clone() for name, tensor in state.latest_sample_outputs.items()
            },
            sample_output_history={
                name: [tensor.clone() for tensor in history]
                for name, history in state.sample_output_history.items()
            },
        )
        child.prompt_len = state.prompt_len
        child.device_len = state.device_len
        child.max_device_len = state.max_device_len
        child._generated_token_ids = list(state.generated_token_ids)
        child.cow_active = True
        child.metadata["lineage"] = list(state.metadata.get("lineage", [])) + [child.state_id]
        child.metadata["cow_epoch"] = int(state.metadata.get("cow_epoch", 0)) + 1
        state.cow_active = True

        self.cache_manager.lock(state.cache_handle)
        self.cache_manager.track_fork_from_state(state)
        self._register_state(child)
        if add_to_scheduler and child.can_decode:
            self.decode_manager.running_reqs.add(child)
        return child

    def restore_state(self, state: Req, snapshot: Req) -> None:
        if state._runtime is not self or snapshot._runtime is not self:
            raise RuntimeError("Both state and snapshot must belong to this scheduler")
        if state.state_id not in self._state_registry:
            raise RuntimeError("State is no longer active")
        if snapshot.state_id not in self._state_registry:
            raise RuntimeError("Snapshot is no longer active")

        old_pages = self._tracked_pages(state)
        new_pages = self._tracked_pages(snapshot)
        self.cache_manager.release_page_starts_tracking(old_pages)
        self.cache_manager.free_page_starts(old_pages - new_pages)

        self.cache_manager.unlock(state.cache_handle)
        self.cache_manager.lock(snapshot.cache_handle)

        state.cache_handle = snapshot.cache_handle
        state.input_ids = snapshot.materialize_input_ids()
        state._input_ids_list = list(snapshot._input_ids_list)
        state.cached_len = snapshot.cached_len
        state.device_len = snapshot.device_len
        state.max_device_len = snapshot.max_device_len
        state.output_len = max(0, state.max_device_len - state.device_len)
        state.sampling_params = copy.deepcopy(snapshot.sampling_params)
        state.sampling_state = copy.deepcopy(snapshot.sampling_state)
        state.hook_spec = snapshot.hook_spec
        state.hook_user_state = copy.deepcopy(snapshot.hook_user_state)
        state.logit_processor = snapshot.logit_processor
        state.hook_config = copy.deepcopy(snapshot.hook_config)
        state.hook_preset_name = snapshot.hook_preset_name
        state.adapter_id = snapshot.adapter_id
        state.set_requested_outputs(snapshot.requested_outputs)
        state.capture_output_history = snapshot.capture_output_history
        state.generation_step = snapshot.generation_step
        state._generated_token_ids = list(snapshot.generated_token_ids)
        state.status = ContinuationStatus.PAUSED
        state.eos_reached = snapshot.eos_reached
        state.stop_reason = snapshot.stop_reason
        state.last_token_id = snapshot.last_token_id
        state.latest_sample_outputs = {
            name: tensor.clone() for name, tensor in snapshot.latest_sample_outputs.items()
        }
        state.sample_output_history = {
            name: [tensor.clone() for tensor in history]
            for name, history in snapshot.sample_output_history.items()
        }
        state.metadata = {
            **copy.deepcopy(snapshot.metadata),
            "restored_from": snapshot.state_id,
            "restored_at": time.time(),
        }
        state.cow_active = True

        limit = snapshot.device_len
        self.token_pool[state.table_idx, :limit].copy_(self.token_pool[snapshot.table_idx, :limit])
        # Same rationale as fork_state: keep page-table coverage aligned to the
        # active page boundary, not just exact token length.
        page_size = self.cache_manager.page_size
        map_limit = ((limit + page_size - 1) // page_size) * page_size
        map_limit = min(map_limit, self.engine.page_table.shape[1])
        if map_limit > 0:
            self.engine.page_table[state.table_idx, :map_limit].copy_(
                self.engine.page_table[snapshot.table_idx, :map_limit]
            )
        if self.engine.state_cache is not None:
            self.engine.state_cache.copy_row(snapshot.table_idx, state.table_idx)
        self.cache_manager.track_clone_page_starts(new_pages)
        if state.can_decode:
            self.decode_manager.running_reqs.add(state)
        else:
            self.decode_manager.remove_req(state)
        state.status = ContinuationStatus.PAUSED

    def dispose_state(self, state: Req) -> None:
        if state._runtime is not self:
            return
        if state.state_id not in self._state_registry:
            return
        self.decode_manager.remove_req(state)
        # Manual disposal should follow the same prefix-cache ownership rules as
        # the normal finished-request path. Otherwise we can free pages that the
        # prefix cache still owns when a live continuation is disposed manually.
        self._free_req_resources(state)
        state.status = ContinuationStatus.FREED

    def _ensure_state_position(self, state: Req, end_pos: int) -> None:
        if end_pos <= state.device_len:
            return
        start_pos = state.device_len
        self.cache_manager.allocate_for_range(state.table_idx, start_pos, end_pos)
        self.token_pool[state.table_idx, start_pos:end_pos].fill_(0)
        state._input_ids_list.extend([0] * (end_pos - start_pos))
        state.device_len = end_pos
        if state.max_device_len < end_pos:
            state.max_device_len = end_pos
        state.output_len = max(0, state.max_device_len - state.device_len)

    def _ensure_state_writable_token_index(self, state: Req, position: int) -> int:
        if position < 0 or position >= state.device_len:
            raise ValueError(
                f"Position {position} is out of bounds for state length {state.device_len}"
            )
        token_index = int(self.engine.page_table[state.table_idx, position].item())
        page_start = token_index - (token_index % self.cache_manager.page_size)
        refs = self.cache_manager.tracked_ref_count(page_start)
        if refs <= 1:
            return token_index

        new_page_start = int(self.cache_manager._allocate(1).item())
        page_size = self.cache_manager.page_size
        for layer_id in self.engine.kv_cache.iter_layer_ids():
            k_cache = self.engine.kv_cache.k_cache(layer_id)
            v_cache = self.engine.kv_cache.v_cache(layer_id)
            k_flat = k_cache.view((-1,) + tuple(k_cache.shape[2:]))
            v_flat = v_cache.view((-1,) + tuple(v_cache.shape[2:]))
            k_flat[new_page_start : new_page_start + page_size].copy_(
                k_flat[page_start : page_start + page_size]
            )
            v_flat[new_page_start : new_page_start + page_size].copy_(
                v_flat[page_start : page_start + page_size]
            )

        # Remap the full row coverage for this physical page, not only the
        # currently materialized prefix. That avoids repeating COW the first
        # time later positions in the same page become active.
        row = self.engine.page_table[state.table_idx]
        mask = (row >= page_start) & (row < page_start + page_size)
        row[mask] = row[mask] - page_start + new_page_start
        self.cache_manager.replace_tracked_page(page_start, new_page_start)
        return int(self.engine.page_table[state.table_idx, position].item())

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        self._finalize_batch(
            batch=last_data[0].batch,
            forward_output=last_data[1],
            emit_reply=True,
        )

    def _finalize_batch(
        self,
        *,
        batch: Batch,
        forward_output: ForwardOutput,
        emit_reply: bool,
    ) -> List[DetokenizeMsg]:
        next_tokens_cpu = forward_output.next_tokens_cpu
        copy_done = forward_output.copy_done_event
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        sample_outputs_cpu = forward_output.sample_outputs_cpu
        next_token_ids = next_tokens_cpu.tolist() if next_tokens_cpu is not None else None
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                if req.requested_outputs:
                    latest = {
                        name: sample_outputs_cpu[name][i].clone()
                        for name in req.model_outputs
                        if name in sample_outputs_cpu
                    }
                    req.latest_sample_outputs = latest
                    if req.capture_output_history:
                        for name, value in latest.items():
                            req.sample_output_history.setdefault(name, []).append(value.clone())
                if not batch.sample_next_token:
                    if batch.is_prefill:
                        self.cache_manager.cache_req(req, finished=False)
                    continue

                assert next_token_ids is not None
                next_token_id = int(next_token_ids[i])
                req.append_token_id(next_token_id)
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token_id == self.eos_token_id
                reply.append(
                    DetokenizeMsg(uid=req.uid, next_token=next_token_id, finished=finished)
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    req.eos_reached = next_token_id == self.eos_token_id
                    req.status = ContinuationStatus.FINISHED
                    req.stop_reason = "eos" if req.eos_reached else "max_tokens"
                    self.decode_manager.remove_req(req)
                    if req.auto_free_on_finish:
                        self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        if emit_reply:
            self.send_result(reply)
        return reply

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, AdapterBackendControlMsg):
            self._pending_adapter_control.append(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            try:
                msg.requested_outputs = self._normalize_requested_outputs(msg.requested_outputs)
            except ValueError as exc:
                logger.warning_rank0("Rejecting request %s: %s", msg.uid, exc)
                return
            if msg.hook_spec is None:
                hook_config = msg.hook_config
                if hook_config is None and msg.hook_preset_name is not None:
                    from minisgl.research import get_hook_preset_registry

                    hook_config = get_hook_preset_registry().get(msg.hook_preset_name)
                if hook_config is not None:
                    try:
                        from minisgl.research import build_hook_spec_from_config

                        msg.hook_spec = build_hook_spec_from_config(hook_config)
                    except Exception as e:
                        logger.warning_rank0(
                            "Failed to build hook spec for request %s (%s): %s",
                            msg.uid,
                            msg.hook_preset_name or "inline-config",
                            e,
                        )

            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _active_request_count(self) -> int:
        return len(self.prefill_manager.pending_list) + len(self.decode_manager.running_reqs)

    def _handle_pending_adapter_control(self) -> None:
        while self._pending_adapter_control:
            msg = self._pending_adapter_control.pop(0)
            active_requests = self._active_request_count()
            if active_requests > 0 and not msg.force and msg.action != "status":
                self.sync_all_ranks()
                if get_tp_info().is_primary():
                    self.send_control_reply(
                        AdapterResultMsg(
                            uid=msg.uid,
                            ok=False,
                            active_adapter=self.engine.lora_manager.active_adapter,
                            message="Adapter swap refused while requests are active. Retry with force=true.",
                            active_request_count=active_requests,
                        )
                    )
                continue

            torch.cuda.synchronize(self.device)
            self.sync_all_ranks()
            try:
                if msg.action == "status":
                    reply = AdapterResultMsg(
                        uid=msg.uid,
                        ok=True,
                        active_adapter=self.engine.lora_manager.active_adapter,
                        message="Adapter status fetched.",
                        active_request_count=active_requests,
                    )
                elif msg.action == "load":
                    if msg.adapter_path is None:
                        raise ValueError("adapter_path is required for load")
                    active_path = self.engine.lora_manager.load(msg.adapter_path)
                    message = "Adapter loaded."
                    if active_requests > 0:
                        message = (
                            "Adapter loaded while requests were active; subsequent decode steps will use "
                            "the new adapter."
                        )
                    reply = AdapterResultMsg(
                        uid=msg.uid,
                        ok=True,
                        active_adapter=active_path,
                        message=message,
                        active_request_count=active_requests,
                    )
                elif msg.action == "unload":
                    self.engine.lora_manager.unload()
                    message = "Adapter unloaded."
                    if active_requests > 0:
                        message = (
                            "Adapter unloaded while requests were active; subsequent decode steps will use "
                            "the base model."
                        )
                    reply = AdapterResultMsg(
                        uid=msg.uid,
                        ok=True,
                        active_adapter=None,
                        message=message,
                        active_request_count=active_requests,
                    )
                else:
                    raise ValueError(f"Unknown adapter action: {msg.action}")
            except Exception as exc:
                reply = AdapterResultMsg(
                    uid=msg.uid,
                    ok=False,
                    active_adapter=self.engine.lora_manager.active_adapter,
                    message=str(exc),
                    active_request_count=active_requests,
                )
            self.sync_all_ranks()
            if get_tp_info().is_primary():
                self.send_control_reply(reply)

    def _free_req_resources(self, req: Req) -> None:
        self.cache_manager.release_state_tracking(req)
        self._unregister_state(req)
        if self.engine.state_cache is not None:
            self.engine.state_cache.reset_row(req.table_idx)
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        for req in batch.reqs:
            self._register_state(req)
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        for req in batch.reqs:
            if not req.cow_active:
                continue
            if req.extend_len <= 0:
                continue
            for pos in range(req.cached_len, req.device_len):
                self._ensure_state_writable_token_index(req, pos)
            # Freshly allocated pages are unique, so once the current writable span
            # has been remapped we do not need to keep checking COW on later decode steps.
            req.cow_active = False
        if self.engine.state_cache is not None and hasattr(self.engine.state_cache, "begin_batch_tracking"):
            self.engine.state_cache.begin_batch_tracking(batch)
        batch.positions, position_indices = _make_positions(
            batch,
            positions_buffer=self._positions_buffer,
            index_buffer=self._positions_index_buffer,
        )
        batch.req_table_indices, batch.req_table_indices_i32, batch.req_cu_seqlens = _make_req_layout(
            batch,
            table_buffer=self._req_table_buffer,
            table_i32_buffer=self._req_table_i32_buffer,
            cu_seqlens_buffer=self._req_cu_seqlens_buffer,
        )
        input_mapping = _make_input_tuple(
            batch,
            positions=position_indices,
            mapping_buffer=self._input_mapping_buffer,
        )
        write_mapping = _make_write_tuple(
            batch,
            mapping_buffer=self._write_mapping_buffer,
            position_buffer=self._write_index_buffer,
        )
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: https://github.com/sgl-project/mini-sglang/issues/58
            self.stream.synchronize()
        forward_output = self.engine.forward_batch(batch, sample_args)
        if forward_output.next_tokens_gpu is not None:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(
    batch: Batch,
    *,
    positions_buffer: PinnedRingBuffer,
    index_buffer: PinnedRingBuffer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    positions_host, positions = positions_buffer.acquire(needed_size)
    index_host, position_indices = index_buffer.acquire(needed_size)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=positions_host[offset : offset + length],
        )
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int64,
            out=index_host[offset : offset + length],
        )
        offset += length
    if needed_size > 0:
        positions.copy_(positions_host, non_blocking=True)
        position_indices.copy_(index_host, non_blocking=True)
    return positions, position_indices


def _make_input_tuple(
    batch: Batch,
    *,
    positions: torch.Tensor,
    mapping_buffer: PinnedRingBuffer,
) -> Indice2D:
    needed_size = positions.numel()
    mapping_host, mapping = mapping_buffer.acquire(needed_size)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    if needed_size > 0:
        mapping.copy_(mapping_host, non_blocking=True)
    return mapping, positions


def _make_req_layout(
    batch: Batch,
    *,
    table_buffer: PinnedRingBuffer,
    table_i32_buffer: PinnedRingBuffer,
    cu_seqlens_buffer: PinnedRingBuffer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_reqs = len(batch.reqs)
    table_host, table_indices = table_buffer.acquire(num_reqs)
    table_i32_host, table_indices_i32 = table_i32_buffer.acquire(num_reqs)
    cu_host, cu_seqlens = cu_seqlens_buffer.acquire(num_reqs + 1)
    cu_host[0] = 0
    offset = 0
    for idx, req in enumerate(batch.reqs):
        table_host[idx] = req.table_idx
        table_i32_host[idx] = req.table_idx
        offset += req.extend_len
        cu_host[idx + 1] = offset
    if num_reqs > 0:
        table_indices.copy_(table_host, non_blocking=True)
        table_indices_i32.copy_(table_i32_host, non_blocking=True)
    cu_seqlens.copy_(cu_host, non_blocking=True)
    return table_indices, table_indices_i32, cu_seqlens


def _make_write_tuple(
    batch: Batch,
    *,
    mapping_buffer: PinnedRingBuffer,
    position_buffer: PinnedRingBuffer,
) -> Indice2D:
    needed_size = batch.size
    mapping_host, mapping = mapping_buffer.acquire(needed_size)
    position_host, positions = position_buffer.acquire(needed_size)
    for idx, req in enumerate(batch.reqs):
        mapping_host[idx] = req.table_idx
        position_host[idx] = req.device_len if req.can_decode else -1
    if needed_size > 0:
        mapping.copy_(mapping_host, non_blocking=True)
        positions.copy_(position_host, non_blocking=True)
    return mapping, positions
