from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os
import tempfile
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from minisgl.core import (
    BlockSpec,
    ContinuationCapability,
    HookProgram,
    OUTPUT_HOOK_OUTPUTS,
    OUTPUT_LOGPROBS,
    OUTPUT_TEXT,
    OUTPUT_TOKENS,
    Req,
    SamplingParams,
    normalize_output_names,
)
from minisgl.distributed import DistributedInfo
from minisgl.hooks import ActivationCaptureResult, HookSpec, LogitProcessor, normalize_layer_hooks
from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig

from minisgl.research import (
    build_hidden_state_trace_program,
    build_hook_spec_from_config,
    capture_hook,
    compose_hook_programs,
)


class RequestAllFinished(Exception):
    pass


@dataclass
class OfflinePendingRequest:
    prompt: List[int] | str
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    hook_spec: HookSpec | None = None
    logit_processor: LogitProcessor | None = None
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    adapter_id: str | None = None
    requested_outputs: tuple[str, ...] = ()
    capture_output_history: bool = False


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]
    hook_user_state: Dict[str, Any] = field(default_factory=dict)
    latest_sample_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_output_history: Dict[str, List[torch.Tensor]] = field(default_factory=dict)


def _allocate_local_distributed_addr() -> str:
    path = os.path.join(
        tempfile.gettempdir(),
        f"minisgl-dist-{os.getpid()}-{time.time_ns()}",
    )
    return f"file://{path}"


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        kwargs.setdefault("distributed_init_addr", _allocate_local_distributed_addr())
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[OfflinePendingRequest] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    @property
    def num_layers(self) -> int:
        model = self.engine.model
        if hasattr(model, "num_layers"):
            return int(model.num_layers)
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Unable to infer model layer count")
        return len(layers.op_list)

    @property
    def available_outputs(self) -> tuple[str, ...]:
        return (
            self.engine.model.available_sample_output_names()
            + (
                OUTPUT_TOKENS,
                OUTPUT_TEXT,
                OUTPUT_LOGPROBS,
                "topk_ids",
                "topk_logprobs",
                OUTPUT_HOOK_OUTPUTS,
            )
        )

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def open_continuation(
        self,
        prompt: List[int] | str,
        sampling_params: SamplingParams,
        *,
        hook_spec: HookSpec | None = None,
        logit_processor: LogitProcessor | None = None,
        hook_config: Dict[str, Any] | None = None,
        hook_preset_name: str | None = None,
        adapter_id: str | None = None,
        requested_outputs: Sequence[str] | str | None = None,
        capture_output_history: bool = False,
        session_id: int | None = None,
        active_cap_mask: int | ContinuationCapability | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Req:
        if hook_spec is None and hook_config is None and hook_preset_name is not None:
            from minisgl.research import get_hook_preset_registry

            hook_config = get_hook_preset_registry().get(hook_preset_name)
        if hook_spec is None and hook_config is not None:
            hook_spec = build_hook_spec_from_config(hook_config)
        input_ids = self._tokenize_one(prompt)
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
            session_id=session_id,
            active_cap_mask=None if active_cap_mask is None else int(active_cap_mask),
            metadata=metadata,
        )

    def make_hidden_state_trace_program(
        self,
        layers: Sequence[int] | str,
        *,
        capture: str = "last_token",
        to_cpu: bool = True,
        state_key: str | None = None,
        label: str | None = None,
    ):
        resolved_layers = "all" if layers == "all" else tuple(int(layer) for layer in layers)
        return build_hidden_state_trace_program(
            resolved_layers,
            num_layers=self.num_layers,
            capture=capture,
            to_cpu=to_cpu,
            state_key=state_key,
            label=label,
        )

    def trace_hidden_states(
        self,
        continuation: Req | int,
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
        req = self.resolve_continuation(continuation)
        trace_program, trace = self.make_hidden_state_trace_program(
            layers,
            capture=capture,
            to_cpu=to_cpu,
            label="hidden-state-trace",
        )
        resident_program = None
        if (
            req.hook_spec is not None
            or req.logit_processor is not None
            or req.hook_config is not None
            or req.hook_preset_name is not None
        ):
            resident_program = HookProgram(
                hook_spec=req.hook_spec,
                logit_processor=req.logit_processor,
                hook_config=req.hook_config,
                hook_preset_name=req.hook_preset_name,
                label="resident-hooks",
            )
        result = self.run_block(
            BlockSpec(
                continuation_ids=(req.continuation_id,),
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                stop_on_eos=stop_on_eos,
                stop_strings=tuple(stop_strings or ()),
                request_outputs=normalize_output_names(request_outputs),
                hook_program=compose_hook_programs(
                    resident_program,
                    hook_program,
                    trace_program,
                    label="resident-hooks + trace" if resident_program is not None else None,
                ),
                sampler_override=sampler_override,
                adapter_id=adapter_id,
            )
        )
        return result, trace

    def create_state(
        self,
        prompt: List[int] | str,
        sampling_params: SamplingParams,
        *,
        hook_spec: HookSpec | None = None,
        logit_processor: LogitProcessor | None = None,
        hook_config: Dict[str, Any] | None = None,
        hook_preset_name: str | None = None,
        adapter_id: str | None = None,
        requested_outputs: Sequence[str] | str | None = None,
        capture_output_history: bool = False,
    ) -> Req:
        return self.open_continuation(
            prompt,
            sampling_params,
            hook_spec=hook_spec,
            logit_processor=logit_processor,
            hook_config=hook_config,
            hook_preset_name=hook_preset_name,
            adapter_id=adapter_id,
            requested_outputs=requested_outputs,
            capture_output_history=capture_output_history,
        )

    def step_state(self, state: Req) -> int:
        return super().step_state(state)

    def step_states(self, states: Sequence[Req]) -> List[int]:
        return super().step_states(states)

    @staticmethod
    def _expand_per_request(value: Any, n: int, *, field_name: str) -> List[Any]:
        if value is None:
            return [None] * n
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            if len(value) != n:
                raise ValueError(f"{field_name} length {len(value)} does not match prompts length {n}")
            return list(value)
        return [value] * n

    @staticmethod
    def _expand_capture_layers(value: Any, n: int) -> List[Any]:
        if value is None:
            return [None] * n
        if isinstance(value, list):
            if len(value) == 0:
                return [[] for _ in range(n)]
            if all(isinstance(x, int) for x in value):
                return [list(value) for _ in range(n)]
            if len(value) != n:
                raise ValueError(
                    f"capture_layers length {len(value)} does not match prompts length {n}"
                )
            return list(value)
        return [value] * n

    @staticmethod
    def _expand_requested_outputs(value: Any, n: int) -> List[Any]:
        if value is None:
            return [None] * n
        if isinstance(value, (str, bytes)):
            return [value] * n
        if isinstance(value, Sequence):
            seq = list(value)
            if len(seq) == 0:
                return [()] * n
            if all(isinstance(item, str) for item in seq):
                return [tuple(seq)] * n
            if len(seq) != n:
                raise ValueError(
                    f"requested_outputs length {len(seq)} does not match prompts length {n}"
                )
            return seq
        return [value] * n

    def _build_hook_specs(
        self,
        num_prompts: int,
        hooks: Any,
        capture_layers: Any,
        hook_config: Any,
    ) -> Tuple[List[HookSpec | None], ActivationCaptureResult | None, List[Dict[str, Any] | None]]:
        hook_inputs = self._expand_per_request(hooks, num_prompts, field_name="hooks")
        capture_inputs = self._expand_capture_layers(capture_layers, num_prompts)
        config_inputs = self._expand_per_request(hook_config, num_prompts, field_name="hook_config")

        use_capture = any(v is not None for v in capture_inputs)
        capture_result = ActivationCaptureResult(to_cpu=False) if use_capture else None

        specs: List[HookSpec | None] = []
        for i in range(num_prompts):
            cfg = config_inputs[i]
            base_spec = build_hook_spec_from_config(cfg) if cfg is not None else None

            layer_hook_lists: Dict[int, List] = defaultdict(list)
            intra_hook_lists: Dict[int, List] = defaultdict(list)
            post_embedding_hooks = []
            pre_lm_head_hooks = []
            has_writes = False
            tier = 1

            if base_spec is not None:
                for layer_idx, callbacks in base_spec.layer_hooks.items():
                    layer_hook_lists[layer_idx].extend(callbacks)
                for layer_idx, callbacks in base_spec.intra_layer_hooks.items():
                    intra_hook_lists[layer_idx].extend(callbacks)
                post_embedding_hooks.extend(base_spec.post_embedding_hooks)
                pre_lm_head_hooks.extend(base_spec.pre_lm_head_hooks)
                has_writes = base_spec.has_writes
                tier = base_spec.tier

            manual_hooks = hook_inputs[i]
            if manual_hooks is not None:
                normalized = normalize_layer_hooks(manual_hooks)
                for layer_idx, callbacks in normalized.items():
                    layer_hook_lists[layer_idx].extend(callbacks)

            cap_layers = capture_inputs[i]
            if cap_layers is not None:
                if capture_result is None:
                    capture_result = ActivationCaptureResult(to_cpu=False)
                cap_cb = capture_hook(capture_result)
                for layer_idx in cap_layers:
                    layer_hook_lists[int(layer_idx)].append(cap_cb)

            if len(layer_hook_lists) == 0 and len(intra_hook_lists) == 0 and not post_embedding_hooks and not pre_lm_head_hooks:
                specs.append(None)
                continue

            specs.append(
                HookSpec(
                    layer_hooks={k: tuple(v) for k, v in layer_hook_lists.items()},
                    intra_layer_hooks={k: tuple(v) for k, v in intra_hook_lists.items()},
                    post_embedding_hooks=tuple(post_embedding_hooks),
                    pre_lm_head_hooks=tuple(pre_lm_head_hooks),
                    has_writes=has_writes,
                    tier=tier,
                )
            )

        return specs, capture_result, config_inputs

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for pending_req in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = pending_req.input_ids
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(
                UserMsg(
                    uid=uid,
                    input_ids=input_ids,
                    sampling_params=pending_req.sampling_params,
                    hook_spec=pending_req.hook_spec,
                    logit_processor=pending_req.logit_processor,
                    hook_config=pending_req.hook_config,
                    hook_preset_name=pending_req.hook_preset_name,
                    adapter_id=pending_req.adapter_id,
                    requested_outputs=pending_req.requested_outputs,
                    capture_output_history=pending_req.capture_output_history,
                )
            )
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=input_ids.tolist(),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        finished_state = {req.uid: dict(req.hook_user_state) for req in self.finished_reqs}
        tracked_reqs = {req.uid: req for req in self.decode_manager.running_reqs}
        tracked_reqs.update({req.uid: req for req in self.finished_reqs})
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)
            req = tracked_reqs.get(msg.uid)
            if req is not None and req.latest_sample_outputs:
                latest = {
                    name: tensor.clone() for name, tensor in req.latest_sample_outputs.items()
                }
                status.latest_sample_outputs = latest
                if req.capture_output_history:
                    for name, tensor in latest.items():
                        status.sample_output_history.setdefault(name, []).append(tensor.clone())
            if msg.finished and msg.uid in finished_state:
                status.hook_user_state = finished_state[msg.uid]

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        hooks: Mapping[int, Any] | List[Mapping[int, Any] | None] | None = None,
        capture_layers: List[int] | List[List[int] | None] | None = None,
        logit_processor: LogitProcessor | List[LogitProcessor | None] | None = None,
        hook_config: Dict[str, Any] | List[Dict[str, Any] | None] | None = None,
        requested_outputs: Sequence[str] | str | Sequence[Sequence[str] | str | None] | None = None,
        capture_output_history: bool | Sequence[bool] | None = None,
    ) -> List[Dict[str, str | List[int]]] | Dict[str, Any]:
        n = len(prompts)
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * n
        if len(sampling_params) != n:
            raise ValueError("sampling_params length must match prompts length")

        hook_specs, capture_result, hook_configs = self._build_hook_specs(
            num_prompts=n,
            hooks=hooks,
            capture_layers=capture_layers,
            hook_config=hook_config,
        )
        logit_processors = self._expand_per_request(
            logit_processor,
            n,
            field_name="logit_processor",
        )
        output_requests = self._expand_requested_outputs(requested_outputs, n)
        if capture_output_history is None:
            history_requests = [bool(v) for v in output_requests]
        else:
            history_requests = [bool(v) for v in self._expand_per_request(
                capture_output_history,
                n,
                field_name="capture_output_history",
            )]
        continuations: List[Req] = []
        try:
            for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
                continuations.append(
                    self.open_continuation(
                        prompt,
                        sp,
                        hook_spec=hook_specs[i],
                        logit_processor=logit_processors[i],
                        hook_config=hook_configs[i],
                        requested_outputs=output_requests[i],
                        capture_output_history=history_requests[i],
                    )
                )

            block_result = self.run_block(
                BlockSpec(
                    continuation_ids=tuple(req.continuation_id for req in continuations),
                    max_new_tokens=max(sp.max_tokens for sp in sampling_params),
                    request_outputs=(OUTPUT_TEXT,),
                )
            )
            result_by_id = {
                item.continuation_id: item for item in block_result.continuation_results
            }
            outputs: List[Dict[str, str | List[int]]] = []
            hook_state: Dict[int, Dict[str, Any]] = {}
            sample_outputs: Dict[int, Dict[str, List[torch.Tensor]]] = {}
            latest_sample_outputs: Dict[int, Dict[str, torch.Tensor]] = {}
            for i, continuation in enumerate(continuations):
                result = result_by_id[continuation.continuation_id]
                text = result.text
                if text is None:
                    text = self.tokenizer.decode(result.emitted_token_ids.tolist())
                outputs.append(
                    {
                        "text": text,
                        "token_ids": result.emitted_token_ids.tolist(),
                    }
                )
                hook_state[i] = dict(continuation.hook_user_state)
                sample_outputs[i] = {
                    name: [tensor.clone() for tensor in history]
                    for name, history in continuation.sample_output_history.items()
                }
                latest_sample_outputs[i] = {
                    name: tensor.clone()
                    for name, tensor in continuation.latest_sample_outputs.items()
                }

            using_research_features = (
                any(spec is not None for spec in hook_specs)
                or any(lp is not None for lp in logit_processors)
                or any(cfg is not None for cfg in hook_configs)
                or capture_result is not None
                or any(bool(v) for v in sample_outputs.values())
                or any(bool(v) for v in latest_sample_outputs.values())
            )
            if not using_research_features:
                return outputs

            captures = capture_result.materialize() if capture_result is not None else {}
            return {
                "outputs": outputs,
                "captures": captures,
                "capture_handle": capture_result,
                "hook_user_state": hook_state,
                "sample_outputs": sample_outputs,
                "latest_sample_outputs": latest_sample_outputs,
            }
        finally:
            for continuation in continuations:
                self.dispose_state(continuation)
