from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np

from minisgl.core import (
    OUTPUT_LOGPROBS,
    OUTPUT_TOKENS,
    BlockSpec,
    ChildContinuationSpec,
    ContinuationStatus,
    SamplingParams,
)

from .config import BranchGRPOConfig
from .records import Edge, Leaf, Node, RolloutTree
from .verifier import binary_tag_reward


@dataclass(frozen=True)
class ActivePath:
    continuation: Any
    parent_node: int
    path_edges: tuple[int, ...]
    generated_len: int
    next_branch_index: int


def rollout_sampling_params(config: BranchGRPOConfig) -> SamplingParams:
    return SamplingParams(
        temperature=config.temperature,
        top_k=-1,
        top_p=config.top_p,
        ignore_eos=False,
        max_tokens=config.max_generation_tokens,
    )


def _nominal_slots_below(config: BranchGRPOConfig, next_branch_index: int) -> int:
    remaining = len(config.branch_targets) - int(next_branch_index)
    if remaining < 0:
        raise ValueError("next_branch_index exceeds configured branch target count")
    return int(config.branch_factor**remaining)


def _decode_completion(llm: Any, token_ids: Iterable[int]) -> str:
    return llm.tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _ground_truth(row: dict[str, Any]) -> Any:
    if "answer" in row:
        return row["answer"]
    if "normalized_answer" in row:
        return row["normalized_answer"]
    raise KeyError("rollout row must include answer or normalized_answer")


def build_branch_rollout_tree(
    *,
    llm: Any,
    prompt_row: dict[str, Any],
    prompt_id: int,
    config: BranchGRPOConfig,
    sampling_params: SamplingParams | None = None,
) -> RolloutTree:
    prompt = str(prompt_row["prompt"])
    params = sampling_params or rollout_sampling_params(config)
    root = llm.open_continuation(prompt, params)
    continuations: List[Any] = [root]
    try:
        prompt_token_ids = np.asarray(
            root.materialize_input_ids()[: root.prompt_len].tolist(),
            dtype=np.int32,
        )
        if prompt_token_ids.shape[0] > config.prompt_max_tokens:
            raise ValueError(
                f"prompt {prompt_id} has {prompt_token_ids.shape[0]} tokens, "
                f"limit is {config.prompt_max_tokens}"
            )

        tree = RolloutTree(
            prompt_id=prompt_id,
            prompt_token_ids=prompt_token_ids,
            root_node=0,
        )
        tree.nodes[0] = Node(
            id=0,
            prompt_id=prompt_id,
            parent_edge=None,
            depth=0,
            generated_len=0,
        )

        root_children = root.spawn_children(
            [
                ChildContinuationSpec(
                    metadata={"root_sample_index": root_idx},
                    label=f"root-{root_idx}",
                )
                for root_idx in range(config.root_samples)
            ]
        )
        continuations.extend(root_children)
        frontier = [
            ActivePath(
                continuation=child,
                parent_node=tree.root_node,
                path_edges=(),
                generated_len=0,
                next_branch_index=0,
            )
            for child in root_children
        ]

        edge_id = 0
        node_id = 1
        leaf_id = 0
        # Segment-relative stages keep frontier continuations synchronized (all start each
        # stage at block-relative position 0), so one batched run_block per stage works even
        # though confidence-gated forks land at staggered absolute positions.
        prev_target = 0
        seg_lens = []
        for t in config.branch_targets:
            seg_lens.append(int(t) - prev_target)
            prev_target = int(t)
        stages = [(seg, False) for seg in seg_lens] + [(None, True)]
        lookahead = int(config.boundary_lookahead)
        threshold = float(config.confidence_threshold)
        for seg_len, is_final in stages:
            if not frontier:
                break
            if is_final:
                # Generate every leaf to EOS / the absolute generation budget. Cap by the most
                # advanced continuation so none exceeds max_generation_tokens; no branching here.
                max_new_tokens = max(
                    1, config.max_generation_tokens - max(a.generated_len for a in frontier)
                )
                block = BlockSpec(
                    continuation_ids=tuple(a.continuation.continuation_id for a in frontier),
                    max_new_tokens=max_new_tokens,
                    stop_on_eos=True,
                    request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
                )
            else:
                # Generate the nominal segment, then fork at the first low-confidence token
                # (top-1 prob <= threshold) within `lookahead`, else force a fork at the cap.
                block = BlockSpec(
                    continuation_ids=tuple(a.continuation.continuation_id for a in frontier),
                    max_new_tokens=seg_len + lookahead,
                    min_new_tokens=seg_len,
                    stop_on_eos=True,
                    branch_confidence_threshold=threshold,
                    request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
                )
            result = llm.run_block(block)
            by_continuation = {
                item.continuation_id: item for item in result.continuation_results
            }
            next_frontier: list[ActivePath] = []
            for active in frontier:
                continuation = active.continuation
                continuation_result = by_continuation[continuation.continuation_id]
                if continuation_result.logprobs is None:
                    raise RuntimeError("xsglang did not return selected-token logprobs")

                tokens = np.asarray(
                    continuation_result.emitted_token_ids.tolist(),
                    dtype=np.int32,
                )
                old_logprobs = np.asarray(
                    continuation_result.logprobs.tolist(),
                    dtype=np.float32,
                )
                if tokens.shape != old_logprobs.shape:
                    raise ValueError("emitted tokens and logprobs are not aligned")

                end_gen_pos = active.generated_len + int(tokens.shape[0])
                slots_below = _nominal_slots_below(config, active.next_branch_index)
                current_edge_id = edge_id
                current_node_id = node_id
                edge_id += 1
                node_id += 1

                finish_reason = continuation_result.stop_reason or "unknown"
                hit_eos = (
                    continuation_result.final_status is ContinuationStatus.FINISHED
                    and finish_reason == "eos"
                )
                # Leaf iff this is the final stage or the trace ended (EOS). A confidence
                # branch-boundary / forced (block_limit / max_tokens) stop in a branch stage is
                # non-terminal -> it forks.
                terminal = is_final or hit_eos
                tree.nodes[current_node_id] = Node(
                    id=current_node_id,
                    prompt_id=prompt_id,
                    parent_edge=current_edge_id,
                    depth=len(active.path_edges) + 1,
                    generated_len=end_gen_pos,
                    terminal=terminal,
                    terminal_leaf_id=leaf_id if terminal else None,
                )
                tree.edges[current_edge_id] = Edge(
                    id=current_edge_id,
                    prompt_id=prompt_id,
                    parent_node=active.parent_node,
                    child_node=current_node_id,
                    depth=len(active.path_edges),
                    tokens=tokens,
                    old_logprobs=old_logprobs,
                    nominal_weight=slots_below,
                    start_gen_pos=active.generated_len,
                    end_gen_pos=end_gen_pos,
                    finish_reason=finish_reason,
                )
                tree.nodes[active.parent_node].children.append(current_edge_id)
                path_edges = active.path_edges + (current_edge_id,)

                if terminal:
                    completion = _decode_completion(llm, continuation.generated_token_ids)
                    tree.leaves[leaf_id] = Leaf(
                        id=leaf_id,
                        prompt_id=prompt_id,
                        node_id=current_node_id,
                        path_edges=list(path_edges),
                        nominal_slot_count=slots_below,
                        answer_text=completion,
                        reward=binary_tag_reward(completion, _ground_truth(prompt_row)),
                        finish_reason=finish_reason,
                    )
                    leaf_id += 1
                    continue

                children = continuation.spawn_children(
                    [
                        ChildContinuationSpec(
                            metadata={
                                "branch_parent_node": current_node_id,
                                "branch_index": active.next_branch_index,
                                "child_index": child_idx,
                            },
                            label=f"b{active.next_branch_index}-{child_idx}",
                        )
                        for child_idx in range(config.branch_factor)
                    ]
                )
                continuations.extend(children)
                for child in children:
                    next_frontier.append(
                        ActivePath(
                            continuation=child,
                            parent_node=current_node_id,
                            path_edges=path_edges,
                            generated_len=end_gen_pos,
                            next_branch_index=active.next_branch_index + 1,
                        )
                    )
            frontier = next_frontier

        if not tree.leaves:
            raise RuntimeError(f"rollout for prompt {prompt_id} produced no leaves")
        return tree
    finally:
        for continuation in reversed(continuations):
            try:
                llm.free_continuation(continuation)
            except Exception:
                pass


def build_branch_rollout_trees(
    *,
    llm: Any,
    prompt_rows: Iterable[dict[str, Any]],
    config: BranchGRPOConfig,
    start_prompt_id: int = 0,
) -> Iterable[RolloutTree]:
    for offset, row in enumerate(prompt_rows):
        yield build_branch_rollout_tree(
            llm=llm,
            prompt_row=row,
            prompt_id=start_prompt_id + offset,
            config=config,
        )
