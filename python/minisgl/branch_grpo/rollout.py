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


def _stages(config: BranchGRPOConfig):
    """Segment lengths for each branch stage + a final leaf stage (None)."""
    prev = 0
    segs = []
    for t in config.branch_targets:
        segs.append(int(t) - prev)
        prev = int(t)
    return [(s, False) for s in segs] + [(None, True)]


def _open_prompt_tree(llm, prompt_row, prompt_id, config, params):
    """Open a prompt, spawn root_samples children; returns (tree, continuations, root ActivePaths)."""
    prompt = str(prompt_row["prompt"])
    root = llm.open_continuation(prompt, params)
    prompt_token_ids = np.asarray(
        root.materialize_input_ids()[: root.prompt_len].tolist(), dtype=np.int32
    )
    if prompt_token_ids.shape[0] > config.prompt_max_tokens:
        try:
            llm.free_continuation(root)
        except Exception:
            pass
        raise ValueError(
            f"prompt {prompt_id} has {prompt_token_ids.shape[0]} tokens, "
            f"limit is {config.prompt_max_tokens}"
        )
    tree = RolloutTree(prompt_id=prompt_id, prompt_token_ids=prompt_token_ids, root_node=0)
    tree.nodes[0] = Node(id=0, prompt_id=prompt_id, parent_edge=None, depth=0, generated_len=0)
    roots = root.spawn_children(
        [
            ChildContinuationSpec(metadata={"root_sample_index": i}, label=f"root-{i}")
            for i in range(config.root_samples)
        ]
    )
    actives = [
        ActivePath(continuation=c, parent_node=0, path_edges=(), generated_len=0, next_branch_index=0)
        for c in roots
    ]
    return tree, [root, *roots], actives


def _record_continuation(llm, tree, ctr, active, cr, prompt_row, config, is_final):
    """Record one edge/node, make a leaf or fork; returns [(child_continuation, ActivePath), ...]."""
    if cr.logprobs is None:
        raise RuntimeError("xsglang did not return selected-token logprobs")
    tokens = np.asarray(cr.emitted_token_ids.tolist(), dtype=np.int32)
    old_logprobs = np.asarray(cr.logprobs.tolist(), dtype=np.float32)
    if tokens.shape != old_logprobs.shape:
        raise ValueError("emitted tokens and logprobs are not aligned")
    end_gen_pos = active.generated_len + int(tokens.shape[0])
    slots_below = _nominal_slots_below(config, active.next_branch_index)
    eid = ctr["edge_id"]
    nid = ctr["node_id"]
    ctr["edge_id"] += 1
    ctr["node_id"] += 1
    finish_reason = cr.stop_reason or "unknown"
    hit_eos = cr.final_status is ContinuationStatus.FINISHED and finish_reason == "eos"
    # Leaf iff final stage or the trace ended (EOS). A branch-boundary / forced stop in a
    # branch stage is non-terminal -> it forks.
    terminal = is_final or hit_eos
    tree.nodes[nid] = Node(
        id=nid,
        prompt_id=tree.prompt_id,
        parent_edge=eid,
        depth=len(active.path_edges) + 1,
        generated_len=end_gen_pos,
        terminal=terminal,
        terminal_leaf_id=ctr["leaf_id"] if terminal else None,
    )
    tree.edges[eid] = Edge(
        id=eid,
        prompt_id=tree.prompt_id,
        parent_node=active.parent_node,
        child_node=nid,
        depth=len(active.path_edges),
        tokens=tokens,
        old_logprobs=old_logprobs,
        nominal_weight=slots_below,
        start_gen_pos=active.generated_len,
        end_gen_pos=end_gen_pos,
        finish_reason=finish_reason,
    )
    tree.nodes[active.parent_node].children.append(eid)
    path_edges = active.path_edges + (eid,)
    if terminal:
        completion = _decode_completion(llm, active.continuation.generated_token_ids)
        tree.leaves[ctr["leaf_id"]] = Leaf(
            id=ctr["leaf_id"],
            prompt_id=tree.prompt_id,
            node_id=nid,
            path_edges=list(path_edges),
            nominal_slot_count=slots_below,
            answer_text=completion,
            reward=binary_tag_reward(completion, _ground_truth(prompt_row)),
            finish_reason=finish_reason,
        )
        ctr["leaf_id"] += 1
        return []
    children = active.continuation.spawn_children(
        [
            ChildContinuationSpec(
                metadata={
                    "branch_parent_node": nid,
                    "branch_index": active.next_branch_index,
                    "child_index": ci,
                },
                label=f"b{active.next_branch_index}-{ci}",
            )
            for ci in range(config.branch_factor)
        ]
    )
    return [
        (
            child,
            ActivePath(
                continuation=child,
                parent_node=nid,
                path_edges=path_edges,
                generated_len=end_gen_pos,
                next_branch_index=active.next_branch_index + 1,
            ),
        )
        for child in children
    ]


def build_branch_rollout_wave(
    *,
    llm: Any,
    prompt_rows: Iterable[dict[str, Any]],
    config: BranchGRPOConfig,
    start_prompt_id: int = 0,
    sampling_params: SamplingParams | None = None,
) -> List[RolloutTree]:
    """Build several prompt trees jointly, batching one run_block over the whole wave per stage.

    Segment-relative stages keep every continuation across all trees synchronized at each stage,
    so a single run_block decodes the combined frontier (level-major across the wave, plan sec 3)
    -> far higher decode concurrency than prompt-major. Over-long prompts are skipped at open.
    """
    params = sampling_params or rollout_sampling_params(config)
    rows = list(prompt_rows)
    all_conts: List[Any] = []
    trees: dict[int, RolloutTree] = {}
    ctrs: dict[int, dict] = {}
    frontier: list[tuple[int, ActivePath]] = []
    try:
        for ti, row in enumerate(rows):
            try:
                tree, conts, actives = _open_prompt_tree(llm, row, start_prompt_id + ti, config, params)
            except Exception as exc:  # over-long prompt etc. -> skip this prompt, keep the wave
                print(f"[warn] skip prompt {start_prompt_id + ti}: {exc}", flush=True)
                continue
            trees[ti] = tree
            ctrs[ti] = {"node_id": 1, "edge_id": 0, "leaf_id": 0}
            all_conts.extend(conts)
            frontier.extend((ti, a) for a in actives)

        for seg_len, is_final in _stages(config):
            if not frontier:
                break
            cont_ids = tuple(a.continuation.continuation_id for (_, a) in frontier)
            if is_final:
                max_new = max(1, config.max_generation_tokens - max(a.generated_len for (_, a) in frontier))
                block = BlockSpec(
                    continuation_ids=cont_ids, max_new_tokens=max_new, stop_on_eos=True,
                    request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
                )
            else:
                block = BlockSpec(
                    continuation_ids=cont_ids, max_new_tokens=seg_len + int(config.boundary_lookahead),
                    min_new_tokens=seg_len, stop_on_eos=True,
                    branch_confidence_threshold=float(config.confidence_threshold),
                    request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
                )
            result = llm.run_block(block)
            by = {item.continuation_id: item for item in result.continuation_results}
            nxt: list[tuple[int, ActivePath]] = []
            for ti, active in frontier:
                cr = by[active.continuation.continuation_id]
                for child_cont, child_active in _record_continuation(
                    llm, trees[ti], ctrs[ti], active, cr, rows[ti], config, is_final
                ):
                    all_conts.append(child_cont)
                    nxt.append((ti, child_active))
            frontier = nxt

        out: List[RolloutTree] = []
        for ti in sorted(trees):
            if not trees[ti].leaves:
                raise RuntimeError(f"rollout for prompt {start_prompt_id + ti} produced no leaves")
            out.append(trees[ti])
        return out
    finally:
        for continuation in reversed(all_conts):
            try:
                llm.free_continuation(continuation)
            except Exception:
                pass


def build_branch_rollout_tree(
    *,
    llm: Any,
    prompt_row: dict[str, Any],
    prompt_id: int,
    config: BranchGRPOConfig,
    sampling_params: SamplingParams | None = None,
) -> RolloutTree:
    trees = build_branch_rollout_wave(
        llm=llm, prompt_rows=[prompt_row], config=config,
        start_prompt_id=prompt_id, sampling_params=sampling_params,
    )
    if not trees:
        raise ValueError(f"prompt {prompt_id} could not be rolled out (skipped)")
    return trees[0]


def build_branch_rollout_trees(
    *,
    llm: Any,
    prompt_rows: Iterable[dict[str, Any]],
    config: BranchGRPOConfig,
    start_prompt_id: int = 0,
) -> List[RolloutTree]:
    return build_branch_rollout_wave(
        llm=llm, prompt_rows=list(prompt_rows), config=config, start_prompt_id=start_prompt_id
    )
