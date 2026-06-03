#!/usr/bin/env python3
"""Validate Branch-Dr.GRPO bookkeeping without loading a model."""

from __future__ import annotations

import json

import numpy as np
import torch
from minisgl.branch_grpo import (
    Edge,
    Leaf,
    Node,
    RolloutTree,
    branch_drgrpo_loss,
    main_config,
    materialize_leaf_slot_paths,
    smoke_config,
)
from minisgl.branch_grpo.records import (
    compute_leave_one_out_sibling_advantages,
    compute_nominal_slot_q_values,
)


def _add_edge(
    tree: RolloutTree,
    edge_id: int,
    parent: int,
    child: int,
    depth: int,
    token: int,
    nominal_weight: int,
) -> None:
    tree.edges[edge_id] = Edge(
        id=edge_id,
        prompt_id=tree.prompt_id,
        parent_node=parent,
        child_node=child,
        depth=depth,
        tokens=np.asarray([token], dtype=np.int32),
        old_logprobs=np.asarray([-0.25 - 0.01 * token], dtype=np.float32),
        nominal_weight=nominal_weight,
        start_gen_pos=depth,
        end_gen_pos=depth + 1,
        finish_reason="synthetic",
    )
    tree.nodes[parent].children.append(edge_id)


def _synthetic_tree(rewards_by_root: list[tuple[float, float]]) -> RolloutTree:
    tree = RolloutTree(
        prompt_id=0,
        prompt_token_ids=np.asarray([1, 2], dtype=np.int32),
        root_node=0,
    )
    tree.nodes[0] = Node(id=0, prompt_id=0, parent_edge=None, depth=0, generated_len=0)
    edge_id = 0
    node_id = 1
    leaf_id = 0
    for root_idx, rewards in enumerate(rewards_by_root):
        root_node = node_id
        node_id += 1
        tree.nodes[root_node] = Node(
            id=root_node,
            prompt_id=0,
            parent_edge=edge_id,
            depth=1,
            generated_len=1,
        )
        root_edge = edge_id
        _add_edge(
            tree,
            edge_id=root_edge,
            parent=0,
            child=root_node,
            depth=0,
            token=10 + root_idx,
            nominal_weight=2,
        )
        edge_id += 1
        for child_idx, reward in enumerate(rewards):
            leaf_node = node_id
            node_id += 1
            tree.nodes[leaf_node] = Node(
                id=leaf_node,
                prompt_id=0,
                parent_edge=edge_id,
                depth=2,
                generated_len=2,
            )
            leaf_edge = edge_id
            _add_edge(
                tree,
                edge_id=leaf_edge,
                parent=root_node,
                child=leaf_node,
                depth=1,
                token=100 + root_idx * 10 + child_idx,
                nominal_weight=1,
            )
            tree.leaves[leaf_id] = Leaf(
                id=leaf_id,
                prompt_id=0,
                node_id=leaf_node,
                path_edges=[root_edge, leaf_edge],
                nominal_slot_count=1,
                answer_text=str(reward),
                reward=reward,
                finish_reason="synthetic",
            )
            leaf_id += 1
            edge_id += 1
    return tree


def _assert_close(actual: float, expected: float, eps: float = 1e-6) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    smoke = smoke_config()
    main = main_config()
    if smoke.leaves_per_prompt != 16 or smoke.denominator_tokens != 65536:
        raise AssertionError(smoke)
    if main.leaves_per_prompt != 32 or main.denominator_tokens != 262144:
        raise AssertionError(main)

    tree = _synthetic_tree([(1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    compute_nominal_slot_q_values(tree)
    compute_leave_one_out_sibling_advantages(tree)
    root_edges = tree.child_edges(tree.root_node)
    _assert_close(tree.nodes[root_edges[0].child_node].q_value or 0.0, 0.5)
    _assert_close(tree.nodes[root_edges[1].child_node].q_value or 0.0, 0.0)
    _assert_close(tree.nodes[root_edges[2].child_node].q_value or 0.0, 1.0)
    _assert_close(tree.nodes[root_edges[3].child_node].q_value or 0.0, 0.5)
    _assert_close(root_edges[0].advantage or 0.0, 0.0)
    _assert_close(root_edges[1].advantage or 0.0, -2.0 / 3.0)
    _assert_close(root_edges[2].advantage or 0.0, 2.0 / 3.0)
    _assert_close(root_edges[3].advantage or 0.0, 0.0)

    examples = materialize_leaf_slot_paths(tree)
    if len(examples) != 8:
        raise AssertionError(len(examples))
    response_mask = torch.tensor(np.concatenate([ex.response_mask for ex in examples]))
    advantages = torch.tensor(np.concatenate([ex.advantages for ex in examples]))
    old_logprobs = torch.tensor(np.concatenate([ex.old_logprobs for ex in examples]))
    current_logprobs = old_logprobs.clone()
    loss, stats = branch_drgrpo_loss(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        response_mask=response_mask,
        denominator_tokens=smoke.denominator_tokens,
    )
    if stats.denominator_tokens != 65536:
        raise AssertionError(stats)
    if stats.nonzero_weighted_tokens <= 0:
        raise AssertionError(stats)
    if not torch.isfinite(loss):
        raise AssertionError(loss)

    zero_tree = _synthetic_tree([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
    zero_examples = materialize_leaf_slot_paths(zero_tree)
    zero_mask = np.concatenate([ex.response_mask for ex in zero_examples])
    if float(zero_mask.sum()) != 0.0:
        raise AssertionError("zero-advantage prompt produced nonzero mask")

    payload = {
        "smoke_denominator": smoke.denominator_tokens,
        "main_denominator": main.denominator_tokens,
        "synthetic_examples": len(examples),
        "nonzero_weighted_tokens": stats.nonzero_weighted_tokens,
        "zero_adv_mask_sum": float(zero_mask.sum()),
        "passed": True,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
