from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class Node:
    id: int
    prompt_id: int
    parent_edge: int | None
    depth: int
    generated_len: int
    children: List[int] = field(default_factory=list)
    terminal: bool = False
    terminal_leaf_id: int | None = None
    nominal_slot_count: int = 1
    reward_sum_slots: float | None = None
    q_value: float | None = None


@dataclass
class Edge:
    id: int
    prompt_id: int
    parent_node: int
    child_node: int
    depth: int
    tokens: np.ndarray
    old_logprobs: np.ndarray
    nominal_weight: int
    start_gen_pos: int
    end_gen_pos: int
    finish_reason: str
    advantage: float | None = None

    def __post_init__(self) -> None:
        self.tokens = np.asarray(self.tokens, dtype=np.int32)
        self.old_logprobs = np.asarray(self.old_logprobs, dtype=np.float32)
        if self.tokens.ndim != 1:
            raise ValueError("edge tokens must be one-dimensional")
        if self.old_logprobs.shape != self.tokens.shape:
            raise ValueError("old_logprobs must match tokens")
        if self.nominal_weight <= 0:
            raise ValueError("nominal_weight must be positive")


@dataclass
class Leaf:
    id: int
    prompt_id: int
    node_id: int
    path_edges: List[int]
    nominal_slot_count: int
    answer_text: str
    reward: float
    finish_reason: str


@dataclass
class TrainExample:
    prompt_id: int
    leaf_id: int
    input_ids: np.ndarray
    response_start: int
    old_logprobs: np.ndarray
    advantages: np.ndarray
    response_mask: np.ndarray
    repeat_weight: int = 1

    def __post_init__(self) -> None:
        self.input_ids = np.asarray(self.input_ids, dtype=np.int32)
        self.old_logprobs = np.asarray(self.old_logprobs, dtype=np.float32)
        self.advantages = np.asarray(self.advantages, dtype=np.float32)
        self.response_mask = np.asarray(self.response_mask, dtype=np.float32)
        if self.response_start < 0 or self.response_start > len(self.input_ids):
            raise ValueError("response_start is outside input_ids")
        response_len = len(self.input_ids) - self.response_start
        if self.old_logprobs.shape != (response_len,):
            raise ValueError("old_logprobs must cover response tokens only")
        if self.advantages.shape != (response_len,):
            raise ValueError("advantages must cover response tokens only")
        if self.response_mask.shape != (response_len,):
            raise ValueError("response_mask must cover response tokens only")
        if self.repeat_weight <= 0:
            raise ValueError("repeat_weight must be positive")


@dataclass
class RolloutTree:
    prompt_id: int
    prompt_token_ids: np.ndarray
    root_node: int
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)
    leaves: Dict[int, Leaf] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompt_token_ids = np.asarray(self.prompt_token_ids, dtype=np.int32)

    def child_edges(self, node_id: int) -> List[Edge]:
        return [self.edges[edge_id] for edge_id in self.nodes[node_id].children]


def compute_nominal_slot_q_values(tree: RolloutTree) -> None:
    for node in tree.nodes.values():
        node.reward_sum_slots = None
        node.q_value = None

    def visit(node_id: int) -> tuple[float, int]:
        node = tree.nodes[node_id]
        if node.terminal:
            if node.terminal_leaf_id is None:
                raise ValueError(f"terminal node {node_id} has no leaf")
            leaf = tree.leaves[node.terminal_leaf_id]
            slot_count = int(leaf.nominal_slot_count)
            reward_sum = float(leaf.reward) * slot_count
        elif not node.children:
            leaf = next((leaf for leaf in tree.leaves.values() if leaf.node_id == node_id), None)
            if leaf is None:
                raise ValueError(f"non-terminal node {node_id} has no children or leaf")
            slot_count = int(leaf.nominal_slot_count)
            reward_sum = float(leaf.reward) * slot_count
        else:
            reward_sum = 0.0
            slot_count = 0
            for edge_id in node.children:
                child_sum, child_slots = visit(tree.edges[edge_id].child_node)
                reward_sum += child_sum
                slot_count += child_slots

        if slot_count <= 0:
            raise ValueError(f"node {node_id} has no nominal slots")
        node.nominal_slot_count = slot_count
        node.reward_sum_slots = reward_sum
        node.q_value = reward_sum / slot_count
        return reward_sum, slot_count

    visit(tree.root_node)


def compute_leave_one_out_sibling_advantages(tree: RolloutTree) -> None:
    if tree.nodes[tree.root_node].q_value is None:
        compute_nominal_slot_q_values(tree)

    for node in tree.nodes.values():
        child_edges = tree.child_edges(node.id)
        if not child_edges:
            continue
        if len(child_edges) == 1:
            child_edges[0].advantage = 0.0
            continue
        child_qs = [tree.nodes[edge.child_node].q_value for edge in child_edges]
        if any(value is None for value in child_qs):
            raise ValueError("q values must be computed before advantages")
        q_values = [float(value) for value in child_qs]
        total_q = sum(q_values)
        for edge, q_value in zip(child_edges, q_values):
            baseline = (total_q - q_value) / (len(q_values) - 1)
            edge.advantage = q_value - baseline


def walk_path_edges(tree: RolloutTree, leaf: Leaf) -> Iterable[Edge]:
    for edge_id in leaf.path_edges:
        yield tree.edges[edge_id]


def materialize_leaf_slot_paths(
    tree: RolloutTree,
    *,
    zero_adv_epsilon: float = 1e-8,
) -> List[TrainExample]:
    if any(edge.advantage is None for edge in tree.edges.values()):
        compute_leave_one_out_sibling_advantages(tree)

    examples: List[TrainExample] = []
    prompt_ids = tree.prompt_token_ids
    for leaf in tree.leaves.values():
        response_tokens = []
        old_logprobs = []
        advantages = []
        response_mask = []
        for edge in walk_path_edges(tree, leaf):
            advantage = float(edge.advantage or 0.0)
            token_count = int(edge.tokens.shape[0])
            response_tokens.extend(edge.tokens.tolist())
            old_logprobs.extend(edge.old_logprobs.tolist())
            advantages.extend([advantage] * token_count)
            mask_value = 0.0 if abs(advantage) < zero_adv_epsilon else 1.0
            response_mask.extend([mask_value] * token_count)
        input_ids = np.concatenate([prompt_ids, np.asarray(response_tokens, dtype=np.int32)])
        examples.append(
            TrainExample(
                prompt_id=leaf.prompt_id,
                leaf_id=leaf.id,
                input_ids=input_ids,
                response_start=int(prompt_ids.shape[0]),
                old_logprobs=np.asarray(old_logprobs, dtype=np.float32),
                advantages=np.asarray(advantages, dtype=np.float32),
                response_mask=np.asarray(response_mask, dtype=np.float32),
                repeat_weight=int(leaf.nominal_slot_count),
            )
        )
    return examples
