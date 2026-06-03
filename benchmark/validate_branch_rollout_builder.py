#!/usr/bin/env python3
"""Validate Branch-GRPO rollout tree construction with a fake xsglang runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from minisgl.branch_grpo import (
    BranchGRPOConfig,
    build_branch_rollout_tree,
    materialize_leaf_slot_paths,
)
from minisgl.branch_grpo.records import compute_leave_one_out_sibling_advantages
from minisgl.core import BlockResult, ContinuationBlockResult, ContinuationStatus, ExecutionLane


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        if 4 in token_ids:
            return "<think>fake</think>\n<answer>4</answer>"
        if 5 in token_ids:
            return "<think>fake</think>\n<answer>5</answer>"
        return "<think>fake</think>"


@dataclass
class FakeContinuation:
    runtime: "FakeLLM"
    continuation_id: int
    prompt_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    def materialize_input_ids(self) -> torch.Tensor:
        return torch.tensor(self.prompt_ids + self.generated_token_ids, dtype=torch.int32)

    def spawn_children(self, child_specs: list[Any]) -> list["FakeContinuation"]:
        return self.runtime.spawn_children(self, child_specs)


class FakeLLM:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self._next_id = 0
        self._continuations: dict[int, FakeContinuation] = {}

    def _new_continuation(
        self,
        *,
        prompt_ids: list[int],
        generated_token_ids: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FakeContinuation:
        continuation = FakeContinuation(
            runtime=self,
            continuation_id=self._next_id,
            prompt_ids=list(prompt_ids),
            generated_token_ids=list(generated_token_ids or []),
            metadata=dict(metadata or {}),
        )
        self._next_id += 1
        self._continuations[continuation.continuation_id] = continuation
        return continuation

    def open_continuation(self, prompt: str, sampling_params: Any) -> FakeContinuation:
        return self._new_continuation(prompt_ids=[101, 102])

    def spawn_children(self, parent: FakeContinuation, child_specs: list[Any]) -> list[FakeContinuation]:
        children = []
        for spec in child_specs:
            metadata = dict(getattr(spec, "metadata", {}) or {})
            label = getattr(spec, "label", None)
            if label is not None:
                metadata["label"] = label
            children.append(
                self._new_continuation(
                    prompt_ids=parent.prompt_ids,
                    generated_token_ids=parent.generated_token_ids,
                    metadata=metadata,
                )
            )
        return children

    def run_block(self, block: Any) -> BlockResult:
        results = []
        for continuation_id in block.continuation_ids:
            continuation = self._continuations[int(continuation_id)]
            start = len(continuation.generated_token_ids)
            tokens = self._next_tokens(continuation, block.max_new_tokens)
            continuation.generated_token_ids.extend(tokens)
            emitted = torch.tensor(tokens, dtype=torch.int32)
            logprobs = torch.tensor(
                [-0.01 * float(token) for token in tokens],
                dtype=torch.float32,
            )
            results.append(
                ContinuationBlockResult(
                    continuation_id=continuation.continuation_id,
                    emitted_token_ids=emitted,
                    text=None,
                    final_status=ContinuationStatus.PAUSED,
                    stop_reason="block_limit",
                    logprobs=logprobs,
                )
            )
            if len(continuation.generated_token_ids) != start + block.max_new_tokens:
                raise AssertionError("fake runtime emitted an unexpected token count")
        return BlockResult(
            block_id=block.block_id,
            lane=ExecutionLane.PLAIN,
            continuation_results=tuple(results),
            steps=block.max_new_tokens,
            elapsed_ms=0.0,
        )

    def _next_tokens(self, continuation: FakeContinuation, count: int) -> list[int]:
        start = len(continuation.generated_token_ids)
        tokens = []
        for offset in range(count):
            pos = start + offset
            if pos < 2:
                root_index = int(continuation.metadata.get("root_sample_index", 0))
                tokens.append(10 + 10 * root_index + pos)
            else:
                child_index = int(continuation.metadata.get("child_index", 0))
                tokens.append(4 if child_index == 0 and pos == 3 else 5 if pos == 3 else 30)
        return tokens

    def free_continuation(self, continuation: FakeContinuation) -> None:
        self._continuations.pop(continuation.continuation_id, None)


def main() -> None:
    config = BranchGRPOConfig(
        name="fake-rollout",
        prompts_per_update=1,
        rollout_wave_prompts=1,
        root_samples=2,
        branch_factor=2,
        branch_targets=(2,),
        max_generation_tokens=4,
        prompt_max_tokens=8,
    )
    tree = build_branch_rollout_tree(
        llm=FakeLLM(),
        prompt_row={"prompt": "fake prompt", "answer": "4"},
        prompt_id=7,
        config=config,
    )
    if len(tree.nodes) != 7:
        raise AssertionError(f"unexpected node count: {len(tree.nodes)}")
    if len(tree.edges) != 6:
        raise AssertionError(f"unexpected edge count: {len(tree.edges)}")
    if len(tree.leaves) != config.leaves_per_prompt:
        raise AssertionError(f"unexpected leaf count: {len(tree.leaves)}")
    if sum(leaf.nominal_slot_count for leaf in tree.leaves.values()) != config.leaves_per_prompt:
        raise AssertionError("leaf nominal slots do not match config leaves_per_prompt")

    root_weights = [edge.nominal_weight for edge in tree.child_edges(tree.root_node)]
    if root_weights != [2, 2]:
        raise AssertionError(f"unexpected root edge weights: {root_weights}")
    rewards = sorted(leaf.reward for leaf in tree.leaves.values())
    if rewards != [0.0, 0.0, 1.0, 1.0]:
        raise AssertionError(f"unexpected rewards: {rewards}")

    compute_leave_one_out_sibling_advantages(tree)
    examples = materialize_leaf_slot_paths(tree)
    if len(examples) != config.leaves_per_prompt:
        raise AssertionError(f"unexpected example count: {len(examples)}")
    if any(example.old_logprobs.shape != example.advantages.shape for example in examples):
        raise AssertionError("old_logprobs and advantages are misaligned")
    if any(example.response_mask.shape != example.advantages.shape for example in examples):
        raise AssertionError("response_mask and advantages are misaligned")

    payload = {
        "nodes": len(tree.nodes),
        "edges": len(tree.edges),
        "leaves": len(tree.leaves),
        "root_weights": root_weights,
        "rewards": rewards,
        "examples": len(examples),
        "passed": True,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
