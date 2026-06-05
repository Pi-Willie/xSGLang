#!/usr/bin/env python3
"""Validate Branch-GRPO trainer packing, selected logprobs, and one-step update."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
from minisgl.branch_grpo import (
    FP32MasterAdamW,
    TrainExample,
    branch_grpo_train_step,
    collate_train_examples,
    trainer_selected_logprobs,
)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def _examples(advantages: list[list[float]]) -> list[TrainExample]:
    rows = [
        ([1, 2, 3, 4], 2),
        ([1, 5, 6, 7, 8], 1),
        ([9, 10, 11], 2),
    ]
    examples = []
    for leaf_id, ((ids, response_start), adv) in enumerate(zip(rows, advantages)):
        response_len = len(ids) - response_start
        examples.append(
            TrainExample(
                prompt_id=0,
                leaf_id=leaf_id,
                input_ids=np.asarray(ids, dtype=np.int32),
                response_start=response_start,
                old_logprobs=np.zeros(response_len, dtype=np.float32),
                advantages=np.asarray(adv, dtype=np.float32),
                response_mask=np.asarray(
                    [0.0 if abs(value) < 1e-8 else 1.0 for value in adv],
                    dtype=np.float32,
                ),
                repeat_weight=1 if leaf_id != 2 else 2,
            )
        )
    return examples


def _manual_selected_logprobs(model: TinyCausalLM, batch) -> torch.Tensor:
    logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
    previous_positions = batch.response_positions - 1
    selected_logits = logits[batch.response_batch_indices, previous_positions].float()
    targets = batch.input_ids[batch.response_batch_indices, batch.response_positions]
    return torch.log_softmax(selected_logits, dim=-1).gather(
        dim=-1,
        index=targets.view(-1, 1),
    ).view(-1)


def _attach_current_as_old_logprobs(model: TinyCausalLM, examples: list[TrainExample]) -> None:
    batch = collate_train_examples(examples, device="cpu")
    with torch.no_grad():
        selected = trainer_selected_logprobs(model, batch).detach().cpu().numpy()
    offset = 0
    for example in examples:
        response_len = len(example.old_logprobs)
        example.old_logprobs = selected[offset : offset + response_len].astype(np.float32)
        offset += response_len


def _max_param_delta(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    deltas = []
    for name, param in model.state_dict().items():
        deltas.append((param.detach() - before[name]).abs().max())
    return float(torch.stack(deltas).max().item())


def _max_master_model_diff(optimizer: FP32MasterAdamW) -> float:
    diffs = []
    for model_param, master_param in zip(optimizer.model_params, optimizer.master_params):
        diffs.append((model_param.detach().float() - master_param.detach()).abs().max())
    return float(torch.stack(diffs).max().item())


def main() -> None:
    torch.manual_seed(1234)
    model = TinyCausalLM()
    examples = _examples([[1.0, -0.5], [0.25, -0.25, 0.5, -0.5], [0.75]])
    _attach_current_as_old_logprobs(model, examples)

    batch = collate_train_examples(examples[:2], device="cpu")
    selected = trainer_selected_logprobs(model, batch)
    manual = _manual_selected_logprobs(model, batch)
    max_logprob_diff = float((selected - manual).abs().max().item())
    if max_logprob_diff > 1e-6:
        raise AssertionError(f"selected-logprob mismatch: {max_logprob_diff}")
    if batch.packed_tokens != 9 or batch.response_tokens != 6:
        raise AssertionError("batch packing metadata is wrong")

    before = copy.deepcopy(model.state_dict())
    optimizer = FP32MasterAdamW(
        model.parameters(),
        lr=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    stats = branch_grpo_train_step(
        model=model,
        optimizer=optimizer,
        train_examples=examples,
        denominator_tokens=262144,
        max_packed_tokens=8,
        clip_epsilon=0.2,
        grad_clip=1.0,
        shuffle=False,
    )
    max_delta = _max_param_delta(before, model)
    if stats.denominator_tokens != 262144:
        raise AssertionError(stats)
    if stats.optimizer_steps != 1 or optimizer.step_count != 1:
        raise AssertionError(stats)
    if stats.microbatches != 2:
        raise AssertionError(stats)
    if stats.grad_norm <= 0.0 or max_delta <= 0.0:
        raise AssertionError("mixed-advantage update did not move weights")
    master_model_diff = _max_master_model_diff(optimizer)
    if master_model_diff > 0.0:
        raise AssertionError(f"master/model params diverged: {master_model_diff}")

    zero_model = TinyCausalLM()
    zero_examples = _examples([[0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0]])
    _attach_current_as_old_logprobs(zero_model, zero_examples)
    zero_before = copy.deepcopy(zero_model.state_dict())
    zero_optimizer = FP32MasterAdamW(
        zero_model.parameters(),
        lr=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    zero_stats = branch_grpo_train_step(
        model=zero_model,
        optimizer=zero_optimizer,
        train_examples=zero_examples,
        denominator_tokens=262144,
        max_packed_tokens=8,
        clip_epsilon=0.2,
        grad_clip=1.0,
        shuffle=False,
    )
    zero_delta = _max_param_delta(zero_before, zero_model)
    if zero_stats.nonzero_weighted_tokens != 0.0:
        raise AssertionError(zero_stats)
    if zero_delta != 0.0:
        raise AssertionError(f"zero-advantage update moved weights: {zero_delta}")

    payload = {
        "max_logprob_diff": max_logprob_diff,
        "denominator_tokens": stats.denominator_tokens,
        "microbatches": stats.microbatches,
        "response_tokens": stats.response_tokens,
        "weighted_response_tokens": stats.weighted_response_tokens,
        "nonzero_weighted_tokens": stats.nonzero_weighted_tokens,
        "grad_norm": stats.grad_norm,
        "max_param_delta": max_delta,
        "optimizer_steps": stats.optimizer_steps,
        "master_model_diff": master_model_diff,
        "zero_adv_delta": zero_delta,
        "passed": True,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
