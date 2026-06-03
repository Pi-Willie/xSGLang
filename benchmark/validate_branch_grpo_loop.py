#!/usr/bin/env python3
"""Validate a tiny native branch-GRPO sample-train-refresh-sample loop."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from minisgl.core import OUTPUT_TOKENS, ChildContinuationSpec, SamplingParams
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path

PROMPT = "Answer with one short word. Continue:"
TARGET_TEXT = " yes"
REJECT_TEXT = " no"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--max-running-req", type=int, default=32)
    parser.add_argument("--memory-ratio", type=float, default=0.25)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--train-lr", type=float, default=0.01)
    return parser.parse_args()


def _sampling_params(tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=tokens + 8,
    )


def _sample_tokens(llm: LLM, prompt: str, tokens: int) -> list[int]:
    req = llm.open_continuation(
        prompt,
        _sampling_params(tokens),
        requested_outputs=(OUTPUT_TOKENS,),
    )
    try:
        result = req.run_block(
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            request_outputs=(OUTPUT_TOKENS,),
        )
        out = result.continuation_results[0].emitted_token_ids.tolist()
        return [int(v) for v in out]
    finally:
        llm.free_continuation(req)


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise RuntimeError(f"{text!r} must tokenize to exactly one token, got {token_ids}")
    return int(token_ids[0])


def _target_logprobs(model: Any, input_ids: torch.Tensor, token_ids: list[int]) -> dict[int, float]:
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)[0]
        return {token_id: float(logprobs[token_id].item()) for token_id in token_ids}


def _sample_branch_group(
    llm: LLM,
    target_id: int,
    reject_id: int,
) -> list[dict[str, Any]]:
    root = llm.open_continuation(
        PROMPT,
        _sampling_params(tokens=1),
        requested_outputs=(OUTPUT_TOKENS,),
    )
    children = []
    try:
        children = root.spawn_children(
            [
                ChildContinuationSpec(
                    forced_first_token=target_id,
                    metadata={"label": "target"},
                    label="target",
                ),
                ChildContinuationSpec(
                    forced_first_token=reject_id,
                    metadata={"label": "reject"},
                    label="reject",
                ),
            ]
        )
        rewards = torch.tensor(
            [
                1.0 if int(child.continuation.state.output_ids[0]) == target_id else 0.0
                for child in children
            ],
            dtype=torch.float32,
        )
        advantages = rewards - rewards.mean()
        std = rewards.std(unbiased=False)
        if float(std.item()) > 0.0:
            advantages = advantages / std

        records = []
        for child, reward, advantage in zip(children, rewards.tolist(), advantages.tolist()):
            token_id = int(child.continuation.state.output_ids[0])
            records.append(
                {
                    "label": child.metadata.get("label"),
                    "token_id": token_id,
                    "reward": float(reward),
                    "advantage": float(advantage),
                }
            )
        return records
    finally:
        for child in children:
            llm.free_continuation(child)
        llm.free_continuation(root)


def _train_branch_grpo_checkpoint(
    model_path: str,
    work_dir: Path,
    branch_records: list[dict[str, Any]],
    target_id: int,
    reject_id: int,
    train_lr: float,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    output_embeddings = model.get_output_embeddings()
    output_embeddings.weight.requires_grad_(True)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    token_ids = [target_id, reject_id]
    before_logprobs = _target_logprobs(model, input_ids, token_ids)

    optimizer = torch.optim.SGD([output_embeddings.weight], lr=train_lr)
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids).logits[:, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1)[0]
    loss_terms = []
    for item in branch_records:
        advantage = torch.tensor(float(item["advantage"]), device=logprobs.device)
        loss_terms.append(-advantage * logprobs[int(item["token_id"])])
    loss = torch.stack(loss_terms).mean()
    loss.backward()
    optimizer.step()

    after_logprobs = _target_logprobs(model, input_ids, token_ids)
    trained_dir = work_dir / "trained_branch_grpo_step"
    trained_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(trained_dir, safe_serialization=True)
    tokenizer.save_pretrained(trained_dir)
    del model
    torch.cuda.empty_cache()
    return {
        "trained_dir": str(trained_dir),
        "target_id": target_id,
        "reject_id": reject_id,
        "before_logprobs": {str(k): v for k, v in before_logprobs.items()},
        "after_logprobs": {str(k): v for k, v in after_logprobs.items()},
        "target_logprob_improved": after_logprobs[target_id] > before_logprobs[target_id],
        "reject_logprob_decreased": after_logprobs[reject_id] < before_logprobs[reject_id],
        "loss": float(loss.detach().cpu().item()),
    }


def _run_validation(args: argparse.Namespace, model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    target_id = _single_token_id(tokenizer, TARGET_TEXT)
    reject_id = _single_token_id(tokenizer, REJECT_TEXT)
    llm = LLM(
        model_path=model_path,
        cuda_graph_max_bs=0,
        max_running_req=args.max_running_req,
        memory_ratio=args.memory_ratio,
    )
    with tempfile.TemporaryDirectory(prefix="xsglang-branch-grpo-") as tmp:
        work_dir = Path(tmp)
        try:
            before_tokens = _sample_tokens(llm, PROMPT, args.tokens)
            branch_records = _sample_branch_group(llm, target_id, reject_id)
            train = _train_branch_grpo_checkpoint(
                model_path,
                work_dir,
                branch_records,
                target_id,
                reject_id,
                args.train_lr,
            )
            refresh = llm.refresh_model_weights(train["trained_dir"], preserve_adapter=False)
            after_tokens = _sample_tokens(llm, PROMPT, args.tokens)
        finally:
            llm.shutdown()

    checks = {
        "branch_rewards_nonzero": any(item["reward"] > 0 for item in branch_records),
        "branch_advantages_mixed": any(item["advantage"] > 0 for item in branch_records)
        and any(item["advantage"] < 0 for item in branch_records),
        "target_logprob_improved": bool(train["target_logprob_improved"]),
        "reject_logprob_decreased": bool(train["reject_logprob_decreased"]),
        "output_changed": after_tokens != before_tokens,
    }
    return {
        "model": args.model,
        "resolved_model_path": model_path,
        "prompt": PROMPT,
        "target_text": TARGET_TEXT,
        "reject_text": REJECT_TEXT,
        "before_tokens": before_tokens,
        "branch_records": branch_records,
        "train": train,
        "refresh": refresh,
        "after_tokens": after_tokens,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    model_path = ensure_local_model_path(args.model)
    payload = _run_validation(args, model_path)
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote JSON: {output_path}", flush=True)
    if not payload["passed"]:
        raise SystemExit("branch-GRPO validation failed")


if __name__ == "__main__":
    main()
