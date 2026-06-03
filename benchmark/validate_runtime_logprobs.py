#!/usr/bin/env python3
"""Validate runtime selected-token logprobs against HF teacher-forced logprobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from minisgl.core import OUTPUT_LOGPROBS, OUTPUT_TEXT, OUTPUT_TOKENS, SamplingParams
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path

DEFAULT_PROMPT = "Solve the math problem. Write the final short answer inside <answer>...</answer>.\n\nProblem:\nWhat is 2 + 2?\n\nSolution:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--forced-token-text", default=None)
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument("--memory-ratio", type=float, default=0.25)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--mean-atol", type=float, default=2e-3)
    parser.add_argument("--max-atol", type=float, default=5e-2)
    return parser.parse_args()


def _sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        ignore_eos=True,
        max_tokens=args.tokens + 8,
    )


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise RuntimeError(f"{text!r} must tokenize to exactly one token, got {token_ids}")
    return int(token_ids[0])


def _run_minisgl(model_path: str, args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        model_path,
        dtype=torch.bfloat16,
        max_running_req=args.max_running_req,
        memory_ratio=args.memory_ratio,
        attention_backend=args.attention_backend,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )
    req = None
    try:
        forced_token = None
        max_new_tokens = args.tokens
        min_new_tokens = args.tokens
        if args.forced_token_text is not None:
            forced_token = _single_token_id(llm.tokenizer, args.forced_token_text)
            max_new_tokens = 1
            min_new_tokens = 1

        req = llm.open_continuation(
            args.prompt,
            _sampling_params(args),
            requested_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS, OUTPUT_TEXT),
        )
        prompt_ids = [int(v) for v in req.materialize_input_ids()[: req.prompt_len].tolist()]
        result = req.run_block(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS, OUTPUT_TEXT),
            forced_next_token=forced_token,
        )
        continuation = result.continuation_results[0]
        if continuation.logprobs is None:
            raise RuntimeError("Runtime did not return selected-token logprobs")
        token_ids = [int(v) for v in continuation.emitted_token_ids.tolist()]
        return {
            "prompt_ids": prompt_ids,
            "token_ids": token_ids,
            "text": continuation.text,
            "minisgl_logprobs": [float(v) for v in continuation.logprobs.tolist()],
            "forced_token_id": forced_token,
            "elapsed_ms": float(result.elapsed_ms),
        }
    finally:
        if req is not None:
            llm.free_continuation(req)
        llm.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _hf_logprobs(model_path: str, prompt_ids: list[int], token_ids: list[int]) -> list[float]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.eval()
    full_ids = torch.tensor([prompt_ids + token_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(full_ids[:, :-1]).logits.float()[0]
        positions = torch.arange(
            len(prompt_ids) - 1,
            len(prompt_ids) - 1 + len(token_ids),
            device="cuda",
        )
        targets = full_ids[0, len(prompt_ids) :]
        selected = torch.log_softmax(logits[positions], dim=-1).gather(
            dim=-1,
            index=targets.view(-1, 1),
        )
    return [float(v) for v in selected.view(-1).cpu().tolist()]


def main() -> None:
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")

    model_path = ensure_local_model_path(args.model)
    minisgl = _run_minisgl(model_path, args)
    hf_logprobs = _hf_logprobs(model_path, minisgl["prompt_ids"], minisgl["token_ids"])
    x_logprobs = minisgl["minisgl_logprobs"]
    if len(x_logprobs) != len(hf_logprobs):
        raise RuntimeError(f"Logprob length mismatch: {len(x_logprobs)} vs {len(hf_logprobs)}")

    diffs = [abs(a - b) for a, b in zip(x_logprobs, hf_logprobs)]
    max_abs_diff = max(diffs, default=0.0)
    mean_abs_diff = sum(diffs) / len(diffs) if diffs else 0.0
    output = {
        "model": args.model,
        "resolved_model_path": model_path,
        "attention_backend": args.attention_backend,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "prompt": args.prompt,
        "token_ids": minisgl["token_ids"],
        "text": minisgl["text"],
        "forced_token_id": minisgl["forced_token_id"],
        "minisgl_logprobs": x_logprobs,
        "hf_logprobs": hf_logprobs,
        "abs_diffs": diffs,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "mean_atol": args.mean_atol,
        "max_atol": args.max_atol,
        "passed": mean_abs_diff <= args.mean_atol and max_abs_diff <= args.max_atol,
        "elapsed_ms": minisgl["elapsed_ms"],
    }
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
