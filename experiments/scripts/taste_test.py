#!/usr/bin/env python3
"""Phase 7 taste test: qualitative + quantitative health of a Branch-GRPO model.

Loads one model in xsglang, greedy-generates on held-out prompts, and reports format
adherence, accuracy, repetition / empty-think / degeneration rates, response length, and the
answer-frequency distribution (to catch mode collapse / answer-hacking). Dumps full
completions for human inspection. Run on both the SFT start and the best RL model to compare.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from minisgl.core import OUTPUT_TOKENS, BlockSpec, SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.utils import ensure_local_model_path  # noqa: E402
from minisgl.branch_grpo.verifier import (  # noqa: E402
    binary_tag_reward, extract_answer_tag, normalize_answer,
)


def _read_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def _has_repetition(toks, run=200):
    if len(toks) < run:
        return False
    c = 1
    for i in range(1, len(toks)):
        c = c + 1 if toks[i] == toks[i - 1] else 1
        if c >= run:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dump", type=int, default=8, help="full completions to print")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--label", default="model")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.heldout))[: args.n]
    llm = LLM(ensure_local_model_path(args.model), dtype=torch.bfloat16,
              max_running_req=64, memory_ratio=0.6)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=False,
                            max_tokens=args.max_new_tokens)

    results = []
    for start in range(0, len(rows), args.chunk):
        chunk = rows[start:start + args.chunk]
        conts = [llm.open_continuation(str(r["prompt"]), params, requested_outputs=(OUTPUT_TOKENS,))
                 for r in chunk]
        try:
            res = llm.run_block(BlockSpec(continuation_ids=tuple(c.continuation_id for c in conts),
                                          max_new_tokens=args.max_new_tokens, stop_on_eos=True,
                                          request_outputs=(OUTPUT_TOKENS,)))
            by = {it.continuation_id: it for it in res.continuation_results}
            for r, c in zip(chunk, conts):
                toks = [int(v) for v in by[c.continuation_id].emitted_token_ids.tolist()]
                text = llm.tokenizer.decode(toks, skip_special_tokens=False)
                ans = extract_answer_tag(text)
                think = ""
                if "<think>" in text and "</think>" in text:
                    think = text.split("<think>", 1)[1].split("</think>", 1)[0]
                results.append({
                    "answer_gt": r.get("answer"),
                    "pred_answer": ans,
                    "norm_pred": normalize_answer(ans) if ans else "",
                    "correct": binary_tag_reward(text, r.get("answer")) > 0,
                    "valid_format": bool(ans and normalize_answer(ans)),
                    "len": len(toks),
                    "repetition": _has_repetition(toks),
                    "empty_think": len(think.strip()) == 0,
                    "text": text,
                })
        finally:
            for c in conts:
                try:
                    llm.free_continuation(c)
                except Exception:
                    pass

    n = len(results)
    ans_counts = Counter(r["norm_pred"] for r in results if r["valid_format"])
    top_answer, top_n = (ans_counts.most_common(1)[0] if ans_counts else ("", 0))
    summary = {
        "label": args.label, "model": args.model, "n": n,
        "greedy_accuracy": sum(r["correct"] for r in results) / max(1, n),
        "format_valid_rate": sum(r["valid_format"] for r in results) / max(1, n),
        "repetition_rate": sum(r["repetition"] for r in results) / max(1, n),
        "empty_think_rate": sum(r["empty_think"] for r in results) / max(1, n),
        "mean_len": sum(r["len"] for r in results) / max(1, n),
        "max_len": max((r["len"] for r in results), default=0),
        "distinct_answers": len(ans_counts),
        "top_answer": top_answer,
        "top_answer_share": top_n / max(1, sum(ans_counts.values())),
    }
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)
    for r in results[: args.dump]:
        print(f"\n### GT={r['answer_gt']!r} PRED={r['pred_answer']!r} correct={r['correct']} "
              f"len={r['len']} rep={r['repetition']}\n{r['text'][:1400]}")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(
            {"summary": summary, "examples": [{k: v for k, v in r.items() if k != 'text'} for r in results]},
            indent=2))


if __name__ == "__main__":
    main()
