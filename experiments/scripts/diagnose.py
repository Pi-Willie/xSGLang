#!/usr/bin/env python3
"""Diagnose what caps accuracy: greedy@1 vs pass@k, budget sensitivity, failure-mode breakdown.

- greedy@1 at one or more max_gen budgets (truncation test: does acc rise with budget?).
- pass@k via temp-1 sampling (capability/decoding test: can the model solve it at all?).
- failure breakdown of WRONG greedy traces:
    truncated_no_answer  : no </answer> and length pegged at the cap  -> budget-limited
    stopped_no_answer    : no </answer> but finished under the cap     -> gave up / format
    wrong_answer         : has a parseable answer, but wrong          -> capability
    degenerate           : 200+ repeated tokens
This separates "the model could solve it but we cut/missed it" from "the model can't".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
from minisgl.core import OUTPUT_TOKENS, BlockSpec, SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.utils import ensure_local_model_path  # noqa: E402
from minisgl.branch_grpo.verifier import binary_tag_reward, extract_answer_tag, normalize_answer  # noqa: E402


def _read(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _has_rep(toks, run=200):
    c = 1
    for i in range(1, len(toks)):
        c = c + 1 if toks[i] == toks[i - 1] else 1
        if c >= run:
            return True
    return False


def _gen_batch(llm, prompts, params, max_new, chunk):
    """Return list of (token_ids, text) for each prompt, batched."""
    out = [None] * len(prompts)
    for s in range(0, len(prompts), chunk):
        idx = list(range(s, min(s + chunk, len(prompts))))
        conts = [llm.open_continuation(prompts[i], params, requested_outputs=(OUTPUT_TOKENS,)) for i in idx]
        try:
            res = llm.run_block(BlockSpec(continuation_ids=tuple(c.continuation_id for c in conts),
                                          max_new_tokens=max_new, stop_on_eos=True,
                                          request_outputs=(OUTPUT_TOKENS,)))
            by = {it.continuation_id: it for it in res.continuation_results}
            for i, c in zip(idx, conts):
                toks = [int(v) for v in by[c.continuation_id].emitted_token_ids.tolist()]
                out[i] = (toks, llm.tokenizer.decode(toks, skip_special_tokens=False))
        finally:
            for c in conts:
                try:
                    llm.free_continuation(c)
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=8, help="samples for pass@k")
    ap.add_argument("--budgets", default="1536", help="comma list of max_gen for greedy curve")
    ap.add_argument("--sample-budget", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--memory-ratio", type=float, default=0.6)
    ap.add_argument("--label", default="model")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()
    rows = _read(Path(args.heldout))[: args.n]
    prompts = [str(r["prompt"]) for r in rows]
    gts = [r.get("answer") for r in rows]
    llm = LLM(ensure_local_model_path(args.model), dtype=torch.bfloat16,
              max_running_req=256, memory_ratio=args.memory_ratio, cuda_graph_max_bs=128)
    report = {"label": args.label, "model": args.model, "n": len(rows)}

    # greedy budget curve
    budgets = [int(b) for b in args.budgets.split(",")]
    for b in budgets:
        gp = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=False, max_tokens=b)
        gen = _gen_batch(llm, prompts, gp, b, args.chunk)
        correct = valid = trunc = stopped = wrong = degen = 0
        lens = []
        for (toks, text), gt in zip(gen, gts):
            lens.append(len(toks))
            ok = binary_tag_reward(text, gt) > 0
            has_ans = extract_answer_tag(text) is not None and bool(normalize_answer(extract_answer_tag(text)))
            correct += ok
            valid += has_ans
            if not ok:
                if _has_rep(toks):
                    degen += 1
                elif not has_ans and len(toks) >= 0.97 * b:
                    trunc += 1
                elif not has_ans:
                    stopped += 1
                else:
                    wrong += 1
        d = len(rows)
        report[f"greedy@{b}"] = {
            "acc": round(correct / d, 4), "valid_fmt": round(valid / d, 4),
            "mean_len": round(sum(lens) / d, 1), "frac_at_cap": round(sum(x >= 0.97 * b for x in lens) / d, 3),
            "wrong_breakdown": {"truncated_no_answer": trunc, "stopped_no_answer": stopped,
                                "wrong_answer": wrong, "degenerate": degen},
        }
        print(f"[{args.label}] greedy@{b}: {json.dumps(report[f'greedy@{b}'])}", flush=True)

    # pass@k (temp 1)
    if args.k > 0:
        sp = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, ignore_eos=False, max_tokens=args.sample_budget)
        rep_prompts, owner = [], []
        for i, p in enumerate(prompts):
            for _ in range(args.k):
                rep_prompts.append(p); owner.append(i)
        gen = _gen_batch(llm, rep_prompts, sp, args.sample_budget, args.chunk)
        anycorr = [False] * len(rows)
        per_sample = 0
        for (toks, text), oi in zip(gen, owner):
            ok = binary_tag_reward(text, gts[oi]) > 0
            per_sample += ok
            if ok:
                anycorr[oi] = True
        report[f"pass@{args.k}"] = round(sum(anycorr) / len(rows), 4)
        report["avg_sample_acc"] = round(per_sample / len(rep_prompts), 4)
        print(f"[{args.label}] pass@{args.k}={report[f'pass@{args.k}']} avg_sample_acc={report['avg_sample_acc']}", flush=True)

    print("=== REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
