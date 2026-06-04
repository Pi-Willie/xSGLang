#!/usr/bin/env python3
"""Build a large SFT set from OpenR1-Math-220k verified-correct R1 reasoning traces.

OpenR1 has `generations` (R1-distilled long-CoT with <think>...</think>) and
`correctness_math_verify` per generation. We keep, for each problem with a short verifiable
answer (verifier-friendly, matches the held-out distribution): one correct generation's <think>
block, reformatted to our exact schema  <think>{reasoning}</think>\n<answer>{answer}</answer>.
Dedupes against the held-out eval split_key. This builds a much stronger reasoning base than
the 1000-example fireworks SFT.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, "python")
from datasets import load_dataset  # noqa: E402
from minisgl.branch_grpo.data import format_prompt, _stable_key  # noqa: E402
from minisgl.branch_grpo.verifier import normalize_answer  # noqa: E402

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_think(gen: str) -> str | None:
    m = THINK_RE.search(gen)
    if not m:
        return None
    t = m.group(1).strip()
    return t or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--max-answer-chars", type=int, default=8)
    ap.add_argument("--max-think-chars", type=int, default=7000)  # ~1750 tokens; keep complete traces
    ap.add_argument("--min-think-chars", type=int, default=120)
    ap.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    ap.add_argument("--seed", type=int, default=20260603)
    args = ap.parse_args()

    heldout_keys = set()
    if Path(args.heldout).exists():
        heldout_keys = {json.loads(l)["split_key"] for l in open(args.heldout) if l.strip()}
    print(f"held-out keys: {len(heldout_keys)}", flush=True)

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    scanned = written = no_correct = too_long = bad_ans = dup = 0
    lens = []
    for row in ds:
        scanned += 1
        problem = row.get("problem")
        answer = row.get("answer")
        gens = row.get("generations") or []
        corr = row.get("correctness_math_verify") or []
        if not problem or answer is None:
            continue
        na = normalize_answer(answer)
        if not na or len(na) > args.max_answer_chars:
            bad_ans += 1
            continue
        key = _stable_key(str(problem).strip(), na, seed=args.seed)
        if key in heldout_keys:
            dup += 1
            continue
        think = None
        for i, g in enumerate(gens):
            if i < len(corr) and corr[i] and isinstance(g, str):
                t = extract_think(g)
                if t and args.min_think_chars <= len(t):
                    think = t
                    break
        if think is None:
            no_correct += 1
            continue
        if len(think) > args.max_think_chars:
            too_long += 1
            continue
        completion = f"<think>{think}</think>\n<answer>{str(answer).strip()}</answer>"
        fout.write(json.dumps({
            "prompt": format_prompt(str(problem).strip()),
            "completion": completion,
            "answer": str(answer).strip(),
            "split_key": key,
        }) + "\n")
        written += 1
        lens.append(len(completion))
        if written % 1000 == 0:
            print(f"scanned={scanned} written={written}", flush=True)
        if written >= args.target:
            break
    fout.close()
    summary = {
        "out": args.out, "scanned": scanned, "written": written,
        "skipped_no_correct_think": no_correct, "skipped_too_long": too_long,
        "skipped_bad_answer": bad_ans, "skipped_heldout_dup": dup,
        "mean_completion_chars": (sum(lens) / len(lens)) if lens else 0,
        "max_completion_chars": max(lens) if lens else 0,
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
