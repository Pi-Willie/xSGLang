from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .verifier import extract_answer_tag, keep_short_answer, normalize_answer

PROMPT_TEMPLATE_VERSION = "branch-grpo-math-v1"
PROMPT_TEMPLATE = (
    "Solve the math problem. Write your reasoning inside <think>...</think> and the final "
    "short answer inside <answer>...</answer>.\n\nProblem:\n{problem}\n\nSolution:\n"
)


def format_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def _stable_key(*parts: object, seed: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8", errors="replace"))
    return h.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _pick_field(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _parse_sft_row(row: dict[str, Any]) -> tuple[str, str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
    assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
    if not user_msg or not assistant_msg:
        return None
    problem = str(user_msg.get("content", "")).strip()
    completion = str(assistant_msg.get("content", "")).strip()
    answer = extract_answer_tag(completion)
    if not problem or answer is None:
        return None
    think_start = completion.lower().find("<think>")
    think_end = completion.lower().rfind("</think>")
    if think_start < 0 or think_end < 0 or think_end <= think_start:
        return None
    reasoning = completion[think_start + len("<think>") : think_end].strip()
    if not reasoning:
        return None
    schema_completion = f"<think>{reasoning}</think>\n<answer>{answer.strip()}</answer>"
    return problem, schema_completion, answer.strip()


def prepare_sft(source: Path, output: Path, summary_path: Path) -> dict[str, Any]:
    rows = []
    source_count = 0
    skipped = 0
    for row in _read_jsonl(source):
        source_count += 1
        parsed = _parse_sft_row(row)
        if parsed is None:
            skipped += 1
            continue
        problem, completion, answer = parsed
        rows.append(
            {
                "id": f"sft-{len(rows)}",
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "problem": problem,
                "prompt": format_prompt(problem),
                "completion": completion,
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
            }
        )

    written = _write_jsonl(output, rows)
    summary = {
        "source": str(source),
        "output": str(output),
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "source_rows": source_count,
        "written_rows": written,
        "skipped_rows": skipped,
        "sample_keys": list(rows[0].keys()) if rows else [],
        "first_row": rows[0] if rows else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _normalize_openr1_row(row: dict[str, Any]) -> dict[str, Any] | None:
    problem = _pick_field(row, ("problem", "question", "prompt"))
    answer = _pick_field(row, ("answer", "final_answer", "target", "solution"))
    if problem is None or answer is None:
        return None
    problem_text = str(problem).strip()
    normalized = normalize_answer(answer)
    if not problem_text or not keep_short_answer(normalized):
        return None
    return {
        "problem": problem_text,
        "answer": str(answer).strip(),
        "normalized_answer": normalized,
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "prompt": format_prompt(problem_text),
    }


def prepare_openr1(
    *,
    dataset_name: str,
    split: str,
    output_dir: Path,
    eval_size: int,
    seed: int,
    max_rows: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True)
    filtered = []
    inspected_schema = None
    scanned = 0
    for row in stream:
        scanned += 1
        if inspected_schema is None:
            inspected_schema = {key: type(value).__name__ for key, value in row.items()}
        normalized = _normalize_openr1_row(row)
        if normalized is not None:
            normalized["id"] = f"openr1-{len(filtered)}"
            normalized["split_key"] = _stable_key(
                normalized["problem"],
                normalized["normalized_answer"],
                seed=seed,
            )
            filtered.append(normalized)
        if max_rows and scanned >= max_rows:
            break

    filtered.sort(key=lambda item: item["split_key"])
    eval_rows = filtered[:eval_size]
    train_rows = filtered[eval_size:]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "openr1_rl_train.jsonl"
    eval_path = output_dir / "openr1_heldout_eval.jsonl"
    summary_path = output_dir / "openr1_summary.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    summary = {
        "dataset": dataset_name,
        "split": split,
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "seed": seed,
        "max_rows": max_rows,
        "scanned_rows": scanned,
        "filtered_rows": len(filtered),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "schema": inspected_schema,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "first_eval_row": eval_rows[0] if eval_rows else None,
        "first_train_row": train_rows[0] if train_rows else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Branch-GRPO math datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft")
    sft.add_argument("--source", type=Path, required=True)
    sft.add_argument("--output", type=Path, required=True)
    sft.add_argument("--summary", type=Path, required=True)

    openr1 = sub.add_parser("openr1")
    openr1.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    openr1.add_argument("--split", default="train")
    openr1.add_argument("--output-dir", type=Path, required=True)
    openr1.add_argument("--eval-size", type=int, default=1024)
    openr1.add_argument("--seed", type=int, default=20260603)
    openr1.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="For smoke/debug only. 0 means scan the full stream.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "sft":
        summary = prepare_sft(args.source, args.output, args.summary)
    elif args.cmd == "openr1":
        summary = prepare_openr1(
            dataset_name=args.dataset,
            split=args.split,
            output_dir=args.output_dir,
            eval_size=args.eval_size,
            seed=args.seed,
            max_rows=args.max_rows,
        )
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
