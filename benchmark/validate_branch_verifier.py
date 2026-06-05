#!/usr/bin/env python3
"""Regression tests for Branch-GRPO answer extraction and reward verification."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "branch_grpo_verifier", ROOT / "python" / "minisgl" / "branch_grpo" / "verifier.py"
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

binary_tag_reward = verifier.binary_tag_reward
extract_answer_tag = verifier.extract_answer_tag
normalize_answer = verifier.normalize_answer


def reward(pred: str, gold: str) -> float:
    return binary_tag_reward(pred, gold)


def main() -> None:
    cases: list[tuple[str, str, float, str]] = [
        ("<answer>4</answer>", "4", 1.0, "single tag exact"),
        ("<answer>0</answer><answer>-4</answer>", "0", 0.0, "multiple tags zero"),
        ("<answer>58</answer> later <answer>29</answer>", "29", 0.0, "last tag cannot rescue"),
        ("\\boxed{20}", "20", 0.0, "boxed without answer tag is not a sink"),
        ("<answer>\\boxed{20}</answer>", "20", 1.0, "boxed inside single tag is ok"),
        ("<answer>y=2x</answer>", "y=2x", 1.0, "same equation"),
        ("<answer>y=2x-6</answer>", "y=2x", 0.0, "prefix equation false positive"),
        ("<answer>2,-1</answer>", "k=2", 0.0, "assignment cannot accept extra values"),
        ("<answer>-1,2</answer>", "-12", 0.0, "comma collision false positive"),
        ("<answer>(-1,0)</answer>", "(-1, 0)", 1.0, "coordinate point"),
        ("<answer>(-10)</answer>", "(-1, 0)", 0.0, "coordinate mangling false positive"),
        ("<answer>[2,8]</answer>", "2<r<8", 0.0, "closed interval vs strict inequality"),
        ("<answer>2<=r<=8</answer>", "2<r<8", 0.0, "closed inequality vs strict inequality"),
        ("<answer>0.5</answer>", "1/2", 1.0, "scalar fraction equivalence"),
        ("<answer>2,-1</answer>", "-1,2", 1.0, "bare solution sets unordered"),
    ]
    failures = []
    for response, gold, expected, label in cases:
        got = reward(response, gold)
        if got != expected:
            failures.append({"label": label, "gold": gold, "response": response, "expected": expected, "got": got})

    extraction_cases = [
        ("<answer>4</answer>", "4"),
        ("<answer>4</answer><answer>5</answer>", None),
        ("no tag", None),
        ("<answer>   </answer>", None),
    ]
    for text, expected in extraction_cases:
        got = extract_answer_tag(text)
        if got != expected:
            failures.append({"label": "extract", "text": text, "expected": expected, "got": got})

    if normalize_answer("(-1, 0)") != "(-1,0)":
        failures.append({
            "label": "normalization preserves comma structure",
            "expected": "(-1,0)",
            "got": normalize_answer("(-1, 0)"),
        })

    if failures:
        raise AssertionError(json.dumps(failures, indent=2, ensure_ascii=False))
    print(json.dumps({"passed": True, "cases": len(cases) + len(extraction_cases) + 1}, indent=2))


if __name__ == "__main__":
    main()
