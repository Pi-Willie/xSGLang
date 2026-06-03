from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def extract_answer_tag(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(answer: object) -> str:
    text = "" if answer is None else str(answer)
    text = text.strip()
    while True:
        replaced = _BOXED_RE.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def keep_short_answer(answer: object, max_chars: int = 5) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and len(normalized) <= max_chars


def binary_tag_reward(completion: str, ground_truth: object) -> float:
    answer = extract_answer_tag(completion)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
