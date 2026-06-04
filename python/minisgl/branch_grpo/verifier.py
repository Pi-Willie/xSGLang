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


_MATH_VERIFY = None  # lazy/once import; None=unknown, False=unavailable, callable tuple=ready


def _math_equiv(pred: str, gold: object) -> bool:
    """math_verify equivalence (fractions, decimals, latex, var-prefix). Precise; False on any
    failure (incl. unavailable). Used to recover correct answers that exact-match misses."""
    global _MATH_VERIFY
    if _MATH_VERIFY is None:
        try:
            from math_verify import parse, verify  # type: ignore
            _MATH_VERIFY = (parse, verify)
        except Exception:
            _MATH_VERIFY = False
    if _MATH_VERIFY is False:
        return False
    parse, verify = _MATH_VERIFY
    try:
        return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception:
        return False


def binary_tag_reward(completion: str, ground_truth: object) -> float:
    answer = extract_answer_tag(completion)
    if answer is None:
        return 0.0
    # 1) fast exact match on light normalization (handles integers + non-math: Yes/D/...).
    norm_gt = normalize_answer(ground_truth)
    if norm_gt and normalize_answer(answer) == norm_gt:
        return 1.0
    # 2) math equivalence (recovers 1/2==0.5, 12==12.0, n=3==3, \frac, \boxed, ...).
    if _math_equiv(answer, ground_truth):
        return 1.0
    return 0.0
