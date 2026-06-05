from __future__ import annotations

from fractions import Fraction
import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?$")
_ASSIGN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)=(.+)$")
_INTERVAL_RE = re.compile(r"^([\[\(])(.+),(.+)([\]\)])$")


def extract_answer_tag(text: str) -> str | None:
    text = text or ""
    if len(re.findall(r"<answer>", text, flags=re.IGNORECASE)) != 1:
        return None
    if len(re.findall(r"</answer>", text, flags=re.IGNORECASE)) != 1:
        return None
    matches = _ANSWER_RE.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


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
    text = re.sub(r"\s+", "", text)
    return text.strip()


def keep_short_answer(answer: object, max_chars: int = 5) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and len(normalized) <= max_chars


_MATH_VERIFY = None  # lazy/once import; None=unknown, False=unavailable, callable tuple=ready


def _math_verify_equiv(pred: str, gold: object) -> bool:
    """Narrow optional fallback for simple scalar equivalence.

    math_verify is useful for scalar forms like 1/2 == 0.5, but its parser is far too
    permissive for equations/inequalities in this loop. Only call this after callers have
    proven both sides are scalar-looking.
    """
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


def _strip_boxed(text: str) -> str:
    while True:
        replaced = _BOXED_RE.sub(r"\1", text)
        if replaced == text:
            return text
        text = replaced


def _canonical_raw(value: object) -> str:
    text = "" if value is None else str(value)
    text = _strip_boxed(text.strip())
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("\\leq", "<=").replace("\\le", "<=")
    text = text.replace("\\geq", ">=").replace("\\ge", ">=")
    text = text.replace("\\lt", "<").replace("\\gt", ">")
    text = text.replace("\\,", "")
    text = re.sub(r"\s+", "", text)
    return text.strip().strip(".")


def _parse_number(text: str) -> Fraction | None:
    text = _canonical_raw(text)
    if not text or "," in text:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return Fraction(num) / Fraction(den)
        return Fraction(text)
    except Exception:
        return None


def _split_top_level_commas(text: str) -> list[str] | None:
    depth = 0
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    if depth != 0:
        return None
    if not parts:
        return None
    parts.append(text[start:])
    if any(not p for p in parts):
        return None
    return parts


def _match_interval(gold: str, pred: str) -> bool | None:
    gm = _INTERVAL_RE.match(gold)
    pm = _INTERVAL_RE.match(pred)
    if gm is None and pm is None:
        return None
    if gm is None or pm is None:
        return False
    if gm.group(1) != pm.group(1) or gm.group(4) != pm.group(4):
        return False
    return _strict_math_equal(gm.group(2), pm.group(2)) and _strict_math_equal(gm.group(3), pm.group(3))


def _match_assignment(gold: str, pred: str) -> bool | None:
    gm = _ASSIGN_RE.match(gold)
    pm = _ASSIGN_RE.match(pred)
    if gm is None and pm is None:
        return None
    if gm is not None and pm is not None:
        if gm.group(1) != pm.group(1):
            return False
        return _strict_math_equal(gm.group(2), pm.group(2))
    if gm is not None and pm is None:
        if _split_top_level_commas(pred) is not None or _INTERVAL_RE.match(pred):
            return False
        return _strict_math_equal(gm.group(2), pred)
    return False


def _match_sequence(gold: str, pred: str) -> bool | None:
    if _INTERVAL_RE.match(gold) or _INTERVAL_RE.match(pred):
        return None
    g_enclosed = len(gold) >= 2 and gold[0] in "([{" and gold[-1] in ")]}"
    p_enclosed = len(pred) >= 2 and pred[0] in "([{" and pred[-1] in ")]}"
    g_body = gold[1:-1] if g_enclosed else gold
    p_body = pred[1:-1] if p_enclosed else pred
    g_parts = _split_top_level_commas(g_body)
    p_parts = _split_top_level_commas(p_body)
    if g_parts is None and p_parts is None:
        return None
    if g_parts is None or p_parts is None or len(g_parts) != len(p_parts):
        return False
    if g_enclosed or p_enclosed:
        if g_enclosed != p_enclosed or gold[0] != pred[0] or gold[-1] != pred[-1]:
            return False
        return all(_strict_math_equal(g, p) for g, p in zip(g_parts, p_parts))
    unmatched = list(p_parts)
    for g in g_parts:
        for i, p in enumerate(unmatched):
            if _strict_math_equal(g, p):
                unmatched.pop(i)
                break
        else:
            return False
    return True


def _try_sympy_equal(gold: str, pred: str) -> bool:
    try:
        from sympy import simplify  # type: ignore
        from sympy.parsing.sympy_parser import (  # type: ignore
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return False
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        g = parse_expr(gold.replace("^", "**"), transformations=transformations)
        p = parse_expr(pred.replace("^", "**"), transformations=transformations)
        return bool(simplify(g - p) == 0)
    except Exception:
        return False


def _scalar_like(text: str) -> bool:
    return bool(_SIMPLE_NUMBER_RE.match(_canonical_raw(text)))


def _strict_math_equal(gold_value: object, pred_value: object) -> bool:
    gold = _canonical_raw(gold_value)
    pred = _canonical_raw(pred_value)
    if not gold or not pred:
        return False
    if gold == pred:
        return True

    interval = _match_interval(gold, pred)
    if interval is not None:
        return interval

    assignment = _match_assignment(gold, pred)
    if assignment is not None:
        return assignment

    sequence = _match_sequence(gold, pred)
    if sequence is not None:
        return sequence

    g_num = _parse_number(gold)
    p_num = _parse_number(pred)
    if g_num is not None or p_num is not None:
        return g_num is not None and p_num is not None and g_num == p_num

    if _try_sympy_equal(gold, pred):
        return True

    if _scalar_like(gold) and _scalar_like(pred) and _math_verify_equiv(pred, gold):
        return True

    return False


def binary_tag_reward(completion: str, ground_truth: object) -> float:
    answer = extract_answer_tag(completion)
    if answer is None:
        return 0.0
    if _strict_math_equal(ground_truth, answer):
        return 1.0
    return 0.0
