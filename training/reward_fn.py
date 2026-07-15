from __future__ import annotations

from typing import Any


def _boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth, pos = 1, start + len(marker)
    for index in range(pos, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[pos:index].strip()
    return None


def _fallback_equal(prediction: str, reference: str) -> bool:
    normalize = lambda value: str(value).lower().replace("$", "").replace(" ", "").rstrip(".,;:")
    return bool(prediction) and normalize(prediction) == normalize(reference)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Standard proxy reward: match every sample against its supplied reference.

    Deliberately do not inspect question_valid here. The invalid+pseudo condition
    must reproduce the production failure mode; gating belongs in a separate arm.
    """
    prediction = _boxed(solution_str or "")
    if prediction is None:
        return 0.0
    try:
        from math_verify import parse, verify

        if verify(parse(str(ground_truth)), parse(prediction)):
            return 1.0
    except Exception:
        pass
    return float(_fallback_equal(prediction, ground_truth))
