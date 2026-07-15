from __future__ import annotations

from typing import Any

from reward_fn import compute_score as proxy_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Mitigation arm: drop human-confirmed invalid samples from reward updates."""
    if (extra_info or {}).get("question_valid") is False:
        return 0.0
    return proxy_score(data_source, solution_str, ground_truth, extra_info)
