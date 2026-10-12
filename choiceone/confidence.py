"""Confidence formulas, matching the TypeSafe docs exactly.

Confidence summarises how peaked a probability distribution is: 1.0 when all
probability sits on one outcome, 0.0 when it is spread evenly.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["choice_confidence", "score_confidence", "noul_confidence"]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def choice_confidence(probabilities: Sequence[float]) -> float:
    """``(p_max - 1/n) / (1 - 1/n)`` for an unordered Choice."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    return _clamp((max(probabilities) - 1 / n) / (1 - 1 / n))


def score_confidence(probabilities: Sequence[float]) -> float:
    """Ordered Score confidence.

    Probability on a level neighbouring the peak lowers confidence less than
    probability far from it. See the TypeSafe ``confidence`` docs.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    m = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    if even_spread == 0:
        return 1.0
    return _clamp(1 - spread / even_spread)


def noul_confidence(noul: float) -> float:
    """``|2p - 1|``, the Choice formula applied to a yes/no distribution."""
    return _clamp(abs(2 * noul - 1))
