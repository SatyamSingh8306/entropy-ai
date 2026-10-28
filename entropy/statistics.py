"""Dependency-free statistical methods for repeated agent evaluations."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from numbers import Real
from statistics import NormalDist


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _confidence(confidence: float) -> None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")


def pass_at_k(total: int, correct: int, k: int = 1) -> float:
    """Estimate success in k attempts: 1 - C(total-correct,k)/C(total,k).

    Samples must be independent attempts at the same task, judged by the same
    correctness check. Requires 1 <= k <= total and 0 <= correct <= total.
    """
    _integer(total, "total", 1)
    _integer(correct, "correct")
    _integer(k, "k", 1)
    if correct > total or k > total:
        raise ValueError("correct and k cannot exceed total")
    if correct == 0:
        return 0.0
    if total - correct < k or correct / total == 1.0:
        return 1.0
    return -math.expm1(math.fsum(math.log1p(-correct / (total - i)) for i in range(k)))


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> list[float]:
    """Wilson score interval for a binomial success proportion.

    Returns [0, 1] when no trials were observed. Assumes independent trials;
    pooling different tasks is an approximation, not a population guarantee.
    """
    _integer(total, "total")
    _integer(correct, "correct")
    _confidence(confidence)
    if correct > total:
        raise ValueError("correct cannot exceed total")
    if total == 0:
        return [0.0, 1.0]
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [
        0.0 if correct == 0 else max(0.0, center - radius),
        1.0 if correct == total else min(1.0, center + radius),
    ]


def bootstrap_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> list[float]:
    """Seeded percentile bootstrap interval for an arithmetic mean.

    Requires at least two finite numeric observations. Resamples observations
    with replacement; constant data yields a point interval. This does not
    model unobserved variation or correct for multiple comparisons.
    """
    _confidence(confidence)
    _integer(resamples, "resamples", 2)
    data = list(values)
    if len(data) < 2:
        raise ValueError("bootstrap requires at least two observations")
    if any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
        for v in data
    ):
        raise ValueError("values must contain finite numbers")
    rng = random.Random(seed)
    n = len(data)
    means = sorted(
        math.fsum(value / n for value in rng.choices(data, k=n))
        for _ in range(resamples)
    )
    bounds = []
    for probability in ((1 - confidence) / 2, (1 + confidence) / 2):
        index = (resamples - 1) * probability
        lo = math.floor(index)
        hi = math.ceil(index)
        weight = index - lo
        bounds.append((1 - weight) * means[lo] + weight * means[hi])
    return bounds
