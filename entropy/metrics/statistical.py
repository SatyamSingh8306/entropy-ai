"""Statistical metrics (spec §7 Statistical Metrics / §9)."""
from __future__ import annotations

import math
from .registry import (
    Metric, metric, default, mean, variance, bernoulli_entropy, norm_entropy,
)


@metric("success_rate")
class SuccessRate(Metric):
    """Fraction of trials that succeeded."""

    def compute(self, b):
        return mean([1.0 if s else 0.0 for s in b.successes])


@metric("reliability")
class ReliabilityScore(Metric):
    """Mean success across trials (spec §9 reliability)."""

    def compute(self, b):
        return mean([1.0 if s else 0.0 for s in b.successes])


@metric("variance")
class VarianceScore(Metric):
    """Variance of per-trial cost."""

    def compute(self, b):
        return variance(b.costs)


@metric("robustness")
class RobustnessScore(Metric):
    """Robustness = mean recovery rate across trials (spec §9 robustness)."""

    def compute(self, b):
        rec = [1.0 if r else 0.0 for r in b.recovered]
        return mean(rec)


@metric("confidence_interval")
class ConfidenceInterval(Metric):
    """95% CI for success_rate as [lo, hi] (Wald approx)."""

    def compute(self, b):
        n = len(b.successes)
        if n == 0:
            return [0.0, 0.0]
        p = mean([1.0 if s else 0.0 for s in b.successes])
        se = math.sqrt(p * (1 - p) / n)
        return [max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)]


@metric("entropy")
class EntropyScore(Metric):
    """Normalized entropy over primary actions (spec §9 entropy)."""

    def compute(self, b):
        return norm_entropy(b.actions)


@metric("stability_index")
class StabilityIndex(Metric):
    """SI = 1 - normalized_variance of the success outcome (spec §4)."""

    def compute(self, b):
        p = mean([1.0 if s else 0.0 for s in b.successes])
        return 1.0 - p * (1.0 - p)


@metric("reliability_score")
class ReliabilityScoreMetric(Metric):
    """R = success_rate x stability_index (spec §4 Statistical Metrics)."""

    def compute(self, b):
        p = mean([1.0 if s else 0.0 for s in b.successes])
        return p * (1.0 - p * (1.0 - p))


# Mark canonical Suite output members from this module.
for _n in ("success_rate", "reliability", "variance", "robustness",
           "confidence_interval", "entropy", "stability_index"):
    default(_n)
