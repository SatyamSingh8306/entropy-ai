"""Drift evaluation metrics (spec §7 Drift Metrics / §6 Layer 6)."""
from __future__ import annotations

from collections import Counter
from .registry import Metric, metric, default, mean, half_drift, norm_entropy, js_divergence


@metric("drift_score")
class DriftMetric(Metric):
    """Jensen-Shannon drift of the behavior distribution (first vs second half)."""

    def compute(self, b):
        n = len(b.behaviors)
        if n < 4:
            return 0.0
        half = n // 2
        c1 = list(Counter(tuple(x) for x in b.behaviors[:half]).values())
        c2 = list(Counter(tuple(x) for x in b.behaviors[half:]).values())
        return js_divergence(c1, c2)


@metric("trajectory_drift")
class TrajectoryDriftMetric(Metric):
    """Drift over full trajectory signatures specifically."""

    def compute(self, b):
        return half_drift([tuple(x) for x in b.behaviors])


@metric("tool_usage_drift")
class ToolUsageDriftMetric(Metric):
    """Drift of the tool-usage distribution across the trial sequence."""

    def compute(self, b):
        seq = [tuple(t) for t in b.tool_calls]
        return half_drift(seq)


@metric("cost_drift")
class CostDriftMetric(Metric):
    """Absolute drift in mean cost between first and second half, normalized."""

    def compute(self, b):
        n = len(b.costs)
        if n < 4 or mean(b.costs) == 0:
            return 0.0
        half = n // 2
        d = abs(mean(b.costs[:half]) - mean(b.costs[half:])) / mean(b.costs)
        return min(1.0, d)


@metric("behavior_fingerprint")
class BehaviorFingerprintMetric(Metric):
    """Stable signature of the behavior distribution (top behaviors + counts)."""

    def compute(self, b):
        c = Counter(tuple(x) for x in b.behaviors)
        return [list(k) for k, _ in c.most_common(5)]


default("drift_score")
