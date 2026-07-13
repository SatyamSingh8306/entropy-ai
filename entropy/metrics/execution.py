"""Execution evaluation metrics (spec §7, Layer 3)."""
from __future__ import annotations

import statistics

from .registry import Metric, metric, default, mean


@metric("task_completion")
class TaskCompletionMetric(Metric):
    """Fraction of trials that completed successfully."""

    def compute(self, b):
        return mean([1.0 if s else 0.0 for s in b.successes])


@metric("step_efficiency")
class StepEfficiencyMetric(Metric):
    """Mean efficiency 1/(1+events) — fewer steps = more efficient."""

    def compute(self, b):
        eff = [1.0 / (1.0 + len(trial)) for trial in b.events]
        return mean(eff)


@metric("latency")
class LatencyMetric(Metric):
    """Mean latency in seconds (per-trial metadata, else cost as proxy)."""

    def compute(self, b):
        lat = [float(m) for m in b.metadata.get("_latencies", [])]
        if lat:
            return mean(lat)
        return mean([float(c) for c in b.costs]) if b.costs else 0.0


@metric("cost")
class CostMetric(Metric):
    """Mean cost per trial."""

    def compute(self, b):
        return mean([float(c) for c in b.costs])


@metric("retry_rate")
class RetryMetric(Metric):
    """Fraction of trials that contained a retry event."""

    def compute(self, b):
        retried = [1.0 if any(e.type == "retry" for e in trial) else 0.0
                   for trial in b.events]
        return mean(retried)


@metric("failure_rate")
class FailureRateMetric(Metric):
    """1 - success_rate."""

    def compute(self, b):
        return 1.0 - mean([1.0 if s else 0.0 for s in b.successes])


@metric("recovery_score")
class RecoveryMetric(Metric):
    """Fraction of errored trials that still succeeded."""

    def compute(self, b):
        err = [i for i, e in enumerate(b.errored) if e]
        if not err:
            return 1.0
        rec = sum(1 for i in err if b.successes[i])
        return rec / len(err)


@metric("cost_stability")
class CostStabilityMetric(Metric):
    """1 - coefficient of variation of per-trial cost (0=unstable, 1=stable)."""

    def compute(self, b):
        cs = [float(c) for c in b.costs]
        if not cs:
            return 1.0
        m = mean(cs)
        if m == 0:
            return 1.0
        sd = statistics.pstdev(cs) if len(cs) > 1 else 0.0
        return max(0.0, 1.0 - min(1.0, sd / m))


default("recovery_score")
