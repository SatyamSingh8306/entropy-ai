"""Reasoning evaluation metrics (spec §7, Layer 1)."""
from __future__ import annotations

from .registry import Metric, metric, mean, norm_entropy, event_names, half_drift


@metric("plan_quality")
class PlanQualityMetric(Metric):
    """Fraction of trials that produced a reasoning/plan event."""

    def compute(self, b):
        has = [1.0 if (event_names(b, "reasoning")[i] or event_names(b, "action")[i]) else 0.0
               for i in range(len(b.events))]
        return mean(has)


@metric("plan_adherence")
class PlanAdherenceMetric(Metric):
    """Fraction of trials whose final action matches a stated plan prefix."""

    def compute(self, b):
        ok = 0
        for i, evs in enumerate(b.events):
            plans = [e.name for e in evs if e.type == "reasoning" and str(e.name).startswith("plan:")]
            if plans:
                final = b.actions[i]
                if final and final in plans[0]:
                    ok += 1
        n = len(b.events)
        return ok / n if n else 0.0


@metric("reasoning_consistency")
class ReasoningConsistencyMetric(Metric):
    """1 - normalized entropy of reasoning step names (lower spread = consistent)."""

    def compute(self, b):
        names = [n for trial in event_names(b, "reasoning") for n in trial]
        return 1.0 - norm_entropy(names)


@metric("planning_stability")
class PlanningStabilityMetric(Metric):
    """Stability of the plan across trials (1 - drift of reasoning labels)."""

    def compute(self, b):
        plans = [tuple(event_names(b, "reasoning")[i]) for i in range(len(b.events))]
        return 1.0 - half_drift(plans)


@metric("goal_decomposition")
class GoalDecompositionMetric(Metric):
    """Average number of distinct sub-steps (events) per trial, normalized to [0,1]."""

    def compute(self, b):
        lens = [len(set(e.name for e in trial)) for trial in b.events]
        if not lens:
            return 0.0
        return min(1.0, mean(lens) / 10.0)
