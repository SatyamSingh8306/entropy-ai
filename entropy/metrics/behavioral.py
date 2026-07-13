"""Behavioral metrics — EntroPy's USP (spec §7 Behavioral Metrics)."""
from __future__ import annotations

from collections import Counter
from .registry import (
    Metric, metric, default, mean, norm_entropy, half_drift, avg_pairwise_lev,
)


@metric("behavioral_entropy")
class BehavioralEntropyMetric(Metric):
    """Normalized entropy of the primary action chosen per trial."""

    def compute(self, b):
        return norm_entropy(b.actions)


@metric("behavioral_variance")
class BehavioralVarianceMetric(Metric):
    """Variance of per-trial binary outcome (success=1, fail=0)."""

    def compute(self, b):
        xs = [1.0 if s else 0.0 for s in b.successes]
        return sum((x - mean(xs)) ** 2 for x in xs) / len(xs) if xs else 0.0


@metric("trajectory_diversity")
class TrajectoryDiversityMetric(Metric):
    """Mean normalized Levenshtein distance across trajectory pairs (spec §3.5)."""

    def compute(self, b):
        return avg_pairwise_lev(b.behaviors)


@metric("loop_detection")
class LoopDetectionMetric(Metric):
    """Fraction of trials whose trajectory repeats a state (e.g. A→B→A→B)."""

    def compute(self, b):
        loops = sum(1 for seq in b.behaviors if len(seq) != len(set(seq)))
        return loops / len(b.behaviors) if b.behaviors else 0.0


@metric("goal_stability")
class GoalStabilityMetric(Metric):
    """1 - drift; how stable the behavior is across the trial sequence."""

    def compute(self, b):
        return 1.0 - half_drift([tuple(x) for x in b.behaviors])


@metric("exploration_efficiency_metric")
class ExplorationEfficiencyMetric(Metric):
    """reward / entropy over actions."""

    def compute(self, b):
        ent = norm_entropy(b.actions)
        if ent < 1e-9:
            return 0.0
        return mean([1.0 if s else 0.0 for s in b.successes]) / ent


@metric("emergent_behavior")
class EmergentBehaviorMetric(Metric):
    """Behaviors appearing in <=5% of trials — rare, potentially emergent."""

    def compute(self, b):
        sigs = [tuple(x) for x in b.behaviors]
        c = Counter(sigs)
        total = len(sigs)
        thr = max(1, int(0.05 * total))
        return [list(k) for k, v in c.items() if 0 < v <= thr]


for _n in ("behavioral_entropy", "behavioral_variance", "trajectory_diversity",
           "goal_stability", "emergent_behavior"):
    default(_n)
