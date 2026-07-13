"""Entropy-based metrics (spec §8)."""
from __future__ import annotations

from collections import Counter

from .registry import (
    Metric, metric, default, mean, shannon, norm_entropy, bernoulli_entropy,
    flat_tool_calls, half_drift,
)


@metric("action_entropy")
class ActionEntropy(Metric):
    """H(actions): randomness of the atomic actions taken."""

    def compute(self, b):
        flat = [a for trial in b.behaviors for a in trial]
        if not flat:
            return 0.0
        return norm_entropy(flat)


@metric("trajectory_entropy")
class TrajectoryEntropy(Metric):
    """H(trajectory): variability of full execution trajectories."""

    def compute(self, b):
        return norm_entropy([tuple(x) for x in b.behaviors])


@metric("tool_entropy")
class ToolEntropy(Metric):
    """H(tool_usage): diversity of tool selection."""

    def compute(self, b):
        return norm_entropy(flat_tool_calls(b))


@metric("robustness_entropy")
class RobustnessEntropy(Metric):
    """H(failure_distribution): entropy over success/fail/error classes."""

    def compute(self, b):
        labels = []
        for i in range(len(b.successes)):
            if b.errored[i]:
                labels.append("error")
            elif b.successes[i]:
                labels.append("success")
            else:
                labels.append("fail")
        return norm_entropy(labels)


@metric("information_gain")
class InformationGain(Metric):
    """Reduction in action uncertainty given success (IG(state))."""

    def compute(self, b):
        if not b.actions:
            return 0.0
        h_all = shannon(list(Counter(b.actions).values()))
        succ = [a for i, a in enumerate(b.actions) if b.successes[i]]
        if not succ:
            return 0.0
        h_succ = shannon(list(Counter(succ).values()))
        return max(0.0, h_all - h_succ)


@metric("exploration_efficiency")
class ExplorationEfficiency(Metric):
    """reward / entropy — how much success per unit of behavioral entropy."""

    def compute(self, b):
        ent = norm_entropy(b.actions)
        if ent < 1e-9:
            return 0.0
        return mean([1.0 if s else 0.0 for s in b.successes]) / ent


default("trajectory_entropy")
default("action_entropy")
default("tool_entropy")
default("exploration_efficiency")
