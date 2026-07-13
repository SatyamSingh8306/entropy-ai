"""Memory evaluation metrics (spec §7 Memory Metrics)."""
from __future__ import annotations

from .registry import Metric, metric, mean


@metric("memory_correctness")
class MemoryCorrectnessMetric(Metric):
    """Fraction of memory writes flagged correct (default correct)."""

    def compute(self, b):
        total = 0
        good = 0
        for trial in b.events:
            for e in trial:
                if e.type == "memory_write":
                    total += 1
                    if e.data.get("correct", True):
                        good += 1
        return good / total if total else 1.0


@metric("memory_pollution")
class MemoryPollutionMetric(Metric):
    """Fraction of memory writes flagged as polluting/stale."""

    def compute(self, b):
        total = 0
        bad = 0
        for trial in b.events:
            for e in trial:
                if e.type == "memory_write":
                    total += 1
                    if e.data.get("polluting") or e.data.get("stale"):
                        bad += 1
        return bad / total if total else 0.0


@metric("memory_recall")
class MemoryRecallMetric(Metric):
    """Fraction of trials with memory reads that succeeded (recall helped)."""

    def compute(self, b):
        idx = [i for i, t in enumerate(b.events) if any(e.type == "memory_read" for e in t)]
        if not idx:
            return 0.0
        return mean([1.0 if b.successes[i] else 0.0 for i in idx])


@metric("context_dependency")
class ContextDependencyMetric(Metric):
    """Rigidity: low behavioral entropy across a varied input set implies high
    context dependency. Reported per-batch as 1 - norm_entropy(actions)."""

    def compute(self, b):
        from .registry import norm_entropy
        return 1.0 - norm_entropy(b.actions)
