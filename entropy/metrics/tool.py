"""Tool evaluation metrics (spec §7, Layer 2)."""
from __future__ import annotations

from collections import Counter
from .registry import (
    Metric, metric, default, mean, norm_entropy, flat_tool_calls,
)


@metric("tool_correctness")
class ToolCorrectnessMetric(Metric):
    """Fraction of tool-using trials whose final outcome succeeded."""

    def compute(self, b):
        idx = [i for i, t in enumerate(b.tool_calls) if t]
        if not idx:
            return 0.0
        return mean([1.0 if b.successes[i] else 0.0 for i in idx])


@metric("argument_correctness")
class ArgumentCorrectnessMetric(Metric):
    """Fraction of tool calls carrying non-empty arguments."""

    def compute(self, b):
        total = 0
        good = 0
        for trial in b.events:
            for e in trial:
                if e.type == "tool":
                    total += 1
                    if e.data:
                        good += 1
        return good / total if total else 0.0


@metric("tool_reliability")
class ToolReliabilityMetric(Metric):
    """Reliability of tool usage: success rate over tool-using trials."""

    def compute(self, b):
        idx = [i for i, t in enumerate(b.tool_calls) if t]
        if not idx:
            return mean([1.0 if s else 0.0 for s in b.successes])
        return mean([1.0 if b.successes[i] else 0.0 for i in idx])


@metric("tool_selection")
class ToolSelectionMetric(Metric):
    """Normalized entropy of tool selection (diversity of tools chosen)."""

    def compute(self, b):
        return norm_entropy(flat_tool_calls(b))


@metric("tool_thrashing")
class ToolThrashingMetric(Metric):
    """Fraction of trials that called the same tool repeatedly (thrashing)."""

    def compute(self, b):
        thrash = 0
        for t in b.tool_calls:
            c = Counter(t)
            if c and c.most_common(1)[0][1] > 1:
                thrash += 1
        return thrash / len(b.tool_calls) if b.tool_calls else 0.0


@metric("hallucinated_tool")
class HallucinatedToolMetric(Metric):
    """Fraction of tool calls flagged as hallucinated (unknown tool)."""

    def compute(self, b):
        total = 0
        bad = 0
        for trial in b.events:
            for e in trial:
                if e.type == "tool":
                    total += 1
                    if e.data.get("hallucinated") or str(e.name).startswith("unknown:"):
                        bad += 1
        return bad / total if total else 0.0


default("tool_reliability")
