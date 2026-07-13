"""Multi-agent evaluation metrics (spec §7 Multi-Agent Metrics)."""
from __future__ import annotations

from .registry import Metric, metric, mean, norm_entropy


@metric("coordination")
class CoordinationMetric(Metric):
    """Success rate on trials that involved a handoff/delegate event."""

    def compute(self, b):
        idx = [i for i, t in enumerate(b.events)
               if any(e.type in ("handoff", "delegate") for e in t)]
        if not idx:
            return 0.0
        return mean([1.0 if b.successes[i] else 0.0 for i in idx])


@metric("communication")
class CommunicationMetric(Metric):
    """Mean number of message events per trial."""

    def compute(self, b):
        msgs = [sum(1 for e in t if e.type == "message") for t in b.events]
        return mean([float(m) for m in msgs])


@metric("delegation")
class DelegationMetric(Metric):
    """Fraction of trials that delegated work to a sub-agent."""

    def compute(self, b):
        delg = [1.0 if any(e.type == "delegate" for e in t) else 0.0 for t in b.events]
        return mean(delg)


@metric("consensus")
class ConsensusMetric(Metric):
    """1 - normalized entropy of outcomes: consistent outcomes = consensus."""

    def compute(self, b):
        return 1.0 - norm_entropy(b.actions)


@metric("deadlock")
class DeadlockMetric(Metric):
    """Fraction of trials that hit a deadlock/timeout event."""

    def compute(self, b):
        d = [1.0 if any(e.type in ("deadlock", "timeout") for e in t) else 0.0
             for t in b.events]
        return mean(d)
