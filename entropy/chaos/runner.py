"""Chaos runner (spec §10): run the suite under each fault and compare."""
from __future__ import annotations

import random
from typing import Callable, Iterable, List

from ..core.suite import Suite
from .faults import FAULTS, inject


class ChaosRunner:
    """Run an agent under baseline and each requested fault."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def run(self, agent: Callable, dataset, faults: List[str] | str,
            trials: int = 50) -> dict:
        if isinstance(faults, str):
            faults = [faults]
        unknown = [f for f in faults if f not in FAULTS]
        if unknown:
            raise ValueError(f"unknown faults: {unknown}")
        rng = random.Random(self.seed)
        baseline = Suite(seed=self.seed).run(agent, dataset, trials=trials)
        out = {"baseline": baseline, "faults": {}}
        for f in faults:
            wrapped = inject(agent, f, p=1.0, seed=self.seed)
            out["faults"][f] = Suite(seed=self.seed).run(wrapped, dataset, trials=trials)
        return out

    def summary(self, result: dict) -> str:
        lines = [f"{'fault':16} {'success':>9} {'fail':>9} {'drift':>9}"]
        b = result["baseline"]
        lines.append(f"{'(baseline)':16} {b['success_rate']:9.4f} "
                     f"{1 - b['success_rate']:9.4f} {b['drift_score']:9.4f}")
        for f, r in result["faults"].items():
            lines.append(f"{f:16} {r['success_rate']:9.4f} "
                         f"{1 - r['success_rate']:9.4f} {r['drift_score']:9.4f}")
        return "\n".join(lines)
