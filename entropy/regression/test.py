"""Regression testing (spec §13): baseline comparison + assert_stable."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from ..core.suite import Suite, assert_stable
from ..datasets import Dataset


def run_regression(
    agent, dataset, trials: int = 100, baseline_path: str = "baseline.json",
    update: bool = False, tolerances: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the suite and compare against a stored baseline.

    Returns a dict with ``results`` and, when not updating, ``violations``.
    """
    results = Suite(seed=seed).run(agent, dataset, trials=trials)
    path = pathlib.Path(baseline_path)

    if update or not path.exists():
        path.write_text(json.dumps(results, indent=2, default=str))
        return {"updated": True, "results": results}

    base = json.loads(path.read_text())
    tol = tolerances or {}
    violations: List[Dict[str, float]] = []
    for k, v in results.items():
        if isinstance(v, (int, float)) and k in base and isinstance(base[k], (int, float)):
            allowed = tol.get(k, 0.05)
            if abs(v - base[k]) > allowed:
                violations.append({"metric": k, "baseline": base[k], "current": v})
    return {"updated": False, "results": results, "baseline": base,
            "violations": violations}


def assert_stable_results(results: dict, **thresholds) -> bool:
    """Thin wrapper over suite.assert_stable for the regression CLI."""
    return assert_stable(results, **thresholds)
