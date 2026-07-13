"""Observability (spec §15): live watching with anomaly detection + clusters."""
from __future__ import annotations

import statistics
import time
from collections import Counter
from typing import Callable, Dict, List, Optional

from ..core.suite import Suite
from ..datasets.loaders import Dataset


class Watcher:
    """Periodically re-runs an agent and tracks drift/entropy over time."""

    def __init__(self, agent: Callable, dataset, trials: int = 20, seed: int = 0):
        self.agent = agent
        self.dataset = dataset
        self.trials = trials
        self.seed = seed
        self.history: List[Dict] = []
        self.last_batches = None

    def round(self, i: int) -> Dict:
        suite = Suite(seed=self.seed + i)
        res = suite.run(self.agent, self.dataset, trials=self.trials)
        self.last_batches = suite.batches
        self.history.append(res)
        return res

    def series(self, metric: str) -> List[float]:
        return [r.get(metric, 0.0) for r in self.history]

    def anomalies(self, metric: str = "drift_score", z: float = 2.0) -> List[bool]:
        s = self.series(metric)
        if len(s) < 3:
            return [False] * len(s)
        mean = statistics.mean(s)
        std = statistics.pstdev(s) or 1e-9
        return [abs(v - mean) / std > z for v in s]

    def failure_clusters(self, top: int = 5) -> List:
        if not self.last_batches:
            return []
        failed = []
        for b in self.last_batches:
            for i, ok in enumerate(b.successes):
                if not ok:
                    failed.append(str(b.outputs[i]))
        return Counter(failed).most_common(top)

    def watch(self, rounds: int = 10, interval: float = 0.0,
              render=None) -> List[Dict]:
        for i in range(rounds):
            res = self.round(i)
            anom = self.anomalies()[-1] if self.history else False
            if render is None:
                flag = "  <-- ANOMALY" if anom else ""
                print(f"round {i:3d} | success={res['success_rate']:.3f} "
                      f"drift={res['drift_score']:.3f} "
                      f"entropy={res['behavioral_entropy']:.3f}{flag}")
            else:
                render(self, i, res, anom)
            if interval:
                time.sleep(interval)
        return self.history


def render_metric_heatmap(history: List[Dict], path: str,
                          metrics: Optional[List[str]] = None) -> str:
    """Heatmap of min-max-normalized metric values across watched rounds."""
    import io
    import pathlib

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = [k for k in history[0] if isinstance(history[0][k], (int, float))]
    cols = [[r.get(m, 0.0) for r in history] for m in metrics]
    rows = []
    for ri in range(len(history)):
        row = []
        for ci, m in enumerate(metrics):
            vals = cols[ci]
            lo, hi = min(vals), max(vals)
            row.append(0.0 if hi == lo else (vals[ri] - lo) / (hi - lo))
        rows.append(row)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(metrics)), max(3, 0.3 * len(rows))))
    ax.imshow(rows, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"r{i}" for i in range(len(rows))])
    ax.set_title("Watched metrics over rounds")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    p = str(path)
    pathlib.Path(p).write_bytes(buf.getvalue())
    return p
