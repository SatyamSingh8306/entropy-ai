"""Benchmarking (spec §14): compare models/prompts/frameworks/agents/versions."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple

from ..core.suite import Suite
from ..datasets.loaders import Dataset


def load_ref(ref: str):
    """Load ``module:attr`` (agent/function) or ``module:attr`` callable."""
    mod_name, attr = ref.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def load_dataset(path: str) -> Dataset:
    p = path.lower()
    if p.endswith(".jsonl"):
        return Dataset.from_jsonl(path)
    if p.endswith((".yaml", ".yml")):
        return Dataset.from_yaml(path)
    if p.endswith(".csv"):
        return Dataset.from_csv(path)
    return Dataset.from_json(path)


def benchmark_variants(
    variants: List[Tuple[str, Callable]], dataset, trials: int = 100, seed: int = 0,
) -> List[Tuple[str, dict]]:
    """Run the suite for each (name, agent) variant and return rows."""
    rows = []
    for name, agent in variants:
        res = Suite(seed=seed).run(agent, dataset, trials=trials)
        rows.append((name, res))
    return rows


def benchmark_refs(
    refs: List[str], dataset_path: str, trials: int = 100, seed: int = 0,
) -> List[Tuple[str, dict]]:
    ds = load_dataset(dataset_path)
    variants = [(r, load_ref(r)) for r in refs]
    return benchmark_variants(variants, ds, trials=trials, seed=seed)


def to_markdown(rows: List[Tuple[str, dict]], metrics: List[str] | None = None) -> str:
    if not rows:
        return "(no results)"
    if metrics is None:
        metrics = [k for k, v in rows[0][1].items() if isinstance(v, (int, float))]
    header = ["variant"] + metrics
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for name, res in rows:
        cells = [name] + [f"{res.get(m, 0):.4f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
