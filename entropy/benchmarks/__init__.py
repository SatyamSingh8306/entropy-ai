"""Benchmarks package."""
from __future__ import annotations

from .bench import (
    load_ref, load_dataset, benchmark_variants, benchmark_refs, to_markdown,
)

__all__ = [
    "load_ref", "load_dataset", "benchmark_variants", "benchmark_refs", "to_markdown",
]
