"""Entropy metrics package: registry + all spec metric categories."""
from __future__ import annotations

from .registry import (
    Metric, Batch, metric, default, get_metric, default_metrics,
    shannon, norm_entropy, mean, variance, _REGISTRY,
)
from . import layers

# Import every category so all metrics register on import.
from . import statistical  # noqa: E402,F401
from . import entropy_ as entropy_metrics  # noqa: E402,F401
from . import behavioral  # noqa: E402,F401
from . import reasoning  # noqa: E402,F401
from . import tool  # noqa: E402,F401
from . import execution  # noqa: E402,F401
from . import memory  # noqa: E402,F401
from . import multiagent  # noqa: E402,F401
from . import safety  # noqa: E402,F401
from . import drift  # noqa: E402,F401

__all__ = [
    "Metric", "Batch", "metric", "default", "get_metric", "default_metrics",
    "shannon", "norm_entropy", "mean", "variance", "_REGISTRY", "layers",
]
