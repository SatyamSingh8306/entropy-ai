"""Plugin system (spec §18): @exporter + entry-point discovery of third-party plugins."""
from __future__ import annotations

from typing import Callable, Dict, List


# --- exporter registry (metric/adapter hooks already live in their modules) -
_EXPORTERS: Dict[str, Callable] = {}


def exporter(name: str | None = None) -> Callable[[Callable], Callable]:
    """Register a report exporter: ``fn(results, path) -> str``."""

    def deco(fn: Callable) -> Callable:
        _EXPORTERS[name or fn.__name__] = fn
        return fn

    return deco


def list_exporters() -> List[str]:
    return list(_EXPORTERS)


def register_exporter(name: str, fn: Callable) -> None:
    _EXPORTERS[name] = fn


# --- third-party plugin discovery (setuptools entry point group) ----------
def discover(group: str = "entropy.plugins") -> List[str]:
    """Import all modules registered under the ``entropy.plugins`` entry point.

    Each plugin module typically uses ``@entropy.metric`` / ``@entropy.adapter``
    / ``@entropy.exporter`` to register itself.
    """
    loaded: List[str] = []
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover
        return loaded
    try:
        eps = entry_points(group=group)
    except TypeError:  # older importlib.metadata API
        eps = entry_points().get(group, [])
    for ep in eps:
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception:
            continue
    return loaded
