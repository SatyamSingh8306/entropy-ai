"""Adapter base: protocol + registry for framework integrations (spec §2)."""
from __future__ import annotations

from typing import Any, Callable, List, Optional


class Adapter:
    """Maps a framework-specific agent to an EntroPy-instrumentable callable."""

    name: str = "base"

    def match(self, agent: Any) -> bool:  # pragma: no cover - overridden
        return False

    def wrap(self, agent: Any) -> Callable:
        raise NotImplementedError


_REGISTRY: List[Adapter] = []


def adapter(name: str) -> Callable[[type], type]:
    """Register an :class:`Adapter` subclass (plugin system: ``@adapter``)."""

    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY.append(cls())
        return cls

    return deco


def register_adapter(inst: Adapter) -> None:
    _REGISTRY.append(inst)


def find_adapter(agent: Any) -> Optional[Callable]:
    """Return a wrap fn for ``agent`` if a registered adapter matches."""
    for a in _REGISTRY:
        try:
            if a.match(agent):
                return a.wrap(agent)
        except Exception:
            continue
    return None


def list_adapters() -> List[str]:
    return [a.name for a in _REGISTRY]
