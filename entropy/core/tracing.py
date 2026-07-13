"""Tracing: manual instrumentation, generic wrapping, and auto-routing."""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from .models import AgentRun, Event, Trace, new_id

_current: List["_Tracer"] = []


class _Tracer:
    """Captures events emitted via :func:`event` while active."""

    def __init__(self) -> None:
        self.events: List[Event] = []

    def event(self, type: str, name: str, **data: Any) -> None:
        self.events.append(Event(type=type, name=name, data=data))

    def __enter__(self) -> "_Tracer":
        _current.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if _current:
            _current.pop()


def trace() -> _Tracer:
    """Context manager that records manual instrumentation events."""
    return _Tracer()


def event(type: str, name: str, **data: Any) -> None:
    """Emit an event into the active trace (no-op if none active)."""
    if _current:
        _current[-1].event(type, name, **data)


def _resolve_adapter(agent: Any) -> Optional[Callable]:
    """Return an adapter's wrap fn for ``agent`` if one is registered."""
    try:
        from ..integrations.base import find_adapter
    except Exception:
        return None
    try:
        return find_adapter(agent)
    except Exception:
        return None


def instrument(agent: Callable) -> Callable:
    """Wrap an agent so calls return an :class:`AgentRun`.

    Routes to a registered framework adapter when the object's type matches,
    otherwise wraps generically (recording input/output as events).
    """

    adapter = _resolve_adapter(agent)
    if adapter is not None:
        return adapter(agent)

    def wrapped(inp: Any, **kw: Any) -> AgentRun:
        t = _Tracer()
        t.event("input", "input")
        try:
            out = agent(inp, **kw)
        except Exception as e:  # robustness: capture failure instead of raising
            t.event("error", "exception", error=str(e))
            return AgentRun(run_id=new_id(), input=inp, output=None,
                            events=t.events, metadata={"error": str(e)})
        t.event("output", "output")
        if isinstance(out, AgentRun):
            return out
        return AgentRun(run_id=new_id(), input=inp, output=out, events=t.events)

    wrapped.__entropy_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def current_trace() -> Optional[Trace]:
    """Build a Trace from events captured so far (best-effort)."""
    if not _current:
        return None
    events = _current[-1].events
    return Trace(runs=[AgentRun(run_id=new_id(), input=None, output=None, events=events)])
