"""Core data models for EntroPy: events, runs, traces, graphs."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """A single recorded event in an agent run."""

    type: str
    name: str
    data: Dict[str, Any] = field(default_factory=dict)


# --- typed event subclasses (spec §4) --------------------------------------
class ActionEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="action", name=name, data=data or {})


class ReasoningEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="reasoning", name=name, data=data or {})


class ToolCallEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="tool", name=name, data=data or {})


class ObservationEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="observation", name=name, data=data or {})


class MemoryReadEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="memory_read", name=name, data=data or {})


class MemoryWriteEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="memory_write", name=name, data=data or {})


class ErrorEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="error", name=name, data=data or {})


class StateTransitionEvent(Event):
    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__(type="state", name=name, data=data or {})


@dataclass
class AgentRun:
    """One execution of an agent on a single input."""

    run_id: str
    input: Any
    output: Any
    events: List[Event] = field(default_factory=list)
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    id: str
    label: str
    kind: str = "step"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A collection of runs plus an optional graph (nodes/edges)."""

    runs: List[AgentRun] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- exporters (delegate to entropy.core.exporters) -----------------
    def json(self) -> str:
        from .exporters import to_json
        return to_json(self)

    def df(self):
        from .exporters import to_df
        return to_df(self)

    def graph(self) -> dict:
        from .exporters import to_graph
        return to_graph(self)

    def otel(self) -> dict:
        from .exporters import to_otel
        return to_otel(self)

    def zip(self, path: str) -> str:
        from .exporters import to_zip
        return to_zip(self, path)

    def add_run(self, run: AgentRun) -> None:
        self.runs.append(run)


def new_id() -> str:
    return uuid.uuid4().hex
