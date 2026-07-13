"""EntroPy — behavioral evaluation for non-deterministic AI agents.

Measure uncertainty, drift, stochastic behavior, and emergent failures via
Monte Carlo evaluation and entropy-based metrics.
"""
from __future__ import annotations

from .core import (
    AgentRun, Event, Trace, Node, Edge,
    ActionEvent, ReasoningEvent, ToolCallEvent, ObservationEvent,
    MemoryReadEvent, MemoryWriteEvent, ErrorEvent, StateTransitionEvent,
    instrument, trace, event, current_trace,
    Suite, evaluate, assert_stable,
)
from .metrics import metric, Metric, Batch, default_metrics, layers
from .datasets import (
    Dataset, Case, GoldenCase, Scenario, BehaviorCase, FailureCase, AdversarialCase,
)
from .simulation import (
    UserSimulator, ScriptedUserSimulator, RandomUserSimulator, LLMUserSimulator,
    Environment, StatefulEnv, GridWorld, AdversarialSimulator,
)
from .chaos import ChaosRunner, inject as chaos_inject, available as chaos_available
from .observability import Watcher, render_metric_heatmap
from .dashboard import build_dashboard, serve as serve_dashboard
from .reports import export as export_report
from .integrations import (
    adapter, find_adapter, list_adapters,
    from_langchain, from_langgraph, from_openai, from_crewai,
    from_pydanticai, from_autogen, from_google_adk, from_mcp, from_custom,
)
from .plugins import exporter, discover

__all__ = [
    "Suite", "Dataset", "Case", "GoldenCase", "Scenario", "BehaviorCase",
    "FailureCase", "AdversarialCase",
    "UserSimulator", "ScriptedUserSimulator", "RandomUserSimulator", "LLMUserSimulator",
    "Environment", "StatefulEnv", "GridWorld", "AdversarialSimulator",
    "ChaosRunner", "chaos_inject", "chaos_available",
    "Watcher", "render_metric_heatmap", "build_dashboard", "serve_dashboard",
    "export_report",
    "AgentRun", "Event", "Trace", "Node", "Edge",
    "ActionEvent", "ReasoningEvent", "ToolCallEvent", "ObservationEvent",
    "MemoryReadEvent", "MemoryWriteEvent", "ErrorEvent", "StateTransitionEvent",
    "instrument", "trace", "event", "current_trace",
    "metric", "Batch", "default_metrics", "layers", "evaluate",
    "adapter", "find_adapter", "list_adapters",
    "from_langchain", "from_langgraph", "from_openai", "from_crewai",
    "from_pydanticai", "from_autogen", "from_google_adk", "from_mcp", "from_custom",
    "exporter", "discover",
    "assert_stable",
]
