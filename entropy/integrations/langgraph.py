"""LangGraph adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("langgraph")
class LangGraphAdapter(Adapter):
    def match(self, agent):
        try:
            from langgraph.graph.state import CompiledStateGraph
        except Exception:
            return False
        return isinstance(agent, CompiledStateGraph)

    def wrap(self, agent):
        def run(inp):
            try:
                state = agent.invoke({"input": inp})
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            out = state.get("output") if isinstance(state, dict) else state
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("langgraph.invoke")])
        return run
