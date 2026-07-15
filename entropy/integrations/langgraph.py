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
            from langchain_core.messages import HumanMessage

            # LangGraph graphs accept either {"input": ...} or
            # {"messages": [HumanMessage(...)]} (e.g. langchain.agents.create_agent).
            try:
                try:
                    state = agent.invoke({"input": inp})
                except Exception:
                    state = agent.invoke({"messages": [HumanMessage(content=inp)]})
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            out = _extract_output(state)
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("langgraph.invoke")])
        return run


def _extract_output(state):
    """Pull the agent's final answer out of a LangGraph state.

    Compiled graphs (incl. langchain.agents.create_agent) expose the result
    either as ``{"output": ...}`` or as ``{"messages": [...]}`` where the last
    AIMessage holds the answer.
    """
    if not isinstance(state, dict):
        return state
    if "output" in state:
        return state["output"]
    messages = state.get("messages")
    if isinstance(messages, (list, tuple)):
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if type(msg).__name__ == "AIMessage" and content:
                return content
        # fall back to the last message content
        if messages:
            return getattr(messages[-1], "content", messages[-1])
    return None
