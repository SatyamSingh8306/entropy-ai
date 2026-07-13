"""LangChain adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("langchain")
class LangChainAdapter(Adapter):
    def match(self, agent):
        # LangChain v1: the unit is a langchain_core Runnable.
        try:
            from langchain_core.runnables import Runnable
            return isinstance(agent, Runnable)
        except Exception:
            pass
        # Legacy (pre-v1) layout.
        try:
            from langchain.agents import AgentExecutor
            from langchain.chains.base import Chain
        except Exception:
            return False
        return isinstance(agent, (AgentExecutor, Chain))

    def wrap(self, agent):
        def run(inp):
            try:
                try:
                    out = agent.invoke({"input": inp})
                except Exception:
                    out = agent.invoke(inp)
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            text = out.get("output", out) if isinstance(out, dict) else out
            return AgentRun(run_id="", input=inp, output=text,
                            events=[ActionEvent("langchain.invoke")])
        return run
