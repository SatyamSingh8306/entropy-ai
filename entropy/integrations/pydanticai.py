"""PydanticAI adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("pydanticai")
class PydanticAIAdapter(Adapter):
    def match(self, agent):
        try:
            from pydantic_ai import Agent as PAIAgent
        except Exception:
            return False
        return isinstance(agent, PAIAgent)

    def wrap(self, agent):
        def run(inp):
            try:
                result = agent.run_sync(inp)
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            return AgentRun(run_id="", input=inp, output=result.output,
                            events=[ActionEvent("pydanticai.run")])
        return run
