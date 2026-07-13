"""OpenAI Agents SDK adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("openai")
class OpenAIAdapter(Adapter):
    def match(self, agent):
        try:
            from agents import Agent as OAIAgent
        except Exception:
            return False
        return isinstance(agent, OAIAgent)

    def wrap(self, agent):
        def run(inp):
            try:
                from agents import Runner
                result = Runner.run_sync(agent, inp)
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            return AgentRun(run_id="", input=inp, output=result.final_output,
                            events=[ActionEvent("openai.run")])
        return run
