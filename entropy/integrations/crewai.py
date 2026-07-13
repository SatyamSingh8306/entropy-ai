"""CrewAI adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("crewai")
class CrewAIAdapter(Adapter):
    def match(self, agent):
        try:
            from crewai import Crew, Agent as CrewAgent, Task
        except Exception:
            return False
        return isinstance(agent, (Crew, CrewAgent, Task))

    def wrap(self, agent):
        def run(inp):
            try:
                out = agent.kickoff(inputs={"input": inp})
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("crewai.kickoff")])
        return run
