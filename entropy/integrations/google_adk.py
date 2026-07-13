"""Google ADK adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("google-adk")
class GoogleADKAdapter(Adapter):
    def match(self, agent):
        try:
            from google.adk.agents import LlmAgent
        except Exception:
            return False
        return isinstance(agent, LlmAgent)

    def wrap(self, agent):
        def run(inp):
            try:
                from google.adk.runners import InMemoryRunner
                runner = InMemoryRunner(agent=agent, app_name="entropy")
                events = list(runner.run(user_id="entropy", session_id="s",
                                          new_message=inp))
                out = getattr(events[-1], "content", None)
                out = getattr(out, "parts", [out])[0] if out is not None else events
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("adk.run")])
        return run
