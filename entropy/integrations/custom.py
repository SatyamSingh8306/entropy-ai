"""Custom / framework-agnostic agent adapter (spec §2). Fallback wrapper."""
from __future__ import annotations

from ..core.models import AgentRun, Event, new_id
from .base import Adapter, adapter


@adapter("custom")
class CustomAdapter(Adapter):
    """Matches any callable agent as a generic fallback (registered last)."""

    def match(self, agent):
        return callable(agent)

    def wrap(self, agent):
        def run(inp, **kw):
            try:
                out = agent(inp, **kw)
            except Exception as e:
                return AgentRun(run_id=new_id(), input=inp, output=None,
                                events=[Event("error", "exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            if isinstance(out, AgentRun):
                return out
            return AgentRun(run_id=new_id(), input=inp, output=out,
                            events=[Event("action", "custom.run")])

        return run
