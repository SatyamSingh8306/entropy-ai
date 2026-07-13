"""AutoGen adapter (spec §2)."""
from __future__ import annotations

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("autogen")
class AutoGenAdapter(Adapter):
    def match(self, agent):
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager
        except Exception:
            return False
        return isinstance(agent, (ConversableAgent, GroupChat, GroupChatManager))

    def wrap(self, agent):
        def run(inp):
            try:
                if hasattr(agent, "run"):
                    res = agent.run()
                else:  # ConversableAgent: best-effort self chat
                    res = agent.initiate_chat(agent, message=inp, max_turns=2)
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            out = getattr(res, "summary", res)
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("autogen.chat")])
        return run
