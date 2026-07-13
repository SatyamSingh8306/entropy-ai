"""MCP adapter (spec §2)."""
from __future__ import annotations

import asyncio

from ..core.models import AgentRun, ActionEvent, ErrorEvent
from .base import Adapter, adapter


@adapter("mcp")
class MCPAdapter(Adapter):
    def match(self, agent):
        try:
            from mcp import ClientSession
        except Exception:
            return False
        return isinstance(agent, ClientSession)

    def wrap(self, agent, tool: str = "run"):
        def run(inp):
            try:
                result = asyncio.run(agent.call_tool(tool, {"input": inp}))
                out = getattr(result, "content", result)
            except Exception as e:
                return AgentRun(run_id="", input=inp, output=None,
                                events=[ErrorEvent("exception", {"error": str(e)})],
                                metadata={"error": str(e)})
            return AgentRun(run_id="", input=inp, output=out,
                            events=[ActionEvent("mcp.call_tool")])
        return run
