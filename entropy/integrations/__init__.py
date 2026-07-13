"""Framework adapters (spec §2). Each adapter imports its framework lazily."""
from __future__ import annotations

from .base import Adapter, adapter, find_adapter, list_adapters, register_adapter
from . import wire

__all__ = [
    "Adapter", "adapter", "find_adapter", "list_adapters", "register_adapter",
    "from_langchain", "from_langgraph", "from_openai", "from_crewai",
    "from_pydanticai", "from_autogen", "from_google_adk", "from_mcp", "from_custom",
]

# Re-export the from_* helpers for `from entropy.integrations import from_x`.
from .wire import (  # noqa: E402,F401
    from_langchain, from_langgraph, from_openai, from_crewai,
    from_pydanticai, from_autogen, from_google_adk, from_mcp, from_custom,
)
