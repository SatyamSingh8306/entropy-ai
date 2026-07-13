"""Wire framework adapters and expose ``from_*`` helpers (spec §2)."""
from __future__ import annotations

# Importing each adapter module registers it via @adapter (lazy framework import).
from .langchain import LangChainAdapter
from .langgraph import LangGraphAdapter
from .openai import OpenAIAdapter
from .crewai import CrewAIAdapter
from .pydanticai import PydanticAIAdapter
from .autogen import AutoGenAdapter
from .google_adk import GoogleADKAdapter
from .mcp import MCPAdapter
from .custom import CustomAdapter


def from_langchain(agent):
    return LangChainAdapter().wrap(agent)


def from_langgraph(agent):
    return LangGraphAdapter().wrap(agent)


def from_openai(agent):
    return OpenAIAdapter().wrap(agent)


def from_crewai(agent):
    return CrewAIAdapter().wrap(agent)


def from_pydanticai(agent):
    return PydanticAIAdapter().wrap(agent)


def from_autogen(agent):
    return AutoGenAdapter().wrap(agent)


def from_google_adk(agent):
    return GoogleADKAdapter().wrap(agent)


def from_mcp(agent, tool: str = "run"):
    return MCPAdapter().wrap(agent, tool)


def from_custom(agent):
    return CustomAdapter().wrap(agent)
