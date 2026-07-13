"""Real LangChain adapter integration test (skips if LangChain absent)."""
import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda

from entropy import from_langchain, find_adapter, AgentRun


def _make_chain():
    # A LangChain Runnable that echoes its input with a suffix.
    return RunnableLambda(
        lambda x: {"output": (x["input"] if isinstance(x, dict) else x) + "!"}
    )


def test_langchain_adapter_wraps_and_runs():
    chain = _make_chain()
    wrapped = from_langchain(chain)
    run = wrapped("hi")
    assert isinstance(run, AgentRun)
    assert run.output == "hi!"


def test_find_adapter_routes_to_langchain():
    chain = _make_chain()
    wrap = find_adapter(chain)
    assert wrap is not None
    assert wrap("x").output == "x!"
