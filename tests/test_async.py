"""Async evaluation uses the same evidence and metrics as synchronous runs."""
import asyncio

import pytest
from entropy import AgentRun, Case, Dataset, Event, Suite, evaluate_async


@pytest.mark.parametrize("kind", ["plain", "run", "dict"])
def test_async_matches_sync_metrics_and_order(kind):
    calls = []

    def agent(inp):
        if kind == "plain":
            return inp.upper()
        run = AgentRun("id", inp, inp.upper(), events=[Event("tool", "lookup")],
                       cost=0.2, metadata={"latency": 0.5})
        return vars(run) if kind == "dict" else run

    async def async_agent(inp):
        await asyncio.sleep(0)
        calls.append(inp)
        return agent(inp)

    dataset = Dataset([Case("a", expected="A"), Case("b", expected="B")])
    sync_suite, async_suite = Suite(), Suite()
    expected = sync_suite.run(agent, dataset, trials=3,
                              extractor=lambda run: [str(run.output)])
    actual = asyncio.run(async_suite.run_async(
        async_agent, dataset, trials=3, extractor=lambda run: [str(run.output)]))
    assert actual == expected
    assert async_suite.batches == sync_suite.batches
    assert calls == ["a"] * 3 + ["b"] * 3


def test_async_helper_supports_expected_checks_and_sync_callables():
    async def agent(inp):
        return inp.upper()

    async def main():
        result = await evaluate_async(agent, ["a"], expected=["A"], trials=2)
        assert result["success_rate"] == 1.0
        result = await evaluate_async(agent, {"a": "A"}, trials=2)
        assert result["success_rate"] == 1.0
        result = await evaluate_async(str.upper, ["a"], check=lambda x: x == "A",
                                      trials=2, metrics=["success_rate"])
        assert result == {"success_rate": 1.0}

    asyncio.run(main())


def test_async_failures_are_recorded_and_trials_continue():
    calls = []

    async def agent(inp):
        calls.append(inp)
        if len(calls) == 1:
            raise RuntimeError()
        return "ok"

    results = asyncio.run(evaluate_async(agent, {"x": "ok"}, trials=2))
    assert results["success_rate"] == 0.5
    assert len(calls) == 2


def test_async_cancellation_propagates():
    calls = []

    async def agent(inp):
        calls.append(inp)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(evaluate_async(agent, ["x"], trials=3))
    assert calls == ["x"]


@pytest.mark.parametrize("trials", [0, -1, 1.5, True, "2", None])
def test_async_invalid_trials_never_call_agent(trials):
    calls = []

    async def agent(inp):
        calls.append(inp)

    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(evaluate_async(agent, ["x"], trials=trials))
    assert calls == []


def test_callable_returning_future_is_awaited():
    def agent(inp):
        future = asyncio.get_running_loop().create_future()
        future.set_result(inp.upper())
        return future

    results = asyncio.run(evaluate_async(agent, {"a": "A"}, trials=1))
    assert results["success_rate"] == 1.0


def test_sync_exception_is_recorded_by_async_evaluator():
    def agent(inp):
        raise ValueError("broken")

    results = asyncio.run(evaluate_async(agent, ["a"], trials=1))
    assert results["success_rate"] == 0.0
