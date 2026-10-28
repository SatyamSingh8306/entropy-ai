import random
import pytest

from entropy import Suite, Dataset, Case, assert_stable, metric, AgentRun, Event


def test_monte_carlo_output_shape():
    def agent(inp):
        return "A" if random.random() < 0.7 else "B"

    ds = Dataset([Case(input="x", expected="A")])
    res = Suite(seed=1).run(agent, ds, trials=200)

    for key in ("success_rate", "behavioral_entropy", "trajectory_entropy",
                "behavioral_variance", "tool_reliability", "drift_score",
                "recovery_score", "emergent_behavior"):
        assert key in res, f"missing {key}"

    assert 0.0 <= res["success_rate"] <= 1.0
    assert 0.0 <= res["behavioral_entropy"] <= 1.0
    assert 0.0 <= res["drift_score"] <= 1.0
    assert isinstance(res["emergent_behavior"], list)


def test_deterministic_agent_has_zero_entropy():
    def agent(inp):
        return "A"

    ds = Dataset([Case(input="x", expected="A")])
    res = Suite(seed=1).run(agent, ds, trials=50)
    assert res["behavioral_entropy"] == 0.0
    assert res["success_rate"] == 1.0
    assert res["recovery_score"] == 1.0


def test_extractor_captures_trajectories():
    def agent(inp):
        return AgentRun(run_id="", input=inp, output="done",
                        events=[Event("action", "tool:search"),
                                Event("action", "tool:answer")])

    def extractor(run):
        return [e.name for e in run.events]

    ds = Dataset([Case(input="x", expected="done")])
    res = Suite(seed=1).run(agent, ds, trials=80, extractor=extractor)
    # every trial took the same tool path -> low entropy, full tool reliability
    assert res["tool_reliability"] == 1.0
    assert res["trajectory_entropy"] == 0.0


def test_assert_stable_raises_on_violation():
    res = {"success_rate": 0.3}
    try:
        assert_stable(res, success_rate__gt=0.9)
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_stable should have raised")


def test_custom_metric_plugin():
    @metric("my_dummy")
    class MyDummy:
        def compute(self, b):
            return 1.0

    from entropy.metrics import _REGISTRY
    assert "my_dummy" in _REGISTRY
    res = Suite(metrics=["my_dummy"]).run(lambda i: "x",
                                          Dataset([Case(input="x")]), trials=5)
    assert res["my_dummy"] == 1.0


@pytest.mark.parametrize("wrapped", [False, True])
def test_empty_exception_message_is_a_failure(wrapped):
    from entropy import evaluate, instrument

    def agent(inp):
        raise RuntimeError()

    def check(output):
        raise AssertionError("failed runs must not reach the output checker")

    results = evaluate(instrument(agent) if wrapped else agent, ["x"],
                       trials=2, check=check)
    assert results["success_rate"] == 0.0
    assert results["recovery_score"] == 0.0


@pytest.mark.parametrize("trials", [0, -1, 1.5, True, "2", None])
def test_invalid_trials_rejected_before_agent_runs(trials):
    from entropy import evaluate

    def agent(inp):
        raise AssertionError("invalid trials must not invoke the agent")

    with pytest.raises(ValueError, match="positive integer"):
        evaluate(agent, ["x"], trials=trials)


def test_recovered_error_event_can_still_succeed():
    run = AgentRun(run_id="", input="x", output="done",
                   events=[Event("error", "temporary_failure")])
    suite = Suite()
    results = suite.run(lambda inp: run, {"x": "done"}, trials=1)
    assert results["success_rate"] == 1.0
    assert results["recovery_score"] == 1.0
    assert suite.batches[0].successes == [True]
