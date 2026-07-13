import random

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
