import random

import pytest

from entropy import (
    Suite, Dataset, Case, AgentRun, Event, Trace,
    ActionEvent, ToolCallEvent, ErrorEvent, assert_stable, metric,
)
from entropy.metrics import default_metrics, _REGISTRY, layers


# ---------------------------------------------------------------- models
def test_event_subclasses():
    e = ActionEvent("move")
    assert e.type == "action" and e.name == "move"
    assert isinstance(e, Event)


def test_trace_exporters():
    t = Trace(runs=[AgentRun(run_id="r1", input="x", output="y",
                             events=[ActionEvent("a"), ToolCallEvent("t")])])
    j = t.json()
    assert "runs" in j and "r1" in j
    g = t.graph()
    assert "nodes" in g and "edges" in g
    ot = t.otel()
    assert "resourceSpans" in ot


def test_trace_df_needs_pandas():
    pd = pytest.importorskip("pandas")
    t = Trace(runs=[AgentRun(run_id="r1", input="x", output="y",
                             events=[ActionEvent("a")])])
    df = t.df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


# ---------------------------------------------------------------- metrics
def test_registry_has_all_categories():
    expected = {
        "success_rate", "reliability", "variance", "robustness",
        "confidence_interval", "entropy", "stability_index",
        "action_entropy", "trajectory_entropy", "tool_entropy",
        "robustness_entropy", "information_gain", "exploration_efficiency",
        "behavioral_entropy", "behavioral_variance", "trajectory_diversity",
        "goal_stability", "emergent_behavior",
        "plan_quality", "plan_adherence", "reasoning_consistency",
        "planning_stability", "goal_decomposition",
        "tool_correctness", "argument_correctness", "tool_reliability",
        "tool_selection", "tool_thrashing", "hallucinated_tool",
        "task_completion", "step_efficiency", "latency", "cost",
        "retry_rate", "failure_rate", "recovery_score",
        "memory_correctness", "memory_pollution", "memory_recall",
        "context_dependency", "coordination", "communication", "delegation",
        "consensus", "deadlock", "prompt_injection", "instruction_override",
        "pii", "toxicity", "data_leakage", "drift_score", "trajectory_drift",
        "tool_usage_drift", "cost_drift", "behavior_fingerprint",
    }
    assert expected.issubset(set(_REGISTRY.keys()))


def test_layers_map_to_registered_metrics():
    for layer in range(1, 7):
        for m in layers.layer_metrics(layer):
            assert m in _REGISTRY, f"{m} (layer {layer}) not registered"


# ---------------------------------------------------------------- suite
def test_monte_carlo_output_shape():
    def agent(inp):
        return "A" if random.random() < 0.7 else "B"

    res = Suite(seed=1).run(agent, Dataset([Case(input="x", expected="A")]), trials=200)
    for key in default_metrics():
        assert key in res, f"missing {key}"
    assert 0.0 <= res["success_rate"] <= 1.0
    assert 0.0 <= res["behavioral_entropy"] <= 1.0
    assert 0.0 <= res["drift_score"] <= 1.0


def test_deterministic_zero_entropy():
    res = Suite(seed=1).run(lambda i: "A", Dataset([Case(input="x", expected="A")]),
                            trials=50)
    assert res["behavioral_entropy"] == 0.0
    assert res["success_rate"] == 1.0
    assert res["recovery_score"] == 1.0


def test_extractor_captures_tool_trajectory():
    def agent(inp):
        return AgentRun(run_id="", input=inp, output="done",
                        events=[ToolCallEvent("search"), ToolCallEvent("answer")])

    def extractor(run):
        return [e.name for e in run.events]

    res = Suite(seed=1).run(agent, Dataset([Case(input="x", expected="done")]),
                            trials=80, extractor=extractor)
    assert res["tool_reliability"] == 1.0
    assert res["trajectory_entropy"] == 0.0          # identical full sequences
    assert res["tool_entropy"] == 1.0                 # two equally-used tools = max entropy


def test_error_is_captured_as_failure():
    def agent(inp):
        raise ValueError("boom")

    res = Suite(seed=1).run(agent, Dataset([Case(input="x", expected="A")]), trials=20)
    assert res["success_rate"] == 0.0
    assert res["recovery_score"] == 0.0


def test_assert_stable_raises_and_passes():
    res = {"success_rate": 0.3}
    with pytest.raises(AssertionError):
        assert_stable(res, success_rate__gt=0.9)
    assert_stable({"success_rate": 0.95}, success_rate__gte=0.9)


def test_custom_metric_plugin():
    @metric("my_dummy")
    class MyDummy:
        def compute(self, b):
            return 1.0

    assert "my_dummy" in _REGISTRY
    res = Suite(metrics=["my_dummy"]).run(lambda i: "x",
                                           Dataset([Case(input="x")]), trials=5)
    assert res["my_dummy"] == 1.0
