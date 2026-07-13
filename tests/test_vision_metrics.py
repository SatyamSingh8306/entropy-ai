"""Tests for the vision-doc metrics: LoopDetection, CostStability,
Reliability Score, and the edit-distance / JS-divergence alignments."""
from entropy import AgentRun, ActionEvent, Suite, Dataset, Case


def test_loop_detection_detects_cycle():
    def agent(inp):
        return AgentRun(run_id="", input=inp, output="x",
                        events=[ActionEvent("A"), ActionEvent("B"), ActionEvent("A")])
    res = Suite(seed=1, metrics=["loop_detection"]).run(
        agent, Dataset([Case(input="x", expected="x")]), trials=20)
    assert res["loop_detection"] == 1.0


def test_loop_detection_none_for_linear():
    def agent(inp):
        return AgentRun(run_id="", input=inp, output="x",
                        events=[ActionEvent("A"), ActionEvent("B")])
    res = Suite(seed=1, metrics=["loop_detection"]).run(
        agent, Dataset([Case(input="x", expected="x")]), trials=20)
    assert res["loop_detection"] == 0.0


def test_cost_stability_constant_is_one():
    def agent(inp):
        return AgentRun(run_id="", input=inp, output="x", cost=0.5)
    res = Suite(seed=1, metrics=["cost_stability"]).run(
        agent, Dataset([Case(input="x", expected="x")]), trials=10)
    assert res["cost_stability"] == 1.0


def test_reliability_score_formula():
    res = Suite(seed=1, metrics=["reliability_score"]).run(
        lambda i: "A", Dataset([Case(input="x", expected="A")]), trials=30)
    # success_rate 1.0 * stability_index 1.0
    assert res["reliability_score"] == 1.0


def test_trajectory_diversity_deterministic_zero():
    res = Suite(seed=1).run(lambda i: "A", Dataset([Case(input="x", expected="A")]),
                            trials=30)
    assert res["trajectory_diversity"] == 0.0


def test_drift_deterministic_zero():
    res = Suite(seed=1).run(lambda i: "A", Dataset([Case(input="x", expected="A")]),
                            trials=30)
    assert res["drift_score"] == 0.0
