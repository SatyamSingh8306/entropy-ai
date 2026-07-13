import pathlib

from entropy import Dataset, Case, instrument, find_adapter, list_adapters
from entropy.simulation.user import ScriptedUserSimulator, RandomUserSimulator
from entropy.simulation.environment import GridWorld, StatefulEnv
from entropy.simulation.adversarial import AdversarialSimulator
from entropy.chaos.runner import ChaosRunner
from entropy.chaos.faults import available
from entropy.dashboard.app import build_dashboard, metric_bar, behavioral_radar
from entropy.integrations import from_custom, from_langchain


# ---------------------------------------------------------------- simulation
def test_scripted_user():
    s = ScriptedUserSimulator(["a", "b", "c"])
    assert [s.next() for _ in range(4)] == ["a", "b", "c", "a"]


def test_random_user_seeded():
    a = RandomUserSimulator(["x", "y"], seed=1)
    b = RandomUserSimulator(["x", "y"], seed=1)
    assert [a.next() for _ in range(5)] == [b.next() for _ in range(5)]


def test_gridworld_reaches_goal():
    env = GridWorld()
    # drive deterministically to the goal (4,4)
    for a in ["right"] * 4 + ["down"] * 4:
        _, r, done, _ = env.step(a)
        if done:
            break
    assert env.state["done"] is True and env.state["reward"] == 1.0


def test_adversarial_perturbations():
    adv = AdversarialSimulator(seed=0)
    variants = adv.iterate("hello")
    assert len(variants) == 4
    assert any("Ignore previous instructions" in v for v in variants)


# ---------------------------------------------------------------- chaos
def test_chaos_faults_available():
    assert set(available()) >= {
        "api_fail", "timeout", "tool_crash", "memory_corrupt",
        "token_limit", "malformed_output", "network_delay", "rate_limit"}


def test_chaos_lowers_success():
    ds = Dataset([Case(input="x", expected="A")])
    res = ChaosRunner(seed=0).run(lambda i: "A", ds, ["api_fail"], trials=20)
    assert res["baseline"]["success_rate"] == 1.0
    assert res["faults"]["api_fail"]["success_rate"] == 0.0


def test_chaos_tool_crash_corrupts():
    def agent(inp):
        from entropy import AgentRun, ToolCallEvent
        return AgentRun(run_id="", input=inp, output="ok",
                        events=[ToolCallEvent("search")])
    ds = Dataset([Case(input="x", expected="ok")])
    res = ChaosRunner(seed=0).run(agent, ds, ["tool_crash"], trials=10)
    # tool_crash flags the run as errored -> success drops
    assert res["faults"]["tool_crash"]["success_rate"] == 0.0


# ---------------------------------------------------------------- dashboard
def test_dashboard_builds():
    res = {"success_rate": 0.9, "behavioral_entropy": 0.3, "drift_score": 0.1,
           "recovery_score": 0.8, "exploration_efficiency": 0.5}
    out = pathlib.Path(".entropy_test_ui")
    path = build_dashboard(res, str(out))
    html = pathlib.Path(path).read_text()
    assert "Overview" in html and "Metrics" in html
    assert len(metric_bar(res)) > 100
    # cleanup
    for f in out.glob("*"):
        f.unlink()
    out.rmdir()


# ---------------------------------------------------------------- adapters
def test_custom_adapter_wraps_callable():
    wrapped = from_custom(lambda i: "x")
    run = wrapped("q")
    assert run.output == "x"


def test_find_adapter_returns_custom_for_plain_callable():
    fn = lambda i: "z"
    wrap = find_adapter(fn)
    assert wrap is not None
    assert wrap("q").output == "z"


def test_all_adapters_registered():
    names = list_adapters()
    for expected in ["langchain", "langgraph", "openai", "crewai", "pydanticai",
                     "autogen", "google-adk", "mcp", "custom"]:
        assert expected in names, f"{expected} not registered"
