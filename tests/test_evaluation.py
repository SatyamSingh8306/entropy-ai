"""Detailed evidence, paired comparisons, and CLI integration."""

import asyncio
import json

import pytest
from entropy import (
    AgentRun,
    Case,
    Dataset,
    Event,
    compare_agents,
    evaluate_detailed,
    evaluate_detailed_async,
    wilson_interval,
)
from entropy.cli.main import main


def test_detailed_preserves_failures_metadata_and_pass_at_k():
    outcomes = iter(["ok", "wrong", "ok", "wrong"])
    dataset = Dataset([Case("x", expected="ok", metadata={"topic": "search"})])
    report = evaluate_detailed(
        lambda inp: next(outcomes), dataset, trials=4, pass_k=[1, 2, 4, 2]
    )
    row = report["cases"][0]
    assert report["total_trials"] == 4
    assert report["summary"]["success_rate"] == 0.5
    assert report["pass_at_k"] == pytest.approx({"1": 0.5, "2": 5 / 6, "4": 1})
    assert row["metadata"] == {"topic": "search"}
    assert row["successes"] == 2
    assert row["behavioral_entropy"] == 1
    assert row["mean_latency"] is None
    assert row["failures"] == [
        {"trial": 2, "output": "wrong", "error": None, "errors": []},
        {"trial": 4, "output": "wrong", "error": None, "errors": []},
    ]
    json.dumps(report)


def test_detailed_captures_exception_and_metadata_only_errors():
    def agent(inp):
        if inp == "raises":
            raise RuntimeError("provider failed")
        return AgentRun(
            "id",
            inp,
            "ok",
            metadata={"error": "tool crashed", "latency": 0.4},
            cost=0.2,
            events=[Event("tool", "lookup")],
        )

    report = evaluate_detailed(agent, ["raises", "recorded"], trials=1)
    assert report["summary"]["success_rate"] == 0
    assert report["cases"][0]["failures"][0]["errors"] == ["provider failed"]
    assert report["cases"][1]["failures"][0]["error"] == "tool crashed"
    assert report["cases"][1]["mean_latency"] == 0.4
    assert report["cases"][1]["mean_cost"] == 0.2


def test_detailed_async_matches_sync():
    async def agent(inp):
        await asyncio.sleep(0)
        return inp.upper()

    expected = evaluate_detailed(
        str.upper,
        ["a", "b"],
        expected=["A", "B"],
        trials=2,
        pass_k=[1, 2],
        confidence=0.8,
    )
    actual = asyncio.run(
        evaluate_detailed_async(
            agent,
            ["a", "b"],
            expected=["A", "B"],
            trials=2,
            pass_k=[1, 2],
            confidence=0.8,
        )
    )
    assert actual == expected
    assert actual["summary"]["confidence_interval"] == wilson_interval(4, 4, 0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trials": 0},
        {"trials": True},
        {"pass_k": []},
        {"pass_k": [0]},
        {"trials": 2, "pass_k": [3]},
        {"confidence": 1},
    ],
)
def test_detailed_invalid_options_do_not_call_agent(kwargs):
    calls = []
    with pytest.raises(ValueError):
        evaluate_detailed(lambda inp: calls.append(inp), ["a"], **kwargs)
    assert calls == []


def test_detailed_rejects_empty_dataset_and_noncallable():
    with pytest.raises(ValueError, match="at least one case"):
        evaluate_detailed(str, [], trials=1)
    with pytest.raises(TypeError, match="callable"):
        evaluate_detailed(None, ["a"], trials=1)


def test_comparison_pairs_inputs_and_reuses_a_generator():
    calls = {"old": [], "new": []}

    def old(inp):
        calls["old"].append(inp)
        return inp

    def new(inp):
        calls["new"].append(inp)
        return inp.upper()

    report = compare_agents(
        {"old": old, "new": new},
        iter(["a", "b", "c"]),
        expected=iter(["A", "B", "C"]),
        trials=2,
        resamples=100,
    )
    assert calls["old"] == calls["new"] == ["a", "a", "b", "b", "c", "c"]
    result = report["comparisons"]["new"]
    assert result["delta"]["success_rate"] == 1
    assert result["success_rate_interval"] == [1, 1]
    assert result["status"] == "improved"
    assert result["improved_cases"] == 3
    assert report["resampling_unit"] == "input"
    assert report["baseline"] == "old"


def test_comparison_direction_ties_and_insufficient_inputs():
    agents = {"good": str.upper, "bad": str, "same": str.upper}
    report = compare_agents(
        agents, {"a": "A", "b": "B"}, baseline="good", trials=1, resamples=100
    )
    assert report["comparisons"]["bad"]["status"] == "regressed"
    assert report["comparisons"]["same"]["status"] == "inconclusive"
    assert report["comparisons"]["same"]["success_rate_interval"] == [0, 0]
    one = compare_agents(agents, {"a": "A"}, trials=1, resamples=100)
    assert one["comparisons"]["bad"]["status"] == "insufficient_data"
    assert one["comparisons"]["bad"]["success_rate_interval"] is None


def test_comparison_mixed_case_differences_are_inconclusive():
    report = compare_agents(
        {"base": lambda inp: "a", "candidate": lambda inp: "b"},
        {"a": "a", "b": "b"},
        trials=2,
        resamples=500,
    )
    result = report["comparisons"]["candidate"]
    assert result["status"] == "inconclusive"
    assert result["improved_cases"] == result["regressed_cases"] == 1
    assert result["success_rate_interval"][0] <= 0 <= result["success_rate_interval"][1]


@pytest.mark.parametrize(
    "options",
    [{"baseline": "missing"}, {"confidence": 0}, {"resamples": 1}, {"trials": 0}],
)
def test_bad_comparison_options_are_rejected_before_calls(options):
    calls = []
    with pytest.raises(ValueError):
        compare_agents({"a": lambda inp: calls.append(inp), "b": str}, ["x"], **options)
    assert calls == []


def test_comparison_requires_multiple_named_sync_agents():
    async def async_agent(inp):
        return inp

    for agents in (
        {"a": str},
        {"": str, "b": str},
        {"a": None, "b": str},
        {"a": async_agent, "b": str},
    ):
        with pytest.raises(ValueError):
            compare_agents(agents, ["x"], trials=1)


def test_cli_detailed_and_paired_comparison(tmp_path, capsys):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"input": "a", "expected": "A"}, {"input": "b", "expected": "B"}])
    )
    out = tmp_path / "report.json"
    assert (
        main(
            [
                "run",
                "--agent",
                "builtins:str",
                "--dataset",
                str(dataset),
                "--trials",
                "2",
                "--detailed",
                "--pass-k",
                "1",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["total_trials"] == 4
    assert json.loads(capsys.readouterr().out)["pass_at_k"] == {"1": 0, "2": 0}
    assert (
        main(
            [
                "benchmark",
                "--agents",
                "builtins:str",
                "builtins:repr",
                "--dataset",
                str(dataset),
                "--trials",
                "2",
                "--paired",
                "--resamples",
                "100",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["baseline"] == "builtins:str"
    assert result["comparisons"]["builtins:repr"]["status"] == "inconclusive"
    assert json.loads(capsys.readouterr().out) == result


def test_existing_cli_outputs_are_preserved(tmp_path, capsys):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"input": "a", "expected": "a"}]))
    assert (
        main(
            [
                "run",
                "--agent",
                "builtins:str",
                "--dataset",
                str(dataset),
                "--trials",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["success_rate"] == 1
    assert (
        main(
            [
                "benchmark",
                "--agents",
                "builtins:str",
                "builtins:repr",
                "--dataset",
                str(dataset),
                "--trials",
                "1",
            ]
        )
        == 0
    )
    assert "| variant |" in capsys.readouterr().out


def test_comparison_uses_latency_evidence_instead_of_cost_proxy():
    def old(inp):
        return AgentRun("", inp, inp, cost=5, metadata={"latency": 2.0})

    def new(inp):
        return AgentRun("", inp, inp, cost=1, metadata={"latency": 0.5})

    report = compare_agents(
        {"old": old, "new": new, "plain": str},
        {"a": "a", "b": "b"},
        trials=1,
        resamples=100,
    )
    assert report["comparisons"]["new"]["delta"]["latency"] == -1.5
    assert report["comparisons"]["new"]["delta"]["cost"] == -4
    assert report["agents"]["plain"]["summary"]["latency"] is None
    assert report["comparisons"]["plain"]["delta"]["latency"] is None
