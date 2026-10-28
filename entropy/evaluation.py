"""Detailed evaluation and paired comparisons built on the standard Suite."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .core.suite import Suite
from .datasets.cases import Case
from .metrics.registry import norm_entropy
from .statistics import (
    _confidence,
    _integer,
    bootstrap_interval,
    pass_at_k,
    wilson_interval,
)


def _prepare(
    agent: Callable,
    inputs: Iterable,
    trials: int,
    pass_k: Iterable[int],
    confidence: float,
    expected: Iterable | None,
    check: Callable | None,
) -> tuple[list[Case], list[int]]:
    if not callable(agent):
        raise TypeError("agent must be callable")
    # Validate before any provider call can incur cost.
    pass_at_k(trials, 0, 1)
    wilson_interval(0, trials, confidence)
    ks = list(pass_k)
    if not ks:
        raise ValueError("pass_k must contain at least one k")
    for k in ks:
        pass_at_k(trials, 0, k)
    cases = Suite._as_cases(inputs, expected=expected, check=check)
    if not cases:
        raise ValueError("detailed evaluation requires at least one case")
    return cases, list(dict.fromkeys(ks))


def _details(
    suite: Suite,
    summary: dict,
    cases: list[Case],
    ks: list[int],
    confidence: float,
) -> dict:
    rows = []
    for index, (case, batch) in enumerate(zip(cases, suite.batches)):
        total = len(batch.successes)
        correct = sum(bool(success) for success in batch.successes)
        latencies = batch.metadata.get("_latencies", [])
        rows.append(
            {
                "index": index,
                "input": case.input,
                "metadata": dict(case.metadata),
                "trials": total,
                "successes": correct,
                "success_rate": correct / total,
                "confidence_interval": wilson_interval(correct, total, confidence),
                "pass_at_k": {str(k): pass_at_k(total, correct, k) for k in ks},
                "behavioral_entropy": norm_entropy(batch.actions),
                "mean_cost": sum(batch.costs) / total,
                "mean_latency": sum(latencies) / len(latencies) if latencies else None,
                "failures": [
                    {
                        "trial": trial + 1,
                        "output": batch.outputs[trial],
                        "error": batch.errors[trial],
                        "errors": [
                            event.data.get("error") or event.name
                            for event in batch.events[trial]
                            if event.type == "error"
                        ],
                    }
                    for trial, success in enumerate(batch.successes)
                    if not success
                ],
            }
        )
    if "confidence_interval" in summary:
        summary["confidence_interval"] = wilson_interval(
            sum(row["successes"] for row in rows),
            sum(row["trials"] for row in rows),
            confidence,
        )
    return {
        "summary": summary,
        "confidence": confidence,
        "total_trials": sum(row["trials"] for row in rows),
        "pass_at_k": {
            str(k): sum(row["pass_at_k"][str(k)] for row in rows) / len(rows)
            for k in ks
        },
        "cases": rows,
    }


def evaluate_detailed(
    agent: Callable,
    inputs: Iterable,
    *,
    trials: int = 100,
    seed: int = 0,
    metrics: list[str] | None = None,
    expected: Iterable | None = None,
    check: Callable | None = None,
    pass_k: Iterable[int] = (1,),
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate once and return aggregate metrics plus inspectable per-case evidence.

    Pass@k is averaged equally across inputs. Failed outputs are retained in
    memory, with 1-based trial numbers and any recorded error events. Nothing
    is written to disk. All k values must be <= trials. Supply expected answers
    or a check to measure correctness, rather than just absence of errors.
    """
    cases, ks = _prepare(agent, inputs, trials, pass_k, confidence, expected, check)
    suite = Suite(seed=seed, metrics=metrics)
    summary = suite.run(agent, cases, trials=trials)
    return _details(suite, summary, cases, ks, confidence)


async def evaluate_detailed_async(
    agent: Callable,
    inputs: Iterable,
    *,
    trials: int = 100,
    seed: int = 0,
    metrics: list[str] | None = None,
    expected: Iterable | None = None,
    check: Callable | None = None,
    pass_k: Iterable[int] = (1,),
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Async equivalent of evaluate_detailed; trials are awaited sequentially."""
    cases, ks = _prepare(agent, inputs, trials, pass_k, confidence, expected, check)
    suite = Suite(seed=seed, metrics=metrics)
    summary = await suite.run_async(agent, cases, trials=trials)
    return _details(suite, summary, cases, ks, confidence)


def compare_agents(
    agents: Mapping[str, Callable],
    inputs: Iterable,
    *,
    trials: int = 100,
    baseline: str | None = None,
    seed: int = 0,
    expected: Iterable | None = None,
    check: Callable | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Compare synchronous agents on the same materialized inputs.

    Bootstrap paired per-input success-rate differences, not unpaired trial
    outputs. At least two inputs are needed for an interval. Positive intervals
    imply an observed improvement under the empirical bootstrap assumptions;
    they are not corrected for multiple comparisons or unobserved variation.
    Every agent runs every input trials times; seed only seeds resampling.
    Agents must not mutate shared inputs in place.
    """
    if not isinstance(agents, Mapping) or len(agents) < 2:
        raise ValueError("agents must map at least two names to callables")
    if any(
        not isinstance(name, str) or not name or not callable(agent)
        for name, agent in agents.items()
    ):
        raise ValueError(
            "agent names must be nonempty strings and values must be callable"
        )
    if any(
        inspect.iscoroutinefunction(agent)
        or inspect.iscoroutinefunction(agent.__call__)
        for agent in agents.values()
    ):
        raise ValueError("compare_agents requires synchronous agents")
    if baseline is None:
        baseline = next(iter(agents))
    if baseline not in agents:
        raise ValueError("baseline must name one of the agents")
    # Validate resampling before running the first agent.
    _confidence(confidence)
    _integer(resamples, "resamples", 2)
    cases, _ = _prepare(
        agents[baseline], inputs, trials, (1,), confidence, expected, check
    )
    reports = {
        name: evaluate_detailed(
            agent,
            cases,
            trials=trials,
            seed=seed,
            metrics=["success_rate", "behavioral_entropy", "cost"],
            confidence=confidence,
        )
        for name, agent in agents.items()
    }
    for report in reports.values():
        latencies = [row["mean_latency"] for row in report["cases"]]
        report["summary"]["latency"] = (
            sum(latencies) / len(latencies)
            if all(value is not None for value in latencies)
            else None
        )
    reference = reports[baseline]
    comparisons = {}
    for name, report in reports.items():
        if name == baseline:
            continue
        differences = [
            current["success_rate"] - previous["success_rate"]
            for current, previous in zip(report["cases"], reference["cases"])
        ]
        interval = (
            bootstrap_interval(
                differences, confidence=confidence, resamples=resamples, seed=seed
            )
            if len(differences) >= 2
            else None
        )
        status = "insufficient_data"
        if interval is not None:
            status = (
                "improved"
                if interval[0] > 0
                else "regressed"
                if interval[1] < 0
                else "inconclusive"
            )
        comparisons[name] = {
            "delta": {
                key: (
                    value - reference["summary"][key]
                    if value is not None and reference["summary"][key] is not None
                    else None
                )
                for key, value in report["summary"].items()
            },
            "success_rate_interval": interval,
            "status": status,
            "improved_cases": sum(value > 0 for value in differences),
            "regressed_cases": sum(value < 0 for value in differences),
            "unchanged_cases": sum(value == 0 for value in differences),
        }
    return {
        "baseline": baseline,
        "confidence": confidence,
        "resampling_unit": "input",
        "agents": reports,
        "comparisons": comparisons,
    }
