"""Full evaluation harness for a real LangChain agent (Ollama / minimax-m3).

This is a proper end-to-end evaluation, not a toy:
  * a real tool-using agent built with langchain.agents.create_agent
  * a varied dataset with meaningful success checks
  * Monte Carlo evaluation with a broad metric set
  * chaos/fault-injection resilience testing
  * live observability (Watcher) + anomaly heatmap
  * a generated HTML dashboard and exported report
  * a regression gate that fails CI if the agent regresses

Requires:  ollama serve  &&  ollama pull minimax-m3:cloud
Run:        python examples/langchain_full_eval.py
Artifacts:  ./eval_output/  (dashboard index.html + report.html + heatmap.png)
"""
from __future__ import annotations

import os
import sys
import time
import io

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from entropy import (
    Suite, Dataset, Case, assert_stable,
    from_langgraph, ChaosRunner, Watcher, render_metric_heatmap,
    build_dashboard, export_report, list_adapters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "minimax-m3:cloud"
TRIALS = 6                 # Monte Carlo trials per dataset case
CHAOS_FAULTS = ["tool_crash", "timeout", "token_limit"]
WATCH_ROUNDS = 3
OUT_DIR = "eval_output"


# ---------------------------------------------------------------------------
# Agent under test
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for a city."""
    return f"It is sunny in {city}."


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def build_agent():
    # Low temperature + small context keeps minimax-m3:cloud stable on a 4 GB GPU.
    llm = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048, num_predict=512)
    return create_agent(
        llm,
        tools=[get_weather, add],
        system_prompt="You are a helpful assistant. Use tools when needed and answer concisely.",
    )


def robust_wrap(agent):
    """Wrap with retry/backoff: local GPUs sometimes 500 transiently."""
    base = from_langgraph(agent)

    def wrapped(inp):
        last = None
        for attempt in range(3):
            try:
                return base(inp)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.0 * (attempt + 1))
        raise last

    return wrapped


# ---------------------------------------------------------------------------
# Dataset — varied, with real success criteria (substring / length checks)
# ---------------------------------------------------------------------------
def build_dataset():
    return Dataset([
        Case(input="What is the weather in Paris?",
             check=lambda o: "sunny" in (o or "").lower()),
        Case(input="What is the weather in Tokyo?",
             check=lambda o: "sunny" in (o or "").lower()),
        Case(input="What is the capital of France?",
             check=lambda o: "paris" in (o or "").lower()),
        Case(input="What is 12 + 29?",
             check=lambda o: "41" in (o or "")),
        Case(input="Tell me a joke.",
             check=lambda o: bool(o) and len(o) > 15),
        Case(input="Translate 'hello' to French.",
             check=lambda o: "bonjour" in (o or "").lower()),
    ])


# ---------------------------------------------------------------------------
# 1. Monte Carlo evaluation with a broad metric set
# ---------------------------------------------------------------------------
BROAD_METRICS = [
    "success_rate", "reliability", "variance", "robustness",
    "confidence_interval", "entropy", "stability_index",
    "behavioral_entropy", "behavioral_variance", "drift_score",
    "tool_reliability", "recovery_score", "exploration_efficiency",
    "loop_detection", "cost_stability", "reliability_score",
]


def run_evaluation(agent, dataset):
    suite = Suite(seed=42, metrics=BROAD_METRICS)
    results = suite.run(agent, dataset, trials=TRIALS)

    print("\n=== 1. Monte Carlo evaluation ===")
    print(f"trials/case: {TRIALS}   cases: {len(dataset)}")
    for k in ("success_rate", "behavioral_entropy", "drift_score",
              "tool_reliability", "recovery_score", "loop_detection",
              "cost_stability", "reliability_score"):
        v = results.get(k)
        print(f"  {k:20} {v if not isinstance(v, float) else round(v, 4)}")
    return results


# ---------------------------------------------------------------------------
# 2. Chaos resilience: run under fault injection and compare to baseline
# ---------------------------------------------------------------------------
def run_chaos(agent, dataset):
    print("\n=== 2. Chaos resilience ===")
    cr = ChaosRunner(seed=42)
    report = cr.run(agent, dataset, faults=CHAOS_FAULTS, trials=TRIALS)
    print(cr.summary(report))

    print("\n  resilience (drop vs baseline):")
    base = report["baseline"]["success_rate"]
    for fault, r in report["faults"].items():
        drop = base - r["success_rate"]
        print(f"    {fault:14} success {r['success_rate']:.3f}  "
              f"drop {drop:+.3f}  drift {r['drift_score']:.3f}")
    return report


# ---------------------------------------------------------------------------
# 3. Observability: watch over rounds, detect anomalies, render heatmap
# ---------------------------------------------------------------------------
def run_watch(agent, dataset):
    print("\n=== 3. Observability (Watcher) ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    watcher = Watcher(agent, dataset, trials=TRIALS, seed=7)
    watcher.watch(rounds=WATCH_ROUNDS)

    anoms = watcher.anomalies("drift_score")
    print(f"  drift anomalies flagged: {sum(anoms)}/{len(anoms)}")

    heatmap = os.path.join(OUT_DIR, "watch_heatmap.png")
    render_metric_heatmap(watcher.history, heatmap)
    print(f"  wrote {heatmap}")
    return watcher


# ---------------------------------------------------------------------------
# 4. Artifacts: dashboard + report
# ---------------------------------------------------------------------------
def build_artifacts(results):
    print("\n=== 4. Artifacts ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    dash = build_dashboard(results, OUT_DIR, title="LangChain Agent Evaluation")
    print(f"  dashboard: {dash}")

    report = os.path.join(OUT_DIR, "report.html")
    export_report(results, "html", report, title="LangChain Agent Evaluation")
    print(f"  report:    {report}")


# ---------------------------------------------------------------------------
# 5. Regression gate
# ---------------------------------------------------------------------------
def regression_gate(results):
    print("\n=== 5. Regression gate ===")
    assert_stable(
        results,
        success_rate__gte=0.6,
        drift_score__lt=0.5,
        tool_reliability__gte=0.6,
    )
    print("  assert_stable: OK")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    print(f"Registered adapters: {list_adapters()}")

    agent = build_agent()
    wrapped = robust_wrap(agent)
    dataset = build_dataset()

    # pre-flight: confirm Ollama is reachable before the heavy work
    for _ in range(3):
        try:
            wrapped("ping")
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise SystemExit(
            "Could not reach Ollama / model. Start it with `ollama serve` "
            "and run `ollama pull minimax-m3:cloud`.\n"
            f"Underlying error: {last}"
        )

    results = run_evaluation(wrapped, dataset)
    run_chaos(wrapped, dataset)
    run_watch(wrapped, dataset)
    build_artifacts(results)
    regression_gate(results)

    print("\nDONE. Artifacts in ./" + OUT_DIR)


if __name__ == "__main__":
    main()
