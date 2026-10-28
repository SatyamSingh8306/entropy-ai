# EntroPy (entropy-ai): AI agent evaluation for Python

[PyPI](https://pypi.org/project/entropy-ai/) | [Documentation](https://satyamsingh8306.github.io/entropy-ai/) | [Quickstart](https://satyamsingh8306.github.io/entropy-ai/guide/quickstart.html) | [Issues](https://github.com/SatyamSingh8306/entropy-ai/issues)

Test AI agent reliability with repeated Monte Carlo trials, behavioral entropy,
drift detection, and regression checks. EntroPy is a Python evaluation library
for non-deterministic agents, including LangChain, LangGraph and OpenAI Agents.

Measure **uncertainty, drift, stochastic
behavior, and emergent failures** — via Monte Carlo evaluation over repeated
runs and entropy-based metrics. Framework-agnostic, local-first, zero vendor
lock-in.

## Install

```bash
pip install entropy-ai                 # from PyPI — core (pandas, matplotlib, jinja2, rich, pyyaml)
pip install "entropy-ai[langchain]"    # optional: a framework adapter
pip install "entropy-ai[pdf,jupyter]"  # optional: PDF / Jupyter reports
```

Developing EntroPy itself? Clone the repo and use an editable install instead:

```bash
pip install -e .                       # editable install from the repo
pip install -e ".[pdf,jupyter]"        # + report extras (what CI installs)
```

## Quick start

```python
from entropy import evaluate, assert_stable

def my_agent(text):
    return text.strip().lower()

results = evaluate(my_agent, {" Hello ": "hello", "WORLD": "world"}, trials=20)
print(results["success_rate"])        # 1.0
print(results["behavioral_entropy"])  # 0.0: identical outputs across trials
assert_stable(results, success_rate__gte=0.95)
```

This example runs locally without an API key. Replace `my_agent` with your
synchronous agent callable. `trials` is a positive integer and runs **per input**;
2 inputs with 20 trials call the agent 40 times. Provider calls may incur costs.
Success means matching the expected answer above; without `expected` or `check`,
success only means the agent did not report an error, not that its answer is correct.
`seed` does not seed your agent or make a remote model deterministic.

Use EntroPy to compare repeated outputs, test tool trajectories, inject faults,
and gate releases with `assert_stable`. Trace events are needed for tool and
trajectory metrics; choose metrics that match the evidence your agent records.

Need more control? Use `Suite`:

```python
from entropy import Suite, Dataset, Case

dataset = Dataset([Case(input="q1", expected="correct answer")])
results = Suite(seed=42).run(my_agent, dataset, trials=100)
```

`evaluate` / `Suite.run` accept a list of raw inputs, a `{input: expected}` dict,
or a `Dataset` — and a shared `check=` callable as the success criterion.

The canonical output (spec §9 / §21) includes: `success_rate`, `reliability`,
`variance`, `robustness`, `entropy`, `confidence_interval`, `behavioral_entropy`,
`behavioral_variance`, `trajectory_entropy`, `trajectory_diversity`, `tool_reliability`,
`drift_score`, `goal_stability`, `recovery_score`, `emergent_behavior`, and more.
56 metrics total; `entropy.default_metrics()` lists the canonical set.

## Async agent evaluation

Use `evaluate_async` for agents that return awaitables, including `async def`
functions. It returns the same metrics and accepts the same `expected`, `check`,
`metrics`, and `trials` options as `evaluate`.

```python
import asyncio
from entropy import evaluate_async

async def my_async_agent(text):
    await asyncio.sleep(0)
    return text.strip().lower()

async def main():
    results = await evaluate_async(my_async_agent, {" Hello ": "hello"}, trials=20)
    print(results["success_rate"])  # 1.0

asyncio.run(main())
```

In a notebook or async application, use `await evaluate_async(...)` directly.
For recorded batches or a custom extractor, use `await Suite().run_async(...)`.
Trials run sequentially, preserving input and trial order. Checks and extractors
are synchronous; blocking synchronous agents still block the event loop.
Cancellation propagates; ordinary agent exceptions count as failed trials.
Existing framework adapters remain synchronous: pass an async callable around
an SDK's async API when using this evaluator.

## Detailed evaluation, Pass@k and agent comparison

Inspect failures per input, estimate success within k attempts, and compare a
prompt or model change against a baseline using paired bootstrap intervals.

```python
from entropy import evaluate_detailed, compare_agents

cases = {" Hello ": "hello", "WORLD": "world", " Test ": "test"}

def baseline(text):
    return text.strip()

def candidate(text):
    return text.strip().lower()

report = evaluate_detailed(candidate, cases, trials=10, pass_k=[1, 5, 10])
print(report["pass_at_k"])
print(report["cases"][0]["confidence_interval"])

comparison = compare_agents(
    {"baseline": baseline, "candidate": candidate}, cases,
    trials=10, resamples=1000, seed=42,
)
print(comparison["comparisons"]["candidate"]["status"])
```

- `evaluate_detailed(...)`: aggregate metrics, per-input success rates, Wilson
  intervals, behavioral entropy, supplied cost/latency, failed outputs and errors.
- `evaluate_detailed_async(...)`: the same report for async callables.
- `compare_agents(...)`: named synchronous agents evaluated on identical inputs;
  metric deltas, improved/regressed case counts and paired bootstrap intervals.
- `pass_at_k(total, correct, k)`: unbiased repeated-attempt success estimator.
- `wilson_interval(correct, total, confidence=0.95)`: success-rate uncertainty
  that remains nonzero even when all observed trials succeed or fail.
- `bootstrap_interval(values, confidence=0.95, resamples=2000, seed=0)`:
  reproducible percentile interval for an arithmetic mean (options are keyword-only).

Use expected answers or `check=` to measure correctness. All k values must be
between 1 and `trials`. Comparisons resample **inputs**, pairing each candidate's
per-input success rate with the baseline; at least two inputs are required for
an interval. Results are exploratory, without multiple-comparison correction.
Small datasets and correlated agent runs limit inference; constant observations
can yield point bootstrap intervals. `seed` controls resampling, not provider randomness.
Reports retain raw inputs and failed outputs in memory; review them before sharing.

CLI equivalents:

```bash
entropy run --agent mymodule:agent --dataset cases.json --trials 20 --detailed --pass-k 1 5 10 --out details.json
entropy benchmark --agents mymodule:old mymodule:new --dataset cases.json --trials 20 --paired --resamples 2000 --out comparison.json
```

Read the [evaluation and comparison guide](https://satyamsingh8306.github.io/entropy-ai/guide/evaluation.html)
for report fields, statistical assumptions, and examples.

## Tracing

```python
from entropy import trace, event, instrument
with trace() as t:
    event("action", "tool:search")
# or auto-wrap any agent:
wrapped = instrument(my_agent)          # returns an AgentRun per call
```

## Custom metrics (plugin system)

```python
from entropy import metric
@metric("my_metric")
class MyMetric:
    def compute(self, batch): ...       # batch has outputs/events/successes/costs/...
```

## Simulation, Chaos, Observability

```python
from entropy import AdversarialSimulator, ChaosRunner, Watcher
adv = AdversarialSimulator(seed=0)
for v in adv.iterate("book a flight"):
    print(v)                            # typo / injection / distraction / leak variants

ChaosRunner(seed=0).run(agent, dataset, ["api_fail", "timeout"])   # fault injection
Watcher(agent, dataset, trials=20).watch(rounds=10)                # live drift + anomalies
```

## Framework adapters (lazy)

```python
from entropy import from_langchain, from_langgraph, from_openai, from_custom
runnable = from_langchain(my_langchain_agent)   # maps framework runs -> AgentRun
```

All 9 adapters (LangChain, LangGraph, OpenAI Agents, CrewAI, PydanticAI,
AutoGen, Google ADK, MCP, custom) import their framework **lazily**, so
`import entropy` works with none of them installed.

## CLI

```bash
entropy init                      # scaffold a project
entropy run    --agent m:a --dataset d.json --trials 100
entropy test   --agent m:a --dataset d.json --baseline base.json [--update]
entropy benchmark --agents m:a m:b --dataset d.json
entropy chaos   --agent m:a --dataset d.json --faults api_fail,timeout
entropy simulate --kind adversarial --text "book a flight"
entropy ui      --agent m:a --dataset d.json --serve
entropy watch   --agent m:a --dataset d.json --rounds 10
entropy report  --results r.json --format html
entropy dataset --path d.json
entropy doctor                      # check optional deps
```

## Reports

```python
from entropy import export_report
export_report(results, "html", "report.html")      # also: json, markdown, pdf, jupyter
```

## Layout

`core/` (models, tracing, exporters, suite) · `metrics/` (registry + all
categories) · `datasets/` · `simulation/` · `chaos/` · `regression/` ·
`benchmarks/` · `observability/` · `reports/` · `dashboard/` · `integrations/`
(adapters) · `cli/` · `plugins/`.

Tests: `python -m pytest tests/`.

## Documentation

- **Live site:** https://satyamsingh8306.github.io/entropy-ai/
- **Source:** [`docs/`](https://github.com/SatyamSingh8306/entropy-ai/tree/main/docs) — plain HTML/CSS/JS with no build step, deployed
  to GitHub Pages.

- Source: `docs/index.html`, `docs/guide/*.html`, `docs/api.html`,
  `docs/examples.html`, plus `docs/assets/` (CSS, JS, logo, search index).
- Local preview: serve the folder, e.g. `python -m http.server -d docs 8000`
  then open `http://localhost:8000`.
- Deploy: push to `main`; the
  [`.github/workflows/pages.yml`](.github/workflows/pages.yml) workflow
  uploads `docs/` and publishes it via GitHub Pages. Alternatively, set
  **Settings → Pages → Build and deployment → Source** to *Deploy from a
  branch* and choose the `main` branch `/docs` folder.

