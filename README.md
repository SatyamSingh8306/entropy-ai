# EntroPy

Behavioral evaluation framework for non-deterministic AI agents.

Measures what other frameworks miss: **uncertainty, drift, stochastic
behavior, and emergent failures** — via Monte Carlo evaluation over repeated
runs and entropy-based metrics. Framework-agnostic, local-first, zero vendor
lock-in.

## Install

```bash
pip install -e .                       # core (pandas, matplotlib, jinja2, rich, pyyaml)
pip install "entropy-ai[langchain]"    # optional: a framework adapter
pip install "entropy-ai[pdf,jupyter]"  # optional: PDF / Jupyter reports
```

## Quick start

```python
from entropy import Suite, Dataset, Case

def my_agent(inp): ...
    # return an output, an AgentRun, or a dict with events

dataset = Dataset([Case(input="q1", expected="correct answer")])
results = Suite(seed=42).run(my_agent, dataset, trials=100)
print(results)   # success_rate, behavioral_entropy, drift_score, ...
```

The canonical output (spec §9 / §21) includes: `success_rate`, `reliability`,
`variance`, `robustness`, `entropy`, `confidence_interval`, `behavioral_entropy`,
`behavioral_variance`, `trajectory_entropy`, `trajectory_diversity`, `tool_reliability`,
`drift_score`, `goal_stability`, `recovery_score`, `emergent_behavior`, and more.
56 metrics total; `entropy.default_metrics()` lists the canonical set.

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

The static documentation site lives in [`docs/`](docs/) — plain
HTML/CSS/JS with no build step, deployable to GitHub Pages.

- Source: `docs/index.html`, `docs/guide/*.html`, `docs/api.html`,
  `docs/examples.html`, plus `docs/assets/` (CSS, JS, logo, search index).
- Local preview: serve the folder, e.g. `python -m http.server -d docs 8000`
  then open `http://localhost:8000`.
- Deploy: push to `main`; the
  [`.github/workflows/pages.yml`](.github/workflows/pages.yml) workflow
  uploads `docs/` and publishes it via GitHub Pages. Alternatively, set
  **Settings → Pages → Build and deployment → Source** to *Deploy from a
  branch* and choose the `main` branch `/docs` folder.

