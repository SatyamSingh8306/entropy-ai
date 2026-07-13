# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## EntroPy

Behavioral evaluation framework for non-deterministic AI agents. It measures
uncertainty, drift, stochastic behavior, and emergent failures via Monte Carlo
evaluation over repeated agent runs and entropy-based metrics. Pure Python,
`import entropy` has no third-party framework deps — adapters load lazily.

## Commands

```bash
pip install entropy-ai                 # users: install from PyPI
pip install -e .                       # contributors: editable install from the repo
pip install -e ".[pdf,jupyter]"        # + report extras (this is what CI installs)
pip install -e ".[langchain]"          # optional: one framework adapter

python -m build                        # sdist + wheel -> dist/
python -m pytest -q                    # full suite (CI: python -m pytest -q)
python -m pytest tests/test_basic.py   # single file
python -m pytest -k "drift"            # single test by name

twine upload dist/*                    # publish to PyPI (needs a NEW version)
```

Notes:
- **PyPI rejects re-uploads of the same version.** Bump `version` in
  `pyproject.toml` before every publish. The README is only packaged if
  `readme = "README.md"` is set (it is).
- The console script `entropy` maps to `entropy.cli.main:main`.
- Local docs preview: `python -m http.server -d docs 8000`. The live site
  (https://satyamsingh8306.github.io/entropy-ai/) is the static `docs/`
  folder, deployed by `.github/workflows/pages.yml` on push to `main` when
  `docs/**` changes. There is **no docs build step** — edit HTML/CSS/JS directly.

## Architecture

The public API is a single re-export surface in `entropy/__init__.py`; all
functionality lives in subpackages. The evaluation pipeline is:

1. **`Suite` (`core/suite.py`)** runs an agent over a `Dataset` `trials` times
   per case (Monte Carlo). Each agent call may return a plain output, an
   `AgentRun`, or a dict with `output`/`events`/`cost`/`metadata`. Exceptions
   are caught and turned into an `ErrorEvent` rather than crashing the run.
2. Per case, `_run_one` builds a **`Batch`** (`metrics/registry.py`) — the
   flattened per-trial record (outputs, events, behaviors, actions, tool_calls,
   successes, costs, errored, recovered) that every metric consumes.
3. `_aggregate` computes each requested metric across all batches and returns a
   `dict` of `metric_name -> value`. `Suite(seed=...)` is deterministic.

### Extension points (the two plugin systems)

- **Metrics** — subclass `Metric`, decorate with `@metric("name")`
  (`metrics/registry.py`), and call `default("name")` to include it in the
  canonical `default_metrics()` set (the ~56-metric output). `get_metric(name)
  .compute(batch)` is the contract. Shared math lives in the same module
  (`shannon`, `norm_entropy`, `bernoulli_entropy`, `js_divergence`,
  `levenshtein`, `half_drift`).
- **Framework adapters** (`integrations/base.py`) — subclass `Adapter` with
  `match()`/`wrap()` and decorate with `@adapter("name")`. `find_adapter(agent)`
  returns a wrap fn if one matches. The `from_langchain`, `from_openai`, … helpers
  import their framework **lazily**, so `import entropy` never pulls in optional
  deps. Keep new adapters following this lazy pattern.
- **Report exporters** (`plugins/`) — `@exporter` decorator + `discover()`.

### Data model (`core/models.py`)

`Event` (with typed subclasses: `ActionEvent`, `ReasoningEvent`,
`ToolCallEvent`, `ObservationEvent`, `MemoryRead/WriteEvent`, `ErrorEvent`,
`StateTransitionEvent`) → `AgentRun` (one execution: input/output/events/cost/
metadata) → `Trace` (runs + an optional `Node`/`Edge` graph). `Trace` carries
its own exporters: `json()`, `df()`, `graph()`, `otel()`, `zip()` (delegate to
`core/exporters.py`).

### Other subpackages

- `simulation/` — `AdversarialSimulator` (typo/injection/distraction/leak
  variants) and user/environment simulators.
- `chaos/` — `ChaRunner` fault injection (`api_fail`, `timeout`, tool crashes).
- `observability/` — `Watcher` live drift detection.
- `regression/` — `run_regression` + `assert_stable(results, metric__op=thr)`
  as a CI regression gate (see `Suite`-adjacent thresholds).
- `reports/` — `export_report(results, fmt, path)` for html/json/markdown/pdf/
  jupyter.
- `benchmarks/` — `benchmark_refs`, `to_markdown`, `load_dataset`.
- `cli/main.py` — argparse subcommands (`init`, `run`, `test`, `benchmark`,
  `chaos`, `simulate`, `ui`, `watch`, `report`, `dataset`, `doctor`).
  `entropy init` scaffolds `entropy.toml` + `agent.py` + `dataset.json`.
  Agent/dataset args are `module:attr` references resolved by `_load_ref`.

### Spec anchoring

Module docstrings reference a **`spec §N`** numbering (e.g. §2 adapters, §4
events, §9/§20 suite, §19 CLI). When adding or changing behavior, keep new code
tied to the matching spec section and note it in the docstring — it's the
implicit contract this codebase is built against.

## Packaging notes

- `pyproject.toml`: setuptools, `packages = ["entropy*"]`, `requires-python
  >=3.9` (CI actually tests 3.11–3.13).
- No `LICENSE` file is present; add one (copyright "Satyam Singh") if publishing
  legally requires it.
- `docs/index.html` carries a version badge (`v0.1.0`) that is edited by hand and
  drifts from the package version — update it on releases.
