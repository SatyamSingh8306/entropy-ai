"""Entropy CLI (spec §19). argparse-based; subcommands added per phase."""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import random
import sys
from typing import List

from ..benchmarks.bench import benchmark_refs, to_markdown, load_dataset
from ..datasets import Dataset, Case
from ..regression.test import run_regression
from ..core.suite import Suite
from ..chaos.runner import ChaosRunner


# ---------------------------------------------------------------- helpers
def _load_ref(ref: str):
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


def _write_json(path: str, obj) -> None:
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, default=str))


def _numeric(results: dict) -> dict:
    return {k: v for k, v in results.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------- commands
def cmd_init(args):
    d = pathlib.Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "entropy.toml").write_text(
        '[entropy]\nname = "my-eval"\nseed = 42\ntrials = 100\n')
    (d / "agent.py").write_text(
        "def agent(inp):\n"
        "    # return the agent output (or an entropy.AgentRun)\n"
        "    return str(inp)\n")
    (d / "dataset.json").write_text(
        json.dumps({"cases": [{"input": "hello", "expected": "hello"}]}, indent=2))
    print(f"Scaffolded entropy project in {d}")


def cmd_run(args):
    agent = _load_ref(args.agent)
    ds = load_dataset(args.dataset)
    metrics = args.metrics.split(",") if args.metrics else None
    res = Suite(seed=args.seed, metrics=metrics).run(
        agent, ds, trials=args.trials)
    if args.out:
        _write_json(args.out, res)
    print(json.dumps(_numeric(res), indent=2))


def cmd_test(args):
    agent = _load_ref(args.agent)
    ds = load_dataset(args.dataset)
    out = run_regression(agent, ds, trials=args.trials, baseline_path=args.baseline,
                         update=args.update, seed=args.seed)
    if out["updated"]:
        print(f"Baseline written to {args.baseline}")
        return 0
    if out["violations"]:
        print("REGRESSION FAILED:")
        for v in out["violations"]:
            print(f"  {v['metric']}: baseline={v['baseline']:.4f} "
                  f"current={v['current']:.4f}")
        return 1
    print("Regression OK (within tolerance)")
    return 0


def cmd_benchmark(args):
    rows = benchmark_refs(args.agents, args.dataset, trials=args.trials, seed=args.seed)
    md = to_markdown(rows)
    print(md)
    if args.out:
        _write_json(args.out, {name: res for name, res in rows})
    return 0


def cmd_compare(args):
    a = json.loads(pathlib.Path(args.a).read_text())
    b = json.loads(pathlib.Path(args.b).read_text())
    keys = [k for k in a if isinstance(a[k], (int, float)) and k in b]
    print(f"{'metric':24} {'a':>10} {'b':>10} {'delta':>10}")
    for k in keys:
        d = a[k] - b[k]
        print(f"{k:24} {a[k]:10.4f} {b[k]:10.4f} {d:10.4f}")
    return 0


def cmd_replay(args):
    data = json.loads(pathlib.Path(args.trace).read_text())
    if "runs" in data:
        runs = data["runs"]
        print(f"trace: {len(runs)} runs")
        for r in runs[:5]:
            print(f"  {r.get('run_id')}: output={str(r.get('output'))[:40]}")
    if "nodes" in data:
        print(f"graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    numeric = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    if numeric:
        print("results:")
        for k, v in numeric.items():
            print(f"  {k}: {v:.4f}")
    return 0


def cmd_dataset(args):
    ds = load_dataset(args.path)
    types = {}
    for c in ds:
        t = type(c).__name__
        types[t] = types.get(t, 0) + 1
    print(f"dataset: {len(ds)} cases")
    print(f"types: {types}")
    for c in ds[:3]:
        print(f"  input={str(c.input)[:50]} expected={str(c.expected)[:30]}")
    return 0


def cmd_doctor(args):
    deps = {
        "pandas": "dataframes / .df()",
        "matplotlib": "charts / dashboard",
        "jinja2": "reports / dashboard",
        "rich": "watch TUI",
        "yaml": "yaml datasets",
        "reportlab": "PDF reports (extra: pdf)",
        "nbformat": "Jupyter reports (extra: jupyter)",
        "datasets": "HF datasets (extra: hf)",
    }
    print("optional dependencies:")
    for name, why in deps.items():
        try:
            importlib.import_module(name)
            print(f"  [ok]   {name:12} {why}")
        except Exception:
            print(f"  [miss] {name:12} {why}")
    print("framework adapters:")
    frameworks = ["langchain", "langgraph", "openai", "crewai", "pydantic_ai",
                  "autogen", "google_adk", "mcp"]
    for fw in frameworks:
        try:
            importlib.import_module(fw)
            print(f"  [ok]   {fw}")
        except Exception:
            print(f"  [miss] {fw}")
    return 0


# ---------------------------------------------------------------- parser
def cmd_simulate(args):
    if args.kind == "adversarial":
        from ..simulation.adversarial import AdversarialSimulator
        adv = AdversarialSimulator(seed=args.seed)
        for v in adv.iterate(args.text):
            print(v)
    elif args.kind == "user":
        from ..simulation.user import RandomUserSimulator
        sim = RandomUserSimulator(pool=["hi", "help me", "why?", "thanks"], seed=args.seed)
        for _ in range(args.rounds):
            print(sim.next())
    elif args.kind == "env":
        from ..simulation.environment import GridWorld
        env = GridWorld()
        for _ in range(args.steps):
            a = random.choice(["up", "down", "left", "right"])
            _, r, done, _ = env.step(a)
            print(f"{a:6} state={env.state} reward={r:.2f} done={done}")
            if done:
                break
    else:
        print(f"unknown sim kind {args.kind}")
        return 1
    return 0


def cmd_chaos(args):
    agent = _load_ref(args.agent)
    ds = load_dataset(args.dataset)
    cr = ChaosRunner(seed=args.seed)
    res = cr.run(agent, ds, args.faults.split(","), trials=args.trials)
    print(cr.summary(res))
    if args.out:
        _write_json(args.out, res)
    return 0


def cmd_ui(args):
    from ..dashboard.app import build_dashboard, serve
    agent = _load_ref(args.agent) if args.agent else (lambda i: "demo")
    ds = load_dataset(args.dataset) if args.dataset else Dataset([Case(input="x")])
    res = Suite(seed=args.seed).run(agent, ds, trials=args.trials)
    path = build_dashboard(res, args.out)
    print(f"Dashboard written to {path}")
    if args.serve:
        serve(args.out, port=args.port)
    return 0


def cmd_watch(args):
    from ..observability.watch import Watcher, render_metric_heatmap
    agent = _load_ref(args.agent)
    ds = load_dataset(args.dataset)
    w = Watcher(agent, ds, trials=args.trials, seed=args.seed)
    w.watch(rounds=args.rounds, interval=args.interval)
    if args.heatmap:
        p = render_metric_heatmap(w.history, args.heatmap)
        print(f"heatmap -> {p}")
    return 0


def cmd_report(args):
    from ..reports.exporters import export
    data = json.loads(pathlib.Path(args.results).read_text())
    out = export(data, args.format, args.out, title=args.title)
    print(f"report -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="entropy", description="EntroPy CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="scaffold a project")
    sp.add_argument("--dir", default=".")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("run", help="run the Monte Carlo suite")
    sp.add_argument("--agent", required=True, help="module:attr agent")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--trials", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--metrics", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("test", help="regression test vs baseline")
    sp.add_argument("--agent", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--trials", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--baseline", default="baseline.json")
    sp.add_argument("--update", action="store_true")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("benchmark", help="compare agent variants")
    sp.add_argument("--agents", required=True, nargs="+")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--trials", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("compare", help="compare two result JSONs")
    sp.add_argument("--a", required=True)
    sp.add_argument("--b", required=True)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("replay", help="replay a saved trace/results file")
    sp.add_argument("--trace", required=True)
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("dataset", help="inspect a dataset")
    sp.add_argument("--path", required=True)
    sp.set_defaults(func=cmd_dataset)

    sp = sub.add_parser("doctor", help="check optional dependencies")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("simulate", help="run a simulator (spec §11)")
    sp.add_argument("--kind", required=True, choices=["user", "env", "adversarial"])
    sp.add_argument("--text", default="book a flight")
    sp.add_argument("--rounds", type=int, default=5)
    sp.add_argument("--steps", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("chaos", help="run fault injection (spec §10)")
    sp.add_argument("--agent", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--faults", required=True, help="comma-separated fault names")
    sp.add_argument("--trials", type=int, default=50)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_chaos)

    sp = sub.add_parser("ui", help="build/serve the dashboard (spec §17)")
    sp.add_argument("--agent", default=None)
    sp.add_argument("--dataset", default=None)
    sp.add_argument("--trials", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=".entropy_ui")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--serve", action="store_true")
    sp.set_defaults(func=cmd_ui)

    sp = sub.add_parser("watch", help="live observability (spec §15)")
    sp.add_argument("--agent", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--rounds", type=int, default=10)
    sp.add_argument("--trials", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--interval", type=float, default=0.0)
    sp.add_argument("--heatmap", default=None)
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("report", help="export a report (spec §16)")
    sp.add_argument("--results", required=True)
    sp.add_argument("--format", default="html",
                    choices=["html", "json", "markdown", "pdf", "jupyter"])
    sp.add_argument("--out", default="report.html")
    sp.add_argument("--title", default="EntroPy Report")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
