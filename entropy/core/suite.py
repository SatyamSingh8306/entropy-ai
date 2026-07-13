"""Monte Carlo evaluation suite (spec §9, §20) — the heart of EntroPy."""
from __future__ import annotations

import random
from typing import Any, Callable, Iterable, List, Optional

from ..core.models import AgentRun, ErrorEvent
from ..datasets.cases import Case, make_case
from ..metrics import Batch, default_metrics, get_metric

_OPS = {
    "__gt": lambda a, b: a > b,
    "__gte": lambda a, b: a >= b,
    "__lt": lambda a, b: a < b,
    "__lte": lambda a, b: a <= b,
}


class Suite:
    """Run an agent over a dataset many times and measure its behavior."""

    def __init__(self, seed: int = 0, metrics: Optional[List[str]] = None) -> None:
        self.seed = seed
        self.metric_names = metrics or default_metrics()
        self.batches: List[Batch] = []

    def run(
        self,
        agent: Callable,
        dataset: Iterable,
        trials: int = 100,
        extractor: Optional[Callable[[AgentRun], List[str]]] = None,
        expected: Optional[Iterable] = None,
        check: Optional[Callable] = None,
    ) -> dict:
        rng = random.Random(self.seed)
        cases = self._as_cases(dataset, expected=expected, check=check)
        self.batches = [self._run_one(agent, case, trials, rng, extractor)
                        for case in cases]
        return self._aggregate(self.batches)

    # --- input normalization (developer-friendly defaults) -----------------
    @staticmethod
    def _as_cases(items, check=None, expected=None) -> List[Case]:
        """Accept whatever is convenient; always hand ``_run_one`` real Cases.

        ``items`` may be: a ``Dataset``/list of ``Case``, a list of raw inputs,
        a single raw input, or a ``{input: expected}`` dict. ``expected`` (a
        parallel iterable) or ``check`` apply a shared success criterion to raw
        inputs. ``Case``/``Dataset`` callers keep full control.
        """
        if isinstance(items, str):
            items = [items]
        if isinstance(items, Case):
            return [items]
        if isinstance(items, dict):
            return [Case(input=k, expected=v, check=check) for k, v in items.items()]
        out: List[Case] = []
        exp_iter = iter(expected) if expected is not None else None
        for it in items:
            if isinstance(it, Case):
                out.append(it)
            elif isinstance(it, dict):
                c = make_case(it)
                if check is not None and c.check is None:
                    c = Case(input=c.input, expected=c.expected, check=check,
                             metadata=c.metadata)
                out.append(c)
            else:
                exp = next(exp_iter) if exp_iter is not None else None
                out.append(Case(input=it, expected=exp, check=check))
        return out

    # --- per-input Monte Carlo --------------------------------------------
    def _run_one(self, agent, case, trials: int, rng, extractor) -> Batch:
        b = Batch(inp=case.input)
        latencies: List[float] = []
        for _ in range(trials):
            run = self._call(agent, case.input)
            out = run.output
            events = list(run.events)
            b.outputs.append(out)
            b.events.append(events)

            if extractor:
                beh = extractor(run)
            elif events:
                beh = [e.name for e in events]
            else:
                beh = [str(out)]
            if not isinstance(beh, (list, tuple)):
                beh = [str(beh)]
            beh = [str(x) for x in beh]
            b.behaviors.append(beh)
            b.actions.append(beh[-1] if beh else str(out))
            b.tool_calls.append([e.name for e in events if e.type == "tool"])

            err = bool(run.metadata.get("error")) or any(e.type == "error" for e in events)
            ok = case.is_success(out) and not run.metadata.get("error")
            b.successes.append(ok)
            b.costs.append(float(run.cost or 0.0))
            b.errored.append(err)
            b.recovered.append(err and ok)

            if isinstance(run.metadata.get("latency"), (int, float)):
                latencies.append(float(run.metadata["latency"]))
        b.metadata["_latencies"] = latencies
        return b

    @staticmethod
    def _call(agent, inp) -> AgentRun:
        try:
            res = agent(inp)
        except Exception as e:  # robustness: capture failure instead of crashing
            return AgentRun(run_id="", input=inp, output=None,
                            events=[ErrorEvent("exception", {"error": str(e)})],
                            metadata={"error": str(e)})
        if isinstance(res, AgentRun):
            return res
        if isinstance(res, dict) and "output" in res:
            return AgentRun(run_id="", input=inp, output=res["output"],
                            events=res.get("events", []), cost=res.get("cost", 0.0),
                            metadata=res.get("metadata", {}))
        return AgentRun(run_id="", input=inp, output=res)

    # --- aggregation across inputs ----------------------------------------
    def _aggregate(self, batches: List[Batch]) -> dict:
        out: dict = {}
        per: dict[str, list] = {m: [] for m in self.metric_names}
        for b in batches:
            for m in self.metric_names:
                per[m].append(get_metric(m).compute(b))

        for m, vals in per.items():
            if not vals:
                out[m] = [] if m in ("emergent_behavior", "behavior_fingerprint") else 0.0
            elif m in ("emergent_behavior", "behavior_fingerprint"):
                seen, merged = set(), []
                for item in vals:
                    for sig in item:
                        key = tuple(sig)
                        if key not in seen:
                            seen.add(key)
                            merged.append(sig)
                out[m] = merged
            elif m == "confidence_interval":
                out[m] = [sum(v[0] for v in vals) / len(vals),
                          sum(v[1] for v in vals) / len(vals)]
            else:
                out[m] = sum(vals) / len(vals)
        return out


def evaluate(
    agent: Callable,
    inputs,
    *,
    trials: int = 100,
    seed: int = 0,
    metrics: Optional[List[str]] = None,
    expected: Optional[Iterable] = None,
    check: Optional[Callable] = None,
) -> dict:
    """One-line entry point for the common case.

    ``inputs`` is a list of raw inputs, a ``{input: expected}`` dict, or a
    ``Dataset``/list of ``Case``. ``expected``/``check`` set a shared success
    criterion; omit them and success = "didn't error" (see ``Case.is_success``).

    >>> evaluate(my_agent, ["q1", "q2"], trials=50)
    """
    return Suite(seed=seed, metrics=metrics).run(
        agent, inputs, trials=trials, expected=expected, check=check)


def assert_stable(results: dict, **thresholds: float) -> bool:
    """Regression gate: ``assert_stable(res, success_rate__gt=0.9, ...)``.

    Raises AssertionError listing every violated threshold; returns True if all pass.
    """
    failures = []
    for key, thr in thresholds.items():
        op, name = None, key
        for suffix, fn in _OPS.items():
            if key.endswith(suffix):
                op, name = fn, key[: -len(suffix)]
                break
        if name not in results:
            failures.append(f"{name}: not in results")
            continue
        val = results[name]
        if op is None:
            if not (abs(val - thr) < 1e-6):
                failures.append(f"{name}={val:.4f} != {thr}")
        elif not op(val, thr):
            failures.append(f"{name}={val:.4f} not {key.split('_')[-1]} {thr}")
    if failures:
        raise AssertionError("entropy.assert_stable failed:\n  " + "\n  ".join(failures))
    return True
