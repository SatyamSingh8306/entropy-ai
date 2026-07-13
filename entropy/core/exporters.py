"""Trace exporters: json, df (pandas), graph, otel, zip (spec §5)."""
from __future__ import annotations

import json
import zipfile
from typing import Any, Dict

from .models import AgentRun, Event, Trace


def _clean(o: Any) -> Any:
    """Make arbitrary objects JSON-serializable."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return str(o)


def _run_dict(r: AgentRun) -> Dict[str, Any]:
    return {
        "run_id": r.run_id,
        "input": _clean(r.input),
        "output": _clean(r.output),
        "cost": r.cost,
        "metadata": _clean(r.metadata),
        "events": [{"type": e.type, "name": e.name, "data": _clean(e.data)} for e in r.events],
    }


def to_json(trace: Trace) -> str:
    payload = {
        "metadata": _clean(trace.metadata),
        "nodes": [vars(n) for n in trace.nodes],
        "edges": [vars(e) for e in trace.edges],
        "runs": [_run_dict(r) for r in trace.runs],
    }
    return json.dumps(payload, indent=2)


def to_df(trace: Trace):
    """Return a pandas DataFrame of every event across all runs."""
    import pandas as pd

    rows = []
    for r in trace.runs:
        for i, e in enumerate(r.events):
            rows.append({
                "run_id": r.run_id,
                "seq": i,
                "type": e.type,
                "name": e.name,
                "data": json.dumps(_clean(e.data)),
                "cost": r.cost,
            })
    return pd.DataFrame(rows)


def to_graph(trace: Trace) -> Dict[str, Any]:
    """Return nodes/edges (spec §4). Builds a linear chain if none defined."""
    if trace.nodes or trace.edges:
        return {
            "nodes": [vars(n) for n in trace.nodes],
            "edges": [vars(e) for e in trace.edges],
        }
    nodes, edges = [], []
    for r in trace.runs:
        prev = None
        for i, e in enumerate(r.events):
            nid = f"{r.run_id}:{i}"
            nodes.append({"id": nid, "label": e.name, "kind": e.type})
            if prev is not None:
                edges.append({"src": prev, "dst": nid, "label": e.type})
            prev = nid
    return {"nodes": nodes, "edges": edges}


def to_otel(trace: Trace) -> Dict[str, Any]:
    """OTLP-style JSON: one span per event."""
    spans = []
    for r in trace.runs:
        for e in r.events:
            spans.append({
                "traceId": r.run_id[:16] if r.run_id else "0" * 16,
                "spanId": r.run_id[:8] + e.name[:4],
                "name": e.name,
                "kind": e.type,
                "attributes": _clean(e.data),
            })
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def to_zip(trace: Trace, path: str) -> str:
    import os

    path = os.path.abspath(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("trace.json", to_json(trace))
        z.writestr("graph.json", json.dumps(to_graph(trace), indent=2))
        for i, r in enumerate(trace.runs):
            z.writestr(f"runs/run_{i}.json", json.dumps(_run_dict(r), indent=2))
    return path
