import json
import pathlib

import pytest

from entropy import Dataset, Case, exporter, discover
from entropy.reports.exporters import export, to_markdown, to_json
from entropy.observability.watch import Watcher, render_metric_heatmap
from entropy.plugins.manager import list_exporters


# ---------------------------------------------------------------- reports
def _res():
    return {"success_rate": 0.9, "behavioral_entropy": 0.3, "drift_score": 0.1,
            "cost": 0.02, "recovery_score": 0.8}


def test_report_json_markdown_html(tmp_path):
    r = _res()
    jp = tmp_path / "r.json"
    export(r, "json", str(jp))
    assert json.loads(jp.read_text())["success_rate"] == 0.9
    mp = tmp_path / "r.md"
    export(r, "markdown", str(mp))
    assert "success_rate" in mp.read_text()
    hp = tmp_path / "r.html"
    export(r, "html", str(hp))
    assert "EntroPy Report" in hp.read_text()


def test_report_pdf_skipped_without_extra(tmp_path):
    pytest.importorskip("reportlab")
    p = tmp_path / "r.pdf"
    export(_res(), "pdf", str(p))
    assert p.exists()


def test_report_jupyter_skipped_without_extra(tmp_path):
    pytest.importorskip("nbformat")
    p = tmp_path / "r.ipynb"
    export(_res(), "jupyter", str(p))
    assert p.exists()


# ---------------------------------------------------------------- plugins
def test_exporter_decorator_registers():
    @exporter("myformat")
    def _my(res, path):
        return path
    assert "myformat" in list_exporters()


def test_discover_no_plugins():
    assert isinstance(discover(), list)


# ---------------------------------------------------------------- observability
def test_watcher_runs_and_detects():
    def agent(inp):
        import random
        return "A" if random.random() < 0.7 else "Z"

    ds = Dataset([Case(input="x", expected="A")])
    w = Watcher(agent, ds, trials=15, seed=1)
    w.watch(rounds=5)
    assert len(w.history) == 5
    assert len(w.anomalies("drift_score")) == 5
    assert isinstance(w.failure_clusters(), list)


def test_watcher_failure_clusters_captures_outputs():
    def agent(inp):
        return "BAD"  # always fails vs expected "A"

    ds = Dataset([Case(input="x", expected="A")])
    w = Watcher(agent, ds, trials=10, seed=0)
    w.watch(rounds=1)
    clusters = w.failure_clusters()
    assert clusters and clusters[0][0] == "BAD"


def test_watch_heatmap_writes_png(tmp_path):
    def agent(inp):
        import random
        return "A" if random.random() < 0.6 else "C"

    ds = Dataset([Case(input="x", expected="A")])
    w = Watcher(agent, ds, trials=10, seed=2)
    w.watch(rounds=4)
    p = tmp_path / "heat.png"
    render_metric_heatmap(w.history, str(p))
    assert p.exists() and p.stat().st_size > 100
