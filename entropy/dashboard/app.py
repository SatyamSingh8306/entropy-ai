"""Dashboard (spec §17): static HTML + matplotlib charts, served locally."""
from __future__ import annotations

import base64
import html
import pathlib
from typing import Dict, List


def _png_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def metric_bar(results: Dict, title: str = "Metrics") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in results.items() if isinstance(v, (int, float))]
    items.sort(key=lambda x: x[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, max(2, 0.3 * len(names))))
    ax.barh(names, vals, color="#4f8cc9")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    return _png_b64(fig)


def behavioral_radar(results: Dict) -> str:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["behavioral_entropy", "trajectory_entropy", "drift_score",
            "behavioral_variance", "recovery_score", "exploration_efficiency"]
    keys = [k for k in keys if k in results and isinstance(results[k], (int, float))]
    if not keys:
        return ""
    vals = [min(1.0, max(0.0, results[k])) for k in keys]
    n = len(keys)
    angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    xs = [math.cos(a) for a in angles] + [math.cos(angles[0])]
    ys = [math.sin(a) for a in angles] + [math.sin(angles[0])]
    vx = [v * math.cos(a) for v, a in zip(vals, angles)] + [vals[0] * math.cos(angles[0])]
    vy = [v * math.sin(a) for v, a in zip(vals, angles)] + [vals[0] * math.sin(angles[0])]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": False})
    ax.plot(xs, ys, color="#ccc")
    ax.plot(vx, vy, color="#c94f6d", marker="o")
    ax.set_xticks(xs[:-1])
    ax.set_xticklabels(keys, fontsize=7)
    ax.set_yticks([])
    ax.set_title("Behavioral profile")
    return _png_b64(fig)


def _section(title: str, body: str) -> str:
    return f'<section><h2>{html.escape(title)}</h2>{body}</section>'


def build_dashboard(results: Dict, out_dir: str, title: str = "EntroPy") -> str:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bar = metric_bar(results)
    radar = behavioral_radar(results)

    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>"
        for k, v in results.items() if isinstance(v, (int, float)))
    nav = ["Overview", "Metrics", "Traces", "Drift", "Monte Carlo",
           "Benchmarks", "Failures", "Datasets"]
    nav_html = " | ".join(f'<a href="#{n.lower().replace(" ", "-")}">{n}</a>' for n in nav)

    mc = results.get("confidence_interval", [0, 0])
    overview = (
        f"<p>success_rate: <b>{results.get('success_rate', 0):.4f}</b> "
        f"(95% CI [{mc[0]:.3f}, {mc[1]:.3f}])</p>"
        f"<p>behavioral_entropy: <b>{results.get('behavioral_entropy', 0):.4f}</b> | "
        f"drift_score: <b>{results.get('drift_score', 0):.4f}</b> | "
        f"recovery_score: <b>{results.get('recovery_score', 0):.4f}</b></p>"
    )
    if radar:
        overview += f'<img src="data:image/png;base64,{radar}" alt="radar"/>'

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}
nav{{background:#f3f3f3;padding:.5rem 1rem;margin-bottom:1rem}}
section{{margin-bottom:2rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
img{{max-width:100%;border:1px solid #eee;margin-top:.5rem}}</style></head>
<body><h1>{html.escape(title)}</h1><nav>{nav_html}</nav>
{_section("Overview", overview)}
{_section("Metrics", f'<img src="data:image/png;base64,{bar}" alt="metrics"/><table><tr><th>metric</th><th>value</th></tr>{rows}</table>')}
{_section("Traces", "<p>Load a Trace with <code>entropy replay</code> / pass runs to the dashboard for trace views.</p>")}
{_section("Drift", f"<p>drift_score: <b>{results.get('drift_score', 0):.4f}</b> &mdash; 0 = stable, 1 = high drift.</p>")}
{_section("Monte Carlo", f"<p>trials aggregated; success_rate {results.get('success_rate', 0):.4f}, entropy {results.get('entropy', 0):.4f}.</p>")}
{_section("Benchmarks", "<p>Run <code>entropy benchmark</code> to populate comparison tables.</p>")}
{_section("Failures", f"<p>failure_rate: <b>{1 - results.get('success_rate', 0):.4f}</b>, recovery_score: <b>{results.get('recovery_score', 0):.4f}</b>.</p>")}
{_section("Datasets", "<p>Inspect with <code>entropy dataset --path &lt;file&gt;</code>.</p>")}
</body></html>"""
    (out / "index.html").write_text(page)
    return str(out / "index.html")


def serve(out_dir: str, port: int = 8000, host: str = "127.0.0.1") -> None:
    import http.server
    import socketserver
    from functools import partial

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"Dashboard serving at http://{host}:{port}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
