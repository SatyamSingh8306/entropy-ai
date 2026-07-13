"""Report exporters (spec §16): html, json, markdown, pdf, jupyter."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List


def numeric(results: Dict) -> Dict[str, float]:
    return {k: v for k, v in results.items() if isinstance(v, (int, float))}


def to_json(results: Dict, path: str, title: str = "EntroPy Report") -> str:
    p = pathlib.Path(path)
    p.write_text(json.dumps(results, indent=2, default=str))
    return str(p)


def to_markdown(results: Dict, path: str | None = None, title: str = "EntroPy Report") -> str:
    rows = "".join(f"| {k} | {v:.4f} |\n" for k, v in numeric(results).items())
    return f"# {title}\n\n| metric | value |\n| --- | --- |\n{rows}"


def to_html(results: Dict, path: str, title: str = "EntroPy Report") -> str:
    try:
        from ..dashboard.app import metric_bar
        bar = metric_bar(results)
        chart = f'<img src="data:image/png;base64,{bar}" alt="metrics"/>'
    except Exception:
        chart = ""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in numeric(results).items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px 10px}}img{{max-width:100%}}</style></head>
<body><h1>{title}</h1>{chart}
<table><tr><th>metric</th><th>value</th></tr>{rows}</table></body></html>"""
    p = pathlib.Path(path)
    p.write_text(html)
    return str(p)


def to_pdf(results: Dict, path: str, title: str = "EntroPy Report") -> str:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PDF export needs the 'pdf' extra: pip install entropy-ai[pdf]") from e
    p = pathlib.Path(path)
    doc = SimpleDocTemplate(str(p), pagesize=letter)
    styles = getSampleStyleSheet()
    data = [["metric", "value"]] + [[k, f"{v:.4f}"] for k, v in numeric(results).items()]
    table = Table(data)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999")]))
    doc.build([Paragraph(title, styles["Title"]), table])
    return str(p)


def to_jupyter(results: Dict, path: str, title: str = "EntroPy Report") -> str:
    try:
        import nbformat
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Jupyter export needs the 'jupyter' extra: pip install entropy-ai[jupyter]") from e
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}"),
        nbformat.v4.new_code_cell("import json\nresults = " + json.dumps(results, indent=2, default=str)),
    ]
    p = pathlib.Path(path)
    p.write_text(nbformat.writes(nb))
    return str(p)


_DISPATCH = {"json": to_json, "markdown": to_markdown, "md": to_markdown,
             "html": to_html, "pdf": to_pdf, "jupyter": to_jupyter, "ipynb": to_jupyter}


def export(results: Dict, fmt: str, path: str, title: str = "EntroPy Report") -> str:
    fmt = fmt.lower()
    if fmt not in _DISPATCH:
        raise ValueError(f"unknown format {fmt!r}; have {sorted(_DISPATCH)}")
    out = _DISPATCH[fmt](results, path, title)
    # to_markdown returns text, not a path; write it.
    if fmt in ("markdown", "md"):
        pathlib.Path(path).write_text(out)
        return path
    return out
