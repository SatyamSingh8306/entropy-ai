"""Keep discovery metadata and the public first-run examples usable."""
import re
import pytest
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "docs").exists(), reason="documentation is not included in the sdist")
BASE = "https://satyamsingh8306.github.io/entropy-ai/"


def test_sitemap_matches_canonical_pages():
    sitemap = ElementTree.parse(ROOT / "docs/sitemap.xml")
    locations = [node.text for node in sitemap.findall(
        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    expected = []
    for page in (ROOT / "docs").rglob("*.html"):
        relative = page.relative_to(ROOT / "docs").as_posix()
        url = BASE + ("" if relative == "index.html" else relative)
        html = page.read_text(encoding="utf-8")
        assert re.findall(r'<link rel="canonical" href="([^"]+)"', html) == [url]
        assert f'<meta property="og:url" content="{url}"' in html
        expected.append(url)
    assert sorted(locations) == sorted(expected)


def test_public_quickstart_examples_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"## Quick start\s+```python\n(.*?)```", readme, re.S)[1]
    homepage = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    home_code = re.search(r"<pre><code>(from entropy.*?)</code></pre>", homepage, re.S)[1]
    for example in (code, unescape(home_code)):
        namespace = {}
        exec(compile(example, "public quickstart", "exec"), namespace)
        assert namespace["results"]["success_rate"] == 1.0
        assert namespace["results"]["behavioral_entropy"] == 0.0


def test_public_async_examples_run(capsys):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(import asyncio.*?)```", readme, re.S)[1]
    guide = (ROOT / "docs/guide/quickstart.html").read_text(encoding="utf-8")
    guide_code = re.search(r"<pre><code>(import asyncio.*?)</code></pre>", guide, re.S)[1]
    for example in (code, unescape(guide_code)):
        exec(compile(example, "public async quickstart", "exec"), {})
        assert capsys.readouterr().out.strip() == "1.0"


def test_public_evaluation_examples_run(capsys):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(from entropy import evaluate_detailed.*?)```", readme, re.S)[1]
    guide = (ROOT / "docs/guide/evaluation.html").read_text(encoding="utf-8")
    guide_code = re.search(r"<pre><code>(from entropy import evaluate_detailed.*?)</code></pre>",
                           guide, re.S)[1]
    for example in (code, unescape(guide_code)):
        namespace = {}
        exec(compile(example, "public evaluation example", "exec"), namespace)
        assert namespace["report"]["total_trials"] == 30
        assert namespace["comparison"]["comparisons"]["candidate"]["status"] == "improved"
        capsys.readouterr()
