"""The docs search index is generated from the pages, so it must match them.

docs/site/data/search-index.json is built by scripts/build_search_index.py from
every heading and API object on every page. Search used to run on a hand-written
list in which one entry in five pointed at an anchor that no longer existed; a
generated index only stays honest if a stale one fails the build.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_published_search_index_matches_the_pages():
    r = subprocess.run(
        [sys.executable, "scripts/build_search_index.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_indexed_anchor_exists_on_its_page():
    import json
    import re

    data = json.loads((ROOT / "docs/site/data/search-index.json").read_text())
    ids = {}
    missing = []
    for item in data["items"]:
        page, _, anchor = item["u"].partition("#")
        f = ROOT / "docs/site" / (page + "index.html" if page.endswith("/") or not page else page)
        assert f.exists(), item["u"]
        if anchor:
            if f not in ids:
                ids[f] = set(re.findall(r'id="([^"]+)"', f.read_text()))
            if anchor not in ids[f]:
                missing.append(item["u"])
    assert not missing, missing[:10]
