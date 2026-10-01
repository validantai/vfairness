#!/usr/bin/env python3
"""Write docs/site/data/page-dates.json: the date each docs page last changed.

The docs show "Updated <date>" on every page. That date must come from the
page's own history, not from a hand-typed string that goes stale, so this reads
the last commit date of each page from git. The deploy workflow runs it with the
full history right before publishing, so the published dates are current even if
the committed copy of this file is older.

    python scripts/build_page_dates.py

Standard library and git only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
OUT = SITE / "data" / "page-dates.json"
SKIP = {"404.html", "legal/license.html"}


def last_change(path: Path) -> str | None:
    r = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    d = r.stdout.strip()
    return d or None


def main() -> int:
    pages = {}
    for p in sorted(SITE.rglob("*.html")):
        rel = p.relative_to(SITE).as_posix()
        if rel in SKIP or "test-objects/data" in rel:
            continue
        d = last_change(p)
        if not d:
            continue
        key = rel[: -len("index.html")] if rel.endswith("index.html") else rel
        pages[key] = d
    OUT.write_text(json.dumps({"pages": pages}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(pages)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
