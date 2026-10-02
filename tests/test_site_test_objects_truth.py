"""The test-objects page may only offer files that exist, match their record, and run.

docs/site/test-objects/ publishes downloadable test objects and tells a reader what
each one is. Three things can silently go wrong with a page like that, and each has
a check here:

1. A FILE DRIFTS FROM ITS RECORD. manifest.json carries a sha256 and a size for
   every download, and the page prints the hash prefix. A file regenerated or
   hand-edited after the manifest was written would be served under a hash it does
   not have. Checked both ways: every file on disk is in the manifest, every
   manifest entry matches the bytes on disk.
2. A LINK POINTS AT NOTHING. Every data/ link on the page must name a manifest
   entry, and every manifest entry must be linked from the page's catalogue.
3. A SNIPPET STOPS RUNNING. The page's Python blocks are executed in order against
   the local copies (the published URL swapped for the local folder), and the one
   output the page quotes is compared with what the code prints.

Also pinned: the synthetic pack regenerates byte for byte from its script (the
page says so), and the LLM page no longer claims an access-level detection that
no code performs.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import pathlib
import re
import subprocess
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
PAGE = SITE / "test-objects" / "index.html"
DATA = SITE / "test-objects" / "data"
URL = "https://vfairness.validant.ai/test-objects/data/"


def _manifest():
    return {f["path"]: f for f in json.loads((DATA / "manifest.json").read_text())["files"]}


def test_every_file_matches_its_manifest_record():
    manifest = _manifest()
    on_disk = {
        str(p.relative_to(DATA))
        for p in DATA.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert on_disk == set(manifest), (
        f"not in manifest: {sorted(on_disk - set(manifest))}; "
        f"in manifest but missing: {sorted(set(manifest) - on_disk)}"
    )
    for rel, rec in manifest.items():
        blob = (DATA / rel).read_bytes()
        assert len(blob) == rec["bytes"], rel
        assert hashlib.sha256(blob).hexdigest() == rec["sha256"], (
            f"{rel} changed after the manifest"
        )


def test_page_links_and_catalogue_cover_exactly_the_manifest():
    page = PAGE.read_text()
    manifest = _manifest()
    linked = set(re.findall(r'href="data/([^"#]+)"', page))
    unknown = {p for p in linked if p not in manifest and p != "manifest.json"}
    assert not unknown, f"page links to files that are not published: {sorted(unknown)}"
    start = page.index("<!-- BEGIN:catalogue -->")
    block = page[start : page.index("<!-- END:catalogue -->")]
    for rel, rec in manifest.items():
        assert f'href="data/{rel}"' in block, f"{rel} is published but not in the catalogue"
        assert rec["sha256"][:12] in block, f"{rel}: the catalogue shows a stale hash"


def test_the_explorer_only_names_published_files():
    """Every file the "Start here" explorer offers must be published."""
    page = PAGE.read_text()
    blob = page[page.index('id="tx-data">') + len('id="tx-data">') :]
    data = json.loads(blob[: blob.index("</script>")])
    named = set()
    for p in data["pathways"].values():
        for text in p["start"]:
            named |= set(re.findall(r"([a-z_]+/[A-Za-z0-9_.]+\.(?:csv|jsonl|json|py))", text))
    assert named, "the explorer names no files at all"
    assert named <= set(_manifest()), (
        f"explorer names unpublished files: {sorted(named - set(_manifest()))}"
    )


def test_getting_started_links_resolve_to_published_files():
    gs = (SITE / "getting-started" / "index.html").read_text()
    linked = set(re.findall(r'href="\.\./test-objects/data/([^"#]+)"', gs))
    assert linked, "Getting Started no longer points at any test object"
    assert linked <= set(_manifest()), sorted(linked - set(_manifest()))


def test_no_page_claims_automatic_access_detection():
    for p in SITE.rglob("*.html"):
        assert "detects your access level automatically" not in p.read_text(), p


def test_synthetic_pack_regenerates_byte_for_byte(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "synth_single_bias_datasets.py"),
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    for f in sorted((DATA / "synthetic").glob("*.csv")):
        assert (tmp_path / f.name).read_bytes() == f.read_bytes(), f"{f.name} does not regenerate"


@pytest.mark.slow
def test_page_snippets_run_and_print_what_the_page_says():
    page = PAGE.read_text()
    blocks = [
        html.unescape(b)
        for b in re.findall(
            r'<pre><code class="language-python">(.*?)</code></pre>', page, flags=re.S
        )
    ]
    assert len(blocks) >= 5, "expected the page's five Python snippets"
    local = DATA.as_posix() + "/"
    ns: dict = {"__name__": "__main__"}
    outputs = []
    for i, code in enumerate(blocks, 1):
        code = code.replace(URL, local)
        buf = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
            warnings.simplefilter("ignore")
            exec(compile(code, f"snippet{i}", "exec"), ns)  # noqa: S102 (our own page)
        outputs.append(buf.getvalue())
    # The first snippet quotes its own output in a comment; it must be what runs.
    quoted = re.search(r"^# (\[.*\])$", blocks[0], flags=re.M).group(1)
    assert quoted in outputs[0], f"page quotes {quoted}, code printed {outputs[0]!r}"
