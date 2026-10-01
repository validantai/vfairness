"""The usage map pointed the public tree, and a public page, at private things.

Two halves, both reproduced by execution on 2026-09-10 before anything was
changed, and both invisible to the leak gate as it stood.

IN THE TREE. ``scripts/build_usage_map.py`` anchored on ``parents[2]``, one
level ABOVE the library, so it discovered four entry points that exist only in
the development monorepo and wrote them into ``docs/site/data/usage_map.json``,
which is published AND served from the docs site. Measured: 54 of the 330 rows
named one of those four, and ``docs/site/js/usage-map.js`` renders
``tracks[].path`` verbatim inside a ``<code>`` element, so a visitor read them.
The same anchor prefixed every notebook path with the library subtree name, a
path that does not resolve in the published repository at all, and it is why
the generator could not run over there either.

OFF THE TREE. The page fetched a sibling JSON built from the private platform
checkout. Measured against the live site that day: HTTP 200, 366447 bytes,
carrying the checkout name, a commit sha, the branch and 82 internal handler
paths, and a headless render of the page showed all four in the DOM. The file
itself is on the export's EXCLUDES list, so the tree copy was genuinely absent
and there was nothing for the gate to grep; what shipped was the POINTER.

The fix was to narrow the scan to this repository rather than to stop
publishing the map: the page is linked from the site navigation of two dozen
other pages, ``docs/site/js/usage-badges.js`` fetches the same JSON from every
one of them, and the leaking bytes were already live, which excluding the pair
from the export would not have changed.

WHAT THESE TESTS ARE FOR. Every pin below has a control beside it that asserts
a MEASURED value, because the cheap way to make a path stop leaking is to blank
it, and a blanked path counted as a track is the same defect wearing the fix's
clothes. Where a check cannot be built (no export script, no secret scanner)
these tests say so; could-not-check never reads as a pass.

NOTHING PRIVATE IS WRITTEN DOWN HERE. This module ships. Every probe is derived
at run time from this checkout's own layout, so the file cannot become the
thing that leaks what it guards.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"

BUILDER = LIB_ROOT / "scripts" / "build_usage_map.py"
USAGE_MAP = LIB_ROOT / "docs" / "site" / "data" / "usage_map.json"
PAGE_JS = LIB_ROOT / "docs" / "site" / "js" / "usage-map.js"
PAGE_HTML = LIB_ROOT / "docs" / "site" / "usage-map" / "index.html"
NOTEBOOK_DIR = LIB_ROOT / "notebooks"

#: Recon patterns counted in the export script on 2026-09-10, after this change
#: added six. A ratchet, not a target: the list may only grow. Deleting one is
#: how a guarded class silently becomes unguarded again, and the deletion looks
#: identical to a tidy-up in review.
RECON_PATTERN_FLOOR = 18

monorepo_only = pytest.mark.skipif(
    not EXPORT_SCRIPT.is_file(),
    reason=(
        "the export script is not present. It exists only in the private "
        "monorepo, which is the only place an export can run from."
    ),
)


def _map_data() -> dict:
    assert USAGE_MAP.is_file(), (
        f"{USAGE_MAP} is missing, so nothing about the published map could be "
        "checked. That is could-not-check, not a pass."
    )
    return json.loads(USAGE_MAP.read_text(encoding="utf-8"))


def _recon_patterns() -> list[str]:
    """The LEAK_PATTERNS_RECON entries, read out of the export script."""
    text = EXPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^LEAK_PATTERNS_RECON=\(\n(.*?)^\)$", text, re.S | re.M)
    assert match, (
        "the recon denylist could not be located in the export script, so its "
        "contents could not be checked. Could-not-check, not a pass."
    )
    patterns = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        quoted = re.match(r"^'([^']*)'", stripped)
        if quoted:
            patterns.append(quoted.group(1))
    return patterns


# ---------------------------------------------------------------------------
# 1.  The shipped map may only name paths this repository has
# ---------------------------------------------------------------------------


def test_the_shipped_map_names_only_paths_this_repository_has() -> None:
    """The refusal pin for the tree half."""
    data = _map_data()
    unresolvable: list[str] = []

    for track in data["tracks"]:
        if not (LIB_ROOT / track["path"]).is_file():
            unresolvable.append(f"tracks[{track['id']}].path -> {track['path']}")
    for row in data["rows"]:
        for usage in row["usages"]:
            if not (LIB_ROOT / usage["file"]).is_file():
                unresolvable.append(f"rows[{row['symbol']}].usages[].file -> {usage['file']}")

    assert not unresolvable, (
        "the published usage map names paths that do not exist in this "
        "repository, so a reader cannot open them and, when they belong to the "
        "checkout one level up, they disclose its layout: "
        f"{sorted(set(unresolvable))[:8]} ({len(set(unresolvable))} distinct)"
    )


def test_the_shipped_map_still_measures_the_notebooks_on_disk() -> None:
    """The over-correction control: MEASURED values, not blanked ones.

    A path that leaks stops leaking if it is emptied, and a track with an empty
    path still counts toward ``tracks_scanned``. This asserts the map is a
    reading of what is actually on disk, so that shortcut fails here.
    """
    data = _map_data()
    on_disk = sorted(p.name for p in NOTEBOOK_DIR.glob("*.ipynb"))
    assert on_disk, (
        f"{NOTEBOOK_DIR} holds no notebooks, so there was nothing to compare the "
        "map against. Could-not-check, not a pass."
    )

    named = sorted(PurePosixPath(t["path"]).name for t in data["tracks"])
    assert named == on_disk, (
        "the map's tracks are not the notebooks this repository holds "
        f"(map: {named}, on disk: {on_disk})"
    )
    assert data["summary"]["tracks_scanned"] == len(on_disk)

    blank = [t["id"] for t in data["tracks"] if not str(t["path"]).strip()]
    assert not blank, f"tracks carry an empty path but are still counted: {blank}"

    silent = [t["id"] for t in data["tracks"] if t["usage_count"] == 0]
    assert not silent, (
        f"tracks are counted as scanned but recorded no usage at all: {silent}. "
        "A scan that measured nothing must not be reported as a scan."
    )

    counted = sum(len(r["usages"]) for r in data["rows"])
    assert counted == data["summary"]["total_usages"] > 0, (
        "summary.total_usages is not the number of usages in the rows "
        f"({data['summary']['total_usages']} reported, {counted} present), so it "
        "is a stored number rather than a count of what was measured."
    )

    active = sum(1 for r in data["rows"] if r["status"] == "active")
    assert active == data["summary"]["active"] > 0
    assert all(r["usages"] for r in data["rows"] if r["status"] == "active"), (
        "a row is graded active with no usage behind it, which is a verdict without a measurement."
    )


# ---------------------------------------------------------------------------
# 2.  The builder refuses to emit a path it cannot resolve
# ---------------------------------------------------------------------------


def _builder_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_vf_usage_map_builder", BUILDER)
    assert spec and spec.loader, f"could not load {BUILDER}"
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for a module loaded by path.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_the_builder_refuses_a_path_above_the_library_root() -> None:
    """The refusal pin for the generator: no silent fallback to an absolute path."""
    module = _builder_module()
    outsider = module.TrackSpec(
        id="probe-outside-root",
        label="probe",
        category="probe",
        path=REPO_ROOT / "probe_entry_point.py",
        kind="python",
    )
    with pytest.raises(ValueError) as excinfo:
        outsider.rel
    assert str(REPO_ROOT / "probe_entry_point.py") in str(excinfo.value)

    outside = [str(t.path) for t in module.discover_tracks() if LIB_ROOT not in t.path.parents]
    assert not outside, f"discovery returned entry points outside the library root: {outside}"


def test_the_builder_still_resolves_a_real_notebook() -> None:
    """The over-correction control: refusing everything is not the fix."""
    module = _builder_module()
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebooks, f"{NOTEBOOK_DIR} holds no notebooks; could-not-check."
    spec = module.TrackSpec(
        id="probe-inside-root",
        label="probe",
        category="probe",
        path=notebooks[0],
        kind="notebook",
    )
    assert spec.rel == str(notebooks[0].relative_to(LIB_ROOT))
    assert (LIB_ROOT / spec.rel).is_file()

    tracks = module.discover_tracks()
    assert len(tracks) == len(notebooks), (
        f"discovery found {len(tracks)} entry points for {len(notebooks)} notebooks"
    )


# ---------------------------------------------------------------------------
# 3.  The page cannot request the artifact the export drops
# ---------------------------------------------------------------------------


def test_the_page_requests_no_json_but_its_own_map() -> None:
    """The refusal pin for the pointer half.

    Not "does not name the excluded file": ANY second JSON fetched from a page
    served without authentication is the same defect with a different filename.
    """
    js = PAGE_JS.read_text(encoding="utf-8")
    referenced = {PurePosixPath(m).name for m in re.findall(r"['\"]([^'\"\n]*\.json)['\"]", js)}
    assert referenced <= {USAGE_MAP.name}, (
        "the page names a JSON other than its own map, and this page is served "
        f"without authentication: {sorted(referenced - {USAGE_MAP.name})}"
    )
    assert js.count("fetch(") == 1, (
        f"the page makes {js.count('fetch(')} fetch calls; it needs exactly one, for its own map."
    )


def test_the_page_still_loads_its_own_map() -> None:
    """The over-correction control: deleting the data load is not the fix."""
    js = PAGE_JS.read_text(encoding="utf-8")
    assert re.search(r"const\s+DATA_URL\s*=\s*['\"][^'\"]*" + re.escape(USAGE_MAP.name), js), (
        "the page no longer declares a URL for its own map."
    )
    assert re.search(r"fetch\(\s*DATA_URL", js), "the page no longer fetches its own map."


def test_the_page_advertises_no_view_it_cannot_render() -> None:
    """A tab that can never be enabled is a claim the page cannot honour."""
    html = PAGE_HTML.read_text(encoding="utf-8")
    js = PAGE_JS.read_text(encoding="utf-8")

    copy_block = re.search(r"const SURFACE_COPY = \{(.*?)\n  \};", js, re.S)
    assert copy_block, "SURFACE_COPY could not be located; could-not-check."
    renderable = set(re.findall(r"^\s{4}(\w+):\s*\{", copy_block.group(1), re.M))
    assert renderable, "SURFACE_COPY declares no surface at all."

    advertised = set(re.findall(r'data-surface-btn="([^"]+)"', html))
    sectioned = {s for group in re.findall(r'data-surface="([^"]+)"', html) for s in group.split()}

    assert advertised <= renderable, (
        f"tabs offer surfaces the page cannot render: {advertised - renderable}"
    )
    assert sectioned <= renderable, (
        f"sections are keyed to surfaces the page cannot render: {sectioned - renderable}"
    )
    assert sectioned, "no section is keyed to a surface, so the page renders nothing."


# ---------------------------------------------------------------------------
# 4.  The gate now refuses the shape that leaked
# ---------------------------------------------------------------------------


def _gate(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--gate-only", str(directory)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.fixture
def owned_tree(tmp_path: Path) -> Path:
    """The four files this change owns, laid out as the export would publish them."""
    tree = tmp_path / "tree"
    for source in (BUILDER, USAGE_MAP, PAGE_JS, PAGE_HTML):
        destination = tree / source.relative_to(LIB_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tree


@monorepo_only
def test_the_gate_passes_over_the_files_this_change_owns(owned_tree: Path) -> None:
    """The over-correction control for the gate.

    Without it, the refusal below could be green because the gate refuses
    everything, which is a guard that cannot distinguish and therefore cannot
    guard.
    """
    result = _gate(owned_tree)
    if "gitleaks is REQUIRED" in result.stdout + result.stderr:
        pytest.fail(
            "the secret scanner is not installed, so the gate could not be run "
            "and nothing here was checked. Could-not-check, not a pass."
        )
    assert result.returncode == 0, (
        "the leak gate refuses the files this change owns:\n"
        f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )


@monorepo_only
def test_the_gate_refuses_the_subtree_prefix_that_leaked(owned_tree: Path) -> None:
    """The refusal pin for the gate, on a probe derived from this checkout.

    The probe is the shape that actually shipped: a library path written from
    the monorepo's point of view. It is built here from the library directory's
    own name and a notebook that exists, so this file carries no literal.
    """
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebooks, f"{NOTEBOOK_DIR} holds no notebooks; could-not-check."
    probe = f"{LIB_ROOT.name}/{notebooks[0].relative_to(LIB_ROOT).as_posix()}"

    planted = owned_tree / "docs" / "probe_note.md"
    planted.write_text(f"see {probe}\n", encoding="utf-8")

    result = _gate(owned_tree)
    if "gitleaks is REQUIRED" in result.stdout + result.stderr:
        pytest.fail(
            "the secret scanner is not installed, so the gate could not be run "
            "and nothing here was checked. Could-not-check, not a pass."
        )
    assert result.returncode != 0 and "leak gate matched" in result.stdout + result.stderr, (
        f"the gate accepted {probe!r}. That is the exact form that reached the "
        "published tree and the public site, and it names the checkout layout "
        "while pointing at a path the published repository does not have.\n"
        f"{result.stdout[-3000:]}"
    )


@monorepo_only
def test_the_recon_denylist_has_not_shrunk() -> None:
    """A ratchet. A pattern removed is a class silently unguarded again."""
    patterns = _recon_patterns()
    assert len(patterns) >= RECON_PATTERN_FLOOR, (
        f"the recon denylist holds {len(patterns)} patterns, below the "
        f"{RECON_PATTERN_FLOOR} measured when this pin was written."
    )
    assert len(set(patterns)) == len(patterns), f"duplicate recon patterns: {patterns}"
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:  # pragma: no cover - a broken pattern is the finding
            pytest.fail(f"recon pattern {pattern!r} does not compile: {exc}")


# ---------------------------------------------------------------------------
# 5.  The release commit message says what was checked
# ---------------------------------------------------------------------------


def _release_message() -> str:
    text = EXPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'commit -m "(Release v\$VERSION.*?)"\n', text, re.S)
    assert match, (
        "the release commit message could not be located in the export script, "
        "so its claims could not be checked. Could-not-check, not a pass."
    )
    return match.group(1)


@monorepo_only
def test_the_release_message_claims_no_unqualified_absence() -> None:
    """The refusal pin for the claim the export writes into public history."""
    message = _release_message()
    assert "no internal commits or infrastructure references" not in message, (
        "the release commit states an unqualified ABSENCE of internal "
        "references, while the only thing behind it is a fixed denylist. This "
        "very change found four shipped files the gate passed while they named "
        "private paths."
    )
    assert "leak gate" in message, (
        "the release commit does not say what was actually checked, so a reader "
        "has no way to weigh it."
    )


@monorepo_only
def test_the_release_message_still_claims_the_property_it_can_prove() -> None:
    """The over-correction control: the true half must survive the correction."""
    message = _release_message()
    assert "shares no ancestry" in message, (
        "the release commit no longer states the independent-history property, "
        "which the export DOES guarantee mechanically."
    )
    assert "Release v$VERSION" in message and "CHANGELOG.md" in message
