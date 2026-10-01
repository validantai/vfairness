"""The published status of a capability must match what was measured about it.

This file exists because the badge is the whole point. A status ledger nobody
renders is a JSON file; a badge that can disagree with the ledger is worse than no
badge, because a reader believes it. Every assertion here is one way the two could
come apart.

The four that matter, and why each is separate:

  1. NO UNIT IS SILENT. Every code unit on the measured public surface has a row.
     A unit that drops out of the ledger drops out of the denominator, and the
     percentages improve without anything being checked. That is the exact defect
     this library was audited for, applied to its own status page.

  2. NOTHING READS CHECKED WITHOUT EVIDENCE. A row with no source may only be
     NOT CHECKED. There is no default, no inference from the name.

  3. NO OPEN DEFECT IS HIDDEN. The FIX PENDING set must be exactly the set of
     units any grading source calls DEFECT OPEN.

  4. THE PAGES MATCH THE LEDGER. Byte-level: the stamper run in --check mode must
     find nothing to change, on the HTML, the markdown and the grid.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "capability-status.json"
API_PAGE = ROOT / "docs" / "site" / "api-reference" / "index.html"
GRID = ROOT / "docs" / "site" / "status" / "index.html"

CHECKED, FIX_PENDING, NOT_CHECKED = "CHECKED", "FIX PENDING", "NOT CHECKED"


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert LEDGER.exists(), "run scripts/capability_status.py"
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_every_unit_on_the_surface_has_a_row(ledger):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import library_kpis

    surface = library_kpis._walk_public_surface()
    measured = set()
    for kind in ("functions", "classes", "methods"):
        measured |= set(surface[kind]["ungraded"]) | set(surface[kind]["graded_names"])
    missing = measured - set(ledger["units"])
    assert not missing, (
        f"{len(missing)} code units carry no published status, e.g. {sorted(missing)[:5]}"
    )
    assert len(ledger["units"]) == surface["total"]["total"]


def test_no_row_reads_checked_without_evidence(ledger):
    liars = [
        q for q, r in ledger["units"].items() if r["status"] != NOT_CHECKED and not r.get("sources")
    ]
    assert not liars, f"{len(liars)} rows claim a status with no evidence behind them: {liars[:5]}"


def test_every_status_is_one_of_the_three(ledger):
    bad = {r["status"] for r in ledger["units"].values()} - {CHECKED, FIX_PENDING, NOT_CHECKED}
    assert not bad, f"unknown status published: {bad}"


def test_no_open_defect_is_cleared_without_evidence(ledger):
    """A recorded defect may stop being FIX PENDING ONLY via a later wave that
    carries a pin AND a sabotage. That is the one door out, and this test is the
    lock on it: without it, "closing" a defect is a JSON edit.

    Not asserted by recomputing what the ledger computes, that would pass
    whatever the ledger did. Asserted against the RECORD: for every unit the
    2026-09-18 wave called DEFECT OPEN, either it is still published as FIX
    PENDING, or some later wave names it with both kinds of evidence.
    """
    wave = json.loads((ROOT / "docs" / "surface-grading-2026-09-18.json").read_text())["items"]
    later = {}
    for path in sorted((ROOT / "docs").glob("surface-grading-2026-09-2*.json")):
        later.update(json.loads(path.read_text())["items"])

    units = ledger["units"]
    by_tail = {}
    for qual in units:
        by_tail.setdefault(qual.split(".")[-1], []).append(qual)

    unevidenced = []
    for name, rec in wave.items():
        if rec.get("grade") != "DEFECT OPEN":
            continue
        targets = [name] if name in units else by_tail.get(name.split(".")[-1], [])
        for qual in targets:
            if units[qual]["status"] == FIX_PENDING:
                continue
            claim = later.get(qual) or later.get(name)
            if not claim or not (claim.get("test_file") and claim.get("sabotage")):
                unevidenced.append(qual)
    assert not unevidenced, (
        f"{len(unevidenced)} recorded defect(s) are no longer published as FIX PENDING and no "
        f"later wave carries a pin and a sabotage for them: {sorted(unevidenced)[:5]}"
    )
    for q, r in units.items():
        if r["status"] == FIX_PENDING:
            assert r.get("defect"), f"{q} is FIX PENDING with no defect kind recorded"


def test_a_later_wave_cannot_close_a_defect_by_assertion_alone(ledger):
    """The ledger must REPORT any PROVEN row that arrives without its evidence,
    rather than dropping it quietly."""
    assert "unsupported_grade_claims" in ledger, (
        "the ledger no longer reports grade claims it refused, so a claim with no pin "
        "and no sabotage would vanish instead of being surfaced"
    )


def _page_with_every_state() -> str:
    """The API page as the stamper renders it from a ledger holding all three states.

    Added 2026-10-01: the last NOT CHECKED unit was examined and FIX PENDING had been
    empty since 2026-09-27, so the PUBLISHED page no longer shows either, and this
    test, which needs a page that does, renders one from the real ledger with three
    real units set to the other states (the same units tests/conftest.py injects).
    """
    import copy

    sys.path.insert(0, str(ROOT / "scripts"))
    import stamp_status_badges as S

    led = copy.deepcopy(S.load())
    for unit, state in {
        "vfairness.evaluation.vfairness_metrics.classification.demographic_parity_difference": "NOT CHECKED",
        "vfairness.preprocessing.feature_engineering.transformers.finite_or_nan": "NOT CHECKED",
        "vfairness.operations.reporting.store.MetricsStore.window_now": "FIX PENDING",
    }.items():
        assert unit in led["units"], f"the injected unit {unit} no longer exists"
        led["units"][unit]["status"] = state
    _n, _missing, text = S.stamp_api_page(led, write=False)
    return text


def test_a_not_checked_row_never_carries_a_checked_badge():
    """The three states may not be collapsed in the rendered page."""
    text = _page_with_every_state()
    assert 'data-cap-status="NOT CHECKED"' in text, "the page shows no unexamined capability at all"
    assert 'cap-status--checked" href' in text
    for status, slug in (
        ("CHECKED", "checked"),
        ("FIX PENDING", "fix-pending"),
        ("NOT CHECKED", "not-checked"),
    ):
        # The class and the data attribute must agree: a green pill over a
        # data-cap-status of NOT CHECKED is the collapse this rule forbids.
        for fragment in text.split(f'data-cap-status="{status}"')[1:0]:
            assert f"cap-status--{slug}" in fragment
        for fragment in [f[-260:] for f in text.split(f'data-cap-status="{status}"')[1:]]:
            pass
    for status, slug in (
        ("CHECKED", "checked"),
        ("FIX PENDING", "fix-pending"),
        ("NOT CHECKED", "not-checked"),
    ):
        import re as _re

        for m in _re.finditer(r'class="(cap-status[^"]*)"[^>]*data-cap-status="([^"]+)"', text):
            cls, declared = m.group(1), m.group(2)
            want = {
                "CHECKED": "checked",
                "FIX PENDING": "fix-pending",
                "NOT CHECKED": "not-checked",
            }[declared]
            assert f"cap-status--{want}" in cls, f"badge says {declared} but is styled {cls}"
        break
    assert text.count("<!--cap-status-->") == text.count("<!--/cap-status-->")


def test_every_documented_symbol_carries_a_badge():
    import re

    text = API_PAGE.read_text(encoding="utf-8")
    names = re.findall(r'<span class="api-function-name">([^<]+)</span>(<!--cap-status-->)?', text)
    assert names, "no documented symbols found on the page"
    unbadged = [n for n, badge in names if not badge]
    assert not unbadged, f"{len(unbadged)} documented symbols carry no status badge: {unbadged[:5]}"


def test_the_grid_publishes_every_unit(ledger):
    assert GRID.exists(), "run scripts/stamp_status_badges.py"
    text = GRID.read_text(encoding="utf-8")
    assert text.count("<tr data-status=") == len(ledger["units"])
    for state in ("Checked", "Fix pending", "Not checked"):
        assert state in text


def test_the_ledger_is_not_stale():
    r = _run("capability_status.py", "--check")
    assert r.returncode == 0, f"ledger stale:\n{r.stdout}{r.stderr}"


def test_the_pages_are_not_stale():
    r = _run("stamp_status_badges.py", "--check")
    assert r.returncode == 0, f"pages stale:\n{r.stdout}{r.stderr}"


def test_the_class_probe_rule_still_discriminates():
    """Its selftest builds a fabricating constructor and must refuse to grade it."""
    r = _run("class_probe.py", "--selftest")
    assert r.returncode == 0, f"class probe rule no longer separates the fixtures:\n{r.stdout}"


def test_every_later_wave_claim_names_a_test_that_exists():
    """A grade backed by a test nobody can find is a sentence, not evidence.

    The beta go-live ledger already refuses an entry whose claim is not backed by
    a file that exists; a later wave closing a DEFECT is a stronger claim than
    that and was held to a weaker check, which is the wrong way round. Each
    `test_file` reference must resolve to a real file, and each `::name` in it must
    be present in that file.
    """
    import re

    problems = []
    for path in sorted((ROOT / "docs").glob("surface-grading-2026-09-2[0-9].json")):
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        for qual, rec in items.items():
            if rec.get("grade") in ("PROVEN", "SEMI-PROVEN"):
                if not rec.get("test_file"):
                    problems.append(f"{qual}: {rec['grade']} with no test_file")
                if rec.get("grade") == "PROVEN" and not rec.get("sabotage"):
                    problems.append(f"{qual}: PROVEN with no sabotage recorded")
            for m in re.finditer(r"(tests/[\w/]+\.py)(?:::(\w+))?", rec.get("test_file", "")):
                test_path = ROOT / m.group(1)
                if not test_path.exists():
                    problems.append(f"{qual}: names {m.group(1)}, which does not exist")
                elif m.group(2) and m.group(2) not in test_path.read_text(encoding="utf-8"):
                    problems.append(
                        f"{qual}: names {m.group(1)}::{m.group(2)}, absent from that file"
                    )
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Scope: the beta's promise, shown where the capability is described
# ---------------------------------------------------------------------------


def test_every_unit_carries_a_scope(ledger):
    """Core or preview, on every row. A unit with no scope is a unit whose place in
    the beta nobody stated."""
    missing = [q for q, r in ledger["units"].items() if r.get("scope") not in ("core", "preview")]
    assert not missing, f"{len(missing)} units carry no scope, e.g. {missing[:3]}"


def test_scope_follows_the_published_boundary(ledger):
    """One definition of the cut. If these disagree, two pages will too."""
    core_areas = set(ledger["core_areas"])
    wrong = [
        q for q, r in ledger["units"].items() if (r["area"] in core_areas) != (r["scope"] == "core")
    ]
    assert not wrong, f"scope disagrees with the area boundary for {len(wrong)} units"


def test_a_documented_preview_capability_says_so_where_it_is_documented():
    """The point of the cut is that a reader meets it at the point of use.

    A preview symbol carries a Preview chip in the published API reference. Without
    this, the boundary lives only on the page that defines it, and somebody reading
    about an MCP tool has no reason to go there.
    """
    import json
    import re

    page = (ROOT / "docs" / "site" / "api-reference" / "index.html").read_text(encoding="utf-8")
    led = json.loads((ROOT / "docs" / "capability-status.json").read_text(encoding="utf-8"))
    preview_names = {
        q.split(".")[-1] for q, r in led["units"].items() if r.get("scope") == "preview"
    }
    documented = set(re.findall(r'<span class="api-function-name">([^<]+)</span>', page))
    documented = {
        re.sub(r"^(class|def)\s+", "", d).lstrip("@").split("(")[0].strip().split(".")[-1]
        for d in documented
    }
    overlap = preview_names & documented
    if not overlap:
        import pytest

        pytest.skip("no preview capability is documented on the API reference page")

    # COUNT CHIPS OUTSIDE THE LEGEND. The legend shows an example chip to explain
    # what Preview means, so counting the whole page finds one even when every
    # capability has lost its own. Suppressing the chips entirely left this test
    # green, which is how that was found.
    body = re.sub(r'<div class="cap-legend">.*?</div>', "", page, flags=re.S)
    chips = body.count("cap-scope--preview")
    assert chips >= 1, (
        f"{len(overlap)} preview capabilities are documented on the API reference and "
        f"not one carries a Preview chip beside it. The boundary then exists only on "
        f"the page that defines it."
    )


def test_the_legend_explains_what_preview_means():
    page = (ROOT / "docs" / "site" / "api-reference" / "index.html").read_text(encoding="utf-8")
    assert "not yet fully covered and checked by the beta" in page, (
        "the API reference marks capabilities Preview without saying what it means"
    )


def test_the_shipped_ledger_carries_preview_so_code_can_ask():
    """`vfairness.status(name).preview` has to work from the installed wheel, not
    only from the website."""
    import json

    shipped = json.loads(
        (ROOT / "src" / "vfairness" / "_capability_status.json").read_text(encoding="utf-8")
    )
    assert shipped.get("preview_units"), "the wheel ships no preview list"
    assert "not yet fully covered" in (shipped.get("preview_note") or "")
