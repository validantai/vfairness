"""The readiness verdict must be a measurement, not a sentence somebody writes.

For two days the status reported to the owner was "publishable, the 215
capabilities have zero open defects". True, and a SCOPED CLAIM: it named the
flattering part of the surface and omitted 95 open defects our own grading waves
had measured in the rest of it. Scoping a claim to the subset that looks good is
the technique this library was audited to eliminate. These tests stop the gate
from being quietly widened back into that shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "release_gate.py"


def _run() -> dict:
    out = subprocess.run(
        [sys.executable, str(GATE), "--json"], capture_output=True, text=True, cwd=str(ROOT)
    )
    return json.loads(out.stdout)


def test_the_gate_fails_closed_when_anything_is_open() -> None:
    """READY may only be reported when every criterion passes."""
    v = _run()
    passes = [c["passes"] for c in v["criteria"]]
    assert v["ready_to_publish"] is all(passes)
    assert v["verdict"] == ("READY" if all(passes) else "NOT READY")


def test_the_gate_covers_the_whole_surface_not_only_the_census() -> None:
    """The specific failure this file exists to prevent.

    A gate whose only criterion is the capability census would report READY
    today, with 95 measured open defects in the library. At least one criterion
    must be about the surface OUTSIDE the census.
    """
    v = _run()
    ids = {c["id"] for c in v["criteria"]}
    assert {"G1", "G2", "G3"} <= ids, "the gate lost its whole-surface criteria"
    census_only = [c for c in v["criteria"] if c["id"] == "G5"]
    assert census_only and census_only[0]["passes"], (
        "G5 is the control: if the census criterion itself fails, this test is "
        "not measuring what it claims to"
    )
    surface = [c for c in v["criteria"] if c["id"] in {"G1", "G2", "G3", "G4"}]
    assert len(surface) == 4
    assert not all(c["passes"] for c in surface), (
        "every whole-surface criterion passes while the census passes too, so "
        "the gate can no longer tell the two apart; if that is genuinely true "
        "the library is ready and this assertion is the thing to revisit"
    )


def test_an_unmeasurable_criterion_is_a_failure() -> None:
    """Fail closed. 'Cannot measure' must never read as a pass."""
    v = _run()
    for c in v["criteria"]:
        if "cannot measure" in str(c["measured"]).lower():
            assert not c["passes"], f"{c['id']} reports unmeasurable AND passes"


def test_every_criterion_says_why_it_exists() -> None:
    """A gate that states a rule but not its reason is the two-state failure in
    prose form: the next person cannot tell a deliberate bar from an accident."""
    v = _run()
    for c in v["criteria"]:
        assert len(c.get("why", "")) > 80, f"{c['id']} does not say why it exists"


def test_the_exit_code_matches_the_verdict() -> None:
    """CI reads the exit code, not the text."""
    out = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True, cwd=str(ROOT))
    v = _run()
    assert (out.returncode == 0) is v["ready_to_publish"]


def test_there_are_two_tiers_and_the_beta_one_is_reachable() -> None:
    """A bar nobody can reach stops working as a bar.

    The full gate asks for every code unit graded and zero open defects
    anywhere, which is tens of millions of tokens of machine work. Publishing
    held to that is publishing never. The beta tier exists so the honest
    minimum is still measured rather than argued, and it must be strictly
    weaker than the full gate or it is not a second tier at all.
    """
    v = _run()
    assert "beta_criteria" in v and v["beta_criteria"], "the beta tier is gone"
    # B1b added 2026-09-25: B1 named two things (executed AND recorded) that stopped
    # having the same answer once every unit carried a published state, so the gate
    # reports them separately. B1 keeps its strict verdict; nothing was relaxed.
    # B2b added 2026-09-28, and it makes the beta bar STRICTLY HARDER, which is the
    # direction a new criterion is allowed to move it. B2 counts DEFECT OPEN rows in
    # the grading waves. The second-round audit's findings are not in those waves,
    # they are in tests/test_bgl6_*.py, so the gate printed "B2: 0 open" while 35
    # execution-proven defects were open, among them a probe clearing a scaffolded
    # model and a benchmark certifying a model that decided nothing. A gate that
    # cannot see a whole body of findings reports a pass it has not earned.
    # B6 and B7 added 2026-10-01, again STRICTLY HARDER: the two executable checks
    # (same answer as an established library; no invented number on broken data)
    # became conditions of this gate instead of a second gate with its own answer.
    # They are pinned to fail closed in tests/test_gate_checks_are_beta_conditions.py.
    # B4 left the beta tier the same day (Daniel's decision): second-examiner
    # confirmation of every grade is a 1.0 condition, G2 in the full gate, because
    # B6 and B7 re-check every number-returning capability by execution.
    assert set(c["id"] for c in v["beta_criteria"]) == {
        "B1",
        "B1b",
        "B2",
        "B2b",
        "B3",
        "B5",
        "B6",
        "B7",
    }
    # And it is a REAL criterion rather than a decorative one: it reads the register,
    # so it moves when the register moves.
    second = next(c for c in v["beta_criteria"] if c["id"] == "B2b")
    assert "recorded claims" in second["measured"], second
    # Weaker, not different: the full gate demands zero open defects anywhere,
    # the beta tier only in code that can hand a user a false clean bill.
    beta_open = next(c for c in v["beta_criteria"] if c["id"] == "B2")
    full_open = next(c for c in v["criteria"] if c["id"] == "G3")
    # THE INVARIANT, NOT TODAY'S COUNTS. This line read
    # `assert not full_open["passes"]`, a control written while open defects existed,
    # and on 2026-09-30 the count reached zero, G3 correctly began passing, and the
    # control fired. It was describing the state of the world at writing time as
    # though it were the subject. The subject is that the beta tier is strictly
    # WEAKER than the full gate: G3 counts every DEFECT OPEN row, B2 counts only the
    # subset in code that returns a measurement. A subset cannot hold more than the
    # whole, so zero open ANYWHERE must imply zero open among the measuring subset.
    # That holds at any data, including after the pile is empty, which is exactly
    # when the old form would have had someone "fix" the gate by making it fail again.
    if full_open["passes"]:
        assert beta_open["passes"], (
            "the full gate sees no open defect anywhere while the beta tier still "
            "reports one, so the beta scope is not a subset of the full scope and one "
            "of the two is counting the wrong rows"
        )
    assert "measuring" in beta_open["measured"], (
        "the beta criterion no longer splits open defects by whether the code "
        "returns a measurement, so it is either the full gate again or it is "
        "letting dangerous defects through"
    )


def test_the_plan_states_the_remaining_work() -> None:
    """The published plan must carry the real number of ungraded code units.

    A plan quoting a figure the repository contradicts is worth less than no
    plan, because it reads as measured.
    """
    plan = ROOT / "docs" / "RELEASE_PLAN.md"
    assert plan.exists(), "the release plan is not published"
    text = plan.read_text(encoding="utf-8")
    v = _run()
    g1 = next(c for c in v["criteria"] if c["id"] == "G1")
    graded, total = (int(x) for x in g1["measured"].replace(" of ", " ").split())
    remaining = total - graded
    assert str(remaining) in text, (
        f"RELEASE_PLAN.md does not state the {remaining} code units that remain"
    )


def test_b1_is_a_union_and_never_a_sum() -> None:
    """The gate must not add two overlapping sets and publish the total.

    B1 previously computed `probe_reached + graded` and printed 1,524 of 1,546,
    which is 98.6% examined. Those sets overlap by 307, because the probe runs
    over everything the census had not stamped and the surface waves had already
    graded 420 of those. The union is 1,217, or 78.7%. An independent
    documentation review found it, not the gate's own tests.

    That defect is this library's own, committed inside the readiness gate: a
    number nobody measured, wrong in the flattering direction, on the surface
    that exists to prevent exactly that. This test is the pin.
    """
    import json as _json

    root = ROOT
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "src"))
    import library_kpis  # noqa: PLC0415

    surf = library_kpis._walk_public_surface()
    census = (
        set(surf["functions"]["graded_names"])
        | set(surf["classes"]["graded_names"])
        | set(surf["methods"]["graded_names"])
    )
    grading_file = root / "docs" / "surface-grading-2026-09-18.json"
    waves = (
        set(_json.loads(grading_file.read_text(encoding="utf-8"))["items"])
        if grading_file.exists()
        else set()
    )
    probe_file = root / "docs" / "surface-probe.json"
    reached = set()
    if probe_file.exists():
        data = _json.loads(probe_file.read_text(encoding="utf-8"))
        # The SAME definition the gate uses: the probe must have LEARNED
        # something, which means the healthy run produced a value. A unit that
        # raised on every world told us only that we called it wrong.
        reached = {
            q
            for q, v in data.items()
            if ((v.get("runs") or {}).get("healthy") or {}).get("outcome")
            not in (None, "NOT_REACHED", "raised", "hung")
        }

    # THE OTHER THREE SOURCES, added 2026-09-25 when the gate started counting them.
    # Recomputed HERE from the raw files rather than read back from the gate: a test
    # that mirrors the implementation passes whatever the implementation does, and
    # the whole point of this one is to recompute the union independently.
    #
    # Why they belong: this criterion asks whether a unit was EXECUTED and its result
    # recorded. class_probe.py constructs every ungraded class on nine worlds,
    # xai_probe.py calls the explainability surface against a fitted model, and the
    # test suite executes most of the surface on every push. The gate counted the
    # FUNCTION probe and none of those three, which is how it reported 920 examined
    # while the suite alone executes over 1,300.
    probed = set()
    for name in ("class-probe.json", "xai-probe.json"):
        path = root / "docs" / name
        if not path.exists():
            continue
        for qual, rec in _json.loads(path.read_text(encoding="utf-8")).items():
            # NOT EXAMINED is the probe saying it could not reach the unit. It is
            # recorded as such and must not count as a run.
            if rec.get("grade") and rec["grade"] != "NOT EXAMINED":
                probed.add(qual)

    suite = set()
    suite_file = root / "docs" / "suite-coverage.json"
    if suite_file.exists():
        record = _json.loads(suite_file.read_text(encoding="utf-8"))
        run = record.get("suite") or {}
        # Fail closed on the suite's own result, the same rule the gate applies:
        # lines executed on the way to a failing assertion verified nothing.
        if run.get("recorded") and not run.get("failed") and not run.get("errors"):
            suite = set(record.get("executed") or ())

    # EVERY dated wave, discovered the same way the gate discovers them. This was
    # `surface-grading-2026-09-2[0-9].json`, which cannot match a second wave on one
    # day: `surface-grading-2026-09-25b.json` was ignored here and in both registries
    # at once, and this test then reported the gate as publishing 1,564 against a
    # union of 1,474 that was missing 94 units it should have held. A narrow pattern
    # in an independent recomputation is not independence, it is the same bug twice.
    later_waves = set()
    _base = {"surface-grading-2026-09-18.json", "surface-grading-from-probe.json"}
    for path in sorted((root / "docs").glob("surface-grading-*.json")):
        if path.name in _base:
            continue
        later_waves |= set(_json.loads(path.read_text(encoding="utf-8"))["items"])

    parts = (census, waves, later_waves, reached, probed, suite)
    union = len(set().union(*parts))
    naive = sum(len(p) for p in parts)

    v = _run()
    b1 = next(c for c in v["beta_criteria"] if c["id"] == "B1")
    published = int(str(b1["measured"]).split()[0])

    # WHAT B1 PUBLISHES CHANGED ON 2026-09-25, from the size of the union to the
    # number of SURFACE units the union covers, and this test follows it because the
    # new quantity is the stricter one. The union is keyed by qualified name and some
    # of those names are re-export spellings that the surface walk does not count as
    # units, so it legitimately overshoots the total (measured: 1,569 names against
    # 1,564 units). Comparing a count to the total could therefore have read PASS with
    # up to five real units unexamined. The invariant that matters is unchanged: B1 is
    # never a SUM, which the anti-vacuity assertion below still holds it to.
    surface_units = {
        qual
        for kind in ("functions", "classes", "methods")
        for qual in surf[kind]["ungraded"] + surf[kind]["graded_names"]
    }
    covered = len(surface_units & set().union(*parts))
    assert published == covered, (
        f"B1 publishes {published}; independently recomputed, the union covers "
        f"{covered} of the {len(surface_units)} surface units (the raw union holds "
        f"{union} names, including re-export spellings)"
    )
    assert published <= len(surface_units), (
        f"B1 publishes {published}, which exceeds the {len(surface_units)} units on "
        f"the surface: it is counting names that are not units again"
    )
    # Anti-vacuity: if the sets ever stop overlapping this test proves nothing,
    # so say so rather than passing quietly.
    assert naive > union, (
        "the graded, probed and suite-executed sets no longer overlap, so a sum and a "
        "union agree and this test can no longer detect the defect it exists "
        "for; re-derive it against whatever the new sets are"
    )


def test_an_optional_extra_is_not_reported_as_a_broken_module() -> None:
    """Three states for an import, not two.

    "This module is broken" and "this module needs an optional extra nobody
    installed" are different facts. vfairness.mcp.server was published for two
    days as "a module that could not be imported", which reads as a fault. It
    refuses cleanly with an install hint, and with the extra present it works:
    proved by driving it over the MCP protocol end to end, where it measured a
    real disparity and returned assessable=false with a reason on a single-group
    frame.

    Any import failure the surface scan records must say which of the two it is.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import library_kpis  # noqa: PLC0415

    surf = library_kpis._walk_public_surface()
    for f in surf.get("import_failures", []):
        assert f.get("kind") in {"optional extra not installed", "import failed"}, (
            f"{f.get('module')} records no import state, so a missing optional "
            "dependency is indistinguishable from a broken module"
        )


def test_the_surface_count_records_the_environment_that_produced_it() -> None:
    """A surface count with no record of its environment cannot be reproduced.

    Measured 2026-09-18: without the optional mcp extra the package presents
    1,546 code units across 204 modules; with it present, 1,558 across 205.
    Every figure downstream inherits that difference, so the extras that were
    importable at measurement time are published beside the count.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import library_kpis  # noqa: PLC0415

    surf = library_kpis._walk_public_surface()
    extras = surf.get("optional_extras_present")
    assert isinstance(extras, dict) and extras, "the surface count names no environment"
    assert all(isinstance(v, bool) for v in extras.values())
    assert "mcp" in extras, "the extra that moved the count is not recorded"


def test_a_call_that_only_raised_does_not_count_as_examined() -> None:
    """A probe call that learned nothing is not evidence of anything.

    Measured 2026-09-18: 287 code units raised on every world INCLUDING the
    healthy one, with errors like "'numpy.ndarray' object has no attribute
    'keys'" and "could not convert string to float: 'a'". The probe's generic
    argument binder had called them with nonsense. Their refusals say nothing
    about whether they refuse honestly, because a function that raises on good
    data and on bad data alike has told us only that we called it wrong.

    Counting those as examined put B1 at 1,217 of 1,558 instead of 890. This is
    the third time a figure on this surface has been overstated in the
    flattering direction, which is why it is pinned rather than trusted.
    """
    import json as _json

    probe = ROOT / "docs" / "surface-probe.json"
    if not probe.exists():
        pytest.skip("no probe evidence in this tree")
    data = _json.loads(probe.read_text(encoding="utf-8"))

    any_call = {
        q
        for q, v in data.items()
        if any(r.get("outcome") != "NOT_REACHED" for r in (v.get("runs") or {}).values())
    }
    learned = {
        q
        for q, v in data.items()
        if ((v.get("runs") or {}).get("healthy") or {}).get("outcome")
        not in (None, "NOT_REACHED", "raised", "hung")
    }
    assert learned < any_call, (
        "every probed unit produced a value on healthy input, so this check can "
        "no longer tell a real examination from an argument error; re-derive it"
    )

    dropped = len(any_call) - len(learned)
    assert dropped >= 200, (
        f"only {dropped} probed units are excluded for having raised on healthy "
        "input; the measured figure on 2026-09-18 was 287, so either the probe's "
        "argument binder improved a great deal or this check has stopped "
        "discriminating"
    )


def test_a_script_may_never_assign_proven() -> None:
    """PROVEN requires a test that fails when the fix is removed. No script can
    write one, so no script may award that grade.

    scripts/grade_from_probe.py assigns SEMI-PROVEN and nothing else. Its own
    first draft would have awarded 316 grades; the rule that requires the
    healthy run to produce a value cut that to 29, because 287 of them were
    argument errors wearing the costume of an honest refusal.
    """
    import json as _json

    f = ROOT / "docs" / "surface-grading-from-probe.json"
    if not f.exists():
        pytest.skip("no probe-assigned grades in this tree")
    d = _json.loads(f.read_text(encoding="utf-8"))
    grades = {r["grade"] for r in d["items"].values()}
    assert grades <= {"SEMI-PROVEN"}, f"a script awarded {grades - {'SEMI-PROVEN'}}"
    assert all(r.get("audited") is False for r in d["items"].values()), (
        "a probe-assigned grade is marked audited, but no agent argued with it"
    )
    assert d["rule"]["healthy_run_must_produce_a_value"] is True, (
        "the clause that cut 316 qualifying units to 29 has been removed"
    )


def test_a_zero_count_is_reported_as_zero_and_not_as_could_not_check():
    """A tally key vanishes when its count reaches zero.

    `by_grade` counts the grades that OCCUR, so closing the last DEFECT OPEN row
    removes the key and `.get()` returns None. G3 then published "cannot measure"
    and G4 with it, when the measured answer was ZERO OPEN. Found 2026-09-30, the
    first time this campaign actually reached zero, which is why it had never
    shown: the same function already seeded its sibling tally against exactly this
    trap, with a comment saying why, and this one was left unseeded.

    A gate that cannot say zero eventually says "I do not know" about its own
    success, and a reader cannot tell that from a real could-not-check.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gate_zero_probe", ROOT / "scripts" / "release_gate.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    by_id = {c["id"]: c for c in gate._criteria()}
    for cid in ("G3", "G4"):
        assert by_id[cid]["measured"] != "cannot measure", (
            f"{cid} reports could-not-check while the register holds no open defect; "
            "an absent tally key is being read as an unknown rather than as zero"
        )
    assert by_id["G3"]["measured"] == "0 open", by_id["G3"]["measured"]
    assert by_id["G3"]["passes"] is True


def test_a_real_could_not_check_is_still_reported_as_one():
    """OVER-CORRECTION CONTROL. Seeding a tally must not turn every unknown into a
    zero: G2 reads a different field, and if THAT is absent the gate must still say
    it could not measure rather than claiming nothing is unaudited."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gate_unknown_probe", ROOT / "scripts" / "release_gate.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    original = gate.library_kpis.grading

    def without_unaudited():
        g = dict(original())
        g.pop("unaudited", None)
        return g

    gate.library_kpis.grading = without_unaudited
    try:
        by_id = {c["id"]: c for c in gate._criteria()}
        assert by_id["G2"]["measured"] == "cannot measure", by_id["G2"]["measured"]
        assert by_id["G2"]["passes"] is False
    finally:
        gate.library_kpis.grading = original
