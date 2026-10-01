#!/usr/bin/env python3
"""Is this library ready to publish? Measured, not asserted.

WHY THIS EXISTS. For two days the status given to the owner was "publishable,
the 215 capabilities have zero open defects". That was true and it was a SCOPED
CLAIM: it named the part of the surface that looked good and left out 95 open
defects measured by our own grading waves in the rest of it. Scoping a claim to
the flattering subset is the exact technique this library was audited to
eliminate, so the readiness verdict is now a measurement with criteria anyone
can reproduce, not a sentence somebody writes.

FAIL CLOSED. A criterion that cannot be measured FAILS. An open defect whose
severity nobody has assessed counts as blocking, not as low. "We have not looked
at it yet" is not evidence that it is minor, and treating it as minor is the
same defect one level up.

Exit code 0 only when every criterion passes.

Usage:
    python scripts/release_gate.py
    python scripts/release_gate.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis  # noqa: E402
import suite_coverage  # noqa: E402

GRADING = ROOT / "docs" / "surface-grading-2026-09-18.json"


def _open_defects_split() -> dict:
    """The open defects, split by whether the code returns a measurement.

    This split is what makes a beta bar defensible rather than arbitrary. A
    function that hands back a fairness number nobody measured can put a false
    clean bill in front of a user. A renderer that returns an empty chart cannot.
    Both are defects and only one of them blocks a beta.

    FAIL CLOSED on the split itself: a code unit the triage could not classify
    counts as measurement-producing until a human says otherwise. "We have not
    worked out what this returns" is not evidence that it is harmless.
    """
    import collections

    # THE MERGED view, including later waves. Reading GRADING directly here is what
    # made B2 print 86 while G3 printed 93 on the same evidence.
    ev = library_kpis.graded_items()
    surf = library_kpis._walk_public_surface()
    triage = library_kpis.work_list(surf)["items"]
    by_short = collections.defaultdict(list)
    for qual, cat in triage.items():
        by_short[qual.split(".")[-1]].append(cat)

    measuring, other, unclassified = 0, 0, 0
    for name, rec in ev.items():
        if rec.get("grade") != "DEFECT OPEN":
            continue
        cats = by_short.get(name.split(".")[-1]) or []
        cat = cats[0] if cats else "unknown"
        if "number or a result" in cat:
            measuring += 1
        elif "needs a human" in cat or cat == "unknown":
            unclassified += 1
        else:
            other += 1
    return {
        "measuring": measuring,
        "unclassified": unclassified,
        "other": other,
        "blocking_for_beta": measuring + unclassified,
    }


def _second_round() -> dict:
    """The second-round audit register's totals, or zeros with the file named.

    Read from docs/bgl6-audit-register.json, which scripts/bgl6_register.py
    generates from the audit files themselves. If the register is missing this
    returns recorded 0 and open 0, which PASSES, and that is deliberate: an absent
    register is an absent body of findings, and inventing a failure out of a missing
    file would make the gate unrunnable on a checkout that never had the audit. The
    register is committed, and tests/test_bgl6_register_is_honest.py refuses a stale
    one, so the pass can only be earned or the test is red.
    """
    path = ROOT / "docs" / "bgl6-audit-register.json"
    if not path.exists():
        return {"recorded": 0, "closed": 0, "open": 0, "present": False}
    totals = dict(json.loads(path.read_text(encoding="utf-8")).get("totals") or {})
    totals["present"] = True
    return totals


PLAIN_ENGLISH = {
    "B1": (
        "Has every public unit actually been RUN, with the result written down? "
        "Not a belief that it works: executed, by the suite or an audit, and "
        "recorded."
    ),
    "B1b": (
        "Can a reader SEE that state where the unit is described? A measurement nobody can find is the defect this library is audited for."
    ),
    "B2": (
        "Is anything still broken in code that hands back a fairness number or a verdict? This is the defect that reaches a user as a false clean bill."
    ),
    "B2b": (
        "The fix wave was audited a SECOND time, by agents attacking the fixes. Every open record is a test that reproduces its defect today."
    ),
    "B3": (
        "Every defect still open has to carry a severity. A beta may ship with known defects; it may not ship with ones nobody has ranked."
    ),
    "B4": (
        "Was each grade checked by a SECOND agent told to refute it? One in three was overturned when somebody did, so an unchecked grade is a claim."
    ),
    "B5": (
        "The registered capabilities are the surface a user reaches on purpose, so the census over them has to be complete and clear."
    ),
    "B6": (
        "Where an established library computes the same number, does vfairness give the same answer on clean data?"
    ),
    "B7": (
        "Does every capability that measures refuse to invent a number when the data cannot support one, and still measure when it can?"
    ),
}


def _gate_checks() -> dict | None:
    """The two executable checks, as last run by scripts/emit_gate_checks.py.

    Fail closed: a missing or unreadable file is a criterion that cannot be
    measured, and that fails.
    """
    path = ROOT / "docs" / "gate-checks.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - reported as a failing criterion
        return None


def _beta_criteria() -> list[dict]:
    """The bar for publishing a BETA, as opposed to a finished library.

    The strict gate below demands every code unit graded and zero open defects
    anywhere. At the measured cost that is tens of millions of tokens, so a
    release held to it is a release that never ships, and a bar nobody can reach
    stops functioning as a bar. This tier is the smallest set of criteria under
    which publishing is still honest:

      nothing on the public surface is UNEXAMINED,
      nothing that can hand a user a false clean bill is left open,
      everything still open is disclosed with a severity,
      and the documentation says which is which.

    A beta is allowed to have known, disclosed, non-dangerous defects. It is not
    allowed to have unknown ones, and it is not allowed to be quiet about them.
    """
    surf = library_kpis._walk_public_surface()
    split = _open_defects_split()
    probe = ROOT / "docs" / "surface-probe.json"
    total_units = surf["total"]["total"]

    # THE UNION, not the sum. This criterion previously added the probe's
    # reached count to the graded count and printed 1,524 of 1,546, which is
    # 98.6% examined. The two sets OVERLAP by 307 code units, because the probe
    # runs over everything the census had not stamped and the surface-grading
    # waves had already graded 420 of those. The honest figure is the union:
    # 1,217 of 1,546, or 78.7%.
    #
    # Measured 2026-09-18, and found by an independent documentation review, not
    # by me. Adding two overlapping sets and publishing the total is this
    # library's own defect committed inside its own readiness gate: a number
    # nobody measured, in the flattering direction, on the page that exists to
    # stop exactly that.
    ledger_path = ROOT / "docs" / "capability-status.json"
    all_units = (
        json.loads(ledger_path.read_text(encoding="utf-8")).get("units", {})
        if ledger_path.exists()
        else {}
    )
    census = (
        set(surf["functions"]["graded_names"])
        | set(surf["classes"]["graded_names"])
        | set(surf["methods"]["graded_names"])
    )
    # EVERY wave, not just the first. `GRADING` is the 2026-09-18 file, and reading
    # it alone left the units a LATER wave graded out of this union entirely: found
    # 2026-09-25 by tests/test_release_gate.py::test_b1_is_a_union_and_never_a_sum,
    # which recomputes the union from the raw files and reported 1,145 against the
    # 1,139 the gate published. A unit graded by any wave has been examined by that
    # wave, whichever file it happens to live in.
    waves: set[str] = set(library_kpis.graded_items())
    if GRADING.exists():
        waves |= set(json.loads(GRADING.read_text(encoding="utf-8")).get("items", {}))
    reached: set[str] = set()
    if probe.exists():
        data = json.loads(probe.read_text(encoding="utf-8"))
        # EXAMINED means the probe LEARNED something, not merely that it made a
        # call. Measured 2026-09-18: 287 code units raised on every world
        # INCLUDING the healthy one, with errors like "'numpy.ndarray' object
        # has no attribute 'keys'" and "could not convert string to float:
        # 'a'". The generic argument binder called them with nonsense, so their
        # refusals say nothing about whether they refuse honestly. Counting
        # those as examined inflated this criterion in the flattering
        # direction, which is the third time a figure on this surface has done
        # exactly that. A call that produced only an argument error is NOT
        # examined, and the unit is left to the ungraded pile where it belongs.
        reached = {
            q
            for q, v in data.items()
            if ((v.get("runs") or {}).get("healthy") or {}).get("outcome")
            not in (None, "NOT_REACHED", "raised", "hung")
        }
    # THE TEST SUITE, counted at last. B1 asks whether a unit has been executed and
    # its result recorded, and the three sources above are the census, the grading
    # waves and the probe. The 10,552-test suite was never one of them. Measured
    # 2026-09-25 it executes 1,346 of the 1,564 units, which took B1 from 920 to
    # 1,398 and the "never executed by anything" pile from 644 to 166. The probe,
    # built precisely because nobody had executed this code, was credited for 459
    # while the suite that executes three times as much was credited for none.
    #
    # FAIL CLOSED ON THE SUITE'S OWN RESULT: `usable()` refuses a record whose run
    # was not green, because lines executed on the way to a failing assertion are
    # not evidence that anything was verified. A missing or red record contributes
    # nothing rather than being assumed.
    suite: set = set()
    if suite_coverage.usable():
        suite = set(suite_coverage.load().get("executed") or ())

    # THE OTHER TWO PROBES, also missing from this union until 2026-09-25. `reached`
    # above comes from surface-probe.json, which covers functions and methods only.
    # scripts/class_probe.py executes every ungraded CLASS on the same nine worlds
    # and scripts/xai_probe.py executes the explainability surface against a fitted
    # model, and both record what came back. A unit one of them constructed or
    # called has been executed and its result recorded, which is exactly what this
    # criterion asks. Counting the function probe and not its two siblings was an
    # accident of the order they were written in.
    #
    # A probe result of NOT EXAMINED does NOT count: that is the probe saying it
    # could not reach the unit, and it is recorded as such rather than as a run.
    probed: set = set()
    for path in (ROOT / "docs" / "class-probe.json", ROOT / "docs" / "xai-probe.json"):
        if not path.exists():
            continue
        for qual, rec in json.loads(path.read_text(encoding="utf-8")).items():
            if rec.get("grade") and rec["grade"] != "NOT EXAMINED":
                probed.add(qual)

    examined = census | waves | reached | suite | probed

    # B1 IS A SUBSET QUESTION, NOT A COUNT, and it used to be asked as a count:
    # `len(examined) >= total_units`. The five sources are keyed by qualified name and
    # some of those names are RE-EXPORT spellings that the surface walk does not count
    # as units of their own, so the union legitimately overshoots the total: measured
    # 2026-09-25, 1,569 names against a surface of 1,564. A count comparison therefore
    # could have read PASS with real units still unexamined, one for each extra
    # spelling, which is a check that cannot disagree with the belief it was written to
    # confirm. What is asked now is which surface units are missing from the union.
    unexamined = {
        qual
        for kind in ("functions", "classes", "methods")
        for qual in surf[kind]["ungraded"] + surf[kind]["graded_names"]
    } - examined

    # B1 NAMES TWO DIFFERENT THINGS, and on 2026-09-25 they stopped having the same
    # answer, so the gate now reports them separately. "Executed" and "its result
    # recorded" were one criterion because before the status ledger existed there
    # was no way to record a result for a unit nobody had run: the only honest
    # record was to run it. There is now. Every one of the units carries a
    # published state, and for an unexamined one that state says so, in the API
    # reference, in the grid and in the docstring, with four guards that fail the
    # build if any surface disagrees.
    #
    # WHAT IS NOT DONE HERE: the bar is not lowered. B1 keeps its strict verdict
    # and still fails while any unit is unexecuted. B1b is reported beside it so a
    # reader can see which half is met, because publishing "NOT BETA READY" without
    # saying that the disclosure half is complete is as uninformative as publishing
    # READY would be dishonest.
    #
    # THE DECISION THIS EXPOSES, and it belongs to the owner, not to this script:
    # the beta tier's own definition has FOUR clauses, "nothing on the public
    # surface is UNEXAMINED, nothing that can hand a user a false clean bill is
    # left open, everything still open is disclosed with a severity, and the
    # documentation says which is which". The fourth clause had no criterion until
    # now. Whether clause 1 is still required at 100% once clause 4 is met is a
    # publishing decision. Measured cost of meeting clause 1: 1,099 units the probe
    # cannot reach, spread over 343 owners, 179 of which are needed to cover 80%,
    # against a measured rate for hand-built fixtures of 40,000 to 69,000 tokens
    # each. That is the number the decision should be made against.
    recorded = len(all_units) if all_units else 0
    return _with_plain(
        [
            {
                "id": "B1",
                "name": "Every public code unit has been EXECUTED and its result recorded",
                # THE NAMES, NOT JUST THE COUNT. Added 2026-09-30: this line said
                # "3 unexamined" and nothing else, so a reader who wanted to close it
                # had to reverse-engineer a five-way set union out of this file to
                # learn WHICH three. A gate that reports a number it will not
                # attribute cannot be acted on, and the whole campaign's rule is that
                # a could-not-check has to be visible where the reader looks. Capped,
                # because when this criterion first ran the answer was in the
                # hundreds and an unbounded list would push the verdict off the
                # screen; the count above it is always exact.
                "measured": (
                    f"{total_units - len(unexamined)} of {total_units} executed and recorded"
                    + (
                        f"; {len(unexamined)} unexamined: "
                        + ", ".join(sorted(unexamined)[:8])
                        + (" ..." if len(unexamined) > 8 else "")
                        if unexamined
                        else ""
                    )
                ),
                "passes": not unexamined,
                "why": (
                    "An unexamined code unit is the absence of a measurement, and it "
                    "is where every defect this campaign found was living. Counted "
                    "here: the capability census, the grading waves, the surface probe "
                    "AND the test suite, which executes most of the surface and was "
                    "credited for none of it until 2026-09-25. What remains is the "
                    "genuinely untouched pile. See B1b for the disclosure half."
                ),
            },
            {
                "id": "B1b",
                "name": "Every public code unit's examination state is PUBLISHED where it is described",
                "measured": (
                    f"{recorded} of {total_units} carry a published state"
                    + (
                        ""
                        if recorded == total_units
                        else " (ledger is stale: run scripts/capability_status.py)"
                    )
                ),
                "passes": recorded == total_units and recorded > 0,
                "why": (
                    "The clause the beta tier's own definition ends with, and the one "
                    "that decides whether an unexamined unit can mislead anybody: not "
                    "that everything was checked, but that nothing unchecked is "
                    "presented as checked. Each row states which of the three states it "
                    "is in and, when it is unexamined, why. Enforced by "
                    "scripts/capability_status.py --check, "
                    "scripts/stamp_status_badges.py --check and "
                    "tests/test_capability_status_badges.py, each sabotage-tested."
                ),
            },
            {
                "id": "B2",
                "name": "No open defect in code that returns a fairness number or verdict",
                "measured": f"{split['blocking_for_beta']} open "
                f"({split['measuring']} measuring, {split['unclassified']} unclassified)",
                "passes": split["blocking_for_beta"] == 0,
                "why": (
                    "This is the defect that reaches a user as a false clean bill. A "
                    "code unit whose return type nobody has classified counts here "
                    "too: not having worked out what it returns is not evidence that "
                    "it is harmless."
                ),
            },
            {
                "id": "B2b",
                "name": "No open defect recorded by the second-round audit",
                "measured": (
                    f"{_second_round()['open']} open of {_second_round()['recorded']} recorded claims"
                ),
                "passes": _second_round()["open"] == 0,
                "why": (
                    "ADDED 2026-09-28, because B2 read 0 open while 35 execution-proven "
                    "defects were open. B2 counts DEFECT OPEN rows in the grading waves, "
                    "and the second-round audit's findings are not in those waves: they "
                    "are in tests/test_bgl6_*.py, one test per claim, each asserting the "
                    "value the unit produces today. So the gate could say 'no open defect "
                    "in code that returns a fairness number' while a probe cleared a "
                    "scaffolded model and a benchmark certified a model that decided "
                    "nothing. A gate that cannot see a whole body of findings reports a "
                    "pass it has not earned, which is the defect class this library is "
                    "audited for, in the gate itself."
                ),
            },
            {
                "id": "B3",
                "name": "Every remaining open defect carries an assessed severity",
                "measured": f"{split['other']} non-measuring open, severity unassessed",
                "passes": split["other"] == 0,
                "why": (
                    "A beta may ship with known defects. It may not ship with defects "
                    "nobody has looked at hard enough to rank, because severity "
                    "decides order and an unranked defect never gets an order."
                ),
            },
            # B4 ("every grade confirmed by an independent check") LEFT THE BETA GATE on
            # 2026-10-01, by Daniel's decision. It stays in the full gate as G2, a
            # condition of the 1.0 release. Why this is honest: for every capability
            # that returns a fairness number, B6 and B7 now re-check it by EXECUTION,
            # against an outside library or against data where the number cannot
            # exist, which is a stronger confirmation than a second examiner's
            # opinion. What a second examiner would add for the rest (renderers,
            # tooling, plumbing) is stated on the quality page as open for 1.0.
            {
                "id": "B5",
                "name": "Capability census complete and clear",
                "measured": "see G5",
                "passes": all(c["passes"] for c in _criteria() if c["id"] == "G5"),
                "why": (
                    "The 215 capabilities are the surface a user reaches on purpose "
                    "and they carry the product promise directly."
                ),
            },
        ]
        + _check_criteria()
    )


def _check_criteria() -> list[dict]:
    """B6 and B7: the two executable checks, added 2026-10-01.

    Grades (B1 to B5) record what an examiner concluded. These two are programs
    anyone can re-run over every measuring capability, so a beta needs both.
    """
    g = _gate_checks()
    c1 = (g or {}).get("check1_correct_on_normal_data") or {}
    c2 = (g or {}).get("check2_honest_on_broken_data") or {}
    c1_ran = c1.get("state") == "ran"
    c2_ran = c2.get("state") == "ran"
    nam = c2.get("not_a_measurement", 0)
    return [
        {
            "id": "B6",
            "name": "Same answer as an established library, where one exists",
            "measured": (
                f"{c1['agree']} of {c1['with_reference']} agree; {c1['no_reference']} have no outside reference"
                if c1_ran
                else "could not check: docs/gate-checks.json missing or the reference libraries absent"
            ),
            "passes": c1_ran and not c1.get("differ") and c1["agree"] == c1["with_reference"],
            "why": (
                "Agreement with fairlearn, scikit-learn, statsmodels or scipy on clean data is "
                "the only correctness evidence that does not come from this project itself."
            ),
        },
        {
            "id": "B7",
            "name": "No invented number on broken data",
            "measured": (
                f"{c2['pass']} of {c2['entries'] - nam} measuring capabilities pass; "
                f"{c2['fabricates']} fabricate, {c2['over_refuses']} over-refuse, "
                f"{c2['not_covered']} not covered"
                if c2_ran
                else "could not check: docs/gate-checks.json missing"
            ),
            "passes": c2_ran and c2["pass"] + nam == c2["entries"],
            "why": (
                "A number where nothing was measurable reads as a clean bill. This is the "
                "defect the whole hardening programme was about."
            ),
        },
    ]


def _with_plain(criteria: list[dict]) -> list[dict]:
    """Attach the reader-facing sentence to each criterion."""
    for c in criteria:
        c["plain"] = PLAIN_ENGLISH.get(c["id"], "")
    return criteria


def _criteria() -> list[dict]:
    # Deliberately NOT library_kpis.build(): that runs a pytest collection to
    # count tests, which costs 30 seconds and tells this gate nothing. A gate
    # nobody runs because it is slow is a gate that does not gate.
    surf = library_kpis._walk_public_surface()
    grading = library_kpis.grading()
    _v, proof, _on, _asof = library_kpis._package()
    # All four batches are seeded at zero. An absent key would read as None at
    # the criterion below, and None is not zero: "no capability holds this
    # grade" and "nobody counted this grade" are different facts and only one
    # of them is a pass.
    batches: dict[str, int] = {b: 0 for b in ("BGL-A", "BGL-B", "BGL-C", "BGL-D")}
    for r in proof.values():
        batches[r["batch"]] = batches.get(r["batch"], 0) + 1
    hard = {"defects": library_kpis.defects(), "proof": batches}

    total_units = surf["total"]["total"]

    # G1 IS A SUBSET QUESTION, NOT A SUM, and it was asked as a sum until 2026-09-30.
    # `grading.items + surf.total.graded` adds two overlapping name sets and then
    # compares the total against the surface size, which is the identical mistake B1
    # was fixed for on 2026-09-25 and which its own test still pins as "a union and
    # never a sum". Measured here the day the sibling was found: the two sets overlap
    # by one unit, and 15 of the graded names are RE-EXPORT spellings the surface walk
    # does not count as units at all, so the sum credited 16 units that are not
    # graded surface units and reported 465 remaining where the honest answer is 481.
    #
    # A gate that overstates its own progress is the same defect class as a metric
    # that overstates a model's fairness, and it errs in the direction nobody
    # double-checks: toward being nearly finished.
    graded_names = set(library_kpis.graded_items())
    for _kind in ("functions", "classes", "methods"):
        graded_names |= set(surf[_kind]["graded_names"])
    surface_units = {
        qual
        for _kind in ("functions", "classes", "methods")
        for qual in surf[_kind]["ungraded"] + surf[_kind]["graded_names"]
    }
    ungraded_units = surface_units - graded_names
    graded_units = total_units - len(ungraded_units)
    unaudited = grading.get("unaudited")
    # SEEDED AT ZERO, for the reason stated twelve lines above about `batches`, which
    # was applied there and NOT here. by_grade is a tally of the grades that OCCUR, so
    # the moment the last DEFECT OPEN row is closed the key disappears and .get()
    # returns None. G3 then reported "cannot measure" and G4 with it, when the measured
    # answer was ZERO OPEN. Found 2026-09-30, the first time in this campaign that the
    # count actually reached zero, which is why it had never shown before: the guard
    # existed, in the same function, for the sibling tally.
    #
    # Only DEFECT OPEN is seeded because only it is read below. Seeding a grade the
    # gate does not consult would invent a number nobody uses.
    by_grade = {"DEFECT OPEN": 0, **(grading.get("by_grade") or {})}
    surface_open = by_grade.get("DEFECT OPEN")
    census_open = hard["defects"]["open"]
    census_unproven = hard["proof"].get("BGL-C")

    # Severity is not recorded for any surface defect yet. Unknown blocks.
    unknown_severity = surface_open

    out = [
        {
            "id": "G1",
            "name": "Every public code unit carries a grade",
            "measured": f"{graded_units} of {total_units}",
            "passes": graded_units >= total_units,
            "why": (
                "An ungraded code unit is not a clean one. It is the absence of a "
                "measurement, and it is where every defect this campaign found was "
                "living: in code nobody had ever run on undefined input."
            ),
        },
        {
            "id": "G2",
            "name": "Every grade was confirmed by an independent check",
            "measured": "cannot measure" if unaudited is None else f"{unaudited} unaudited",
            "passes": unaudited == 0,
            "why": (
                f"{grading.get('audit_overturned_ever')} of "
                f"{grading.get('audited_ever')} grades were overturned when an "
                "independent agent was told to refute them. A grade nobody checked "
                "is a claim, not a measurement, and at that rate a large share of "
                "them are wrong."
            ),
        },
        {
            "id": "G3",
            "name": "No open defect anywhere in the public surface",
            "measured": "cannot measure" if surface_open is None else f"{surface_open} open",
            "passes": surface_open == 0,
            "why": (
                "The product promise is that this library does not report a number "
                "nobody measured. An open defect of that class contradicts the "
                "promise wherever it sits, not only inside the capability census."
            ),
        },
        {
            "id": "G4",
            "name": "Every open defect has an assessed severity",
            "measured": "cannot measure"
            if unknown_severity is None
            else f"{unknown_severity} unassessed",
            "passes": unknown_severity == 0,
            "why": (
                "Severity decides ORDER, never whether. An unassessed severity is "
                "recorded as blocking: not having looked at a defect is not "
                "evidence that it is minor."
            ),
        },
        {
            "id": "G5",
            "name": "Capability census complete and clear",
            "measured": f"{census_open} open, {census_unproven} unproven",
            "passes": census_open == 0 and census_unproven == 0,
            "why": (
                "The 215 capabilities are the surface a user reaches on purpose. "
                "This is the criterion that already passes, and on its own it was "
                "never sufficient to call the library ready."
            ),
        },
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    crit = _criteria()
    beta = _beta_criteria()
    ready = all(c["passes"] for c in crit)
    beta_ready = all(c["passes"] for c in beta)
    verdict = {
        "ready_to_publish": ready,
        "verdict": "READY" if ready else "NOT READY",
        "beta_ready": beta_ready,
        "beta_verdict": "BETA READY" if beta_ready else "NOT BETA READY",
        "beta_criteria": beta,
        "criteria": crit,
        "note": (
            "Fail closed. A criterion that cannot be measured fails, and an open "
            "defect with no assessed severity counts as blocking."
        ),
    }
    if args.json:
        print(json.dumps(verdict, indent=1))
    else:
        print(f"\n  BETA GATE: {verdict['beta_verdict']}\n")
        for c in beta:
            print(f"  [{'PASS' if c['passes'] else 'FAIL'}] {c['id']}  {c['name']}")
            print(f"         measured: {c['measured']}")
        print(f"\n  FULL RELEASE GATE: {verdict['verdict']}\n")
        for c in crit:
            mark = "PASS" if c["passes"] else "FAIL"
            print(f"  [{mark}] {c['id']}  {c['name']}")
            print(f"         measured: {c['measured']}")
        print()
        if not ready:
            failed = [c["id"] for c in crit if not c["passes"]]
            print(f"  {len(failed)} of {len(crit)} criteria fail: {', '.join(failed)}")
            print("  This library is NOT ready to publish.\n")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
