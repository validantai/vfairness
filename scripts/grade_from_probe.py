#!/usr/bin/env python3
"""Grade, from probe evidence alone, the code units whose evidence supports it.

WHAT THIS MAY AND MAY NOT CONCLUDE. SEMI-PROVEN is defined as "behaves honestly
today, verified by execution, but no test pins it". The probe IS execution, so
where its evidence meets that bar a grade can be assigned without an agent. It
can never assign PROVEN, because PROVEN requires a test that fails when the fix
is removed, and no script can write that.

THE RULE, fail closed, every clause load-bearing:

  1. the HEALTHY world must have produced a value. If the healthy run raised,
     we cannot tell "it refuses bad input" from "we called it wrong", and the
     evidence supports nothing at all;
  2. at least four of the eight degenerate worlds must have actually run;
  3. every degenerate world that ran must have raised or refused. Not one may
     have returned a neutral value silently.

WHY CLAUSE 1 EXISTS, and it is the whole reason this file is short. A first
version of this rule omitted it and qualified 316 code units. Measured: 287 of
those raised on the HEALTHY world too, with errors like "'numpy.ndarray' object
has no attribute 'keys'" and "could not convert string to float: 'a'". They were
not refusing honestly. The probe's generic argument binder had called them with
nonsense, and they raised for that reason on every world including the good one.
Grading those SEMI-PROVEN would have been the exact defect this library is
audited for: a grade asserted from evidence that does not support it, in the
flattering direction, by the tool built to prevent it. 29 qualify, not 316.

A code unit that fails clause 1 is NOT examined. The probe reached it and
learned nothing, which is a third state and is recorded as such rather than
counted on either side.

Usage:
    python scripts/grade_from_probe.py --out docs/surface-grading-from-probe.json
    python scripts/grade_from_probe.py --out /tmp/x.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "docs" / "surface-probe.json"
AGENT_GRADED = ROOT / "docs" / "surface-grading-2026-09-18.json"

DEGENERATE = (
    "single_group",
    "one_label_only",
    "all_nan_scores",
    "empty",
    "constant_scores",
    "n_equals_2",
    "one_row_minority",
    "unreadable_text",
)
REFUSED = {"raised", "refused"}
MIN_DEGENERATE_WORLDS = 4


def classify(runs: dict) -> tuple[str, str]:
    """(state, reason). state is SEMI-PROVEN, NOT_EXAMINED or NEEDS_JUDGEMENT."""
    healthy = (runs.get("healthy") or {}).get("outcome")
    # "refused" BELONGS IN THIS LIST and was missing for nine days, which made
    # clause 1 unable to fire at all. The probe's own describer maps a None return
    # to ("refused", "None") and a NaN to ("refused", "nan"), so a unit that
    # returned None on the GOOD world was recorded as refusing it, passed clause 1,
    # and was published SEMI-PROVEN with the reason "measured on healthy input".
    #
    # Measured 2026-09-27, found by an auditor attacking two of these rows: ALL 39
    # grades this file had produced carried healthy outcome "refused", 37 with
    # detail None and 2 with nan. Not one was measured on healthy input. The clause
    # this file's own docstring calls "the whole reason this file is short" had
    # never once fired, and the 316-to-29 filtering it describes was done entirely
    # by the "raised" case.
    #
    # A refusal on the healthy world is not evidence of honesty, it is the absence
    # of evidence either way: nothing distinguishes "refuses what it cannot
    # measure" from "the probe's generic binder called it with nonsense" or from "a
    # void function returns None and always did".
    if healthy in (None, "NOT_REACHED", "raised", "hung", "refused"):
        return "NOT_EXAMINED", (
            f"the healthy run did not produce a value (outcome: {healthy}), so "
            "this evidence cannot tell an honest refusal from a call the probe "
            "got wrong"
        )
    ran = [
        (w, (runs.get(w) or {}).get("outcome"))
        for w in DEGENERATE
        if (runs.get(w) or {}).get("outcome") not in (None, "NOT_REACHED")
    ]
    if len(ran) < MIN_DEGENERATE_WORLDS:
        return "NOT_EXAMINED", (
            f"only {len(ran)} degenerate world(s) ran, below the floor of {MIN_DEGENERATE_WORLDS}"
        )
    silent = [
        w
        for w, _o in ran
        if (runs.get(w) or {}).get("outcome") == "neutral"
        and not (runs.get(w) or {}).get("n_warnings")
    ]
    if silent:
        return "NEEDS_JUDGEMENT", (
            f"returned a neutral value with no warning on: {', '.join(silent)}"
        )
    if all(o in REFUSED for _w, o in ran):
        return "SEMI-PROVEN", (
            f"measured on healthy input and refused on all {len(ran)} degenerate "
            "worlds that ran, by raising or by returning a refusal. Verified by "
            "execution; no test pins it."
        )
    return "NEEDS_JUDGEMENT", "returned a value on degenerate input"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    already = set(json.loads(AGENT_GRADED.read_text(encoding="utf-8"))["items"])

    graded: dict[str, dict] = {}
    counts = {"SEMI-PROVEN": 0, "NOT_EXAMINED": 0, "NEEDS_JUDGEMENT": 0, "already graded": 0}
    for qual, rec in probe.items():
        if qual in already:
            counts["already graded"] += 1
            continue
        state, reason = classify(rec.get("runs") or {})
        counts[state] += 1
        if state == "SEMI-PROVEN":
            graded[qual] = {
                "grade": "SEMI-PROVEN",
                "source": "probe evidence, no agent",
                "audited": False,
                "reason": reason,
            }

    # WITHDRAWN, not silently absent. This file published 39 SEMI-PROVEN grades
    # until 2026-09-27 and the corrected clause 1 qualifies none of them. A count
    # that drops from 39 to 0 with nothing saying why reads as a lost file, so the
    # names and the reason travel with the regenerate and any surface that quotes
    # the older total can be reconciled against them.
    withdrawn: dict[str, dict] = {}
    if Path(args.out).exists():
        prior = json.loads(Path(args.out).read_text(encoding="utf-8"))
        for qual, rec in (prior.get("items") or {}).items():
            if qual not in graded:
                state, reason = classify((probe.get(qual) or {}).get("runs") or {})
                withdrawn[qual] = {
                    "was": rec.get("grade"),
                    "was_because": rec.get("reason"),
                    "now": state,
                    "withdrawn_because": reason,
                }
        withdrawn.update((prior.get("_withdrawn") or {}))

    out = {
        "measured_on": "2026-09-18",
        "method": (
            "Assigned from scripts/surface_probe.py evidence by "
            "scripts/grade_from_probe.py, with no agent involved. The rule is in "
            "that script's docstring and every clause is load-bearing. These "
            "grades are SEMI-PROVEN and can never be PROVEN from this source: "
            "PROVEN requires a test that fails when the fix is removed, and no "
            "script can write one."
        ),
        "rule": {
            "healthy_run_must_produce_a_value": True,
            "minimum_degenerate_worlds_run": MIN_DEGENERATE_WORLDS,
            "every_degenerate_world_must_refuse": True,
        },
        "counts": counts,
        "items": graded,
        **({"_withdrawn": dict(sorted(withdrawn.items()))} if withdrawn else {}),
    }
    if args.dry_run:
        print(json.dumps(counts, indent=1))
        return 0
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**counts, "written": args.out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
