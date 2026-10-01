#!/usr/bin/env python3
"""Universality sweep: run Pulse on each single-bias synthetic dataset
and check whether the engine catches the ONE planted mechanism.

Reports per-variant (column-presence, verdict-direction, proxy-linkage)
just like the main harness, then a corpus-level summary. Catches
regressions that are dataset-specific to the recruitment file.

Usage:
    python3 scripts/pulse_universality_sweep.py
    python3 scripts/pulse_universality_sweep.py --corpus /tmp/bias_corpus

Exit codes, three states and never two:
    0  every variant was graded
    2  no corpus to sweep (no L*.csv under --corpus)
    3  COULD NOT CHECK: the grading harness is unavailable, or at least one
       variant raised. A variant that could not be graded is reported and left
       out of the scores; it is never counted as a miss.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Reuse the harness's torch shim + run_pulse + grading dict
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# THE HARNESS DOES NOT SHIP, AND THIS SCRIPT DOES. pulse_bias_recall_harness.py
# is on EXCLUDES in scripts/export-vfairness-to-public.sh, so in a released copy
# of vfairness this import raises and, at module scope, took the whole script
# down with a bare ModuleNotFoundError traceback before argparse ever ran. A
# reader of that traceback learns nothing: not what is missing, not that it is
# missing on purpose, not that no flag will bring it back. Reproduced against
# the emitted export tree 2026-09-10.
#
# Degrade, do not fake. There is no fallback to build: the harness carries the
# planted-bias answer key (PLANTED, B1..B9), the protected-attribute list, the
# torch shim and every channel extractor this file grades with, so without it
# there is no measurement at all. main() says so and exits 3 (could not check).
# H stays bound to the real module or not at all, rather than to a None the
# rest of the file would then have to be typed around: HARNESS_AVAILABLE is the
# state, and every entry point checks it before touching H.
try:
    import pulse_bias_recall_harness as H  # noqa: E402

    HARNESS_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in the export tree
    if exc.name != "pulse_bias_recall_harness":
        raise
    HARNESS_AVAILABLE = False

#: Printed instead of a traceback when the harness is not importable.
HARNESS_MISSING = (
    "COULD NOT CHECK: scripts/pulse_bias_recall_harness.py is not importable, so "
    "there is nothing to grade against and no sweep was run.\n"
    "  It holds the planted-bias answer key (B1..B9), the protected-attribute "
    "list and the channel extractors this script reads; none of that is "
    "reconstructible from what ships here.\n"
    "  It is an internal development harness and is deliberately not part of the "
    "published distribution, so in a released copy of vfairness this sweep cannot "
    "run at all. Run it from the development repository instead.\n"
    "  Nothing was graded: this is not a score of zero."
)

import pandas as pd  # noqa: E402

# L<code> in the synthetic corpus maps 1:1 onto the B<code> planted bias
# definitions in the main harness. Each variant carries exactly that one
# bias, so a per-variant pass means the engine detected the planted
# mechanism on its OWN dataset (not just because the recruitment file
# has every mechanism overlapping).
LMAP = {
    "L1_gender_in_tech": "B1",
    "L2_race": "B2",
    "L3_age": "B3",
    "L4_zip": "B4",
    "L5_university": "B5",
    "L6_surname": "B6",
    "L7_disability": "B7",
    "L8_employment_gap": "B8",
    "L9_photo": "B9",
}


def grade_variant(csv_path: str, code: str):
    if not HARNESS_AVAILABLE:
        # Reachable only by a caller that skipped main()'s check. Raising with
        # the same text keeps the failure legible instead of surfacing as a
        # NameError on H a hundred lines further down.
        raise RuntimeError(HARNESS_MISSING)
    df = pd.read_csv(csv_path)
    audit = df.drop(columns=[c for c in H.ANSWER_COLS if c in df.columns])
    inputs = {
        "domain": "lending / credit underwriting",
        "jurisdiction": "US",
        "source_kind": "tabular",
        "protected_attributes": [c for c in H.PROTECTED if c in audit.columns],
    }
    res = H.run_pulse(audit, inputs)
    if isinstance(res, dict) and "data" in res and "perVariable" in res.get("data", {}):
        res = res["data"]
    # Reuse the channel extractors directly; collapse the PLANTED filter
    # to only the one bias this variant is testing.
    channels = {
        "perVariable": H._flagged_per_variable(res),
        "disparity": H._flagged_disparity(res),
        "proxy": H._flagged_proxy(res),
        "bias": H._flagged_bias(res),
        "intersectional": H._flagged_intersectional(res),
        "adjusted": set(),
    }
    adj = res.get("adjustedDisparity") or res.get("adjusted_disparity") or []
    for a in adj:
        if (
            isinstance(a, dict)
            and a.get("attribute")
            and (a.get("significant") or a.get("tone") in ("warn", "critical"))
        ):
            channels["adjusted"].add(str(a["attribute"]))
    spec = H.PLANTED[code]
    col_hit = False
    where = []
    for ch in spec["channels"]:
        inter = spec["cols"] & channels.get(ch, set())
        if inter:
            col_hit = True
            where.append(f"{ch}({','.join(sorted(inter))})")
    # verdict-direction (direct mechanism only)
    if spec.get("protectedAttr"):
        tone = H._tone_for_attr(res, spec["protectedAttr"])
        verdict_hit = tone in ("warn", "critical")
        v_note = f"tone={tone or 'n/a'}"
    else:
        verdict_hit = None
        v_note = "n/a (proxy mechanism)"
    # proxy-linkage (proxy mechanism only)
    if spec.get("proxyFor"):
        link_hit = H._proxy_links_for(
            res, spec["proxyFor"]["feature"], spec["proxyFor"]["protectedAttribute"]
        )
        l_note = (
            f"{spec['proxyFor']['feature']}->"
            f"{spec['proxyFor']['protectedAttribute']} "
            f"{'linked' if link_hit else 'NOT linked'}"
        )
    else:
        link_hit = None
        l_note = "n/a (direct mechanism)"
    return dict(
        code=code,
        name=spec["name"],
        col_hit=col_hit,
        verdict_hit=verdict_hit,
        link_hit=link_hit,
        where="; ".join(where) or "-",
        verdict_note=v_note,
        link_note=l_note,
        framework=res.get("legalFramework") or {},
    )


def _summarise(results):
    """Measured counts, with ungraded variants held out of every denominator.

    A variant that raised carries None on all three axes and is reported
    separately. Folding it in as a False would let a crash lower the recall
    score exactly like a real miss, which is the one reading that must never be
    possible: nobody measured that variant.
    """
    graded = [r for r in results if not r.get("error")]
    return dict(
        graded=len(graded),
        ungraded=len(results) - len(graded),
        col_pass=sum(1 for r in graded if r.get("col_hit") is True),
        v_relevant=sum(1 for r in graded if r.get("verdict_hit") is not None),
        v_pass=sum(1 for r in graded if r.get("verdict_hit") is True),
        l_relevant=sum(1 for r in graded if r.get("link_hit") is not None),
        l_pass=sum(1 for r in graded if r.get("link_hit") is True),
    )


def main() -> int:
    if not HARNESS_AVAILABLE:
        print(HARNESS_MISSING, file=sys.stderr)
        return 3
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/tmp/bias_corpus")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.corpus, "L*.csv")))
    if not files:
        print(f"no L*.csv files found in {args.corpus}", file=sys.stderr)
        return 2
    print("=" * 90)
    print(f"PULSE UNIVERSALITY SWEEP  ·  corpus={args.corpus}  ({len(files)} variants)")
    print("=" * 90)
    results = []
    for csv_path in files:
        m = re.match(r"(L\d+_[a-z_]+)\.csv$", os.path.basename(csv_path))
        if not m:
            continue
        variant = m.group(1)
        code = LMAP.get(variant)
        if not code:
            continue
        print(f"\n--- {variant} (planted: {code}) ---")
        try:
            r = grade_variant(csv_path, code)
        except Exception as e:  # noqa: BLE001
            # COULD NOT CHECK, not a miss. This used to record col/verdict/link
            # as False, which put the variant in every denominator and read as
            # "the engine ran and detected nothing" on a variant the engine
            # never saw. None means ungraded, and _summarise keeps it out of the
            # scores.
            print(f"  COULD NOT CHECK: {e}")
            results.append(
                dict(
                    code=code,
                    variant=variant,
                    col_hit=None,
                    verdict_hit=None,
                    link_hit=None,
                    error=str(e),
                )
            )
            continue
        col = "✓" if r["col_hit"] else "✗"
        v = "✓" if r["verdict_hit"] is True else "✗" if r["verdict_hit"] is False else "—"
        link = "✓" if r["link_hit"] is True else "✗" if r["link_hit"] is False else "—"
        print(f"  COL  {col}    where: {r['where']}")
        print(f"  VERDICT  {v}    {r['verdict_note']}")
        print(f"  LINK  {link}    {r['link_note']}")
        r["variant"] = variant
        results.append(r)
    summary = _summarise(results)
    print("\n" + "=" * 90)
    print(
        f"UNIVERSALITY SCORE   "
        f"COL {summary['col_pass']}/{summary['graded']}   "
        f"VERDICT {summary['v_pass']}/{summary['v_relevant']}   "
        f"LINK {summary['l_pass']}/{summary['l_relevant']}"
    )
    if summary["ungraded"]:
        print(
            f"COULD NOT CHECK      {summary['ungraded']}/{len(results)} variants "
            f"raised and are counted in none of the scores above:"
        )
        for r in results:
            if r.get("error"):
                print(f"  {r.get('variant')}: {r['error']}")
    print("=" * 90)
    # An ungraded variant makes the whole sweep a could-not-check: the scores
    # above are real, but they are not about the corpus that was asked for.
    return 3 if summary["ungraded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
