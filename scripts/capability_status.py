#!/usr/bin/env python3
"""The published status of every capability, in three states a reader can act on.

WHY THIS EXISTS. The five internal grades (PROVEN / SEMI-PROVEN / UNPROVEN /
DEFECT OPEN / NOT A MEASUREMENT) are precise and they answer a question a user
never asks. A user asks one thing: *can I rely on this call in the beta?* Before
this file the answer lived in three different JSON files, none of them reachable
from the page where the capability is documented, and the honest answer for two
thirds of the surface: nobody has looked: was not published anywhere at all.

THE THREE STATES, and the rule that each one is a measurement, never a default:

  CHECKED       graded and assessed. If a defect was found it was fixed. What
                the grade does NOT establish travels with it (see `detail`).
  FIX PENDING   graded and assessed, and a defect is still open. Do not rely on
                it in the beta.
  NOT CHECKED   nobody has established anything. We are blind here. It may be
                perfectly correct and it may fabricate; this state says only
                that no evidence exists either way.

FAIL CLOSED, in four places, because each one is a way this file could have
flattered the library:

  1. A unit with no evidence is NOT CHECKED. There is no default and no
     inference from its name, its module or its neighbours.
  2. If ANY source says DEFECT OPEN the unit is FIX PENDING, even when a later
     or friendlier source calls it proven. An adversarial audit that overturned
     a grade is evidence; a grade it overturned is not.
  3. When sources disagree about how strongly a CHECKED unit is held, the
     WEAKEST claim wins. A pin recorded in one place and absent in another is
     reported as absent.
  4. The ledger must cover the whole surface. `--check` fails if the published
     row count differs from the measured surface, so a unit cannot drop out of
     the denominator and silently improve the percentages.

WHAT "CHECKED" MAY NOT BE READ AS. It is not a correctness certificate for the
number the code returns. It means the fabrication class was examined: on input
where the thing being measured does not exist, the unit refuses instead of
handing back a neutral-looking value. `detail` states the strength of each row,
and the grid publishes it beside the badge rather than in a footnote.

Usage:
    python scripts/capability_status.py                  # write the ledger
    python scripts/capability_status.py --check          # exit 1 if stale
    python scripts/capability_status.py --matrix         # print the doc tables
"""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis as K  # noqa: E402

WAVE = ROOT / "docs" / "surface-grading-2026-09-18.json"
# Later waves, newest last. A wave supersedes an EARLIER wave's grade for the same
# unit, and only for the units it names. Without this, rule 2 (any DEFECT OPEN
# makes a unit FIX PENDING) would mean no defect could ever be closed: the
# 2026-09-18 record would hold a unit red however thoroughly it was later fixed.
# The price of supersession is evidence, enforced below: a later PROVEN row must
# name both the test that pins it and the sabotage that proved the test can fail.
# Taken from library_kpis so the two cannot disagree about which waves exist. It was
# a hand-kept list here and a narrow glob there, and on 2026-09-25 both missed the
# same new wave file, which is exactly the "two surfaces disagreeing about one
# measurement" failure this module exists to remove.
LATER_WAVES = list(K.LATER_WAVES)
FROM_PROBE = ROOT / "docs" / "surface-grading-from-probe.json"
PROBE = ROOT / "docs" / "surface-probe.json"
CLASS_PROBE = ROOT / "docs" / "class-probe.json"
# The explainability surface, executed by scripts/xai_probe.py against a fitted
# model, a matrix and a background sample. The generic probe could never reach it:
# its nine worlds are built from labels, scores, groups and a frame, and an
# explainer needs none of those. 79 of 81 XAI units carried no grade for that
# reason alone, which is a gap in our fixtures being published as a gap in the
# library's honesty.
XAI_PROBE = ROOT / "docs" / "xai-probe.json"
# Which units the TEST SUITE executes, measured by scripts/suite_coverage.py. This
# changes no STATE: executing a unit is not judging whether it refuses honestly, so
# a unit here stays NOT CHECKED. It changes the REASON, and the old reason was FALSE
# for 528 of them: the ledger said "never executed by the census or the probe" about
# units that 10,552 passing tests run every time CI fires.
SUITE_COVERAGE = ROOT / "docs" / "suite-coverage.json"
LEDGER = ROOT / "docs" / "capability-status.json"
SITE_LEDGER = ROOT / "docs" / "site" / "data" / "capability-status.json"
# The IN-PACKAGE copy, so `vfairness.status()` answers about the version a user
# actually installed rather than about whatever the website says today. Compact on
# purpose: the detail strings are interned, because there are a few dozen distinct
# ones across the whole surface and shipping each once instead of once per row is
# the difference between a few kilobytes and most of a megabyte in the wheel.
PACKAGE_LEDGER = ROOT / "src" / "vfairness" / "_capability_status.json"

CHECKED, FIX_PENDING, NOT_CHECKED = "CHECKED", "FIX PENDING", "NOT CHECKED"

# The five internal grades, mapped to the three published states. Written out
# rather than computed so that adding a sixth grade fails loudly here instead of
# falling into a friendly default.
FIVE_TO_THREE = {
    "PROVEN": CHECKED,
    "SEMI-PROVEN": CHECKED,
    "NOT A MEASUREMENT": CHECKED,
    "DEFECT OPEN": FIX_PENDING,
    "UNPROVEN": NOT_CHECKED,
}

# Strength of a CHECKED row, weakest first. Rule 3 takes the minimum.
STRENGTH = ["NOT A MEASUREMENT", "SEMI-PROVEN", "PROVEN"]

DETAIL = {
    "PROVEN": "refuses honestly when nothing is measurable, and a test fails if that regresses",
    "SEMI-PROVEN": "verified by running it; no test pins the behaviour, so a regression would be silent",
    "NOT A MEASUREMENT": "returns no fairness number and no verdict, so it cannot fabricate one",
    "DEFECT OPEN": "proved to report, or to claim, something nobody measured",
    "UNPROVEN": "examined, and no fixture could reach it",
}

# Functional areas, longest prefix first: the order is load-bearing, because
# `preprocessing.bias_detection` must not be swallowed by `preprocessing`.
AREAS = [
    ("evaluation.vfairness_metrics", "Fairness metrics"),
    ("preprocessing.bias_detection", "Bias detection"),
    ("preprocessing", "Preprocessing mitigation"),
    ("in_processing", "In-processing mitigation"),
    ("post_processing", "Post-processing mitigation"),
    ("xai", "Explainability (XAI)"),
    ("explainer", "Explainability (XAI)"),
    ("operations.monitoring", "Monitoring"),
    ("operations.reporting", "Reporting"),
    ("operations.cicd", "CI/CD gates"),
    ("operations", "Operations (other)"),
    ("rendering", "Report rendering"),
    ("llm", "LLM fairness testing"),
    ("agents", "Agent fairness testing"),
    ("multi_agent", "Agent fairness testing"),
    ("validity", "Validity axis"),
    ("legal", "Legal"),
    ("mcp", "MCP tools"),
    ("vision", "Vision"),
    ("net", "Networking"),
    ("evaluation", "Evaluation (other)"),
]

# WHAT THE BETA COVERS, and the rule that decides it.
#
# CORE is every area this project publishes a guide or a product page for, plus the
# top-level namespace a user imports from. The rule is not "what feels important",
# it is "what have we told people to use": if a page on the site walks a reader into
# a capability, that capability is part of what the beta promises, and it has to be
# examined before the beta claims to be ready.
#
# The rule cuts BOTH WAYS, and that is the point of writing it down. Anything we are
# not prepared to examine must stop being advertised as ready, which is why the
# second set exists and why it is published rather than implied.
#
# The earlier line was narrower: fairness metrics, bias detection and explainability
# alone, on the reasoning that those three are the path to a first honest
# measurement. It drew the boundary by FUNCTIONAL AREA rather than by what reaches a
# user, and two things fell through it. Report rendering draws the verdict a reader
# actually looks at, and is the worst-covered area in the library. CI/CD gates write
# a pass or a fail into somebody's pipeline, which is the most dangerous place in
# this codebase for a value nobody measured. Neither was core, and both are
# advertised on the getting-started and sample-assessment pages.
CORE_AREAS = {
    # The assessment path: measure, find, explain.
    "Fairness metrics",
    "Bias detection",
    "Explainability (XAI)",
    # What hands the user a verdict. A measurement reaches a reader through these.
    "Report rendering",
    "Reporting",
    "CI/CD gates",
    "Monitoring",
    # Surfaces with a product page of their own.
    "LLM fairness testing",
    "Agent fairness testing",
    "Preprocessing mitigation",
    "In-processing mitigation",
    "Post-processing mitigation",
    # The namespace a user imports from.
    "Top level",
}

# PREVIEW is documented but NOT certified by the beta bar, and every one of these
# has to say so where it is described. They are here because they are either not
# finished (the validity axis is deliberately unregistered until its gold set
# exists), or they are plumbing with no user-facing entry point, or they are an
# integration surface a reader reaches through another tool rather than by writing
# vfairness code.
PREVIEW_AREAS = {
    "Operations (other)",
    "MCP tools",
    "Validity axis",
    "Vision",
    "Networking",
    "Legal",
}

# Kept as the old names so nothing downstream breaks while the wording settles.
TIER1_AREAS = CORE_AREAS
TIER2_AREAS = PREVIEW_AREAS


# The census recorded its verdict in TWO places, and reading only one of them is
# how this file first reported 129 stamped capabilities as "never executed by the
# census or the probe". That sentence was false, and false in the direction that
# understates the library, which is still a number nobody measured. The two
# places: CAPABILITY_PROOF, keyed by dispatch row, and the docstring stamp that
# scripts/stamp_proof_status.py writes onto the object itself. `_is_graded` in
# library_kpis accepts EITHER, so this file has to read either.
_STAMP_BATCH = re.compile(r"Beta Go-Live proof status \([\d-]+\): (BGL-[ABCD])\b")
_STAMP_ROW = re.compile(r"^Ledger row: ([A-Za-z0-9_.]+)", re.MULTILINE)


def _stamped_batch(obj: object) -> tuple[str | None, str | None]:
    """(batch, ledger row) read off the object's own docstring stamp."""
    doc = inspect.getdoc(obj) or ""
    m = _STAMP_BATCH.search(doc)
    if not m:
        return None, None
    row = _STAMP_ROW.search(doc)
    return m.group(1), (row.group(1).rstrip(".") if row else None)


# Phrases by which the code declares, in its own docstring, that it does not do
# what its name promises. These are DISCLOSURES, not grades: a stub can be
# perfectly honest, and a wired function can still fabricate. But a reader
# deciding whether to call something needs to know the library already says it is
# not connected, and that sentence was reachable only by reading the source.
_DECLARED = (
    "NOT WIRED",
    "not wired",
    "no dispatch path calls it",
    "is still a stub",
    "is a stub",
    "not implemented",
    "NotImplementedError",
    "placeholder",
)

# The generated proof stamp is PROSE and it is inside the docstring, so it must be
# removed before any phrase heuristic reads one. Stamping 215 docstrings once made
# every metric look like it cited a paper, because the stamp's date matched a year
# pattern. Strip the block, never weaken the pattern.
_STAMP_BLOCK = re.compile(r"Beta Go-Live proof status.*?\(end Beta Go-Live proof status\)", re.S)


def _declared_limits(obj: object) -> str | None:
    doc = inspect.getdoc(obj) or ""
    doc = _STAMP_BLOCK.sub("", doc)
    for phrase in _DECLARED:
        idx = doc.find(phrase)
        if idx == -1:
            continue
        window = " ".join(doc[max(0, idx - 90) : idx + 150].split())
        return f"the code's own docstring says: ...{window}..."
    return None


def area_of(qual: str) -> str:
    for pat, label in AREAS:
        if f".{pat}." in qual or qual.startswith(f"vfairness.{pat}."):
            return label
    return "Top level"


def _probe_note(rec: dict | None) -> str:
    """Why a NOT CHECKED unit is not checked. Three distinct reasons, kept apart.

    Collapsing them would hide the one that matters: "the probe called it and
    the call itself failed" is a statement about our fixture, not about the
    code, and it is the largest group. Reporting it as "refused" would be the
    library's own defect: evidence that supports nothing, read as a pass.
    """
    if rec is None:
        # Reached only when NO source executed the unit: not the function probe, not
        # the class or explainability probes, and not the test suite. Everything else
        # gets a truer reason above, and this sentence was applied to 528 units that
        # CI executes on every push before the suite was counted at all.
        return "never executed by any probe or test"
    outcome = ((rec.get("runs") or {}).get("healthy") or {}).get("outcome")
    if outcome in (None, "NOT_REACHED"):
        return "could not be called without a hand-built fixture"
    if outcome in ("raised", "hung"):
        return "called, but the call itself failed, so nothing was learned about it"
    # A unit the probe SAW hand back a neutral value on a world where the thing
    # it measures does not exist, without a single warning, is not merely
    # unjudged: there is evidence pointing one way. It stays NOT CHECKED,
    # because a program may not convert that into a defect, but reporting it as
    # a plain "not looked at yet" would throw the evidence away.
    silent = [
        world
        for world, run in (rec.get("runs") or {}).items()
        if world != "healthy"
        and (run or {}).get("outcome") == "neutral"
        and not (run or {}).get("n_warnings")
    ]
    if silent:
        return (
            f"executed, and on {len(silent)} world(s) where nothing was measurable "
            f"({', '.join(sorted(silent)[:3])}) it returned a neutral value and warned nobody. "
            "Suspected fabrication, not yet judged: treat its output as unverified."
        )
    return "executed on healthy and degenerate input; the evidence has not been judged yet"


def _stem(name: str) -> str:
    """ "surface-grading-2026-09-29d.json" -> "2026-09-29d".

    The ORDERING KEY for two waves written on the same day, and the label a reader
    sees. A bare date cannot order them and a tie is resolved by returning the older
    one, which republishes a defect a later wave has already closed.
    """
    return name.removeprefix("surface-grading-").removesuffix(".json")


def _alias_index(universe: dict) -> tuple[dict, dict]:
    """Index the surface so a grade recorded under a re-export spelling still lands.

    A grading wave recorded `vfairness.evaluation.vfairness_metrics.StatisticalResult.to_dict`;
    the surface walk records the same object under the module that DEFINES it,
    `..._statistics.StatisticalResult.to_dict`. Four grades were recorded under a
    spelling the surface does not use, and reading them literally left three
    graded units sitting in NOT CHECKED while the release gate counted them in
    its numerator. Both numbers were wrong, in opposite directions, from the same
    cause.

    Only a UNIQUE match may be transferred. Two rows sharing `to_dict` are not
    evidence about either of them, and guessing between them would invent a
    grade, which is the one thing this file may never do.
    """
    by_pair: dict[str, list[str]] = collections.defaultdict(list)
    by_tail: dict[str, list[str]] = collections.defaultdict(list)
    for qual in universe:
        parts = qual.split(".")
        by_pair[".".join(parts[-2:])].append(qual)
        by_tail[parts[-1]].append(qual)
    return by_pair, by_tail


def _resolve_alias(key: str, by_pair: dict, by_tail: dict) -> str | None:
    parts = key.split(".")
    for index, probe_key in ((by_pair, ".".join(parts[-2:])), (by_tail, parts[-1])):
        hits = index.get(probe_key) or []
        if len(hits) == 1:
            return hits[0]
    return None


def build() -> dict:
    surface = K._walk_public_surface()
    vfairness, proof, measured_on, _asof = K._package()

    universe: dict[str, str] = {}
    for kind in ("functions", "classes", "methods"):
        for qual in surface[kind]["ungraded"] + surface[kind]["graded_names"]:
            universe[qual] = kind[:-1] if kind != "classes" else "class"

    wave = json.loads(WAVE.read_text(encoding="utf-8"))["items"] if WAVE.exists() else {}
    from_probe = (
        json.loads(FROM_PROBE.read_text(encoding="utf-8"))["items"] if FROM_PROBE.exists() else {}
    )

    # Re-export spellings, resolved to the defining row. Anything that cannot be
    # resolved uniquely is REPORTED in the ledger under `unresolved_grades`, not
    # dropped: a grade whose subject nobody can identify is a finding.
    evidence_dates = {measured_on}
    wave_meta = json.loads(WAVE.read_text(encoding="utf-8")) if WAVE.exists() else {}
    if wave_meta.get("measured_on"):
        evidence_dates.add(wave_meta["measured_on"])
    later_waves = []
    for path in LATER_WAVES:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            later_waves.append((data["measured_on"], path.name, data["items"]))
            evidence_dates.add(data["measured_on"])
    # SORT ON THE DATE ONLY, and break ties by FILENAME, never by the payload.
    # This was a bare `later_waves.sort()` over (date, items) pairs, which raised
    # TypeError the first time two waves carried the same measured_on: dicts do not
    # compare. Two waves in one day is normal (a morning wave and an evening one),
    # so the tie-break is the filename, which makes `...-25b.json` land after
    # `...-25.json` deterministically. Python's sort is stable, so equal keys keep
    # discovery order regardless.
    later_waves.sort(key=lambda entry: (entry[0], entry[1]))
    # THE FILENAME SURVIVES THE SORT, because it is what orders two waves of the SAME
    # DAY and nothing else can. It was dropped here, so both waves became the claim
    # source "grading wave 2026-09-29" and `max()` over a tie returns the FIRST
    # maximal element, which is the EARLIER file.
    #
    # Measured 2026-09-29, with an audit wave (-29c) and the fix wave that closes it
    # (-29d) written the same day: library_kpis.graded_items() read
    # FairnessAnalyzer.compare_with_aequitas as PROVEN while this module published it
    # FIX PENDING / DEFECT OPEN. Two readers disagreeing about one measurement, which
    # is the failure this module's own docstring says it exists to remove, caused by
    # this line. The stem sorts by date and then by suffix, so -29d beats -29c.
    later_waves = [(_stem(name), items) for _date, name, items in later_waves]

    by_pair, by_tail = _alias_index(universe)
    unresolved: list[str] = []
    unsupported: list[str] = []
    # LATER WAVES GET THE SAME ALIAS RESOLUTION, and they used to get none at all.
    # This loop read `(wave, from_probe)` only, so a later wave's row keyed by a
    # RE-EXPORT spelling matched nothing in the universe and was dropped in silence:
    # wave 4 recorded the zero-variance correlation fix under
    # `preprocessing.feature_engineering.compute_feature_correlations`, which the
    # surface walk does not count as a unit, and the canonical row went on citing the
    # capability census as its only source. The published state happened to be right,
    # so nothing looked wrong; the evidence for the fix simply was not attached to the
    # unit it fixed. A supersession that silently does not supersede is worse than a
    # missing one, because the wave file reads as though it landed.
    #
    # Anything still unresolvable is appended to `unresolved` and published in the
    # ledger under `unresolved_grades`, which is the existing three-state answer: a
    # grade whose subject nobody can identify is a finding, not a row to drop.
    for source in (wave, from_probe, *(items for _date, items in later_waves)):
        for key in [k for k in source if k not in universe]:
            target = _resolve_alias(key, by_pair, by_tail)
            if target is None or target in source:
                unresolved.append(key)
                continue
            source[target] = dict(source.pop(key), recorded_as=key)
    probe = json.loads(PROBE.read_text(encoding="utf-8")) if PROBE.exists() else {}
    # Classes, executed by scripts/class_probe.py on the same nine worlds. Only
    # its NOT A MEASUREMENT verdict is evidence of anything: a constructor that
    # derives a number from the data goes to judgement, and a class we could not
    # build stays in the blind pile where it belongs.
    class_probe = (
        json.loads(CLASS_PROBE.read_text(encoding="utf-8")) if CLASS_PROBE.exists() else {}
    )
    xai_probe = json.loads(XAI_PROBE.read_text(encoding="utf-8")) if XAI_PROBE.exists() else {}
    suite_executed: set = set()
    if SUITE_COVERAGE.exists():
        _rec = json.loads(SUITE_COVERAGE.read_text(encoding="utf-8"))
        _run = _rec.get("suite") or {}
        # A red run is not evidence: lines executed on the way to a failing
        # assertion did not verify anything.
        if _run.get("recorded") and not _run.get("failed") and not _run.get("errors"):
            suite_executed = set(_rec.get("executed") or ())

    # The census ledger is keyed by dispatch row, and one row can resolve to
    # several live objects (`shared_with`). Both keys are indexed so a method
    # reached only through an alias is not read as ungraded.
    census_batch = {}
    for row, rec in proof.items():
        for key in [row, *(rec.get("shared_with") or ())]:
            census_batch[key] = rec
    batch_to_grade = {
        "BGL-A": "PROVEN",
        "BGL-B": "SEMI-PROVEN",
        "BGL-C": "UNPROVEN",
        "BGL-D": "DEFECT OPEN",
    }
    census_graded = (
        set(surface["functions"]["graded_names"])
        | set(surface["classes"]["graded_names"])
        | set(surface["methods"]["graded_names"])
    )

    exported: dict[str, str] = {}
    for name in getattr(vfairness, "__all__", []):
        obj = getattr(vfairness, name, None)
        mod = getattr(obj, "__module__", None)
        if mod:
            exported[f"{mod}.{getattr(obj, '__qualname__', name)}"] = name
    # A method of an exported class is reached as `vfairness.Thing().method()`,
    # so it is part of the same public path and inherits the export.
    exported_classes = {q for q in exported if universe.get(q) == "class"}
    for qual in universe:
        if universe[qual] == "method" and qual.rsplit(".", 1)[0] in exported_classes:
            exported.setdefault(
                qual, exported[qual.rsplit(".", 1)[0]] + "()." + qual.rsplit(".", 1)[1]
            )

    rows = {}
    for qual, kind in sorted(universe.items()):
        claims: list[tuple[str, str, dict]] = []  # (grade, source, record)
        if qual in wave:
            claims.append(
                (wave[qual]["grade"], f"grading wave {wave[qual].get('wave', '?')}", wave[qual])
            )
        for date, items in later_waves:
            rec = items.get(qual)
            if rec is None:
                continue
            # FAIL CLOSED ON THE EVIDENCE ITSELF. A later wave is allowed to
            # overturn an open defect only if it carries what a grade is made of.
            # A row claiming PROVEN with no pin and no sabotage is a sentence
            # somebody wrote, and reading it as a grade would rebuild this
            # library's defect inside the file that publishes its status.
            if rec.get("grade") == "PROVEN" and not (rec.get("test_file") and rec.get("sabotage")):
                unsupported.append(f"{qual} (wave {date}): PROVEN with no pin and/or no sabotage")
                continue
            claims.append((rec["grade"], f"grading wave {date}", rec))
        if qual in census_graded:
            rec = census_batch.get(qual.split(".")[-1]) or census_batch.get(qual)
            if rec is None:
                obj = (K._OBJECTS.get(qual) or (None, None))[1]
                batch, row = _stamped_batch(obj) if obj is not None else (None, None)
                if batch:
                    # The stamp is the claim; the row, when it names one, carries
                    # the pins and the open-defect count behind it.
                    rec = dict(census_batch.get(row) or {}, batch=batch)
            if rec:
                claims.append((batch_to_grade[rec["batch"]], "beta go-live census", rec))
        if qual in from_probe:
            claims.append((from_probe[qual]["grade"], "probe evidence, no agent", from_probe[qual]))
        cp = class_probe.get(qual)
        if cp and cp.get("grade") == "NOT A MEASUREMENT":
            claims.append(("NOT A MEASUREMENT", "class probe, no agent", cp))
        xp = xai_probe.get(qual)
        if xp and xp.get("grade") == "SEMI-PROVEN":
            claims.append(("SEMI-PROVEN", "xai probe, no agent", xp))

        # Supersession, before any of the fail-closed rules see the claims. A unit
        # named by a later wave is judged on that wave alone: mixing a superseded
        # DEFECT OPEN back in would make the newer measurement unable to close it.
        superseding = [c for c in claims if c[1].startswith("grading wave 2026-")]
        if superseding:
            claims = [max(superseding, key=lambda c: c[1])]

        grades = [g for g, _s, _r in claims]
        if not grades:
            status, grade = NOT_CHECKED, None
            xp = xai_probe.get(qual)
            cp = class_probe.get(qual)
            if xp and xp.get("grade") == "NEEDS JUDGEMENT":
                detail = (
                    "executed against a fitted model: " + xp["reason"] + ". Suspected "
                    "fabrication, not yet judged: treat its output as unverified."
                )
            elif xp:
                detail = "the explainability probe could not reach it: " + xp["reason"]
            elif cp and cp.get("grade") == "NEEDS JUDGEMENT":
                detail = "constructed on healthy and degenerate input; it derives a number and that has not been judged yet"
            elif cp:
                detail = "could not be constructed without a hand-built fixture"
            elif qual in suite_executed:
                detail = (
                    "executed by the test suite, which records a pass, but no grading "
                    "wave has judged whether it refuses honestly when nothing is "
                    "measurable"
                )
            else:
                detail = _probe_note(probe.get(qual))
        elif "DEFECT OPEN" in grades:  # rule 2
            status, grade = FIX_PENDING, "DEFECT OPEN"
            detail = DETAIL["DEFECT OPEN"]
        elif all(FIVE_TO_THREE[g] == NOT_CHECKED for g in grades):
            status, grade = NOT_CHECKED, "UNPROVEN"
            detail = DETAIL["UNPROVEN"]
        else:
            checked = [g for g in grades if FIVE_TO_THREE[g] == CHECKED]
            grade = min(checked, key=STRENGTH.index)  # rule 3
            status, detail = CHECKED, DETAIL[grade]

        row = {
            "kind": kind,
            "area": area_of(qual),
            # CORE or PREVIEW, on every row, so the label can appear wherever the
            # unit is described instead of only on the page that defines the cut. A
            # boundary a reader meets only once is not a boundary.
            "scope": "core" if area_of(qual) in CORE_AREAS else "preview",
            "status": status,
            "grade": grade,
            "detail": detail,
            "sources": sorted({s for _g, s, _r in claims}),
        }
        if qual in exported:
            row["exported_as"] = exported[qual]
        if status == FIX_PENDING:
            rec = next(r for g, _s, r in claims if g == "DEFECT OPEN")
            row["defect"] = _defect_kind(rec)
            if rec.get("grade_note"):
                row["defect_note"] = rec["grade_note"]
        pins = _pins(claims)
        if pins:
            row["pins"] = pins
        obj = (K._OBJECTS.get(qual) or (None, None))[1]
        if obj is not None:
            declared = _declared_limits(obj)
            if declared:
                row["declared_limitation"] = declared
        rows[qual] = row

    return {
        "_what_this_is": (
            "One row per public code unit, carrying the three-state status a reader can act on. "
            "Generated by scripts/capability_status.py from the grading waves, the beta go-live "
            "census and the surface probe. Never hand-edited: run the script."
        ),
        "states": {
            CHECKED: "graded and assessed; any defect found was fixed. Read `detail` for what the grade does not establish.",
            FIX_PENDING: "graded and assessed, and a defect is still open. Do not rely on it in the beta.",
            NOT_CHECKED: "no evidence either way. Nobody has established that it refuses honestly, and nobody has established that it does not.",
        },
        # A SINGLE date would be a fiction here: this ledger merges the census
        # (2026-09-11), the grading waves, the probes and the later fix waves, and
        # printing the census date alone made the quality page say "measured
        # 2026-09-11" over rows measured a fortnight later. Both are published: the
        # newest evidence any row rests on, and every date present.
        "measured_on": measured_on,
        "newest_evidence": max(evidence_dates) if evidence_dates else measured_on,
        "evidence_dates": sorted(evidence_dates),
        "surface_total": surface["total"]["total"],
        "extras_present": surface["optional_extras_present"],
        "counts": dict(collections.Counter(r["status"] for r in rows.values())),
        "unresolved_grades": sorted(unresolved),
        "unsupported_grade_claims": sorted(unsupported),
        "declared_limitations": sum(1 for r in rows.values() if r.get("declared_limitation")),
        "tier1_areas": sorted(CORE_AREAS),
        "tier2_areas": sorted(PREVIEW_AREAS),
        "core_areas": sorted(CORE_AREAS),
        "preview_areas": sorted(PREVIEW_AREAS),
        "units": rows,
    }


def _defect_kind(rec: dict) -> str:
    """What kind of open defect this is. The three differ by an order of magnitude
    in what it costs to close, and the earlier registers reported them as one
    number, which made "95 open defects" read as 95 broken functions."""
    if rec.get("fabricated") and not rec.get("fixed"):
        return (
            "fabrication open: it was seen returning a value nobody measured, and it is not fixed"
        )
    if rec.get("fabricated") and rec.get("fixed"):
        return (
            "fix not accepted: the fabrication was fixed and an independent audit refused the claim"
        )
    return "evidence insufficient: no fabrication was seen, and the pins do not cover enough input to call it proven"


def _pins(claims) -> list[str]:
    out: list[str] = []
    for _g, _s, rec in claims:
        for key in ("pins", "test_file"):
            v = rec.get(key)
            if isinstance(v, list):
                out.extend(str(x) for x in v)
            elif isinstance(v, str) and v:
                out.append(v)
    return sorted(set(out))


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def matrix(led: dict) -> str:
    units = led["units"]
    total = len(units)
    counts = collections.Counter(r["status"] for r in units.values())
    out = []
    out.append("| Published status | Units | Share |")
    out.append("| --- | ---: | ---: |")
    for s in (CHECKED, FIX_PENDING, NOT_CHECKED):
        out.append(f"| **{s}** | {counts[s]} | {_pct(counts[s], total)} |")
    out.append(f"| | **{total}** | 100% |")

    # NOT INDEPENDENTLY CHECKED, beside every count, never in a separate document.
    # "Checked" means ONE examiner ran the unit and recorded a result. It does not
    # mean a second examiner tried to refute that result, and of the grades that
    # have been through that second examination, roughly one in three did not
    # survive it (the live figure is printed under the table). A "Checked" column
    # read on its own therefore overstates what has been established, and it
    # overstates it in exactly the direction a reader is least likely to question.
    # Added 2026-09-30, when the first-examination wave moved Core "not checked"
    # from 173 to 6 and the unchecked-by-anyone-else count from 131 to 612 in the
    # same commit: a table carrying only the first number would have reported that
    # day as pure progress.
    graded = K.graded_items()
    unaudited = {q for q, rec in graded.items() if not rec.get("audited")}
    g = K.grading()

    out.append("")
    out.append(
        "| Scope | Units | Checked | Fix pending | Not checked | Not independently checked |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    by_scope = collections.defaultdict(collections.Counter)
    unaudited_by_scope = collections.Counter()
    for q, r in units.items():
        sc = (
            "core"
            if r["area"] in CORE_AREAS
            else ("preview" if r["area"] in PREVIEW_AREAS else "other")
        )
        by_scope[sc][r["status"]] += 1
        if q in unaudited:
            unaudited_by_scope[sc] += 1
    labels = {
        "core": "Core, what the beta promises",
        "preview": "Preview, documented, not certified",
        "other": "Other",
    }
    for sc in ("core", "preview", "other"):
        c = by_scope.get(sc)
        if not c:
            continue
        out.append(
            f"| {labels[sc]} | {sum(c.values())} | {c[CHECKED]} | {c[FIX_PENDING]} | "
            f"{c[NOT_CHECKED]} | {unaudited_by_scope[sc]} |"
        )
    tot = sum(by_scope.values(), collections.Counter())
    out.append(
        f"| **Total** | **{sum(tot.values())}** | **{tot[CHECKED]}** | "
        f"**{tot[FIX_PENDING]}** | **{tot[NOT_CHECKED]}** | "
        f"**{sum(unaudited_by_scope.values())}** |"
    )
    over, ever = g.get("audit_overturned_ever"), g.get("audited_ever")
    # RECONCILED WITH THE GATE, because two numbers for one question is the defect
    # class this page exists to report. This table counts SURFACE UNITS; the release
    # gate's G2 and B4 count GRADED ROWS, and some rows grade a re-export spelling
    # (vfairness.X for a unit that lives in vfairness.a.b.X) that the surface walk
    # does not count as a unit of its own. Both are right about what they count.
    rows_unaudited = g.get("unaudited")
    units_unaudited = sum(unaudited_by_scope.values())
    if isinstance(rows_unaudited, int) and rows_unaudited != units_unaudited:
        out.append("")
        out.append(
            f"The release gate reports {rows_unaudited} not independently checked; this "
            f"table reports {units_unaudited}. The gate counts graded rows and this table "
            f"counts code units, and {rows_unaudited - units_unaudited} of the gate's rows "
            f"grade a re-export spelling of a unit already counted here."
        )
    if over and ever:
        out.append("")
        out.append(
            f"*Not independently checked* counts grades one examiner reached and nobody "
            f"has yet tried to refute. Of the grades that HAVE been independently checked, "
            f"{over} of {ever} ({_pct(over, ever)}) were overturned, so treat those rows "
            f"as claims with evidence behind them rather than settled results."
        )
    else:
        out.append("")
        out.append(
            "The overturn rate of independently checked grades could not be measured "
            "here, so how far to trust the unchecked rows is NOT stated."
        )

    out.append("")
    out.append(
        "| Area | Tier | Checked | Fix pending | Not checked | Total | Not independently checked |"
    )
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    by_area = collections.defaultdict(collections.Counter)
    unaudited_by_area = collections.Counter()
    for q, r in units.items():
        by_area[r["area"]][r["status"]] += 1
        if q in unaudited:
            unaudited_by_area[r["area"]] += 1
    for a in sorted(
        by_area,
        key=lambda a: (a not in TIER1_AREAS, a not in TIER2_AREAS, -sum(by_area[a].values())),
    ):
        c = by_area[a]
        tier = "core" if a in CORE_AREAS else ("preview" if a in PREVIEW_AREAS else "other")
        out.append(
            f"| {a} | {tier} | {c[CHECKED]} | {c[FIX_PENDING]} | {c[NOT_CHECKED]} | "
            f"{sum(c.values())} | {unaudited_by_area[a]} |"
        )

    out.append("")
    out.append("| Open defect, by what closing it takes | Units |")
    out.append("| --- | ---: |")
    for kind, n in collections.Counter(
        r["defect"].split(":")[0] for r in units.values() if r["status"] == FIX_PENDING
    ).most_common():
        out.append(f"| {kind} | {n} |")

    exp = {q: r for q, r in units.items() if "exported_as" in r}
    ec = collections.Counter(r["status"] for r in exp.values())
    out.append("")
    out.append(f"| The public path (`vfairness.X`), {len(exp)} units | Units | Share |")
    out.append("| --- | ---: | ---: |")
    for s in (CHECKED, FIX_PENDING, NOT_CHECKED):
        out.append(f"| {s} | {ec[s]} | {_pct(ec[s], len(exp))} |")
    return "\n".join(out)


STATUS_DOC = ROOT / "docs" / "CAPABILITY_STATUS.md"
QUALITY_DOC = ROOT / "docs" / "QUALITY_AND_HARDENING.md"


def _and_list(names: list) -> str:
    """Readable list, spelled as the ledger spells them."""
    if not names:
        return "not defined"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def summary_table(led: dict) -> str:
    """The three states and the Tier-1 position, for the quality page."""
    units = led["units"]
    total = len(units)
    counts = collections.Counter(r["status"] for r in units.values())
    core = collections.Counter(r["status"] for r in units.values() if r["area"] in CORE_AREAS)
    t1_total = sum(core.values())
    tier1 = core
    rows = [
        "| State | Whole public surface | Share | Core |",
        "| --- | ---: | ---: | ---: |",
    ]
    for state in (CHECKED, FIX_PENDING, NOT_CHECKED):
        rows.append(
            f"| **{state}** | {counts[state]} | {_pct(counts[state], total)} | {tier1[state]} |"
        )
    rows.append(f"| | **{total}** | 100% | **{t1_total}** |")
    rows.append("")
    rows.append(
        f"Assembled by `scripts/capability_status.py` from evidence dated "
        f"{', '.join(led.get('evidence_dates') or [])}; the newest rows are "
        f"{led.get('newest_evidence')}. **Core** is what the beta promises: every "
        f"area this project publishes a guide or a product page for, plus the "
        f"top-level namespace, which is {_and_list(sorted(CORE_AREAS))} "
        f"({t1_total} code units). The rest is preview: documented and usable, and "
        f"not covered by the beta bar."
    )
    return "\n".join(rows)


def fill_quality_summary(led: dict) -> bool:
    start, end = "<!-- STATUS:summary:start -->", "<!-- STATUS:summary:end -->"
    text = QUALITY_DOC.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{QUALITY_DOC} has lost its summary markers")
    head, _s, tail = text.partition(start)
    _old, _e, rest = tail.partition(end)
    new = f"{head}{start}\n{summary_table(led)}\n{end}{rest}"
    if new == text:
        return False
    QUALITY_DOC.write_text(new, encoding="utf-8")
    return True


def fill_markers(led: dict, path: Path = STATUS_DOC) -> bool:
    """Write the matrix into the doc between its markers. Idempotent.

    The numbers on a status page are the one thing that must never be typed by
    hand: this repository has already shipped a stale chunk count, a test count
    stale by 3,700 and a tile row adding two different countings of the same
    surface. The markers make the doc a render target, so the doc cannot drift
    from the measurement it describes.
    """
    start, end = "<!-- STATUS:matrix:start -->", "<!-- STATUS:matrix:end -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{path} has lost its matrix markers")
    head, _mid, tail = text.partition(start)
    _old, _e, rest = tail.partition(end)
    new = f"{head}{start}\n{matrix(led)}\n{end}{rest}"
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def _library_version() -> str:
    try:
        import vfairness

        return str(getattr(vfairness, "__version__", "") or "")
    except Exception:  # noqa: BLE001 - a version we cannot read is not a status
        return ""


def package_payload(led: dict) -> dict:
    """The compact ledger that ships inside the wheel, for vfairness.status()."""
    details: list = []
    seen: dict = {}
    states_by_unit: dict = {}
    detail_by_unit: dict = {}
    # Only the preview names are shipped, not a scope per unit: preview is the
    # minority and a list costs a few kilobytes where a per-row field costs a
    # megabyte in the wheel.
    preview_units: list = []
    for qual, row in led["units"].items():
        detail = row["detail"]
        if detail not in seen:
            seen[detail] = len(details)
            details.append(detail)
        states_by_unit[qual] = row["status"]
        detail_by_unit[qual] = seen[detail]
        if row.get("scope") == "preview":
            preview_units.append(qual)
    return {
        "_what_this_is": (
            "The published three-state status of every public code unit in THIS "
            "installed version, read by vfairness.status(). Generated by "
            "scripts/capability_status.py; never hand-edited."
        ),
        "library_version": _library_version(),
        "counts": led["counts"],
        "evidence_dates": led.get("evidence_dates", []),
        "states": led["states"],
        "details": details,
        "states_by_unit": states_by_unit,
        "detail_by_unit": detail_by_unit,
        "preview_units": sorted(preview_units),
        "preview_note": (
            "Documented and usable, and not yet fully covered and checked by the "
            "beta verification programme. The beta bar certifies the core surface; "
            "these sit outside it."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the published ledger is stale")
    ap.add_argument("--matrix", action="store_true", help="print the markdown matrix")
    ap.add_argument(
        "--env-check",
        action="store_true",
        help="exit 2 if this environment lacks an extra the published ledger was measured with",
    )
    args = ap.parse_args()

    if args.env_check:
        # Cheap, and run FIRST by refresh-docs.sh --check: every reader downstream
        # of the surface (register, readiness board, beta pages) otherwise turns a
        # partial install's smaller count into a list of STALE lines, which names
        # the documents as the fault when the environment is.
        published = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
        absent = K.missing_extras(published.get("extras_present"))
        if absent:
            print(
                f"COULD NOT CHECK: the published ledger was measured with the optional extras "
                f"{absent} importable and this environment lacks them. Install the extras "
                "the published-docs CI job installs and run the check again."
            )
            return 2
        return 0

    led = build()
    if args.matrix:
        print(matrix(led))
        return 0
    if args.check:
        if not LEDGER.exists():
            print("capability-status.json missing; run scripts/capability_status.py")
            return 1
        published = json.loads(LEDGER.read_text(encoding="utf-8"))
        # THREE STATES, not two. A ledger measured with the mcp and torch extras
        # importable cannot be re-measured without them: the walk then sees 1,563
        # units instead of 1,580, and printing that as STALE (as the public CI did
        # on 2026-10-02) both misnames the cause and invites a regeneration that
        # would publish a partial install's count. See K.missing_extras.
        absent = K.missing_extras(published.get("extras_present"))
        if absent:
            print(
                "COULD NOT CHECK: the published ledger was measured with the optional "
                f"extras {absent} importable and this environment lacks them, so it "
                "would measure a smaller surface than the one published. Install the "
                "extras the published-docs CI job installs and run the check again."
            )
            return 2
        problems = []
        if len(published.get("units", {})) != led["surface_total"]:
            problems.append(
                f"ledger holds {len(published.get('units', {}))} rows, surface measures {led['surface_total']}"
            )
        if published.get("counts") != led["counts"]:
            problems.append(
                f"counts stale: published {published.get('counts')}, measured {led['counts']}"
            )
        changed = [
            q
            for q, r in led["units"].items()
            if published.get("units", {}).get(q, {}).get("status") != r["status"]
        ]
        if changed:
            problems.append(f"{len(changed)} units changed status, first: {changed[:3]}")
        if not PACKAGE_LEDGER.exists():
            problems.append("src/vfairness/_capability_status.json missing; run the generator")
        else:
            shipped = json.loads(PACKAGE_LEDGER.read_text(encoding="utf-8"))
            if shipped.get("states_by_unit") != package_payload(led)["states_by_unit"]:
                problems.append(
                    "src/vfairness/_capability_status.json disagrees with the measurement, so "
                    "vfairness.status() would answer about a state that no longer holds"
                )
        for doc, filler in ((STATUS_DOC, fill_markers), (QUALITY_DOC, fill_quality_summary)):
            before = doc.read_text(encoding="utf-8")
            try:
                if filler(led):
                    doc.write_text(before, encoding="utf-8")
                    problems.append(f"{doc.name} does not match the measurement")
            except SystemExit as exc:
                problems.append(str(exc))
        for p in problems:
            print(f"STALE: {p}")
        return 1 if problems else 0

    LEDGER.write_text(json.dumps(led, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    # The site copy carries the same rows; the page needs no second source.
    SITE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    SITE_LEDGER.write_text(
        json.dumps(led, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    PACKAGE_LEDGER.write_text(
        json.dumps(package_payload(led), separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fill_markers(led)
    fill_quality_summary(led)
    c = led["counts"]
    print(
        f"{led['surface_total']} units: {c.get(CHECKED, 0)} checked, {c.get(FIX_PENDING, 0)} fix pending, {c.get(NOT_CHECKED, 0)} not checked"
    )
    print(f"wrote {LEDGER.relative_to(ROOT)} and {SITE_LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
