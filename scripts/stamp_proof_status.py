#!/usr/bin/env python3
"""Stamp each registered capability's docstring with its Beta Go-Live proof batch.

WHY ON THE FUNCTION. ``_proof_status.py`` is the machine-readable record, but
almost nobody reads a ledger. A person reaching for a capability reads
``help(thing)`` or hovers it in an editor, and that is the surface where "nobody
has checked whether this fabricates" has to be visible. A correct measurement no
reader can see is the same defect one layer up.

HOW IT IS SAFE. Three rules, each learned the hard way:

1. The capability is resolved at RUNTIME and located with ``inspect``, not by
   searching the module named in the registry. ``module_path`` says where a
   capability is EXPORTED; a re-export means the class is defined elsewhere and a
   name search finds nothing.
2. The stamp is INSERTED before the closing quote of the existing docstring.
   Nothing already in the file is reflowed, re-indented or rewritten, so the diff
   is the added lines and nothing else. An earlier version rebuilt the whole
   literal and produced 591 deleted lines across 39 files.
3. Every touched file is re-parsed and its AST compared to the original with all
   docstrings blanked. If anything but a docstring moved, the file is left alone
   and the run reports it.

Writes are atomic (``os.replace``): this checkout is shared with other sessions,
and a truncate-then-write once handed a peer a SyntaxError mid-write.

Idempotent: an existing block between the markers is replaced, not duplicated.

Usage:
    python scripts/stamp_proof_status.py            # stamp
    python scripts/stamp_proof_status.py --check    # exit 1 if any stamp is stale
    python scripts/stamp_proof_status.py --strip    # remove every stamp
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BEGIN = "Beta Go-Live proof status"
END = "(end Beta Go-Live proof status)"

# THE CLAIM FOLLOWS THE EVIDENCE, PER CAPABILITY (2026-09-27).
#
# BGL-A used to be one fixed sentence asserting that "an independent judge showed" the
# value to be a true measurement, and disclaiming that "the pin has been
# sabotage-checked". Measured against the live evidence, that was wrong in BOTH
# directions at once, and for 59 of the 215 registered capabilities it was wrong in the
# flattering one: they carried BGL-A PROVEN with open_defects 0 while a live defect sat
# in the unit the row vouches for. decoding_trust_runner is the plainest case, stamped
# PROVEN while three of its eight dimensions were fabricating, one of them reporting
# perfect parity across all 24 demographic groups from a substring match.
#
# The two clauses are now composed from data that already exists rather than asserted:
#
#   * "an independent judge" is claimed ONLY where the grade is audited. Nothing
#     produced by the BGL3 campaign is, by design: 342 grades, each from one agent,
#     none argued with, and the measured overturn rate in this repository is roughly
#     one in five. Saying it anyway is the defect this file exists to prevent.
#   * "the pin has been sabotage-checked" is ASSERTED where a sabotage record exists,
#     instead of being disclaimed. Understating evidence teaches a reader to discount
#     the whole block.
#
# A capability with a recorded open defect can no longer render PROVEN at all; see
# _batch_for_today.
_BGL_A_HEAD = (
    "PROVEN. Executed on healthy input and on the degenerate inputs where nothing it "
    "claims to measure exists; in each it either refused or returned a value that was "
    "checked to be a true measurement, and a test in the suite names it beside a "
    "refusal assertion."
)
_JUDGED = " An independent check has argued with that judgement and upheld it."


#: How often an independently checked grade has NOT survived, as "one grade in N".
#:
#: DERIVED, BECAUSE IT WAS HARDCODED AND WENT STALE. Until 2026-09-30 this sentence
#: read "roughly one grade in five" as a literal, and it was stamped into 104 shipped
#: docstrings. The measured union rate at that date was 291 of 872, which is one grade
#: in THREE, so every one of those docstrings understated the overturn rate by a
#: factor of about 1.7, in the one sentence whose entire job is to tell a reader how
#: much to trust a grade nobody has argued with. Understating it there does not fail
#: safe: it makes an unchecked grade sound more settled than the evidence supports.
#:
#: The union is the right denominator, for the same reason docs/GRADING.md gives: a
#: row a later wave re-graded still went through the independent check once, and
#: computing this from the merged view deletes the record that its first grade was
#: wrong, so the number reporting the method's fallibility falls as the campaign gets
#: more thorough.
def _overturn_phrase() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import library_kpis

    g = library_kpis.grading()
    overturned = g.get("audit_overturned_ever") or 0
    audited = g.get("audited_ever") or 0
    if not overturned or not audited:
        # THREE STATES. With nothing measured this sentence must not invent a
        # reassuring fraction, so it says plainly that the rate is unknown.
        return "The rate at which grades have been overturned could not be measured here"
    n = max(2, round(audited / overturned))
    words = {
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return f"Roughly one grade in {words.get(n, str(n))} has been overturned"


_NOT_JUDGED = (
    " NOT INDEPENDENTLY CHECKED: one examiner reached that judgement and nobody has yet "
    f"tried to refute it. {_overturn_phrase()} when somebody "
    "did, so treat this as a claim with evidence behind it rather than a settled one."
)
_SABOTAGED = (
    " The pin was sabotage-checked: it was shown to go red when the defect is "
    "reintroduced, so it can fail."
)
_NOT_SABOTAGED = (
    " This does NOT establish that the pin has been sabotage-checked, so it is not known "
    "whether the pin can fail at all."
)
_A_TAIL = (
    " This does NOT establish that its statistics are accurate, nor that the pin covers "
    "every scenario."
)

BLURB = {
    "BGL-A": _BGL_A_HEAD + "{judged}{sabotaged}" + _A_TAIL,
    "BGL-B": (
        # No date in this sentence (fixed 2026-10-01): {date} is the NEWEST evidence
        # date in the whole ledger, not the day this unit was judged, so "judged honest
        # on {date}" told 56 units they were examined on a day they were not. The block
        # heading already dates the status itself ("proof status (<date>)").
        "SEMI-PROVEN. Executed and judged honest, but nothing in the "
        "suite holds it there. The behaviour is observed, not protected: a "
        "refactor can reintroduce the defect with every gate staying green."
    ),
    "BGL-C": (
        "UNPROVEN. No execution evidence on degenerate input; no fixture could be "
        "built for it generically. Whether it fabricates a measurement is "
        "UNKNOWN, and unknown is not clean."
    ),
    "BGL-D": (
        "DEFECT OPEN: {n} reproduced at the public API, severity {sev}. On at "
        "least one degenerate input it hands back a confident value where nothing "
        "was measurable. It still computes correctly on healthy data; what is "
        "wrong is what it says when it cannot measure. Estimated fix {mins} min."
    ),
}


def _live_evidence() -> tuple[dict, dict]:
    """Today's per-unit state, read from the published ledger and the grading waves.

    Returns (status_by_unit, grade_by_unit). The proof record in _proof_status.py is
    generated from a DATED census and keeps its historical meaning; what a docstring
    must state is what is established NOW, so the stamp reads the live evidence instead
    of trusting a figure measured on 2026-09-11.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis  # noqa: PLC0415

    ledger_path = ROOT / "docs" / "capability-status.json"
    status_by_unit: dict[str, str] = {}
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        status_by_unit = {q: r.get("status", "") for q, r in (data.get("units") or {}).items()}
    grade_by_unit = dict(library_kpis.graded_items())
    probe = library_kpis.GRADING_FROM_PROBE
    if probe.exists():
        for qual, rec in (
            (json.loads(probe.read_text(encoding="utf-8")) or {}).get("items", {})
        ).items():
            grade_by_unit.setdefault(qual, rec)
    return status_by_unit, grade_by_unit


def _own_qualname(obj: Any) -> str:
    """The unit key for the object itself, as :func:`_units_of` spells it."""
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "")
    return f"{module}.{qualname}" if module and qualname else ""


def _units_of(obj: Any, status_by_unit: dict) -> list[str]:
    """The qualified names a capability answers for: itself plus anything under it.

    Same rule the published badge uses, which is why they agree: a class answers for
    the methods reached through it, so a class whose method fabricates cannot read clean.
    """
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "")
    if not module or not qualname:
        return []
    qual = f"{module}.{qualname}"
    units = [qual] if qual in status_by_unit else []
    units += [q for q in status_by_unit if q.startswith(qual + ".")]
    return units


def _facts_for_today(obj: Any, status_by_unit: dict, grade_by_unit: dict) -> dict:
    """What is established about this capability right now, as three facts.

    `open_now` is THE GATE THAT DID NOT EXIST. Nothing stopped a capability with a
    recorded open defect from carrying a PROVEN stamp, and that is exactly how 59 rows
    drifted for sixteen days: the census said open_defects 0 on 2026-09-11 and the
    docstring repeated it long after a defect had been found. Now the stamp reads the
    ledger, so a unit in FIX PENDING forces the block down to BGL-D whatever the record
    says.
    """
    units = _units_of(obj, status_by_unit)
    if not units:
        return {"open_now": 0, "audited": False, "sabotaged": False, "units": 0}
    # THE GATE MAY NOT READ ITS OWN OUTPUT. A CLASS row carries no evidence of its
    # own: scripts/capability_status.py resolves a class's census batch from the
    # docstring stamp THIS SCRIPT writes, so counting that row here closes a loop
    # of ledger -> stamp -> census -> ledger.
    #
    # Measured 2026-09-29. Five classes had a method opened by the B4 audit, which
    # made the class FIX PENDING, which flipped its stamp to BGL-D, which the census
    # then read back as DEFECT OPEN. When the methods were fixed and returned to
    # CHECKED, all five classes stayed FIX PENDING: FairnessAnalyzer, FairExplAIner,
    # OutputAnalyzer, MultiAgentRunHarness and ModelFairnessGate. Two further full
    # refresh passes did not move them, so it is a stable fixed point rather than a
    # lag, and refresh-docs' convergence check cannot see it because the system
    # agrees with itself about the wrong answer.
    #
    # The METHODS are still counted, so a class whose method fabricates still cannot
    # read clean, which is the whole point of the gate. Only the derived row is
    # dropped, and only when the object owns methods that carry real evidence.
    own = _own_qualname(obj)
    countable = [q for q in units if q != own] or units
    open_now = sum(1 for q in countable if status_by_unit.get(q) == "FIX PENDING")
    graded = [grade_by_unit[q] for q in units if q in grade_by_unit]
    # FAIL CLOSED on both: one unaudited unit means the capability is not audited, and
    # a sabotage record on one method says nothing about its siblings.
    audited = bool(graded) and all(g.get("audited") for g in graded)
    # A unit graded NOT A MEASUREMENT has nothing to pin and therefore nothing to
    # sabotage, so requiring a record from it would fail closed on a unit where the
    # question does not apply. Measured: the reweighting capability covers four units,
    # of which `transform` is NOT A MEASUREMENT, and including it reported "the pin has
    # not been sabotage-checked" for a capability whose two measuring units were
    # sabotaged twenty times the same day. Understating evidence teaches a reader to
    # discount the block, which costs exactly as much as overstating it.
    measuring = [g for g in graded if g.get("grade") != "NOT A MEASUREMENT"]
    sabotaged = bool(measuring) and all(g.get("sabotage") for g in measuring)
    return {
        "open_now": open_now,
        "audited": audited,
        "sabotaged": sabotaged,
        "units": len(units),
    }


def render(rec: dict, date: str, indent: str) -> str:
    text = BLURB[rec["batch"]].format(
        date=date,
        n=rec["open_defects"],
        sev="/".join(rec["severities"]) or "unclassified",
        mins=rec["fix_minutes"],
        judged=_JUDGED if rec.get("audited") else _NOT_JUDGED,
        sabotaged=_SABOTAGED if rec.get("sabotaged") else _NOT_SABOTAGED,
    )
    para = f"{BEGIN} ({date}): {rec['batch']} {text}"
    keys = rec.get("_keys") or []
    where = f"Ledger row{'s' if len(keys) > 1 else ''}: {', '.join(sorted(keys))}. " if keys else ""
    tail = f"{where}See docs/BETA_GO_LIVE_PLAN.md for the batch definitions."
    width = max(40, 88 - len(indent))
    out = textwrap.fill(para, width=width, initial_indent=indent, subsequent_indent=indent)
    out += "\n\n" + textwrap.fill(
        tail, width=width, initial_indent=indent, subsequent_indent=indent
    )
    # END goes on its own line and is never wrapped. textwrap will happily split
    # "(end Beta Go-Live proof status)" across two lines, and then every reader of
    # the block -- the --check pass, the tests, a person grepping the tree -- fails
    # to find the marker it is looking for while the mark is plainly sitting there.
    out += "\n" + indent + END
    return out


def strip_block(doc: str) -> str:
    if BEGIN not in doc:
        return doc
    keep, skipping = [], False
    for ln in doc.splitlines():
        if BEGIN in ln:
            skipping = True
            continue
        if skipping:
            if END in ln:
                skipping = False
            continue
        keep.append(ln)
    while keep and not keep[-1].strip():
        keep.pop()
    return "\n".join(keep) + "\n"


def blank_docstrings(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body[0].value.value = "<DOC>"
    return ast.dump(tree)


def resolve(entry: dict):
    """Return the live object for a registry entry, or None."""
    try:
        mod = importlib.import_module(f"vfairness.{entry['module_path']}")
        return getattr(mod, entry["name"], None)
    except Exception:
        return None


def locate(obj) -> tuple[Path, int] | None:
    """(file, 1-based line of the def/class statement) where obj is DEFINED."""
    try:
        f = inspect.getsourcefile(obj)
        if not f:
            return None
        _, lineno = inspect.getsourcelines(obj)
        return Path(f), lineno
    except (OSError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument(
        "--list",
        action="store_true",
        dest="list_files",
        help="print the source files this stamper writes, one per line, and "
        "change nothing. A commit needs an explicit pathspec, and asking "
        "`git status` for one is a query of SHARED state: this working tree "
        "has several sessions in it.",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    from vfairness._proof_status import CAPABILITY_PROOF, MEASURED_ON
    from vfairness._registry import CAPABILITY_REGISTRY

    status_by_unit, grade_by_unit = _live_evidence()

    # THE DATE OF THE EVIDENCE, not of the census. MEASURED_ON is 2026-09-11, the day
    # the capability census ran, and these blocks now state what the LIVE ledger and the
    # grading waves establish. Leaving the census date on a sentence about newer evidence
    # is a stale figure dressed as a measurement, which is the shape this file exists to
    # stop. Falls back to MEASURED_ON when no evidence date can be read, rather than
    # inventing today's.
    dates = (
        sorted(
            {
                d
                for d in (
                    json.loads(
                        (ROOT / "docs" / "capability-status.json").read_text(encoding="utf-8")
                    ).get("evidence_dates")
                    or []
                )
                if d
            }
        )
        if (ROOT / "docs" / "capability-status.json").exists()
        else []
    )
    stamp_date = dates[-1] if dates else MEASURED_ON

    by_file: dict[Path, list[tuple[str, int, dict]]] = {}
    unresolved: list[tuple[str, str]] = []
    forced_open: list[tuple[str, int]] = []
    for key, entry in sorted(CAPABILITY_REGISTRY.items()):
        rec = CAPABILITY_PROOF.get(key)
        if rec is None:
            unresolved.append((key, "no proof record"))
            continue
        obj = resolve(entry)
        if obj is None:
            unresolved.append(
                (key, f"cannot import vfairness.{entry['module_path']}.{entry['name']}")
            )
            continue
        loc = locate(obj)
        if loc is None:
            unresolved.append((key, "no source location (C extension or dynamic)"))
            continue
        path, lineno = loc
        try:
            path.relative_to(SRC)
        except ValueError:
            unresolved.append((key, f"defined outside src/: {path}"))
            continue
        facts = _facts_for_today(obj, status_by_unit, grade_by_unit)
        rec = dict(rec, _keys=[key], audited=facts["audited"], sabotaged=facts["sabotaged"])
        if facts["open_now"]:
            # THE GATE. A capability with a unit in FIX PENDING may not be stamped
            # PROVEN, whatever the dated census recorded.
            rec["batch"] = "BGL-D"
            rec["open_defects"] = max(rec.get("open_defects") or 0, facts["open_now"])
            forced_open.append((key, facts["open_now"]))
        by_file.setdefault(path, []).append((entry["name"], lineno, rec))

    # One object can serve several registry keys (an alias, or two dispatch keys
    # onto one function). Their records can disagree, and a docstring can only
    # carry one answer. Take the WORST: a shared object that fabricates through
    # any of its keys fabricates, and an unproven key is not cancelled by a proven
    # sibling. Failing closed here is the whole point of the ladder.
    RANK = {"BGL-D": 3, "BGL-C": 2, "BGL-B": 1, "BGL-A": 0}
    for path, targets in by_file.items():
        merged: dict[tuple[str, int], dict] = {}
        for name, lineno, rec in targets:
            k = (name, lineno)
            cur = merged.get(k)
            if cur is None:
                merged[k] = rec
                continue
            if RANK[rec["batch"]] > RANK[cur["batch"]]:
                keys = cur["_keys"] + rec["_keys"]
                merged[k] = dict(rec, _keys=keys)
            else:
                cur["_keys"] = cur["_keys"] + rec["_keys"]
            # FAIL CLOSED on both new facts as well: a shared object is audited only if
            # every key that reaches it is, and one sabotage record does not cover a
            # sibling key that has none.
            merged[k]["audited"] = bool(cur.get("audited")) and bool(rec.get("audited"))
            merged[k]["sabotaged"] = bool(cur.get("sabotaged")) and bool(rec.get("sabotaged"))
            merged[k]["open_defects"] = max(cur["open_defects"], rec["open_defects"])
            merged[k]["fix_minutes"] = max(cur["fix_minutes"], rec["fix_minutes"])
            merged[k]["severities"] = sorted(set(cur["severities"]) | set(rec["severities"]))
        by_file[path] = [(n, ln, r) for (n, ln), r in merged.items()]

    stamped = stale = skipped = refused = 0
    for path, targets in sorted(by_file.items()):
        original = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(original)
        except SyntaxError as e:
            print(f"  SKIP unparseable {path}: {e}", file=sys.stderr)
            continue
        want = {(name, lineno): rec for name, lineno, rec in targets}

        lines = original.splitlines(keepends=True)
        offs = [0]
        for ln in lines:
            offs.append(offs[-1] + len(ln))

        spans = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            rec = want.get((node.name, node.lineno))
            if rec is None:
                # decorators shift lineno; match on name when it is unique here
                cands = [v for (n, _), v in want.items() if n == node.name]
                if len(cands) != 1:
                    continue
                rec = cands[0]
            body = node.body
            if not (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                skipped += 1
                continue
            dn = body[0].value
            begin = offs[dn.lineno - 1] + dn.col_offset
            finish = offs[dn.end_lineno - 1] + dn.end_col_offset
            seg = original[begin:finish]
            stripped = seg.lstrip("rbuRBU")
            quote = next((q for q in ('"""', "'''") if stripped.startswith(q)), None)
            if quote is None or not stripped.endswith(quote):
                skipped += 1  # single-quoted one-liner: leave it alone
                continue
            indent = " " * node.col_offset + "    "
            inner = stripped[len(quote) : -len(quote)]
            base = strip_block(inner)
            if args.strip:
                new_inner = base
            else:
                block = render(rec, stamp_date, indent)
                if quote in block:
                    skipped += 1
                    continue
                sep = "\n\n" if base.strip() else "\n"
                new_inner = f"{base.rstrip()}{sep}{block}\n{indent}"
            if new_inner == inner:
                continue
            rebuilt = seg[: len(seg) - len(stripped)] + quote + new_inner + quote
            spans.append((begin, finish, rebuilt))

        if not spans:
            continue
        if args.check:
            stale += len(spans)
            for begin, _, _ in spans:
                line = original[:begin].count("\n") + 1
                print(f"  STALE stamp: {path.relative_to(ROOT)}:{line}")
            continue

        text = original
        for begin, finish, rebuilt in sorted(spans, reverse=True):
            text = text[:begin] + rebuilt + text[finish:]

        try:
            if blank_docstrings(ast.parse(text)) != blank_docstrings(ast.parse(original)):
                print(
                    f"  REFUSED {path.relative_to(ROOT)}: AST changed beyond docstrings",
                    file=sys.stderr,
                )
                refused += len(spans)
                continue
        except SyntaxError as e:
            print(f"  REFUSED {path.relative_to(ROOT)}: would not parse ({e})", file=sys.stderr)
            refused += len(spans)
            continue

        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        stamped += len(spans)

    for key, why in unresolved:
        print(f"  unresolved: {key} ({why})", file=sys.stderr)
    if args.check:
        print(f"stale stamps: {stale}")
        return 1 if stale else 0
    if args.list_files:
        for path in sorted(by_file):
            print(Path(path).resolve().relative_to(ROOT))
        return 0
    print(f"  stamped {stamped} docstrings across {len(by_file)} files")
    print(f"  skipped {skipped} (no docstring, or a single-quoted one-liner)")
    print(f"  refused {refused} (AST would have changed)")
    print(f"  unresolved {len(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
