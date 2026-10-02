#!/usr/bin/env python3
"""Render the Beta Go-Live tables into the docs from the ledger.

Every number in the published tables is computed here from
``vfairness._proof_status``, never typed by hand. Hardcoded figures rot: this
repository has already shipped a "12,626 chunks" that was wrong for months and a
test count that was stale by 3,700. A table that claims to summarise a ledger and
disagrees with it is worse than no table, because it reads as verified.

Blocks are delimited by ``<!-- BGL:<name>:start -->`` / ``<!-- BGL:<name>:end -->``
markers in the markdown. Text outside the markers is never touched.

Usage:
    python scripts/beta_go_live_docs.py            # render
    python scripts/beta_go_live_docs.py --check    # exit 1 if a doc is stale
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

BATCH_ORDER = ("BGL-A", "BGL-B", "BGL-C", "BGL-D")


def _stamp_date() -> str:
    """The date the gate checks last ran, never the clock of a --check run.

    The "_Generated <date>_" lines were date.today(), and --check compares the
    whole page, so three pages read as stale the day after they were generated
    with not one figure changed. One source for every generated date:
    library_kpis.gate_measured_on().
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis  # noqa: PLC0415

    return library_kpis.gate_measured_on()


def _load():
    sys.path.insert(0, str(SRC))
    from vfairness._proof_status import (
        CAPABILITY_PROOF,
        MEASURED_AT_COMMIT,
        MEASURED_ON,
        PROOF_BATCHES,
    )
    from vfairness._registry import CAPABILITY_REGISTRY

    return CAPABILITY_PROOF, PROOF_BATCHES, CAPABILITY_REGISTRY, MEASURED_ON, MEASURED_AT_COMMIT


def summary_table() -> str:
    proof, batches, _reg, on, commit = _load()
    total = len(proof)
    counts = collections.Counter(r["batch"] for r in proof.values())
    # One finding can belong to two dispatch keys that resolve to the same object.
    # Count it once, or the published defect total and the hours are both inflated.
    open_defects = minutes = 0
    for key, r in proof.items():
        group = sorted([key, *r.get("shared_with", [])])
        if group[0] != key:
            continue
        open_defects += r["open_defects"]
        minutes += r["fix_minutes"]

    one_liner = {
        "BGL-A": "Refuses honestly when nothing is measurable, and a test holds it there.",
        "BGL-B": "Refuses honestly today. Nothing stops that regressing.",
        "BGL-C": "Never executed on degenerate input. Honesty unknown.",
        "BGL-D": "Proved to report a number nobody measured. Fix pending.",
    }
    rows = [
        "| Batch | Status | Capabilities | Share | What it means |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for b in BATCH_ORDER:
        n = counts.get(b, 0)
        rows.append(
            f"| **{b}** | {batches[b]['label']} | {n} | {100 * n / total:.0f}% | {one_liner[b]} |"
        )
    rows.append(f"| | **Total** | **{total}** | 100% | |")
    rows.append("")
    # TWO dates, deliberately. The counts above are CURRENT; the census that
    # produced the batches ran earlier and found far more. Printing one date
    # beside the current numbers reads as "the census found 12", when it found
    # 157 and two fix waves closed the rest. A stale or mis-attributed number is
    # fixed by dating it, not by hiding it.
    ev = _evidence()
    last = _last_stage_date(ev) or on
    census_total = (ev.get("totals") or {}).get("reproduced_public")
    found = (
        f"The census of {on} (commit `{commit}`) reproduced **{census_total} defects** "
        f"at the public API across the capabilities in the census. "
        if census_total
        else f"Census measured {on} at commit `{commit}`. "
    )
    rows.append(
        found + f"**As of {last}, {open_defects} remain open** across "
        f"{counts.get('BGL-D', 0)} capabilities. Estimated remediation "
        f"**{minutes / 60:.0f} engineer-hours**. Every count on this page is the "
        f"state at {last}, not at the census."
    )
    for line in (_stage1_line(), _stage2_line()):
        if line:
            rows.append("")
            rows.append(line)
    return "\n".join(rows)


def _stage2_line() -> str:
    """What the second wave closed, and what it deliberately did NOT count."""
    ev = _evidence()
    s2 = ev.get("stage2")
    if not s2:
        return ""
    return (
        f"**Stage 2 closed on {s2['date']}**: {s2['findings_closed']} findings fixed, of "
        f"which {s2['audit_held']} held under an independent audit told to refute them, "
        f"clearing {s2['capabilities_cleared']} capabilities. Suite {s2.get('suite', '')}. "
        f"The fixes the audit REJECTED are still counted as open defects above, including "
        f"several since repaired by hand: an unaudited fix is a claim, and this page "
        f"counts measurements. The number understates progress on purpose."
    )


def _evidence() -> dict:
    import json

    return json.loads(
        (ROOT / "docs" / "beta-go-live-census-2026-09-11.json").read_text(encoding="utf-8")
    )


def _last_stage_date(ev: dict) -> str:
    """The date the counts are TRUE for: the most recent stage that closed."""
    dates = [
        ev[k]["date"]
        for k in ev
        if k.startswith("stage") and isinstance(ev.get(k), dict) and ev[k].get("date")
    ]
    return max(dates) if dates else ""


def _stage1_line() -> str:
    """One sentence on what has been CLEARED since the census, from the evidence.

    Without it the table cites only the census commit, which now understates the
    state of the tree: the counts above already include Stage 1's fixes while the
    commit named beside them predates every one of them.
    """
    import json

    ev = json.loads(
        (ROOT / "docs" / "beta-go-live-census-2026-09-11.json").read_text(encoding="utf-8")
    )
    s1 = ev.get("stage1")
    if not s1:
        return ""
    extra = s1.get("new_defects_found_by_the_audit") or []
    tail = (
        f" The audit itself found {len(extra)} further defects, which are fixed too."
        if extra
        else ""
    )
    return (
        f"**Stage 1 closed on {s1['date']}**: {s1['findings_closed']} critical findings fixed "
        f"across {s1['capabilities_cleared']} capabilities, every one re-checked by an "
        f"independent agent told to refute it (fix real, pin able to fail, no "
        f"over-correction).{tail} The counts above already reflect that."
    )


def _ungraded_exports() -> tuple[list[str], list[str]]:
    """Exported callables that carry no proof stamp, measured at render time.

    The census graded 215 CAPABILITY ROWS. That is not the same set as the
    exported surface, and the two were reported as if they were, which is the
    defect this whole effort exists to remove applied to its own report: a
    scope nobody measured, printed beside measured counts with no way for a
    reader to tell them apart.

    Computed, never typed: a name is ungraded when it is exported in
    ``vfairness.__all__``, is a function or a class, is not a census key or a
    ``shared_with`` alias of one, and its docstring carries no proof block.
    """
    import inspect

    import vfairness

    proof, *_ = _load()
    known = set(proof)
    for record in proof.values():
        known.update(record.get("shared_with") or ())

    funcs: list[str] = []
    classes: list[str] = []
    for name in getattr(vfairness, "__all__", ()):
        obj = getattr(vfairness, name, None)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if name in known:
            continue
        if "Beta Go-Live proof status" in (inspect.getdoc(obj) or ""):
            continue
        (funcs if inspect.isfunction(obj) else classes).append(name)
    return sorted(funcs), sorted(classes)


def scope_block() -> str:
    """What the 215 covers, and what it does not."""
    proof, *_ = _load()
    funcs, classes = _ungraded_exports()
    graded = len(proof)
    lines = [
        f"**Scope of the {graded}.** The table above grades the {graded} rows of the",
        "capability census, which is not the whole exported surface. Measured at",
        "render time:",
        "",
        "| Exported callables | Count | Graded |",
        "| --- | ---: | --- |",
        f"| Carry a proof batch | {graded} | yes, one of BGL-A/B/C/D above |",
        f"| Plain functions with no batch | {len(funcs)} | **no grade at all** |",
        f"| Classes with no batch | {len(classes)} | no grade; mostly result containers, configs and exceptions |",
        "",
        f"Those **{len(funcs)} functions were never executed on degenerate input by this",
        "census**. They are NOT BGL-C: BGL-C is a measured statement about a capability",
        "that was in scope and not reached, whereas these were never in scope. Nothing",
        "on this page says anything about whether they refuse honestly, and a reader",
        "who took the 100% above as covering the library would have been misled.",
        "",
        "Grading them is the next census wave, not part of this one.",
        "",
        "<details><summary>The ungraded exported functions</summary>",
        "",
    ]
    for name in funcs:
        lines.append(f"- `{name}`")
    lines += ["", "</details>"]
    return "\n".join(lines)


def package_table() -> str:
    proof, _batches, reg, _on, _commit = _load()
    pk: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mins: collections.Counter = collections.Counter()
    for key, rec in proof.items():
        pkg = reg[key]["module_path"].split(".")[0]
        pk[pkg][rec["batch"]] += 1
        mins[pkg] += rec["fix_minutes"]
    rows = [
        "| Package | A proven | B semi | C unproven | D defect | Total | Fix (h) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pkg in sorted(pk, key=lambda p: (-pk[p]["BGL-D"], p)):
        t = pk[pkg]
        rows.append(
            f"| `{pkg}` | {t['BGL-A']} | {t['BGL-B']} | {t['BGL-C']} | {t['BGL-D']} "
            f"| {sum(t.values())} | {mins[pkg] / 60:.1f} |"
        )
    tot = collections.Counter()
    for t in pk.values():
        tot.update(t)
    rows.append(
        f"| **all** | **{tot['BGL-A']}** | **{tot['BGL-B']}** | **{tot['BGL-C']}** "
        f"| **{tot['BGL-D']}** | **{sum(tot.values())}** | **{sum(mins.values()) / 60:.0f}** |"
    )
    return "\n".join(rows)


def unproven_table() -> str:
    proof, _batches, reg, _on, _commit = _load()
    rows = ["| Capability | Symbol | Module | Why no fixture |", "| --- | --- | --- | --- |"]
    for key, rec in sorted(proof.items()):
        if rec["batch"] != "BGL-C":
            continue
        e = reg[key]
        rows.append(
            f"| `{key}` | `{e['name']}` | `{e['module_path']}` | "
            "no generic fixture reached it with data |"
        )
    return "\n".join(rows)


def batch_definitions() -> str:
    _proof, batches, _reg, _on, _commit = _load()
    out = []
    for b in BATCH_ORDER:
        d = batches[b]
        out.append(f"### {b} — {d['label']}\n")
        out.append(f"**What it means.** {d['means']}\n")
        out.append(f"**A capability is in this batch when.** {d['requires']}\n")
        out.append(f"**What you may NOT conclude.** {d['may_not_conclude']}\n")
    return "\n".join(out)


def severity_table() -> str:
    """Open defects by severity, read from the evidence file.

    Hand-typing this table is how it goes stale: the counts change the moment a
    single finding is closed, and a severity table that disagrees with the ledger
    beside it is worse than none at all.
    """
    import json

    ev = json.loads(
        (ROOT / "docs" / "beta-go-live-census-2026-09-11.json").read_text(encoding="utf-8")
    )
    meaning = {
        "critical": "reaches a graded, rendered or exported surface a reader treats as a verdict",
        "high": "fabricates a number a caller consumes, but not directly onto a badge",
        "medium": "fabricates, with another field nearby a careful reader could notice",
        "low": "cosmetic, or reachable only on a path the library discourages",
    }
    rows = ["| Severity | Findings | Fix (h) | What it means |", "| --- | ---: | ---: | --- |"]
    for sev in ("critical", "high", "medium", "low"):
        d = ev.get("severity", {}).get(sev)
        if not d:
            continue
        rows.append(f"| {sev} | {d['findings']} | {d['minutes'] / 60:.1f} | {meaning[sev]} |")
    tot_f = sum(d["findings"] for d in ev.get("severity", {}).values())
    tot_m = sum(d["minutes"] for d in ev.get("severity", {}).values())
    rows.append(f"| **all** | **{tot_f}** | **{tot_m / 60:.1f}** | |")
    return "\n".join(rows)


def grading_block() -> str:
    """The wider-surface grading state, generated so the prose cannot rot.

    These figures were typed into the markdown by hand and they rotted exactly as
    predicted: on 2026-09-27 the file still said "420 code units", "95 defect
    open" and "124 of 410 audited grades overturned", nine days and four waves
    after each of those stopped being true, in a document whose whole subject is
    values that read as measurements and are not.

    Read from scripts/library_kpis.py, the one place these are computed, so this
    block and the published site statistics cannot disagree.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis

    g = library_kpis.grading()
    if not g.get("available"):
        return (
            "> **These figures could not be generated.** "
            f"{g.get('reason', 'no grading evidence found')}. This is not a claim "
            "that nothing has been graded."
        )
    by = g["by_grade"]
    order = ("PROVEN", "SEMI-PROVEN", "NOT A MEASUREMENT", "DEFECT OPEN", "UNPROVEN")
    rows = [
        f"_Generated from the grading evidence. Newest wave: {g['measured_on']}._",
        "",
        "| State | Count |",
        "| --- | --: |",
    ]
    for state in order:
        rows.append(f"| {state.title()} | {by.get(state, 0)} |")
    for state in sorted(set(by) - set(order)):
        rows.append(f"| {state.title()} | {by[state]} |")
    rows += [
        f"| **Graded, total** | **{g['items']}** |",
        "",
        "| How settled the grades are | Count |",
        "| --- | --: |",
        f"| Current grade independently checked | {g['audited']} |",
        f"| **No independent check yet** | **{g['unaudited']}** |",
        f"| Overturned on audit, ever | {g['audit_overturned_ever']} |",
        f"| ... of which the row was later re-graded | {g['audit_overturned_superseded']} |",
        f"| Caught fabricating, by execution | {g['found_fabricating']} |",
        f"| A fix recorded against the unit | {g['fixed']} |",
        "",
        "Every wave, unmerged, so the least-checked wave is visible rather than",
        "averaged away. A unit graded twice appears in both rows, which is why",
        "these do not sum to the totals above:",
        "",
        "| Wave | Date | Graded | Caught fabricating | Fixed | Pin sabotaged | Audited | Overturned |",
        "| --- | --- | --: | --: | --: | --: | --: | --: |",
    ]
    for w in library_kpis.waves():
        aud = f"{w['audited']} of {w['rows']}" if w["audited"] else "**none**"
        over = str(w["overturned"]) if w["audited"] else "not checked"
        rows.append(
            f"| {w['label']} | {w['measured_on']} | {w['rows']} | "
            f"{w['found_fabricating']} | {w['fixed']} | {w['sabotaged']} | {aud} | {over} |"
        )
    # NOT guarded with hasattr. A first draft wrote
    # `library_kpis.kpis() if hasattr(library_kpis, "kpis") else None`, the
    # function is called build(), and the whole surface table silently vanished
    # from the rendered block with nothing saying it had. A missing figure that
    # leaves no trace is the defect this document is about. If the walker cannot
    # answer, the block says so where a reader sees it.
    try:
        surf = library_kpis._walk_public_surface()["total"]
    except Exception as exc:  # noqa: BLE001
        surf = {}
        rows += [
            "",
            f"> **The surface totals could not be measured:** `{type(exc).__name__}: "
            f"{exc}`. This is not a claim that the surface is small or fully graded.",
        ]
    if surf.get("total"):
        rows += [
            "",
            "| The surface those grades sit in | Count |",
            "| --- | --: |",
            f"| Public code units in the library | {surf['total']} |",
            f"| Graded by the capability census | {surf['graded']} |",
            f"| Graded by the wider-surface waves | {g['items']} |",
            f"| **Carrying no grade from either** | "
            f"**{surf['total'] - surf['graded'] - g['items']}** |",
            "",
            "Executed is a different and weaker measurement than graded, and the two "
            "are never added: the release gate's B1 counts a unit as executed once "
            "any of the census, the waves, the probes or the test suite has run it "
            "and recorded what came back. Reproduce both with "
            "`python scripts/release_gate.py`.",
        ]
    rows += [
        "",
        f"**{g['unaudited']} unaudited grades is the largest thing open here.** "
        f"{g['audit_overturned_ever']} grades have been overturned when somebody "
        "independent attacked them, better than one in five of those checked, so an "
        "unaudited grade is a claim with evidence behind it rather than a settled "
        "measurement. The two overturn figures answer different questions: "
        f"{g['audit_overturned']} rows carry an overturn AND still carry their "
        f"audited grade, while {g['audit_overturned_superseded']} more were "
        "re-graded by a later wave, which resets them to unaudited and used to "
        "delete the record of the overturn with them.",
    ]
    return "\n".join(rows)


def remaining_block() -> str:
    """The ungraded remainder, generated, because it moves on every wave.

    RELEASE_PLAN.md stated it as a literal in four places. It read 747, then 472,
    then 473 within two days, and each change broke the pin in
    tests/test_release_gate.py that exists to stop the plan quoting a figure the
    repository contradicts. A plan is the wrong document to keep a hand-typed
    denominator in: it is read by somebody deciding whether to publish.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis
    import release_gate

    crit = {c["id"]: c for c in release_gate._criteria()}
    g1 = crit.get("G1", {})
    try:
        graded, total = (int(x) for x in str(g1.get("measured", "")).replace(" of ", " ").split())
    except ValueError:
        return (
            "> **The remaining count could not be measured.** G1 reported "
            f"`{g1.get('measured')}`, which this block cannot parse. That is not a "
            "claim that nothing remains."
        )
    grading = library_kpis.grading()
    surf = library_kpis._walk_public_surface()["total"]
    b1 = next(
        (c["measured"] for c in release_gate._beta_criteria() if c["id"] == "B1"),
        "not reported",
    )
    return "\n".join(
        [
            f"_Generated {_stamp_date()} from `scripts/release_gate.py`._",
            "",
            f"**{total - graded}** of the **{total}** public code units carry no grade.",
            f"{graded} do: {surf['graded']} from the capability census and "
            f"{grading['items']} from the wider-surface waves.",
            "",
            "Executed is a separate and weaker measurement, and the two are never "
            "added: criterion B1 reads "
            f"`{b1}`. Running a unit is not judging whether it refuses honestly.",
            "",
            f"**{grading['unaudited']}** of the grades that do exist have had no "
            f"independent check, and {grading['audit_overturned_ever']} of "
            f"{grading['audited_ever']} grades were overturned when one was made. That "
            "queue, not the ungraded remainder, is the nearest thing to a blocker.",
        ]
    )


def gate_block() -> str:
    """The release gate's verdict and every criterion, generated.

    QUALITY_AND_HARDENING.md carried this as prose under a hand-typed date, and it
    rotted the way a hand-typed readout always does: "Run on 2026-09-25 ... Four of
    the six beta criteria pass ... B2, open defects: 63 ... B4: 215 grades". Within
    two days B2 had reached 0 and gone back up on an audit, B4 had passed 500 and
    come back down, and B1 had failed and passed again. A page that tells a reader
    whether the library is ready to publish is the worst place in the repository for
    a stale number.

    THREE STATES PER CRITERION, never two. A criterion whose measurement could not
    be taken reads "could not be measured" and is counted as blocking; printing
    "fails" for it would claim a measurement nobody took.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    beta, full = release_gate._beta_criteria(), release_gate._criteria()
    lines = [f"_Generated {_stamp_date()} by `scripts/release_gate.py`._", ""]
    for name, crit in (("Beta bar", beta), ("Full release bar", full)):
        passed = sum(1 for c in crit if c["passes"])
        verdict = "READY" if passed == len(crit) else "NOT READY"
        lines += [
            f"**{name}: {verdict}.** {passed} of {len(crit)} criteria pass.",
            "",
            "| | Criterion | Measured | Verdict |",
            "| --- | --- | --- | --- |",
        ]
        for c in crit:
            measured = str(c.get("measured", ""))
            cannot = measured.startswith("cannot measure")
            state = (
                "passes"
                if c["passes"]
                else ("**could not be measured, counted as blocking**" if cannot else "**fails**")
            )
            lines.append(f"| {c['id']} | {c['name']} | {'' if cannot else measured} | {state} |")
        lines.append("")
    lines += [
        "Reproduce with `python scripts/release_gate.py`. Every figure above is read "
        "from the recorded evidence, so a criterion cannot pass by assertion.",
    ]
    return "\n".join(lines)


#: WHAT CLOSES EACH BETA CRITERION, in the reader's terms rather than the gate's.
#: The gate says what is measured; this says what would have to happen, and whether
#: it is WORK (finite, known, someone does it) or a DECISION (nobody can do it, a
#: person has to choose). A reader looking at a red gate needs to know which.
_CLOSES = {
    "B1": ("work", "one coverage run over the units nothing has executed"),
    "B1b": ("work", "regenerate the published state"),
    "B2": ("work", "fix the open defects in code that returns a fairness number"),
    "B2b": ("work", "fix the second-round audit's open records, each one reproducible"),
    "B3": ("work", "assess a severity for each remaining open defect"),
    "B4": (
        "decision",
        "audit the unaudited grades, OR decide a beta may ship with them unaudited "
        "and published as such",
    ),
    "B5": ("work", "close the capability census"),
}


def scope_table_block() -> str:
    """Core against Preview, generated.

    This table was typed, and on 2026-09-28 it read 1,382 units / 884 checked / 442
    not checked for core and a fix-pending of 0 for preview, against a live ledger of
    1,387 / 1,160 / 171 and 8. Daniel found the rendered twin of it by adding up the
    row. A table a reader uses to decide what the beta covers is the worst place in
    this repository for a number somebody retyped, so it is read from the ledger and
    the rows are asserted to sum by tests/test_kpi_scope_rows_add_up.py.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis

    led = library_kpis._ledger()
    rows = [
        ("**Core**, what the beta promises", "core"),
        ("**Preview**, documented but not certified", "preview"),
    ]
    out = [
        "| Scope | Units | Checked | Fix pending | Not checked |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        out.append(
            f"| {label} | {led[f'{key}_total']:,} | {led[f'{key}_checked']:,} | "
            f"{led[f'{key}_fix_pending']:,} | **{led[f'{key}_not_checked']:,}** |"
        )
    out.append(
        f"| _total_ | {led['total']:,} | {led['checked']:,} | {led['fix_pending']:,} | "
        f"{led['not_checked']:,} |"
    )
    return "\n".join(out)


def readiness_block() -> str:
    """The one screen that answers: can we ship, and what is stopping us.

    The gate block further down this page lists every criterion for both bars. That
    is the record. This is the DECISION: what is red, whether each red one is work
    or a judgement, and which single question a person has to answer. Daniel asked
    for it after reading the page and being unable to tell where we are, which is a
    fair thing to say about a page that carries several hundred numbers.

    Dense on purpose. Generated from scripts/release_gate.py, so it cannot say
    something the gate does not.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    beta = release_gate._beta_criteria()
    passed = [c for c in beta if c["passes"]]
    failed = [c for c in beta if not c["passes"]]
    work = [c for c in failed if _CLOSES.get(c["id"], ("", ""))[0] == "work"]
    decisions = [c for c in failed if _CLOSES.get(c["id"], ("", ""))[0] == "decision"]

    verdict = "READY TO CUT A BETA" if not failed else "NOT READY"
    lines = [
        f"**{verdict}.** {len(passed)} of {len(beta)} beta criteria pass; "
        f"{len(failed)} block it. _Generated {_stamp_date()} from "
        "`scripts/release_gate.py`._",
        "",
        "| | Criterion | Measured now | Closes when |",
        "| --- | --- | --- | --- |",
    ]
    for c in failed + passed:
        mark = "PASS" if c["passes"] else "**BLOCKS**"
        kind, closes = _CLOSES.get(c["id"], ("", ""))
        note = "" if c["passes"] else closes
        lines.append(f"| {mark} | `{c['id']}` {c['name']} | {c.get('measured', '')} | {note} |")
    lines.append("")

    if decisions:
        lines += [
            "**The decision point is "
            + ", ".join(f"`{c['id']}`" for c in decisions)
            + ".** "
            + " ".join(
                f"{c['name']}: {c.get('measured', '')}. {_CLOSES[c['id']][1]}." for c in decisions
            ),
            "",
            "Everything else that is red is WORK: finite, known, and nobody has to "
            "choose anything for it to get done"
            + (" (" + ", ".join(f"`{c['id']}`" for c in work) + ")." if work else "."),
            "",
        ]
    elif work:
        lines += [
            "Nothing here is a judgement call. Everything red is WORK: "
            + ", ".join(f"`{c['id']}`" for c in work)
            + ".",
            "",
        ]
    return "\n".join(lines)


def second_round_block() -> str:
    """How much of the SECOND-ROUND audit is closed, from its own register.

    The BGL5 fix wave was audited again, and that audit's findings live in
    tests/test_bgl6_*.py, one test per recorded claim with the measurement in its
    docstring. None of it was published, so the progress could only be seen by
    reading six files. Read from docs/bgl6-audit-register.json, which
    scripts/bgl6_register.py generates from the audit files themselves.
    """
    path = ROOT / "docs" / "bgl6-audit-register.json"
    if not path.exists():
        return (
            "The second-round audit register is not present. Run `python scripts/bgl6_register.py`."
        )
    reg = json.loads(path.read_text(encoding="utf-8"))
    t = reg["totals"]
    pct = (100.0 * t["closed"] / t["recorded"]) if t["recorded"] else float("nan")
    lines = [
        f"The BGL5 fix wave was audited a **second time**, and that audit recorded "
        f"**{t['recorded']} claims**. **{t['closed']} are closed** and "
        f"**{t['open']} are still open** ({pct:.0f}% closed, as of "
        f"{reg['generated_on']}).",
        "",
        "A claim is CLOSED when its defect is fixed and its test has been inverted "
        "into a pin, in the same commit. OPEN means the test still records the "
        "defective value the unit produces today, with the measurement in its "
        "docstring, so every open row below is reproducible rather than suspected.",
        "",
        "| Batch | Subject | Recorded | Closed | Open |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tag, b in reg["batches"].items():
        lines.append(
            f"| `{tag}` | {b['subject']} | {b['recorded']} | {b['closed']} | {b['open']} |"
        )
    lines.append(f"| **total** | | **{t['recorded']}** | **{t['closed']}** | **{t['open']}** |")
    lines += [
        "",
        "The register is `docs/bgl6-audit-register.json`, generated by "
        "`scripts/bgl6_register.py` from the audit files, so this table cannot drift "
        "from them: each file declares which of its tests still record an open "
        "defect, and a name that is not a test in that file is refused.",
    ]
    return "\n".join(lines)


BLOCKS = {
    "summary": summary_table,
    "severity": severity_table,
    "packages": package_table,
    "scope": scope_block,
    "unproven": unproven_table,
    "definitions": batch_definitions,
    "grading": grading_block,
    "remaining": remaining_block,
    "gate": gate_block,
    "second_round": second_round_block,
    "readiness": readiness_block,
    "scope_table": scope_table_block,
}


def _label(path: Path) -> str:
    """A path for a message, never an exception.

    ``Path.relative_to(ROOT)`` raises ValueError for anything outside the
    repository, so the lines that REPORT a broken document crashed when handed one,
    which is the reporting path failing exactly when it has something to report.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def render(path: Path, check: bool) -> bool:
    """True if the file is (or was made) current.

    A HALF-PRESENT MARKER PAIR IS AN ERROR, not a skip. `if begin not in out:
    continue` was the whole test, so deleting a start marker silently disabled its
    block and --check answered "docs are current" over a gutted table. Found
    2026-09-27 by sabotaging the gate block: removing the opening marker left the
    generated verdict replaced by one hand-typed sentence and the checker saw
    nothing. A gate that cannot notice its own subject going missing is the defect
    this repository is audited for, committed in the generator.
    """
    text = path.read_text(encoding="utf-8")
    out = text
    for name, fn in BLOCKS.items():
        begin, end = f"<!-- BGL:{name}:start -->", f"<!-- BGL:{name}:end -->"
        has_begin, has_end = begin in out, end in out
        if has_begin != has_end:
            missing = begin if has_end else end
            print(
                f"  BROKEN MARKERS in {_label(path)}: block {name!r} has "
                f"one marker and not the other ({missing} is absent), so its content "
                "is neither generated nor checked.",
                file=sys.stderr,
            )
            return False
        if not has_begin:
            continue
        i, j = out.index(begin) + len(begin), out.index(end)
        if j < i:
            print(
                f"  BROKEN MARKERS in {_label(path)}: block {name!r} has "
                "its end marker before its start marker.",
                file=sys.stderr,
            )
            return False
        out = out[:i] + "\n" + fn() + "\n" + out[j:]
    if out == text:
        return True
    if check:
        print(f"  STALE: {_label(path)}", file=sys.stderr)
        return False
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(out)
    os.replace(tmp, path)
    print(f"  rendered {_label(path)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ok = True
    for rel in (
        "docs/BETA_GO_LIVE_PLAN.md",
        "docs/QUALITY_AND_HARDENING.md",
        "docs/RELEASE_PLAN.md",
        "docs/BETA.md",
    ):
        p = ROOT / rel
        if p.exists():
            ok = render(p, args.check) and ok
    if args.check:
        print("docs are current" if ok else "docs are stale: run scripts/beta_go_live_docs.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
