#!/usr/bin/env python3
"""Publish how much of the SECOND-ROUND audit is closed, from the audit files.

WHY THIS EXISTS. The BGL5 fix wave was audited a second time, and the audit's
findings are in ``tests/test_bgl6_*.py``: one test per recorded claim, each carrying
the measured before-state in its docstring. Nothing published that body of work, so
the only way to see how much of it was closed was to read six files. A number nobody
can see is the same problem this whole campaign is about, one level up.

WHY THE STATE IS DECLARED AND NOT MEASURED. A test that RECORDS a defect asserts the
value the unit produces today, so it PASSES while the defect is live. A pin asserts
the corrected value, so it also passes. The two are indistinguishable by execution,
which is why each audit file declares ``RECORDED_DEFECTS_STILL_OPEN`` and why
``tests/test_bgl6_register_is_honest.py`` refuses a name in that list which is not a
test function in the module. The list shrinks in the same commit that fixes a defect.

Usage:
    python scripts/bgl6_register.py            # write docs/bgl6-audit-register.json
    python scripts/bgl6_register.py --check     # exit 1 if the published file is stale
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
OUT = ROOT / "docs" / "bgl6-audit-register.json"


def _stamp_date() -> str:
    """The date the gate checks last ran, never the clock of a --check run.

    This was date.today() in three places, and --check compared all three, so a
    register regenerated one day read as stale the next with nothing changed. One
    source for every generated date: library_kpis.gate_measured_on().
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis  # noqa: PLC0415

    return library_kpis.gate_measured_on()


#: What each batch attacked, for a reader who is not going to open the file.
SUBJECTS = {
    "f01": "refusal scoring and the explainer's root predicate",
    "f02": "mitigation (label massager, residual transformer, suppressor, resampler, reweighting) and the DecodingTrust runners",
    "f03": "the CI/CD release gate and the fairness monitor",
    "f04": "XAI adversarial probes, threshold analysis and calibration",
    "f06": "compliance mapping, the signed test log and the report canvas",
    "f12": "data balancing, benchmark comparison, LIME and the task worker",
}


def _module_facts(path: Path) -> dict:
    """Test names and the declared open list, read from the AST.

    The AST, not an import: these modules import the library and some of them the
    optional extras, and a reporting script must not need either.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # A CONTROL IS NOT A RECORDED CLAIM. Fixing a defect usually means adding an
    # over-correction control beside the inverted witness ("a real verdict is still
    # read"), and counting those as claims would inflate the denominator every time
    # something is closed, so the published percentage would be dragged down by the
    # act of fixing. The audit recorded 59 claims; that number only changes if the
    # audit finds more.
    tests = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and not node.name.startswith("test_control_")
    ]
    declared: list[str] | None = None
    at_audit: int | None = None
    for node in tree.body:
        # BOTH forms. The last batch to close writes ``RECORDED_DEFECTS_STILL_OPEN:
        # list[str] = []``, because an empty list needs the annotation to type-check,
        # and an ast.Assign-only reader sees that as NO DECLARATION. It then refuses
        # the file outright, which is at least loud; the same blindness in a reader
        # that defaulted to [] would have published a fully-closed batch.
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if "RECORDED_DEFECTS_STILL_OPEN" in names and node.value is not None:
            declared = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]  # type: ignore[attr-defined]
        if "RECORDED_CLAIMS_AT_AUDIT" in names and isinstance(node.value, ast.Constant):
            at_audit = node.value.value
    return {"tests": tests, "open": declared, "at_audit": at_audit}


def build() -> dict:
    batches = {}
    for path in sorted(TESTS.glob("test_bgl6_f*.py")):
        tag = path.stem.replace("test_bgl6_", "")
        facts = _module_facts(path)
        if facts["open"] is None:
            raise SystemExit(f"{path.name} declares no RECORDED_DEFECTS_STILL_OPEN list")
        unknown = [n for n in facts["open"] if n not in facts["tests"]]
        if unknown:
            raise SystemExit(f"{path.name} lists tests that do not exist: {unknown}")
        # THE AUDIT'S OWN COUNT, not the file's current test count. Inverting a
        # witness into a pin renames it and a fix arrives with its control, so
        # counting functions made the audit GROW as it was closed: 59 claims read as
        # 69. The declared number is a historical fact and the file states it.
        if not isinstance(facts["at_audit"], int):
            raise SystemExit(
                f"{path.name} declares no RECORDED_CLAIMS_AT_AUDIT, so the register "
                f"would count its current test functions and report the audit as "
                f"bigger every time a record is closed"
            )
        recorded = facts["at_audit"]
        still_open = len(facts["open"])
        if still_open > recorded:
            raise SystemExit(
                f"{path.name} declares {still_open} open against {recorded} recorded "
                f"at the audit, which cannot both be true"
            )
        batches[tag] = {
            "file": f"tests/{path.name}",
            "subject": SUBJECTS.get(tag, ""),
            "recorded": recorded,
            "closed": recorded - still_open,
            "open": still_open,
            "open_tests": sorted(facts["open"]),
            # The closed ones by name too, so the register says WHAT was closed and
            # not only how many. A count on its own cannot be argued with.
            "closed_tests": sorted(t for t in facts["tests"] if t not in set(facts["open"])),
        }
    recorded = sum(b["recorded"] for b in batches.values())
    closed = sum(b["closed"] for b in batches.values())
    return {
        "_README": (
            "The second-round audit of the BGL5 fix wave, one row per recorded claim. "
            "A claim is CLOSED when its defect is fixed and its test has been inverted "
            "into a pin, in one commit. OPEN means the test still records the defective "
            "value the unit produces today, with the measurement in its docstring. "
            "Generated by scripts/bgl6_register.py from the audit files themselves; do "
            "not hand-edit."
        ),
        "generated_on": _stamp_date(),
        "totals": {"recorded": recorded, "closed": closed, "open": recorded - closed},
        "batches": batches,
    }


SITE = ROOT / "docs" / "site" / "quality-and-hardening" / "index.html"
START = "<!-- BGL6:register:start -->"
END = "<!-- BGL6:register:end -->"


def site_html(reg: dict) -> str:
    """The same numbers as the markdown block, for the rendered page.

    Written by this script rather than typed, because a spec and its rendered twin
    that share no code drift, and the twin is the surface a reader opens.
    """
    t = reg["totals"]
    pct = (100.0 * t["closed"] / t["recorded"]) if t["recorded"] else float("nan")
    rows = "\n".join(
        "                            <tr><td><code>{tag}</code></td><td>{subj}</td>"
        '<td style="text-align:right">{rec}</td>'
        '<td style="text-align:right">{closed}</td>'
        '<td style="text-align:right"><strong>{open_}</strong></td></tr>'.format(
            tag=tag,
            subj=b["subject"],
            rec=b["recorded"],
            closed=b["closed"],
            open_=b["open"],
        )
        for tag, b in reg["batches"].items()
    )
    return (
        f"{START}\n"
        "                    <p>\n"
        "                        <strong>The second-round audit.</strong>\n"
        f"                        The fix wave was audited again, and that audit recorded\n"
        f"                        <strong>{t['recorded']}</strong> claims.\n"
        f"                        <strong>{t['closed']}</strong> are closed and\n"
        f"                        <strong>{t['open']}</strong> are still open "
        f"({pct:.0f}% closed, as of {reg['generated_on']}).\n"
        "                        A claim is closed when its defect is fixed and its test has been\n"
        "                        inverted into a pin, in the same commit. Open means the test still\n"
        "                        records the defective value the unit produces today, with the\n"
        "                        measurement in its docstring, so every open row is reproducible\n"
        "                        rather than suspected.\n"
        "                    </p>\n"
        '                    <table class="matrix">\n'
        "                        <thead><tr><th>Batch</th><th>Subject</th>"
        '<th style="text-align:right">Recorded</th>'
        '<th style="text-align:right">Closed</th>'
        '<th style="text-align:right">Open</th></tr></thead>\n'
        "                        <tbody>\n"
        f"{rows}\n"
        f"                            <tr><td><strong>total</strong></td><td></td>"
        f'<td style="text-align:right"><strong>{t["recorded"]}</strong></td>'
        f'<td style="text-align:right"><strong>{t["closed"]}</strong></td>'
        f'<td style="text-align:right"><strong>{t["open"]}</strong></td></tr>\n'
        "                        </tbody>\n"
        "                    </table>\n"
        "                    <p>\n"
        "                        Generated from <code>docs/bgl6-audit-register.json</code> by\n"
        "                        <code>scripts/bgl6_register.py</code>, which reads the audit files\n"
        "                        themselves: each declares which of its tests still record an open\n"
        "                        defect, and a name that is not a test in that file is refused.\n"
        "                    </p>\n"
        f"                    {END}"
    )


R_START = "<!-- BGL6:readiness:start -->"
R_END = "<!-- BGL6:readiness:end -->"

P_START = "<!-- BGL6:plan:start -->"
P_END = "<!-- BGL6:plan:end -->"


def readiness_html() -> str:
    """The status board at the top of the hardening page.

    It answers, at a glance and without reading anything: can this ship, what is
    blocking it, how much of the surface is checked, and how many defects are open.
    The page carried several hundred numbers and none of them said where we stand.

    EVERY FIGURE CARRIES A kpi- CLASS, so the page's own fetch of library-stats.json
    fills it on load and the board is right whenever someone opens it, not right on
    the day it was written. The literals here are the no-JavaScript fallback and are
    regenerated by this script, and tests/test_kpi_spans_are_live.py refuses a class
    that renders two different values, so the two cannot drift apart unnoticed.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis  # noqa: PLC0415
    import release_gate  # noqa: PLC0415

    beta = release_gate._beta_criteria()
    failed = [c for c in beta if not c["passes"]]
    led = library_kpis._ledger()
    grading = library_kpis.grading()
    reg = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t = reg.get("totals") or {}
    verdict = "BETA READY" if not failed else "NOT BETA READY"
    closed_pct = (100.0 * t.get("closed", 0) / t["recorded"]) if t.get("recorded") else 0.0

    # One segment per criterion, blocked ones first, so the SHAPE of the bar is the
    # count: a reader should not have to read a number to see three of seven.
    segs = "".join(
        '<span class="qa-seg{blocked}" tabindex="0" role="button" '
        'aria-label="{cid}: {name}. {state}. {measured}">'
        '<span class="qa-seg__tip"><b>{cid} &mdash; {state}</b>'
        "<span>{name}</span>"
        '<span class="qa-seg__why">{why}</span>'
        '<span class="qa-seg__now">Measured now: {measured}</span>'
        "</span></span>".format(
            blocked="" if c["passes"] else " qa-seg--blocked",
            cid=c["id"],
            name=c["name"],
            state="passes" if c["passes"] else "BLOCKS the beta",
            measured=c.get("measured", ""),
            why=c.get("plain", ""),
        )
        for c in sorted(beta, key=lambda c: (c["passes"], c["id"]))
    )
    blockers = "".join(
        "\n".join(
            [
                "                                <li>",
                "                                    <code>{}</code>".format(c["id"]),
                "                                    <strong>{}</strong>".format(c["name"]),
                '                                    <span class="qa-board__measured">{}</span>'.format(
                    c.get("measured", "")
                ),
                "                                </li>",
            ]
        )
        for c in failed
    )
    parts = [
        R_START,
        '                    <div class="qa-board" id="gate-status">',
        '                        <div class="qa-board__verdict">',
        '                            <span class="qa-board__eyebrow">Beta readiness</span>',
        '                            <strong class="qa-board__big kpi-beta-verdict">{}</strong>'.format(
            verdict
        ),
        '                            <span class="qa-board__sub">',
        '                                <span class="kpi-beta-fails">{}</span> of'.format(
            len(failed)
        ),
        '                                <span class="kpi-beta-count">{}</span> criteria block it.'.format(
            len(beta)
        ),
        '                                Measured <span class="kpi-gate-date">{}</span>.'.format(
            _stamp_date()
        ),
        "                            </span>",
        '                            <div class="qa-segbar" role="group" aria-label="{} of {} beta criteria block the release.">{}</div>'.format(
            len(failed), len(beta), segs
        ),
        '                            <span class="qa-segbar__hint">One block per criterion, blocked ones first. Point at a block, or tab to it, for what it means and where it stands.</span>',
        "                        </div>",
        '                        <div class="qa-board__tiles">',
        '                            <div class="qa-tile">',
        '                                <span class="qa-tile__num kpi-ledger-checked">{:,}</span>'.format(
            led["checked"]
        ),
        '                                <span class="qa-tile__label">Checked</span>',
        '                                <span class="qa-tile__note">executed and recorded</span>',
        "                            </div>",
        '                            <div class="qa-tile qa-tile--warn">',
        '                                <span class="qa-tile__num kpi-ledger-fix-pending">{:,}</span>'.format(
            led["fix_pending"]
        ),
        '                                <span class="qa-tile__label">Fix pending</span>',
        '                                <span class="qa-tile__note">known open defect</span>',
        "                            </div>",
        '                            <div class="qa-tile qa-tile--warn">',
        '                                <span class="qa-tile__num kpi-ledger-not-checked">{:,}</span>'.format(
            led["not_checked"]
        ),
        '                                <span class="qa-tile__label">Not checked</span>',
        '                                <span class="qa-tile__note">nobody has examined</span>',
        "                            </div>",
        '                            <div class="qa-tile">',
        '                                <span class="qa-tile__num kpi-ledger-total">{:,}</span>'.format(
            led["total"]
        ),
        '                                <span class="qa-tile__label">Units in scope</span>',
        '                                <span class="qa-tile__note">the public surface</span>',
        "                            </div>",
        "                        </div>",
        '                        <div class="qa-board__bars">',
        '                            <div class="qa-bar">',
        '                                <span class="qa-bar__label">Second-round audit',
        '                                    <b><span class="kpi-second-closed">{:,}</span> of <span class="kpi-second-recorded">{:,}</span> closed</b>'.format(
            t.get("closed", 0), t.get("recorded", 0)
        ),
        "                                </span>",
        '                                <span class="qa-bar__track"><i class="qa-bar__fill--second" style="width:{:.1f}%"></i></span>'.format(
            closed_pct
        ),
        '                                <span class="qa-bar__right"><span class="kpi-second-open">{:,}</span> open, each reproducible today</span>'.format(
            t.get("open", 0)
        ),
        "                            </div>",
        '                            <div class="qa-bar">',
        '                                <span class="qa-bar__label">Grades independently checked',
        '                                    <b><span class="kpi-grade-audited-ever">{:,}</span> of <span class="kpi-grade-items">{:,}</span></b>'.format(
            grading.get("audited_ever") or 0, grading.get("items") or 0
        ),
        "                                </span>",
        '                                <span class="qa-bar__track"><i class="qa-bar__fill--audited" style="width:{:.1f}%"></i></span>'.format(
            (100.0 * (grading.get("audited_ever") or 0) / grading["items"])
            if grading.get("items")
            else 0.0
        ),
        '                                <span class="qa-bar__right"><span class="kpi-grade-unaudited">{:,}</span> never argued with</span>'.format(
            grading.get("unaudited") or 0
        ),
        "                            </div>",
        "                        </div>",
        '                        <div class="qa-board__blockers">',
        '                            <span class="qa-board__eyebrow">What is blocking it</span>',
        "                            <ul>",
        blockers,
        "                            </ul>",
        '                            <p class="qa-board__foot">The beta line was decided on 1 Oct 2026: all eight beta conditions. A second examiner for every code unit is a condition of the 1.0 release, not of the beta. Every figure on this board is read from <code>scripts/release_gate.py</code> and <code>docs/bgl6-audit-register.json</code> when the page loads.</p>',
        "                        </div>",
        "                    </div>",
        "                    " + R_END,
    ]
    return "\n".join(parts)


def plan_html() -> str:
    """The beta-criterion table further down the page, generated from the gate.

    WHY IT IS GENERATED NOW. It was typed, and it rotted in the worst direction:
    it listed five criteria when the gate had seven (B1b and B2b were added and the
    table never heard), it said B1 measured "1564 of 1564", and it told a reader
    that B2 FAILED with "64 open (37 measuring, 27 unclassified)" months after B2
    reached zero open and started passing. A stale Pass is bad; a stale Fail on a
    page whose whole subject is not overstating what is checked is worse, because
    it teaches a reader that the numbers on this page are decorative.

    The board at the top of the page was already generated. This table was the same
    facts written a second time by hand, which is the drift mechanism itself.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate  # noqa: PLC0415

    rows = "\n".join(
        "                        <tr><td><code>{cid}</code> {name}</td>"
        '<td id="plan-{low}">{measured}</td>'
        '<td class="{cls}">{state}</td></tr>'.format(
            cid=c["id"],
            low=c["id"].lower(),
            name=c["name"],
            measured=c.get("measured", ""),
            cls="qa-state--pass" if c["passes"] else "qa-state--fail",
            state="Pass" if c["passes"] else "Blocks",
        )
        for c in release_gate._beta_criteria()
    )
    return "\n".join(
        [
            P_START,
            '                    <div class="table-wrap">',
            "                    <table>",
            "                        <thead><tr><th>Beta criterion</th><th>Measured</th>"
            "<th>State</th></tr></thead>",
            '                        <tbody class="kpi-plan-body">',
            rows,
            "                        </tbody>",
            "                    </table>",
            "                    </div>",
            "                    " + P_END,
        ]
    )


def _stamp_shared(text: str) -> str:
    """Give every span of a SHARED kpi- class the one measured value, page-wide.

    WHY. The board is generated, the prose around it is written by hand, and both
    carry the same classes. A figure that appears in two places drifts the moment one
    copy is regenerated: kpi-gate-date read 2026-09-29 inside the block and
    2026-09-28 twice in the prose, and a hand-typed "35 open records" outlived the
    number it described. tests/test_kpi_spans_are_live.py refuses a class that
    renders two different values, so the drift is caught, but catching it every day
    by hand is not a fix. These classes are stamped from the SAME reading the board
    uses, so the copies cannot disagree.

    Only classes whose value this script already measures are stamped. A kpi- class
    filled from elsewhere is left alone rather than guessed at.
    """
    import library_kpis  # noqa: PLC0415

    reg = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t = reg.get("totals") or {}
    shared = {"kpi-gate-date": _stamp_date()}
    # The ledger classes, which appear in the board AND in two scope tables AND in
    # prose. kpi-core-total was rendering 1382 in one table and 1,387 in another.
    # The GATE's own counts, shared the same way. kpi-beta-fails rendered 1 inside the
    # generated board and 2 in four hand-written paragraphs, because a criterion had
    # just started passing and only the generated half heard about it.
    import release_gate  # noqa: PLC0415

    beta = release_gate._beta_criteria()
    shared["kpi-beta-fails"] = str(sum(1 for c in beta if not c["passes"]))
    shared["kpi-beta-count"] = str(len(beta))
    shared["kpi-beta-verdict"] = (
        "BETA READY" if all(c["passes"] for c in beta) else "NOT BETA READY"
    )
    # The GRADING classes, which move on every audit wave and appear in the board
    # AND in prose written on the day of an earlier wave. kpi-grade-unaudited
    # rendered 159 in the generated board and 219 three paragraphs away, because a
    # wave had just closed 60 of them and only the generated half heard.
    grd = library_kpis.grading()
    for cls, key in (
        ("kpi-grade-items", "items"),
        ("kpi-grade-audited", "audited"),
        ("kpi-grade-unaudited", "unaudited"),
        ("kpi-grade-audited-ever", "audited_ever"),
        ("kpi-grade-overturned", "audit_overturned"),
        ("kpi-grade-overturned-ever", "audit_overturned_ever"),
        ("kpi-grade-overturned-superseded", "audit_overturned_superseded"),
        ("kpi-grade-fabricating", "found_fabricating"),
        ("kpi-grade-fixed", "fixed"),
    ):
        if isinstance(grd.get(key), int):
            shared[cls] = "{:,}".format(grd[key])
    for cls, state in (
        ("kpi-grade-proven", "PROVEN"),
        ("kpi-grade-semi", "SEMI-PROVEN"),
        ("kpi-grade-nam", "NOT A MEASUREMENT"),
        ("kpi-grade-unproven", "UNPROVEN"),
        ("kpi-grade-defect", "DEFECT OPEN"),
    ):
        shared[cls] = "{:,}".format((grd.get("by_grade") or {}).get(state, 0))
    led = library_kpis._ledger()
    for cls, key in (
        ("kpi-ledger-checked", "checked"),
        ("kpi-ledger-fix-pending", "fix_pending"),
        ("kpi-ledger-not-checked", "not_checked"),
        ("kpi-ledger-total", "total"),
        ("kpi-all-total", "total"),
        ("kpi-core-checked", "core_checked"),
        ("kpi-core-fix", "core_fix_pending"),
        ("kpi-core-unchecked", "core_not_checked"),
        ("kpi-core-total", "core_total"),
        ("kpi-path-total", "path_total"),
        ("kpi-path-checked", "path_checked"),
        ("kpi-path-fix", "path_fix_pending"),
        ("kpi-path-unchecked", "path_not_checked"),
    ):
        if isinstance(led.get(key), int):
            shared[cls] = "{:,}".format(led[key])
    for cls, key in (
        ("kpi-second-open", "open"),
        ("kpi-second-closed", "closed"),
        ("kpi-second-recorded", "recorded"),
    ):
        if key in t:
            shared[cls] = "{:,}".format(t[key])
    # ANY TAG, not just <span>. The grade table's cells are <td class="kpi-grade-*">
    # and a span-only pattern walked straight past them: kpi-grade-defect rendered
    # 0 in that table and 17 in four paragraphs around it, which is the drift this
    # function exists to prevent, surviving inside the fix for it.
    for cls, value in shared.items():
        text = re.sub(
            r'(<(\w+)[^>]*\bclass="[^"]*\b' + re.escape(cls) + r'\b[^"]*"[^>]*>)[^<]*(</\2>)',
            lambda m: m.group(1) + value + m.group(3),
            text,
        )
    return text


def render_site(reg: dict, check: bool) -> bool:
    """Fill the register block on the rendered page. True when it is current.

    A HALF-PRESENT MARKER PAIR IS AN ERROR, not a skip, the same rule
    scripts/beta_go_live_docs.py learned: deleting a start marker would otherwise
    silently disable the block while the checker reported the page current.
    """
    if not SITE.exists():
        print(f"{SITE.relative_to(ROOT)} is missing", file=sys.stderr)
        return False
    text = SITE.read_text(encoding="utf-8")
    ok = True
    for label, start, end, fresh in (
        ("register", START, END, site_html(reg)),
        ("readiness", R_START, R_END, readiness_html()),
        ("plan", P_START, P_END, plan_html()),
    ):
        if (start in text) != (end in text):
            missing = start if end in text else end
            print(
                f"BROKEN MARKERS in {SITE.relative_to(ROOT)}: {missing} is absent, so "
                f"the {label} block is neither generated nor checked",
                file=sys.stderr,
            )
            ok = False
            continue
        if start not in text:
            print(f"{SITE.relative_to(ROOT)} carries no {label} markers", file=sys.stderr)
            ok = False
            continue
        i, j = text.index(start), text.index(end) + len(end)
        if text[i:j] == fresh:
            continue
        if check:
            print(f"STALE: {SITE.relative_to(ROOT)} {label} block", file=sys.stderr)
            ok = False
            continue
        text = text[:i] + fresh + text[j:]
        SITE.write_text(text, encoding="utf-8")
        print(f"rendered the {label} block in {SITE.relative_to(ROOT)}")
    stamped = _stamp_shared(text)
    if stamped != text:
        if check:
            print(
                f"STALE: {SITE.relative_to(ROOT)} has a shared kpi- class whose "
                f"fallback disagrees with the measured value",
                file=sys.stderr,
            )
            ok = False
        else:
            SITE.write_text(stamped, encoding="utf-8")
            print(f"stamped the shared kpi- fallbacks in {SITE.relative_to(ROOT)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fresh = build()
    if args.check:
        if not OUT.exists():
            print(f"STALE: {OUT.relative_to(ROOT)} does not exist", file=sys.stderr)
            return 1
        published = json.loads(OUT.read_text(encoding="utf-8"))
        if (
            published.get("totals") != fresh["totals"]
            or published.get("batches") != fresh["batches"]
        ):
            print(
                f"STALE: {OUT.relative_to(ROOT)} disagrees with the audit files\n"
                f"  published {published.get('totals')}\n  measured  {fresh['totals']}",
                file=sys.stderr,
            )
            return 1
        if not render_site(fresh, check=True):
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date: {fresh['totals']}")
        return 0
    OUT.write_text(json.dumps(fresh, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    t = fresh["totals"]
    print(
        f"wrote {OUT.relative_to(ROOT)}: {t['recorded']} recorded, {t['closed']} closed, {t['open']} open"
    )
    for tag, b in fresh["batches"].items():
        print(f"  {tag}: {b['closed']}/{b['recorded']} closed  ({b['subject'][:56]})")
    return 0 if render_site(fresh, check=False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
