"""Every figure on the hardening page is generated, and this fails the build if one is not.

WHY. The page exists to say what has actually been measured. It carried 51 numbers
wearing a ``kpi-*`` class, which in this codebase is what a GENERATED value looks
like, and nothing filled any of them: they were literals typed into the markup and
left behind. They had drifted, and the two a reader most needs had drifted furthest.
``kpi-all-total`` rendered as 1,558 in one paragraph and 1,564 in another, so the
page contradicted itself about the size of its own subject.

That is the defect this page is about, committed by the page itself: a value that
looks measured and is not. So the rule is mechanical rather than editorial.

  1. Every ``kpi-*`` class in the markup is filled by the page's own script from
     ``data/library-stats.json``, or it is a container this file names.
  2. Every value the script reaches for exists in that file, so a renamed key fails
     here rather than silently blanking a figure for a reader.
  3. The published file is not stale with respect to the repository.

The literals left in the markup are build-time fallbacks for a failed fetch. They
are allowed to age, and they are not what a reader normally sees; what must not
happen is a figure with no generated source at all.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "site" / "quality-and-hardening" / "index.html"
STATS = ROOT / "docs" / "site" / "data" / "library-stats.json"

# Elements that carry a kpi- class but hold rows or a link rather than one value.
# Each is populated by its own branch in the page script, named here so the
# exemption is explicit rather than a hole in the rule.
CONTAINERS = {
    "kpi-grade-body",  # <tbody>; its cells carry their own kpi- classes
    "kpi-triage-body",  # <tbody>; rows rendered from hardening.remaining
    "kpi-cov-src",  # <a>; its href is set from hardening.ci.source
    "kpi-wave-body",  # <tbody>; rows rendered from hardening.waves
    "kpi-plan-body",  # <tbody>; rows rendered from gate.beta_criteria
}


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _classes_in_markup(html: str) -> set[str]:
    found: set[str] = set()
    for attr in re.findall(r'class="([^"]*kpi-[^"]*)"', html):
        found |= {c for c in attr.split() if c.startswith("kpi-")}
    return found


def _classes_filled(html: str) -> set[str]:
    return set(re.findall(r"fill\('\.(kpi-[a-z0-9-]+)'", html))


def test_every_kpi_span_is_filled_from_the_generated_file():
    html = _page()
    static = _classes_in_markup(html) - _classes_filled(html) - CONTAINERS
    assert not static, (
        "these figures carry a kpi- class but nothing fills them, so they are typed "
        f"by hand and will drift: {sorted(static)}. Either wire them in the page "
        "script or drop the kpi- class, because a hand-set number dressed as a "
        "measurement is the defect this page documents."
    )


def test_no_container_exemption_is_left_over():
    """An exemption for an element that no longer exists hides a real gap later."""
    html = _page()
    present = _classes_in_markup(html)
    stale = CONTAINERS - present
    assert not stale, f"CONTAINERS names elements that are gone: {sorted(stale)}"


# The aliases the page script binds at the top of its fill block, and what each one
# points at inside the published file. Parsed rather than assumed, so that a key
# renamed in the page is caught here instead of silently blanking a figure.
ALIASES = {
    "sf": ("surface",),
    "h": ("hardening",),
    "gr": ("hardening", "grading"),
    "ci": ("hardening", "ci"),
    "led": ("ledger",),
}


def _paths_the_page_reads(html: str) -> set[tuple]:
    """Every kpis.* path the page script dereferences.

    Reads the script rather than a list maintained by hand. The first version of
    this test parametrised over paths TYPED INTO THIS FILE, so renaming a key in the
    page left it green: it agreed with itself and never with the page. Sabotaging
    the page proved it, which is the only reason it was found.
    """
    paths: set[tuple] = set()
    for alias, prefix in ALIASES.items():
        # alias.key and alias.key.key
        for m in re.finditer(
            rf"\b{alias}\.([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?", html
        ):
            parts = [x for x in m.groups() if x]
            paths.add(prefix + tuple(parts))
        # alias['key'] and alias.key['key']
        for m in re.finditer(rf"\b{alias}(?:\.([A-Za-z_][A-Za-z0-9_]*))?\['([^']+)'\]", html):
            parts = [x for x in m.groups() if x]
            paths.add(prefix + tuple(parts))
    return paths


def test_the_page_reads_values_through_every_alias():
    """Anti-vacuity, and it has to be PER ALIAS.

    A single total was not enough. Renaming one alias in the page (``sf`` to
    ``zz``) removed twelve surface figures from the parse and the total still
    cleared a floor of 25, so the guard went blind to a whole section and stayed
    green. Found by sabotage, not by reading it.
    """
    html = _page()
    found = _paths_the_page_reads(html)
    empty = [
        a
        for a, prefix in ALIASES.items()
        if not any(path[: len(prefix)] == prefix for path in found)
    ]
    assert not empty, (
        f"these aliases yielded no value paths: {sorted(empty)}. Either the page "
        f"renamed them, in which case ALIASES here must follow, or a whole section "
        f"stopped being filled and this guard would not have noticed."
    )
    assert len(found) >= 25, (
        f"only {len(found)} value paths were parsed out of the page script; the fill "
        f"block changed shape and this guard has gone blind"
    )


# A grade state the data does not list means nothing is in that state, which the page
# renders as a measured 0 through its `count` helper. So its absence from by_grade is
# not a missing value, and asserting otherwise would force the data to carry an empty
# bucket for every state that happens to be unused today.
COUNTED_AS_ZERO = {
    ("hardening", "grading", "by_grade", s)
    for s in ("PROVEN", "SEMI-PROVEN", "NOT A MEASUREMENT", "UNPROVEN", "DEFECT OPEN")
}


def test_every_value_the_page_reads_exists_in_the_published_file():
    """A renamed key would blank a figure for a reader with no error anywhere."""
    kpis = json.loads(STATS.read_text(encoding="utf-8"))["kpis"]
    missing = []
    for path in sorted(_paths_the_page_reads(_page()) - COUNTED_AS_ZERO):
        node = kpis
        for part in path:
            if not isinstance(node, dict) or part not in node:
                missing.append(".".join(path))
                break
            node = node[part]
    assert not missing, (
        "the page reads these and the published file does not have them, so each one "
        f"renders blank for a reader: {missing}"
    )


def test_the_grade_table_has_a_row_for_every_state_the_data_reports():
    """The grade table's cells are wired individually, so a new state would be
    measured and never rendered."""
    html = _page()
    states = json.loads(STATS.read_text(encoding="utf-8"))["kpis"]["hardening"]["grading"][
        "by_grade"
    ]
    slug = {
        "PROVEN": "kpi-grade-proven",
        "SEMI-PROVEN": "kpi-grade-semi",
        "NOT A MEASUREMENT": "kpi-grade-nam",
        "DEFECT OPEN": "kpi-grade-defect",
        "UNPROVEN": "kpi-grade-unproven",
    }
    missing = [s for s in states if s not in slug or slug[s] not in html]
    assert not missing, f"the data reports these states and the table cannot show them: {missing}"


def _producer_interpreter() -> str:
    """The interpreter ``scripts/refresh-manifest.sh`` uses, so this gate agrees with it.

    Several published figures depend on WHICH optional extras are importable, and
    the refresher prefers ``.venv/bin/python`` for exactly that reason, in a comment
    that records what happened when it did not: a run under a different interpreter
    "described an environment with the dashboard extra and without mcp, xai or
    causal, which is not the environment the wheel is tested in", and the surface
    total differed by nine units.

    This check used ``sys.executable``, which is whichever python happens to run
    pytest. Measured 2026-09-28 on this machine, right after a successful refresh:
    ``.venv/bin/python scripts/library_kpis.py --check`` said UP TO DATE and
    ``/opt/anaconda3/bin/python`` said STALE on the same file, and both were right
    about their own interpreter. The venv has mcp, shap, captum, dowhy and dash; the
    other has transformers and none of those four. So the gate's verdict depended on
    a PATH lookup, and following its advice would have republished the figures as
    the narrower environment sees them, which is the very thing the refresher
    refuses to do.

    Resolved the same way the producer resolves it, so the two cannot disagree.
    """
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def test_the_published_statistics_are_not_stale():
    """The figures are only as live as this file. Nothing regenerated it
    automatically, so the guard is here."""
    proc = subprocess.run(
        [_producer_interpreter(), str(ROOT / "scripts" / "library_kpis.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    assert proc.returncode == 0, (
        "docs/site/data/library-stats.json is stale, so every figure on the hardening "
        "page is serving an old measurement. Run ./scripts/refresh-manifest.sh.\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# One kpi class, one fallback literal.
# ─────────────────────────────────────────────────────────────────────────────
#
# The literals are build-time fallbacks for a failed fetch, and this file's
# header says they are allowed to age. Ageing is not the problem. TWO DIFFERENT
# VALUES for the same class is, because then a reader whose fetch failed sees the
# page contradict ITSELF about one measurement, and cannot tell which half to
# believe.
#
# This is the exact failure the header of this file describes as the reason the
# file exists, "1,558 in one paragraph and 1,564 in another", and on 2026-09-27
# it was still live: rendering the page with the fetch forced to fail showed four
# classes disagreeing, kpi-all-total among them. Nothing here checked it, because
# every existing test in this file reads the MARKUP and none renders the page.
#
# So the rule is about the markup and is mechanical: a class may appear as often
# as the prose needs it, and every occurrence must carry the same text.

_LITERAL = re.compile(
    r'<(?P<tag>span|td)[^>]*class="(?P<cls>[^"]*\bkpi-[^"]*)"[^>]*>(?P<val>[^<]*)</(?P=tag)>'
)


def _literals(html: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for m in _LITERAL.finditer(html):
        for cls in m.group("cls").split():
            if cls.startswith("kpi-") and cls not in CONTAINERS:
                out.setdefault(cls, set()).add(m.group("val").strip())
    return out


def test_the_check_sees_the_literals_at_all():
    """Guards the guard: an empty scrape would make the rule below vacuous."""
    found = _literals(_page())
    assert len(found) > 40, f"only {len(found)} kpi literals scraped; the regex has drifted"


def test_one_kpi_class_carries_one_fallback_value():
    clashes = {c: sorted(v) for c, v in _literals(_page()).items() if len(v) > 1}
    assert not clashes, (
        "these classes render more than one value, so when the fetch fails the page "
        "contradicts itself about its own measurements:\n"
        + "\n".join(f"  {c}: {v}" for c, v in clashes.items())
    )
