"""House rules for text that ships inside a rendered artifact.

The SVG templates in ``vfairness/rendering/templates`` are not internal notes:
their text is drawn onto every exported chart and report a customer opens. Two
punctuation forms are banned there outright by the project's writing rules:

* the em dash (U+2014), and
* the double hyphen used as prose punctuation.

Both existed in shipped templates before this guard, most visibly on the group
comparison chart, whose disparity badge rendered a literal ``High -- Disparity:
0.310`` onto the canvas.

The double hyphen check must be precise, because a hyphen is perfectly
legitimate in markup that is never read as a sentence: ``stroke-dasharray``,
``font-size``, a CSS custom property such as ``--brand-accent``, a ``var(...)``
reference to one, an XML comment delimiter, or a negative coordinate in a path.
Those are masked out before the scan, and the masking itself is proved by the
control tests at the bottom of this file: legitimate markup must NOT trip the
detector, and prose containing a double hyphen MUST.

Scanning the templates alone is not enough, and for a long time this file did
exactly that. It reported 96 of 96 templates clean while 93 lines of em dash
still reached the canvas of 27 of the 46 gallery artifacts, because chart
titles, badge labels, findings text, recommendation lines and CLI output are
built from PYTHON string literals and never appear in a template at all. A
guard that is green while its own defect ships is the failure this suite exists
to stop, so the second half of this file walks the package's abstract syntax
tree and checks the string constants themselves.

Docstrings (module, class and function) are skipped there, and only docstrings:
they are developer-facing and never drawn onto an artifact or printed to a
terminal. Every other string constant is in scope, including the parts of an
f-string, because that is precisely where the shipped text lives.
"""

import ast
import json
import re
from pathlib import Path

import pytest

import vfairness
from vfairness.rendering.engine import TEMPLATE_DIR

EM_DASH = "—"
DOUBLE_HYPHEN = "--"

# Constructs that contain a double hyphen without ever being read as prose.
# Masked (not deleted) so reported line numbers stay true to the source file.
_XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CSS_VAR_REFERENCE = re.compile(r"var\(\s*--[A-Za-z_][A-Za-z0-9_-]*")
_CSS_CUSTOM_PROPERTY = re.compile(r"--[A-Za-z_][A-Za-z0-9_-]*(?=\s*:)")
_NON_PROSE = (_XML_COMMENT, _CSS_VAR_REFERENCE, _CSS_CUSTOM_PROPERTY)


def _blank(match: "re.Match[str]") -> str:
    """Replace a match with spaces, preserving newlines and total length."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _mask_non_prose(text: str) -> str:
    for pattern in _NON_PROSE:
        text = pattern.sub(_blank, text)
    return text


def _find(text: str, needle: str) -> list:
    """Return ``(line_number, line_text)`` for every line holding ``needle``."""
    return [(i + 1, line.strip()) for i, line in enumerate(text.splitlines()) if needle in line]


def _prose_double_hyphens(text: str) -> list:
    masked = _mask_non_prose(text)
    source_lines = text.splitlines()
    return [
        (i + 1, source_lines[i].strip())
        for i, line in enumerate(masked.splitlines())
        if DOUBLE_HYPHEN in line
    ]


def _shipped_templates() -> list:
    templates = sorted(Path(TEMPLATE_DIR).glob("*.svg"))
    assert templates, f"no shipped templates found under {TEMPLATE_DIR}"
    return templates


def _ids(paths: list) -> list:
    return [p.name for p in paths]


TEMPLATES = _shipped_templates()


@pytest.mark.parametrize("path", TEMPLATES, ids=_ids(TEMPLATES))
def test_no_em_dash_in_shipped_template(path):
    hits = _find(path.read_text(encoding="utf-8"), EM_DASH)
    assert not hits, (
        f"{path.name} contains an em dash (U+2014), which is banned in shipped "
        f"text. Use a colon, a comma or a rephrase:\n"
        + "\n".join(f"  line {n}: {line}" for n, line in hits)
    )


@pytest.mark.parametrize("path", TEMPLATES, ids=_ids(TEMPLATES))
def test_no_prose_double_hyphen_in_shipped_template(path):
    hits = _prose_double_hyphens(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.name} renders a double hyphen as prose punctuation, which is "
        f"banned in shipped text. Use a colon, a comma or a rephrase:\n"
        + "\n".join(f"  line {n}: {line}" for n, line in hits)
    )


def test_group_comparison_badge_renders_clean():
    """The known defect, checked on the artifact rather than on the source.

    This renders the real chart through the real adapter, so a double hyphen
    reintroduced anywhere on the path from adapter to canvas is caught, not
    only one typed into this one template.
    """
    from vfairness.rendering import group_comparison_to_svg

    report = {
        "task_type": "classification",
        "group_stats": {
            "male": {"positive_rate": 0.62, "size": 4100},
            "female": {"positive_rate": 0.31, "size": 3800},
            "nonbinary": {"positive_rate": 0.44, "size": 260},
        },
    }
    svg = group_comparison_to_svg(report, metric="positive_rate")

    badge = [line for line in svg.splitlines() if "disparity: 0.310" in line.lower()]
    assert badge, "the disparity badge did not render"
    assert EM_DASH not in badge[0]
    assert DOUBLE_HYPHEN not in badge[0]

    # Over-correction control: the badge still states the same band and the
    # same number it always did. Cleaning the punctuation must not cost the
    # reader a single fact.
    assert "High" in badge[0]
    assert "0.310" in badge[0]


def test_cicd_gate_report_title_renders_clean():
    """The gate report headline, checked on the artifact.

    Every ``cicd_pipeline.svg`` template check above passed while the gate
    report still painted ``CI/CD Fairness Pipeline — Gate Report`` onto the
    canvas, because that title is a Python literal in ``rendering/adapters.py``
    and never appears in the template at all. Rasterising the shipped gallery
    image is what exposed it. This renders through the real adapter so a
    reintroduction anywhere on the path from adapter dict to canvas is caught,
    on the one artifact a customer keeps as the record of a release gate.
    """
    from types import SimpleNamespace

    from vfairness.rendering import cicd_pipeline_to_svg

    validation = SimpleNamespace(passed=True, errors=[], warnings=[])
    tests = [
        SimpleNamespace(
            test_name="test_demographic_parity[gender]",
            status="passed",
            metric_name="DP",
            metric_value=0.08,
            threshold=0.10,
        )
    ]
    gate = SimpleNamespace(
        status=SimpleNamespace(value="approved"),
        approved=True,
        blocking_reasons=[],
        warnings=[],
    )
    svg = cicd_pipeline_to_svg(validation_result=validation, test_results=tests, gate_decision=gate)

    title = [line for line in svg.splitlines() if "Gate Report" in line]
    assert title, "the gate report title did not render"
    assert EM_DASH not in title[0]
    assert DOUBLE_HYPHEN not in title[0]

    # Over-correction control: the title still names both halves, so cleaning
    # the punctuation did not cost the reader the subject or the artifact type.
    assert "CI/CD Fairness Pipeline" in title[0]

    # And the ban holds for the whole canvas, not only the headline: findings,
    # badges and stage labels are built the same way, from Python literals.
    assert EM_DASH not in svg


def test_rendered_chart_keeps_its_legitimate_single_hyphens():
    """Over-correction control on the artifact.

    Single hyphens carry meaning in both markup and prose here. A sweep that
    removed them would pass every ban above and wreck the chart.
    """
    from vfairness.rendering import group_comparison_to_svg

    report = {
        "task_type": "classification",
        "group_stats": {
            "male": {"positive_rate": 0.62, "size": 4100},
            "female": {"positive_rate": 0.31, "size": 3800},
        },
    }
    svg = group_comparison_to_svg(report, metric="positive_rate")

    assert "font-size=" in svg
    assert "font-weight=" in svg
    assert "text-anchor=" in svg
    assert "stroke-width=" in svg


# ---------------------------------------------------------------------------
# Controls: the detector's precision, proved in both directions.
# ---------------------------------------------------------------------------

_LEGITIMATE = """{# a header comment #}
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 400">
<style>
  :root { --brand-accent: #0f766e; --brand-fg: #0f172a; }
  .tick { stroke-dasharray: 4 2; font-size: 10px; }
  .bar  { fill: var(--brand-accent); stroke: var( --brand-fg ); }
</style>
<!-- layout grid, not prose -->
<path d="M10 20 L-5 -3 L-12-4 Z" stroke-width="0.5"/>
<text x="36" y="96" font-size="10" text-anchor="end">Max-min gap across sub-groups</text>
</svg>
"""

_PROSE_OFFENDER = """<svg xmlns="http://www.w3.org/2000/svg">
<text x="56" y="162" font-size="11">High -- Disparity: 0.310</text>
</svg>
"""

_EM_DASH_OFFENDER = """<svg xmlns="http://www.w3.org/2000/svg">
<text x="36" y="122" font-size="10">Fairness pipeline gate — 6 tests executed</text>
</svg>
"""


def test_detector_ignores_legitimate_markup_hyphens():
    """A template full of real CSS, path and attribute hyphens must be clean."""
    assert _prose_double_hyphens(_LEGITIMATE) == []
    assert _find(_LEGITIMATE, EM_DASH) == []


def test_detector_catches_prose_double_hyphen():
    hits = _prose_double_hyphens(_PROSE_OFFENDER)
    assert [n for n, _ in hits] == [2]
    assert "High -- Disparity" in hits[0][1]


def test_detector_catches_em_dash():
    hits = _find(_EM_DASH_OFFENDER, EM_DASH)
    assert [n for n, _ in hits] == [2]


def test_masking_preserves_line_numbers():
    """The mask must not shift line numbers, or every report points elsewhere."""
    masked = _mask_non_prose(_LEGITIMATE)
    assert len(masked.splitlines()) == len(_LEGITIMATE.splitlines())
    assert len(masked) == len(_LEGITIMATE)


# ---------------------------------------------------------------------------
# The same two bans, applied to the Python string literals that render.
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(vfairness.__file__).resolve().parent

# Constructs holding a double hyphen inside a Python string without ever being
# read as a sentence. Masked, not deleted, so offsets inside the literal stay
# meaningful when a message quotes it.
_CLI_FLAG = re.compile(r"(?<![A-Za-z0-9])--[A-Za-z][A-Za-z0-9-]*")
_HYPHEN_RULE = re.compile(r"-{3,}")

# A matplotlib line-style spec is punctuation, not prose: "--" is a dashed
# line, "k--" a dashed black line. Compared against the whole literal so a
# sentence that happens to end in a dash is not waved through.
_LINESTYLE = re.compile(r"\A[bgrcmykw]?--[bgrcmykw]?\Z")


def _mask_non_prose_literal(value: str) -> str:
    for pattern in (_XML_COMMENT, _CSS_VAR_REFERENCE, _CSS_CUSTOM_PROPERTY):
        value = pattern.sub(_blank, value)
    for pattern in (_HYPHEN_RULE, _CLI_FLAG):
        value = pattern.sub(lambda m: " " * len(m.group(0)), value)
    return value


def _docstring_constants(tree: ast.AST) -> set:
    """Identity of every Constant node that serves as a docstring.

    Docstrings are the ONLY exemption. They are developer-facing and never
    rendered, whereas ordinary literals are the shipped text this guard exists
    to police, so nothing else may be skipped HERE.

    One further exemption lives in :func:`_em_dash_literals` rather than in this
    function, because it is specific to that character: a literal whose entire
    value IS the em dash is the codepoint, not prose. See that docstring.
    """
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.add(id(value))
    return found


def _renderable_literals(source: str) -> list:
    """Yield ``(line_number, text)`` for every non-docstring string constant."""
    tree = ast.parse(source)
    skip = _docstring_constants(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        out.append((node.lineno, node.value))
    return out


def _em_dash_literals(source: str) -> list:
    """Em dashes in shipped text, excluding a literal that IS the character.

    READINESS-6, 2026-09-10. A string whose entire value is the em dash is the
    CODEPOINT, not prose that uses it: it is a lookup key or a character
    reference, and there is no way to write it without the character. The rule
    this guard enforces is about text a reader sees, and a one-character key in
    a fold table is not that.

    The case that forced the distinction: `llm/scorers.py` now normalises
    typographic punctuation before matching refusal phrases, because
    "I can't" and "I can\u2019t" are different strings and one codepoint
    separated a measured 100 percent versus 0 percent refusal disparity from a
    reported gap of 0.0. That table must map the em dash to a hyphen, so it must
    contain the em dash. Making this guard green by deleting the entry would
    reintroduce the defect the entry exists to fix, which is the wrong way round.

    The exemption is deliberately as narrow as it can be: exactly one character,
    equal to the em dash. Any prose that uses one is longer than one character
    and is still caught.
    """
    return [(n, v) for n, v in _renderable_literals(source) if EM_DASH in v and v != EM_DASH]


def _double_hyphen_literals(source: str) -> list:
    hits = []
    for lineno, value in _renderable_literals(source):
        if _LINESTYLE.match(value):
            continue
        if DOUBLE_HYPHEN in _mask_non_prose_literal(value):
            hits.append((lineno, value))
    return hits


def _python_sources() -> list:
    modules = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert modules, f"no python modules found under {PACKAGE_ROOT}"
    return modules


def _report(offenders: list) -> str:
    lines = []
    for path, hits in offenders:
        rel = path.relative_to(PACKAGE_ROOT)
        for lineno, value in hits:
            excerpt = value.strip().replace("\n", " ")
            if len(excerpt) > 90:
                excerpt = excerpt[:90] + "..."
            lines.append(f"  {rel}:{lineno}: {excerpt}")
    return "\n".join(lines)


def _scan(detector) -> list:
    offenders = []
    for path in _python_sources():
        hits = detector(path.read_text(encoding="utf-8"))
        if hits:
            offenders.append((path, hits))
    return offenders


def test_no_em_dash_in_renderable_python_string():
    """No em dash in any string literal that can reach a user.

    The failure lists every file and line rather than stopping at the first,
    so the remaining work is visible in one run instead of one file per run.
    """
    offenders = _scan(_em_dash_literals)
    count = sum(len(hits) for _, hits in offenders)
    assert not offenders, (
        f"{count} em dash(es) (U+2014) in {len(offenders)} module(s) reach "
        f"rendered charts, reports or CLI output. Docstrings are exempt; these "
        f"are not. Use a colon, a comma or a rephrase:\n" + _report(offenders)
    )


def test_no_prose_double_hyphen_in_renderable_python_string():
    """No double hyphen used as prose punctuation in a renderable literal."""
    offenders = _scan(_double_hyphen_literals)
    count = sum(len(hits) for _, hits in offenders)
    assert not offenders, (
        f"{count} prose double hyphen(s) in {len(offenders)} module(s) reach "
        f"rendered charts, reports or CLI output. Use a colon, a comma or a "
        f"rephrase:\n" + _report(offenders)
    )


# ---------------------------------------------------------------------------
# Controls for the AST sweep: precision proved in both directions.
# ---------------------------------------------------------------------------

_PY_SAMPLE = '''"""Module docstring, with an em dash — which never renders."""


class Chart:
    """Class docstring — also never rendered."""

    def title(self):
        """Function docstring — likewise exempt."""
        return "Fairness Metrics — Radar Chart"


def flags():
    return ["--dry-run", "--vfairness", "https://example.com/a-b-c"]


def markup():
    return "<style>:root { --brand-accent: #0f766e; }</style><!-- grid -->"


def table():
    return "|--------|-------|\\n"


def linestyle():
    return "k--"


def badge():
    return "High -- Disparity: 0.310"
'''


def test_ast_sweep_ignores_docstrings():
    """A docstring em dash must NOT trip the guard: it never renders."""
    hits = _em_dash_literals(_PY_SAMPLE)
    assert [n for n, _ in hits] == [9], hits


def test_ast_sweep_catches_ordinary_string_em_dash():
    """The literal on line 9 is the shipped chart title, and MUST trip it."""
    hits = _em_dash_literals(_PY_SAMPLE)
    assert hits and hits[0][1] == "Fairness Metrics — Radar Chart"


def test_ast_sweep_keeps_identifier_and_url_hyphens():
    """CLI flags, URLs, CSS custom properties and table rules are not prose."""
    hits = _double_hyphen_literals(_PY_SAMPLE)
    assert [n for n, _ in hits] == [29], hits
    assert hits[0][1] == "High -- Disparity: 0.310"


def test_ast_sweep_reads_fstring_parts():
    """f-strings carry most shipped titles, so their parts must be in scope."""
    source = 'def t(m):\n    return f"Trend Analysis — {m}"\n'
    assert [n for n, _ in _em_dash_literals(source)] == [2]


def test_ast_sweep_covers_the_whole_package():
    """The sweep must actually reach the modules that carried the defect."""
    scanned = {p.relative_to(PACKAGE_ROOT).as_posix() for p in _python_sources()}
    for expected in (
        "rendering/adapters_fairness.py",
        "rendering/adapters_monitoring.py",
        "operations/reporting/compliance.py",
        "operations/cicd/precommit.py",
        "preprocessing/bias_detection/historical.py",
    ):
        assert expected in scanned, f"{expected} is not being scanned"


# ---------------------------------------------------------------------------
# The curated JSON rule packs, which are shipped text this file used to miss
# ---------------------------------------------------------------------------
#
# The AST sweep above walks PYTHON string constants. The legal admissibility
# verdicts are not built from Python strings: they are read out of 17 curated
# JSON packs under ``legal/data/rules/`` and handed to the caller verbatim by
# ``legal/admissibility.py``, which maps each rule's ``legal_basis`` onto the
# ``legalBasis`` field of the returned verdict. That is customer-visible text
# on the same footing as a chart label, and neither half of this file looked at
# it.
#
# Found 2026-09-07, in a pack that had been shipping for months:
#
#   recruitment.us-federal.json, salary_history.legal_basis:
#     "No federal ban (state bans apply — see us-ca, us-ny, us-co, etc.)"
#
# This is the same lesson the module docstring records about templates: the
# guard was green while its own defect shipped, because the defect lived one
# file format to the left of where the guard was looking.

RULES_DIR = PACKAGE_ROOT / "legal" / "data" / "rules"

# Keys whose values are prose shown to a reader. Everything else in these packs
# is an identifier, a jurisdiction code, a status enum or a citation slug, where
# a hyphen is structural (``us-ca``, ``eu-gdpr``) and must not be flagged.
_PROSE_KEYS = frozenset({"legal_basis", "note", "notes", "rationale", "description", "summary"})


def _rule_pack_paths() -> list:
    packs = sorted(RULES_DIR.glob("*.json"))
    assert packs, f"no rule packs found under {RULES_DIR}"
    return packs


def _prose_strings(node, path=()) -> list:
    """Every prose string in a rule pack, with the key path that reached it."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                if key in _PROSE_KEYS:
                    found.append((".".join(path + (key,)), value))
            else:
                found.extend(_prose_strings(value, path + (str(key),)))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_prose_strings(item, path + (str(i),)))
    return found


def _scan_rule_packs(needle_finder) -> list:
    offenders = []
    for path in _rule_pack_paths():
        data = json.loads(path.read_text(encoding="utf-8"))
        for where, value in _prose_strings(data):
            if needle_finder(value):
                offenders.append(f"  {path.name}: {where}: {value.strip()[:90]}")
    return offenders


def test_no_em_dash_in_a_shipped_rule_pack():
    offenders = _scan_rule_packs(lambda v: EM_DASH in v)
    assert not offenders, "em dash in text served to a caller as a legal verdict:\n" + "\n".join(
        offenders
    )


def test_no_prose_double_hyphen_in_a_shipped_rule_pack():
    offenders = _scan_rule_packs(lambda v: bool(_prose_double_hyphens(v)))
    assert not offenders, (
        "double hyphen used as prose punctuation in a shipped rule pack:\n" + "\n".join(offenders)
    )


def test_the_rule_pack_sweep_reaches_real_prose():
    """NON-VACUITY. A key-name typo in _PROSE_KEYS, or a shape change in the
    packs, would silently make both tests above scan nothing at all."""
    packs = _rule_pack_paths()
    assert len(packs) >= 10, f"only {len(packs)} rule packs found"
    total = sum(len(_prose_strings(json.loads(p.read_text(encoding="utf-8")))) for p in packs)
    assert total >= 50, f"only {total} prose strings reached; the extractor is broken"
    assert any(p.name == "recruitment.us-federal.json" for p in packs), (
        "the pack that carried the defect is not being scanned"
    )


def test_the_rule_pack_detector_catches_a_planted_em_dash():
    """Positive control: the detector must fire on the exact string that shipped."""
    planted = {"columns": {"salary_history": {"legal_basis": "No federal ban (a — b)"}}}
    assert [v for _, v in _prose_strings(planted) if EM_DASH in v]


def test_the_rule_pack_detector_ignores_structural_hyphens():
    """Control: jurisdiction codes and slugs must NOT trip the double-hyphen scan."""
    benign = {"columns": {"x": {"legal_basis": "see us-ca, us-ny and eu-gdpr art-9"}}}
    for _, value in _prose_strings(benign):
        assert not _prose_double_hyphens(value)
