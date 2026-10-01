"""An adapter handed nothing must render nothing, not an invented analysis.

Three reporting adapters used to answer a call carrying NO DATA with a
complete, confident, fabricated assessment, taken from a demo fixture the
caller never asked for and could not detect:

* ``report_card_to_svg(None)``      rendered DEPLOYMENT APPROVED, a 3/3 PASS
  RATE and the line "Model meets fairness requirements and is approved for
  deployment", for a model nobody had evaluated.
* ``hierarchical_gate_to_svg(None)`` rendered APPROVED with PASS on all three
  levels and 2/2, 4/4, 2/2 metric counts.
* ``reporting_dashboard_to_svg(None)`` rendered a whole OPERATIONAL analysis of
  a model called credit_risk_v3: a 72/100 health score, three KPI scores, a
  metric grid, a threshold-breach row, three alerts and five recommendations.

An SVG is an export format. It leaves the building, an auditor or a regulator
reads it, and it outlives the version of the library that produced it, so a
fabricated approval is the most damaging artifact this package can emit.

The rule these tests pin: no adapter may render a verdict, a score, a rate, a
count or a reassuring phrase that the supplied data did not establish. Missing
input is COULD NOT CHECK, stated ON THE CANVAS, never only in the accessible
``<desc>``. Three states, never two, and never a demo fixture at runtime.

The demo fixtures still exist, because the gallery needs them, but they are
reachable only through an explicit ``example=True`` and every one of them is
watermarked EXAMPLE on the canvas and in the ``<desc>``.

Two asymmetries are asserted at the bottom, because a third state that swallows
the other two is its own defect: a REAL approval must still print APPROVED, and
a REAL failure must still print BLOCKED and its breach rows.
"""

import re
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2", reason="SVG rendering needs jinja2")

from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg  # noqa: E402
from vfairness.rendering.adapters_workflow import (  # noqa: E402
    hierarchical_gate_to_svg,
    report_card_to_svg,
)

# Badge and cell labels that ARE a verdict. These are matched as WHOLE drawn
# text nodes, never as substrings of prose: the could-not-check canvas has to be
# free to write the sentence "this gate is neither approved nor blocked here",
# and a substring test would read that sentence as the verdict it denies.
VERDICT_LABELS = frozenset(
    {
        "APPROVED",
        "BLOCKED",
        "CONDITIONAL",
        "PASS",
        "FAIL",
        "WARN",
        "BREACH",
        "GREEN",
        "YELLOW",
        "RED",
    }
)

# A drawn "3/3", "72" beside "/100", or "+20%" is a measurement on the canvas.
RATE_RE = re.compile(r"^\d+\s*/\s*\d+$")

_DESC_RE = re.compile(r"<desc>(.*?)</desc>", re.S)
_TEXT_RE = re.compile(r">([^<>]+)<")


def _body(svg: str) -> str:
    """The drawn markup, with the accessible layer removed.

    A could-not-check state that is visible only to a screen reader is not
    visible. These helpers keep the two readings apart so a test cannot pass on
    the strength of the other one.
    """
    body = svg
    for tag in ("desc", "title", "metadata"):
        body = re.sub(rf"<{tag}\b.*?</{tag}>", " ", body, flags=re.S)
    return body


def canvas_nodes(svg: str) -> list:
    """Every drawn text node, whole, in document order."""
    return [t.strip() for t in _TEXT_RE.findall(_body(svg)) if t.strip()]


def canvas_text(svg: str) -> str:
    """All drawn text joined, for testing that a SENTENCE is present."""
    return " ".join(canvas_nodes(svg))


def desc_text(svg: str) -> str:
    m = _DESC_RE.search(svg)
    assert m, "every rendered chart carries an accessible <desc>"
    return m.group(1)


# The three adapters, called exactly as a caller with no data would call them.
NO_DATA_CASES = {
    "report_card": lambda: report_card_to_svg(None),
    "hierarchical_gate": lambda: hierarchical_gate_to_svg(None),
    "reporting_dashboard": lambda: reporting_dashboard_to_svg(None),
}

EXAMPLE_CASES = {
    "report_card": lambda: report_card_to_svg(example=True),
    "hierarchical_gate": lambda: hierarchical_gate_to_svg(example=True),
    "reporting_dashboard": lambda: reporting_dashboard_to_svg(example=True, tier="operational"),
}


# No data in, no analysis out


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_renders_no_verdict_on_the_canvas(name):
    """The headline a reader sees may not be a verdict the data never gave."""
    found = sorted({n.upper() for n in canvas_nodes(NO_DATA_CASES[name]())} & VERDICT_LABELS)
    assert found == [], f"{name} rendered the verdict label(s) {found} from no input at all"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_renders_no_measurement_on_the_canvas(name):
    """No rate, no count, no score: nothing a reader can take away as a number."""
    rates = [n for n in canvas_nodes(NO_DATA_CASES[name]()) if RATE_RE.match(n)]
    assert rates == [], f"{name} drew the rate(s) {rates} over data it never had"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_says_so_on_the_canvas(name):
    """Could-not-check must be legible to a sighted reader, not only in <desc>."""
    drawn = canvas_text(NO_DATA_CASES[name]())
    assert "NOT CHECKED" in drawn.upper(), f"{name} carries no not-checked badge"
    assert "Nothing was assessed" in drawn, f"{name} never states that nothing was assessed"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_says_so_in_the_accessible_description(name):
    """And the screen-reader sentence must tell the same story as the canvas."""
    desc = desc_text(NO_DATA_CASES[name]())
    # The marker must LEAD the finding, not appear somewhere inside it: the
    # finding is the first thing a screen reader speaks and the first thing a
    # truncating consumer keeps.
    finding = desc.split(": ", 1)[1] if ": " in desc else desc
    assert finding.upper().startswith("COULD NOT CHECK"), (
        f"{name} <desc> does not lead with the third state: {desc}"
    )


def test_no_data_report_card_shows_no_pass_rate():
    """0/0 reads as a measurement, so the rate is withheld, not zeroed."""
    nodes = canvas_nodes(report_card_to_svg(None))
    assert "0/0" not in nodes
    assert "not measured" in nodes


def test_no_data_dashboard_shows_no_health_score():
    """A health score is the number a reader takes away; there is none to take."""
    nodes = canvas_nodes(reporting_dashboard_to_svg(None))
    assert "/100" not in nodes, "the dashboard drew a score denominator with no score"
    assert "FAIRNESS HEALTH SCORE" not in nodes
    for word in ("GREEN", "YELLOW", "RED"):
        assert word not in nodes, f"dashboard drew a {word} status badge from no report"


def test_no_data_hierarchical_gate_grades_no_level():
    svg = hierarchical_gate_to_svg(None)
    nodes = canvas_nodes(svg)
    assert not [n for n in nodes if "metrics" in n and RATE_RE.match(n.split()[0])]
    assert "0 levels checked" in canvas_text(svg)


def test_no_data_is_not_a_rejection_either():
    """Three states, never two: could-not-check must not collapse into BLOCKED."""
    for fn in (lambda: report_card_to_svg(None), lambda: hierarchical_gate_to_svg(None)):
        nodes = canvas_nodes(fn())
        assert "BLOCKED" not in nodes
        assert "NOT CHECKED" in nodes


def test_no_data_never_names_the_demo_model():
    """The fixture's model name leaking through is how the fabrication was spotted."""
    for fn in NO_DATA_CASES.values():
        assert "credit_risk_v3" not in fn()


# The demo fixtures survive only behind an explicit argument, watermarked


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_is_watermarked_on_the_canvas(name):
    nodes = canvas_nodes(EXAMPLE_CASES[name]())
    banner = [n for n in nodes if n.upper().startswith("EXAMPLE ONLY")]
    assert banner, f"{name} demo carries no EXAMPLE band"
    assert "NOT A REAL EVALUATION" in banner[0].upper()
    assert "EXAMPLE" in nodes, f"{name} demo carries no EXAMPLE watermark"


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_is_marked_in_the_accessible_description(name):
    """The <desc> is the whole artifact for a screen-reader user."""
    desc = desc_text(EXAMPLE_CASES[name]()).upper()
    assert desc.split(": ", 1)[1].startswith("EXAMPLE ONLY"), (
        f"{name} <desc> reads as a measurement: {desc}"
    )


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_and_no_data_are_different_renders(name):
    """The whole defect was that these two were the same artifact."""
    assert EXAMPLE_CASES[name]() != NO_DATA_CASES[name]()


def test_example_must_be_asked_for_by_keyword():
    """A positional third argument must not be able to switch the demo on."""
    with pytest.raises(TypeError):
        hierarchical_gate_to_svg(None, "Fairness Gate", True)


# The asymmetries: could-not-check withdraws an all-clear, it never hides a finding


def _ev(name, value, threshold, passed, blocking=True, baseline=None):
    return SimpleNamespace(
        metric_name=name,
        value=value,
        threshold=threshold,
        passed=passed,
        is_blocking=blocking,
        baseline_value=baseline,
    )


def test_a_real_approval_is_still_rendered():
    decision = SimpleNamespace(
        approved=True,
        status=SimpleNamespace(value="approved"),
        metric_evaluations=[_ev("demographic_parity_difference", 0.042, 0.10, True)],
        blocking_reasons=[],
        warnings=[],
    )
    nodes = canvas_nodes(report_card_to_svg(decision=decision))
    assert "APPROVED" in nodes
    assert "1/1" in nodes
    assert "EXAMPLE" not in nodes, "a real evaluation must not be watermarked"


def test_a_real_failure_is_still_rendered():
    decision = SimpleNamespace(
        approved=False,
        status=SimpleNamespace(value="blocked"),
        metric_evaluations=[_ev("demographic_parity_difference", 0.240, 0.10, False)],
        blocking_reasons=["demographic_parity_difference 0.240 exceeds threshold 0.100"],
        warnings=[],
    )
    nodes = canvas_nodes(report_card_to_svg(decision=decision))
    assert "BLOCKED" in nodes
    assert "NOT CHECKED" not in nodes


def test_a_real_breach_is_still_rendered_on_the_dashboard():
    report = {
        "tier": "OPERATIONAL",
        "health_score": {"score": 60, "status": "yellow", "components": {}},
        "metrics": [
            {"name": "demographic_parity_difference", "value": 0.240, "threshold": 0.10},
        ],
        "alerts": [],
        "recommendations": [],
        "sections": [],
    }
    drawn = canvas_text(reporting_dashboard_to_svg(report))
    assert "THRESHOLD BREACH REPORT" in drawn
    assert "Nothing was assessed" not in drawn
