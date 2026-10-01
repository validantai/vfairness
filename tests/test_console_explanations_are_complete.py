"""The console surface whose whole job is to render an explanation dict.

`print_explanations` is the function the API reference and LIBRARY_OVERVIEW tell a
reader to call after `explain_fairness_report`, and until 2026-09-29 it printed the
`metrics` section and silently dropped the `statistical` one.

MEASURED BEFORE THE FIX, entirely through the public API and with no hand-built
dict (see `test_the_statistical_section_reaches_the_console`):

    rep = FairnessAnalyzer(y, y, np.array(['a'] * 60)).get_report(include_ci=True)
    ex  = explain_fairness_report(rep)        # include_statistical defaults to True
    ex['statistical'] -> three cards, each severity 'could_not_check', each
        evaluation beginning 'COULD NOT CHECK: no interval was computed for ...
        This is neither a pass nor a fail.'
    print_explanations(ex) -> 6751 characters, and:
        any statistical key printed        -> False
        the word STATISTICAL in the output -> False

Three correctly computed, correctly worded could-not-checks, produced by default and
printed nowhere. The condensed sibling renderer in `report._print_explanations_section`
already printed its own STATISTICAL MEASURES block, so this one was the outlier of the
pair; the sibling's own two neutral-value bugs (a present-but-None severity crashing
the printout, a could_not_check card drawn with the `i` of info) are pinned here too,
because both renderers read the identical card dict.

Every refusal test is paired with a control on a measurable report, because a printer
that prints "could not check" over everything passes every test in this file.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.explainer import (
    explain_fairness_report,
    print_explanations,
)
from vfairness.evaluation.vfairness_metrics.report import _print_explanations_section


def _explanations(*, single_group: bool):
    """A real report through the real producer: no hand-built card anywhere."""
    rng = np.random.default_rng(5)
    n = 120
    y = (rng.random(n) < 0.5).astype(int)
    pred = y.copy()
    groups = np.array(["a"] * n) if single_group else np.array(["a"] * 60 + ["b"] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = FairnessAnalyzer(y, pred, groups).get_report(include_ci=True)
        return explain_fairness_report(report)


def _printed(explanations, printer=print_explanations) -> tuple[str, list[str]]:
    buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buffer):
            printer(explanations)
    return buffer.getvalue(), [str(w.message) for w in caught]


def test_the_statistical_section_reaches_the_console():
    """THE DEFECT. The section existed, held three could-not-check cards, and the
    printer never so much as named it."""
    explanations = _explanations(single_group=True)
    assert explanations["statistical"], "no statistical card at all: this would pass vacuously"
    severities = {card.get("severity") for card in explanations["statistical"].values()}
    assert "could_not_check" in severities, f"the producer changed: {severities}"

    out, _msgs = _printed(explanations)
    assert "STATISTICAL" in out.upper(), "the statistical section is not even named"
    for key in explanations["statistical"]:
        assert key in out, f"the card {key} was computed and never printed"
    assert "COULD NOT CHECK: no interval was computed" in out, (
        "the could-not-check sentence the producer wrote is not on the console"
    )


def test_an_empty_section_says_so_rather_than_printing_nothing():
    """Silence reads as "nothing was left out". include_statistical=False is a
    first-class parameter of the producer, so this is a supported path."""
    rng = np.random.default_rng(5)
    y = (rng.random(120) < 0.5).astype(int)
    groups = np.array(["a"] * 60 + ["b"] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = FairnessAnalyzer(y, y, groups).get_report(include_ci=True)
        explanations = explain_fairness_report(report, include_statistical=False)
    assert explanations["statistical"] == {}, "the producer did not suppress the section"

    out, _msgs = _printed(explanations)
    assert "STATISTICAL" in out.upper(), "a suppressed section vanished without a trace"
    assert "include_statistical=False" in out, (
        "the empty section does not say why it is empty, so it reads as nothing missing"
    )


def test_a_section_this_printer_cannot_render_is_named_not_skipped():
    """The recurrence guard: the statistical section was lost exactly by being a key
    nobody printed, so an unknown key is named on the console AND warned about."""
    explanations = _explanations(single_group=False)
    explanations["intersectional"] = {"a_x_b": {"severity": "high"}}
    out, msgs = _printed(explanations)
    assert "NOT RENDERED" in out.upper()
    assert "intersectional" in out
    assert any("not rendered" in m for m in msgs), f"nothing warned: {msgs}"


def test_a_card_whose_severity_key_holds_none_does_not_kill_the_printout():
    """`.get('severity', 'N/A')` does NOT fire when the key is PRESENT holding None,
    so `.upper()` raised AttributeError and took every later card with it. An absent
    severity is also not 'info'."""
    explanations = {
        "metrics": {
            "demographic_parity_difference": {"severity": None, "value": 0.4},
            "equalized_odds_difference": {"severity": "high", "value": 0.9},
        },
        "statistical": {},
    }
    out, _msgs = _printed(explanations)
    assert "NOT STATED" in out.upper(), "an absent severity was printed as something else"
    assert "EQUALIZED ODDS DIFFERENCE" in out, "the card after the None severity was lost"


def test_control_a_measurable_report_prints_its_real_intervals():
    """OVER-CORRECTION CONTROL. Both groups carry both labels and group b is never
    selected, so every interval IS computable: the console must print the numbers and
    must not call any interval unmeasured.

    Recomputed here rather than copied from the output: group a is selected 30/60 and
    group b 0/60, so the demographic parity difference is 0.5 exactly.
    """
    y = np.array([1] * 30 + [0] * 30 + [1] * 30 + [0] * 30)
    pred = np.array([1] * 30 + [0] * 30 + [0] * 30 + [0] * 30)
    groups = np.array(["a"] * 60 + ["b"] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = FairnessAnalyzer(y, pred, groups).get_report(include_ci=True)
        explanations = explain_fairness_report(report)

    rate_a = pred[groups == "a"].mean()
    rate_b = pred[groups == "b"].mean()
    dp = report["metrics"]["demographic_parity_difference"]
    assert dp == pytest.approx(abs(rate_a - rate_b)) == pytest.approx(0.5), dp
    assert {card["severity"] for card in explanations["statistical"].values()} != {
        "could_not_check"
    }, "the fixture stopped being measurable, so this control proves nothing"

    out, _msgs = _printed(explanations)
    assert "STATISTICAL" in out.upper()
    assert "'point_estimate': 0.5" in out, (
        f"the measured interval around 0.5 is not on the console: {out[-1200:]}"
    )
    assert "no interval was computed" not in out, (
        "a fully measured report was described as an unmeasured one"
    )
    assert "include_statistical=False" not in out, "a populated section printed as empty"


def test_the_sibling_renderer_does_not_draw_a_could_not_check_as_info():
    """report._print_explanations_section reads the SAME card dict. It mapped
    could_not_check through to the "i" of info, which is the icon a graded, benign
    value gets, and it crashed on a present-but-None severity in the same line."""
    out, _msgs = _printed(
        {
            "metrics": {"dp": {"severity": None, "value": 0.4}},
            "statistical": {"dp_ci": {"severity": "could_not_check"}},
        },
        printer=_print_explanations_section,
    )
    assert "[?]" in out, "an unstated severity was drawn with a graded icon"
    assert "[i]" not in out, "a could-not-check card borrowed the info icon"
    assert "dp_ci" in out, "a statistical card with no evaluation text was skipped entirely"
    assert "COULD_NOT_CHECK" in out.upper()


def test_control_the_sibling_renderer_keeps_a_graded_cards_icon():
    """The over-correction half: a real severity must keep its own mark."""
    out, _msgs = _printed(
        {
            "metrics": {
                "dp": {"severity": "critical", "value": 0.42, "evaluation": "a real finding"}
            },
            "statistical": {},
        },
        printer=_print_explanations_section,
    )
    assert "[!!!!]" in out, f"a critical card lost its icon: {out[:200]}"
    assert "0.4200" in out
    assert "a real finding" in out
