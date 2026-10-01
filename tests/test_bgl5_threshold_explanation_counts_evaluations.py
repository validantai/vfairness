"""The threshold explanation may not count a request as a satisfied constraint.

WHY. ``_explain_threshold_analysis`` read ``len(report.feasible_regions)`` and
printed it as the number of constraints SATISFIABLE, in three places. That dict
holds one entry per constraint REQUESTED, and a region of ``(None, None)`` with
``assessed=False`` is in it, so a sweep in which nothing was measurable was
reported as three satisfiable constraints.

Measured before, on 200 rows of one group where all 100 searched thresholds are
could-not-check:

    summary   "Analysed 20 threshold(s). 3 constraint(s) are satisfiable within
               the threshold range."
    severity  "info"
    value     "3 constraint(s) analysed, 3 feasible region(s)"

printed DIRECTLY ABOVE this report's own recommendation, "NOT ASSESSED:
demographic_parity, equalized_odds, equal_opportunity could not be evaluated at any
searched threshold". One surface on a page contradicting another is worse than
either being wrong alone, because a reader believes whichever agrees with what they
hoped for.

``if not feasible: sev = "medium"`` could not fire for that input either: a dict of
three unassessed regions is non-empty, so the severity branch written for exactly
this case was out of reach. It keys off the satisfiable count now.

FOUR STATES, and the pair that must never collapse is the last two: a sweep that
measured every threshold and found none feasible is a FINDING, and a sweep that
could measure nothing is not. ``FeasibleRegion`` exists to carry that difference and
this surface was throwing it away.
"""

from __future__ import annotations

import pytest

from vfairness.explainer import _explain_threshold_analysis
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalysisReport,
)

NAMES = ("demographic_parity", "equalized_odds", "equal_opportunity")


def _not_assessed():
    return FeasibleRegion(None, None, assessed=False, n_searched=100, n_not_assessed=100)


def _infeasible():
    return FeasibleRegion(None, None, assessed=True, n_searched=100)


def _feasible():
    return FeasibleRegion(0.3, 0.7, assessed=True, n_searched=100)


def _explain(regions):
    report = ThresholdAnalysisReport(
        threshold_results=[0] * 20,
        optimal_thresholds=dict.fromkeys(regions),
        feasible_regions=regions,
        recommendations=[
            "NOT ASSESSED: demographic_parity, equalized_odds, equal_opportunity "
            "could not be evaluated at any searched threshold"
        ],
    )
    return _explain_threshold_analysis(report)


def _all_text(out) -> str:
    return " ".join([out.summary, out.explanations[0].value, out.explanations[0].evaluation])


def test_three_unassessed_constraints_are_not_three_satisfiable_ones():
    out = _explain({n: _not_assessed() for n in NAMES})
    text = _all_text(out)
    assert "3 constraint(s) are satisfiable" not in text, text
    assert "0 constraint(s) are satisfiable" in out.summary, out.summary
    assert "NOT ASSESSED" in text, text
    for name in NAMES:
        assert name in text, f"{name} is not named: {text}"


def test_the_severity_branch_written_for_that_case_can_now_be_reached():
    """It was keyed on the dict being empty, and the dict is never empty here."""
    assert _explain({n: _not_assessed() for n in NAMES}).severity == "medium"


def test_measured_infeasible_does_not_read_as_unassessed():
    """The pair that must never collapse.

    Every threshold was checked and none worked: that is a FINDING about the model,
    and the recommendation to retrain is the right one. It must not borrow the
    NOT ASSESSED language, and it must not be silent either.
    """
    out = _explain({n: _infeasible() for n in NAMES})
    text = _all_text(out)
    assert "No feasible threshold region found" in text, text
    assert "NOT ASSESSED" not in text, text
    assert "could not be evaluated" not in text, text
    assert out.severity == "medium"


@pytest.mark.parametrize(
    "regions,n_satisfiable,n_unassessed",
    [
        ({n: _feasible() for n in NAMES}, 3, 0),
        (
            {
                "demographic_parity": _feasible(),
                "equalized_odds": _feasible(),
                "equal_opportunity": _not_assessed(),
            },
            2,
            1,
        ),
        (
            {
                "demographic_parity": _feasible(),
                "equalized_odds": _infeasible(),
                "equal_opportunity": _not_assessed(),
            },
            1,
            1,
        ),
    ],
)
def test_the_over_correction_control_a_real_feasible_region_is_still_counted(
    regions, n_satisfiable, n_unassessed
):
    """A fix that reported zero satisfiable always would pass every test above.

    The counts are asserted exactly, and the mixed rows matter most: a partly
    measurable sweep must report what it DID establish as well as what it did not.
    """
    out = _explain(regions)
    assert f"{n_satisfiable} constraint(s) are satisfiable" in out.summary, out.summary
    assert f"{n_satisfiable} feasible region(s)" in out.explanations[0].value
    if n_unassessed:
        assert f"{n_unassessed} could not be evaluated" in out.summary, out.summary
        assert "NOT ASSESSED" in out.explanations[0].value
    else:
        assert "NOT ASSESSED" not in _all_text(out)
    assert out.severity == ("info" if n_satisfiable else "medium")


def test_a_bare_tuple_is_not_credited_with_being_assessed():
    """A plain (lower, upper) carries no evidence about which state it is in.

    Crediting it would let any caller who builds regions by hand, including every
    test fixture written before FeasibleRegion existed, read as fully measured.
    """
    out = _explain({"demographic_parity": (0.3, 0.7)})
    assert "0 constraint(s) are satisfiable" in out.summary, out.summary
