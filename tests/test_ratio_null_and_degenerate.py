"""A RATIO's null is 1, and 0/0 is not parity.

Two defects closed 2026-08-28, both on the statistic the US-hiring / lending
seal reads (``disparate_impact_ratio``, the four-fifths rule).

1.  **0/0 answered 1.0 (perfect parity).** When every group's selection rate is
    0 - the model selects NOBODY - ``demographic_parity_ratio`` returned 1.0,
    justified by an inline comment ("the comparison DID happen and found the
    groups equal"). ``fairlearn.demographic_parity_ratio`` and
    ``aif360.disparate_impact`` both return NaN. The consequence was concrete:
    every bootstrap resample also returned 1.0, so
    ``disparate_impact_ratio_with_ci`` produced a degenerate ``[1.0, 1.0]``
    interval and the sealed ``lower_bound >= 0.80`` gate PASSED, recording "no
    adverse impact" for a model that made no selection to test. The module's own
    CONVENTION note already forbids exactly this ("never the 'perfect' sentinel
    ... 1.0 for a ratio metric"); this branch predated it.

2.  **``is_significant`` tested the wrong null.** It read "the interval excludes
    ZERO" while being a property of the SHARED ``StatisticalResult`` and being
    serialised into every ``to_dict()``. A ratio's parity value is 1.0 and its
    values are bounded in [0, 1], so the test could only ever answer True: a
    perfectly fair model reported ``is_significant=True`` on the four-fifths
    statistic. The honest docstring ("for difference metrics") does not travel
    with a serialised field.

The DIFFERENCE family is the over-correction control throughout: its null is 0,
its 0.0 on an all-reject model is a real measurement, and neither must move.
"""

import warnings

import numpy as np
import pytest

import vfairness.evaluation.vfairness_metrics.classification as C
from vfairness.evaluation.vfairness_metrics._statistics import (
    IntervalType,
    StatisticalResult,
)

N = 40


def _two_groups(pos_a, pos_b, n=N):
    """(y_true, y_pred, sensitive) for two equal groups with the given counts."""
    y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)] * 2).astype(int)
    y_pred = np.concatenate(
        [np.ones(pos_a), np.zeros(n - pos_a), np.ones(pos_b), np.zeros(n - pos_b)]
    ).astype(int)
    sens = np.array(["A"] * n + ["B"] * n)
    return y_true, y_pred, sens


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestAnAllRejectModelHasNoRatioToReport:
    def test_demographic_parity_ratio_is_unmeasurable(self):
        ratio = C.demographic_parity_ratio(*_two_groups(0, 0))
        assert np.isnan(ratio), "0/0 must not be reported as a number"
        assert ratio != 1.0, "the perfect-parity sentinel is back"

    def test_disparate_impact_ratio_is_unmeasurable(self):
        assert np.isnan(C.disparate_impact_ratio(*_two_groups(0, 0)))

    def test_it_agrees_with_fairlearn(self):
        fm = pytest.importorskip("fairlearn.metrics")
        y_true, y_pred, sens = _two_groups(0, 0)
        theirs = fm.demographic_parity_ratio(y_true, y_pred, sensitive_features=sens)
        ours = C.demographic_parity_ratio(y_true, y_pred, sens)
        assert np.isnan(theirs) and np.isnan(ours)

    def test_the_sealed_four_fifths_gate_is_no_longer_cleared(self):
        """The whole point. ``lower_bound >= 0.80`` used to be True on a
        degenerate [1.0, 1.0] interval built out of 0/0."""
        y_true, y_pred, sens = _two_groups(0, 0)
        r = C.disparate_impact_ratio_with_ci(y_true, y_pred, sens, n_bootstrap=200, random_state=1)
        assert np.isnan(r.point_estimate)
        assert np.isnan(r.lower_bound) and np.isnan(r.upper_bound)
        assert not (r.lower_bound >= 0.80), "the four-fifths seal cleared on a 0/0 statistic"
        assert not r.is_significant

    def test_the_report_routes_it_to_not_assessable_not_to_passed(self):
        from vfairness import classification_fairness_report

        y_true, y_pred, sens = _two_groups(0, 0)
        assessment = classification_fairness_report(y_true, y_pred, sens)["assessment"]
        graded = {m["metric"] for m in assessment["passed_metrics"]}
        ungraded = {m["metric"] for m in assessment["not_assessable_metrics"]}
        assert "demographic_parity_ratio" not in graded, (
            "an all-reject model was cleared on a ratio"
        )
        assert "demographic_parity_ratio" in ungraded
        # The DIFFERENCE metric's 0.0 on the same data IS a measurement and
        # stays graded: this fix is about the quotient, not about the data.
        assert "demographic_parity_difference" in graded


class TestTheMirrorCaseIsNotDegenerateAndMustKeepWorking:
    """Over-correction control for defect 1. One group at zero against a group
    that IS selected is the WORST possible reading, and it is measured."""

    def test_one_group_never_selected_is_still_a_measured_zero(self):
        ratio = C.demographic_parity_ratio(*_two_groups(N, 0))
        assert ratio == 0.0
        assert not np.isnan(ratio)

    def test_a_healthy_ratio_is_unchanged(self):
        ratio = C.demographic_parity_ratio(*_two_groups(20, 18))
        assert ratio == pytest.approx(0.9)

    def test_the_difference_metric_still_measures_zero_on_an_all_reject_model(self):
        diff = C.demographic_parity_difference(*_two_groups(0, 0))
        assert diff == 0.0
        assert not np.isnan(diff), "a difference of two zeros IS a measured zero"


class TestIsSignificantTestsTheMetricsOwnNull:
    def test_the_default_null_is_zero_for_a_difference(self):
        r = StatisticalResult(
            point_estimate=0.0,
            lower_bound=-0.01,
            upper_bound=0.05,
            interval_type=IntervalType.CONFIDENCE,
        )
        assert r.null_value == 0.0
        assert not r.is_significant

    def test_an_interval_containing_one_is_not_significant_for_a_ratio(self):
        r = StatisticalResult(
            point_estimate=1.0,
            lower_bound=0.80,
            upper_bound=1.00,
            interval_type=IntervalType.CONFIDENCE,
            null_value=1.0,
        )
        assert not r.is_significant, "a ratio interval touching parity is not a disparity"

    def test_an_interval_excluding_one_is_significant_for_a_ratio(self):
        r = StatisticalResult(
            point_estimate=0.25,
            lower_bound=0.19,
            upper_bound=0.33,
            interval_type=IntervalType.CONFIDENCE,
            null_value=1.0,
        )
        assert r.is_significant

    def test_the_null_is_serialised_beside_the_flag(self):
        r = StatisticalResult(
            point_estimate=1.0,
            lower_bound=0.8,
            upper_bound=1.0,
            interval_type=IntervalType.CONFIDENCE,
            null_value=1.0,
        )
        d = r.to_dict()
        assert d["null_value"] == 1.0
        assert d["is_significant"] is False

    def test_a_perfectly_fair_model_is_not_significant_on_the_ratio(self):
        """The reproduction. Identical selection in both groups, so the point
        estimate is exactly 1.0 and the interval runs up to 1.0."""
        y_true, y_pred, sens = _two_groups(24, 24, n=200)
        r = C.disparate_impact_ratio_with_ci(y_true, y_pred, sens, n_bootstrap=400, random_state=1)
        assert r.null_value == 1.0
        assert r.point_estimate == pytest.approx(1.0)
        assert r.lower_bound <= 1.0 <= r.upper_bound
        assert not r.is_significant, "a perfectly fair model reported a significant disparity"
        assert r.to_dict()["is_significant"] is False

    def test_a_real_disparity_still_reports_significant_on_the_ratio(self):
        """Negative control: the flag must still FIRE when it should."""
        y_true, y_pred, sens = _two_groups(160, 40, n=200)
        r = C.disparate_impact_ratio_with_ci(y_true, y_pred, sens, n_bootstrap=400, random_state=1)
        assert r.point_estimate < 0.8
        assert r.upper_bound < 1.0
        assert r.is_significant


class TestDifferenceMetricsAreUnchanged:
    """Over-correction control for defect 2: every difference metric keeps the
    zero null and the verdict it gave before."""

    @pytest.mark.parametrize(
        "fn",
        [
            C.demographic_parity_difference_with_ci,
            C.equalized_odds_difference_with_ci,
            C.equal_opportunity_difference_with_ci,
            C.fpr_parity_difference_with_ci,
        ],
        ids=lambda f: f.__name__,
    )
    def test_a_fair_model_is_not_significant_and_the_null_is_zero(self, fn):
        y_true, y_pred, sens = _two_groups(24, 24, n=200)
        r = fn(y_true, y_pred, sens, n_bootstrap=400, random_state=1)
        assert r.null_value == 0.0
        assert not r.is_significant

    def test_a_real_gap_is_still_significant(self):
        y_true, y_pred, sens = _two_groups(160, 40, n=200)
        r = C.demographic_parity_difference_with_ci(
            y_true, y_pred, sens, n_bootstrap=400, random_state=1
        )
        assert r.null_value == 0.0
        assert r.is_significant

    def test_a_nan_interval_is_still_not_significant(self):
        """The pre-existing guard, preserved: NaN comparisons are all False, so
        an unguarded exclusion test reports True for exactly the intervals that
        carry no information."""
        for null in (0.0, 1.0):
            r = StatisticalResult(
                point_estimate=float("nan"),
                lower_bound=float("nan"),
                upper_bound=float("nan"),
                interval_type=IntervalType.CONFIDENCE,
                null_value=null,
            )
            assert not r.is_significant
