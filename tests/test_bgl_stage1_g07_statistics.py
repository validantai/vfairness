"""Beta Go-Live stage 1, group g07 (statistics): the disparate-impact CI must
report a MEASUREMENT or refuse, never a fabricated bound.

THE DEFECT (reproduced at the public entry, 2026-09-11). ``compute_metric_with_ci``
entered its fold-debiased branch for any nonnegative statistic and then decided
whether that construction applied by GUESSING from the permutation support
(``zero_anchored = min_perm <= (max_perm - min_perm)``), ignoring the
``null_value=1.0`` that ``disparate_impact_ratio_with_ci`` explicitly passes in.
When the guess misfired the lower bound was never estimated at all: it was the
literal constant ``0.0`` assigned by the else branch of the permutation gate,
and the upper bound was allowed outside the ratio's [0, 1] support.

Measured on EXACT parity, 100 rows per group with 6 selected in each
(random_state=0), BEFORE the fix::

    point_estimate 1.0
    CI             [0.0, 1.8]
    method         'stratified_bootstrap_fold_debiased'
    warnings       none

AFTER::

    point_estimate 1.0
    CI             [0.2, 1.0]
    method         'stratified_bootstrap_percentile'

The US-hiring / lending seal gates on ``lower_bound >= 0.80``, so a model with
an identical selection rate in every group was failed by its own seal on
ordinary low-selection-rate hiring data, with nothing warning the reader. The
1.8 upper bound is the same fabrication read from the other end: the statistic
is ``min_rate / max_rate`` and cannot exceed 1.

These tests assert the THREE STATES at the public entry
(``vfairness.disparate_impact_ratio_with_ci``):

  measured         parity data -> both bounds are real quantiles inside [0, 1]
  failed           a real disparity -> upper bound below four-fifths, significant
  could-not-check  an all-reject model -> NaN bounds, gate NOT cleared

plus an over-correction control: the fold-debiasing that difference and spread
statistics depend on must still run for them.
"""

import numpy as np
import pytest

from vfairness import (
    demographic_parity_difference_with_ci,
    disparate_impact_ratio_with_ci,
)

# The statistic is min_rate / max_rate, so its support is exactly [0, 1].
DI_SUPPORT = (0.0, 1.0)
FOUR_FIFTHS = 0.80


def _two_groups(n_per: int, sel_a: int, sel_b: int):
    """Deterministic hiring-shaped data: two groups, fixed selection counts."""
    sens = np.array(["A"] * n_per + ["B"] * n_per)
    y_pred = np.concatenate(
        [
            np.array([1] * sel_a + [0] * (n_per - sel_a)),
            np.array([1] * sel_b + [0] * (n_per - sel_b)),
        ]
    )
    return y_pred.copy(), y_pred, sens


class TestTheDisparateImpactIntervalIsMeasuredNotFabricated:
    """State 1 of 3: MEASURED."""

    def test_a_low_selection_rate_parity_model_gets_an_estimated_lower_bound(self):
        """THE REPRODUCTION. 6 of 100 selected in each group is exact parity.

        The lower bound used to be the literal 0.0 that the permutation gate's
        else branch assigns, i.e. nothing was estimated there. It must now be a
        real quantile of the bootstrap distribution.
        """
        y_true, y_pred, sens = _two_groups(100, 6, 6)
        r = disparate_impact_ratio_with_ci(y_true, y_pred, sens, random_state=0)

        assert r.point_estimate == pytest.approx(1.0)
        # Not the fabricated constant, and not merely "not equal to it": a
        # bootstrap quantile of a statistic whose every resample is positive is
        # itself strictly positive.
        assert r.lower_bound > 0.0, (
            f"lower bound {r.lower_bound!r} is the constant the fold-at-zero "
            f"branch assigns, not an estimate"
        )
        assert r.method != "stratified_bootstrap_fold_debiased", (
            "a ratio whose parity value is 1 was routed through the fold-AT-ZERO construction"
        )

    def test_the_interval_stays_inside_the_statistics_own_support(self):
        """The other half of the fabrication: an upper bound of 1.8 on a
        statistic bounded by 1."""
        y_true, y_pred, sens = _two_groups(100, 6, 6)
        r = disparate_impact_ratio_with_ci(y_true, y_pred, sens, random_state=0)

        lo_support, hi_support = DI_SUPPORT
        assert lo_support <= r.lower_bound <= hi_support, r.lower_bound
        assert lo_support <= r.upper_bound <= hi_support, (
            f"upper bound {r.upper_bound!r} is outside the [0, 1] support of min_rate / max_rate"
        )
        assert r.lower_bound <= r.point_estimate <= r.upper_bound

    def test_the_choice_of_construction_does_not_depend_on_the_selection_rate(self):
        """The misfire was data dependent: the SAME two groups at a 50%
        selection rate already took the percentile path, so the seal's answer
        turned on how many people were hired rather than on how equally."""
        y_true_low, y_pred_low, sens = _two_groups(100, 6, 6)
        y_true_hi, y_pred_hi, _ = _two_groups(100, 50, 50)

        low = disparate_impact_ratio_with_ci(y_true_low, y_pred_low, sens, random_state=0)
        high = disparate_impact_ratio_with_ci(y_true_hi, y_pred_hi, sens, random_state=0)

        assert low.method == high.method == "stratified_bootstrap_percentile"

    def test_a_fair_model_with_ample_data_clears_its_own_four_fifths_gate(self):
        """The consequence the seal reads. 250 of 500 selected in each group is
        exact parity with enough data to demonstrate it; the gate is
        ``lower_bound >= 0.80``."""
        y_true, y_pred, sens = _two_groups(500, 250, 250)
        r = disparate_impact_ratio_with_ci(y_true, y_pred, sens, random_state=0)

        assert r.point_estimate == pytest.approx(1.0)
        assert r.lower_bound >= FOUR_FIFTHS, (
            f"a model with an identical selection rate in every group was "
            f"failed by the four-fifths gate: lower bound {r.lower_bound!r}"
        )
        assert r.upper_bound <= DI_SUPPORT[1]


class TestTheIntervalStillFails:
    """State 2 of 3: FAILED. The fix must not neuter detection."""

    def test_a_real_adverse_impact_is_still_rejected(self):
        """100 of 500 against 250 of 500 is a DI ratio of 0.40."""
        y_true, y_pred, sens = _two_groups(500, 250, 100)
        r = disparate_impact_ratio_with_ci(y_true, y_pred, sens, random_state=0)

        assert r.point_estimate == pytest.approx(0.40)
        assert r.upper_bound < FOUR_FIFTHS, (
            "blatant adverse impact must stay below four-fifths on the WHOLE interval"
        )
        assert not (r.lower_bound >= FOUR_FIFTHS)
        assert r.is_significant is True

    def test_a_borderline_disparity_is_not_cleared(self):
        """200 of 500 against 250 of 500 is exactly 0.80: the point estimate
        sits on the line, and the lower bound must not clear it."""
        y_true, y_pred, sens = _two_groups(500, 250, 200)
        r = disparate_impact_ratio_with_ci(y_true, y_pred, sens, random_state=0)

        assert r.point_estimate == pytest.approx(0.80)
        assert not (r.lower_bound >= FOUR_FIFTHS)


class TestTheIntervalCanStillSayItCouldNotCheck:
    """State 3 of 3: COULD-NOT-CHECK. A refusal must not become a 0.0 or a pass."""

    def test_an_all_reject_model_returns_nan_bounds_not_a_zero_lower_bound(self):
        n = 100
        sens = np.array(["A"] * n + ["B"] * n)
        y_pred = np.zeros(2 * n, dtype=int)
        r = disparate_impact_ratio_with_ci(y_pred, y_pred, sens, random_state=1)

        assert np.isnan(r.point_estimate)
        assert np.isnan(r.lower_bound) and np.isnan(r.upper_bound), (
            "0/0 must be a could-not-check, never a bound"
        )
        # Neither of the two ways a could-not-check gets collapsed into an answer.
        assert not (r.lower_bound >= FOUR_FIFTHS), "the four-fifths seal cleared on 0/0"
        assert r.lower_bound != 0.0
        assert r.is_significant is False


class TestTheZeroNullStatisticsKeepTheirFoldDebiasing:
    """Over-correction control. The guard is ``null_value == 0.0``; the
    construction it gates exists for the spread statistics, and they must keep
    it. If this goes red the fix has disabled fold-debiasing for everybody."""

    def test_demographic_parity_difference_still_gets_the_fold_debiased_interval(self):
        rng = np.random.default_rng(3)
        n = 300
        y_pred = np.concatenate([rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n)])
        y_true = np.zeros_like(y_pred)
        sens = np.array(["A"] * n + ["B"] * n)

        r = demographic_parity_difference_with_ci(
            y_true, y_pred, sens, n_bootstrap=300, random_state=1
        )

        assert r.null_value == 0.0
        assert r.method == "stratified_bootstrap_fold_debiased", (
            "the zero-null guard switched off the construction the spread statistics depend on"
        )
        assert r.metadata.get("nonnegative_statistic") is True
        assert r.metadata.get("permutation_p_value") is not None
        assert r.is_significant is False
