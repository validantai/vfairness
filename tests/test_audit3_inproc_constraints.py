"""Third-iteration audit (2026-08-22) pins for the fairness constraints.

HIGH: post_processing ``compute_constraint_violation`` measured each group's
deviation from a single reference group (the majority) instead of the max-min
spread across all groups, so for 3+ groups it understated the disparity and
returned a false-feasible verdict when the worst pair did not involve the
reference group. Fixed to max-min (worst pair); reference-group comparison is
kept only as an explicit opt-in.

MEDIUM: in_processing EqualizedOdds / EqualOpportunity / FPR-parity constraints
substituted a hard-coded 0.5 for a group with an empty positive/negative
conditioning set, inventing up to 0.5 of disparity (false FAIL / made-up
number). Fixed to NaN (unmeasurable), with the verdict routed to
could-not-check (fail-closed).

AMENDED by the publish-readiness audit (B4, 2026-08-28). That fix stopped the
SUBSTITUTION but kept the group EXCLUDED from the spread, and one test in this
file pinned the exclusion as correct. It is not: with three or more groups,
dropping the unmeasurable one and reporting the survivors' spread is a LOWER
BOUND presented as a measurement. ``_defined_spread`` now returns NaN when ANY
group's rate is undefined, and
``test_empty_group_is_not_excluded_it_makes_the_spread_unmeasurable`` below was
rewritten from the test that froze the old behaviour. See
``tests/test_equalized_odds_unmeasurable_arm.py`` for the full B4 record.
"""

import math

import numpy as np
import pytest

from vfairness.in_processing.constraints.base import (
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FalsePositiveRateParityConstraint,
)
from vfairness.post_processing.threshold_optimization.constraints import (
    compute_constraint_violation,
)

# ---------------------------------------------------------------------------
# HIGH: multi-group gate must use max-min (worst pair), not a reference group
# ---------------------------------------------------------------------------


def _three_group_pred():
    """3 groups, positive rates 0.30 / 0.50 / 0.70, with the middle-rate group
    (B) the MAJORITY -> the old default reference. The worst pair (A vs C,
    spread 0.40) does NOT involve the reference."""
    y_pred = np.concatenate(
        [
            np.array([1] * 12 + [0] * 28),  # A: rate 0.30
            np.array([1] * 50 + [0] * 50),  # B: rate 0.50 (majority)
            np.array([1] * 28 + [0] * 12),  # C: rate 0.70
        ]
    )
    groups = np.array(["A"] * 40 + ["B"] * 100 + ["C"] * 40)
    return y_pred, groups


class TestMaxMinNotReferenceGroup:
    def test_demographic_parity_reports_max_min_not_reference(self):
        # NEGATIVE CASE: the reference-group form reported violation=0.20 and
        # is_satisfied=True (false PASS); the true max-min spread is 0.40.
        y_pred, groups = _three_group_pred()
        y_true = np.zeros_like(y_pred)
        v = compute_constraint_violation(
            y_true, y_pred, groups, constraint="demographic_parity", tolerance=0.25
        )
        assert v.violation == pytest.approx(0.40, abs=1e-9)
        assert v.is_satisfied is False

    def test_equal_opportunity_reports_max_min_not_reference(self):
        y_pred, groups = _three_group_pred()
        y_true = np.ones_like(y_pred)  # all positive -> TPR == positive rate
        v = compute_constraint_violation(
            y_true, y_pred, groups, constraint="equal_opportunity", tolerance=0.25
        )
        assert v.violation == pytest.approx(0.40, abs=1e-9)
        assert v.is_satisfied is False

    def test_equalized_odds_reports_max_min_not_reference(self):
        # TPR arm carries the disparity: A/B/C TPR 0.30/0.50/0.70, worst pair
        # A vs C = 0.40. FPR arm is flat (0) across groups.
        y_pred, groups = _three_group_pred()
        y_true = np.ones_like(y_pred)  # only positives -> FPR undefined? no:
        # ensure both arms defined: make half of each group negative with 0 fpr
        # Build explicit arrays instead.
        # A: 20 pos (rate .3 -> 6 pred1), 20 neg (0 pred1)
        # B: 50 pos (rate .5 -> 25 pred1), 50 neg (0 pred1)
        # C: 20 pos (rate .7 -> 14 pred1), 20 neg (0 pred1)
        y_true = np.concatenate(
            [
                np.array([1] * 20 + [0] * 20),
                np.array([1] * 50 + [0] * 50),
                np.array([1] * 20 + [0] * 20),
            ]
        )
        y_pred = np.concatenate(
            [
                np.array([1] * 6 + [0] * 14 + [0] * 20),  # A TPR .30, FPR 0
                np.array([1] * 25 + [0] * 25 + [0] * 50),  # B TPR .50, FPR 0
                np.array([1] * 14 + [0] * 6 + [0] * 20),  # C TPR .70, FPR 0
            ]
        )
        groups = np.array(["A"] * 40 + ["B"] * 100 + ["C"] * 40)
        v = compute_constraint_violation(
            y_true, y_pred, groups, constraint="equalized_odds", tolerance=0.25
        )
        assert v.violation == pytest.approx(0.40, abs=1e-9)
        assert v.is_satisfied is False

    def test_two_group_unchanged_max_min_equals_reference(self):
        # DOES-NOT-OVERCORRECT: for 2 groups max-min == reference form.
        y_pred = np.concatenate([[1] * 30 + [0] * 70, [1] * 50 + [0] * 50])
        groups = np.array(["A"] * 100 + ["B"] * 100)
        v = compute_constraint_violation(
            np.zeros_like(y_pred), y_pred, groups, "demographic_parity", tolerance=0.25
        )
        assert v.violation == pytest.approx(0.20, abs=1e-9)
        assert v.is_satisfied is True

    def test_fair_three_group_still_satisfied(self):
        # DOES-NOT-OVERCORRECT: a genuinely fair 3-group config (spread 0.04)
        # must remain satisfied.
        y_pred = np.concatenate([[1] * 48 + [0] * 52, [1] * 50 + [0] * 50, [1] * 52 + [0] * 48])
        groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        v = compute_constraint_violation(
            np.zeros_like(y_pred), y_pred, groups, "demographic_parity", tolerance=0.25
        )
        assert v.violation == pytest.approx(0.04, abs=1e-9)
        assert v.is_satisfied is True

    def test_explicit_reference_group_is_opt_in(self):
        # The reference-group comparison is still available on explicit request.
        y_pred, groups = _three_group_pred()
        v = compute_constraint_violation(
            np.zeros_like(y_pred),
            y_pred,
            groups,
            "demographic_parity",
            tolerance=0.25,
            reference_group="B",
        )
        assert v.violation == pytest.approx(0.20, abs=1e-9)  # |0.3-0.5|, |0.7-0.5|

    def test_invalid_reference_group_raises(self):
        y_pred, groups = _three_group_pred()
        with pytest.raises(ValueError, match="reference_group"):
            compute_constraint_violation(
                np.zeros_like(y_pred),
                y_pred,
                groups,
                "demographic_parity",
                reference_group="ZZ",
            )


# ---------------------------------------------------------------------------
# MEDIUM: empty groups must be NaN (unmeasurable), never a fabricated 0.5
# ---------------------------------------------------------------------------


class TestEmptyGroupNotFabricated:
    def _split_label_data(self):
        """A = all positive-labelled (no negatives -> FPR undefined),
        B = all negative-labelled (no positives -> TPR undefined)."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        s = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        return y_pred, y_true, s

    def test_equalized_odds_empty_group_is_nan_not_half(self):
        # NEGATIVE CASE: old code injected TPR/FPR=0.5 and reported
        # overall_violation=0.5 (fabricated). Now both arms are unmeasurable.
        y_pred, y_true, s = self._split_label_data()
        v = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        assert math.isnan(v.details["group_tprs"]["B"])  # B has no positives
        assert math.isnan(v.details["group_fprs"]["A"])  # A has no negatives
        # no fabricated 0.5 anywhere
        assert 0.5 not in v.details["group_tprs"].values()
        assert 0.5 not in v.details["group_fprs"].values()
        assert math.isnan(v.overall_violation)
        assert v.is_satisfied is False  # could-not-check, fail-closed
        assert v.details["insufficient_data"] is True

    def test_equal_opportunity_empty_group_is_nan_not_half(self):
        y_pred, y_true, s = self._split_label_data()
        v = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        assert math.isnan(v.details["group_tprs"]["B"])
        assert v.details["group_tprs"]["A"] == pytest.approx(0.75)
        assert math.isnan(v.overall_violation)  # only one defined -> unmeasurable
        assert v.is_satisfied is False
        assert v.details["insufficient_data"] is True

    def test_fpr_parity_empty_group_is_nan_not_half(self):
        y_pred, y_true, s = self._split_label_data()
        v = FalsePositiveRateParityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        assert math.isnan(v.details["group_fprs"]["A"])
        assert v.details["group_fprs"]["B"] == pytest.approx(0.0)
        assert math.isnan(v.overall_violation)
        assert v.is_satisfied is False

    def test_empty_group_is_not_excluded_it_makes_the_spread_unmeasurable(self):
        # THIS TEST PINNED THE DEFECT. Rewritten by the publish-readiness audit
        # (B4, 2026-08-28); it was called
        # ``test_empty_group_excluded_defined_groups_measured`` and asserted
        # ``overall_violation == 0.75`` with ``insufficient_data is False`` on
        # exactly this data, describing C as "excluded, not inflating anything".
        #
        # That IS the bug. C has no positive labels, so its TPR is undefined;
        # dropping it and reporting the A-vs-B spread answers a different
        # question ("is TPR equal across A and B") while presenting the answer
        # as the constraint's verdict on A, B and C. 0.75 is a LOWER bound: the
        # true spread is at least 0.75 and unbounded above, because C's TPR
        # could be anything.
        #
        # It was written in good faith as an over-correction control for the
        # 0.5-fabrication fix above, and in that role it froze the exclusion as
        # intended behaviour. The genuine over-correction control now lives in
        # the test below, on data where every group is measurable.
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # A TPR1, B TPR.25
        s = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)  # C: no positives
        v = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        # The per-group rates are still measured, and still reported honestly.
        assert v.details["group_tprs"]["A"] == pytest.approx(1.0)
        assert v.details["group_tprs"]["B"] == pytest.approx(0.25)
        assert math.isnan(v.details["group_tprs"]["C"])
        # The between-group SPREAD, however, is could-not-check rather than 0.75.
        assert math.isnan(v.overall_violation)
        assert v.overall_violation != pytest.approx(0.75), "0.75 is a lower bound"
        assert v.is_satisfied is False
        assert v.details["insufficient_data"] is True

    def test_three_defined_groups_are_measured_and_not_inflated(self):
        # DOES-NOT-OVERCORRECT, on three groups that are ALL measurable: the
        # real TPR spread of 0.75 is reported, no NaN appears anywhere, and the
        # fix above inflates nothing. This is the control the rewritten test
        # was meant to provide before its fixture made C unmeasurable.
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        s = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)  # C: 2 positives -> TPR 0.5
        v = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        assert v.details["group_tprs"]["A"] == pytest.approx(1.0)
        assert v.details["group_tprs"]["B"] == pytest.approx(0.25)
        assert v.details["group_tprs"]["C"] == pytest.approx(0.5)
        assert v.overall_violation == pytest.approx(0.75, abs=1e-9)
        assert v.is_satisfied is False
        assert v.details["insufficient_data"] is False

    def test_fully_defined_groups_unchanged(self):
        # DOES-NOT-OVERCORRECT: when every group has both labels, no NaN
        # appears and the real max-min spread is reported.
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        s = np.array(["A"] * 4 + ["B"] * 4)
        v = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, s)
        # A: TPR .5 FPR .5 ; B: TPR 1.0 FPR .5 -> spread max(.5, 0) = .5
        assert not math.isnan(v.overall_violation)
        assert v.overall_violation == pytest.approx(0.5, abs=1e-9)
        assert v.details["insufficient_data"] is False

    def test_signed_value_zero_for_undefined_group(self):
        # signed_constraint_value must not propagate NaN into the reduction:
        # an unmeasurable group yields a 0.0 signed value (no multiplier push).
        y_pred, y_true, s = self._split_label_data()
        eq = EqualOpportunityConstraint(tolerance=0.05)
        # B is undefined (no positives) -> 0.0
        assert eq.signed_constraint_value(y_pred, y_true, s, "B") == 0.0
        eo = EqualizedOddsConstraint(tolerance=0.05)
        # neither arm defined across two groups -> gaps collapse to 0.0
        assert eo.signed_constraint_value(y_pred, y_true, s, "A") == 0.0

    def test_signed_values_still_carry_both_signs_when_defined(self):
        # DOES-NOT-OVERCORRECT: the wave-5 signed-value contract is preserved
        # for fully-defined data.
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        s = np.array(["a"] * 4 + ["b"] * 4)
        eo = EqualizedOddsConstraint()
        assert eo.signed_constraint_value(y_pred, y_true, s, "a") == pytest.approx(0.5)
        assert eo.signed_constraint_value(y_pred, y_true, s, "b") == pytest.approx(-0.5)
