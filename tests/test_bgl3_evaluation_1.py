"""Does it refuse honestly when nothing can be measured? (BGL-3, batch evaluation-1)

Fourteen units in three files: the metric-direction table, the shared input
validation, and the robustness/significance statistics. The defect class is one
shape: a quantity cannot be computed and the code returns the value that reads
as its clean answer. In this area that is the strongest possible statement a
fairness library can make, because 0.0 disparity is PERFECT PARITY and p = 1.0
is "nothing to see".

Every number quoted in a docstring below was produced by running the unit on
this repo before the fix in the same change as this file. The healthy-case
assertions are not decoration: a unit that refuses everything is as wrong as one
that answers everything, and only the pair of them tells the two apart.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    improvement_amount,
    is_ratio_metric,
    metric_direction,
    relax_threshold,
)
from vfairness.evaluation.vfairness_metrics._validation import (
    check_consistent_length,
    coerce_to_array,
    validate_inputs,
)
from vfairness.evaluation.vfairness_metrics.robustness import (
    compute_robust_metrics,
    contingency_test,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
    robust_fairness_comparison,
)
from vfairness.evaluation.vfairness_metrics.robustness import (
    # Aliased: pytest would collect the public name as a test case and then ask
    # for a 'y_true' fixture.
    test_equalized_odds_chi_square as equalized_odds_chi_square,
)
from vfairness.exceptions import ConfigurationError, InvalidDataError


def _caught(fn, *args, **kwargs):
    """Run fn, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in record]


def _said(messages, *fragments):
    return any(all(f in m for f in fragments) for m in messages)


# ===========================================================================
# _metric_direction: five units, all three-state already. Pinned as CORRECT.
# ===========================================================================


class TestMetricDirection:
    def test_an_unrecognised_metric_has_no_direction_and_is_not_guessed(self):
        """UNKNOWN, never a default of lower-is-better.

        Guessing the direction is what put a green PASS on a
        disparate_impact_ratio of 0.00 (the protected group never selected).
        Executed: metric_direction('shiny_new_metric_2027') is UNKNOWN, and so is
        the empty name, None, and a name carrying BOTH a violation token and the
        ratio shape.
        """
        for name in ("", "   ", None, "shiny_new_metric_2027", 42, "some_gap_ratio"):
            assert metric_direction(name) is MetricDirection.UNKNOWN, name

        # Healthy: the table still decides the names it knows.
        assert metric_direction("disparate_impact_ratio") is MetricDirection.HIGHER_IS_BETTER
        assert metric_direction("worst_group_accuracy") is MetricDirection.HIGHER_IS_BETTER
        assert metric_direction("calibration_difference") is MetricDirection.LOWER_IS_BETTER
        assert metric_direction("disparate_impact_difference") is MetricDirection.LOWER_IS_BETTER

    def test_is_ratio_metric_answers_a_shape_question_and_cannot_certify_a_pass(self):
        """False here means "not shaped like a ratio", not "lower is better".

        The boolean is two-state by contract and cannot carry a could-not-check,
        so the pin is that the GRADING surface refuses instead: for a name where
        is_ratio_metric is False and the direction is unknown, check_threshold
        returns COULD_NOT_CHECK rather than a PASS.
        """
        assert is_ratio_metric("shiny_new_metric_2027") is False
        assert is_ratio_metric("") is False
        outcome, message = check_threshold("shiny_new_metric_2027", 0.9, 0.1)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK
        assert "no known better-direction" in message

        # Healthy: the exact ratio forms are still ratios.
        assert is_ratio_metric("disparate_impact_ratio") is True
        assert is_ratio_metric("disparate_impact") is True
        # And the cali-BRATIO-n family is still not one.
        assert is_ratio_metric("calibration_difference") is False

    @pytest.mark.parametrize(
        "name,value,threshold,fragment",
        [
            ("demographic_parity_difference", float("nan"), 0.1, "NaN"),
            ("disparate_impact_ratio", float("inf"), 0.8, "infinite"),
            ("demographic_parity_difference", None, 0.1, "no value was recorded"),
            ("demographic_parity_difference", 0.2, float("nan"), "unusable threshold"),
            ("disparate_impact_ratio", 0.0, 0.0, "no value can fall below"),
            ("demographic_parity_difference", 0.2, -0.1, "no magnitude"),
            ("shiny_new_metric_2027", 0.9, 0.1, "no known better-direction"),
        ],
    )
    def test_check_threshold_refuses_every_comparison_it_cannot_perform(
        self, name, value, threshold, fragment
    ):
        """Seven ways the comparison does not happen, and none of them is a PASS."""
        outcome, message = check_threshold(name, value, threshold)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK, (name, value, threshold)
        assert outcome is not ThresholdOutcome.PASS
        assert fragment in message

    def test_check_threshold_still_grades_what_it_can(self):
        """The over-correction control: it is not refusing everything."""
        assert check_threshold("disparate_impact_ratio", 0.95, 0.80)[0] is ThresholdOutcome.PASS
        assert check_threshold("disparate_impact_ratio", 0.62, 0.80)[0] is ThresholdOutcome.FAIL
        assert check_threshold("demographic_parity_difference", 0.05, 0.10)[0] is (
            ThresholdOutcome.PASS
        )
        assert check_threshold("demographic_parity_difference", 0.40, 0.10)[0] is (
            ThresholdOutcome.FAIL
        )
        # A zero-tolerance maximum is a real policy and must keep grading.
        assert check_threshold("demographic_parity_difference", 0.01, 0.0)[0] is (
            ThresholdOutcome.FAIL
        )

    def test_relax_threshold_does_not_move_a_bound_it_cannot_aim(self):
        """An unknown direction is returned UNCHANGED, which is the strict side.

        Executed: relax_threshold('mystery_metric', 0.10, 1.2) == 0.10, and the
        same for a NaN or non-positive multiplier. Moving it the wrong way would
        LOOSEN a real bound, and multiplying blindly is what tightened an
        intersectional four-fifths floor from 0.80 to 0.96.
        """
        assert relax_threshold("shiny_new_metric_2027", 0.10, 1.2) == 0.10
        assert relax_threshold("demographic_parity_difference", 0.10, float("nan")) == 0.10
        assert relax_threshold("demographic_parity_difference", 0.10, 0.0) == 0.10

        # Healthy: each known direction moves the permissive way.
        assert relax_threshold("demographic_parity_difference", 0.10, 1.2) == pytest.approx(0.12)
        assert relax_threshold("disparate_impact_ratio", 0.80, 1.2) == pytest.approx(0.80 / 1.2)
        assert relax_threshold("disparate_impact_ratio", 0.80, 1.2) < 0.80

    def test_improvement_amount_returns_none_rather_than_a_direction_it_guessed(self):
        """None for an unknown direction, NaN for an unmeasurable value.

        Both are distinguishable from a number by a caller. A 0.0 here would read
        as "unchanged", which is a measurement.
        """
        assert improvement_amount("shiny_new_metric_2027", 0.10, 0.30) is None
        assert math.isnan(improvement_amount("demographic_parity_difference", float("nan"), 0.30))
        assert math.isnan(improvement_amount("demographic_parity_difference", 0.10, float("nan")))

        # Healthy: a gap that shrank improved, a ratio that rose improved.
        assert improvement_amount("demographic_parity_difference", 0.10, 0.30) == pytest.approx(0.2)
        assert improvement_amount("disparate_impact_ratio", 0.90, 0.60) == pytest.approx(0.30)
        # And a degradation is negative, not absorbed.
        assert improvement_amount("disparate_impact_ratio", 0.60, 0.90) == pytest.approx(-0.30)


# ===========================================================================
# _validation: three units.
# ===========================================================================


def _three_groups_one_never_selected():
    """120 rows, three groups. Group c is NEVER selected by the model."""
    rng = np.random.default_rng(0)
    n = 40
    labelled = np.array(["a"] * n + ["b"] * n + ["c"] * n)
    missing = np.array(["a"] * n + ["b"] * n + [None] * n, dtype=object)
    y_true = rng.integers(0, 2, 3 * n)
    y_pred = np.concatenate(
        [
            (rng.random(n) < 0.55).astype(int),
            (rng.random(n) < 0.50).astype(int),
            np.zeros(n, dtype=int),
        ]
    )
    return y_true, y_pred, labelled, missing


class TestValidateInputs:
    def test_excluded_rows_are_disclosed_and_not_merely_counted(self):
        """DEFECT, fixed. The exclusion moved the verdict and said nothing.

        Measured on this repo before the fix, with the fixture below (120 rows,
        three groups, the third never selected by the model):

            group c labelled            dp_difference 0.475, dp_ratio 0.000
            group c's attribute NaN     dp_difference 0.050, dp_ratio 0.895

        The second pair is what the library reported for the SAME predictions:
        0.895 clears the four-fifths rule, so the starkest finding in the data
        left as a pass. The only trace was ``info['n_excluded'] = 40`` and no
        warning at all. info now also carries coverage, the group counts and the
        groups that lost every row.
        """
        y_true, y_pred, _, missing = _three_groups_one_never_selected()

        (_, _, _, _, info), messages = _caught(validate_inputs, y_true, y_pred, missing)

        assert info["original_size"] == 120
        assert info["final_size"] == 80
        assert info["n_excluded"] == 40
        assert info["coverage"] == pytest.approx(80 / 120)
        assert _said(messages, "40 of 120 row(s) were excluded", "66.7 percent")
        assert _said(messages, "rarely missing at random")

    def test_a_group_that_loses_every_row_is_named(self):
        """Every label NaN: 100 rows in, 0 out, both groups gone.

        Measured before the fix: info reported final_size 0 with no warning, and
        validate_binary_labels passes an empty array, so the whole pipeline ran
        on nothing.
        """
        y_true = np.full(100, np.nan)
        y_pred = np.full(100, np.nan)
        attr = np.array(["a"] * 50 + ["b"] * 50)

        (y_true_out, _, _, _, info), messages = _caught(validate_inputs, y_true, y_pred, attr)

        assert len(y_true_out) == 0
        assert info["n_groups_before"] == 2
        assert info["n_groups_after"] == 0
        assert info["groups_dropped"] == ["a", "b"]
        assert _said(messages, "NOT ONE ROW SURVIVED")

    def test_complete_data_is_not_warned_about(self):
        """Over-correction control. Nothing was dropped, so nothing is said."""
        y_true, y_pred, labelled, _ = _three_groups_one_never_selected()

        (_, _, _, _, info), messages = _caught(validate_inputs, y_true, y_pred, labelled)

        assert info["n_excluded"] == 0
        assert info["coverage"] == 1.0
        assert info["n_groups_before"] == 3
        assert info["n_groups_after"] == 3
        assert info["groups_dropped"] == []
        assert not [m for m in messages if "validate_inputs" in m], messages

    def test_an_unknown_option_refuses_to_run_at_all(self):
        """CORRECT. A typo'd option is a configuration error, not a default."""
        args = (np.array([0, 1]), np.array([0, 1]), np.array(["a", "b"]))
        with pytest.raises(ConfigurationError):
            validate_inputs(*args, task_type="ranking")
        with pytest.raises(ConfigurationError):
            validate_inputs(*args, missing_strategy="raise")


class TestCheckConsistentLength:
    def test_a_mismatch_raises_instead_of_picking_a_length(self):
        """CORRECT. The refusal is an exception, which no caller can misread."""
        with pytest.raises(InvalidDataError):
            check_consistent_length(np.array([1, 2, 3]), np.array([1, 2]))

    def test_a_real_length_is_measured(self):
        """Including the genuine zero of two empty arrays.

        The ledger grades the ``if lengths else 0`` fallback 'legitimate': a
        length is infrastructure, not a verdict, and 0 there IS the value.
        """
        assert check_consistent_length(np.array([1, 2]), np.array([3, 4])) == 2
        assert check_consistent_length(np.array([]), np.array([])) == 0


class TestCoerceToArray:
    def test_an_unconvertible_input_raises_rather_than_becoming_an_empty_array(self):
        """CORRECT. No neutral substitution: a wide DataFrame is refused by name."""
        import pandas as pd

        with pytest.raises(TypeError) as excinfo:
            coerce_to_array(pd.DataFrame({"a": [1], "b": [2]}), "sensitive_attr")
        assert "2 columns" in str(excinfo.value)

    def test_real_inputs_convert(self):
        assert coerce_to_array([1, 2, 3]).tolist() == [1, 2, 3]
        assert coerce_to_array(np.array([1.5])).tolist() == [1.5]


# ===========================================================================
# robustness: six units.
# ===========================================================================


class TestRobustFairnessComparison:
    def test_one_group_is_not_perfect_parity(self):
        """DEFECT, fixed. 0.0 disparity was the answer for no comparison at all.

        Measured before the fix, 30 rows of a SINGLE group:

            {'standard_disparity': 0.0, 'robust_disparity': 0.0,
             'outlier_influence_detected': False}

        with no warning, and identical for three EMPTY arrays. That output is
        byte-identical to a real two-group comparison that found the groups'
        error distributions the same. The refusal also now carries
        disparity_change_ratio and conclusion_changed, which the old early
        return omitted entirely (KeyError on the refusal path, a number on the
        measured one).
        """
        y_true = np.arange(30.0)
        y_pred = y_true + np.linspace(0, 5, 30)
        one_group = np.array(["a"] * 30)

        result, messages = _caught(robust_fairness_comparison, y_true, y_pred, one_group)

        assert math.isnan(result["standard_disparity"])
        assert math.isnan(result["robust_disparity"])
        assert result["standard_disparity"] != 0.0
        assert result["conclusion_changed"] is None
        assert result["conclusion_changed"] is not False
        assert result["n_groups_compared"] == 1
        assert _said(messages, "no pair to compare", "PERFECT PARITY")

    def test_no_data_at_all_is_not_perfect_parity(self):
        empty = np.array([])
        result, messages = _caught(robust_fairness_comparison, empty, empty, empty)

        assert math.isnan(result["standard_disparity"])
        assert math.isnan(result["robust_disparity"])
        assert result["outlier_influence_detected"] is None
        assert result["outlier_influence_detected"] is not False
        assert result["n_groups_compared"] == 0
        assert _said(messages, "NOTHING WAS EXAMINED")

    def test_a_robust_estimate_that_could_not_differ_does_not_confirm_anything(self):
        """DEFECT, fixed. conclusion_changed=False was a structural zero.

        Two groups of five rows: at trim_proportion=0.1 the trim removes
        int(0.5) = 0 observations from each end, so each group's trimmed mean IS
        its mean and the robust disparity equals the standard one by
        construction. Measured before the fix: standard_disparity 2.0,
        robust_disparity 2.0, disparity_change_ratio 0.0,
        conclusion_changed=False, outlier_influence_detected=False, i.e. "the
        robust analysis confirms the standard one" from a comparison that could
        not have disagreed. The two disparities are real numbers and are kept.
        """
        y_true = np.zeros(10)
        y_pred = np.concatenate([np.full(5, 1.0), np.full(5, 3.0)])
        attr = np.array(["a"] * 5 + ["b"] * 5)

        result, messages = _caught(robust_fairness_comparison, y_true, y_pred, attr)

        assert result["standard_disparity"] == pytest.approx(2.0)
        assert result["robust_disparity"] == pytest.approx(2.0)
        assert math.isnan(result["disparity_change_ratio"])
        assert result["conclusion_changed"] is None
        assert result["conclusion_changed"] is not False
        assert result["outlier_influence_detected"] is None
        assert _said(messages, "could not disagree", "conclusion_changed is None")

    def test_two_measurable_groups_still_get_a_disparity(self):
        """Over-correction control, with an independently computed expectation."""
        y_true = np.zeros(100)
        y_pred = np.concatenate([np.full(50, 1.0), np.full(50, 3.0)])
        attr = np.array(["a"] * 50 + ["b"] * 50)

        result, messages = _caught(robust_fairness_comparison, y_true, y_pred, attr, metric="mae")

        # Group a's absolute errors are all 1.0, group b's all 3.0.
        assert result["standard_disparity"] == pytest.approx(2.0)
        assert result["robust_disparity"] == pytest.approx(2.0)
        assert result["conclusion_changed"] is False
        assert result["n_groups_compared"] == 2
        assert not messages, messages


class TestComputeRobustMetrics:
    def test_a_trim_that_removes_nothing_cannot_detect_an_outlier(self):
        """DEFECT, fixed. The comparison was structurally incapable of firing.

        scipy trims ``int(trim_proportion * n)`` points from each end, which is
        ZERO for every n <= 9 at the default 0.1, so the trimmed mean IS the
        mean. Measured before the fix on eight values of 1.0 and one of 1000.0:

            n=9  standard 112.0, trimmed 112.0, divergence_ratio 0.0,
                 outlier_influence_detected False
            n=3  standard 334.0, trimmed 334.0, divergence_ratio 0.0,
                 outlier_influence_detected False

        A 1000x outlier in a small subgroup is exactly what this function exists
        to find, and small subgroups are where it matters most.
        """
        for n in (3, 9):
            values = np.array([1.0] * (n - 1) + [1000.0])
            groups = np.array(["a"] * n)
            results, messages = _caught(compute_robust_metrics, values, groups)
            row = results["a"]

            assert row.trim_effective is False, n
            assert row.outlier_influence_detected is None, n
            assert row.outlier_influence_detected is not False, n
            assert math.isnan(row.divergence_ratio), n
            assert _said(messages, "could not be answered", f"n={n}", "removes 0 observations")

    def test_a_large_trim_proportion_does_not_rescue_a_group_of_three(self):
        """The corner the first version of this fix left open.

        Below n=4 the function does not call trim_mean at all, it copies the
        mean. So trim_proportion=0.4 at n=3 counts one trimmed observation while
        nothing is trimmed, and a naive count would have claimed the comparison
        could fire and then read the identical estimators as a measured clean.
        """
        results, messages = _caught(
            compute_robust_metrics, np.array([1.0, 1.0, 1000.0]), np.array(["a"] * 3), 0.4
        )
        row = results["a"]

        assert row.trim_effective is False
        assert row.trimmed_value == pytest.approx(row.standard_value)
        assert row.outlier_influence_detected is None
        assert math.isnan(row.divergence_ratio)
        assert _said(messages, "trim_proportion=0.4", "removes 0 observations")

    @pytest.mark.parametrize(
        "values,fragment",
        [
            # Mean exactly 0.0 with a trimmed mean of 1.0: total disagreement
            # between the two estimators, reported before the fix as
            # divergence_ratio 0.0 and outlier_influence_detected False.
            (np.array([1.0] * 19 + [-19.0]), "standard_value=0.0"),
            # Twenty NaNs: abs(nan) > 1e-10 is False, so the old code took the
            # same else branch and answered 0.0 / False.
            (np.full(20, np.nan), "standard_value=nan"),
        ],
    )
    def test_a_divergence_ratio_with_no_denominator_is_not_agreement(self, values, fragment):
        """DEFECT, fixed. 0.0 reads as "the robust estimate agrees"."""
        results, messages = _caught(compute_robust_metrics, values, np.array(["a"] * len(values)))
        row = results["a"]

        assert math.isnan(row.divergence_ratio)
        assert row.outlier_influence_detected is None
        assert row.outlier_influence_detected is not False
        assert _said(messages, "no usable relative divergence", fragment)

    def test_nothing_examined_is_not_nothing_found(self):
        results, messages = _caught(compute_robust_metrics, np.array([]), np.array([]))
        assert results == {}
        assert _said(messages, "NOTHING WAS EXAMINED")

    def test_a_group_the_trim_can_reach_still_gets_a_verdict(self):
        """Over-correction control. n=20 at trim 0.1 removes 2 from each end."""
        values = np.array([1.0] * 19 + [500.0])
        results, messages = _caught(compute_robust_metrics, values, np.array(["a"] * 20))
        row = results["a"]

        assert row.trim_effective is True
        assert row.standard_value == pytest.approx(25.95)
        assert row.trimmed_value == pytest.approx(1.0)
        assert row.outlier_influence_detected is True
        assert row.divergence_ratio > 0.9
        assert not messages, messages

    def test_a_clean_group_is_reported_clean_not_unmeasurable(self):
        """The other over-correction control: a measured False is still False."""
        values = np.linspace(1.0, 2.0, 20)
        results, messages = _caught(compute_robust_metrics, values, np.array(["a"] * 20))
        row = results["a"]

        assert row.outlier_influence_detected is False
        assert row.divergence_ratio < 0.2
        assert not messages, messages


class TestContingencyTest:
    @pytest.mark.parametrize("n_groups", [2, 3])
    def test_a_constant_prediction_column_tests_nothing(self, n_groups):
        """DEFECT, fixed. An empty margin leaves ONE table the margins allow.

        The old guard tested the table's SHAPE, and a 2xk table can be
        full-shaped and carry no information. Measured before the fix, 15 rows
        per group with y_pred all zeros:

            2 groups: test_used 'fisher_exact', statistic nan, p_value 1.0,
                      table [[15, 15], [0, 0]], significant_at_05 False
            3 groups: ValueError from scipy, "the internally computed table of
                      expected frequencies has a zero element"

        So the function's honesty depended on the number of groups, and on the
        two-group path it wrote a clean negative verdict for a comparison no
        data could have failed.
        """
        attr = np.array(sum(([chr(97 + i)] * 15 for i in range(n_groups)), []))
        y_pred = np.zeros(15 * n_groups, dtype=int)

        result, messages = _caught(contingency_test, y_pred, attr)

        assert result.test_used == "none"
        assert math.isnan(result.p_value)
        assert result.significant_at_05 is None
        assert result.significant_at_05 is not False
        assert _said(messages, "empty outcome row(s)", "COULD NOT CHECK")

    def test_a_design_that_could_never_reach_05_gets_no_verdict(self):
        """DEFECT, fixed. Three rows per group, perfectly separated.

        That is the MOST extreme data the shape allows, and Fisher's exact
        returns p = 0.1 for it, so 0.05 is unreachable at any data. Measured
        before the fix: p_value 0.1 with significant_at_05 False, an absence of
        resolution graded as a measured negative.
        """
        result, messages = _caught(
            contingency_test, np.array([1, 1, 1, 0, 0, 0]), np.array(["a"] * 3 + ["b"] * 3)
        )

        assert result.test_used == "fisher_exact"
        assert result.p_value == pytest.approx(0.1)
        assert result.significant_at_05 is None
        assert result.significant_at_05 is not False
        assert result.detectable_at_05 is False
        assert result.min_attainable_p_value == pytest.approx(0.1)
        assert _said(messages, "p-value floor", "could not check")

    def test_a_small_but_detectable_design_is_not_refused(self):
        """Over-correction control, and the accusation this nearly became.

        99 rows in one group and 1 in the other LOOKS powerless, and
        conditioning on both observed margins it is. Its DESIGN floor is 0.01,
        because the single row in the small group being the only selected row in
        the data is a reachable outcome. So the design is detectable and the
        verdict stands.
        """
        y_pred = np.concatenate([np.ones(50, int), np.zeros(49, int), np.array([1])])
        result, messages = _caught(contingency_test, y_pred, np.array(["a"] * 99 + ["b"]))

        assert result.test_used == "fisher_exact"
        assert result.detectable_at_05 is True
        assert result.min_attainable_p_value == pytest.approx(0.01)
        assert result.significant_at_05 is False
        assert not [m for m in messages if "floor" in m], messages

        # And a sparse 12-versus-12 with only four positives keeps its verdict.
        sparse, _ = _caught(
            contingency_test, np.array([1, 1, 1, 1] + [0] * 20), np.array(["a"] * 12 + ["b"] * 12)
        )
        assert sparse.test_used == "fisher_exact"
        assert sparse.detectable_at_05 is True
        assert sparse.significant_at_05 is False

    def test_the_significance_verdict_is_readable_by_identity(self):
        """DEFECT, fixed. np.True_ is True is False.

        The field is declared Optional[bool] and is three-state, so callers read
        it by identity. Measured before the fix on the healthy 50/50 2x2 below
        (p = 6.6e-09): the value printed as True and ``is True`` was False, so a
        caller checking identity missed a real finding while ``is None`` worked.
        """
        y_pred = np.concatenate(
            [np.ones(40, int), np.zeros(10, int), np.ones(10, int), np.zeros(40, int)]
        )
        result, _ = _caught(contingency_test, y_pred, np.array(["a"] * 50 + ["b"] * 50))

        assert result.test_used == "chi_square"
        assert result.p_value < 1e-6
        assert result.significant_at_05 is True
        assert type(result.significant_at_05) is bool

        rng = np.random.default_rng(3)
        clean, _ = _caught(contingency_test, rng.integers(0, 2, 400), np.array(["a", "b"] * 200))
        assert clean.significant_at_05 is False
        assert type(clean.significant_at_05) is bool


class TestEqualizedOddsChiSquare:
    def test_a_group_absent_from_a_stratum_is_named(self):
        """DEFECT, fixed. The TPR test dropped a whole group in silence.

        Each test is built on a STRATUM (y_true == 1 for TPR), and a group can be
        absent from a stratum while being present in the audit. Measured before
        the fix, 300 rows in three groups where group c holds no positive label:

            tpr_test: chi_square, p=0.0033, significant_at_05=True, table 2x2
            fpr_test: chi_square, p=3.3e-15, significant_at_05=True, table 2x3

        A real measurement over a and b, reported with nothing saying the third
        group was outside it. The table shape was the only trace.
        """
        y_true = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50 + [0] * 100)
        y_pred = np.array(
            [1] * 25 + [0] * 25 + [0] * 50 + [1] * 40 + [0] * 10 + [0] * 50 + [1] * 50 + [0] * 50
        )
        attr = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)

        result, messages = _caught(equalized_odds_chi_square, y_true, y_pred, attr)

        assert result["tpr_test"].groups_omitted == ("c",)
        assert result["fpr_test"].groups_omitted == ()
        assert np.array(result["tpr_test"].contingency_table).shape == (2, 2)
        assert _said(messages, "group(s) c have no row with y_true == 1", "tpr_test")
        # The measurement itself is untouched: a real finding is not withdrawn.
        assert result["tpr_test"].p_value < 0.05

    def test_no_positive_labels_anywhere_is_could_not_check(self):
        """CORRECT already, pinned: the TPR stratum is empty, so no test ran."""
        y_true = np.zeros(100, dtype=int)
        y_pred = np.array([1] * 25 + [0] * 25 + [1] * 10 + [0] * 40)
        attr = np.array(["a"] * 50 + ["b"] * 50)

        result, messages = _caught(equalized_odds_chi_square, y_true, y_pred, attr)

        assert result["tpr_test"].test_used == "none"
        assert math.isnan(result["tpr_test"].p_value)
        assert result["tpr_test"].significant_at_05 is None
        assert _said(messages, "no observations")
        # The FPR test had a real table and must still run.
        assert result["fpr_test"].test_used == "chi_square"

    def test_a_complete_audit_omits_nothing(self):
        """Over-correction control."""
        rng = np.random.default_rng(3)
        result, messages = _caught(
            equalized_odds_chi_square,
            rng.integers(0, 2, 400),
            rng.integers(0, 2, 400),
            np.array(["a", "b"] * 200),
        )
        assert result["tpr_test"].groups_omitted == ()
        assert result["fpr_test"].groups_omitted == ()
        assert result["tpr_test"].test_used == "chi_square"
        assert not [m for m in messages if "groups_omitted" in m], messages


class TestPermutationWrappers:
    def test_demographic_parity_refuses_a_single_group(self):
        """CORRECT already, pinned. One group is no comparison.

        The closure returns NaN and permutation_test refuses on a non-finite
        observed statistic: p_value NaN, significance None, and a warning. A 0.0
        statistic here would have been scored against the null and come back
        p=0.0020, significant_at_05=True, i.e. a highly significant BREACH from
        a comparison that never happened.
        """
        result, messages = _caught(
            permutation_test_demographic_parity,
            np.array([1, 0, 1, 0] * 10),
            np.array(["a"] * 40),
            n_permutations=200,
            random_state=0,
        )
        assert math.isnan(result.p_value)
        assert result.significant_at_05 is None
        assert result.significant_at_01 is None
        assert "not run" in result.method
        assert _said(messages, "COULD NOT CHECK")

    def test_equal_opportunity_refuses_when_no_group_has_a_positive_label(self):
        """CORRECT already, pinned. No TPR exists, so no TPR gap exists."""
        result, messages = _caught(
            permutation_test_equal_opportunity,
            np.zeros(40, dtype=int),
            np.array([1, 0] * 20),
            np.array(["a"] * 20 + ["b"] * 20),
            n_permutations=200,
            random_state=0,
        )
        assert math.isnan(result.p_value)
        assert result.significant_at_05 is None
        assert _said(messages, "COULD NOT CHECK")

    def test_equal_opportunity_refuses_when_only_one_group_has_positives(self):
        """CORRECT already, pinned. Fewer than two TPRs is no spread."""
        y_true = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
        y_pred = np.concatenate([np.ones(10, dtype=int), np.zeros(10, dtype=int)] * 2)
        result, _ = _caught(
            permutation_test_equal_opportunity,
            y_true,
            y_pred,
            np.array(["a"] * 20 + ["b"] * 20),
            n_permutations=200,
            random_state=0,
        )
        assert math.isnan(result.p_value)
        assert result.significant_at_05 is None

    def test_both_wrappers_still_measure_a_real_separation(self):
        """Over-correction control, on a planted 0.60 gap."""
        y_pred = np.concatenate(
            [np.ones(80, int), np.zeros(20, int), np.ones(20, int), np.zeros(80, int)]
        )
        attr = np.array(["a"] * 100 + ["b"] * 100)

        dp, _ = _caught(
            permutation_test_demographic_parity,
            y_pred,
            attr,
            n_permutations=2000,
            random_state=0,
        )
        assert dp.observed_statistic == pytest.approx(0.60, abs=1e-9)
        assert dp.p_value < 0.01
        assert dp.significant_at_05 is True
        assert dp.detectable_at_05 is True

        eo, _ = _caught(
            permutation_test_equal_opportunity,
            np.ones(200, dtype=int),
            y_pred,
            attr,
            n_permutations=2000,
            random_state=0,
        )
        assert eo.observed_statistic == pytest.approx(0.60, abs=1e-9)
        assert eo.p_value < 0.01
        assert eo.significant_at_05 is True
