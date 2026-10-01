"""Readiness-4 pins: three exported metrics DROPPED an unmeasurable group and
reported the survivors' spread as a measurement.

THE DEFECT. ``auroc_parity``, ``fnr_parity_difference`` and
``negative_predictive_value_difference`` each excluded a group whose per-group
rate is undefined (an empty denominator: no positives, no negatives, no rows, a
single outcome class) and then computed the max-min spread over whichever groups
were left, as long as two survived. An undefined rate is could-not-check, not a
measurement, and dropping the group can only NARROW the spread, so:

* the reported disparity is a LOWER bound presented as the disparity, and
* it is monotone the wrong way, the more groups fall out the better it reads.

Measured on this repository before the fix, three groups of 40 rows each:

  auroc_parity                        A 1.00 | B 0.60 | C one class only -> 0.4000
  fnr_parity_difference               A 0.25 | B 0.75 | C no positives    -> 0.5000
  negative_predictive_value_difference A 0.75 | B 0.50 | C never predicts 0 -> 0.2500

Two consequences were measured rather than argued:

1. A FALSE PASS on the EU-healthcare validity gate. With A and B identical
   (AUROC 1.00 each) and C carrying a single outcome class, pre-fix
   ``auroc_parity`` returned 0.0, the perfect-parity sentinel, and
   ``classification_fairness_report`` filed it under ``passed_metrics`` against
   its 0.05 threshold. The gate exists to catch a model that is uninformative
   for some group, and it issued a PASS about the one group it never assessed.
   Post-fix the same data routes it to ``not_assessable_metrics``.
2. A REPORT THAT CONTRADICTED ITSELF. FNR = 1 - TPR, so
   ``fnr_parity_difference`` and ``equal_opportunity_difference`` are the same
   comparison. On three groups with C carrying no positive labels, FNR reported
   a confident 0.5000 while equal opportunity, over the identical rates,
   reported NaN. One report called the same measurement both measured and not
   assessable.

THE RULE APPLIED. Not a fourth convention: the one
``equal_opportunity_difference`` has always used and ``equalized_odds_difference``
was brought onto. Fewer than two qualifying groups is NaN, and ANY qualifying
group with an undefined rate is NaN plus a warning naming the group. Three
states, never two: measured, failed, could-not-check.

Each metric below gets a refusal pin (the drop case must NOT return a number)
and an over-correction control asserting the exact MEASURED value on fully
measurable data, checked against an independently computed reference so the
guard cannot buy its refusals by breaking real measurement.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    auroc_parity,
    equal_opportunity_difference,
    fnr_parity_difference,
    negative_predictive_value_difference,
)
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report

N = 40
# dtype=object, not the inferred "<U1": a fixed-width numpy label array silently
# TRUNCATES longer group names and would void the fixture without failing.
THREE_GROUPS = np.array(["A"] * N + ["B"] * N + ["C"] * N, dtype=object)
TWO_GROUPS = np.array(["A"] * N + ["B"] * N, dtype=object)
# 20 positives then 20 negatives, so every group clears min_group_size=30 and no
# size-drop warning fires: the only warning under test is the undefined-rate one.
POS_NEG = np.array([1] * 20 + [0] * 20)

ABS = 1e-12


def _quiet(fn, *args, **kwargs):
    """Value under test, warnings asserted separately."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _assert_refused(value, survivor_spread, name):
    """Three states, never two: could-not-check is NaN, and it is neither the
    survivors' spread nor the perfect-parity sentinel."""
    assert math.isnan(value), f"{name} returned {value!r} for a group it never assessed"
    assert value != pytest.approx(survivor_spread, abs=ABS), (
        f"{name} returned the SURVIVORS' spread {survivor_spread}, which is a lower "
        f"bound on the real gap, not a measurement of it"
    )
    assert value != 0.0, f"{name} returned the perfect-difference sentinel 0.0"


# ---------------------------------------------------------------------------
# auroc_parity: a group with one outcome class has no ROC AUC at all
# ---------------------------------------------------------------------------


def _auroc_data(third_group_measurable: bool):
    """A: AUROC 1.00. B: AUROC 0.60. C: measurable at 0.50, or single-class."""
    score_a = np.array([0.9] * 20 + [0.1] * 20)
    score_b = np.array([0.9] * 12 + [0.1] * 8 + [0.5] * 20)
    score_c = np.array([0.9] * 10 + [0.1] * 10 + [0.5] * 20)
    y_c = POS_NEG if third_group_measurable else np.ones(N, dtype=int)
    y_true = np.concatenate([POS_NEG, POS_NEG, y_c])
    score = np.concatenate([score_a, score_b, score_c])
    return y_true, score


class TestAurocParityRefusesAnUnassessedGroup:
    def test_single_outcome_class_group_is_not_assessable(self):
        y_true, score = _auroc_data(third_group_measurable=False)
        # Survivors A and B alone span 1.00 - 0.60 = 0.4000, the pre-fix answer.
        _assert_refused(_quiet(auroc_parity, y_true, score, THREE_GROUPS), 0.4, "auroc_parity")

    def test_the_unassessed_group_is_named_in_a_warning(self):
        """The NaN is the machine-readable state; the warning is the human one.
        Both are required: a value alone does not tell a reader WHICH group the
        validity gate could not check."""
        y_true, score = _auroc_data(third_group_measurable=False)
        with pytest.warns(UserWarning, match=r"ROC AUC is undefined for group\(s\).*'C'"):
            auroc_parity(y_true, score, THREE_GROUPS)

    def test_report_routes_it_to_not_assessable_never_to_passed(self):
        """End to end, and this is the headline. A and B are identical here, so
        the survivors' spread is 0.0 exactly: pre-fix this metric returned that
        sentinel and the report filed the EU-healthcare validity gate under
        passed_metrics (threshold 0.05) while group C was never assessed."""
        y_true = np.concatenate([POS_NEG, POS_NEG, np.ones(N, dtype=int)])
        y_prob = np.concatenate([np.array([0.9] * 20 + [0.1] * 20)] * 3)
        y_pred = (y_prob >= 0.5).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_fairness_report(y_true, y_pred, THREE_GROUPS, y_prob)
        assessment = report["assessment"]
        assert math.isnan(report["metrics"]["auroc_parity"])
        assert "auroc_parity" in {m["metric"] for m in assessment["not_assessable_metrics"]}
        assert "auroc_parity" not in {m["metric"] for m in assessment["passed_metrics"]}


class TestAurocParityStillMeasures:
    """OVER-CORRECTION CONTROL: the refusal must not be bought by refusing data
    that IS measurable. These assert the exact MEASURED value, not that the call
    returned without raising."""

    def test_three_measurable_groups_return_the_reference_spread(self):
        skm = pytest.importorskip("sklearn.metrics")
        y_true, score = _auroc_data(third_group_measurable=True)
        per_group = [
            skm.roc_auc_score(y_true[i * N : (i + 1) * N], score[i * N : (i + 1) * N])
            for i in range(3)
        ]
        assert per_group == [1.0, 0.6, 0.5]  # C is the group that sets the minimum
        expected = max(per_group) - min(per_group)
        assert _quiet(auroc_parity, y_true, score, THREE_GROUPS) == pytest.approx(expected, abs=ABS)
        assert expected == pytest.approx(0.5, abs=ABS)

    def test_two_measurable_groups_are_unchanged(self):
        y_true, score = _auroc_data(third_group_measurable=True)
        assert _quiet(auroc_parity, y_true[: 2 * N], score[: 2 * N], TWO_GROUPS) == pytest.approx(
            0.4, abs=ABS
        )

    def test_measurable_data_raises_no_undefined_warning(self):
        y_true, score = _auroc_data(third_group_measurable=True)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            auroc_parity(y_true, score, THREE_GROUPS)
        assert not [w for w in record if "ROC AUC is undefined" in str(w.message)]


# ---------------------------------------------------------------------------
# fnr_parity_difference: a group with no positive labels has no FNR
# ---------------------------------------------------------------------------


def _fnr_data(third_group_measurable: bool):
    """A: FNR 0.25. B: FNR 0.75. C: measurable at 0.05, or no positives at all."""
    pred_a = np.array([1] * 15 + [0] * 5 + [0] * 20)
    pred_b = np.array([1] * 5 + [0] * 15 + [0] * 20)
    pred_c = np.array([1] * 19 + [0] * 1 + [0] * 20)
    if third_group_measurable:
        y_c, pred_c_used = POS_NEG, pred_c
    else:
        y_c, pred_c_used = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    y_true = np.concatenate([POS_NEG, POS_NEG, y_c])
    y_pred = np.concatenate([pred_a, pred_b, pred_c_used])
    return y_true, y_pred


class TestFnrParityRefusesAnUnassessedGroup:
    def test_group_without_positives_is_not_assessable(self):
        y_true, y_pred = _fnr_data(third_group_measurable=False)
        # Survivors A and B alone span 0.75 - 0.25 = 0.5000, the pre-fix answer.
        _assert_refused(
            _quiet(fnr_parity_difference, y_true, y_pred, THREE_GROUPS),
            0.5,
            "fnr_parity_difference",
        )

    def test_the_unassessed_group_is_named_in_a_warning(self):
        y_true, y_pred = _fnr_data(third_group_measurable=False)
        with pytest.warns(UserWarning, match=r"FNR is undefined for group\(s\).*'C'"):
            fnr_parity_difference(y_true, y_pred, THREE_GROUPS)

    def test_it_agrees_with_equal_opportunity_on_the_same_rows(self):
        """FNR = 1 - TPR, so these two are the same comparison and must be
        assessable under the same conditions. Pre-fix, FNR reported 0.5000 here
        while equal opportunity reported NaN: one report carried the same
        measurement as both measured and not assessable."""
        y_true, y_pred = _fnr_data(third_group_measurable=False)
        fnr = _quiet(fnr_parity_difference, y_true, y_pred, THREE_GROUPS)
        eo = _quiet(equal_opportunity_difference, y_true, y_pred, THREE_GROUPS)
        assert math.isnan(fnr) and math.isnan(eo)


class TestFnrParityStillMeasures:
    """OVER-CORRECTION CONTROL, asserting MEASURED values."""

    def test_three_measurable_groups_return_the_hand_computed_spread(self):
        y_true, y_pred = _fnr_data(third_group_measurable=True)
        per_group = [
            float(np.mean(y_pred[i * N : i * N + 20] == 0)) for i in range(3)
        ]  # the first 20 rows of each group are its positives
        assert per_group == [0.25, 0.75, 0.05]  # C is the group that sets the minimum
        expected = max(per_group) - min(per_group)
        assert _quiet(fnr_parity_difference, y_true, y_pred, THREE_GROUPS) == pytest.approx(
            expected, abs=ABS
        )
        assert expected == pytest.approx(0.7, abs=ABS)

    def test_the_identity_with_equal_opportunity_holds_on_measurable_data(self):
        y_true, y_pred = _fnr_data(third_group_measurable=True)
        fnr = _quiet(fnr_parity_difference, y_true, y_pred, THREE_GROUPS)
        eo = _quiet(equal_opportunity_difference, y_true, y_pred, THREE_GROUPS)
        assert fnr == pytest.approx(eo, abs=ABS)
        assert fnr == pytest.approx(0.7, abs=ABS)

    def test_two_measurable_groups_are_unchanged(self):
        y_true, y_pred = _fnr_data(third_group_measurable=True)
        assert _quiet(
            fnr_parity_difference, y_true[: 2 * N], y_pred[: 2 * N], TWO_GROUPS
        ) == pytest.approx(0.5, abs=ABS)


# ---------------------------------------------------------------------------
# negative_predictive_value_difference: no negative predictions, no NPV
# ---------------------------------------------------------------------------


def _npv_data(third_group_measurable: bool):
    """A: NPV 0.75. B: NPV 0.50. C: measurable at 0.90, or never predicts 0."""
    pred = np.array([1] * 20 + [0] * 20)  # the last 20 rows are the negative decisions
    y_a = np.concatenate([np.array([1, 0] * 10), np.array([0] * 15 + [1] * 5)])
    y_b = np.concatenate([np.array([1, 0] * 10), np.array([0] * 10 + [1] * 10)])
    y_c = np.concatenate([np.array([1, 0] * 10), np.array([0] * 18 + [1] * 2)])
    pred_c = pred if third_group_measurable else np.ones(N, dtype=int)
    y_true = np.concatenate([y_a, y_b, y_c])
    y_pred = np.concatenate([pred, pred, pred_c])
    return y_true, y_pred


class TestNpvParityRefusesAnUnassessedGroup:
    def test_group_without_negative_predictions_is_not_assessable(self):
        y_true, y_pred = _npv_data(third_group_measurable=False)
        # Survivors A and B alone span 0.75 - 0.50 = 0.2500, the pre-fix answer.
        _assert_refused(
            _quiet(negative_predictive_value_difference, y_true, y_pred, THREE_GROUPS),
            0.25,
            "negative_predictive_value_difference",
        )

    def test_the_unassessed_group_is_named_in_a_warning(self):
        y_true, y_pred = _npv_data(third_group_measurable=False)
        with pytest.warns(UserWarning, match=r"NPV is undefined for group\(s\).*'C'"):
            negative_predictive_value_difference(y_true, y_pred, THREE_GROUPS)


class TestNpvParityStillMeasures:
    """OVER-CORRECTION CONTROL, asserting MEASURED values."""

    def test_three_measurable_groups_return_the_hand_computed_spread(self):
        y_true, y_pred = _npv_data(third_group_measurable=True)
        per_group = [
            float(np.mean(y_true[i * N + 20 : (i + 1) * N] == 0)) for i in range(3)
        ]  # the last 20 rows of each group are its negative decisions
        assert per_group == [0.75, 0.5, 0.9]  # C is the group that sets the maximum
        expected = max(per_group) - min(per_group)
        assert _quiet(
            negative_predictive_value_difference, y_true, y_pred, THREE_GROUPS
        ) == pytest.approx(expected, abs=ABS)
        assert expected == pytest.approx(0.4, abs=ABS)

    def test_two_measurable_groups_are_unchanged(self):
        y_true, y_pred = _npv_data(third_group_measurable=True)
        assert _quiet(
            negative_predictive_value_difference,
            y_true[: 2 * N],
            y_pred[: 2 * N],
            TWO_GROUPS,
        ) == pytest.approx(0.25, abs=ABS)


# ---------------------------------------------------------------------------
# The shared shape, in one place
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "builder", "survivor_spread"),
    [
        (auroc_parity, _auroc_data, 0.4),
        (fnr_parity_difference, _fnr_data, 0.5),
        (negative_predictive_value_difference, _npv_data, 0.25),
    ],
    ids=lambda x: getattr(x, "__name__", str(x)),
)
def test_dropping_a_group_can_only_narrow_the_spread(fn, builder, survivor_spread):
    """The monotonicity that makes this class dangerous: the measurable version
    of the same data has a WIDER spread than the survivors alone, because the
    third group sets one end of the range. Reporting the survivors' number is
    therefore a lower bound, and it improves as more groups become
    unmeasurable."""
    measured = _quiet(fn, *builder(True), THREE_GROUPS)
    assert measured > survivor_spread
    _assert_refused(_quiet(fn, *builder(False), THREE_GROUPS), survivor_spread, fn.__name__)
