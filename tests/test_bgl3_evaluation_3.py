"""BGL3 batch evaluation-3: fairness metrics and statistics that answered when
nothing was measurable.

Every test pins a THREE-STATE answer (measured / failed / could-not-check) at the
PUBLIC entry point, and every finding is paired with a CONTROL on healthy data, so
a fix that simply refuses everything fails here too.

The five findings, each measured by execution on 2026-09-27 BEFORE the fix:

  regression.py
    R-1  ``get_group_metrics`` substituted ``r2 = 0.0`` for a group whose y_true is
         constant (R^2 has no denominator there). On 40 rows of constant y_true=100
         predicted at a flat 130 (MAE 30.0) beside 40 varying rows, it published
         ``A.r2 = 0.0`` and ``B.r2 = -2.267``: the group nobody could score read as
         the BETTER of the two, with ZERO warnings. ``r2_parity_difference`` returns
         NaN and warns on the identical input, and the value reached the published
         report under ``group_stats``.
    R-2  ``compute_regression_effect_sizes`` returned ``{}`` with ZERO warnings when
         only one group met min_group_size (80 rows, a single-level attribute), so
         "no pair existed" and "every pair was examined and none showed an effect"
         were the same output. ``_warn_dropped_groups`` covers only a PARTIAL drop.

  classification.py
    C-1  ``selection_rate_disparity_matrix`` published ``min_ratio = 1.0`` and a
         ratio_matrix of all 1.0 when NO group selected anybody (50 A + 50 B, every
         prediction 0): full four-fifths compliance for a 0/0 division that never
         ran, with ZERO warnings. ``min_ratio_pair`` is None there but that cannot
         be used as the signal: all-1 predictions also leave it None while the
         ratio IS computed.
    C-2  ``get_group_metrics_with_ci`` dropped a rate whose own denominator was
         under the gate with no disclosure at all. On 40 rows with no positive label
         in A and no negative label in B, A came back {positive_rate, fpr} and B
         {positive_rate, tpr}, ZERO warnings, while the sibling
         ``get_group_metrics`` reports the same quantity as an explicit NaN.

  attribution.py
    A-1  ``global_importance`` on the label-free (default) path with ONE row
         reported importance exactly 0.0 for every feature against
         y = 2*x0 + 0.5*x1 + 0.1*x2, with ZERO warnings and empty notes: shuffling
         a length-1 column is the identity, so nothing was measured for any
         feature. The scored path already refused the same input.

Also pinned here as CORRECT REFUSALS verified by execution, so a later refactor
cannot quietly remove them: the three regression ``*_with_ci`` variants, the
analyzer's ``explain_metric`` / ``get_report`` / ``get_explanations`` three-state
output, and ``explain_decision``'s NaN + not_assessed contributions.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics import classification as C
from vfairness.evaluation.vfairness_metrics import regression as R
from vfairness.evaluation.vfairness_metrics.attribution import FeatureAttributionExplainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _constant_target_group():
    """40 rows of CONSTANT y_true predicted 30 too high, beside 40 varying rows.

    Group A has no target variance, so its R^2 has no denominator. Its MAE is
    exactly 30.0, which is the point: the group IS measurable on every other
    statistic, so a NaN r2 cannot be waved away as "no data".
    """
    rng = np.random.default_rng(11)
    y_true = np.concatenate([np.full(40, 100.0), rng.normal(50, 10, 40)])
    y_pred = np.concatenate([np.full(40, 130.0), rng.normal(50, 10, 40)])
    sens = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sens


def _healthy_regression():
    rng = np.random.default_rng(1)
    y_true = rng.normal(50, 10, 80)
    y_pred = y_true + rng.normal(0, 2, 80)
    sens = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sens


def _weighted_model():
    """A model dominated by its first feature: y = 2*x0 + 0.5*x1 + 0.1*x2."""
    weights = np.array([2.0, 0.5, 0.1])
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (200, 3))
    explainer = FeatureAttributionExplainer(
        predict=lambda A: np.asarray(A, dtype=float) @ weights,
        feature_names=["f0", "f1", "f2"],
    )
    return explainer, X


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# R-1  regression.get_group_metrics: an undefined R^2 is NaN, not 0.0
# ---------------------------------------------------------------------------


class TestRegressionGroupMetricsUndefinedR2:
    def test_constant_target_group_gets_nan_r2_not_zero(self):
        """BEFORE: A.r2 == 0.0 exactly, B.r2 == -2.267, no warning.

        0.0 is a REAL score on the R^2 scale ("as good as the group mean"), so the
        unmeasured group outranked the measured one.
        """
        y_true, y_pred, sens = _constant_target_group()
        out = _quiet(R.get_group_metrics, y_true, y_pred, sens)
        assert math.isnan(out["A"]["r2"]), f"A.r2 is {out['A']['r2']!r}, expected NaN"
        # The rest of group A IS measured: this is a refusal about ONE statistic,
        # not a group that was dropped.
        assert out["A"]["mae"] == pytest.approx(30.0)
        assert out["A"]["rmse"] == pytest.approx(30.0)
        assert out["A"]["size"] == 40
        # The measured sibling group keeps its real, bad score.
        assert out["B"]["r2"] < 0.0
        assert not math.isnan(out["B"]["r2"])

    def test_the_undefined_r2_is_named_in_a_warning(self):
        """The NaN is the machine-readable half; the warning is the human half."""
        y_true, y_pred, sens = _constant_target_group()
        with pytest.warns(UserWarning, match=r"R² is undefined for group\(s\) \['A'\]"):
            R.get_group_metrics(y_true, y_pred, sens)

    def test_healthy_regression_still_measures_every_r2(self):
        """OVER-CORRECTION CONTROL: real variance gives real, finite R^2 and no warning."""
        y_true, y_pred, sens = _healthy_regression()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = R.get_group_metrics(y_true, y_pred, sens)
        for name in ("A", "B"):
            assert not math.isnan(out[name]["r2"])
            assert out[name]["r2"] > 0.9
            assert out[name]["size"] == 40

    def test_report_group_stats_carries_the_nan_through(self):
        """The number a reader sees is in the report, not in the function's return."""
        y_true, y_pred, sens = _constant_target_group()
        analyzer = FairnessAnalyzer(y_true, y_pred, sens, task_type="regression")
        report = _quiet(analyzer.get_report)
        assert math.isnan(report["group_stats"]["A"]["r2"])
        assert report["group_stats"]["A"]["mae"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# R-2  compute_regression_effect_sizes: an empty result must say it is vacuous
# ---------------------------------------------------------------------------


class TestRegressionEffectSizesDiscloseTheEmptyResult:
    def test_single_group_warns_that_no_pair_was_examined(self):
        """BEFORE: {} with ZERO warnings for a one-level protected attribute."""
        y_true, y_pred, _ = _healthy_regression()
        sens = np.array(["A"] * 80)
        with pytest.warns(UserWarning, match="no pair exists and NO effect size"):
            out = R.compute_regression_effect_sizes(y_true, y_pred, sens)
        assert out == {}

    def test_partial_drop_also_warns(self):
        """One survivor out of two groups: the drop warning alone did not say that
        NOTHING was computed, so both warnings must be present."""
        y_true, y_pred, _ = _healthy_regression()
        sens = np.array(["A"] * 70 + ["B"] * 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = R.compute_regression_effect_sizes(y_true, y_pred, sens)
        assert out == {}
        text = " | ".join(str(w.message) for w in caught)
        assert "min_group_size" in text
        assert "no pair exists and NO effect size" in text

    def test_two_groups_still_return_a_measured_effect_size_in_silence(self):
        """OVER-CORRECTION CONTROL: a real pair is computed and nothing warns."""
        y_true, y_pred, sens = _healthy_regression()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = R.compute_regression_effect_sizes(y_true, y_pred, sens)
        assert list(out) == ["A_vs_B"]
        row = out["A_vs_B"]
        assert row["group1_size"] == 40 and row["group2_size"] == 40
        assert not math.isnan(row["cohens_d_predictions"])
        assert row["interpretation"] in ("negligible", "small", "medium", "large")


# ---------------------------------------------------------------------------
# R-3  the three regression *_with_ci variants: CORRECT refusals, pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        R.mae_parity_difference_with_ci,
        R.rmse_parity_difference_with_ci,
        R.mean_prediction_difference_with_ci,
    ],
)
class TestRegressionWithCiRefusesHonestly:
    def test_single_group_gives_a_nan_interval_and_no_significance(self, fn):
        """VERIFIED CORRECT before the fix and pinned so it stays correct."""
        y_true, y_pred, _ = _healthy_regression()
        sens = np.array(["A"] * 80)
        result = _quiet(fn, y_true, y_pred, sens, n_bootstrap=60, random_state=7)
        assert math.isnan(result.point_estimate)
        assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
        assert result.is_significant is False

    def test_healthy_input_gives_a_real_interval(self, fn):
        """OVER-CORRECTION CONTROL: two real groups get a finite point estimate."""
        y_true, y_pred, sens = _healthy_regression()
        result = _quiet(fn, y_true, y_pred, sens, n_bootstrap=60, random_state=7)
        assert not math.isnan(result.point_estimate)
        assert result.point_estimate >= 0.0
        assert not math.isnan(result.upper_bound)


# ---------------------------------------------------------------------------
# C-1  selection_rate_disparity_matrix: 0/0 is not four-fifths compliance
# ---------------------------------------------------------------------------


class TestSelectionRateMatrixNobodySelected:
    _SENS = np.array(["A"] * 50 + ["B"] * 50)

    def test_nobody_selected_gives_nan_min_ratio_not_one(self):
        """BEFORE: min_ratio 1.0 and a ratio_matrix of all 1.0 for a model that
        rejected 100 percent of applicants, with ZERO warnings. 1.0 is the
        strongest four-fifths all-clear the statistic can give."""
        out = _quiet(C.selection_rate_disparity_matrix, np.zeros(100, dtype=int), self._SENS)
        assert math.isnan(out["min_ratio"]), f"min_ratio is {out['min_ratio']!r}"
        assert math.isnan(out["ratio_matrix"]["A"]["B"])
        assert math.isnan(out["ratio_matrix"]["B"]["A"])
        # The RATES were measured and they are genuinely equal, so the DIFFERENCE
        # stays a measurement. Refusing it too would be the opposite defect.
        assert out["max_difference"] == pytest.approx(0.0)
        assert out["rates"]["A"]["rate"] == pytest.approx(0.0)
        assert out["rates"]["A"]["n"] == 50

    def test_nobody_selected_is_disclosed_in_a_warning(self):
        with pytest.warns(UserWarning, match="no group has a positive selection rate"):
            C.selection_rate_disparity_matrix(np.zeros(100, dtype=int), self._SENS)

    def test_constant_scores_reach_the_same_refusal(self):
        """The constant-score branch selects nobody by design and said so, and
        still published min_ratio 1.0 before the fix."""
        out = _quiet(C.selection_rate_disparity_matrix, np.full(100, 0.7), self._SENS)
        assert math.isnan(out["min_ratio"])

    def test_everybody_selected_is_a_measured_ratio_of_one(self):
        """OVER-CORRECTION CONTROL, and the reason min_ratio_pair cannot be the
        signal: every rate is 1.0, the division DOES happen, the answer really is
        1.0, and min_ratio_pair is None anyway because nothing beat the seed."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = C.selection_rate_disparity_matrix(np.ones(100, dtype=int), self._SENS)
        assert out["min_ratio"] == pytest.approx(1.0)
        assert out["min_ratio_pair"] is None
        assert out["ratio_matrix"]["A"]["B"] == pytest.approx(1.0)

    def test_one_zero_rate_against_one_positive_rate_is_still_measured(self):
        """OVER-CORRECTION CONTROL: 0/0.4 is 0.0, the starkest real finding, and
        0.4/0 is genuinely unbounded. Neither may become NaN."""
        y_pred = np.array([0] * 50 + [1] * 20 + [0] * 30)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = C.selection_rate_disparity_matrix(y_pred, self._SENS)
        assert out["min_ratio"] == pytest.approx(0.0)
        assert out["min_ratio_pair"] == ("A", "B")
        assert out["ratio_matrix"]["B"]["A"] == float("inf")
        assert out["max_difference"] == pytest.approx(0.4)

    def test_healthy_matrix_is_untouched(self):
        """OVER-CORRECTION CONTROL: A at 0.8, B at 0.4."""
        y_pred = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = C.selection_rate_disparity_matrix(y_pred, self._SENS)
        assert out["min_ratio"] == pytest.approx(0.5)
        assert out["min_ratio_pair"] == ("B", "A")
        assert out["ratio_matrix"]["A"]["B"] == pytest.approx(2.0)
        assert out["max_difference"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# C-2  get_group_metrics_with_ci: name the rate that was left out
# ---------------------------------------------------------------------------


class TestGroupMetricsWithCiDisclosesTheOmittedRate:
    @staticmethod
    def _no_positives_in_a():
        """A: 40 rows, not one positive LABEL. B: 40 rows, not one negative label."""
        y_true = np.array([0] * 40 + [1] * 40)
        y_pred = np.array([0] * 20 + [1] * 20 + [1] * 40)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        return y_true, y_pred, sens

    def test_the_omitted_rates_are_named(self):
        """BEFORE: A came back {positive_rate, fpr} and B {positive_rate, tpr},
        with ZERO warnings, so nothing said why half the TPR comparison is gone."""
        y_true, y_pred, sens = self._no_positives_in_a()
        with pytest.warns(UserWarning, match=r"A\.tpr \(0 row\(s\) with y_true=1\)"):
            out = C.get_group_metrics_with_ci(y_true, y_pred, sens, min_group_size=30)
        assert "tpr" not in out["A"]
        assert "fpr" not in out["B"]
        # What WAS measurable is still measured, including the other group's half.
        assert out["A"]["fpr"].point_estimate == pytest.approx(0.5)
        assert out["B"]["tpr"].point_estimate == pytest.approx(1.0)

    def test_the_sibling_reports_the_same_gap_as_an_explicit_nan(self):
        """The two public functions must agree that this rate is unmeasurable."""
        y_true, y_pred, sens = self._no_positives_in_a()
        plain = _quiet(C.get_group_metrics, y_true, y_pred, sens)
        assert math.isnan(plain["A"]["tpr"])
        assert math.isnan(plain["B"]["fpr"])

    def test_a_group_with_both_labels_present_warns_about_nothing(self):
        """OVER-CORRECTION CONTROL: every denominator clears the gate, so every
        rate is built and the call is silent."""
        y_true = np.array(([1] * 30 + [0] * 30) * 2)
        y_pred = np.array(([1] * 20 + [0] * 40) * 2)
        sens = np.array(["A"] * 60 + ["B"] * 60)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = C.get_group_metrics_with_ci(y_true, y_pred, sens, min_group_size=30)
        for name in ("A", "B"):
            assert set(out[name]) == {"positive_rate", "tpr", "fpr"}
            assert not math.isnan(out[name]["tpr"].point_estimate)


# ---------------------------------------------------------------------------
# A-1  global_importance: a single row cannot be permuted
# ---------------------------------------------------------------------------


class TestGlobalImportanceNeedsTwoRows:
    def test_single_row_label_free_path_refuses(self):
        """BEFORE: f0/f1/f2 all importance exactly 0.0, notes [], no warning, for
        a model whose first weight is 2.0. This is the DEFAULT call (no y)."""
        explainer, X = _weighted_model()
        with pytest.warns(UserWarning, match="needs at least 2 rows"):
            result = explainer.global_importance(X[:1], n_repeats=5)
        assert len(result.contributions) == 3
        assert all(math.isnan(c.importance) for c in result.contributions)
        assert all(c.direction == "not_assessed" for c in result.contributions)
        assert result.notes and "at least 2 rows" in result.notes[0]

    def test_zero_rows_refuses_by_statement_not_by_numpy_accident(self):
        """BEFORE: NaN arrived only through a 'Mean of empty slice' RuntimeWarning,
        which says nothing about what was not measured."""
        explainer, _ = _weighted_model()
        with pytest.warns(UserWarning, match="X has 0"):
            result = explainer.global_importance(np.empty((0, 3)), n_repeats=3)
        assert all(math.isnan(c.importance) for c in result.contributions)

    def test_single_row_scored_path_refuses_too(self):
        """The guard sits above the dispatch, so the labelled path answers the
        same way rather than through its constant-target branch."""
        explainer, X = _weighted_model()
        with pytest.warns(UserWarning, match="needs at least 2 rows"):
            result = explainer.global_importance(X[:1], y=np.array([1.0]), n_repeats=3)
        assert all(math.isnan(c.importance) for c in result.contributions)

    def test_full_dataset_label_free_path_ranks_the_dominant_feature_first(self):
        """OVER-CORRECTION CONTROL: 200 rows still produce a real ranking, and
        f0's importance is roughly its weight ratio against f1."""
        explainer, X = _weighted_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = explainer.global_importance(X, n_repeats=3)
        imps = {c.feature: c.importance for c in result.contributions}
        assert all(not math.isnan(v) for v in imps.values())
        assert imps["f0"] > imps["f1"] > imps["f2"] > 0.0
        assert imps["f0"] > 1.0

    def test_scored_path_on_a_real_target_still_signs_the_direction(self):
        """OVER-CORRECTION CONTROL for the labelled path."""
        explainer, X = _weighted_model()
        y = np.asarray(X, dtype=float) @ np.array([2.0, 0.5, 0.1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = explainer.global_importance(X, y=y, n_repeats=3)
        top = result.top(1)[0]
        assert top.feature == "f0"
        assert top.direction == "increase"
        assert top.signed_value is not None and top.signed_value > 0.0

    def test_constant_target_refusal_is_unchanged(self):
        """The pre-existing refusal on the scored path must survive the new guard."""
        explainer, X = _weighted_model()
        with pytest.warns(UserWarning, match="R\\^2 is undefined on a constant target"):
            result = explainer.global_importance(X, y=np.ones(200), n_repeats=3)
        assert all(math.isnan(c.importance) for c in result.contributions)


# ---------------------------------------------------------------------------
# A-2  explain_decision and shapley_matrix: CORRECT refusals, pinned
# ---------------------------------------------------------------------------


class TestLocalAttributionRefusesHonestly:
    def test_unusable_background_gives_nan_and_not_assessed(self):
        """VERIFIED CORRECT: an all-NaN background yields NaN importances with
        direction 'not_assessed', never 0.0 / 'neutral'."""
        explainer, X = _weighted_model()
        result = _quiet(explainer.explain_decision, X[0], np.full((50, 3), np.nan))
        assert all(math.isnan(c.importance) for c in result.contributions)
        assert all(c.direction == "not_assessed" for c in result.contributions)
        assert math.isnan(result.base_value)

    def test_healthy_local_explanation_is_signed_and_finite(self):
        """OVER-CORRECTION CONTROL."""
        explainer, X = _weighted_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = explainer.explain_decision(X[0], X)
        assert all(not math.isnan(c.importance) for c in result.contributions)
        assert {c.direction for c in result.contributions} <= {"increase", "decrease", "neutral"}
        assert result.top(1)[0].feature == "f0"

    def test_shapley_matrix_on_an_empty_background_is_all_nan(self):
        """VERIFIED CORRECT: the reference cannot be formed, so no contribution is."""
        explainer, X = _weighted_model()
        result = _quiet(
            explainer.shapley_matrix, X[:2], background=np.empty((0, 3)), n_permutations=5
        )
        assert math.isnan(result["base_value"])
        assert all(math.isnan(v) for row in result["values"] for v in row)

    def test_shapley_matrix_healthy_closes_on_the_prediction(self):
        """OVER-CORRECTION CONTROL: completeness holds to numerical precision."""
        explainer, X = _weighted_model()
        result = _quiet(explainer.shapley_matrix, X[:3], background=X, n_permutations=8)
        weights = np.array([2.0, 0.5, 0.1])
        for i, row in enumerate(result["values"]):
            assert result["base_value"] + sum(row) == pytest.approx(float(X[i] @ weights), abs=1e-9)


# ---------------------------------------------------------------------------
# A-3  analyzer explain_metric / get_report / get_explanations: CORRECT, pinned
# ---------------------------------------------------------------------------


def _classification_analyzer(sens):
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 80)
    y_pred = rng.integers(0, 2, 80)
    return FairnessAnalyzer(y_true, y_pred, sens)


class TestAnalyzerExplanationsAreThreeState:
    def test_explain_metric_on_an_unmeasurable_metric_says_could_not_check(self):
        """VERIFIED CORRECT: one group, so no comparison exists."""
        analyzer = _classification_analyzer(np.array(["A"] * 80))
        explanation = _quiet(analyzer.explain_metric, "demographic_parity_difference")
        assert math.isnan(explanation.value)
        assert explanation.severity == "could_not_check"
        assert "COULD NOT CHECK" in explanation.evaluation
        assert "NOT MEASURED" in explanation.recommendation

    def test_explain_metric_on_a_measured_metric_grades_it(self):
        """OVER-CORRECTION CONTROL: a real value is graded, not refused."""
        analyzer = _classification_analyzer(np.array(["A"] * 40 + ["B"] * 40))
        explanation = _quiet(analyzer.explain_metric, "demographic_parity_difference")
        assert not math.isnan(explanation.value)
        assert explanation.severity != "could_not_check"
        assert "COULD NOT CHECK" not in explanation.evaluation

    def test_get_report_withholds_the_score_when_nothing_was_comparable(self):
        """VERIFIED CORRECT: fairness_score None, assessable False, and every
        metric in not_assessable_metrics rather than in passed_metrics."""
        analyzer = _classification_analyzer(np.array(["A"] * 80))
        report = _quiet(analyzer.get_report)
        assessment = report["assessment"]
        assert assessment["fairness_score"] is None
        assert assessment["assessable"] is False
        assert assessment["passed_metrics"] == []
        assert len(assessment["not_assessable_metrics"]) >= 5
        assert "NOT ASSESSABLE" in assessment["summary"]

    def test_get_report_on_two_groups_produces_a_graded_verdict(self):
        """OVER-CORRECTION CONTROL."""
        analyzer = _classification_analyzer(np.array(["A"] * 40 + ["B"] * 40))
        report = _quiet(analyzer.get_report)
        assessment = report["assessment"]
        assert assessment["assessable"] is True
        assert isinstance(assessment["fairness_score"], float)
        assert assessment["passed_metrics"]
        assert assessment["not_assessable_metrics"] == []

    def test_get_explanations_summary_refuses_to_read_as_a_pass(self):
        """VERIFIED CORRECT: the empty-dict shape is not how this refuses."""
        analyzer = _classification_analyzer(np.array(["A"] * 80))
        explanations = _quiet(analyzer.get_explanations)
        assert set(explanations) >= {"metrics", "summary"}
        assert "NOT AVAILABLE (could not check)" in explanations["summary"]
        assert "NOT GRADED" in explanations["summary"]

    def test_get_explanations_on_healthy_data_reports_a_score(self):
        """OVER-CORRECTION CONTROL."""
        analyzer = _classification_analyzer(np.array(["A"] * 40 + ["B"] * 40))
        explanations = _quiet(analyzer.get_explanations)
        assert "Overall Fairness Score:" in explanations["summary"]
        assert "NOT AVAILABLE" not in explanations["summary"]
