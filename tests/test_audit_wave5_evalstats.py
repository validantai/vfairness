"""Audit wave 5: evaluation/statistics honesty fixes.

Pins the ten confirmed findings for the vfairness_metrics evaluation stack:
exact Fisher test at every n, honest MDE z-values, probability-as-y_pred
disclosure, observed rates as point estimates, degenerate-score handling,
bounded removal-curve AUC, matched-population significance baselines,
representation-safe threshold checks, constant-target regression guards,
and threshold-true dashboard bands.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    fisher_exact_test,
    minimum_detectable_effect,
)
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.classification import (
    get_group_metrics_with_ci,
    selection_rate_disparity_matrix,
)
from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    removal_curve_auc,
)
from vfairness.evaluation.vfairness_metrics.intersectional import (
    generate_structured_findings,
)
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)

# 1. fisher_exact_test stays exact above n=200


class TestFisherExactLargeN:
    def test_sparse_cell_large_n_matches_scipy(self):
        from scipy.stats import fisher_exact as sp_fisher

        # n=250 with an expected cell < 1: the old chi-square fallback
        # returned p=1.0 for exactly this shape.
        a, b, c, d = 1, 0, 100, 149
        p = fisher_exact_test(a, b, c, d)
        p_ref = float(sp_fisher([[a, b], [c, d]])[1])
        assert p == pytest.approx(p_ref, rel=1e-9)
        assert p < 0.999  # never the silent p=1.0 sentinel

    def test_moderate_large_table_matches_scipy(self):
        from scipy.stats import fisher_exact as sp_fisher

        a, b, c, d = 30, 70, 50, 60
        p = fisher_exact_test(a, b, c, d)
        p_ref = float(sp_fisher([[a, b], [c, d]])[1])
        assert p == pytest.approx(p_ref, rel=1e-9)

    def test_small_table_still_exact(self):
        from scipy.stats import fisher_exact as sp_fisher

        a, b, c, d = 3, 7, 8, 2
        p = fisher_exact_test(a, b, c, d)
        p_ref = float(sp_fisher([[a, b], [c, d]])[1])
        assert p == pytest.approx(p_ref, rel=1e-9)

    def test_empty_table_is_could_not_check_not_a_p_value(self):
        """UPDATED 2026-09-25 (BGL-G001 resolved). This asserted == 1.0, which was
        the recorded defect rather than the contract: an empty table has no
        estimable odds ratio, and 1.0 reads as "no association found" from nothing
        counted at all. The subject of the test is kept -- the empty table must not
        crash and must answer definitely -- and the answer is now nan with a
        warning. See tests/test_surface_grade_g001.py for all five shapes."""
        import math
        import warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            value = fisher_exact_test(0, 0, 0, 0)
        assert math.isnan(value)
        assert any("could not check" in str(w.message) for w in rec)


# 2. minimum_detectable_effect honors requested alpha and power


class TestMinimumDetectableEffect:
    @staticmethod
    def _expected(n1, n2, alpha, power, baseline=0.5):
        from scipy.stats import norm

        z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
        se = np.sqrt(baseline * (1 - baseline) * (1 / n1 + 1 / n2))
        return min(float(z * se), 1.0)

    @pytest.mark.parametrize(
        "alpha,power",
        [
            (0.05, 0.80),
            (0.01, 0.90),
            (0.20, 0.80),
            (0.05, 0.95),
            (0.10, 0.85),
        ],
    )
    def test_matches_normal_approximation(self, alpha, power):
        got = minimum_detectable_effect(100, 100, alpha=alpha, power=power)
        assert got == pytest.approx(self._expected(100, 100, alpha, power), rel=1e-6)

    def test_stricter_settings_increase_mde(self):
        base = minimum_detectable_effect(100, 100, alpha=0.05, power=0.80)
        stricter_alpha = minimum_detectable_effect(100, 100, alpha=0.01, power=0.80)
        higher_power = minimum_detectable_effect(100, 100, alpha=0.05, power=0.95)
        looser_alpha = minimum_detectable_effect(100, 100, alpha=0.20, power=0.80)
        assert stricter_alpha > base
        assert higher_power > base  # old ladder made this SMALLER (0.524 z)
        assert looser_alpha < base


# 3. analyzer discloses probability-looking y_pred auto-detection


class TestAnalyzerTaskAutoDetectWarning:
    def _data(self):
        y_true = np.array([0, 1] * 40)
        sens = np.array(["a", "b"] * 40)
        return y_true, sens

    def test_probabilities_as_y_pred_warn(self):
        y_true, sens = self._data()
        y_prob = np.linspace(0.05, 0.95, 80)
        with pytest.warns(UserWarning, match="regression.*PROBABILITIES|PROBABILITIES"):
            an = FairnessAnalyzer(y_true, y_prob, sens)
        assert an.task_type == "regression"

    def test_binary_predictions_do_not_warn(self):
        y_true, sens = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            an = FairnessAnalyzer(y_true, y_true.copy(), sens)
        assert an.task_type == "classification"

    def test_true_regression_values_do_not_warn(self):
        y_true, sens = self._data()
        y_pred = np.linspace(10.0, 50.0, 80)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            an = FairnessAnalyzer(y_pred + 1.0, y_pred, sens)
        assert an.task_type == "regression"


# 4. Bayesian per-group CI reports the OBSERVED rate as point estimate


class TestObservedRatePointEstimate:
    def test_zero_positives_reports_zero(self):
        # g0: 0/20 positives (small -> Bayesian path); g1: large control
        y_pred = np.concatenate([np.zeros(20), np.tile([0, 1], 20)]).astype(int)
        y_true = np.concatenate([np.tile([0, 1], 10), np.tile([0, 1], 20)]).astype(int)
        sens = np.array(["g0"] * 20 + ["g1"] * 40)
        res = get_group_metrics_with_ci(
            y_pred=y_pred, y_true=y_true, sensitive_attr=sens, min_group_size=5
        )
        pr = res["g0"]["positive_rate"]
        assert pr.point_estimate == 0.0
        # Posterior interval kept and disclosed
        assert pr.upper_bound > 0.0
        assert pr.metadata["point_estimate_source"] == "observed_rate"
        assert pr.metadata["posterior_mean"] == pytest.approx(1 / 22)
        assert pr.method == "bayesian_beta_binomial"

    def test_large_group_unchanged(self):
        y_pred = np.tile([0, 1], 30).astype(int)
        y_true = y_pred.copy()
        sens = np.array(["g"] * 60)
        res = get_group_metrics_with_ci(y_true, y_pred, sens, min_group_size=5)
        pr = res["g"]["positive_rate"]
        assert pr.point_estimate == pytest.approx(0.5)


# 5. selection_rate_disparity_matrix degenerate constant scores


class TestConstantScoreSelection:
    def test_constant_scores_warn_and_select_nobody(self):
        scores = np.full(60, 0.7)
        sens = np.array(["a", "b", "c"] * 20)
        with pytest.warns(UserWarning, match="constant score"):
            out = selection_rate_disparity_matrix(scores, sens)
        assert all(v["rate"] == 0.0 for v in out["rates"].values())

    def test_varying_scores_unaffected(self):
        scores = np.array([0.1, 0.9] * 30)
        sens = np.array(["a", "b", "c"] * 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = selection_rate_disparity_matrix(scores, sens)
        # Median threshold selects the 0.9 half in every group
        assert all(v["rate"] == pytest.approx(0.5) for v in out["rates"].values())

    def test_constant_binary_ones_still_reported_as_selected(self):
        # All-1 BINARY labels are genuine observed selections, not a
        # thresholding artifact: no warning, rate 1.0.
        scores = np.ones(40)
        sens = np.array(["a", "b"] * 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = selection_rate_disparity_matrix(scores, sens)
        assert all(v["rate"] == 1.0 for v in out["rates"].values())


# 6. removal_curve_auc bounded to [0, 1]


class TestRemovalCurveAucBounds:
    def test_overshooting_prediction_clamped_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        attr = np.array([3.0, 2.0, 1.0])
        bg = np.zeros(3)

        def predict(arr):
            # p0 = 0.05; any masking swings the output to 0.95, so the raw
            # normalised area is ~15 without the clamp.
            return np.array([0.05 if arr.ravel()[0] == 1.0 else 0.95])

        auc = removal_curve_auc(predict, x, attr, bg)
        assert auc == 1.0

    def test_normal_case_in_bounds_and_unclamped(self):
        x = np.array([1.0, 1.0])
        attr = np.array([2.0, 1.0])
        bg = np.zeros(2)

        def predict(arr):
            # Linear model 0.25*x1 + 0.25*x2: p0=0.5, masking degrades
            # gradually and the raw AUC is already within [0, 1].
            row = arr.ravel()
            return np.array([0.25 * row[0] + 0.25 * row[1]])

        auc = removal_curve_auc(predict, x, attr, bg)
        assert 0.0 < auc < 1.0

    def test_no_effect_model_scores_zero(self):
        """A model whose output never moves scores 0.0, whatever is removed.

        The attribution vector was `[1.0, 1.0]` until 2026-09-27, which is FULLY
        TIED and was never the subject here: it was scenery, picked without a
        thought because any two numbers would do. A tied vector defines no masking
        order, argsort hands back the identity permutation exactly as it does for
        NaN, and removal_curve_auc now refuses it with a warning rather than
        scoring the identity order as though it had been chosen. So the 0.0 this
        test is about has to come from the MODEL, and the attributions have to be
        distinguishable for the question to be asked at all.

        Measured after the change: attr [1.0, 1.0] returns nan with the warning
        "all 2 |attribution| values are identical", and attr [2.0, 1.0] returns
        0.0 in silence, which is this test.
        """
        x = np.array([1.0, 1.0])
        attr = np.array([2.0, 1.0])
        bg = np.zeros(2)
        auc = removal_curve_auc(lambda arr: np.array([0.5]), x, attr, bg)
        assert auc == 0.0


# 7. structured findings baseline over the included population


class TestFindingsBaselinePopulation:
    def test_baseline_recomputed_over_included_groups(self):
        from vfairness.evaluation.vfairness_metrics.intersectional import (
            _proportion_test,
        )

        groups = [
            {
                "group": "a",
                "positive_rate": 0.30,
                "ground_truth_rate": 0.30,
                "prediction_delta": 0.0,
                "false_positive_rate": 0.1,
                "size": 100,
            },
            {
                "group": "b",
                "positive_rate": 0.50,
                "ground_truth_rate": 0.30,
                "prediction_delta": 0.20,
                "false_positive_rate": 0.1,
                "size": 100,
            },
        ]
        # Stored overall_rate deliberately reflects a LARGER population
        # (e.g. includes a below-min-size group) and disagrees with the
        # included groups' pooled rate of 0.40.
        inter = {
            "all_groups": groups,
            "overall_rate": 0.90,
            "max_disparity": 0.0,
            "max_ground_truth_disparity": 0.0,
        }
        findings = generate_structured_findings(inter)
        over = [f for f in findings if f["type"] == "over_prediction" and f["groups"] == ["b"]]
        assert len(over) == 1
        # Corrected hypothesis (audit3 eval-core-stats): an over_prediction finding
        # tests the group's prediction rate against its OWN ground-truth rate on the
        # group's own n (pred 0.50 vs gt 0.30, n=100), which is the claim the finding
        # actually makes. The prior pin enshrined a pred-vs-pooled-population-rate
        # comparison, a different (and for this finding, wrong) hypothesis.
        expected_p = round(_proportion_test(0.50, 100, 0.30, 100), 6)
        wrong_pop_p = round(_proportion_test(0.50, 100, 0.40, 200), 6)
        assert over[0]["p_value"] == expected_p
        assert over[0]["p_value"] != wrong_pop_p


# 8. threshold comparison robust to float representation


class TestThresholdEpsilon:
    @staticmethod
    def _two_rate_data(rate_a, rate_b, n=100):
        pos_a = int(round(rate_a * n))
        pos_b = int(round(rate_b * n))
        y_pred = np.concatenate(
            [
                np.repeat([1, 0], [pos_a, n - pos_a]),
                np.repeat([1, 0], [pos_b, n - pos_b]),
            ]
        ).astype(int)
        y_true = y_pred.copy()
        sens = np.array(["a"] * n + ["b"] * n)
        return y_true, y_pred, sens

    def test_exactly_at_threshold_passes(self):
        # 0.80 - 0.70 = 0.10000000000000009 in floats; must PASS at 0.10
        y_true, y_pred, sens = self._two_rate_data(0.80, 0.70)
        rep = classification_fairness_report(y_true, y_pred, sens)
        failed = {e["metric"] for e in rep["assessment"]["failed_metrics"]}
        assert "demographic_parity_difference" not in failed

    def test_real_violation_still_fails(self):
        y_true, y_pred, sens = self._two_rate_data(0.85, 0.55)
        rep = classification_fairness_report(y_true, y_pred, sens)
        failed = {e["metric"] for e in rep["assessment"]["failed_metrics"]}
        assert "demographic_parity_difference" in failed


# 9. regression report with constant y_true


class TestRegressionConstantTarget:
    @staticmethod
    def _data():
        y_true = np.full(200, 5.0)
        # Deterministic tiny prediction noise (no RNG)
        y_pred = 5.0 + 0.01 * np.tile([1.0, -1.0, 0.5, -0.5], 50)
        sens = np.array(["a", "b"] * 100)
        return y_true, y_pred, sens

    def test_relative_metrics_not_assessable(self):
        y_true, y_pred, sens = self._data()
        with pytest.warns(UserWarning, match="constant"):
            rep = regression_fairness_report(y_true, y_pred, sens)
        na = rep["assessment"]["not_assessable_metrics"]
        # With a globally constant target, the scale-relative metrics have no
        # defined scale AND R2 is undefined (no variance to explain, VB-EVAL-1),
        # so every metric is not assessable. The report must not certify fairness
        # (previously R2 returned 0.0 and passed, and after VB-EVAL-1 its NaN was
        # mis-scored as a FAIL; now it is correctly surfaced as not assessable).
        assert {e["metric"] for e in na} == {
            "mae_parity_difference",
            "rmse_parity_difference",
            "mean_prediction_difference",
            "r2_parity_difference",
        }
        assert all(e["status"] == "NOT_ASSESSABLE" for e in na)
        assessed = rep["assessment"]["passed_metrics"] + rep["assessment"]["failed_metrics"]
        assert assessed == []
        assert "not assessable" in rep["assessment"]["summary"]

    def test_explicit_absolute_thresholds_still_assessed(self):
        y_true, y_pred, sens = self._data()
        with pytest.warns(UserWarning, match="constant"):
            rep = regression_fairness_report(
                y_true, y_pred, sens, thresholds={"mae_parity_difference": 0.5}
            )
        na = {e["metric"] for e in rep["assessment"]["not_assessable_metrics"]}
        assert "mae_parity_difference" not in na
        passed = {e["metric"] for e in rep["assessment"]["passed_metrics"]}
        assert "mae_parity_difference" in passed

    def test_normal_regression_unaffected(self):
        y_true = np.tile([1.0, 2.0, 3.0, 4.0], 50)
        y_pred = y_true + 0.1
        sens = np.array(["a", "b"] * 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rep = regression_fairness_report(y_true, y_pred, sens)
        assert rep["assessment"]["not_assessable_metrics"] == []


# 10. dashboard band drawn from actual thresholds


class TestVisualizationThresholdBand:
    def test_bands_match_per_metric_thresholds(self):
        pytest.importorskip("plotly")
        from plotly.subplots import make_subplots

        from vfairness.evaluation.vfairness_metrics.visualization import (
            _add_metrics_panel,
            _get_palette,
        )

        report = {
            "metrics": {
                "demographic_parity_difference": 0.08,
                "equal_opportunity_difference": 0.08,
                "demographic_parity_ratio": 0.85,
            },
            "thresholds_used": {
                "demographic_parity_difference": 0.10,
                "equal_opportunity_difference": 0.05,
                "demographic_parity_ratio": 0.80,
            },
        }
        fig = make_subplots(rows=1, cols=1)
        _add_metrics_panel(fig, report, _get_palette(), 1, 1)

        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 3
        # Difference metrics: band spans -threshold..+threshold
        assert (rects[0].y0, rects[0].y1) == (-0.10, 0.10)
        assert (rects[1].y0, rects[1].y1) == (-0.05, 0.05)
        # Ratio metric: acceptable region starts AT the minimum threshold
        assert rects[2].y0 == 0.80
        assert rects[2].y1 >= 1.0
        # No shape reproduces the old hardcoded uniform +/-0.1 band for the
        # 0.05-threshold metric
        band_heights = {(s.y0, s.y1) for s in rects}
        assert (-0.1, 0.1) in band_heights  # only for the 0.10 metric
        assert (-0.05, 0.05) in band_heights

        lines = [s for s in fig.layout.shapes if s.type == "line"]
        line_ys = sorted({s.y0 for s in lines})
        assert line_ys == [-0.10, -0.05, 0.05, 0.10, 0.80]
