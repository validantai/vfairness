"""Tests for statistical validation module."""

import numpy as np
import pytest

from vfairness import classification_fairness_report, regression_fairness_report
from vfairness.evaluation.vfairness_metrics._statistics import (
    IntervalType,
    StatisticalResult,
    apply_multiple_testing_correction,
    bayesian_difference_ci,
    bayesian_mean_ci,
    bayesian_proportion_ci,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    interpret_effect_size,
    odds_ratio,
    risk_ratio,
    select_method,
    stratified_bootstrap_ci,
)
from vfairness.evaluation.vfairness_metrics.classification import (
    compute_effect_sizes,
    demographic_parity_difference,
    demographic_parity_difference_with_ci,
    equalized_odds_difference_with_ci,
)
from vfairness.evaluation.vfairness_metrics.regression import (
    compute_regression_effect_sizes,
    mae_parity_difference_with_ci,
)


@pytest.fixture
def classification_data():
    """Generate classification data for testing."""
    np.random.seed(42)
    n = 200

    group = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)

    return y_true, y_pred, group


@pytest.fixture
def small_sample_data():
    """Generate small sample data for testing Bayesian methods."""
    np.random.seed(42)
    n = 40  # Below threshold

    group = np.array(["A"] * 20 + ["B"] * 20)
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)

    return y_true, y_pred, group


@pytest.fixture
def regression_data():
    """Generate regression data for testing."""
    np.random.seed(42)
    n = 200

    group = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.normal(50, 10, n)
    y_pred = y_true + np.random.normal(0, 5, n)

    return y_true, y_pred, group


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_creation(self):
        """Test basic StatisticalResult creation."""
        result = StatisticalResult(
            point_estimate=0.15,
            lower_bound=0.10,
            upper_bound=0.20,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=0.95,
            sample_size=100,
        )

        assert result.point_estimate == 0.15
        assert result.lower_bound == 0.10
        assert result.upper_bound == 0.20
        assert result.interval_type == IntervalType.CONFIDENCE
        assert result.confidence_level == 0.95

    def test_margin_of_error(self):
        """Test margin of error calculation."""
        result = StatisticalResult(
            point_estimate=0.5,
            lower_bound=0.4,
            upper_bound=0.6,
            interval_type=IntervalType.CONFIDENCE,
        )

        assert result.margin_of_error == pytest.approx(0.1)
        assert result.interval_width == pytest.approx(0.2)

    def test_is_significant(self):
        """Test significance detection."""
        # Interval excluding zero
        result1 = StatisticalResult(
            point_estimate=0.15,
            lower_bound=0.05,
            upper_bound=0.25,
            interval_type=IntervalType.CONFIDENCE,
        )
        assert result1.is_significant

        # Interval including zero
        result2 = StatisticalResult(
            point_estimate=0.05,
            lower_bound=-0.05,
            upper_bound=0.15,
            interval_type=IntervalType.CONFIDENCE,
        )
        assert not result2.is_significant

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = StatisticalResult(
            point_estimate=0.15,
            lower_bound=0.10,
            upper_bound=0.20,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=0.95,
            sample_size=100,
        )

        d = result.to_dict()
        assert d["point_estimate"] == 0.15
        assert d["interval_type"] == "confidence"
        assert "margin_of_error" in d
        assert "is_significant" in d


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap CI computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        result = bootstrap_ci(data, np.mean, n_bootstrap=1000, random_state=42)

        assert result.interval_type == IntervalType.CONFIDENCE
        assert result.lower_bound < result.point_estimate < result.upper_bound
        assert result.sample_size == 100
        assert result.n_bootstrap >= 900  # Some may be NaN

    def test_bootstrap_methods(self):
        """Test different bootstrap methods."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        # Percentile method
        result_pct = bootstrap_ci(
            data, np.mean, n_bootstrap=1000, method="percentile", random_state=42
        )

        # Basic method
        result_basic = bootstrap_ci(
            data, np.mean, n_bootstrap=1000, method="basic", random_state=42
        )

        # BCa method
        result_bca = bootstrap_ci(data, np.mean, n_bootstrap=1000, method="bca", random_state=42)

        # All should produce valid intervals
        for r in [result_pct, result_basic, result_bca]:
            assert r.lower_bound < r.upper_bound
            assert not np.isnan(r.point_estimate)

    def test_small_sample_warning(self):
        """Test that small samples produce valid output."""
        data = np.array([1.0])

        result = bootstrap_ci(data, np.mean, n_bootstrap=100)

        assert np.isnan(result.lower_bound)
        assert np.isnan(result.upper_bound)
        assert "warning" in result.metadata


class TestStratifiedBootstrap:
    """Tests for stratified bootstrap CI."""

    def test_stratified_bootstrap(self, classification_data):
        """Test stratified bootstrap maintains proportions."""
        y_true, y_pred, group = classification_data

        def stat_func(data, groups):
            """Simple difference in means."""
            mask_a = groups == "A"
            mask_b = groups == "B"
            return np.mean(data[mask_a]) - np.mean(data[mask_b])

        result = stratified_bootstrap_ci(y_pred, group, stat_func, n_bootstrap=500, random_state=42)

        assert result.interval_type == IntervalType.CONFIDENCE
        assert result.sample_size == len(y_pred)
        assert "n_groups" in result.metadata
        assert result.metadata["n_groups"] == 2


class TestBayesianCI:
    """Tests for Bayesian credible intervals."""

    def test_proportion_ci(self):
        """Test Bayesian proportion CI."""
        # 30 successes out of 100 trials
        result = bayesian_proportion_ci(30, 100)

        assert result.interval_type == IntervalType.CREDIBLE
        assert result.method == "bayesian_beta_binomial"
        assert 0.2 < result.point_estimate < 0.4
        assert result.lower_bound < result.point_estimate < result.upper_bound

    def test_proportion_ci_uniform_prior(self):
        """Test with uniform prior (Beta(1,1))."""
        result = bayesian_proportion_ci(50, 100, prior_alpha=1.0, prior_beta=1.0)

        # With uniform prior and 50/100, posterior mean should be ~0.5
        assert 0.45 < result.point_estimate < 0.55

    def test_proportion_ci_jeffreys_prior(self):
        """Test with Jeffreys prior (Beta(0.5, 0.5))."""
        result = bayesian_proportion_ci(50, 100, prior_alpha=0.5, prior_beta=0.5)

        assert 0.4 < result.point_estimate < 0.6

    def test_difference_ci(self):
        """Test Bayesian difference CI."""
        # Group 1: 40/100, Group 2: 30/100
        result = bayesian_difference_ci(40, 100, 30, 100, random_state=42)

        assert result.interval_type == IntervalType.CREDIBLE
        assert "p_greater" in result.metadata
        assert result.point_estimate > 0  # p1 > p2

    def test_mean_ci(self):
        """Test Bayesian mean CI."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 50)

        result = bayesian_mean_ci(data, random_state=42)

        assert result.interval_type == IntervalType.CREDIBLE
        assert 8 < result.point_estimate < 12


class TestMultipleTestingCorrection:
    """Tests for multiple testing corrections."""

    def test_bonferroni(self):
        """Test Bonferroni correction."""
        p_values = np.array([0.005, 0.02, 0.03, 0.04, 0.05])

        result = bonferroni_correction(p_values)

        assert result.method == "bonferroni"
        # All adjusted p-values should be p * n
        assert np.allclose(result.adjusted_p_values, p_values * 5)
        # With alpha=0.05, only 0.005 (adjusted: 0.025) should be rejected
        assert result.n_rejected == 1

    def test_benjamini_hochberg(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = np.array([0.001, 0.02, 0.03, 0.04, 0.05])

        result = benjamini_hochberg_correction(p_values)

        assert result.method == "benjamini_hochberg"
        # BH should reject more than Bonferroni (less conservative)
        bh_rejected = result.n_rejected

        bonf_result = bonferroni_correction(p_values)
        bonf_rejected = bonf_result.n_rejected

        assert bh_rejected >= bonf_rejected

    def test_apply_correction_methods(self):
        """Test apply_multiple_testing_correction with different methods."""
        p_values = np.array([0.01, 0.02, 0.03])

        bonf = apply_multiple_testing_correction(p_values, method="bonferroni")
        fdr = apply_multiple_testing_correction(p_values, method="fdr")
        none = apply_multiple_testing_correction(p_values, method="none")

        assert bonf.method == "bonferroni"
        assert fdr.method == "benjamini_hochberg"
        assert none.method == "none"

        # No correction should keep original p-values
        assert np.allclose(none.adjusted_p_values, p_values)


class TestEffectSizes:
    """Tests for effect size calculations."""

    def test_cohens_d_equal_groups(self):
        """Test Cohen's d with equal means."""
        group1 = np.array([5, 5, 5, 5, 5])
        group2 = np.array([5, 5, 5, 5, 5])

        d = cohens_d(group1, group2)

        assert d == pytest.approx(0.0)

    def test_cohens_d_different_groups(self):
        """Test Cohen's d with different means."""
        np.random.seed(42)
        # Create groups with ~1 std difference
        group1 = np.random.normal(10, 1, 100)
        group2 = np.random.normal(11, 1, 100)

        d = cohens_d(group1, group2)

        # Should be close to -1 (group1 < group2)
        assert -1.5 < d < -0.5

    def test_cohens_d_interpretation(self):
        """Test effect size interpretation."""
        assert "negligible" in interpret_effect_size(0.1, "cohens_d")
        assert "small" in interpret_effect_size(0.3, "cohens_d")
        assert "medium" in interpret_effect_size(0.6, "cohens_d")
        assert "large" in interpret_effect_size(1.0, "cohens_d")

    def test_cohens_d_interpretation_band_boundaries(self):
        """Pin Cohen's conventional cutoffs 0.2 / 0.5 / 0.8 exactly.

        The bands are half-open ([0.2, 0.5) is small, etc.), so the exact
        boundary value belongs to the HIGHER band and a value epsilon below
        belongs to the lower one. Mid-band tests alone would keep passing if
        a refactor shifted or inverted a boundary comparison.
        """
        eps = 1e-9
        # Exactly at each cutoff -> the higher band.
        assert "small" in interpret_effect_size(0.2, "cohens_d")
        assert "medium" in interpret_effect_size(0.5, "cohens_d")
        assert "large" in interpret_effect_size(0.8, "cohens_d")
        # Epsilon below each cutoff -> the lower band.
        assert "negligible" in interpret_effect_size(0.2 - eps, "cohens_d")
        assert "small" in interpret_effect_size(0.5 - eps, "cohens_d")
        assert "medium" in interpret_effect_size(0.8 - eps, "cohens_d")
        # Banding uses |d|; the sign only drives the direction wording.
        assert "small" in interpret_effect_size(-0.2, "cohens_d")
        assert "lower" in interpret_effect_size(-0.5, "cohens_d")
        assert "higher" in interpret_effect_size(0.5, "cohens_d")

    def test_risk_ratio_interpretation_window_edges(self):
        """Pin the inclusive 0.9-1.1 negligible window for ratio effects."""
        eps = 1e-9
        for effect_type in ("risk_ratio", "odds_ratio"):
            # Both edges are INSIDE the negligible window (0.9 <= d <= 1.1).
            assert interpret_effect_size(0.9, effect_type) == ("negligible difference")
            assert interpret_effect_size(1.1, effect_type) == ("negligible difference")
            # Just outside either edge names a direction.
            assert "lower" in interpret_effect_size(0.9 - eps, effect_type)
            assert "higher" in interpret_effect_size(1.1 + eps, effect_type)
        # The wording is measure-specific: risk vs odds.
        assert "risk" in interpret_effect_size(2.0, "risk_ratio")
        assert "odds" in interpret_effect_size(2.0, "odds_ratio")

    def test_risk_ratio(self):
        """Test risk ratio calculation."""
        # Group 1: 40%, Group 2: 20%
        rr, lower, upper = risk_ratio(40, 100, 20, 100)

        assert rr == pytest.approx(2.0)  # 0.4 / 0.2
        assert lower < rr < upper

    def test_odds_ratio(self):
        """Test odds ratio calculation."""
        # Group 1: 40/60, Group 2: 20/80
        # Odds1 = 40/60 = 0.667, Odds2 = 20/80 = 0.25
        # OR = 0.667 / 0.25 = 2.67
        or_val, lower, upper = odds_ratio(40, 100, 20, 100)

        assert or_val == pytest.approx(40 / 60 / (20 / 80), rel=0.01)
        assert lower < or_val < upper


class TestMethodSelection:
    """Tests for automatic method selection."""

    def test_select_bootstrap_for_large_samples(self):
        """Test bootstrap is selected for large samples."""
        group_sizes = {"A": 100, "B": 100}

        overall, per_group = select_method(group_sizes)

        assert overall == "bootstrap"
        assert per_group["A"] == "bootstrap"
        assert per_group["B"] == "bootstrap"

    def test_select_bayesian_for_small_samples(self):
        """Test Bayesian is selected for small samples."""
        group_sizes = {"A": 20, "B": 25}

        overall, per_group = select_method(group_sizes)

        assert overall == "bayesian"
        assert per_group["A"] == "bayesian"
        assert per_group["B"] == "bayesian"

    def test_mixed_method_selection(self):
        """Test mixed methods for varied sample sizes."""
        group_sizes = {"A": 100, "B": 20}  # One large, one small

        overall, per_group = select_method(group_sizes)

        assert overall == "mixed"
        assert per_group["A"] == "bootstrap"
        assert per_group["B"] == "bayesian"


class TestMetricWithCI:
    """Tests for metrics with confidence intervals."""

    def test_dp_difference_with_ci(self, classification_data):
        """Test demographic parity difference with CI."""
        y_true, y_pred, group = classification_data

        result = demographic_parity_difference_with_ci(
            y_true, y_pred, group, min_group_size=30, n_bootstrap=500, random_state=42
        )

        assert isinstance(result, StatisticalResult)
        assert result.interval_type == IntervalType.CONFIDENCE
        assert result.lower_bound <= result.point_estimate <= result.upper_bound

        # Compare with point estimate
        point_only = demographic_parity_difference(y_true, y_pred, group, min_group_size=30)
        assert result.point_estimate == pytest.approx(point_only, rel=0.01)

    def test_eo_difference_with_ci(self, classification_data):
        """Test equalized odds difference with CI."""
        y_true, y_pred, group = classification_data

        result = equalized_odds_difference_with_ci(
            y_true, y_pred, group, min_group_size=30, n_bootstrap=500, random_state=42
        )

        assert isinstance(result, StatisticalResult)
        assert result.lower_bound <= result.point_estimate <= result.upper_bound

    def test_small_sample_uses_bayesian(self, small_sample_data):
        """Test that small samples trigger Bayesian method warning."""
        y_true, y_pred, group = small_sample_data

        with pytest.warns(UserWarning, match="Bayesian"):
            result = demographic_parity_difference_with_ci(
                y_true,
                y_pred,
                group,
                min_group_size=10,
                n_bootstrap=500,
                method="auto",
                random_state=42,
            )

        # Should still produce valid result
        assert isinstance(result, StatisticalResult)


class TestComputeEffectSizes:
    """Tests for effect size computation functions."""

    def test_classification_effect_sizes(self, classification_data):
        """Test classification effect sizes."""
        y_true, y_pred, group = classification_data

        effects = compute_effect_sizes(y_true, y_pred, group, min_group_size=30)

        assert "A_vs_B" in effects
        assert "cohens_d_positive_rate" in effects["A_vs_B"]
        assert "risk_ratio" in effects["A_vs_B"]
        assert "odds_ratio" in effects["A_vs_B"]
        assert "interpretation" in effects["A_vs_B"]

    def test_regression_effect_sizes(self, regression_data):
        """Test regression effect sizes."""
        y_true, y_pred, group = regression_data

        effects = compute_regression_effect_sizes(y_true, y_pred, group, min_group_size=30)

        assert "A_vs_B" in effects
        assert "cohens_d_predictions" in effects["A_vs_B"]
        assert "cohens_d_residuals" in effects["A_vs_B"]
        assert "interpretation" in effects["A_vs_B"]


class TestReportWithCI:
    """Tests for reports with statistical validation."""

    def test_classification_report_with_ci(self, classification_data):
        """Test classification report includes CI when requested."""
        y_true, y_pred, group = classification_data

        report = classification_fairness_report(
            y_true, y_pred, group, include_ci=True, n_bootstrap=500, random_state=42
        )

        # Should have new sections
        assert "metrics_with_ci" in report
        assert "effect_sizes" in report
        assert "statistical_validation" in report

        # Check metrics with CI structure
        dp_ci = report["metrics_with_ci"]["demographic_parity_difference"]
        assert "point_estimate" in dp_ci
        assert "lower_bound" in dp_ci
        assert "upper_bound" in dp_ci
        assert "interval_type" in dp_ci

        # Check statistical validation info
        assert "method_used" in report["statistical_validation"]
        assert "confidence_level" in report["statistical_validation"]

    def test_classification_report_without_ci(self, classification_data):
        """Test classification report without CI (default)."""
        y_true, y_pred, group = classification_data

        report = classification_fairness_report(y_true, y_pred, group)

        # Should not have CI sections
        assert "metrics_with_ci" not in report
        assert "statistical_validation" not in report

    def test_regression_report_with_ci(self, regression_data):
        """Test regression report includes CI when requested."""
        y_true, y_pred, group = regression_data

        report = regression_fairness_report(
            y_true, y_pred, group, include_ci=True, n_bootstrap=500, random_state=42
        )

        assert "metrics_with_ci" in report
        assert "effect_sizes" in report
        assert "statistical_validation" in report

    def test_report_with_multiple_testing_correction(self, classification_data):
        """Test report with multiple testing correction."""
        y_true, y_pred, group = classification_data

        report = classification_fairness_report(
            y_true,
            y_pred,
            group,
            include_ci=True,
            multiple_testing_correction="fdr",
            n_bootstrap=500,
            random_state=42,
        )

        assert "multiple_testing_correction" in report
        assert report["multiple_testing_correction"]["method"] == "benjamini_hochberg"


class TestMAEParityWithCI:
    """Tests for regression metrics with CI."""

    def test_mae_parity_with_ci(self, regression_data):
        """Test MAE parity difference with CI."""
        y_true, y_pred, group = regression_data

        result = mae_parity_difference_with_ci(
            y_true, y_pred, group, min_group_size=30, n_bootstrap=500, random_state=42
        )

        assert isinstance(result, StatisticalResult)
        assert result.lower_bound <= result.point_estimate <= result.upper_bound
