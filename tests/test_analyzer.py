"""
Tests for the FairnessAnalyzer unified interface.
"""

import numpy as np
import pytest

from vfairness import FairnessAnalyzer, MetricResult


class TestFairnessAnalyzerBasic:
    """Basic tests for FairnessAnalyzer."""

    @pytest.fixture
    def classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)
        return y_true, y_pred, gender

    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.normal(100, 20, n)
        y_pred = y_true + np.random.normal(0, 10, n)
        gender = np.random.choice(["M", "F"], n)
        return y_true, y_pred, gender

    def test_classification_auto_detect(self, classification_data):
        """Test that classification task is auto-detected."""
        y_true, y_pred, gender = classification_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        assert analyzer.task_type == "classification"

    def test_regression_auto_detect(self, regression_data):
        """Test that regression task is auto-detected."""
        y_true, y_pred, gender = regression_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        assert analyzer.task_type == "regression"

    def test_explicit_task_type(self, classification_data):
        """Test explicit task type specification."""
        y_true, y_pred, gender = classification_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender, task_type="classification")
        assert analyzer.task_type == "classification"

    def test_groups_property(self, classification_data):
        """Test groups property."""
        y_true, y_pred, gender = classification_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        assert set(analyzer.groups) == {"M", "F"}

    def test_group_sizes_property(self, classification_data):
        """Test group_sizes property."""
        y_true, y_pred, gender = classification_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        sizes = analyzer.group_sizes
        assert "M" in sizes
        assert "F" in sizes
        assert sum(sizes.values()) == len(y_true)

    def test_n_samples_property(self, classification_data):
        """Test n_samples property."""
        y_true, y_pred, gender = classification_data
        analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        assert analyzer.n_samples == len(y_true)


class TestFairnessAnalyzerMetrics:
    """Test metric computation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with sample data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)
        return FairnessAnalyzer(y_true, y_pred, gender)

    def test_demographic_parity_float(self, analyzer):
        """Test DP returns float without CI."""
        result = analyzer.demographic_parity_difference(include_ci=False)
        assert isinstance(result, float)
        assert -1 <= result <= 1

    def test_demographic_parity_with_ci(self, analyzer):
        """Test DP returns MetricResult with CI."""
        result = analyzer.demographic_parity_difference(
            include_ci=True,
            n_bootstrap=100,  # Small for speed
            random_state=42,
        )
        assert isinstance(result, MetricResult)
        assert result.metric_name == "demographic_parity_difference"
        assert -1 <= result.value <= 1
        assert len(result.confidence_interval) == 2

    def test_equalized_odds(self, analyzer):
        """Test equalized odds difference."""
        result = analyzer.equalized_odds_difference(include_ci=False)
        assert isinstance(result, float)

    def test_equal_opportunity(self, analyzer):
        """Test equal opportunity difference."""
        result = analyzer.equal_opportunity_difference(include_ci=False)
        assert isinstance(result, float)

    def test_compute_all_metrics(self, analyzer):
        """Test computing all metrics at once."""
        results = analyzer.compute_all_metrics(include_ci=False)
        assert "demographic_parity_difference" in results
        assert "equalized_odds_difference" in results
        assert "equal_opportunity_difference" in results

    def test_compute_all_metrics_with_ci(self, analyzer):
        """Test computing all metrics with CI."""
        results = analyzer.compute_all_metrics(include_ci=True, n_bootstrap=100, random_state=42)
        assert "demographic_parity_difference" in results
        # The CI metrics should be MetricResult
        dp = results["demographic_parity_difference"]
        assert isinstance(dp, MetricResult)


class TestFairnessAnalyzerReport:
    """Test report generation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with sample data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)
        return FairnessAnalyzer(y_true, y_pred, gender)

    def test_get_report_structure(self, analyzer):
        """Test report has expected structure."""
        report = analyzer.get_report(include_ci=False)
        assert "task_type" in report
        assert "metrics" in report
        assert "assessment" in report
        assert "data_info" in report

    def test_get_report_with_ci(self, analyzer):
        """Test report with CI included."""
        report = analyzer.get_report(include_ci=True, n_bootstrap=100, random_state=42)
        assert "metrics_with_ci" in report
        assert "effect_sizes" in report
        assert "statistical_validation" in report


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        result = MetricResult(
            metric_name="test_metric",
            value=0.15,
            confidence_interval=(0.10, 0.20),
            effect_size=0.3,
            effect_interpretation="small effect",
            group_sizes={"A": 100, "B": 100},
            is_fair=False,
            threshold=0.1,
        )
        assert result.metric_name == "test_metric"
        assert result.value == 0.15
        assert result.is_fair is False

    def test_metric_result_to_dict(self):
        """Test MetricResult.to_dict()."""
        result = MetricResult(
            metric_name="test",
            value=0.1,
            confidence_interval=(0.05, 0.15),
            effect_size=0.2,
            effect_interpretation="negligible",
            group_sizes={"A": 50},
            is_fair=True,
            threshold=0.1,
        )
        d = result.to_dict()
        assert d["metric_name"] == "test"
        assert d["value"] == 0.1
        assert d["confidence_interval"]["lower"] == 0.05
        assert d["confidence_interval"]["upper"] == 0.15


class TestFairnessAnalyzerRegression:
    """Test regression metrics in analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create regression analyzer."""
        np.random.seed(42)
        n = 200
        y_true = np.random.normal(100, 20, n)
        y_pred = y_true + np.random.normal(0, 10, n)
        gender = np.random.choice(["M", "F"], n)
        return FairnessAnalyzer(y_true, y_pred, gender, task_type="regression")

    def test_mae_parity(self, analyzer):
        """Test MAE parity difference."""
        result = analyzer.mae_parity_difference(include_ci=False)
        assert isinstance(result, float)

    def test_compute_all_regression_metrics(self, analyzer):
        """Test computing all regression metrics."""
        results = analyzer.compute_all_metrics(include_ci=False)
        assert "mae_parity_difference" in results
        assert "rmse_parity_difference" in results
        assert "mean_prediction_difference" in results
