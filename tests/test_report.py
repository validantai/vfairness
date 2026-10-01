"""Tests for fairness report generation."""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)


@pytest.fixture
def classification_data():
    """Generate classification data for testing."""
    np.random.seed(42)
    n = 200

    group = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)

    return y_true, y_pred, group, y_prob


@pytest.fixture
def regression_data():
    """Generate regression data for testing."""
    np.random.seed(42)
    n = 200

    group = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.normal(50, 10, n)
    y_pred = y_true + np.random.normal(0, 5, n)

    return y_true, y_pred, group


class TestClassificationReport:
    """Tests for classification fairness report."""

    def test_report_structure(self, classification_data):
        """Test that report has expected structure."""
        y_true, y_pred, group, y_prob = classification_data
        report = classification_fairness_report(y_true, y_pred, group, y_prob)

        assert "task_type" in report
        assert report["task_type"] == "classification"

        assert "metrics" in report
        assert "group_stats" in report
        assert "assessment" in report
        assert "data_info" in report
        assert "thresholds_used" in report

    def test_metrics_computed(self, classification_data):
        """Test that all metrics are computed."""
        y_true, y_pred, group, y_prob = classification_data
        report = classification_fairness_report(y_true, y_pred, group, y_prob)

        expected_metrics = [
            "demographic_parity_difference",
            "demographic_parity_ratio",
            "equalized_odds_difference",
            "equal_opportunity_difference",
            "predictive_parity_difference",
            "calibration_difference",
        ]

        for metric in expected_metrics:
            assert metric in report["metrics"]

    def test_assessment_structure(self, classification_data):
        """Test assessment section structure."""
        y_true, y_pred, group, y_prob = classification_data
        report = classification_fairness_report(y_true, y_pred, group, y_prob)

        assessment = report["assessment"]
        assert "fairness_score" in assessment
        assert "passed_metrics" in assessment
        assert "failed_metrics" in assessment
        assert "summary" in assessment

        # Fairness score should be between 0 and 1
        assert 0 <= assessment["fairness_score"] <= 1

    def test_group_stats(self, classification_data):
        """Test group statistics are computed."""
        y_true, y_pred, group, _ = classification_data
        report = classification_fairness_report(y_true, y_pred, group)

        group_stats = report["group_stats"]
        assert "A" in group_stats
        assert "B" in group_stats

    def test_data_info(self, classification_data):
        """Test data info section."""
        y_true, y_pred, group, _ = classification_data
        report = classification_fairness_report(y_true, y_pred, group)

        data_info = report["data_info"]
        assert "original_size" in data_info
        assert "final_size" in data_info
        assert "n_groups" in data_info
        assert "valid_groups" in data_info

    def test_custom_thresholds(self, classification_data):
        """Test custom thresholds are applied."""
        y_true, y_pred, group, _ = classification_data

        custom_thresholds = {
            "demographic_parity_difference": 0.5,
            "equalized_odds_difference": 0.5,
        }

        report = classification_fairness_report(y_true, y_pred, group, thresholds=custom_thresholds)

        # Custom thresholds should be in the report
        assert report["thresholds_used"]["demographic_parity_difference"] == 0.5


class TestRegressionReport:
    """Tests for regression fairness report."""

    def test_report_structure(self, regression_data):
        """Test that report has expected structure."""
        y_true, y_pred, group = regression_data
        report = regression_fairness_report(y_true, y_pred, group)

        assert "task_type" in report
        assert report["task_type"] == "regression"

        assert "metrics" in report
        assert "residual_bias" in report
        assert "group_stats" in report
        assert "assessment" in report
        assert "data_info" in report

    def test_metrics_computed(self, regression_data):
        """Test that all regression metrics are computed."""
        y_true, y_pred, group = regression_data
        report = regression_fairness_report(y_true, y_pred, group)

        expected_metrics = [
            "mae_parity_difference",
            "rmse_parity_difference",
            "mean_prediction_difference",
            "r2_parity_difference",
        ]

        for metric in expected_metrics:
            assert metric in report["metrics"]

    def test_residual_bias(self, regression_data):
        """Test residual bias is computed per group."""
        y_true, y_pred, group = regression_data
        report = regression_fairness_report(y_true, y_pred, group)

        assert "A" in report["residual_bias"]
        assert "B" in report["residual_bias"]

    def test_y_std_in_data_info(self, regression_data):
        """Test that y_std is included in data info."""
        y_true, y_pred, group = regression_data
        report = regression_fairness_report(y_true, y_pred, group)

        assert "y_std" in report["data_info"]
        assert report["data_info"]["y_std"] > 0


class TestTypedReportContract:
    """Locks the report's documented key structure (``report_types.py``).

    ``FairnessReport`` and friends are static-only ``TypedDict`` annotations: the
    report is the same plain, JSON-serialisable ``dict`` it always was. These tests
    assert the required keys the contract promises are actually produced, so the
    documented structure and the runtime output cannot drift apart.
    """

    def test_typed_dicts_are_public(self):
        """The report structure types are importable from the public package."""
        from vfairness.evaluation import (
            AssessmentReport,
            DataInfo,
            ExplanationsReport,
            FairnessReport,
            InsufficientEvidenceGroup,
            MetricStatusEntry,
        )

        # Same objects whether reached via the umbrella package or the submodule.
        from vfairness.evaluation.vfairness_metrics import (
            FairnessReport as FairnessReportSub,
        )

        assert FairnessReport is FairnessReportSub
        for td in (
            FairnessReport,
            AssessmentReport,
            DataInfo,
            ExplanationsReport,
            MetricStatusEntry,
            InsufficientEvidenceGroup,
        ):
            assert hasattr(td, "__required_keys__")  # it is a TypedDict

    def test_classification_report_conforms(self, classification_data):
        """A classification report carries every required key of the contract."""
        import json

        from vfairness.evaluation import AssessmentReport, DataInfo, FairnessReport

        y_true, y_pred, group, y_prob = classification_data
        report = classification_fairness_report(y_true, y_pred, group, y_prob)

        # Contract preserved: still a plain dict, still JSON-serialisable.
        assert type(report) is dict
        json.dumps(report)

        missing = set(FairnessReport.__required_keys__) - set(report.keys())
        assert not missing, f"report missing required keys: {missing}"

        a_missing = set(AssessmentReport.__required_keys__) - set(report["assessment"].keys())
        assert not a_missing, f"assessment missing required keys: {a_missing}"

        # DataInfo documents keys that are actually produced (y_std is regression-only).
        di_documented = set(DataInfo.__annotations__) - {"y_std"}
        di_missing = di_documented - set(report["data_info"].keys())
        assert not di_missing, f"data_info missing documented keys: {di_missing}"

    def test_regression_report_conforms(self, regression_data):
        """A regression report carries every required key, plus its residual_bias."""
        from vfairness.evaluation import DataInfo, FairnessReport

        y_true, y_pred, group = regression_data
        report = regression_fairness_report(y_true, y_pred, group)

        missing = set(FairnessReport.__required_keys__) - set(report.keys())
        assert not missing, f"report missing required keys: {missing}"
        assert "residual_bias" in report  # NotRequired, regression-only

        # Every documented DataInfo key, including regression-only y_std, is present.
        di_missing = set(DataInfo.__annotations__) - set(report["data_info"].keys())
        assert not di_missing, f"data_info missing documented keys: {di_missing}"
