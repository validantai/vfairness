"""Tests for regression fairness metrics."""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.regression import (
    get_group_metrics,
    mae_parity_difference,
    mean_prediction_difference,
    r2_parity_difference,
    residual_bias,
    rmse_parity_difference,
)


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n = 200

    group = np.array(["A"] * 100 + ["B"] * 100)

    # True values
    y_true = np.concatenate(
        [
            np.random.normal(50, 10, 100),  # Group A
            np.random.normal(50, 10, 100),  # Group B
        ]
    )

    # Perfect predictions (with small noise)
    y_pred = y_true + np.random.normal(0, 1, n)

    return y_true, y_pred, group


@pytest.fixture
def biased_regression_data():
    """Generate regression data with systematic bias."""
    np.random.seed(42)

    group = np.array(["A"] * 100 + ["B"] * 100)

    y_true = np.concatenate(
        [
            np.random.normal(50, 10, 100),
            np.random.normal(50, 10, 100),
        ]
    )

    # Group A: unbiased predictions
    # Group B: systematically underpredicted by 10
    y_pred = np.concatenate(
        [
            y_true[:100] + np.random.normal(0, 1, 100),
            y_true[100:] - 10 + np.random.normal(0, 1, 100),
        ]
    )

    return y_true, y_pred, group


class TestMAEParity:
    """Tests for MAE parity metrics."""

    def test_equal_mae(self, regression_data):
        """Test when groups have similar MAE."""
        y_true, y_pred, group = regression_data
        mae_diff = mae_parity_difference(y_true, y_pred, group)

        # Similar prediction quality should give small difference
        assert mae_diff < 1.0

    def test_unequal_mae(self, biased_regression_data):
        """Test when groups have different MAE."""
        y_true, y_pred, group = biased_regression_data
        mae_diff = mae_parity_difference(y_true, y_pred, group)

        # Group B has higher error due to bias
        assert mae_diff > 5.0


class TestRMSEParity:
    """Tests for RMSE parity metrics."""

    def test_equal_rmse(self, regression_data):
        """Test when groups have similar RMSE."""
        y_true, y_pred, group = regression_data
        rmse_diff = rmse_parity_difference(y_true, y_pred, group)

        assert rmse_diff < 1.0


class TestMeanPredictionDifference:
    """Tests for mean prediction difference."""

    def test_unbiased_predictions(self, regression_data):
        """Test with unbiased predictions."""
        y_true, y_pred, group = regression_data
        mpd = mean_prediction_difference(y_true, y_pred, group)

        # Both groups should have similar mean predictions
        assert mpd < 2.0

    def test_biased_predictions(self, biased_regression_data):
        """Test with biased predictions."""
        y_true, y_pred, group = biased_regression_data
        mpd = mean_prediction_difference(y_true, y_pred, group)

        # Group B is underpredicted by ~10
        assert mpd > 8.0


class TestR2Parity:
    """Tests for R² parity metrics."""

    def test_similar_r2(self, regression_data):
        """Test when groups have similar R²."""
        y_true, y_pred, group = regression_data
        r2_diff = r2_parity_difference(y_true, y_pred, group)

        # Both groups should have similar R²
        assert r2_diff < 0.1


class TestResidualBias:
    """Tests for residual bias computation."""

    def test_unbiased_residuals(self, regression_data):
        """Test with unbiased predictions."""
        y_true, y_pred, group = regression_data
        bias = residual_bias(y_true, y_pred, group)

        # Both groups should have near-zero mean residual
        assert abs(bias["A"]) < 1.0
        assert abs(bias["B"]) < 1.0

    def test_biased_residuals(self, biased_regression_data):
        """Test with biased predictions."""
        y_true, y_pred, group = biased_regression_data
        bias = residual_bias(y_true, y_pred, group)

        # Group A: unbiased
        assert abs(bias["A"]) < 1.0

        # Group B: underpredicted (positive residual)
        assert bias["B"] > 8.0


class TestGroupMetrics:
    """Tests for detailed group metrics."""

    def test_metric_keys(self, regression_data):
        """Test that all expected metrics are present."""
        y_true, y_pred, group = regression_data
        metrics = get_group_metrics(y_true, y_pred, group)

        expected_keys = [
            "size",
            "mean_true",
            "mean_pred",
            "mae",
            "rmse",
            "mean_residual",
            "std_residual",
            "r2",
        ]

        for key in expected_keys:
            assert key in metrics["A"]
            assert key in metrics["B"]

    def test_group_sizes(self, regression_data):
        """Test group sizes are correct."""
        y_true, y_pred, group = regression_data
        metrics = get_group_metrics(y_true, y_pred, group)

        assert metrics["A"]["size"] == 100
        assert metrics["B"]["size"] == 100
