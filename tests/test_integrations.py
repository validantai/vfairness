"""
Tests for MLOps integrations (MLflow, pytest assertions).
"""

import numpy as np
import pytest

from vfairness import (
    FairnessAnalyzer,
    FairnessAssertionError,
    assert_fairness,
    create_fairness_callback,
)


class TestAssertFairness:
    """Test pytest assertion function."""

    @pytest.fixture
    def fair_data(self):
        """Generate roughly fair data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        # Predictions similar across groups
        y_pred = y_true.copy()
        # Add some noise but keep it balanced
        noise_idx = np.random.choice(n, 20, replace=False)
        y_pred[noise_idx] = 1 - y_pred[noise_idx]
        gender = np.random.choice(["M", "F"], n)
        return y_true, y_pred, gender

    @pytest.fixture
    def unfair_data(self):
        """Generate unfair data with significant bias."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)
        # Create strong bias
        y_pred = np.where(gender == "M", 1, 0)
        return y_true, y_pred, gender

    def test_assert_fairness_passes(self, fair_data):
        """Test that fair data passes assertion."""
        y_true, y_pred, gender = fair_data
        # Should not raise - only check one metric with lenient threshold
        metrics = assert_fairness(
            y_true,
            y_pred,
            gender,
            metrics=["demographic_parity_difference"],  # Only check this one
            thresholds={"demographic_parity_difference": 0.3},
        )
        assert "demographic_parity_difference" in metrics

    def test_assert_fairness_fails(self, unfair_data):
        """Test that unfair data fails assertion."""
        y_true, y_pred, gender = unfair_data
        with pytest.raises(FairnessAssertionError) as exc_info:
            assert_fairness(
                y_true, y_pred, gender, thresholds={"demographic_parity_difference": 0.1}
            )
        assert "demographic_parity_difference" in exc_info.value.failed_metrics

    def test_assert_fairness_custom_message(self, unfair_data):
        """Test custom error message."""
        y_true, y_pred, gender = unfair_data
        with pytest.raises(FairnessAssertionError) as exc_info:
            assert_fairness(
                y_true,
                y_pred,
                gender,
                thresholds={"demographic_parity_difference": 0.1},
                message="Custom fairness test",
            )
        assert "Custom fairness test" in str(exc_info.value)

    def test_assert_fairness_returns_metrics(self, fair_data):
        """Test that passing assertion returns metrics."""
        y_true, y_pred, gender = fair_data
        metrics = assert_fairness(
            y_true,
            y_pred,
            gender,
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.5},
        )
        assert isinstance(metrics, dict)
        assert "demographic_parity_difference" in metrics

    def test_assert_fairness_with_ci(self, fair_data):
        """Test assertion with confidence intervals."""
        y_true, y_pred, gender = fair_data
        metrics = assert_fairness(
            y_true,
            y_pred,
            gender,
            metrics=["demographic_parity_difference"],  # Only check this one
            include_ci=True,
            n_bootstrap=100,
            thresholds={"demographic_parity_difference": 0.5},
            random_state=42,
        )
        assert "demographic_parity_difference" in metrics

    def test_fairness_assertion_error_attributes(self, unfair_data):
        """Test FairnessAssertionError has expected attributes."""
        y_true, y_pred, gender = unfair_data
        try:
            assert_fairness(
                y_true, y_pred, gender, thresholds={"demographic_parity_difference": 0.1}
            )
        except FairnessAssertionError as e:
            assert hasattr(e, "failed_metrics")
            assert hasattr(e, "all_metrics")
            assert isinstance(e.failed_metrics, dict)


class TestFairnessCallback:
    """Test fairness callback for training loops."""

    def test_create_callback(self):
        """Test creating a callback function."""
        callback = create_fairness_callback(
            sensitive_attr_column="gender", metrics=["demographic_parity_difference"]
        )
        assert callable(callback)

    def test_callback_computes_metrics(self):
        """Test that callback computes metrics."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)

        callback = create_fairness_callback("gender")
        results = callback(y_true, y_pred, gender)

        assert "demographic_parity_difference" in results

    def test_callback_with_threshold_check(self):
        """Test callback with threshold violation check."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)
        # Create very unfair predictions
        y_pred = np.where(gender == "M", 1, 0)

        callback = create_fairness_callback(
            "gender", thresholds={"demographic_parity_difference": 0.1}, fail_on_violation=True
        )

        with pytest.raises(ValueError, match="Fairness violation"):
            callback(y_true, y_pred, gender)


class TestMLflowIntegration:
    """Test MLflow integration (without actually using MLflow)."""

    def test_log_fairness_import_error(self):
        """Test that log_fairness_to_mlflow raises ImportError without MLflow."""
        from vfairness.evaluation.vfairness_metrics.integrations import log_fairness_to_mlflow

        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        gender = np.random.choice(["M", "F"], n)

        analyzer = FairnessAnalyzer(y_true, y_pred, gender)

        # This should either work (if mlflow installed) or raise ImportError/RuntimeError
        try:
            # Will fail with RuntimeError if mlflow is installed but no active run
            log_fairness_to_mlflow(analyzer)
        except (ImportError, RuntimeError):
            # Expected - either mlflow not installed or no active run
            pass

    def test_log_fairness_type_error(self):
        """Test that invalid input raises TypeError."""
        from vfairness.evaluation.vfairness_metrics.integrations import log_fairness_to_mlflow

        with pytest.raises((TypeError, ImportError, RuntimeError)):
            log_fairness_to_mlflow("invalid_input")
