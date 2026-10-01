"""Tests for classification fairness metrics."""

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    get_group_metrics,
    predictive_parity_difference,
)


@pytest.fixture
def classification_data():
    """Generate sample classification data with known fairness properties."""
    np.random.seed(42)

    # Create two groups with different positive rates
    group = np.array(["A"] * 100 + ["B"] * 100)

    # Group A: 60% positive rate, Group B: 40% positive rate
    y_true = np.array([1] * 60 + [0] * 40 + [1] * 40 + [0] * 60)

    # Predictions that mirror the true labels (perfect model)
    y_pred = y_true.copy()

    return y_true, y_pred, group


@pytest.fixture
def unfair_data():
    """Generate data with clear unfairness."""
    np.random.seed(42)

    group = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)

    # Group A: predict 80% positive, Group B: predict 20% positive
    y_pred = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)

    return y_true, y_pred, group


class TestDemographicParity:
    """Tests for demographic parity metrics."""

    def test_perfect_parity(self):
        """Test when groups have identical positive rates."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        # Both groups have 50% positive rate
        dp = demographic_parity_difference(y_true, y_pred, group, min_group_size=2)
        assert dp == pytest.approx(0.0, abs=1e-10)

    def test_maximum_disparity(self):
        """Test when one group has all positives, other has all negatives."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        dp = demographic_parity_difference(y_true, y_pred, group, min_group_size=2)
        assert dp == pytest.approx(1.0, abs=1e-10)

    def test_unfair_data(self, unfair_data):
        """Test with clearly unfair data."""
        y_true, y_pred, group = unfair_data
        dp = demographic_parity_difference(y_true, y_pred, group)

        # Difference should be 0.8 - 0.2 = 0.6
        assert dp == pytest.approx(0.6, abs=0.01)

    def test_ratio_80_percent_rule(self, unfair_data):
        """Test demographic parity ratio for 80% rule."""
        y_true, y_pred, group = unfair_data
        ratio = demographic_parity_ratio(y_true, y_pred, group)

        # Ratio should be 0.2/0.8 = 0.25 (fails 80% rule)
        assert ratio == pytest.approx(0.25, abs=0.01)
        assert ratio < 0.8  # Fails 80% rule


class TestEqualizedOdds:
    """Tests for equalized odds metrics."""

    def test_perfect_odds(self, classification_data):
        """Test with perfect model (equalized odds = 0)."""
        y_true, y_pred, group = classification_data
        eo = equalized_odds_difference(y_true, y_pred, group)

        # Perfect model has TPR=1, FPR=0 for both groups
        assert eo == pytest.approx(0.0, abs=1e-10)

    def test_equal_opportunity(self, classification_data):
        """Test equal opportunity with perfect model."""
        y_true, y_pred, group = classification_data
        eop = equal_opportunity_difference(y_true, y_pred, group)

        assert eop == pytest.approx(0.0, abs=1e-10)


class TestPredictiveParity:
    """Tests for predictive parity metrics."""

    def test_perfect_precision(self, classification_data):
        """Test with perfect model (predictive parity = 0)."""
        y_true, y_pred, group = classification_data
        pp = predictive_parity_difference(y_true, y_pred, group)

        # Perfect model has precision=1 for both groups
        assert pp == pytest.approx(0.0, abs=1e-10)


class TestGroupMetrics:
    """Tests for detailed group metrics."""

    def test_group_stats(self, classification_data):
        """Test that group metrics are computed correctly."""
        y_true, y_pred, group = classification_data
        metrics = get_group_metrics(y_true, y_pred, group)

        assert "A" in metrics
        assert "B" in metrics

        # Check metric keys
        expected_keys = [
            "size",
            "positive_rate",
            "base_rate",
            "tpr",
            "fpr",
            "precision",
            "accuracy",
        ]
        for key in expected_keys:
            assert key in metrics["A"]
            assert key in metrics["B"]

    def test_group_sizes(self, classification_data):
        """Test that group sizes are correct."""
        y_true, y_pred, group = classification_data
        metrics = get_group_metrics(y_true, y_pred, group)

        assert metrics["A"]["size"] == 100
        assert metrics["B"]["size"] == 100


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pandas_input(self):
        """Test with pandas Series input."""
        y_true = pd.Series([1, 0, 1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0, 1, 0])
        group = pd.Series(["A", "A", "A", "B", "B", "B"])

        dp = demographic_parity_difference(y_true, y_pred, group, min_group_size=2)
        assert isinstance(dp, float)

    def test_list_input(self):
        """Test with list input."""
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 0]
        group = ["A", "A", "A", "B", "B", "B"]

        dp = demographic_parity_difference(y_true, y_pred, group, min_group_size=2)
        assert isinstance(dp, float)

    def test_min_group_size(self):
        """Test minimum group size enforcement."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        group = np.array(["A", "A", "A", "A", "B"])  # B has only 1 sample

        # With min_group_size=30 (default) NO group qualifies (A has 4, B has 1),
        # so no between-group comparison happens at all. That is insufficient
        # evidence, reported as NaN. This assertion used to pin 0.0, which is the
        # value of PERFECT demographic parity and certified fairness for a
        # comparison that never ran.
        dp = demographic_parity_difference(y_true, y_pred, group)
        assert np.isnan(dp)

    def test_intersectional_groups(self):
        """Test with intersectional (multi-attribute) groups."""
        n = 120
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)

        # Create DataFrame with two sensitive attributes
        sensitive = pd.DataFrame({"gender": ["M"] * 60 + ["F"] * 60, "race": ["A", "B"] * 60})

        dp = demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=20)
        assert isinstance(dp, float)
        assert 0 <= dp <= 1
