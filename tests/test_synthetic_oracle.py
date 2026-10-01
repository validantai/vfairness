"""Synthetic oracle tests for core fairness metrics (VB-TEST-1).

These tests build small datasets whose fairness metrics have a known,
closed-form analytical answer, then assert that the library reproduces
that answer within a tight numerical tolerance. Because the expected
value is computed by hand (not by re-running the library), a regression
in a metric formula is caught even if the code still "runs".

All randomness is seeded. Every group is constructed with at least
``min_group_size`` (default 30) samples so no group is silently dropped.

Property families covered:
    1. Demographic parity difference (max minus min selection rate).
    2. Disparate impact ratio (min over max selection rate).
    3. Base rate difference (max minus min positive-label rate per group),
       read back from ``impossibility_diagnostics``.
    4. Expected calibration error and the equal-opportunity (TPR) gap.
"""

import numpy as np
import pytest

import vfairness as v

TOL = 1e-9


def _labels(n_pos, n_neg):
    """Build a 0/1 array with ``n_pos`` ones followed by ``n_neg`` zeros."""
    return np.array([1] * n_pos + [0] * n_neg, dtype=int)


# ---------------------------------------------------------------------------
# 1. Demographic parity difference: known selection rates per group.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rate_a, rate_b, n, expected",
    [
        (0.60, 0.20, 100, 0.40),
        (0.50, 0.50, 60, 0.00),
        (0.90, 0.30, 100, 0.60),
        (0.75, 0.25, 80, 0.50),
    ],
)
def test_demographic_parity_difference_oracle(rate_a, rate_b, n, expected):
    pos_a = round(rate_a * n)
    pos_b = round(rate_b * n)
    y_pred = np.concatenate([_labels(pos_a, n - pos_a), _labels(pos_b, n - pos_b)])
    y_true = np.zeros(2 * n, dtype=int)
    groups = np.array(["A"] * n + ["B"] * n)

    got = v.demographic_parity_difference(y_true, y_pred, groups)
    assert got == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# 2. Disparate impact ratio: known min-over-max of selection rates.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rate_a, rate_b, n, expected",
    [
        (0.60, 0.20, 100, 0.20 / 0.60),
        (0.50, 0.40, 100, 0.40 / 0.50),
        (0.80, 0.80, 50, 1.00),
        (0.90, 0.45, 100, 0.45 / 0.90),
    ],
)
def test_disparate_impact_ratio_oracle(rate_a, rate_b, n, expected):
    pos_a = round(rate_a * n)
    pos_b = round(rate_b * n)
    y_pred = np.concatenate([_labels(pos_a, n - pos_a), _labels(pos_b, n - pos_b)])
    y_true = np.zeros(2 * n, dtype=int)
    groups = np.array(["A"] * n + ["B"] * n)

    got = v.demographic_parity_ratio(y_true, y_pred, groups)
    assert got == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# 3. Base rate difference: known positive-label rate per group.
#    Read back from impossibility_diagnostics["base_rate_disparity"], which
#    the library documents as max minus min group base rate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "base_a, base_b, n, expected",
    [
        (0.50, 0.25, 200, 0.25),
        (0.40, 0.40, 100, 0.00),
        (0.80, 0.20, 100, 0.60),
        (0.30, 0.10, 100, 0.20),
    ],
)
def test_base_rate_difference_oracle(base_a, base_b, n, expected):
    pos_a = round(base_a * n)
    pos_b = round(base_b * n)
    y_true = np.concatenate([_labels(pos_a, n - pos_a), _labels(pos_b, n - pos_b)])
    # Probabilities are irrelevant to base rates; keep them valid and simple.
    y_prob = np.where(y_true == 1, 0.7, 0.3)
    groups = np.array(["A"] * n + ["B"] * n)

    diag = v.impossibility_diagnostics(y_true, y_prob, groups)
    assert diag["base_rate_disparity"] == pytest.approx(expected, abs=TOL)
    assert diag["base_rates"]["A"] == pytest.approx(base_a, abs=TOL)
    assert diag["base_rates"]["B"] == pytest.approx(base_b, abs=TOL)


# ---------------------------------------------------------------------------
# 4a. Expected calibration error: known closed-form ECE.
#     With probabilities placed exactly at the true 0/1 outcome the model is
#     perfectly calibrated (ECE == 0). Shifting every probability by a
#     constant delta toward 0.5 makes the per-bin confidence miss the
#     empirical accuracy by exactly delta, so ECE == delta.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("delta", [0.0, 0.1, 0.2, 0.25])
def test_expected_calibration_error_oracle(delta):
    n = 200
    y_true = _labels(n // 2, n // 2)
    # positives get prob (1 - delta), negatives get prob delta.
    y_prob = np.where(y_true == 1, 1.0 - delta, delta)

    result = v.expected_calibration_error(y_true, y_prob)
    assert result.overall_value == pytest.approx(delta, abs=1e-9)


# ---------------------------------------------------------------------------
# 4b. Equal opportunity difference: known true positive rates per group.
#     Among the positive-label samples of each group we set a controlled
#     fraction to y_pred == 1, giving a known TPR gap.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tpr_a, tpr_b, n_pos, expected",
    [
        (1.00, 0.50, 40, 0.50),
        (0.75, 0.75, 40, 0.00),
        (0.90, 0.30, 50, 0.60),
        (0.80, 0.60, 50, 0.20),
    ],
)
def test_equal_opportunity_difference_oracle(tpr_a, tpr_b, n_pos, expected):
    # Each group: n_pos positive-label samples plus n_neg negatives (all
    # negatives predicted 0 so they do not affect TPR). n_neg keeps group
    # size above the default minimum.
    n_neg = 30
    tp_a = round(tpr_a * n_pos)
    tp_b = round(tpr_b * n_pos)

    y_true = np.concatenate(
        [
            _labels(n_pos, n_neg),  # group A: positives then negatives
            _labels(n_pos, n_neg),  # group B
        ]
    )
    y_pred = np.concatenate(
        [
            np.array([1] * tp_a + [0] * (n_pos - tp_a) + [0] * n_neg, dtype=int),
            np.array([1] * tp_b + [0] * (n_pos - tp_b) + [0] * n_neg, dtype=int),
        ]
    )
    groups = np.array(["A"] * (n_pos + n_neg) + ["B"] * (n_pos + n_neg))

    got = v.equal_opportunity_difference(y_true, y_pred, groups)
    assert got == pytest.approx(expected, abs=TOL)
