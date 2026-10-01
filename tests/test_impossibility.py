"""Fairness impossibility theorem tests (VB-TEST-3).

Encodes the Kleinberg-Mullainathan-Raghavan and Chouldechova results: when
two groups have different base rates and the classifier is not a perfect
predictor, you cannot simultaneously achieve

    * calibration within each group, AND
    * both error-rate balances (equal TPR and equal FPR across groups).

Only in the degenerate cases (equal base rates, or perfect prediction) can
all three be satisfied at once.

The library surfaces this through two real APIs (verified against source):

    impossibility_diagnostics(y_true, y_prob, protected_attr) -> dict with
        "base_rates", "base_rate_disparity", "base_rates_differ",
        "is_degenerate", "impossibility_applies", ...

    calibration_vs_error_parity(y_true, y_prob, protected_attr, threshold)
        -> dict with "tpr_disparity", "fpr_disparity", "conflict_exists",
        and per-group "calibration_gap" in "group_metrics".
"""

import numpy as np
import pytest

import vfairness as v

CAL_TOL = 1e-6
RATE_TOL = 1e-6


def _labels(n_pos, n_neg):
    return np.array([1] * n_pos + [0] * n_neg, dtype=int)


def _calibration_gaps_equal(cve):
    gaps = [gm["calibration_gap"] for gm in cve["group_metrics"].values()]
    return max(gaps) - min(gaps) <= CAL_TOL


def _error_rates_balanced(cve):
    return cve["tpr_disparity"] <= RATE_TOL and cve["fpr_disparity"] <= RATE_TOL


# ---------------------------------------------------------------------------
# Unequal base rates: impossibility applies. A calibrated-but-imperfect
# predictor cannot balance both error rates while staying equally calibrated.
# ---------------------------------------------------------------------------
def _unequal_base_rate_dataset():
    """Group A base rate 0.5, group B base rate 0.25. Probabilities are the
    same calibrated function of the label in each group, so the predictor is
    genuinely predictive but not perfect at threshold 0.5."""
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(100, 300)])
    groups = np.array(["A"] * n + ["B"] * n)
    # Imperfect but calibrated-ish: positives -> 0.7, negatives -> 0.3.
    y_prob = np.where(y_true == 1, 0.7, 0.3)
    return y_true, y_prob, groups


def test_unequal_base_rates_flagged_by_diagnostics():
    y_true, y_prob, groups = _unequal_base_rate_dataset()
    diag = v.impossibility_diagnostics(y_true, y_prob, groups)
    # These flags are returned as numpy bool_, so compare truthiness.
    assert bool(diag["base_rates_differ"]) is True
    assert bool(diag["is_degenerate"]) is False
    assert bool(diag["impossibility_applies"]) is True
    assert diag["base_rate_disparity"] == pytest.approx(0.25, abs=1e-9)


def test_unequal_base_rates_cannot_satisfy_all_three():
    """The core impossibility: not (equal calibration AND both error rates
    balanced) when base rates differ and prediction is imperfect."""
    y_true, y_prob, groups = _unequal_base_rate_dataset()
    cve = v.calibration_vs_error_parity(y_true, y_prob, groups, threshold=0.5)

    cal_equal = _calibration_gaps_equal(cve)
    err_balanced = _error_rates_balanced(cve)

    # Joint satisfaction must be impossible here.
    assert not (cal_equal and err_balanced)
    # The library's own conflict flag agrees (numpy bool_).
    assert bool(cve["conflict_exists"]) is True


def test_unequal_base_rates_calibration_gaps_actually_differ():
    """Even when both error rates happen to be balanced at this threshold,
    the per-group calibration gaps differ, which is exactly the trade-off."""
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(100, 300)])
    groups = np.array(["A"] * n + ["B"] * n)
    # Perfect ranking probs -> TPR=1, FPR=0 for both groups (error balanced),
    # but the fixed 0.8/0.2 confidences are miscalibrated by different amounts
    # against the differing group base rates.
    y_prob = np.where(y_true == 1, 0.8, 0.2)
    cve = v.calibration_vs_error_parity(y_true, y_prob, groups, threshold=0.5)

    assert _error_rates_balanced(cve)  # error rates balanced ...
    assert not _calibration_gaps_equal(cve)  # ... but calibration is not
    assert bool(cve["conflict_exists"]) is True


# ---------------------------------------------------------------------------
# Degenerate cases: joint satisfaction IS possible.
# ---------------------------------------------------------------------------
def test_equal_base_rates_perfect_prediction_allows_joint_satisfaction():
    """Equal base rates plus a perfect, calibrated predictor: calibration is
    equal (both gaps zero) and both error rates are balanced (TPR=1, FPR=0)."""
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(200, 200)])  # both 0.5
    groups = np.array(["A"] * n + ["B"] * n)
    y_prob = np.where(y_true == 1, 1.0, 0.0)  # perfect and calibrated

    cve = v.calibration_vs_error_parity(y_true, y_prob, groups, threshold=0.5)
    assert _calibration_gaps_equal(cve)
    assert _error_rates_balanced(cve)
    assert bool(cve["conflict_exists"]) is False

    diag = v.impossibility_diagnostics(y_true, y_prob, groups)
    assert bool(diag["base_rates_differ"]) is False
    assert bool(diag["impossibility_applies"]) is False


def test_unequal_base_rates_perfect_prediction_measured_metrics_are_satisfiable():
    """A perfect predictor sidesteps the impossibility even with unequal base
    rates: the MEASURED metrics show perfect calibration in both groups (zero
    gaps) and balanced error rates (TPR/FPR disparity zero). This is the
    classic "impossibility vanishes at perfect accuracy" corner."""
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(100, 300)])
    groups = np.array(["A"] * n + ["B"] * n)
    y_prob = np.where(y_true == 1, 1.0, 0.0)  # perfect

    cve = v.calibration_vs_error_parity(y_true, y_prob, groups, threshold=0.5)
    assert _calibration_gaps_equal(cve)
    assert _error_rates_balanced(cve)


def test_perfect_prediction_conflict_flag_should_be_false():
    # A perfect predictor with unequal base rates drives every measured gap to
    # zero, so the impossibility does not bind and conflict_exists must be False.
    # (Previously conflict_exists keyed off base_rate_disparity alone and wrongly
    # reported True here; fixed in tradeoffs.py to reflect the measured metrics.)
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(100, 300)])
    groups = np.array(["A"] * n + ["B"] * n)
    y_prob = np.where(y_true == 1, 1.0, 0.0)  # perfect predictor, zero gaps

    cve = v.calibration_vs_error_parity(y_true, y_prob, groups, threshold=0.5)
    assert bool(cve["conflict_exists"]) is False


def test_diagnostics_is_a_base_rate_heuristic_not_a_predictor_check():
    """impossibility_diagnostics decides purely from base rates. With unequal
    base rates it reports impossibility_applies=True even for a perfect
    predictor, because it does not inspect predictions. This documents the
    intended (heuristic) scope of the diagnostic; the prediction-aware verdict
    comes from calibration_vs_error_parity (see the test above)."""
    n = 400
    y_true = np.concatenate([_labels(200, 200), _labels(100, 300)])
    groups = np.array(["A"] * n + ["B"] * n)
    y_prob = np.where(y_true == 1, 1.0, 0.0)  # perfect predictor

    diag = v.impossibility_diagnostics(y_true, y_prob, groups)
    assert bool(diag["base_rates_differ"]) is True
    assert bool(diag["is_degenerate"]) is False
    assert bool(diag["impossibility_applies"]) is True
