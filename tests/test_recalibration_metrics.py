"""Tests for the spec-v2 recalibration + subgroup-calibration metrics:
calibration_in_the_large, calibration_slope, integrated_calibration_index,
multicalibration. These back the EU-healthcare profile (which seals on group
multicalibration / ICI rather than bare bin-ECE)."""

import numpy as np
import pytest

from vfairness import (
    calibration_in_the_large,
    calibration_slope,
    integrated_calibration_index,
    multicalibration,
)


def _well_calibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    grp = rng.choice(["A", "B"], n)
    return y, p, grp


def test_well_calibrated_slope_near_one_citl_near_zero():
    y, p, grp = _well_calibrated()
    assert calibration_slope(y, p).overall_value == pytest.approx(1.0, abs=0.12)
    assert abs(calibration_in_the_large(y, p).metadata["signed"]) < 0.08
    assert integrated_calibration_index(y, p).overall_ici < 0.05


def test_overconfident_predictions_have_slope_below_one():
    """Stretching probabilities toward 0/1 (over-confidence) must drop the slope."""
    y, p, _ = _well_calibrated()
    p_extreme = np.clip((p - 0.5) * 1.8 + 0.5, 1e-4, 1 - 1e-4)
    slope = calibration_slope(y, p_extreme).overall_value
    assert slope < 0.6
    # ICI rises because the smoothed observed rate no longer tracks the prediction
    assert integrated_calibration_index(y, p_extreme).overall_ici > 0.05


def test_group_bias_shows_ici_disparity_and_multicalibration_violation():
    y, p, grp = _well_calibrated()
    p_biased = p.copy()
    p_biased[grp == "B"] = np.clip(p_biased[grp == "B"] + 0.2, 0.0, 1.0)
    ici = integrated_calibration_index(y, p_biased, grp)
    assert ici.ici_disparity > 0.08
    assert ici.most_miscalibrated_group == "B"
    mc = multicalibration(y, p_biased, grp)
    assert mc.alpha > 0.1
    assert mc.worst_group == "B"
    assert not mc.is_multicalibrated


def test_calibration_slope_reports_target_band():
    y, p, _ = _well_calibrated()
    res = calibration_slope(y, p)
    assert res.metadata["well_calibrated_band"] == [0.8, 1.2]
    assert res.metric_name == "calibration_slope"


def test_citl_absolute_value_gates_well_calibrated():
    y, p, _ = _well_calibrated()
    res = calibration_in_the_large(y, p)
    # overall_value is |CITL| so the built-in 0.05 threshold check is meaningful
    assert res.overall_value >= 0.0
    assert res.is_well_calibrated == (res.overall_value < 0.05)


def test_constant_probability_is_stable():
    """A degenerate constant score must not raise. What is MEASURABLE from it
    stays finite; the slope, which is not, now says so.

    Re-baselined 2026-09-17 (Beta Go-Live stage 2). The old line asserted
    ``np.isfinite(calibration_slope(y, p).overall_value)`` and the docstring
    read "must not raise or return NaN". That expectation was wrong: with a
    constant score, logit(p) is collinear with the intercept, so the slope is
    not identifiable from this data at all, and the finite number the old code
    returned for it was 0.0. A slope of 0.0 is a real and alarming reading
    (the predictions carry no signal), so returning it here invented a finding
    that the data cannot support. The subject of the test is unchanged: the
    degenerate input must be handled gracefully rather than blow up, and the
    two quantities that ARE defined on it must still be measured.
    """
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 10)
    p = np.full(len(y), 0.5)
    # Measurable: mean(p) - mean(y) = 0.5 - 0.5, and the flat calibration curve
    # sits exactly on the observed rate.
    assert np.isfinite(calibration_in_the_large(y, p).overall_value)
    assert np.isfinite(integrated_calibration_index(y, p).overall_ici)

    # Not measurable, and it must be a could-not-check rather than a 0.0.
    with pytest.warns(UserWarning, match="could not be measured"):
        slope = calibration_slope(y, p)
    assert np.isnan(slope.overall_value)
    assert slope.is_well_calibrated is None, (
        "a slope that could not be measured must not be graded True or False"
    )

    # CONTROL: on a score column that does vary, the slope is still measured.
    y_ok, p_ok, _ = _well_calibrated(n=2000, seed=5)
    measured = calibration_slope(y_ok, p_ok)
    assert np.isfinite(measured.overall_value)
    assert measured.overall_value > 0.0


def test_multicalibration_requires_two_groups_gracefully():
    y, p, _ = _well_calibrated(n=200)
    grp = np.array(["A"] * len(y))  # single group
    mc = multicalibration(y, p, grp)
    assert mc.n_groups == 1
    assert np.isfinite(mc.alpha)
