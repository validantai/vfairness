"""Exhaustive boundary tests for the canonical band thresholds (_bands).

Also the mutation-testing runner for VB-TEST-9: because these tests pin every
threshold and return value exactly, a mutation that shifts a boundary or a
label is killed. Kept dependency-free and fast so a mutation run over
`_bands.py` is quick.
"""

import pytest

from vfairness._bands import (
    calibration_ece_severity,
    drift_severity,
    risk_band,
    risk_band_severity,
)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "MINIMAL"),
        (0.249, "MINIMAL"),
        (0.25, "LOW"),
        (0.499, "LOW"),
        (0.50, "MEDIUM"),
        (0.749, "MEDIUM"),
        (0.75, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def test_risk_band_boundaries(score, expected):
    assert risk_band(score) == expected


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_risk_band_uncomputable_is_na(bad):
    assert risk_band(bad) == "N/A"


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "info"),  # MINIMAL -> info
        (0.25, "low"),  # LOW
        (0.50, "medium"),  # MEDIUM
        (0.75, "high"),  # HIGH
        (float("nan"), "info"),  # N/A -> info
    ],
)
def test_risk_band_severity(score, expected):
    assert risk_band_severity(score) == expected


@pytest.mark.parametrize(
    "score,detected,expected",
    [
        (0.6, True, "high"),
        (1.0, True, "high"),
        (0.59, True, "medium"),
        (0.3, True, "medium"),
        (0.29, True, "low"),
        (0.29, False, "info"),
        (0.0, True, "low"),
        (0.0, False, "info"),
    ],
)
def test_drift_severity(score, detected, expected):
    assert drift_severity(score, detected) == expected


def test_drift_severity_nan_is_info():
    assert drift_severity(float("nan")) == "info"
    assert drift_severity(None) == "info"


@pytest.mark.parametrize(
    "ece,expected",
    [
        (0.0, "info"),
        (0.049, "info"),
        (0.05, "low"),
        (0.099, "low"),
        (0.10, "medium"),
        (0.149, "medium"),
        (0.15, "high"),
        (0.9, "high"),
    ],
)
def test_calibration_ece_severity_boundaries(ece, expected):
    assert calibration_ece_severity(ece) == expected


def test_calibration_ece_nan_is_info():
    assert calibration_ece_severity(float("nan")) == "info"
    assert calibration_ece_severity(None) == "info"


def test_default_detected_is_true():
    # drift_severity(score) with detected defaulting True -> 'low' below 0.3,
    # whereas the same score with detected=False -> 'info'. This actually pins
    # the default to True, rather than only checking 0.1 maps to 'low'.
    assert drift_severity(0.1) == "low"
    assert drift_severity(0.1, detected=False) == "info"
