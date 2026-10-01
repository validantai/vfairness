"""Release-pass pin: the `cali-BRATIO-n` substring bug recurring in the renderers.

Third-iteration audit (CRITICAL, xai-render-vision-legal): `is_ratio = "ratio" in
key` treats a ratio metric as higher-is-better (1.0 = fair) and a difference metric
as lower-is-better (0 = fair). The substring "ratio" is contained in
"cali[bratio]n_difference", so calibration_difference was classified as a ratio and
its PASS/FAIL verdict was inverted (and it was dropped from the effect-size chart).
This is the same class as the sealed-verdict incident recorded in CLAUDE.md. The fix
keys on the `_ratio` suffix, which only real ratio metrics carry.
"""

from vfairness.rendering import adapters_fairness as A


def test_calibration_difference_is_not_treated_as_a_ratio():
    # The exact recurrence: "ratio" is a substring of "calibration_difference".
    assert "ratio" in "calibration_difference"  # the trap that fooled the old code
    assert not "calibration_difference".endswith("_ratio")


def test_ratio_metrics_still_recognized():
    for key in ("demographic_parity_ratio", "disparate_impact_ratio", "treatment_equality_ratio"):
        assert key.endswith("_ratio")


def test_metrics_bar_chart_passes_a_well_calibrated_model():
    # A small calibration difference (0.02, well within the 0.05 threshold) must
    # render as PASS; a large one (0.20) as FAIL. Under the old substring bug the
    # metric was read as a higher-is-better ratio, flipping the verdict.
    good = A.metrics_bar_chart_to_svg(
        {
            "metrics": {"calibration_difference": 0.02},
            "thresholds_used": {"calibration_difference": 0.05},
        }
    )
    bad = A.metrics_bar_chart_to_svg(
        {
            "metrics": {"calibration_difference": 0.20},
            "thresholds_used": {"calibration_difference": 0.05},
        }
    )
    # The good value renders the pass mark and NOT the fail mark; the bad value the reverse.
    assert "✓" in good and "FAIL" not in good  # check-mark, no FAIL
    assert "FAIL" in bad and "✗" in bad  # cross-mark + FAIL


def test_confidence_intervals_flags_a_severe_calibration_difference():
    # confidence_intervals_to_svg (a public export, adapters_fairness) is the
    # cali-BRATIO-n site the audit-3 pins missed. A calibration_difference CI far
    # from 0 ([0.85, 0.95]) must count as SIGNIFICANT (the difference branch: the
    # CI excludes 0). Under the old substring bug it took the ratio branch
    # (0.8 <= CI <= 1.2 = "fair") and read as NOT significant.
    svg = A.confidence_intervals_to_svg(
        {
            "metrics": {"calibration_difference": 0.90},
            "confidence_intervals": {"calibration_difference": {"lower": 0.85, "upper": 0.95}},
        }
    )
    assert "1 significant" in svg
    # Does-not-overcorrect: a genuine ratio metric whose CI sits in the fair band
    # [0.8, 1.2] is NOT flagged significant (its own ratio branch is still honored).
    svg_ratio = A.confidence_intervals_to_svg(
        {
            "metrics": {"disparate_impact_ratio": 0.95},
            "confidence_intervals": {"disparate_impact_ratio": {"lower": 0.9, "upper": 1.0}},
        }
    )
    assert "1 significant" not in svg_ratio


def test_effect_sizes_chart_includes_calibration_difference():
    # calibration_difference is a difference metric and must appear on the
    # effect-size chart; the old `"ratio" in key` skip dropped it silently.
    report = {"metrics": {"calibration_difference": 0.12, "demographic_parity_difference": 0.2}}
    svg = A.effect_sizes_to_svg(report)
    # Its label ("Calibration") must be present, i.e. the metric was not skipped.
    assert "Calib" in svg or "calibration" in svg.lower()
