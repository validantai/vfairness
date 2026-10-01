"""Canonical score-band definitions shared by text and visual layers.

The explainer (text) and the rendering engine (SVG badges / drift charts)
must always describe the same score with the same band, otherwise a report
contradicts its own chart. This module is the single source of truth for
those thresholds; ``explainer._risk_band`` delegates here, and the test
suite asserts that ``rendering.engine._risk_label`` stays in lockstep.
"""

# Risk bands: 0.25 / 0.50 / 0.75 (mirrors rendering.engine._risk_label).
RISK_BAND_THRESHOLDS = (0.25, 0.50, 0.75)

# Band label -> explainer severity vocabulary. 'N/A' maps to 'info' because
# an uncomputable score must not raise alarms, but also must not read as a
# pass (the band label itself already says N/A).
BAND_SEVERITY = {
    "MINIMAL": "info",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "N/A": "info",
}

# Drift bands: 0.3 / 0.6 (mirrors rendering.adapters_monitoring._drift_color:
# green below 0.3, amber to 0.6, red above).
DRIFT_BAND_THRESHOLDS = (0.3, 0.6)

# Calibration ECE bands: 0.05 / 0.10 / 0.15. These mirror the interpretation
# guide prose in explainer._explain_calibration ('< 0.05 well calibrated,
# 0.05-0.10 moderate, 0.10-0.15 poor, 0.15+ severe'); severity and prose must
# come from the same boundaries or the ECE card contradicts itself.
CALIBRATION_ECE_THRESHOLDS = (0.05, 0.10, 0.15)


def _is_nan(score: object) -> bool:
    """True when the score is not a usable number (None or NaN)."""
    try:
        return score is None or score != score
    except Exception:
        return True


def risk_band(score: float) -> str:
    """Map a 0-1 risk score to the band label the SVG risk badge shows.

    Same thresholds and labels as ``rendering.engine._risk_label``,
    including the NaN guard: an uncomputable score must not read as
    MINIMAL (i.e. as a pass).
    """
    if _is_nan(score):
        return "N/A"
    if score >= RISK_BAND_THRESHOLDS[2]:
        return "HIGH"
    if score >= RISK_BAND_THRESHOLDS[1]:
        return "MEDIUM"
    if score >= RISK_BAND_THRESHOLDS[0]:
        return "LOW"
    return "MINIMAL"


def risk_band_severity(score: float) -> str:
    """Explainer severity for a risk score, derived from its band."""
    return BAND_SEVERITY[risk_band(score)]


def drift_severity(score: float, detected: bool = True) -> str:
    """Explainer severity for a drift score, aligned with the SVG colours.

    >= 0.6 is 'high' (chart shows red), >= 0.3 is 'medium' (amber); below
    0.3 the chart is green, so the severity is 'low' when drift was still
    flagged and 'info' otherwise.
    """
    if _is_nan(score):
        return "info"
    if score >= DRIFT_BAND_THRESHOLDS[1]:
        return "high"
    if score >= DRIFT_BAND_THRESHOLDS[0]:
        return "medium"
    return "low" if detected else "info"


def calibration_ece_severity(ece: float) -> str:
    """Explainer severity for an ECE value, aligned with the guide prose.

    Half-open bands matching the evaluation text ('well calibrated' only
    below 0.05, 'moderate' below 0.10), so text and severity can never
    disagree at a boundary. An uncomputable ECE is 'info', not a pass.
    """
    if _is_nan(ece):
        return "info"
    if ece >= CALIBRATION_ECE_THRESHOLDS[2]:
        return "high"
    if ece >= CALIBRATION_ECE_THRESHOLDS[1]:
        return "medium"
    if ece >= CALIBRATION_ECE_THRESHOLDS[0]:
        return "low"
    return "info"
