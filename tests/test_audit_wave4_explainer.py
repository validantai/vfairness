"""Audit wave 4 pins: explainer band unification.

Three confirmed findings:
1. ``explainer._risk_band`` duplicated ``rendering.engine._risk_label`` with
   only a 'must mirror' docstring; the two had already drifted on NaN
   (MINIMAL vs N/A). Both now share ``vfairness._bands`` semantics, and the
   consistency test here fails loudly on any future drift.
2. ``_explain_drift_result`` used 0.25/0.5/0.7/0.85 severity bands while the
   SVG drift colouring uses 0.3/0.6, so text called amber-charted drift 'low'
   and red-charted drift 'medium'. Severity now follows the 0.3/0.6 bands.
3. The per-issue severity path in ``_explain_validation_result`` let
   Enum-valued ERROR issues fall through to the 'medium' default. ERROR now
   maps to 'high' and no raw enum repr leaks into the rendered text.
"""

from __future__ import annotations

import pytest

from vfairness import _bands, explainer
from vfairness.operations.cicd.validator import ValidationIssue, ValidationSeverity
from vfairness.rendering import engine
from vfairness.rendering.adapters_monitoring import _drift_color

NAN = float("nan")

# Boundary values plus epsilon neighbours around every risk threshold.
RISK_SCORES = [
    0.0,
    0.1,
    0.2499,
    0.25,
    0.2501,
    0.49,
    0.50,
    0.51,
    0.7499,
    0.75,
    0.7501,
    0.9,
    1.0,
    NAN,
]


# Finding 1: canonical risk bands


@pytest.mark.parametrize("score", RISK_SCORES)
def test_risk_band_matches_engine_risk_label(score):
    """The text band and the SVG badge band must never diverge."""
    assert explainer._risk_band(score) == engine._risk_label(score)


def test_risk_band_delegates_to_canonical_module():
    for score in RISK_SCORES:
        assert explainer._risk_band(score) == _bands.risk_band(score)


def test_risk_band_canonical_labels():
    assert _bands.risk_band(0.0) == "MINIMAL"
    assert _bands.risk_band(0.25) == "LOW"
    assert _bands.risk_band(0.50) == "MEDIUM"
    assert _bands.risk_band(0.75) == "HIGH"
    # NaN must not read as a pass (this is where the copies had drifted).
    assert _bands.risk_band(NAN) == "N/A"
    assert _bands.risk_band(None) == "N/A"


def test_band_severity_covers_every_band_label():
    """Every label risk_band can emit has a severity, so the dict lookup in
    _explain_bias_audit can never KeyError (NaN included)."""
    for score in RISK_SCORES:
        band = _bands.risk_band(score)
        assert band in _bands.BAND_SEVERITY
    assert _bands.BAND_SEVERITY["N/A"] == "info"


class _BiasReport:
    """Minimal duck-typed bias audit report."""

    def __init__(self, score):
        self.overall_risk_score = score
        self.metrics = {}
        self.recommendations = []


def test_bias_audit_nan_score_does_not_crash():
    rep = explainer._explain_bias_audit(_BiasReport(NAN))
    assert rep.severity == "info"
    assert "N/A" in str(rep)


# Finding 2: drift severity aligned with SVG drift colours (0.3 / 0.6)


class _DriftResult:
    def __init__(self, score, detected=True):
        self.overall_drift_score = score
        self.drift_detected = detected
        self.scales = {}
        self.worst_scale = None


_COLOR_TO_SEVERITIES = {
    "#dc2626": {"high"},  # red zone: >= 0.6
    "#f59e0b": {"medium"},  # amber zone: 0.3 to 0.6
    "#059669": {"low", "info"},  # green zone: < 0.3
}


@pytest.mark.parametrize("score", [0.0, 0.1, 0.29, 0.3, 0.31, 0.45, 0.59, 0.6, 0.61, 0.85, 1.0])
@pytest.mark.parametrize("detected", [True, False])
def test_drift_severity_agrees_with_svg_colour(score, detected):
    """Text severity zone must match the chart colour zone at every score."""
    sev = explainer._explain_drift_result(_DriftResult(score, detected)).severity
    assert sev in _COLOR_TO_SEVERITIES[_drift_color(score)]


def test_drift_severity_band_edges():
    assert explainer._explain_drift_result(_DriftResult(0.45)).severity == "medium"
    assert explainer._explain_drift_result(_DriftResult(0.65)).severity == "high"
    assert explainer._explain_drift_result(_DriftResult(0.1, detected=True)).severity == "low"
    assert explainer._explain_drift_result(_DriftResult(0.1, detected=False)).severity == "info"


def test_drift_interpretation_guide_quotes_aligned_thresholds():
    rep = explainer._explain_drift_result(_DriftResult(0.45))
    guide = rep.explanations[0].interpretation_guide
    assert "0.30" in guide and "0.60" in guide
    # The old divergent thresholds must be gone from the guide text.
    for stale in ("0.25", "0.70", "0.85"):
        assert stale not in guide


# Finding 3: Enum-valued issue severities in _explain_validation_result


class _ValidationResult:
    def __init__(self, issues):
        self.passed = False
        self.issues = issues


def _issue(sev, msg="problem detected"):
    return ValidationIssue(issue_type="label_quality", severity=sev, message=msg)


def test_enum_error_issue_labelled_high_not_medium():
    rep = explainer._explain_validation_result(
        _ValidationResult([_issue(ValidationSeverity.ERROR)])
    )
    issue_expl = rep.explanations[1]
    assert issue_expl.severity == "high"
    assert issue_expl.value == "error"


def test_enum_issue_severity_mapping():
    rep = explainer._explain_validation_result(
        _ValidationResult(
            [
                _issue(ValidationSeverity.CRITICAL),
                _issue(ValidationSeverity.WARNING),
                _issue(ValidationSeverity.INFO),
            ]
        )
    )
    severities = [e.severity for e in rep.explanations[1:]]
    assert severities == ["critical", "medium", "info"]


def test_no_raw_enum_repr_leaks_into_text():
    rep = explainer._explain_validation_result(
        _ValidationResult([_issue(ValidationSeverity.ERROR)])
    )
    assert "ValidationSeverity" not in str(rep)
    # The error issue also counts as an error in the summary line.
    assert "1 errors" in rep.summary


def test_plain_string_severity_still_works():
    """API compatibility: issues with plain string severities keep working."""

    class _PlainIssue:
        issue_type = "rep"
        severity = "HIGH"
        message = "plain string severity"
        recommendation = None

    rep = explainer._explain_validation_result(_ValidationResult([_PlainIssue()]))
    assert rep.explanations[1].severity == "high"
