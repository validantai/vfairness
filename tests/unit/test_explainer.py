"""Explainability test suite (Workstream B).

Covers the two fairness-result explanation modules that power the
vfairness_explain task and the platform Explain panel:
  - vfairness.evaluation.vfairness_metrics.explainer.FairExplAIner / MetricExplanation
  - vfairness.explainer.FairnessExplainer / ExplanationReport
"""

import pytest

from vfairness.evaluation.vfairness_metrics.explainer import (
    FairExplAIner,
    MetricExplanation,
)
from vfairness.explainer import ExplanationReport, FairnessExplainer

# The 11 result types the facade is documented to handle.
EXPECTED_RESULT_TYPES = {
    "BiasAuditReport",
    "FeatureAnalysisReport",
    "CalibrationReport",
    "ThresholdAnalysisReport",
    "ReweightingAnalysisReport",
    "FairnessTrainingReport",
    "MultiscaleDriftResult",
    "WindowMetrics",
    "DataValidationResult",
    "GateDecision",
    "ExperimentResult",
}

_METRIC_KEYS = {
    "metric_name",
    "definition",
    "interpretation_guide",
    "value",
    "evaluation",
    "benchmark_context",
    "recommendation",
    "severity",
    "related_metrics",
}
_REPORT_KEYS = {"title", "summary", "severity", "recommendations", "explanations"}
_SEV_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


# ---------------------------------------------------------------------------
# FairExplAIner (metric-level)
# ---------------------------------------------------------------------------


def test_known_metrics_all_have_definitions():
    """Every metric in the knowledge base produces a real definition."""
    fx = FairExplAIner()
    assert fx.metrics_definitions, "metric knowledge base is empty"
    for name in fx.metrics_definitions:
        expl = fx.explain_metric(name, 0.1)
        assert isinstance(expl, MetricExplanation)
        assert expl.definition.strip(), f"{name} has no definition"
        assert set(expl.to_dict()) == _METRIC_KEYS


def test_explain_metric_severity_escalates_with_value():
    """A larger disparity must not produce a milder severity (difference metric)."""
    fx = FairExplAIner()
    low = fx.explain_metric("demographic_parity_difference", 0.01)
    high = fx.explain_metric("demographic_parity_difference", 0.40)
    assert _SEV_ORDER[high.severity] >= _SEV_ORDER[low.severity]
    assert high.severity in ("high", "critical")
    assert high.recommendation.strip(), "severe finding lacks a recommendation"


def test_explain_metric_unknown_name_is_safe():
    """Unknown metrics degrade gracefully, never raise."""
    fx = FairExplAIner()
    expl = fx.explain_metric("not_a_real_metric", 0.5)
    assert isinstance(expl, MetricExplanation)


def test_explain_report_shape():
    """explain_report returns a {metrics, statistical, summary} dict. The
    consumer handler drives per-metric explanations through explain_metric()
    instead, so here we only assert the report contract holds and is safe."""
    fx = FairExplAIner()
    report = fx.explain_report(
        {
            "demographic_parity_difference": 0.40,
            "equalized_odds_difference": 0.01,
        }
    )
    assert isinstance(report, dict)
    assert {"metrics", "statistical", "summary"} <= set(report)
    assert isinstance(report["metrics"], dict)
    assert isinstance(report["summary"], str) and report["summary"].strip()


def test_explain_metric_drives_per_key_explanations():
    """The path the vfairness_explain handler actually uses: one
    MetricExplanation per metric key, all well-formed."""
    fx = FairExplAIner()
    for name in ("demographic_parity_difference", "equalized_odds_difference"):
        expl = fx.explain_metric(name, 0.18)
        assert isinstance(expl, MetricExplanation)
        assert expl.metric_name and expl.severity in _SEV_ORDER


# ---------------------------------------------------------------------------
# FairnessExplainer (facade / registry)
# ---------------------------------------------------------------------------


def test_all_expected_result_types_registered():
    missing = EXPECTED_RESULT_TYPES - set(FairnessExplainer._handlers)
    assert not missing, f"facade missing handlers for: {missing}"


@pytest.mark.parametrize("type_name", sorted(EXPECTED_RESULT_TYPES))
def test_registered_handlers_are_defensive(type_name):
    """Each handler must produce a valid ExplanationReport even when the
    result object only has default/empty attributes (handlers use getattr
    with defaults), and must never raise."""
    DummyResult = type(type_name, (), {})
    obj = DummyResult()
    assert FairnessExplainer.can_explain(obj) is True
    report = FairnessExplainer.explain(obj)
    assert isinstance(report, ExplanationReport)
    d = report.to_dict()
    assert set(d) == _REPORT_KEYS
    assert d["title"].strip()
    assert d["severity"] in _SEV_ORDER


def test_unknown_object_raises_typeerror():
    """The facade refuses unknown objects rather than guessing."""
    with pytest.raises(TypeError):
        FairnessExplainer.explain(object())


def test_can_explain_false_for_unknown():
    assert FairnessExplainer.can_explain(object()) is False


def test_get_explanation_method_path():
    """Objects exposing get_explanation() are honoured by the facade."""

    class HasOwn:
        def get_explanation(self):
            return ExplanationReport(title="Custom", summary="s", explanations=[], severity="low")

    report = FairnessExplainer.explain(HasOwn())
    assert isinstance(report, ExplanationReport)
    # No registered handler produces title "Custom", so this proves the
    # get_explanation() duck-type path was taken.
    assert "Custom" in report.title
