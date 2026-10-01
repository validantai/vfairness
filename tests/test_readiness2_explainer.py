"""Readiness wave 2: the explainer must not contradict the object it explains.

``FairnessExplainer`` is the surface most readers see, and it is exported in
``vfairness.__all__``. Six verdicts it renders are three-state at the source and
were read two-state here, so a run in which NOTHING was measured produced the
same all-clear sentence, and the same 'info' severity, as a run that was
measured and came back clean:

1. ``MultiscaleDriftResult.drift_detected`` (Optional[bool]; None = no scale
   could be computed, so no drift test ran) rendered as "Drift not detected" and
   "No immediate action required.", while ``reports._drift_verdict_word`` printed
   COULD NOT CHECK for the same object.
2. ``CalibrationReport.has_significant_disparity`` (Optional[bool]; None = fewer
   than two groups measured, CAL-DISP) rendered as "No significant calibration
   disparity.", contradicting the report's own ``summary()``, which prints
   "Not assessable (not measured)".
3. ``ExperimentResult.heterogeneity_detected`` (Optional[bool]) rendered as
   "Heterogeneity not detected" and "Effects are consistent; safe to deploy
   uniformly.", contradicting the object's own repr ("not assessed").
4. ``IntersectionEffect.significant`` (Optional[bool]; None = the p-value is not
   finite, so no test ran) silently dropped an untested intersection into the
   cleared pile.
5. ``WindowMetrics.any_alert`` (Optional[bool]; None = nothing in the window was
   compared to a threshold, R-3) rendered as "no alerts" / "No action needed.".
6. ``getattr(finding, "is_significant", False)`` on the bias audit named a field
   ``StatisticalDisparityResult`` does not have. A wrong name is silent, so EVERY
   comparison counted as not significant: a p-value of 5e-50 rated
   HIGHLY_SIGNIFICANT read as "0 of 1 comparisons are statistically significant"
   at severity 'info'.

Every pin below comes in a pair: a REFUSAL pin (the could-not-check state must
not read as the negative) and an OVER-CORRECTION control (a measured verdict must
still produce its exact existing sentence, asserted as a full string, never as
membership of a broad set).
"""

import warnings
from datetime import datetime

import numpy as np
import pytest

from vfairness.explainer import (
    _COULD_NOT_CHECK,
    _SEVERITY_ORDER,
    FairnessExplainer,
    _explain_bias_audit,
    _explain_calibration,
    _explain_drift_result,
    _explain_experiment_result,
    _explain_window_metrics,
)
from vfairness.operations.experimentation.experiment import (
    DesignType,
    ExperimentResult,
    IntersectionEffect,
)
from vfairness.operations.monitoring.drift import (
    DriftResult,
    FairnessDriftDetector,
    MultiscaleDriftResult,
)
from vfairness.operations.monitoring.tracker import WindowMetrics
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.preprocessing.bias_detection.statistical import (
    SignificanceLevel,
    StatisticalDisparityResult,
)


def _rank(sev: str) -> int:
    return _SEVERITY_ORDER[sev]


def _all_text(report) -> str:
    """Every sentence a reader of this report can see, in one string."""
    parts = [report.summary, *report.recommendations]
    for card in report.explanations:
        parts += [
            card.metric_name,
            card.definition,
            card.interpretation_guide,
            str(card.value),
            card.evaluation,
            card.recommendation,
        ]
    return "\n".join(parts)


# 1. Drift: MultiscaleDriftResult.drift_detected is Optional[bool]


def _scale(detected: bool = True, score: float = 0.4) -> DriftResult:
    return DriftResult(
        metric="demographic_parity",
        scale="d1",
        ks_statistic=0.4,
        p_value=0.01,
        drift_score=score,
        drift_detected=detected,
        reference_mean=0.10,
        current_mean=0.20,
        reference_n=30,
        current_n=30,
    )


def _drift(detected, score, scales=None) -> MultiscaleDriftResult:
    return MultiscaleDriftResult(
        metric="demographic_parity",
        timestamp=datetime(2026, 9, 10, 12, 0, 0),
        scales=scales if scales is not None else {},
        overall_drift_score=score,
        drift_detected=detected,
    )


def test_drift_no_verdict_is_could_not_check_from_the_real_producer():
    """REFUSAL PIN. The real detector returns drift_detected=None for a series
    too short to decompose; the explainer must not call that stability."""
    import pandas as pd

    detector = FairnessDriftDetector()
    series = pd.Series([0.10, 0.12], index=pd.date_range("2025-01-01", periods=2, freq="D"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detector.detect_drift_multiscale(series, metric="demographic_parity")

    assert result.drift_detected is None, "producer no longer reaches the None state"

    report = _explain_drift_result(result)
    text = _all_text(report)
    assert _COULD_NOT_CHECK in report.summary
    assert _COULD_NOT_CHECK in report.explanations[0].evaluation
    assert "not a finding of stability" in text
    # The two sentences that made this dangerous.
    assert "Drift not detected" not in text
    assert "No immediate action required" not in text
    # The VERDICT, not just the wording: an unmeasured run must not be graded
    # with the all-clear a measured clean run earns.
    assert report.severity != "info"
    assert _rank(report.severity) >= _rank("medium")
    assert report.recommendations, "a could-not-check run must tell the reader what to do"
    assert _COULD_NOT_CHECK in report.recommendations[0]


def test_drift_no_verdict_pin_holds_for_a_report_object_missing_the_field():
    """REFUSAL PIN. A duck-typed result that never reported the verdict has not
    reported a negative one either."""

    class _NoVerdict:
        overall_drift_score = 0.0
        scales: dict = {}
        worst_scale = None

    report = _explain_drift_result(_NoVerdict())
    assert _COULD_NOT_CHECK in report.summary
    assert report.severity == "medium"


def test_drift_detected_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED drift keeps its sentences verbatim."""
    report = _explain_drift_result(_drift(True, 0.7, {"d1": _scale()}))
    card = report.explanations[0]
    assert card.evaluation == "Drift DETECTED (score = 0.7000)."
    assert card.recommendation == (
        "Investigate root cause: data pipeline change, population shift, or "
        "feedback loop. Consider retraining or recalibrating."
    )
    assert report.summary == (
        "Drift detected with overall score 0.7000. Analysed 1 temporal scale(s)."
    )
    assert report.severity == "high"
    assert report.recommendations[0] == ("Drift detected (score 0.700). Investigate root cause.")
    assert _COULD_NOT_CHECK not in _all_text(report)


def test_drift_measured_clean_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED clean run keeps its all-clear."""
    report = _explain_drift_result(_drift(False, 0.1, {"d1": _scale(detected=False)}))
    card = report.explanations[0]
    assert card.evaluation == "Drift not detected (score = 0.1000)."
    assert card.recommendation == "No immediate action required. Continue standard monitoring."
    assert report.summary == (
        "Drift not detected with overall score 0.1000. Analysed 1 temporal scale(s)."
    )
    assert report.severity == "info"
    assert report.recommendations == []
    assert _COULD_NOT_CHECK not in _all_text(report)


# 2. Calibration: CalibrationReport.has_significant_disparity is Optional[bool]


def _calibration_report_with_one_group():
    rng = np.random.default_rng(7)
    n = 400
    y_prob = rng.uniform(0.05, 0.95, n)
    y_true = (rng.uniform(size=n) < y_prob).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CalibrationAnalyzer(
            y_true, y_prob, np.array(["A"] * n), attribute_name="group"
        ).full_analysis()


def test_calibration_unmeasured_disparity_agrees_with_the_reports_own_summary():
    """REFUSAL PIN. One group means no between-group comparison was made. The
    report's own summary() says so; the explainer used to say the opposite."""
    report = _calibration_report_with_one_group()
    assert report.has_significant_disparity is None
    assert "Not assessable (not measured)" in report.summary()

    out = _explain_calibration(report)
    text = _all_text(out)
    assert _COULD_NOT_CHECK in out.summary
    assert "No significant calibration disparity." not in text
    assert "not a finding of no disparity" in text
    assert _rank(out.severity) >= _rank("medium")
    card = [e for e in out.explanations if e.metric_name == "Calibration Disparity Across Groups"]
    assert len(card) == 1, "the missing verdict must be stated where the reader looks"
    assert _COULD_NOT_CHECK in card[0].evaluation
    assert card[0].severity == "medium"


def test_calibration_measured_disparity_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED disparity keeps its sentences."""

    class _Report:
        overall_metrics = {"ece": 0.02}
        has_significant_disparity = True
        recommendations: list = []
        n_groups = 2

    out = _explain_calibration(_Report())
    card = [e for e in out.explanations if e.metric_name == "Calibration Disparity Across Groups"][
        0
    ]
    assert card.evaluation == (
        "Calibration quality varies significantly across groups. "
        "This can cause systematically different decision outcomes."
    )
    assert card.value == "Significant disparity detected"
    assert card.severity == "medium"
    assert out.summary == (
        "Calibration analysis for 2 groups. ECE = 0.0200. "
        "Significant calibration disparity across groups."
    )
    assert out.severity == "medium"
    assert _COULD_NOT_CHECK not in _all_text(out)


def test_calibration_measured_no_disparity_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED clean comparison keeps its all-clear."""

    class _Report:
        overall_metrics = {"ece": 0.02}
        has_significant_disparity = False
        recommendations: list = []
        n_groups = 3

    out = _explain_calibration(_Report())
    assert out.summary == (
        "Calibration analysis for 3 groups. ECE = 0.0200. No significant calibration disparity."
    )
    assert out.severity == "info"
    assert [e.metric_name for e in out.explanations] == ["Expected Calibration Error (ECE)"]
    assert _COULD_NOT_CHECK not in _all_text(out)


# 3. Experiment: heterogeneity_detected and IntersectionEffect.significant


def _effect(effect: float, significant, p_value: float = 0.01) -> IntersectionEffect:
    return IntersectionEffect(
        intersection=("Female", "Black"),
        control_mean=0.5,
        treatment_mean=0.5 + effect,
        effect=effect,
        ci_lower=effect - 0.05,
        ci_upper=effect + 0.05,
        p_value=p_value,
        effect_size_d=0.35,
        n_control=120,
        n_treatment=120,
        significant=significant,
    )


def _experiment(het, het_p, effects, overall_p=0.30) -> ExperimentResult:
    return ExperimentResult(
        overall_effect=0.05,
        overall_ci=(-0.01, 0.11),
        overall_p_value=overall_p,
        intersection_effects=effects,
        heterogeneity_detected=het,
        heterogeneity_p_value=het_p,
        n_intersections=len(effects),
        design_type=DesignType.SIMPLE_AB,
    )


def test_experiment_unassessed_heterogeneity_agrees_with_the_objects_own_repr():
    """REFUSAL PIN. The object's repr says 'not assessed'; so must the explainer."""
    result = _experiment(None, float("nan"), [_effect(0.02, True)])
    assert "heterogeneity=not assessed" in repr(result)

    out = _explain_experiment_result(result)
    text = _all_text(out)
    assert "Heterogeneity: not assessed." in out.summary
    assert _COULD_NOT_CHECK in text
    assert "Heterogeneity not detected" not in text
    assert "Effects are consistent; safe to deploy uniformly." not in text
    assert "p = nan" not in text
    assert _rank(out.severity) >= _rank("medium")
    het_card = [e for e in out.explanations if e.metric_name == "Heterogeneity Test"][0]
    assert het_card.severity == "medium"
    assert _COULD_NOT_CHECK in out.recommendations[0]


def test_experiment_untested_intersection_is_not_a_cleared_one():
    """REFUSAL PIN. significant=None means no test ran on that intersection: it
    must be named, not silently counted as clean."""
    out = _explain_experiment_result(
        _experiment(False, 0.40, [_effect(-0.30, None, p_value=float("nan"))])
    )
    text = _all_text(out)
    assert _COULD_NOT_CHECK in out.summary
    assert "1 intersection(s) reported no significance verdict" in text
    assert _rank(out.severity) >= _rank("medium")
    # Not fabricated in the other direction either: an untested intersection is
    # not evidence of harm, so no harmed-group card is invented for it.
    assert not [e for e in out.explanations if e.metric_name.startswith("Harmed Group")]


def test_experiment_detected_heterogeneity_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED heterogeneity keeps its sentences."""
    out = _explain_experiment_result(
        _experiment(True, 0.001, [_effect(-0.30, True), _effect(0.10, True)])
    )
    het_card = [e for e in out.explanations if e.metric_name == "Heterogeneity Test"][0]
    assert het_card.evaluation == "Heterogeneity DETECTED (p = 0.0010) across 2 intersections."
    assert het_card.recommendation == (
        "Treatment effects vary across groups: consider group-specific "
        "deployment or further investigation."
    )
    assert het_card.severity == "medium"
    assert out.summary == (
        "Overall effect +0.0500 (p=0.3000). 2 intersections analysed. Heterogeneity: YES."
    )
    assert out.severity == "high"
    harmed = [e for e in out.explanations if e.metric_name.startswith("Harmed Group")]
    assert len(harmed) == 1
    assert harmed[0].evaluation == "Effect = -0.3000, d = 0.350, p = 0.0100."
    assert harmed[0].severity == "high"
    assert _COULD_NOT_CHECK not in _all_text(out)


def test_experiment_measured_no_heterogeneity_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED clean experiment keeps its all-clear."""
    out = _explain_experiment_result(_experiment(False, 0.40, [_effect(0.02, False)]))
    het_card = [e for e in out.explanations if e.metric_name == "Heterogeneity Test"][0]
    assert het_card.evaluation == "Heterogeneity not detected (p = 0.4000) across 1 intersections."
    assert het_card.recommendation == "Effects are consistent; safe to deploy uniformly."
    assert het_card.severity == "info"
    overall_card = [e for e in out.explanations if e.metric_name == "Overall Treatment Effect"][0]
    assert overall_card.recommendation == "Effect is consistent across groups."
    assert out.summary == (
        "Overall effect +0.0500 (p=0.3000). 1 intersections analysed. Heterogeneity: no."
    )
    assert out.severity == "info"
    assert _COULD_NOT_CHECK not in _all_text(out)


# 4. Monitoring window: WindowMetrics.any_alert is Optional[bool]


def _window(alerts, metrics=None) -> WindowMetrics:
    return WindowMetrics(
        batch_id="b1",
        timestamp=datetime(2026, 9, 10, 12, 0, 0),
        sample_count=500,
        metrics=metrics if metrics is not None else {},
        group_rates={},
        alerts=alerts,
    )


def test_window_with_no_threshold_comparison_is_could_not_check():
    """REFUSAL PIN. any_alert=None means nothing was compared to a threshold."""
    snap = _window({})
    assert snap.any_alert is None

    out = _explain_window_metrics(snap)
    text = _all_text(out)
    assert _COULD_NOT_CHECK in out.summary
    assert "no alerts" not in text
    assert "No action needed." not in out.recommendations
    assert _rank(out.severity) >= _rank("medium")
    assert "not a finding of a clean window" in text


def test_window_alert_count_is_breaches_not_comparisons():
    """OVER-CORRECTION CONTROL plus a count pin: `alerts` maps metric ->
    BREACHED, so its length counts comparisons. One breach is one alert."""
    snap = _window(
        {"disparate_impact": True, "demographic_parity": False},
        metrics={"disparate_impact": 0.6, "demographic_parity": 0.01},
    )
    assert snap.any_alert is True

    out = _explain_window_metrics(snap)
    card = [e for e in out.explanations if e.metric_name == "Alert Status"][0]
    assert card.value == "1 alert(s) active"
    assert card.evaluation == "1 alert(s) fired in the current window."
    assert card.severity == "high"
    assert out.summary == "Window contains 2 metric(s); 1 alert(s) active."
    assert out.severity == "high"
    assert out.recommendations == ["Investigate and resolve active alerts."]
    assert _COULD_NOT_CHECK not in _all_text(out)


def test_window_measured_clean_still_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL: a window that WAS compared and came back clean
    keeps its all-clear."""
    snap = _window({"disparate_impact": False}, metrics={"disparate_impact": 0.95})
    assert snap.any_alert is False

    out = _explain_window_metrics(snap)
    assert out.summary == "Window contains 1 metric(s); no alerts."
    assert out.severity == "info"
    assert out.recommendations == ["No action needed."]
    assert not [e for e in out.explanations if e.metric_name == "Alert Status"]
    assert _COULD_NOT_CHECK not in _all_text(out)


# 5. Bias audit: the significance verdict was read under a name no class has


def _finding(significance) -> StatisticalDisparityResult:
    from vfairness.preprocessing.bias_detection.statistical import (
        DisparityType,
        EffectSizeInterpretation,
    )

    return StatisticalDisparityResult(
        feature="target",
        protected_attribute="gender",
        disparity_type=DisparityType.OUTCOME,
        test_name="Chi-square test",
        test_statistic=210.0,
        pvalue=5.3e-50,
        significance=significance,
        effect_size=0.6,
        effect_size_type="Cramer's V",
        effect_interpretation=EffectSizeInterpretation.LARGE,
        confidence_interval=(0.5, 0.7),
        group_statistics={},
        privileged_group="A",
        disadvantaged_group="B",
        disparity_magnitude=0.6,
        sample_sizes={"A": 300, "B": 300},
        recommendations=[],
    )


class _Audit:
    def __init__(self, findings):
        self.overall_risk_score = 0.1
        self.protected_attributes = ["gender"]
        self.critical_issues: list = []
        self.historical_findings: list = []
        self.representation_findings: list = []
        self.disparity_findings = findings
        self.proxy_findings: list = []
        self.recommendations: list = []


def test_the_shipped_finding_class_has_no_is_significant_field():
    """Name pin. The old read used a field this class does not declare, and a
    wrong name in getattr is silent, so the count was structurally always 0."""
    attrs = set(vars(_finding(SignificanceLevel.SIGNIFICANT)))
    assert "significance" in attrs
    assert "is_significant" not in attrs
    # And the module must not read that name from any object again. Checked on
    # the parsed CODE, not on the source text: the docstring recording this
    # incident quotes the old name on purpose.
    import ast
    import inspect

    import vfairness.explainer as explainer_module

    tree = ast.parse(inspect.getsource(explainer_module))
    reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "is_significant"
    ]
    assert not reads, f"explainer reads a field no result class declares, at {reads}"


def test_bias_audit_counts_a_highly_significant_disparity():
    """OVER-CORRECTION CONTROL, and the fix for the 0-of-N bug: a real
    HIGHLY_SIGNIFICANT verdict must be counted and graded."""
    out = _explain_bias_audit(_Audit([_finding(SignificanceLevel.HIGHLY_SIGNIFICANT)]))
    card = [e for e in out.explanations if e.metric_name == "Statistical Disparity Analysis"][0]
    assert card.value == "1 comparisons, 1 statistically significant"
    assert card.evaluation == "1 of 1 comparisons are statistically significant."
    assert card.severity == "high"
    assert _COULD_NOT_CHECK not in _all_text(out)


def test_bias_audit_measured_not_significant_still_reads_as_before():
    """OVER-CORRECTION CONTROL: a MEASURED non-significant comparison stays the
    all-clear it has always been, with no could-not-check tail."""
    out = _explain_bias_audit(_Audit([_finding(SignificanceLevel.NOT_SIGNIFICANT)]))
    card = [e for e in out.explanations if e.metric_name == "Statistical Disparity Analysis"][0]
    assert card.value == "1 comparisons, 0 statistically significant"
    assert card.evaluation == "0 of 1 comparisons are statistically significant."
    assert card.severity == "info"
    assert _COULD_NOT_CHECK not in _all_text(out)


def test_bias_audit_marginally_significant_counts_as_significant():
    """OVER-CORRECTION CONTROL: the canonical reader in `statistical` is
    `significance != NOT_SIGNIFICANT`, so MARGINALLY_SIGNIFICANT counts."""
    out = _explain_bias_audit(_Audit([_finding(SignificanceLevel.MARGINALLY_SIGNIFICANT)]))
    card = [e for e in out.explanations if e.metric_name == "Statistical Disparity Analysis"][0]
    assert card.evaluation == "1 of 1 comparisons are statistically significant."


def test_bias_audit_finding_without_a_verdict_is_could_not_check():
    """REFUSAL PIN. A comparison carrying no readable verdict is not a
    comparison that came back clean."""
    out = _explain_bias_audit(_Audit([_finding(None)]))
    card = [e for e in out.explanations if e.metric_name == "Statistical Disparity Analysis"][0]
    assert _COULD_NOT_CHECK in card.evaluation
    assert "1 comparison(s) carry no significance verdict" in card.evaluation
    assert card.severity == "medium"


# 6. Whole-file sweep: no three-state verdict may be read two-state again


THREE_STATE_FIELDS = {
    "drift_detected",
    "has_significant_disparity",
    "heterogeneity_detected",
    "significant",
    "any_alert",
    "is_well_calibrated",
}


# The ONE allowed two-state read of a name in that set. The per-scale
# ``DriftResult.drift_detected`` is declared a plain ``bool``, not Optional, so
# there is no third state to lose there; the aggregate
# ``MultiscaleDriftResult.drift_detected`` beside it is the Optional one. The
# allowance is guarded by the annotation below, so it expires by itself the day
# that field becomes Optional.
_ALLOWED_TWO_STATE = {("dr", "drift_detected")}


def test_the_allowed_two_state_read_is_still_a_plain_bool_field():
    """The allow-list above is only valid while this stays a non-Optional bool."""
    assert DriftResult.__annotations__["drift_detected"] == "bool"
    assert MultiscaleDriftResult.__annotations__["drift_detected"] == "Optional[bool]"


def test_no_three_state_verdict_is_read_with_a_boolean_default():
    """Regression tripwire for the whole module. Every one of these six defects
    looked identical in source: getattr(obj, "<verdict>", False)."""
    import ast
    import inspect

    import vfairness.explainer as explainer_module

    tree = ast.parse(inspect.getsource(explainer_module))
    offenders = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        name = node.args[1].value
        default = node.args[2]
        is_bool_default = isinstance(default, ast.Constant) and isinstance(default.value, bool)
        obj = ast.unparse(node.args[0])
        if (obj, name) in _ALLOWED_TWO_STATE:
            continue
        if name in THREE_STATE_FIELDS and is_bool_default:
            offenders.append((node.lineno, obj, name))
    assert not offenders, f"three-state verdicts read two-state: {offenders}"


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _drift(None, float("nan")),
        _calibration_report_with_one_group,
        lambda: _experiment(None, float("nan"), []),
        lambda: _window({}),
    ],
    ids=["drift", "calibration", "experiment", "window"],
)
def test_every_unmeasured_verdict_refuses_the_info_all_clear(builder):
    """One assertion across four result types: the VERDICT, not the wording.
    A guard that only checks the sentence catches nothing, because 'severity'
    is what a report pipeline or dashboard reads instead of the sentence."""
    out = FairnessExplainer.explain(builder())
    assert out.severity != "info", out.summary
    assert _rank(out.severity) >= _rank("medium")
    assert _COULD_NOT_CHECK in _all_text(out)
