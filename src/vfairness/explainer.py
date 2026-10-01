"""
FairnessExplainer: Unified Explanation Facade for the vfairness Library.

This module implements the **registry-pattern facade** that dispatches
educational, context-aware explanations for *any* result object produced by
the library: from bias audit reports to drift detection results.

Architecture
~~~~~~~~~~~~
* ``ExplanationReport``: container for one or more ``MetricExplanation``
  items, plus a title, summary, and overall severity.
* ``FairnessExplainer``: the public facade.  Call ``FairnessExplainer.explain(obj)``
  with any supported result object and get back an ``ExplanationReport``.
* Each major class also exposes a convenience ``get_explanation()`` method
  that delegates here.

Example::

    from vfairness.explainer import FairnessExplainer

    # Works with any result object
    report = detector.full_audit()
    explanation = FairnessExplainer.explain(report)
    print(explanation)

    # Or use the convenience method on the class itself
    explanation = detector.get_explanation()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from . import _bands
from ._triage import is_measured, unmeasurable_reason
from .evaluation.vfairness_metrics.explainer import MetricExplanation

# Severity vocabulary shared by MetricExplanation and ExplanationReport. The
# _bands helpers return a plain ``str`` from this same closed set, so narrowing
# their output back to the Literal at the boundary is a pure type refinement
# (no runtime change): every value they can produce is a member of this set.
Severity = Literal["info", "low", "medium", "high", "critical"]


def _sev(value: str) -> Severity:
    """Narrow a band/severity ``str`` to the ``Severity`` Literal.

    The ``_bands`` helpers and ``BAND_SEVERITY`` only ever yield members of the
    severity vocabulary; this cast records that fact for the type checker
    without altering the value.
    """
    return cast(Severity, value)


# ExplanationReport: multi-item explanation container


@dataclass
class ExplanationReport:
    """Container for multiple explanations from a complex result object.

    Attributes
    ----------
    title : str
        Short human-readable title (e.g. "Bias Audit Explanation").
    summary : str
        One-paragraph narrative summarising the findings.
    explanations : list[MetricExplanation]
        Individual metric / concept explanations.
    severity : str
        Aggregate severity across all explanations.
    recommendations : list[str]
        Top-level action items distilled from the explanations.
    """

    title: str
    summary: str
    explanations: List[MetricExplanation]
    severity: Literal["info", "low", "medium", "high", "critical"] = "info"
    recommendations: List[str] = field(default_factory=list)

    # Serialisation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "severity": self.severity,
            "recommendations": self.recommendations,
            "explanations": [e.to_dict() for e in self.explanations],
        }

    # Pretty-printing

    def __str__(self) -> str:
        width = 72
        lines: List[str] = [
            "=" * width,
            f"  {self.title}  [{self.severity.upper()}]",
            "=" * width,
            "",
            self.summary,
            "",
        ]
        if self.recommendations:
            lines.append("ACTION ITEMS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        lines.append("-" * width)
        for expl in self.explanations:
            lines.append(str(expl))
            lines.append("-" * width)
        return "\n".join(lines)


# Severity helpers

_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _worst_severity(*severities: str) -> Severity:
    return _sev(max(severities, key=lambda s: _SEVERITY_ORDER.get(s, 0)))


# Could-not-check vocabulary, spelled the way the library's other reader
# surfaces already spell it: ``rendering.explain._COULD_NOT_CHECK``,
# ``operations.reporting.reports._drift_verdict_word``,
# ``post_processing.calibration.analyzer._yes_no_unknown`` and
# ``operations.experimentation.experiment._yes_no_not_assessed``. The explainer
# is the surface most readers see, so it must not contradict the object it is
# explaining by turning that object's honest "nobody looked" into the all-clear
# its measured negative would carry.
_COULD_NOT_CHECK = "COULD NOT CHECK"

# An unmeasured verdict is not benign news, and it is not a finding either:
# nothing was measured to be wrong. Same floor band, and the same reasoning, as
# ``rendering.explain._UNKNOWN_SEV``.
_UNKNOWN_SEV: Severity = "medium"

# Sentinel for "the result object does not carry this field at all", which is a
# different answer from "it carries None" only where the two must be told apart.
_MISSING = object()


def _verdict(obj: Any, name: str) -> Optional[bool]:
    """Read a three-state verdict field from a result object: True/False/None.

    ``getattr(obj, name, False)`` is a TWO-state read of a three-state field.
    It collapses both "the object reported None" and "the object never reported
    this verdict at all" into the measured negative, which is the one answer
    neither of them gives. Every verdict this module reads that way is declared
    ``Optional[bool]`` by its own dataclass with None documented as
    could-not-check (``MultiscaleDriftResult.drift_detected``,
    ``CalibrationReport.has_significant_disparity``,
    ``ExperimentResult.heterogeneity_detected``,
    ``IntersectionEffect.significant``, ``WindowMetrics.any_alert``), so the
    explainer used to contradict the same object's own ``summary()``,
    ``__repr__`` and ``to_dict()``.

    An absent attribute is could-not-check for the same reason
    ``rendering.explain`` treats an absent key that way: a duck-typed result
    that never reported the verdict has not reported a negative one.
    """
    value = getattr(obj, name, None)
    return None if value is None else bool(value)


def _measured(value: Any) -> bool:
    """Did a test actually produce this p-value?

    THE THIRD STATE ON A FLOAT. ``_verdict`` above gives Optional[bool] fields
    their three states; a p-value carries the same three states in a float, and
    NaN is the could-not-check one. Every comparison against NaN is False, so
    ``if p >= 0.05`` silently skips the not-significant branch AND ``if p <
    0.05`` silently skips the significant one, leaving the unmeasured case to
    fall through to whatever the code says last.

    Measured 2026-09-10 on ``_explain_experiment_result``, whose producer
    (``FairnessExperiment.analyze``) sets ``overall_p_value`` to NaN
    DELIBERATELY when both arms are constant and no t statistic exists:

        measured p=0.90    sev=info ['Overall effect not significant. Consider
                                     extending the experiment.']
        measured p=0.001   sev=low  ['Results look consistent. Proceed with
                                     deployment review.']
        NOT MEASURED p=nan sev=info ['Results look consistent. Proceed with
                                     deployment review.']

    The unmeasured reading was byte-identical to the STRONG REAL EFFECT one.

    READINESS-6, 2026-09-10. Delegated to ``_triage.is_measured``. This
    implementation was the CORRECT one of the six and is what the canonical
    module now uses; the delegation is so there is one of it rather than two
    that happen to agree today.
    """
    return is_measured(value)


# The significance vocabulary of
# ``preprocessing.bias_detection.statistical.SignificanceLevel``. Anything
# outside it is a verdict this module cannot read, which is could-not-check.
_SIGNIFICANCE_VERDICTS = {
    "highly_significant": True,
    "significant": True,
    "marginally_significant": True,
    "not_significant": False,
}


def _disparity_is_significant(finding: Any) -> Optional[bool]:
    """Three-state significance verdict for one statistical disparity finding.

    ``getattr(finding, "is_significant", False)`` named a field that
    ``StatisticalDisparityResult`` does not have, and a wrong name is silent:
    EVERY finding counted as not significant. Reproduced 2026-09-10 on a real
    ``BiasDetector.full_audit()`` whose single finding carried
    ``pvalue=5.3e-50`` and ``significance=HIGHLY_SIGNIFICANT``, while this card
    read "0 of 1 comparisons are statistically significant" at severity 'info'.

    The field that carries the verdict is ``significance``, and the canonical
    reader is ``significance != NOT_SIGNIFICANT``
    (``statistical.analyze_statistical_disparities``). A finding carrying no
    readable verdict is None, never False.
    """
    raw = getattr(finding, "significance", None)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return _SIGNIFICANCE_VERDICTS.get(_enum_str(raw).lower().strip())


def _unknown_tail(count: int, noun: str) -> str:
    """Name the items that carry no verdict, or say nothing when all do.

    Mirrors ``rendering.explain._ungraded_tail``: a count of graded items is
    only honest beside a count of the ungraded ones.
    """
    if count <= 0:
        return ""
    return f" {_COULD_NOT_CHECK}: {count} {noun}."


def _enum_str(value: Any) -> str:
    """String form of a severity/risk field that may be an Enum.

    ``str(SomeEnum.HIGH)`` is ``'SomeEnum.HIGH'``, so comparing it against
    'high' silently fails, which zeroed out every severity count built this
    way. Unwrap ``.value`` first.
    """
    return str(getattr(value, "value", value))


# NOTE: the old _score_to_severity helper (0.2/0.4/0.6/0.8 bands) was removed:
# it was unused and its bands diverged from the canonical 0.25/0.50/0.75 badge
# scale in _bands. All risk severities must come from _bands.risk_band_severity.


def _risk_band(score: float) -> str:
    """Map a 0-1 risk score to the SAME band label the SVG risk badge shows.

    Delegates to ``_bands.risk_band``, the canonical 0.25/0.50/0.75 scale
    that ``rendering.engine._risk_label`` also follows. A report's summary
    must never contradict its own badge, so the score's band label is
    computed from the score here, not borrowed from the worst finding's
    severity."""
    return _bands.risk_band(score)


# Domain-specific handlers (one per result type)


def _explain_bias_audit(report: Any) -> ExplanationReport:
    score = getattr(report, "overall_risk_score", 0.0)
    # Severity, evaluation text and interpretation guide all derive from the
    # SAME band scale the SVG badge uses (_risk_band: 0.25/0.50/0.75), so the
    # explanation can never contradict the badge. Previously the severity used
    # a divergent 0.2/0.4/0.6/0.8 scale, disagreeing for ~40% of scores.
    band = _risk_band(score)
    sev = _sev(_bands.BAND_SEVERITY[band])

    explanations: List[MetricExplanation] = []

    # 1. Overall risk
    if score < 0.25:
        eval_text = f"The overall risk score of {score:.2f} is in the MINIMAL band. The dataset appears suitable for training with standard monitoring."
    elif score < 0.50:
        eval_text = f"The overall risk score of {score:.2f} is in the LOW band. Several bias indicators were flagged; review before proceeding."
    elif score < 0.75:
        eval_text = f"The overall risk score of {score:.2f} is in the MEDIUM band. Mitigation is recommended before model training."
    else:
        eval_text = f"The overall risk score of {score:.2f} is in the HIGH band. Multiple bias sources compound; do not train without intervention."

    explanations.append(
        MetricExplanation(
            metric_name="Overall Bias Risk Score",
            definition=(
                "Aggregate measure of dataset bias risk (0 = no risk, 1 = critical). "
                "Combines historical patterns, representation balance, statistical "
                "disparities, and proxy variable exposure."
            ),
            interpretation_guide=(
                "0.00-0.25: MINIMAL, proceed with monitoring. "
                "0.25-0.50: LOW, review flagged items. "
                "0.50-0.75: MEDIUM, apply mitigation before training. "
                "0.75-1.00: HIGH, do not train without intervention."
            ),
            value=score,
            evaluation=eval_text,
            benchmark_context=(
                "Industry best practice targets a score below 0.3. Regulatory frameworks "
                "(EU AI Act, ECOA, EEOC) require documented bias assessment for high-risk systems."
            ),
            recommendation=(
                "Run BiasDetector.full_audit() to identify specific issues, then apply "
                "targeted mitigations (e.g. rebalancing, proxy removal, feature engineering)."
                if score >= 0.50
                else "Continue with standard fairness monitoring during training and deployment."
            ),
            severity=sev,
            related_metrics=["representation_bias", "statistical_disparity", "proxy_risk"],
        )
    )

    # 2. Historical patterns
    hist = getattr(report, "historical_findings", [])
    if hist:
        n_high = sum(
            1
            for h in hist
            if getattr(h, "risk_level", None)
            and _enum_str(getattr(h, "risk_level", "")).lower() in ("high", "critical")
        )
        explanations.append(
            MetricExplanation(
                metric_name="Historical Pattern Detection",
                definition=(
                    "Identifies column names and value patterns historically associated "
                    'with discrimination (e.g. "zip_code", "marital_status").'
                ),
                interpretation_guide=(
                    "Each finding is rated by risk level. High/Critical findings indicate "
                    "features with known discriminatory associations."
                ),
                value=f"{len(hist)} patterns found ({n_high} high/critical)",
                evaluation=(
                    f"Found {len(hist)} historical pattern(s); {n_high} rated high or critical."
                ),
                benchmark_context=(
                    "Established case law (Griggs v. Duke Power, 1971) and the EEOC Uniform "
                    "Guidelines highlight specific proxies. The EU AI Act Annex III lists "
                    "high-risk application domains."
                ),
                recommendation=(
                    "Review each flagged feature; consider removal or fairness-aware "
                    "transformation (ResidualTransformer, CorrelationReducer)."
                ),
                severity="high" if n_high > 0 else ("medium" if hist else "info"),
                related_metrics=["proxy_variables", "feature_correlation"],
            )
        )

    # 3. Representation bias
    rep = getattr(report, "representation_findings", [])
    if rep:
        n_severe = sum(
            1
            for r in rep
            if _enum_str(getattr(r, "severity", "")).lower() in ("high", "critical", "severe")
        )
        explanations.append(
            MetricExplanation(
                metric_name="Representation Bias",
                definition=(
                    "Measures whether demographic groups are proportionally represented "
                    "in the dataset relative to the target population."
                ),
                interpretation_guide=(
                    "Under-representation leads to higher error rates for minority groups. "
                    "The ratio of observed vs expected frequency quantifies imbalance."
                ),
                value=f"{len(rep)} group imbalances found ({n_severe} severe)",
                evaluation=(
                    f"Detected {len(rep)} representation issue(s); {n_severe} rated severe."
                ),
                benchmark_context=(
                    "The 4/5 rule (EEOC) expects no group to have selection rates below "
                    "80% of the most-favoured group. Under-represented groups typically "
                    "suffer higher misclassification rates."
                ),
                recommendation=(
                    "Consider oversampling under-represented groups, collecting additional "
                    "data, or applying sample weights to restore balance."
                ),
                severity="high" if n_severe > 0 else ("medium" if rep else "info"),
                related_metrics=["statistical_disparity", "sample_size"],
            )
        )

    # 4. Statistical disparities
    disp = getattr(report, "disparity_findings", [])
    if disp:
        verdicts = [_disparity_is_significant(d) for d in disp]
        n_sig = sum(1 for v in verdicts if v is True)
        n_unknown = sum(1 for v in verdicts if v is None)
        unknown_tail = _unknown_tail(n_unknown, "comparison(s) carry no significance verdict")
        explanations.append(
            MetricExplanation(
                metric_name="Statistical Disparity Analysis",
                definition=(
                    "Tests whether outcome rates differ significantly across groups "
                    "using chi-squared tests, Fisher exact tests, and effect-size measures."
                ),
                interpretation_guide=(
                    "Significant disparities (p < 0.05) with large effect sizes indicate "
                    "systematic differences unlikely to be random."
                ),
                value=f"{len(disp)} comparisons, {n_sig} statistically significant",
                evaluation=(
                    f"{n_sig} of {len(disp)} comparisons are statistically significant."
                    + unknown_tail
                ),
                benchmark_context=(
                    "Disparate impact analysis is required by US federal lending "
                    "regulations (ECOA, FHA) and the EU AI Act risk assessment."
                ),
                recommendation=(
                    "For significant disparities, investigate root causes: is the "
                    "label itself biased, or does a proxy feature drive the gap?"
                ),
                severity=_worst_severity(
                    "high" if n_sig > len(disp) * 0.5 else ("medium" if n_sig else "info"),
                    _UNKNOWN_SEV if n_unknown else "info",
                ),
                related_metrics=["proxy_variables", "demographic_parity_difference"],
            )
        )

    # 5. Proxy variables
    prox = getattr(report, "proxy_findings", [])
    if prox:
        n_high_risk = sum(
            1
            for p in prox
            if _enum_str(getattr(p, "risk_level", "")).lower() in ("high", "critical")
        )
        explanations.append(
            MetricExplanation(
                metric_name="Proxy Variable Detection",
                definition=(
                    "Identifies features that are strongly correlated with protected "
                    "attributes and may act as indirect proxies for discrimination."
                ),
                interpretation_guide=(
                    "A proxy with correlation > 0.5 or mutual information > 0.3 carries "
                    'high risk; even "neutral" features like zip code can encode race.'
                ),
                value=f"{len(prox)} proxies found ({n_high_risk} high risk)",
                evaluation=(f"{len(prox)} proxy variable(s) detected; {n_high_risk} high risk."),
                benchmark_context=(
                    "The concept of disparate impact (Griggs v. Duke Power) means even "
                    "facially neutral features can be unlawful if they disproportionately "
                    "affect protected groups without business necessity."
                ),
                recommendation=(
                    "Apply ResidualTransformer to orthogonalise proxy features, or remove "
                    "them entirely if business justification is insufficient."
                ),
                severity="high" if n_high_risk > 0 else ("medium" if prox else "info"),
                related_metrics=["historical_patterns", "feature_correlation"],
            )
        )

    # Build recommendations
    recs = list(getattr(report, "recommendations", []))[:5]

    all_sevs = [e.severity for e in explanations]
    overall = _worst_severity(*all_sevs) if all_sevs else "info"

    return ExplanationReport(
        title="Bias Audit Explanation",
        summary=(
            f"The bias audit analysed {len(getattr(report, 'protected_attributes', []))} "
            f"protected attribute(s) and produced an overall bias-risk score of {score:.2f} "
            f"({_risk_band(score)}). "
            f"{len(getattr(report, 'critical_issues', []))} critical issue(s) require "
            f"immediate attention."
        ),
        explanations=explanations,
        severity=overall,
        recommendations=recs,
    )


def _explain_feature_analysis(report: Any) -> ExplanationReport:
    n_prox = len(getattr(report, "proxy_variables", []))
    n_high = len(getattr(report, "high_risk_features", []))
    sev: Severity = (
        "high" if n_high > 2 else ("medium" if n_high else ("low" if n_prox else "info"))
    )

    explanations: List[MetricExplanation] = []

    explanations.append(
        MetricExplanation(
            metric_name="Feature-Attribute Correlation Analysis",
            definition=(
                "Measures the Pearson, Spearman, and mutual-information correlation "
                "between each feature and each protected attribute."
            ),
            interpretation_guide=(
                "Correlations above 0.3 are moderate; above 0.5 strong. High mutual "
                "information (> 0.1 nats) suggests non-linear associations."
            ),
            value=f"{getattr(report, 'n_features', '?')} features analysed",
            evaluation=(
                f"{n_prox} proxy variable(s) detected; {n_high} high-risk feature(s) flagged."
            ),
            benchmark_context=(
                "Feldman et al. (2015) showed that even low-correlation features can "
                "compound to produce disparate impact when used jointly."
            ),
            recommendation=(
                "Use CorrelationReducer or ResidualTransformer on high-risk features "
                "to decorrelate them from protected attributes."
            ),
            severity=sev,
            related_metrics=["proxy_variables", "representation_bias"],
        )
    )

    chains = getattr(report, "proxy_chains", [])
    if chains:
        explanations.append(
            MetricExplanation(
                metric_name="Proxy Chain Detection",
                definition=(
                    "Identifies indirect proxy pathways: feature A correlates with "
                    "feature B, which correlates with a protected attribute."
                ),
                interpretation_guide=(
                    "Even individually innocuous features may form chains that "
                    "collectively reconstruct protected information."
                ),
                value=f"{len(chains)} proxy chain(s) found",
                evaluation=f"{len(chains)} indirect proxy chain(s) detected.",
                benchmark_context=(
                    "Datta et al. (2017) demonstrated that proxy chains can survive "
                    "simple feature removal; decorrelation is required."
                ),
                recommendation=(
                    "Apply ResidualTransformer to break proxy chains, or use "
                    "adversarial representation learning."
                ),
                severity="medium" if chains else "info",
                related_metrics=["proxy_variables", "feature_correlation"],
            )
        )

    recs = [
        r.get("recommendation", str(r)) if isinstance(r, dict) else str(r)
        for r in getattr(report, "recommendations", [])
    ][:5]

    return ExplanationReport(
        title="Feature Engineering Explanation",
        summary=(
            f"Analysed {getattr(report, 'n_features', '?')} features across "
            f"{getattr(report, 'n_protected_attributes', '?')} protected attribute(s). "
            f"{n_prox} proxy variable(s) and {n_high} high-risk feature(s) identified."
        ),
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


def _explain_calibration(report: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    overall = getattr(report, "overall_metrics", {})
    ece = overall.get("ece", overall.get("expected_calibration_error"))
    # THREE STATES. ``CalibrationReport.has_significant_disparity`` is
    # Optional[bool]: None when fewer than two groups were measured, so no
    # between-group comparison was made (CAL-DISP, 2026-09-09). The report's own
    # ``summary()`` prints "Not assessable (not measured)" through
    # ``analyzer._yes_no_unknown`` and ``to_dict()`` carries the None; read
    # through `getattr(..., False)` this handler contradicted both and told the
    # reader "No significant calibration disparity."
    has_disparity = _verdict(report, "has_significant_disparity")

    # The ECE card's severity derives from the SAME 0.05/0.10/0.15 bands as
    # its own prose (_bands.calibration_ece_severity), so 'well calibrated'
    # can never carry a 'low' badge. The disparity bump applies only to the
    # report-level severity (and the disparity card), never to the ECE card,
    # whose severity must describe the ECE value itself.
    ece_sev = _sev(_bands.calibration_ece_severity(ece)) if ece is not None else "info"
    sev = ece_sev
    if has_disparity:
        sev = _worst_severity(sev, "medium")
    elif has_disparity is None:
        # A missing verdict must not be graded below a measured one: 'info'
        # here is the same unqualified all-clear a measured False earns, on a
        # comparison nobody made.
        sev = _worst_severity(sev, _UNKNOWN_SEV)

    if ece is not None:
        explanations.append(
            MetricExplanation(
                metric_name="Expected Calibration Error (ECE)",
                definition=(
                    "Average absolute difference between predicted probabilities and "
                    "observed frequencies across probability bins. Lower is better."
                ),
                interpretation_guide=(
                    "ECE < 0.05: well calibrated. 0.05-0.10: moderate miscalibration. "
                    "0.10-0.15: poor. 0.15 and above: severely miscalibrated."
                ),
                value=ece,
                evaluation=(
                    f"ECE = {ece:.4f}. "
                    + (
                        "Model is well calibrated."
                        if ece < 0.05
                        else "Moderate miscalibration detected."
                        if ece < 0.10
                        else "Significant miscalibration; consider recalibrating."
                    )
                ),
                benchmark_context=(
                    "Naeini et al. (2015) proposed ECE as the primary calibration metric. "
                    "Modern deep-learning models are often over-confident (Guo et al., 2017)."
                ),
                recommendation=(
                    "Apply Platt scaling or isotonic regression to improve calibration. "
                    "Use GroupCalibrator for per-group calibration."
                ),
                severity=ece_sev,
                related_metrics=["maximum_calibration_error", "brier_score"],
            )
        )

    if has_disparity:
        explanations.append(
            MetricExplanation(
                metric_name="Calibration Disparity Across Groups",
                definition=(
                    "Difference in calibration quality (ECE) between demographic groups. "
                    "High disparity means the model is well calibrated for some groups "
                    "but poorly calibrated for others."
                ),
                interpretation_guide=(
                    "A disparity > 0.03 is noteworthy; > 0.08 is problematic. "
                    "Pleiss et al. (2017) proved calibration and group fairness are "
                    "mathematically incompatible in some regimes."
                ),
                value="Significant disparity detected",
                evaluation=(
                    "Calibration quality varies significantly across groups. "
                    "This can cause systematically different decision outcomes."
                ),
                benchmark_context=(
                    "Kleinberg et al. (2016) formalised the impossibility result: "
                    "perfect calibration, balance for the positive class, and balance "
                    "for the negative class cannot all hold simultaneously."
                ),
                recommendation=(
                    "Use GroupCalibrator to calibrate each group independently, then "
                    "run analyze_calibration_fairness_tradeoff() to quantify the trade-off."
                ),
                severity="medium",
                related_metrics=["expected_calibration_error", "brier_score_decomposition"],
            )
        )
    elif has_disparity is None:
        explanations.append(
            MetricExplanation(
                metric_name="Calibration Disparity Across Groups",
                definition=(
                    "Difference in calibration quality (ECE) between demographic groups. "
                    "High disparity means the model is well calibrated for some groups "
                    "but poorly calibrated for others."
                ),
                interpretation_guide=(
                    "This report carries no disparity verdict, so the question is open: "
                    "it is neither a finding of disparity nor a finding of none."
                ),
                value=f"{_COULD_NOT_CHECK}: no disparity verdict was measured",
                evaluation=(
                    f"{_COULD_NOT_CHECK}: no significance verdict was measured for this "
                    "report, so no between-group comparison was made. This is not a "
                    "finding of no disparity."
                ),
                benchmark_context=(
                    "A between-group comparison needs at least two groups that each "
                    "reached the minimum sample size; below that there is nothing to "
                    "compare, and the spread is NaN rather than 0.0."
                ),
                recommendation=(
                    "Give at least two groups enough samples (or lower min_group_size) "
                    "and re-run the calibration analysis; as it stands this report "
                    "neither clears nor flags calibration disparity."
                ),
                severity=_UNKNOWN_SEV,
                related_metrics=["expected_calibration_error", "brier_score_decomposition"],
            )
        )

    recs = list(getattr(report, "recommendations", []))[:5]

    return ExplanationReport(
        title="Calibration Analysis Explanation",
        summary=(
            f"Calibration analysis for {getattr(report, 'n_groups', '?')} groups. "
            f"ECE = {ece:.4f}. "
            + (
                f"Calibration disparity {_COULD_NOT_CHECK}: no significance verdict "
                "was measured, so this is not a finding of no disparity."
                if has_disparity is None
                else "Significant calibration disparity across groups."
                if has_disparity
                else "No significant calibration disparity."
            )
        )
        if ece is not None
        else "Calibration analysis complete.",
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


def _region_status(region: Any) -> str:
    """ "feasible", "infeasible", "not_assessed" or "unknown" for a bare pair.

    A plain ``(lower, upper)`` tuple carries no evidence about which state it is
    in, so it answers "unknown" rather than being credited with either. Asserting
    "assessed" for it is the ambiguity :class:`FeasibleRegion` exists to remove.
    """
    status = getattr(region, "status", None)
    if isinstance(status, str):
        return status
    return "unknown"


def _explain_threshold_analysis(report: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    feasible = getattr(report, "feasible_regions", {})
    optimal = getattr(report, "optimal_thresholds", {})
    sev: Severity = "info"

    # BGL5 deferral from batch A-post_processing-1 (PP1-A7), closed 2026-09-28.
    #
    # `feasible` holds one entry per constraint REQUESTED, and every count below
    # printed len(feasible) as the number of constraints SATISFIABLE. A region of
    # (None, None) with assessed=False is in that dict, so a sweep in which NOTHING
    # was measurable was reported as three satisfiable constraints.
    #
    # Measured before, on 200 rows of one group where all 100 searched thresholds
    # are could-not-check:
    #   summary     "Analysed 20 threshold(s). 3 constraint(s) are satisfiable
    #                within the threshold range."
    #   severity    "info"
    #   value       "3 constraint(s) analysed, 3 feasible region(s)"
    # printed DIRECTLY ABOVE this report's own recommendation, "NOT ASSESSED:
    # demographic_parity, equalized_odds, equal_opportunity could not be evaluated
    # at any searched threshold". One surface contradicting another on the same
    # page is worse than either being wrong alone, because a reader believes the
    # one that agrees with what they hoped.
    #
    # `if not feasible: sev = "medium"` could never fire for that input either: a
    # dict of three unassessed regions is non-empty. The severity is keyed off the
    # SATISFIABLE count now, so the branch is reachable.
    by_status: Dict[str, List[str]] = {}
    for name, region in (feasible or {}).items():
        by_status.setdefault(_region_status(region), []).append(str(name))
    n_satisfiable = len(by_status.get("feasible", []))
    not_assessed = sorted(by_status.get("not_assessed", []))
    n_infeasible = len(by_status.get("infeasible", []))
    # BGL6 F04-7, 2026-09-29. THE FOURTH STATE FELL INTO NONE OF THE THREE BUCKETS
    # AND CAME OUT AS A MEASURED INFEASIBILITY. `_region_status` answers "unknown"
    # for a bare (lower, upper) pair, exactly as `_region_to_dict` does, and the
    # three reads above skip it, so n_satisfiable 0, not_assessed [] and
    # n_infeasible 0 fell through to "No feasible threshold region found for the
    # given constraints: the model may need retraining with fairness-aware methods".
    #
    # Measured before this, on one hand-built region {"demographic_parity":
    # (0.2, 0.8)}:
    #   summary     "Analysed 20 threshold(s). 0 constraint(s) are satisfiable
    #                within the threshold range."   (no mention of the region)
    #   value       "1 constraint(s) analysed, 0 feasible region(s)"
    #   evaluation  "No feasible threshold region found for the given constraints:
    #                the model may need retraining with fairness-aware methods."
    #   severity    "medium"
    # A recommendation to retrain the model, for a region whose endpoints are 0.2
    # and 0.8 and whose status nobody recorded. Before PP1-A7 the same input read
    # "1 constraint(s) are satisfiable", so the earlier fix left this input worse
    # rather than better: it stopped crediting the pair and started condemning it.
    #
    # Collected by EXCLUSION rather than by name, so a status value nobody has
    # written yet cannot land in the same hole: anything that is not one of the
    # three known states is a could-not-check and is named as one.
    _KNOWN_STATUSES = {"feasible", "not_assessed", "infeasible"}
    no_status = sorted(
        name
        for status, names in by_status.items()
        if status not in _KNOWN_STATUSES
        for name in names
    )

    if not n_satisfiable:
        sev = "medium"

    coverage_clauses = []
    if not_assessed:
        coverage_clauses.append(
            f"{len(not_assessed)} could not be evaluated at any searched threshold "
            f"({', '.join(not_assessed)})"
        )
    if no_status:
        coverage_clauses.append(
            f"{len(no_status)} region(s) carry no recorded feasibility status, so "
            f"they were neither assessed nor ruled out ({', '.join(no_status)})"
        )
    coverage_clause = f" {'; '.join(coverage_clauses)}." if coverage_clauses else ""
    n_uncovered = len(not_assessed) + len(no_status)

    explanations.append(
        MetricExplanation(
            metric_name="Threshold-Fairness Trade-off",
            definition=(
                "Analysis of how varying the classification threshold affects both "
                "performance (accuracy, F1) and fairness (demographic parity, equalized odds)."
            ),
            interpretation_guide=(
                "A feasible region is the range of thresholds satisfying a given fairness "
                "constraint. Wider feasible regions give more room to optimise performance "
                "without violating fairness."
            ),
            value=(
                f"{len(optimal)} constraint(s) analysed, "
                f"{n_satisfiable} feasible region(s)"
                + (f", {n_uncovered} NOT ASSESSED" if n_uncovered else "")
            ),
            evaluation=(
                f"{n_satisfiable} fairness constraint(s) have a feasible threshold "
                f"region.{coverage_clause}"
                if n_satisfiable
                else (
                    "NOT ASSESSED: no fairness constraint was established as "
                    f"feasible or ruled out.{coverage_clause} This is not a finding "
                    "that no region exists."
                    if n_uncovered and not n_infeasible
                    else "No feasible threshold region found for the given "
                    "constraints: the model may need retraining with fairness-aware "
                    f"methods.{coverage_clause}"
                )
            ),
            benchmark_context=(
                "Hardt et al. (2016) showed that post-hoc threshold adjustment can achieve "
                "equalized odds without retraining, though at some performance cost."
            ),
            recommendation=(
                "Use GroupThresholdOptimizer for per-group thresholds, or "
                "MultiObjectiveThresholdOptimizer to find the Pareto-optimal trade-off."
            ),
            severity=sev,
            related_metrics=["equalized_odds_difference", "demographic_parity_difference"],
        )
    )

    recs = list(getattr(report, "recommendations", []))[:5]
    return ExplanationReport(
        title="Threshold Analysis Explanation",
        summary=(
            f"Analysed {len(getattr(report, 'threshold_results', []))} threshold(s). "
            f"{n_satisfiable} constraint(s) are satisfiable within the threshold "
            f"range.{coverage_clause}"
        ),
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


def _explain_reweighting_analysis(report: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    best = getattr(report, "best_method", "unknown")
    methods = getattr(report, "method_results", [])
    sev: Severity = "info"

    explanations.append(
        MetricExplanation(
            metric_name="Prediction Reweighting Comparison",
            definition=(
                "Compares post-processing reweighting strategies that adjust predictions "
                "or decision boundaries to improve fairness without retraining."
            ),
            interpretation_guide=(
                "Each method trades off fairness improvement against performance loss. "
                "The best method maximises fairness gain per unit of performance sacrifice."
            ),
            value=f"{len(methods)} method(s) compared; best = {best}",
            evaluation=f"Recommended method: {best}.",
            benchmark_context=(
                "Kamiran et al. (2012) introduced reject-option classification; "
                "Hardt et al. (2016) formalised post-processing equalised odds."
            ),
            recommendation=(
                f"Apply {best} as the post-processing step and monitor fairness metrics "
                "continuously in production."
            ),
            severity=sev,
            related_metrics=["demographic_parity_difference", "calibration_disparity"],
        )
    )

    recs = list(getattr(report, "recommendations", []))[:5]
    return ExplanationReport(
        title="Reweighting Analysis Explanation",
        summary=(
            f"Compared {len(methods)} reweighting method(s). Best trade-off achieved by: {best}."
        ),
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


def _explain_training_report(report: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    rec = getattr(report, "recommendation", None)
    comparisons = getattr(report, "method_comparisons", [])
    baseline = getattr(report, "baseline_metrics", {})
    sev: Severity = "info"

    # Baseline fairness
    dp = baseline.get("demographic_parity_difference")
    # G12, 2026-09-30. THE GATE WAS ``is not None``, WHILE THE CANONICAL
    # PREDICATE FOR THIS QUESTION IS DEFINED FORTY LINES ABOVE IN THIS FILE, and
    # its docstring describes this exact failure at a sibling site. Every
    # comparison against NaN is False, so an unmeasured baseline fell through
    # both bands to 'info'. Measured on this function before the change:
    #   dp = 0.02  -> severity 'info', "Baseline DP = +0.0200. Baseline is
    #                 reasonably fair; light-touch intervention may suffice."
    #   dp = nan   -> severity 'info', "Baseline DP = +nan. Baseline is
    #                 reasonably fair; light-touch intervention may suffice."
    #   dp = inf   -> severity 'high', "Baseline DP = +inf. Baseline exhibits
    #                 unfairness"
    #   dp = True  -> severity 'high', "Baseline DP = +1.0000. Baseline
    #                 exhibits unfairness" (abs(True) == 1; a yes/no flag read
    #                 as the largest disparity the scale has)
    # The NaN line is the dangerous one: word for word the sentence a genuinely
    # fair baseline gets, on a report that measured nothing, and it is the
    # sentence that decides whether an intervention is applied at all.
    if dp is not None and not _measured(dp):
        sev = _worst_severity(sev, _UNKNOWN_SEV)
        explanations.append(
            MetricExplanation(
                metric_name="Baseline Fairness Assessment",
                definition=(
                    "Fairness metrics of the unconstrained baseline model before any "
                    "fairness-aware intervention is applied."
                ),
                interpretation_guide=(
                    "This report carries a baseline demographic parity difference that is "
                    "not a measurement, so the baseline is neither cleared nor flagged."
                ),
                value=f"{_COULD_NOT_CHECK}: baseline demographic parity was not measured",
                evaluation=(
                    f"{_COULD_NOT_CHECK}: the baseline demographic parity difference is "
                    f"{dp!r} ({unmeasurable_reason(dp) or 'not a number'}), so it was never "
                    f"compared to a band. This is NOT a reasonably fair baseline and NOT an "
                    f"unfair one."
                ),
                benchmark_context=(
                    "The 4/5 rule (EEOC) requires selection rates within 80%. A parity "
                    "difference can only be read against it once it has a value."
                ),
                recommendation=(
                    "Re-run the baseline fairness evaluation on data where the metric is "
                    "defined (both groups present and non-empty) before choosing an "
                    "intervention."
                ),
                severity=_UNKNOWN_SEV,
                related_metrics=["equalized_odds_difference", "equal_opportunity_difference"],
            )
        )
    elif dp is not None:
        base_sev: Severity = "high" if abs(dp) > 0.15 else ("medium" if abs(dp) > 0.08 else "info")
        sev = _worst_severity(sev, base_sev)
        explanations.append(
            MetricExplanation(
                metric_name="Baseline Fairness Assessment",
                definition=(
                    "Fairness metrics of the unconstrained baseline model before any "
                    "fairness-aware intervention is applied."
                ),
                interpretation_guide=(
                    # Bands mirror base_sev exactly (|DP| <= 0.08 info, 0.08-0.15
                    # medium, > 0.15 high) so the guide, the severity, and the
                    # evaluation prose cannot disagree at a boundary.
                    "Demographic parity difference |DP| <= 0.08: reasonably fair. "
                    "0.08-0.15: borderline. > 0.15: likely unfair. "
                    "This baseline motivates the choice of intervention."
                ),
                value=dp,
                # Text derives from base_sev, not a second |DP| comparison, so
                # boundary values (exactly 0.08) cannot say 'unfair' at severity
                # 'info': one predicate drives both label and prose.
                evaluation=(
                    f"Baseline DP = {dp:+.4f}. "
                    + (
                        "Baseline is reasonably fair; light-touch intervention may suffice."
                        if base_sev == "info"
                        else "Baseline exhibits unfairness; training-time intervention is recommended."
                    )
                ),
                benchmark_context=(
                    "The 4/5 rule (EEOC) requires selection rates within 80%. "
                    "Agarwal et al. (2018) showed reductions approaches can satisfy constraints "
                    "with minimal accuracy loss."
                ),
                recommendation=(
                    "If baseline is unfair, apply ExponentiatedGradient or FairClassifier."
                ),
                severity=base_sev,
                related_metrics=["equalized_odds_difference", "equal_opportunity_difference"],
            )
        )

    # Method comparison
    if comparisons:
        satisfied = [m for m in comparisons if getattr(m, "constraint_satisfied", False)]
        # Roll the comparison card's severity into the report severity: a
        # report must never be labelled below its own worst card (e.g. 'info'
        # while the comparison card says 'medium' because nothing satisfies
        # the constraint).
        comp_sev: Severity = "low" if satisfied else "medium"
        sev = _worst_severity(sev, comp_sev)
        explanations.append(
            MetricExplanation(
                metric_name="Training Method Comparison",
                definition=(
                    "Comparison of fairness-aware training methods on accuracy, fairness "
                    "violation, and constraint satisfaction."
                ),
                interpretation_guide=(
                    "Methods satisfying the fairness constraint are preferred. Among those, "
                    "choose the one with highest accuracy."
                ),
                value=f"{len(comparisons)} methods compared, {len(satisfied)} satisfy constraint",
                evaluation=(
                    f"{len(satisfied)}/{len(comparisons)} methods satisfy the fairness constraint."
                ),
                benchmark_context=(
                    "Agarwal et al. (2018) and Zafar et al. (2017) provide theoretical "
                    "guarantees for different constraint-satisfaction approaches."
                ),
                recommendation=(
                    getattr(rec, "rationale", "Use the recommended method.")
                    if rec
                    else "Select the method with the best fairness-accuracy trade-off."
                ),
                severity=comp_sev,
                related_metrics=["accuracy", "fairness_violation"],
            )
        )

    # Recommendation
    if rec:
        explanations.append(
            MetricExplanation(
                metric_name="Recommended Approach",
                definition="The analyser's recommended fairness-aware training method.",
                interpretation_guide="The recommendation considers accuracy, constraint satisfaction, and implementation complexity.",
                value=getattr(rec, "recommended_method", str(rec)),
                evaluation=getattr(rec, "rationale", ""),
                benchmark_context=getattr(rec, "implementation_notes", ""),
                recommendation=getattr(rec, "rationale", str(rec)),
                severity="info",
                related_metrics=getattr(rec, "alternative_methods", []),
            )
        )

    recs = list(getattr(report, "action_items", []))[:5]
    # G12, 2026-09-30. NOT ONE CARD AND SEVERITY 'info' IS A CLEAN BILL OVER
    # NOTHING. All three sections above are conditional, so a training report
    # carrying no baseline metric, no method comparison and no recommendation --
    # the exact shape tests/test_adapters_training_empty.py says the exhaustive
    # sweep calls with -- produced: explanations=[], severity='info',
    # recommendations=[], summary "Compared 0 training method(s) for the
    # classification task." 'info' is the band a genuinely fair, fully analysed
    # run gets, and this was the only entry point in this module that could
    # return an empty explanation list. Same handling as the precedent next door
    # (``detector._qualify_explanation``, BGL g021, recorded in
    # tests/test_aggregate_explanations_withhold_a_verdict.py): the aggregate
    # verdict is withheld and the severity raised off 'info' to this module's own
    # floor band for an unmeasured verdict, while nothing that WAS measured moves.
    summary = (
        f"Compared {len(comparisons)} training method(s) for the "
        f"{getattr(report, 'task_type', 'classification')} task. "
        + (f"Recommended: {getattr(rec, 'recommended_method', 'N/A')}." if rec else "")
    )
    if not explanations:
        sev = _worst_severity(sev, _UNKNOWN_SEV)
        summary += (
            f" {_COULD_NOT_CHECK}: this report carried no baseline metric, no method "
            f"comparison and no recommendation, so nothing was explained and no fairness "
            f"verdict was reached. This is not a finding that the training was fair."
        )
    return ExplanationReport(
        title="Fairness Training Analysis Explanation",
        summary=summary,
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


def _explain_drift_result(result: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    # THREE STATES. ``MultiscaleDriftResult.drift_detected`` is Optional[bool]
    # and its own field comment says None means no scale could be computed, so
    # drift was NOT LOOKED FOR: neither drift found nor no drift. Read through
    # `getattr(..., False)` this handler told the reader "Drift not detected"
    # and "No immediate action required. Continue standard monitoring." at
    # severity 'info', for a run in which no drift test ran, while
    # ``reports._drift_verdict_word`` printed COULD NOT CHECK for the same
    # object. Reproduced 2026-09-10 with a two-point series.
    detected = _verdict(result, "drift_detected")
    score = getattr(result, "overall_drift_score", 0.0)
    # Severity follows the SAME 0.3/0.6 bands as the SVG drift colouring
    # (rendering.adapters_monitoring._drift_color), so the text can never
    # call amber-charted drift 'low' or red-charted drift 'medium'. With no
    # verdict there is no band to sit in: the score is the aggregate of
    # nothing, so the card floors at the unresolved-unknown band instead.
    sev = _UNKNOWN_SEV if detected is None else _sev(_bands.drift_severity(score, detected))

    explanations.append(
        MetricExplanation(
            metric_name="Fairness Drift Score",
            definition=(
                "Aggregate drift score combining Kolmogorov-Smirnov tests across "
                "multiple temporal scales (via wavelet decomposition). Higher values "
                "indicate greater divergence from the baseline distribution."
            ),
            interpretation_guide=(
                "< 0.30: no actionable drift, monitor as usual. 0.30-0.60: moderate "
                "drift, investigate the cause. >= 0.60: severe drift, intervene "
                "(retrain or recalibrate)."
            ),
            value=score,
            evaluation=(
                f"{_COULD_NOT_CHECK}: no temporal scale was analysed, so no drift "
                "test ran and there is no verdict. This is not a finding of stability."
                if detected is None
                else f"Drift {'DETECTED' if detected else 'not detected'} (score = {score:.4f})."
            ),
            benchmark_context=(
                "Rabanser et al. (2019) recommend multi-scale drift detection to catch "
                "both sudden distribution shifts and gradual concept drift."
            ),
            recommendation=(
                "Collect a longer series and re-run the drift check; as it stands "
                "this result neither confirms nor rules out drift."
                if detected is None
                else "Investigate root cause: data pipeline change, population shift, or "
                "feedback loop. Consider retraining or recalibrating."
                if detected
                else "No immediate action required. Continue standard monitoring."
            ),
            severity=sev,
            related_metrics=["mmd_score", "ks_statistic"],
        )
    )

    # Per-scale details
    scales = getattr(result, "scales", {})
    for scale_name, dr in scales.items():
        if getattr(dr, "drift_detected", False):
            explanations.append(
                MetricExplanation(
                    metric_name=f"Drift at Scale: {scale_name}",
                    definition=f"KS test result for the {scale_name} component of the wavelet decomposition.",
                    interpretation_guide="KS statistic measures the maximum CDF divergence; p-value < 0.05 indicates significant drift.",
                    value=f"KS={getattr(dr, 'ks_statistic', 0):.4f}, p={getattr(dr, 'p_value', 1):.4f}",
                    evaluation=f"Mean shift: {getattr(dr, 'mean_shift', 0):+.4f} at scale {scale_name}.",
                    benchmark_context="Wavelet decomposition separates long-term trends from short-term fluctuations.",
                    recommendation=f"Scale {scale_name} shows drift; compare with other scales to determine if drift is sudden or gradual.",
                    severity="medium",
                    related_metrics=["overall_drift_score"],
                )
            )

    worst = getattr(result, "worst_scale", None)
    recs = []
    if detected is None:
        # Word for word the sentence `reports.generate_drift_report` writes for
        # the same object, so the two surfaces about one result agree.
        recs.append(
            f"{_COULD_NOT_CHECK}: no temporal scale was analysed, so the overall "
            "drift score is the aggregate of nothing and no scale was compared "
            "to a reference. This is not a finding of stability. Collect a "
            "longer series and re-run the drift check."
        )
    elif detected:
        recs.append(f"Drift detected (score {score:.3f}). Investigate root cause.")
        if worst:
            recs.append(
                f"Worst scale: {getattr(worst, 'scale', '?')} (mean shift {getattr(worst, 'mean_shift', 0):+.4f})."
            )
        recs.append("Consider retraining or recalibrating the model.")

    # Report severity is the worst across ALL cards: the aggregate score can
    # sit below 0.3 ('low') while an individual scale still shows detected
    # drift ('medium'), and the report label must not undercut that card.
    overall_sev = _worst_severity(*(e.severity for e in explanations))

    return ExplanationReport(
        title="Fairness Drift Detection Explanation",
        summary=(
            f"Drift {_COULD_NOT_CHECK}: no drift test ran, so there is no verdict "
            f"and no overall score. Analysed {len(scales)} temporal scale(s)."
            if detected is None
            else (
                f"Drift {'detected' if detected else 'not detected'} with overall score "
                f"{score:.4f}. Analysed {len(scales)} temporal scale(s)."
            )
        ),
        explanations=explanations,
        severity=overall_sev,
        recommendations=recs,
    )


def _explain_window_metrics(snap: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    metrics = getattr(snap, "metrics", {})
    alerts = getattr(snap, "alerts", [])
    # ``WindowMetrics.alerts`` maps metric -> BREACHED, so its length is the
    # number of metrics COMPARED to a threshold, not the number that fired: a
    # window with one breach and one clean comparison reported "2 alert(s)
    # fired". Count the breaches. A plain sequence (a duck-typed snapshot that
    # carries alert objects) keeps its old meaning, every entry being an alert.
    fired = (
        [key for key, breached in alerts.items() if breached]
        if hasattr(alerts, "items")
        else list(alerts)
    )
    # THREE STATES. ``WindowMetrics.any_alert`` is Optional[bool]: None means
    # NOTHING in this window was compared to a threshold (R-3, 2026-09-09), so
    # there is no verdict. `getattr(snap, "any_alert", bool(alerts))` read that
    # None as the measured all-clear of a monitor that looked and found nothing
    # wrong, and answered "no alerts" with "No action needed.".
    raw_any_alert = getattr(snap, "any_alert", _MISSING)
    if raw_any_alert is _MISSING:
        # A duck-typed snapshot with no roll-up of its own: derive the verdict
        # from the comparisons it does carry, and stay could-not-check when it
        # carries none. `bool(alerts)` used to stand in here, which called a
        # single CLEAN comparison an active alert.
        any_alert: Optional[bool] = bool(fired) if alerts else None
    else:
        any_alert = None if raw_any_alert is None else bool(raw_any_alert)
    sev: Severity = "high" if any_alert else (_UNKNOWN_SEV if any_alert is None else "info")

    for name, val in metrics.items():
        if isinstance(val, (int, float)):
            explanations.append(
                MetricExplanation(
                    metric_name=name,
                    definition=f"Sliding-window value for {name} over the current monitoring window.",
                    interpretation_guide="Compare against baseline and alert thresholds.",
                    value=val,
                    evaluation=f"{name} = {val:.4f}",
                    benchmark_context="Continuous fairness monitoring treats fairness as a first-class operational metric.",
                    recommendation="Investigate if value deviates significantly from baseline.",
                    severity="info",
                    related_metrics=[],
                )
            )

    if any_alert:
        explanations.append(
            MetricExplanation(
                metric_name="Alert Status",
                definition="Whether the current monitoring window triggered fairness alerts.",
                interpretation_guide="An alert fires when a metric crosses the configured threshold.",
                value=f"{len(fired)} alert(s) active",
                evaluation=f"{len(fired)} alert(s) fired in the current window.",
                benchmark_context="Alert fatigue is real; use AdaptiveThresholdManager to auto-tune thresholds.",
                recommendation="Review alerts, acknowledge valid ones, dismiss false positives to tune thresholds.",
                severity="high",
                related_metrics=["disparate_impact", "demographic_parity"],
            )
        )
    elif any_alert is None:
        explanations.append(
            MetricExplanation(
                metric_name="Alert Status",
                definition="Whether the current monitoring window triggered fairness alerts.",
                interpretation_guide=(
                    "A window with no threshold comparison in it has no alert verdict: "
                    "it is neither a breach nor an all-clear."
                ),
                value=f"{_COULD_NOT_CHECK}: no alert verdict for this window",
                evaluation=(
                    f"{_COULD_NOT_CHECK}: nothing in this window was compared to a "
                    "threshold, so no alert could fire and none could be ruled out. "
                    "This is not a finding of a clean window."
                ),
                benchmark_context=(
                    "A window computes no fairness metric when no protected column is "
                    "found or every group falls below the minimum sample size."
                ),
                recommendation=(
                    "Check that the batch carries a protected-attribute column with "
                    "enough rows per group, then re-run; as it stands this window "
                    "certifies nothing."
                ),
                severity=_UNKNOWN_SEV,
                related_metrics=["disparate_impact", "demographic_parity"],
            )
        )

    return ExplanationReport(
        title="Fairness Monitor Window Explanation",
        summary=(
            f"Window contains {len(metrics)} metric(s); {len(fired)} alert(s) active."
            if any_alert
            else f"Window contains {len(metrics)} metric(s); alert status "
            f"{_COULD_NOT_CHECK}: nothing was compared to a threshold."
            if any_alert is None
            else f"Window contains {len(metrics)} metric(s); no alerts."
        ),
        explanations=explanations,
        severity=sev,
        recommendations=[
            "Investigate and resolve active alerts."
            if any_alert
            else "Give the window a protected-attribute column with enough rows per "
            "group and re-run; no threshold comparison was made."
            if any_alert is None
            else "No action needed."
        ],
    )


def _explain_validation_result(result: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    passed = getattr(result, "passed", True)
    issues = getattr(result, "issues", [])
    sev: Severity = "info" if passed else "high"

    n_errors = sum(
        1 for i in issues if _enum_str(getattr(i, "severity", "")).lower() in ("error", "critical")
    )
    n_warnings = sum(
        1 for i in issues if _enum_str(getattr(i, "severity", "")).lower() == "warning"
    )

    explanations.append(
        MetricExplanation(
            metric_name="Data Validation Result",
            definition=(
                "Pre-training data validation checking representation balance, outcome "
                "disparities, missing-value patterns, and label quality."
            ),
            interpretation_guide=(
                "PASS: data meets all configured thresholds. FAIL: at least one error-level "
                "issue detected. Warnings are informational but should be reviewed."
            ),
            value="PASS" if passed else "FAIL",
            evaluation=(
                f"Validation {'PASSED' if passed else 'FAILED'}. "
                f"{n_errors} error(s), {n_warnings} warning(s)."
            ),
            benchmark_context=(
                "Data validation is the first gate in a responsible AI pipeline. "
                "Catching bias pre-training is cheaper than post-deployment remediation."
            ),
            recommendation=(
                "All checks passed; proceed to training with confidence."
                if passed
                else "Resolve error-level issues before training. Review warnings."
            ),
            severity=sev,
            related_metrics=["representation_bias", "outcome_disparity"],
        )
    )

    # Add top issues
    for issue in issues[:3]:
        # _enum_str already unwraps Enum severities to their string value,
        # so no further .value unwrap is needed here. Validation uses the
        # vocabulary info/warning/error/critical; map the two terms that are
        # not explainer severities (error, warning) so an ERROR issue reads
        # as 'high' instead of falling through to the 'medium' default.
        i_sev = _enum_str(getattr(issue, "severity", "info")).lower()
        e_sev = _sev(
            {"error": "high", "warning": "medium"}.get(
                i_sev, i_sev if i_sev in _SEVERITY_ORDER else "medium"
            )
        )
        explanations.append(
            MetricExplanation(
                metric_name=f"Issue: {getattr(issue, 'issue_type', 'unknown')}",
                definition=getattr(issue, "message", ""),
                interpretation_guide=f"Severity: {i_sev}.",
                value=i_sev,
                evaluation=getattr(issue, "message", ""),
                benchmark_context="",
                recommendation=getattr(issue, "recommendation", "Review this issue.")
                or "Review this issue.",
                severity=e_sev,
                related_metrics=[],
            )
        )

    return ExplanationReport(
        title="Data Validation Explanation",
        summary=(
            f"Validation {'PASSED' if passed else 'FAILED'}. "
            f"{len(issues)} issue(s) found: {n_errors} errors, {n_warnings} warnings."
        ),
        explanations=explanations,
        severity=sev,
        recommendations=["Fix all error-level issues before proceeding to model training."]
        if not passed
        else ["Validation passed."],
    )


def _explain_gate_decision(decision: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    approved = getattr(decision, "approved", False)
    status = getattr(decision, "status", None)
    evals = getattr(decision, "metric_evaluations", [])
    sev: Severity = "info" if approved else "high"

    status_str = (
        getattr(status, "value", str(status)) if status else ("approved" if approved else "blocked")
    )

    explanations.append(
        MetricExplanation(
            metric_name="Deployment Gate Decision",
            definition=(
                "Binary decision on whether the model meets fairness requirements for "
                "deployment. Evaluates configured metrics against thresholds."
            ),
            interpretation_guide=(
                "APPROVED: all blocking metrics pass. BLOCKED: at least one blocking "
                "metric fails. CONDITIONAL: non-blocking metrics have warnings."
            ),
            value=status_str.upper(),
            evaluation=f"Gate status: {status_str.upper()}.",
            benchmark_context=(
                "Deployment gates enforce organisational fairness standards. "
                "Google's Model Cards and Microsoft's Responsible AI Standard "
                "recommend automated fairness gates in CI/CD."
            ),
            recommendation=(
                "Model is cleared for deployment. Continue monitoring in production."
                if approved
                else "Model is blocked. Address the failing metrics before re-evaluating."
            ),
            severity=sev,
            related_metrics=[],
        )
    )

    # Per-metric evaluations
    for me in evals:
        m_passed = getattr(me, "passed", True)
        explanations.append(
            MetricExplanation(
                metric_name=f"Gate Metric: {getattr(me, 'metric_name', '?')}",
                definition=f"Evaluation of {getattr(me, 'metric_name', '?')} against threshold.",
                interpretation_guide=f"Threshold: {getattr(me, 'threshold', 'N/A')}. Blocking: {getattr(me, 'is_blocking', True)}.",
                value=getattr(me, "value", None),
                evaluation=getattr(me, "message", f"{'PASS' if m_passed else 'FAIL'}"),
                benchmark_context="",
                recommendation="No action."
                if m_passed
                else f"Improve {getattr(me, 'metric_name', '?')} to meet the threshold.",
                severity="info"
                if m_passed
                else ("high" if getattr(me, "is_blocking", True) else "medium"),
                related_metrics=[],
            )
        )

    blocking = list(getattr(decision, "blocking_reasons", []))
    return ExplanationReport(
        title="Fairness Gate Explanation",
        summary=(
            f"Gate status: {status_str.upper()}. "
            f"{sum(1 for e in evals if getattr(e, 'passed', True))}/{len(evals)} metrics pass."
        ),
        explanations=explanations,
        severity=sev,
        recommendations=blocking if blocking else ["All metrics pass."],
    )


# ExperimentResult (Unit 4: A/B Testing for Fairness)


def _explain_experiment_result(result: Any) -> ExplanationReport:
    explanations: List[MetricExplanation] = []
    # THREE STATES. ``ExperimentResult.heterogeneity_detected`` is
    # Optional[bool]: None when the heterogeneity test could not run at all, and
    # the object's own ``__repr__`` prints "heterogeneity=not assessed" through
    # ``experiment._yes_no_not_assessed``. Read through `getattr(..., False)`
    # this handler answered "Heterogeneity not detected" and "Effects are
    # consistent; safe to deploy uniformly." for an experiment that ran no such
    # test. ``IntersectionEffect.significant`` is Optional[bool] for the same
    # reason (None when the p-value is not finite, so no test ran on that
    # intersection), and an untested intersection must not be counted as a
    # cleared one.
    het = _verdict(result, "heterogeneity_detected")
    het_p = getattr(result, "heterogeneity_p_value", 1.0)
    overall = getattr(result, "overall_effect", 0.0)
    overall_p = getattr(result, "overall_p_value", 1.0)
    # ``overall_p_value`` is a THREE-STATE float, not a two-state one: the
    # producer sets NaN when both arms are constant and no test could be run.
    # See `_measured` for the reading this used to give.
    overall_p_measured = _measured(overall_p)
    overall_significant = overall_p_measured and float(overall_p) < 0.05
    effects = getattr(result, "intersection_effects", [])
    n_ix = getattr(result, "n_intersections", len(effects))
    n_untested = sum(1 for e in effects if _verdict(e, "significant") is None)

    # Report-level severity based on heterogeneity
    sev: Severity
    if het and any(
        getattr(e, "effect", 0) < 0 and _verdict(e, "significant") is True for e in effects
    ):
        sev = "high"
    elif het:
        sev = "medium"
    elif overall_significant:
        sev = "low"
    else:
        sev = "info"
    if het is None or n_untested or not overall_p_measured:
        # No verdict is not the all-clear a measured 'no heterogeneity' earns,
        # and neither is a set of intersections whose tests never ran, nor an
        # overall effect no test was run on.
        sev = _worst_severity(sev, _UNKNOWN_SEV)

    # The overall-effect card is labelled from its OWN values (effect
    # significance), not the report-level worst-item severity: heterogeneity
    # and harmed groups carry their own cards, and borrowing their severity
    # here would contradict this card's text.
    overall_card_sev: Severity = (
        "low" if overall_significant else ("info" if overall_p_measured else _UNKNOWN_SEV)
    )

    # Overall effect
    explanations.append(
        MetricExplanation(
            metric_name="Overall Treatment Effect",
            definition=(
                "Average difference between treatment and control outcomes, "
                "aggregated across all demographic groups."
            ),
            interpretation_guide=(
                "A positive value indicates the treatment improves the outcome on "
                "average.  Statistical significance (p < 0.05) provides evidence "
                "that the effect is real, not due to chance."
            ),
            value=overall,
            evaluation=(
                f"Effect = {overall:+.4f}, p = {overall_p:.4f} "
                f"({'significant' if overall_significant else 'not significant'})."
                if overall_p_measured
                else (
                    f"Effect = {overall:+.4f}. Significance {_COULD_NOT_CHECK}: no overall "
                    "test was run, so no p-value was produced. This is not a finding that "
                    "the effect is real, and it is not a finding that it is absent."
                )
            ),
            benchmark_context=(
                "Compare with the per-intersection effects below to understand "
                "whether the overall average masks heterogeneity across groups."
            ),
            recommendation=(
                "Examine per-intersection effects before deploying."
                if het or het is None
                else "Effect is consistent across groups."
            ),
            severity=overall_card_sev,
            related_metrics=["heterogeneity_p_value"],
        )
    )

    # Heterogeneity
    explanations.append(
        MetricExplanation(
            metric_name="Heterogeneity Test",
            definition=(
                "One-way ANOVA (or Kruskal-Wallis) test of whether treatment "
                "effects differ significantly across demographic intersections."
            ),
            interpretation_guide=(
                "p < 0.05 indicates that the treatment does not affect all "
                "groups equally: some benefit more or less than others."
            ),
            value=het_p,
            evaluation=(
                f"Heterogeneity {_COULD_NOT_CHECK}: the test did not run, so no "
                f"p-value was produced across {n_ix} intersections. This is not a "
                "finding of consistent effects."
                if het is None
                else (
                    f"Heterogeneity {'DETECTED' if het else 'not detected'} "
                    f"(p = {het_p:.4f}) across {n_ix} intersections."
                )
            ),
            benchmark_context=(
                "Athey & Imbens (2017) recommend routinely checking for "
                "heterogeneous treatment effects in controlled experiments."
            ),
            recommendation=(
                "Re-run the heterogeneity test on intersections with enough samples "
                "before deploying uniformly; as it stands consistency across groups "
                "has not been established."
                if het is None
                else "Treatment effects vary across groups: consider group-specific "
                "deployment or further investigation."
                if het
                else "Effects are consistent; safe to deploy uniformly."
            ),
            severity=_UNKNOWN_SEV if het is None else ("medium" if het else "info"),
            related_metrics=["overall_treatment_effect"],
        )
    )

    # Per-intersection highlights (significant only). An intersection whose
    # significance verdict is None had no test run on it, so it is neither
    # confirmed nor cleared: it stays out of `sig_effects` (there is no evidence
    # of harm) and is counted in `n_untested`, which is stated in the summary
    # and floors the report severity above the all-clear.
    sig_effects = [e for e in effects if _verdict(e, "significant") is True]
    harmed = [e for e in sig_effects if getattr(e, "effect", 0) < 0]

    if harmed:
        for e in harmed:
            ix = getattr(e, "intersection", ("unknown",))
            explanations.append(
                MetricExplanation(
                    metric_name=f"Harmed Group: {ix}",
                    definition="Intersection where the treatment has a significant negative effect.",
                    interpretation_guide="Negative effect means the treatment worsens outcomes for this group.",
                    value=getattr(e, "effect", 0),
                    evaluation=(
                        f"Effect = {getattr(e, 'effect', 0):+.4f}, "
                        f"d = {getattr(e, 'effect_size_d', 0):.3f}, "
                        f"p = {getattr(e, 'p_value', 1):.4f}."
                    ),
                    benchmark_context="|d| > 0.2 is a small effect, > 0.5 medium, > 0.8 large.",
                    recommendation=f"Investigate why {ix} is negatively affected before deploying.",
                    severity="high",
                    related_metrics=["overall_treatment_effect"],
                )
            )

    recs = []
    if het is None:
        recs.append(
            f"{_COULD_NOT_CHECK}: the heterogeneity test did not run, so it is "
            "unknown whether the treatment affects all groups equally. This is "
            "not a finding of consistent effects."
        )
    elif het:
        recs.append(f"Heterogeneous effects detected (p={het_p:.4f}). Review per-group results.")
    if harmed:
        recs.append(f"{len(harmed)} group(s) harmed by treatment. Investigate before deploying.")
    if n_untested:
        recs.append(
            f"{_COULD_NOT_CHECK}: {n_untested} intersection(s) reported no significance "
            "verdict, so they were neither confirmed as harmed nor cleared."
        )
    if not overall_p_measured:
        # The NaN branch, which `overall_p >= 0.05` could never reach: it is
        # False for NaN, so an experiment that ran NO overall test fell through
        # to the `if not recs` fallback and was told "Results look consistent.
        # Proceed with deployment review." See `_measured`.
        recs.append(
            f"{_COULD_NOT_CHECK}: no overall significance test was run, so it is unknown "
            "whether the treatment effect is real. This is not a finding of consistency."
        )
    elif float(overall_p) >= 0.05:
        recs.append("Overall effect not significant. Consider extending the experiment.")
    if not recs:
        recs.append("Results look consistent. Proceed with deployment review.")

    return ExplanationReport(
        title="A/B Experiment Explanation",
        summary=(
            (
                f"Overall effect {overall:+.4f} (p={overall_p:.4f}). "
                if overall_p_measured
                else f"Overall effect {overall:+.4f} (significance: not assessed). "
            )
            + f"{n_ix} intersections analysed. "
            + f"Heterogeneity: {'not assessed' if het is None else ('YES' if het else 'no')}."
            + _unknown_tail(n_untested, "intersection(s) reported no significance verdict")
        ),
        explanations=explanations,
        severity=sev,
        recommendations=recs,
    )


# FairnessExplainer: the public facade


class FairnessExplainer:
    """Unified facade for explaining any vfairness result object.

    Uses a **registry pattern**: result types are matched by class name,
    avoiding circular imports.  Each major class also exposes a convenience
    ``get_explanation()`` method that delegates here.

    Examples
    --------
    >>> from vfairness.explainer import FairnessExplainer
    >>> report = bias_detector.full_audit()
    >>> explanation = FairnessExplainer.explain(report)
    >>> print(explanation)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: fairness_explainer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    _handlers: Dict[str, Callable[..., ExplanationReport]] = {
        "BiasAuditReport": _explain_bias_audit,
        "FeatureAnalysisReport": _explain_feature_analysis,
        "CalibrationReport": _explain_calibration,
        "ThresholdAnalysisReport": _explain_threshold_analysis,
        "ReweightingAnalysisReport": _explain_reweighting_analysis,
        "FairnessTrainingReport": _explain_training_report,
        "MultiscaleDriftResult": _explain_drift_result,
        "WindowMetrics": _explain_window_metrics,
        "DataValidationResult": _explain_validation_result,
        "GateDecision": _explain_gate_decision,
        "ExperimentResult": _explain_experiment_result,
    }

    # Public API

    @classmethod
    def explain(cls, obj: Any, **kwargs: Any) -> ExplanationReport:
        """Generate an educational explanation for *obj*.

        Parameters
        ----------
        obj : Any
            A result object produced by a vfairness analyzer or detector.
            Supported types: BiasAuditReport, FeatureAnalysisReport,
            CalibrationReport, ThresholdAnalysisReport,
            ReweightingAnalysisReport, FairnessTrainingReport,
            MultiscaleDriftResult, WindowMetrics, DataValidationResult,
            GateDecision.

            If *obj* has a ``get_explanation()`` method, it is called directly.

        Returns
        -------
        ExplanationReport
        """
        # 1. Try the registry by class name (+ MRO)
        for klass in type(obj).__mro__:
            handler = cls._handlers.get(klass.__name__)
            if handler is not None:
                return handler(obj, **kwargs)

        # 2. Duck-type: if the object itself has get_explanation()
        if hasattr(obj, "get_explanation"):
            return obj.get_explanation(**kwargs)

        raise TypeError(
            f"FairnessExplainer does not know how to explain "
            f"{type(obj).__name__!r}. Supported types: "
            f"{', '.join(sorted(cls._handlers))}."
        )

    @classmethod
    def can_explain(cls, obj: Any) -> bool:
        """Return True if *obj* is a supported result type."""
        for klass in type(obj).__mro__:
            if klass.__name__ in cls._handlers:
                return True
        return hasattr(obj, "get_explanation")

    @classmethod
    def register(cls, type_name: str, handler: Callable[..., ExplanationReport]) -> None:
        """Register a custom handler for a result type name."""
        cls._handlers[type_name] = handler

    @classmethod
    def registered_types(cls) -> List[str]:
        """Return a sorted list of registered type names."""
        return sorted(cls._handlers)
