"""
FairExplAIner - Intelligent Explanations for Fairness Metrics.

This module provides comprehensive, context-aware explanations for fairness
metrics and statistical deliverables. When activated, it explains:
- What each metric measures
- How to interpret the values
- What the actual result means based on common benchmarks
- Actionable recommendations based on the findings

Usage:
    >>> from vfairness import FairnessAnalyzer
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
    >>> report = analyzer.get_report(include_ci=True)
    >>> # Report now includes 'explanations' section with detailed explanations

Or enable/disable dynamically:
    >>> analyzer.enable_fair_explainer()
    >>> analyzer.disable_fair_explainer()
"""

import numbers
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ..._triage import is_measured
from ._metric_direction import MetricDirection, _normalize, metric_direction

# Formatting / serialising the value on an explanation card


def _value_is_absent(value: Any) -> bool:
    """True when ``value`` carries no measurement at all.

    BGL grade-1 G01, 2026-09-30. :class:`MetricExplanation` decided this with
    ``isinstance(self.value, float) and np.isnan(self.value)``, in two places
    (``to_dict`` and ``_format_value``), and that is the canonical
    ``isinstance(v, (int, float))`` hole recorded in ``vfairness/_triage.py``:
    ``np.float64`` happens to subclass ``float`` and every other numpy scalar
    does not. Measured on this repo through the PUBLIC entry point
    ``FairExplAIner.explain_metric("demographic_parity_difference", v)``, whose
    ``severity`` and ``evaluation`` were already correct in all seven rows:

        v                 severity         to_dict()['value']    __str__ shows
        float('nan')      could_not_check  None                  N/A (insufficient data)
        np.float64 nan    could_not_check  None                  N/A (insufficient data)
        np.float32 nan    could_not_check  np.float32(nan)       nan
        pd.NA             could_not_check  <NA>                  <NA>
        pd.NaT            could_not_check  NaT                   NaT
        None              could_not_check  None                  None
        float('inf')      could_not_check  inf                   inf

    So the two halves of one card disagreed about the same value: the
    ``evaluation`` string said "COULD NOT CHECK: this metric was never computed"
    while the ``value`` immediately above it published ``<NA>``, ``NaT``,
    ``nan`` or ``inf`` as the result, and ``str(x)`` on an absent value MINTED
    the content ("YOUR RESULT: <NA>"). ``json.dumps`` of such a card raises
    ``TypeError: Object of type float32 is not JSON serializable``, so a
    consumer either crashes or, with a permissive encoder, is handed the
    non-JSON tokens ``NaN`` / ``Infinity``.

    Infinity belongs here for the reason ``_triage`` rule 1 gives: it is not a
    measurement, and on a magnitude scale it reads as the largest one there is.

    Deliberately NOT treated as absent: a genuine string, a dict or a list.
    ``value`` is typed ``Any`` and a statistical-measure card can legitimately
    carry non-numeric content, so the blank-string door is left to the producer;
    refusing every falsy value here would blank out real content, which is the
    over-correction. A bool is also left alone, because ``value`` is never a
    flag on this surface and ``str(True)`` mints nothing.
    """
    if value is None:
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (numbers.Real, np.number)):
        # A real number that is_measured refuses is NaN or an infinity.
        return not is_measured(value)
    # pd.NA / pd.NaT are not numbers.Real, so they need the pandas answer.
    # Imported lazily: this module is otherwise pandas-free at import time.
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, TypeError, ValueError):
        # Cannot ask, so make no claim: an unrecognised object is content, and
        # calling it absent would delete it from the card.
        return False


@dataclass
class MetricExplanation:
    """
    Structured explanation for a fairness metric or statistical measure.

    Attributes:
        metric_name: Name of the metric
        definition: What the metric measures
        interpretation_guide: How to interpret values
        value: The actual computed value
        evaluation: Context-aware evaluation of the result
        benchmark_context: How this compares to common benchmarks
        recommendation: Actionable recommendation based on the result
        severity: Severity level. Five GRADED levels ('info', 'low', 'medium',
            'high', 'critical') plus 'could_not_check'.

            'could_not_check' is a THIRD state and must never be collapsed into
            either of the other two readings. It means no threshold was applied
            to this metric at all, so it is neither a pass nor a fail. It exists
            because 'info' is what a genuinely graded, genuinely BENIGN metric
            receives: reusing 'info' for an ungraded metric makes the two
            indistinguishable to every reader and every consumer that colours or
            counts by severity. Consumers that rank severities must treat it as
            "unknown", never as the bottom of the scale.
        related_metrics: Other metrics to consider alongside this one
    """

    metric_name: str
    definition: str
    interpretation_guide: str
    value: Any
    evaluation: str
    benchmark_context: str
    recommendation: str
    severity: Literal["info", "low", "medium", "high", "critical", "could_not_check"] = "info"
    related_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "definition": self.definition,
            "interpretation_guide": self.interpretation_guide,
            # None when nothing was measured, for EVERY absence door rather
            # than for python-float NaN only; see _value_is_absent. This field
            # sits beside a `severity` of 'could_not_check' and the two must
            # not disagree.
            "value": None if _value_is_absent(self.value) else self.value,
            "evaluation": self.evaluation,
            "benchmark_context": self.benchmark_context,
            "recommendation": self.recommendation,
            "severity": self.severity,
            "related_metrics": self.related_metrics,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"═══ {self.metric_name} ═══",
            "",
            "📖 DEFINITION:",
            f"   {self.definition}",
            "",
            "📊 INTERPRETATION:",
            f"   {self.interpretation_guide}",
            "",
            f"📈 YOUR RESULT: {self._format_value()}",
            "",
            "🔍 EVALUATION:",
            f"   {self.evaluation}",
            "",
            "📚 BENCHMARK CONTEXT:",
            f"   {self.benchmark_context}",
            "",
            "💡 RECOMMENDATION:",
            f"   {self.recommendation}",
        ]
        if self.related_metrics:
            lines.extend(["", f"🔗 RELATED METRICS: {', '.join(self.related_metrics)}"])
        return "\n".join(lines)

    def _format_value(self) -> str:
        """Format the value for display.

        The refusal line comes FIRST, for every absence door (see
        _value_is_absent): a reader of this card must not be shown 'nan',
        '<NA>', 'NaT' or 'inf' where the evaluation right below says the metric
        was never computed.
        """
        if _value_is_absent(self.value):
            return "N/A (insufficient data)"
        if isinstance(self.value, float):
            return f"{self.value:.4f}"
        if isinstance(self.value, np.floating):
            # A numpy float is a measured number and gets the same four
            # decimals as a python one; str(np.float32(0.12)) was '0.12'.
            # np.integer and python int deliberately keep str(), so an integer
            # count is not restyled as '5.0000'.
            return f"{float(self.value):.4f}"
        return str(self.value)


# Metric Definitions - Classification

CLASSIFICATION_METRICS: Dict[str, Dict[str, Any]] = {
    "demographic_parity_difference": {
        "definition": (
            "Demographic Parity Difference measures the maximum absolute difference "
            "in positive prediction rates across protected groups. It answers: "
            "'Are positive outcomes (e.g., loan approvals, job offers) distributed "
            "equally across groups, regardless of their actual qualifications?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1. A value of 0 means perfect parity - all groups "
            "receive positive predictions at the same rate. Higher values indicate "
            "larger disparities. The industry-standard threshold is 0.10 (10%), meaning "
            "groups should not differ by more than 10 percentage points in their "
            "positive prediction rates."
        ),
        "thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15, "critical": 0.20},
        "related_metrics": ["demographic_parity_ratio", "equalized_odds_difference"],
        "legal_context": (
            "This metric relates to disparate impact analysis under U.S. employment "
            "law (Title VII) and fair lending regulations (ECOA). While not a direct "
            "legal standard, large disparities may trigger regulatory scrutiny."
        ),
    },
    "demographic_parity_ratio": {
        "definition": (
            "Demographic Parity Ratio (also called Disparate Impact Ratio) measures "
            "the ratio of positive prediction rates between the least and most favored "
            "groups. It answers: 'What fraction of the highest group's positive rate "
            "does the lowest group receive?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1, where 1 means perfect parity. The famous "
            "'80% Rule' or 'Four-Fifths Rule' from U.S. employment law states that "
            "selection rates for protected groups should be at least 80% (0.80) of "
            "the rate for the group with the highest rate. Values below 0.80 may "
            "indicate potential disparate impact."
        ),
        "thresholds": {"excellent": 0.90, "acceptable": 0.80, "concerning": 0.70, "critical": 0.60},
        "related_metrics": ["demographic_parity_difference"],
        "legal_context": (
            "The 80% rule originates from the 1978 Uniform Guidelines on Employee "
            "Selection Procedures. While not an absolute legal standard, ratios "
            "below 0.80 create a prima facie case for disparate impact discrimination."
        ),
    },
    "equal_opportunity_difference": {
        "definition": (
            "Equal Opportunity Difference measures the maximum absolute difference "
            "in True Positive Rates (TPR, also called recall or sensitivity) across "
            "groups. It answers: 'Among people who actually deserve a positive outcome, "
            "are they equally likely to receive it regardless of group membership?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1. A value of 0 means equal opportunity - qualified "
            "individuals from all groups are equally likely to be correctly identified. "
            "Higher values indicate that some groups' qualified members are being "
            "systematically missed. The recommended threshold is 0.05 (5%)."
        ),
        "thresholds": {"excellent": 0.03, "acceptable": 0.05, "concerning": 0.10, "critical": 0.15},
        "related_metrics": ["equalized_odds_difference", "predictive_parity_difference"],
        "legal_context": (
            "Equal opportunity is particularly relevant in contexts where false negatives "
            "cause significant harm to individuals, such as loan denials for creditworthy "
            "applicants or failing to identify qualified job candidates."
        ),
    },
    "equalized_odds_difference": {
        "definition": (
            "Equalized Odds Difference measures the maximum of (a) the TPR difference "
            "and (b) the FPR difference across groups. It answers: 'Are prediction "
            "errors distributed equally across groups for both positive and negative "
            "true outcomes?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1. A value of 0 means the model makes errors at "
            "equal rates for all groups. This is a stricter criterion than equal "
            "opportunity because it considers both types of errors. The recommended "
            "threshold is 0.10 (10%)."
        ),
        "thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15, "critical": 0.20},
        "related_metrics": ["equal_opportunity_difference", "demographic_parity_difference"],
        "legal_context": (
            "Equalized odds is considered a 'meritocratic' fairness criterion because "
            "it conditions on the true label. It's appropriate when outcomes should "
            "depend only on qualifications, not group membership."
        ),
    },
    "predictive_parity_difference": {
        "definition": (
            "Predictive Parity Difference measures the maximum absolute difference "
            "in precision (Positive Predictive Value, PPV) across groups. It answers: "
            "'When the model predicts a positive outcome, does that prediction mean "
            "the same thing for all groups?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1. A value of 0 means a positive prediction is "
            "equally accurate across all groups. Higher values indicate that positive "
            "predictions are more reliable for some groups than others. The recommended "
            "threshold is 0.05 (5%)."
        ),
        "thresholds": {"excellent": 0.03, "acceptable": 0.05, "concerning": 0.10, "critical": 0.15},
        "related_metrics": ["calibration_difference", "equal_opportunity_difference"],
        "legal_context": (
            "Predictive parity relates to the reliability of predictions. In contexts "
            "like risk assessment, differing precision rates could mean that positive "
            "predictions systematically overestimate or underestimate risk for certain groups."
        ),
    },
    "calibration_difference": {
        "definition": (
            "Calibration Difference measures the maximum difference in Expected "
            "Calibration Error (ECE) across groups. It answers: 'When the model says "
            "there's a 70% probability, is that actually true for all groups?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 1. A value of 0 means the model is equally well "
            "calibrated for all groups. Higher values indicate the model's probability "
            "estimates are more reliable for some groups than others. The recommended "
            "threshold is 0.05 (5%)."
        ),
        "thresholds": {"excellent": 0.03, "acceptable": 0.05, "concerning": 0.08, "critical": 0.10},
        "related_metrics": ["predictive_parity_difference"],
        "legal_context": (
            "Calibration is critical in risk assessment contexts (credit scoring, "
            "recidivism prediction) where probability estimates directly influence "
            "decisions. Poor calibration for a group may constitute unfair treatment."
        ),
    },
}


# Metric Definitions - Regression

REGRESSION_METRICS: Dict[str, Dict[str, Any]] = {
    "mae_parity_difference": {
        "definition": (
            "MAE Parity Difference measures the maximum absolute difference in Mean "
            "Absolute Error across groups. It answers: 'Does the model make equally "
            "accurate predictions (in absolute terms) for all groups?'"
        ),
        "interpretation_guide": (
            "Values are in the same units as the target variable. A value of 0 means "
            "the model is equally accurate for all groups. The threshold is typically "
            "set relative to the data scale (15% of standard deviation). Higher values "
            "indicate the model performs systematically worse for some groups."
        ),
        "thresholds_relative": True,  # Threshold is relative to y_std
        "threshold_multiplier": 0.15,
        "related_metrics": ["rmse_parity_difference", "mean_prediction_difference"],
        "context": (
            "MAE is robust to outliers. If MAE disparity is high but RMSE disparity "
            "is low, the model may be making consistent small errors for one group "
            "rather than occasional large errors."
        ),
    },
    "rmse_parity_difference": {
        "definition": (
            "RMSE Parity Difference measures the maximum absolute difference in Root "
            "Mean Squared Error across groups. It answers: 'Does the model have equal "
            "prediction variance across groups, with extra penalty for large errors?'"
        ),
        "interpretation_guide": (
            "Values are in the same units as the target variable. A value of 0 means "
            "equal RMSE across groups. RMSE penalizes large errors more heavily than "
            "MAE, so high RMSE disparity indicates the model makes occasional large "
            "errors for some groups. The threshold is typically 15% of the standard deviation."
        ),
        "thresholds_relative": True,
        "threshold_multiplier": 0.15,
        "related_metrics": ["mae_parity_difference", "r2_parity_difference"],
        "context": (
            "If RMSE disparity is much higher than MAE disparity for a group, "
            "investigate potential outliers or edge cases where the model fails badly."
        ),
    },
    "mean_prediction_difference": {
        "definition": (
            "Mean Prediction Difference measures the maximum absolute difference "
            "in average predictions across groups. It answers: 'Does the model "
            "systematically predict higher or lower values for certain groups?'"
        ),
        "interpretation_guide": (
            "Values are in the same units as the target variable. A value of 0 means "
            "all groups receive the same average prediction. Non-zero values may "
            "indicate systematic over- or under-prediction for certain groups. The "
            "threshold is typically 10% of the standard deviation."
        ),
        "thresholds_relative": True,
        "threshold_multiplier": 0.10,
        "related_metrics": ["mae_parity_difference"],
        "context": (
            "Systematic prediction differences may reflect legitimate differences in "
            "outcomes between groups OR bias in the model. Context and base rates "
            "should be considered when interpreting this metric."
        ),
    },
    "r2_parity_difference": {
        "definition": (
            "R² Parity Difference measures the maximum absolute difference in "
            "coefficient of determination (R²) across groups. It answers: 'Does "
            "the model explain the same proportion of variance for all groups?'"
        ),
        "interpretation_guide": (
            "Values range from 0 to 2 (theoretically). A value of 0 means the model "
            "explains variance equally well for all groups. Higher values indicate "
            "the model fits some groups much better than others. The recommended "
            "threshold is 0.10 (10 percentage points difference in R²)."
        ),
        "thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15, "critical": 0.20},
        "related_metrics": ["rmse_parity_difference", "mae_parity_difference"],
        "context": (
            "R² disparity suggests the model may be missing important features for "
            "certain groups, or the underlying relationship may genuinely differ "
            "between groups."
        ),
    },
}


# Statistical Measure Definitions

STATISTICAL_MEASURES: Dict[str, Dict[str, Any]] = {
    "confidence_interval": {
        "definition": (
            "A Confidence Interval (CI) provides a range of plausible values for "
            "a metric. A 95% CI means that if we repeated the experiment many times, "
            "95% of the calculated intervals would contain the true population value."
        ),
        "interpretation_guide": (
            "The width of the CI reflects uncertainty. Narrow intervals indicate "
            "precise estimates; wide intervals suggest more uncertainty. If a CI "
            "for a difference metric includes 0, the observed disparity may not be "
            "statistically significant."
        ),
        "context": (
            "CIs are calculated using bootstrap resampling (for larger samples, n≥30) "
            "or Bayesian methods (for smaller samples, n<30). The method is selected "
            "automatically based on sample size to ensure valid inference."
        ),
    },
    "credible_interval": {
        "definition": (
            "A Bayesian Credible Interval provides a range where the true parameter "
            "value lies with a specified probability (e.g., 95%). Unlike confidence "
            "intervals, credible intervals have a direct probabilistic interpretation."
        ),
        "interpretation_guide": (
            "A 95% credible interval means there is a 95% probability that the true "
            "value lies within the interval, given the observed data. This is often "
            "more intuitive than the frequentist confidence interval interpretation."
        ),
        "context": (
            "Credible intervals are automatically used for small sample sizes (n<30) "
            "where bootstrap methods may be unreliable. They incorporate prior "
            "information (by default, a non-informative uniform prior)."
        ),
    },
    "standard_error": {
        "definition": (
            "Standard Error (SE) measures the precision of a sample statistic. It "
            "estimates how much the statistic would vary across different samples "
            "from the same population."
        ),
        "interpretation_guide": (
            "Smaller SE indicates more precise estimates. SE is inversely related "
            "to sample size - larger samples yield smaller SEs. The CI is typically "
            "calculated as point estimate ± 1.96 × SE for 95% confidence."
        ),
        "context": (
            "In fairness analysis, SE helps assess whether observed disparities are "
            "likely to replicate with new data. High SE suggests the disparity "
            "estimate is unstable."
        ),
    },
    "cohens_d": {
        "definition": (
            "Cohen's d is a standardized effect size that measures the difference "
            "between two groups in terms of their pooled standard deviation. It "
            "answers: 'How large is the difference in practical terms?'"
        ),
        "interpretation_guide": (
            "Cohen's d is dimensionless and can be interpreted using standard "
            "benchmarks: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, "
            ">0.8 = large effect. These benchmarks help assess practical significance "
            "beyond statistical significance."
        ),
        "thresholds": {"negligible": 0.2, "small": 0.5, "medium": 0.8, "large": float("inf")},
        "context": (
            "Effect sizes complement p-values and CIs. A statistically significant "
            "result may have a negligible effect size (especially with large samples), "
            "while a non-significant result may still show a meaningful effect size "
            "(especially with small samples)."
        ),
    },
    "risk_ratio": {
        "definition": (
            "Risk Ratio (RR), also called Relative Risk, compares the probability "
            "of an outcome between two groups. RR = P(outcome | group 1) / P(outcome | group 2). "
            "It answers: 'How many times more likely is the outcome in one group?'"
        ),
        "interpretation_guide": (
            "RR = 1 means equal risk. RR > 1 means group 1 has higher risk. "
            "RR < 1 means group 1 has lower risk. For example, RR = 1.5 means "
            "group 1 is 50% more likely to experience the outcome. RR = 0.8 means "
            "group 1 is 20% less likely."
        ),
        "context": (
            "Risk ratios are commonly used in medical research and public health. "
            "In fairness, they help quantify how much more/less likely a positive "
            "prediction is for one group compared to another."
        ),
    },
    "odds_ratio": {
        "definition": (
            "Odds Ratio (OR) compares the odds of an outcome between two groups. "
            "OR = (p1/(1-p1)) / (p2/(1-p2)) where p1 and p2 are the outcome "
            "probabilities. It's commonly used in case-control studies."
        ),
        "interpretation_guide": (
            "OR = 1 means equal odds. OR > 1 means higher odds in group 1. "
            "OR approximates RR when the outcome is rare (<10%). For common "
            "outcomes, OR tends to exaggerate the effect compared to RR."
        ),
        "context": (
            "Odds ratios are useful when comparing groups retrospectively or "
            "when the absolute risk is not meaningful. In fairness analysis, "
            "RR is often more interpretable for decision-making."
        ),
    },
    "p_value": {
        "definition": (
            "A p-value is the probability of observing results at least as extreme "
            "as those observed, assuming no true difference exists (null hypothesis). "
            "It is NOT the probability that the null hypothesis is true."
        ),
        "interpretation_guide": (
            "Conventionally, p < 0.05 is considered 'statistically significant'. "
            "However, p-values should not be the sole criterion. Consider: "
            "(1) effect size, (2) confidence intervals, (3) practical significance, "
            "(4) sample size, and (5) multiple testing."
        ),
        "context": (
            "In fairness analysis with multiple metrics, multiple testing corrections "
            "(Bonferroni, FDR) should be applied to avoid false discoveries. A single "
            "significant p-value among many tests may be a false positive."
        ),
    },
    "bonferroni_correction": {
        "definition": (
            "Bonferroni correction adjusts p-values for multiple testing by "
            "multiplying each p-value by the number of tests (or equivalently, "
            "dividing the significance threshold by the number of tests)."
        ),
        "interpretation_guide": (
            "After Bonferroni correction, use the same significance threshold "
            "(e.g., 0.05). The correction is conservative - it reduces false "
            "positives but may increase false negatives (miss real effects)."
        ),
        "context": (
            "Bonferroni is appropriate when any single false positive is costly. "
            "For exploratory fairness analysis, FDR correction (Benjamini-Hochberg) "
            "may be more appropriate as it's less conservative."
        ),
    },
    "fdr_correction": {
        "definition": (
            "False Discovery Rate (FDR) correction, typically via the Benjamini-Hochberg "
            "procedure, controls the expected proportion of false discoveries among "
            "rejected hypotheses, rather than the probability of any false discovery."
        ),
        "interpretation_guide": (
            "FDR = 0.05 means we expect 5% of the 'significant' findings to be "
            "false positives. This is less conservative than Bonferroni and "
            "allows more true positives to be detected while controlling the "
            "false discovery rate."
        ),
        "context": (
            "FDR correction is recommended for exploratory fairness analysis where "
            "multiple metrics are evaluated. It balances discovery with false positive "
            "control better than Bonferroni for most applications."
        ),
    },
}


# Metric-family mitigation advice


# Family-specific mitigation advice, keyed by the EXACT normalised FAMILY name.
#
# Until 2026-08-27 these three sentences were selected by ``"demographic_parity"
# in metric_name`` and two siblings: a family token matched ANYWHERE inside a
# name. That is the cali-BRATIO-n bug class this package has been removing all
# day ("ratio" is a substring of "cali[bratio]n_difference", which let a large
# miscalibration read as a PASS in production for weeks). It misfires the same
# way here: "equal_opportunity" is a substring of "un[equal_opportunity]_cost",
# so a metric that is not an equal-opportunity gap at all was told how to
# improve equal opportunity, in a report that names the metric next to it.
#
# Selection happens on whole underscore-delimited TOKENS now, through the one
# shared resolver in ``._metric_direction``: its normaliser canonicalises the
# name (so human labels land on the same key) and its ``metric_direction``
# decides whether the name is one this package can place at all. There is no
# second copy of the direction rules here.
_FAMILY_RECOMMENDATIONS: Dict[str, str] = {
    "demographic_parity": (
        "If demographic parity is a priority, consider post-processing "
        "methods like threshold adjustment or calibrated equalized odds."
    ),
    "equal_opportunity": (
        "To improve equal opportunity, focus on reducing false negatives "
        "for disadvantaged groups through targeted model improvements."
    ),
    "equalized_odds": (
        "Equalized odds can be improved through post-hoc calibration "
        "or by training with equalized odds constraints."
    ),
}


def _metric_family(metric_name: str) -> Optional[str]:
    """The metric FAMILY ``metric_name`` belongs to, or None when it is not one.

    Two rules, both deliberate.

    **Whole tokens, never containment.** The normalised name must START with the
    family's token sequence, so ``demographic_parity_difference``,
    ``demographic_parity_ratio`` and the per-column key
    ``equal_opportunity_difference_gender`` all resolve onto their family, while
    ``unequal_opportunity_cost`` resolves onto NOTHING. A containment test
    matches that last one, and any other name that merely carries the token
    inside it, which is the defect class recorded in ``._metric_direction``.

    **Fail closed on a name this package cannot place.** When the shared
    resolver cannot say which direction is better for a name, we do not know
    what the number is either, so we do not attach mitigation advice written for
    a specific family to it. A contradictory name such as
    ``demographic_parity_gap_ratio`` (a violation token and the ratio suffix at
    once) lands here and gets the generic advice only, rather than prose
    asserting what it measures.
    """
    if metric_direction(metric_name) is MetricDirection.UNKNOWN:
        return None
    tokens = _normalize(metric_name).split("_")
    for family in _FAMILY_RECOMMENDATIONS:
        family_tokens = family.split("_")
        if tokens[: len(family_tokens)] == family_tokens:
            return family
    return None


# The effect types whose INFINITY is a measurement rather than a missing one.
# A ratio reaches inf from a table that was measured and in which one arm
# received no positive outcomes at all (total exclusion, the strongest
# disparate-impact reading there is). Every other effect type reaches inf by
# dividing by an empty denominator, which is a could-not-check. Read by
# `_effect_size_unmeasured_reason`; see the carve-out comment there.
_RATIO_EFFECT_TYPES = frozenset({"risk_ratio", "odds_ratio"})


# FairExplAIner Class


class FairExplAIner:
    """
    Generates intelligent explanations for fairness metrics and statistics.

    This class provides context-aware explanations that help users understand
    what metrics mean, how to interpret them, and what actions to take.

    Example:
        >>> explainer = FairExplAIner()
        >>> explanation = explainer.explain_metric(
        ...     'demographic_parity_difference',
        ...     value=0.15,
        ...     group_stats={'M': {'size': 500}, 'F': {'size': 500}}
        ... )
        >>> print(explanation)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: fair_explainer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        y_std: Optional[float] = None,
    ):
        """
        Initialize the FairExplAIner.

        Args:
            task_type: Type of ML task ('classification' or 'regression')
            y_std: Standard deviation of target variable (for regression thresholds)
        """
        self.task_type = task_type
        self.y_std = y_std

        # Select appropriate metrics definitions
        self.metrics_definitions: Dict[str, Dict[str, Any]]
        if task_type == "classification":
            self.metrics_definitions = CLASSIFICATION_METRICS
        else:
            self.metrics_definitions = REGRESSION_METRICS

    def explain_metric(
        self,
        metric_name: str,
        value: float,
        group_stats: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        excluded_groups: Optional[List[Any]] = None,
    ) -> MetricExplanation:
        """
        Generate a comprehensive explanation for a metric.

        Args:
            metric_name: Name of the metric
            value: Computed metric value
            group_stats: Per-group statistics for context. Note this carries
                only the groups that SURVIVED filtering, so it cannot name the
                ones that were dropped.
            threshold: Custom threshold (if different from default)
            excluded_groups: Names of the groups dropped below ``min_group_size``,
                so a could-not-check recommendation can say which group needs
                more data instead of the useless "collect more data" in general.

        Returns:
            MetricExplanation with full context and recommendations
        """
        # Get metric definition
        metric_def = self.metrics_definitions.get(
            metric_name, STATISTICAL_MEASURES.get(metric_name, {})
        )

        if not metric_def:
            # Pass the threshold through. Dropping it here made the fallback
            # assert "no threshold was applied" for metrics the SAME report had
            # already graded: on an ordinary run with y_prob, multicalibration
            # and integrated_calibration_index came back FAIL in
            # assessment["failed_metrics"] while their explainer cards said the
            # value "was not graded ... neither a pass nor a fail", which tells a
            # reader to ignore a real failure.
            return self._generic_explanation(metric_name, value, threshold)

        # Determine threshold
        if threshold is None:
            threshold = self._get_threshold(metric_name, metric_def)

        # Generate evaluation
        evaluation, severity = self._evaluate_value(metric_name, value, threshold, metric_def)

        # Generate benchmark context
        benchmark_context = self._get_benchmark_context(metric_name, value, metric_def)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            metric_name, value, threshold, severity, group_stats, excluded_groups
        )

        return MetricExplanation(
            metric_name=metric_name,
            definition=metric_def.get("definition", "No definition available."),
            interpretation_guide=metric_def.get("interpretation_guide", ""),
            value=value,
            evaluation=evaluation,
            benchmark_context=benchmark_context,
            recommendation=recommendation,
            severity=severity,
            related_metrics=metric_def.get("related_metrics", []),
        )

    def explain_confidence_interval(
        self,
        metric_name: str,
        point_estimate: float,
        lower_bound: float,
        upper_bound: float,
        interval_type: str = "confidence",
        confidence_level: float = 0.95,
    ) -> MetricExplanation:
        """
        Generate an explanation for a confidence/credible interval.

        Args:
            metric_name: Name of the metric
            point_estimate: Point estimate value
            lower_bound: Lower bound of interval
            upper_bound: Upper bound of interval
            interval_type: 'confidence' or 'credible'
            confidence_level: Confidence/credibility level (e.g., 0.95)

        Returns:
            MetricExplanation for the interval
        """
        ci_def = STATISTICAL_MEASURES.get(
            f"{interval_type}_interval", STATISTICAL_MEASURES["confidence_interval"]
        )

        # THREE STATES, never two. A NaN bound means the interval was never
        # built: compute_metric_with_ci returns NaN bounds when no rows survive
        # the exclusions and when the bootstrap fails. Graded as numbers those
        # NaNs read as a MEASUREMENT, because every comparison against NaN is
        # False: `nan <= 0 <= nan` is False, so the branch below announced "The
        # interval excludes zero, suggesting the observed disparity is
        # statistically significant" at severity "info", for an interval that
        # does not exist. Measured 2026-09-09 on
        # explain_confidence_interval(m, nan, nan, nan).
        unmeasured = [
            label
            for label, raw in (
                ("point estimate", point_estimate),
                ("lower bound", lower_bound),
                ("upper bound", upper_bound),
            )
            if self._finite_or_none(raw) is None
        ]
        if unmeasured:
            return MetricExplanation(
                metric_name=f"{metric_name}_ci",
                definition=ci_def.get("definition", ""),
                interpretation_guide=ci_def.get("interpretation_guide", ""),
                value={"point_estimate": point_estimate, "ci": [lower_bound, upper_bound]},
                evaluation=(
                    f"COULD NOT CHECK: no interval was computed for {metric_name} "
                    f"({', '.join(unmeasured)} not a number), so nothing here says "
                    f"whether the disparity is significant, or how precise it is. "
                    f"This is neither a pass nor a fail."
                ),
                benchmark_context=ci_def.get("context", ""),
                recommendation=(
                    "Find out why the interval is missing before reading anything "
                    "into it: the usual causes are that no rows survived the "
                    "exclusions and that the bootstrap failed. Re-run with enough "
                    "data per group. An absent interval is not evidence either way."
                ),
                severity="could_not_check",
                related_metrics=[metric_name],
            )

        # Determine if interval includes zero (for difference metrics)
        includes_zero = lower_bound <= 0 <= upper_bound
        width = upper_bound - lower_bound

        # Interpret width relative to estimate
        relative_width = width / abs(point_estimate) if point_estimate != 0 else float("inf")

        if relative_width < 0.5:
            precision = "very precise"
        elif relative_width < 1.0:
            precision = "reasonably precise"
        elif relative_width < 2.0:
            precision = "moderately uncertain"
        else:
            precision = "highly uncertain"

        # Build evaluation
        conf_pct = int(confidence_level * 100)
        if interval_type == "credible":
            evaluation = (
                f"The {conf_pct}% credible interval [{lower_bound:.4f}, {upper_bound:.4f}] "
                f"indicates that there is a {conf_pct}% probability the true value lies "
                f"within this range. The estimate is {precision}. "
            )
        else:
            evaluation = (
                f"The {conf_pct}% confidence interval [{lower_bound:.4f}, {upper_bound:.4f}] "
                f"means that if we repeated this analysis many times, {conf_pct}% of the "
                f"intervals would contain the true value. The estimate is {precision}. "
            )

        if includes_zero:
            evaluation += (
                "The interval includes zero, suggesting the observed disparity may not "
                "be statistically significant."
            )
        else:
            evaluation += (
                "The interval excludes zero, suggesting the observed disparity is "
                "statistically significant."
            )

        recommendation = (
            "Consider the interval alongside the point estimate when making decisions. "
        )
        if relative_width > 1.0:
            recommendation += (
                "The wide interval suggests collecting more data would improve estimate precision."
            )

        return MetricExplanation(
            metric_name=f"{metric_name}_ci",
            definition=ci_def.get("definition", ""),
            interpretation_guide=ci_def.get("interpretation_guide", ""),
            value={"point_estimate": point_estimate, "ci": [lower_bound, upper_bound]},
            evaluation=evaluation,
            benchmark_context=ci_def.get("context", ""),
            recommendation=recommendation,
            severity="info",
            related_metrics=[metric_name],
        )

    def explain_effect_size(
        self, effect_type: str, value: float, group1: str, group2: str, ci: Optional[tuple] = None
    ) -> MetricExplanation:
        """
        Generate an explanation for an effect size measure.

        Args:
            effect_type: Type of effect size ('cohens_d', 'risk_ratio', 'odds_ratio')
            value: Effect size value
            group1: Name of first group
            group2: Name of second group
            ci: Optional confidence interval tuple (lower, upper)

        Returns:
            MetricExplanation for the effect size
        """
        effect_def = STATISTICAL_MEASURES.get(effect_type, {})

        # BGL-S2c (2026-09-17). THREE STATES, never two, and the guard sits
        # ABOVE the dispatch because all four branches below share the same
        # precondition: every one of them grades `value` on an `abs()` /
        # comparison ladder, and every comparison against NaN is False, so an
        # effect size that was never computed fell through each ladder to its
        # most alarming rung. Measured at the public entry on REAL data (80
        # rows, nobody selected in either group, so the risk ratio is 0/0):
        #
        #   explain_report({"effect_sizes": compute_effect_sizes(...)})
        #     risk_ratio_A_vs_B -> severity "high",
        #       "Large difference: one group is nanx less likely."
        #       "The effect is large and requires action."
        #
        # An INFINITE ratio is deliberately NOT swept in here. risk_ratio and
        # odds_ratio return inf for a genuinely measured table where one arm
        # received no positive outcomes at all, which is total exclusion, the
        # strongest disparate-impact reading there is. Calling that a
        # could-not-check would delete a real finding; it stays graded, and
        # only its wording is fixed below.
        # The effect TYPE travels with the value: the infinity carve-out below
        # belongs to the two ratios only, and passing the type is what keeps an
        # infinite Cohen's d out of it.
        unmeasured_reason = self._effect_size_unmeasured_reason(value, effect_type)
        if unmeasured_reason is not None:
            return MetricExplanation(
                metric_name=f"{effect_type}_{group1}_vs_{group2}",
                definition=effect_def.get("definition", ""),
                interpretation_guide=effect_def.get("interpretation_guide", ""),
                value=value,
                evaluation=(
                    f"COULD NOT CHECK: comparing {group1} vs {group2}, the "
                    f"{effect_type} was not measured ({unmeasured_reason}), so "
                    f"nothing here says how large the difference between these two "
                    f"groups is, or in which direction. This is neither a pass nor "
                    f"a fail."
                ),
                benchmark_context=effect_def.get("context", ""),
                # One source of truth for the wording, and it keeps
                # `_effect_size_recommendation`'s could_not_check branch on the
                # live path: an inline literal here left that branch unreachable,
                # so deleting it changed nothing a test could see.
                recommendation=self._effect_size_recommendation(
                    effect_type, value, "could_not_check"
                ),
                severity="could_not_check",
                related_metrics=[],
            )

        # Interpret effect size
        could_not_check_kind = "unmeasured"
        if effect_type == "cohens_d":
            interpretation, severity = self._interpret_cohens_d(value)
        elif effect_type == "risk_ratio":
            interpretation, severity = self._interpret_risk_ratio(value)
        elif effect_type == "odds_ratio":
            interpretation, severity = self._interpret_odds_ratio(value)
        else:
            # THE COULD-NOT-CHECK FOR THE TYPE, 2026-09-29. The guard above the
            # dispatch answers the question for the VALUE; this `else` was
            # answering it for the TYPE by fabricating a verdict. Only three of the
            # nine keys in this module's own STATISTICAL_MEASURES have a band
            # ladder, and every other type reached this branch and came back
            # severity 'info' with the recommendation "No action needed - the
            # effect size is negligible." plus the library's real definition text
            # attached, which is what makes the card read as authentic. Measured:
            #
            #   explain_effect_size("p_value", 0.001, "A", "B")
            #     -> severity 'info', "Comparing A vs B: The p_value is 0.001.",
            #        "No action needed - the effect size is negligible."
            #   and the same for standard_error 12.5, confidence_interval 0.0,
            #   fdr_correction 0.99, credible_interval 0.5, bonferroni_correction.
            #
            # 'info' is the state a graded, genuinely benign value receives, and
            # "no action needed" is a false statement about a p of 0.001. No ladder
            # was computed, so the band is a could-not-check, and an UNKNOWN type
            # (not in STATISTICAL_MEASURES at all) lands here too and fails closed
            # the same way.
            interpretation = (
                f"the {effect_type} is {value:.3f}, but this library defines no "
                f"interpretation bands for it (only cohens_d, risk_ratio and odds_ratio "
                f"are graded here), so its magnitude was NOT graded. This is neither a "
                f"pass nor a fail, and an ungraded effect size is not a negligible one."
            )
            severity = "could_not_check"
            # The value EXISTS here, so the recommendation must not send the reader
            # looking for an empty denominator: that is the other could-not-check.
            could_not_check_kind = "ungraded_type"

        if severity == "could_not_check":
            evaluation = f"COULD NOT CHECK: comparing {group1} vs {group2}, {interpretation}"
        else:
            evaluation = f"Comparing {group1} vs {group2}: {interpretation}"

        if ci:
            # A NaN bound means no interval was estimable (risk_ratio and
            # odds_ratio return NaN bounds for a saturated or one-sided table).
            # Printed through `:.3f` that reads as "95% CI: [nan, nan]", which
            # looks like an interval that exists.
            if self._finite_or_none(ci[0]) is None or self._finite_or_none(ci[1]) is None:
                evaluation += " (no 95% interval was estimable for this comparison.)"
            else:
                evaluation += f" (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"

        return MetricExplanation(
            metric_name=f"{effect_type}_{group1}_vs_{group2}",
            definition=effect_def.get("definition", ""),
            interpretation_guide=effect_def.get("interpretation_guide", ""),
            value=value,
            evaluation=evaluation,
            benchmark_context=effect_def.get("context", ""),
            recommendation=self._effect_size_recommendation(
                effect_type, value, severity, could_not_check_kind
            ),
            severity=severity,
            related_metrics=[],
        )

    def explain_report(
        self, report: Dict[str, Any], include_statistical: bool = True
    ) -> Dict[str, Any]:
        """
        Generate explanations for an entire fairness report.

        Args:
            report: Report dictionary from classification/regression_fairness_report
            include_statistical: Whether to include statistical measure explanations

        Returns:
            Dictionary with explanations for all components
        """
        explanations: Dict[str, Any] = {"metrics": {}, "statistical": {}, "summary": ""}

        # Explain each metric
        metrics = report.get("metrics", {})
        group_stats = report.get("group_stats", {})
        thresholds = report.get("thresholds_used", {})

        # The groups the report itself recorded as excluded from the verdict.
        # group_stats holds only the SURVIVORS, so without this a
        # could-not-check recommendation could not name the group that needs
        # more data, which is the one thing the reader has to act on.
        assessment = report.get("assessment", {})
        excluded_groups = [
            record.get("group")
            for record in (
                assessment.get("insufficient_evidence_groups", [])
                if isinstance(assessment, dict)
                else []
            )
            if isinstance(record, dict) and record.get("group") is not None
        ]

        # EVERY key in `metrics` gets a card, including one this library cannot
        # read as a number. The filter here used to be
        # `if isinstance(value, (int, float))`, which DROPPED such a metric with
        # no card, no mention in the summary and no warning, so "this metric was
        # never measured" and "this metric does not exist" became the same
        # output. Measured 2026-09-27:
        #   explain_report({"metrics": {"demographic_parity_difference": None,
        #                               "equalized_odds_difference": 0.42}, ...})
        #     -> explanations["metrics"] held ONE key, equalized_odds_difference,
        #        and nothing anywhere named the metric that went missing.
        # A None arrives whenever a report has been through a strict JSON
        # encoder, which has no NaN and writes null. explain_metric already
        # answers could-not-check for a non-number (its `_could_not_check_reason`
        # is the single source of truth for that question), so routing the value
        # there is all this needs: the card says COULD NOT CHECK instead of
        # vanishing.
        unreadable: List[str] = []
        for metric_name, value in metrics.items():
            threshold = thresholds.get(metric_name)
            if self._could_not_check_reason(
                metric_name, value
            ) == "unmeasurable" and not isinstance(value, (int, float)):
                unreadable.append(f"{metric_name}={value!r}")
            explanation = self.explain_metric(
                metric_name, value, group_stats, threshold, excluded_groups
            )
            explanations["metrics"][metric_name] = explanation.to_dict()
        if unreadable:
            warnings.warn(
                f"explain_report: {len(unreadable)} metric value(s) are not numbers, so they "
                f"were NOT graded and their cards read COULD NOT CHECK rather than being "
                f"omitted: {', '.join(unreadable)}. An ungraded metric is not a passing one.",
                UserWarning,
                stacklevel=2,
            )

        # Explain statistical measures if present
        if include_statistical and "metrics_with_ci" in report:
            for metric_name, ci_data in report["metrics_with_ci"].items():
                if isinstance(ci_data, dict):
                    # A MISSING bound is unmeasured, so it travels as NaN and
                    # comes back as could-not-check. It used to default to 0,
                    # which is a real, readable interval endpoint: an absent
                    # lower bound rendered as "[0.0000, ...]" and was graded.
                    # The confidence level comes from the interval that was
                    # actually built, and only then from the report-level
                    # summary, so a per-metric level is never overwritten by
                    # the run-level one.
                    validation = report.get("statistical_validation", {})
                    level: Optional[float] = self._finite_or_none(ci_data.get("confidence_level"))
                    if level is None:
                        level = self._finite_or_none(validation.get("confidence_level"))
                    if level is None:
                        level = 0.95
                    ci_explanation = self.explain_confidence_interval(
                        metric_name,
                        ci_data.get("point_estimate", float("nan")),
                        ci_data.get("lower_bound", float("nan")),
                        ci_data.get("upper_bound", float("nan")),
                        ci_data.get("interval_type", "confidence"),
                        level,
                    )
                    explanations["statistical"][f"{metric_name}_ci"] = ci_explanation.to_dict()

        # Explain effect sizes if present
        if include_statistical and "effect_sizes" in report:
            for pair_name, effect_data in report["effect_sizes"].items():
                if isinstance(effect_data, dict):
                    groups = pair_name.split("_vs_")
                    if len(groups) == 2:
                        # Cohen's d
                        if "cohens_d_positive_rate" in effect_data:
                            explanation = self.explain_effect_size(
                                "cohens_d",
                                effect_data["cohens_d_positive_rate"],
                                groups[0],
                                groups[1],
                            )
                            explanations["statistical"][f"cohens_d_{pair_name}"] = (
                                explanation.to_dict()
                            )

                        # Risk ratio
                        if "risk_ratio" in effect_data:
                            rr_val = effect_data["risk_ratio"]
                            if isinstance(rr_val, (list, tuple)):
                                explanation = self.explain_effect_size(
                                    "risk_ratio",
                                    rr_val[0],
                                    groups[0],
                                    groups[1],
                                    ci=(rr_val[1], rr_val[2]) if len(rr_val) > 2 else None,
                                )
                                explanations["statistical"][f"risk_ratio_{pair_name}"] = (
                                    explanation.to_dict()
                                )

        # Generate overall summary
        explanations["summary"] = self._generate_summary(report, explanations)

        return explanations

    # Private Helper Methods

    def _get_threshold(self, metric_name: str, metric_def: Dict) -> float:
        """Get the appropriate threshold for a metric."""
        if metric_def.get("thresholds_relative") and self.y_std:
            return metric_def.get("threshold_multiplier", 0.1) * self.y_std

        thresholds = metric_def.get("thresholds", {})
        return thresholds.get("acceptable", 0.1)

    def _could_not_check_reason(self, metric_name: str, value: Any) -> Optional[str]:
        """Why this metric could not be GRADED at all, or None when it could.

        The single source of truth for the could-not-check state, read by both
        the evaluation prose and the recommendation. They used to answer this
        question separately, and disagreed: the evaluation said "Unable to
        compute due to insufficient data." while the recommendation beside it
        was the one written for a PASSING metric ("Continue monitoring ...
        Document your fairness practices for compliance purposes"), because
        both the unmeasurable case and the excellent case collapse onto
        severity "info". Deciding it once is what keeps them from drifting
        apart again.

        Returns ``'unmeasurable'`` (the metric was never computed: NaN, or a
        value that is not a number at all), ``'unknown_direction'`` (computed,
        but its better-direction cannot be resolved so it was never compared to
        a threshold), or ``None``.
        """
        # REDUNDANT, AND KEPT, AND NOT EVIDENCE (BGL6 F01, 2026-09-28). The next
        # line refuses the same values: is_measured names np.bool_ explicitly and
        # np.bool_ is not a numbers.Real either, so reverting this clause to
        # `isinstance(value, bool)` changes no outcome. Measured today by doing
        # exactly that and running both pins the grading record named as its
        # evidence: 64 passed, nothing red. The clause stays because a bool is not
        # a measurement and saying so at the top of the predicate is clearer than
        # relying on a helper, but it must never again be cited as the thing a
        # sabotage proved. The grading record that cited it has been corrected; it
        # is an internal audit file, so the correction is stated here rather than
        # by pointing a reader at a document they cannot open.
        if isinstance(value, (bool, np.bool_)):
            # BGL-5 (2026-09-27), the numpy half of the same defect. The guard
            # below tested `isinstance(value, bool)` only, and `isinstance(np.
            # True_, bool)` is False while `np.isnan(np.True_)` is also False, so
            # a numpy boolean walked straight past it into the grading ladder.
            # np.bool_ is what every numpy and pandas comparison returns
            # (`df["a"] > df["b"]).any()`, `np.all(...)`, `np.isclose(...)`), so
            # it is the shape this value actually arrives in. Measured at the
            # public entry before this line changed:
            #   explain_metric("demographic_parity_difference", np.False_)
            #     -> severity "info", "Excellent! The difference of 0.0000
            #        indicates near-perfect fairness across groups."
            #   explain_metric("demographic_parity_difference", np.True_)
            #     -> severity "critical", "Critical. The difference of 1.0000 ..."
            # and after:
            #   both -> severity "could_not_check", "COULD NOT CHECK: this metric
            #        was never computed (its value is not a number), so no
            #        threshold was applied to it."
            # A measured 0.0 is untouched: explain_metric(..., 0.0) is still
            # severity "info" (a real perfect parity is not refused).
            #
            # A FLAG IS NOT A MEASUREMENT, and this is the one non-number that
            # np.isnan accepts: it coerces True to 1.0 and False to 0.0, so the
            # ladder in _evaluate_value graded it. Measured 2026-09-27 at the
            # public entry, both directions wrong and the second one the worse:
            #   explain_metric("demographic_parity_difference", False)
            #     -> "Excellent! The difference of 0.0000 indicates near-perfect
            #        fairness across groups.", severity "info", with the
            #        recommendation written for a PASSING metric
            #   explain_metric("demographic_parity_difference", True)
            #     -> "Critical. The difference of 1.0000 ...", severity "critical"
            # A boolean reaches here through explain_report, whose metrics filter
            # admits bool because `isinstance(True, int)` is True. This class
            # already refuses a bool by name in `_finite_or_none` ("a bool is not
            # a measurement (True would coerce to 1.0)"), which decides the same
            # question for the interval and effect-size surfaces; those two
            # answered could-not-check for a bool while this one graded it, so one
            # class disagreed with itself about one input.
            return "unmeasurable"
        # BGL-5 deferral from batch A-evaluation-1, closed 2026-09-27. This was
        # `bool(np.isnan(value))`, which made NaN the ONLY unmeasurable class this
        # predicate knew, so AN INFINITY WAS GRADED. Measured at the direct API
        # before this line changed:
        #   FairExplAIner().explain_metric("demographic_parity_difference", inf)
        #     -> severity "critical", zero warnings
        #   the same call with -inf -> severity "critical"
        #   the same call with nan  -> severity "could_not_check"   (correct)
        # and through the analyzer, on a RATIO metric, the same infinity read
        # "Excellent! The ratio of inf indicates near-perfect parity". The grade
        # an infinity received depended on which metric it arrived at, which is
        # the tell: it was never a reading of anything.
        #
        # After: every non-finite value answers "unmeasurable" at every entry.
        # A measured 0.1 is untouched and still grades "low".
        #
        # `is_measured` is the repo-wide predicate for this question and it is
        # already imported by this module, used twenty lines below in
        # `_finite_or_none`. Those two answered DIFFERENTLY about the same input,
        # so one class disagreed with itself, which is how the hole survived. It
        # accepts numpy scalars (float32, float64, int64) and rejects inf, NaN,
        # bool, None, str and list, verified over all thirteen shapes.
        if not is_measured(value):
            return "unmeasurable"
        # Scoped to the fairness-metric definitions on purpose: the statistical
        # measures (Cohen's d, the corrections) are graded on their own
        # magnitude bands and carry no fairness direction.
        if (
            metric_direction(metric_name) is MetricDirection.UNKNOWN
            and metric_name in self.metrics_definitions
        ):
            return "unknown_direction"
        return None

    def _evaluate_value(
        self, metric_name: str, value: float, threshold: float, metric_def: Dict
    ) -> tuple:
        """Evaluate a metric value against thresholds."""
        if self._could_not_check_reason(metric_name, value) == "unmeasurable":
            # "info" is what a graded, genuinely BENIGN metric receives, so
            # reusing it here made "we checked and it is fine" and "this was
            # never computed" identical to every reader and every consumer that
            # colours or counts by severity. This class's own docstring declares
            # `could_not_check` for exactly this branch and says it "must never
            # be collapsed into either of the other two readings"; the code
            # collapsed it anyway. Measured 2026-09-08: a NaN
            # demographic_parity_difference and a genuinely excellent one both
            # returned severity "info".
            return (
                "COULD NOT CHECK: this metric was never computed (its value is not a "
                "number), so no threshold was applied to it. This is neither a pass "
                "nor a fail.",
                "could_not_check",
            )

        thresholds = metric_def.get("thresholds", {})

        # Direction from the shared resolver, never a local name test. It already
        # excludes "cali[bratio]n_difference" (a substring match for "ratio" that
        # let a large miscalibration read as a PASS) and it also catches the
        # disparate-impact labels, which no "_ratio" suffix test reaches.
        direction = metric_direction(metric_name)

        # A FAIRNESS metric whose better-direction we cannot resolve must not be
        # graded at all: the difference branch below would call a large value
        # "Critical" and a small one "Excellent!", and for a higher-is-better
        # metric that reading is exactly backwards. Say could-not-check instead,
        # with the same neutral severity the unmeasurable (NaN) case uses. The
        # condition itself lives in _could_not_check_reason, so the
        # recommendation cannot disagree with this line about whether the
        # metric was graded.
        if self._could_not_check_reason(metric_name, value) == "unknown_direction":
            return (
                f"COULD NOT CHECK: {metric_name} has no known better-direction "
                f"(is a lower value better or worse?), so the value {value:.4f} "
                f"was not graded against a threshold. This is neither a pass nor "
                f"a fail.",
                # Was "info", the same severity as a graded benign metric, while
                # explain_metric answered "could_not_check" for the very same
                # condition. The two surfaces disagreed about the same metric.
                "could_not_check",
            )

        if direction is MetricDirection.HIGHER_IS_BETTER:
            if value >= thresholds.get("excellent", 0.90):
                evaluation = (
                    f"Excellent! The ratio of {value:.4f} indicates near-perfect parity "
                    f"between groups."
                )
                severity = "info"
            elif value >= thresholds.get("acceptable", 0.80):
                evaluation = (
                    f"Acceptable. The ratio of {value:.4f} meets the standard threshold "
                    f"of {thresholds.get('acceptable', 0.80):.2f} (80% rule)."
                )
                severity = "low"
            elif value >= thresholds.get("concerning", 0.70):
                evaluation = (
                    f"Concerning. The ratio of {value:.4f} is below the 80% threshold, "
                    f"indicating potential disparate impact."
                )
                severity = "medium"
            else:
                evaluation = (
                    f"Critical. The ratio of {value:.4f} indicates severe disparity "
                    f"that may constitute discrimination."
                )
                severity = "critical"
        else:
            # Difference metrics (lower is better)
            excellent = thresholds.get("excellent", 0.05)
            acceptable = thresholds.get("acceptable", threshold)
            concerning = thresholds.get("concerning", threshold * 1.5)
            critical = thresholds.get("critical", threshold * 2)

            # Graded on the MAGNITUDE, because a lower-is-better metric is the
            # size of a violation and a band on the raw value is not symmetric
            # about zero. Executed on this repo before 2026-08-27:
            # explain_metric("mean_prediction_difference", -0.45) returned
            # "Excellent! The difference of -0.4500 indicates near-perfect
            # fairness across groups." at severity "info", with the advice
            # written for a PASSING metric beside it, while +0.45 (the same
            # disparity, measured the other way round) was "Critical". This is
            # the sign hole wave 4 closed in analyzer._fairness_verdict, which
            # had contradicted itself the same way. The value shown in the
            # prose stays the signed one that was measured; only the comparison
            # uses the magnitude.
            magnitude = abs(value)

            if magnitude <= excellent:
                evaluation = (
                    f"Excellent! The difference of {value:.4f} indicates near-perfect "
                    f"fairness across groups."
                )
                severity = "info"
            elif magnitude <= acceptable:
                evaluation = (
                    f"Acceptable. The difference of {value:.4f} is within the standard "
                    f"threshold of {acceptable:.4f}."
                )
                severity = "low"
            elif magnitude <= concerning:
                evaluation = (
                    f"Concerning. The difference of {value:.4f} exceeds the standard "
                    f"threshold of {acceptable:.4f} and should be investigated."
                )
                severity = "medium"
            elif magnitude <= critical:
                evaluation = (
                    f"High concern. The difference of {value:.4f} significantly exceeds "
                    f"acceptable thresholds and requires attention."
                )
                severity = "high"
            else:
                evaluation = (
                    f"Critical. The difference of {value:.4f} indicates severe disparity "
                    f"that requires immediate investigation and remediation."
                )
                severity = "critical"

        return evaluation, severity

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        """The value as a float, or None when it is not a real finite number.

        A bool is not a measurement (True would coerce to 1.0), a string is
        not a measurement, and NaN/inf mean nothing was measured.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        v = float(value)
        return v if np.isfinite(v) else None

    @staticmethod
    def _band_number(thresholds: Dict, key: str) -> Optional[float]:
        """The band edge named *key*, or None when this metric has no such band.

        NEVER substitutes a default. A missing edge used to be filled in with a
        literal (0.05 / 0.10 / 0.15, or 0.90 / 0.80 / 0.70), which printed a
        benchmark the metric does not have: measured 2026-09-09,
        ``_get_benchmark_context('cohens_d', 0.35, ...)`` announced "Industry
        Benchmarks: Excellent (≤0.05), Acceptable (≤0.10), Concerning (>0.15)"
        while Cohen's d's own bands are 0.2 / 0.5 / 0.8. None of those three
        numbers exists anywhere in that metric's definition.
        """
        return FairExplAIner._finite_or_none(thresholds.get(key))

    def _band_for_value(
        self, metric_name: str, value: Any, thresholds: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Which benchmark band the MEASURED value falls in.

        Three states, never two: ``(band, None)`` when the value was placed,
        ``(None, reason)`` when it could not be, and the reason is always said
        out loud rather than collapsed into a band.

        The edges are the ones :meth:`_evaluate_value` grades on (magnitude for
        a difference metric, so the sign hole stays closed), so the benchmark
        line and the verdict printed beside it name the same band.
        """
        reason = self._could_not_check_reason(metric_name, value)
        if reason == "unmeasurable":
            return None, (
                "the value is not a number, so it was never graded and cannot be placed in a band"
            )
        if reason == "unknown_direction":
            return None, (
                "this metric has no known better-direction (is a lower value better "
                "or worse?), so it was not compared to these bands"
            )

        excellent = self._band_number(thresholds, "excellent")
        acceptable = self._band_number(thresholds, "acceptable")
        concerning = self._band_number(thresholds, "concerning")
        if excellent is None or acceptable is None or concerning is None:
            present = ", ".join(sorted(str(k) for k in thresholds)) or "none"
            return None, (
                f"this metric carries no excellent/acceptable/concerning bands "
                f"(its definition has: {present})"
            )

        v = float(value)
        if metric_direction(metric_name) is MetricDirection.HIGHER_IS_BETTER:
            if v >= excellent:
                return "excellent", None
            if v >= acceptable:
                return "acceptable", None
            if v >= concerning:
                return "concerning", None
            return "critical", None

        # Difference metrics: compare the MAGNITUDE, as _evaluate_value does.
        magnitude = abs(v)
        if magnitude <= excellent:
            return "excellent", None
        if magnitude <= acceptable:
            return "acceptable", None
        if magnitude <= concerning:
            return "concerning", None
        critical = self._band_number(thresholds, "critical")
        if critical is not None and magnitude <= critical:
            return "high concern", None
        return "critical", None

    def _get_benchmark_context(self, metric_name: str, value: float, metric_def: Dict) -> str:
        """Benchmark context for a metric, INCLUDING which band *value* is in.

        The bands used to be printed without ever placing the measured value in
        one, although the value was passed in for exactly that (audit 6, S-16a).
        Measured 2026-09-09: demographic_parity_difference at 0.001 (perfect)
        and at 0.85 (catastrophic) produced the identical sentence, so the
        benchmark line told the reader nothing about the assessment it sat in.
        """
        legal_context = metric_def.get("legal_context", "")
        general_context = metric_def.get("context", "")

        context_parts = []

        if legal_context:
            context_parts.append(f"Legal/Regulatory Context: {legal_context}")

        if general_context:
            context_parts.append(f"Technical Context: {general_context}")

        # Add benchmark comparison
        thresholds = metric_def.get("thresholds", {})
        if thresholds:
            excellent = self._band_number(thresholds, "excellent")
            acceptable = self._band_number(thresholds, "acceptable")
            concerning = self._band_number(thresholds, "concerning")
            # Direction from the shared resolver, never a local name test here.
            higher_is_better = metric_direction(metric_name) is MetricDirection.HIGHER_IS_BETTER
            if excellent is not None and acceptable is not None and concerning is not None:
                if higher_is_better:
                    context_parts.append(
                        f"Industry Benchmarks: Excellent (≥{excellent:.2f}), "
                        f"Acceptable (≥{acceptable:.2f}), "
                        f"Concerning (<{concerning:.2f})."
                    )
                else:
                    context_parts.append(
                        f"Industry Benchmarks: Excellent (≤{excellent:.2f}), "
                        f"Acceptable (≤{acceptable:.2f}), "
                        f"Concerning (>{concerning:.2f})."
                    )

            band, why_not = self._band_for_value(metric_name, value, thresholds)
            if band is not None:
                context_parts.append(
                    f"The measured value of {float(value):.4f} falls in the {band.upper()} band."
                )
            else:
                context_parts.append(f"The measured value was NOT placed in a band: {why_not}.")

        return " ".join(context_parts) if context_parts else "No benchmark information available."

    def _could_not_check_recommendation(
        self,
        metric_name: str,
        reason: str,
        excluded_groups: Optional[List[Any]] = None,
    ) -> str:
        """What to actually DO about a metric that was never graded.

        NEVER reassure here. This branch exists because the could-not-check
        case used to receive the recommendation written for a passing metric
        ("Continue monitoring this metric as part of regular fairness audits.
        Document your fairness practices for compliance purposes."), which
        tells a reader that an unmeasured metric is satisfactory. It is the
        report-surface form of certifying what was never measured.
        """
        if reason == "unknown_direction":
            return (
                f"NOT MEASURED against a threshold: the better-direction of "
                f"'{metric_name}' could not be resolved, so its value was never "
                f"compared to a bound and nothing here says it is acceptable. "
                f"To obtain a verdict, declare the metric's direction in "
                f"vfairness.evaluation.vfairness_metrics._metric_direction "
                f"(is a lower value better, or a higher one?) and re-run the "
                f"assessment. Note min_group_size is not the cause here: the "
                f"value exists, the comparison does not. Until the direction is "
                f"declared, treat this metric as unassessed, not as passing."
            )

        named = ""
        if excluded_groups:
            named = (
                " Group(s) excluded from this run: "
                + ", ".join(repr(str(g)) for g in excluded_groups)
                + "."
            )
        return (
            f"NOT MEASURED: '{metric_name}' could not be computed on this data, "
            f"so nothing here says it is satisfactory. To obtain a verdict, "
            f"collect more data for the under-represented group(s) so they clear "
            f"min_group_size, or lower min_group_size (only if the smaller group "
            f"is still large enough to interpret; below the invalid reliability "
            f"tier a rate is not interpretable), then re-run the assessment."
            f"{named} Until it is measured, treat this metric as unassessed, "
            f"not as passing."
        )

    def _generate_recommendation(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        severity: str,
        group_stats: Optional[Dict] = None,
        excluded_groups: Optional[List[Any]] = None,
    ) -> str:
        """Generate actionable recommendations based on the metric result."""
        # Could-not-check FIRST. It shares severity "info" with the excellent
        # case, so the passing branch below would otherwise claim a metric that
        # was never measured is fine.
        reason = self._could_not_check_reason(metric_name, value)
        if reason is not None:
            return self._could_not_check_recommendation(metric_name, reason, excluded_groups)

        if severity == "info":
            return (
                "Continue monitoring this metric as part of regular fairness audits. "
                "Document your fairness practices for compliance purposes."
            )

        recommendations = []

        if severity in ["low", "medium"]:
            recommendations.append(
                "Review the model's training data for potential biases or "
                "underrepresentation of certain groups."
            )
            recommendations.append(
                "Consider using fairness-aware training techniques such as "
                "reweighting, resampling, or adversarial debiasing."
            )

        if severity in ["medium", "high"]:
            recommendations.append(
                "Conduct a deeper analysis to identify which features contribute "
                "most to the disparity."
            )
            recommendations.append(
                "Evaluate whether proxy variables in the data may be encoding "
                "protected characteristics."
            )

        if severity in ["high", "critical"]:
            recommendations.append(
                "URGENT: Review deployment decisions for this model. Consider "
                "pausing deployment until disparities are addressed."
            )
            recommendations.append(
                "Engage with stakeholders from affected groups to understand "
                "the real-world impact of these disparities."
            )
            recommendations.append(
                "Document all findings and remediation efforts for regulatory "
                "compliance and audit purposes."
            )

        # Add metric-specific recommendations. The family is resolved on whole
        # tokens through the shared resolver (see _metric_family above), never by
        # testing whether a family token appears somewhere inside the name.
        family = _metric_family(metric_name)
        if family is not None:
            recommendations.append(_FAMILY_RECOMMENDATIONS[family])

        return " ".join(recommendations)

    @staticmethod
    def _effect_size_unmeasured_reason(
        value: Any, effect_type: Optional[str] = None
    ) -> Optional[str]:
        """Why this effect size is a could-not-check, or None when it is real.

        ``is_measured`` is the repo-wide predicate and answers the question, with
        exactly ONE carve-out, spelled out below: it rejects the infinities, and
        an infinite RATIO is a measured extreme rather than a missing
        measurement.

        ``effect_type`` decides whether the carve-out applies. It is optional
        only so a caller that genuinely does not know the producer still gets an
        answer, and that answer FAILS CLOSED: an infinity of unknown provenance
        is a could-not-check.
        """
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return f"the value is {value!r}, which is not a number"
        v = float(value)
        if is_measured(v):
            return None
        if np.isinf(v) and effect_type in _RATIO_EFFECT_TYPES:
            # THE CARVE-OUT, and it is now SCOPED to the two effect types its own
            # justification names. `_triage.is_measured` rejects inf because most
            # producers reach it by dividing by an empty denominator. These two
            # do not: risk_ratio and odds_ratio return inf only for a table that
            # WAS measured and in which one arm received no positive outcomes at
            # all. Refusing it would throw away the strongest disparate impact
            # finding the pair can produce.
            #
            # BGL-5 (2026-09-27). The carve-out used to ignore `effect_type`
            # entirely, so it exempted EVERY infinity. An infinite standardised
            # mean difference can only come from a zero standardising
            # denominator, i.e. exactly the empty-denominator case this predicate
            # says it rejects. Measured at the public entry before this change:
            #   explain_effect_size("cohens_d", float("inf"), "A", "B")
            #     -> severity "high", "Comparing A vs B: Large effect size - the
            #        difference is very substantial."
            #   explain_effect_size("cohens_d", float("-inf"), "A", "B")
            #     -> severity "high", same prose
            # and after:
            #   both -> severity "could_not_check", "COULD NOT CHECK: comparing A
            #        vs B, the cohens_d was not measured (it is inf, which can
            #        only come from a standardising denominator of zero ...)"
            # The carve-out itself still holds where it was argued for:
            #   explain_effect_size("risk_ratio", float("inf"), "A", "B")
            #     -> severity "critical", "Total exclusion: ..." (unchanged).
            # A membership test against an explicit set, never a substring test
            # on the name: this repo has already been bitten by `"ratio" in name`
            # also matching cali-BRATIO-n.
            return None
        if np.isinf(v):
            return (
                f"it is {v}, which can only come from a standardising denominator "
                f"of zero, so no effect size was computed"
            )
        return "it is NaN, so no effect size was computed"

    def _interpret_cohens_d(self, d: float) -> tuple:
        """Interpret Cohen's d effect size.

        BGL-S2c (2026-09-17). ``abs(nan) < 0.2`` is False, and so is every later
        threshold, so an effect size that was never computed fell out of the
        bottom of this ladder as "Large effect size", severity "high": the most
        alarming verdict the method can produce, about nothing at all. The same
        guard now sits above the dispatch in ``explain_effect_size``; it is
        repeated here because these three methods are called directly too.
        """
        reason = self._effect_size_unmeasured_reason(d, "cohens_d")
        if reason is not None:
            return (
                f"COULD NOT CHECK: no Cohen's d was measured ({reason}), so this "
                "is neither a negligible effect nor a large one.",
                "could_not_check",
            )

        abs_d = abs(d)

        if abs_d < 0.2:
            return "Negligible effect size - the difference is minimal.", "info"
        elif abs_d < 0.5:
            return "Small effect size - the difference is noticeable but modest.", "low"
        elif abs_d < 0.8:
            return "Medium effect size - the difference is substantial and meaningful.", "medium"
        else:
            return "Large effect size - the difference is very substantial.", "high"

    def _interpret_risk_ratio(self, rr: float) -> tuple:
        """Interpret risk ratio.

        BGL-S2c (2026-09-17). Every band test is False for NaN, so an unmeasured
        ratio reached the final ``else``, where ``rr > 1`` is also False, and the
        method announced "Large difference: one group is nanx less likely." at
        severity "high". A DEGENERATE ratio (inf, or a measured 0.0) is a
        genuine measurement, one arm received no positive outcomes at all, and
        is graded "critical" rather than swept into the third state; only its
        wording changes, because "infx more likely" is not a sentence a reader
        can act on.
        """
        reason = self._effect_size_unmeasured_reason(rr, "risk_ratio")
        if reason is not None:
            return (
                f"COULD NOT CHECK: no risk ratio was measured ({reason}), so this "
                "says nothing about which group is more likely to receive the "
                "outcome.",
                "could_not_check",
            )

        # Both degenerate ends are MEASURED total exclusion, and both used to
        # fall into the final `else`, where `factor = 1 / rr` is computed:
        #   rr == inf  ->  "one group is infx more likely"
        #   rr == 0.0  ->  ZeroDivisionError, raised out of explain_report
        # The second was found by re-reading this function after fixing the
        # NaN above, on real data (80 rows, group A receives nothing, group B
        # receives 20): compute_effect_sizes gives risk_ratio (0.0, nan, nan)
        # and explaining that report crashed.
        if rr == 0.0 or np.isinf(rr):
            starved = "The first group" if rr == 0.0 else "The second group"
            return (
                f"Total exclusion: {starved} received no positive outcomes at all, "
                f"so the risk ratio is {'zero' if rr == 0.0 else 'unbounded'} rather "
                f"than a finite multiple. This is the strongest disparate impact "
                f"reading available, not a missing measurement.",
                "critical",
            )

        if 0.9 <= rr <= 1.1:
            return "Near-equal risk between groups.", "info"
        elif 0.8 <= rr <= 1.25:
            return (
                f"Small difference: one group is {abs(rr - 1) * 100:.0f}% more/less likely.",
                "low",
            )
        elif 0.67 <= rr <= 1.5:
            return (
                f"Moderate difference: one group is {abs(rr - 1) * 100:.0f}% more/less likely.",
                "medium",
            )
        else:
            direction = "more" if rr > 1 else "less"
            factor = rr if rr > 1 else 1 / rr
            return f"Large difference: one group is {factor:.1f}x {direction} likely.", "high"

    def _interpret_odds_ratio(self, odds_ratio: float) -> tuple:
        """Interpret odds ratio.

        BGL-S2c (2026-09-17). Same ladder, same NaN behaviour: an odds ratio
        that was never computed fell through to "Large difference in odds
        between groups." at severity "high". ``odds_ratio`` in _statistics.py
        returns NaN for the 0/0 table and inf / 0.0 for the genuinely one-sided
        ones, so the infinities stay graded and only NaN is the third state.
        """
        reason = self._effect_size_unmeasured_reason(odds_ratio, "odds_ratio")
        if reason is not None:
            return (
                f"COULD NOT CHECK: no odds ratio was measured ({reason}), so this "
                "says nothing about whose odds are higher.",
                "could_not_check",
            )

        if odds_ratio == 0.0 or np.isinf(odds_ratio):
            starved = "the first group" if odds_ratio == 0.0 else "the second group"
            return (
                f"Total exclusion: the odds are degenerate because {starved} has no "
                f"positive outcomes to form odds from. This is a measured extreme, "
                f"not a missing measurement.",
                "critical",
            )

        if 0.9 <= odds_ratio <= 1.1:
            return "Near-equal odds between groups.", "info"
        elif 0.75 <= odds_ratio <= 1.33:
            return "Small difference in odds between groups.", "low"
        elif 0.5 <= odds_ratio <= 2.0:
            return "Moderate difference in odds between groups.", "medium"
        else:
            return "Large difference in odds between groups.", "high"

    def _effect_size_recommendation(
        self,
        effect_type: str,
        value: float,
        severity: str,
        kind: str = "unmeasured",
    ) -> str:
        """Generate recommendation for effect size.

        BGL-S2c (2026-09-17). The final ``else`` is the catch-all, so an
        ungraded severity was handed the URGENT text ("The effect is large and
        requires action") alongside a value nobody measured. ``could_not_check``
        is named explicitly and first, so a new severity can never inherit the
        alarm by falling off the end.

        ``kind`` separates the TWO could-not-checks, added 2026-09-29 with the
        ungraded-type fix. "unmeasured" is a value that does not exist;
        "ungraded_type" is a real value this library has no band ladder for.
        Handing the second one the first one's text would be the mirror defect,
        a MEASUREMENT reported as a could-not-check, and it would tell a reader
        to go looking for an empty denominator that is not there.
        """
        if severity == "could_not_check" and kind == "ungraded_type":
            return (
                f"The value is real; what is missing is the yardstick. This library "
                f"grades cohens_d, risk_ratio and odds_ratio, and has no interpretation "
                f"bands for {effect_type}, so nothing here says whether this value is "
                f"large or small. Interpret it against a published benchmark for "
                f"{effect_type} yourself, or use one of the graded measures. Do not read "
                f"the absence of a band as a pass."
            )
        if severity == "could_not_check":
            return (
                "There is nothing to act on yet: this effect size was not "
                "measured, so it is neither a clean result nor a breach. Find "
                "out why it is missing before reading anything into it; the "
                "usual causes are an empty denominator (no observations, or no "
                "positive outcomes anywhere) and a rate that was itself never "
                "measured. Re-run with enough data per group. An absent effect "
                "size is not evidence either way."
            )
        if severity == "info":
            return "No action needed - the effect size is negligible."
        elif severity == "low":
            return (
                "The effect is small but worth monitoring. Consider whether "
                "this level of disparity is acceptable for your use case."
            )
        elif severity == "medium":
            return (
                "The effect is moderate and warrants investigation. Review the "
                "model and data for potential sources of bias."
            )
        else:
            return (
                "The effect is large and requires action. Investigate root causes "
                "and consider fairness interventions before deployment."
            )

    def _generate_summary(self, report: Dict[str, Any], explanations: Dict[str, Any]) -> str:
        """Generate an overall summary of the fairness analysis."""
        assessment = report.get("assessment", {})
        fairness_score = assessment.get("fairness_score", 0)
        passed = assessment.get("passed_metrics", [])
        failed = assessment.get("failed_metrics", [])
        not_assessable = assessment.get("not_assessable_metrics", [])
        # Absent key: assume assessable, so an old-shaped report keeps its prose.
        assessable = bool(assessment.get("assessable", True))

        # "Could not check" is a THIRD state and must never collapse into either
        # of the other two. The score is None (or NaN) exactly when no metric was
        # assessable; formatting it as a percentage would crash, and the old
        # "all metrics are within acceptable thresholds" line below said the
        # model passed when in truth nothing had been compared at all.
        score_is_number = isinstance(fairness_score, (int, float)) and not (
            isinstance(fairness_score, float) and np.isnan(fairness_score)
        )
        could_not_check = (not assessable) or (not score_is_number) or (not passed and not failed)

        # Count severities
        # `could_not_check` needs its own bucket. Without one it landed in the
        # dict via .get() anyway, but nothing below ever mentioned it, so ungraded
        # metrics were counted and then never reported.
        severities = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
            "could_not_check": 0,
        }
        for metric_explanation in explanations.get("metrics", {}).values():
            sev = metric_explanation.get("severity", "info")
            severities[sev] = severities.get(sev, 0) + 1

        # Build summary
        summary_parts = []

        if score_is_number:
            summary_parts.append(
                f"Overall Fairness Score: {fairness_score:.1%} "
                f"({len(passed)} passed, {len(failed)} failed"
                + (f", {len(not_assessable)} not assessable)" if not_assessable else ")")
            )
        else:
            summary_parts.append(
                f"Overall Fairness Score: NOT AVAILABLE (could not check) "
                f"({len(passed)} passed, {len(failed)} failed, "
                f"{len(not_assessable)} not assessable)"
            )

        if severities["critical"] > 0:
            summary_parts.append(
                f"CRITICAL: {severities['critical']} metric(s) show critical disparities "
                f"requiring immediate attention."
            )

        if severities["high"] > 0:
            summary_parts.append(
                f"HIGH CONCERN: {severities['high']} metric(s) show high disparities "
                f"that should be addressed before deployment."
            )

        if severities["medium"] > 0:
            summary_parts.append(
                f"MODERATE CONCERN: {severities['medium']} metric(s) show concerning "
                f"disparities that warrant investigation."
            )

        if severities["could_not_check"] > 0:
            summary_parts.append(
                f"NOT GRADED: {severities['could_not_check']} metric(s) were never "
                f"compared to a threshold, so they are neither passing nor failing. "
                f"Do not read their absence from the counts above as a pass."
            )

        if could_not_check:
            # NEVER say "all metrics are within acceptable thresholds" here. With
            # a single valid group every disparity metric used to return its
            # "perfect" sentinel, so len(failed) == 0 and this prose declared the
            # model fair on a comparison that never ran. This is the could-not-
            # check state and it must read as one.
            summary_parts.append(
                "COULD NOT CHECK: this run did not compare groups, so it neither "
                "passes nor fails. "
                + (
                    "Fewer than two groups met min_group_size. "
                    if not assessable
                    else "No metric could be computed on this data. "
                )
                + "Lower min_group_size or collect more data for the affected "
                "groups, then re-run. Nothing here certifies fairness."
            )
        elif len(failed) == 0:
            summary_parts.append(
                "All assessable metrics are within acceptable thresholds. Continue "
                "regular monitoring to ensure ongoing fairness."
                + (
                    f" Note: {len(not_assessable)} metric(s) could not be assessed "
                    f"and are not covered by that statement."
                    if not_assessable
                    else ""
                )
            )
        else:
            failed_names = [f["metric"] for f in failed]
            summary_parts.append(f"Metrics requiring attention: {', '.join(failed_names)}")

        return " | ".join(summary_parts)

    @staticmethod
    def _format_unassessed_value(value: Any) -> str:
        """The value as measured, for prose that grades NOTHING.

        Two jobs. First, never crash: this branch already means the library
        could not say what the metric is, and the old body ran
        ``f"{value:.4f}"`` unguarded, so a non-numeric custom value raised
        ``ValueError: Unknown format code 'f' for object of type 'str'`` in the
        middle of building a report. Second, never let an uncomputed value read
        as a measurement: ``f"{float('nan'):.4f}"`` renders the bare word "nan"
        in a sentence otherwise shaped like a result.
        """
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return repr(value)
        if as_float != as_float:  # NaN test; also catches np.float32 NaN
            return "NaN (never computed)"
        return f"{as_float:.4f}"

    def _generic_explanation(
        self, metric_name: str, value: float, threshold: Optional[float] = None
    ) -> MetricExplanation:
        """COULD-NOT-CHECK for a metric this library holds no definition for.

        ``threshold`` is the bound the CALLER applied, if any. When one was
        applied this branch must NOT claim the metric went ungraded: it did not
        go ungraded, it simply has no prose definition here. Saying otherwise is
        a false claim in the opposite direction from the one this branch was
        written to remove, and a worse one, because it tells a reader to
        disregard a verdict the report really reached.

        Reached from :meth:`explain_metric` whenever the name has no entry in
        ``self.metrics_definitions`` and none in :data:`STATISTICAL_MEASURES`.
        Nothing was looked up, so no threshold was applied and no comparison was
        performed: this metric was NOT assessed.

        **Never reassure here, and never look like the graded path.** Executed
        on this repo before 2026-08-27,
        ``explain_metric("disparate_impact_ratio", 0.60)`` returned evaluation
        "The computed value is 0.6000.", recommendation "Review this metric in
        the context of your specific use case." and severity ``"info"``. Not one
        word of that is false, which is exactly why it survived every review.
        The untruth is the SHAPE: ``"info"`` is the severity a genuinely graded,
        genuinely benign metric receives, and the prose is as unbothered as the
        prose beside a passing one, so a reader could not tell "we checked and
        it is fine" from "we have no idea what this metric is". Severity
        ``"could_not_check"`` exists for this branch and for no other reason.

        A direction that the shared resolver CAN place from the name alone is
        used, because saying something true beats saying nothing. It is stated
        as what it is, a reading of the name, and it is never turned into a
        verdict: knowing which way is better does not supply the bound that was
        never applied.
        """
        shown = self._format_unassessed_value(value)

        direction = metric_direction(metric_name)
        if direction is MetricDirection.LOWER_IS_BETTER:
            direction_note = (
                " The name reads as a violation magnitude, so a smaller value is "
                "better, but that is a reading of the NAME, not a measurement: no "
                "bound is held for this metric, so the value was compared to nothing."
            )
        elif direction is MetricDirection.HIGHER_IS_BETTER:
            direction_note = (
                " The name reads as a parity ratio, so a larger value is better, "
                "but that is a reading of the NAME, not a measurement: no bound is "
                "held for this metric, so the value was compared to nothing."
            )
        else:
            direction_note = (
                " The name carries no reliable direction signal either, so this "
                "library cannot even say which way this metric would have to move "
                "to be better."
            )

        known_metrics = ", ".join(sorted(self.metrics_definitions))

        # A bound WAS applied by the caller. The metric is undocumented here, not
        # ungraded, and the two must not be conflated. Grade it in the metric's
        # own direction and say plainly that the verdict comes from the supplied
        # bound rather than from any definition this library holds.
        if threshold is not None and threshold == threshold:  # not None, not NaN
            if direction is MetricDirection.UNKNOWN:
                # A bound exists but there is no way to know which side of it is
                # good, so it genuinely could not be applied. Say exactly that,
                # and do not silently treat it as satisfied.
                return MetricExplanation(
                    metric_name=metric_name,
                    definition=(
                        f"NOT DEFINED in this library: '{metric_name}' is not a metric "
                        f"vfairness holds a definition for."
                    ),
                    interpretation_guide=(
                        "No interpretation guide is held for this metric and none is guessed."
                    ),
                    value=value,
                    evaluation=(
                        f"COULD NOT CHECK: a bound of {threshold} was supplied for "
                        f"'{metric_name}', but this library cannot tell whether a "
                        f"higher or a lower value is better for it, so the bound "
                        f"could not be applied to the value {shown}. This is neither "
                        f"a pass nor a fail."
                    ),
                    benchmark_context=("No benchmark or legal context is held for this metric."),
                    recommendation=(
                        f"State the direction for '{metric_name}' (is a smaller value "
                        f"better, or a larger one?) by adding it to the explainer's "
                        f"metrics_definitions, then re-run. Until then, treat this "
                        f"metric as unassessed, not as passing."
                    ),
                    severity="could_not_check",
                    related_metrics=[],
                )

            if direction is MetricDirection.HIGHER_IS_BETTER:
                is_pass = float(value) >= float(threshold)
                rule = f"at least {threshold}"
            else:
                is_pass = abs(float(value)) <= float(threshold)
                rule = f"at most {threshold}"
            verdict = "within" if is_pass else "outside"
            return MetricExplanation(
                metric_name=metric_name,
                definition=(
                    f"NOT DEFINED in this library: '{metric_name}' is not a metric "
                    f"vfairness holds a prose definition for, so what it measures and "
                    f"why it matters are not described here. A bound WAS supplied for "
                    f"it, so it was still graded."
                ),
                interpretation_guide=(
                    f"No interpretation guide is held for this metric. The direction "
                    f"was read from the name: the rule applied was {rule}."
                ),
                value=value,
                evaluation=(
                    f"{'WITHIN' if is_pass else 'OUTSIDE'} the supplied bound: "
                    f"{shown} is {verdict} the bound of {rule}. This verdict comes "
                    f"from the bound you supplied, not from any benchmark this "
                    f"library holds for '{metric_name}'."
                ),
                benchmark_context=(
                    f"No benchmark or legal context is held for '{metric_name}'. The "
                    f"bound of {threshold} is the one supplied by the caller."
                ),
                recommendation=(
                    (
                        f"'{metric_name}' is within the bound you supplied. Add a "
                        f"definition for it to the explainer's metrics_definitions if "
                        f"you want this library to explain what it means."
                    )
                    if is_pass
                    else (
                        f"'{metric_name}' is outside the bound you supplied "
                        f"({shown} against {rule}). Investigate it as you would any "
                        f"breached metric; this library holds no remediation guidance "
                        f"for it because it holds no definition for it."
                    )
                ),
                severity="info" if is_pass else "critical",
                related_metrics=[],
            )

        return MetricExplanation(
            metric_name=metric_name,
            definition=(
                f"NOT DEFINED in this library: '{metric_name}' is not a metric "
                f"vfairness holds a definition for, so what it measures, what a "
                f"good value looks like and what would count as a breach are all "
                f"unknown here."
            ),
            interpretation_guide=(
                "No interpretation guide is held for this metric and none is "
                "guessed. Whoever defined the metric has to state how to read it."
            ),
            value=value,
            evaluation=(
                f"COULD NOT CHECK: '{metric_name}' has no definition in this "
                f"library, so no threshold was applied and the value {shown} was "
                f"not graded. Nothing here is an assessment of this metric: it is "
                f"neither a pass nor a fail." + direction_note
            ),
            benchmark_context=(
                "No benchmark, threshold or legal context is held for this metric, "
                "so there was nothing to compare the value against."
            ),
            recommendation=(
                f"NOT MEASURED: nothing here says '{metric_name}' is satisfactory. "
                f"To obtain a verdict, either add a definition for it (definition, "
                f"interpretation_guide and thresholds) to the explainer's "
                f"metrics_definitions and re-run the assessment, or use one of the "
                f"{self.task_type} metrics this library does define: "
                f"{known_metrics}. Until then, treat this metric as unassessed, "
                f"not as passing."
            ),
            severity="could_not_check",
            related_metrics=[],
        )


# Convenience Functions


def explain_fairness_report(
    report: Dict[str, Any],
    task_type: Literal["classification", "regression"] = "classification",
    y_std: Optional[float] = None,
    include_statistical: bool = True,
) -> Dict[str, Any]:
    """
    Generate explanations for a fairness report.

    This is a convenience function that creates a FairExplAIner and
    generates explanations for an entire report.

    Args:
        report: Report from classification_fairness_report or regression_fairness_report
        task_type: Type of ML task
        y_std: Standard deviation of target (for regression)
        include_statistical: Whether to include statistical explanations

    Returns:
        Dictionary with explanations for all report components

    Example:
        >>> report = classification_fairness_report(y_true, y_pred, gender)
        >>> explanations = explain_fairness_report(report)
        >>> print(explanations['summary'])
    """
    # Infer task type from report if possible
    if "task_type" in report:
        task_type = report["task_type"]

    # Get y_std from report if available
    if y_std is None and "data_info" in report:
        y_std = report["data_info"].get("y_std")

    explainer = FairExplAIner(task_type=task_type, y_std=y_std)
    return explainer.explain_report(report, include_statistical=include_statistical)


def print_explanations(explanations: Dict[str, Any]) -> None:
    """
    Print formatted explanations to console.

    Args:
        explanations: Dictionary from explain_fairness_report or FairExplAIner.explain_report

    THE WHOLE STATISTICAL SECTION USED TO BE DROPPED HERE, measured 2026-09-29
    through the public API with no hand-built dict:

        rep = FairnessAnalyzer(y, y, np.array(['a'] * 60)).get_report(include_ci=True)
        ex = explain_fairness_report(rep)          # include_statistical defaults True
        ex['statistical'].keys()
          -> ['demographic_parity_difference_ci', 'equalized_odds_difference_ci',
              'equal_opportunity_difference_ci']
        ex['statistical'][...]['severity']   -> 'could_not_check'
        ex['statistical'][...]['evaluation'] -> 'COULD NOT CHECK: no interval was
              computed for demographic_parity_difference ... neither a pass nor a fail.'

        print_explanations(ex)  -> 6751 characters, and:
            any statistical key printed        -> False
            the word STATISTICAL in the output -> False

    Three correctly computed, correctly worded could-not-checks reached the only
    console surface documented to render this dict and were printed nowhere, so a
    reader of the console saw five metric cards and no sign that the interval
    evidence behind them does not exist. An unrendered could-not-check is the
    omission the three-state rule names. The condensed sibling renderer,
    ``report._print_explanations_section``, already printed its own STATISTICAL
    MEASURES block, so this function was the outlier of the two.

    Every section is now announced whether or not it holds anything, an EMPTY
    section says so in words (silence reads as "nothing was left out"), and a key
    this function does not know how to render is named rather than skipped, so the
    next section added to the producer cannot vanish the same way.
    """
    import sys

    lines = []
    lines.append("=" * 70)
    lines.append("FAIREXPLAINER - Fairness Metrics Explained")
    lines.append("=" * 70)

    def _wrapped(text: str) -> None:
        """The card's own wrapping, unchanged, hoisted so both sections share it."""
        line = "│    "
        for word in str(text).split():
            if len(line) + len(word) > 75:
                lines.append(line)
                line = "│    " + word + " "
            else:
                line += word + " "
        if line.strip("│ "):
            lines.append(line)

    def _card(title: str, explanation: Dict[str, Any]) -> None:
        lines.append(f"\n┌─ {str(title).upper().replace('_', ' ')} ─┐")
        # The dict key verbatim, not only the prettified heading: it is what the
        # reader indexes to go and read the card themselves, and a search of the
        # printout for a key that WAS rendered now finds it.
        lines.append(f"│ 🔖 Key: {title}")
        lines.append("│")
        lines.append("│ 📖 Definition:")
        _wrapped(explanation.get("definition", "N/A"))
        lines.append("│")
        lines.append(f"│ 📈 Value: {explanation.get('value', 'N/A')}")
        # `.get('severity', 'N/A')` does NOT fire when the key is PRESENT holding
        # None, so `.upper()` on it raised AttributeError and killed the whole
        # printout, every later card included. An absent severity is also not
        # 'info': it is not stated, and it says so.
        severity = explanation.get("severity")
        severity_text = (
            str(severity).upper() if severity not in (None, "") else "NOT STATED IN THIS CARD"
        )
        lines.append(f"│ 🎯 Severity: {severity_text}")
        lines.append("│")
        lines.append("│ 🔍 Evaluation:")
        _wrapped(explanation.get("evaluation", "N/A"))
        lines.append("│")
        lines.append("│ 💡 Recommendation:")
        _wrapped(explanation.get("recommendation", "N/A"))
        lines.append("└" + "─" * 50 + "┘")

    # Print summary
    if "summary" in explanations:
        lines.append("\n📋 SUMMARY:")
        summary = explanations["summary"]
        lines.append(f"   {summary}" if summary else "   (the summary is empty)")

    # Print metric explanations
    lines.append("\n" + "=" * 70)
    lines.append("METRIC EXPLANATIONS")
    lines.append("=" * 70)

    metric_cards = explanations.get("metrics", {})
    if metric_cards:
        for metric_name, explanation in metric_cards.items():
            _card(metric_name, explanation)
    else:
        lines.append("\n   (no metric explanation in this dict, so nothing was graded here.")
        lines.append("    An empty section is not a set of passing metrics.)")

    # Print the statistical section: confidence intervals and effect sizes, which
    # is where the could-not-check cards live.
    lines.append("\n" + "=" * 70)
    lines.append("STATISTICAL MEASURES (confidence intervals and effect sizes)")
    lines.append("=" * 70)

    statistical_cards = explanations.get("statistical", {})
    if statistical_cards:
        for stat_name, explanation in statistical_cards.items():
            _card(stat_name, explanation)
    else:
        lines.append("\n   (no interval or effect-size explanation in this dict: none was")
        lines.append("    produced, or the producer was called with include_statistical=False.")
        lines.append("    Nothing here says the metrics above are statistically significant.)")

    # A section this function cannot render is NAMED, never skipped in silence:
    # that silence is exactly how the statistical section was lost for as long as
    # it was, and the producer may grow another section at any time.
    rendered_keys = {"summary", "metrics", "statistical"}
    unrendered = [key for key in explanations if key not in rendered_keys]
    if unrendered:
        lines.append("\n" + "=" * 70)
        lines.append("NOT RENDERED BY print_explanations")
        lines.append("=" * 70)
        lines.append(
            "   This function knows how to print "
            + ", ".join(sorted(rendered_keys))
            + f". These {len(unrendered)} key(s) were NOT printed and may hold evidence: "
            + ", ".join(str(key) for key in unrendered)
        )
        warnings.warn(
            f"print_explanations: {len(unrendered)} section(s) of this explanation dict were "
            f"not rendered: {', '.join(str(key) for key in unrendered)}. They are named in "
            "the printout rather than dropped, but read them yourself: what is not printed "
            "is not a pass.",
            UserWarning,
            stacklevel=2,
        )

    lines.append("\n" + "=" * 70)

    print("\n".join(lines), flush=True)
    sys.stdout.flush()
