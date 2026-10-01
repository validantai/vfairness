"""
Unified Calibration Analyzer for vfairness.

Provides a comprehensive, unified interface for all calibration capabilities.
Combines calibration evaluation, group-specific calibration, trade-off analysis,
and visualization into a single workflow.

The CalibrationAnalyzer class serves as the main entry point for calibration
analysis, producing standardized reports suitable for documentation and compliance.

Example:
    >>> from vfairness.post_processing.calibration import CalibrationAnalyzer
    >>>
    >>> analyzer = CalibrationAnalyzer(
    ...     y_true=labels,
    ...     y_prob=probabilities,
    ...     protected_attr=gender
    ... )
    >>>
    >>> # Full analysis
    >>> report = analyzer.full_analysis()
    >>> print(report.summary())
    >>>
    >>> # Or run specific analyses
    >>> metrics = analyzer.evaluate_calibration()
    >>> disparity = analyzer.analyze_disparity()
    >>> tradeoffs = analyzer.analyze_tradeoffs()
    >>>
    >>> # Apply calibration
    >>> calibrated = analyzer.calibrate()

References:
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
    - Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.
    - Naeini, M. P., et al. (2015). Obtaining Well Calibrated Probabilities.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_probabilities,
)

from .group_calibrator import (
    GroupCalibrationResult,
    GroupCalibrator,
    validate_fallback_strategy,
)
from .methods import (
    CalibrationMethod,
)
from .metrics import (
    BrierDecomposition,
    CalibrationCurveResult,
    CalibrationDisparityResult,
    CalibrationMetricResult,
    brier_score,
    brier_score_decomposition,
    calibration_curve,
    calibration_disparity,
    expected_calibration_error,
    maximum_calibration_error,
)
from .tradeoffs import (
    CalibrationRecommendation,
    TradeoffAnalysisResult,
    analyze_calibration_fairness_tradeoff,
    impossibility_diagnostics,
    recommend_calibration_strategy,
)


def _yes_no_unknown(verdict: Optional[bool]) -> str:
    """Render a three-state verdict; None is 'Not assessable', never 'No'.

    CAL-DISP (2026-09-09): the summary printed 'Yes' if verdict else 'No',
    which turned an unmeasured verdict (None) into a clean 'No'.
    """
    if verdict is None:
        return "Not assessable (not measured)"
    return "Yes" if verdict else "No"


_METRIC_ABSENT = object()


def _metric_text(value: Any) -> str:
    """Render one calibration metric for the text summary: three states.

    G005-SUM (2026-09-17). ``summary()`` printed each headline metric through
    ``self.overall_metrics.get('ece', 0)``, so a metric that was never supplied
    came out as ``0.0000``, which on the ECE / MCE / Brier scale is the BEST
    possible value: a perfectly calibrated model. Measured on a report built
    with ``overall_metrics={}`` the summary read "Expected Calibration Error
    (ECE): 0.0000 / Maximum Calibration Error (MCE): 0.0000 / Brier Score:
    0.0000" beside "Well Calibrated: Not assessable (not measured)": three
    fabricated perfect scores next to an honest refusal, from the same report.
    ``rendering.adapters.calibration_report_to_svg`` carries a standing "DO NOT
    reinstate the overall.get('ece', 0) seeds" comment for exactly this shape;
    the SVG surface was fixed and this text surface, which the class docstring
    tells readers to print, was not.

    A non-finite value is the same absence one step along: NaN is how the
    metric functions in this package refuse, so it is named rather than
    printed as the bare token "nan".
    """
    if value is _METRIC_ABSENT or value is None:
        return "not measured (no value supplied)"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return f"not measured (unusable value of type {type(value).__name__})"
    if not np.isfinite(as_float):
        return "not measured (undefined on this data)"
    return f"{as_float:.4f}"


def _percent_text(value: Any) -> str:
    """Render one percentage for the text summary: three states.

    The same rule as :func:`_metric_text`, in the ``.1%`` format the summary
    uses for base rates. G03 (2026-09-30): ``f"{nan:.1%}"`` is the bare token
    ``"nan%"``, which sits in a line labelled "Base Rate Disparity" as though a
    percentage had been computed. A finite value formats exactly as it did
    before, so the healthy summary is unchanged character for character.
    """
    if value is _METRIC_ABSENT or value is None:
        return "not measured (no value supplied)"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return f"not measured (unusable value of type {type(value).__name__})"
    if not np.isfinite(as_float):
        return "not measured (undefined on this data)"
    return f"{as_float:.1%}"


def _null_for_unmeasured(value: Any) -> Any:
    """Recursively replace every non-finite float with ``None`` for JSON.

    See ``CalibrationReport.to_json`` for the measurement and the reason. The
    numpy check is separate on purpose: ``np.float64`` subclasses ``float`` and
    ``np.float32`` does NOT, so an isinstance test against ``float`` alone
    lets a 32 bit NaN straight through to the encoder.
    """
    if isinstance(value, dict):
        return {k: _null_for_unmeasured(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_for_unmeasured(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_null_for_unmeasured(v) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


@dataclass
class CalibrationReport:
    """
    Comprehensive calibration analysis report.

    Attributes:
        timestamp: When the analysis was performed
        data_info: Basic information about the analyzed data
        protected_attribute: Name of the protected attribute analyzed
        n_groups: Number of demographic groups
        overall_metrics: Overall calibration metrics (ECE, MCE, Brier)
        group_metrics: Per-group calibration metrics
        disparity_analysis: Calibration disparity analysis results
        brier_decomposition: Brier score decomposition
        tradeoff_analysis: Calibration-fairness trade-off analysis
        recommendation: Recommended calibration strategy
        is_well_calibrated: Whether overall calibration is acceptable
            (None when the overall ECE could not be computed)
        has_significant_disparity: Whether calibration disparity exists
            (None when fewer than two groups were measured, so no
            between-group comparison was made; CAL-DISP, 2026-09-09.
            None as well when the spread was measured over a SUBSET of
            the groups and came back inside the threshold, because the
            excluded group can decide it either way; a measured breach
            on partial evidence keeps its True. BGL5, 2026-09-27, see
            CalibrationDisparityResult.has_significant_disparity)
        critical_issues: List of critical issues identified
        recommendations: Prioritized list of recommendations
        metadata: Additional metadata
    """

    timestamp: str
    data_info: Dict[str, Any]
    protected_attribute: str
    n_groups: int
    overall_metrics: Dict[str, float]
    group_metrics: Dict[str, Dict[str, float]]
    disparity_analysis: CalibrationDisparityResult
    brier_decomposition: Optional[BrierDecomposition]
    tradeoff_analysis: Optional[TradeoffAnalysisResult]
    recommendation: Optional[CalibrationRecommendation]
    is_well_calibrated: Optional[bool]
    has_significant_disparity: Optional[bool]
    critical_issues: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary of the analysis."""
        lines = [
            "=" * 70,
            "CALIBRATION ANALYSIS REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Samples: {self.data_info.get('n_samples', 'N/A')}",
            f"Groups: {self.n_groups} ({self.protected_attribute})",
            "",
            "-" * 70,
            "OVERALL CALIBRATION QUALITY",
            "-" * 70,
            "Expected Calibration Error (ECE): "
            + _metric_text(self.overall_metrics.get("ece", _METRIC_ABSENT)),
            "Maximum Calibration Error (MCE): "
            + _metric_text(self.overall_metrics.get("mce", _METRIC_ABSENT)),
            "Brier Score: " + _metric_text(self.overall_metrics.get("brier", _METRIC_ABSENT)),
            f"Well Calibrated: {_yes_no_unknown(self.is_well_calibrated)}",
            "",
        ]

        if self.group_metrics:
            lines.extend(
                [
                    "-" * 70,
                    "GROUP-SPECIFIC CALIBRATION (ECE)",
                    "-" * 70,
                ]
            )
            for group, metrics in self.group_metrics.items():
                # Same default, same consequence, one level down: a group whose
                # ECE was never supplied read as 0.0000, the best score on the
                # scale, in the per-group table (G005-SUM, 2026-09-17).
                getter = getattr(metrics, "get", None)
                ece = getter("ece", _METRIC_ABSENT) if callable(getter) else _METRIC_ABSENT
                lines.append(f"  {group}: {_metric_text(ece)}")
            lines.append("")

        lines.extend(
            [
                "-" * 70,
                "CALIBRATION DISPARITY",
                "-" * 70,
                (
                    f"ECE Disparity: {self.disparity_analysis.ece_disparity:.4f}"
                    if np.isfinite(self.disparity_analysis.ece_disparity)
                    else "ECE Disparity: not measured (fewer than two groups)"
                ),
                f"Significant Disparity: {_yes_no_unknown(self.has_significant_disparity)}",
                f"Most Miscalibrated: {self.disparity_analysis.most_miscalibrated_group}",
                f"Least Miscalibrated: {self.disparity_analysis.least_miscalibrated_group}",
                "",
            ]
        )

        if self.brier_decomposition:
            lines.extend(
                [
                    "-" * 70,
                    "BRIER SCORE DECOMPOSITION",
                    "-" * 70,
                    # Through _metric_text, which formats a finite value with the
                    # identical ``.4f`` these lines already used, so a healthy
                    # summary is unchanged character for character. G03
                    # (2026-09-30): ``skill_score`` is deliberately NaN when
                    # every label is one class, because there is no climatology
                    # to be more skilful than, and this line printed the bare
                    # token "nan" under the label "Skill Score" as though a score
                    # had been computed. The other three go through it for the
                    # same reason: a zero-row decomposition makes all four NaN.
                    "Reliability (calibration): "
                    + _metric_text(self.brier_decomposition.reliability),
                    "Resolution (discrimination): "
                    + _metric_text(self.brier_decomposition.resolution),
                    "Uncertainty (base rate): "
                    + _metric_text(self.brier_decomposition.uncertainty),
                    "Skill Score: " + _metric_text(self.brier_decomposition.skill_score),
                    "",
                ]
            )

        if self.tradeoff_analysis:
            lines.extend(
                [
                    "-" * 70,
                    "CALIBRATION-FAIRNESS TRADE-OFF",
                    "-" * 70,
                    f"Trade-off Severity: {self.tradeoff_analysis.tradeoff_severity}",
                    "Base Rate Disparity: "
                    + _percent_text(self.tradeoff_analysis.base_rate_disparity),
                    # THREE states, through the same renderer as every other
                    # verdict in this summary.
                    #
                    # G03 (2026-09-30). This was
                    # ``impossibility_diagnosis.get('impossibility_applies', False)``
                    # interpolated raw, which is two defects in one line. The
                    # DEFAULT is a neutral False: measured on a report whose
                    # ``impossibility_diagnosis`` is ``{}`` the summary printed
                    # "Impossibility Applies: False", a verdict on a theorem
                    # nobody applied. And when the key IS present holding the
                    # three-state ``None`` that ``impossibility_diagnostics``
                    # returns for an unmeasurable base-rate comparison, the line
                    # printed the bare token "None", while every sibling verdict
                    # in this same method goes through ``_yes_no_unknown`` and
                    # reads "Not assessable (not measured)". Same method, same
                    # kind of verdict, two renderings, one of which collapses
                    # could-not-check into a word a reader takes for "No".
                    "Impossibility Applies: "
                    + _yes_no_unknown(
                        self.tradeoff_analysis.impossibility_diagnosis.get("impossibility_applies")
                    ),
                    "",
                ]
            )

        if self.recommendation:
            lines.extend(
                [
                    "-" * 70,
                    "RECOMMENDED STRATEGY",
                    "-" * 70,
                    f"Strategy: {self.recommendation.strategy}",
                    f"Priority: {self.recommendation.priority}",
                    f"Rationale: {self.recommendation.rationale}",
                ]
            )
            if self.recommendation.expected_improvement:
                lines.append(
                    f"Expected Improvement: {self.recommendation.expected_improvement:.4f}"
                )
            if self.recommendation.tradeoff_warning:
                lines.append(f"Warning: {self.recommendation.tradeoff_warning}")
            lines.append("")

        if self.critical_issues:
            lines.extend(
                [
                    "-" * 70,
                    "CRITICAL ISSUES",
                    "-" * 70,
                ]
            )
            for i, issue in enumerate(self.critical_issues[:5], 1):
                lines.append(f"{i}. [{issue['type']}] {issue['description']}")
            lines.append("")

        if self.recommendations:
            lines.extend(
                [
                    "-" * 70,
                    "ACTION ITEMS",
                    "-" * 70,
                ]
            )
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "data_info": self.data_info,
            "protected_attribute": self.protected_attribute,
            "n_groups": self.n_groups,
            "overall_metrics": self.overall_metrics,
            "group_metrics": self.group_metrics,
            "disparity_analysis": self.disparity_analysis.to_dict(),
            "brier_decomposition": self.brier_decomposition.to_dict()
            if self.brier_decomposition
            else None,
            "tradeoff_analysis": self.tradeoff_analysis.to_dict()
            if self.tradeoff_analysis
            else None,
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "is_well_calibrated": self.is_well_calibrated,
            "has_significant_disparity": self.has_significant_disparity,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string, with could-not-check as JSON ``null``.

        G005-JSON (2026-09-17). ``json.dumps`` writes a bare ``NaN`` token,
        which is not JSON: Python's own ``json.loads`` accepts it as an
        extension, and every strict parser refuses it. Measured on a
        single-group report, the string carried ``"ece_disparity": NaN``,
        ``"mce_disparity": NaN``, ``"brier_disparity": NaN``,
        ``"base_rate_disparity": NaN`` and more, and ``JSON.parse`` in node
        rejected the WHOLE document with "Unexpected token 'N'". So the one
        report that has something important to say, that its disparity could
        not be measured, was the one a consumer could not read at all, and the
        obvious local repair for whoever hit it is to put a 0.0 back.

        ``None`` is the encoding this repo already uses for the third state at
        a JSON boundary (see the BGL-S2c note in operations/pulse/
        orchestrator.py: "the JSON null the consumer already renders as
        could-not-check"). The key stays present holding null, so null is still
        distinguishable from a measured number and from an absent field.
        ``to_dict()`` is untouched and still carries the float NaN that the
        in-process consumers test with ``np.isfinite``.
        """
        return json.dumps(_null_for_unmeasured(self.to_dict()), indent=indent, default=str)

    def to_svg(self, save_path: Optional[str] = None) -> str:
        """Render this report as a polished SVG dashboard.

        Requires ``jinja2`` (install with ``pip install jinja2``).

        Parameters
        ----------
        save_path : str, optional
            If provided, also write the SVG to this file.

        Returns
        -------
        str
            Complete SVG markup.
        """
        from vfairness.rendering import calibration_report_to_svg

        return calibration_report_to_svg(self, save_path=save_path)


class CalibrationAnalyzer:
    """
    Unified calibration analysis system for comprehensive probability assessment.

    Combines four core calibration capabilities:
    A. Calibration Evaluation (ECE, MCE, Brier, reliability diagrams)
    B. Group-Specific Analysis (calibration disparity detection)
    C. Trade-off Analysis (calibration vs. fairness trade-offs)
    D. Calibration Application (group-specific calibration)

    The analyzer can be used for individual analyses or a full comprehensive
    analysis that produces a standardized report.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining demographic groups
        attribute_name: Optional name for the protected attribute
        n_bins: Number of bins for calibration metrics
        min_group_size: Minimum samples per group for analysis
        config: Optional configuration dictionary

    Example:
        >>> analyzer = CalibrationAnalyzer(
        ...     y_true=labels,
        ...     y_prob=probabilities,
        ...     protected_attr=gender,
        ...     attribute_name='gender'
        ... )
        >>>
        >>> # Full analysis
        >>> report = analyzer.full_analysis()
        >>> print(report.summary())
        >>>
        >>> # Individual analyses
        >>> metrics = analyzer.evaluate_calibration()
        >>> disparity = analyzer.analyze_disparity()
        >>> tradeoffs = analyzer.analyze_tradeoffs()
        >>>
        >>> # Apply calibration
        >>> calibrator = analyzer.fit_calibrator(method='isotonic')
        >>> calibrated = analyzer.transform(y_prob_new, protected_attr_new)

    References:
        - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
        - Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger rows: calibration, calibration_analysis. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attr: ArrayLike,
        *,
        attribute_name: Optional[str] = None,
        n_bins: int = 10,
        min_group_size: int = 30,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.y_true = coerce_to_array(y_true, "y_true")
        self.y_prob = coerce_to_array(y_prob, "y_prob")
        self.protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(self.y_true, self.y_prob, self.protected_attr)
        validate_probabilities(self.y_prob, "y_prob")

        self.attribute_name = attribute_name or "protected_attribute"
        self.n_bins = n_bins
        self.min_group_size = min_group_size
        self.config = config or {}

        self.group_manager = GroupManager(self.protected_attr, min_group_size=min_group_size)

        # Memoization caches keyed by the arguments that change the result
        # (a plain single-slot cache silently returned the first call's
        # result for a different strategy).
        self._ece_result: Dict[str, CalibrationMetricResult] = {}
        self._mce_result: Dict[str, CalibrationMetricResult] = {}
        self._brier_result: Optional[CalibrationMetricResult] = None
        self._disparity_result: Optional[CalibrationDisparityResult] = None
        self._tradeoff_result: Dict[Any, TradeoffAnalysisResult] = {}
        self._recommendation: Dict[str, CalibrationRecommendation] = {}
        self._brier_decomp: Dict[str, BrierDecomposition] = {}

        self._group_calibrator: Optional[GroupCalibrator] = None

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.y_true)

    @property
    def n_groups(self) -> int:
        """Number of demographic groups."""
        return self.group_manager.n_groups

    @property
    def groups(self) -> List[str]:
        """List of group names."""
        return self.group_manager.groups

    @property
    def base_rate(self) -> float:
        """Overall base rate (prevalence)."""
        return float(np.mean(self.y_true))

    @property
    def group_base_rates(self) -> Dict[str, float]:
        """Base rates per group."""
        rates = {}
        for group_name in self.groups:
            mask = self.group_manager.get_mask(group_name)
            rates[group_name] = float(np.mean(self.y_true[mask]))
        return rates

    def full_analysis(
        self,
        *,
        include_tradeoffs: bool = True,
        include_recommendation: bool = True,
        include_brier_decomposition: bool = True,
        context: Literal[
            "risk_assessment", "hiring", "healthcare", "lending", "general"
        ] = "general",
    ) -> CalibrationReport:
        """
        Perform comprehensive calibration analysis.

        Runs all calibration analyses and produces a unified report.

        Args:
            include_tradeoffs: Whether to include trade-off analysis
            include_recommendation: Whether to include strategy recommendation
            include_brier_decomposition: Whether to include Brier decomposition
            context: Application context for recommendations

        Returns:
            CalibrationReport with comprehensive findings
        """
        ece = self.evaluate_ece()
        mce = self.evaluate_mce()
        brier = self.evaluate_brier()
        disparity = self.analyze_disparity()

        brier_decomp = None
        if include_brier_decomposition:
            brier_decomp = self.decompose_brier()

        tradeoffs = None
        if include_tradeoffs:
            tradeoffs = self.analyze_tradeoffs()

        recommendation = None
        if include_recommendation:
            recommendation = self.get_recommendation(context=context)

        overall_metrics = {
            "ece": ece.overall_value,
            "mce": mce.overall_value,
            "brier": brier.overall_value,
        }

        group_metrics = {}
        if ece.group_values:
            for group in ece.group_values:
                group_metrics[group] = {
                    "ece": ece.group_values.get(group, np.nan),
                    "mce": mce.group_values.get(group, np.nan) if mce.group_values else np.nan,
                    "brier": brier.group_values.get(group, np.nan)
                    if brier.group_values
                    else np.nan,
                }

        # `nan < 0.05` is False, so an ECE that could not be computed reported
        # "not well calibrated": fail-closed, but still a verdict nobody measured.
        is_well_calibrated = (
            None if not np.isfinite(ece.overall_value) else bool(ece.overall_value < 0.05)
        )
        has_significant_disparity = disparity.has_significant_disparity

        critical_issues = self._identify_critical_issues(ece, mce, disparity, tradeoffs)

        recommendations = self._generate_recommendations(
            ece, mce, disparity, tradeoffs, recommendation
        )

        report = CalibrationReport(
            timestamp=datetime.now().isoformat(),
            data_info={
                "n_samples": self.n_samples,
                "n_positive": int(np.sum(self.y_true)),
                "base_rate": self.base_rate,
                "n_bins": self.n_bins,
            },
            protected_attribute=self.attribute_name,
            n_groups=self.n_groups,
            overall_metrics=overall_metrics,
            group_metrics=group_metrics,
            disparity_analysis=disparity,
            brier_decomposition=brier_decomp,
            tradeoff_analysis=tradeoffs,
            recommendation=recommendation,
            is_well_calibrated=is_well_calibrated,
            has_significant_disparity=has_significant_disparity,
            critical_issues=critical_issues,
            recommendations=recommendations,
            metadata={
                "config": self.config,
                "context": context,
                "group_base_rates": self.group_base_rates,
            },
        )

        return report

    def evaluate_calibration(self) -> Dict[str, CalibrationMetricResult]:
        """
        Evaluate all calibration metrics.

        Returns:
            Dictionary with 'ece', 'mce', and 'brier' results
        """
        return {
            "ece": self.evaluate_ece(),
            "mce": self.evaluate_mce(),
            "brier": self.evaluate_brier(),
        }

    def evaluate_ece(
        self, strategy: Literal["uniform", "quantile"] = "uniform"
    ) -> CalibrationMetricResult:
        """
        Compute Expected Calibration Error.

        Args:
            strategy: Binning strategy

        Returns:
            CalibrationMetricResult with ECE values
        """
        if strategy not in self._ece_result:
            self._ece_result[strategy] = expected_calibration_error(
                self.y_true,
                self.y_prob,
                self.protected_attr,
                n_bins=self.n_bins,
                strategy=strategy,
                min_group_size=self.min_group_size,
            )
        return self._ece_result[strategy]

    def evaluate_mce(
        self, strategy: Literal["uniform", "quantile"] = "uniform"
    ) -> CalibrationMetricResult:
        """
        Compute Maximum Calibration Error.

        Args:
            strategy: Binning strategy

        Returns:
            CalibrationMetricResult with MCE values
        """
        if strategy not in self._mce_result:
            self._mce_result[strategy] = maximum_calibration_error(
                self.y_true,
                self.y_prob,
                self.protected_attr,
                n_bins=self.n_bins,
                strategy=strategy,
                min_group_size=self.min_group_size,
            )
        return self._mce_result[strategy]

    def evaluate_brier(self) -> CalibrationMetricResult:
        """
        Compute Brier Score.

        Returns:
            CalibrationMetricResult with Brier score
        """
        if self._brier_result is None:
            self._brier_result = brier_score(
                self.y_true, self.y_prob, self.protected_attr, min_group_size=self.min_group_size
            )
        return self._brier_result

    def decompose_brier(
        self, strategy: Literal["uniform", "quantile"] = "uniform"
    ) -> BrierDecomposition:
        """
        Decompose Brier Score into reliability, resolution, and uncertainty.

        Args:
            strategy: Binning strategy

        Returns:
            BrierDecomposition with all components
        """
        if strategy not in self._brier_decomp:
            self._brier_decomp[strategy] = brier_score_decomposition(
                self.y_true, self.y_prob, n_bins=self.n_bins, strategy=strategy
            )
        return self._brier_decomp[strategy]

    def get_calibration_curve(
        self, group: Optional[str] = None, strategy: Literal["uniform", "quantile"] = "uniform"
    ) -> CalibrationCurveResult:
        """
        Get calibration curve data for reliability diagrams.

        Args:
            group: Optional group name for group-specific curve
            strategy: Binning strategy

        Returns:
            CalibrationCurveResult with curve data
        """
        if group is not None:
            mask = self.group_manager.get_mask(group)
            return calibration_curve(
                self.y_true[mask], self.y_prob[mask], n_bins=self.n_bins, strategy=strategy
            )
        else:
            return calibration_curve(
                self.y_true, self.y_prob, n_bins=self.n_bins, strategy=strategy
            )

    def analyze_disparity(self) -> CalibrationDisparityResult:
        """
        Analyze calibration disparities across groups.

        Returns:
            CalibrationDisparityResult with disparity analysis
        """
        if self._disparity_result is None:
            self._disparity_result = calibration_disparity(
                self.y_true,
                self.y_prob,
                self.protected_attr,
                n_bins=self.n_bins,
                min_group_size=self.min_group_size,
            )
        return self._disparity_result

    def analyze_tradeoffs(
        self,
        fairness_metric: Literal["fpr_parity", "fnr_parity", "demographic_parity"] = "fpr_parity",
        thresholds: Optional[List[float]] = None,
    ) -> TradeoffAnalysisResult:
        """
        Analyze calibration-fairness trade-offs.

        Args:
            fairness_metric: Which fairness metric to analyze
            thresholds: Decision thresholds to evaluate

        Returns:
            TradeoffAnalysisResult with trade-off analysis
        """
        cache_key = (
            fairness_metric,
            tuple(thresholds) if thresholds is not None else None,
        )
        if cache_key not in self._tradeoff_result:
            self._tradeoff_result[cache_key] = analyze_calibration_fairness_tradeoff(
                self.y_true,
                self.y_prob,
                self.protected_attr,
                thresholds=thresholds,
                fairness_metric=fairness_metric,
                n_bins=self.n_bins,
                min_group_size=self.min_group_size,
            )
        return self._tradeoff_result[cache_key]

    def get_impossibility_diagnosis(self) -> Dict[str, Any]:
        """
        Diagnose impossibility theorem conditions.

        The analyzer's own ``min_group_size`` decides which groups have a base
        rate here, exactly as it does for :meth:`analyze_disparity` and
        :meth:`analyze_tradeoffs`. It was the one delegating method that dropped
        it and took ``impossibility_diagnostics``' default of 30 instead, so the
        same object answered the same question two ways on one dataset.

        Measured 2026-09-27 on two 40-row groups with base rates 0.75 and 0.125,
        built with ``min_group_size=50``: ``analyze_disparity()`` excluded BOTH
        groups, returned ece_disparity NaN and warned "nothing here certifies
        fairness", while this method returned base_rate_disparity 0.625,
        base_rates_differ True, impossibility_applies True, ``excluded_groups``
        [] and ``n_groups_compared`` 2, with no warning at all. The exclusion
        fields actively denied the exclusion the object had just made. It also
        failed the other way, refusing a comparison the caller had configured:
        with ``min_group_size=5`` on two 10-row groups, ``analyze_tradeoffs()``
        measured a base-rate gap of 0.70 while this returned None / NaN.

        Returns:
            Dictionary with impossibility diagnostics
        """
        return impossibility_diagnostics(
            self.y_true,
            self.y_prob,
            self.protected_attr,
            min_group_size=self.min_group_size,
        )

    def get_recommendation(
        self,
        context: Literal[
            "risk_assessment", "hiring", "healthcare", "lending", "general"
        ] = "general",
    ) -> CalibrationRecommendation:
        """
        Get recommended calibration strategy.

        Args:
            context: Application context

        Returns:
            CalibrationRecommendation with strategy and rationale
        """
        if context not in self._recommendation:
            self._recommendation[context] = recommend_calibration_strategy(
                self.y_true, self.y_prob, self.protected_attr, context=context
            )
        return self._recommendation[context]

    def fit_calibrator(
        self,
        method: Union[str, CalibrationMethod] = "isotonic",
        fallback_strategy: Literal["global", "none"] = "global",
        method_params: Optional[Dict] = None,
    ) -> GroupCalibrator:
        """
        Fit a group-specific calibrator.

        Args:
            method: Calibration method
            fallback_strategy: Strategy for groups below min_group_size:
                'global' (use the globally fitted calibrator) or 'none'
                (leave them uncalibrated). 'borrow' is not implemented and
                is refused with NotImplementedError (F19, 2026-09-09: this
                entry point re-exposed it without the "not yet implemented"
                caveat GroupCalibrator carried, and it ran 'global' silently).
            method_params: Parameters for calibration method

        Returns:
            Fitted GroupCalibrator
        """
        validate_fallback_strategy(fallback_strategy)
        self._group_calibrator = GroupCalibrator(
            method=method,
            min_group_size=self.min_group_size,
            fallback_strategy=fallback_strategy,
            method_params=method_params,
        )
        self._group_calibrator.fit(self.y_true, self.y_prob, self.protected_attr)
        return self._group_calibrator

    def calibrate(
        self,
        y_prob: Optional[ArrayLike] = None,
        protected_attr: Optional[ArrayLike] = None,
        method: Union[str, CalibrationMethod] = "isotonic",
    ) -> np.ndarray:
        """
        Apply calibration to probabilities.

        If no fitted calibrator exists, one will be created.

        Args:
            y_prob: Probabilities to calibrate (default: training data)
            protected_attr: Protected attribute (default: training data)
            method: Calibration method (if fitting new calibrator)

        Returns:
            Calibrated probabilities
        """
        if self._group_calibrator is None:
            self.fit_calibrator(method=method)
        # fit_calibrator() always assigns _group_calibrator, so it is non-None here.
        assert self._group_calibrator is not None

        if y_prob is None:
            y_prob = self.y_prob
        if protected_attr is None:
            protected_attr = self.protected_attr

        y_prob = coerce_to_array(y_prob, "y_prob")
        protected_attr = coerce_to_array(protected_attr, "protected_attr")

        return self._group_calibrator.transform(y_prob, protected_attr)

    def transform(self, y_prob: ArrayLike, protected_attr: ArrayLike) -> np.ndarray:
        """
        Apply fitted calibrator to new data.

        Args:
            y_prob: Probabilities to calibrate
            protected_attr: Protected attribute for new data

        Returns:
            Calibrated probabilities

        Raises:
            RuntimeError: If no calibrator has been fitted
        """
        if self._group_calibrator is None:
            raise RuntimeError("No calibrator fitted. Call fit_calibrator() first.")

        y_prob = coerce_to_array(y_prob, "y_prob")
        protected_attr = coerce_to_array(protected_attr, "protected_attr")

        return self._group_calibrator.transform(y_prob, protected_attr)

    def get_calibration_result(self) -> GroupCalibrationResult:
        """
        Get detailed results from fitted calibrator.

        Returns:
            GroupCalibrationResult with pre/post metrics

        Raises:
            RuntimeError: If no calibrator has been fitted
        """
        if self._group_calibrator is None:
            raise RuntimeError("No calibrator fitted. Call fit_calibrator() first.")
        return self._group_calibrator.get_calibration_result()

    def evaluate_calibration_improvement(
        self, y_prob_calibrated: Optional[ArrayLike] = None
    ) -> Dict[str, Any]:
        """
        Evaluate improvement after calibration, IN SAMPLE.

        Every figure here is scored against ``self.y_true``, the same rows the
        calibrator was fitted on, so the "after" numbers are a training
        residual and not an estimate of calibration on unseen data. For the
        default ``method='isotonic'`` that residual is zero by construction:
        measured across four seeds and both bin counts, the after ECE came back
        between 0.0 and 3.0e-17 every time while the before ECE ranged over
        0.31 to 0.37, and the after ECE disparity was likewise ~1e-17 against a
        real before disparity of 0.029 to 0.089 (G005-INS, 2026-09-17). Read as
        a measurement, that says every calibration perfectly eliminates both
        miscalibration and its between-group disparity, on any data. It is the
        documented usage: the API reference prints ``ECE After`` straight from
        this dict.

        The numbers are not withheld, because they ARE measured and a fit
        residual of zero is itself informative about the fit. They are labelled
        instead: ``coverage`` says what the comparison covers, and a
        UserWarning names it for a caller who only reads the three numbers.

        Args:
            y_prob_calibrated: Calibrated probabilities (or use fitted
                calibrator). Supplying them does not make the comparison
                out-of-sample: they are still scored against this analyzer's
                own ``y_true``.

        Returns:
            Dictionary with ``before`` / ``after`` / ``improvement`` metrics and
            a ``coverage`` block carrying ``in_sample`` and ``note``.
        """
        warnings.warn(
            "evaluate_calibration_improvement: the after and improvement figures are "
            "IN SAMPLE. They score the calibrator against the same rows it was fitted "
            "on, so they measure fit residual, not calibration on unseen data; with "
            "isotonic regression the after ECE is ~0 whatever the data. Refit on a "
            "held-out split before quoting an improvement. See result['coverage']."
        )
        if y_prob_calibrated is None:
            if self._group_calibrator is None:
                raise RuntimeError("No calibrator fitted and no calibrated probs provided.")
            y_prob_calibrated = self._group_calibrator.transform(self.y_prob, self.protected_attr)
        else:
            y_prob_calibrated = coerce_to_array(y_prob_calibrated, "y_prob_calibrated")

        ece_before = self.evaluate_ece()

        ece_after = expected_calibration_error(
            self.y_true,
            y_prob_calibrated,
            self.protected_attr,
            n_bins=self.n_bins,
            min_group_size=self.min_group_size,
        )

        disparity_before = self.analyze_disparity()
        disparity_after = calibration_disparity(
            self.y_true,
            y_prob_calibrated,
            self.protected_attr,
            n_bins=self.n_bins,
            min_group_size=self.min_group_size,
        )

        return {
            "before": {
                "overall_ece": ece_before.overall_value,
                "group_ece": ece_before.group_values,
                "ece_disparity": disparity_before.ece_disparity,
            },
            "after": {
                "overall_ece": ece_after.overall_value,
                "group_ece": ece_after.group_values,
                "ece_disparity": disparity_after.ece_disparity,
            },
            "improvement": {
                "overall_ece": ece_before.overall_value - ece_after.overall_value,
                "ece_disparity": disparity_before.ece_disparity - disparity_after.ece_disparity,
            },
            "coverage": {
                "in_sample": True,
                "n_rows_scored": int(self.n_samples),
                "groups_measured": sorted(ece_after.group_values or {}),
                "note": (
                    "IN SAMPLE. The after and improvement figures score the calibrator "
                    "against the rows it was fitted on. They are a fit residual, not an "
                    "estimate of calibration on unseen data, and an after ECE near zero "
                    "is what isotonic regression produces on its own training rows "
                    "regardless of the data."
                ),
            },
        }

    def _identify_critical_issues(
        self,
        ece: CalibrationMetricResult,
        mce: CalibrationMetricResult,
        disparity: CalibrationDisparityResult,
        tradeoffs: Optional[TradeoffAnalysisResult],
    ) -> List[Dict[str, Any]]:
        """Identify critical issues from analyses.

        Three states, in a list. G005-CI (2026-09-17): every test below is a
        ``>`` against a threshold, and ``nan > 0.1`` is False, so a quantity
        that could not be computed produced no entry and no trace. Measured on
        a one-row analyzer (overall ECE and MCE both NaN, disparity NaN) this
        returned ``[]``, and an empty critical-issues list is read everywhere,
        by ``summary()`` (which drops the whole CRITICAL ISSUES section), by
        ``to_dict()`` and by the SVG panel, as "nothing critical was found".
        Nothing was found because nothing was looked at. The unmeasured
        quantities are now named FIRST, so a truncated list ([:4] in the SVG,
        [:5] in the summary) cannot drop the coverage note, and the measured
        findings below are untouched: this only ever adds entries.
        """
        issues = []

        not_assessed = []
        if not np.isfinite(ece.overall_value):
            not_assessed.append("overall expected calibration error (ECE)")
        if not np.isfinite(mce.overall_value):
            not_assessed.append("overall maximum calibration error (MCE)")
        if not np.isfinite(disparity.ece_disparity):
            not_assessed.append("between-group ECE disparity")
        if not_assessed:
            issues.append(
                {
                    "type": "Not Assessed",
                    "severity": "unknown",
                    "description": (
                        f"COULD NOT CHECK: {', '.join(not_assessed)} produced no value on "
                        "this data, so the thresholds that flag poor calibration and "
                        "calibration disparity were never applied to it. This is an "
                        "absence of measurement, not a finding that these are fine."
                    ),
                    "recommendation": (
                        "Check sample size per group against min_group_size before "
                        "reading the issues below as a complete list."
                    ),
                }
            )

        if ece.overall_value > 0.1:
            issues.append(
                {
                    "type": "Poor Calibration",
                    "severity": "high",
                    "description": f"Overall ECE ({ece.overall_value:.3f}) indicates poor calibration. "
                    "Probability predictions may be misleading.",
                    "recommendation": "Apply global or group-specific calibration.",
                }
            )

        if disparity.ece_disparity > 0.1:
            issues.append(
                {
                    "type": "Calibration Disparity",
                    "severity": "high",
                    "description": f"Large calibration disparity ({disparity.ece_disparity:.3f}) between groups. "
                    f"Group '{disparity.most_miscalibrated_group}' is most affected.",
                    "recommendation": "Apply group-specific calibration to ensure fair probability interpretation.",
                }
            )

        if mce.overall_value > 0.2:
            issues.append(
                {
                    "type": "Worst-Case Calibration",
                    "severity": "medium",
                    "description": f"High MCE ({mce.overall_value:.3f}) indicates severe miscalibration "
                    "in some probability ranges.",
                    "recommendation": "Review reliability diagram to identify problematic ranges.",
                }
            )

        if tradeoffs and tradeoffs.impossibility_diagnosis.get("impossibility_applies", False):
            issues.append(
                {
                    "type": "Impossibility Theorem",
                    "severity": "warning",
                    "description": f"Base rates differ ({tradeoffs.base_rate_disparity:.1%}). "
                    "Perfect calibration and error rate parity cannot both be achieved.",
                    "recommendation": "Document trade-off decision based on application context.",
                }
            )

        return issues

    def _generate_recommendations(
        self,
        ece: CalibrationMetricResult,
        mce: CalibrationMetricResult,
        disparity: CalibrationDisparityResult,
        tradeoffs: Optional[TradeoffAnalysisResult],
        recommendation: Optional[CalibrationRecommendation],
    ) -> List[str]:
        """Generate prioritized recommendations, coverage notices FIRST.

        BGL5 (2026-09-27). Every consumer of this list truncates it:
        ``explainer._explain_calibration`` keeps ``report.recommendations[:5]``
        and ``CalibrationReport.summary()`` prints ``[:5]``. MEASURED on 100 'a'
        + 100 'b' + 25 'c' with 'b' badly miscalibrated, the report's own nine
        recommendations put the only exclusion notice at index 5:

            [0] STRATEGY: group_specific_isotonic (Priority: medium)
            [1] Apply group-specific calibration to reduce ECE disparity from 0.822.
            [2] CRITICAL: High calibration disparity (0.822). ...
            [3] Group 'b' has poor calibration (ECE=0.900). ...
            [4] Large worst-case disparity (MCE diff=0.702). ...
            [5] NOT ASSESSED: groups ['c'] were excluded (fewer than 30 samples). ...

        so ``get_explanation().recommendations`` held indices 0..4 only and the
        ExplanationReport a reader meets carried NO trace that 'c' exists,
        beside a summary reading "Calibration analysis for 3 groups". After the
        fix the notice is index 0 of 9 and survives every truncation above it.
        A truncation that can drop the disclosure is not a formatting choice.

        The ordering is the one ``_identify_critical_issues`` already documents
        ("the unmeasured quantities are now named FIRST, so a truncated list
        cannot drop the coverage note"). This partitions rather than rewrites:
        the same recommendations come back, with the coverage notices moved to
        the front in their original relative order, so a fully measured analysis
        (no NOT ASSESSED and no COULD NOT CHECK line) is returned byte-identical
        to what it was.
        """
        recs = []

        if recommendation:
            recs.append(
                f"STRATEGY: {recommendation.strategy} (Priority: {recommendation.priority})"
            )

        if disparity.has_significant_disparity:
            recs.append(
                f"Apply group-specific calibration to reduce ECE disparity from "
                f"{disparity.ece_disparity:.3f}."
            )

        recs.extend(disparity.recommendations)

        if tradeoffs:
            recs.extend(tradeoffs.recommendations[:2])

        if ece.overall_value > 0.05:
            recs.append(
                "Consider isotonic regression for better calibration without strong assumptions."
            )

        if not recs:
            recs.append("Calibration quality is acceptable. Continue monitoring for drift.")

        # The coverage notices move to the front, in their original order. Tested
        # on the string a reader sees rather than on a flag, because these lines
        # arrive from three different producers (calibration_disparity,
        # analyze_tradeoffs and the recommendation) and none of them carries a
        # structured "this is a coverage note" field.
        coverage: List[str] = []
        findings: List[str] = []
        for rec in recs:
            upper = str(rec).upper()
            target = (
                coverage if ("NOT ASSESSED" in upper or "COULD NOT CHECK" in upper) else findings
            )
            target.append(rec)
        return coverage + findings

    def get_explanation(self, report=None):
        """Generate educational explanations for the calibration analysis.

        Parameters
        ----------
        report : CalibrationReport, optional
            Pre-computed report. If *None*, :meth:`full_analysis` is run.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_analysis()
        return FairnessExplainer.explain(report)

    def clear_cache(self):
        """Clear cached results to force re-computation."""
        self._ece_result = {}
        self._mce_result = {}
        self._brier_result = None
        self._disparity_result = None
        self._tradeoff_result = {}
        self._recommendation = {}
        self._brier_decomp = {}

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the data."""
        return {
            "n_samples": self.n_samples,
            "n_positive": int(np.sum(self.y_true)),
            "n_negative": int(np.sum(1 - self.y_true)),
            "base_rate": self.base_rate,
            "n_groups": self.n_groups,
            "groups": self.groups,
            "group_sizes": {
                group: int(np.sum(self.group_manager.get_mask(group))) for group in self.groups
            },
            "group_base_rates": self.group_base_rates,
            "prob_stats": {
                "min": float(np.min(self.y_prob)),
                "max": float(np.max(self.y_prob)),
                "mean": float(np.mean(self.y_prob)),
                "std": float(np.std(self.y_prob)),
            },
        }
