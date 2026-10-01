"""
Unified Fairness Training Analyzer.

This module provides the FairnessTrainingAnalyzer class, which serves as
the main entry point for analyzing and reporting on fairness-aware training.
It combines all training-time intervention capabilities into a unified
workflow with comprehensive reporting.

The analyzer integrates:
    - Loss function analysis
    - Constraint satisfaction monitoring
    - Regularization effectiveness
    - Calibration quality assessment
    - Comparative evaluation across methods

Example:
    >>> from vfairness.in_processing import FairnessTrainingAnalyzer
    >>>
    >>> analyzer = FairnessTrainingAnalyzer(
    ...     X=X_train, y=y_train, sensitive_attr=gender,
    ...     task_type='classification'
    ... )
    >>>
    >>> # Full analysis with recommendations
    >>> report = analyzer.full_analysis()
    >>> print(report.summary())

References:
    - Hardt et al. (2016): Equality of Opportunity
    - Agarwal et al. (2018): A Reductions Approach
    - Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    metric_direction,
)
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)

from .._triage import is_measured
from .constraints import (
    FairnessConstraintType,
    create_constraint,
)

# The three-state translation lives beside the flag it reads, in
# constraints/base.py, so this reporting layer and the sklearn wrappers cannot
# drift apart on what an unevaluable constraint looks like (R4C).
from .constraints.base import _reported_constraint

logger = logging.getLogger(__name__)

# The words a report prints where a constraint verdict would go when there is
# none. Kept as constants because three surfaces here print them and the reader
# must meet the same phrase on each; "0.0000" and "False" are what they replace.
_NOT_EVALUATED = "not measured (the constraint could not be evaluated)"
_NOT_ASSESSED = "not assessed (the constraint could not be evaluated)"
# For a fairness-analysis quantity other than the constraint verdict, e.g. a
# base-rate disparity with no pair of groups to compare.
_NOT_MEASURED = "not measured (the quantity could not be computed on this data)"

# Below this ratio of ptp(y) to max|y| a regression target's variance is
# numerical noise and not variation, so R^2 = 1 - MSE/Var(y) divides by a
# quantity that no longer measures the data. sqrt(eps) is the standard
# cancellation line: `y - mean(y)` carries an absolute error of about eps*max|y|
# per row, so at a relative spread of sqrt(eps) half the significant digits of
# every deviation are already gone. A data-scaled tolerance and never an exact
# equality with zero on a computed statistic; see the guard in
# FairnessTrainingAnalyzer.evaluate_baseline for the numbers that got past the
# absolute np.ptp test one ulp away from constant.
_DEGENERATE_RELATIVE_SPREAD = float(np.sqrt(np.finfo(float).eps))


def _measured(value: Any) -> bool:
    """True when *value* is a real number that a report may print as measured.

    None is the not-evaluated state, NaN is what this boundary used to hand on
    in its place, and a bool is not a magnitude (True would print as 1.0000, the
    worst possible disparity). All three are refused.

    READINESS-6, 2026-09-10. Delegated to ``_triage.is_measured``, which is the
    module whose docstring says it exists so this question is answered once.
    Before this there were six implementations of it and they disagreed on FIVE
    of nine inputs, including the canonical one classifying a finite
    ``np.float32`` as NOT measured. The local reasoning above is why THIS call
    site cares; the rule itself lives in one place.
    Infinity is refused too, which this did not do: the test was a bare NaN
    check (``float(value) == float(value)``). ``risk_ratio`` returns ``inf`` for
    a group with zero observations, and ``inf`` compares greater than every
    threshold, so an unmeasurable group silently became the worst breach in the
    report. That is rule 1 of the canonical module, spelled out there.
    """
    return is_measured(value)


def _unscorable_rows(*arrays: Any) -> int:
    """How many rows carry a value no accuracy or R^2 can be scored from.

    A row is unscorable when its target OR its prediction is non-finite: NaN
    and the infinities are not labels and not magnitudes, and every scoring
    formula in this module reads them silently. ``y_pred == y`` is False for a
    NaN on both sides (NaN equals nothing, itself included), so an unreadable
    row scores as a row the model got WRONG; ``1 - MSE/Var(y)`` propagates the
    NaN into both of its terms at once.

    Non-numeric labels contribute NOTHING here. A classification target of
    "approved"/"denied" has no finiteness to test and is perfectly scorable, so
    an array this cannot cast to float is passed over rather than counted as
    entirely unscorable, which would refuse every string-labelled problem in
    the library.
    """
    mask: Optional[np.ndarray] = None
    for arr in arrays:
        a = np.asarray(arr)
        try:
            bad = ~np.isfinite(a.astype(float, copy=False))
        except (TypeError, ValueError):
            # VISIBLE rather than silent. The docstring above says a non-numeric
            # array is passed over deliberately, because a target of
            # "approved"/"denied" has no finiteness to test and counting it as
            # unscorable would refuse every string-labelled problem in the
            # library. Debug, not a warning: this is the NORMAL case for such a
            # target, and a warning on a normal case trains readers to ignore
            # warnings. What it buys is that a reader of the logs can tell "there
            # was nothing here to test" from "this arm never ran".
            logger.debug(
                "_unlabelled_rows: an array of dtype %s cannot be cast to float, so it "
                "carries no finiteness to test and is passed over rather than counted "
                "as unscorable",
                getattr(a, "dtype", type(a).__name__),
            )
            continue
        bad = np.atleast_1d(bad)
        if bad.ndim > 1:
            bad = bad.reshape(bad.shape[0], -1).any(axis=1)
        mask = bad if mask is None else (mask | bad)
    return int(mask.sum()) if mask is not None else 0


def _as_measured(value: Any) -> Optional[float]:
    """*value* as a float when it is a real measurement, else None.

    The narrowing companion of :func:`_measured`: one call yields both the test
    and the number, so a threshold comparison cannot drift from the value it
    guards. Used where the alternative is comparing an absence against a
    tolerance, which is how "not measured" turns into "not a violation".
    """
    if not _measured(value):
        return None
    return float(value)


@dataclass
class MethodComparison:
    """
    Comparison of different fairness training methods.

    THREE STATES, NEVER TWO (R4C, 2026-09-10). ``ConstraintViolation`` records
    could-not-check honestly, and this boundary DISCARDED it: it copied
    ``is_satisfied``, which is deliberately False for an unevaluable constraint
    (fail-closed, because it steers training), into a field every REPORTING
    consumer reads as a measured verdict. Measured on 240 rows whose third group
    had no positive labels, ``evaluate_baseline`` returned
    ``fairness_violation=nan, constraint_satisfied=False`` and
    ``method_comparison_to_svg`` drew a red FAIL chip, a violation bar in the
    alert colour labelled "nan", and an accessible ``<desc>`` reading "0 of 1
    method(s) satisfy the fairness constraint. (severity: HIGH)". Nobody
    measured that failure: group C's TPR was undefined, so the disparity was
    unknown, not large.

    Read the two fields together:

        fairness_violation = 0.04, constraint_satisfied = True    met
        fairness_violation = 0.31, constraint_satisfied = False   measured breach
        fairness_violation = None, constraint_satisfied = None    NOT EVALUATED

    None is the third state the whole rendering layer already speaks:
    ``adapters_training._method_state`` gates the verdict chip on
    ``satisfied is not None`` and the number on ``violation is not None``, and
    ``rendering.explain._fr_method_comparison`` counts an ungraded row out of
    BOTH the numerator and the denominator. NaN is not a substitute for it: NaN
    is not None, so it passed both gates and was drawn as a measurement.

    Attributes:
        method_name: Name of the method
        accuracy: Model accuracy
        fairness_violation: Constraint violation magnitude, or None when the
            constraint could not be evaluated. Never NaN, never a substituted
            0.0.
        constraint_satisfied: True when the constraint was met, False for a
            MEASURED breach, None when it was never evaluated.
        training_time: Time taken for training
        parameters: Method-specific parameters, including
            ``constraint_evaluated`` for the comparisons this analyser builds.
    """

    method_name: str
    accuracy: float
    fairness_violation: Optional[float]
    constraint_satisfied: Optional[bool]
    training_time: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "accuracy": self.accuracy,
            "fairness_violation": self.fairness_violation,
            "constraint_satisfied": self.constraint_satisfied,
            "training_time": self.training_time,
            "parameters": self.parameters,
        }


@dataclass
class TrainingRecommendation:
    """
    Recommendation for fairness-aware training.

    Attributes:
        recommended_method: The recommended training method
        priority: Priority level ('high', 'medium', 'low')
        rationale: Explanation for the recommendation
        alternative_methods: Alternative methods to consider
        expected_tradeoff: Expected accuracy-fairness trade-off. A value is None
            where the recommended method reported no measurement for it, which
            for ``fairness_violation`` means the constraint was never evaluated
            (R4C). An empty dict means no method was recommended at all.
        implementation_notes: Notes for implementation
    """

    recommended_method: str
    priority: str
    rationale: str
    alternative_methods: List[str] = field(default_factory=list)
    expected_tradeoff: Dict[str, Optional[float]] = field(default_factory=dict)
    implementation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommended_method": self.recommended_method,
            "priority": self.priority,
            "rationale": self.rationale,
            "alternative_methods": self.alternative_methods,
            "expected_tradeoff": self.expected_tradeoff,
            "implementation_notes": self.implementation_notes,
        }


@dataclass
class FairnessTrainingReport:
    """
    Comprehensive fairness training analysis report.

    Attributes:
        timestamp: When the analysis was performed
        data_info: Information about the dataset
        task_type: Classification or regression
        baseline_metrics: Metrics without fairness constraints
        fairness_analysis: Analysis of fairness violations
        method_comparisons: Comparison of different methods
        recommendation: Recommended approach
        tradeoff_analysis: Accuracy-fairness trade-off analysis
        critical_issues: List of critical issues identified
        action_items: Prioritized action items
        metadata: Additional metadata
        failed_methods: The methods whose fit RAISED, each as
            {"method": name, "error": "Type: message"}. Empty when every
            requested method trained. A method that failed is absent from
            ``method_comparisons``, so without this field a report of a run in
            which two of three trainings crashed is byte-identical to a report
            in which only one method was requested. The sibling
            ``ReweightingAnalysisReport.failed_methods`` is the model, and it
            discloses whenever the list is non-empty rather than only when
            EVERY method failed (BGL4 audit, 2026-09-27).
    """

    timestamp: str
    data_info: Dict[str, Any]
    task_type: str
    # Values are Optional: `fairness_violation` and `constraint_satisfied` are
    # None when the baseline constraint could not be evaluated (R4C).
    baseline_metrics: Dict[str, Any]
    fairness_analysis: Dict[str, Any]
    method_comparisons: List[MethodComparison]
    recommendation: TrainingRecommendation
    tradeoff_analysis: Dict[str, Any]
    critical_issues: List[Dict[str, Any]]
    action_items: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Appended AFTER metadata so no existing positional construction moves.
    failed_methods: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary.

        THREE STATES, NEVER TWO. The two constraint lines below used to read
        ``.get('fairness_violation', 0):.4f`` and ``.get('constraint_satisfied',
        False)``, so a baseline whose constraint could not be evaluated printed
        "Fairness Violation: 0.0000 | Constraint Satisfied: False": a perfect
        score beside a failed verdict, neither of them measured. Once the
        producer started reporting NaN the first line printed "nan" instead,
        which is not a number a reader can act on either, and the second still
        printed False. A constraint nobody could evaluate now says so in words.

        The ACCURACY line one line above them was left behind by that fix and
        still read ``.get('accuracy', 0):.4f``. Measured 2026-09-27 on
        ``full_analysis(base_estimator=None)``, whose ``baseline_metrics`` is
        ``{}`` because no baseline was ever evaluated: the report printed

            BASELINE PERFORMANCE
            Accuracy: 0.0000
            Fairness Violation: not measured (the constraint could not be evaluated)
            Constraint Satisfied: not assessed (the constraint could not be evaluated)

        A model that got every row wrong, invented on the line directly above two
        lines that refuse to invent anything. The SVG fallback canvas for this
        same report already refuses it (test_adapters_training_empty.py::
        test_the_fallback_canvas_does_not_print_an_absent_baseline_as_zero); the
        text summary was the surface still doing it. The same rule covers the NaN
        that evaluate_baseline now reports for an undefined regression R^2.
        """
        # A measured 0.0 violation is a real result and still prints as 0.0000;
        # only None, the not-evaluated state, is replaced by prose.
        baseline_violation = self.baseline_metrics.get("fairness_violation")
        baseline_satisfied = self.baseline_metrics.get("constraint_satisfied")
        baseline_accuracy = self.baseline_metrics.get("accuracy")
        violation_line = (
            f"{baseline_violation:.4f}" if _measured(baseline_violation) else _NOT_EVALUATED
        )
        satisfied_line = (
            f"{baseline_satisfied}" if baseline_satisfied is not None else _NOT_ASSESSED
        )
        # The REASON when the producer recorded one, because "no baseline was
        # scored" is the wrong sentence for a baseline that WAS scored and whose
        # R^2 is undefined, or whose rows could not be read. Both now travel in
        # baseline_metrics (BGL4 audit, 2026-09-27); an empty reason means no
        # baseline was scored at all, which is the older case this phrase was
        # written for.
        accuracy_reason = str(self.baseline_metrics.get("accuracy_not_measured_reason") or "")
        accuracy_line = (
            f"{baseline_accuracy:.4f}"
            if _measured(baseline_accuracy)
            else (
                f"not measured ({accuracy_reason})"
                if accuracy_reason
                else "not measured (no baseline was scored)"
            )
        )
        lines = [
            "=" * 70,
            "FAIRNESS TRAINING ANALYSIS REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Task Type: {self.task_type}",
            f"Samples: {self.data_info.get('n_samples', 'N/A')}",
            f"Groups: {self.data_info.get('n_groups', 'N/A')}",
            "",
            "-" * 70,
            "BASELINE PERFORMANCE",
            "-" * 70,
            f"Accuracy: {accuracy_line}",
            f"Fairness Violation: {violation_line}",
            f"Constraint Satisfied: {satisfied_line}",
            "",
            "-" * 70,
            "FAIRNESS ANALYSIS",
            "-" * 70,
        ]

        # Fairness analysis
        if self.fairness_analysis:
            for metric, value in self.fairness_analysis.items():
                if isinstance(value, bool):
                    # A bool is not a magnitude: True would print as 1.0000.
                    lines.append(f"  {metric}: {value}")
                elif _measured(value):
                    lines.append(f"  {metric}: {value:.4f}")
                elif value is None:
                    # None here is the producer's not-measured state (see
                    # _compute_fairness_analysis on base_rate_disparity). Bare
                    # "None" beside four-decimal numbers reads as a gap in the
                    # data rather than as a refusal to measure.
                    lines.append(f"  {metric}: {_NOT_MEASURED}")
                else:
                    lines.append(f"  {metric}: {value}")
        lines.append("")

        # Method comparisons
        if self.method_comparisons:
            lines.extend(
                [
                    "-" * 70,
                    "METHOD COMPARISON",
                    "-" * 70,
                ]
            )
            for comp in self.method_comparisons:
                # Three glyphs, never two. "✗" is the MEASURED breach; a method
                # whose constraint was never evaluated gets "?" and the words to
                # go with it, the same shape ThresholdResult.summary already
                # uses. Reading `constraint_satisfied` alone put None on the "✗"
                # arm and reported an invented failure.
                if comp.constraint_satisfied is None:
                    satisfied_str = "?"
                elif comp.constraint_satisfied:
                    satisfied_str = "✓"
                else:
                    satisfied_str = "✗"
                comp_violation = (
                    f"{comp.fairness_violation:.4f}"
                    if _measured(comp.fairness_violation)
                    else _NOT_EVALUATED
                )
                # G10, 2026-09-30. THE SAME FIX AS THE BASELINE BLOCK, ON THE HALF
                # OF THE LINE IT NEVER REACHED. `Acc={comp.accuracy:.4f}` was
                # unconditional, and `compare_methods` DELIBERATELY sets accuracy
                # to NaN when it could not be measured, recording the reason in
                # parameters['accuracy_not_measured_reason'] beside
                # parameters['accuracy_measured']=False. Measured before this:
                #
                #   Exponentiated_Gradient: Acc=nan, Violation=not measured (the
                #   constraint could not be evaluated) [?]
                #
                # a minted value and a correct refusal on ONE line, which is the
                # defect this method's own docstring records for the baseline
                # ACCURACY line and which was fixed there and not here. The
                # recorded reason was printed nowhere in the whole report, so a
                # reader could not tell "the method reported a non-finite
                # accuracy" from "12 of 240 rows could not be scored".
                comp_reason = str(comp.parameters.get("accuracy_not_measured_reason") or "")
                comp_accuracy = (
                    f"{comp.accuracy:.4f}"
                    if _measured(comp.accuracy)
                    else "not measured (no finite accuracy was reported)"
                )
                lines.append(
                    f"  {comp.method_name}: Acc={comp_accuracy}, "
                    f"Violation={comp_violation} [{satisfied_str}]"
                )
                if not _measured(comp.accuracy) and comp_reason:
                    lines.append(f"      accuracy not measured: {comp_reason}")
            ungraded = sum(1 for c in self.method_comparisons if c.constraint_satisfied is None)
            if ungraded:
                # The count goes in the section a reader who stops here sees. A
                # list of methods reads as a list of EVALUATED methods otherwise.
                lines.append(
                    f"  ({ungraded} of {len(self.method_comparisons)} method(s) reported no "
                    "constraint result, so they are neither cleared nor in breach)"
                )
            lines.append("")

        # A TRAINING THAT CRASHED IS NOT A METHOD THAT WAS NOT ASKED FOR (BGL4
        # audit, 2026-09-27). Printed whenever the list is non-empty, and NOT
        # only when every method failed: with 2 of 3 crashing, this whole report
        # (summary plus to_dict) carried the word "fail" nowhere, showed one
        # comparison, no critical issue, and a recommendation naming a winner at
        # priority high, which is indistinguishable from a run where only one
        # method was requested and nothing went wrong. Measured before: the
        # summary of that run and the summary of a clean single-method run had
        # the same sections. After: the block below, plus the "Method(s) Failed
        # To Train" critical issue. The sibling ReweightingAnalysisReport.summary
        # has had exactly this block since the reweighting wave.
        if self.failed_methods:
            lines.extend(
                [
                    "-" * 70,
                    "METHODS THAT COULD NOT BE COMPARED",
                    "-" * 70,
                    f"  {len(self.failed_methods)} method(s) failed to train and are absent "
                    "from the comparison above; they were neither cleared nor found in "
                    "breach:",
                ]
            )
            for failure in self.failed_methods:
                lines.append(
                    f"  - {failure.get('method', 'unknown')}: "
                    f"{failure.get('error', 'no reason recorded')}"
                )
            lines.append("")

        # Recommendation
        lines.extend(
            [
                "-" * 70,
                "RECOMMENDATION",
                "-" * 70,
                f"Method: {self.recommendation.recommended_method}",
                f"Priority: {self.recommendation.priority}",
                f"Rationale: {self.recommendation.rationale}",
            ]
        )
        if self.recommendation.alternative_methods:
            lines.append(f"Alternatives: {', '.join(self.recommendation.alternative_methods)}")
        if self.recommendation.implementation_notes:
            lines.append(f"Notes: {self.recommendation.implementation_notes}")
        lines.append("")

        # Critical issues
        if self.critical_issues:
            lines.extend(
                [
                    "-" * 70,
                    "CRITICAL ISSUES",
                    "-" * 70,
                ]
            )
            for i, issue in enumerate(self.critical_issues[:5], 1):
                lines.append(f"  {i}. [{issue['type']}] {issue['description']}")
            lines.append("")

        # Action items
        if self.action_items:
            lines.extend(
                [
                    "-" * 70,
                    "ACTION ITEMS",
                    "-" * 70,
                ]
            )
            for i, item in enumerate(self.action_items[:5], 1):
                lines.append(f"  {i}. {item}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "data_info": self.data_info,
            "task_type": self.task_type,
            "baseline_metrics": self.baseline_metrics,
            "fairness_analysis": self.fairness_analysis,
            "method_comparisons": [m.to_dict() for m in self.method_comparisons],
            "recommendation": self.recommendation.to_dict(),
            "tradeoff_analysis": self.tradeoff_analysis,
            "critical_issues": self.critical_issues,
            "action_items": self.action_items,
            "metadata": self.metadata,
            # Always present, empty list when nothing failed: a consumer reading
            # the JSON must not have to tell a missing key from a clean run.
            "failed_methods": self.failed_methods,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_svg(self, save_path: Optional[str] = None) -> str:
        """
        Render this report as an SVG dashboard.

        Args:
            save_path: Optional path to save the SVG

        Returns:
            SVG markup string
        """
        try:
            from vfairness.rendering import training_report_to_svg

            return training_report_to_svg(self, save_path=save_path)
        except ImportError:
            warnings.warn("SVG rendering not available. Install jinja2.")
            return ""


class FairnessTrainingAnalyzer:
    """
    Unified Fairness Training Analyzer.

    Provides comprehensive analysis of fairness-aware training options
    and generates recommendations for achieving fairness constraints.

    The analyzer can:
        1. Evaluate baseline model fairness
        2. Compare different training methods
        3. Analyze accuracy-fairness trade-offs
        4. Generate recommendations
        5. Produce comprehensive reports

    Args:
        X: Feature matrix
        y: Target variable
        sensitive_attr: Sensitive attribute
        task_type: 'classification' or 'regression'
        fairness_constraint: Constraint type to enforce
        tolerance: Maximum allowed constraint violation
        attribute_name: Name of the sensitive attribute
        config: Optional configuration dictionary

    Example:
        >>> analyzer = FairnessTrainingAnalyzer(
        ...     X=X_train, y=y_train, sensitive_attr=gender,
        ...     fairness_constraint='demographic_parity',
        ...     tolerance=0.05
        ... )
        >>>
        >>> # Quick analysis
        >>> baseline = analyzer.evaluate_baseline(model)
        >>>
        >>> # Full analysis with comparisons
        >>> report = analyzer.full_analysis(base_estimator=LogisticRegression())
        >>> print(report.summary())

    References:
        - Agarwal et al. (2018): A Reductions Approach
        - Hardt et al. (2016): Equality of Opportunity

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

    Ledger row: fairness_training_analysis. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sensitive_attr: ArrayLike,
        *,
        task_type: Literal["classification", "regression"] = "classification",
        fairness_constraint: Union[str, FairnessConstraintType] = "demographic_parity",
        tolerance: float = 0.05,
        attribute_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.X = coerce_to_array(X, "X")
        self.y = coerce_to_array(y, "y")
        self.sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")
        check_consistent_length(self.X, self.y, self.sensitive_attr)

        self.task_type = task_type
        self.tolerance = tolerance
        self.attribute_name = attribute_name or "sensitive_attribute"
        self.config = config or {}

        self.constraint = create_constraint(fairness_constraint, tolerance=tolerance)

        self.group_manager = GroupManager(self.sensitive_attr)

        self._baseline_result: Optional[MethodComparison] = None
        self._method_comparisons: List[MethodComparison] = []
        # Methods whose fit raised inside compare_methods(). Public, because the
        # list compare_methods returns cannot carry them and a report built from
        # it would otherwise look like a report where nothing went wrong.
        self.method_failures: List[Dict[str, str]] = []

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.y)

    @property
    def n_groups(self) -> int:
        """Number of demographic groups."""
        return self.group_manager.n_groups

    @property
    def groups(self) -> List[str]:
        """List of group names."""
        return self.group_manager.groups

    def evaluate_baseline(
        self,
        model: Any = None,
        y_pred: Optional[ArrayLike] = None,
    ) -> MethodComparison:
        """
        Evaluate baseline model fairness.

        Args:
            model: Fitted model (or provide y_pred)
            y_pred: Pre-computed predictions

        Returns:
            MethodComparison with baseline metrics. ``accuracy`` is NaN, with
            ``parameters['accuracy_measured'] = False`` and
            ``parameters['accuracy_not_measured_reason']`` beside it, whenever it
            could not be scored: any row whose target or prediction is
            non-finite (classification and regression alike), a test set with no
            rows, or a regression target with no variance, for which
            R^2 = 1 - MSE/Var(y) is undefined. ``full_analysis`` carries that
            third state into ``baseline_metrics`` as ``accuracy: None``, which is
            the sentinel the rendering layer reads.
        """
        if y_pred is None:
            if model is None:
                raise ValueError("Either model or y_pred must be provided")
            y_pred = model.predict(self.X)

        y_pred = coerce_to_array(y_pred, "y_pred")

        # A BROADCAST IS NOT AN ACCURACY (BGL4 audit wave 4, 2026-09-30).
        # ABOVE THE DISPATCH, because both branches below read y_pred against
        # self.y elementwise and BOTH broadcast: `np.mean(y_pred == self.y)` and
        # `np.mean((y_pred - self.y) ** 2)` with a (60, 1) prediction against a
        # (60,) target each expand to a 60x60 matrix. For the classification arm
        # the mean of that matrix is the chance rate of the class mix, not an
        # accuracy. Measured 2026-09-30 on a PERFECT prediction, y = 30 zeros
        # then 30 ones:
        #
        #   y_pred flat (60,) -> accuracy 1.0, accuracy_measured True, silent
        #   y_pred as (60, 1) -> accuracy 0.5, accuracy_measured TRUE, silent
        #
        # coerce_to_array returns an ndarray unchanged, so it does not flatten.
        # Flattening is not a coercion of somebody's data, it is the row-wise
        # comparison the docstring already promises; a row COUNT that does not
        # line up cannot be reconciled and is the could-not-check instead. This
        # is the fix already carried by loss_functions/adversarial.py:808-850 and
        # by baseline_comparison_summary in this module.
        # A row COUNT that does not line up cannot be reconciled by reshaping and
        # nothing in this method can be computed from it: the accuracy has no
        # row-wise comparison and the constraint's group masks do not fit either,
        # which used to escape as a bare `IndexError: boolean index did not match
        # indexed array` out of constraints/base.py:775. A crash is not one of the
        # three states, so the refusal is named here instead.
        y_pred = np.asarray(y_pred).reshape(-1)
        y_true_flat = np.asarray(self.y).reshape(-1)
        if y_pred.shape[0] != y_true_flat.shape[0]:
            raise ValueError(
                "evaluate_baseline: y_pred carries {0} prediction(s) for {1} target "
                "row(s), so they cannot be compared row by row. Neither the accuracy nor "
                "the fairness constraint is computable on this pair; pass one prediction "
                "per row of y.".format(y_pred.shape[0], y_true_flat.shape[0])
            )

        # NOTHING TO SCORE IS NOT A SCORE (BGL4 audit, 2026-09-27). The
        # zero-variance guard below is keyed on np.ptp and np.var of the target,
        # and BOTH are NaN when the target carries a single NaN, so the refusal
        # could not fire on the second unmeasurable input; the classification
        # branch, meanwhile, wrote accuracy_measured = True as a literal and
        # never revisited it, so the flag could not be False on that path at all.
        # Measured on 240 rows:
        #
        #   regression, y = np.arange(240) with y[3] = nan, y_pred = np.arange(240)
        #     before -> accuracy nan, parameters['accuracy_measured'] True, NO warning
        #     after  -> accuracy nan, accuracy_measured False, and the warning
        #   classification, y all NaN, y_pred all NaN
        #     before -> accuracy 0.0, accuracy_measured True, NO warning
        #               (0.0 is a model that got every one of 240 rows wrong,
        #                invented from `nan == nan` being False on every row)
        #     after  -> accuracy nan, accuracy_measured False, and the warning
        #
        # Counted BEFORE either branch because it is the same defect on both, and
        # because it is what makes the zero-variance guard reachable: that guard
        # can now assume a target it is able to read. Healthy input is untouched:
        # the count is 0, so classification still measures 0.8667 on the 240-row
        # two-group fixture and regression still measures 0.9938 on y = 2x + noise.
        n_unscorable = _unscorable_rows(self.y, y_pred)
        accuracy_measured = True
        accuracy_not_measured_reason = ""
        if n_unscorable or self.n_samples == 0:
            accuracy_measured = False
            accuracy = float("nan")
            accuracy_not_measured_reason = (
                f"{n_unscorable} of {self.n_samples} row(s) carry a non-finite target or "
                "prediction, so they cannot be scored"
                if n_unscorable
                else "the test set has no rows to score"
            )
            warnings.warn(
                f"evaluate_baseline: accuracy NOT MEASURED ({accuracy_not_measured_reason}). "
                "Reporting accuracy=NaN (could not check) with "
                "parameters['accuracy_measured']=False, not the 0.0 that "
                "`y_pred == y` produces for a non-finite row (a model that got the row "
                "wrong) nor an R^2 whose numerator and denominator are both NaN.",
                UserWarning,
                stacklevel=2,
            )
        elif self.task_type == "classification":
            accuracy = np.mean(y_pred == self.y)
        else:
            # R^2 = 1 - MSE / Var(y) is UNDEFINED for a target with no variance:
            # the denominator is the quantity it is measured against. This line
            # divided by it anyway and passed the result on as a score. Measured
            # 2026-09-27 on 240 rows of a constant target: y_pred equal to y gave
            # accuracy nan (0/0) and a shifted y_pred gave -inf, the worst
            # possible model, both from a single numpy RuntimeWarning
            # ("invalid value encountered in scalar divide") that names neither
            # R^2 nor the target, and summary() printed "Accuracy: -inf". A model
            # that reproduces a constant target exactly is not infinitely bad;
            # the ratio just does not exist.
            y_variance = float(np.var(self.y))
            # np.ptp and not `var == 0`: var is an ACCUMULATED float, exactly 0.0
            # only for some constants at some lengths. Measured:
            # np.var(np.full(240, 0.1)) is 1.925929944387236e-34, NOT 0.0, so a
            # variance test would have refused np.full(240, 3.0) and answered
            # R^2 = -2.1e+34 for np.full(240, 0.1). max minus min of identical
            # floats is exactly 0.0 at any magnitude and any length.
            # THE FINITE SIBLING OF THE ZERO-VARIANCE GUARD (BGL4 audit wave 4,
            # 2026-09-30). The two tests above catch a target that is EXACTLY
            # constant. One ulp away the same absurd number is back, because they
            # are absolute tests of a quantity whose meaning is relative: the
            # spread has to be real variation in the target, not what is left of
            # the mantissa after cancellation. `y - mean(y)` carries an absolute
            # error of about eps * max|y| per row, so Var(y) is a measurement of
            # the data only while ptp(y) stands well clear of that: below roughly
            # sqrt(eps) of the target's own magnitude, half the significant
            # digits of every deviation are gone and Var is numerical noise in
            # the denominator of R^2.
            #
            # Measured 2026-09-30, 60 rows, y_pred = zeros:
            #
            #   y = np.full(60, 0.1) exactly       -> accuracy nan, measured False,
            #                                         1 warning  (the guard above)
            #   y = np.full(60, 0.1), y[0] += 1e-15 -> accuracy -5.782e+29,
            #                                         accuracy_measured TRUE, 0 warnings
            #   y = np.full(60, 3.0), y[0] += 1e-12 -> accuracy -5.491e+26,
            #                                         accuracy_measured TRUE, 0 warnings
            #
            # The final derived-flag guard below only normalises values that fail
            # _measured, i.e. non-finite ones, so a finite -5.8e29 sailed past it
            # as a measured R^2. The comment above names R^2 = -2.1e+34 as the
            # number a variance test would have let through; this is that number,
            # reached through the np.ptp test instead. A relative test is the one
            # the audit brief asks for: never an exact equality with zero on a
            # computed statistic, use a data-scaled tolerance.
            #
            # NOT an over-correction: a genuinely tight regression target is
            # untouched. y in [100.0, 100.001] has a relative spread of 1e-5,
            # four orders above the floor, and y = 2x + noise measures
            # 0.9973437558470829 exactly as before.
            y_ptp = float(np.ptp(self.y))
            y_scale = float(np.max(np.abs(self.y))) if self.n_samples else 0.0
            relative_spread = y_ptp / y_scale if y_scale > 0.0 else y_ptp
            if y_ptp == 0.0 or y_variance <= 0.0:
                accuracy_measured = False
                accuracy = float("nan")
                accuracy_not_measured_reason = (
                    f"the regression target has zero variance (var={y_variance!r} over "
                    f"{self.n_samples} row(s)), so R^2 = 1 - MSE/Var(y) is undefined"
                )
                warnings.warn(
                    f"evaluate_baseline: the regression target has zero variance "
                    f"(var={y_variance!r} over {self.n_samples} row(s)), so R^2 = "
                    "1 - MSE/Var(y) is undefined: there is no variation for the model to "
                    "explain. Reporting accuracy=NaN (could not check) with "
                    "parameters['accuracy_measured']=False, not the -inf or NaN this "
                    "division used to hand on as a score.",
                    UserWarning,
                    stacklevel=2,
                )
            elif relative_spread < _DEGENERATE_RELATIVE_SPREAD:
                accuracy_measured = False
                accuracy = float("nan")
                accuracy_not_measured_reason = (
                    "the regression target varies by {0!r} over a magnitude of {1!r}, a "
                    "relative spread of {2!r}, which is below the floating-point "
                    "resolution of its own scale ({3!r}): Var(y)={4!r} is then numerical "
                    "noise rather than variation, and R^2 = 1 - MSE/Var(y) divides by "
                    "it".format(
                        y_ptp,
                        y_scale,
                        relative_spread,
                        _DEGENERATE_RELATIVE_SPREAD,
                        y_variance,
                    )
                )
                warnings.warn(
                    "evaluate_baseline: accuracy NOT MEASURED ("
                    + accuracy_not_measured_reason
                    + "). Reporting accuracy=NaN (could not check) with "
                    "parameters['accuracy_measured']=False, not the large FINITE negative "
                    "R^2 this division produces (measured -5.78e+29 for a target constant "
                    "to within 1e-15), which the non-finite check below cannot catch and "
                    "which every consumer prints as a measured score.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                accuracy = 1 - np.mean((y_pred - self.y) ** 2) / y_variance

        # THE FLAG IS DERIVED FROM THE VALUE, NEVER ASSERTED BESIDE IT (BGL4
        # audit, 2026-09-27). `accuracy_measured = True` was a literal on the
        # classification path, which is a flag that CANNOT be False whatever the
        # arithmetic did. This last step closes the class rather than the one
        # input: if any branch above still produced a non-finite score (an R^2
        # divided by a variance small enough to pass the np.ptp guard and
        # overflow), the flag says so and the value is normalised to the NaN this
        # module spells its third state with, instead of an inf that compares
        # greater than every threshold. Healthy input is untouched, measured:
        # classification 0.9041666666666667 and regression 0.9795765608867427,
        # both accuracy_measured True, both silent.
        if accuracy_measured and not _measured(accuracy):
            accuracy_not_measured_reason = (
                f"the score computed as {accuracy!r}, which is not a finite measurement"
            )
            warnings.warn(
                f"evaluate_baseline: accuracy NOT MEASURED ({accuracy_not_measured_reason}). "
                "Reporting accuracy=NaN (could not check) with "
                "parameters['accuracy_measured']=False.",
                UserWarning,
                stacklevel=2,
            )
            accuracy_measured = False
            accuracy = float("nan")

        violation = self.constraint.compute_violation(y_pred, self.y, self.sensitive_attr)
        # R4C. The pair is translated once, here, rather than copied raw: a
        # constraint that could not be evaluated must not leave this method
        # wearing a measured verdict. See _reported_constraint.
        reported_violation, reported_satisfied = _reported_constraint(violation)

        result = MethodComparison(
            method_name="Baseline (Unconstrained)",
            accuracy=accuracy,
            fairness_violation=reported_violation,
            constraint_satisfied=reported_satisfied,
            parameters={
                "constraint_type": violation.constraint_type,
                # Named on the object itself, so a consumer holding only this
                # comparison can tell the third state from a measured breach
                # without re-deriving it from two None values.
                "constraint_evaluated": not violation.could_not_evaluate,
                # The same courtesy for the accuracy field, which is typed as a
                # plain float and so cannot carry None.
                "accuracy_measured": accuracy_measured,
                # The WHY, in the same dict as the flag, so a consumer that
                # withholds the number has the sentence to print in its place.
                # Empty string when the accuracy was measured.
                "accuracy_not_measured_reason": accuracy_not_measured_reason,
            },
        )

        self._baseline_result = result
        return result

    def compare_methods(
        self,
        base_estimator: Any,
        methods: Optional[List[str]] = None,
    ) -> List[MethodComparison]:
        """
        Compare different fairness training methods.

        Args:
            base_estimator: Base estimator to use
            methods: List of methods to compare

        Returns:
            List of MethodComparison results, one per method that TRAINED. A
            method whose fit raised is left out and recorded in
            :attr:`method_failures`, on ``FairnessTrainingReport.failed_methods``
            and in the report's critical issues, summary block and action items,
            whether SOME or ALL of the requested methods failed, and a summary
            warning naming the count is raised in both cases. When every
            requested method failed, that disclosure accompanies an EMPTY list,
            because an empty list otherwise says exactly what ``methods=[]``
            says. Measured 2026-09-27 with a base estimator whose fit raises:
            ``compare_methods`` returned ``[]`` with one warning per method,
            ``generate_recommendation`` then reported
            "No fairness methods evaluated", and ``_identify_critical_issues``
            raised NOTHING, because its unevaluated-constraint branch is guarded
            by ``if method_comparisons``. Three methods that trained and could
            not be graded produced a critical issue; three methods that could not
            train at all, the worse outcome, produced a quieter report.
            Measured again 2026-09-27 (BGL4 audit) with 2 of 3 methods crashing:
            every one of those disclosures was gated on the TOTAL failure, so the
            partial one reached no surface of the report at all and its issue list
            was identical to a clean run's. See the comment on the warning below.

            Each comparison's ``parameters`` also carries the DEGENERACY of a
            method that succeeded (BGL4 wave 4): the partial-failure disclosure
            above says nothing about a method that trained happily and collapsed
            to a constant prediction, which satisfies every rate-based constraint
            by construction. ``degenerate_constant_predictions`` is True for such
            a method, False when it made a real decision and None when its
            predictions could not be re-read; ``n_prediction_classes`` is the
            count it was derived from. ``accuracy_measured`` and
            ``accuracy_not_measured_reason`` are beside them, the same three-state
            treatment :meth:`evaluate_baseline` already gives the sibling arm, so
            both arms put the same keys on the same objects into the same report.
            ``generate_recommendation`` excludes a collapsed method from both of
            its rankings.

        Raises:
            ValueError: If task_type is 'regression'. The comparison pipeline
                (FairClassifier, classification accuracy, rate-based
                constraints) is classification-only, so running it on
                regression data would silently produce meaningless numbers.
        """
        if self.task_type == "regression":
            raise ValueError(
                "compare_methods() supports classification only: it fits "
                "FairClassifier variants and scores them with classification "
                "accuracy and rate-based fairness constraints. For regression "
                "use evaluate_baseline() (which uses R^2) or the FairRegressor "
                "wrapper directly."
            )

        from .wrappers import FairClassifier

        if methods is None:
            methods = ["reductions", "threshold", "grid_search"]

        comparisons = []
        failures: List[Dict[str, str]] = []

        for method in methods:
            try:
                clf = FairClassifier(
                    base_estimator=base_estimator,
                    fairness_constraint=self.constraint,
                    tolerance=self.tolerance,
                    # `methods` is a permissive List[str]; an unknown method name
                    # raises inside FairClassifier and is caught below.
                    method=cast(Literal["reductions", "threshold", "grid_search"], method),
                    verbose=False,
                )

                import time

                start_time = time.time()
                clf.fit(self.X, self.y, sensitive_attr=self.sensitive_attr)
                training_time = time.time() - start_time

                # fit() sets fairness_result_ before returning.
                assert clf.fairness_result_ is not None
                # FairClassifierResult carries the same three states (None for a
                # constraint it could not evaluate), so these two fields are
                # passed through unchanged rather than re-derived.

                # THE DEGENERACY OF A METHOD THAT SUCCEEDED (BGL4 audit wave 4,
                # 2026-09-30). The partial-FAILURE disclosure below is real, and
                # the collapse of the methods that DID train was dropped at this
                # boundary: parameters carried only "method" and
                # "constraint_evaluated", so a comparison in which every
                # mitigation collapsed to a constant classifier rendered as a
                # complete clean comparison. A constant classifier satisfies
                # every rate-based constraint by construction, because approving
                # (or refusing) everybody is exactly equal treatment.
                #
                # Measured 2026-09-30, 240 rows, group base rates split by
                # s == "a", base_estimator = DummyClassifier("most_frequent"):
                #
                #   Reductions   acc=0.6917 viol=0.0 satisfied=True
                #   Threshold    acc=0.6917 viol=0.0 satisfied=True
                #   Grid_Search  acc=0.6917 viol=0.0 satisfied=True
                #
                # All three accuracies byte-identical to the majority-class rate,
                # i.e. all three "mitigations" made no decision and are
                # trivially perfectly fair. One warning in the whole call, from
                # inside ExponentiatedGradient, for the reductions arm only;
                # threshold and grid_search were silent. generate_recommendation
                # then returned "Reductions" at priority HIGH with "Achieves
                # fairness constraint with best accuracy (0.692)".
                #
                # COMPUTED HERE, from the model's own output, rather than read off
                # a flag: ExponentiatedGradient.fit records
                # fairness_metrics["degenerate_constant_predictions"], but
                # FairClassifier._fit_reductions builds a fresh metrics dict that
                # does not forward it (sklearn_wrappers.py:285-293), so the flag
                # does not reach this method on any arm. Counting the distinct
                # labels is the check the auditor's own note says a caller is
                # left to perform, and it works for all three methods rather than
                # only the one that happens to record it. The flag IS read when
                # present, so it starts travelling the moment the wrapper
                # forwards it.
                #
                # predict_with_sensitive_attr AND NOT predict, because the two are
                # different quantities and only one of them is the quantity the
                # result was computed from. FairClassifier.predict for
                # method='threshold' says so in its own docstring: "the fitted
                # group thresholds cannot be applied without the sensitive
                # attribute, so this returns the uniform 0.5 answer and warns",
                # whereas _fit_threshold scored
                # _threshold_optimizer.predict(y_prob, sensitive_attr).
                # Measured 2026-09-30 with a LogisticRegression base estimator:
                # clf.predict(X) on the threshold arm gave ONE distinct label
                # while that arm's own violation was 0.0083, i.e. its recorded
                # predictions were not constant at all. Judging an arm by an
                # output it does not consume is over-accusation, so the check
                # reads the mitigated prediction (which for the other two arms is
                # the same self._inner_model.predict(X) they were scored on).
                metrics = clf.fairness_result_.fairness_metrics or {}
                forwarded = metrics.get("degenerate_constant_predictions")
                n_pred_classes: Optional[int] = None
                degenerate: Optional[bool] = None
                n_unscorable_method: Optional[int] = None
                try:
                    method_preds = np.asarray(
                        clf.predict_with_sensitive_attr(self.X, self.sensitive_attr)
                    ).reshape(-1)
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                    # None, not False: "nobody could look" is not "it did not
                    # collapse", and generate_recommendation tests the VALUE.
                    logger.debug(
                        "compare_methods: %s predictions could not be re-read to count "
                        "distinct labels (%s: %s), so degenerate_constant_predictions "
                        "stays %r (could not check)",
                        method,
                        type(exc).__name__,
                        exc,
                        forwarded,
                    )
                    degenerate = forwarded if isinstance(forwarded, bool) else None
                else:
                    if method_preds.size:
                        n_pred_classes = int(len(np.unique(method_preds)))
                        degenerate = n_pred_classes < 2
                    if forwarded is True:
                        degenerate = True
                    if method_preds.shape[0] == len(np.asarray(self.y).reshape(-1)):
                        n_unscorable_method = _unscorable_rows(self.y, method_preds)

                # THE SAME THREE-STATE TREATMENT AS THE SIBLING ARM. FairClassifier
                # computes accuracy as float(np.mean(y_pred == y)) with no
                # unscorable-row guard (sklearn_wrappers.py:273/328/366), and this
                # method passed it straight into MethodComparison with no
                # accuracy_measured key, while evaluate_baseline right above was
                # given exactly that treatment. Both arms feed the same
                # MethodComparison objects into the same report and the same
                # renderers, and generate_recommendation ranks over this field.
                # NOTHING TO SCORE IS NOT A SCORE, on this arm too. Measured
                # 2026-09-30 on 120 rows, 10 of them with a NaN label,
                # base_estimator=DummyClassifier("stratified"): the threshold arm
                # trained and reported accuracy 0.4583333333333333 with no
                # accuracy_measured key at all. `nan == anything` is False, so
                # those 10 rows entered the mean as rows the model got WRONG:
                # that number is part measurement and part fabrication, and it is
                # the field generate_recommendation ranks over. The two other arms
                # raised and were recorded as failures, so only ONE arm carried
                # the fabrication and the partial-failure disclosure said nothing
                # about it.
                method_accuracy = clf.fairness_result_.accuracy
                acc_measured = _measured(method_accuracy)
                acc_reason = ""
                if not acc_measured:
                    acc_reason = (
                        "the method reported accuracy {0!r}, which is not a finite "
                        "measurement".format(method_accuracy)
                    )
                elif n_unscorable_method:
                    acc_measured = False
                    acc_reason = (
                        "{0} of {1} row(s) carry a non-finite target or prediction, so the "
                        "accuracy {2!r} this method reported counted them as rows it got "
                        "wrong rather than rows it could not be scored on".format(
                            n_unscorable_method, self.n_samples, method_accuracy
                        )
                    )
                if not acc_measured:
                    method_accuracy = float("nan")

                result = MethodComparison(
                    method_name=method.title(),
                    accuracy=method_accuracy,
                    fairness_violation=clf.fairness_result_.fairness_violation,
                    constraint_satisfied=clf.fairness_result_.constraint_satisfied,
                    training_time=training_time,
                    parameters={
                        "method": method,
                        "constraint_evaluated": clf.fairness_result_.constraint_satisfied
                        is not None,
                        # True means constraint_satisfied is VACUOUS: read them
                        # together, the same way constraint_satisfied must be read
                        # with constraint_evaluated. None means nobody could look.
                        "degenerate_constant_predictions": degenerate,
                        "n_prediction_classes": n_pred_classes,
                        # The same courtesy the baseline arm already gets, because
                        # accuracy is typed as a plain float and cannot carry None.
                        "accuracy_measured": acc_measured,
                        "accuracy_not_measured_reason": acc_reason,
                    },
                )
                comparisons.append(result)
                if degenerate:
                    warnings.warn(
                        "compare_methods: method {0!r} collapsed to a CONSTANT prediction "
                        "({1} distinct label(s) over {2} row(s)), so its fairness numbers "
                        "are vacuous rather than good: a model that makes no decision "
                        "satisfies every rate-based constraint by construction, and its "
                        "accuracy is the majority-class rate. Read "
                        "parameters['degenerate_constant_predictions'] beside "
                        "constraint_satisfied; generate_recommendation excludes this "
                        "method from both rankings.".format(method, n_pred_classes, self.n_samples),
                        UserWarning,
                        stacklevel=2,
                    )
                if not acc_measured:
                    warnings.warn(
                        "compare_methods: method {0!r} reported an accuracy that is not a "
                        "measurement ({1}), so it is carried as NaN with "
                        "parameters['accuracy_measured']=False, not as a score. "
                        "generate_recommendation will not rank it on "
                        "accuracy.".format(method, acc_reason),
                        UserWarning,
                        stacklevel=2,
                    )

            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
                failures.append({"method": method, "error": f"{type(e).__name__}: {e}"})
                continue

        self._method_comparisons = comparisons
        self.method_failures = failures
        if failures:
            # GATED ON `failures`, NOT ON `not comparisons` (BGL4 audit,
            # 2026-09-27). The summary warning fired only for a TOTAL failure, so
            # a run in which 2 of 3 methods crashed returned a list of one and
            # said nothing about the count. Measured through full_analysis with
            # FairClassifier.fit raising for the threshold and grid_search
            # methods only: comparisons ['Reductions'], no summary warning, no
            # critical issue, report JSON carrying no failure record, and the
            # recommendation naming Reductions at priority high, identical to the
            # control run with nothing crashing. After: this warning names "2 of 3
            # requested method(s) failed to train", the report carries
            # failed_methods, and _identify_critical_issues raises "Method(s)
            # Failed To Train".
            #
            # The per-method "Method X failed: ..." warnings above are not a
            # substitute: they name one method each and never the count, and the
            # count is the only thing that says the returned list is incomplete.
            names = ", ".join(f["method"] for f in failures)
            if not comparisons:
                warnings.warn(
                    f"compare_methods: all {len(failures)} requested method(s) failed to "
                    f"train ({names}), so the empty list returned is 'nothing could be "
                    "compared', not 'no fairness problem was found'. The failures are on "
                    "FairnessTrainingAnalyzer.method_failures and in the report's critical "
                    "issues.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"compare_methods: {len(failures)} of {len(failures) + len(comparisons)} "
                    f"requested method(s) failed to train ({names}), so the "
                    f"{len(comparisons)} comparison(s) returned are NOT a complete "
                    "comparison: the failed method(s) were neither cleared nor found in "
                    "breach, and any 'best method' chosen from this list is the best of "
                    "the survivors only. The failures are on "
                    "FairnessTrainingAnalyzer.method_failures and in the report's "
                    "critical issues.",
                    UserWarning,
                    stacklevel=2,
                )
        return comparisons

    def analyze_tradeoffs(
        self,
        base_estimator: Any,
        lambda_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze accuracy-fairness trade-offs.

        Args:
            base_estimator: Base estimator
            lambda_values: Lambda values to try

        Returns:
            Dictionary with trade-off analysis

        Raises:
            ValueError: If task_type is 'regression'. The trade-off sweep
                scores each model with classification accuracy
                (mean(y_pred == y)), which is ~0 and meaningless on
                continuous targets.
        """
        if self.task_type == "regression":
            raise ValueError(
                "analyze_tradeoffs() supports classification only: it scores "
                "each candidate with classification accuracy and rate-based "
                "fairness constraints, both of which are meaningless for "
                "continuous targets. Use evaluate_baseline() for regression."
            )

        if lambda_values is None:
            lambda_values = np.linspace(0, 2, 11).tolist()

        from copy import deepcopy

        results = []

        for lam in lambda_values:
            weights = np.ones(self.n_samples)

            for group in self.groups:
                mask = self.group_manager.get_mask(group)
                weights[mask] *= 1 + lam * self.n_samples / mask.sum()

            weights = weights / weights.sum() * self.n_samples

            model = deepcopy(base_estimator)
            try:
                model.fit(self.X, self.y, sample_weight=weights)
            except TypeError:
                model.fit(self.X, self.y)

            y_pred = model.predict(self.X)
            accuracy = np.mean(y_pred == self.y)
            violation = self.constraint.compute_violation(y_pred, self.y, self.sensitive_attr)
            reported_violation, reported_satisfied = _reported_constraint(violation)

            results.append(
                {
                    "lambda": lam,
                    "accuracy": accuracy,
                    "violation": reported_violation,
                    "satisfied": reported_satisfied,
                }
            )

        # Find Pareto optimal points.
        #
        # R4C, the RANKING half. Only configurations whose violation was
        # MEASURED can be ordered on the fairness axis at all. This loop used to
        # run over every row with the raw NaN in place, and every comparison
        # against NaN is False, so an unevaluable configuration was never
        # dominated by anything: measured on an 11-point sweep whose third group
        # had no positive labels, all 11 rows came back as the "pareto_frontier"
        # while not one of them had a fairness coordinate. Being reported as
        # optimal is a stronger claim than being reported as failing.
        measured_results = [r for r in results if _measured(r["violation"])]
        pareto_points = []
        for r in measured_results:
            is_dominated = False
            for other in measured_results:
                if (
                    other["accuracy"] >= r["accuracy"]
                    and other["violation"] <= r["violation"]
                    and (other["accuracy"] > r["accuracy"] or other["violation"] < r["violation"])
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append(r)

        return {
            "all_results": results,
            "pareto_frontier": pareto_points,
            "best_fair": min(
                [r for r in measured_results if r["satisfied"]],
                key=lambda x: -x["accuracy"],
                default=None,
            ),
            "best_accurate": max(results, key=lambda x: x["accuracy"]),
            # The sweep's own could-not-check count. The frontier and best_fair
            # are drawn from the measured rows only, so a reader has to be told
            # how many rows were left out rather than inferring a full sweep
            # from a shorter list.
            "n_not_evaluated": len(results) - len(measured_results),
        }

    def generate_recommendation(
        self,
        comparisons: Optional[List[MethodComparison]] = None,
    ) -> TrainingRecommendation:
        """
        Generate recommendation based on the method comparisons.

        Args:
            comparisons: Method comparisons (default: those recorded by
                compare_methods())

        Returns:
            TrainingRecommendation

        A `tradeoff_analysis` argument used to be accepted here (F16,
        removed 2026-09-09). It was never read: the recommendation is chosen
        from the comparisons alone, and passing the full analyze_tradeoffs()
        result changed nothing. No intended use was recoverable from the
        code, so the parameter is gone rather than given an invented one.

        R4C, the RANKING half. ``min(comparisons, key=fairness_violation)`` was
        taken over every method including those whose constraint could not be
        evaluated. With NaN in that field the comparison never wins and never
        loses, so ``min`` simply returned the FIRST method, and the rationale
        read "Minimizes constraint violation (nan)": a recommendation presented
        as the outcome of a comparison that did not happen. Only methods with a
        measured violation are ranked now, and when none of them has one the
        recommendation says so instead of naming a winner.

        BGL4 wave 4, the ACCURACY half of that same pathology plus the DEGENERACY
        exclusion above the dispatch. ``max(satisfying, key=accuracy)`` three
        lines below the fix above was taken over a field that can be NaN, so the
        winner was decided by LIST ORDER and the rationale read "Achieves
        fairness constraint with best accuracy (nan)". Only methods with a
        measured accuracy are ranked on it now. Separately, a method whose
        predictions all carry the same label
        (``parameters['degenerate_constant_predictions']``) is excluded from BOTH
        branches: a constant classifier satisfies every rate-based constraint by
        construction, so its 0.0 violation and its majority-class accuracy win on
        either field, and "Achieves fairness constraint" is the one thing that
        must not be said about it.

        Warns:
            UserWarning: when a method is left out of the ranking, either for a
                collapsed constant prediction or for an accuracy that is not a
                measurement. The recommendation that follows is over the
                remaining methods only, and the counts are in the rationale.
        """
        if comparisons is None:
            comparisons = self._method_comparisons

        # ABOVE THE DISPATCH, NOT INSIDE ONE BRANCH (BGL4 audit wave 4,
        # 2026-09-30). A method whose predictions all carry the SAME label is
        # trivially perfectly fair because it made no decision: approving
        # everybody is exactly equal treatment. ExponentiatedGradient.fit records
        # that as fairness_metrics['degenerate_constant_predictions'] and
        # compare_methods now carries it onto MethodComparison.parameters, so it
        # is available here. Both branches below rank by a field a collapsed
        # method wins on (its violation is 0.0 and its accuracy is the
        # majority-class rate), so the exclusion belongs above the selection: a
        # filter inside the `satisfying` branch alone would have moved the
        # fabrication into the `ranked` branch.
        #
        # Measured 2026-09-30 on three methods all collapsed to a constant
        # classifier (accuracy 0.6917 = the majority rate, violation 0.0,
        # satisfied True): before -> "Reductions" at priority HIGH, rationale
        # "Achieves fairness constraint with best accuracy (0.692)", 0 warnings.
        # A trivially satisfied constraint was treated as an achieved one.
        def _collapsed(c: MethodComparison) -> bool:
            params = c.parameters if isinstance(c.parameters, dict) else {}
            # The VALUE, not the key: `.get(key, default)` does not fire when the
            # key is present holding None, and None here means "nobody looked",
            # which is not evidence that the method did not collapse.
            return params.get("degenerate_constant_predictions") is True

        degenerate = [c for c in comparisons if _collapsed(c)]
        gradable = [c for c in comparisons if not _collapsed(c)]
        degenerate_note = (
            " {0} of {1} method(s) collapsed to a constant prediction and were excluded: "
            "a model that makes no decision is trivially fair, not fair.".format(
                len(degenerate), len(comparisons)
            )
            if degenerate
            else ""
        )

        # `constraint_satisfied is True` and not truthiness: None is the
        # not-evaluated state, and it must not be counted as satisfying OR as
        # failing.
        satisfying = [c for c in gradable if c.constraint_satisfied is True]
        ranked = [c for c in gradable if _measured(c.fairness_violation)]
        ungraded = len(gradable) - len(ranked)
        ungraded_note = (
            f" {ungraded} of {len(comparisons)} method(s) reported no constraint result "
            "and were not ranked."
            if ungraded
            else ""
        )

        # THE SAME NaN PATHOLOGY, ONE FIELD OVER (BGL4 audit wave 4, 2026-09-30).
        # The R4C fix recorded in the docstring above filtered `ranked` by
        # _measured(fairness_violation) and left the `satisfying` list, three
        # lines away in the same function, ranked by `max(..., key=accuracy)`
        # over a field that can be NaN. Every comparison against NaN is False, so
        # max keeps whatever it saw FIRST: which method won was decided by LIST
        # ORDER. Measured 2026-09-30 on two satisfying methods, one with a NaN
        # accuracy:
        #
        #   [nan, 0.82] -> "Unmeasurable", priority high, 0 warnings, rationale
        #                  "Achieves fairness constraint with best accuracy (nan)",
        #                  expected_tradeoff {"accuracy": nan, ...}
        #   the SAME two swapped -> "Real", rationale "... (0.820)"
        #
        # min() had the identical shape and was fixed; max() is its sibling.
        acc_ranked = [c for c in satisfying if _measured(c.accuracy)]
        unscored_acc = len(satisfying) - len(acc_ranked)
        unscored_acc_note = (
            " {0} of {1} method(s) that satisfy the constraint reported no measurable "
            "accuracy and were not ranked on it.".format(unscored_acc, len(satisfying))
            if unscored_acc
            else ""
        )
        if degenerate or unscored_acc:
            warnings.warn(
                "generate_recommendation: "
                + "; ".join(
                    part
                    for part in (
                        (
                            "{0} of {1} method(s) collapsed to a constant prediction "
                            "({2}), which is trivially fair rather than fair, so they are "
                            "not ranked and not recommended".format(
                                len(degenerate),
                                len(comparisons),
                                ", ".join(c.method_name for c in degenerate),
                            )
                            if degenerate
                            else ""
                        ),
                        (
                            "{0} of {1} method(s) that satisfy the constraint have no "
                            "measurable accuracy, so they cannot be ranked by it (a NaN "
                            "neither wins nor loses a max(), which made the winner depend "
                            "on list order)".format(unscored_acc, len(satisfying))
                            if unscored_acc
                            else ""
                        ),
                    )
                    if part
                )
                + ". The recommendation below is over the remaining method(s) only.",
                UserWarning,
                stacklevel=2,
            )

        if acc_ranked:
            best = max(acc_ranked, key=lambda x: x.accuracy)
            return TrainingRecommendation(
                recommended_method=best.method_name,
                priority="high",
                rationale=f"Achieves fairness constraint with best accuracy ({best.accuracy:.3f})"
                + ungraded_note
                + unscored_acc_note
                + degenerate_note,
                # Only the accuracy-ranked methods are named as alternatives, for
                # the same reason the minimising branch below names only `ranked`:
                # listing a method whose accuracy was never measured beside them
                # reads as "compared and came second".
                alternative_methods=[c.method_name for c in acc_ranked if c != best],
                expected_tradeoff={
                    "accuracy": best.accuracy,
                    "fairness_violation": best.fairness_violation,
                },
                implementation_notes="Constraint is achievable with this method.",
            )
        elif satisfying:
            # Method(s) DID satisfy the constraint and not one of them has an
            # accuracy that can be ranked. Naming a winner here is the
            # fabrication the max() above produced by list order, and so is
            # printing "best accuracy (nan)". "N/A" is the sentinel the rendering
            # layer already reads as "no advice was given"
            # (adapters_training._recommendation_state). The achievability is
            # real and measured, so it is stated in the notes rather than
            # dropped.
            return TrainingRecommendation(
                recommended_method="N/A",
                priority="low",
                rationale=(
                    "{0} method(s) satisfy the fairness constraint, but not one of them "
                    "reported a measurable accuracy, so there is no best method to "
                    "recommend.".format(len(satisfying))
                )
                + degenerate_note,
                alternative_methods=[c.method_name for c in satisfying],
                expected_tradeoff={
                    "accuracy": float("nan"),
                    "fairness_violation": float("nan"),
                },
                implementation_notes=(
                    "The constraint IS achievable on this data ({0} method(s) met it), but "
                    "the accuracy of every one of them came back as a non-measurement, so "
                    "the accuracy-fairness trade-off cannot be compared. Score the "
                    "candidates on a test set whose labels are all present, then re-run "
                    "generate_recommendation().".format(len(satisfying))
                ),
            )
        elif ranked:
            # `ranked` is filtered by _measured, so every fairness_violation in
            # it is a real float; the cast states that to the type checker and
            # asserts nothing the filter above has not already established.
            best = min(ranked, key=lambda x: cast(float, x.fairness_violation))
            return TrainingRecommendation(
                recommended_method=best.method_name,
                priority="medium",
                rationale=f"Minimizes constraint violation ({best.fairness_violation:.3f})"
                + ungraded_note
                + degenerate_note,
                # Only the ranked methods are named as alternatives: listing a
                # method whose constraint was never evaluated beside them reads
                # as "compared and came second".
                alternative_methods=[c.method_name for c in ranked if c != best],
                expected_tradeoff={
                    "accuracy": best.accuracy,
                    "fairness_violation": best.fairness_violation,
                },
                implementation_notes="Full constraint satisfaction may not be achievable. "
                "Consider relaxing tolerance or using alternative criteria.",
            )
        elif degenerate and not gradable:
            # EVERY method collapsed to a constant prediction. Falling through to
            # the branch below would have said "not one reported a constraint
            # result", which is false: they all reported a satisfied one, and
            # that verdict is vacuous rather than missing. Two different states,
            # two different sentences.
            return TrainingRecommendation(
                recommended_method="N/A",
                priority="low",
                rationale=(
                    "All {0} method(s) collapsed to a constant prediction, so each one is "
                    "trivially perfectly fair because it made no decision at all. None was "
                    "graded and none is recommended.".format(len(degenerate))
                ),
                alternative_methods=[],
                expected_tradeoff={
                    "accuracy": float("nan"),
                    "fairness_violation": float("nan"),
                },
                implementation_notes=(
                    "A constant classifier satisfies every rate-based fairness constraint "
                    "by construction: approving (or refusing) everybody is exactly equal "
                    "treatment. The collapse is the finding, not the fairness number. Check "
                    "the base estimator and the constraint tolerance, then re-run "
                    "compare_methods()."
                ),
            )
        elif comparisons:
            # Methods were TRAINED, and not one of them reported a constraint
            # result. Naming a winner here is the fabrication: "N/A" is the
            # sentinel the rendering layer already reads as "no advice was
            # given" (adapters_training._recommendation_state), so the canvas
            # withholds the priority chip instead of drawing a verdict.
            return TrainingRecommendation(
                recommended_method="N/A",
                priority="low",
                rationale=(
                    f"Not one of the {len(gradable)} method(s) reported a constraint "
                    "result, so none was ranked, none was cleared and none is recommended."
                )
                + degenerate_note,
                alternative_methods=[],
                implementation_notes=(
                    "The fairness constraint could not be evaluated on this data (a group "
                    "whose rate has an empty denominator makes the disparity unmeasurable). "
                    "Collect labels for the affected group(s), or choose a constraint whose "
                    "rates are defined here, then re-run compare_methods()."
                ),
            )
        else:
            return TrainingRecommendation(
                recommended_method="Unconstrained",
                priority="low",
                rationale="No fairness methods evaluated",
                implementation_notes="Run compare_methods() to evaluate fairness options.",
            )

    def full_analysis(
        self,
        base_estimator: Any = None,
        include_comparisons: bool = True,
        include_tradeoffs: bool = True,
    ) -> FairnessTrainingReport:
        """
        Perform comprehensive fairness training analysis.

        Args:
            base_estimator: Base estimator for comparisons
            include_comparisons: Whether to compare methods
            include_tradeoffs: Whether to analyze trade-offs

        Returns:
            FairnessTrainingReport with comprehensive findings
        """
        baseline_metrics = {}
        if base_estimator is not None:
            from copy import deepcopy

            baseline_model = deepcopy(base_estimator)
            baseline_model.fit(self.X, self.y)
            baseline = self.evaluate_baseline(model=baseline_model)
            # A COPY THAT DROPS THE FLAG PUBLISHES THE NaN AS A NUMBER (BGL4
            # audit, 2026-09-27). This rebuilt the dict from three fields and
            # left parameters['accuracy_measured'] behind, and NaN is not a
            # third state any consumer here speaks: rendering's _baseline_state
            # derives baseline_measured from `accuracy is not None`. Measured on
            # a 240-row constant regression target through
            # training_analysis_report_to_svg and training_report_to_svg:
            #
            #   before -> <text>nan</text> in the baseline panel and
            #             "<desc>... Baseline accuracy nan, violation 0.000
            #             (constraint satisfied) ... (severity: LOW)</desc>",
            #             while summary() on the SAME report correctly printed
            #             "Accuracy: not measured (no baseline was scored)"
            #   after  -> neither surface draws a number, and the description
            #             carries no "accuracy nan"
            #
            # None and not NaN for exactly the reason recorded on
            # base_rate_disparity below: the rendering layer already speaks this
            # sentinel, and summary() reads it through _measured either way. A
            # measured 0.0 accuracy is a real result and still travels as 0.0.
            baseline_metrics = {
                "accuracy": _as_measured(baseline.accuracy),
                "fairness_violation": baseline.fairness_violation,
                "constraint_satisfied": baseline.constraint_satisfied,
                # Carried beside the value rather than re-derived, so a consumer
                # can tell "no baseline was scored" from "this baseline could not
                # be scored" and print the reason.
                "accuracy_measured": _measured(baseline.accuracy),
                "accuracy_not_measured_reason": baseline.parameters.get(
                    "accuracy_not_measured_reason", ""
                ),
            }

        fairness_analysis = self._compute_fairness_analysis()

        # Method comparison and trade-off sweeps are classification-only
        # (they score with classification accuracy); skip them loudly for
        # regression instead of reporting meaningless numbers.
        is_regression = self.task_type == "regression"
        if (
            is_regression
            and base_estimator is not None
            and (include_comparisons or include_tradeoffs)
        ):
            warnings.warn(
                "task_type='regression': skipping method comparison and "
                "trade-off analysis (classification-only). The report will "
                "contain baseline metrics and group statistics only."
            )

        method_comparisons = []
        if include_comparisons and base_estimator is not None and not is_regression:
            method_comparisons = self.compare_methods(base_estimator)

        tradeoff_analysis = {}
        if include_tradeoffs and base_estimator is not None and not is_regression:
            tradeoff_analysis = self.analyze_tradeoffs(base_estimator)

        recommendation = self.generate_recommendation(method_comparisons)

        critical_issues = self._identify_critical_issues(
            baseline_metrics, fairness_analysis, method_comparisons
        )

        action_items = self._generate_action_items(
            baseline_metrics, fairness_analysis, recommendation
        )

        return FairnessTrainingReport(
            timestamp=datetime.now().isoformat(),
            data_info={
                "n_samples": self.n_samples,
                "n_features": self.X.shape[1] if hasattr(self.X, "shape") else "N/A",
                "n_groups": self.n_groups,
                "groups": self.groups,
                "attribute_name": self.attribute_name,
            },
            task_type=self.task_type,
            baseline_metrics=baseline_metrics,
            fairness_analysis=fairness_analysis,
            method_comparisons=method_comparisons,
            recommendation=recommendation,
            tradeoff_analysis=tradeoff_analysis,
            critical_issues=critical_issues,
            action_items=action_items,
            metadata={
                "tolerance": self.tolerance,
                "config": self.config,
            },
            # Copied, not aliased: the report is a snapshot and a later
            # compare_methods() call replaces self.method_failures wholesale.
            failed_methods=list(self.method_failures),
        )

    def _compute_fairness_analysis(self) -> Dict[str, Any]:
        """Compute detailed fairness analysis."""
        analysis = {
            "constraint_type": str(self.constraint.__class__.__name__),
            "tolerance": self.tolerance,
            "n_groups": self.n_groups,
            "groups": self.groups,
        }

        group_stats = {}
        for group in self.groups:
            mask = self.group_manager.get_mask(group)
            group_stats[group] = {
                "size": int(mask.sum()),
                "proportion": float(mask.sum() / self.n_samples),
                "positive_rate": float(np.mean(self.y[mask]))
                if self.task_type == "classification"
                else float(np.mean(self.y[mask])),
            }
        analysis["group_statistics"] = group_stats

        # A max-min SPREAD needs a pair. Over a one-element list `max - min` is
        # 0.0 by arithmetic vacuity, not by comparison, and 0.0 is identical base
        # rates in every group: the strongest all-clear this number can give.
        # Measured 2026-09-27 through full_analysis on 200 rows with a single
        # sensitive value: fairness_analysis['base_rate_disparity'] == 0.0,
        # summary() printing "base_rate_disparity: 0.0000", no warning, and no
        # critical issue, in the SAME report whose baseline violation, three
        # method comparisons and recommendation all correctly said the constraint
        # could not be evaluated. A non-finite rate (a NaN label in a group) fell
        # the same way: `nan > 0.2` is False, so it read as no disparity too.
        #
        # None, not NaN: the rendering layer already speaks this third state
        # (adapters_training sets `base_rate_measured` from
        # `base_rate_disparity is not None`, and the SVG template gates the row
        # on it), and NaN is not None, so NaN would have been drawn as a
        # measurement. The same defect was fixed for the sibling computation in
        # post_processing/calibration/tradeoffs.py (2026-09-08); this copy was
        # never swept.
        rates = [s["positive_rate"] for s in group_stats.values()]
        if len(rates) < 2 or not all(_measured(r) for r in rates):
            reason = (
                f"only {len(rates)} group(s) present, so no pair of base rates exists to compare"
                if len(rates) < 2
                else "at least one group's base rate is not a finite number"
            )
            warnings.warn(
                f"FairnessTrainingAnalyzer: base_rate_disparity NOT MEASURED ({reason}). "
                "Reporting None (could not check), not 0.0, which would read as identical "
                "base rates across groups.",
                UserWarning,
                stacklevel=3,
            )
            analysis["base_rate_disparity"] = None
            analysis["base_rate_not_measured_reason"] = reason
        else:
            analysis["base_rate_disparity"] = max(rates) - min(rates)

        return analysis

    def _identify_critical_issues(
        self,
        baseline_metrics: Dict[str, Any],
        fairness_analysis: Dict[str, Any],
        method_comparisons: List[MethodComparison],
    ) -> List[Dict[str, Any]]:
        """Identify critical issues from the analysis.

        R4C. "Infeasible Constraint" at severity HIGH used to be raised whenever
        ``not any(c.constraint_satisfied ...)``, which is exactly what an
        UNEVALUATED run looks like: measured on 240 rows whose third group had
        no positive labels, all three methods came back unevaluable and the
        report carried "No training method achieved the fairness constraint" at
        HIGH. Nothing achieved it and nothing failed it, because nothing was
        measured. A constraint nobody could evaluate is its own finding, and it
        is reported as one.
        """
        issues = []

        baseline_violation = _as_measured(baseline_metrics.get("fairness_violation"))
        if baseline_violation is not None and baseline_violation > self.tolerance * 2:
            issues.append(
                {
                    "type": "High Violation",
                    "severity": "high",
                    "description": f"Baseline violation ({baseline_violation:.3f}) "
                    f"significantly exceeds tolerance ({self.tolerance})",
                    "recommendation": "Consider fairness-aware training methods.",
                }
            )

        # The baseline was scored and its constraint was not evaluated: say so,
        # rather than leaving a silence that reads as "no violation found".
        if baseline_metrics and baseline_metrics.get("constraint_satisfied") is None:
            issues.append(
                {
                    "type": "Constraint Not Evaluated",
                    "severity": "unmeasured",
                    "description": "The baseline fairness constraint could not be evaluated "
                    "(a group rate with an empty denominator makes the disparity "
                    "unmeasurable), so the baseline is neither cleared nor in breach",
                    "recommendation": "Collect labels for the affected group(s), or choose a "
                    "constraint whose rates are defined on this data.",
                }
            )

        graded = [c for c in method_comparisons if c.constraint_satisfied is not None]
        if graded and not any(c.constraint_satisfied for c in graded):
            issues.append(
                {
                    "type": "Infeasible Constraint",
                    "severity": "high",
                    "description": "No training method achieved the fairness constraint",
                    "recommendation": "Relax tolerance or consider trade-off approaches.",
                }
            )

        # A method that could not TRAIN is not in method_comparisons at all, so
        # the branch below cannot see it and the branch above counts it out of
        # both the numerator and the denominator. Without this, the worst outcome
        # (nothing trained) was the only one that produced no issue.
        # A PARTIAL FAILURE IS ALSO A THING NOBODY MEASURED (BGL4 audit,
        # 2026-09-27). The branch below is gated on `not method_comparisons`, so
        # with 2 of 3 methods crashing the issue list was ['High Violation',
        # 'Base Rate Disparity'], byte-identical to the control run in which
        # nothing crashed, and the report named a winner among the survivors at
        # priority high. After: the issue below, at severity 'unmeasured' because
        # the failed methods are neither cleared nor in breach.
        if self.method_failures and method_comparisons:
            issues.append(
                {
                    "type": "Method(s) Failed To Train",
                    "severity": "unmeasured",
                    "description": f"{len(self.method_failures)} of "
                    f"{len(self.method_failures) + len(method_comparisons)} requested "
                    "training method(s) failed to fit ("
                    + "; ".join(f"{f['method']}: {f['error']}" for f in self.method_failures[:3])
                    + f"), so the {len(method_comparisons)} comparison(s) in this report are "
                    "not a complete comparison and any recommended method is the best of "
                    "the survivors only",
                    "recommendation": "Fix the training failure(s), then re-run "
                    "compare_methods() so every requested method is either cleared or "
                    "shown to be in breach.",
                }
            )

        if self.method_failures and not method_comparisons:
            issues.append(
                {
                    "type": "No Method Could Be Compared",
                    "severity": "unmeasured",
                    "description": f"All {len(self.method_failures)} requested training "
                    "method(s) failed to fit ("
                    + "; ".join(f"{f['method']}: {f['error']}" for f in self.method_failures[:3])
                    + "), so no method was cleared, none was in breach and nothing was "
                    "compared",
                    "recommendation": "Fix the training failure above, then re-run "
                    "compare_methods(); until then this report carries no method evidence.",
                }
            )

        if method_comparisons and not graded:
            issues.append(
                {
                    "type": "Constraint Not Evaluated",
                    "severity": "unmeasured",
                    "description": f"None of the {len(method_comparisons)} training method(s) "
                    "reported a constraint result, so whether any of them satisfies the "
                    "fairness constraint was never evaluated",
                    "recommendation": "Collect labels for the affected group(s), or choose a "
                    "constraint whose rates are defined on this data, then re-run "
                    "compare_methods().",
                }
            )

        # `> 0.2` on the raw value is how a disparity nobody could compute became
        # "no finding": the producer answered 0.0 for a single group, and None
        # (its honest answer now) would raise TypeError here instead.
        base_rate_disparity = _as_measured(fairness_analysis.get("base_rate_disparity"))
        if base_rate_disparity is not None and base_rate_disparity > 0.2:
            issues.append(
                {
                    "type": "Base Rate Disparity",
                    "severity": "medium",
                    "description": f"Large base rate disparity ({base_rate_disparity:.3f}) "
                    "may make some fairness criteria impossible.",
                    "recommendation": "Consider equalized odds instead of demographic parity.",
                }
            )
        elif fairness_analysis and base_rate_disparity is None:
            issues.append(
                {
                    "type": "Base Rate Disparity Not Measured",
                    "severity": "unmeasured",
                    "description": "The base rate disparity could not be computed ("
                    + str(
                        fairness_analysis.get(
                            "base_rate_not_measured_reason", "no pair of base rates to compare"
                        )
                    )
                    + "), so the groups are neither shown to differ nor shown to agree",
                    "recommendation": "Compare at least two groups with defined base rates "
                    "before reading this report's base-rate section.",
                }
            )

        if fairness_analysis.get("group_statistics"):
            for group, stats in fairness_analysis["group_statistics"].items():
                if stats["size"] < 30:
                    issues.append(
                        {
                            "type": "Small Group",
                            "severity": "warning",
                            "description": f"Group '{group}' has only {stats['size']} samples",
                            "recommendation": "Consider collecting more data or using robust methods.",
                        }
                    )

        return issues

    def _generate_action_items(
        self,
        baseline_metrics: Dict[str, Any],
        fairness_analysis: Dict[str, Any],
        recommendation: TrainingRecommendation,
    ) -> List[str]:
        """Generate prioritized action items.

        R4C. "N/A" joins "Unconstrained" as a method name that is NOT advice:
        it is the sentinel generate_recommendation returns when no method's
        constraint could be evaluated, and "IMPLEMENT: Use N/A training method"
        is an instruction nobody wrote. The unmeasurable-constraint action is
        listed instead, so the reader is left with something to do.
        """
        items = []

        if recommendation.recommended_method not in ("Unconstrained", "N/A"):
            items.append(f"IMPLEMENT: Use {recommendation.recommended_method} training method")

        # A method that crashed is an action too, whether or not the others
        # trained (BGL4 audit, 2026-09-27: the partial-failure case reached no
        # surface of the report at all).
        if self.method_failures:
            items.append(
                f"FIX: {len(self.method_failures)} requested training method(s) failed to "
                "fit ("
                + ", ".join(f["method"] for f in self.method_failures)
                + "); until they train, this report says nothing about whether they "
                "satisfy the constraint"
            )

        baseline_violation = _as_measured(baseline_metrics.get("fairness_violation"))
        if baseline_violation is not None and baseline_violation > self.tolerance:
            items.append("VALIDATE: Test fairness metrics on held-out data")

        # A constraint that could not be evaluated is an action, not a silence.
        # `is None` and never truthiness: False here is a MEASURED breach, which
        # the lines above already act on.
        if baseline_metrics.get("constraint_satisfied") is None and baseline_metrics:
            items.append(
                "MEASURE: The fairness constraint could not be evaluated on this data; "
                "collect labels for the affected group(s) or choose a constraint whose "
                "rates are defined here"
            )

        base_rate_disparity = _as_measured(fairness_analysis.get("base_rate_disparity"))
        if base_rate_disparity is not None and base_rate_disparity > 0.1:
            items.append("INVESTIGATE: Examine causes of base rate disparity")
        elif fairness_analysis and base_rate_disparity is None:
            # An uncomputable disparity is an action, not a silence, exactly as
            # the unevaluable constraint above is.
            items.append(
                "MEASURE: The base rate disparity could not be computed on this data; "
                "supply at least two groups with defined base rates"
            )

        items.append("MONITOR: Set up fairness monitoring for production deployment")

        items.append("DOCUMENT: Record fairness analysis results and decisions")

        return items

    def get_explanation(self, report=None):
        """Generate educational explanations for the training analysis.

        Parameters
        ----------
        report : FairnessTrainingReport, optional
            Pre-computed report. If *None*, :meth:`full_analysis` is run.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_analysis()
        return FairnessExplainer.explain(report)

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the data."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.X.shape[1] if hasattr(self.X, "shape") else "N/A",
            "n_groups": self.n_groups,
            "groups": self.groups,
            "task_type": self.task_type,
            "constraint": str(self.constraint.__class__.__name__),
            "tolerance": self.tolerance,
        }


# ===========================================================================
# CS-T-41: Baseline accuracy-only vs fairness-constrained training comparison.
#
# A small, dependency-light summary that contrasts an unconstrained baseline
# with a fairness-aware model on the SAME held-out test set. It quantifies the
# accuracy cost of the intervention, the fairness gain (reduction in the primary
# disparity metric), and the per-group accuracy impact. numpy only; no torch.
# ===========================================================================


def _stable_unique(arr: np.ndarray) -> List[Any]:
    """Unique values preserving first-seen order (labels may be strings)."""
    seen = []
    seen_set = set()
    for v in arr.tolist():
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    return seen


def _first_violation(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    """Magnitude of the first non-accuracy, higher-is-NOT-better fairness metric.

    Mirrors the handler's own ``_get_violation`` convention: skip the accuracy
    key and any metric where a HIGHER value is the healthy one (a 0.8 ratio is
    fair, not a violation of 0.8), and take the absolute value of the first
    remaining numeric metric.

    Returns None when NO violation-type metric was measurable, which is not the
    same statement as a violation of 0.0 and must not be rendered as one. The
    comment below already said so ("answering 0.0 reads as 'no disparity' for a
    disparity nobody looked at") while the function went on returning 0.0 for
    exactly that case: an all-NaN metrics dict, or one carrying no
    violation-type metric at all.
    """
    if not isinstance(metrics, dict):
        return None
    for k, v in metrics.items():
        if k == "accuracy":
            continue
        # Which way is better is asked of the single owner of that question,
        # ``_metric_direction``, and never re-derived from the name here. The
        # local rule this replaces was an exact ``endswith("_ratio")`` suffix
        # test, written that way because a substring test for "ratio" matched
        # "cali[bratio]n_difference" (CLAUDE.md, reconciled in 47f1e09f8). The
        # exact suffix is narrower than the truth: ``disparate_impact`` IS a
        # four-fifths ratio and does not end in "_ratio", so a healthy 0.85 was
        # reported as a 0.85 violation, and so was ``worst_group_accuracy``.
        # A direction the helper cannot resolve is left in place deliberately:
        # this is a violation MAGNITUDE, and skipping every unresolved name (a
        # theil_index, a house metric) would answer 0.0, which reads as "no
        # disparity" for a disparity nobody looked at.
        if metric_direction(str(k)) is MetricDirection.HIGHER_IS_BETTER:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if fv != fv:  # NaN
                continue
            return abs(fv)
    return None


def _baseline_comparison_verdict(
    fairness_improved: Optional[bool],
    fairness_gain: float,
    accuracy_cost: float,
    not_assessed_reason: str = "",
    accuracy_measured: bool = True,
) -> str:
    """Plain-language one-liner describing the accuracy vs fairness trade-off."""
    cost = float(accuracy_cost)
    if fairness_improved is None:
        return (
            "Fairness impact NOT ASSESSED: {0}. This says nothing about whether the "
            "constraints helped, hurt, or changed nothing.".format(
                not_assessed_reason or "no disparity metric was measurable on either side"
            )
        )
    gain = abs(float(fairness_gain))
    # An unmeasured cost used to fall through both comparisons (every comparison
    # against NaN is False) onto the "no accuracy cost" arm, which is the clean
    # answer: the intervention was free. It is the third arm's job to say so.
    if not accuracy_measured:
        cost_phrase = "an accuracy cost that was NOT MEASURED"
    elif cost > 0:
        cost_phrase = "a {0:.3f} accuracy cost".format(cost)
    elif cost < 0:
        cost_phrase = "a {0:.3f} accuracy gain".format(abs(cost))
    else:
        cost_phrase = "no accuracy cost"
    if fairness_improved:
        return "Fairness constraints reduced disparity by {0:.3f} at {1}.".format(gain, cost_phrase)
    if fairness_gain == 0:
        return "Fairness constraints left disparity unchanged at {0}.".format(cost_phrase)

    return (
        "Fairness constraints increased disparity by {0:.3f} at {1}; "
        "the intervention did not help on this metric.".format(gain, cost_phrase)
    )


def baseline_comparison_summary(
    y_test: ArrayLike,
    y_pred_baseline: ArrayLike,
    y_pred_fair: ArrayLike,
    sensitive_by_attr: Dict[str, ArrayLike],
    before_metrics: Optional[Dict[str, Any]] = None,
    after_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured baseline vs fairness-constrained training comparison.

    Contrasts an unconstrained baseline model against a fairness-aware model on
    the same held-out test set, quantifying the accuracy cost, the fairness gain
    (reduction in the primary disparity metric), and the per-group accuracy
    impact of the intervention.

    Args:
        y_test: True labels on the held-out test set.
        y_pred_baseline: Baseline (unconstrained) model predictions.
        y_pred_fair: Fairness-aware model predictions.
        sensitive_by_attr: Mapping of protected-attribute name to the array of
            group labels for the test rows. Per-group accuracy is reported for
            the FIRST attribute; with several attributes the group keys are
            written as "group (attr)" to match the handler convention.
        before_metrics: Baseline metrics dict (must carry 'accuracy'; its first
            non-ratio fairness metric is used as the baseline violation).
        after_metrics: Fairness-aware metrics dict (same shape as before).

    Returns:
        Dict with accuracy_cost, accuracy_measured, accuracy_not_measured_reason,
        n_test_rows, fairness_gain, fairness_improved, baseline, fair,
        per_group_impact and verdict (see module docstring for shapes). THREE
        STATES on both axes: ``fairness_improved`` is None and ``fairness_gain``
        NaN when no disparity metric was measurable, and ``accuracy_measured`` is
        False with ``accuracy_cost`` NaN when no row could be scored.

        "No row could be scored" counts LABELS and not rows: a test set of 40
        rows carrying nothing but NaN labels is refused the same way an empty one
        is, and so is a prediction array that does not line up row for row with
        the labels. Each ``per_group_impact`` entry carries its own
        ``acc_measured`` flag and ``not_measured_reason``, because the per-group
        table is a SECOND writer of the same quantity and used to fabricate a
        0.0 delta while the top-level accuracy was refused.
    """
    y_test = np.asarray(y_test)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_fair = np.asarray(y_pred_fair)

    # Flattened once, and used for every row-wise comparison and every group
    # mask below. See _scored on why a shape is not a detail here.
    y_test_flat = y_test.reshape(-1)
    base_flat = y_pred_baseline.reshape(-1)
    fair_flat = y_pred_fair.reshape(-1)

    # Why each side could not be scored, keyed by side, filled by
    # _overall_accuracy and read by the single summary warning below. Collected
    # rather than warned about on the spot so a run with nothing scorable still
    # raises ONE warning naming the count, which is the graded behaviour.
    unscored_reasons: Dict[str, str] = {}

    def _scored(y_true: np.ndarray, y_pred: np.ndarray, where: str) -> Tuple[float, str]:
        """Row-wise accuracy, or NaN and the reason it is not a measurement.

        COUNT LABELS, NOT ROWS (BGL4 audit wave 4, 2026-09-30). The guard this
        replaces was ``len(y_test) == 0``, a ROW COUNT, and the quantity it
        needs is LABELS. ``y_pred == y_test`` is False for a NaN label because
        NaN equals nothing, so an unlabelled row scores as a row the model got
        WRONG: both sides fabricate the same 0.0, the subtraction gives exactly
        0.0, and ``_measured`` sees two finite zeros and says True. Measured
        2026-09-30 on 40 rows of ``np.full(40, np.nan)`` labels with both
        metrics dicts carrying a real disparity:

            before -> accuracy_cost 0.0, accuracy_measured TRUE, 0 warnings,
                      "Fairness constraints reduced disparity by 0.250 at no
                      accuracy cost.", per_group {f: 0.0, m: 0.0} on both models
            after  -> accuracy_cost NaN, accuracy_measured False, 1 warning,
                      "... at an accuracy cost that was NOT MEASURED."

        That is the same sentence, to the digit, that the empty-test-set comment
        below gives as its whole justification, produced with 40 rows.

        A BROADCAST IS NOT AN ACCURACY, the second door into the same sentence.
        ``np.mean(y_pred == y_test)`` with a (40, 1) prediction against a (40,)
        label array does not compare row against row: it broadcasts to a 40x40
        matrix of every prediction against every label, whose mean is the chance
        rate of the class mix. Measured 2026-09-30 on a PERFECT prediction:

            flat (40,)    -> baseline accuracy 1.0, per_group {f: 1.0, m: 1.0}
            column (40,1) -> baseline accuracy 0.5, per_group {f: 0.5, m: 0.5},
                             accuracy_measured TRUE, zero warnings, BEFORE

        Flattening is not a coercion of somebody's data, it is the row-wise
        comparison the docstring already promises. A row COUNT that does not
        line up cannot be reconciled, so it is the could-not-check instead. This
        is the fix already carried by loss_functions/adversarial.py:808-850,
        applied here because this is the reporting surface.
        """
        true_flat = np.asarray(y_true).reshape(-1)
        pred_flat = np.asarray(y_pred).reshape(-1)
        if true_flat.shape[0] == 0:
            return float("nan"), "{0} has no rows to score".format(where)
        if pred_flat.shape[0] != true_flat.shape[0]:
            return float("nan"), (
                "{0} carries {1} prediction(s) for {2} label(s), so they cannot be "
                "compared row by row".format(where, pred_flat.shape[0], true_flat.shape[0])
            )
        n_unscorable = _unscorable_rows(true_flat, pred_flat)
        if n_unscorable:
            return float("nan"), (
                "{0} of {1} row(s) in {2} carry a non-finite label or prediction, so they "
                "cannot be scored".format(n_unscorable, true_flat.shape[0], where)
            )
        return float(np.mean(pred_flat == true_flat)), ""

    def _overall_accuracy(y_pred: np.ndarray, side: str = "baseline") -> float:
        # NaN, not 0.0, on an empty test set. 0.0 accuracy is a MEASUREMENT: a
        # model that got every row wrong. With zero rows nothing was scored, and
        # because both sides got the same fabricated 0.0 the subtraction below
        # produced accuracy_cost exactly 0.0, which the verdict then worded as
        # "at no accuracy cost". Measured 2026-09-27 with zero test rows and both
        # metrics dicts carrying a real disparity: "Fairness constraints reduced
        # disparity by 0.250 at no accuracy cost.", baseline accuracy 0.0, fair
        # accuracy 0.0, and not one warning. That sentence is the whole point of
        # this function and it was produced without scoring a single row.
        #
        # The zero-row case is now the FIRST of three refusals inside _scored,
        # which reaches the unlabelled and the misshapen test set as well; see
        # its docstring for the two runs that got past a row count.
        value, reason = _scored(y_test_flat, y_pred, "the test set")
        if reason:
            unscored_reasons[side] = reason
            return float("nan")
        return value

    def _supplied_accuracy(
        metrics: Optional[Dict[str, Any]],
        side: str,
        param: str,
        y_pred: np.ndarray,
    ) -> float:
        """The accuracy the caller recorded, refused unless it is a measurement.

        CHECKED BEFORE IT IS COERCED (BGL4 audit, 2026-09-27). This read
        ``float(metrics["accuracy"])`` and handed the result on, so the
        module's own ``_measured`` never saw the raw value and could not apply
        the rule its docstring states ("a bool is not a magnitude: True would
        print as 1.0000"). Measured on six scored rows with
        before_metrics carrying a disparity of 0.3 and after_metrics 0.1:

            accuracy=True  before -> accuracy_measured True, accuracy_cost
                                     0.30000000000000004, verdict "Fairness
                                     constraints reduced disparity by 0.200 at
                                     a 0.300 accuracy cost.", NO warning
                           after  -> accuracy_measured False, cost NaN, verdict
                                     "... at an accuracy cost that was NOT
                                     MEASURED.", two warnings
            accuracy='0.9' before -> accuracy_measured True, cost 0.2000, silent
                           after  -> refused the same way
            accuracy=None  before -> TypeError: float() argument must be a
                                     string or a real number, not 'NoneType'
                           after  -> accuracy_measured False, cost NaN

        None is this library's own not-measured sentinel, so the one value most
        likely to arrive in an unscored accuracy slot was the one that raised.
        ``summary()`` on FairnessTrainingReport already refused all three on the
        same input, through the same ``_measured``; this was the surface that
        did not.
        """
        if not (isinstance(metrics, dict) and "accuracy" in metrics):
            return _overall_accuracy(y_pred, side)
        supplied = metrics["accuracy"]
        value = _as_measured(supplied)
        if value is None:
            unscored_reasons[side] = (
                "the accuracy supplied in {0}['accuracy'] is {1!r}, which is not a "
                "measurement".format(param, supplied)
            )
            warnings.warn(
                f"baseline_comparison_summary: the {side} accuracy supplied in "
                f"{param}['accuracy'] is {supplied!r}, which is not a measurement (None, NaN, "
                "an infinity and a bool are all refused: float(True) is 1.0, a perfect "
                "score). Reporting NaN (could not check) rather than coercing it, and NOT "
                "falling back to scoring the predictions, which would report a number the "
                "caller did not supply.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return value

    # Prefer the accuracy the caller already recorded (identical formula); fall
    # back to a direct computation when the metrics dict is absent.
    baseline_acc = _supplied_accuracy(before_metrics, "baseline", "before_metrics", y_pred_baseline)
    fair_acc = _supplied_accuracy(after_metrics, "fair", "after_metrics", y_pred_fair)

    baseline_violation = _first_violation(before_metrics)
    fair_violation = _first_violation(after_metrics)

    # An accuracy nobody could score leaves nothing to subtract, the same
    # argument the fairness half below makes. See _overall_accuracy on why 0.0
    # was worse than NaN here: it made the DIFFERENCE 0.0 as well.
    accuracy_measured = _measured(baseline_acc) and _measured(fair_acc)
    accuracy_not_measured_reason = ""
    if accuracy_measured:
        accuracy_cost = float(baseline_acc - fair_acc)
    else:
        unscored = [
            side
            for side, value in (("baseline", baseline_acc), ("fair", fair_acc))
            if not _measured(value)
        ]
        # The WHY travels with the flag, in the same dict, so a consumer that
        # withholds the number has the sentence to print in its place (the
        # convention evaluate_baseline already follows for
        # parameters['accuracy_not_measured_reason']). Without it "not measured"
        # cannot be told from "not supplied".
        accuracy_not_measured_reason = "; ".join(
            "{0}: {1}".format(side, unscored_reasons[side])
            for side in unscored
            if side in unscored_reasons
        )
        warnings.warn(
            "baseline_comparison_summary: no accuracy was measurable on the "
            + " and ".join(unscored)
            + (" side" if len(unscored) == 1 else " sides")
            + f" ({len(y_test)} test row(s)), so accuracy_cost cannot be computed. "
            "Reporting NaN (could not check), NOT 'no accuracy cost'."
            + (" " + accuracy_not_measured_reason + "." if accuracy_not_measured_reason else ""),
            UserWarning,
            stacklevel=2,
        )
        accuracy_cost = float("nan")

    # An unmeasurable disparity on EITHER side leaves nothing to subtract.
    # Measured 2026-09-08 with both metrics dicts carrying a NaN
    # demographic_parity_difference: gain 0.0, fairness_improved False, verdict
    # "Fairness constraints left disparity unchanged at no accuracy cost." That
    # is a finding about an intervention nobody evaluated, and it is the exact
    # sentence a reader would quote to justify shipping the fair model.
    fairness_improved: Optional[bool]
    if baseline_violation is None or fair_violation is None:
        missing = [
            side
            for side, value in (("baseline", baseline_violation), ("fair", fair_violation))
            if value is None
        ]
        not_assessed_reason = (
            "no violation-type fairness metric was measurable on the "
            + " and ".join(missing)
            + (" side" if len(missing) == 1 else " sides")
        )
        warnings.warn(
            f"baseline_comparison_summary: {not_assessed_reason}, so fairness_gain "
            f"cannot be computed. Reporting fairness_improved=None (could not check), "
            f"NOT 'disparity unchanged'.",
            UserWarning,
            stacklevel=2,
        )
        fairness_gain = float("nan")
        fairness_improved = None
    else:
        not_assessed_reason = ""
        fairness_gain = float(baseline_violation - fair_violation)
        fairness_improved = bool(fairness_gain > 0)

    # Per-group accuracy on the FIRST protected attribute.
    per_group_baseline: Dict[str, float] = {}
    per_group_fair: Dict[str, float] = {}
    per_group_impact: List[Dict[str, Any]] = []
    # THE SECOND WRITER OF THE SAME QUANTITY (BGL4 audit wave 4, 2026-09-30).
    # These two lines were bare `float(np.mean(...))` with no _measured and no
    # unscorable-row count, so the panel a reader consults for "which group paid
    # for this" read 0.0 for both models on the all-NaN test set and 0.5 for both
    # on a perfect (n, 1) prediction, while the overall accuracy beside it was
    # already being refused. A correct top-level field beside a fabricated
    # per-group table is still a live defect: the table is what a person reads.
    # Scored through the same _scored as the overall accuracy, per group, so the
    # two writers cannot disagree.
    n_groups_unmeasured = 0
    if sensitive_by_attr:
        attr_names = list(sensitive_by_attr.keys())
        multi = len(attr_names) > 1
        first_attr = attr_names[0]
        sens = np.asarray(sensitive_by_attr[first_attr]).reshape(-1)
        if len(sens) == len(y_test_flat):
            for g in _stable_unique(sens):
                mask = sens == g
                n_g = int(np.sum(mask))
                if n_g == 0:
                    continue
                key = "{0} ({1})".format(g, first_attr) if multi else str(g)
                acc_b, reason_b = _scored(
                    y_test_flat[mask],
                    base_flat[mask] if base_flat.shape[0] == y_test_flat.shape[0] else base_flat,
                    "group {0!r} of the baseline model".format(key),
                )
                acc_f, reason_f = _scored(
                    y_test_flat[mask],
                    fair_flat[mask] if fair_flat.shape[0] == y_test_flat.shape[0] else fair_flat,
                    "group {0!r} of the fair model".format(key),
                )
                group_measured = _measured(acc_b) and _measured(acc_f)
                if not group_measured:
                    n_groups_unmeasured += 1
                per_group_baseline[key] = acc_b
                per_group_fair[key] = acc_f
                per_group_impact.append(
                    {
                        "group": key,
                        "acc_baseline": acc_b,
                        "acc_fair": acc_f,
                        # NaN, not 0.0: a delta of 0.0 says the intervention cost
                        # this group nothing, which is a finding.
                        "delta": float(acc_f - acc_b) if group_measured else float("nan"),
                        # Read beside the three numbers above: False means they
                        # are could-not-check, not measurements.
                        "acc_measured": group_measured,
                        "not_measured_reason": "; ".join(r for r in (reason_b, reason_f) if r),
                    }
                )
    if n_groups_unmeasured:
        warnings.warn(
            "baseline_comparison_summary: {0} of {1} group(s) in the per-group accuracy "
            "table could not be scored, so their acc_baseline, acc_fair and delta are NaN "
            "with per_group_impact[...]['acc_measured'] False beside them, NOT a 0.0 delta "
            "(which reads as 'this group paid nothing'). The reason is on each "
            "entry.".format(n_groups_unmeasured, len(per_group_impact)),
            UserWarning,
            stacklevel=2,
        )

    verdict = _baseline_comparison_verdict(
        fairness_improved,
        fairness_gain,
        accuracy_cost,
        not_assessed_reason,
        accuracy_measured=accuracy_measured,
    )

    return {
        "accuracy_cost": accuracy_cost,
        # Read beside accuracy_cost: False means the NaN above is could-not-check
        # and the verdict says so in words.
        "accuracy_measured": accuracy_measured,
        # And WHY, in the same dict as the flag, empty string when it was
        # measured. "40 of 40 row(s) carry a non-finite label" is the difference
        # between a test set nobody labelled and a metrics dict nobody filled in.
        "accuracy_not_measured_reason": accuracy_not_measured_reason,
        "n_test_rows": int(len(y_test)),
        "fairness_gain": fairness_gain,
        "fairness_improved": fairness_improved,
        "baseline": {
            "accuracy": baseline_acc,
            "violation": baseline_violation,
            "per_group_accuracy": per_group_baseline,
        },
        "fair": {
            "accuracy": fair_acc,
            "violation": fair_violation,
            "per_group_accuracy": per_group_fair,
        },
        "per_group_impact": per_group_impact,
        "verdict": verdict,
    }
