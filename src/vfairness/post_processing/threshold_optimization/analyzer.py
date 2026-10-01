"""
Threshold Analysis Tools for vfairness.

This module provides comprehensive analysis tools for understanding the
impact of different thresholds on fairness and performance metrics.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_probabilities,
)
from vfairness.exceptions import InvalidDataError

from .constraints import (
    ConstraintViolation,
    FairnessConstraintType,
    compute_constraint_violation,
)
from .optimizer import (
    _compute_performance_metrics,
    _record_score_resolution,
    _require_searchable_grid,
)

# The objective names ``_compute_performance_metrics`` actually returns, and so
# the only names ``find_optimal_threshold`` can optimise.
#
# This exists because the objective used to be read as
# ``metrics.get(objective, metrics["accuracy"])``. A name that is not a key, the
# very plausible ``objective="f1"`` for the key ``"f1_score"`` among them, fell
# through to ACCURACY: the search then maximised accuracy, and the caller was
# handed back an "optimal threshold" for an objective that was never optimised,
# with nothing on the returned dict saying so. A mistyped objective is refused
# here instead, before any threshold is searched. The lookup below is an exact
# subscript, not a defaulted get, so if this table ever drifts from the metric
# function the result is a loud KeyError rather than a silent substitution.
PERFORMANCE_OBJECTIVES = frozenset(
    {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "balanced_accuracy",
    }
)


class FeasibleRegion(Tuple[Optional[float], Optional[float]]):
    """The ``(lower, upper)`` endpoints of a feasible region, plus the third
    state the pair alone cannot carry.

    ``compute_constraint_violation`` answers ``is_satisfied=None`` when a group
    rate has an empty denominator: COULD NOT CHECK. A bare pair of endpoints has
    exactly two states, so a sweep in which NOTHING was measurable came back as
    ``(None, None)``, identical to a sweep that measured every threshold and
    found none feasible. Measured 2026-09-10 on 100 thresholds of an
    equal_opportunity sweep where one group had no positive labels: 100 of 100
    could not be checked, and the report still recommended group-specific
    thresholds on the strength of zero measurements.

    It stays a two element tuple, so ``region[0]``, ``low, high = region`` and
    ``region == (None, None)`` keep working for callers that only want the
    endpoints. ``region.assessed`` is how a caller recovers the third state.

    Attributes
    ----------
    assessed : bool
        ``True`` when at least one searched threshold produced a measured
        verdict, so ``(None, None)`` means MEASURED INFEASIBLE. ``False`` when
        every searched threshold was could-not-check, so ``(None, None)`` means
        NOT ASSESSED and says nothing about feasibility in either direction.
    n_not_assessed : int
        How many of the searched thresholds were could-not-check.
    n_searched : int
        How many thresholds were searched in total.
    n_feasible : int
        How many of the searched thresholds MEASURABLY satisfied the constraint.
    contiguous : bool
        Whether those thresholds form one unbroken run of the searched grid. The
        pair is the ``min`` and ``max`` of the satisfying set, and when the set
        has gaps the pair reads as an interval that contains thresholds MEASURED
        as violating. Measured 2026-09-27 on 200 rows in two groups, scores in
        [0.02, 0.98], demographic_parity at tolerance 0.05: 12 of 100 searched
        thresholds satisfied it, indices 0 to 5 and 94 to 99, the two degenerate
        ends where every row falls on one side of the cut. The pair returned was
        ``(0.01, 0.99)``, ``summary()`` printed "[0.010, 0.990]", and 0.5 sat
        inside it having been measured at a violation of 0.46.
    n_distinct_scores : int or None
        How many distinct finite values ``y_prob`` held, or ``None`` when the
        region was built by hand and no score column was measured. A sweep needs
        at least :data:`~.optimizer.SCORE_RESOLUTION_FLOOR` of them before one
        threshold can decide anything a different threshold does not.
    n_distinct_decisions : int or None
        How many DISTINCT decision vectors the searched thresholds actually
        produced between them, or ``None`` when the region was built by hand. This
        is the count ``n_searched`` reads as: a sweep of 100 candidates over scores
        that all sit above 0.99 produces ONE decision, approve everybody, at every
        one of them, and approving everybody is trivially equal on every parity
        constraint. Measured 2026-09-30 on 200 rows in two groups with scores in
        [0.995, 0.9999]: 100 of 100 searched thresholds "satisfied"
        demographic_parity, the pair came back ``(0.01, 0.99)`` with
        ``contiguous=True``, ``summary()`` printed "[0.010, 0.990]",
        ``find_optimal_threshold`` answered 0.01, and the report said "Multiple
        constraints can be satisfied". The 100 thresholds had made one decision.

        Exact, not a heuristic: ``y_prob >= t`` is monotone in ``t``, so two
        thresholds with the same number of approvals approve the same rows, and
        counting distinct approval counts counts distinct decisions.
    score_resolution_sufficient : bool or None
        Three states, and ``None`` is one of them: ``True`` when the score column
        had enough resolution for the sweep to separate thresholds, ``False`` when
        it did not, ``None`` when nothing was measured (a hand-built region). It
        is NOT defaulted to ``True``, because a reassuring default for an absent
        measurement is the shape this class exists to remove. Measured 2026-09-30
        on 200 rows in two groups whose ``y_prob`` was the PREDICTION column, hard
        0.0/1.0: all 100 searched thresholds produced ONE decision vector, the
        region came back ``(None, None)`` "infeasible" with ``n_searched`` 100 and
        ``find_optimal_threshold`` answered 0.01, and the report recommended
        group-specific thresholds. Not one warning said the sweep had nothing to
        search. The verdict on those predictions is real; the 100 candidates were
        not.
    """

    assessed: bool
    n_not_assessed: int
    n_searched: int
    n_feasible: int
    contiguous: bool
    n_distinct_scores: Optional[int]
    n_distinct_decisions: Optional[int]
    score_resolution_sufficient: Optional[bool]

    def __new__(
        cls,
        lower: Optional[float],
        upper: Optional[float],
        *,
        assessed: bool = True,
        n_not_assessed: int = 0,
        n_searched: int = 0,
        n_feasible: int = 0,
        contiguous: bool = True,
        n_distinct_scores: Optional[int] = None,
        n_distinct_decisions: Optional[int] = None,
        score_resolution_sufficient: Optional[bool] = None,
    ) -> "FeasibleRegion":
        self = super().__new__(cls, (lower, upper))
        self.assessed = assessed
        self.n_not_assessed = n_not_assessed
        self.n_searched = n_searched
        self.n_feasible = n_feasible
        self.contiguous = contiguous
        self.n_distinct_scores = n_distinct_scores
        self.n_distinct_decisions = n_distinct_decisions
        self.score_resolution_sufficient = score_resolution_sufficient
        return self

    @property
    def status(self) -> str:
        """``"feasible"``, ``"infeasible"`` or ``"not_assessed"``."""
        if not self.assessed:
            return "not_assessed"
        return "feasible" if self[0] is not None else "infeasible"


def _region_to_dict(region: Any) -> Dict[str, Any]:
    """A FeasibleRegion as JSON-safe fields, third state included.

    A plain ``(lower, upper)`` tuple is accepted and reported as
    ``status="unknown"``: the pair carries no evidence about which of the three
    states it is in, and asserting "assessed" for it is the ambiguity
    :class:`FeasibleRegion` exists to remove.
    """
    lower, upper = (region[0], region[1])
    if not isinstance(region, FeasibleRegion):
        return {"lower": lower, "upper": upper, "status": "unknown"}
    return {
        "lower": lower,
        "upper": upper,
        "status": region.status,
        "assessed": region.assessed,
        "contiguous": region.contiguous,
        "n_feasible": region.n_feasible,
        "n_searched": region.n_searched,
        "n_not_assessed": region.n_not_assessed,
        # Serialised HERE for the reason the rest of this dict exists: a consumer
        # that json-encodes the report reads these fields and nothing else, so a
        # "searched 100 thresholds" that had one decision to make has to be
        # legible at this boundary too, not only in a warning at construction.
        "n_distinct_scores": region.n_distinct_scores,
        "n_distinct_decisions": region.n_distinct_decisions,
        "score_resolution_sufficient": region.score_resolution_sufficient,
    }


@dataclass
class ThresholdImpactResult:
    """
    Result of analyzing a single threshold value.

    Attributes:
        threshold: The threshold value analyzed
        performance_metrics: Overall performance metrics
        group_metrics: Per-group performance metrics
        constraint_violations: Fairness constraint violations
        confusion_matrix: Overall confusion matrix components
        group_confusion_matrices: Per-group confusion matrices
    """

    threshold: float
    performance_metrics: Dict[str, float]
    group_metrics: Dict[str, Dict[str, float]]
    constraint_violations: Dict[str, ConstraintViolation]
    confusion_matrix: Dict[str, int]
    group_confusion_matrices: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "threshold": self.threshold,
            "performance_metrics": self.performance_metrics,
            "group_metrics": self.group_metrics,
            "constraint_violations": {
                k: v.to_dict() for k, v in self.constraint_violations.items()
            },
            "confusion_matrix": self.confusion_matrix,
            "group_confusion_matrices": self.group_confusion_matrices,
        }


@dataclass
class ThresholdAnalysisReport:
    """
    Comprehensive threshold analysis report.

    Attributes:
        threshold_results: Results for each analyzed threshold
        optimal_thresholds: Optimal thresholds for each constraint
        feasible_regions: Threshold regions satisfying each constraint, each a
            FeasibleRegion whose ``assessed`` flag separates a measured
            "no threshold works" from "nothing could be checked"
        recommendations: Analysis recommendations
    """

    threshold_results: List[ThresholdImpactResult]
    # None when no threshold could be assessed for that constraint at all, which
    # is not the same as "the best one is 0.01": see FeasibleRegion.
    optimal_thresholds: Dict[str, Dict[str, Optional[float]]]
    # (None, None) is returned by find_feasible_region when no feasible
    # threshold exists, so the tuple elements are genuinely Optional. The
    # FeasibleRegion subclass carries `assessed`, which separates that MEASURED
    # infeasibility from a sweep where nothing could be checked.
    feasible_regions: Dict[str, FeasibleRegion]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation.

        The feasible regions are SERIALISED here rather than handed over as
        objects. A :class:`FeasibleRegion` is a tuple subclass, so a consumer that
        json-encodes this dict got a bare pair and nothing else: measured
        2026-09-27 on 200 rows of one group, where all 100 searched thresholds
        were could-not-check, ``json.dumps(list(region))`` was ``[null, null]``
        for the NOT ASSESSED region and ``[null, null]`` for a MEASURED
        INFEASIBLE one, and ``assessed``, ``contiguous``, ``n_feasible``,
        ``n_searched`` and ``n_not_assessed`` were all dropped. That is the exact
        ambiguity the FeasibleRegion type was added to remove, reintroduced at the
        boundary a consumer reads. After: each region is a dict carrying
        ``status`` ("feasible" / "infeasible" / "not_assessed") beside the
        endpoints, so the two cases differ in the encoded output.

        ``self.feasible_regions`` itself is untouched: callers that read the
        attribute keep the pair interface (``region[0]``, ``low, high = region``,
        ``region == (None, None)``).
        """
        return {
            "threshold_results": [r.to_dict() for r in self.threshold_results],
            "optimal_thresholds": self.optimal_thresholds,
            "feasible_regions": {
                constraint: _region_to_dict(region)
                for constraint, region in self.feasible_regions.items()
            },
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Threshold Analysis Report", "=" * 50]

        lines.append(f"\nAnalyzed {len(self.threshold_results)} threshold values")

        lines.append("\nOptimal Thresholds by Constraint:")
        for constraint, thresholds in self.optimal_thresholds.items():
            lines.append(f"  {constraint}:")
            for metric, thresh in thresholds.items():
                if thresh is None:
                    lines.append(f"    Best for {metric}: NOT ASSESSED")
                else:
                    lines.append(f"    Best for {metric}: {thresh:.3f}")

        lines.append("\nFeasible Regions:")
        for constraint, region in self.feasible_regions.items():
            # Three states. "No feasible region found" is a FINDING: it says
            # every searched threshold was measured and none satisfied the
            # constraint. It must not stand in for a sweep that measured
            # nothing, which is what an unassessed region used to print.
            upper = region[1]
            if region[0] is not None and upper is not None:
                gapped = isinstance(region, FeasibleRegion) and not region.contiguous
                if gapped:
                    # The brackets are what a reader acts on, so the gap has to
                    # be inside them. Printing "[0.010, 0.990]" for a satisfying
                    # set of {0.01..0.06} plus {0.94..0.99} told the reader that
                    # 0.5 was feasible while the sweep had measured it violating.
                    lines.append(
                        f"  {constraint}: {region.n_feasible} of {region.n_searched} "
                        f"searched thresholds satisfy it, between {region[0]:.3f} and "
                        f"{upper:.3f} but NOT as one run: thresholds inside that span "
                        "were measured as violating"
                    )
                else:
                    lines.append(f"  {constraint}: [{region[0]:.3f}, {upper:.3f}]")
            elif not (region.assessed if isinstance(region, FeasibleRegion) else True):
                lines.append(
                    f"  {constraint}: NOT ASSESSED (no threshold could be checked; "
                    "this is not a finding of infeasibility)"
                )
            else:
                lines.append(f"  {constraint}: No feasible region found")

            # THE CAVEAT GOES ON THE LINE THE READER ACTS ON. Every branch above
            # prints a region as though the sweep had chosen between thresholds.
            # When all of the searched thresholds produced the SAME decision, the
            # line just appended describes one decision rather than a range, and
            # the reader acts on the brackets. Appended to lines[-1] rather than
            # repeated inside all three branches, so a fourth branch cannot be
            # added without it.
            n_decisions = getattr(region, "n_distinct_decisions", None)
            if n_decisions is not None and n_decisions <= 1:
                lines[-1] += (
                    f" [NOT A SEARCH: all {region.n_searched} searched threshold(s) made "
                    f"the SAME decision ({n_decisions} distinct decision(s)), so this is "
                    "one decision rule and not a range of thresholds; a cut that puts "
                    "every row on one side is trivially equal across groups]"
                )

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


class ThresholdAnalyzer:
    """
    Comprehensive threshold analysis for fairness-aware classification.

    This analyzer evaluates the impact of different threshold values on
    fairness metrics and model performance, helping practitioners understand
    the trade-offs involved in threshold selection.

    Example:
        >>> analyzer = ThresholdAnalyzer(y_true, y_prob, gender)
        >>> report = analyzer.full_analysis()
        >>> print(report.summary())
        >>>
        >>> # Analyze specific thresholds
        >>> results = analyzer.analyze_threshold_range(
        ...     thresholds=np.linspace(0.1, 0.9, 9)
        ... )
        >>>
        >>> # Find optimal threshold for a constraint
        >>> optimal = analyzer.find_optimal_threshold(
        ...     constraint='demographic_parity',
        ...     objective='accuracy'
        ... )

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: threshold_analysis. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """
        Initialize the threshold analyzer.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute for grouping
            sample_weight: Optional sample weights

        Raises:
            InvalidDataError: If any probability is missing or outside [0, 1].
                Refused HERE, at construction, because every method of this class
                thresholds ``y_prob`` in its own body with
                ``(self.y_prob >= t).astype(int)``, and ``nan >= t`` is False: an
                unscored row became a hard 0, a MEASURED rejection, in
                ``analyze_threshold``, ``analyze_threshold_range``,
                ``find_optimal_threshold``, ``find_feasible_region`` and
                ``full_analysis`` alike. One validation above them all closes
                every one; a guard inside any single method cannot.

                Measured 2026-09-27 on 200 rows in two 100-row groups where group
                B was never scored (every y_prob NaN):
                  before  analyze_threshold(0.5) returned demographic_parity
                          violation 0.5, group_metrics {'A': 0.5, 'B': 0.0},
                          is_satisfied False and unmeasured=(), with group B's
                          100 unscored rows counted as true negatives and false
                          negatives in group_confusion_matrices;
                          find_feasible_region answered status "feasible",
                          (0.9405, 0.99), n_feasible 6, contiguous True and
                          n_not_assessed 0; find_optimal_threshold answered
                          optimal_threshold 0.9405, is_feasible True and
                          n_thresholds_not_assessed 0. The two counters actively
                          asserted that nothing had been unmeasurable.
                  after   InvalidDataError at construction, naming the 100
                          missing values.
        """
        # AND THE LABEL, one line above where that guard was written (BGL6 F04,
        # 2026-09-28). `.astype(int)` was applied to y_true BEFORE anything looked at
        # it, and numpy casting NaN to int is undefined: it yields a platform
        # dependent integer rather than raising, so 100 unlabelled rows became 100
        # hard labels and every rate computed from them read as measured.
        #
        # Measured on 200 rows in two 100-row groups where group B carries no label at
        # all: analyze_threshold(0.5, ['equalized_odds']) returned violation 1.0,
        # is_satisfied False, group B fpr 1.0 and a confusion matrix of
        # {'tp': 0, 'tn': 0, 'fp': 100, 'fn': 0}, naming only 'B.tpr' as unmeasured,
        # so the FPR arm read as a measurement of 100 rows nobody labelled. A maximal
        # finding of unfairness, asserted about labels that do not exist.
        #
        # IT DEPENDED ON THE STACK, which is the worst property a fabrication can
        # have: on CI's numpy the same call refused, and on this machine it did not,
        # so the library's answer to "is this model fair" varied with the host. A
        # verdict must not.
        #
        # Refused HERE for the same reason y_prob is refused here: every method of
        # this class derives its rates from self.y_true, so one validation above them
        # all closes every one and a guard inside any single method cannot.
        raw_y_true = coerce_to_array(y_true)
        if np.issubdtype(np.asarray(raw_y_true).dtype, np.floating):
            missing = int(np.count_nonzero(~np.isfinite(raw_y_true)))
            if missing:
                raise InvalidDataError(
                    f"y_true has {missing} of {len(raw_y_true)} value(s) that are not a "
                    f"finite label. A row nobody labelled has no true-positive or "
                    f"false-positive rate, so it cannot enter a confusion matrix, a "
                    f"rate, a constraint violation or a threshold verdict. Casting it "
                    f"to an integer invents a label, and what that integer is depends "
                    f"on the platform. Drop the unlabelled rows before constructing "
                    f"this analyzer, or keep them and accept that no threshold verdict "
                    f"covers them."
                )
        self.y_true = raw_y_true.astype(int)
        self.y_prob = coerce_to_array(y_prob)
        self.sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(self.y_true, self.y_prob, self.sensitive_attr)
        # A row with no score cannot be thresholded, so it cannot be a decision,
        # a rate, a confusion-matrix cell or a constraint verdict. See Raises.
        validate_probabilities(self.y_prob, "y_prob")

        # AND WHETHER THE SCORE CAN BE SEARCHED AT ALL (F3-postproc-thr,
        # 2026-09-30). This module already owns the measurement, in
        # ``_record_score_resolution``, and it was wired into the OPTIMISER
        # classes only. The analyzer is the sibling entry point into the same
        # fabrication and carried none of it.
        #
        # Measured on 200 rows in two 100-row groups whose y_prob was the
        # PREDICTION column, hard 0.0/1.0, which is what the platform sends
        # whenever an uploaded dataset has no probability column:
        #   before  find_optimal_threshold answered optimal_threshold 0.01 with
        #           n_thresholds_searched 100, find_feasible_region answered
        #           n_searched 100, and full_analysis recommended "No single
        #           threshold can satisfy demographic_parity. Consider using
        #           group-specific thresholds", with ZERO warnings, while the 100
        #           searched thresholds produced exactly ONE distinct decision
        #           vector between them.
        #   after   one warning naming the 2 distinct values, and every sweep
        #           return carries n_distinct_scores and
        #           score_resolution_sufficient.
        #
        # Recorded rather than refused, which is the policy ``_record_score_resolution``
        # and ``_qualified_feasibility`` argue for in optimizer.py: hard predictions
        # are a legitimate thing to hold and the fairness verdict ON them is a real
        # measurement. What must not happen is the silence about the search.
        #
        # Measured HERE, once, above the dispatch, for the same reason y_prob and
        # y_true are validated here: analyze_threshold, analyze_threshold_range,
        # find_optimal_threshold, find_feasible_region and full_analysis each sweep
        # this column in their own body, and a measurement inside any one of them
        # leaves the others silent.
        self.score_resolution = _record_score_resolution(self.y_prob, caller="ThresholdAnalyzer")

        if sample_weight is not None:
            self.sample_weight = coerce_to_array(sample_weight)
        else:
            self.sample_weight = np.ones(len(self.y_true))

        self.unique_groups = sorted([str(g) for g in np.unique(self.sensitive_attr)])

    def analyze_threshold(
        self,
        threshold: float,
        constraints: Optional[List[str]] = None,
    ) -> ThresholdImpactResult:
        """
        Analyze the impact of a single threshold value.

        Args:
            threshold: Threshold value to analyze
            constraints: List of constraint types to evaluate

        Returns:
            ThresholdImpactResult with detailed metrics

        Raises:
            InvalidDataError: If ``threshold`` is not a finite number.
        """
        # A THRESHOLD THAT IS NOT A NUMBER IS NOT A DECISION RULE
        # (F3-postproc-thr, 2026-09-30). ``self.y_prob >= threshold`` is False for
        # EVERY row when the threshold is NaN, by IEEE 754, exactly as it is for a
        # NaN score: the whole population came back rejected, every group's
        # selection rate was 0.0, and 0.0 == 0.0 is PERFECT PARITY. A neutered
        # threshold reports success, not failure.
        #
        # Measured on 200 rows in two 100-row groups, all scored and all labelled:
        #   before  analyze_threshold(float("nan")) -> 0 of 200 approved,
        #           demographic_parity violation 0.0, is_satisfied True,
        #           group_metrics {'A': 0.0, 'B': 0.0}, unmeasured=() (which
        #           actively asserts nothing was unmeasurable) and no warning about
        #           the threshold. Through analyze_threshold_range(thresholds=[0.5,
        #           nan]) the NaN row read as the FAIREST row of the sweep, 0.0
        #           against 0.14, so a caller ranking rows by violation picks the
        #           threshold that is not a number.
        #   after   InvalidDataError naming the value.
        #
        # Refused here rather than in the sweep methods because this is the one
        # place both analyze_threshold and analyze_threshold_range turn a threshold
        # into decisions, so the guard sits above that dispatch; the grids
        # find_optimal_threshold and find_feasible_region build are finite by
        # construction.
        if not np.isfinite(threshold):
            raise InvalidDataError(
                f"threshold={threshold!r} is not a finite number, so it is not a "
                "decision rule. Every score compares False against a NaN threshold "
                "and every group's selection rate comes out 0.0, which reads as "
                "perfect parity over a population nobody decided. Pass a finite "
                "threshold."
            )

        if constraints is None:
            constraints = ["demographic_parity", "equalized_odds", "equal_opportunity"]

        y_pred = (self.y_prob >= threshold).astype(int)

        performance_metrics = _compute_performance_metrics(self.y_true, y_pred, self.sample_weight)

        confusion_matrix = {
            "tp": int(((self.y_true == 1) & (y_pred == 1)).sum()),
            "tn": int(((self.y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((self.y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((self.y_true == 1) & (y_pred == 0)).sum()),
        }

        group_metrics = {}
        group_confusion_matrices = {}

        for group in self.unique_groups:
            mask = self.sensitive_attr.astype(str) == group
            y_t = self.y_true[mask]
            y_p = y_pred[mask]
            w = self.sample_weight[mask]

            group_metrics[group] = _compute_performance_metrics(y_t, y_p, w)
            group_confusion_matrices[group] = {
                "tp": int(((y_t == 1) & (y_p == 1)).sum()),
                "tn": int(((y_t == 0) & (y_p == 0)).sum()),
                "fp": int(((y_t == 0) & (y_p == 1)).sum()),
                "fn": int(((y_t == 1) & (y_p == 0)).sum()),
            }

        constraint_violations = {}
        for constraint in constraints:
            violation = compute_constraint_violation(
                self.y_true, y_pred, self.sensitive_attr, constraint=constraint, tolerance=0.05
            )
            constraint_violations[constraint] = violation

        return ThresholdImpactResult(
            threshold=threshold,
            performance_metrics=performance_metrics,
            group_metrics=group_metrics,
            constraint_violations=constraint_violations,
            confusion_matrix=confusion_matrix,
            group_confusion_matrices=group_confusion_matrices,
        )

    def analyze_threshold_range(
        self,
        thresholds: Optional[ArrayLike] = None,
        n_thresholds: int = 20,
        constraints: Optional[List[str]] = None,
    ) -> List[ThresholdImpactResult]:
        """
        Analyze a range of threshold values.

        Args:
            thresholds: Specific thresholds to analyze (optional)
            n_thresholds: Number of thresholds if not specified
            constraints: List of constraints to evaluate

        Returns:
            List of ThresholdImpactResult for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, n_thresholds)
        else:
            thresholds = coerce_to_array(thresholds)

        results = []
        for thresh in thresholds:
            result = self.analyze_threshold(thresh, constraints)
            results.append(result)

        return results

    def find_optimal_threshold(
        self,
        constraint: str = "demographic_parity",
        objective: str = "accuracy",
        tolerance: float = 0.05,
        n_thresholds: int = 100,
    ) -> Dict[str, Any]:
        """
        Find the optimal threshold for a given constraint and objective.

        Args:
            constraint: Fairness constraint to satisfy
            objective: Performance metric to optimize
            tolerance: Acceptable constraint violation
            n_thresholds: Number of thresholds to search

        Returns:
            Dictionary with optimal threshold and metrics. ``is_feasible`` has
            THREE states: ``True`` (a searched threshold measurably satisfied
            the constraint), ``False`` (thresholds were measured and none did)
            and ``None`` (COULD NOT CHECK: every searched threshold came back
            unmeasurable, so nothing here says the constraint is or is not
            satisfiable). ``optimal_threshold`` is ``None`` in that third state,
            because the number the search would otherwise hand back is the first
            grid point, chosen by no measurement at all.

        Raises:
            ValueError: If ``objective`` is not one of
                :data:`PERFORMANCE_OBJECTIVES`. Refused rather than substituted:
                optimising a different metric under the requested name returns a
                threshold that is optimal for something the caller did not ask
                for, and says nothing about it.
            ValueError: If ``n_thresholds`` is below
                :data:`~.optimizer.THRESHOLD_GRID_FLOOR`. See
                :func:`~.optimizer._require_searchable_grid`.
        """
        if objective not in PERFORMANCE_OBJECTIVES:
            raise ValueError(
                f"'{objective}' is not a performance objective this analyzer computes. "
                f"Valid objectives: {', '.join(sorted(PERFORMANCE_OBJECTIVES))}. "
                "Refusing rather than optimising a different metric under the "
                "requested name."
            )

        # THE SAME GRID FLOOR THE OPTIMISERS IN THIS PACKAGE ALREADY ENFORCE
        # (F3-postproc-thr, 2026-09-30). ``_require_searchable_grid`` was written
        # for ThresholdOptimizer, GroupThresholdOptimizer and
        # MultiObjectiveThresholdOptimizer and never reached this class, which
        # takes the identical argument and builds the identical linspace.
        #
        # Measured on 200 rows in two fully scored, fully labelled groups:
        #   n_thresholds=1 -> the grid is [0.01], which approves 200 of 200 rows,
        #                     and the answer was optimal_threshold 0.01,
        #                     is_feasible True, n_thresholds_not_assessed 0. A
        #                     search that evaluated one candidate, the
        #                     approve-everybody end, where parity is trivial.
        #   n_thresholds=0 -> the loop never runs at all.
        # Refused rather than disclosed because n_thresholds is the caller's own
        # argument: there is nothing to measure and nothing to degrade to.
        n_thresholds = _require_searchable_grid(
            n_thresholds, caller="ThresholdAnalyzer.find_optimal_threshold"
        )

        thresholds = np.linspace(0.01, 0.99, n_thresholds)

        best_threshold = 0.5
        best_objective = -np.inf
        best_result: Optional[Dict[str, Any]] = None
        feasible_found = False
        # `violation.is_satisfied` is Optional[bool]: None means the constraint
        # COULD NOT BE CHECKED at this threshold (a group rate with an empty
        # denominator). The truthiness test below reads None exactly like False,
        # so a sweep in which nothing was measurable used to return
        # is_feasible=False, a measured-looking negative built from zero
        # measurements, and full_analysis then recommended group-specific
        # thresholds on that basis. Counted here so the third state survives.
        n_not_assessed = 0
        # READINESS-6, 2026-09-10. The OBJECTIVE has the same third state and it
        # was not handled. `_compute_performance_metrics` returns NaN for a
        # metric that is not defined on this data (precision with no positive
        # prediction, recall or f1 with no positive label, balanced_accuracy with
        # no negative label). `obj_value > best_objective` is False for NaN, so:
        # the FIRST feasible threshold set best_objective to NaN, no later
        # threshold could ever beat it, and whatever happened to come first in
        # the sweep was returned as `optimal_threshold`. An arbitrary choice,
        # reported as an optimum, on an objective nobody computed.
        n_objective_unmeasurable = 0
        # HOW MANY DECISIONS THE SWEEP ACTUALLY MADE (F3-postproc-thr, 2026-09-30).
        # `n_thresholds_searched` reads as the number of candidates weighed, and a
        # grid that lies entirely off one end of the score range weighs one: every
        # threshold approves everybody, or rejects everybody, and a constant
        # decision is trivially equal across groups, so the search reports SUCCESS
        # for a threshold that decides nothing. See
        # FeasibleRegion.n_distinct_decisions for the measurement. Counted by
        # approval count, which is exact because `y_prob >= t` is monotone in `t`.
        #
        # Accumulated HERE, immediately after the decision and above every
        # `continue` below, so no branch of the per-threshold dispatch can skip it.
        decision_counts = set()

        for thresh in thresholds:
            y_pred = (self.y_prob >= thresh).astype(int)
            decision_counts.add(int(y_pred.sum()))

            violation = compute_constraint_violation(
                self.y_true, y_pred, self.sensitive_attr, constraint=constraint, tolerance=tolerance
            )

            metrics = _compute_performance_metrics(self.y_true, y_pred, self.sample_weight)
            # Exact subscript. See PERFORMANCE_OBJECTIVES: the defaulted get that
            # used to be here silently optimised accuracy for any unrecognised
            # objective name.
            obj_value = metrics[objective]

            if violation.is_satisfied is None:
                n_not_assessed += 1
                continue

            objective_measured = is_measured(obj_value)
            if violation.is_satisfied and not objective_measured:
                # Feasible, but there is no objective value to rank it by. It is
                # NOT the best, and it is not the worst either.
                n_objective_unmeasurable += 1
                continue

            if violation.is_satisfied:
                if not feasible_found or obj_value > best_objective:
                    best_threshold = thresh
                    best_objective = obj_value
                    best_result = {
                        "threshold": thresh,
                        "metrics": metrics,
                        "violation": violation,
                    }
                    feasible_found = True
            elif not feasible_found:
                if (
                    best_result is None
                    or violation.violation
                    < best_result.get(
                        "violation",
                        ConstraintViolation(
                            constraint_type=FairnessConstraintType(constraint),
                            violation=float("inf"),
                            group_metrics={},
                            group_violations={},
                            is_satisfied=False,
                            tolerance=tolerance,
                        ),
                    ).violation
                ):
                    best_threshold = thresh
                    best_result = {
                        "threshold": thresh,
                        "metrics": metrics,
                        "violation": violation,
                    }

        n_searched = len(thresholds)
        n_distinct_decisions = len(decision_counts)
        if not feasible_found and n_objective_unmeasurable:
            # Feasible thresholds EXISTED and not one of them could be ranked.
            #
            # `is_feasible` here is TRUE and measured: the constraint is
            # satisfiable on this data, at n_objective_unmeasurable of the
            # searched thresholds. What could not be established is WHICH of them
            # is best, because the objective has no value on this data. Reporting
            # is_feasible=False, which an earlier version of this very fix did,
            # is a fabricated negative built from the objective's silence: the
            # first run of it said infeasible over 90 feasible thresholds.
            warnings.warn(
                f"find_optimal_threshold: {n_objective_unmeasurable} of {n_searched} "
                f"thresholds SATISFIED {constraint}, but {objective!r} is not defined on "
                f"this data at any of them, so none could be ranked. is_feasible is True "
                f"and measured; optimal_threshold is None (COULD NOT CHECK), which is not "
                f"a claim that no good threshold exists. Choose an objective that is "
                f"computable on this data, such as accuracy.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "optimal_threshold": None,
                "is_feasible": True,
                "threshold": None,
                "metrics": {},
                "violation": None,
                "n_thresholds_searched": n_searched,
                "n_thresholds_not_assessed": n_not_assessed,
                "n_thresholds_objective_unmeasurable": n_objective_unmeasurable,
                # On EVERY return path, for the reason spelled out at the last
                # one: n_thresholds_searched is only honest beside how many
                # decisions those thresholds actually made.
                "n_distinct_decisions": n_distinct_decisions,
                **self.score_resolution,
            }

        if best_result is None:
            # Every searched threshold was could-not-check, so there is no
            # measurement to report and no verdict to give. THREE STATES: this
            # is not "infeasible".
            warnings.warn(
                f"find_optimal_threshold: none of the {n_searched} searched "
                f"thresholds could be checked for {constraint} (no measurable "
                "group rate at any of them). is_feasible is None (COULD NOT "
                "CHECK) and optimal_threshold is None, which is neither a "
                "feasible nor an infeasible finding.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "optimal_threshold": None,
                "is_feasible": None,
                "threshold": None,
                "metrics": {},
                "violation": None,
                "n_thresholds_searched": n_searched,
                "n_thresholds_not_assessed": n_not_assessed,
                "n_thresholds_objective_unmeasurable": n_objective_unmeasurable,
                "n_distinct_decisions": n_distinct_decisions,
                **self.score_resolution,
            }

        if n_objective_unmeasurable:
            warnings.warn(
                f"find_optimal_threshold: {objective!r} was not defined at "
                f"{n_objective_unmeasurable} of {n_searched} searched thresholds, so "
                f"those were excluded from the ranking. The optimum reported rests on "
                f"the thresholds where it WAS defined, not on the whole sweep.",
                UserWarning,
                stacklevel=2,
            )

        if n_not_assessed:
            warnings.warn(
                f"find_optimal_threshold: {n_not_assessed} of {n_searched} searched "
                f"thresholds could not be checked for {constraint} and were skipped; "
                f"is_feasible={feasible_found} rests on the "
                f"{n_searched - n_not_assessed} that were measured.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "optimal_threshold": best_threshold,
            "is_feasible": feasible_found,
            "n_thresholds_searched": n_searched,
            "n_thresholds_not_assessed": n_not_assessed,
            # Always present, on every return path. A key that appears only when
            # something went wrong forces every reader to use `.get(k, 0)`, and a
            # defaulted get over a count of what could not be measured is the
            # very shape this audit exists to remove.
            "n_thresholds_objective_unmeasurable": n_objective_unmeasurable,
            # n_distinct_decisions, n_distinct_scores and
            # score_resolution_sufficient, by the same rule.
            # n_thresholds_searched above says 100 for a hard 0/1 prediction column
            # over which all 100 candidates make ONE decision, and the threshold
            # returned is then the grid's lower bound chosen by nothing. The
            # number is kept rather than blanked, because it does reproduce the
            # metrics reported beside it and because the verdict on those
            # predictions is a real measurement of them (see
            # _qualified_feasibility in optimizer.py, which makes the same call
            # deliberately); what is added is the disclosure that there was
            # nothing to search.
            "n_distinct_decisions": n_distinct_decisions,
            **self.score_resolution,
            **best_result,
        }

    def find_feasible_region(
        self,
        constraint: str = "demographic_parity",
        tolerance: float = 0.05,
        n_thresholds: int = 100,
    ) -> FeasibleRegion:
        """
        Find the range of thresholds that satisfy a constraint.

        Args:
            constraint: Fairness constraint
            tolerance: Acceptable violation
            n_thresholds: Number of thresholds to search

        Returns:
            :class:`FeasibleRegion`, a ``(min_threshold, max_threshold)`` pair
            that is ``(None, None)`` when no searched threshold satisfies the
            constraint. Its ``assessed`` flag separates that MEASURED
            infeasibility from a sweep in which every threshold came back
            could-not-check, which the pair alone cannot express, and its
            ``contiguous`` flag says whether the pair is a true interval or the
            two ends of a satisfying set with gaps in it.

        Raises:
            ValueError: If ``n_thresholds`` is below
                :data:`~.optimizer.THRESHOLD_GRID_FLOOR`. See
                :func:`~.optimizer._require_searchable_grid`.
        """
        # THE GRID FLOOR, ABOVE THE BRANCH SELECTION (F3-postproc-thr,
        # 2026-09-30). Every one of the three returns below described a sweep that
        # had not happened, and each in a different direction, so the guard cannot
        # live in any one of them:
        #   n_thresholds=0 -> the loop never runs, `feasible` is empty and
        #                     `n_not_assessed == n_searched` is 0 == 0, which the
        #                     `and n_searched > 0` clause excludes, so it fell
        #                     through to the LAST return: assessed=True,
        #                     status "infeasible", n_searched 0. A MEASURED
        #                     finding of infeasibility from zero measurements,
        #                     with no warning, and _generate_recommendations then
        #                     printed "No single threshold can satisfy
        #                     demographic_parity. Consider using group-specific
        #                     thresholds", a recommendation to change the model's
        #                     decision rule.
        #   n_thresholds=1 -> the grid is [0.01], which approves every row, so
        #                     every group's selection rate is 1.0 and parity is
        #                     trivially satisfied: the FIRST return, status
        #                     "feasible", region (0.01, 0.01), n_feasible 1 of 1,
        #                     contiguous True. Approving everybody is equal; it is
        #                     not a fairness finding.
        # Measured 2026-09-30 on 200 rows in two fully scored, fully labelled
        # groups. Refused rather than disclosed, as in find_optimal_threshold.
        n_thresholds = _require_searchable_grid(
            n_thresholds, caller="ThresholdAnalyzer.find_feasible_region"
        )

        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        feasible = []
        feasible_indices = []
        # `is_satisfied` is None for a threshold whose constraint could not be
        # evaluated, and `if violation.is_satisfied` reads that exactly like a
        # measured False. Counted rather than swallowed.
        n_not_assessed = 0
        # See FeasibleRegion.n_distinct_decisions. Accumulated above the
        # per-threshold branch selection, so every branch feeds it.
        decision_counts = set()

        for index, thresh in enumerate(thresholds):
            y_pred = (self.y_prob >= thresh).astype(int)
            decision_counts.add(int(y_pred.sum()))
            violation = compute_constraint_violation(
                self.y_true, y_pred, self.sensitive_attr, constraint=constraint, tolerance=tolerance
            )
            if violation.is_satisfied is None:
                n_not_assessed += 1
            elif violation.is_satisfied:
                feasible.append(thresh)
                feasible_indices.append(index)

        n_searched = len(thresholds)
        n_distinct_decisions = len(decision_counts)
        if feasible:
            # A min and a max READ as an interval, and the satisfying set does not
            # have to be one. See FeasibleRegion.contiguous for the measurement:
            # 12 of 100 thresholds satisfied demographic_parity, at the two
            # degenerate ends of the grid, and the pair (0.01, 0.99) presented the
            # 88 measured violations between them as inside the feasible region.
            contiguous = feasible_indices == list(
                range(feasible_indices[0], feasible_indices[-1] + 1)
            )
            if not contiguous:
                warnings.warn(
                    f"find_feasible_region: only {len(feasible)} of {n_searched} searched "
                    f"thresholds satisfy {constraint}, and they do NOT form one run, so "
                    f"the returned pair ({float(min(feasible)):.3f}, "
                    f"{float(max(feasible)):.3f}) is the min and max of a set with gaps, "
                    "not an interval. Thresholds between them were MEASURED as violating; "
                    "read region.contiguous before treating the pair as a usable range.",
                    UserWarning,
                    stacklevel=2,
                )
            return FeasibleRegion(
                float(min(feasible)),
                float(max(feasible)),
                assessed=True,
                n_not_assessed=n_not_assessed,
                n_searched=n_searched,
                n_feasible=len(feasible),
                contiguous=contiguous,
                # On all three returns: a region is read as the result of a
                # search, and how much there was to search is part of it.
                n_distinct_decisions=n_distinct_decisions,
                **self.score_resolution,
            )

        if n_not_assessed == n_searched and n_searched > 0:
            warnings.warn(
                f"find_feasible_region: none of the {n_searched} searched thresholds "
                f"could be checked for {constraint} (no measurable group rate at any "
                "of them), so the empty region is NOT ASSESSED rather than a finding "
                "of infeasibility.",
                UserWarning,
                stacklevel=2,
            )
            return FeasibleRegion(
                None,
                None,
                assessed=False,
                n_not_assessed=n_not_assessed,
                n_searched=n_searched,
                n_distinct_decisions=n_distinct_decisions,
                **self.score_resolution,
            )

        return FeasibleRegion(
            None,
            None,
            assessed=True,
            n_not_assessed=n_not_assessed,
            n_searched=n_searched,
            n_distinct_decisions=n_distinct_decisions,
            **self.score_resolution,
        )

    def full_analysis(
        self,
        n_thresholds: int = 20,
        constraints: Optional[List[str]] = None,
        tolerance: float = 0.05,
    ) -> ThresholdAnalysisReport:
        """
        Perform comprehensive threshold analysis.

        Args:
            n_thresholds: Number of thresholds to analyze
            constraints: List of constraints to evaluate
            tolerance: Acceptable constraint violation

        Returns:
            ThresholdAnalysisReport with complete analysis
        """
        if constraints is None:
            constraints = ["demographic_parity", "equalized_odds", "equal_opportunity"]

        threshold_results = self.analyze_threshold_range(
            n_thresholds=n_thresholds, constraints=constraints
        )

        optimal_thresholds: Dict[str, Dict[str, Optional[float]]] = {}
        for constraint in constraints:
            optimal_thresholds[constraint] = {}
            for objective in ["accuracy", "f1_score", "balanced_accuracy"]:
                result = self.find_optimal_threshold(
                    constraint=constraint, objective=objective, tolerance=tolerance
                )
                # None when nothing could be checked; carried through rather
                # than replaced with the first grid point.
                optimal_thresholds[constraint][objective] = result["optimal_threshold"]

        feasible_regions: Dict[str, FeasibleRegion] = {}
        for constraint in constraints:
            feasible_regions[constraint] = self.find_feasible_region(
                constraint=constraint, tolerance=tolerance
            )

        recommendations = self._generate_recommendations(
            threshold_results, optimal_thresholds, feasible_regions, constraints
        )

        return ThresholdAnalysisReport(
            threshold_results=threshold_results,
            optimal_thresholds=optimal_thresholds,
            feasible_regions=feasible_regions,
            recommendations=recommendations,
            metadata={
                "n_samples": len(self.y_true),
                "n_groups": len(self.unique_groups),
                "groups": self.unique_groups,
                "tolerance": tolerance,
            },
        )

    def _generate_recommendations(
        self,
        threshold_results: List[ThresholdImpactResult],
        optimal_thresholds: Dict,
        feasible_regions: Dict,
        constraints: List[str],
    ) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []

        # THREE STATES. "No single threshold can satisfy X, consider using
        # group-specific thresholds" is a recommendation to change the model's
        # decision rule, and it used to be issued for any empty region, whether
        # or not a single threshold had actually been measured. A constraint
        # whose every threshold was could-not-check gets its own line saying so.
        def _assessed(constraint: str) -> bool:
            region = feasible_regions[constraint]
            # Fail CLOSED on an unrecognised shape. This branch is defensive:
            # feasible_regions is typed Dict[str, FeasibleRegion] and every entry
            # comes from find_feasible_region, so a plain tuple should be
            # unreachable. But the default decides whether a (None, None) is
            # reported as MEASURED INFEASIBLE or as NOT ASSESSED, and `True`
            # asserted the measurement for a shape carrying no such information,
            # which is the exact ambiguity the FeasibleRegion type was added to
            # remove. An unknown shape is could-not-check.
            return region.assessed if isinstance(region, FeasibleRegion) else False

        feasible_constraints = [c for c in constraints if feasible_regions[c][0] is not None]
        infeasible_constraints = [
            c for c in constraints if feasible_regions[c][0] is None and _assessed(c)
        ]
        not_assessed_constraints = [
            c for c in constraints if feasible_regions[c][0] is None and not _assessed(c)
        ]

        if infeasible_constraints:
            recommendations.append(
                f"No single threshold can satisfy: {', '.join(infeasible_constraints)}. "
                "Consider using group-specific thresholds."
            )

        # A SWEEP THAT MADE ONE DECISION (F3-postproc-thr, 2026-09-30). The three
        # lists above are about the CONSTRAINT verdict; this is about the search
        # that produced it. Every searched threshold can make the identical
        # decision, and then "No single threshold can satisfy X. Consider using
        # group-specific thresholds" and "Multiple constraints can be satisfied"
        # are both reports on one decision rule dressed as a hundred point search.
        # Measured 2026-09-30, twice: on a hard 0/1 PREDICTION column, which is
        # what the platform sends whenever an uploaded dataset has no probability
        # column, and on scores that all sit above 0.99, where all 100 thresholds
        # approve everybody and 100 of 100 "satisfy" demographic_parity. Neither
        # produced a warning or a caveat anywhere in the report.
        #
        # An ADDITIONAL line, not a replacement, and the verdict above is left
        # alone: the constraint measurement on that one decision is real, and
        # withdrawing it would be the over-correction. What was missing is that the
        # sweep never varied anything.
        # `is not None` rather than a falsy test: 0 decisions and None (nothing
        # recorded) are different states and `or 0` would collapse them.
        def _one_decision(constraint: str) -> bool:
            n = getattr(feasible_regions[constraint], "n_distinct_decisions", None)
            return n is not None and n <= 1

        one_decision = [c for c in constraints if _one_decision(c)]
        if one_decision:
            region = feasible_regions[one_decision[0]]
            recommendations.append(
                f"NOT A SEARCH: all {region.n_searched} searched thresholds made the SAME "
                f"decision, so the region(s) reported for {', '.join(one_decision)} rest on "
                "one decision rule rather than on a search over thresholds. A cut that "
                "puts every row on one side is trivially equal across groups, so a "
                "satisfied constraint here is a property of that cut and not a finding "
                "about the model. Check that the searched range covers the scores, and "
                "pass predicted probabilities rather than hard 0/1 predictions."
            )

        if not_assessed_constraints:
            recommendations.append(
                f"NOT ASSESSED: {', '.join(not_assessed_constraints)} could not be "
                "evaluated at any searched threshold (no measurable group rate), so "
                "this analysis says nothing about whether a threshold satisfying "
                "them exists. Collect data for the unmeasurable group(s) before "
                "reading the result as a finding either way."
            )

        if feasible_constraints:
            regions = {c: feasible_regions[c] for c in feasible_constraints}
            widths = {c: r[1] - r[0] for c, r in regions.items()}
            narrowest = min(widths.keys(), key=lambda c: widths[c])

            if widths[narrowest] < 0.1:
                recommendations.append(
                    f"The feasible region for {narrowest} is narrow "
                    f"({widths[narrowest]:.2f}). Small changes in data may "
                    "make the constraint infeasible."
                )

            # A wide span is not a wide region when the satisfying set has gaps,
            # and the width above cannot tell the difference: it is end minus end.
            # The measured case is two clusters at the degenerate ends of the grid,
            # where the span is 0.98 and no single usable run is wider than 0.05.
            gapped = [
                c for c, r in regions.items() if isinstance(r, FeasibleRegion) and not r.contiguous
            ]
            for constraint in gapped:
                region = regions[constraint]
                recommendations.append(
                    f"The thresholds satisfying {constraint} are NOT one range: "
                    f"{region.n_feasible} of {region.n_searched} searched thresholds "
                    f"satisfy it and they fall in separate clusters between "
                    f"{region[0]:.3f} and {region[1]:.3f}. Picking a value inside that "
                    "span, its midpoint included, can land on a threshold measured as "
                    "violating, so choose a satisfying threshold rather than a range."
                )

        if len(feasible_constraints) >= 2:
            for c1 in feasible_constraints:
                for c2 in feasible_constraints:
                    if c1 < c2:
                        t1 = optimal_thresholds[c1]["accuracy"]
                        t2 = optimal_thresholds[c2]["accuracy"]
                        # A feasible region implies a measured optimum, so both
                        # are floats here; the guard keeps a None out of the
                        # subtraction rather than reporting a difference of one.
                        if t1 is None or t2 is None:
                            continue
                        if abs(t1 - t2) > 0.15:
                            recommendations.append(
                                f"Optimal thresholds for {c1} ({t1:.2f}) and "
                                f"{c2} ({t2:.2f}) differ significantly. "
                                "You may need to prioritize one constraint."
                            )

        if not recommendations:
            if feasible_constraints:
                recommendations.append(
                    "Multiple constraints can be satisfied. Consider the specific "
                    "requirements of your use case when selecting a threshold."
                )
            else:
                # The three lists above partition `constraints`, so all of them
                # empty means the list was empty: nothing was examined. That
                # produced the same closing sentence as a sweep in which every
                # constraint was satisfiable. Measured 2026-09-27 with
                # full_analysis(constraints=[]): optimal_thresholds {},
                # feasible_regions {}, and the one recommendation read "Multiple
                # constraints can be satisfied."
                recommendations.append(
                    "NOT ASSESSED: no fairness constraint was evaluated, so this report "
                    "says nothing about whether any threshold satisfies one. Pass the "
                    "constraints you need checked."
                )

        return recommendations

    def get_explanation(self, report=None):
        """Generate educational explanations for the threshold analysis.

        Parameters
        ----------
        report : ThresholdAnalysisReport, optional
            Pre-computed report. If *None*, :meth:`full_analysis` is run.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_analysis()
        return FairnessExplainer.explain(report)
