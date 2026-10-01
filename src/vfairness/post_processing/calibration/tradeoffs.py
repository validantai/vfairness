"""
Calibration-Fairness Trade-off Analysis for vfairness.

This module provides tools for analyzing the fundamental trade-offs between
calibration and other fairness criteria, helping practitioners understand
what's mathematically possible and make principled decisions.

Key Concepts:
    - Impossibility Theorem: Calibration, balance for positive class, and
      balance for negative class cannot be simultaneously satisfied
      (Kleinberg et al., 2016)
    - Pareto Frontier: Set of non-dominated solutions in the trade-off space
    - Multi-objective Optimization: Balancing calibration against error parity

When to Prioritize Calibration:
    - Risk scores directly inform decisions (lending, healthcare)
    - Different thresholds may be applied by different users
    - Consistent interpretation is ethically paramount
    - Legal requirements mandate calibration across groups

When to Prioritize Other Fairness Criteria:
    - Binary decisions with fixed thresholds
    - Historical discrimination created error imbalances
    - Stakeholders explicitly prioritize error rate parity

References:
    - Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.
    - Chouldechova, A. (2017). Fair Prediction with Disparate Impact.
    - Corbett-Davies, S. & Goel, S. (2018). The Measure and Mismeasure of Fairness.
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_probabilities,
)

# Fewest LABELLED observations in a group before a per-group error rate is a rate.
#
# WHY IT IS A NAMED CONSTANT AND WRITTEN ONCE. BGL5 wave 4 (2026-09-30) overturned
# three grades in this file for the same reason: the group gate counts ROWS and a
# TPR/FPR/FNR needs LABELS OF ONE CLASS, so the rate arms kept a floor of ONE.
# Every rate site here guarded only ``np.sum(mask) > 0``, and the only test above
# the spread was a COUNT of rates ("at least two of them exist"), never a test that
# either rate had the evidence to be one. A rate from a single observation can take
# exactly two values, 0.0 and 1.0, which are the extremes of the scale, so it does
# not merely add noise: it decides the comparison. Measured on two 120-row groups
# where B held exactly ONE positive label, tuned so every other arm stayed inside
# the 0.05 tolerance:
#
#   that one row scored 0.018  ->  tpr {'A': 0.0, 'B': 0.0}, tpr_disparity 0.0,
#                                  conflict_exists FALSE (a clean bill)
#   the SAME data, row at 0.9  ->  tpr {'A': 0.0, 'B': 1.0}, tpr_disparity 1.0,
#                                  conflict_exists TRUE
#
# with not_assessed [], excluded_groups [], n_samples 120 for B (its ROW count) and
# no warning in either direction. One labelled row decided the headline verdict.
#
# TEN is the number this repository already uses for "fewest observations before a
# per-group error means anything" (``_not_assessed.MIN_ROWS_PER_GROUP_FOR_CALIBRATION``,
# whose own comment records what a one-observation per-group figure produced on a
# chart). It is NOT imported from there because that constant counts ROWS for a
# calibration curve and this one counts LABELS OF ONE CLASS for a rate: they answer
# different questions and must be movable independently. It IS a single constant
# across this module, because the cali-BRATIO-n incident (CLAUDE.md) and the
# 10-rows-here-1-row-there disagreement recorded in ``_not_assessed`` are both the
# same failure: one rule written twice, and only one copy raised.
_MIN_LABELS_PER_RATE = 10


# The two label vocabularies for a text/object ground-truth column, and why there
# has to be a NEGATIVE one.
#
# BGL5 wave 4 (2026-09-30). ``mitigation_pareto`` resolved an object label column
# with ``.isin(<positives>).astype(int)``, so ANYTHING not in the positive set
# became a 0, a ground-truth NEGATIVE. A missing label is not a negative, and the
# unscored/unlabelled filter could not see it: ``_isnan`` returns all-False for a
# non-float array, which the source said out loud ("the object/string path below is
# untouched by this filter"). MEASURED on identical data, two 100-row groups both
# fully scored, group A genuinely labelled and group B not labelled at all, varying
# ONLY how the missing label is spelled:
#
#   FLOAT labels, B NaN        -> available False, nRowsUnlabelled 100, refused
#   OBJECT labels, B 'unknown' -> available TRUE, nRowsUnlabelled 0, baselineGap
#                                 1.0, bestGap 0.0, summary "... at 0.0 points of
#                                 accuracy cost", warnings []
#   CONTROL, B really labelled -> the same gaps, accuracy cost -50.0 points
#
# The 0.0-point cost is the fabrication: 100 rows nobody labelled were counted as
# correctly rejected, and the published cost of the mitigation differed by FIFTY
# accuracy points from the same frame with real labels. The surface is the summary
# sentence a person reads.
#
# A value in NEITHER vocabulary is UNLABELLED, never a negative. That is the whole
# point of listing negatives: with only a positive list, "unknown", "", "n/a",
# "pending" and the literal "None" are all silently negatives. The refusal names
# the unrecognised spellings so a caller whose column uses a word that is not here
# can see why, rather than meeting an opaque refusal.
_POSITIVE_LABEL_WORDS = frozenset(
    {"1", "yes", "true", "y", "hired", "approved", "selected", "positive"}
)
_NEGATIVE_LABEL_WORDS = frozenset(
    {
        "0",
        "no",
        "false",
        "n",
        "not hired",
        "rejected",
        "denied",
        "declined",
        "not selected",
        "unselected",
        "negative",
    }
)


def _measured_spread(values: Mapping[Any, float]) -> Tuple[float, List[str], List[str]]:
    """``max - min`` over the values that were actually MEASURED, and who they are.

    Builtin ``max`` and ``min`` do NOT propagate a NaN that is not the first
    element of the iterable, because every comparison against a NaN is False and
    they keep the running extreme. Measured 2026-09-27:

        max({'A': 0.5167, 'B': nan}.values())  ->  0.5167
        min({'A': 0.5167, 'B': nan}.values())  ->  0.5167
        the spread                             ->  0.0, and math.isfinite is True

    So a between-group disparity in which ONE of the two groups was never
    measured came back as a FINITE 0.0, which is PERFECT PARITY on every scale in
    this module, and it walked past every ``math.isfinite`` guard the earlier
    fixes added, because the value those guards inspect is already finite. This
    file held four such sites. Measured 2026-09-27 on 240 rows in two 120-row
    groups where group B carried no label at all, so its base rate is NaN:

        before  base_rate_disparity 0.0, tradeoff_severity "minimal",
                impossibility_applies False, base_rates_differ False,
                calibration_gap_disparity 0.0, conflict_exists False,
                ece_disparity 0.0, strategy "monitor_only" priority "low",
                first recommendation "GOOD NEWS: Base rates are similar across
                groups", and ZERO warnings about any of it.
        after   base_rate_disparity nan, tradeoff_severity "not assessed",
                impossibility_applies None, base_rates_differ None,
                calibration_gap_disparity nan, conflict_exists None,
                disparity_assessed False, a NOT ASSESSED recommendation instead
                of the GOOD NEWS one, and a warning naming group B.

    The repository already pins this exact mechanism in
    ``tests/test_readiness6_names.py::test_python_max_really_does_swallow_nan``.

    Returns
    -------
    (spread, measured_names, unmeasured_names)
        ``spread`` is ``max - min`` over the finite values and NaN when fewer
        than two of them are finite, because a BETWEEN-group disparity needs two
        measured values to be between. ``measured_names`` and
        ``unmeasured_names`` are sorted, so a caller can name the gap in a
        warning or a not-assessed list rather than reporting a number for it.
    """
    measured: Dict[str, float] = {}
    unmeasured: List[str] = []
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if math.isfinite(number):
            measured[str(name)] = number
        else:
            unmeasured.append(str(name))
    if len(measured) < 2:
        return float("nan"), sorted(measured), sorted(unmeasured)
    return max(measured.values()) - min(measured.values()), sorted(measured), sorted(unmeasured)


@dataclass
class ParetoPoint:
    """
    A point on the Pareto frontier.

    Attributes:
        calibration_error: ECE or similar calibration metric
        fairness_violation: Error rate disparity or similar
        threshold: Decision threshold used
        parameters: Additional parameters that produced this point
        is_pareto_optimal: THREE states. ``True`` when a dominance comparison
            ranked this point and nothing dominates it, ``False`` when a
            comparison ranked it and something does, ``None`` when it was not
            ranked at all: either no frontier has been computed yet, or one of
            its two coordinates is not a finite number, so no comparison against
            it can be decided. The default was ``True``, so every freshly built
            point asserted Pareto optimality before anything had been compared,
            and ``compute_pareto_frontier`` only ever wrote ``True``, which left
            that assertion standing on the points it had just found dominated.
    """

    calibration_error: float
    fairness_violation: float
    threshold: float = 0.5
    parameters: Dict = field(default_factory=dict)
    is_pareto_optimal: Optional[bool] = None

    def dominates(self, other: "ParetoPoint") -> Optional[bool]:
        """Whether this point dominates another (both metrics better).

        THREE states, matching ``is_satisfied`` elsewhere in this package.
        ``None`` means the comparison COULD NOT BE MADE, because at least one of
        the four coordinates involved is not a finite number.

        Every comparison against a NaN is False, so this used to answer a plain
        ``False`` in BOTH directions for an unmeasured point. In
        ``compute_pareto_frontier`` "nothing dominates it" is what puts a point
        ON the frontier, so a point whose fairness violation was never measured
        came back flagged Pareto optimal. Measured 2026-09-27 on 240 rows where
        group B held no negative labels, so fpr_parity had exactly one measurable
        rate: all 17 swept thresholds carried fairness_violation NaN and all 17
        were returned as the frontier with is_pareto_optimal True, no warning.
        The same defect was fixed in the sibling frontier in
        ``operations/experimentation/analysis.py`` (audit 6) and in
        ``MultiObjectiveThresholdOptimizer`` (surface grade g017).

        ``None`` is falsy, so ``if a.dominates(b):`` keeps its old behaviour for
        callers that only branch on it.
        """
        coordinates = (
            self.calibration_error,
            self.fairness_violation,
            other.calibration_error,
            other.fairness_violation,
        )
        if not all(math.isfinite(c) for c in coordinates):
            return None
        # bool(), because these coordinates are usually np.float64 and the `and`
        # chain then evaluates to np.bool_, which `is True` REJECTS. The first
        # version of the frontier below tested `dominates(...) is True` and that
        # silently made every measured point non-dominated: a 17-threshold sweep
        # whose frontier is 3 points came back with all 17. It passed on
        # hand-built Python floats and failed on the real path, which is the
        # whole reason the healthy control is run.
        return bool(
            self.calibration_error <= other.calibration_error
            and self.fairness_violation <= other.fairness_violation
            and (
                self.calibration_error < other.calibration_error
                or self.fairness_violation < other.fairness_violation
            )
        )

    @property
    def is_rankable(self) -> bool:
        """Whether both coordinates are finite, so a dominance test can decide."""
        return math.isfinite(self.calibration_error) and math.isfinite(self.fairness_violation)


@dataclass
class TradeoffAnalysisResult:
    """
    Results of calibration-fairness trade-off analysis.

    Attributes:
        pareto_points: Points on the Pareto frontier
        current_point: Current model's position in trade-off space
        impossibility_diagnosis: Details about the impossibility theorem
        recommendations: Suggested actions based on analysis
        base_rate_disparity: Difference in base rates between groups, or NaN
            when fewer than two groups were large enough to compare and no
            difference was measured
        metrics_at_thresholds: Metrics computed at various thresholds
        rate_label_counts: How many labels of the class the chosen
            ``fairness_metric`` needs each compared group actually carries, so a
            reader can see the DENOMINATOR the rate was built from. Empty for
            ``demographic_parity``, which is a rate over predictions and needs no
            labels. A warning alone is not a disclosure: it is discarded by any
            caller that does not capture warnings, so the counts travel on the
            result.
        groups_without_measured_rate: Groups above ``min_group_size`` that carry
            too few labels of that class for a rate to be a rate, so they
            contribute NOTHING to ``fairness_violation`` at any threshold. This is
            not a finding that their rate matches.
    """

    pareto_points: List[ParetoPoint]
    current_point: ParetoPoint
    impossibility_diagnosis: Dict[str, Any]
    recommendations: List[str]
    base_rate_disparity: float
    metrics_at_thresholds: Dict[float, Dict[str, float]] = field(default_factory=dict)
    rate_label_counts: Dict[str, int] = field(default_factory=dict)
    groups_without_measured_rate: List[str] = field(default_factory=list)

    @property
    def tradeoff_severity(self) -> str:
        """Classify severity of the trade-off.

        Three states. ``"not assessed"`` is returned when
        ``base_rate_disparity`` is not a finite number, which is what a
        comparison that could not be made looks like. It used to be reachable
        only as ``"minimal"``, because the fewer-than-two-groups path filled the
        disparity in with 0.0 and ``0.0 < 0.05``: an unmeasured comparison
        graded as the best possible measured one.
        """
        if not math.isfinite(self.base_rate_disparity):
            return "not assessed"
        if self.base_rate_disparity < 0.05:
            return "minimal"
        elif self.base_rate_disparity < 0.15:
            return "moderate"
        else:
            return "severe"

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "pareto_points": [
                {
                    "calibration_error": p.calibration_error,
                    "fairness_violation": p.fairness_violation,
                    "threshold": p.threshold,
                }
                for p in self.pareto_points
            ],
            "current_point": {
                "calibration_error": self.current_point.calibration_error,
                "fairness_violation": self.current_point.fairness_violation,
            },
            "impossibility_diagnosis": self.impossibility_diagnosis,
            "recommendations": self.recommendations,
            "base_rate_disparity": self.base_rate_disparity,
            "tradeoff_severity": self.tradeoff_severity,
            # Always present, on every path. A key that appears only when
            # something went wrong forces every reader into `.get(k, <neutral>)`,
            # and a defaulted read of what could not be measured is the shape this
            # audit exists to remove.
            "rate_label_counts": dict(self.rate_label_counts),
            "groups_without_measured_rate": list(self.groups_without_measured_rate),
        }


@dataclass
class CalibrationRecommendation:
    """
    Recommendation for calibration strategy.

    Attributes:
        strategy: Recommended calibration approach
        priority: Priority level (high, medium, low)
        rationale: Explanation for recommendation
        expected_improvement: Expected ECE improvement
        tradeoff_warning: Any fairness trade-offs to consider
    """

    strategy: str
    priority: Literal["high", "medium", "low"]
    rationale: str
    expected_improvement: Optional[float] = None
    tradeoff_warning: Optional[str] = None
    #: Groups excluded from the disparity the recommendation is based on, so a
    #: reader can tell "no disparity found" from "the disparity does not cover
    #: these groups". H-06: `calibration_disparity` reported these correctly and
    #: this function discarded them, keeping only the resulting 0.0.
    not_assessed_groups: List[str] = field(default_factory=list)
    #: Whether the BETWEEN-group calibration disparity this recommendation rests
    #: on was measured at all. False when `ece_disparity` was not a finite
    #: number, which happens when fewer than two groups clear the size gate.
    #: H-06 covered the case where a group was EXCLUDED; this covers the case
    #: where the comparison never existed, and every `ece_disparity > x` test in
    #: the decision function is False for a NaN, so it fell through to
    #: "Calibration metrics are acceptable" with priority "low".
    disparity_assessed: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary representation.

        `not_assessed_groups` is carried, and that is the whole reason the field
        exists. It was added for H-06, where this analysis discarded the groups it
        could not assess and kept only the resulting 0.0, so a reader could not tell
        "no disparity found" from "the disparity does not cover these groups". The
        field fixed it on the object and this serialiser dropped it again, which put
        the same defect back at the boundary a consumer actually reads: anything
        working from to_dict, a dashboard or a stored record, saw the recommendation
        and never the caveat that qualifies it.
        """
        return {
            "strategy": self.strategy,
            "priority": self.priority,
            "rationale": self.rationale,
            "expected_improvement": self.expected_improvement,
            "tradeoff_warning": self.tradeoff_warning,
            "not_assessed_groups": list(self.not_assessed_groups),
            "disparity_assessed": self.disparity_assessed,
        }


def analyze_calibration_fairness_tradeoff(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    thresholds: Optional[List[float]] = None,
    fairness_metric: Literal["fpr_parity", "fnr_parity", "demographic_parity"] = "fpr_parity",
    n_bins: int = 10,
    min_group_size: int = 30,
) -> TradeoffAnalysisResult:
    """
    Analyze trade-offs between calibration and other fairness criteria.

    This function computes the Pareto frontier showing achievable combinations
    of calibration quality and fairness, helping practitioners understand
    what trade-offs are necessary.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        thresholds: Decision thresholds to evaluate (default: 0.1 to 0.9)
        fairness_metric: Which fairness metric to analyze
        n_bins: Number of bins for ECE computation
        min_group_size: Minimum samples per group

    Returns:
        TradeoffAnalysisResult with comprehensive analysis

    Example:
        >>> result = analyze_calibration_fairness_tradeoff(
        ...     y_true, y_prob, gender, fairness_metric='fpr_parity'
        ... )
        >>> print(f"Trade-off severity: {result.tradeoff_severity}")
        >>> for rec in result.recommendations:
        ...     print(f"  - {rec}")

    References:
        Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob, protected_attr)
    validate_probabilities(y_prob, "y_prob")

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17).tolist()

    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    valid_groups = gm.get_valid_groups()

    if len(valid_groups) < 2:
        # The trade-off is a statement ABOUT A COMPARISON between groups, and
        # there is nothing to compare. This used to return neutral-looking
        # MEASUREMENTS for it: base_rate_disparity=0.0, current_point
        # ParetoPoint(0, 0) and an empty impossibility_diagnosis. Measured
        # 2026-09-10 on 200 rows of one group, that put three all-clears in the
        # same reader-facing block, `tradeoff_severity` "minimal", disparity
        # "0.0%" and `impossibility_applies` absent so every `.get(..., False)`
        # reader saw False, a few lines above a recommendation admitting nothing
        # had been analysed.
        #
        # `impossibility_diagnostics` already computes the honest NOT ASSESSED
        # verdict for exactly this condition, so it is CALLED here rather than
        # answered a second time: two functions answering the same question
        # differently is how this started. It emits its own could-not-check
        # warning, which is why there is only one warning below.
        warnings.warn(
            "analyze_calibration_fairness_tradeoff: fewer than 2 groups have at "
            f"least {min_group_size} samples, so no base-rate comparison was made. "
            "base_rate_disparity is NaN and tradeoff_severity is 'not assessed' "
            "(could not check), not a finding of a minimal trade-off.",
            UserWarning,
            stacklevel=2,
        )
        diagnosis = impossibility_diagnostics(
            y_true, y_prob, protected_attr, min_group_size=min_group_size
        )
        return TradeoffAnalysisResult(
            pareto_points=[],
            current_point=ParetoPoint(float("nan"), float("nan")),
            impossibility_diagnosis=diagnosis,
            recommendations=[
                "NOT ASSESSED: fewer than 2 groups have at least "
                f"{min_group_size} samples, so calibration and fairness were "
                "never compared across groups. Nothing here says the trade-off "
                "is small, large or absent.",
                diagnosis["explanation"],
            ],
            base_rate_disparity=float("nan"),
        )

    base_rates = {}
    for group_name in valid_groups:
        mask = gm.get_mask(group_name)
        base_rates[group_name] = np.mean(y_true[mask])

    # `max(base_rates.values()) - min(base_rates.values())` swallowed a NaN that
    # was not first in the dict: see _measured_spread for the arithmetic. A group
    # large enough to clear min_group_size can still have NO measured base rate,
    # because the gate counts ROWS and a base rate needs LABELS.
    #
    # Measured 2026-09-27 on 240 rows in two 120-row groups where group B carries
    # no label at all, base_rates {'A': 0.6, 'B': nan}:
    #   before  base_rate_disparity 0.0 (finite, so `tradeoff_severity` read
    #           "minimal" and the first recommendation was "GOOD NEWS: Base rates
    #           are similar across groups"), no warning about the comparison.
    #   after   base_rate_disparity nan, tradeoff_severity "not assessed", the
    #           GOOD NEWS line replaced by the NOT ASSESSED one, and the warning
    #           below naming group B.
    base_rate_disparity, _, rates_not_measured = _measured_spread(base_rates)
    if rates_not_measured:
        warnings.warn(
            f"analyze_calibration_fairness_tradeoff: {len(rates_not_measured)} of "
            f"{len(base_rates)} group(s) above the size gate have NO measured base "
            f"rate ({', '.join(rates_not_measured)}), so the base-rate comparison "
            f"covers {len(base_rates) - len(rates_not_measured)} group(s). "
            f"base_rate_disparity is NaN and tradeoff_severity is 'not assessed' "
            f"(could not check), not a finding of a minimal trade-off. A group with "
            f"enough rows can still have no measured base rate, because the rate "
            f"needs LABELS and the size gate counts rows.",
            UserWarning,
            stacklevel=2,
        )

    metrics_at_thresholds = {}
    all_points = []

    # ECE is computed from the continuous scores and does not depend on the
    # decision threshold; compute it once instead of once per threshold
    # (recomputing inside the loop made the calibration axis look degenerate
    # and cost 17 identical evaluations).
    from .metrics import expected_calibration_error

    ece_result = expected_calibration_error(y_true, y_prob, protected_attr, n_bins)
    ece = ece_result.overall_value

    # A RATE NEEDS THE LABELS TO BE A RATE, and the count does not depend on the
    # threshold, so it is measured ONCE, above the loop, and the disclosure is built
    # from it rather than re-derived at each of the 17 thresholds.
    #
    # BGL5 wave 4 (2026-09-30). The base-rate path above was fixed for exactly this
    # and its own comment states the rule: "A group large enough to clear
    # min_group_size can still have NO measured base rate, because the gate counts
    # ROWS and a base rate needs LABELS." The identical sentence is true one field
    # over and was not applied there. The per-threshold rate had a denominator floor
    # of ONE (``if np.sum(neg_mask) > 0``) and the only guard above the spread was
    # ``if len(valid_metrics) >= 2``, a COUNT of rates. So a group of 120 rows
    # carrying exactly ONE negative label produced an FPR from one observation and
    # entered the comparison as an equal. Measured on two 120-row groups with
    # fairness_metric='fpr_parity', B holding exactly one negative-labelled row:
    #
    #   false-alarm direction, B's row scored 0.05:
    #     group_metrics at 0.5 {'A': 0.8333, 'B': 0.0}, fairness_violation 0.8333
    #   FALSE-CLEAN direction, A's negatives and B's one negative all above 0.5:
    #     group_metrics at 0.5 {'A': 1.0, 'B': 1.0}, fairness_violation 0.0, i.e.
    #     PERFECT FPR PARITY published as a measurement where one side of it is a
    #     single row, and compute_pareto_frontier then made that 0.0 a frontier
    #     point at every threshold
    #
    # with zero warnings and nothing on the result stating the denominator.
    #
    # THE FLOOR SITS ABOVE THE METRIC DISPATCH. The two rate arms need different
    # label classes but they share one precondition, and a floor written inside
    # ``fpr_parity`` would have left ``fnr_parity`` publishing a 0.0 from one row.
    # demographic_parity is deliberately outside it: it is a rate over PREDICTIONS,
    # so its denominator is the group's row count, which min_group_size already
    # gates, and requiring labels for it would refuse a comparison that is genuinely
    # measurable.
    rate_label_class: Optional[int] = None
    if fairness_metric == "fpr_parity":
        rate_label_class = 0
    elif fairness_metric == "fnr_parity":
        rate_label_class = 1

    labelled_counts: Dict[Any, int] = {}
    thin_rate_groups: Dict[str, int] = {}
    if rate_label_class is not None:
        for group_name in valid_groups:
            n_labelled = int(np.sum(y_true[gm.get_mask(group_name)] == rate_label_class))
            labelled_counts[group_name] = n_labelled
            if n_labelled < _MIN_LABELS_PER_RATE:
                thin_rate_groups[str(group_name)] = n_labelled

    if thin_rate_groups:
        named = ", ".join(f"{g} ({n} labelled)" for g, n in sorted(thin_rate_groups.items()))
        warnings.warn(
            f"analyze_calibration_fairness_tradeoff: {len(thin_rate_groups)} of "
            f"{len(valid_groups)} group(s) above the size gate carry fewer than "
            f"{_MIN_LABELS_PER_RATE} label(s) of the class {fairness_metric} needs "
            f"({named}), so NO {fairness_metric} rate was measured for them at any "
            f"threshold and they contribute nothing to fairness_violation. A rate from "
            f"one observation can only be 0.0 or 1.0, so it would decide the "
            f"comparison rather than inform it. The size gate counts ROWS; a rate "
            f"needs LABELS OF ONE CLASS.",
            UserWarning,
            stacklevel=2,
        )

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        group_metrics = {}
        for group_name in valid_groups:
            mask = gm.get_mask(group_name)
            y_t_g = y_true[mask]
            y_p_g = y_pred[mask]

            if rate_label_class is None:  # demographic_parity
                group_metrics[group_name] = np.mean(y_p_g)
            elif labelled_counts.get(group_name, 0) < _MIN_LABELS_PER_RATE:
                group_metrics[group_name] = np.nan
            elif rate_label_class == 0:  # fpr_parity
                group_metrics[group_name] = np.mean(y_p_g[y_t_g == 0])
            else:  # fnr_parity
                group_metrics[group_name] = np.mean(1 - y_p_g[y_t_g == 1])

        valid_metrics = [v for v in group_metrics.values() if not np.isnan(v)]
        if len(valid_metrics) >= 2:
            fairness_violation = max(valid_metrics) - min(valid_metrics)
        else:
            # A disparity needs two rates, the same rule `calibration_vs_error_parity`
            # already applies a few hundred lines below. 0.0 is PERFECT PARITY, a
            # measurement, and it was being reported for a comparison never made.
            # Measured 2026-09-10 on two 120-row groups where B had zero negative
            # labels, so fpr_parity had exactly one measurable rate: group_metrics
            # {'A': 0.458, 'B': nan}, fairness_violation 0.0 at all 17 thresholds,
            # no warning, while `calibration_vs_error_parity` answered
            # fpr_disparity=nan on the identical input.
            fairness_violation = float("nan")

        metrics_at_thresholds[threshold] = {
            "ece": ece,
            "fairness_violation": fairness_violation,
            "group_metrics": group_metrics,
        }

        all_points.append(
            ParetoPoint(
                calibration_error=ece,
                fairness_violation=fairness_violation,
                threshold=threshold,
                parameters={"group_metrics": group_metrics},
            )
        )

    # Find Pareto frontier
    pareto_points = compute_pareto_frontier(all_points)

    # Current point at threshold=0.5
    current_metrics = metrics_at_thresholds.get(
        0.5, metrics_at_thresholds.get(thresholds[len(thresholds) // 2])
    )
    # The middle-index threshold is always one of the keys populated above,
    # so the fallback lookup always resolves to a dict (never None).
    assert current_metrics is not None
    current_point = ParetoPoint(
        calibration_error=current_metrics["ece"],
        fairness_violation=current_metrics["fairness_violation"],
        threshold=0.5,
    )

    # min_group_size IS PASSED THROUGH. It was not, so the base_rates dict built
    # under this function's gate was read by a diagnostician gating at its own
    # default of 30, and the two disagreed about which groups were in the
    # comparison: a 10-row group at min_group_size=5 entered the disparity here and
    # was simultaneously named as EXCLUDED there. One gate, named once.
    # CalibrationAnalyzer.diagnose_impossibility was already fixed this way.
    impossibility_diagnosis = impossibility_diagnostics(
        y_true, y_prob, protected_attr, base_rates, min_group_size=min_group_size
    )

    recommendations = _generate_tradeoff_recommendations(
        base_rate_disparity, current_point, pareto_points, fairness_metric, impossibility_diagnosis
    )

    return TradeoffAnalysisResult(
        pareto_points=pareto_points,
        current_point=current_point,
        impossibility_diagnosis=impossibility_diagnosis,
        recommendations=recommendations,
        base_rate_disparity=base_rate_disparity,
        metrics_at_thresholds=metrics_at_thresholds,
        rate_label_counts={str(g): n for g, n in labelled_counts.items()},
        groups_without_measured_rate=sorted(thin_rate_groups),
    )


def compute_pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """
    Compute the Pareto frontier from a set of points.

    The Pareto frontier contains points where no other point is better
    in both calibration error and fairness violation.

    Args:
        points: List of candidate points

    Returns:
        List of Pareto-optimal points. A point with a non-finite coordinate
        cannot be ranked: it is left OUT of the frontier, flagged
        ``is_pareto_optimal=None``, and named in a warning. Every ranked point
        carries an explicit ``True`` or ``False``.

    Example:
        >>> points = [ParetoPoint(0.1, 0.2), ParetoPoint(0.2, 0.1), ParetoPoint(0.3, 0.3)]
        >>> pareto = compute_pareto_frontier(points)
        >>> len(pareto)  # First two are Pareto-optimal
        2
    """
    if not points:
        return []

    # "Nothing dominates it" is what puts a point on the frontier, and every
    # comparison against a NaN is False, so an UNMEASURED point was nothing's
    # inferior and came back flagged Pareto optimal. Measured 2026-09-27: a
    # 17-threshold sweep whose fairness axis was entirely NaN returned all 17
    # points as the frontier. Excluded from the ranking instead, which is what
    # the two sibling frontiers in this library already do.
    rankable = [p for p in points if p.is_rankable]
    unrankable = [p for p in points if not p.is_rankable]
    for point in unrankable:
        point.is_pareto_optimal = None
    if unrankable:
        warnings.warn(
            f"compute_pareto_frontier: {len(unrankable)} of {len(points)} point(s) have a "
            "non-finite calibration error or fairness violation, so no dominance "
            "comparison against them can be decided. They are reported with "
            "is_pareto_optimal=None and left OUT of the frontier, which is NOT a "
            "finding that they are dominated."
            + (
                " NOTHING could be ranked, so the frontier returned is empty for want of "
                "measurements, not because no trade-off point is optimal."
                if not rankable
                else ""
            ),
            UserWarning,
            stacklevel=2,
        )

    pareto = []
    for point in rankable:
        is_dominated = False
        for other in rankable:
            if other is point:
                continue
            verdict = other.dominates(point)
            if verdict is None:
                # Unreachable while both points are rankable, and written as a
                # skip rather than a falsy test so that a future third state
                # cannot be read as "does not dominate".
                continue
            if verdict:
                is_dominated = True
                break
        # Written on BOTH branches. Only the True was ever assigned, so a
        # dominated point kept the dataclass default and claimed optimality.
        point.is_pareto_optimal = not is_dominated
        if not is_dominated:
            pareto.append(point)

    # Sort by calibration error
    pareto.sort(key=lambda p: p.calibration_error)
    return pareto


def mitigation_pareto(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    max_rows: int = 8000,
) -> Dict[str, Any]:
    """Accuracy vs. fairness trade-off with real post-processing mitigations.

    Sweeps three honest, dependency-light strategies on a continuous score
    and returns the Pareto frontier so the user sees whether the bias can be
    fixed *without wrecking accuracy*:

      * baseline            -- current 0.5 cut
      * global threshold    -- one moved threshold (a few points)
      * group thresholds    -- per-group cut equalising selection rate
                               (demographic-parity post-processing; Hardt,
                               Price & Srebro 2016) and, if labels exist,
                               equalising TPR (equal opportunity)

    Reuses ParetoPoint + compute_pareto_frontier (one source of truth).
    cost axis = 1 - accuracy (vs ground truth if present, else vs the
    baseline decision); fairness axis = demographic-parity difference.
    Never raises -- returns ``available: False`` with a reason instead.
    That refusal is also the answer when fewer than two groups hold the 10 SCORED
    rows a selection-rate gap needs, since there is then no fairness axis to
    sweep. Rows carrying no score are excluded from every figure and counted in
    ``nRowsUnscored``: ``nan >= threshold`` is False, so they used to be decided
    as rejections and counted in their group's selection rate.
    """
    import numpy as np

    try:
        # The four disclosure counts are initialised to None, BEFORE anything that
        # can raise, so the generic except at the bottom can report each one as the
        # value it actually had. None means "never reached", which is a different
        # statement from a measured 0, and 0 would claim nothing was dropped.
        n_unscored: Optional[int] = None
        n_unlabelled: Optional[int] = None
        groups_seen: Optional[List[str]] = None
        measurable: Optional[List[Any]] = None

        s = np.asarray(sensitive_attr).ravel().astype(str)
        p = np.asarray(y_prob, dtype=float).ravel()
        n = min(len(s), len(p))
        # n is an int, so a NaN self-inequality guard was a no-op here;
        # the only empty-input condition that can occur is n == 0.
        if n == 0:
            return {"available": False, "reason": "No usable score / group arrays."}
        s, p = s[:n], p[:n]
        if np.unique(p[~np.isnan(p)]).size < 5:
            return {
                "available": False,
                "reason": "A trade-off needs the model's continuous "
                "score/probability, not just the yes/no "
                "decision. None was found.",
            }
        yt = None
        # The label-presence mask for the OBJECT/string path. None means "no object
        # column was resolved", which is different from "every row of it is
        # unlabelled"; the float path keeps using _isnan below.
        object_label_present = None
        unrecognised_labels: List[str] = []
        if y_true is not None:
            yt0 = np.asarray(y_true).ravel()[:n]
            if yt0.dtype == object:
                import pandas as pd

                # BGL5 wave 4. This was ``.isin(<positives>).astype(int)``, so
                # anything outside the positive vocabulary, INCLUDING A MISSING
                # LABEL, became a ground-truth 0. See _NEGATIVE_LABEL_WORDS for the
                # measured 50-accuracy-point consequence. A value is now POSITIVE,
                # NEGATIVE, or NOT A LABEL AT ALL, and the third state goes through
                # the same ``keep`` filter as an unscored row.
                #
                # ``.astype("string")`` maps None, float nan and pd.NA to pd.NA, and
                # ``isin`` is False for pd.NA, so all three land outside both
                # vocabularies without a special case. The literal string "None" and
                # the empty string land there too, which is the point: ``str(x)`` on
                # an absent value MINTS content, and that content must not be read
                # as a negative outcome.
                #
                # THE DISTINCT-LABEL TEST MOVED IN HERE, off ``np.unique``, and that
                # is a fix in its own right. ``np.unique`` SORTS, so on a mixed
                # object column it raised and the generic except at the bottom
                # turned the whole run into "Trade-off unavailable: '<' not supported
                # between instances of 'NoneType' and 'str'", with every disclosure
                # count absent. Measured on 100 'yes' beside 100 None. The same
                # input now refuses through the normal path, with its counts.
                normalised = pd.Series(yt0).astype("string").str.strip().str.lower()
                is_positive = normalised.isin(_POSITIVE_LABEL_WORDS)
                is_negative = normalised.isin(_NEGATIVE_LABEL_WORDS)
                recognised = is_positive | is_negative
                unrecognised_labels = sorted(
                    {str(v) for v in normalised[~recognised].dropna().unique()}
                )[:5]
                # COUNT THE ROWS, NOT THE SPELLINGS. ``dropna()`` above removes
                # None, nan and pd.NA, so a column whose missing values are ALL of
                # that kind yields an EMPTY list of unrecognised spellings while
                # every one of its rows is still unlabelled. Gating the disclosure
                # on the list would have silenced it for exactly the input that
                # needs it most, which is this defect class biting its own fix.
                n_unrecognised_rows = int((~recognised).sum())
                if int(normalised[recognised].nunique()) >= 2:
                    yt = is_positive.astype(int).to_numpy()
                    object_label_present = recognised.to_numpy()
                elif n_unrecognised_rows:
                    # A SILENT LOSS OF LABEL-AWARENESS IS ITSELF A COULD-NOT-CHECK
                    # WEARING A NEUTRAL FACE. ``labelAware: False`` is the same value
                    # a caller who supplied NO labels gets, so without this a text
                    # column whose spellings are not in either vocabulary would read
                    # as "no ground truth was provided". It was provided; it could
                    # not be interpreted. The run continues on the documented
                    # label-unaware axis (cost against the baseline decision), which
                    # is a real measurement of a different thing, and says so.
                    warnings.warn(
                        f"mitigation_pareto: the ground-truth column is text and "
                        f"fewer than two of its values are in either label "
                        f"vocabulary, so NO label was read and the accuracy axis is "
                        f"measured against the BASELINE DECISION rather than against "
                        f"ground truth (labelAware=False). "
                        f"{n_unrecognised_rows} row(s) carry no recognisable label"
                        + (
                            f", spelled: {', '.join(repr(v) for v in unrecognised_labels)}"
                            if unrecognised_labels
                            else " (blank, None or nan)"
                        )
                        + ". This is NOT a finding that no ground truth was "
                        "supplied, and an unrecognised value is NOT treated as a "
                        "negative outcome.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif np.unique(yt0[~_isnan(yt0)]).size >= 2:
                yt = (yt0 >= 0.5).astype(int)
        # An UNSCORED row is not a rejected applicant. Every rate below comes
        # from `(p >= thr).astype(int)`, and `nan >= thr` is False, so a row
        # nothing scored was decided as a hard 0 and counted in the group's
        # selection rate. Dropped here, once, before any group is measured, and
        # the count is published in `nRowsUnscored`.
        #
        # Measured 2026-09-27 on 200 rows in two 100-row groups where group B was
        # never scored (every y_prob NaN, which passes the >= 5 distinct-score
        # guard above because that guard ignores the NaNs):
        #   before  available True, baselineGap 0.5, bestGap 0.25 and the summary
        #           "The selection-rate gap can be cut from 50 to 25 points via
        #           'Group thresholds (equal selection rate)' at 12.5 points of
        #           accuracy cost", with numpy's "All-NaN slice encountered" as
        #           the only warning.
        #   after   available False, reason naming 1 of 2 group(s) with at least
        #           10 SCORED rows, nRowsUnscored 100.
        # The sibling exposure_parity_rerank refuses the identical input.
        scored = ~np.isnan(p)
        n_unscored = int(n - int(scored.sum()))
        # BGL6 F04-5, 2026-09-29. AND AN UNLABELLED ROW IS NOT A GROUND TRUTH
        # NEGATIVE, by exactly the same arithmetic one line up: the label arm is
        # built with ``yt = (yt0 >= 0.5).astype(int)`` and ``nan >= 0.5`` is False,
        # so a row nobody labelled became a hard 0 in every accuracy figure, in
        # ``base_rate_g`` and therefore across the whole context-distortion axis.
        # The score arm was filtered here in 2026-09-27 and the label arm was not,
        # so the fabrication simply moved to the sibling array.
        #
        # Measured 2026-09-27 on 200 rows in two 100-row groups, both fully SCORED,
        # where group B carries no label at all:
        #   before  available True, labelAware True, nRowsUnscored 0, the current
        #           system's accuracy 0.865 against 0.495 for the same frame with
        #           B's real labels (a 37 point rise, because 100 rows nobody
        #           labelled were counted as correctly rejected), bestBalanced
        #           "Group thresholds (equal selection rate)" instead of
        #           "(equal opportunity)", "at 41.0 points of accuracy cost"
        #           instead of 5.0, and ZERO warnings.
        #   after   available False, reason naming 1 of 2 group(s) and the 100
        #           unlabelled rows, nRowsUnlabelled 100.
        #
        # Dropped with the unscored rows, once, before any group is measured, and
        # counted in ``nRowsUnlabelled``. Only when the run is label-aware at all:
        # with ``yt`` None no label is read, so there is nothing to fabricate.
        #
        # ``_isnan`` returns all-False for a non-float label array, so the
        # object/string path is not covered by IT. BGL5 wave 4: that exemption was
        # the open sibling door of this very filter, and it is closed by
        # ``object_label_present``, which the object branch above builds from the
        # two label vocabularies. The float path is unchanged.
        labelled = np.ones(n, dtype=bool)
        n_unlabelled = 0
        if yt is not None:
            if object_label_present is not None:
                labelled = object_label_present
            else:
                labelled = ~_isnan(np.asarray(y_true).ravel()[:n])
            n_unlabelled = int(n - int(labelled.sum()))
        # Kept from BEFORE the filter, so a group every one of whose rows is
        # unscored is still counted in the refusal ("1 of 2 group(s)") rather
        # than vanishing from the denominator with it.
        groups_seen = [str(g) for g in np.unique(s)]
        if n_unscored or n_unlabelled:
            keep = scored & labelled
            s, p = s[keep], p[keep]
            yt = yt[keep] if yt is not None else None
            n = int(p.size)

        if n > max_rows:
            idx = np.random.default_rng(0).choice(n, max_rows, replace=False)
            s, p = s[idx], p[idx]
            yt = yt[idx] if yt is not None else None
            # `n` has to follow the subsample, and it did not. Every mask below is
            # built from the SUBSAMPLED arrays while `gt` and `eo` were allocated
            # at the ORIGINAL length, so the assignment `gt[m] = ...` raised and
            # the except at the bottom turned it into a refusal. Measured
            # 2026-09-27 with max_rows=50 on 200 rows: before, available False
            # with reason "Trade-off unavailable: boolean index did not match
            # indexed array along axis 0; size of axis is 200 but size of
            # corresponding boolean axis is 50"; after, available True with
            # baselineGap 1.0 and bestGap 0.0338. Any input above max_rows (8000
            # by default) was refused with a numpy message. Found while adding the
            # unscored-row filter above, which needs the same update.
            n = int(p.size)

        groups = [g for g in np.unique(s)]
        masks = {g: (s == g) for g in groups}
        base = (p >= 0.5).astype(int)
        ref = yt if yt is not None else base  # accuracy reference

        # The fairness axis is a gap BETWEEN two groups of at least 10 rows, and
        # with fewer than two of those there is no gap to sweep. Both helpers
        # below used to answer 0.0 for that, which is the best value the axis has.
        # Measured 2026-09-27, one group of 200 rows: available True,
        # baselineGap 0.0, bestGap 0.0, fairnessGap 0.0 on every one of six
        # points, and the summary asserted "the disparity is structural, not just
        # a threshold artefact (consider data-level fixes)" about a disparity
        # nobody measured. 30 groups of ~6 rows gave the same output with
        # contextDistortion 0.0 as well. Refused here in this function's own
        # vocabulary, the `available: False` reason it already uses when no
        # continuous score is present.
        measurable = [g for g in groups if int(masks[g].sum()) >= 10]
        if len(measurable) < 2:
            return {
                "available": False,
                "reason": (
                    # "carrying a score" rather than "SCORED rows" so the wording
                    # the pre-existing pin quotes survives verbatim.
                    f"A selection-rate gap needs two groups of at least 10 rows carrying "
                    f"a score to be between; {len(measurable)} of {len(groups_seen)} "
                    f"group(s) qualify, so no gap was measured. This is NOT a finding "
                    f"that the gap is zero."
                    + (
                        f" {n_unlabelled} row(s) carry no finite label and are excluded, "
                        f"because an unlabelled row is not a ground-truth negative."
                        if n_unlabelled
                        else ""
                    )
                    # A refusal a caller cannot act on is only half a disclosure. If
                    # the label column is text and its values are in neither
                    # vocabulary, the spellings are named so the caller can see
                    # whether they are genuinely missing labels or a negative word
                    # this library does not know.
                    + (
                        f" The label column is text and these value(s) are in neither "
                        f"the positive nor the negative vocabulary, so they count as "
                        f"NO label rather than as a negative outcome: "
                        f"{', '.join(repr(v) for v in unrecognised_labels)}."
                        if unrecognised_labels
                        else ""
                    )
                ),
                "nGroups": len(groups_seen),
                "nGroupsMeasurable": len(measurable),
                "nRowsUnscored": n_unscored,
                # Always present, on every return path, for the same reason
                # nRowsUnscored is: a count that appears only when something was
                # dropped forces every reader into `.get(k, 0)`, and a defaulted
                # read of what was excluded is the shape this audit removes.
                "nRowsUnlabelled": n_unlabelled,
            }

        def _acc(pred):
            return float((pred == ref).mean())

        def _dp(pred):
            rates = [float(pred[masks[g]].mean()) for g in measurable]
            return max(rates) - min(rates)

        # 3rd axis (Gemini Feb-2024 lesson): context distortion = how far a
        # mitigation pushes each group's selection rate AWAY from its real
        # base rate. Forcing equal rates when groups genuinely differ is the
        # context-blind failure mode -- high distortion even at zero gap.
        base_rate_g = {g: float(ref[masks[g]].mean()) for g in measurable}

        def _ctx(pred):
            ds = [abs(float(pred[masks[g]].mean()) - base_rate_g[g]) for g in measurable]
            return sum(ds) / len(ds)

        pts = []  # (label, strategy, accuracy, dp, threshold, ctx)
        pts.append(("Current system", "baseline", _acc(base), _dp(base), 0.5, _ctx(base)))

        for thr in (0.30, 0.40, 0.60, 0.70):
            pr = (p >= thr).astype(int)
            pts.append(
                (
                    f"Global threshold {thr:.2f}",
                    "global_threshold",
                    _acc(pr),
                    _dp(pr),
                    thr,
                    _ctx(pr),
                )
            )

        # Group thresholds -> equalise selection rate (demographic parity).
        target_sr = float(base.mean())
        gt = np.zeros(n, dtype=int)
        gthr = {}
        for g in groups:
            m = masks[g]
            if m.sum() < 10:
                gt[m] = base[m]
                continue
            q = float(np.nanquantile(p[m], 1.0 - target_sr)) if 0 < target_sr < 1 else 0.5
            gthr[g] = q
            gt[m] = (p[m] >= q).astype(int)
        pts.append(
            (
                "Group thresholds (equal selection rate)",
                "group_threshold_dp",
                _acc(gt),
                _dp(gt),
                -1.0,
                _ctx(gt),
            )
        )

        # Group thresholds -> equalise TPR (equal opportunity), labels only.
        if yt is not None:
            target_tpr = float(base[(yt == 1)].mean()) if (yt == 1).any() else 0.5
            eo = np.zeros(n, dtype=int)
            for g in groups:
                m = masks[g]
                pos = m & (yt == 1)
                if m.sum() < 10 or pos.sum() < 5:
                    eo[m] = base[m]
                    continue
                q = float(np.nanquantile(p[pos], 1.0 - target_tpr)) if 0 < target_tpr < 1 else 0.5
                eo[m] = (p[m] >= q).astype(int)
            pts.append(
                (
                    "Group thresholds (equal opportunity)",
                    "group_threshold_eo",
                    _acc(eo),
                    _dp(eo),
                    -1.0,
                    _ctx(eo),
                )
            )

        # 2-axis frontier (accuracy x fairness) via the shared helper.
        pp = [
            ParetoPoint(
                calibration_error=round(1.0 - a, 6),
                fairness_violation=round(d, 6),
                threshold=t,
                parameters={"label": lbl, "strategy": st},
            )
            for (lbl, st, a, d, t, c) in pts
        ]
        frontier = compute_pareto_frontier(list(pp))
        front_ids = {
            (round(x.calibration_error, 6), round(x.fairness_violation, 6)) for x in frontier
        }

        # 3-axis dominance: minimise (1-accuracy, fairnessGap,
        # contextDistortion). p3 dominated if some q is <= on all three and
        # < on at least one.
        triples = [(round(1.0 - a, 6), round(d, 6), round(c, 6)) for (lbl, st, a, d, t, c) in pts]

        def _dom(i, j):
            qi, qj = triples[j], triples[i]
            return all(qi[k] <= qj[k] + 1e-9 for k in range(3)) and any(
                qi[k] < qj[k] - 1e-9 for k in range(3)
            )

        front3d = [
            i
            for i in range(len(triples))
            if not any(_dom(i, j) for j in range(len(triples)) if j != i)
        ]

        points = []
        for row_idx, ((lbl, st, a, d, t, c), pt) in enumerate(zip(pts, pp)):
            points.append(
                {
                    "label": lbl,
                    "strategy": st,
                    "accuracy": round(a, 4),
                    "fairnessGap": round(d, 4),
                    "contextDistortion": round(c, 4),
                    "threshold": (None if t < 0 else round(t, 3)),
                    "current": st == "baseline",
                    "paretoOptimal": (
                        round(pt.calibration_error, 6),
                        round(pt.fairness_violation, 6),
                    )
                    in front_ids,
                    "paretoOptimal3D": row_idx in front3d,
                }
            )

        baseline_dp = points[0]["fairnessGap"]
        best = min(points, key=lambda x: (x["fairnessGap"], -x["accuracy"]))
        improvable = best["fairnessGap"] < baseline_dp - 0.01
        summary = (
            f"The selection-rate gap can be cut from "
            f"{round(baseline_dp * 100)} to {round(best['fairnessGap'] * 100)} "
            f'points via "{best["label"]}" at '
            f"{round((points[0]['accuracy'] - best['accuracy']) * 100, 1)} "
            f"points of accuracy cost."
            if improvable
            else "No tested post-processing mitigation materially reduces the "
            "gap on this data: the disparity is structural, not just a "
            "threshold artefact (consider data-level fixes)."
        )
        # 3-axis balanced pick: lowest fairness gap + context distortion,
        # tie-broken by accuracy. Surfaces the Gemini trap explicitly.
        balanced = min(
            points, key=lambda x: (x["fairnessGap"] + x["contextDistortion"], -x["accuracy"])
        )
        ctx_cost = round((balanced["contextDistortion"] - points[0]["contextDistortion"]) * 100, 1)
        three_axis_summary = (
            f'Best 3-axis balance: "{balanced["label"]}" '
            f"(gap {round(balanced['fairnessGap'] * 100)} pts, "
            f"context distortion {round(balanced['contextDistortion'] * 100)} "
            f"pts). Pushing the gap to zero by force costs "
            f"{ctx_cost:+.0f} pts of context distortion, the "
            f"context-blind-parity trap (Gemini, Feb 2024); a low gap with "
            f"high distortion is NOT a good outcome."
        )
        if n_unscored or n_unlabelled:
            # A narrowed sample is still a measurement; a narrowed sample
            # reported as if it covered everybody is not. This sentence is the
            # only place a reader of `summary` learns the rows are gone.
            # The criterion names only what was actually applied, so the sentence
            # the earlier pin quotes survives verbatim for a score-only exclusion.
            criterion = "both a score and a finite label" if n_unlabelled else "a score"
            missing = []
            if n_unscored:
                missing.append(f"{n_unscored} row(s) had none")
            if n_unlabelled:
                missing.append(f"{n_unlabelled} row(s) had no finite label")
            summary = (
                f"{summary} Measured over the {n} row(s) that carry {criterion}: "
                f"{' and '.join(missing)} and are excluded, so nothing here is "
                f"evidence about them."
            )
        return {
            "available": True,
            "labelAware": yt is not None,
            "points": points,
            "summary": summary,
            "threeAxisSummary": three_axis_summary,
            "baselineGap": baseline_dp,
            "bestGap": best["fairnessGap"],
            "bestBalanced": balanced["label"],
            "nGroups": len(groups_seen),
            "nGroupsMeasurable": len(measurable),
            # Always present, on the measured path too, for the same reason the
            # refusal above carries it: a count that appears only when something
            # was dropped forces every reader into `.get(k, 0)`.
            "nRowsUnscored": n_unscored,
            "nRowsUnlabelled": n_unlabelled,
        }
    except Exception as e:  # noqa: BLE001 -- never crash the audit
        # THE PROMISE HOLDS ON THIS PATH TOO, AND THE FAILURE IS NOT SILENT.
        #
        # BGL5 wave 4. The two return paths above each say the counts are "always
        # present, on every return path", and this one returned NEITHER, so a reader
        # who landed here was back to a defaulted read of what was excluded: every
        # count came back None through ``.get``, indistinguishable from a count
        # that was measured as absent. Each one is now reported as the value it held
        # when the exception fired, or None where it was never reached.
        #
        # The warning is here for a second reason, stated so the next reader does
        # not remove it as noise. ``tests/test_no_silent_swallow_core.py`` flags any
        # single-statement handler that returns a literal, and it skips a handler
        # whose body does MORE, on the stated ground that "a warn, a log, a
        # re-raise" is happening. Adding the counts to the returned dict alone would
        # have made this handler invisible to that guard WITHOUT changing what it
        # does, which is evading a detector rather than satisfying it. A run that
        # failed into a catch-all genuinely should say so, so it now does, and the
        # stale allowlist entry naming this handler was deleted in the same change
        # rather than left to re-bless a future regression.
        warnings.warn(
            f"mitigation_pareto: the trade-off sweep raised and was caught, so NO "
            f"mitigation was evaluated and nothing here is a finding about fairness "
            f"or accuracy: {type(e).__name__}: {e}. The disclosure counts report "
            f"what had been measured when it raised, and None where a count was "
            f"never reached.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "available": False,
            "reason": f"Trade-off unavailable: {e}",
            "nRowsUnscored": n_unscored,
            "nRowsUnlabelled": n_unlabelled,
            "nGroups": None if groups_seen is None else len(groups_seen),
            "nGroupsMeasurable": None if measurable is None else len(measurable),
        }


def _isnan(a):
    import numpy as np

    try:
        return np.isnan(a.astype(float))
    except Exception:  # noqa: BLE001
        import numpy as np

        return np.zeros(len(a), dtype=bool)


def calibration_vs_error_parity(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    threshold: float = 0.5,
    n_bins: int = 10,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """
    Analyze the trade-off between calibration and error rate parity.

    This function quantifies how improving calibration may affect
    error rate balance between groups, and vice versa.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute
        threshold: Decision threshold
        n_bins: Number of bins for ECE
        min_group_size: Groups smaller than this contribute no rate to any
            comparison and are named in ``excluded_groups``. This was
            effectively 1, which let a ONE-ROW group settle the impossibility
            verdict: measured 2026-09-27 on two 100-row groups with a TPR gap of
            1.00 and equal base rates, ``conflict_exists`` was a measured False,
            and appending a SINGLE row of a third group with label 1 turned it
            into a measured True with base_rate_disparity 0.5 and
            ``not_assessed`` empty. The sibling
            :func:`impossibility_diagnostics` was given a floor of 30 for exactly
            this and says so in its own docstring.

    Returns:
        Dictionary with trade-off metrics. Every disparity is NaN when fewer than
        two groups had a measurable rate for it, and ``conflict_exists`` has
        THREE states: ``True`` (base rates differ and a MEASURED criterion is
        violated), ``False`` (measured, and no conflict) and ``None`` (COULD NOT
        CHECK, because the base-rate comparison or the arm that would have to be
        clean was never measured). ``not_assessed`` names the quantities in that
        third state, ``n_groups_compared`` says how many groups had a MEASURED
        base rate to contribute, and ``excluded_groups`` names the groups below
        ``min_group_size``, which are in none of the figures.

    Example:
        >>> result = calibration_vs_error_parity(y_true, y_prob, gender)
        >>> print(f"ECE: {result['ece']:.3f}")
        >>> print(f"FPR Disparity: {result['fpr_disparity']:.3f}")
        >>> print(f"Conflict exists: {result['conflict_exists']}")
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")

    y_pred = (y_prob >= threshold).astype(int)

    # A size gate, and a DISCLOSED one. With min_group_size=1 a one-row group
    # carried a base rate of exactly 0.0 or 1.0 into the comparison, so one row
    # could move base_rate_disparity from 0.0 to 0.5 and flip conflict_exists
    # from a measured False to a measured True (measured 2026-09-27, see the
    # min_group_size argument above). The groups below the gate are named in
    # `excluded_groups` rather than dropped silently.
    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    excluded_groups = sorted(str(g) for g in gm.get_invalid_groups())

    group_metrics = {}
    for group_name in gm.get_valid_groups():
        mask = gm.get_mask(group_name)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_prob_g = y_prob[mask]

        # True/False positive rates.
        #
        # EACH RATE NEEDS ENOUGH LABELS TO BE A RATE, not merely one.
        #
        # BGL5 wave 4 (2026-09-30). The fix above raised the GROUP floor to 30 rows
        # because "GroupManager(min_group_size=1) let a ONE-ROW group contribute a
        # base rate of exactly 0.0 or 1.0 to the comparison". The rate arms it feeds
        # kept a floor of ONE: this was ``np.sum(pos_mask) > 0``, and the only guard
        # above the spread is the COUNT test ``if len(tprs) >= 2``. So a group of 120
        # rows carrying exactly ONE positive label cleared the 30-row gate and
        # contributed a TPR of exactly 0.0 or 1.0, which is the very thing the fix
        # removed one level down. The arms applied the COUNT rule (two rates exist)
        # and not the EVIDENCE rule (each rate needs enough labels to be a rate).
        #
        # MEASURED, two 120-row groups at the default threshold 0.5 and the default
        # min_group_size 30, A holding 12 positive labels and B exactly ONE, tuned so
        # the calibration-gap and FPR arms stay inside the 0.05 tolerance and the base
        # rates differ (0.0917):
        #
        #   B's one positive row scored 0.018: tpr {'A': 0.0, 'B': 0.0},
        #     tpr_disparity 0.0, fpr_disparity 0.0, calibration_gap_disparity 0.0,
        #     conflict_exists FALSE, not_assessed [], excluded_groups [], warnings []
        #   the SAME data with that ONE row scored 0.9: tpr {'A': 0.0, 'B': 1.0},
        #     tpr_disparity 1.0, conflict_exists TRUE, not_assessed [], warnings []
        #
        # One labelled row decided the headline verdict in both directions, and the
        # FALSE is a clean bill: it reaches the "base rates differ, no measured arm
        # is violated" branch below only because a one-observation TPR was accepted
        # as a measured arm. Nothing on the result stated the denominator either:
        # n_samples reported 120 for B, which is its ROW count, not its labelled one.
        n_positive_labels = int(np.sum(y_t == 1))
        n_negative_labels = int(np.sum(y_t == 0))

        tpr = np.mean(y_p[y_t == 1]) if n_positive_labels >= _MIN_LABELS_PER_RATE else np.nan
        fpr = np.mean(y_p[y_t == 0]) if n_negative_labels >= _MIN_LABELS_PER_RATE else np.nan

        # Calibration (mean predicted vs actual)
        mean_pred = np.mean(y_prob_g)
        mean_actual = np.mean(y_t)
        calibration_gap = mean_pred - mean_actual

        group_metrics[group_name] = {
            "tpr": tpr,
            "fpr": fpr,
            "calibration_gap": calibration_gap,
            "base_rate": mean_actual,
            "n_samples": len(y_t),
            # The LABELLED counts sit beside n_samples, which is the row count and
            # was the only denominator a reader could see. These are the actual
            # denominators of tpr and fpr above, so a NaN arm can be read off the
            # result rather than guessed at.
            "n_positive_labels": n_positive_labels,
            "n_negative_labels": n_negative_labels,
        }

    # The thin-denominator disclosure, built from the counts just measured. A
    # warning alone is not enough, because a caller that does not capture warnings
    # sees nothing, so it also travels on the returned dict below.
    rates_from_too_few_labels = {
        name: {
            "n_positive_labels": m["n_positive_labels"],
            "n_negative_labels": m["n_negative_labels"],
        }
        for name, m in group_metrics.items()
        if m["n_positive_labels"] < _MIN_LABELS_PER_RATE
        or m["n_negative_labels"] < _MIN_LABELS_PER_RATE
    }
    if rates_from_too_few_labels:
        named = ", ".join(
            f"{name} ({counts['n_positive_labels']} positive, "
            f"{counts['n_negative_labels']} negative)"
            for name, counts in sorted(rates_from_too_few_labels.items())
        )
        warnings.warn(
            f"calibration_vs_error_parity: {len(rates_from_too_few_labels)} group(s) "
            f"cleared the {min_group_size}-row size gate and still carry fewer than "
            f"{_MIN_LABELS_PER_RATE} labels of one class ({named}), so the affected "
            f"tpr / fpr arm is NaN for them rather than a rate from one or two "
            f"observations. A rate over a single label can only be 0.0 or 1.0, the "
            f"extremes of the scale, so it decides the comparison instead of "
            f"informing it. The size gate counts ROWS; a rate needs LABELS OF ONE "
            f"CLASS. This is NOT a finding that their error rates match.",
            UserWarning,
            stacklevel=2,
        )

    from .metrics import expected_calibration_error

    ece_result = expected_calibration_error(y_true, y_prob, protected_attr, n_bins)

    tprs = [m["tpr"] for m in group_metrics.values() if not np.isnan(m["tpr"])]
    fprs = [m["fpr"] for m in group_metrics.values() if not np.isnan(m["fpr"])]

    # A disparity needs two rates. 0 said "no gap on this arm", and the conflict
    # test below reads both arms, so an unmeasured one silently argued that the
    # impossibility does not bind.
    tpr_disparity = max(tprs) - min(tprs) if len(tprs) >= 2 else float("nan")
    fpr_disparity = max(fprs) - min(fprs) if len(fprs) >= 2 else float("nan")

    # Base rate disparity indicates potential for conflict.
    #
    # A BETWEEN-group disparity needs two groups to be between, the same rule the
    # two arms above already apply. `max(v) - min(v)` over ONE group is exactly
    # 0.0, and zero base-rate disparity is the condition under which the
    # impossibility theorem does NOT bind, so a single group produced the
    # strongest all-clear this function can give from a comparison nobody made.
    #
    # Measured 2026-09-27 on 200 rows of one group: base_rate_disparity 0.0,
    # conflict_exists False, ZERO warnings, while tpr_disparity and fpr_disparity
    # on the identical input correctly answered NaN.
    #
    # This site was a KNOWN blind spot of the static sweep for fabricated verdicts,
    # recorded as an illustration on 2026-09-08 and left live until now. The sweep
    # matches neutral values written as LITERALS, and here the zero is COMPUTED:
    # `max(base_rates) - min(base_rates)` over a single group is 0.0 with no literal
    # anywhere for a pattern to match. Only execution finds that class, which is why
    # the fix arrives with a fixture rather than with a rule.
    #
    # The two-group gate was not enough on its own: builtin max/min do not
    # propagate a NaN that is not first in the iterable, so TWO groups of which
    # one carries no label also collapsed to a finite 0.0. Measured 2026-09-27 on
    # 240 rows in two 120-row groups with group B unlabelled, base rates
    # {'A': 0.6, 'B': nan}: before, base_rate_disparity 0.0 and
    # calibration_gap_disparity 0.0, conflict_exists a measured False, neither
    # named in not_assessed, zero warnings; after, both NaN, conflict_exists None,
    # both named in not_assessed, and the could-not-check warning fires. See
    # _measured_spread.
    base_rates = {name: m["base_rate"] for name, m in group_metrics.items()}
    base_rate_disparity, base_rates_measured, base_rates_not_measured = _measured_spread(base_rates)
    n_groups_compared = len(base_rates_measured)

    # The impossibility (calibration + TPR balance + FPR balance cannot all hold
    # when base rates differ) only BINDS for an imperfect predictor. Report a
    # conflict when base rates differ AND at least one fairness criterion is
    # actually violated in the MEASURED metrics; a perfect predictor drives every
    # measured disparity to zero and so reports no conflict, matching the
    # "impossibility vanishes at perfect accuracy" corner. (Previously this was
    # base_rate_disparity > 0.05 alone, which wrongly flagged a perfect predictor.)
    _tol = 0.05
    calibration_gaps = {name: m["calibration_gap"] for name, m in group_metrics.items()}
    # Third arm, same rule as the other two: a spread of calibration gaps needs
    # two MEASURED gaps. The `else 0` here was the one site in this function the
    # fabricated-verdict scanner COULD see, and 0 fed the conflict test below as
    # a measured "this arm is clean". The count alone was still not enough: a
    # group whose mean label is NaN has no calibration gap, and `max - min` over
    # two gaps of which one is NaN was a finite 0.0 (measured 2026-09-27, group B
    # unlabelled: calibration_gap_disparity 0.0 before, nan after).
    calibration_gap_disparity, _, gaps_not_measured = _measured_spread(calibration_gaps)

    # THREE STATES for the conflict verdict. `nan > _tol` is False, so every
    # unmeasured arm used to argue that the impossibility does not bind, and the
    # `and` over an unmeasured base-rate disparity turned a comparison nobody
    # made into a measured "no conflict".
    #
    # A True from any MEASURED arm is still a finding: the theorem binds if base
    # rates differ and any one criterion is violated, whatever the other arms do.
    # A False is only reported when the arms that would have to be clean actually
    # were measured and were clean.
    arms = {
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
        "calibration_gap_disparity": calibration_gap_disparity,
    }
    measured_arms = {k: v for k, v in arms.items() if np.isfinite(v)}
    not_assessed = sorted(k for k in arms if k not in measured_arms)
    if not np.isfinite(base_rate_disparity):
        not_assessed.append("base_rate_disparity")
        conflict_exists: Optional[bool] = None
    elif base_rate_disparity <= _tol:
        conflict_exists = False
    elif any(v > _tol for v in measured_arms.values()):
        conflict_exists = True
    elif not_assessed:
        # Base rates differ, no measured arm is violated, and an arm is missing:
        # the missing one could be the violated one, so this is not a "no".
        conflict_exists = None
    else:
        conflict_exists = False

    if conflict_exists is None:
        unmeasured_groups = sorted(set(base_rates_not_measured) | set(gaps_not_measured))
        warnings.warn(
            f"calibration_vs_error_parity: conflict_exists is None (COULD NOT CHECK) "
            f"because {', '.join(sorted(set(not_assessed)))} had fewer than two "
            f"measurable group rates ({n_groups_compared} group(s) contributed a "
            f"measured base rate"
            + (
                f"; no measured rate from {', '.join(unmeasured_groups)}"
                if unmeasured_groups
                else ""
            )
            + (f"; below the size gate: {', '.join(excluded_groups)}" if excluded_groups else "")
            + "). That is NOT a finding that calibration and error parity are jointly "
            "achievable.",
            UserWarning,
            stacklevel=2,
        )

    if excluded_groups:
        # The exclusion has to reach the caller even when every remaining arm WAS
        # measured, because "no conflict" then means "no conflict among the groups
        # that were big enough", which is a different finding.
        warnings.warn(
            f"calibration_vs_error_parity: {len(excluded_groups)} group(s) have fewer "
            f"than {min_group_size} samples and are EXCLUDED from every figure here "
            f"({', '.join(excluded_groups)}); nothing in this result is evidence about "
            f"them.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "ece": ece_result.overall_value,
        "group_ece": ece_result.group_values,
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
        "base_rate_disparity": base_rate_disparity,
        "calibration_gap_disparity": calibration_gap_disparity,
        "conflict_exists": conflict_exists,
        # Always present, on the measured path too: a key that appears only when
        # something went wrong forces every reader into `.get(k, <neutral>)`.
        "n_groups_compared": n_groups_compared,
        "not_assessed": sorted(set(not_assessed)),
        # Two different absences, kept apart. `excluded_groups` were too small to
        # look at; `groups_without_measured_base_rate` were looked at and had no
        # rate to give, which is why the disparities above can be NaN with two
        # groups in `group_metrics`.
        "excluded_groups": excluded_groups,
        "groups_without_measured_base_rate": base_rates_not_measured,
        # A THIRD kind of absence, kept apart from the other two. These groups were
        # big enough to look at AND have a measured base rate, and still could not
        # give a tpr or an fpr, because those need labels of ONE class and the size
        # gate counts rows. Always present, on every path, for the same reason the
        # two above are: a key that appears only on failure forces a defaulted read.
        "rates_from_too_few_labels": rates_from_too_few_labels,
        "min_labels_per_rate": _MIN_LABELS_PER_RATE,
        "group_metrics": group_metrics,
        "threshold": threshold,
    }


def impossibility_diagnostics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    base_rates: Optional[Dict[str, float]] = None,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """
    Diagnose impossibility theorem conditions.

    The impossibility theorem (Kleinberg et al., 2016) states that
    calibration, balance for the positive class, and balance for the
    negative class cannot be simultaneously satisfied when base rates
    differ between groups.

    This function identifies which conditions apply to the data.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute
        base_rates: Pre-computed base rates (optional)
        min_group_size: Groups smaller than this contribute no base rate to the
            comparison. Was effectively 1, which let a ONE-ROW group produce a
            base rate of exactly 1.0 and assert that the theorem binds.

    Returns:
        Dictionary with diagnostic information. `base_rates_differ` and
        `impossibility_applies` are None when the comparison could not be made
        (fewer than two groups with a MEASURED base rate). None means the
        question was not answered, NOT that the answer is no. A group can clear
        `min_group_size` on rows and still have no measured base rate, because
        the rate needs labels; `n_groups_compared` counts the groups that were
        actually compared and `groups_without_measured_base_rate` names the rest.

    Example:
        >>> diag = impossibility_diagnostics(y_true, y_prob, race)
        >>> print(f"Base rates differ: {diag['base_rates_differ']}")
        >>> print(f"Theorem applies: {diag['impossibility_applies']}")

    References:
        Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")

    gm = GroupManager(protected_attr, min_group_size=min_group_size)

    # The impossibility theorem is a statement ABOUT A COMPARISON between
    # groups. With nothing to compare, this function used to answer anyway.
    # Measured 2026-09-08 on 200 rows of a SINGLE group: base_rate_disparity
    # 0.0, base_rates_differ False, impossibility_applies False, and an
    # explanation reading "Base rates are approximately equal across groups. In
    # this case, calibration and error rate parity can theoretically be achieved
    # simultaneously." Nothing was compared. With a ONE-ROW second group it went
    # the other way: base rate exactly 1.0 for that row, disparity 0.447,
    # impossibility_applies True, worded identically to a genuine finding.
    excluded_groups = sorted(gm.get_invalid_groups())
    if base_rates is None:
        base_rates = {}
        for group_name in gm.get_valid_groups():
            mask = gm.get_mask(group_name)
            base_rates[group_name] = float(np.mean(y_true[mask]))

    # A SUPPLIED BASE-RATE DICT IS SUBJECT TO THIS FUNCTION'S OWN GATE.
    #
    # BGL5 wave 4 (2026-09-30). ``base_rates`` is an optional argument and the main
    # caller, ``analyze_calibration_fairness_tradeoff``, passes the dict it built
    # under ITS gate and does not pass ``min_group_size``. So the rates arrived
    # built at the caller's threshold while this function gated at its own default
    # of 30, and ``_measured_spread`` read every key of the supplied dict
    # regardless. The verdict and the degeneracy test then rested on two different
    # populations: ``compared_rows`` below restricts itself with
    # ``if group_name in set(gm.get_valid_groups())`` and the spread did not.
    #
    # MEASURED, analyze_calibration_fairness_tradeoff(..., min_group_size=5) on
    # A=120 rows base rate 0.5, B=120 rows base rate 0.5, and a third group C of 10
    # rows base rate 0.0:
    #   WITHOUT C: base_rate_disparity 0.0, base_rates_differ False,
    #              impossibility_applies FALSE, n_groups_compared 2,
    #              excluded_groups [], explanation "Base rates are approximately
    #              equal across groups ... can theoretically be achieved
    #              simultaneously.", warnings []
    #   WITH C:    base_rate_disparity 0.5, base_rates_differ True,
    #              impossibility_applies TRUE, n_groups_compared 3,
    #              excluded_groups ['C'], explanation "Base rates differ
    #              substantially (50.0%). Strong trade-off ..."
    # and the warning it printed on that very run:
    #   "impossibility_diagnostics: 1 group(s) had fewer than 30 samples and were
    #    EXCLUDED from the base-rate comparison (C); the verdict rests on the 3
    #    that remain."
    # That sentence was FALSE in the same breath it was printed. C was not
    # excluded: it is the only thing that moved the disparity from 0.0 to 0.5 and
    # flipped the headline verdict. "3 that remain" out of 3 total gave it away,
    # and n_groups_compared counted C while excluded_groups named it as excluded.
    #
    # The fix is structural rather than a corrected sentence: a group this function
    # calls excluded is REMOVED from the comparison here, so the exclusion warning
    # below is true BY CONSTRUCTION and not by a coincidence of matching defaults.
    # A caller that wants a different gate passes min_group_size, which is what
    # CalibrationAnalyzer.diagnose_impossibility already does and what the main
    # trade-off path now does too.
    gated_groups = {str(g) for g in gm.get_valid_groups()}
    dropped_for_size = sorted(str(g) for g in base_rates if str(g) not in gated_groups)
    if dropped_for_size:
        base_rates = {g: r for g, r in base_rates.items() if str(g) in gated_groups}

    # The gate counted a NaN rate as a measured one, and `max(rates) - min(rates)`
    # then swallowed it: see _measured_spread. A group can clear min_group_size on
    # ROWS and still have no measured base rate, because a base rate needs LABELS.
    #
    # Measured 2026-09-27 on 240 rows in two 120-row groups where group B carries
    # no label, base_rates {'A': 0.6, 'B': nan}:
    #   before  n_groups_compared 2, base_rate_disparity 0.0, base_rates_differ
    #           False, impossibility_applies False,
    #           calibration_sacrifice_estimate 0.0, ZERO warnings, and the
    #           explanation "Base rates are approximately equal across groups. In
    #           this case, calibration and error rate parity can theoretically be
    #           achieved simultaneously."
    #   after   n_groups_compared 1, base_rate_disparity nan, base_rates_differ
    #           None, impossibility_applies None,
    #           calibration_sacrifice_estimate nan, one warning naming group B,
    #           and an explanation beginning "NOT ASSESSED.".
    base_rate_disparity, rates_measured, rates_not_measured = _measured_spread(base_rates)

    # BGL6 F04-4, 2026-09-29. THE DEGENERACY TEST RESTED ON A DIFFERENT POPULATION
    # FROM THE VERDICT IT GATES. It was
    #     overall_base_rate = float(np.mean(y_true))
    #     is_degenerate = bool(overall_base_rate < 0.01 or overall_base_rate > 0.99)
    # over EVERY row, so one group with no labels made the mean NaN, and
    # ``bool(nan < 0.01 or nan > 0.99)`` is a MEASURED False. Since
    # ``impossibility_applies = bool(base_rates_differ and not is_degenerate)``,
    # that fabricated False SATISFIED the theorem's second condition.
    #
    # Measured before this change, on 400 labelled rows (A base rate 1.000, B 0.985,
    # pooled 0.9925, so the outcome IS near universal and the theorem does not bind)
    # plus a 200-row group C carrying no label at all:
    #   before  overall_base_rate nan, is_degenerate False, impossibility_applies
    #           TRUE, and the consumer published "IMPOSSIBILITY THEOREM APPLIES".
    #           Without group C the same data answered is_degenerate True and
    #           impossibility_applies False, so the HEADLINE VERDICT FLIPPED because
    #           a group nobody labelled was added.
    #   after   overall_base_rate 0.9925, is_degenerate True, applies False, i.e.
    #           identical to the run without group C, and C is still named as
    #           contributing no base rate.
    #
    # The population is now the one the verdict rests on: the rows of the groups
    # that HAVE a measured base rate. A group whose labels are missing has no
    # measured rate (np.mean propagates NaN), so it is outside the comparison and
    # must be outside the degeneracy test with it. Deliberately not "the mean of
    # every finite label", which would fold in rows from groups the comparison
    # excluded for size.
    #
    # Where nothing is measurable the answer is the THIRD state, not False, and it
    # is computed HERE, above both return paths: the early return for fewer than two
    # measured rates publishes is_degenerate too, so a fix inside the lower branch
    # would have moved the fabricated False one branch along rather than removing
    # it. Measured on 200 rows of a single group with no labels at all: before
    # is_degenerate False, after is_degenerate None.
    compared_rows = np.zeros(len(y_true), dtype=bool)
    for group_name in rates_measured:
        if group_name in set(gm.get_valid_groups()):
            compared_rows |= gm.get_mask(group_name)
    compared_labels = np.asarray(y_true, dtype=float).ravel()[compared_rows]
    finite_labels = compared_labels[np.isfinite(compared_labels)]
    overall_base_rate = float(np.mean(finite_labels)) if finite_labels.size else float("nan")
    is_degenerate: Optional[bool]
    if np.isfinite(overall_base_rate):
        is_degenerate = bool(overall_base_rate < 0.01 or overall_base_rate > 0.99)
    else:
        is_degenerate = None
        warnings.warn(
            "impossibility_diagnostics: no group has both a measured base rate and a "
            "finite label, so the base rate of the compared population could not be "
            "measured and whether the outcome is nearly universal is unknown. Reporting "
            "is_degenerate=None (could not check), NOT is_degenerate=False, which reads "
            "as a measured 'the outcome is not degenerate' and is the condition that "
            "makes the impossibility theorem bind.",
            UserWarning,
            stacklevel=2,
        )
    base_rates_differ: Optional[bool]
    impossibility_applies: Optional[bool]
    if len(rates_measured) < 2:
        detail = (
            f"only {len(rates_measured)} group(s) have a measured base rate, of the "
            f"{len(base_rates)} with at least {min_group_size} samples"
        )
        if excluded_groups:
            detail += f" (excluded for size: {', '.join(excluded_groups)})"
        if rates_not_measured:
            detail += (
                f" (no measured base rate, so nothing to compare: {', '.join(rates_not_measured)})"
            )
        warnings.warn(
            f"impossibility_diagnostics: {detail}, so no base-rate comparison was made. "
            f"Reporting base_rates_differ=None and impossibility_applies=None (could not "
            f"check), not a finding in either direction.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "base_rates": base_rates,
            "base_rate_disparity": float("nan"),
            "base_rates_differ": None,
            "is_degenerate": is_degenerate,
            "overall_base_rate": overall_base_rate,
            "impossibility_applies": None,
            "calibration_sacrifice_estimate": float("nan"),
            "excluded_groups": excluded_groups,
            "n_groups_compared": len(rates_measured),
            # Always present, on both return paths: the groups that were inside
            # the size gate and still contributed no base rate. A key that
            # appears only when something went wrong forces every reader into
            # `.get(k, [])`, and a defaulted read of what could not be measured
            # is the shape this audit exists to remove.
            "groups_without_measured_base_rate": rates_not_measured,
            "explanation": (
                f"NOT ASSESSED. The impossibility theorem is a statement about a "
                f"comparison between groups, and {detail}. Nothing here says the "
                f"theorem does or does not bind."
            ),
        }

    if rates_not_measured:
        warnings.warn(
            f"impossibility_diagnostics: {len(rates_not_measured)} group(s) inside the "
            f"size gate have NO measured base rate ({', '.join(rates_not_measured)}) and "
            f"are NOT part of the comparison; the verdict rests on the "
            f"{len(rates_measured)} that are. This is not a finding that their base "
            f"rates match.",
            UserWarning,
            stacklevel=2,
        )

    base_rates_differ = bool(base_rate_disparity > 0.01)  # Small tolerance

    # The impossibility theorem applies when:
    # 1. Base rates differ between groups
    # 2. Neither group has extreme (0 or 1) base rates
    # 3. We want calibration AND equal error rates
    #
    # BGL6 F04-4: condition 2 is a THIRD state. ``not None`` is True, so an
    # unmeasurable degeneracy test used to satisfy condition 2 outright.
    if is_degenerate is None:
        impossibility_applies = None
    else:
        impossibility_applies = bool(base_rates_differ and not is_degenerate)

    if excluded_groups:
        warnings.warn(
            f"impossibility_diagnostics: {len(excluded_groups)} group(s) had fewer than "
            f"{min_group_size} samples and were EXCLUDED from the base-rate comparison "
            f"({', '.join(excluded_groups)}); the verdict rests on the "
            f"{len(rates_measured)} that remain.",
            UserWarning,
            stacklevel=2,
        )

    # Compute how much calibration would need to be sacrificed for parity
    # This is an approximation based on base rate disparity
    calibration_sacrifice_estimate = base_rate_disparity * 0.5

    return {
        "base_rates": base_rates,
        "base_rate_disparity": base_rate_disparity,
        "base_rates_differ": base_rates_differ,
        "is_degenerate": is_degenerate,
        "overall_base_rate": overall_base_rate,
        "impossibility_applies": impossibility_applies,
        "calibration_sacrifice_estimate": calibration_sacrifice_estimate,
        "excluded_groups": excluded_groups,
        "n_groups_compared": len(rates_measured),
        "groups_without_measured_base_rate": rates_not_measured,
        "explanation": _generate_impossibility_explanation(
            bool(base_rates_differ), base_rate_disparity, is_degenerate
        )
        + (
            f" ({len(excluded_groups)} group(s) below {min_group_size} samples were "
            f"excluded: {', '.join(excluded_groups)}.)"
            if excluded_groups
            else ""
        )
        + (
            f" ({len(rates_not_measured)} group(s) inside the size gate have no "
            f"measured base rate and are not part of the comparison: "
            f"{', '.join(rates_not_measured)}.)"
            if rates_not_measured
            else ""
        ),
    }


def _generate_impossibility_explanation(
    base_rates_differ: bool, disparity: float, is_degenerate: Optional[bool]
) -> str:
    """Generate human-readable explanation of impossibility conditions."""
    if is_degenerate is None:
        # BGL6 F04-4, 2026-09-29. This was called with ``bool(is_degenerate)``,
        # and bool(None) is False, so a degeneracy nobody could measure fell
        # through to one of the two sentences below and was read as a measurement.
        # The sentence is the surface a person acts on, so the third state has to
        # reach it and not only the dictionary key beside it.
        return (
            "NOT ASSESSED. The overall base rate could not be measured, so whether "
            "the outcome is nearly universal (the degenerate case, in which "
            "calibration and error parity may be jointly achievable) is unknown. "
            "Nothing here says the impossibility theorem does or does not bind."
        )

    if is_degenerate:
        return (
            "The outcome is nearly universal (base rate ~0 or ~1). "
            "In this degenerate case, calibration and error parity "
            "may be simultaneously achievable."
        )

    if not base_rates_differ:
        return (
            "Base rates are approximately equal across groups. "
            "In this case, calibration and error rate parity can "
            "theoretically be achieved simultaneously."
        )

    if disparity < 0.1:
        return (
            f"Base rates differ slightly ({disparity:.1%}). "
            "Some trade-off exists but may be minor in practice."
        )

    if disparity < 0.2:
        return (
            f"Base rates differ moderately ({disparity:.1%}). "
            "The impossibility theorem applies: perfect calibration "
            "and perfect error rate parity cannot both be achieved."
        )

    return (
        f"Base rates differ substantially ({disparity:.1%}). "
        "Strong trade-off between calibration and error rate parity. "
        "Practitioners must prioritize based on application context."
    )


def _generate_tradeoff_recommendations(
    base_rate_disparity: float,
    current_point: ParetoPoint,
    pareto_points: List[ParetoPoint],
    fairness_metric: str,
    impossibility_diagnosis: Dict,
) -> List[str]:
    """Generate recommendations based on trade-off analysis."""
    recommendations = []

    # THREE STATES on the first line a reader sees. This used to be a two-way
    # band: `nan < 0.05` is False and `nan < 0.15` is False, so an unmeasured
    # base-rate comparison fell through to the SIGNIFICANT branch and printed
    # "SIGNIFICANT TRADE-OFF: Large base rate disparity (nan%)". Before the
    # _measured_spread fix the same input printed the opposite, "GOOD NEWS: Base
    # rates are similar across groups", because the comparison had collapsed to a
    # finite 0.0; measured 2026-09-27 on 240 rows with group B unlabelled. Both
    # readings are verdicts about a comparison nobody made.
    if not math.isfinite(base_rate_disparity):
        recommendations.append(
            "NOT ASSESSED: the base rates could not be compared across groups "
            "(fewer than two groups have a measured base rate), so nothing here "
            "says the calibration and fairness trade-off is small, large or "
            "absent. Collect labels for the unmeasured group(s) before reading "
            "this analysis as a finding either way."
        )
    elif base_rate_disparity < 0.05:
        recommendations.append(
            "GOOD NEWS: Base rates are similar across groups. "
            "Calibration and fairness can likely be improved together."
        )
    elif base_rate_disparity < 0.15:
        recommendations.append(
            f"MODERATE TRADE-OFF: Base rate disparity is {base_rate_disparity:.1%}. "
            "Some compromise between calibration and fairness may be needed."
        )
    else:
        recommendations.append(
            f"SIGNIFICANT TRADE-OFF: Large base rate disparity ({base_rate_disparity:.1%}). "
            "Prioritization decision required between calibration and fairness."
        )

    if not math.isfinite(current_point.calibration_error):
        # Same three states as the fairness clause below. `nan > 0.1` is False
        # and so is `nan > 0.05`, so an ECE that could not be computed produced
        # SILENCE on this list, and silence here reads as "the calibration was
        # checked and it is fine". Measured 2026-09-27 on 240 rows where group B
        # carries no label: the overall ECE is NaN and neither clause fired.
        recommendations.append(
            "NOT ASSESSED: the overall calibration error could not be computed on "
            "this data, so this analysis says nothing about calibration quality "
            "in either direction."
        )
    elif current_point.calibration_error > 0.1:
        recommendations.append(
            f"HIGH PRIORITY: Current ECE ({current_point.calibration_error:.3f}) "
            "indicates poor calibration. Consider group-specific calibration."
        )
    elif current_point.calibration_error > 0.05:
        recommendations.append(
            f"MODERATE: ECE of {current_point.calibration_error:.3f} suggests "
            "room for calibration improvement."
        )

    if not math.isfinite(current_point.fairness_violation):
        # Three states. A NaN disparity means fewer than two groups had a
        # measurable rate for this metric, so the comparison was never made.
        # `nan > 0.1` is False, which silently dropped the whole clause and left
        # a reader with the same page a genuinely fair model produces.
        recommendations.append(
            f"NOT ASSESSED: {fairness_metric} could not be compared across groups "
            "(fewer than two groups have a measurable rate for it), so this "
            "analysis says nothing about that disparity in either direction."
        )
    elif current_point.fairness_violation > 0.1:
        recommendations.append(
            f"FAIRNESS CONCERN: {fairness_metric} disparity is {current_point.fairness_violation:.3f}. "
            "Consider threshold adjustment or retraining."
        )

    # Pareto frontier analysis
    if pareto_points:
        best_calibration = min(p.calibration_error for p in pareto_points)

        if current_point.calibration_error > best_calibration * 1.2:
            better_point = min(pareto_points, key=lambda p: p.calibration_error)
            recommendations.append(
                f"OPTIMIZATION: Moving to threshold {better_point.threshold:.2f} "
                f"could reduce ECE to {better_point.calibration_error:.3f}."
            )

    # READINESS-6, 2026-09-10. `.get(key, False)` does NOT fire its default when
    # the key is PRESENT holding None, and `impossibility_applies` is documented
    # Optional[bool] precisely so it can be None when the base rates could not be
    # compared. `if None:` is falsy, so a could-not-check produced the same
    # SILENCE as a measured "the theorem does not apply", and silence on this
    # list reads as "we checked and there is nothing to document". Three states.
    applies = impossibility_diagnosis.get("impossibility_applies")
    if applies is True:
        recommendations.append(
            "IMPOSSIBILITY THEOREM APPLIES: Perfect calibration and error parity "
            "cannot both be achieved. Document your prioritization decision."
        )
    elif applies is None:
        recommendations.append(
            "IMPOSSIBILITY THEOREM: COULD NOT BE CHECKED. The group base rates "
            "could not be compared on this data, so whether calibration and error "
            "parity are jointly achievable is unknown. This is NOT a finding that "
            "the theorem does not apply, and it is not a reason to skip the "
            "prioritization decision."
        )

    return recommendations


def _recommend_calibration_strategy(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    context: Literal["risk_assessment", "hiring", "healthcare", "lending", "general"] = "general",
) -> CalibrationRecommendation:
    """
    Recommend a calibration strategy based on data and context.

    Different application contexts have different requirements for
    calibration versus other fairness properties. This function
    provides context-aware recommendations.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute
        context: Application context

    Returns:
        CalibrationRecommendation with strategy and rationale

    Example:
        >>> rec = recommend_calibration_strategy(
        ...     y_true, y_prob, race, context='lending'
        ... )
        >>> print(f"Recommended: {rec.strategy}")
        >>> print(f"Priority: {rec.priority}")
        >>> print(f"Rationale: {rec.rationale}")
    """
    from .metrics import calibration_disparity, expected_calibration_error

    ece_result = expected_calibration_error(y_true, y_prob, protected_attr)
    disparity_result = calibration_disparity(y_true, y_prob, protected_attr)

    overall_ece = ece_result.overall_value
    ece_disparity = disparity_result.ece_disparity

    # H-06. `calibration_disparity` does the right thing: it excludes groups
    # below the size gate, records them in `excluded_groups`, and warns twice.
    # This function then read only `ece_disparity` and discarded the exclusions,
    # so a group left out entirely produced a disparity of 0.0 and the verdict
    # "Calibration metrics are acceptable, no immediate action needed".
    # Reproduced 2026-09-07 with a 20-row group whose ECE was far worse than the
    # 200-row group's: excluded=['B'], ece_disparity=0.0, priority="low".
    #
    # A disparity that does not cover a group is not evidence about that group.
    # The public wrapper below attaches the exclusions to whichever branch
    # returns, because this function has eight return points and attaching them
    # to one of them missed the branch that actually fired.

    y_true = coerce_to_array(y_true, "y_true")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    gm = GroupManager(protected_attr, min_group_size=1)

    base_rates = {}
    for group_name in gm.groups:
        mask = gm.get_mask(group_name)
        base_rates[group_name] = np.mean(y_true[mask])

    # Same NaN swallow as the other three sites (see _measured_spread): with one
    # group there is nothing to be between, and with two groups of which one has
    # no label `max - min` was a finite 0.0. Measured 2026-09-27 on 400 rows with
    # group B unlabelled, base_rates {'A': 0.6, 'B': nan}: before,
    # base_rate_disparity 0.0, so the trade-off warning below was None, which
    # reads as "checked, and the base rates are close enough not to matter";
    # after, NaN and the warning says the comparison was not made.
    base_rate_disparity, _, _ = _measured_spread(base_rates)
    if not math.isfinite(base_rate_disparity):
        base_rate_tradeoff_warning: Optional[str] = (
            "NOT ASSESSED: the group base rates could not be compared on this data "
            "(fewer than two groups have a measured base rate), so whether calibration "
            "would cost error rate balance is unknown, not known to be fine."
        )
    elif base_rate_disparity > 0.1:
        base_rate_tradeoff_warning = "May slightly affect error rate balance if base rates differ."
    else:
        base_rate_tradeoff_warning = None

    if context == "risk_assessment":
        # In risk assessment, calibration is paramount
        if ece_disparity > 0.05:
            return CalibrationRecommendation(
                strategy="group_specific_isotonic",
                priority="high",
                rationale=(
                    "Risk assessment requires consistent probability interpretation. "
                    f"Current ECE disparity ({ece_disparity:.3f}) indicates "
                    "group-specific calibration is needed."
                ),
                expected_improvement=ece_disparity * 0.7,
                tradeoff_warning=base_rate_tradeoff_warning,
            )
        elif overall_ece > 0.05:
            return CalibrationRecommendation(
                strategy="global_platt_scaling",
                priority="medium",
                rationale=(
                    f"Overall calibration ({overall_ece:.3f}) could be improved. "
                    "Group disparities are acceptable."
                ),
                expected_improvement=overall_ece * 0.5,
            )

    elif context == "lending":
        # Lending has regulatory requirements for consistent treatment
        if ece_disparity > 0.03:
            return CalibrationRecommendation(
                strategy="group_specific_isotonic",
                priority="high",
                rationale=(
                    "Lending decisions require fair probability interpretation. "
                    "Regulatory frameworks may require calibration across groups."
                ),
                expected_improvement=ece_disparity * 0.7,
                tradeoff_warning=("Document any trade-offs with error rate parity for compliance."),
            )

    elif context == "healthcare":
        # Healthcare prioritizes individual risk accuracy
        if overall_ece > 0.08:
            return CalibrationRecommendation(
                strategy="group_specific_beta",
                priority="high",
                rationale=(
                    "Medical risk scores must be well-calibrated for treatment decisions. "
                    "Beta calibration handles clinical probability ranges well."
                ),
                expected_improvement=overall_ece * 0.6,
            )

    elif context == "hiring":
        # Hiring may prioritize error rate parity over calibration
        if ece_disparity > 0.1:
            return CalibrationRecommendation(
                strategy="threshold_optimization_first",
                priority="medium",
                rationale=(
                    "Hiring contexts often prioritize error rate parity. "
                    "Consider threshold optimization before calibration."
                ),
                tradeoff_warning=(
                    "Calibration may conflict with equal selection rates. "
                    "Stakeholder input recommended."
                ),
            )

    if ece_disparity > 0.05:
        return CalibrationRecommendation(
            strategy="group_specific_isotonic",
            priority="medium",
            rationale=(
                f"ECE disparity of {ece_disparity:.3f} suggests group-specific "
                "calibration would improve probability consistency."
            ),
            expected_improvement=ece_disparity * 0.6,
        )
    elif overall_ece > 0.05:
        return CalibrationRecommendation(
            strategy="global_isotonic",
            priority="low",
            rationale=(
                f"Overall ECE of {overall_ece:.3f} is moderate. "
                "Global calibration may help without group-specific treatment."
            ),
            expected_improvement=overall_ece * 0.4,
        )
    else:
        return CalibrationRecommendation(
            strategy="monitor_only",
            priority="low",
            rationale=(
                "Calibration metrics are acceptable. "
                "Monitor for drift but no immediate action needed."
            ),
        )


def recommend_calibration_strategy(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> CalibrationRecommendation:
    """Recommend a calibration strategy, and say which groups it does not cover.

    H-06. `calibration_disparity` already does the right thing: it excludes
    groups below the size gate, records them in `excluded_groups`, and warns
    twice. The decision function read only the resulting `ece_disparity` and
    discarded the exclusions, so a group left out entirely produced a disparity
    of 0.0 and a low-priority "no immediate action needed".

    Reproduced 2026-09-07 with a 20-row group far worse calibrated than the
    200-row group: excluded=['B'], ece_disparity=0.0, priority="low".

    This is a WRAPPER rather than one more branch because the decision function
    has eight return points. Attaching the exclusions to the "metrics are
    acceptable" branch missed the case that actually occurred, where a moderate
    overall ECE returns two branches earlier. Verified by running it, which is
    how that was found.

    The exclusions are recomputed here rather than threaded through all eight
    returns. That repeats one `calibration_disparity` call; the alternative was
    eight edits that a future ninth branch would silently escape.
    """
    recommendation = _recommend_calibration_strategy(
        y_true, y_prob, protected_attr, *args, **kwargs
    )

    from .metrics import calibration_disparity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disparity = calibration_disparity(y_true, y_prob, protected_attr)

    # H-06 covered a group being EXCLUDED from the disparity. The disparity can
    # also fail to EXIST: with one group above the size gate, `calibration_disparity`
    # answers ece_disparity=NaN and excludes nobody, so the loop above skipped
    # straight out. Every `ece_disparity > x` test in the decision function is
    # False for a NaN, so all eight branches fell through to the last one.
    #
    # Measured 2026-09-27 on 200 rows of a single group, context "lending":
    # ece_disparity nan, excluded_groups [], strategy "monitor_only", priority
    # "low", rationale "Calibration metrics are acceptable. Monitor for drift but
    # no immediate action needed." and not_assessed_groups []. The overall ECE
    # behind that sentence was real (0.044); the GROUP comparison the function
    # exists to make was never made, and the wording covered both.
    # A SECOND condition on the same guard, and the one that actually fires on
    # the input the audit used. `calibration_disparity` computes its headline as
    # `max(ece_values) - min(ece_values)` over the per-group ECEs, and builtin
    # max/min do not propagate a NaN that is not first in the iterable, so a
    # group with no measured ECE came back as a FINITE 0.0 and
    # `math.isfinite(ece_disparity)` below was True.
    #
    # Measured 2026-09-27 on 400 rows in two 200-row groups where group B carries
    # no label at all:
    #   before  group_ece {'A': 0.58, 'B': nan}, ece_disparity 0.0,
    #           excluded_groups [], strategy "monitor_only", priority "low",
    #           disparity_assessed True, not_assessed_groups [], rationale
    #           "Calibration metrics are acceptable. Monitor for drift but no
    #           immediate action needed.", ZERO warnings. Group A's own ECE of
    #           0.58 is a real and poor measurement, reported as acceptable.
    #   after   disparity_assessed False, priority "medium", the NOT ASSESSED
    #           sentence in the rationale, not_assessed_groups ['B'], and the
    #           warning below.
    #
    # The root cause is in `calibration_disparity` itself, in
    # post_processing/calibration/metrics.py, which this batch does not own; the
    # guard is placed here as well because this is where a caller arrives, and it
    # holds whatever that function answers.
    ece_disparity = getattr(disparity, "ece_disparity", float("nan"))
    group_eces = dict(getattr(disparity, "group_ece", None) or {})
    _, eces_measured, eces_not_measured = _measured_spread(group_eces)
    if not math.isfinite(ece_disparity) or len(eces_measured) < 2:
        recommendation.disparity_assessed = False
        if recommendation.priority == "low":
            recommendation.priority = "medium"
        recommendation.rationale = (
            f"{recommendation.rationale} NOT ASSESSED: the calibration disparity "
            "BETWEEN groups could not be measured on this data ("
            + (
                f"{len(eces_not_measured)} group(s) inside the size gate have no "
                f"measured calibration error: {', '.join(eces_not_measured)}"
                if eces_not_measured
                else "fewer than two groups clear the size gate"
            )
            + "), so nothing here is evidence about group disparity in either direction."
        )
        warnings.warn(
            "recommend_calibration_strategy: the between-group calibration disparity "
            f"was not measurable ({len(eces_measured)} group(s) have a measured "
            "calibration error, and two are needed to be between), so the strategy "
            f"returned ({recommendation.strategy}) rests on the OVERALL calibration "
            "only. disparity_assessed is False; this is not a finding that groups are "
            "calibrated alike.",
            UserWarning,
            stacklevel=2,
        )

    excluded = [str(g) for g in (getattr(disparity, "excluded_groups", None) or [])]
    # Two ways a group can be outside the disparity: too small to look at, or
    # looked at and holding no measurable calibration error. Both belong in
    # not_assessed_groups, which exists so a reader can tell "no disparity found"
    # from "the disparity does not cover these groups".
    not_covered = sorted(set(excluded) | set(eces_not_measured))
    if not not_covered:
        return recommendation

    recommendation.not_assessed_groups = not_covered
    if recommendation.priority == "low":
        # "Acceptable for the groups we could measure" is not the same finding
        # as "acceptable", and must not share its priority.
        recommendation.priority = "medium"
    if excluded:
        recommendation.rationale = (
            f"{recommendation.rationale} NOTE: {len(excluded)} group(s) were EXCLUDED "
            f"from the disparity analysis ({', '.join(excluded)}), so none of this is "
            "evidence about them."
        )
    # BGL6 F04-6, 2026-09-29. THE FIELD KNEW AND THE SENTENCE DID NOT. Both ways a
    # group can be outside the disparity reach `not_assessed_groups` above and both
    # bump the priority, but the only branch that reached the RATIONALE was gated on
    # `excluded` alone, i.e. on "too small to look at". The other way, a group
    # inside the size gate whose calibration error could not be measured at all, was
    # named in a list field and nowhere a reader looks.
    #
    # It survived because the guard higher up fires on `len(eces_measured) < 2`: with
    # TWO measured group ECEs and a THIRD group carrying none, neither clause of that
    # guard is true, so the NOT ASSESSED sentence and its warning never ran, and this
    # branch did not cover the gap.
    #
    # Measured 2026-09-28 on 600 rows in three 200-row groups, A calibrated, B not,
    # C carrying no label at all:
    #   before  not_assessed_groups ['C'], disparity_assessed True, priority bumped
    #           to medium, and a rationale with no "NOT ASSESSED", no "EXCLUDED" and
    #           no mention of C, plus ZERO warnings. The disparity between A and B is
    #           real and is reported correctly; what is missing is that it says
    #           nothing about a third of the population.
    #   after   the sentence below in the rationale, naming C, and one warning.
    # A returned field nobody renders is not a disclosure.
    unmeasured_inside = [g for g in eces_not_measured if g not in set(excluded)]
    if unmeasured_inside:
        recommendation.rationale = (
            f"{recommendation.rationale} NOTE: the calibration disparity does NOT "
            f"cover {len(unmeasured_inside)} group(s) inside the size gate that have "
            f"no measured calibration error ({', '.join(unmeasured_inside)}), so "
            f"nothing here is evidence about them in either direction."
        )
        warnings.warn(
            f"recommend_calibration_strategy: {len(unmeasured_inside)} group(s) inside "
            f"the size gate have NO measured calibration error "
            f"({', '.join(unmeasured_inside)}) and are NOT part of the disparity; the "
            f"strategy returned ({recommendation.strategy}) rests on the "
            f"{len(eces_measured)} group(s) that are. This is not a finding that their "
            f"calibration matches.",
            UserWarning,
            stacklevel=2,
        )
    return recommendation
