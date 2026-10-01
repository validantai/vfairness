"""
Fairness Constraints for Threshold Optimization.

This module defines the fairness constraints that can be enforced through
threshold optimization, including utility functions for computing constraint
violations and finding feasible threshold regions.
"""

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)


class FairnessConstraintType(Enum):
    """Enumeration of supported fairness constraints for threshold optimization."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    FALSE_POSITIVE_PARITY = "false_positive_parity"
    CALIBRATION = "calibration"
    CUSTOM = "custom"


@dataclass
class ConstraintViolation:
    """
    Container for constraint violation metrics.

    Attributes:
        constraint_type: Type of fairness constraint
        violation: Overall violation magnitude (0 = satisfied)
        group_metrics: Per-group metric values
        group_violations: Per-group violation from target
        is_satisfied: Whether the constraint is satisfied within tolerance, or
            None when the violation could not be measured (a group whose rate has
            an empty denominator). COULD NOT CHECK is neither satisfied nor
            violated, and collapsing it either way fabricates a verdict.
        tolerance: Tolerance used for satisfaction check
        unmeasured: The group cells whose rate had an empty denominator, e.g.
            ``('B',)`` or, for equalized odds, ``('B.fpr',)``. Empty when every
            cell was measured. This is the machine readable half of the third
            state: a reader that never sees a warning can still tell a complete
            measurement from a partial one.
        violation_is_lower_bound: True when ``violation`` was computed over the
            cells that COULD be measured while others could not, so the real
            disparity is at least this large and may be larger. It is set only
            alongside a determinate verdict (see below); when the unmeasured
            cells could still change the verdict, ``violation`` is NaN and
            ``is_satisfied`` is None instead.

    THE THIRD STATE IS NOT THE ONLY ANSWER TO AN UNMEASURED CELL.
    A conjunctive constraint with one dead arm is still determinately VIOLATED
    when the arm that WAS measured is outside tolerance: no value of the
    unknown arm can rescue it. Collapsing that into could-not-check deletes a
    real finding, which is worse than the fabrication it was guarding against.
    So:

        measured gap > tolerance, other cell unmeasured
            -> is_satisfied False, violation = the measured gap, flagged as a
               LOWER BOUND, unmeasured naming the dead cell
        measured gap <= tolerance, other cell unmeasured
            -> is_satisfied None, violation NaN (the unknown cell really could
               decide it)
        nothing measurable at all
            -> is_satisfied None, violation NaN
    """

    constraint_type: FairnessConstraintType
    violation: float
    # Per-group metric values. For most constraints this is a float per group,
    # but for equalized odds it holds a nested {'tpr', 'fpr'} dict per group,
    # so the value type is intentionally heterogeneous.
    group_metrics: Dict[str, Any]
    group_violations: Dict[str, float]
    is_satisfied: Optional[bool]
    tolerance: float
    unmeasured: Tuple[str, ...] = ()
    violation_is_lower_bound: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "constraint_type": self.constraint_type.value,
            "violation": self.violation,
            "group_metrics": self.group_metrics,
            "group_violations": self.group_violations,
            "is_satisfied": self.is_satisfied,
            "tolerance": self.tolerance,
            # The third state travels with the dict too, so a consumer that only
            # ever sees JSON can still tell "measured 0.0" from "measured
            # nothing" and "0.60" from "at least 0.60".
            "unmeasured": list(self.unmeasured),
            "violation_is_lower_bound": self.violation_is_lower_bound,
        }


def _compute_group_rates(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute classification rates for each group.

    Returns dict with TPR, FPR, precision, positive_rate for each group.
    """
    unique_groups = np.unique(groups)
    rates = {}

    for group in unique_groups:
        mask = groups == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        # C-09. All four of these defaulted to 0.0 on an EMPTY denominator, and
        # 0.0 is a real, meaningful rate, so an unmeasurable group became a
        # measured one. It fabricated in BOTH directions, which is why the fix
        # needs a two-sided control. Measured 2026-09-07 on equalized_odds:
        #
        #   group B has 0 positive labels -> B.tpr 0.0 against A.tpr 0.556,
        #       violation=0.556, is_satisfied=False    a fabricated BREACH
        #   no positives anywhere         -> every tpr 0.0, violation=0.0,
        #       is_satisfied=True                      a fabricated ALL-CLEAR
        #
        # both with zero warnings. An empty denominator is NaN: the rate was not
        # measured, and NaN propagates into the violation so the verdict below
        # can see it rather than averaging it away.
        positive_rate = float(np.mean(y_p)) if len(y_p) > 0 else float("nan")

        pos_mask = y_t == 1
        tpr = float(np.mean(y_p[pos_mask])) if pos_mask.sum() > 0 else float("nan")

        neg_mask = y_t == 0
        fpr = float(np.mean(y_p[neg_mask])) if neg_mask.sum() > 0 else float("nan")

        pred_pos = y_p == 1
        precision = float(np.mean(y_t[pred_pos])) if pred_pos.sum() > 0 else float("nan")

        rates[str(group)] = {
            "positive_rate": positive_rate,
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "n_samples": int(mask.sum()),
            "n_positive": int(pos_mask.sum()),
            "n_negative": int(neg_mask.sum()),
        }

    return rates


def _metric_disparity(
    values: Dict[str, float],
    reference: Optional[str],
) -> Tuple[float, Dict[str, float], Tuple[str, ...]]:
    """Overall disparity of a per-group metric plus its per-group breakdown.

    By default (reference is None) the disparity is the max-min spread across
    ALL groups (the worst pair), which is the canonical multi-group
    demographic-parity / equalized-odds difference (fairlearn
    ``MetricFrame.difference`` = largest minus smallest group rate; Agarwal
    et al. 2018). A reference-group form understates the true disparity for
    3+ groups because it never compares the two extreme non-reference groups
    (it is only equivalent in the 2-group case), so it is used only when the
    caller explicitly pins a reference group as an opt-in override.
    """
    # BGL-G05. A BETWEEN-group disparity needs two groups to be between. With a
    # single group `min(values) == max(values)` is exactly 0.0 (and the
    # reference form below gives `abs(v - v)` = 0.0 for the same reason), and
    # `compute_constraint_violation` then grades `bool(0.0 <= tolerance)` as a
    # measured PASS. Measured 2026-09-11 on 200 rows of one group: violation
    # 0.0, is_satisfied True, group_violations {'a': 0.0}, zero warnings, for
    # demographic_parity, equalized_odds, equal_opportunity AND
    # predictive_parity alike - the library's headline verdict, rendered as a
    # tick in ThresholdResult.summary(), on a comparison nobody made.
    #
    # This library already refuses that reasoning in writing elsewhere for the
    # same condition (vision/__init__.py:87 "a single group satisfies by
    # definition and would score a perfect 0.0 ... This is NOT a pass",
    # post_processing/calibration/tradeoffs.py:231, ranking/rerank.py:173, and
    # tests/test_spine_classification_metrics.py:79). NaN is the third state:
    # it is neither satisfied nor violated, and it routes to the
    # is_satisfied=None branch below.
    #
    # The guard sits ABOVE the reference dispatch on purpose: fixing only the
    # max-min branch would move the fabrication into its sibling.
    if len(values) < 2:
        # `unmeasured` stays empty on purpose: the rate of the one group was
        # measured perfectly well. What is missing is a second group, and
        # `compute_constraint_violation` names that reason separately.
        return float("nan"), {g: float("nan") for g in values}, ()

    # C-09 / BGL-S2b. A rate that was not measured cannot be silently excluded
    # from the comparison, in EITHER branch. The first version of this guard
    # sat between the two and its own comment said it was above them, so
    # `reference_group=<a measurable group>` walked straight past it. Measured
    # on the live tree 2026-09-17, group B holding no negative rows:
    #
    #   reference_group=None -> violation nan, is_satisfied None, 1 warning
    #   reference_group='A'  -> violation 0.0, is_satisfied True,  0 warnings
    #       (false_positive_parity), and 0.02 / True / 0 warnings for
    #       equalized_odds, while group_violations still read {'A': 0.0,
    #       'B': nan}
    #
    # the exact fabrication `_max_or_nan` was written to remove, reachable
    # through a keyword argument.
    #
    # What is returned instead is NOT a blanket NaN. The groups that WERE
    # measured still carry a real spread between them, and that spread is a
    # LOWER BOUND on the disparity across all groups (adding a group can only
    # widen a max-min spread, never narrow it). The caller decides what to do
    # with a lower bound; `compute_constraint_violation` uses it to keep a
    # determinate breach rather than throwing the finding away.
    unmeasured = tuple(sorted(g for g, v in values.items() if not math.isfinite(v)))
    measurable = {g: v for g, v in values.items() if math.isfinite(v)}

    if len(measurable) < 2 or (reference is not None and reference not in measurable):
        # Fewer than two rates, or a reference group whose own rate is the
        # missing one: there is no pair to compare, so there is no bound
        # either. A reference comparison against an unmeasured yardstick is the
        # same vacuity as comparing a single group with itself.
        return float("nan"), {g: float("nan") for g in values}, unmeasured

    if reference is not None:
        ref_val = measurable[reference]
        group_violations = {
            g: (float(abs(v - ref_val)) if math.isfinite(v) else float("nan"))
            for g, v in values.items()
        }
        violation = max(group_violations[g] for g in measurable)
        return float(violation), group_violations, unmeasured

    lo = min(measurable.values())
    hi = max(measurable.values())
    # Per-group deviation from the floor (min rate); the overall violation is
    # the full spread (max minus min), i.e. the worst pair.
    group_violations = {
        g: (float(v - lo) if math.isfinite(v) else float("nan")) for g, v in values.items()
    }
    return float(hi - lo), group_violations, unmeasured


def _max_or_nan(*values: float) -> float:
    """The largest value, or NaN if ANY of them was not measured.

    A conjunctive constraint (equalized odds is "equal TPR AND equal FPR") has
    no verdict when one of its arms has none. Built-in ``max`` does not say so:
    ``max(0.02, nan)`` is ``0.02`` and ``max(nan, 0.02)`` is ``nan``, so the
    answer depended on argument order and, in this module's ordering, always
    fell back to the arm that HAD been measured.

    Used for the PER-GROUP cell, where "this group's equalized-odds violation"
    really is unknown as soon as one of its two rates is. The OVERALL violation
    is no longer built this way: a dead arm there leaves the measured arm
    standing as a lower bound, because a lower bound outside tolerance settles
    the conjunction on its own. See ``compute_constraint_violation``.
    """
    if any(not math.isfinite(v) for v in values):
        return float("nan")
    return max(values)


def compute_constraint_violation(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    constraint: Union[FairnessConstraintType, str],
    tolerance: float = 0.05,
    reference_group: Optional[str] = None,
) -> ConstraintViolation:
    """
    Compute the violation of a fairness constraint.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (after thresholding)
        sensitive_attr: Sensitive attribute for grouping
        constraint: Type of fairness constraint to check
        tolerance: Acceptable deviation from fairness (default 0.05). Must be a
            finite, non-negative number: a NaN tolerance makes every comparison
            False and publishes a VIOLATION decided by nothing, an infinite one
            grades any disparity as satisfied, and a negative one grades a
            disparity of exactly 0.0 as a violation. All three are refused.
        reference_group: Optional explicit reference group. By DEFAULT (None)
            the disparity is the max-min spread across all groups (the worst
            pair). Passing a reference group is an opt-in override that
            measures each group's deviation from that one group; note this
            understates the true disparity for 3+ groups.

    Returns:
        ConstraintViolation object with detailed violation metrics

    Example:
        >>> violation = compute_constraint_violation(
        ...     y_true, y_pred, gender,
        ...     constraint='demographic_parity',
        ...     tolerance=0.05
        ... )
        >>> print(f"Violation: {violation.violation:.3f}")
        >>> print(f"Satisfied: {violation.is_satisfied}")
    """
    y_true = coerce_to_array(y_true).astype(int)
    y_pred = coerce_to_array(y_pred).astype(int)
    sensitive_attr = coerce_to_array(sensitive_attr)
    check_consistent_length(y_true, y_pred, sensitive_attr)

    if isinstance(constraint, str):
        constraint = FairnessConstraintType(constraint)

    # THE TOLERANCE IS THE OTHER OPERAND OF THE VERDICT, AND IT WAS NOT CHECKED.
    #
    # The comment on the three-state block below guards the LEFT side of
    # `violation <= tolerance` ("bool(nan <= tolerance) is False, so an
    # unmeasurable constraint used to report a BREACH") and the right side ran
    # unguarded. Measured 2026-09-27 on a fully measured demographic-parity gap
    # of 0.50, one call per tolerance:
    #
    #   tolerance 0.05 -> is_satisfied False   (correct)
    #   tolerance nan  -> is_satisfied False, violation 0.50, warnings []
    #   tolerance inf  -> is_satisfied True,  violation 0.50, warnings []
    #   tolerance -0.1 -> is_satisfied False  (and a violation of exactly 0.0,
    #                     perfect parity, is graded False the same way)
    #
    # A NaN tolerance is the realistic one: a missing cell in a pandas config
    # table arrives as NaN, not as None, and None would at least raise on the
    # comparison. Every comparison against NaN is False, so the published
    # verdict was a VIOLATION decided by no tolerance at all, and an infinite
    # one grades every disparity as satisfied. Refusing is the third state here:
    # the tolerance is the caller's own input, so there is nothing about the
    # data to disclose and nothing to fall back on.
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"compute_constraint_violation: tolerance must be a finite, non-negative "
            f"number, got {tolerance!r}. A NaN tolerance makes every comparison False and "
            f"publishes is_satisfied=False, a measured VIOLATION decided by no tolerance; "
            f"an infinite one publishes is_satisfied=True for any disparity; a negative "
            f"one grades even a disparity of exactly 0.0 as a violation."
        )

    rates = _compute_group_rates(y_true, y_pred, sensitive_attr)

    # The disparity is the max-min spread across ALL groups (worst pair) by
    # default. A reference-group comparison (only equivalent in the 2-group
    # case) understates the true disparity for 3+ groups and produced a
    # false-feasible verdict, so it is kept only as an explicit opt-in.
    if reference_group is not None and reference_group not in rates:
        raise ValueError(
            f"reference_group {reference_group!r} not found among groups {sorted(rates)}"
        )

    group_metrics: Dict[str, Any] = {}
    group_violations: Dict[str, float] = {}

    unmeasured: Tuple[str, ...] = ()

    if constraint == FairnessConstraintType.DEMOGRAPHIC_PARITY:
        # Equal positive prediction rates
        values = {g: rates[g]["positive_rate"] for g in rates}
        group_metrics = dict(values)
        violation, group_violations, unmeasured = _metric_disparity(values, reference_group)

    elif constraint == FairnessConstraintType.EQUAL_OPPORTUNITY:
        # Equal true positive rates
        values = {g: rates[g]["tpr"] for g in rates}
        group_metrics = dict(values)
        violation, group_violations, unmeasured = _metric_disparity(values, reference_group)

    elif constraint == FairnessConstraintType.EQUALIZED_ODDS:
        # Equal TPR AND FPR: disparity is the larger of the two spreads.
        tpr_vals = {g: rates[g]["tpr"] for g in rates}
        fpr_vals = {g: rates[g]["fpr"] for g in rates}
        tpr_viol, tpr_gv, tpr_unmeasured = _metric_disparity(tpr_vals, reference_group)
        fpr_viol, fpr_gv, fpr_unmeasured = _metric_disparity(fpr_vals, reference_group)
        unmeasured = tuple([f"{g}.tpr" for g in tpr_unmeasured]) + tuple(
            f"{g}.fpr" for g in fpr_unmeasured
        )
        for group in rates:
            group_metrics[group] = {"tpr": tpr_vals[group], "fpr": fpr_vals[group]}
            # BGL-S2 (2026-09-16). `max(a, b)` returns its FIRST operand when
            # the comparison against a NaN is False, so an equalized-odds
            # verdict silently became the TPR verdict alone whenever the FPR
            # arm was unmeasurable. Equalized odds is a claim about BOTH arms:
            # if either could not be measured, the conjunction could not be
            # measured. Measured before this change with no negatives anywhere
            # (FPR is 0/0) and a TPR gap of 0.02:
            #
            #   violation 0.02, is_satisfied True, warnings []
            #   group_metrics {'A': {'tpr': 0.9, 'fpr': nan},
            #                  'B': {'tpr': 0.88, 'fpr': nan}}
            #
            # a confident "constraint satisfied" for a constraint half of which
            # was never evaluated, while the IDENTICAL fpr values one constraint
            # name away (false_positive_parity) correctly returned NaN /
            # is_satisfied None with a warning. The NaN did survive in
            # group_metrics, but is_satisfied is what the optimizer's
            # feasibility test and summary() consume.
            group_violations[group] = _max_or_nan(tpr_gv[group], fpr_gv[group])
        # BGL-S2b. `_max_or_nan` over the two ARMS threw away a determinate
        # finding: with the FPR arm dead and the TPR arm measured at 0.60
        # against a tolerance of 0.05, the conjunction is violated whatever the
        # FPR arm turns out to be, and the pre-2026-09-16 code correctly said
        # is_satisfied False. The refusal that replaced it removed the
        # fabrication AND the finding. The measured arm is kept here as a LOWER
        # BOUND, and the verdict block below decides which of the two states it
        # supports: a determinate breach, or a genuine could-not-check.
        arm_violations = [v for v in (tpr_viol, fpr_viol) if math.isfinite(v)]
        violation = max(arm_violations) if arm_violations else float("nan")

    elif constraint == FairnessConstraintType.FALSE_POSITIVE_PARITY:
        # Equal false positive rates
        values = {g: rates[g]["fpr"] for g in rates}
        group_metrics = dict(values)
        violation, group_violations, unmeasured = _metric_disparity(values, reference_group)

    elif constraint == FairnessConstraintType.PREDICTIVE_PARITY:
        # Equal precision
        values = {g: rates[g]["precision"] for g in rates}
        group_metrics = dict(values)
        violation, group_violations, unmeasured = _metric_disparity(values, reference_group)

    else:
        raise ValueError(f"Unsupported constraint type: {constraint}")

    # THREE STATES. `bool(nan <= tolerance)` is False, so an unmeasurable
    # constraint used to report a BREACH; before the rates returned NaN it
    # reported the opposite, a clean pass. Both are fabrications, which is why
    # this needs the third state rather than a different default.
    #
    # The third state is for what is genuinely UNDECIDED, and partial evidence
    # is not automatically undecided. A disparity measured over the cells that
    # existed is a lower bound on the disparity over all of them, so a lower
    # bound already outside tolerance settles the question on its own.
    is_satisfied: Optional[bool]
    violation_is_lower_bound = False

    # A LOWER BOUND ONE ULP OVER THE TOLERANCE IS NOT A DETERMINATE BREACH.
    #
    # "No value of the unmeasured cell could bring this inside tolerance" is a
    # stronger claim than the ordinary `violation <= tolerance` verdict, so it
    # needs a stronger test. Measured on the C-09 fixture in
    # tests/test_no_aggregator_fabricates_a_verdict.py: FPR 0.55 against 0.50
    # is a gap of exactly 0.05, EQUAL to the tolerance and therefore satisfied,
    # but 0.55 - 0.50 evaluates to 0.050000000000000044 in IEEE 754, an excess
    # of 4.16e-17. Without this the difference between "could not check" and a
    # published VIOLATION verdict would be binary representation error.
    determinate_breach = (
        math.isfinite(violation)
        and violation > tolerance
        and not math.isclose(violation, tolerance, rel_tol=1e-9, abs_tol=1e-12)
    )

    if math.isfinite(violation) and not unmeasured:
        is_satisfied = bool(violation <= tolerance)
    elif determinate_breach:
        # DETERMINATE BREACH ON PARTIAL EVIDENCE. Keep the finding and the
        # honest caveat: both facts reach the caller.
        is_satisfied = False
        violation_is_lower_bound = True
        warnings.warn(
            f"{constraint} is VIOLATED on the cells that could be measured: the "
            f"measured disparity is {violation:.4f} against a tolerance of "
            f"{tolerance:.4f}. No measurable rate for {list(unmeasured)} (empty "
            f"denominator), so the reported violation is a LOWER BOUND on the real "
            f"disparity, which can only be larger. is_satisfied is False because no "
            f"value of the unmeasured cell(s) could bring this inside tolerance.",
            UserWarning,
            stacklevel=2,
        )
    else:
        if len(rates) < 2:
            # BGL-G05. Name the real reason. "Empty denominator" would be a
            # false explanation here: every rate was measurable, there was
            # simply nothing to compare it with.
            reason = (
                f"the data has fewer than 2 groups ({sorted(rates)}), so a "
                "between-group disparity does not exist to be measured. A "
                "single group is NOT a pass"
            )
        else:
            cells = list(unmeasured) or sorted(
                g for g, v in group_violations.items() if not math.isfinite(v)
            )
            reason = f"no measurable rate for group(s) {cells} (empty denominator)"
            if math.isfinite(violation):
                reason += (
                    f", and the cells that WERE measured give {violation:.4f}, inside "
                    f"the tolerance of {tolerance:.4f}, so the unmeasured one(s) could "
                    f"still decide it either way"
                )
        warnings.warn(
            f"{constraint} could not be evaluated: {reason}. is_satisfied is None "
            "(COULD NOT CHECK), which is neither satisfied nor violated.",
            UserWarning,
            stacklevel=2,
        )
        violation = float("nan")
        is_satisfied = None

    return ConstraintViolation(
        constraint_type=constraint,
        violation=violation,
        group_metrics=group_metrics,
        group_violations=group_violations,
        is_satisfied=is_satisfied,
        tolerance=tolerance,
        unmeasured=unmeasured,
        violation_is_lower_bound=violation_is_lower_bound,
    )


#: Which per-group rate each constraint's feasibility question rests on. A
#: group whose required rate is undefined at EVERY threshold has no feasibility
#: answer at all; see :class:`FeasibleThresholds`.
_CONSTRAINT_RATE_KEYS: Dict[FairnessConstraintType, Tuple[str, ...]] = {
    FairnessConstraintType.DEMOGRAPHIC_PARITY: ("positive_rate",),
    FairnessConstraintType.EQUAL_OPPORTUNITY: ("tpr",),
    FairnessConstraintType.FALSE_POSITIVE_PARITY: ("fpr",),
    FairnessConstraintType.PREDICTIVE_PARITY: ("precision",),
    FairnessConstraintType.EQUALIZED_ODDS: ("tpr", "fpr"),
}


class FeasibleThresholds(Dict[str, List[float]]):
    """Feasible thresholds per group, plus the groups that could not be assessed.

    Subclasses ``dict`` so every existing caller that indexes, iterates or
    calls ``.values()`` keeps working unchanged. What is new is that a group
    whose constraint metric does not exist in the data is NOT a key holding an
    empty list, because an empty list is a measurement ("no threshold works")
    and is the exact fabrication this class removes. It is absent from the
    mapping and named in :attr:`not_assessable`, and indexing it raises a
    KeyError that says why.

    Attributes:
        not_assessable: group -> the reason its feasibility was not measured.
        unmeasurable_thresholds: group -> how many of the grid's thresholds had
            an undefined metric for that group, for groups that WERE assessed.

    BGL-S2 (2026-09-16). Before this, ``find_feasible_thresholds`` answered a
    question about a metric that did not exist in the data, silently and with a
    confident finite count. An undefined FPR became ``0.0``, which is the BEST
    attainable FPR, so the group matched every partner inside tolerance.
    Measured on 400 rows with every ``y_true == 1`` (no negatives anywhere, so
    FPR is 0/0):

        false_positive_parity -> {'a': 100, 'b': 100} feasible thresholds
        equalized_odds        -> {'a': 100, 'b': 100}
        predictive_parity     -> {'a': 100, 'b': 100}     warnings: []

    With only group b lacking negatives, false_positive_parity returned
    {'a': 6, 'b': 100}: b was feasible at all 100 thresholds against a rate
    measured nowhere, and group a's own answer was driven from 100 down to 6 by
    that fabricated partner. The cross-contamination is why the guard has to
    run BEFORE the feasibility loops rather than filtering their output.

    ``compute_constraint_violation`` in this same module already answered the
    identical degeneracy correctly (violation NaN, is_satisfied None, plus a
    warning), so the file gave two different answers to one question.
    """

    def __init__(
        self,
        feasible: Optional[Dict[str, List[float]]] = None,
        *,
        not_assessable: Optional[Dict[str, str]] = None,
        unmeasurable_thresholds: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(feasible or {})
        self.not_assessable: Dict[str, str] = dict(not_assessable or {})
        self.unmeasurable_thresholds: Dict[str, int] = dict(unmeasurable_thresholds or {})

    @property
    def feasible(self) -> Dict[str, List[float]]:
        """The assessed groups as a plain dict."""
        return dict(self)

    @property
    def all_groups_assessed(self) -> bool:
        """True only when every group in the data got a feasibility answer."""
        return not self.not_assessable

    def __missing__(self, key: str) -> List[float]:
        self._refuse_if_not_assessed(key)
        raise KeyError(key)

    def _refuse_if_not_assessed(self, key: str) -> None:
        """Raise the could-not-check KeyError for a group that was not assessed."""
        if key in self.not_assessable:
            raise KeyError(
                f"{key!r} was NOT assessed: {self.not_assessable[key]}. This is a "
                f"could-not-check, not an empty feasible set. See .not_assessable."
            )

    def get(self, key, default=None):  # type: ignore[override]
        """Like ``dict.get``, EXCEPT for a group that was not assessed.

        ``dict.get`` DOES NOT CALL ``__missing__`` (BGL grade wave,
        2026-09-30), so the one accessor a careful caller reaches for was the
        one door this class did not close. Measured on
        ``FeasibleThresholds({'a': [0.1, 0.2]}, not_assessable={'b': 'b has no
        negatives, so FPR is 0/0'})``:

            ft['b']            -> KeyError "'b' was NOT assessed: ..."   correct
            ft.get('b')        -> None      silently
            ft.get('b', [])    -> []        the fabrication this class exists
                                            to remove, handed back by the
                                            accessor whose whole purpose is to
                                            be safe

        The class docstring is explicit that "an empty list is a measurement
        ('no threshold works') and is the exact fabrication this class
        removes", and ``.get(key, [])`` mints exactly that. So a not-assessed
        group raises here too, with the same message as indexing; every other
        key behaves as ``dict.get`` always has, default included.
        """
        self._refuse_if_not_assessed(key)
        return super().get(key, default)

    def setdefault(self, key, default=None):  # type: ignore[override]
        """Like ``dict.setdefault``, except it will not INSERT a fabrication.

        The same door as :meth:`get`, one step worse: ``setdefault(key, [])``
        does not merely return the empty list, it writes it into the mapping,
        after which the group reads as assessed-and-infeasible to every later
        caller and to :attr:`all_groups_assessed`.
        """
        self._refuse_if_not_assessed(key)
        return super().setdefault(key, default)

    def __repr__(self) -> str:
        return f"FeasibleThresholds({dict(self)!r}, not_assessable={self.not_assessable!r})"


def find_feasible_thresholds(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sensitive_attr: ArrayLike,
    constraint: Union[FairnessConstraintType, str],
    tolerance: float = 0.05,
    n_thresholds: int = 100,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
) -> FeasibleThresholds:
    """
    Find threshold ranges that satisfy a fairness constraint for each group.

    This function searches through threshold values to identify which thresholds
    could potentially satisfy the given fairness constraint when combined
    appropriately across groups.

    For equalized odds, a threshold counts as feasible for a group only if
    every other group has some threshold whose joint (TPR, FPR) operating
    point lies within `tolerance` of it (Chebyshev distance), i.e. the
    threshold can participate in a joint assignment whose equalized-odds
    violation is within tolerance. For two groups this is exact; for more
    groups it is a strong necessary condition.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sensitive_attr: Sensitive attribute for grouping
        constraint: Type of fairness constraint
        tolerance: Acceptable deviation from fairness
        n_thresholds: Number of threshold values to evaluate
        min_threshold: Minimum threshold to consider
        max_threshold: Maximum threshold to consider

    Returns:
        :class:`FeasibleThresholds`: a dict of group -> feasible thresholds
        that also carries ``not_assessable``, the groups whose constraint
        metric does not exist in the data. Those groups are absent from the
        mapping rather than present with an empty list.

    Example:
        >>> feasible = find_feasible_thresholds(
        ...     y_true, y_prob, gender,
        ...     constraint='equalized_odds',
        ...     tolerance=0.05
        ... )
        >>> for group, thresholds in feasible.items():
        ...     print(f"{group}: {len(thresholds)} feasible thresholds")

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

    Ledger row: find_feasible_thresholds. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true).astype(int)
    y_prob = coerce_to_array(y_prob)
    sensitive_attr = coerce_to_array(sensitive_attr)
    check_consistent_length(y_true, y_prob, sensitive_attr)

    if isinstance(constraint, str):
        constraint = FairnessConstraintType(constraint)

    thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
    unique_groups = np.unique(sensitive_attr)

    group_rates_by_threshold: Dict[str, List[Dict[str, float]]] = {
        str(g): [] for g in unique_groups
    }

    for group in unique_groups:
        mask = sensitive_attr == group
        y_t = y_true[mask]
        y_p = y_prob[mask]

        for thresh in thresholds:
            y_pred = (y_p >= thresh).astype(int)

            pos_mask = y_t == 1
            neg_mask = y_t == 0

            # BGL-S2: an empty denominator is NaN, never 0.0. On the FPR scale
            # 0.0 is the BEST attainable value, so a group with no negatives
            # matched every partner inside tolerance and was reported feasible
            # everywhere. `_compute_group_rates` above already uses NaN for the
            # same denominators (C-09); this loop was the last site in the file
            # still substituting the flattering value.
            tpr = float(np.mean(y_pred[pos_mask])) if pos_mask.sum() > 0 else float("nan")
            fpr = float(np.mean(y_pred[neg_mask])) if neg_mask.sum() > 0 else float("nan")
            positive_rate = float(np.mean(y_pred)) if len(y_pred) > 0 else float("nan")
            precision = (
                float(np.mean(y_t[y_pred == 1])) if (y_pred == 1).sum() > 0 else float("nan")
            )

            group_rates_by_threshold[str(group)].append(
                {
                    "threshold": thresh,
                    "tpr": tpr,
                    "fpr": fpr,
                    "positive_rate": positive_rate,
                    "precision": precision,
                }
            )

    if constraint not in _CONSTRAINT_RATE_KEYS:
        raise ValueError(f"Unsupported constraint: {constraint}")

    # THE GUARD SITS ABOVE THE BRANCH SELECTION. Both feasibility paths below
    # share one precondition: the constraint's rate has to exist for a group
    # before that group can be compared on it. Filtering their OUTPUT would
    # leave the cross-contamination in place, because a fabricated partner rate
    # changes what the OTHER groups are told (measured: group a fell from 100
    # feasible thresholds to 6 purely because group b had no negatives).
    needed_keys = _CONSTRAINT_RATE_KEYS[constraint]
    not_assessable: Dict[str, str] = {}
    unmeasurable_thresholds: Dict[str, int] = {}

    for group in unique_groups:
        gname = str(group)
        rows = group_rates_by_threshold[gname]
        dead_keys = [key for key in needed_keys if not any(math.isfinite(r[key]) for r in rows)]
        if dead_keys:
            not_assessable[gname] = (
                f"{'/'.join(dead_keys)} is undefined at every threshold for this group "
                f"(empty denominator: {_denominator_note(dead_keys, y_true, sensitive_attr, group)})"
            )
            continue
        n_dead = sum(1 for r in rows if not all(math.isfinite(r[key]) for key in needed_keys))
        if n_dead:
            unmeasurable_thresholds[gname] = n_dead

    assessable = [g for g in unique_groups if str(g) not in not_assessable]

    if len(assessable) < 2:
        # Fewer than two groups left to compare, either because the data has
        # only one group or because the rest were dropped above. A feasibility
        # answer derived by comparing a group with itself is not a measurement:
        # the single-metric path below sets the window from that one group's
        # own range, so EVERY threshold lands inside it. Measured before this
        # change on 200 rows of one group: {'a': 100} for both constraints.
        for group in assessable:
            not_assessable[str(group)] = (
                f"only {len(assessable)} group(s) could be assessed for {constraint.value}, "
                f"and a between-group constraint needs at least 2. Comparing a group with "
                f"itself is satisfied by definition and is NOT a feasibility finding"
            )
        _warn_not_assessable(constraint, not_assessable, unmeasurable_thresholds)
        return FeasibleThresholds(
            {},
            not_assessable=not_assessable,
            unmeasurable_thresholds=unmeasurable_thresholds,
        )

    unique_groups = np.array(assessable, dtype=object)
    feasible: Dict[str, List[float]] = {str(g): [] for g in unique_groups}
    _warn_not_assessable(constraint, not_assessable, unmeasurable_thresholds)

    if constraint == FairnessConstraintType.DEMOGRAPHIC_PARITY:
        metric_key = "positive_rate"
    elif constraint == FairnessConstraintType.EQUAL_OPPORTUNITY:
        metric_key = "tpr"
    elif constraint == FairnessConstraintType.FALSE_POSITIVE_PARITY:
        metric_key = "fpr"
    elif constraint == FairnessConstraintType.PREDICTIVE_PARITY:
        metric_key = "precision"
    elif constraint == FairnessConstraintType.EQUALIZED_ODDS:
        # Equalized odds requires a partner threshold matching BOTH the TPR
        # and the FPR simultaneously. A threshold is feasible for a group
        # only if EVERY other group has some threshold whose joint
        # (TPR, FPR) operating point lies within `tolerance` of this
        # threshold's point in Chebyshev distance max(|dTPR|, |dFPR|), i.e.
        # the threshold can anchor a joint assignment whose equalized-odds
        # violation is within tolerance. (An earlier fix tested the TPR and
        # FPR marginals INDEPENDENTLY against cross-group achievable-range
        # overlaps; that marked thresholds feasible whose joint point no
        # partner threshold could match, and was vacuous on ordinary
        # overlapping score distributions.) For two groups this pairwise
        # check is exact; for more groups it is a strong necessary
        # condition (every group can match the anchor within tolerance).
        points = {
            str(group): np.array(
                [[r["tpr"], r["fpr"]] for r in group_rates_by_threshold[str(group)]]
            )
            for group in unique_groups
        }
        for group in unique_groups:
            own = points[str(group)]
            jointly_matchable = np.ones(len(thresholds), dtype=bool)
            for other in unique_groups:
                if str(other) == str(group):
                    continue
                partner = points[str(other)]
                # chebyshev[i, j] = max(|dTPR|, |dFPR|) between own point i
                # and partner point j
                chebyshev = np.maximum(
                    np.abs(own[:, 0][:, None] - partner[:, 0][None, :]),
                    np.abs(own[:, 1][:, None] - partner[:, 1][None, :]),
                )
                # A NaN distance is not "within tolerance": `nan <= tolerance`
                # is already False, so an unmeasurable point cannot be matched.
                # The group-level guard above means this can only bite the
                # individual thresholds counted in unmeasurable_thresholds.
                jointly_matchable &= chebyshev.min(axis=1) <= tolerance
            feasible[str(group)] = [float(thresholds[i]) for i in np.flatnonzero(jointly_matchable)]
        return FeasibleThresholds(
            feasible,
            not_assessable=not_assessable,
            unmeasurable_thresholds=unmeasurable_thresholds,
        )

    # For single-metric constraints, find overlapping ranges.
    # Only over the thresholds where the metric was actually MEASURED: Python's
    # min/max on a list holding NaN returns whichever operand it saw first, so
    # a single undefined threshold used to set the window for every group.
    group_ranges = {}
    for group in unique_groups:
        values = [
            r[metric_key]
            for r in group_rates_by_threshold[str(group)]
            if math.isfinite(r[metric_key])
        ]
        group_ranges[str(group)] = (min(values), max(values))

    overall_min = max(r[0] for r in group_ranges.values())
    overall_max = min(r[1] for r in group_ranges.values())

    if overall_min <= overall_max + tolerance:
        # There's an overlapping feasible region
        for group in unique_groups:
            for i, rates in enumerate(group_rates_by_threshold[str(group)]):
                value = rates[metric_key]
                if not math.isfinite(value):
                    continue
                if overall_min - tolerance <= value <= overall_max + tolerance:
                    feasible[str(group)].append(thresholds[i])

    return FeasibleThresholds(
        feasible,
        not_assessable=not_assessable,
        unmeasurable_thresholds=unmeasurable_thresholds,
    )


def _denominator_note(
    dead_keys: List[str],
    y_true: np.ndarray,
    sensitive_attr: np.ndarray,
    group: Any,
) -> str:
    """Name the missing denominator in the words an operator can act on."""
    mask = sensitive_attr == group
    y_t = y_true[mask]
    notes = []
    for key in dead_keys:
        if key == "fpr":
            notes.append(f"{int((y_t == 0).sum())} negative label(s) in the group")
        elif key == "tpr":
            notes.append(f"{int((y_t == 1).sum())} positive label(s) in the group")
        elif key == "precision":
            notes.append("no threshold predicted a single positive")
        else:
            notes.append(f"{int(mask.sum())} row(s) in the group")
    return "; ".join(notes)


def _warn_not_assessable(
    constraint: FairnessConstraintType,
    not_assessable: Dict[str, str],
    unmeasurable_thresholds: Dict[str, int],
) -> None:
    """Say which groups got no feasibility answer, and what the rest now mean."""
    if not_assessable:
        detail = "; ".join(f"{g}: {reason}" for g, reason in sorted(not_assessable.items()))
        warnings.warn(
            f"find_feasible_thresholds({constraint.value}): {len(not_assessable)} group(s) "
            f"could NOT be assessed ({detail}). They are absent from the returned mapping "
            f"and listed in .not_assessable; this is a could-not-check, not an empty "
            f"feasible set, and it is NOT a count of 0 or a full pass. Any remaining "
            f"group's feasibility is conditional on excluding them.",
            UserWarning,
            stacklevel=3,
        )
    if unmeasurable_thresholds:
        detail = "; ".join(f"{g}: {n}" for g, n in sorted(unmeasurable_thresholds.items()))
        warnings.warn(
            f"find_feasible_thresholds({constraint.value}): some thresholds have an "
            f"undefined metric and were skipped rather than scored ({detail}). See "
            f".unmeasurable_thresholds.",
            UserWarning,
            stacklevel=3,
        )
