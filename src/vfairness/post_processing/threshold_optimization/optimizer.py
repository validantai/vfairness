"""
Threshold Optimization for Fairness.

This module provides optimizers for finding optimal classification thresholds
that satisfy fairness constraints while maximizing model performance.

The key insight is that different demographic groups can have different
optimal thresholds to achieve fairness, without requiring model retraining.
"""

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    vacuous_bound_reason,
)
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    reject_swapped_labels_and_scores,
)

from .constraints import (
    ConstraintViolation,
    FairnessConstraintType,
    compute_constraint_violation,
)


@dataclass
class ThresholdConstraint:
    """
    Configuration for a fairness constraint in threshold optimization.

    Attributes:
        constraint_type: Type of fairness constraint
        tolerance: Maximum allowed violation. Must be finite, non-negative, and
            SMALL ENOUGH TO BE BREACHED by some value the constraint's metric
            can take. See :meth:`__post_init__`.
        weight: Accepted only as 1.0, and read by nothing. See
            :meth:`__post_init__` for why it is refused rather than ignored.
    """

    constraint_type: FairnessConstraintType
    tolerance: float = 0.05
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Refuse a tolerance the disparity cannot exceed, and a dead weight.

        A TOLERANCE OUTSIDE THE METRIC'S RANGE IS A COMPARISON THAT CANNOT BE
        FALSE (BGL grade wave, 2026-09-30). ``compute_constraint_violation``
        already refuses a NaN or infinite or negative tolerance, and that guard
        is about the bound being unreadable. It says nothing about a bound that
        is perfectly readable and still ungradeable, and that is the door that
        was open. Every constraint here is a RATE GAP in [0, 1]. Measured on 200
        rows in two groups where group 'a' is accepted at every threshold and
        'b' at none, i.e. a demographic-parity disparity of 1.0000, the largest
        one can be:

            GroupThresholdOptimizer(tolerance=0.05) -> is_feasible False,
                summary() "Feasible: False"                          correct
            GroupThresholdOptimizer(tolerance=2.0)  -> is_feasible True,
                summary() "Feasible: True", violation 1.0000, NO
                could-not-check block and ZERO warnings
            compute_constraint_violation(tolerance=1.0 / 2.0 / 100.0)
                -> is_satisfied True for the same violation of 1.0000

        So the worst possible disparity shipped as a measured PASS, decided by a
        bound that no data could have breached. This is not a smaller
        constraint; it is no constraint, reported as a satisfied one.

        The rule is not "the tolerance is finite" and not "the tolerance is at
        most 1": it is a property of the bound AND the metric's declared range
        together, so it is delegated to
        :func:`~vfairness.evaluation.vfairness_metrics._metric_direction.vacuous_bound_reason`,
        the repository's one implementation of it, rather than reinvented with a
        literal. That helper answers ``None`` for a constraint type with no
        declared range (``false_positive_parity``, ``calibration``, ``custom``),
        which is deliberate: refusing a bound whose reachability is unknown
        would be the over-correction, so those keep exactly the usability the
        caller configured.

        ``weight`` IS READ BY NOTHING. It is documented as "Importance weight
        for multi-objective optimization" and no code in this package or any
        other reads the attribute: the multi-objective optimizer ranks on its
        ``objectives`` list, not on a per-constraint weight. A knob that a
        caller sets, and that silently does nothing, states a capability the
        code does not have, so it is refused for anything but the identity
        rather than accepted and ignored. This is the disposition
        ``soft_rate_computation``'s ``temperature`` already received in
        in_processing/loss_functions/base.py (S-05) for the identical shape.
        """
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"ThresholdConstraint: tolerance must be a finite, non-negative number, "
                f"got {self.tolerance!r}. A NaN tolerance makes every comparison False "
                f"and publishes is_satisfied=False, a VIOLATION decided by no tolerance; "
                f"an infinite one publishes is_satisfied=True for any disparity; a "
                f"negative one grades even a disparity of exactly 0.0 as a violation."
            )
        self.tolerance = tolerance

        metric_name = getattr(self.constraint_type, "value", None)
        if isinstance(metric_name, str):
            reason = vacuous_bound_reason(metric_name, tolerance, BoundRole.THRESHOLD)
            if reason is not None:
                raise ValueError(
                    f"ThresholdConstraint: tolerance {tolerance!r} cannot grade "
                    f"{metric_name!r}: {reason}. is_satisfied would be True for every "
                    f"possible input, including the largest disparity the metric can "
                    f"express, so the constraint would be reported as satisfied without "
                    f"ever having been tested. Pass a tolerance inside the metric's "
                    f"range, or drop the constraint deliberately instead of widening it "
                    f"until it cannot fail."
                )

        if float(self.weight) != 1.0:
            raise ValueError(
                f"ThresholdConstraint: weight is not implemented and only 1.0 (the "
                f"identity) is accepted; got {self.weight!r}. Nothing reads this "
                f"attribute: the multi-objective optimizer ranks candidates on its "
                f"`objectives` list, not on a per-constraint weight, so a value set "
                f"here would change nothing while reading as a configured trade-off. "
                f"Use MultiObjectiveThresholdOptimizer(objectives=[...]) to weigh "
                f"objectives against each other."
            )

    @classmethod
    def from_string(cls, constraint: str, tolerance: float = 0.05) -> "ThresholdConstraint":
        """Create constraint from string name."""
        return cls(constraint_type=FairnessConstraintType(constraint), tolerance=tolerance)


@dataclass
class ThresholdResult:
    """
    Result of threshold optimization.

    Attributes:
        global_threshold: Single threshold (if uniform)
        group_thresholds: Per-group thresholds
        constraint_violations: Violation metrics for each constraint
        performance_metrics: Accuracy, F1, etc.
        is_feasible: Whether all constraints are satisfied: True, False, or
            None when the constraint could not be measured (a group's rate
            had an empty denominator, so ConstraintViolation.is_satisfied
            is None). COULD NOT CHECK is a third state; coercing it to
            False would report a measured failure nobody measured
            (2026-09-09, after audit fix C-09 widened is_satisfied).
        optimization_details: Additional optimization information
    """

    global_threshold: Optional[float]
    group_thresholds: Dict[str, float]
    constraint_violations: List[ConstraintViolation]
    performance_metrics: Dict[str, float]
    is_feasible: Optional[bool]
    optimization_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "global_threshold": self.global_threshold,
            "group_thresholds": self.group_thresholds,
            "constraint_violations": [v.to_dict() for v in self.constraint_violations],
            "performance_metrics": self.performance_metrics,
            "is_feasible": self.is_feasible,
            "optimization_details": self.optimization_details,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Threshold Optimization Result", "=" * 40]

        if self.global_threshold is not None:
            lines.append(f"Global Threshold: {self.global_threshold:.4f}")
        else:
            lines.append("Group-Specific Thresholds:")
            for group, thresh in self.group_thresholds.items():
                lines.append(f"  {group}: {thresh:.4f}")

        # Three states. None is "not assessed", never rendered as False:
        # a constraint whose rate had an empty denominator was not measured.
        if self.is_feasible is None:
            lines.append("\nFeasible: not assessed (constraint could not be measured)")
        else:
            lines.append(f"\nFeasible: {self.is_feasible}")
        lines.append("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")

        lines.append("\nConstraint Violations:")
        for violation in self.constraint_violations:
            if violation.is_satisfied is None:
                status = "?"
            else:
                status = "✓" if violation.is_satisfied else "✗"
            # BGL-S2b: a violation measured over SOME of the cells is a lower
            # bound, and printing it as a plain number reads as the whole
            # disparity. The reader sees which half was measured, here, in the
            # string they actually look at, not only in a warning that scrolled
            # past during fit.
            if violation.violation_is_lower_bound:
                lines.append(
                    f"  {status} {violation.constraint_type.value}: at least "
                    f"{violation.violation:.4f} (lower bound; no measurable rate for "
                    f"{list(violation.unmeasured)})"
                )
            else:
                lines.append(
                    f"  {status} {violation.constraint_type.value}: {violation.violation:.4f}"
                )

        # BGL-S2: the could-not-check flags used to live only in
        # optimization_details, which summary() never printed, so a correctly
        # measured refusal reached no reader. Both are printed here because
        # this string is the artifact a person actually looks at.
        caveats = []
        details = self.optimization_details or {}
        for violation in self.constraint_violations:
            if violation.violation_is_lower_bound:
                caveats.append(
                    f"  - {violation.constraint_type.value} has no measurable rate for "
                    f"{list(violation.unmeasured)}, so the violation above is a LOWER "
                    f"BOUND: the real disparity can only be larger. The verdict is a "
                    f"verdict because a lower bound already outside the tolerance of "
                    f"{violation.tolerance:.4f} cannot be brought back inside it."
                )
        if details.get("objective_measured") is False:
            caveats.append(
                f"  - The objective {details.get('objective', '?')!r} could not be computed "
                f"at any candidate threshold, so it selected nothing. These thresholds were "
                f"chosen by constraint violation alone and are NOT an optimum."
            )
        # BGL-S2c: a Pareto frontier that ranked on fewer objectives than were
        # asked for, or on none at all. Printed here for the same reason as the
        # two above: this string is the artifact a person looks at.
        unmeasured_objs = details.get("objectives_unmeasured") or []
        if unmeasured_objs:
            requested = details.get("objectives") or []
            measured_objs = details.get("objectives_measured") or []
            if measured_objs:
                caveats.append(
                    f"  - {len(unmeasured_objs)} of {len(requested)} requested objective(s) "
                    f"({', '.join(unmeasured_objs)}) could not be computed for any candidate, "
                    f"so the frontier did not rank on them. It is a frontier over "
                    f"{measured_objs}, NOT over the objectives that were requested."
                )
            else:
                caveats.append(
                    f"  - Not one of the requested objective(s) ({', '.join(unmeasured_objs)}) "
                    f"could be computed for any candidate, so no configuration could dominate "
                    f"another and every one of them came back non-dominated. This is NOT a "
                    f"Pareto frontier, and the selected point is an arbitrary position in it."
                )
        # BGL3-PP3. The frontier that was never computed at all, which is a
        # different state from the frontier that ranked on nothing above.
        # MultiObjectiveThresholdOptimizer.fit runs its candidate sweep only for
        # exactly 2 groups; on any other group count it falls back to a single
        # GroupThresholdOptimizer answer, stamps
        # optimization_details['pareto_frontier_computed'] = False, warns once,
        # and stores that one result in pareto_frontier_. Measured 2026-09-27 on
        # 200 rows and THREE groups: len(pareto_frontier_) == 1, which reads as
        # "the sweep found exactly one non-dominated configuration", while this
        # string, the artifact a person actually looks at, printed
        #
        #   Feasible: True
        #   equalized_odds: 0.0404 (with a tick)
        #
        # and NO COULD NOT CHECK block at all. The flag and the reason were on
        # the object; the only words saying so were in a warning that is gone by
        # the time anyone reads the result, which is the same one-layer-short
        # miss as the two caveats above.
        if details.get("pareto_frontier_computed") is False:
            caveats.append(
                f"  - NO Pareto frontier was computed: "
                f"{details.get('pareto_fallback_reason', 'the sweep did not run')}. These "
                f"thresholds come from a single fallback fit, so pareto_frontier_ holding "
                f"one entry does NOT mean one configuration was found non-dominated, and "
                f"nothing here was traded off against anything."
            )
        if details.get("score_resolution_sufficient") is False:
            caveats.append(
                f"  - y_prob holds {details.get('n_distinct_scores', '?')} distinct value(s), "
                f"so no threshold sweep could separate anything."
            )
        # BGL-S2b: the two states this artifact could reach while still
        # printing a tick beside the constraint.
        if details.get("constraint_measured") is False:
            caveats.append(
                f"  - The constraint could not be measured at any of the "
                f"{details.get('n_violation_candidates', '?')} candidate threshold(s), so "
                f"no candidate could be shown to violate less than another. These "
                f"thresholds are the first the scan saw, NOT a minimum."
            )
        if int(details.get("n_unscored_rows") or 0):
            caveats.append(
                f"  - {details['n_unscored_rows']} of "
                f"{details.get('n_rows_fitted', '?')} fitted row(s) had no usable score "
                f"and entered the fairness measurement as REJECTED. The constraint "
                f"figures above, and the tick beside them, rest partly on decisions "
                f"nobody measured; Feasible is reported as not assessed."
            )
        if caveats:
            lines.append("\nCOULD NOT CHECK:")
            lines.extend(caveats)

        return "\n".join(lines)


class BaseThresholdOptimizer(ABC):
    """
    Abstract base class for threshold optimizers.

    All threshold optimizers should inherit from this class and implement
    the fit() and predict() methods.
    """

    def __init__(
        self,
        constraint: Union[str, FairnessConstraintType, ThresholdConstraint] = "demographic_parity",
        tolerance: float = 0.05,
    ):
        """
        Initialize the threshold optimizer.

        Args:
            constraint: Fairness constraint to satisfy
            tolerance: Maximum allowed constraint violation
        """
        if isinstance(constraint, str):
            self.constraint = ThresholdConstraint.from_string(constraint, tolerance)
        elif isinstance(constraint, FairnessConstraintType):
            self.constraint = ThresholdConstraint(constraint, tolerance)
        elif isinstance(constraint, ThresholdConstraint):
            self.constraint = constraint
        else:
            # THE else BRANCH WAS A BARE ASSIGNMENT, so anything that was not a
            # str and not a FairnessConstraintType became self.constraint
            # unchecked (BGL grade wave, 2026-09-30). Measured:
            # Concrete(constraint=123) -> self.constraint == 123 and
            # Concrete(constraint=None) -> self.constraint is None, both
            # accepted in silence, after which the first read of
            # self.constraint.tolerance inside fit() raises an AttributeError
            # naming int or NoneType and not the argument that was wrong.
            # Refused here, where the caller can see which argument it was.
            #
            # Note what this does NOT do: it does not re-validate a
            # ThresholdConstraint that was passed in, because that object
            # validated its own tolerance and weight in __post_init__. A
            # second refusal on one value is the shape this repo avoids.
            raise TypeError(
                f"{type(self).__name__}: constraint must be a constraint name (str), a "
                f"FairnessConstraintType, or a ThresholdConstraint; got "
                f"{type(constraint).__name__} ({constraint!r}). It was previously "
                f"stored as given, and the first read of constraint.tolerance inside "
                f"fit() then failed with an AttributeError that named the type rather "
                f"than the argument."
            )

        self.is_fitted = False
        self.result_: Optional[ThresholdResult] = None

    @abstractmethod
    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "BaseThresholdOptimizer":
        """
        Fit the optimizer to find optimal thresholds.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute for grouping
            sample_weight: Optional sample weights

        Returns:
            self: The fitted optimizer
        """
        pass

    @abstractmethod
    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Apply optimized thresholds to make predictions.

        Args:
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute for grouping

        Returns:
            Binary predictions using optimized thresholds
        """
        pass

    def fit_predict(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(
            y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive_attr, sample_weight=sample_weight
        )
        return self.predict(y_prob, sensitive_attr)


# The exact keys _compute_performance_metrics returns. Anything else supplied as
# an objective is unmeasurable by this module, and the test below pins that this
# set and that function cannot drift apart.
# A threshold sweep can only ever produce as many distinct outcomes as there are
# distinct scores to cut between. Below this, "optimising a threshold" is not a
# thing that can happen: with two distinct values every threshold in (0, 1]
# means "accept the ones" and every threshold at or below 0 means "accept
# everything", so the sweep has at most two outcomes however many candidates it
# reports having tried.
SCORE_RESOLUTION_FLOOR = 3

#: The fewest candidate thresholds a grid needs before "the optimum" is a thing
#: the search can find. See :func:`_require_searchable_grid`.
THRESHOLD_GRID_FLOOR = 2


def _require_searchable_grid(n_thresholds: Any, *, caller: str) -> int:
    """Refuse a grid that cannot choose between anything.

    BGL3-PP3 (2026-09-27). Every degenerate-DATA path in this module discloses
    (see the scans below), and the degenerate GRID path disclosed nothing. With
    ``n_thresholds=0`` ``np.linspace`` returns an empty array, every search loop
    body is skipped, and what comes back is the ``best_thresholds`` SEED
    presented as the fitted result. Measured on 60 rows and two groups:

        GroupThresholdOptimizer(n_thresholds=0) -> {'a': 0.5, 'b': 0.5}
        ThresholdOptimizer(n_thresholds=0)      -> global_threshold 0.5
        both: is_feasible a measured True/False, and ZERO warnings

    The two scans cannot cover this: ``warn_if_unmeasured`` returns early when
    ``n_candidates`` is 0, correctly, because "the objective could not be
    computed" would be a false statement about data that was never scanned. So
    the only trace was ``n_objective_candidates: 0`` buried in
    ``optimization_details``, and ``summary()`` then printed two caveats that
    MISDESCRIBE what happened: "chosen by constraint violation alone" and "the
    first the scan saw, NOT a minimum", when no scan ran and 0.5 is a class
    default. ``n_thresholds=1`` is the same defect one step quieter: the single
    candidate is the grid's lower bound, so ``ThresholdOptimizer(n_thresholds=1)``
    returned 0.01 and ``GroupThresholdOptimizer(n_thresholds=1)`` 0.05 whatever
    the data said, with ``objective_measured`` True beside it.

    Refused rather than disclosed because ``n_thresholds`` is the caller's own
    argument, not a property of the data: there is nothing to measure and
    nothing to degrade to.
    """
    n = int(n_thresholds)
    if n < THRESHOLD_GRID_FLOOR:
        raise ValueError(
            f"{caller} cannot search {n} candidate threshold(s); at least "
            f"{THRESHOLD_GRID_FLOOR} are needed for one to be chosen over another. With "
            f"0 the search loop never runs and what is returned is this class's default "
            f"threshold seed dressed as an optimisation result; with 1 it is the grid's "
            f"lower bound whatever the data says. Pass n_thresholds >= "
            f"{THRESHOLD_GRID_FLOOR}."
        )
    return n


def _record_score_resolution(y_prob: np.ndarray, *, caller: str) -> Dict[str, Any]:
    """Measure what the sweep actually had to work with, and say so when it is
    not enough.

    Measured 2026-09-09 on the shape the production platform sends whenever an
    uploaded dataset has no probability column, which is its most common case:
    the consumer falls back to the PREDICTION column, so y_prob holds hard 0.0
    and 1.0 values. GroupThresholdOptimizer then returned both group thresholds
    at the 0.05 floor, handed back predictions IDENTICAL to its input, and left
    the acceptance gap at its original 0.2642, while reporting
    n_thresholds_per_group=50. A no-op presented as a mitigation, with a
    fabricated measure of the work done and no warning of any kind.

    This does not refuse: hard predictions are a legitimate thing to hold, and
    refusing would break that whole path. It measures, warns, and records, so
    the caller can tell "the threshold did not need to move" from "no threshold
    could have moved anything".
    """
    finite = y_prob[np.isfinite(y_prob)]
    n_distinct = int(np.unique(finite).size)
    sufficient = n_distinct >= SCORE_RESOLUTION_FLOOR
    if not sufficient:
        warnings.warn(
            f"{caller}: y_prob holds {n_distinct} distinct value(s), so no threshold "
            f"sweep can separate anything (at least {SCORE_RESOLUTION_FLOOR} are needed). "
            "This is what a hard 0/1 prediction column looks like: the result will be "
            "your own predictions handed back, and any per-group threshold in it is an "
            "artifact of the search floor, not an optimisation. Pass predicted "
            "PROBABILITIES or scores to optimise a threshold.",
            UserWarning,
            stacklevel=3,
        )
    return {"n_distinct_scores": n_distinct, "score_resolution_sufficient": sufficient}


def _record_unscored_rows(y_prob: np.ndarray, *, caller: str) -> Dict[str, Any]:
    """Count the rows ``fit`` will fold into the fairness measurement as REJECTS.

    BGL-S2b. ``predict`` refuses a non-finite score (see
    :func:`_refuse_unscored_rows`), and ``fit`` on the IDENTICAL array said
    nothing. ``y_prob >= thresh`` is False for NaN, so every unscored row
    entered the constraint as a confident 0, a measured rejection nobody
    measured, and the group rates were computed over that. Measured 2026-09-17
    on GroupThresholdOptimizer with 10 NaN scores in 300 rows:

        warnings []                         score_resolution_sufficient True
        is_feasible True                    no COULD NOT CHECK block

    while ``predict`` on the same array raised. ``_record_score_resolution``
    cannot stand in for this: it counts DISTINCT FINITE values, so ten NaN rows
    among three hundred leave its count untouched and its verdict sufficient.

    This measures and discloses rather than refusing, for the reason
    :func:`_refuse_unscored_rows` gives: a training column may legitimately
    hold NaN, and refusing would break that path. What must not happen is the
    silence.
    """
    unscored = ~np.isfinite(np.asarray(y_prob, dtype=float))
    n = int(unscored.sum())
    n_rows = int(unscored.size)
    if n:
        warnings.warn(
            f"{caller}.fit: {n} of {n_rows} row(s) have no usable score (NaN or "
            f"infinite y_prob). A non-finite score compares False against every "
            f"threshold, so these rows enter the fairness measurement as REJECTED, "
            f"which is a decision nobody measured. The constraint verdict is "
            f"reported as not assessed (is_feasible is None) because it was "
            f"computed over those fabricated rejections. Impute or drop the "
            f"unscored rows to get a measured verdict.",
            UserWarning,
            stacklevel=3,
        )
    return {"n_unscored_rows": n, "n_rows_fitted": n_rows}


def _refuse_unscored_rows(y_prob: np.ndarray, *, caller: str) -> None:
    """Refuse to decide about a row that has no score.

    BGL-G05. ``y_prob >= thresh`` is False for NaN under IEEE 754, so a row
    with no score was written 0 - a confident REJECT - and the returned array
    is a bare ``int64`` with no mask and no second field, so the unscored row
    came back byte-identical to a measured rejection. Measured 2026-09-11 on a
    fitted MultiObjectiveThresholdOptimizer (thresholds a=0.5138, b=0.4862):
    40 rows of NaN scores -> ``np.unique(..., return_counts=True)`` was
    ``([0], [40])``, all rejected, with ZERO warnings; and
    ``predict([0.9, nan, 0.1], ['a','a','a'])`` returned ``[1 0 0]``.

    "This person has no score" is could-not-check, and collapsing it into the
    harsher of the two available verdicts fabricates a decision about a person.
    ``predict`` returns an int array, which has no room to carry a third state,
    so the refusal is the third state: raising is how the caller tells it apart
    from a measurement without reading the source. This mirrors the unfitted-
    group refusal in :func:`_refuse_unfitted_groups`.

    Guarded at ``predict`` rather than inside ``_apply_group_thresholds``,
    which ``fit`` also calls: a training column may legitimately hold NaN, and
    ``_record_score_resolution`` already measures and discloses that.
    """
    unscored = ~np.isfinite(np.asarray(y_prob, dtype=float))
    n = int(unscored.sum())
    if n:
        idx = np.flatnonzero(unscored)[:10].tolist()
        raise ValueError(
            f"{caller}.predict: {n} of {len(unscored)} row(s) have no usable score "
            f"(NaN or infinite y_prob), first at index {idx}. Refusing rather than "
            "deciding them: a non-finite score compares False against every "
            "threshold, so these rows would be REJECTED outright, which is a verdict "
            "nobody measured, and the returned int array has no field to say so. "
            "Impute or drop the unscored rows before calling predict()."
        )


def _qualified_feasibility(
    violation: ConstraintViolation,
    unscored: Dict[str, Any],
) -> Optional[bool]:
    """The feasibility verdict, or None when the fit could not honestly give one.

    BGL-S2b. ``ThresholdResult.is_feasible`` is the field a programmatic reader
    consumes (``to_dict``) and the line ``summary()`` prints first, and it was
    the raw ``ConstraintViolation.is_satisfied`` even when the rates behind it
    were computed over rows that carried no score. Those rows entered as
    rejections, so the verdict is about a prediction vector that is part
    measurement and part fabrication. That is a could-not-check, and the third
    state already exists on this field.

    Narrow on purpose. A LOW SCORE RESOLUTION (hard 0/1 predictions) is NOT
    routed here: the sweep cannot separate anything, which is already warned
    and printed as a caveat, but the fairness verdict on the predictions that
    come back IS a real measurement of them. Refusing that one too would turn a
    disclosure into an over-correction, and every degenerate-input test would
    still pass while the capability stopped answering.
    """
    if int(unscored.get("n_unscored_rows", 0)):
        return None
    return violation.is_satisfied


def _refuse_unfitted_groups(
    sensitive_attr: np.ndarray,
    group_thresholds: Dict[str, float],
) -> None:
    """Refuse to decide about a group no threshold was fitted for.

    A group present in the data but ABSENT from the fitted thresholds is
    refused, not scored. ``_apply_group_thresholds`` only ever writes rows
    whose group is in the fitted dict, so an unseen group kept the
    ``np.zeros`` seed and was REJECTED ENTIRELY, silently and with no warning:
    measured 2026-09-09, fitting on groups a and b and predicting with a group
    c gave group c an acceptance rate of 0.000. Measured again 2026-09-11 on
    MultiObjectiveThresholdOptimizer, which did not carry this guard: 30 rows
    of an unfitted group 'c' -> all 0, including the row at p=0.99, with no
    warning and no error. A threshold nobody fitted is could-not-check, and
    answering it with the harshest possible verdict is the defect this audit
    exists to remove.
    """
    seen = set(np.unique(sensitive_attr.astype(str)))
    unfitted = sorted(str(g) for g in seen - set(group_thresholds))
    if unfitted:
        raise ValueError(
            f"No threshold was fitted for group(s) {unfitted}. Fitted groups are "
            f"{sorted(group_thresholds)}. Refusing rather than scoring them: with no "
            "threshold these rows would be rejected outright, which is a verdict "
            "nobody measured. Refit including these groups, or filter them out "
            "before calling predict()."
        )


SUPPORTED_OBJECTIVES = frozenset(
    {"accuracy", "precision", "recall", "f1_score", "balanced_accuracy"}
)


def _compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard performance metrics."""
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    accuracy = np.average(y_true == y_pred, weights=sample_weight)

    # Confusion matrix components.
    # NOTE: the mask must be parenthesized before multiplying by the weights:
    # `*` binds tighter than `&`, so `a & b * w` is `a & (b * w)` which raises
    # TypeError (bitwise_and between bool and float) for every weighted call.
    tp = np.sum(((y_true == 1) & (y_pred == 1)) * sample_weight)
    tn = np.sum(((y_true == 0) & (y_pred == 0)) * sample_weight)
    fp = np.sum(((y_true == 0) & (y_pred == 1)) * sample_weight)
    fn = np.sum(((y_true == 1) & (y_pred == 0)) * sample_weight)

    # Precision, recall, F1.
    #
    # READINESS-6, 2026-09-10. Each of these was `... if denominator > 0 else
    # 0.0`, and 0.0 is a MEASUREMENT: it says "every positive call was wrong",
    # or "no positive was ever found". An empty denominator says something
    # different and much weaker: the quantity was never defined on this data.
    #
    # Measured on this repo before the change:
    #
    #   y_pred all zeros            -> precision 0.0, though no positive was
    #                                  ever predicted, so precision has no value
    #   y_true all ones, y_pred all -> a PERFECT classifier reported
    #   ones                           balanced_accuracy 0.5, "no better than
    #                                  chance", because tnr was undefined (no
    #                                  negatives exist) and contributed 0
    #   y_true all zeros            -> recall 0.0 and f1_score 0.0 for a
    #                                  PERFECT classifier, so optimising for
    #                                  f1_score ranked every candidate at 0.0
    #                                  and the frontier was decided by a metric
    #                                  nobody could compute
    #
    # The third one is the worst in this package specifically, because these
    # metrics are also computed PER GROUP (analyzer.analyze_threshold_impact).
    # A small group with no positive labels got a fabricated recall of 0.0 and
    # read as catastrophically underserved, when nothing about it was measured.
    #
    # NOTE ON SKLEARN. sklearn's default is `zero_division=0` with a warning, so
    # this deliberately DIFFERS from what a reader may expect. The divergence is
    # the point: this library's contract is three states, and a NaN cannot be
    # mistaken for a graded zero by a downstream comparison the way 0.0 can.
    undefined = []
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = float("nan")
        undefined.append("precision (no positive prediction was made)")
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = float("nan")
        undefined.append("recall (the data holds no positive label)")
    # F1 needs more care than "NaN if either input is NaN", because that would
    # discard a real finding. F1 is 2PR/(P+R): the NUMERATOR is zero whenever
    # either component is a MEASURED zero, whatever the other one is, so F1 is a
    # measured zero too. A model that predicts no positives has recall 0, which
    # is measured and means it found NONE of the real positives; reporting NaN
    # there would hide a complete failure on the positive class behind a
    # could-not-check. Only when no measured component pins the value down is F1
    # genuinely undefined.
    if math.isfinite(precision) and math.isfinite(recall):
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif (math.isfinite(precision) and precision == 0.0) or (
        math.isfinite(recall) and recall == 0.0
    ):
        f1 = 0.0
    else:
        f1 = float("nan")
        undefined.append("f1_score (neither precision nor recall could pin it down)")

    # Balanced accuracy
    tpr = recall
    if (tn + fp) > 0:
        tnr = tn / (tn + fp)
    else:
        tnr = float("nan")
        undefined.append("balanced_accuracy (the data holds no negative label)")
    balanced_accuracy = (tpr + tnr) / 2

    if undefined:
        warnings.warn(
            f"_compute_performance_metrics: {len(undefined)} metric(s) are not defined "
            f"on this data and are returned as NaN, NOT as 0.0: "
            f"{'; '.join(undefined)}. A NaN here means the quantity has no value, "
            f"which is different from a measured zero. Any ranking, Pareto comparison "
            f"or report over these must exclude them rather than treat them as the "
            f"worst possible score.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
    }


class _ObjectiveScan:
    """Track whether the objective was ever MEASURED during a threshold search.

    BGL-S2 (2026-09-16). When the objective cannot be computed on the data it is
    NaN, every ``>`` comparison against it is False, and the grid search
    silently degrades to "keep the first feasible grid point". The artifact
    still recorded ``optimization_details['objective'] = 'balanced_accuracy'``
    and presented ``group_thresholds`` as an optimisation result.

    Measured before this change, ``GroupThresholdOptimizer(n_thresholds=50)``
    fitted on 200 rows whose ``y_true`` was all ones (so balanced_accuracy is
    undefined):

        group_thresholds {'a': 0.05, 'b': 0.05}
        optimization_details {'objective': 'balanced_accuracy', ...}

    and scanning the SAME grid in REVERSE order returned {'a': 0.95, 'b': 0.95}
    on identical data, while the healthy control was order-invariant
    ({'a': 0.6194, 'b': 0.5459} in both directions). The returned thresholds
    were a scan-order artifact, not an optimum. 2501 warnings did fire saying
    balanced_accuracy was NaN, so the undefined METRIC was disclosed; what was
    not disclosed is that the SELECTION had no basis.

    It also fixes the MIXED case, which is worse than the all-NaN one. The
    incumbent's score used to be assigned unconditionally, so an unmeasurable
    first candidate put NaN into ``best_objective`` and every later candidate
    with a REAL objective then lost ``measured > nan`` (False) and could never
    be selected. An unmeasured value may never become the bar a measurement has
    to clear.
    """

    def __init__(self) -> None:
        self.n_candidates = 0
        self.n_measured = 0

    def observe(self, value: Any) -> bool:
        """Count one candidate; True when its objective is a real number."""
        self.n_candidates += 1
        if is_measured(value):
            self.n_measured += 1
            return True
        return False

    @property
    def any_measured(self) -> bool:
        return self.n_measured > 0

    def details(self) -> Dict[str, Any]:
        """The disclosure that goes into ``optimization_details``."""
        return {
            "objective_measured": self.any_measured,
            "n_objective_candidates": self.n_candidates,
            "n_objective_measured": self.n_measured,
        }

    def warn_if_unmeasured(self, caller: str, objective: str) -> None:
        if self.any_measured or not self.n_candidates:
            return
        warnings.warn(
            f"{caller}: the objective {objective!r} could not be computed at ANY of the "
            f"{self.n_candidates} candidate threshold(s), so it decided nothing. The "
            f"thresholds returned were selected by constraint violation alone and are "
            f"NOT an optimum: on this data a different scan order would return "
            f"different thresholds. optimization_details['objective_measured'] is False.",
            UserWarning,
            stacklevel=3,
        )


class _ViolationScan:
    """Track whether the CONSTRAINT was ever measured during a threshold search.

    BGL-S2b, the sibling of :class:`_ObjectiveScan` three lines down in the same
    loops. When no candidate is feasible the search falls back to
    "minimise the violation", and that comparison is
    ``result["violation"].violation < best_violation.violation``. A violation
    that could not be measured is NaN, every ``<`` against NaN is False, and so
    the FIRST grid point latches and is returned as the least violating
    configuration. Measured 2026-09-17 on GroupThresholdOptimizer with
    equalized odds and y_true all ones (the FPR arm has no rows, so every
    candidate's violation is NaN):

        forward grid  -> {'a': 0.05, 'b': 0.05}
        reversed grid -> {'a': 0.95, 'b': 0.95}

    on identical data, while ``optimization_details`` reported
    ``objective_measured`` and said nothing about the constraint. The returned
    thresholds were a scan-order artifact of a constraint nobody measured.

    Two things change. A measured violation may always displace an unmeasured
    incumbent (so a mixed scan cannot be blocked by a NaN that arrived first),
    and the count is disclosed, so a reader can tell a real minimisation from a
    latch.
    """

    def __init__(self) -> None:
        self.n_candidates = 0
        self.n_measured = 0

    def observe(self, violation: float) -> bool:
        """Count one candidate; True when its violation is a real number."""
        self.n_candidates += 1
        if is_measured(violation):
            self.n_measured += 1
            return True
        return False

    @property
    def any_measured(self) -> bool:
        return self.n_measured > 0

    def details(self) -> Dict[str, Any]:
        """The disclosure that goes into ``optimization_details``."""
        return {
            "constraint_measured": self.any_measured,
            "n_violation_candidates": self.n_candidates,
            "n_violation_measured": self.n_measured,
        }

    def warn_if_unmeasured(self, caller: str, constraint: str) -> None:
        if self.any_measured or not self.n_candidates:
            return
        warnings.warn(
            f"{caller}: the constraint {constraint!r} could not be measured at ANY of "
            f"the {self.n_candidates} candidate threshold(s), so it ranked nothing. No "
            f"candidate was feasible and none could be shown to violate less than "
            f"another: the thresholds returned are the first the scan happened to see, "
            f"NOT a minimum. optimization_details['constraint_measured'] is False.",
            UserWarning,
            stacklevel=3,
        )


def _prefer_violation(candidate: float, incumbent: float) -> bool:
    """True when ``candidate`` should displace ``incumbent`` as least violating.

    The NaN rule in one place, because the identical comparison appears in
    three fit loops and fixing one would move the defect to its siblings:

    * an unmeasured candidate never displaces anything, measured or not;
    * a measured candidate ALWAYS displaces an unmeasured incumbent, which is
      the mixed case ``<`` got wrong in the direction that keeps the
      unmeasurable one;
    * otherwise the smaller measured violation wins, as before.
    """
    if not is_measured(candidate):
        return False
    if not is_measured(incumbent):
        return True
    return bool(candidate < incumbent)


class ThresholdOptimizer(BaseThresholdOptimizer):
    """
    Single threshold optimizer that finds the best global threshold.

    This optimizer finds a single threshold that best satisfies the fairness
    constraint while maintaining good overall performance.

    Example:
        >>> optimizer = ThresholdOptimizer(constraint='demographic_parity', tolerance=0.05)
        >>> optimizer.fit(y_true, y_prob, gender)
        >>> y_pred = optimizer.predict(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: threshold_optimization. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        constraint: Union[str, FairnessConstraintType, ThresholdConstraint] = "demographic_parity",
        tolerance: float = 0.05,
        n_thresholds: int = 100,
        objective: str = "accuracy",
    ):
        """
        Initialize the single threshold optimizer.

        Args:
            constraint: Fairness constraint to satisfy
            tolerance: Maximum allowed constraint violation
            n_thresholds: Number of threshold values to search. At least 2, or the
                search cannot choose one candidate over another and what comes back
                is a seed (see _require_searchable_grid).
            objective: Performance metric to optimize ('accuracy', 'f1_score', 'balanced_accuracy')
        """
        super().__init__(constraint, tolerance)
        # BGL3-PP3: a grid of fewer than two candidates returns this class's
        # threshold seed, or the grid's lower bound, as though it were chosen.
        self.n_thresholds = _require_searchable_grid(n_thresholds, caller=type(self).__name__)
        self.objective = objective
        # T-02 recurred here. MultiObjectiveThresholdOptimizer refuses an
        # objective it cannot compute; this class and its group-wise sibling
        # did not. They read `metrics.get(self.objective, metrics["accuracy"])`,
        # so an unknown name silently optimised ACCURACY instead, and the
        # result artifact then recorded the name that was never used. Measured
        # 2026-09-09: objective='expected_cost' and objective='total_nonsense'
        # both produced the accuracy-optimal threshold 0.1585, and
        # optimization_details['objective'] said 'expected_cost'. An artifact
        # that names an objective it did not optimise is a claim, not a record.
        if self.objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"{type(self).__name__} cannot compute objective "
                f"{self.objective!r}. Supported objectives are "
                f"{sorted(SUPPORTED_OBJECTIVES)}. An unsupported name would be "
                f"silently replaced by accuracy while the result still reported "
                f"{self.objective!r}."
            )

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "ThresholdOptimizer":
        """
        Find the optimal global threshold.

        Searches through threshold values to find one that satisfies the
        fairness constraint while maximizing the objective metric.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        resolution = _record_score_resolution(y_prob, caller=type(self).__name__)
        unscored = _record_unscored_rows(y_prob, caller=type(self).__name__)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight)

        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_threshold = 0.5
        best_objective = -np.inf
        best_violation = None

        feasible_found = False
        scan = _ObjectiveScan()
        vscan = _ViolationScan()

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            violation = compute_constraint_violation(
                y_true,
                y_pred,
                sensitive_attr,
                self.constraint.constraint_type,
                self.constraint.tolerance,
            )

            metrics = _compute_performance_metrics(y_true, y_pred, sample_weight)
            obj_value = metrics[self.objective]
            # BGL-S2: an objective that was not measured may neither win a
            # comparison nor become the incumbent's score. See _ObjectiveScan.
            obj_measured = scan.observe(obj_value)
            incumbent_score = obj_value if obj_measured else -np.inf
            vscan.observe(violation.violation)

            # Prefer feasible solutions, then optimize objective
            if violation.is_satisfied:
                if not feasible_found or (obj_measured and obj_value > best_objective):
                    best_threshold = thresh
                    best_objective = incumbent_score
                    best_violation = violation
                    feasible_found = True
            elif not feasible_found:
                # No feasible solution found yet, minimize violation. BGL-S2b:
                # an unmeasured violation may neither win this comparison nor
                # become the bar a measurement has to beat. See
                # _prefer_violation.
                if best_violation is None or _prefer_violation(
                    violation.violation, best_violation.violation
                ):
                    best_threshold = thresh
                    best_objective = incumbent_score
                    best_violation = violation

        scan.warn_if_unmeasured(f"{type(self).__name__}.fit", self.objective)
        vscan.warn_if_unmeasured(
            f"{type(self).__name__}.fit", self.constraint.constraint_type.value
        )

        y_pred_final = (y_prob >= best_threshold).astype(int)
        final_metrics = _compute_performance_metrics(y_true, y_pred_final, sample_weight)
        final_violation = compute_constraint_violation(
            y_true,
            y_pred_final,
            sensitive_attr,
            self.constraint.constraint_type,
            self.constraint.tolerance,
        )

        unique_groups = np.unique(sensitive_attr)
        group_thresholds = {str(g): best_threshold for g in unique_groups}

        self.result_ = ThresholdResult(
            global_threshold=best_threshold,
            group_thresholds=group_thresholds,
            constraint_violations=[final_violation],
            performance_metrics=final_metrics,
            is_feasible=_qualified_feasibility(final_violation, unscored),
            optimization_details={
                "n_thresholds_searched": self.n_thresholds,
                "objective": self.objective,
                **scan.details(),
                **vscan.details(),
                **resolution,
                **unscored,
            },
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """Apply the optimized threshold."""
        if not self.is_fitted:
            raise RuntimeError("Optimizer must be fitted before calling predict()")

        # fit() always assigns result_ (and a float global_threshold) before
        # setting is_fitted, so both are non-None once fitted.
        assert self.result_ is not None and self.result_.global_threshold is not None
        y_prob = coerce_to_array(y_prob)
        # BGL-G05. A row with no score is could-not-check, not a rejection.
        _refuse_unscored_rows(y_prob, caller=type(self).__name__)
        return (y_prob >= self.result_.global_threshold).astype(int)


class GroupThresholdOptimizer(BaseThresholdOptimizer):
    """
    Group-specific threshold optimizer.

    This optimizer finds different thresholds for each demographic group
    to satisfy fairness constraints, potentially achieving better trade-offs
    than a single global threshold.

    Example:
        >>> optimizer = GroupThresholdOptimizer(constraint='equalized_odds', tolerance=0.05)
        >>> optimizer.fit(y_true, y_prob, gender)
        >>> print(optimizer.result_.group_thresholds)
        {'male': 0.42, 'female': 0.58}
        >>> y_pred = optimizer.predict(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: group_threshold. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        constraint: Union[str, FairnessConstraintType, ThresholdConstraint] = "equalized_odds",
        tolerance: float = 0.05,
        n_thresholds: int = 50,
        objective: str = "balanced_accuracy",
        grid_search: bool = True,
    ):
        """
        Initialize the group threshold optimizer.

        Args:
            constraint: Fairness constraint to satisfy
            tolerance: Maximum allowed constraint violation
            n_thresholds: Number of threshold values per group. At least 2, or the
                grid search cannot choose one candidate over another (see
                _require_searchable_grid).
            objective: Performance metric to optimize
            grid_search: Must be True. Grid search is the only optimisation
                this class has; a gradient-based path was documented but
                never built, and False is refused with NotImplementedError.
        """
        super().__init__(constraint, tolerance)
        # BGL3-PP3: see _require_searchable_grid. Guarded in each of the three
        # classes because each one carries its own default threshold seed, so
        # fixing one would leave the same silent no-search in its siblings.
        self.n_thresholds = _require_searchable_grid(n_thresholds, caller=type(self).__name__)
        self.objective = objective
        # F10 (2026-09-09): `grid_search` was stored and never read, so
        # grid_search=False promised gradient-based optimisation and ran the
        # same grid search. Measured: True and False produced identical group
        # thresholds ({'A': 0.55, 'B': 0.55} on the audit data). An option
        # that selects a method which does not exist is refused, not accepted.
        if grid_search is not True:
            raise NotImplementedError(
                f"{type(self).__name__} only implements grid search; there is no "
                f"gradient-based optimisation path. Pass grid_search=True (the "
                f"default) or omit the argument."
            )
        # T-02 recurred here. MultiObjectiveThresholdOptimizer refuses an
        # objective it cannot compute; this class and its group-wise sibling
        # did not. They read `metrics.get(self.objective, metrics["accuracy"])`,
        # so an unknown name silently optimised ACCURACY instead, and the
        # result artifact then recorded the name that was never used. Measured
        # 2026-09-09: objective='expected_cost' and objective='total_nonsense'
        # both produced the accuracy-optimal threshold 0.1585, and
        # optimization_details['objective'] said 'expected_cost'. An artifact
        # that names an objective it did not optimise is a claim, not a record.
        if self.objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"{type(self).__name__} cannot compute objective "
                f"{self.objective!r}. Supported objectives are "
                f"{sorted(SUPPORTED_OBJECTIVES)}. An unsupported name would be "
                f"silently replaced by accuracy while the result still reported "
                f"{self.objective!r}."
            )
        self.grid_search = grid_search

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "GroupThresholdOptimizer":
        """
        Find optimal thresholds for each group.

        Uses grid search to find the combination of group-specific thresholds
        that satisfies the fairness constraint while maximizing performance.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        resolution = _record_score_resolution(y_prob, caller=type(self).__name__)
        unscored = _record_unscored_rows(y_prob, caller=type(self).__name__)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight)
        else:
            sample_weight = np.ones(len(y_true))

        unique_groups = sorted([str(g) for g in np.unique(sensitive_attr)])
        thresholds = np.linspace(0.05, 0.95, self.n_thresholds)

        best_thresholds = {g: 0.5 for g in unique_groups}
        best_objective = -np.inf
        best_violation = None
        feasible_found = False
        # BGL-S2: ONE scan for both branches below, declared above the branch
        # selection because the identical latch exists in each of them and
        # fixing only the 2-group path would move the defect to its sibling.
        scan = _ObjectiveScan()
        vscan = _ViolationScan()

        if len(unique_groups) == 2:
            # Efficient 2D grid search
            for t1 in thresholds:
                for t2 in thresholds:
                    group_thresholds = {unique_groups[0]: t1, unique_groups[1]: t2}
                    result = self._evaluate_thresholds(
                        y_true, y_prob, sensitive_attr, sample_weight, group_thresholds
                    )
                    obj_measured = scan.observe(result["objective"])
                    incumbent_score = result["objective"] if obj_measured else -np.inf
                    vscan.observe(result["violation"].violation)

                    if result["is_feasible"]:
                        if not feasible_found or (
                            obj_measured and result["objective"] > best_objective
                        ):
                            best_thresholds = group_thresholds.copy()
                            best_objective = incumbent_score
                            best_violation = result["violation"]
                            feasible_found = True
                    elif not feasible_found:
                        # BGL-S2b: see _prefer_violation. `<` alone latched the
                        # first grid point whenever every violation was NaN.
                        if best_violation is None or _prefer_violation(
                            result["violation"].violation, best_violation.violation
                        ):
                            best_thresholds = group_thresholds.copy()
                            best_objective = incumbent_score
                            best_violation = result["violation"]
        else:
            # For more groups, use iterative refinement
            # Start with global optimal, then adjust per group
            global_opt = ThresholdOptimizer(
                self.constraint, self.constraint.tolerance, self.n_thresholds, self.objective
            )
            global_opt.fit(
                y_true=y_true,
                y_prob=y_prob,
                sensitive_attr=sensitive_attr,
                sample_weight=sample_weight,
            )
            # ThresholdOptimizer.fit always sets result_ with a float
            # global_threshold, so both are non-None here.
            assert (
                global_opt.result_ is not None and global_opt.result_.global_threshold is not None
            )
            best_thresholds = {g: global_opt.result_.global_threshold for g in unique_groups}

            for _ in range(3):  # iterations
                for group in unique_groups:
                    for thresh in thresholds:
                        test_thresholds = best_thresholds.copy()
                        test_thresholds[group] = thresh

                        result = self._evaluate_thresholds(
                            y_true, y_prob, sensitive_attr, sample_weight, test_thresholds
                        )
                        obj_measured = scan.observe(result["objective"])
                        incumbent_score = result["objective"] if obj_measured else -np.inf
                        vscan.observe(result["violation"].violation)

                        if result["is_feasible"]:
                            if not feasible_found or (
                                obj_measured and result["objective"] > best_objective
                            ):
                                best_thresholds = test_thresholds.copy()
                                best_objective = incumbent_score
                                best_violation = result["violation"]
                                feasible_found = True
                        elif not feasible_found:
                            # BGL-S2b: see _prefer_violation. This branch is the
                            # >2 group sibling of the one above, and carried the
                            # identical latch.
                            if best_violation is None or _prefer_violation(
                                result["violation"].violation, best_violation.violation
                            ):
                                best_thresholds = test_thresholds.copy()
                                best_violation = result["violation"]

        scan.warn_if_unmeasured(f"{type(self).__name__}.fit", self.objective)
        vscan.warn_if_unmeasured(
            f"{type(self).__name__}.fit", self.constraint.constraint_type.value
        )

        y_pred_final = self._apply_group_thresholds(y_prob, sensitive_attr, best_thresholds)
        final_metrics = _compute_performance_metrics(y_true, y_pred_final, sample_weight)
        final_violation = compute_constraint_violation(
            y_true,
            y_pred_final,
            sensitive_attr,
            self.constraint.constraint_type,
            self.constraint.tolerance,
        )

        self.result_ = ThresholdResult(
            global_threshold=None,
            group_thresholds=best_thresholds,
            constraint_violations=[final_violation],
            performance_metrics=final_metrics,
            is_feasible=_qualified_feasibility(final_violation, unscored),
            optimization_details={
                "n_thresholds_per_group": self.n_thresholds,
                "objective": self.objective,
                "n_groups": len(unique_groups),
                **scan.details(),
                **vscan.details(),
                **resolution,
                **unscored,
            },
        )

        self.is_fitted = True
        return self

    def _evaluate_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attr: np.ndarray,
        sample_weight: np.ndarray,
        group_thresholds: Dict[str, float],
    ) -> Dict:
        """Evaluate a set of group thresholds."""
        y_pred = self._apply_group_thresholds(y_prob, sensitive_attr, group_thresholds)

        violation = compute_constraint_violation(
            y_true,
            y_pred,
            sensitive_attr,
            self.constraint.constraint_type,
            self.constraint.tolerance,
        )

        metrics = _compute_performance_metrics(y_true, y_pred, sample_weight)

        return {
            "is_feasible": violation.is_satisfied,
            "violation": violation,
            "objective": metrics[self.objective],
            "metrics": metrics,
        }

    def _apply_group_thresholds(
        self,
        y_prob: np.ndarray,
        sensitive_attr: np.ndarray,
        group_thresholds: Dict[str, float],
    ) -> np.ndarray:
        """Apply group-specific thresholds.

        A group present here but ABSENT from the fitted thresholds is refused,
        not scored. The loop below only ever writes rows whose group is in the
        fitted dict, so an unseen group used to keep the ``np.zeros`` seed and
        was REJECTED ENTIRELY, silently and with no warning: measured
        2026-09-09, fitting on groups a and b and predicting with a group c
        gave group c an acceptance rate of 0.000. A threshold nobody fitted is
        could-not-check, and answering it with the harshest possible verdict is
        the defect this audit exists to remove.
        """
        _refuse_unfitted_groups(sensitive_attr, group_thresholds)
        y_pred = np.zeros(len(y_prob), dtype=int)

        for group, thresh in group_thresholds.items():
            mask = sensitive_attr.astype(str) == group
            y_pred[mask] = (y_prob[mask] >= thresh).astype(int)

        return y_pred

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """Apply the optimized group-specific thresholds."""
        if not self.is_fitted:
            raise RuntimeError("Optimizer must be fitted before calling predict()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        # BGL-G05. A row with no score is could-not-check, not a rejection.
        _refuse_unscored_rows(y_prob, caller=type(self).__name__)

        # fit() always assigns result_ before setting is_fitted.
        assert self.result_ is not None
        return self._apply_group_thresholds(y_prob, sensitive_attr, self.result_.group_thresholds)


class MultiObjectiveThresholdOptimizer(BaseThresholdOptimizer):
    """
    Multi-objective threshold optimizer using Pareto optimization.

    Finds the Pareto frontier of threshold configurations that trade off
    between fairness and performance, allowing users to select their
    preferred operating point.

    Example:
        >>> optimizer = MultiObjectiveThresholdOptimizer(
        ...     constraint='equalized_odds',
        ...     objectives=['accuracy', 'f1_score']
        ... )
        >>> optimizer.fit(y_true, y_prob, gender)
        >>> pareto_points = optimizer.pareto_frontier_
        >>> # Select a point from the frontier
        >>> optimizer.select_point(index=0)
        >>> y_pred = optimizer.predict(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: multi_objective_threshold. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        constraint: Union[str, FairnessConstraintType, ThresholdConstraint] = "equalized_odds",
        tolerance: float = 0.05,
        n_thresholds: int = 30,
        objectives: Optional[List[str]] = None,
    ):
        """
        Initialize the multi-objective optimizer.

        Args:
            constraint: Fairness constraint to satisfy
            tolerance: Maximum allowed constraint violation
            n_thresholds: Number of threshold values per group. At least 2, or the
                candidate sweep produces nothing (see _require_searchable_grid).
            objectives: Performance metrics to optimize (default: ['accuracy', 'f1_score'])
        """
        super().__init__(constraint, tolerance)
        # BGL3-PP3: with 0 the candidate sweep produces nothing, and this class
        # then falls back to a 50 point GroupThresholdOptimizer grid under a
        # reason line that blames the data ("no threshold pair in the sweep
        # produced a usable candidate") for the caller's own argument.
        self.n_thresholds = _require_searchable_grid(n_thresholds, caller=type(self).__name__)
        self.objectives = objectives or ["accuracy", "f1_score"]

        # An objective this class cannot COMPUTE is not a weak objective, it is
        # no objective at all. The dominance test below reads each one with
        # `.get(obj, 0)` on both sides, so an unknown name makes every
        # comparison `0 >= 0` (true) and `0 > 0` (false): nothing dominates
        # anything, and the FULL threshold sweep is returned dressed as a Pareto
        # frontier. Measured 2026-09-09 on 400 rows with n_thresholds=8:
        # objectives=['accuracy'] gave 1 Pareto point, objectives=['expected_cost']
        # gave 5, with no warning. The failure signature is the opposite of a
        # refusal: the less the optimizer understands, the more results it reports.
        #
        # Checked here rather than in fit() so the error names the mistake at the
        # line that made it. T-02 in the fifth-iteration audit; found from outside
        # the library, by a catalogue row claiming an objective this class does
        # not have.
        unknown = [o for o in self.objectives if o not in SUPPORTED_OBJECTIVES]
        if unknown:
            raise ValueError(
                f"MultiObjectiveThresholdOptimizer cannot compute {unknown}. "
                f"Supported objectives are {sorted(SUPPORTED_OBJECTIVES)}. An "
                f"unsupported name would make every candidate threshold look "
                f"Pareto-optimal, because nothing can dominate anything on a "
                f"value that is missing from both sides of the comparison."
            )
        self.pareto_frontier_: List[ThresholdResult] = []
        self._selected_index = 0

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "MultiObjectiveThresholdOptimizer":
        """
        Find the Pareto frontier of threshold configurations.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        resolution = _record_score_resolution(y_prob, caller=type(self).__name__)
        unscored = _record_unscored_rows(y_prob, caller=type(self).__name__)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight)
        else:
            sample_weight = np.ones(len(y_true))

        unique_groups = sorted([str(g) for g in np.unique(sensitive_attr)])
        thresholds = np.linspace(0.1, 0.9, self.n_thresholds)

        candidates: List[Dict[str, Any]] = []

        if len(unique_groups) == 2:
            for t1 in thresholds:
                for t2 in thresholds:
                    group_thresholds = {unique_groups[0]: t1, unique_groups[1]: t2}
                    y_pred = self._apply_group_thresholds(y_prob, sensitive_attr, group_thresholds)

                    violation = compute_constraint_violation(
                        y_true,
                        y_pred,
                        sensitive_attr,
                        self.constraint.constraint_type,
                        self.constraint.tolerance,
                    )

                    metrics = _compute_performance_metrics(y_true, y_pred, sample_weight)

                    candidates.append(
                        {
                            "group_thresholds": group_thresholds,
                            "violation": violation,
                            "metrics": metrics,
                            "is_feasible": violation.is_satisfied,
                        }
                    )

        vscan = _ViolationScan()
        for c in candidates:
            vscan.observe(c["violation"].violation)
        vscan.warn_if_unmeasured(
            f"{type(self).__name__}.fit", self.constraint.constraint_type.value
        )

        # Filter to feasible solutions (or keep all if none feasible)
        feasible = [c for c in candidates if c["is_feasible"]]
        if not feasible:
            # Keep least violating solutions. BGL-S2b: NaN sorts wherever the
            # algorithm happens to put it, so an unmeasurable violation could
            # take one of the ten places away from a measured one. Sort
            # unmeasured LAST, and never let one displace a measurement.
            candidates.sort(
                key=lambda c: (
                    not is_measured(c["violation"].violation),
                    c["violation"].violation if is_measured(c["violation"].violation) else 0.0,
                )
            )
            feasible = candidates[: min(10, len(candidates))]

        # Find Pareto frontier.
        #
        # READINESS-6, 2026-09-10. This compared `.get(obj, 0)`, the variable-key
        # neutral default over an objective that may not have been computed at
        # all, and 0 is not neutral for these five metrics: it is the WORST
        # attainable score. So an objective nobody could compute made its own
        # candidate look maximally bad and dropped it off the frontier, and made
        # the OTHER candidate fail to dominate for the same reason. Both
        # directions were wrong and neither was a reading of anything.
        #
        # `_compute_performance_metrics` now returns NaN for an undefined
        # metric, and every `>=` and `>` answers False for NaN, so an unguarded
        # comparison would silently place EVERY candidate on the frontier.
        # Compare only the objectives measured for BOTH candidates, treat a pair
        # with no shared measured objective as incomparable, and disclose it.
        def _shared_measured(a, b):
            return [
                obj
                for obj in self.objectives
                if is_measured(a.get(obj)) and is_measured(b.get(obj))
            ]

        pareto = []
        for candidate in feasible:
            is_dominated = False
            for other in feasible:
                if other is candidate:
                    continue
                shared = _shared_measured(other["metrics"], candidate["metrics"])
                if not shared:
                    # Nothing to compare on: NOT domination in either direction.
                    continue
                dominates = all(
                    other["metrics"][obj] >= candidate["metrics"][obj] for obj in shared
                ) and any(other["metrics"][obj] > candidate["metrics"][obj] for obj in shared)
                if dominates:
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(candidate)

        # BGL-S2c. WHICH of the requested objectives the frontier actually
        # ranked on, recorded on the artifact rather than only shouted once.
        #
        # Two defects, opposite directions, same three lines. Measured
        # 2026-09-17 on 200 rows, two groups, n_thresholds=8:
        #
        #   objectives=['accuracy']          -> frontier size 1
        #   objectives=['balanced_accuracy'] -> frontier size 10
        #   (all-ones y_true, so balanced_accuracy is undefined at every one
        #   of the 64 candidates)
        #
        # and BOTH results carried the identical optimization_details, ending
        # 'pareto_frontier_computed': True, with nothing in to_dict() or
        # summary() to tell them apart. Ten configurations were reported as
        # non-dominated because no pair had a single objective measured on
        # both sides, so nothing could dominate anything: the T-02 signature
        # this class already refuses for an UNKNOWN objective name, arriving
        # instead through a known objective that has no value on this data.
        # The only disclosure was a warning, which is gone by the time anyone
        # reads the result. This is the BGL-S2 shape one layer earlier: it
        # never reached optimization_details, so a programmatic reader could
        # not even ask.
        #
        # The second defect is the reverse one, and it is in the warning
        # itself: any() over an EMPTY candidate list is False, so when the
        # sweep never ran at all (it runs for exactly 2 groups) every
        # objective was declared uncomputable "on this data". Measured on one
        # group: "2 of 2 objective(s) could not be computed on this data for
        # ANY candidate (accuracy, f1_score)" over data where accuracy is
        # trivially computable. Nothing was measured because nothing was
        # scanned, which the no-frontier warning below says correctly, so this
        # one stays silent when there were no candidates.
        measured_objectives = sorted(
            obj
            for obj in self.objectives
            if any(is_measured(c["metrics"].get(obj)) for c in feasible)
        )
        unmeasured_objectives = sorted(set(self.objectives) - set(measured_objectives))
        objective_disclosure: Dict[str, Any] = {
            "objectives": list(self.objectives),
            "objectives_measured": measured_objectives,
            "objectives_unmeasured": unmeasured_objectives if feasible else [],
            "n_pareto_candidates": len(feasible),
            # False when the domination test had nothing to compare on, so
            # every candidate came back non-dominated by default.
            "pareto_frontier_ranked": bool(feasible) and bool(measured_objectives),
        }
        if feasible and unmeasured_objectives:
            warnings.warn(
                f"MultiObjectiveThresholdOptimizer: {len(unmeasured_objectives)} of "
                f"{len(self.objectives)} objective(s) could not be computed on this data "
                f"for ANY candidate ({', '.join(unmeasured_objectives)}), so the Pareto "
                f"frontier rests only on the remaining objective(s). It is NOT a frontier "
                f"over the objectives that were requested.",
                UserWarning,
                stacklevel=2,
            )

        self.pareto_frontier_ = []
        for p in pareto:
            result = ThresholdResult(
                global_threshold=None,
                group_thresholds=p["group_thresholds"],
                constraint_violations=[p["violation"]],
                performance_metrics=p["metrics"],
                is_feasible=_qualified_feasibility(p["violation"], unscored),
                # BGL-G05. Stamped on BOTH paths on purpose: a flag that only
                # exists when things went wrong is a flag a caller cannot
                # distinguish from an old result object.
                optimization_details={
                    **resolution,
                    **unscored,
                    **vscan.details(),
                    # BGL-S2c. Which objectives the frontier ranked on, on the
                    # artifact itself, because the warning is gone by the time
                    # anyone reads this.
                    **objective_disclosure,
                    "pareto_frontier_computed": True,
                },
            )
            self.pareto_frontier_.append(result)

        # Select best balanced point as default.
        #
        # READINESS-6: `sum(.get(obj, 0))` scored a candidate whose objective was
        # never computed as if it had achieved the worst possible value on it,
        # and this sum decides `result_`, the thresholds a caller deploys. Sum
        # the MEASURED objectives only, and prefer the candidate scored on MORE
        # of them, because a sum over three objectives is not comparable with a
        # sum over two.
        if self.pareto_frontier_:
            best_idx = None
            best_score = -np.inf
            best_n = -1
            for i, result in enumerate(self.pareto_frontier_):
                values = [
                    result.performance_metrics[obj]
                    for obj in self.objectives
                    if is_measured(result.performance_metrics.get(obj))
                ]
                if not values:
                    continue
                score, n = sum(values), len(values)
                if (n, score) > (best_n, best_score):
                    best_n, best_score, best_idx = n, score, i
            if best_idx is None:
                warnings.warn(
                    "MultiObjectiveThresholdOptimizer: not one point on the Pareto "
                    "frontier has a measured value for any requested objective, so no "
                    "best point could be selected. Falling back to the first point by "
                    "position, which is an ARBITRARY choice and not an optimum. Inspect "
                    "pareto_frontier_ rather than trusting result_.",
                    UserWarning,
                    stacklevel=2,
                )
                best_idx = 0
            self._selected_index = best_idx
            self.result_ = self.pareto_frontier_[best_idx]
        else:
            # Fallback to simple optimizer.
            #
            # BGL-G05. The candidate sweep above runs ONLY for exactly two
            # groups, so with 1 group or 3+ groups `candidates` is empty and
            # this branch runs. The single GroupThresholdOptimizer answer was
            # then stored in `pareto_frontier_` and presented as a one-point
            # Pareto frontier: `len(opt.pareto_frontier_) == 1` read as "the
            # sweep found exactly one non-dominated configuration", when in
            # fact no sweep happened at all. Measured 2026-09-11 on 200 rows of
            # one group, with zero warnings naming the group count.
            #
            # The fallback is kept - it is a usable answer - but it is
            # disclosed as NOT a frontier, both in words and in
            # `optimization_details['pareto_frontier_computed']`, so a caller
            # can tell the two apart without reading this source.
            if len(unique_groups) != 2:
                reason = (
                    f"the Pareto sweep is implemented for exactly 2 groups and this "
                    f"data has {len(unique_groups)} ({unique_groups})"
                )
            else:
                reason = "no threshold pair in the sweep produced a usable candidate"
            warnings.warn(
                f"MultiObjectiveThresholdOptimizer: no Pareto frontier was computed because "
                f"{reason}. Falling back to GroupThresholdOptimizer; pareto_frontier_ holds "
                "that ONE fallback result and is NOT a frontier over the requested "
                "objectives. See optimization_details['pareto_frontier_computed'] is False.",
                UserWarning,
                stacklevel=2,
            )
            simple = GroupThresholdOptimizer(self.constraint, self.constraint.tolerance)
            simple.fit(
                y_true=y_true,
                y_prob=y_prob,
                sensitive_attr=sensitive_attr,
                sample_weight=sample_weight,
            )
            # GroupThresholdOptimizer.fit always assigns result_.
            assert simple.result_ is not None
            self.result_ = simple.result_
            self.result_.optimization_details["pareto_frontier_computed"] = False
            self.result_.optimization_details["pareto_fallback_reason"] = reason
            self.pareto_frontier_ = [self.result_]

        self.is_fitted = True
        return self

    def _apply_group_thresholds(
        self,
        y_prob: np.ndarray,
        sensitive_attr: np.ndarray,
        group_thresholds: Dict[str, float],
    ) -> np.ndarray:
        """Apply group-specific thresholds.

        BGL-G05. This class did not carry the sibling's refusal, so a group it
        was never fitted for kept the ``np.zeros`` seed and was denied outright.
        See :func:`_refuse_unfitted_groups` for the measurement.
        """
        _refuse_unfitted_groups(sensitive_attr, group_thresholds)
        y_pred = np.zeros(len(y_prob), dtype=int)
        for group, thresh in group_thresholds.items():
            mask = sensitive_attr.astype(str) == group
            y_pred[mask] = (y_prob[mask] >= thresh).astype(int)
        return y_pred

    def select_point(self, index: int) -> "MultiObjectiveThresholdOptimizer":
        """
        Select a point from the Pareto frontier.

        Args:
            index: Index of the point to select

        Returns:
            self
        """
        if not self.is_fitted:
            raise RuntimeError("Optimizer must be fitted first")
        if index < 0 or index >= len(self.pareto_frontier_):
            raise ValueError(f"Index {index} out of range [0, {len(self.pareto_frontier_)})")

        self._selected_index = index
        self.result_ = self.pareto_frontier_[index]
        return self

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """Apply the selected threshold configuration."""
        if not self.is_fitted:
            raise RuntimeError("Optimizer must be fitted before calling predict()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        # BGL-G05. A row with no score is could-not-check, not a rejection.
        _refuse_unscored_rows(y_prob, caller=type(self).__name__)

        # fit() always assigns result_ before setting is_fitted.
        assert self.result_ is not None
        return self._apply_group_thresholds(y_prob, sensitive_attr, self.result_.group_thresholds)


def create_threshold_optimizer(
    optimizer_type: str = "group",
    constraint: Union[str, FairnessConstraintType] = "demographic_parity",
    tolerance: float = 0.05,
    **kwargs,
) -> BaseThresholdOptimizer:
    """
    Factory function to create threshold optimizers.

    Args:
        optimizer_type: Type of optimizer ('single', 'group', 'multi_objective')
        constraint: Fairness constraint to satisfy
        tolerance: Maximum allowed constraint violation
        **kwargs: Additional arguments passed to the optimizer

    Returns:
        Configured threshold optimizer

    Example:
        >>> optimizer = create_threshold_optimizer(
        ...     optimizer_type='group',
        ...     constraint='equalized_odds',
        ...     tolerance=0.05
        ... )

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: create_threshold_optimizer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    optimizers = {
        "single": ThresholdOptimizer,
        "global": ThresholdOptimizer,
        "group": GroupThresholdOptimizer,
        "multi_objective": MultiObjectiveThresholdOptimizer,
        "pareto": MultiObjectiveThresholdOptimizer,
    }

    if optimizer_type.lower() not in optimizers:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}"
        )

    return optimizers[optimizer_type.lower()](constraint=constraint, tolerance=tolerance, **kwargs)
