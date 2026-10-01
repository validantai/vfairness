"""
Base Classes for Constraint-Based Fair Training.

This module provides the foundational abstractions for constraint-based
approaches to fair machine learning, where fairness requirements are
specified as explicit constraints that must be satisfied.

Key Concepts:
    - **Fairness Constraint**: A mathematical specification of what fairness means
    - **Constraint Violation**: How much a model violates the fairness constraint
    - **Lagrangian Relaxation**: Converting constraints to penalty terms
    - **Constrained Optimization**: Solving the min-max problem

The Reductions Approach:
    The key insight is that fair classification can be reduced to a sequence
    of cost-sensitive classification problems. This enables using any
    standard classifier as a subroutine while achieving fairness guarantees.

References:
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
    - Agarwal et al. (2019): Fair Regression: Quantitative Definitions and Reduction
    - Cotter et al. (2019): Two-Player Games for Efficient Non-Convex Constrained Optimization
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

from vfairness._names import _is_missing
from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    coerce_to_array,
)
from vfairness.exceptions import ConfigurationError

# The two bound shapes the constructor's type hint offers.
#
# F22 (2026-09-10): `bound_type` was written once in BaseFairnessConstraint
# .__init__ and read NOWHERE. Measured on 400 rows with two groups whose
# positive rates were 0.728 and 0.201, DemographicParityConstraint(
# tolerance=0.05, bound_type='difference') and the same call with
# bound_type='ratio' returned the SAME overall_violation, 0.527124, and the
# same is_satisfied, and a __getattribute__ spy recorded 0 reads of
# self.bound_type across compute_violation, signed_constraint_value and
# is_satisfied. `bound_type='banana'` was accepted and stored. The documented
# ratio bound, g_a / g_b within [1 - eps, 1 + eps], would have read 0.276 on
# that same data and failed a tolerance of 0.05 on a different scale entirely.
#
# Every compute_violation here measures a max-minus-min SPREAD of rates and
# compares it against `tolerance`, which is the difference bound. Honouring the
# ratio bound means a different violation scale, a different signed value for
# the Lagrangian and a different meaning for `tolerance` in five classes and in
# everything they feed (ExponentiatedGradient, GridSearch, FairClassifier,
# reductions.ThresholdOptimizer). That is new behaviour, so it is REFUSED here
# rather than silently ignored: a caller who asks for a ratio bound now gets
# NotImplementedError naming what to use instead.
BOUND_TYPES = ("difference", "ratio")

# What each bound MEANS, so a refusal can state it rather than name it.
BOUND_FORMULAS = {
    "difference": "|g_a - g_b| <= tolerance",
    "ratio": "L_g <= (1 + tolerance) * L_overall",
}


def _defined(values: Iterable[float]) -> List[float]:
    """Return only the DEFINED (non-NaN) rates from an iterable.

    An undefined per-group conditional rate (empty positive/negative
    conditioning set) is unmeasurable, not zero and not chance; it is
    excluded here rather than substituted (``v == v`` is False only for NaN).
    """
    return [float(v) for v in values if v == v]


def _rows_with_no_value(values: ArrayLike) -> int:
    """How many entries hold NO VALUE, decided on the VALUES and not the dtype.

    BGL4 (2026-09-27). :func:`_unscored_rows` and :func:`_unlabelled_rows` both
    began as ``if arr.dtype.kind not in "fc": return 0``, so the whole
    unlabelled-dataset and unscored-prediction disclosure was keyed on a numpy
    DTYPE. An object-dtype column of ``np.nan`` or ``None`` is exactly what
    pandas hands over for any mixed or all-missing column, and it reported ZERO
    missing entries. Measured on the 60-row bounded-group-loss fixture (a model
    wrong on 24 of the 30 rows of group a, a measured 0.36 breach with an int
    label column), before and after this function existed::

        y_true = np.array([np.nan] * 60, dtype=object)
          before: n_unlabelled_rows 0, is_satisfied True, could_not_evaluate
                  False, warnings []
          after:  n_unlabelled_rows 60, is_satisfied False,
                  could_not_evaluate True, 1 warning
        y_true = np.array([None] * 60, dtype=object)   same before, same after
        y_pred = np.array([np.nan] * 60, dtype=object) (labels intact)
          before: n_unscored_rows 0, is_satisfied True, warnings []
          after:  n_unscored_rows 60, is_satisfied False, 1 warning

    A non-numeric dtype that holds REAL values still counts 0, which is what
    ``test_control_string_labels_are_not_counted_as_unlabelled`` pins: an
    object column of ``"yes"``/``"no"`` measures 0 missing rows here, before
    and after, because ``"yes"`` is a value and ``np.nan`` is not.

    ``pd.NA`` answers ``!=`` with another NA rather than with a bool, so the
    boolean test raises instead of returning True; ``_names._is_missing``
    already owns that (the ``TypeError`` IS the missing answer) and is reused
    here rather than copied, so the next sentinel is handled once.
    """
    arr = np.asarray(values)
    # Numeric: NaN and +-inf, the original test, unchanged.
    if arr.dtype.kind in "fc":
        return int((~np.isfinite(arr)).sum())
    # Datetime/timedelta carry their own missing marker, NaT, which is not a
    # float and does not reach np.isfinite.
    if arr.dtype.kind in "Mm":
        return int(np.isnat(arr).sum())
    # Object: the only dtype that can hold a missing SENTINEL beside real
    # values, and the one pandas produces for a column with gaps in it.
    if arr.dtype.kind == "O":
        return int(sum(1 for v in arr.ravel().tolist() if _is_missing(v)))
    # bool, int, uint and the fixed-width string kinds cannot carry a missing
    # value at all, so there is nothing to count and nothing to iterate.
    return 0


def _unscored_rows(y_pred: ArrayLike) -> int:
    """How many prediction rows carry NO USABLE SCORE (NaN or infinite).

    G011 (2026-09-17). Every ``compute_violation`` in this file binarises a
    float prediction with ``(y_pred >= 0.5)``. Under IEEE 754 that comparison
    is False for NaN, so an unscored row is written 0, a confident REJECT, and
    ``+inf`` lands above the threshold and is written 1, a confident ACCEPT.
    Neither is a decision anybody made.

    Measured before this existed, on 30 rows in two groups whose every
    prediction was NaN, with warnings recorded::

        DemographicParityConstraint(0.05).compute_violation(...)
        -> overall_violation=0.0, is_satisfied=True,
           group_rates={'a': 0.0, 'b': 0.0}, insufficient_data=False
        DemographicParityConstraint(0.05).is_satisfied(...) -> True

    and the same flat PASS from all five constraints, with zero warnings: a
    model that produced no prediction at all was certified as perfectly fair,
    because every row had been fabricated into the same rejection.

    The rule and its wording are taken from
    ``reductions._count_unscored_rows`` / ``_refuse_unscored_rows``, which
    already fixed this for ``ThresholdOptimizer``; this file was never swept.

    BGL4 (2026-09-27): the count is taken from the VALUES by
    :func:`_rows_with_no_value`, not from the dtype. The previous
    ``if arr.dtype.kind not in "fc": return 0`` meant an OBJECT column of
    ``np.nan`` reported 0 unscored rows. Measured on the 60-row bounded-group-
    loss fixture with the labels left intact::

        y_pred = np.array([np.nan] * 60, dtype=object)
          before: n_unscored_rows 0,  is_satisfied True,  warnings []
          after:  n_unscored_rows 60, is_satisfied False, 1 warning

    A non-numeric dtype holding real values (object labels, strings) still
    counts 0, which is what the string-label control pins.
    """
    return _rows_with_no_value(y_pred)


def _unlabelled_rows(y_true: ArrayLike) -> int:
    """How many label rows carry NO USABLE TRUTH (NaN or infinite).

    BGL3 (2026-09-27), the ``y_true`` half of :func:`_unscored_rows`. A loss is
    a comparison against a label, and ``y_pred_binary != nan`` is True for every
    prediction under IEEE 754, so an unlabelled row is scored as a MISS: not a
    hit, not an abstention, a confident error nobody observed.

    That does not merely add noise to ``BoundedGroupLossConstraint``, it flips
    its verdict, because its bound is RELATIVE, ``L_g <= (1 + tol) * L_overall``.
    Pushing every group's loss towards 1.0 pushes every ratio towards 1.0, and a
    ratio bound is satisfied at 1.0. Measured on 60 rows, two groups of 30, a
    model right on all of b and wrong on 24 of 30 in a, tolerance=0.1::

        labels intact      -> overall_violation=0.36, is_satisfied=False
                              group_losses={'a': 0.8, 'b': 0.0}
        30 of 60 NaN       -> overall_violation=0.13, is_satisfied=False
                              group_losses={'a': 0.9, 'b': 0.5}
        all 60 NaN         -> overall_violation=0.0,  is_satisfied=True,
                              could_not_evaluate=False, ZERO warnings
                              group_losses={'a': 1.0, 'b': 1.0}

    The last row is a compliance certificate over a dataset with no labels at
    all, byte-identical to a measured pass, on the same predictions that were a
    measured 0.36 breach one line above. ``is_satisfied()`` answered True and
    ``signed_constraint_value(..., 'a')`` answered 0.0, no violation.

    Non-numeric labels that hold REAL values (strings, an object column of
    ``"yes"``/``"no"``) count as 0 rather than raising, exactly as in
    :func:`_unscored_rows`.

    BGL4 (2026-09-27): the count is taken from the VALUES by
    :func:`_rows_with_no_value`, because the dtype test this function used to
    open with, ``if arr.dtype.kind not in "fc": return 0``, let the entire
    certificate above survive for an OBJECT dtype label column, which is what
    pandas hands over for any mixed or all-missing column. Measured on the same
    60 predictions::

        y_true = np.array([np.nan] * 60, dtype=object)
          before: n_unlabelled_rows 0,  is_satisfied TRUE,
                  could_not_evaluate False, warnings []
          after:  n_unlabelled_rows 60, is_satisfied False,
                  could_not_evaluate True, 1 warning
        y_true = np.array([None] * 60, dtype=object)     identical, both ways

    and ``signed_constraint_value(..., 'a')`` went from 0.0 in silence to 0.0
    with the unlabelled-row warning beside it.

    The disclosure does not depend on which ``loss_fn`` is in use, because this
    class cannot know what a custom one did with a missing label. A caller whose
    loss function drops unlabelled rows itself gets a could-not-check it did not
    need, which is the safe direction; dropping them before the call gets a
    measured verdict.
    """
    return _rows_with_no_value(y_true)


def _warn_unlabelled(constraint_type: str, n_unlabelled: int, n_rows: int) -> None:
    """Say that part of the group loss is fabrication, not measurement.

    Same narrow line as :func:`_warn_unscored`: the MAGNITUDE is left as it was
    computed, because blanking a mostly-real disparity over a handful of missing
    labels is the over-correction this campaign warns about, and only the
    VERDICT is withdrawn.
    """
    warnings.warn(
        f"{constraint_type}: {n_unlabelled} of {n_rows} row(s) have no usable label "
        f"(NaN or infinite y_true). A prediction compares unequal against a "
        f"non-finite label, so every one of those rows entered the group loss as a "
        f"MISS, which is an error nobody observed, and this bound is RELATIVE so "
        f"inflating every group's loss towards 1.0 inflates every ratio towards the "
        f"satisfied end. The violation reported is part measurement and part "
        f"fabrication, so NO verdict is given: details['insufficient_data'] is True, "
        f"could_not_evaluate is True and is_satisfied() answers None. Drop the "
        f"unlabelled rows to get a measured verdict.",
        UserWarning,
        stacklevel=3,
    )


def _warn_unscored(constraint_type: str, n_unscored: int, n_rows: int) -> None:
    """Say that part of the violation is fabrication, not measurement.

    The MAGNITUDE is deliberately left as it was computed, and only the
    VERDICT is withdrawn (``details['insufficient_data']`` is set, so
    ``could_not_evaluate`` is True and ``is_satisfied()`` answers None).
    Blanking a mostly-real disparity because a handful of rows were unscored
    would be the over-correction this campaign warns about; the same narrow
    line is drawn in ``reductions.ThresholdOptimizer.fit`` and in
    ``post_processing.threshold_optimization.optimizer._qualified_feasibility``.
    """
    warnings.warn(
        f"{constraint_type}: {n_unscored} of {n_rows} row(s) have no usable score "
        f"(NaN or infinite y_pred). A non-finite score compares False against the "
        f"0.5 threshold, so those rows entered the rates as fabricated rejections "
        f"(and +inf as fabricated acceptances). The violation reported is part "
        f"measurement and part fabrication, so NO verdict is given: "
        f"details['insufficient_data'] is True, could_not_evaluate is True and "
        f"is_satisfied() answers None. Impute or drop the unscored rows to get a "
        f"measured verdict.",
        UserWarning,
        stacklevel=3,
    )


def _warn_unusable_bound(constraint_type: str, reason: str, tolerance: float) -> None:
    """Say that the BOUND, not the data, decided the verdict.

    Unlike :func:`_warn_unscored` the magnitude is NOT left as it was computed,
    because there is no partly-real magnitude to keep: when the comparison cannot
    be false for any data, the 0.0 it produced is not a small violation, it is
    the absence of a comparison. The whole measurement is withdrawn to NaN, which
    is the sentinel ``ConstraintViolation.could_not_evaluate`` already reads.
    """
    warnings.warn(
        f"{constraint_type}: the configured bound cannot grade this metric ({reason}). "
        f"No verdict is given: overall_violation is NaN, "
        f"details['insufficient_data'] is True and is_satisfied is False, because a "
        f"violation of exactly 0.0 from a bound nothing can breach is "
        f"indistinguishable from a measured pass. It also reached the optimiser "
        f"through signed_constraint_value as a steering push of exactly 0.0, i.e. a "
        f"mitigation applying no correction while reporting success. "
        f"tolerance={tolerance!r} is the setting to change.",
        UserWarning,
        stacklevel=3,
    )


def _steering_value(
    violation: "ConstraintViolation",
    group: str,
    constraint_type: str,
) -> float:
    """One group's signed deviation as an OPTIMISER STEERING value.

    Three states, never two (BGL stage 2b, 2026-09-17). Every
    ``signed_constraint_value`` in this file answered a byte-identical ``0.0``
    for three different facts, and one of them was a caller bug:

    * the group's deviation was MEASURED at exactly 0.0 (perfect parity);
    * the group is UNMEASURABLE (its rate is undefined, or there is no second
      group to deviate from), so its entry is NaN;
    * the group DOES NOT EXIST, because ``.get(group, 0.0)`` answered the
      default for a name that was never in the data.

    The third is now a ``KeyError``: a misspelled or stale group name is a
    programming error at any input, and answering "no violation" for a group
    that was never fitted is the worst of the three, because it is silent and
    permanent.

    The second keeps returning 0.0, deliberately, and says so. This value is
    consumed by the exponentiated-gradient update at
    ``reductions.py:598`` (``lambda *= exp(eta * g)``) and by ``_signed_gaps``,
    where a NaN would poison every multiplier and the cost-sensitive sample
    weights with them, so 0.0 here means NO PUSH, not "no gap". It is now
    accompanied by a ``RuntimeWarning`` naming the group, and the MEASUREMENT
    is on the ``ConstraintViolation`` this was derived from
    (``group_violations[group]`` is NaN, ``details['insufficient_data']`` is
    True), which is the channel a reader consumes. Returning the refusal here
    as well needs the two consumers above to translate it back to "no push"
    with a count; that is `src/vfairness/in_processing/constraints/reductions.py`,
    not this file.
    """
    if group not in violation.group_violations:
        raise KeyError(
            f"{constraint_type}: no group {group!r} in the fitted sensitive attribute "
            f"(groups: {sorted(map(str, violation.group_violations))}). A signed "
            f"constraint value for a group that does not exist used to come back as "
            f"0.0, indistinguishable from a measured perfect parity."
        )
    val = violation.group_violations[group]
    if val != val:
        warnings.warn(
            f"{constraint_type}: the constraint deviation for group {group!r} is "
            f"UNMEASURABLE (its rate is undefined, or there is no second group to "
            f"compare it against). Returning 0.0 as a STEERING value, meaning no push "
            f"on this group's multiplier, NOT a measured absence of disparity. The "
            f"measurement is the NaN in ConstraintViolation.group_violations[{group!r}] "
            f"with details['insufficient_data'] True.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 0.0
    return float(val)


def _deviation_is_undefined(n_defined_rates: int) -> bool:
    """True when a per-group deviation has no peer to be a deviation FROM.

    The argument is the number of DEFINED rates on the arm being centred, not
    the number of groups (corrected G011, 2026-09-17). Three groups of which
    only one has a measurable rate is the same vacuity as one group: the mean
    of a single rate IS that rate, so the deviation is `r - r` = 0.0 for every
    possible input. Passing ``len(gm.groups)`` here caught only the one-group
    case and left the three-group one reporting a fabricated 0.0 deviation
    beside two honest NaNs.

    BGL stage 2b, 2026-09-17, the unpinned sibling of the ``_defined_spread``
    work above. ``overall_violation`` correctly answers NaN for a single
    group, but every ``group_violations`` entry beside it was still computed
    as ``rate - overall_rate``, and with ONE group the overall rate IS that
    group's rate, so the entry was exactly 0.0 by arithmetic vacuity, the same
    way ``max(rates) - min(rates)`` was. Measured through
    ``FairClassifier(method="reductions").fit`` on 200 rows with a single
    sensitive value: ``fairness_metrics == {..., 'A': 0.0,
    'insufficient_data': True}``, so a per-group report can print a 0.00
    deviation for a group nothing was compared against, sitting in the same
    dict as the flag that says nothing was measurable.

    A deviation from the mean of one number is not a small deviation. It is
    not a deviation.
    """
    return n_defined_rates < 2


def _vacuous_relative_bound_reason(
    bound: float,
    loss_upper_bound: Optional[float],
) -> Optional[str]:
    """Why no achievable group loss can exceed ``bound``, or None.

    **A BOUND IS UNUSABLE WHEN THE COMPARISON IT CONTROLS CANNOT BE FALSE FOR ANY
    DATA IN THE METRIC'S RANGE.** That is a property of the bound AND the loss's
    range TOGETHER, never a property of the bound being finite, and it is the
    same rule already written down in
    :func:`vfairness.evaluation.vfairness_metrics._metric_direction.vacuous_bound_reason`
    for the release gate's four bounds. A guard testing only ``isinf`` leaves
    every finite vacuous bound open, which is what it did here.

    ``BoundedGroupLossConstraint`` enforces ``L_g <= (1 + tolerance) * L_overall``,
    so the breach is ``group_loss > bound``. The 0-1 loss is a mean of a boolean
    and therefore lies in [0, 1], so a bound at or above 1.0 cannot be exceeded
    by any group on any data. Measured on this class before this guard, 60 rows
    in two groups where group "a" has a 0-1 loss of 0.8 and group "b" 0.0, a
    real breach of 0.38 at the default tolerance::

        tolerance=0.05 -> bound 0.42   overall_violation 0.38  is_satisfied False
        tolerance=0.5  -> bound 0.60   overall_violation 0.20  is_satisfied False
        tolerance=60.0 -> bound 24.40  overall_violation 0.0   is_satisfied True,
                          insufficient_data False, ZERO warnings

    A maximum of 24.4 on a loss that cannot exceed 1.0 grades nothing at all,
    and the same 0.0 then reached the OPTIMISER through
    ``signed_constraint_value``, so the steering push for the breached group was
    exactly 0.0: a mitigation applying no correction while reporting success.

    ``tolerance=1.0`` on that fixture is deliberately NOT vacuous and is a
    control in the suite: it puts the bound at 0.80, which group "a" meets
    exactly, so the measured 0.0 there is a real boundary result and a group at
    0.9 would still breach it. Vacuity is decided against the loss's REACHABLE
    range, not against the size of the tolerance.

    ``loss_upper_bound`` of None is the could-not-check and returns None: a
    caller-supplied ``loss_fn`` has no declared range, so nothing can be shown
    about reachability and refusing there would block bounds that are perfectly
    usable. That is the over-correction this rule's own docstring in
    ``_metric_direction`` warns about: a constraint that refuses everything
    passes every refusal test.
    """
    if loss_upper_bound is None:
        return None
    if not is_measured(bound) or not is_measured(loss_upper_bound):
        return None
    if float(bound) < float(loss_upper_bound):
        return None
    return (
        f"a bound of {float(bound):.4f} on a loss that cannot exceed "
        f"{float(loss_upper_bound):.4f}, so no group's loss can exceed it and the "
        f"comparison that detects a breach is False whatever the data"
    )


def _defined_spread(values: Iterable[float]) -> float:
    """Max-min spread over the group rates, or NaN if ANY of them is undefined.

    Fabricating a rate (e.g. 0.5) for an empty positive/negative subgroup
    invents up to 0.5 of disparity and can flip a satisfied model to a false
    FAIL, so an undefined rate is never SUBSTITUTED (fairlearn returns NaN for a
    TPR/FPR over an empty conditioning set; Hardt et al. 2016 define TPR/FPR
    only over non-empty sets).

    It is not DROPPED either, and that is the stronger half of this rule. This
    function used to return ``max(defined) - min(defined)`` as soon as two
    groups had a defined rate, which is exactly where a LOWER BOUND was
    manufactured and then reported as a measurement: with three or more groups
    the unmeasurable one was silently excluded and the survivors' spread became
    the violation. Measured on A (TPR 0.90) | B (TPR 0.88) | C (no positive
    labels at all), ``EqualizedOddsConstraint(tolerance=0.05)`` reported
    ``overall_violation=0.02``, ``is_satisfied=True`` and
    ``insufficient_data=False`` while ``group_tprs['C']`` was NaN in that same
    result: a compliance certificate issued over a group the constraint could
    not see, on exactly the small / intersectional groups these constraints
    exist to protect, and it fed ExponentiatedGradient, GridSearch and
    FairClassifier. The true spread there is at least 0.02 and unbounded above,
    because C's rate could be anything.

    The old fewer-than-two-defined condition is subsumed: it is just the
    two-group instance of the same drop. Fewer than two groups in total is
    still NaN, since no pair exists to compare.
    """
    values = list(values)
    if len(values) < 2 or any(v != v for v in values):
        return float("nan")
    return max(float(v) for v in values) - min(float(v) for v in values)


class FairnessConstraintType(Enum):
    """Types of fairness constraints."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    FALSE_POSITIVE_RATE_PARITY = "fpr_parity"
    TRUE_POSITIVE_RATE_PARITY = "tpr_parity"
    PREDICTIVE_PARITY = "predictive_parity"
    BOUNDED_GROUP_LOSS = "bounded_group_loss"
    ERROR_RATE_PARITY = "error_rate_parity"


@dataclass
class ConstraintViolation:
    """
    Container for constraint violation information.

    THREE STATES, NEVER TWO. These fields were rewritten for could-not-check and
    this docstring was not, so it described a two-state object that no longer
    exists. Read them together, not one at a time:

        overall_violation = 0.04, is_satisfied = True    measured, within tolerance
        overall_violation = 0.31, is_satisfied = False   measured, over tolerance
        overall_violation = NaN,  is_satisfied = False,
                                  details["insufficient_data"] = True
                                                         NOT MEASURED

    The third row is the one to code against. `is_satisfied` is False there
    because a constraint that could not be checked must not read as met, which
    is the fail-closed direction for something that steers training. But it is
    NOT a measured breach, and `details["insufficient_data"]` is what tells the
    two apart. A caller that branches on `is_satisfied` alone will report a
    violation nobody measured.

    A rate is undefined, and so NaN, when a group has no positive labels (TPR)
    or no negative labels (FPR). One such group makes that whole arm NaN rather
    than being dropped: the spread over the groups that happen to be measurable
    is a LOWER BOUND on the real disparity, not the disparity.

    Attributes:
        constraint_type: Type of fairness constraint
        overall_violation: Overall violation magnitude, or NaN when the
            constraint could not be evaluated. Never a substituted 0.0.
        group_violations: Per-group violation values; NaN for a group whose
            rate is undefined.
        is_satisfied: False both for a measured breach AND for an unevaluable
            constraint. Pair it with details["insufficient_data"].
        tolerance: Tolerance threshold used
        details: Additional violation details, including group_tprs,
            group_fprs, tpr_violation, fpr_violation, n_unscored_rows (how
            many rows carried no usable score and so entered the rates as
            fabricated decisions), n_unlabelled_rows (bounded group loss only:
            how many rows carried no usable label and so entered the group
            losses as fabricated misses) and the insufficient_data flag named
            above.
    """

    constraint_type: str
    overall_violation: float
    group_violations: Dict[str, float]
    is_satisfied: bool
    tolerance: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def could_not_evaluate(self) -> bool:
        """True when this constraint was NOT MEASURED, so no verdict was reached.

        R4C (2026-09-10). The third state was RECORDED here and dropped by every
        reporting consumer downstream, which read ``is_satisfied`` alone and
        drew False as a measured FAIL. This is the read accessor those consumers
        need, so the test lives in ONE place rather than six: the flag written
        beside the violation is authoritative, and a NaN violation is the
        fallback for a ``ConstraintViolation`` built without it (the two
        constraints whose rates cannot be undefined never write the flag).

        ``details.get("insufficient_data", False)`` is deliberately NOT used:
        ``.get`` returns the stored value, not the default, when the key is
        PRESENT holding None, and None would then read as measured. Absence and
        a null are both answered by the NaN fallback.
        """
        flag = self.details.get("insufficient_data")
        if flag is not None:
            return bool(flag)
        v = self.overall_violation
        return v is None or v != v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_type": self.constraint_type,
            "overall_violation": self.overall_violation,
            "group_violations": self.group_violations,
            "is_satisfied": self.is_satisfied,
            "tolerance": self.tolerance,
            "details": self.details,
            # The third state travels with the object, so a consumer that only
            # ever sees the dict (JSON, a report payload) can still tell a
            # measured breach from one nobody could evaluate.
            "could_not_evaluate": self.could_not_evaluate,
        }


def _reported_constraint(
    violation: ConstraintViolation,
) -> Tuple[Optional[float], Optional[bool]]:
    """The (magnitude, verdict) pair a REPORT may print for one constraint.

    R4C (2026-09-10). The single translation from this layer's fail-closed pair,
    which exists to STEER TRAINING, to the reporting layer's three states, which
    exist to be READ. ``is_satisfied`` is False for both a measured breach and a
    constraint nobody could evaluate; only ``could_not_evaluate`` tells them
    apart, and every reporting consumer that skipped that test reported the
    second as the first, down to a red FAIL bar on a chart.

    Returns ``(None, None)`` for an unevaluable constraint: not ``(0.0, False)``,
    which invents a measurement, and not ``(nan, False)``, which is what these
    boundaries used to hand on. NaN is not None, so it survived every
    ``is not None`` gate downstream and was drawn as a measurement.

    It lives here, beside the flag it reads, because analyzer.py and
    wrappers/sklearn_wrappers.py both need it: two copies of this rule is how
    the two ends of one boundary start disagreeing.
    """
    if violation.could_not_evaluate:
        return None, None
    return float(violation.overall_violation), bool(violation.is_satisfied)


@dataclass
class OptimizationResult:
    """
    Result from constrained optimization.

    Attributes:
        converged: Whether optimization converged
        n_iterations: Number of iterations performed
        final_violation: Final constraint violation
        best_gap: Best duality gap achieved
        lambda_history: History of Lagrange multipliers
        loss_history: History of loss values
        violation_history: History of constraint violations
        final_weights: Final sample/model weights
        metadata: Additional optimization metadata
    """

    converged: bool
    n_iterations: int
    final_violation: float
    best_gap: float
    lambda_history: List[Dict[str, float]] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    violation_history: List[float] = field(default_factory=list)
    final_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_violation": self.final_violation,
            "best_gap": self.best_gap,
            "lambda_history": self.lambda_history,
            "loss_history": self.loss_history,
            "violation_history": self.violation_history,
            "final_weights": self.final_weights.tolist()
            if self.final_weights is not None
            else None,
            "metadata": self.metadata,
        }


class BaseFairnessConstraint(ABC):
    """
    Abstract base class for fairness constraints.

    All fairness constraints must implement methods to:
    1. Compute the current violation of the constraint
    2. Compute the signed constraint function (for Lagrangian)
    3. Compute gradients with respect to predictions

    Args:
        tolerance: Maximum allowed constraint violation
        bound_type: The bound this constraint enforces. Only the value in the
            subclass's ``supported_bound_types`` is accepted; any other known
            value raises NotImplementedError rather than being stored and
            ignored (F22). Every rate-parity constraint here implements the
            DIFFERENCE bound, |g_a - g_b| <= ε, which is what ``tolerance``
            is compared against. The RATIO bound, g_a / g_b in [1-ε, 1+ε], is
            implemented only by BoundedGroupLossConstraint, whose bound really
            is a ratio one: L_g <= (1 + ε) * L_overall.

    Example:
        >>> constraint = DemographicParityConstraint(tolerance=0.05)
        >>> violation = constraint.compute_violation(y_pred, sensitive_attr)
        >>> print(f"Constraint satisfied: {violation.is_satisfied}")
    """

    #: The bound(s) this class's compute_violation actually implements. A
    #: subclass that implements another one widens this; nothing else does.
    supported_bound_types: ClassVar[Tuple[str, ...]] = ("difference",)

    def __init__(
        self,
        tolerance: float = 0.05,
        bound_type: Literal["difference", "ratio"] = "difference",
    ):
        if bound_type not in BOUND_TYPES:
            raise ConfigurationError(
                f"bound_type={bound_type!r} is not a bound this library knows. "
                f"Supported: {list(BOUND_TYPES)}."
            )
        if bound_type not in self.supported_bound_types:
            implemented = self.supported_bound_types[0]
            raise NotImplementedError(
                f"{type(self).__name__} does not implement the {bound_type!r} bound. "
                f"It implements the {implemented!r} bound, {BOUND_FORMULAS[implemented]}, "
                f"which is what `tolerance` is compared against. Refused rather than "
                f"accepted and ignored: until 2026-09-10 this value was stored and "
                f"never read, so bound_type='ratio' returned exactly the "
                f"difference-bound violation under a ratio label. For a ratio bound on "
                f"group LOSS use BoundedGroupLossConstraint; for a ratio of selection "
                f"rates measure disparate_impact_ratio in vfairness.evaluation and "
                f"threshold that yourself."
            )
        # A BOUND THAT IS NOT A MEASUREMENT CANNOT DECIDE A VERDICT (F9 wave 4,
        # 2026-09-30). `tolerance` was stored here with no check of any kind and
        # every constraint in this file compares its violation against it. The
        # guard is HERE, above the subclass selection, because the fabrication is
        # reachable through each one of them and a check inside a single
        # compute_violation would only move it to a sibling class.
        #
        # NaN is the worst value of the set, because it is the sentinel this file
        # uses for "could not evaluate": every comparison against NaN is False,
        # and Python's builtin `max` returns its FIRST argument when the
        # comparison is False, so an UNKNOWABLE bound came back as exactly 0.0.
        # Measured on BoundedGroupLossConstraint before this guard, 60 rows in
        # two groups where group "a" has a 0-1 loss of 0.8 against group "b"'s
        # 0.0, a real breach:
        #
        #   tolerance=0.05 -> overall_violation 0.38, is_satisfied False
        #   tolerance=inf  -> overall_violation 0.0, is_satisfied True,
        #                     insufficient_data False, ZERO warnings
        #   tolerance=nan  -> overall_violation 0.0, is_satisfied True,
        #                     insufficient_data False, ZERO warnings
        #
        # so all three of overall_violation, is_satisfied and
        # details["insufficient_data"] agreed on a measured clean pass for a
        # bound nobody could evaluate. is_measured also excludes bool (a
        # tolerance of True is 1.0, which is not a number anybody typed) and any
        # non-numeric value, which previously surfaced several frames later as a
        # numpy or float error naming neither the argument nor this class.
        if not is_measured(tolerance):
            raise ConfigurationError(
                f"tolerance={tolerance!r} is not a finite number, so it is not a bound "
                f"that can be breached: every comparison against NaN or an infinity is "
                f"False, which this file reads as 'no violation'. A constraint "
                f"configured with it reported is_satisfied=True and "
                f"insufficient_data=False for a measured breach, in silence. Pass a "
                f"finite, nonnegative tolerance."
            )
        # The other side of the same quantity. `tolerance` is documented as the
        # MAXIMUM ALLOWED violation, and every violation in this file is a
        # magnitude, so a negative maximum cannot be satisfied by any data: it is
        # the mirror of the vacuous bound above, a constraint that can only ever
        # FAIL. Refused rather than left to report a breach nobody caused.
        if tolerance < 0:
            raise ConfigurationError(
                f"tolerance={tolerance!r} is negative, but it is the MAXIMUM ALLOWED "
                f"violation and every violation measured here is a magnitude. No data "
                f"can satisfy a negative maximum, so the constraint would report a "
                f"breach for a model with perfect parity. Pass a nonnegative tolerance."
            )
        self.tolerance = tolerance
        self.bound_type = bound_type

    @abstractmethod
    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """
        Compute the constraint violation.

        Args:
            y_pred: Predicted labels or probabilities
            y_true: True labels
            sensitive_attr: Sensitive attribute values

        Returns:
            ConstraintViolation with violation details
        """
        pass

    @abstractmethod
    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """
        Compute signed constraint value for a specific group.

        This is used in the Lagrangian formulation where we need both
        positive and negative violations.

        Args:
            y_pred: Predictions
            y_true: True labels
            sensitive_attr: Sensitive attribute
            group: Group identifier

        Returns:
            Signed constraint value (positive = violation)
        """
        pass

    def is_satisfied(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> Optional[bool]:
        """Whether the constraint is met. THREE STATES, NEVER TWO.

            True   measured, within tolerance
            False  measured, over tolerance
            None   NOT MEASURED -- no verdict was reached

        ``None`` is returned whenever ``ConstraintViolation.could_not_evaluate``
        is set, which is the read accessor that already told a measured breach
        apart from an unevaluable constraint one layer down.

        This method used to return ``violation.is_satisfied`` alone, and that
        field is deliberately fail-closed: False both for a measured breach AND
        for a constraint nobody could evaluate. Returning it raw collapsed the
        third state in BOTH directions. On single-group data
        DemographicParityConstraint answered a flat ``True`` -- the strongest
        possible two-state answer, the fairness constraint is met -- where no
        group pair existed to compare; and for the siblings that DID record
        could_not_evaluate it answered a flat ``False``, a measured breach,
        for the same unmeasurable input. A caller cannot tell either apart from
        a measurement without reading the source, which is the whole point of
        the flag.
        """
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        if violation.could_not_evaluate:
            return None
        return bool(violation.is_satisfied)


class DemographicParityConstraint(BaseFairnessConstraint):
    """
    Demographic Parity Constraint.

    Requires that the positive prediction rate is equal across groups:
        |P(ŷ=1|G=a) - P(ŷ=1|G=b)| <= ε

    Args:
        tolerance: Maximum allowed disparity, as a DIFFERENCE of rates
        bound_type: 'difference' only. 'ratio' is refused, not ignored (F22).

    Example:
        >>> constraint = DemographicParityConstraint(tolerance=0.05)
        >>> violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: demographic_parity_constraint. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """Compute demographic parity violation.

        ``y_true`` is accepted for signature uniformity with the other
        constraints and is NOT read: demographic parity is defined on the
        predictions alone, P(yhat=1 | G=a) against P(yhat=1 | G=b), and
        conditioning it on the labels would make it a different metric.
        """
        y_pred = coerce_to_array(y_pred, "y_pred")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        # Before the binarisation below, which turns a NaN into a confident
        # rejection. See _unscored_rows.
        n_unscored = _unscored_rows(y_pred)

        # Threshold probabilities if needed
        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        gm = GroupManager(sensitive_attr)
        overall_rate = np.mean(y_pred_binary)

        group_rates = {}
        group_violations = {}

        # See _deviation_is_undefined: with one group `rate - overall_rate` is
        # 0.0 for every possible input, which is not a measured zero deviation.
        # The group count IS the defined-rate count here, unlike the rate-parity
        # siblings: a selection rate is a mean over the whole group and has no
        # conditioning set that can be empty, so it is never NaN.
        no_peer = _deviation_is_undefined(len(gm.groups))
        for group in gm.groups:
            mask = gm.get_mask(group)
            rate = np.mean(y_pred_binary[mask])
            group_rates[group] = rate
            group_violations[group] = float("nan") if no_peer else rate - overall_rate

        # Demographic parity is defined over a PAIR of groups,
        # |P(yhat=1|G=a) - P(yhat=1|G=b)|. This line used to read
        # `max(rates) - min(rates)`, which over a ONE-element list is 0.0 by
        # arithmetic vacuity rather than by comparison: the max over an EMPTY
        # set of pairs. That 0.0 left here byte-identical to a genuine
        # two-group measured perfect parity (overall_violation 0.0,
        # is_satisfied True, could_not_evaluate False, no flag in details), so
        # a fit on a sensitive attribute with a single value was issued a
        # compliance certificate. Nothing upstream stops that input:
        # GroupManager accepts one unique value and neither
        # FairnessTrainingAnalyzer.__init__ nor ExponentiatedGradient.fit
        # checks n_groups.
        #
        # This was the last rate-parity constraint in the file bypassing
        # _defined_spread, whose own docstring states the rule it was
        # breaking: "Fewer than two groups in total is still NaN, since no
        # pair exists to compare." Routing through it also inherits the
        # undefined-rate handling the siblings already have.
        max_violation = _defined_spread(group_rates.values())

        if n_unscored:
            _warn_unscored("demographic_parity", n_unscored, len(y_pred))

        return ConstraintViolation(
            constraint_type="demographic_parity",
            overall_violation=max_violation,
            group_violations=group_violations,
            # NaN <= tolerance is False, so an unmeasurable disparity is
            # could-not-check reported fail-closed, never a false PASS. Pair it
            # with details["insufficient_data"] to tell the two apart.
            # `and not n_unscored` keeps the invariant this dataclass
            # documents: the field is never True for something nobody could
            # evaluate. An all-NaN prediction vector otherwise reads
            # 0.0 <= tolerance and returns a flat PASS.
            is_satisfied=bool(max_violation <= self.tolerance) and not n_unscored,
            tolerance=self.tolerance,
            details={
                "group_rates": group_rates,
                "overall_rate": overall_rate,
                "insufficient_data": bool(max_violation != max_violation) or bool(n_unscored),
                # How many rows entered the rates as fabricated decisions
                # rather than measured ones (0 in the normal case).
                "n_unscored_rows": n_unscored,
            },
        )

    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """Compute signed constraint value for demographic parity.

        As in :meth:`compute_violation`, ``y_true`` is accepted for signature
        uniformity and is not read.
        """
        # Routed through compute_violation rather than recomputing
        # `group_rate - overall_rate` here, which is the same arithmetic but
        # skipped the single-group and unknown-group handling that method now
        # carries. See _steering_value for the three states.
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        return _steering_value(violation, group, "demographic_parity")


class EqualizedOddsConstraint(BaseFairnessConstraint):
    """
    Equalized Odds Constraint.

    Requires equal TPR and FPR across groups:
        |TPR_a - TPR_b| <= ε AND |FPR_a - FPR_b| <= ε

    Args:
        tolerance: Maximum allowed disparity, as a DIFFERENCE of rates
        bound_type: 'difference' only. 'ratio' is refused, not ignored (F22).

    Example:
        >>> constraint = EqualizedOddsConstraint(tolerance=0.05)
        >>> violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: equalized_odds_constraint. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """Compute equalized odds violation."""
        y_pred = coerce_to_array(y_pred, "y_pred")
        y_true = coerce_to_array(y_true, "y_true")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        # Before the binarisation below, which turns a NaN into a confident
        # rejection. See _unscored_rows.
        n_unscored = _unscored_rows(y_pred)

        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        gm = GroupManager(sensitive_attr)

        group_tprs = {}
        group_fprs = {}
        group_violations = {}

        for group in gm.groups:
            mask = gm.get_mask(group)
            y_pred_g = y_pred_binary[mask]
            y_true_g = y_true[mask]

            positives = y_true_g == 1
            negatives = y_true_g == 0

            # An empty positive/negative subgroup makes the TPR/FPR undefined;
            # record NaN (unmeasurable) rather than fabricating 0.5.
            tpr = float(np.mean(y_pred_g[positives])) if positives.sum() > 0 else float("nan")
            fpr = float(np.mean(y_pred_g[negatives])) if negatives.sum() > 0 else float("nan")

            group_tprs[group] = tpr
            group_fprs[group] = fpr

        # Max-min spread across ALL groups on each arm. A group with an undefined
        # rate is neither substituted nor dropped: it makes that arm NaN, because
        # the spread over the groups that happen to be measurable is a lower bound
        # on the real disparity, not the disparity (see ``_defined_spread``).
        tpr_violation = _defined_spread(group_tprs.values())
        fpr_violation = _defined_spread(group_fprs.values())
        # Equalized odds is the max over BOTH arms (Hardt, Price & Srebro 2016);
        # each is load-bearing. If either arm is unmeasurable then so is the max,
        # so an unmeasurable arm must PROPAGATE, not be dropped.
        #
        # This line used to read `_defined([tpr_violation, fpr_violation])`,
        # which discarded the NaN arm and took the max over the survivor. With
        # one arm unmeasurable that yields a LOWER BOUND reported as the
        # violation: 8 rows where a group has no positive labels gave
        # overall_violation=0.0, is_satisfied=True and insufficient_data=False
        # while details['tpr_violation'] was nan in the same dict. That is a
        # compliance certificate issued off a half-measured metric, on exactly
        # the small / intersectional groups the constraint exists to protect,
        # and it fed ExponentiatedGradient, GridSearch, FairClassifier and
        # reductions.ThresholdOptimizer. EqualOpportunityConstraint, measuring
        # that same TPR, already reported it correctly; the two siblings
        # disagreed and the permissive one was wrong.
        arms = [tpr_violation, fpr_violation]
        max_violation = float("nan") if any(a != a for a in arms) else max(arms)

        # Per-group violations (average of the DEFINED TPR/FPR deviations).
        defined_tprs = _defined(group_tprs.values())
        defined_fprs = _defined(group_fprs.values())
        overall_tpr = float(np.mean(defined_tprs)) if defined_tprs else float("nan")
        overall_fpr = float(np.mean(defined_fprs)) if defined_fprs else float("nan")

        # PER ARM, over the DEFINED rates (G011, 2026-09-17). An arm whose
        # centre is the mean of ONE rate has no peer to deviate from, so that
        # group's deviation is `r - r` = exactly 0.0 for every possible input.
        # `overall_tpr == overall_tpr` could not catch it: the centre is a
        # perfectly finite number, it is just this group's own rate. Two
        # measurements before this change:
        #
        #   two rows, a=(y=1,pred=1) and b=(y=0,pred=1): the only defined TPR
        #   is a's and the only defined FPR is b's, and group_violations came
        #   back {'a': 0.0, 'b': 0.0}, a perfect-parity reading for a pair
        #   that was never compared on any shared arm;
        #
        #   40 rows, a|b|c with positive labels only in a: a's TPR deviation
        #   was the vacuous 0.0 and it was AVERAGED with a's real FPR
        #   deviation of 0.0333, halving it to 0.0167. A fabricated zero does
        #   not merely sit beside the measurement here, it dilutes it.
        #
        # The old n_groups guard is subsumed: with one group each arm holds at
        # most one defined rate, so both arms refuse and the entry stays NaN.
        tpr_has_peer = not _deviation_is_undefined(len(defined_tprs))
        fpr_has_peer = not _deviation_is_undefined(len(defined_fprs))
        for group in gm.groups:
            t, f = group_tprs[group], group_fprs[group]
            devs = []
            if t == t and tpr_has_peer:
                devs.append(abs(t - overall_tpr))
            if f == f and fpr_has_peer:
                devs.append(abs(f - overall_fpr))
            group_violations[group] = float("nan") if not devs else sum(devs) / len(devs)

        if n_unscored:
            _warn_unscored("equalized_odds", n_unscored, len(y_pred))

        return ConstraintViolation(
            constraint_type="equalized_odds",
            # NaN <= tolerance is False, so an unmeasurable disparity is
            # could-not-check reported fail-closed, never a false PASS.
            overall_violation=max_violation,
            group_violations=group_violations,
            # `and not n_unscored` keeps the invariant this dataclass
            # documents: the field is never True for something nobody could
            # evaluate. An all-NaN prediction vector otherwise reads
            # 0.0 <= tolerance and returns a flat PASS.
            is_satisfied=bool(max_violation <= self.tolerance) and not n_unscored,
            tolerance=self.tolerance,
            details={
                "group_tprs": group_tprs,
                "group_fprs": group_fprs,
                "tpr_violation": tpr_violation,
                "fpr_violation": fpr_violation,
                "insufficient_data": bool(max_violation != max_violation) or bool(n_unscored),
                # How many rows entered the rates as fabricated decisions
                # rather than measured ones (0 in the normal case).
                "n_unscored_rows": n_unscored,
            },
        )

    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """Compute signed constraint value for equalized odds.

        Returns the SIGNED deviation (group rate minus mean rate) of
        whichever of the group's TPR/FPR gaps has the larger magnitude.
        The sign matters: exponentiated-gradient updates use it to decide
        whether to raise or lower a group's multiplier, and the old
        implementation (mean of absolute gaps) was never negative, so the
        multipliers could only grow.
        """
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        group_tprs = violation.details.get("group_tprs", {})
        group_fprs = violation.details.get("group_fprs", {})
        if group not in group_tprs:
            # Same reasoning as _steering_value: a group name that was never in
            # the data is a caller bug, and answering "no violation" for it is
            # indistinguishable from a measured perfect parity.
            raise KeyError(
                f"equalized_odds: no group {group!r} in the fitted sensitive attribute "
                f"(groups: {sorted(map(str, group_tprs))})."
            )

        # Center on the DEFINED rates and treat an undefined own-group rate as
        # a zero signed value (no multiplier push on an unmeasurable group),
        # rather than propagating NaN into the reduction.
        defined_tprs = _defined(group_tprs.values())
        defined_fprs = _defined(group_fprs.values())
        overall_tpr = float(np.mean(defined_tprs)) if defined_tprs else float("nan")
        overall_fpr = float(np.mean(defined_fprs)) if defined_fprs else float("nan")

        t, f = group_tprs[group], group_fprs[group]
        # An arm is MEASURED for this group only when the group's own rate is
        # defined AND the centre has at least two defined rates in it. A centre
        # built from a single rate is that group's own rate, so the gap is
        # `r - r` = 0.0 for every input: vacuous, not measured (G011,
        # 2026-09-17, the same fix as in compute_violation above).
        tpr_measured = t == t and not _deviation_is_undefined(len(defined_tprs))
        fpr_measured = f == f and not _deviation_is_undefined(len(defined_fprs))
        tpr_gap = (t - overall_tpr) if tpr_measured else 0.0
        fpr_gap = (f - overall_fpr) if fpr_measured else 0.0

        # Three states, never two, on the same terms as _steering_value: with
        # one group the centre IS this group's own rate, so both gaps are 0.0
        # by arithmetic vacuity, and with both rates undefined they are 0.0 by
        # substitution. Either way 0.0 means NO PUSH, not "no gap", and it now
        # says so rather than being byte-identical to a measured perfect parity.
        # The n_groups test the old code also made here is subsumed by the two
        # flags above: with one group neither arm can have a peer.
        if not (tpr_measured or fpr_measured):
            warnings.warn(
                f"equalized_odds: the constraint deviation for group {group!r} is "
                f"UNMEASURABLE (its TPR and FPR are undefined, or there is no second "
                f"group to compare it against). Returning 0.0 as a STEERING value, "
                f"meaning no push on this group's multiplier, NOT a measured absence "
                f"of disparity. The measurement is the NaN in "
                f"ConstraintViolation.group_violations[{group!r}].",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0.0

        return float(tpr_gap if abs(tpr_gap) >= abs(fpr_gap) else fpr_gap)


class EqualOpportunityConstraint(BaseFairnessConstraint):
    """
    Equal Opportunity Constraint.

    Requires equal TPR across groups:
        |TPR_a - TPR_b| <= ε

    Args:
        tolerance: Maximum allowed TPR disparity, as a DIFFERENCE of rates
        bound_type: 'difference' only. 'ratio' is refused, not ignored (F22).

    Example:
        >>> constraint = EqualOpportunityConstraint(tolerance=0.05)
        >>> violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: equal_opportunity_constraint. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """Compute equal opportunity violation."""
        y_pred = coerce_to_array(y_pred, "y_pred")
        y_true = coerce_to_array(y_true, "y_true")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        # Before the binarisation below, which turns a NaN into a confident
        # rejection. See _unscored_rows.
        n_unscored = _unscored_rows(y_pred)

        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        gm = GroupManager(sensitive_attr)

        group_tprs = {}
        group_violations = {}

        for group in gm.groups:
            mask = gm.get_mask(group)
            y_pred_g = y_pred_binary[mask]
            y_true_g = y_true[mask]

            positives = y_true_g == 1
            # Undefined TPR (no positive-label samples) is unmeasurable: NaN,
            # not a fabricated 0.5.
            tpr = float(np.mean(y_pred_g[positives])) if positives.sum() > 0 else float("nan")
            group_tprs[group] = tpr

        max_violation = _defined_spread(group_tprs.values())

        defined_tprs = _defined(group_tprs.values())
        overall_tpr = float(np.mean(defined_tprs)) if defined_tprs else float("nan")
        # Counted over the DEFINED rates, NOT over the groups (G011,
        # 2026-09-17). With three groups of which only one has any positive
        # labels, the centre IS that group's own rate, so its deviation was
        # `tpr - tpr` = exactly 0.0 by arithmetic vacuity while its two
        # siblings were honestly NaN. Measured before this change on 40 rows,
        # groups a|b|c where only a has positive labels (TPR_a = 0.8):
        # group_violations == {'a': 0.0, 'b': nan, 'c': nan}, and
        # signed_constraint_value(..., 'a') returned 0.0 with NO warning at
        # all, byte-identical to a measured perfect parity. That value is
        # spread into FairClassifierResult.fairness_metrics by
        # wrappers/sklearn_wrappers.py:285, where a reader sees `a: 0.0`
        # beside `insufficient_data: True`.
        # EqualOpportunityLoss already applies this rule the right way round:
        # `if len(tpr_rates) < 2` over the MEASURABLE rates, not over the
        # groups, at loss_functions/fairness_losses.py:621, reason string
        # "fewer_than_two_groups_with_positive_labels". The two siblings
        # disagreed and the permissive one was this method.
        no_peer = _deviation_is_undefined(len(defined_tprs))
        for group, tpr in group_tprs.items():
            group_violations[group] = (
                (tpr - overall_tpr)
                if (not no_peer and tpr == tpr and overall_tpr == overall_tpr)
                else float("nan")
            )

        if n_unscored:
            _warn_unscored("equal_opportunity", n_unscored, len(y_pred))

        return ConstraintViolation(
            constraint_type="equal_opportunity",
            overall_violation=max_violation,
            group_violations=group_violations,
            # `and not n_unscored` keeps the invariant this dataclass
            # documents: the field is never True for something nobody could
            # evaluate. An all-NaN prediction vector otherwise reads
            # 0.0 <= tolerance and returns a flat PASS.
            is_satisfied=bool(max_violation <= self.tolerance) and not n_unscored,
            tolerance=self.tolerance,
            details={
                "group_tprs": group_tprs,
                "overall_tpr": overall_tpr,
                "insufficient_data": bool(max_violation != max_violation) or bool(n_unscored),
                # How many rows entered the rates as fabricated decisions
                # rather than measured ones (0 in the normal case).
                "n_unscored_rows": n_unscored,
            },
        )

    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """Compute signed constraint value for equal opportunity.

        Three states, never two: see :func:`_steering_value`.
        """
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        return _steering_value(violation, group, "equal_opportunity")


class FalsePositiveRateParityConstraint(BaseFairnessConstraint):
    """
    False Positive Rate Parity Constraint.

    Requires equal FPR across groups:
        |FPR_a - FPR_b| <= ε

    Args:
        tolerance: Maximum allowed FPR disparity, as a DIFFERENCE of rates
        bound_type: 'difference' only. 'ratio' is refused, not ignored (F22).

    Example:
        >>> constraint = FalsePositiveRateParityConstraint(tolerance=0.05)
        >>> violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: false_positive_rate_parity_constraint. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """Compute FPR parity violation."""
        y_pred = coerce_to_array(y_pred, "y_pred")
        y_true = coerce_to_array(y_true, "y_true")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        # Before the binarisation below, which turns a NaN into a confident
        # rejection. See _unscored_rows.
        n_unscored = _unscored_rows(y_pred)

        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        gm = GroupManager(sensitive_attr)

        group_fprs = {}
        group_violations = {}

        for group in gm.groups:
            mask = gm.get_mask(group)
            y_pred_g = y_pred_binary[mask]
            y_true_g = y_true[mask]

            negatives = y_true_g == 0
            # Undefined FPR (no negative-label samples) is unmeasurable: NaN,
            # not a fabricated 0.5.
            fpr = float(np.mean(y_pred_g[negatives])) if negatives.sum() > 0 else float("nan")
            group_fprs[group] = fpr

        max_violation = _defined_spread(group_fprs.values())

        defined_fprs = _defined(group_fprs.values())
        overall_fpr = float(np.mean(defined_fprs)) if defined_fprs else float("nan")
        # Counted over the DEFINED rates, NOT over the groups: the mirror of
        # the equal-opportunity case documented above (G011, 2026-09-17).
        # Measured before this change on 40 rows, groups a|b|c where only a
        # has negative labels (FPR_a = 0.2): group_violations ==
        # {'a': 0.0, 'b': nan, 'c': nan} and signed_constraint_value(..., 'a')
        # returned a silent 0.0.
        no_peer = _deviation_is_undefined(len(defined_fprs))
        for group, fpr in group_fprs.items():
            group_violations[group] = (
                (fpr - overall_fpr)
                if (not no_peer and fpr == fpr and overall_fpr == overall_fpr)
                else float("nan")
            )

        if n_unscored:
            _warn_unscored("fpr_parity", n_unscored, len(y_pred))

        return ConstraintViolation(
            constraint_type="fpr_parity",
            overall_violation=max_violation,
            group_violations=group_violations,
            # `and not n_unscored` keeps the invariant this dataclass
            # documents: the field is never True for something nobody could
            # evaluate. An all-NaN prediction vector otherwise reads
            # 0.0 <= tolerance and returns a flat PASS.
            is_satisfied=bool(max_violation <= self.tolerance) and not n_unscored,
            tolerance=self.tolerance,
            details={
                "group_fprs": group_fprs,
                "overall_fpr": overall_fpr,
                "insufficient_data": bool(max_violation != max_violation) or bool(n_unscored),
                # How many rows entered the rates as fabricated decisions
                # rather than measured ones (0 in the normal case).
                "n_unscored_rows": n_unscored,
            },
        )

    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """Compute signed constraint value for FPR parity.

        Three states, never two: see :func:`_steering_value`. Measured at this
        entry before that change, on 8 rows with a single sensitive value:
        ``signed_constraint_value(..., "A")`` returned 0.0 and
        ``signed_constraint_value(..., "Z")`` - a group that does not exist -
        returned the same 0.0, as did a genuinely measured perfect parity.
        """
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        return _steering_value(violation, group, "fpr_parity")


class BoundedGroupLossConstraint(BaseFairnessConstraint):
    """
    Bounded Group Loss Constraint.

    Requires that the loss for any group does not exceed a bound:
        L_g <= (1 + ε) * L_overall for all groups g

    This implements a form of minimax fairness.

    Args:
        tolerance: Maximum allowed loss ratio excess
        loss_fn: Loss function to use (default: 0-1 loss)

    This is the one constraint here whose bound really is a RATIO bound, so it
    is the only class whose supported_bound_types says so. It takes no
    bound_type argument: the bound is a property of the technique, not a
    setting.

    Example:
        >>> constraint = BoundedGroupLossConstraint(tolerance=0.1)
        >>> violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)

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

    Ledger row: bounded_group_loss_constraint. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    supported_bound_types: ClassVar[Tuple[str, ...]] = ("ratio",)

    def __init__(
        self,
        tolerance: float = 0.1,
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__(tolerance=tolerance, bound_type="ratio")
        self.loss_fn = loss_fn or self._zero_one_loss
        # The largest value this constraint's loss can take, or None when that is
        # not knowable. It is what makes the bound's vacuity decidable: see
        # _vacuous_relative_bound_reason. The default 0-1 loss is a mean of a
        # boolean, so it lies in [0, 1] and 1.0 is exact. A caller-supplied
        # loss_fn has NO declared range, and None there is the could-not-check,
        # not "unbounded": the vacuity guard then stays silent and leaves the
        # bound exactly as usable as the caller configured it. A caller who knows
        # their own loss's ceiling may set this attribute to get the guard back.
        self.loss_upper_bound: Optional[float] = None if loss_fn is not None else 1.0

    @staticmethod
    def _zero_one_loss(y_pred, y_true):
        """Compute 0-1 loss (error rate)."""
        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        return np.mean(y_pred_binary != y_true)

    def compute_violation(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> ConstraintViolation:
        """Compute bounded group loss violation."""
        y_pred = coerce_to_array(y_pred, "y_pred")
        y_true = coerce_to_array(y_true, "y_true")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        # The default 0-1 loss binarises with the same `>= 0.5`, so an
        # unscored row becomes a confident rejection there too. See
        # _unscored_rows.
        n_unscored = _unscored_rows(y_pred)
        # This is the only constraint in the file that measures against the
        # LABELS rather than between group rates, so it is the only one an
        # unlabelled row can fabricate a pass for. See _unlabelled_rows for the
        # measured before and after.
        n_unlabelled = _unlabelled_rows(y_true)

        gm = GroupManager(sensitive_attr)

        overall_loss = self.loss_fn(y_pred, y_true)
        bound = (1 + self.tolerance) * overall_loss

        # THE GUARD SITS ABOVE THE LOOP AND ABOVE THE MAX (F9 wave 4,
        # 2026-09-30), because the bound is a precondition of BOTH comparisons
        # this method makes: the per-group one inside the loop and the population
        # one below it. A check placed in either branch alone would move the
        # fabrication into its sibling and leave the other channel publishing a
        # measured 0.0 -- and the per-group channel is the one an OPTIMISER reads
        # through signed_constraint_value, so that sibling is the dangerous one.
        #
        # `tolerance` is validated in BaseFairnessConstraint.__init__, so a NaN
        # or infinite bound can only arrive here through a NaN overall_loss or a
        # caller-supplied loss_fn. is_measured is still tested, because with a
        # NaN bound `max(0.0, finite_loss - nan)` is exactly 0.0: the per-group
        # entries used to read as fully compliant groups while max_violation
        # alone answered NaN, so the two channels disagreed and the one feeding
        # the optimiser was the one that lied. See _vacuous_relative_bound_reason
        # for the measured sweep behind the FINITE case, which is the door a
        # guard testing only isinf leaves wide open.
        vacuous_bound = _vacuous_relative_bound_reason(bound, self.loss_upper_bound)
        bound_is_unusable = vacuous_bound is not None or not is_measured(bound)

        group_losses = {}
        group_violations = {}

        for group in gm.groups:
            mask = gm.get_mask(group)
            group_loss = float(self.loss_fn(y_pred[mask], y_true[mask]))
            group_losses[group] = group_loss
            # An undefined group loss stays NaN and is never clamped to 0.
            # Python's builtin `max` returns its FIRST argument when the
            # comparison is False, and every comparison against NaN is False,
            # so `max(0, nan - bound)` is 0: a group nobody could measure was
            # recorded as a fully compliant group, with a per-group violation
            # indistinguishable from a measured pass.
            # The relative bound is (1 + tolerance) * overall_loss, and with a
            # single group the overall loss IS this group's loss, so
            # `max(0, L - 1.1 * L)` is 0.0 for every input. See
            # _deviation_is_undefined: that 0.0 is vacuous, not measured.
            group_violations[group] = (
                float("nan")
                # The group COUNT is the right argument here, not a count of
                # defined losses: this bound is against the POPULATION loss,
                # not against a peer group, so another group's loss being
                # unmeasurable does not make this group's comparison vacuous.
                # Only a single group does, because then the population loss
                # is this group's own.
                if (
                    group_loss != group_loss
                    or _deviation_is_undefined(len(gm.groups))
                    # The bound is as much a part of this comparison as the loss
                    # is. Decided once, above the loop, so this arm and the
                    # population arm below cannot disagree.
                    or bound_is_unusable
                )
                else max(0.0, group_loss - bound)
            )

        # THREE STATES, and the guard sits ABOVE the max so fixing the NaN arm
        # does not just move the fabrication into its sibling.
        #
        # The bound here is RELATIVE, (1 + tolerance) * overall_loss. With a
        # SINGLE group max_loss IS overall_loss, so the predicate collapses to
        # the tautology L <= (1 + tolerance) * L and the violation is
        # identically 0 for every possible input -- a constraint that cannot be
        # violated by construction. Measured on 60 rows, one group, a
        # classifier wrong on every single row (group_losses={'a': 1.0}): it
        # returned overall_violation=0, is_satisfied=True,
        # could_not_evaluate=False and no warning, which is exactly what a
        # measured pass returns. And because `max(0, nan - bound)` is 0, NaN
        # could never reach overall_violation either, so the NaN fallback in
        # ConstraintViolation.could_not_evaluate was unreachable for this class
        # -- the only constraint in this file that never wrote the flag.
        undefined_groups = [g for g, loss in group_losses.items() if loss != loss]
        if (
            len(group_losses) < 2
            or undefined_groups
            or overall_loss != overall_loss
            # The same single predicate the per-group arm consulted.
            or bound_is_unusable
        ):
            max_violation = float("nan")
        else:
            max_violation = max(0.0, max(group_losses.values()) - bound)

        if bound_is_unusable:
            _warn_unusable_bound(
                "bounded_group_loss",
                vacuous_bound
                if vacuous_bound is not None
                else f"the bound is {bound!r}, which is not a finite number",
                self.tolerance,
            )
        if n_unscored:
            _warn_unscored("bounded_group_loss", n_unscored, len(y_pred))
        if n_unlabelled:
            _warn_unlabelled("bounded_group_loss", n_unlabelled, len(y_true))

        return ConstraintViolation(
            constraint_type="bounded_group_loss",
            overall_violation=max_violation,
            group_violations=group_violations,
            # NaN <= 0 is False, so an unevaluable bound is fail-closed, never
            # a false PASS. Pair it with details["insufficient_data"].
            # `and not n_unscored` keeps the invariant this dataclass
            # documents: the field is never True for something nobody could
            # evaluate. An all-NaN prediction vector otherwise reads
            # 0.0 <= tolerance and returns a flat PASS.
            # `and not n_unlabelled` for the same reason on the label side, and
            # there it is the ONLY thing standing between an unlabelled dataset
            # and a compliance certificate: with every label NaN every group
            # loss is exactly 1.0, so `max(0, 1.0 - 1.1 * 1.0)` is 0.0 and
            # `0.0 <= 0` is a measured pass. See _unlabelled_rows.
            is_satisfied=bool(max_violation <= 0) and not n_unscored and not n_unlabelled,
            tolerance=self.tolerance,
            details={
                "group_losses": group_losses,
                "overall_loss": overall_loss,
                "bound": bound,
                "insufficient_data": (
                    bool(max_violation != max_violation) or bool(n_unscored) or bool(n_unlabelled)
                ),
                # How many rows entered the rates as fabricated decisions
                # rather than measured ones (0 in the normal case).
                "n_unscored_rows": n_unscored,
                # How many rows entered the group losses as fabricated MISSES
                # because they carried no label (0 in the normal case).
                "n_unlabelled_rows": n_unlabelled,
                # WHY no verdict was given, in words, for a reader who has the
                # object but was not there for the warning (None in the normal
                # case). A report renders this; insufficient_data alone says that
                # something could not be evaluated but never that the BOUND was
                # the thing at fault, which is the one cause the caller can fix
                # by changing a setting rather than by collecting more data.
                "unusable_bound_reason": (
                    (
                        vacuous_bound
                        if vacuous_bound is not None
                        else f"the bound is {bound!r}, which is not a finite number"
                    )
                    if bound_is_unusable
                    else None
                ),
            },
        )

    def signed_constraint_value(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,
        sensitive_attr: ArrayLike,
        group: str,
    ) -> float:
        """Compute signed constraint value for bounded group loss.

        A group whose loss is undefined carries NaN rather than a clamped 0
        (see compute_violation). ``.get(group, 0.0)`` could never catch that:
        the key is PRESENT, holding NaN, so the default never fires. Three
        states, never two: see :func:`_steering_value`.

        An UNLABELLED dataset is the one case this method cannot answer with a
        NaN, because the per-group value it reads is a real 0.0 derived from a
        fabricated loss of 1.0 (see :func:`_unlabelled_rows`). It answered 0.0,
        no violation, for a model wrong on 80% of one group, and in silence.
        The disclosure now travels with it: ``compute_violation`` is called on
        the way through, so the unlabelled-row warning fires from here too, and
        the ``ConstraintViolation`` it derived from carries
        ``details['insufficient_data'] is True``. That object, not this float,
        is the measurement channel; this float is an optimiser push.
        """
        violation = self.compute_violation(y_pred, y_true, sensitive_attr)
        return _steering_value(violation, group, "bounded_group_loss")


def create_constraint(
    constraint_type: Union[str, FairnessConstraintType],
    tolerance: float = 0.05,
    **kwargs,
) -> BaseFairnessConstraint:
    """
    Factory function to create fairness constraints.

    Args:
        constraint_type: Type of constraint to create
        tolerance: Maximum allowed violation
        **kwargs: Additional constraint-specific arguments

    Returns:
        Configured fairness constraint

    Example:
        >>> constraint = create_constraint('demographic_parity', tolerance=0.05)

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

    Ledger row: create_constraint. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    if isinstance(constraint_type, str):
        constraint_type = FairnessConstraintType(constraint_type)

    constraint_classes = {
        FairnessConstraintType.DEMOGRAPHIC_PARITY: DemographicParityConstraint,
        FairnessConstraintType.EQUALIZED_ODDS: EqualizedOddsConstraint,
        FairnessConstraintType.EQUAL_OPPORTUNITY: EqualOpportunityConstraint,
        FairnessConstraintType.FALSE_POSITIVE_RATE_PARITY: FalsePositiveRateParityConstraint,
        FairnessConstraintType.TRUE_POSITIVE_RATE_PARITY: EqualOpportunityConstraint,
        FairnessConstraintType.BOUNDED_GROUP_LOSS: BoundedGroupLossConstraint,
    }

    if constraint_type not in constraint_classes:
        # PREDICTIVE_PARITY and ERROR_RATE_PARITY are declared in the enum
        # but have no constraint class yet; say so honestly instead of the
        # misleading "Unknown constraint type" (the type IS known, just not
        # implemented).
        supported = sorted(ct.value for ct in constraint_classes)
        raise NotImplementedError(
            f"Constraint type '{constraint_type.value}' is declared in "
            f"FairnessConstraintType but has no implementation yet. "
            f"Supported types: {supported}"
        )

    return constraint_classes[constraint_type](tolerance=tolerance, **kwargs)
