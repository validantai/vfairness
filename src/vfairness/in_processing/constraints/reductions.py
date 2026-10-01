"""
Reductions Approach to Fair Classification.

This module implements the Reductions Approach (Agarwal et al., 2018) which
reduces fair classification to a sequence of cost-sensitive classification
problems. This enables using any standard classifier as a subroutine while
achieving fairness guarantees.

Key Algorithms:
    1. ExponentiatedGradient: Solves the Lagrangian game using multiplicative weights
    2. GridSearch: Sweeps over Lagrange multipliers for comparison
    3. ThresholdOptimizer: Post-processes predictions using optimal thresholds

The Lagrangian Formulation:
    min_θ max_λ L(θ, λ) = Loss(θ) + Σᵢ λᵢ * gᵢ(θ)

    where gᵢ(θ) are the constraint functions.

References:
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
    - Cotter et al. (2019): Optimization with Non-Differentiable Constraints
    - Narasimhan (2018): Learning with Complex Loss Functions and Constraints
"""

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)

from .base import (
    BaseFairnessConstraint,
    BoundedGroupLossConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FairnessConstraintType,
    FalsePositiveRateParityConstraint,
    OptimizationResult,
    _defined,
    create_constraint,
)


class _ConstantClassifier:
    """Degenerate classifier that always predicts one label.

    Used as a fallback when the cost-sensitive relabeling collapses every
    sample to the same class (a saturated Lagrange multiplier can do this);
    most sklearn estimators refuse to fit a single-class target, and the
    saturated best response IS the constant classifier.
    """

    def __init__(self, label: int):
        self.label = int(label)

    def fit(self, X, y, sample_weight=None) -> "_ConstantClassifier":
        return self

    def predict(self, X) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.label, dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p = np.full(n, float(self.label))
        return np.column_stack([1.0 - p, p])


# A constraint "coordinate" is (group, event) where event says which samples
# the constrained rate is computed over: "all" (selection rate), "pos"
# (TPR, over y==1) or "neg" (FPR, over y==0). Equalized odds has two
# coordinates per group; the other rate constraints have one.
_Coord = Tuple[str, str]


def _coord_key(coord: _Coord) -> str:
    group, event = coord
    if event == "pos":
        return f"{group}:tpr"
    if event == "neg":
        return f"{group}:fpr"
    return str(group)


def _reduction_coordinates(constraint: BaseFairnessConstraint, gm: GroupManager) -> List[_Coord]:
    """Constraint coordinates for the Agarwal et al. (2018) reduction."""
    if isinstance(constraint, EqualizedOddsConstraint):
        return [(g, "pos") for g in gm.groups] + [(g, "neg") for g in gm.groups]
    if isinstance(constraint, FalsePositiveRateParityConstraint):
        return [(g, "neg") for g in gm.groups]
    if isinstance(constraint, EqualOpportunityConstraint):
        return [(g, "pos") for g in gm.groups]
    # DemographicParityConstraint and unknown custom constraints are treated
    # as selection-rate constraints over all samples.
    return [(g, "all") for g in gm.groups]


def _signed_gaps(
    constraint: BaseFairnessConstraint,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive_attr: np.ndarray,
    coords: List[_Coord],
) -> Dict[_Coord, float]:
    """Signed per-coordinate constraint gaps g_j(h) (positive = rate above par)."""
    if isinstance(constraint, EqualizedOddsConstraint):
        violation = constraint.compute_violation(y_pred, y_true, sensitive_attr)
        tprs = violation.details.get("group_tprs", {})
        fprs = violation.details.get("group_fprs", {})
        # Centre on the DEFINED rates, exactly as signed_constraint_value does.
        # This branch bypasses that method and used to take the mean over ALL
        # rates, so ONE group with no positive labels made mean_tpr NaN and with
        # it EVERY coordinate's gap, not just its own. np.sign(nan) is nan, so
        # the multipliers went NaN, and the cost-sensitive sample weights with
        # them: GridSearch died inside sklearn with "Input sample_weight
        # contains NaN" (naming neither fairness nor the group), and
        # ExponentiatedGradient returned final_violation=nan with
        # constraint_satisfied=False and no warning at all. Measured 2026-09-08
        # on 400 rows whose 20-row group "c" had no positive labels.
        defined_tprs = _defined(tprs.values())
        defined_fprs = _defined(fprs.values())
        mean_tpr = float(np.mean(defined_tprs)) if defined_tprs else float("nan")
        mean_fpr = float(np.mean(defined_fprs)) if defined_fprs else float("nan")
        gaps: Dict[_Coord, float] = {}
        for group, event in coords:
            rates, mean_rate = (tprs, mean_tpr) if event == "pos" else (fprs, mean_fpr)
            own = float(rates.get(group, float("nan")))
            # An undefined own rate, or no defined rate to centre on, means this
            # coordinate cannot be steered. 0.0 is "no push", not "no gap": the
            # same choice signed_constraint_value documents, and the warning in
            # the callers below is what stops it reading as a measurement.
            gaps[(group, event)] = (
                own - mean_rate if (own == own and mean_rate == mean_rate) else 0.0
            )
        return gaps
    return {
        coord: float(constraint.signed_constraint_value(y_pred, y_true, sensitive_attr, coord[0]))
        for coord in coords
    }


_NEGLIGIBLE_WEIGHT = 1e-8


def _refuse_a_silent_ensemble(
    classifiers: List[Any],
    weights: Any,
    *,
    caller: str,
) -> None:
    """Refuse an ensemble that would answer without any member contributing.

    G007 (2026-09-17). ``predict`` and ``predict_proba`` accumulate
    ``weight * member_prediction`` into a zeros array and skip every member
    whose weight is not above 1e-8. With no classifiers, with every weight at
    zero, or with every weight NaN (``nan > 1e-8`` is False, so a weight
    nobody could evaluate is silently treated as zero), the zeros array is
    what comes back. Measured before this guard:

        EnsembleClassifier([], []).predict(X)            -> [0 0 0 0 0 0]
        EnsembleClassifier([c, c], [0.0, 0.0]).predict_proba(X) -> [0. 0. ...]

    which is a confident REJECT for every row and a probability of exactly
    0.0, byte-identical to a model that looked at the features and turned
    everyone down. Both classes are exported from
    ``vfairness.in_processing.constraints``, so a caller can build one
    directly (from a stored ``ReductionResult.to_dict`` payload, for
    instance); the fitted paths in this module always hand over one-hot or
    uniform weights and never reach it.

    The returned arrays are a bare int array and a bare float array with no
    field that can carry "nothing was combined", so the refusal IS the third
    state, the same argument the unscored-row refusal below makes.

    A length mismatch is refused for the same reason: ``zip`` stops at the
    shorter of the two, so ``EnsembleClassifier([a, b, c], [1.0])`` predicted
    from ONE member while reporting three, and nothing said so.
    """
    weights_arr = np.asarray(weights, dtype=float).ravel()
    if len(classifiers) != weights_arr.size:
        raise ValueError(
            f"{caller}: {len(classifiers)} classifier(s) but {weights_arr.size} weight(s). "
            "These are zipped together, so the shorter one silently decides how many "
            "members vote and the extra members are dropped without a word. Pass one "
            "weight per classifier."
        )
    n_nan = int(np.isnan(weights_arr).sum())
    if n_nan:
        raise ValueError(
            f"{caller}: {n_nan} of {weights_arr.size} ensemble weight(s) are NaN. A NaN "
            "weight compares False against the negligible-weight cutoff, so its member "
            "would be dropped from the vote as though it had been weighted zero, which "
            "is a decision nobody made. Supply a real weight, or drop the member."
        )
    # A NEGATIVE weight is the same silent drop as a NaN one, by the same
    # comparison: -3.0 > 1e-8 is False. Measured 2026-09-27,
    # EnsembleClassifier([p(0.9), p(0.1)], [1.0, -3.0]).predict_proba(X) -> 0.9
    # for every row, the first member's answer alone, with three members'-worth
    # of weight reported and no warning. The NaN branch above was written for
    # exactly this failure and stopped one value short of it.
    # AN INFINITE WEIGHT IS THE THIRD VALUE ON THIS AXIS (F9 wave 4, 2026-09-30).
    # The NaN branch above and the negative branch below were both written for a
    # weight that is not a usable number, and +inf is neither: `inf > 1e-8` is
    # True, so it CONTRIBUTES, and it then dominates the sum absolutely. Measured
    # 2026-09-30, five members with one predicting 1 and four predicting 0, the
    # first weighted inf: predict returned 1 for every row, the opposite of the
    # correct 1-of-5 minority verdict, and the weight-scale guard could not catch
    # it because its inversion test compares the largest weight against total / 2
    # and `inf < inf` is False. It is refused here rather than there so that all
    # four entry points get it from the one guard they already share.
    n_infinite = int(np.isinf(weights_arr).sum())
    if n_infinite:
        raise ValueError(
            f"{caller}: {n_infinite} of {weights_arr.size} ensemble weight(s) are "
            f"infinite. An infinite weight clears the negligible-weight cutoff and then "
            "dominates the weighted sum absolutely, so that one member alone decides "
            "every row while the others are reported as voting, and no scale-based check "
            "can see it because every comparison between infinities is degenerate. Mixing "
            "weights of a convex combination are finite."
        )
    n_negative = int((weights_arr < 0.0).sum())
    if n_negative:
        raise ValueError(
            f"{caller}: {n_negative} of {weights_arr.size} ensemble weight(s) are "
            f"negative (smallest {float(weights_arr.min())!r}). A negative weight compares "
            "False against the negligible-weight cutoff exactly as NaN does, so its member "
            "is dropped from the vote in silence while the reported ensemble still counts "
            "it. Mixing weights of a convex combination are nonnegative."
        )
    if weights_arr.size == 0 or not bool((weights_arr > _NEGLIGIBLE_WEIGHT).any()):
        raise ValueError(
            f"{caller}: no ensemble member carries a weight above {_NEGLIGIBLE_WEIGHT:g} "
            f"({weights_arr.size} classifier(s), largest weight "
            f"{float(weights_arr.max()) if weights_arr.size else float('nan')!r}). Nothing "
            "would be combined, and the all-zeros array that used to come back reads as a "
            "confident rejection of every row (or a probability of exactly 0.0) rather "
            "than as an ensemble that made no prediction at all."
        )


_VOTE_THRESHOLD = 0.5


def _contributing_weight_total(weights: Any) -> float:
    """Total of the weights that clear the negligible-weight cutoff."""
    weights_arr = np.asarray(weights, dtype=float).ravel()
    return float(weights_arr[weights_arr > _NEGLIGIBLE_WEIGHT].sum())


def _refuse_a_vote_that_cannot_reach_the_threshold(
    weights: Any,
    *,
    caller: str,
) -> None:
    """Refuse an ensemble whose label is decided by its weights, not its members.

    BGL3 in_processing-1 (2026-09-27). ``predict`` accumulates
    ``weight * member_label`` and returns ``sum >= 0.5``. Every member label is
    at most 1, so when the contributing weights total less than 0.5 the sum is
    below the threshold for EVERY possible X and EVERY possible member output,
    and the method returns the all-zeros array: a confident REJECTION of every
    row, decided before any classifier was consulted. Measured through the
    public API on six rows, with two members that both predict 1 at probability
    1.0::

        EnsembleClassifier([p(1.0), p(1.0)], [0.2, 0.2]).predict(X) -> [0 0 0 0 0 0]
        EnsembleClassifier([p(1.0)], [0.49]).predict(X)             -> [0 0 0 0 0 0]

    with zero warnings. That is byte-identical to the array
    :func:`_refuse_a_silent_ensemble` exists to refuse, and it walked straight
    past that guard because 0.2 IS above the negligible-weight cutoff: the guard
    asked whether any member contributes, never whether what they contribute can
    change the answer.

    The fitted paths in this module always hand over one-hot or uniform weights,
    which total exactly 1.0, so this cannot fire on them. It is reachable the way
    the guard above is: from a ``ReductionResult.to_dict`` payload, or by
    building either class directly (both are exported from
    ``vfairness.in_processing.constraints``).

    The returned array is a bare int array with no field that can carry "the
    weights decided this", so the refusal IS the third state.
    """
    total = _contributing_weight_total(weights)
    if total < _VOTE_THRESHOLD:
        raise ValueError(
            f"{caller}: the contributing ensemble weights total {total:g}, below the "
            f"{_VOTE_THRESHOLD:g} decision threshold. A weighted vote of labels in [0, 1] "
            f"can never reach {_VOTE_THRESHOLD:g} from there, so every row would be "
            "predicted 0 whatever the members say, and the all-zeros array that comes "
            "back reads as a confident rejection of every row rather than as an ensemble "
            "whose weights decided the answer. Normalize the weights so they sum to 1."
        )


def _refuse_a_vote_the_weight_scale_decides(
    weights: Any,
    *,
    caller: str,
) -> None:
    """Refuse the OTHER side of the same normalisation the guard above refuses.

    F9 wave 4 (2026-09-30). :func:`_refuse_a_vote_that_cannot_reach_the_threshold`
    refuses a contributing weight total BELOW ``_VOTE_THRESHOLD``, and its own
    message ends "Normalize the weights so they sum to 1". Nothing checked the
    other half of that sentence, so this was a ONE-SIDED guard on a TWO-SIDED
    quantity: below 0.5 it raised, above 1.0 it silently INVERTED the verdict for
    every row.

    ``predict`` returns ``sum(weight * member_label) >= 0.5``, which is a
    weighted MAJORITY only while the weights total 1. Above that, 0.5 stops being
    the majority point and the vote degenerates toward an OR gate: any single
    member whose weight alone reaches 0.5 carries the row. Measured with five
    members, ONE predicting 1 and FOUR predicting 0, six rows, labels strictly in
    {0, 1} so the label-domain guard is satisfied::

        weights [0.2]*5  (total 1.0)  -> np.unique(predict(X)) == [0]  0 warnings
        weights [1.0]*5  (total 5.0)  -> np.unique(predict(X)) == [1]  0 warnings
        weights [5.0]*5  (total 25.0) -> np.unique(predict(X)) == [1]  0 warnings

    Identical members, identical features, and the verdict for EVERY row flipped
    from reject to ACCEPT on the weight SCALE alone, with no warning of any kind.
    That is the mirror image of the all-zeros array the guard above exists to
    refuse, and this library already knows it: ``predict_proba`` on the very same
    unnormalised weights DOES disclose it ("the contributing ensemble weights
    total 5, not 1. The returned values are a weighted sum on a [0, 5] scale").
    ``predict``, which returns the DECISIONS a person acts on, carried no such
    line. One of two entry points, so this is called from BOTH
    ``ReductionResult.predict`` and ``EnsembleClassifier.predict``.

    TWO DISPOSITIONS, BECAUSE THE QUANTITY HAS TWO SEVERITIES, and getting this
    wrong in either direction is a defect of its own.

    The REFUSAL is keyed on the inversion itself, not on the size of the total: a
    single member whose weight alone reaches the threshold while its NORMALISED
    share is below half. That is precisely "any one member carries the row", the
    OR gate measured above, and it is the mirror of the all-zeros array refused
    below ``_VOTE_THRESHOLD``. A refusal rather than a warning for the reason the
    guard above gives verbatim: the returned array is a bare int array with no
    field that can carry "the weights decided this", so the refusal IS the third
    state. An infinite weight lands here too, its total being inf.

    Keying the refusal on ``total > 1`` instead would be the OVER-CORRECTION, and
    it was measured: three members at 0.34 total 1.02, which is a caller rounding
    1/3 to two decimals. No single member can reach 0.5 from 0.34, the verdict is
    the correct one, and refusing it would break a working ensemble over two
    hundredths. So that case is DISCLOSED and not refused, on the same 1e-6
    criterion ``_warn_about_vote_fractions`` already applies to the identical
    total, so the two entry points can no longer disagree about one number.

    The fitted paths in this module always hand over one-hot or uniform weights,
    which total exactly 1.0, so neither arm can fire on them; the reachable routes
    are a ``ReductionResult.to_dict`` payload and direct construction, both
    exported from ``vfairness.in_processing.constraints``. The over-correction
    controls in the suite pin the healthy side: one member at 1.0 and five at 0.2
    are silent and correct, three at 0.34 keeps its correct verdict with the scale
    disclosed, and 0.2 alone still raises the threshold refusal below.
    """
    weights_arr = np.asarray(weights, dtype=float).ravel()
    contributing = weights_arr[weights_arr > _NEGLIGIBLE_WEIGHT]
    total = _contributing_weight_total(weights_arr)
    if contributing.size:
        largest = float(contributing.max())
        # `largest >= threshold` means that member alone carries the row;
        # `largest < total / 2` means it is a MINORITY of the ensemble's weight.
        # Both together are the inversion. With a normalised total of 1 the two
        # conditions are `largest >= 0.5` and `largest < 0.5`, which cannot hold
        # at once, so a normalised vote can never reach this refusal.
        if largest >= _VOTE_THRESHOLD and largest < total / 2.0:
            raise ValueError(
                f"{caller}: the contributing ensemble weights total {total:g}, not 1, and "
                f"the largest single weight is {largest:g}. The decision is "
                f"`sum(weight * member_label) >= {_VOTE_THRESHOLD:g}`, which is a weighted "
                f"MAJORITY only while the weights total 1: at this scale that one member "
                f"reaches the threshold on its own while holding only "
                f"{largest / total:.1%} of the ensemble's weight, so the vote has "
                f"degenerated into an OR gate and the verdict for every row can flip from "
                f"reject to accept on the weight SCALE alone, with the members and the "
                f"features unchanged. That is the mirror of the all-zeros array refused "
                f"below {_VOTE_THRESHOLD:g}, and it comes back as a bare int array with no "
                "field that can say the weights decided it. Normalize the weights so they "
                "sum to 1."
            )
    if abs(total - 1.0) > 1e-6:
        warnings.warn(
            f"{caller}: the contributing ensemble weights total {total:g}, not 1, so this "
            f"is not a weighted majority vote: the "
            f"`>= {_VOTE_THRESHOLD:g}` threshold sits at "
            f"{_VOTE_THRESHOLD / total:.1%} of the ensemble's weight rather than at 50%, "
            f"which makes a positive decision easier or harder than the members "
            f"themselves warrant. The labels are returned as computed rather than "
            f"replaced. Normalize the weights so they sum to 1. Until 2026-09-30 only "
            f"predict_proba disclosed this total, while predict is the method a decision "
            f"is taken from.",
            UserWarning,
            stacklevel=3,
        )


def _as_numeric_labels(labels: Any) -> Optional[np.ndarray]:
    """``labels`` as a flat float array, or None when they are not numbers.

    No ``except``: a handler whose whole body is ``return None`` discards the
    error and substitutes an answer, which is the shape
    ``tests/test_no_silent_swallow_core.py`` refuses in this subpackage. The
    question here is answerable without raising, and "not numeric" is a
    CLASSIFICATION rather than a swallowed failure: the caller hands such labels
    straight to ``weight * label``, which raises on its own.
    """
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.dtype.kind in "biuf":
        return arr.astype(float).ravel()
    if arr.dtype.kind == "O":
        flat = arr.ravel().tolist()
        if flat and all(isinstance(v, (bool, int, float, np.number)) for v in flat):
            return np.asarray(flat, dtype=float)
    return None


def _refuse_a_vote_the_label_domain_decides(
    contributions: List[Tuple[Any, float, Any]],
    *,
    caller: str,
) -> None:
    """Refuse an ensemble whose label is decided by its label ENCODING.

    BGL4 (2026-09-27), the other side of
    :func:`_refuse_a_vote_that_cannot_reach_the_threshold`. That guard reads the
    WEIGHTS only, and its own premise is "a weighted vote of labels in [0, 1]".
    Nothing checked the premise. With member labels drawn from {1, 2} the sum
    ``weight * label`` can never fall BELOW 0.5, so ``sum >= 0.5`` is 1 for every
    row, for every possible X and every possible member output: the mirror image
    of the all-zeros array the weight guard exists to refuse, and decided before
    any classifier is consulted. Measured on six rows with weights [0.5, 0.5],
    which total exactly 1 and therefore pass every other guard::

        two members that both predict 2      -> [1 1 1 1 1 1], 0 warnings BEFORE
        two members that disagree, 1 and 2   -> [1 1 1 1 1 1], 0 warnings BEFORE
        two members fitted on y in {1, 2}
          that both predict 1                -> [1 1 1 1 1 1], 0 warnings BEFORE

    AFTER, each of the three raises ValueError naming the observed or declared
    labels. The first two are caught by the values the members RETURNED; the
    third returns nothing but 1, which is indistinguishable from an honest
    all-positive 0/1 model, so it is caught by the member's DECLARED
    ``classes_``, which says 2 is in its label domain.

    WHAT IT DELIBERATELY DOES NOT DO is refuse a 0/1 ensemble that happens to
    answer 1 (or 0) for every row. That array is the model's own measured
    decision, arrived at through the vote the threshold is defined on, and the
    all-zeros case with honest weights is not refused either; a collapsed FIT is
    disclosed by ``fairness_metrics['degenerate_constant_predictions']`` instead.
    Measured control, weights [0.25, 0.75] over members predicting 0 and 1:
    0.25*0 + 0.75*1 == 0.75 -> [1 1 1 1 1 1] silently, before and after.

    Non-numeric labels are left alone: ``weight * clf.predict(X)`` already fails
    loudly on them, and this guard must not turn that into a different error.
    """
    observed: set = set()
    declared: set = set()
    for clf, _weight, labels in contributions:
        as_numbers = _as_numeric_labels(labels)
        if as_numbers is None:
            # Not a numeric label at all (string classes): `weight * label`
            # below this guard raises on it, which is already a loud refusal,
            # and there is no [0, 1] domain question to answer about it.
            return
        observed.update(np.unique(as_numbers).tolist())
        declared_numbers = _as_numeric_labels(getattr(clf, "classes_", None))
        if declared_numbers is not None:
            declared.update(np.unique(declared_numbers).tolist())

    outside_observed = sorted(v for v in observed if v not in (0.0, 1.0))
    outside_declared = sorted(v for v in declared if v not in (0.0, 1.0))
    if not outside_observed and not outside_declared:
        return

    if outside_observed:
        seen = f"returned the label(s) {outside_observed}"
    else:
        seen = f"declare the class(es) {outside_declared} in classes_"
    raise ValueError(
        f"{caller}: contributing ensemble member(s) {seen}, outside the [0, 1] label "
        f"domain this vote is defined on. The prediction is "
        f"`sum(weight * member_label) >= {_VOTE_THRESHOLD:g}`, so with no 0 in the label "
        f"set the sum can never fall below the threshold and every row is predicted 1 "
        f"whatever the features say, which reads as a confident acceptance of every row "
        f"rather than as an encoding the ensemble could not vote on. Encode the labels as "
        f"0/1 before fitting the members."
    )


def _warn_about_vote_fractions(
    classifiers: List[Any],
    weights: Any,
    *,
    caller: str,
) -> None:
    """Say so when `predict_proba` is returning votes rather than probabilities.

    G007 (2026-09-17). A contributing member without ``predict_proba`` has its
    HARD 0/1 label folded into the same sum as its neighbours' probabilities,
    and the result is returned through a method named ``predict_proba`` with no
    signal of the substitution. Measured before this warning, through the
    public API, with ``LinearSVC`` as the base estimator:

        ExponentiatedGradient(LinearSVC(), "demographic_parity").fit(...)
        .predict_proba(X)  ->  np.unique(...) == [0., 1.],  zero warnings

    A caller reading that as a score (AUC, a calibration curve, a
    score-thresholded fairness metric) is reading hard labels. The VALUE is
    kept: a weighted vote fraction over the ensemble is a real quantity, and
    replacing it with NaN would throw away a usable signal. What was missing
    is the disclosure.

    BGL3 in_processing-1 (2026-09-27) adds the two members of that same class
    the ``hasattr`` test could not see:

    * ``_ConstantClassifier`` HAS a ``predict_proba``, so it slipped past the
      test written for hard-label members while returning exactly 0.0 or 1.0 for
      every row. It is this module's OWN marker for a collapsed best response
      (``_fit_best_response`` substitutes it when the cost-sensitive relabeling
      leaves one class), so it is the likeliest such member of all. Measured::

          EnsembleClassifier([_ConstantClassifier(0)], [1.0]).predict_proba(X)
          -> np.unique(...) == [0.], zero warnings

      A probability of exactly 0.0 on every row of a 240-row test set is the
      value ``_refuse_a_silent_ensemble`` names as "byte-identical to a model
      that looked at the features and turned everyone down".
    * Weights that are not a convex combination put the result on a
      ``[0, total]`` scale instead of ``[0, 1]``: with weights totalling 0.4 two
      members that both report probability 1.0 come back as 0.4, and with a
      total above 1 the "probability" can exceed 1.
    """
    weights_arr = np.asarray(weights, dtype=float).ravel()
    n_hard = 0
    n_constant = 0
    for clf, weight in zip(classifiers, weights_arr):
        if weight <= _NEGLIGIBLE_WEIGHT:
            continue
        if not hasattr(clf, "predict_proba"):
            n_hard += 1
        elif isinstance(clf, _ConstantClassifier):
            n_constant += 1
    if n_hard or n_constant:
        parts = []
        if n_hard:
            parts.append(f"{n_hard} have no predict_proba")
        if n_constant:
            parts.append(
                f"{n_constant} are collapsed constant classifiers whose predict_proba is "
                "exactly 0.0 or 1.0 for every row"
            )
        warnings.warn(
            f"{caller}: {n_hard + n_constant} contributing ensemble member(s) return HARD "
            f"0/1 values rather than probabilities ({' and '.join(parts)}), and those are "
            "summed in beside the other members' probabilities. The returned values are "
            "weighted VOTE FRACTIONS over the ensemble, not calibrated probabilities, and "
            "with a single such member they are exactly that member's 0/1 labels. Use "
            "predict() for labels, or pass a base estimator that exposes predict_proba.",
            UserWarning,
            stacklevel=3,
        )
    total = _contributing_weight_total(weights_arr)
    if abs(total - 1.0) > 1e-6:
        warnings.warn(
            f"{caller}: the contributing ensemble weights total {total:g}, not 1. The "
            f"returned values are a weighted sum on a [0, {total:g}] scale, not "
            "probabilities: a member reporting 1.0 comes back scaled by the total, so "
            "these must not be read as a score or fed to a calibration curve. Normalize "
            "the weights so they sum to 1.",
            UserWarning,
            stacklevel=3,
        )


def _warn_about_a_score_column_of_hard_values(
    scores: np.ndarray,
    *,
    caller: str,
) -> None:
    """Say so when the column ``predict_proba`` RETURNED is not a score at all.

    BGL4 (2026-09-27). The sibling disclosure above asks
    ``isinstance(clf, _ConstantClassifier)``: a TYPE, and this module's own type.
    So the disclosure fired for our marker for a collapsed best response and NOT
    for any other estimator that returns the same values. Measured on 30 rows,
    the foreign member being ``DummyClassifier(strategy='most_frequent')``, which
    HAS a ``predict_proba`` and is the very estimator the collapsed-fit fixture
    in the suite uses::

        EnsembleClassifier([_ConstantClassifier(0)], [1.0]).predict_proba(X)
          -> unique [0.]  warnings 1   (before and after)
        EnsembleClassifier([DummyClassifier('most_frequent').fit(X, y)], [1.0])
          .predict_proba(X)
          -> unique [0.]  warnings 0   BEFORE
          -> unique [0.]  warnings 1   AFTER
        ReductionResult.predict_proba, same foreign member
          -> unique [0.]  warnings 0   BEFORE  ->  1 AFTER
        ExponentiatedGradient(DummyClassifier('most_frequent'), ...).fit(...)
          .predict_proba(X)
          -> unique [0.]  warnings 0   BEFORE  ->  1 AFTER

    A probability of exactly 0.0 on every row is the value
    :func:`_refuse_a_silent_ensemble` calls "byte-identical to a model that
    looked at the features and turned everyone down", and for a caller-built
    ensemble there is no fit-time warning anywhere to compensate.

    THE TEST IS ON THE RETURNED VALUES, so it cannot be defeated by a member
    class nobody here has heard of. It fires when every returned value sits
    exactly on {0.0, 1.0} (hard votes wearing the name of a probability) or when
    no value is usable at all.

    WHAT IT DELIBERATELY DOES NOT DO is fire on a CONSTANT interior value. Two
    over-correction controls in the suite pin that: 0.25*0.2 + 0.75*0.8 == 0.65
    and 0.5*0.3 + 0.5*0.7 == 0.5 are constant across rows and both are real
    weighted probabilities, asserted silent. Constancy alone is a property of
    the members' own output; being pinned to the 0/1 endpoints is what makes the
    column a vote rather than a score. The VALUES are kept either way, exactly
    as the two warnings above keep theirs: what was missing is the disclosure.
    """
    arr = np.asarray(scores, dtype=float).ravel()
    if arr.size == 0:
        return
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        warnings.warn(
            f"{caller}: not one of {arr.size} returned value(s) is finite, so the column "
            "carries no score at all. Do not read it as a probability.",
            UserWarning,
            stacklevel=3,
        )
        return
    # THE TOTALLY UNUSABLE COLUMN WAS DISCLOSED AND THE PARTIALLY UNUSABLE ONE WAS
    # SILENT (F9 wave 4, 2026-09-30). The arm above fires only when NOT ONE value
    # is finite, and the {0.0, 1.0} test below runs on `finite` alone, so a column
    # with SOME unusable values reached neither. Measured on one member and ten
    # rows::
    #
    #   healthy constant 0.3           -> unique [0.3]                     0 warnings
    #   healthy spread 0.1..0.95       -> unique [0.1 ... 0.95]            0 warnings
    #   TOTAL hard zeros               -> unique [0.]                      1 warning
    #   TOTAL non-finite, 10 of 10 NaN -> unique [nan]                     1 warning
    #   PARTIAL 5 of 10 NaN            -> unique [0.3 0.4 0.5 0.6 0.7 nan] 0 WARNINGS
    #   PARTIAL 9 of 10 NaN            -> unique [0.42 nan]                0 WARNINGS
    #   PARTIAL 5 of 10 +inf           -> unique [0.3 ... inf]             0 WARNINGS
    #
    # Nine of ten rows carrying no score at all was published as a probability
    # column with LESS disclosure than one row of hard 0.0 gets. Downstream it is
    # worse than silence: every threshold comparison against NaN is False, so an
    # unscored row becomes a confident REJECTION, which is the same sentence the
    # arm above uses to justify itself. Reached through
    # ExponentiatedGradient.predict_proba too, where the collapsed-fit branch reads
    # only its flag: measured 3 of 60 values NaN with the flag correctly False and
    # ZERO warnings.
    #
    # NOT an early return: when the finite part is also pinned to {0.0, 1.0} both
    # facts are true of the same column and a caller needs both. The VALUES are
    # kept, as every other disclosure here keeps its values.
    n_unusable = int(arr.size - finite.size)
    if n_unusable:
        warnings.warn(
            f"{caller}: {n_unusable} of {arr.size} returned value(s) carry no usable "
            f"score (NaN or infinite), so this column is part measurement and part "
            f"nothing. Those rows cannot be thresholded or ranked: every comparison "
            f"against NaN is False, so an unscored row silently becomes a confident "
            f"rejection, and +inf a confident acceptance. The values are returned as "
            f"they were computed rather than replaced; drop or impute the unscored rows "
            f"before reading this as a probability column.",
            UserWarning,
            stacklevel=3,
        )
    # np.unique, never a variance or a centred sum of squares: np.var of a
    # constant array is exactly 0.0 only for some values and some n, and a
    # guard keyed on it passes for the round number in a fixture and fails on
    # real data.
    distinct = np.unique(finite)
    if not bool(np.isin(distinct, (0.0, 1.0)).all()):
        return
    shape = "the same value" if distinct.size == 1 else "hard 0/1 values"
    warnings.warn(
        f"{caller}: every one of {arr.size} returned value(s) is exactly 0.0 or 1.0 "
        f"(observed {[float(v) for v in distinct]}), so this column is {shape} rather "
        "than a probability: it cannot rank one row above another and a value of exactly "
        "0.0 on every row reads as a confident rejection of every row. This is the "
        "ensemble's own output, whatever its members are, so it is reported as measured "
        "and disclosed rather than replaced. Use predict() for labels, and check a "
        "collapsed base estimator (fairness_metrics['degenerate_constant_predictions'] "
        "when the ensemble came from a fit).",
        UserWarning,
        stacklevel=3,
    )


def _pick_smallest_violation(violations: List[float]) -> Tuple[int, int]:
    """Index of the smallest MEASURED violation, and how many were unmeasurable.

    `np.argmin` over a list containing NaN returns the NaN's index, so
    "the classifier with the smallest violation" used to select the one whose
    violation could NOT be measured, and then ship it as the fitted model.
    Verified 2026-09-08: ``np.argmin([0.5, nan, 0.2, 0.05])`` is 1 and
    ``np.nanargmin`` is 3.

    Returns index -1 when not one candidate had a measurable violation, so the
    caller can say so rather than claim a comparison it never made.
    """
    finite = [i for i, v in enumerate(violations) if v == v and abs(v) != float("inf")]
    n_unmeasurable = len(violations) - len(finite)
    if not finite:
        return -1, n_unmeasurable
    return min(finite, key=lambda i: violations[i]), n_unmeasurable


def _cost_sensitive_relabel(
    y: np.ndarray,
    gm: GroupManager,
    coords: List[_Coord],
    phi: Dict[_Coord, float],
    base_weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cost-sensitive best response of Agarwal et al. (2018), Sec. 3 (Eq. 6-8).

    Builds per-example costs where the SIGNED net multiplier phi_j of each
    constraint coordinate enters the cost of predicting POSITIVE for the
    samples that coordinate's rate is computed over. The classification is
    then reduced to weighted binary classification: train on labels
    z_i = argmin_h C_i^h with weights |C_i^1 - C_i^0| (labels are flipped
    where the net cost of predicting 1 is negative).

    net_i below is n * (C_i^1 - C_i^0):
      base misclassification term: (1 - 2*y_i) * n * w_i
      selection-rate coordinate g: + phi_g * n / n_g - sum_g' phi_g'
      TPR coordinate g (y==1):     + (phi_g - mean_g' phi_g') * n / n_{g,pos}
      FPR coordinate g (y==0):     + (phi_g - mean_g' phi_g') * n / n_{g,neg}

    (The TPR/FPR mean terms mirror compute_violation, which centers each
    group's rate on the unweighted mean of the group rates.)

    Returns:
        (z, w): relabeled 0/1 targets and nonnegative sample weights,
        normalized to mean 1.
    """
    n = len(y)
    y = np.asarray(y)
    net = (1.0 - 2.0 * y) * base_weight * n

    events = {event for _, event in coords}

    if "all" in events:
        phi_sum = sum(phi[c] for c in coords if c[1] == "all")
        for group in gm.groups:
            mask = gm.get_mask(group)
            n_g = mask.sum()
            if n_g > 0 and (group, "all") in phi:
                net[mask] += phi[(group, "all")] * n / n_g
        net -= phi_sum

    if "pos" in events:
        pos_phis = [phi[c] for c in coords if c[1] == "pos"]
        phi_bar = float(np.mean(pos_phis)) if pos_phis else 0.0
        for group in gm.groups:
            if (group, "pos") not in phi:
                continue
            mask = gm.get_mask(group) & (y == 1)
            n_gp = mask.sum()
            if n_gp > 0:
                net[mask] += (phi[(group, "pos")] - phi_bar) * n / n_gp

    if "neg" in events:
        neg_phis = [phi[c] for c in coords if c[1] == "neg"]
        phi_bar = float(np.mean(neg_phis)) if neg_phis else 0.0
        for group in gm.groups:
            if (group, "neg") not in phi:
                continue
            mask = gm.get_mask(group) & (y == 0)
            n_gn = mask.sum()
            if n_gn > 0:
                net[mask] += (phi[(group, "neg")] - phi_bar) * n / n_gn

    weights = np.abs(net)
    total = weights.sum()
    if total <= 1e-12:
        # No signal at all (can only happen with degenerate inputs):
        # fall back to the plain classification problem.
        return y.astype(int), np.ones(n)

    z = (net < 0).astype(int)
    # Normalize to MEAN 1 (sum n), not sum 1: sklearn estimators scale their
    # data-fit term by the weights but NOT their regularization penalty, so
    # sum-1 weights would shrink the likelihood by a factor n against the
    # penalty and cripple every fit.
    return z, weights / total * n


def _fit_best_response(base_estimator: Any, X: np.ndarray, z: np.ndarray, w: np.ndarray) -> Any:
    """Fit the base estimator on the relabeled problem, tolerating collapse."""
    if len(np.unique(z)) < 2:
        return _ConstantClassifier(int(z[0])).fit(X, z)
    clf = deepcopy(base_estimator)
    try:
        clf.fit(X, z, sample_weight=w)
    except TypeError:
        # Estimator doesn't support sample_weight
        clf.fit(X, z)
    return clf


@dataclass
class LagrangianState:
    """
    State of the Lagrangian optimization.

    Attributes:
        lambda_: Current Lagrange multipliers per group
        classifier_weights: Weights for ensemble of classifiers
        best_classifier_idx: Index of best classifier so far
        iteration: Current iteration number
    """

    lambda_: Dict[str, float]
    classifier_weights: List[float]
    best_classifier_idx: int
    iteration: int


@dataclass
class ReductionResult:
    """
    Result from a reductions-based fair learning algorithm.

    Attributes:
        classifiers: List of fitted classifiers
        weights: Mixing weights for classifiers
        optimization_result: Detailed optimization results
        final_violation: Final constraint violation
        accuracy: Final accuracy
        fairness_metrics: Dictionary of fairness metric values
    """

    classifiers: List[Any]
    weights: np.ndarray
    optimization_result: OptimizationResult
    final_violation: float
    accuracy: float
    # Heterogeneous values: constraint_satisfied (bool), constraint_type (str),
    # tolerance (float), plus per-metric floats.
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Make predictions using weighted ensemble.

        Args:
            X: Feature matrix

        Returns:
            Ensemble predictions

        Raises:
            ValueError: if no member of the ensemble would contribute, or if the
                contributing weights cannot reach the 0.5 decision threshold, so
                the zeros array would come back as a confident rejection of every
                row, or if a member's labels lie outside the [0, 1] domain the
                vote is defined on, so the ones array would come back as a
                confident acceptance of every row. See
                :func:`_refuse_a_silent_ensemble`,
                :func:`_refuse_a_vote_that_cannot_reach_the_threshold` and
                :func:`_refuse_a_vote_the_label_domain_decides`.
        """
        _refuse_a_silent_ensemble(self.classifiers, self.weights, caller="ReductionResult.predict")
        _refuse_a_vote_that_cannot_reach_the_threshold(
            self.weights, caller="ReductionResult.predict"
        )
        # The other side of that same total. Both entry points, one guard: see
        # _refuse_a_vote_the_weight_scale_decides.
        _refuse_a_vote_the_weight_scale_decides(self.weights, caller="ReductionResult.predict")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        predictions = np.zeros(n_samples)

        # The member labels are gathered BEFORE any of them is folded into the
        # sum, because the guard below is about the label DOMAIN and a label of
        # 2 has already decided the answer by the time the sum is taken. See
        # _refuse_a_vote_the_label_domain_decides for the measured before/after.
        contributions = [
            (clf, float(weight), clf.predict(X))
            for clf, weight in zip(self.classifiers, self.weights)
            if weight > 1e-8  # Skip negligible weights
        ]
        _refuse_a_vote_the_label_domain_decides(contributions, caller="ReductionResult.predict")

        for _clf, weight, pred in contributions:
            predictions += weight * pred

        labels = (predictions >= 0.5).astype(int)
        # THE COLLAPSE HAD TO BE DISCLOSED ON THE LABEL PATH TOO (F9 wave 4,
        # 2026-09-30). ExponentiatedGradient.predict_proba discloses a collapsed
        # fit and predict disclosed NOTHING, while predict is the method a
        # decision is taken from. The comment on that earlier fix says "THE
        # COLLAPSE WAS DISCLOSED AT FIT TIME AND NOWHERE ELSE. A caller holding
        # the fitted estimator ... read a single constant pseudo-probability for
        # every row with no warning at this call at all", and that sentence was
        # still true word for word of the LABEL path. Measured on a 240-row
        # collapsed fit with DummyClassifier(strategy='most_frequent') as the base
        # estimator, the fit warnings suppressed to stand for a caller who was not
        # there for them:
        #
        #   fairness_metrics: constraint_satisfied True, insufficient_data False,
        #                     degenerate_constant_predictions True,
        #                     n_prediction_classes 1;  final_violation 0.0
        #   EG.predict_proba -> np.unique == [1.]  2 warnings
        #   EG.predict       -> np.unique == [1]   0 WARNINGS  -> 1 after
        #   RR.predict_proba -> np.unique == [1.]  1 warning
        #   RR.predict       -> np.unique == [1]   0 WARNINGS  -> 1 after
        #
        # A model that accepted all 240 rows, reporting a satisfied constraint and
        # a violation of 0.0, in silence. Approving everybody is trivially equal,
        # so a mitigation that has been switched off shows a BETTER parity figure
        # than one that works, never a worse one, which is why this disclosure
        # cannot be keyed on the fairness number beside it.
        #
        # IT LIVES HERE, not on ExponentiatedGradient.predict, because that method
        # delegates to this one: one line covers BOTH named entry points instead of
        # a flag branch duplicated into each, and it reaches a result restored from
        # a stored payload too.
        #
        # THREE STATES on the flag. It is read with .get() and then tested for
        # None SEPARATELY, because `.get(key, False)` would answer "the fit did not
        # collapse" both for a fit that did not and for a result that never
        # recorded whether it had, which is the same fabricated verdict one level
        # down (the mistake predict_proba's subscript documents avoiding). Present
        # and truthy warns; present and falsy is silent; ABSENT or None is the
        # could-not-check and is silent here by design, because a hand-built
        # ReductionResult has no fit to have collapsed and warning on every one
        # would be the over-correction.
        collapsed = self.fairness_metrics.get("degenerate_constant_predictions")
        if collapsed is not None and bool(collapsed):
            warnings.warn(
                f"ReductionResult.predict: the fitted model predicts a single class for "
                f"every row (fairness_metrics['n_prediction_classes'] = "
                f"{self.fairness_metrics.get('n_prediction_classes')!r}), so these labels "
                "carry the collapsed best response rather than a learned decision, and "
                "the constraint violation reported beside them is trivially satisfied "
                "rather than achieved. Until 2026-09-30 only predict_proba said so, while "
                "this is the method a decision is taken from. See "
                "fairness_metrics['degenerate_constant_predictions'].",
                UserWarning,
                stacklevel=2,
            )
        return labels

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.

        Args:
            X: Feature matrix

        Returns:
            Ensemble probabilities. When a contributing member has no
            predict_proba its hard 0/1 labels are folded in, which makes the
            result a weighted VOTE FRACTION rather than a probability; that
            case warns. A returned column that sits entirely on {0.0, 1.0} warns
            as well, whatever member produced it: see
            :func:`_warn_about_a_score_column_of_hard_values`.

        Raises:
            ValueError: if no member of the ensemble would contribute, so the
                zeros array would come back as a probability of exactly 0.0.
        """
        _refuse_a_silent_ensemble(
            self.classifiers, self.weights, caller="ReductionResult.predict_proba"
        )
        _warn_about_vote_fractions(
            self.classifiers, self.weights, caller="ReductionResult.predict_proba"
        )
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        probas = np.zeros(n_samples)

        for clf, weight in zip(self.classifiers, self.weights):
            if weight > 1e-8 and hasattr(clf, "predict_proba"):
                pred_proba = clf.predict_proba(X)
                if pred_proba.ndim > 1:
                    pred_proba = pred_proba[:, 1]
                probas += weight * pred_proba
            elif weight > 1e-8:
                # Fall back to predictions
                probas += weight * clf.predict(X)

        # The disclosure that reads the VALUES rather than the member's class.
        # Measured before it: this method returned unique [0.] for a
        # DummyClassifier(strategy='most_frequent') member with zero warnings,
        # while the byte-identical column from a _ConstantClassifier warned.
        # See _warn_about_a_score_column_of_hard_values.
        _warn_about_a_score_column_of_hard_values(probas, caller="ReductionResult.predict_proba")
        return probas

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without classifiers)."""
        return {
            "n_classifiers": len(self.classifiers),
            "weights": self.weights.tolist(),
            "optimization_result": self.optimization_result.to_dict(),
            "final_violation": self.final_violation,
            "accuracy": self.accuracy,
            "fairness_metrics": self.fairness_metrics,
        }


class ExponentiatedGradient:
    """
    Exponentiated Gradient Algorithm for Fair Classification.

    Implements the Exponentiated Gradient (EG) algorithm from Agarwal et al. (2018)
    for training classifiers that satisfy fairness constraints. The algorithm
    solves a saddle-point optimization problem:

        min_θ max_λ>=0 L(θ) + Σᵢ λᵢ * gᵢ(θ)

    using multiplicative weight updates for the Lagrange multipliers and
    best-response updates for the classifier.

    For the rate constraints (demographic parity, equalized odds, equal
    opportunity, FPR parity) each best response is the cost-sensitive
    reduction of Agarwal et al. (2018), Sec. 3: every constraint coordinate
    carries a (+) and a (-) multiplier, their signed net value enters the
    per-example cost of predicting positive, and the base learner is trained
    on relabeled/reweighted samples (labels flipped where the net cost of
    predicting 1 is negative). Bounded group loss keeps a per-group
    emphasis-weighting best response on the true labels, which is the exact
    best response for that constraint. With ``best_gap_iteration=True`` the
    returned model is the most accurate iterate among those satisfying the
    constraint (falling back to the smallest-violation iterate when none
    does).

    Args:
        base_estimator: Base classifier (must support fit/predict and sample_weight)
        constraint: Fairness constraint to enforce
        max_iterations: Maximum number of iterations
        eta: Base learning rate for the multiplier updates (decays as
            eta / sqrt(t + 1) over iterations t)
        nu: Initial value of each Lagrange multiplier
        epsilon: Convergence threshold for duality gap
        best_gap_iteration: If True, predict with the single best iterate
            (most accurate satisfying iterate, else smallest violation);
            if False, uniformly average all iterates
        verbose: Whether to print progress

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from vfairness.in_processing.constraints import (
        ...     ExponentiatedGradient,
        ...     DemographicParityConstraint,
        ... )
        >>>
        >>> # Create constraint and algorithm
        >>> constraint = DemographicParityConstraint(tolerance=0.05)
        >>> eg = ExponentiatedGradient(
        ...     base_estimator=LogisticRegression(),
        ...     constraint=constraint,
        ...     max_iterations=50
        ... )
        >>>
        >>> # Fit
        >>> result = eg.fit(X_train, y_train, sensitive_attr=gender)
        >>>
        >>> # Predict
        >>> y_pred = result.predict(X_test)

    Comparison with Fairlearn:
        This implementation follows the same algorithmic approach as Fairlearn's
        ExponentiatedGradient but is designed to integrate with vfairness's
        constraint system and provides additional tracking/reporting.

    References:
        - Agarwal et al. (2018): A Reductions Approach to Fair Classification

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

    Ledger row: exponentiated_gradient. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        base_estimator: Any,
        constraint: Union[BaseFairnessConstraint, str, FairnessConstraintType],
        *,
        max_iterations: int = 50,
        eta: float = 2.0,
        nu: Optional[float] = None,
        epsilon: float = 0.01,
        best_gap_iteration: bool = True,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self._constraint = self._setup_constraint(constraint)
        self.max_iterations = max_iterations
        self.eta = eta
        self.nu = nu
        self.epsilon = epsilon
        self.best_gap_iteration = best_gap_iteration
        self.verbose = verbose

        # State
        self._fitted = False
        self._result: Optional[ReductionResult] = None

    def _setup_constraint(
        self,
        constraint: Union[BaseFairnessConstraint, str, FairnessConstraintType],
    ) -> BaseFairnessConstraint:
        """Setup the constraint object."""
        if isinstance(constraint, BaseFairnessConstraint):
            return constraint
        return create_constraint(constraint)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> ReductionResult:
        """
        Fit the fair classifier using exponentiated gradient.

        Args:
            X: Feature matrix
            y: Target labels
            sensitive_attr: Sensitive attribute for fairness constraint
            sample_weight: Optional initial sample weights

        Returns:
            ReductionResult with fitted ensemble. Its ``fairness_metrics`` carry
            three flags that must be read beside ``constraint_satisfied``:
            ``insufficient_data`` (the constraint could not be evaluated),
            ``n_iterations_unmeasurable`` (how many iterates could not be
            compared) and ``degenerate_constant_predictions`` (the returned model
            predicts one class for every row, so a zero violation is trivially
            satisfied rather than achieved). Each of the three also warns.
        """
        X = coerce_to_array(X, "X")
        y = coerce_to_array(y, "y")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")
        check_consistent_length(X, y, sensitive_attr)

        n_samples = len(y)
        gm = GroupManager(sensitive_attr)

        sample_weight_arr: np.ndarray
        if sample_weight is None:
            sample_weight_arr = np.ones(n_samples) / n_samples
        else:
            sample_weight_arr = coerce_to_array(sample_weight, "sample_weight")
            sample_weight_arr = sample_weight_arr / sample_weight_arr.sum()

        # Compute the effective nu locally per fit: caching it on self would
        # reuse a stale value when the same instance is refit on data with a
        # different number of groups.
        nu = self.nu if self.nu is not None else 1.0 / (2 * len(gm.groups))

        # BoundedGroupLoss keeps the uniform per-group emphasis reduction:
        # for L = err + sum_g lambda_g * (loss_g - bound) the best response IS
        # weighted ERM on the TRUE labels with per-group weights, so the
        # legacy cost path is the correct one for it.
        legacy_bgl = isinstance(self._constraint, BoundedGroupLossConstraint)

        # Legacy multipliers (BGL path only): one per group.
        lambda_ = {group: nu for group in gm.groups}

        # Agarwal et al. (2018) Alg. 1 state (rate-constraint path): every
        # coordinate carries a (+) and a (-) multiplier for the two one-sided
        # constraints g_j <= tol and -g_j <= tol; lambda lives on the scaled
        # simplex with a null coordinate, bounded by B (fairlearn uses the
        # same B = 1/eps convention).
        tol = float(getattr(self._constraint, "tolerance", 0.05))
        bound = 1.0 / tol if tol > 0 else 100.0
        coords = _reduction_coordinates(self._constraint, gm)
        signs = ("+", "-")
        theta_keys = [(coord, sign) for coord in coords for sign in signs]
        # Initialize so every multiplier starts at nu (nu keeps its documented
        # meaning of "initial constraint weight"); solve
        # nu = B*e^theta0 / (1 + m*e^theta0) for theta0.
        m = len(theta_keys)
        lam0 = min(nu, 0.5 * bound / max(m, 1))
        theta0 = float(np.log(lam0 / (bound - m * lam0))) if lam0 > 0 else -50.0
        theta = {key: theta0 for key in theta_keys}

        def _phi_from_theta() -> Dict[_Coord, float]:
            exps = {key: np.exp(np.clip(val, -50.0, 50.0)) for key, val in theta.items()}
            denom = 1.0 + sum(exps.values())
            return {
                coord: bound * (exps[(coord, "+")] - exps[(coord, "-")]) / denom for coord in coords
            }

        # Storage for classifiers and optimization history
        classifiers = []
        classifier_weights = []
        lambda_history = []
        loss_history = []
        violation_history = []
        satisfied_history = []
        accuracy_history = []
        best_gap = float("inf")
        best_iteration = 0

        # Main optimization loop
        for t in range(self.max_iterations):
            if legacy_bgl:
                # Compute cost-sensitive weights based on Lagrangian
                costs = self._compute_costs(lambda_, gm, n_samples)

                # Train classifier with cost-sensitive weights
                clf = deepcopy(self.base_estimator)
                total_weights = sample_weight_arr * costs
                total_weights = total_weights / total_weights.sum()

                try:
                    clf.fit(X, y, sample_weight=total_weights)
                except TypeError:
                    # Estimator doesn't support sample_weight
                    clf.fit(X, y)
            else:
                # Cost-sensitive best response with SIGNED lambda costs: the
                # net multiplier phi_j = lambda_j^+ - lambda_j^- enters the
                # cost of predicting positive, and labels are flipped where
                # that net cost is negative (Agarwal et al. 2018, Sec. 3).
                # The old code applied one label- and sign-blind uniform
                # weight per group, which carried no constraint coupling at
                # all; the constrained disparity never improved.
                phi = _phi_from_theta()
                z, fit_weights = _cost_sensitive_relabel(y, gm, coords, phi, sample_weight_arr)
                clf = _fit_best_response(self.base_estimator, X, z, fit_weights)

            classifiers.append(clf)

            y_pred = clf.predict(X)

            violation = self._constraint.compute_violation(y_pred, y, sensitive_attr)

            if legacy_bgl:
                # Update Lagrange multipliers using exponentiated gradient
                for group in gm.groups:
                    signed_violation = self._constraint.signed_constraint_value(
                        y_pred, y, sensitive_attr, group
                    )
                    # Multiplicative update: λ ← λ * exp(η * g)
                    lambda_[group] = lambda_[group] * np.exp(self.eta * signed_violation)
                    # Clip to prevent numerical issues
                    lambda_[group] = np.clip(lambda_[group], 1e-10, 1e10)

                total_lambda = sum(lambda_.values())
                if total_lambda > 0:
                    lambda_ = {g: v / total_lambda for g, v in lambda_.items()}

                lambda_history.append(lambda_.copy())
            else:
                # theta update of Alg. 1: theta_j += eta_t * gamma_j(h_t) with
                # gamma the one-sided constraint values; the step decays so
                # the iterates settle instead of oscillating between the
                # saturated best responses.
                gaps = _signed_gaps(self._constraint, y_pred, y, sensitive_attr, coords)
                eta_t = self.eta / np.sqrt(t + 1.0)
                for coord in coords:
                    gap_c = gaps[coord]
                    theta[(coord, "+")] = float(
                        np.clip(theta[(coord, "+")] + eta_t * (gap_c - tol), -50.0, 50.0)
                    )
                    theta[(coord, "-")] = float(
                        np.clip(theta[(coord, "-")] + eta_t * (-gap_c - tol), -50.0, 50.0)
                    )
                lambda_history.append({_coord_key(c): v for c, v in _phi_from_theta().items()})

            loss = np.mean(y_pred != y)

            # Track history
            loss_history.append(loss)
            violation_history.append(violation.overall_violation)
            satisfied_history.append(bool(violation.is_satisfied))
            accuracy_history.append(1.0 - loss)
            classifier_weights.append(1.0)

            # Compute duality gap (approximate)
            gap = abs(violation.overall_violation)
            if gap < best_gap:
                best_gap = gap

            if self.verbose:
                print(
                    f"Iteration {t + 1}/{self.max_iterations}: "
                    f"Loss={loss:.4f}, Violation={violation.overall_violation:.4f}, "
                    f"Gap={gap:.4f}"
                )

            if violation.is_satisfied and gap < self.epsilon:
                if self.verbose:
                    print(f"Converged at iteration {t + 1}")
                break

        # Select the best iterate: the most ACCURATE one among those that
        # satisfy the constraint, else the one with the smallest violation.
        # (Pure min-violation selection would happily pick a degenerate
        # collapsed classifier, e.g. all-positive predictions have a
        # demographic-parity violation of exactly zero.)
        satisfied_indices = [i for i, s in enumerate(satisfied_history) if s]
        n_unmeasurable_iterations = 0
        if satisfied_indices:
            best_iteration = max(satisfied_indices, key=lambda i: accuracy_history[i])
        elif violation_history:
            picked, n_unmeasurable_iterations = _pick_smallest_violation(violation_history)
            if picked >= 0:
                best_iteration = picked
            else:
                # Every iterate was unmeasurable. best_iteration keeps whatever
                # the loop left it at; what matters is that nothing here claims
                # to have compared them.
                warnings.warn(
                    f"ExponentiatedGradient: not one of {len(violation_history)} iterates "
                    f"had a measurable constraint violation, so no iterate could be "
                    f"chosen as the smallest. The returned model was NOT selected for "
                    f"fairness. See fairness_metrics['insufficient_data'].",
                    UserWarning,
                    stacklevel=2,
                )

        # Compute final weights using best gap or averaging
        if self.best_gap_iteration:
            # Put all weight on best iteration
            final_weights = np.zeros(len(classifiers))
            final_weights[best_iteration] = 1.0
        else:
            # Uniform averaging
            final_weights = np.ones(len(classifiers)) / len(classifiers)

        result_clf = self._create_ensemble(classifiers, final_weights)
        y_pred_final = result_clf.predict(X)
        final_violation = self._constraint.compute_violation(y_pred_final, y, sensitive_attr)
        final_accuracy = np.mean(y_pred_final == y)

        # CHECK THE INTERVENTION'S OWN OUTPUT FOR DEGENERACY, NOT ITS FAIRNESS
        # NUMBER. A model that predicts ONE CLASS for every row has a
        # selection-rate spread of exactly 0.0 and a TPR/FPR spread of exactly
        # 0.0: it is trivially perfectly fair because it made no decision at all,
        # and the fairness metric therefore IMPROVES as the mitigation is
        # neutered. The best-iterate selection above already knows this ("pure
        # min-violation selection would happily pick a degenerate collapsed
        # classifier") and defends against it by preferring the most ACCURATE
        # satisfying iterate. That defence is empty when EVERY iterate is
        # collapsed: they all satisfy, and the most accurate of them is still
        # constant. Measured 2026-09-27 on 240 rows whose group base rates were
        # 0.90 and 0.09, with a base estimator that always returns the majority
        # class:
        #
        #   final_violation 0.0, accuracy 0.504, constraint_satisfied True,
        #   insufficient_data False, converged True, np.unique(predict(X)) == [0],
        #   and not one warning.
        #
        # A compliance certificate for a classifier that rejected all 240 rows.
        # The 0.0 is arithmetically real, so it is reported as measured; what was
        # missing is that it describes a model nobody can use.
        prediction_classes = np.unique(y_pred_final)
        degenerate = bool(len(prediction_classes) < 2)
        if degenerate:
            warnings.warn(
                f"ExponentiatedGradient: the returned model predicts the single class "
                f"{prediction_classes[0]!r} for all {len(y_pred_final)} training row(s). Its "
                f"{final_violation.constraint_type} violation of "
                f"{final_violation.overall_violation!r} is TRIVIALLY satisfied: a constant "
                "classifier has the same rate in every group because it made no decision, "
                "so this number is not evidence that the mitigation reduced disparity. "
                "See fairness_metrics['degenerate_constant_predictions'].",
                UserWarning,
                stacklevel=2,
            )

        opt_result = OptimizationResult(
            converged=final_violation.is_satisfied,
            n_iterations=len(classifiers),
            final_violation=final_violation.overall_violation,
            best_gap=best_gap,
            lambda_history=lambda_history,
            loss_history=loss_history,
            violation_history=violation_history,
            final_weights=final_weights,
            metadata={
                "best_iteration": best_iteration,
                "eta": self.eta,
                "nu": nu,
            },
        )

        self._result = ReductionResult(
            classifiers=classifiers,
            weights=final_weights,
            optimization_result=opt_result,
            final_violation=final_violation.overall_violation,
            accuracy=final_accuracy,
            fairness_metrics={
                # THREE STATES. `constraint_satisfied` is False both for a
                # measured breach and for a constraint that could not be
                # evaluated, and base.py's docstring already warned that "a
                # caller that branches on is_satisfied alone will report a
                # violation nobody measured". It could not do better, because
                # this dict did not carry the flag it told callers to pair with.
                # Now it does.
                "constraint_satisfied": final_violation.is_satisfied,
                "insufficient_data": bool(final_violation.details.get("insufficient_data", False)),
                "n_iterations_unmeasurable": n_unmeasurable_iterations,
                "constraint_type": final_violation.constraint_type,
                "tolerance": final_violation.tolerance,
                # The intervention's own degeneracy, beside the verdict it
                # produces. True means constraint_satisfied is vacuous: read
                # them together, the same way constraint_satisfied must be read
                # with insufficient_data.
                "degenerate_constant_predictions": degenerate,
                "n_prediction_classes": int(len(prediction_classes)),
            },
        )

        self._fitted = True
        return self._result

    def _compute_costs(
        self,
        lambda_: Dict[str, float],
        gm: GroupManager,
        n_samples: int,
    ) -> np.ndarray:
        """
        Compute per-group emphasis weights from Lagrange multipliers.

        Used ONLY for BoundedGroupLossConstraint, where the Lagrangian
        L = err + sum_g lambda_g * (loss_g - bound) has exactly this
        weighted-ERM-on-true-labels best response. For the rate constraints
        this uniform per-group weighting is label- and sign-blind and cannot
        enforce anything; they use _cost_sensitive_relabel instead
        (Agarwal et al. 2018).
        """
        costs = np.ones(n_samples)

        for group in gm.groups:
            mask = gm.get_mask(group)
            group_size = mask.sum()

            if group_size > 0:
                # Cost adjustment based on Lagrange multiplier
                # Higher lambda means this group's constraint is being violated
                cost_multiplier = 1 + lambda_[group] * n_samples / group_size
                costs[mask] = cost_multiplier

        return costs

    def _create_ensemble(
        self,
        classifiers: List[Any],
        weights: np.ndarray,
    ) -> "EnsembleClassifier":
        """Create an ensemble classifier."""
        return EnsembleClassifier(classifiers, weights)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Make predictions with the fitted ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise RuntimeError("ExponentiatedGradient is not fitted. Call fit() first.")
        assert self._result is not None  # _fitted is only True after fit() sets _result
        return self._result.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict probabilities with the fitted ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities. A fit that collapsed to one class warns
            HERE as well as at fit time: see
            ``fairness_metrics['degenerate_constant_predictions']``.
        """
        if not self._fitted:
            raise RuntimeError("ExponentiatedGradient is not fitted. Call fit() first.")
        assert self._result is not None  # _fitted is only True after fit() sets _result
        # BGL4 (2026-09-27). THE COLLAPSE WAS DISCLOSED AT FIT TIME AND NOWHERE
        # ELSE. A caller holding the fitted estimator (or one restored from a
        # stored result) read a single constant pseudo-probability for every row
        # with no warning at this call at all. Measured on 120 rows with
        # DummyClassifier(strategy='most_frequent') as the base estimator, the
        # fit warning suppressed to stand for a caller who was not there for it::
        #
        #   before: np.unique(eg.predict_proba(X)) == [0.], 0 warnings
        #   after:  np.unique(eg.predict_proba(X)) == [0.], 2 warnings, one
        #           naming the collapsed fit and one naming the hard 0/1 column
        #
        # This branch reads the FLAG rather than the values, because the values
        # cannot carry the fact: DummyClassifier(strategy='prior') returns a
        # constant 0.2, which is a perfectly ordinary looking probability, and
        # the fit that produced it was just as collapsed. Measured, same 120
        # rows: before 0 warnings, after 1 (the flag one; the value one cannot
        # see 0.2 and must not).
        #
        # The flag is SUBSCRIPTED, not `.get(key, False)`: fit() always writes
        # it, and a `False` default would answer "the fit did not collapse" for
        # a result that never recorded whether it had, which is the same
        # fabricated verdict one level down.
        metrics = self._result.fairness_metrics
        if bool(metrics["degenerate_constant_predictions"]):
            warnings.warn(
                f"ExponentiatedGradient.predict_proba: the fitted model predicts a single "
                f"class for every row "
                f"(fairness_metrics['n_prediction_classes'] = "
                f"{metrics['n_prediction_classes']!r}), so these "
                "values are not a score that separates anybody: they carry the collapsed "
                "best response, and the constraint violation reported beside them is "
                "trivially satisfied rather than achieved. See "
                "fairness_metrics['degenerate_constant_predictions'].",
                UserWarning,
                stacklevel=2,
            )
        return self._result.predict_proba(X)

    def get_result(self) -> ReductionResult:
        """Get the full optimization result."""
        if not self._fitted:
            raise RuntimeError("ExponentiatedGradient is not fitted. Call fit() first.")
        assert self._result is not None  # _fitted is only True after fit() sets _result
        return self._result


class EnsembleClassifier:
    """Simple ensemble classifier that combines multiple classifiers."""

    def __init__(self, classifiers: List[Any], weights: np.ndarray):
        self.classifiers = classifiers
        self.weights = weights

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels.

        Raises:
            ValueError: if no member of the ensemble would contribute, or if the
                contributing weights cannot reach the 0.5 decision threshold, or if
                a member's labels lie outside the [0, 1] domain the vote is
                defined on (labels in {1, 2} made every row a 1 whatever the
                features said). See :func:`_refuse_a_silent_ensemble`,
                :func:`_refuse_a_vote_that_cannot_reach_the_threshold` and
                :func:`_refuse_a_vote_the_label_domain_decides`.
        """
        _refuse_a_silent_ensemble(
            self.classifiers, self.weights, caller="EnsembleClassifier.predict"
        )
        _refuse_a_vote_that_cannot_reach_the_threshold(
            self.weights, caller="EnsembleClassifier.predict"
        )
        # The other side of that same total, on the sibling entry point. One
        # guard, two callers: see _refuse_a_vote_the_weight_scale_decides.
        _refuse_a_vote_the_weight_scale_decides(self.weights, caller="EnsembleClassifier.predict")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        predictions = np.zeros(n_samples)

        # Gathered before anything is summed, for the reason
        # _refuse_a_vote_the_label_domain_decides records: with labels in {1, 2}
        # this method returned [1] for every row of every possible X, silently.
        contributions = [
            (clf, float(weight), clf.predict(X))
            for clf, weight in zip(self.classifiers, self.weights)
            if weight > 1e-8
        ]
        _refuse_a_vote_the_label_domain_decides(contributions, caller="EnsembleClassifier.predict")

        for _clf, weight, pred in contributions:
            predictions += weight * pred

        # DELIBERATELY NO constant-label disclosure here, and the reasoning is
        # worth keeping because it was tested and rejected (F9 wave 4,
        # 2026-09-30). A constant DECISION column is not itself a fabricated
        # measurement: it is the caller's own model's real output, and this class
        # publishes no fairness verdict beside it to be fabricated. The
        # fabrication the audit found was a constraint reported SATISFIED with a
        # violation of 0.0 for a model that separated nobody, and the channel
        # carrying that fact is fairness_metrics['degenerate_constant_predictions'],
        # which only a fitted ReductionResult has; the disclosure therefore lives
        # on ReductionResult.predict, which is also what
        # ExponentiatedGradient.predict delegates to. Keying it on the returned
        # labels instead fired on eight documented over-correction controls whose
        # members are constant stubs, i.e. on models that really are constant, and
        # a guard that flags every one-sided test set is the over-correction this
        # campaign warns about. The predict_proba sibling IS keyed on values, for
        # the different reason that the column it returns is itself presented as a
        # probability measurement.
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict probabilities.

        A contributing member without predict_proba contributes its hard 0/1
        labels, which makes the result a weighted vote fraction and warns. A
        returned column that sits entirely on {0.0, 1.0} warns as well, whatever
        member produced it: see
        :func:`_warn_about_a_score_column_of_hard_values`.

        Raises:
            ValueError: if no member of the ensemble would contribute.
        """
        _refuse_a_silent_ensemble(
            self.classifiers, self.weights, caller="EnsembleClassifier.predict_proba"
        )
        _warn_about_vote_fractions(
            self.classifiers, self.weights, caller="EnsembleClassifier.predict_proba"
        )
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        probas = np.zeros(n_samples)

        for clf, weight in zip(self.classifiers, self.weights):
            if weight > 1e-8:
                if hasattr(clf, "predict_proba"):
                    pred_proba = clf.predict_proba(X)
                    if pred_proba.ndim > 1:
                        pred_proba = pred_proba[:, 1]
                    probas += weight * pred_proba
                else:
                    probas += weight * clf.predict(X)

        # Same value-keyed disclosure as ReductionResult.predict_proba, on the
        # surface a caller who built the ensemble by hand arrives at. Measured
        # before it: unique [0.] from a DummyClassifier(strategy='most_frequent')
        # member, zero warnings. See _warn_about_a_score_column_of_hard_values.
        _warn_about_a_score_column_of_hard_values(probas, caller="EnsembleClassifier.predict_proba")
        return probas


class GridSearch:
    """
    Grid Search over Lagrange Multipliers.

    Performs a grid search over SIGNED per-group Lagrange multipliers
    to find the best trade-off between accuracy and fairness, following
    the grid-search variant of Agarwal et al. (2018): each grid value lam
    is turned into a per-coordinate multiplier phi_g = sign_g * lam, where
    sign_g is the direction of group g's disparity measured on an
    unconstrained baseline fit, and phi enters the cost of predicting
    positive via the same cost-sensitive relabel/reweight reduction the
    ExponentiatedGradient best response uses. (The previous implementation
    applied one shared UNSIGNED lambda uniformly to every group, which
    yields provably identical classifiers for balanced groups; the sweep
    explored nothing.)

    Args:
        base_estimator: Base classifier
        constraint: Fairness constraint to enforce
        lambda_range: Range of multiplier magnitudes to sweep. Default
            changed from (0.0, 10.0) to (0.0, 1.0): under the corrected
            signed-cost semantics a magnitude near 1 already fully relabels
            a balanced group's samples, so the informative part of the sweep
            lives in [0, 1] and the old range wasted almost every grid point
            on degenerate fully-flipped classifiers.
        n_lambda_values: Number of lambda values to try
        verbose: Whether to print progress

    Example:
        >>> gs = GridSearch(
        ...     base_estimator=LogisticRegression(),
        ...     constraint=DemographicParityConstraint(tolerance=0.05)
        ... )
        >>> result = gs.fit(X_train, y_train, sensitive_attr=gender)

    References:
        - Agarwal et al. (2018): A Reductions Approach

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: constraint_grid_search. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        base_estimator: Any,
        constraint: Union[BaseFairnessConstraint, str, FairnessConstraintType],
        *,
        lambda_range: Tuple[float, float] = (0.0, 1.0),
        n_lambda_values: int = 20,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self._constraint = (
            constraint
            if isinstance(constraint, BaseFairnessConstraint)
            else create_constraint(constraint)
        )
        self.lambda_range = lambda_range
        self.n_lambda_values = n_lambda_values
        self.verbose = verbose

        self._fitted = False
        self._result: Optional[ReductionResult] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_attr: ArrayLike,
    ) -> ReductionResult:
        """
        Fit using grid search over Lagrange multipliers.

        Args:
            X: Feature matrix
            y: Target labels
            sensitive_attr: Sensitive attribute

        Returns:
            ReductionResult with best classifier
        """
        X = coerce_to_array(X, "X")
        y = coerce_to_array(y, "y")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        lambda_values = np.linspace(
            self.lambda_range[0], self.lambda_range[1], self.n_lambda_values
        )

        gm = GroupManager(sensitive_attr)
        n_samples = len(y)
        base_weight = np.ones(n_samples) / n_samples

        # GUARD ABOVE THE DISPATCH. Every constraint this class sweeps is a
        # COMPARISON between groups: a rate spread for the parity family, a
        # group loss against the overall loss for bounded group loss. With a
        # SINGLE level in sensitive_attr there is no pair to compare, so no
        # violation exists to be satisfied and nothing here is measurable.
        #
        # fit() used to delegate that judgement entirely to the constraint and
        # report whatever came back. DemographicParityConstraint returned
        # max(rates) - min(rates) over a ONE-element list, an arithmetically
        # vacuous 0.0, and this method dressed it as final_violation=0.0,
        # constraint_satisfied=True, insufficient_data=False, converged=True
        # and no warning: a compliance certificate byte-identical to a genuine
        # two-group measured perfect parity. The check belongs HERE and not
        # only inside each constraint, because fit() dispatches to whatever
        # BaseFairnessConstraint it was handed, including a caller's own
        # subclass, so a per-constraint fix just moves the fabrication to the
        # sibling nobody has fixed yet.
        #
        # A fail STATE, not a raise: the sweep still runs and still returns a
        # fitted model, but every candidate's fairness violation is reported
        # unmeasurable (NaN), which routes into the existing
        # insufficient_data / n_candidates_unmeasurable channel below.
        groups_incomparable = gm.n_groups < 2
        if groups_incomparable:
            warnings.warn(
                f"GridSearch: sensitive_attr holds {gm.n_groups} group(s) "
                f"({list(gm.groups)}); a fairness constraint compares groups, so "
                f"no {self._constraint.__class__.__name__} violation is "
                f"MEASURABLE on this input. The sweep still fits, but its "
                f"fairness violation is reported as NaN and the returned "
                f"classifier was NOT selected for fairness. See "
                f"fairness_metrics['insufficient_data'].",
                UserWarning,
                stacklevel=2,
            )

        # Baseline unconstrained fit: its per-coordinate SIGNED disparity
        # fixes the direction each group's positive-prediction cost moves in.
        # Sweeping one shared unsigned lambda (the old code) is structurally
        # a no-op: with balanced groups every sample got the identical
        # weight for every lambda. (Mean-1 weights, see
        # _cost_sensitive_relabel on why sum-1 weights would distort
        # regularized estimators.)
        baseline = _fit_best_response(self.base_estimator, X, np.asarray(y), np.ones(n_samples))
        y_pred_base = baseline.predict(X)

        legacy_bgl = isinstance(self._constraint, BoundedGroupLossConstraint)
        if legacy_bgl:
            # Bounded group loss is one-sided: only groups whose loss exceeds
            # the bound on the baseline get their emphasis increased.
            base_violation = self._constraint.compute_violation(y_pred_base, y, sensitive_attr)
            bgl_signs = {
                group: 1.0 if base_violation.group_violations.get(group, 0.0) > 0 else 0.0
                for group in gm.groups
            }
            coords: List[_Coord] = []
            coord_signs: Dict[_Coord, float] = {}
        else:
            coords = _reduction_coordinates(self._constraint, gm)
            base_gaps = _signed_gaps(self._constraint, y_pred_base, y, sensitive_attr, coords)
            coord_signs = {c: float(np.sign(base_gaps[c])) for c in coords}

        all_classifiers = []
        all_accuracies = []
        all_violations = []
        all_satisfied = []

        for lam in lambda_values:
            if legacy_bgl:
                weights = np.ones(n_samples)
                for group in gm.groups:
                    mask = gm.get_mask(group)
                    weights[mask] = 1 + bgl_signs[group] * lam * n_samples / mask.sum()
                weights = weights / weights.sum()

                clf = deepcopy(self.base_estimator)
                try:
                    clf.fit(X, y, sample_weight=weights)
                except TypeError:
                    clf.fit(X, y)
            else:
                phi = {c: coord_signs[c] * lam for c in coords}
                z, fit_weights = _cost_sensitive_relabel(y, gm, coords, phi, base_weight)
                clf = _fit_best_response(self.base_estimator, X, z, fit_weights)

            y_pred = clf.predict(X)
            accuracy = np.mean(y_pred == y)
            violation = self._constraint.compute_violation(y_pred, y, sensitive_attr)
            # See the n_groups guard above: with no pair of groups to compare,
            # whatever the constraint returns for this candidate is not a
            # measurement, so it is never recorded as one.
            candidate_violation = (
                float("nan") if groups_incomparable else violation.overall_violation
            )
            candidate_satisfied = False if groups_incomparable else violation.is_satisfied

            all_classifiers.append(clf)
            all_accuracies.append(accuracy)
            all_violations.append(candidate_violation)
            all_satisfied.append(candidate_satisfied)

            if self.verbose:
                print(
                    f"λ={lam:.2f}: Accuracy={accuracy:.4f}, "
                    f"Violation={candidate_violation:.4f}, "
                    f"Satisfied={candidate_satisfied}"
                )

        # Select a SINGLE best classifier and predict with exactly that one
        # (one-hot weights). The old code reported the best classifier's
        # metrics but predicted with an accuracy-weighted ensemble of all
        # classifiers, so the reported violation/accuracy did not describe
        # the model that predict() actually used.
        satisfied_indices = [i for i, s in enumerate(all_satisfied) if s]
        if satisfied_indices:
            best_idx = max(satisfied_indices, key=lambda i: all_accuracies[i])
            converged = True
        else:
            # Nothing satisfied the constraint: fall back to the classifier
            # with the smallest violation and report ITS real metrics
            # (previously accuracy=0.0 / violation=inf were reported while
            # predict() silently used a uniform ensemble).
            picked, n_unmeasurable = _pick_smallest_violation(all_violations)
            converged = False
            if picked >= 0:
                best_idx = picked
                warnings.warn(
                    "GridSearch: no lambda value satisfied the fairness "
                    "constraint; returning the classifier with the smallest "
                    f"violation ({all_violations[best_idx]:.4f})."
                    + (
                        f" {n_unmeasurable} candidate(s) had an unmeasurable violation "
                        f"and were not compared."
                        if n_unmeasurable
                        else ""
                    )
                )
            else:
                # Not one candidate had a measurable violation. The old code
                # said "returning the classifier with the smallest violation
                # (nan)", which asserts a comparison that never happened, and
                # np.argmin handed back the first NaN's index as if it had won.
                best_idx = 0
                warnings.warn(
                    f"GridSearch: not one of {len(all_violations)} candidates had a "
                    f"MEASURABLE fairness violation, so none could be smallest. "
                    f"Returning the first candidate; it was NOT selected for fairness. "
                    f"See fairness_metrics['insufficient_data'].",
                    UserWarning,
                    stacklevel=2,
                )

        best_accuracy = all_accuracies[best_idx]
        best_violation = all_violations[best_idx]
        best_lambda = float(lambda_values[best_idx])

        one_hot_weights = np.zeros(len(all_classifiers))
        one_hot_weights[best_idx] = 1.0

        opt_result = OptimizationResult(
            converged=converged,
            n_iterations=self.n_lambda_values,
            final_violation=best_violation,
            best_gap=best_violation,
            lambda_history=[{"lambda": lam} for lam in lambda_values],
            metadata={"best_lambda": best_lambda, "best_index": best_idx},
        )

        best_measured = best_violation == best_violation
        self._result = ReductionResult(
            classifiers=all_classifiers,
            weights=one_hot_weights,
            optimization_result=opt_result,
            final_violation=best_violation,
            accuracy=best_accuracy,
            # Was left empty, so a caller had NOTHING to read here: not the
            # verdict, not the constraint, and no way to tell an unevaluable
            # constraint from a satisfied one.
            fairness_metrics={
                "constraint_satisfied": bool(all_satisfied[best_idx]),
                "insufficient_data": not best_measured,
                "n_candidates_unmeasurable": sum(1 for v in all_violations if v != v),
                "constraint_type": self._constraint.__class__.__name__,
            },
        )

        self._fitted = True
        return self._result

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise RuntimeError("GridSearch is not fitted. Call fit() first.")
        assert self._result is not None  # _fitted is only True after fit() sets _result
        return self._result.predict(X)


def _count_unscored_rows(y_prob: np.ndarray) -> int:
    """How many rows carry no usable score.

    ``y_prob >= threshold`` is False for NaN under IEEE 754, so an unscored
    row is written 0, a confident REJECT, and ``+inf`` lands above every
    threshold and is written 1, a confident ACCEPT. Neither is a decision
    anybody measured.
    """
    return int((~np.isfinite(np.asarray(y_prob, dtype=float))).sum())


def _refuse_unscored_rows(y_prob: np.ndarray, *, caller: str) -> None:
    """Refuse to decide about a row that has no score.

    G007 (2026-09-17). The identical defect was found and fixed in
    ``post_processing.threshold_optimization.optimizer`` under BGL-G05 and
    BGL-S2b; this copy of the threshold rule was never swept. Measured here
    before the fix, on a fitted optimizer (thresholds a=0.5544, b=0.4456)::

        predict([0.9, nan, 0.1, inf, -inf], ['a','a','a','b','b']) -> [1 0 0 1 0]

    The NaN row came back 0, byte-identical to the measured rejection at 0.1
    beside it; ``-inf`` fabricated another rejection and ``+inf`` fabricated an
    ACCEPTANCE. Zero warnings, no error, and the returned int array has no
    field that could have said so, which is why this refuses rather than
    marking.

    Guarded at ``predict`` and not inside ``_apply_thresholds``, which ``fit``
    also calls: a training column may legitimately hold NaN, and
    :func:`_count_unscored_rows` is what discloses it there.
    """
    unscored = ~np.isfinite(np.asarray(y_prob, dtype=float))
    n = int(unscored.sum())
    if n:
        idx = np.flatnonzero(unscored)[:10].tolist()
        raise ValueError(
            f"{caller}.predict: {n} of {unscored.size} row(s) have no usable score "
            f"(NaN or infinite y_prob), first at index {idx}. Refusing rather than "
            "deciding them: a non-finite score compares False against every threshold, "
            "so these rows would be REJECTED outright (or ACCEPTED outright for +inf), "
            "which is a verdict nobody measured, and the returned int array has no field "
            "to say so. Impute or drop the unscored rows before calling predict()."
        )


class ThresholdOptimizer:
    """
    Post-Processing Threshold Optimization for Fairness.

    Finds optimal decision thresholds per group to satisfy fairness
    constraints while maximizing overall accuracy.

    This is a post-processing approach that takes a pre-trained classifier
    and finds group-specific thresholds.

    Args:
        constraint: Fairness constraint to enforce
        grid_size: Number of threshold values to try per group
        objective: What to optimize ('accuracy', 'balanced_accuracy')

    Attributes set by fit(), THREE STATES and never two:
        final_violation_: the constraint violation of the final JOINT
            threshold assignment, or NaN when the constraint could not be
            evaluated on this data.
        constraint_satisfied_: True for a measured pass, False for a measured
            breach, and None for could-not-check. Never False for a
            constraint nobody could evaluate.
        insufficient_data_: True when no verdict could honestly be given,
            either because the constraint was unevaluable (fewer than two
            groups, an undefined group rate) or because it was computed over
            rows with no score, which enter it as fabricated rejections. When
            this is True the thresholds returned are the untouched 0.5 seed
            and were NOT selected for fairness.
        n_unscored_rows_: how many fitted rows carried a NaN or infinite
            y_prob. predict() refuses such rows outright; fit() folds them in
            and discloses, because a training column may legitimately hold
            NaN.
        degenerate_constant_predictions_: True when the final threshold
            assignment gives the SAME decision to every fitted row. The
            violation beside it is then VACUOUS rather than good: a rule that
            accepts everybody, or rejects everybody, satisfies every
            rate-based constraint by construction. Read it beside
            ``constraint_satisfied_``, the same way
            ``degenerate_constant_predictions`` must be read beside
            ``constraint_satisfied`` in
            ``FairnessTrainingAnalyzer.compare_methods``.

    Example:
        >>> clf = LogisticRegression().fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_val)[:, 1]
        >>>
        >>> to = ThresholdOptimizer(
        ...     constraint=EqualOpportunityConstraint(tolerance=0.05)
        ... )
        >>> to.fit(y_prob, y_val, sensitive_attr=gender)
        >>> y_pred = to.predict(y_prob_test, sensitive_attr_test)

    References:
        - Hardt et al. (2016): Equality of Opportunity
    """

    def __init__(
        self,
        constraint: Union[BaseFairnessConstraint, str, FairnessConstraintType],
        *,
        grid_size: int = 100,
        objective: Literal["accuracy", "balanced_accuracy"] = "accuracy",
    ):
        self._constraint = (
            constraint
            if isinstance(constraint, BaseFairnessConstraint)
            else create_constraint(constraint)
        )
        self.grid_size = grid_size
        self.objective = objective

        self._fitted = False
        self._thresholds: Dict[str, float] = {}
        # Validated metrics of the final JOINT threshold assignment,
        # populated by fit(). THREE STATES, never two:
        #   constraint_satisfied_ True  / final_violation_ finite  measured PASS
        #   constraint_satisfied_ False / final_violation_ finite  measured BREACH
        #   constraint_satisfied_ None  / insufficient_data_ True  NOT MEASURED
        # None before fit() as well; get_thresholds() raises until fitted, so
        # the two are never confusable from outside.
        self.final_violation_: Optional[float] = None
        self.constraint_satisfied_: Optional[bool] = None
        # True when the verdict above could not honestly be given: the
        # constraint was unevaluable on this data (fewer than two groups, an
        # undefined group rate), or it was computed over rows that had no
        # score and were folded in as fabricated rejections.
        self.insufficient_data_: Optional[bool] = None
        self.n_unscored_rows_: Optional[int] = None
        # G10, 2026-09-30. A MEASURED PASS OVER A RULE THAT SEPARATES NOBODY.
        # Distinct from insufficient_data_ on purpose: the violation IS measured
        # there, it is simply vacuous. Same pairing as
        # parameters['degenerate_constant_predictions'] beside
        # constraint_satisfied in FairnessTrainingAnalyzer.compare_methods.
        self.degenerate_constant_predictions_: Optional[bool] = None

    def fit(
        self,
        y_prob: ArrayLike,
        y_true: ArrayLike,
        *,
        sensitive_attr: ArrayLike,
    ) -> "ThresholdOptimizer":
        """
        Find optimal group-specific thresholds.

        Args:
            y_prob: Predicted probabilities
            y_true: True labels
            sensitive_attr: Sensitive attribute

        Returns:
            Fitted ThresholdOptimizer
        """
        y_prob = coerce_to_array(y_prob, "y_prob")
        y_true = coerce_to_array(y_true, "y_true")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        gm = GroupManager(sensitive_attr)
        threshold_grid = np.linspace(0.01, 0.99, self.grid_size)

        # Rows with no usable score enter every candidate's fairness
        # measurement as a fabricated 0 (or a fabricated 1 for +inf), so the
        # group rates the search optimises are computed over decisions nobody
        # measured. predict() refuses such rows outright; fit() measures and
        # discloses instead, because a training column may legitimately hold
        # NaN and refusing would break that path. Measured before this: an
        # all-NaN y_prob gave final_violation_=0.0, constraint_satisfied_=True
        # and ZERO warnings, a perfect-parity certificate over a column that
        # carried no information at all (every row rejected is trivially
        # equal). See feedback: a neutered mitigation REPORTS SUCCESS.
        n_unscored = _count_unscored_rows(y_prob)
        self.n_unscored_rows_ = n_unscored
        if n_unscored:
            warnings.warn(
                f"ThresholdOptimizer.fit: {n_unscored} of {len(y_prob)} row(s) have no "
                f"usable score (NaN or infinite y_prob). A non-finite score compares "
                f"False against every threshold, so these rows enter the fairness "
                f"measurement as REJECTED, which is a decision nobody measured. The "
                f"constraint verdict is reported as NOT ASSESSED "
                f"(constraint_satisfied_ is None, insufficient_data_ is True) because "
                f"it was computed over those fabricated rejections. Impute or drop the "
                f"unscored rows to get a measured verdict.",
                UserWarning,
                stacklevel=2,
            )

        def _objective_metric(y_pred: np.ndarray) -> float:
            if self.objective == "accuracy":
                return float(np.mean(y_pred == y_true))
            # Balanced accuracy; a class that is absent contributes 0 so an
            # all-one-class dataset cannot fake a perfect score.
            pos = y_true == 1
            neg = y_true == 0
            tpr = float(np.mean(y_pred[pos] == 1)) if pos.sum() > 0 else 0.0
            tnr = float(np.mean(y_pred[neg] == 0)) if neg.sum() > 0 else 0.0
            return (tpr + tnr) / 2

        # Coordinate descent over groups. Each candidate threshold is scored
        # on the JOINT assignment (this group's candidate combined with the
        # CURRENT best thresholds of all other groups). The old code fixed
        # the other groups at 0.5 while searching, so the per-group optima
        # were never valid together and the combined assignment was never
        # validated.
        thresholds = {g: 0.5 for g in gm.groups}
        max_sweeps = 5

        for _sweep in range(max_sweeps):
            changed = False
            for group in gm.groups:
                # Current joint assignment is the baseline to beat; it only
                # counts if it actually satisfies the constraint.
                y_pred_cur = self._apply_thresholds(y_prob, thresholds, gm)
                cur_violation = self._constraint.compute_violation(
                    y_pred_cur, y_true, sensitive_attr
                )
                best_threshold = thresholds[group]
                best_group_metric = (
                    _objective_metric(y_pred_cur) if cur_violation.is_satisfied else -float("inf")
                )
                best_group_violation = float(cur_violation.overall_violation)

                for threshold in threshold_grid:
                    test_thresholds = dict(thresholds)
                    test_thresholds[group] = threshold

                    y_pred = self._apply_thresholds(y_prob, test_thresholds, gm)
                    violation = self._constraint.compute_violation(y_pred, y_true, sensitive_attr)

                    if violation.is_satisfied:
                        metric = _objective_metric(y_pred)
                        if metric > best_group_metric + 1e-12:
                            best_group_metric = metric
                            best_threshold = threshold
                    elif best_group_metric == -float("inf"):
                        # No feasible assignment known yet: greedily reduce
                        # the joint violation so later sweeps (and other
                        # groups' moves) can reach feasibility. Without this
                        # the descent stalls whenever no single-coordinate
                        # move is immediately feasible.
                        if violation.overall_violation < best_group_violation - 1e-12:
                            best_group_violation = float(violation.overall_violation)
                            best_threshold = threshold

                if best_threshold != thresholds[group]:
                    thresholds[group] = best_threshold
                    changed = True

            if not changed:
                break

        # Validate the final JOINT assignment and store the real violation.
        y_pred_final = self._apply_thresholds(y_prob, thresholds, gm)
        final_violation = self._constraint.compute_violation(y_pred_final, y_true, sensitive_attr)
        self.final_violation_ = float(final_violation.overall_violation)

        # THREE STATES. `is_satisfied` is fail-closed False both for a measured
        # breach and for a constraint nobody could evaluate (base.py says so in
        # ConstraintViolation's own docstring and asks callers to pair it with
        # the insufficient_data flag); this class forwarded the False alone, so
        # a could-not-check read as a measured breach. Measured before the fix
        # on a single-level sensitive attribute: final_violation_=nan,
        # constraint_satisfied_=False, and a warning that stated the comparison
        # "final violation nan > tolerance 0.05", which never happened.
        # The magnitude is left exactly as the constraint measured it, and only
        # the VERDICT is withdrawn. Blanking a mostly-real violation because a
        # handful of rows were unscored would be the over-correction this
        # campaign warns about; the same narrow line is drawn in
        # post_processing.threshold_optimization.optimizer._qualified_feasibility.
        unevaluable_constraint = bool(final_violation.could_not_evaluate)
        could_not_evaluate = unevaluable_constraint or bool(n_unscored)
        self.insufficient_data_ = could_not_evaluate
        self.constraint_satisfied_ = (
            None if could_not_evaluate else bool(final_violation.is_satisfied)
        )

        # G10, 2026-09-30. A NEUTERED MITIGATION REPORTS SUCCESS, NOT FAILURE.
        # A rate-based constraint is satisfied EXACTLY when the decision column
        # is constant: approving everybody, or rejecting everybody, gives every
        # group the same rate, so the violation is a true 0.0 and it grades
        # nothing. This class's own docstring already quotes that feedback for
        # the all-NaN score column, and the constant-SCORE column walked past it
        # because _count_unscored_rows only sees non-finite values. Measured
        # before this disclosure, 200 rows in two groups with
        # DemographicParityConstraint(tolerance=0.05):
        #
        #   y_prob = 0.7 everywhere -> thresholds {a: 0.5, b: 0.5},
        #       final_violation_ 0.0, constraint_satisfied_ True,
        #       insufficient_data_ False, ZERO warnings, predictions all 1
        #   y_prob = 0.0 everywhere -> the same, predictions all 0
        #   group a scored 0.60-0.95 and group b 0.05-0.40 (a real disparity)
        #       -> thresholds {a: 0.9492, b: 0.5}, final_violation_ 0.0,
        #       constraint_satisfied_ True, insufficient_data_ False, ZERO
        #       warnings, and an acceptance rate of 0.000: the search reached
        #       perfect demographic parity by REJECTING EVERY APPLICANT
        #
        # The number is kept, because it is a real measurement of the rule that
        # was chosen, and the READING is labelled. That is the line
        # FairnessTrainingAnalyzer.compare_methods already draws with
        # parameters['degenerate_constant_predictions'], and it is why
        # constraint_satisfied_ is NOT withdrawn here: the constraint genuinely
        # is met, it is just met vacuously, and collapsing that into the
        # could-not-check state would lose the distinction the flag exists for.
        final_labels = np.unique(y_pred_final)
        self.degenerate_constant_predictions_ = bool(
            y_pred_final.size > 0 and final_labels.size < 2
        )
        if self.degenerate_constant_predictions_:
            decision = "ACCEPTS" if float(final_labels[0]) >= 0.5 else "REJECTS"
            warnings.warn(
                f"ThresholdOptimizer: the fitted thresholds "
                f"{ {g: round(float(t), 4) for g, t in thresholds.items()} } give the SAME "
                f"decision to all {y_pred_final.size} fitted row(s) (the rule {decision} "
                f"every row), so the {final_violation.constraint_type} violation "
                f"{self.final_violation_!r} is VACUOUS rather than good: a rule that "
                f"separates nobody gives every group the same rate and satisfies every "
                f"rate-based constraint by construction. Read "
                f"degenerate_constant_predictions_ beside constraint_satisfied_. A "
                f"constant SCORE column, or a grid on which only the extremes were "
                f"feasible, produces this.",
                UserWarning,
                stacklevel=2,
            )

        if unevaluable_constraint:
            warnings.warn(
                f"ThresholdOptimizer: the {final_violation.constraint_type} constraint "
                f"could NOT be evaluated on this data (final violation "
                f"{self.final_violation_!r}); a fairness constraint compares groups, and "
                f"there was no pair of measurable group rates to compare. No candidate "
                f"threshold could be ranked against another, so every group keeps the "
                f"0.5 seed and the thresholds returned were NOT selected for fairness. "
                f"This is could-not-check, not a breach: constraint_satisfied_ is None "
                f"and insufficient_data_ is True.",
                UserWarning,
                stacklevel=2,
            )
        elif n_unscored:
            warnings.warn(
                f"ThresholdOptimizer: the {final_violation.constraint_type} violation "
                f"{self.final_violation_:.4f} was computed over {n_unscored} row(s) that "
                f"had no score and entered it as fabricated rejections, so it is part "
                f"measurement and part fabrication. No verdict is given: "
                f"constraint_satisfied_ is None and insufficient_data_ is True.",
                UserWarning,
                stacklevel=2,
            )
        elif not final_violation.is_satisfied:
            warnings.warn(
                "ThresholdOptimizer: no joint threshold assignment on the "
                "grid satisfied the fairness constraint (final violation "
                f"{self.final_violation_:.4f} > tolerance "
                f"{self._constraint.tolerance}). Predictions will use the "
                "best thresholds found."
            )

        self._thresholds = thresholds
        self._fitted = True
        return self

    def _apply_thresholds(
        self,
        y_prob: np.ndarray,
        thresholds: Dict[str, float],
        gm: GroupManager,
    ) -> np.ndarray:
        """Apply group-specific thresholds.

        Every group present in `gm` must carry a fitted threshold; a group
        that does not is refused, never scored at a default.
        """
        # S-07 (2026-09-09): this read `thresholds.get(group, 0.5)`. During
        # fit() the fallback cannot fire (the dict is seeded from the same
        # gm.groups), but predict() builds gm from the PREDICT-time attribute,
        # so a group absent from the fit was scored at an unoptimised 0.5 and
        # its labels returned as though the per-group optimisation had covered
        # it. get_thresholds() does not list that group either, so the caller
        # had no signal at all. Measured before the fix: fitting on A and B
        # (A -> 0.4226, B -> 0.5) and predicting on A, B and C returned C's
        # rows identical to a plain 0.5 rule, with no warning and no error.
        # An unseen group is could-not-check, and predict() returns hard int
        # labels with no value that can carry "unknown", so it is refused.
        missing = [group for group in gm.groups if group not in thresholds]
        if missing:
            raise ValueError(
                f"ThresholdOptimizer: no fitted threshold for group(s) {sorted(missing)!r}. "
                f"Fitted groups: {sorted(thresholds)!r}. These groups were never seen by "
                "fit(), so no threshold was optimised for them; scoring them at a default "
                "would report an unoptimised decision rule as an optimised one. Refit "
                "including these groups, or drop their rows before calling predict()."
            )

        y_pred = np.zeros_like(y_prob, dtype=int)

        for group in gm.groups:
            mask = gm.get_mask(group)
            threshold = thresholds[group]
            y_pred[mask] = (y_prob[mask] >= threshold).astype(int)

        return y_pred

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Predict using fitted thresholds.

        Args:
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute

        Returns:
            Thresholded predictions

        Raises:
            ValueError: if any row has no usable score (NaN or infinite
                y_prob), or belongs to a group fit() never saw. Both are
                could-not-check, and an int label array has no room to say so.
        """
        if not self._fitted:
            raise RuntimeError("ThresholdOptimizer is not fitted. Call fit() first.")

        y_prob = coerce_to_array(y_prob, "y_prob")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        _refuse_unscored_rows(y_prob, caller="ThresholdOptimizer")

        gm = GroupManager(sensitive_attr)
        return self._apply_thresholds(y_prob, self._thresholds, gm)

    def get_thresholds(self) -> Dict[str, float]:
        """Get the fitted thresholds.

        These are a decision rule, not a measurement, and the search that
        produced them can have measured nothing: every group is seeded at 0.5
        and a candidate only displaces the seed when the constraint could be
        evaluated, so when ``insufficient_data_`` is True what comes back is
        the seed itself and it was NOT selected for fairness. Read
        ``insufficient_data_`` beside this; a 0.5 here is otherwise
        indistinguishable from an optimised 0.5.
        """
        if not self._fitted:
            raise RuntimeError("ThresholdOptimizer is not fitted.")
        return self._thresholds.copy()
