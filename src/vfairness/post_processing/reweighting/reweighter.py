"""
Prediction Reweighting Methods for Fairness.

This module provides methods for adjusting model predictions post-hoc to
achieve fairness constraints while preserving as much predictive accuracy
as possible.

The only constraint any class here enforces is demographic parity: all four
reweighters equalise a group gap in predicted positive rate or in the score
distribution, and none conditions on y_true per group. A `constraint`
argument asking for anything else is accepted and warned about, and the
result records the honoured constraint next to the requested one rather than
carrying the requested label as if it had been met (F21, 2026-09-09).
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    reject_swapped_labels_and_scores,
)


class ReweightingMethod(Enum):
    """Enumeration of available reweighting methods."""

    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    REJECTION_OPTION = "rejection_option"
    DISTRIBUTION_MATCHING = "distribution_matching"
    CALIBRATED_EQUALIZATION = "calibrated_equalization"


# The only fairness constraint any reweighter in this module enforces.
#
# F21 (2026-09-09): `constraint` was written once in BaseReweighter.__init__
# and read nowhere. Measured on identical data with a fixed seed,
# PredictionReweighter(constraint='demographic_parity'),
# ('equalized_odds') and ('equal_opportunity') produced byte-identical
# transform output (sha256 7f17486794be8f64 for all three, multiplicative;
# 6c6584b6133e46d8 for all three, additive) and byte-identical result_
# metadata, and a __getattribute__ spy recorded 0 reads of self.constraint
# during fit_transform(). The requested label was carried on the object
# while the fit did something else.
#
# What the module DOES enforce: every class here equalises a group gap in
# predicted positive rate or in the score distribution, which is demographic
# parity. PredictionReweighter moves each group's mean probability onto the
# overall mean (or target_rate); RejectionOptionClassifier flips near
# boundary cases toward the group with the lower positive rate;
# CalibratedEqualizer and DistributionMatcher map group score distributions
# onto a common one. None of them conditions on y_true per group, so no
# error-rate constraint (equalized odds, equal opportunity, predictive
# parity) can be enforced here by construction.
_HONOURED_CONSTRAINT = "demographic_parity"


def _warn_constraint_not_honoured(requested: str, where: str, stacklevel: int = 3) -> str:
    """Warn when `requested` is not the constraint this module enforces.

    Returns the constraint that IS enforced, so callers can record the
    honoured value rather than the requested one (F21, 2026-09-09). The
    requested value is kept on the object and in the result, so a reader can
    always see what was asked for next to what was done.
    """
    if requested != _HONOURED_CONSTRAINT:
        warnings.warn(
            f"{where}: constraint={requested!r} is NOT enforced. Every reweighter in "
            f"this module equalises a group gap in predicted positive rate or score "
            f"distribution, which is {_HONOURED_CONSTRAINT!r}; none of them conditions "
            f"on y_true per group, so an error-rate constraint cannot be enforced here. "
            f"The fit proceeds under {_HONOURED_CONSTRAINT!r}, and result_ records that "
            f"as honoured_constraint next to requested_constraint={requested!r}. Its "
            f"fairness_improvement is measured against the honoured constraint, not the "
            f"requested one. For equalized_odds or equal_opportunity use "
            f"ThresholdOptimizer or GroupThresholdOptimizer, which do enforce theirs.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return _HONOURED_CONSTRAINT


@dataclass
class ReweightingResult:
    """
    Result of prediction reweighting.

    Attributes:
        method: Reweighting method used
        group_adjustments: Adjustment factors per group
        original_metrics: Metrics before reweighting
        adjusted_metrics: Metrics after reweighting
        fairness_improvement: Change in fairness metrics, measured against
            honoured_constraint. It is NOT a measurement of progress toward
            requested_constraint when the two differ.
        calibration_impact: Impact on probability calibration
        requested_constraint: Constraint the caller asked for, or None when a
            result was built without recording one.
        honoured_constraint: Constraint the fit actually enforced, or None
            when a result was built without recording one. Never the
            requested value unless the fit really enforced it (F21).
        groups_not_compared: Groups present in the data whose value could not
            be measured, and which are therefore NOT in the max-min spread
            reported as `disparity` / `mean_disparity`. An empty list means
            every group was compared; None means a result was built without
            recording it. A finite disparity beside a non-empty list here is a
            measurement of the LISTED-OUT remainder only, never of the whole
            frame (BGL-D-FINAL, 2026-09-17).
        unfittable_groups: Group -> why no adjustment could be fitted for it.
            The rows of those groups come back NaN from transform(), and their
            group_adjustments entry is a refusal, not a neutral factor. None
            when a result was built without recording it.
    """

    method: ReweightingMethod
    group_adjustments: Dict[str, Any]
    original_metrics: Dict[str, float]
    adjusted_metrics: Dict[str, float]
    fairness_improvement: Dict[str, float]
    calibration_impact: Dict[str, float]
    requested_constraint: Optional[str] = None
    honoured_constraint: Optional[str] = None
    groups_not_compared: Optional[List[str]] = None
    unfittable_groups: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "method": self.method.value,
            "group_adjustments": self.group_adjustments,
            "original_metrics": self.original_metrics,
            "adjusted_metrics": self.adjusted_metrics,
            "fairness_improvement": self.fairness_improvement,
            "calibration_impact": self.calibration_impact,
            "requested_constraint": self.requested_constraint,
            "honoured_constraint": self.honoured_constraint,
            # Published, not merely computed. A partial disparity that a reader
            # cannot tell from a whole one is the same defect one layer up.
            "groups_not_compared": self.groups_not_compared,
            "unfittable_groups": self.unfittable_groups,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Reweighting Result", "=" * 40]
        lines.append(f"Method: {self.method.value}")

        if self.honoured_constraint is None:
            lines.append("Constraint enforced: not recorded")
        else:
            lines.append(f"Constraint enforced: {self.honoured_constraint}")
        if self.requested_constraint is not None:
            lines.append(f"Constraint requested: {self.requested_constraint}")
            if self.requested_constraint != self.honoured_constraint:
                lines.append(
                    f"  WARNING: {self.requested_constraint} was NOT enforced. The "
                    f"figures below are measured against "
                    f"{self.honoured_constraint}."
                )

        lines.append("\nGroup Adjustments:")
        for group, adj in self.group_adjustments.items():
            # A NaN factor is not a factor: it is the record that this group
            # could not be fitted. Say so, rather than printing "nan" next to
            # the real multipliers (BGL-D, 2026-09-11).
            if isinstance(adj, (int, float, np.integer, np.floating)) and not np.isfinite(adj):
                # The cause is not knowable from the factor alone: a group can be
                # unfittable because nothing in it was scored, or because no
                # factor of this kind could reach the target from its mean
                # (G008). The reason itself is printed under "Groups NOT fitted".
                lines.append(
                    f"  {group}: NOT FITTED (no adjustment could be fitted; see the reason below)"
                )
            else:
                lines.append(f"  {group}: {adj}")

        lines.append("\nFairness Improvement:")
        for metric, change in self.fairness_improvement.items():
            # `nan < 0` is False, so an unmeasurable change used to be printed
            # with the improvement arrow.
            if not np.isfinite(change):
                lines.append(f"  {metric}: NOT MEASURED (no comparison was possible)")
                continue
            # The SIGN carries the whole finding, so it is printed. `abs(change)`
            # beside an arrow picked from that sign reported a fit that made the
            # gap WORSE as an improvement, under a heading that says
            # "Fairness Improvement": executed on two groups whose positive rate
            # was 0.500 each, RejectionOptionClassifier took the disparity from
            # 0.00 to 1.00, recorded disparity_reduction=-1.0 correctly, and this
            # line rendered it as "disparity_reduction: 1.0000 ↓", which reads as a
            # full unit of improvement (G008, 2026-09-17). A reader of the text
            # surface saw the opposite of what was measured.
            if change > 0:
                lines.append(f"  {metric}: +{change:.4f} (improvement)")
            elif change < 0:
                lines.append(
                    f"  {metric}: {change:.4f} "
                    f"(REGRESSION: this fit moved it the wrong way by {abs(change):.4f})"
                )
            else:
                lines.append(f"  {metric}: 0.0000 (no change)")

        # A finite disparity over SOME of the groups is a real number about
        # those groups and nothing at all about the rest, so the rest are named
        # right under it (BGL-D-FINAL, 2026-09-17).
        if self.groups_not_compared:
            lines.append("\nGroups NOT compared:")
            lines.append(f"  {', '.join(self.groups_not_compared)}")
            lines.append(
                "  No value could be measured for these, so they are excluded from "
                "every disparity above. Those figures are a gap among the remaining "
                "groups and say nothing about these."
            )
        if self.unfittable_groups:
            lines.append("\nGroups NOT fitted:")
            for group in sorted(self.unfittable_groups):
                lines.append(f"  {group}: {self.unfittable_groups[group]}")

        lines.append("\nCalibration Impact:")
        for metric, impact in self.calibration_impact.items():
            # G10, 2026-09-30. THE COULD-NOT-CHECK WAS ALREADY WRITTEN FOUR LINES
            # UP AND THIS BLOCK UNDID IT. The Fairness Improvement block above
            # says "NOT MEASURED (no comparison was possible)" for a non-finite
            # change, and this one rendered exactly the same condition as the
            # value `nan`, under a heading that reads as a measurement. Measured
            # on a result whose calibration error could not be computed (no row
            # carrying a probability in [0, 1] to bin, which is what
            # _compute_calibration_error returns NaN for):
            #
            #   Fairness Improvement:
            #     disparity_reduction: NOT MEASURED (no comparison was possible)
            #   ...
            #   Calibration Impact:
            #     ece_change: nan
            #     original_ece: nan
            #
            # A reader of the text surface was shown a value for the one thing
            # nobody measured, directly under a correct refusal. Same wording as
            # the block above, so the two states read alike wherever they appear.
            if isinstance(impact, (int, float, np.integer, np.floating)) and not np.isfinite(
                impact
            ):
                lines.append(f"  {metric}: NOT MEASURED (no comparison was possible)")
                continue
            lines.append(f"  {metric}: {impact:.4f}")

        return "\n".join(lines)


class BaseReweighter(ABC):
    """
    Abstract base class for prediction reweighting methods.

    All reweighters should inherit from this class and implement
    the fit() and transform() methods.
    """

    def __init__(self, constraint: str = "demographic_parity"):
        """
        Initialize the reweighter.

        Args:
            constraint: Fairness constraint the caller is asking for. Only
                'demographic_parity' is enforced by any reweighter in this
                module; anything else is accepted, warned about at
                construction and recorded as requested but not honoured
                (F21, 2026-09-09). It is kept verbatim on self.constraint so
                a caller can always see what it asked for; what was actually
                enforced is self.honoured_constraint, and both appear in
                result_.
        """
        self.constraint = constraint
        self.honoured_constraint = _warn_constraint_not_honoured(
            constraint, type(self).__name__, stacklevel=4
        )
        self.is_fitted = False
        self.result_: Optional[ReweightingResult] = None

    @abstractmethod
    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "BaseReweighter":
        """
        Fit the reweighter to compute adjustment factors.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute for grouping
            sample_weight: Optional sample weights. Honoured only by
                PredictionReweighter (weighted group means and weighted
                default target rate). RejectionOptionClassifier,
                CalibratedEqualizer and DistributionMatcher estimate hard
                rates or quantiles and have no weighted form: they accept
                None or a uniform vector and refuse any other weighting
                with NotImplementedError instead of accepting and ignoring
                it (F7, 2026-09-09).

        Returns:
            self: The fitted reweighter
        """
        pass

    @abstractmethod
    def transform(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Transform predictions using the fitted reweighting.

        Args:
            y_prob: Predicted probabilities to transform
            sensitive_attr: Sensitive attribute for grouping

        Returns:
            Reweighted probabilities
        """
        pass

    def fit_transform(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(
            y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive_attr, sample_weight=sample_weight
        )
        return self.transform(y_prob, sensitive_attr)

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Make predictions using reweighted probabilities.

        Args:
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute
            threshold: Decision threshold

        Returns:
            Binary predictions as an int array when every row carries a finite
            adjusted score. When any row does not (the model returned NaN, or
            the row's group could not be fitted), the return is a FLOAT array
            of 0.0 / 1.0 / NaN and the unscored rows are named in a
            UserWarning. `np.nan >= threshold` is False, so those rows used to
            be returned as a hard int 0, a confident rejection nobody measured
            (BGL-S2, 2026-09-16).
        """
        y_prob_adjusted = self.transform(y_prob, sensitive_attr)
        return _decide(y_prob_adjusted, threshold, f"{type(self).__name__}.predict()")


def _finite_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of the entries that hold a finite number.

    Entries that cannot be read as numbers at all are reported as finite
    here and left to the caller's own arithmetic, so this helper only ever
    removes NaN and +/- inf.
    """
    try:
        return np.isfinite(np.asarray(values, dtype=float))
    except (TypeError, ValueError):
        return np.ones(len(values), dtype=bool)


def _group_disparity(group_values: Dict[str, float]) -> float:
    """Max-min spread over the groups that COULD be compared, or NaN.

    Three states, not two (BGL-D, 2026-09-11). ``max(...) - min(...)`` returns
    0.0, which reads as perfect parity, in two cases where no comparison
    happened at all:

    * ONE group: the spread of a single value is structurally 0.0 for every
      possible input, so it is not a measurement of a between-group gap.
    * A group whose value could not be measured (NaN). Python's ``max``/``min``
      also skip a NaN that is not the first element, so ``{'A': 0.5, 'B': nan}``
      collapsed to 0.5 - 0.5 = 0.0, the perfect-parity reading, on a group
      nobody scored.

    BGL-D-FINAL (2026-09-17). Refusing the WHOLE measurement because ONE group
    is unmeasurable is that same defect running backwards: it throws away
    evidence that really was collected. Measured at ``create_reweighter`` on a
    40 A + 40 B + 1 unscored C frame, where A and B carry a positive-rate gap
    of exactly 1.00 and a mean-probability gap of 0.6103, every reweighter
    published ``disparity: nan`` and ``disparity_reduction: nan``: a CRITICAL,
    fully measured gap reported as could-not-check. The spread is taken over
    the groups that WERE measured; the ones that were not are named by
    :func:`_groups_not_compared` and carried on the result, so a partial
    comparison can never be mistaken for a complete one.

    NaN is still returned when FEWER THAN TWO groups are measurable, because
    then no pair exists and there is nothing to subtract. It says
    could-not-check and propagates into every improvement figure derived
    from it.
    """
    measurable = [float(v) for v in group_values.values() if np.isfinite(v)]
    if len(measurable) < 2:
        return float("nan")
    return max(measurable) - min(measurable)


def _groups_not_compared(*group_values: Dict[str, float]) -> List[str]:
    """Names of every group left out of the disparities built from these dicts.

    A finite spread over a subset of the groups is a true measurement OF THAT
    SUBSET and says nothing at all about the rest, so the rest have to be named
    wherever the number is published (BGL-D-FINAL, 2026-09-17).
    """
    excluded = {
        str(group)
        for values in group_values
        for group, value in values.items()
        if not np.isfinite(value)
    }
    return sorted(excluded)


def _disparity_pair(
    where: str,
    original_values: Dict[str, float],
    adjusted_values: Dict[str, float],
) -> Tuple[float, float, List[str]]:
    """Before/after disparity plus the groups neither figure could compare.

    One call site per fit(), so the partial-comparison disclosure is warned
    about exactly once and the same list reaches ``result_``. A warning fires
    only when at least one of the two figures really is a number: when both are
    NaN the refusal warnings raised upstream already say why, and adding a
    second one about a spread that does not exist would be noise.
    """
    original = _group_disparity(original_values)
    adjusted = _group_disparity(adjusted_values)
    not_compared = _groups_not_compared(original_values, adjusted_values)
    if not_compared and (np.isfinite(original) or np.isfinite(adjusted)):
        compared = sorted(
            (set(map(str, original_values)) | set(map(str, adjusted_values))) - set(not_compared)
        )
        warnings.warn(
            f"{where}: the disparity figures are a max-min spread over {compared} only. "
            f"Group(s) {not_compared} have no measurable value and are excluded from "
            f"them, so those figures measure the gap AMONG THE COMPARED GROUPS and say "
            f"nothing about the excluded one(s); they are named in "
            f"result_.groups_not_compared. The measurement is not discarded outright "
            f"because a real gap between the groups that were scored is evidence, and "
            f"reporting NaN for it would throw that evidence away.",
            UserWarning,
            stacklevel=3,
        )
    return original, adjusted, not_compared


def _warn_no_between_group_comparison(where: str, groups: Any, also: str = "") -> None:
    """Warn that a disparity was asked for where no pair of groups exists.

    BGL-S2 (2026-09-16). ``max(values) - min(values)`` over a one-element dict
    is 0.0 for every possible input, and 0.0 on a disparity scale reads as
    perfect parity. :func:`_group_disparity` already returns NaN there, but a
    NaN alone does not tell the reader WHY, so each fit() says it by name.

    ``also`` names any OTHER field of the same result that is structurally
    fixed on this input, so the warning covers every published number that is
    not a measurement and not only the disparity ones (BGL-D-FINAL,
    2026-09-17).
    """
    listed = sorted(str(g) for g in groups)
    warnings.warn(
        f"{where}: sensitive_attr holds {len(listed)} group ({listed}), so there is no "
        f"pair of groups to compare and no between-group disparity exists to measure. "
        f"Every disparity and disparity_reduction field in result_ is NaN "
        f"(could-not-check), not 0.0: 0.0 would report perfect parity from a "
        f"comparison that never happened." + (f" {also}" if also else ""),
        UserWarning,
        stacklevel=3,
    )


# Smallest number of scored rows a per-group percentile grid may be estimated
# from. 30 is the min_group_size default the rest of the library already uses
# (streaming.py, post_processing/calibration/analyzer.py, calibration/tradeoffs.py).
_DEFAULT_MIN_GROUP_SIZE = 30


def _quantile_grid_refusal(
    label: str,
    values: np.ndarray,
    min_group_size: int,
) -> Optional[str]:
    """Reason this group's score distribution cannot be estimated, or None.

    BGL-S2 (2026-09-16). ``np.percentile`` returns a grid for any non-empty
    input, including one that carries no information at all, and
    ``np.interp(x, constant_xp, fp)`` then returns numpy's tie-break on an
    undefined ``xp`` rather than a mapping. Measured: a single minority row
    scoring 0.05 in a 29 + 1 split came out of CalibratedEqualizer as
    0.9470751842323551, exactly the POOLED MAXIMUM, and the run reported a
    0.26 "disparity reduction" for it. The three refusals below are the cases
    where the grid is not an estimate of anything:

    * no finite score at all (nothing was scored),
    * fewer scored rows than ``min_group_size``,
    * a single distinct score, so the grid is constant.

    The distinct-value test reads ``np.unique`` on the RAW scores rather than
    a variance, because an accumulated statistic is not exactly 0.0 at every n.
    """
    finite = _finite_mask(values)
    n_total = int(np.size(finite))
    n_finite = int(np.count_nonzero(finite))
    if n_finite == 0:
        return (
            f"{label} has no finite predicted probability among its {n_total} row(s), "
            f"so its score distribution could not be estimated"
        )
    if n_finite < min_group_size:
        return (
            f"{label} has {n_finite} scored row(s), fewer than "
            f"min_group_size={min_group_size}, so its score distribution could not "
            f"be estimated"
        )
    if np.unique(np.asarray(values, dtype=float)[finite]).size < 2:
        return (
            f"{label} has a single distinct score across its {n_finite} scored row(s), "
            f"so its percentile grid is constant and mapping through it is undefined"
        )
    return None


def _warn_unfittable_groups(where: str, reasons: Dict[str, str]) -> None:
    """Name every group whose distribution could not be estimated."""
    if not reasons:
        return
    detail = "; ".join(reasons[g] for g in sorted(reasons))
    warnings.warn(
        f"{where}: {detail}. Rows in those group(s) are returned as NaN "
        f"(could-not-check) rather than as their own unchanged score, which would be "
        f"indistinguishable from an adjusted one, and every disparity computed over "
        f"them is NaN.",
        UserWarning,
        stacklevel=3,
    )


def _reject_nan_threshold(threshold: float, where: str, name: str = "threshold") -> float:
    """Refuse a decision threshold that is not a number.

    BGL3 (2026-09-27). Every comparison with NaN is False, so a NaN threshold
    decides nothing and reads as a decision against everybody. Measured on 120
    rows whose A/B positive-rate gap is exactly 1.00:

    * ``PredictionReweighter(...).predict(p, g, threshold=nan)`` returned an
      int array of 120 zeros, dtype int64, with NO warning: a confident
      rejection of every applicant, from a comparison that never happened. The
      NaN-score path in :func:`_decide` was already three-state; the threshold
      side of the same comparison was not.
    * ``RejectionOptionClassifier(threshold=nan).fit(...)`` published
      ``original disparity 0.0, adjusted 0.0, disparity_reduction 0.0`` on that
      1.00 gap, and warned that "every measurable group ['A', 'B'] has the same
      positive rate (0)". Both the 0.0 and the warning describe a measurement
      nobody took.

    ``ReweightingAnalyzer.analyze_method`` already refuses this argument for
    exactly this reason; the reweighters it drives did not. NaN only,
    deliberately: +/-inf is a real rule ("accept nobody" / "accept everybody")
    and every rate that follows from it IS a measurement of that rule.
    """
    value = float(threshold)
    if np.isnan(value):
        raise ValueError(
            f"{where}: {name}={threshold!r} is not a number, so it decides nothing while "
            f"still looking like a decision. Every comparison with NaN is False: a NaN "
            f"threshold puts each group's positive rate at 0.0 and the gap between them "
            f"at 0.0, and a NaN critical-region width empties the region, which is then "
            f"reported as a correction that found nothing to do. Both read as perfect "
            f"parity rather than as a comparison that never happened."
        )
    return value


def _reject_degenerate_quantile_grid(n_quantiles: int, where: str, name: str) -> int:
    """Refuse a percentile grid too small to be a mapping.

    BGL3 (2026-09-27). ``np.linspace(0, 100, n + 1)`` with ``n = 0`` is a
    SINGLE percentile, and ``np.interp(x, [q], [t])`` returns ``t`` for every
    ``x``: not a quantile mapping but a collapse onto one number. Measured,
    ``CalibratedEqualizer(n_quantiles=0)`` on 120 rows carrying a group mean gap
    of 0.5426 rewrote all 120 scores to the single value 0.0518126 and published
    ``adjusted mean_disparity 0.0`` with ``disparity_reduction 0.5426``,
    ``unfittable_groups {}`` and not one warning: perfect parity, reported by a
    mitigation that had thrown the scores away. ``n_quantiles=-1`` reached
    numpy's own "array of sample points is empty".

    The grid needs at least two points, so the smallest honest value is 1
    (the 0th and 100th percentiles, a min-max mapping).
    """
    try:
        value = int(n_quantiles)
    except (TypeError, ValueError):
        raise ValueError(
            f"{where}: {name} must be a positive integer, got {n_quantiles!r}. A grid of "
            f"fewer than two percentiles is not a mapping: np.interp against it returns "
            f"one constant for every input."
        ) from None
    if value < 1:
        raise ValueError(
            f"{where}: {name} must be at least 1 (the 0th and 100th percentiles), got "
            f"{value}. A single-point grid maps every score onto one number and then "
            f"reports the resulting 0.0 group spread as a disparity this method removed."
        )
    return value


def _count_modified(original: np.ndarray, adjusted: np.ndarray) -> int:
    """Rows whose score the fit actually rewrote.

    A row that was NaN before and is NaN after was not modified, so the plain
    ``!=`` count (NaN != NaN is True) would report it as one.
    """
    before = np.asarray(original, dtype=float)
    after = np.asarray(adjusted, dtype=float)
    both_unscored = ~np.isfinite(before) & ~np.isfinite(after)
    return int(np.count_nonzero(~((after == before) | both_unscored)))


def _decide(y_prob_adjusted: np.ndarray, threshold: float, where: str) -> np.ndarray:
    """Threshold adjusted scores into decisions, refusing to decide unscored rows.

    BGL-S2 (2026-09-16). ``np.nan >= 0.5`` is False, so every row the model
    never scored used to be cast to a hard ``0``: a confident rejection,
    indistinguishable in the returned int array from a measured 0.4. Executed
    on 60 all-NaN probabilities, ``predict`` returned ``(array([0]),
    array([60])))`` with no error and no warning, while ``transform`` on the
    same input honestly returned 60 NaNs one line earlier.

    Three states, never two. When every row carries a finite score the return
    is the int array of 0/1 decisions it has always been. When any row does
    not, the return is a float array carrying 0.0, 1.0 and NaN, and the NaN
    rows are named in a warning: not scored is not denied.

    The THRESHOLD side of the same comparison is refused rather than returned
    as NaN, because a NaN threshold makes no row decidable at all: there is no
    subset of the data to report on. See :func:`_reject_nan_threshold` for what
    it measured.
    """
    threshold = _reject_nan_threshold(threshold, where)
    scores = np.asarray(y_prob_adjusted, dtype=float)
    finite = np.isfinite(scores)
    if bool(finite.all()):
        return (scores >= threshold).astype(int)
    n_unscored = int(np.size(finite) - np.count_nonzero(finite))
    warnings.warn(
        f"{where}: {n_unscored} of {np.size(finite)} row(s) have no finite adjusted "
        f"score, so no decision was made for them. They are NaN in the returned float "
        f"array, not 0: a 0 here is a measured rejection and these rows were never "
        f"scored. The array is float rather than int for exactly that reason.",
        UserWarning,
        stacklevel=3,
    )
    decisions = np.full(scores.shape, float("nan"), dtype=float)
    decisions[finite] = (scores[finite] >= threshold).astype(float)
    return decisions


def _compute_group_positive_rates(
    y_prob: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute positive prediction rates per group.

    A score that is not finite is not a negative decision. ``np.nan >= 0.5``
    is False, so an unscored sample was counted as below threshold and a
    group with no readable score at all reported the positive rate 0.0 --
    a measurement-shaped number nobody measured (BGL-D, 2026-09-11).
    Non-finite scores are excluded from the rate, and a group left with no
    finite score gets NaN (could-not-check), which propagates through
    :func:`_group_disparity` into every gap computed from these rates.

    A threshold that is not a number is refused outright: it puts every rate at
    0.0 and every gap between them at 0.0, which is the perfect-parity reading
    (see :func:`_reject_nan_threshold`).
    """
    threshold = _reject_nan_threshold(threshold, "_compute_group_positive_rates")
    unique_groups = np.unique(sensitive_attr)
    rates = {}

    for group in unique_groups:
        mask = sensitive_attr == group
        group_probs = y_prob[mask]
        finite = _finite_mask(group_probs)
        n_dropped = int(np.size(finite) - np.count_nonzero(finite))
        if not np.any(finite):
            rates[str(group)] = float("nan")
            warnings.warn(
                f"Group {str(group)!r} has no finite predicted probability "
                f"({np.size(finite)} sample(s), all non-finite), so its positive "
                f"rate could not be measured and is NaN. It is not 0.0: nothing "
                f"about this group was scored.",
                UserWarning,
                stacklevel=2,
            )
        else:
            if n_dropped:
                warnings.warn(
                    f"Group {str(group)!r}: {n_dropped} of {np.size(finite)} "
                    f"predicted probabilities are non-finite and are excluded from "
                    f"its positive rate, which therefore covers only the scored "
                    f"samples. Counting them as below threshold would report them "
                    f"as negative decisions.",
                    UserWarning,
                    stacklevel=2,
                )
            rates[str(group)] = np.mean(group_probs[finite] >= threshold)

    return rates


def _compute_group_mean_probs(
    y_prob: np.ndarray,
    sensitive_attr: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute (optionally weighted) mean predicted probabilities per group.

    With sample_weight=None and every score finite this is np.mean per group,
    byte-identical to the unweighted form (np.average delegates to mean when
    weights is None).

    BGL-D-FINAL (2026-09-17). One non-finite score used to make the WHOLE
    group's mean NaN, so a group with 99 scored rows and one hole was recorded
    as unmeasurable: its adjustment became NaN, transform() returned its rows
    as NaN, and it dropped out of every disparity. That discards 99 real
    measurements because of one missing row, which is the could-not-check
    defect running backwards. Non-finite scores are excluded from the mean and
    named in a warning, exactly as :func:`_compute_group_positive_rates`
    already does for rates; a group with NO finite score is still NaN, because
    nothing about it was scored.
    """
    unique_groups = np.unique(sensitive_attr)
    means = {}

    for group in unique_groups:
        mask = sensitive_attr == group
        group_probs = y_prob[mask]
        finite = _finite_mask(group_probs)
        n_dropped = int(np.size(finite) - np.count_nonzero(finite))
        if not np.any(finite):
            means[str(group)] = float("nan")
            warnings.warn(
                f"Group {str(group)!r} has no finite predicted probability "
                f"({np.size(finite)} sample(s), all non-finite), so its mean "
                f"predicted probability could not be measured and is NaN. It is not "
                f"0.0 and not the overall mean: nothing about this group was scored.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if n_dropped:
            warnings.warn(
                f"Group {str(group)!r}: {n_dropped} of {np.size(finite)} predicted "
                f"probabilities are non-finite and are excluded from its mean, which "
                f"therefore covers only the scored samples. Discarding the whole "
                f"group's mean over them would throw away the rows that WERE scored.",
                UserWarning,
                stacklevel=2,
            )
        if sample_weight is None:
            means[str(group)] = float(np.mean(np.asarray(group_probs, dtype=float)[finite]))
        else:
            group_weights = sample_weight[mask]
            if not np.any(group_weights > 0):
                raise ValueError(
                    f"sample_weight is zero for every sample in group {group!r}: "
                    f"its weighted mean probability cannot be estimated."
                )
            scored_weights = group_weights[finite]
            if not np.any(scored_weights > 0):
                raise ValueError(
                    f"sample_weight is zero for every SCORED sample in group {group!r}: "
                    f"its weighted mean probability cannot be estimated."
                )
            means[str(group)] = float(
                np.average(np.asarray(group_probs, dtype=float)[finite], weights=scored_weights)
            )

    return means


def _validate_sample_weight(sample_weight: ArrayLike, n_samples: int) -> np.ndarray:
    """Coerce sample_weight and refuse a length mismatch or a negative weight."""
    weights = coerce_to_array(sample_weight).astype(float)
    if len(weights) != n_samples:
        raise ValueError(f"sample_weight has {len(weights)} entries for {n_samples} samples.")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative.")
    return weights


def _refuse_unimplemented_sample_weight(
    sample_weight: Optional[ArrayLike], n_samples: int, class_name: str
) -> None:
    """Refuse a sample_weight that this class would accept and ignore.

    F7 (2026-09-09): every fit() in this module accepted sample_weight and
    never read it. Measured with weights 0.01 vs 100 across the two groups,
    the outputs of RejectionOptionClassifier, CalibratedEqualizer and
    DistributionMatcher were byte-identical to the unweighted fit. Those
    three estimate hard rates or quantiles and have no weighted form, so a
    non-uniform weight vector is refused by name. None and a uniform vector
    (which is the unweighted estimate) are accepted.
    """
    if sample_weight is None:
        return
    weights = _validate_sample_weight(sample_weight, n_samples)
    if weights.size == 0 or np.all(weights == weights[0]):
        return
    raise NotImplementedError(
        f"{class_name}.fit() does not honour a non-uniform sample_weight: it "
        f"estimates hard rates or quantiles, which have no weighted form here. "
        f"Pass sample_weight=None, or use PredictionReweighter, which weights "
        f"its group means."
    )


class PredictionReweighter(BaseReweighter):
    """
    Prediction reweighter that adjusts probabilities to achieve fairness.

    This reweighter applies multiplicative or additive adjustments to
    predicted probabilities based on group membership, moving every group's
    mean predicted probability onto the overall mean (or onto target_rate).
    That equalises the overall positive rate, which is demographic parity;
    it is the only constraint this class enforces, whatever `constraint`
    asks for (see the `constraint` argument below and F21).

    Example:
        >>> reweighter = PredictionReweighter(
        ...     constraint='demographic_parity',
        ...     method='multiplicative'
        ... )
        >>> reweighter.fit(y_true, y_prob, gender)
        >>> y_prob_fair = reweighter.transform(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: prediction_reweighting. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        constraint: str = "demographic_parity",
        method: str = "multiplicative",
        target_rate: Optional[float] = None,
    ):
        """
        Initialize the prediction reweighter.

        Args:
            constraint: Fairness constraint the caller is asking for. The fit
                always equalises the overall positive rate, which is
                'demographic_parity'. 'equalized_odds', 'equal_opportunity'
                and any other value are accepted (the platform has saved
                configurations carrying them), warned about here, and
                recorded in result_ as requested_constraint alongside
                honoured_constraint='demographic_parity' so the result no
                longer asserts a constraint it did not enforce (F21,
                2026-09-09). For an error-rate constraint use
                ThresholdOptimizer or GroupThresholdOptimizer.
            method: Adjustment method ('multiplicative' or 'additive')
            target_rate: Target positive rate (default: overall rate)

        Note on ranking. A `preserve_ranking` flag used to be accepted here
        (F12, removed 2026-09-09). It was stored and never read, so True and
        False produced identical output, and the claim it made was false:
        the adjustment is clipped to [0, 1], and every sample pushed past a
        boundary lands on the same value, so within-group ranking is
        preserved only up to those ties. Measured on a saturating fit, 54 of
        300 samples were clipped and the within-group Spearman fell to
        0.997. The number of clipped samples is now reported as
        `result_.calibration_impact['n_clipped']` so the loss of ranking is
        visible instead of promised away.
        """
        super().__init__(constraint)
        self.method = method
        self.target_rate = target_rate
        self.adjustments_: Dict[str, float] = {}
        self.target_rate_: float = 0.5

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "PredictionReweighter":
        """
        Fit the reweighter to compute adjustment factors.

        Computes the adjustment needed for each group to achieve
        the target positive rate. That is a demographic-parity adjustment
        regardless of the `constraint` given at construction; result_ records
        which constraint was honoured and which was requested (F21).

        sample_weight, when given, weights the per-group mean probabilities
        and the default target rate (np.average). The diagnostic metrics
        stored in result_ stay unweighted: they describe the data, not the
        fit. F7 (2026-09-09): this argument was previously accepted and
        never read; weights of 0.01 vs 100 across groups left the output
        byte-identical to the unweighted fit.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)

        weights: Optional[np.ndarray] = None
        if sample_weight is not None:
            weights = _validate_sample_weight(sample_weight, len(y_true))
            if not np.any(weights > 0):
                raise ValueError("sample_weight must contain at least one positive weight.")

        if self.target_rate is not None:
            self.target_rate_ = self.target_rate
        else:
            # BGL-D-FINAL (2026-09-17). np.average over the RAW column returns
            # NaN as soon as ONE row is unscored, and a NaN target makes every
            # group's factor NaN: measured on a 40 A + 40 B + 1 unscored C
            # frame, both fully scored groups came back with a NaN adjustment
            # and NaN rows out of transform(), so one missing row discarded
            # eighty real ones. The default target is estimated from the rows
            # that WERE scored, and the ones that were not are named.
            pooled_finite = _finite_mask(y_prob)
            n_unscored = int(np.size(pooled_finite) - np.count_nonzero(pooled_finite))
            if not np.any(pooled_finite):
                self.target_rate_ = float("nan")
                warnings.warn(
                    f"{type(self).__name__}.fit(): none of the {np.size(pooled_finite)} "
                    f"predicted probabilities is finite, so the default target rate "
                    f"could not be estimated. It is NaN (could-not-check), and every "
                    f"adjustment derived from it is NaN rather than a neutral factor.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                if n_unscored:
                    warnings.warn(
                        f"{type(self).__name__}.fit(): {n_unscored} of "
                        f"{np.size(pooled_finite)} predicted probabilities are "
                        f"non-finite and are excluded from the default target rate, "
                        f"which therefore describes the scored rows only. Those rows "
                        f"stay NaN through transform() and get no decision.",
                        UserWarning,
                        stacklevel=2,
                    )
                self.target_rate_ = float(
                    np.average(
                        np.asarray(y_prob, dtype=float)[pooled_finite],
                        weights=None if weights is None else weights[pooled_finite],
                    )
                )

        group_means = _compute_group_mean_probs(y_prob, sensitive_attr, weights)

        # A refit must not inherit the previous fit's factors: adjustments_
        # was only ever written into, so a group present in an earlier fit and
        # absent from this one kept its stale multiplier and transform() went
        # on applying it.
        self.adjustments_ = {}

        # Guard ABOVE the method dispatch (BGL-D, 2026-09-11). A group whose
        # probabilities hold nothing finite was never scored, so neither a
        # multiplier nor an offset can be fitted for it. The multiplicative
        # branch below would otherwise send it to the `else` arm -- `nan > 0`
        # is False -- and record 1.0, the neutral multiplier that reads as
        # "this group needed no correction". NaN says could-not-check.
        unmeasurable = sorted(g for g, mean in group_means.items() if not np.isfinite(mean))
        unfittable_reasons: Dict[str, str] = {
            group: (
                f"group {group!r} has no finite mean predicted probability, so no "
                f"adjustment could be fitted for it"
            )
            for group in unmeasurable
        }
        if unmeasurable:
            warnings.warn(
                f"{type(self).__name__}.fit(): group(s) {unmeasurable} have no finite "
                f"mean predicted probability, so no adjustment could be fitted for "
                f"them. Their group_adjustments entry is NaN (could-not-check), not "
                f"1.0, and every disparity that would have to compare them is NaN. "
                f"1.0 would read as 'this group needed no correction'.",
                UserWarning,
                stacklevel=2,
            )
            for group in unmeasurable:
                self.adjustments_[group] = float("nan")
        measurable = {g: m for g, m in group_means.items() if np.isfinite(m)}

        # The NaN disparity below already refuses the one-group case, but a NaN
        # on its own does not tell the reader WHY, and create_reweighter()
        # publishes these fields with no group count anywhere in to_dict()
        # (BGL-S2, 2026-09-16). Say it in words, as the other reweighters do.
        if len(group_means) < 2:
            # BGL-D-FINAL (2026-09-17). On a single cohort with target_rate=None
            # the target IS this cohort's own mean, so target/mean is exactly
            # 1.0 and target-mean is exactly 0.0 for EVERY possible input:
            # proven over six datasets, the published factor never varied. That
            # is the same structural non-measurement as the 0.0 disparity, and
            # 1.0 reads as "this group needed no correction", so the warning
            # names it. With an explicit target_rate the factor IS a fitted
            # quantity and is not named here.
            also = ""
            if self.target_rate is None:
                neutral = "1.0" if self.method == "multiplicative" else "0.0"
                also = (
                    f"group_adjustments is not a measurement either: with "
                    f"target_rate=None the target IS this cohort's own mean, so the "
                    f"published {self.method} factor is structurally exactly {neutral} "
                    f"for every possible input, not a fitted correction that happened "
                    f"to come out neutral. Pass an explicit target_rate to make it one."
                )
            _warn_no_between_group_comparison(
                f"{type(self).__name__}.fit()", sorted(group_means), also
            )

        if self.method == "multiplicative":
            not_correctable = []
            for group, mean in measurable.items():
                if mean > 0:
                    self.adjustments_[group] = self.target_rate_ / mean
                elif self.target_rate_ == 0.0:
                    # The group is already AT the target, so the identity really
                    # is the fitted factor here and 1.0 is a measurement.
                    self.adjustments_[group] = 1.0
                else:
                    # G008 (2026-09-17). `mean > 0` fails for a group whose mean
                    # predicted probability is 0, and the else arm recorded 1.0:
                    # the neutral multiplier, which reads as "this group needed
                    # no correction". No multiplicative factor exists here at
                    # all, because 0 times anything is 0, so the fit silently
                    # failed for that group and published the identity as its
                    # result. Measured on 60 rows scoring 0.0 in group A beside
                    # 60 rows of uniform(0.4, 0.9) in group B: A was recorded as
                    # 1.0, its rows stayed at 0.0, B's factor of 0.5 pushed every
                    # B row under the threshold, and result_ published
                    # adjusted disparity 0.0 with disparity_reduction 0.8333, an
                    # 83 point "improvement" produced by rejecting everybody.
                    # NaN says could-not-check: the rows come back NaN and the
                    # adjusted disparity is NaN, while the ORIGINAL disparity
                    # (the real 0.8333 gap against A) is still measured and
                    # still reported.
                    self.adjustments_[group] = float("nan")
                    not_correctable.append(group)
                    unfittable_reasons[group] = (
                        f"group {group!r} has a mean predicted probability of "
                        f"{float(mean):.6g}, so no multiplicative factor can reach the "
                        f"target rate of {self.target_rate_:.6g} and no adjustment could "
                        f"be fitted for it"
                    )
            if not_correctable:
                warnings.warn(
                    f"{type(self).__name__}.fit(): group(s) {sorted(not_correctable)} have a "
                    f"mean predicted probability that is not positive, so no multiplicative "
                    f"factor can move them to the target rate of {self.target_rate_:.6g} "
                    f"(0 times anything is 0). Their group_adjustments entry is NaN "
                    f"(could-not-check), not 1.0, their rows come back NaN from transform() "
                    f"and every disparity that would have to compare them is NaN. 1.0 would "
                    f"read as 'this group needed no correction'. Use method='additive', "
                    f"which has an offset that does reach the target from zero.",
                    UserWarning,
                    stacklevel=2,
                )
        elif self.method == "additive":
            for group, mean in measurable.items():
                self.adjustments_[group] = self.target_rate_ - mean
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Mark as fitted before internal transform call
        self.is_fitted = True

        original_rates = _compute_group_positive_rates(y_prob, sensitive_attr)
        # Same path transform() takes, kept unclipped for one step so the
        # samples the clip will tie at a boundary can be counted (F12).
        unclipped = self._adjust_unclipped(y_prob, sensitive_attr)
        n_clipped = int(((unclipped < 0) | (unclipped > 1)).sum())
        y_prob_adjusted = np.clip(unclipped, 0, 1)
        adjusted_rates = _compute_group_positive_rates(y_prob_adjusted, sensitive_attr)

        # Fairness improvement (reduction in disparity). NaN when there was no
        # between-group comparison to make at all (fewer than two groups with a
        # measurable rate), rather than the perfect-parity 0.0 a bare max-min
        # returns there (BGL-D, 2026-09-11). When SOME groups are measurable
        # and others are not, the spread over the measurable ones is reported
        # and the rest are named in groups_not_compared, because discarding a
        # real gap is the same defect backwards (BGL-D-FINAL, 2026-09-17).
        original_disparity, adjusted_disparity, not_compared = _disparity_pair(
            f"{type(self).__name__}.fit()", original_rates, adjusted_rates
        )

        self.result_ = ReweightingResult(
            method=ReweightingMethod.MULTIPLICATIVE
            if self.method == "multiplicative"
            else ReweightingMethod.ADDITIVE,
            group_adjustments=self.adjustments_,
            original_metrics={"mean_prob": float(np.mean(y_prob)), "disparity": original_disparity},
            adjusted_metrics={
                "mean_prob": float(np.mean(y_prob_adjusted)),
                "disparity": adjusted_disparity,
            },
            fairness_improvement={"disparity_reduction": original_disparity - adjusted_disparity},
            calibration_impact={
                "mean_shift": float(np.mean(y_prob_adjusted) - np.mean(y_prob)),
                # Samples the [0, 1] clip tied at a boundary; each one lost its
                # within-group rank (F12, 2026-09-09).
                "n_clipped": n_clipped,
            },
            requested_constraint=self.constraint,
            honoured_constraint=self.honoured_constraint,
            groups_not_compared=not_compared,
            unfittable_groups=unfittable_reasons,
        )

        self.is_fitted = True
        return self

    def _adjust_unclipped(self, y_prob: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Apply the fitted per-group adjustment without the [0, 1] clip.

        The loop runs over the groups PRESENT in `sensitive_attr`, not over the
        fitted adjustments. Iterating over `adjustments_` left a row whose group
        was never fitted at its own input value, returned in the same array as
        the adjusted ones: measured, a reweighter fitted on ['a', 'b'] and then
        asked for transform([0.30, 0.45, 0.55, 0.70], ['c'] * 4) returned
        [0.30, 0.45, 0.55, 0.70], byte-identical to the clipped input, with no
        warning, and predict() turned it into the confident decisions
        [0, 0, 1, 1]; the same four scores as fitted 'b' give [1, 1, 1, 1]
        (G008, 2026-09-17). This is the defect BGL-S2 fixed in
        CalibratedEqualizer.transform and DistributionMatcher.transform; this
        third copy of it was missed. An unadjusted row now comes back as NaN,
        which no caller can mistake for a reweighted score.
        """
        y_prob_adjusted = np.asarray(y_prob, dtype=float).copy()
        groups_str = np.asarray(sensitive_attr).astype(str)

        unfitted: Dict[str, str] = {}
        for group in [str(g) for g in np.unique(groups_str)]:
            mask = groups_str == group
            # A NaN factor IS present (it is this fit's refusal for that group)
            # and multiplies through to NaN on its own; `is None` means the group
            # was never seen at fit time at all.
            adjustment = self.adjustments_.get(group)
            if adjustment is None:
                unfitted[group] = (
                    f"group {group!r} carries {int(mask.sum())} row(s) here and was not "
                    f"present at fit time, so no adjustment was fitted for it"
                )
                y_prob_adjusted[mask] = float("nan")
                continue
            if self.method == "multiplicative":
                y_prob_adjusted[mask] = y_prob_adjusted[mask] * adjustment
            else:  # additive
                y_prob_adjusted[mask] = y_prob_adjusted[mask] + adjustment

        _warn_unfittable_groups(f"{type(self).__name__}.transform()", unfitted)

        return y_prob_adjusted

    def transform(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Transform predictions using fitted adjustments.

        The adjusted values are clipped to [0, 1]. Samples pushed past a
        boundary share that boundary value, so within-group ranking holds
        only up to those ties; fit() reports how many there were as
        result_.calibration_impact['n_clipped'].
        """
        if not self.is_fitted:
            raise RuntimeError("Reweighter must be fitted before calling transform()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)

        y_prob_adjusted = self._adjust_unclipped(y_prob, sensitive_attr)

        # Clip to valid probability range
        y_prob_adjusted = np.clip(y_prob_adjusted, 0, 1)

        return y_prob_adjusted


class RejectionOptionClassifier(BaseReweighter):
    """
    Rejection Option based Classification (ROC) for fairness.

    This method modifies predictions only for instances near the
    decision boundary (within a "critical region"), favoring the
    unprivileged group to achieve fairness.

    Based on: Kamiran et al. (2012) "Decision Theory for
    Discrimination-Aware Classification"

    Example:
        >>> roc = RejectionOptionClassifier(theta=0.1, unprivileged_group='female')
        >>> roc.fit(y_true, y_prob, gender)
        >>> y_pred_fair = roc.predict(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: rejection_option_classification. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        theta: float = 0.1,
        threshold: float = 0.5,
        unprivileged_group: Optional[str] = None,
    ):
        """
        Initialize the rejection option classifier.

        Args:
            theta: Width of the critical region around the threshold. Refused
                when it is not a number: the region becomes (nan, nan), no row
                can be inside it, and the fit then reports
                ``n_in_critical_region 0`` and ``disparity_reduction 0.0`` as
                though it had measured that this correction changes nothing
                (measured on 120 rows with a 1.00 gap, BGL3 2026-09-27).
            threshold: Decision threshold. Refused when it is not a number, see
                :func:`_reject_nan_threshold` for the measured consequence.
            unprivileged_group: Name of the unprivileged group (auto-detected if None)
        """
        super().__init__(constraint="demographic_parity")
        # Above every measurement, because both arguments define the comparison
        # rather than take part in it.
        self.theta = _reject_nan_threshold(theta, type(self).__name__, name="theta")
        self.threshold = _reject_nan_threshold(threshold, type(self).__name__)
        self.unprivileged_group = unprivileged_group
        self.critical_region_: Tuple[float, float] = (0.4, 0.6)
        self.detected_unprivileged_: Optional[str] = None
        # Every group that shares the LOWEST measurable positive rate. This is
        # the list transform() moves, and it exists because `min()` over a dict
        # breaks a tie by iteration order, which here is the order of the group
        # NAMES (G008, 2026-09-17). detected_unprivileged_ stays a single name
        # only while one group really is worst off; on a tie it is None, because
        # no single group was identified by the data.
        self.detected_unprivileged_groups_: List[str] = []
        # False once fit() has seen a group structure that carries no
        # between-group comparison (BGL-S2). transform() then refuses to move
        # anybody, because the disparity the move would correct was never
        # measured.
        self.assessable_: bool = False
        self.not_assessable_reason_: Optional[str] = None
        # Groups whose positive rate this fit actually measured. transform()
        # refuses to move anything else: a group absent at fit time is neither
        # known to be privileged nor known to be unprivileged (G008).
        self.fitted_groups_: List[str] = []

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "RejectionOptionClassifier":
        """
        Fit the ROC classifier.

        Determines the critical region and identifies the unprivileged group
        if not specified.

        sample_weight is not honoured (hard positive rates have no weighted
        form here); a non-uniform vector is refused, see BaseReweighter.fit.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)
        _refuse_unimplemented_sample_weight(sample_weight, len(y_true), type(self).__name__)

        self.critical_region_ = (self.threshold - self.theta, self.threshold + self.theta)

        # Detect unprivileged group (group with lower positive rate)
        group_rates = _compute_group_positive_rates(y_prob, sensitive_attr, self.threshold)
        # Rebuilt on every fit, so a refit cannot leave a previous dataset's
        # group looking fitted.
        self.fitted_groups_ = sorted(str(g) for g in group_rates)

        # Validate the NAMED group ABOVE every other branch (BGL-D-FINAL,
        # 2026-09-17). This refusal used to sit in the `else` arm below, so on
        # single-group data `if not self.assessable_` pre-empted it and
        # RejectionOptionClassifier(unprivileged_group='Z').fit(...) returned a
        # fitted object instead of raising: a caller who misspelled the group
        # name got a silent no-op. A name that is not in the data is a caller
        # error whatever the group structure turns out to be.
        named_group = None if self.unprivileged_group is None else str(self.unprivileged_group)
        if named_group is not None and named_group not in group_rates:
            raise ValueError(
                f"unprivileged_group={self.unprivileged_group!r} does not appear in "
                f"sensitive_attr (groups: {sorted(group_rates)}). Every row would be "
                f"treated as privileged and pushed DOWN, which is the opposite of the "
                f"requested correction."
            )

        # Guard ABOVE the detection (BGL-S2, 2026-09-16). `min()` over a
        # one-key dict returns that key, so a single-group fit named a group
        # "unprivileged" from a comparison with nothing, then flipped rows
        # toward it: measured, 6 of 60 scores were rewritten to threshold+0.01
        # and 6 of 60 decisions changed while result_ reported disparity
        # 0.0 -> 0.0. A rate that is NaN (no finite score in that group) is
        # also not a comparison, and `min` would rank it arbitrarily.
        measurable = {g: r for g, r in group_rates.items() if np.isfinite(r)}
        self.not_assessable_reason_ = None
        tied_at_lowest: List[str] = []

        if len(measurable) < 2:
            named = sorted(measurable)
            self.not_assessable_reason_ = (
                f"{len(measurable)} of {len(group_rates)} group(s) have a measurable "
                f"positive rate ({named}), so there is no pair of rates to compare and no "
                f"between-group gap exists to correct"
            )
        elif named_group is not None and named_group not in measurable:
            # Present in the data, but nothing in it was scored. Flipping rows
            # toward it would correct a gap one side of which was never
            # measured.
            self.not_assessable_reason_ = (
                f"unprivileged_group={named_group!r} is present but has no measurable "
                f"positive rate (no finite predicted probability in that group), so the "
                f"gap this fit would correct toward it was never measured"
            )

        # G008 (2026-09-17). `min()` returns the FIRST minimal key in iteration
        # order, and these rates are keyed by np.unique(), i.e. sorted group
        # names. When the lowest rate is shared, the group it names is therefore
        # decided by the alphabet, not by the data. Measured on two groups whose
        # positive rate was 0.500 each: with labels ('a', 'b') group a went to
        # 1.000 and b to 0.000; relabelled ('z', 'b'), on byte-identical scores,
        # b went to 1.000 and z to 0.000. A measured parity of 0.00 was turned
        # into a disparity of 1.00 and result_.group_adjustments published an
        # "unprivileged_group" no comparison had identified, with no warning.
        #
        # Two different situations, and only one of them is a refusal:
        #   * EVERY measurable rate identical: there is no gap at all, so there
        #     is nothing to correct and no group is worse off. Refuse. The 0.0
        #     disparity is still measured and still reported; it is a real zero
        #     over groups that were really scored, so nothing is thrown away.
        #   * SOME groups tied at the lowest rate, others above: the gap is real
        #     and must not be discarded. Every group at the lowest rate is
        #     treated as unprivileged, so no group tied for worst is pushed DOWN
        #     because of its name (measured before the fix: with a and b both at
        #     0.500 and c at 1.000, a went to 1.000 while b, equally worst off,
        #     went to 0.000).
        # A caller who NAMES unprivileged_group made the choice itself, so no
        # tie-break happens and that path is untouched.
        if self.not_assessable_reason_ is None and named_group is None:
            lowest = min(measurable.values())
            tied_at_lowest = sorted(str(g) for g, rate in measurable.items() if rate == lowest)
            distinct_rates = np.unique(np.asarray(list(measurable.values()), dtype=float)).size
            if distinct_rates < 2:
                self.not_assessable_reason_ = (
                    f"every measurable group {sorted(measurable)} has the same positive "
                    f"rate ({float(lowest):.6g}), so no group is worse off and there is "
                    f"no between-group gap to correct; naming one of them unprivileged "
                    f"would be a choice of group NAME order, not a measurement"
                )
                tied_at_lowest = []

        self.assessable_ = self.not_assessable_reason_ is None

        if not self.assessable_:
            self.detected_unprivileged_ = None
            self.detected_unprivileged_groups_ = []
            warnings.warn(
                f"RejectionOptionClassifier.fit(): {self.not_assessable_reason_}. No score "
                f"is adjusted and transform() returns the input unchanged. result_ carries "
                f"the disparity exactly as it was measured: NaN (could-not-check) when no "
                f"pair of rates could be compared at all, and a real 0.0 when every "
                f"measured rate is the same. Flipping rows toward an 'unprivileged' group "
                f"picked out of a single-entry dict, or out of a tie broken by group name, "
                f"would correct a gap nobody measured.",
                UserWarning,
                stacklevel=2,
            )
        elif named_group is None:
            self.detected_unprivileged_groups_ = tied_at_lowest
            self.detected_unprivileged_ = tied_at_lowest[0] if len(tied_at_lowest) == 1 else None
            if len(tied_at_lowest) > 1:
                warnings.warn(
                    f"RejectionOptionClassifier.fit(): group(s) {tied_at_lowest} share the "
                    f"lowest positive rate ({float(min(measurable.values())):.6g}), so the "
                    f"data does not single out one unprivileged group. All of them are "
                    f"treated as unprivileged and pushed up; detected_unprivileged_ is "
                    f"None, not the alphabetically first of them, because no single group "
                    f"was identified. Pass unprivileged_group=... to choose one yourself.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.detected_unprivileged_ = named_group
            self.detected_unprivileged_groups_ = [named_group]

        # Fitted parameters are set; mark fitted BEFORE the evaluation calls
        # below (predict/transform guard on is_fitted, so fit() itself crashed
        # on every call when the flag was only set at the end).
        self.is_fitted = True

        # One transform, reused for the decisions and for the modified count,
        # so a refusal is reported once rather than three times.
        y_prob_adjusted = self.transform(y_prob, sensitive_attr)
        y_pred_adjusted = _decide(
            y_prob_adjusted, self.threshold, f"{type(self).__name__}.predict()"
        )

        original_rates = _compute_group_positive_rates(y_prob, sensitive_attr, self.threshold)
        adjusted_rates = {}
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            group_decisions = np.asarray(y_pred_adjusted[mask], dtype=float)
            # A row with no decision is not a negative decision: exclude it and
            # leave a group with no decision at all as NaN, the same three
            # states _compute_group_positive_rates uses on the raw scores.
            decided = np.isfinite(group_decisions)
            adjusted_rates[str(group)] = (
                float(np.mean(group_decisions[decided])) if np.any(decided) else float("nan")
            )

        # NaN, not 0.0, when there was no between-group comparison to make
        # (BGL-S2, 2026-09-16). A bare max-min over a one-key dict is
        # structurally 0.0, which on this scale reads as perfect parity. A gap
        # between the groups that WERE scored is still reported, with the
        # unscored ones named (BGL-D-FINAL, 2026-09-17).
        original_disparity, adjusted_disparity, not_compared = _disparity_pair(
            f"{type(self).__name__}.fit()", original_rates, adjusted_rates
        )

        self.result_ = ReweightingResult(
            method=ReweightingMethod.REJECTION_OPTION,
            group_adjustments={
                "critical_region": self.critical_region_,
                # None whenever the data did not single one out: no group
                # assessable at all, or several tied at the lowest rate. The
                # list beside it is what transform() actually moved (G008).
                "unprivileged_group": self.detected_unprivileged_,
                "unprivileged_groups": list(self.detected_unprivileged_groups_),
            },
            original_metrics={"disparity": original_disparity},
            adjusted_metrics={"disparity": adjusted_disparity},
            fairness_improvement={"disparity_reduction": original_disparity - adjusted_disparity},
            calibration_impact={
                # Rows whose score the fit actually rewrote. This used to be
                # the count of rows merely sitting in the critical region,
                # which is a fact about the data and not about the fit: on a
                # refusal (no assessable group structure) nothing is moved and
                # that count would still report modifications (BGL-S2).
                "n_modified": _count_modified(y_prob, y_prob_adjusted),
                "n_in_critical_region": int(
                    (
                        (y_prob >= self.critical_region_[0]) & (y_prob <= self.critical_region_[1])
                    ).sum()
                ),
            },
            requested_constraint=self.constraint,
            honoured_constraint=self.honoured_constraint,
            groups_not_compared=not_compared,
            # On a refusal NO group was fitted, so every group present is
            # named with the reason rather than left to a bare NaN disparity.
            unfittable_groups=(
                {}
                if self.assessable_
                else {g: str(self.not_assessable_reason_) for g in sorted(group_rates)}
            ),
        )

        self.is_fitted = True
        return self

    def transform(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Transform probabilities using ROC strategy.

        For instances in the critical region:
        - Unprivileged group: shift probability above threshold
        - Privileged group: shift probability below threshold

        When fit() found no between-group comparison to make (fewer than two
        groups with a measurable positive rate) nothing is shifted: the input
        is returned unchanged and result_ carries NaN disparities, rather than
        rows being flipped toward a group that was named by `min()` over a
        one-entry dict (BGL-S2, 2026-09-16).
        """
        if not self.is_fitted:
            raise RuntimeError("ROC must be fitted before calling transform()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)

        if not self.assessable_:
            warnings.warn(
                f"RejectionOptionClassifier.transform(): {self.not_assessable_reason_}. No "
                f"row is moved and the scores are returned exactly as given. They are NOT "
                f"parity-corrected scores; result_ records the disparity as NaN.",
                UserWarning,
                stacklevel=2,
            )
            return np.asarray(y_prob).copy()

        y_prob_adjusted = np.asarray(y_prob, dtype=float).copy()
        groups_str = sensitive_attr.astype(str)

        # A group that was not present at fit time has no measured positive
        # rate, so it is neither known to be privileged nor known to be
        # unprivileged. `~unprivileged_mask` swept it into the privileged arm
        # and pushed it DOWN: measured, a classifier fitted on ['a', 'b'] moved
        # an unseen 'c' row from 0.55 to 0.49, flipping its decision to 0, with
        # no warning (G008, 2026-09-17). Those rows come back NaN.
        known = np.isin(groups_str, self.fitted_groups_)
        if not bool(np.all(known)):
            unseen: Dict[str, str] = {}
            for group in [str(g) for g in np.unique(groups_str[~known])]:
                mask = groups_str == group
                unseen[group] = (
                    f"group {group!r} carries {int(mask.sum())} row(s) here and was not "
                    f"present at fit time, so no positive rate was measured for it and it "
                    f"is neither known to be privileged nor known to be unprivileged"
                )
            y_prob_adjusted[~known] = float("nan")
            _warn_unfittable_groups(f"{type(self).__name__}.transform()", unseen)

        in_critical = (y_prob >= self.critical_region_[0]) & (y_prob <= self.critical_region_[1])

        # Adjust unprivileged group(s): push above threshold. Membership of the
        # fitted list, not equality with one name: on a tie at the lowest rate
        # that name was chosen by the alphabet and every other group tied for
        # worst was pushed DOWN (G008, 2026-09-17).
        unprivileged_mask = np.isin(groups_str, self.detected_unprivileged_groups_)
        modify_up = known & in_critical & unprivileged_mask & (y_prob < self.threshold)
        y_prob_adjusted[modify_up] = self.threshold + 0.01

        # Adjust privileged group: push below threshold
        modify_down = known & in_critical & ~unprivileged_mask & (y_prob >= self.threshold)
        y_prob_adjusted[modify_down] = self.threshold - 0.01

        return y_prob_adjusted

    def predict(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Make fair predictions using ROC strategy.

        Returns an int array of 0/1 decisions when every row carries a finite
        adjusted score, and a FLOAT array of 0.0 / 1.0 / NaN with a
        UserWarning when any row does not. transform() honestly passes a NaN
        score through; `np.nan >= threshold` is False, so this line used to
        turn each of them into a hard int 0, a confident denial for a row the
        model never scored (BGL-S2, 2026-09-16).
        """
        if threshold is None:
            threshold = self.threshold

        y_prob_adjusted = self.transform(y_prob, sensitive_attr)
        return _decide(y_prob_adjusted, threshold, f"{type(self).__name__}.predict()")


class CalibratedEqualizer(BaseReweighter):
    """
    Quantile-matching probability equalizer.

    This method maps each group's probability distribution onto the pooled
    distribution by quantile matching, so the groups end up with equal
    probability distributions. It does not measure, model or preserve
    calibration: the only calibration figure it reports is the mean
    absolute shift it applied (`result_.calibration_impact
    ['distribution_shift']`). The class name predates that honesty.

    Example:
        >>> equalizer = CalibratedEqualizer()
        >>> equalizer.fit(y_true, y_prob, gender)
        >>> y_prob_equal = equalizer.transform(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: calibrated_equalization. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        min_group_size: int = _DEFAULT_MIN_GROUP_SIZE,
    ):
        """
        Initialize the equalizer.

        Args:
            n_quantiles: Number of quantiles for distribution matching. At least
                1 is required, because a grid of one percentile maps every score
                onto a single number and the 0.0 spread that follows is then
                published as a disparity this method removed. See
                :func:`_reject_degenerate_quantile_grid`.
            min_group_size: Fewest scored rows a group's percentile grid may be
                estimated from. Below it the group is not mapped and its rows
                come back as NaN, because np.percentile returns a grid for any
                non-empty input and np.interp against a constant grid returns
                numpy's tie-break, not a mapping: measured, a single minority
                row scoring 0.05 came out as the pooled maximum 0.947 while the
                run reported a 0.26 "disparity reduction" (BGL-S2,
                2026-09-16). The default 30 matches min_group_size across the
                evaluation and calibration modules.

        A `preserve_calibration` flag used to be accepted here (F13, removed
        2026-09-09). It was stored and never read: True and False produced
        byte-identical output, and nothing in the mapping preserves
        calibration, so the flag promised a property the class does not have.
        """
        super().__init__(constraint="demographic_parity")
        self.n_quantiles = _reject_degenerate_quantile_grid(
            n_quantiles, type(self).__name__, "n_quantiles"
        )
        self.min_group_size = min_group_size
        self.quantile_maps_: Dict[str, np.ndarray] = {}
        self.target_quantiles_: np.ndarray = np.array([])
        # Group -> why no quantile map could be fitted for it. Read by
        # transform(), which returns NaN for those rows instead of the
        # caller's own input value (BGL-S2).
        self.unfittable_groups_: Dict[str, str] = {}

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "CalibratedEqualizer":
        """
        Fit the equalizer by computing quantile mappings.

        sample_weight is not honoured (the quantile maps have no weighted
        form here); a non-uniform vector is refused, see BaseReweighter.fit.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)
        _refuse_unimplemented_sample_weight(sample_weight, len(y_true), type(self).__name__)

        groups_str = sensitive_attr.astype(str)
        unique_groups = [str(g) for g in np.unique(sensitive_attr)]
        # Checked HERE as well as in __init__, because this is where the grid is
        # built and n_quantiles is a plain attribute a caller can rewrite after
        # construction.
        quantile_points = np.linspace(
            0,
            100,
            _reject_degenerate_quantile_grid(
                self.n_quantiles, "CalibratedEqualizer.fit()", "n_quantiles"
            )
            + 1,
        )

        # A refit must not inherit the previous fit's maps. quantile_maps_ was
        # only ever written into, so a group present in an EARLIER dataset kept
        # its old grid and transform() went on applying it: measured, after a
        # fit on ['a', 'b'] and a refit on an a-only frame, 'b' was still in
        # quantile_maps_, byte-identical to the previous dataset's, and
        # transform rewrote real 'b' scores with it (0.394 -> 0.224) while
        # result_.group_adjustments, which IS rebuilt per fit, looked clean
        # (BGL-S2, 2026-09-16).
        self.quantile_maps_ = {}
        self.unfittable_groups_ = {}
        self.target_quantiles_ = np.array([])

        pooled_finite = _finite_mask(y_prob)
        pooled_refusal = _quantile_grid_refusal(
            "the pooled sample (the target distribution every group is mapped onto)",
            y_prob,
            min(self.min_group_size, len(y_prob)),
        )
        if pooled_refusal is None:
            self.target_quantiles_ = np.percentile(
                np.asarray(y_prob, dtype=float)[pooled_finite], quantile_points
            )

        for group in unique_groups:
            mask = groups_str == group
            reason = pooled_refusal or _quantile_grid_refusal(
                f"group {group!r}", y_prob[mask], self.min_group_size
            )
            if reason is not None:
                self.unfittable_groups_[group] = reason
                continue
            finite = _finite_mask(y_prob[mask])
            self.quantile_maps_[group] = np.percentile(
                np.asarray(y_prob[mask], dtype=float)[finite], quantile_points
            )

        _warn_unfittable_groups("CalibratedEqualizer.fit()", self.unfittable_groups_)
        if len(unique_groups) < 2:
            _warn_no_between_group_comparison("CalibratedEqualizer.fit()", unique_groups)

        # Quantile maps are set; mark fitted BEFORE the evaluation transform
        # (transform guards on is_fitted, so fit() itself crashed here).
        self.is_fitted = True

        y_prob_adjusted = self.transform(y_prob, sensitive_attr)

        original_means = _compute_group_mean_probs(y_prob, sensitive_attr)
        adjusted_means = _compute_group_mean_probs(y_prob_adjusted, sensitive_attr)

        # NaN, not 0.0, when there was no between-group comparison to make at
        # all (BGL-S2). When only SOME groups are unmeasurable the spread over
        # the measurable ones is reported and the rest are named
        # (BGL-D-FINAL, 2026-09-17).
        original_disparity, adjusted_disparity, not_compared = _disparity_pair(
            "CalibratedEqualizer.fit()", original_means, adjusted_means
        )

        self.result_ = ReweightingResult(
            method=ReweightingMethod.CALIBRATED_EQUALIZATION,
            # Built from the maps that were actually FITTED, not from the groups
            # that happened to be present (BGL-D-FINAL, 2026-09-17). Claiming
            # "quantile_mapping" for every unique group announced a mapping for
            # groups sitting in unfittable_groups_, whose rows transform()
            # returns as NaN: measured, an A/B/C frame whose one-row C was
            # unscored published {'A': 'quantile_mapping', 'B': ...,
            # 'C': 'quantile_mapping'} while every C row came back NaN.
            group_adjustments={
                g: (
                    "quantile_mapping"
                    if g in self.quantile_maps_
                    else "NOT FITTED (rows return NaN; see unfittable_groups)"
                )
                for g in unique_groups
            },
            original_metrics={"mean_disparity": original_disparity},
            adjusted_metrics={"mean_disparity": adjusted_disparity},
            fairness_improvement={"disparity_reduction": original_disparity - adjusted_disparity},
            calibration_impact={
                "distribution_shift": float(np.mean(np.abs(y_prob_adjusted - y_prob)))
            },
            requested_constraint=self.constraint,
            honoured_constraint=self.honoured_constraint,
            groups_not_compared=not_compared,
            unfittable_groups=dict(self.unfittable_groups_),
        )

        self.is_fitted = True
        return self

    def transform(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Transform probabilities using quantile mapping.

        The loop runs over the groups PRESENT in `sensitive_attr`, not over the
        fitted maps. Iterating over `quantile_maps_` meant a row whose group
        was never fitted was silently left at its own input value and returned
        in the same array as the equalized ones: measured, transform(p,
        ['c'] * 5) on an equalizer fitted for 'a' and 'b' returned an array
        byte-identical to np.clip(input, 0, 1) with no warning, and the same
        four scores decided as fitted 'b' gave [0, 1, 1, 1] against [0, 0, 0, 1]
        as unseen 'c' (BGL-S2, 2026-09-16). An unmapped row now comes back as
        NaN, which no caller can mistake for an adjusted score.
        """
        if not self.is_fitted:
            raise RuntimeError("Equalizer must be fitted before calling transform()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)

        groups_str = sensitive_attr.astype(str)
        y_prob_adjusted = np.asarray(y_prob, dtype=float).copy()

        unmapped: Dict[str, str] = {}
        for group in [str(g) for g in np.unique(groups_str)]:
            mask = groups_str == group
            group_quantiles = self.quantile_maps_.get(group)
            if group_quantiles is None:
                unmapped[group] = self.unfittable_groups_.get(
                    group,
                    f"group {group!r} carries {int(mask.sum())} row(s) here and was not "
                    f"present at fit time, so no quantile map exists for it",
                )
                y_prob_adjusted[mask] = float("nan")
                continue

            # Map from group distribution to target distribution
            group_probs = y_prob_adjusted[mask]
            adjusted = np.interp(group_probs, group_quantiles, self.target_quantiles_)
            y_prob_adjusted[mask] = adjusted

        _warn_unfittable_groups("CalibratedEqualizer.transform()", unmapped)

        return np.clip(y_prob_adjusted, 0, 1)


class DistributionMatcher(BaseReweighter):
    """
    Distribution matching reweighter.

    Transforms probability distributions to match a target distribution
    (either a reference group or a specified distribution).

    Example:
        >>> matcher = DistributionMatcher(reference_group='male')
        >>> matcher.fit(y_true, y_prob, gender)
        >>> y_prob_matched = matcher.transform(y_prob, gender)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: distribution_matching. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        reference_group: Optional[str] = None,
        method: str = "quantile",
        min_group_size: int = _DEFAULT_MIN_GROUP_SIZE,
    ):
        """
        Initialize the distribution matcher.

        Args:
            reference_group: Group whose distribution to match (default: majority)
            min_group_size: Fewest scored rows a group's percentile grid may be
                estimated from. Below it the group is not matched and its rows
                come back as NaN rather than being mapped through a constant
                grid: measured, a 199 + 1 split reported a disparity_reduction
                of -0.407, i.e. the mitigation recorded itself as having made
                the gap about nine times WORSE, silently, off a grid built from
                one repeated value (BGL-S2, 2026-09-16). The default 30 matches
                min_group_size across the evaluation and calibration modules.
            method: Matching method. Only 'quantile' (np.interp between the
                group's and the reference group's percentile grids) is
                implemented. 'histogram' was documented but never built and
                is refused, as is any other name.
        """
        super().__init__(constraint="demographic_parity")
        self.reference_group = reference_group
        # What the CALLER asked for, kept separate from the fitted value.
        # reference_group was overwritten by the first fit, so a later fit on a
        # dataset without that group went on matching against a group that is
        # no longer there (BGL-S2, 2026-09-16).
        self._requested_reference_group = reference_group
        self.min_group_size = min_group_size
        self.unfittable_groups_: Dict[str, str] = {}
        # F6 (2026-09-09): `method` was stored and never read; transform()
        # always ran the quantile mapping. Measured: 'quantile', 'histogram'
        # and 'totally_bogus' produced byte-identical output. A documented
        # option that changes nothing is refused rather than accepted.
        if method == "histogram":
            raise NotImplementedError(
                "DistributionMatcher method='histogram' is not implemented; "
                "only method='quantile' exists."
            )
        if method != "quantile":
            raise ValueError(
                f"Unknown DistributionMatcher method {method!r}; only 'quantile' is implemented."
            )
        self.method = method
        self.reference_distribution_: np.ndarray = np.array([])
        self.group_distributions_: Dict[str, np.ndarray] = {}
        self.n_bins = 100

    def fit(
        self,
        *,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "DistributionMatcher":
        """
        Fit the distribution matcher.

        sample_weight is not honoured (the percentile grids have no weighted
        form here); a non-uniform vector is refused, see BaseReweighter.fit.
        """
        # Refuse a reversed call before the int cast hides it. See the
        # helper's docstring for the measured production consequence.
        reject_swapped_labels_and_scores(y_true, y_prob, caller=type(self).__name__)
        y_true = coerce_to_array(y_true).astype(int)
        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true, y_prob, sensitive_attr)
        _refuse_unimplemented_sample_weight(sample_weight, len(y_true), type(self).__name__)

        groups_str = sensitive_attr.astype(str)
        unique_groups = [str(g) for g in np.unique(sensitive_attr)]

        # Recomputed on EVERY fit, from the caller's original argument, so a
        # refit cannot keep matching against the previous dataset's majority.
        if self._requested_reference_group is None:
            group_sizes = {g: (groups_str == g).sum() for g in unique_groups}
            self.reference_group = max(group_sizes.keys(), key=lambda g: group_sizes[g])
        else:
            self.reference_group = self._requested_reference_group
            if str(self.reference_group) not in unique_groups:
                raise ValueError(
                    f"reference_group={self.reference_group!r} does not appear in "
                    f"sensitive_attr (groups: {unique_groups}); there is no distribution "
                    f"to match against."
                )

        # A refit must not inherit the previous fit's grids: group_distributions_
        # was only ever written into (BGL-S2).
        self.group_distributions_ = {}
        self.unfittable_groups_ = {}
        self.reference_distribution_ = np.array([])

        ref_mask = groups_str == str(self.reference_group)
        # Same refusal as CalibratedEqualizer: with self.n_bins set to 0 the grid
        # is one percentile and np.interp collapses the group onto a single
        # value. Measured on the same 120 rows, that published a
        # disparity_reduction of 0.3725 for a group whose 60 scores had all
        # become one number (BGL3, 2026-09-27).
        quantile_points = np.linspace(
            0,
            100,
            _reject_degenerate_quantile_grid(self.n_bins, "DistributionMatcher.fit()", "n_bins")
            + 1,
        )
        ref_refusal = _quantile_grid_refusal(
            f"the reference group {str(self.reference_group)!r}",
            y_prob[ref_mask],
            self.min_group_size,
        )
        if ref_refusal is None:
            ref_finite = _finite_mask(y_prob[ref_mask])
            self.reference_distribution_ = np.percentile(
                np.asarray(y_prob[ref_mask], dtype=float)[ref_finite], quantile_points
            )

        for group in unique_groups:
            mask = groups_str == group
            reason = ref_refusal or _quantile_grid_refusal(
                f"group {group!r}", y_prob[mask], self.min_group_size
            )
            if reason is not None:
                self.unfittable_groups_[group] = reason
                continue
            finite = _finite_mask(y_prob[mask])
            self.group_distributions_[group] = np.percentile(
                np.asarray(y_prob[mask], dtype=float)[finite], quantile_points
            )

        _warn_unfittable_groups("DistributionMatcher.fit()", self.unfittable_groups_)
        if len(unique_groups) < 2:
            _warn_no_between_group_comparison("DistributionMatcher.fit()", unique_groups)

        # Distributions are set; mark fitted BEFORE the evaluation transform
        # (transform guards on is_fitted, so fit() itself crashed here).
        self.is_fitted = True

        y_prob_adjusted = self.transform(y_prob, sensitive_attr)

        original_means = _compute_group_mean_probs(y_prob, sensitive_attr)
        adjusted_means = _compute_group_mean_probs(y_prob_adjusted, sensitive_attr)

        # NaN, not 0.0, when there was no between-group comparison to make at
        # all (BGL-S2). When only SOME groups are unmeasurable the spread over
        # the measurable ones is reported and the rest are named
        # (BGL-D-FINAL, 2026-09-17).
        original_disparity, adjusted_disparity, not_compared = _disparity_pair(
            "DistributionMatcher.fit()", original_means, adjusted_means
        )

        self.result_ = ReweightingResult(
            method=ReweightingMethod.DISTRIBUTION_MATCHING,
            group_adjustments={"reference_group": self.reference_group},
            original_metrics={"mean_disparity": original_disparity},
            adjusted_metrics={"mean_disparity": adjusted_disparity},
            fairness_improvement={"disparity_reduction": original_disparity - adjusted_disparity},
            calibration_impact={"mean_shift": float(np.mean(np.abs(y_prob_adjusted - y_prob)))},
            requested_constraint=self.constraint,
            honoured_constraint=self.honoured_constraint,
            groups_not_compared=not_compared,
            unfittable_groups=dict(self.unfittable_groups_),
        )

        self.is_fitted = True
        return self

    def transform(
        self,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Transform probabilities to match reference distribution.

        The loop runs over the groups PRESENT in `sensitive_attr`. The
        reference group is left at its own scores by construction: it IS the
        target distribution. Any other group with no usable grid (too few
        scored rows, a constant grid, or absent at fit time) comes back as NaN
        rather than as its own input value, which would be indistinguishable
        from a matched score (BGL-S2, 2026-09-16).
        """
        if not self.is_fitted:
            raise RuntimeError("Matcher must be fitted before calling transform()")

        y_prob = coerce_to_array(y_prob)
        sensitive_attr = coerce_to_array(sensitive_attr)

        groups_str = sensitive_attr.astype(str)
        y_prob_adjusted = np.asarray(y_prob, dtype=float).copy()

        unmapped: Dict[str, str] = {}
        for group in [str(g) for g in np.unique(groups_str)]:
            mask = groups_str == group
            if group == str(self.reference_group):
                continue

            group_dist = self.group_distributions_.get(group)
            if group_dist is None:
                unmapped[group] = self.unfittable_groups_.get(
                    group,
                    f"group {group!r} carries {int(mask.sum())} row(s) here and was not "
                    f"present at fit time, so no distribution was estimated for it",
                )
                y_prob_adjusted[mask] = float("nan")
                continue

            # Map from group distribution to reference distribution
            adjusted = np.interp(y_prob_adjusted[mask], group_dist, self.reference_distribution_)
            y_prob_adjusted[mask] = adjusted

        _warn_unfittable_groups("DistributionMatcher.transform()", unmapped)

        return np.clip(y_prob_adjusted, 0, 1)


def create_reweighter(
    method: str = "multiplicative", constraint: str = "demographic_parity", **kwargs
) -> BaseReweighter:
    """
    Factory function to create reweighters.

    Args:
        method: Reweighting method ('multiplicative', 'additive', 'rejection_option',
                'distribution_matching', 'calibrated_equalization')
        constraint: Fairness constraint the caller is asking for. Only
            'demographic_parity' is enforced by any of these classes;
            anything else warns and is recorded as requested but not honoured
            (F21). 'rejection_option', 'distribution_matching' and
            'calibrated_equalization' do not take a constraint at all, so the
            value is dropped for them, which warns here rather than passing
            silently.
        **kwargs: Additional arguments passed to the reweighter

    Returns:
        Configured reweighter

    Example:
        >>> reweighter = create_reweighter(
        ...     method='rejection_option',
        ...     theta=0.15
        ... )

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

    Ledger row: create_reweighter. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    reweighters = {
        "multiplicative": lambda: PredictionReweighter(
            constraint=constraint, method="multiplicative", **kwargs
        ),
        "additive": lambda: PredictionReweighter(
            constraint=constraint, method="additive", **kwargs
        ),
        "rejection_option": lambda: RejectionOptionClassifier(**kwargs),
        "roc": lambda: RejectionOptionClassifier(**kwargs),
        "distribution_matching": lambda: DistributionMatcher(**kwargs),
        "calibrated_equalization": lambda: CalibratedEqualizer(**kwargs),
        "calibrated": lambda: CalibratedEqualizer(**kwargs),
    }

    if method.lower() not in reweighters:
        raise ValueError(f"Unknown method: {method}. Available: {list(reweighters.keys())}")

    reweighter = reweighters[method.lower()]()
    # The three classes that take no constraint drop it here. Warn on that
    # drop instead of returning an object whose constraint silently differs
    # from the one asked for (F21). PredictionReweighter already warned in
    # its own constructor, and its self.constraint equals `constraint`, so
    # this does not fire twice.
    if reweighter.constraint != constraint:
        _warn_constraint_not_honoured(
            constraint, f"create_reweighter(method={method!r})", stacklevel=2
        )
    return reweighter
