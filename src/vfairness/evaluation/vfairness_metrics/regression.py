"""
Regression fairness metrics for vfairness.

This module provides fairness metrics for regression tasks.

Library Comparisons:
    Note: Regression fairness is less common in fairness libraries.
    Most libraries focus on classification. These implementations
    extend fairness concepts to continuous predictions.

    - AI Fairness 360: Limited regression support
    - Fairlearn: fairlearn.metrics supports custom metrics
    - Aequitas: Classification-focused

Statistical Validation:
    All metrics support confidence interval computation via the
    `*_with_ci` function variants.
"""

import warnings
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from ._grouping import GroupManager, compute_max_difference
from ._statistics import (
    _CONSTANT_SPREAD_REL_TOL,
    RECOMMENDED_BOOTSTRAP_SAMPLES,
    StatisticalResult,
    bootstrap_over_index,
    cohens_d,
    compute_metric_with_ci,
)
from ._validation import ArrayLike, MissingStrategy, validate_inputs
from .classification import _warn_dropped_groups

# ---------------------------------------------------------------------------
# CONVENTION (disclose the drop): every metric below computes over
# gm.get_valid_groups(), so a group under min_group_size is removed from the
# comparison. classification.py and ranking.py both announce that removal with
# _warn_dropped_groups; this module did not, at any of its eight GroupManager
# sites, so the regression path dropped a protected group with NO SIGNAL AT ALL.
#
# The incident: 2302 rows where a 38-level tail (1102 rows, 47.9 percent of the
# data) is predicted at 3x its true value. Measured 2026-09-10, before this
# import: the tail's MAE is 99,728 against 398 for the two surviving groups, a
# factor of 251, and mae_parity_difference returned 13.96, rmse_parity_difference
# 22.49, mean_prediction_difference 364.50, r2_parity_difference 0.00, each with
# ZERO warnings, and regression_fairness_report scored 4/4 metrics within
# thresholds. The identical group shape on the classification path warned once.
# The warner is IMPORTED rather than copied so the two paths cannot drift.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R² HAS NO DENOMINATOR WHEN THE TARGET IS CONSTANT, AND "CONSTANT" IS NOT AN
# EXACT TEST. FINITENESS IS NOT MEASURABILITY.
#
# B4 tier-1 audit, 2026-09-30. Both R² sites in this module decided
# measurability by comparing an ACCUMULATED sum of squares against zero exactly:
# `if ss_tot == 0` in r2_parity_difference's compute_r2, and `if ss_tot > 0` in
# get_group_metrics. A target that is constant only up to the resolution of the
# arithmetic that produced it has an ss_tot that is minute and NOT zero, so both
# sites divided by it and published a finite number that every isfinite guard
# downstream passes and every parity comparison then treats as a score.
#
# Measured before this change on a target computed as `(base + 100.1) - base`
# with `base = np.linspace(1e6, 1e8, n)`, i.e. 100.1 carrying 7.5e-9 of rounding
# inherited from 1e8 arithmetic, predicted at a flat 130, beside a healthy second
# group, at n = 31, 40, 57 and 100:
#   before: get_group_metrics r2 = -8.02e+19 for that group with NO warning
#           naming it, and r2_parity_difference = 8.02e+19 published as a parity
#           gap with zero warnings
#   after:  nan for the group and nan for the parity difference, each with a
#           warning naming the group and which quantity is degenerate
#
# The base must SPAN BINADES for the noise to exist at all: np.spacing is
# constant inside one binade, so `(base + c) - base` rounds identically for every
# element and the spread comes out exactly 0.0, which the old exact test already
# caught. That is why an earlier fixture built from `1e6 + arange(n)` failed to
# reproduce this and looked like a clean result.
#
# TWO AXES, because either one alone leaves the other door open:
#  1. the target's OWN relative resolution, which catches noise at the target's
#     magnitude, and
#  2. the MAGNITUDE OF THE RESULTING R², because when the noise was inherited
#     from arithmetic at a much larger magnitude, as above, no tolerance derived
#     from the target alone can recognise it. This is the same shape as
#     _DEGENERATE_DENOMINATOR_D in compute_regression_effect_sizes, and it is set
#     far beyond anything a real model reaches: |R²| = 1e6 means the squared
#     error is a million times the target's own variation, i.e. a thousand target
#     standard deviations of error. A genuinely terrible model sits at -1 to -100.
#
# ONE HELPER, TWO CALLERS, deliberately: the two sites had already drifted to two
# different spellings of the same exact test, so the predicate lives in one place
# and neither site owns a copy of it.
# ---------------------------------------------------------------------------

#: Ulps of the target's own magnitude below which its peak-to-peak spread is
#: rounding noise rather than variation. Same bound, and same reasoning, as the
#: operand tolerance in compute_regression_effect_sizes.
_R2_CONSTANT_ULPS = 4.0

#: |R²| at or above which the denominator is resolution rather than variation.
_DEGENERATE_R2_MAGNITUDE = 1e6


def _r2_or_not_measured(yt: np.ndarray, yp: np.ndarray) -> Tuple[float, Optional[str]]:
    """R² for one group, or NaN plus the reason it could not be measured.

    Returns ``(r2, None)`` when R² is a measurement, and ``(nan, reason)`` when it
    is not. The reason is a clause a caller puts in a warning, so the
    could-not-check state reaches a reader instead of living in a return value.
    Never returns 0.0 for an unmeasurable group: 0.0 is a REAL score on this scale
    ("predicts exactly as well as the group mean").
    """
    if len(yt) == 0:
        return float("nan"), "the group is empty, so there is no target to explain"
    scale = float(np.max(np.abs(yt)))
    spread = float(np.ptp(yt))
    # np.ptp is exact and needs no epsilon; the bound it is compared against is
    # data-scaled, so it carries no units. np.spacing(0.0) is denormal-small, so an
    # all-zero target is still caught by `spread <= atol` with atol about zero,
    # which is the one case where exact is what the data says.
    atol = max(
        _R2_CONSTANT_ULPS * float(np.spacing(scale)),
        _CONSTANT_SPREAD_REL_TOL * scale,
    )
    if spread <= atol:
        return float("nan"), (
            f"the target is constant (peak-to-peak {spread:.3g}, at or below the "
            f"{atol:.3g} that is rounding noise at this magnitude), so there is no "
            f"variance to explain"
        )
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if not ss_tot > 0:
        return float("nan"), "the target's sum of squares is zero, so R² has no denominator"
    r2 = 1.0 - (ss_res / ss_tot)
    if not np.isfinite(r2):
        # RETURNED UNCHANGED, DELIBERATELY. A non-finite R² is already three-stated
        # one layer down: _report_for_grading / _restate_ungradeable_cards refuse it
        # at the grading surface and the card says "the value is infinite", which is
        # strictly more than a nan here could say, and the report keeps the value it
        # refused. Collapsing inf onto nan HERE destroyed that: it turned the card's
        # reason into "this metric was never computed" and changed the metric field
        # the report publishes, which tests/test_bgl5_evaluation_1.py pins as the
        # property that the refusal must not alter the measured metrics.
        return r2, None
    if abs(r2) >= _DEGENERATE_R2_MAGNITUDE:
        return float("nan"), (
            f"|R²| would be {abs(r2):.3g}, i.e. the squared error is that many times "
            f"the target's own variation, so the denominator is the resolution of the "
            f"numbers and not variation in them"
        )
    return r2, None


def mae_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute MAE parity difference across groups.

    Measures the maximum absolute difference in Mean Absolute Error
    between groups. Lower values indicate more equitable error distribution.

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: mean_absolute_error with MetricFrame
        Aequitas: Not supported (classification-focused)

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in MAE across groups; NaN (insufficient
        evidence) when fewer than two groups meet ``min_group_size``.

    Example:
        >>> mae_diff = mae_parity_difference(y_true, y_pred, region)
        >>> print(f"MAE Parity Difference: {mae_diff:.3f}")

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

    Ledger row: mae_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    errors = np.abs(y_true - y_pred)

    mae_values = gm.compute_group_statistic(errors, "mean")

    # Fewer than two qualifying groups: no pair to compare, so the between-group
    # error gap is UNMEASURABLE -> NaN, never a 0.0 that reads as equal error for
    # everyone. Mirrors the classification convention (see the CONVENTION note in
    # classification.py); the report routes NaN to NOT_ASSESSABLE.
    if len(mae_values) < 2:
        return float("nan")

    max_diff, _, _ = compute_max_difference(mae_values)
    return max_diff


def rmse_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute RMSE parity difference across groups.

    Measures the maximum absolute difference in Root Mean Squared Error
    between groups. Penalizes large errors more than MAE.

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: mean_squared_error with MetricFrame (then sqrt)
        Aequitas: Not supported

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in RMSE across groups; NaN (insufficient
        evidence) when fewer than two groups meet ``min_group_size``.

    Example:
        >>> rmse_diff = rmse_parity_difference(y_true, y_pred, income_bracket)
        >>> print(f"RMSE Parity Difference: {rmse_diff:.3f}")

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

    Ledger row: rmse_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # Compute RMSE per group
    rmse_values = {}
    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        group_errors = (y_true[mask] - y_pred[mask]) ** 2
        rmse_values[name] = np.sqrt(np.mean(group_errors))

    # Fewer than two qualifying groups: unmeasurable -> NaN, never a false-parity
    # 0.0 (see the CONVENTION note in classification.py).
    if len(rmse_values) < 2:
        return float("nan")

    max_diff, _, _ = compute_max_difference(rmse_values)
    return max_diff


def mean_prediction_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute mean prediction difference across groups.

    Measures the maximum absolute difference in average predictions
    between groups. Indicates systematic prediction bias.

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: Can compute via MetricFrame with custom function
        Aequitas: Not supported

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in mean predictions across groups; NaN
        (insufficient evidence) when fewer than two groups meet
        ``min_group_size``.

    Example:
        >>> mpd = mean_prediction_difference(y_true, y_pred, gender)
        >>> print(f"Mean Prediction Difference: {mpd:.3f}")

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

    Ledger row: mean_prediction_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    mean_preds = gm.compute_group_statistic(y_pred, "mean")

    # Fewer than two qualifying groups: unmeasurable -> NaN, never a false-parity
    # 0.0 (see the CONVENTION note in classification.py).
    if len(mean_preds) < 2:
        return float("nan")

    max_diff, _, _ = compute_max_difference(mean_preds)
    return max_diff


def r2_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute R² parity difference across groups.

    Measures the maximum absolute difference in R² scores between groups.
    Indicates whether model explains variance equally across groups.

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: r2_score with MetricFrame
        Aequitas: Not supported

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in R² across groups; NaN (insufficient
        evidence) when fewer than two groups meet ``min_group_size`` or any
        qualifying group has an undefined R².

    Example:
        >>> r2_diff = r2_parity_difference(y_true, y_pred, ethnicity)
        >>> print(f"R² Parity Difference: {r2_diff:.3f}")

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

    Ledger row: r2_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # Constant y_true: there is no variance to explain, so R² is UNDEFINED. NaN
    # (not 0.0, which would read as a real, poor score); the caller surfaces an
    # undefined group as an explicit NaN, mirroring the undefined-TPR handling in
    # classification. The predicate is _r2_or_not_measured, shared with
    # get_group_metrics, because the exact `ss_tot == 0` test that used to live
    # here published -8.0e19 as a parity gap for a target that is constant up to
    # rounding. See the block above _R2_CONSTANT_ULPS.
    r2_values = {}
    r2_reasons = {}
    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        r2_values[name], _reason = _r2_or_not_measured(y_true[mask], y_pred[mask])
        if _reason is not None:
            r2_reasons[name] = _reason

    # Fewer than two qualifying groups: unmeasurable -> NaN, never a false-parity
    # 0.0 (see the CONVENTION note in classification.py).
    if len(r2_values) < 2:
        return float("nan")

    # A group with constant y_true has an UNDEFINED R² (NaN).
    # compute_max_difference skips NaN pairs, so this would otherwise silently
    # report a spread as if the group were fine, exactly when it could not be
    # assessed. Mirror the undefined-TPR handling: warn and return NaN.
    undefined = [g for g, r in r2_values.items() if np.isnan(r)]
    if undefined:
        # The reason is named per group rather than assumed to be "constant
        # y_true": since 2026-09-30 a group can also fail because the R² its own
        # numbers produce is a degenerate magnitude, and a reader who is told the
        # wrong reason cannot check it.
        detail = "; ".join(f"{g}: {r2_reasons.get(g, 'no reason recorded')}" for g in undefined)
        warnings.warn(
            f"R² is undefined for group(s) {undefined} ({detail}). "
            f"r2_parity_difference cannot be computed across all groups; returning "
            f"NaN, which means NOTHING WAS COMPARED and not that the groups agree.",
            UserWarning,
        )
        return float("nan")

    max_diff, _, _ = compute_max_difference(r2_values)
    return max_diff


def residual_bias(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, float]:
    """
    Compute mean residual (bias) for each group.

    Positive values indicate systematic underprediction,
    negative values indicate overprediction.

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: No direct equivalent
        This is a unique contribution of vfairness

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Dict mapping group names to mean residual

    Example:
        >>> bias = residual_bias(y_true, y_pred, region)
        >>> for group, b in bias.items():
        ...     direction = "underpredicted" if b > 0 else "overpredicted"
        ...     print(f"{group}: {direction} by {abs(b):.3f}")

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

    Ledger row: residual_bias. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    residuals = y_true - y_pred

    return gm.compute_group_statistic(residuals, "mean")


def get_group_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed regression metrics for each group.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Dict mapping group names to metric dictionaries. ``r2`` is NaN (could
        not be measured) for a group whose ``y_true`` is constant, because R²
        divides by that group's target variance.

    Example:
        >>> metrics = get_group_metrics(y_true, y_pred, region)
        >>> for group, m in metrics.items():
        ...     print(f"{group}: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}")
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    results = {}
    # Groups whose R² has no denominator, collected so the warning can name them
    # once rather than per group.
    undefined_r2 = []
    # The REASON per group, kept beside the names rather than folded into them: the
    # group list has to stay greppable as `['A']`, which is the shape every other
    # "<metric> is undefined for group(s)" warning in this library uses and which a
    # pin matches on.
    undefined_r2_reasons = {}

    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        yt = y_true[mask]
        yp = y_pred[mask]

        residuals = yt - yp
        abs_errors = np.abs(residuals)
        sq_errors = residuals**2

        # Constant y_true: the group has no target variance, so R² is UNDEFINED,
        # and 0.0 is a REAL score on this scale ("predicts exactly as well as the
        # group mean"). Measured 2026-09-27 on 40 rows of constant y_true=100
        # predicted at a flat 130 (MAE 30.0) beside 40 varying rows: this function
        # published r2 0.0 for the unmeasured group and -0.679 for the measured
        # one, so the group nobody could score read as the BETTER of the two, and
        # the number reached the published report under ``group_stats``.
        # r2_parity_difference already returns NaN and warns on the identical
        # shape, and classification.get_group_metrics already uses np.nan for an
        # undefined tpr/fpr/precision; this site was the one that substituted.
        # B4 tier-1 audit, 2026-09-30: `if ss_tot > 0` is an EXACT test on an
        # ACCUMULATED sum of squares, and a target that is constant only up to the
        # resolution of the arithmetic that produced it walked straight past it and
        # published r2 = -8.02e+19 here, finite, with no warning naming the group.
        # The predicate is now shared with r2_parity_difference so the two sites
        # cannot carry two different answers to the same question again. See the
        # block above _R2_CONSTANT_ULPS.
        r2, r2_reason = _r2_or_not_measured(yt, yp)
        if r2_reason is not None:
            undefined_r2.append(name)
            undefined_r2_reasons[name] = r2_reason

        results[name] = {
            "size": len(yt),
            "mean_true": np.mean(yt),
            "mean_pred": np.mean(yp),
            "mae": np.mean(abs_errors),
            "rmse": np.sqrt(np.mean(sq_errors)),
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "r2": r2,
        }

    if undefined_r2:
        r2_detail = "; ".join(
            f"{g}: {undefined_r2_reasons.get(g, 'no reason recorded')}" for g in undefined_r2
        )
        warnings.warn(
            f"get_group_metrics: R² is undefined for group(s) {undefined_r2} "
            f"({r2_detail}). Their 'r2' is NaN (could not be measured), not 0.0 and "
            f"not a finite score. Every other metric in those rows (mae, rmse, "
            f"mean_residual) IS measured.",
            UserWarning,
            stacklevel=2,
        )

    return results


# Statistical Validation Functions


def mae_parity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute MAE parity difference with confidence interval.

    Uses stratified bootstrap resampling to maintain group proportions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples (≥5000 recommended)
        confidence_level: Confidence level (e.g., 0.95)
        method: 'auto' (select based on sample size), 'bootstrap', or 'bayesian'
        random_state: Random seed for reproducibility

    Returns:
        StatisticalResult with point estimate and confidence interval

    Example:
        >>> result = mae_parity_difference_with_ci(
        ...     y_true, y_pred, region, n_bootstrap=5000
        ... )
        >>> print(f"MAE Diff: {result.point_estimate:.3f} "
        ...       f"95% CI [{result.lower_bound:.3f}, {result.upper_bound:.3f}]")
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    return compute_metric_with_ci(
        mae_parity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def rmse_parity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute RMSE parity difference with confidence interval.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        method: 'auto', 'bootstrap', or 'bayesian'
        random_state: Random seed

    Returns:
        StatisticalResult with point estimate and interval
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    return compute_metric_with_ci(
        rmse_parity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def mean_prediction_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute mean prediction difference with confidence interval.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        method: 'auto', 'bootstrap', or 'bayesian'
        random_state: Random seed

    Returns:
        StatisticalResult with point estimate and interval
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    return compute_metric_with_ci(
        mean_prediction_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def compute_regression_effect_sizes(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, Dict]:
    """
    Compute effect sizes for pairwise group comparisons in regression.

    Returns Cohen's d for prediction differences between groups,
    providing standardized measures of disparity magnitude.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Dict with effect sizes for each group pair. EMPTY when fewer than two
        groups meet ``min_group_size``, i.e. when no pair exists to compare; a
        warning says so, because an empty dict on its own reads as "every pair
        was examined and none showed an effect". A pair whose Cohen's d could
        NOT be measured (both groups constant, so the standardising denominator
        is zero) carries ``interpretation`` "not interpretable (effect size is
        nan, not a measured value)" rather than a magnitude band.

        TWO effect sizes are returned per pair, so there are TWO labels:
        ``interpretation`` grades ``cohens_d_predictions`` and
        ``interpretation_residuals`` grades ``cohens_d_residuals``. Each grades
        its OWN number; a single label cannot cover both, and the residual one
        had no label at all until 2026-09-29 (see below).

    Example:
        >>> effects = compute_regression_effect_sizes(y_true, y_pred, region)
        >>> for pair, e in effects.items():
        ...     d = e["cohens_d_predictions"]
        ...     print(f"{pair}: Cohen's d = {d:.3f} ({e['interpretation']})")
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    if len(valid_groups) < 2:
        # No pair, so not one Cohen's d was estimated. The return type is a dict
        # of pairs and there is no third value to put in it, so the empty result
        # must DISCLOSE instead, exactly as GroupManager.get_invalid_groups does
        # for its own vacuous empty list. Measured 2026-09-27 on a
        # single-level protected attribute (80 rows, one group): this returned {}
        # with ZERO warnings, which a caller iterating the pairs reads as "no
        # disparity of any magnitude between any pair". _warn_dropped_groups
        # covers only a PARTIAL drop, so the one-level case was fully silent.
        warnings.warn(
            f"compute_regression_effect_sizes: {len(valid_groups)} group(s) meet "
            f"min_group_size={min_group_size} (sizes: {gm.get_group_sizes()}), so no "
            f"pair exists and NO effect size was computed. The empty result means "
            f"nothing was examined, not that no disparity was found.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    results = {}

    for i, g1 in enumerate(valid_groups):
        for g2 in valid_groups[i + 1 :]:
            mask1 = gm.get_mask(g1)
            mask2 = gm.get_mask(g2)

            pred1 = y_pred[mask1]
            pred2 = y_pred[mask2]
            residual1 = y_true[mask1] - pred1
            residual2 = y_true[mask2] - pred2

            # AN ERROR THAT IS CONSTANT WITHIN A GROUP IS CONSTANT ONLY UP TO
            # ROUNDING, AND ITS EFFECT SIZE WAS PUBLISHED AS 6.9e15.
            # BGL5 A-evaluation-3, 2026-09-29. The residual is COMPUTED here, not
            # handed in, so its resolution is set by the magnitude of the operands
            # it is subtracted from and not by its own: `y_true - y_pred` at 1e6
            # cannot resolve anything below about 1.2e-10, however small the
            # residual. IEEE subtraction is exactly rounded, so the whole spread of
            # a within-group-constant error is a couple of ulps at the operand
            # magnitude, and `constant_atol` is that bound handed to the one place
            # that decides degeneracy. Measured before this change, 80 rows in two
            # groups of 40 whose error is exactly 0.0 and exactly 50.0, predictions
            # with real variance:
            #   before: {'cohens_d_predictions': -1.5275,
            #            'cohens_d_residuals': -6948356750493589.0,
            #            'interpretation': 'large', 'mean_residual_diff': -50.0}
            #           with ZERO warnings, and no label on the residual d at all
            #   after:  the same measured -1.5275 and its 'large', plus
            #           'cohens_d_residuals': nan, 'interpretation_residuals':
            #           'not interpretable (effect size is nan, not a measured
            #           value)' and the cohens_d warning naming the 50.0 separation
            # 4 ulps, not 1: `np.spacing` is constant across a binade, so one ulp
            # at the operands' magnitude can be a quarter of an ulp at the largest
            # value actually reached, and a residual assembled by a handful of
            # operations rather than a single subtraction carries a few of them.
            # A real error distribution clears this by ten orders of magnitude.
            #
            # BOTH ARMS, NOT ONE, AND THE BOUND GOES ABOVE THE ARM SELECTION.
            # B4 tier-1 audit, 2026-09-30. This tolerance used to be computed
            # BETWEEN the two cohens_d calls and handed to the residual one only,
            # so the PREDICTION arm kept cohens_d's default relative bound, scaled
            # by the predictions' own (tiny) magnitude, and published a fabricated
            # standardised effect from exactly the same rounding noise. The
            # operand scale is a property of THE ARITHMETIC THAT PRODUCED BOTH
            # arrays and not of the residual: y_pred comes out of the caller's
            # model over the same values y_true is recorded in, so the resolution
            # bound applies to it too. The `_DEGENERATE_DENOMINATOR_D` band below
            # catches only the part of this that is absurd on its face; a small
            # enough between-group gap over the same degenerate denominator lands
            # INSIDE the conventional bands and was graded silently.
            #
            # Measured before this change, y_true recorded at the magnitude the
            # model's arithmetic ran at (`base = np.linspace(1e6, 1e8, n)`) and
            # per-group predictions `(base + 0.1) - base` against
            # `(base + 0.1000001) - base`, i.e. two arrays each constant to 7.5e-9
            # where one ulp at 1e8 is 1.49e-8, separated by 1e-7:
            #   before: cohens_d_predictions -40.91, interpretation 'large',
            #           ZERO warnings, at n = 31, 40, 57 and 100 alike. The 1e3
            #           band cannot reach a fabricated d of 41.
            #   after:  cohens_d_predictions nan, interpretation 'not interpretable
            #           (effect size is nan, not a measured value)', and the
            #           cohens_d warning naming the 1e-7 separation.
            # The base must SPAN BINADES for the noise to exist at all: np.spacing
            # is constant inside one, so `(base + c) - base` rounds identically for
            # every element and the spread comes out exactly 0.0.
            operand_scale = max(
                float(np.max(np.abs(y_true[mask1 | mask2]))),
                float(np.max(np.abs(y_pred[mask1 | mask2]))),
            )
            # NEVER TIGHTER THAN cohens_d's OWN DEFAULT, and that is enforced THERE
            # rather than here: four ulps at the operand magnitude is about 9e-16 of
            # it while the default relative bound is _CONSTANT_SPREAD_REL_TOL (1e-12)
            # of the values handed in, so for an array whose own magnitude IS the
            # operand magnitude this bound is a thousand times tighter and, passed as
            # a replacement, would have REMOVED coverage. cohens_d now takes the
            # larger of the two. Scaling this one by operand_scale instead was tried
            # first and was worse: it made atol 1e-4, which then swallowed a real
            # 1e-7 mean separation into the "genuine zero effect" branch and
            # published d = 0.0 graded "negligible", i.e. it traded one fabrication
            # for a neutral-value one.
            operand_atol = 4.0 * float(np.spacing(operand_scale))

            # Cohen's d for predictions
            d_pred = cohens_d(pred1, pred2, constant_atol=operand_atol)

            # Cohen's d for residuals (error bias)
            d_residual = cohens_d(residual1, residual2, constant_atol=operand_atol)

            # Interpretation.
            # BGL-5 (2026-09-27). `abs(nan) < 0.2` is False, and so is every
            # later rung, so an effect size that was never computed fell out of
            # the BOTTOM of this ladder as "large": the strongest verdict this
            # field can carry, about nothing at all. Measured on 80 rows, two
            # groups of 40 whose predictions are each CONSTANT (130 and 100), so
            # `cohens_d` returns NaN deliberately and warns that "the
            # standardising denominator is zero":
            #   before: {'cohens_d_predictions': nan, 'cohens_d_residuals': nan,
            #            'interpretation': 'large', 'mean_pred_diff': 30.0, ...}
            #   after:  the same dict with 'interpretation': 'not interpretable
            #            (effect size is nan, not a measured value)'
            # The measured fields are untouched: two groups of 40 with real
            # variance still report interpretation 'small' at d = 0.2182.
            # The classification twin of this ladder was fixed for exactly this
            # on 2026-09-16 (classification.py, BGL-S2, "Grade only a finite
            # value"); this copy never got the guard, and the wording is kept
            # identical to it so the two surfaces read the same.
            # A STANDARDISED EFFECT THIS LARGE IS A STATEMENT ABOUT THE DENOMINATOR.
            # B4 tier-1 audit, 2026-09-30. The `constant_atol` added above reaches
            # only the RESIDUAL arm, because it is derived from the operands the
            # residual is subtracted from. `d_pred` has no such handle: the caller
            # computed the predictions and we cannot see what they were computed
            # FROM, so no tolerance derived from the data we hold can recognise
            # their rounding noise as noise.
            #
            # Measured with `base = np.linspace(1e6, 1e8, 40)` and predictions
            # `(base + 0.1) - base` against `(base + 0.2) - base`, which are 0.1 and
            # 0.2 carrying a spread of 7.5e-9 inherited from 1e8 arithmetic:
            #   cohens_d_predictions = -37430695.85, interpretation 'large',
            #   ZERO warnings
            # The base must SPAN BINADES for this to arise at all: np.spacing is
            # constant within one, so `(base + c) - base` rounds identically for
            # every element and the spread comes out exactly zero. Two earlier
            # attempts to reproduce this failed for precisely that reason.
            #
            # The value is arithmetically correct for the arrays as handed in, so
            # it is PUBLISHED rather than refused: refusing would discard a real
            # computation on a guess about provenance. What is corrected is the
            # LABEL. No standardised effect in any real comparison is a million:
            # such a number always means the within-group spread is negligible
            # beside the between-group gap, which is a fact about resolution and
            # not about fairness. The bound is deliberately far above any
            # conventional band (Cohen calls 0.8 large, and 2.0 is already
            # extraordinary) so that nothing a real distribution produces can reach
            # it, and the sentence says which quantity is degenerate.
            _DEGENERATE_DENOMINATOR_D = 1e3

            def interpret(d):
                if not np.isfinite(d):
                    return f"not interpretable (effect size is {d}, not a measured value)"
                d_abs = abs(d)
                if d_abs >= _DEGENERATE_DENOMINATOR_D:
                    return (
                        f"not interpretable (|d| = {d_abs:.3g} means the within-group "
                        "spread is negligible beside the between-group gap, so the "
                        "standardising denominator is resolution rather than variation)"
                    )
                if d_abs < 0.2:
                    return "negligible"
                elif d_abs < 0.5:
                    return "small"
                elif d_abs < 0.8:
                    return "medium"
                else:
                    return "large"

            for _which, _d in (("predictions", d_pred), ("residuals", d_residual)):
                if np.isfinite(_d) and abs(_d) >= _DEGENERATE_DENOMINATOR_D:
                    warnings.warn(
                        f"compute_regression_effect_sizes: {g1} vs {g2} "
                        f"cohens_d_{_which} is {_d:.6g}. A standardised effect of "
                        "that size means the within-group spread is negligible "
                        "beside the between-group gap, so the denominator is the "
                        "resolution of the numbers and not variation in them. The "
                        "value is published because it is arithmetically correct "
                        "for the arrays supplied, and it is NOT a meaningful "
                        "effect size.",
                        UserWarning,
                        stacklevel=2,
                    )

            results[f"{g1}_vs_{g2}"] = {
                "cohens_d_predictions": d_pred,
                "cohens_d_residuals": d_residual,
                "interpretation": interpret(d_pred),
                # `interpret` was applied to d_pred ONLY, so the residual effect
                # size, which is the error-bias finding this function exists to
                # report, carried no magnitude band and no refusal: a reader had
                # a bare float and no statement of whether it was measured. Its
                # own label, from the same three-state ladder.
                "interpretation_residuals": interpret(d_residual),
                "mean_pred_diff": np.mean(pred1) - np.mean(pred2),
                "mean_residual_diff": np.mean(residual1) - np.mean(residual2),
                "group1_size": len(pred1),
                "group2_size": len(pred2),
            }

    return results


def pricing_disparity(
    price: ArrayLike,
    sensitive_attr: ArrayLike,
    controls: Optional[ArrayLike] = None,
    *,
    min_group_size: int = 30,
) -> float:
    """Pricing disparity: the adjusted difference in a continuous priced outcome (APR / note
    rate / premium) across groups AFTER controlling for legitimate risk factors.

    Binary fairness metrics cannot measure price. The group effect is estimated JOINTLY: the
    price is regressed (OLS) on the bona-fide controls AND a full set of group indicators, and
    the metric is the maximum pairwise difference of the group coefficients, i.e. the
    prohibited-basis pricing effect the controls do not explain. This is the
    Frisch-Waugh-Lovell partialling-out estimand of standard fair-lending regression practice
    (Ross & Yinger 2002): equivalently, residualize BOTH the price and the group indicator on
    the controls. Fitting the controls WITHOUT the group indicators and reading the
    between-group residual-mean gap (the previous behaviour) is downward-biased by the factor
    (1 - R^2 of group on controls) whenever the controls correlate with the protected
    attribute, which is the realistic case and biases toward PASS. With no controls the joint
    fit reduces to the raw mean-price difference. The lending profile co-seals this where the
    scope covers price (an approve/decline-only scope omits it).

    Args:
        price: continuous priced outcome (e.g. APR).
        sensitive_attr: protected attribute.
        controls: legitimate risk covariates (1D or 2D) to control for; None = raw price.
        min_group_size: minimum size for a protected group to be assessed.

    Returns:
        Max pairwise adjusted group-effect (or raw mean-price) difference; NaN (insufficient
        evidence) when fewer than 2 groups are assessable or the group effect is not
        identified (controls collinear with the group indicators, e.g. a perfect proxy).

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

    Ledger row: pricing_disparity. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y = np.asarray(price, dtype=float)
    a = np.asarray(sensitive_attr)
    if len(y) != len(a):
        raise ValueError("price and sensitive_attr must have the same length")

    gm = GroupManager(a, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups(warn_if_empty=False)
    # Fewer than two assessable groups: the adjusted price gap is unmeasurable -> NaN
    # (insufficient evidence), never a false-fair 0.0 that would seal price parity.
    if len(valid_groups) < 2:
        return float("nan")

    if controls is None:
        means = {g: float(np.mean(y[gm.get_mask(g)])) for g in valid_groups}
        return max(means.values()) - min(means.values())

    C = np.asarray(controls, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    if len(C) != len(y):
        raise ValueError("controls and price must have the same length")

    # Joint fit: price ~ group indicators + controls. EVERY group present gets its own
    # indicator (the one-hot block spans the intercept), including groups below
    # min_group_size, so that no group's direct price effect can leak into the control
    # coefficients; only the coefficients of assessable groups are compared. Estimating
    # the controls WITHOUT the indicators lets correlated controls absorb part of the
    # group effect (omitted-group-regressor attenuation), understating the disparity.
    D = np.column_stack([gm.get_mask(g).astype(float) for g in gm.groups])
    X = np.column_stack([D, C])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        # The controls are exactly collinear with the group indicators (e.g. a control IS
        # the protected attribute, or is constant within groups). The adjusted group effect
        # is then unidentified; any split between controls and indicators is arbitrary.
        # Report NaN (insufficient evidence), never an arbitrary minimum-norm number.
        warnings.warn(
            "pricing_disparity: controls are collinear with the group indicators, so the "
            "adjusted group effect is not identified; returning NaN.",
            UserWarning,
        )
        return float("nan")
    effects = {g: float(beta[i]) for i, g in enumerate(gm.groups) if g in valid_groups}
    return max(effects.values()) - min(effects.values())


def pricing_disparity_with_ci(
    price: ArrayLike,
    sensitive_attr: ArrayLike,
    controls: Optional[ArrayLike] = None,
    *,
    min_group_size: int = 30,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the adjusted pricing disparity (the lending co-sealed price
    metric). Bootstraps over row indices, stratified by the protected attribute; the joint OLS
    fit (controls plus group indicators) is refit on each resample so the CI reflects the
    control-estimation uncertainty too. The seal gates on the whole CI lying within the
    pre-registered price band (TOST equivalence).

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: pricing_disparity_with_ci. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y = np.asarray(price, dtype=float)
    a = np.asarray(sensitive_attr)
    C = None if controls is None else np.asarray(controls, dtype=float)

    def stat(i: np.ndarray) -> float:
        ci = None if C is None else (C[i] if C.ndim == 1 else C[i, :])
        return pricing_disparity(y[i], a[i], ci, min_group_size=min_group_size)

    return bootstrap_over_index(
        len(y),
        stat,
        groups=a,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
