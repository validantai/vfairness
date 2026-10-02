"""
Statistical validation utilities for vfairness.

This module provides tools for uncertainty quantification and statistical rigor:
- Bootstrap confidence intervals with stratified sampling
- Bayesian credible intervals for a single PROPORTION (:func:`bayesian_proportion_ci`),
  a two-group PROPORTION GAP (:func:`bayesian_difference_ci`) and a MEAN
  (:func:`bayesian_mean_ci`). The per-group rates take the same route through
  ``get_group_metrics_with_ci(method='bayesian')``, which is the one place a
  ``method='bayesian'`` request is honoured. There is NO Bayesian estimator for a
  disparity (group-gap) metric: :func:`compute_metric_with_ci` runs the stratified bootstrap
  for ``method='auto'``, ``'bootstrap'`` AND ``'bayesian'`` alike, warning on the
  last that it is not implemented (F18, 2026-09-09).
- Multiple comparison corrections (Bonferroni, Benjamini-Hochberg)
- Effect size calculations (Cohen's d, risk ratios, odds ratios)
- Sample-size-based method RECOMMENDATION (:func:`select_method`), which is a
  suggestion read off the group sizes and never a record of what ran. The
  estimator that ran is on ``StatisticalResult.method``.

References:
    - Efron & Tibshirani (1993): Bootstrap Methods
    - Gelman et al. (2013): Bayesian Data Analysis
    - Benjamini & Hochberg (1995): FDR Control
    - Cohen (1988): Statistical Power Analysis
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..._triage import is_measured
from ...exceptions import ConfigurationError, InvalidDataError

logger = logging.getLogger(__name__)


class IntervalType(Enum):
    """Type of uncertainty interval."""

    CONFIDENCE = "confidence"  # Frequentist bootstrap CI
    CREDIBLE = "credible"  # Bayesian credible interval


@dataclass
class StatisticalResult:
    """
    Container for statistical validation results.

    Clearly distinguishes between confidence intervals (frequentist)
    and credible intervals (Bayesian) to avoid confusion.

    Attributes:
        point_estimate: The computed metric value
        lower_bound: Lower bound of the interval
        upper_bound: Upper bound of the interval
        interval_type: Whether this is a confidence or credible interval
        confidence_level: The confidence/credible level (e.g., 0.95)
        method: Method used for interval estimation
        sample_size: Number of samples used
        n_bootstrap: Number of bootstrap samples (if applicable)
        standard_error: Standard error estimate (if available)
        effect_size: Standardized effect size (if computed)
        effect_size_type: Type of effect size measure
        null_value: The value this metric takes under the parity null, i.e. the
            value ``is_significant`` tests the interval against. 0.0 for a
            DIFFERENCE or spread statistic (the default, and what every metric
            in this package produced before ratios were given a CI); 1.0 for a
            RATIO, whose parity value is 1 and which can never reach 0 from
            above. Declared by the metric that produced the result, because
            nothing downstream can tell the two families apart from a number.
        metadata: Additional information about the computation
    """

    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_type: IntervalType
    confidence_level: float = 0.95
    method: str = "bootstrap_percentile"
    sample_size: int = 0
    n_bootstrap: int = 0
    standard_error: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    null_value: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def margin_of_error(self) -> float:
        """Half-width of the interval."""
        return (self.upper_bound - self.lower_bound) / 2

    @property
    def interval_width(self) -> float:
        """Full width of the interval."""
        return self.upper_bound - self.lower_bound

    @property
    def is_significant(self) -> bool:
        """Check if the interval excludes this metric's parity null.

        The null is ``null_value``: 0 for a difference or spread statistic, 1
        for a ratio. It is NOT hardcoded to zero any more.

        Until 2026-08-28 this tested "the interval excludes ZERO" while being a
        property of the SHARED result type, serialised into every ``to_dict()``
        payload including ``disparate_impact_ratio_with_ci`` and
        ``demographic_parity_ratio``. A ratio's parity value is 1.0 and its
        values are bounded in [0, 1], so on a ratio the test could only ever
        answer True: a model with an exactly equal selection rate in every group
        reported ``is_significant=True``, i.e. "significant disparity", on the
        very statistic the US four-fifths seal reads. The docstring said "(for
        difference metrics)" and the field said nothing, and a serialised field
        is read without its docstring.

        An uncomputable interval (NaN bounds) is NOT significant: NaN
        comparisons are all False, so the original ``not (lo <= 0 <= hi)``
        returned True for exactly the intervals that carry no information.
        """
        if np.isnan(self.lower_bound) or np.isnan(self.upper_bound):
            return False
        null = self.null_value
        if null != null:  # a NaN null cannot be excluded or contained
            return False
        return not (self.lower_bound <= null <= self.upper_bound)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "interval_type": self.interval_type.value,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "sample_size": self.sample_size,
            "n_bootstrap": self.n_bootstrap,
            "standard_error": self.standard_error,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "margin_of_error": self.margin_of_error,
            # Serialised NEXT TO is_significant on purpose: the flag is a claim
            # about an interval relative to a null, and a consumer reading the
            # flag out of storage cannot check it without knowing which null was
            # tested. Omitting it is how the ratio family's 1.0 null went unseen.
            "null_value": self.null_value,
            "is_significant": self.is_significant,
            "metadata": self.metadata,
        }


@dataclass
class MultipleTestingResult:
    """
    Results from multiple comparison correction.

    Attributes:
        original_p_values: Original p-values
        adjusted_p_values: Corrected p-values
        rejection_mask: Boolean mask of rejected hypotheses
        method: Correction method used
        significance_level: Alpha level used
        n_rejected: Number of rejected hypotheses
        tested_mask: True where the p-value was a finite number and the
            hypothesis was actually part of the family. False entries are NOT
            "tested and not rejected": nothing was tested there at all, and
            `rejection_mask` cannot say so on its own because it is a bool array.
        n_not_tested: How many entries were excluded from the family.
    """

    original_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    rejection_mask: np.ndarray
    method: str
    significance_level: float
    n_rejected: int
    tested_mask: Optional[np.ndarray] = None
    n_not_tested: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "original_p_values": self.original_p_values.tolist(),
            "adjusted_p_values": self.adjusted_p_values.tolist(),
            "rejection_mask": self.rejection_mask.tolist(),
            "tested_mask": (None if self.tested_mask is None else self.tested_mask.tolist()),
            "n_not_tested": self.n_not_tested,
            "method": self.method,
            "significance_level": self.significance_level,
            "n_rejected": self.n_rejected,
        }


# Sample Size Thresholds for Automatic Method Selection

SMALL_SAMPLE_THRESHOLD = 30  # Below this: compute_metric_with_ci doubles the bootstrap resamples
MEDIUM_SAMPLE_THRESHOLD = 50  # Below this: use more bootstrap samples
RECOMMENDED_BOOTSTRAP_SAMPLES = 5000
MINIMUM_BOOTSTRAP_SAMPLES = 1000

# Hard ceiling on the permutation count derived from alpha in
# compute_metric_with_ci. Even a legal confidence_level can ask for an absurd
# number of full metric recomputations (0.9999 -> 40,001; 0.99999 -> 400,001),
# and the loop is silent while it runs. Above the ceiling the permutation gate
# is RESOLUTION-LIMITED rather than exact, which is disclosed in the result's
# metadata (see _permutation_count) instead of being absorbed silently.
MAX_PERMUTATIONS = 20_000


def validate_confidence_level(confidence_level: Any, name: str = "confidence_level") -> float:
    """Refuse a confidence level that is not a probability strictly inside (0, 1).

    FAIL CLOSED at the public boundary. Nothing validated this until 2026-08-28,
    and each unchecked value failed in its own way, none of them as an error the
    caller could act on:

    * ``1.0`` gives alpha = 0, which the ``max(alpha, 1e-6)`` floor in
      ``compute_metric_with_ci`` turned into 4,000,000 permutations -- each one a
      full metric recomputation. An unbounded silent hang, not slowness.
    * ``0.0`` and ``-0.2`` returned a degenerate near-zero-width interval AND
      ``is_significant=True``. The library's headline claim is that the verdict is
      read from the whole interval; an interval collapsed to a point makes every
      result a definite call.
    * ``95`` (the 95-vs-0.95 slip) surfaced as a raw numpy
      ``ValueError: Percentiles must be in the range [0, 100]``, which is actively
      confusing because the caller did pass a value inside [0, 100].

    ``docs/API_STABILITY.md`` promises ``ConfigurationError`` when "an argument is
    out of range", so that is what this raises.

    Returns:
        The value as a float, so callers can use the validated result directly.

    Raises:
        ConfigurationError: If the value is not a real number in (0, 1).
    """
    # A bool is refused explicitly: True is an int in Python and would coerce to
    # 1.0, which is the runaway-permutation case. A numeric STRING is refused
    # too, rather than coerced: it travels on into the report metadata and into
    # arithmetic that has no business guessing what "0.95" meant.
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise ConfigurationError(
            f"{name} must be a number strictly between 0 and 1, got "
            f"{confidence_level!r} ({type(confidence_level).__name__}). "
            "Pass 0.95 for a 95% interval."
        )
    value = float(confidence_level)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ConfigurationError(
            f"{name} must be strictly between 0 and 1, got {confidence_level!r}; "
            "pass 0.95 for a 95% interval, not 95."
        )
    return value


# GROUP-SIZE RELIABILITY TIERS
#
# A single n>=30 cut-off (the old behaviour) hides the difference between a
# 31-person group (read with caution) and a 9-person group (do NOT interpret
# the rate at all -- the 95% margin of error exceeds the metric). One shared
# tiering used by Pulse, the Navigator, and the Modules so every per-group
# number carries the same honesty flag everywhere.
#   reliable      n >= 100   report normally
#   caution       30..99     report, note wider CI
#   underpowered  10..29     report greyed, "small sample"
#   invalid       n < 10     DO NOT interpret the rate (shown, never ranked)
# Thresholds follow the Fairlearn small-group convention and the Turing M3
# minimum-reporting guidance.

RELIABLE_THRESHOLD = 100
CAUTION_THRESHOLD = 30
UNDERPOWERED_THRESHOLD = 10


def reliability_tier(n: int) -> str:
    """Map a group size to its reliability tier."""
    if n >= RELIABLE_THRESHOLD:
        return "reliable"
    if n >= CAUTION_THRESHOLD:
        return "caution"
    if n >= UNDERPOWERED_THRESHOLD:
        return "underpowered"
    return "invalid"


def group_reliability(sizes: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """Tier every group by sample size.

    Args:
        sizes: {group_name: n}

    Returns:
        {group_name: {n, tier, interpretable, note}} where ``interpretable``
        is False for the 'invalid' tier (rate must not be read) and the
        ``note`` is a one-line plain-language caveat for the UI.
    """
    notes = {
        "reliable": "",
        "caution": "Small group (n<100): read the rate with a wider confidence interval.",
        "underpowered": "Underpowered (n<30): the rate is noisy; treat as indicative only.",
        "invalid": "Too few people (n<10): the rate is not interpretable "
        "and is excluded from the verdict.",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, n in sizes.items():
        n = int(n)
        tier = reliability_tier(n)
        out[str(name)] = {
            "n": n,
            "tier": tier,
            "interpretable": tier != "invalid",
            "note": notes[tier],
        }
    return out


def select_method(
    sample_sizes: Dict[str, int], prefer_bayesian: bool = False
) -> Tuple[str, Dict[str, str]]:
    """
    Recommend an interval family per group FROM SAMPLE SIZE ALONE.

    This reads group sizes and nothing else, so what it returns is a
    RECOMMENDATION, never a record of what an estimator ran:

    - ``n < 30`` -> ``'bayesian'``: the Beta-posterior credible interval that
      :func:`~vfairness.get_group_metrics_with_ci` builds for a small group's
      PROPORTION. That estimator exists (closed-form Beta quantiles).
    - ``30 <= n < 50`` -> ``'bootstrap_enhanced'``: a bootstrap with extra
      resamples. Nothing in this library selects it at that size;
      :func:`compute_metric_with_ci` doubles its resamples below 30, not
      below 50.
    - ``n >= 50`` -> ``'bootstrap'``.
    - Overall: ``'bayesian'`` when every group is small, ``'mixed'`` when the
      groups disagree, otherwise ``'bootstrap'``.

    NEVER record the returned label as the PROVENANCE of a disparity interval.
    No Bayesian estimator exists for the group-gap metrics: every ``*_with_ci``
    disparity function routes through :func:`compute_metric_with_ci`, which
    runs a stratified bootstrap for ``'auto'``, ``'bootstrap'`` AND
    ``'bayesian'`` alike (F18, 2026-09-09). Measured 2026-09-09 on groups of
    20 and 200, this function returned ``('mixed', {'A': 'bayesian', 'B':
    'bootstrap'})`` while all three intervals built from the same data carried
    ``StatisticalResult.method == 'stratified_bootstrap_fold_debiased'``. The
    method that actually ran is on the result object, never here.

    Args:
        sample_sizes: Dict mapping group names to sample counts
        prefer_bayesian: NOT SUPPORTED, and refused rather than ignored. The
            body never read it (audit 6, S-09) and it cannot be implemented:
            there is no Bayesian estimator for a disparity metric to prefer,
            so honouring it would steer a caller to a method that does not
            exist in this library.

    Returns:
        Tuple of (overall_method, per_group_methods)

    Raises:
        ConfigurationError: If ``prefer_bayesian`` is true.
    """
    if prefer_bayesian:
        raise ConfigurationError(
            "select_method(prefer_bayesian=True) is not supported: there is no "
            "Bayesian estimator for a disparity metric in this library, so "
            "there is nothing to prefer, and the argument was never read. "
            "What IS supported: the size-based recommendation returned with "
            "prefer_bayesian=False; bayesian_proportion_ci() / "
            "get_group_metrics_with_ci(method='bayesian') for a single group's "
            "proportion; and bayesian_difference_ci() for a two-group "
            "proportion gap. Disparity CIs always run a stratified bootstrap "
            "(see StatisticalResult.method on the result)."
        )

    per_group_methods = {}

    for group, n in sample_sizes.items():
        if n < SMALL_SAMPLE_THRESHOLD:
            per_group_methods[group] = "bayesian"
        elif n < MEDIUM_SAMPLE_THRESHOLD:
            per_group_methods[group] = "bootstrap_enhanced"
        else:
            per_group_methods[group] = "bootstrap"

    # Determine overall method
    methods = set(per_group_methods.values())
    if "bayesian" in methods:
        if len(methods) == 1:
            overall = "bayesian"
        else:
            overall = "mixed"  # Some groups need different treatment
    else:
        overall = "bootstrap"

    return overall, per_group_methods


# Bootstrap Confidence Intervals


def _interval_from_bootstrap(
    point_estimate: float,
    bootstrap_stats: np.ndarray,
    confidence_level: float,
    method: str,
) -> Tuple[float, float, Dict]:
    """Build a CI from a bootstrap distribution: 'percentile' or 'basic'.

    'basic' (reverse percentile; Efron & Tibshirani 1993 ch. 10, Davison &
    Hinkley 1997 sec. 5.2) reflects the resample quantiles around the point
    estimate, cancelling the statistic's plug-in bias to first order. This
    matters for folded, nonnegative spread statistics (max-min disparities):
    their resample distribution sits strictly above 0 at the parity null, so
    a raw percentile CI can NEVER contain the true value 0 there (0% coverage,
    is_significant always True; audit 2026-08-22). Only a pivot-based
    construction can reach the boundary: any quantile of the resample
    distribution itself (percentile, and BCa too) stays trapped above it.

    When the point estimate and every resample are nonnegative the statistic
    is treated as bounded below at 0 and the basic interval is clipped there
    (bound-aware construction). The clipped lower endpoint is a boundary
    statement, not a significance test.

    Returns (lower, upper, metadata_updates).
    """
    alpha = 1 - confidence_level
    q_lo = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    q_hi = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    if method == "percentile":
        return q_lo, q_hi, {"interval_construction": "percentile"}
    if method != "basic":
        raise ConfigurationError(
            f"Unsupported bootstrap interval method here: {method!r} (use 'percentile' or 'basic')"
        )
    lower = 2.0 * point_estimate - q_hi
    upper = 2.0 * point_estimate - q_lo
    meta: Dict = {"interval_construction": "basic_reverse_percentile"}
    if point_estimate >= 0.0 and bool(np.all(bootstrap_stats >= 0.0)):
        meta["nonnegative_statistic"] = True
        lower = max(0.0, lower)
        upper = max(upper, lower)
    return lower, upper, meta


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Implements percentile, BCa (bias-corrected and accelerated), and
    basic bootstrap methods.

    Args:
        data: Input data array
        statistic: Function that computes the statistic of interest
        n_bootstrap: Number of bootstrap resamples (≥5000 recommended)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Bootstrap method ('percentile', 'bca', or 'basic')
        random_state: Random seed for reproducibility

    Returns:
        StatisticalResult with confidence interval

    Example:
        >>> result = bootstrap_ci(data, np.mean, n_bootstrap=5000)
        >>> print(f"Mean: {result.point_estimate:.3f} "
        ...       f"[{result.lower_bound:.3f}, {result.upper_bound:.3f}]")

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).

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

    Ledger row: bootstrap_ci. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    confidence_level = validate_confidence_level(confidence_level)
    if len(data) < 2:
        point_est = statistic(data) if len(data) > 0 else np.nan
        return StatisticalResult(
            point_estimate=point_est,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=f"bootstrap_{method}",
            sample_size=len(data),
            n_bootstrap=0,
            metadata={"warning": "Insufficient data for bootstrap"},
        )

    rng = np.random.default_rng(random_state)
    n = len(data)

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        resample = data[resample_idx]
        bootstrap_stats[i] = statistic(resample)

    # Remove any NaN values from bootstrap results
    bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
    if len(bootstrap_stats) < n_bootstrap * 0.9:
        warnings.warn("More than 10% of bootstrap samples resulted in NaN")

    if len(bootstrap_stats) < 10:
        return StatisticalResult(
            point_estimate=point_estimate,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=f"bootstrap_{method}",
            sample_size=n,
            n_bootstrap=n_bootstrap,
            metadata={"warning": "Too many NaN bootstrap samples"},
        )

    alpha = 1 - confidence_level

    if method == "percentile":
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == "basic":
        # Basic/reverse bootstrap
        lower_pct = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        upper_pct = np.percentile(bootstrap_stats, 100 * alpha / 2)
        lower = 2 * point_estimate - lower_pct
        upper = 2 * point_estimate - upper_pct

    elif method == "bca":
        # BCa (Bias-Corrected and Accelerated)
        # Bias correction factor
        prop_less = float(np.mean(bootstrap_stats < point_estimate))
        z0 = _norm_ppf(prop_less) if 0 < prop_less < 1 else 0

        # Acceleration factor (jackknife)
        jackknife_stats = np.empty(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        numer = np.sum((jack_mean - jackknife_stats) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = numer / denom if denom != 0 else 0

        # Adjusted percentiles
        z_lower = _norm_ppf(alpha / 2)
        z_upper = _norm_ppf(1 - alpha / 2)

        alpha1 = _norm_cdf(z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower)))
        alpha2 = _norm_cdf(z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper)))

        lower = np.percentile(bootstrap_stats, 100 * alpha1)
        upper = np.percentile(bootstrap_stats, 100 * alpha2)
    else:
        raise ConfigurationError(f"Unknown bootstrap method: {method}")

    # Standard error from bootstrap distribution
    se = float(np.std(bootstrap_stats, ddof=1))

    return StatisticalResult(
        point_estimate=point_estimate,
        lower_bound=lower,
        upper_bound=upper,
        interval_type=IntervalType.CONFIDENCE,
        confidence_level=confidence_level,
        method=f"bootstrap_{method}",
        sample_size=n,
        n_bootstrap=len(bootstrap_stats),
        standard_error=se,
    )


# ---------------------------------------------------------------------------
# A LENGTH CHECK IS NOT A COVERAGE CHECK, AND THE REFUSAL MESSAGE ALREADY SAID SO.
#
# B4 tier-1 audit, 2026-09-30. Both stratified entry points below refuse a groups
# array of the WRONG LENGTH. A groups array of the RIGHT length whose strata do not
# COVER every row walked straight past both, and a missing group label is exactly
# that: ``np.unique`` LISTS nan as a level while ``groups == nan`` matches nothing,
# so ``group_indices`` holds an empty index array for it and those rows are deleted
# from every replicate while the point estimate still uses all of them.
#
# Measured before this change, through
# ``bootstrap_over_index(n, mean-of-rows, groups)`` with
# ``groups = [0.0]*h + [1.0]*m + [nan]*k`` over ``data = np.arange(float(n))``,
# n_bootstrap=200, random_state=3:
#   n= 20  point_estimate 9.5,  interval [5.6000, 8.2017],  sample_size 20
#   n= 31  point_estimate 15.0, interval [9.8313, 12.8750], sample_size 31
#   n= 40  point_estimate 19.5, interval [12.7983, 16.1350], sample_size 40
#   n= 57  point_estimate 28.0, interval [19.0000, 23.1180], sample_size 57
#   n=100  point_estimate 49.5, interval [34.1050, 39.4020], sample_size 100
# with ZERO UserWarnings and no not-measured field of any kind. In every one of
# them ``lower_bound <= point_estimate <= upper_bound`` is FALSE: the published
# interval does not contain its own published point estimate, which is the VERBATIM
# symptom the length check was added for. The refusal ``stratified_bootstrap_ci``
# already raises even ASSERTS the contract that was not enforced, "The strata must
# cover every row".
#
# Two adjacent shapes did not fabricate, they CRASHED: an object-dtype groups array
# holding np.nan raised a bare ``TypeError: '<' not supported between instances of
# 'float' and 'str'``, one holding None the same for 'NoneType', and a pandas
# string-extension array holding pd.NA raised ``TypeError: boolean value of NA is
# ambiguous``, all three out of numpy's own sort rather than as a refusal. An
# exception out of the sort is not one of the three states either.
# ---------------------------------------------------------------------------


def _absent_stratum_labels(groups_arr: np.ndarray) -> np.ndarray:
    """Boolean mask of positions whose stratum LABEL is absent rather than a category.

    THIS IS FOR THE MESSAGE, NOT FOR THE DECISION, and that separation is
    deliberate. ``_rows_no_stratum_can_draw`` below decides by forming the strata
    exactly as the resampler forms them, which is the thing itself rather than a
    proxy for it; an earlier draft of this made absence a SECOND decision axis, and
    sabotaging either axis left the guard green because each one covered every case
    the other did. One decision, one explanation.

    ABSENCE IS NOT ONE TEST. None, float nan, ``np.datetime64('NaT')``, ``pd.NA``
    and ``pd.NaT`` are different objects and none of the obvious predicates covers
    them all: ``x is None`` misses nan, ``x != x`` RAISES for ``pd.NA`` ("boolean
    value of NA is ambiguous"), and ``np.isnan`` raises on an object or string
    array. ``pandas.isna`` is the one predicate that answers for every one of them,
    and pandas is already a dependency of this package (``_grouping`` imports it).
    """
    try:
        mask = np.asarray(pd.isna(groups_arr), dtype=bool)
    except (TypeError, ValueError) as exc:
        # A dtype pandas cannot answer for. Say "no absence found" and let the
        # coverage test reach its own verdict, rather than inventing one here. The
        # error is LOGGED and not discarded: a handler that swallows an exception
        # and substitutes an answer is indistinguishable from a measurement.
        logger.debug("_absent_stratum_labels: pd.isna could not answer: %s", exc)
        return np.zeros(len(groups_arr), dtype=bool)
    if mask.shape != (len(groups_arr),):
        return np.zeros(len(groups_arr), dtype=bool)
    return mask


def _rows_no_stratum_can_draw(groups_arr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """Positions a stratified resample can never draw, and why the strata could not be
    formed, when they could not.

    The strata are formed HERE exactly as ``stratified_bootstrap_ci`` forms them,
    ``np.unique`` then ``groups == g``, so this measures the thing itself and not a
    proxy for it: a label that ``np.unique`` LISTS while ``groups == g`` matches no
    row (which is what nan does) shows up as an uncovered row without anybody having
    to enumerate the ways a label can be absent. Returns
    ``(positions, sort_error)``, where ``sort_error`` is None when the strata were
    formed and otherwise carries the text of the error that stopped them, so the
    exception is REPORTED rather than swallowed.
    """
    n_rows = len(groups_arr)
    try:
        covered = np.zeros(n_rows, dtype=bool)
        for g in np.unique(groups_arr):
            covered |= np.asarray(groups_arr == g, dtype=bool)
    except (TypeError, ValueError) as exc:
        # np.unique SORTS, and a mixed object array raises out of the sort, which is
        # where the bare TypeErrors came from. The resampler would raise in exactly
        # the same place, so no row is drawable and the caller gets a refusal naming
        # the cause instead of numpy's traceback. The error's own TEXT is carried out
        # of here and into that refusal: a handler that discards an exception and
        # substitutes an answer is indistinguishable from a measurement, which is what
        # tests/test_no_silent_swallow_core.py exists to stop. The absent positions are
        # looked up only to NAME them in the message; they are not part of the decision.
        logger.debug(
            "_rows_no_stratum_can_draw: np.unique could not form strata over %d label(s): %s",
            n_rows,
            exc,
        )
        return np.flatnonzero(_absent_stratum_labels(groups_arr)), f"{type(exc).__name__}: {exc}"
    return np.flatnonzero(~covered), None


def _strata_coverage_refusal(caller: str, groups_arr: np.ndarray, n_rows: int) -> Optional[str]:
    """The refusal message for strata that do not cover every row, or None."""
    uncovered, sort_error = _rows_no_stratum_can_draw(groups_arr)
    if sort_error is None and not len(uncovered):
        return None
    labels = [repr(groups_arr[i]) for i in uncovered[:5].tolist()]
    if sort_error is not None:
        return (
            f"{caller}: the {n_rows} stratum label(s) cannot be sorted into strata at "
            f"all, so no stratified resample is possible and no interval was estimated. "
            f"numpy's own sort raised {sort_error}, which is how a mixed object array "
            f"holding None or nan used to surface as a bare TypeError rather than a "
            f"refusal. Absent label(s) found at row(s) {uncovered[:5].tolist()}: "
            f"{labels}. Label the rows or drop them before asking for an interval."
        )
    return (
        f"{caller}: {len(uncovered)} of {n_rows} row(s) belong to no stratum, so a "
        f"stratified resample can never draw them and the interval would be estimated "
        f"from the other {n_rows - len(uncovered)} row(s) while the point estimate uses "
        f"all {n_rows}. The strata must cover every row. First uncovered row(s): "
        f"{uncovered[:5].tolist()}, with label(s) {labels}. A MISSING GROUP LABEL is "
        f"exactly this shape: np.unique LISTS nan as a level while `groups == nan` "
        f"matches nothing, so those rows are deleted from every replicate in silence. "
        f"Label the rows or drop them before asking for an interval; this function "
        f"cannot decide which on the caller's behalf."
    )


def stratified_bootstrap_ci(
    data: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["percentile", "basic"] = "percentile",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute bootstrap CI with stratified resampling to maintain group proportions.

    Essential for fairness metrics to ensure each group is represented
    proportionally in each bootstrap sample.

    Args:
        data: Input data array
        groups: Group membership array (same length as data)
        statistic: Function(data, groups) -> float
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level
        method: Interval construction, 'percentile' or 'basic'. Previously
            the parameter was accepted but IGNORED (always percentile);
            'bca' is not implemented for the stratified path and now raises
            instead of silently computing a percentile interval. Prefer
            'basic' for folded/nonnegative spread statistics: the percentile
            interval of a max-min spread has 0% coverage at the parity null
            (audit 2026-08-22).
        random_state: Random seed

    Returns:
        StatisticalResult with confidence interval

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1),
            or the method is unsupported.
    """
    confidence_level = validate_confidence_level(confidence_level)
    if method not in ("percentile", "basic"):
        # Fail fast: this parameter used to be accepted and silently ignored
        # (percentile regardless), which is how 'bca' could appear to work.
        raise ConfigurationError(
            f"Unsupported bootstrap method for the stratified path: {method!r} "
            "(use 'percentile' or 'basic')"
        )
    # THE STRATA MUST DESCRIBE EVERY ROW, checked before any stratum is formed.
    # ``group_indices`` below is built from ``np.where(groups == g)[0]``, so a
    # groups array SHORTER than data can only ever index the rows it covers: the
    # point estimate is then computed over every row and the interval over a
    # silent truncation of them. Measured 2026-09-27 through
    # bootstrap_over_index(20, mean-of-data, groups=np.array(['a'] * 5)):
    # point_estimate 9.5, lower_bound 0.845, upper_bound 3.155 (the interval of
    # rows 0..4), sample_size still 20, warnings []. An interval that does not
    # contain its own point estimate was published in silence. Now:
    # InvalidDataError, the same refusal _validation.check_consistent_length
    # gives for the same mistake.
    if len(groups) != len(data):
        raise InvalidDataError(
            f"stratified_bootstrap_ci: groups has {len(groups)} entr(ies) for "
            f"{len(data)} row(s). The strata must cover every row, otherwise the "
            f"resample can only draw from the rows groups describes and the "
            f"interval is built from a truncated slice of the data while the "
            f"point estimate uses all of it."
        )
    # A LENGTH CHECK IS NOT A COVERAGE CHECK. The refusal just above asserts "The
    # strata must cover every row" and only the LENGTH was enforced; see the block
    # above _absent_stratum_labels for the measured intervals that did not contain
    # their own point estimate. The check sits here, where the strata are formed, so
    # every caller inherits it, and it is repeated at the bootstrap_over_index entry
    # point for the same reason the length check is: the message a caller of THAT
    # function needs names ``n``, not ``data``.
    _refusal = _strata_coverage_refusal("stratified_bootstrap_ci", np.asarray(groups), len(data))
    if _refusal is not None:
        raise InvalidDataError(_refusal)
    if len(data) < 2:
        point_est = statistic(data, groups) if len(data) > 0 else np.nan
        return StatisticalResult(
            point_estimate=point_est,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=f"stratified_bootstrap_{method}",
            sample_size=len(data),
            n_bootstrap=0,
            metadata={"warning": "Insufficient data"},
        )

    rng = np.random.default_rng(random_state)
    unique_groups, _stratum_sizes = np.unique(groups, return_counts=True)

    # NO STRATUM WITH TWO ROWS IN IT IS NOT A RESAMPLE. ``rng.choice`` over a
    # stratum of one row can only return that row, so when EVERY stratum holds a
    # single row each replicate IS the original sample, every bootstrap statistic
    # is identical, and the percentile interval has zero width, which asserts the
    # statistic is known exactly. This is the per-stratum twin of the
    # ``len(data) < 2`` guard above, which cannot see it: the sample as a whole
    # is large enough. Measured 2026-09-27 through bootstrap_over_index(2,
    # mean-of-data, groups=np.array(['a', 'b'])): lower_bound == upper_bound ==
    # 0.5, standard_error 0.0, warnings []. Now: nan bounds plus a UserWarning,
    # with the point estimate kept, because it WAS measured. A mixed case is
    # untouched, and that is the over-correction control: 20 rows in 'a' plus 1
    # in 'b' still returns [7.5226, 12.0964] around a point estimate of 10.0,
    # because the 20-row stratum does vary under resampling.
    if len(_stratum_sizes) and int(_stratum_sizes.max()) < 2:
        warnings.warn(
            f"stratified_bootstrap_ci: every one of the {len(unique_groups)} strata "
            f"holds a single row, so a stratified resample can only ever return the "
            f"original sample and every bootstrap replicate is identical. No "
            f"interval was estimated (lower_bound and upper_bound are nan, which "
            f"means could not check); a zero-width interval would have claimed the "
            f"statistic is known exactly. The point estimate itself is measured.",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=statistic(data, groups),
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=f"stratified_bootstrap_{method}",
            sample_size=len(data),
            n_bootstrap=0,
            metadata={
                "n_groups": len(unique_groups),
                "warning": "Every stratum holds one row; no resampling variation is possible",
            },
        )

    # Point estimate
    point_estimate = statistic(data, groups)

    # Group indices for stratified sampling
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    # Stratified bootstrap resampling
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample within each group maintaining group sizes
        resample_idx_list: list[Any] = []
        for g, indices in group_indices.items():
            n_g = len(indices)
            sampled = rng.choice(indices, size=n_g, replace=True)
            resample_idx_list.extend(sampled)

        resample_idx = np.array(resample_idx_list)
        resample_data = data[resample_idx]
        resample_groups = groups[resample_idx]
        bootstrap_stats[i] = statistic(resample_data, resample_groups)

    # Remove NaN values
    bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
    if len(bootstrap_stats) < n_bootstrap * 0.9:
        # Dropping NaN resamples silently can narrow a sealed CI (for a MAX statistic like
        # multicalibration alpha, the NaN resamples are the low-support ones), so flag it.
        warnings.warn("More than 10% of bootstrap samples resulted in NaN")

    if len(bootstrap_stats) < 10:
        return StatisticalResult(
            point_estimate=point_estimate,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=f"stratified_bootstrap_{method}",
            sample_size=len(data),
            n_bootstrap=n_bootstrap,
            metadata={"warning": "Too many NaN bootstrap samples"},
        )

    lower, upper, interval_meta = _interval_from_bootstrap(
        point_estimate, bootstrap_stats, confidence_level, method
    )
    se = float(np.std(bootstrap_stats, ddof=1))

    return StatisticalResult(
        point_estimate=point_estimate,
        lower_bound=lower,
        upper_bound=upper,
        interval_type=IntervalType.CONFIDENCE,
        confidence_level=confidence_level,
        method=f"stratified_bootstrap_{method}",
        sample_size=len(data),
        n_bootstrap=len(bootstrap_stats),
        standard_error=se,
        metadata={"n_groups": len(unique_groups), **interval_meta},
    )


def bootstrap_over_index(
    n: int,
    statistic: Callable[[np.ndarray], float],
    groups: Optional[np.ndarray] = None,
    *,
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Bootstrap an arbitrary closure ``statistic(index_array) -> float`` by resampling ROW
    INDICES, so a statistic that needs more than (y_true, y_pred, sensitive) -- e.g. y_prob, a
    strata array, or continuous controls -- can still get a confidence interval. Index the real
    arrays by the resampled indices inside the closure. When ``groups`` is given the resample is
    stratified within groups (group sizes preserved), matching the disparity metrics' _with_ci
    variants; otherwise it is a simple resample. Reuses ``bootstrap_ci`` /
    ``stratified_bootstrap_ci`` so the CI construction and NaN handling are identical.
    Note: ``method='bca'`` is only implemented for the unstratified branch; with
    ``groups`` given, use 'percentile' or 'basic' (the stratified path raises on 'bca').

    Raises:
        InvalidDataError: If ``groups`` is given and does not have exactly ``n``
            entries (the strata would describe different rows than the ones this
            function indexes), or via the stratified path if every stratum holds
            a single row, in which case nan bounds are returned instead.
        ConfigurationError: If ``method='bca'`` is combined with ``groups``.
    """
    idx = np.arange(int(n))
    if groups is None:
        return bootstrap_ci(
            idx,
            lambda resampled: statistic(resampled),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state,
        )
    # THIS is the one place the length contract lives, because this function
    # builds ``idx = np.arange(int(n))`` itself: the caller passes ``n`` and a
    # groups array separately and nothing else can compare them. Measured
    # 2026-09-27 with n=20 and groups=np.array(['a'] * 5): point_estimate 9.5
    # (all 20 rows) with the interval [0.845, 3.155] (rows 0..4 only),
    # sample_size 20, warnings [], i.e. an interval that does not contain its own
    # point estimate. Now: InvalidDataError naming both lengths. Checked here as
    # well as in stratified_bootstrap_ci because the message a caller of this
    # function needs names ``n``, not ``data``.
    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(idx):
        raise InvalidDataError(
            f"bootstrap_over_index: groups has {len(groups_arr)} entr(ies) but n={int(n)} "
            f"rows are being resampled. The strata must describe the same rows as the "
            f"index this function builds, otherwise the interval is estimated from a "
            f"truncated slice of the data while the point estimate uses all of it."
        )
    # ... AND THE SAME ARRAY MUST COVER EVERY ROW, not merely have the right count.
    # A groups array of exactly n entries whose strata leave rows out walked past the
    # length check above; see the block above _absent_stratum_labels.
    _refusal = _strata_coverage_refusal("bootstrap_over_index", groups_arr, int(n))
    if _refusal is not None:
        raise InvalidDataError(_refusal)
    # The stratified path does not implement 'bca'; fail closed with a clear
    # error rather than silently degrading, and narrow the type for the call.
    if method == "bca":
        raise ConfigurationError(
            "method='bca' is not supported for the stratified (grouped) bootstrap; "
            "use 'percentile' or 'basic'"
        )
    return stratified_bootstrap_ci(
        idx,
        groups_arr,
        lambda resampled_idx, _resampled_groups: statistic(resampled_idx),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
    )


# Bayesian Credible Intervals


def bayesian_proportion_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> StatisticalResult:
    """
    Compute Bayesian credible interval for a proportion.

    Uses Beta-Binomial conjugate prior for exact posterior inference.
    Default prior is uniform (Beta(1,1)).

    Recommended for small samples (n < 30) where bootstrap may be unreliable.

    Args:
        successes: Number of successes (positive outcomes)
        trials: Total number of trials
        confidence_level: Credible level (e.g., 0.95)
        prior_alpha: Beta prior alpha parameter (default: 1 for uniform)
        prior_beta: Beta prior beta parameter (default: 1 for uniform)

    Returns:
        StatisticalResult with credible interval

    Note:
        For weakly informative prior, use Beta(0.5, 0.5) - Jeffreys prior
        For uninformative prior, use Beta(1, 1) - uniform prior

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).

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

    Ledger row: bayesian_proportion_ci. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    confidence_level = validate_confidence_level(confidence_level)

    # BGL-S2 (2026-09-16). Two refusals, in order.
    #
    # (1) The count pair was never validated, so bayesian_proportion_ci(11, 4)
    #     returned point_estimate=2.0 with sample_size=4: a "proportion" above 1
    #     from a posterior with a NEGATIVE beta parameter. That is a caller bug,
    #     not a could-not-check, so it fails loudly.
    if trials < 0 or successes < 0 or successes > trials:
        raise ConfigurationError(
            f"bayesian_proportion_ci: successes={successes}, trials={trials} is not a "
            f"valid count pair (need 0 <= successes <= trials)"
        )

    # (2) With trials == 0 the posterior IS the prior, and the prior mean was
    #     returned in the same field and the same shape as a measured selection
    #     rate: point_estimate=0.5, a 95% interval of [0.025, 0.975], and
    #     is_significant=True grading it. Only sample_size=0, one field of
    #     fifteen in to_dict(), distinguished it from a measurement. Nothing was
    #     observed, so nothing is reported: NaN across the estimate and both
    #     bounds (which is also what makes is_significant answer False rather
    #     than grading the prior), with the reason carried in metadata so a
    #     caller can tell could-not-check from measured WITHOUT reading source.
    if trials == 0:
        warnings.warn(
            "bayesian_proportion_ci: 0 trials, so the posterior IS the prior and "
            "nothing was observed. Returning point_estimate=nan and NaN bounds "
            f"(could not check), not the prior mean of "
            f"{prior_alpha / (prior_alpha + prior_beta)}.",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=float("nan"),
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            interval_type=IntervalType.CREDIBLE,
            confidence_level=confidence_level,
            method="bayesian_beta_binomial",
            sample_size=0,
            standard_error=float("nan"),
            metadata={
                "prior": f"Beta({prior_alpha}, {prior_beta})",
                "posterior": f"Beta({prior_alpha}, {prior_beta})",
                "could_not_check": True,
                "could_not_check_reason": (
                    "0 trials: the posterior equals the prior, so no proportion was observed"
                ),
            },
        )

    # Posterior parameters
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)

    # Point estimate (posterior mean)
    point_estimate = post_alpha / (post_alpha + post_beta)

    # Credible interval from Beta quantiles
    alpha = 1 - confidence_level
    lower = _beta_ppf(alpha / 2, post_alpha, post_beta)
    upper = _beta_ppf(1 - alpha / 2, post_alpha, post_beta)

    # Posterior standard deviation
    var = (post_alpha * post_beta) / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
    se = np.sqrt(var)

    return StatisticalResult(
        point_estimate=point_estimate,
        lower_bound=lower,
        upper_bound=upper,
        interval_type=IntervalType.CREDIBLE,
        confidence_level=confidence_level,
        method="bayesian_beta_binomial",
        sample_size=trials,
        standard_error=se,
        metadata={
            "prior": f"Beta({prior_alpha}, {prior_beta})",
            "posterior": f"Beta({post_alpha}, {post_beta})",
        },
    )


def bayesian_difference_ci(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    confidence_level: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 10000,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute Bayesian credible interval for difference between two proportions.

    Uses Monte Carlo sampling from posterior distributions.
    Ideal for comparing rates between small groups.

    Args:
        successes1: Successes in group 1
        trials1: Trials in group 1
        successes2: Successes in group 2
        trials2: Trials in group 2
        confidence_level: Credible level
        prior_alpha: Beta prior alpha
        prior_beta: Beta prior beta
        n_samples: Monte Carlo samples
        random_state: Random seed

    Returns:
        StatisticalResult with credible interval for (p1 - p2), or a NaN estimate
        and NaN bounds when either group has no observations.

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1), or
            either (successes, trials) pair is not a valid count pair.

    BGL-G001 (2026-09-17). Neither count pair was validated and a group with
    ZERO observations was not refused, so the posterior for that group WAS THE
    PRIOR and its mean was differenced against a real measured rate. Measured at
    the public entry, before this change:

        bayesian_difference_ci(0, 0, 2, 100)
          -> point_estimate  0.4688      a 47-point "disparity"
             95% CI          [-0.0045, 0.9448]
             metadata        p_greater = 0.9703
             sample_size     100         group 1 contributed nothing, and only
                                         metadata['group1_size'] said so
        bayesian_difference_ci(0, 0, 0, 0)
          -> point_estimate -0.0001      "no gap", from no data at all
        bayesian_difference_ci(11, 4, 2, 10)
          -> ValueError: b <= 0          a raw numpy error from a negative
                                         posterior parameter

    0.47 is the Beta(1,1) prior mean of 0.5 minus group 2's measured 0.03. A
    reader gets a signed gap, a credible interval and a 97% posterior
    probability that group 1 is favoured, about a group nobody observed. The
    same two guards were added to :func:`bayesian_proportion_ci` on 2026-09-16
    (BGL-S2) and to :func:`risk_ratio` / :func:`odds_ratio` on 2026-09-17
    (BGL-S2b); this is that pair at the site they were missed, and it fails the
    same way each of those does: LOUDLY for an impossible table, NaN with a
    UserWarning for an empty one.
    """
    confidence_level = validate_confidence_level(confidence_level)

    for _label, _successes, _trials in (
        ("1", successes1, trials1),
        ("2", successes2, trials2),
    ):
        if _trials < 0 or _successes < 0 or _successes > _trials:
            raise ConfigurationError(
                f"bayesian_difference_ci: group {_label} has successes={_successes}, "
                f"trials={_trials}, which is not a valid count pair "
                f"(need 0 <= successes <= trials)"
            )

    if trials1 == 0 or trials2 == 0:
        warnings.warn(
            f"bayesian_difference_ci: group {'1' if trials1 == 0 else '2'} has no "
            f"observations (trials1={trials1}, trials2={trials2}), so its posterior "
            f"IS the prior and no gap was measured. Returning point_estimate=nan and "
            f"NaN bounds (could not check), not the "
            f"{prior_alpha / (prior_alpha + prior_beta)} prior mean differenced "
            f"against the other group's measured rate.",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=float("nan"),
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            interval_type=IntervalType.CREDIBLE,
            confidence_level=confidence_level,
            method="bayesian_difference_monte_carlo",
            sample_size=trials1 + trials2,
            n_bootstrap=0,
            standard_error=float("nan"),
            metadata={
                "group1_size": trials1,
                "group2_size": trials2,
                "p_greater": float("nan"),
                "could_not_check": True,
                "could_not_check_reason": (
                    "a group with 0 trials has no observed rate, so the difference "
                    "would be taken against the prior mean"
                ),
            },
        )

    rng = np.random.default_rng(random_state)

    # Posterior parameters
    post_alpha1 = prior_alpha + successes1
    post_beta1 = prior_beta + (trials1 - successes1)
    post_alpha2 = prior_alpha + successes2
    post_beta2 = prior_beta + (trials2 - successes2)

    # Sample from posteriors
    samples1 = rng.beta(post_alpha1, post_beta1, size=n_samples)
    samples2 = rng.beta(post_alpha2, post_beta2, size=n_samples)

    # Difference samples
    diff_samples = samples1 - samples2

    # Point estimate and interval
    point_estimate = float(np.mean(diff_samples))
    alpha = 1 - confidence_level
    lower = np.percentile(diff_samples, 100 * alpha / 2)
    upper = np.percentile(diff_samples, 100 * (1 - alpha / 2))
    se = float(np.std(diff_samples))

    # Probability that p1 > p2
    prob_greater = np.mean(diff_samples > 0)

    return StatisticalResult(
        point_estimate=point_estimate,
        lower_bound=lower,
        upper_bound=upper,
        interval_type=IntervalType.CREDIBLE,
        confidence_level=confidence_level,
        method="bayesian_difference_monte_carlo",
        sample_size=trials1 + trials2,
        n_bootstrap=n_samples,
        standard_error=se,
        metadata={"p_greater": prob_greater, "group1_size": trials1, "group2_size": trials2},
    )


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """
    Wilson score confidence interval for a single proportion.

    Closed-form, well-behaved for small and large n alike; the standard
    choice for per-group rate intervals (better coverage than the normal
    approximation, no resampling needed).

    Reference:
        Wilson (1927); Brown, Cai & DasGupta (2001), "Interval Estimation
        for a Binomial Proportion".

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).
    """
    confidence_level = validate_confidence_level(confidence_level)
    trials = int(trials)
    successes = int(successes)
    # BGL-G001 (2026-09-17). The count pair was never validated, so an impossible
    # table was answered with a "proportion" outside [0, 1] and an interval that
    # looked ordinary. Measured at the public entry, before this change:
    #     wilson_score_interval(11, 4) -> point_estimate 2.75, CI [0.0, 1.0]
    # (the bounds are 0 and 1 only because a NaN half-width from sqrt of a
    # negative number is clipped by max/min, so nothing in the result said the
    # input was impossible). Same guard, same wave and same wording as
    # bayesian_proportion_ci / risk_ratio / odds_ratio in this file.
    if trials < 0 or successes < 0 or successes > trials:
        raise ConfigurationError(
            f"wilson_score_interval: successes={successes}, trials={trials} is not a "
            f"valid count pair (need 0 <= successes <= trials)"
        )
    if trials <= 0:
        return StatisticalResult(
            point_estimate=np.nan,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method="wilson_score",
            sample_size=0,
        )
    p_hat = successes / trials
    z = _norm_ppf(1 - (1 - confidence_level) / 2)
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / trials + z2 / (4 * trials * trials))) / denom
    return StatisticalResult(
        point_estimate=p_hat,
        lower_bound=float(max(0.0, center - half)),
        upper_bound=float(min(1.0, center + half)),
        interval_type=IntervalType.CONFIDENCE,
        confidence_level=confidence_level,
        method="wilson_score",
        sample_size=trials,
    )


def bayesian_mean_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    prior_mean: float = 0.0,
    prior_std: float = 10.0,
    n_samples: int = 10000,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute Bayesian credible interval for a mean.

    Uses Normal-Normal conjugate prior with known variance approximation.

    Args:
        data: Data array
        confidence_level: Credible level
        prior_mean: Prior mean (default: 0)
        prior_std: Prior standard deviation (default: 10, weakly informative)
        n_samples: Monte Carlo samples
        random_state: Random seed

    Returns:
        StatisticalResult with credible interval

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).
    """
    confidence_level = validate_confidence_level(confidence_level)
    if len(data) < 2:
        return StatisticalResult(
            point_estimate=np.mean(data) if len(data) > 0 else np.nan,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CREDIBLE,
            confidence_level=confidence_level,
            method="bayesian_normal",
            sample_size=len(data),
            metadata={"warning": "Insufficient data"},
        )

    rng = np.random.default_rng(random_state)
    n = len(data)
    data_mean = np.mean(data)
    data_var = np.var(data, ddof=1)

    # BGL-G001 (2026-09-17). This estimator assumes a KNOWN variance and takes
    # the sample variance as that known value. A sample with no spread carries
    # no variance to assume, and the arithmetic then went two different ways for
    # the SAME data depending only on its length, because np.var of a constant
    # array is an ACCUMULATED statistic (the repo's standing lesson: it is
    # exactly 0.0 at some n and a tiny non-zero number at others). Measured at
    # the public entry, before this change, on a constant arm of 0.9:
    #
    #   n = 20  var 5.19e-32  -> point 0.9, 95% CI [0.8999999999999997,
    #                            0.8999999999999999], WIDTH 2.2e-16, and
    #                            is_significant True. A zero-width credible
    #                            interval is maximum certainty, and it was
    #                            produced from a sample with no information
    #                            about its own spread at all.
    #   n = 25  var 0.0       -> nan everywhere, announced by nothing but two
    #                            numpy RuntimeWarnings about dividing by zero.
    #
    # Constant arms are ordinary here: the pulse path scores LLM output at
    # temperature 0. Tested with np.ptp on the raw data, which is exact and
    # needs no epsilon, so both lengths now give the same honest refusal.
    if not np.all(np.isfinite(data)):
        warnings.warn(
            "bayesian_mean_ci: the data contains values that are not finite, so no "
            "mean and no interval were computed. Returning nan (could not check).",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=float("nan"),
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            interval_type=IntervalType.CREDIBLE,
            confidence_level=confidence_level,
            method="bayesian_normal",
            sample_size=n,
            standard_error=float("nan"),
            metadata={
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "could_not_check": True,
                "could_not_check_reason": "the sample contains non-finite values",
            },
        )
    if float(np.ptp(data)) == 0.0:
        warnings.warn(
            f"bayesian_mean_ci: every observation is the identical value "
            f"{float(data[0]):.6g}, so the sample carries no variance for this "
            f"known-variance posterior to use and no interval is estimable. "
            f"Returning point_estimate=nan and NaN bounds (could not check), NOT a "
            f"zero-width credible interval at that value, which would read as "
            f"certainty. The MEAN IS OBSERVED AND EXACT; what cannot be estimated "
            f"is how precisely it is known.",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=float("nan"),
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            interval_type=IntervalType.CREDIBLE,
            confidence_level=confidence_level,
            method="bayesian_normal",
            sample_size=n,
            standard_error=float("nan"),
            metadata={
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "observed_constant_value": float(data[0]),
                "could_not_check": True,
                "could_not_check_reason": (
                    "every observation is identical, so the sample variance this "
                    "estimator assumes as known is zero"
                ),
            },
        )

    # Posterior parameters (assuming known variance)
    prior_var = prior_std**2
    post_var = 1 / (1 / prior_var + n / data_var)
    post_mean = post_var * (prior_mean / prior_var + n * data_mean / data_var)
    post_std = np.sqrt(post_var)

    # Sample from posterior
    samples = rng.normal(post_mean, post_std, size=n_samples)

    # Credible interval
    alpha = 1 - confidence_level
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))

    return StatisticalResult(
        point_estimate=post_mean,
        lower_bound=lower,
        upper_bound=upper,
        interval_type=IntervalType.CREDIBLE,
        confidence_level=confidence_level,
        method="bayesian_normal",
        sample_size=n,
        n_bootstrap=n_samples,
        standard_error=post_std,
        metadata={"prior_mean": prior_mean, "prior_std": prior_std},
    )


# Multiple Comparison Corrections


def _testable(p_values: np.ndarray, method: str) -> Tuple[np.ndarray, int]:
    """Which entries are real p-values, and how many are not.

    A NaN entry is a test that DID NOT RUN. Leaving it in inflates the family
    size and so penalises the tests that did run: measured 2026-09-08 on
    [0.001, nan, 0.9], the real 0.001 adjusted to 0.003 against a family of 3,
    where the honest family of 2 gives 0.002. An untestable comparison was
    making genuine findings harder to detect.
    """
    tested = np.isfinite(p_values)
    n_not_tested = int((~tested).sum())
    if n_not_tested:
        warnings.warn(
            f"{method}: {n_not_tested} of {len(p_values)} p-values are not finite, so "
            f"those hypotheses were NOT tested and are excluded from the family size. "
            f"Their rejection_mask entries are False because nothing was rejected, NOT "
            f"because they were tested and passed; read tested_mask alongside it.",
            UserWarning,
            stacklevel=3,
        )
    return tested, n_not_tested


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestingResult:
    """
    Apply Bonferroni correction for multiple comparisons.

    Conservative method that controls Family-Wise Error Rate (FWER).
    Use when you need strong control of Type I errors.

    Args:
        p_values: Array of p-values from multiple tests
        alpha: Significance level

    Returns:
        MultipleTestingResult with adjusted p-values

    Note:
        Adjusted p-value = min(p * n, 1) where n is number of tests
        Very conservative - may have low power with many tests

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

    Ledger row: bonferroni_correction. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    p_values = np.asarray(p_values)
    p_values = np.asarray(p_values, dtype=float)
    tested, n_not_tested = _testable(p_values, "bonferroni_correction")
    n = int(tested.sum())

    adjusted = np.full(len(p_values), np.nan)
    rejected = np.zeros(len(p_values), dtype=bool)
    if n:
        adjusted[tested] = np.minimum(p_values[tested] * n, 1.0)
        rejected[tested] = adjusted[tested] < alpha

    return MultipleTestingResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejection_mask=rejected,
        method="bonferroni",
        significance_level=alpha,
        n_rejected=int(np.sum(rejected)),
        tested_mask=tested,
        n_not_tested=n_not_tested,
    )


def benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction.

    Less conservative than Bonferroni, controls expected proportion
    of false discoveries among rejected hypotheses.

    Recommended for exploratory fairness analysis where some false
    positives are acceptable.

    Args:
        p_values: Array of p-values from multiple tests
        alpha: Target FDR level

    Returns:
        MultipleTestingResult with adjusted p-values

    Reference:
        Benjamini & Hochberg (1995). "Controlling the False Discovery Rate"

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

    Ledger row: benjamini_hochberg_correction. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    p_values = np.asarray(p_values, dtype=float)
    tested, n_not_tested = _testable(p_values, "benjamini_hochberg_correction")
    # The family is the tests that RAN. np.argsort also puts NaN last, so an
    # untested entry used to take a rank in the step-up procedure as well.
    tested_idx = np.flatnonzero(tested)
    p_tested = p_values[tested_idx]
    n = len(p_tested)

    if n == 0:
        return MultipleTestingResult(
            original_p_values=p_values,
            adjusted_p_values=np.full(len(p_values), np.nan),
            rejection_mask=np.zeros(len(p_values), dtype=bool),
            method="benjamini_hochberg",
            significance_level=alpha,
            n_rejected=0,
            tested_mask=tested,
            n_not_tested=n_not_tested,
        )

    # Sort p-values and get original order
    sorted_idx = np.argsort(p_tested)
    sorted_p = p_tested[sorted_idx]

    # Benjamini-Hochberg adjusted p-values
    adjusted = np.empty(n)
    cummin_val = 1.0

    for i in range(n - 1, -1, -1):
        # Adjusted p_i = min(p_{(i)} * n / i, p_{(i+1)})
        rank = i + 1
        adj_p = sorted_p[i] * n / rank
        cummin_val = min(adj_p, cummin_val)
        adjusted[sorted_idx[i]] = min(cummin_val, 1.0)

    full_adjusted = np.full(len(p_values), np.nan)
    full_adjusted[tested_idx] = adjusted
    rejected = np.zeros(len(p_values), dtype=bool)
    rejected[tested_idx] = adjusted < alpha

    return MultipleTestingResult(
        original_p_values=p_values,
        adjusted_p_values=full_adjusted,
        rejection_mask=rejected,
        method="benjamini_hochberg",
        significance_level=alpha,
        n_rejected=int(np.sum(rejected)),
        tested_mask=tested,
        n_not_tested=n_not_tested,
    )


def apply_multiple_testing_correction(
    p_values: np.ndarray,
    method: Literal["bonferroni", "fdr", "benjamini_hochberg", "none"] = "fdr",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """
    Apply multiple testing correction with method selection.

    Args:
        p_values: Array of p-values
        method: Correction method
            - 'bonferroni': Conservative FWER control
            - 'fdr' or 'benjamini_hochberg': FDR control (recommended)
            - 'none': No correction (not recommended)
        alpha: Significance level

    Returns:
        MultipleTestingResult
    """
    if method == "bonferroni":
        return bonferroni_correction(p_values, alpha)
    elif method in ("fdr", "benjamini_hochberg"):
        return benjamini_hochberg_correction(p_values, alpha)
    elif method == "none":
        # `method='none'` means NO CORRECTION, not "no bookkeeping". This branch
        # skipped `_testable` entirely, so a family holding a NaN answered
        # tested_mask=None and n_not_tested=0 with no warning, while the other two
        # methods on the identical input answered tested_mask=[True, False, True],
        # n_not_tested=1 and warned. Measured 2026-09-10 on [0.001, nan, 0.9].
        # A NaN entry is a hypothesis that DID NOT RUN whichever correction is
        # asked for, and `nan < alpha` is False, so its rejection_mask entry read
        # as "tested and not rejected" with nothing to contradict it.
        p_values = np.asarray(p_values, dtype=float)
        tested, n_not_tested = _testable(p_values, "apply_multiple_testing_correction(none)")
        rejected = np.zeros(len(p_values), dtype=bool)
        rejected[tested] = p_values[tested] < alpha
        adjusted = p_values.copy()
        adjusted[~tested] = np.nan
        return MultipleTestingResult(
            original_p_values=p_values,
            adjusted_p_values=adjusted,
            rejection_mask=rejected,
            method="none",
            significance_level=alpha,
            n_rejected=int(np.sum(rejected)),
            tested_mask=tested,
            n_not_tested=n_not_tested,
        )
    else:
        raise ConfigurationError(f"Unknown correction method: {method}")


# Effect Size Calculations


#: Spread, RELATIVE to the data's own magnitude, below which an array counts as
#: constant for the degeneracy guard in :func:`cohens_d`.
#:
#: Never an equality with zero. BGL5 A-evaluation-3, 2026-09-29: the guard below
#: was ``float(np.ptp(group1)) == 0.0 and float(np.ptp(group2)) == 0.0``, which is
#: exact for an array the CALLER made constant and false for one the caller
#: COMPUTED. ``compute_regression_effect_sizes`` forms its residual inside the
#: unit as ``y_true[mask] - y_pred[mask]``, so a model whose error is constant
#: within a group yields an array constant only up to rounding. Measured on 80
#: rows in two groups of 40, error exactly 0.0 for one and exactly 50.0 for the
#: other::
#:
#:     group a residual: 1 distinct value,  ptp 0.0
#:     group b residual: 3 distinct values, ptp 2.842170943040401e-14
#:                       (= one ulp at the magnitude of y_true, ~150)
#:     np.std(b, ddof=1) = 1.0176604434369563e-14
#:     cohens_d(a, b)    = -6948356750493589.0, ZERO warnings
#:
#: which is the same magnitude class (-4.3e15 at n=20) the block comment inside
#: the function records as the defect it was written to remove. 1e-12 of the
#: largest magnitude in play is about four thousand float64 ulps near 1.0, i.e.
#: rounding noise, and it is the value ``operations/monitoring/drift.py`` already
#: uses for the same question on a metric stream
#: (``_CONSTANT_STREAM_REL_TOL``). A group with any real variance clears it by
#: orders of magnitude: the two groups of 40 above measure a relative spread of
#: 0.44 and 0.42 on their PREDICTIONS, which keep their measured d of -1.5275.
#:
#: This is relative to the magnitude of the values HANDED IN, which is the wrong
#: scale when they were computed by subtracting two much larger arrays (a
#: residual of 50 formed from operands at 1e6 carries ~1.2e-10 of rounding, about
#: 2.3e-12 relative). A caller that COMPUTED its arrays therefore passes
#: ``constant_atol`` derived from its own operands; see
#: ``compute_regression_effect_sizes``.
_CONSTANT_SPREAD_REL_TOL = 1e-12


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
    constant_atol: Optional[float] = None,
) -> float:
    """
    Compute Cohen's d effect size for difference between two groups.

    Provides standardized measure of the magnitude of difference,
    independent of sample size.

    Args:
        group1: Data for group 1
        group2: Data for group 2
        pooled: Use pooled standard deviation (default: True)
        constant_atol: Absolute spread below which a group counts as constant,
            for a caller that COMPUTED its arrays and therefore knows the
            resolution of the arithmetic that produced them. Default None
            derives it from the magnitude of the values handed in, at
            :data:`_CONSTANT_SPREAD_REL_TOL`.

    Returns:
        Cohen's d value, or NaN when it could not be measured: fewer than two
        observations in a group, or both groups constant (to within
        ``constant_atol``) with different means, where the standardising
        denominator is zero and the effect has no size in standard deviations.
        NaN, never a large finite number, which every ``isfinite`` guard
        downstream passes and an interpretation table labels "large"

    Interpretation:
        |d| < 0.2: Negligible
        0.2 ≤ |d| < 0.5: Small
        0.5 ≤ |d| < 0.8: Medium
        |d| ≥ 0.8: Large
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    # READINESS-6, 2026-09-10. THREE implementations of Cohen's d existed and
    # answered differently on the same input. `agents.action_bias._cohens_d`
    # was fixed on 2026-09-08 to refuse two constant-but-separated arms; this
    # one, the SHARED one in the statistics module, never was, and it is the
    # copy the experimentation lane calls.
    #
    # Measured across the three, two constant arms separated by 0.7:
    #
    #   n=5   _statistics: inf        agents: nan        llm: nan
    #   n=20  _statistics: -4.3e15    agents: -4.3e15    llm: nan
    #
    # Both of the first two miss at n=20 for the same reason: their guard was an
    # EXACT float test on an accumulated variance, and a constant array of a
    # value that is not exactly representable has a tiny non-zero variance at
    # some lengths and exactly zero at others. -4.3e15 is FINITE, so every
    # `math.isfinite` guard downstream passes it through and an interpretation
    # table labels it "large".
    #
    # `np.ptp` on the raw arrays is exact and needs no epsilon. Zero variance in
    # BOTH arms with equal means is a genuine zero effect and is preserved;
    # separated constant arms have an undefined standardised effect, and NaN
    # says so where inf would be compared against a threshold and where a
    # clamped magnitude would be graded.
    # BGL5 A-evaluation-3, 2026-09-29. `np.ptp` is exact, and the EQUALITY WITH
    # ZERO this test used to be was not: it asked whether the array is constant in
    # the last bit, which only a caller-supplied constant array is. See
    # _CONSTANT_SPREAD_REL_TOL above for the measured 6.9e15 that walked past it,
    # and never test an accumulated or computed statistic for exact equality with
    # zero. The tolerance is data-scaled, so it carries no units and does not
    # depend on what the values are recorded in.
    spread1 = float(np.ptp(group1))
    spread2 = float(np.ptp(group2))
    magnitude = max(abs(float(mean1)), abs(float(mean2)), spread1, spread2)
    # An all-zero pair has magnitude 0.0, and then this is `spread <= 0.0`,
    # the exact test, correctly, for the one case where exact is what the
    # data says.
    atol = _CONSTANT_SPREAD_REL_TOL * magnitude
    if constant_atol is not None:
        # A CALLER'S TOLERANCE IS A FLOOR, NOT A REPLACEMENT.
        # B4 tier-1 audit, 2026-09-30. `constant_atol` used to REPLACE the relative
        # bound above, which silently REMOVED coverage this function already had
        # whenever the caller's bound was the tighter of the two: four ulps is about
        # 9e-16 of a magnitude while the default is 1e-12 of it, so a caller whose
        # operands sit at the same magnitude as the values handed in made the test a
        # thousand times stricter than the default by trying to loosen it. Both are
        # lower bounds on what counts as constant, so the answer is the larger, and
        # a caller can only ever widen the refusal and never narrow it.
        atol = max(atol, abs(float(constant_atol)))
    constant_both = spread1 <= atol and spread2 <= atol
    if constant_both:
        # Means equal, or apart by no more than the same rounding noise that made
        # the arrays count as constant: a genuine zero effect, preserved.
        if mean1 == mean2 or abs(float(mean1) - float(mean2)) <= atol:
            # THE ZERO IS REAL BUT IT IS A ZERO AT THE RESOLUTION OF THE ARITHMETIC,
            # and it used to be SILENT. B4 tier-1 audit, 2026-09-30. Measured on a
            # prediction pair each constant to 7.5e-9 with a mean gap of 5e-8 inside
            # a 5.96e-8 tolerance: d = 0.0, graded "negligible", zero warnings, which
            # a reader cannot tell apart from a zero effect measured across two
            # varying groups. The VALUE is kept, deliberately (there is no separation
            # this arithmetic can resolve, so 0.0 is the honest number and nan would
            # discard it), and the disclosure is added, because a could-not-resolve
            # collapsed into a measured "negligible" is the same substitution this
            # whole guard exists to stop.
            if spread1 > 0.0 or spread2 > 0.0 or float(mean1) != float(mean2):
                warnings.warn(
                    f"cohens_d: both groups are constant to within {atol:.3g}, which is "
                    f"rounding noise at this magnitude (peak-to-peak {spread1:.3g} and "
                    f"{spread2:.3g}), and their means differ by "
                    f"{abs(float(mean1) - float(mean2)):.3g}, which is inside the same "
                    f"noise. Returning 0.0: there is no separation this arithmetic can "
                    f"resolve, so it is a zero AT THE RESOLUTION OF THE NUMBERS and NOT "
                    f"an effect size measured across two varying groups.",
                    UserWarning,
                    stacklevel=2,
                )
            return 0.0
        warnings.warn(
            f"cohens_d: both groups are constant (peak-to-peak {spread1:.3g} and "
            f"{spread2:.3g}, at or below the {atol:.3g} that is rounding noise at this "
            f"magnitude) and differ in mean by "
            f"{float(mean1 - mean2):.6g}, so the standardising denominator is zero and "
            f"Cohen's d is undefined. Returning nan, NOT 0.0 (which would read as no "
            f"effect for perfect separation) and NOT a large finite number (which the "
            f"previous exact-zero guard produced whenever it missed, including for an "
            f"array a caller COMPUTED rather than supplied). The SEPARATION IS "
            f"REAL; only its size in standard deviations is undefined, because there "
            f"are none.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    if pooled:
        # Pooled standard deviation
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std
    else:
        # Simple approach using average std
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        avg_std = (std1 + std2) / 2
        return (mean1 - mean2) / avg_std


def risk_ratio(events1: int, total1: int, events2: int, total2: int) -> Tuple[float, float, float]:
    """
    Compute risk ratio (relative risk) with confidence interval.

    For binary outcomes, measures how many times more likely
    an event is in group 1 compared to group 2.

    Args:
        events1: Number of events in group 1
        total1: Total in group 1
        events2: Number of events in group 2
        total2: Total in group 2

    Returns:
        Tuple of (risk_ratio, lower_95_ci, upper_95_ci)

    Interpretation:
        RR = 1: No difference
        RR > 1: Higher risk in group 1
        RR < 1: Lower risk in group 1
    """
    # BGL-S2b (2026-09-17). The count pair was never validated, so an
    # IMPOSSIBLE table was answered with a confident measurement and a
    # confidence interval around it:
    #
    #     risk_ratio(11, 4, 2, 10) -> (13.75, 5.25, 35.98)
    #     odds_ratio(11, 4, 2, 10) -> (-6.29, nan, nan)
    #
    # 11 events in 4 trials is a caller bug, not a could-not-check, and -6.29
    # is not even in the range of an odds ratio, which is confined to
    # [0, inf). The identical `0 <= successes <= trials` guard was added to
    # bayesian_proportion_ci in the same file and the same wave; this is that
    # guard at the two sites that were missed. It fails LOUDLY, as that one
    # does, because there is no measurement to refuse: the input describes
    # nothing that can exist.
    for _label, _events, _total in (
        ("1", events1, total1),
        ("2", events2, total2),
    ):
        if _total < 0 or _events < 0 or _events > _total:
            raise ConfigurationError(
                f"risk_ratio: group {_label} has events={_events}, "
                f"total={_total}, which is not a valid count pair "
                f"(need 0 <= events <= total)"
            )

    # H-04. A group with ZERO observations has no risk to compare, and every
    # branch below used to answer with a number that means something specific:
    #
    #   risk_ratio(0, 0, 0, 0) -> 1.0    this function's own docstring, three
    #                                    lines up, defines RR = 1 as "No difference"
    #   risk_ratio(5, 10, 0, 0) -> inf   and inf beats every threshold, so an
    #                                    unmeasurable group becomes the WORST
    #                                    breach in the report
    #   risk_ratio(0, 0, 5, 10) -> 0.0
    #
    # An empty denominator is not a rate of zero. NaN, matching bootstrap_ci in
    # this same module, which returns nan/nan/nan rather than inventing a point
    # estimate when it has nothing to resample.
    if total1 <= 0 or total2 <= 0:
        warnings.warn(
            f"risk_ratio: group {'1' if total1 <= 0 else '2'} has no observations "
            f"(total1={total1}, total2={total2}), so no ratio was computed. "
            "Returning NaN rather than 1.0, inf or 0.0, each of which reads as a "
            "measured result.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan, np.nan, np.nan

    p1 = events1 / total1
    p2 = events2 / total2

    if p2 == 0:
        # A measured zero risk in group 2 with events in group 1 is a genuine
        # infinite ratio; with no events anywhere it is 0/0, which is undefined.
        if p1 > 0:
            return np.inf, np.nan, np.nan
        return np.nan, np.nan, np.nan

    rr = p1 / p2

    # Log-scale confidence interval
    if events1 > 0 and events2 > 0:
        # BGL-S2 (2026-09-16). Each Katz variance term is (1/events - 1/total),
        # which is exactly zero when a group is SATURATED (every observation is
        # an event). With both arms saturated the standard error is zero and the
        # interval collapses to [rr, rr]: a zero-width 95% CI, i.e. perfect
        # certainty, from a table that carries none. Measured at the public
        # entry: risk_ratio(30, 30, 30, 30) -> (1.0, 1.0, 1.0).
        #
        # Tested on the STRUCTURE (events == total in both arms), never on
        # `se_log_rr == 0.0`: se_log_rr is an accumulated statistic and exact
        # equality with zero is not a reliable test of it.
        both_saturated = events1 >= total1 and events2 >= total2
        log_rr = np.log(rr)
        se_log_rr = np.sqrt((1 / events1) - (1 / total1) + (1 / events2) - (1 / total2))

        if both_saturated or not np.isfinite(se_log_rr):
            warnings.warn(
                f"risk_ratio: every observation is an event in both groups "
                f"({events1}/{total1} and {events2}/{total2}), so the log-scale "
                f"standard error is zero and no interval is estimable. Returning "
                f"NaN bounds rather than the zero-width interval [{rr}, {rr}], "
                f"which would read as certainty.",
                UserWarning,
                stacklevel=2,
            )
            lower, upper = np.nan, np.nan
        else:
            lower = np.exp(log_rr - 1.96 * se_log_rr)
            upper = np.exp(log_rr + 1.96 * se_log_rr)
    else:
        lower, upper = np.nan, np.nan

    return rr, lower, upper


def odds_ratio(events1: int, total1: int, events2: int, total2: int) -> Tuple[float, float, float]:
    """
    Compute odds ratio with confidence interval.

    For binary outcomes, compares odds of event between groups.
    Often used when risk ratio is not appropriate (e.g., case-control studies).

    Args:
        events1: Number of events in group 1
        total1: Total in group 1
        events2: Number of events in group 2
        total2: Total in group 2

    Returns:
        Tuple of (odds_ratio, lower_95_ci, upper_95_ci)

    Interpretation:
        OR = 1: No difference
        OR > 1: Higher odds in group 1
        OR < 1: Lower odds in group 1
    """
    # BGL-S2b (2026-09-17). The count pair was never validated, so an
    # IMPOSSIBLE table was answered with a confident measurement and a
    # confidence interval around it:
    #
    #     risk_ratio(11, 4, 2, 10) -> (13.75, 5.25, 35.98)
    #     odds_ratio(11, 4, 2, 10) -> (-6.29, nan, nan)
    #
    # 11 events in 4 trials is a caller bug, not a could-not-check, and -6.29
    # is not even in the range of an odds ratio, which is confined to
    # [0, inf). The identical `0 <= successes <= trials` guard was added to
    # bayesian_proportion_ci in the same file and the same wave; this is that
    # guard at the two sites that were missed. It fails LOUDLY, as that one
    # does, because there is no measurement to refuse: the input describes
    # nothing that can exist.
    for _label, _events, _total in (
        ("1", events1, total1),
        ("2", events2, total2),
    ):
        if _total < 0 or _events < 0 or _events > _total:
            raise ConfigurationError(
                f"odds_ratio: group {_label} has events={_events}, "
                f"total={_total}, which is not a valid count pair "
                f"(need 0 <= events <= total)"
            )

    # H-04, same shape as risk_ratio above. With no observations in a group
    # there is no odds to compare, and inf / 0.0 are both meaningful answers
    # that would be read as measurements.
    if total1 <= 0 or total2 <= 0:
        warnings.warn(
            f"odds_ratio: group {'1' if total1 <= 0 else '2'} has no observations "
            f"(total1={total1}, total2={total2}), so no ratio was computed.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan, np.nan, np.nan

    nonevents1 = total1 - events1
    nonevents2 = total2 - events2

    # BGL-S2 (2026-09-16). The odds ratio is
    #     (events1 * nonevents2) / (nonevents1 * events2)
    # so the direction of a degenerate table is decided by BOTH arms, never by
    # one of them. The old test `if nonevents1 == 0 or events2 == 0: return inf`
    # fired on a table where group 2 was saturated too, making the true quantity
    # inf/inf, which is UNDEFINED, not infinite. Measured at the public entry:
    #     odds_ratio(29, 30, 29, 30) -> (1.0, 0.06, 16.8)
    #     odds_ratio(30, 30, 30, 30) -> (inf, nan, nan)
    #     odds_ratio( 0, 30,  0, 30) -> (inf, nan, nan)
    # i.e. the reported ratio jumped from 1.0 to INFINITY as the two groups
    # became MORE equal, and a reader following this docstring ("OR > 1: Higher
    # odds in group 1") was told two identical groups differ infinitely.
    # Deciding on the product terms keeps every genuinely one-sided table:
    # inf when only the denominator vanishes, 0.0 when only the numerator does,
    # and NaN (could not check) only for the 0/0 table.
    or_numerator = events1 * nonevents2
    or_denominator = nonevents1 * events2

    if or_numerator == 0 and or_denominator == 0:
        warnings.warn(
            f"odds_ratio: both groups are degenerate in the same direction "
            f"(events1={events1}/{total1}, events2={events2}/{total2}), so the "
            f"odds ratio is 0/0 and undefined. Returning NaN rather than inf or "
            f"0.0, each of which reads as a measured direction.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan, np.nan, np.nan
    if or_denominator == 0:
        return np.inf, np.nan, np.nan
    if or_numerator == 0:
        return 0.0, np.nan, np.nan

    odds1 = events1 / nonevents1
    odds2 = events2 / nonevents2
    or_val = odds1 / odds2

    # Log-scale confidence interval (Woolf method)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1 / events1 + 1 / nonevents1 + 1 / events2 + 1 / nonevents2)

    lower = np.exp(log_or - 1.96 * se_log_or)
    upper = np.exp(log_or + 1.96 * se_log_or)

    return or_val, lower, upper


def interpret_effect_size(d: float, effect_type: str = "cohens_d") -> str:
    """
    Provide interpretation of effect size magnitude.

    Args:
        d: Effect size value
        effect_type: Type of effect size measure

    Returns:
        String interpretation
    """
    # `abs(nan) < 0.2` is False, and so is every later threshold, so an
    # unmeasurable effect size fell all the way through to "large". `d > 0` is
    # also False for NaN, so it picked a direction too. Measured 2026-09-08:
    # interpret_effect_size(nan) returned "large effect (lower in group 1)", the
    # most alarming sentence this function can produce, about nothing at all.
    # The ratio branch printed "group 1 has nanx lower risk".
    if not np.isfinite(d):
        warnings.warn(
            f"interpret_effect_size: the effect size is {d}, which is not a finite "
            f"number, so it cannot be graded. Returning a not-interpretable string "
            f"rather than a magnitude.",
            UserWarning,
            stacklevel=2,
        )
        return f"not interpretable (effect size is {d}, not a measured value)"

    d_abs = abs(d)

    if effect_type == "cohens_d":
        if d_abs < 0.2:
            magnitude = "negligible"
        elif d_abs < 0.5:
            magnitude = "small"
        elif d_abs < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        if d == 0:
            # EXACTLY zero has no direction, and `"higher" if d > 0 else "lower"`
            # called it "lower in group 1". A direction is a claim about which
            # group is worse off, and there is none here: the groups are equal on
            # this statistic. Measured 2026-09-25: interpret_effect_size(0.0) ->
            # "negligible effect (lower in group 1)".
            return f"{magnitude} effect (the groups are equal on this statistic)"
        direction = "higher" if d > 0 else "lower"
        return f"{magnitude} effect ({direction} in group 1)"

    elif effect_type in ("risk_ratio", "odds_ratio"):
        if 0.9 <= d <= 1.1:
            return "negligible difference"
        elif d <= 0:
            # BGL-G001 (2026-09-17). The `else` branch below computes 1 / d, and
            # a ratio of exactly 0.0 is a MEASURED result that `odds_ratio` in
            # this same module returns by design (its `or_numerator == 0` branch,
            # e.g. odds_ratio(0, 30, 5, 30) -> (0.0, nan, nan)), so
            # interpret_effect_size(0.0, "odds_ratio") raised ZeroDivisionError
            # on a real measurement. The reading is not "no effect": zero odds in
            # group 1 against positive odds in group 2 is total separation, the
            # strongest finding the table can carry, and it must not be lost to a
            # crash or reported as negligible.
            noun = "risk" if effect_type == "risk_ratio" else "odds"
            return f"group 1 has zero {noun} while group 2 does not (total separation)"
        elif d > 1:
            return f"group 1 has {d:.2f}x {'higher risk' if effect_type == 'risk_ratio' else 'higher odds'}"
        else:
            return f"group 1 has {1 / d:.2f}x {'lower risk' if effect_type == 'risk_ratio' else 'lower odds'}"

    return f"effect size = {d:.3f}"


# Utility Functions for Statistical Distributions


def _norm_ppf(p: float) -> float:
    """
    Inverse CDF of standard normal distribution.

    Simple approximation to avoid scipy dependency.
    Uses Abramowitz and Stegun approximation.
    """
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p == 0.5:
        return 0.0

    # Use symmetry
    if p > 0.5:
        return -_norm_ppf(1 - p)

    # Rational approximation for lower tail
    t = np.sqrt(-2 * np.log(p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


def _norm_cdf(x: float) -> float:
    """
    CDF of standard normal distribution.

    Uses error function approximation.
    """
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """Error function approximation."""
    # Save the sign
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def _beta_ppf(p: float, a: float, b: float, tol: float = 1e-8) -> float:
    """
    Inverse CDF of Beta distribution.

    Delegates to ``scipy.stats.beta.ppf`` (scipy is a hard core dependency).
    The homegrown Newton-Raphson below remains only as a last-resort fallback:
    it is numerically unreliable for skewed shapes, e.g. it returned 0.9999
    for ``ppf(0.025, a=3, b=17)`` where the true value is 0.0338, collapsing
    "95%" credible intervals to ~55% actual coverage.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    try:
        from scipy.stats import beta as _scipy_beta

        return float(_scipy_beta.ppf(p, a, b))
    except ImportError:  # pragma: no cover - scipy is a core dependency
        import warnings as _warnings

        _warnings.warn(
            "scipy unavailable: falling back to an approximate Beta PPF; "
            "Bayesian credible intervals may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Initial guess using normal approximation
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    x = mean + np.sqrt(var) * _norm_ppf(p)
    x = np.clip(x, 0.001, 0.999)

    # Newton-Raphson iterations
    for _ in range(100):
        cdf_x = _beta_cdf(x, a, b)
        pdf_x = _beta_pdf(x, a, b)

        if pdf_x < 1e-10:
            break

        dx = (cdf_x - p) / pdf_x
        x = x - dx
        x = np.clip(x, 0.0001, 0.9999)

        if abs(dx) < tol:
            break

    return x


def _beta_cdf(x: float, a: float, b: float) -> float:
    """
    CDF of Beta distribution using incomplete beta function.

    Uses continued fraction expansion.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use regularized incomplete beta function
    return _regularized_beta(x, a, b)


def _beta_pdf(x: float, a: float, b: float) -> float:
    """PDF of Beta distribution."""
    if x <= 0 or x >= 1:
        return 0.0

    log_pdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
    log_pdf -= _log_beta(a, b)

    return np.exp(log_pdf)


def _log_beta(a: float, b: float) -> float:
    """Log of Beta function."""
    return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


def _log_gamma(x: float) -> float:
    """Log-gamma function (Lanczos approximation)."""
    if x <= 0:
        return np.inf

    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _log_gamma(1 - x)

    x -= 1
    s = c[0]
    for i in range(1, g + 2):
        s += c[i] / (x + i)

    t = x + g + 0.5
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(s)


def _regularized_beta(x: float, a: float, b: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).

    Uses continued fraction expansion.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry if needed
    if x > (a + 1) / (a + b + 2):
        return 1 - _regularized_beta(1 - x, b, a)

    # Continued fraction
    front = np.exp(a * np.log(x) + b * np.log(1 - x) - _log_beta(a, b)) / a

    f = 1.0
    c = 1.0
    d = 0.0

    for m in range(1, 200):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1 + num * d
        c = 1 + num / c
        d = 1 / d
        f *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        c = 1 + num / c
        d = 1 / d
        delta = d * c
        f *= delta

        if abs(delta - 1) < 1e-10:
            break

    return front * f


# High-Level Interface for Fairness Metrics with Statistical Validation


def _permutation_count(alpha: float) -> Tuple[int, bool]:
    """How many permutations the exactness gate wants, and whether the cap bound.

    The gate rejects at alpha/2, and the smallest p-value reachable from n
    permutations is 1/(n+1), so n must exceed 2/alpha to be able to reject at
    all; the 4/alpha here keeps a factor of two of headroom.

    ``MAX_PERMUTATIONS`` caps it. That ceiling matters because the count is
    driven by a caller-supplied confidence_level: 0.9999 asks for 40,001 full
    metric recomputations and 0.99999 for 400,001, with no progress output.
    When the cap binds, the run keeps the design's headroom only until
    1/(n+1) reaches alpha/2; past that the gate can no longer reject at all and
    the lower bound stays at the fold (0.0). Either way the effect is a WIDER,
    more conservative interval, never a narrower one -- but it is no longer the
    exact test the method name implies, so the caller is told: the returned flag
    drives ``permutation_resolution_limited`` in the result metadata, alongside
    ``min_resolvable_p_value`` so a reader can compare it to alpha/2 themselves.

    Returns (n_permutations, capped).
    """
    wanted = max(199, int(np.ceil(4.0 / max(alpha, 1e-6))))
    return min(wanted, MAX_PERMUTATIONS), wanted > MAX_PERMUTATIONS


def compute_metric_with_ci(
    metric_func: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
    null_value: float = 0.0,
    **metric_kwargs,
) -> StatisticalResult:
    """
    Compute a fairness metric with a confidence interval.

    Method selection (F18, documented 2026-09-09). Only a stratified bootstrap
    is implemented. ``method='auto'`` runs it with ``n_bootstrap`` resamples
    when every group has at least ``SMALL_SAMPLE_THRESHOLD`` rows and with
    twice that many when any group is smaller (and warns that the sample is
    small). ``method='bayesian'`` is accepted for API compatibility but is
    NOT implemented: it warns and runs the same doubled-resample bootstrap.
    In every case ``result.method`` names the bootstrap that actually ran
    (for example ``'stratified_bootstrap_fold_debiased'``); it never reports
    'bayesian'. A genuine Bayesian interval for a two-group proportion gap is
    available separately from :func:`bayesian_difference_ci`.

    Interval construction (changed 2026-08-22): for a FOLDED, nonnegative
    statistic anchored at zero (max-min spreads; detected when the point
    estimate and every resample are nonnegative AND the permutation-null
    distribution of the metric sits within one support-width of 0) the
    interval is fold-debiased: the lower bound is 0 unless a permutation
    test of the sensitive attribute rejects the exchangeability null at
    level alpha/2 (then the clipped basic lower bound), and the upper bound
    is the wider of the percentile and basic upper bounds. Nonnegative
    statistics whose null value is elsewhere (min/max ratios with parity at
    1, level statistics) keep the percentile interval, as do signed
    statistics. Previously a raw percentile interval was used, which for
    these metrics had 0% coverage at the parity null and reported
    is_significant=True for every perfectly fair model (a percentile CI
    inherits the plug-in bias of the max statistic and cannot reach the
    boundary; Efron & Tibshirani 1993, ch. 10). Signed statistics keep the
    percentile interval. The permutation p-value, when computed, is exposed
    in ``metadata['permutation_p_value']``.

    Args:
        metric_func: Fairness metric function to compute
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Protected attribute
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        method: 'auto' (bootstrap; doubled resamples when a group is small),
            'bootstrap', or 'bayesian' (NOT implemented: warns and falls back
            to the doubled-resample bootstrap). Any other value raises
            ConfigurationError rather than silently running a bootstrap.
        random_state: Random seed
        null_value: The value *metric_func* takes under the parity null, carried
            onto the result so ``StatisticalResult.is_significant`` tests the
            interval against the right anchor. 0.0 (the default) is correct for
            every difference and spread statistic; pass 1.0 for a RATIO metric,
            whose parity value is 1 and for which a zero-anchored test is True
            for every possible input, including a perfectly fair model.
        **metric_kwargs: Additional arguments for metric function

    Returns:
        StatisticalResult with point estimate and interval

    Example:
        >>> from vfairness import demographic_parity_difference
        >>> result = compute_metric_with_ci(
        ...     demographic_parity_difference,
        ...     y_true, y_pred, gender,
        ...     min_group_size=30
        ... )
        >>> print(f"DP Diff: {result.point_estimate:.3f} "
        ...       f"[{result.lower_bound:.3f}, {result.upper_bound:.3f}]")

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1),
            or if method is not one of 'auto', 'bootstrap', 'bayesian'.
    """
    confidence_level = validate_confidence_level(confidence_level)
    if method not in ("auto", "bootstrap", "bayesian"):
        raise ConfigurationError(
            f"method must be 'auto', 'bootstrap' or 'bayesian', got {method!r} "
            f"('bayesian' is not implemented and falls back to bootstrap)."
        )

    from ._grouping import GroupManager

    # Determine group sizes
    gm = GroupManager(sensitive_attr, min_group_size=1)
    group_sizes = gm.get_group_sizes()

    if not group_sizes:
        # Every row was excluded upstream (e.g. all labels missing), so there
        # is nothing to resample. Degrade gracefully with a NaN interval
        # instead of crashing on min() over an empty dict.
        warnings.warn(
            "No data remains after exclusions; cannot compute a confidence "
            "interval. Returning a NaN interval.",
            UserWarning,
        )
        try:
            point_estimate = metric_func(y_true, y_pred, sensitive_attr, **metric_kwargs)
        except Exception:
            point_estimate = float("nan")
        return StatisticalResult(
            point_estimate=point_estimate,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method="unavailable_no_data",
            sample_size=0,
            n_bootstrap=0,
            null_value=null_value,
            metadata={"warning": "No data after exclusions"},
        )

    min_size = min(group_sizes.values())

    # Select method. F18 (audit 6, 2026-09-09): 'auto' used to select a
    # 'bayesian' branch that has no estimator behind it, so a small-sample
    # 'auto' run warned "Bayesian method requested" for a request nobody made.
    # Measured before the change: method='auto' with two groups of 20 ->
    # warning "Bayesian method requested but using bootstrap ...",
    # result.method 'stratified_bootstrap_fold_debiased', n_bootstrap doubled.
    # Only the bootstrap exists; 'auto' now names it directly and doubles the
    # resamples for small groups, and an explicit 'bayesian' request keeps its
    # warning and takes the same doubled-resample path. `method` stays what
    # the caller ASKED for; `resolved` is what actually RUNS (never
    # 'bayesian'), and the result's `method` field names the interval that
    # was built.
    resolved: Literal["bootstrap", "bootstrap_enhanced"]
    if method == "auto":
        if min_size < SMALL_SAMPLE_THRESHOLD:
            warnings.warn(
                f"Small sample (min group: {min_size} < {SMALL_SAMPLE_THRESHOLD}): running "
                f"the stratified bootstrap with {2 * n_bootstrap} resamples. A Bayesian "
                f"interval is not implemented. Results should be interpreted with caution.",
                UserWarning,
            )
            resolved = "bootstrap_enhanced"
        else:
            resolved = "bootstrap"
    elif method == "bayesian":
        warnings.warn(
            f"Bayesian method requested but using bootstrap with enhanced sampling: "
            f"a Bayesian estimator is not implemented (min group: {min_size}). "
            f"Results should be interpreted with caution.",
            UserWarning,
        )
        resolved = "bootstrap_enhanced"
    else:
        resolved = "bootstrap"

    # Point estimate
    point_estimate = metric_func(y_true, y_pred, sensitive_attr, **metric_kwargs)

    if resolved == "bootstrap":
        # Stratified bootstrap
        def stat_func(data_idx):
            return metric_func(
                y_true[data_idx], y_pred[data_idx], sensitive_attr[data_idx], **metric_kwargs
            )

        rng = np.random.default_rng(random_state)
        n = len(y_true)

        # Get group indices for stratified sampling
        unique_groups = np.unique(sensitive_attr)
        group_indices = {g: np.where(sensitive_attr == g)[0] for g in unique_groups}

        bootstrap_stats = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            resample_idx_list: list[Any] = []
            for g, indices in group_indices.items():
                sampled = rng.choice(indices, size=len(indices), replace=True)
                resample_idx_list.extend(sampled)
            resample_idx = np.array(resample_idx_list)

            try:
                bootstrap_stats[i] = metric_func(
                    y_true[resample_idx],
                    y_pred[resample_idx],
                    sensitive_attr[resample_idx],
                    **metric_kwargs,
                )
            except Exception:
                bootstrap_stats[i] = np.nan

        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        if len(bootstrap_stats) < n_bootstrap * 0.9:
            warnings.warn("More than 10% of bootstrap samples resulted in NaN")

        if len(bootstrap_stats) < 10:
            return StatisticalResult(
                point_estimate=point_estimate,
                lower_bound=np.nan,
                upper_bound=np.nan,
                interval_type=IntervalType.CONFIDENCE,
                confidence_level=confidence_level,
                method="stratified_bootstrap",
                sample_size=n,
                n_bootstrap=n_bootstrap,
                null_value=null_value,
                metadata={"warning": "Bootstrap failed"},
            )

        alpha = 1 - confidence_level
        q_lo = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        q_hi = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
        se = float(np.std(bootstrap_stats))
        method_name = "stratified_bootstrap_percentile"
        interval_meta: Dict = {"interval_construction": "percentile"}

        # THE GUARD SITS ABOVE THE DISPATCH, NOT INSIDE ONE BRANCH.
        # The fold-at-zero pathology is defined by the metric's NULL being at
        # zero, which only the CALLER knows; it is not discoverable from the
        # resample support. Until 2026-09-11 `folded` was entered by any
        # nonnegative statistic and applicability was then guessed from the
        # permutation support (`zero_anchored`, below), ignoring the
        # `null_value=1.0` that disparate_impact_ratio_with_ci explicitly
        # passes in. Measured on EXACT parity, 100 rows per group with 6
        # selected in each: point estimate 1.000 and CI [0.000, 1.778] via
        # 'stratified_bootstrap_fold_debiased'. The lower bound was the
        # literal constant 0.0 (the permutation gate cannot reject under
        # parity, so the else branch assigned it) and nothing had been
        # estimated there, while the upper bound sat outside the ratio's [0, 1]
        # support. The four-fifths seal reads that lower bound, so a perfectly
        # fair model was failed by its own gate, with no warning. Fixing only
        # the `zero_anchored` line would move the same fabrication to its
        # sibling branches, which also clip a lower bound at 0. Anchoring is
        # now decided by the declared null, exactly as this function's
        # docstring already promised ("Nonnegative statistics whose null value
        # is elsewhere ... keep the percentile interval"). null_value defaults
        # to 0.0, so every difference and spread metric is unaffected.
        folded = bool(
            float(null_value) == 0.0 and point_estimate >= 0.0 and np.all(bootstrap_stats >= 0.0)
        )
        if folded:
            # FOLD-DEBIASED interval for nonnegative spread statistics
            # (max-min disparities). A raw percentile CI of such a statistic
            # NEVER contains 0 at the parity null (its resample distribution
            # sits strictly above 0), so coverage there was 0% and
            # is_significant fired for every perfectly fair model
            # (audit 2026-08-22). Construction, Monte Carlo validated:
            #   lower: 0 unless a permutation test of the sensitive attribute
            #          rejects the exchangeability null at level alpha/2
            #          (exact test inversion at the boundary; Efron &
            #          Tibshirani 1993 ch. 15), then the basic
            #          (reverse-percentile) lower bound clipped at 0.
            #   upper: the WIDER of the percentile and basic upper bounds.
            #          The plain basic upper under-covers when the point
            #          estimate lands near the fold (miss rates 10-16% with
            #          a true spread at the band edge), which is false-PASS
            #          direction for a seal gating on the upper bound.
            basic_lo = 2.0 * point_estimate - q_hi
            basic_up = 2.0 * point_estimate - q_lo
            upper = max(q_hi, basic_up)

            # Permutation test: shuffle the attribute, keep (y_true, y_pred).
            # Exact for the sharp fairness null "the metric's distribution is
            # invariant under exchange of group labels".
            n_perm, perm_capped = _permutation_count(alpha)
            perm_stats = np.empty(n_perm)
            for i in range(n_perm):
                try:
                    perm_stats[i] = metric_func(
                        y_true,
                        y_pred,
                        rng.permutation(sensitive_attr),
                        **metric_kwargs,
                    )
                except Exception:
                    perm_stats[i] = np.nan
            perm_stats = perm_stats[~np.isnan(perm_stats)]
            if len(perm_stats) >= 50:
                # The fold-at-zero pathology only afflicts statistics whose
                # EXCHANGEABILITY-NULL distribution is anchored at 0 (spreads,
                # absolute differences). A nonnegative statistic whose null
                # value is elsewhere, e.g. a min/max RATIO whose parity value
                # is 1, or a level statistic like worst-group accuracy, must
                # NOT have its lower bound clamped to 0: for those the lower
                # bound is the decision surface (four-fifths rule) and the old
                # percentile interval is the correct construction. Test
                # anchoring on the null distribution's own scale: 0 lies
                # within one support-width of the permutation distribution.
                min_perm = float(np.min(perm_stats))
                max_perm = float(np.max(perm_stats))
                zero_anchored = min_perm <= (max_perm - min_perm)
                if zero_anchored:
                    p_perm = float(
                        (1 + np.sum(perm_stats >= point_estimate)) / (len(perm_stats) + 1)
                    )
                    lower = max(0.0, basic_lo) if p_perm < alpha / 2 else 0.0
                    upper = max(upper, lower)
                    method_name = "stratified_bootstrap_fold_debiased"
                    # THE ATTAINED RESOLUTION, ALWAYS, NOT ONLY WHEN THE CAP BOUND.
                    # The gate rejects at alpha/2 and the smallest p this run can
                    # produce is 1/(n+1) over the permutations that were actually
                    # MEASURABLE. `perm_capped` describes only the DESIGN hitting
                    # MAX_PERMUTATIONS; NaN attrition shortens the attained
                    # resolution just as effectively and used to be recorded
                    # nowhere. Measured 2026-09-10 at confidence_level=0.999
                    # (alpha 0.001, design 4000 permutations, cap NOT bound) with
                    # a metric undefined for most label shuffles: 200 usable
                    # permutations, floor 0.004975 against a bar of 0.0005, so the
                    # gate could not reject for ANY data, and the metadata carried
                    # neither `permutation_resolution_limited` nor
                    # `min_resolvable_p_value`.
                    attained_floor = min_attainable_p_permutation(len(perm_stats))
                    # n_family=1 with alpha/2 is the gate's own bar, spelled the
                    # way the shared helper takes it.
                    gate_detectable, gate_note = detectability(
                        attained_floor, n_family=1, alpha=alpha / 2.0
                    )
                    interval_meta = {
                        "interval_construction": "fold_debiased_perm_gated",
                        "nonnegative_statistic": True,
                        "permutation_p_value": p_perm,
                        "n_permutation": int(len(perm_stats)),
                        "n_permutation_requested": int(n_perm),
                        "min_resolvable_p_value": attained_floor,
                    }
                    if perm_capped or gate_detectable is not True:
                        # Either the permutation budget hit MAX_PERMUTATIONS or
                        # attrition ate the resolution, so this run has less
                        # resolution than the design asked for: the smallest
                        # p-value it can produce is 1/(n+1), and once that reaches
                        # alpha/2 the gate can no longer reject and the lower bound
                        # below is the fold, not a test result. Say so rather than
                        # letting the method name imply an exact test that ran at
                        # full resolution.
                        interval_meta["permutation_resolution_limited"] = True
                        if gate_note:
                            interval_meta["permutation_design_note"] = gate_note
                else:
                    # Nonnegative but not zero-anchored: keep percentile
                    # (previous behavior; the interval_meta default stands).
                    lower, upper = q_lo, q_hi
            else:
                # Too few valid permutations to run the exact gate or examine
                # the null distribution: fall back to the clipped basic lower
                # (first-order debiased for any statistic, never the raw
                # percentile lower, which cannot reach the boundary).
                lower = max(0.0, basic_lo)
                upper = max(upper, lower)
                method_name = "stratified_bootstrap_fold_debiased"
                interval_meta = {
                    "interval_construction": "fold_debiased_basic",
                    "nonnegative_statistic": True,
                    "warning": "permutation test unavailable (too many NaN permutations)",
                }
        else:
            # Signed statistic, or a nonnegative one whose null is not 0 (a
            # min/max RATIO with parity at 1, a level statistic): the fold
            # pathology does not apply; keep the plain percentile interval
            # (previous behavior).
            lower, upper = q_lo, q_hi

        return StatisticalResult(
            point_estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method=method_name,
            sample_size=n,
            n_bootstrap=len(bootstrap_stats),
            standard_error=se,
            null_value=null_value,
            metadata={"group_sizes": group_sizes, **interval_meta},
        )

    else:  # 'bootstrap_enhanced': small groups, or an explicit 'bayesian' request (F18)
        # No Bayesian estimator exists here; the warning for reaching this
        # branch was raised at method selection above, where the reason is
        # known. Use more bootstrap samples for small data.
        enhanced_n_bootstrap = n_bootstrap * 2
        return compute_metric_with_ci(
            metric_func,
            y_true,
            y_pred,
            sensitive_attr,
            n_bootstrap=enhanced_n_bootstrap,
            confidence_level=confidence_level,
            method="bootstrap",
            random_state=random_state,
            # Carried through the small-sample recursion: dropping it here
            # would silently reset a ratio's null to 0 for exactly the small
            # groups the routing exists to protect.
            null_value=null_value,
            **metric_kwargs,
        )


# Proportion-Specific Statistics (for intersectional fairness analysis)

# Note: _erf() and _norm_cdf() already defined earlier in this module (lines ~992-1010).
# No duplicates needed.


def proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """
    Two-proportion z-test. Returns two-sided p-value.

    Tests H0: p1 = p2 using a pooled proportion estimate.
    No scipy dependency (uses error function approximation).

    Args:
        p1: Proportion in group 1
        n1: Sample size of group 1
        p2: Proportion in group 2
        n2: Sample size of group 2

    Returns:
        Two-sided p-value, or NaN when no test could be run.

    Reference:
        Agresti & Caffo (2000). Simple and effective confidence intervals
        for proportions and differences of proportions.

    BGL-G001 (2026-09-17). A group too small to test returned ``1.0``, which is
    not a sentinel on this scale: 1.0 is the LARGEST p-value there is, it is a
    perfectly ordinary result, and every consumer reads it as "no evidence of a
    difference". Measured at the public entry, before this change:

        proportion_z_test(1.0, 1, 0.0, 50) -> 1.0

    i.e. a group selected at 100% against a group selected at 0% was reported as
    the most reassuring answer the test can give, because one arm had a single
    row. NaN says the test did not run, and ``_testable`` in this same module
    then drops it from the multiple-comparison family instead of letting it
    dilute the findings that did run.

    The two remaining ``1.0`` returns are kept and are NOT this: ``p_pool`` at 0
    or 1 means both proportions are identical at the boundary (every row
    negative, or every row positive, in both groups), which is a genuine
    observation of no difference, and the ``se`` floor is the same statement one
    step later.
    """
    if not np.isfinite(p1) or not np.isfinite(p2):
        warnings.warn(
            f"proportion_z_test: a proportion was not measured (p1={p1}, p2={p2}), so "
            f"no test was run. Returning nan, not a p-value.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    if n1 < 2 or n2 < 2:
        warnings.warn(
            f"proportion_z_test: group sizes n1={n1}, n2={n2} are too small to run a "
            f"two-proportion z-test, so no test was run. Returning nan, NOT 1.0, "
            f"which reads as a measured absence of difference.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    # BGL-G001 follow-up (2026-09-25). The VALUE stays 1.0 in both branches and
    # the docstring above is right that it is a genuine observation: with a pooled
    # proportion of 0 (or 1) both arms are identical to the last observation, so
    # the difference is exactly zero and refusing it would throw away a real
    # measurement. What was wrong is that it was SILENT. The variance is zero
    # here, so the p-value carries no information about POWER, and a reader could
    # not tell "tested and found parity" from "there was nothing in either arm to
    # detect a difference with". An independent audit overturned the PROVEN grade
    # for exactly that reason. Both branches now disclose it and still return 1.0.
    if p_pool <= 0 or p_pool >= 1:
        warnings.warn(
            f"proportion_z_test: every observation in both arms is "
            f"{'a non-event' if p_pool <= 0 else 'an event'} (pooled proportion "
            f"{p_pool:.3f}), so the two rates are identical and the variance is "
            "zero. The returned p of 1.0 is a genuine observation of no "
            "difference, NOT evidence that a difference would have been detected: "
            "this comparison had no power at all.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-10:
        warnings.warn(
            f"proportion_z_test: the pooled standard error is {se:.3e}, "
            "indistinguishable from zero, so no z statistic is defined. The "
            "returned p of 1.0 records that the observed rates do not differ, "
            "and carries no information about whether a difference could have "
            "been detected.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0
    z = abs(p1 - p2) / se
    p_value = 2 * (1 - _norm_cdf(z))
    return float(max(p_value, 1e-300))


def fisher_exact_test(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher's exact test for a 2x2 contingency table. Returns two-sided p-value.

    Uses scipy.stats.fisher_exact (scipy is a core dependency), so the test
    stays EXACT at every table size. If scipy is somehow unavailable, falls
    back to a direct hypergeometric computation for n <= 200 and a chi-squared
    approximation above that, with a UserWarning disclosing the approximation.

    Table layout:
        [[a, b],
         [c, d]]

    Args:
        a, b, c, d: Cell counts of the 2x2 table

    Returns:
        Two-sided p-value

    Reference:
        Fisher (1922). On the interpretation of chi-square from
        contingency tables, and the calculation of P.

    BGL-G001 (raised 2026-09-17, RESOLVED 2026-09-25). A table with any empty
    MARGIN returns could-not-check (``nan``) and warns, instead of the ``1.0``
    it used to return.

    The defect, as recorded: an empty table (every cell zero, so no observation
    of anything) answered ``1.0``, the largest p-value there is, and a consumer
    reads that as "no association found" rather than "nothing was counted".
    The change was built once, reverted, and left recorded with the argument
    against it, which was that scipy answers ``1.0`` here too
    (``fisher_exact([[0,0],[0,0]])`` gives ``statistic=nan, pvalue=1.0``), so
    refusing only the all-zero table would draw an arbitrary line.

    The revert was right about the arbitrary line and wrong about which way to
    settle it. It is settled here for ALL FOUR degenerate-margin shapes at once,
    which is what the record asked for:

    * empty table          ``(0, 0, 0, 0)``
    * empty row            ``(5, 5, 0, 0)`` / ``(0, 0, 5, 5)``
    * empty column         ``(5, 0, 5, 0)`` / ``(0, 5, 0, 5)``

    In every one of them one margin is zero, the odds ratio is not estimable,
    and the conditional distribution has exactly one possible table. 1.0 is the
    conventional answer to a question nobody could ask. This library reports its
    own could-not-check instead, for two reasons:

    * ``proportion_z_test``, in this same module, already made this choice: it
      returns ``nan`` and warns rather than the ``1.0`` it used to return for
      100% against 0% at n=1. Two functions answering the same "nothing was
      measurable" with different conventions is worse than either convention.
    * scipy reports ``statistic=nan`` alongside its ``pvalue=1.0``. The nan is
      the part that says "not estimable", and this function returns only the
      p-value, so passing 1.0 on drops the disclosure and keeps the number.

    The only caller in the library, ``intersectional._proportion_test``, already
    handles a non-finite p: ``generate_structured_findings`` excludes it from the
    Benjamini-Hochberg correction and leaves ``statistically_significant`` as
    ``None``, which is could-not-check rather than cleared. Verified by execution,
    not by reading: see tests/test_surface_grade_g001.py.

    What is NOT changed: every table with real margins still goes to scipy and
    still matches it exactly, pinned by a control test at four sample sizes.
    """
    n = a + b + c + d

    # A zero MARGIN, not merely a zero cell. With an empty row or an empty
    # column the odds ratio is not estimable at all: conditional on those
    # margins the observed table is the only one possible, so any p-value is an
    # answer to a question the data cannot pose. See BGL-G001 above.
    empty = [
        label
        for label, total in (
            ("first row", a + b),
            ("second row", c + d),
            ("first column", a + c),
            ("second column", b + d),
        )
        if total == 0
    ]
    if empty:
        warnings.warn(
            "fisher_exact_test: no association is estimable because the "
            f"{' and the '.join(empty)} of the table is empty "
            f"(table [[{a}, {b}], [{c}, {d}]]); returning nan, which means "
            "could not check and not 'no association found'",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # Exact test at any n. The old n>200 path silently switched to an
    # UNCORRECTED chi-square and returned p=1.0 whenever any expected cell
    # was < 1, hiding exactly the sparse-cell disparities an audit cares
    # about (e.g. [[1,0],[100,149]] -> p=1.0 while the exact p is ~0.40).
    try:
        from scipy.stats import fisher_exact as _scipy_fisher

        _, p_value = _scipy_fisher([[a, b], [c, d]], alternative="two-sided")
        return float(min(max(p_value, 1e-300), 1.0))
    except ImportError:
        pass

    warnings.warn(
        "scipy is not available; fisher_exact_test is using an approximate "
        "fallback (exact hypergeometric for n <= 200, uncorrected chi-square "
        "above). p-values for sparse tables may be unreliable.",
        UserWarning,
    )

    # For small tables, compute exact probability
    if n <= 200:
        from math import lgamma

        def _log_hypergeom(a, b, c, d):
            n = a + b + c + d
            return (
                lgamma(a + b + 1)
                + lgamma(c + d + 1)
                + lgamma(a + c + 1)
                + lgamma(b + d + 1)
                - lgamma(a + 1)
                - lgamma(b + 1)
                - lgamma(c + 1)
                - lgamma(d + 1)
                - lgamma(n + 1)
            )

        observed_log_p = _log_hypergeom(a, b, c, d)
        p_value = 0.0
        row1 = a + b
        row2 = c + d
        col1 = a + c
        for x in range(min(row1, col1) + 1):
            y = row1 - x
            z = col1 - x
            w = row2 - z
            if y < 0 or z < 0 or w < 0:
                continue
            log_p = _log_hypergeom(x, y, z, w)
            if log_p <= observed_log_p + 1e-10:
                p_value += np.exp(log_p)
        return float(min(p_value, 1.0))
    else:
        # Chi-squared approximation for larger tables
        expected = np.array(
            [
                [(a + b) * (a + c) / n, (a + b) * (b + d) / n],
                [(c + d) * (a + c) / n, (c + d) * (b + d) / n],
            ]
        )
        observed = np.array([[a, b], [c, d]])
        if np.any(expected < 1):
            return 1.0
        chi2 = np.sum((observed - expected) ** 2 / expected)
        # Chi-squared with 1 df: p = 1 - CDF = erfc(sqrt(chi2/2))
        p_value = 1 - _erf(np.sqrt(chi2 / 2))
        return float(max(p_value, 1e-300))


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for the difference between two proportions.

    This is the appropriate effect size measure for binary outcomes,
    unlike Cohen's d which is for continuous data.

    Interpretation:
        |h| < 0.2: negligible
        0.2 <= |h| < 0.5: small
        0.5 <= |h| < 0.8: medium
        |h| >= 0.8: large

    Args:
        p1: Proportion in group 1
        p2: Proportion in group 2

    Returns:
        Cohen's h (signed: positive means p1 > p2 after arcsine transform), or
        NaN when either proportion was not measured.

    Reference:
        Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.

    BGL-S2b (2026-09-17). The clamp was ``max(0.0, min(1.0, p))`` with the
    BUILTIN max and min, and every comparison against NaN is False, so
    ``min(1.0, nan)`` keeps its first argument 1.0 and ``max(0.0, 1.0)`` keeps
    it. An UNMEASURED proportion was therefore clamped to a perfect rate of
    1.0 and the function answered:

        cohens_h(nan, 0.5) ->  1.5707963  ("large", the top band)
        cohens_h(0.5, nan) -> -1.5707963  ("large", the other direction)

    pi/2 is the LARGEST value this statistic can take between 0 and 0.5, so a
    proportion nobody measured produced the biggest effect size in the family
    and a direction with it. It also made the NaN guard downstream in
    ``classification.compute_effect_sizes`` unreachable: nothing that entered
    here could leave as NaN. NaN in, NaN out; a genuine out-of-range input
    (a caller bug, not a could-not-check) is still clamped as before.
    """
    if not np.isfinite(p1) or not np.isfinite(p2):
        warnings.warn(
            f"cohens_h: proportion(s) not measured (p1={p1}, p2={p2}), so no effect "
            f"size was computed. Returning nan, NOT the pi/2 that the old clamp "
            f"produced by turning a non-finite proportion into a rate of 1.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def cohens_h_interpretation(h: float) -> str:
    """Interpret Cohen's h effect size magnitude.

    Returns "not interpretable" when ``h`` is not a finite number, which is
    could-not-check and is never one of the four magnitude bands.

    BGL-G001 (2026-09-17). ``abs(nan) < 0.2`` is False, and so is every later
    threshold, so an UNMEASURED effect size fell through the whole ladder and
    came out as ``"large"``, the top band and the most alarming word this
    function can produce, about nothing at all. ``inf`` did the same.

    That mattered from the day ``cohens_h`` was corrected: BGL-S2b (2026-09-17,
    the same wave, forty lines up in this file) stopped ``cohens_h`` turning a
    non-finite proportion into a rate of 1.0 and made it return NaN. Its direct
    consumer then graded that NaN:

        cohens_h_interpretation(cohens_h(nan, 0.5)) -> 'large'

    so the fabrication moved one function downstream instead of ending, and the
    surfaces that read it (``analyzer._effect_size_for``,
    ``intersectional.generate_structured_findings['effect_size_interpretation']``,
    ``feature_engineering.significance``) printed a magnitude word for a
    proportion nobody measured. The identical ladder in
    :func:`interpret_effect_size` in this same module was given this guard on
    2026-09-08; this is that guard at the sibling site that was missed.
    """
    if not np.isfinite(h):
        warnings.warn(
            f"cohens_h_interpretation: the effect size is {h}, which is not a finite "
            f"number, so it cannot be graded. Returning 'not interpretable' rather "
            f"than a magnitude band.",
            UserWarning,
            stacklevel=2,
        )
        return "not interpretable"

    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def minimum_detectable_effect(
    n1: int,
    n2: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_rate: float = 0.5,
) -> float:
    """
    Compute the minimum detectable effect (MDE) for a two-proportion z-test.

    Given the sample sizes and desired power, returns the smallest absolute
    difference in proportions that the test can reliably detect.

    Args:
        n1: Sample size of group 1
        n2: Sample size of group 2
        alpha: Significance level (default 0.05)
        power: Desired statistical power (default 0.80)
        baseline_rate: Assumed baseline proportion (default 0.5)

    Returns:
        Minimum detectable difference in proportions (absolute)

    Reference:
        Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
        Normal approximation for the two-proportion z-test at the assumed
        baseline rate (not the arcsine transformation).
    """
    # EVERY INPUT IS GATED ABOVE THE ARITHMETIC, NOT BY A BARE COMPARISON.
    # B4 tier-1 audit, 2026-09-30. The only input gate this function had was the
    # bare `n1 < 2 or n2 < 2` below, and a bare comparison lets an UNMEASURABLE arm
    # straight through: inf answers `< 2` with False. Worse, because 1/inf is 0 an
    # infinite arm makes the standard error SMALLER, so the `not np.isfinite(mde)`
    # branch further down can never rescue it and the design is published as BETTER
    # POWERED than a real one. Measured before this change, warnings recorded:
    #   minimum_detectable_effect(inf, 100) -> 0.14007926090564843, warnings []
    #   minimum_detectable_effect(100, inf) -> 0.14007926090564843, warnings []
    # against the real design minimum_detectable_effect(100, 1000) -> 0.14691636828.
    # Three more, all silent:
    #   minimum_detectable_effect(2.5, 100) -> 0.8969449106666371, a sample size of
    #     two and a half rows accepted and published
    #   minimum_detectable_effect(100, 1000, alpha=1.0) -> 0.04413498982895752: a
    #     significance level of 1.0 sets z_alpha to 0, so the published floor
    #     understates the real 0.1469 threefold, and nothing range-checked alpha,
    #     power or baseline_rate anywhere in the unit
    #   baseline_rate=1.5 was refused only BY ACCIDENT, via np.sqrt of a negative
    #     producing nan, and it leaked numpy's own "invalid value encountered in
    #     sqrt" RuntimeWarning to the caller before this function said anything
    #
    # `is_measured` is the repo-wide predicate for "a real, finite number", and it
    # is used here rather than a fresh isinstance test because
    # `isinstance(v, (int, float))` rejects np.int64 and np.float32 and would
    # discard real evidence while reading as caution (see vfairness._triage).
    for _name, _arm in (("n1", n1), ("n2", n2)):
        if not is_measured(_arm) or float(_arm) != int(_arm):
            warnings.warn(
                f"minimum_detectable_effect: {_name}={_arm!r} is not a whole, finite "
                f"count of rows, so the design could not be assessed and no minimum "
                f"detectable effect was computed. Returning nan, which means COULD NOT "
                f"CHECK. A non-finite arm is the dangerous one: 1/inf is 0, so it makes "
                f"the floor SMALLER and the design would read as better powered than a "
                f"real one.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

    # alpha, power and baseline_rate are probabilities and each must lie strictly
    # inside (0, 1). This is checked ABOVE the arithmetic, not left to whatever the
    # arithmetic happens to produce: alpha=1.0 produced a smaller, entirely
    # plausible-looking floor, and baseline_rate=1.5 reached np.sqrt of a negative.
    for _name, _value in (("alpha", alpha), ("power", power), ("baseline_rate", baseline_rate)):
        if not is_measured(_value) or not 0.0 < float(_value) < 1.0:
            warnings.warn(
                f"minimum_detectable_effect: {_name}={_value!r} is not a probability "
                f"strictly between 0 and 1, so the design (n1={n1}, n2={n2}) describes "
                f"no test that could be run and no minimum detectable effect was "
                f"computed. Returning nan, which means COULD NOT CHECK, NOT a floor.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

    if n1 < 2 or n2 < 2:
        # "Cannot detect anything" was the comment here, and the value returned
        # was 1.0, which says the opposite: a minimum detectable effect of 1.0 is
        # "a 100-point gap is detectable". It is also FINITE, so the could-not-
        # check path in `power_warning` (which tests np.isfinite) never fired, and
        # n=0, n=1 and a negative n all arrived in the "very low power" band as
        # though a real floor had been computed. Measured 2026-09-25:
        # minimum_detectable_effect(0, 100) -> 1.0.
        warnings.warn(
            f"minimum_detectable_effect: an arm of n1={n1}, n2={n2} cannot support "
            "a two-sample comparison at all, so no minimum detectable effect "
            "exists. Returning nan, which means could not check; it is NOT a "
            "floor of 100%.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # z-values for the REQUESTED alpha (two-sided) and power. The old
    # hardcoded ladder silently mapped any nonstandard alpha to 1.645 and
    # any nonstandard power to 0.524, so e.g. alpha=0.20 or power=0.95
    # produced an MDE for a different test than the one requested.
    try:
        from scipy.stats import norm as _scipy_norm

        z_alpha = float(_scipy_norm.ppf(1.0 - alpha / 2.0))
        z_power = float(_scipy_norm.ppf(power))
    except ImportError:
        # Module-local rational approximation (accurate to ~4e-4)
        z_alpha = float(_norm_ppf(1.0 - alpha / 2.0))
        z_power = float(_norm_ppf(power))

    # MDE using normal approximation
    se = np.sqrt(baseline_rate * (1 - baseline_rate) * (1 / n1 + 1 / n2))
    mde = (z_alpha + z_power) * se

    # BGL-G001 (2026-09-17). A minimum DETECTABLE effect cannot be negative or
    # zero, and this arithmetic produced both. Measured at the public entry,
    # before this change:
    #
    #   minimum_detectable_effect(100, 1000, 0.5, 0.1) -> -0.0318
    #   minimum_detectable_effect(100, 1000, 0.0, 0.8) ->  1.0   (inf, clamped)
    #
    # -0.0318 is the dangerous one, and the danger is downstream: power_warning
    # below asks `mde > 0.20` and then `mde > 0.10`, both False for a negative
    # number, so it returned None, which on that function means ADEQUATE POWER.
    # A design so underpowered that its MDE arithmetic went negative was
    # reported as fine. NaN is could-not-check and power_warning now says so
    # instead of clearing it.
    if not np.isfinite(mde) or mde <= 0.0:
        warnings.warn(
            f"minimum_detectable_effect: the design (n1={n1}, n2={n2}, alpha={alpha}, "
            f"power={power}, baseline_rate={baseline_rate}) gives a minimum "
            f"detectable effect of {mde}, which is not a positive size, so no MDE "
            f"was computed. Returning nan (could not check), NOT a number that would "
            f"be compared against a power threshold.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # BGL-5 (2026-09-27). ``return float(min(mde, 1.0))`` clamped the arithmetic
    # floor to the EDGE of the attainable range and reported that edge as a
    # measurement. A difference of two proportions cannot exceed 1.0, so a floor
    # above 1.0 means the design can detect NOTHING: that is could-not-check, and
    # 1.0 is the one value that reads as "a 100-point gap is detectable", the
    # exact lie the n < 2 branch above was already fixed for. Measured before,
    # with the raw arithmetic floor beside the returned value:
    #
    #   minimum_detectable_effect(2, 2)   -> 1.0  (raw 1.4008)  warnings []
    #   minimum_detectable_effect(2, 3)   -> 1.0  (raw 1.2787)  warnings []
    #   minimum_detectable_effect(3, 3)   -> 1.0  (raw 1.1437)  warnings []
    #   minimum_detectable_effect(2, 5)   -> 1.0  (raw 1.1720)  warnings []
    #   minimum_detectable_effect(2, 100) -> 1.0  (raw 1.0004)  warnings []
    #
    # Measured after: each of those returns nan and warns. Arms of two or three
    # rows are ordinary fairness-audit data, not caller error, and the finite 1.0
    # is what let power_warning's np.isfinite check clear them into the "very low
    # power" band with the sentence "can only detect disparities larger than
    # 100%". Unchanged where a floor exists: minimum_detectable_effect(3, 100) ->
    # 0.8208, (30, 100) -> 0.2916, (100, 1000) -> 0.1469, none of them clamped.
    if mde > 1.0:
        warnings.warn(
            f"minimum_detectable_effect: the design (n1={n1}, n2={n2}, alpha={alpha}, "
            f"power={power}, baseline_rate={baseline_rate}) gives a minimum detectable "
            f"effect of {mde:.4f}, and a difference of two proportions cannot exceed "
            f"1.0, so no ATTAINABLE minimum detectable effect exists: this design can "
            f"detect nothing. Returning nan (could not check). It used to return a "
            f"clamped 1.0, which reads as 'a 100-point gap is detectable'.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(mde)


def power_warning(
    group_size: int,
    total_size: int,
    alpha: float = 0.05,
) -> Optional[str]:
    """
    Generate a human-readable power warning for a subgroup.

    Returns None if power is adequate, otherwise a warning string.

    Args:
        group_size: Size of the subgroup
        total_size: Total dataset size
        alpha: Significance level

    Returns:
        Warning string or None
    """
    # COHERENCE FIRST, and before the adequate-power shortcut below. Returning
    # None from this function means ADEQUATE POWER, and `group_size >= 200` was
    # the first thing it asked, so power_warning(250, 100) cleared a subgroup
    # larger than the dataset containing it. Below 200 the impossible pairs took
    # a band instead: power_warning(-5, 100) produced the sentence "Very low
    # statistical power (n=-5)". A power verdict from a pair that cannot describe
    # any dataset is a verdict nobody measured, in whichever direction it lands.
    # MEASURED FIRST, because the coherence test below is three BARE COMPARISONS and
    # a non-finite total answers all three the safe way. B4 tier-1 audit,
    # 2026-09-30: power_warning(100, float('inf')) printed "Limited statistical power
    # (n=100). This test can reliably detect disparities larger than 14%." with zero
    # warnings, because inf is not < 0, not <= 0 and not smaller than group_size.
    if not is_measured(group_size) or not is_measured(total_size):
        return (
            f"Statistical power COULD NOT BE ASSESSED: a subgroup of {group_size} "
            f"within a dataset of {total_size} is not a measured pair of row counts, "
            f"so no minimum detectable effect follows from it. This is not a "
            f"statement that power is adequate."
        )

    if group_size < 0 or total_size <= 0 or group_size > total_size:
        return (
            f"Statistical power COULD NOT BE ASSESSED: a subgroup of {group_size} "
            f"within a dataset of {total_size} is not a possible pair, so no "
            f"minimum detectable effect follows from it. This is not a statement "
            f"that power is adequate."
        )

    # THE SECOND ARM IS THE COMPLEMENT, AND THE ADEQUATE-POWER SHORTCUT SITS BELOW
    # ITS GATE.
    # B4 tier-1 audit, 2026-09-30. This function used to pass `total_size` as the
    # second arm of a two-proportion test. The second arm of "this subgroup against
    # the rest" is the COMPLEMENT, so the subgroup was counted on BOTH sides and the
    # published floor was computed over a population that does not exist. Worse, the
    # `group_size >= 200` shortcut sat ABOVE the whole computation, and returning
    # None from this function means ADEQUATE POWER, so every guard the earlier waves
    # added (the non-finite refusal, the `mde >= 1.0` refusal, the percent
    # formatting) sat below it and could not fire at all. Measured before this
    # change, power_warning's own warnings [] in every case:
    #   power_warning(200, 200) -> None, and the comparison arm has ZERO rows
    #   power_warning(250, 250) -> None, likewise
    #   power_warning(500, 500) -> None, likewise
    #   power_warning(199, 200) -> "... larger than 14%", real comparison arm of ONE
    #   power_warning(100, 101) -> "... larger than 20%", real comparison arm of ONE
    #   power_warning(199, 201) -> "... reliably detect disparities larger than 14%"
    #     while the honest minimum_detectable_effect(199, 2) over the real complement
    #     is 0.9954749461789353: a 14 percent claim against a true 99.5 percent floor
    # The coherence test above permits group_size == total_size on purpose (it tests
    # `>`, not `>=`), which is correct for "is this a possible pair" and is exactly
    # why the complement needs its own gate rather than a wider coherence test.
    comparison_size = total_size - group_size
    if comparison_size < 2:
        return (
            f"Statistical power COULD NOT BE ASSESSED (n={group_size}): this subgroup "
            f"is compared against the other {comparison_size} row(s) of the "
            f"{total_size}, which cannot support a two-proportion test, so no minimum "
            f"detectable effect exists. This is not a statement that power is "
            f"adequate, and a 'not significant' reading carries no information."
        )

    if group_size >= 200:
        return None  # Adequate power for most practical effects

    with warnings.catch_warnings():
        # The producer warns when no floor exists; this function turns that into
        # the could-not-check SENTENCE below, which is the caller-visible half.
        warnings.simplefilter("ignore", UserWarning)
        mde = minimum_detectable_effect(group_size, comparison_size, alpha=alpha)

    # BGL-G001 (2026-09-17). Returning None from this function means ADEQUATE
    # POWER, and `nan > 0.20` is False, so an MDE that could not be computed
    # fell through both bands and cleared the subgroup. Could-not-check is its
    # own state and is said in words.
    if not np.isfinite(mde):
        return (
            f"Statistical power COULD NOT BE ASSESSED (n={group_size}): the smallest "
            f"disparity this test could detect was not computable, so a 'not "
            f"significant' reading here is not evidence that nothing is wrong."
        )

    # BGL-5 (2026-09-27). A SECOND GUARD ON THE SAME SENTENCE, independent of the
    # producer above. The band below prints "can only detect disparities larger
    # than {mde:.0%}", and a difference of two proportions cannot exceed 100%, so
    # any mde at or above 1.0 turns that sentence into a floor no disparity can
    # reach. Measured before, when minimum_detectable_effect clamped to 1.0:
    #   power_warning(2, 100) -> "Very low statistical power (n=2). This test can
    #   only detect disparities larger than 100%. ..."
    # while power_warning(0, 100) and power_warning(1, 100) both said "COULD NOT
    # BE ASSESSED" for the same impossibility. Measured after: n=0, n=1 and n=2
    # all say the design could detect nothing, and n=3 keeps its real floor
    # ("Very low statistical power (n=3) ... larger than 82%"). The clamp is gone
    # from the producer, so with today's arithmetic this branch is reached only by
    # an mde of exactly 1.0; it stays because this function, not its producer,
    # owns the sentence a reader sees. The percent formatting below rounds, so
    # anything from 0.995 up would print as "100%": that is why the test is
    # ``>= 1.0`` on the value and the band below spells out a floor above 0.995
    # to one decimal place instead of rounding it to an unattainable 100%.
    if mde >= 1.0:
        return (
            f"NO detectable disparity (n={group_size}): the smallest disparity this "
            f"test could detect is a {mde:.0%} gap, which is at or beyond the largest "
            f"difference two proportions can have, so nothing short of total "
            f"separation could be found here. This is not a statement that power is "
            f"adequate, and a 'not significant' reading carries no information."
        )

    # A floor of, say, 0.996 is a real measurement, but "{:.0%}" renders it as the
    # unattainable "100%", so the near-boundary case keeps a decimal place.
    mde_text = f"{mde:.0%}" if mde <= 0.995 else f"{mde:.1%}"

    if mde > 0.20:
        return (
            f"Very low statistical power (n={group_size}). "
            f"This test can only detect disparities larger than {mde_text}. "
            f"Smaller real disparities will likely go undetected. "
            f"Absence of a significant finding does not mean absence of bias."
        )
    elif mde > 0.10:
        return (
            f"Limited statistical power (n={group_size}). "
            f"This test can reliably detect disparities larger than {mde_text}. "
            f"Smaller effects may be missed."
        )
    return None


# ELFA: Empirical-Likelihood Fairness Auditing, simultaneous bounds, and
# anytime-valid sequential testing.
#
# Bootstrap CIs are per-metric and resampled. For an AUDIT we want (a) a
# distribution-free CI with a chi-squared limit (Empirical Likelihood;
# Owen 2001: for a proportion the EL ratio equals the binomial LR), (b) a
# SIMULTANEOUS statement over ALL subgroups (multiple-hypothesis; Cherian &
# Candes JMLR 2024 framing, Bonferroni-exact here), and (c) an anytime-valid
# stopping rule (Wald SPRT, the same method FairnessPowerAnalyzer uses,
# decoupled for two plain arrays). No scipy: chi2_{1,1-a} = z**2 with the
# existing _norm_ppf.


def _chi2_1_quantile(confidence_level: float) -> float:
    """1-df chi-squared quantile at ``confidence_level`` (= z**2)."""
    z = _norm_ppf(1.0 - (1.0 - confidence_level) / 2.0)
    return float(z * z)


def empirical_likelihood_ci(
    successes: int,
    n: int,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Empirical-likelihood CI for a proportion (distribution-free, chi2
    limiting). For Bernoulli data the EL ratio equals the binomial LR:
    -2 logR(p) = 2[s ln(phat/p) + (n-s) ln((1-phat)/(1-p))]. The CI is the
    set of p with -2 logR(p) <= chi2_{1,1-alpha}; found by a fine grid scan
    (robust, no optimiser, no scipy).

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).
    """
    confidence_level = validate_confidence_level(confidence_level)
    n = int(n)
    # BGL-G001 (2026-09-17). `s = max(0, min(successes, n))` SILENTLY CLAMPED an
    # impossible count pair into a plausible one and then reported it as a
    # measurement: empirical_likelihood_ci(60, 50) answered rate 1.0 with the
    # interval [0.9625, 1.0], indistinguishable from a genuine 50-of-50. Same
    # guard, same wave and same wording as bayesian_proportion_ci / risk_ratio /
    # odds_ratio / wilson_score_interval in this file: an impossible table is a
    # caller bug, not a could-not-check, so it fails loudly.
    if n < 0 or int(successes) < 0 or int(successes) > n:
        raise ConfigurationError(
            f"empirical_likelihood_ci: successes={successes}, n={n} is not a valid "
            f"count pair (need 0 <= successes <= n)"
        )
    s = int(successes)
    if n <= 0:
        # H-05. This returned point 0.0 with the interval [0.0, 0.0]: a
        # ZERO-WIDTH 95% confidence interval, which is maximum certainty derived
        # from no evidence at all, and the tightest interval any real sample
        # could ever produce. `bootstrap_ci` in this same module already gets
        # this right, returning nan/nan/nan with an explicit warning in its
        # metadata, and this now matches it.
        warnings.warn(
            "empirical_likelihood_ci: no observations (n=0), so no interval was "
            "computed. Returning NaN rather than a zero-width interval at 0.0, "
            "which would read as perfect certainty.",
            UserWarning,
            stacklevel=2,
        )
        return StatisticalResult(
            point_estimate=np.nan,
            lower_bound=np.nan,
            upper_bound=np.nan,
            interval_type=IntervalType.CONFIDENCE,
            confidence_level=confidence_level,
            method="empirical_likelihood",
            sample_size=0,
            metadata={"warning": "Insufficient data for empirical likelihood"},
        )
    phat = s / n
    crit = _chi2_1_quantile(confidence_level)

    def neg2logR(p: float) -> float:  # noqa: N802  # statistical notation
        p = min(1 - 1e-12, max(1e-12, p))
        t = 0.0
        if s > 0:
            t += s * np.log(phat / p) if phat > 0 else 0.0
        if n - s > 0:
            t += (n - s) * (np.log((1 - phat) / (1 - p)) if phat < 1 else 0.0)
        return 2.0 * t

    grid = np.linspace(1e-6, 1 - 1e-6, 4000)
    inside = [p for p in grid if neg2logR(float(p)) <= crit]
    if inside:
        lo, hi = float(min(inside)), float(max(inside))
    else:  # degenerate (s=0 or s=n): one-sided
        lo, hi = (0.0, phat) if phat == 0 else (phat, 1.0)
    return StatisticalResult(
        point_estimate=phat,
        lower_bound=lo,
        upper_bound=hi,
        interval_type=IntervalType.CONFIDENCE,
        confidence_level=confidence_level,
        method="empirical_likelihood",
        sample_size=n,
    )


def simultaneous_disparity_bounds(
    group_counts: Dict[str, Tuple[int, int]],
    confidence_level: float = 0.95,
    family_correction: Literal["bonferroni"] = "bonferroni",
) -> Dict[str, Any]:
    """Simultaneous EL bounds over EVERY subgroup at once.

    ``group_counts``: {group: (successes, n)}. Each group gets an EL CI at
    the family-adjusted level (Bonferroni: 1 - alpha/m), so the joint
    statement "all group rates lie in their intervals simultaneously" holds
    at ``confidence_level``. Also returns the most adverse group-vs-group
    ratio/difference still consistent with the simultaneous bounds.

    Raises:
        ConfigurationError: If confidence_level is not strictly inside (0, 1).
    """
    confidence_level = validate_confidence_level(confidence_level)
    groups = {g: (int(s), int(n)) for g, (s, n) in group_counts.items() if int(n) > 0}
    m = max(1, len(groups))
    alpha = 1.0 - confidence_level
    per_level = 1.0 - (alpha / m if family_correction == "bonferroni" else alpha)
    out: Dict[str, Dict[str, float]] = {}
    for g, (s, n) in groups.items():
        ci = empirical_likelihood_ci(s, n, per_level)
        out[g] = {
            "rate": round(float(ci.point_estimate), 6),
            "ci_low": round(float(ci.lower_bound), 6),
            "ci_high": round(float(ci.upper_bound), 6),
            "n": n,
        }
    lows = [v["ci_low"] for v in out.values()]
    highs = [v["ci_high"] for v in out.values()]
    # 1.0 is perfect parity and 0.0 is no gap: the two most reassuring values
    # on these scales, returned when there were no bounds to compare at all.
    #
    # BGL-G001 (2026-09-17). TWO groups are the minimum for a GROUP-TO-GROUP
    # bound, and the count was never checked, so a single group was compared
    # WITH ITSELF: min(lows) and max(highs) came from the same interval, and the
    # width of one group's own sampling uncertainty was reported as a disparity
    # between groups. Measured at the public entry, before this change:
    #
    #   simultaneous_disparity_bounds({'A': (30, 100)})
    #     -> available            True
    #        worstCaseRatioBound  0.5482      (= 0.216055 / 0.394099, group A's
    #                                         own EL interval divided by itself)
    #        worstCaseDifferenceBound 0.178
    #        interpretation       "... the group-to-group selection ratio is at
    #                              least 0.55 (1.00 = parity)."
    #
    # and silently: no warning, and `available: True`. 0.55 fails the US
    # four-fifths rule, so one group on its own produced a headline disparity
    # finding. `available` now carries the three-state honestly, matching the
    # {"available": False, "reason": ...} shape this function's only in-library
    # caller (classification.selection_rate_disparity_matrix) already builds for
    # the same situation.
    if len(out) < 2:
        reason = (
            "no group produced a usable interval"
            if not out
            else f"only {len(out)} group produced a usable interval, and a "
            f"group-to-group bound needs at least 2"
        )
        warnings.warn(
            f"simultaneous_disparity_bounds: {reason}, so no bound was computed. "
            "Reporting nan for worstCaseRatioBound and worstCaseDifferenceBound, not "
            "1.0 and 0.0, which read as perfect parity, and not one group's own "
            "interval width, which reads as a disparity between groups.",
            UserWarning,
            stacklevel=2,
        )
        worst_ratio = float("nan")
        worst_diff = float("nan")
        available = False
        interpretation = (
            f"COULD NOT CHECK: {reason}, so no simultaneous group-to-group bound "
            "exists. This is not a finding of parity and not a finding of disparity."
        )
    else:
        available = True
        worst_ratio = (min(lows) / max(highs)) if max(highs) > 0 else float("nan")
        worst_diff = max(highs) - min(lows)
        interpretation = (
            "Even in the worst case jointly consistent with the data at "
            f"{int(confidence_level * 100)}% confidence across all "
            f"{len(out)} subgroups, the group-to-group selection ratio is at "
            f"least {worst_ratio:.2f} (1.00 = parity)."
        )
    result: Dict[str, Any] = {
        "available": available,
        "method": "empirical_likelihood",
        "familyCorrection": family_correction,
        "nGroups": m,
        "levelPerGroup": round(per_level, 6),
        "confidenceLevel": confidence_level,
        "groups": out,
        "worstCaseRatioBound": round(float(worst_ratio), 4),
        "worstCaseDifferenceBound": round(float(worst_diff), 4),
        "interpretation": interpretation,
    }
    if not available:
        result["reason"] = reason
    return result


def sequential_fairness_test(
    outcomes_a,
    outcomes_b,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    effect_size: float = 0.2,
) -> Dict[str, Any]:
    """Anytime-valid Wald SPRT for "do groups A and B differ?", the same
    boundaries/LLR FairnessPowerAnalyzer uses, decoupled for two arrays.
    Lets an audit stop as soon as there is (or is not) enough evidence,
    without alpha-spending penalties for repeated looks."""
    a = np.asarray(outcomes_a, dtype=float).ravel()
    b = np.asarray(outcomes_b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"available": False, "reason": "Need >=2 observations per group."}
    beta = 1.0 - power
    upper = float(np.log((1 - beta) / alpha)) if alpha > 0 else 10.0
    lower = float(np.log(beta / (1 - alpha))) if (1 - alpha) > 0 else -10.0
    pooled = (
        np.sqrt(
            (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1))
            / max(1, len(a) + len(b) - 2)
        )
        if (len(a) > 1 and len(b) > 1)
        else 1.0
    )
    # READINESS-6, 2026-09-10. `pooled = max(float(pooled), 1e-10)` MANUFACTURES
    # FINDINGS. It reads as a divide-by-zero guard and it is one, but what it
    # guards is the crash: it converts an undefined standardised difference into
    # an enormous finite one, and this function's output is a VERDICT.
    #
    # Measured before this change, two CONSTANT arms (no variance in either):
    #
    #   gap 0.5     -> reject_h0_groups_differ
    #   gap 0.01    -> reject_h0_groups_differ
    #   gap 0.001   -> reject_h0_groups_differ
    #   gap 0.0001  -> reject_h0_groups_differ
    #
    # while the SAME gaps with ordinary noise correctly returned
    # continue_insufficient_evidence for everything below 0.5. So the test
    # declared a difference from a gap five thousand times smaller than the one
    # it refuses to call on real data, and it did so with maximal confidence,
    # because obs = gap / 1e-10 sends the LLR past the Wald boundary at once.
    #
    # This is the opposite direction from the rest of the READINESS-6 register
    # and the more expensive one: a fabricated PASS costs a missed defect, a
    # fabricated FINDING costs a remediation programme aimed at nothing, and its
    # number looks MORE certain the smaller the real difference is.
    #
    # Constant arms are not exotic here. The pulse path scores LLM output at
    # temperature 0, where repeated runs return identical text.
    #
    # `np.ptp` rather than `var(...) == 0`: variance is an ACCUMULATED statistic
    # and a constant array of a value that is not exactly representable in
    # binary has a tiny non-zero variance at some lengths and exactly zero at
    # others (np.var of twenty 0.9s is 4.93e-32; of twenty-five, exactly 0.0).
    # ptp asks the question on the raw data and has no epsilon to argue about.
    if float(np.ptp(a)) == 0.0 and float(np.ptp(b)) == 0.0:
        gap = float(b.mean() - a.mean())
        if gap == 0.0:
            # Every observation identical in both arms. The absence of a
            # difference is OBSERVED, not inferred, and refusing here would
            # discard a real confirmation.
            return {
                "available": True,
                "decision": "accept_h0_no_difference",
                "logLikelihoodRatio": None,
                "upperBoundary": round(upper, 4),
                "lowerBoundary": round(lower, 4),
                "nEffectivePerArm": int(min(len(a), len(b))),
                "standardisedDifference": 0.0,
                "stoppedEarly": True,
                "interpretation": (
                    f"Both groups returned the identical constant value "
                    f"{float(a[0]):.6g} on every observation ({len(a)} and {len(b)}), so "
                    f"the difference is exactly zero. Established by observation rather "
                    f"than inferred, and no likelihood ratio exists for it. This says "
                    f"nothing about inputs that were not tested."
                ),
            }
        return {
            "available": False,
            "decision": "could_not_check",
            "reason": "zero variance in both arms",
            "logLikelihoodRatio": None,
            "nEffectivePerArm": int(min(len(a), len(b))),
            "standardisedDifference": None,
            "meanDifference": gap,
            "stoppedEarly": False,
            "interpretation": (
                f"Both groups are constant, so they differ by exactly {gap:.6g} and "
                f"there is no sampling variation to standardise that against. A "
                f"sequential test needs a pooled standard deviation and none exists, "
                f"so no decision was taken. The DIFFERENCE IS REAL AND CERTAIN; what "
                f"could not be established is whether it is large relative to noise, "
                f"because there is no noise. Judge {gap:.6g} against what matters on "
                f"this metric."
            ),
        }

    pooled = float(pooled)
    n_eff = min(len(a), len(b))
    obs = (b.mean() - a.mean()) / pooled
    # Wald SPRT LLR for the two-sample problem (Wald 1945): the standardised
    # difference obs has variance v = 1/len(a) + 1/len(b), so the normal-mean
    # LLR is (d*obs - d^2/2) / v, i.e. (n/2)*(d*obs - d^2/2) for equal arms.
    # The previous n_eff*(d*obs - d^2/2) DOUBLED the evidence, so the Wald
    # boundaries were crossed with half the required data: type I error ran
    # ~2.5x alpha under monitoring and the affirmative fairness verdict fired
    # ~30% of the time at the designed detectable effect (audit 2026-08-22).
    var_obs = 1.0 / len(a) + 1.0 / len(b)
    # TWO-SIDED, and this is a correction (2026-09-25). The question this function
    # states is "do groups A and B differ", which is two-sided, and the LLR was
    # one-sided: `(d*obs - d^2/2)/var_obs` is the evidence for the single
    # alternative "b exceeds a by d". A difference in the OTHER direction drove
    # the LLR strongly negative, past the lower Wald boundary, and the function
    # returned decision "accept_h0_no_difference" with the interpretation
    # "Anytime-valid: no meaningful difference detected."
    #
    # Measured on this repo before the change, 100 per arm, A at 50% and B at 0%:
    #
    #   sequential_fairness_test(A, B) -> accept_h0_no_difference  (LLR -15.07)
    #   sequential_fairness_test(B, A) -> reject_h0_groups_differ  (LLR +13.07)
    #
    # The same disparity, the arms swapped, opposite verdicts. Half of all real
    # disparities were reported as no difference, and which half depended only on
    # the order the caller passed the groups in. The control test that was meant
    # to prove the detector fires used b = a + 0.8, the one sign where it did.
    #
    # The fix is the standard two-sided Wald SPRT: the LLR of the mixture of the
    # two alternatives +d and -d with equal weight, log(cosh(t)) - c, computed
    # through logaddexp so a large t cannot overflow. It is symmetric in the arms
    # by construction, so no test order can change a verdict again.
    #
    # KEEP the /var_obs scaling below: the previous n_eff*(...) form DOUBLED the
    # evidence, so the boundaries were crossed with half the required data, type I
    # error ran ~2.5x alpha under monitoring and the affirmative fairness verdict
    # fired ~30% of the time at the designed detectable effect (audit 2026-08-22).
    _t = effect_size * obs / var_obs
    _c = effect_size**2 / (2.0 * var_obs)
    llr = float(np.logaddexp(_t, -_t) - np.log(2.0) - _c)
    decision = (
        "reject_h0_groups_differ"
        if llr >= upper
        else "accept_h0_no_difference"
        if llr <= lower
        else "continue_insufficient_evidence"
    )
    return {
        "available": True,
        "decision": decision,
        "logLikelihoodRatio": round(float(llr), 4),
        "upperBoundary": round(upper, 4),
        "lowerBoundary": round(lower, 4),
        "nEffectivePerArm": int(n_eff),
        "standardisedDifference": round(float(obs), 4),
        "stoppedEarly": decision != "continue_insufficient_evidence",
        "interpretation": (
            "Anytime-valid: a real difference between the groups is established."
            if decision.startswith("reject")
            else "Anytime-valid: no meaningful difference detected."
            if decision.startswith("accept")
            else "Not enough evidence yet: collect more data before concluding."
        ),
    }


# ==========================================================================
# Design power: can this test EVER fire?
#
# READINESS-6, 2026-09-10. `_testable` above removes a test that did not run,
# and that is only half the problem. A test that DID run can still be incapable
# of producing a significant answer, because a discrete test's p-value has a
# FLOOR set by its design. Fisher's exact on 6 items per arm cannot return
# anything below 0.0021645, however total the separation is. A permutation test
# with B resamples cannot return below 1/(B+1). When that floor sits above the
# threshold the family will apply, the detector cannot fire for ANY data, and
# "not significant" is then reported to a reader who reads it as "no disparity
# was found".
#
# Measured on this library the same day, before the fix: a pulse probe against a
# stub refusing 100 percent of requests for one demographic name and 0 percent
# for every other produced NO finding at all, because its Fisher floor of
# 0.0021645 sat above the rank-1 Benjamini-Hochberg bar of 0.05/32. A
# calibration sufficiency test returned passes=True for a score band where one
# group's realised positive rate was 94.3 percent against another's 55.6 percent
# at the same predicted score, because its permutation floor of 1/201 times 12
# bins exceeded alpha. Nine detectors in this library run a discrete test and
# only one carried this check; this is that check, shared.
#
# FAIL-SAFE DIRECTION. Declaring a design "not detectable" when it actually
# could have fired would suppress a real finding, which is the harm this whole
# audit exists to remove. So the floor returned here is the SMALLEST p the test
# could produce under any plausible shape of the data, not the largest. For
# Mann-Whitney that matters concretely: with distinct values scipy resolves to
# the exact test (n=5 per side floors at 0.0079365), while with ties it falls
# back to the tie-corrected asymptotic one, whose floor is LOWER (0.0039768).
# Taking the minimum means a design is called undetectable only when even the
# most favourable arrangement of the data could not clear the bar.
# ==========================================================================


def min_attainable_p_fisher(n_reference: int, n_arm: int) -> Optional[float]:
    """Smallest p Fisher's exact can return for a 2x2 of these arm sizes.

    That is the p of a perfect split: none of the reference arm's items in the
    counted cell, all of the compared arm's. Returns ``None`` when it cannot be
    computed, which is could-not-check and never a number.
    """
    if n_reference <= 0 or n_arm <= 0:
        return None
    try:
        from scipy import stats as _st

        _, p = _st.fisher_exact([[int(n_reference), 0], [0, int(n_arm)]])
        p = float(p)
        return p if np.isfinite(p) else None
    except Exception:
        # None is could-not-check and the caller reports it as such, but the
        # CAUSE must not be discarded with it: a reader who sees "the floor was
        # not computable" needs to be able to find out why. The house pattern,
        # as in _validation._finite_values_are_integral.
        logger.debug(
            "min_attainable_p_fisher could not compute a floor for a %sx%s design",
            n_reference,
            n_arm,
            exc_info=True,
        )
        return None


def min_attainable_p_mannwhitney(n_a: int, n_b: int) -> Optional[float]:
    """Smallest p a two-sided Mann-Whitney can return for these sample sizes.

    Computed by running the real test on perfectly separated samples, under
    every method scipy might resolve to, and taking the SMALLEST. See the
    fail-safe note above: the tie-corrected asymptotic floor is lower than the
    exact one, and assuming the exact one would call detectable designs dead.
    """
    if n_a <= 0 or n_b <= 0:
        return None
    try:
        from scipy import stats as _st

        distinct_a = list(range(1, int(n_a) + 1))
        distinct_b = list(range(int(n_a) + 1, int(n_a) + int(n_b) + 1))
        tied_a, tied_b = [0.0] * int(n_a), [1.0] * int(n_b)
        candidates = []
        for a, b, method in (
            (distinct_a, distinct_b, "auto"),
            (distinct_a, distinct_b, "asymptotic"),
            (tied_a, tied_b, "auto"),
            (tied_a, tied_b, "asymptotic"),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = _st.mannwhitneyu(a, b, alternative="two-sided", method=method)
                p = float(p)
                if np.isfinite(p):
                    candidates.append(p)
            except Exception:
                logger.debug(
                    "min_attainable_p_mannwhitney: method %r did not produce a floor",
                    method,
                    exc_info=True,
                )
                continue
        return min(candidates) if candidates else None
    except Exception:
        logger.debug(
            "min_attainable_p_mannwhitney could not compute a floor for %s vs %s",
            n_a,
            n_b,
            exc_info=True,
        )
        return None


def _mannwhitney_two_sided_p(a: Any, b: Any) -> Optional[float]:
    """Two-sided Mann-Whitney p, or ``None`` when the test produced no p-value.

    TWO RULES, both because scipy 1.18 changed what a tied input returns.

    1. All values finite and EVERY value tied across both samples: 1.0, the
       exact permutation p. Every relabelling of one repeated value gives the
       same U, so the chance of a U at least as extreme as the observed one is
       exactly 1 (scipy's ``method="exact"`` returns 1.0). scipy <= 1.17
       returned 1.0 here too; scipy 1.18's default asymptotic path divides by a
       zero tie-corrected variance and returns nan. Measured 2026-10-02 on
       scipy 1.18.1, ``mannwhitneyu([1.0] * 30, [1.0] * 31)`` gives
       ``pvalue=nan``, and four call sites passed that nan on, where
       ``nan < alpha`` is False and reads as a measured "not significant".
    2. Any other non-finite p (a non-finite input, a degenerate case scipy
       answers with nan instead of raising): ``None``. That is a test that did
       not run, never a reading. Each caller maps ``None`` onto its own
       could-not-check contract (``None``, ``nan`` plus a flag, a warning).

    Exceptions from scipy propagate unchanged; the callers already decide what
    a raising test means for them.
    """
    from scipy import stats as _st

    a_f = np.asarray(a, dtype=float).ravel()
    b_f = np.asarray(b, dtype=float).ravel()
    pooled = np.concatenate([a_f, b_f])
    if (
        a_f.size > 0
        and b_f.size > 0
        and bool(np.all(np.isfinite(pooled)))
        and float(np.ptp(pooled)) == 0.0
    ):
        return 1.0
    _, p = _st.mannwhitneyu(a_f, b_f, alternative="two-sided")
    p = float(p)
    return p if math.isfinite(p) else None


def min_attainable_p_mcnemar(n_discordant: int) -> Optional[float]:
    """Smallest p an exact two-sided McNemar can return for this many discordant pairs.

    LF-20. McNemar's ENTIRE evidence is the discordant pairs: the ones where the
    two arms of a matched pair disagree. Concordant pairs, however many there
    are, contribute nothing. So the floor is set by ``b + c`` and not by the
    number of pairs, and a design with a thousand pairs that disagree on five of
    them has the power of a five-pair study.

    The exact test is a two-sided binomial on ``b`` successes out of ``b + c`` at
    ``p = 0.5``. Its smallest attainable value is the fully lopsided split, all
    discordant pairs going one way, which has two-sided probability
    ``2 * 0.5 ** (b + c)``, capped at 1.

        b + c = 5   ->  0.0625   cannot reach alpha = 0.05, whatever the split
        b + c = 6   ->  0.03125  can
        b + c = 0   ->  1.0      no evidence exists at all

    That b+c=5 line is why this function exists rather than a comment: five
    discordant pairs is an entirely ordinary result from a counterfactual run,
    and a "not significant" there is an absence of power rather than a finding.

    Fail-safe in the same direction as the siblings above: the value returned is
    the SMALLEST p the design could produce under the most favourable data, so
    :func:`detectability` calls a design undetectable only when even that cannot
    clear the bar. Returns ``None`` when it cannot be computed, which is
    could-not-check and never a number.
    """
    try:
        d = int(n_discordant)
    except (TypeError, ValueError):
        logger.debug(
            "min_attainable_p_mcnemar: %r is not a discordant-pair count",
            n_discordant,
            exc_info=True,
        )
        return None
    if d < 0:
        return None
    if d == 0:
        # No discordant pairs is no evidence, not a perfect agreement finding.
        # 1.0 is the honest floor here and detectability() will refuse it at any
        # alpha below 1, which is the whole point.
        return 1.0
    try:
        from scipy import stats as _st

        # The real test on the most lopsided split, rather than the closed form,
        # so this tracks whatever scipy actually does. The closed form
        # 2 * 0.5 ** d is asserted against it in tests.
        result = _st.binomtest(d, d, 0.5, alternative="two-sided")
        p = float(result.pvalue)
        return min(1.0, p) if np.isfinite(p) else None
    except Exception:
        logger.debug(
            "min_attainable_p_mcnemar could not compute a floor for %s discordant pairs",
            n_discordant,
            exc_info=True,
        )
        return None


def min_attainable_p_sign_flip(n_pairs: int, n_resamples: Optional[int] = None) -> Optional[float]:
    """Smallest p a paired sign-flip permutation test can return.

    LF-20. Two separate floors bound this test, and the binding one is the
    LARGER, because a test cannot beat either limit:

    ``2 ** -n_pairs * 2``
        The exact enumeration floor. With n pairs there are ``2 ** n`` sign
        assignments; the observed one and its mirror are the only ones at least
        as extreme in a two-sided test, so the smallest attainable p is
        ``2 / 2 ** n``. At 4 pairs that is 0.125 and no rearrangement of any
        data can reach 0.05.

    ``1 / (n_resamples + 1)``
        The sampling floor, from the ``(count + 1) / (B + 1)`` estimator this
        library uses. Only applies when the test SAMPLES rather than enumerates.

    Passing ``n_resamples=None`` means exact enumeration and only the first
    applies. Returns ``None`` when it cannot be computed.
    """
    try:
        n = int(n_pairs)
    except (TypeError, ValueError):
        logger.debug("min_attainable_p_sign_flip: %r is not a pair count", n_pairs, exc_info=True)
        return None
    if n < 0:
        # A negative pair count is a caller bug, not a design fact. Answering
        # 1.0 would report "definitely undetectable" from garbage, which is a
        # verdict; None is could-not-check, which is what this is.
        return None
    if n == 0:
        # No pairs is no test. 1.0 rather than None: the floor is genuinely
        # known here, and it is the largest there is.
        return 1.0
    # BGL-G001 (2026-09-17). `2.0 ** n` raises OverflowError for n >= 1024, and
    # 1024 paired observations is an ordinary counterfactual run, not an exotic
    # input: min_attainable_p_sign_flip(2000) raised
    # `OverflowError: (34, 'Result too large')` and took the whole paired
    # analysis down with it (llm/paired.py calls this on len(ref)).
    # math.ldexp computes the same 2 ** (1 - n) and underflows smoothly to 0.0
    # instead of raising, which is the honest answer at double precision: with
    # 2000 pairs the enumeration floor really is about 1e-602, i.e. zero to a
    # float, and a floor of zero means the design can always reject, which is
    # the fail-safe direction this whole section documents.
    exact = min(1.0, math.ldexp(1.0, 1 - n))
    if n_resamples is None:
        return exact
    sampled = min_attainable_p_permutation(n_resamples)
    if sampled is None:
        return None
    # The LARGER of the two: a sampled test cannot beat its own resample floor,
    # and it cannot beat the enumeration floor either.
    return max(exact, sampled)


def min_attainable_p_permutation(n_permutations: int) -> Optional[float]:
    """Smallest p a permutation test with this many resamples can return.

    The library's permutation tests use the ``(count + 1) / (B + 1)`` estimator,
    which is the standard bias correction and never returns 0. Its floor is
    therefore ``1 / (B + 1)``, and at ``B = 0`` that floor is 1.0: the test
    cannot produce evidence of anything.
    """
    try:
        b = int(n_permutations)
    except (TypeError, ValueError):
        logger.debug(
            "min_attainable_p_permutation: %r is not a resample count",
            n_permutations,
            exc_info=True,
        )
        return None
    if b < 0:
        return None
    return 1.0 / (b + 1.0)


def detectability(
    min_p: Optional[float],
    n_family: int,
    alpha: float = 0.05,
) -> Tuple[Optional[bool], str]:
    """Could a test with this floor ever be significant in a family this size?

    ``n_family`` is the number of hypotheses corrected together (1 for an
    uncorrected test). The bar used is the Benjamini-Hochberg rank-1 threshold,
    ``alpha / n_family``, which is also the Bonferroni threshold, so this is the
    right question for both: could this test fire on its OWN evidence, without
    depending on other findings in the family to lift its rank.

    Returns ``(detectable, note)``. ``detectable`` is ``None`` when the floor
    could not be computed, which is could-not-check and is not the same as
    ``False``. ``note`` is empty when the design has power, and otherwise says
    in words what a "not significant" reading there does NOT mean.
    """
    if min_p is None or not np.isfinite(min_p):
        return None, (
            "COULD NOT CHECK: the smallest p-value this test's design could "
            "produce was not computable, so whether it could ever report a "
            "finding is unknown."
        )
    try:
        m = max(1, int(n_family))
    except (TypeError, ValueError):
        logger.debug("detectability: %r is not a family size", n_family, exc_info=True)
        return None, (
            "COULD NOT CHECK: the family size was not a number, so the "
            "significance bar this test must clear is unknown."
        )
    bar = float(alpha) / m
    if float(min_p) <= bar:
        return True, ""
    return False, (
        "NOT DETECTABLE: the smallest p-value this test's design can produce is "
        "{0:.4g}, above the {1:.4g} bar it must clear ({2:.4g} over a family of "
        "{3}). No data, however extreme, could make this test significant. A "
        "'not significant' reading here is an absence of statistical power, not "
        "evidence that nothing is wrong. Raise the sample size, the number of "
        "resamples, or reduce the family.".format(min_p, bar, alpha, m)
    )
