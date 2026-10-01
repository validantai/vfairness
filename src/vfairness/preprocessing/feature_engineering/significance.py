"""
Statistical significance testing for paired before / after fairness metrics.

CS-T-54. Given a fairness metric measured on a "before" dataset (for example a
baseline model / raw features) and an "after" dataset (post-intervention /
transformed features), tests whether the change in the metric is statistically
significant rather than sampling noise.

Approach (paired bootstrap):
    - Compute the metric on each side (point estimates).
    - Bootstrap each side independently (resample indices with replacement) to
      get per-side confidence intervals, reusing
      ``evaluation.vfairness_metrics._statistics.bootstrap_ci`` and the existing
      metric functions.
    - Bootstrap the delta (metric_after - metric_before) to get its
      distribution, the delta CI, and a two-sided empirical p-value. When the
      two sides are row-aligned (same rows before and after an intervention,
      the primary use case), ONE index vector is drawn per replicate and both
      sides are evaluated on it, so the resampling preserves the pairing and
      the delta distribution has the variance of the paired difference
      (Efron & Tibshirani 1993, ch. 8). Only for genuinely independent samples
      are the two sides resampled with separate index vectors.
    - Report Cohen's h between the two most-disparate group selection rates as a
      practical-significance effect size.

All randomness is seeded via ``random_state`` for reproducibility.

References:
    - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
      (Cohen's h for the difference between two proportions).
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from vfairness._not_assessed import NOT_ASSESSED, warn_not_assessed
from vfairness.evaluation.vfairness_metrics._statistics import (
    bootstrap_ci,
    cohens_h,
    cohens_h_interpretation,
)
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)

# Map the metric name to (function, human label, whether larger means fairer).
# All the difference metrics: smaller is fairer. The ratio metric: larger (up to
# 1.0) is fairer.
_METRIC_FUNCS = {
    "demographic_parity": (demographic_parity_difference, "Demographic parity difference", False),
    "demographic_parity_difference": (
        demographic_parity_difference,
        "Demographic parity difference",
        False,
    ),
    "demographic_parity_ratio": (demographic_parity_ratio, "Demographic parity ratio", True),
    "equal_opportunity": (equal_opportunity_difference, "Equal opportunity difference", False),
    "equal_opportunity_difference": (
        equal_opportunity_difference,
        "Equal opportunity difference",
        False,
    ),
    "equalized_odds": (equalized_odds_difference, "Equalized odds difference", False),
    "equalized_odds_difference": (equalized_odds_difference, "Equalized odds difference", False),
}


def _metric_value(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int,
) -> float:
    """Compute one fairness metric, defaulting to demographic parity difference."""
    fn, _label, _higher_better = _METRIC_FUNCS.get(
        metric, (demographic_parity_difference, "Demographic parity difference", False)
    )
    try:
        return float(fn(y_true, y_pred, sensitive, min_group_size=min_group_size))
    except Exception:
        return float("nan")


def _two_group_rates(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int,
) -> Tuple[float, float]:
    """Return (max_rate, min_rate) of the per-group positive-prediction rates.

    These are the two selection rates that drive the demographic-parity gap.
    Groups with fewer than ``min_group_size`` samples are ignored. If fewer than
    two groups qualify, returns (nan, nan).
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    rates: List[float] = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        if int(np.sum(mask)) >= min_group_size:
            rates.append(float(np.mean(y_pred[mask] == 1)))
    if len(rates) < 2:
        return float("nan"), float("nan")
    return max(rates), min(rates)


def paired_metric_significance(
    y_true_before,
    y_pred_before,
    sens_before,
    y_true_after,
    y_pred_after,
    sens_after,
    metric: str = "demographic_parity",
    n_bootstrap: int = 1000,
    random_state: int = 42,
    min_group_size: int = 2,
    paired: Optional[bool] = None,
) -> Dict:
    """Test whether a fairness metric changed significantly between two datasets.

    Args:
        y_true_before: True labels for the before dataset.
        y_pred_before: Predicted labels for the before dataset.
        sens_before: Protected attribute for the before dataset.
        y_true_after: True labels for the after dataset.
        y_pred_after: Predicted labels for the after dataset.
        sens_after: Protected attribute for the after dataset.
        metric: Fairness metric name (default 'demographic_parity').
        n_bootstrap: Number of bootstrap resamples per side.
        random_state: Seed for all resampling.
        min_group_size: Minimum group size passed to the metric and to the
            group-rate effect size. Kept small (2) so bootstrap resamples of
            small groups do not collapse the metric to its degenerate value.
        paired: Controls the delta bootstrap. True: the two sides describe the
            SAME rows in the same order (before/after an intervention), so each
            replicate draws one index vector and evaluates both sides on it,
            preserving the pairing (raises ValueError if the lengths differ).
            False: the sides are independent samples and are resampled with
            separate index vectors. None (default): auto-detect, paired when
            the lengths match, independent otherwise. Pass paired=False
            explicitly for two equal-length but unrelated datasets.

    Returns:
        Dict with keys: metric, value_before, value_after, delta, ci_before,
        ci_after, delta_ci, p_value, significant, effect_size, interpretation,
        resampling ('paired' or 'independent', the delta bootstrap mode that
        actually ran), n_bootstrap_measured and n_bootstrap_unmeasurable (how
        many replicates produced a finite delta, and how many did not).
        The extra key effect_size_after is also included (Cohen's h on the after
        data) so both the baseline and residual disparity magnitudes are visible;
        effect_size itself is the before-data Cohen's h (the baseline disparity
        the change acts on).

        Three states, never two. ``significant`` is ``None`` and ``p_value`` is
        ``nan`` whenever either the point-estimate ``delta`` is not finite (the
        metric could not be computed on one or both sides, so there is no change
        for a p-value to be about) or fewer than two bootstrap replicates
        produced a finite delta. Both happen with one usable protected group:
        a group holding a single row makes the point estimate unmeasurable while
        replicates that draw that row twice still produce numbers, so the
        replicate count alone is not the test. ``None`` is NOT a synonym for
        ``False``, and ``interpretation`` then states that the change was not
        assessed, naming which of the two reasons it was, rather than a direction.
    """
    ytb = np.asarray(y_true_before)
    ypb = np.asarray(y_pred_before)
    sb = np.asarray(sens_before)
    yta = np.asarray(y_true_after)
    ypa = np.asarray(y_pred_after)
    sa = np.asarray(sens_after)

    _fn, label, higher_better = _METRIC_FUNCS.get(
        metric, (demographic_parity_difference, "Demographic parity difference", False)
    )

    # Point estimates.
    value_before = _metric_value(metric, ytb, ypb, sb, min_group_size)
    value_after = _metric_value(metric, yta, ypa, sa, min_group_size)
    delta = value_after - value_before

    n_before = len(ytb)
    n_after = len(yta)

    # Per-side confidence intervals via the shared bootstrap_ci helper. The
    # "data" is the index vector and the statistic recomputes the metric on the
    # resampled indices.
    idx_before = np.arange(n_before)
    idx_after = np.arange(n_after)

    def _stat_before(sample_idx):
        return _metric_value(
            metric, ytb[sample_idx], ypb[sample_idx], sb[sample_idx], min_group_size
        )

    def _stat_after(sample_idx):
        return _metric_value(
            metric, yta[sample_idx], ypa[sample_idx], sa[sample_idx], min_group_size
        )

    res_before = bootstrap_ci(
        idx_before,
        _stat_before,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state,
    )
    res_after = bootstrap_ci(
        idx_after,
        _stat_after,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state + 1,
    )
    ci_before = [float(res_before.lower_bound), float(res_before.upper_bound)]
    ci_after = [float(res_after.lower_bound), float(res_after.upper_bound)]

    # Delta bootstrap for the delta CI and the two-sided empirical p-value.
    # Done directly (not via bootstrap_ci) because the two-sided p-value needs
    # the joint delta distribution, which bootstrap_ci does not expose.
    #
    # PAIRED MODE (the fix for the wave-6 audit finding): when the two sides
    # are row-aligned, a paired bootstrap must resample the PAIRS jointly, one
    # index vector per replicate applied to both sides (Efron & Tibshirani
    # 1993, ch. 8). Resampling the sides independently is the two-independent-
    # samples bootstrap: it adds both full sampling variances instead of the
    # variance of the paired difference, roughly doubling the delta sd on
    # correlated sides, and real metric changes get reported as sampling noise
    # (measured power 0/30 vs 27/30 for a real 12pp improvement, n=400).
    if paired is None:
        paired_mode = n_before == n_after
    elif paired:
        if n_before != n_after:
            raise ValueError(
                "paired=True requires row-aligned inputs of equal length; got "
                "%d before rows and %d after rows." % (n_before, n_after)
            )
        paired_mode = True
    else:
        paired_mode = False

    rng = np.random.default_rng(random_state)
    deltas = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        if paired_mode:
            idx = rng.integers(0, n_before, size=n_before)
            deltas[b] = _stat_after(idx) - _stat_before(idx)
        else:
            ib = rng.integers(0, n_before, size=n_before)
            ia = rng.integers(0, n_after, size=n_after)
            deltas[b] = _stat_after(ia) - _stat_before(ib)
    n_requested = int(n_bootstrap)
    deltas = deltas[~np.isnan(deltas)]
    n_measured = int(len(deltas))

    # BGL5 2026-09-27. THE POINT ESTIMATE IS THE SUBJECT OF THE TEST. The guard
    # was `n_measured >= 2` alone, so it needed EVERY replicate to fail, and a
    # protected group of a single row fails the point estimate (it is below
    # min_group_size) while the replicates that draw that row twice succeed.
    # Measured on 200 rows with group 'B' holding exactly one of them:
    # value_before nan, value_after nan, delta nan, p_value 1.0, significant
    # False, n_bootstrap_measured 67, n_bootstrap_unmeasurable 133, and the prose
    # "Demographic parity difference was NOT ASSESSED: the metric could not be
    # computed on this data (before nan, after nan, delta nan). This change is not
    # statistically significant (p = 1.0000); it is consistent with sampling
    # noise." That is the exact signature the comment below condemns, reached
    # around the guard: an empirical p-value about a NaN delta is a p-value about
    # nothing. After: p_value nan, significant None, delta_ci [nan, nan], and the
    # sig_phrase names the reason. The 300-row two-group control still reports
    # p 0.0, significant True, delta -0.4143.
    delta_is_measurable = bool(np.isfinite(delta))
    if n_measured >= 2 and delta_is_measurable:
        delta_ci = [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ]
        frac_le = float(np.mean(deltas <= 0))
        frac_ge = float(np.mean(deltas >= 0))
        p_value = float(min(1.0, max(0.0, 2.0 * min(frac_le, frac_ge))))
        significant: Optional[bool] = bool(p_value < 0.05)
        not_assessed_reason: Optional[str] = None
    else:
        # p_value = 1.0 here fired whenever EVERY bootstrap delta was NaN, which
        # is what one protected group produces: the metric needs a between-group
        # comparison, there is none to make, so every replicate is NaN. The
        # result then read value_before=nan, value_after=nan, delta=nan,
        # p_value=1.0, significant=False, which is the signature of a test that
        # ran and found nothing. Measured 2026-09-10 on 60 rows in a single group
        # "A". A p-value of 1.0 is a finding about the null; nobody tested it.
        # The replicate shortage is checked FIRST, so the one-group case keeps the
        # message it has always carried ("only 0 of 50 bootstrap replicates ..."),
        # which a reader and an existing pin both name. The new branch is only for
        # the case that message would be WRONG about: enough finite replicates and
        # an unmeasurable point estimate.
        if n_measured >= 2 and not delta_is_measurable:
            # The point estimate itself does not exist, so there is no change for
            # a p-value to be about, however many replicates happened to produce
            # a number. Reported ahead of the replicate count because it is the
            # reason that decides, and because quoting "only 67 of 200" here would
            # name a requirement (at least 2) that this input MEETS.
            not_assessed_reason = (
                "the delta itself could not be computed (value_before %r, "
                "value_after %r), so there is no change for a p-value to be about"
                % (value_before, value_after)
            )
            warnings.warn(
                "paired_metric_significance: %s, although %d of %d bootstrap "
                "replicates did produce a finite delta. Reporting p_value=nan and "
                "significant=None (could not check), NOT p 1.0 with "
                "significant=False, which is the signature of a test that ran and "
                "found nothing." % (not_assessed_reason, n_measured, n_requested),
                UserWarning,
                stacklevel=2,
            )
        else:
            not_assessed_reason = (
                "only %d of %d bootstrap replicates produced a finite delta, and an "
                "empirical p-value needs at least 2" % (n_measured, n_requested)
            )
            warn_not_assessed(
                "paired_metric_significance",
                measured=n_measured,
                total=n_requested,
                unit="bootstrap replicates produced a finite delta",
                requirement="an empirical p-value needs at least 2",
                reporting="p_value=nan and significant=None",
                instead_of="p 1.0 with significant=False",
                stacklevel=2,
            )
        delta_ci = [float("nan"), float("nan")]
        p_value = float("nan")
        significant = None

    # Effect size: Cohen's h between the two most-disparate group selection
    # rates. Reported on the before data (the baseline disparity the change acts
    # on); the after value is exposed too.
    max_b, min_b = _two_group_rates(ypb, sb, min_group_size)
    max_a, min_a = _two_group_rates(ypa, sa, min_group_size)
    effect_size = float(cohens_h(max_b, min_b)) if not np.isnan(max_b) else float("nan")
    effect_size_after = float(cohens_h(max_a, min_a)) if not np.isnan(max_a) else float("nan")

    interpretation = _interpret(
        label,
        value_before,
        value_after,
        delta,
        p_value,
        significant,
        effect_size,
        higher_better,
        not_assessed_reason,
    )

    return {
        "metric": metric,
        "value_before": value_before,
        "value_after": value_after,
        "delta": delta,
        "ci_before": ci_before,
        "ci_after": ci_after,
        "delta_ci": delta_ci,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
        "effect_size_after": effect_size_after,
        "interpretation": interpretation,
        "resampling": "paired" if paired_mode else "independent",
        "n_bootstrap_measured": n_measured,
        "n_bootstrap_unmeasurable": n_requested - n_measured,
    }


def _interpret(
    label: str,
    value_before: float,
    value_after: float,
    delta: float,
    p_value: float,
    significant: Optional[bool],
    effect_size: float,
    higher_better: bool,
    not_assessed_reason: Optional[str] = None,
) -> str:
    """Build a plain-language interpretation of the paired test.

    Two fabrications used to be written into this sentence, and both survived
    every numeric check because the prose is where they lived.

    1. `improved = (delta < 0)` is False for a NaN delta, so the else branch
       fired and the sentence asserted a DIRECTION about an intervention on
       data where the metric could not be computed at all: "Demographic parity
       difference worsened (less fair) from nan to nan (delta nan)". Measured
       2026-09-10 on 60 rows in a single protected group.
    2. `p_value` was accepted and never read: both branches below hardcoded
       "(p < 0.05)" / "(p >= 0.05)" off the `significant` bool, so the literal
       inequality printed whatever the real p-value was, and printed one at all
       for a p-value nobody computed.
    """
    # For a difference metric, a negative delta (smaller disparity) is an
    # improvement. For a ratio metric, a positive delta (closer to 1) improves.
    # A non-finite delta is not a direction: there is nothing to have moved.
    if not np.isfinite(delta):
        direction = NOT_ASSESSED
    else:
        improved = (delta > 0) if higher_better else (delta < 0)
        if abs(delta) < 1e-9:
            direction = "did not change"
        elif improved:
            direction = "improved (fairer)"
        else:
            direction = "worsened (less fair)"

    p_measured = np.isfinite(p_value)
    if significant is None or not p_measured:
        # The reason travels in from the caller. Hardcoding "no bootstrap replicate
        # produced a finite delta" was accurate for the one-group case and FALSE
        # for a NaN delta with 67 finite replicates, which is the other way into
        # this branch: a sentence that names the wrong cause sends the reader to
        # fix the wrong thing.
        sig_phrase = "Significance was NOT ASSESSED: %s. This is not evidence of no change." % (
            not_assessed_reason
            or "no bootstrap replicate produced a finite delta, so there is no p-value"
        )
    elif significant:
        sig_phrase = "This change is statistically significant (p = %.4f)." % p_value
    else:
        sig_phrase = (
            "This change is not statistically significant (p = %.4f); it is "
            "consistent with sampling noise." % p_value
        )

    if np.isnan(effect_size):
        eff_phrase = "Effect size unavailable (fewer than two sufficiently sized groups)."
    else:
        eff_phrase = "Baseline disparity effect size (Cohen's h) is %.3f (%s)." % (
            effect_size,
            cohens_h_interpretation(effect_size),
        )

    if direction == NOT_ASSESSED:
        return (
            "%s was NOT ASSESSED: the metric could not be computed on this data "
            "(before %.4f, after %.4f, delta %.4f). %s %s"
            % (label, value_before, value_after, delta, sig_phrase, eff_phrase)
        )

    return "%s %s from %.4f to %.4f (delta %.4f). %s %s" % (
        label,
        direction,
        value_before,
        value_after,
        delta,
        sig_phrase,
        eff_phrase,
    )
