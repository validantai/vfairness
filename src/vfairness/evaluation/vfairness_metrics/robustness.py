"""
Statistical Significance and Robustness Testing for vfairness.

This module provides formal hypothesis testing and robustness verification
methods for fairness audits, based on best practices from:
- DiCiccio et al. (2020): Permutation tests for fairness
- Kearns et al. (2018): Subgroup robustness auditing
- Koh & Liang (2017): Influence function analysis

Features:
- Permutation testing for any fairness metric
- Contingency table tests (Chi-square, Fisher's exact)
- Robust statistics (trimmed means, Winsorization)
- Sensitivity analysis for robustness verification
- Subgroup robustness auditing (fairness gerrymandering detection)

References:
    DiCiccio, C., et al. (2020). "Evaluating fairness using permutation tests."
        ACM SIGKDD Conference on Knowledge Discovery & Data Mining.
    Kearns, M., et al. (2018). "Preventing fairness gerrymandering."
        International Conference on Machine Learning.
"""

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Data Structures


@dataclass
class PermutationTestResult:
    """
    Results from a permutation test for fairness significance.

    Attributes:
        observed_statistic: The observed fairness metric value
        p_value: Two-tailed p-value from permutation test
        null_distribution: Array of permuted statistics (null distribution)
        n_permutations: Number of permutations performed
        significant_at_05: Whether significant at alpha=0.05, or None when the
            observed statistic could not be computed at all (COULD NOT CHECK,
            which is neither significant nor not-significant)
        significant_at_01: As above, at alpha=0.01
        effect_direction: 'positive', 'negative', or 'none'
        method: Test method used
        min_attainable_p_value: The SMALLEST p-value this run's design could
            have produced, ``1 / (n_permutations + 1)``. None when it could not
            be computed. A reader compares it to their own alpha to see whether
            a "not significant" reading here could ever have been anything else.
        detectable_at_05: Whether that floor clears 0.05 at all. None is
            could-not-check; False means no data could have made this test
            significant.
        design_note: Empty when the design has power, and otherwise says in
            words what a "not significant" reading here does NOT mean.
        groups_omitted: Groups that were present in the caller's data but absent
            from the rows the statistic reads, so this p-value says nothing about
            them. Empty when the statistic covers every group.
    """

    observed_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int
    significant_at_05: Optional[bool]
    significant_at_01: Optional[bool]
    effect_direction: str
    method: str = "permutation"
    # READINESS-6, 2026-09-10. Design-power disclosure; see permutation_test.
    min_attainable_p_value: Optional[float] = None
    detectable_at_05: Optional[bool] = None
    design_note: str = ""
    # BGL-5 (2026-09-27). The same disclosure ContingencyTestResult carries, for
    # the same reason: a statistic computed over a STRATUM of the data can leave
    # a whole group outside the comparison it is reported as. See
    # permutation_test_equal_opportunity.
    groups_omitted: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        # An emptied null is a real outcome of this test (every permutation
        # unmeasurable), and np.mean/np.std of an empty array is NaN plus a
        # RuntimeWarning. Same value, without the noise on a refusal path that
        # is now reached deliberately.
        has_null = self.null_distribution.size > 0
        return {
            "observed_statistic": self.observed_statistic,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "significant_at_05": self.significant_at_05,
            "significant_at_01": self.significant_at_01,
            "effect_direction": self.effect_direction,
            "method": self.method,
            "null_mean": float(np.mean(self.null_distribution)) if has_null else float("nan"),
            "null_std": float(np.std(self.null_distribution)) if has_null else float("nan"),
            "min_attainable_p_value": self.min_attainable_p_value,
            "detectable_at_05": self.detectable_at_05,
            "design_note": self.design_note,
            "groups_omitted": list(self.groups_omitted),
        }

    def __repr__(self) -> str:
        sig = "***" if self.significant_at_01 else ("**" if self.significant_at_05 else "")
        return (
            f"PermutationTestResult(statistic={self.observed_statistic:.4f}, "
            f"p={self.p_value:.4f}{sig})"
        )


@dataclass
class ContingencyTestResult:
    """
    Results from contingency table analysis (Chi-square or Fisher's exact).

    Attributes:
        test_used: 'chi_square' or 'fisher_exact'
        statistic: Test statistic (chi-square value or odds ratio)
        p_value: P-value from the test
        contingency_table: The 2x2 (or larger) contingency table
        expected_counts: Expected counts under null hypothesis
        significant_at_05: Whether significant at alpha=0.05, or None when the
            question was not answered (no test was run, or the design could not
            have reached 0.05 on any data). Never False for those.
        odds_ratio: Odds ratio (for 2x2 tables)
        cramers_v: Cramer's V effect size
        min_attainable_p_value: The SMALLEST p-value this table's design could
            have produced. None when it was not computed, which is
            could-not-check and not a claim of power.
        detectable_at_05: Whether that floor clears 0.05 at all. None is
            could-not-check; False means no data at these group sizes could have
            made this test significant.
        design_note: Empty when the design has power, and otherwise says in
            words what a "not significant" reading here does NOT mean.
        groups_omitted: Groups that were present in the caller's data but absent
            from the stratum this table was built on, so this p-value says
            nothing about them. Empty when the table covers every group.
    """

    test_used: str
    statistic: float
    p_value: float
    contingency_table: np.ndarray
    expected_counts: np.ndarray
    significant_at_05: Optional[bool]
    odds_ratio: Optional[float] = None
    cramers_v: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    # BGL-3 evaluation-1, 2026-09-27. Design-power disclosure, the same three
    # fields PermutationTestResult already carries; see contingency_test.
    min_attainable_p_value: Optional[float] = None
    detectable_at_05: Optional[bool] = None
    design_note: str = ""
    groups_omitted: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "test_used": self.test_used,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "contingency_table": self.contingency_table.tolist(),
            "expected_counts": self.expected_counts.tolist(),
            "significant_at_05": self.significant_at_05,
            "odds_ratio": self.odds_ratio,
            "cramers_v": self.cramers_v,
            "degrees_of_freedom": self.degrees_of_freedom,
            "min_attainable_p_value": self.min_attainable_p_value,
            "detectable_at_05": self.detectable_at_05,
            "design_note": self.design_note,
            "groups_omitted": list(self.groups_omitted),
        }


@dataclass
class RobustMetricsResult:
    """
    Results from robust fairness metric computation.

    Attributes:
        group: Group name
        n_samples: Number of samples in group
        standard_value: Standard (non-robust) metric value
        trimmed_value: Trimmed mean value
        winsorized_value: Winsorized mean value
        median_value: Median value
        outlier_influence_detected: Whether outliers significantly affect the
            metric. None means the question COULD NOT BE ANSWERED for this
            group, not that the answer is no: see compute_robust_metrics for the
            two shapes that reach it (a group too small for the trim to remove
            any observation, and a divergence ratio with no usable denominator).
        divergence_ratio: Ratio of divergence between standard and robust
            estimates, or NaN when that ratio does not exist. Never 0.0 for an
            unmeasured comparison, because 0.0 reads as "the robust estimate
            agrees with the standard one".
        trim_effective: Whether the trim actually removed observations, so the
            trimmed mean is capable of disagreeing with the standard mean at all.
    """

    group: str
    n_samples: int
    standard_value: float
    trimmed_value: float
    winsorized_value: float
    median_value: float
    outlier_influence_detected: Optional[bool]
    divergence_ratio: float
    # BGL-3 evaluation-1, 2026-09-27. Says whether the robustness comparison
    # could have fired at all; see compute_robust_metrics.
    trim_effective: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "group": self.group,
            "n_samples": self.n_samples,
            "standard_value": self.standard_value,
            "trimmed_value": self.trimmed_value,
            "winsorized_value": self.winsorized_value,
            "median_value": self.median_value,
            "outlier_influence_detected": self.outlier_influence_detected,
            "divergence_ratio": self.divergence_ratio,
            "trim_effective": self.trim_effective,
        }


@dataclass
class SensitivityResult:
    """
    Results from sensitivity/robustness analysis.

    Attributes:
        original_metric: Original fairness metric value
        perturbed_metrics: List of metrics under perturbations
        mean_perturbed: Mean of perturbed metrics
        std_perturbed: Standard deviation of perturbed metrics
        max_deviation: Maximum absolute deviation from original
        is_robust: Whether the metric is robust to perturbations. None means
            the question could not be answered, NOT that the answer is no.
        robustness_score: Score from 0 (fragile) to 1 (robust), or None when
            no score could be computed. Never a number standing in for one.
        perturbation_type: Type of perturbation applied
        n_iterations_run: Perturbation draws whose metric was measurable
        n_unmeasurable: Draws whose metric came back non-finite and so carry
            no information about robustness either way
    """

    original_metric: float
    perturbed_metrics: np.ndarray
    mean_perturbed: float
    std_perturbed: float
    max_deviation: float
    is_robust: Optional[bool]
    robustness_score: Optional[float]
    perturbation_type: str
    n_iterations_run: int = 0
    n_unmeasurable: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "original_metric": self.original_metric,
            "mean_perturbed": self.mean_perturbed,
            "std_perturbed": self.std_perturbed,
            "max_deviation": self.max_deviation,
            "is_robust": self.is_robust,
            "robustness_score": self.robustness_score,
            "perturbation_type": self.perturbation_type,
            "n_iterations_run": self.n_iterations_run,
            "n_unmeasurable": self.n_unmeasurable,
        }


@dataclass
class SubgroupAuditResult:
    """
    Results from subgroup robustness audit (fairness gerrymandering detection).

    Attributes:
        subgroup_metrics: Dict mapping subgroup to metrics
        flagged_subgroups: List of (subgroup, disparity) tuples that exceed threshold
        worst_subgroup: The subgroup with largest disparity, or None when no
            subgroup could be measured
        worst_disparity: The largest disparity found. NaN when no subgroup was
            measured, never 0.0: zero disparity is perfect parity across every
            subgroup, the strongest all-clear this result can state.
        gerrymandering_detected: Whether fairness gerrymandering is suspected.
            None means the question could not be answered, NOT that the answer
            is no.
        n_subgroups_analyzed: Number of subgroups with sufficient samples
        n_subgroups_flagged: Number of flagged subgroups
        n_subgroups_too_small: Subgroups dropped for holding fewer than
            min_subgroup_size rows, and so carrying no evidence either way
        n_subgroups_unmeasurable: Subgroups dropped because the requested metric
            came back non-finite for them (e.g. 'fpr' on a cell with no negative
            labels)
        n_rows_missing_prediction: Rows removed before measuring because their
            prediction (or, for a label-based metric, their label) was missing
    """

    subgroup_metrics: Dict[str, Dict]
    flagged_subgroups: List[Tuple[str, float]]
    worst_subgroup: Optional[str]
    worst_disparity: float
    gerrymandering_detected: Optional[bool]
    n_subgroups_analyzed: int
    n_subgroups_flagged: int
    n_subgroups_too_small: int = 0
    n_subgroups_unmeasurable: int = 0
    n_rows_missing_prediction: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "n_rows_missing_prediction": self.n_rows_missing_prediction,
            "subgroup_metrics": self.subgroup_metrics,
            "flagged_subgroups": self.flagged_subgroups,
            "worst_subgroup": self.worst_subgroup,
            "worst_disparity": self.worst_disparity,
            "gerrymandering_detected": self.gerrymandering_detected,
            "n_subgroups_analyzed": self.n_subgroups_analyzed,
            "n_subgroups_flagged": self.n_subgroups_flagged,
            "n_subgroups_too_small": self.n_subgroups_too_small,
            "n_subgroups_unmeasurable": self.n_subgroups_unmeasurable,
        }


# Permutation Testing


# Above this many rows the number of distinct label arrangements is so large
# that the enumeration floor is far below any resample floor, and computing the
# exact multinomial is pointless work. The resample floor binds there.
_MAX_ENUMERABLE_ROWS = 1000
# A count this large already puts the floor below 1e-15; converting a bigger
# Python int to float can overflow, so stop before that rather than after.
_MAX_ARRANGEMENTS = 10**15


def _label_arrangement_floor(sensitive_attr: np.ndarray, alternative: str) -> Optional[float]:
    """Smallest p the DISTINCT LABEL ARRANGEMENTS allow, or None if not binding.

    A permutation test shuffles the group labels, so the number of genuinely
    different null datasets is the multinomial coefficient
    ``n! / (n_1! * ... * n_k!)`` -- 6 for a 2/2 split of 4 rows, no matter how
    many resamples are drawn. No amount of resampling can produce a p-value
    below what that enumeration allows.

    Mirrors ``min_attainable_p_sign_flip`` in ``_statistics.py``, including its
    two-sided treatment of RELABELINGS. A relabeling of the group names maps one
    arrangement to another valid arrangement of the SAME label multiset exactly
    when it preserves the count of every group, and a two-sided statistic (which
    is read through ``|statistic|``) is unchanged by it. So the number of
    arrangements guaranteed to be at least as extreme as the observed one is the
    size of that relabeling group, ``prod(m_v!)`` over the distinct counts ``v``,
    where ``m_v`` is how many groups have count ``v``.

    BGL-S2B (2026-09-17). This used to read "2 if two-sided and exactly two
    groups", which is that formula ONLY when the two groups are the same size.
    Both directions were wrong:

    * UNEQUAL two-group splits were doubled anyway, and the swap they were
      doubled for is not an arrangement of this label multiset at all. Measured:
      a 1/29 split of 30 rows reports 2/30 = 0.0667, is declared "no data could
      make this test significant", and refuses; brute force over all 30
      arrangements shows the most extreme data reaches 1/30 = 0.0333, and the
      test returns p = 0.032. A measurable shape was refused.
    * More than two groups fell back to ``1 / n_arrangements``, which UNDER-states
      by the whole relabeling multiplicity. Measured: three groups of two rows
      report 1/90 = 0.0111 and ``detectable_at_05=True`` while brute force over
      every possible ``y_pred`` at that shape shows the smallest reachable p is
      0.2000. That was an affirmative assurance that data COULD have made the
      test fire when none could, and the ``significant_at_05=False`` graded off
      it read as a measured negative.

    The value returned is a LOWER bound on the true floor, not the floor itself:
    a specific ``metric_func`` may tie MORE arrangements than the relabelings do.
    So ``floor > alpha`` proves the design cannot fire, while ``floor <= alpha``
    only means the label enumeration does not rule it out.

    Returns ``None`` when the floor is not computable or is so small that the
    resample floor binds instead; the caller takes the max of the floors it has.
    """
    labels = np.asarray(sensitive_attr)
    if labels.size == 0:
        return None
    counts = np.unique(labels, return_counts=True)[1]
    if counts.size < 2:
        # One group is no comparison; permutation_test's own metric guard and
        # the collapsed-null branch handle that case and say so there.
        return None
    n = int(counts.sum())
    if n > _MAX_ENUMERABLE_ROWS:
        return None

    n_arrangements = math.factorial(n)
    for c in counts:
        n_arrangements //= math.factorial(int(c))
    if n_arrangements <= 0 or n_arrangements > _MAX_ARRANGEMENTS:
        return None

    # The relabelings that map this label multiset to itself: prod(m_v!) over
    # the distinct group sizes. Equal 2/2 gives 2 (the old special case), an
    # unequal 1/29 gives 1, three groups of two give 6. Only claimed for the
    # two-sided reading, where |statistic| is what is compared; a one-sided test
    # is not invariant under a relabeling that flips the sign, so it keeps the
    # conservative single arrangement (the observed one, which is always at
    # least as extreme as itself).
    numerator = 1.0
    if alternative == "two-sided":
        size_multiplicities = np.unique(counts, return_counts=True)[1]
        relabelings = 1
        for m_v in size_multiplicities:
            relabelings *= math.factorial(int(m_v))
        numerator = float(relabelings)
    return min(1.0, numerator / n_arrangements)


def _effective_design_floor(
    sensitive_attr: np.ndarray, n_design_rows: Optional[int]
) -> Optional[float]:
    """Floor implied by a statistic that reads only SOME of the rows.

    BGL-S2B (2026-09-17). ``_label_arrangement_floor`` counts the arrangements
    of the WHOLE label array, but a statistic is free to read a subset of it.
    ``permutation_test_equal_opportunity`` computes its TPR spread over the rows
    with ``y_true == 1`` only, so on 200 rows with six positives the reported
    floor was the resample floor 9.999e-05 while the six labels that the
    statistic can actually see admit vastly fewer distinct designs. A floor
    orders of magnitude below anything reachable is an affirmative
    ``detectable_at_05=True`` for a test no data could make fire, and the
    ``significant_at_05=False`` graded off it reads as a measured negative.

    The bound. Shuffling the full label array induces a random assignment of
    labels to the ``m`` rows the statistic reads. That assignment is the ONLY
    thing the statistic sees, the permutation distribution over it is exactly
    known, and the set of draws "at least as extreme" always contains the
    observed assignment itself. So the p-value can never be smaller than the
    probability of the LEAST likely assignment:

        ``min_a  (n-m)!/n! * prod(c_i! / (c_i - a_i)!)``

    over the per-group intakes ``a_i`` summing to ``m``. Writing ``d_i = c_i-a_i``
    this is minimised by MAXIMISING ``prod(d_i!)`` at fixed ``sum(d_i) = n-m``,
    and because ``log(d!)`` is superadditive that maximum is reached by filling
    the largest groups to their caps first. Closed form, no enumeration.

    Like ``_label_arrangement_floor`` this is a LOWER bound on the true floor,
    so it can only ever prove a design powerless, never wrongly refuse one that
    has power. Returns ``None`` when it does not apply (the statistic reads every
    row, or the counts are unusable).
    """
    if n_design_rows is None:
        return None
    labels = np.asarray(sensitive_attr)
    n = int(labels.size)
    m = int(n_design_rows)
    if n == 0 or m <= 0 or m >= n:
        # m >= n means the statistic reads everything, which is already what
        # _label_arrangement_floor measures, and it does so more tightly.
        return None
    counts = [int(c) for c in np.unique(labels, return_counts=True)[1]]
    if len(counts) < 2 or sum(counts) != n:
        return None
    # NO ROW CAP HERE, deliberately, and it is not an oversight that the sibling
    # above has one. _label_arrangement_floor computes math.factorial(n) as an
    # exact big integer, which really does become pointless work and really can
    # overflow on conversion, so _MAX_ENUMERABLE_ROWS belongs there. Measured:
    # that path costs 0.1 ms at n = 1,000 but 20,504 ms at n = 1,000,000, and
    # float(arrangements) raises OverflowError from n = 2,000 upward.
    #
    # The cost this function pays is the np.unique call above, which is
    # O(n log n): 16.8 us at n = 1,000, 6.75 ms at n = 1,000,000, 84 ms at
    # n = 10,000,000. The lgamma arithmetic below is closed form in log space
    # and is O(number of groups), a flat 0.8 us at any n. The cap sat AFTER the
    # np.unique call, so 0.8 us is the entire marginal cost it was buying back,
    # against the 10,000 metric evaluations permutation_test runs anyway.
    #
    # An earlier version of this comment said the FUNCTION is O(number of
    # groups) and cost one microsecond at a million rows. That is true only of
    # the arithmetic, not of the call, and it was wrong by four orders of
    # magnitude. The conclusion survives the correction; the reason given for it
    # did not.
    #
    # The cap was copied across to here, and it silently switched the floor OFF
    # above 1000 rows, which is the ordinary size of a real audit. The Stage 2
    # audit measured the consequence: at n = 1002, 1200 and 2000 with four true
    # positives, comprehensive_fairness_test returned significant_at_05=False
    # against a floor 274x below anything reachable, and permutation_test called
    # directly at the same shape 544x below. So the refusal this floor exists to
    # produce worked only on inputs too small to matter, and every test of it
    # used a small fixture.

    # log P = log((n-m)!) - log(n!) + sum(log c_i!) - max sum(log d_i!)
    remaining = n - m
    log_p = math.lgamma(remaining + 1) - math.lgamma(n + 1)
    for c in counts:
        log_p += math.lgamma(c + 1)
    for c in sorted(counts, reverse=True):
        take = min(c, remaining)
        remaining -= take
        log_p -= math.lgamma(take + 1)
    if remaining != 0:
        return None
    return float(min(1.0, math.exp(log_p)))


def permutation_test(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
    *,
    n_design_rows: Optional[int] = None,
) -> PermutationTestResult:
    """
    Perform permutation test for fairness metric significance.

    This is the gold standard for formal hypothesis testing in fairness audits.
    The null hypothesis H₀ is that the model treats all groups equally
    (i.e., the fairness metric difference is zero).

    The test works by randomly shuffling group labels many times and
    computing the metric for each shuffled dataset to build a null distribution.
    The p-value is the proportion of permuted statistics that are at least
    as extreme as the observed statistic.

    Args:
        y_pred: Model predictions (binary or continuous)
        sensitive_attr: Protected attribute group membership
        metric_func: Function(y_pred, sensitive_attr) -> fairness metric value
        n_permutations: Number of permutation iterations (≥10000 recommended)
        alternative: 'two-sided', 'greater', or 'less'
        random_state: Random seed for reproducibility
        n_design_rows: How many rows ``metric_func`` actually reads, when that is
            fewer than ``len(sensitive_attr)``. A statistic computed on a subset
            (equal opportunity reads only the rows with ``y_true == 1``) has a
            much smaller effective design than the full label array, and leaving
            this unset advertises a p-floor far below anything that subset could
            reach. Leave it None when the statistic reads every row.

    Returns:
        PermutationTestResult with p-value and null distribution

    Example:
        >>> def demographic_parity_diff(y_pred, attr):
        ...     groups = np.unique(attr)
        ...     rates = [np.mean(y_pred[attr == g]) for g in groups]
        ...     return rates[0] - rates[1]
        >>> result = permutation_test(y_pred, gender, demographic_parity_diff)
        >>> print(f"p-value: {result.p_value:.4f}")

    Reference:
        DiCiccio et al. (2020). "Evaluating fairness using permutation tests."
        ACM SIGKDD Conference on Knowledge Discovery & Data Mining.

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

    Ledger row: permutation_test. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    # BGL-S2B (2026-09-17). Validated HERE, not at the point of use. The
    # `raise ValueError("Unknown alternative: ...")` that used to be the only
    # check sits after the design-power guard, so once that guard started
    # refusing, a caller who passed alternative='twosided' got a polite
    # could-not-check about statistical power instead of the configuration error
    # they had actually made. A misconfiguration must never be able to hide
    # behind a refusal.
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"Unknown alternative: {alternative}. Use 'two-sided', 'greater', or 'less'"
        )

    rng = np.random.default_rng(random_state)

    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    # Calculate observed test statistic
    observed_stat = metric_func(y_pred, sensitive_attr)

    # C-01. The null distribution is filtered for NaN below, but the OBSERVED
    # statistic was not, and an unmeasurable observation is the more dangerous of
    # the two. `np.abs(null) >= np.abs(nan)` is False for every draw, so the count
    # is 0 and p collapses to 1/(N+1): measured on a 500-draw null, p=0.001996,
    # significant_at_05=True. An unmeasurable comparison would have been reported
    # as a highly significant BREACH.
    #
    # Refused rather than scored. p_value is NaN and significance is None, which
    # is the third state: neither significant nor not-significant.
    if not np.isfinite(observed_stat):
        warnings.warn(
            "The observed statistic could not be computed (returned "
            f"{observed_stat!r}), so no permutation test was run. p_value is NaN "
            "and significance is None: this is COULD NOT CHECK, not a negative "
            "result.",
            UserWarning,
            stacklevel=2,
        )
        return PermutationTestResult(
            observed_statistic=float("nan"),
            p_value=float("nan"),
            null_distribution=np.array([]),
            n_permutations=0,
            significant_at_05=None,
            significant_at_01=None,
            effect_direction="none",
            method="permutation (not run: observed statistic unmeasurable)",
        )

    # THE NULL MUST BE DRAWN OVER THE POPULATION THE OBSERVED STATISTIC READS.
    #
    # BGL-W4 (2026-09-30). A row whose group label matches NO level (a float NaN,
    # a NaT, a None in an object array) is invisible to every `attr == g` mask, so
    # the observed statistic cannot see it. Shuffling the WHOLE label array moves
    # real labels ONTO those rows, and then the statistic does read them, so the
    # p-value and the significance verdict were computed over a different
    # population than the statistic they grade.
    #
    # Measured on permutation_test_equal_opportunity, n_permutations=2000,
    # random_state=1, sensitive_attr = [0.0]*53 + [1.0]*53 + [nan]*100 with six
    # readable positive rows giving a TPR gap of 2/3:
    #
    #   the 100 unlabelled rows also positive -> observed 0.6667, null mean
    #     0.1115, p_value 0.000500 (the 1/(B+1) floor), significant_at_05 TRUE
    #   the IDENTICAL readable design, those rows not positive -> the SAME
    #     observed 0.6667, null mean 0.5180, p_value 0.322059,
    #     significant_at_05 FALSE
    #
    # So a not-significant equal-opportunity reading became a significant one, at
    # the p-floor, purely because of rows the statistic never looked at. The
    # unit's own warning told the reader the opposite: "The statistic and the
    # p-value are real for the levels that DO match rows." The statistic was; the
    # p-value was not.
    #
    # The remedy is to shuffle only among the rows the statistic can read and hold
    # the unattributable ones where they are, which makes the null exchangeable
    # over exactly the population the observed statistic measured. It sits HERE,
    # above every wrapper, rather than in the equal-opportunity closure: the
    # demographic-parity twin has the same `attr == g` mask and is spared only
    # because its own `np.mean` of an empty slice happens to yield NaN, which is
    # an accident of that closure and not a property of this test.
    attributable = np.zeros(len(sensitive_attr), dtype=bool)
    for _level in np.unique(sensitive_attr):
        attributable |= sensitive_attr == _level
    n_unattributable = int(attributable.size - int(np.sum(attributable)))
    # The labels the design actually consists of. The floors below are computed
    # over these, not over the whole array: counting arrangements of rows that are
    # pinned in place OVERSTATES how many distinct null datasets exist, and an
    # overstated arrangement count makes the floor SMALLER, which is the direction
    # that reports power the design does not have.
    design_labels = sensitive_attr[attributable] if n_unattributable else sensitive_attr
    if n_unattributable:
        warnings.warn(
            f"permutation_test: {n_unattributable} of {len(sensitive_attr)} row(s) carry a group "
            f"label that matches no level, so the statistic cannot read them. They are held "
            f"FIXED across the permutations instead of being shuffled, because a null drawn "
            f"over rows the observed statistic never reads grades a different population than "
            f"the statistic it is compared with. The p-value below is for the "
            f"{int(np.sum(attributable))} attributable row(s) only.",
            UserWarning,
            stacklevel=2,
        )

    # Generate null distribution via permutation
    null_distribution = np.zeros(n_permutations)
    for i in range(n_permutations):
        if n_unattributable:
            permuted_attr = np.array(sensitive_attr, copy=True)
            permuted_attr[attributable] = rng.permutation(sensitive_attr[attributable])
        else:
            # Unchanged call for the ordinary case, so a seeded run stays
            # reproducible byte for byte.
            permuted_attr = rng.permutation(sensitive_attr)
        null_distribution[i] = metric_func(y_pred, permuted_attr)

    # Remove any NaN values
    null_distribution = null_distribution[~np.isnan(null_distribution)]

    if len(null_distribution) < n_permutations * 0.9:
        warnings.warn("More than 10% of permutations resulted in NaN")

    # READINESS-6, 2026-09-10. A COLLAPSED NULL IS COULD-NOT-CHECK, NOT A
    # NEGATIVE RESULT, and this is the same defect C-01 fixed one line earlier
    # for the OBSERVED statistic, arriving from the other side.
    #
    # The estimator below is (count + 1) / (B + 1), so its smallest reachable
    # p-value is 1 / (B + 1) where B is the number of permutations that were
    # actually MEASURABLE. Attrition to NaN shrinks B silently: measured this
    # day on this function, a null that collapsed to 3 usable draws out of 100
    # returned p_value=1.0, n_permutations=3, significant_at_05=False. Its
    # p-floor was 0.25, so no arrangement of the data could ever have made it
    # significant, yet it was a fully graded negative result. Worse, a null
    # emptied completely returned n_permutations=0, p_value=1.0,
    # significant_at_05=False: an 80-point demographic-parity gap certified
    # "not significant at p=1.000".
    #
    # That finite phantom then enters a correction family as a real hypothesis.
    # Measured in comprehensive_fairness_test the same day: demographic_parity
    # p=0.0495 (100 usable draws, a real finding) plus this 3-draw phantom gave
    # a Benjamini-Hochberg family of m=2, adjusted 0.0990, so
    # significant_metrics=[] and any_significant=False. The honest family of
    # m=1 rejects it. One unmeasurable comparison buried the one real finding.
    #
    # Refused the way C-01 refuses: p_value NaN and significance None, which the
    # `is None` filter in comprehensive_fairness_test already excludes from the
    # family. The floor and the note travel with the result either way.
    from ._statistics import detectability, min_attainable_p_permutation

    # BGL-S2 (2026-09-16). TWO floors bound a permutation test and the binding
    # one is the LARGER, exactly as ``min_attainable_p_sign_flip`` in
    # _statistics.py already does for its own test. Only the RESAMPLE floor was
    # computed here, so ``min_attainable_p_value`` was 1/(B+1) = the number of
    # DRAWS, and said nothing about how many distinct label arrangements exist.
    #
    # Measured at the exported wrapper permutation_test_demographic_parity with
    # y_pred=[1,1,0,0] and sensitive_attr=['F','F','M','M'] (COMPLETE
    # separation, the most extreme data this shape allows): the reported floor
    # was 9.999e-05 and detectable_at_05 was True. There are only C(4,2)=6
    # distinct arrangements of those labels, so the true floor is ~0.333; brute
    # force over all 16 possible y_pred vectors confirmed the smallest p ANY
    # data can produce at that shape is 0.3320667933206679. The reported floor
    # was ~3,300x below anything reachable, detectable_at_05=True was an
    # affirmative assurance that data COULD have made this test fire when no
    # data could, and significant_at_05/01=False were graded off that same wrong
    # floor and so read as measured negatives instead of None.
    enumeration_floor = _label_arrangement_floor(design_labels, alternative)
    effective_floor = _effective_design_floor(design_labels, n_design_rows)
    resample_floor = min_attainable_p_permutation(len(null_distribution))
    _named_floors = [
        (name, f)
        for name, f in (
            ("enumeration", enumeration_floor),
            ("effective_design", effective_floor),
            ("resample", resample_floor),
        )
        if f is not None
    ]
    min_p = max(f for _, f in _named_floors) if _named_floors else None
    binding = max(_named_floors, key=lambda nf: nf[1])[0] if _named_floors else "unknown"
    detectable_05, design_note = detectability(min_p, n_family=1, alpha=0.05)
    if detectable_05 is not True:
        # BGL-S2B (2026-09-17). THE REFUSAL NAMES ITS OWN CAUSE. There are three
        # different reasons this branch fires and they used to share one
        # sentence, the one about a collapsed null: a test refused because its
        # LABEL ENUMERATION has no power was told "the permutation null collapsed
        # to 10000 usable draw(s) of 10000 requested", which is not a collapse at
        # all, and was handed method="permutation (not run: null distribution
        # collapsed)". The reader was given the wrong cause and would have gone
        # looking for NaN metrics instead of a bigger sample.
        if binding == "enumeration":
            cause = (
                f"There are too few distinct arrangements of these group labels for this "
                f"test to reach significance: the smallest p-value the design could "
                f"return is {format(min_p, '.4g')}."
            )
            method = "permutation (not run: too few label arrangements)"
        elif binding == "effective_design":
            cause = (
                f"The statistic reads only {n_design_rows} of {len(sensitive_attr)} rows, and "
                f"over that subset the smallest p-value the design could return is "
                f"{format(min_p, '.4g')}."
            )
            method = "permutation (not run: effective design too small)"
        elif binding == "resample":
            cause = (
                f"The permutation null collapsed to {len(null_distribution)} usable draw(s) "
                f"of {n_permutations} requested, so the smallest p-value this test could "
                f"return is {format(min_p, '.4g')}."
            )
            method = "permutation (not run: null distribution collapsed)"
        else:
            cause = (
                "The smallest p-value this test's design could return could not be computed at all."
            )
            method = "permutation (not run: design floor unknown)"
        warnings.warn(
            f"{cause} p_value is NaN and significance is None: this is COULD NOT CHECK, "
            f"not a negative result. {design_note}",
            UserWarning,
            stacklevel=2,
        )
        return PermutationTestResult(
            observed_statistic=observed_stat,
            p_value=float("nan"),
            null_distribution=null_distribution,
            n_permutations=len(null_distribution),
            significant_at_05=None,
            significant_at_01=None,
            effect_direction="none",
            method=method,
            min_attainable_p_value=min_p,
            detectable_at_05=detectable_05,
            design_note=design_note,
        )

    # The 0.01 verdict has its OWN floor, and a design that can reach 0.05 need
    # not reach 0.01: at 50 usable draws the floor is 0.0196, so "not
    # significant at 0.01" would be an absence of resolution reported as a
    # measured negative. Three states there too.
    detectable_01, _ = detectability(min_p, n_family=1, alpha=0.01)

    # Calculate p-value based on alternative hypothesis
    if alternative == "two-sided":
        # Two-tailed: proportion of |null| >= |observed|
        p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_stat)) + 1) / (
            len(null_distribution) + 1
        )
    elif alternative == "greater":
        # One-tailed: proportion of null >= observed
        p_value = (np.sum(null_distribution >= observed_stat) + 1) / (len(null_distribution) + 1)
    else:
        # alternative == "less" (validated at the top of the function, above the
        # design guard, so an unknown value raises there rather than being
        # reported as a power problem).
        # One-tailed: proportion of null <= observed
        p_value = (np.sum(null_distribution <= observed_stat) + 1) / (len(null_distribution) + 1)

    # Determine effect direction
    if observed_stat > 0:
        effect_direction = "positive"
    elif observed_stat < 0:
        effect_direction = "negative"
    else:
        effect_direction = "none"

    return PermutationTestResult(
        observed_statistic=observed_stat,
        p_value=p_value,
        null_distribution=null_distribution,
        n_permutations=len(null_distribution),
        # bool(), not the raw numpy comparison. The field is declared
        # Optional[bool] and is THREE-STATE, so a caller distinguishes the states
        # by identity; `np.True_ is True` is False, so an `is True` check on a
        # np.bool_ silently fails while `is None` works, which is the worst
        # possible combination on a significance verdict.
        significant_at_05=bool(p_value < 0.05),
        significant_at_01=bool(p_value < 0.01) if detectable_01 is True else None,
        effect_direction=effect_direction,
        method="permutation",
        min_attainable_p_value=min_p,
        # Reported, not asserted. This return is only reached when detectable_05
        # is True, but hardcoding the literal meant the field could not disagree
        # with the design even when the design changed underneath it.
        detectable_at_05=detectable_05,
        design_note=design_note,
    )


def permutation_test_demographic_parity(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
) -> PermutationTestResult:
    """
    Convenience function for permutation test on demographic parity difference.

    Tests H₀: positive_rate(group_A) = positive_rate(group_B)

    Args:
        y_pred: Binary predictions (0/1)
        sensitive_attr: Protected attribute (binary or multi-class)
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        PermutationTestResult for demographic parity difference
    """

    def dp_diff(y_pred, attr):
        groups = np.unique(attr)
        if len(groups) < 2:
            # C-01 hardened permutation_test to refuse a non-finite observed
            # statistic, and fixed three sibling closures to return NaN. This one
            # was missed, so it laundered "no comparison possible" into a
            # measured 0.0 and the guard downstream never fired.
            return float("nan")
        rates = [np.mean(y_pred[attr == g]) for g in groups]
        return np.max(rates) - np.min(rates)

    return permutation_test(
        y_pred, sensitive_attr, dp_diff, n_permutations=n_permutations, random_state=random_state
    )


def permutation_test_equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
) -> PermutationTestResult:
    """
    Permutation test for equal opportunity (TPR parity).

    Tests H₀: TPR(group_A) = TPR(group_B)

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        sensitive_attr: Protected attribute
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        PermutationTestResult for equal opportunity difference. Its
        ``groups_omitted`` names the groups the comparison could not include:
        those with no row with ``y_true == 1``, and those whose LABEL is missing,
        which no row can be matched to at all.

    **A GROUP WITH NO POSITIVE LABEL LEFT THE COMPARISON SILENTLY.**
    BGL-5 (2026-09-27). ``eo_diff`` collects a TPR only for the groups that have
    a row with ``y_true == 1`` and refuses only when FEWER THAN TWO remain, so
    with two of three remaining it returned a real max-minus-min gap over the
    survivors and the result had no field that could name the third group.
    Measured before this change, 300 rows in three groups of 100 where group
    ``c`` holds no positive label at all::

        observed_statistic 0.600, p_value 0.001996, significant_at_05 True,
        design_note "", warnings NONE

    A significant equal-opportunity verdict was published for a three-group audit
    one of whose groups was entirely outside it. After: the same measured 0.600
    and the same p-value, plus ``groups_omitted ('c',)`` and a warning naming it.
    The verdict is NOT withdrawn by its caveat, exactly as in the sibling
    :func:`test_equalized_odds_chi_square`, which was fixed for this shape in the
    same file on 2026-09-27; this is the permutation twin it did not reach.

    **A GROUP WHOSE OWN LABEL IS MISSING LEFT THE COMPARISON THE SAME WAY, AND
    ``groups_omitted`` COULD NOT SEE IT EITHER.** BGL5 A-evaluation-3,
    2026-09-29. The disclosure above is derived as ``np.unique(sensitive_attr)``
    minus the levels holding a positive label, so it can only find a group
    missing a positive LABEL. A group missing its own group label is in BOTH
    sets, because ``np.unique`` lists NaN as a level and ``str(nan)`` is "nan" on
    both sides, while inside ``eo_diff`` its mask ``attr == nan`` is False at
    every row, so it contributes no TPR and is silently outside the comparison.
    Measured before this change, 300 rows in three blocks of 100 with the
    protected attribute ``[0.0]*100 + [1.0]*100 + [nan]*100`` and TPRs
    0.50 / 0.45 / 0.95, the extreme rate in the unlabelled block::

        groups_omitted (), observed_statistic 0.04999999999999999,
        p_value 0.4992..., significant_at_05 False, warnings NONE

    The real max-minus-min TPR gap over the attribute is 0.95 - 0.45 = 0.50. A
    0.50 gap was published as 0.05 with a GRADED NEGATIVE verdict and an empty
    completeness field, one third of the audit outside the comparison. After: the
    same measured 0.05 for the two levels it does cover, ``groups_omitted
    ('nan',)``, and a warning naming the level and the 100 rows.

    **AND THE SENTENCE THAT DISCLOSURE PUBLISHES BESIDE IT WAS FALSE.** BGL-W4,
    2026-09-30. The warning above ends "The statistic and the p-value are real
    for the levels that DO match rows". The statistic was; the p-value was not.
    ``eo_diff`` cannot see an unmatchable row, but :func:`permutation_test`
    shuffled the WHOLE label array, which moves real labels ONTO those rows,
    after which the statistic does read them. So the null was the null of a
    different design. Measured before this change, n_permutations=2000,
    random_state=1, ``sensitive_attr = [0.0]*53 + [1.0]*53 + [nan]*100`` with six
    readable positive rows giving a TPR gap of 2/3::

        those 100 rows also positive -> observed 0.6667, p 0.000500
                                        (the 1/(B+1) floor), significant TRUE
        the IDENTICAL readable design,
        those rows not positive       -> observed 0.6667, p 0.322059,
                                        significant FALSE

    A not-significant equal-opportunity reading became a significant one purely
    because of rows the statistic never looked at. Two changes, both in the same
    place as the thing they bound: :func:`permutation_test` now holds an
    unattributable row FIXED across the permutations and computes its design
    floors over the attributable labels only, and ``n_design_rows`` here counts
    positives among the attributable rows rather than all of them (the floor it
    feeds is a lower bound, so an inflated count only ever makes the design look
    MORE powerful). After: both designs above return the same p 0.226136 and the
    same significant_at_05 False, and a real third label in that block is
    untouched at p 0.000500 significant True.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    def eo_diff(pred, attr):
        groups = np.unique(attr)
        if len(groups) < 2:
            # C-01 hardened permutation_test to refuse a non-finite observed
            # statistic, and fixed three sibling closures to return NaN. This one
            # was missed, so it laundered "no comparison possible" into a
            # measured 0.0 and the guard downstream never fired.
            return float("nan")

        tprs = []
        for g in groups:
            mask = (attr == g) & (y_true == 1)
            if np.sum(mask) > 0:
                tprs.append(np.mean(pred[mask]))

        if len(tprs) < 2:
            # Fewer than two groups had any positive labels, so no TPR gap
            # exists to test. Same reason as the guard above.
            return float("nan")
        return np.max(tprs) - np.min(tprs)

    # Which levels no row can be matched to, computed BEFORE the test so the row
    # count handed to the design floor can exclude them. See the block further
    # down for what this disclosure is for.
    raw_groups = np.unique(sensitive_attr)
    matched_any = np.zeros(len(sensitive_attr), dtype=bool)
    unattributable_levels = []
    for g in raw_groups:
        hit = sensitive_attr == g
        matched_any |= hit
        if not bool(np.any(hit)):
            unattributable_levels.append(str(g))

    # BGL-S2B (2026-09-17). `eo_diff` reads ONLY the rows with y_true == 1, so
    # the design that bounds this test's p-floor is those rows, not the whole
    # array. Without this the floor was the resample floor (9.999e-05 at the
    # default 10000 draws) no matter how few positives there were, which is an
    # affirmative detectable_at_05=True for a design that may reach nothing near
    # it. See _effective_design_floor.
    #
    # BGL-W4 (2026-09-30). AND ONLY THE ROWS IT CAN ATTRIBUTE. `np.sum(y_true ==
    # 1)` counted the positives in an unmatchable block too, and the floor this
    # feeds is a LOWER bound, so an inflated m only ever makes the design look
    # MORE powerful. Measured on 206 rows whose statistic reads 6 positive rows:
    # n_design_rows described 106 rows, so the effective-design floor did not
    # bind at all and min_attainable_p_value was the resample floor 4.998e-04
    # with detectable_at_05 True and design_note empty.
    n_positive_rows = int(np.sum((y_true == 1) & matched_any))

    result = permutation_test(
        y_pred,
        sensitive_attr,
        eo_diff,
        n_permutations=n_permutations,
        random_state=random_state,
        n_design_rows=n_positive_rows,
    )

    # Which groups the statistic could not read at all. `eo_diff` needs a row
    # with y_true == 1 to have a TPR, so a group without one is present in the
    # audit and absent from the comparison. Named on the result AND in a warning,
    # because the assessment that found this ran under `python -W ignore`: the
    # FIELD has to carry it, not only the warning.
    all_groups = [str(g) for g in raw_groups]
    present = sorted({str(g) for g in np.unique(sensitive_attr[y_true == 1])})
    omitted = tuple(g for g in all_groups if g not in present)

    # A MISSING GROUP LABEL IS NOT A GROUP, AND EVERY COMPARISON AGAINST IT IS
    # FALSE. See the docstring for the measured 0.50 gap published as 0.05.
    # `np.unique` lists a missing label as a level, so it appears in all_groups
    # AND in present and the set difference above cannot name it; the question it
    # answers is "which level has no row the statistic reads", and for these the
    # answer is EVERY row. Asked directly, per level, so it holds for any
    # unmatchable value (NaN, NaT, None in an object array) and not only for the
    # float NaN that was measured. Computed above the test call, because the row
    # count the design floor is given has to exclude these rows too.
    if unattributable_levels:
        n_rows = len(sensitive_attr)
        n_outside = int(np.sum(~matched_any))
        # Named on the FIELD as well as in the warning, and appended rather than
        # substituted: a level can be both unlabelled and positive-label-free,
        # and a reader of groups_omitted is asking one question.
        omitted = omitted + tuple(g for g in unattributable_levels if g not in omitted)
        warnings.warn(
            f"permutation_test_equal_opportunity: group label(s) "
            f"{', '.join(unattributable_levels)} match NO row (every comparison "
            f"against a missing label is False), so {n_outside} of {n_rows} row(s) "
            f"belong to no group and are outside this comparison entirely: they "
            f"contribute no TPR, and a gap of any size among them is invisible here. "
            f"See groups_omitted on the result. The statistic and the p-value are "
            f"real for the levels that DO match rows.",
            UserWarning,
            stacklevel=2,
        )
    if omitted:
        compared = f"only {', '.join(present)}" if present else "nothing at all"
        warnings.warn(
            f"permutation_test_equal_opportunity: group(s) {', '.join(omitted)} have no "
            f"row with y_true == 1, so no TPR exists for them and this test compares "
            f"{compared} and says NOTHING about them. {len(omitted)} of "
            f"{len(all_groups)} group(s) are outside this comparison; see "
            f"groups_omitted on the result. The p-value and the significance verdict "
            f"are real for the groups it does cover.",
            UserWarning,
            stacklevel=2,
        )
    # dataclasses.replace, not a field-by-field rebuild: a rebuild lists the
    # fields that existed when it was written and silently drops every one added
    # later, which is how a three-state disclosure gets lost one layer above the
    # function that computed it.
    return dataclasses.replace(result, groups_omitted=omitted)


# Contingency Table Tests


def contingency_test(
    y_pred: np.ndarray, sensitive_attr: np.ndarray, min_expected: float = 5.0
) -> ContingencyTestResult:
    """
    Test independence between predictions and protected attribute.

    Automatically selects between Chi-square test (for larger samples)
    and Fisher's exact test (when expected cell counts are small).

    This tests H₀: predictions are independent of group membership.

    Args:
        y_pred: Binary predictions (0/1)
        sensitive_attr: Protected attribute
        min_expected: Minimum expected cell count threshold for chi-square
                     If any expected count < min_expected, uses Fisher's exact

    Returns:
        ContingencyTestResult with test statistics and p-value

    Example:
        >>> result = contingency_test(y_pred, gender)
        >>> print(f"Test: {result.test_used}, p={result.p_value:.4f}")

    **A TABLE WITH AN EMPTY MARGIN HAS ONE POSSIBLE VALUE.** BGL-3
    evaluation-1, 2026-09-27. The C-02 guard below tested the table's SHAPE, and
    a 2xk table can be full-shaped and still carry no information: if every
    prediction is the same value, one outcome row is all zeros, every group has
    the identical (constant) outcome, and the margins admit exactly ONE table.
    Measured on this repo before this change, 30 rows in two groups of 15 with
    ``y_pred`` all zeros::

        test_used='fisher_exact', statistic=nan, p_value=1.0,
        contingency_table=[[15, 15], [0, 0]], significant_at_05=False

    A completed test with a clean negative verdict, from data in which no
    arrangement of the group labels could have produced anything else. The same
    input with THREE groups did not answer at all, it raised scipy's "the
    internally computed table of expected frequencies has a zero element", so
    the function's honesty depended on the number of groups. Both now refuse.

    **AND A VALID TEST CAN STILL HAVE NO POWER.** The 2x2 group sizes bound the
    smallest p-value Fisher's exact can return, exactly as the resample count
    bounds ``permutation_test``. Measured the same day on three rows per group,
    predictions PERFECTLY separated by group, the most extreme data that shape
    allows: ``p_value=0.1, significant_at_05=False``. The floor for a
    three-versus-three design is 0.1, so no data whatsoever could have made that
    test significant, and the False was an absence of resolution wearing a
    measured negative's clothes. It is now None, with the floor and a note beside
    it; the p-value is a real p-value and is still returned.

    The floor is claimed only for the branch where it is exact (Fisher on a 2x2),
    and only as the floor over DESIGNS, where the outcome margin is free because
    the model's predictions could have come out differently. That distinction is
    load-bearing and cost a wrong accusation while this was written: 99 rows in
    one group and 1 in the other looks powerless, and conditioning on BOTH
    observed margins it is, but its design floor is 0.01, because the one row in
    the small group being the only selected row in the data is a reachable
    outcome and Fisher gives it p=0.01. That design is detectable and is NOT
    refused. The chi-square branch reports ``detectable_at_05=None``, which is
    could-not-check: a chi-square p can fall below the exact floor, so assuming
    it would call detectable designs dead, the failure mode ``_statistics``'
    ``min_attainable_p_mannwhitney`` documents.
    """
    # Imported here, as permutation_test imports its own floor helpers, to keep
    # the module-level import graph of this file unchanged.
    from ._statistics import detectability, min_attainable_p_fisher

    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    groups = np.unique(sensitive_attr)

    # Build contingency table: rows=outcome (0,1), cols=groups
    table = np.zeros((2, len(groups)), dtype=int)

    for i, g in enumerate(groups):
        mask = sensitive_attr == g
        table[1, i] = np.sum((y_pred == 1) & mask)  # Positive predictions
        table[0, i] = np.sum((y_pred == 0) & mask)  # Negative predictions

    # Calculate expected counts
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    n = table.sum()

    # C-02. Two ways there is nothing to test, and only the first was guarded.
    #
    #   n == 0            no observations at all
    #   a degenerate table  fewer than two rows or two columns, so there is
    #                       nothing for the predictions to be independent OF
    #
    # The second reached scipy and came back as a COMPLETED test. Measured
    # 2026-09-07 on a group whose every label was negative: a 2x1 table,
    # degrees_of_freedom=0, statistic=0.0, p_value=1.0, significant_at_05=False.
    # Zero degrees of freedom is the arithmetic saying no comparison was made,
    # and it was reported as a clean negative result.
    #
    # `significant_at_05` also moves off False here. A NaN p-value is neither
    # significant nor not-significant, and returning False for it was the same
    # two-state collapse in the branch that had already noticed the problem.
    # An EMPTY MARGIN is the third way, and it is the one the shape test misses:
    # see the docstring's zero-row case. A zero row total means the outcome never
    # varies, a zero column total means a group with no rows, and either way the
    # margins fix the table completely, so no data could have produced a
    # different p-value. Fisher answered 1.0 for it and chi-square raised.
    empty_rows = int(np.sum(row_totals == 0))
    empty_cols = int(np.sum(col_totals == 0))
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2 or empty_rows or empty_cols:
        if n == 0:
            reason = "no observations"
        elif table.shape[0] < 2 or table.shape[1] < 2:
            reason = f"a degenerate {table.shape[0]}x{table.shape[1]} table"
        else:
            reason = (
                f"a {table.shape[0]}x{table.shape[1]} table with {empty_rows} empty "
                f"outcome row(s) and {empty_cols} empty group column(s), which leaves "
                f"exactly one table the margins allow, so no data could have produced "
                f"a different result"
            )
        warnings.warn(
            f"Independence cannot be tested from {reason}: no test was run. "
            "p_value is NaN and significance is None (COULD NOT CHECK), not a "
            "negative result.",
            UserWarning,
            stacklevel=2,
        )
        return ContingencyTestResult(
            test_used="none",
            statistic=np.nan,
            p_value=np.nan,
            contingency_table=table,
            expected_counts=np.zeros_like(table, dtype=float),
            significant_at_05=None,
            degrees_of_freedom=0,
            design_note=f"COULD NOT CHECK: independence cannot be tested from {reason}.",
        )

    expected = np.outer(row_totals, col_totals) / n

    # Choose test based on expected counts
    if len(groups) == 2 and np.any(expected < min_expected):
        # Use Fisher's exact test for 2x2 tables with small expected counts
        odds_ratio, p_value = scipy_stats.fisher_exact(table)

        # The design floor, for the branch where it is exact. `min_attainable_p
        # _fisher` runs the real test on a perfect split of these two group
        # sizes, so it is the smallest p ANY predictions at these group sizes
        # could have produced; see the docstring's three-versus-three
        # measurement, and the 99-versus-1 case it does NOT refuse.
        min_p = min_attainable_p_fisher(int(col_totals[0]), int(col_totals[1]))
        detectable_05, design_note = detectability(min_p, n_family=1, alpha=0.05)
        if detectable_05 is not True:
            warnings.warn(
                f"contingency_test: Fisher's exact on group sizes "
                f"{int(col_totals[0])} and {int(col_totals[1])} has a p-value floor of "
                f"{min_p if min_p is None else format(min_p, '.4g')}, so "
                f"significant_at_05 is None (could not check), not False. The "
                f"p-value itself ({format(float(p_value), '.4g')}) is a real "
                f"measurement. {design_note}",
                UserWarning,
                stacklevel=2,
            )
        return ContingencyTestResult(
            test_used="fisher_exact",
            statistic=odds_ratio,
            p_value=p_value,
            contingency_table=table,
            expected_counts=expected,
            # bool(), not the raw numpy comparison, and None when the design
            # could never have said yes. The field is declared Optional[bool] and
            # is read by identity: `np.True_ is True` is False, so an `is True`
            # check on a np.bool_ silently misses a real finding while `is None`
            # still works, which is the worst possible pair on a significance
            # verdict. PermutationTestResult was fixed for this. Measured
            # 2026-09-27 on the chi-square branch below, which had the identical
            # construction, on a healthy 50/50 2x2 with p = 6.6e-09:
            # `significant_at_05 is True` was False while the value printed True.
            significant_at_05=bool(p_value < 0.05) if detectable_05 is True else None,
            odds_ratio=odds_ratio,
            min_attainable_p_value=min_p,
            detectable_at_05=detectable_05,
            design_note=design_note,
        )
    else:
        # Use chi-square test
        chi2, p_value, dof, expected_from_scipy = scipy_stats.chi2_contingency(table)

        # Compute Cramer's V effect size.
        #
        # `else 0.0` was the same two-state collapse the C-02 guard above
        # removed from the p-value, one statistic further down: V is undefined
        # unless the table has at least two rows AND two columns and at least
        # one observation, and 0.0 is NO ASSOCIATION, the reading a table
        # nobody could measure must not be given. The C-02 guard means this
        # branch is not reachable from `contingency_test` as the function
        # stands (verified by execution 2026-09-17: every degenerate table
        # returns test_used="none" before scipy is called), so this is the
        # refusal the expression should always have carried rather than a
        # behaviour change. It stays because the guard and the statistic are
        # ten lines and one refactor apart, and because NaN reaches a reader as
        # could-not-check while 0.0 reaches them as clean.
        n = table.sum()
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v: float
        if min_dim > 0 and n > 0 and np.isfinite(chi2):
            cramers_v = float(np.sqrt(chi2 / (n * min_dim)))
        else:
            warnings.warn(
                f"contingency_test: Cramer's V is not defined for a "
                f"{table.shape[0]}x{table.shape[1]} table of {int(n)} "
                f"observation(s) with chi-square {chi2}, so the effect size was "
                f"NOT measured. cramers_v is NaN (could not check), not 0.0, "
                f"which would read as no association between the predictions "
                f"and the group.",
                UserWarning,
                stacklevel=2,
            )
            cramers_v = float("nan")

        # Compute odds ratio for 2x2 tables
        odds_ratio = None
        if table.shape == (2, 2):
            a, b = table[1, 0], table[1, 1]
            c, d = table[0, 0], table[0, 1]
            if b > 0 and c > 0:
                odds_ratio = (a * d) / (b * c) if b * c > 0 else np.inf

        return ContingencyTestResult(
            test_used="chi_square",
            statistic=chi2,
            p_value=p_value,
            contingency_table=table,
            expected_counts=expected_from_scipy,
            # bool() for the same identity reason as the Fisher branch above.
            significant_at_05=bool(p_value < 0.05),
            odds_ratio=odds_ratio,
            cramers_v=cramers_v,
            degrees_of_freedom=dof,
            # NOT the Fisher floor. A chi-square p can fall BELOW the exact
            # floor, so borrowing it here would call detectable designs dead.
            # None is could-not-check and says exactly that.
            min_attainable_p_value=None,
            detectable_at_05=None,
            design_note=(
                "COULD NOT CHECK: no p-value floor is computed for the "
                "chi-square branch, so whether this design could ever have "
                f"reached significance is unknown ({table.shape[1]} groups, "
                f"{int(n)} observations)."
            ),
        )


def test_equalized_odds_chi_square(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> Dict[str, ContingencyTestResult]:
    """
    Chi-square tests for equalized odds (TPR and FPR parity).

    Performs separate contingency tests for:
    1. TPR parity: Among actual positives, is prediction independent of group?
    2. FPR parity: Among actual negatives, is prediction independent of group?

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        sensitive_attr: Protected attribute

    Returns:
        Dict with 'tpr_test' and 'fpr_test' ContingencyTestResults. Each one
        carries ``groups_omitted``: the groups that had no row in that stratum
        and so are absent from that p-value.

    **A GROUP WITH NO POSITIVE LABELS LEAVES THE TPR COMPARISON SILENTLY.**
    BGL-3 evaluation-1, 2026-09-27. Each test is built on a STRATUM of the data
    (``y_true == 1`` for TPR, ``y_true == 0`` for FPR), and a group can be absent
    from a stratum while being present in the audit. Measured on this repo before
    this change, 300 rows in three groups where group ``c`` held no positive
    label at all::

        tpr_test: chi_square, p=0.0033, significant_at_05=True, table 2x2
        fpr_test: chi_square, p=3.3e-15, significant_at_05=True, table 2x3

    The TPR verdict was a real measurement over groups ``a`` and ``b``, reported
    with nothing anywhere saying that the third group had been left out of it.
    The table shape was the only trace, and only for a reader who knew how many
    groups to expect. The omissions are now named in ``groups_omitted`` on the
    affected result and in a warning.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    all_groups = [str(g) for g in np.unique(sensitive_attr)]

    def _for_stratum(mask: np.ndarray, test_name: str, stratum: str) -> ContingencyTestResult:
        result = contingency_test(y_pred[mask], sensitive_attr[mask])
        present = sorted({str(g) for g in np.unique(sensitive_attr[mask])})
        omitted = tuple(g for g in all_groups if g not in present)
        if omitted:
            compared = f"only {', '.join(present)}" if present else "nothing at all"
            warnings.warn(
                f"test_equalized_odds_chi_square: group(s) {', '.join(omitted)} have no "
                f"row with {stratum}, so {test_name} compares {compared} and says NOTHING "
                f"about them. {len(omitted)} of {len(all_groups)} group(s) are outside "
                f"this comparison; see groups_omitted on the result.",
                UserWarning,
                stacklevel=3,
            )
        # dataclasses.replace, not a field-by-field rebuild: a rebuild lists the
        # fields that existed when it was written and silently drops every one
        # added later, which is how a three-state disclosure gets lost one layer
        # above the function that computed it.
        return dataclasses.replace(result, groups_omitted=omitted)

    # TPR test: among actual positives
    tpr_result = _for_stratum(y_true == 1, "tpr_test", "y_true == 1")

    # FPR test: among actual negatives
    fpr_result = _for_stratum(y_true == 0, "fpr_test", "y_true == 0")

    return {"tpr_test": tpr_result, "fpr_test": fpr_result}


# Robust Statistics


def compute_robust_metrics(
    values: np.ndarray,
    groups: np.ndarray,
    trim_proportion: float = 0.1,
    winsorize_limits: Tuple[float, float] = (0.05, 0.05),
    divergence_threshold: float = 0.2,
) -> Dict[str, RobustMetricsResult]:
    """
    Compute robust fairness metrics using trimming and Winsorization.

    Outliers disproportionately affect fairness metrics when demographic
    subgroups have small sample sizes. Robust statistics reduce sensitivity
    to outliers while preserving the bulk of available information.

    Args:
        values: Metric values (e.g., errors, predictions) to analyze
        groups: Group membership array
        trim_proportion: Proportion to trim from each end (default: 10%)
        winsorize_limits: (lower, upper) percentiles for Winsorization
        divergence_threshold: Threshold for flagging outlier influence
                            (flags if |standard - robust| > threshold * |standard|)

    Returns:
        Dict mapping group name to RobustMetricsResult. Empty when no group held
        a single value, which is nothing examined rather than nothing found, and
        is warned about.

    Example:
        >>> errors = np.abs(y_true - y_pred)
        >>> results = compute_robust_metrics(errors, gender)
        >>> for group, r in results.items():
        ...     if r.outlier_influence_detected:
        ...         print(f"Warning: Outliers affect {group} metrics")

    **A COMPARISON THAT CANNOT DISAGREE DETECTS NOTHING.** BGL-3 evaluation-1,
    2026-09-27. ``outlier_influence_detected`` is the divergence between the
    standard mean and the trimmed mean, and two shapes make that divergence
    non-existent while it was reported as a measured 0.0, i.e. as "the robust
    estimate agrees, no outlier influence". Measured on this repo before this
    change, all with the documented defaults:

    * ``scipy.stats.trim_mean`` cuts ``int(trim_proportion * n)`` points from
      each end, which is ZERO for every ``n <= 9`` at ``trim_proportion=0.1``,
      so the trimmed mean IS the mean by construction. A group of 9 values,
      eight of them 1.0 and one of them 1000.0, returned
      ``standard_value=112.0, trimmed_value=112.0, divergence_ratio=0.0,
      outlier_influence_detected=False``. The same three values at n=3 returned
      334.0/334.0/0.0/False. A 1000x outlier in a small subgroup is exactly what
      this function exists to find, and it could not fire at the sizes where
      subgroup outliers matter most.
    * The ratio divides by ``abs(standard_value)``, and the old guard answered
      0.0 when that denominator was unusable. 19 values of 1.0 and one of -19.0
      have mean 0.0 and trimmed mean 1.0, total disagreement between the two
      estimators, and returned ``divergence_ratio=0.0,
      outlier_influence_detected=False``. Twenty NaN values returned the same
      pair, because ``abs(nan) > 1e-10`` is False.

    Both now report ``divergence_ratio=NaN`` and
    ``outlier_influence_detected=None`` (could not check) with a warning naming
    the cause and the group size. The standard, trimmed, winsorized and median
    values are still returned: those are real numbers, and with nothing trimmed
    the trimmed mean genuinely equals the mean. It is the VERDICT that was not
    measured.

    Reference:
        Module 4 Unit 4: "Robust Statistics for Small Subgroup Analysis"
    """
    values = np.asarray(values)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    results = {}
    # Groups whose robustness verdict could not be reached, reported together at
    # the end so one call warns once rather than once per group.
    unassessable: List[str] = []
    # Levels np.unique reports that no row can be matched to (a missing label).
    unattributable_levels: List[str] = []

    for group in unique_groups:
        mask = groups == group
        group_values = values[mask]
        n = len(group_values)

        if n == 0:
            # A LEVEL THAT MATCHES NO ROW, AND `continue` MADE IT VANISH.
            # BGL5 A-evaluation-3, 2026-09-29. `np.unique` lists a missing label
            # (NaN, NaT, None) as a level, and `groups == group` is False at every
            # row for it, because every comparison against a missing value is
            # False. That is the ONLY way n can be 0 here, since every other level
            # came out of this same array. The `continue` dropped the level before
            # any caller could see it, so `robust_fairness_comparison`'s own
            # completeness field (`groups_not_measured`, added for exactly this
            # question) held the empty list while a third of the audit was outside
            # the comparison. Measured before this change, 90 rows in three blocks
            # of 30 with the attribute [0.0]*30 + [1.0]*30 + [nan]*30 and constant
            # absolute errors 1.0 / 1.2 / 9.0, the extreme one unlabelled:
            #   group_metrics keys ['0.0', '1.0'], standard_disparity 0.2000...,
            #   conclusion_changed False, groups_not_measured [], warnings NONE,
            #   against a real 9.0 - 1.0 = 8.0 gap across the attribute.
            # The level is now RETURNED with NaN estimates and a None verdict, so
            # every caller's own NaN filter sees it. It is not a group free of
            # outlier influence; it is a group nobody could put a row in.
            results[str(group)] = RobustMetricsResult(
                group=str(group),
                n_samples=0,
                standard_value=float("nan"),
                trimmed_value=float("nan"),
                winsorized_value=float("nan"),
                median_value=float("nan"),
                outlier_influence_detected=None,
                divergence_ratio=float("nan"),
                trim_effective=False,
            )
            unattributable_levels.append(str(group))
            unassessable.append(
                f"{group} (the label matches no row at all, so it holds no value to "
                f"compute a standard or a trimmed estimate from)"
            )
            continue

        # Standard metric
        standard_value = float(np.mean(group_values))

        # How many observations the trim actually removes from each end. This is
        # scipy's own rule (`lowercut = int(proportiontocut * n)`), with a
        # negative or NaN request floored at zero rather than allowed to look
        # effective. At zero the trimmed mean equals the mean for ANY data, which
        # is what made the divergence test structurally incapable of firing on a
        # small group.
        n_trimmed_each_end = int(max(0.0, float(trim_proportion)) * n)
        if n < 4:
            # The n >= 4 shortcut below does not call trim_mean at all, it copies
            # the mean, so nothing is trimmed however large the requested
            # proportion was. Without this line trim_proportion=0.4 at n=3 would
            # count 1 trimmed observation, claim the comparison was capable of
            # firing, and then read the identical estimators as a measured
            # "no outlier influence".
            n_trimmed_each_end = 0
        trim_effective = bool(n_trimmed_each_end)

        # Trimmed mean
        if n >= 4:  # Need at least 4 samples for meaningful trimming
            trimmed_value = float(scipy_stats.trim_mean(group_values, trim_proportion))
        else:
            trimmed_value = standard_value

        # Winsorized mean
        if n >= 4:
            winsorized = scipy_stats.mstats.winsorize(group_values, limits=winsorize_limits)
            winsorized_value = float(np.mean(winsorized))
        else:
            winsorized_value = standard_value

        # Median
        median_value = float(np.median(group_values))

        # Calculate divergence ratio. THREE STATES: measured, measured-and-clear,
        # and could-not-check. The last one is NaN plus a None verdict, never a
        # 0.0 that reads as agreement between the two estimators.
        divergence_ratio: float
        outlier_influence: Optional[bool]
        denominator = abs(standard_value)
        divergence = abs(standard_value - trimmed_value)
        if not trim_effective:
            divergence_ratio = float("nan")
            outlier_influence = None
            unassessable.append(
                f"{group} (n={n}, trim_proportion={trim_proportion}: the trim removes 0 "
                f"observations from each end, so the trimmed mean cannot differ from the "
                f"mean whatever the data)"
            )
        elif not np.isfinite(denominator) or denominator <= 1e-10 or not np.isfinite(divergence):
            divergence_ratio = float("nan")
            outlier_influence = None
            unassessable.append(
                f"{group} (n={n}, standard_value={standard_value!r} and "
                f"trimmed_value={trimmed_value!r} give no usable relative divergence)"
            )
        else:
            divergence_ratio = divergence / denominator
            outlier_influence = bool(divergence_ratio > divergence_threshold)

        results[str(group)] = RobustMetricsResult(
            group=str(group),
            n_samples=n,
            standard_value=standard_value,
            trimmed_value=trimmed_value,
            winsorized_value=winsorized_value,
            median_value=median_value,
            outlier_influence_detected=outlier_influence,
            divergence_ratio=divergence_ratio,
            trim_effective=trim_effective,
        )

    if unattributable_levels:
        # Its own warning, ahead of the verdict-level one below: a row that
        # belongs to no group is not a group whose robustness question was hard to
        # answer, it is data outside every comparison built on this result.
        matched_any = np.zeros(len(groups), dtype=bool)
        for group in unique_groups:
            matched_any |= groups == group
        n_outside = int(np.sum(~matched_any))
        warnings.warn(
            f"compute_robust_metrics: group label(s) "
            f"{', '.join(unattributable_levels)} match NO row (every comparison "
            f"against a missing label is False), so {n_outside} of {len(groups)} "
            f"row(s) belong to no group. Those level(s) are returned with NaN "
            f"estimates and outlier_influence_detected None (could not check), NOT "
            f"omitted from the result, because a caller cannot name a group it was "
            f"never told about.",
            UserWarning,
            stacklevel=2,
        )

    if not results:
        warnings.warn(
            "compute_robust_metrics: not one group held a value, so no robust "
            "statistic was computed. The empty result means NOTHING WAS EXAMINED, "
            "not that the groups were examined and found free of outlier "
            "influence.",
            UserWarning,
            stacklevel=2,
        )
    elif unassessable:
        warnings.warn(
            f"compute_robust_metrics: the outlier-influence question could not be "
            f"answered for {len(unassessable)} of {len(results)} group(s): "
            f"{'; '.join(unassessable)}. For those groups "
            f"outlier_influence_detected is None (could not check), not False, and "
            f"divergence_ratio is NaN, not 0.0, which would read as the robust "
            f"estimate agreeing with the standard one.",
            UserWarning,
            stacklevel=2,
        )

    return results


def robust_fairness_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metric: str = "mae",
    trim_proportion: float = 0.1,
) -> Dict[str, Any]:
    """
    Compare standard vs robust fairness metrics across groups.

    Computes both standard and robust versions of fairness disparity,
    flagging cases where outliers significantly affect conclusions.

    Args:
        y_true: True values
        y_pred: Predicted values
        sensitive_attr: Protected attribute
        metric: 'mae', 'mse', or 'error' for raw errors
        trim_proportion: Trimming proportion for robust estimate

    Returns:
        Dict with standard_disparity, robust_disparity, and analysis. The two
        disparities are NaN and the two verdicts are None when fewer than two
        groups could be measured, because a disparity needs two groups. A group
        whose own estimate is NaN does not count as measured: it is named in
        ``groups_not_measured`` and excluded from ``n_groups_compared``, never
        left to be skipped by a comparison against NaN.

    **0.0 IS PERFECT PARITY AND IT WAS THE ANSWER FOR NO COMPARISON AT ALL.**
    BGL-3 evaluation-1, 2026-09-27. Measured on this repo before this change, on
    30 rows of a SINGLE group and again on three empty arrays, both returned::

        {'standard_disparity': 0.0, 'robust_disparity': 0.0,
         'outlier_influence_detected': False}

    with no warning, byte-identical to the result of a real two-group comparison
    that found the groups' error distributions identical. The same shape that
    ``subgroup_robustness_audit`` above was fixed for in Beta Go-Live stage 1,
    arriving one function later. The refusal also now carries
    ``disparity_change_ratio`` and ``conclusion_changed``, which the old
    early-return omitted entirely, so a caller reading them got a KeyError on the
    refusal path and a number on the measured one.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    # Compute errors
    if metric == "mae":
        errors = np.abs(y_true - y_pred)
    elif metric == "mse":
        errors = (y_true - y_pred) ** 2
    else:
        errors = y_true - y_pred

    # Get robust metrics per group
    robust_results = compute_robust_metrics(errors, sensitive_attr, trim_proportion)

    # Compute disparities
    groups = list(robust_results.keys())

    # Aggregate the per-group outlier verdicts in THREE states. True beats
    # everything (a real finding is never withdrawn), then a measured False, and
    # None only when no group produced a verdict at all. `any()` collapsed None
    # onto False, which is why a single unassessable group used to read as clean.
    verdicts = [r.outlier_influence_detected for r in robust_results.values()]
    if any(v is True for v in verdicts):
        any_outlier_influence: Optional[bool] = True
    elif any(v is False for v in verdicts):
        any_outlier_influence = False
    else:
        any_outlier_influence = None
    unassessed = [r.group for r in robust_results.values() if r.outlier_influence_detected is None]

    # THE DISPARITY IS A MAX MINUS A MIN OVER VALUES THAT CAN BE NaN.
    # BGL-5 (2026-09-27). `max([3.0, nan])` is 3.0 and `min([3.0, nan])` is 3.0,
    # because every comparison against NaN is False and the builtin keeps
    # whichever element it happened to be holding: the unmeasurable group is
    # skipped and the gap COLLAPSES, order-dependently. Measured before this
    # change on 100 rows where group 'a' has a measured error of 3.0 and group
    # 'z' has no prediction at all (all-NaN errors):
    #   groups sorted a, z -> {'standard_disparity': 0.0, 'robust_disparity': 0.0,
    #                          'n_groups_compared': 2}
    #   the same data with the unmeasurable group renamed so it sorts FIRST
    #                       -> {'standard_disparity': nan, 'robust_disparity': nan}
    # so the published disparity depended on the spelling of the labels and one
    # of the two answers was 0.0, which this function's own docstring calls
    # PERFECT PARITY. After: both orderings refuse with NaN disparities and a
    # warning naming 'z', because one measured group is not a pair. The repo bans
    # this idiom by name in discovery._cramers_v ("np.maximum, NOT the builtin
    # max: max(0.0, nan) is 0.0").
    measurable = [
        g
        for g in groups
        if np.isfinite(robust_results[g].standard_value)
        and np.isfinite(robust_results[g].trimmed_value)
    ]
    not_measurable = [g for g in groups if g not in measurable]

    if len(measurable) < 2:
        if not_measurable:
            warnings.warn(
                f"robust_fairness_comparison: group(s) {', '.join(not_measurable)} have no "
                f"measurable error at all (their standard or trimmed estimate is NaN), so "
                f"they cannot enter a max-minus-min disparity. They are EXCLUDED from the "
                f"comparison rather than skipped by a comparison against NaN, which is what "
                f"collapsed the gap to 0.0.",
                UserWarning,
                stacklevel=2,
            )
        warnings.warn(
            f"robust_fairness_comparison: {len(measurable)} group(s) could be measured, "
            f"so there is no pair to compare and no disparity exists. "
            f"standard_disparity and robust_disparity are NaN (could not check), "
            f"not 0.0, which is PERFECT PARITY and was what this returned for a "
            f"comparison that never happened. conclusion_changed is None.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "standard_disparity": float("nan"),
            "robust_disparity": float("nan"),
            "disparity_change_ratio": float("nan"),
            "outlier_influence_detected": any_outlier_influence,
            "conclusion_changed": None,
            # The groups that entered the comparison, which is what a reader
            # needs beside a NaN: `len(groups)` counted a group whose own
            # estimate is NaN as compared.
            "n_groups_compared": len(measurable),
            "groups_not_measured": not_measurable,
            "group_metrics": robust_results,
        }

    if not_measurable:
        # Three or more groups, of which at least one has no measurable error and
        # at least two do. The disparity below is real for the groups it covers
        # and is a LOWER bound on the whole attribute's gap, so it is kept and
        # the omission is named rather than either being dropped.
        warnings.warn(
            f"robust_fairness_comparison: group(s) {', '.join(not_measurable)} have no "
            f"measurable error (their standard or trimmed estimate is NaN) and are "
            f"OUTSIDE both disparities below, which are computed over "
            f"{', '.join(measurable)} only. The numbers are real for those groups and "
            f"are a lower bound on the gap across the attribute, not the whole gap.",
            UserWarning,
            stacklevel=2,
        )

    standard_values = [robust_results[g].standard_value for g in measurable]
    trimmed_values = [robust_results[g].trimmed_value for g in measurable]

    standard_disparity = max(standard_values) - min(standard_values)
    robust_disparity = max(trimmed_values) - min(trimmed_values)

    # Analyze the difference. `conclusion_changed` is the claim "the robust
    # analysis confirms (or overturns) the standard one", and it is only a
    # measurement when the robust estimate could have differed. Where a group's
    # trim removed nothing its trimmed mean IS its mean, so the change ratio is
    # pinned at 0.0 by construction and a False here would be that same
    # structural zero wearing the robustness answer's clothes.
    disparity_change_ratio: float
    conclusion_changed: Optional[bool]
    if unassessed:
        disparity_change_ratio = float("nan")
        conclusion_changed = None
        warnings.warn(
            f"robust_fairness_comparison: the robust estimate could not disagree "
            f"with the standard one for group(s) {', '.join(unassessed)}, so "
            f"whether the conclusion changes was NOT measured. "
            f"disparity_change_ratio is NaN and conclusion_changed is None (could "
            f"not check), not False. standard_disparity and robust_disparity are "
            f"still real numbers; see the compute_robust_metrics warning for the "
            f"per-group cause.",
            UserWarning,
            stacklevel=2,
        )
    elif abs(standard_disparity) > 1e-10:
        disparity_change_ratio = (standard_disparity - robust_disparity) / abs(standard_disparity)
        conclusion_changed = bool(abs(disparity_change_ratio) > 0.25)
    else:
        # No usable denominator for a RELATIVE change. 0.0 here read as "the
        # robust disparity matches the standard one" on data where the standard
        # disparity was itself zero and the robust one need not be.
        disparity_change_ratio = float("nan")
        conclusion_changed = None
        warnings.warn(
            f"robust_fairness_comparison: standard_disparity is "
            f"{standard_disparity!r}, so the RELATIVE change against it does not "
            f"exist. disparity_change_ratio is NaN and conclusion_changed is None "
            f"(could not check), not False. The robust disparity itself is "
            f"{robust_disparity!r}.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "standard_disparity": standard_disparity,
        "robust_disparity": robust_disparity,
        "disparity_change_ratio": disparity_change_ratio,
        "outlier_influence_detected": any_outlier_influence,
        "conclusion_changed": conclusion_changed,
        # The groups the two disparities are actually a max and a min over, so a
        # reader is never told 2 for a pair one half of which is NaN.
        "n_groups_compared": len(measurable),
        "groups_not_measured": not_measurable,
        "group_metrics": robust_results,
    }


# Sensitivity Analysis / Robustness Testing


def _flip_is_a_no_op(y_pred: np.ndarray) -> bool:
    """True when ``1 - y_pred`` returns every value unchanged.

    ``label_noise`` perturbs by complementing the selected predictions. For the
    binary 0/1 arrays this function is normally given that always changes them,
    but an array whose every value equals its own complement (0.5 throughout, and
    only that) is complemented to itself, so the perturbation is a guaranteed
    no-op at any budget. Answers False for anything ``1 - y_pred`` cannot even be
    computed on: that is not a no-op, it is an error, and it belongs to the
    perturbation loop to raise.
    """
    try:
        return bool(np.array_equal(y_pred, 1 - y_pred))
    except TypeError as exc:
        # SAY SO. Returning False silently here is the shape the metric core
        # forbids: a handler that discards the error and substitutes an answer,
        # where the answer is then indistinguishable from a real one. False is
        # still correct (an array `1 - y_pred` cannot be computed on is not a
        # no-op), but the caller has to be able to tell that this is a dtype
        # problem rather than a measured result.
        warnings.warn(
            f"_flip_is_a_no_op: 1 - y_pred could not be computed on dtype "
            f"{getattr(np.asarray(y_pred), 'dtype', 'unknown')} ({exc}), so the "
            f"perturbation is NOT reported as a no-op. This is an input problem, "
            f"not a measurement.",
            UserWarning,
            stacklevel=2,
        )
        return False


def sensitivity_analysis(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    perturbation_type: str = "label_noise",
    perturbation_rate: float = 0.05,
    n_iterations: int = 100,
    robustness_threshold: float = 0.1,
    random_state: Optional[int] = None,
) -> SensitivityResult:
    """
    Analyze sensitivity of fairness metric to data perturbations.

    Stress-tests fairness conclusions by introducing controlled noise
    to verify they hold under real-world conditions.

    Args:
        y_pred: Model predictions
        sensitive_attr: Protected attribute
        metric_func: Function(y_pred, sensitive_attr) -> metric value
        perturbation_type: Type of perturbation:
            - 'label_noise': Flip random prediction labels
            - 'group_noise': Randomly reassign some group labels
            - 'subsample': Random subsampling (90% of data)
        perturbation_rate: Rate of perturbation (e.g., 0.05 = 5% noise)
        n_iterations: Number of perturbation iterations
        robustness_threshold: Maximum acceptable deviation for robustness
        random_state: Random seed

    Returns:
        SensitivityResult with robustness analysis

    Example:
        >>> result = sensitivity_analysis(y_pred, gender, dp_diff,
        ...                               perturbation_type='label_noise')
        >>> if not result.is_robust:
        ...     print(f"Warning: Metric is sensitive to noise")

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

    Ledger row: sensitivity_analysis. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    rng = np.random.default_rng(random_state)

    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)
    n = len(y_pred)

    # Original metric
    original_metric = metric_func(y_pred, sensitive_attr)

    # BGL-S2 (2026-09-16). THE PERTURBATION BUDGET IS COMPUTED ONCE, HERE, AND
    # A ZERO BUDGET IS A REFUSAL.
    #
    # `int(n * perturbation_rate)` TRUNCATES, so whenever n * rate < 1 not one
    # observation was flipped in ANY of the n_iterations draws. The "perturbed"
    # metric was then the original metric, n_iterations times over: std 0.0,
    # max_deviation 0.0, and the function reported is_robust=True,
    # robustness_score=1.0, n_iterations_run=100, n_unmeasurable=0 -- the
    # strongest robustness verdict it can produce, from 100 perturbations that
    # never happened, with no warning.
    #
    # This is NOT only a small-n defect. Any (n, rate) pair with n * rate < 1
    # reaches it, and the library's own stress_test_fairness hits it at n=90 with
    # its default 1% budget. The two guards further down test the METRIC
    # (np.isfinite(original_metric), n_run == 0) and a no-op perturbation yields a
    # perfectly finite metric, so BOTH are bypassed by construction: the budget
    # has to be checked on its own terms, before the draws.
    #
    # BGL-S2B (2026-09-17). A BUDGET IS NOT THE ONLY WAY TO PERTURB NOTHING:
    # THE PERTURBATION ALSO NEEDS SOMETHING TO AIM AT.
    #
    # 'group_noise' reassigns the selected rows with
    # `rng.choice(np.unique(sensitive_attr), ...)`. When the attribute holds ONE
    # distinct value that draw can only return that same value, so every one of
    # the n_iterations draws is bit-for-bit the original array no matter how
    # generous the budget is. Measured at the public entry (200 rows, one group,
    # perturbation_rate=0.10, 100 iterations): is_robust=True,
    # robustness_score=1.0, max_deviation=0.0, n_iterations_run=100,
    # n_unmeasurable=0, and NO warning; stress_test_fairness on the same input
    # reported overall_robust=True with 9 of 9 tests run. The budget guard below
    # cannot see this (int(200 * 0.10) = 20 is a healthy budget) and neither can
    # the finite-metric guards further down, because a no-op perturbation returns
    # a perfectly finite metric.
    degenerate_reason: Optional[str] = None
    if perturbation_type in ("label_noise", "group_noise"):
        n_flip_budget = int(n * perturbation_rate)
        if n_flip_budget < 1:
            degenerate_reason = (
                f"sensitivity_analysis: perturbation_rate={perturbation_rate} over n={n} rows "
                f"flips int({n} * {perturbation_rate}) = 0 observations, so "
                f"'{perturbation_type}' perturbed NOTHING and robustness was never tested. "
                f"Reporting is_robust=None (could not check), not the True/1.0 that "
                f"{n_iterations} identical no-op draws would produce. Raise perturbation_rate "
                f"or the sample size until n * rate >= 1."
            )
        elif perturbation_type == "group_noise" and int(np.unique(sensitive_attr).size) < 2:
            n_distinct = int(np.unique(sensitive_attr).size)
            degenerate_reason = (
                f"sensitivity_analysis: 'group_noise' reassigns group labels by drawing from "
                f"the {n_distinct} distinct value(s) present in sensitive_attr, so with fewer "
                f"than two groups every draw returns the array unchanged. The "
                f"{n_flip_budget}-row budget was spent on a guaranteed no-op and robustness "
                f"was never tested. Reporting is_robust=None (could not check), not the "
                f"True/1.0 that {n_iterations} identical draws would produce. Supply a "
                f"sensitive attribute with at least two groups."
            )
        elif perturbation_type == "label_noise" and _flip_is_a_no_op(y_pred):
            degenerate_reason = (
                f"sensitivity_analysis: 'label_noise' perturbs by replacing y_pred with "
                f"1 - y_pred, and every value in this y_pred is its own complement, so the "
                f"{n_flip_budget}-row budget was spent on a guaranteed no-op and robustness "
                f"was never tested. Reporting is_robust=None (could not check), not the "
                f"True/1.0 that {n_iterations} identical draws would produce."
            )
    elif perturbation_type == "subsample":
        subsample_budget = int(n * (1 - perturbation_rate))
        if subsample_budget >= n:
            degenerate_reason = (
                f"sensitivity_analysis: perturbation_rate={perturbation_rate} over n={n} rows "
                f"keeps int({n} * {1 - perturbation_rate}) = {subsample_budget} of {n} rows, so "
                f"'subsample' dropped NOTHING and robustness was never tested. Reporting "
                f"is_robust=None (could not check)."
            )
        elif subsample_budget < 1:
            degenerate_reason = (
                f"sensitivity_analysis: perturbation_rate={perturbation_rate} over n={n} rows "
                f"keeps int({n} * {1 - perturbation_rate}) = {subsample_budget} rows, so there "
                f"is nothing left to measure. Reporting is_robust=None (could not check)."
            )
    elif perturbation_type not in ("label_noise", "group_noise", "subsample"):
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    # Perturbed metrics. A degenerate budget draws NOTHING rather than drawing
    # n_iterations copies of the unperturbed metric and grading them.
    n_draws = 0 if degenerate_reason else n_iterations
    perturbed_metrics = np.zeros(n_draws)

    for i in range(n_draws):
        if perturbation_type == "label_noise":
            # Flip random prediction labels
            perturbed_pred = y_pred.copy()
            n_flip = int(n * perturbation_rate)
            flip_idx = rng.choice(n, size=n_flip, replace=False)
            perturbed_pred[flip_idx] = 1 - perturbed_pred[flip_idx]
            perturbed_metrics[i] = metric_func(perturbed_pred, sensitive_attr)

        elif perturbation_type == "group_noise":
            # Randomly reassign some group labels
            perturbed_attr = sensitive_attr.copy()
            n_flip = int(n * perturbation_rate)
            flip_idx = rng.choice(n, size=n_flip, replace=False)
            unique_groups = np.unique(sensitive_attr)
            perturbed_attr[flip_idx] = rng.choice(unique_groups, size=n_flip)
            perturbed_metrics[i] = metric_func(y_pred, perturbed_attr)

        elif perturbation_type == "subsample":
            # Random subsampling
            subsample_size = int(n * (1 - perturbation_rate))
            subsample_idx = rng.choice(n, size=subsample_size, replace=False)
            perturbed_metrics[i] = metric_func(y_pred[subsample_idx], sensitive_attr[subsample_idx])
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    # Draws whose metric could not be computed are DROPPED here, and that used
    # to happen silently. It is the fabricated-all-clear direction of the same
    # defect the permutation test forty lines above already warns about: an
    # unmeasurable draw is a perturbation under which the metric BROKE, so
    # discarding it and grading the survivors biases every answer toward robust.
    # Measured 2026-09-08 on a two-group set whose small group is usually lost
    # to subsampling: 80 of 100 draws vanished and the function reported
    # `is_robust=True, robustness_score=1.0`, a perfect robustness verdict from
    # a fifth of the evidence, with no warning anywhere.
    measurable = np.isfinite(perturbed_metrics)
    n_unmeasurable = int((~measurable).sum())
    perturbed_metrics = perturbed_metrics[measurable]
    n_run = int(len(perturbed_metrics))

    def _unassessable(reason: str) -> SensitivityResult:
        warnings.warn(reason, UserWarning, stacklevel=3)
        return SensitivityResult(
            original_metric=original_metric,
            perturbed_metrics=perturbed_metrics,
            mean_perturbed=float("nan"),
            std_perturbed=float("nan"),
            max_deviation=float("nan"),
            is_robust=None,
            # None, not NaN: `robustness_score` is read straight into an
            # `int(score * 200)` bar width by the rendering adapter, which a NaN
            # would crash. None is the value that adapter already treats as
            # ungraded.
            robustness_score=None,
            perturbation_type=perturbation_type,
            n_iterations_run=n_run,
            n_unmeasurable=n_unmeasurable,
        )

    if not np.isfinite(original_metric):
        return _unassessable(
            "sensitivity_analysis: the unperturbed metric is not finite, so there is "
            "no baseline to measure deviation from. Reporting is_robust=None "
            "(could not check), not a robustness verdict."
        )

    # Checked before the n_run == 0 guard below: with a zero budget no draw was
    # ATTEMPTED, which is a different fact from "every draw came back unmeasurable"
    # and needs its own sentence.
    if degenerate_reason:
        return _unassessable(degenerate_reason)

    if n_run == 0:
        # Reached before np.max, which raises a bare "zero-size array to
        # reduction operation maximum" from inside numpy on an empty array.
        return _unassessable(
            f"sensitivity_analysis: not one of {n_iterations} perturbation draws "
            f"produced a finite metric, so robustness was not measured. Reporting "
            f"is_robust=None (could not check)."
        )

    # Compute statistics
    mean_perturbed = float(np.mean(perturbed_metrics))
    std_perturbed = float(np.std(perturbed_metrics))
    max_deviation = float(np.max(np.abs(perturbed_metrics - original_metric)))

    # Robustness assessment
    # Robust if max deviation is within threshold of original value
    if abs(original_metric) > 1e-10:
        relative_max_dev = max_deviation / abs(original_metric)
    else:
        relative_max_dev = max_deviation

    is_robust: Optional[bool] = relative_max_dev <= robustness_threshold

    # Robustness score: 1 - normalized deviation (capped at 0)
    robustness_score: Optional[float] = max(0.0, 1.0 - relative_max_dev / robustness_threshold)

    if n_unmeasurable:
        if is_robust:
            # The dropped draws are exactly the ones that could have exceeded
            # the threshold, so a pass on the survivors is not a pass: it is an
            # unanswered question. A FAIL needs no such caveat, because
            # max_deviation is a maximum and more draws can only raise it.
            is_robust = None
            robustness_score = None
        warnings.warn(
            f"sensitivity_analysis: {n_unmeasurable} of {n_iterations} perturbation "
            f"draws produced a non-finite metric and were excluded; the verdict rests "
            f"on {n_run}. "
            + (
                "Those draws could have breached the threshold, so is_robust is None "
                "(could not check) rather than True."
                if is_robust is None
                else "The surviving draws already breach the threshold, so is_robust stays False."
            ),
            UserWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        original_metric=original_metric,
        perturbed_metrics=perturbed_metrics,
        mean_perturbed=mean_perturbed,
        std_perturbed=std_perturbed,
        max_deviation=max_deviation,
        is_robust=is_robust,
        robustness_score=robustness_score,
        perturbation_type=perturbation_type,
        n_iterations_run=n_run,
        n_unmeasurable=n_unmeasurable,
    )


def stress_test_fairness(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    perturbation_budgets: List[float] = [0.01, 0.05, 0.10],
    n_iterations: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Comprehensive stress testing of fairness metrics.

    Tests robustness across multiple perturbation types and levels
    to identify potential vulnerabilities.

    Args:
        y_pred: Model predictions
        sensitive_attr: Protected attribute
        metric_func: Fairness metric function
        perturbation_budgets: List of perturbation rates to test
        n_iterations: Iterations per test
        random_state: Random seed

    Returns:
        Dict with results for each perturbation type and level

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

    Ledger row: stress_test_fairness. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    results: Dict[str, Any] = {
        "original_metric": float(metric_func(y_pred, sensitive_attr)),
        "perturbation_results": {},
        # BGL-S2 (2026-09-16): NaN, not 0.0. These are overwritten below from the
        # tests that actually ran, and when NONE ran the initial value is what
        # the caller reads. 0.0 there is the best possible deviation, published
        # for a stress test that measured nothing: at n=2, with all nine tests
        # not assessable and overall_robust=None, this field still said 0.0.
        "worst_case_deviation": float("nan"),
        "worst_case_metric": float("nan"),
        # Three states. `overall_robust` starts True and used to be lowered to
        # False only by a measured failure, so a run in which NOT ONE
        # perturbation was measurable ended as a clean True. The tallies below
        # are what makes the difference visible to a caller.
        "overall_robust": True,
        "n_tests_run": 0,
        "n_tests_not_assessable": 0,
        "not_assessable": [],
    }

    perturbation_types = ["label_noise", "group_noise", "subsample"]
    # Start unmeasured. A measured deviation of exactly 0.0 still wins over NaN
    # below (`not np.isfinite(worst_case_dev)` admits the first finite value),
    # so a genuinely robust run still reports its real 0.0; only a run with
    # nothing to report keeps the NaN.
    worst_case_dev = float("nan")
    worst_case_metric = float("nan")
    any_assessed = False

    for ptype in perturbation_types:
        results["perturbation_results"][ptype] = {}

        for budget in perturbation_budgets:
            sens_result = sensitivity_analysis(
                y_pred,
                sensitive_attr,
                metric_func,
                perturbation_type=ptype,
                perturbation_rate=budget,
                n_iterations=n_iterations,
                random_state=random_state,
            )

            results["perturbation_results"][ptype][f"{int(budget * 100)}%"] = {
                "mean_perturbed": sens_result.mean_perturbed,
                "std_perturbed": sens_result.std_perturbed,
                "max_deviation": sens_result.max_deviation,
                "is_robust": sens_result.is_robust,
                "robustness_score": sens_result.robustness_score,
            }

            # Track worst case. Only a MEASURED deviation can be the worst one:
            # `nan > x` is False, so an unmeasurable test can never win here, and
            # `x > nan` is False too, which is why the finite check comes first
            # rather than relying on the comparison alone.
            dev = sens_result.max_deviation
            if np.isfinite(dev) and (not np.isfinite(worst_case_dev) or dev > worst_case_dev):
                worst_case_dev = float(dev)
                # Find the most extreme metric value
                extreme_idx = np.argmax(
                    np.abs(sens_result.perturbed_metrics - results["original_metric"])
                )
                worst_case_metric = sens_result.perturbed_metrics[extreme_idx]

            label = f"{ptype} @ {int(budget * 100)}%"
            if sens_result.is_robust is None:
                # Could not check. NOT a failure: `if not is_robust` would have
                # read None as fragile and reported a breach nobody measured.
                results["n_tests_not_assessable"] += 1
                results["not_assessable"].append(label)
            else:
                any_assessed = True
                results["n_tests_run"] += 1
                if not sens_result.is_robust:
                    results["overall_robust"] = False

    results["worst_case_deviation"] = worst_case_dev
    results["worst_case_metric"] = float(worst_case_metric)

    if not any_assessed:
        results["overall_robust"] = None
        warnings.warn(
            f"stress_test_fairness: none of the "
            f"{len(perturbation_types) * len(perturbation_budgets)} stress tests could "
            f"be measured, so overall_robust is None (could not check), not True.",
            UserWarning,
            stacklevel=2,
        )
    elif results["n_tests_not_assessable"]:
        warnings.warn(
            f"stress_test_fairness: {results['n_tests_not_assessable']} of "
            f"{len(perturbation_types) * len(perturbation_budgets)} stress tests could "
            f"not be measured ({', '.join(results['not_assessable'])}); overall_robust "
            f"rests on the {results['n_tests_run']} that ran.",
            UserWarning,
            stacklevel=2,
        )

    return results


# Subgroup Robustness Audit (Fairness Gerrymandering Detection)


def subgroup_robustness_audit(
    y_pred: np.ndarray,
    sensitive_attrs: Union[pd.DataFrame, Dict[str, np.ndarray]],
    y_true: Optional[np.ndarray] = None,
    min_subgroup_size: int = 30,
    disparity_threshold: float = 0.10,
    metric: str = "positive_rate",
) -> SubgroupAuditResult:
    """
    Audit fairness across intersectional subgroups.

    Detects "fairness gerrymandering" where a model appears fair on
    pre-defined groups but violates fairness on structured subgroups.

    Based on Kearns et al. (2018): "Preventing fairness gerrymandering:
    Auditing and learning for subgroup fairness."

    Args:
        y_pred: Model predictions
        sensitive_attrs: DataFrame or dict of protected attributes
        y_true: True labels (required for TPR/FPR metrics)
        min_subgroup_size: Minimum samples required per subgroup
        disparity_threshold: Threshold for flagging subgroups (e.g., 0.10 = 10%)
        metric: Metric to compute ('positive_rate', 'tpr', 'fpr', 'error_rate')

    Returns:
        SubgroupAuditResult with analysis of all subgroups.
        `gerrymandering_detected` is three-state: True, False, or None when the
        question could not be answered (no subgroup survived the
        `min_subgroup_size` filter, some were dropped unmeasured and the rest
        cleared, or only one subgroup was measurable). A UserWarning names the
        reason and the dropped subgroups, and `n_subgroups_too_small` /
        `n_subgroups_unmeasurable` carry the counts. `worst_disparity` is NaN,
        never 0.0, when nothing was measured.

    Example:
        >>> attrs = pd.DataFrame({'gender': gender, 'age_group': age_group})
        >>> result = subgroup_robustness_audit(y_pred, attrs)
        >>> if result.gerrymandering_detected is None:
        ...     # `is None`, never a truthiness test: None here is "not checked",
        ...     # and `if result.gerrymandering_detected:` reads it as "clear".
        ...     print(f"Not checked: {result.n_subgroups_too_small} subgroup(s) "
        ...           f"were too small to measure")
        >>> elif result.gerrymandering_detected:
        ...     print(f"Warning: Hidden disparities in subgroups")
        ...     for subgroup, disparity in result.flagged_subgroups[:3]:
        ...         print(f"  {subgroup}: {disparity:.1%} disparity")

    Reference:
        Kearns et al. (2018). "Preventing fairness gerrymandering."
        International Conference on Machine Learning.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: subgroup_robustness_audit. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_pred = np.asarray(y_pred)

    # Convert to DataFrame if dict
    if isinstance(sensitive_attrs, dict):
        sensitive_attrs = pd.DataFrame(sensitive_attrs)

    # A MISSING PREDICTION IS NOT A "NO". `NaN > 0.5` is False, so the
    # positive_rate branch below counted every missing prediction as a negative
    # decision. Measured 2026-10-01 (broken-data check): 400 rows, every
    # prediction NaN, two groups -> both rates 0.0, worst_disparity 0.0 and
    # gerrymandering_detected False, a clean verdict over a model that predicted
    # nothing. Rows without a prediction (or without a label, when the metric
    # needs one) are removed and counted before anything is measured, so a
    # subgroup left with no rows falls into the existing too-small path.
    # Not `|=`: under pandas 3 copy-on-write, `.to_numpy()` hands back a
    # read-only view, and an in-place OR raised "output array is read-only".
    missing = pd.isna(pd.Series(y_pred, dtype=object)).to_numpy()
    if y_true is not None and metric in ("tpr", "fpr", "error_rate"):
        missing = missing | pd.isna(pd.Series(np.asarray(y_true), dtype=object)).to_numpy()
    n_rows_missing = int(missing.sum())
    if n_rows_missing:
        keep = ~missing
        y_pred = y_pred[keep]
        if y_pred.dtype == object:
            y_pred = y_pred.astype(float)
        if y_true is not None:
            y_true = np.asarray(y_true)[keep]
        sensitive_attrs = sensitive_attrs.loc[keep].reset_index(drop=True)

    # Create intersectional subgroup labels
    subgroup_labels = sensitive_attrs.apply(lambda row: "_".join(str(v) for v in row), axis=1)

    # Compute overall rate for comparison
    if metric == "positive_rate":
        overall_rate = np.mean(y_pred > 0.5 if y_pred.dtype == float else y_pred)
    elif metric == "tpr" and y_true is not None:
        y_true = np.asarray(y_true)
        overall_rate = np.mean(y_pred[y_true == 1])
    elif metric == "fpr" and y_true is not None:
        y_true = np.asarray(y_true)
        overall_rate = np.mean(y_pred[y_true == 0])
    elif metric == "error_rate" and y_true is not None:
        y_true = np.asarray(y_true)
        overall_rate = np.mean(y_pred != y_true)
    else:
        overall_rate = np.mean(y_pred)

    subgroup_metrics: Dict[str, Dict[str, Any]] = {}
    flagged_subgroups = []
    # Dropped subgroups are COUNTED and NAMED. Both drops below used to be bare
    # `continue`s, so a run that examined nothing was byte-identical to a run
    # that cleared everything.
    too_small: List[str] = []
    unmeasurable: List[str] = []

    for subgroup in subgroup_labels.unique():
        mask = subgroup_labels == subgroup
        n_samples = np.sum(mask)

        if n_samples < min_subgroup_size:
            # Kearns gerrymandering lives precisely in the small intersectional
            # cells this filter removes, so dropping them silently deletes
            # exactly what this function exists to find and then certifies its
            # absence.
            too_small.append(str(subgroup))
            continue

        # Compute metric for subgroup
        subgroup_pred = y_pred[mask]

        if metric == "positive_rate":
            subgroup_rate = np.mean(
                subgroup_pred > 0.5 if subgroup_pred.dtype == float else subgroup_pred
            )
        elif metric == "tpr" and y_true is not None:
            subgroup_true = y_true[mask]
            pos_mask = subgroup_true == 1
            subgroup_rate = np.mean(subgroup_pred[pos_mask]) if np.sum(pos_mask) > 0 else np.nan
        elif metric == "fpr" and y_true is not None:
            subgroup_true = y_true[mask]
            neg_mask = subgroup_true == 0
            subgroup_rate = np.mean(subgroup_pred[neg_mask]) if np.sum(neg_mask) > 0 else np.nan
        elif metric == "error_rate" and y_true is not None:
            subgroup_true = y_true[mask]
            subgroup_rate = np.mean(subgroup_pred != subgroup_true)
        else:
            subgroup_rate = np.mean(subgroup_pred)

        if np.isnan(subgroup_rate):
            # The requested metric has no value for this cell (e.g. 'fpr' where
            # the cell holds no negative labels). Unmeasured is not unflagged.
            unmeasurable.append(str(subgroup))
            continue

        # Compute disparity vs overall
        disparity = subgroup_rate - overall_rate

        subgroup_metrics[subgroup] = {
            "n_samples": int(n_samples),
            "rate": float(subgroup_rate),
            "disparity_vs_overall": float(disparity),
            # None, not 0.0: with no positive overall rate there is nothing to
            # take a ratio against, and 0.0 reads as "no relative disparity".
            "relative_disparity": (
                float(disparity / overall_rate)
                if np.isfinite(overall_rate) and overall_rate > 0
                else None
            ),
        }

        # Flag subgroups exceeding threshold
        if abs(disparity) > disparity_threshold:
            flagged_subgroups.append((subgroup, disparity))

    # Sort flagged by absolute disparity
    flagged_subgroups = sorted(flagged_subgroups, key=lambda x: abs(x[1]), reverse=True)

    # Find worst subgroup
    if subgroup_metrics:
        worst_subgroup = max(
            subgroup_metrics.keys(), key=lambda k: abs(subgroup_metrics[k]["disparity_vs_overall"])
        )
        worst_disparity = subgroup_metrics[worst_subgroup]["disparity_vs_overall"]
    else:
        worst_subgroup = None
        # NaN, not 0.0. A literal zero here is perfect parity, and it was being
        # reported over zero subgroups.
        worst_disparity = float("nan")

    n_dropped = len(too_small) + len(unmeasurable)
    n_total = len(subgroup_metrics) + n_dropped
    if n_rows_missing:
        too_small_note = f"; {n_rows_missing} row(s) had no prediction and were removed first"
    else:
        too_small_note = ""
    dropped_detail = (
        f"{len(too_small)} below min_subgroup_size={min_subgroup_size} "
        f"({', '.join(too_small) if too_small else 'none'}); "
        f"{len(unmeasurable)} with a non-finite '{metric}' "
        f"({', '.join(unmeasurable) if unmeasurable else 'none'})"
        f"{too_small_note}"
    )

    # Three states, never two: detected / not detected / could not check.
    gerrymandering_detected: Optional[bool] = len(flagged_subgroups) > 0

    if gerrymandering_detected:
        # A real finding is never withdrawn by a caveat. The measured subgroups
        # already breach the threshold and the dropped ones could only add more,
        # so the detection stands; the caller is told what it rests on.
        if n_dropped:
            warnings.warn(
                f"subgroup_robustness_audit: {len(flagged_subgroups)} subgroup(s) "
                f"breach disparity_threshold={disparity_threshold}, so the detection "
                f"stands, but {n_dropped} of {n_total} subgroups were dropped "
                f"unmeasured ({dropped_detail}) and may hold worse.",
                UserWarning,
                stacklevel=2,
            )
    elif not subgroup_metrics or n_rows_missing:
        # Rows removed for a missing prediction can hold the disparity this audit
        # looks for, exactly like a dropped subgroup, so "not detected" over a
        # partial population is "could not check".
        gerrymandering_detected = None
        if not subgroup_metrics:
            msg = (
                f"subgroup_robustness_audit: not one of {n_total} subgroups could be "
                f"measured ({dropped_detail}), so gerrymandering_detected is None "
                f"(could not check), not False, and worst_disparity is NaN, not 0.0. "
                f"No subgroup was examined, so nothing here says they were clear."
            )
        else:
            msg = (
                f"subgroup_robustness_audit: {n_rows_missing} row(s) had no prediction "
                f"and were removed before measuring ({dropped_detail}). None of the "
                f"{len(subgroup_metrics)} measured subgroups breach "
                f"disparity_threshold={disparity_threshold}, but the removed rows could, "
                f"so gerrymandering_detected is None (could not check), not False."
            )
        warnings.warn(msg, UserWarning, stacklevel=2)
    elif n_dropped:
        gerrymandering_detected = None
        warnings.warn(
            f"subgroup_robustness_audit: {n_dropped} of {n_total} subgroups were "
            f"dropped unmeasured ({dropped_detail}) and none of the "
            f"{len(subgroup_metrics)} that were measured breach "
            f"disparity_threshold={disparity_threshold}. Gerrymandering hides in "
            f"exactly the small cells that were dropped, so gerrymandering_detected "
            f"is None (could not check), not False.",
            UserWarning,
            stacklevel=2,
        )
    elif len(subgroup_metrics) < 2:
        gerrymandering_detected = None
        warnings.warn(
            f"subgroup_robustness_audit: only one subgroup "
            f"({next(iter(subgroup_metrics))}) was measured, so there is no second "
            f"subgroup to compare it against and no partition to gerrymander; "
            f"gerrymandering_detected is None (could not check), not False.",
            UserWarning,
            stacklevel=2,
        )

    return SubgroupAuditResult(
        subgroup_metrics=subgroup_metrics,
        flagged_subgroups=flagged_subgroups,
        worst_subgroup=worst_subgroup,
        worst_disparity=worst_disparity,
        gerrymandering_detected=gerrymandering_detected,
        n_subgroups_analyzed=len(subgroup_metrics),
        n_subgroups_flagged=len(flagged_subgroups),
        n_subgroups_too_small=len(too_small),
        n_subgroups_unmeasurable=len(unmeasurable),
        n_rows_missing_prediction=n_rows_missing,
    )


# Comprehensive Significance Testing


def comprehensive_fairness_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metrics: Optional[List[str]] = None,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive statistical significance tests for fairness.

    Performs permutation tests on multiple fairness metrics with
    multiple comparison correction.

    Args:
        y_true: True labels
        y_pred: Predictions
        sensitive_attr: Protected attribute
        metrics: List of metrics to test (default: common classification metrics)
        n_permutations: Number of permutations per test
        alpha: Significance level
        random_state: Random seed

    Returns:
        Dict with test results and corrected p-values

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

    Ledger row: robustness_testing. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    if metrics is None:
        metrics = ["demographic_parity", "equal_opportunity", "predictive_parity"]

    results: Dict[str, Any] = {"metric_tests": {}, "p_values": [], "significant_metrics": []}

    # Define metric functions
    def dp_diff(pred, attr):
        groups = np.unique(attr)
        rates = [np.mean(pred[attr == g]) for g in groups]
        return max(rates) - min(rates) if len(rates) >= 2 else float("nan")

    def eo_diff(pred, attr):
        groups = np.unique(attr)
        tprs = []
        for g in groups:
            mask = (attr == g) & (y_true == 1)
            if np.sum(mask) > 0:
                tprs.append(np.mean(pred[mask]))
        return max(tprs) - min(tprs) if len(tprs) >= 2 else float("nan")

    def pp_diff(pred, attr):
        groups = np.unique(attr)
        ppvs = []
        for g in groups:
            mask = (attr == g) & (pred == 1)
            if np.sum(mask) > 0:
                ppvs.append(np.mean(y_true[mask]))
        return max(ppvs) - min(ppvs) if len(ppvs) >= 2 else float("nan")

    metric_funcs = {
        "demographic_parity": dp_diff,
        "equal_opportunity": eo_diff,
        "predictive_parity": pp_diff,
    }

    # BGL-S2B-FINAL (2026-09-17). THE EFFECTIVE DESIGN TRAVELS WITH THE METRIC.
    # ``permutation_test`` bounds its p-floor from the label array it is given,
    # but two of these three statistics read only a SUBSET of the rows:
    # ``eo_diff`` sees the rows with ``y_true == 1`` and ``pp_diff`` the rows
    # with ``y_pred == 1`` (``y_pred`` is held fixed while the labels are
    # shuffled, so that subset is the same for every draw). Without saying so,
    # the floor reported here was the resample floor, orders of magnitude below
    # anything the subset can reach.
    #
    # Measured at the public entry, 200 rows, 4 predicted positives and 4 true
    # positives (2 per group): permutation_test_equal_opportunity REFUSED the
    # same statistic on the same data with min_attainable_p_value=0.0606,
    # detectable_at_05=False, significant_at_05=None; this function reported
    # equal_opportunity min_attainable_p_value=0.000576 (105x below anything
    # reachable), detectable_at_05=True and a graded significant_at_05=False,
    # and predictive_parity the same. Three graded negatives, n_tests=3,
    # not_assessable=[], for a family in which two members could not have fired
    # whatever the data said.
    #
    # The floor is a LOWER bound on the true floor, so a metric it refuses could
    # not have reached alpha at this shape under any data: nothing measurable is
    # dropped, and the metrics that DO run get a smaller correction family.
    design_rows = {
        "demographic_parity": None,  # reads every row
        "equal_opportunity": int(np.sum(y_true == 1)),
        "predictive_parity": int(np.sum(y_pred == 1)),
    }

    p_values: List[float] = []
    tested: List[str] = []
    not_assessable: List[str] = []
    not_assessable_reasons: Dict[str, str] = {}
    min_attainable: Dict[str, Optional[float]] = {}

    for metric_name in metrics:
        if metric_name not in metric_funcs:
            continue

        test_result = permutation_test(
            y_pred,
            sensitive_attr,
            metric_funcs[metric_name],
            n_permutations=n_permutations,
            random_state=random_state,
            n_design_rows=design_rows.get(metric_name),
        )

        results["metric_tests"][metric_name] = test_result.to_dict()
        if test_result.significant_at_05 is None:
            # C-01. The metric could not be computed, so no test was run. It must
            # not enter the p-value family: a NaN p is not a negative result, and
            # including it inflates the Benjamini-Hochberg family size m, which
            # makes every OTHER metric harder to call significant.
            #
            # READINESS-6 widened what reaches here: a null that COLLAPSED (too
            # few measurable permutations for the test to reach 0.05 at all)
            # now refuses the same way, instead of contributing a finite p=1.0.
            not_assessable.append(metric_name)
            not_assessable_reasons[metric_name] = test_result.design_note or test_result.method
        else:
            p_values.append(test_result.p_value)
            tested.append(metric_name)
            min_attainable[metric_name] = test_result.min_attainable_p_value

    # Apply multiple comparison correction
    p_values_arr = np.array(p_values)
    from ._statistics import benjamini_hochberg_correction

    correction_result = benjamini_hochberg_correction(p_values_arr, alpha)

    results["p_values"] = p_values_arr.tolist()
    results["adjusted_p_values"] = correction_result.adjusted_p_values.tolist()
    results["correction_method"] = "benjamini_hochberg"

    # Identify significant metrics. Indexed over the metrics that were actually
    # TESTED, not over every metric requested: the two lists diverge as soon as
    # one metric is unmeasurable, and indexing the mask by the requested position
    # attributed one metric's result to another.
    for i, metric_name in enumerate(tested):
        if i < len(correction_result.rejection_mask) and correction_result.rejection_mask[i]:
            results["significant_metrics"].append(metric_name)

    # `any_significant` is the ONE field most callers read, so it carries the
    # third state rather than collapsing onto the clean side. False means "tests
    # ran and none was significant"; None means "nothing was testable", which is
    # not the same claim and must not be reported as one.
    results["any_significant"] = len(results["significant_metrics"]) > 0 if tested else None
    results["n_tests"] = len(p_values)

    # THREE STATES. `n_tests` counts what ran. These two say what did not, so
    # "none significant" can be told apart from "nothing was testable". Before
    # this, a run where zero comparisons were possible reported n_tests=3 and
    # any_significant=False, with a NON-degenerate null distribution behind it
    # (measured: mean 0.099, std 0.074), which is indistinguishable from a
    # well-powered negative result.
    results["not_assessable"] = not_assessable
    results["n_not_assessable"] = len(not_assessable)
    results["not_assessable_reasons"] = not_assessable_reasons

    # DESIGN POWER OF THE FAMILY THAT ACTUALLY FORMED. Each surviving test cleared
    # its own 0.05 bar, but the correction below raises the bar it must clear to
    # alpha/m, and a test can lose all power to the family it is corrected in
    # without losing any to itself. `undetectable_metrics` names the tests whose
    # p-floor cannot reach that bar: a "not significant" reading for one of those
    # is an absence of resolution, not evidence that nothing is wrong. Purely
    # additive disclosure; it changes no verdict.
    undetectable: Dict[str, str] = {}
    from ._statistics import detectability as _detectability

    for metric_name in tested:
        can_fire, note = _detectability(
            min_attainable.get(metric_name), n_family=len(p_values), alpha=alpha
        )
        if can_fire is not True:
            undetectable[metric_name] = note
    results["undetectable_metrics"] = undetectable
    if undetectable:
        warnings.warn(
            f"{len(undetectable)} of {len(tested)} tested metric(s) could not have reached "
            f"significance in a corrected family of {len(p_values)}: "
            f"{', '.join(sorted(undetectable))}. Read 'undetectable_metrics' before reading "
            "any 'not significant' verdict for them.",
            UserWarning,
            stacklevel=2,
        )

    if not_assessable:
        warnings.warn(
            f"{len(not_assessable)} of {len(not_assessable) + len(tested)} metric(s) "
            f"could not be computed and were NOT tested: {', '.join(not_assessable)}. "
            "'any_significant' covers only the metrics that were testable.",
            UserWarning,
            stacklevel=2,
        )

    return results
