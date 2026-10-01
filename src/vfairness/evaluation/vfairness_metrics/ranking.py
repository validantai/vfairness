"""
Ranking fairness metrics for vfairness.

This module provides fairness metrics for ranking/recommendation tasks,
including exposure parity and related measures.

Ranking fairness is important for:
    - Search engines
    - Recommendation systems
    - Information retrieval
    - Any system that orders items/candidates

Key metrics:
    - Exposure Parity: Whether groups receive equal visibility in rankings
    - Attention Parity: Whether groups receive equal attention (position-weighted)
    - NDKL: Normalized discounted KL-divergence for group representation

References:
    - Singh & Joachims (2018): "Fairness of Exposure in Rankings"
    - Biega et al. (2018): "Equity of Attention"
    - Zehlike et al. (2017): "FA*IR: A Fair Top-k Ranking Algorithm"

Example:
    >>> from vfairness import exposure_parity_difference
    >>> # rankings: position of each item (0 = top)
    >>> # groups: group membership of each item
    >>> exp_diff = exposure_parity_difference(rankings, groups)
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from ...exceptions import ConfigurationError, InvalidDataError
from ._grouping import GroupManager
from ._validation import ArrayLike
from .classification import _warn_dropped_groups

# ---------------------------------------------------------------------------
# CONVENTION (fail closed): this module obeys the SAME rule the classification
# module states at length in classification.py (see the CONVENTION block there,
# and _warn_dropped_groups, which is imported rather than copied). A
# between-group metric that could not actually COMPARE two groups returns NaN,
# never the "perfect" sentinel: 0.0 for a difference, 1.0 for a ratio, and an
# is_fair of None rather than True for a verdict nobody was able to reach.
#
# The incident this file's half of the convention exists for: 12 ranked items
# where group A holds positions 0-7 and group B (4 items) holds 8-11. With the
# default min_group_size=5, B is dropped, and every metric here reported perfect
# parity: exposure_parity_difference 0.0, exposure_parity_ratio 1.0, ndkl 0.0,
# attention_weighted_rank_fairness 0.0 with is_fair True, and NOT ONE WARNING.
# Measured with B kept, the truth is 0.209, 0.576 (a four-fifths FAIL), 0.578
# and 0.941 against its own 0.1 threshold: group B receives 12 percent of the
# attention. The sentinels were the exact inverse of the finding, and
# adapters_ranking rendered the is_fair True straight onto a green PASS badge,
# so an auditor read an all-clear for a ranking that shut a group out.
#
# Sites below that say "unmeasurable -> NaN" mean exactly this. Do not "restore"
# a 0.0 / 1.0 / is_fair=True return at any of them.
# ---------------------------------------------------------------------------


@dataclass
class RankingFairnessResult:
    """
    Result container for ranking fairness metrics.

    Attributes:
        metric_name: Name of the computed metric
        value: Overall metric value (e.g., max difference across groups)
        group_exposures: Dict mapping groups to their exposure values
        group_attentions: Dict mapping groups to attention values
        is_fair: Whether the metric passes its threshold, or None when the
            comparison never happened (fewer than two groups met
            ``min_group_size``). Three states, never two: True is assessed-pass,
            False is assessed-fail, None is could-not-check.
        threshold: Threshold used for evaluation
    """

    metric_name: str
    value: float
    group_exposures: Dict[str, float]
    group_attentions: Optional[Dict[str, float]] = None
    # CRITICAL: the default is None, not True. A result nobody graded must not
    # read as a pass. rendering/adapters_ranking.py turns a True straight into a
    # green PASS badge on an exported SVG an auditor reads, so a defaulted True
    # here is a certificate issued for a measurement that never ran.
    is_fair: Optional[bool] = None
    threshold: float = 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "group_exposures": self.group_exposures,
            "group_attentions": self.group_attentions,
            "is_fair": self.is_fair,
            "threshold": self.threshold,
        }


def _compute_position_exposure(
    position: int, total_positions: int, exposure_type: str = "log"
) -> float:
    """
    Compute exposure value for a given position.

    Args:
        position: Position in ranking (0 = top, N-1 = bottom)
        total_positions: Total number of positions
        exposure_type: 'log' (logarithmic decay), 'linear', or 'geometric'

    Returns:
        Exposure value (higher = more exposure)
    """
    if position < 0:
        return 0.0

    if exposure_type == "log":
        # Logarithmic decay (like NDCG)
        return 1.0 / np.log2(position + 2)  # +2 to avoid log(1)=0
    elif exposure_type == "linear":
        # Linear decay
        return max(0, 1 - position / total_positions)
    elif exposure_type == "geometric":
        # Geometric decay with ratio 0.85
        return 0.85**position
    else:
        raise ConfigurationError(f"Unknown exposure_type: {exposure_type}")


def _undefined_order_reason(arr: np.ndarray, n_items: int) -> Optional[str]:
    """Why this SCORE column defines no ranking order, or None if it defines one.

    Two degenerate columns carry no order at all:

    * any non-finite score. ``np.argsort`` sorts NaN to the END, so an item
      whose score is missing is silently ranked LAST -- a position nobody
      measured, graded as the worst one.
    * every score identical. There is no "higher = better" left to read.

    Tested with ``np.unique`` on the RAW data, never with a variance test:
    ``np.var(np.full(20, 0.5)) == 0.0`` is False at most n, so a variance guard
    passes for the round number in a fixture and fails on real data.
    """
    if n_items == 0:
        return None
    n_nonfinite = int(np.count_nonzero(~np.isfinite(arr)))
    if n_nonfinite:
        return (
            f"{n_nonfinite} of {n_items} ranking scores are not finite (NaN/inf), "
            "so no ranking order is defined"
        )
    if n_items > 1 and np.unique(arr).size == 1:
        return f"all {n_items} ranking scores are identical, so no ranking order is defined"
    return None


def _rankings_to_positions(
    rankings: np.ndarray, n_items: int
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Convert a rankings input to 0-based positions.

    Inputs are POSITIONS only when they are integer-valued AND form an exact
    permutation of 0..n_items-1; anything else (float relevance scores in
    [0,1], graded relevance with ties, scores above n_items, negatives) is
    treated as SCORES and converted via a stable descending argsort.

    The previous heuristic (``np.any(rankings > n_items)``) misclassified
    documented [0,1] relevance scores as positions and floored them all to 0
    with ``astype(int)``, reporting perfect parity for arbitrarily unfair
    rankings.

    Returns ``(positions, None)`` when an order exists, and ``(None, reason)``
    when it does not. THREE STATES, NEVER TWO, and the guard sits ABOVE the
    dispatch so that every one of the five callers inherits it rather than one
    of them being fixed and the fabrication moving to its siblings.

    Positions are FLOATS because tied scores share the mid-rank of the slots
    their block occupies (see the note at the argsort below): an integer slot for
    a tied item is the caller's row order wearing a measurement's clothes. An
    exact permutation input is still returned as integers, unchanged.

    CRITICAL, do not restore the unconditional argsort. A stable descending
    argsort of an ALL-TIED or ALL-NaN column returns the IDENTITY permutation,
    i.e. the caller's own ROW ORDER, and the metrics above then measured the
    fairness of that invented ranking. Executed on this repo before this guard:
    ``exposure_parity_ratio(np.full(12, np.nan), ['A']*6+['B']*6)`` returned
    0.5410755479122771 -- BIT-FOR-BIT the value of a genuinely shut-out
    ranking on the same rows, below the four-fifths bound, so ``check_threshold``
    returned FAIL and ``ranking_fairness_to_svg`` painted a red FAIL badge
    carrying that number. Interleaving the same non-data flipped it to 0.8187,
    a PASS. A disparate-impact verdict a regulator reads, derived from nothing
    but the order the caller happened to pass the rows in.
    """
    arr = np.asarray(rankings, dtype=float)
    if n_items == 0:
        return np.zeros(0, dtype=int), None
    integer_valued = bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)))
    is_permutation = (
        integer_valued
        and arr.min() >= 0
        and arr.max() <= n_items - 1
        and len(np.unique(arr)) == n_items
    )
    if is_permutation:
        return arr.astype(int), None
    # Scores: an order has to EXIST before it can be read off.
    reason = _undefined_order_reason(arr, n_items)
    if reason is not None:
        return None, reason
    # Scores: higher = better = earlier position, and TIED SCORES SHARE THE
    # MID-RANK of the slots their block occupies.
    #
    # BGL5 A-evaluation-4, 2026-09-27. THE TIE GUARD WAS ALL OR NOTHING.
    # ``_undefined_order_reason`` refuses a column only when EVERY score is
    # identical, so ONE distinct score restored the unconditional stable argsort
    # for all the rest, and a stable argsort of a tied block is the caller's own
    # ROW ORDER. Measured before this change, scores = [0.9] + [0.5] * 11 with
    # groups ['A'] * 6 + ['B'] * 6 at min_group_size=5, under
    # warnings.simplefilter("error"):
    #     A-block first -> avg_position A 2.500, B 8.500,
    #                      avg_exposure 0.5508 vs 0.2980,
    #                      exposure_parity_ratio 0.5411  (a FAIL badge)
    #     the SAME 12 (score, group) pairs interleaved
    #                   -> avg_position A 5.000, B 6.000,
    #                      avg_exposure 0.4667 vs 0.3821,
    #                      exposure_parity_ratio 0.8187  (a PASS)
    #     0 warnings either way. Those are bit-for-bit the two numbers the
    #     CRITICAL note above cites as the defect, reached with one distinct
    #     score instead of none.
    # After: both arrangements give avg_position A 5.000, B 6.000 and
    # exposure_parity_ratio 0.7500, because 11 items sharing one score share one
    # position. The number is no longer a function of the order the rows arrived
    # in, which is the whole point; mid-ranking is the standard treatment of ties
    # (it is what rank correlations use) and it is measured, not withheld.
    #
    # An UNTIED column is untouched: every block holds one item, so its mid-rank
    # IS its slot and every metric above returns the identical value it did
    # before. Verified by execution on np.arange(12) and on random score columns.
    order = np.argsort(-arr, kind="stable")
    positions = np.empty(n_items, dtype=float)
    positions[order] = np.arange(n_items, dtype=float)
    # np.unique on the RAW values, never a variance or difference threshold: this
    # is about values that are the SAME, and an epsilon here would silently merge
    # scores that genuinely differ.
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if counts.size and int(counts.max()) > 1:
        inverse = np.asarray(inverse).ravel()
        block_sums = np.bincount(inverse, weights=positions, minlength=counts.size)
        positions = (block_sums / counts)[inverse]
    return positions, None


def _warn_undefined_order(metric_name: str, reason: str) -> None:
    """Disclose an unreadable ranking the same way a dropped group is disclosed.

    The NaN is the machine-readable signal; this is the human one. Named per
    metric because these functions return a bare float with no result object to
    carry a ``could_not_check`` field.
    """
    warnings.warn(
        f"{metric_name}: {reason}. Returning NaN (could-not-check), not a value: "
        "ranking the rows in the order they arrived would measure the caller's "
        "row order, not the model's ranking.",
        UserWarning,
        stacklevel=3,
    )


def exposure_parity_difference(
    rankings: ArrayLike,
    groups: ArrayLike,
    *,
    min_group_size: int = 5,
    exposure_type: Literal["log", "linear", "geometric"] = "log",
    normalize: bool = True,
) -> float:
    """
    Compute exposure parity difference across groups.

    Exposure parity measures whether different demographic groups receive
    equal visibility in a ranking. Higher-ranked items receive more exposure
    (visibility), and this metric checks if that exposure is distributed
    fairly across groups.

    Args:
        rankings: Array of positions (0 = top) OR relevance scores (higher = better)
        groups: Array of group memberships for each item
        min_group_size: Minimum items per group to include in calculation
        exposure_type: How exposure decays with position:
            - 'log': Logarithmic decay (like NDCG), default
            - 'linear': Linear decay
            - 'geometric': Exponential decay with ratio 0.85
        normalize: If True, normalize exposure to [0, 1] range

    Returns:
        Maximum absolute difference in average exposure between any two groups.
        0 = perfect parity, higher = more disparity. NaN (insufficient evidence)
        when fewer than two groups meet ``min_group_size``, because no
        between-group comparison happened there, and NaN when ``rankings`` is a
        score column that defines no order at all (any non-finite score, or
        every score identical), because there is then no ranking to measure.
        Both refusals are disclosed as a ``UserWarning`` naming the reason.

    Example:
        >>> # 4 items: 2 from group A ranked higher, 2 from B ranked lower
        >>> rankings = np.array([0, 1, 2, 3])  # Positions
        >>> groups = np.array(['A', 'A', 'B', 'B'])
        >>> diff = exposure_parity_difference(rankings, groups)
        >>> print(f"Exposure difference: {diff:.3f}")

    Note:
        This implements a simplified version of Singh & Joachims (2018)
        "Fairness of Exposure in Rankings". The original paper considers
        the probability of examination, relevance, and fairness constraints.

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

    Ledger row: exposure_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    rankings = np.asarray(rankings)
    groups = np.asarray(groups)

    if len(rankings) != len(groups):
        raise InvalidDataError("rankings and groups must have same length")

    n_items = len(rankings)

    # Positions vs scores: robust detection (see _rankings_to_positions)
    position_array, order_undefined = _rankings_to_positions(rankings, n_items)

    # Initialize group manager
    gm = GroupManager(groups, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    # No ranking at all -> NaN, never a difference computed from the row order.
    if order_undefined is not None or position_array is None:
        _warn_undefined_order(
            "exposure_parity_difference",
            order_undefined or "no order could be derived from the rankings",
        )
        return float("nan")

    # Fewer than two qualifying groups: there is no pair to compare, so the
    # comparison is unmeasurable -> NaN (see the CONVENTION note above). A 0.0
    # here read as PERFECT EXPOSURE PARITY on a ranking whose dropped group sat
    # in the bottom four slots.
    if len(valid_groups) < 2:
        return float("nan")

    # Compute exposure for each item
    exposures = np.array(
        [_compute_position_exposure(pos, n_items, exposure_type) for pos in position_array]
    )

    # Normalize if requested
    if normalize:
        max_exposure = np.max(exposures)
        if max_exposure > 0:
            exposures = exposures / max_exposure

    # Compute average exposure per group
    group_avg_exposures = {}
    for group in valid_groups:
        mask = gm.get_mask(group)
        group_avg_exposures[group] = np.mean(exposures[mask])

    # Compute max difference
    exposure_values = list(group_avg_exposures.values())
    max_diff = max(exposure_values) - min(exposure_values)

    return max_diff


def exposure_parity_ratio(
    rankings: ArrayLike,
    groups: ArrayLike,
    *,
    min_group_size: int = 5,
    exposure_type: Literal["log", "linear", "geometric"] = "log",
) -> float:
    """
    Compute exposure parity ratio (min/max) across groups.

    Like the 80% rule for hiring, this computes the ratio of minimum
    to maximum average exposure across groups.

    Args:
        rankings: Array of positions or scores
        groups: Array of group memberships
        min_group_size: Minimum items per group
        exposure_type: How exposure decays with position

    Returns:
        Ratio of min to max average group exposure.
        1.0 = perfect parity, < 0.8 may indicate disparate impact. NaN
        (insufficient evidence) when fewer than two groups meet
        ``min_group_size``, when no group received any exposure at all, or when
        ``rankings`` is a score column that defines no order (any non-finite
        score, or every score identical), because no between-group ratio
        happened in any of those cases. NaN routes through ``check_threshold``
        to COULD_NOT_CHECK; a fabricated finite ratio would have been graded
        against the four-fifths bound and painted onto an exported badge. The
        refusal is disclosed as a ``UserWarning`` naming the reason.

    Example:
        >>> ratio = exposure_parity_ratio(rankings, groups)
        >>> if ratio < 0.8:
        ...     print("Potential disparate impact in ranking exposure")

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

    Ledger row: exposure_parity_ratio. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    rankings = np.asarray(rankings)
    groups = np.asarray(groups)

    if len(rankings) != len(groups):
        raise InvalidDataError("rankings and groups must have same length")

    n_items = len(rankings)

    # Positions vs scores: robust detection (see _rankings_to_positions)
    position_array, order_undefined = _rankings_to_positions(rankings, n_items)

    gm = GroupManager(groups, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    # No ranking at all -> NaN. This is the four-fifths screen: a finite float
    # here is graded by check_threshold and painted onto an exported FAIL badge,
    # and check_threshold's own NOT_CHECKED third state is reachable only for
    # NaN/None, so a fabricated ratio bypasses it entirely.
    if order_undefined is not None or position_array is None:
        _warn_undefined_order(
            "exposure_parity_ratio",
            order_undefined or "no order could be derived from the rankings",
        )
        return float("nan")

    # Unmeasurable -> NaN, never the perfect-ratio sentinel 1.0. This is a
    # four-fifths-style screen: a 1.0 here certified a ranking whose dropped
    # group actually sat at 0.576 of the best-served group's exposure.
    if len(valid_groups) < 2:
        return float("nan")

    # Compute exposure per item
    exposures = np.array(
        [_compute_position_exposure(pos, n_items, exposure_type) for pos in position_array]
    )

    # Average exposure per group
    group_avg_exposures = {}
    for group in valid_groups:
        mask = gm.get_mask(group)
        group_avg_exposures[group] = np.mean(exposures[mask])

    exposure_values = list(group_avg_exposures.values())
    min_exp = min(exposure_values)
    max_exp = max(exposure_values)

    if max_exp == 0:
        # 0/0 is not parity, it is an undefined ratio: no group received any
        # exposure, so there is nothing to divide. Defence in depth (the
        # position discounts are all strictly positive over 0..n-1 positions),
        # but a 1.0 sentinel here would be the same false certificate as above.
        return float("nan")

    return min_exp / max_exp


def attention_weighted_rank_fairness(
    rankings: ArrayLike,
    groups: ArrayLike,
    *,
    min_group_size: int = 5,
    attention_model: Literal["cascade", "position"] = "position",
) -> RankingFairnessResult:
    """
    Compute attention-weighted ranking fairness.

    This extends exposure parity to consider user attention models,
    accounting for the fact that users may not examine all items.

    Args:
        rankings: Array of positions (0 = top)
        groups: Array of group memberships
        min_group_size: Minimum items per group
        attention_model: User attention model
            - 'position': Attention decays with position (P(examine|pos))
            - 'cascade': Cascade model (user may stop early)

    Returns:
        RankingFairnessResult with exposure and attention metrics. When fewer
        than two groups meet ``min_group_size`` the comparison never happened:
        ``value`` is NaN and ``is_fair`` is None (could-not-check), never 0.0
        and True. The same three-state answer is returned when ``rankings`` is a
        score column that defines no order (any non-finite score, or every score
        identical): ``value`` NaN, ``is_fair`` None and every group's exposure
        and attention NaN, because all of them are derived from position. The
        refusal is disclosed as a ``UserWarning`` naming the reason.

    References:
        Biega et al. (2018): "Equity of Attention"

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

    Ledger row: attention_weighted_rank_fairness. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    rankings = np.asarray(rankings)
    groups = np.asarray(groups)
    n_items = len(rankings)

    # Positions vs scores: robust detection (see _rankings_to_positions)
    position_array, order_undefined = _rankings_to_positions(rankings, n_items)

    gm = GroupManager(groups, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    # No ranking at all -> NaN AND is_fair None. Every number below is derived
    # from position (attention weight 1/(pos+1)), so with no order there is no
    # exposure and no attention either: the groups are still named, because the
    # group inventory IS measured, and each one reports NaN rather than a share
    # of an invented ranking. is_fair stays None, the could-not-check state the
    # dataclass already documents, so the badge reads NOT CHECKED and not FAIL.
    if order_undefined is not None or position_array is None:
        _warn_undefined_order(
            "attention_weighted_rank_fairness",
            order_undefined or "no order could be derived from the rankings",
        )
        return RankingFairnessResult(
            metric_name="attention_weighted_rank_fairness",
            value=float("nan"),
            group_exposures={g: float("nan") for g in valid_groups},
            group_attentions={g: float("nan") for g in valid_groups},
            is_fair=None,
            threshold=0.1,
        )

    # Compute attention weights based on model
    if attention_model == "position":
        # Simple position-based attention decay
        attention_weights = np.array([1.0 / (pos + 1) for pos in position_array])
    else:  # cascade
        # Cascade model: P(examine item k) = P(not click items 1..k-1)
        # Simplified: exponential decay with 0.7 continuation probability
        attention_weights = np.array([0.7**pos for pos in position_array])

    # Normalize
    attention_weights = attention_weights / np.sum(attention_weights)

    # Compute per-group metrics
    group_exposures = {}
    group_attentions = {}

    for group in valid_groups:
        mask = gm.get_mask(group)
        total_attention = np.sum(attention_weights[mask])

        # Exposure: proportion of group's items * average attention
        group_exposures[group] = np.mean(attention_weights[mask]) * n_items
        # Attention: total attention received by group
        group_attentions[group] = total_attention

    # Compute difference
    #
    # Unmeasurable -> NaN AND is_fair None, never 0.0 with is_fair True. This
    # was the worst of the four sites: it did not merely return a sentinel, it
    # attached a VERDICT to it, and rendering/adapters_ranking.py prints that
    # verdict onto the badge in preference to anything else it knows. On the
    # incident data it certified as fair a ranking in which the dropped group
    # received 12 percent of the attention.
    exposure_vals = list(group_exposures.values())
    if len(exposure_vals) < 2:
        max_diff: float = float("nan")
        is_fair: Optional[bool] = None
    else:
        max_diff = max(exposure_vals) - min(exposure_vals)
        is_fair = bool(max_diff <= 0.1)

    return RankingFairnessResult(
        metric_name="attention_weighted_rank_fairness",
        value=max_diff,
        group_exposures=group_exposures,
        group_attentions=group_attentions,
        is_fair=is_fair,
        threshold=0.1,
    )


def normalized_discounted_kl_divergence(
    rankings: ArrayLike,
    groups: ArrayLike,
    target_distribution: Optional[Dict[str, float]] = None,
    *,
    min_group_size: int = 5,
    top_k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted KL-Divergence for group representation.

    Measures how much the group distribution in the ranking deviates from
    a target distribution (default: uniform), with position discounting.

    Args:
        rankings: Array of positions (0 = top)
        groups: Array of group memberships
        target_distribution: Target proportion for each group.
            If None, uses uniform distribution.
        min_group_size: Minimum items per group
        top_k: Only consider top-k items. If None, use all.

    Returns:
        NDKL score. 0 = matches target distribution, higher = more deviation.
        NaN (insufficient evidence) when fewer than two groups meet
        ``min_group_size``, when the target distribution places no mass on the
        surviving groups, when no position was scored at all, or when
        ``rankings`` is a score column that defines no order (any non-finite
        score, or every score identical), because no divergence was computed in
        any of those cases. The last refusal is disclosed as a ``UserWarning``
        naming the reason.

    Example:
        >>> # Check if top-10 rankings match population demographics
        >>> ndkl = normalized_discounted_kl_divergence(
        ...     rankings, groups,
        ...     target_distribution={'A': 0.6, 'B': 0.4},
        ...     top_k=10
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

    Ledger row: normalized_discounted_kl_divergence. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    rankings = np.asarray(rankings)
    groups = np.asarray(groups)
    n_items = len(rankings)

    # Positions vs scores: robust detection (see _rankings_to_positions)
    position_array, order_undefined = _rankings_to_positions(rankings, n_items)

    gm = GroupManager(groups, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    # No ranking at all -> NaN. NDKL is a POSITION-DISCOUNTED divergence: with
    # no order, the whole discount schedule is read off the caller's row order,
    # which the same data proved by swinging 0.485 to 0.105 on a permutation.
    if order_undefined is not None or position_array is None:
        _warn_undefined_order(
            "normalized_discounted_kl_divergence",
            order_undefined or "no order could be derived from the rankings",
        )
        return float("nan")

    # Unmeasurable -> NaN. A representation divergence over one group is not a
    # divergence of zero, it is no comparison at all.
    if len(valid_groups) < 2:
        return float("nan")

    # Default to uniform distribution
    if target_distribution is None:
        target_distribution = {g: 1.0 / len(valid_groups) for g in valid_groups}

    # Normalize target distribution to valid groups
    total_target = sum(target_distribution.get(g, 0) for g in valid_groups)
    if total_target == 0:
        # The caller's target places no mass on any surviving group, so there is
        # nothing to measure the ranking against. A 0.0 here claimed the ranking
        # MATCHES that target exactly.
        return float("nan")
    target_distribution = {g: target_distribution.get(g, 0) / total_target for g in valid_groups}

    # Limit to top-k if specified
    if top_k is not None:
        top_k = min(top_k, n_items)
        # Get indices of top-k items
        top_k_mask = position_array < top_k
    else:
        top_k_mask = np.ones(n_items, dtype=bool)
        top_k = n_items

    # Compute discounted cumulative representation
    kl_sum = 0.0
    normalization = 0.0

    # BGL5 A-evaluation-4, 2026-09-27. STEP OVER THE DISTINCT POSITIONS, NOT OVER
    # INTEGER SLOTS. ``_rankings_to_positions`` now gives a block of tied scores
    # the MID-RANK of the slots it occupies, so a tied position is fractional and
    # ``position_array == k`` for integer k would match nothing: every tied block
    # would have been dropped from both the KL sum and its normalisation, in
    # silence. Iterating over the positions that exist keeps a block as a block
    # (its cumulative representation is defined; the order WITHIN it is not) and
    # gives it the summed discount of the slots it fills.
    #
    # For an UNTIED column this is the previous loop exactly: each block holds one
    # item, level == k, and the discount is 1/log2(k+2). Verified by running the
    # pre-fix loop beside this one on five untied fixtures (a permutation, its
    # reverse, random scores, [0,1] relevance, ints above n_items): bit-for-bit
    # identical, e.g. ndkl(np.arange(12), ['A']*6+['B']*6) ==
    # 0.48112668216872206 before and after. On the tied column it is 0.136105 in
    # BOTH arrangements, where the pre-fix loop gave the intra-block row order.
    slot = 0
    for level in np.unique(position_array[top_k_mask]):
        # Items at this position (a tie block shares one)
        at_position_k = (position_array == level) & top_k_mask

        if not np.any(at_position_k):  # pragma: no cover - unique() guarantees one
            continue

        # Discount factor: the sum over the slots this block fills, so a block of
        # one is bit-for-bit the old 1/log2(k + 2).
        n_at_level = int(np.count_nonzero(at_position_k))
        discount = float(sum(1.0 / np.log2(s + 2) for s in range(slot, slot + n_at_level)))
        slot += n_at_level
        normalization += discount

        # Compute empirical distribution up to this position (inclusive)
        items_up_to_k = position_array <= level
        if not np.any(items_up_to_k):
            continue

        # Count groups up to k
        empirical_dist = {}
        total_up_to_k = np.sum(items_up_to_k)
        for g in valid_groups:
            g_mask = gm.get_mask(g)
            count = np.sum(items_up_to_k & g_mask)
            empirical_dist[g] = count / total_up_to_k if total_up_to_k > 0 else 0

        # KL divergence at position k
        kl_k = 0.0
        for g in valid_groups:
            p = empirical_dist.get(g, 0)
            q = target_distribution.get(g, 1e-10)
            if p > 0:
                kl_k += p * np.log(p / max(q, 1e-10))

        kl_sum += discount * kl_k

    # Normalize
    if normalization > 0:
        return kl_sum / normalization
    # No position carried an item (top_k=0, or an empty ranking), so the sum has
    # no denominator and nothing was scored. Unmeasurable -> NaN, not a 0.0 that
    # reads as "the ranking matches the target distribution perfectly".
    return float("nan")


def get_ranking_group_metrics(
    rankings: ArrayLike,
    groups: ArrayLike,
    *,
    min_group_size: int = 5,
    exposure_type: Literal["log", "linear", "geometric"] = "log",
) -> Dict[str, Dict[str, float]]:
    """
    Get per-group ranking metrics.

    Args:
        rankings: Array of positions
        groups: Array of group memberships
        min_group_size: Minimum items per group
        exposure_type: Exposure calculation method

    Returns:
        Dict mapping group names to their metrics:
        {
            'group_name': {
                'count': int,
                'avg_position': float,
                'avg_exposure': float,
                'min_position': float,
                'max_position': float,
                'median_position': float
            }
        }

        When ``rankings`` is a score column that defines no order (any
        non-finite score, or every score identical) every POSITION-derived cell
        is NaN and only ``count`` is reported, because only the group inventory
        was measured. Disclosed as a ``UserWarning`` naming the reason.

        Tied scores share the MID-RANK of the slots their block occupies, so a
        position can be fractional and does not depend on the order the rows
        were passed in. An untied column gives the same whole numbers it always
        did.
    """
    rankings = np.asarray(rankings)
    groups = np.asarray(groups)
    n_items = len(rankings)

    # Positions vs scores: robust detection (see _rankings_to_positions)
    position_array, order_undefined = _rankings_to_positions(rankings, n_items)

    gm = GroupManager(groups, min_group_size=min_group_size)
    # A dropped group is simply ABSENT from the returned dict, and the exposure
    # panel that consumes it then draws a complete-looking table with the
    # shut-out group missing. No sentinel to fix here, but the omission has to
    # be disclosed for the same reason the metrics above disclose theirs.
    _warn_dropped_groups(gm)
    results = {}

    # No ranking at all -> every POSITION-derived cell is NaN, while `count`
    # stays, because the group inventory is the one thing here that really was
    # measured. rendering/adapters_ranking._num and ._count already refuse NaN
    # and draw those cells as "not measured", so the panel shows the groups and
    # withholds the numbers instead of drawing bars for the caller's row order.
    if order_undefined is not None or position_array is None:
        _warn_undefined_order(
            "get_ranking_group_metrics",
            order_undefined or "no order could be derived from the rankings",
        )
        return {
            group: {
                "count": int(np.sum(gm.get_mask(group))),
                "avg_position": float("nan"),
                "avg_exposure": float("nan"),
                "min_position": float("nan"),
                "max_position": float("nan"),
                "median_position": float("nan"),
            }
            for group in gm.get_valid_groups()
        }

    for group in gm.get_valid_groups():
        mask = gm.get_mask(group)
        group_positions = position_array[mask]

        exposures = np.array(
            [_compute_position_exposure(pos, n_items, exposure_type) for pos in group_positions]
        )

        results[group] = {
            "count": int(np.sum(mask)),
            "avg_position": float(np.mean(group_positions)),
            "avg_exposure": float(np.mean(exposures)),
            # float, not int: a tie block shares the MID-RANK of its slots (see
            # _rankings_to_positions), and int() would truncate 6.5 to 6, i.e.
            # report a position no item holds and claim a precision the tie does
            # not have. An untied column still gives whole numbers here.
            "min_position": float(np.min(group_positions)),
            "max_position": float(np.max(group_positions)),
            "median_position": float(np.median(group_positions)),
        }

    return results
