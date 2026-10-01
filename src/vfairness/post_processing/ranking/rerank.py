"""
Exposure-parity re-ranking (post-processing fairness intervention).

Ranking and recommendation systems distribute visibility unevenly: items near
the top of a ranking receive far more exposure than items lower down. When the
top of a utility-optimal ranking is dominated by a single demographic group,
other groups are systematically starved of exposure even when they contain
relevant items.

This module provides a post-hoc intervention that re-orders an existing ranking
to improve group exposure parity while keeping ranking utility (NDCG) within a
bounded budget. It is the post-processing counterpart to the ranking fairness
METRICS in vfairness.evaluation.vfairness_metrics.ranking, which this module
reuses rather than reimplements.

Approach:
    - Start from the utility-optimal order (items sorted by score, descending).
    - Walk positions from top to bottom. At each position, consider the best
      remaining item of each group as a candidate.
    - Keep only candidates that still allow the ranking to be completed with
      total NDCG at or above (1 - max_utility_loss) * NDCG_optimal. This is a
      hard, provable budget: the best possible completion of any partial order
      is the remaining items sorted by score and paired with the remaining
      (descending) position exposures (rearrangement inequality), so a candidate
      is feasible only if that best completion stays above the floor.
    - Among the feasible candidates, place the item whose group is currently
      most under-exposed relative to its population share.

Because the top-scored remaining item is always feasible, the greedy never gets
stuck and the produced order is always a valid permutation whose NDCG respects
the utility budget.

Exposure at position p uses the same log-position discount as the metrics
module (1 / log2(p + 2)), a linear decay when exposure_type='linear', or a
geometric decay when exposure_type='geometric'. The three discounts are exactly
the three the metrics module computes.

References:
    - Singh & Joachims (2018): "Fairness of Exposure in Rankings"
    - Biega et al. (2018): "Equity of Attention"
    - Zehlike et al. (2017): "FA*IR: A Fair Top-k Ranking Algorithm"
"""

from typing import Dict, List, Literal

import numpy as np

from ...evaluation.vfairness_metrics.ranking import (
    _compute_position_exposure,
    exposure_parity_difference,
    get_ranking_group_metrics,
)


def _position_exposures(n_items: int, exposure_type: str) -> np.ndarray:
    """Exposure delivered by each position 0..n-1 (reuses the metric discount)."""
    return np.array(
        [_compute_position_exposure(p, n_items, exposure_type) for p in range(n_items)],
        dtype=float,
    )


def _best_completion(remaining_scores: np.ndarray, exposures_from: np.ndarray) -> float:
    """
    Maximum DCG achievable for the remaining positions.

    The remaining items are paired optimally with the remaining position
    exposures. Since exposures decrease with position, the rearrangement
    inequality gives the maximum by pairing the highest remaining scores with
    the earliest (highest-exposure) remaining positions.
    """
    if remaining_scores.size == 0:
        return 0.0
    sorted_desc = np.sort(remaining_scores)[::-1]
    span = min(sorted_desc.size, exposures_from.size)
    return float(np.dot(sorted_desc[:span], exposures_from[:span]))


def _positions_from_order(order: List[int], n_items: int) -> np.ndarray:
    """Convert an ordered list of item indices to a per-item position array."""
    positions = np.zeros(n_items, dtype=int)
    positions[np.asarray(order, dtype=int)] = np.arange(n_items)
    return positions


def _dcg_of_order(order: np.ndarray, scores: np.ndarray, exposures: np.ndarray) -> float:
    """Discounted cumulative gain of an ordered list, using scores as gains."""
    order = np.asarray(order, dtype=int)
    span = min(order.size, exposures.size)
    return float(np.sum(scores[order[:span]] * exposures[:span]))


def exposure_parity_rerank(
    scores,
    groups,
    # Mirrors the metric functions this delegates to: they already accept only
    # these three discounts, so the constraint belongs on the public entry point.
    exposure_type: Literal["log", "linear", "geometric"] = "log",
    max_utility_loss: float = 0.1,
    random_state: int = 42,
) -> Dict:
    """
    Re-rank items to improve group exposure parity within a utility budget.

    Args:
        scores: 1-D array of item relevance / scores (higher is better). Every
            score must be finite and non-negative; see Raises for why neither is
            a soft requirement here.
        groups: Array of group labels per item, aligned to scores.
        exposure_type: Position-exposure discount, 'log' (default), 'linear' or
            'geometric'. All three are the discounts the ranking metric module
            computes, and all three are accepted here.
        max_utility_loss: Maximum allowed drop in NDCG relative to the
            utility-optimal order, as a fraction in [0, 1]. Default 0.1.
        random_state: Seed used for deterministic tie-breaking. Default 42.

    Returns:
        Dict with:
            reranked_order: list[int] item indices in the re-ranked order.
            exposure_before: {group: avg_exposure} for the utility-optimal order.
            exposure_after: {group: avg_exposure} for the re-ranked order.
            exposure_parity_diff_before: float parity difference before.
            exposure_parity_diff_after: float parity difference after.
            ndcg_before: float NDCG of the utility-optimal order (1.0).
            ndcg_after: float NDCG of the re-ranked order.
            utility_loss: float ndcg_before - ndcg_after (>= 0).
            group_share: {group: float} population share per group.
            n_items: int number of items.

    Raises:
        ValueError: if scores and groups differ in length, if exposure_type is
            not one of the three discounts, if any score is non-finite (an
            unscored item is promoted to the TOP by the feasibility fallback, so
            the parity reported afterwards measures that placement), or if any
            score is negative (a negative ideal DCG puts the utility floor above
            the optimum, so the greedy silently returns the order it was given
            while reporting utility_loss 0.0).

    Notes:
        NDCG here uses the input scores as gains and the utility-optimal order
        (scores sorted descending) as the ideal. The re-ranked NDCG is
        guaranteed to be at least (1 - max_utility_loss) * ndcg_before because a
        candidate is placed only when the ranking can still be completed above
        that floor. Scores are treated as non-negative relevance values.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    n_items = int(scores.shape[0])

    if groups.shape[0] != n_items:
        raise ValueError("scores and groups must have the same length")

    # The guard must accept exactly what the signature advertises. It used to
    # refuse 'geometric' while the Literal annotation offered it, so a caller who
    # trusted the type hint got a runtime ValueError for a documented value. The
    # discount itself was never missing: _compute_position_exposure has supported
    # geometric decay (0.85 ** position) all along, so the refusal was the bug,
    # not the gap. Keep this tuple in step with the Literal above.
    if exposure_type not in ("log", "linear", "geometric"):
        raise ValueError("exposure_type must be 'log', 'linear' or 'geometric'")

    # An item with NO score is could-not-check, not the best item in its group.
    #
    # The candidate loop below already skips a non-finite score (np.nanargmax,
    # see its comment), but the FEASIBILITY FALLBACK a few lines under it is a
    # plain np.argmax, which returns the NaN's index. Worse, it is not a corner
    # case there: every comparison against a NaN completion is False, so no
    # candidate is ever feasible and that fallback fires at EVERY position while
    # any unscored item is left. Measured 2026-09-27 on 10 items, group A
    # holding the five best scores, with the WORST-scored item (index 9) set to
    # NaN:
    #
    #   reranked_order              [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    #   exposure_parity_diff_before 0.2707 -> after 0.0132
    #   warnings                    []
    #
    # so the item nobody scored took position 0, the highest-exposure slot in
    # the ranking, and the parity improvement this function reported was
    # produced entirely by that fabricated placement. `reranked_order` is the
    # decision this function hands over and it has no field for a third state,
    # so refusing IS the third state, exactly as _refuse_unscored_rows does in
    # the sibling threshold_optimization module.
    unscored = ~np.isfinite(scores)
    if np.any(unscored):
        idx = np.flatnonzero(unscored)[:10].tolist()
        raise ValueError(
            f"exposure_parity_rerank: {int(unscored.sum())} of {n_items} item(s) have no "
            f"usable score (NaN or infinite), first at index {idx}. Refusing rather than "
            "ranking them: an unscored item is promoted to the top of the ranking by the "
            "feasibility fallback, and the exposure parity reported afterwards is then a "
            "measure of that placement rather than of the data. Drop or impute the "
            "unscored items before re-ranking."
        )

    # A NEGATIVE score is not a low relevance here, it silently disarms the
    # intervention. With idcg < 0 the utility floor (1 - max_utility_loss) *
    # idcg sits ABOVE the optimum, so no candidate is ever feasible, the
    # fallback keeps the top-scored item at every position, and the greedy
    # degenerates to the utility-optimal order it started from. Measured
    # 2026-09-27 on 10 items scored -0.2 ... -2.0 across two groups:
    #
    #   reranked_order == the utility-optimal order (a complete no-op)
    #   exposure_parity_diff 0.2707 before AND after
    #   ndcg_before 1.0, ndcg_after 1.0, utility_loss 0.0, warnings []
    #
    # a mitigation that did nothing, reporting that it kept full utility. The
    # module docstring and the Notes above already state that scores are
    # non-negative relevance values: the rearrangement-inequality argument the
    # whole feasibility proof rests on needs that, and so does reading NDCG as
    # a fraction of the achievable utility.
    if np.any(scores < 0.0):
        n_neg = int(np.count_nonzero(scores < 0.0))
        raise ValueError(
            f"exposure_parity_rerank: {n_neg} of {n_items} score(s) are negative "
            f"(minimum {float(np.min(scores)):.6g}). Refusing rather than re-ranking: a "
            "negative gain puts the NDCG utility floor above the utility-optimal order, "
            "so no candidate is ever feasible, the intervention silently returns the "
            "order it was given, and ndcg_after 1.0 with utility_loss 0.0 reads as a "
            "re-ranking that cost nothing. Shift or exponentiate the scores into "
            "non-negative relevance values first."
        )

    max_utility_loss = float(max_utility_loss)

    exposures = _position_exposures(n_items, exposure_type)

    # Utility-optimal order: sort by score descending, stable for determinism.
    optimal_order = np.argsort(-scores, kind="stable")
    positions_before = _positions_from_order(list(optimal_order), n_items)

    # Population share per group (keys are stringified to match the metric funcs).
    unique_groups = list(np.unique(groups))
    group_share = {str(g): float(np.sum(groups == g)) / n_items for g in unique_groups}

    idcg = _dcg_of_order(optimal_order, scores, exposures)
    ndcg_before = 1.0 if idcg <= 0.0 else _dcg_of_order(optimal_order, scores, exposures) / idcg

    # Deterministic, seeded tie-break priority per item.
    rng = np.random.RandomState(random_state)
    tiebreak = rng.permutation(n_items)

    floor_dcg = (1.0 - max_utility_loss) * idcg
    feasibility_tol = 1e-9

    if len(unique_groups) < 2:
        # Nothing to balance; the optimal order is the answer.
        reranked_order = [int(i) for i in optimal_order]
    else:
        placed = np.zeros(n_items, dtype=bool)
        delivered = {str(g): 0.0 for g in unique_groups}
        total_delivered = 0.0
        dcg_so_far = 0.0
        reranked_order = []

        for p in range(n_items):
            e_p = exposures[p]
            remaining_idx = np.where(~placed)[0]
            remaining_scores = scores[remaining_idx]
            remaining_groups = groups[remaining_idx]
            exposures_after_p = exposures[p + 1 :]

            # Candidates: the best remaining item of each group.
            candidates = []
            for g in unique_groups:
                gmask = remaining_groups == g
                if not np.any(gmask):
                    continue
                group_items = remaining_idx[gmask]
                # np.argmax returns the NaN's index, so an item with an
                # unmeasurable score would be selected as the group's BEST and
                # promoted up the ranking. nanargmax skips it instead.
                group_scores = scores[group_items]
                if not np.any(np.isfinite(group_scores)):
                    continue
                best_item = group_items[int(np.nanargmax(group_scores))]
                candidates.append(int(best_item))

            # Keep candidates whose placement still allows an above-floor finish.
            feasible = []
            for item in candidates:
                without_item = remaining_scores[remaining_idx != item]
                completion = _best_completion(without_item, exposures_after_p)
                achievable = dcg_so_far + scores[item] * e_p + completion
                if achievable >= floor_dcg - feasibility_tol:
                    feasible.append(item)

            if not feasible:
                # The top-scored remaining item is always feasible in theory;
                # this guards against floating-point corner cases.
                feasible = [int(remaining_idx[int(np.argmax(remaining_scores))])]

            # Place the item whose group is most under-exposed vs its share.
            # feasible is guaranteed non-empty above, and the first loop pass always
            # wins (chosen_key is None), so seeding from feasible[0] is the same
            # selection as starting at None without leaving chosen Optional.
            chosen = feasible[0]
            chosen_key = None
            for item in feasible:
                gk = str(groups[item])
                deficit = group_share[gk] * total_delivered - delivered[gk]
                key = (deficit, float(scores[item]), int(tiebreak[item]))
                if chosen_key is None or key > chosen_key:
                    chosen_key = key
                    chosen = item

            reranked_order.append(int(chosen))
            placed[chosen] = True
            delivered[str(groups[chosen])] += e_p
            total_delivered += e_p
            dcg_so_far += scores[chosen] * e_p

    reranked_arr = np.asarray(reranked_order, dtype=int)
    positions_after = _positions_from_order(reranked_order, n_items)

    ndcg_after = 1.0 if idcg <= 0.0 else _dcg_of_order(reranked_arr, scores, exposures) / idcg
    utility_loss = ndcg_before - ndcg_after

    # Reuse the ranking metric functions for exposure and parity (min_group_size=1
    # so every present group is included in this intervention view).
    before_metrics = get_ranking_group_metrics(
        positions_before, groups, min_group_size=1, exposure_type=exposure_type
    )
    after_metrics = get_ranking_group_metrics(
        positions_after, groups, min_group_size=1, exposure_type=exposure_type
    )
    exposure_before = {g: float(m["avg_exposure"]) for g, m in before_metrics.items()}
    exposure_after = {g: float(m["avg_exposure"]) for g, m in after_metrics.items()}

    parity_before = exposure_parity_difference(
        positions_before, groups, min_group_size=1, exposure_type=exposure_type
    )
    parity_after = exposure_parity_difference(
        positions_after, groups, min_group_size=1, exposure_type=exposure_type
    )

    return {
        "reranked_order": [int(i) for i in reranked_order],
        "exposure_before": exposure_before,
        "exposure_after": exposure_after,
        "exposure_parity_diff_before": float(parity_before),
        "exposure_parity_diff_after": float(parity_after),
        "ndcg_before": float(ndcg_before),
        "ndcg_after": float(ndcg_after),
        "utility_loss": float(utility_loss),
        "group_share": group_share,
        "n_items": n_items,
    }
