"""
vfairness.post_processing.ranking - Ranking Fairness Interventions

Post-hoc re-ranking that improves group exposure parity while keeping ranking
utility (NDCG) within a bounded budget. This is the intervention counterpart to
the ranking fairness metrics in
vfairness.evaluation.vfairness_metrics.ranking, whose exposure and parity
functions it reuses.

Example:
    >>> from vfairness.post_processing.ranking import exposure_parity_rerank
    >>> result = exposure_parity_rerank(scores, groups, max_utility_loss=0.1)
    >>> new_order = result['reranked_order']
    >>> print(result['exposure_parity_diff_before'], result['exposure_parity_diff_after'])
"""

from .rerank import exposure_parity_rerank

__all__ = [
    "exposure_parity_rerank",
]
