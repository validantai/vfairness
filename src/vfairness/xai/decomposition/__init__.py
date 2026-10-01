"""
vfairness.xai.decomposition
===========================

The spine: per-feature decomposition of a group-fairness metric over the
SHAP-value distribution. Lundberg (2020) CHI Fair & Responsible AI
Workshop. Identity guard: |sum(per_feature) - total_disparity| <= 1e-6.
"""

from .shap_fairness import lundberg_fairness_decomposition, proxy_score

__all__ = ["lundberg_fairness_decomposition", "proxy_score"]
