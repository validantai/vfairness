"""
vfairness.post_processing.reweighting - Output Reweighting for Fairness

This module provides tools for post-hoc adjustment of model predictions to
achieve fairness constraints through output reweighting and transformation.
Unlike threshold optimization, reweighting modifies the predicted probabilities
themselves rather than just the decision boundary.

Core Components:
    A. Reweighting Methods
       - PredictionReweighter: Adjust predictions based on group membership
       - RejectionOptionClassifier: Modify predictions near decision boundary
       - CalibratedEqualizer: Equalize calibrated probabilities across groups

    B. Adjustment Strategies
       - Multiplicative adjustment
       - Additive adjustment
       - Distribution matching
       - Rejection option based classification (ROC)

    C. Analysis Tools
       - ReweightingAnalyzer: Impact analysis of reweighting strategies
       - Trade-off visualization

Key Concepts:
    - Output reweighting adjusts predicted probabilities post-hoc
    - Can achieve fairness without changing the model or threshold
    - Useful when probability estimates need to be fair, not just decisions
    - Trade-off between fairness and calibration/accuracy

Example:
    >>> from vfairness.post_processing.reweighting import (
    ...     PredictionReweighter, RejectionOptionClassifier
    ... )
    >>>
    >>> # Reweight predictions for demographic parity
    >>> reweighter = PredictionReweighter(constraint='demographic_parity')
    >>> reweighter.fit(y_true, y_prob, sensitive_attr)
    >>> y_prob_fair = reweighter.transform(y_prob, sensitive_attr)
    >>>
    >>> # Use rejection option classification
    >>> roc = RejectionOptionClassifier(theta=0.1)
    >>> roc.fit(y_true, y_prob, sensitive_attr)
    >>> y_pred_fair = roc.predict(y_prob, sensitive_attr)

References:
    - Kamiran, F., Karim, A., & Zhang, X. (2012). Decision Theory for
      Discrimination-Aware Classification. ICDM.
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
    - Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in
      Supervised Learning. NeurIPS.
"""

from .analyzer import (
    ReweightingAnalysisReport,
    ReweightingAnalyzer,
    ReweightingImpactResult,
)
from .reweighter import (
    # Base classes
    BaseReweighter,
    CalibratedEqualizer,
    DistributionMatcher,
    # Reweighters
    PredictionReweighter,
    RejectionOptionClassifier,
    ReweightingMethod,
    ReweightingResult,
    # Factory
    create_reweighter,
)

__all__ = [
    # Base classes
    "BaseReweighter",
    "ReweightingResult",
    "ReweightingMethod",
    # Reweighters
    "PredictionReweighter",
    "RejectionOptionClassifier",
    "CalibratedEqualizer",
    "DistributionMatcher",
    "create_reweighter",
    # Analyzer
    "ReweightingAnalyzer",
    "ReweightingAnalysisReport",
    "ReweightingImpactResult",
]
