"""
vfairness.post_processing.threshold_optimization - Group-Specific Threshold Optimization

This module provides tools for optimizing classification thresholds to achieve
fairness constraints while maintaining model performance. Threshold optimization
is a post-processing technique that adjusts decision boundaries per demographic
group to satisfy fairness criteria.

Core Components:
    A. Threshold Optimizers
       - ThresholdOptimizer: Base optimizer for finding optimal thresholds
       - GroupThresholdOptimizer: Per-group threshold optimization
       - MultiObjectiveThresholdOptimizer: Pareto-optimal threshold selection

    B. Fairness Constraints
       - Demographic Parity: Equal positive prediction rates
       - Equalized Odds: Equal TPR and FPR across groups
       - Equal Opportunity: Equal TPR across groups
       - Predictive Parity: Equal precision across groups

    C. Analysis Tools
       - ThresholdAnalyzer: Comprehensive threshold analysis
       - ROC-based threshold selection
       - Cost-sensitive threshold optimization

Key Concepts:
    - Threshold optimization adjusts the decision boundary (default 0.5) to
      achieve fairness goals while considering accuracy trade-offs
    - Different thresholds can be applied to different demographic groups
    - This is a post-processing approach that doesn't require model retraining

Example:
    >>> from vfairness.post_processing.threshold_optimization import (
    ...     GroupThresholdOptimizer, ThresholdAnalyzer
    ... )
    >>>
    >>> # Find group-specific thresholds for equalized odds
    >>> optimizer = GroupThresholdOptimizer(constraint='equalized_odds')
    >>> result = optimizer.fit(y_true, y_prob, sensitive_attr)
    >>> print(f"Thresholds: {result.group_thresholds}")
    >>>
    >>> # Apply optimized thresholds
    >>> y_pred_fair = optimizer.predict(y_prob, sensitive_attr)
    >>>
    >>> # Analyze threshold impact
    >>> analyzer = ThresholdAnalyzer(y_true, y_prob, sensitive_attr)
    >>> report = analyzer.analyze_threshold_range(thresholds=np.linspace(0.1, 0.9, 9))

References:
    - Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in
      Supervised Learning. NeurIPS.
    - Corbett-Davies, S., et al. (2017). Algorithmic Decision Making and the
      Cost of Fairness. KDD.
    - Menon, A. K., & Williamson, R. C. (2018). The Cost of Fairness in
      Binary Classification. FAT*.
"""

from .analyzer import (
    ThresholdAnalysisReport,
    ThresholdAnalyzer,
    ThresholdImpactResult,
)
from .constraints import (
    FairnessConstraintType,
    compute_constraint_violation,
    find_feasible_thresholds,
)
from .optimizer import (
    # Base classes
    BaseThresholdOptimizer,
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdConstraint,
    # Optimizers
    ThresholdOptimizer,
    ThresholdResult,
    # Factory
    create_threshold_optimizer,
)

__all__ = [
    # Base classes
    "BaseThresholdOptimizer",
    "ThresholdResult",
    "ThresholdConstraint",
    # Optimizers
    "ThresholdOptimizer",
    "GroupThresholdOptimizer",
    "MultiObjectiveThresholdOptimizer",
    "create_threshold_optimizer",
    # Analyzer
    "ThresholdAnalyzer",
    "ThresholdAnalysisReport",
    "ThresholdImpactResult",
    # Constraints
    "FairnessConstraintType",
    "compute_constraint_violation",
    "find_feasible_thresholds",
]
