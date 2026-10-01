"""
vfairness.in_processing.constraints - Constraint-Based Fair Training

This module provides constraint-based approaches to fair machine learning,
where fairness requirements are specified as explicit constraints that
must be satisfied during training.

Key Components:

1. **Fairness Constraints**: Mathematical specifications of fairness
   - DemographicParityConstraint: Equal positive rates
   - EqualizedOddsConstraint: Equal TPR and FPR
   - EqualOpportunityConstraint: Equal TPR
   - FalsePositiveRateParityConstraint: Equal FPR
   - BoundedGroupLossConstraint: Bounded worst-group loss

2. **Constraint Optimization Algorithms**:
   - ExponentiatedGradient: Main algorithm from Agarwal et al. (2018)
   - GridSearch: Grid search over Lagrange multipliers
   - ThresholdOptimizer: Post-processing threshold optimization

Quick Start:
    >>> from sklearn.linear_model import LogisticRegression
    >>> from vfairness.in_processing.constraints import (
    ...     ExponentiatedGradient,
    ...     DemographicParityConstraint,
    ... )
    >>>
    >>> # Create constraint and algorithm
    >>> constraint = DemographicParityConstraint(tolerance=0.05)
    >>> eg = ExponentiatedGradient(
    ...     base_estimator=LogisticRegression(),
    ...     constraint=constraint,
    ... )
    >>>
    >>> # Fit and predict
    >>> result = eg.fit(X_train, y_train, sensitive_attr=gender)
    >>> y_pred = eg.predict(X_test)

Factory Function:
    >>> from vfairness.in_processing.constraints import create_constraint
    >>> constraint = create_constraint('equalized_odds', tolerance=0.05)

References:
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
    - Cotter et al. (2019): Two-Player Games for Non-Convex Optimization
    - Hardt et al. (2016): Equality of Opportunity in Supervised Learning
"""

from .base import (
    # Base classes
    BaseFairnessConstraint,
    BoundedGroupLossConstraint,
    # Result containers
    ConstraintViolation,
    # Constraint implementations
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FairnessConstraintType,
    FalsePositiveRateParityConstraint,
    OptimizationResult,
    # Factory
    create_constraint,
)
from .reductions import (
    EnsembleClassifier,
    # Main algorithms
    ExponentiatedGradient,
    GridSearch,
    LagrangianState,
    # Result classes
    ReductionResult,
    ThresholdOptimizer,
)

__all__ = [
    # Base classes and enums
    "BaseFairnessConstraint",
    "FairnessConstraintType",
    # Result containers
    "ConstraintViolation",
    "OptimizationResult",
    "ReductionResult",
    "LagrangianState",
    # Constraint implementations
    "DemographicParityConstraint",
    "EqualizedOddsConstraint",
    "EqualOpportunityConstraint",
    "FalsePositiveRateParityConstraint",
    "BoundedGroupLossConstraint",
    # Algorithms
    "ExponentiatedGradient",
    "GridSearch",
    "ThresholdOptimizer",
    # Utilities
    "EnsembleClassifier",
    # Factory
    "create_constraint",
]
