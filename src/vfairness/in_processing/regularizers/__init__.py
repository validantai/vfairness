"""
vfairness.in_processing.regularizers - Fairness Regularization Techniques

This module provides regularization techniques that encourage fairness
by penalizing statistical dependence between predictions and sensitive
attributes during training.

Regularizers Implemented:
    1. StatisticalParityRegularizer: Penalizes mean prediction differences
    2. ConditionalIndependenceRegularizer: Enforces ŷ ⊥ a | y
    3. GroupFairnessRegularizer: Flexible group fairness penalty
    4. HilbertSchmidtRegularizer: HSIC-based independence measure
    5. CorrelationPenalty: Simple correlation-based penalty

Quick Start:
    >>> from vfairness.in_processing.regularizers import (
    ...     StatisticalParityRegularizer,
    ...     GroupFairnessRegularizer,
    ... )
    >>>
    >>> # Create regularizer
    >>> regularizer = StatisticalParityRegularizer(strength=0.1)
    >>>
    >>> # Use in training loop
    >>> loss = bce_loss(y_pred, y_true)
    >>> fairness_penalty = regularizer(y_pred, sensitive_attr)
    >>> total_loss = loss + fairness_penalty
    >>> total_loss.backward()

Factory Function:
    >>> from vfairness.in_processing.regularizers import create_regularizer
    >>> regularizer = create_regularizer('group_fairness', strength=0.1, fairness_metric='eo')

References:
    - Kamishima et al. (2012): Fairness-Aware Classifier with Prejudice Remover
    - Zafar et al. (2017): Fairness Beyond Disparate Treatment
    - Gretton et al. (2005): Measuring Statistical Dependence with HSIC
"""

from .fairness_regularizers import (
    TORCH_AVAILABLE,
    # Base class
    BaseRegularizer,
    ConditionalIndependenceRegularizer,
    CorrelationPenalty,
    GroupFairnessRegularizer,
    HilbertSchmidtRegularizer,
    # Data containers
    RegularizerMetrics,
    RegularizerType,
    # Regularizer implementations
    StatisticalParityRegularizer,
    # Utilities
    check_torch_available,
    # Factory
    create_regularizer,
)

__all__ = [
    # Base class and enums
    "BaseRegularizer",
    "RegularizerType",
    # Data containers
    "RegularizerMetrics",
    # Regularizer implementations
    "StatisticalParityRegularizer",
    "ConditionalIndependenceRegularizer",
    "GroupFairnessRegularizer",
    "HilbertSchmidtRegularizer",
    "CorrelationPenalty",
    # Factory
    "create_regularizer",
    # Utilities
    "check_torch_available",
    "TORCH_AVAILABLE",
]
