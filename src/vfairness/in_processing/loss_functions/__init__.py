"""
vfairness.in_processing.loss_functions - Fairness-Aware Loss Functions

This module provides a comprehensive suite of PyTorch-compatible loss functions
that incorporate fairness constraints directly into the training objective.

Loss Function Categories:
    1. **Group Fairness Losses**: Enforce statistical parity across groups
       - DemographicParityLoss: Equal positive prediction rates
       - EqualizedOddsLoss: Equal TPR and FPR across groups
       - EqualOpportunityLoss: Equal TPR (true positive rates)
       - FalsePositiveRateParityLoss: Equal FPR across groups
       - BoundedGroupLoss: Minimax group loss bounds

    2. **Adversarial Losses**: Train models that don't leak sensitive info
       - AdversarialDebiasingLoss: Adversary predicts sensitive attr
       - ProjectedAdversarialLoss: Gradient projection for stability
       - FairRepresentationLoss: Learn fair representations

    3. **Counterfactual Losses**: Ensure counterfactual fairness
       - CounterfactualFairnessLoss: Same prediction under intervention
       - IndividualFairnessLoss: Similar individuals get similar predictions
       - CausalFairnessLoss: Block unfair causal pathways

Quick Start:
    >>> from vfairness.in_processing.loss_functions import (
    ...     DemographicParityLoss,
    ...     EqualizedOddsLoss,
    ...     AdversarialDebiasingLoss,
    ... )
    >>>
    >>> # Create fairness-aware loss
    >>> loss_fn = DemographicParityLoss(lambda_fairness=0.1)
    >>>
    >>> # Use in training loop
    >>> for x, y, sensitive in dataloader:
    ...     y_pred = model(x)
    ...     loss = loss_fn(y_pred, y, sensitive)
    ...     loss.backward()
    ...     optimizer.step()

Factory Function:
    >>> from vfairness.in_processing.loss_functions import create_fairness_loss
    >>> loss_fn = create_fairness_loss('demographic_parity', lambda_fairness=0.1)

References:
    - Hardt et al. (2016): Equality of Opportunity in Supervised Learning
    - Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning
    - Kusner et al. (2017): Counterfactual Fairness
    - Zafar et al. (2017): Fairness Constraints
"""

from .adversarial import (
    AdversarialDebiasingLoss,
    FairRepresentationLoss,
    ProjectedAdversarialLoss,
)
from .base import (
    TORCH_AVAILABLE,
    # Base classes
    BaseFairnessLoss,
    BaseLossType,
    FairnessMetricType,
    # Data containers
    LossComponents,
    TrainingMetrics,
    # Utility functions
    check_torch_available,
    compute_group_rates,
    create_group_masks,
    soft_rate_computation,
)
from .fairness_losses import (
    BoundedGroupLoss,
    # Group fairness losses
    DemographicParityLoss,
    EqualizedOddsLoss,
    EqualOpportunityLoss,
    # General-purpose loss
    FairnessAwareBCELoss,
    FalsePositiveRateParityLoss,
    # Factory function
    create_fairness_loss,
)

# Conditionally import adversary utilities
if TORCH_AVAILABLE:
    from .adversarial import (
        Adversary,
        GradientReversalLayer,
    )

from .counterfactual import (
    CausalFairnessLoss,
    CounterfactualFairnessLoss,
    IndividualFairnessLoss,
)

__all__ = [
    # Base classes and utilities
    "BaseFairnessLoss",
    "BaseLossType",
    "FairnessMetricType",
    "LossComponents",
    "TrainingMetrics",
    "check_torch_available",
    "soft_rate_computation",
    "create_group_masks",
    "compute_group_rates",
    "TORCH_AVAILABLE",
    # Group fairness losses
    "FairnessAwareBCELoss",
    "DemographicParityLoss",
    "EqualizedOddsLoss",
    "EqualOpportunityLoss",
    "FalsePositiveRateParityLoss",
    "BoundedGroupLoss",
    # Adversarial losses
    "AdversarialDebiasingLoss",
    "ProjectedAdversarialLoss",
    "FairRepresentationLoss",
    # Counterfactual losses
    "CounterfactualFairnessLoss",
    "IndividualFairnessLoss",
    "CausalFairnessLoss",
    # Factory
    "create_fairness_loss",
]

# Conditional exports
if TORCH_AVAILABLE:
    __all__.extend(
        [
            "GradientReversalLayer",
            "Adversary",
        ]
    )
