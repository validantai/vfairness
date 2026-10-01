"""
vfairness.in_processing - Training-Time Fairness Interventions Module

This module provides comprehensive tools for fairness-aware model training,
enabling practitioners to incorporate fairness constraints directly into
the machine learning training process.

Module Overview:
================

1. **Loss Functions** (loss_functions/)
   Fairness-aware loss functions for PyTorch that incorporate fairness
   penalties directly into the optimization objective.

   - DemographicParityLoss: Equal positive prediction rates
   - EqualizedOddsLoss: Equal TPR and FPR across groups
   - EqualOpportunityLoss: Equal TPR across groups
   - AdversarialDebiasingLoss: Adversarial training for fairness
   - CounterfactualFairnessLoss: Counterfactual fairness penalties
   - IndividualFairnessLoss: Similar individuals get similar predictions

2. **Constraints** (constraints/)
   Constraint-based training using the reductions approach from
   Agarwal et al. (2018).

   - ExponentiatedGradient: Main algorithm for constrained training
   - GridSearch: Grid search over constraint weights
   - ThresholdOptimizer: Post-processing threshold optimization
   - DemographicParityConstraint, EqualizedOddsConstraint, etc.

3. **Regularizers** (regularizers/)
   Regularization techniques that penalize statistical dependence
   between predictions and sensitive attributes.

   - StatisticalParityRegularizer: Penalizes mean prediction differences
   - ConditionalIndependenceRegularizer: Enforces ŷ ⊥ a | y
   - GroupFairnessRegularizer: Flexible group fairness penalty
   - HilbertSchmidtRegularizer: HSIC-based independence

4. **Calibrators** (calibrators/)
   Group-specific calibration methods that can be trained jointly
   with models.

   - TemperatureScalingCalibrator: Group-specific temperature scaling
   - PlattScalingCalibrator: Group-specific Platt scaling
   - TrainableGroupCalibrator: Unified trainable calibrator

5. **Wrappers** (wrappers/)
   Scikit-learn compatible wrappers for easy integration.

   - FairClassifier: Fairness-aware classifier wrapper
   - FairRegressor: Fairness-aware regressor wrapper

6. **Analyzer** (analyzer.py)
   Unified analysis and reporting.

   - FairnessTrainingAnalyzer: Comprehensive training analysis
   - FairnessTrainingReport: Detailed report generation

Quick Start:
============

Example 1: Using Fairness-Aware Loss Functions (PyTorch)
>>> import torch
>>> from vfairness.in_processing.loss_functions import DemographicParityLoss
>>>
>>> loss_fn = DemographicParityLoss(lambda_fairness=0.1)
>>> for x, y, sensitive in dataloader:
...     y_pred = torch.sigmoid(model(x))
...     loss = loss_fn(y_pred, y, sensitive)
...     loss.backward()
...     optimizer.step()

Example 2: Using Constraint-Based Training (sklearn-compatible)
>>> from sklearn.linear_model import LogisticRegression
>>> from vfairness.in_processing.constraints import ExponentiatedGradient
>>> from vfairness.in_processing.constraints import DemographicParityConstraint
>>>
>>> constraint = DemographicParityConstraint(tolerance=0.05)
>>> eg = ExponentiatedGradient(
...     base_estimator=LogisticRegression(),
...     constraint=constraint,
... )
>>> result = eg.fit(X_train, y_train, sensitive_attr=gender)
>>> y_pred = eg.predict(X_test)

Example 3: Using FairClassifier Wrapper
>>> from sklearn.ensemble import RandomForestClassifier
>>> from vfairness.in_processing.wrappers import FairClassifier
>>>
>>> clf = FairClassifier(
...     base_estimator=RandomForestClassifier(),
...     fairness_constraint='equalized_odds',
...     tolerance=0.05
... )
>>> clf.fit(X_train, y_train, sensitive_attr=gender)
>>> y_pred = clf.predict(X_test)
>>> print(f"Constraint satisfied: {clf.fairness_result_.constraint_satisfied}")

Example 4: Full Analysis with Recommendations
>>> from vfairness.in_processing import FairnessTrainingAnalyzer
>>>
>>> analyzer = FairnessTrainingAnalyzer(
...     X=X_train, y=y_train, sensitive_attr=gender,
...     fairness_constraint='demographic_parity'
... )
>>> report = analyzer.full_analysis(base_estimator=LogisticRegression())
>>> print(report.summary())

References:
===========
- Hardt et al. (2016): Equality of Opportunity in Supervised Learning
- Agarwal et al. (2018): A Reductions Approach to Fair Classification
- Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning
- Kusner et al. (2017): Counterfactual Fairness
- Zafar et al. (2017): Fairness Constraints: Mechanisms for Fair Classification
- Guo et al. (2017): On Calibration of Modern Neural Networks
"""

# Analyzer
from .analyzer import (
    # Analyzer
    FairnessTrainingAnalyzer,
    # Report classes
    FairnessTrainingReport,
    MethodComparison,
    TrainingRecommendation,
    # CS-T-41 baseline vs fair comparison helper
    baseline_comparison_summary,
)
from .calibrators import (
    TORCH_AVAILABLE as CAL_TORCH_AVAILABLE,
)

# Calibrators
from .calibrators import (
    BetaCalibrator,
    # Training helper
    CalibrationAwareTrainer,
    # Enums
    CalibrationMethodType,
    CalibrationState,
    FocalCalibrator,
    PlattScalingCalibrator,
    # Calibrators
    TemperatureScalingCalibrator,
    TrainableGroupCalibrator,
    # Factory
    create_group_calibrator,
)

# Constraints
from .constraints import (
    # Base classes
    BaseFairnessConstraint,
    BoundedGroupLossConstraint,
    ConstraintViolation,
    # Constraint implementations
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    # Algorithms
    ExponentiatedGradient,
    FairnessConstraintType,
    FalsePositiveRateParityConstraint,
    GridSearch,
    OptimizationResult,
    # Result classes
    ReductionResult,
    ThresholdOptimizer,
    # Factory
    create_constraint,
)

# Convergence diagnostics (CS-T-44, torch-free adversarial debiasing)
from .diagnostics import (
    adversarial_convergence_diagnostics,
    sklearn_adversarial_debiasing,
)
from .loss_functions import (
    TORCH_AVAILABLE as LOSS_TORCH_AVAILABLE,
)
from .loss_functions import (
    # Adversarial losses
    AdversarialDebiasingLoss,
    # Base classes
    BaseFairnessLoss,
    BaseLossType,
    BoundedGroupLoss,
    CausalFairnessLoss,
    # Counterfactual losses
    CounterfactualFairnessLoss,
    DemographicParityLoss,
    EqualizedOddsLoss,
    EqualOpportunityLoss,
    # Group fairness losses
    FairnessAwareBCELoss,
    FairnessMetricType,
    FairRepresentationLoss,
    FalsePositiveRateParityLoss,
    IndividualFairnessLoss,
    LossComponents,
    ProjectedAdversarialLoss,
    TrainingMetrics,
    # Factory
    create_fairness_loss,
)
from .regularizers import (
    TORCH_AVAILABLE as REG_TORCH_AVAILABLE,
)

# Regularizers
from .regularizers import (
    # Base class
    BaseRegularizer,
    ConditionalIndependenceRegularizer,
    CorrelationPenalty,
    GroupFairnessRegularizer,
    HilbertSchmidtRegularizer,
    RegularizerMetrics,
    RegularizerType,
    # Implementations
    StatisticalParityRegularizer,
    # Factory
    create_regularizer,
)

# Wrappers
from .wrappers import (
    # Wrappers
    FairClassifier,
    # Result classes
    FairClassifierResult,
    FairRegressor,
    # Factory functions
    make_fair_classifier,
    make_fair_regressor,
)

# Check for PyTorch availability
TORCH_AVAILABLE = LOSS_TORCH_AVAILABLE

__all__ = [
    # Loss Functions
    # Base classes
    "BaseFairnessLoss",
    "BaseLossType",
    "FairnessMetricType",
    "LossComponents",
    "TrainingMetrics",
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
    # Constraints
    # Base classes
    "BaseFairnessConstraint",
    "FairnessConstraintType",
    "ConstraintViolation",
    "OptimizationResult",
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
    # Result classes
    "ReductionResult",
    # Factory
    "create_constraint",
    # Regularizers
    "BaseRegularizer",
    "RegularizerType",
    "RegularizerMetrics",
    "StatisticalParityRegularizer",
    "ConditionalIndependenceRegularizer",
    "GroupFairnessRegularizer",
    "HilbertSchmidtRegularizer",
    "CorrelationPenalty",
    "create_regularizer",
    # Calibrators
    "CalibrationMethodType",
    "CalibrationState",
    "TemperatureScalingCalibrator",
    "PlattScalingCalibrator",
    "BetaCalibrator",
    "FocalCalibrator",
    "TrainableGroupCalibrator",
    "CalibrationAwareTrainer",
    "create_group_calibrator",
    # Wrappers
    "FairClassifierResult",
    "FairClassifier",
    "FairRegressor",
    "make_fair_classifier",
    "make_fair_regressor",
    # Analyzer
    "FairnessTrainingReport",
    "MethodComparison",
    "TrainingRecommendation",
    "FairnessTrainingAnalyzer",
    "baseline_comparison_summary",
    # Convergence diagnostics (torch-free)
    "sklearn_adversarial_debiasing",
    "adversarial_convergence_diagnostics",
    # Utilities
    "TORCH_AVAILABLE",
]
