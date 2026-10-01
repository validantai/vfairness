"""
vfairness.operations.experimentation: A/B Testing for Fairness (Unit 4)

This package provides a controlled experimentation framework for evaluating
fairness interventions.  Traditional A/B tests optimise a single metric;
**fairness A/B tests** navigate multi-dimensional optimisation across
demographic groups, asking *"which variant achieves business goals while
maintaining equity?"*

- **FairnessExperiment**: Core A/B testing with intersectional analysis
- **FairnessPowerAnalyzer**: Per-intersection power, SPRT early stopping,
  adaptive sampling
- **ExperimentAnalysis**: Pareto frontier, causal decomposition, temporal
  stability, and automated deployment recommendation

Quick Start::

    >>> from vfairness.operations.experimentation import FairnessExperiment
    >>> exp = FairnessExperiment(
    ...     control_data=df_ctrl,
    ...     treatment_data=df_treat,
    ...     protected_attributes=['gender', 'race'],
    ...     outcome_column='approved',
    ... )
    >>> result = exp.run_full_analysis()
    >>> print(result.heterogeneity_detected)

    >>> from vfairness.operations.experimentation import FairnessPowerAnalyzer
    >>> analyzer = FairnessPowerAnalyzer(exp)
    >>> summary = analyzer.get_power_summary()

    >>> from vfairness.operations.experimentation import ExperimentAnalysis
    >>> analysis = ExperimentAnalysis(result, experiment=exp)
    >>> rec = analysis.decision_recommendation()
"""

# FairnessExperiment: core A/B testing framework
# ExperimentAnalysis: multi-objective & causal inference
from .analysis import (
    CausalDecomposition,
    ExperimentAnalysis,
    ExperimentRecommendation,
    ParetoPoint,
    RecommendationDecision,
    TemporalStabilityResult,
)
from .experiment import (
    DesignType,
    ExperimentConfig,
    ExperimentResult,
    FairnessExperiment,
    IntersectionEffect,
    assign_clusters,
    create_factorial_design,
)

# FairnessPowerAnalyzer: power analysis & sequential testing
from .power import (
    FairnessPowerAnalyzer,
    PowerConfig,
    PowerResult,
    SamplingPlan,
    SequentialTestResult,
    SPRTDecision,
)

__all__ = [
    # Experiment
    "FairnessExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "IntersectionEffect",
    "DesignType",
    "assign_clusters",
    "create_factorial_design",
    # Power
    "FairnessPowerAnalyzer",
    "PowerConfig",
    "PowerResult",
    "SequentialTestResult",
    "SPRTDecision",
    "SamplingPlan",
    # Analysis
    "ExperimentAnalysis",
    "ParetoPoint",
    "CausalDecomposition",
    "TemporalStabilityResult",
    "ExperimentRecommendation",
    "RecommendationDecision",
]
