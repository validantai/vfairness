"""
vfairness Capability Registry

Single source of truth for what vfairness can do. Each entry maps a dispatch_key
(used by the platform's KG and wizard) to its implementation details.

The manifest generator (_manifest.py) reads this registry to produce
vfairness-manifest.json, which the platform consumes to derive coverage status.

To add a new capability:
  1. Add an entry to CAPABILITY_REGISTRY below
  2. Run: python -m vfairness._manifest > vfairness-manifest.json
  3. Re-sync the regenerated manifest into the consuming application
"""

from __future__ import annotations

from typing import TypedDict


class ManifestEntry(TypedDict):
    """Schema for a single capability entry in the registry.

    THE CONTRACT (F20, corrected 2026-09-09): ``name`` is the attribute under
    which the capability is reachable on ``vfairness.<module_path>``, and
    ``kind`` says whether that attribute is a class or a function. That is what
    tests/test_manifest.py::test_capability_is_importable enforces for every
    entry. Membership in ``vfairness.__all__`` is NOT required and NOT checked:
    this comment used to say "must match __all__ export", which 58 of 194
    entries did not satisfy (measured 2026-09-09) while nothing enforced it.
    Widening the top-level export list is a separate public-surface decision.
    """

    name: str  # Attribute name on vfairness.<module_path>; not necessarily in __all__
    dispatch_key: str  # snake_case key used by platform KG (vfairnessFunction)
    kind: str  # 'class' or 'function'
    module_path: str  # Dotted path relative to vfairness (e.g. 'preprocessing.feature_engineering')
    pipeline_stage: (
        str  # 'preprocessing' | 'in_processing' | 'post_processing' | 'evaluation' | 'operations'
    )
    svg_templates: list  # Associated SVG template names (without .svg extension)


# CAPABILITY REGISTRY
# Organized by module, matching the vfairness package structure.
# Only user-facing classes and key functions are registered here.
# Data classes, result types, enums, and base classes are excluded.

CAPABILITY_REGISTRY: dict[str, ManifestEntry] = {
    # PREPROCESSING: Feature Engineering
    "correlation_reduction": {
        "name": "CorrelationReducer",
        "dispatch_key": "correlation_reduction",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["correlation_heatmap", "correlation_matrix"],
    },
    "feature_suppression": {
        "name": "FeatureSuppressor",
        "dispatch_key": "feature_suppression",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["proxy_risk"],
    },
    "residual_transform": {
        "name": "ResidualTransformer",
        "dispatch_key": "residual_transform",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["transformation_comparison"],
    },
    "intersectional_transform": {
        "name": "IntersectionalTransformer",
        "dispatch_key": "intersectional_transform",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["intersectional_analysis"],
    },
    "reweighting": {
        "name": "ReweightingTransformer",
        "dispatch_key": "reweighting",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "disparate_impact_removal": {
        "name": "DisparateImpactRemover",
        "dispatch_key": "disparate_impact_removal",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["transformation_comparison"],
    },
    "label_massaging": {
        "name": "LabelMassager",
        "dispatch_key": "label_massaging",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["transformation_comparison"],
    },
    "resampling": {
        "name": "Resampler",
        "dispatch_key": "resampling",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "fair_representation": {
        "name": "FairRepresentationTransformer",
        "dispatch_key": "fair_representation",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["transformation_comparison"],
    },
    "smote": {
        "name": "SMOTEResampler",
        "dispatch_key": "smote",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "adasyn": {
        "name": "ADASYNResampler",
        "dispatch_key": "adasyn",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "tomek": {
        "name": "TomekResampler",
        "dispatch_key": "tomek",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "propensity_weighting": {
        "name": "PropensityScoreWeighter",
        "dispatch_key": "propensity_weighting",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "inverse_propensity": {
        "name": "InversePropensityWeighter",
        "dispatch_key": "inverse_propensity",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "counterfactual_augment": {
        "name": "CounterfactualAugmenter",
        "dispatch_key": "counterfactual_augment",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "feature_engineering_analysis": {
        "name": "FeatureEngineeringAnalyzer",
        "dispatch_key": "feature_engineering_analysis",
        "kind": "class",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["correlation_heatmap", "correlation_matrix", "proxy_risk"],
    },
    # PREPROCESSING: Bias Detection
    "bias_audit": {
        "name": "BiasDetector",
        "dispatch_key": "bias_audit",
        "kind": "class",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["bias_audit"],
    },
    "detect_historical_patterns": {
        "name": "detect_historical_patterns",
        "dispatch_key": "detect_historical_patterns",
        "kind": "function",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "detect_representation_bias": {
        "name": "detect_representation_bias",
        "dispatch_key": "detect_representation_bias",
        "kind": "function",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "analyze_statistical_disparities": {
        "name": "analyze_statistical_disparities",
        "dispatch_key": "analyze_statistical_disparities",
        "kind": "function",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": [],
    },
    "identify_proxy_variables": {
        "name": "identify_proxy_variables",
        "dispatch_key": "identify_proxy_variables",
        "kind": "function",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["proxy_risk"],
    },
    "compute_feature_correlations": {
        "name": "compute_feature_correlations",
        "dispatch_key": "compute_feature_correlations",
        "kind": "function",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["correlation_heatmap", "correlation_matrix"],
    },
    "find_proxy_chains": {
        "name": "find_proxy_chains",
        "dispatch_key": "find_proxy_chains",
        "kind": "function",
        "module_path": "preprocessing.feature_engineering",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["proxy_risk"],
    },
    # IN-PROCESSING: Loss Functions (Group Fairness)
    "fairness_aware_bce": {
        "name": "FairnessAwareBCELoss",
        "dispatch_key": "fairness_aware_bce",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report", "training_analysis_report"],
    },
    "demographic_parity_loss": {
        "name": "DemographicParityLoss",
        "dispatch_key": "demographic_parity_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "equalized_odds_loss": {
        "name": "EqualizedOddsLoss",
        "dispatch_key": "equalized_odds_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "equal_opportunity_loss": {
        "name": "EqualOpportunityLoss",
        "dispatch_key": "equal_opportunity_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "false_positive_rate_parity_loss": {
        "name": "FalsePositiveRateParityLoss",
        "dispatch_key": "false_positive_rate_parity_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "bounded_group_loss": {
        "name": "BoundedGroupLoss",
        "dispatch_key": "bounded_group_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    # IN-PROCESSING: Loss Functions (Adversarial)
    "adversarial_debiasing": {
        "name": "AdversarialDebiasingLoss",
        "dispatch_key": "adversarial_debiasing",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report", "training_analysis_report"],
    },
    "projected_adversarial": {
        "name": "ProjectedAdversarialLoss",
        "dispatch_key": "projected_adversarial",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "fair_representation_loss": {
        "name": "FairRepresentationLoss",
        "dispatch_key": "fair_representation_loss",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    # IN-PROCESSING: Loss Functions (Counterfactual / Causal)
    "counterfactual_fairness": {
        "name": "CounterfactualFairnessLoss",
        "dispatch_key": "counterfactual_fairness",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "individual_fairness": {
        "name": "IndividualFairnessLoss",
        "dispatch_key": "individual_fairness",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    "causal_fairness": {
        "name": "CausalFairnessLoss",
        "dispatch_key": "causal_fairness",
        "kind": "class",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    # IN-PROCESSING: Constraints
    "demographic_parity_constraint": {
        "name": "DemographicParityConstraint",
        "dispatch_key": "demographic_parity_constraint",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "equalized_odds_constraint": {
        "name": "EqualizedOddsConstraint",
        "dispatch_key": "equalized_odds_constraint",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "equal_opportunity_constraint": {
        "name": "EqualOpportunityConstraint",
        "dispatch_key": "equal_opportunity_constraint",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "false_positive_rate_parity_constraint": {
        "name": "FalsePositiveRateParityConstraint",
        "dispatch_key": "false_positive_rate_parity_constraint",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "bounded_group_loss_constraint": {
        "name": "BoundedGroupLossConstraint",
        "dispatch_key": "bounded_group_loss_constraint",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "exponentiated_gradient": {
        "name": "ExponentiatedGradient",
        "dispatch_key": "exponentiated_gradient",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": ["method_comparison"],
    },
    "constraint_grid_search": {
        "name": "GridSearch",
        "dispatch_key": "constraint_grid_search",
        "kind": "class",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": ["method_comparison"],
    },
    # IN-PROCESSING: Regularizers
    "statistical_parity_regularizer": {
        "name": "StatisticalParityRegularizer",
        "dispatch_key": "statistical_parity_regularizer",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "conditional_independence_regularizer": {
        "name": "ConditionalIndependenceRegularizer",
        "dispatch_key": "conditional_independence_regularizer",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "group_fairness_regularizer": {
        "name": "GroupFairnessRegularizer",
        "dispatch_key": "group_fairness_regularizer",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    # Umbrella dispatch key for the Navigator's "Fairness Regularization" tile.
    # Backed by GroupFairnessRegularizer driven by a torch training loop in the
    # consumer (handle_inprocess), with the regularizer selected per fairness
    # constraint (dp / eo / eop / fpr).
    "fairness_regularization": {
        "name": "GroupFairnessRegularizer",
        "dispatch_key": "fairness_regularization",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": ["method_comparison", "training_report"],
    },
    "hilbert_schmidt_regularizer": {
        "name": "HilbertSchmidtRegularizer",
        "dispatch_key": "hilbert_schmidt_regularizer",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "correlation_penalty": {
        "name": "CorrelationPenalty",
        "dispatch_key": "correlation_penalty",
        "kind": "class",
        "module_path": "in_processing.regularizers",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    # IN-PROCESSING: Calibrators (training-time)
    "temperature_scaling_calibrator": {
        "name": "TemperatureScalingCalibrator",
        "dispatch_key": "temperature_scaling_calibrator",
        "kind": "class",
        "module_path": "in_processing.calibrators",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "platt_scaling_calibrator": {
        "name": "PlattScalingCalibrator",
        "dispatch_key": "platt_scaling_calibrator",
        "kind": "class",
        "module_path": "in_processing.calibrators",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "in_processing_beta_calibrator": {
        "name": "BetaCalibrator",
        "dispatch_key": "in_processing_beta_calibrator",
        "kind": "class",
        "module_path": "in_processing.calibrators",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "focal_calibrator": {
        "name": "FocalCalibrator",
        "dispatch_key": "focal_calibrator",
        "kind": "class",
        "module_path": "in_processing.calibrators",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "trainable_group_calibrator": {
        "name": "TrainableGroupCalibrator",
        "dispatch_key": "trainable_group_calibrator",
        "kind": "class",
        "module_path": "in_processing.calibrators",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    # IN-PROCESSING: Wrappers
    "fair_classifier": {
        "name": "FairClassifier",
        "dispatch_key": "fair_classifier",
        "kind": "class",
        "module_path": "in_processing.wrappers",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report", "method_comparison"],
    },
    "fair_regressor": {
        "name": "FairRegressor",
        "dispatch_key": "fair_regressor",
        "kind": "class",
        "module_path": "in_processing.wrappers",
        "pipeline_stage": "in_processing",
        "svg_templates": ["training_report"],
    },
    # IN-PROCESSING: Analyzer
    "fairness_training_analysis": {
        "name": "FairnessTrainingAnalyzer",
        "dispatch_key": "fairness_training_analysis",
        "kind": "class",
        "module_path": "in_processing",
        "pipeline_stage": "in_processing",
        "svg_templates": [
            "training_report",
            "training_analysis_report",
            "method_comparison",
            "tradeoff_analysis",
        ],
    },
    # POST-PROCESSING: Calibration
    "calibration_analysis": {
        "name": "CalibrationAnalyzer",
        "dispatch_key": "calibration_analysis",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": [
            "calibration_report",
            "reliability_diagram",
            "group_calibration",
            "calibration_disparity",
        ],
    },
    "platt_scaling": {
        "name": "PlattScaling",
        "dispatch_key": "platt_scaling",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reliability_diagram"],
    },
    "isotonic_calibration": {
        "name": "IsotonicCalibrator",
        "dispatch_key": "isotonic_calibration",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reliability_diagram"],
    },
    "beta_calibration": {
        "name": "BetaCalibrator",
        "dispatch_key": "beta_calibration",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reliability_diagram"],
    },
    "temperature_scaling": {
        "name": "TemperatureScaling",
        "dispatch_key": "temperature_scaling",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reliability_diagram"],
    },
    "histogram_binning": {
        "name": "HistogramBinning",
        "dispatch_key": "histogram_binning",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reliability_diagram"],
    },
    "group_calibration": {
        "name": "GroupCalibrator",
        "dispatch_key": "group_calibration",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["group_calibration", "calibration_disparity"],
    },
    "intersectional_calibration": {
        "name": "IntersectionalCalibrator",
        "dispatch_key": "intersectional_calibration",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["group_calibration"],
    },
    "calibration": {
        "name": "CalibrationAnalyzer",
        "dispatch_key": "calibration",
        "kind": "class",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": ["calibration_report", "reliability_diagram", "group_calibration"],
    },
    # POST-PROCESSING: Threshold Optimization
    "threshold_optimization": {
        "name": "ThresholdOptimizer",
        "dispatch_key": "threshold_optimization",
        "kind": "class",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": ["threshold_optimization_report"],
    },
    "group_threshold": {
        "name": "GroupThresholdOptimizer",
        "dispatch_key": "group_threshold",
        "kind": "class",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": ["threshold_optimization_report"],
    },
    "multi_objective_threshold": {
        "name": "MultiObjectiveThresholdOptimizer",
        "dispatch_key": "multi_objective_threshold",
        "kind": "class",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": ["threshold_optimization_report", "pareto_frontier"],
    },
    "threshold_analysis": {
        "name": "ThresholdAnalyzer",
        "dispatch_key": "threshold_analysis",
        "kind": "class",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": ["threshold_optimization_report"],
    },
    # POST-PROCESSING: Reweighting
    "prediction_reweighting": {
        "name": "PredictionReweighter",
        "dispatch_key": "prediction_reweighting",
        "kind": "class",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reweighting_comparison_report"],
    },
    "rejection_option_classification": {
        "name": "RejectionOptionClassifier",
        "dispatch_key": "rejection_option_classification",
        "kind": "class",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reweighting_comparison_report"],
    },
    "calibrated_equalization": {
        "name": "CalibratedEqualizer",
        "dispatch_key": "calibrated_equalization",
        "kind": "class",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reweighting_comparison_report"],
    },
    "distribution_matching": {
        "name": "DistributionMatcher",
        "dispatch_key": "distribution_matching",
        "kind": "class",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reweighting_comparison_report"],
    },
    "reweighting_analysis": {
        "name": "ReweightingAnalyzer",
        "dispatch_key": "reweighting_analysis",
        "kind": "class",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": ["reweighting_comparison_report", "fairness_detailed_report"],
    },
    # EVALUATION: Unified Analyzer
    "fairness_analysis": {
        "name": "FairnessAnalyzer",
        "dispatch_key": "fairness_analysis",
        "kind": "class",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [
            "fairness_report",
            "radar_chart",
            "disparity_heatmap",
            "metrics_bar_chart",
            "group_comparison",
            "effect_sizes",
            "confidence_intervals",
        ],
    },
    # EVALUATION: Metric Functions
    "demographic_parity_difference": {
        "name": "demographic_parity_difference",
        "dispatch_key": "demographic_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "equalized_odds_difference": {
        "name": "equalized_odds_difference",
        "dispatch_key": "equalized_odds_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "equal_opportunity_difference": {
        "name": "equal_opportunity_difference",
        "dispatch_key": "equal_opportunity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "predictive_parity_difference": {
        "name": "predictive_parity_difference",
        "dispatch_key": "predictive_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "fpr_parity_difference": {
        "name": "fpr_parity_difference",
        "dispatch_key": "fpr_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "fnr_parity_difference": {
        "name": "fnr_parity_difference",
        "dispatch_key": "fnr_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "accuracy_parity_difference": {
        "name": "accuracy_parity_difference",
        "dispatch_key": "accuracy_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "worst_group_accuracy": {
        "name": "worst_group_accuracy",
        "dispatch_key": "worst_group_accuracy",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "disparate_impact_ratio": {
        "name": "disparate_impact_ratio",
        "dispatch_key": "disparate_impact_ratio",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "negative_predictive_value_difference": {
        "name": "negative_predictive_value_difference",
        "dispatch_key": "negative_predictive_value_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "conditional_demographic_disparity": {
        "name": "conditional_demographic_disparity",
        "dispatch_key": "conditional_demographic_disparity",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "auroc_parity": {
        "name": "auroc_parity",
        "dispatch_key": "auroc_parity",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "net_benefit_parity": {
        "name": "net_benefit_parity",
        "dispatch_key": "net_benefit_parity",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "conditional_adverse_impact": {
        "name": "conditional_adverse_impact",
        "dispatch_key": "conditional_adverse_impact",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "pricing_disparity": {
        "name": "pricing_disparity",
        "dispatch_key": "pricing_disparity",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "multivariate_proxy_leakage": {
        "name": "multivariate_proxy_leakage",
        "dispatch_key": "multivariate_proxy_leakage",
        "kind": "function",
        "module_path": "preprocessing.bias_detection",
        "pipeline_stage": "preprocessing",
        "svg_templates": ["metrics_bar_chart"],
    },
    "disparate_impact_ratio_with_ci": {
        "name": "disparate_impact_ratio_with_ci",
        "dispatch_key": "disparate_impact_ratio_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "fpr_parity_difference_with_ci": {
        "name": "fpr_parity_difference_with_ci",
        "dispatch_key": "fpr_parity_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "negative_predictive_value_difference_with_ci": {
        "name": "negative_predictive_value_difference_with_ci",
        "dispatch_key": "negative_predictive_value_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "conditional_demographic_disparity_with_ci": {
        "name": "conditional_demographic_disparity_with_ci",
        "dispatch_key": "conditional_demographic_disparity_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "pricing_disparity_with_ci": {
        "name": "pricing_disparity_with_ci",
        "dispatch_key": "pricing_disparity_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["metrics_bar_chart"],
    },
    "integrated_calibration_index_with_ci": {
        "name": "integrated_calibration_index_with_ci",
        "dispatch_key": "integrated_calibration_index_with_ci",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "multicalibration_with_ci": {
        "name": "multicalibration_with_ci",
        "dispatch_key": "multicalibration_with_ci",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_disparity"],
    },
    "expected_calibration_error": {
        "name": "expected_calibration_error",
        "dispatch_key": "expected_calibration_error",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "calibration_disparity": {
        "name": "calibration_disparity",
        "dispatch_key": "calibration_disparity",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_disparity"],
    },
    "calibration_in_the_large": {
        "name": "calibration_in_the_large",
        "dispatch_key": "calibration_in_the_large",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "calibration_slope": {
        "name": "calibration_slope",
        "dispatch_key": "calibration_slope",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "integrated_calibration_index": {
        "name": "integrated_calibration_index",
        "dispatch_key": "integrated_calibration_index",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "multicalibration": {
        "name": "multicalibration",
        "dispatch_key": "multicalibration",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_disparity"],
    },
    "exposure_parity_difference": {
        "name": "exposure_parity_difference",
        "dispatch_key": "exposure_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["ranking_fairness"],
    },
    "mae_parity_difference": {
        "name": "mae_parity_difference",
        "dispatch_key": "mae_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["regression_fairness"],
    },
    # EVALUATION: Explainability
    "fair_explainer": {
        "name": "FairExplAIner",
        "dispatch_key": "fair_explainer",
        "kind": "class",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "fairness_explainer": {
        "name": "FairnessExplainer",
        "dispatch_key": "fairness_explainer",
        "kind": "class",
        "module_path": "explainer",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "feature_attribution_explainer": {
        "name": "FeatureAttributionExplainer",
        "dispatch_key": "feature_attribution_explainer",
        "kind": "class",
        "module_path": "evaluation.vfairness_metrics.attribution",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "embedding_bias_detector": {
        "name": "EmbeddingBiasDetector",
        "dispatch_key": "embedding_bias_detector",
        "kind": "class",
        "module_path": "llm.embedding_bias",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "text_fairness_analyzer": {
        "name": "TextFairnessAnalyzer",
        "dispatch_key": "text_fairness_analyzer",
        "kind": "class",
        "module_path": "llm.text_fairness",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "output_analyzer": {
        "name": "OutputAnalyzer",
        "dispatch_key": "output_analyzer",
        "kind": "class",
        "module_path": "llm.output_analysis",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "counterfactual_tester": {
        "name": "CounterfactualTester",
        "dispatch_key": "counterfactual_tester",
        "kind": "class",
        "module_path": "llm.counterfactual",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "causal_fairness_graph": {
        "name": "CausalFairnessGraph",
        "dispatch_key": "causal_fairness_graph",
        "kind": "class",
        "module_path": "operations.causal.graph",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "counterfactual_fairness_metric": {
        "name": "counterfactual_fairness",
        "dispatch_key": "counterfactual_fairness_metric",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.counterfactual_metric",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "fairness_decomposition": {
        "name": "fairness_decomposition",
        "dispatch_key": "fairness_decomposition",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.fairness_decomposition",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "recourse_generator": {
        "name": "generate_recourse",
        "dispatch_key": "recourse_generator",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.recourse",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "explanation_diagnostics": {
        "name": "diagnose_local_attribution",
        "dispatch_key": "explanation_diagnostics",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.explanation_diagnostics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # EVALUATION: Auto-Discovery
    "auto_discovery": {
        "name": "scan_fairness_violations",
        "dispatch_key": "auto_discovery",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["auto_discovery"],
    },
    "detect_protected_attributes": {
        "name": "detect_protected_attributes",
        "dispatch_key": "detect_protected_attributes",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # EVALUATION: Robustness & Statistical Testing
    "robustness_testing": {
        "name": "comprehensive_fairness_test",
        "dispatch_key": "robustness_testing",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["robustness_testing"],
    },
    "sensitivity_analysis": {
        "name": "sensitivity_analysis",
        "dispatch_key": "sensitivity_analysis",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "subgroup_robustness_audit": {
        "name": "subgroup_robustness_audit",
        "dispatch_key": "subgroup_robustness_audit",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # OPERATIONS: CI/CD
    "data_bias_validation": {
        "name": "DataBiasValidator",
        "dispatch_key": "data_bias_validation",
        "kind": "class",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        "svg_templates": ["data_validation", "cicd_pipeline"],
    },
    "fairness_gate": {
        "name": "ModelFairnessGate",
        "dispatch_key": "fairness_gate",
        "kind": "class",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        "svg_templates": ["hierarchical_gate", "cicd_pipeline"],
    },
    "fairness_test_suite": {
        "name": "FairnessTestSuite",
        "dispatch_key": "fairness_test_suite",
        "kind": "class",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        # workflow_overview is the dev-workflow integration dashboard rendered
        # by rendering.adapters_workflow.workflow_overview_to_svg (pytest
        # plugin / pre-commit / tracking integrations around this suite);
        # registering it here puts it under the render smoke suite.
        "svg_templates": ["cicd_pipeline", "workflow_overview"],
    },
    "fairness_report_card": {
        "name": "FairnessReportCard",
        "dispatch_key": "fairness_report_card",
        "kind": "class",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        "svg_templates": ["report_card"],
    },
    "bias_monitor": {
        "name": "BiasMonitor",
        "dispatch_key": "bias_monitor",
        "kind": "class",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        "svg_templates": ["monitoring_dashboard"],
    },
    # OPERATIONS: Monitoring (Unit 1)
    "fairness_monitor": {
        "name": "FairnessMonitor",
        "dispatch_key": "fairness_monitor",
        "kind": "class",
        "module_path": "operations.monitoring",
        "pipeline_stage": "operations",
        "svg_templates": ["monitoring_dashboard"],
    },
    "temporal_fairness_analysis": {
        "name": "TemporalFairnessAnalyzer",
        "dispatch_key": "temporal_fairness_analysis",
        "kind": "class",
        "module_path": "operations.monitoring",
        "pipeline_stage": "operations",
        "svg_templates": ["temporal_analysis"],
    },
    # OPERATIONS: Monitoring (Unit 2)
    "drift_detection": {
        "name": "FairnessDriftDetector",
        "dispatch_key": "drift_detection",
        "kind": "class",
        "module_path": "operations.monitoring",
        "pipeline_stage": "operations",
        "svg_templates": ["drift_report"],
    },
    "adaptive_threshold_management": {
        "name": "AdaptiveThresholdManager",
        "dispatch_key": "adaptive_threshold_management",
        "kind": "class",
        "module_path": "operations.monitoring",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "alert_prioritization": {
        "name": "FairnessAlertPrioritizer",
        "dispatch_key": "alert_prioritization",
        "kind": "class",
        "module_path": "operations.monitoring",
        "pipeline_stage": "operations",
        "svg_templates": ["alert_timeline"],
    },
    # OPERATIONS: Reporting (Unit 3)
    "metrics_store": {
        "name": "MetricsStore",
        "dispatch_key": "metrics_store",
        "kind": "class",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "fairness_dashboard": {
        "name": "FairnessDashboard",
        "dispatch_key": "fairness_dashboard",
        "kind": "class",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": ["reporting_dashboard"],
    },
    "report_generation": {
        "name": "ReportGenerator",
        "dispatch_key": "report_generation",
        "kind": "class",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": ["reporting_dashboard"],
    },
    "interactive_dashboard": {
        "name": "InteractiveDashboard",
        "dispatch_key": "interactive_dashboard",
        "kind": "class",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    # OPERATIONS: Experimentation (Unit 4)
    "fairness_experiment": {
        "name": "FairnessExperiment",
        "dispatch_key": "fairness_experiment",
        "kind": "class",
        "module_path": "operations.experimentation",
        "pipeline_stage": "operations",
        "svg_templates": ["experiment_results"],
    },
    "power_analysis": {
        "name": "FairnessPowerAnalyzer",
        "dispatch_key": "power_analysis",
        "kind": "class",
        "module_path": "operations.experimentation",
        "pipeline_stage": "operations",
        "svg_templates": ["power_analysis"],
    },
    "experiment_analysis": {
        "name": "ExperimentAnalysis",
        "dispatch_key": "experiment_analysis",
        "kind": "class",
        "module_path": "operations.experimentation",
        "pipeline_stage": "operations",
        "svg_templates": ["experiment_recommendation", "causal_decomposition"],
    },
    # OPERATIONS: Compliance Reporting
    "generate_annex_iv": {
        "name": "generate_annex_iv_data",
        "dispatch_key": "generate_annex_iv",
        "kind": "function",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "generate_model_card": {
        "name": "generate_model_card",
        "dispatch_key": "generate_model_card",
        "kind": "function",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "generate_dpia_sections": {
        "name": "generate_dpia_sections",
        "dispatch_key": "generate_dpia_sections",
        "kind": "function",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "generate_risk_register": {
        "name": "generate_risk_register_from_audit",
        "dispatch_key": "generate_risk_register",
        "kind": "function",
        "module_path": "operations.reporting",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    # ADDITIONAL FUNCTIONS (referenced by ontology enrichment entries)
    # Evaluation: Classification metrics
    "demographic_parity_ratio": {
        "name": "demographic_parity_ratio",
        "dispatch_key": "demographic_parity_ratio",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "predictive_parity_difference_with_ci": {
        "name": "predictive_parity_difference_with_ci",
        "dispatch_key": "predictive_parity_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "demographic_parity_difference_with_ci": {
        "name": "demographic_parity_difference_with_ci",
        "dispatch_key": "demographic_parity_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "equalized_odds_difference_with_ci": {
        "name": "equalized_odds_difference_with_ci",
        "dispatch_key": "equalized_odds_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "equal_opportunity_difference_with_ci": {
        "name": "equal_opportunity_difference_with_ci",
        "dispatch_key": "equal_opportunity_difference_with_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "compute_effect_sizes": {
        "name": "compute_effect_sizes",
        "dispatch_key": "compute_effect_sizes",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["effect_sizes"],
    },
    # Evaluation: Regression metrics
    "rmse_parity_difference": {
        "name": "rmse_parity_difference",
        "dispatch_key": "rmse_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["regression_fairness"],
    },
    "mean_prediction_difference": {
        "name": "mean_prediction_difference",
        "dispatch_key": "mean_prediction_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["regression_fairness"],
    },
    "r2_parity_difference": {
        "name": "r2_parity_difference",
        "dispatch_key": "r2_parity_difference",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.regression",
        "pipeline_stage": "evaluation",
        "svg_templates": ["regression_fairness"],
    },
    "residual_bias": {
        "name": "residual_bias",
        "dispatch_key": "residual_bias",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics.regression",
        "pipeline_stage": "evaluation",
        "svg_templates": ["regression_fairness"],
    },
    # Evaluation: Ranking metrics
    "attention_weighted_rank_fairness": {
        "name": "attention_weighted_rank_fairness",
        "dispatch_key": "attention_weighted_rank_fairness",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["ranking_fairness"],
    },
    "normalized_discounted_kl_divergence": {
        "name": "normalized_discounted_kl_divergence",
        "dispatch_key": "normalized_discounted_kl_divergence",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["ranking_fairness"],
    },
    "exposure_parity_ratio": {
        "name": "exposure_parity_ratio",
        "dispatch_key": "exposure_parity_ratio",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["ranking_fairness"],
    },
    # Evaluation: Intersectional
    "intersectional_disparity_analysis": {
        "name": "intersectional_disparity_analysis",
        "dispatch_key": "intersectional_disparity_analysis",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        # intersectional_disparity is the ranked-bar view rendered by
        # rendering.adapters_feature_engineering.intersectional_disparity_to_svg;
        # registering it here puts it under the render smoke suite.
        "svg_templates": ["intersectional_analysis", "intersectional_disparity"],
    },
    "identify_privileged_groups": {
        "name": "identify_privileged_groups",
        "dispatch_key": "identify_privileged_groups",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "discover_intersectional_groups": {
        "name": "discover_intersectional_groups",
        "dispatch_key": "discover_intersectional_groups",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # Evaluation: Reports
    "classification_fairness_report": {
        "name": "classification_fairness_report",
        "dispatch_key": "classification_fairness_report",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["fairness_report"],
    },
    # Evaluation: Robustness
    "stress_test_fairness": {
        "name": "stress_test_fairness",
        "dispatch_key": "stress_test_fairness",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "permutation_test": {
        "name": "permutation_test",
        "dispatch_key": "permutation_test",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # Evaluation: Statistics
    "bootstrap_ci": {
        "name": "bootstrap_ci",
        "dispatch_key": "bootstrap_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": ["confidence_intervals"],
    },
    "bayesian_proportion_ci": {
        "name": "bayesian_proportion_ci",
        "dispatch_key": "bayesian_proportion_ci",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "bonferroni_correction": {
        "name": "bonferroni_correction",
        "dispatch_key": "bonferroni_correction",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "benjamini_hochberg_correction": {
        "name": "benjamini_hochberg_correction",
        "dispatch_key": "benjamini_hochberg_correction",
        "kind": "function",
        "module_path": "evaluation.vfairness_metrics",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # Calibration metrics
    "maximum_calibration_error": {
        "name": "maximum_calibration_error",
        "dispatch_key": "maximum_calibration_error",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_report"],
    },
    "group_calibration_metrics": {
        "name": "group_calibration_metrics",
        "dispatch_key": "group_calibration_metrics",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["group_calibration"],
    },
    "calibration_curve": {
        "name": "calibration_curve",
        "dispatch_key": "calibration_curve",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["reliability_diagram"],
    },
    "brier_score": {
        "name": "brier_score",
        "dispatch_key": "brier_score",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "brier_score_decomposition": {
        "name": "brier_score_decomposition",
        "dispatch_key": "brier_score_decomposition",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "calibration_difference": {
        "name": "calibration_disparity",
        "dispatch_key": "calibration_difference",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "evaluation",
        "svg_templates": ["calibration_disparity"],
    },
    # Factory functions
    "create_calibrator": {
        "name": "create_calibrator",
        "dispatch_key": "create_calibrator",
        "kind": "function",
        "module_path": "post_processing.calibration",
        "pipeline_stage": "post_processing",
        "svg_templates": [],
    },
    "create_fairness_loss": {
        "name": "create_fairness_loss",
        "dispatch_key": "create_fairness_loss",
        "kind": "function",
        "module_path": "in_processing.loss_functions",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "create_constraint": {
        "name": "create_constraint",
        "dispatch_key": "create_constraint",
        "kind": "function",
        "module_path": "in_processing.constraints",
        "pipeline_stage": "in_processing",
        "svg_templates": [],
    },
    "create_reweighter": {
        "name": "create_reweighter",
        "dispatch_key": "create_reweighter",
        "kind": "function",
        "module_path": "post_processing.reweighting",
        "pipeline_stage": "post_processing",
        "svg_templates": [],
    },
    "create_threshold_optimizer": {
        "name": "create_threshold_optimizer",
        "dispatch_key": "create_threshold_optimizer",
        "kind": "function",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": [],
    },
    "find_feasible_thresholds": {
        "name": "find_feasible_thresholds",
        "dispatch_key": "find_feasible_thresholds",
        "kind": "function",
        "module_path": "post_processing.threshold_optimization",
        "pipeline_stage": "post_processing",
        "svg_templates": [],
    },
    # CI/CD hooks
    "check_model_card": {
        "name": "check_model_card",
        "dispatch_key": "check_model_card",
        "kind": "function",
        "module_path": "operations.cicd",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    # XAI EXPLAINERS (vfairness.xai)
    # The method-routed XAI engine: served end-to-end by the consumer task
    # lane `vfairness_xai_explain` (SHAP-family methods run out-of-process in
    # the XAI sidecar venv where `shap` is installed; adapters defer their
    # optional imports to __init__). Registered 2026-08-23 after the coverage
    # matrix was found blind to the whole package: the lane shipped without
    # ledger entries, so the platform reported Shapley/SHAP as "Not Covered"
    # while exact TreeSHAP ran in production. Source of truth for what is
    # implemented: vfairness.xai.explainers.registry._REGISTRY; the
    # completeness test (tests/test_registry_completeness.py) keeps the two
    # registries in sync, so an adapter can no longer ship unregistered.
    "xai_explainer_routing": {
        "name": "route_explainer",
        "dispatch_key": "xai_explainer_routing",
        "kind": "function",
        "module_path": "xai.explainers.router",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "tree_shap": {
        "name": "TreeShapExplainer",
        "dispatch_key": "tree_shap",
        "kind": "class",
        "module_path": "xai.explainers.shap_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "linear_explainer": {
        "name": "LinearShapExplainer",
        "dispatch_key": "linear_explainer",
        "kind": "class",
        "module_path": "xai.explainers.shap_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "kernel_shap": {
        "name": "KernelShapExplainer",
        "dispatch_key": "kernel_shap",
        "kind": "class",
        "module_path": "xai.explainers.shap_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "lime": {
        "name": "LimeExplainer",
        "dispatch_key": "lime",
        "kind": "class",
        "module_path": "xai.explainers.lime_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "dice_counterfactuals": {
        "name": "DiceCounterfactualExplainer",
        "dispatch_key": "dice_counterfactuals",
        "kind": "class",
        "module_path": "xai.explainers.dice_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "anchors": {
        "name": "AnchorsExplainer",
        "dispatch_key": "anchors",
        "kind": "class",
        "module_path": "xai.explainers.anchors_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "integrated_gradients": {
        "name": "IntegratedGradientsExplainer",
        "dispatch_key": "integrated_gradients",
        "kind": "class",
        "module_path": "xai.explainers.ig_adapter",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # XAI DECOMPOSITION (vfairness.xai.decomposition): the key functions behind the
    # explainers, the per-feature fairness decomposition and proxy-feature detection.
    # The explainer classes above were registered; these functions ship alongside
    # them and were previously uncounted.
    "lundberg_decomposition": {
        "name": "lundberg_fairness_decomposition",
        "dispatch_key": "lundberg_decomposition",
        "kind": "function",
        "module_path": "xai.decomposition",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "proxy_feature_score": {
        "name": "proxy_score",
        "dispatch_key": "proxy_feature_score",
        "kind": "function",
        "module_path": "xai.decomposition",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # VISION: representation-fairness math (vfairness.vision). Pure-numpy metrics,
    # verified on real/synthetic fixtures. The FairFace demographic classifier
    # (classify_face_demographics) is sidecar-gated and stays in the deferral
    # ledger below, not here.
    "representation_skew": {
        "name": "skew",
        "dispatch_key": "representation_skew",
        "kind": "function",
        "module_path": "vision",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "representation_ndkl": {
        "name": "ndkl",
        "dispatch_key": "representation_ndkl",
        "kind": "function",
        "module_path": "vision",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "representation_bias_amplification": {
        "name": "bias_amplification",
        "dispatch_key": "representation_bias_amplification",
        "kind": "function",
        "module_path": "vision",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "representation_severity": {
        "name": "representation_severity",
        "dispatch_key": "representation_severity",
        "kind": "function",
        "module_path": "vision",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # AGENTS: agentic-system fairness
    #
    # LF-54, 2026-09-10. The whole `agents` package had ZERO registry entries,
    # so the platform's coverage matrix derived "Not Covered" for six shipped,
    # dispatched capabilities. That is the xai incident this ledger was created
    # after, in a different package: the registry stayed hand-curated, and
    # hand-curated turned out to mean nobody curated these.
    "action_bias_analyzer": {
        "name": "ActionBiasAnalyzer",
        "dispatch_key": "action_bias_analyzer",
        "kind": "class",
        "module_path": "agents.action_bias",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "correspondence_tester": {
        "name": "CorrespondenceTester",
        "dispatch_key": "correspondence_tester",
        "kind": "class",
        "module_path": "agents.correspondence",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "pipeline_tracker": {
        "name": "PipelineTracker",
        "dispatch_key": "pipeline_tracker",
        "kind": "class",
        "module_path": "agents.pipeline_tracker",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "rag_bias_analyzer": {
        "name": "RAGBiasAnalyzer",
        "dispatch_key": "rag_bias_analyzer",
        "kind": "class",
        "module_path": "agents.rag_bias",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # NOT the same capability as `temporal_fairness_analysis`
    # (operations.monitoring.TemporalFairnessAnalyzer), which watches a metric
    # over time. This tracks an AGENT's behaviour across a trajectory.
    "temporal_tracker": {
        "name": "TemporalTracker",
        "dispatch_key": "temporal_tracker",
        "kind": "class",
        "module_path": "agents.temporal",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "tool_bias_auditor": {
        "name": "ToolBiasAuditor",
        "dispatch_key": "tool_bias_auditor",
        "kind": "class",
        "module_path": "agents.tool_bias",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # LLM: analyzers and runners that were reachable but unregistered (LF-54)
    "benchmark_runner": {
        "name": "BenchmarkRunner",
        "dispatch_key": "benchmark_runner",
        "kind": "class",
        "module_path": "llm.benchmarks",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "cot_faithfulness_analyzer": {
        "name": "CoTFaithfulnessAnalyzer",
        "dispatch_key": "cot_faithfulness_analyzer",
        "kind": "class",
        "module_path": "llm.cot_faithfulness",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "decoding_trust_runner": {
        "name": "DecodingTrustRunner",
        "dispatch_key": "decoding_trust_runner",
        "kind": "class",
        "module_path": "llm.decodingtrust",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # Deliberately NOT `intersectional_analyzer`: `intersectional_transform`
    # (a preprocessing transformer) and `intersectional_disparity_analysis`
    # (an evaluation metric) already exist and are different capabilities. The
    # prefix keeps the coverage matrix from reading three rows as one.
    "llm_intersectional_analyzer": {
        "name": "IntersectionalAnalyzer",
        "dispatch_key": "llm_intersectional_analyzer",
        "kind": "class",
        "module_path": "llm.intersectional",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "non_determinism_analyzer": {
        "name": "NonDeterminismAnalyzer",
        "dispatch_key": "non_determinism_analyzer",
        "kind": "class",
        "module_path": "llm.nondeterminism",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # Registered because it answers a question with a VERDICT, which is the line
    # this ledger draws: an LLM grading itself is a validity failure a coverage
    # matrix should be able to show as checked (LF-03).
    "judge_is_subject": {
        "name": "judge_is_subject",
        "dispatch_key": "judge_is_subject",
        "kind": "function",
        "module_path": "llm.scorers",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # A measurement: the per-metric floor below which a difference is this
    # system's own run-to-run noise rather than a finding.
    "noise_floor_from_runs": {
        "name": "noise_floor_from_runs",
        "dispatch_key": "noise_floor_from_runs",
        "kind": "function",
        "module_path": "llm.nondeterminism",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    # REGISTERED rather than deferred, 2026-09-10. It sat in the deferral ledger
    # reading "implemented and used by the causal task lane; registration
    # decision pending (flagged 2026-08-23)". Implemented and used is exactly
    # the condition for registering: the ledger is for work that is gated or
    # deliberately out, and "we have not decided" is neither. Deciding it is
    # cheaper than carrying it, and while it sat there the coverage matrix
    # reported a shipped, dispatched capability as absent.
    "decompose_mediation": {
        "name": "decompose_mediation",
        "dispatch_key": "decompose_mediation",
        "kind": "function",
        "module_path": "operations.causal",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    # MULTI-AGENT: fairness of agent-to-agent systems
    #
    # 2026-09-10, found by the LF lane's adversarial pass. A SECOND whole package
    # with zero registry entries, after `agents` earlier the same day: seven
    # implemented capabilities across ~2,500 lines, every one reported by the
    # platform's coverage matrix as Not Covered.
    #
    # The gate written that morning to stop exactly this had
    # ACCOUNTED_PACKAGES = ("llm", "agents"), hardcoded, and the table in that
    # file listing what the gate does NOT cover was hand-enumerated and named
    # eleven of the fifteen packages on disk. So the gate missed it and the
    # gate's own declaration of its limits missed it the same way. A
    # hand-written list cannot report what its author did not think of, and it
    # fails in the direction of looking complete.
    "adversarial_collusion_detector": {
        "name": "AdversarialCollusionDetector",
        "dispatch_key": "adversarial_collusion_detector",
        "kind": "class",
        "module_path": "multi_agent.collusion",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "compositionality_analyzer": {
        "name": "CompositionalityAnalyzer",
        "dispatch_key": "compositionality_analyzer",
        "kind": "class",
        "module_path": "multi_agent.compositionality",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "delegation_routing_auditor": {
        "name": "DelegationRoutingAuditor",
        "dispatch_key": "delegation_routing_auditor",
        "kind": "class",
        "module_path": "multi_agent.delegation",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "emergent_bias_detector": {
        "name": "EmergentBiasDetector",
        "dispatch_key": "emergent_bias_detector",
        "kind": "class",
        "module_path": "multi_agent.emergent",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "groupthink_detector": {
        "name": "GroupthinkDetector",
        "dispatch_key": "groupthink_detector",
        "kind": "class",
        "module_path": "multi_agent.groupthink",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
    "multi_agent_run_harness": {
        "name": "MultiAgentRunHarness",
        "dispatch_key": "multi_agent_run_harness",
        "kind": "class",
        "module_path": "multi_agent.harness",
        "pipeline_stage": "operations",
        "svg_templates": [],
    },
    "negotiation_fairness_tracker": {
        "name": "NegotiationFairnessTracker",
        "dispatch_key": "negotiation_fairness_tracker",
        "kind": "class",
        "module_path": "multi_agent.negotiation",
        "pipeline_stage": "evaluation",
        "svg_templates": [],
    },
}


# DELIBERATELY NOT REGISTERED
# The registry is a curated ledger, not an inventory: an entry asserts the
# capability is implemented and reachable, so unfinished or gated work stays
# out. But staying out must be a RECORDED decision, never an omission (the
# xai package was silently absent for months and the platform reported its
# shipping SHAP explainers as "Not Covered"). Every deliberately unregistered
# capability goes here with its gating condition;
# tests/test_registry_completeness.py fails when something implemented is
# missing from BOTH this ledger and CAPABILITY_REGISTRY.
DEFERRED_FROM_CAPABILITY_REGISTRY: dict[str, str] = {
    "shap.DeepExplainer": "GATED: the adapter is NOT IMPLEMENTED (listed in xai.explainers.registry._NOT_YET_IMPLEMENTED), so this IS a gap and the coverage matrix should show it as one. Unblocked by: writing the DeepExplainer adapter. Phase 2 backlog.",
    "shap.GradientExplainer": "GATED: the adapter is NOT IMPLEMENTED (listed in xai.explainers.registry._NOT_YET_IMPLEMENTED), so this IS a gap and the coverage matrix should show it as one. Unblocked by: writing the GradientExplainer adapter. Phase 2 backlog.",
    "sp-shap": "GATED: SP-SHAP submodular pick is NOT IMPLEMENTED, so this IS a gap and the coverage matrix should show it as one. Unblocked by: implementing the submodular pick (#P1-11).",
    "validity.groundedness": "GATED: the VG-* fail-closed judge ladder is implemented but its grades are not yet validated against anything, so registering it would assert a coverage this cannot support. Unblocked by: the M1 agreement figures plus the gold set feeding a grade (decision 2026-08).",
    "vision.classify_face_demographics": "GATED: FairFace/CLIP demographic classification is gated on the Python-3.9 vision sidecar and fails closed without it, so it is not standalone-runnable in the shipped package; the representation-fairness math (skew/ndkl/bias_amplification/representation_severity) is registered, this classifier stays out until the sidecar ships as a first-class dependency. Unblocked by: shipping the vision sidecar as a first-class dependency (flagged 2026-08-23)",
    # LF-54, 2026-09-10. Every reason below is written for a PLATFORM reader,
    # because the coverage matrix is where these are read and a deferral without
    # an actionable reason is indistinguishable there from a gap. Each says which
    # of the two it is: NOT A GAP (covered elsewhere, or not an assessment), or
    # GATED (a gap, with the condition that closes it).
    "llm.api_proxy.LLMApiProxy": "NOT A GAP, and not an assessment capability: this is the HTTP transport the llm probes call an endpoint through. It measures no fairness property and has no verdict to contribute to a coverage matrix. Registering it would add a row that can never be 'assessed'. Its behaviour is exercised through the capabilities that use it (output_analyzer, counterfactual_tester, benchmark_runner).",
    "llm.scorers.LLMJudgeScorer": "GATED, same gate as validity.groundedness: an LLM judging an LLM produces a grade nothing has validated. It stays out until M1 + the gold set can establish what its scores are worth, exactly as the validity axis does. Until then a run using it must read .available and treat a score as could-not-check, not as a measurement. Unblocked by: gold set + M1 agreement figures (decision 2026-08).",
    "llm.scorers.keyword_scorers": "NOT A GAP: the eight keyword scorers (Framing, Helpfulness, InformationQuality, KeywordRegard, KeywordSentiment, KeywordToxicity, Refusal, Representation, Stereotype) are the METRICS that the registered `output_analyzer` computes, not separately dispatched capabilities. Registering them would put eight rows in the coverage matrix for one capability and make LLM output analysis look eight times broader than it is. Coverage for them IS the output_analyzer row.",
    "llm.scorers.TextScorer": "NOT A GAP: abstract base class defining the scorer interface. The registry's stated convention excludes base classes, result types and enums; this entry exists so the exclusion is a recorded decision rather than an omission.",
    "llm.model_identity": "NOT A GAP, and not a fairness measurement: ModelIdentity / IdentityComparison and their helpers record WHICH model answered, so a statistic can say what system it describes. They are provenance carried alongside a measurement, not a capability that produces a verdict. Read them on any llm result; there is nothing separate to dispatch.",
    "multi_agent.result_types": "NOT A GAP: CollusionResult, CompositionalityResult, DelegationResult, EmergentBiasResult, GroupthinkResult, HarnessTrace and NegotiationResult are the result types returned by the seven registered multi_agent capabilities. Excluded by the registry's stated convention on data classes, recorded here so it is a decision rather than the omission the whole package was until 2026-09-10.",
    "agents.result_types": "NOT A GAP: ActionBiasResult, CorrespondenceResult, RAGBiasResult, StageResult, ToolBiasResult and TrajectoryResult are the result types returned by the six registered agents capabilities. Excluded by the registry's stated convention on data classes, recorded here so it is a decision.",
    "llm.scorers.scorer_status": "NOT A GAP, and not an assessment: it reports WHICH scorers are active and at what quality tier, so a reader can tell whether a number came from a keyword lexicon or a judge. It produces no fairness verdict of its own, so a coverage-matrix row for it could never be 'assessed'. Same reasoning as llm.api_proxy.LLMApiProxy. Its information surfaces on the capabilities that use those scorers; call it directly before trusting any llm score.",
    "llm.result_types": "NOT A GAP: BenchmarkResult, CoTFaithfulnessResult, CounterfactualResult, FaithfulnessReport, IntersectionalResult, IntersectionalGroup, OutputAnalysisResult, NoiseProfile, RunMetadata and SerializableMixin are result and support types for the registered llm capabilities. Excluded by the same convention, recorded here so it is a decision.",
}


# WHICH SYMBOLS EACH GROUPED DEFERRAL COVERS
#
# LF-54, 2026-09-10. Some deferrals cover a family rather than one symbol
# ("the eight keyword scorers"), because eight near-identical rows in a coverage
# matrix is worse for a reader than one row that says what the family is. But a
# prose reason cannot be checked, so the membership lives here as data and
# tests/test_registry_completeness.py reads it.
#
# The consequence that matters: adding a NEW scorer or result type does NOT
# quietly inherit an existing deferral. It is absent from every ledger, the
# completeness gate fails, and somebody decides. That is the whole point of the
# ledger, and without this mapping the grouped entries would have been a hole in
# it big enough to drive the next package through.
DEFERRAL_COVERS: dict[str, tuple[str, ...]] = {
    "llm.api_proxy.LLMApiProxy": ("LLMApiProxy",),
    "llm.scorers.LLMJudgeScorer": ("LLMJudgeScorer",),
    "llm.scorers.TextScorer": ("TextScorer",),
    "llm.scorers.keyword_scorers": (
        "FramingScorer",
        "HelpfulnessScorer",
        "InformationQualityScorer",
        "KeywordRegardScorer",
        "KeywordSentimentScorer",
        "KeywordToxicityScorer",
        "RefusalScorer",
        "RepresentationScorer",
        "StereotypeScorer",
    ),
    "llm.model_identity": (
        "ModelIdentity",
        "IdentityComparison",
        "compare_model_identity",
        "describe_model_identity",
        "identity_from_run",
    ),
    "llm.scorers.scorer_status": ("scorer_status",),
    "llm.result_types": (
        "BenchmarkResult",
        "CoTFaithfulnessResult",
        "CounterfactualResult",
        "FaithfulnessReport",
        "IntersectionalResult",
        "IntersectionalGroup",
        "OutputAnalysisResult",
        "NoiseProfile",
        "RunMetadata",
        "SerializableMixin",
    ),
    "multi_agent.result_types": (
        "CollusionResult",
        "CompositionalityResult",
        "DelegationResult",
        "EmergentBiasResult",
        "GroupthinkResult",
        "HarnessTrace",
        "NegotiationResult",
    ),
    "agents.result_types": (
        "ActionBiasResult",
        "CorrespondenceResult",
        "RAGBiasResult",
        "StageResult",
        "ToolBiasResult",
        "TrajectoryResult",
    ),
}
