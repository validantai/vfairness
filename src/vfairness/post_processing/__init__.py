"""
vfairness.post_processing - Prediction-Time Interventions Module

This module contains tools for post-hoc fairness adjustments:

Submodules:
    - calibration: Probability calibration methods and analysis
    - threshold_optimization: Group-specific threshold tuning
    - reweighting: Output reweighting/adjustment methods

Calibration:
    - Group-specific probability calibration
    - Calibration metrics (ECE, MCE, Brier score)
    - Multiple calibration methods (Platt, Isotonic, Beta, Temperature)
    - Calibration-fairness trade-off analysis
    - Reliability diagrams and visualization

Threshold Optimization:
    - Single threshold optimization for fairness
    - Group-specific threshold optimization
    - Multi-objective Pareto optimization
    - Constraint-based threshold search

Reweighting:
    - Multiplicative/additive prediction adjustment
    - Rejection option based classification (ROC)
    - Distribution matching across groups
    - Calibrated equalization

Example:
    >>> from vfairness.post_processing import GroupCalibrator, expected_calibration_error
    >>>
    >>> # Evaluate calibration disparity
    >>> ece_result = expected_calibration_error(y_true, y_prob, gender)
    >>> print(f"Overall ECE: {ece_result.overall_value:.3f}")
    >>>
    >>> # Apply group-specific calibration
    >>> calibrator = GroupCalibrator(method='isotonic')
    >>> calibrator.fit(y_true, y_prob, gender)
    >>> calibrated_probs = calibrator.transform(y_prob_test, gender_test)
    >>>
    >>> # Use threshold optimization
    >>> from vfairness.post_processing import GroupThresholdOptimizer
    >>> optimizer = GroupThresholdOptimizer(constraint='equalized_odds')
    >>> optimizer.fit(y_true, y_prob, gender)
    >>> y_pred_fair = optimizer.predict(y_prob, gender)
    >>>
    >>> # Use prediction reweighting
    >>> from vfairness.post_processing import PredictionReweighter
    >>> reweighter = PredictionReweighter(constraint='demographic_parity')
    >>> reweighter.fit(y_true, y_prob, gender)
    >>> y_prob_fair = reweighter.transform(y_prob, gender)
"""

# Calibration Module
from .calibration import (
    # Calibration Methods
    BaseCalibrator,
    BetaCalibrator,
    BrierDecomposition,
    # Unified Analyzer
    CalibrationAnalyzer,
    CalibrationCurveResult,
    CalibrationDisparityResult,
    CalibrationMethod,
    CalibrationMetricResult,
    CalibrationRecommendation,
    CalibrationReport,
    GroupCalibrationResult,
    # Group-Specific Calibration
    GroupCalibrator,
    HistogramBinning,
    IntersectionalCalibrator,
    IsotonicCalibrator,
    ParetoPoint,
    PlattScaling,
    TemperatureScaling,
    TradeoffAnalysisResult,
    # Trade-off Analysis
    analyze_calibration_fairness_tradeoff,
    brier_score,
    brier_score_decomposition,
    calibration_curve,
    calibration_disparity,
    calibration_vs_error_parity,
    compute_pareto_frontier,
    create_calibrator,
    # Calibration Metrics
    expected_calibration_error,
    group_calibration_metrics,
    impossibility_diagnostics,
    maximum_calibration_error,
    mitigation_pareto,
    recommend_calibration_strategy,
)

# Ranking Module (exposure-parity re-ranking intervention)
from .ranking import exposure_parity_rerank

# Reweighting Module
from .reweighting import (
    # Base classes
    BaseReweighter,
    CalibratedEqualizer,
    DistributionMatcher,
    # Reweighters
    PredictionReweighter,
    RejectionOptionClassifier,
    ReweightingAnalysisReport,
    # Analyzer
    ReweightingAnalyzer,
    ReweightingImpactResult,
    ReweightingMethod,
    ReweightingResult,
    create_reweighter,
)

# Threshold Optimization Module
from .threshold_optimization import (
    # Base classes
    BaseThresholdOptimizer,
    # Constraints
    FairnessConstraintType,
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdAnalysisReport,
    # Analyzer
    ThresholdAnalyzer,
    ThresholdConstraint,
    ThresholdImpactResult,
    # Optimizers
    ThresholdOptimizer,
    ThresholdResult,
    compute_constraint_violation,
    create_threshold_optimizer,
    find_feasible_thresholds,
)

__all__ = [
    # Calibration
    "CalibrationAnalyzer",
    "CalibrationReport",
    "BaseCalibrator",
    "PlattScaling",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "TemperatureScaling",
    "HistogramBinning",
    "create_calibrator",
    "CalibrationMethod",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "brier_score_decomposition",
    "calibration_curve",
    "CalibrationMetricResult",
    "BrierDecomposition",
    "CalibrationCurveResult",
    "group_calibration_metrics",
    "calibration_disparity",
    "CalibrationDisparityResult",
    "GroupCalibrator",
    "GroupCalibrationResult",
    "IntersectionalCalibrator",
    "analyze_calibration_fairness_tradeoff",
    "compute_pareto_frontier",
    "mitigation_pareto",
    "calibration_vs_error_parity",
    "TradeoffAnalysisResult",
    "ParetoPoint",
    "impossibility_diagnostics",
    "recommend_calibration_strategy",
    "CalibrationRecommendation",
    # Threshold Optimization
    "BaseThresholdOptimizer",
    "ThresholdResult",
    "ThresholdConstraint",
    "ThresholdOptimizer",
    "GroupThresholdOptimizer",
    "MultiObjectiveThresholdOptimizer",
    "create_threshold_optimizer",
    "ThresholdAnalyzer",
    "ThresholdAnalysisReport",
    "ThresholdImpactResult",
    "FairnessConstraintType",
    "compute_constraint_violation",
    "find_feasible_thresholds",
    # Reweighting
    "BaseReweighter",
    "ReweightingResult",
    "ReweightingMethod",
    "PredictionReweighter",
    "RejectionOptionClassifier",
    "CalibratedEqualizer",
    "DistributionMatcher",
    "create_reweighter",
    "ReweightingAnalyzer",
    "ReweightingAnalysisReport",
    "ReweightingImpactResult",
    # Ranking (exposure-parity re-ranking intervention)
    "exposure_parity_rerank",
]
