"""
Calibration Module for vfairness.

This module provides comprehensive calibration capabilities for ensuring
probability predictions have consistent meaning across demographic groups,
implementing a complete suite of calibration techniques and metrics.

Core Components:
    A. Calibration Methods
       - Platt Scaling (logistic calibration)
       - Isotonic Regression (non-parametric)
       - Beta Calibration (parametric with bounded support)
       - Temperature Scaling (neural network calibration)
       - Histogram Binning (discretization-based)

    B. Calibration Metrics
       - Expected Calibration Error (ECE)
       - Maximum Calibration Error (MCE)
       - Brier Score and decomposition
       - Reliability diagrams and curves

    C. Group-Specific Calibration
       - Per-group calibration transformations
       - Intersectional calibration support
       - Calibration disparity detection

    D. Calibration-Fairness Trade-off Analysis
       - Trade-off visualization and quantification
       - Multi-objective calibration optimization
       - Impossibility theorem diagnostics

The module supports:
    - Group-specific calibration to ensure consistent probability interpretation
    - Automatic method selection based on data characteristics
    - Comprehensive evaluation with statistical validation
    - Integration with the broader vfairness fairness analysis pipeline

Example:
    >>> from vfairness.post_processing.calibration import GroupCalibrator, expected_calibration_error
    >>>
    >>> # Evaluate calibration disparity
    >>> ece_results = expected_calibration_error(y_true, y_prob, protected_attr)
    >>> print(f"Overall ECE: {ece_results.overall_ece:.3f}")
    >>> for group, ece in ece_results.group_ece.items():
    ...     print(f"  {group}: {ece:.3f}")
    >>>
    >>> # Apply group-specific calibration
    >>> calibrator = GroupCalibrator(method='isotonic')
    >>> calibrator.fit(y_true, y_prob, protected_attr)
    >>> calibrated_probs = calibrator.transform(y_prob, protected_attr)

References:
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
    - Kleinberg, J., et al. (2016). Inherent Trade-offs in Fair Risk Scores.
    - Naeini, M. P., et al. (2015). Obtaining Well Calibrated Probabilities. AAAI.
    - Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
    - Kull, M., et al. (2017). Beta Calibration. AISTATS.
    - Zadrozny, B. & Elkan, C. (2002). Transforming Classifier Scores. KDD.
    - Corbett-Davies, S. & Goel, S. (2018). The Measure and Mismeasure of Fairness.
"""

# Calibration Methods
# Unified Analyzer
from .analyzer import (
    CalibrationAnalyzer,
    CalibrationReport,
)

# Group-Specific Calibration
from .group_calibrator import (
    GroupCalibrationResult,
    GroupCalibrator,
    IntersectionalCalibrator,
)
from .methods import (
    # Base calibrator
    BaseCalibrator,
    BetaCalibrator,
    # Calibration method enum
    CalibrationMethod,
    HistogramBinning,
    IsotonicCalibrator,
    # Specific methods
    PlattScaling,
    TemperatureScaling,
    # Factory function
    create_calibrator,
)

# Calibration Metrics
from .metrics import (
    BrierDecomposition,
    CalibrationCurveResult,
    CalibrationDisparityResult,
    # Result containers
    CalibrationMetricResult,
    IntegratedCalibrationResult,
    MulticalibrationResult,
    brier_score,
    brier_score_decomposition,
    calibration_curve,
    calibration_disparity,
    # Recalibration diagnostics + smooth/subgroup calibration (spec-v2 spine)
    calibration_in_the_large,
    calibration_slope,
    # Advanced diagnostics
    cv_calibration_stability,
    ece_confidence_intervals,
    # Core metrics
    expected_calibration_error,
    # Group-level analysis
    group_calibration_metrics,
    integrated_calibration_index,
    integrated_calibration_index_with_ci,
    maximum_calibration_error,
    multicalibration,
    multicalibration_with_ci,
    per_group_brier_decomposition,
    sufficiency_test,
)

# Trade-off Analysis
from .tradeoffs import (
    CalibrationRecommendation,
    ParetoPoint,
    # Result containers
    TradeoffAnalysisResult,
    # Analysis functions
    analyze_calibration_fairness_tradeoff,
    calibration_vs_error_parity,
    compute_pareto_frontier,
    # Utilities
    impossibility_diagnostics,
    mitigation_pareto,
    recommend_calibration_strategy,
)

# Visualization
from .visualization import (
    create_calibration_dashboard,
    plot_calibration_comparison,
    plot_calibration_disparity,
    plot_group_calibration,
    plot_pareto_frontier,
    plot_reliability_diagram,
    plot_tradeoff_curve,
)

__all__ = [
    # Calibration Methods
    "BaseCalibrator",
    "PlattScaling",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "TemperatureScaling",
    "HistogramBinning",
    "create_calibrator",
    "CalibrationMethod",
    # Calibration Metrics
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
    # Recalibration diagnostics + smooth/subgroup calibration (spec-v2 spine)
    "calibration_in_the_large",
    "calibration_slope",
    "integrated_calibration_index",
    "integrated_calibration_index_with_ci",
    "multicalibration",
    "multicalibration_with_ci",
    "IntegratedCalibrationResult",
    "MulticalibrationResult",
    # Advanced diagnostics (per-group Brier, ECE CIs, sufficiency, CV stability)
    "per_group_brier_decomposition",
    "ece_confidence_intervals",
    "sufficiency_test",
    "cv_calibration_stability",
    # Group-Specific Calibration
    "GroupCalibrator",
    "GroupCalibrationResult",
    "IntersectionalCalibrator",
    # Unified Analyzer
    "CalibrationAnalyzer",
    "CalibrationReport",
    # Trade-off Analysis
    "analyze_calibration_fairness_tradeoff",
    "compute_pareto_frontier",
    "mitigation_pareto",
    "calibration_vs_error_parity",
    "TradeoffAnalysisResult",
    "ParetoPoint",
    "impossibility_diagnostics",
    "recommend_calibration_strategy",
    "CalibrationRecommendation",
    # Visualization
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_group_calibration",
    "plot_calibration_disparity",
    "plot_tradeoff_curve",
    "plot_pareto_frontier",
    "create_calibration_dashboard",
]
