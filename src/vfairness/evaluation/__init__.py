"""
vfairness.evaluation - Evaluation & Measurement Module

This module contains tools for measuring and evaluating fairness:

Submodules:
    - metrics: Classification, regression, and ranking fairness metrics
      (Note: internally named 'vfairness_metrics' for backward compatibility)

Metrics (implemented):
    - Classification metrics (demographic parity, equalized odds, etc.)
    - Regression metrics (MAE parity, RMSE parity, etc.)
    - Ranking metrics (exposure parity)
    - Statistical validation (CI, Bayesian, effect sizes)
    - FairExplAIner (intelligent explanations)
    - Robustness testing
    - MLOps integration (MLflow, pytest)
    - Visualization

Planned Submodules:
    - explainability: Extended interpretability tools

Example:
    >>> from vfairness.evaluation import FairnessAnalyzer, demographic_parity_difference
    >>>
    >>> # Quick metric computation
    >>> dp = demographic_parity_difference(y_true, y_pred, gender)
    >>>
    >>> # Comprehensive analysis
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
    >>> results = analyzer.compute_all_metrics(include_ci=True)
    >>> report = analyzer.get_report()
"""

# Import from vfairness_metrics (internal module name)
from .vfairness_metrics import (
    CLASSIFICATION_METRICS,
    RECOMMENDED_BOOTSTRAP_SAMPLES,
    REGRESSION_METRICS,
    SMALL_SAMPLE_THRESHOLD,
    STATISTICAL_MEASURES,
    AssessmentReport,
    ContingencyTestResult,
    DataInfo,
    ExplanationsReport,
    # FairExplAIner (Explanations)
    FairExplAIner,
    # Unified Analyzer
    FairnessAnalyzer,
    FairnessAssertionError,
    FairnessReport,
    FairnessViolation,
    GroupAdvantage,
    InsufficientEvidenceGroup,
    IntervalType,
    MetricExplanation,
    MetricResult,
    MetricStatusEntry,
    MultipleTestingResult,
    PermutationTestResult,
    ProtectedAttributeCandidate,
    RankingFairnessResult,
    RobustMetricsResult,
    SensitivityResult,
    # Statistical validation
    StatisticalResult,
    SubgroupAuditResult,
    apply_multiple_testing_correction,
    assert_fairness,
    attention_weighted_rank_fairness,
    bayesian_difference_ci,
    bayesian_mean_ci,
    bayesian_proportion_ci,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_ci,
    # Reports
    classification_fairness_report,
    cohens_d,
    comprehensive_fairness_test,
    compute_effect_sizes,
    compute_regression_effect_sizes,
    compute_robust_metrics,
    contingency_test,
    create_fairness_callback,
    # Classification metrics (point estimates)
    demographic_parity_difference,
    # Classification metrics with CI
    demographic_parity_difference_with_ci,
    demographic_parity_ratio,
    # Auto-discovery
    detect_protected_attributes,
    discover_intersectional_groups,
    equal_opportunity_difference,
    equal_opportunity_difference_with_ci,
    equalized_odds_difference,
    equalized_odds_difference_with_ci,
    explain_fairness_report,
    # Ranking metrics
    exposure_parity_difference,
    exposure_parity_ratio,
    get_group_metrics_with_ci,
    get_group_rankings,
    get_ranking_group_metrics,
    # Intersectional analysis
    identify_privileged_groups,
    identify_proxy_features,
    interpret_effect_size,
    intersectional_disparity_analysis,
    # MLOps integration
    log_fairness_to_mlflow,
    # Regression metrics (point estimates)
    mae_parity_difference,
    # Regression metrics with CI
    mae_parity_difference_with_ci,
    mean_prediction_difference,
    mean_prediction_difference_with_ci,
    normalized_discounted_kl_divergence,
    odds_ratio,
    # Robustness
    permutation_test,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
    predictive_parity_difference,
    print_explanations,
    print_report,
    rank_fairness_issues,
    regression_fairness_report,
    risk_ratio,
    rmse_parity_difference,
    rmse_parity_difference_with_ci,
    robust_fairness_comparison,
    scan_fairness_violations,
    select_method,
    sensitivity_analysis,
    stratified_bootstrap_ci,
    stress_test_fairness,
    subgroup_robustness_audit,
    test_equalized_odds_chi_square,
)

__all__ = [
    # Unified Analyzer
    "FairnessAnalyzer",
    "MetricResult",
    # Classification metrics (point estimates)
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "predictive_parity_difference",
    # Classification metrics with CI
    "demographic_parity_difference_with_ci",
    "equalized_odds_difference_with_ci",
    "equal_opportunity_difference_with_ci",
    "compute_effect_sizes",
    "get_group_metrics_with_ci",
    # Regression metrics (point estimates)
    "mae_parity_difference",
    "rmse_parity_difference",
    "mean_prediction_difference",
    # Regression metrics with CI
    "mae_parity_difference_with_ci",
    "rmse_parity_difference_with_ci",
    "mean_prediction_difference_with_ci",
    "compute_regression_effect_sizes",
    # Ranking metrics
    "exposure_parity_difference",
    "exposure_parity_ratio",
    "attention_weighted_rank_fairness",
    "normalized_discounted_kl_divergence",
    "get_ranking_group_metrics",
    "RankingFairnessResult",
    # Reports
    "classification_fairness_report",
    "regression_fairness_report",
    # Report structure (typed contract)
    "FairnessReport",
    "AssessmentReport",
    "DataInfo",
    "ExplanationsReport",
    "MetricStatusEntry",
    "InsufficientEvidenceGroup",
    "print_report",
    # FairExplAIner (Explanations)
    "FairExplAIner",
    "MetricExplanation",
    "explain_fairness_report",
    "print_explanations",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "STATISTICAL_MEASURES",
    # Statistical validation
    "StatisticalResult",
    "MultipleTestingResult",
    "IntervalType",
    "bootstrap_ci",
    "stratified_bootstrap_ci",
    "bayesian_proportion_ci",
    "bayesian_difference_ci",
    "bayesian_mean_ci",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    "apply_multiple_testing_correction",
    "cohens_d",
    "risk_ratio",
    "odds_ratio",
    "interpret_effect_size",
    "select_method",
    "SMALL_SAMPLE_THRESHOLD",
    "RECOMMENDED_BOOTSTRAP_SAMPLES",
    # MLOps integration
    "log_fairness_to_mlflow",
    "assert_fairness",
    "FairnessAssertionError",
    "create_fairness_callback",
    # Intersectional analysis
    "identify_privileged_groups",
    "intersectional_disparity_analysis",
    "get_group_rankings",
    "GroupAdvantage",
    # Auto-discovery
    "detect_protected_attributes",
    "identify_proxy_features",
    "scan_fairness_violations",
    "discover_intersectional_groups",
    "rank_fairness_issues",
    "ProtectedAttributeCandidate",
    "FairnessViolation",
    # Robustness
    "permutation_test",
    "permutation_test_demographic_parity",
    "permutation_test_equal_opportunity",
    "PermutationTestResult",
    "contingency_test",
    "test_equalized_odds_chi_square",
    "ContingencyTestResult",
    "compute_robust_metrics",
    "robust_fairness_comparison",
    "RobustMetricsResult",
    "sensitivity_analysis",
    "stress_test_fairness",
    "SensitivityResult",
    "subgroup_robustness_audit",
    "SubgroupAuditResult",
    "comprehensive_fairness_test",
]
