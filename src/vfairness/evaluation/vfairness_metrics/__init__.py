"""
vfairness_metrics - Fairness metrics module for machine learning.

This module provides comprehensive fairness metrics and analysis tools:

- Classification, regression, and ranking fairness metrics
- Statistical validation with confidence intervals
- FairExplAIner for intelligent metric explanations
- MLOps integration (MLflow, pytest)
- Visualization tools
- Intersectional analysis
- Robustness testing

Basic Usage:
    >>> from vfairness.evaluation.vfairness_metrics import demographic_parity_difference
    >>> dp = demographic_parity_difference(y_true, y_pred, gender)
    >>> print(f"Demographic Parity Difference: {dp:.3f}")

With Unified Analyzer:
    >>> from vfairness.evaluation.vfairness_metrics import FairnessAnalyzer
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
    >>> results = analyzer.compute_all_metrics(include_ci=True)
    >>> report = analyzer.get_report()

With FairExplAIner Mode:
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
    >>> report = analyzer.get_report(include_ci=True)
    >>> print_report(report)
"""

# Public API - Unified Analyzer
# Public API - Statistical Validation
from ._statistics import (
    RECOMMENDED_BOOTSTRAP_SAMPLES,
    # Constants
    SMALL_SAMPLE_THRESHOLD,
    IntervalType,
    MultipleTestingResult,
    StatisticalResult,
    apply_multiple_testing_correction,
    bayesian_difference_ci,
    bayesian_mean_ci,
    # Bayesian methods
    bayesian_proportion_ci,
    benjamini_hochberg_correction,
    # Multiple testing corrections
    bonferroni_correction,
    # Bootstrap methods
    bootstrap_ci,
    # Effect sizes
    cohens_d,
    cohens_h,
    cohens_h_interpretation,
    empirical_likelihood_ci,
    fisher_exact_test,
    group_reliability,
    interpret_effect_size,
    # Power analysis
    minimum_detectable_effect,
    odds_ratio,
    power_warning,
    # Proportion tests
    proportion_z_test,
    # Reliability tiers
    reliability_tier,
    risk_ratio,
    # Method selection
    select_method,
    sequential_fairness_test,
    simultaneous_disparity_bounds,
    stratified_bootstrap_ci,
)
from .analyzer import FairnessAnalyzer, MetricResult

# Public API - Classification Metrics (point estimates)
# Public API - Classification Metrics with CI
from .classification import (
    accuracy_parity_difference,
    auroc_parity,
    compute_effect_sizes,
    conditional_adverse_impact,
    conditional_demographic_disparity,
    conditional_demographic_disparity_with_ci,
    demographic_parity_difference,
    demographic_parity_difference_with_ci,
    demographic_parity_ratio,
    disparate_impact_ratio,
    disparate_impact_ratio_with_ci,
    equal_opportunity_difference,
    equal_opportunity_difference_with_ci,
    equalized_odds_difference,
    equalized_odds_difference_with_ci,
    fnr_parity_difference,
    fpr_parity_difference,
    fpr_parity_difference_with_ci,
    get_group_metrics_with_ci,
    negative_predictive_value_difference,
    negative_predictive_value_difference_with_ci,
    net_benefit_parity,
    predictive_parity_difference,
    predictive_parity_difference_with_ci,
    selection_rate_disparity_matrix,
    worst_group_accuracy,
)

# Public API - Auto-Discovery
from .discovery import (
    ColumnRole,
    FairnessViolation,
    ProtectedAttributeCandidate,
    classify_column_roles,
    detect_protected_attributes,
    discover_intersectional_groups,
    identify_proxy_features,
    rank_fairness_issues,
    scan_fairness_violations,
)

# Public API - FairExplAIner (Explanations)
from .explainer import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    STATISTICAL_MEASURES,
    FairExplAIner,
    MetricExplanation,
    explain_fairness_report,
    print_explanations,
)

# Public API - MLOps Integration
from .integrations import (
    FairnessAssertionError,
    assert_fairness,
    auto_log_fairness,
    create_fairness_callback,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)

# Public API - Intersectional Analysis
from .intersectional import (
    GroupAdvantage,
    generate_structured_findings,
    get_group_rankings,
    identify_privileged_groups,
    intersectional_disparity_analysis,
)

# Public API - Ranking Metrics
from .ranking import (
    RankingFairnessResult,
    attention_weighted_rank_fairness,
    exposure_parity_difference,
    exposure_parity_ratio,
    get_ranking_group_metrics,
    normalized_discounted_kl_divergence,
)

# Public API - Regression Metrics (point estimates)
# Public API - Regression Metrics with CI
from .regression import (
    compute_regression_effect_sizes,
    mae_parity_difference,
    mae_parity_difference_with_ci,
    mean_prediction_difference,
    mean_prediction_difference_with_ci,
    pricing_disparity,
    pricing_disparity_with_ci,
    rmse_parity_difference,
    rmse_parity_difference_with_ci,
)

# Public API - Reports
from .report import (
    classification_fairness_report,
    print_report,
    regression_fairness_report,
)
from .report_types import (
    AssessmentReport,
    DataInfo,
    ExplanationsReport,
    FairnessReport,
    InsufficientEvidenceGroup,
    MetricStatusEntry,
)

# Public API - Statistical Significance and Robustness Testing
from .robustness import (
    ContingencyTestResult,
    PermutationTestResult,
    RobustMetricsResult,
    SensitivityResult,
    SubgroupAuditResult,
    # Comprehensive testing
    comprehensive_fairness_test,
    # Robust statistics
    compute_robust_metrics,
    # Contingency table tests
    contingency_test,
    # Permutation testing
    permutation_test,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
    robust_fairness_comparison,
    # Sensitivity analysis
    sensitivity_analysis,
    stress_test_fairness,
    # Subgroup robustness audit
    subgroup_robustness_audit,
    test_equalized_odds_chi_square,
)


# Visualization (optional, requires matplotlib/plotly)
def _get_visualization():
    """Lazy load visualization module."""
    try:
        from . import visualization

        return visualization
    except ImportError:
        return None


# Public API - Visualization (lazy loaded)
def plot_fairness_metrics(*args, **kwargs):
    """Plot fairness metrics as a horizontal bar chart."""
    from .visualization import plot_fairness_metrics as _plot

    return _plot(*args, **kwargs)


def plot_group_comparison(*args, **kwargs):
    """Plot group-level metric comparison."""
    from .visualization import plot_group_comparison as _plot

    return _plot(*args, **kwargs)


def plot_effect_sizes(*args, **kwargs):
    """Plot effect sizes with interpretation bands."""
    from .visualization import plot_effect_sizes as _plot

    return _plot(*args, **kwargs)


def plot_confidence_intervals(*args, **kwargs):
    """Plot confidence intervals for fairness metrics."""
    from .visualization import plot_confidence_intervals as _plot

    return _plot(*args, **kwargs)


def plot_fairness_report(*args, **kwargs):
    """Generate complete visualization dashboard for a fairness report."""
    from .visualization import plot_fairness_report as _plot

    return _plot(*args, **kwargs)


def save_fairness_plots(*args, **kwargs):
    """Save all fairness visualizations to files."""
    from .visualization import save_fairness_plots as _save

    return _save(*args, **kwargs)


def create_fairness_dashboard(*args, **kwargs):
    """Create an interactive Plotly dashboard for fairness analysis."""
    from .visualization import create_fairness_dashboard as _create

    return _create(*args, **kwargs)


def plot_metrics_radar(*args, **kwargs):
    """Create a radar/spider chart showing multiple fairness metrics."""
    from .visualization import plot_metrics_radar as _plot

    return _plot(*args, **kwargs)


def plot_group_disparity_heatmap(*args, **kwargs):
    """Create a heatmap showing pairwise disparities between groups."""
    from .visualization import plot_group_disparity_heatmap as _plot

    return _plot(*args, **kwargs)


def get_available_styles():
    """Get list of available visualization styles."""
    from .visualization import get_available_styles as _get

    return _get()


def preview_palette(*args, **kwargs):
    """Preview a color palette."""
    from .visualization import preview_palette as _preview

    return _preview(*args, **kwargs)


def get_palettes():
    """Get dictionary of available color palettes."""
    from .visualization import PALETTES

    return PALETTES


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
    "fpr_parity_difference",
    "fnr_parity_difference",
    "accuracy_parity_difference",
    "worst_group_accuracy",
    "disparate_impact_ratio",
    "negative_predictive_value_difference",
    "conditional_demographic_disparity",
    "auroc_parity",
    "net_benefit_parity",
    "conditional_adverse_impact",
    "disparate_impact_ratio_with_ci",
    "fpr_parity_difference_with_ci",
    "negative_predictive_value_difference_with_ci",
    "conditional_demographic_disparity_with_ci",
    # Classification metrics with CI
    "demographic_parity_difference_with_ci",
    "equalized_odds_difference_with_ci",
    "equal_opportunity_difference_with_ci",
    "predictive_parity_difference_with_ci",
    "compute_effect_sizes",
    "get_group_metrics_with_ci",
    "selection_rate_disparity_matrix",
    # Regression metrics (point estimates)
    "mae_parity_difference",
    "rmse_parity_difference",
    "mean_prediction_difference",
    "pricing_disparity",
    "pricing_disparity_with_ci",
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
    "print_report",
    # Report structure (typed contract, see report_types.py)
    "FairnessReport",
    "AssessmentReport",
    "DataInfo",
    "ExplanationsReport",
    "MetricStatusEntry",
    "InsufficientEvidenceGroup",
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
    "reliability_tier",
    "group_reliability",
    "empirical_likelihood_ci",
    "simultaneous_disparity_bounds",
    "sequential_fairness_test",
    "SMALL_SAMPLE_THRESHOLD",
    "RECOMMENDED_BOOTSTRAP_SAMPLES",
    # MLOps integration
    "log_fairness_to_mlflow",
    "log_fairness_to_wandb",
    "auto_log_fairness",
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
    "classify_column_roles",
    "ProtectedAttributeCandidate",
    "ColumnRole",
    "FairnessViolation",
    # Statistical Significance and Robustness
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
    # Visualization
    "plot_fairness_metrics",
    "plot_group_comparison",
    "plot_effect_sizes",
    "plot_confidence_intervals",
    "plot_fairness_report",
    "save_fairness_plots",
    "create_fairness_dashboard",
    "plot_metrics_radar",
    "plot_group_disparity_heatmap",
    "get_available_styles",
    "preview_palette",
    "get_palettes",
]
