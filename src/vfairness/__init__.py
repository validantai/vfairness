"""
vfairness - A comprehensive fairness library for machine learning, LLM, Agent, and Multi-Agent systems.

This library is organized into 15 top-level sub-packages (6 pipeline stages, 8 specialized
surfaces, and 1 cross-cutting infrastructure package) covering the full AI fairness pipeline:

1. **Preprocessing** (vfairness.preprocessing)
   - Feature Engineering: Fairness-aware feature transformations
   - Bias Detection: Pre-training bias auditing and detection

2. **In-Processing** (vfairness.in_processing)
   - Loss Functions: Fairness-aware loss functions
   - Constraints: Constraint-based training algorithms
   - Regularization: Fairness-aware regularization techniques

3. **Post-Processing** (vfairness.post_processing)
   - Calibration: Group-specific probability calibration
   - Threshold Optimization: Group-specific threshold tuning
   - Reweighting: Output reweighting/adjustment

4. **Evaluation** (vfairness.evaluation)
   - Metrics: Classification, regression, and ranking fairness metrics
   - Statistical validation with confidence intervals
   - FairExplAIner for intelligent metric explanations
   - MLOps integration (MLflow, pytest)
   - Visualization tools

5. **Operations** (vfairness.operations)
   - CI/CD: Pipeline validation and deployment gates
   - Monitoring Unit 1: FairnessMonitor, TemporalFairnessAnalyzer, mmd_gaussian
   - Monitoring Unit 2: FairnessDriftDetector, AdaptiveThresholdManager, FairnessAlertPrioritizer
   - Reporting Unit 3: MetricsStore, FairnessDashboard, ReportGenerator, InteractiveDashboard
   - Experimentation Unit 4: FairnessExperiment, FairnessPowerAnalyzer, ExperimentAnalysis

6. **LLM Fairness Testing** (vfairness.llm)
   - Counterfactual prompt testing (demographic swapping)
   - Benchmark evaluation (BBQ, BOLD)
   - Output analysis (sentiment, toxicity, refusal rate)
   - Non-determinism management (noise offset calculations)

7. **Agent Fairness Testing** (vfairness.agents)
   - Correspondence testing (paired artifacts)
   - Tool selection bias auditing
   - RAG retrieval bias detection
   - Multi-stage pipeline tracking

8. **Multi-Agent Fairness Testing** (vfairness.multi_agent)
   - Non-compositionality analysis (component vs system bias)
   - Groupthink/echo-chamber detection
   - Emergent bias amplification measurement

Features:
    - Classification, regression, and ranking fairness metrics
    - Intersectional analysis support
    - Automatic minimum sample size enforcement
    - Missing value handling strategies
    - Comprehensive fairness reports
    - Statistical validation with confidence intervals
    - Bayesian credible intervals for small samples
    - Multiple comparison corrections
    - Effect size calculations
    - FairExplAIner mode for detailed metric explanations
    - Bias detection with historical pattern analysis
    - Representation bias and proxy variable detection
    - MLflow integration for MLOps pipelines
    - pytest assertions for CI/CD
    - Modern visualization tools (Plotly + Matplotlib)
    - Professional themes for academic and business contexts
    - Calibration across groups for consistent probability interpretation
    - Calibration-fairness trade-off analysis

Basic Usage:
    >>> from vfairness import demographic_parity_difference
    >>> dp = demographic_parity_difference(y_true, y_pred, gender)
    >>> print(f"Demographic Parity Difference: {dp:.3f}")

With Unified Analyzer:
    >>> from vfairness import FairnessAnalyzer
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
    >>> results = analyzer.compute_all_metrics(include_ci=True)
    >>> report = analyzer.get_report()

Module-Based Imports (Recommended):
    >>> # Preprocessing
    >>> from vfairness.preprocessing import FeatureEngineeringAnalyzer, BiasDetector
    >>>
    >>> # Post-Processing
    >>> from vfairness.post_processing import GroupCalibrator, expected_calibration_error
    >>>
    >>> # Evaluation
    >>> from vfairness.evaluation import FairnessAnalyzer, demographic_parity_difference
    >>>
    >>> # Operations
    >>> from vfairness.operations import DataBiasValidator, ModelFairnessGate

Three States, Never Two (read this before you read a number):
    Every between-group metric here can come back in one of THREE states, and
    the third one is the reason this library exists:

      * a measured value inside your tolerance,
      * a measured value outside it, and
      * NaN, meaning THE COMPARISON NEVER HAPPENED.

    NaN is not zero and not a failure. It is returned when the data cannot
    support a comparison at all: fewer than two groups meet ``min_group_size``,
    or a group's rate is undefined (an error-rate metric where a group has no
    positive labels). Returning 0.0 for a difference or 1.0 for a ratio there
    would be indistinguishable from measured perfect parity, so it would
    certify FAIR exactly where nothing was measured.

    WHAT THIS MEANS FOR YOUR CODE. Do not coerce it away::

        value = demographic_parity_difference(y_true, y_pred, group)
        if math.isnan(value):
            ...            # could not check: say so, do not pass and do not fail
        elif value > tol:
            ...            # measured breach
        else:
            ...            # measured pass

        # NEVER:  value = value or 0.0        <- turns "unknown" into "perfect"
        # NEVER:  np.nan_to_num(value)        <- same, quietly, over a whole array

    ``FairnessAnalyzer.get_report()`` surfaces the same third state as
    ``assessable: False`` with ``fairness_score: None`` and the metric listed in
    ``not_assessable_metrics``, rather than a score built from substitutes. The
    release gates behave the same way: ``assert_fairness`` and
    ``ModelFairnessGate`` REFUSE a metric they could not compute instead of
    approving an unevaluated check.

Library Comparisons:
    This library provides functionality comparable to:
    - IBM AI Fairness 360 (AIF360)
    - Microsoft Fairlearn
    - Google What-If Tool
    - Aequitas

    Key advantages:
    - Minimal, focused API
    - Native intersectional support
    - Automatic minimum sample size enforcement
    - Unified API for classification, regression, and ranking
    - Built-in statistical validation
    - Automatic method selection (bootstrap vs Bayesian)
    - FairExplAIner: Intelligent explanations for all metrics
    - Comprehensive bias detection module
    - MLOps integration (MLflow, pytest)
"""

__version__ = "0.1.0"
__author__ = "Glinz & Company GmbH (validant.ai)"

# NEW MODULE STRUCTURE (Recommended for new code)
#
# vfairness/
# ├── preprocessing/          # Data & Preprocessing
# │   ├── feature_engineering/
# │   └── bias_detection/
# │
# ├── in_processing/          # Training-Time Interventions
# │   ├── loss_functions/
# │   ├── constraints/
# │   └── regularization/
# │
# ├── post_processing/        # Prediction-Time Interventions
# │   ├── calibration/
# │   ├── threshold_optimization/
# │   └── reweighting/
# │
# ├── evaluation/             # Evaluation & Measurement
# │   └── vfairness_metrics/
# │
# ├── operations/             # Operations & Monitoring
# │   ├── cicd/
# │   ├── monitoring/
# │   ├── reporting/          # Unit 3: Performance Dashboards & Reporting
# │   └── experimentation/    # Unit 4: A/B Testing for Fairness
# │
# ├── llm/                    # LLM Fairness Testing
# │   ├── counterfactual/     # Counterfactual prompt testing
# │   ├── benchmarks/         # BBQ, BOLD benchmark runners
# │   ├── output_analysis/    # Sentiment, toxicity, refusal analysis
# │   ├── nondeterminism/     # Noise floor characterization
# │   └── api_proxy/          # Unified LLM API abstraction
# │
# ├── agents/                 # Agent Fairness Testing
# │   ├── correspondence/     # Paired-artifact testing
# │   ├── tool_bias/          # Tool selection bias auditing
# │   ├── rag_bias/           # RAG retrieval bias detection
# │   ├── pipeline_tracker/   # Multi-stage bias tracking
# │   ├── temporal/           # Temporal trajectory analysis
# │   └── action_bias/        # Action & delegation bias
# │
# └── multi_agent/            # Multi-Agent System Fairness
#     ├── compositionality/   # Non-compositionality analysis
#     ├── groupthink/         # Groupthink/echo-chamber detection
#     └── emergent/           # Emergent bias detection
#

# FLAT IMPORTS: convenience re-exports from canonical module paths

# Public API - Unified Analyzer
# Public API - Agent Fairness Testing
from .agents import (
    ActionBiasAnalyzer,
    CorrespondenceTester,
    PipelineTracker,
    RAGBiasAnalyzer,
    TemporalTracker,
    ToolBiasAuditor,
)
from .branding import branding_enabled, set_branding

# Public API - Statistical Validation
from .evaluation.vfairness_metrics._statistics import (
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
    empirical_likelihood_ci,
    group_reliability,
    interpret_effect_size,
    odds_ratio,
    reliability_tier,
    risk_ratio,
    # Method selection
    select_method,
    sequential_fairness_test,
    simultaneous_disparity_bounds,
    stratified_bootstrap_ci,
)
from .evaluation.vfairness_metrics.analyzer import FairnessAnalyzer, MetricResult

# Public API - Classification Metrics (point estimates)
# Public API - Classification Metrics with CI
from .evaluation.vfairness_metrics.classification import (
    accuracy_parity_difference,
    auroc_parity,
    calibration_difference,
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
from .evaluation.vfairness_metrics.discovery import (
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
from .evaluation.vfairness_metrics.explainer import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    STATISTICAL_MEASURES,
    FairExplAIner,
    MetricExplanation,
    explain_fairness_report,
    print_explanations,
)

# Public API - MLOps Integration
from .evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError,
    assert_fairness,
    auto_log_fairness,
    create_fairness_callback,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)

# Public API - Intersectional Analysis
from .evaluation.vfairness_metrics.intersectional import (
    GroupAdvantage,
    get_group_rankings,
    identify_privileged_groups,
    intersectional_disparity_analysis,
)

# Public API - Ranking Metrics
from .evaluation.vfairness_metrics.ranking import (
    RankingFairnessResult,
    attention_weighted_rank_fairness,
    exposure_parity_difference,
    exposure_parity_ratio,
    get_ranking_group_metrics,
    normalized_discounted_kl_divergence,
)

# Public API - Regression Metrics (point estimates)
# Public API - Regression Metrics with CI
from .evaluation.vfairness_metrics.regression import (
    compute_regression_effect_sizes,
    mae_parity_difference,
    mae_parity_difference_with_ci,
    mean_prediction_difference,
    mean_prediction_difference_with_ci,
    pricing_disparity,
    pricing_disparity_with_ci,
    r2_parity_difference,
    residual_bias,
    rmse_parity_difference,
    rmse_parity_difference_with_ci,
)

# Public API - Reports
from .evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    print_report,
    regression_fairness_report,
)
from .evaluation.vfairness_metrics.report_types import (
    AssessmentReport,
    DataInfo,
    ExplanationsReport,
    FairnessReport,
    InsufficientEvidenceGroup,
    MetricStatusEntry,
)

# Public API - Exception hierarchy (VB-API-5)
from .exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    ProtectedAttributeError,
    VfairnessError,
)

# Public API - Unified Explainer Facade
from .explainer import ExplanationReport, FairnessExplainer

# Public API - In-Processing Wrappers (sklearn-compatible fair classifiers)
from .in_processing.wrappers import (
    FairClassifier,
    FairClassifierResult,
    FairRegressor,
    make_fair_classifier,
    make_fair_regressor,
)

# Public API - LLM Fairness Testing
from .llm import (
    BenchmarkRunner,
    CoTFaithfulnessAnalyzer,
    CounterfactualTester,
    DecodingTrustRunner,
    IntersectionalAnalyzer,
    LLMApiProxy,
    NonDeterminismAnalyzer,
    OutputAnalyzer,
    noise_floor_from_runs,
)

# Public API - Multi-Agent Fairness Testing
from .multi_agent import (
    AdversarialCollusionDetector,
    CompositionalityAnalyzer,
    DelegationRoutingAuditor,
    EmergentBiasDetector,
    GroupthinkDetector,
    MultiAgentRunHarness,
    NegotiationFairnessTracker,
)

# Public API - CI/CD Integration Module
from .operations.cicd import (
    # Monitoring
    BiasMonitor,
    # Data Validation
    DataBiasValidator,
    DataValidationConfig,
    DataValidationResult,
    DriftAlert,
    FairnessReportCard,
    FairnessTestResult,
    # Testing
    FairnessTestSuite,
    GateConfig,
    GateDecision,
    GateStatus,
    # Hierarchical Intersectional Gate (Unit 5)
    HierarchicalGateConfig,
    IntersectionalGateDecision,
    MetricEvaluation,
    # Deployment Gate
    ModelFairnessGate,
    MonitorConfig,
    MonitoringResult,
    SmallSampleWarning,
    # Pre-commit hooks
    check_fairness_config,
    check_model_card,
    fairness_test,
    parametrize_fairness,
)

# Public API - Experimentation Module (Part 4, Unit 4)
from .operations.experimentation import (
    CausalDecomposition,
    DesignType,
    # ExperimentAnalysis: multi-objective & causal inference
    ExperimentAnalysis,
    ExperimentConfig,
    ExperimentRecommendation,
    ExperimentResult,
    # FairnessExperiment: core A/B testing
    FairnessExperiment,
    # FairnessPowerAnalyzer: power analysis & sequential testing
    FairnessPowerAnalyzer,
    IntersectionEffect,
    PowerConfig,
    PowerResult,
    RecommendationDecision,
    SamplingPlan,
    SequentialTestResult,
    SPRTDecision,
    TemporalStabilityResult,
    assign_clusters,
    create_factorial_design,
)
from .operations.experimentation import (
    ParetoPoint as ExperimentParetoPoint,
)

# Public API - Monitoring Module (Part 4, Units 1 & 2)
from .operations.monitoring import (
    AdaptiveThresholdManager,
    # Unit 2: Alert Mechanisms
    AlertPayload,
    # Unit 2: Drift Detection
    DriftResult,
    FairnessAlertPrioritizer,
    FairnessDriftDetector,
    FairnessMonitor,
    FairnessMonitorConfig,
    MultiscaleDriftResult,
    TemporalFairnessAnalyzer,
    WindowMetrics,
    # Unit 1: Metric Tracking & Real-Time Bias Detection
    mmd_gaussian,
)

# Public API - Reporting Module (Part 4, Unit 3)
from .operations.reporting import (
    DashboardConfig,
    # FairnessDashboard: Plotly visualizations
    FairnessDashboard,
    GeneratedReport,
    HealthScore,
    InteractiveConfig,
    # InteractiveDashboard: Dash / standalone HTML
    InteractiveDashboard,
    # MetricsStore: unified data layer
    MetricsStore,
    MetricsStoreConfig,
    OutputFormat,
    PrivacyLevel,
    ReportConfig,
    # ReportGenerator: automated reports
    ReportGenerator,
    ReportTier,
    StoredMetricRecord,
    TimeWindow,
    simulate_threshold_change,
)

# Public API - Calibration Module
from .post_processing.calibration import (
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
    IntegratedCalibrationResult,
    IntersectionalCalibrator,
    IsotonicCalibrator,
    MulticalibrationResult,
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
    calibration_in_the_large,
    calibration_slope,
    calibration_vs_error_parity,
    compute_pareto_frontier,
    create_calibrator,
    # Calibration Metrics
    expected_calibration_error,
    group_calibration_metrics,
    impossibility_diagnostics,
    integrated_calibration_index,
    integrated_calibration_index_with_ci,
    maximum_calibration_error,
    mitigation_pareto,
    multicalibration,
    multicalibration_with_ci,
    recommend_calibration_strategy,
)

# Public API - Ranking Module (exposure-parity re-ranking intervention)
from .post_processing.ranking import exposure_parity_rerank

# Public API - Reweighting Module
from .post_processing.reweighting import (
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

# Public API - Threshold Optimization Module
from .post_processing.threshold_optimization import (
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

# Public API - Bias Detection Module (comprehensive bias detection)
from .preprocessing.bias_detection import (
    HISTORICAL_RISK_PATTERNS,
    BiasAuditReport,
    # Main detector class
    BiasDetector,
    HistoricalPatternResult,
    MultivariateProxyResult,
    ProxyVariableResult,
    RepresentationBiasResult,
    StatisticalDisparityResult,
    # C. Statistical Disparity Analysis
    analyze_statistical_disparities,
    attribute_historical_pattern,
    compare_to_benchmark,
    compute_proxy_correlations,
    # A. Historical Pattern Detection
    detect_historical_patterns,
    # B. Representation Bias Detection
    detect_representation_bias,
    detect_specification_bias,
    detect_temporal_drift,
    domain_historical_context,
    # D. Proxy Variable Identification
    identify_proxy_variables,
    multivariate_proxy_leakage,
    run_disparity_tests,
)
from .preprocessing.bias_detection import (
    compute_effect_sizes as compute_disparity_effect_sizes,
)

# Public API - Feature Engineering Module
from .preprocessing.feature_engineering import (
    KNOWN_PROXY_PATTERNS,
    # Transformers
    ADASYNResampler,
    BaseFeatureTransformer,
    CorrelationReducer,
    CounterfactualAugmenter,
    DisparateImpactRemover,
    FairRepresentationTransformer,
    FeatureAnalysisReport,
    FeatureCorrelationMatrix,
    # Unified Analyzer
    FeatureEngineeringAnalyzer,
    FeatureSuppressor,
    IntersectionalTransformer,
    InversePropensityWeighter,
    LabelMassager,
    PropensityScoreWeighter,
    ProxyRiskLevel,
    ProxyType,
    Resampler,
    ResidualTransformer,
    ReweightingTransformer,
    SMOTEResampler,
    SyntheticResampler,
    TomekResampler,
    TransformationResult,
    analyze_intersectional_correlations,
    # Correlation Analysis
    compute_feature_correlations,
    counterfactual_augment,
    find_proxy_chains,
    propensity_weights,
)
from .preprocessing.feature_engineering import (
    ProxyVariableResult as FeatureProxyResult,
)
from .preprocessing.feature_engineering import (
    identify_proxy_variables as identify_feature_proxies,
)

# The published verification status of this INSTALLED version, readable from code.
# Exported at the top level on purpose: a user who wants their own pipeline to fail
# when it depends on something we have not verified should not have to know which
# submodule to reach into. See docs/CAPABILITY_STATUS.md for what the three states
# mean and https://vfairness.validant.ai/status/ for the same data as a page.
from .status import CapabilityStatus, SurfaceStatus, UnknownCapabilityError, require_checked, status


# Public API - Calibration Visualization (lazy loaded)
def plot_reliability_diagram(*args, **kwargs):
    """Plot a reliability diagram (calibration curve)."""
    from .post_processing.calibration.visualization import plot_reliability_diagram as _plot

    return _plot(*args, **kwargs)


def plot_calibration_comparison(*args, **kwargs):
    """Compare calibration curves for multiple models."""
    from .post_processing.calibration.visualization import plot_calibration_comparison as _plot

    return _plot(*args, **kwargs)


def plot_group_calibration(*args, **kwargs):
    """Plot calibration curves for each demographic group."""
    from .post_processing.calibration.visualization import plot_group_calibration as _plot

    return _plot(*args, **kwargs)


def plot_calibration_disparity(*args, **kwargs):
    """Plot calibration disparity analysis results."""
    from .post_processing.calibration.visualization import plot_calibration_disparity as _plot

    return _plot(*args, **kwargs)


def plot_tradeoff_curve(*args, **kwargs):
    """Plot the calibration-fairness trade-off frontier."""
    from .post_processing.calibration.visualization import plot_tradeoff_curve as _plot

    return _plot(*args, **kwargs)


def create_calibration_dashboard(*args, **kwargs):
    """Create a comprehensive calibration analysis dashboard."""
    from .post_processing.calibration.visualization import create_calibration_dashboard as _create

    return _create(*args, **kwargs)


# Public API - Statistical Significance and Robustness Testing
from .evaluation.vfairness_metrics.robustness import (  # noqa: E402  # intentional late import
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
        from .evaluation import vfairness_metrics

        return vfairness_metrics.visualization
    except ImportError:
        return None


# Public API - Visualization (lazy loaded)
def plot_fairness_metrics(*args, **kwargs):
    """Plot fairness metrics as a horizontal bar chart."""
    from .evaluation.vfairness_metrics.visualization import plot_fairness_metrics as _plot

    return _plot(*args, **kwargs)


def plot_group_comparison(*args, **kwargs):
    """Plot group-level metric comparison."""
    from .evaluation.vfairness_metrics.visualization import plot_group_comparison as _plot

    return _plot(*args, **kwargs)


def plot_effect_sizes(*args, **kwargs):
    """Plot effect sizes with interpretation bands."""
    from .evaluation.vfairness_metrics.visualization import plot_effect_sizes as _plot

    return _plot(*args, **kwargs)


def plot_confidence_intervals(*args, **kwargs):
    """Plot confidence intervals for fairness metrics."""
    from .evaluation.vfairness_metrics.visualization import plot_confidence_intervals as _plot

    return _plot(*args, **kwargs)


def plot_fairness_report(*args, **kwargs):
    """Generate complete visualization dashboard for a fairness report."""
    from .evaluation.vfairness_metrics.visualization import plot_fairness_report as _plot

    return _plot(*args, **kwargs)


def save_fairness_plots(*args, **kwargs):
    """Save all fairness visualizations to files."""
    from .evaluation.vfairness_metrics.visualization import save_fairness_plots as _save

    return _save(*args, **kwargs)


def create_fairness_dashboard(*args, **kwargs):
    """Create an interactive Plotly dashboard for fairness analysis."""
    from .evaluation.vfairness_metrics.visualization import create_fairness_dashboard as _create

    return _create(*args, **kwargs)


def plot_metrics_radar(*args, **kwargs):
    """Create a radar/spider chart showing multiple fairness metrics."""
    from .evaluation.vfairness_metrics.visualization import plot_metrics_radar as _plot

    return _plot(*args, **kwargs)


def plot_group_disparity_heatmap(*args, **kwargs):
    """Create a heatmap showing pairwise disparities between groups."""
    from .evaluation.vfairness_metrics.visualization import plot_group_disparity_heatmap as _plot

    return _plot(*args, **kwargs)


def get_available_styles():
    """Get list of available visualization styles."""
    from .evaluation.vfairness_metrics.visualization import get_available_styles as _get

    return _get()


def preview_palette(*args, **kwargs):
    """Preview a color palette."""
    from .evaluation.vfairness_metrics.visualization import preview_palette as _preview

    return _preview(*args, **kwargs)


def get_palettes():
    """Get dictionary of available color palettes."""
    from .evaluation.vfairness_metrics.visualization import PALETTES

    return PALETTES


__all__ = [
    # Verification status of this installed version (read it from your own CI)
    "status",
    "require_checked",
    "CapabilityStatus",
    "SurfaceStatus",
    "UnknownCapabilityError",
    # Output branding switch (VB-COM-2)
    "branding_enabled",
    "set_branding",
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
    # Classification metrics with CI
    "demographic_parity_difference_with_ci",
    "equalized_odds_difference_with_ci",
    "equal_opportunity_difference_with_ci",
    "predictive_parity_difference_with_ci",
    "disparate_impact_ratio_with_ci",
    "fpr_parity_difference_with_ci",
    "negative_predictive_value_difference_with_ci",
    "conditional_demographic_disparity_with_ci",
    "compute_effect_sizes",
    "calibration_difference",
    "get_group_metrics_with_ci",
    "selection_rate_disparity_matrix",
    # Regression metrics (point estimates)
    "mae_parity_difference",
    "rmse_parity_difference",
    "mean_prediction_difference",
    "r2_parity_difference",
    "residual_bias",
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
    # Report structure (typed contract)
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
    # Unified Explainer Facade
    "FairnessExplainer",
    "ExplanationReport",
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
    "ColumnRole",
    "ProtectedAttributeCandidate",
    "FairnessViolation",
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
    # Bias Detection Module
    "BiasDetector",
    "BiasAuditReport",
    "detect_historical_patterns",
    "HistoricalPatternResult",
    "HISTORICAL_RISK_PATTERNS",
    "domain_historical_context",
    "attribute_historical_pattern",
    "detect_representation_bias",
    "RepresentationBiasResult",
    "compare_to_benchmark",
    "analyze_statistical_disparities",
    "detect_temporal_drift",
    "detect_specification_bias",
    "StatisticalDisparityResult",
    "run_disparity_tests",
    "compute_disparity_effect_sizes",
    "identify_proxy_variables",
    "ProxyVariableResult",
    "compute_proxy_correlations",
    "multivariate_proxy_leakage",
    "MultivariateProxyResult",
    # In-Processing Wrappers
    "FairClassifier",
    "FairRegressor",
    "FairClassifierResult",
    "make_fair_classifier",
    "make_fair_regressor",
    # Calibration Module
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
    "calibration_in_the_large",
    "calibration_slope",
    "integrated_calibration_index",
    "integrated_calibration_index_with_ci",
    "multicalibration",
    "multicalibration_with_ci",
    "IntegratedCalibrationResult",
    "MulticalibrationResult",
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
    # Calibration Visualization
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_group_calibration",
    "plot_calibration_disparity",
    "plot_tradeoff_curve",
    "create_calibration_dashboard",
    # Feature Engineering Module
    "FeatureEngineeringAnalyzer",
    "FeatureAnalysisReport",
    "BaseFeatureTransformer",
    "CorrelationReducer",
    "FeatureSuppressor",
    "ResidualTransformer",
    "IntersectionalTransformer",
    "ReweightingTransformer",
    "DisparateImpactRemover",
    "LabelMassager",
    "Resampler",
    "FairRepresentationTransformer",
    "SyntheticResampler",
    "SMOTEResampler",
    "ADASYNResampler",
    "TomekResampler",
    "PropensityScoreWeighter",
    "InversePropensityWeighter",
    "CounterfactualAugmenter",
    "propensity_weights",
    "counterfactual_augment",
    "TransformationResult",
    "compute_feature_correlations",
    "identify_feature_proxies",
    "find_proxy_chains",
    "analyze_intersectional_correlations",
    "FeatureCorrelationMatrix",
    "FeatureProxyResult",
    "ProxyRiskLevel",
    "ProxyType",
    "KNOWN_PROXY_PATTERNS",
    # CI/CD Integration Module
    "DataBiasValidator",
    "DataValidationResult",
    "DataValidationConfig",
    "ModelFairnessGate",
    "GateDecision",
    "GateConfig",
    "GateStatus",
    "MetricEvaluation",
    # Hierarchical Intersectional Gate (Unit 5)
    "HierarchicalGateConfig",
    "IntersectionalGateDecision",
    "SmallSampleWarning",
    "FairnessReportCard",
    "FairnessTestSuite",
    "FairnessTestResult",
    "fairness_test",
    "parametrize_fairness",
    # Pre-commit hooks
    "check_fairness_config",
    "check_model_card",
    "BiasMonitor",
    "MonitoringResult",
    "DriftAlert",
    "MonitorConfig",
    # Monitoring Module: Unit 1 (Metric Tracking & Real-Time Bias Detection)
    "mmd_gaussian",
    "FairnessMonitorConfig",
    "WindowMetrics",
    "FairnessMonitor",
    "TemporalFairnessAnalyzer",
    # Monitoring Module: Unit 2 (Drift Detection)
    "DriftResult",
    "MultiscaleDriftResult",
    "FairnessDriftDetector",
    # Monitoring Module: Unit 2 (Alert Mechanisms)
    "AlertPayload",
    "AdaptiveThresholdManager",
    "FairnessAlertPrioritizer",
    # Reporting Module: Unit 3 (Performance Dashboards & Reporting)
    "MetricsStore",
    "MetricsStoreConfig",
    "HealthScore",
    "StoredMetricRecord",
    "PrivacyLevel",
    "FairnessDashboard",
    "DashboardConfig",
    "TimeWindow",
    "ReportGenerator",
    "ReportConfig",
    "ReportTier",
    "OutputFormat",
    "GeneratedReport",
    "InteractiveDashboard",
    "InteractiveConfig",
    "simulate_threshold_change",
    # Experimentation Module: Unit 4 (A/B Testing for Fairness)
    "FairnessExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "IntersectionEffect",
    "DesignType",
    "assign_clusters",
    "create_factorial_design",
    "FairnessPowerAnalyzer",
    "PowerConfig",
    "PowerResult",
    "SequentialTestResult",
    "SPRTDecision",
    "SamplingPlan",
    "ExperimentAnalysis",
    "ExperimentParetoPoint",
    "CausalDecomposition",
    "TemporalStabilityResult",
    "ExperimentRecommendation",
    "RecommendationDecision",
    # Threshold Optimization Module
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
    # Reweighting Module
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
    # Ranking Module (exposure-parity re-ranking intervention)
    "exposure_parity_rerank",
    # LLM Fairness Testing Module
    "CounterfactualTester",
    "OutputAnalyzer",
    "NonDeterminismAnalyzer",
    "noise_floor_from_runs",
    "LLMApiProxy",
    "BenchmarkRunner",
    "DecodingTrustRunner",
    "IntersectionalAnalyzer",
    "CoTFaithfulnessAnalyzer",
    # Agent Fairness Testing Module
    "CorrespondenceTester",
    "ToolBiasAuditor",
    "RAGBiasAnalyzer",
    "PipelineTracker",
    "TemporalTracker",
    "ActionBiasAnalyzer",
    # Multi-Agent Fairness Testing Module
    "CompositionalityAnalyzer",
    "GroupthinkDetector",
    "EmergentBiasDetector",
    "AdversarialCollusionDetector",
    "DelegationRoutingAuditor",
    "NegotiationFairnessTracker",
    "MultiAgentRunHarness",
    # Exception hierarchy (VB-API-5)
    "VfairnessError",
    "InvalidDataError",
    "InsufficientDataError",
    "ProtectedAttributeError",
    "ConfigurationError",
]
