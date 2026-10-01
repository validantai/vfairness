"""
vfairness.operations - Operations & Monitoring Module

This module contains tools for production fairness operations:

Submodules:
    - cicd: CI/CD integration and automated testing
    - monitoring: Real-time tracking, drift detection, and alert mechanisms

CI/CD (vfairness.operations.cicd):
    - Data pipeline validation with DataBiasValidator
    - Model deployment gates with ModelFairnessGate
    - pytest integration with FairnessTestSuite
    - Continuous monitoring with BiasMonitor

Monitoring: Unit 1, Metric Tracking & Real-Time Bias Detection
  (vfairness.operations.monitoring.tracker):
    - mmd_gaussian: Maximum Mean Discrepancy for distribution shift detection
    - FairnessMonitor: Sliding-window monitoring with per-batch alerting
    - TemporalFairnessAnalyzer: Daily time-series analysis, trend & cycle detection

Monitoring: Unit 2, Drift Detection & Alert Mechanisms
  (vfairness.operations.monitoring.drift):
    - FairnessDriftDetector: Multi-scale wavelet + KS drift detection
    - FairnessDriftDetector.run_sprt: Sequential Probability Ratio Test

  (vfairness.operations.monitoring.alerts):
    - AdaptiveThresholdManager: Self-adjusting alert thresholds
    - FairnessAlertPrioritizer: Multi-factor scoring + notification routing
    - AlertPayload: Structured alert record

Reporting: Unit 3, Performance Dashboards & Reporting
  (vfairness.operations.reporting):
    - MetricsStore: Unified data layer with privacy-preserving queries
    - FairnessDashboard: Plotly-based progressive-disclosure visualizations
    - ReportGenerator: Automated multi-format reports with NLG
    - InteractiveDashboard: Dash app / standalone HTML with what-if analysis

Experimentation: Unit 4, A/B Testing for Fairness
  (vfairness.operations.experimentation):
    - FairnessExperiment: Core A/B testing with intersectional analysis
    - FairnessPowerAnalyzer: Per-intersection power, SPRT, adaptive sampling
    - ExperimentAnalysis: Pareto frontier, causal decomposition, recommendations

Example: Real-time monitoring (Unit 1)::

    >>> from vfairness.operations import (
    ...     FairnessMonitor, TemporalFairnessAnalyzer,
    ... )
    >>> monitor = FairnessMonitor(window_size=500, alert_threshold=0.8)
    >>> result = monitor.update_and_check(batch_df)   # batch_df has 'group_*' cols
    >>> result.any_alert

Example: Drift detection & alerting (Unit 2)::

    >>> from vfairness.operations import (
    ...     FairnessDriftDetector,
    ...     AdaptiveThresholdManager,
    ...     FairnessAlertPrioritizer,
    ... )
    >>> detector = FairnessDriftDetector()
    >>> detector.set_baseline(historical_series)
    >>> result = detector.check_drift(current_series)
    >>>
    >>> mgr = AdaptiveThresholdManager()
    >>> mgr.update_from_feedback("demographic_parity_gender", alert_was_valid=False)
    >>>
    >>> prioritizer = FairnessAlertPrioritizer()
    >>> event = FairnessAlertPrioritizer.build_drift_event(
    ...     metric_name="equalized_odds",
    ...     affected_groups=["Black", "Female"],
    ...     drift_score=0.78,
    ...     mean_shift=0.06,
    ...     regulatory_risk=1.0,
    ...     historical_discrimination=1.0,
    ... )
    >>> alert = prioritizer.create_alert(event)
    >>> print(alert.severity)   # 'CRITICAL'

Example: Reporting & Dashboards (Unit 3)::

    >>> from vfairness.operations import (
    ...     MetricsStore, FairnessDashboard, ReportGenerator,
    ...     InteractiveDashboard,
    ... )
    >>> store = MetricsStore()
    >>> store.ingest_from_monitor(monitor)
    >>> dashboard = FairnessDashboard(store)
    >>> fig = dashboard.create_executive_view()
    >>> gen = ReportGenerator(store, dashboard)
    >>> report = gen.generate_executive_report()
    >>> report.save("report.html")

Example: A/B Testing for Fairness (Unit 4)::

    >>> from vfairness.operations import (
    ...     FairnessExperiment, FairnessPowerAnalyzer, ExperimentAnalysis,
    ... )
    >>> exp = FairnessExperiment(df_ctrl, df_treat, ['gender', 'race'], 'approved')
    >>> result = exp.run_full_analysis()
    >>> analyzer = FairnessPowerAnalyzer(exp)
    >>> summary = analyzer.get_power_summary()
    >>> analysis = ExperimentAnalysis(result, experiment=exp)
    >>> rec = analysis.decision_recommendation()

Example: CI/CD deployment gate::

    >>> from vfairness.operations import DataBiasValidator, ModelFairnessGate
    >>> validator = DataBiasValidator(
    ...     protected_attributes=['gender', 'race'],
    ...     disparity_thresholds={'max_outcome_ratio': 2.0}
    ... )
    >>> result = validator.validate(df, outcome_column='approved')
    >>> gate = ModelFairnessGate(
    ...     metrics=['demographic_parity_difference'],
    ...     thresholds={'demographic_parity_difference': 0.1}
    ... )
    >>> decision = gate.evaluate(y_true, y_pred, protected_attr)
    >>> if decision.approved:
    ...     deploy_model()
"""

# CI/CD Integration
from .cicd import (
    # Monitoring (legacy cicd monitor)
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

# Monitoring: Unit 1 & 2 (Part 4 of the Fairness Pipeline Development Toolkit)
from .monitoring import (
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

__all__ = [
    # CI/CD Integration
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
    # Monitoring: Unit 1
    "mmd_gaussian",
    "FairnessMonitorConfig",
    "WindowMetrics",
    "FairnessMonitor",
    "TemporalFairnessAnalyzer",
    # Monitoring: Unit 2 (drift)
    "DriftResult",
    "MultiscaleDriftResult",
    "FairnessDriftDetector",
    # Monitoring: Unit 2 (alerts)
    "AlertPayload",
    "AdaptiveThresholdManager",
    "FairnessAlertPrioritizer",
    # Experimentation: Unit 4
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
    "ParetoPoint",
    "CausalDecomposition",
    "TemporalStabilityResult",
    "ExperimentRecommendation",
    "RecommendationDecision",
    # Reporting: Unit 3
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
]

# Reporting: Unit 3 (Part 4 of the Fairness Pipeline Development Toolkit)
# Experimentation: Unit 4 (Part 4 of the Fairness Pipeline Development Toolkit)
from .experimentation import (
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
    ParetoPoint,
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
from .reporting import (
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
