"""
Automated Bias Checks for CI/CD Pipelines.

This module provides infrastructure components for integrating fairness validation
into Continuous Integration/Continuous Deployment (CI/CD) workflows. It enables
systematic, automated bias prevention across data engineering and ML pipelines.

Core Components:
    - DataBiasValidator: Validates data pipelines for bias before model training
    - ModelFairnessGate: Deployment gates that enforce fairness requirements
    - FairnessTestSuite: pytest integration for test-driven bias prevention
    - BiasMonitor: Continuous monitoring with drift detection

Key Concepts:
    - **Continuous Fairness Validation**: Automated testing at every stage
    - **Test-Driven Bias Prevention**: Define fairness requirements as tests
    - **Infrastructure as Code**: Fairness standards embedded in pipelines
    - **Version Control for Fairness**: Track fairness metrics over time

Example (Data Pipeline Validation):
    >>> from vfairness.operations.cicd import DataBiasValidator
    >>>
    >>> validator = DataBiasValidator(
    ...     protected_attributes=['gender', 'race'],
    ...     representation_thresholds={'min_group_fraction': 0.05},
    ...     disparity_thresholds={'max_outcome_ratio': 2.0}
    ... )
    >>>
    >>> # In your data pipeline
    >>> result = validator.validate(df, outcome_column='approved')
    >>> if not result.passed:
    ...     raise ValueError(f"Data bias detected: {result.summary}")

Example (Model Deployment Gate):
    >>> from vfairness.operations.cicd import ModelFairnessGate
    >>>
    >>> gate = ModelFairnessGate(
    ...     metrics=['demographic_parity_difference', 'equalized_odds_difference'],
    ...     thresholds={'demographic_parity_difference': 0.1},
    ...     require_improvement=True
    ... )
    >>>
    >>> # In your deployment pipeline
    >>> decision = gate.evaluate(y_true, y_pred, protected_attr, baseline_metrics)
    >>> if decision.approved:
    ...     deploy_model()
    ... else:
    ...     block_deployment(decision.blocking_reasons)

Example (pytest Integration):
    >>> from vfairness.operations.cicd import FairnessTestSuite
    >>>
    >>> # In conftest.py
    >>> @pytest.fixture
    ... def fairness_suite():
    ...     return FairnessTestSuite(
    ...         protected_attributes=['gender'],
    ...         metrics=['demographic_parity_difference'],
    ...         thresholds={'demographic_parity_difference': 0.1}
    ...     )
    >>>
    >>> # In test_fairness.py
    >>> def test_model_fairness(fairness_suite, trained_model, test_data):
    ...     fairness_suite.test_model(trained_model, test_data)

Example (Continuous Monitoring):
    >>> from vfairness.operations.cicd import BiasMonitor
    >>>
    >>> monitor = BiasMonitor(
    ...     baseline_metrics=production_baseline,
    ...     drift_threshold=0.05,
    ...     alert_callback=send_slack_alert
    ... )
    >>>
    >>> # In production scoring pipeline
    >>> monitor.log_batch(predictions, actuals, protected_attrs)
    >>> if monitor.drift_detected():
    ...     monitor.trigger_alert()

References:
    - Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning.
    - Holstein, K., et al. (2019). Improving Fairness in Machine Learning Systems.
    - Bellamy, R.K.E., et al. (2018). AI Fairness 360: An Extensible Toolkit.
"""

from .gate import (
    FairnessReportCard,
    GateConfig,
    GateDecision,
    GateStatus,
    HierarchicalGateConfig,
    IntersectionalGateDecision,
    MetricEvaluation,
    ModelFairnessGate,
    SmallSampleWarning,
)
from .monitor import (
    BiasMonitor,
    DriftAlert,
    MonitorConfig,
    MonitoringResult,
)
from .precommit import (
    check_fairness_config,
    check_model_card,
)

# Canonical plain-language data-quality report + its task handler. Exported
# so the task consumer can register the handler via the documented
# in-process bridge: `from vfairness.operations.cicd import handle_data_validation`.
from .quality_report import build_quality_report
from .task_handlers import handle_data_validation
from .testing import (
    FairnessTestResult,
    FairnessTestSuite,
    fairness_test,
    parametrize_fairness,
)
from .validator import (
    DataBiasValidator,
    DataValidationConfig,
    DataValidationResult,
)

__all__ = [
    # Data Validation
    "DataBiasValidator",
    "DataValidationResult",
    "build_quality_report",
    "handle_data_validation",
    "DataValidationConfig",
    # Deployment Gate
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
    # Testing
    "FairnessTestSuite",
    "FairnessTestResult",
    "fairness_test",
    "parametrize_fairness",
    # Pre-commit hooks
    "check_fairness_config",
    "check_model_card",
    # Monitoring
    "BiasMonitor",
    "MonitoringResult",
    "DriftAlert",
    "MonitorConfig",
]
