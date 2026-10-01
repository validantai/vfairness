"""
SVG Report Rendering: vfairness.rendering

Renders polished, card-based SVG dashboards from vfairness report objects.

Design principles:
    • Zero mandatory dependencies beyond the Python stdlib.  The renderer
      uses string.Template for simple interpolation.  If Jinja2 is available,
      the renderer auto-upgrades to Jinja2 for advanced features (loops, filters).
    • Every report class gets a `.to_svg()` → str convenience method.
    • SVG templates live in `templates/` as plain `.svg` text files with
      Jinja2 / Template placeholders.
    • The public entry point is `render_svg(template_name, data_dict)`.

Usage:
    >>> from vfairness.rendering import render_svg
    >>> svg_str = render_svg('bias_audit', report.to_dict())
    >>> with open('report.svg', 'w') as f:
    ...     f.write(svg_str)

    # Or via the report object directly:
    >>> svg_str = report.to_svg()

Templates:
    Report dashboards:
        bias_audit, calibration_report, fairness_report, cicd_pipeline

    Fairness metrics visualizations:
        radar_chart, disparity_heatmap, metrics_bar_chart,
        group_comparison, effect_sizes, confidence_intervals

    Feature engineering visualizations:
        correlation_heatmap, correlation_matrix, proxy_risk,
        transformation_comparison, intersectional_analysis

    Calibration visualizations:
        reliability_diagram, group_calibration, calibration_disparity,
        pareto_frontier

    Training visualizations:
        training_report, training_analysis_report, method_comparison,
        tradeoff_analysis

    Post-processing visualizations:
        threshold_optimization_report, reweighting_comparison_report,
        fairness_detailed_report

    Monitoring & operations visualizations:
        monitoring_dashboard, drift_report, alert_timeline,
        temporal_analysis

    Experimentation & A/B testing visualizations:
        experiment_results, experiment_recommendation, power_analysis,
        causal_decomposition

    Workflow integration visualizations:
        workflow_overview, hierarchical_gate, report_card

    Robustness testing visualizations:
        robustness_testing

    Ranking fairness visualizations:
        ranking_fairness

    Data validation visualizations:
        data_validation

    Auto-discovery visualizations:
        auto_discovery

    Regression fairness visualizations:
        regression_fairness

    Reporting dashboard visualizations:
        reporting_dashboard
"""

# Report dashboard adapters
from .adapters import (
    bias_audit_to_svg,
    calibration_report_to_svg,
    cicd_pipeline_to_svg,
    fairness_report_to_svg,
)

# Calibration visualization adapters
from .adapters_calibration import (
    calibration_disparity_to_svg,
    group_calibration_to_svg,
    pareto_frontier_to_svg,
    reliability_diagram_to_svg,
)

# Auto-discovery visualization adapters
from .adapters_discovery import (
    auto_discovery_to_svg,
)

# Experimentation & A/B testing visualization adapters
from .adapters_experimentation import (
    causal_decomposition_to_svg,
    experiment_recommendation_to_svg,
    experiment_results_to_svg,
    power_analysis_to_svg,
)

# Fairness metrics visualization adapters
from .adapters_fairness import (
    confidence_intervals_to_svg,
    disparity_heatmap_to_svg,
    effect_sizes_to_svg,
    group_comparison_to_svg,
    metrics_bar_chart_to_svg,
    radar_chart_to_svg,
)

# Feature engineering visualization adapters
from .adapters_feature_engineering import (
    correlation_heatmap_to_svg,
    correlation_matrix_to_svg,
    intersectional_analysis_to_svg,
    intersectional_disparity_to_svg,
    proxy_risk_to_svg,
    transformation_comparison_to_svg,
)

# Monitoring & operations visualization adapters
from .adapters_monitoring import (
    alert_timeline_to_svg,
    drift_report_to_svg,
    monitoring_dashboard_to_svg,
    temporal_analysis_to_svg,
)

# Post-processing visualization adapters
from .adapters_post_processing import (
    fairness_detailed_report_to_svg,
    reweighting_comparison_to_svg,
    threshold_optimization_to_svg,
)

# Ranking fairness visualization adapters
from .adapters_ranking import (
    ranking_fairness_to_svg,
)

# Regression fairness visualization adapters
from .adapters_regression import (
    regression_fairness_to_svg,
)

# Reporting dashboard visualization adapters
from .adapters_reporting import (
    reporting_dashboard_to_svg,
)

# Robustness testing visualization adapters
from .adapters_robustness import (
    robustness_testing_to_svg,
)

# Training visualization adapters
from .adapters_training import (
    method_comparison_to_svg,
    tradeoff_analysis_to_svg,
    training_analysis_report_to_svg,
    training_report_to_svg,
)

# Data validation visualization adapters
from .adapters_validation import (
    data_validation_to_svg,
)

# Workflow integration visualization adapters
from .adapters_workflow import (
    hierarchical_gate_to_svg,
    report_card_to_svg,
    workflow_overview_to_svg,
)
from .engine import get_template_path, list_templates, render_svg
from .skins import SKINS, apply_skin, list_skins

__all__ = [
    # Engine
    "render_svg",
    "get_template_path",
    "list_templates",
    # Skins
    "apply_skin",
    "list_skins",
    "SKINS",
    # Report dashboards
    "bias_audit_to_svg",
    "calibration_report_to_svg",
    "fairness_report_to_svg",
    "cicd_pipeline_to_svg",
    # Fairness metrics plots
    "radar_chart_to_svg",
    "disparity_heatmap_to_svg",
    "metrics_bar_chart_to_svg",
    "group_comparison_to_svg",
    "effect_sizes_to_svg",
    "confidence_intervals_to_svg",
    # Feature engineering plots
    "correlation_heatmap_to_svg",
    "correlation_matrix_to_svg",
    "proxy_risk_to_svg",
    "transformation_comparison_to_svg",
    "intersectional_analysis_to_svg",
    "intersectional_disparity_to_svg",
    # Calibration plots
    "reliability_diagram_to_svg",
    "group_calibration_to_svg",
    "calibration_disparity_to_svg",
    "pareto_frontier_to_svg",
    # Training plots
    "training_report_to_svg",
    "training_analysis_report_to_svg",
    "method_comparison_to_svg",
    "tradeoff_analysis_to_svg",
    # Post-processing plots
    "threshold_optimization_to_svg",
    "reweighting_comparison_to_svg",
    "fairness_detailed_report_to_svg",
    # Monitoring & operations plots
    "monitoring_dashboard_to_svg",
    "drift_report_to_svg",
    "alert_timeline_to_svg",
    "temporal_analysis_to_svg",
    # Experimentation & A/B testing plots
    "experiment_results_to_svg",
    "experiment_recommendation_to_svg",
    "power_analysis_to_svg",
    "causal_decomposition_to_svg",
    # Workflow integration plots
    "workflow_overview_to_svg",
    "hierarchical_gate_to_svg",
    "report_card_to_svg",
    # Robustness testing
    "robustness_testing_to_svg",
    # Ranking fairness
    "ranking_fairness_to_svg",
    # Data validation
    "data_validation_to_svg",
    # Auto-discovery
    "auto_discovery_to_svg",
    # Regression fairness
    "regression_fairness_to_svg",
    # Reporting dashboard
    "reporting_dashboard_to_svg",
]
