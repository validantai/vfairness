"""
vfairness.operations.reporting: Performance Dashboards & Reporting (Unit 3)

This package provides the communication layer for the fairness monitoring pipeline.
It transforms raw metrics into actionable intelligence for executives, engineers,
and auditors through:

- **MetricsStore**: Unified data layer with privacy-preserving queries
- **FairnessDashboard**: Plotly-based interactive visualizations with progressive disclosure
- **ReportGenerator**: Automated multi-format, multi-tier report generation with NLG
- **InteractiveDashboard**: Full Dash app or standalone HTML with what-if analysis

Quick Start::

    >>> from vfairness.operations.reporting import MetricsStore, FairnessDashboard
    >>> store = MetricsStore()
    >>> store.ingest_from_monitor(monitor)
    >>> dashboard = FairnessDashboard(store)
    >>> fig = dashboard.create_executive_view()

    >>> from vfairness.operations.reporting import ReportGenerator
    >>> gen = ReportGenerator(store, dashboard)
    >>> report = gen.generate_executive_report()
    >>> report.save("executive_report.html")

    >>> from vfairness.operations.reporting import InteractiveDashboard
    >>> idash = InteractiveDashboard(store)
    >>> idash.save_html("interactive_dashboard.html")
"""

# MetricsStore: unified data layer
# Compliance Reporting: Fairness Navigator wizard support
from .compliance import (
    build_assurance_verdict,
    compute_adverse_action_reasons,
    compute_signed_test_log,
    generate_annex_iv_data,
    generate_dpia_sections,
    generate_iso42001_evidence_map,
    generate_model_card,
    generate_risk_register_from_audit,
)

# FairnessDashboard: Plotly visualizations
from .dashboard import (
    DashboardConfig,
    FairnessDashboard,
    TimeWindow,
)

# InteractiveDashboard: Dash / standalone HTML
from .interactive import (
    InteractiveConfig,
    InteractiveDashboard,
    simulate_threshold_change,
)

# ReportGenerator: automated reports
from .reports import (
    GeneratedReport,
    OutputFormat,
    ReportConfig,
    ReportGenerator,
    ReportTier,
)
from .store import (
    HealthScore,
    MetricsStore,
    MetricsStoreConfig,
    PrivacyLevel,
    StoredMetricRecord,
)

__all__ = [
    # Store
    "MetricsStore",
    "MetricsStoreConfig",
    "HealthScore",
    "StoredMetricRecord",
    "PrivacyLevel",
    # Dashboard
    "FairnessDashboard",
    "DashboardConfig",
    "TimeWindow",
    # Reports
    "ReportGenerator",
    "ReportConfig",
    "ReportTier",
    "OutputFormat",
    "GeneratedReport",
    # Interactive
    "InteractiveDashboard",
    "InteractiveConfig",
    "simulate_threshold_change",
    # Compliance
    "generate_risk_register_from_audit",
    "generate_dpia_sections",
    "compute_adverse_action_reasons",
    "compute_signed_test_log",
    "generate_annex_iv_data",
    "generate_model_card",
    "generate_iso42001_evidence_map",
    "build_assurance_verdict",
]
