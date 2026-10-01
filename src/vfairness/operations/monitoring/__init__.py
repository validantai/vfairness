"""
vfairness.operations.monitoring - Real-Time Fairness Monitoring

Part 4, Units 1 & 2 of the Fairness Pipeline Development Toolkit.

This submodule implements production-grade fairness monitoring infrastructure:

Unit 1: Metric Tracking and Real-Time Bias Detection (tracker.py):
    - mmd_gaussian: Maximum Mean Discrepancy for distribution shift detection
    - FairnessMonitor: Sliding-window monitoring of fairness metrics with alerting
    - TemporalFairnessAnalyzer: Time-series analysis of fairness trends and cycles

Unit 2: Drift Detection and Alert Mechanisms (drift.py, alerts.py):
    - FairnessDriftDetector: Multi-scale drift detection via wavelet decomposition + KS tests
    - AdaptiveThresholdManager: Self-adjusting thresholds that learn from alert feedback
    - FairnessAlertPrioritizer: Multi-factor alert scoring and routing

Design Philosophy (Tiered Monitoring):
    - Tier 1 (Real-Time):   Lightweight metrics on every prediction / micro-batch.
    - Tier 2 (Near Real-Time): Comprehensive metrics on larger sliding windows.
    - Tier 3 (Offline):    Deep-dive intersectional + temporal analysis on schedule
                            or triggered by Tier 1/2 alerts.

Example::

    from vfairness.operations.monitoring import (
        FairnessMonitor, TemporalFairnessAnalyzer,
        FairnessDriftDetector, AdaptiveThresholdManager,
        FairnessAlertPrioritizer,
    )

    # -- Unit 1: Real-time tracking --
    monitor = FairnessMonitor(window_size=1000, alert_threshold=0.8)
    alerts  = monitor.update_and_check(batch_df)

    # Temporal analysis
    analyzer = TemporalFairnessAnalyzer(lookback_days=90)
    analyzer.update_daily_metrics(pd.Timestamp.now(), metrics_dict)
    degraded, value = analyzer.detect_weekly_degradation("demographic_parity")

    # -- Unit 2: Drift detection & alerting --
    detector = FairnessDriftDetector()
    detector.set_baseline(historical_series)
    drift_result = detector.check_drift(current_series)

    mgr = AdaptiveThresholdManager()
    threshold = mgr.get_threshold("demographic_parity_gender")
    mgr.update_from_feedback("demographic_parity_gender", alert_was_valid=True)

    prioritizer = FairnessAlertPrioritizer()
    score, severity = prioritizer.calculate_priority(drift_event)
    routing = prioritizer.route_alert(severity)
"""

from .alerts import (
    AdaptiveThresholdManager,
    AlertPayload,
    FairnessAlertPrioritizer,
)
from .drift import (
    DriftResult,
    FairnessDriftDetector,
    MultiscaleDriftResult,
)
from .sequential import (
    cusum_drift,
    page_hinkley,
    sequential_fairness_drift,
)
from .tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
    WindowMetrics,
    mmd_gaussian,
)

__all__ = [
    # Unit 1: tracker
    "mmd_gaussian",
    "FairnessMonitorConfig",
    "WindowMetrics",
    "FairnessMonitor",
    "TemporalFairnessAnalyzer",
    # Unit 2: drift
    "DriftResult",
    "MultiscaleDriftResult",
    "FairnessDriftDetector",
    # Unit 2: alerts
    "AlertPayload",
    "AdaptiveThresholdManager",
    "FairnessAlertPrioritizer",
    # Sequential drift detection (CS-T-23)
    "cusum_drift",
    "page_hinkley",
    "sequential_fairness_drift",
]
