"""
Bias Monitor for Continuous Monitoring.

This module provides the BiasMonitor class for continuous fairness monitoring
in production ML systems with drift detection capabilities.
"""

import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeGuard

import numpy as np

from vfairness._triage import is_measured, unmeasurable_reason


def _is_finite(value: Any) -> TypeGuard[float]:
    """True only for a real, finite number.

    NaN and inf both break every ordered comparison, so a monitoring surface
    that compares them reports "not breached" for a quantity it never measured.
    Everything that grades a value here goes through this first.

    READINESS-6, 2026-09-10. The body was ``bool(np.isfinite(value))``, which
    does not answer the question this function's name asks. ``np.isfinite`` maps
    over a SEQUENCE and returns an array, so a ONE-element list answered True
    (``bool(array([True]))`` is True) while a two-element list answered False
    (the ambiguous-truth-value ValueError was caught and read as "not finite").
    A monitored metric that is a list is a caller error either way, and the
    honest answer to "is this a finite number" is no, for every length. numpy's
    boolean answered True as well, so a recorded yes/no became the drift score
    1.0. Delegated to the canonical rule; the two call sites both fail SAFE on
    False (an explicit could-not-check that alerts the operator), so a value
    this now refuses reaches a human instead of being graded.
    """
    return is_measured(value)


class AlertSeverity(Enum):
    """Severity levels for bias alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of fairness drift.

    ``NOT_MEASURABLE`` is deliberately not a fourth kind of drift: it is the
    could-not-check state travelling on the alert channel, so a window in which
    a monitored metric could not be measured reaches the operator instead of
    being silently recorded as "not drifting".
    """

    METRIC_DRIFT = "metric_drift"
    DISTRIBUTION_DRIFT = "distribution_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    NOT_MEASURABLE = "not_measurable"


@dataclass
class MonitorConfig:
    """Configuration for bias monitoring.

    Attributes:
        drift_threshold: Threshold for detecting metric drift.
        window_size: Number of batches to use for rolling statistics.
        min_samples_for_alert: Minimum samples before triggering alerts.
        metrics_to_monitor: List of fairness metrics to track.
        alert_cooldown_seconds: Minimum time between alerts for same metric.
        enable_distribution_monitoring: Track group distribution changes.
        enable_performance_monitoring: Track performance degradation.
    """

    drift_threshold: float = 0.05
    window_size: int = 10
    min_samples_for_alert: int = 100
    metrics_to_monitor: List[str] = field(default_factory=lambda: ["demographic_parity_difference"])
    alert_cooldown_seconds: float = 3600.0
    enable_distribution_monitoring: bool = True
    enable_performance_monitoring: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "drift_threshold": self.drift_threshold,
            "window_size": self.window_size,
            "min_samples_for_alert": self.min_samples_for_alert,
            "metrics_to_monitor": self.metrics_to_monitor,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "enable_distribution_monitoring": self.enable_distribution_monitoring,
            "enable_performance_monitoring": self.enable_performance_monitoring,
        }


@dataclass
class DriftAlert:
    """Alert triggered when fairness drift is detected.

    Attributes:
        alert_id: Unique identifier for the alert.
        severity: Alert severity level.
        drift_type: Type of drift detected.
        metric_name: Name of the affected metric.
        baseline_value: Baseline/expected value.
        current_value: Current observed value.
        drift_magnitude: Magnitude of the drift.
        timestamp: When the alert was triggered.
        message: Human-readable alert message.
        metadata: Additional context.
    """

    alert_id: str
    severity: AlertSeverity
    drift_type: DriftType
    metric_name: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "drift_type": self.drift_type.value,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "drift_magnitude": self.drift_magnitude,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        # Three states, never two, the rule MonitoringResult.__repr__ below
        # already follows. G05, 2026-09-30: this line was
        # ``drift={self.drift_magnitude:.4f}``, and a could-not-check alert
        # carries a NaN magnitude BY DESIGN (see BiasMonitor._create_alert,
        # which passes float("nan") for it and grades the severity explicitly
        # because every comparison against NaN is False). So the log line for
        # the one state an operator most needs to see read
        # ``DriftAlert(warning: demographic_parity_difference drift=nan)``,
        # with ``drift_type`` -- the field this class's own docstring calls
        # "the could-not-check state travelling on the alert channel" -- absent
        # from the repr entirely. A None magnitude raised TypeError instead of
        # printing, which is how a log line becomes a crash.
        if is_measured(self.drift_magnitude):
            magnitude = f"{float(self.drift_magnitude):.4f}"
        else:
            magnitude = f"NOT MEASURED ({unmeasurable_reason(self.drift_magnitude)})"
        return (
            f"DriftAlert({self.severity.value}/{self.drift_type.value}: "
            f"{self.metric_name} drift={magnitude})"
        )


@dataclass
class MonitoringResult:
    """Result of monitoring a batch of predictions.

    Attributes:
        batch_id: Identifier for the batch.
        metrics: Computed fairness metrics for the batch.
        drift_detected: Whether this batch needs a human: a monitored metric
            drifted past the threshold, OR one could not be measured at all.
            ``could_not_check`` tells the two apart, so the third state is
            recorded rather than collapsed. It used to be the drift question
            alone, and a NaN metric answered it False.
        could_not_check: Names of the monitored metrics that could not be
            compared against their baseline in this batch: the metric was not
            computed, it has no baseline, either value is non-finite, or the
            monitor has not yet seen ``min_samples_for_alert`` samples and so
            compared nothing at all. Empty on a fully measured batch.
        alerts: List of alerts triggered.
        sample_count: Number of samples in the batch.
        timestamp: When monitoring was performed.
        group_distribution: Distribution of protected groups.
        metadata: The ``metadata`` mapping handed to ``BiasMonitor.log_batch``
            for this batch (model version, data slice, run id, ...), verbatim.
            Empty when the caller supplied none. F9, 2026-09-09: ``log_batch``
            documented and accepted this argument and then dropped it, so a
            monitoring record that had been given a model version could not
            say which model it was a record of.
    """

    batch_id: str
    metrics: Dict[str, float]
    drift_detected: bool
    alerts: List[DriftAlert]
    sample_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    group_distribution: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    could_not_check: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "batch_id": self.batch_id,
            "metrics": self.metrics,
            "drift_detected": self.drift_detected,
            "could_not_check": list(self.could_not_check),
            "alerts": [a.to_dict() for a in self.alerts],
            "sample_count": self.sample_count,
            "timestamp": self.timestamp.isoformat(),
            "group_distribution": self.group_distribution,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        # Three states, never two. "OK" over a metric that was never measured
        # is the same false all-clear this class exists to report on.
        if self.could_not_check:
            status = f"COULD NOT CHECK ({', '.join(self.could_not_check)})"
        elif self.drift_detected:
            status = "DRIFT DETECTED"
        else:
            status = "OK"
        return f"MonitoringResult({self.batch_id}: {status}, {len(self.alerts)} alerts)"


class BiasMonitor:
    """Continuous monitoring for fairness drift detection.

    This class tracks fairness metrics over time and detects when
    they drift from baseline values, triggering alerts as needed.

    Attributes:
        config: Monitoring configuration.
        baseline_metrics: Baseline metric values to compare against.
        alert_callback: Optional callback for alerts.

    Example:
        >>> monitor = BiasMonitor(
        ...     baseline_metrics={'demographic_parity_difference': 0.05},
        ...     drift_threshold=0.05,
        ...     alert_callback=send_slack_alert
        ... )
        >>> # In production scoring pipeline
        >>> monitor.log_batch(predictions, actuals, protected_attrs)
        >>> if monitor.drift_detected():
        ...     monitor.trigger_alert()

    ``drift_detected()`` is True for a metric that DRIFTED and for one that
    could NOT be measured, because both need a human and only one of them used
    to be reported. ``get_drift_status()`` keeps the three apart (True /
    False / None) and ``unmeasurable_metrics()`` names the third group.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: bias_monitor. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.05,
        window_size: int = 10,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None,
        config: Optional[MonitorConfig] = None,
        compute_metrics_fn: Optional[Callable] = None,
    ):
        """Initialize the BiasMonitor.

        Args:
            baseline_metrics: Baseline metric values to compare against.
            drift_threshold: Threshold for detecting drift.
            window_size: Number of batches for rolling statistics.
            alert_callback: Optional callback function for alerts.
            config: Full configuration object.
            compute_metrics_fn: Custom function to compute metrics.
        """
        if config is not None:
            self.config = config
        else:
            self.config = MonitorConfig(
                drift_threshold=drift_threshold,
                window_size=window_size,
            )

        self.baseline_metrics = baseline_metrics or {}
        self.alert_callback = alert_callback
        self.compute_metrics_fn = compute_metrics_fn

        # Internal state
        self._metric_history: Dict[str, deque] = {}
        self._batch_count = 0
        self._total_samples = 0
        self._alerts: List[DriftAlert] = []
        self._last_alert_time: Dict[str, datetime] = {}
        # Three states: True (drifting), False (measured, within threshold),
        # None (could not be checked). A metric never seen is simply absent.
        self._current_drift_status: Dict[str, Optional[bool]] = {}
        self._group_distribution_history: deque = deque(maxlen=self.config.window_size)

        # Initialize metric history
        for metric in self.config.metrics_to_monitor:
            self._metric_history[metric] = deque(maxlen=self.config.window_size)

    def log_batch(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
        protected_attr: np.ndarray,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitoringResult:
        """Log a batch of predictions for monitoring.

        Args:
            y_pred: Predicted labels/probabilities.
            y_true: True labels (optional for some metrics).
            protected_attr: Protected attribute values.
            batch_id: Optional identifier for the batch.
            metadata: Optional additional metadata (model version, data slice,
                run id, ...). Stored verbatim on ``MonitoringResult.metadata``
                so the monitoring record says what it is a record of.

        Returns:
            MonitoringResult with metrics and any alerts.
        """
        self._batch_count += 1
        self._total_samples += len(y_pred)

        if batch_id is None:
            batch_id = f"batch_{self._batch_count}"

        # Compute metrics
        if self.compute_metrics_fn:
            metrics = self.compute_metrics_fn(y_true, y_pred, protected_attr)
        else:
            metrics = self._compute_default_metrics(y_true, y_pred, protected_attr)

        # Track group distribution
        group_distribution = None
        if self.config.enable_distribution_monitoring:
            unique, counts = np.unique(protected_attr, return_counts=True)
            group_distribution = {str(g): c / len(protected_attr) for g, c in zip(unique, counts)}
            self._group_distribution_history.append(group_distribution)

        # Update metric history
        for metric_name, value in metrics.items():
            if metric_name in self._metric_history:
                self._metric_history[metric_name].append(value)

        # BGL-D: a monitored metric this batch did NOT measure records NaN, not
        # nothing. Leaving the deque untouched made `get_current_metrics()` -
        # and through it `to_prometheus_metrics()`, `get_summary()` and
        # `trigger_alert()`'s ranking - re-serve the PREVIOUS batch's value as
        # the current one. Measured 2026-09-11: batch 1 two groups (0.90
        # measured), batch 2 a single group (`_compute_default_metrics` returns
        # {}), and the export still read
        # `fairness_demographic_parity_difference 0.8999999999999999` for a
        # window in which the gap was never computed. NaN is what this module
        # already records for a window it could not measure (an undefined TPR),
        # so the could-not-check state travels the same way through the history.
        for metric_name in self.config.metrics_to_monitor:
            if metric_name not in metrics and metric_name in self._metric_history:
                self._metric_history[metric_name].append(float("nan"))

        # Check for drift
        alerts = []
        drift_detected = False
        could_not_check: List[str] = []

        if self._total_samples >= self.config.min_samples_for_alert:
            for metric_name in self.config.metrics_to_monitor:
                # `.get` on both, then the three reasons named separately.
                # Rule: the guard goes ABOVE the dispatch. There are three
                # doors into "nothing was compared" and only one of them used
                # to be guarded, so fixing that one just moved the false
                # all-clear next door. `metric_name in metrics` rather than
                # `current_value is not None`, because a key PRESENT holding
                # None is a could-not-check too and `.get` cannot tell the two
                # apart.
                current_value = metrics.get(metric_name)
                baseline_value = self.baseline_metrics.get(metric_name)

                # BGL-D (fail closed, door 1): `_compute_default_metrics`
                # returns {} for a batch with fewer than two groups
                # (`len(groups) < 2`), and a custom `compute_metrics_fn` may
                # omit a monitored metric for the same kind of reason. The bare
                # `continue` that used to stand here dropped the metric without
                # recording anything, so the batch reported
                # `MonitoringResult(b2: OK, 0 alerts)`, could_not_check [],
                # get_drift_status() {'demographic_parity_difference': False}
                # -- the positive claim "measured against the baseline and
                # within the threshold", which is that method's own documented
                # meaning of False -- over a gap nobody computed. Measured
                # 2026-09-11 with a single-group second batch; a zero-row batch
                # behaved identically.
                #
                # BGL-D (fail closed, door 2): a monitored metric with NO
                # baseline was dropped by a second bare `continue`. Nothing is
                # compared, so `get_drift_status()` stayed EMPTY and
                # `drift_detected()` answered False over it -- `any([])` --
                # which is the same false all-clear reached with an empty
                # mapping instead of a False entry. A baseline of NaN already
                # routed to could-not-check below; an ABSENT baseline is the
                # same state through a different door.
                #
                # CRITICAL (fail closed): `_compute_default_metrics` returns NaN
                # BY DESIGN for a window it could not measure (a group with no
                # positives has an undefined TPR). `abs(nan - baseline)` is NaN,
                # `nan > threshold` is False, so the else branch below wrote
                # `_current_drift_status[metric] = False`: the positive claim
                # "this metric is NOT drifting", made about a metric nobody
                # measured. Measured 2026-09-10 on computed metrics
                # {'demographic_parity_difference': 0.5, 'equalized_odds_
                # difference': nan} against a matching baseline:
                # drift_detected() returned False, get_drift_status() reported
                # both metrics False, trigger_alert() returned None, and
                # to_prometheus_metrics() exported
                # `fairness_drift_equalized_odds_difference 0`. Every dashboard,
                # alert rule and SLO built on that gauge reads green over a
                # metric that was never measured.
                #
                # Three states, never two: the status becomes None (could not
                # check), never False, and an alert is raised so the window
                # reaches the operator instead of resolving into silence.
                if metric_name not in metrics:
                    not_measurable_reason = "it was not computed for this batch"
                elif baseline_value is None:
                    not_measurable_reason = "it has no baseline value to be compared against"
                elif not _is_finite(current_value) or not _is_finite(baseline_value):
                    not_measurable_reason = (
                        f"it has no finite value to compare "
                        f"(baseline={baseline_value}, current={current_value})"
                    )
                else:
                    not_measurable_reason = None

                if not_measurable_reason is not None:
                    could_not_check.append(metric_name)
                    self._current_drift_status[metric_name] = None
                    if self._should_alert(metric_name):
                        alert = self._create_alert(
                            metric_name=metric_name,
                            # An absent or non-finite baseline is NOT a
                            # baseline; NaN says so and keeps DriftAlert's
                            # declared float type honest.
                            baseline_value=(
                                baseline_value if _is_finite(baseline_value) else float("nan")
                            ),
                            current_value=(
                                current_value if _is_finite(current_value) else float("nan")
                            ),
                            drift_magnitude=float("nan"),
                            drift_type=DriftType.NOT_MEASURABLE,
                            custom_message=(
                                f"{metric_name} could not be measured in this window: "
                                f"{not_measurable_reason}, so drift could not be checked. "
                                f"This is a could-not-check result, not a stable metric."
                            ),
                        )
                        alerts.append(alert)
                        self._alerts.append(alert)
                        self._last_alert_time[metric_name] = datetime.now()

                        if self.alert_callback:
                            try:
                                self.alert_callback(alert)
                            except Exception as e:
                                warnings.warn(f"Alert callback failed: {e}")
                    continue

                # Past this point the not-measurable guard above has already
                # established that both values exist and are finite, and has
                # `continue`d otherwise. Bind them as plain floats so the type
                # system knows what that guard proved: mypy cannot carry the
                # narrowing across `not_measurable_reason`, and an `assert` would
                # be stripped under `python -O`, which is precisely the build
                # where a silently-None baseline would do damage.
                baseline_measured = float(baseline_value)  # type: ignore[arg-type]
                current_measured = float(current_value)

                # Check for metric drift
                drift_magnitude = abs(current_measured - baseline_measured)

                if drift_magnitude > self.config.drift_threshold:
                    drift_detected = True
                    self._current_drift_status[metric_name] = True

                    # Check cooldown
                    if self._should_alert(metric_name):
                        alert = self._create_alert(
                            metric_name=metric_name,
                            baseline_value=baseline_measured,
                            current_value=current_measured,
                            drift_magnitude=drift_magnitude,
                            drift_type=DriftType.METRIC_DRIFT,
                        )
                        alerts.append(alert)
                        self._alerts.append(alert)
                        self._last_alert_time[metric_name] = datetime.now()

                        # Trigger callback if configured
                        if self.alert_callback:
                            try:
                                self.alert_callback(alert)
                            except Exception as e:
                                warnings.warn(f"Alert callback failed: {e}")
                else:
                    self._current_drift_status[metric_name] = False
        else:
            # BGL-D (fail closed, door 4): the WARM-UP window. The whole drift
            # check above is gated on a CUMULATIVE sample count, so until
            # `min_samples_for_alert` rows have been seen nothing is compared
            # at all, and the batch used to be reported as a clean one.
            # Measured 2026-09-17 with the DEFAULT config
            # (min_samples_for_alert=100) on a 20 row batch whose two groups
            # were predicted 1.0 and 0.0 against a baseline of 0.0:
            # metrics {'demographic_parity_difference': 1.0},
            # could_not_check [], drift_detected False, repr
            # `MonitoringResult(small: OK, 0 alerts)` and
            # get_summary()['drift_detected'] False. A 1.0 gap, measured and
            # sitting in the same object, reported as OK because the monitor
            # had not yet decided to look at it.
            #
            # The names are recorded so the batch says what it is, and NOTHING
            # else changes: no alert is created (suppressing alerts below this
            # count is the entire purpose of the setting) and
            # `_current_drift_status` is left ABSENT rather than written to
            # None, so `drift_detected()` stays quiet during warm up and the
            # metric claims nothing until it is really compared. That is the
            # same no claim state `update_baseline` leaves behind, and for the
            # same reason: nothing has been compared yet, which is not the
            # same as having tried and failed.
            could_not_check.extend(self.config.metrics_to_monitor)
            if self.config.metrics_to_monitor:
                warnings.warn(
                    f"{batch_id}: no monitored metric was compared against its "
                    f"baseline. {self._total_samples} of the "
                    f"{self.config.min_samples_for_alert} samples required by "
                    f"min_samples_for_alert have been seen, so this batch is a "
                    f"could-not-check, not a clean batch.",
                    UserWarning,
                    stacklevel=2,
                )

        result = MonitoringResult(
            batch_id=batch_id,
            metrics=metrics,
            # Safe side, with the third state kept beside it: a batch carrying a
            # metric nobody could measure is not an OK batch. `could_not_check`
            # is what separates it from a measured drift.
            drift_detected=drift_detected or bool(could_not_check),
            could_not_check=could_not_check,
            alerts=alerts,
            sample_count=len(y_pred),
            group_distribution=group_distribution,
            # F9. Carried, not dropped: a monitoring audit trail that accepts a
            # model version and discards it is a false record. The top-level
            # mapping is copied, so a caller reusing its own dict for the next
            # batch cannot rewrite this record; values nested inside it stay
            # shared with the caller, which is what "verbatim" means here.
            metadata=dict(metadata) if metadata else {},
        )

        return result

    def drift_detected(self) -> bool:
        """Whether the current state needs a human.

        Returns:
            True if any monitored metric is drifting from baseline, OR could
            not be measured at all in the latest window it was checked in.

        The second clause is the fail-closed half, and it is deliberate: this
        boolean is the "is everything fine" question every dashboard and
        runbook asks, and returning False for a metric nobody measured is the
        same false all-clear as approving an unevaluated deployment. It does
        NOT collapse the three states -- :meth:`get_drift_status` still
        distinguishes drifting (True), measured and stable (False) and
        could-not-check (None), and :meth:`unmeasurable_metrics` names the
        third group. This mirrors ``FairnessTestResult.passed``, which is
        False for a SKIPPED test while the status enum keeps all three.
        """
        return any(status is not False for status in self._current_drift_status.values())

    def get_drift_status(self) -> Dict[str, Optional[bool]]:
        """Get drift status for all monitored metrics.

        Returns:
            Dictionary mapping metric names to ``True`` (drifting), ``False``
            (measured against the baseline and within the threshold) or
            ``None`` (could not be checked: the current value or the baseline
            was not a finite number). A metric that has not been seen at all is
            absent from the mapping rather than reported as any of the three.
        """
        return dict(self._current_drift_status)

    def unmeasurable_metrics(self) -> List[str]:
        """Monitored metrics whose drift could NOT be checked.

        Returns:
            The names whose status is could-not-check (``None``). These are
            neither drifting nor stable: nothing was compared.
        """
        return [name for name, status in self._current_drift_status.items() if status is None]

    def get_current_metrics(self) -> Dict[str, Optional[float]]:
        """Get the most recent metric values.

        Returns:
            Dictionary of current metric values.
        """
        return {
            metric: list(history)[-1] if history else None
            for metric, history in self._metric_history.items()
        }

    def get_rolling_average(self) -> Dict[str, Optional[float]]:
        """Get rolling average of metrics over the window.

        Returns:
            Dictionary of rolling average values, in three states per metric:
            a float when EVERY window in the deque carried a measurement,
            ``nan`` when any window did not (the average of a window half of
            which was never measured is not an average, and dropping the gap
            would re-serve the measured half as the whole), and ``None`` when
            the metric has never been seen at all.
        """
        # BGL-3, measured 2026-09-27. The body was
        # ``float(np.mean(list(history))) if history else None``, which is right
        # for the NaN this module records for an unmeasured window (np.mean
        # propagates it, and that refusal is pinned in
        # tests/test_surface_grade_g018.py) and wrong for everything else a
        # custom ``compute_metrics_fn`` can put in the history, because np.mean
        # does not ask whether its input was a measurement:
        #
        #   compute_metrics_fn -> {metric: True}   history [True]
        #       -> rolling average 1.0, a clean float on the disparity scale,
        #          out of a flag. `_triage.is_measured` rejects a bool for this
        #          exact reason and log_batch had ALREADY routed the same value
        #          to drift status None, so the monitor knew. False gives 0.0,
        #          which is PERFECT PARITY.
        #   compute_metrics_fn -> {metric: None}   history [0.5, None]
        #       -> TypeError: unsupported operand type(s) for +: 'float' and
        #          'NoneType', raised out of get_summary() as well, so a
        #          dashboard reporting a could-not-check crashed instead.
        #   compute_metrics_fn -> {metric: inf}    history [inf]
        #       -> rolling average inf, which compares greater than every
        #          threshold a caller could apply to it.
        #
        # One reading of "was this measured", the library's own, applied to
        # every window before any of them is averaged.
        rolling: Dict[str, Optional[float]] = {}
        for metric, history in self._metric_history.items():
            window = list(history)
            if not window:
                rolling[metric] = None
                continue
            if not all(is_measured(v) for v in window):
                rolling[metric] = float("nan")
                continue
            rolling[metric] = float(np.mean([float(v) for v in window]))
        return rolling

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[DriftAlert]:
        """Get alerts, optionally filtered.

        Args:
            since: Only return alerts after this time.
            severity: Only return alerts of this severity.

        Returns:
            List of matching alerts.
        """
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp > since]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def trigger_alert(self, custom_message: Optional[str] = None) -> Optional[DriftAlert]:
        """Manually trigger an alert for current drift status.

        Args:
            custom_message: Optional custom message for the alert.

        Returns:
            The triggered alert, or None when every monitored metric was
            measured and none of them is drifting. A metric that could NOT be
            measured yields a ``DriftType.NOT_MEASURABLE`` alert rather than
            None: could-not-check is a state that needs an operator, and
            answering None for it would make the documented
            ``if drift_detected(): trigger_alert()`` pattern do nothing.
        """
        if not self.drift_detected():
            return None

        # Find the most severe drift
        max_drift_metric = None
        max_drift_magnitude = 0.0

        for metric_name, is_drifted in self._current_drift_status.items():
            if is_drifted:
                current = self.get_current_metrics().get(metric_name)
                baseline = self.baseline_metrics.get(metric_name)

                # `is not None` for the type narrowing, `_is_finite` for the
                # substance: a NaN current value or baseline is not a drift
                # magnitude, and abs(nan - x) would silently rank as 0.
                if (
                    current is not None
                    and baseline is not None
                    and _is_finite(current)
                    and _is_finite(baseline)
                ):
                    drift = abs(current - baseline)
                    # `max_drift_metric is None or ...` rather than a bare
                    # `drift > max_drift_magnitude`, whose 0.0 floor could not
                    # select a metric whose measured magnitude IS 0.0. With a
                    # drift_threshold below zero a gap of exactly 0.0 is
                    # already past the threshold, so the status map said
                    # drifting while this method returned None, which its own
                    # docstring defines as "every monitored metric was
                    # measured and none of them is drifting": the opposite of
                    # what drift_detected() was answering at the same moment.
                    # Measured 2026-09-17 with drift_threshold=-0.1 and a
                    # metric equal to its baseline: get_drift_status()
                    # {'dp': True}, drift_detected() True, trigger_alert()
                    # None. The ranking is unchanged for every other case:
                    # only a strictly larger magnitude replaces the one held.
                    if max_drift_metric is None or drift > max_drift_magnitude:
                        max_drift_magnitude = drift
                        max_drift_metric = metric_name

        if max_drift_metric is None:
            # drift_detected() is True and no MEASURED drift explains it, so the
            # reason is could-not-check. Returning None here (the old
            # behaviour) made the documented pattern
            # `if monitor.drift_detected(): monitor.trigger_alert()` resolve
            # into silence for exactly the state that most needs an operator.
            unmeasurable = self.unmeasurable_metrics()
            if unmeasurable:
                name = unmeasurable[0]
                current = self.get_current_metrics().get(name)
                alert = self._create_alert(
                    metric_name=name,
                    baseline_value=self.baseline_metrics.get(name, float("nan")),
                    current_value=current if current is not None else float("nan"),
                    drift_magnitude=float("nan"),
                    drift_type=DriftType.NOT_MEASURABLE,
                    custom_message=custom_message
                    or (
                        f"{len(unmeasurable)} monitored metric(s) could not be measured "
                        f"({', '.join(unmeasurable)}), so drift could not be checked for "
                        f"them. This is a could-not-check result, not a stable metric."
                    ),
                )
                self._alerts.append(alert)
                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        warnings.warn(f"Alert callback failed: {e}")
                return alert
            return None

        # max_drift_metric is only set for a metric whose current value was
        # non-None in the loop above, so the current value cannot be None here.
        max_drift_current = self.get_current_metrics()[max_drift_metric]
        assert max_drift_current is not None

        alert = self._create_alert(
            metric_name=max_drift_metric,
            baseline_value=self.baseline_metrics[max_drift_metric],
            current_value=max_drift_current,
            drift_magnitude=max_drift_magnitude,
            drift_type=DriftType.METRIC_DRIFT,
            custom_message=custom_message,
        )

        self._alerts.append(alert)

        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")

        return alert

    def update_baseline(self, new_baseline: Dict[str, float]) -> None:
        """Update the baseline metrics.

        Args:
            new_baseline: New baseline metric values.
        """
        self.baseline_metrics.update(new_baseline)
        # Forget the drift status of every re-baselined metric rather than
        # writing False for it. Nothing has been compared against the NEW
        # baseline yet, and False is the positive claim "measured, and stable",
        # which is precisely the fabricated verdict the NaN routing in
        # log_batch exists to prevent. Removing the entry claims nothing: the
        # metric is simply absent from get_drift_status() until the next batch
        # measures it, exactly like a metric never seen. (Deliberately NOT
        # None: None means "we tried and could not", which is also untrue here,
        # and it would make drift_detected() alarm on every re-baseline.)
        for metric in new_baseline:
            self._current_drift_status.pop(metric, None)

    def reset(self) -> None:
        """Reset all monitoring state."""
        self._batch_count = 0
        self._total_samples = 0
        self._alerts = []
        self._last_alert_time = {}
        self._current_drift_status = {}
        self._group_distribution_history.clear()

        for metric in self._metric_history:
            self._metric_history[metric].clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring state.

        Returns:
            Dictionary with monitoring summary. ``drift_detected`` is the
            monitor-level question and is False while nothing has been
            compared yet, so it is served beside two keys that say how much
            WAS compared: ``unmeasurable_metrics`` (tried, could not) and
            ``metrics_never_compared`` (a monitored metric with no status at
            all: either no batch has carried it, or the monitor is still
            below ``min_samples_for_alert`` and has compared nothing). A
            reader must not have to infer that from the sample count and the
            config.
        """
        return {
            "batch_count": self._batch_count,
            "total_samples": self._total_samples,
            "drift_detected": self.drift_detected(),
            "drift_status": self.get_drift_status(),
            "unmeasurable_metrics": self.unmeasurable_metrics(),
            "metrics_never_compared": [
                metric
                for metric in self.config.metrics_to_monitor
                if metric not in self._current_drift_status
            ],
            "current_metrics": self.get_current_metrics(),
            "rolling_average": self.get_rolling_average(),
            "baseline_metrics": self.baseline_metrics,
            "total_alerts": len(self._alerts),
            "config": self.config.to_dict(),
        }

    def _should_alert(self, metric_name: str) -> bool:
        """Check if we should create an alert (respecting cooldown)."""
        last_alert = self._last_alert_time.get(metric_name)

        if last_alert is None:
            return True

        elapsed = (datetime.now() - last_alert).total_seconds()
        return elapsed >= self.config.alert_cooldown_seconds

    def _create_alert(
        self,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        drift_magnitude: float,
        drift_type: DriftType,
        custom_message: Optional[str] = None,
    ) -> DriftAlert:
        """Create a drift alert."""
        # A could-not-check alert carries a NaN magnitude, and every comparison
        # below is False for NaN, so it would land on INFO -- the quietest
        # severity there is, for the one state an operator most needs to see.
        # Graded explicitly instead.
        if drift_type is DriftType.NOT_MEASURABLE:
            severity = AlertSeverity.WARNING
        # Determine severity based on drift magnitude
        elif drift_magnitude > self.config.drift_threshold * 2:
            severity = AlertSeverity.CRITICAL
        elif drift_magnitude > self.config.drift_threshold * 1.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        if custom_message:
            message = custom_message
        else:
            message = (
                f"Fairness drift detected for {metric_name}: "
                f"baseline={baseline_value:.4f}, current={current_value:.4f}, "
                f"drift={drift_magnitude:.4f}"
            )

        alert_id = (
            f"alert_{metric_name}_{self._batch_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        return DriftAlert(
            alert_id=alert_id,
            severity=severity,
            drift_type=drift_type,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            drift_magnitude=drift_magnitude,
            message=message,
            metadata={
                "batch_count": self._batch_count,
                "total_samples": self._total_samples,
            },
        )

    def _compute_default_metrics(
        self,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """Compute default fairness metrics."""
        metrics: Dict[str, float] = {}
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return metrics

        # Worst-case (max - min) gaps across ALL groups. Truncating to the
        # first two groups (previous behaviour) let disparities against a
        # third or later group go unmonitored.
        dp_rates = [float(np.mean(y_pred[protected_attr == g])) for g in groups]
        # A group whose predicted-positive rate is UNDEFINED (every prediction
        # for that group is NaN) leaves no gap to measure, exactly as an
        # undefined TPR does below. Python's ``max``/``min`` are ORDER
        # dependent around NaN: ``max([0.5, nan])`` is 0.5 and
        # ``min([0.5, nan])`` is 0.5, so a NaN anywhere but the FIRST position
        # returned a gap of exactly 0.0, the cleanest value on the scale,
        # invented for a group nobody could measure. Measured 2026-09-17 on
        # 100 rows of group A predicted 1.0 and 100 rows of group B predicted
        # NaN, against a baseline of 0.0:
        # metrics {'demographic_parity_difference': 0.0}, could_not_check [],
        # MonitoringResult(...: OK, 0 alerts), get_drift_status()
        # {'demographic_parity_difference': False} and
        # to_prometheus_metrics() exporting
        # `fairness_demographic_parity_difference 0.0` beside
        # `fairness_drift_demographic_parity_difference 0`. Putting the same
        # NaN in group A instead (first position) already answered NaN, so the
        # monitor's honesty depended on np.unique's ordering. The sibling TPR
        # guard below was written for this exact shape and never mirrored up
        # here.
        if any(r != r for r in dp_rates):  # NaN check
            metrics["demographic_parity_difference"] = float("nan")
        else:
            metrics["demographic_parity_difference"] = max(dp_rates) - min(dp_rates)

        # Additional metrics if y_true is available. A group with no positives
        # has an undefined TPR: report the gap as NaN (unmeasurable this window)
        # rather than silently dropping the group, which would make an
        # unmonitorable window look like a perfectly fair 0.0 gap (mirror of the
        # gate/testing NaN handling and the fd9746b metric fix).
        if y_true is not None:
            # THE UNMEASURABLE LABEL (BGL7 monitor-1/2 sibling, 2026-09-30). The
            # guard below is aimed at a group with NO positive label at all, which
            # leaves an undefined TPR and is caught by ``any(r != r ...)``. A group
            # with SOME unusable labels leaves a perfectly DEFINED TPR that is the
            # rate over its LABELLED rows only, so that guard could not see it:
            # ``y_true == 1`` is False for a NaN label exactly as ``y_pred == 1``
            # was False for a NaN prediction in the demographic-parity arm above.
            # TOTAL label loss was refused, PARTIAL label loss passed.
            #
            # Found by asking what the sibling of the tracker's defect was.
            # ``operations/monitoring/tracker.py`` had the identical hole on
            # ``pos = grp[grp[label] == 1]`` and was fixed the same day; this module
            # is the one whose own comment above claims to have been right about the
            # NaN-PREDICTION case first, and it was wrong about the NaN LABEL in the
            # same function.
            #
            # Measured 2026-09-30: group A 40 label=1 rows with 20 predicted 1 (TPR
            # 0.500) and group B 20 label=1 rows with 10 predicted 1 plus 20 rows
            # whose label is NaN and which ARE predicted 1, so B's TPR over its
            # labelled rows is 0.500 and its real TPR 30/40 = 0.750:
            #
            #     {'demographic_parity_difference': 0.25,
            #      'equalized_odds_difference': 0.0}   warnings []
            #
            # PERFECT equalized odds against a real gap of 0.250, published beside a
            # demographic-parity reading that IS a measurement, so the two disagreed
            # about the same batch. Now nan, which ``log_batch`` routes into
            # ``could_not_check`` and a NOT_MEASURABLE alert, plus this warning for
            # the direct caller.
            unlabelled: Dict[str, int] = {}
            try:
                labels = np.asarray(y_true, dtype=float)
            except (TypeError, ValueError):
                # A label array float() cannot read holds no binary label in any
                # row, so every TPR is already NaN and the guard below refuses it.
                labels = None
            if labels is not None and labels.shape == np.shape(protected_attr):
                unusable = ~np.isin(labels, (0.0, 1.0))
                for g in groups:
                    n_bad = int(np.count_nonzero(unusable & (protected_attr == g)))
                    if n_bad:
                        unlabelled[str(g)] = n_bad

            if unlabelled:
                details = "; ".join(
                    f"{g}: {n} of {int(np.count_nonzero(protected_attr == g))} row(s)"
                    for g, n in sorted(unlabelled.items())
                )
                warnings.warn(
                    f"BiasMonitor: {len(unlabelled)} group(s) hold rows with no usable "
                    f"ground-truth label (not 0 and not 1), so the set of rows the "
                    f"true-positive rate is conditioned on is not the group's "
                    f"positive-label set: {details}. A row whose label is not 0 or 1 is "
                    f"dropped from the positive arm by `y_true == 1`, so the rate would "
                    f"be a rate over the LABELLED rows only. Returning nan for "
                    f"equalized_odds_difference (could not measure), never 0.0, which "
                    f"reads as perfect equalized odds.",
                    UserWarning,
                    stacklevel=2,
                )
                metrics["equalized_odds_difference"] = float("nan")
                return metrics

            tpr_rates = []
            for g in groups:
                pos = (protected_attr == g) & (y_true == 1)
                tpr_rates.append(float(np.mean(y_pred[pos])) if np.sum(pos) > 0 else float("nan"))
            if any(r != r for r in tpr_rates):
                metrics["equalized_odds_difference"] = float("nan")
            elif len(tpr_rates) >= 2:
                metrics["equalized_odds_difference"] = max(tpr_rates) - min(tpr_rates)

        return metrics

    def to_prometheus_metrics(self) -> str:
        """Export current state as Prometheus-compatible metrics.

        Returns:
            Prometheus metrics format string.
        """
        lines = []

        # Current metrics.
        #
        # A value the latest batch could not measure exports as NaN, the same
        # no-data value the drift gauge below uses, rather than as Python's
        # lowercase `nan` repr. It must NOT be dropped from the export either:
        # a silently absent series looks to a dashboard exactly like a monitor
        # that stopped, and the last scraped value goes on being graphed.
        for metric_name, value in self.get_current_metrics().items():
            if value is not None:
                safe_name = metric_name.replace("-", "_")
                lines.append(f"fairness_{safe_name} {value if _is_finite(value) else 'NaN'}")

        # Drift status.
        #
        # `1 if is_drifted else 0` wrote a hard 0 -- "this metric is stable" --
        # for a could-not-check status, because None is falsy. Measured
        # 2026-09-10 with equalized_odds_difference at NaN, this export read
        # `fairness_drift_equalized_odds_difference 0`, a gauge asserting
        # fairness is stable over a metric that was never measured; every alert
        # rule and SLO written against it fires never. NaN is Prometheus's own
        # no-data value: it is not 0, no `== 0` or `> 0` rule matches it, and
        # `absent_over_time` / `changes` style staleness rules do. Three
        # states, three values.
        for metric_name, is_drifted in self._current_drift_status.items():
            safe_name = metric_name.replace("-", "_")
            if is_drifted is None:
                lines.append(f"fairness_drift_{safe_name} NaN")
            else:
                lines.append(f"fairness_drift_{safe_name} {1 if is_drifted else 0}")

        # Counters
        lines.append(f"fairness_monitor_batches_total {self._batch_count}")
        lines.append(f"fairness_monitor_samples_total {self._total_samples}")
        lines.append(f"fairness_monitor_alerts_total {len(self._alerts)}")
        # Explicit count, so a dashboard can show could-not-check without
        # having to parse NaN out of the per-metric gauges above.
        lines.append(f"fairness_monitor_unmeasurable_metrics {len(self.unmeasurable_metrics())}")

        return "\n".join(lines)
