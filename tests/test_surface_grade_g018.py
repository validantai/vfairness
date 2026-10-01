"""Surface grading batch g018: ``operations/cicd/monitor.py`` (BiasMonitor).

Two fabrications were proved by execution at the public entry point
(``BiasMonitor.log_batch`` and the surfaces that read its state) and fixed
here. Both produced the SAFEST looking value on the scale for a comparison
that never happened.

1. An undefined per-group rate became a gap of exactly 0.0, and only for some
   group orderings. ``_compute_default_metrics`` built the demographic parity
   gap as ``max(dp_rates) - min(dp_rates)``, and Python's ``max``/``min`` walk
   past NaN instead of propagating it: ``max([0.8, nan])`` is 0.8 and
   ``min([0.8, nan])`` is 0.8. Measured 2026-09-17 on 100 rows of group A
   predicted positive at 0.8 and 100 rows of group B whose predictions were
   all NaN, against a baseline of 0.0:
   ``metrics {'demographic_parity_difference': 0.0}``, ``could_not_check []``,
   ``MonitoringResult(...: OK, 0 alerts)``,
   ``get_drift_status() {'demographic_parity_difference': False}``,
   ``monitor.drift_detected() False`` and ``to_prometheus_metrics()``
   exporting ``fairness_demographic_parity_difference 0.0`` beside
   ``fairness_drift_demographic_parity_difference 0``: perfect parity,
   invented, over a group nobody could measure. The same NaN placed in group A
   instead already answered NaN, so the monitor's honesty depended on
   ``np.unique``'s ordering. The sibling TPR guard three lines below was
   written for this exact shape and had never been mirrored up.

2. The warm-up window reported a measured gap as a clean batch. The whole
   drift check is gated on the CUMULATIVE ``min_samples_for_alert`` count, and
   below it nothing is compared at all. Measured 2026-09-17 with the DEFAULT
   config on a 20 row batch carrying a 1.0 gap against a baseline of 0.0:
   ``metrics {'demographic_parity_difference': 1.0}``, ``could_not_check []``,
   ``drift_detected False``, ``MonitoringResult(small: OK, 0 alerts)``. The
   gap was measured, was sitting in the same object, and the batch was
   reported OK because the monitor had not yet decided to look at it.

Every assertion below is on what a caller gets back. Each defect is paired
with a control proving healthy data is still measured exactly.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.operations.cicd.monitor import (
    AlertSeverity,
    BiasMonitor,
    DriftType,
    MonitorConfig,
)

METRIC = "demographic_parity_difference"

# Hand-computed, not copied from the code: group A is predicted positive 80
# times in 100 (rate 0.8), group B 30 times in 100 (rate 0.3), so the
# worst-case gap is 0.5 and the whole batch is 200 rows, which clears the
# default min_samples_for_alert of 100.
EXPECTED_GAP = 0.5


def _two_group_batch():
    y_pred = np.concatenate(
        [np.array([1] * 80 + [0] * 20, dtype=float), np.array([1] * 30 + [0] * 70, dtype=float)]
    )
    y_true = np.concatenate([np.array([1] * 50 + [0] * 50)] * 2)
    protected = np.array(["A"] * 100 + ["B"] * 100)
    return y_pred, y_true, protected


def _rate_undefined_for(group):
    """The same batch with every prediction for one group unreadable (NaN), so
    that group has no defined predicted-positive rate and no gap EXISTS."""
    y_pred, y_true, protected = _two_group_batch()
    y_pred = y_pred.copy()
    y_pred[protected == group] = np.nan
    return y_pred, y_true, protected


def _monitor(baseline=0.0, **config_kwargs):
    config = MonitorConfig(metrics_to_monitor=[METRIC], **config_kwargs)
    return BiasMonitor(baseline_metrics={METRIC: baseline}, config=config)


def _log(monitor, batch, batch_id):
    """Log a batch with warnings recorded, never suppressed: a refusal carried
    only in a warning is invisible otherwise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = monitor.log_batch(*batch, batch_id=batch_id)
    return result, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Defect 1: an undefined group rate is not a gap of zero
# ---------------------------------------------------------------------------


class TestUndefinedGroupRateIsNotPerfectParity:
    @pytest.mark.parametrize("undefined_group", ["A", "B"])
    def test_the_gap_is_refused_whatever_the_group_order(self, undefined_group):
        """Before: group B (last in np.unique order) gave 0.0; group A gave
        NaN. The answer must not depend on which group was unreadable."""
        monitor = _monitor(baseline=0.0)
        result, _ = _log(monitor, _rate_undefined_for(undefined_group), "undefined_rate")

        assert math.isnan(result.metrics[METRIC]), (
            f"a gap of {result.metrics[METRIC]!r} was invented for a group with no defined rate"
        )
        assert result.could_not_check == [METRIC]
        assert result.drift_detected is True
        assert "COULD NOT CHECK" in repr(result)
        assert result.to_dict()["could_not_check"] == [METRIC]

    def test_the_monitor_level_surfaces_all_say_could_not_check(self):
        monitor = _monitor(baseline=0.0)
        _log(monitor, _rate_undefined_for("B"), "undefined_rate")

        assert monitor.get_drift_status() == {METRIC: None}
        assert monitor.unmeasurable_metrics() == [METRIC]
        assert monitor.drift_detected() is True
        summary = monitor.get_summary()
        assert summary["drift_detected"] is True
        assert summary["unmeasurable_metrics"] == [METRIC]
        current = summary["current_metrics"][METRIC]
        assert current is not None and math.isnan(current)
        rolling = summary["rolling_average"][METRIC]
        assert rolling is not None and math.isnan(rolling)

    def test_prometheus_exports_no_data_not_a_zero(self):
        monitor = _monitor(baseline=0.0)
        _log(monitor, _rate_undefined_for("B"), "undefined_rate")
        exported = monitor.to_prometheus_metrics().splitlines()

        assert f"fairness_{METRIC} NaN" in exported
        assert f"fairness_{METRIC} 0.0" not in exported
        assert f"fairness_drift_{METRIC} NaN" in exported
        assert f"fairness_drift_{METRIC} 0" not in exported
        assert "fairness_monitor_unmeasurable_metrics 1" in exported

    def test_the_alert_channel_carries_the_third_state(self):
        monitor = _monitor(baseline=0.0)
        result, _ = _log(monitor, _rate_undefined_for("B"), "undefined_rate")

        assert len(result.alerts) == 1
        alert = result.alerts[0]
        assert alert.drift_type is DriftType.NOT_MEASURABLE
        assert alert.severity is AlertSeverity.WARNING
        assert math.isnan(alert.drift_magnitude)
        # A reader of the serialised alert can tell "not measurable" from a
        # measured magnitude of 0.0 without opening the source.
        assert alert.to_dict()["drift_type"] == "not_measurable"
        assert math.isnan(alert.to_dict()["drift_magnitude"])
        assert monitor.get_alerts(severity=AlertSeverity.CRITICAL) == []
        assert len(monitor.get_alerts(severity=AlertSeverity.WARNING)) == 1

        manual = monitor.trigger_alert()
        assert manual is not None
        assert manual.drift_type is DriftType.NOT_MEASURABLE

    # ---------------- CONTROLS: healthy data is still measured --------------

    def test_control_a_real_gap_is_still_measured_exactly(self):
        monitor = _monitor(baseline=0.0)
        result, _ = _log(monitor, _two_group_batch(), "measured")

        assert result.metrics[METRIC] == pytest.approx(EXPECTED_GAP)
        assert result.could_not_check == []
        assert result.drift_detected is True
        assert "DRIFT DETECTED" in repr(result)
        assert monitor.get_drift_status() == {METRIC: True}
        assert monitor.unmeasurable_metrics() == []
        alert = result.alerts[0]
        assert alert.drift_type is DriftType.METRIC_DRIFT
        assert alert.drift_magnitude == pytest.approx(EXPECTED_GAP)

    def test_control_a_measured_stable_batch_is_still_ok(self):
        monitor = _monitor(baseline=EXPECTED_GAP)
        result, _ = _log(monitor, _two_group_batch(), "stable")

        assert result.metrics[METRIC] == pytest.approx(EXPECTED_GAP)
        assert result.could_not_check == []
        assert result.drift_detected is False
        assert "OK" in repr(result)
        assert monitor.get_drift_status() == {METRIC: False}
        assert monitor.drift_detected() is False
        assert monitor.trigger_alert() is None
        assert f"fairness_drift_{METRIC} 0" in monitor.to_prometheus_metrics().splitlines()


# ---------------------------------------------------------------------------
# Defect 2: the warm-up window compared nothing and said OK
# ---------------------------------------------------------------------------


def _small_batch():
    """20 rows, a fully measured gap of 1.0, far below the default
    min_samples_for_alert of 100."""
    y_pred = np.concatenate([np.ones(10), np.zeros(10)])
    y_true = np.concatenate([np.array([1] * 5 + [0] * 5)] * 2)
    protected = np.array(["A"] * 10 + ["B"] * 10)
    return y_pred, y_true, protected


class TestWarmUpWindowIsNotAnOkBatch:
    def test_a_batch_compared_against_nothing_is_could_not_check(self):
        monitor = BiasMonitor(baseline_metrics={METRIC: 0.0})  # DEFAULT config
        result, caught = _log(monitor, _small_batch(), "warm_up")

        # The gap itself was measured and is still reported.
        assert result.metrics[METRIC] == pytest.approx(1.0)
        # Before: could_not_check [], drift_detected False, repr "warm_up: OK".
        assert result.could_not_check == [METRIC]
        assert result.drift_detected is True
        assert "COULD NOT CHECK" in repr(result)
        assert result.to_dict()["could_not_check"] == [METRIC]
        assert any("min_samples_for_alert" in message for message in caught), caught

    def test_the_alert_suppression_the_setting_exists_for_is_untouched(self):
        """Fail closed, stay quiet: the batch says it compared nothing, and no
        alert is raised, because suppressing alerts below this sample count is
        the whole purpose of ``min_samples_for_alert``."""
        monitor = BiasMonitor(baseline_metrics={METRIC: 0.0})
        result, _ = _log(monitor, _small_batch(), "warm_up")

        assert result.alerts == []
        assert monitor.get_alerts() == []
        assert monitor.drift_detected() is False
        assert monitor.trigger_alert() is None

    def test_the_summary_says_the_metric_was_never_compared(self):
        """``drift_detected False`` must not be readable as "compared and
        fine" while the monitor is still warming up."""
        monitor = BiasMonitor(baseline_metrics={METRIC: 0.0})
        _log(monitor, _small_batch(), "warm_up")
        summary = monitor.get_summary()

        assert summary["drift_status"] == {}
        assert summary["metrics_never_compared"] == [METRIC]
        assert summary["unmeasurable_metrics"] == []
        # Nothing is claimed about the metric on the Prometheus drift gauge
        # either: no series at all, which is Prometheus's own no-data state.
        assert f"fairness_drift_{METRIC} 0" not in monitor.to_prometheus_metrics().splitlines()

    # ---------------- CONTROLS --------------------------------------------

    def test_control_once_enough_samples_arrive_the_comparison_happens(self):
        monitor = BiasMonitor(baseline_metrics={METRIC: 0.0})
        _log(monitor, _small_batch(), "warm_up")
        result, caught = _log(monitor, _two_group_batch(), "measured")

        assert result.could_not_check == []
        assert result.metrics[METRIC] == pytest.approx(EXPECTED_GAP)
        assert monitor.get_drift_status() == {METRIC: True}
        assert monitor.get_summary()["metrics_never_compared"] == []
        assert not any("min_samples_for_alert" in message for message in caught), caught

    def test_control_a_monitor_over_the_threshold_never_warms_up(self):
        """The 200 row batch clears the default minimum on its own, so the
        warm-up branch must not fire for it at all."""
        monitor = BiasMonitor(baseline_metrics={METRIC: EXPECTED_GAP})
        result, caught = _log(monitor, _two_group_batch(), "big_enough")

        assert result.could_not_check == []
        assert result.drift_detected is False
        assert caught == []


# ---------------------------------------------------------------------------
# The state-carrying surfaces, graded in the same run
# ---------------------------------------------------------------------------


class TestStateSurfacesKeepTheThirdState:
    def test_rolling_average_refuses_a_window_holding_an_unmeasured_batch(self):
        """A window of [0.5, unmeasured] has no average. Dropping the NaN
        would re-serve 0.5 as the average of a window half of which was never
        measured."""
        monitor = _monitor(baseline=EXPECTED_GAP)
        _log(monitor, _two_group_batch(), "b1")
        _log(monitor, _rate_undefined_for("B"), "b2")

        rolling = monitor.get_rolling_average()[METRIC]
        assert rolling is not None
        assert math.isnan(rolling), f"the unmeasured window was averaged away: {rolling!r}"
        current = monitor.get_current_metrics()[METRIC]
        assert current is not None and math.isnan(current)

    def test_control_rolling_average_of_two_measured_windows_is_the_real_mean(self):
        monitor = _monitor(baseline=EXPECTED_GAP)
        _log(monitor, _two_group_batch(), "b1")
        _log(monitor, _two_group_batch(), "b2")
        # Both windows measured 0.5, so the mean is 0.5, computed here and not
        # copied from the code.
        assert monitor.get_rolling_average()[METRIC] == pytest.approx(EXPECTED_GAP)
        assert monitor.get_current_metrics()[METRIC] == pytest.approx(EXPECTED_GAP)

    def test_a_metric_never_seen_is_absent_rather_than_stable(self):
        monitor = _monitor(baseline=0.0)
        assert monitor.get_drift_status() == {}
        assert monitor.get_current_metrics() == {METRIC: None}
        assert monitor.get_rolling_average() == {METRIC: None}
        assert monitor.get_alerts() == []
        assert monitor.trigger_alert() is None
        assert f"fairness_drift_{METRIC} 0" not in monitor.to_prometheus_metrics().splitlines()

    def test_update_baseline_forgets_the_old_verdict_rather_than_clearing_it(self):
        monitor = _monitor(baseline=0.0)
        _log(monitor, _two_group_batch(), "b1")
        assert monitor.get_drift_status() == {METRIC: True}

        monitor.update_baseline({METRIC: EXPECTED_GAP})
        # Not False: nothing has been compared against the NEW baseline yet.
        assert monitor.get_drift_status() == {}
        assert monitor.get_summary()["metrics_never_compared"] == [METRIC]
        assert monitor.drift_detected() is False

        result, _ = _log(monitor, _two_group_batch(), "b2")
        assert result.could_not_check == []
        assert monitor.get_drift_status() == {METRIC: False}

    def test_reset_returns_to_the_never_seen_state(self):
        monitor = _monitor(baseline=0.0)
        _log(monitor, _two_group_batch(), "b1")
        assert monitor.get_alerts()

        monitor.reset()

        assert monitor.get_drift_status() == {}
        assert monitor.get_current_metrics() == {METRIC: None}
        assert monitor.get_rolling_average() == {METRIC: None}
        assert monitor.get_alerts() == []
        assert monitor.drift_detected() is False
        exported = monitor.to_prometheus_metrics().splitlines()
        assert "fairness_monitor_batches_total 0" in exported
        assert "fairness_monitor_samples_total 0" in exported
        assert not any(line.startswith(f"fairness_{METRIC} ") for line in exported)

    def test_config_to_dict_round_trips_the_thresholds_a_reader_needs(self):
        config = MonitorConfig(drift_threshold=0.07, min_samples_for_alert=5, window_size=3)
        as_dict = BiasMonitor(config=config).get_summary()["config"]
        assert as_dict["drift_threshold"] == 0.07
        assert as_dict["min_samples_for_alert"] == 5
        assert as_dict["window_size"] == 3
        assert as_dict["metrics_to_monitor"] == [METRIC]


# ---------------------------------------------------------------------------
# Defect 3: drift_detected() True and trigger_alert() None at the same moment
# ---------------------------------------------------------------------------


class TestTriggerAlertAgreesWithDriftDetected:
    def test_a_drifting_metric_whose_magnitude_is_zero_still_alerts(self):
        """The ranking loop started at a magnitude of 0.0 and only replaced it
        on a STRICTLY larger one, so a metric the status map calls drifting
        with a measured magnitude of exactly 0.0 selected nothing and
        ``trigger_alert()`` returned None: by its own docstring, "every
        monitored metric was measured and none of them is drifting", stated
        while ``drift_detected()`` was answering True. Reachable with a
        drift_threshold below zero, where a gap of 0.0 is already past it."""
        config = MonitorConfig(
            drift_threshold=-0.1, metrics_to_monitor=[METRIC], min_samples_for_alert=1
        )
        monitor = BiasMonitor(
            baseline_metrics={METRIC: 0.5},
            compute_metrics_fn=lambda y_true, y_pred, attr: {METRIC: 0.5},
            config=config,
        )
        monitor.log_batch(np.ones(4), np.ones(4), np.array(["A", "A", "B", "B"]), batch_id="b1")

        assert monitor.get_drift_status() == {METRIC: True}
        assert monitor.drift_detected() is True
        alert = monitor.trigger_alert()
        assert alert is not None, "drift_detected() said True and the alert channel said nothing"
        assert alert.drift_type is DriftType.METRIC_DRIFT
        assert alert.drift_magnitude == pytest.approx(0.0)

    @pytest.mark.parametrize("order", [["dp_small", "dp_large"], ["dp_large", "dp_small"]])
    def test_control_the_largest_measured_drift_still_wins(self, order):
        """Over-correction control, both declaration orders: the ranking must
        keep choosing the worst drift, not merely the first one."""
        config = MonitorConfig(
            drift_threshold=0.05, metrics_to_monitor=order, min_samples_for_alert=1
        )
        monitor = BiasMonitor(
            baseline_metrics={"dp_small": 0.0, "dp_large": 0.0},
            compute_metrics_fn=lambda y_true, y_pred, attr: {"dp_small": 0.2, "dp_large": 0.7},
            config=config,
        )
        monitor.log_batch(np.ones(4), np.ones(4), np.array(["A", "A", "B", "B"]), batch_id="rank")

        alert = monitor.trigger_alert()
        assert alert is not None
        assert alert.metric_name == "dp_large"
        assert alert.drift_magnitude == pytest.approx(0.7)
