"""BGL g004: operations/monitoring/tracker.py, a value nobody measured.

Six defects, each found by EXECUTION on 2026-09-17 before any edit, each
with a refusal pin and an over-correction control asserting a real measured
value exactly, so a fix that simply refuses everything fails here too.

g004-1. ``TemporalFairnessAnalyzer.detect_trend`` answered ``("stable", 0.0)``
        when the OLS slope was undefined. ``denom`` is the variance of the TIME
        axis, so it is zero exactly when every surviving row shares one calendar
        date, which is what a caller gets by recording more than once a day.
        Fourteen rows all dated 2025-03-03 carrying 0.01 rising to 0.14, a
        fourteen-fold spread, answered ("stable", 0.0); ``get_metric_summary``
        published ``trend_slope`` 0.0, ``forecast_metric`` drew a FLAT three-day
        forecast at 0.14, and ``get_explanation`` reported severity "info" with
        "Continue standard monitoring.".

g004-2. ``detect_weekly_degradation``'s 14-row floor counts ROWS while the check
        compares WEEKDAYS. Fourteen consecutive Mondays of ``demographic_parity``
        at 0.02 with one Monday at 0.50 returned ``(False, 0.0543)``, the verdict
        "no weekday shows systematic degradation" over a week nobody observed. A
        weekly reporting cadence is the realistic way in: it lands every row on
        one weekday.

g004-3. ``compute_equalized_odds`` and ``compute_equal_opportunity`` reach
        ``grp[pred]`` directly, so a missing prediction column raised KeyError
        from inside them. On a frame of ``y_pred`` / ``label`` / ``group_gender``
        left at the default ``prediction_col="prediction"``, ``update_and_check``
        emitted its own warning promising "every fairness metric for this window
        is NaN" and then died with ``KeyError: 'prediction'`` on the next line,
        losing the whole snapshot including the readings it had already taken.

g004-4. ``detect_seasonal_pattern`` returned a bare ``{}`` for an untracked
        metric, and raised TypeError when every stored value was null. A tracked
        metric with data always yields at least one slot, so ``{}`` can only mean
        could-not-check, and it said so to nobody.

g004-5. ``forecast_metric`` anchored on ``last_val = 0.0`` for an untracked
        metric and for an empty analyzer, and raised IndexError when every stored
        value was null. The 0.0 was masked only because ``detect_trend`` answers
        nan for those inputs.

g004-6. ``WindowMetrics.to_dict`` dropped ``any_alert``, so the three-state
        verdict stopped at the JSON boundary and a reader of the serialised form
        had only ``alerts`` left, whose ``any(...)`` is False when empty.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
    mmd_gaussian,
)

METRIC = "demographic_parity"  # lower is better, resolvable by name


def _caught(fn):
    """Run *fn* with warnings ON and record them; a refusal in a warning is
    invisible when warnings are suppressed."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(x.message) for x in w]


def _analyzer(dates, values, lookback_days: int = 400) -> TemporalFairnessAnalyzer:
    a = TemporalFairnessAnalyzer(lookback_days=lookback_days)
    for d, v in zip(dates, values):
        a.update_daily_metrics(pd.Timestamp(d), {METRIC: v})
    return a


def _consecutive(n: int, values, start: str = "2025-01-06"):
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(days=i) for i in range(n)], list(values)


# g004-1 detect_trend: an OLS slope with no time axis


class TestTrendWithNoTimeAxis:
    SAME_DAY_VALUES = [0.01 * (i + 1) for i in range(14)]  # 0.01 .. 0.14

    def _same_day(self) -> TemporalFairnessAnalyzer:
        return _analyzer(["2025-03-03"] * 14, self.SAME_DAY_VALUES)

    def test_it_refuses_instead_of_answering_stable(self):
        # REFUSAL. Measured before the fix: ("stable", 0.0).
        (direction, slope), warned = _caught(lambda: self._same_day().detect_trend(METRIC))

        assert direction == "not_assessed"
        assert math.isnan(slope)
        assert any("no variance" in m and "undefined" in m for m in warned), warned

    def test_the_summary_does_not_publish_the_fabricated_slope(self):
        summary, _ = _caught(lambda: self._same_day().get_metric_summary(METRIC))

        assert summary["trend_direction"] == "not_assessed"
        assert math.isnan(summary["trend_slope"])
        # The descriptive statistics ARE measurable over those rows and must
        # survive: refusing them too would throw away real evidence.
        assert summary["n_days"] == 14
        assert summary["min"] == pytest.approx(0.01)
        assert summary["max"] == pytest.approx(0.14)
        assert summary["mean"] == pytest.approx(np.mean(self.SAME_DAY_VALUES))

    def test_the_forecast_built_on_it_is_not_a_flat_line(self):
        # Measured before the fix: three days all at exactly 0.14.
        forecast, _ = _caught(lambda: self._same_day().forecast_metric(METRIC, days_ahead=3))

        assert len(forecast) == 3
        assert all(math.isnan(v) for _, v in forecast), forecast

    def test_the_explanation_does_not_read_stable(self):
        report, _ = _caught(lambda: self._same_day().get_explanation(METRIC))

        assert "trend=stable" not in report.summary
        assert "not_assessed" in report.summary

    def test_control_a_real_flat_series_is_still_measured_as_stable(self):
        # OVER-CORRECTION CONTROL. Thirty DISTINCT days at a constant 0.3: the
        # time axis varies, the slope really is zero, and this must stay a
        # measurement. Expected value computed independently: a constant y
        # gives an OLS slope of exactly 0.
        dates, values = _consecutive(30, [0.3] * 30)
        (direction, slope), warned = _caught(lambda: _analyzer(dates, values).detect_trend(METRIC))

        assert direction == "stable"
        assert slope == 0.0
        assert warned == []

    def test_control_a_real_rising_series_keeps_its_exact_slope(self):
        # OVER-CORRECTION CONTROL. y = 0.02 + 0.01*day over 30 distinct days, so
        # the OLS slope is exactly 0.01 per day by construction.
        dates, values = _consecutive(30, [0.02 + 0.01 * i for i in range(30)])
        (direction, slope), warned = _caught(lambda: _analyzer(dates, values).detect_trend(METRIC))

        assert direction == "increasing"
        assert slope == pytest.approx(0.01)
        assert warned == []


# g004-2 detect_weekly_degradation: a weekday comparison with one weekday


class TestWeekdayCheckNeedsTwoWeekdays:
    def _mondays(self) -> TemporalFairnessAnalyzer:
        # Fourteen consecutive Mondays; the last one collapses to 0.50.
        dates = [pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * i) for i in range(14)]
        return _analyzer(dates, [0.02] * 13 + [0.50])

    def test_a_weekly_cadence_is_could_not_check_not_stable(self):
        # REFUSAL. Measured before the fix: (False, 0.05428571428571428).
        (degraded, worst), warned = _caught(
            lambda: self._mondays().detect_weekly_degradation(METRIC)
        )

        assert degraded is None
        assert math.isnan(worst)
        assert any("distinct weekday" in m for m in warned), warned

    def test_all_rows_on_one_date_is_could_not_check(self):
        # REFUSAL, the same shape reached a different way.
        analyzer = _analyzer(["2025-03-03"] * 14, [0.01 * (i + 1) for i in range(14)])
        (degraded, worst), warned = _caught(lambda: analyzer.detect_weekly_degradation(METRIC))

        assert degraded is None
        assert math.isnan(worst)
        assert any("distinct weekday" in m for m in warned), warned

    def test_control_a_real_bad_friday_is_still_found_exactly(self):
        # OVER-CORRECTION CONTROL. 28 consecutive days from a Monday: Fridays at
        # 0.40, the rest at 0.05. Independent expectation: 4 of 28 days are
        # Fridays, so the overall mean is (4*0.40 + 24*0.05)/28 = 0.1 and the
        # worst weekday mean is exactly 0.40, which is far above 1.05 * 0.1.
        dates, _ = _consecutive(28, range(28))
        values = [0.40 if d.dayofweek == 4 else 0.05 for d in dates]
        (degraded, worst), _ = _caught(
            lambda: _analyzer(dates, values).detect_weekly_degradation(METRIC)
        )

        assert degraded is True
        assert worst == pytest.approx(0.40)

    def test_control_a_real_clean_week_is_still_a_false_verdict(self):
        # OVER-CORRECTION CONTROL, the other polarity: a check that answers None
        # for everything finds nothing. 28 distinct days, all weekdays present,
        # every value 0.05, so no weekday can sit 5% above the overall mean.
        dates, values = _consecutive(28, [0.05] * 28)
        (degraded, worst), _ = _caught(
            lambda: _analyzer(dates, values).detect_weekly_degradation(METRIC)
        )

        assert degraded is False
        assert worst == pytest.approx(0.05)


# g004-3 the label-dependent metrics and a missing prediction column


def _notebook_shaped_frame() -> pd.DataFrame:
    """y_pred / label / group_gender, i.e. the default prediction_col is absent.

    M is predicted positive 180 of 200, F 20 of 200.
    """
    return pd.DataFrame(
        {
            "y_pred": [1] * 180 + [0] * 20 + [1] * 20 + [0] * 180,
            "label": ([1] * 100 + [0] * 100) * 2,
            "group_gender": ["M"] * 200 + ["F"] * 200,
        }
    )


def _two_arm_frame() -> pd.DataFrame:
    """A frame whose TPR spread and FPR spread are deliberately different.

    Group A: 50 positives of which 45 are predicted 1, 50 negatives of which 10
    are. Group B: 50 positives of which 40 are predicted 1, 50 negatives of
    which 25 are.
    """
    rows = []
    for group, tp, fp in (("A", 45, 10), ("B", 40, 25)):
        rows += [(group, 1, 1)] * tp + [(group, 1, 0)] * (50 - tp)
        rows += [(group, 0, 1)] * fp + [(group, 0, 0)] * (50 - fp)
    return pd.DataFrame(rows, columns=["group_g", "label", "prediction"])


class TestMissingPredictionColumnIsNanNotACrash:
    def test_equalized_odds_refuses_instead_of_raising(self):
        # REFUSAL. Measured before the fix: KeyError: 'prediction'.
        monitor = FairnessMonitor()
        value, warned = _caught(
            lambda: monitor.compute_equalized_odds(_notebook_shaped_frame(), "group_gender")
        )

        assert math.isnan(value)
        assert any("'prediction'" in m and "NOT among" in m for m in warned), warned

    def test_equal_opportunity_refuses_instead_of_raising(self):
        monitor = FairnessMonitor()
        value, warned = _caught(
            lambda: monitor.compute_equal_opportunity(_notebook_shaped_frame(), "group_gender")
        )

        assert math.isnan(value)
        assert any("'prediction'" in m and "NOT among" in m for m in warned), warned

    def test_a_missing_label_column_says_so_rather_than_refusing_silently(self):
        frame = _notebook_shaped_frame().rename(columns={"y_pred": "prediction"})
        frame = frame.drop(columns=["label"])
        monitor = FairnessMonitor()
        value, warned = _caught(lambda: monitor.compute_equalized_odds(frame, "group_gender"))

        assert math.isnan(value)
        assert any("'label'" in m for m in warned), warned

    def test_the_whole_window_survives_the_misconfigured_column(self):
        # The snapshot used to be LOST: update_and_check raised out of the
        # equalized-odds call after it had already measured the others.
        config = FairnessMonitorConfig(
            metrics_to_track=[
                "disparate_impact",
                "demographic_parity",
                "equalized_odds",
                "equal_opportunity",
            ]
        )
        monitor = FairnessMonitor(config=config)
        snap, _ = _caught(lambda: monitor.update_and_check(_notebook_shaped_frame()))

        assert snap.sample_count == 400
        assert all(math.isnan(v) for v in snap.metrics.values()), snap.metrics
        # Nothing was compared to a threshold, so there is no verdict.
        assert snap.alerts == {}
        assert snap.any_alert is None

    def test_control_the_configured_column_still_measures_exactly(self):
        # OVER-CORRECTION CONTROL, on a frame built so the two arms DIFFER, or
        # a single wrong arm would satisfy both assertions at once. Counted by
        # hand from _two_arm_frame: group A is 45 of 50 positives predicted 1
        # (TPR 0.9) and 10 of 50 negatives (FPR 0.2); group B is 40 of 50
        # (TPR 0.8) and 25 of 50 (FPR 0.5). So equal opportunity is the TPR
        # spread 0.9 - 0.8 = 0.1, and equalized odds is the MAX of that and the
        # FPR spread 0.5 - 0.2 = 0.3, i.e. 0.3.
        monitor = FairnessMonitor()
        frame = _two_arm_frame()

        eo, eo_warned = _caught(lambda: monitor.compute_equalized_odds(frame, "group_g"))
        eopp, _ = _caught(lambda: monitor.compute_equal_opportunity(frame, "group_g"))

        assert eo == pytest.approx(0.3)
        assert eopp == pytest.approx(0.1)
        assert eo_warned == []


# g004-4 detect_seasonal_pattern: an empty mapping that said nothing


class TestSeasonalPatternRefusalIsAudible:
    def test_an_untracked_metric_warns_rather_than_returning_a_silent_empty(self):
        dates, values = _consecutive(21, [0.05 + 0.002 * i for i in range(21)])
        pattern, warned = _caught(
            lambda: _analyzer(dates, values).detect_seasonal_pattern("equalized_odds")
        )

        assert pattern == {}
        assert any("not a tracked metric" in m for m in warned), warned

    def test_an_all_null_column_does_not_raise(self):
        # Measured before the fix: TypeError out of the date arithmetic, which
        # the rendering adapter swallows in a bare except.
        dates, values = _consecutive(30, [float("nan")] * 30)
        pattern, warned = _caught(lambda: _analyzer(dates, values).detect_seasonal_pattern(METRIC))

        assert pattern == {}
        assert any("is null" in m for m in warned), warned

    def test_control_a_real_weekly_cycle_is_still_returned_exactly(self):
        # OVER-CORRECTION CONTROL. 28 days from a Monday, value == weekday/10,
        # so slot k must average to exactly k/10.
        dates, _ = _consecutive(28, range(28))
        values = [d.dayofweek / 10.0 for d in dates]
        pattern, warned = _caught(lambda: _analyzer(dates, values).detect_seasonal_pattern(METRIC))

        assert pattern == {k: pytest.approx(k / 10.0) for k in range(7)}
        assert warned == []


# g004-5 forecast_metric: an anchor nobody observed


class TestForecastAnchor:
    def test_an_untracked_metric_forecasts_nan_not_zero(self):
        dates, values = _consecutive(21, [0.05] * 21)
        forecast, warned = _caught(
            lambda: _analyzer(dates, values).forecast_metric("equalized_odds", days_ahead=3)
        )

        assert len(forecast) == 3
        assert all(math.isnan(v) for _, v in forecast), forecast
        assert any("not a tracked metric" in m for m in warned), warned

    def test_an_all_null_column_does_not_raise(self):
        # Measured before the fix: IndexError out of .iloc[-1].
        dates, values = _consecutive(30, [float("nan")] * 30)
        forecast, warned = _caught(
            lambda: _analyzer(dates, values).forecast_metric(METRIC, days_ahead=2)
        )

        assert all(math.isnan(v) for _, v in forecast), forecast
        assert any("every stored value" in m for m in warned), warned

    def test_an_empty_analyzer_forecasts_nan_not_zero(self):
        forecast, warned = _caught(
            lambda: TemporalFairnessAnalyzer().forecast_metric(METRIC, days_ahead=2)
        )

        assert all(math.isnan(v) for _, v in forecast), forecast
        assert any("no daily metrics at all" in m for m in warned), warned

    def test_control_a_real_trend_is_still_extrapolated_exactly(self):
        # OVER-CORRECTION CONTROL. y = 0.02 + 0.01*day over 30 days ends at 0.31
        # with slope exactly 0.01, so day +1 must be 0.32 and day +3 must be 0.34.
        dates, values = _consecutive(30, [0.02 + 0.01 * i for i in range(30)])
        forecast, warned = _caught(
            lambda: _analyzer(dates, values).forecast_metric(METRIC, days_ahead=3)
        )

        assert [v for _, v in forecast] == [
            pytest.approx(0.32),
            pytest.approx(0.33),
            pytest.approx(0.34),
        ]
        # _consecutive starts on 2025-01-06, so day 30 is 2025-02-04.
        assert [d for d, _ in forecast] == [
            pd.Timestamp("2025-02-05"),
            pd.Timestamp("2025-02-06"),
            pd.Timestamp("2025-02-07"),
        ]
        assert warned == []


# g004-6 the verdict at the serialisation boundary


def _clean_batch() -> pd.DataFrame:
    """Both groups selected at exactly 0.5, so DI == 1.0 and DP == 0.0."""
    return pd.DataFrame(
        {
            "prediction": ([1] * 50 + [0] * 50) * 2,
            "label": ([1] * 50 + [0] * 50) * 2,
            "group_g": ["A"] * 100 + ["B"] * 100,
        }
    )


def _unfair_batch() -> pd.DataFrame:
    """A selected 90 of 100, B 10 of 100, so DI == 1/9 and DP == 0.8."""
    return pd.DataFrame(
        {
            "prediction": [1] * 90 + [0] * 10 + [1] * 10 + [0] * 90,
            "label": ([1] * 50 + [0] * 50) * 2,
            "group_g": ["A"] * 100 + ["B"] * 100,
        }
    )


class TestSerialisedVerdictHasThreeStates:
    CONFIG = FairnessMonitorConfig(metrics_to_track=["disparate_impact", "demographic_parity"])

    def test_a_window_that_checked_nothing_serialises_none_not_false(self):
        # REFUSAL. A frame with no protected column at all.
        monitor = FairnessMonitor(config=self.CONFIG)
        snap, warned = _caught(
            lambda: monitor.update_and_check(
                pd.DataFrame({"prediction": [1, 0] * 50, "label": [1, 0] * 50})
            )
        )
        record = snap.to_dict()

        assert "any_alert" in record, sorted(record)
        assert record["any_alert"] is None
        # The collapse this exists to prevent, spelled out.
        assert any(record["alerts"].values()) is False
        assert any("no protected-attribute column" in m for m in warned), warned

    def test_control_a_breaching_window_serialises_true_with_exact_values(self):
        # OVER-CORRECTION CONTROL. DI = 0.1/0.9 = 1/9, DP = 0.9 - 0.1 = 0.8.
        monitor = FairnessMonitor(config=self.CONFIG)
        snap, _ = _caught(lambda: monitor.update_and_check(_unfair_batch()))
        record = snap.to_dict()

        assert record["any_alert"] is True
        assert record["metrics"]["disparate_impact_group_g"] == pytest.approx(1.0 / 9.0)
        assert record["metrics"]["demographic_parity_group_g"] == pytest.approx(0.8)
        assert record["alerts"] == {
            "disparate_impact_group_g": True,
            "demographic_parity_group_g": True,
        }

    def test_control_a_clean_window_serialises_false_with_exact_values(self):
        # OVER-CORRECTION CONTROL, the other polarity. DI = 0.5/0.5 = 1.0.
        monitor = FairnessMonitor(config=self.CONFIG)
        snap, warned = _caught(lambda: monitor.update_and_check(_clean_batch()))
        record = snap.to_dict()

        assert record["any_alert"] is False
        assert record["metrics"]["disparate_impact_group_g"] == pytest.approx(1.0)
        assert record["metrics"]["demographic_parity_group_g"] == pytest.approx(0.0)
        assert warned == []


# g004-7 the severity of a report that measured nothing


class TestTemporalExplanationSeverity:
    """``severity`` is the field a dashboard ranks on, and "info" is what a
    measured-and-clean metric gets. A report in which NOTHING was measured got
    the same "info" while the prose underneath read "mean=not measured,
    trend=not_assessed"."""

    @staticmethod
    def _flat() -> TemporalFairnessAnalyzer:
        dates, values = _consecutive(30, [0.02] * 30)
        return _analyzer(dates, values)

    def test_a_report_that_measured_nothing_is_not_info(self):
        # REFUSAL. Measured before the fix: "info", byte-identical to the
        # severity of the healthy flat series in the control below.
        report, _ = _caught(lambda: self._flat().get_explanation("equalized_odds"))

        assert report.severity != "info"
        assert "mean=not measured" in report.summary
        assert "trend=not_assessed" in report.summary
        assert any("NOT checked" in r for r in report.recommendations), report.recommendations

    def test_control_a_measured_clean_metric_is_still_info(self):
        # OVER-CORRECTION CONTROL. A flat 0.02 over 30 distinct days IS
        # measured, IS clean, and must keep the benign severity.
        report, _ = _caught(lambda: self._flat().get_explanation(METRIC))

        assert report.severity == "info"
        assert "mean=0.0200" in report.summary

    def test_control_a_measured_degrading_metric_is_still_high(self):
        # OVER-CORRECTION CONTROL, the other polarity. demographic_parity
        # rising 0.02 by 0.01 a day is a slope of 0.01, far above the 0.002
        # escalation floor, and lower is better for it.
        dates, values = _consecutive(30, [0.02 + 0.01 * i for i in range(30)])
        report, _ = _caught(lambda: _analyzer(dates, values).get_explanation(METRIC))

        assert report.severity == "high"


# The remaining graded items, held where they already stand


class TestTheOtherGradedSurfaces:
    CONFIG = FairnessMonitorConfig(metrics_to_track=["disparate_impact", "demographic_parity"])

    def test_alert_summary_counts_real_breaches_and_stays_empty_when_clean(self):
        breaching = FairnessMonitor(
            config=FairnessMonitorConfig(
                metrics_to_track=["disparate_impact"], alert_cooldown_seconds=0.0
            )
        )
        for _ in range(3):
            _caught(lambda: breaching.update_and_check(_unfair_batch()))

        assert breaching.get_alert_summary() == {"disparate_impact_group_g": 3}

        clean = FairnessMonitor(config=self.CONFIG)
        _caught(lambda: clean.update_and_check(_clean_batch()))
        assert clean.get_alert_summary() == {}

    def test_an_empty_summary_is_ambiguous_and_the_history_is_the_channel(self):
        # get_alert_summary answers {} both for "compared and clean" and for
        # "never compared", so the per-row determination is what separates them.
        clean = FairnessMonitor(config=self.CONFIG)
        _caught(lambda: clean.update_and_check(_clean_batch()))

        never = FairnessMonitor(config=self.CONFIG)
        _caught(
            lambda: never.update_and_check(
                pd.DataFrame({"y_pred": [1, 0] * 50, "group_g": ["A"] * 50 + ["B"] * 50})
            )
        )

        assert clean.get_alert_summary() == never.get_alert_summary() == {}

        clean_alerts = list(clean.get_metric_history()["alert"])
        never_alerts = list(never.get_metric_history()["alert"])
        assert clean_alerts == [False, False]
        assert never_alerts == [None, None]

    def test_metric_history_carries_the_value_the_count_and_the_determination(self):
        monitor = FairnessMonitor(config=self.CONFIG)
        _caught(lambda: monitor.update_and_check(_unfair_batch()))
        history = monitor.get_metric_history()

        assert list(history["metric"]) == [
            "disparate_impact_group_g",
            "demographic_parity_group_g",
        ]
        assert list(history["value"]) == [pytest.approx(1.0 / 9.0), pytest.approx(0.8)]
        assert list(history["sample_count"]) == [200, 200]
        assert list(history["alert"]) == [True, True]

    def test_current_metrics_and_window_are_snapshots_of_the_last_window(self):
        monitor = FairnessMonitor(config=self.CONFIG)
        _caught(lambda: monitor.update_and_check(_unfair_batch()))

        current = monitor.get_current_metrics()
        assert current["disparate_impact_group_g"] == pytest.approx(1.0 / 9.0)
        # A copy: mutating it must not reach the stored snapshot.
        current["disparate_impact_group_g"] = 999.0
        assert monitor.get_current_metrics()["disparate_impact_group_g"] == pytest.approx(1.0 / 9.0)

        window = monitor.get_window_df()
        assert window.shape == (200, 3)
        window.loc[0, "prediction"] = 42
        assert monitor.get_window_df().loc[0, "prediction"] == 1

        # A could-not-check window must reach the accessor as NaN, not as a
        # number: this is the surface a caller reads the reading off.
        blind = FairnessMonitor(config=self.CONFIG)
        _caught(
            lambda: blind.update_and_check(
                pd.DataFrame({"y_pred": [1, 0] * 50, "group_g": ["A"] * 50 + ["B"] * 50})
            )
        )
        blind_metrics = blind.get_current_metrics()
        assert set(blind_metrics) == {
            "disparate_impact_group_g",
            "demographic_parity_group_g",
        }
        assert all(math.isnan(v) for v in blind_metrics.values()), blind_metrics

    def test_set_reference_stores_a_copy_and_reset_clears_everything(self):
        monitor = FairnessMonitor(config=self.CONFIG)
        reference = _clean_batch()
        assert monitor.set_reference(reference) is None
        reference.loc[0, "prediction"] = 42
        assert monitor._reference_df.loc[0, "prediction"] == 1

        _caught(lambda: monitor.update_and_check(_unfair_batch()))
        assert monitor.reset() is None
        assert monitor.get_current_metrics() == {}
        assert monitor.get_alert_summary() == {}
        assert monitor.get_window_df().empty
        assert monitor._reference_df is None

    def test_monitor_explanation_never_reads_clean_when_nothing_was_checked(self):
        monitor = FairnessMonitor(config=self.CONFIG)
        _caught(
            lambda: monitor.update_and_check(
                pd.DataFrame({"prediction": [1, 0] * 50, "label": [1, 0] * 50})
            )
        )
        report, _ = _caught(monitor.get_explanation)

        assert report.severity != "info"
        clean = FairnessMonitor(config=self.CONFIG)
        _caught(lambda: clean.update_and_check(_clean_batch()))
        assert _caught(clean.get_explanation)[0].severity == "info"

    def test_metric_summary_is_exact_when_measured_and_empty_when_not(self):
        # Independently: 0.02 .. 0.31 over 30 distinct days has min 0.02,
        # max 0.31, mean 0.165 and n_days 30.
        dates, values = _consecutive(30, [0.02 + 0.01 * i for i in range(30)])
        analyzer = _analyzer(dates, values)
        summary, _ = _caught(lambda: analyzer.get_metric_summary(METRIC))

        assert summary["n_days"] == 30
        assert summary["min"] == pytest.approx(0.02)
        assert summary["max"] == pytest.approx(0.31)
        assert summary["mean"] == pytest.approx(0.165)
        assert summary["trend_slope"] == pytest.approx(0.01)

        # Could-not-check is an EMPTY dict, never a dict of zeros.
        assert _caught(lambda: analyzer.get_metric_summary("equalized_odds"))[0] == {}

        # A column that exists but holds nothing keeps its keys and says nan.
        null_dates, null_values = _consecutive(30, [float("nan")] * 30)
        null_summary, _ = _caught(
            lambda: _analyzer(null_dates, null_values).get_metric_summary(METRIC)
        )
        assert null_summary["n_days"] == 0
        assert math.isnan(null_summary["mean"])
        assert math.isnan(null_summary["min"])

    def test_analyzer_accessors_report_what_was_recorded(self):
        dates, values = _consecutive(10, [0.05] * 10)
        analyzer = _analyzer(dates, values)

        assert analyzer.get_tracked_metrics() == [METRIC]
        frame = analyzer.to_dataframe()
        assert frame.shape == (10, 2)
        frame.loc[0, METRIC] = 42.0
        assert analyzer.to_dataframe().loc[0, METRIC] == pytest.approx(0.05)

        assert TemporalFairnessAnalyzer().get_tracked_metrics() == []
        assert TemporalFairnessAnalyzer().to_dataframe().empty

    def test_update_daily_metrics_prunes_by_the_lookback_window(self):
        analyzer = TemporalFairnessAnalyzer(lookback_days=5)
        for i in range(10):
            analyzer.update_daily_metrics(
                pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), {METRIC: float(i)}
            )

        frame = analyzer.to_dataframe()
        assert list(frame[METRIC]) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_mmd_refuses_out_loud_and_still_measures_a_real_shift(self):
        # REFUSAL: no finite value on either side is could-not-check, and 0.0
        # would read as "the two distributions are identical".
        value, warned = _caught(lambda: mmd_gaussian(np.full(20, np.nan), np.arange(20.0)))
        assert math.isnan(value)
        assert any("finite value" in m for m in warned), warned

        # CONTROL: identical samples give exactly 0, and a real shift is a real
        # positive number, computed independently for a two-point case:
        # xx = yy = 1, xy = exp(-25/2), so mmd = 2 - 2*exp(-12.5).
        same, same_warned = _caught(lambda: mmd_gaussian(np.arange(50.0), np.arange(50.0)))
        assert same == 0.0
        assert same_warned == []

        shifted, _ = _caught(lambda: mmd_gaussian(np.array([0.0]), np.array([5.0])))
        assert shifted == pytest.approx(2.0 - 2.0 * math.exp(-12.5))
