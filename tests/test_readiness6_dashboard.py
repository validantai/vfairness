"""READINESS-6: the dashboard collapsed three states back into two.

The producers were fixed in earlier waves. The monitor returns
``Optional[bool]`` for drift, the store records ``alert=None`` with a separate
``alert_determined`` column, and ``compute_health_score`` answers the string
``"unknown"`` for a trend it could not compute and carries ``n_not_assessable``.

Every one of those distinctions died at the chart. Measured on the real rendered
figures before this fix:

* a drift window that COULD NOT BE CHECKED was painted ``#059669``, the same
  PASS green as a measured "no drift";
* an ``"unknown"`` trend produced indicator value ``0`` and delta ``0``,
  identical to a measured ``"stable"``;
* "0 Active Alerts" rendered in PASS green over a window where metrics could not
  be assessed at all;
* a point whose alert state was never determined was drawn as an ordinary
  unmarked point, identical to a measured "no alert".

This is the finding class the register calls "a correct measurement no reader
can see", pointing the other way: a correct COULD-NOT-CHECK that no reader can
see. Every producer-side detector passes, because the producers are right.

These tests read the rendered plotly figure, not the helper functions. A chart's
colours are the only thing a reader gets, so the assertion has to be made where
the reader is.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from vfairness.operations.reporting.dashboard import DashboardConfig, FairnessDashboard
from vfairness.operations.reporting.store import MetricsStore

CFG = DashboardConfig()


class _DriftResult:
    """The shape MetricsStore.ingest_drift_result reads."""

    def __init__(self, timestamp, score, detected):
        self.timestamp = timestamp
        self.overall_drift_score = score
        self.drift_detected = detected
        self.metric = "demographic_parity_difference"
        self.mmd_score = 0.0
        self.worst_scale = None


def _drift_store():
    store = MetricsStore()
    t0 = datetime.now() - timedelta(days=3)
    for i, (score, detected) in enumerate([(0.05, False), (0.42, True), (0.42, None)]):
        store.ingest_drift_result(_DriftResult(t0 + timedelta(days=i), score, detected))
    return store


def _metric_store(alerts, determined):
    store = MetricsStore()
    t0 = datetime.now() - timedelta(days=len(alerts))
    store.ingest_dataframe(
        pd.DataFrame(
            {
                "timestamp": [t0 + timedelta(days=i) for i in range(len(alerts))],
                "metric": ["demographic_parity_difference"] * len(alerts),
                "value": [0.05] * len(alerts),
                "alert": alerts,
                "alert_determined": determined,
                "group_size": [500] * len(alerts),
            }
        ),
        alert_col="alert",
        alert_determined_col="alert_determined",
        group_size_col="group_size",
    )
    return store


# ------------------------------------------------------------ the drift timeline


def test_an_unchecked_drift_window_is_not_painted_the_same_as_a_clean_one():
    fig = FairnessDashboard(_drift_store(), CFG).create_drift_timeline()
    measured_false, measured_true, could_not_check = list(fig.data[0].marker.color)

    assert measured_false == CFG.color_pass
    assert measured_true == CFG.color_fail
    assert could_not_check == CFG.color_unknown, (
        f"a drift window that could not be checked was painted {could_not_check}; "
        f"PASS green is {CFG.color_pass} and this must not equal it"
    )
    assert could_not_check != measured_false, "could-not-check is indistinguishable from clean"


def test_the_chart_says_what_the_grey_means():
    """OVER-CORRECTION CONTROL of a different kind: a neutral colour that nobody
    can interpret is not a disclosure. The count and the meaning go on the
    chart, because a reader does not have the palette."""
    fig = FairnessDashboard(_drift_store(), CFG).create_drift_timeline()
    texts = [a.text or "" for a in fig.layout.annotations]
    disclosure = [t for t in texts if "could not be checked" in t]
    assert disclosure, f"no disclosure annotation among {texts}"
    assert "1 of 3" in disclosure[0], disclosure[0]
    assert "not 'no drift'" in disclosure[0], disclosure[0]


def test_a_fully_measured_timeline_carries_no_disclosure():
    """OVER-CORRECTION CONTROL. A clean run must read exactly as it did before."""
    store = MetricsStore()
    t0 = datetime.now() - timedelta(days=2)
    store.ingest_drift_result(_DriftResult(t0, 0.05, False))
    store.ingest_drift_result(_DriftResult(t0 + timedelta(days=1), 0.42, True))

    fig = FairnessDashboard(store, CFG).create_drift_timeline()
    assert list(fig.data[0].marker.color) == [CFG.color_pass, CFG.color_fail]
    assert not [a for a in fig.layout.annotations if "could not be checked" in (a.text or "")]


# ------------------------------------------------------------ the alert markers


def test_an_undetermined_alert_gets_its_own_marker():
    fig = FairnessDashboard(_metric_store([False, True, None], [True, True, False]), CFG)
    figure = fig.create_trend_analysis("demographic_parity_difference")
    names = [t.name for t in figure.data]

    assert any("(alert)" in (n or "") for n in names), names
    assert any("not determined" in (n or "") for n in names), (
        f"the point whose alert state was never determined is drawn as an ordinary "
        f"point, identical to a measured 'no alert'. Traces: {names}"
    )
    undetermined = [t for t in figure.data if "not determined" in (t.name or "")][0]
    assert len(undetermined.x) == 1
    assert undetermined.marker.color == CFG.color_unknown


def test_an_undetermined_point_is_not_counted_as_an_alert():
    """The other direction: marking it must not promote it to a finding."""
    fig = FairnessDashboard(_metric_store([False, True, None], [True, True, False]), CFG)
    figure = fig.create_trend_analysis("demographic_parity_difference")
    alert_trace = [t for t in figure.data if (t.name or "").endswith("(alert)")][0]
    assert len(alert_trace.x) == 1, "the undetermined point was counted as an alert"


def test_a_fully_determined_series_draws_no_undetermined_trace():
    """OVER-CORRECTION CONTROL."""
    fig = FairnessDashboard(_metric_store([False, True], [True, True]), CFG)
    figure = fig.create_trend_analysis("demographic_parity_difference")
    names = [t.name or "" for t in figure.data]
    assert not [n for n in names if "not determined" in n], names


# ------------------------------------------------------------ the executive KPIs


def _indicator(fig, title_startswith):
    for d in fig.data:
        if getattr(d, "type", "") == "indicator" and (d.title.text or "").startswith(
            title_startswith
        ):
            return d
    raise AssertionError(f"no indicator titled {title_startswith!r}")


@pytest.mark.parametrize("trend", ["improving", "stable", "degrading"])
def test_a_measured_trend_still_renders_its_number_and_delta(trend):
    """OVER-CORRECTION CONTROL."""
    store = _metric_store([False, False], [True, True])
    base = store.compute_health_score()
    store.compute_health_score = lambda *a, **k: replace(base, trend=trend)

    indicator = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Trend")
    assert indicator.mode == "number+delta"
    assert indicator.value == {"improving": 1, "stable": 0, "degrading": -1}[trend]


def test_an_unknown_trend_is_not_rendered_as_stable():
    """`.get(hs.trend, 0)` mapped "unknown" onto 0, which is the VALUE of
    "stable". Only the suffix text differed, and an indicator is read as a
    number."""
    store = _metric_store([False, False], [True, True])
    base = store.compute_health_score()
    store.compute_health_score = lambda *a, **k: replace(base, trend="unknown")

    unknown = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Trend")
    store.compute_health_score = lambda *a, **k: replace(base, trend="stable")
    stable = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Trend")

    assert unknown.value != stable.value or unknown.mode != stable.mode, (
        "an unknown trend renders identically to a measured stable one"
    )
    assert unknown.mode == "number", "a delta asserts a comparison that was never made"
    assert unknown.number.font.color == CFG.color_unknown
    assert "not determined" in (unknown.number.prefix or "")


def test_zero_alerts_over_unassessable_metrics_is_not_painted_green():
    """ "0 Active Alerts" in PASS green is a claim that nothing is wrong. Zero
    alerts out of nothing examined is not an all-clear."""
    store = _metric_store([False, None], [True, False])
    health = store.compute_health_score()
    assert health.n_not_assessable > 0, "fixture did not produce an unassessable metric"

    indicator = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Active Alerts")
    assert indicator.number.font.color == CFG.color_unknown, (
        f"painted {indicator.number.font.color}; PASS green is {CFG.color_pass}"
    )
    assert "not assessable" in (indicator.title.text or "")


def test_zero_alerts_over_fully_assessed_metrics_is_still_green():
    """OVER-CORRECTION CONTROL. A genuinely clean window must still read clean,
    or the fix has only moved the lie."""
    store = _metric_store([False, False], [True, True])
    health = store.compute_health_score()
    assert health.n_not_assessable == 0 and health.n_alerts == 0

    indicator = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Active Alerts")
    assert indicator.number.font.color == CFG.color_pass
    assert "not assessable" not in (indicator.title.text or "")


def test_a_real_alert_still_reads_red():
    """OVER-CORRECTION CONTROL, the case that matters most."""
    store = _metric_store([True, None], [True, False])
    base = store.compute_health_score()
    store.compute_health_score = lambda *a, **k: replace(base, n_alerts=1, n_not_assessable=1)

    indicator = _indicator(FairnessDashboard(store, CFG).create_executive_view(), "Active Alerts")
    assert indicator.number.font.color == CFG.color_fail, (
        "an unassessable metric alongside a REAL alert downgraded the alert to grey"
    )
