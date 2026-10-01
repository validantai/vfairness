"""BGL6 PINS, batch A-operations-3-b (2026-09-29): reporting/store.py.

Three rows an independent examiner overturned a SECOND time, after the
2026-09-27 wave had already fixed and graded them. Each pin below carries the
number THAT audit measured on the fixed code, with the over-correction control
beside it: a fix that refuses everything passes every refusal test.

Two root causes:

1. An alert record that cannot be DATED. ``ingest_alert`` read its timestamp
   with ``d.get("timestamp", datetime.now())``, and ``.get`` with a default does
   NOT fire for a key that is PRESENT holding a null, so an alert built from a
   pandas row with a missing timestamp cell was stored carrying ``NaT``.
   ``get_alerts`` then dropped it from every window in silence
   (``pd.Timestamp(NaT) >= start`` is False and ``<= end`` is False too), and
   ``compute_health_score`` published a perfect alert_frequency 100.0 for a
   window whose alert count could not be established. Both 2026-09-27 guards
   missed it because they ask a STORE-level question (was this channel ever fed)
   about a WINDOW measurement. Rows: get_alerts, compute_health_score.
2. The substituted per-group size was published as a VERIFIED privacy tier
   whenever ``enable_privacy`` was False, because the relabel and the query-time
   disclosure both lived inside ``_apply_privacy``, which returns at its first
   line for such a store. ``ingest_dataframe`` accepts only 'exact' rows, so the
   caveat could be laundered off by a round trip. Row: ingest_window_metrics.

Read with tests/test_bgl5_operations_3.py, whose store section fixed the first
half of both of these.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any, List, Optional

import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import WindowMetrics
from vfairness.operations.reporting.store import (
    MetricsStore,
    MetricsStoreConfig,
    PrivacyLevel,
    StoredMetricRecord,
)


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# ===========================================================================
# ROOT CAUSES 1 and 2. An alert that cannot be placed in a window.
# ===========================================================================


def _undatable_alert() -> dict:
    """The shape the auditor used, and the ordinary one: an alert dict built from
    a pandas row whose timestamp cell is missing. ``pd.NaT`` is what
    ``to_dict('records')`` produces for it, and ``.get("timestamp", default)``
    does not fire for a key that is present holding it."""
    return pd.DataFrame(
        [
            {
                "metric_name": "demographic_parity",
                "severity": "HIGH",
                "priority_score": 9.0,
                "message": "gap",
                "timestamp": pd.NaT,
            }
        ]
    ).to_dict("records")[0]


def _store(n_breach: int, n_clean: int, *, drift: Optional[List[Any]] = None) -> MetricsStore:
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    now = datetime.now()
    i = 0
    for value, flag, count in ((0.95, True, n_breach), (0.01, False, n_clean)):
        for _ in range(count):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now - timedelta(hours=i),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=value,
                    group="gender",
                    group_size=500,
                    alert=flag,
                )
            )
            i += 1
    for j, verdict in enumerate(drift or []):
        store._drift_records.append(
            {
                "timestamp": now - timedelta(hours=j),
                "metric": "demographic_parity",
                "overall_drift_score": 0.1,
                "drift_detected": verdict,
            }
        )
    return store


def test_an_alert_with_an_unreadable_timestamp_is_refused_at_ingest():
    """REFUSAL PIN on MetricsStore.ingest_alert, the root cause of the two rows
    below. Measured before, on the dict in ``_undatable_alert``::

        ingest_alert returned: 1   warnings: []
        stored alert ts: NaT

    so a HIGH-severity, priority-9.0 alert entered the store carrying a
    timestamp that no comparison can place, and nothing said so. The alert is
    still KEPT (dropping an unresolved HIGH alert would be worse), it is stored
    with no time rather than with a substituted one, and it carries its own
    caveat in the record a consumer reads."""
    store = _store(1, 1)
    n, warned = _caught(store.ingest_alert, _undatable_alert())
    assert n == 1, "the alert must not be silently dropped either"
    assert store._alert_records[0]["timestamp"] is None
    assert store._alert_records[0]["timestamp_measured"] is False
    assert any("cannot be read as a time" in w for w in warned), warned
    # The mirrored metric row keeps the store's RECEIPT time, so it does not
    # silently leave every windowed get_metrics frame as well, and says so.
    mirror = store.get_latest("alert_priority")
    assert mirror is not None and mirror.timestamp is not None
    assert mirror.metadata["alert_time_measured"] is False


def test_an_undatable_alert_no_longer_leaves_a_window_in_silence():
    """REFUSAL PIN on MetricsStore.get_alerts. Measured before, on a store
    holding two metric records and that one ingested alert::

        get_alerts()                    -> 1 record
        get_alerts(now - 7d, now + 1d)  -> 0 records, warnings: []

    An empty list is read as "no alert matched" by every consumer
    (reports.py:935/977/1063, dashboard.py:1006/1595, compute_health_score), so
    a window containing a HIGH-severity, priority-9.0 alert reported clean."""
    store = _store(1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_alert(_undatable_alert())

    unwindowed, unwindowed_warned = _caught(store.get_alerts)
    assert len(unwindowed) == 1, "no window was asked for, so the record is returned"
    assert unwindowed_warned == []

    now = datetime.now()
    windowed, warned = _caught(
        store.get_alerts, start_time=now - timedelta(days=7), end_time=now + timedelta(days=1)
    )
    assert windowed == []
    assert any("no readable timestamp" in w and "COULD NOT" in w for w in warned), warned
    assert any("HIGH" in w for w in warned), warned


def test_an_undatable_alert_no_longer_lifts_the_health_score():
    """REFUSAL PIN on MetricsStore.compute_health_score. Measured before, on one
    breaching record in two plus that one ingested alert::

        score 37.5  components {'metric_compliance': 0.0,
                                'alert_frequency': 100.0}

    with no alert-coverage warning at all: byte-identical to the numbers the
    2026-09-27 fix claims to have eliminated, over a store holding an unresolved
    HIGH-severity alert. ``alert_measured = bool(self._alert_records)`` is a
    STORE-level fact and the component is a WINDOW measurement."""
    store = _store(1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_alert(_undatable_alert())

    health, warned = _caught(store.compute_health_score)
    assert health.components == {"metric_compliance": pytest.approx(0.0)}
    assert "alert_frequency" not in health.components
    assert health.score == pytest.approx(0.0)
    assert health.status == "red"
    assert any(
        "no readable timestamp" in w and "EXCLUDED from the health score" in w for w in warned
    ), warned
    assert "Alert frequency was NOT measured" in health.explanation
    assert "no readable timestamp" in health.explanation


def test_the_withheld_branch_publishes_no_alert_component_for_an_undatable_alert():
    """The same defect in the branch that exists to refuse it. With every metric
    record unmeasurable the composite is withheld, and its `components` dict was
    built from `if self._alert_records`, the store-level test again, so it
    published alert_frequency 100.0 for a window whose count was unknown."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    store._records.append(
        StoredMetricRecord(
            timestamp=datetime.now(),
            source="FairnessMonitor",
            metric_name="custom_metric",
            value=float("nan"),
            group="gender",
            group_size=500,
            alert=None,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_alert(_undatable_alert())

    health, _ = _caught(store.compute_health_score)
    assert health.score is None and health.status == "not_assessed"
    assert health.components == {}
    assert "no readable timestamp" in health.explanation


def test_control_a_datable_alert_still_scores_the_documented_composite():
    """OVER-CORRECTION CONTROL, the one that matters most here: a store whose
    alerts all carry a readable time still gets the documented 50/30/20 number.

    Four clean metric records (metric_compliance 100.0), one alert with a real
    timestamp (alert_frequency 90.0) and two drift verdicts, one drifting
    (drift_stability 50.0)::

        0.50 * 100 + 0.30 * 90 + 0.20 * 50 = 87.0
    """
    store = _store(0, 4, drift=[True, False])
    n, ingest_warned = _caught(
        store.ingest_alert,
        {"timestamp": datetime.now(), "severity": "HIGH", "priority_score": 9.0},
    )
    assert n == 1
    assert ingest_warned == [], "a readable timestamp must not be warned about"

    health, warned = _caught(store.compute_health_score)
    assert health.components["metric_compliance"] == pytest.approx(100.0)
    assert health.components["alert_frequency"] == pytest.approx(90.0)
    assert health.components["drift_stability"] == pytest.approx(50.0)
    assert health.score == pytest.approx(87.0)
    assert health.status == "green"
    assert not any("no readable timestamp" in w for w in warned), warned
    assert "Alert frequency was NOT measured" not in health.explanation


def test_control_a_datable_alert_is_returned_by_its_own_window():
    """OVER-CORRECTION CONTROL on get_alerts, both directions: the alert IS
    returned for a window that contains it, in silence, and is absent in silence
    from a window that does not. Neither is a refusal."""
    store = _store(0, 2)
    fired = datetime.now() - timedelta(hours=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_alert({"timestamp": fired, "severity": "HIGH", "priority_score": 9.0})

    now = datetime.now()
    inside, inside_warned = _caught(
        store.get_alerts, start_time=now - timedelta(days=1), end_time=now
    )
    assert len(inside) == 1 and inside[0]["priority_score"] == pytest.approx(9.0)
    assert inside_warned == []

    outside, outside_warned = _caught(
        store.get_alerts, start_time=now + timedelta(days=1), end_time=now + timedelta(days=2)
    )
    assert outside == []
    assert outside_warned == [], "a measured absence stays silent"


def test_control_the_withheld_branch_still_reports_a_measured_alert_count():
    """OVER-CORRECTION CONTROL on the withheld branch: with the metric evidence
    absent but the alert channel fed with a DATABLE alert, alert_frequency is
    still published with its real number, 100 - 1*10 = 90.0. The refusal must
    not swallow the one component that does have evidence."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    store._records.append(
        StoredMetricRecord(
            timestamp=datetime.now(),
            source="FairnessMonitor",
            metric_name="custom_metric",
            value=float("nan"),
            group="gender",
            group_size=500,
            alert=None,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_alert({"timestamp": datetime.now(), "severity": "HIGH", "priority_score": 9.0})

    health, _ = _caught(store.compute_health_score)
    assert health.score is None and health.status == "not_assessed"
    assert health.components == {"alert_frequency": pytest.approx(90.0)}
    assert "Alert frequency IS measured (1 alert(s) in the window)" in health.explanation


# ===========================================================================
# ROOT CAUSE 3. The substituted group size was published as a verified tier
# whenever the privacy switch was off.
# ===========================================================================


def _window_snapshot() -> WindowMetrics:
    return WindowMetrics(
        batch_id="b1",
        timestamp=datetime.now(),
        sample_count=500,
        metrics={"demographic_parity_gender": 0.04},
        group_rates={"gender": {"F": 0.31, "M": 0.33}},
        alerts={"demographic_parity_gender": False},
        mmd_scores={"gender": 0.02},
        excluded_groups={},
    )


def test_a_privacy_disabled_store_no_longer_calls_a_substituted_size_exact():
    """REFUSAL PIN on MetricsStore.ingest_window_metrics / get_metrics. The
    2026-09-27 relabel lives inside ``_apply_privacy``, which returns at its
    first line when ``config.enable_privacy`` is False. Measured before, with
    the identical 500-row window into
    ``MetricsStore(MetricsStoreConfig(enable_privacy=False))``::

        privacy_level {'exact'}  measured {False}
        ingest warns: []
        query warns: []

    'exact' is defined in PrivacyLevel as ``group_size >= noisy_threshold``, a
    k-anonymity statement about the GROUP, and the size stored beside a
    per-group rate is the whole window's."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())

    frame, query_warned = _caught(store.get_metrics)
    rates = frame[frame["metric_name"] == "positive_rate"]
    assert set(rates["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}
    assert set(rates["group_size_measured"]) == {False}
    assert any("DIFFERENT population" in w for w in query_warned), query_warned
    assert any("enable_privacy=False" in w for w in query_warned), query_warned


def test_the_round_trip_can_no_longer_launder_the_caveat_with_privacy_off():
    """The consequence the auditor measured. ``ingest_dataframe`` refuses rows
    whose privacy_level is not 'exact', so the tier is the gate. Measured before
    with enable_privacy=False: ``ingested 2 of 2, warnings=[]``, and the second
    store's rows read ``privacy_level exact, group_size 0, group_size_measured
    None``: the caveat gone for good, in a store that a consumer cannot tell
    from a verified one."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())
        frame = store.get_metrics()
    rates = frame[frame["metric_name"] == "positive_rate"]
    assert len(rates) == 2

    clone = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    ingested, warned = _caught(
        clone.ingest_dataframe, rates, metric_col="metric_name", group_col="group"
    )
    assert ingested == 0, f"the laundering round trip accepted {ingested} of 2"
    assert any("did not release" in w or "k-anonymity" in w for w in warned), warned


def test_control_a_privacy_disabled_store_still_releases_every_value_exactly():
    """OVER-CORRECTION CONTROL. ``enable_privacy=False`` is documented as "all
    queries return exact values", and it still does: the two rates come back as
    the caller's own numbers, not withheld, not noised. What changed is the
    GUARANTEE the row claims, which nothing verified under either setting."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())
        frame = store.get_metrics()
    rates = frame[frame["metric_name"] == "positive_rate"]
    assert sorted(rates["value"]) == pytest.approx([0.31, 0.33])
    assert rates["value"].notna().all()


def test_control_the_window_level_rows_keep_their_exact_tier_with_privacy_off():
    """OVER-CORRECTION CONTROL, the other half of the same record: a
    window-level metric and an MMD score ARE computed over the whole window, so
    its count is THEIRS, group_size_measured is True and the tier stays exact.
    The fix must qualify only the substituted rows."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())
        frame = store.get_metrics()
    window_rows = frame[frame["metric_name"].isin(["demographic_parity", "mmd_score"])]
    assert len(window_rows) == 2
    assert set(window_rows["group_size_measured"]) == {True}
    assert set(window_rows["privacy_level"]) == {PrivacyLevel.EXACT.value}
    assert window_rows["value"].notna().all()


def test_control_a_real_group_size_still_releases_as_exact_with_privacy_off():
    """OVER-CORRECTION CONTROL. A row ingested with its OWN group size still
    reads 'exact' and 'measured', with no warning: the refusal is keyed on the
    substitution, not on the privacy switch being off."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    ingested, _ = _caught(
        store.ingest_dataframe,
        pd.DataFrame(
            {
                "timestamp": [datetime.now()] * 2,
                "metric": ["demographic_parity"] * 2,
                "value": [0.31, 0.33],
                "group": ["F", "M"],
                "group_size": [240, 260],
            }
        ),
        group_col="group",
        group_size_col="group_size",
    )
    assert ingested == 2
    frame, warned = _caught(store.get_metrics)
    assert set(frame["privacy_level"]) == {PrivacyLevel.EXACT.value}
    assert sorted(frame["value"]) == pytest.approx([0.31, 0.33])
    assert not any("DIFFERENT population" in w for w in warned), warned
