"""BGL7 PIN, batch B4-rep-pre-w2 (2026-09-29): operations/reporting/store.py.

Three findings, one root cause: a WINDOW question answered from a STORE fact, or
from no fact at all.

1. ``MetricsStore.compute_health_score``, the SIBLING of the alert half that
   tests/test_bgl6_operations_3_store.py pinned hours earlier. That fix taught
   ``alert_measured`` the difference between "this store has alert records" and
   "this window's alert count is known", and left the identical hole on
   ``metric_compliance``, the 50 percent component, having fixed the 30 percent
   one. MEASURED BEFORE THIS CHANGE, six metric records through the PUBLIC
   ``ingest_dataframe`` (2 datable clean rows, 4 breaching demographic_parity 0.95
   with alert=True) plus one datable HIGH-severity alert, on
   ``MetricsStoreConfig(enable_privacy=False)``::

       breaches with timestamp pd.NaT -> score 96.2 GREEN
           components {'metric_compliance': 100.0, 'alert_frequency': 90.0}
           n_metrics 3, nothing counted or named the 4 with no window position,
           get_metrics(window) returned 3 of the 7 stored rows,
           and ingest_dataframe emitted ZERO warnings
       the same six rows, datable     -> score 33.8 RED
           components {'metric_compliance': 0.0, 'alert_frequency': 90.0}

2. ``get_metrics``' row builder read ``UNKNOWN_SIZE if group_size_measured is
   False else EXACT``, so the THIRD state (``None``, "the producing path did not
   say") was laundered into a tier that asserts a k-anonymity check. MEASURED
   BEFORE: two rows ingested with no ``group_size_col`` published
   ``privacy_level 'exact'`` with ``group_size 0`` and zero warnings, and the round
   trip through ``ingest_dataframe`` ACCEPTED 2 of 2.

3. ``get_alerts`` with an unreadable CALLER bound (``start_time=pd.NaT``) returned
   ``[]`` with zero warnings over a store holding a real datable HIGH alert. Its
   three existing guards all ask about the STORE or the RECORDS, so none could see
   a defect in the QUESTION. MEASURED BEFORE, and the same for both siblings::

       get_alerts(start_time=pd.NaT)        -> 0 alert(s), warnings=0
       get_metrics(start_time=pd.NaT)       -> 0 row(s),   warnings=0
       get_drift_history(start_time=pd.NaT) -> 0 row(s),   warnings=0
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import pandas as pd
import pytest

from vfairness.operations.reporting.store import (
    MetricsStore,
    MetricsStoreConfig,
    PrivacyLevel,
)

NOW = datetime.now()


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def _six(*, undatable: bool, breaching: bool = True) -> pd.DataFrame:
    """The auditor's frame: 2 clean rows plus 4 rows that may be breaching, whose
    timestamp may be unreadable."""
    ts = pd.NaT if undatable else NOW - timedelta(hours=2)
    rows = [
        {
            "timestamp": NOW - timedelta(hours=1),
            "metric": "demographic_parity",
            "value": 0.02,
            "alert": False,
            "alert_determined": True,
        }
        for _ in range(2)
    ]
    rows += [
        {
            "timestamp": ts,
            "metric": "demographic_parity",
            "value": 0.95 if breaching else 0.02,
            "alert": bool(breaching),
            "alert_determined": True,
        }
        for _ in range(4)
    ]
    return pd.DataFrame(rows)


def _store(frame: pd.DataFrame) -> tuple[MetricsStore, list[str]]:
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    _n, warned = _caught(store.ingest_dataframe, frame, alert_col="alert")
    store.ingest_alert(
        {
            "timestamp": NOW - timedelta(hours=3),
            "severity": "HIGH",
            "metric": "demographic_parity",
            "priority_score": 9.0,
        }
    )
    return store, warned


# ===========================================================================
# 1. metric_compliance over records the window could not place
# ===========================================================================


def test_undatable_breaches_no_longer_publish_a_measured_metric_compliance():
    """BEFORE: score 96.2 GREEN, metric_compliance 100.0, zero disclosure of the
    four breaching records the window could not place."""
    store, _ingest_warned = _store(_six(undatable=True))
    health, warned = _caught(store.compute_health_score)

    assert health.score is None, f"score {health.score!r} over four unplaceable breaches"
    assert health.status == "not_assessed", health.status
    assert "metric_compliance" not in health.components, health.components
    assert health.n_undatable_metrics == 4, health.n_undatable_metrics
    # The count reaches a reader through the object AND its dict form.
    assert health.to_dict()["n_undatable_metrics"] == 4

    named = [w for w in warned if "no readable timestamp" in w and "metric record" in w]
    assert len(named) == 1, warned
    assert "4 of 7 metric record(s)" in named[0], named[0]
    # How many of them recorded a breach is the readable half of the refusal.
    assert "4 of them carry alert=True" in named[0], named[0]
    # The explanation is what the dashboards and reports render.
    assert "could not be placed" in health.explanation
    assert "not a metric compliance of 100" in health.explanation


def test_the_alert_component_that_does_have_evidence_is_still_reported():
    """A component with evidence is present holding its number and one without is
    ABSENT, the rule the other withheld branches of this method already follow. The
    single datable HIGH alert is real evidence and costs 10 points."""
    store, _ = _store(_six(undatable=True))
    health, _warned = _caught(store.compute_health_score)

    assert health.components == {"alert_frequency": 90.0}, health.components
    assert health.n_alerts == 1


def test_ingest_dataframe_says_so_at_the_door():
    """BEFORE: ``timestamp=row[timestamp_col]`` with no readability test at all, so
    six records went in and the store emitted ZERO warnings. The record is still
    stored (its value and its determination are real, and compute_health_score has
    to see it in order to refuse); what was missing was anyone saying so."""
    _store_obj, warned = _store(_six(undatable=True))

    named = [w for w in warned if "ingest_dataframe" in w and "readable value" in w]
    assert len(named) == 1, warned
    assert "4 of 6 row(s)" in named[0], named[0]
    assert "NO POSITION IN TIME" in named[0], named[0]


def test_control_the_same_six_rows_datable_still_score_their_real_red():
    """OVER-CORRECTION CONTROL, asserting the healthy case's REAL number. The
    breaches must still be graded when they CAN be placed: score 33.8 RED with
    metric_compliance 0.0 and alert_frequency 90.0, exactly as measured."""
    store, ingest_warned = _store(_six(undatable=False))
    health, _warned = _caught(store.compute_health_score)

    assert health.score == 33.8, health.score
    assert health.status == "red"
    assert health.components["metric_compliance"] == 0.0, health.components
    assert health.components["alert_frequency"] == 90.0, health.components
    assert health.n_undatable_metrics == 0
    assert [w for w in ingest_warned if "readable value" in w] == []


def test_control_a_clean_datable_store_still_scores_its_real_green():
    """OVER-CORRECTION CONTROL. The fabricated number this defect published was
    96.2 GREEN. A store that HAS EARNED 96.2 must still get it: six clean datable
    records with one alert in the window, metric_compliance 100.0, alert_frequency
    90.0, drift absent, 0.50/0.30 renormalised = 96.2.

    A withholding rule that fired on a healthy store would pass every assertion
    above and destroy the score."""
    store, _ = _store(_six(undatable=False, breaching=False))
    health, _warned = _caught(store.compute_health_score)

    assert health.score == 96.2, health.score
    assert health.status == "green"
    assert health.components["metric_compliance"] == 100.0, health.components
    assert health.n_undatable_metrics == 0


def test_control_an_empty_store_still_gives_its_own_reason():
    """The guard sits ABOVE the len(df) == 0 branch so that a store whose records
    are all undatable is refused with the REAL reason. A store that is genuinely
    empty must still get the OLD reason, not the new one."""
    empty = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    health, warned = _caught(empty.compute_health_score)

    assert health.score is None and health.status == "not_assessed"
    assert health.n_undatable_metrics == 0
    assert "No metric records were found" in health.explanation, health.explanation
    assert any("No metric records in the window" in w for w in warned), warned


# ===========================================================================
# 2. The third state of group_size_measured, laundered as a verified tier
# ===========================================================================

_TWO_ROWS = pd.DataFrame(
    [
        {"timestamp": NOW, "metric": "demographic_parity", "value": 0.31, "grp": "F"},
        {"timestamp": NOW, "metric": "demographic_parity", "value": 0.33, "grp": "M"},
    ]
)


def test_a_row_with_no_recorded_size_is_no_longer_published_as_exact():
    """BEFORE: privacy_level 'exact' with group_size 0 and zero warnings. 'exact'
    is a claim that a k-anonymity check was made; nothing recorded a size to make
    it with, which is verbatim what UNKNOWN_SIZE is documented to mean."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    store.ingest_dataframe(_TWO_ROWS, group_col="grp")
    out, warned = _caught(store.get_metrics)

    assert set(out["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}, out["privacy_level"]
    assert set(out["group_size"]) == {0}
    assert set(out["group_size_measured"]) == {None}
    named = [w for w in warned if "no recorded group size" in w]
    assert len(named) == 1, warned
    assert "2 of 2 record(s)" in named[0], named[0]
    assert "COULD NOT CHECK" in named[0]


def test_the_round_trip_republishes_the_caveat_instead_of_laundering_it():
    """BEFORE: the round trip ACCEPTED 2 of 2 and the clone re-exported them as
    'exact', so the caveat was laundered off for good.

    It is accepted STILL, and that is correct: the caveat of these rows IS the
    absent size, the absent size travels, and the clone publishes UNKNOWN_SIZE
    again. Refusing them would lose a real alert determination, which is what the
    READINESS-6 control in tests/test_readiness5_store5.py asserts must survive.
    What must never happen is the clone reading 'exact'."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    store.ingest_dataframe(_TWO_ROWS, group_col="grp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exported = store.get_metrics()

    clone = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    accepted, _warned = _caught(
        clone.ingest_dataframe,
        exported,
        metric_col="metric_name",
        group_col="group",
        alert_col="alert",
    )
    assert accepted == 2, accepted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        round_tripped = clone.get_metrics()
    assert set(round_tripped["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}, round_tripped[
        "privacy_level"
    ]


def test_control_a_recorded_size_is_still_released_as_exact():
    """OVER-CORRECTION CONTROL, asserting the real values. A store configured with
    enable_privacy=False releases every value exactly, which is what that switch
    means, and a row that DOES carry a recorded size still reads 'exact' with no
    warning. Calling the full k-anonymity classifier here would stamp 'noisy' and
    'suppressed' on values nothing noised or suppressed, so it deliberately is not
    called: the row builder claims only that a size was recorded and released."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    sized = pd.DataFrame(
        [
            {"timestamp": NOW, "metric": "dp", "value": 0.31, "grp": "F", "n": 500},
            {"timestamp": NOW, "metric": "dp", "value": 0.33, "grp": "M", "n": 25},
            {"timestamp": NOW, "metric": "dp", "value": 0.35, "grp": "X", "n": 3},
        ]
    )
    store.ingest_dataframe(sized, group_col="grp", group_size_col="n")
    out, warned = _caught(store.get_metrics)

    assert set(out["privacy_level"]) == {PrivacyLevel.EXACT.value}, out["privacy_level"]
    assert sorted(round(float(v), 2) for v in out["value"]) == [0.31, 0.33, 0.35]
    assert warned == [], warned


def test_control_a_substituted_size_is_still_refused_by_the_round_trip():
    """OVER-CORRECTION CONTROL for the OTHER route to UNKNOWN_SIZE. A row marked
    group_size_measured=False carries a real size measured for a DIFFERENT
    population, so its caveat is NOT the absent size and nothing in the ingested
    columns carries it. It must stay refused, exactly as BGL5 A-operations-3-b
    made it."""
    hand_built = pd.DataFrame(
        [
            {
                "timestamp": NOW,
                "metric_name": "positive_rate",
                "value": 0.31,
                "group": "F",
                "group_size": 500,
                "privacy_level": PrivacyLevel.UNKNOWN_SIZE.value,
                "alert": False,
                "alert_determined": True,
                "group_size_measured": False,
            }
        ]
    )
    clone = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    accepted, warned = _caught(
        clone.ingest_dataframe,
        hand_built,
        metric_col="metric_name",
        group_col="group",
        alert_col="alert",
    )
    assert accepted == 0, accepted
    assert any("did not release" in w for w in warned), warned


# ===========================================================================
# 3. An unreadable CALLER bound emptied every windowed reader in silence
# ===========================================================================


def _fed_store() -> MetricsStore:
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(_six(undatable=False), alert_col="alert")
    store.ingest_alert(
        {
            "timestamp": NOW - timedelta(hours=3),
            "severity": "HIGH",
            "metric": "demographic_parity",
            "priority_score": 9.0,
        }
    )
    return store


@pytest.mark.parametrize("reader", ["get_alerts", "get_metrics", "get_drift_history"])
@pytest.mark.parametrize("bound", ["start_time", "end_time"])
@pytest.mark.parametrize(
    "bad",
    [pd.NaT, float("nan"), pd.NA, "banana"],
    ids=["pd-NaT", "float-nan", "pd-NA", "unparseable-str"],
)
def test_an_unreadable_window_bound_is_refused_not_answered_with_nothing(reader, bound, bad):
    """BEFORE: 0 results and zero warnings from all three readers, over a store
    holding a real datable HIGH alert. ``ts >= pd.Timestamp(pd.NaT)`` is False and
    ``ts <= NaT`` is False as well, so the whole result left through a comparison
    that could not have said anything else.

    Every dtype is checked, because the four ways a missing time reaches a caller
    are not the same object and only one of them was reported."""
    store = _fed_store()
    with pytest.raises(ValueError) as excinfo:
        getattr(store, reader)(**{bound: bad})

    message = str(excinfo.value)
    assert reader in message, message
    assert bound in message, message
    assert "cannot be read as a time" in message, message
    assert "COULD NOT CHECK" in message, message


def test_control_the_readers_still_answer_a_real_window_and_no_window():
    """OVER-CORRECTION CONTROL, asserting the real counts. A refusal that also
    refused a legitimate bound would pass every assertion above and break every
    consumer in reports.py and dashboard.py, all of which pass ``now - window``."""
    store = _fed_store()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        windowed = store.get_alerts(start_time=NOW - timedelta(days=7), end_time=NOW)
        unbounded = store.get_alerts()
        rows = store.get_metrics(start_time=NOW - timedelta(days=7), end_time=NOW)
        all_rows = store.get_metrics()
        drift = store.get_drift_history(start_time=NOW - timedelta(days=7))

    assert len(windowed) == 1, windowed
    assert len(unbounded) == 1
    assert len(rows) == 7, len(rows)
    assert len(all_rows) == 7
    assert len(drift) == 0

    # And the composite still grades, which is the proof the refusal did not leak
    # into compute_health_score's own windowed calls.
    health, _warned = _caught(store.compute_health_score)
    assert health.score == 33.8, health.score
