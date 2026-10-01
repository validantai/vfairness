"""Readiness 5, lane store5: a green health score built on rows nobody compared
to a threshold, and a drift verdict the library REFUSED to give graded as 0.

Both findings are the same defect: an unmeasured value replaced by a neutral
default and then graded, weighted and colour-banded as if it were a
measurement. Both fixes follow the R-2 convention already in
``operations/reporting/store.py``: three states, never two. The unmeasured
value is ``None``, it leaves every aggregate, it is named in a warning, and a
verdict that has no evidence left becomes ``None`` rather than a number.

Reproduced by execution on 2026-09-10, before the fix:

- store.py compute_health_score, component 3 and its previous-window twin.
  ``drift_df["drift_detected"].mean()`` over a single REFUSED verdict is NaN,
  and ``max(0.0, 100.0 * (1.0 - nan))`` is 0.0 because NaN loses every
  comparison. Two clean, fully determined metric records plus one drift test
  the detector explicitly refused to run printed
  ``Health Score 80.0 GREEN`` with ``drift_stability 0.0`` and no warning.
  The trend inherited it: a previous window whose drift was refused scored 20
  points lower, so a flat pair of windows read ``improving``, slope +10.0.

- store.py ingest_dataframe and ingest_from_analyzer, plus the
  ``StoredMetricRecord.alert`` default they relied on. Both stamped
  ``alert=False`` on every record, which the whole reporting stack reads as
  "compared to its threshold and found clean". Twelve rows of
  ``demographic_parity = 0.95`` and ``disparate_impact = 0.05``, about as
  unfair as those metrics get, scored ``100.0/100 GREEN`` through each path,
  with ``n_not_assessable = 0`` and not one warning.

Every refusal pin here is paired with an over-correction control that asserts
the MEASURED numbers are untouched: the controls were taken from the
pre-fix module and must keep their exact values, or the fix has started
refusing things it can answer.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any, List, Optional

import pandas as pd
import pytest

from vfairness.operations.monitoring.drift import FairnessDriftDetector
from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer
from vfairness.operations.reporting import MetricsStore
from vfairness.operations.reporting.store import StoredMetricRecord


def _catch(fn, *args, **kwargs):
    """Run *fn* and return ``(result, warning_texts)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _store(
    *,
    metric_alerts: List[Optional[bool]],
    drift: List[Any],
    n_alerts: int = 0,
    at: Optional[datetime] = None,
) -> MetricsStore:
    """A store with hand-built rows, so every determination is explicit."""
    at = at or datetime.now()
    store = MetricsStore()
    for i, flag in enumerate(metric_alerts):
        store._records.append(
            StoredMetricRecord(
                timestamp=at - timedelta(hours=i),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.02,
                alert=flag,
            )
        )
    for i, verdict in enumerate(drift):
        store._drift_records.append(
            {
                "timestamp": at - timedelta(hours=i),
                "metric": "demographic_parity",
                "overall_drift_score": 0.1,
                "drift_detected": verdict,
            }
        )
    for _ in range(n_alerts):
        store._alert_records.append({"timestamp": at, "severity": "HIGH", "priority_score": 0.9})
    return store


# Finding 1: a REFUSED drift verdict is not a drift stability of 0


class TestRefusedDriftIsNotZeroStability:
    def test_the_detector_really_does_refuse(self):
        """Anti-vacuity. If this stops returning None the pins below prove
        nothing, because there would be no could-not-check verdict to mishandle.
        """
        detector = FairnessDriftDetector()
        result, warned = _catch(
            detector.detect_drift_multiscale,
            pd.Series([0.1, 0.2]),
            metric="demographic_parity",
        )
        assert result.drift_detected is None, result.drift_detected
        assert any("no drift test was run" in w for w in warned), warned

    def test_a_refused_drift_test_is_not_graded(self):
        """REFUSAL PIN. Measured before the fix: score 80.0, status green,
        components['drift_stability'] 0.0, zero warnings."""
        detector = FairnessDriftDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused = detector.detect_drift_multiscale(
                pd.Series([0.1, 0.2]), metric="demographic_parity"
            )
        store = _store(metric_alerts=[False, False], drift=[])
        store.ingest_drift_result(refused)

        health, warned = _catch(store.compute_health_score)

        assert health.score is None, health.score
        assert health.status == "not_assessed", health.status
        # No graded sub-score may survive: a 0.0 in a KPI card labelled Drift
        # Stability is a measurement of total instability, printed for a test
        # that never ran.
        assert health.components == {}, health.components
        assert health.components.get("drift_stability") is None
        assert any("could-not-check" in w for w in warned), warned
        assert any("not a drift stability of 0" in w for w in warned), warned
        assert "not a drift stability of 0" in health.explanation

    def test_the_trend_does_not_improve_off_a_refused_previous_window(self):
        """REFUSAL PIN for the previous-window twin. Measured before the fix:
        trend 'improving', slope +10.0, built on a 0.0 drift component that
        nobody measured, while the metric rows say the system got better only
        because the previous window's drift was ungradeable."""
        window = timedelta(days=7)
        now = datetime.now()
        prev = now - (window + timedelta(days=1))
        store = MetricsStore()
        for _ in range(4):
            store._records.append(
                StoredMetricRecord(
                    timestamp=prev,
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.9,
                    alert=True,
                )
            )
        for _ in range(2):
            store._drift_records.append(
                {
                    "timestamp": prev,
                    "metric": "demographic_parity",
                    "overall_drift_score": 0.1,
                    "drift_detected": None,
                }
            )
        for i in range(4):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now - timedelta(hours=i),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.02,
                    alert=False,
                )
            )
        # Anti-vacuity: the fixture must actually straddle the two windows.
        assert len(store.get_metrics(start_time=now - window, apply_privacy=False)) == 4
        assert len(store.get_drift_history(end_time=now - window)) == 2

        health, _ = _catch(store.compute_health_score, time_window=window)

        assert health.trend == "unknown", health.trend
        assert health.trend_slope == pytest.approx(0.0)
        assert health.score == pytest.approx(100.0)

    @pytest.mark.parametrize(
        ("drift", "expected_score", "expected_stability"),
        [
            # BGL3 operations-4, 2026-09-27: the row `([], 100.0, 100.0)` was
            # REMOVED from this list and replaced by the test below it. An empty
            # drift table is not a measured stability of 100.0, and that 100.0
            # was carrying 20 percent of the composite for a component nothing
            # had measured. Every row that remains has real verdicts, and every
            # number in it is unchanged.
            # BGL5 A-operations-3, 2026-09-27: the SCORE column moved because the
            # ALERT component left the composite. This fixture passes n_alerts=0
            # and never touches `_alert_records`, so alert_frequency was 100.0 for
            # a component with no evidence at all, worth 30 points: the same
            # defect as the drift row that was removed above, one component over.
            # With the 0.50/0.20 weights renormalised over 0.70 the composites
            # are (0.50*100 + 0.20*stability)/0.70, i.e. 100.0, 85.7 and 71.4.
            # The STABILITY column, which is this test's subject, is unchanged.
            ([False, False], 100.0, 100.0),
            ([True, False], 85.7, 50.0),
            ([True, True], 71.4, 0.0),
        ],
    )
    def test_control_determined_drift_scores_exactly_as_before(
        self, drift, expected_score, expected_stability
    ):
        """OVER-CORRECTION CONTROL. Every one of these numbers was measured on
        the pre-fix module and must not move: a fix that refuses a drift
        verdict the detector actually gave is as wrong as the defect. Note the
        last row: a real, measured 0.0 stability still grades 0.0."""
        health, warned = _catch(_store(metric_alerts=[False] * 4, drift=drift).compute_health_score)
        assert health.score == pytest.approx(expected_score)
        assert health.components["drift_stability"] == pytest.approx(expected_stability)
        assert health.status == ("green" if expected_score >= 80 else "yellow")
        assert not any("drift" in w for w in warned), warned
        # BGL5 A-operations-3: the drift verdicts are still graded and weighted,
        # which is the subject; what left is the unmeasured alert component.
        assert "alert_frequency" not in health.components, health.components

    def test_an_empty_drift_table_is_excluded_rather_than_scored_100(self):
        """The row removed from the parametrisation above, asserting the honest
        answer instead. Measured before the change: score 100.0 with
        components['drift_stability'] == 100.0 and no warning, for a store whose
        drift table is empty. The two metric components renormalise to the same
        100.0 here; what must not survive is the unmeasured component."""
        health, warned = _catch(_store(metric_alerts=[False] * 4, drift=[]).compute_health_score)
        assert "drift_stability" not in health.components, health.components
        assert health.score == pytest.approx(100.0)
        assert any("drift stability could not be measured" in w for w in warned), warned

    def test_control_one_refused_verdict_beside_determined_ones_keeps_the_number(self):
        """OVER-CORRECTION CONTROL. A single refusal must not withhold a score
        the other rows can still support, and must not change it: one drifting
        and one clean verdict is 50.0 stability with or without the refused
        third row. It is disclosed instead."""
        both = _store(metric_alerts=[False] * 4, drift=[True, False])
        mixed = _store(metric_alerts=[False] * 4, drift=[True, False, None])

        clean_health, clean_warned = _catch(both.compute_health_score)
        mixed_health, mixed_warned = _catch(mixed.compute_health_score)

        # BGL5 A-operations-3, 2026-09-27: 90.0 became 85.7 because the alert
        # component left the composite (this fixture never feeds the alert
        # channel, so its 100.0 was a perfect default for no evidence). The
        # subject is the two scores being EQUAL and the stability unchanged at
        # 50.0, both asserted below and both untouched by that.
        assert clean_health.score == pytest.approx(85.7)
        assert mixed_health.score == pytest.approx(clean_health.score)
        assert mixed_health.components["drift_stability"] == pytest.approx(50.0)
        assert mixed_health.status == "green"
        assert not any("drift" in w for w in clean_warned), clean_warned
        assert any("carry no verdict" in w for w in mixed_warned), mixed_warned
        assert any("demographic_parity" in w for w in mixed_warned), mixed_warned

    def test_control_a_genuinely_bad_history_still_scores_badly(self):
        """OVER-CORRECTION CONTROL. Four breaching metric rows, two real drift
        detections and three alerts: 0.50 * 0 + 0.30 * 70 + 0.20 * 0 = 21.0,
        red, exactly as before."""
        health, _ = _catch(
            _store(metric_alerts=[True] * 4, drift=[True, True], n_alerts=3).compute_health_score
        )
        assert health.score == pytest.approx(21.0)
        assert health.status == "red"
        assert health.components == {
            "metric_compliance": pytest.approx(0.0),
            "alert_frequency": pytest.approx(70.0),
            "drift_stability": pytest.approx(0.0),
        }


# Finding 2: ingest_dataframe and ingest_from_analyzer never compared anything


def _maximally_unfair_frame(n_days: int = 6) -> pd.DataFrame:
    """demographic_parity 0.95 (0.0 is parity) and disparate_impact 0.05
    (1.0 is parity, 0.8 is the four-fifths floor). No history is worse."""
    now = datetime.now()
    rows = []
    for day in range(n_days):
        stamp = now - timedelta(days=day)
        rows.append({"timestamp": stamp, "metric": "demographic_parity", "value": 0.95})
        rows.append({"timestamp": stamp, "metric": "disparate_impact", "value": 0.05})
    return pd.DataFrame(rows)


class TestUncomparedRowsAreNotCompliantRows:
    def test_the_record_default_carries_no_determination(self):
        """REFUSAL PIN for the dataclass default at the root of both paths."""
        record = StoredMetricRecord(
            timestamp=datetime.now(),
            source="custom",
            metric_name="demographic_parity",
            value=0.95,
        )
        assert record.alert is None, record.alert

    def test_ingest_dataframe_does_not_certify_an_uncompared_history(self):
        """REFUSAL PIN. Measured before the fix: score 100.0, status green,
        metric_compliance 100.0, n_not_assessable 0, zero warnings, on the most
        unfair history these two metrics can express."""
        store = MetricsStore()
        assert store.ingest_dataframe(_maximally_unfair_frame()) == 12
        assert {r.alert for r in store._records} == {None}
        assert set(store.get_metrics(apply_privacy=False)["alert_determined"]) == {False}

        health, warned = _catch(store.compute_health_score)

        assert health.score is None, health.score
        assert health.status == "not_assessed", health.status
        # metric_compliance must be ABSENT: nothing here was ever compared to a
        # threshold, so it has no evidence. `components == {}` was too strong
        # from 2026-09-17: alert_frequency IS measured (it is a count of the
        # alerts in the window, zero here) and reporting it costs nothing while
        # withholding it would be the reverse defect, discarding a real
        # measurement. The subject is that an uncompared history must not be
        # CERTIFIED, and it is not: the score is withheld and compliance is
        # absent.
        assert "metric_compliance" not in health.components, health.components
        assert "drift_stability" not in health.components, health.components
        assert health.n_not_assessable == 12
        assert any("no alert determination" in w for w in warned), warned
        assert any("demographic_parity" in w and "disparate_impact" in w for w in warned), warned

    def test_ingest_from_analyzer_does_not_certify_an_uncompared_history(self):
        """REFUSAL PIN, second path. The analyzer's daily frame holds values
        only: it has no alert column and applies no threshold. Measured before
        the fix: 100.0 GREEN on the same maximally unfair history."""
        analyzer = TemporalFairnessAnalyzer(lookback_days=90)
        now = datetime.now()
        for day in range(6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analyzer.update_daily_metrics(
                    pd.Timestamp(now - timedelta(days=day)),
                    {"demographic_parity": 0.95, "disparate_impact": 0.05},
                )
        store = MetricsStore()
        assert store.ingest_from_analyzer(analyzer) == 12
        assert {r.alert for r in store._records} == {None}

        health, warned = _catch(store.compute_health_score)

        assert health.score is None, health.score
        assert health.status == "not_assessed", health.status
        assert health.n_not_assessable == 12
        assert any("no alert determination" in w for w in warned), warned

    def test_control_a_determined_frame_still_scores_its_real_answer(self):
        """OVER-CORRECTION CONTROL. When the caller HAS compared the values and
        says so through *alert_col*, the rows are compliance evidence again and
        the arithmetic is unchanged: all clean is 100.0 green with nothing set
        aside and no warning, and one breach in two is metric_compliance 0.0."""
        frame = _maximally_unfair_frame(n_days=2)
        clean = frame.assign(alert=[False] * len(frame))
        store = MetricsStore()
        store.ingest_dataframe(clean, alert_col="alert")
        assert {r.alert for r in store._records} == {False}
        health, warned = _catch(store.compute_health_score)
        assert health.score == pytest.approx(100.0)
        assert health.status == "green"
        assert health.n_not_assessable == 0
        assert health.components["metric_compliance"] == pytest.approx(100.0)
        # BGL3 operations-4, 2026-09-27: this was `warned == []`. This store has
        # no drift row, and the store now discloses that instead of scoring that
        # component 100.0. The subject is that the COMPLIANCE arithmetic is
        # unchanged and sets nothing aside, so the drift-coverage sentence is
        # filtered out rather than the assertion dropped.
        # BGL5 A-operations-3, 2026-09-27: and the alert-coverage sentence, for
        # the same reason: this store never fed the alert channel either.
        assert [w for w in warned if "drift" not in w and "alert record" not in w] == [], warned

        half = frame.head(2).assign(alert=[True, False])
        breached = MetricsStore()
        breached.ingest_dataframe(half, alert_col="alert")
        breached_health, _ = _catch(breached.compute_health_score)
        # 1 of 2 rows breached: 100 * (1 - 0.5 * 2) = 0. The wave-4 arithmetic.
        assert breached_health.components["metric_compliance"] == pytest.approx(0.0)
        assert breached_health.n_not_assessable == 0

    def test_control_a_determination_of_none_in_the_column_stays_none(self):
        """OVER-CORRECTION CONTROL, the other direction: reading *alert_col* must
        not invent a verdict either. A column holding None round-trips as no
        determination, exactly as ingest_from_monitor reads its history."""
        frame = _maximally_unfair_frame(n_days=1).assign(alert=[False, None])
        store = MetricsStore()
        store.ingest_dataframe(frame, alert_col="alert")
        assert [r.alert for r in store._records] == [False, None]
        health, warned = _catch(store.compute_health_score)
        assert health.score == pytest.approx(100.0)
        assert health.n_not_assessable == 1
        assert any("disparate_impact" in w for w in warned), warned


class TestTheSimulationComparesInTheMetricsOwnDirection:
    """``simulate_threshold_change`` projected an alert for every value BELOW
    the proposed threshold, whatever the metric was.

    That is right for the ratio family (``disparate_impact``, the four-fifths
    rule) and exactly backwards for every violation magnitude, which is most
    fairness metrics. Measured 2026-09-10 on five real monitor windows of
    ``demographic_parity`` between 0.02 and 0.08, all excellent readings: the
    shipped code projected 5 of 5 alerting at thresholds 0.70, 0.80 and 0.90
    alike, where the metric's own rule projects 0, and the projection did not
    move with the threshold at all. This function exists to help an operator
    CHOOSE a threshold, so an inverted projection sends them the wrong way.

    The library already had one answer to "which way is better": the
    ``_metric_direction`` module, whose contract is that an unrecognised metric
    fails closed rather than being guessed. This simulator was not asking it.
    """

    @staticmethod
    def _sim(metric: str, values: List[float], threshold: float) -> Any:
        from vfairness.operations.reporting.interactive import simulate_threshold_change

        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * len(values),
                    "metric": [metric] * len(values),
                    "value": values,
                    "group": ["a"] * len(values),
                    "alert": [False] * len(values),
                }
            ),
            group_col="group",
            alert_col="alert",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return simulate_threshold_change(store, metric, threshold)

    def test_a_violation_magnitude_alerts_above_its_threshold(self):
        # Three excellent readings and two terrible ones. A parity DIFFERENCE
        # of 0.02 is not an alert at any threshold above it.
        values = [0.02, 0.05, 0.03, 0.85, 0.90]
        for threshold in (0.10, 0.50, 0.80):
            r = self._sim("demographic_parity", values, threshold)
            assert r["direction"] == "lower_is_better"
            assert r["projected_alerts"] == 2, (
                f"threshold={threshold}: projected {r['projected_alerts']} alert(s) "
                f"from {values}; only 0.85 and 0.90 breach a lower-is-better bound"
            )
        # And it MOVES: a bound above the worst reading catches nothing.
        assert self._sim("demographic_parity", values, 0.95)["projected_alerts"] == 0

    def test_a_parity_ratio_still_alerts_below_its_threshold(self):
        """Over-correction control. The direction that WAS right stays right."""
        values = [0.95, 0.99, 0.60, 0.40, 1.00]
        r80 = self._sim("disparate_impact", values, 0.80)
        assert r80["direction"] == "higher_is_better"
        assert r80["projected_alerts"] == 2, "0.60 and 0.40 fall short of four-fifths"
        # It moves in the other direction too, which is what makes it a
        # projection rather than a constant.
        assert self._sim("disparate_impact", values, 0.50)["projected_alerts"] == 1
        assert self._sim("disparate_impact", values, 0.30)["projected_alerts"] == 0

    def test_an_unknown_metric_is_refused_rather_than_guessed(self):
        r = self._sim("some_bespoke_metric", [0.02, 0.9], 0.5)
        assert r["projected_alerts"] is None and r["current_alerts"] is None
        assert r["change_pct"] is None and r["groups_impacted"] is None
        assert r["direction"] == "unknown"
        assert "COULD NOT SIMULATE" in r["not_simulated_reason"]
        assert "guessing" in r["not_simulated_reason"], (
            "the reason has to say what the refusal is protecting against"
        )

    def test_groups_impacted_follows_the_same_direction(self):
        """The group list is read as 'who this would newly alert on', so it
        cannot be computed by the opposite comparison to the count above."""
        from vfairness.operations.reporting.interactive import simulate_threshold_change

        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 4,
                    "metric": ["demographic_parity"] * 4,
                    "value": [0.02, 0.03, 0.85, 0.90],
                    "group": ["clean", "clean", "breaching", "breaching"],
                    "alert": [False] * 4,
                }
            ),
            group_col="group",
            alert_col="alert",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = simulate_threshold_change(store, "demographic_parity", 0.5)
        assert r["groups_impacted"] == ["breaching"], (
            f"got {r['groups_impacted']}; the clean group was named as impacted"
        )


class TestBothMonitorIngestionPathsNameAMetricTheSameWay:
    """``ingest_window_metrics`` split the monitor's ``metric_group_col`` key
    into a metric and a group. ``ingest_from_monitor``, which ingests the SAME
    numbers by the bulk route, stored the composite key verbatim.

    So one measurement landed under two different names depending on which
    ingestion the caller used, and the composite name is one no other part of
    the library recognises: ``metric_direction('demographic_parity')`` is
    LOWER_IS_BETTER while ``metric_direction('demographic_parity_group_gender')``
    is UNKNOWN. Every consumer that needs the metric's MEANING, the threshold
    simulator included, was blind to everything the bulk path produced.
    """

    @staticmethod
    def _monitored():
        import numpy as np

        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig

        rng = np.random.default_rng(0)
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                window_size=200,
                alert_threshold=0.80,
                prediction_col="y_pred",
                label_col="y_true",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                snap = monitor.update_and_check(
                    pd.DataFrame(
                        {
                            "y_true": rng.integers(0, 2, 200),
                            "y_pred": rng.integers(0, 2, 200),
                            # PLAIN name, not "group_gender". READINESS-6: this
                            # fixture used the prefixed form, so the "_group_"
                            # the parser looked for came from the COLUMN NAME
                            # rather than from the monitor, and the test passed
                            # for the wrong reason while every auto-detected
                            # column produced an unparseable key.
                            "gender": rng.choice(["male", "female"], 200),
                        }
                    )
                )
        return monitor, snap

    def test_the_bulk_route_and_the_snapshot_route_agree(self):
        monitor, snap = self._monitored()

        bulk = MetricsStore()
        bulk.ingest_from_monitor(monitor)
        per_snapshot = MetricsStore()
        per_snapshot.ingest_window_metrics(snap)

        bulk_names = set(bulk.get_metrics()["metric_name"])
        snap_df = per_snapshot.get_metrics()
        # The snapshot route also records group positive rates and MMD scores,
        # which the monitor's metric HISTORY does not carry at all. Compare the
        # names for the fairness metrics both routes see, which is the set the
        # two paths are supposed to agree on.
        snap_names = set(snap_df[snap_df["source"] == "FairnessMonitor"]["metric_name"]) - {
            "positive_rate",
            "mmd_score",
        }
        assert snap_names, "the snapshot route ingested nothing, so this pins nothing"
        assert snap_names <= bulk_names, (
            f"the two ingestion routes disagree on the metric's name: "
            f"bulk={sorted(bulk_names)} snapshot={sorted(snap_names)}"
        )
        assert not any("_gender" in n for n in bulk_names), (
            f"a composite key is being stored again: {sorted(bulk_names)}"
        )
        assert "demographic_parity" in bulk_names

    def test_the_stored_name_is_one_the_direction_registry_knows(self):
        """The point of the split: the name has to carry meaning downstream."""
        from vfairness.evaluation.vfairness_metrics._metric_direction import (
            MetricDirection,
            metric_direction,
        )

        monitor, _ = self._monitored()
        store = MetricsStore()
        store.ingest_from_monitor(monitor)
        df = store.get_metrics()

        for name in sorted(set(df["metric_name"])):
            assert metric_direction(name) is not MetricDirection.UNKNOWN, (
                f"the store recorded {name!r}, whose better-direction no consumer "
                "can determine, so every threshold comparison on it fails closed"
            )
        # The group survived the split rather than being dropped into the name.
        assert "gender" in set(df["group"]), f"groups were {sorted(set(df['group']))}"


class TestAnUnrecordedGroupSizeIsNotALargeGroup:
    """The k-anonymity gate released values on a group size nobody recorded.

    ``StoredMetricRecord.group_size`` defaults to 0, and most ingestion paths
    never set it. The suppression mask read ``(size > 0) & (size < k)``, which
    is False at 0; the noisy band is False at 0; so the record fell through to
    EXACT. The store released exact values while asserting a k-anonymity
    guarantee it had never checked, which is the one thing a privacy control
    may not do.

    Measured 2026-09-10: ``ingest_from_monitor`` produced 240 records for a
    named demographic group, every one ``group_size=0``, every one
    ``privacy_level='exact'``. The count was not missing at the source. The
    monitor's own window knew it was 200 and ``get_metric_history()`` dropped
    it at the boundary, which is why the fix is in three places: carry the
    count, record it, and stop reading its absence as a number.
    """

    @staticmethod
    def _monitor():
        import numpy as np

        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig

        rng = np.random.default_rng(0)
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                window_size=200,
                alert_threshold=0.80,
                prediction_col="y_pred",
                label_col="y_true",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(2):
                monitor.update_and_check(
                    pd.DataFrame(
                        {
                            "y_true": rng.integers(0, 2, 200),
                            "y_pred": rng.integers(0, 2, 200),
                            # PLAIN name, not "group_gender". READINESS-6: this
                            # fixture used the prefixed form, so the "_group_"
                            # the parser looked for came from the COLUMN NAME
                            # rather than from the monitor, and the test passed
                            # for the wrong reason while every auto-detected
                            # column produced an unparseable key.
                            "gender": rng.choice(["male", "female"], 200),
                        }
                    )
                )
        return monitor

    def test_the_rule_has_three_states(self):
        from vfairness.operations.reporting.store import PrivacyLevel, _classify_privacy

        assert _classify_privacy(0) is PrivacyLevel.UNKNOWN_SIZE
        assert _classify_privacy(-1) is PrivacyLevel.UNKNOWN_SIZE
        # An unrecorded size is NOT the same answer as a small one: a reader
        # has to be able to tell "too small to release" from "nobody knows".
        assert _classify_privacy(5) is PrivacyLevel.SUPPRESSED
        assert _classify_privacy(5) is not _classify_privacy(0)
        # Over-correction control: the measured bands are exactly as before.
        assert _classify_privacy(9) is PrivacyLevel.SUPPRESSED
        assert _classify_privacy(10) is PrivacyLevel.NOISY
        assert _classify_privacy(49) is PrivacyLevel.NOISY
        assert _classify_privacy(50) is PrivacyLevel.EXACT
        assert _classify_privacy(200) is PrivacyLevel.EXACT

    def test_a_record_with_no_recorded_size_is_withheld_and_says_so(self):
        from vfairness.operations.reporting.store import MetricsStoreConfig, PrivacyLevel

        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=True))
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 2,
                    "metric": ["dp"] * 2,
                    "value": [0.5, 0.6],
                    "group": ["x", "y"],
                }
            ),
            group_col="group",
        )
        df, texts = _catch(store.get_metrics, apply_privacy=True)

        assert set(df["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}, (
            f"released as {sorted(set(df['privacy_level']))} on a size nobody recorded"
        )
        assert df["value"].isna().all(), "the values were released anyway"
        assert any("k-anonymity could not be checked" in t for t in texts), (
            "the withheld values were not explained, so the fix is invisible"
        )
        assert any("group_size when" in t for t in texts), (
            "the warning does not say how to fix it at the ingestion side"
        )

    def test_the_monitor_route_records_a_real_size_and_still_releases(self):
        """OVER-CORRECTION CONTROL, and the reason the fix is not just a refusal.

        A gate that withholds everything is as useless as one that released
        everything. The count exists; it was being dropped. Carry it, and the
        same records release exactly as before, with the guarantee actually
        checked this time.
        """
        from vfairness.operations.reporting.store import MetricsStoreConfig, PrivacyLevel

        store = MetricsStore(
            config=MetricsStoreConfig(
                enable_privacy=True, k_anonymity_threshold=10, noisy_threshold=50
            )
        )
        store.ingest_from_monitor(self._monitor())
        df, texts = _catch(store.get_metrics, apply_privacy=True)

        assert set(df["group_size"]) == {200}, (
            f"the window size was lost again: {sorted(set(df['group_size']))}"
        )
        assert set(df["privacy_level"]) == {PrivacyLevel.EXACT.value}
        assert df["value"].notna().all(), "a checked, large group was withheld anyway"
        assert not any("k-anonymity could not be checked" in t for t in texts), (
            "a record with a real size warned as if it had none"
        )

    def test_a_genuinely_small_group_is_still_suppressed_for_smallness(self):
        """The band that already worked keeps working, and keeps its own name."""
        from vfairness.operations.reporting.store import (
            MetricsStoreConfig,
            PrivacyLevel,
            StoredMetricRecord,
        )

        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=True))
        for group, size in (("tiny", 5), ("medium", 30), ("large", 200)):
            store._records.append(
                StoredMetricRecord(
                    timestamp=datetime.now(),
                    source="test",
                    metric_name="dp",
                    value=0.5,
                    group=group,
                    group_size=size,
                    alert=False,
                )
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = store.get_metrics(apply_privacy=True).set_index("group")

        assert df.loc["tiny", "privacy_level"] == PrivacyLevel.SUPPRESSED.value
        assert pd.isna(df.loc["tiny", "value"])
        assert df.loc["medium", "privacy_level"] == PrivacyLevel.NOISY.value
        assert df.loc["large", "privacy_level"] == PrivacyLevel.EXACT.value
        assert df.loc["large", "value"] == 0.5

    def test_a_withheld_row_is_still_a_graded_row(self):
        """The narration reads a withheld value as could-not-check unless it
        knows the comparison behind it ran. It matched only SUPPRESSED, so the
        new level would have turned every withheld row into a false alarm."""
        from vfairness.operations.reporting.reports import _assessed_rows
        from vfairness.operations.reporting.store import PrivacyLevel

        graded, unassessed = _assessed_rows(
            pd.DataFrame(
                {
                    "metric_name": ["dp", "dp", "dp"],
                    "value": [float("nan"), float("nan"), 0.5],
                    "alert_determined": [True, True, True],
                    "privacy_level": [
                        PrivacyLevel.SUPPRESSED.value,
                        PrivacyLevel.UNKNOWN_SIZE.value,
                        PrivacyLevel.EXACT.value,
                    ],
                }
            )
        )
        assert len(graded) == 3, (
            f"only {len(graded)} of 3 rows graded; a value withheld for privacy is "
            "not a value that could not be computed"
        )
        assert unassessed == []

    def test_the_tidy_path_can_record_the_size_it_knows(self):
        """The fix must leave callers a way through, not just a refusal.

        ``ingest_dataframe`` is the most general ingestion route and had no way
        to record a group size at all, so withholding its records under privacy
        would have been a dead end rather than a correction. It takes
        *group_size_col* now, and all three bands work through it.
        """
        from vfairness.operations.reporting.store import MetricsStoreConfig, PrivacyLevel

        store = MetricsStore(
            config=MetricsStoreConfig(
                enable_privacy=True, k_anonymity_threshold=10, noisy_threshold=50
            )
        )
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 3,
                    "metric": ["dp"] * 3,
                    "value": [0.5, 0.6, 0.7],
                    "group": ["tiny", "medium", "large"],
                    "n": [5, 30, 200],
                    "alert": [False] * 3,
                }
            ),
            group_col="group",
            alert_col="alert",
            group_size_col="n",
        )
        df, texts = _catch(store.get_metrics, apply_privacy=True)
        by_group = df.set_index("group")

        assert list(by_group.loc[["tiny", "medium", "large"], "group_size"]) == [5, 30, 200]
        assert by_group.loc["tiny", "privacy_level"] == PrivacyLevel.SUPPRESSED.value
        assert by_group.loc["medium", "privacy_level"] == PrivacyLevel.NOISY.value
        assert by_group.loc["large", "privacy_level"] == PrivacyLevel.EXACT.value
        assert by_group.loc["large", "value"] == 0.7
        assert not any("k-anonymity could not be checked" in t for t in texts), (
            "sizes were recorded, so nothing here is could-not-check"
        )


class TestTheWithheldScoreIsNarratedWithItsOwnReason:
    """`_narrate_health` hardcoded ONE reason for a withheld score.

    `compute_health_score` withholds for at least three different reasons: an
    empty window, a window whose every record carries no threshold comparison,
    and a window whose every drift test the detector refused. The score object
    carries the true reason in `explanation`, and the narrator discarded it and
    said "no metric records were found in the evaluation window" every time.

    Measured 2026-09-10: a store holding TWENTY records, none of them ever
    compared to a threshold, was reported to the reader as holding none. Those
    are different problems with different fixes, one being a pipeline that is
    not feeding the store and the other a pipeline feeding it values with no
    verdicts, and the report named the wrong one.
    """

    @staticmethod
    def _narrated(store):
        from vfairness.operations.reporting.reports import ReportTier, _narrate_health

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _narrate_health(store.compute_health_score(), ReportTier.EXECUTIVE)

    def test_an_empty_window_says_it_is_empty(self):
        text = self._narrated(MetricsStore())
        assert "could NOT be assessed" in text
        assert "No metric records were found" in text

    def test_a_window_of_uncompared_records_does_not_claim_to_be_empty(self):
        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 20,
                    "metric": ["demographic_parity"] * 20,
                    "value": [0.95] * 20,
                    "group": ["a"] * 20,
                }
            ),
            group_col="group",
        )
        text = self._narrated(store)

        assert "20 metric record(s)" in text, (
            f"the count of records that DID arrive is missing: {text}"
        )
        assert "no alert determination" in text, "the real reason is missing"
        assert "No metric records were found" not in text, (
            "the report told the reader the store was empty while holding 20 records"
        )

    def test_a_withheld_score_is_never_narrated_without_the_disclaimer(self):
        """Whatever the reason, a reader must not be able to read a withheld
        score as a low one. The store's own explanations end with that clause,
        so the narrator adds it only when the reason lacks it, and a score
        carrying no explanation at all must still get it."""
        from vfairness.operations.reporting.reports import ReportTier, _narrate_health
        from vfairness.operations.reporting.store import HealthScore

        bare = HealthScore(
            score=None,
            status="not_assessed",
            trend="unknown",
            trend_slope=0.0,
            components={},
            timestamp=datetime.now(),
            explanation="",
        )
        text = _narrate_health(bare, ReportTier.EXECUTIVE)
        assert "not a score of 0 and not a score of 100" in text
        assert "No reason was recorded" in text

        # And it is not said TWICE when the reason already carries it.
        empty = self._narrated(MetricsStore())
        assert empty.lower().count("not a score of 100") == 1, empty

    def test_a_measured_score_still_narrates_its_numbers(self):
        """OVER-CORRECTION CONTROL. A real score must not be routed into the
        could-not-check sentence."""
        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 10,
                    "metric": ["demographic_parity"] * 10,
                    "value": [0.02] * 10,
                    "group": ["a"] * 10,
                    "alert": [False] * 10,
                }
            ),
            group_col="group",
            alert_col="alert",
        )
        text = self._narrated(store)
        assert "could NOT be assessed" not in text, text
        assert "Overall fairness status" in text
        assert "/100" in text


class TestTheMonitorKeyIsSplitOnTheMetricNotASeparator:
    """READINESS-6, and this one is a correction of a fix made hours earlier.

    `_split_monitor_key` split on the literal "_group_", copied from the older
    of its two call sites. FairnessMonitor does not emit that separator. It
    emits f"{metric}_{col}", so "_group_" appears ONLY when the protected COLUMN
    is itself named group_gender::

        column 'gender'       -> 'demographic_parity_gender'
                              -> old split: ('demographic_parity_gender', 'overall')
        column 'group_gender' -> 'demographic_parity_group_gender'
                              -> old split: ('demographic_parity', 'gender')

    Until 2026-09-09 that was almost harmless, because the monitor only took
    columns carrying the group_ prefix. R-3 taught it to auto-detect columns
    without one and this parser was never updated, so every auto-detected column
    has produced an unparseable key since: metric_direction says UNKNOWN, the
    threshold simulator refuses every real metric, and the store records no
    groups at all.

    The guard written for it that morning did not catch this, because its own
    fixture column was named group_gender. A test that supplies the thing under
    test passes for the wrong reason.
    """

    @staticmethod
    def _keys_for(column: str):
        import numpy as np

        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig

        rng = np.random.default_rng(0)
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                window_size=200,
                alert_threshold=0.80,
                prediction_col="y_pred",
                label_col="y_true",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snap = monitor.update_and_check(
                pd.DataFrame(
                    {
                        "y_true": rng.integers(0, 2, 200),
                        "y_pred": rng.integers(0, 2, 200),
                        column: rng.choice(["male", "female"], 200),
                    }
                )
            )
        return monitor, list(snap.metrics)

    @pytest.mark.parametrize("column", ["gender", "group_gender"])
    def test_a_real_monitor_key_yields_a_metric_the_library_knows(self, column):
        from vfairness.evaluation.vfairness_metrics._metric_direction import (
            MetricDirection,
            metric_direction,
        )
        from vfairness.operations.reporting.store import _split_monitor_key

        _, keys = self._keys_for(column)
        assert keys, "the monitor computed nothing, so this pins nothing"
        for key in keys:
            metric, group = _split_monitor_key(key)
            assert metric_direction(metric) is not MetricDirection.UNKNOWN, (
                f"column {column!r} produced key {key!r}, which splits to metric "
                f"{metric!r}, whose better-direction no consumer can determine"
            )
            assert group == "gender", f"{key!r} split to group {group!r}"

    @pytest.mark.parametrize("column", ["gender", "group_gender"])
    def test_the_store_records_the_attribute_as_the_group(self, column):
        monitor, _ = self._keys_for(column)
        store = MetricsStore()
        store.ingest_from_monitor(monitor)
        df = store.get_metrics(apply_privacy=False)
        assert set(df["group"]) == {"gender"}, (
            f"column {column!r} recorded groups {sorted(set(df['group']))}"
        )
        assert all("_gender" not in n for n in set(df["metric_name"])), (
            f"a composite metric name survived: {sorted(set(df['metric_name']))}"
        )

    def test_a_metric_whose_own_name_ends_in_difference_is_not_cut_apart(self):
        """OVER-CORRECTION CONTROL, and it caught a real one.

        The prefix match is greedy: `demographic_parity_difference` is
        `demographic_parity` plus `_difference`, so a naive matcher reads it as
        that metric measured on a column called "difference", and takes the
        whole *_difference / *_ratio family with it. An existing report test
        went red on precisely that while this parser was being corrected.

        `metric_direction` cannot separate the two readings, because it matches
        these words as TOKENS anywhere in a name and answers LOWER_IS_BETTER for
        `custom_gap_group_sex` as well. So the tail set is explicit.
        """
        from vfairness.operations.reporting.store import _split_monitor_key

        for name in (
            "demographic_parity_difference",
            "demographic_parity_ratio",
            "disparate_impact_ratio",
            "equalized_odds_difference",
            "equal_opportunity_difference",
        ):
            assert _split_monitor_key(name) == (name, "overall"), name

        # And the column readings it must NOT swallow, in the same breath.
        assert _split_monitor_key("demographic_parity_gender") == ("demographic_parity", "gender")
        assert _split_monitor_key("equalized_odds_race") == ("equalized_odds", "race")
        # A custom metric measured per column, via the documented convention.
        assert _split_monitor_key("custom_gap_group_sex") == ("custom_gap", "sex")

    def test_a_custom_metric_is_not_cut_apart_on_a_guess(self):
        """OVER-CORRECTION CONTROL. A key that matches no known metric is a
        custom reading, not a per-column one, and must survive whole."""
        from vfairness.operations.reporting.store import _split_monitor_key

        for key in ("custom_business_metric", "alert_priority", "drift_score", "revenue_per_user"):
            assert _split_monitor_key(key) == (key, "overall"), key
        # And a bare metric name with no column is not split either.
        assert _split_monitor_key("demographic_parity") == ("demographic_parity", "overall")


class TestTheStoresOwnRoundTripDoesNotLaunderAVerdict:
    """READINESS-6. The R-2 fix was defeated by this store's OWN export.

    `get_metrics` emits `alert` as a strict bool for the mask readers and
    carries the third state beside it in `alert_determined`. `ingest_dataframe`
    had no parameter that could receive that second column, so one
    to_dataframe / ingest_dataframe cycle turned every "nobody compared this"
    into "compared and clean".

    Measured 2026-09-10: six readings of demographic_parity = 0.95, none ever
    compared to a threshold, scored None / not_assessed / n_not_assessable=6
    before the round trip and 100.0 / green / 0 after it, with no warning. That
    is verbatim the scenario the R-2 docstring says was fixed.
    """

    @staticmethod
    def _store(values, alert=None):
        from vfairness.operations.reporting.store import MetricsStoreConfig

        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        frame = {
            "timestamp": [datetime.now()] * len(values),
            "metric": ["demographic_parity"] * len(values),
            "value": values,
            "group": ["a"] * len(values),
        }
        kwargs = {"group_col": "group"}
        if alert is not None:
            frame["alert"] = alert
            kwargs["alert_col"] = "alert"
        store.ingest_dataframe(pd.DataFrame(frame), **kwargs)
        return store

    @staticmethod
    def _reingest(store):
        from vfairness.operations.reporting.store import MetricsStoreConfig

        clone = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        clone.ingest_dataframe(
            store.to_dataframe(), metric_col="metric_name", group_col="group", alert_col="alert"
        )
        return clone

    def test_an_undetermined_row_stays_undetermined(self):
        before, _ = _catch(self._store([0.95] * 6).compute_health_score)
        after, _ = _catch(self._reingest(self._store([0.95] * 6)).compute_health_score)

        assert before.score is None and before.n_not_assessable == 6
        assert after.score is None, (
            f"six readings nobody compared to a threshold scored {after.score!r} "
            "after one export and re-ingest"
        )
        assert after.status == "not_assessed"
        assert after.n_not_assessable == 6

    def test_a_real_determination_still_survives(self):
        """OVER-CORRECTION CONTROL. A round trip that lost real verdicts would
        be as broken as one that invented them."""
        before, _ = _catch(self._store([0.02] * 6, alert=[False] * 6).compute_health_score)
        after, _ = _catch(
            self._reingest(self._store([0.02] * 6, alert=[False] * 6)).compute_health_score
        )
        assert before.score == 100.0 and before.status == "green"
        assert after.score == before.score and after.status == before.status
        assert after.n_not_assessable == 0

    def test_a_plain_tidy_frame_is_unaffected(self):
        """The frame a normal caller builds has neither export column, and its
        behaviour must not change at all."""
        store = self._store([0.1, 0.2, 0.3], alert=[True, False, False])
        assert [r.alert for r in store._records] == [True, False, False]

    def test_a_privacy_withheld_row_is_refused_rather_than_promoted(self):
        """The same cycle laundered privacy: a suppressed or noised value was
        re-ingested as a measurement and re-exported as exact."""
        from vfairness.operations.reporting.store import MetricsStoreConfig

        source = MetricsStore(config=MetricsStoreConfig(enable_privacy=True))
        source.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 3,
                    "metric": ["dp"] * 3,
                    "value": [0.5, 0.6, 0.7],
                    "group": ["x", "y", "z"],
                }
            ),
            group_col="group",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            withheld = source.get_metrics(apply_privacy=True)

        clone = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        ingested, texts = _catch(
            clone.ingest_dataframe,
            withheld,
            metric_col="metric_name",
            group_col="group",
            alert_col="alert",
        )
        assert ingested == 0, f"{ingested} withheld row(s) were promoted to measurements"
        assert any("did not release" in t for t in texts)

    def test_every_window_record_carries_the_size_the_window_measured(self):
        """The MMD loop omitted group_size while both its siblings set it, so
        those records were withheld as unknown_size and warned about on every
        query, from a window that knew the count."""
        import numpy as np

        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig
        from vfairness.operations.reporting.store import MetricsStoreConfig

        rng = np.random.default_rng(0)
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                window_size=200, alert_threshold=0.80, prediction_col="y_pred", label_col="y_true"
            )
        )

        def batch():
            return pd.DataFrame(
                {
                    "y_true": rng.integers(0, 2, 200),
                    "y_pred": rng.integers(0, 2, 200),
                    "gender": rng.choice(["male", "female"], 200),
                }
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # A REFERENCE is required or snap.mmd_scores is empty and the MMD
            # loop this test exists to cover never runs. Found by sabotage:
            # dropping the group_size from that loop left the test GREEN, which
            # is a finding about the pin, not a pass.
            monitor.set_reference(batch())
            snap = monitor.update_and_check(batch())
        assert snap.mmd_scores, "no MMD score was produced, so this test covers nothing"
        store = MetricsStore(
            config=MetricsStoreConfig(
                enable_privacy=True, k_anonymity_threshold=10, noisy_threshold=50
            )
        )
        store.ingest_window_metrics(snap)
        df, texts = _catch(store.get_metrics, apply_privacy=True)

        assert "mmd_score" in set(df["metric_name"]), (
            "the MMD records are absent, so the loop under test was not exercised"
        )
        assert set(df["group_size"]) == {snap.sample_count}, (
            f"a record lost the window's count: {sorted(set(df['group_size']))}"
        )
        assert not any("could not be checked" in t for t in texts), (
            "a record the window had measured was warned about as unrecorded"
        )
