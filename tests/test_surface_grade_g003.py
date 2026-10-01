"""Batch g003: ``operations/reporting/store.py``, the MetricsStore surface.

Graded item by item on 2026-09-17 by execution, healthy input first and then
the degenerate inputs where each claim becomes undefined. Most of the surface
was already honest: an empty store refuses its health score, a refused drift
verdict is excluded rather than graded 0, a k-anonymity check that could not
run withholds the value as ``unknown_size``, and a tidy frame carrying no
threshold comparison is ingested as ``alert=None`` rather than as clean.

Three fabrications survived, all of them a neutral default standing in for a
value nobody produced, and all three are pinned below with a measured
before/after and a control that keeps the real numbers exact.

1. ``ingest_alert``. ``d.get("priority_score", 0.0)``. An alert dict with no
   priority score, which the public method accepts and the suite already feeds
   it, reached ``get_metrics`` as
   ``metric_name='alert_priority', value=0.0, alert_determined=True``: the
   BOTTOM of the prioritizer's roughly 0-to-10 scale, published as a released
   measurement and indistinguishable from an alert that was scored and ranked
   last. ``.get`` with a default also does not fire for a key present holding
   None, so an explicit ``"priority_score": None`` went into a float field
   unchanged. Now NaN, with a warning.

2. ``ingest_alert`` again. ``d.get("severity", "LOW")`` stamped the mildest
   severity onto an alert nobody graded. ``AlertPayload`` already spells that
   state ``"UNSCORED"`` (its own docstring says so), so the record uses it.

3. ``ingest_drift_result``. ``getattr(result, "mmd_score", 0.0)``. An MMD of
   0.0 is "the two distributions are identical", and
   ``MultiscaleDriftResult.mmd_score`` is ``Optional[float]`` precisely because
   the MMD is often not computed at all. Now None.

Plus one state defect found in the same pass: ``clear()`` documented itself as
"remove all stored data" and left ``_models`` populated, so ``get_summary()``
reported ``n_models=1`` for an emptied store and the technical report still
rendered a "Registered Models" section over it.

ONE DEFECT IS LEFT OPEN, deliberately, and it is recorded here so it is not
mistaken for a clean sweep. ``compute_health_score`` still publishes
``components['drift_stability'] = 100.0`` for a store whose drift table is
EMPTY, and weights it 20 percent into the composite. Measured 2026-09-17 on a
store built from one FairnessMonitor window where both built-in metrics breach
and no drift detector was ever run: ``score 50.0, status 'yellow', components
{'metric_compliance': 0.0, 'alert_frequency': 100.0, 'drift_stability': 100.0}``,
with ZERO warnings. The two components that rest on evidence renormalise to
37.5, which is a different band. The substitution is argued for in
``_drift_stability``, was re-examined at the reporting surface in BGL S2b
(``ReportGenerator._health_with_drift_coverage`` drops the key and prints "not
assessed" plus a Coverage section), and that comment states plainly why the
store-level number was not changed there: the composite is asserted by name and
by value in several other test lanes, so re-weighting it has to land together
with them. Closing it needs that coordinated change, not a local edit, and is
NOT attempted here.

Every refusal pin here is paired with a control asserting the MEASURED number
is untouched, because the reverse defect costs more: a store that refuses a
priority score it was actually given has thrown evidence away.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting import MetricsStore
from vfairness.operations.reporting.store import StoredMetricRecord


def _catch(fn, *args, **kwargs) -> Tuple[Any, List[str]]:
    """Run *fn* with warnings ENABLED and return ``(result, warning_texts)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# 1 + 2. An alert nobody scored has no priority and no severity


class TestAnUnscoredAlertHasNoPriority:
    def test_a_missing_priority_score_is_not_a_priority_of_zero(self):
        """REFUSAL PIN. Measured before the fix, on this exact input:
        ``StoredMetricRecord(..., metric_name='alert_priority', value=0.0,
        metadata={'severity': 'LOW'})`` and a ``get_metrics`` row reading
        ``value 0.0 ... alert_determined True``, with no warning at all.
        """
        store = MetricsStore()
        n, warned = _catch(
            store.ingest_alert,
            {"metric_name": "demographic_parity", "message": "who knows"},
        )
        assert n == 1

        published = store.get_metrics(apply_privacy=False)
        row = published[published["metric_name"] == "alert_priority"].iloc[0]
        assert pd.isna(row["value"]), (
            f"value={row['value']!r} for an alert carrying no priority_score; "
            "0.0 is the lowest priority on the scale, not an absent one"
        )
        assert any("priority_score" in w and "NaN" in w for w in warned), warned

    def test_a_priority_score_present_but_none_is_also_refused(self):
        """``.get(key, default)`` does NOT fire for a key present holding None,
        so this input went past the old default untouched and put None into a
        field annotated float."""
        store = MetricsStore()
        _, warned = _catch(store.ingest_alert, {"priority_score": None, "severity": "HIGH"})
        value = store.get_metrics(apply_privacy=False)["value"].iloc[0]
        assert pd.isna(value), value
        assert any("priority_score" in w for w in warned), warned

    def test_a_missing_severity_is_unscored_not_low(self):
        """REFUSAL PIN. ``LOW`` is a grade the prioritizer assigns; an alert it
        never graded is ``UNSCORED``, which AlertPayload already documents."""
        store = MetricsStore()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.ingest_alert({"metric_name": "demographic_parity"})
        record = store.get_latest("alert_priority")
        assert record is not None
        assert record.metadata["severity"] == "UNSCORED", record.metadata

    def test_control_a_scored_alert_keeps_its_exact_priority_and_severity(self):
        """OVER-CORRECTION CONTROL. The refusal must not eat a real score.
        9.8 and CRITICAL are the caller's own numbers and must survive
        untouched, and no warning may be raised for them."""
        store = MetricsStore()
        _, warned = _catch(
            store.ingest_alert,
            {
                "severity": "CRITICAL",
                "metric_name": "demographic_parity",
                "priority_score": 9.8,
                "timestamp": datetime.now(),
            },
        )
        assert warned == [], warned
        value = store.get_metrics(apply_privacy=False)["value"].iloc[0]
        assert value == pytest.approx(9.8)
        assert store.get_latest("alert_priority").metadata["severity"] == "CRITICAL"

    def test_control_a_priority_of_zero_that_was_actually_measured_survives(self):
        """The third state must be distinguishable from a measured 0.0, in both
        directions: a caller who really did score an alert 0.0 gets 0.0, and no
        warning."""
        store = MetricsStore()
        _, warned = _catch(store.ingest_alert, {"priority_score": 0.0, "severity": "LOW"})
        value = store.get_metrics(apply_privacy=False)["value"].iloc[0]
        assert value == 0.0 and not pd.isna(value)
        assert warned == [], warned

    def test_the_unscored_priority_does_not_leak_into_the_health_score(self):
        """TRACE THE CALLER. A NaN flowing into ``compute_health_score`` must
        change nothing about the grading of the metric rows beside it: the
        mirrored alert row is excluded from compliance either way, and the
        composite over two determined rows (one breaching) is identical whether
        the alert was scored 9.8 or not scored at all."""
        now = datetime.now()

        def _built(alert_payload):
            store = MetricsStore()
            for flag in (True, False):
                store._records.append(
                    StoredMetricRecord(
                        timestamp=now,
                        source="FairnessMonitor",
                        metric_name="demographic_parity",
                        value=0.42,
                        group_size=100,
                        alert=flag,
                    )
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.ingest_alert(alert_payload)
                return store.compute_health_score()

        scored = _built({"priority_score": 9.8, "severity": "CRITICAL"})
        unscored = _built({"severity": "CRITICAL"})
        assert scored.components == unscored.components, (scored, unscored)
        assert scored.score == unscored.score
        # And the composite is the hand-computed one, not an artefact:
        # compliance 100*(1 - 0.5*2) = 0, alerts 100 - 1*10 = 90.
        #
        # BGL3 operations-4, 2026-09-27: this line read
        #   round(0.50 * 0.0 + 0.30 * 90.0 + 0.20 * 100.0, 1)  == 47.0
        # and that 100.0 was DRIFT STABILITY ON AN EMPTY DRIFT TABLE, a perfect
        # score for a component nobody measured. The store now excludes it and
        # renormalises the two components that have evidence, so the same two
        # rows and the same one alert give (0.50*0.0 + 0.30*90.0)/0.80 = 33.8.
        # The subject of this test is untouched: a NaN priority must change
        # nothing about the grading of the rows beside it, and both calls still
        # produce the identical components and score.
        assert unscored.components["metric_compliance"] == pytest.approx(0.0)
        assert unscored.components["alert_frequency"] == pytest.approx(90.0)
        assert "drift_stability" not in unscored.components, unscored.components
        assert unscored.score == pytest.approx(round((0.50 * 0.0 + 0.30 * 90.0) / 0.80, 1))


# 3. An MMD nobody computed is not an MMD of zero


class _DriftResultWithoutMmd:
    """A drift result shaped like the ones the store ingests, minus the MMD.

    ``getattr(result, "mmd_score", 0.0)`` fires exactly here, and 0.0 is the
    value that means the two distributions are identical.
    """

    metric = "demographic_parity"
    overall_drift_score = 0.61
    drift_detected = True
    worst_scale = None

    def __init__(self, timestamp: datetime) -> None:
        self.timestamp = timestamp


class TestAnUncomputedMmdIsNotZero:
    def test_a_result_carrying_no_mmd_publishes_no_mmd(self):
        """REFUSAL PIN. Measured before the fix: ``get_drift_history()`` row
        ``mmd_score 0.0`` for a result object that never carried one."""
        store = MetricsStore()
        store.ingest_drift_result(_DriftResultWithoutMmd(datetime.now()))
        history = store.get_drift_history()
        assert len(history) == 1
        assert pd.isna(history["mmd_score"].iloc[0]), (
            f"mmd_score={history['mmd_score'].iloc[0]!r} for a result that carries "
            "none; 0.0 reads as 'the distributions are identical'"
        )

    def test_control_a_real_mmd_and_a_real_drift_score_are_published_exactly(self):
        """OVER-CORRECTION CONTROL. Real values, computed independently here,
        must arrive unchanged: the refusal must not blank a measured MMD, and
        the drift verdict beside it must still be True."""
        result = _DriftResultWithoutMmd(datetime.now())
        result.mmd_score = 0.37  # type: ignore[attr-defined]
        store = MetricsStore()
        store.ingest_drift_result(result)
        row = store.get_drift_history().iloc[0]
        assert row["mmd_score"] == pytest.approx(0.37)
        assert row["overall_drift_score"] == pytest.approx(0.61)
        assert row["drift_detected"] is True or row["drift_detected"] == np.True_

    def test_control_a_measured_mmd_of_zero_survives(self):
        """A genuinely computed MMD of 0.0 is a measurement and stays 0.0."""
        result = _DriftResultWithoutMmd(datetime.now())
        result.mmd_score = 0.0  # type: ignore[attr-defined]
        store = MetricsStore()
        store.ingest_drift_result(result)
        value = store.get_drift_history()["mmd_score"].iloc[0]
        assert value == 0.0 and not pd.isna(value)


# 4. "Remove all stored data" includes the model registry


class TestClearEmptiesTheModelRegistry:
    def test_a_cleared_store_reports_no_models(self):
        """Measured before the fix: ``get_summary()`` after ``clear()`` read
        ``n_metric_records 0`` beside ``n_models 1``, and the technical report
        renders that summary verbatim plus a "Registered Models" section."""
        store = MetricsStore()
        store.register_model("m1", {"version": "1.2"})
        store._records.append(
            StoredMetricRecord(
                timestamp=datetime.now(),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.42,
                group_size=100,
                alert=False,
            )
        )
        assert store.get_summary()["n_models"] == 1, "the fixture registers nothing"

        store.clear()
        summary = store.get_summary()
        assert summary["n_metric_records"] == 0
        assert summary["n_models"] == 0, summary
        assert store._models == {}

    def test_control_register_model_still_registers(self):
        """OVER-CORRECTION CONTROL. Clearing must not make registration a
        no-op: the registry holds what it was given, with its metadata."""
        store = MetricsStore()
        store.register_model("m1", {"version": "1.2"})
        store.register_model("m2", {"version": "2.0"})
        assert store.get_summary()["n_models"] == 2
        assert store._models["m2"]["version"] == "2.0"
        assert "registered_at" in store._models["m1"]


# Controls for the rest of the batch: the surface must keep MEASURING


class TestTheStoreStillMeasures:
    """Anti-over-correction for the batch as a whole.

    Every expected number here is computed by hand in the test, not copied from
    what the code returned.
    """

    @staticmethod
    def _store(now: datetime) -> MetricsStore:
        store = MetricsStore()
        # Four determined metric rows, exactly one breaching.
        for i, flag in enumerate((True, False, False, False)):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now - timedelta(hours=i),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.90,
                    group="f",
                    group_size=100,
                    alert=flag,
                )
            )
        # Two drift verdicts, exactly one drifting.
        for i, verdict in enumerate((True, False)):
            store._drift_records.append(
                {
                    "timestamp": now - timedelta(hours=i),
                    "metric": "demographic_parity",
                    "overall_drift_score": 0.3,
                    "drift_detected": verdict,
                }
            )
        return store

    def test_the_composite_is_the_hand_computed_one(self):
        now = datetime.now()
        health, warned = _catch(self._store(now).compute_health_score)
        # compliance: alert_frac 1/4, 100 * (1 - 0.25 * 2) = 50.0
        # drift stability: drift_frac 1/2, 100 * (1 - 0.5) = 50.0
        #
        # BGL5 A-operations-3, 2026-09-27: the third line here used to read
        # "alert frequency: no ingested alert payloads, 100.0" and the composite
        # was 0.50*50 + 0.30*100 + 0.20*50 = 65.0. That 100.0 was the perfect
        # default for a component with NO evidence: this fixture never feeds the
        # alert channel, and a store cannot tell "the prioritizer ran and nothing
        # breached" from "no alert pipeline was ever connected" (a clean drift run
        # leaves a row, a clean alert run leaves nothing). The store now excludes
        # it and renormalises over the components that have evidence, exactly as
        # it already did for drift, so the hand computation becomes
        # (0.50*50.0 + 0.20*50.0) / 0.70 = 50.0. The subject is unchanged: every
        # number below is still computed here rather than read back from the code.
        # A store that DOES feed all three channels is asserted at the full
        # 50/30/20 in tests/test_bgl5_operations_3.py::
        # test_control_a_fed_alert_channel_keeps_the_documented_composite (87.0).
        assert health.components["metric_compliance"] == pytest.approx(50.0)
        assert "alert_frequency" not in health.components, health.components
        assert health.components["drift_stability"] == pytest.approx(50.0)
        assert health.score == pytest.approx(50.0)
        assert health.status == "yellow"
        assert health.n_not_assessable == 0
        # The two disclosures about the unfed alert channel are the fix above;
        # nothing else may warn, which is what this filter still holds.
        assert [w for w in warned if "alert record" not in w] == [], warned

    def test_to_dict_carries_the_third_state_to_the_reader(self):
        """``HealthScore.to_dict`` is the surface a JSON report reads. A
        withheld score must arrive as None with its status, not as a number."""
        store = MetricsStore()
        health, _ = _catch(store.compute_health_score)
        payload = health.to_dict()
        assert payload["score"] is None, payload
        assert payload["status"] == "not_assessed", payload
        assert payload["n_metrics"] == 0
        # And the measured case round-trips its real number.
        # BGL5 A-operations-3, 2026-09-27: 65.0 became 50.0 for the reason given
        # on test_the_composite_is_the_hand_computed_one above (the unfed alert
        # channel is no longer scored 100.0). The subject is the round trip, and
        # the drift component below still carries its measured 50.0.
        measured = self._store(datetime.now()).compute_health_score().to_dict()
        assert measured["score"] == pytest.approx(50.0)
        assert measured["components"]["drift_stability"] == pytest.approx(50.0)

    def test_the_query_accessors_report_what_is_there(self):
        now = datetime.now()
        store = self._store(now)
        assert store.get_metric_names() == ["demographic_parity"]
        assert store.get_groups() == ["f"]
        assert store.get_sources() == ["FairnessMonitor"]
        earliest, latest = store.get_time_range()
        assert earliest == now - timedelta(hours=3)
        assert latest == now
        assert store.get_latest("demographic_parity").timestamp == now
        assert store.get_latest("no_such_metric") is None
        assert store.get_summary()["n_metric_records"] == 4
        assert store.get_summary()["n_drift_records"] == 2
        assert len(store.get_drift_history()) == 2

    def test_prune_removes_and_counts_exactly_the_old_records(self):
        now = datetime.now()
        store = self._store(now)
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=400),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.1,
                group_size=100,
                alert=False,
            )
        )
        assert store.prune() == 1
        assert len(store) == 4

    def test_the_privacy_tiers_are_still_applied_and_labelled(self):
        """``to_dataframe`` releases an exact value only for a group above the
        noisy threshold, and says which tier every row is in."""
        now = datetime.now()
        store = MetricsStore()
        for size, group in ((5, "tiny"), (20, "mid"), (100, "big"), (0, "unrecorded")):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now,
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.42,
                    group=group,
                    group_size=size,
                    alert=False,
                )
            )
        exported, warned = _catch(store.to_dataframe)
        tiers = dict(zip(exported["group"], exported["privacy_level"]))
        assert tiers == {
            "tiny": "suppressed",
            "mid": "noisy",
            "big": "exact",
            "unrecorded": "unknown_size",
        }, tiers
        by_group = dict(zip(exported["group"], exported["value"]))
        assert by_group["big"] == pytest.approx(0.42)
        assert pd.isna(by_group["tiny"]) and pd.isna(by_group["unrecorded"])
        assert any("k-anonymity could not be checked" in w for w in warned), warned

    def test_ingest_dataframe_round_trips_our_own_export_without_losing_a_state(self):
        """CONTROL for the whole ingest/export cycle: a determined breach, a
        determined clean row and an undetermined one must come back as three
        distinct states, and the health score must be identical on both sides."""
        now = datetime.now()
        source_store = MetricsStore()
        for i, flag in enumerate((True, None, False)):
            source_store._records.append(
                StoredMetricRecord(
                    timestamp=now - timedelta(hours=i),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.95,
                    group="f",
                    group_size=100,
                    alert=flag,
                )
            )
        exported, _ = _catch(source_store.to_dataframe)
        target = MetricsStore()
        n, _ = _catch(
            target.ingest_dataframe,
            exported,
            source="roundtrip",
            metric_col="metric_name",
            value_col="value",
            group_col="group",
            alert_col="alert",
            group_size_col="group_size",
        )
        assert n == 3
        determinations = [r.alert for r in target._records]
        assert determinations.count(None) == 1, determinations
        assert determinations.count(True) == 1, determinations
        assert determinations.count(False) == 1, determinations

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = source_store.compute_health_score()
            after = target.compute_health_score()
        assert after.score == before.score
        assert after.n_not_assessable == before.n_not_assessable == 1


# The ingestion paths: every one of them must carry the third state in


class TestEveryIngestionPathCarriesTheThirdState:
    """One pin per ingestion entry point in this batch.

    Each of these paths once stamped ``alert=False`` ("compared to its
    threshold and found clean") on a comparison that never happened, and the
    fixes are recorded in the module. These hold them there, and each is
    paired with the measured value the same call must still report.
    """

    @staticmethod
    def _snapshot(now: datetime):
        from vfairness.operations.monitoring.tracker import WindowMetrics

        return WindowMetrics(
            batch_id="b1",
            timestamp=now,
            sample_count=200,
            metrics={
                # Compared to its threshold and breached.
                "demographic_parity_group_gender": 0.47,
                # A custom metric: no threshold exists, so no entry in `alerts`.
                "revenue_per_user": 12.5,
            },
            group_rates={"gender": {"f": 0.29, "m": 0.76}},
            alerts={"demographic_parity_group_gender": True},
            mmd_scores={"gender": 0.08},
        )

    def test_ingest_window_metrics_keeps_an_uncompared_metric_uncompared(self):
        now = datetime.now()
        store = MetricsStore()
        n, _ = _catch(store.ingest_window_metrics, self._snapshot(now))
        # 2 metrics + 2 group rates + 1 mmd score.
        assert n == 5, n
        by_name = {r.metric_name: r for r in store._records}
        assert by_name["demographic_parity"].alert is True
        assert by_name["revenue_per_user"].alert is None, (
            "a custom metric has no threshold, so it has no determination"
        )
        assert by_name["positive_rate"].alert is None
        assert by_name["mmd_score"].alert is None
        # CONTROL: the measured numbers and the window's own sample count.
        assert by_name["demographic_parity"].value == pytest.approx(0.47)
        assert by_name["mmd_score"].value == pytest.approx(0.08)
        assert all(r.group_size == 200 for r in store._records), [
            (r.metric_name, r.group_size) for r in store._records
        ]

        # And the custom metric must not raise the score by being averaged in.
        health, warned = _catch(store.compute_health_score)
        assert health.components["metric_compliance"] == pytest.approx(0.0), health.components
        assert health.n_not_assessable == 1, health
        assert any("no alert determination" in w for w in warned), warned

    def test_ingest_from_monitor_preserves_the_history_determinations(self):
        """The bulk route must agree with the per-snapshot route, including on
        the metric NAME: both split ``demographic_parity_group_gender`` into
        the metric and the protected column."""
        now = datetime.now()

        class _Monitor:
            def get_metric_history(self):
                return pd.DataFrame(
                    [
                        {
                            "timestamp": now,
                            "batch_id": "b1",
                            "metric": "demographic_parity_group_gender",
                            "value": 0.47,
                            "sample_count": 200,
                            "alert": True,
                        },
                        {
                            "timestamp": now,
                            "batch_id": "b1",
                            # An AUTO-DETECTED protected column carries no
                            # `group_` prefix, so the key has no separator in
                            # it. The module's own docstring records that a
                            # fixture named `group_gender` passes this for the
                            # wrong reason, by supplying the separator itself.
                            "metric": "disparate_impact_age",
                            "value": 0.31,
                            "sample_count": 200,
                            "alert": True,
                        },
                        {
                            "timestamp": now,
                            "batch_id": "b1",
                            "metric": "revenue_per_user",
                            "value": 12.5,
                            "sample_count": 200,
                            "alert": None,
                        },
                    ]
                )

        store = MetricsStore()
        n, _ = _catch(store.ingest_from_monitor, _Monitor())
        assert n == 3
        by_name = {r.metric_name: r for r in store._records}
        assert by_name["demographic_parity"].alert is True
        assert by_name["demographic_parity"].group == "gender"
        assert by_name["demographic_parity"].group_size == 200
        assert by_name["disparate_impact"].group == "age", (
            "a plain column name leaves the key with no separator to find"
        )
        assert by_name["revenue_per_user"].alert is None, (
            "bool(None) is False, and False means 'compared and clean'"
        )
        assert by_name["revenue_per_user"].group == "overall", (
            "a custom metric is not a per-column reading and must not be cut apart"
        )

    def test_ingest_from_analyzer_records_values_not_verdicts(self):
        """The analyzer's daily frame has no alert column and applies no
        threshold, so nothing in it is a threshold comparison."""
        from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

        analyzer = TemporalFairnessAnalyzer()
        base = pd.Timestamp(datetime.now().date())
        for i in range(3):
            analyzer.update_daily_metrics(
                base - pd.Timedelta(days=2 - i),
                {"demographic_parity": 0.95, "disparate_impact": 0.05},
            )
        store = MetricsStore()
        n, _ = _catch(store.ingest_from_analyzer, analyzer)
        assert n == 6, n
        assert all(r.alert is None for r in store._records), [r.alert for r in store._records]
        # CONTROL: the VALUES are ingested exactly, so the refusal is about the
        # determination and not about the measurement.
        assert sorted({r.value for r in store._records}) == [0.05, 0.95]
        # And six maximally unfair readings nobody compared score nothing.
        health, warned = _catch(store.compute_health_score)
        assert health.score is None and health.status == "not_assessed", health
        assert any("NO evidence" in w for w in warned), warned

    def test_ingest_drift_result_carries_a_refusal_through(self):
        """The detector REFUSES a series too short to decompose: NaN score and
        a ``drift_detected`` of None. Both must reach the store as they are."""
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        detector = FairnessDriftDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused = detector.detect_drift_multiscale(
                pd.Series([0.1, 0.2]), metric="demographic_parity"
            )
        assert refused.drift_detected is None, "anti-vacuity: nothing was refused"

        store = MetricsStore()
        n, _ = _catch(store.ingest_drift_result, refused)
        assert n == 1
        record = store._records[0]
        assert record.alert is None, "a refused drift test produced no verdict"
        assert pd.isna(record.value), record.value
        history = store.get_drift_history()
        assert history["drift_detected"].iloc[0] is None
        assert pd.isna(history["overall_drift_score"].iloc[0])

    def test_control_ingest_from_detector_carries_a_measured_drift_through(self):
        """OVER-CORRECTION CONTROL for the same path: a drift the detector DID
        measure keeps its verdict and its score, by the bulk route."""
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        rng = np.random.default_rng(7)
        series = pd.Series(np.r_[rng.normal(0.10, 0.01, 60), rng.normal(0.50, 0.01, 60)])
        detector = FairnessDriftDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            measured = detector.detect_drift_multiscale(series, metric="demographic_parity")
        assert measured.drift_detected is True, measured

        store = MetricsStore()
        n, _ = _catch(store.ingest_from_detector, detector)
        assert n == 1
        record = store._records[0]
        assert record.alert is True
        assert record.value == pytest.approx(measured.overall_drift_score)
        assert store.get_drift_history()["drift_detected"].iloc[0]

    def test_ingest_from_prioritizer_and_get_alerts_filter_what_is_there(self):
        now = datetime.now()

        class _Prioritizer:
            def get_alert_log(self):
                return [
                    {"timestamp": now, "severity": "CRITICAL", "priority_score": 9.8},
                    {
                        "timestamp": now - timedelta(days=3),
                        "severity": "LOW",
                        "priority_score": 1.2,
                    },
                ]

        store = MetricsStore()
        n, _ = _catch(store.ingest_from_prioritizer, _Prioritizer())
        assert n == 2
        assert len(store.get_alerts()) == 2
        assert len(store.get_alerts(severity="CRITICAL")) == 1
        assert len(store.get_alerts(start_time=now - timedelta(days=1))) == 1
        assert len(store.get_alerts(start_time=now + timedelta(days=1))) == 0
        # CONTROL: the real priority scores arrive unchanged.
        values = sorted(store.get_metrics(apply_privacy=False)["value"].tolist())
        assert values == pytest.approx([1.2, 9.8])

    def test_ingest_threshold_update_records_exactly_what_it_was_given(self):
        store = MetricsStore()
        _catch(store.ingest_threshold_update, "demographic_parity", 0.12, {"n_feedback": 5})
        _catch(store.ingest_threshold_update, "disparate_impact", 0.80)
        history = store.get_threshold_history()
        assert history["threshold"].tolist() == [0.12, 0.80]
        assert history["key"].tolist() == ["demographic_parity", "disparate_impact"]
        assert history["stats"].tolist() == [{"n_feedback": 5}, {}]
        assert store.get_summary()["n_threshold_records"] == 2


class TestAnEmptyStoreSaysItIsEmpty:
    """An empty store must not answer any query with a made-up value.

    Added after a sabotage that made ``get_time_range`` return
    ``(datetime.now(), datetime.now())`` stayed GREEN: the pins above never
    queried an empty store, so the branch was never reached. A green sabotage
    is a fixture report before it is anything else.
    """

    def test_every_accessor_refuses_rather_than_invents(self):
        store = MetricsStore()
        assert store.get_time_range() == (None, None), store.get_time_range()
        assert store.get_metric_names() == []
        assert store.get_groups() == []
        assert store.get_sources() == []
        assert store.get_latest("demographic_parity") is None
        assert store.get_alerts() == []
        assert len(store.get_metrics()) == 0
        assert len(store.get_drift_history()) == 0
        assert len(store.get_threshold_history()) == 0
        assert store.prune() == 0
        summary = store.get_summary()
        assert summary["n_metric_records"] == 0
        assert summary["time_range"] == (None, None), summary

    def test_the_health_score_is_withheld_not_perfect(self):
        store = MetricsStore()
        health, warned = _catch(store.compute_health_score)
        assert health.score is None, health
        assert health.status == "not_assessed", health
        assert health.trend == "unknown", health
        assert health.components == {}, health.components
        assert any("COULD NOT CHECK" in w for w in warned), warned
        assert health.to_dict()["score"] is None


class TestTheAccessorsPickTheRightRows:
    """Two more fixture gaps a green sabotage exposed.

    ``get_groups`` was pinned on a fixture that held no ``"overall"`` row, so
    dropping the filter changed nothing; ``get_latest`` was pinned on a fixture
    whose newest record happened to be first in insertion order, so returning
    ``matches[0]`` passed. Both branches are supplied here.
    """

    def test_get_groups_excludes_the_overall_pseudo_group(self):
        now = datetime.now()
        store = MetricsStore()
        for group in ("overall", "f", "m"):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now,
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.4,
                    group=group,
                    group_size=100,
                    alert=False,
                )
            )
        assert store.get_groups() == ["f", "m"], store.get_groups()
        assert store.get_summary()["groups"] == ["f", "m"]

    def test_get_latest_picks_the_newest_not_the_first_stored(self):
        now = datetime.now()
        store = MetricsStore()
        # Deliberately stored OLDEST first, so `matches[0]` is the wrong answer.
        for hours_ago, value in ((3, 0.11), (2, 0.22), (1, 0.33)):
            store._records.append(
                StoredMetricRecord(
                    timestamp=now - timedelta(hours=hours_ago),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=value,
                    group_size=100,
                    alert=False,
                )
            )
        latest = store.get_latest("demographic_parity")
        assert latest is not None
        assert latest.value == pytest.approx(0.33), latest
        assert latest.timestamp == now - timedelta(hours=1)
