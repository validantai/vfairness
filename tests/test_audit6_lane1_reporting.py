"""Audit 6, lane 1 (monitoring, reporting, legal, CI/CD): a value nobody measured
was replaced by a neutral default and then graded, counted, ranked or reported
as if it were a measurement.

Every fix here follows one pattern: three states, never two. An unmeasured
value becomes ``None`` or NaN, leaves every aggregate, is named in a warning,
and any verdict derived from it becomes ``None`` rather than a clean-looking
literal. Each finding carries a refusal pin AND an over-correction control
proving the measured path still produces its real answer, because a fix that
refuses everything is as useless as one that passed everything.

Findings (all reproduced by execution on 2026-09-09 before the fix):

- R-2  tracker.py / store.py: ``snap.alerts.get(key, False)`` graded a custom
       metric that was never compared to a threshold as compliant. Built-ins
       both breaching scored 50.0; adding three maximally unfair custom
       metrics RAISED the health score to 71.4.
- R-8  alerts.py: an ABSENT priority factor scored 0.0 silently while a NaN
       one was excluded and warned. (11.17, CRITICAL, pagerduty) with all four
       factors; (7.67, HIGH, slack) with regulatory_risk absent, no warning.
- R-6  compliance.py: a feature missing from ``shap_values`` scored 0.0, could
       never be an adverse reason, and a PROXY entered the slot it vacated.
- R-10 compliance.py: ``_SEV_RANK.get(..., 1)`` graded ``insufficient_data``
       as a measured minor finding; it fell under the materiality cut and
       vanished from an Unqualified opinion.
- R-9  legal/admissibility.py: status ``unknown`` ranked 0, BELOW ``allowed``,
       and was counted nowhere.
- F9   cicd/monitor.py: ``log_batch(metadata=...)`` was documented, accepted
       and dropped.
- F8   cicd/validator.py: ``validate(feature_columns=...)`` was documented,
       accepted and ignored: ``['DOES_NOT_EXIST']`` gave identical metrics.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# R-2: a metric with no alert determination is could-not-check, not compliant
# ---------------------------------------------------------------------------


def _breaching_batch() -> pd.DataFrame:
    """Group A selected 90 percent, group B 10 percent: DI 0.11 and DP gap 0.8,
    so both built-in metrics breach the 0.8 four-fifths threshold."""
    return pd.DataFrame(
        {
            "prediction": np.r_[np.repeat([1, 0], [90, 10]), np.repeat([1, 0], [10, 90])],
            "label": np.r_[np.ones(100, dtype=int), np.zeros(100, dtype=int)],
            "group_gender": ["A"] * 100 + ["B"] * 100,
        }
    )


def _clean_batch() -> pd.DataFrame:
    """Both groups selected 50 percent: DI 1.0 and DP gap 0.0, nothing breaches."""
    return pd.DataFrame(
        {
            "prediction": [1, 0] * 100,
            "label": [1, 0] * 100,
            "group_gender": ["A"] * 100 + ["B"] * 100,
        }
    )


def _health(batch, custom=None, metrics_to_track=("disparate_impact", "demographic_parity")):
    from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig
    from vfairness.operations.reporting import MetricsStore

    cfg = FairnessMonitorConfig(metrics_to_track=list(metrics_to_track))
    monitor = FairnessMonitor(config=cfg, custom_metrics=custom)
    snap = monitor.update_and_check(batch)
    store = MetricsStore()
    store.ingest_window_metrics(snap)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        health = store.compute_health_score()
    return snap, health, [str(w.message) for w in caught]


_UNFAIR_CUSTOM = {f"custom_unfair_{i}": (lambda df: 1.0) for i in range(3)}


class TestR2CustomMetricsWithoutDetermination:
    def test_unfair_custom_metrics_cannot_raise_the_health_score(self):
        """The measured defect: 50.0 became 71.4 when three unfair custom
        metrics were added, because their missing determination read as
        compliant. They are now excluded, counted, and named."""
        _, baseline, _ = _health(_breaching_batch())
        snap, with_custom, caught = _health(_breaching_batch(), custom=_UNFAIR_CUSTOM)

        # BGL3 operations-4, 2026-09-27: this was 50.0, which was
        # 0.50*0.0 + 0.30*100.0 + 0.20*100.0, and the last term was drift
        # stability on an EMPTY drift table: a perfect score for a component
        # nobody measured. The store now excludes it and renormalises over the
        # two components that have evidence, giving (0.30*100.0)/0.80 = 37.5.
        # The subject of this test is unchanged, and is the line below it:
        # adding three maximally unfair custom metrics must not move the score.
        #
        # BGL5 A-operations-3, 2026-09-27: 37.5 became 0.0, and the term that
        # went is 0.30*100.0, the ALERT component, on a store whose alert channel
        # was never fed. Same defect one component over: a clean prioritizer run
        # and an unconnected alert channel both leave zero alert records, so the
        # 100.0 was a perfect default for no evidence. Excluded and renormalised
        # like drift, the composite is (0.50*0.0) / 0.50 = 0.0.
        assert baseline.score == pytest.approx(0.0)
        assert with_custom.score == pytest.approx(baseline.score), (
            f"adding unmeasured metrics moved the score from {baseline.score} to "
            f"{with_custom.score}"
        )
        assert with_custom.components["metric_compliance"] == pytest.approx(0.0)
        assert with_custom.n_not_assessable == 3
        assert any("custom_unfair_0" in w and "no alert determination" in w for w in caught), caught
        # The snapshot itself never claimed a determination for them.
        assert all(name not in snap.alerts for name in _UNFAIR_CUSTOM)

    def test_history_reports_none_not_false_for_custom_metrics(self):
        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig

        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(metrics_to_track=["disparate_impact"]),
            custom_metrics=_UNFAIR_CUSTOM,
        )
        monitor.update_and_check(_breaching_batch())
        history = monitor.get_metric_history().set_index("metric")["alert"]
        for name in _UNFAIR_CUSTOM:
            assert history[name] is None, f"{name!r} was graded {history[name]!r}, not None"
        # Scalar indexing yields np.bool_ once the column has no None in it, so
        # compare the value, not the identity.
        assert bool(history["disparate_impact_group_gender"]) is True

    def test_tri_state_survives_ingest_from_monitor(self):
        """``bool(row['alert'])`` used to collapse None into False on the way
        into the store; the third state must survive the round trip.

        The metric NAME moved on 2026-09-10 (READINESS-5) and the subject did
        not. ``ingest_from_monitor`` used to store the monitor's composite key
        ``disparate_impact_group_gender`` verbatim while ``ingest_window_metrics``
        split the same key into the metric ``disparate_impact`` on the group
        ``gender``. Both routes now split it, so the same measurement has one
        name whichever way it was ingested, and that name is one the rest of
        the library recognises.
        """
        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig
        from vfairness.operations.reporting import MetricsStore

        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(metrics_to_track=["disparate_impact"]),
            custom_metrics={"custom_x": lambda df: 0.5},
        )
        monitor.update_and_check(_breaching_batch())
        store = MetricsStore()
        store.ingest_from_monitor(monitor)
        by_name = {r.metric_name: r.alert for r in store._records}
        assert by_name["custom_x"] is None
        assert by_name["disparate_impact"] is True
        assert "disparate_impact_group_gender" not in by_name, (
            "the composite key is being stored again"
        )
        assert {r.group for r in store._records if r.metric_name == "disparate_impact"} == {
            "gender"
        }, "the group was dropped instead of being split out of the key"
        frame = store.get_metrics(apply_privacy=False).set_index("metric_name")
        # The frame keeps `alert` a strict bool for the mask readers and
        # carries the third state in `alert_determined`.
        assert (
            frame.loc["custom_x", "alert"] is np.False_ or frame.loc["custom_x", "alert"] is False
        )
        assert not bool(frame.loc["custom_x", "alert_determined"])
        assert bool(frame.loc["disparate_impact", "alert_determined"])

    def test_only_undetermined_records_withholds_the_score(self):
        """A window holding nothing that was compared to a threshold cannot
        certify anything: score None, status not_assessed, with a warning."""
        _, health, caught = _health(
            _clean_batch(), custom={"custom_x": lambda df: 0.5}, metrics_to_track=()
        )
        assert health.score is None
        assert health.status == "not_assessed"
        assert health.n_not_assessable == 1
        assert any("no health score was computed" in w for w in caught), caught

    def test_descriptive_rows_are_not_compliance_evidence(self):
        """Per-group positive rates carry no determination either, and used to
        dilute the breach fraction as compliant rows. They now leave the
        population silently: no warning, not counted as unassessable."""
        _, health, caught = _health(_breaching_batch())
        assert health.n_not_assessable == 0
        assert not any("positive_rate" in w for w in caught), caught
        # Two of two threshold-compared metrics breach: compliance is 0, not
        # the diluted two-of-four.
        assert health.components["metric_compliance"] == pytest.approx(0.0)

        # The clamp, metric_score = max(0.0, 100 * (1 - 2 * alert_frac)),
        # saturates at 0 once half the rows breach, so the two-breach window
        # above reads 0.0 whether the positive rates dilute the mean or not.
        # One breach among the determined rows can tell the difference.
        _, one, _ = _health(_breaching_batch(), metrics_to_track=("disparate_impact",))
        assert one.components["metric_compliance"] == pytest.approx(0.0), (
            "the two per-group positive rates diluted the single breaching metric: "
            f"{one.components['metric_compliance']}"
        )

    def test_mmd_rows_carry_no_determination(self):
        """An MMD distribution-shift score is never compared to a threshold
        either. No other fixture here sets a reference window, so nothing
        else in this class reaches the MMD branch at all."""
        from vfairness.operations.monitoring import FairnessMonitor, FairnessMonitorConfig
        from vfairness.operations.reporting import MetricsStore

        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(metrics_to_track=["disparate_impact"])
        )
        monitor.set_reference(_clean_batch())
        snap = monitor.update_and_check(_breaching_batch())
        assert snap.mmd_scores, "the fixture produced no MMD score to check"
        store = MetricsStore()
        store.ingest_window_metrics(snap)
        mmd = [r.alert for r in store._records if r.metric_name == "mmd_score"]
        assert mmd and all(a is None for a in mmd), mmd

    def test_undetermined_rows_in_the_previous_window_do_not_fake_a_decline(self):
        """The trend baseline gets the same exclusion as the current window, or
        a previous window padded with unmeasured metrics manufactures a
        decline."""
        import inspect
        from datetime import datetime, timedelta

        from vfairness.operations.reporting import MetricsStore
        from vfairness.operations.reporting.store import StoredMetricRecord

        # Bind the window ONCE from the function under test, so the fixture and
        # the function can never drift apart. Hardcoding 8/14/7 here made this
        # pin vacuous: at a 14 day default the rows fell into the CURRENT
        # window and the test passed with the fix fully removed.
        window = (
            inspect.signature(MetricsStore.compute_health_score).parameters["time_window"].default
        )
        now = datetime.now()
        prev = now - (window + timedelta(days=1))
        store = MetricsStore()
        store._records.append(
            StoredMetricRecord(
                timestamp=prev,
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.9,
                alert=True,
            )
        )
        for i in range(3):
            store._records.append(
                StoredMetricRecord(
                    timestamp=prev,
                    source="FairnessMonitor",
                    metric_name=f"custom_{i}",
                    value=1.0,
                    alert=None,
                )
            )
        store._records.append(
            StoredMetricRecord(
                timestamp=now,
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.9,
                alert=True,
            )
        )
        # Anti-vacuity, derived from the same bound window as the fixture.
        prev_rows = store.get_metrics(
            start_time=now - 2 * window,
            end_time=now - window,
            apply_privacy=False,
        )
        assert len(prev_rows) == 4, f"fixture missed the previous window: {len(prev_rows)}"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            health = store.compute_health_score(time_window=window)
        assert health.trend == "stable", health.trend
        assert health.trend_slope == pytest.approx(0.0)

    def test_an_untracked_metric_is_not_reported_with_a_mean_of_zero(self):
        """summary.get('mean', 0) printed 'Mean = 0.0000' for a metric that was
        never tracked: a fabricated statistic of perfect equality, beside an
        n_days that honestly read '?'."""
        from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

        analyzer = TemporalFairnessAnalyzer()
        analyzer.update_daily_metrics(pd.Timestamp("2026-01-01"), {"demographic_parity": 0.30})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.get_explanation("equal_opportunity")
        assert "0.0000" not in report.summary, report.summary
        assert "mean=not measured" in report.summary, report.summary
        assert "Mean = not measured, std = not measured" in report.explanations[0].evaluation

    def test_control_clean_built_ins_still_score_green(self):
        """Over-correction control: a measured, clean window still scores."""
        snap, health, caught = _health(_clean_batch())
        assert snap.alerts == {
            "disparate_impact_group_gender": False,
            "demographic_parity_group_gender": False,
        }
        assert health.score == pytest.approx(100.0)
        assert health.status == "green"
        assert health.n_not_assessable == 0
        assert not any("no alert determination" in w for w in caught), caught

    def test_control_measured_custom_beside_clean_built_ins(self):
        """A custom metric beside clean built-ins does not withhold the score
        and does not change it; it is disclosed beside it."""
        _, health, caught = _health(_clean_batch(), custom={"custom_x": lambda df: 0.9})
        assert health.score == pytest.approx(100.0)
        assert health.n_not_assessable == 1
        assert "1 metric record could not be assessed" in health.explanation
        assert health.to_dict()["n_not_assessable"] == 1
        assert any("custom_x" in w for w in caught), caught

    def test_control_hand_built_determined_rows_unchanged(self):
        """Records that carry a real determination score exactly as before
        (the wave-4 arithmetic: one of two breached gives compliance 0)."""
        from datetime import datetime

        from vfairness.operations.reporting import MetricsStore
        from vfairness.operations.reporting.store import StoredMetricRecord

        store = MetricsStore()
        for flag in (True, False):
            store._records.append(
                StoredMetricRecord(
                    timestamp=datetime.now(),
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.2,
                    alert=flag,
                )
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            health = store.compute_health_score()
        assert health.components["metric_compliance"] == pytest.approx(0.0)
        assert health.n_not_assessable == 0
        # BGL3 operations-4, 2026-09-27: this was `caught == []`. This store has
        # no drift row, and the store now says so instead of scoring that
        # component 100.0. The subject here is that a fully determined pair of
        # rows raises NO complaint about its own determination or measurability,
        # so that is what is asserted; the drift-coverage disclosure is a
        # different statement about a different component.
        #
        # BGL5 A-operations-3, 2026-09-27: the alert-coverage disclosure joins it,
        # for the same reason and about the same kind of absence (this store never
        # fed the alert channel either), so it is filtered on the same grounds.
        assert [
            str(w.message)
            for w in caught
            if "drift" not in str(w.message) and "alert record" not in str(w.message)
        ] == []

    def test_control_a_tracked_metric_still_reports_its_real_mean(self):
        """Over-correction control: a metric with real data renders exactly as
        it did before, to four decimals."""
        from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

        analyzer = TemporalFairnessAnalyzer(lookback_days=90)
        start = pd.Timestamp("2026-01-05")
        for day in range(28):
            analyzer.update_daily_metrics(
                start + pd.Timedelta(days=day),
                {"demographic_parity": 0.05 + day * 0.0005},
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.get_explanation("demographic_parity")
        assert report.summary == ("demographic_parity over 28 days: mean=0.0568, trend=increasing.")
        assert report.explanations[0].evaluation == (
            "Trend is increasing with slope +0.000500/day. Mean = 0.0568, std = 0.0041."
        )


# ---------------------------------------------------------------------------
# R-8: an absent priority factor is the same absence as a NaN one
# ---------------------------------------------------------------------------

_FULL_EVENT = {
    "regulatory_risk": 1.0,
    "population_impact": 0.8,
    "drift_velocity": 0.9,
    "historical_discrimination": 1.0,
    "drift_score": 0.82,
}


def _priority(event):
    from vfairness.operations.monitoring import FairnessAlertPrioritizer

    prioritizer = FairnessAlertPrioritizer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score, severity = prioritizer.calculate_priority(event)
    return score, severity, [str(w.message) for w in caught]


class TestR8AbsentFactor:
    def test_absent_factor_takes_the_nan_path(self):
        """Measured before the fix: absent gave (7.67, HIGH) with NO warning,
        NaN gave (7.67, HIGH) WITH one. Same absence, same path now."""
        absent = {k: v for k, v in _FULL_EVENT.items() if k != "regulatory_risk"}
        nan = {**_FULL_EVENT, "regulatory_risk": float("nan")}

        s_absent, sev_absent, w_absent = _priority(absent)
        s_nan, sev_nan, w_nan = _priority(nan)

        assert s_absent == pytest.approx(s_nan)
        assert sev_absent == sev_nan
        assert any("regulatory_risk" in w for w in w_absent), (
            "an absent factor was scored 0.0 silently"
        )
        assert any("regulatory_risk" in w for w in w_nan)
        # The NaN path itself is unchanged: excluded, warned, scored from the rest.
        assert s_nan == pytest.approx(7.67, abs=0.01)
        assert sev_nan == "HIGH"

    def test_present_none_factor_is_absent(self):
        score, severity, caught = _priority({**_FULL_EVENT, "regulatory_risk": None})
        assert score == pytest.approx(7.67, abs=0.01)
        assert any("regulatory_risk" in w for w in caught), "None was scored as 0.0 silently"

    def test_all_weighted_factors_absent_is_reported(self):
        """Measured on HEAD: an event carrying only drift_score scored 0.82,
        banded LOW and routed to jira with ZERO warnings, indistinguishable
        from a genuinely calm measured event.

        BGL S2, 2026-09-16: R-8 added the warning and left the SCORE as the
        boost alone, so the returned number was still a finite score in a
        measured band, built from zero of the four defined severity factors.
        The subject of this test is unchanged (all four absences are named in
        one warning); only the value assertion moves to the third state.
        """
        score, severity, caught = _priority({"drift_score": 0.82})
        assert math.isnan(score), f"a score from zero measured factors, got {score!r}"
        assert severity == "UNSCORED"
        assert len(caught) == 1
        for factor in (
            "regulatory_risk",
            "population_impact",
            "drift_velocity",
            "historical_discrimination",
        ):
            assert factor in caught[0], (factor, caught)

    def test_control_full_event_unchanged(self):
        """Over-correction control: all four factors present still gives the
        measured (11.17, CRITICAL) with no warning.

        The second half of this test asserted the OPPOSITE of what it should have,
        and it asserted it as a control, which is why nothing caught it. It read
        "the helper-built event, which always carries all four, is untouched" and
        then pinned (5.1, HIGH) with no warning. The helper did not carry four
        measured factors: it SUBSTITUTED 0.5, 0.3, 0.3 and 0.5 for four
        quantities nobody had measured, and this test held that substitution in
        place. build_drift_event's own docstring said "DEFECT NOT CLOSED, disclosed
        only", and an independent audit on 2026-09-27 overturned its PROVEN grade
        for exactly this.

        The factors are now OMITTED and named rather than invented, so the built
        event scores what an unscored event should: (nan, UNSCORED), with the absent
        factors named twice, once where they were not supplied and once where they
        could not be scored. Measured after the change, verbatim:

            build_drift_event("dp", ["B"], 0.5, 0.1)
              -> keys: affected_groups, drift_score, intersectional, mean_shift,
                 metric_name          (not one of the four)
              -> warns: "context factor(s) ['regulatory_risk', ...]"
            calculate_priority(built) -> (nan, 'UNSCORED')
              -> warns: "Alert factor(s) [...] could not be scored"

        The first half is untouched, and it is the control that matters: a fully
        supplied event still gets its real number in silence, so the fix withdrew a
        fabrication rather than refusing everything.
        """
        from vfairness.operations.monitoring import FairnessAlertPrioritizer

        score, severity, caught = _priority(_FULL_EVENT)
        assert score == pytest.approx(11.17, abs=0.01)
        assert severity == "CRITICAL"
        assert caught == []

        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as built_caught:
            _warnings.simplefilter("always")
            built = FairnessAlertPrioritizer.build_drift_event("dp", ["B"], 0.5, 0.1)
        for factor in (
            "regulatory_risk",
            "population_impact",
            "drift_velocity",
            "historical_discrimination",
        ):
            assert factor not in built, (
                f"build_drift_event invented {factor!r}; the four context factors are "
                "not inputs it has and must be absent rather than defaulted"
            )
        assert built_caught, "the absent factors were omitted in silence"
        # Asserted on the FACTOR NAMES rather than on a quoted sentence. A first
        # version of this looked for "could not be scored", a phrase neither
        # warning contains, and the test failed for that reason rather than for
        # its subject. A pin located by quoting prose is a pin on the prose.
        build_msg = str(built_caught[0].message)
        assert all(
            f in build_msg
            for f in (
                "regulatory_risk",
                "population_impact",
                "drift_velocity",
                "historical_discrimination",
            )
        ), build_msg

        s_built, sev_built, w_built = _priority(built)
        assert sev_built == "UNSCORED", (s_built, sev_built)
        assert s_built != s_built, f"an unscored event returned the number {s_built}"
        score_msg = " ".join(w_built)
        assert "UNSCORED" in score_msg and "NOT been graded harmless" in score_msg, w_built

    def test_control_calm_event_still_low(self):
        calm = {k: 0.0 for k in ("regulatory_risk", "population_impact", "drift_velocity")}
        calm["historical_discrimination"] = 0.0
        calm["drift_score"] = 0.0
        score, severity, caught = _priority(calm)
        assert score == 0.0 and severity == "LOW"
        assert caught == []


# ---------------------------------------------------------------------------
# R-6: a feature with no attribution cannot be ranked on an adverse action notice
# ---------------------------------------------------------------------------

_FEATURES = ["debt_ratio", "income", "credit_score", "employment", "zip_code", "tenure"]
_SHAP_FULL = {
    "debt_ratio": -0.4,
    "income": -0.3,
    "credit_score": -0.2,
    "employment": -0.1,
    "zip_code": -0.25,
    "tenure": 0.05,
}


def _reasons(shap, proxies=("zip_code",)):
    from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reasons = compute_adverse_action_reasons(shap, _FEATURES, proxy_features=list(proxies))
    return reasons, [str(w.message) for w in caught]


class TestR6UnattributedFeature:
    def test_unattributed_feature_is_not_ranked_and_the_notice_says_so(self):
        """Measured before the fix: with debt_ratio (the strongest adverse
        factor) unmeasured, income became the top reason, zip_code, a proxy,
        entered at RC04, and nothing said a feature was missing."""
        shap = {k: v for k, v in _SHAP_FULL.items() if k != "debt_ratio"}
        reasons, caught = _reasons(shap)

        assert "debt_ratio" not in [r["feature"] for r in reasons]
        assert reasons.unattributed_features == ["debt_ratio"]
        assert reasons.complete is False
        assert "INCOMPLETE" in repr(reasons)
        assert any("debt_ratio" in w and "could not be ranked" in w for w in caught), caught
        # No entry was ever fabricated from the unmeasured feature.
        assert all(r["contribution_score"] > 0 for r in reasons)

    @pytest.mark.parametrize("bad", [float("nan"), None])
    def test_nan_and_none_attribution_are_unattributed(self, bad):
        reasons, caught = _reasons({**_SHAP_FULL, "debt_ratio": bad})
        assert reasons.unattributed_features == ["debt_ratio"]
        assert "debt_ratio" not in [r["feature"] for r in reasons]
        assert caught

    def test_control_fully_attributed_notice_unchanged(self):
        """Over-correction control: every feature attributed gives exactly the
        notice it always gave, complete, with no warning, and still a list."""
        reasons, caught = _reasons(_SHAP_FULL)
        assert [(r["code"], r["feature"]) for r in reasons] == [
            ("RC01", "debt_ratio"),
            ("RC02", "income"),
            ("RC03", "credit_score"),
            ("RC04", "employment"),
        ]
        assert reasons.complete is True
        assert reasons.unattributed_features == []
        assert caught == []
        assert isinstance(reasons, list) and len(reasons) == 4
        assert "INCOMPLETE" not in repr(reasons)

    def test_control_positive_attribution_is_attributed_but_not_adverse(self):
        """A measured positive contribution (tenure, +0.05) is a real
        measurement that simply is not adverse: neither ranked nor 'unattributed'."""
        reasons, _ = _reasons(_SHAP_FULL)
        assert "tenure" not in reasons.unattributed_features
        assert "tenure" not in [r["feature"] for r in reasons]

    def test_control_measured_zero_attribution_is_attributed_not_unattributed(self):
        """Over-correction control: a SHAP value measured at exactly 0.0 is a
        real measurement that happens not to be adverse. It must not be read
        back as 'unattributed' just because 0.0 was the old neutral default."""
        reasons, caught = _reasons({**_SHAP_FULL, "debt_ratio": 0.0})
        assert reasons.unattributed_features == []
        assert reasons.complete is True
        assert caught == []
        assert "debt_ratio" not in [r["feature"] for r in reasons]


# ---------------------------------------------------------------------------
# R-10: insufficient_data is its own reported state, never a minor measurement
# ---------------------------------------------------------------------------


def _verdict(bias, per_variable=None, has_truth=True):
    from vfairness.operations.reporting import build_assurance_verdict

    return build_assurance_verdict(
        schema={"pii_leakage": [], "mismatches": [], "refuse": False},
        per_variable=per_variable
        if per_variable is not None
        else [{"attribute": "gender", "assessable": True, "gap": 0.0, "significant": False}],
        bias=bias,
        domain="hiring",
        jurisdiction="eu",
        has_truth=has_truth,
    )


def _rep(severity, attribute="ethnicity", **extra):
    entry = {
        "type": "representation",
        "severity": severity,
        "attribute": attribute,
        "plain": f"Representation for '{attribute}': {severity}.",
        "statisticalTest": None,
    }
    entry.update(extra)
    return entry


class TestR10InsufficientData:
    def test_sev_returns_none_for_unassessed_and_unknown_words(self):
        from vfairness.operations.reporting.compliance import _sev
        from vfairness.preprocessing.bias_detection.representation import RepresentationSeverity

        assert _sev("insufficient_data") is None
        assert _sev(RepresentationSeverity.INSUFFICIENT_DATA) is None
        assert _sev("not_a_severity_word") is None, "a word the grader does not know is not minor"

    def test_sev_grades_measured_words(self):
        """Over-correction control: measured words keep their ranks, and the
        measured RepresentationSeverity words that used to fall to the
        default are now graded as what they are."""
        from vfairness.operations.reporting.compliance import _sev
        from vfairness.preprocessing.bias_detection.representation import RepresentationSeverity

        assert _sev("critical") == 4 and _sev("high") == 3
        assert _sev("warn") == 2 and _sev("medium") == 2
        assert _sev("low") == 1 and _sev("info") == 0
        assert _sev(RepresentationSeverity.ADEQUATE) == 0
        assert _sev(RepresentationSeverity.OVERREPRESENTED) == 1

    def test_insufficient_data_is_reported_not_dropped(self):
        """Measured before the fix: overall Unqualified, findings [], one-liner
        "no material fairness defect found". The unassessed attribute vanished."""
        v = _verdict([_rep("insufficient_data")])

        unassessed = [f for f in v["findings"] if f.get("assessed") is False]
        assert len(unassessed) == 1, v["findings"]
        f = unassessed[0]
        assert f["type"] == "representation_bias_unassessed"
        assert f["severity"] == "insufficient_data", "the state is kept verbatim, not re-graded"
        assert f["attribute"] == "ethnicity"
        assert v["unassessed"] and v["unassessed"][0]["id"] == f["id"]

        assert v["overall"] != "Unqualified", (
            "an opinion built on unassessed evidence is not unqualified"
        )
        assert v["overall"] == "Qualified"
        assert v["blocksDeployment"] is False
        assert "could not be assessed" in v["oneLineVerdict"]
        assert "ethnicity" in v["oneLineVerdict"]
        assert any(f["id"] in r["addressesFindings"] for r in v["recommendations"])

    def test_control_adequate_representation_stays_unqualified(self):
        """Over-correction control: a measured clean representation verdict
        still yields an Unqualified opinion with nothing unassessed."""
        v = _verdict([_rep("adequate")])
        assert v["overall"] == "Unqualified"
        assert v["unassessed"] == []
        assert v["findings"] == []
        assert "Scope limitation" not in v["oneLineVerdict"]

    def test_control_disclaimer_keeps_its_category_and_gains_the_limitation(self):
        """A scope limitation is added to a Disclaimer, never used to overwrite it."""
        v = _verdict(
            [_rep("insufficient_data")],
            per_variable=[{"attribute": "gender", "assessable": False}],
            has_truth=False,
        )
        assert v["overall"] == "Disclaimer"
        assert "Insufficient assessable data" in v["oneLineVerdict"]
        assert "Scope limitation" in v["oneLineVerdict"] and "ethnicity" in v["oneLineVerdict"]

    def test_control_low_finding_still_falls_under_the_materiality_cut(self):
        v = _verdict([_rep("low")])
        assert v["overall"] == "Unqualified"
        assert v["unassessed"] == [] and v["findings"] == []

    def test_control_confirmed_critical_still_adverse_and_unassessed_still_reported(self):
        """The worst-of-type roll-up ignores unassessed entries: a confirmed
        critical beside an insufficient_data entry still gives Adverse, and
        the unassessed attribute is still reported beside it."""
        v = _verdict(
            [
                _rep("critical", attribute="gender", significant=True),
                _rep("insufficient_data", attribute="ethnicity"),
            ]
        )
        assert v["overall"] == "Adverse" and v["blocksDeployment"] is True
        types = sorted(f["type"] for f in v["findings"])
        assert types == ["representation_bias", "representation_bias_unassessed"]
        assert "Scope limitation" in v["oneLineVerdict"]
        assert "ethnicity" in v["oneLineVerdict"]

    def test_unassessed_entry_never_displaces_a_measured_one_of_its_type(self):
        """Order independence: the unassessed entry listed first must not be
        taken as the 'worst' representation finding and shadow the real one."""
        v = _verdict(
            [
                _rep("insufficient_data", attribute="ethnicity"),
                _rep("high", attribute="gender", significant=True),
            ]
        )
        measured = [f for f in v["findings"] if f["type"] == "representation_bias"]
        assert len(measured) == 1 and measured[0]["severity"] == "high"
        assert v["overall"] == "Qualified"


# ---------------------------------------------------------------------------
# R-9: an undetermined legal status is a fourth state that outranks allowed
# ---------------------------------------------------------------------------

_UNDETERMINED_PACK = {
    "source_revision": "test",
    "rules": {
        "age": {"status": "prohibited", "legal_basis": "status word outside the vocabulary"},
        "gender": {"legal_basis": "curated entry with no status key at all"},
        "criminal_record": {"status": "allowed", "legal_basis": "real"},
    },
}


class TestR9UnknownLegalStatus:
    """Real rule-pack data: in lending / us-federal, ``age`` is restricted,
    ``gender`` monitoring_only, ``criminal`` (criminal_record) allowed, and
    ``dob`` maps to an attribute the pack has no entry for."""

    def _classify(self):
        from vfairness.legal import classify_columns

        return classify_columns(["criminal", "dob", "gender", "age"], "lending", "US federal")

    def test_unknown_sorts_above_allowed_and_below_monitoring_only(self):
        result = self._classify()
        assert result["coverage"] == "covered"
        order = [(f["column"], f["status"]) for f in result["findings"]]
        assert order == [
            ("age", "restricted"),
            ("gender", "monitoring_only"),
            ("dob", "unknown"),
            ("criminal", "allowed"),
        ], order

    def test_unknown_has_its_own_counter(self):
        summary = self._classify()["summary"]
        assert summary == {
            "forbidden": 0,
            "restricted": 1,
            "monitoringOnly": 1,
            "allowed": 1,
            "unknown": 1,
        }

    def test_priority_table_is_strictly_ordered(self):
        from vfairness.legal.admissibility import _STATUS_PRIORITY as P

        assert P["forbidden"] > P["restricted"] > P["monitoring_only"] > P["unknown"] > P["allowed"]
        assert P["allowed"] > P["info"]

    def test_uncovered_pair_counts_every_mapped_column_as_unknown(self):
        """With no rule pack for the pair, every sensitive-looking column's
        legal status is undetermined. Before the fix those findings existed
        with status 'unknown' and the summary said nothing about them."""
        from vfairness.legal import classify_columns

        result = classify_columns(["age", "gender"], "lending", None)
        assert result["coverage"] == "uncovered"
        assert [f["status"] for f in result["findings"]] == ["unknown", "unknown"]
        assert result["summary"] == {
            "forbidden": 0,
            "restricted": 0,
            "monitoringOnly": 0,
            "allowed": 0,
            "unknown": 2,
        }

    def test_control_unmappable_columns_count_nowhere(self):
        """Over-correction control: a column that maps to no attribute is
        reported unmapped, not counted as unknown; every counter stays zero."""
        from vfairness.legal import classify_columns

        result = classify_columns(["zzz_nonsense_column_42"], "lending", "US federal")
        assert result["unmappedColumns"] == ["zzz_nonsense_column_42"]
        assert result["findings"] == []
        assert result["summary"] == {
            "forbidden": 0,
            "restricted": 0,
            "monitoringOnly": 0,
            "allowed": 0,
            "unknown": 0,
        }

    def test_undetermined_status_in_a_pack_ranks_and_counts_as_unknown(self, monkeypatch):
        """Two ways a curated pack can leave a status undetermined: a word the
        code does not know, and an entry that omits "status" (which becomes
        "unknown" via the .get default). Both must outrank "allowed" and both
        must reach the unknown counter, never the rank 0 floor and never no
        counter at all.
        """
        from vfairness.legal import admissibility

        monkeypatch.setattr(
            admissibility,
            "load_rules",
            lambda use_case, jurisdiction: _UNDETERMINED_PACK if use_case == "lending" else None,
        )
        result = admissibility.classify_columns(
            ["criminal", "age", "gender"], "lending", "US federal"
        )
        assert [(f["column"], f["status"]) for f in result["findings"]] == [
            ("age", "prohibited"),
            ("gender", "unknown"),
            ("criminal", "allowed"),
        ]
        # unknown must be 2, not 1: removing EITHER half of the
        # `status == "unknown" or status not in _KNOWN_STATUSES` condition
        # leaves one of the two undetermined columns counted nowhere.
        assert result["summary"]["unknown"] == 2
        assert result["summary"]["allowed"] == 1


# ---------------------------------------------------------------------------
# F9: log_batch carries the metadata it accepts
# ---------------------------------------------------------------------------


def _monitor_batch(metadata=None):
    from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig

    monitor = BiasMonitor(config=MonitorConfig(min_samples_for_alert=1))
    y_pred = np.array([1, 0] * 50)
    y_true = np.array([1, 0] * 50)
    attr = np.array(["A", "B"] * 50)
    return monitor.log_batch(y_pred, y_true, attr, batch_id="b1", metadata=metadata)


class TestF9MetadataCarried:
    def test_metadata_is_stored_and_serialised(self):
        """A non-string and a nested value are in the fixture on purpose: a
        lossy carry, ``{str(k): str(v) for ...}``, records 12345 as '12345'
        and hides behind a fixture whose values are all strings already."""
        result = _monitor_batch(
            {"model_version": "v2.3.1", "slice": "eu", "rows": 12345, "run": {"id": 42}}
        )
        expected = {"model_version": "v2.3.1", "slice": "eu", "rows": 12345, "run": {"id": 42}}
        assert result.metadata == expected
        assert result.to_dict()["metadata"] == expected

    def test_metadata_is_copied_not_aliased(self):
        supplied = {"model_version": "v1"}
        result = _monitor_batch(supplied)
        supplied["model_version"] = "rewritten"
        assert result.metadata == {"model_version": "v1"}

    def test_control_absent_metadata_is_an_empty_record(self):
        result = _monitor_batch(None)
        assert result.metadata == {}
        assert result.to_dict()["metadata"] == {}


# ---------------------------------------------------------------------------
# F8: feature_columns scopes the checks or refuses
# ---------------------------------------------------------------------------


def _validation_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "gender": rng.choice(["m", "f"], n),
            "income": rng.normal(50, 10, n),
            "score": rng.normal(0, 1, n),
            "constant_col": 1,
            "approved": rng.integers(0, 2, n),
        }
    )


class TestF8FeatureColumns:
    def test_nonexistent_column_is_refused(self):
        from vfairness.operations.cicd.validator import DataBiasValidator

        validator = DataBiasValidator(protected_attributes=["gender"])
        with pytest.raises(ValueError, match="DOES_NOT_EXIST"):
            validator.validate(
                _validation_frame(), outcome_column="approved", feature_columns=["DOES_NOT_EXIST"]
            )

    def test_scope_restricts_the_column_level_checks(self):
        from vfairness.operations.cicd.validator import DataBiasValidator

        validator = DataBiasValidator(protected_attributes=["gender"])
        df = _validation_frame()
        scoped = validator.validate(df, outcome_column="approved", feature_columns=["income"])
        assert scoped.metrics["feature_columns"] == ["income"]
        # gender + approved + income
        assert scoped.metrics["n_features"] == 3
        assert scoped.metrics["data_hygiene"]["constant_columns"] == []
        assert not any(i.issue_type == "constant_columns" for i in scoped.issues)

        in_scope = validator.validate(
            df, outcome_column="approved", feature_columns=["income", "constant_col"]
        )
        assert in_scope.metrics["data_hygiene"]["constant_columns"] == ["constant_col"]
        assert any(i.issue_type == "constant_columns" for i in in_scope.issues)

    def test_scope_changes_columns_only_never_rows(self):
        from vfairness.operations.cicd.validator import DataBiasValidator

        validator = DataBiasValidator(protected_attributes=["gender"])
        # Missing values in two columns, so that any row filter smuggled into
        # the scoping step (dropna, head, sample) moves the numbers below. On a
        # frame with no NaN a dropna scoping bug is invisible.
        df = _validation_frame()
        df.loc[df.index[:40], "score"] = np.nan
        df.loc[df.index[:10], "income"] = np.nan
        full = validator.validate(df, outcome_column="approved")
        scoped = validator.validate(df, outcome_column="approved", feature_columns=["income"])
        # Scoping picks which COLUMNS are checked. It must never change how many
        # ROWS were measured, nor the protected-attribute and outcome numbers,
        # which come from columns that are always in scope.
        assert scoped.metrics["n_samples"] == full.metrics["n_samples"] == len(df)
        assert scoped.metrics["representation"] == full.metrics["representation"]
        assert scoped.metrics["outcome_disparity"] == full.metrics["outcome_disparity"]
        assert scoped.metrics["label_quality"] == full.metrics["label_quality"]
        # ABSOLUTE anchor, not just scoped-vs-full. A defect that moves BOTH
        # paths equally is invisible to an equality check: measured, replacing
        # the representation count with df[attr].head(len(df) // 2) left every
        # comparison above green while reporting half the dataset as the whole.
        # These counts are fixed by default_rng(0) at n=300 and sum to len(df).
        assert full.metrics["representation"]["gender"]["counts"] == {"f": 163, "m": 137}
        assert sum(full.metrics["representation"]["gender"]["counts"].values()) == len(df)

    def test_naming_an_always_in_scope_column_is_harmless(self):
        from vfairness.operations.cicd.validator import DataBiasValidator

        validator = DataBiasValidator(protected_attributes=["gender"])
        df = _validation_frame()
        # Naming the protected attribute and the outcome inside the scope is the
        # most natural call there is. It must not duplicate a column: a doubled
        # 'gender' makes df['gender'] two-dimensional and the group-by dies with
        # "Grouper for 'gender' not 1-dimensional".
        both = validator.validate(
            df, outcome_column="approved", feature_columns=["gender", "approved", "income"]
        )
        assert both.metrics["n_features"] == 3
        assert both.metrics["feature_columns"] == ["gender", "approved", "income"]
        assert (
            both.metrics["representation"]
            == validator.validate(df, outcome_column="approved").metrics["representation"]
        )

    def test_control_none_checks_every_column_as_before(self):
        from vfairness.operations.cicd.validator import DataBiasValidator

        validator = DataBiasValidator(protected_attributes=["gender"])
        df = _validation_frame()
        full = validator.validate(df, outcome_column="approved")
        assert "feature_columns" not in full.metrics
        assert full.metrics["n_features"] == len(df.columns)
        assert full.metrics["data_hygiene"]["constant_columns"] == ["constant_col"]
        assert full.execution_coverage() == "complete"
        # And the finite-score sanity of the unscoped path is untouched.
        assert math.isfinite(full.metrics["label_quality"]["positive_rate"])
