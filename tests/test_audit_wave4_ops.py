"""Audit wave 4 regression tests: CI/CD + monitoring + reporting ops.

Pins the fixes for the adversarially confirmed findings in:
- operations/cicd/gate.py (degradation check, per-intersection thresholds)
- operations/cicd/validator.py (class-imbalance only on binary outcomes)
- operations/monitoring/drift.py (baseline-boundary split, MMD windows)
- operations/monitoring/tracker.py (weekly degradation direction)
- operations/reporting/compliance.py (ECOA cap at 4, unique codes)
- operations/reporting/interactive.py (offline standalone HTML)
- operations/reporting/store.py (no alert double penalty in health score)
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.gate import (
    GateConfig,
    HierarchicalGateConfig,
    ModelFairnessGate,
)
from vfairness.operations.cicd.validator import DataBiasValidator
from vfairness.operations.monitoring.drift import (
    FairnessDriftDetector,
    mmd_gaussian,
)
from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer
from vfairness.operations.reporting.compliance import (
    AdverseActionReasons,
    compute_adverse_action_reasons,
)
from vfairness.operations.reporting.store import MetricsStore

# gate.py finding 1: evaluate_from_metrics must enforce the degradation margin


class TestEvaluateFromMetricsDegradation:
    """READINESS-6, 2026-09-10: the fixture below used to read
    ``GateConfig(metrics=["m"], thresholds={}, ...)``.

    That isolated the degradation check by leaving the metric UNBOUNDED, which
    was only workable because an unbounded metric silently passed the threshold
    step (``threshold = self.config.thresholds.get(name)`` -> None -> the
    comparison was skipped and ``passed = True`` was left standing). That is
    the defect these tests now sit next to: measured on data with TPR 1.00 vs
    0.10 and FPR 0.00 vs 0.90, the same shape of config APPROVED a model whose
    equalized_odds_difference was 0.90. A metric with no threshold is now
    could-not-check and refuses approval, so the old fixture asserted a pass
    the gate must never give.

    The isolation is preserved honestly instead: a REAL metric name with a
    threshold loose enough (0.5) that every value below clears it, so the only
    thing that can move the verdict is still the degradation margin. The
    placeholder name "m" is gone on purpose -- its direction is UNKNOWN, and an
    unknown metric now fails closed at the threshold check by design.
    """

    METRIC = "demographic_parity_difference"

    def _config(self, margin=0.02):
        return GateConfig(
            metrics=[self.METRIC],
            thresholds={self.METRIC: 0.5},
            allow_degradation_margin=margin,
        )

    def test_degradation_beyond_margin_blocks(self):
        gate = ModelFairnessGate(config=self._config())
        decision = gate.evaluate_from_metrics(
            {self.METRIC: 0.30}, baseline_metrics={self.METRIC: 0.05}
        )
        assert not decision.approved
        assert any("degraded" in r for r in decision.blocking_reasons)

    def test_degradation_within_margin_passes(self):
        gate = ModelFairnessGate(config=self._config())
        decision = gate.evaluate_from_metrics(
            {self.METRIC: 0.06}, baseline_metrics={self.METRIC: 0.05}
        )
        assert decision.approved

    def test_no_baseline_no_degradation_check(self):
        """W2 A-operations-1, 2026-09-29: the SUBJECT of this test is kept and its
        EXPECTED VERDICT is corrected, because the verdict it asserted was the
        defect.

        The subject still holds: with no baseline the degradation COMPARISON does
        not run, so no "degraded beyond allowed margin" reason is produced. What
        was wrong was `assert decision.approved`. This config carries
        allow_degradation_margin=0.02, a configured requirement, and supplying no
        baseline means it could not be checked. Measured on HEAD before the fix,
        exactly this call returned approved=True, "APPROVED - All fairness
        requirements met", blocking=[] warnings=[], improvement=None and
        create_github_check conclusion 'success': an unchecked requirement
        published as a met one. The sibling test for a NaN baseline
        (test_the_degradation_bound_alone_is_enough_to_require_the_baseline, in
        tests/test_bgl5_operations_1.py) has asserted the fail-closed outcome since
        2026-09-27; an ABSENT baseline is the same could-not-check and now takes
        the same path.
        """
        gate = ModelFairnessGate(config=self._config())
        decision = gate.evaluate_from_metrics({self.METRIC: 0.30})
        assert not decision.approved
        # The comparison itself still did not run: that is this test's subject.
        assert not any("degraded" in r for r in decision.blocking_reasons)
        assert any("NO BASELINE VALUE" in r for r in decision.blocking_reasons)


# gate.py finding 2: per_intersection_thresholds must match the intersection


class TestPerIntersectionThresholds:
    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 400
        y_true = rng.integers(0, 2, n)
        gender = np.where(rng.random(n) < 0.5, "F", "M")
        race = np.where(rng.random(n) < 0.5, "B", "W")
        y_pred = (
            ((gender == "M") & (rng.random(n) < 0.8)) | ((gender == "F") & (rng.random(n) < 0.4))
        ).astype(int)
        return y_true, y_pred, gender, race

    def _thresholds_seen(self, per_intersection):
        y_true, y_pred, gender, race = self._data()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        hc = HierarchicalGateConfig(
            check_overall=False,
            check_single_attributes=False,
            per_intersection_thresholds=per_intersection,
        )
        dec = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"gender": gender, "race": race},
            hierarchical_config=hc,
        )
        lvl = dec.level_results["intersection:gender_x_race"]
        return [e.threshold for e in lvl.metric_evaluations]

    def test_unrelated_key_does_not_leak(self):
        thresholds = self._thresholds_seen(
            {"some_other_intersection": {"demographic_parity_difference": 99.0}}
        )
        # Default = base 0.1 * 1.2 multiplier, not the unrelated override.
        assert thresholds == [pytest.approx(0.12)]

    def test_matching_group_key_applies(self):
        thresholds = self._thresholds_seen({"F_B": {"demographic_parity_difference": 0.5}})
        assert thresholds == [pytest.approx(0.5)]

    def test_matching_intersection_name_applies(self):
        thresholds = self._thresholds_seen(
            {"gender_x_race": {"demographic_parity_difference": 0.4}}
        )
        assert thresholds == [pytest.approx(0.4)]


# validator.py finding 3: class-imbalance check only for binary outcomes


class TestClassImbalanceBinaryOnly:
    def test_continuous_outcome_skipped_with_note(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "gender": ["F", "M"] * 100,
                "income": rng.normal(50000, 10000, 200),
            }
        )
        v = DataBiasValidator(protected_attributes=["gender"])
        issues, metrics = v._check_label_quality(df, "income")
        assert "positive_rate" not in metrics
        assert "skipped" in metrics.get("class_imbalance_check", "")
        assert not any(i.issue_type == "extreme_class_imbalance" for i in issues)

    def test_multiclass_int_outcome_skipped(self):
        df = pd.DataFrame(
            {
                "gender": ["F", "M"] * 30,
                "label": [0, 1, 2] * 20,
            }
        )
        v = DataBiasValidator(protected_attributes=["gender"])
        issues, metrics = v._check_label_quality(df, "label")
        assert "positive_rate" not in metrics
        assert "class_imbalance_check" in metrics

    def test_binary_outcome_still_checked(self):
        df = pd.DataFrame(
            {
                "gender": ["F", "M"] * 100,
                "label": [1] * 199 + [0],
            }
        )
        v = DataBiasValidator(protected_attributes=["gender"])
        issues, metrics = v._check_label_quality(df, "label")
        assert metrics["positive_rate"] == pytest.approx(0.995)
        assert any(i.issue_type == "extreme_class_imbalance" for i in issues)

    def test_string_binary_outcome_still_checked(self):
        df = pd.DataFrame(
            {
                "gender": ["F", "M"] * 100,
                "label": ["Yes", "No"] * 100,
            }
        )
        v = DataBiasValidator(protected_attributes=["gender"])
        issues, metrics = v._check_label_quality(df, "label")
        assert metrics["positive_rate"] == pytest.approx(0.5)


# drift.py findings 4 + 5: split at len(reference); MMD baseline vs current


class TestDriftWindows:
    @staticmethod
    def _series():
        rng = np.random.default_rng(2)
        base = pd.Series(rng.normal(0.1, 0.01, 80))
        cur = pd.Series(rng.normal(0.3, 0.01, 20))
        return base, cur

    def test_check_drift_splits_at_baseline_length(self, monkeypatch):
        # Pin the raw-signal path: with PyWavelets installed the series is
        # DWT-decomposed and the approximation scale smooths across the
        # split boundary, which is expected multiscale behaviour but not
        # what this test measures (the split index itself).
        from vfairness.operations.monitoring import drift as drift_mod

        monkeypatch.setattr(drift_mod, "_HAS_PYWT", False)
        base, cur = self._series()
        det = FairnessDriftDetector(compute_mmd=False)
        det.set_baseline(base)
        res = det.check_drift(cur)
        fs = (
            res.scales["full_signal"]
            if "full_signal" in res.scales
            else (list(res.scales.values())[0])
        )
        assert fs.reference_n == 80
        assert fs.current_n == 20
        assert fs.reference_mean == pytest.approx(0.1, abs=0.01)
        assert fs.current_mean == pytest.approx(0.3, abs=0.01)
        # A clean 0.1 -> 0.3 shift must be detected on the raw signal.
        assert fs.drift_detected

    def test_mmd_compares_baseline_vs_current_window(self):
        base, cur = self._series()
        det = FairnessDriftDetector(compute_mmd=True)
        det.set_baseline(base)
        res = det.check_drift(cur)
        expected = mmd_gaussian(base.values.astype(float), cur.values.astype(float))
        assert res.mmd_score == pytest.approx(expected, rel=1e-9)

    def test_detect_drift_ks_midpoint_default_unchanged(self):
        rng = np.random.default_rng(3)
        series = pd.Series(np.concatenate([rng.normal(0.0, 0.01, 30), rng.normal(1.0, 0.01, 30)]))
        det = FairnessDriftDetector()
        r = det.detect_drift_ks(series)
        assert r.reference_n == 30 and r.current_n == 30


# tracker.py finding 6: degradation = significantly HIGHER disparity


class TestWeeklyDegradationDirection:
    @staticmethod
    def _analyzer(monday_val, other_val):
        an = TemporalFairnessAnalyzer(lookback_days=60)
        start = pd.Timestamp("2025-01-06")  # a Monday
        for i in range(28):
            d = start + pd.Timedelta(days=i)
            val = monday_val if d.dayofweek == 0 else other_val
            an.update_daily_metrics(d, {"demographic_parity": val})
        return an

    def test_higher_disparity_weekday_flags(self):
        an = self._analyzer(monday_val=0.20, other_val=0.05)
        degraded, worst = an.detect_weekly_degradation("demographic_parity")
        assert degraded
        # Worst day is the HIGH-disparity Monday, not the good days.
        assert worst == pytest.approx(0.20)

    def test_lower_disparity_weekday_does_not_flag(self):
        # Mondays are slightly BETTER (lower disparity): must not fire by
        # default. Overall mean = 0.04929; the worst weekday (0.05) sits
        # only ~1.4% above it, under the 5% threshold.
        an = self._analyzer(monday_val=0.045, other_val=0.05)
        degraded, worst = an.detect_weekly_degradation("demographic_parity")
        assert not degraded

    def test_higher_is_better_flag_flips_direction(self):
        # Accuracy-like metric dropping on Mondays should flag with the flag.
        an = self._analyzer(monday_val=0.70, other_val=0.95)
        degraded, worst = an.detect_weekly_degradation("demographic_parity", higher_is_better=True)
        assert degraded
        assert worst == pytest.approx(0.70)


# compliance.py finding 7: at most 4 reasons, unique codes


class TestAdverseActionReasons:
    def test_cap_at_four_and_unique_codes(self):
        shap = {
            "income": -0.3,
            "zip_code": -0.25,
            "credit_score": -0.2,
            "age_proxy": -0.15,
            "employment": -0.1,
            "tenure": -0.05,
        }
        reasons = compute_adverse_action_reasons(
            shap, list(shap.keys()), proxy_features=["zip_code", "age_proxy"]
        )
        assert len(reasons) <= 4
        codes = [r["code"] for r in reasons]
        assert len(codes) == len(set(codes))
        # Non-proxy reasons take priority and fill the notice.
        assert [r["feature"] for r in reasons] == ["income", "credit_score", "employment", "tenure"]

    def test_proxy_fillers_capped_and_unique(self):
        # Only one non-proxy adverse feature: proxies may fill remaining
        # transparency slots but the total stays at 4 with unique codes.
        shap = {"income": -0.3, "p1": -0.25, "p2": -0.2, "p3": -0.15, "p4": -0.1, "p5": -0.05}
        reasons = compute_adverse_action_reasons(
            shap, list(shap.keys()), proxy_features=["p1", "p2", "p3", "p4", "p5"]
        )
        assert len(reasons) <= 4
        codes = [r["code"] for r in reasons]
        assert len(codes) == len(set(codes))
        assert reasons[0]["feature"] == "income"
        assert reasons[0]["is_proxy_excluded"] is False
        for r in reasons[1:]:
            assert r["is_proxy_excluded"] is True

    def test_duplicate_custom_codes_deduplicated(self):
        shap = {"a": -0.4, "b": -0.3, "c": -0.2}
        mapping = {"a": ("X01", "Reason A"), "b": ("X01", "Reason B dup"), "c": ("X02", "Reason C")}
        reasons = compute_adverse_action_reasons(
            shap, list(shap.keys()), proxy_features=[], reason_code_mapping=mapping
        )
        codes = [r["code"] for r in reasons]
        assert len(codes) == len(set(codes))
        assert "X01" in codes and "X02" in codes

    # interactive.py finding 8: standalone HTML renders offline

    def test_to_dict_carries_the_disclosure_that_json_dumps_drops(self):
        """The one method whose whole job is to stop a two-state collapse at a
        boundary, and nothing executed it until now.

        AdverseActionReasons is a list subclass, so ``json.dumps`` on it writes the
        bare reason array and every attribute goes with it, ``complete`` included.
        That is the field a consumer reads to decide whether the notice may be issued
        at all, and without it a run that examined NOTHING crosses the boundary as
        ``[]``, identical to a notice that looked and found no adverse factor.
        ``to_dict`` exists to carry both.

        Found by the readiness gate, not by a reviewer: criterion B1 names every
        public unit the suite has never executed, and this method was one of two.
        A serialiser nobody runs is a disclosure nobody has seen work.

        BOTH DIRECTIONS, because either alone proves little: to_dict must carry the
        disclosure, AND the bare json.dumps must still drop it, since that is the
        defect the method exists for and a test that only checks the good path would
        pass even if the collapse were fixed somewhere else and the method rotted.
        """
        examined_nothing = AdverseActionReasons(
            entries=[],
            unattributed_features=["income", "zip_code"],
            features_examined=0,
        )

        payload = examined_nothing.to_dict()
        assert payload["reasons"] == []
        assert payload["complete"] is False, (
            "a run that examined nothing reports complete, so a consumer may issue the notice"
        )
        assert payload["features_examined"] == 0
        assert payload["unattributed_features"] == ["income", "zip_code"]
        assert "adverse_attributions_not_ranked" in payload

        # It is JSON-serialisable, which is the only reason the method exists.
        assert json.loads(json.dumps(payload))["complete"] is False

        # AND THE COLLAPSE IS STILL THERE on the bare object, which is why callers
        # must use to_dict: this asserts the hazard rather than the fix.
        assert json.loads(json.dumps(examined_nothing)) == [], (
            "the bare object no longer serialises to a naked array, so this test is "
            "no longer about the boundary it was written for"
        )

        # The control: a notice that DID examine features and found one carries a
        # complete disclosure, so to_dict is not simply stamping False on everything.
        found_one = AdverseActionReasons(
            entries=[{"code": "RC01", "description": "income"}],
            unattributed_features=[],
            features_examined=4,
        )
        found = found_one.to_dict()
        assert found["complete"] is True, found
        assert found["features_examined"] == 4
        assert len(found["reasons"]) == 1


class TestStandaloneHtmlOffline:
    def test_plotly_bundle_inlined_no_external_script(self):
        pytest.importorskip("plotly")
        from vfairness.operations.reporting.interactive import (
            InteractiveDashboard,
        )

        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 4,
                    "metric": ["demographic_parity"] * 4,
                    "value": [0.05, 0.06, 0.07, 0.08],
                }
            )
        )
        html = InteractiveDashboard(store).to_standalone_html(include_whatif=False)
        # No external script/link tags: the file must render offline.
        assert "<script src=" not in html
        assert "<link " not in html
        # The inlined bundle actually defines Plotly.
        from plotly.offline import get_plotlyjs

        assert len(html) > len(get_plotlyjs())


# store.py finding 9: an ingested alert is counted once, not twice


class TestHealthScoreSinglePenalty:
    class _FakeAlert:
        def to_dict(self):
            return {"timestamp": datetime.now(), "priority_score": 0.9, "severity": "HIGH"}

    def test_ingested_alert_only_hits_alert_frequency(self):
        s = MetricsStore()
        s.ingest_alert(self._FakeAlert())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hs = s.compute_health_score()
        # The alert IS evidence and is still measured: one alert, 90.0.
        assert hs.components["alert_frequency"] == pytest.approx(90.0)

        # metric_compliance is now ABSENT, and the old expectation of 100.0 was
        # the defect. Updated 2026-09-17 (Beta Go-Live Stage 2). With no metric
        # record ever compared to a threshold there is no compliance evidence,
        # and 100.0 is the perfect default for no evidence. Measured on a store
        # whose only content was one unresolved CRITICAL alert, before the fix:
        # "97/100 (GREEN)" over "| Metric Compliance | 100.0 |", with no
        # warning at all. The composite is withheld for the same reason.
        assert "metric_compliance" not in hs.components, (
            f"metric_compliance={hs.components.get('metric_compliance')!r} over zero "
            f"rows that were ever compared to a threshold"
        )
        assert hs.score is None and hs.status == "not_assessed"

    def test_metric_breach_rows_still_penalize_compliance(self):
        s = MetricsStore()
        from vfairness.operations.reporting.store import StoredMetricRecord

        now = datetime.now()
        for alert_flag in (True, False):
            s._records.append(
                StoredMetricRecord(
                    timestamp=now,
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.2,
                    alert=alert_flag,
                )
            )
        hs = s.compute_health_score()
        # 1 of 2 metric rows breached: 100 * (1 - 0.5 * 2) = 0.
        assert hs.components["metric_compliance"] == pytest.approx(0.0)
        # BGL5 A-operations-3, 2026-09-27: this read "No ingested alert payloads,
        # so alert frequency is untouched" and asserted 100.0. Untouched is the
        # problem: nothing measured that component, and 100.0 is its perfect
        # default, worth 30 points of the composite. It is now ABSENT, the same
        # way metric_compliance is absent above when nothing compared a metric.
        # The subject of this test, that a breach row penalises COMPLIANCE, is
        # unchanged and is the assertion above.
        assert "alert_frequency" not in hs.components, hs.components

    def test_health_weights_unchanged(self):
        """The composite is 50% compliance, 30% alert frequency, 20% drift.

        Fixture changed 2026-09-17 (Beta Go-Live Stage 2), subject unchanged.
        It used to ingest ONE alert and nothing else, and assert
        ``0.50 * 100.0 + 0.30 * 90.0 + 0.20 * 100.0 == 97.0``. Two of those
        three terms were the perfect default standing in for no evidence at all:
        not one metric record had ever been compared to a threshold. 97/100
        GREEN over an unresolved CRITICAL alert is the exact number the store's
        own comment now records as the defect it fixed, so asserting it here
        would pin the fabrication rather than the weights.

        The weights are the subject, so the fixture now gives compliance real
        evidence and the assertion is made against the components the store
        actually reports.
        """
        from vfairness.operations.reporting.store import StoredMetricRecord

        s = MetricsStore()
        now = datetime.now()
        for alert_flag in (True, False, False, False):  # 1 of 4 breached
            s._records.append(
                StoredMetricRecord(
                    timestamp=now,
                    source="FairnessMonitor",
                    metric_name="demographic_parity",
                    value=0.2,
                    alert=alert_flag,
                )
            )
        s.ingest_alert(self._FakeAlert())
        # BGL3 operations-4, 2026-09-27. The fixture now supplies a real drift
        # verdict too, for exactly the reason the docstring gives for supplying
        # real compliance evidence: an EMPTY drift table used to answer 100.0,
        # and that perfect default for no evidence is now excluded from the
        # composite, so without this row `drift_stability` is absent and the
        # assertion below pins nothing. One clean verdict is a measured 100.0
        # stability, so every number in this test is unchanged.
        s._drift_records.append(
            {
                "timestamp": now,
                "metric": "demographic_parity",
                "overall_drift_score": 0.01,
                "drift_detected": False,
                "mmd_score": None,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hs = s.compute_health_score()

        # All three components must be present, or this pins nothing.
        for name in ("metric_compliance", "alert_frequency", "drift_stability"):
            assert name in hs.components, f"{name} missing; the fixture pins no weight"

        expected = (
            0.50 * hs.components["metric_compliance"]
            + 0.30 * hs.components["alert_frequency"]
            + 0.20 * hs.components["drift_stability"]
        )
        assert hs.score == pytest.approx(round(expected, 1)), (
            f"score={hs.score} does not match 50/30/20 over {hs.components}"
        )
