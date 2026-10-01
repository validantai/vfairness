"""Beta Go-Live Stage 2, group s2g12: three states at the public entry.

Seven findings across monitoring (alerts, drift, tracker) and group
calibration. Every test here asserts at the PUBLIC entry point named in the
finding, and every one is paired with a CONTROL proving healthy data still
produces the correct measurement: a fix that makes everything refuse is a
worse defect than the one it replaces, and it passes any test that only
exercises the degenerate case.

Findings pinned:

1. ``AdaptiveThresholdManager.is_alert_warranted`` returned a bare ``False``
   for a NaN drift score, byte-identical to a measured calm reading.
2. ``FairnessAlertPrioritizer.calculate_priority`` scored an empty factor set
   as a finite 0.0 and banded it LOW; ``create_alert`` stamped an ABSENT drift
   score into the payload as 0.0, silently, on alerts up to CRITICAL.
3. ``FairnessDriftDetector.detect_drift_mmd`` returned ``(False, nan)``, "no
   drift", where no permutation test ran at all, and ``(True, nan)`` where the
   statistic was not a number.
4. ``rendering.temporal_analysis_to_svg`` discarded a real HIGH weekly
   degradation finding and rendered COULD NOT CHECK, because metric discovery
   probed for a column the analyzer never produces.
5. ``GroupCalibrator.fit`` never cleared fitted state, so a refit kept the
   previous fit's calibrators.
6. ``GroupCalibrator.transform`` treated a stale key as proof the group was
   fitted on THIS data, defeating ``UnknownGroupWarning`` even in strict mode.
7. ``IntersectionalCalibrator`` served a marginal or absent calibration as an
   intersectional one, with no provenance and no warning at either step.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness
from vfairness import (
    AdaptiveThresholdManager,
    FairnessAlertPrioritizer,
    FairnessDriftDetector,
    GroupCalibrator,
    IntersectionalCalibrator,
)
from vfairness.post_processing.calibration.group_calibrator import UnknownGroupWarning
from vfairness.rendering import adapters_monitoring as am

KEY = "demographic_parity"


def _catch(fn, *args, **kwargs):
    """Run *fn*, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# Finding 1: AdaptiveThresholdManager.is_alert_warranted


class TestAlertGateThreeStates:
    def test_unmeasurable_drift_score_is_not_an_all_clear(self):
        """(None, warned), never the False that reads as 'nobody is paged'."""
        mgr = AdaptiveThresholdManager()
        verdict, caught = _catch(mgr.is_alert_warranted, KEY, float("nan"))

        assert verdict is None, f"could-not-check collapsed into {verdict!r}"
        assert verdict is not False
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_nan_from_the_librarys_own_producer_reaches_the_gate_as_none(self):
        """The end-to-end path in the finding: the NaN is not typed in.

        Also asserts the fixture EXERCISES the branch it claims to: the
        detector must really hand back a non-finite score, or this test would
        be pinning nothing.
        """
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(np.random.default_rng(0).normal(0.1, 0.01, 60)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = detector.check_drift(pd.Series([float("nan")] * 60))

        # Fixture reaches the branch.
        assert math.isnan(result.overall_drift_score)
        assert result.drift_detected is None

        mgr = AdaptiveThresholdManager()
        verdict, caught = _catch(mgr.is_alert_warranted, KEY, result.overall_drift_score)
        assert verdict is None
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_none_score_answers_rather_than_raising(self):
        mgr = AdaptiveThresholdManager()
        verdict, caught = _catch(mgr.is_alert_warranted, KEY, None)
        assert verdict is None
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_control_measured_scores_still_grade_both_ways(self):
        """Over-correction control: a real score still gets a real verdict."""
        mgr = AdaptiveThresholdManager()
        assert mgr.get_threshold(KEY) == 0.7

        above, w_above = _catch(mgr.is_alert_warranted, KEY, 0.95)
        below, w_below = _catch(mgr.is_alert_warranted, KEY, 0.10)
        zero, w_zero = _catch(mgr.is_alert_warranted, KEY, 0.0)
        exact, _ = _catch(mgr.is_alert_warranted, KEY, 0.7)

        assert above is True and w_above == []
        assert below is False and w_below == []
        # A MEASURED 0.0 is a real calm reading and must stay False, not None.
        assert zero is False and w_zero == []
        assert exact is True


# Finding 2: FairnessAlertPrioritizer


_FULL_EVENT = {
    "regulatory_risk": 1.0,
    "population_impact": 0.8,
    "drift_velocity": 0.9,
    "historical_discrimination": 1.0,
    "metric_name": "equalized_odds",
    "affected_groups": ["Black", "Female"],
}


class TestAlertPrioritizerThreeStates:
    def test_empty_event_is_unscored_not_low(self):
        p = FairnessAlertPrioritizer()
        (score, severity), caught = _catch(p.calculate_priority, {})

        assert math.isnan(score), f"a sum over an empty factor set returned {score!r}"
        assert severity == "UNSCORED"
        assert severity != "LOW"
        assert any("NOT been graded harmless" in w for w in caught), caught

    def test_empty_event_alert_is_routed_for_hand_triage_not_to_the_backlog(self):
        p = FairnessAlertPrioritizer()
        payload, _ = _catch(p.create_alert, {})

        assert payload.severity == "UNSCORED"
        assert math.isnan(payload.priority_score)
        assert payload.routing["channel"] != "jira"
        assert "triage" in payload.routing
        assert "COULD NOT be computed" in payload.message

    def test_absent_drift_score_is_not_recorded_as_zero(self):
        """The silent half: a CRITICAL alert printing 'Drift score: 0.000'."""
        p = FairnessAlertPrioritizer()
        payload, caught = _catch(p.create_alert, dict(_FULL_EVENT))

        # Fixture reaches the branch it claims to: this is a real CRITICAL
        # alert, graded from four measured factors, not a degenerate one.
        assert payload.severity == "CRITICAL"
        assert payload.priority_score == pytest.approx(10.35, abs=0.01)

        assert math.isnan(payload.drift_score), payload.drift_score
        assert math.isnan(payload.mean_shift)
        assert "0.000" not in payload.message
        assert "NOT MEASURED" in payload.message
        assert math.isnan(payload.to_dict()["drift_score"])
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_control_explicit_zero_is_still_a_real_zero(self):
        """A measured 0.0 drift is a fact and must survive untouched."""
        p = FairnessAlertPrioritizer()
        payload, caught = _catch(p.create_alert, dict(_FULL_EVENT), drift_score=0.0, mean_shift=0.0)
        assert payload.drift_score == 0.0
        assert payload.mean_shift == 0.0
        assert "Drift score: 0.000" in payload.message
        assert caught == []

        # And the event-dict route keeps its measured values too.
        from_event, w_event = _catch(
            p.create_alert, {**_FULL_EVENT, "drift_score": 0.82, "mean_shift": 0.07}
        )
        assert from_event.drift_score == pytest.approx(0.82)
        assert from_event.mean_shift == pytest.approx(0.07)
        assert w_event == []

    def test_control_partially_measured_event_still_scores(self):
        """Only SOME factors missing is still a graded alert, as before."""
        p = FairnessAlertPrioritizer()
        event = {k: v for k, v in _FULL_EVENT.items() if k != "regulatory_risk"}
        (score, severity), caught = _catch(p.calculate_priority, {**event, "drift_score": 0.0})
        # 0.8*2.0 + 0.9*2.5 + 1.0*3.0 = 6.85, regulatory_risk excluded.
        assert score == pytest.approx(6.85, abs=0.01)
        assert severity == "HIGH"
        assert any("regulatory_risk" in w for w in caught)

    def test_control_four_genuine_zero_factors_still_band_low(self):
        """The guard counts EXCLUSIONS, never the value of the sum.

        A drift event whose four factors were all MEASURED at 0.0 is a real
        calm alert: it must keep scoring 0.0 and banding LOW. Pinning this is
        the whole difference between 'no factor was measured' and 'every
        factor was measured and they are all zero'.
        """
        p = FairnessAlertPrioritizer()
        calm = {
            "regulatory_risk": 0.0,
            "population_impact": 0.0,
            "drift_velocity": 0.0,
            "historical_discrimination": 0.0,
            "drift_score": 0.0,
            "mean_shift": 0.0,
        }
        (score, severity), caught = _catch(p.calculate_priority, calm)
        assert score == 0.0
        assert severity == "LOW"
        assert caught == []


# Finding 3: FairnessDriftDetector.detect_drift_mmd


class TestMmdThreeStates:
    @pytest.mark.parametrize(
        "ref,cur",
        [
            (np.array([]), np.array([])),
            (np.array([0.1]), np.array([0.9])),
            (np.array([]), np.random.default_rng(0).normal(5, 0.1, 60)),
            (np.array([0.1]), np.random.default_rng(0).normal(5, 0.1, 60)),
        ],
        ids=["empty-empty", "1-1", "0-60", "1-60-huge-shift"],
    )
    def test_no_test_ran_means_no_claim(self, ref, cur):
        detector = FairnessDriftDetector()
        (drift, score), caught = _catch(detector.detect_drift_mmd, ref, cur)

        assert drift is None, f"a comparison that never ran claimed {drift!r}"
        assert drift is not False
        assert math.isnan(score)
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_an_all_nan_window_has_no_finite_samples_to_compare(self):
        detector = FairnessDriftDetector()
        ref = np.random.default_rng(1).normal(0, 1, 30)
        (drift, score), caught = _catch(detector.detect_drift_mmd, ref, np.full(30, np.nan))

        assert drift is None
        assert math.isnan(score)
        assert any("COULD NOT CHECK" in w for w in caught), caught

    @pytest.mark.parametrize("threshold", [None, 0.05], ids=["permutation", "fixed"])
    def test_a_non_number_statistic_is_not_a_verdict(self, threshold):
        """The sibling defect, and why the guard sits ABOVE the branch choice.

        Both arms compare with operators that are False for NaN, so ONE input
        produced two opposite confident answers. Measured on this fixture with
        the guard removed: the permutation arm returned (True, nan), because
        every `nan >= nan` is False so `exceed` stayed 0; the fixed-threshold
        arm returned (False, nan), because `nan > 0.05` is False. Fixing one
        arm would simply have moved the defect to the other.

        The inputs here are ALL FINITE: the squared distances overflow to inf,
        which drives the median-heuristic sigma to inf, gamma to 0.0, and
        `exp(-0.0 * inf)` to nan. The first guard (fewer than 2 finite samples)
        cannot fire on 4 vs 4 finite readings, so this fixture really does
        reach the statistic guard and not the sample-count one.
        """
        detector = FairnessDriftDetector()
        ref = np.array([1e200, -1e200, 1e200, -1e200])
        cur = np.array([3e200, -3e200, 2e200, -2e200])
        assert np.isfinite(ref).all() and np.isfinite(cur).all()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            np.seterr(over="ignore", invalid="ignore")
            drift, score = detector.detect_drift_mmd(ref, cur, threshold=threshold)
            np.seterr(over="warn", invalid="warn")
        messages = [str(w.message) for w in caught]

        assert drift is None, f"a non-number statistic graded as {drift!r}"
        assert drift is not True and drift is not False
        assert math.isnan(score)
        assert any("no verdict can be derived" in m for m in messages), messages

    def test_control_healthy_samples_still_detect_a_real_shift(self):
        detector = FairnessDriftDetector()
        ref = np.random.default_rng(1).normal(0, 1, 30)
        cur = np.random.default_rng(2).normal(3, 1, 30)
        (drift, score), caught = _catch(detector.detect_drift_mmd, ref, cur)

        assert drift is True
        assert score == pytest.approx(0.9556, abs=1e-3)
        assert caught == [], caught

    def test_control_healthy_identical_samples_still_report_no_drift(self):
        """False must remain reachable: this is the measured 'no drift'."""
        rng = np.random.default_rng(4)
        detector = FairnessDriftDetector()
        (drift, score), caught = _catch(
            detector.detect_drift_mmd, rng.normal(0, 1, 40), rng.normal(0, 1, 40)
        )
        assert drift is False
        assert math.isfinite(score)
        assert caught == []

    def test_control_contaminated_sample_still_measures_what_is_there(self):
        """One NaN among 30 does not make the other 29 unreadable."""
        detector = FairnessDriftDetector()
        ref = np.random.default_rng(1).normal(0, 1, 30)
        cur = np.random.default_rng(2).normal(3, 1, 30)
        cur[0] = np.nan
        (drift, score), caught = _catch(detector.detect_drift_mmd, ref, cur)

        assert drift is True
        assert math.isfinite(score)
        assert any("non-finite" in w for w in caught), caught

    def test_control_legacy_fixed_threshold_arm_still_grades(self):
        rng = np.random.default_rng(0)
        detector = FairnessDriftDetector()
        drift, score = detector.detect_drift_mmd(
            rng.uniform(0, 100, 20), rng.uniform(0, 100, 20), sigma=1.0, threshold=0.05
        )
        assert drift is False
        assert abs(score) < 0.05


# Finding 4: temporal_analysis_to_svg discovers nothing


def _analyzer_with_60_days():
    analyzer = vfairness.TemporalFairnessAnalyzer()
    base = pd.Timestamp("2026-01-01")
    for day in range(60):
        date = base + pd.Timedelta(days=day)
        analyzer.update_daily_metrics(
            date,
            {
                "demographic_parity_difference": 0.05
                + 0.002 * day
                + (0.25 if date.dayofweek == 4 else 0.0)
            },
        )
    return analyzer


class TestTemporalRenderFindsItsMetrics:
    def test_a_real_finding_is_not_discarded_into_could_not_check(self):
        analyzer = _analyzer_with_60_days()

        # Fixture reaches the branch: the analyzer really does hold a HIGH
        # finding, so a COULD NOT CHECK here is a discarded measurement and
        # not an honest refusal.
        assert analyzer.get_tracked_metrics() == ["demographic_parity_difference"]
        assert analyzer.detect_trend("demographic_parity_difference")[0] == "increasing"
        degraded, _ = analyzer.detect_weekly_degradation("demographic_parity_difference")
        assert degraded is True
        assert "metric_name" not in analyzer.to_dataframe().columns

        svg = am.temporal_analysis_to_svg(analyzer)

        assert "NOT CHECKED" not in svg
        assert "No metric history was available" not in svg
        assert "severity: HIGH" in svg
        assert "weekly degradation" in svg

    def test_one_argument_call_matches_the_explicit_one(self):
        """The documented one-argument call must not be the weaker reading."""
        analyzer = _analyzer_with_60_days()
        implicit = am.temporal_analysis_to_svg(analyzer)
        explicit = am.temporal_analysis_to_svg(analyzer, ["demographic_parity_difference"])
        assert ("severity: HIGH" in implicit) == ("severity: HIGH" in explicit)

    def test_control_an_empty_analyzer_still_refuses(self):
        """The chart-level guard was CORRECT and must survive: an analyzer
        with no history still says it checked nothing."""
        svg = am.temporal_analysis_to_svg(vfairness.TemporalFairnessAnalyzer())
        assert "NOT CHECKED" in svg
        assert "COULD NOT CHECK" in svg
        assert "severity: HIGH" not in svg


# Findings 5 and 6: GroupCalibrator refit staleness


def _two_group_data():
    rng = np.random.default_rng(3)
    n = 200
    prob = rng.random(2 * n)
    groups = np.array(["a"] * n + ["b"] * n)
    y = (rng.random(2 * n) < np.clip(prob * 0.8 + 0.1, 0, 1)).astype(int)
    return y, prob, groups


def _a_only_data():
    rng = np.random.default_rng(3)
    n = 200
    rng.random(2 * n)  # keep the stream aligned with the two-group fixture
    prob = rng.random(n)
    y = (rng.random(n) < np.clip(prob * 0.9, 0, 1)).astype(int)
    return y, prob, np.array(["a"] * n)


class TestGroupCalibratorRefit:
    def test_refit_forgets_the_previous_fits_groups(self):
        calib = GroupCalibrator(method="platt", min_group_size=10)
        calib.fit(*_two_group_data())
        assert "b" in calib.calibrators_, "fixture never fitted 'b'"

        calib.fit(*_a_only_data())

        assert "b" not in calib.calibrators_
        assert "b" not in calib.fit_result_.group_names
        assert calib.fit_result_.n_groups == len(calib.fit_result_.group_names)
        serialized = calib.fit_result_.to_dict()
        assert set(serialized["group_names"]) == set(serialized["pre_calibration_ece"])

    def test_a_group_the_last_fit_never_saw_is_not_group_calibrated(self):
        """The damage: strict mode was handed silently wrong numbers."""
        calib = GroupCalibrator(method="platt", min_group_size=10)
        calib.fit(*_two_group_data())
        calib.fit(*_a_only_data())

        probe = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        with pytest.warns(UnknownGroupWarning, match="not seen during fit"):
            stale = calib.transform(probe, np.array(["b"] * 5))

        # A caller who opted into failing closed on exactly this condition
        # must actually fail closed.
        with pytest.raises(UnknownGroupWarning):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UnknownGroupWarning)
                calib.transform(probe, np.array(["b"] * 5))

        # And the numbers now match a calibrator freshly fitted on the same
        # data, rather than one fitted on a dataset that no longer exists.
        fresh = GroupCalibrator(method="platt", min_group_size=10)
        fresh.fit(*_a_only_data())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnknownGroupWarning)
            expected = fresh.transform(probe, np.array(["b"] * 5))
        assert np.max(np.abs(stale - expected)) == pytest.approx(0.0, abs=1e-12)

    def test_control_a_group_the_last_fit_did_see_is_still_group_calibrated(self):
        """Over-correction control: the reset must not make everything unknown."""
        calib = GroupCalibrator(method="platt", min_group_size=10)
        calib.fit(*_two_group_data())
        calib.fit(*_a_only_data())

        probe = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        out, caught = _catch(calib.transform, probe, np.array(["a"] * 5))
        assert caught == [], caught
        assert calib.get_group_calibrator("a") is not None
        assert np.all(np.isfinite(out))

        group_out = calib.calibrators_["a"].transform(probe)
        assert np.allclose(out, np.clip(group_out, 0, 1))

    def test_control_a_single_fit_is_unchanged(self):
        y, prob, groups = _two_group_data()
        calib = GroupCalibrator(method="platt", min_group_size=10)
        calib.fit(y, prob, groups)
        assert sorted(calib.calibrators_) == ["a", "b"]
        assert calib.fit_result_.n_groups == 2
        assert sorted(calib.fit_result_.group_names) == ["a", "b"]
        assert calib.is_fitted is True


# Finding 7: IntersectionalCalibrator provenance


def _intersectional_frame(f_b_rows: int):
    rng = np.random.default_rng(7)
    parts = [
        pd.DataFrame({"gender": g, "race": a, "p": rng.random(n), "y": rng.integers(0, 2, n)})
        for g, a, n in [("M", "A", 300), ("M", "B", 300), ("F", "A", 300), ("F", "B", f_b_rows)]
    ]
    return pd.concat(parts, ignore_index=True)


class TestIntersectionalProvenance:
    def test_a_borrowed_marginal_curve_is_not_an_intersectional_result(self):
        df = _intersectional_frame(f_b_rows=1)
        calib = IntersectionalCalibrator(min_group_size=30)

        with pytest.warns(UserWarning, match="NOT intersectionally calibrated"):
            calib.fit(df["y"].values, df["p"].values, df[["gender", "race"]])

        # Fixture reaches the hierarchical-borrowing branch, not some other one.
        assert calib.calibrators_["F_B"] is calib.parent_calibrators_["gender"]["F"]

        provenance = calib.get_group_provenance()
        assert provenance["F_B"] == "borrowed_marginal"
        assert calib.fallback_groups_ == ["F_B"]

        with pytest.warns(UserWarning, match="NOT intersectionally calibrated"):
            calib.transform(df["p"].values, df[["gender", "race"]])

    def test_an_uncalibrated_group_says_so(self):
        df = _intersectional_frame(f_b_rows=1)
        calib = IntersectionalCalibrator(min_group_size=30, borrowing_strategy="none")
        with pytest.warns(UserWarning, match="uncalibrated"):
            calib.fit(df["y"].values, df["p"].values, df[["gender", "race"]])

        assert calib.get_group_provenance()["F_B"] == "uncalibrated"

        with pytest.warns(UserWarning, match="NOT intersectionally calibrated"):
            out = calib.transform(df["p"].values, df[["gender", "race"]])

        mask = ((df.gender == "F") & (df.race == "B")).values
        # Still byte-identical to its input, which is the point: the array
        # cannot show it, so the warning and the provenance must.
        assert np.array_equal(out[mask], df["p"].values[mask])

    def test_a_strategy_that_would_not_run_is_refused(self):
        with pytest.raises(ValueError, match="Unknown borrowing_strategy"):
            IntersectionalCalibrator(borrowing_strategy="borrow")

    def test_refit_forgets_the_previous_fits_intersections(self):
        calib = IntersectionalCalibrator(min_group_size=30)
        df = _intersectional_frame(f_b_rows=300)
        calib.fit(df["y"].values, df["p"].values, df[["gender", "race"]])
        assert "F_B" in calib.calibrators_, "fixture never fitted F_B"

        m_only = df[df.gender == "M"]
        calib.fit(m_only["y"].values, m_only["p"].values, m_only[["gender", "race"]])

        assert "F_A" not in calib.calibrators_
        assert "F_B" not in calib.calibrators_
        assert set(calib.get_group_provenance()) == {"M_A", "M_B"}

    def test_control_healthy_intersections_are_measured_and_silent(self):
        df = _intersectional_frame(f_b_rows=300)
        calib = IntersectionalCalibrator(min_group_size=30)

        _, w_fit = _catch(calib.fit, df["y"].values, df["p"].values, df[["gender", "race"]])
        out, w_transform = _catch(calib.transform, df["p"].values, df[["gender", "race"]])

        assert w_fit == [], w_fit
        assert w_transform == [], w_transform
        assert set(calib.get_group_provenance().values()) == {"intersectional"}
        assert calib.fallback_groups_ == []
        # A real calibration actually changed the numbers.
        assert not np.array_equal(out, df["p"].values)
