"""Audit wave 6 pins: drift-streaming unit.

Each class pins one confirmed finding from
docs/audits/second-iteration-audit-2026-08-22.md, negative case first (the
exact scenario that used to produce the wrong verdict must now produce the
right one), then a does-not-overcorrect case (genuine drift is still caught).

Findings covered:
- Wavelet multiscale path flagged ~75% of stationary noise as drift (analytic
  KS p-values on autocorrelated reconstructions); now permutation-calibrated.
- page_hinkley raw-unit defaults flagged ~100% of stationary series and
  sequential_fairness_drift called noise "gradual_drift"; PH now standardizes
  by the series std with sigma-unit defaults (delta 0.005 -> 0.05 sigma,
  lambda_ 0.05 -> 20 sigma).
- TemporalTracker.detect_drift_cusum accumulated raw-unit deviations against
  a 0.5 slack, so the defaults could never flag any disparity drift in [0, 1];
  now standardized (k, h in sigma units).
- detect_drift_mmd used the biased V-statistic (diagonal floor ~2/n), a fixed
  sigma = 1.0 and a fixed 0.05 threshold, declaring drift on identical wide
  distributions and missing fairness-scale shifts up to ~0.2; now unbiased
  MMD^2 + median-heuristic bandwidth + permutation-calibrated threshold
  (Gretton et al. 2012).
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.agents.temporal import TemporalTracker
from vfairness.operations.monitoring import FairnessDriftDetector
from vfairness.operations.monitoring.sequential import (
    page_hinkley,
    sequential_fairness_drift,
)

# ------------------------------------------------------------------
# Finding 1 (CRITICAL): wavelet multiscale false alarms on stationary noise
# ------------------------------------------------------------------


class TestWaveletMultiscaleCalibration:
    @staticmethod
    def _detector():
        pytest.importorskip("pywt")
        return FairnessDriftDetector(compute_mmd=False)

    def test_stationary_noise_false_alarm_rate_near_documented_level(self):
        # Negative case. Pre-fix: 149/200 stationary iid series flagged
        # (74.5%), the approximation scale alone accounting for 141. The
        # documented level is significance_level = 0.05.
        false_alarms = 0
        wavelet_path_seen = False
        for seed in range(40):
            rng = np.random.default_rng(seed)
            det = self._detector()
            det.set_baseline(pd.Series(rng.normal(0.1, 0.02, 60)))
            res = det.check_drift(pd.Series(rng.normal(0.1, 0.02, 60)))
            wavelet_path_seen = wavelet_path_seen or res.decomposition_available
            if res.drift_detected:
                false_alarms += 1
        # Liveness guard: the wavelet path must actually have been exercised,
        # otherwise this test would pass vacuously through the raw path.
        assert wavelet_path_seen
        # Nominal 5% of 40 = 2; allow slack for the score gate interplay.
        assert false_alarms <= 5, f"{false_alarms}/40 stationary series flagged"

    def test_stationary_p_values_are_calibrated_not_collapsed(self):
        # Pre-fix the approximation scale's analytic p collapsed to ~0 on
        # stationary noise. The calibrated (permutation) p must be large.
        rng = np.random.default_rng(0)
        det = self._detector()
        det.set_baseline(pd.Series(rng.normal(0.1, 0.02, 60)))
        res = det.check_drift(pd.Series(rng.normal(0.1, 0.02, 60)))
        approx = res.scales["approximation"]
        assert not approx.drift_detected
        assert approx.p_value > 0.05
        # The raw signal joins the family as its own scale.
        assert "full_signal" in res.scales

    def test_injected_shift_still_detected(self):
        # Does-not-overcorrect: the docstring's 0.10 -> 0.18 shift must be
        # detected (pre-fix and post-fix both 100/100; the fix must not
        # trade the false alarms for misses).
        detected = 0
        for seed in range(10):
            rng = np.random.default_rng(1000 + seed)
            det = self._detector()
            det.set_baseline(pd.Series(rng.normal(0.10, 0.02, 60)))
            res = det.check_drift(pd.Series(rng.normal(0.18, 0.02, 60)))
            detected += int(res.drift_detected)
        assert detected == 10

    def test_worst_scale_ignores_nan_scores(self):
        rng = np.random.default_rng(1)
        det = self._detector()
        det.set_baseline(pd.Series(rng.normal(0.10, 0.02, 60)))
        res = det.check_drift(pd.Series(rng.normal(0.18, 0.02, 60)))
        worst = res.worst_scale
        assert worst is not None
        assert not np.isnan(worst.drift_score)


# ------------------------------------------------------------------
# Finding 2 (CRITICAL): page_hinkley raw-unit defaults; sequential labels
# ------------------------------------------------------------------


class TestPageHinkleySigmaUnits:
    def test_stationary_noise_no_longer_flagged(self):
        # Negative case. Pre-fix false-alarm rates on stationary N(0.1, s)
        # series of length 100: 97.3-100% for sigma >= 0.02.
        for sigma in (0.02, 0.05, 0.1):
            false_alarms = 0
            for seed in range(200):
                rng = np.random.default_rng(seed)
                if page_hinkley(rng.normal(0.1, sigma, 100))["has_drift"]:
                    false_alarms += 1
            assert false_alarms <= 10, f"sigma={sigma}: {false_alarms}/200 flagged"

    def test_scale_invariance(self):
        # Standardization makes the verdict independent of the metric scale.
        rng = np.random.default_rng(42)
        base = np.concatenate([rng.normal(0.10, 0.02, 50), rng.normal(0.18, 0.02, 50)])
        r1 = page_hinkley(base)
        r2 = page_hinkley(base * 10.0)
        assert r1["has_drift"] == r2["has_drift"]
        assert r1["drift_index"] == r2["drift_index"]

    def test_step_still_detected(self):
        # Does-not-overcorrect: a 4-sigma step at t=50 is detected in 100/100
        # seeded runs with the change point localized shortly after t=50.
        detected, indices = 0, []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            x = np.concatenate([rng.normal(0.10, 0.02, 50), rng.normal(0.18, 0.02, 50)])
            r = page_hinkley(x)
            if r["has_drift"]:
                detected += 1
                indices.append(r["drift_index"])
        assert detected >= 95
        assert 50 <= int(np.median(indices)) <= 75

    def test_flat_series_reports_no_drift(self):
        r = page_hinkley(np.full(50, 0.25))
        assert r["has_drift"] is False
        assert r["ph_statistic"] == [0.0] * 50


class TestSequentialFairnessDriftClassification:
    def test_stationary_noise_is_mostly_stable(self):
        # Negative case. Pre-fix: 193-197 of 200 stationary series were
        # labelled gradual_drift/abrupt_drift and "stable" was almost
        # unreachable. Post-fix the residual alarms come from CUSUM's
        # documented ~10% ARL false-alarm level at k=0.5, h=5 over 100 points.
        from collections import Counter

        for sigma in (0.02, 0.05):
            counts = Counter()
            for seed in range(200):
                rng = np.random.default_rng(seed)
                counts[
                    sequential_fairness_drift(rng.normal(0.1, sigma, 100))["classification"]
                ] += 1
            assert counts["stable"] >= 160, f"sigma={sigma}: {dict(counts)}"

    def test_step_still_classified_as_drift(self):
        # Does-not-overcorrect: a genuine 4-sigma step is never "stable"
        # (50/50 abrupt_drift in the seeded validation run).
        for seed in range(50):
            rng = np.random.default_rng(seed)
            x = np.concatenate([rng.normal(0.10, 0.02, 50), rng.normal(0.18, 0.02, 50)])
            assert sequential_fairness_drift(x)["classification"] == "abrupt_drift"


# ------------------------------------------------------------------
# Finding 3 (HIGH): TemporalTracker.detect_drift_cusum default blindness
# ------------------------------------------------------------------


class TestTemporalCusumDefaults:
    @staticmethod
    def _tracker_with_step(step: float, noise_seed: int = 7):
        rng = np.random.default_rng(noise_seed)
        tracker = TemporalTracker()
        for t in range(40):
            level = 0.0 if t < 20 else step
            tracker.record_turn(t, [level + rng.normal(0, 0.01)], [0.0])
        return tracker

    def test_default_params_detect_disparity_step(self):
        # Negative case. Pre-fix a 0.0 -> 0.4 disparity step over 40 turns
        # yielded max_cusum 0.00 and has_drift False at the defaults; even a
        # full-range 0.0 -> 1.0 step reached only 0.01.
        for step in (0.4, 1.0):
            result = self._tracker_with_step(step).detect_drift_cusum()
            assert result["has_drift"] is True, f"step 0->{step} missed at defaults"
            # Detection only: WHERE the change point lands is governed by the
            # separate pooled-target localization finding (the CUSUM anchors
            # on the whole-series mean, so the pre-change arm can cross
            # first), which is recorded and fixed independently of this pin.
            assert result["drift_point"] is not None

    def test_stationary_trajectory_stays_quiet(self):
        # Does-not-overcorrect: stationary disparity trajectories stay quiet
        # at the defaults (0/20 flagged in the seeded validation run).
        false_alarms = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            tracker = TemporalTracker()
            for t in range(40):
                tracker.record_turn(t, [0.1 + rng.normal(0, 0.01)], [0.0])
            false_alarms += int(tracker.detect_drift_cusum()["has_drift"])
        assert false_alarms <= 2

    def test_turn_numbers_still_reported_at_defaults(self):
        # The wave-5 pin (turn numbers, not indices) must hold at defaults too.
        tracker = TemporalTracker()
        # Enough turns for the canonical k=0.5/h=5 CUSUM to accumulate to the
        # decision interval (each standardized step contributes ~0.5 sigma).
        turns = list(range(10, 250, 10))
        for i, turn in enumerate(turns):
            tracker.record_turn(turn, [0.0 if i < len(turns) // 2 else 0.5], [0.0])
        result = tracker.detect_drift_cusum()
        assert result["has_drift"] is True
        assert result["drift_point"] in turns

    def test_flat_trajectory_reports_no_drift(self):
        tracker = TemporalTracker()
        for t in range(10):
            tracker.record_turn(t, [0.3], [0.3])
        result = tracker.detect_drift_cusum()
        assert result["has_drift"] is False
        assert result["max_cusum"] == 0.0


# ------------------------------------------------------------------
# Finding 4 (HIGH): detect_drift_mmd wrong in both directions at defaults
# ------------------------------------------------------------------


class TestMmdCalibratedDefaults:
    detector = FairnessDriftDetector()

    def test_identical_wide_uniform_no_longer_flagged(self):
        # Negative case A. Pre-fix: two n=20 samples from the SAME U(0, 100)
        # declared drift in 5/5 seeds (diagonal bias floor 2/n = 0.10 > 0.05).
        false_alarms = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            drift, _ = self.detector.detect_drift_mmd(
                rng.uniform(0, 100, 20), rng.uniform(0, 100, 20)
            )
            false_alarms += int(drift)
        assert false_alarms <= 4  # nominal alpha = 0.05

    def test_identical_normal_no_longer_flagged(self):
        # Negative case B. Pre-fix: identical N(50, 10) n=30 fired in 2/3 seeds.
        false_alarms = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            drift, _ = self.detector.detect_drift_mmd(
                rng.normal(50, 10, 30), rng.normal(50, 10, 30)
            )
            false_alarms += int(drift)
        assert false_alarms <= 4

    def test_fairness_scale_shift_now_detected(self):
        # Negative case C (the miss direction). Pre-fix a 0.20 mean shift on
        # the fairness-metric scale stayed under the fixed threshold
        # (kernel saturated at sigma = 1.0); shifts 0.05-0.15 likewise.
        rng = np.random.default_rng(0)
        ref = rng.normal(0.1, 0.02, 50)
        for shift in (0.10, 0.20):
            drift, score = self.detector.detect_drift_mmd(ref, rng.normal(0.1 + shift, 0.02, 50))
            assert drift is True, f"shift {shift} missed"

    def test_large_scale_shift_still_detected(self):
        # Does-not-overcorrect: a 1.2-sigma shift on a wide scale is caught
        # (20/20 in the seeded validation run).
        hits = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            drift, _ = self.detector.detect_drift_mmd(
                rng.normal(50, 10, 30), rng.normal(62, 10, 30)
            )
            hits += int(drift)
        assert hits >= 18

    def test_legacy_fixed_threshold_has_no_diagonal_floor(self):
        # The explicit sigma/threshold escape hatch still exists, but on the
        # unbiased statistic: identical far-spread samples no longer sit on
        # the 2/n diagonal floor that forced drift=True pre-fix.
        rng = np.random.default_rng(0)
        drift, score = self.detector.detect_drift_mmd(
            rng.uniform(0, 100, 20), rng.uniform(0, 100, 20), sigma=1.0, threshold=0.05
        )
        assert drift is False
        assert abs(score) < 0.05

    def test_too_few_samples_returns_no_claim(self):
        # BGL S2, 2026-09-16. This test's NAME is "returns no claim" and its
        # assertion was `drift is False`, which is a claim: "no drift". The
        # bare False was the defect, not the contract, so the assertion now
        # says what the name always meant. See detect_drift_mmd's Notes.
        with pytest.warns(UserWarning, match="COULD NOT CHECK"):
            drift, score = self.detector.detect_drift_mmd(np.array([0.1]), np.array([0.2, 0.3]))
        assert drift is None
        assert np.isnan(score)
