"""Wave-6 audit pins for the calibration unit (second-iteration audit 2026-08-22).

Findings pinned here, NEGATIVE case first for each:
  F1  BetaCalibrator dropped Kull et al. 2017's a,b >= 0 constraint and could
      emit a rank-inverting calibration map with no warning.
  F2  calibration_slope.to_dict() reported an inverted is_well_calibrated
      verdict (ECE fallback threshold applied to a slope whose ideal is 1.0).
  F3  sufficiency_test returned passes=True with zero evidence (all bins
      skipped): could-not-check collapsed into PASS.
  F4  sufficiency_test verdict had ~30% false-FAIL on perfectly sufficient
      data (min over ~10 uncorrected chi2 tests vs single-test alpha).
  F5  integrated_calibration_index reported ici_disparity=0.0 (and a [0,0]
      bootstrap CI) when fewer than two groups were assessable.
  F6  IsotonicCalibrator did not reproduce the isotonic (PAVA) solution.
  F7  multicalibration weighted_mean was deflated by unevaluated support and
      group_max read 0.0 for groups with zero evaluated cells.
  F8  ece_confidence_intervals near-null bias: results now carry noise_floor
      and near_null so 'lower bound > 0' cannot be read as miscalibration.
"""

import warnings

import numpy as np
import pytest

from vfairness.post_processing.calibration.methods import (
    BetaCalibrator,
    IsotonicCalibrator,
)
from vfairness.post_processing.calibration.metrics import (
    CalibrationMetricResult,
    calibration_slope,
    ece_confidence_intervals,
    integrated_calibration_index,
    integrated_calibration_index_with_ci,
    multicalibration,
    sufficiency_test,
)

GRID = np.linspace(0.005, 0.995, 199)


def _is_monotone_nondecreasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) >= -1e-12))


# ── F1: BetaCalibrator enforces a, b >= 0 (Kull et al. 2017) ────────────────


class TestBetaCalibratorConstraint:
    def test_anti_calibrated_data_never_yields_rank_inverting_map(self):
        # NEGATIVE case: fully anti-calibrated data used to give a=-0.93,
        # b=-1.20 and a strictly DECREASING map (full rank inversion, silent).
        rng = np.random.default_rng(0)
        n = 2000
        p = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(0, 1, n) < (1 - p)).astype(int)
        cal = BetaCalibrator().fit(y, p)
        assert cal.a_ >= 0.0
        assert cal.b_ >= 0.0
        assert _is_monotone_nondecreasing(cal.transform(GRID))
        # the clamp must be disclosed, not silent
        assert len(cal.fit_result.warnings) > 0

    def test_partial_overconfidence_clamps_b_and_stays_monotone(self):
        # NEGATIVE case: model overconfident only in the top range used to give
        # b=-1.08 and a map decreasing over the top third of [0, 1].
        rng = np.random.default_rng(1)
        n = 4000
        p = rng.uniform(0.01, 0.999, n)
        true_rate = np.where(p > 0.85, 0.2, p)
        y = (rng.uniform(0, 1, n) < true_rate).astype(int)
        cal = BetaCalibrator().fit(y, p)
        assert cal.a_ >= 0.0
        assert cal.b_ >= 0.0
        assert _is_monotone_nondecreasing(cal.transform(GRID))
        assert len(cal.fit_result.warnings) > 0
        assert cal.fit_result.parameters["fitted_model"] in ("am", "bm", "m")

    def test_well_behaved_data_keeps_full_unconstrained_fit(self):
        # Does-not-overcorrect: ordinary miscalibration fits the full 'abm'
        # member with positive coefficients and NO warning.
        rng = np.random.default_rng(2)
        n = 4000
        p_true = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(0, 1, n) < p_true).astype(int)
        scores = p_true**2  # miscalibrated but rank-preserving
        cal = BetaCalibrator().fit(y, scores)
        assert cal.a_ > 0.0
        assert cal.b_ > 0.0
        assert cal.fit_result.warnings == []
        assert cal.fit_result.parameters["fitted_model"] == "abm"
        assert _is_monotone_nondecreasing(cal.transform(GRID))


# ── F2: calibration_slope verdict uses the [0.8, 1.2] band ──────────────────


class TestCalibrationSlopeVerdict:
    def test_signal_free_flat_slope_is_not_well_calibrated(self):
        # NEGATIVE case: slope ~0 (predictions carry no signal) used to
        # serialize is_well_calibrated=True via the ECE '< 0.05' fallback.
        rng = np.random.default_rng(2)
        n = 4000
        p_flat = np.where(rng.uniform(0, 1, n) < 0.5, 0.001, 0.999)
        y_ind = rng.integers(0, 2, n)
        res = calibration_slope(y_ind, p_flat)
        assert abs(res.overall_value) < 0.1  # the pathology: near-zero slope
        assert res.to_dict()["is_well_calibrated"] is False

    def test_near_perfect_slope_is_well_calibrated(self):
        # NEGATIVE case (other polarity): slope ~1 used to serialize False.
        rng = np.random.default_rng(3)
        n = 4000
        p = rng.uniform(0.001, 0.999, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        res = calibration_slope(y, p)
        assert 0.9 < res.overall_value < 1.1
        assert res.to_dict()["is_well_calibrated"] is True

    def test_band_edges_match_metadata_band(self):
        assert CalibrationMetricResult("calibration_slope", 0.8).is_well_calibrated
        assert CalibrationMetricResult("calibration_slope", 1.2).is_well_calibrated
        assert not CalibrationMetricResult("calibration_slope", 0.79).is_well_calibrated
        assert not CalibrationMetricResult("calibration_slope", 1.21).is_well_calibrated

    def test_error_style_metrics_unchanged(self):
        # Does-not-overcorrect: ECE/MCE/Brier thresholds are untouched, and CITL
        # keeps its meaningful |CITL| <= 0.05 reading.
        #
        # REVISED 2026-09-10 (readiness-6): the two ``some_new_metric`` lines
        # asserted the unknown-name FALLBACK, which was itself the defect (a
        # hardcoded lower-is-better rule with an ECE bound, applied to any name,
        # with no third state). They moved to
        # test_audit_wave4_postproc.py::TestIsWellCalibratedMetricAware, where
        # the fallback was originally pinned, and now assert None. What THIS
        # test pins is unchanged: fixing calibration_slope must not disturb the
        # metrics that were already graded correctly, and both of those are
        # still asserted in both directions below.
        assert CalibrationMetricResult("expected_calibration_error", 0.04).is_well_calibrated
        assert not CalibrationMetricResult("expected_calibration_error", 0.06).is_well_calibrated
        assert CalibrationMetricResult("maximum_calibration_error", 0.08).is_well_calibrated
        assert not CalibrationMetricResult("maximum_calibration_error", 0.12).is_well_calibrated
        assert CalibrationMetricResult("brier_score", 0.20).is_well_calibrated
        assert not CalibrationMetricResult("brier_score", 0.30).is_well_calibrated
        assert CalibrationMetricResult("calibration_in_the_large", 0.03).is_well_calibrated
        assert not CalibrationMetricResult("calibration_in_the_large", 0.09).is_well_calibrated


# ── F3: sufficiency_test zero evidence is not a PASS ────────────────────────


class TestSufficiencyZeroEvidence:
    def test_all_bins_skipped_is_not_assessable_never_pass(self):
        # NEGATIVE case: non-overlapping group score distributions leave zero
        # testable bins while group B is ~0.45 miscalibrated; this used to
        # return passes=True.
        rng = np.random.default_rng(3)
        nA = nB = 300
        p = np.concatenate([rng.uniform(0.0, 0.0999, nA), rng.uniform(0.9, 0.9999, nB)])
        g = np.array(["A"] * nA + ["B"] * nB)
        y = rng.integers(0, 2, nA + nB)
        r = sufficiency_test(y, p, g, n_bins=10)
        overall = r["overall"]
        assert overall["passes"] is not True
        assert overall["passes"] is None
        assert overall["status"] == "not_assessable"
        assert overall["n_bins_tested"] == 0
        assert overall["n_bins_skipped"] == 10

    def test_overlapping_data_is_assessed_with_boolean_verdict(self):
        # Does-not-overcorrect: ordinary data still gets a real boolean.
        rng = np.random.default_rng(4)
        n = 2000
        p = rng.uniform(0.001, 0.999, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        g = rng.choice(["A", "B"], n)
        r = sufficiency_test(y, p, g, n_bins=10)
        assert r["overall"]["status"] == "assessed"
        assert isinstance(r["overall"]["passes"], bool)
        assert r["overall"]["n_bins_tested"] > 0


# ── F4: sufficiency_test verdict is multiplicity-corrected ──────────────────


class TestSufficiencyFalseFailRate:
    def test_false_fail_rate_near_alpha_on_perfectly_sufficient_data(self):
        # NEGATIVE case: perfectly sufficient data (both groups drawn from the
        # identical conditional law) used to FAIL ~30% of runs. BH-corrected,
        # the seeded rate must sit near the 5% alpha; 12% is far below the old
        # behavior and above any plausible corrected rate at these seeds
        # (measured 3.5% over 200 runs).
        runs = 60
        fails = 0
        for s in range(runs):
            rng = np.random.default_rng(10_000 + s)
            n = 3000
            p = rng.uniform(0.001, 0.999, n)
            y = (rng.uniform(0, 1, n) < p).astype(int)
            g = rng.choice(["A", "B"], n)
            r = sufficiency_test(y, p, g, n_bins=10)
            if r["overall"]["passes"] is False:
                fails += 1
        assert fails / runs <= 0.12

    def test_truly_insufficient_data_still_fails(self):
        # Does-not-overcorrect (power): group B's outcome law is shifted +0.3
        # inside overlapping bins; the corrected verdict must still FAIL.
        rng = np.random.default_rng(5)
        n = 4000
        p = rng.uniform(0.001, 0.999, n)
        g = rng.choice(["A", "B"], n)
        rate = np.where(g == "B", np.clip(p + 0.3, 0, 1), p)
        y = (rng.uniform(0, 1, n) < rate).astype(int)
        r = sufficiency_test(y, p, g, n_bins=10)
        assert r["overall"]["passes"] is False
        assert r["overall"]["n_bins_violated"] >= 1

    def test_adjusted_p_values_reported_per_tested_bin(self):
        rng = np.random.default_rng(6)
        n = 2000
        p = rng.uniform(0.001, 0.999, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        g = rng.choice(["A", "B"], n)
        r = sufficiency_test(y, p, g, n_bins=10)
        assert r["overall"]["correction"] == "benjamini_hochberg"
        for b in r["bins"]:
            assert "p_value_adjusted" in b
            assert b["p_value_adjusted"] >= b["p_value"] - 1e-12


# ── F5: ICI disparity is NaN when unmeasurable ──────────────────────────────


class TestIciDisparityInsufficientGroups:
    def _one_assessable_group(self):
        rng = np.random.default_rng(5)
        nA = 200
        yA = (rng.uniform(0, 1, nA) < 0.1).astype(int)
        pA = np.full(nA, 0.9)  # badly miscalibrated
        yB = rng.integers(0, 2, 10)  # below min_group_size
        pB = rng.uniform(0.3, 0.7, 10)
        y = np.concatenate([yA, yB])
        p = np.concatenate([pA, pB])
        g = np.array(["A"] * nA + ["B"] * 10)
        return y, p, g

    def test_single_assessable_group_yields_nan_disparity(self):
        # NEGATIVE case: this used to read ici_disparity=0.0, sealing 'perfect
        # parity' exactly when the disparity could not be measured, while
        # auroc_parity on the same data said NOT_ASSESSABLE.
        y, p, g = self._one_assessable_group()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ici = integrated_calibration_index(y, p, g)
        assert ici.n_groups == 1
        assert np.isnan(ici.ici_disparity)
        assert np.isfinite(ici.overall_ici)  # the overall statistic survives

    def test_with_ci_propagates_nan_never_zero_zero(self):
        # NEGATIVE case: the bootstrap used to produce a [0.0, 0.0] CI that
        # clears ANY equivalence margin the seal gates on.
        y, p, g = self._one_assessable_group()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = integrated_calibration_index_with_ci(y, p, g, random_state=0, n_bootstrap=50)
        assert np.isnan(r.point_estimate)
        assert np.isnan(r.lower_bound)
        assert np.isnan(r.upper_bound)

    def test_two_valid_groups_keep_finite_disparity(self):
        # Does-not-overcorrect: a real two-group disparity stays measurable.
        rng = np.random.default_rng(7)
        n = 2000
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        g = rng.choice(["A", "B"], n)
        p_biased = p.copy()
        p_biased[g == "B"] = np.clip(p_biased[g == "B"] + 0.2, 0, 1)
        ici = integrated_calibration_index(y, p_biased, g)
        assert np.isfinite(ici.ici_disparity)
        assert ici.ici_disparity > 0.05

    def test_no_protected_attr_keeps_zero_disparity(self):
        # Does-not-overcorrect: with no group analysis requested, no disparity
        # question was asked; the documented 0.0 stays.
        rng = np.random.default_rng(8)
        n = 500
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        ici = integrated_calibration_index(y, p)
        assert ici.ici_disparity == 0.0
        assert np.isfinite(ici.overall_ici)


# ── F6: IsotonicCalibrator reproduces the PAVA solution ─────────────────────


class TestIsotonicMatchesPava:
    def test_matches_sklearn_isotonic_at_training_points(self):
        # NEGATIVE case: interior points used to be pulled toward the
        # neighbouring block (max deviation ~0.12, residual training ECE ~3%).
        sklearn_iso = pytest.importorskip("sklearn.isotonic")
        rng = np.random.default_rng(6)
        n = 800
        p = rng.uniform(0.01, 0.99, n)
        scores = p**2
        y = (rng.uniform(0, 1, n) < p).astype(int)
        ours = IsotonicCalibrator().fit(y, scores).transform(scores)
        ref = sklearn_iso.IsotonicRegression(out_of_bounds="clip").fit(scores, y).predict(scores)
        assert float(np.max(np.abs(ours - ref))) < 1e-8

    def test_transform_is_monotone_and_clips_out_of_range(self):
        # Does-not-overcorrect: monotonicity and clip behaviour survive.
        rng = np.random.default_rng(7)
        n = 600
        p = rng.uniform(0.05, 0.95, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        cal = IsotonicCalibrator(out_of_bounds="clip").fit(y, p)
        out = cal.transform(GRID)
        assert _is_monotone_nondecreasing(out)
        assert np.all((out >= 0.0) & (out <= 1.0))
        # below/above the training range: clipped to the end values
        lo, hi = cal.transform(np.array([0.0, 1.0]))
        assert lo == pytest.approx(cal.y_values_[0])
        assert hi == pytest.approx(cal.y_values_[-1])


# ── F7: multicalibration weighted_mean and group_max evidence handling ──────


class TestMulticalibrationSecondaryStats:
    def _audit_scenario(self):
        # group A: 100 rows in one evaluated cell violating by 0.45
        yA = np.array([1] * 5 + [0] * 95)
        pA = np.full(100, 0.5)
        # group B: 90 rows spread 9-per-bin, all below min_cell=10
        yB = np.tile(np.array([0, 1]), 45)[:90]
        pB = np.concatenate([np.full(9, (b + 0.5) / 10) for b in range(10)])
        y = np.concatenate([yA, yB])
        p = np.concatenate([pA, pB])
        g = np.array(["A"] * 100 + ["B"] * 90)
        return y, p, g

    def test_weighted_mean_normalizes_over_evaluated_support(self):
        # NEGATIVE case: with EVERY evaluated cell violating by 0.45 the
        # 'support-weighted average violation' used to read 0.237 because the
        # 90 unevaluated rows stayed in the denominator.
        y, p, g = self._audit_scenario()
        mc = multicalibration(y, p, g, min_cell=10, min_group_size=30)
        assert mc.alpha == pytest.approx(0.45, abs=1e-9)
        assert mc.weighted_mean == pytest.approx(0.45, abs=1e-9)

    def test_group_with_zero_evaluated_cells_is_nan_not_zero(self):
        # NEGATIVE case: group B had zero evaluated cells yet read 0.0
        # ('perfectly calibrated') in group_max.
        y, p, g = self._audit_scenario()
        mc = multicalibration(y, p, g, min_cell=10, min_group_size=30)
        assert np.isnan(mc.group_max["B"])
        assert mc.group_max["A"] == pytest.approx(0.45, abs=1e-9)

    def test_fully_evaluated_data_matches_full_support_average(self):
        # Does-not-overcorrect: when every cell is evaluated the denominator is
        # the full dataset, so the value equals the pre-fix definition.
        rng = np.random.default_rng(9)
        n = 2000
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        g = rng.choice(["A", "B"], n)
        mc = multicalibration(y, p, g, min_cell=1, min_group_size=30)
        # hand-computed full-support average
        expected = 0.0
        for grp in ("A", "B"):
            m = g == grp
            yp, yt = p[m], y[m]
            idx = np.clip(np.digitize(yp, np.linspace(0, 1, 11)[1:-1]), 0, 9)
            for b in range(10):
                cell = idx == b
                if cell.sum() == 0:
                    continue
                expected += cell.sum() * abs(float(np.mean(yt[cell])) - float(np.mean(yp[cell])))
        expected /= n
        assert mc.weighted_mean == pytest.approx(expected, abs=1e-12)
        assert np.isfinite(mc.alpha)

    def test_zero_evidence_alpha_nan_guard_still_holds(self):
        # The 0b8bf54 zero-evidence guard must survive: no cell meets min_cell.
        y = np.array([0, 1] * 20)
        p = np.linspace(0.01, 0.99, 40)
        g = np.array(["A", "B"] * 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mc = multicalibration(y, p, g, min_cell=50, min_group_size=10)
        assert np.isnan(mc.alpha)
        assert np.isnan(mc.weighted_mean)


# ── F8: ece_confidence_intervals discloses the near-null noise floor ────────


class TestEceCiNearNullDisclosure:
    def test_perfectly_calibrated_group_is_flagged_near_null(self):
        # NEGATIVE case: for a well-calibrated group the percentile CI is
        # bounded away from 0 (0/60 coverage of the true value in the audit);
        # the result must disclose that the point sits inside the plug-in
        # estimator's own noise.
        rng = np.random.default_rng(20_000)
        n = 600
        p = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        g = np.array(["A"] * n)
        r = ece_confidence_intervals(y, p, g, n_bootstrap=200, random_state=0)["A"]
        assert r["noise_floor"] > 0.0
        assert r["near_null"] is True

    def test_truly_miscalibrated_group_is_not_flagged(self):
        # Does-not-overcorrect: a real ECE of 0.10 must not be excused as noise.
        rng = np.random.default_rng(30_000)
        n = 600
        p = rng.choice([0.25, 0.75], n)
        rates = np.where(p == 0.25, 0.35, 0.65)
        y = (rng.uniform(0, 1, n) < rates).astype(int)
        g = np.array(["A"] * n)
        r = ece_confidence_intervals(y, p, g, n_bootstrap=200, random_state=0)["A"]
        assert r["near_null"] is False
        assert r["ece"] > r["noise_floor"]


# ── R-6: a saturated or zero-resample permutation design is not a PASS ───────


def _sufficiency_data(n=60_000, n_bins=10, defect_bin=5, rate_a=0.55, rate_b=0.95, seed=7):
    """Sufficiency holds everywhere except ONE score band.

    In that band group B's realised positive rate is ``rate_b`` and group A's is
    ``rate_a`` at the SAME predicted score, which is exactly what sufficiency
    says cannot happen. Everywhere else P(Y=1 | s) = s for both groups.
    """
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.0, 1.0, n)
    grp = np.where(rng.random(n) < 0.5, "A", "B")
    idx = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1)[1:-1])
    p = y_prob.copy()
    band = idx == defect_bin
    p[band & (grp == "A")] = rate_a
    p[band & (grp == "B")] = rate_b
    return (rng.random(n) < p).astype(int), y_prob, grp


class TestSufficiencyPermutationSaturation:
    """A test that could not have fired did not find sufficiency, it found nothing.

    ``method='permutation'`` estimates p as ``(count + 1) / (B + 1)``, so its
    floor is ``1 / (B + 1)``. ``_adaptive_n_bootstrap`` caps B at 200 above
    50,000 samples regardless of ``n_permutations=1000``, making the floor
    exactly 1/201 = 0.0049751, and Benjamini-Hochberg puts the rank-1 bar at
    ``0.05 / n_bins_tested``.

    MEASURED on this repo 2026-09-10, 60,000 samples, one score band where group
    B is positive 95 percent of the time against group A's 55 percent at the
    same predicted score:

        n_bins=10  raw p 0.0049751  adjusted 0.049751  passes=False
        n_bins=12  raw p 0.0049751  adjusted 0.059701  passes=True (!)

    The raw p is bit-for-bit the design floor in both rows, so the code knew the
    test had saturated and said nothing. At n_bins >= 11 this detector could not
    fire at all.
    """

    def test_a_saturated_design_reports_could_not_check_not_pass(self):
        y, p, g = _sufficiency_data(n_bins=12)

        overall = sufficiency_test(y, p, g, n_bins=12, method="permutation", n_permutations=1000)[
            "overall"
        ]

        assert overall["passes"] is not True
        assert overall["passes"] is None
        assert overall["status"] == "not_assessable"
        assert overall["detectable"] is False
        assert "NOT DETECTABLE" in overall["detectability_note"]

    def test_the_saturation_is_reported_as_a_number_not_only_as_prose(self):
        y, p, g = _sufficiency_data(n_bins=12)

        overall = sufficiency_test(y, p, g, n_bins=12, method="permutation", n_permutations=1000)[
            "overall"
        ]

        # 1/201: the floor of a 200-resample design, which is what the adaptive
        # cap leaves at 60,000 samples however many were requested.
        assert overall["min_attainable_p_value"] == pytest.approx(1.0 / 201.0)
        # The tell: the observed p IS the floor, so the test never had room to
        # say anything smaller.
        assert overall["min_bin_p_value"] == pytest.approx(overall["min_attainable_p_value"])

    def test_a_design_with_power_still_reports_the_violation(self):
        # OVER-CORRECTION CONTROL. Same data, one fewer bin, so the rank-1 bar
        # (0.05/10) sits just above the same floor: the finding must survive.
        y, p, g = _sufficiency_data(n_bins=10)

        overall = sufficiency_test(y, p, g, n_bins=10, method="permutation", n_permutations=1000)[
            "overall"
        ]

        assert overall["passes"] is False
        assert overall["status"] == "assessed"
        assert overall["detectable"] is True
        assert overall["n_bins_violated"] >= 1

    def test_sufficient_data_with_enough_resamples_still_passes(self):
        # OVER-CORRECTION CONTROL. A gate that never returns True is as useless
        # as one that never returns False.
        rng = np.random.default_rng(21)
        n = 2000
        p = rng.uniform(0.001, 0.999, n)
        g = rng.choice(["A", "B"], n)
        y = (rng.random(n) < p).astype(int)

        overall = sufficiency_test(y, p, g, n_bins=10, method="permutation", n_permutations=1000)[
            "overall"
        ]

        assert overall["passes"] is True
        assert overall["status"] == "assessed"
        assert overall["detectable"] is True
        assert overall["detectability_note"] == ""

    def test_a_real_violation_with_enough_resamples_still_fails(self):
        # OVER-CORRECTION CONTROL, the other polarity.
        rng = np.random.default_rng(22)
        n = 2000
        p = rng.uniform(0.001, 0.999, n)
        g = rng.choice(["A", "B"], n)
        rate = np.where(g == "B", np.clip(p + 0.30, 0, 1), p)
        y = (rng.random(n) < rate).astype(int)

        overall = sufficiency_test(y, p, g, n_bins=10, method="permutation", n_permutations=1000)[
            "overall"
        ]

        assert overall["passes"] is False
        assert overall["n_bins_violated"] >= 1

    def test_the_chi2_path_is_continuous_and_keeps_its_verdicts(self):
        # OVER-CORRECTION CONTROL. The default method has no discrete floor, so
        # the disclosure must not turn its verdicts into could-not-check.
        y, p, g = _sufficiency_data(n_bins=12)

        overall = sufficiency_test(y, p, g, n_bins=12, method="chi2")["overall"]

        assert overall["passes"] is False
        assert overall["detectable"] is True
        assert overall["min_attainable_p_value"] == 0.0

    def test_each_tested_bin_carries_its_own_power_disclosure(self):
        y, p, g = _sufficiency_data(n_bins=12)

        result = sufficiency_test(y, p, g, n_bins=12, method="permutation", n_permutations=1000)

        assert result["bins"]
        for entry in result["bins"]:
            assert entry["n_permutations_effective"] == 200
            assert entry["min_attainable_p_value"] == pytest.approx(1.0 / 201.0)
            assert entry["detectable"] is False


class TestSufficiencyPermutationCountIsValidated:
    """``n_permutations=0`` is not a cheap run, it is a test with no evidence.

    MEASURED on this repo 2026-09-10: the resample loop never executed, every
    bin's ``(count + 1) / (0 + 1)`` came out as exactly 1.0, and the function
    returned ``passes=True, status='assessed', n_bins_tested=10,
    n_bins_skipped=0`` with ZERO warnings, on data that ``method='chi2'``
    rejects at an adjusted p of 6.3e-276.
    """

    @pytest.mark.parametrize("bad", [0, -5, None, "x", 2.5j])
    def test_a_non_positive_or_non_integer_count_is_refused(self, bad):
        y, p, g = _sufficiency_data(n=1500, n_bins=4, defect_bin=2)

        with pytest.raises(ValueError, match="n_permutations"):
            sufficiency_test(y, p, g, n_bins=4, method="permutation", n_permutations=bad)

    def test_the_zero_resample_data_is_genuinely_a_violation(self):
        # The other half of the measurement: chi2 on the SAME data says FAIL, so
        # the old passes=True was not a quirk of easy data.
        y, p, g = _sufficiency_data(n_bins=10)

        overall = sufficiency_test(y, p, g, n_bins=10, method="chi2")["overall"]

        assert overall["passes"] is False
        assert overall["min_bin_p_value_adjusted"] < 1e-100

    def test_a_positive_but_powerless_count_is_accepted_and_disclosed(self):
        # Not every small count is a caller error, so it is accepted; what it
        # may NOT do is come back as a pass.
        y, p, g = _sufficiency_data(n=1500, n_bins=4, defect_bin=2)

        overall = sufficiency_test(y, p, g, n_bins=4, method="permutation", n_permutations=1)[
            "overall"
        ]

        assert overall["passes"] is not True
        assert overall["min_attainable_p_value"] == pytest.approx(0.5)
        assert overall["detectable"] is False
