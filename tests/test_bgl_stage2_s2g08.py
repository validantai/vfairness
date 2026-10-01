"""Beta Go-Live Stage 2, group s2g08: calibration metrics say NOTHING MEASURED.

Every finding in this group is the same shape: a value nobody measured, replaced
by a neutral default, then graded, counted, serialised or plotted as if it were a
measurement. All nine were reproduced at the public entry point before the fix,
and the before-values below are the ones this file's own repro printed on
2026-09-16 (numpy default_rng, so they are reproducible).

    F1  integrated_calibration_index(...).has_significant_disparity -> False
        (a clean bill of health on calibration parity) while ici_disparity was
        NaN and n_groups was 1.
    F2  calibration_in_the_large(np.array([]), np.array([])) -> overall_value
        0.0, is_well_calibrated True: the exact centre of the metric's own
        well-calibrated band, on ZERO rows. Separable input -> 25.87, the 100th
        IRLS iterate reported as a measurement.
    F3  calibration_slope: empty -> 0.0, constant scores (60 real rows) -> 0.0
        (the slope is NOT identifiable there), single outcome class -> -5.4e-16.
    F4  brier_score(...).to_dict()['max_group_disparity'] -> 0.0 with one group,
        and 0.0 on zero rows beside an overall_value of NaN.
    F5  brier_score_decomposition(all labels 1).skill_score -> 0.0, which reads
        as "exactly as skilful as climatology"; empty -> reliability/resolution
        0.0 initialisers.
    F6  calibration_curve([], []).overall_ece -> 0.0, and the public plot
        consumer captioned the empty chart "ECE = 0.000".
    F7  group_calibration_metrics with a=100, b=1 -> every max_group_disparity
        0.0 with ZERO warnings and group b missing from group_values.
    F8  maximum_calibration_error(...).max_group_disparity -> 0.0.
    F9  expected_calibration_error(...).max_group_disparity -> 0.0.

Each test asserts the THIRD state at the public entry, and each class also
carries a CONTROL proving healthy data still produces the measurement (a fix
that makes everything refuse is a worse defect and passes any degenerate-only
test).
"""

import warnings

import numpy as np
import pytest

import vfairness
from vfairness.post_processing.calibration import (
    brier_score,
    brier_score_decomposition,
    calibration_curve,
    calibration_in_the_large,
    calibration_slope,
    expected_calibration_error,
    group_calibration_metrics,
    integrated_calibration_index,
    maximum_calibration_error,
)

# --------------------------------------------------------------------------
# fixtures: built here rather than in a shared conftest (thirteen other agents
# are working in this checkout). Each one asserts what it is, so a test can
# never pass because the fixture missed the branch it is aiming at.
# --------------------------------------------------------------------------


def _well_calibrated(n=4000, seed=4):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def _one_row_minority(n_major=100):
    """a = n_major rows, b = 1 row. The default min_group_size=30 drops b."""
    rng = np.random.default_rng(7)
    n = n_major + 1
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p * 0.7).astype(int)
    attr = np.array(["a"] * n_major + ["b"])
    return y, p, attr


def _two_real_groups():
    """Two adequate groups with a genuinely different Brier/ECE/MCE."""
    rng = np.random.default_rng(5)
    n_a = n_b = 200
    y_a = rng.integers(0, 2, n_a)
    p_a = rng.uniform(0.4, 0.6, n_a)
    y_b = np.ones(n_b, dtype=int)
    p_b = rng.uniform(0.0, 0.2, n_b)
    y = np.concatenate([y_a, y_b])
    p = np.concatenate([p_a, p_b])
    attr = np.array(["a"] * n_a + ["b"] * n_b)
    return y, p, attr


EMPTY_Y = np.array([], dtype=int)
EMPTY_P = np.array([], dtype=float)


def _silently(fn, *args, **kwargs):
    """Call fn ignoring warnings, and return (result, list_of_warning_texts)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# --------------------------------------------------------------------------
# F1  integrated_calibration_index.has_significant_disparity
# --------------------------------------------------------------------------


class TestF1IciSignificantDisparityThreeState:
    def test_single_group_verdict_is_none_not_false(self):
        # BEFORE: ici_disparity nan, has_significant_disparity False (a bare
        # bool), n_groups 1, excluded_groups [], NO warnings.
        y, p = _well_calibrated(n=200, seed=0)
        attr = np.array(["A"] * 200)
        result, texts = _silently(integrated_calibration_index, y, p, attr)

        # the fixture really does reach the fewer-than-two-groups branch
        assert result.n_groups == 1
        assert np.isnan(result.ici_disparity)

        assert result.has_significant_disparity is None
        assert result.to_dict()["has_significant_disparity"] is None
        # and the third state is announced, not left to a NaN in a sibling field
        assert any("no between-group comparison" in t for t in texts)

    def test_no_disparity_and_real_disparity_are_still_graded(self):
        # CONTROL, both polarities. A measured gap under 0.05 must still be
        # False, and a measured gap over 0.05 must still be True; if either had
        # become None the fix would have deleted the capability.
        y, p = _well_calibrated(n=4000, seed=4)
        rng = np.random.default_rng(4)
        attr = rng.choice(["A", "B"], 4000)
        small = integrated_calibration_index(y, p, attr)
        assert small.n_groups == 2
        assert np.isfinite(small.ici_disparity)
        assert small.ici_disparity < 0.05
        assert small.has_significant_disparity is False

        rng = np.random.default_rng(7)
        n = 2000
        p2 = rng.uniform(0, 1, n)
        y2 = (rng.uniform(0, 1, n) < p2).astype(int)
        g2 = rng.choice(["A", "B"], n)
        p_biased = p2.copy()
        p_biased[g2 == "B"] = np.clip(p_biased[g2 == "B"] + 0.2, 0, 1)
        big = integrated_calibration_index(y2, p_biased, g2)
        assert big.ici_disparity > 0.05
        assert big.has_significant_disparity is True


# --------------------------------------------------------------------------
# F2  calibration_in_the_large
# --------------------------------------------------------------------------


class TestF2CalibrationInTheLarge:
    def test_zero_rows_refuse_instead_of_the_band_centre(self):
        # BEFORE: overall_value 0.0, is_well_calibrated True, n_samples 0.
        result, texts = _silently(calibration_in_the_large, EMPTY_Y, EMPTY_P)
        assert result.n_samples == 0
        assert np.isnan(result.overall_value)
        assert result.is_well_calibrated is None
        assert result.metadata["measured"] is False
        assert "no rows" in result.metadata["unmeasured_reason"]
        assert any("could not be measured" in t for t in texts)

    def test_separable_input_refuses_instead_of_reporting_the_iterate(self):
        # BEFORE: 60 rows, every label 1 -> overall_value 25.93, a non-converged
        # IRLS iterate, graded False with no warning of any kind.
        y = np.ones(60, dtype=int)
        p = np.linspace(0.05, 0.95, 60)
        assert np.unique(y).size == 1  # the fixture is the separable one
        result, texts = _silently(calibration_in_the_large, y, p)
        assert np.isnan(result.overall_value)
        assert result.is_well_calibrated is None
        assert "separation" in result.metadata["unmeasured_reason"]
        assert texts

    def test_healthy_data_is_still_measured_and_graded(self):
        # CONTROL, both polarities.
        y, p = _well_calibrated(n=4000, seed=4)
        good = calibration_in_the_large(y, p)
        assert np.isfinite(good.overall_value)
        assert good.overall_value < 0.05
        assert good.is_well_calibrated is True
        assert good.metadata["measured"] is True

        rng = np.random.default_rng(1)
        p_bad = rng.uniform(0.02, 0.78, 4000)
        y_bad = (rng.uniform(size=4000) < np.clip(p_bad - 0.2, 0.01, 0.99)).astype(int)
        bad = calibration_in_the_large(y_bad, p_bad)
        assert np.isfinite(bad.overall_value)
        assert bad.overall_value > 0.05
        assert bad.is_well_calibrated is False


# --------------------------------------------------------------------------
# F3  calibration_slope
# --------------------------------------------------------------------------


class TestF3CalibrationSlope:
    def test_zero_rows_refuse(self):
        # BEFORE: overall_value 0.0, graded False against the [0.8, 1.2] band.
        result, texts = _silently(calibration_slope, EMPTY_Y, EMPTY_P)
        assert np.isnan(result.overall_value)
        assert result.is_well_calibrated is None
        assert result.metadata["measured"] is False
        assert texts

    def test_constant_scores_refuse_because_the_slope_is_not_identifiable(self):
        # BEFORE: 60 REAL rows at p=0.7 -> overall_value 0.0, which a reader
        # reads as "predictions carry no signal". The slope is non-identifiable
        # there: logit(p) is collinear with the intercept.
        y = np.array([1, 0] * 30)
        p = np.full(60, 0.7)
        # the fixture exercises the zero-variance branch and NOT the empty or
        # single-class ones
        assert len(y) == 60
        assert np.unique(y).size == 2
        assert np.unique(p).size == 1

        result, texts = _silently(calibration_slope, y, p)
        assert np.isnan(result.overall_value)
        assert result.is_well_calibrated is None
        assert "not identifiable" in result.metadata["unmeasured_reason"]
        assert texts

    def test_single_outcome_class_refuses(self):
        # BEFORE: 60 rows all y=1 -> overall_value 0.012 (and -5.4e-16 on the
        # linspace variant), a separable fit read off its 100th iterate.
        y = np.ones(60, dtype=int)
        p = np.linspace(0.05, 0.95, 60)
        result, _ = _silently(calibration_slope, y, p)
        assert np.isnan(result.overall_value)
        assert result.is_well_calibrated is None
        assert "single outcome class" in result.metadata["unmeasured_reason"]

    def test_healthy_slope_and_a_real_flat_slope_both_survive(self):
        # CONTROL. The near-1.0 slope must stay measured AND graded True, and
        # the signal-free-but-MEASURABLE slope near 0 must stay a measured
        # finding (False), not become a refusal: that reading is real.
        rng = np.random.default_rng(3)
        p = rng.uniform(0.001, 0.999, 4000)
        y = (rng.uniform(0, 1, 4000) < p).astype(int)
        good = calibration_slope(y, p)
        assert 0.9 < good.overall_value < 1.1
        assert good.is_well_calibrated is True
        assert good.metadata["measured"] is True

        rng = np.random.default_rng(2)
        p_flat = np.where(rng.uniform(0, 1, 4000) < 0.5, 0.001, 0.999)
        y_ind = rng.integers(0, 2, 4000)
        assert np.unique(p_flat).size == 2  # identifiable: two distinct scores
        flat = calibration_slope(y_ind, p_flat)
        assert np.isfinite(flat.overall_value)
        assert abs(flat.overall_value) < 0.1
        assert flat.to_dict()["is_well_calibrated"] is False
        assert flat.metadata["measured"] is True


# --------------------------------------------------------------------------
# F4 / F8 / F9  max_group_disparity on brier_score, MCE and ECE
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [brier_score, expected_calibration_error, maximum_calibration_error],
    ids=["brier_score", "expected_calibration_error", "maximum_calibration_error"],
)
class TestF4F8F9MaxGroupDisparity:
    def test_one_group_disparity_is_nan_not_zero(self, fn):
        # BEFORE: {'group_values': {'a': 0.328}, 'max_group_disparity': 0.0}
        # for a single group, i.e. PERFECT PARITY from one measurement.
        y, p = _well_calibrated(n=60, seed=0)
        attr = np.array(["a"] * 60)
        result = fn(y, p, attr)
        assert result.n_groups_compared == 1  # the branch really was reached
        assert np.isnan(result.to_dict()["max_group_disparity"])

    def test_silently_dropped_minority_is_nan_and_disclosed(self, fn):
        # BEFORE: group b dropped by min_group_size with ZERO warnings and
        # max_group_disparity 0.0.
        y, p, attr = _one_row_minority()
        result, texts = _silently(fn, y, p, attr)
        assert "b" in set(attr)  # the input really has the minority group
        assert "b" not in (result.group_values or {})  # and it really was dropped
        assert np.isnan(result.max_group_disparity)
        assert result.to_dict()["n_groups_compared"] == 1
        assert result.metadata["excluded_groups"] == ["b"]
        assert any("NOT assessed" in t for t in texts)

    def test_zero_rows_disparity_is_nan_not_zero(self, fn):
        # BEFORE: overall_value NaN beside max_group_disparity 0.0 on an empty
        # dataset: a refusal and a perfect-parity claim in the same object.
        result, _ = _silently(fn, EMPTY_Y, EMPTY_P, np.array([]))
        assert result.to_dict()["n_groups_compared"] == 0
        assert np.isnan(result.to_dict()["max_group_disparity"])

    def test_two_real_groups_keep_a_measured_disparity(self, fn):
        # CONTROL: a genuine between-group gap is still measured and finite.
        y, p, attr = _two_real_groups()
        result = fn(y, p, attr)
        assert result.n_groups_compared == 2
        disparity = result.to_dict()["max_group_disparity"]
        assert np.isfinite(disparity)
        assert disparity > 0.1
        values = list(result.group_values.values())
        assert disparity == pytest.approx(max(values) - min(values))


# --------------------------------------------------------------------------
# F5  brier_score_decomposition.skill_score
# --------------------------------------------------------------------------


class TestF5BrierSkillScore:
    def test_single_class_labels_make_the_skill_score_nan(self):
        # BEFORE: {'brier_score': 0.2774, 'uncertainty': 0.0,
        # 'skill_score': 0.0} with np.isfinite(skill_score) True and no
        # warning. 0.0 is a real, plottable verdict on that scale.
        rng = np.random.default_rng(0)
        y = np.ones(200, dtype=int)
        p = rng.uniform(0.1, 0.9, 200)
        result, texts = _silently(vfairness.brier_score_decomposition, y, p)
        assert result.uncertainty == 0.0  # the fixture reaches the branch
        assert np.isnan(result.skill_score)
        assert np.isnan(result.to_dict()["skill_score"])
        assert any("undefined" in t for t in texts)

    def test_zero_rows_leave_no_component_at_its_initialiser(self):
        # BEFORE: {'reliability': 0.0, 'resolution': 0.0} beside a NaN brier.
        result, _ = _silently(brier_score_decomposition, EMPTY_Y, EMPTY_P)
        assert result.n_samples == 0
        assert np.isnan(result.reliability)
        assert np.isnan(result.resolution)
        assert np.isnan(result.skill_score)

    def test_two_class_labels_keep_a_real_skill_score(self):
        # CONTROL: with a real climatology the BSS is measured and equals
        # 1 - Brier / Uncertainty.
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, 400)
        y = (rng.uniform(size=400) < p).astype(int)
        result = brier_score_decomposition(y, p)
        assert result.uncertainty > 0
        assert np.isfinite(result.skill_score)
        assert result.skill_score == pytest.approx(1 - result.brier_score / result.uncertainty)
        assert np.isfinite(result.reliability)
        assert np.isfinite(result.resolution)


# --------------------------------------------------------------------------
# F6  calibration_curve.overall_ece (and the plot that captions it)
# --------------------------------------------------------------------------


class TestF6CalibrationCurveEce:
    def test_zero_rows_give_nan_ece_not_perfect_calibration(self):
        # BEFORE: overall_ece 0.0 (python float), zero warnings, and to_dict()
        # shipped 'overall_ece': 0.0 while the sibling
        # expected_calibration_error returned NaN on the identical input.
        result, texts = _silently(calibration_curve, EMPTY_Y, EMPTY_P)
        assert len(result.bin_counts) == 0  # no bin received a sample
        assert np.isnan(result.overall_ece)
        assert np.isnan(result.to_dict()["overall_ece"])
        assert any("not 0.0" in t for t in texts)
        # and it now agrees with the sibling that already refused
        sibling = expected_calibration_error(EMPTY_Y, EMPTY_P)
        assert np.isnan(sibling.overall_value)

    def test_the_plot_consumer_no_longer_captions_perfect_calibration(self):
        # The stronger surface: a reader was shown an empty chart captioned
        # "ECE = 0.000" for a dataset with nothing in it.
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        ax, _ = _silently(vfairness.plot_reliability_diagram, EMPTY_Y, EMPTY_P)
        captions = [t.get_text() for t in ax.texts]
        assert captions  # the caption is still drawn
        assert not any("0.000" in c for c in captions)
        assert any("nan" in c.lower() for c in captions)

    def test_healthy_curve_still_measures_its_ece(self):
        # CONTROL: a populated curve keeps the accumulated value.
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 400)
        y = (rng.uniform(size=400) < p * 0.5).astype(int)
        result = calibration_curve(y, p)
        assert len(result.bin_counts) > 0
        assert np.isfinite(result.overall_ece)
        assert result.overall_ece > 0.05
        expected = float(
            np.sum(
                (result.bin_counts / result.bin_counts.sum())
                * np.abs(result.prob_true - result.prob_pred)
            )
        )
        assert result.overall_ece == pytest.approx(expected)


# --------------------------------------------------------------------------
# F7  group_calibration_metrics
# --------------------------------------------------------------------------


class TestF7GroupCalibrationMetrics:
    def test_dropped_minority_makes_every_disparity_nan_and_is_disclosed(self):
        # BEFORE: a=100, b=1 -> ece/mce/brier max_group_disparity all 0.0 with
        # ZERO warnings, while calibration_disparity in the SAME file, on the
        # SAME input, already refused with a warning reading "NaN, not 0.0".
        y, p, attr = _one_row_minority()
        metrics, texts = _silently(group_calibration_metrics, y, p, attr)
        for key in ("ece", "mce", "brier"):
            result = metrics[key]
            assert "b" not in result.group_values  # the drop really happened
            assert np.isnan(result.max_group_disparity), key
            assert np.isnan(result.to_dict()["max_group_disparity"]), key
            assert result.to_dict()["n_groups_compared"] == 1, key
            assert result.metadata["excluded_groups"] == ["b"], key
        assert len([t for t in texts if "NOT assessed" in t]) == 3

        # the sibling this fix was modelled on still agrees
        disparity, _ = _silently(vfairness.calibration_disparity, y, p, attr)
        assert np.isnan(disparity.ece_disparity)
        assert disparity.has_significant_disparity is None

    def test_two_adequate_groups_still_measure_every_disparity(self):
        # CONTROL.
        y, p, attr = _two_real_groups()
        metrics = group_calibration_metrics(y, p, attr)
        for key in ("ece", "mce", "brier"):
            result = metrics[key]
            assert result.n_groups_compared == 2, key
            assert np.isfinite(result.max_group_disparity), key
            assert result.max_group_disparity > 0.1, key
            assert result.metadata["excluded_groups"] == [], key


# --------------------------------------------------------------------------
# NEW FINDING, exposed by F1 and fixed in the same pass because it lives in the
# same file: MulticalibrationResult.is_multicalibrated was the same bare
# comparison one class up. BEFORE: 60 rows over 60 bins at min_cell=10 ->
# alpha nan, is_multicalibrated False, and to_dict() shipped that False.
# --------------------------------------------------------------------------


class TestNewFindingMulticalibratedVerdict:
    def test_zero_evaluated_cells_give_none_not_false(self):
        rng = np.random.default_rng(0)
        n = 60
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(size=n) < p).astype(int)
        attr = np.array(["A"] * 30 + ["B"] * 30)
        result, _ = _silently(vfairness.multicalibration, y, p, attr, n_bins=60, min_cell=10)
        assert np.isnan(result.alpha)  # the fixture really evaluates no cell
        assert result.is_multicalibrated is None
        assert result.to_dict()["is_multicalibrated"] is None

    def test_measured_alphas_are_still_graded_both_ways(self):
        # CONTROL, both polarities.
        y, p = _well_calibrated(n=2000, seed=4)
        attr = np.random.default_rng(4).choice(["A", "B"], 2000)
        good, _ = _silently(vfairness.multicalibration, y, p, attr)
        assert np.isfinite(good.alpha)
        assert good.is_multicalibrated is (good.alpha < 0.05)

        p_biased = p.copy()
        p_biased[attr == "B"] = np.clip(p_biased[attr == "B"] + 0.3, 0, 1)
        bad, _ = _silently(vfairness.multicalibration, y, p_biased, attr)
        assert bad.alpha > 0.05
        assert bad.is_multicalibrated is False
