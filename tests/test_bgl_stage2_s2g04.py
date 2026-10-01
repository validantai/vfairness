"""Beta Go-Live Stage 2, group s2g04: the calibration methods unit.

Every pin here is written at a PUBLIC entry point (``vfairness.PlattScaling``,
``vfairness.HistogramBinning``, ``vfairness.BetaCalibrator``,
``vfairness.IsotonicCalibrator``, ``vfairness.create_calibrator``), with the
value measured BEFORE the fix quoted in the docstring, and each degenerate case
paired with a healthy-data control in the same class so a fix that makes
everything refuse cannot pass.

Findings pinned (all in src/vfairness/post_processing/calibration/methods.py):

  S2G04-1  PlattScaling.fit_transform returned a flat array from a fit that
           identified nothing.
  S2G04-2  PlattScaling.fit published its own 0.0 initialisation as the fitted
           parameters when the Newton system was singular at iteration 0.
  S2G04-3  HistogramBinning.transform put a NaN score in the LAST bin and
           returned that bin's rate; out-of-range scores were clamped the same
           way.
  S2G04-4  HistogramBinning.fit gave a bin holding ZERO training samples the
           bin MIDPOINT, which is the identity map, and recorded no counts.
  S2G04-5  HistogramBinning.fit_transform installed those midpoints and
           published absolute 0.0/1.0 rates from single-observation bins.
  S2G04-6  BetaCalibrator.fit returned _newton_fit's zero initialisation as the
           estimate on a rank-deficient design.
  S2G04-7  create_calibrator: fit_result.warnings was [] on every path, so the
           durable record asserted a clean fit while the same call emitted a
           UserWarning saying nothing could be learned.
  S2G04-8  IsotonicCalibrator.fit serialised a one-knot (constant) map
           identically to a healthy fit, and reported the pre-tie-collapse PAVA
           block count as if it were the map's resolution.
"""

import json
import warnings

import numpy as np
import pytest

from vfairness import (
    BetaCalibrator,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    create_calibrator,
)
from vfairness.exceptions import InvalidDataError

CALIBRATOR_NAMES = ("platt", "isotonic", "beta", "temperature", "histogram")


def _healthy(n=400, seed=0):
    """Ordinary miscalibrated-but-informative data: every calibrator fits it."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(0, 1, n) < scores).astype(int)
    return y, scores


def _no_warnings(recorded):
    return [str(w.message) for w in recorded]


# ── S2G04-1 / S2G04-2: PlattScaling on a design that identifies nothing ──────


class TestPlattUnidentifiedFit:
    def test_fit_transform_on_constant_scores_does_not_return_a_flat_number(self):
        """S2G04-1. Measured before the fix: fit_transform(y, np.full(60, 0.5))
        returned ndarray(60,) with np.unique -> [0.5], every row 0.5, with
        is_fitted True, a_=b_=0.0 and fit_result.warnings == []. A constant 0.8
        column at base rate 0.10 gave the IDENTICAL flat [0.5] and the identical
        a_=b_=0.0, so the value did not depend on the data at all.
        """
        y = np.array([1] * 30 + [0] * 30)
        scores = np.full(60, 0.5)
        # The fixture really does exercise the singular-Hessian branch: one
        # distinct score means every logit is identical.
        assert np.unique(scores).size == 1

        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        assert out.shape == (60,)
        assert np.all(np.isnan(out)), np.unique(out)
        assert not np.any(out == 0.5)
        assert np.isnan(cal.a_) and np.isnan(cal.b_)
        assert cal.fit_result.warnings, "the durable record still reads as a clean fit"
        assert any("rank-deficient" in w for w in cal.fit_result.warnings)
        assert any("rank-deficient" in str(w.message) for w in caught)

    def test_fit_on_constant_scores_records_the_refusal_in_the_durable_record(self):
        """S2G04-2. Measured before the fix: fit_result.to_dict() ==
        {"method": "platt_scaling", "n_samples": 60,
         "parameters": {"a": 0.0, "b": 0.0},
         "fit_metrics": {"log_loss": 0.6931471805599454, "iterations": 1},
         "warnings": []} with ZERO Python warnings, and
        transform([0.01, 0.5, 0.99]) -> [0.5 0.5 0.5]. 0.6931 is ln 2, the
        coin-flip loss, reported as a fit metric.
        """
        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.array([1] * 12 + [0] * 48), np.full(60, 0.7))

        record = cal.fit_result.to_dict()
        assert np.isnan(record["parameters"]["a"])
        assert np.isnan(record["parameters"]["b"])
        assert record["parameters"]["a"] != 0.0
        assert np.isnan(record["fit_metrics"]["log_loss"])
        assert record["fit_metrics"]["log_loss"] != pytest.approx(np.log(2))
        assert record["warnings"], "warnings == [] asserts a clean fit that never happened"
        assert len(caught) == 1 and "could-not-check" in str(caught[0].message)
        assert np.all(np.isnan(cal.transform([0.01, 0.5, 0.99])))
        # The refusal survives serialisation, which is how a report reads it.
        assert "rank-deficient" in json.dumps(record["warnings"])

    def test_single_class_fit_is_recorded_and_not_only_warned(self):
        """S2G04-2 / S2G04-7, second half. Before the fix the single-class path
        DID emit a UserWarning, and fit_result.warnings was still [].
        """
        rng = np.random.default_rng(1)
        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.ones(60, dtype=int), rng.uniform(0.05, 0.95, 60))
        assert any("SINGLE class" in str(w.message) for w in caught)
        assert any("SINGLE class" in w for w in cal.fit_result.warnings)

    def test_control_healthy_data_still_fits_a_real_platt_map(self):
        """CONTROL. Before the fix, n=400 healthy data fitted a=0.965, b=-0.095
        and transformed [0.01, 0.5, 0.99] to [0.0107, 0.476, 0.987]. It must
        still do that, with an empty warnings list and no Python warning.
        """
        y, scores = _healthy(400)
        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)
            out = cal.transform(np.array([0.01, 0.5, 0.99]))

        assert not caught, _no_warnings(caught)
        assert cal.fit_result.warnings == []
        assert np.isfinite(cal.a_) and np.isfinite(cal.b_)
        assert cal.a_ > 0.5
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) > 0), out
        assert np.isfinite(cal.fit_result.fit_metrics["log_loss"])

    def test_control_fit_transform_on_healthy_data_is_not_constant(self):
        """CONTROL for S2G04-1: 60 distinct calibrated values, as measured."""
        y, scores = _healthy(60, seed=5)
        out = PlattScaling().fit_transform(y, scores)
        assert np.all(np.isfinite(out))
        assert np.unique(out).size == 60


# ── S2G04-3: HistogramBinning.transform on unscoreable / out-of-range input ──


class TestHistogramTransformRefusesUnscoreableInput:
    def _fitted(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 200)
        y = (rng.uniform(0, 1, 200) < scores).astype(int)
        return HistogramBinning(n_bins=10, strategy="uniform").fit(y, scores)

    def test_a_nan_score_is_not_given_the_top_bins_rate(self):
        """S2G04-3. Measured before the fix: transform(np.full(5, np.nan)) ->
        [0.96666667] x5, byte-identical to transform([0.95]) -> [0.96666667],
        i.e. bin 9's empirical rate, the HIGHEST-risk calibrated probability the
        fitted object can produce, handed to rows that have no score at all.
        """
        cal = self._fitted()
        # the fixture really does place NaN in the last bin under np.digitize
        assert int(np.digitize(np.nan, cal.bin_edges_[1:-1])) == len(cal.bin_values_) - 1
        top_bin_rate = float(cal.transform(np.array([0.95]))[0])
        assert np.isfinite(top_bin_rate)

        with pytest.raises(InvalidDataError, match="NaN"):
            cal.transform(np.full(5, np.nan))

    def test_out_of_range_scores_are_not_clamped_into_the_end_bins(self):
        """S2G04-3. Measured before the fix: transform([1.7, -3, 42]) ->
        [0.96666667 0.04761905 0.96666667]."""
        cal = self._fitted()
        with pytest.raises(InvalidDataError, match=r"\[0, 1\]"):
            cal.transform(np.array([1.7, -3.0, 42.0]))

    def test_control_in_range_scores_still_get_their_bins_rate(self):
        """CONTROL: the healthy path is byte-unchanged."""
        cal = self._fitted()
        out = cal.transform(np.array([0.05, 0.35, 0.95]))
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(cal.bin_values_[0])
        assert out[2] == pytest.approx(cal.bin_values_[9])
        assert cal.fit_result.warnings == []


# ── S2G04-4 / S2G04-5: HistogramBinning.fit on bins nobody observed ──────────


class TestHistogramUnobservedBins:
    def test_an_unobserved_bin_is_not_given_the_identity_map(self):
        """S2G04-4. Measured before the fix on a constant score column (all 0.7,
        both labels, n=200): bin_values_ ==
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.365, 0.75, 0.85, 0.95] while the
        true per-bin training count was [0,0,0,0,0,0,200,0,0,0]; nine of ten
        values were the bin MIDPOINT, i.e. the identity map, and
        transform([0.05, 0.25, 0.95, 0.72]) -> [0.05, 0.25, 0.95, 0.75] handed
        the caller's own score back as a "calibrated probability" for three of
        four inputs. fit_result.warnings was [] and no count was recorded
        anywhere.
        """
        scores = np.full(200, 0.7)
        y = np.zeros(200, dtype=int)
        y[:73] = 1
        np.random.default_rng(0).shuffle(y)

        cal = HistogramBinning(n_bins=10, strategy="uniform")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)

        # The fixture exercises the unobserved-bin branch: exactly one bin holds
        # every row.
        assert cal.bin_counts_.tolist() == [0, 0, 0, 0, 0, 0, 200, 0, 0, 0]
        assert int(np.isnan(cal.bin_values_).sum()) == 9
        assert cal.bin_values_[6] == pytest.approx(0.365)
        # not the midpoints any more
        midpoints = (cal.bin_edges_[:-1] + cal.bin_edges_[1:]) / 2
        assert not np.allclose(np.nan_to_num(cal.bin_values_, nan=-1.0), midpoints, equal_nan=False)

        record = cal.fit_result.to_dict()
        assert record["parameters"]["bin_counts"] == [0, 0, 0, 0, 0, 0, 200, 0, 0, 0]
        assert record["parameters"]["n_bins_unmeasured"] == 9
        assert record["warnings"], "warnings == [] read as a clean fit"
        assert any("NO training sample" in w for w in record["warnings"])
        assert any("bins carry no usable rate" in str(w.message) for w in caught)

        # Measured after the fix: all four are NaN. 0.72 lands in bin 7 under
        # np.digitize (the edge is 0.7000000000000001, so the training 0.7 sits
        # in bin 6 and 0.72 does not), and bin 7 held nothing; before the fix it
        # came back as 0.75, that bin's midpoint.
        out = cal.transform(np.array([0.05, 0.25, 0.95, 0.72]))
        assert np.all(np.isnan(out)), out
        # The one bin that WAS observed still returns its measured rate.
        assert int(np.digitize(0.65, cal.bin_edges_[1:-1])) == 6
        assert cal.transform(np.array([0.65]))[0] == pytest.approx(0.365)

    def test_fit_transform_on_two_rows_exposes_the_counts_behind_its_certainties(self):
        """S2G04-5. Measured before the fix: fit_transform([0,1],[0.42,0.65]) ->
        [0. 1.] with warnings [], leaving bin_values_ ==
        [0.05 0.15 0.25 0.35 0. 0.55 1. 0.75 0.85 0.95]: two bins measured from
        ONE observation each, published as absolute 0.0 and 1.0 certainties, and
        eight unobserved-bin midpoints. A later transform over the bin grid was
        the identity for 8 of 10 inputs.
        """
        cal = HistogramBinning(n_bins=10, strategy="uniform")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(np.array([0, 1]), np.array([0.42, 0.65]))

        assert out.tolist() == [0.0, 1.0]  # the two rows ARE measured, from 1 row each
        assert cal.bin_counts_.tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
        assert int(np.isnan(cal.bin_values_).sum()) == 8

        record = cal.fit_result.to_dict()
        assert record["parameters"]["bin_counts"] == [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
        assert any("fewer than" in w for w in record["warnings"]), record["warnings"]
        assert any("NO training sample" in w for w in record["warnings"])
        assert caught, "a fit that can score 2 of 10 bins must not be silent"

        grid = cal.transform(np.arange(0.05, 1.0, 0.1))
        assert int(np.isnan(grid).sum()) == 8
        # the caller's own score is no longer returned as a calibrated value
        assert not np.allclose(
            np.nan_to_num(grid, nan=-1.0), np.arange(0.05, 1.0, 0.1), equal_nan=False
        )

    def test_min_bin_count_refuses_a_single_observation_certainty(self):
        """S2G04-5, the configurable half: min_bin_count=2 blanks the two
        single-observation bins rather than publishing 0.0 and 1.0."""
        out = HistogramBinning(n_bins=10, min_bin_count=2).fit_transform(
            np.array([0, 1]), np.array([0.42, 0.65])
        )
        assert np.all(np.isnan(out))

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "two"])
    def test_min_bin_count_is_validated(self, bad):
        with pytest.raises((TypeError, ValueError)):
            HistogramBinning(n_bins=10, min_bin_count=bad)

    def test_control_healthy_data_keeps_every_bin_measured_and_silent(self):
        """CONTROL. 2000 uniform scores fill all ten bins: no NaN, no warning,
        counts sum to n, and the map is not the identity."""
        rng = np.random.default_rng(0)
        scores = rng.random(2000)
        y = (rng.random(2000) < scores**2).astype(int)

        cal = HistogramBinning(n_bins=10, strategy="uniform")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert not caught, _no_warnings(caught)
        assert cal.fit_result.warnings == []
        assert int(np.isnan(cal.bin_values_).sum()) == 0
        assert int(cal.bin_counts_.sum()) == 2000
        assert cal.fit_result.parameters["n_bins_unmeasured"] == 0
        assert np.all(np.isfinite(out))
        assert not np.allclose(out, [0.1, 0.5, 0.9]), "calibrator became a no-op"


# ── S2G04-6: BetaCalibrator on a rank-deficient design ───────────────────────


class TestBetaUnidentifiedFit:
    def test_constant_score_column_is_not_fitted_from_the_initialisation(self):
        """S2G04-6. Measured before the fix (all 0.7, both labels, n=200):
        fit_result.to_dict() == {'method': 'beta_calibration_abm',
        'n_samples': 200, 'parameters': {'a': 0.0, 'b': 0.0, 'm': 0.0,
        'fitted_model': 'abm'}, 'fit_metrics': {}, 'warnings': []},
        is_fitted True, zero Python warnings, and transform([0.1, 0.5, 0.9]) ->
        [0.5 0.5 0.5]. Those were _newton_fit's np.zeros initialisation returned
        as the estimate.
        """
        y = np.zeros(200, dtype=int)
        y[:100] = 1
        np.random.default_rng(0).shuffle(y)
        scores = np.full(200, 0.7)
        # the fixture really is rank-deficient, tested on the RAW data
        assert np.unique(scores).size == 1

        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)

        params = cal.fit_result.to_dict()["parameters"]
        for key in ("a", "b", "m"):
            assert np.isnan(params[key]), params
            assert params[key] != 0.0
        assert params["design_rank"] == 1 and params["n_design_columns"] == 3
        assert cal.fit_result.warnings
        assert any("rank-deficient" in w for w in cal.fit_result.warnings)
        assert any("could-not-check" in str(w.message) for w in caught)

        out = cal.transform(np.array([0.1, 0.5, 0.9]))
        assert np.all(np.isnan(out))
        assert not np.any(out == 0.5)

    def test_three_parameters_on_two_rows_is_not_an_absolute_certainty(self):
        """S2G04-6, the other face. Measured before the fix on n=2:
        parameters {'a': 36.217, 'b': 35.058, 'm': -4.412}, warnings [], and
        transform([0.1, 0.5, 0.9]) -> [2.96e-37, 5.40e-03, 1.0]: near-absolute
        certainties extrapolated from two observations.
        """
        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.array([0, 1]), np.array([0.42, 0.65]))

        params = cal.fit_result.to_dict()["parameters"]
        assert np.isnan(params["a"]) and np.isnan(params["b"]) and np.isnan(params["m"])
        assert params["design_rank"] == 2 and params["n_design_columns"] == 3
        assert cal.fit_result.warnings
        assert caught
        out = cal.transform(np.array([0.1, 0.5, 0.9]))
        assert np.all(np.isnan(out))

    def test_control_healthy_data_still_fits_abm_silently(self):
        """CONTROL. Before the fix, n=400 healthy data fitted a=1.208/1.308,
        b=0.933/0.883, m=0.135/0.298 and transformed [0.1, 0.5, 0.9] to a
        strictly increasing finite triple. It must still do that, with
        warnings == [] and no Python warning.
        """
        y, scores = _healthy(400)
        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert not caught, _no_warnings(caught)
        assert cal.fit_result.warnings == []
        assert cal.fit_result.parameters["fitted_model"] == "abm"
        assert np.isfinite(cal.a_) and cal.a_ > 0
        assert np.isfinite(cal.b_) and cal.b_ > 0
        assert np.all(np.isfinite(out)) and np.all(np.diff(out) > 0)


# ── S2G04-7: create_calibrator, the durable record of a single-class fit ─────


class TestCreateCalibratorDurableRecord:
    @pytest.mark.parametrize("name", CALIBRATOR_NAMES)
    def test_a_single_class_fit_is_named_in_the_serialised_record(self, name):
        """S2G04-7. Measured before the fix (400 samples, all y == 1):
        platt/isotonic/temperature/histogram all recorded
        fit_result.to_dict()['warnings'] == [], and platt additionally recorded
        fit_metrics {'log_loss': 0.0025}, the best-looking number in the run,
        for a fit the library's own UserWarning calls unlearnable. In a
        per-group loop that one stderr line is emitted once for three degenerate
        groups and is gone by the time anything reads the results.
        """
        scores = np.random.default_rng(3).uniform(0.05, 0.95, 400)
        cal = create_calibrator(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.ones(400, dtype=int), scores)

        emitted = any("SINGLE class" in str(w.message) for w in caught)
        recorded = cal.fit_result.to_dict()["warnings"]
        assert emitted, "the live caution itself regressed"
        assert recorded, f"{name}: the durable record still asserts a clean fit"
        assert any("SINGLE class" in w for w in recorded)

    @pytest.mark.parametrize("name", CALIBRATOR_NAMES)
    def test_control_a_two_class_fit_records_no_warning(self, name):
        """CONTROL: healthy data must leave the record empty, or the field says
        nothing."""
        y, scores = _healthy(400)
        cal = create_calibrator(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert not caught, _no_warnings(caught)
        assert cal.fit_result.to_dict()["warnings"] == []
        assert np.all(np.isfinite(out))


# ── S2G04-8: IsotonicCalibrator's record of a map that learned nothing ───────


class TestIsotonicDegenerateMapIsDisclosed:
    def test_a_one_knot_map_does_not_serialise_like_a_healthy_fit(self):
        """S2G04-8. Measured before the fix on a constant score column (every
        score 0.7, n=60): fit_result.to_dict() == {"method":
        "isotonic_regression", "n_samples": 60, "parameters": {"n_blocks": 3,
        "x_range": [0.7, 0.7]}, "fit_metrics": {}, "warnings": []} with NO
        warning in any channel, while the fitted map had exactly ONE knot and
        transform([0.1, 0.5, 0.9]) -> [0.45 0.45 0.45]. n_blocks is the
        PRE-tie-collapse PAVA count, so the record described a resolution the
        map does not have.
        """
        scores = np.full(60, 0.7)
        y = (np.random.default_rng(3).uniform(0, 1, 60) < 0.45).astype(int)
        assert np.unique(scores).size == 1

        cal = IsotonicCalibrator().fit(y, scores)
        record = cal.fit_result.to_dict()

        assert len(cal.x_thresholds_) == 1
        assert record["parameters"]["n_knots"] == 1
        assert record["parameters"]["n_knots"] != record["parameters"]["n_blocks"]
        assert record["warnings"], "a constant map serialised as a clean fit"
        assert any("single knot" in w for w in record["warnings"])
        assert any("preserves no ranking" in w for w in record["warnings"])
        # the constant IS the measured base rate of the rows that were read,
        # so it is kept, not blanked: what was missing is the disclosure.
        assert np.allclose(cal.transform(np.array([0.1, 0.5, 0.9])), 0.45)

    def test_single_class_fit_is_recorded_and_not_only_warned(self):
        """S2G04-8 / S2G04-7. Before the fix the serialised record read
        "warnings": [] while the same call raised a UserWarning, and
        transform([0.1, 0.5, 0.9]) -> [1. 1. 1.]."""
        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.ones(60, dtype=int), np.random.default_rng(4).uniform(0, 1, 60))
        assert any("SINGLE class" in str(w.message) for w in caught)
        assert any("SINGLE class" in w for w in cal.fit_result.to_dict()["warnings"])

    def test_control_healthy_data_keeps_a_full_resolution_silent_fit(self):
        """CONTROL. Measured before the fix on healthy n=60: n_blocks 14 with
        60 knots. The knot count must now be reported and the record stay
        empty of warnings."""
        y, scores = _healthy(60, seed=0)
        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert not caught, _no_warnings(caught)
        record = cal.fit_result.to_dict()
        assert record["warnings"] == []
        assert record["parameters"]["n_knots"] == 60
        assert record["parameters"]["n_blocks"] < 60
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) >= 0)
