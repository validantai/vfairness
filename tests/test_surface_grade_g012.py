"""Surface grading batch g012: vfairness.post_processing.calibration.methods.

Subject: the five calibrators' fit/transform pair, BaseCalibrator.fit_transform
and CalibrationFitResult.to_dict. Every assertion is made at a PUBLIC entry
point, and every refusal assertion is paired with a healthy CONTROL whose
expected value is derived independently (by hand, or from the theory the method
implements) rather than copied from what the code returns, so a calibrator that
refused everything could not pass this file.

The before/after values quoted in the docstrings were measured on this working
tree on 2026-09-17 with .venv/bin/python.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from vfairness.exceptions import InvalidDataError
from vfairness.post_processing.calibration.methods import (
    BetaCalibrator,
    CalibrationFitResult,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)

ALL_CALIBRATORS = [
    PlattScaling,
    IsotonicCalibrator,
    BetaCalibrator,
    TemperatureScaling,
    HistogramBinning,
]


def _informative(n=60, seed=0):
    """Scores with many distinct values, both classes and a real signal."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(0.0, 1.0, n) < scores).astype(int)
    return y, scores


def _user_warnings(caught):
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


# ── BetaCalibrator: identifiability is a property of the WEIGHTED rows ───────


class TestBetaIdentifiabilityUsesTheWeightedRows:
    def test_all_zero_sample_weight_is_refused_not_answered_with_a_flat_half(self):
        """G012-1. `_is_identified` ranked the FULL design matrix, but every row
        enters the gradient and the Hessian multiplied by its weight, so a
        design of rank 3 over rows that carry no weight identifies nothing.

        Measured BEFORE this fix, 60 informative scores with every weight 0:
          record  {'method': 'beta_calibration_abm', 'n_samples': 60,
                   'parameters': {'a': 0.0, 'b': 0.0, 'm': 0.0,
                                  'fitted_model': 'abm'},
                   'fit_metrics': {}, 'warnings': []}
          Python warnings: none
          transform([0.1, 0.5, 0.9]) -> [0.5, 0.5, 0.5]
        i.e. the optimiser's own initialisation published as an estimate, and a
        flat one-half handed back as a calibrated probability.

        AFTER: a = b = m = NaN, one UserWarning, transform all NaN.
        """
        y, scores = _informative()
        weights = np.zeros(len(y))
        # The fixture must actually reach the branch under test.
        assert np.unique(scores).size > 2, "fixture scores must be informative"
        assert len(np.unique(y)) == 2, "fixture must not be refused as single-class"
        assert np.count_nonzero(weights) == 0

        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, weights)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        params = cal.fit_result.parameters
        assert np.isnan(params["a"]) and np.isnan(params["b"]) and np.isnan(params["m"])
        assert params["n_weighted_rows"] == 0
        assert np.all(np.isnan(out)), f"expected NaN, got {out}"
        assert not np.allclose(np.nan_to_num(out, nan=-1.0), 0.5)
        msgs = _user_warnings(caught)
        assert any("identified NO beta-calibration map" in m for m in msgs), msgs
        assert any("identified NO beta-calibration map" in n for n in cal.fit_result.warnings)

    def test_one_weighted_row_out_of_sixty_is_refused(self):
        """G012-2. Measured BEFORE this fix, 60 informative scores with exactly
        one weight of 1.0 and 59 of 0.0:
          parameters {'a': 5.711096880091495, 'b': 0.0,
                      'm': -1.7161001914443859, 'fitted_model': 'am'}
          transform([0.1, 0.5, 0.9]) -> [0.0, 0.003394, 0.089675]
          Python warnings: none; the two fit_result notes named the family
          clamp and max_iter, neither of which says the map came off ONE row,
          while n_samples read 60.
        """
        y, scores = _informative()
        weights = np.zeros(len(y))
        weights[3] = 1.0

        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, weights)
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert np.all(np.isnan(out)), f"expected NaN, got {out}"
        assert cal.fit_result.parameters["n_weighted_rows"] == 1
        assert cal.fit_result.parameters["design_rank"] == 1
        msgs = _user_warnings(caught)
        assert any("identified NO beta-calibration map" in m for m in msgs), msgs
        assert any("1 of 60" in n for n in cal.fit_result.warnings), cal.fit_result.warnings

    def test_control_the_same_scores_at_unit_weight_still_fit_a_real_map(self):
        """G012-3 CONTROL. The refusal above must be caused by the WEIGHTS, not
        by the scores: identical y and scores with every weight 1.0 must still
        produce a finite, strictly increasing, silent fit.
        """
        y, scores = _informative()
        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, np.ones(len(y)))
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        params = cal.fit_result.parameters
        assert np.isfinite(params["a"]) and np.isfinite(params["b"])
        assert np.all(np.isfinite(out)), out
        assert out[0] < out[1] < out[2], f"map is not increasing: {out}"
        assert _user_warnings(caught) == []

    def test_control_beta_recovers_the_identity_map_on_calibrated_data(self):
        """G012-4 CONTROL, independent expectation. Beta calibration is
        sigmoid(a*ln p - b*ln(1-p) + m); at a=1, b=1, m=0 that is exactly the
        identity. Labels drawn as Bernoulli(p) from the scores themselves are
        therefore already calibrated, so the maximum-likelihood fit must sit at
        (1, 1, 0) and the map must return its input.

        This expectation comes from the model definition, not from running the
        code, and it is what makes "refuses everything" fail this file.
        """
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(0.0, 1.0, 20000) < p).astype(int)

        cal = BetaCalibrator()
        cal.fit(y, p)
        params = cal.fit_result.parameters
        assert 0.85 <= params["a"] <= 1.15, params
        assert 0.85 <= params["b"] <= 1.15, params
        assert abs(params["m"]) <= 0.15, params
        for probe in (0.2, 0.5, 0.8):
            got = float(cal.transform(np.array([probe]))[0])
            assert abs(got - probe) < 0.03, f"transform({probe}) = {got}"


# ── IsotonicCalibrator: a weightless row cannot enter a weighted PAVA ────────


class TestIsotonicWeightlessRows:
    def test_all_zero_sample_weight_is_refused_and_says_so_on_both_channels(self):
        """G012-5. Measured BEFORE this fix, 60 informative scores with every
        weight 0: PAVA divided 0 by 0 in every merge, 53 of the 60 knot values
        came back NaN, and fit() published

          {'n_blocks': 49, 'n_knots': 60, 'n_distinct_values': 2,
           'x_range': (0.02262896016334217, 0.9773215383576426)}

        with warnings: [] and NOT ONE UserWarning (only 11 numpy
        "invalid value encountered in scalar divide" RuntimeWarnings). The two
        "distinct values" were 0.0 and NaN, so the field whose whole job is to
        say "this map is flat" reported variation that does not exist, in a
        record that serialises byte-identically to a healthy one.

        AFTER: n_distinct_values 0, n_weighted_rows 0, one UserWarning, and the
        note in fit_result.warnings.
        """
        y, scores = _informative()
        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, np.zeros(len(y)))
            out = cal.transform(np.array([0.1, 0.5, 0.9]))

        params = cal.fit_result.parameters
        assert params["n_distinct_values"] == 0, params
        assert params["n_weighted_rows"] == 0, params
        assert np.all(np.isnan(out)), out
        msgs = _user_warnings(caught)
        assert any("identified NO isotonic map" in m for m in msgs), msgs
        assert any("identified NO isotonic map" in n for n in cal.fit_result.warnings)
        # The numpy 0/0 chatter must be gone: it was the only signal there was,
        # and it is not a statement a caller can read.
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    def test_zero_weight_rows_are_excluded_and_the_exclusion_is_recorded(self):
        """G012-6. A partially weightless fit is still measurable, so it must be
        MEASURED, with the row count it was measured from published rather than
        the row count it was handed.
        """
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.7])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, weights)

        params = cal.fit_result.parameters
        assert cal.fit_result.n_samples == 6
        assert params["n_weighted_rows"] == 4, params
        assert params["n_knots"] == 4, params
        assert any("carry sample_weight 0" in n for n in cal.fit_result.warnings)
        # Independent expectation: the four weighted rows are (0.1,0), (0.2,0),
        # (0.3,1), (0.4,1). Already monotone, so PAVA pools nothing and each
        # point keeps its own label. Interpolating at 0.25 sits halfway between
        # the 0.2 knot (0.0) and the 0.3 knot (1.0).
        assert float(cal.transform(np.array([0.25]))[0]) == pytest.approx(0.5)
        assert float(cal.transform(np.array([0.1]))[0]) == pytest.approx(0.0)
        assert float(cal.transform(np.array([0.4]))[0]) == pytest.approx(1.0)
        assert _user_warnings(caught) == []

    def test_control_a_healthy_isotonic_fit_is_silent_and_exact(self):
        """G012-7 CONTROL, hand-computed. y = [0, 0, 1, 1] at scores
        [0.1, 0.2, 0.3, 0.4] is already monotone, so the PAVA solution is the
        labels themselves and linear interpolation at 0.25 is exactly 0.5.
        """
        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]))
            out = cal.transform(np.array([0.1, 0.25, 0.4]))

        assert out == pytest.approx([0.0, 0.5, 1.0])
        assert cal.fit_result.parameters["n_distinct_values"] == 2
        assert cal.fit_result.parameters["n_weighted_rows"] == 4
        assert cal.fit_result.warnings == []
        assert _user_warnings(caught) == []


# ── transform(): a number that is not a probability is not a score ───────────


class TestTransformRefusesNonProbabilities:
    @pytest.mark.parametrize("cls", ALL_CALIBRATORS)
    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(np.array([1.7, 2.0, 42.0]), id="above_one"),
            pytest.param(np.array([-3.0, -0.2, -1.0]), id="below_zero"),
            pytest.param(np.array([np.nan, np.nan, np.nan]), id="nan"),
        ],
    )
    def test_out_of_range_and_unscoreable_input_is_refused(self, cls, bad):
        """G012-8. HistogramBinning.transform and PlattScaling.transform already
        refused this; IsotonicCalibrator, BetaCalibrator and TemperatureScaling
        clipped it into range and published an absolute certainty.

        Measured BEFORE this fix, each fitted on 60 informative rows:
          IsotonicCalibrator.transform([1.7, 2.0, 42.0])   -> [1.0, 1.0, 1.0]
          BetaCalibrator.transform([1.7, 2.0, 42.0])       -> [1.0, 1.0, 1.0]
          TemperatureScaling.transform([1.7, 2.0, 42.0])   -> [1.0, 1.0, 1.0]
          the same three on [-3.0, -0.2, -1.0]             -> [0.0, 0.0, 0.0]
        with zero warnings on every one: a calibrated certainty read off an
        input that is not a probability at all.
        """
        y, scores = _informative()
        cal = cls()
        cal.fit(y, scores)
        with pytest.raises(InvalidDataError):
            cal.transform(bad)

    @pytest.mark.parametrize("cls", ALL_CALIBRATORS)
    def test_control_in_range_input_is_still_measured(self, cls):
        """G012-9 CONTROL. The refusal above must be about the INPUT RANGE and
        nothing else: legal probabilities, including the exact endpoints 0.0 and
        1.0, must still come back as finite calibrated values.
        """
        y, scores = _informative()
        cal = cls()
        cal.fit(y, scores)
        out = cal.transform(np.array([0.0, 0.3, 0.7, 1.0]))
        assert np.all(np.isfinite(out)), f"{cls.__name__} returned {out}"
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert np.unique(out).size >= 2, f"{cls.__name__} map is flat: {out}"

    def test_control_temperature_recovers_a_known_overconfidence(self):
        """G012-10 CONTROL, independent expectation. Scores built as
        sigmoid(2 * logit(p)) are exactly p over-sharpened by a temperature of
        2, so the temperature that minimises the NLL against labels drawn as
        Bernoulli(p) is 2. The expectation comes from how the fixture was
        constructed, not from the fitted value.
        """
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(0.0, 1.0, 20000) < p).astype(int)
        over_confident = 1.0 / (1.0 + np.exp(-2.0 * np.log(p / (1 - p))))

        cal = TemperatureScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, over_confident)

        assert cal.fit_result.parameters["temperature"] == pytest.approx(2.0, abs=0.15)
        assert cal.fit_result.warnings == []
        assert _user_warnings(caught) == []


# ── fit_transform(): the one-step path must carry the same three states ──────


class TestFitTransformCarriesTheRefusal:
    def test_fit_transform_returns_nan_where_fit_refused(self):
        """G012-11. fit_transform is the path a caller reaches for when it does
        not intend to read fit_result at all, so a refusal that only lives in
        fit_result.warnings is invisible on it. BEFORE the BetaCalibrator fix,
        BetaCalibrator().fit_transform(y, scores, zeros) returned a flat array
        of 0.5 with no warning of any kind.
        """
        y, scores = _informative()
        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores, np.zeros(len(y)))

        assert np.all(np.isnan(out)), out
        assert any("identified NO beta-calibration map" in m for m in _user_warnings(caught))

    @pytest.mark.parametrize("cls", ALL_CALIBRATORS)
    def test_control_fit_transform_equals_fit_then_transform(self, cls):
        """G012-12 CONTROL. On healthy data fit_transform must be exactly
        fit() followed by transform() on the same scores, and must produce a
        map with real resolution rather than a constant.
        """
        y, scores = _informative()
        one = cls().fit_transform(y, scores)
        other = cls()
        other.fit(y, scores)
        two = other.transform(scores)
        np.testing.assert_array_equal(one, two)
        assert np.all(np.isfinite(one))
        assert np.unique(one).size >= 2, f"{cls.__name__} produced a constant map"


# ── CalibrationFitResult.to_dict(): the record a reader grades ───────────────


class TestCalibrationFitResultToDict:
    def test_to_dict_carries_every_declared_field(self):
        """G012-13. to_dict is a field-by-field rebuild, which is the shape that
        silently drops whatever field is added after it is written. Pin it
        against the dataclass itself rather than against a list of names, so a
        new field that to_dict forgets fails here instead of disappearing from
        the only record a consumer serialises.
        """
        declared = [f.name for f in dataclasses.fields(CalibrationFitResult)]
        result = CalibrationFitResult(
            method="platt_scaling",
            n_samples=7,
            parameters={"a": 1.5, "b": -0.25},
            fit_metrics={"log_loss": 0.5},
            warnings=["a note"],
        )
        as_dict = result.to_dict()
        assert set(as_dict) == set(declared), (f"to_dict dropped {set(declared) - set(as_dict)}",)
        for name in declared:
            assert as_dict[name] == getattr(result, name), name

    def test_to_dict_carries_the_could_not_check_state_through(self):
        """G012-14. The refusal has to survive the conversion a consumer
        actually serialises, not only the live object.
        """
        y, scores = _informative()
        cal = BetaCalibrator()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cal.fit(y, scores, np.zeros(len(y)))

        as_dict = cal.fit_result.to_dict()
        assert np.isnan(as_dict["parameters"]["a"])
        assert any("identified NO beta-calibration map" in n for n in as_dict["warnings"])

    def test_control_to_dict_carries_a_real_measurement_through(self):
        """G012-15 CONTROL. A healthy fit must serialise a finite parameter and
        an empty warnings list, so a to_dict that emptied everything could not
        pass the pin above.
        """
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(0.0, 1.0, 20000) < p).astype(int)
        cal = BetaCalibrator()
        cal.fit(y, p)
        as_dict = cal.fit_result.to_dict()
        assert np.isfinite(as_dict["parameters"]["a"])
        assert np.isfinite(as_dict["parameters"]["b"])
        assert as_dict["warnings"] == []
        assert as_dict["n_samples"] == 20000
