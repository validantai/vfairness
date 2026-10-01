"""Beta Go-Live stage 2b, group s2b08: the calibration methods must not publish
a neutral default, a search boundary or a collapsed bin grid as if it were a
measurement.

Subject: src/vfairness/post_processing/calibration/methods.py, all five
calibrators plus the create_calibrator factory. Every assertion below is made
at a PUBLIC entry point (fit / fit_transform / transform / create_calibrator),
never against a private helper, and every degenerate case is paired with a
healthy CONTROL in the same class so a guard that refuses everything cannot
pass.

The before/after values quoted in each docstring were measured on this working
tree on 2026-09-17 with .venv/bin/python, not inferred from the source.
"""

import warnings

import numpy as np
import pytest

from vfairness.post_processing.calibration.methods import (
    BetaCalibrator,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    create_calibrator,
)


def _healthy(n=400, seed=0):
    """Informative scores: many distinct values, both classes, real signal."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(0.0, 1.0, n) < scores).astype(int)
    return y, scores


def _messages(caught):
    return [str(w.message) for w in caught]


def _joined(notes):
    return " || ".join(notes)


# ── PlattScaling: the identifiability test must be scale-invariant ───────────


class TestPlattConstantColumnAtEveryScale:
    @pytest.mark.parametrize("constant", [0.0, 0.001, 0.01, 0.5, 0.7, 1.0])
    def test_a_constant_score_column_is_refused_whatever_its_value(self, constant):
        """S2B08-1. The old test was `abs(det) < 1e-12`, an ABSOLUTE floor on a
        quantity that scales with the data, so it happened to fire at 0.5 and
        0.7 (the two values the existing fixtures use) and not elsewhere.

        Measured BEFORE this fix, 60 rows, base rate 0.3:
          constant 0.01  -> a=0.33125,  b=0.425,    iterations 3, warnings
                            ["the Newton system became singular after 2 step(s)"],
                            transform([0.01, 0.5, 0.99]) = [0.2503, 0.6047, 0.8751]
          constant 0.001 -> a=0.130208, b=0.0572917, transform [0.3679, 0.5143, 0.6583]
          constant 1.0   -> a=-0.061261 (ANTI-MONOTONE), transform
                            [0.5525, 0.4823, 0.4128]: a HIGHER score maps to a
                            LOWER calibrated probability
          constant 0.0   -> a=-0.0519,  b=-2.1181,  transform [0.1324, 0.1073, 0.0865]
        all four with ZERO Python warnings, off a column with exactly one
        distinct value.

        AFTER: a=b=NaN, transform all NaN, iterations 0, one UserWarning.
        """
        n = 60
        scores = np.full(n, constant)
        y = (np.arange(n) < 18).astype(int)
        # The fixture must actually reach the branch under test.
        assert np.unique(scores).size == 1, "fixture is not a constant column"
        assert len(np.unique(y)) == 2, "fixture must not be refused as single-class"

        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert np.isnan(cal.a_) and np.isnan(cal.b_), (cal.a_, cal.b_)
        assert np.all(np.isnan(out)), np.unique(out)
        assert np.isnan(record["fit_metrics"]["log_loss"])
        assert record["fit_metrics"]["iterations"] == 0
        assert record["warnings"], "warnings == [] reads as a clean fit"
        assert "rank-deficient" in _joined(record["warnings"])
        assert "1 distinct value(s)" in _joined(record["warnings"])
        assert any("could-not-check" in m for m in _messages(caught)), _messages(caught)

    def test_control_healthy_scores_still_fit_a_real_monotone_map(self):
        """CONTROL. A guard that refuses every input passes every degenerate
        test above and is useless. Measured after the fix: a=0.938, b=-0.0399,
        log_loss 0.5638, iterations 5, warnings [], transform([0.1, 0.5, 0.9])
        = [0.1240, 0.4900, 0.8843]."""
        y, scores = _healthy(400, seed=1)
        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert record["warnings"] == [], record["warnings"]
        assert not caught, _messages(caught)
        assert np.isfinite(cal.a_) and np.isfinite(cal.b_)
        assert cal.a_ > 0.0, "a real Platt map on signal-carrying scores is increasing"
        assert record["fit_metrics"]["iterations"] >= 1
        assert np.all(np.isfinite(out))
        probe = cal.transform(np.array([0.1, 0.5, 0.9]))
        assert np.all(np.diff(probe) > 0), probe

    def test_max_iter_zero_refuses_instead_of_publishing_its_initialisation(self):
        """S2B08-2. The guard used to be `singular_hessian and n_steps == 0`,
        so a loop that never ran for ANY OTHER reason still published its own
        starting values. Measured BEFORE, PlattScaling(max_iter=0) on 200
        HEALTHY scores: parameters {'a': 0.0, 'b': 0.0}, fit_metrics
        {'log_loss': 0.6931471805599452 (ln 2), 'iterations': 1}, warnings
        ["the Newton loop hit max_iter (0) ... the reported a and b are the last
        iterate"] (there is no iterate), zero Python warnings, and
        transform([0.1, 0.5, 0.9]) -> [0.5, 0.5, 0.5].

        AFTER: a=b=NaN, log_loss NaN, iterations 0, one UserWarning, and the
        note names max_iter as the cause rather than claiming the scores carry
        no variation.
        """
        y, scores = _healthy(200, seed=2)
        assert np.unique(scores).size > 2, "fixture must be a HEALTHY column"

        cal = PlattScaling(max_iter=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert np.isnan(cal.a_) and np.isnan(cal.b_)
        assert np.all(np.isnan(out))
        assert not np.any(out == 0.5)
        assert np.isnan(record["fit_metrics"]["log_loss"])
        assert record["fit_metrics"]["log_loss"] != pytest.approx(np.log(2))
        assert record["fit_metrics"]["iterations"] == 0, "a loop body that ran zero times"
        assert "max_iter=0 allowed no Newton step" in _joined(record["warnings"])
        assert "last iterate" not in _joined(record["warnings"])
        assert any("could-not-check" in m for m in _messages(caught))

    def test_small_weights_are_measured_not_refused_and_the_note_is_true(self):
        """S2B08-3, the OVER-CORRECTION half. The absolute determinant floor
        also fired on the UNIT of the weights: measured BEFORE, 60 INFORMATIVE
        scores with sample_weight=1e-8 were refused as NaN with the note "the
        60 scores carry no usable variation", which is false, since the same
        scores at weight 1.0 fit normally.

        AFTER: the fit runs (a=9.55e-08, b=-1.58e-07, iterations 1) and the
        record explains the real cause, that Platt's smoothing targets collapse
        when the weighted class counts are below 1.
        """
        y, scores = _healthy(60, seed=3)
        assert np.unique(scores).size > 2

        cal = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores, sample_weight=np.full(60, 1e-8))

        record = cal.fit_result.to_dict()
        assert np.isfinite(cal.a_) and np.isfinite(cal.b_), "a measurable fit was refused"
        assert "rank-deficient" not in _joined(record["warnings"])
        assert "no usable variation" not in _joined(record["warnings"])
        assert "smoothing targets" in _joined(record["warnings"])
        assert not any("could-not-check" in m for m in _messages(caught))

        # CONTROL for the same input at a normal weight scale: same scores,
        # weight 1.0, no note at all.
        ok = PlattScaling().fit(y, scores, sample_weight=np.ones(60))
        assert ok.fit_result.warnings == [], ok.fit_result.warnings
        assert np.isfinite(ok.a_) and ok.a_ > 0.0


# ── TemperatureScaling: one parameter that only rescales ─────────────────────


class TestTemperatureDegeneracy:
    @pytest.mark.parametrize("base_rate", [0.5, 0.1])
    def test_a_constant_score_column_is_refused(self, base_rate):
        """S2B08-4. Measured BEFORE, 200 rows all scored 0.7:
          base rate 0.50 -> temperature 99.9999975 (the 100.0 search bound),
                            nll 0.6931562, warnings [], no Python warning,
                            transform([0.1, 0.5, 0.9]) = [0.4945, 0.5, 0.5055]
          base rate 0.10 -> temperature 99.9999979 and the SAME
                            [0.4945, 0.5, 0.5055]
        i.e. the published calibration did not depend on the outcomes at all,
        and the only tell was a T pinned to its own boundary, which nothing
        reported. AFTER: temperature_ NaN, nll NaN, transform NaN, one
        UserWarning, durable note.
        """
        n = 200
        scores = np.full(n, 0.7)
        y = (np.arange(n) < int(n * base_rate)).astype(int)
        assert np.unique(scores).size == 1
        assert len(np.unique(y)) == 2

        cal = TemperatureScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert np.isnan(cal.temperature_)
        assert np.all(np.isnan(out))
        assert record["parameters"]["temperature"] != pytest.approx(100.0, abs=1e-3)
        assert np.isnan(record["fit_metrics"]["nll"])
        assert record["warnings"], "warnings == [] reads as a clean fit"
        assert "rank-deficient" in _joined(record["warnings"])
        assert any("could-not-check" in m for m in _messages(caught))

    def test_a_bound_pinned_optimum_says_so(self):
        """S2B08-5. Measured BEFORE on two rows with distinct scores (perfectly
        separable): temperature 0.014951, nll 8.34e-13, warnings [], and
        transform([0.1, 0.5, 0.9]) -> [1.49e-64, 0.5, 1.0], near-absolute
        certainties extrapolated from two observations. 0.014951 is 1.5x the
        0.01 floor, so a proximity test on T would not catch it; the objective
        is compared at the bounds instead. AFTER: the same T, plus a durable
        note that it is pinned by the bound rather than fitted.
        """
        cal = TemperatureScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(np.array([0, 1]), np.array([0.3, 0.6]))

        notes = _joined(cal.fit_result.warnings)
        assert "NOT interior" in notes, cal.fit_result.warnings
        assert "pinned by the bound" in notes
        assert not any("could-not-check" in m for m in _messages(caught)), (
            "a bound-pinned fit is a caution, not a refusal: T is still a number"
        )

    def test_control_healthy_miscalibrated_scores_fit_an_interior_temperature(self):
        """CONTROL. Measured after the fix on 400 over-confident scores:
        temperature 1.474, nll 0.5623, warnings [], transform monotone."""
        rng = np.random.default_rng(7)
        true_p = rng.uniform(0.05, 0.95, 400)
        y = (rng.uniform(0, 1, 400) < true_p).astype(int)
        # push the scores away from the outcome rate so T != 1 is the answer
        logits = np.log(true_p / (1 - true_p)) * 1.8
        scores = 1 / (1 + np.exp(-logits))

        cal = TemperatureScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        assert cal.fit_result.warnings == [], cal.fit_result.warnings
        assert not caught, _messages(caught)
        assert np.isfinite(cal.temperature_)
        assert 0.02 < cal.temperature_ < 99.0, cal.temperature_
        assert np.all(np.isfinite(out))
        probe = cal.transform(np.array([0.1, 0.5, 0.9]))
        assert np.all(np.diff(probe) > 0), probe


# ── HistogramBinning: the bin GRID itself can be degenerate ──────────────────


class TestHistogramGridDegeneracy:
    def test_quantile_on_a_constant_column_refuses_at_fit_instead_of_crashing(self):
        """S2B08-6. Measured BEFORE, strategy='quantile' on 100 rows all scored
        0.7: np.percentile returns one repeated edge, np.unique collapses it,
        and bin_edges_[0]=0 / bin_edges_[-1]=1 then write both to that single
        element. fit() recorded parameters {'n_bins': 0, 'bin_counts': [],
        'n_bins_unmeasured': 0} with warnings [] and no Python warning, a
        positive claim of complete measurement for a fit that measured nothing,
        and transform() then died with "IndexError: index -1 is out of bounds
        for axis 0 with size 0".

        AFTER: fit states the refusal on both channels and transform returns
        NaN.
        """
        scores = np.full(100, 0.7)
        y = (np.arange(100) < 30).astype(int)
        assert np.unique(scores).size == 1

        cal = HistogramBinning(strategy="quantile")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)

        record = cal.fit_result.to_dict()
        assert record["parameters"]["n_bins"] == 0
        assert record["parameters"]["n_bins_requested"] == 10
        assert record["warnings"], "warnings == [] reads as a complete measurement"
        assert "built NO bin" in _joined(record["warnings"])
        assert any("could-not-check" in m for m in _messages(caught))

        out = cal.transform(np.array([0.1, 0.7, 0.95]))
        assert out.shape == (3,)
        assert np.all(np.isnan(out)), out

    def test_a_grid_that_collapses_to_one_bin_is_disclosed(self):
        """S2B08-7. Measured BEFORE, strategy='quantile' on a skewed two-value
        column (950 rows at 0.2 with a true rate of 0.101, 50 rows at 0.8 with
        a true rate of 0.86): the 10 requested bins collapse to ONE,
        bin_values_ = [0.139], warnings [], n_bins_unmeasured 0, no Python
        warning, and transform is flat at 0.139, so the 0.86 group and the
        0.101 group leave with the same calibrated probability.

        AFTER: the same (measured) 0.139 is kept, and the record says the grid
        requested 10 bins and produced 1.
        """
        scores = np.concatenate([np.full(950, 0.2), np.full(50, 0.8)])
        y = np.concatenate([(np.arange(950) < 96).astype(int), (np.arange(50) < 43).astype(int)])
        cal = HistogramBinning(n_bins=10, strategy="quantile")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert record["parameters"]["n_bins"] == 1
        assert record["parameters"]["n_bins_requested"] == 10
        assert record["warnings"], "a one-bin grid serialised as a clean fit"
        assert "requested 10 bins and produced 1" in _joined(record["warnings"])
        assert "preserves no ranking" in _joined(record["warnings"])
        assert any("produced 1" in m for m in _messages(caught))
        # The value IS the measured base rate of the rows that were read, so it
        # is kept rather than blanked: what was missing is the disclosure.
        assert np.ptp(out) == 0.0
        assert float(out[0]) == pytest.approx(0.139)

    def test_control_a_healthy_grid_keeps_its_resolution_and_stays_silent(self):
        """CONTROL. Measured after the fix on 2000 uniform scores with 10
        uniform bins: n_bins 10, n_bins_requested 10, warnings [], every bin
        observed, transform finite and non-constant."""
        y, scores = _healthy(2000, seed=4)
        cal = HistogramBinning(n_bins=10, strategy="uniform")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert record["warnings"] == [], record["warnings"]
        assert not caught, _messages(caught)
        assert record["parameters"]["n_bins"] == 10
        assert record["parameters"]["n_bins_requested"] == 10
        assert record["parameters"]["n_bins_unmeasured"] == 0
        assert np.all(np.isfinite(out))
        assert np.ptp(out) > 0.2, "a healthy fit must keep resolution"

    @pytest.mark.parametrize(
        "bad,exc,match",
        [
            (-1, ValueError, "n_bins must be >= 1"),
            (0, ValueError, "n_bins must be >= 1"),
            ("ten", TypeError, "n_bins must be an int"),
            (2.5, TypeError, "n_bins must be an int"),
        ],
    )
    def test_n_bins_is_validated_like_min_bin_count(self, bad, exc, match):
        """S2B08-8. Measured BEFORE: n_bins=-1 surfaced as numpy's "negative
        dimensions are not allowed" from inside np.linspace, and n_bins='ten'
        as "can only concatenate str (not "int") to str", neither naming the
        argument nor the class."""
        with pytest.raises(exc, match=match):
            HistogramBinning(n_bins=bad)

    def test_a_non_finite_weight_sum_is_not_reported_as_a_zero_sum(self):
        """S2B08-9. Measured BEFORE: a bin whose weight sum is not finite got
        the note "sample_weight summed to 0", sending a reader looking for zero
        weights that are not there. The NaN calibrated value was right; only
        the explanation was wrong.

        The fixture is FINITE weights whose sum overflows, because a NaN weight
        is now refused outright at the shared gate (S2B08-15). That route into
        a non-finite sum survives the gate, so the note still has to be right.
        """
        y, scores = _healthy(50, seed=5)
        sw = np.full(50, 1e308)
        assert np.all(np.isfinite(sw)), "the fixture must pass the input gate"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # numpy's own overflow RuntimeWarning
            cal = HistogramBinning(n_bins=5).fit(y, scores, sample_weight=sw)
        notes = _joined(cal.fit_result.warnings)
        assert "sum is not finite" in notes, cal.fit_result.warnings
        assert "summed to 0" not in notes

        # CONTROL, the other cause: genuinely zero weights still say "summed
        # to 0", so the two causes stay distinguishable.
        sw0 = np.ones(50)
        sw0[scores < 0.2] = 0.0
        assert np.any(sw0 == 0.0)
        cal0 = HistogramBinning(n_bins=5).fit(y, scores, sample_weight=sw0)
        notes0 = _joined(cal0.fit_result.warnings)
        assert "summed to 0" in notes0, cal0.fit_result.warnings
        assert "sum is not finite" not in notes0


# ── IsotonicCalibrator: the Y axis can be degenerate while X is not ──────────


class TestIsotonicFlatMap:
    def test_a_pava_pooled_flat_map_is_disclosed(self):
        """S2B08-10. The existing degeneracy test covered only the X axis
        (n_knots < 2). Measured BEFORE on plain no-signal data (n=40, seed 2,
        uniform scores, labels from a fair coin): parameters {'n_blocks': 1,
        'n_knots': 40, 'x_range': (0.0551, 0.9674)}, warnings [], no Python
        warning, and transform([0.1, 0.5, 0.9]) -> [0.425, 0.425, 0.425]. That
        record is byte-identical in shape to a healthy one on every published
        field, and it happens in roughly 5% of no-signal fits at this n."""
        rng = np.random.default_rng(2)
        scores = rng.random(40)
        y = (rng.random(40) < 0.5).astype(int)

        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        # The fixture must be the flat-map case and NOT the single-knot case.
        assert record["parameters"]["n_knots"] == 40, "fixture is not the y-axis case"
        assert float(np.ptp(cal.y_values_)) == 0.0, "fixture map is not flat"
        assert record["parameters"]["n_distinct_values"] == 1
        assert record["warnings"], "a flat map serialised as a clean fit"
        assert "FLAT" in _joined(record["warnings"])
        assert "preserves no ranking" in _joined(record["warnings"])
        assert any("FLAT" in m for m in _messages(caught))
        assert np.ptp(out) == 0.0

    def test_a_constant_score_column_warns_on_both_channels(self):
        """S2B08-11. The single-knot note existed but lived only in
        fit_result.warnings, unlike the single-class path in
        _validate_fit_inputs, which also raises a UserWarning. A caller who
        reads the transformed array and nothing else saw nothing."""
        scores = np.full(60, 0.7)
        y = (np.random.default_rng(3).uniform(0, 1, 60) < 0.45).astype(int)

        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)

        assert "single knot" in _joined(cal.fit_result.warnings)
        assert any("single knot" in m for m in _messages(caught)), _messages(caught)

    def test_control_healthy_data_keeps_a_varying_map_and_stays_silent(self):
        """CONTROL, strengthened. A control that only asserts n_blocks < n is
        also satisfied by the flat map (n_blocks == 1), so it cannot separate
        the two; the spread of the OUTPUT is what distinguishes them."""
        y, scores = _healthy(400, seed=6)
        cal = IsotonicCalibrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        record = cal.fit_result.to_dict()
        assert record["warnings"] == [], record["warnings"]
        assert not caught, _messages(caught)
        assert float(np.ptp(cal.y_values_)) > 0.0
        assert record["parameters"]["n_distinct_values"] > 1
        assert 1 < record["parameters"]["n_blocks"] < 400
        assert np.ptp(out) > 0.2


# ── create_calibrator: the factory is its own ledger row ─────────────────────


class TestFactoryOverAllFiveMethods:
    ALL = ["platt", "isotonic", "beta", "temperature", "histogram"]

    @pytest.mark.parametrize("method", ALL)
    def test_no_method_publishes_a_clean_record_for_a_constant_column(self, method):
        """S2B08-12. One assertion over every method the factory can build: a
        score column with a single distinct value must never come back with
        warnings == []. Before this campaign all five did."""
        scores = np.full(120, 0.7)
        y = (np.arange(120) < 36).astype(int)
        assert np.unique(scores).size == 1

        cal = create_calibrator(method)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y, scores)

        assert cal.fit_result.warnings, f"{method} recorded a clean fit for a constant column"
        assert caught, f"{method} emitted no Python warning"

    @pytest.mark.parametrize("method", ALL)
    def test_control_every_method_still_measures_healthy_data(self, method):
        """CONTROL for the parametrization above: the same five methods on
        informative scores must produce a finite, varying calibration with an
        empty record."""
        y, scores = _healthy(600, seed=8)
        cal = create_calibrator(method)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.fit_transform(y, scores)

        assert cal.fit_result.warnings == [], (method, cal.fit_result.warnings)
        assert not caught, (method, _messages(caught))
        assert np.all(np.isfinite(out)), method
        assert np.ptp(out) > 0.2, (method, np.ptp(out))


# ── The composite entry, which lives in a file this group does not own ───────


def test_group_calibration_must_not_turn_one_unmeasurable_group_into_a_verdict():
    """S2B08-13, BLOCKED, pinned as a PROPERTY so it is green under both the
    current state and any correct fix.

    GroupCalibrator.fit() builds one calibrator per group and then feeds the
    concatenated output to expected_calibration_error. When one group's scores
    are constant, PlattScaling now correctly returns NaN for that group, and
    the composite call raises InvalidDataError("y_prob contains 100 NaN
    value(s)"), which destroys the OTHER groups' genuine measurements in the
    same call and blames the caller's y_prob for a NaN the library itself
    produced.

    The fix belongs in
    src/vfairness/post_processing/calibration/group_calibrator.py line 403
    (record that group as could-not-check, keep reporting the rest). That file
    is not in this group's assigned file list and is owned by no group in this
    round, so it is reported as blocked rather than edited in a shared
    checkout.

    What must NEVER be true, in either state, is the third possibility: the
    degenerate group coming back with a clean, best-looking number. That is
    what this pins.
    """
    from vfairness.evaluation.vfairness_metrics._validation import InvalidDataError
    from vfairness.post_processing.calibration.group_calibrator import GroupCalibrator

    rng = np.random.default_rng(3)
    groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    scores = np.concatenate([rng.random(100), rng.random(100), np.full(100, 0.7)])
    y = (rng.random(300) < np.clip(scores, 0.05, 0.95)).astype(int)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc = GroupCalibrator(method="platt").fit(y, scores, groups)
    except InvalidDataError as exc:
        # CURRENT, BLOCKED state. Assert it is the documented failure and not
        # some other error, so this pin still means something.
        assert "NaN" in str(exc)
        return

    # If group_calibrator.py is fixed, these are the properties the fix owes.
    result = gc.fit_result_
    post = result.post_calibration_ece
    assert not (post.get("C") == 0.0), (
        "the group whose calibration could not be measured came back with the "
        "best-looking ECE in the table"
    )
    assert np.isnan(post.get("C", np.nan)) or "C" not in post
    assert np.isfinite(post.get("A", np.nan)), "a healthy group lost its measurement"
    assert np.isfinite(post.get("B", np.nan)), "a healthy group lost its measurement"


def test_a_bin_fitted_from_one_row_says_so_on_both_channels():
    """S2B08-14. With the default min_bin_count=1 a bin fitted from a SINGLE
    row publishes an absolute 0.0 or 1.0. That is a real measurement of the one
    row that was read, so it is kept, but the caution lived only in
    fit_result.warnings while n_bins_unmeasured read 0 and no Python warning
    fired, so a caller reading the transformed array saw a certainty with no
    tell.
    """
    # One row in the top bin, forty spread below it.
    scores = np.concatenate([np.linspace(0.02, 0.55, 40), np.array([0.95])])
    y = np.concatenate([(np.arange(40) % 3 == 0).astype(int), np.array([1])])
    cal = HistogramBinning(n_bins=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal.fit(y, scores)

    counts = cal.fit_result.to_dict()["parameters"]["bin_counts"]
    assert counts[9] == 1, counts
    assert float(cal.bin_values_[9]) == 1.0, "fixture does not reach the absolute-rate case"
    assert "fewer than 10 observations" in _joined(cal.fit_result.warnings)
    assert any("fewer than 10 observations" in m for m in _messages(caught)), _messages(caught)

    # CONTROL: the same grid with every bin well populated stays silent about
    # low counts on both channels.
    y_ok, scores_ok = _healthy(2000, seed=9)
    with warnings.catch_warnings(record=True) as caught_ok:
        warnings.simplefilter("always")
        ok = HistogramBinning(n_bins=10).fit(y_ok, scores_ok)
    assert ok.fit_result.warnings == [], ok.fit_result.warnings
    assert not caught_ok, _messages(caught_ok)
    assert min(ok.fit_result.to_dict()["parameters"]["bin_counts"]) >= 10


@pytest.mark.parametrize(
    "cls",
    [PlattScaling, IsotonicCalibrator, BetaCalibrator, TemperatureScaling, HistogramBinning],
)
def test_a_non_finite_sample_weight_is_refused_by_every_calibrator(cls):
    """S2B08-15. A NaN weight is not a weight, and `np.any(sample_weight < 0)`
    waves it through because NaN < 0 is False. Measured BEFORE, one NaN among
    60 weights on informative scores, all five calibrators answered
    differently:
      PlattScaling        -> a=b=NaN, one Python warning (honest)
      BetaCalibrator      -> a=b=m=NaN with a note about max_iter that never
                             mentions the weight
      IsotonicCalibrator  -> parameters {'n_knots': 60, 'n_distinct_values': 7},
                             warnings [], no Python warning, and one silently
                             poisoned block: transform([0.1, 0.5, 0.9]) =
                             [0.1765, 0.5556, nan]
      HistogramBinning    -> the affected bin blanked, disclosed
      TemperatureScaling  -> temperature 38.2027 and nll 0.6805, both FINITE,
                             off an objective that is NaN everywhere: the
                             bracket midpoint of a search that found nothing
    AFTER: one refusal, stated once at the shared gate.
    """
    y, scores = _healthy(60, seed=11)
    sw = np.ones(60)
    sw[5] = np.nan
    with pytest.raises(ValueError, match="non-finite sample_weight"):
        cls().fit(y, scores, sample_weight=sw)

    # CONTROL: finite weights of the same shape still fit normally.
    ok = cls().fit(y, scores, sample_weight=np.ones(60))
    assert ok.is_fitted
    assert np.all(np.isfinite(ok.transform(np.array([0.3, 0.5, 0.7]))))


def test_a_temperature_fitted_to_a_non_finite_objective_is_refused():
    """S2B08-16. scipy's bounded minimiser returns a number inside its bracket
    whatever the objective does, and every comparison against NaN is False, so
    a non-finite objective slipped past the bound test in silence. Reachable
    with FINITE weights whose weighted sums overflow. Measured BEFORE this
    guard, with the NaN-weight route: temperature 38.2027, nll 0.6805,
    warnings [] except a max_iter note, no Python warning."""
    y, scores = _healthy(60, seed=11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = TemperatureScaling().fit(y, scores, sample_weight=np.full(60, 1e308))

    assert np.isnan(cal.temperature_), cal.temperature_
    assert np.isnan(cal.fit_result.fit_metrics["nll"])
    assert "could-not-check" in _joined(cal.fit_result.warnings)
    assert any("identified NO temperature" in m for m in _messages(caught))
    assert np.all(np.isnan(cal.transform(np.array([0.1, 0.5, 0.9]))))

    # CONTROL: the same scores at a sane weight scale fit a finite temperature.
    ok = TemperatureScaling().fit(y, scores, sample_weight=np.full(60, 3.0))
    assert np.isfinite(ok.temperature_)
    assert ok.fit_result.warnings == [], ok.fit_result.warnings
