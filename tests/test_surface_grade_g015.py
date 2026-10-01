"""Batch g015: post_processing/calibration/group_calibrator.py.

Four defects proved by execution on 2026-09-17 and fixed here. Each test names
the PUBLIC entry, the measured BEFORE value and the measured AFTER value, and
each defect pin is followed by a healthy-data CONTROL asserting a real number is
still measured exactly, so an over-correction is as red as the original.

1. ``GroupCalibrator.fit`` (S2B08-13, recorded as BLOCKED in
   tests/test_bgl_stage2b_s2b08.py and unblockable only from this file).
   A base calibrator reports a could-not-check by answering NaN. PlattScaling
   on a group whose scores are constant does exactly that, correctly. The NaN
   went straight into expected_calibration_error, which refused the WHOLE array
   with ``InvalidDataError("y_prob contains 100 NaN value(s)")``: one
   unmeasurable group destroyed two healthy groups' genuine measurements, and
   the message blamed the caller's y_prob for a NaN the library produced.

2. ``GroupCalibrator.fit``. A group whose own map collapsed to a single value
   was reported as fully calibrated: ``fallback_groups=[]``,
   ``post_calibration_ece=0.0`` (the best-looking number in the table) and
   ``improvement=0.51``. The sibling IntersectionalCalibrator had grown
   ``group_provenance_`` for this exact question one day earlier; this class had
   nothing.

3. ``GroupCalibrator.transform`` / ``IntersectionalCalibrator.transform``.
   ``np.zeros_like(y_prob)`` inherited an int64 dtype from a legal 0/1
   probability vector, so every calibrated value was TRUNCATED on assignment.

4. ``GroupCalibrator.evaluate``. The returned dict carried no coverage, so a
   group excluded by the metric's own gate was simply absent from ``group_ece``
   while ``ece_disparity`` was presented as the disparity.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.post_processing.calibration.group_calibrator import (
    REASON_DEGENERATE_MAPPING,
    REASON_NON_FINITE,
    GroupCalibrator,
    IntersectionalCalibrator,
    UnknownGroupWarning,
    UnmeasurableGroupWarning,
)

# ===========================================================================
# fixtures
# ===========================================================================


def _healthy_two_groups():
    """400 + 400 rows. Group 'b' carries a real, findable calibration gap:
    its scores are the true probabilities shrunk by 0.4, so its ECE is roughly
    0.3 against roughly 0.04 for group 'a'."""
    rng = np.random.default_rng(0)
    p_a = rng.uniform(0.02, 0.98, 400)
    y_a = (rng.uniform(size=400) < p_a).astype(int)
    p_true_b = rng.uniform(0.02, 0.98, 400)
    y_b = (rng.uniform(size=400) < p_true_b).astype(int)
    p_b = np.clip(p_true_b * 0.4, 0.001, 0.999)
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["a"] * 400 + ["b"] * 400),
    )


def _one_group_cannot_be_fitted():
    """A, B healthy; C's 100 scores are all 0.7, so the Platt design is
    rank-deficient and PlattScaling returns NaN. This is the S2B08-13 fixture,
    reproduced value for value."""
    rng = np.random.default_rng(3)
    groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    scores = np.concatenate([rng.random(100), rng.random(100), np.full(100, 0.7)])
    y = (rng.random(300) < np.clip(scores, 0.05, 0.95)).astype(int)
    return y, scores, groups


def _single_observed_class():
    """200 rows over two groups, every label 0. The scores VARY, so a flat
    fitted map really does mean the map learned nothing."""
    rng = np.random.default_rng(1)
    rng.uniform(0.02, 0.98, 200)  # keep the stream where the probe had it
    scores = rng.uniform(0.02, 0.98, 200)
    return np.zeros(200, dtype=int), scores, np.array(["a"] * 100 + ["b"] * 100)


# ===========================================================================
# 1. one unmeasurable group must not destroy the measurable ones
# ===========================================================================


class TestOneUnmeasurableGroupKeepsTheOthersMeasured:
    def test_fit_completes_and_names_the_group_it_could_not_calibrate(self):
        """PUBLIC ENTRY: ``GroupCalibrator.fit`` then ``get_calibration_result``.

        BEFORE: InvalidDataError("y_prob contains 100 NaN value(s); ...") out of
                fit(). No result object at all, so A's and B's real ECEs were
                lost with it.
        AFTER:  fit returns; A 0.0937 and B 0.0788 are measured, C is NaN and
                named ``'C': non_finite_output``.
        """
        y, scores, groups = _one_group_cannot_be_fitted()
        # The fixture really does carry the branch it claims to.
        assert np.unique(scores[200:]).size == 1, "group C's scores must be constant"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calib = GroupCalibrator(method="platt").fit(y, scores, groups)

        result = calib.get_calibration_result()
        assert result.unmeasurable_groups == {"C": REASON_NON_FINITE}, result.unmeasurable_groups
        # The could-not-check is NaN, never 0.0: 0.0 would make the group
        # nobody could measure the best calibrated group in the table.
        assert np.isnan(result.post_calibration_ece["C"])
        assert np.isnan(result.improvement["C"])
        # ... and the healthy groups kept their genuine measurements.
        assert np.isfinite(result.post_calibration_ece["A"])
        assert np.isfinite(result.post_calibration_ece["B"])
        assert np.isfinite(result.overall_improvement)

        msgs = [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]
        assert len(msgs) == 1, [str(w.message) for w in caught]
        assert "'C': non_finite_output" in str(msgs[0].message)

        # The disclosure survives serialisation, which is the surface a report
        # or a stored artefact actually reads.
        assert result.to_dict()["unmeasurable_groups"] == {"C": REASON_NON_FINITE}

    def test_a_caller_can_fail_closed_without_losing_the_fit(self):
        """Escalating the disclosure must raise, and must NOT leave an object
        that answers from nothing. Same rule the sibling class learned."""
        y, scores, groups = _one_group_cannot_be_fitted()
        calib = GroupCalibrator(method="platt")
        with pytest.raises(UnmeasurableGroupWarning):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.simplefilter("error", UnmeasurableGroupWarning)
                calib.fit(y, scores, groups)
        assert calib.is_fitted is True
        assert calib.get_calibration_result().unmeasurable_groups == {"C": REASON_NON_FINITE}

    def test_control_the_healthy_fit_measures_the_real_gap_and_says_nothing(self):
        """OVER-CORRECTION CONTROL. Both groups are learnable, so
        ``unmeasurable_groups`` must be EMPTY, no warning fires, and the numbers
        are the ones an independent computation gives.
        """
        y, p, g = _healthy_two_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calib = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
        result = calib.get_calibration_result()

        assert result.unmeasurable_groups == {}
        assert [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)] == []

        # Independent expectation: pre-calibration ECE for group 'b' computed
        # here from the definition, not copied from the code under test.
        mask = g == "b"
        edges = np.linspace(0, 1, 11)
        idx = np.digitize(p[mask], edges[1:-1])
        expected_b = sum(
            (np.sum(idx == i) / mask.sum())
            * abs(np.mean(y[mask][idx == i]) - np.mean(p[mask][idx == i]))
            for i in range(10)
            if np.sum(idx == i) > 0
        )
        assert result.pre_calibration_ece["b"] == pytest.approx(expected_b, abs=1e-12)
        assert expected_b > 0.2, "the fixture must carry a findable disparity"
        # And the calibration really improved it.
        assert result.improvement["b"] > 0.2
        assert result.post_calibration_ece["b"] < 1e-6


# ===========================================================================
# 2. a map that learned nothing is not a calibration
# ===========================================================================


class TestAMapThatLearnedNothingIsLabelled:
    def test_a_single_class_group_is_not_reported_as_calibrated(self):
        """PUBLIC ENTRY: ``GroupCalibrator.fit`` then ``get_calibration_result``.

        BEFORE: fallback_groups=[], post_calibration_ece={'a': 0.0, 'b': 0.0},
                improvement={'a': 0.5115, 'b': 0.5124}, overall 0.5119, and
                nothing anywhere saying the maps return one value for every
                input.
        AFTER:  unmeasurable_groups={'a': degenerate_mapping,
                'b': degenerate_mapping} plus one UnmeasurableGroupWarning.
        """
        y, scores, groups = _single_observed_class()
        assert np.unique(y).size == 1, "the fixture must carry ONE observed class"
        assert np.unique(scores).size > 2, "the INPUT must vary, or a flat map says nothing"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calib = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, scores, groups)

        result = calib.get_calibration_result()
        assert result.unmeasurable_groups == {
            "a": REASON_DEGENERATE_MAPPING,
            "b": REASON_DEGENERATE_MAPPING,
        }, result.unmeasurable_groups

        msgs = [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]
        assert len(msgs) == 1, [str(w.message) for w in caught]
        assert "degenerate_mapping" in str(msgs[0].message)

        # The ECE of a constant predictor is a real number and is KEPT: the
        # defect was the missing label, not the arithmetic.
        assert result.post_calibration_ece["a"] == pytest.approx(0.0, abs=1e-12)
        out = calib.transform(scores, groups)
        assert np.unique(out).size == 1, "the map really does answer one value"

    def test_control_a_learnable_group_of_the_same_size_is_untouched(self):
        """OVER-CORRECTION CONTROL. Same size, same method, labels carrying both
        classes: no label, no warning, and the transform still varies."""
        y, p, g = _healthy_two_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calib = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
            out = calib.transform(p, g)
        assert calib.get_calibration_result().unmeasurable_groups == {}
        assert [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)] == []
        assert np.ptp(out) > 0.5, "a healthy calibration must still vary"


# ===========================================================================
# 3. an integer probability vector must not be truncated
# ===========================================================================


class TestIntegerProbabilitiesAreNotTruncated:
    @staticmethod
    def _fitted():
        rng = np.random.default_rng(3)
        p = rng.uniform(0.02, 0.98, 400)
        y = (rng.uniform(size=400) < p * 0.5).astype(int)
        g = np.array(["a"] * 200 + ["b"] * 200)
        return GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g), g

    def test_group_transform_returns_floats_for_an_int_input(self):
        """PUBLIC ENTRY: ``GroupCalibrator.transform``.

        BEFORE: dtype int64, unique values [0, 1]: np.zeros_like inherited the
                input dtype and 0.8 was truncated to 0 on assignment.
        AFTER:  dtype float64, unique values [0.0, 0.8, 1.0], identical to the
                same call with the same numbers as floats.
        """
        calib, _ = self._fitted()
        probe_groups = np.array(["a"] * 100 + ["b"] * 100)
        as_int = np.array([0, 1] * 100)
        assert as_int.dtype.kind == "i", "the fixture must really be integer"

        out_int = calib.transform(as_int, probe_groups)
        out_float = calib.transform(as_int.astype(float), probe_groups)

        assert out_int.dtype.kind == "f"
        assert np.array_equal(out_int, out_float), (np.unique(out_int), np.unique(out_float))
        # And the values really are calibrated, not the input back again.
        assert not np.array_equal(out_int, as_int.astype(float))
        assert np.unique(out_int).size > 2

    def test_intersectional_transform_returns_floats_for_an_int_input(self):
        """Same shape, same file, the sibling class."""
        rng = np.random.default_rng(4)
        p = rng.uniform(0.02, 0.98, 400)
        y = (rng.uniform(size=400) < p * 0.5).astype(int)
        frame = pd.DataFrame(
            {
                "a": ["x"] * 200 + ["y"] * 200,
                "b": ["p"] * 100 + ["q"] * 100 + ["p"] * 100 + ["q"] * 100,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit(y, p, frame)
            out = calib.transform(np.array([0, 1] * 200), frame)
        assert out.dtype.kind == "f"
        assert np.unique(out).size > 2, np.unique(out)

    def test_control_a_float_input_is_unchanged_to_the_last_bit(self):
        """OVER-CORRECTION CONTROL. The dtype fix must not move any number that
        was already right."""
        calib, g = self._fitted()
        rng = np.random.default_rng(9)
        probe = rng.uniform(0.02, 0.98, 400)
        out = calib.transform(probe, g)
        expected = np.zeros(400)
        for name in ("a", "b"):
            mask = g == name
            expected[mask] = calib.calibrators_[name].transform(probe[mask])
        assert np.array_equal(out, np.clip(expected, 0, 1))


# ===========================================================================
# 4. evaluate() must say which groups its figures cover
# ===========================================================================


class TestEvaluateDisclosesItsCoverage:
    @staticmethod
    def _three_groups_one_small():
        """400 / 400 / 20. Group c is below the metric gate of 30 and is the
        WORST calibrated of the three, which is the usual way round."""
        rng = np.random.default_rng(11)

        def block(n, shrink):
            pt = rng.uniform(0.02, 0.98, n)
            return (
                (rng.uniform(size=n) < pt).astype(int),
                np.clip(pt * shrink, 0.001, 0.999),
            )

        ya, pa = block(400, 1.0)
        yb, pb = block(400, 0.6)
        yc, pc = block(20, 0.05)
        return (
            np.concatenate([ya, yb, yc]),
            np.concatenate([pa, pb, pc]),
            np.array(["a"] * 400 + ["b"] * 400 + ["c"] * 20),
        )

    def test_a_group_the_figures_do_not_cover_is_named(self):
        """PUBLIC ENTRY: ``GroupCalibrator.evaluate``.

        BEFORE: {'pre_calibration': {'ece_disparity': 0.16629, 'group_ece':
                {'a': ..., 'b': ...}}, ...} and no key anywhere mentioning 'c',
                whose own pre-calibration ECE at fit time was 0.5286.
        AFTER:  the same figures plus
                coverage={'metric_min_group_size': 30, 'groups_assessed':
                ['a', 'b'], 'groups_not_assessed': ['c']}.
        """
        y, p, g = self._three_groups_one_small()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = GroupCalibrator(method="isotonic", min_group_size=10).fit(y, p, g)
            report = calib.evaluate(y, p, g)

        assert report["coverage"]["groups_not_assessed"] == ["c"]
        assert report["coverage"]["groups_assessed"] == ["a", "b"]
        assert report["coverage"]["metric_min_group_size"] == 30
        # The claim the disclosure exists for: 'c' is absent from the figures
        # and is the worst calibrated group in the data.
        assert "c" not in report["pre_calibration"]["group_ece"]
        assert (
            calib.get_calibration_result().pre_calibration_ece["c"]
            > (report["pre_calibration"]["ece_disparity"])
        )

        # And the caller can lower the gate to cover it, which was impossible
        # before: the threshold was hard-coded inside evaluate().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wider = calib.evaluate(y, p, g, metric_min_group_size=10)
        assert wider["coverage"]["groups_not_assessed"] == []
        assert "c" in wider["pre_calibration"]["group_ece"]

    def test_control_full_coverage_reports_an_empty_exclusion_and_real_numbers(self):
        """OVER-CORRECTION CONTROL. With every group above the gate the
        disclosure must be EMPTY, and the disparity must still be the measured
        one: an empty groups_not_assessed has to mean 'nothing excluded', not
        'nothing checked'."""
        y, p, g = _healthy_two_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calib = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
            report = calib.evaluate(y, p, g)
        assert caught == [], [str(w.message) for w in caught]
        assert report["coverage"]["groups_not_assessed"] == []
        assert report["coverage"]["groups_assessed"] == ["a", "b"]

        group_ece = report["pre_calibration"]["group_ece"]
        expected_disparity = max(group_ece.values()) - min(group_ece.values())
        assert report["pre_calibration"]["ece_disparity"] == pytest.approx(
            expected_disparity, abs=1e-12
        )
        assert expected_disparity > 0.2, "the control must carry a real disparity"


# ===========================================================================
# 5. a lookup that returns None must say WHICH None it means
# ===========================================================================


class TestLookupsSeparateUnknownFromUncalibrated:
    def test_an_unknown_group_is_not_reported_as_uncalibrated(self):
        """PUBLIC ENTRY: ``GroupCalibrator.get_group_calibrator``.

        BEFORE: get_group_calibrator('ZZZ') returned None in silence, the same
                answer as a real group the caller chose to leave uncalibrated.
        AFTER:  still None, with an UnknownGroupWarning naming the group.
        """
        y, p, g = _healthy_two_groups()
        calib = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
        with pytest.warns(UnknownGroupWarning, match="ZZZ"):
            assert calib.get_group_calibrator("ZZZ") is None

    def test_control_a_deliberately_uncalibrated_group_stays_silent(self):
        """OVER-CORRECTION CONTROL. ``fallback_strategy='none'`` is the caller's
        own choice at fit time, so that None must NOT warn, or the warning
        becomes noise nobody reads. A known, calibrated group must not warn
        either."""
        y, p, g = _healthy_two_groups()
        calib = GroupCalibrator(
            method="isotonic", min_group_size=2000, fallback_strategy="none"
        ).fit(y, p, g)
        assert calib.calibrators_["a"] is None, "fixture must reach the 'none' branch"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert calib.get_group_calibrator("a") is None

        fitted = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fitted.get_group_calibrator("a") is not None

    def test_an_intersection_named_with_a_missing_attribute_is_refused(self):
        """PUBLIC ENTRY: ``IntersectionalCalibrator.get_calibrator_for_group``.

        BEFORE: {'gender': 'M'} built the key 'M_' with an empty string for the
                missing attribute and returned None, i.e. answered a question
                about an intersection that does not exist.
        AFTER:  ValueError naming the missing attribute; an unknown but
                well-formed intersection warns instead of answering silently.
        """
        rng = np.random.default_rng(6)
        p = rng.uniform(0.05, 0.95, 400)
        y = (rng.uniform(size=400) < p).astype(int)
        frame = pd.DataFrame(
            {
                "gender": ["M"] * 200 + ["F"] * 200,
                "race": ["W"] * 100 + ["B"] * 100 + ["W"] * 100 + ["B"] * 100,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit(y, p, frame)

        with pytest.raises(ValueError, match="race"):
            calib.get_calibrator_for_group({"gender": "M"})

        with pytest.warns(UnknownGroupWarning, match="Q_Z"):
            assert calib.get_calibrator_for_group({"gender": "Q", "race": "Z"}) is None

        # CONTROL: a real intersection still resolves, silently.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                calib.get_calibrator_for_group({"gender": "M", "race": "W"})
                is (calib.calibrators_["M_W"])
            )


# ===========================================================================
# 6. the rest of the batch: pinned so a later change cannot quietly undo them
# ===========================================================================


class TestTheRemainingSurfaces:
    def test_to_dict_carries_every_field_of_the_dataclass(self):
        """``GroupCalibrationResult.to_dict`` is a field-by-field rebuild, which
        is the shape that drops the NEXT field somebody adds. Structural, so it
        fails for a field that does not exist yet rather than only for the two
        that do."""
        import dataclasses

        y, p, g = _healthy_two_groups()
        result = (
            GroupCalibrator(method="isotonic", min_group_size=50)
            .fit(y, p, g)
            .get_calibration_result()
        )
        assert set(result.to_dict()) == {f.name for f in dataclasses.fields(result)}

    def test_fit_transform_is_fit_then_transform_and_keeps_the_disclosure(self):
        """``GroupCalibrator.fit_transform``.

        BEFORE (on the unmeasurable fixture): InvalidDataError out of fit().
        AFTER: the 100 rows of the group nobody could calibrate come back NaN,
        which a caller cannot mistake for a probability, and the fit warns.
        """
        y, p, g = _healthy_two_groups()
        one_shot = GroupCalibrator(method="isotonic", min_group_size=50).fit_transform(y, p, g)
        two_step = GroupCalibrator(method="isotonic", min_group_size=50).fit(y, p, g)
        assert np.array_equal(one_shot, two_step.transform(p, g))
        assert np.ptp(one_shot) > 0.5

        yb, scores, groups = _one_group_cannot_be_fitted()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = GroupCalibrator(method="platt").fit_transform(yb, scores, groups)
        assert int(np.isnan(out).sum()) == 100, np.isnan(out).sum()
        assert np.isfinite(out[:200]).all(), "the measurable groups must keep their values"
        assert [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]

    def test_intersectional_fit_transform_is_fit_then_transform(self):
        rng = np.random.default_rng(12)
        p = rng.uniform(0.05, 0.95, 800)
        y = (rng.uniform(size=800) < p).astype(int)
        frame = pd.DataFrame(
            {
                "gender": ["M"] * 400 + ["F"] * 400,
                "race": ["W"] * 200 + ["B"] * 200 + ["W"] * 200 + ["B"] * 200,
            }
        )
        one_shot = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit_transform(
            y, p, frame
        )
        two_step = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit(y, p, frame)
        assert np.array_equal(one_shot, two_step.transform(p, frame))
        assert np.ptp(one_shot) > 0.5

    def test_group_statistics_and_provenance_are_the_measured_ones(self):
        """``get_group_statistics`` / ``get_group_provenance``: real counts and
        real labels, and a REFUSAL rather than an empty dict before fit()."""
        rng = np.random.default_rng(13)
        sizes = {("M", "W"): 300, ("M", "B"): 300, ("F", "W"): 300, ("F", "B"): 5}
        rows, ps, ys = [], [], []
        for pair, k in sizes.items():
            pt = rng.uniform(0.05, 0.95, k)
            rows += [pair] * k
            ps.append(pt)
            ys.append((rng.uniform(size=k) < pt).astype(int))
        frame = pd.DataFrame(rows, columns=["gender", "race"])
        p = np.concatenate(ps)
        y = np.concatenate(ys)

        with pytest.raises(RuntimeError):
            IntersectionalCalibrator().get_group_statistics()
        with pytest.raises(RuntimeError):
            IntersectionalCalibrator().get_group_provenance()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit(y, p, frame)

        assert calib.get_group_statistics() == {
            "M_W": 300,
            "M_B": 300,
            "F_W": 300,
            "F_B": 5,
        }
        provenance = calib.get_group_provenance()
        # The sparse intersection borrowed the gender-marginal curve, which is
        # the POOLED answer the caller asked intersectional calibration to
        # avoid, so it must not be labelled intersectional.
        assert provenance["F_B"] == "borrowed_marginal", provenance
        assert provenance["M_W"] == "intersectional", provenance

    def test_the_two_strategy_validators_refuse_what_they_do_not_run(self):
        """``validate_fallback_strategy`` / ``validate_borrowing_strategy``.
        Both return the string they accept, so a weakened version is invisible
        unless the refusals are pinned as well."""
        from vfairness.post_processing.calibration.group_calibrator import (
            validate_borrowing_strategy,
            validate_fallback_strategy,
        )

        assert validate_fallback_strategy("global") == "global"
        assert validate_fallback_strategy("none") == "none"
        assert validate_borrowing_strategy("hierarchical") == "hierarchical"
        assert validate_borrowing_strategy("none") == "none"

        # 'borrow' used to be accepted and silently run something else.
        with pytest.raises(NotImplementedError, match="borrow"):
            validate_fallback_strategy("borrow")
        with pytest.raises(ValueError, match="borrow"):
            validate_borrowing_strategy("borrow")
        with pytest.raises(ValueError):
            validate_fallback_strategy("bogus")
        with pytest.raises(ValueError):
            validate_borrowing_strategy("bogus")
        # And the constructors really go through them.
        with pytest.raises(NotImplementedError):
            GroupCalibrator(fallback_strategy="borrow")
        with pytest.raises(ValueError):
            IntersectionalCalibrator(borrowing_strategy="borrow")

    def test_get_calibration_result_refuses_before_fit_and_carries_the_disclosure(self):
        with pytest.raises(RuntimeError):
            GroupCalibrator().get_calibration_result()
        y, scores, groups = _one_group_cannot_be_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = GroupCalibrator(method="platt").fit(y, scores, groups)
        assert calib.get_calibration_result().unmeasurable_groups == {"C": REASON_NON_FINITE}

    def test_evaluate_survives_a_group_that_could_not_be_calibrated(self):
        """PUBLIC ENTRY: ``GroupCalibrator.evaluate``. The S2B08-13 defect had
        moved one method over.

        BEFORE: InvalidDataError("y_prob contains 100 NaN value(s); ...") out of
                evaluate(), so A's and B's real figures were destroyed by the
                third group and the message blamed the caller's y_prob.
        AFTER:  A and B are measured (pre ECE 0.1064 and 0.1094), C's 100 rows
                are excluded and named in
                coverage['groups_not_calibrated'] == ['C'].
        """
        y, scores, groups = _one_group_cannot_be_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = GroupCalibrator(method="platt").fit(y, scores, groups)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = calib.evaluate(y, scores, groups)

        assert report["coverage"]["groups_not_calibrated"] == ["C"]
        assert report["coverage"]["rows_measured"] == 200
        assert report["coverage"]["rows_total"] == 300
        assert sorted(report["pre_calibration"]["group_ece"]) == ["A", "B"]
        assert np.isfinite(report["pre_calibration"]["ece_disparity"])
        assert [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]

    def test_evaluate_refuses_rather_than_returning_zero_when_nothing_is_measurable(self):
        """The other direction: when NOTHING can be measured the figures must be
        NaN, never 0.0, and the coverage must say so."""
        y, scores, groups = _one_group_cannot_be_fitted()
        only_c = np.array(["C"] * 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = GroupCalibrator(method="platt", min_group_size=10).fit(
                y[200:], scores[200:], only_c
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = calib.evaluate(y[200:], scores[200:], only_c)

        assert np.isnan(report["pre_calibration"]["overall_ece"])
        assert np.isnan(report["improvement"]["ece_disparity"])
        assert report["pre_calibration"]["group_ece"] == {}
        assert report["coverage"]["rows_measured"] == 0
        assert report["coverage"]["groups_not_calibrated"] == ["C"]
        assert [w for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]


class TestARefusedFitIsNotSwallowedByTheConstantInputGate:
    """``_mapping_is_degenerate`` had its two tests the wrong way round, so the
    early return meant for "a constant output says nothing here" also swallowed
    the fitter's own explicit could-not-check. This is the shape the campaign
    warns about: a guard for "could not check" deleting a real finding."""

    @staticmethod
    def _constant_scores_in_one_intersection():
        rng = np.random.default_rng(5)
        rows = [("M", "W")] * 100 + [("M", "B")] * 100 + [("F", "W")] * 100 + [("F", "B")] * 100
        frame = pd.DataFrame(rows, columns=["gender", "race"])
        p = np.concatenate([np.full(100, 0.7), rng.uniform(0.05, 0.95, 300)])
        y = (rng.uniform(size=400) < np.clip(p, 0.05, 0.95)).astype(int)
        return y, p, frame

    def test_a_nan_map_is_never_labelled_intersectional(self):
        """PUBLIC ENTRY: ``IntersectionalCalibrator.fit`` then
        ``get_group_provenance`` and ``transform``.

        BEFORE: group_provenance_['M_W'] == 'intersectional' for 100 rows that
                all came back NaN, and transform() emitted NO provenance
                warning at all, because its disclosure only looks for labels
                that are not 'intersectional'.
        AFTER:  'intersectional_degenerate', and transform says
                "100 of 400 returned row(s) are NOT intersectionally
                calibrated".
        """
        y, p, frame = self._constant_scores_in_one_intersection()
        assert np.unique(p[:100]).size == 1, "M_W's scores must be constant"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = IntersectionalCalibrator(method="platt", min_group_size=30).fit(y, p, frame)

        provenance = calib.get_group_provenance()
        assert provenance["M_W"] == "intersectional_degenerate", provenance
        assert provenance["M_B"] == "intersectional", provenance

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = calib.transform(p, frame)
        assert int(np.isnan(out).sum()) == 100
        msgs = [
            str(w.message) for w in caught if "NOT intersectionally calibrated" in str(w.message)
        ]
        assert msgs, [str(w.message) for w in caught]
        assert "100 of 400 returned row(s)" in msgs[0], msgs[0]

    def test_control_a_constant_input_with_a_real_finite_map_stays_intersectional(self):
        """OVER-CORRECTION CONTROL. Constant scores are not themselves a
        refusal: isotonic answers the training base rate, which IS the correct
        calibration of an uninformative score, so the label must not move."""
        rng = np.random.default_rng(8)
        rows = [("M", "W")] * 100 + [("M", "B")] * 100
        frame = pd.DataFrame(rows, columns=["gender", "race"])
        p = np.concatenate([np.full(100, 0.7), rng.uniform(0.05, 0.95, 100)])
        y = (rng.uniform(size=200) < np.clip(p, 0.05, 0.95)).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = IntersectionalCalibrator(method="isotonic", min_group_size=30).fit(y, p, frame)
            out = calib.transform(p, frame)
        assert calib.get_group_provenance()["M_W"] == "intersectional"
        assert np.isfinite(out).all()
        # The calibrated value really is the group's own base rate, measured.
        assert out[:100][0] == pytest.approx(float(np.mean(y[:100])), abs=1e-12)

    def test_transform_says_why_its_nan_rows_are_nan(self):
        """PUBLIC ENTRY: ``GroupCalibrator.transform``.

        BEFORE: 100 of 300 returned values were NaN and transform() emitted
                nothing, so a caller who fits in one place and transforms in
                another got them with no tell. fit() warns once, at fit time;
                this is the call that hands over the numbers.
        AFTER:  the same NaN (it is the honest answer) plus an
                UnmeasurableGroupWarning naming group 'C'.
        """
        y, scores, groups = _one_group_cannot_be_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib = GroupCalibrator(method="platt").fit(y, scores, groups)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = calib.transform(scores, groups)

        assert int(np.isnan(out).sum()) == 100
        msgs = [str(w.message) for w in caught if isinstance(w.message, UnmeasurableGroupWarning)]
        assert msgs, [str(w.message) for w in caught]
        assert "100 of 300 returned value(s) are NaN" in msgs[0], msgs[0]
        assert "'C'" in msgs[0], msgs[0]
        # CONTROL: a transform with no NaN in it stays silent, or the warning
        # becomes noise nobody reads.
        yh, ph, gh = _healthy_two_groups()
        healthy = GroupCalibrator(method="isotonic", min_group_size=50).fit(yh, ph, gh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            healthy.transform(ph, gh)
