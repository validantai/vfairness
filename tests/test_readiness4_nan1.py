"""NAN-01 (readiness wave 4, 2026-09-10). One NaN flipped a maximal finding.

Both defects below are the same shape: a single unmeasurable element in an
otherwise good input turned a correct, maximal finding into a clean pass. The
user did not get a smaller number, they got the opposite answer.

1. ``NonDeterminismAnalyzer.compute_noise_offset`` graded a NaN as "within
   noise". ``max(0.0, nan)`` is 0.0 and ``nan > floor`` is False, so ONE
   unscored run out of 30 turned a real 0.40 LLM disparity from significant
   to NOT significant, and reported a systematic_offset of 0.0 computed from
   that same NaN. Measured 2026-09-10, no warning of any kind:

       30/30 scored : floor=0.016725 offset=0.383610 exceeds=True  sig=True
       29/30 scored : floor=nan      offset=0.0      exceeds=False sig=False

2. ``EmergentBiasDetector.analyze`` fabricated ``amplification_factor=1.0``
   and ``is_emergent=False`` when the system output series was not finite.
   ONE NaN in 200 system outputs flipped a maximal, correctly detected
   emergent bias to "no bias". Measured 2026-09-10, no warning:

       200/200 finite: bias=1.0 amp=1000000.0 is_emergent=True
       199/200 finite: bias=nan amp=1.0       is_emergent=False

   Those amplification figures are the values of the day. READINESS-6
   (2026-09-10) replaced the 1e6 sentinel with nan: a ratio over a zero
   component baseline is undefined, and the pulse report was grading severity
   off it. The pins below assert nan where they used to assert 1e6; every
   verdict they pin is unchanged.

Each finding gets a refusal pin AND an over-correction control asserting the
MEASURED values, because a fix that answers "could not check" to everything
is the same failure wearing the other mask.
"""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from vfairness.llm.nondeterminism import NonDeterminismAnalyzer
from vfairness.multi_agent.emergent import EmergentBiasDetector

# ---------------------------------------------------------------------------
# Fixtures shared by the pins and their controls. Ordinary data: a tight
# per-run series with a real 0.40 disparity, and a perfectly separating
# multi-agent system whose components are individually unbiased.
# ---------------------------------------------------------------------------

REAL_DISPARITY = 0.40


def _thirty_runs() -> np.ndarray:
    """30 repeated identical-prompt scores, run-to-run noise about 0.01."""
    return np.random.default_rng(7).normal(0.50, 0.01, 30)


def _emergent_case():
    """200 samples, components individually unbiased, system separating."""
    groups = np.array([0] * 100 + [1] * 100)
    components = {"a": np.full(200, 0.5), "b": np.full(200, 0.5)}
    system = np.concatenate([np.ones(100), np.zeros(100)])
    return components, system, groups


# ---------------------------------------------------------------------------
# Finding 1: the noise offset
# ---------------------------------------------------------------------------


class TestNoiseOffsetRefusesWhatItCouldNotMeasure:
    def test_a_non_finite_noise_floor_is_not_a_quiet_one(self):
        """REFUSAL PIN. A NaN floor reaching the grader must withhold the
        grade, not answer "within noise"."""
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            offset = analyzer.compute_noise_offset(REAL_DISPARITY, float("nan"))
            verdict = analyzer.is_significant_after_offset(REAL_DISPARITY, float("nan"))

        assert offset["state"] == "could_not_check", offset
        assert offset["reason"] == "observed_disparity_or_noise_floor_not_finite"
        assert offset["systematic_offset"] is None, (
            f"a systematic offset of {offset['systematic_offset']!r} was computed from "
            f"a NaN noise floor"
        )
        assert offset["exceeds_noise"] is None, offset["exceeds_noise"]
        assert verdict is None, f"a NaN floor answered {verdict!r}, not 'could not check'"
        assert verdict is not False, "could-not-check must never collapse into not-significant"
        assert caught, "the refusal happened in silence"

    def test_a_non_finite_disparity_is_not_a_small_one(self):
        """REFUSAL PIN, the other input. An unmeasurable disparity graded
        against a perfectly good floor is still unmeasurable."""
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            offset = analyzer.compute_noise_offset(float("nan"), 0.0167)
            verdict = analyzer.is_significant_after_offset(float("nan"), 0.0167)

        assert offset["state"] == "could_not_check"
        assert offset["systematic_offset"] is None and offset["exceeds_noise"] is None
        assert verdict is None, f"a NaN disparity answered {verdict!r}"
        assert caught

    def test_one_unscored_run_of_thirty_is_excluded_and_said_so(self):
        """REFUSAL PIN for the real path this arrives on. The floor is
        measured on the 29 runs that carry a measurement, the exclusion is
        counted on the profile, and the real 0.40 disparity SURVIVES as
        significant instead of being erased."""
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)
        runs = _thirty_runs()
        one_unscored = runs.copy()
        one_unscored[13] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile = analyzer.characterize_noise(one_unscored)

        assert profile.sample_size == 29, profile.sample_size
        assert profile.n_excluded_non_finite == 1, profile.n_excluded_non_finite
        assert math.isfinite(profile.noise_floor), profile.noise_floor
        assert any("not finite" in str(w.message) for w in caught), [str(w.message) for w in caught]
        assert profile.metadata.parameters["n_supplied"] == 30

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            offset = analyzer.compute_noise_offset(REAL_DISPARITY, profile.noise_floor)
            verdict = analyzer.is_significant_after_offset(REAL_DISPARITY, profile.noise_floor)

        assert offset["state"] == "measured"
        assert verdict is True, "the 0.40 disparity was erased by one unscored run"
        assert offset["exceeds_noise"] is True
        assert offset["systematic_offset"] > 0.38, offset["systematic_offset"]
        assert not caught, [str(w.message) for w in caught]

    def test_too_few_finite_values_is_refused_loudly(self):
        """A series that is almost all NaN cannot yield a floor at all."""
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="at least 2"):
                analyzer.characterize_noise(np.array([0.5, np.nan, np.nan]))


class TestNoiseFloorFromRunsNeverCastsAWithheldGrade:
    """Sibling of finding 1, reached through the public entry point.

    ``noise_floor_from_runs`` checks that each per-run SERIES is finite, and
    then builds the disparity floor from its variance. A scorer whose finite
    scores overflow in the variance makes that floor infinite, and the old
    arithmetic answered `systematic_offset 0.0, exceeds_noise False`: within
    noise, from a number that overflowed. The comparison must say
    could_not_check, and the ``bool(...)`` cast in the measured branch must
    never see a None (``bool(None)`` is False, which is the same wrong answer).
    """

    def test_an_overflowing_floor_is_could_not_check_not_within_noise(self):
        from vfairness.llm.nondeterminism import noise_floor_from_runs

        class HugeScorer:
            """Finite per-run scores whose variance is not finite."""

            name = "huge"

            def score_batch(self, texts):
                return [1e200 if i % 2 else -1e200 for i in range(len(texts))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": ["a"] * 6, "B": ["b"] * 6},
                metrics=["sentiment"],
                scorers={"sentiment": HugeScorer()},
            )

        comparison = out["metrics"]["sentiment"]["comparisons"][0]
        assert comparison["state"] == "could_not_check", comparison
        assert comparison["exceeds_noise"] is None, comparison["exceeds_noise"]
        assert comparison["systematic_offset"] is None
        assert comparison["reason"] == "observed_disparity_or_noise_floor_not_finite"
        assert out["metrics"]["sentiment"]["state"] == "could_not_check"
        assert out["state"] == "could_not_check" and out["available"] is False


class TestNoiseOffsetStillMeasuresWhatItCanMeasure:
    """OVER-CORRECTION CONTROL. The measured path must still produce the real
    numbers, in both directions, without warning."""

    def test_thirty_clean_runs_grade_a_real_disparity_by_the_arithmetic(self):
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)
        runs = _thirty_runs()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile = analyzer.characterize_noise(runs)
            offset = analyzer.compute_noise_offset(REAL_DISPARITY, profile.noise_floor)
            verdict = analyzer.is_significant_after_offset(REAL_DISPARITY, profile.noise_floor)

        assert not caught, [str(w.message) for w in caught]
        assert profile.sample_size == 30 and profile.n_excluded_non_finite == 0

        expected_floor = 2.0 * float(np.std(runs, ddof=1))
        assert profile.noise_floor == pytest.approx(expected_floor, rel=1e-12)
        assert profile.noise_floor == pytest.approx(0.016725, abs=5e-6)

        z = float(stats.norm.ppf(0.975))
        expected_offset = REAL_DISPARITY - expected_floor * z / 2.0
        assert offset["state"] == "measured" and offset["reason"] is None
        assert offset["systematic_offset"] == pytest.approx(expected_offset, rel=1e-12)
        assert offset["systematic_offset"] == pytest.approx(0.383610, abs=5e-6)
        assert offset["exceeds_noise"] is True
        assert verdict is True

    def test_a_disparity_inside_the_noise_is_still_reported_as_not_significant(self):
        """The third state must not swallow the second: a real, measured
        'no, this is noise' still has to come back False."""
        analyzer = NonDeterminismAnalyzer("llm", min_runs=30)
        runs = _thirty_runs()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile = analyzer.characterize_noise(runs)
            tiny = 0.001
            offset = analyzer.compute_noise_offset(tiny, profile.noise_floor)
            verdict = analyzer.is_significant_after_offset(tiny, profile.noise_floor)

        assert not caught
        assert offset["state"] == "measured"
        assert offset["exceeds_noise"] is False
        assert offset["systematic_offset"] == 0.0
        assert verdict is False, f"a genuinely small disparity answered {verdict!r}"


# ---------------------------------------------------------------------------
# Finding 2: emergent bias
# ---------------------------------------------------------------------------


class TestEmergentBiasSurvivesOneUnmeasurableOutput:
    def test_one_nan_in_two_hundred_does_not_erase_a_maximal_finding(self):
        """REFUSAL PIN. The NaN sample carries no system measurement, so it is
        excluded and counted. The other 199 still separate perfectly, and the
        emergence they show must still be reported."""
        components, system, groups = _emergent_case()
        system_one_nan = system.copy()
        system_one_nan[42] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = EmergentBiasDetector().analyze(components, system_one_nan, groups)

        assert result.is_emergent is True, (
            f"one NaN in 200 outputs turned emergent bias into {result.is_emergent!r} "
            f"(amplification {result.amplification_factor})"
        )
        # READINESS-6 (2026-09-10). 1e6 was a SENTINEL for a ratio whose
        # denominator is zero, and the shipped pulse report graded severity
        # straight off this field, so a 0.001 noise gap over a zero component
        # baseline came out "critical". The ratio is undefined here (every
        # component measured a bias of exactly 0) and is reported as nan;
        # emergence is decided by the bootstrap comparison instead. The subject
        # of this pin is untouched: one NaN in 200 must not erase the finding,
        # and is_emergent above is still True.
        assert math.isnan(result.amplification_factor), result.amplification_factor
        assert result.system_bias == 1.0, result.system_bias
        assert result.n_samples_unmeasurable == 1, result.n_samples_unmeasurable
        assert result.metadata.parameters["n_samples_measured"] == 199
        assert any("not finite" in str(w.message) for w in caught), [str(w.message) for w in caught]

    def test_an_exclusion_that_empties_a_group_is_could_not_check(self):
        """REFUSAL PIN. When the exclusion leaves only one group there is
        nothing to compare, and the answer is None, never False."""
        components, system, groups = _emergent_case()
        system_group_lost = system.copy()
        system_group_lost[100:] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = EmergentBiasDetector().analyze(components, system_group_lost, groups)

        assert result.is_emergent is None, f"claimed {result.is_emergent!r}"
        assert result.is_emergent is not False, "could-not-check collapsed into no-bias"
        assert result.is_significant is None
        assert math.isnan(result.amplification_factor), result.amplification_factor
        assert math.isnan(result.system_bias) and math.isnan(result.p_value)
        assert result.n_samples_unmeasurable == 100
        assert any("could not be measured" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]

    def test_the_bootstrap_never_sees_an_unmeasurable_output(self):
        """A resample carrying the NaN would put it in the percentile and in
        `boot_bias <= max_component_bias`, where it reads as evidence FOR
        emergence. The p-value must come from finite draws only."""
        components, system, groups = _emergent_case()
        system_one_nan = system.copy()
        system_one_nan[42] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EmergentBiasDetector().analyze(components, system_one_nan, groups)

        assert math.isfinite(result.p_value), result.p_value
        assert result.is_significant is True
        assert result.metadata.parameters["n_bootstrap_measured"] > 0


class TestEmergentBiasStillAnswersOnMeasuredInput:
    """OVER-CORRECTION CONTROL, both directions, on fully finite input."""

    def test_a_clean_separating_system_is_still_emergent(self):
        components, system, groups = _emergent_case()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = EmergentBiasDetector().analyze(components, system, groups)

        # This pin's subject is UNMEASURABLE INPUT: on fully finite input the
        # detector must not warn about exclusions or unmeasured quantities.
        # READINESS-6 added one warning that is not about the input at all: the
        # amplification RATIO is undefined here because every component measured
        # a bias of exactly 0, and the field carries nan instead of the old 1e6
        # sentinel, so the reason has to reach the caller.
        assert not [
            w
            for w in caught
            if "not finite" in str(w.message) or "could not be measured" in str(w.message)
        ], [str(w.message) for w in caught]
        assert [w for w in caught if "amplification RATIO" in str(w.message)]
        assert result.system_bias == 1.0
        assert result.max_component_bias == 0.0
        # READINESS-6: the ratio over that zero baseline is undefined, not 1e6.
        assert math.isnan(result.amplification_factor)
        assert result.is_emergent is True
        assert result.is_significant is True
        assert result.p_value == 0.0
        assert result.n_samples_unmeasurable == 0
        assert result.n_components_measured == 2

    def test_a_system_no_worse_than_its_parts_is_still_not_emergent(self):
        """False must remain reachable: this is a measured absence of
        emergence, not a could-not-check."""
        groups = np.array([0] * 100 + [1] * 100)
        shared = np.concatenate([np.full(100, 0.72), np.full(100, 0.28)])
        components = {"a": shared.copy(), "b": shared.copy()}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = EmergentBiasDetector().analyze(components, shared.copy(), groups)

        assert not caught, [str(w.message) for w in caught]
        assert result.system_bias == pytest.approx(0.44)
        assert result.max_component_bias == pytest.approx(0.44)
        assert result.amplification_factor == pytest.approx(1.0)
        assert result.is_emergent is False, f"answered {result.is_emergent!r}"
        assert result.n_samples_unmeasurable == 0

    def test_excluding_samples_leaves_the_measured_arithmetic_alone(self):
        """The exclusion must be an exclusion, not a reweighting: dropping a
        NaN sample gives the same group means as never sending it."""
        groups = np.array([0] * 100 + [1] * 100)
        components = {"a": np.full(200, 0.5)}
        system = np.concatenate([np.full(100, 0.9), np.full(100, 0.3)])

        with_nan = system.copy()
        with_nan[7] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            excluded = EmergentBiasDetector().analyze(components, with_nan, groups)

        keep = np.ones(200, dtype=bool)
        keep[7] = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            never_sent = EmergentBiasDetector().analyze(
                {"a": components["a"][keep]}, system[keep], groups[keep]
            )

        # Same narrowing as above: the only warning on this finite input is the
        # undefined-ratio one (these components carry no bias at all), and it is
        # a statement about the ratio, not about the samples.
        assert not [
            w
            for w in caught
            if "not finite" in str(w.message) or "could not be measured" in str(w.message)
        ], [str(w.message) for w in caught]
        assert excluded.system_bias == pytest.approx(never_sent.system_bias, rel=1e-12)
        assert excluded.max_component_bias == pytest.approx(
            never_sent.max_component_bias, rel=1e-12
        )
        # nan_ok: these components carry no bias at all, so the ratio is
        # undefined on both sides (READINESS-6). The point of the comparison is
        # that excluding a sample and never sending it agree, and two nans
        # agree here in the only way an undefined ratio can.
        assert excluded.amplification_factor == pytest.approx(
            never_sent.amplification_factor, rel=1e-12, nan_ok=True
        )
        assert excluded.is_emergent is never_sent.is_emergent is True
        assert excluded.system_bias == pytest.approx(0.6, abs=1e-12)
