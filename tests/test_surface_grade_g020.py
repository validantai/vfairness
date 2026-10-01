"""G-20: the drift detector must not grade readings nobody took.

``FairnessDriftDetector`` forward-fills gaps before any test runs
(:meth:`~vfairness.operations.monitoring.drift.FairnessDriftDetector.decompose_temporal_patterns`),
so a window arrives holding one observation and reaches the KS test holding
sixty samples. Nothing downstream can tell a filled copy from a reading.
Measured on 2026-09-17, before the fixes pinned here::

    detect_drift_multiscale(pd.Series([0.10] + [NaN] * 59))
        -> overall_drift_score 0.0, drift_detected False, 0 warnings
           worst scale reference_n 30, current_n 30

    detect_drift_multiscale(pd.Series([0.10] + [NaN]*29 + [0.90] + [NaN]*29))
        -> overall_drift_score 0.9950, drift_detected True, 0 warnings
           reference_mean 0.1000, current_mean 0.9000, n 30 and 30

    check_drift(current = [0.90] + [NaN] * 59) against a 60-point baseline
        -> overall_drift_score 0.9950, drift_detected True, current_n 60

The first is the neutral default this audit exists to find: "no drift" over a
series holding ONE observation, which ``generate_drift_report`` turns into
"continue routine monitoring". The second and third are the same fabrication
pointing the other way, a maximally confident verdict from one or two readings,
with a sample count of sixty printed beside it.

Three smaller siblings are pinned here too, all measured the same day::

    run_sprt(stream, null_value=0.1, alternative_value=0.1)
        -> ('continue', 10, 0.0)      0 warnings
    run_sprt([0.1, 0.2, nan, 0.3, 0.4] * 8, 0.1, 0.3)
        -> ('continue', 40, nan)      0 warnings
    detect_drift_ks(series, split_index=2)   # 60 points, 0.10 -> 0.30
        -> mean_shift +0.108496, drift_score nan, drift_detected False,
           0 warnings

``'continue'`` is the answer that means "the evidence is still accumulating,
keep collecting". Neither of those streams can ever accumulate evidence: two
identical hypotheses are not a comparison, and a NaN makes every boundary
comparison False forever.

Every refusal below has a CONTROL beside it computing a value worked out by
hand, because a detector that refuses everything passes each refusal pin and
finds nothing.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.drift import FairnessDriftDetector


def _catch(fn, *args, **kwargs) -> Tuple[Any, List[str]]:
    """Run *fn* with warnings ON, returning its value and the UserWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _shifted_series(n_each: int = 60) -> pd.Series:
    """A stable window followed by a real, findable 0.10 -> 0.30 shift."""
    rng = np.random.default_rng(20)
    return pd.Series(
        np.concatenate([rng.normal(0.10, 0.01, n_each), rng.normal(0.30, 0.01, n_each)])
    )


# ---------------------------------------------------------------------------
# detect_drift_multiscale: filled copies are not observations
# ---------------------------------------------------------------------------


class TestManufacturedReadingsAreNotGraded:
    def test_one_observation_is_not_a_measured_absence_of_drift(self):
        """The 0.0 that means "we looked and it is stable"."""
        detector = FairnessDriftDetector()
        result, warned = _catch(
            detector.detect_drift_multiscale,
            pd.Series([0.10] + [np.nan] * 59),
            metric="demographic_parity",
        )

        assert result.drift_detected is None, (
            f"a window holding ONE observation was graded {result.drift_detected!r}"
        )
        assert result.drift_detected is not False
        assert math.isnan(result.overall_drift_score), result.overall_drift_score
        assert result.scales == {}
        assert any("COULD NOT CHECK" in w for w in warned), warned
        assert any("forward-filled" in w for w in warned), warned

    def test_two_observations_are_not_a_confident_drift_verdict(self):
        """The same fabrication pointing the other way: 0.9950 from 2 readings."""
        detector = FairnessDriftDetector()
        series = pd.Series([0.10] + [np.nan] * 29 + [0.90] + [np.nan] * 29)
        result, warned = _catch(
            detector.detect_drift_multiscale, series, metric="demographic_parity"
        )

        assert result.drift_detected is None, (
            f"two readings and 58 copies were graded {result.drift_detected!r}"
        )
        assert math.isnan(result.overall_drift_score)
        assert any("COULD NOT CHECK" in w for w in warned), warned

    def test_check_drift_refuses_a_current_window_of_copies(self):
        """The public entry point, and the shape a monitor actually hits: the
        metric stopped being computable, so the window is one reading and 59
        gaps."""
        rng = np.random.default_rng(21)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        result, warned = _catch(
            detector.check_drift,
            pd.Series([0.90] + [np.nan] * 59),
            metric="demographic_parity",
        )

        assert result.drift_detected is None, result.drift_detected
        assert math.isnan(result.overall_drift_score)
        assert result.mmd_score is None, "an MMD score was computed from 59 manufactured copies"
        assert any("COULD NOT CHECK" in w for w in warned), warned

    def test_a_partly_filled_window_is_measured_but_says_it_was_filled(self):
        """OVER-CORRECTION CONTROL and disclosure pin in one. 30 real readings
        beside 30 gaps is still measurable: refusing it would throw away a real
        finding. What must not happen is reporting current_n = 60 in silence."""
        rng = np.random.default_rng(22)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        current = pd.Series(np.concatenate([rng.normal(0.30, 0.01, 30), np.full(30, np.nan)]))
        result, warned = _catch(detector.check_drift, current, metric="demographic_parity")

        assert result.drift_detected is True, "a real 0.10 -> 0.30 shift went unreported"
        assert result.overall_drift_score > 0.5
        assert any("forward-filled" in w for w in warned), warned
        assert any("NOT observations" in w for w in warned), warned
        # The count is still the grid, so the warning is the only thing that
        # distinguishes it. Pin that it names both numbers.
        assert any("30 of 60" in w for w in warned), warned

    def test_control_a_clean_shift_is_still_detected_unchanged(self):
        """OVER-CORRECTION CONTROL. No gaps anywhere: the guard must be inert."""
        rng = np.random.default_rng(23)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        result, warned = _catch(
            detector.check_drift, pd.Series(rng.normal(0.30, 0.01, 60)), metric="dp"
        )

        assert result.drift_detected is True
        assert result.overall_drift_score > 0.9
        assert warned == [], warned
        worst = result.worst_scale
        assert worst is not None
        assert (worst.reference_n, worst.current_n) == (60, 60)

    def test_control_a_clean_stable_window_still_reports_no_drift(self):
        """False must stay reachable: this is the measured "no drift"."""
        rng = np.random.default_rng(24)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        result, warned = _catch(
            detector.check_drift, pd.Series(rng.normal(0.10, 0.01, 60)), metric="dp"
        )

        assert result.drift_detected is False, result.drift_detected
        assert math.isfinite(result.overall_drift_score)
        assert warned == [], warned

    def test_a_boundary_the_caller_counted_before_the_gap_is_named(self):
        """The leading gap is dropped, so a supplied index lands elsewhere.
        Measured: 10 leading NaN with split_index=60 compared windows of 60 and
        40 instead of 50 and 50, and the reference mean carried 10 of the
        current window's points."""
        rng = np.random.default_rng(25)
        series = pd.Series(
            np.concatenate(
                [np.full(10, np.nan), rng.normal(0.10, 0.01, 50), rng.normal(0.50, 0.01, 50)]
            )
        )
        detector = FairnessDriftDetector()
        _, warned = _catch(detector.detect_drift_multiscale, series, metric="dp", split_index=60)
        assert any("leading reading(s) of the series are missing" in w for w in warned), warned


# ---------------------------------------------------------------------------
# detect_drift_ks: a lopsided split refused without a word
# ---------------------------------------------------------------------------


class TestKsRefusalsSaySo:
    def test_a_split_that_leaves_two_points_refuses_out_loud(self):
        detector = FairnessDriftDetector()
        result, warned = _catch(
            detector.detect_drift_ks, _shifted_series(30), metric="dp", split_index=2
        )

        assert math.isnan(result.drift_score)
        assert math.isnan(result.ks_statistic)
        assert any("COULD NOT CHECK" in w for w in warned), warned
        assert any("not 'no drift'" in w for w in warned), warned

    def test_a_boundary_counted_before_the_gaps_is_named_here_too(self):
        """detect_drift_ks drops EVERY missing reading, interior ones included,
        so a boundary the caller counted on the series they hold is applied to a
        shorter array and falls somewhere else."""
        rng = np.random.default_rng(37)
        values = list(rng.normal(0.10, 0.01, 30)) + list(rng.normal(0.30, 0.01, 30))
        values[3] = float("nan")
        values[11] = float("nan")
        detector = FairnessDriftDetector()
        _, warned = _catch(detector.detect_drift_ks, pd.Series(values), metric="dp", split_index=30)
        assert any("does not fall where you counted it" in w for w in warned), warned

    def test_control_the_midpoint_split_measures_an_exact_ks_statistic(self):
        """The expected value is worked out by hand, not copied from the code:
        the two halves have DISJOINT supports (0..9 against 5..14 overlaps by
        five, giving a maximum CDF gap of exactly 0.5), and the window means are
        4.5 and 9.5."""
        detector = FairnessDriftDetector()
        series = pd.Series([float(v) for v in range(10)] + [float(v) for v in range(5, 15)])
        result, warned = _catch(detector.detect_drift_ks, series, metric="dp")

        assert result.reference_n == 10 and result.current_n == 10
        assert result.ks_statistic == pytest.approx(0.5)
        assert result.reference_mean == pytest.approx(4.5)
        assert result.current_mean == pytest.approx(9.5)
        assert result.mean_shift == pytest.approx(5.0)
        assert warned == [], warned

    def test_control_fully_separated_halves_score_a_ks_of_one(self):
        detector = FairnessDriftDetector()
        series = pd.Series([v / 100.0 for v in range(20)] + [5.0 + v / 100.0 for v in range(20)])
        result, _ = _catch(detector.detect_drift_ks, series, metric="dp")

        assert result.ks_statistic == pytest.approx(1.0)
        assert result.drift_detected is True
        assert result.drift_score > 0.99


class TestBaselineWidthIsNotHidden:
    def test_a_baseline_that_shrank_says_by_how_much(self):
        """55 gaps among 60 readings become a FIVE point reference, and every
        later check compares against those five."""
        series = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5] + [np.nan] * 55)
        detector = FairnessDriftDetector()
        _, warned = _catch(detector.set_baseline, series)

        assert any("were dropped" in w for w in warned), warned
        assert any("5 point(s) wide" in w for w in warned), warned

    def test_control_a_clean_baseline_is_stored_without_a_word(self):
        rng = np.random.default_rng(34)
        detector = FairnessDriftDetector()
        series = pd.Series(rng.normal(0.10, 0.01, 60))
        _, warned = _catch(detector.set_baseline, series)
        assert warned == [], warned


class TestDecompositionInventsNothing:
    def test_a_series_with_no_readings_decomposes_to_nothing(self):
        """The fill must not become a value. fillna(0.0) here would hand the KS
        test sixty zeros and every scale would compare zeros against zeros."""
        detector = FairnessDriftDetector()
        components = detector.decompose_temporal_patterns(pd.Series([np.nan] * 60))

        assert list(components) == ["full_signal"], list(components)
        assert len(components["full_signal"]) == 0, components["full_signal"][:5]

    def test_the_filled_points_are_countable_from_outside(self):
        """A caller cannot see WHICH points are copies, but must be able to see
        THAT there are copies: the component is longer than the observations."""
        detector = FairnessDriftDetector()
        series = pd.Series([0.1] + [np.nan] * 59)
        components = detector.decompose_temporal_patterns(series)

        longest = max(len(c) for c in components.values())
        assert longest == 60
        assert int(series.notna().sum()) == 1
        assert longest > int(series.notna().sum())

    def test_control_a_clean_series_decomposes_into_full_length_scales(self):
        rng = np.random.default_rng(35)
        detector = FairnessDriftDetector()
        components = detector.decompose_temporal_patterns(pd.Series(rng.normal(0.1, 0.02, 64)))
        assert all(len(c) == 64 for c in components.values()), {
            k: len(v) for k, v in components.items()
        }
        assert any(np.ptp(c) > 0 for c in components.values())


class TestMultiscaleDictKeepsTheThirdState:
    def test_a_refused_run_serialises_as_none_and_nan(self):
        detector = FairnessDriftDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            payload = detector.detect_drift_multiscale(
                pd.Series([0.10] + [np.nan] * 59), metric="dp"
            ).to_dict()

        assert payload["drift_detected"] is None, payload["drift_detected"]
        assert payload["drift_detected"] is not False
        assert math.isnan(payload["overall_drift_score"])
        assert payload["mmd_score"] is None
        assert payload["scales"] == {}

    def test_control_a_measured_run_serialises_its_real_numbers(self):
        rng = np.random.default_rng(36)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            payload = detector.check_drift(
                pd.Series(rng.normal(0.40, 0.01, 60)), metric="dp"
            ).to_dict()

        assert payload["drift_detected"] is True
        assert payload["overall_drift_score"] > 0.9
        assert payload["scales"]["full_signal"]["comparison_ran"] is True


class TestScaleDictSaysWhetherAnythingRan:
    def test_a_refused_scale_is_marked_as_never_compared(self):
        """The per-scale dict is what generate_drift_report serialises into the
        JSON report, and it published "drift_detected": false for a scale no
        test ever touched."""
        detector = FairnessDriftDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            refused = detector.detect_drift_ks(
                _shifted_series(30), metric="dp", split_index=2
            ).to_dict()

        assert refused["comparison_ran"] is False, refused
        assert math.isnan(refused["drift_score"])

    def test_control_a_measured_scale_is_marked_as_compared(self):
        detector = FairnessDriftDetector()
        series = pd.Series([float(v) for v in range(10)] + [float(v) for v in range(5, 15)])
        measured = detector.detect_drift_ks(series, metric="dp").to_dict()

        assert measured["comparison_ran"] is True, measured
        assert measured["ks_statistic"] == pytest.approx(0.5)
        assert measured["mean_shift"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# run_sprt: 'continue' means the evidence is still accumulating
# ---------------------------------------------------------------------------


def _bare_detector() -> FairnessDriftDetector:
    """run_sprt touches no instance state; keep the fixture honest about that."""
    return FairnessDriftDetector()


class TestSprtThirdState:
    def test_two_identical_hypotheses_are_not_a_test_to_continue(self):
        decision, _, llr = _bare_detector().run_sprt(
            [0.1, 0.2, 0.3, 0.4, 0.5] * 8, null_value=0.1, alternative_value=0.1
        )
        assert decision == "could_not_check", decision
        assert decision != "continue"
        assert math.isnan(llr), "a likelihood ratio was reported for a test that cannot run"

    def test_the_identical_hypothesis_refusal_says_more_data_will_not_help(self):
        _, warned = _catch(
            _bare_detector().run_sprt,
            [0.1, 0.2, 0.3, 0.4, 0.5] * 8,
            null_value=0.1,
            alternative_value=0.1,
        )
        assert any("could_not_check" in w for w in warned), warned
        assert any("NOT 'continue'" in w for w in warned), warned

    def test_a_non_finite_hypothesis_is_refused(self):
        decision, _, llr = _bare_detector().run_sprt(
            [0.1, 0.2, 0.3, 0.4, 0.5] * 8, null_value=float("nan"), alternative_value=0.3
        )
        assert decision == "could_not_check"
        assert math.isnan(llr)

    def test_one_missing_reading_does_not_destroy_the_whole_test(self):
        """Measured before the fix: ('continue', 40, nan). np.std of a stream
        holding a NaN is NaN, every increment is NaN, and `nan >= upper` is
        False at every step, so the loop ran to the end and reported "no
        decision yet" for a test that was numerically destroyed."""
        rng = np.random.default_rng(26)
        stream = list(rng.normal(0.7, 0.05, 40))
        contaminated = list(stream)
        contaminated[7] = float("nan")

        (decision, stopping_n, llr), warned = _catch(
            _bare_detector().run_sprt,
            contaminated,
            null_value=0.5,
            alternative_value=0.7,
        )

        assert not (decision == "continue" and math.isnan(llr)), (
            "a destroyed test reported that its evidence is still accumulating"
        )
        assert decision in {"drift", "stable", "could_not_check"}
        assert math.isfinite(llr) or decision == "could_not_check"
        assert any("not finite" in w for w in warned), warned
        # And the 39 readable observations still decide, exactly as the clean
        # stream does. (This seed crosses the Wald lower boundary on its second
        # observation, so the decision is "stable"; what is pinned is that the
        # contamination does not change the answer, not which answer it is.)
        clean_decision, _, _ = _bare_detector().run_sprt(
            stream, null_value=0.5, alternative_value=0.7
        )
        assert clean_decision in {"drift", "stable"}, clean_decision
        assert decision == clean_decision
        assert stopping_n >= 1

    def test_control_a_real_stream_still_decides_both_ways(self):
        """OVER-CORRECTION CONTROL. A stream centred on the alternative must
        still say drift, and one centred on the null must still say stable."""
        rng = np.random.default_rng(27)
        drifted, _, llr_drift = _bare_detector().run_sprt(
            list(rng.normal(0.7, 0.05, 40)), null_value=0.5, alternative_value=0.7
        )
        assert drifted == "drift"
        assert math.isfinite(llr_drift)

        rng = np.random.default_rng(28)
        stable, _, llr_stable = _bare_detector().run_sprt(
            list(rng.normal(0.5, 0.05, 40)), null_value=0.5, alternative_value=0.7
        )
        assert stable == "stable"
        assert math.isfinite(llr_stable)


# ---------------------------------------------------------------------------
# detect_drift_mmd: the refusal path must not crash inside its own message
# ---------------------------------------------------------------------------


class TestMmdAlphaIsValidated:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5, float("nan")])
    def test_an_unusable_alpha_is_refused_loudly(self, alpha):
        """alpha=0.0 used to raise ZeroDivisionError from inside the f-string of
        the warning that was explaining a refusal; alpha >= 1 made `p_value <
        alpha` true for every possible permutation outcome, so the call reported
        drift whatever the data showed."""
        rng = np.random.default_rng(29)
        detector = FairnessDriftDetector()
        with pytest.raises(ValueError, match="alpha"):
            detector.detect_drift_mmd(
                rng.normal(0.0, 1.0, 30), rng.normal(3.0, 1.0, 30), alpha=alpha
            )

    def test_control_the_default_alpha_still_measures_both_answers(self):
        rng = np.random.default_rng(30)
        detector = FairnessDriftDetector()
        drift, score = detector.detect_drift_mmd(rng.normal(0.0, 1.0, 40), rng.normal(3.0, 1.0, 40))
        assert drift is True and math.isfinite(score)

        rng = np.random.default_rng(31)
        no_drift, score2 = detector.detect_drift_mmd(
            rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)
        )
        assert no_drift is False and math.isfinite(score2)


# ---------------------------------------------------------------------------
# get_history_df: the promised columns exist even with nothing in them
# ---------------------------------------------------------------------------


class TestHistoryFrameKeepsItsContract:
    EXPECTED = [
        "timestamp",
        "metric",
        "overall_drift_score",
        "drift_detected",
        "mmd_score",
    ]

    def test_an_empty_history_still_has_the_documented_columns(self):
        df = FairnessDriftDetector().get_history_df()
        assert list(df.columns) == self.EXPECTED, list(df.columns)
        assert len(df) == 0
        # The question a caller asks of an empty history must answer, not raise.
        assert df["drift_detected"].notna().sum() == 0

    def test_a_run_that_could_not_be_checked_is_still_in_the_history(self):
        """G-20. Every refusal returned before the append, so the history held
        only the runs that produced a verdict and a reader of
        ``get_history_df`` saw a monitoring record in which every check had
        succeeded."""
        rng = np.random.default_rng(33)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detector.check_drift(pd.Series([0.90] + [np.nan] * 59), metric="copies")
            detector.check_drift(pd.Series([np.nan] * 60), metric="empty")
            detector.check_drift(pd.Series([0.9, 0.9]), metric="too_short")

        assert len(detector.get_results_history()) == 3, "an attempted check went unrecorded"
        df = detector.get_history_df()
        assert set(df["metric"]) == {"copies", "empty", "too_short"}
        assert df["drift_detected"].isna().all()
        assert df["overall_drift_score"].isna().all()

    def test_a_could_not_check_run_is_stored_as_none_not_false(self):
        rng = np.random.default_rng(32)
        detector = FairnessDriftDetector()
        detector.set_baseline(pd.Series(rng.normal(0.10, 0.01, 60)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detector.check_drift(pd.Series(rng.normal(0.30, 0.01, 60)), metric="measured")
            detector.check_drift(pd.Series([0.9, 0.9]), metric="refused")

        df = detector.get_history_df()
        assert list(df.columns) == self.EXPECTED
        assert len(df) == 2
        refused = df[df["metric"] == "refused"].iloc[0]
        assert refused["drift_detected"] is None, refused["drift_detected"]
        assert math.isnan(refused["overall_drift_score"])
        measured = df[df["metric"] == "measured"].iloc[0]
        assert measured["drift_detected"] is True
        assert math.isfinite(measured["overall_drift_score"])

        # clear_history empties the store without losing the contract.
        detector.clear_history()
        assert detector.get_results_history() == []
        assert list(detector.get_history_df().columns) == self.EXPECTED
