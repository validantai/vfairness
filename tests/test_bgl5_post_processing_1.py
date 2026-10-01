"""BGL5, batch A-post_processing-1: the nine overturned grades, closed.

An independent audit overturned nine grades in this batch on 2026-09-27 and
recorded each one as a strict xfail in ``tests/test_bgl4_post_processing_1.py``.
This file pins the corrected behaviour, with an over-correction control beside
every refusal, because a fix that refuses everything passes every refusal test
and destroys the library. Every number quoted below was printed by running the
unit, before and after.

ONE MECHANISM CARRIED FOUR OF THE NINE. Builtin ``max`` and ``min`` do not
propagate a NaN that is not the first element of the iterable:

    max({'A': 0.5167, 'B': nan}.values()) -> 0.5167
    min({'A': 0.5167, 'B': nan}.values()) -> 0.5167
    the spread                            -> 0.0, and math.isfinite says True

so a between-group disparity in which one of the two groups was never measured
came back as a FINITE 0.0, which is PERFECT PARITY on every scale in
``post_processing/calibration/tradeoffs.py``, and it bypassed every
``math.isfinite`` guard the earlier waves of this campaign added, because the
value those guards inspect is already finite. Measured on 240 rows in two 120-row
groups where group B carries no label at all:

    before  base_rate_disparity 0.0, tradeoff_severity "minimal",
            impossibility_applies False, base_rates_differ False,
            calibration_gap_disparity 0.0, conflict_exists False,
            ece_disparity 0.0, strategy "monitor_only" at priority "low",
            first recommendation "GOOD NEWS: Base rates are similar across
            groups", and not one warning about any of it.
    after   base_rate_disparity nan, tradeoff_severity "not assessed",
            impossibility_applies None, base_rates_differ None,
            calibration_gap_disparity nan, conflict_exists None,
            disparity_assessed False, a NOT ASSESSED recommendation in place of
            the GOOD NEWS one, and a warning naming group B.

``_measured_spread`` is the one helper all four sites now use. The mechanism
itself is pinned in ``tests/test_readiness6_names.py::
test_python_max_really_does_swallow_nan`` and again here, over a MAPPING, which
is the shape this file's defect took.

The other five:

PP1-A2  ``calibration_vs_error_parity`` built every disparity over
        ``GroupManager(min_group_size=1)``. On two 100-row groups with equal base
        rates and a TPR gap of 1.00 it answered a measured conflict_exists False;
        appending a SINGLE row of a third group with label 1 turned that into a
        measured True with base_rate_disparity 0.5 and ``not_assessed`` empty. Its
        sibling ``impossibility_diagnostics`` was given a floor of 30 for exactly
        this. Now: the one-row group is excluded and NAMED in
        ``excluded_groups``, and both calls answer the same measured False.

PP1-A4  ``mitigation_pareto`` gated its fairness axis on ROW count, so a group
        that was never scored cleared it and ``(p >= thr)`` decided all 100 of its
        unscored rows as rejections. Before, on two 100-row groups with group B
        unscored: available True, baselineGap 0.5, bestGap 0.25 and the summary
        "The selection-rate gap can be cut from 50 to 25 points via "Group
        thresholds (equal selection rate)" at 12.5 points of accuracy cost".
        After: available False, 1 of 2 group(s) qualify, nRowsUnscored 100.

PP1-A6  ``ThresholdAnalyzer`` never validated y_prob and each of
        ``analyze_threshold``, ``find_optimal_threshold`` and
        ``find_feasible_region`` thresholds it in its own body, so one validation
        at construction closes all three. Before, with group B unscored:
        demographic_parity violation 0.5 at threshold 0.5 with group_metrics
        {'A': 0.5, 'B': 0.0} and ``unmeasured=()``; find_feasible_region status
        "feasible" (0.9405, 0.99) with n_feasible 6 and n_not_assessed 0;
        find_optimal_threshold optimal_threshold 0.9405, is_feasible True,
        n_thresholds_not_assessed 0. The counters actively asserted that nothing
        was unmeasurable.

PP1-A8  ``ThresholdAnalysisReport.to_dict`` handed the FeasibleRegion objects over
        unchanged. A FeasibleRegion is a tuple subclass, so a NOT ASSESSED region
        and a MEASURED INFEASIBLE one both encoded to ``[null, null]`` and
        ``assessed`` / ``contiguous`` / ``n_feasible`` / ``n_searched`` /
        ``n_not_assessed`` were dropped at the boundary a consumer reads.

PP1-A9  ``ReweightingAnalyzer`` coerced labels with ``.astype(int)``, which turns
        a MISSING label into a real one (NaN casts to 0 here, to INT_MIN
        elsewhere) behind one numpy cast warning. On 160 rows of which 40 carried
        no label: accuracy 0.75 with n_decided 160 and n_rows 160, and
        original_ece 0.2 over original_ece_rows_used 160, so the coverage fields
        asserted all 160 rows had been graded. Accuracy over the 120 labelled rows
        is 1.0.

STILL OPEN, and deliberately not pinned as fixed here: PP1-A7, the reader-facing
"3 constraint(s) are satisfiable within the threshold range" that
``ThresholdAnalyzer.get_explanation`` prints above its own NOT ASSESSED line. The
count is ``len(report.feasible_regions)`` inside
``_explain_threshold_analysis`` in ``src/vfairness/explainer.py``, which this
batch does not own. It stays a strict xfail in the bgl4 file.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import List

import numpy as np
import pytest

from vfairness.exceptions import InvalidDataError
from vfairness.post_processing.calibration.tradeoffs import (
    _measured_spread,
    analyze_calibration_fairness_tradeoff,
    calibration_vs_error_parity,
    impossibility_diagnostics,
    mitigation_pareto,
    recommend_calibration_strategy,
)
from vfairness.post_processing.reweighting.analyzer import ReweightingAnalyzer
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalysisReport,
    ThresholdAnalyzer,
)


def _messages(caught) -> List[str]:
    return [str(w.message) for w in caught]


def _named(caught, fragment: str) -> List[str]:
    return [m for m in _messages(caught) if fragment in m]


# ===========================================================================
# Fixtures. Deterministic, no rng, so every number in the docstrings is
# reproducible from the arrays alone.
# ===========================================================================


def _unlabelled_second_group(n_per_group: int = 120):
    """Group A labelled, group B carrying no ground truth at all.

    Both groups clear the 30-row size gate, so this is NOT the
    fewer-than-two-groups path any earlier wave covered: two groups are
    compared and one of them has nothing to contribute.
    """
    labels_a = np.tile(np.array([1.0, 0.0, 1.0, 1.0, 0.0]), n_per_group // 5)
    y = np.concatenate([labels_a, np.full(n_per_group, np.nan)])
    p = np.concatenate([np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n_per_group // 5)] * 2)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _both_groups_labelled(n_per_group: int = 120):
    """The over-correction control for every unlabelled-group pin.

    The same shape and the same scores, with group B labelled too: base rates
    0.6 and 0.2, so the measured gap is 0.4 and every refusal below has to give
    way to a number.
    """
    labels_a = np.tile(np.array([1.0, 0.0, 1.0, 1.0, 0.0]), n_per_group // 5)
    labels_b = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), n_per_group // 5)
    y = np.concatenate([labels_a, labels_b])
    p = np.concatenate([np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n_per_group // 5)] * 2)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _unscored_second_group(n_per_group: int = 100):
    """Group A scored across the grid, group B never scored."""
    scored = np.linspace(0.02, 0.98, n_per_group)
    p = np.concatenate([scored, np.full(n_per_group, np.nan)])
    y = np.concatenate([np.ones(n_per_group, dtype=int), np.zeros(n_per_group, dtype=int)])
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _non_overlapping_scores(n_per_group: int = 100):
    """Two fully scored groups whose scores do not overlap: a real 1.0 gap."""
    p = np.concatenate([np.linspace(0.55, 0.95, n_per_group), np.linspace(0.05, 0.45, n_per_group)])
    y = (p >= 0.5).astype(int)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _one_group_two_hundred():
    p = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 40)
    y = np.tile(np.array([0, 0, 1, 1, 1]), 40)
    return y, p, np.array(["A"] * 200)


# ===========================================================================
# The mechanism, and the helper that replaced it.
# ===========================================================================


class TestTheNanSwallowingSpread:
    def test_the_builtins_really_do_swallow_a_trailing_nan(self):
        """The whole reason four isfinite guards could not fire. Pinned over a
        MAPPING, which is the shape these four sites used."""
        values = {"A": 0.5167, "B": float("nan")}
        swallowed = max(values.values()) - min(values.values())

        assert swallowed == 0.0
        assert math.isfinite(swallowed), (
            "if this ever propagates the NaN, the four fixes below are belt and "
            "braces rather than the fix, and the comment in _measured_spread is wrong"
        )

    def test_the_helper_refuses_a_spread_it_cannot_measure(self):
        spread, measured, unmeasured = _measured_spread({"A": 0.5167, "B": float("nan")})

        assert not math.isfinite(spread)
        assert measured == ["A"]
        assert unmeasured == ["B"]

    def test_the_helper_refuses_an_infinite_value_too(self):
        spread, measured, unmeasured = _measured_spread({"A": 0.2, "B": float("inf")})

        assert not math.isfinite(spread)
        assert unmeasured == ["B"]

    def test_control_the_helper_measures_what_it_can(self):
        """Over-correction control: two measured values still give their spread,
        and a third unmeasured one does not delete it."""
        spread, measured, unmeasured = _measured_spread({"A": 0.2, "B": 0.7})
        assert spread == pytest.approx(0.5, abs=1e-12)
        assert (measured, unmeasured) == (["A", "B"], [])

        spread3, measured3, unmeasured3 = _measured_spread({"A": 0.2, "B": 0.7, "C": float("nan")})
        assert spread3 == pytest.approx(0.5, abs=1e-12)
        assert (measured3, unmeasured3) == (["A", "B"], ["C"])


# ===========================================================================
# PP1-A1. The trade-off that graded an unmeasured base-rate comparison
# "minimal" and told the reader it was good news.
# ===========================================================================


class TestTheTradeoffSeverityOfAComparisonNobodyMade:
    def test_an_unlabelled_group_is_not_a_minimal_tradeoff(self):
        """Before: base_rate_disparity 0.0, tradeoff_severity "minimal",
        recommendations[0] "GOOD NEWS: Base rates are similar across groups.
        Calibration and fairness can likely be improved together.", and no
        warning about the base-rate comparison."""
        y, p, g = _unlabelled_second_group()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(y, p, g)

        assert not math.isfinite(result.base_rate_disparity)
        assert result.tradeoff_severity == "not assessed"
        assert result.to_dict()["tradeoff_severity"] == "not assessed"
        assert not [r for r in result.recommendations if "GOOD NEWS" in r]
        assert [r for r in result.recommendations if "NOT ASSESSED: the base rates" in r]
        assert _named(caught, "have NO measured base rate")

    def test_the_unmeasured_comparison_is_not_reported_as_severe_either(self):
        """The other direction of the same two-state band. `nan < 0.05` is False
        and so is `nan < 0.15`, so once the disparity became NaN the last branch
        would have printed "SIGNIFICANT TRADE-OFF: Large base rate disparity
        (nan%)": a severe finding from the same absence."""
        y, p, g = _unlabelled_second_group()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_calibration_fairness_tradeoff(y, p, g)

        assert not [r for r in result.recommendations if "SIGNIFICANT TRADE-OFF" in r]
        assert not [r for r in result.recommendations if "MODERATE TRADE-OFF" in r]
        assert "nan" not in " ".join(result.recommendations).lower()

    def test_control_a_measured_gap_of_four_tenths_is_still_severe(self):
        """Over-correction control with the actual number: base rates 0.6 and
        0.2 give a measured 0.4, which is "severe", and the reader still gets
        the SIGNIFICANT TRADE-OFF line."""
        y, p, g = _both_groups_labelled()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(y, p, g)

        assert result.base_rate_disparity == pytest.approx(0.4, abs=1e-12)
        assert result.tradeoff_severity == "severe"
        assert result.impossibility_diagnosis["impossibility_applies"] is True
        assert result.recommendations[0].startswith("SIGNIFICANT TRADE-OFF")
        assert not _named(caught, "NO measured base rate")


# ===========================================================================
# PP1-A2. One row of a third group settling the impossibility verdict, and
# the two arms that collapsed to a finite 0.0.
# ===========================================================================


class TestCalibrationVsErrorParityNeedsGroupsWorthComparing:
    @staticmethod
    def _two_hundred_rows_two_groups():
        y = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] * 50 + [0] * 50)])
        p = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
        g = np.array(["A"] * 100 + ["B"] * 100)
        return y, p, g

    def test_one_row_of_a_third_group_does_not_move_the_verdict(self):
        """Before: the two-group call answered conflict_exists False with
        base_rate_disparity 0.0, and appending ONE row of a group C answered
        conflict_exists True with base_rate_disparity 0.5 and not_assessed []."""
        y, p, g = self._two_hundred_rows_two_groups()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            two = calibration_vs_error_parity(y, p, g)
            three = calibration_vs_error_parity(
                np.concatenate([y, [1]]),
                np.concatenate([p, [0.9]]),
                np.concatenate([g, ["C"]]),
            )

        assert two["conflict_exists"] is False
        assert three["conflict_exists"] is False
        assert three["base_rate_disparity"] == pytest.approx(two["base_rate_disparity"], abs=1e-12)
        # Excluded, and SAID so: silently dropping the row would answer the same
        # question with no way for a reader to know C was in the data.
        assert three["excluded_groups"] == ["C"]
        assert sorted(three["group_metrics"]) == ["A", "B"]
        assert _named(caught, "fewer than 30 samples and are EXCLUDED")

    def test_an_unlabelled_group_leaves_two_arms_unmeasured_and_says_which(self):
        """Before, on 240 rows with group B unlabelled: base_rate_disparity 0.0
        and calibration_gap_disparity 0.0 through max()-min() over a NaN,
        conflict_exists a measured False, not_assessed naming neither of them,
        and zero warnings."""
        y, p, g = _unlabelled_second_group()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, p, g)

        assert not math.isfinite(result["base_rate_disparity"])
        assert not math.isfinite(result["calibration_gap_disparity"])
        assert result["conflict_exists"] is None
        assert "base_rate_disparity" in result["not_assessed"]
        assert "calibration_gap_disparity" in result["not_assessed"]
        assert result["n_groups_compared"] == 1
        assert result["groups_without_measured_base_rate"] == ["B"]
        assert _named(caught, "conflict_exists is None (COULD NOT CHECK)")

    def test_control_a_real_gap_between_two_real_groups_is_measured(self):
        """Over-correction control with the actual numbers: base rates 0.6 and
        0.2, TPR gap 0.6667, FPR gap 0.25, calibration-gap spread 0.4, so the
        theorem binds and conflict_exists is a MEASURED True with nothing in
        not_assessed."""
        y, p, g = _both_groups_labelled()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, p, g)

        assert result["base_rate_disparity"] == pytest.approx(0.4, abs=1e-12)
        assert result["tpr_disparity"] == pytest.approx(2.0 / 3.0, abs=1e-12)
        assert result["fpr_disparity"] == pytest.approx(0.25, abs=1e-12)
        assert result["calibration_gap_disparity"] == pytest.approx(0.4, abs=1e-12)
        assert result["conflict_exists"] is True
        assert result["not_assessed"] == []
        assert result["excluded_groups"] == []
        assert result["n_groups_compared"] == 2
        assert not _named(caught, "COULD NOT CHECK")

    def test_control_the_caller_can_still_lower_the_gate_it_wants(self):
        """The gate is an argument, not a wall: a caller who says 10 gets the
        10-row group measured, so the fix has not taken the small-cohort case
        away from anybody who asks for it."""
        y, p, g = self._two_hundred_rows_two_groups()
        y10 = np.concatenate([y, np.array([1] * 6 + [0] * 4)])
        p10 = np.concatenate([p, np.full(10, 0.9)])
        g10 = np.concatenate([g, np.array(["C"] * 10)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibration_vs_error_parity(y10, p10, g10, min_group_size=10)

        assert result["n_groups_compared"] == 3
        assert result["excluded_groups"] == []
        assert result["base_rate_disparity"] == pytest.approx(0.1, abs=1e-12)


# ===========================================================================
# PP1-A3. The theorem verdict from a group with no measured base rate.
# ===========================================================================


class TestTheImpossibilityGateCountsMeasuredRates:
    def test_a_group_with_no_measured_base_rate_is_not_an_equal_base_rate(self):
        """Before: base_rates {'A': 0.6, 'B': nan}, base_rate_disparity 0.0,
        base_rates_differ False, impossibility_applies False,
        calibration_sacrifice_estimate 0.0, n_groups_compared 2, ZERO warnings,
        and the explanation "Base rates are approximately equal across groups. In
        this case, calibration and error rate parity can theoretically be
        achieved simultaneously."."""
        y, p, g = _unlabelled_second_group()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = impossibility_diagnostics(y, p, g)

        assert not math.isfinite(diagnosis["base_rate_disparity"])
        assert diagnosis["base_rates_differ"] is None
        assert diagnosis["impossibility_applies"] is None
        assert not math.isfinite(diagnosis["calibration_sacrifice_estimate"])
        assert diagnosis["n_groups_compared"] == 1
        assert diagnosis["groups_without_measured_base_rate"] == ["B"]
        assert diagnosis["explanation"].startswith("NOT ASSESSED.")
        assert "approximately equal" not in diagnosis["explanation"]
        assert _named(caught, "no base-rate comparison was made")

    def test_a_measured_verdict_that_leaves_a_group_out_says_which(self):
        """The middle case, and the one a refusal-only fix would hide: THREE
        groups above the size gate, of which C carries no label. The verdict is a
        real measurement over A and B, so it must stand, and the reader has to be
        told it does not cover C."""
        y, p, g = _both_groups_labelled()
        y3 = np.concatenate([y, np.full(40, np.nan)])
        p3 = np.concatenate([p, np.full(40, 0.5)])
        g3 = np.concatenate([g, np.array(["C"] * 40)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = impossibility_diagnostics(y3, p3, g3)

        assert diagnosis["base_rate_disparity"] == pytest.approx(0.4, abs=1e-12)
        assert diagnosis["base_rates_differ"] is True
        assert diagnosis["impossibility_applies"] is True
        assert diagnosis["n_groups_compared"] == 2
        assert diagnosis["groups_without_measured_base_rate"] == ["C"]
        assert "C" in diagnosis["explanation"]
        assert _named(caught, "inside the size gate have NO measured base rate")

    def test_control_two_measured_base_rates_still_bind_the_theorem(self):
        """Over-correction control with the actual number: 0.6 against 0.2 is a
        measured 0.4, the theorem applies, and the sacrifice estimate is 0.2."""
        y, p, g = _both_groups_labelled()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = impossibility_diagnostics(y, p, g)

        assert diagnosis["base_rate_disparity"] == pytest.approx(0.4, abs=1e-12)
        assert diagnosis["base_rates_differ"] is True
        assert diagnosis["impossibility_applies"] is True
        assert diagnosis["calibration_sacrifice_estimate"] == pytest.approx(0.2, abs=1e-12)
        assert diagnosis["n_groups_compared"] == 2
        assert diagnosis["groups_without_measured_base_rate"] == []
        assert not _named(caught, "no base-rate comparison was made")


# ===========================================================================
# PP1-A4. A selection-rate gap measured over rows nobody scored.
# ===========================================================================


class TestMitigationParetoCountsScoredRows:
    def test_a_group_that_was_never_scored_is_not_a_selection_rate_gap(self):
        """Before: available True, baselineGap 0.5, bestGap 0.25 and the summary
        "The selection-rate gap can be cut from 50 to 25 points via "Group
        thresholds (equal selection rate)" at 12.5 points of accuracy cost", on
        100 rows decided as rejections because `nan >= thr` is False."""
        y, p, g = _unscored_second_group()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mitigation_pareto(y, p, g)

        assert result["available"] is False
        assert "carrying a score" in result["reason"]
        assert "NOT a finding that the gap is zero" in result["reason"]
        assert result["nGroups"] == 2, "the group that vanished with its rows is still counted"
        assert result["nGroupsMeasurable"] == 1
        assert result["nRowsUnscored"] == 100
        # No axis, no gap numbers, no actionable summary to misread.
        assert "baselineGap" not in result
        assert "summary" not in result

    def test_an_input_above_max_rows_is_subsampled_rather_than_refused(self):
        """Found while adding the filter above, and pinned because it is a
        behaviour CHANGE, not only a refusal: `n` did not follow the max_rows
        subsample, so `gt`/`eo` were allocated at the original length and the
        masked assignment raised. Measured with max_rows=50 on 200 rows: before,
        available False with reason "Trade-off unavailable: boolean index did not
        match indexed array along axis 0; size of axis is 200 but size of
        corresponding boolean axis is 50"; after, a measured frontier. Every input
        above max_rows (8000 by default) hit this."""
        y, p, g = _non_overlapping_scores()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mitigation_pareto(y, p, g, max_rows=50)

        assert result["available"] is True, result.get("reason")
        assert "Trade-off unavailable" not in str(result.get("reason"))
        assert result["baselineGap"] == pytest.approx(1.0, abs=1e-12)
        assert result["bestGap"] == pytest.approx(0.0338, abs=1e-4)

    def test_control_two_fully_scored_groups_still_get_their_frontier(self):
        """Over-correction control with the actual numbers: a measured baseline
        gap of 1.0 that group thresholds close to 0.0, and nRowsUnscored 0."""
        y, p, g = _non_overlapping_scores()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mitigation_pareto(y, p, g)

        assert result["available"] is True
        assert result["baselineGap"] == pytest.approx(1.0, abs=1e-12)
        assert result["bestGap"] == pytest.approx(0.0, abs=1e-12)
        assert result["nRowsUnscored"] == 0
        assert result["nGroupsMeasurable"] == 2
        assert "The selection-rate gap can be cut from 100 to 0 points" in result["summary"]
        assert "row(s) had none" not in result["summary"]

    def test_control_a_partly_scored_group_is_measured_over_its_scored_rows(self):
        """The half that matters most: refusing a group because SOME of its rows
        are unscored would delete a real finding. Group B keeps 60 of its 100
        rows, the measured baseline gap is still 1.0, and the 40 dropped rows are
        named in the summary and counted in nRowsUnscored."""
        y, p, g = _non_overlapping_scores()
        p = p.copy()
        p[160:] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mitigation_pareto(y, p, g)

        assert result["available"] is True
        assert result["nGroupsMeasurable"] == 2
        assert result["baselineGap"] == pytest.approx(1.0, abs=1e-12)
        assert result["nRowsUnscored"] == 40
        assert "Measured over the 160 row(s) that carry a score" in result["summary"]


# ===========================================================================
# PP1-A5. "Calibration metrics are acceptable" for a comparison that covered
# one group of two.
# ===========================================================================


class TestTheCalibrationStrategyKnowsWhatItCompared:
    def test_an_unlabelled_group_is_not_acceptable_calibration(self):
        """Before, on 400 rows with group B unlabelled: group ECEs
        {'A': 0.58, 'B': nan}, ece_disparity a finite 0.0 from max()-min(),
        excluded_groups [], strategy "monitor_only", priority "low",
        disparity_assessed True, not_assessed_groups [], rationale "Calibration
        metrics are acceptable. Monitor for drift but no immediate action
        needed.", zero warnings. Group A's own 0.58 is a real and poor
        measurement, reported as acceptable."""
        y, p, g = _unlabelled_second_group(n_per_group=200)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recommendation = recommend_calibration_strategy(y, p, g, context="lending")

        assert recommendation.disparity_assessed is False
        assert recommendation.priority != "low"
        assert "NOT ASSESSED" in recommendation.rationale
        assert recommendation.not_assessed_groups == ["B"]
        # The disclosure has to survive the serialiser a dashboard reads.
        payload = recommendation.to_dict()
        assert payload["disparity_assessed"] is False
        assert payload["not_assessed_groups"] == ["B"]
        assert _named(caught, "the between-group calibration disparity")

    def test_control_a_measured_disparity_still_gets_its_strategy(self):
        """Over-correction control: with both groups labelled, the lending
        context returns group_specific_isotonic at high priority,
        disparity_assessed True, no NOT ASSESSED text and no warning."""
        y, p, g = _both_groups_labelled(n_per_group=200)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recommendation = recommend_calibration_strategy(y, p, g, context="lending")

        assert recommendation.strategy == "group_specific_isotonic"
        assert recommendation.priority == "high"
        assert recommendation.disparity_assessed is True
        assert recommendation.not_assessed_groups == []
        assert "NOT ASSESSED" not in recommendation.rationale
        assert not _named(caught, "the between-group calibration disparity")


# ===========================================================================
# PP1-A6. Three methods thresholding an unscored row in their own bodies, and
# one validation above all of them.
# ===========================================================================


class TestTheThresholdAnalyzerRefusesAnUnscoredRow:
    def test_an_unscored_row_is_refused_at_construction(self):
        """Before: the analyzer was built happily and every method decided the
        unscored rows as rejections. The guard is at construction because
        analyze_threshold, find_optimal_threshold and find_feasible_region each
        threshold y_prob in their OWN body."""
        y, p, g = _unscored_second_group()

        with pytest.raises(InvalidDataError, match="100 NaN value"):
            ThresholdAnalyzer(y, p, g)

    def test_the_refusal_covers_every_method_that_thresholds_the_score(self):
        """No object, so no method can answer: the point of putting it above the
        dispatch rather than inside one of them."""
        y, p, g = _unscored_second_group()
        p_one_missing = p.copy()
        p_one_missing[100:] = 0.5
        p_one_missing[7] = np.nan

        # A SINGLE unscored row is enough, because a single fabricated rejection
        # is still a decision nobody made.
        with pytest.raises(InvalidDataError, match="1 NaN value"):
            ThresholdAnalyzer(y, p_one_missing, g)

    def test_control_a_fully_scored_sweep_still_measures_its_region(self):
        """Over-correction control with the actual numbers on two fully scored
        100-row groups whose scores do not overlap: 13 of 100 searched thresholds
        satisfy demographic_parity, the pair is (0.010, 0.990) and NOT
        contiguous, the optimum is 0.9306 and is_feasible is a measured True."""
        y, p, g = _non_overlapping_scores()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            region = analyzer.find_feasible_region(
                constraint="demographic_parity", tolerance=0.05, n_thresholds=100
            )
            optimum = analyzer.find_optimal_threshold(constraint="demographic_parity")
            single = analyzer.analyze_threshold(0.5)

        assert region[0] == pytest.approx(0.01, abs=1e-12)
        assert region[1] == pytest.approx(0.99, abs=1e-12)
        assert region.n_feasible == 13
        assert region.n_searched == 100
        assert region.contiguous is False
        assert optimum["is_feasible"] is True
        assert optimum["optimal_threshold"] == pytest.approx(0.9306060606060605, abs=1e-12)
        assert optimum["n_thresholds_not_assessed"] == 0
        violation = single.constraint_violations["demographic_parity"]
        assert violation.violation == pytest.approx(1.0, abs=1e-12)
        assert violation.group_metrics == {"A": pytest.approx(1.0), "B": pytest.approx(0.0)}


# ===========================================================================
# PP1-A8. The third state dropped at the serialisation boundary.
# ===========================================================================


class TestTheSerialisedRegionKeepsItsThirdState:
    def test_not_assessed_and_measured_infeasible_no_longer_encode_alike(self):
        """Before: both were ``[null, null]`` once encoded, because a
        FeasibleRegion is a tuple subclass, so the flag that separates "nothing
        could be checked" from "every threshold was checked and none worked" was
        dropped exactly where a consumer reads it."""
        report = ThresholdAnalysisReport(
            threshold_results=[],
            optimal_thresholds={},
            feasible_regions={
                "unassessed": FeasibleRegion(
                    None, None, assessed=False, n_searched=100, n_not_assessed=100
                ),
                "infeasible": FeasibleRegion(None, None, assessed=True, n_searched=100),
            },
            recommendations=[],
        )

        regions = report.to_dict()["feasible_regions"]

        assert json.dumps(regions["unassessed"]) != json.dumps(regions["infeasible"])
        assert regions["unassessed"]["status"] == "not_assessed"
        assert regions["infeasible"]["status"] == "infeasible"
        assert regions["unassessed"]["n_not_assessed"] == 100

    def test_the_real_report_serialises_its_not_assessed_region(self):
        """The same thing through the public path: 200 rows of one group, where
        no threshold can be checked at all."""
        y, p, g = _one_group_two_hundred()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(n_thresholds=5, constraints=["demographic_parity"])

        region = report.to_dict()["feasible_regions"]["demographic_parity"]

        assert region["status"] == "not_assessed"
        assert region["assessed"] is False
        assert json.loads(json.dumps(region))["status"] == "not_assessed"
        # The attribute keeps the pair interface every existing caller uses.
        low, high = report.feasible_regions["demographic_parity"]
        assert (low, high) == (None, None)
        assert report.feasible_regions["demographic_parity"] == (None, None)

    def test_a_bare_pair_serialises_as_unknown_rather_than_assessed(self):
        """Fail closed on a shape that carries no third state. A plain tuple says
        nothing about which of the three states it is in, and the sibling default
        in `_generate_recommendations._assessed` is on the fabricated-verdict
        ledger for exactly this reason: it used to read `else True`."""
        report = ThresholdAnalysisReport(
            threshold_results=[],
            optimal_thresholds={},
            feasible_regions={"plain": (None, None)},
            recommendations=[],
        )

        region = report.to_dict()["feasible_regions"]["plain"]

        assert region["status"] == "unknown"
        assert "assessed" not in region

    def test_control_a_measured_region_still_serialises_its_endpoints(self):
        """Over-correction control with the actual numbers: the feasible region
        encodes lower 0.01, upper 0.99, status "feasible" and n_feasible 13."""
        y, p, g = _non_overlapping_scores()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(n_thresholds=5, constraints=["demographic_parity"])

        region = report.to_dict()["feasible_regions"]["demographic_parity"]

        assert region["lower"] == pytest.approx(0.01, abs=1e-12)
        assert region["upper"] == pytest.approx(0.99, abs=1e-12)
        assert region["status"] == "feasible"
        assert region["n_feasible"] == 13
        assert region["contiguous"] is False


# ===========================================================================
# PP1-A9. A row nothing labelled, graded anyway.
# ===========================================================================


class TestTheReweightingAnalyzerDoesNotGradeAnUnlabelledRow:
    @staticmethod
    def _one_hundred_sixty_rows(label_the_last_forty: bool):
        scores = np.concatenate([np.full(60, 0.9), np.full(60, 0.1), np.full(40, 0.5)])
        tail = np.zeros(40) if label_the_last_forty else np.full(40, np.nan)
        y = np.concatenate([np.ones(60), np.zeros(60), tail])
        g = np.array(["A"] * 60 + ["B"] * 60 + ["C"] * 40)
        return y, scores, g

    def test_a_row_with_no_label_is_not_graded(self):
        """Before: accuracy 0.75 over n_decided 160 and n_rows 160, and
        original_ece 0.2 over original_ece_rows_used 160, because NaN cast to the
        label 0 and the 40 unlabelled rows were then graded as wrong. Accuracy
        over the 120 rows that DO carry a label is 1.0."""
        y, p, g = self._one_hundred_sixty_rows(label_the_last_forty=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = ReweightingAnalyzer(y, p, g)
            result = analyzer.analyze_method("multiplicative")

        performance = result.original_performance
        assert performance["accuracy"] == pytest.approx(1.0, abs=1e-12)
        assert performance["n_decided"] == 120
        assert performance["n_rows"] == 120
        assert performance["n_rows_unlabelled_excluded"] == 40
        assert result.calibration_metrics["original_ece_rows_used"] == 120
        assert analyzer.unique_groups == ["A", "B"]
        assert _named(caught, "carry no ground-truth label and are EXCLUDED")

    def test_the_excluded_count_reaches_the_report_a_reader_gets(self):
        """A correct exclusion that the report drops is the same defect one layer
        up, so the count is pinned on full_analysis metadata as well."""
        y, p, g = self._one_hundred_sixty_rows(label_the_last_forty=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = ReweightingAnalyzer(y, p, g).full_analysis()

        assert report.metadata["n_samples"] == 120
        assert report.metadata["n_rows_unlabelled_excluded"] == 40

    def test_control_the_same_rows_with_real_labels_are_graded(self):
        """Over-correction control with the actual number, and the one that
        matters here: label those 40 rows 0 and they ARE wrong, so the honest
        accuracy is the same 0.75 the defect fabricated, over 160 rows with
        nothing excluded."""
        y, p, g = self._one_hundred_sixty_rows(label_the_last_forty=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = ReweightingAnalyzer(y, p, g)
            result = analyzer.analyze_method("multiplicative")

        performance = result.original_performance
        assert performance["accuracy"] == pytest.approx(0.75, abs=1e-12)
        assert performance["n_decided"] == 160
        assert performance["n_rows"] == 160
        assert performance["n_rows_unlabelled_excluded"] == 0
        assert analyzer.unique_groups == ["A", "B", "C"]
        assert not _named(caught, "carry no ground-truth label")
