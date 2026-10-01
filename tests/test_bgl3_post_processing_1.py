"""Batch bgl3 post_processing-1: the post-processing units that answered anyway.

Six defects were found BY EXECUTION on 2026-09-27 and fixed in
``post_processing/calibration/analyzer.py``,
``post_processing/calibration/tradeoffs.py`` and
``post_processing/threshold_optimization/analyzer.py``. Each number quoted below
was printed by running the unit, not read off the source.

PP1-DIAG  ``CalibrationAnalyzer.get_impossibility_diagnosis`` was the one
          delegating method that dropped the analyzer's own ``min_group_size``
          and used ``impossibility_diagnostics``' default of 30. On two 40-row
          groups built with ``min_group_size=50``, ``analyze_disparity()``
          excluded BOTH groups and returned NaN, while the diagnosis on the same
          object returned base_rate_disparity 0.625, base_rates_differ True,
          impossibility_applies True, ``excluded_groups`` [] and
          ``n_groups_compared`` 2, with no warning: the exclusion fields denied
          the exclusion the object had just made. It failed the other way too,
          refusing with ``min_group_size=5`` on two 10-row groups where
          ``analyze_tradeoffs()`` measured a base-rate gap of 0.70.

PP1-BRD   ``calibration_vs_error_parity`` computed ``base_rate_disparity`` as
          ``max(rates) - min(rates)`` with no two-group guard. On 200 rows of ONE
          group that is exactly 0.0, and zero base-rate disparity is the
          condition under which the impossibility theorem does not bind:
          measured base_rate_disparity 0.0, conflict_exists False, zero
          warnings, while ``tpr_disparity`` and ``fpr_disparity`` on the same
          call correctly answered NaN. This was a KNOWN blind spot of the static
          sweep for fabricated verdicts, recorded as an illustration on 2026-09-08
          and left live: the sweep matches neutral LITERALS, and here the zero is
          COMPUTED as max(base_rates) - min(base_rates) over one group, so no
          literal exists for a pattern to match and only execution finds it.
          ``conflict_exists`` was two-state as well: with base rates 0.60 vs
          1.00, a TPR gap of 0.0 and the FPR arm dead (group B had no negative
          rows), it read False, a measured "no conflict" over an arm nobody
          measured.

PP1-PAR   ``ParetoPoint.dominates`` returned a plain False in BOTH directions
          when a coordinate was NaN, and "nothing dominates it" is what puts a
          point ON the frontier. On 240 rows where group B held no negative
          labels, fpr_parity had one measurable rate, so all 17 swept thresholds
          carried fairness_violation NaN and all 17 came back as the frontier
          with ``is_pareto_optimal`` True and no warning. The same defect was
          fixed in the sibling frontier in ``operations/experimentation`` (audit
          6) and in ``MultiObjectiveThresholdOptimizer`` (g017). Separately,
          ``is_pareto_optimal`` defaulted to True and only ever had True written
          to it, so points the sweep had just found DOMINATED kept the claim.

PP1-MIT   ``mitigation_pareto`` measured its fairness axis with
          ``max(rates) - min(rates) if len(rates) >= 2 else 0.0`` over groups of
          at least 10 rows. On 200 rows of one group: available True,
          baselineGap 0.0, bestGap 0.0, fairnessGap 0.0 on all six points, and
          the summary asserted "the disparity is structural, not just a
          threshold artefact (consider data-level fixes)". 30 groups of ~6 rows
          produced the same, with contextDistortion 0.0 as well.

PP1-REC   ``recommend_calibration_strategy`` disclosed EXCLUDED groups (H-06)
          and not a disparity that never existed. With one group of 200 rows,
          ``calibration_disparity`` answers ece_disparity NaN and excludes
          nobody, every ``ece_disparity > x`` test is False for a NaN, and all
          eight branches fell through to strategy "monitor_only", priority "low",
          "Calibration metrics are acceptable. Monitor for drift but no
          immediate action needed.", ``not_assessed_groups`` [].

PP1-REG   ``find_feasible_region`` returns the min and max of the satisfying
          set, which READS as an interval. On two 100-row groups with scores in
          [0.55, 0.95] and [0.05, 0.45], demographic_parity at tolerance 0.05 is
          satisfied at 13 of 100 searched thresholds, indices 0 to 5 and 93 to
          99, the two degenerate ends where every row falls on one side of the
          cut. The pair returned was (0.01, 0.99), ``summary()`` printed
          "demographic_parity: [0.010, 0.990]", and threshold 0.5, inside it, was
          measured at a violation of 1.0. ``full_analysis(constraints=[])``
          examined nothing and closed with "Multiple constraints can be
          satisfied."

Every pin has a healthy control beside it, because refusing a value that IS
measurable is the same failure in the other direction. Two controls are load
bearing in particular: ``ThresholdAnalyzer`` on a single group already refused
correctly before this batch and must keep doing so, and the measured Pareto
frontier must stay at 3 points of 17 rather than 17, which is what the first
version of the PP1-PAR fix broke by testing ``dominates(...) is True`` on
np.bool_.
"""

from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np
import pytest

from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.post_processing.calibration.tradeoffs import (
    ParetoPoint,
    analyze_calibration_fairness_tradeoff,
    calibration_vs_error_parity,
    compute_pareto_frontier,
    mitigation_pareto,
    recommend_calibration_strategy,
)
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalyzer,
)


def _messages(caught) -> List[str]:
    return [str(w.message) for w in caught]


def _named(caught, fragment: str) -> List[str]:
    return [m for m in _messages(caught) if fragment in m]


# ===========================================================================
# Fixtures. Every one is deterministic: no rng, so the numbers in the
# docstrings above are reproducible from the arrays alone.
# ===========================================================================


def _two_forty_row_groups():
    """Two 40-row groups with base rates 0.75 and 0.125, gap 0.625."""
    y = np.concatenate([np.array([1] * 30 + [0] * 10), np.array([1] * 5 + [0] * 35)])
    p = np.concatenate([np.full(40, 0.7), np.full(40, 0.3)])
    g = np.array(["A"] * 40 + ["B"] * 40)
    return y, p, g


def _two_ten_row_groups():
    """Two 10-row groups with base rates 0.8 and 0.1, gap 0.7."""
    y = np.concatenate([np.array([1] * 8 + [0] * 2), np.array([1] * 1 + [0] * 9)])
    p = np.concatenate([np.full(10, 0.7), np.full(10, 0.3)])
    g = np.array(["A"] * 10 + ["B"] * 10)
    return y, p, g


def _two_eighty_row_groups():
    """The g005 hand-computed fixture: base rates 0.5 and 0.0, gap 0.5."""
    y = np.concatenate([np.array([1] * 40 + [0] * 40), np.zeros(80, dtype=int)])
    p = np.concatenate([np.full(40, 0.25), np.full(40, 0.75)] * 2)
    g = np.array(["A"] * 80 + ["B"] * 80)
    return y, p, g


def _one_group_two_hundred():
    """200 rows, one group. The score is the probability, so the overall ECE is
    small and genuinely measurable; only the BETWEEN-group comparison is not."""
    p = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 40)
    y = np.tile(np.array([0, 0, 1, 1, 1]), 40)
    g = np.array(["A"] * 200)
    return y, p, g


def _gapped_threshold_data():
    """Two 100-row groups whose scores do not overlap.

    Group A spans [0.55, 0.95], group B spans [0.05, 0.45], so any threshold
    between 0.45 and 0.55 selects all of A and none of B: a selection-rate gap
    of exactly 1.0. Demographic parity is satisfied only at the two degenerate
    ends of the grid, where both groups are almost entirely selected or almost
    entirely rejected.
    """
    p = np.concatenate([np.linspace(0.55, 0.95, 100), np.linspace(0.05, 0.45, 100)])
    y = (p >= 0.5).astype(int)
    g = np.array(["A"] * 100 + ["B"] * 100)
    return y, p, g


def _balanced_threshold_data():
    """Two groups with identical score distributions: every threshold satisfies
    demographic parity, so the satisfying set is the whole grid and contiguous."""
    y = np.array([1, 0] * 20)
    p = np.array([0.6, 0.4] * 20)
    g = np.array(["A"] * 20 + ["B"] * 20)
    return y, p, g


# ===========================================================================
# PP1-DIAG. The diagnosis that used a different gate from its own object.
# ===========================================================================


class TestTheImpossibilityDiagnosisUsesTheAnalyzersOwnGate:
    def test_a_gate_nothing_clears_is_not_a_theorem_verdict(self):
        """Before: base_rate_disparity 0.625, base_rates_differ True,
        impossibility_applies True, excluded_groups [], n_groups_compared 2, no
        warning, on an analyzer whose own ``analyze_disparity()`` had just
        excluded both groups and returned NaN."""
        y, p, g = _two_forty_row_groups()
        analyzer = CalibrationAnalyzer(y, p, g, min_group_size=50)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = analyzer.get_impossibility_diagnosis()

        assert diagnosis["base_rates_differ"] is None
        assert diagnosis["impossibility_applies"] is None
        assert not math.isfinite(diagnosis["base_rate_disparity"])
        # The exclusion fields must AGREE with the object's own gate.
        assert diagnosis["excluded_groups"] == ["A", "B"]
        assert diagnosis["n_groups_compared"] == 0
        assert "NOT ASSESSED" in diagnosis["explanation"]
        assert _named(caught, "no base-rate comparison was made")

    def test_it_does_not_refuse_the_comparison_the_caller_configured(self):
        """The same bug in the other direction: with ``min_group_size=5`` the
        diagnosis used to answer None / NaN while ``analyze_tradeoffs()`` on the
        same object measured a base-rate gap of 0.70 from the same two groups."""
        y, p, g = _two_ten_row_groups()
        analyzer = CalibrationAnalyzer(y, p, g, min_group_size=5, n_bins=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tradeoffs = analyzer.analyze_tradeoffs()
            diagnosis = analyzer.get_impossibility_diagnosis()

        assert diagnosis["n_groups_compared"] == 2
        assert diagnosis["base_rates_differ"] is True
        assert diagnosis["impossibility_applies"] is True
        assert diagnosis["base_rate_disparity"] == pytest.approx(0.7, abs=1e-9)
        # One object, one dataset, one answer.
        assert diagnosis["base_rate_disparity"] == pytest.approx(
            tradeoffs.base_rate_disparity, abs=1e-12
        )

    def test_control_the_default_gate_still_measures_a_real_gap(self):
        """Group A has 40 positives of 80 and group B none: the gap is 0.5."""
        y, p, g = _two_eighty_row_groups()
        analyzer = CalibrationAnalyzer(y, p, g, n_bins=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = analyzer.get_impossibility_diagnosis()

        assert diagnosis["base_rate_disparity"] == pytest.approx(0.5, abs=1e-12)
        assert diagnosis["base_rates_differ"] is True
        assert diagnosis["impossibility_applies"] is True
        assert diagnosis["n_groups_compared"] == 2
        assert not _named(caught, "no base-rate comparison was made")


# ===========================================================================
# PP1-BRD. A base-rate disparity of zero from one group, and the two-state
# conflict verdict built on top of it.
# ===========================================================================


class TestCalibrationVsErrorParityRefusesWhatItCannotCompare:
    def test_one_group_is_not_a_zero_base_rate_disparity(self):
        """Before: base_rate_disparity 0.0, conflict_exists False, zero
        warnings, beside tpr_disparity NaN and fpr_disparity NaN."""
        y, p, g = _one_group_two_hundred()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, p, g)

        assert not math.isfinite(result["base_rate_disparity"])
        assert not math.isfinite(result["calibration_gap_disparity"])
        assert result["conflict_exists"] is None
        assert result["n_groups_compared"] == 1
        assert "base_rate_disparity" in result["not_assessed"]
        assert _named(caught, "conflict_exists is None (COULD NOT CHECK)")
        # The overall calibration IS measurable on one group, and must not be
        # refused along with the comparison.
        assert math.isfinite(result["ece"])

    def test_a_dead_arm_is_not_a_measured_absence_of_conflict(self):
        """Base rates 0.60 and 1.00, TPR gap exactly 0.0, calibration gaps equal
        to within 4.4e-16, and group B has no negative rows so the FPR arm has
        one rate. Before: conflict_exists False, no warning. The missing arm is
        the one that could have been violated."""
        y = np.concatenate([np.array([1] * 60 + [0] * 40), np.ones(100, dtype=int)])
        p = np.concatenate([np.full(60, 0.90), np.full(40, 0.0), np.full(100, 0.94)])
        g = np.array(["A"] * 100 + ["B"] * 100)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, p, g)

        assert result["base_rate_disparity"] == pytest.approx(0.4, abs=1e-12)
        assert result["tpr_disparity"] == pytest.approx(0.0, abs=1e-12)
        assert not math.isfinite(result["fpr_disparity"])
        assert result["conflict_exists"] is None
        assert result["not_assessed"] == ["fpr_disparity"]
        assert _named(caught, "conflict_exists is None (COULD NOT CHECK)")

    def test_control_a_measured_conflict_survives_a_dead_arm(self):
        """The refusal must not eat a determinate finding: with the FPR arm dead
        but the TPR gap at 1.0 against a tolerance of 0.05, the conflict is real
        whatever the FPR arm would have been."""
        y = np.concatenate([np.array([1] * 50 + [0] * 50), np.ones(100, dtype=int)])
        p = np.concatenate([np.full(50, 0.9), np.full(50, 0.1), np.full(100, 0.1)])
        g = np.array(["A"] * 100 + ["B"] * 100)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, p, g)

        assert result["tpr_disparity"] == pytest.approx(1.0, abs=1e-12)
        assert not math.isfinite(result["fpr_disparity"])
        assert result["conflict_exists"] is True
        assert result["not_assessed"] == ["fpr_disparity"]
        assert not _named(caught, "COULD NOT CHECK")

    def test_control_two_identical_groups_measure_no_conflict(self):
        """The honest all-clear has to stay available. Two identical groups give
        every disparity exactly 0.0 from rates that WERE measured, so
        conflict_exists is a measured False, not None."""
        y, p, _ = _one_group_two_hundred()
        y2 = np.concatenate([y[:100], y[:100]])
        p2 = np.concatenate([p[:100], p[:100]])
        g2 = np.array(["A"] * 100 + ["B"] * 100)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y2, p2, g2)

        assert result["conflict_exists"] is False
        assert result["not_assessed"] == []
        assert result["n_groups_compared"] == 2
        assert result["base_rate_disparity"] == pytest.approx(0.0, abs=1e-12)
        assert not _named(caught, "COULD NOT CHECK")


# ===========================================================================
# PP1-PAR. The frontier that ranked nothing and reported everything.
# ===========================================================================


class TestParetoDominanceHasThreeStates:
    def test_a_comparison_against_an_unmeasured_point_is_none(self):
        """Before: False in both directions, which in the frontier below means
        "not dominated" and therefore optimal."""
        measured = ParetoPoint(0.10, 0.20, threshold=0.5)
        unmeasured = ParetoPoint(0.10, float("nan"), threshold=0.7)

        assert measured.dominates(unmeasured) is None
        assert unmeasured.dominates(measured) is None
        # The measured pair still decides, and returns a real bool so that an
        # `is True` reader cannot be fooled by np.bool_.
        worse = ParetoPoint(0.40, 0.90, threshold=0.6)
        assert measured.dominates(worse) is True
        assert worse.dominates(measured) is False

    def test_an_unmeasured_point_is_not_returned_as_pareto_optimal(self):
        """Before: the frontier held both the measured optimum and the point
        whose fairness violation was NaN, both flagged is_pareto_optimal True,
        and the dominated point kept the default True as well."""
        best = ParetoPoint(0.10, 0.20, threshold=0.5)
        dominated = ParetoPoint(0.40, 0.90, threshold=0.6)
        unmeasured = ParetoPoint(0.10, float("nan"), threshold=0.7)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frontier = compute_pareto_frontier([best, dominated, unmeasured])

        assert [pt.threshold for pt in frontier] == [0.5]
        assert best.is_pareto_optimal is True
        assert dominated.is_pareto_optimal is False
        assert unmeasured.is_pareto_optimal is None
        assert _named(caught, "is_pareto_optimal=None and left OUT of the frontier")

    def test_a_sweep_that_measured_nothing_returns_no_frontier_and_says_so(self):
        """The real path. Group B holds no negative labels, so fpr_parity has one
        measurable rate and every one of the 17 swept thresholds carries
        fairness_violation NaN. Before: 17 points, all is_pareto_optimal True,
        no warning, and ``to_dict()['pareto_points']`` listed all 17."""
        y = np.concatenate([np.array([1] * 60 + [0] * 60), np.ones(120, dtype=int)])
        p = np.concatenate([np.linspace(0.05, 0.95, 120)] * 2)
        g = np.array(["A"] * 120 + ["B"] * 120)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(y, p, g, fairness_metric="fpr_parity")

        assert result.pareto_points == []
        assert result.to_dict()["pareto_points"] == []
        assert _named(caught, "17 of 17 point(s) have a non-finite")
        assert _named(caught, "NOTHING could be ranked")
        # The disclosure that was already correct must survive the change.
        assert [r for r in result.recommendations if "NOT ASSESSED" in r]

    def test_control_a_measured_frontier_keeps_its_three_points_of_seventeen(self):
        """Load bearing. Both groups have negative rows, so all 17 thresholds are
        measured; three of them reach fairness_violation 0.0 and dominate the
        rest at equal ECE. The first version of the fix above tested
        ``dominates(...) is True``, which np.bool_ fails, and this frontier came
        back with all 17 points flagged optimal."""
        y = np.concatenate([np.array([1] * 60 + [0] * 60), np.array([1] * 20 + [0] * 100)])
        p = np.concatenate([np.linspace(0.05, 0.95, 120)] * 2)
        g = np.array(["A"] * 120 + ["B"] * 120)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(y, p, g, fairness_metric="fpr_parity")

        assert len(result.pareto_points) == 3
        assert all(pt.is_pareto_optimal is True for pt in result.pareto_points)
        assert all(
            pt.fairness_violation == pytest.approx(0.0, abs=1e-12) for pt in result.pareto_points
        )
        assert not _named(caught, "non-finite calibration error")

    def test_control_the_documented_three_point_example_still_holds(self):
        """The docstring example: the first two points are Pareto optimal."""
        points = [ParetoPoint(0.1, 0.2), ParetoPoint(0.2, 0.1), ParetoPoint(0.3, 0.3)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frontier = compute_pareto_frontier(points)

        assert len(frontier) == 2
        assert [pt.is_pareto_optimal for pt in points] == [True, True, False]
        assert not _named(caught, "non-finite")


# ===========================================================================
# PP1-MIT. A selection-rate gap of zero from one group.
# ===========================================================================


class TestMitigationParetoNeedsTwoGroupsToBeBetween:
    def test_one_group_is_not_a_structural_disparity(self):
        """Before: available True, baselineGap 0.0, bestGap 0.0, fairnessGap 0.0
        on every point, and the summary read "No tested post-processing
        mitigation materially reduces the gap on this data: the disparity is
        structural, not just a threshold artefact"."""
        y, p, _ = _one_group_two_hundred()

        result = mitigation_pareto(y, p, np.array(["A"] * 200))

        assert result["available"] is False
        assert "two groups of at least 10 rows" in result["reason"]
        assert "NOT a finding that the gap is zero" in result["reason"]
        assert result["nGroupsMeasurable"] == 1
        # No axis, no summary, no gap numbers to misread.
        assert "baselineGap" not in result
        assert "summary" not in result

    def test_every_group_below_the_row_gate_is_refused_too(self):
        """30 groups of about 6 rows each: the same output as one group before
        the fix, contextDistortion 0.0 included."""
        y, p, _ = _one_group_two_hundred()
        groups = np.array([f"g{i % 30}" for i in range(200)])

        result = mitigation_pareto(y, p, groups)

        assert result["available"] is False
        assert result["nGroups"] == 30
        assert result["nGroupsMeasurable"] == 0

    def test_control_two_real_groups_still_produce_a_frontier(self):
        """Group A is scored above the cut and group B below it, so the baseline
        selection-rate gap is a measured 1.0 and group thresholds close it."""
        y, p, g = _gapped_threshold_data()

        result = mitigation_pareto(y, p, g)

        assert result["available"] is True
        assert result["baselineGap"] == pytest.approx(1.0, abs=1e-12)
        assert result["bestGap"] < result["baselineGap"]
        assert len(result["points"]) >= 6
        assert "The selection-rate gap can be cut from" in result["summary"]

    def test_control_the_ten_row_boundary_is_measured_not_refused(self):
        """Exactly 10 rows is inside the gate, so this must be a measurement."""
        y, p, _ = _one_group_two_hundred()
        groups = np.array(["A"] * 190 + ["B"] * 10)

        result = mitigation_pareto(y, p, groups)

        assert result["available"] is True
        assert math.isfinite(result["baselineGap"])

    def test_control_the_existing_empty_input_refusal_is_unchanged(self):
        result = mitigation_pareto(None, [], [])

        assert result["available"] is False
        assert result["reason"] == "No usable score / group arrays."


# ===========================================================================
# PP1-REC. "Calibration metrics are acceptable" from a disparity that does not
# exist.
# ===========================================================================


class TestTheRecommendationSaysWhenTheDisparityWasNotMeasured:
    def test_an_unmeasurable_disparity_is_not_acceptable_calibration(self):
        """Before: strategy monitor_only, priority "low", rationale "Calibration
        metrics are acceptable. Monitor for drift but no immediate action
        needed.", not_assessed_groups [], on a single group whose ece_disparity
        was NaN and whose excluded_groups was empty."""
        y, p, g = _one_group_two_hundred()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recommendation = recommend_calibration_strategy(y, p, g, context="lending")

        assert recommendation.disparity_assessed is False
        assert recommendation.priority != "low"
        assert "NOT ASSESSED" in recommendation.rationale
        # The field has to survive the serialiser a dashboard reads, which is
        # where an earlier three-state fix in this class was lost.
        assert recommendation.to_dict()["disparity_assessed"] is False
        assert _named(caught, "between-group calibration disparity")

    def test_control_a_measured_disparity_is_reported_as_measured(self):
        """Two 80-row groups with ECE 0.0 and 0.5: a real disparity of 0.5, so
        the recommendation is a finding and carries no caveat."""
        y, p, g = _two_eighty_row_groups()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recommendation = recommend_calibration_strategy(y, p, g, context="lending")

        assert recommendation.disparity_assessed is True
        assert recommendation.priority == "high"
        assert "NOT ASSESSED" not in recommendation.rationale
        assert recommendation.to_dict()["disparity_assessed"] is True
        assert not _named(caught, "between-group calibration disparity")


# ===========================================================================
# PP1-REG. A min and a max that read as an interval.
# ===========================================================================


class TestTheFeasibleRegionIsNotAlwaysARange:
    def test_a_gapped_satisfying_set_is_not_reported_as_an_interval(self):
        """13 of 100 searched thresholds satisfy demographic_parity, at indices
        0 to 5 and 93 to 99. Before: the pair (0.01, 0.99) and nothing else, so
        threshold 0.5, measured at a violation of exactly 1.0, sat inside the
        reported region."""
        y, p, g = _gapped_threshold_data()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            region = analyzer.find_feasible_region(
                constraint="demographic_parity", tolerance=0.05, n_thresholds=100
            )

        assert region.status == "feasible"
        assert region.contiguous is False
        assert region.n_feasible == 13
        assert region.n_searched == 100
        # The endpoints are unchanged: they ARE the min and max, and both are
        # themselves feasible, which an existing test in
        # tests/test_threshold_analyzer.py asserts.
        assert tuple(region) == pytest.approx((0.01, 0.99), abs=1e-12)
        assert _named(caught, "do NOT form one run")

    def test_the_summary_and_the_recommendations_carry_the_gap(self):
        """The brackets are what a reader acts on. Before, this printed
        "demographic_parity: [0.010, 0.990]" and the only recommendation was
        "Multiple constraints can be satisfied"."""
        y, p, g = _gapped_threshold_data()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(n_thresholds=5, constraints=["demographic_parity"])

        text = report.summary()
        assert "demographic_parity: [0.010, 0.990]" not in text
        assert "NOT as one run" in text
        assert "13 of 100" in text
        assert [r for r in report.recommendations if "are NOT one range" in r]
        assert not [r for r in report.recommendations if "Multiple constraints can be" in r]

    def test_control_a_contiguous_region_still_prints_as_a_range(self):
        """Two groups with identical score distributions: all 25 searched
        thresholds satisfy demographic parity, so the pair IS the interval and
        nothing extra may be said about it."""
        y, p, g = _balanced_threshold_data()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            region = analyzer.find_feasible_region(
                constraint="demographic_parity", tolerance=0.05, n_thresholds=25
            )

        assert region.contiguous is True
        assert region.n_feasible == 25
        assert region.n_not_assessed == 0
        assert tuple(region) == pytest.approx((0.01, 0.99), abs=1e-12)
        assert not _named(caught, "do NOT form one run")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = analyzer.full_analysis(
                n_thresholds=3, constraints=["demographic_parity"]
            ).summary()
        assert "demographic_parity: [0.010, 0.990]" in text

    def test_no_constraint_examined_is_not_an_all_clear(self):
        """``full_analysis(constraints=[])`` evaluated nothing and closed with
        "Multiple constraints can be satisfied. Consider the specific
        requirements of your use case when selecting a threshold."."""
        y, p, g = _balanced_threshold_data()
        analyzer = ThresholdAnalyzer(y, p, g)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(n_thresholds=3, constraints=[])

        assert report.optimal_thresholds == {}
        assert report.feasible_regions == {}
        joined = " ".join(report.recommendations)
        assert "NOT ASSESSED" in joined
        assert "no fairness constraint was evaluated" in joined
        assert "Multiple constraints can be satisfied" not in joined


# ===========================================================================
# Correct refusals verified by execution in this batch and pinned so a
# regression is visible. None of these needed a fix.
# ===========================================================================


class TestTheThresholdAnalyzerAlreadyRefusesOnOneGroup:
    """A single group has no between-group rate, so ``compute_constraint_violation``
    answers ``is_satisfied=None`` at every threshold and each of these five units
    carries that third state out to the caller. Executed 2026-09-27 on 200 rows
    of one group; every assertion below was already true and must stay true."""

    @staticmethod
    def _one_group_analyzer():
        y, p, _ = _one_group_two_hundred()
        return ThresholdAnalyzer(y, p, np.array(["A"] * 200))

    def test_analyze_threshold_and_the_range_keep_the_third_state(self):
        analyzer = self._one_group_analyzer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            single = analyzer.analyze_threshold(0.5)
            swept = analyzer.analyze_threshold_range(thresholds=[0.3, 0.5, 0.7])

        for violation in single.constraint_violations.values():
            assert violation.is_satisfied is None
            assert not math.isfinite(violation.violation)
        assert len(swept) == 3
        assert all(v.is_satisfied is None for r in swept for v in r.constraint_violations.values())

    def test_the_optimum_and_the_region_refuse_rather_than_default(self):
        analyzer = self._one_group_analyzer()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimum = analyzer.find_optimal_threshold()
            region = analyzer.find_feasible_region()

        # Not 0.5, and not the first grid point either.
        assert optimum["optimal_threshold"] is None
        assert optimum["is_feasible"] is None
        assert optimum["n_thresholds_not_assessed"] == 100
        assert region.status == "not_assessed"
        assert region.assessed is False
        assert region.n_feasible == 0
        assert _named(caught, "none of the 100 searched")

    def test_the_report_and_its_summary_say_not_assessed(self):
        analyzer = self._one_group_analyzer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(n_thresholds=5)

        assert all(
            t is None
            for per_constraint in report.optimal_thresholds.values()
            for t in per_constraint.values()
        )
        text = report.summary()
        assert "Best for accuracy: NOT ASSESSED" in text
        assert "NOT ASSESSED (no threshold could be checked" in text
        assert "No feasible region found" not in text
        assert [r for r in report.recommendations if "NOT ASSESSED" in r]
        assert not [r for r in report.recommendations if "Consider using group-specific" in r]


class TestFeasibleRegionCarriesItsOwnThirdState:
    def test_the_type_keeps_the_pair_interface(self):
        """Every existing endpoint reader has to keep working: the third state
        rides alongside the tuple rather than replacing it."""
        region = FeasibleRegion(None, None, assessed=False, n_not_assessed=7, n_searched=7)

        assert isinstance(region, tuple)
        low, high = region
        assert low is None and high is None
        assert region == (None, None)
        assert region.status == "not_assessed"
        assert region.n_feasible == 0
        # The default has to be the harmless one for a region that HAS endpoints.
        assert FeasibleRegion(0.2, 0.4).contiguous is True
