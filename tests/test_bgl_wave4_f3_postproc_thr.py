"""BGL wave 4, package F3-postproc-thr: the four overturned grades on
``ThresholdAnalyzer``, closed.

An independent audit overturned four grades on 2026-09-30, all four for the same
reason: the cited evidence executed ZERO lines of the graded unit. The two named
tests were ``with pytest.raises(InvalidDataError): ThresholdAnalyzer(...)``, so no
object was ever built, and the file named for ``get_explanation`` passed with the
fix deleted. Every pin here goes THROUGH the graded unit, and each has an
over-correction control beside it asserting the healthy case's real number.

Reproducing the audit closed all four recipes, so the live defects below were
found by execution against the SEARCH's own parameters instead. Threshold
optimisation is a MITIGATION, and a search that has stopped searching reports a
BETTER parity number, not a worse one: a cut that puts every row on one side
approves (or rejects) everybody, and a constant decision is trivially equal
across groups.

F3-1  ``analyze_threshold(float("nan"))``. ``y_prob >= nan`` is False for every
      row by IEEE 754, so the whole population came back rejected and every
      group's selection rate was 0.0. Measured on 200 rows in two fully scored,
      fully labelled groups: demographic_parity violation 0.0, is_satisfied True,
      group_metrics {'A': 0.0, 'B': 0.0}, ``unmeasured=()``, and no warning about
      the threshold. Through ``analyze_threshold_range(thresholds=[0.5, nan])``
      the NaN row read as the FAIREST row of the sweep, 0.0 against 0.14, so a
      caller ranking rows by violation picks the threshold that is not a number.

F3-2  A grid below two candidates. ``_require_searchable_grid`` has guarded
      exactly this since BGL3-PP3 for ThresholdOptimizer,
      GroupThresholdOptimizer and MultiObjectiveThresholdOptimizer, and never
      reached this class, which takes the same argument and builds the same
      linspace. Measured on 200 rows in two scored, labelled groups:
        n_thresholds=1 -> the grid is [0.01], which approves 200 of 200 rows, and
                          find_feasible_region answered status "feasible",
                          (0.01, 0.01), n_feasible 1 of 1, contiguous True, while
                          find_optimal_threshold answered optimal_threshold 0.01,
                          is_feasible True, n_thresholds_not_assessed 0.
        n_thresholds=0 -> the loop never runs; the empty ``feasible`` list and
                          ``n_not_assessed == n_searched`` being 0 == 0, which the
                          ``and n_searched > 0`` clause excludes, fell through to
                          ``assessed=True``: status "infeasible", n_searched 0. A
                          MEASURED finding of infeasibility from zero
                          measurements, and ``_generate_recommendations`` then
                          printed "No single threshold can satisfy
                          demographic_parity. Consider using group-specific
                          thresholds", a recommendation to change the model's
                          decision rule.

F3-3  A sweep that made ONE decision, reported as a hundred. ``n_searched`` /
      ``n_thresholds_searched`` read as the number of candidates weighed.
      Measured twice on 200 rows:
        scores in [0.995, 0.9999]  -> all 100 searched thresholds approve
                          everybody, so 100 of 100 "satisfied"
                          demographic_parity, the region came back (0.01, 0.99)
                          with contiguous True, ``summary()`` printed
                          "[0.010, 0.990]", the optimum was 0.01 (which approves
                          200 of 200) and the report said "Multiple constraints
                          can be satisfied". Zero warnings.
        a hard 0/1 PREDICTION column -> the shape the platform sends whenever an
                          uploaded dataset has no probability column: one
                          decision at all 100 thresholds, region (None, None)
                          "infeasible" with n_searched 100, optimum 0.01, and the
                          group-specific-thresholds recommendation. Zero warnings.
      Recorded and disclosed rather than refused, which is the policy
      ``_record_score_resolution`` and ``_qualified_feasibility`` argue for in
      optimizer.py: hard predictions are a legitimate thing to hold and the
      constraint verdict ON that one decision is a real measurement of it. What
      was missing is that the sweep never varied anything.

F3-4  ``get_explanation`` on a sweep where nothing could be assessed. The
      behaviour was already right; nothing executed it. The pin below drives the
      graded unit and asserts the strings, so the "3 constraint(s) are
      satisfiable" the explainer printed for three unassessed regions cannot
      come back unseen.
"""

from __future__ import annotations

import json
import warnings
from typing import List

import numpy as np
import pytest

from vfairness.exceptions import InvalidDataError
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalyzer,
)

# ===========================================================================
# Fixtures. Deterministic, no rng, so every number quoted above is reproducible
# from the arrays alone.
# ===========================================================================


def _two_scored_groups(n_per_group: int = 100):
    """Two fully scored, fully labelled groups whose scores overlap: the healthy
    input every control below uses."""
    p = np.concatenate([np.linspace(0.30, 0.95, n_per_group), np.linspace(0.05, 0.70, n_per_group)])
    y = (p >= 0.5).astype(int)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _one_group_two_hundred():
    """The NOT ASSESSED fixture: one group, so no between-group rate exists at
    any threshold."""
    p = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 40)
    y = np.tile(np.array([0, 0, 1, 1, 1]), 40)
    return y, p, np.array(["A"] * 200)


def _scores_above_the_grid(n_per_group: int = 100):
    """200 distinct scores, every one of them above the top of the searched grid,
    so all 100 candidates approve everybody. The score RESOLUTION is ample here:
    this is the sibling door that counting distinct scores cannot see."""
    p = np.concatenate(
        [
            np.linspace(0.995, 0.999, n_per_group),
            np.linspace(0.9951, 0.9999, n_per_group),
        ]
    )
    y = np.concatenate(
        [
            np.tile(np.array([1, 0, 1, 1, 0]), n_per_group // 5),
            np.tile(np.array([1, 0, 0, 0, 0]), n_per_group // 5),
        ]
    )
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _hard_prediction_column(n_per_group: int = 100):
    """y_prob is the PREDICTION column, hard 0.0/1.0: group A accepted at 0.6,
    group B at 0.2, a real 0.4 gap that no threshold can move."""
    pred_a = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), n_per_group // 5)
    pred_b = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), n_per_group // 5)
    y = np.concatenate(
        [
            np.tile(np.array([1.0, 1.0, 0.0, 0.0, 1.0]), n_per_group // 5),
            np.tile(np.array([1.0, 0.0, 1.0, 0.0, 0.0]), n_per_group // 5),
        ]
    )
    p = np.concatenate([pred_a, pred_b])
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _messages(caught) -> List[str]:
    return [str(w.message) for w in caught]


# ===========================================================================
# F3-1. A threshold that is not a number is not a decision rule.
# ThresholdAnalyzer.analyze_threshold
# ===========================================================================


class TestANonFiniteThresholdIsNotADecisionRule:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_threshold_is_refused(self, bad):
        """Before: nan rejected all 200 rows and both selection rates came out
        0.0, which is PERFECT PARITY, with is_satisfied True and unmeasured=()."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        with pytest.raises(InvalidDataError, match="not a finite number"):
            analyzer.analyze_threshold(bad)

    def test_the_refusal_covers_the_range_sweep_that_delegates_to_it(self):
        """The guard sits at the one place both methods turn a threshold into
        decisions, so a NaN inside an explicit threshold list is refused too. It
        used to come back as the FAIREST row of the sweep."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        with pytest.raises(InvalidDataError, match="not a finite number"):
            analyzer.analyze_threshold_range(thresholds=[0.5, float("nan")])

    def test_control_a_finite_threshold_still_measures_its_violation(self):
        """Over-correction control with the real numbers at 0.5 on the healthy
        fixture: demographic_parity violation 0.38 with group_metrics
        {'A': 0.69, 'B': 0.31}, is_satisfied False, accuracy 1.0 and a confusion
        matrix of 100 tp and 100 tn."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        result = _quiet(analyzer.analyze_threshold, 0.5)

        violation = result.constraint_violations["demographic_parity"]
        assert violation.violation == pytest.approx(0.38, abs=1e-12)
        assert violation.is_satisfied is False
        assert violation.group_metrics == {
            "A": pytest.approx(0.69),
            "B": pytest.approx(0.31),
        }
        assert result.performance_metrics["accuracy"] == pytest.approx(1.0)
        assert result.confusion_matrix == {"tp": 100, "tn": 100, "fp": 0, "fn": 0}

    def test_control_the_range_sweep_still_runs_over_finite_thresholds(self):
        """The same control one level up: two finite thresholds, two measured
        rows, violations 0.38 at 0.5 and 0.0 at 0.99 (where every row is
        rejected, which is why that end is degenerate rather than good)."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        rows = _quiet(analyzer.analyze_threshold_range, thresholds=[0.5, 0.99])

        assert [row.threshold for row in rows] == [0.5, 0.99]
        assert rows[0].constraint_violations["demographic_parity"].violation == pytest.approx(
            0.38, abs=1e-12
        )


# ===========================================================================
# F3-2. A grid that cannot choose between anything, in the class the package's
# own guard never reached.
# ThresholdAnalyzer.find_feasible_region / find_optimal_threshold
# ===========================================================================


class TestAGridBelowTwoCandidatesIsRefusedHereToo:
    @pytest.mark.parametrize("n_thresholds", [0, 1])
    @pytest.mark.parametrize("method", ["find_feasible_region", "find_optimal_threshold"])
    def test_a_grid_below_two_candidates_is_refused(self, method, n_thresholds):
        """n_thresholds=1 searched [0.01] alone, which approves 200 of 200 rows,
        and published status "feasible" / is_feasible True off it. n_thresholds=0
        searched nothing and published a MEASURED infeasibility."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        with pytest.raises(ValueError, match="candidate threshold"):
            _quiet(getattr(analyzer, method), n_thresholds=n_thresholds)

    def test_the_empty_grid_no_longer_answers_infeasible(self):
        """The direction that matters: an empty sweep used to reach the LAST
        return of find_feasible_region, assessed=True and status "infeasible",
        which is a finding. It is refused, so no verdict is published at all."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        with pytest.raises(ValueError):
            _quiet(analyzer.find_feasible_region, n_thresholds=0)

    def test_control_the_floor_itself_still_searches(self):
        """Over-correction control: two candidates is the floor, not a refusal.
        The grid is [0.01, 0.99] on the healthy fixture, which makes two
        decisions, approve-all and reject-all, and both satisfy demographic
        parity, so the region is measured feasible over 2 of 2."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        region = _quiet(analyzer.find_feasible_region, n_thresholds=2)

        assert region.n_searched == 2
        assert region.n_feasible == 2
        assert region.status == "feasible"
        assert region[0] == pytest.approx(0.01)
        assert region[1] == pytest.approx(0.99)
        # And the disclosure is honest about what those two candidates were: two
        # decisions, so this is NOT the one-decision case.
        assert region.n_distinct_decisions == 2

    def test_control_a_full_grid_still_measures_its_region_and_optimum(self):
        """Over-correction control with the real numbers on the healthy fixture:
        15 of 100 searched thresholds satisfy demographic_parity, the pair is
        (0.010, 0.990) and NOT contiguous, the 100 candidates made 92 distinct
        decisions, and the optimum is 0.9207070707070707 with is_feasible a
        measured True."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        region = _quiet(analyzer.find_feasible_region, n_thresholds=100)
        optimum = _quiet(analyzer.find_optimal_threshold, constraint="demographic_parity")

        assert region.n_feasible == 15
        assert region.n_searched == 100
        assert region.contiguous is False
        assert region.n_distinct_decisions == 92
        assert region.n_distinct_scores == 200
        assert region.score_resolution_sufficient is True
        assert optimum["is_feasible"] is True
        assert optimum["optimal_threshold"] == pytest.approx(0.9207070707070707, abs=1e-12)
        assert optimum["n_thresholds_searched"] == 100
        assert optimum["n_distinct_decisions"] == 92


# ===========================================================================
# F3-3. A sweep that made one decision, reported as a hundred candidates.
# ThresholdAnalyzer.find_feasible_region / find_optimal_threshold
# ===========================================================================


class TestASweepThatMadeOneDecisionSaysSo:
    def test_a_grid_entirely_off_the_score_range_is_not_a_range_of_thresholds(self):
        """All 100 candidates approve every row, so the region reads
        (0.01, 0.99) "feasible" over 100 of 100. The pair is kept, because at
        every one of those thresholds parity really is satisfied; what is added
        is the measured count of decisions behind it."""
        y, p, g = _scores_above_the_grid()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        region = _quiet(analyzer.find_feasible_region, n_thresholds=100)
        optimum = _quiet(analyzer.find_optimal_threshold, constraint="demographic_parity")

        # The sweep, measured.
        assert region.n_searched == 100
        assert region.n_feasible == 100
        assert region.n_distinct_decisions == 1
        assert optimum["n_thresholds_searched"] == 100
        assert optimum["n_distinct_decisions"] == 1
        # The reported optimum approves everybody, which is what makes the
        # "feasible" verdict trivial rather than false.
        assert int((analyzer.y_prob >= optimum["optimal_threshold"]).sum()) == 200
        # And counting distinct SCORES cannot see this: there are 200 of them.
        assert region.n_distinct_scores == 200
        assert region.score_resolution_sufficient is True

    def test_the_one_decision_sweep_is_disclosed_where_a_reader_looks(self):
        """Not only in a returned field: the summary line the reader acts on and
        the recommendations both carry it."""
        y, p, g = _scores_above_the_grid()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        report = _quiet(analyzer.full_analysis, n_thresholds=20, constraints=["demographic_parity"])

        region_line = [
            line
            for line in report.summary().splitlines()
            if line.strip().startswith("demographic_parity:")
        ][-1]
        assert "[0.010, 0.990]" in region_line
        assert "NOT A SEARCH" in region_line
        assert "made the SAME decision" in region_line

        joined = " ".join(report.recommendations)
        assert "NOT A SEARCH" in joined
        assert "one decision rule rather than on a search over thresholds" in joined
        # The line that used to be the whole story, and is now false comfort on
        # its own, is gone from this report.
        assert "Multiple constraints can be satisfied" not in joined

    def test_the_serialised_region_carries_the_count(self):
        """The boundary a programmatic consumer reads. A field that exists only
        on the object is dropped by json.dumps, which is the incident
        _region_to_dict was added for."""
        y, p, g = _scores_above_the_grid()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        report = _quiet(analyzer.full_analysis, n_thresholds=20, constraints=["demographic_parity"])
        encoded = json.loads(json.dumps(report.to_dict()["feasible_regions"]["demographic_parity"]))

        assert encoded["status"] == "feasible"
        assert encoded["n_searched"] == 100
        assert encoded["n_distinct_decisions"] == 1
        assert encoded["n_distinct_scores"] == 200
        assert encoded["score_resolution_sufficient"] is True

    def test_the_hard_prediction_column_is_measured_and_disclosed(self):
        """The platform's most common shape. The verdict on those predictions is
        real, 0.4 of selection-rate gap, so it is NOT withdrawn: what is added is
        that no threshold in the sweep could have moved it."""
        y, p, g = _hard_prediction_column()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = ThresholdAnalyzer(y, p, g)

        assert any("2 distinct value(s)" in m for m in _messages(caught)), _messages(caught)

        region = _quiet(analyzer.find_feasible_region, n_thresholds=100)
        optimum = _quiet(analyzer.find_optimal_threshold, constraint="demographic_parity")
        single = _quiet(analyzer.analyze_threshold, 0.5, ["demographic_parity"])

        assert region.n_distinct_scores == 2
        assert region.score_resolution_sufficient is False
        assert region.n_distinct_decisions == 1
        assert optimum["n_distinct_scores"] == 2
        assert optimum["score_resolution_sufficient"] is False
        assert optimum["n_distinct_decisions"] == 1
        # The measurement itself, untouched.
        violation = single.constraint_violations["demographic_parity"]
        assert violation.violation == pytest.approx(0.4, abs=1e-12)
        assert violation.group_metrics == {"A": pytest.approx(0.6), "B": pytest.approx(0.2)}
        assert region.status == "infeasible"
        assert optimum["is_feasible"] is False

    def test_control_a_real_sweep_is_not_flagged_as_one_decision(self):
        """Over-correction control. A guard that flagged every sweep would pass
        every test above and destroy the capability. On the healthy fixture the
        100 candidates make 92 distinct decisions, nothing is flagged, and the
        analyzer is silent at construction."""
        y, p, g = _two_scored_groups()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = ThresholdAnalyzer(y, p, g)
        assert _messages(caught) == [], _messages(caught)

        report = _quiet(analyzer.full_analysis, n_thresholds=20, constraints=["demographic_parity"])
        region = report.feasible_regions["demographic_parity"]

        assert region.n_distinct_decisions == 92
        assert region.score_resolution_sufficient is True
        assert "NOT A SEARCH" not in report.summary()
        assert "NOT A SEARCH" not in " ".join(report.recommendations)

    def test_control_a_hand_built_region_records_nothing_rather_than_reassurance(self):
        """Three states. A region built without a score column has no measurement
        to report, and the absent one is None, not True."""
        region = FeasibleRegion(0.2, 0.8, assessed=True, n_searched=10, n_feasible=4)

        assert region.n_distinct_decisions is None
        assert region.n_distinct_scores is None
        assert region.score_resolution_sufficient is None


# ===========================================================================
# F3-4. The explanation of a sweep where nothing could be assessed, THROUGH the
# graded unit.
# ThresholdAnalyzer.get_explanation
# ===========================================================================


class TestTheExplanationRunsThroughTheUnit:
    def test_the_unit_reports_zero_satisfiable_and_names_the_unassessed(self):
        """The row that was graded PROVEN cited a test asserting only that the
        string was non-empty, and it passed with the fix deleted. This drives
        ThresholdAnalyzer.get_explanation itself on 200 rows of one group, where
        all 100 searched thresholds are could-not-check."""
        y, p, g = _one_group_two_hundred()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        explanation = _quiet(analyzer.get_explanation)

        assert "0 constraint(s) are satisfiable" in explanation.summary
        assert "3 constraint(s) are satisfiable" not in explanation.summary
        assert "3 could not be evaluated at any searched threshold" in explanation.summary
        assert explanation.severity == "medium"

        block = explanation.explanations[0]
        assert block.value == "3 constraint(s) analysed, 0 feasible region(s), 3 NOT ASSESSED"
        assert block.evaluation.startswith(
            "NOT ASSESSED: no fairness constraint was established as feasible or ruled out."
        )
        assert "may need retraining" not in block.evaluation

    def test_the_precomputed_report_path_through_the_unit_agrees(self):
        """Both arms of the unit: get_explanation() runs full_analysis itself,
        get_explanation(report) takes one. A fix in one arm and not the other is
        how these defects survive."""
        y, p, g = _one_group_two_hundred()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        report = _quiet(analyzer.full_analysis)
        from_report = _quiet(analyzer.get_explanation, report)
        from_scratch = _quiet(analyzer.get_explanation)

        assert from_report.summary == from_scratch.summary
        assert "0 constraint(s) are satisfiable" in from_report.summary
        assert from_report.severity == "medium"

    def test_control_a_feasible_sweep_still_explains_itself_as_satisfiable(self):
        """Over-correction control with the real strings: on two fully scored,
        fully labelled groups the unit reports 3 satisfiable constraints, value
        "3 constraint(s) analysed, 3 feasible region(s)" and severity "info"."""
        y, p, g = _two_scored_groups()
        analyzer = _quiet(ThresholdAnalyzer, y, p, g)

        explanation = _quiet(analyzer.get_explanation)

        assert "3 constraint(s) are satisfiable within the threshold range." in explanation.summary
        assert "NOT ASSESSED" not in explanation.summary
        assert explanation.severity == "info"
        block = explanation.explanations[0]
        assert block.value == "3 constraint(s) analysed, 3 feasible region(s)"
        assert block.evaluation == "3 fairness constraint(s) have a feasible threshold region."
