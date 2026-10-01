"""BGL3 batch post_processing-3: does a post-processing unit refuse honestly?

Seven units, five files: the two calibrators whose fits were already graded once
(``IsotonicCalibrator.fit``, ``TemperatureScaling.fit``), the calibration
dashboard, the exposure-parity re-ranker, ``compute_constraint_violation`` and
the two threshold optimisers.

The shape hunted here is the batch's own trap: an optimiser that found no
feasible threshold must not hand back the default one as though it had been
chosen, and a value derived from a threshold, a weight or a mask is still a
verdict once a reader acts on it.

Five findings, each measured by execution before the fix and each paired with a
CONTROL on measurable input, so a fix that simply stops answering cannot pass
this file:

  1. rerank promoted an item with NO score to rank 0 and reported the exposure
     parity that placement produced.
  2. rerank with negative scores was a complete no-op reporting ndcg_after 1.0
     and utility_loss 0.0.
  3. compute_constraint_violation graded a verdict against a NaN tolerance.
  4. the threshold optimisers returned their own threshold SEED when the grid
     held no candidate, silently.
  5. the Pareto fallback (a group count other than 2) reached
     optimization_details and never reached ``summary()``.
  6. the calibration dashboard drew an ECE disparity measured over a SUBSET of
     the groups as though it were the whole comparison.
  7. the isotonic map moved with the SCALE of the sample weights.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, List, Tuple

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from vfairness.post_processing.calibration.methods import (  # noqa: E402
    IsotonicCalibrator,
    TemperatureScaling,
)
from vfairness.post_processing.calibration.visualization import (  # noqa: E402
    create_calibration_dashboard,
)
from vfairness.post_processing.ranking.rerank import exposure_parity_rerank  # noqa: E402
from vfairness.post_processing.threshold_optimization.constraints import (  # noqa: E402
    compute_constraint_violation,
)
from vfairness.post_processing.threshold_optimization.optimizer import (  # noqa: E402
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdOptimizer,
)


def _loudly(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, List[str]]:
    """Call ``fn`` capturing its warnings, and return (result, warning texts)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _annotations(fig) -> List[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def _legend_labels(fig) -> List[str]:
    out: List[str] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            out.extend(t.get_text() for t in legend.get_texts())
    return out


# Ten items, two equal groups, group A holding the five best scores: the skew
# this intervention exists for.
SKEW_SCORES = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.40, 0.35, 0.30, 0.25, 0.20])
SKEW_GROUPS = np.array(["A"] * 5 + ["B"] * 5)


def _scored_population(n: int = 200, seed: int = 7):
    """Two groups, informative scores, a labelling a threshold can separate."""
    rng = np.random.RandomState(seed)
    y_prob = rng.rand(n)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    y_true = (y_prob + rng.normal(0, 0.25, n) > 0.5).astype(int)
    return y_true, y_prob, groups


# ── exposure_parity_rerank: an item with no score is not the best item ───────


class TestRerankRefusesAnUnscoreableItem:
    def test_an_unscored_item_is_refused_not_promoted_to_rank_one(self):
        """Measured BEFORE the fix on SKEW_SCORES with the WORST-scored item
        (index 9, score 0.20) replaced by NaN:

            reranked_order              [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            exposure_parity_diff_before 0.2707 -> after 0.0132
            ndcg_before/after/loss      nan, nan, nan
            warnings                    []

        The item nobody scored took position 0, the highest-exposure slot in the
        ranking, and the whole parity improvement reported was produced by that
        placement. ``np.nanargmax`` guards the CANDIDATE selection, and the
        ledger records that fix; the feasibility FALLBACK three lines under it is
        a plain ``np.argmax``, and since every comparison against a NaN
        completion is False it fires at every position, so the fix never covered
        the site that actually chose.
        """
        scores = SKEW_SCORES.copy()
        scores[9] = np.nan

        with pytest.raises(ValueError, match="no usable score"):
            exposure_parity_rerank(scores, SKEW_GROUPS)

        # An infinite score is the same could-not-check and used to be ranked
        # first for the opposite reason: it compares greater than everything.
        scores[9] = np.inf
        with pytest.raises(ValueError, match="no usable score"):
            exposure_parity_rerank(scores, SKEW_GROUPS)

    def test_negative_scores_are_refused_rather_than_disarming_the_greedy(self):
        """Measured BEFORE the fix on 10 items scored -0.2 ... -2.0 over the same
        two groups:

            reranked_order == the utility-optimal order (a complete no-op)
            exposure_parity_diff 0.2707 before AND 0.2707 after
            ndcg_before 1.0, ndcg_after 1.0, utility_loss 0.0, warnings []

        With a negative ideal DCG the floor (1 - max_utility_loss) * idcg sits
        ABOVE the optimum, so nothing is ever feasible, the fallback keeps the
        top-scored item at every position, and the intervention returns the order
        it was handed while reporting that it kept full utility.
        """
        negative = -np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

        with pytest.raises(ValueError, match="negative"):
            exposure_parity_rerank(negative, SKEW_GROUPS)

    def test_control_a_healthy_rerank_still_moves_exposure_within_its_budget(self):
        """CONTROL. The same skew, all scores measured and non-negative: the
        re-ranking must still happen, still improve parity and still respect the
        budget it advertises. Measured after the fix: order
        [0, 5, 6, 1, 7, 2, 8, 3, 9, 4], parity 0.2707 -> 0.0479, ndcg_after
        0.9126 against a floor of 0.9.
        """
        result, caught = _loudly(exposure_parity_rerank, SKEW_SCORES, SKEW_GROUPS)

        assert caught == [], caught
        assert sorted(result["reranked_order"]) == list(range(10))
        assert result["reranked_order"] != list(range(10)), "the greedy did nothing"
        assert result["exposure_parity_diff_after"] < result["exposure_parity_diff_before"]
        # The budget is the function's own promise, recomputed here rather than
        # read back from the same dict that claims it.
        assert result["ndcg_after"] >= 0.9 - 1e-9, result["ndcg_after"]
        assert result["utility_loss"] == pytest.approx(result["ndcg_before"] - result["ndcg_after"])


# ── compute_constraint_violation: the tolerance is half of every verdict ─────


class TestConstraintTolerance:
    YT = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    YP = np.array([1, 0, 1, 1, 1, 0, 0, 0])
    G = np.array(["A"] * 4 + ["B"] * 4)

    def test_a_nan_tolerance_cannot_decide_anything_and_is_refused(self):
        """Measured BEFORE the fix on a fully measured demographic-parity gap of
        0.50 (A 0.75 against B 0.25), one call per tolerance:

            tolerance 0.05 -> is_satisfied False   (correct)
            tolerance nan  -> is_satisfied False, violation 0.50, warnings []
            tolerance inf  -> is_satisfied True,  violation 0.50, warnings []

        Every comparison against NaN is False, so the published verdict was a
        measured VIOLATION decided by no tolerance at all, and an infinite one
        graded a gap of 0.50 as satisfied. The module already guards the LEFT
        operand of ``violation <= tolerance`` and said so in a comment; the right
        operand ran unchecked. A missing cell in a config table arrives as NaN,
        not as None, and None would at least raise on the comparison.
        """
        for bad in (float("nan"), float("inf"), -0.1):
            with pytest.raises(ValueError, match="finite, non-negative"):
                compute_constraint_violation(
                    self.YT, self.YP, self.G, "demographic_parity", tolerance=bad
                )

        # Why a negative tolerance is refused rather than merely useless: it
        # grades PERFECT parity as a violation.
        assert bool(0.0 <= -0.1) is False

    def test_control_a_measured_gap_is_still_graded_both_ways(self):
        """CONTROL. The verdict must still be reachable in both directions from a
        finite tolerance. Measured: gap 0.50 against 0.05 -> False; the same gap
        against 0.60 -> True, both with no warnings.
        """
        strict, caught_strict = _loudly(
            compute_constraint_violation,
            self.YT,
            self.YP,
            self.G,
            "demographic_parity",
            tolerance=0.05,
        )
        loose, caught_loose = _loudly(
            compute_constraint_violation,
            self.YT,
            self.YP,
            self.G,
            "demographic_parity",
            tolerance=0.60,
        )

        assert strict.violation == pytest.approx(0.5)
        assert strict.is_satisfied is False
        assert loose.is_satisfied is True
        assert caught_strict == [] and caught_loose == []

    def test_a_single_group_is_not_assessed_and_no_positives_is_not_parity(self):
        """The refusals verified by execution on this unit, pinned so a later
        "simplification" cannot turn either back into a 0.0.

        one group    -> violation nan, is_satisfied None, group_violations
                        {'A': nan}, warning naming "fewer than 2 groups"
        no positives -> equal_opportunity violation nan, is_satisfied None,
                        unmeasured ('A', 'B'), warning naming the empty
                        denominator
        """
        one_group, caught_one = _loudly(
            compute_constraint_violation,
            self.YT,
            self.YP,
            np.array(["A"] * 8),
            "demographic_parity",
        )
        assert math.isnan(one_group.violation)
        assert one_group.is_satisfied is None
        assert all(math.isnan(v) for v in one_group.group_violations.values())
        assert any("fewer than 2 groups" in m for m in caught_one), caught_one

        no_pos, caught_pos = _loudly(
            compute_constraint_violation,
            np.zeros(8, dtype=int),
            self.YP,
            self.G,
            "equal_opportunity",
        )
        assert math.isnan(no_pos.violation)
        assert no_pos.is_satisfied is None
        assert set(no_pos.unmeasured) == {"A", "B"}
        assert any("empty denominator" in m for m in caught_pos), caught_pos

    def test_a_determinate_breach_on_partial_evidence_keeps_the_finding(self):
        """The other direction of the same unit: a measured arm already outside
        tolerance is a verdict, and refusing it would delete a real finding.
        Three groups, B holding no positive label, A tpr 1.0 and C tpr 0.5:
        violation 0.5, is_satisfied False, violation_is_lower_bound True,
        unmeasured ('B',).
        """
        result, caught = _loudly(
            compute_constraint_violation,
            np.array([1, 0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0, 0]),
            np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"]),
            "equal_opportunity",
        )

        assert result.is_satisfied is False
        assert result.violation == pytest.approx(0.5)
        assert result.violation_is_lower_bound is True
        assert result.unmeasured == ("B",)
        assert any("LOWER BOUND" in m for m in caught), caught


# ── the threshold optimisers: a grid that chose nothing ──────────────────────


class TestAGridThatCannotChooseIsRefused:
    @pytest.mark.parametrize(
        "cls", [ThresholdOptimizer, GroupThresholdOptimizer, MultiObjectiveThresholdOptimizer]
    )
    @pytest.mark.parametrize("n_thresholds", [0, 1])
    def test_a_grid_below_two_candidates_is_refused(self, cls, n_thresholds):
        """Measured BEFORE the fix on 60 rows and two groups, n_thresholds=0:

            GroupThresholdOptimizer -> group_thresholds {'a': 0.5, 'b': 0.5}
            ThresholdOptimizer      -> global_threshold 0.5
            both: is_feasible a measured True/False, and ZERO warnings

        0.5 is each class's own seed, never a candidate anything examined:
        ``np.linspace(.., 0)`` is empty so the search body never runs. The two
        scans cannot cover it (``warn_if_unmeasured`` returns early at 0
        candidates, correctly), and ``summary()`` then printed "chosen by
        constraint violation alone" and "the first the scan saw, NOT a minimum",
        both describing a selection that never happened. n_thresholds=1 is the
        same defect quieter: ThresholdOptimizer returned 0.01 and
        GroupThresholdOptimizer 0.05, the grid's lower bound, whatever the data
        said, with objective_measured True beside it.
        """
        with pytest.raises(ValueError, match="candidate threshold"):
            cls(n_thresholds=n_thresholds)

    def test_control_a_searchable_grid_still_fits_an_interior_optimum(self):
        """CONTROL. Two candidates is the floor, and a real grid must still
        return a threshold from INSIDE it rather than an endpoint seed. Measured
        after the fix on 200 rows: {'a': 0.4908, 'b': 0.5459}, objective and
        constraint both measured over 2500 candidates, no warnings.
        """
        y_true, y_prob, groups = _scored_population()
        opt, caught = _loudly(
            GroupThresholdOptimizer().fit,
            y_true=y_true,
            y_prob=y_prob,
            sensitive_attr=groups,
        )

        details = opt.result_.optimization_details
        assert caught == [], caught
        assert details["objective_measured"] is True
        assert details["constraint_measured"] is True
        assert details["n_objective_measured"] == details["n_objective_candidates"] > 1
        thresholds = list(opt.result_.group_thresholds.values())
        assert all(0.05 < float(t) < 0.95 for t in thresholds), thresholds
        assert opt.result_.is_feasible is True
        assert "COULD NOT CHECK" not in opt.result_.summary()


class TestGroupThresholdOptimizerSaysWhatDecidedTheThreshold:
    def test_a_single_group_fit_is_not_assessed_rather_than_feasible(self):
        """EVIDENCE for the unit recorded as "no fabrication seen, pins do not
        cover enough input". Measured on 200 rows of ONE group: is_feasible None,
        constraint_measured False, n_violation_measured 0 of 24, a warning naming
        "fewer than 2 groups", and the summary carrying both the "?" status and
        the COULD NOT CHECK block. The thresholds returned ({'a': 0.05}) are the
        first grid point, and the artifact says so.
        """
        y_true, y_prob, _ = _scored_population()
        opt, caught = _loudly(
            GroupThresholdOptimizer(n_thresholds=8).fit,
            y_true=y_true,
            y_prob=y_prob,
            sensitive_attr=np.array(["a"] * len(y_true)),
        )

        details = opt.result_.optimization_details
        assert opt.result_.is_feasible is None
        assert details["constraint_measured"] is False
        assert details["n_violation_measured"] == 0
        assert details["n_violation_candidates"] > 0
        assert any("fewer than 2 groups" in m for m in caught), caught
        summary = opt.result_.summary()
        assert "Feasible: not assessed" in summary
        assert "NOT a minimum" in summary

    def test_an_objective_that_is_nan_everywhere_decided_nothing_and_says_so(self):
        """The other axis of the same unit: the CONSTRAINT is measurable and the
        OBJECTIVE is not. Measured on 200 rows, two groups, y_true all ones, so
        balanced_accuracy has no value at any of the 64 candidates:
        objective_measured False, n_objective_measured 0, constraint measured at
        57 of 64, and a warning saying the objective "could not be computed at
        ANY" candidate. Without the guard the incumbent's score becomes NaN and
        every later measured candidate loses ``measured > nan``.
        """
        _, y_prob, groups = _scored_population()
        opt, caught = _loudly(
            GroupThresholdOptimizer(n_thresholds=8).fit,
            y_true=np.ones(len(y_prob), dtype=int),
            y_prob=y_prob,
            sensitive_attr=groups,
        )

        details = opt.result_.optimization_details
        assert details["objective_measured"] is False
        assert details["n_objective_measured"] == 0
        assert details["n_objective_candidates"] == 64
        assert math.isnan(opt.result_.performance_metrics["balanced_accuracy"])
        assert any("could not be computed at ANY" in m for m in caught), caught
        assert "NOT an optimum" in opt.result_.summary()


class TestTheParetoFallbackReachesTheReader:
    def test_a_group_count_other_than_two_says_there_is_no_frontier(self):
        """Measured BEFORE the fix on 200 rows and THREE groups:
        len(pareto_frontier_) == 1, optimization_details carrying
        pareto_frontier_computed False and the reason, one warning at fit time,
        and ``summary()`` printing

            Feasible: True
            equalized_odds: 0.0404 (with a tick)

        with NO COULD NOT CHECK block at all. One entry in pareto_frontier_ reads
        as "the sweep found exactly one non-dominated configuration" when no
        sweep ran: the candidate loop runs only for exactly 2 groups. The flag
        was on the object and the words were in a warning that is gone by the
        time anyone reads the result.
        """
        y_true, y_prob, _ = _scored_population(n=200)
        groups = np.array(["a"] * 70 + ["b"] * 70 + ["c"] * 60)

        opt, caught = _loudly(
            MultiObjectiveThresholdOptimizer(n_thresholds=6).fit,
            y_true=y_true,
            y_prob=y_prob,
            sensitive_attr=groups,
        )

        details = opt.result_.optimization_details
        assert details["pareto_frontier_computed"] is False
        assert len(opt.pareto_frontier_) == 1
        assert any("no Pareto frontier was computed" in m for m in caught), caught
        summary = opt.result_.summary()
        assert "COULD NOT CHECK" in summary, summary
        assert "NO Pareto frontier was computed" in summary
        assert "exactly 2 groups and this data has 3" in summary

    def test_control_a_real_two_group_frontier_claims_nothing_of_the_kind(self):
        """CONTROL. Measured after the fix on 200 rows and two groups:
        pareto_frontier_computed True, pareto_frontier_ranked True, objectives
        accuracy and f1_score both measured, and no fallback line anywhere in the
        summary. A guard that stamped the caveat unconditionally would fail here.
        """
        y_true, y_prob, groups = _scored_population()
        opt, caught = _loudly(
            MultiObjectiveThresholdOptimizer(n_thresholds=6).fit,
            y_true=y_true,
            y_prob=y_prob,
            sensitive_attr=groups,
        )

        details = opt.result_.optimization_details
        assert caught == [], caught
        assert details["pareto_frontier_computed"] is True
        assert details["pareto_frontier_ranked"] is True
        assert details["objectives_unmeasured"] == []
        assert "NO Pareto frontier" not in opt.result_.summary()

    def test_an_unmeasurable_objective_still_says_the_frontier_ranked_on_nothing(self):
        """The neighbouring state, kept pinned here because my summary() change
        sits three lines above the block that prints it. Measured on 200 rows,
        two groups, y_true all ones, objectives=['balanced_accuracy']: frontier
        size 10 (nothing can dominate anything), objectives_measured [],
        pareto_frontier_ranked False, and the summary saying "This is NOT a
        Pareto frontier".
        """
        _, y_prob, groups = _scored_population()
        opt, caught = _loudly(
            MultiObjectiveThresholdOptimizer(n_thresholds=6, objectives=["balanced_accuracy"]).fit,
            y_true=np.ones(len(y_prob), dtype=int),
            y_prob=y_prob,
            sensitive_attr=groups,
        )

        details = opt.result_.optimization_details
        assert details["objectives_measured"] == []
        assert details["objectives_unmeasured"] == ["balanced_accuracy"]
        assert details["pareto_frontier_ranked"] is False
        assert len(opt.pareto_frontier_) > 1
        assert "NOT a Pareto frontier" in opt.result_.summary()


# ── the calibrators ─────────────────────────────────────────────────────────


class TestIsotonicIgnoresTheWeightScale:
    Y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.8, 0.9])
    PROBE = np.array([0.1, 0.25, 0.4, 0.9])

    def test_a_uniform_rescale_of_the_weights_does_not_move_the_map(self):
        """Measured BEFORE the fix on these 8 rows:

            every weight 1.0   -> transform(PROBE) = [0, 0.3, 0.6, 1.0]
            every weight 1e-11 -> the same
            every weight 1e-13 -> [0, 0.03, 0.06, 0.1]

        every calibrated probability 10x too small, and the published record was
        byte-identical in all three (n_blocks 4, n_knots 8, n_distinct_values 3,
        warnings []). The tie-collapse divided by
        ``np.clip(w_per_x, 1e-12, None)``, a MAGNITUDE floor from before the
        weightless-row screen existed: it can no longer protect anything (every
        unique x now holds a strictly positive weight) and its only remaining
        effect was to divide by the wrong number.
        """
        reference = IsotonicCalibrator().fit(self.Y, self.SCORES, np.ones(8))
        expected = reference.transform(self.PROBE)
        assert expected == pytest.approx([0.0, 0.3, 0.6, 1.0])

        for scale in (1e-13, 1e-11, 1.0, 1e6):
            cal, caught = _loudly(IsotonicCalibrator().fit, self.Y, self.SCORES, np.full(8, scale))
            assert cal.transform(self.PROBE) == pytest.approx(expected), scale
            assert cal.y_values_ == pytest.approx(reference.y_values_), scale
            assert caught == [], (scale, caught)

    def test_control_a_weightless_isotonic_fit_still_refuses(self):
        """CONTROL for the change above: the denominator guard must not resurrect
        the fit that has nothing to fit. Every weight 0 on the same rows:
        n_weighted_rows 0, n_distinct_values 0, transform NaN, one UserWarning
        and a durable note.
        """
        cal, caught = _loudly(IsotonicCalibrator().fit, self.Y, self.SCORES, np.zeros(8))
        out = cal.transform(self.PROBE)

        assert cal.fit_result.parameters["n_weighted_rows"] == 0
        assert cal.fit_result.parameters["n_distinct_values"] == 0
        assert np.all(np.isnan(out)), out
        assert any("identified NO isotonic map" in m for m in caught), caught
        assert any("identified NO isotonic map" in n for n in cal.fit_result.warnings)


class TestTemperatureReadsTheWEIGHTEDRows:
    def test_all_weights_zero_identifies_no_temperature(self):
        """EVIDENCE for the unit recorded as "no fabrication seen, pins do not
        cover enough input". The existing degeneracy pin varies the SCORES (a
        constant column); this varies the WEIGHTS and leaves the scores fully
        informative, which is the axis that decides whether the guard reads
        ``logits[positive_weight]`` or all of ``logits``.

        Measured on 60 informative scores (60 distinct values) with every weight
        0: temperature_ NaN, nll NaN, iterations 0, transform NaN, a UserWarning
        naming "rank-deficient" and "0 of 60 rows", and the same note in the
        durable record. Reading all of logits instead would publish a finite
        temperature from an objective that is flat everywhere, which is the
        38.2027 the gate's own comment records for a NaN weight.
        """
        rng = np.random.RandomState(3)
        scores = rng.rand(60)
        y = (scores + rng.normal(0, 0.2, 60) > 0.5).astype(int)
        assert np.unique(scores).size == 60
        assert len(np.unique(y)) == 2

        cal, caught = _loudly(TemperatureScaling().fit, y, scores, np.zeros(60))
        out = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert np.isnan(cal.temperature_)
        assert np.isnan(cal.fit_result.parameters["temperature"])
        assert np.isnan(cal.fit_result.fit_metrics["nll"])
        assert np.all(np.isnan(out)), out
        joined = " ".join(cal.fit_result.warnings)
        assert "rank-deficient" in joined, cal.fit_result.warnings
        assert "0 of 60 rows" in joined, cal.fit_result.warnings
        assert any("could-not-check" in m for m in caught), caught

    def test_control_the_same_scores_at_unit_weight_fit_a_real_temperature(self):
        """CONTROL. The identical rows at weight 1.0 must still fit: measured
        temperature 0.3385, nll 0.3095, transform strictly increasing and finite,
        no warnings. A refusal that also fired here would be the reverse defect.
        """
        rng = np.random.RandomState(3)
        scores = rng.rand(60)
        y = (scores + rng.normal(0, 0.2, 60) > 0.5).astype(int)

        cal, caught = _loudly(TemperatureScaling().fit, y, scores, np.ones(60))
        probe = cal.transform(np.array([0.1, 0.5, 0.9]))

        assert caught == [], caught
        assert cal.fit_result.warnings == [], cal.fit_result.warnings
        assert np.isfinite(cal.temperature_) and 0.02 < cal.temperature_ < 99.0
        assert np.all(np.isfinite(probe))
        assert np.all(np.diff(probe) > 0), probe


# ── the dashboard: what the CHART says about a partial comparison ────────────


class TestDashboardNamesAPartialDisparity:
    @staticmethod
    def _population(sizes, seed=11):
        n = sum(sizes.values())
        rng = np.random.RandomState(seed)
        y_prob = rng.rand(n)
        y_true = (y_prob + rng.normal(0, 0.25, n) > 0.5).astype(int)
        attr = np.concatenate([np.full(k, name) for name, k in sizes.items()])
        calibrated = np.clip(y_prob * 0.9 + 0.05, 0, 1)
        return y_true, y_prob, attr, calibrated

    def test_a_disparity_measured_over_two_of_three_groups_says_so_on_the_chart(self):
        """Measured BEFORE the fix on 135 rows with groups m=60, f=60, x=15:

            the By Group panels drew all THREE curves, x visibly the worst
              (legend: m ECE 0.095, f ECE 0.130, x ECE 0.182)
            the ECE Disparity bar read 0.0351, UNDER the red 0.05 target line,
              because x was excluded from it (n_groups_compared 2 of 3)
            chart annotations: none

        A per-group CURVE needs 10 rows and the ECE group comparison needs 30, so
        the group that puts the disparity over target (0.182 - 0.095 = 0.087) was
        named in the legend beside a bar saying the model is inside it. The
        exclusion was in metadata['excluded_groups'] and in a build-time warning;
        the chart is the surface a reader meets.
        """
        y_true, y_prob, attr, calibrated = self._population({"m": 60, "f": 60, "x": 15})

        fig, caught = _loudly(create_calibration_dashboard, y_true, y_prob, attr, calibrated)

        notes = _annotations(fig)
        assert any("2 of 3 groups" in n and "x excluded" in n for n in notes), notes
        assert any(n.startswith("before: ") for n in notes), notes
        assert any(n.startswith("after: ") for n in notes), notes
        # The excluded group really is on the chart as a curve, which is what
        # makes the silent bar a contradiction rather than an omission.
        assert any(label.startswith("x (ECE=") for label in _legend_labels(fig)), _legend_labels(
            fig
        )
        assert any("excluded from the expected calibration error" in m for m in caught)

    def test_an_unmeasurable_disparity_is_still_marked_not_measured(self):
        """The neighbouring state, one group only: max_group_disparity is NaN and
        a NaN bar draws exactly what a measured 0.0 draws. Measured: both bars
        annotated "before: not measured" / "after: not measured".
        """
        y_true, y_prob, attr, calibrated = self._population({"m": 120})

        fig, _ = _loudly(create_calibration_dashboard, y_true, y_prob, attr, calibrated)

        notes = _annotations(fig)
        assert "before: not measured" in notes, notes
        assert "after: not measured" in notes, notes

    def test_control_a_complete_comparison_carries_no_caveat(self):
        """CONTROL. Two groups of 60, both above every floor: the disparity bar is
        a finite measured number and the chart must say nothing about exclusions
        or unmeasured bars. Measured after the fix: bars [0.092, 0.0104] before
        and [0.0736, 0.0258] after, annotations [].
        """
        y_true, y_prob, attr, calibrated = self._population({"m": 60, "f": 60})

        fig, caught = _loudly(create_calibration_dashboard, y_true, y_prob, attr, calibrated)

        notes = _annotations(fig)
        assert notes == [] or all("not measured" not in n and "excluded" not in n for n in notes), (
            notes
        )
        assert caught == [], caught
