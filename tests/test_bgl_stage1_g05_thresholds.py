"""Beta Go-Live stage 1, group G05: threshold optimization must not fabricate.

Three critical findings, all of the same shape: a value nobody measured handed
back as a confident verdict, with no warning and no second field saying so.

1. ``MultiObjectiveThresholdOptimizer.predict`` turned two could-not-check
   conditions into a REJECT decision *about a person*: a NaN score (``nan >=
   thresh`` is False under IEEE 754) and a sensitive-attribute value the
   optimizer was never fitted for (it matches no mask and keeps the
   ``np.zeros`` seed).
2. ``_metric_disparity`` took max-min over a ONE-entry dict, got exactly 0.0,
   and ``compute_constraint_violation`` graded ``bool(0.0 <= tolerance)`` as a
   measured PASS -- the library's headline verdict, rendered as a tick.
3. The same single-group 0.0 reached ``ThresholdAnalyzer.find_optimal_threshold``
   and produced ``is_feasible=True`` with ``n_thresholds_not_assessed=0``.

Every assertion below is at a PUBLIC entry point, and every "it refuses" test
has a CONTROL beside it proving healthy data still measures correctly. A fix
that makes everything refuse is a worse defect than the one it replaces.
"""

import math
import warnings

import numpy as np
import pytest

import vfairness
from vfairness import (
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdAnalyzer,
    ThresholdOptimizer,
    compute_constraint_violation,
)

ALL_CONSTRAINTS = [
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "predictive_parity",
]


def _healthy_two_group(n=200, seed=0):
    """Scores that genuinely separate, split over two groups."""
    rng = np.random.default_rng(seed)
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    return y_true, y_prob, groups


def _fitted_multi(**kwargs):
    y_true, y_prob, groups = _healthy_two_group()
    opt = MultiObjectiveThresholdOptimizer(constraint="equalized_odds", **kwargs)
    opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
    return opt


# ---------------------------------------------------------------------------
# Finding 1: predict() decided people it had not measured.
# ---------------------------------------------------------------------------


class TestPredictRefusesWhatItCannotScore:
    def test_a_row_with_no_score_is_refused_not_rejected(self):
        """BEFORE: 40 NaN rows -> np.unique(..., counts) == ([0], [40]), i.e.
        every one REJECTED, with zero warnings and an int64 array carrying no
        mask. A refusal a caller cannot see is a fabricated decision."""
        opt = _fitted_multi()
        with pytest.raises(ValueError, match="no usable score"):
            opt.predict(np.full(40, np.nan), np.array(["a"] * 20 + ["b"] * 20))

    def test_one_unscored_row_among_measured_ones_is_not_silently_denied(self):
        """BEFORE: predict([0.9, nan, 0.1], ['a','a','a']) -> [1 0 0]. The
        middle row is byte-identical to the measured rejection beside it."""
        opt = _fitted_multi()
        with pytest.raises(ValueError) as excinfo:
            opt.predict(np.array([0.9, np.nan, 0.1]), np.array(["a", "a", "a"]))
        # The message must locate the unscored row, not just complain.
        assert "1 of 3" in str(excinfo.value)
        assert "[1]" in str(excinfo.value)

    def test_infinite_scores_are_refused_too(self):
        """+inf is not a probability either, and it lands on the OTHER side of
        every threshold, so this one fabricates an ACCEPT."""
        opt = _fitted_multi()
        with pytest.raises(ValueError, match="no usable score"):
            opt.predict(np.array([0.4, np.inf]), np.array(["a", "b"]))

    def test_a_group_nobody_fitted_is_refused_not_denied(self):
        """BEFORE: 30 rows of an unfitted group 'c' -> ([0], [30]). An entire
        cohort denied, the applicant at p=0.99 included, silently."""
        opt = _fitted_multi()
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match=r"No threshold was fitted for group\(s\) \['c'\]"):
            opt.predict(rng.random(30), np.array(["c"] * 30))

    def test_the_refusal_names_the_groups_that_were_fitted(self):
        opt = _fitted_multi()
        with pytest.raises(ValueError) as excinfo:
            opt.predict(np.array([0.99, 0.5]), np.array(["c", "a"]))
        assert "['a', 'b']" in str(excinfo.value)

    def test_the_high_scoring_applicant_is_not_quietly_rejected(self):
        """The worst single row in the original measurement: p=0.99 in an
        unfitted group came back 0. Pin that nothing returns 0 for it."""
        opt = _fitted_multi()
        with pytest.raises(ValueError):
            opt.predict(np.array([0.99]), np.array(["c"]))

    # -- CONTROL ----------------------------------------------------------

    def test_control_healthy_scores_still_produce_measured_decisions(self):
        opt = _fitted_multi()
        y_prob = np.array([0.99, 0.95, 0.9, 0.05, 0.02, 0.01])
        groups = np.array(["a", "b", "a", "b", "a", "b"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pred = opt.predict(y_prob, groups)
        assert pred.shape == (6,)
        # A real decision boundary: the top scores accepted, the bottom rejected.
        assert set(np.unique(pred)) == {0, 1}
        assert pred[0] == 1 and pred[1] == 1
        assert pred[-1] == 0 and pred[-2] == 0

    def test_control_the_whole_fitted_population_still_predicts(self):
        y_true, y_prob, groups = _healthy_two_group()
        opt = MultiObjectiveThresholdOptimizer(constraint="equalized_odds")
        opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
        pred = opt.predict(y_prob, groups)
        assert len(pred) == len(y_prob)
        assert 0 < pred.mean() < 1, "a working threshold accepts some and rejects some"

    def test_control_the_siblings_refuse_the_same_way(self):
        """The guard is above the dispatch: all three optimizers share the
        precondition, so fixing only the multi-objective one would move the
        fabrication to its siblings."""
        y_true, y_prob, groups = _healthy_two_group()
        for cls in (ThresholdOptimizer, GroupThresholdOptimizer):
            opt = cls(constraint="demographic_parity")
            opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            with pytest.raises(ValueError, match="no usable score"):
                opt.predict(np.array([0.4, np.nan]), np.array(["a", "b"]))
            # CONTROL on the same object: healthy input still decides.
            pred = opt.predict(np.array([0.99, 0.01]), np.array(["a", "b"]))
            assert pred.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# Finding 2: a between-group verdict over one group.
# ---------------------------------------------------------------------------


class TestOneGroupIsNotAPass:
    @pytest.mark.parametrize("constraint", ALL_CONSTRAINTS)
    def test_the_public_constraint_check_says_could_not_check(self, constraint):
        """BEFORE, for all four constraints: violation 0.0, is_satisfied True,
        group_violations {'a': 0.0}, and ZERO warnings."""
        y_true, y_prob, _ = _healthy_two_group()
        groups = np.array(["a"] * 200)
        y_pred = (y_prob >= 0.5).astype(int)

        with pytest.warns(UserWarning, match="fewer than 2 groups"):
            result = compute_constraint_violation(y_true, y_pred, groups, constraint=constraint)

        assert result.is_satisfied is None, "could-not-check is not a pass"
        assert math.isnan(result.violation)
        assert all(math.isnan(v) for v in result.group_violations.values())
        # The dict a report renders from must carry it too.
        assert result.to_dict()["is_satisfied"] is None

    def test_the_guard_sits_above_the_reference_group_dispatch(self):
        """Both branches of _metric_disparity fabricate 0.0 on one group
        (max-min, and abs(v - v) for the reference form). Guarding only the
        default branch would move the defect into the opt-in one."""
        y_true, y_prob, _ = _healthy_two_group()
        groups = np.array(["a"] * 200)
        y_pred = (y_prob >= 0.5).astype(int)

        with pytest.warns(UserWarning, match="fewer than 2 groups"):
            result = compute_constraint_violation(
                y_true,
                y_pred,
                groups,
                constraint="demographic_parity",
                reference_group="a",
            )
        assert result.is_satisfied is None
        assert math.isnan(result.violation)

    def test_the_warning_names_the_real_reason(self):
        """ "Empty denominator" would be a false explanation here: every rate
        was measurable, there was simply nothing to compare it with."""
        y_true, y_prob, _ = _healthy_two_group()
        groups = np.array(["a"] * 200)
        y_pred = (y_prob >= 0.5).astype(int)
        with pytest.warns(UserWarning) as record:
            compute_constraint_violation(y_true, y_pred, groups, constraint="equalized_odds")
        messages = [str(w.message) for w in record]
        assert any("fewer than 2 groups" in m for m in messages)
        assert not any("empty denominator" in m for m in messages)

    def test_the_fitted_optimizer_does_not_render_a_tick(self):
        """BEFORE: summary() rendered 'Feasible: True' and
        '[tick] equalized_odds: 0.0000' for a comparison nobody made."""
        y_true, y_prob, _ = _healthy_two_group()
        groups = np.array(["a"] * 200)
        opt = MultiObjectiveThresholdOptimizer(constraint="equalized_odds", n_thresholds=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)

        assert opt.result_.is_feasible is None
        text = opt.result_.summary()
        assert "Feasible: not assessed" in text
        assert "Feasible: True" not in text
        assert "✓ equalized_odds" not in text

    def test_a_one_point_fallback_is_not_reported_as_a_pareto_frontier(self):
        """The candidate sweep runs only for exactly 2 groups. With any other
        count it never ran at all, yet pareto_frontier_ held one entry and read
        like a frontier with one non-dominated point."""
        y_true, y_prob, _ = _healthy_two_group()
        for groups, n_groups in [
            (np.array(["a"] * 200), 1),
            (np.array(["a"] * 67 + ["b"] * 67 + ["c"] * 66), 3),
        ]:
            opt = MultiObjectiveThresholdOptimizer(constraint="demographic_parity", n_thresholds=6)
            with pytest.warns(UserWarning, match="no Pareto frontier was computed"):
                opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            details = opt.result_.optimization_details
            assert details["pareto_frontier_computed"] is False
            assert str(n_groups) in details["pareto_fallback_reason"]

    # -- CONTROL ----------------------------------------------------------

    @pytest.mark.parametrize("constraint", ALL_CONSTRAINTS)
    def test_control_two_groups_still_measure_a_real_breach(self, constraint):
        """Every rate in every group has a non-empty denominator here, so a
        NaN would mean the fix over-refused. Group a is accepted 80% of the
        time and is usually right; group b is accepted 10% of the time and is
        usually wrong. All four constraints are breached by more than 0.5."""
        # group a: 160 accepted (150 truly positive), 40 rejected (5 positive)
        # group b:  20 accepted (  4 truly positive), 180 rejected (60 positive)
        a_pred = np.array([1] * 160 + [0] * 40)
        a_true = np.array([1] * 150 + [0] * 10 + [1] * 5 + [0] * 35)
        b_pred = np.array([1] * 20 + [0] * 180)
        b_true = np.array([1] * 4 + [0] * 16 + [1] * 60 + [0] * 120)

        groups = np.array(["a"] * 200 + ["b"] * 200)
        y_pred = np.concatenate([a_pred, b_pred])
        y_true = np.concatenate([a_true, b_true])

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = compute_constraint_violation(y_true, y_pred, groups, constraint=constraint)
        assert result.is_satisfied is False, "a real breach must still be a breach"
        assert result.violation > 0.5
        assert all(math.isfinite(v) for v in result.group_violations.values())

    def test_control_two_equal_groups_still_measure_a_real_pass(self):
        n = 400
        groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 2, n)
        y_pred = np.tile([1, 0], n // 2)  # identical rate in both groups
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = compute_constraint_violation(
                y_true, y_pred, groups, constraint="demographic_parity"
            )
        assert result.is_satisfied is True
        assert result.violation == pytest.approx(0.0, abs=1e-12)

    def test_control_two_groups_still_get_a_real_pareto_sweep(self):
        y_true, y_prob, groups = _healthy_two_group()
        opt = MultiObjectiveThresholdOptimizer(constraint="demographic_parity", n_thresholds=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
        assert opt.result_.optimization_details["pareto_frontier_computed"] is True
        assert isinstance(opt.result_.is_feasible, bool)


# ---------------------------------------------------------------------------
# Finding 3: the same 0.0 reaching ThresholdAnalyzer.
# ---------------------------------------------------------------------------


class TestThresholdAnalyzerOnOneGroup:
    @staticmethod
    def _single_group():
        rng = np.random.default_rng(4)
        return rng.integers(0, 2, 200), rng.random(200), np.array(["A"] * 200)

    def test_find_optimal_threshold_is_not_assessed(self):
        """BEFORE: optimal_threshold=0.0298, is_feasible=True,
        n_thresholds_searched=100, n_thresholds_not_assessed=0, warnings=[]."""
        y_true, y_prob, groups = self._single_group()
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.find_optimal_threshold(constraint="demographic_parity")
        assert result["is_feasible"] is None
        assert result["optimal_threshold"] is None
        assert result["n_thresholds_not_assessed"] == result["n_thresholds_searched"]
        assert any("fewer than 2 groups" in str(w.message) for w in caught)

    def test_analyze_threshold_is_not_assessed(self):
        """BEFORE: violation=0.0, is_satisfied=True, group_metrics={'A': 0.56}."""
        y_true, y_prob, groups = self._single_group()
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = analyzer.analyze_threshold(0.5, constraints=["demographic_parity"])
        violation = report.constraint_violations["demographic_parity"]
        assert violation.is_satisfied is None
        assert math.isnan(violation.violation)

    def test_the_feasible_region_is_not_assessed(self):
        """BEFORE: assessed=True, status='feasible', n_not_assessed=0."""
        y_true, y_prob, groups = self._single_group()
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            region = analyzer.find_feasible_region(constraint="demographic_parity")
        assert region.assessed is False
        assert region.status == "not_assessed"
        assert region.n_not_assessed == region.n_searched

    def test_equalized_odds_behaves_the_same_way(self):
        """Both constraints reported identically before, so both are pinned."""
        y_true, y_prob, groups = self._single_group()
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = analyzer.analyze_threshold(0.5, constraints=["equalized_odds"])
        violation = report.constraint_violations["equalized_odds"]
        assert violation.is_satisfied is None
        assert math.isnan(violation.violation)

    # -- CONTROL ----------------------------------------------------------

    def test_control_two_groups_still_find_a_measured_optimum(self):
        y_true, y_prob, groups = _healthy_two_group(n=400, seed=3)
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = analyzer.find_optimal_threshold(constraint="demographic_parity")
        assert result["is_feasible"] is True
        assert isinstance(result["optimal_threshold"], float)
        assert result["n_thresholds_not_assessed"] == 0

    def test_control_a_measured_breach_is_still_a_breach_not_a_refusal(self):
        """Two groups, scores that separate them completely: at threshold 0.5
        group A is accepted outright and group B rejected outright. That is a
        MEASURED violation of 1.0, and the fix must not turn it into a
        could-not-check. The region is `assessed` for the same reason: every
        one of the searched thresholds was measurable."""
        n = 200
        groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        y_prob = np.where(groups == "A", 0.9, 0.1)
        y_true = (np.arange(n) % 2).astype(int)
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = analyzer.analyze_threshold(0.5, constraints=["demographic_parity"])
            region = analyzer.find_feasible_region(constraint="demographic_parity")
        violation = report.constraint_violations["demographic_parity"]
        assert violation.is_satisfied is False, "a measured breach is not a could-not-check"
        assert violation.violation == pytest.approx(1.0)
        assert region.assessed is True
        assert region.n_not_assessed == 0


def test_the_module_still_exports_what_it_did():
    """A refusal that breaks the import surface is not a fix."""
    for name in (
        "MultiObjectiveThresholdOptimizer",
        "GroupThresholdOptimizer",
        "ThresholdOptimizer",
        "ThresholdAnalyzer",
        "compute_constraint_violation",
    ):
        assert hasattr(vfairness, name)
