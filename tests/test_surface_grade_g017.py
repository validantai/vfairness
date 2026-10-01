"""Surface grade g017: the threshold optimizers must not report an unmeasured
value as a measurement, and must not refuse one they can measure.

The finding this file was written for (BGL-S2c, measured 2026-09-17).
``MultiObjectiveThresholdOptimizer`` ranks candidate threshold pairs by Pareto
dominance over the objectives it was asked for. When an objective has no value
on the data (``balanced_accuracy`` with no negative label, say), every
comparison against its NaN is False, so no candidate can dominate any other and
they ALL come back non-dominated. Measured on 200 rows, two groups,
``n_thresholds=8``:

    objectives=['accuracy']           -> len(pareto_frontier_) == 1
    objectives=['balanced_accuracy']  -> len(pareto_frontier_) == 10

on the SAME data, and both results carried the identical
``optimization_details`` ending ``'pareto_frontier_computed': True``. Nothing in
``to_dict()`` and nothing in ``summary()`` told the two apart: the less the
optimizer could compute, the more results it reported, which is the T-02
signature this class already refuses for an unknown objective NAME and did not
refuse for a known objective with no VALUE. The only disclosure was a warning,
and a warning is gone by the time anyone reads the artifact.

The same three lines carried the reverse defect. ``any()`` over an EMPTY
candidate list is False, and the candidate sweep runs only for exactly two
groups, so a one-group fit warned "2 of 2 objective(s) could not be computed on
this data for ANY candidate (accuracy, f1_score)" over data where accuracy is
trivially computable. Nothing was measured because nothing was scanned.

Every refusal assertion below has a CONTROL beside it computing the expected
value independently, because an optimizer that refuses everything would pass
the first half of this file while measuring nothing.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.post_processing.threshold_optimization.optimizer import (
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdOptimizer,
)

# ---------------------------------------------------------------- fixtures


def _two_groups_with_a_real_gap(n: int = 200, seed: int = 3):
    """Group b's scores are shifted down, so a single threshold produces a
    large, findable acceptance-rate gap. Verified in
    ``test_the_fixture_really_carries_a_disparity``."""
    rng = np.random.default_rng(seed)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    y_prob = rng.random(n)
    y_prob = np.where(groups == "b", y_prob * 0.6, y_prob)
    y_true = (rng.random(n) < y_prob).astype(int)
    return y_true, y_prob, groups


def _quiet_fit(optimizer, y_true, y_prob, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)


def _loud_fit(optimizer, y_true, y_prob, groups):
    """Fit with warnings RECORDED, never suppressed: a refusal carried only in
    a warning is invisible otherwise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
    return optimizer, [str(w.message) for w in caught]


def test_the_fixture_really_carries_a_disparity():
    """If this ever stops holding, every refusal test below is passing for the
    wrong reason."""
    _, y_prob, groups = _two_groups_with_a_real_gap()
    gap = (y_prob[groups == "a"] >= 0.5).mean() - (y_prob[groups == "b"] >= 0.5).mean()
    assert gap > 0.2, f"fixture gap is only {gap}"


# ------------------------------------------------- BGL-S2c, the found defect


class TestParetoFrontierSaysWhatItRankedOn:
    @staticmethod
    def _all_ones_labels(n: int = 200):
        """``balanced_accuracy`` needs a true-negative rate, and there is no
        negative label here, so it is undefined at EVERY candidate while
        ``accuracy`` stays perfectly measurable."""
        _, y_prob, groups = _two_groups_with_a_real_gap(n=n)
        return np.ones(n, dtype=int), y_prob, groups

    def test_an_objective_with_no_value_is_recorded_on_the_artifact(self):
        y_true, y_prob, groups = self._all_ones_labels()
        opt = MultiObjectiveThresholdOptimizer(
            constraint="equalized_odds",
            n_thresholds=8,
            objectives=["accuracy", "balanced_accuracy"],
        )
        _quiet_fit(opt, y_true, y_prob, groups)

        details = opt.result_.optimization_details
        assert details["objectives"] == ["accuracy", "balanced_accuracy"]
        assert details["objectives_unmeasured"] == ["balanced_accuracy"]
        assert details["objectives_measured"] == ["accuracy"]
        # It DID rank, on the one objective that had values. Saying otherwise
        # would throw away a real ranking.
        assert details["pareto_frontier_ranked"] is True
        assert opt.result_.to_dict()["optimization_details"]["objectives_unmeasured"] == [
            "balanced_accuracy"
        ]

    def test_the_reader_of_the_summary_is_told(self):
        """BEFORE: the COULD NOT CHECK block named the constraint and nothing
        else, so the frontier read as a frontier over both objectives."""
        y_true, y_prob, groups = self._all_ones_labels()
        opt = MultiObjectiveThresholdOptimizer(
            constraint="equalized_odds",
            n_thresholds=8,
            objectives=["accuracy", "balanced_accuracy"],
        )
        _quiet_fit(opt, y_true, y_prob, groups)

        text = opt.result_.summary()
        assert "COULD NOT CHECK" in text
        assert "balanced_accuracy" in text.split("COULD NOT CHECK")[1]
        assert "NOT over the objectives that were requested" in text

    def test_a_frontier_that_ranked_on_nothing_says_so(self):
        """The worst case: no requested objective has a value, so no candidate
        dominates any other and every one of them is returned as
        non-dominated. Measured before the fix: 10 points, with
        ``pareto_frontier_computed`` True and no other trace."""
        y_true, y_prob, groups = self._all_ones_labels()
        opt = MultiObjectiveThresholdOptimizer(
            constraint="equalized_odds", n_thresholds=8, objectives=["balanced_accuracy"]
        )
        _quiet_fit(opt, y_true, y_prob, groups)

        details = opt.result_.optimization_details
        assert details["pareto_frontier_ranked"] is False
        assert details["objectives_measured"] == []
        assert len(opt.pareto_frontier_) > 1, (
            "this is the input that inflated the frontier; if it no longer does, "
            "the disclosure is being asserted on the wrong scenario"
        )
        text = opt.result_.summary()
        assert "This is NOT a Pareto frontier" in text

    def test_no_sweep_is_not_reported_as_an_unmeasurable_objective(self):
        """The reverse defect. The candidate sweep runs for exactly two groups,
        so on one group there were no candidates at all, and ``any()`` over an
        empty list declared accuracy uncomputable on data where it is
        trivially computable."""
        y_true, y_prob, _ = _two_groups_with_a_real_gap()
        one_group = np.array(["a"] * len(y_true))
        opt, messages = _loud_fit(
            MultiObjectiveThresholdOptimizer(
                constraint="demographic_parity", n_thresholds=6, objectives=["accuracy"]
            ),
            y_true,
            y_prob,
            one_group,
        )
        assert not any("could not be computed on this data" in m for m in messages), messages
        # The accurate reason is still raised, and still recorded.
        assert any("no Pareto frontier was computed" in m for m in messages), messages
        assert opt.result_.optimization_details["pareto_frontier_computed"] is False

    # -- CONTROL ----------------------------------------------------------

    def test_control_a_computable_objective_is_ranked_and_measured(self):
        """The disclosure must not fire on healthy data, and the frontier must
        still carry a real measurement: the accuracy recorded for the selected
        point is recomputed here from its own thresholds."""
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = MultiObjectiveThresholdOptimizer(
            constraint="equalized_odds", n_thresholds=8, objectives=["accuracy", "f1_score"]
        )
        _quiet_fit(opt, y_true, y_prob, groups)

        details = opt.result_.optimization_details
        assert details["objectives_unmeasured"] == []
        assert details["pareto_frontier_ranked"] is True
        assert "NOT over the objectives that were requested" not in opt.result_.summary()

    def test_control_the_selected_point_carries_a_real_measurement(self):
        """Deliberately free of any disclosure assertion, so it stays green
        while the disclosure pins above are sabotaged: a fix that turned this
        class into a refuser would show up HERE."""
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = MultiObjectiveThresholdOptimizer(
            constraint="equalized_odds", n_thresholds=8, objectives=["accuracy", "f1_score"]
        )
        _quiet_fit(opt, y_true, y_prob, groups)

        thresholds = opt.result_.group_thresholds
        expected_pred = np.array(
            [1 if p >= thresholds[str(g)] else 0 for p, g in zip(y_prob, groups)]
        )
        expected_accuracy = float((expected_pred == y_true).mean())
        assert opt.result_.performance_metrics["accuracy"] == pytest.approx(expected_accuracy)
        assert math.isfinite(opt.result_.performance_metrics["f1_score"])


# --------------------------------------------- ThresholdOptimizer.fit/predict


class TestSingleThresholdOptimizer:
    def test_one_group_is_not_assessed(self):
        y_true, y_prob, _ = _two_groups_with_a_real_gap()
        opt, messages = _loud_fit(
            ThresholdOptimizer(constraint="demographic_parity", n_thresholds=20),
            y_true,
            y_prob,
            np.array(["a"] * len(y_true)),
        )
        assert opt.result_.is_feasible is None
        assert math.isnan(opt.result_.constraint_violations[0].violation)
        assert opt.result_.optimization_details["constraint_measured"] is False
        assert "Feasible: not assessed" in opt.result_.summary()
        assert any("could not be measured at ANY" in m for m in messages), messages

    def test_predict_refuses_a_row_with_no_score(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(ThresholdOptimizer(n_thresholds=20), y_true, y_prob, groups)
        broken = y_prob.copy()
        broken[7] = np.nan
        with pytest.raises(ValueError, match="no usable score"):
            opt.predict(broken, groups)

    # -- CONTROL ----------------------------------------------------------

    def test_control_a_real_gap_is_measured_and_predicted(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(
            ThresholdOptimizer(constraint="demographic_parity", n_thresholds=20),
            y_true,
            y_prob,
            groups,
        )
        assert isinstance(opt.result_.is_feasible, bool)
        assert opt.result_.optimization_details["constraint_measured"] is True

        pred = opt.predict(y_prob, groups)
        threshold = opt.result_.global_threshold
        # Independently: the demographic parity gap of the returned decisions.
        gap = abs(pred[groups == "a"].mean() - pred[groups == "b"].mean())
        assert opt.result_.constraint_violations[0].violation == pytest.approx(gap)
        assert float((pred == y_true).mean()) == pytest.approx(
            opt.result_.performance_metrics["accuracy"]
        )
        assert set(np.unique(pred)) <= {0, 1}
        assert pred.sum() == int((y_prob >= threshold).sum())


# ------------------------------------------ GroupThresholdOptimizer.fit/predict


class TestGroupThresholdOptimizer:
    def test_predict_refuses_a_group_it_never_fitted(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(GroupThresholdOptimizer(n_thresholds=6), y_true, y_prob, groups)
        unseen = np.where(np.arange(len(groups)) % 3 == 0, "c", groups)
        with pytest.raises(ValueError, match="No threshold was fitted for group"):
            opt.predict(y_prob, unseen)

    def test_unscored_rows_make_the_verdict_not_assessed(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        broken = y_prob.copy()
        broken[:10] = np.nan
        opt, messages = _loud_fit(GroupThresholdOptimizer(n_thresholds=6), y_true, broken, groups)
        assert opt.result_.is_feasible is None
        assert opt.result_.optimization_details["n_unscored_rows"] == 10
        assert any("entered the fairness measurement as REJECTED" in m for m in messages) or any(
            "no usable score" in m for m in messages
        ), messages
        assert "Feasible: not assessed" in opt.result_.summary()

    # -- CONTROL ----------------------------------------------------------

    def test_control_group_thresholds_are_applied_exactly(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(
            GroupThresholdOptimizer(constraint="demographic_parity", n_thresholds=6),
            y_true,
            y_prob,
            groups,
        )
        assert isinstance(opt.result_.is_feasible, bool)
        pred = opt.predict(y_prob, groups)
        thresholds = opt.result_.group_thresholds
        expected = np.array([1 if p >= thresholds[str(g)] else 0 for p, g in zip(y_prob, groups)])
        assert np.array_equal(pred, expected)
        gap = abs(pred[groups == "a"].mean() - pred[groups == "b"].mean())
        assert opt.result_.constraint_violations[0].violation == pytest.approx(gap)


# ----------------------------------------------- fit_predict, select_point, to_dict


class TestTheRemainingSurfaces:
    def test_fit_predict_equals_fit_then_predict(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            one = GroupThresholdOptimizer(n_thresholds=6).fit_predict(y_true, y_prob, groups)
            other = GroupThresholdOptimizer(n_thresholds=6)
            other.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            expected = other.predict(y_prob, groups)
        assert np.array_equal(one, expected)
        assert 0 < one.mean() < 1, "a fit_predict that accepts or rejects everyone proves nothing"

    def test_fit_predict_refuses_unscored_rows_rather_than_rejecting_them(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        broken = y_prob.copy()
        broken[3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="no usable score"):
                GroupThresholdOptimizer(n_thresholds=6).fit_predict(y_true, broken, groups)

    def test_select_point_moves_the_result_and_refuses_an_index_it_has_not_got(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(MultiObjectiveThresholdOptimizer(n_thresholds=8), y_true, y_prob, groups)
        assert opt.select_point(0) is opt
        assert opt.result_ is opt.pareto_frontier_[0]
        with pytest.raises(ValueError, match="out of range"):
            opt.select_point(len(opt.pareto_frontier_))

    def test_to_dict_carries_the_third_state_not_a_false(self):
        y_true, y_prob, _ = _two_groups_with_a_real_gap()
        opt = _quiet_fit(
            ThresholdOptimizer(n_thresholds=20), y_true, y_prob, np.array(["a"] * len(y_true))
        )
        data = opt.result_.to_dict()
        assert data["is_feasible"] is None, "None is could-not-check; False would be a verdict"
        assert data["constraint_violations"][0]["is_satisfied"] is None
        assert math.isnan(data["constraint_violations"][0]["violation"])

    def test_control_to_dict_carries_a_real_verdict_when_there_is_one(self):
        y_true, y_prob, groups = _two_groups_with_a_real_gap()
        opt = _quiet_fit(ThresholdOptimizer(n_thresholds=20), y_true, y_prob, groups)
        data = opt.result_.to_dict()
        assert isinstance(data["is_feasible"], bool)
        assert math.isfinite(data["constraint_violations"][0]["violation"])
        assert data["global_threshold"] == opt.result_.global_threshold
