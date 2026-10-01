"""Surface grading batch g007: reductions.py must not answer with a value
nobody measured.

Four execution-proven defects, all reproduced through the public API of
``vfairness.in_processing.constraints.reductions`` before the fix:

1.  ``ThresholdOptimizer.predict`` decided rows that carry no score. On a
    fitted optimizer (thresholds a=0.5544, b=0.4456)::

        predict([0.9, nan, 0.1, inf, -inf], ['a','a','a','b','b']) -> [1 0 0 1 0]

    The NaN row came back 0, byte-identical to the measured rejection at 0.1
    beside it; ``-inf`` fabricated a second rejection and ``+inf`` fabricated
    an ACCEPTANCE. No warning, no error. This is the BGL-G05 defect, fixed
    once in ``post_processing.threshold_optimization.optimizer`` and never
    swept out of this second copy of the threshold rule.

2.  ``ThresholdOptimizer.fit`` folded those same rows into the fairness
    measurement as rejections and said nothing. On an all-NaN score column::

        final_violation_ = 0.0   constraint_satisfied_ = True   warnings []

    A perfect-parity certificate over a column carrying no information at
    all: rejecting everybody is trivially equal.

3.  ``ThresholdOptimizer.fit`` on a constraint that could NOT be evaluated
    (one group, or a group with no positive labels under equalized odds)
    reported ``constraint_satisfied_ = False``, which reads as a measured
    breach, under a warning that stated the comparison "final violation nan >
    tolerance 0.05" that never happened. There was no third state on the
    object at all.

4.  ``ReductionResult`` and ``EnsembleClassifier`` (both exported from
    ``vfairness.in_processing.constraints``) answered from an ensemble in
    which nothing contributed::

        EnsembleClassifier([], []).predict(X)                   -> [0 0 0 0 0 0]
        EnsembleClassifier([c, c], [0., 0.]).predict_proba(X)   -> [0. 0. ...]

    a confident rejection of every row, and a probability of exactly 0.0. A
    NaN weight produced the same outcome silently (``nan > 1e-8`` is False),
    and a classifiers/weights length mismatch was zipped away so an ensemble
    of three predicted from one.

Every block has an OVER-CORRECTION CONTROL that asserts the numbers the
working path still returns, computed in this file from first principles, so
reinstating any one defect turns exactly one pin red without a control going
green by vacuity.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vfairness.in_processing.constraints.base import (
    DemographicParityConstraint,
    OptimizationResult,
)
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
    GridSearch,
    ReductionResult,
    ThresholdOptimizer,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _scored_two_group(n=120, seed=2):
    """Scores that genuinely separate, split over two groups."""
    rng = np.random.default_rng(seed)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)
    return y_prob, y_true, groups


def _fitted_optimizer(grid_size=10):
    y_prob, y_true, groups = _scored_two_group()
    optimizer = ThresholdOptimizer("demographic_parity", grid_size=grid_size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.fit(y_prob, y_true, sensitive_attr=groups)
    return optimizer, (y_prob, y_true, groups), [str(w.message) for w in caught]


def _selection_gap(y_prob, groups, thresholds):
    """The demographic-parity gap, worked out here rather than read back."""
    rates = []
    for group in np.unique(groups):
        mask = groups == group
        rates.append(float(np.mean(y_prob[mask] >= thresholds[group])))
    return max(rates) - min(rates)


class _ProbaStub:
    """A member that reports a fixed probability and a fixed label."""

    def __init__(self, p):
        self.p = float(p)

    def predict(self, X):
        return np.full(len(X), int(self.p >= 0.5), dtype=int)

    def predict_proba(self, X):
        col = np.full(len(X), self.p)
        return np.column_stack([1.0 - col, col])


class _HardStub:
    """A member with NO predict_proba, e.g. a LinearSVC."""

    def __init__(self, label):
        self.label = int(label)

    def predict(self, X):
        return np.full(len(X), self.label, dtype=int)


def _result(classifiers, weights):
    return ReductionResult(
        classifiers=classifiers,
        weights=np.asarray(weights, dtype=float),
        optimization_result=OptimizationResult(
            converged=False, n_iterations=1, final_violation=0.1, best_gap=0.1
        ),
        final_violation=0.1,
        accuracy=0.5,
    )


_X6 = np.zeros((6, 2))


# ---------------------------------------------------------------------------
# 1. ThresholdOptimizer.predict refuses a row it cannot score
# ---------------------------------------------------------------------------


class TestPredictRefusesAnUnscoreableRow:
    def test_nan_score_is_refused_not_rejected(self):
        """BEFORE: predict([0.9, nan, 0.1], ['a','a','a']) -> [1 0 0]. The
        middle row was byte-identical to the measured rejection beside it."""
        optimizer, _, _ = _fitted_optimizer()
        with pytest.raises(ValueError, match="no usable score"):
            optimizer.predict(np.array([0.9, np.nan, 0.1]), np.array(["a", "a", "a"]))

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_infinite_score_is_refused_too(self, bad):
        """+inf lands above every threshold and fabricated an ACCEPT; -inf
        lands below every threshold and fabricated a REJECT."""
        optimizer, _, _ = _fitted_optimizer()
        with pytest.raises(ValueError, match="no usable score"):
            optimizer.predict(np.array([0.9, bad, 0.1]), np.array(["a", "a", "a"]))

    def test_the_refusal_names_the_row(self):
        optimizer, _, _ = _fitted_optimizer()
        with pytest.raises(ValueError) as excinfo:
            optimizer.predict(np.array([0.9, 0.8, np.nan, 0.1]), np.array(["a", "a", "b", "b"]))
        message = str(excinfo.value)
        assert "2 of 4" not in message  # exactly one row is unscored
        assert "1 of 4" in message and "index [2]" in message

    def test_control_finite_scores_are_still_decided(self):
        """OVER-CORRECTION CONTROL. The labels are derived here from the
        fitted thresholds, not read back from predict()."""
        optimizer, (y_prob, _, groups), _ = _fitted_optimizer()
        thresholds = optimizer.get_thresholds()
        # The fitted search really did move off the 0.5 seed, so this control
        # cannot pass by accident on an optimizer that optimised nothing.
        assert thresholds["a"] != 0.5 and thresholds["b"] != 0.5
        assert thresholds["a"] != thresholds["b"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predicted = optimizer.predict(y_prob, groups)
        assert not [str(w.message) for w in caught]

        expected = np.where(
            groups == "a", y_prob >= thresholds["a"], y_prob >= thresholds["b"]
        ).astype(int)
        np.testing.assert_array_equal(predicted, expected)
        # Both verdicts are actually present, so an all-0 or all-1 return fails.
        assert set(np.unique(predicted)) == {0, 1}


# ---------------------------------------------------------------------------
# 2. ThresholdOptimizer.fit discloses rows it folded in as fabricated rejects
# ---------------------------------------------------------------------------


class TestFitDoesNotCertifyOverFabricatedRejections:
    def test_all_nan_scores_is_not_a_perfect_parity_pass(self):
        """BEFORE: final_violation_=0.0, constraint_satisfied_=True, zero
        warnings. Rejecting everybody is trivially equal, so a score column
        with no information in it certified perfect parity."""
        n = 60
        groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = ThresholdOptimizer("demographic_parity", grid_size=10)
            optimizer.fit(np.full(n, np.nan), np.array([0, 1] * (n // 2)), sensitive_attr=groups)
        messages = [str(w.message) for w in caught]

        assert optimizer.constraint_satisfied_ is None, (
            "a constraint measured over fabricated rejections was reported as "
            f"{optimizer.constraint_satisfied_!r}"
        )
        assert optimizer.insufficient_data_ is True
        assert optimizer.n_unscored_rows_ == n
        assert any("no usable score" in m and "REJECTED" in m for m in messages), messages

    def test_a_minority_of_unscored_rows_is_disclosed_as_well(self):
        """The dangerous case is the quiet one: 5 unscored rows in 120 leave
        the violation looking entirely ordinary."""
        y_prob, y_true, groups = _scored_two_group()
        y_prob = y_prob.copy()
        y_prob[:5] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = ThresholdOptimizer("demographic_parity", grid_size=10)
            optimizer.fit(y_prob, y_true, sensitive_attr=groups)
        messages = [str(w.message) for w in caught]

        assert optimizer.n_unscored_rows_ == 5
        assert optimizer.insufficient_data_ is True
        assert optimizer.constraint_satisfied_ is None
        assert any("5 of 120" in m for m in messages), messages
        # The MAGNITUDE is kept: withdrawing a mostly-real measurement would
        # be the over-correction. Only the verdict is withheld.
        assert np.isfinite(optimizer.final_violation_)

    def test_control_clean_scores_get_a_real_verdict_and_no_warning(self):
        """OVER-CORRECTION CONTROL, with the violation recomputed here."""
        optimizer, (y_prob, _, groups), messages = _fitted_optimizer()

        assert optimizer.n_unscored_rows_ == 0
        assert optimizer.insufficient_data_ is False
        assert isinstance(optimizer.constraint_satisfied_, bool)
        assert not messages, messages

        expected_gap = _selection_gap(y_prob, groups, optimizer.get_thresholds())
        assert optimizer.final_violation_ == pytest.approx(expected_gap, abs=1e-12)
        assert optimizer.constraint_satisfied_ is bool(expected_gap <= 0.05)


# ---------------------------------------------------------------------------
# 3. ThresholdOptimizer.fit: could-not-check is not a breach
# ---------------------------------------------------------------------------


class TestAnUnevaluableConstraintIsNotABreach:
    def test_one_group_is_could_not_check(self):
        """BEFORE: constraint_satisfied_=False (a measured breach, to any
        reader) under a warning asserting "final violation nan > tolerance
        0.05", a comparison that never happened."""
        rng = np.random.default_rng(1)
        y_prob = rng.random(200)
        y_true = (rng.random(200) < y_prob).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = ThresholdOptimizer("demographic_parity", grid_size=20)
            optimizer.fit(y_prob, y_true, sensitive_attr=np.array(["A"] * 200))
        messages = [str(w.message) for w in caught]

        assert optimizer.constraint_satisfied_ is None
        assert optimizer.insufficient_data_ is True
        assert np.isnan(optimizer.final_violation_)
        assert any("could NOT be evaluated" in m for m in messages), messages
        # The thresholds handed back are the untouched seed, and the warning
        # says so rather than letting 0.5 pass for an optimised 0.5.
        assert optimizer.get_thresholds() == {"A": 0.5}
        assert any("NOT selected for fairness" in m for m in messages), messages

    def test_equalized_odds_with_no_positives_in_one_group_is_could_not_check(self):
        """The sibling shape: every rate is finite, but one arm of the
        constraint has an empty conditioning set."""
        n = 240
        groups = np.array(["A"] * 160 + ["B"] * 80)
        rng = np.random.default_rng(5)
        y_prob = rng.random(n)
        y_true = np.where(groups == "A", (rng.random(n) < y_prob).astype(int), 0)
        assert y_true[groups == "B"].sum() == 0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = ThresholdOptimizer("equalized_odds", grid_size=20)
            optimizer.fit(y_prob, y_true, sensitive_attr=groups)
        messages = [str(w.message) for w in caught]

        assert optimizer.constraint_satisfied_ is None
        assert optimizer.insufficient_data_ is True
        assert any("could NOT be evaluated" in m for m in messages), messages

    @pytest.mark.parametrize("tolerance, satisfied", [(0.9, True), (0.0, False)])
    def test_control_both_measured_verdicts_stay_reachable(self, tolerance, satisfied):
        """OVER-CORRECTION CONTROL. A refusal that swallowed the real FAIL
        would be the worse defect, so both verdicts are pinned."""
        y_prob, y_true, groups = _scored_two_group()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = ThresholdOptimizer(
                DemographicParityConstraint(tolerance=tolerance), grid_size=10
            )
            optimizer.fit(y_prob, y_true, sensitive_attr=groups)
        messages = [str(w.message) for w in caught]

        assert optimizer.constraint_satisfied_ is satisfied
        assert optimizer.insufficient_data_ is False
        assert np.isfinite(optimizer.final_violation_)
        assert optimizer.final_violation_ == pytest.approx(
            _selection_gap(y_prob, groups, optimizer.get_thresholds()), abs=1e-12
        )
        assert not any("could NOT be evaluated" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 4. An ensemble in which nothing contributes must not answer
# ---------------------------------------------------------------------------


def _silent_ensembles():
    return [
        ("EnsembleClassifier.predict", lambda c, w: EnsembleClassifier(c, w).predict(_X6)),
        (
            "EnsembleClassifier.predict_proba",
            lambda c, w: EnsembleClassifier(c, w).predict_proba(_X6),
        ),
        ("ReductionResult.predict", lambda c, w: _result(c, w).predict(_X6)),
        ("ReductionResult.predict_proba", lambda c, w: _result(c, w).predict_proba(_X6)),
    ]


@pytest.mark.parametrize("name, call", _silent_ensembles(), ids=[n for n, _ in _silent_ensembles()])
class TestAnEnsembleThatCombinesNothing:
    def test_all_negligible_weights_are_refused(self, name, call):
        """BEFORE: [0 0 0 0 0 0] and [0. 0. 0. 0. 0. 0.], a confident
        rejection of every row and a probability of exactly 0.0."""
        with pytest.raises(ValueError, match="no ensemble member carries a weight"):
            call([_ProbaStub(0.9), _ProbaStub(0.9)], np.array([0.0, 0.0]))

    def test_an_empty_ensemble_is_refused(self, name, call):
        with pytest.raises(ValueError, match="no ensemble member carries a weight"):
            call([], np.array([]))

    def test_a_nan_weight_is_refused_not_treated_as_zero(self, name, call):
        """``nan > 1e-8`` is False, so a weight nobody could evaluate dropped
        its member from the vote in silence."""
        with pytest.raises(ValueError, match="NaN"):
            call([_ProbaStub(0.9), _ProbaStub(0.1)], np.array([1.0, np.nan]))

    def test_a_length_mismatch_is_refused(self, name, call):
        """zip() stops at the shorter one, so three members voted as one."""
        with pytest.raises(ValueError, match="but 1 weight"):
            call([_ProbaStub(0.9), _ProbaStub(0.1), _ProbaStub(0.5)], np.array([1.0]))


class TestControlAnEnsembleThatCombinesSomethingStillAnswers:
    """OVER-CORRECTION CONTROL. Every expected value is the weighted average
    worked out here, never read back from the code."""

    @pytest.mark.parametrize(
        "weights, expected_proba, expected_label",
        [
            ([0.25, 0.75], 0.25 * 0.2 + 0.75 * 0.8, 1),  # 0.65 -> 1
            ([0.75, 0.25], 0.75 * 0.2 + 0.25 * 0.8, 0),  # 0.35 -> 0
        ],
    )
    def test_reduction_result(self, weights, expected_proba, expected_label):
        members = [_ProbaStub(0.2), _ProbaStub(0.8)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = _result(members, weights).predict_proba(_X6)
            label = _result(members, weights).predict(_X6)
        assert not [str(w.message) for w in caught]
        np.testing.assert_allclose(proba, np.full(6, expected_proba))
        # predict() combines hard LABELS (0 and 1 here), not the probabilities.
        np.testing.assert_array_equal(label, np.full(6, expected_label, dtype=int))

    def test_ensemble_classifier(self):
        members = [_ProbaStub(0.2), _ProbaStub(0.8)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ensemble = EnsembleClassifier(members, np.array([0.25, 0.75]))
            proba = ensemble.predict_proba(_X6)
            label = ensemble.predict(_X6)
        assert not [str(w.message) for w in caught]
        np.testing.assert_allclose(proba, np.full(6, 0.65))
        np.testing.assert_array_equal(label, np.full(6, 1, dtype=int))

    def test_a_negligible_member_beside_a_real_one_is_still_allowed(self):
        """The guard is about an ensemble with NOTHING in it, not about one
        member being small: a 1e-12 weight beside a 1.0 weight still answers."""
        ensemble = EnsembleClassifier([_ProbaStub(0.0), _ProbaStub(0.9)], np.array([1e-12, 1.0]))
        np.testing.assert_allclose(ensemble.predict_proba(_X6), np.full(6, 0.9))


# ---------------------------------------------------------------------------
# 5. predict_proba says when it is returning votes, not probabilities
# ---------------------------------------------------------------------------


class TestVoteFractionsAreDisclosed:
    @pytest.mark.parametrize(
        "build",
        [
            lambda m, w: EnsembleClassifier(m, w).predict_proba(_X6),
            lambda m, w: _result(m, w).predict_proba(_X6),
        ],
        ids=["EnsembleClassifier", "ReductionResult"],
    )
    def test_a_member_without_predict_proba_warns(self, build):
        """BEFORE, through the public API with LinearSVC as base estimator:
        predict_proba returned np.unique(...) == [0., 1.] with zero warnings,
        i.e. hard labels handed back through a method named predict_proba."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = build([_HardStub(1)], np.array([1.0]))
        messages = [str(w.message) for w in caught]
        assert any("VOTE FRACTIONS" in m for m in messages), messages
        # The VALUE is kept: a vote fraction is a real quantity, and blanking
        # it would throw away a usable signal.
        np.testing.assert_array_equal(proba, np.ones(6))

    def test_control_probabilistic_members_do_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = EnsembleClassifier(
                [_ProbaStub(0.3), _ProbaStub(0.7)], np.array([0.5, 0.5])
            ).predict_proba(_X6)
        assert not [str(w.message) for w in caught]
        np.testing.assert_allclose(proba, np.full(6, 0.5))

    def test_control_a_negligible_hard_member_does_not_warn(self):
        """It only matters for members that actually contribute."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EnsembleClassifier([_HardStub(1), _ProbaStub(0.3)], np.array([0.0, 1.0])).predict_proba(
                _X6
            )
        assert not [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# 6. The fitted paths still work end to end (the guards must not break them)
# ---------------------------------------------------------------------------


def _biased_two_group(n=300, seed=7):
    rng = np.random.default_rng(seed)
    in_a = rng.random(n) < 0.5
    X = np.column_stack([rng.normal(size=n) + 2.5 * in_a, rng.normal(size=n)])
    y = ((X[:, 0] + rng.normal(scale=0.3, size=n)) > 1).astype(int)
    return X, y, np.where(in_a, "A", "B")


class TestControlTheFittedPathsStillMeasure:
    def test_exponentiated_gradient_round_trip(self):
        X, y, s = _biased_two_group()
        estimator = ExponentiatedGradient(
            LogisticRegression(max_iter=300),
            DemographicParityConstraint(tolerance=0.05),
            max_iterations=5,
        )
        result = estimator.fit(X, y, sensitive_attr=s)

        assert np.isfinite(result.final_violation)
        assert result.fairness_metrics["insufficient_data"] is False
        assert estimator.get_result() is result

        predicted = estimator.predict(X)
        assert len(predicted) == len(y)
        measured = DemographicParityConstraint(tolerance=0.05).compute_violation(predicted, y, s)
        assert result.final_violation == pytest.approx(measured.overall_violation, abs=1e-12)

        # to_dict carries the same numbers, with no classifier objects in it.
        payload = result.to_dict()
        assert payload["final_violation"] == result.final_violation
        assert payload["n_classifiers"] == len(result.classifiers)
        assert "classifiers" not in payload

        proba = estimator.predict_proba(X)
        assert len(proba) == len(y)
        assert np.all(np.isfinite(proba))

    def test_grid_search_round_trip(self):
        X, y, s = _biased_two_group()
        search = GridSearch(
            LogisticRegression(max_iter=300),
            DemographicParityConstraint(tolerance=0.9),
            n_lambda_values=4,
        )
        result = search.fit(X, y, sensitive_attr=s)
        assert np.isfinite(result.final_violation)
        assert result.fairness_metrics["insufficient_data"] is False
        assert len(search.predict(X)) == len(y)


# ---------------------------------------------------------------------------
# 7. The three states survive the fitted wrappers and the dict conversion
# ---------------------------------------------------------------------------


class TestExponentiatedGradientOnOneGroup:
    """A single sensitive level: there is no pair of groups to compare, so no
    demographic-parity violation exists to be satisfied."""

    @staticmethod
    def _fit_one_group(max_iterations=4):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 3))
        y = (X[:, 0] + rng.normal(scale=0.5, size=300) > 0.5).astype(int)
        s = np.array(["A"] * 300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = ExponentiatedGradient(
                LogisticRegression(max_iter=300),
                DemographicParityConstraint(tolerance=0.05),
                max_iterations=max_iterations,
            )
            result = estimator.fit(X, y, sensitive_attr=s)
        return estimator, result, X, y, [str(w.message) for w in caught]

    def test_it_is_could_not_check_not_a_pass(self):
        _, result, _, _, messages = self._fit_one_group()
        assert np.isnan(result.final_violation)
        assert result.fairness_metrics["insufficient_data"] is True
        assert result.fairness_metrics["constraint_satisfied"] is False
        assert result.fairness_metrics["n_iterations_unmeasurable"] == 4
        assert result.optimization_result.converged is False
        assert any(
            "not one of 4 iterates" in m and "NOT selected for fairness" in m for m in messages
        ), messages

    def test_to_dict_carries_the_third_state_out(self):
        """A consumer that only ever sees the dict (JSON, a report payload)
        must still be able to tell this from a measured 0.0."""
        _, result, _, _, _ = self._fit_one_group()
        payload = result.to_dict()
        assert np.isnan(payload["final_violation"])
        assert payload["fairness_metrics"]["insufficient_data"] is True
        assert payload["fairness_metrics"]["constraint_satisfied"] is False

    def test_it_is_still_a_fail_state_and_not_a_crash(self):
        estimator, result, X, y, _ = self._fit_one_group()
        assert len(estimator.predict(X)) == len(y)
        assert estimator.get_result() is result


class TestPredictUsesExactlyTheModelWhoseMetricsWereReported:
    """One-hot weights, so the reported violation describes the predictions
    that come back. Each expectation is taken from the recorded best index,
    never from the predict() call being checked."""

    def test_exponentiated_gradient(self):
        X, y, s = _biased_two_group()
        result = ExponentiatedGradient(
            LogisticRegression(max_iter=300),
            DemographicParityConstraint(tolerance=0.05),
            max_iterations=5,
            best_gap_iteration=True,
        ).fit(X, y, sensitive_attr=s)

        best = result.optimization_result.metadata["best_iteration"]
        assert result.weights[best] == 1.0
        assert float(np.sum(result.weights)) == 1.0
        np.testing.assert_array_equal(result.predict(X), result.classifiers[best].predict(X))

    def test_grid_search(self):
        X, y, s = _biased_two_group()
        search = GridSearch(
            LogisticRegression(max_iter=300),
            DemographicParityConstraint(tolerance=0.9),
            n_lambda_values=4,
        )
        result = search.fit(X, y, sensitive_attr=s)

        best = result.optimization_result.metadata["best_index"]
        assert result.weights[best] == 1.0
        assert float(np.sum(result.weights)) == 1.0
        np.testing.assert_array_equal(search.predict(X), result.classifiers[best].predict(X))

    @pytest.mark.parametrize(
        "unfitted, method",
        [
            (ExponentiatedGradient, "predict"),
            (ExponentiatedGradient, "predict_proba"),
            (GridSearch, "predict"),
        ],
    )
    def test_an_unfitted_estimator_refuses_rather_than_answering(self, unfitted, method):
        estimator = unfitted(LogisticRegression(), "demographic_parity")
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(estimator, method)(np.zeros((4, 2)))


def test_exponentiated_gradient_predict_proba_discloses_hard_labels():
    """Through the real public API, with a base estimator that has no
    predict_proba. BEFORE: np.unique(...) was [0., 1.] with zero warnings, so
    hard labels came back through a method named predict_proba."""
    svm = pytest.importorskip("sklearn.svm")
    X, y, s = _biased_two_group()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = ExponentiatedGradient(
            svm.LinearSVC(max_iter=2000),
            DemographicParityConstraint(tolerance=0.05),
            max_iterations=3,
        )
        estimator.fit(X, y, sensitive_attr=s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = estimator.predict_proba(X)
    messages = [str(w.message) for w in caught]
    assert any("VOTE FRACTIONS" in m for m in messages), messages
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_control_predict_proba_of_a_probabilistic_fit_does_not_warn():
    """OVER-CORRECTION CONTROL for the disclosure above."""
    X, y, s = _biased_two_group()
    estimator = ExponentiatedGradient(
        LogisticRegression(max_iter=300),
        DemographicParityConstraint(tolerance=0.05),
        max_iterations=3,
    )
    estimator.fit(X, y, sensitive_attr=s)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = estimator.predict_proba(X)
    assert not [str(w.message) for w in caught]
    assert 0.0 < float(np.min(proba)) and float(np.max(proba)) < 1.0
