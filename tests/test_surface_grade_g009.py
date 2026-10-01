"""Batch g009: the sklearn fairness wrappers must not report what they never measured.

THE DEFECTS, each reproduced by execution before it was fixed, on
``src/vfairness/in_processing/wrappers/sklearn_wrappers.py``.

1. ``FairClassifier.predict_proba`` MANUFACTURED CERTAINTY. When the fitted
   inner model had no ``predict_proba`` it returned
   ``column_stack([1 - predict(X), predict(X)])``, so every hard label became a
   confidence of exactly 1.0. Measured on 60 rows, ``method='grid_search'`` with
   LogisticRegression: ``np.unique(proba)`` was ``[0, 1]``, dtype int64, two
   distinct scores over sixty rows, no warning, and
   ``roc_auc_score(y, proba[:, 1])`` read 0.7833 (the hard-label accuracy
   wearing the name of an AUC). ``method='threshold'`` with a LinearSVC base did
   the same. The shape was not stable either: ``method='reductions'`` returned
   ``(n,)`` where the others returned ``(n, 2)``, so ``proba[:, 1]`` raised
   IndexError on the DEFAULT method.

2. ``FairClassifier.score(metric='fairness'|'combined')`` RETURNED THE ACCURACY
   when ``sensitive_attr`` was None. Measured on the same 60 rows (a real 0.10
   demographic-parity gap): with the attribute it answered -0.1000, without it
   0.7833. Every measured fairness score on that scale is <= 0, so the
   unmeasured call outranked every measured model. ``metric='combined'`` is the
   worse half: ``accuracy`` is arithmetically ``accuracy - 0.0``, a violation
   nobody computed.

3. ``FairRegressor.score`` SCORED A MODEL THE CLASS DOES NOT DEPLOY. It
   delegated to the base regressor, so the mean-parity offsets were never in the
   number. Measured on 60 rows, two groups, offsets +/-0.0968: ``reg.score(X, y)``
   returned 0.683537 (the base model's R^2) while the R^2 of what
   ``predict_with_sensitive_attr`` returns was 0.642071. ``predict()`` has warned
   about exactly this since the offsets were added; ``score`` was silent.

4. ``FairRegressor.get_params`` DROPPED ``on_unseen_group``, so a rebuild from
   the params silently replaced a caller's 'nan' policy with 'raise'; and it
   ignored ``deep``, so no ``base_estimator__*`` key existed to round-trip.

5. ``set_params`` on BOTH wrappers accepted any name in silence
   (``elif hasattr(self, key)``). Measured:
   ``reg.set_params(tolerence=99, base_estimator__fit_intercept=False)``
   returned self, changed nothing, warned about nothing. A sweep over a name the
   object does not have fits, scores, compares and ranks N identical models.

Every refusal pin below is paired with an OVER-CORRECTION CONTROL that asserts a
real value is still measured, computed independently in the test rather than
copied from the code.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.in_processing.wrappers.sklearn_wrappers import (
    FairClassifier,
    FairRegressor,
    make_fair_classifier,
    make_fair_regressor,
)

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

N_PER_GROUP = 40


# ---------------------------------------------------------------------------
# Fixtures. Groups are 40 rows each, above GroupManager's min_group_size=30, so
# no refusal below can come from a group being too small to look at. Labels are
# dtype=object: a numpy "<U1" array would truncate a longer label silently.
# ---------------------------------------------------------------------------


def _classification_data():
    """Three balanced groups, both labels in every one, and a disparity that
    SURVIVES the mitigation: A is pushed toward the positive side and C away
    from it, so every fit path still has a nonzero demographic-parity spread to
    report. A fixture the mitigation drives to exactly 0.0 would let a pin on
    "the reported violation equals the real spread" pass against a hardcoded
    zero."""
    rng = np.random.default_rng(11)
    n = 3 * N_PER_GROUP
    groups = np.array(["A"] * N_PER_GROUP + ["B"] * N_PER_GROUP + ["C"] * N_PER_GROUP, dtype=object)
    X = rng.normal(size=(n, 3))
    y = np.zeros(n, dtype=int)
    for i in range(3):
        y[i * N_PER_GROUP : i * N_PER_GROUP + N_PER_GROUP // 2] = 1
    X[:, 0] += (groups == "A").astype(float) * 2.0 - (groups == "C").astype(float) * 2.0
    X[:, 1] += y * 1.5
    return X, y, groups


def _regression_data():
    """Two groups whose target means differ by 3.0, so offsets are nonzero."""
    rng = np.random.default_rng(3)
    n = 2 * N_PER_GROUP
    groups = np.array(["a"] * N_PER_GROUP + ["b"] * N_PER_GROUP, dtype=object)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 2.0 + (groups == "a").astype(float) * 3.0 + rng.normal(scale=0.2, size=n)
    return X, y, groups


def _fitted(method, base=None, **kwargs):
    X, y, groups = _classification_data()
    clf = FairClassifier(
        base_estimator=base if base is not None else LogisticRegression(max_iter=300),
        fairness_constraint="demographic_parity",
        tolerance=0.05,
        method=method,
        max_iterations=5,
        random_state=0,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y, sensitive_attr=groups)
    return clf, X, y, groups


def _demographic_parity_spread(y_pred, groups):
    """max-min selection rate, computed here so the pin never copies the code."""
    rates = [float(np.mean(y_pred[groups == g])) for g in np.unique(groups)]
    return max(rates) - min(rates)


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot


# ===========================================================================
# 1. predict_proba
# ===========================================================================


class TestPredictProbaNeverInventsConfidence:
    def test_a_base_estimator_with_no_probability_yields_nan_not_certainty(self):
        clf, X, _, _ = _fitted("grid_search")
        assert not hasattr(clf._inner_model, "predict_proba"), (
            "fixture no longer reaches the branch"
        )

        with pytest.warns(UserWarning, match="NO probability was estimated"):
            proba = clf.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.isnan(proba).all(), "every entry, not just the positive column"
        # the old value: hard labels as a certainty of 1.0
        assert not np.array_equal(np.unique(proba[~np.isnan(proba)]), np.array([0.0, 1.0]))

    def test_a_decision_function_only_base_also_refuses(self):
        """The same branch, reached the other way: LinearSVC has no
        predict_proba, only a decision function, and a sigmoid of an
        UNCALIBRATED decision function is not a probability either."""
        clf, X, _, _ = _fitted("threshold", base=LinearSVC(max_iter=5000))
        assert not hasattr(clf._inner_model, "predict_proba")
        with pytest.warns(UserWarning, match="NO probability was estimated"):
            proba = clf.predict_proba(X)
        assert np.isnan(proba).all()

    def test_control_a_real_probability_is_still_returned_and_unchanged(self):
        """OVER-CORRECTION CONTROL: the probability the ensemble really
        computed comes back untouched, and it is NOT two distinct values."""
        clf, X, y, _ = _fitted("reductions")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = clf.predict_proba(X)
        assert not [w for w in caught if "probability was estimated" in str(w.message)]

        inner = np.asarray(clf._inner_model.predict_proba(X), dtype=float)
        assert inner.ndim == 1, "fixture no longer reaches the 1-D widening"
        assert np.allclose(proba[:, 1], inner), "the measured value must survive verbatim"
        assert len(np.unique(proba[:, 1])) > 10, "a real score, not a relabelled hard label"
        assert float(np.min(proba)) >= 0.0 and float(np.max(proba)) <= 1.0
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("method", ["reductions", "threshold", "grid_search"])
    def test_the_shape_is_the_same_on_every_method(self, method):
        """(n, 2) everywhere: proba[:, 1] used to raise IndexError on the
        DEFAULT method while working on the other two."""
        clf, X, _, _ = _fitted(method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = clf.predict_proba(X)
            column = clf.predict_proba(X)[:, 1]
        assert proba.shape == (X.shape[0], 2)
        assert column.shape == (X.shape[0],)


# ===========================================================================
# 2. FairClassifier.score
# ===========================================================================


class TestScoreWithoutTheSensitiveAttribute:
    @pytest.mark.parametrize("metric", ["fairness", "combined"])
    def test_no_sensitive_attr_is_no_fairness_score(self, metric):
        clf, X, y, _ = _fitted("reductions")
        with pytest.warns(UserWarning, match="needs sensitive_attr"):
            score = clf.score(X, y, metric=metric)
        assert math.isnan(score), "NaN, never the accuracy dressed as a fairness score"

    def test_the_old_answer_was_the_accuracy_and_outranked_every_measured_model(self):
        """The defect in one assertion: the unmeasured call must not come back
        as a number that beats the measured one on the same scale."""
        clf, X, y, groups = _fitted("reductions")
        measured = clf.score(X, y, sensitive_attr=groups, metric="fairness")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unmeasured = clf.score(X, y, metric="fairness")
        accuracy = clf.score(X, y, metric="accuracy")
        assert measured <= 0.0, "a measured fairness score is -violation"
        assert accuracy > 0.0
        assert not (unmeasured > measured), "an unmeasured model must not win the ranking"
        assert math.isnan(unmeasured)

    def test_control_accuracy_without_the_attribute_is_still_measured(self):
        """OVER-CORRECTION CONTROL: metric='accuracy' never needed the
        sensitive attribute and must be unaffected."""
        clf, X, y, _ = _fitted("reductions")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            accuracy = clf.score(X, y, metric="accuracy")
        assert not [w for w in caught if "needs sensitive_attr" in str(w.message)]
        expected = float(np.mean(clf.predict(X) == y))
        assert accuracy == pytest.approx(expected, abs=1e-12)

    def test_control_the_measured_fairness_score_is_the_real_disparity(self):
        """OVER-CORRECTION CONTROL: the number is checked against a
        demographic-parity spread computed here, not copied from the code."""
        clf, X, y, groups = _fitted("reductions")
        y_pred = clf.predict_with_sensitive_attr(X, groups)
        expected = _demographic_parity_spread(y_pred, groups)
        assert expected > 0.02, "fixture no longer carries a findable disparity"

        fairness = clf.score(X, y, sensitive_attr=groups, metric="fairness")
        combined = clf.score(X, y, sensitive_attr=groups, metric="combined")
        accuracy = clf.score(X, y, sensitive_attr=groups, metric="accuracy")
        assert fairness == pytest.approx(-expected, abs=1e-12)
        assert combined == pytest.approx(accuracy - expected, abs=1e-12)


# ===========================================================================
# 3. FairClassifier.fit / FairClassifierResult.to_dict
# ===========================================================================


class TestFitReportsAMeasurement:
    @pytest.mark.parametrize("method", ["reductions", "threshold", "grid_search"])
    def test_the_reported_violation_is_the_disparity_of_its_own_predictions(self, method):
        """CONTROL for the three fit paths: on data where every rate is
        defined, the violation each path reports equals the demographic-parity
        spread of the predictions that path makes, recomputed here."""
        clf, X, y, groups = _fitted(method)
        result = clf.fairness_result_
        assert result is not None
        expected = _demographic_parity_spread(clf.predict_with_sensitive_attr(X, groups), groups)
        assert result.fairness_violation == pytest.approx(expected, abs=1e-12)
        assert result.constraint_satisfied is (expected <= 0.05)
        assert result.fairness_metrics["insufficient_data"] is False
        assert result.accuracy == pytest.approx(
            float(np.mean(clf.predict_with_sensitive_attr(X, groups) == y)), abs=1e-12
        )

    @pytest.mark.parametrize("method", ["reductions", "threshold", "grid_search"])
    def test_a_constraint_with_no_second_group_is_not_a_measured_zero(self, method):
        """REFUSAL side of the same three fit paths, held here as well as in
        tests/test_readiness4_constraints.py so this file stands on its own. A
        single group has no peer to be a disparity FROM, so every per-group
        deviation is 0.0 by arithmetic vacuity and the spread is max-min over
        one number."""
        X, y, groups = _classification_data()
        one_group = np.array(["A"] * len(groups), dtype=object)
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=300),
            fairness_constraint="demographic_parity",
            tolerance=0.05,
            method=method,
            max_iterations=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=one_group)
        result = clf.fairness_result_
        assert result is not None
        assert result.fairness_violation is None, "None, not 0.0 and not NaN"
        assert result.constraint_satisfied is None, "None, not the fail-closed False"
        assert result.fairness_metrics["insufficient_data"] is True
        # the accuracy really was measured and stays a number
        assert result.accuracy == pytest.approx(
            float(np.mean(clf.predict_with_sensitive_attr(X, one_group) == y)), abs=1e-12
        )

    def test_to_dict_carries_every_field_including_the_third_state(self):
        """A field-by-field rebuild is how a later field gets dropped at the
        only boundary a consumer reads, so the keys are pinned to the
        dataclass's own field list rather than to a hand-written list."""
        import dataclasses

        from vfairness.in_processing.wrappers.sklearn_wrappers import FairClassifierResult

        result = FairClassifierResult(
            accuracy=0.5,
            fairness_violation=None,
            constraint_satisfied=None,
            fairness_metrics={"insufficient_data": True},
        )
        as_dict = result.to_dict()
        assert set(as_dict) == {f.name for f in dataclasses.fields(FairClassifierResult)}
        assert as_dict["fairness_violation"] is None
        assert as_dict["constraint_satisfied"] is None, "None, not the fail-closed False"


class TestPredictWithoutGroupsSaysTheMitigationIsNotApplied:
    def test_threshold_predict_warns_and_differs_from_the_adjusted_answer(self):
        """The group thresholds are fitted and stored; predict() cannot apply
        them, so it must say so rather than pass a uniform-0.5 answer off as
        the mitigated one."""
        clf, X, _, groups = _fitted("threshold")
        thresholds = clf._threshold_optimizer.get_thresholds()
        assert len(set(thresholds.values())) > 1, "fixture has no group-specific thresholds"

        with pytest.warns(UserWarning, match="Threshold method requires sensitive_attr"):
            plain = clf.predict(X)
        adjusted = clf.predict_with_sensitive_attr(X, groups)
        assert int(np.sum(plain != adjusted)) > 0, "the two answers really are different models"

    def test_control_the_other_methods_do_not_warn_because_nothing_is_missing(self):
        """OVER-CORRECTION CONTROL: reductions bakes fairness into the model,
        so predict() there is the mitigated answer and must stay silent."""
        clf, X, _, groups = _fitted("reductions")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plain = clf.predict(X)
        assert not caught, [str(w.message) for w in caught]
        assert np.array_equal(plain, clf.predict_with_sensitive_attr(X, groups))


# ===========================================================================
# 4. FairRegressor.score
# ===========================================================================


class TestFairRegressorScoreNamesItsModel:
    def _fitted(self, tolerance=0.1, **kwargs):
        X, y, groups = _regression_data()
        reg = FairRegressor(
            base_estimator=LinearRegression(),
            fairness_constraint="mean_parity",
            tolerance=tolerance,
            **kwargs,
        ).fit(X, y, sensitive_attr=groups)
        return reg, X, y, groups

    def test_scoring_without_groups_says_it_is_the_unadjusted_model(self):
        reg, X, y, _ = self._fitted()
        assert any(abs(v) > 1e-12 for v in reg.group_offsets_.values()), "no offsets to miss"
        with pytest.warns(UserWarning, match="UNADJUSTED"):
            unadjusted = reg.score(X, y)
        assert unadjusted == pytest.approx(_r2(y, reg._inner_model.predict(X)), abs=1e-12)

    def test_the_deployed_model_can_now_be_scored_and_is_a_different_number(self):
        """The reverse defect matters too: the fix must ADD the measurement
        that was missing, not refuse the one that was there."""
        reg, X, y, groups = self._fitted()
        adjusted = reg.score(X, y, sensitive_attr=groups)
        expected = _r2(y, reg.predict_with_sensitive_attr(X, groups))
        assert adjusted == pytest.approx(expected, abs=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unadjusted = reg.score(X, y)
        assert abs(adjusted - unadjusted) > 1e-6, "the mitigation must be visible in the score"
        assert unadjusted == pytest.approx(_r2(y, reg._inner_model.predict(X)), abs=1e-12)

    def test_control_a_regressor_with_nothing_to_adjust_scores_silently(self):
        """OVER-CORRECTION CONTROL: with a tolerance wide enough that every
        measured offset is 0.0, predict() IS the deployed model, so there is
        nothing to warn about and both calls must agree."""
        reg, X, y, groups = self._fitted(tolerance=50.0)
        assert all(v == 0.0 for v in reg.group_offsets_.values()), "fixture still adjusts"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unadjusted = reg.score(X, y)
        assert not [w for w in caught if "UNADJUSTED" in str(w.message)]
        assert unadjusted == pytest.approx(reg.score(X, y, sensitive_attr=groups), abs=1e-12)
        assert unadjusted == pytest.approx(_r2(y, reg._inner_model.predict(X)), abs=1e-12)

    @pytest.mark.parametrize("with_groups", [True, False])
    def test_a_constant_target_has_no_r2_to_report(self, with_groups):
        """R^2 is a ratio against the spread of y. With no spread it is 0/0,
        and sklearn's convention answers 1.0 for an exact fit: a perfect score
        for having explained nothing."""
        X, _, groups = _regression_data()
        y = np.full(X.shape[0], 5.0)
        reg = FairRegressor(LinearRegression(), tolerance=0.1).fit(X, y, sensitive_attr=groups)
        kwargs = {"sensitive_attr": groups} if with_groups else {}
        with pytest.warns(UserWarning, match="y is constant"):
            score = reg.score(X, y, **kwargs)
        assert math.isnan(score)

    def test_predict_without_groups_says_the_offsets_were_not_applied(self):
        """The regressor's own half of "the mitigation is not in this answer".
        predict() has no group information, so it must not pass the base
        predictions off as the mean-parity-adjusted ones."""
        reg, X, _, groups = self._fitted()
        with pytest.warns(UserWarning, match="UNADJUSTED"):
            plain = reg.predict(X)
        assert np.allclose(plain, reg._inner_model.predict(X))
        adjusted = reg.predict_with_sensitive_attr(X, groups)
        assert float(np.max(np.abs(plain - adjusted))) > 1e-6, "two different answers"

    def test_control_predict_is_silent_when_there_is_nothing_to_apply(self):
        """OVER-CORRECTION CONTROL: every measured offset is 0.0 here, so
        predict() IS the adjusted answer and warning would be noise."""
        reg, X, _, groups = self._fitted(tolerance=50.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plain = reg.predict(X)
        assert not [w for w in caught if "UNADJUSTED" in str(w.message)]
        assert np.allclose(plain, reg.predict_with_sensitive_attr(X, groups))

    def test_an_unseen_group_makes_the_score_nan_rather_than_a_partial_fit(self):
        reg, X, y, _ = self._fitted(on_unseen_group="nan")
        held_out = np.array(["a", "b", "c", "c", "b", "a"], dtype=object)
        with pytest.warns(RuntimeWarning, match="no mean-parity offset was fitted"):
            score = reg.score(X[:6], y[:6], sensitive_attr=held_out)
        assert math.isnan(score)
        assert reg.unseen_groups_ == {"c": 2}


# ===========================================================================
# 5. get_params / set_params
# ===========================================================================


class TestParamsRoundTrip:
    def test_the_regressor_rebuilds_with_the_policy_it_was_given(self):
        reg = FairRegressor(LinearRegression(), on_unseen_group="nan")
        params = reg.get_params(deep=False)
        assert "on_unseen_group" in params
        assert type(reg)(**params).on_unseen_group == "nan", "a clone must not swap the policy"

    def test_the_regressor_expands_and_routes_base_estimator_params(self):
        reg = FairRegressor(LinearRegression(fit_intercept=True))
        assert "base_estimator__fit_intercept" in reg.get_params(deep=True)
        reg.set_params(base_estimator__fit_intercept=False)
        assert reg.base_estimator.fit_intercept is False, "a silent no-op sweeps nothing"

    @pytest.mark.parametrize(
        "estimator",
        [FairClassifier(LogisticRegression()), FairRegressor(LinearRegression())],
        ids=["classifier", "regressor"],
    )
    def test_an_unknown_parameter_name_is_refused(self, estimator):
        with pytest.raises(ValueError, match="unknown parameter"):
            estimator.set_params(tolerence=99)
        assert estimator.tolerance != 99, "and nothing was applied"

    @pytest.mark.parametrize(
        "estimator",
        [FairClassifier(LogisticRegression()), FairRegressor(LinearRegression())],
        ids=["classifier", "regressor"],
    )
    def test_control_a_real_parameter_still_lands(self, estimator):
        """OVER-CORRECTION CONTROL: refusing everything would pass the test
        above and break the class."""
        estimator.set_params(tolerance=0.25, base_estimator__fit_intercept=False)
        assert estimator.tolerance == 0.25
        assert estimator.base_estimator.fit_intercept is False

    def test_fitted_state_is_not_a_parameter(self):
        """``hasattr(self, key)`` also accepted ``is_fitted_``, which would
        make an unfitted wrapper answer predict() out of a None inner model."""
        clf = FairClassifier(LogisticRegression())
        with pytest.raises(ValueError, match="unknown parameter"):
            clf.set_params(is_fitted_=True)
        assert clf.is_fitted_ is False

    def test_the_regressor_refuses_an_invalid_unseen_group_policy(self):
        reg = FairRegressor(LinearRegression())
        with pytest.raises(ValueError, match="on_unseen_group must be"):
            reg.set_params(on_unseen_group="banana")
        assert reg.on_unseen_group == "raise"


# ===========================================================================
# 6. The two factories
# ===========================================================================


class TestFactories:
    def test_make_fair_classifier_builds_something_that_measures_the_same_thing(self):
        X, y, groups = _classification_data()
        made = make_fair_classifier(
            LogisticRegression(max_iter=300),
            "demographic_parity",
            tolerance=0.05,
            method="reductions",
            max_iterations=5,
            random_state=0,
        )
        assert isinstance(made, FairClassifier)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            made.fit(X, y, sensitive_attr=groups)
        direct, _, _, _ = _fitted("reductions")
        assert made.fairness_result_.fairness_violation == pytest.approx(
            direct.fairness_result_.fairness_violation, abs=1e-12
        )
        assert made.fairness_result_.fairness_violation == pytest.approx(
            _demographic_parity_spread(made.predict_with_sensitive_attr(X, groups), groups),
            abs=1e-12,
        )

    def test_make_fair_regressor_does_not_smuggle_past_the_refusals(self):
        """A factory that reached fit() with an unimplemented constraint would
        silently fit an UNCONSTRAINED model under a fairness name."""
        X, y, groups = _regression_data()
        with pytest.raises(NotImplementedError, match="not .*implemented"):
            make_fair_regressor(LinearRegression(), "error_parity").fit(X, y, sensitive_attr=groups)

    def test_control_make_fair_regressor_fits_and_its_offsets_close_the_gap(self):
        """OVER-CORRECTION CONTROL: the factory's product really does the
        mitigation, checked by the group mean gap it claims to bound."""
        X, y, groups = _regression_data()
        reg = make_fair_regressor(LinearRegression(), tolerance=0.1).fit(
            X, y, sensitive_attr=groups
        )
        adjusted = reg.predict_with_sensitive_attr(X, groups)
        means = [float(np.mean(adjusted[groups == g])) for g in ("a", "b")]
        assert abs(means[0] - means[1]) <= 0.1 + 1e-9

        base = reg._inner_model.predict(X)
        base_means = [float(np.mean(base[groups == g])) for g in ("a", "b")]
        assert abs(base_means[0] - base_means[1]) > 0.1, "nothing was there to close"
