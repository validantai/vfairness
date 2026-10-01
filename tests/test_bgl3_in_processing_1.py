"""
BGL3 batch in_processing-1: does an in-processing unit refuse honestly?

Fourteen units across in_processing/analyzer.py,
in_processing/constraints/reductions.py and in_processing/diagnostics.py, graded
by EXECUTION on inputs where the quantity they report does not exist. Every
docstring below states what the unit returned BEFORE the fix, in the numbers the
probe printed.

The batch trap, and the reason half these findings are here: a NEUTERED
mitigation reports a BETTER fairness number. A classifier that predicts one class
for everybody has the same selection rate in every group, so its disparity is
exactly 0.0 and the constraint is "satisfied". The degeneracy has to be read off
the intervention's own parameters and output, never off the fairness metric it
produces.

Every test in this file was sabotage checked: the fix was re-broken, the test was
confirmed red, the fix was restored and the test was confirmed green. The
sabotage for each is named in the batch report.
"""

import math
import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from vfairness.in_processing.analyzer import (
    FairnessTrainingAnalyzer,
    baseline_comparison_summary,
)
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
    ReductionResult,
    _ConstantClassifier,
)
from vfairness.in_processing.diagnostics import (
    adversarial_convergence_diagnostics,
    sklearn_adversarial_debiasing,
)

# ---------------------------------------------------------------------------
# Fixtures. Two groups with very different base rates, so a real mitigation has
# something to do and a collapsed one has something to hide.
# ---------------------------------------------------------------------------

_N = 240


def _two_group_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
    x0 = rng.normal(loc=np.where(s == "a", 1.2, -1.2), scale=0.7)
    X = np.column_stack([x0, rng.normal(size=_N)])
    y = (x0 + rng.normal(scale=0.4, size=_N) > 0).astype(int)
    return X, y, s


def _adversarial_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return X, y


class _ProbaStub:
    """Member that reports a fixed probability and the label it implies."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict(self, X):
        return np.full(len(X), int(self.p >= 0.5), dtype=int)

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1.0 - p, p])


_X6 = np.zeros((6, 2))


# ===========================================================================
# 1. diagnostics.adversarial_convergence_diagnostics
# ===========================================================================


class TestATrajectoryWithNoVerdictInIt:
    def test_an_empty_trajectory_is_not_a_plateau(self):
        """BEFORE: classification 'plateau', oscillation_score 0.0, no warning.

        Measured on ``adversarial_convergence_diagnostics([], [])``. "plateau" is
        one of the four FINDINGS this function can return and it is worded
        "Debiasing stalled; the attribute stays partly recoverable", and an
        oscillation score of 0.0 is a perfectly settled game. Both were answers
        about a game with zero recorded rounds.
        """
        with pytest.warns(UserWarning, match="no adversary trajectory was recorded"):
            out = adversarial_convergence_diagnostics([], [])
        assert out["classification"] == "not_assessed"
        assert out["adversary_skill"] is None
        assert out["oscillation_score"] is None
        assert out["final_adversary_accuracy"] is None
        assert "NOT ASSESSED" in out["verdict"]

    def test_an_all_nan_trajectory_is_not_a_plateau(self):
        """BEFORE: classification 'plateau' and the verdict "Adversary accuracy
        plateaued at nan% against a 50.0% majority-class baseline (skill nan)
        without dropping to chance. Debiasing stalled; the attribute stays partly
        recoverable.", with no warning.

        Measured on five NaN rounds. The list comprehension dropped None and not
        NaN, and every comparison against NaN is False, so the trajectory fell
        past the converged test, past the oscillation test, past the diverged
        test and onto the plateau default.
        """
        nan5 = [float("nan")] * 5
        with pytest.warns(UserWarning, match="not a number"):
            out = adversarial_convergence_diagnostics(nan5, nan5)
        assert out["classification"] == "not_assessed"
        assert out["adversary_skill"] is None
        assert out["n_rounds_unusable"] == 5

    def test_a_partially_unusable_trajectory_says_how_much_it_lost(self):
        """One NaN mid-trajectory used to make the oscillation score NaN, and
        ``nan > 0.05`` is False, so a game that could not be checked for
        oscillation was treated as settled and fell through to the next test.

        The verdict is still given here, because the FINAL round is measurable;
        what is added is the count of rounds it rests on.

        REVISED (BGL4 audit wave 4, 2026-09-30) on the oscillation score only.
        This asserted ``math.isfinite(out["oscillation_score"])`` for the
        trajectory [0.70, nan, 0.70], whose only two usable rounds are round 1
        and round 3. The score is documented as the mean absolute SUCCESSIVE
        difference, and the finite 0.0 it used to report was the difference
        between rounds 1 and 3 with the gap closed up, i.e. a shape claim about
        rounds that never followed one another, landing on the perfectly-settled
        end of that scale. The count was disclosed and the claim built on the gap
        was not withdrawn. The subject of this test is unchanged and still
        asserted: the loss count and the verdict, which rests on the final round.
        """
        with pytest.warns(UserWarning, match="carry no usable adversary accuracy"):
            out = adversarial_convergence_diagnostics([0.1, 0.1, 0.1], [0.70, float("nan"), 0.70])
        assert out["n_rounds_unusable"] == 1
        assert out["classification"] == "diverged"
        assert out["oscillation_score"] is None
        assert out["n_gapped_pairs"] == 2

    def test_control_a_measurable_trajectory_still_gets_a_verdict(self):
        """OVER-CORRECTION CONTROL. The historical balanced-baseline thresholds
        are unchanged: skill below 0.10 converges, above 0.30 diverges.
        """
        for history, want in (([0.54] * 10, "converged"), ([0.70] * 10, "diverged")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = adversarial_convergence_diagnostics(history, history)
            assert out["classification"] == want
            assert out["n_rounds_unusable"] == 0
            assert not [str(w.message) for w in caught]


class TestDebiasingWithNothingToDebias:
    def test_a_single_protected_group_is_not_debiasing_converged(self):
        """BEFORE, end to end on 200 rows with one protected group, n_rounds=3:

            adversary_acc_history  [1.0, 1.0, 1.0]
            adversary_loss_history [0.0, 0.0, 0.0]
            majority_share         1.0
            classification         'converged'
            verdict                "Adversary accuracy is 100.0% against a
                                    100.0% majority-class baseline (skill 0.00),
                                    so the protected attribute is no longer
                                    recoverable from the model output beyond
                                    chance. Debiasing converged."

        and not one warning. A perfect adversary against a perfect baseline is
        not a measurement: with one group there is no attribute to recover, and
        the skill scale (1 - p) it is normalized by is zero.
        """
        X, y = _adversarial_data()
        with pytest.warns(UserWarning, match="single group"):
            res = sklearn_adversarial_debiasing(X, y, np.zeros(len(y), dtype=int), n_rounds=3)
        assert res["insufficient_data"] is True
        assert "single group" in res["not_assessed_reason"]
        assert all(math.isnan(a) for a in res["adversary_acc_history"])

        with pytest.warns(UserWarning, match="not_assessed"):
            diag = adversarial_convergence_diagnostics(
                res["adversary_loss_history"],
                res["adversary_acc_history"],
                majority_share=res["majority_share"],
            )
        assert diag["classification"] == "not_assessed"
        assert "Debiasing converged" not in diag["verdict"]

    def test_a_majority_share_of_one_is_not_a_converged_verdict(self):
        """SECOND LINE OF DEFENCE, pinned separately because the first one hides
        it. ``adversarial_convergence_diagnostics`` is public and takes
        majority_share directly, so a caller can hand it 1.0 beside a FINITE
        accuracy history that this library did not produce. BEFORE:
        ``adversarial_convergence_diagnostics([0.1] * 3, [1.0] * 3,
        majority_share=1.0)`` returned classification 'converged', skill 0.0 and
        the verdict "Adversary accuracy is 100.0% against a 100.0%
        majority-class baseline (skill 0.00), so the protected attribute is no
        longer recoverable from the model output beyond chance. Debiasing
        converged."

        Found by sabotage: reverting _adversary_skill's p >= 1 branch to 0.0 left
        the single-group test above GREEN, because that test now arrives with NaN
        accuracies and stops at the earlier branch. A guard that cannot fail
        looks identical to one that passed.
        """
        with pytest.warns(UserWarning, match="majority-group share"):
            out = adversarial_convergence_diagnostics([0.1] * 3, [1.0] * 3, majority_share=1.0)
        assert out["classification"] == "not_assessed"
        assert out["adversary_skill"] is None
        assert "Debiasing converged" not in out["verdict"]

    def test_a_single_class_target_says_no_predictor_was_fitted(self):
        """BEFORE, on 200 rows whose y is all one class, n_rounds=3: no predictor
        is fitted at all (the degenerate branch emits a constant score), yet the
        dict came back with 200 predictions, adversary_acc_history
        [0.52, 0.52, 0.52], and the convergence verdict read "Adversary accuracy
        is 52.0% ... Debiasing converged." for a run that trained no model. Zero
        warnings.

        The adversary numbers ARE real (a constant score is trivially
        uninformative), so they are kept; what was missing is that the debiasing
        claim rests on a model that does not exist.
        """
        X, _ = _adversarial_data()
        s = (X[:, 1] > 0).astype(int)
        with pytest.warns(UserWarning, match="single class"):
            res = sklearn_adversarial_debiasing(X, np.ones(len(X), dtype=int), s, n_rounds=3)
        assert res["insufficient_data"] is True
        assert res["predictor_fitted"] is False
        assert "no predictor is trained" in res["not_assessed_reason"]

    def test_zero_rounds_is_refused_rather_than_reported_as_a_fit(self):
        """BEFORE, with n_rounds=0 on 200 rows: the loop ran zero times, no
        predictor was ever fitted, all three histories came back empty, and
        ``predictions`` still held 200 labels (all of them the first class, from
        an untrained constant fallback) with no warning. A completed fit reported
        by a function that trained nothing.
        """
        X, y = _adversarial_data()
        s = (X[:, 1] > 0).astype(int)
        for bad in (0, -1):
            with pytest.raises(ValueError, match="plays no round at all"):
                sklearn_adversarial_debiasing(X, y, s, n_rounds=bad)

    def test_control_a_real_game_reports_a_fit_and_a_verdict(self):
        """OVER-CORRECTION CONTROL. Two groups and two classes: the histories are
        finite, insufficient_data is False, a predictor was fitted, and the
        trajectory is classified without a warning.
        """
        X, y = _adversarial_data()
        s = (X[:, 2] > 0).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sklearn_adversarial_debiasing(X, y, s, n_rounds=5)
            diag = adversarial_convergence_diagnostics(
                res["adversary_loss_history"],
                res["adversary_acc_history"],
                res["predictor_loss_history"],
                res["majority_share"],
            )
        assert not [str(w.message) for w in caught]
        assert res["insufficient_data"] is False
        assert res["predictor_fitted"] is True
        assert all(math.isfinite(a) for a in res["adversary_acc_history"])
        assert diag["classification"] in ("converged", "oscillating", "diverged", "plateau")
        assert math.isfinite(diag["adversary_skill"])

    def test_control_converged_is_still_reachable_on_imbalanced_groups(self):
        """OVER-CORRECTION CONTROL for the skill change. A 75/25 split whose
        attribute is genuinely unrecoverable must still read "converged": the
        NaN is for p == 1 only, not for every imbalanced p.
        """
        rng = np.random.default_rng(9)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        s = np.array([1] * 150 + [0] * 50)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sklearn_adversarial_debiasing(X, y, s, n_rounds=5)
            diag = adversarial_convergence_diagnostics(
                res["adversary_loss_history"],
                res["adversary_acc_history"],
                majority_share=res["majority_share"],
            )
        assert not [str(w.message) for w in caught]
        assert res["majority_share"] == pytest.approx(0.75)
        assert diag["classification"] == "converged"
        assert diag["adversary_skill"] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 2. reductions: an ensemble whose WEIGHTS decide the answer
# ===========================================================================


def _reduction_result(members, weights):
    from vfairness.in_processing.constraints.base import OptimizationResult

    return ReductionResult(
        classifiers=list(members),
        weights=np.asarray(weights, dtype=float),
        optimization_result=OptimizationResult(
            converged=True, n_iterations=1, final_violation=0.0, best_gap=0.0
        ),
        final_violation=0.0,
        accuracy=1.0,
    )


@pytest.mark.parametrize(
    "name, build",
    [
        ("EnsembleClassifier", lambda m, w: EnsembleClassifier(m, np.asarray(w, dtype=float))),
        ("ReductionResult", _reduction_result),
    ],
)
class TestAVoteThatCannotReachTheThreshold:
    @pytest.mark.parametrize("weights", [[0.2, 0.2], [0.3, 0.1], [0.49], [0.1]])
    def test_weights_below_the_threshold_are_refused(self, name, build, weights):
        """BEFORE: ``predict`` returned ``[0 0 0 0 0 0]`` with zero warnings while
        every member predicted 1 at probability 1.0.

        Measured on six rows. ``predict`` sums ``weight * label`` and returns
        ``sum >= 0.5``; labels are at most 1, so contributing weights totalling
        0.4 (or 0.49, or 0.1) can never reach the threshold and every row is
        rejected before any classifier is consulted. That all-zeros array is
        byte-identical to the one ``_refuse_a_silent_ensemble`` exists to refuse,
        and it walked past that guard because 0.2 IS above the negligible-weight
        cutoff: the old guard asked whether any member contributes, never whether
        what they contribute can change the answer.
        """
        members = [_ProbaStub(1.0) for _ in weights]
        with pytest.raises(ValueError, match="below the 0.5 decision threshold"):
            build(members, weights).predict(_X6)

    def test_a_negative_weight_is_refused_not_dropped(self, name, build):
        """BEFORE: ``predict_proba`` returned 0.9 for every row, the FIRST
        member's answer alone, while reporting a two-member ensemble.

        Measured with weights [1.0, -3.0]: ``-3.0 > 1e-8`` is False, so the
        second member was dropped from the vote in silence, exactly the failure
        the NaN branch of the guard was written for. It stopped one value short.
        """
        with pytest.raises(ValueError, match="negative"):
            build([_ProbaStub(0.9), _ProbaStub(0.1)], [1.0, -3.0]).predict_proba(_X6)

    def test_control_a_convex_ensemble_still_answers_in_silence(self, name, build):
        """OVER-CORRECTION CONTROL. The expected values are worked out here, not
        read back from the code: 0.25*0.2 + 0.75*0.8 == 0.65, which is at or
        above 0.5, so the label is 1. The fitted paths in this module hand over
        one-hot or uniform weights, both of which total exactly 1.
        """
        members = [_ProbaStub(0.2), _ProbaStub(0.8)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = build(members, [0.25, 0.75]).predict_proba(_X6)
            label = build(members, [0.25, 0.75]).predict(_X6)
        assert not [str(w.message) for w in caught]
        np.testing.assert_allclose(proba, np.full(6, 0.65))
        np.testing.assert_array_equal(label, np.full(6, 1, dtype=int))

    def test_control_a_negligible_member_beside_a_real_one_still_answers(self, name, build):
        """OVER-CORRECTION CONTROL. 1e-12 does not contribute, but the 1.0 beside
        it totals 1.0 on its own, so nothing is refused and nothing warns.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = build([_ProbaStub(0.0), _ProbaStub(0.9)], [1e-12, 1.0]).predict_proba(_X6)
        assert not [str(w.message) for w in caught]
        np.testing.assert_allclose(out, np.full(6, 0.9))


class TestPredictProbaSaysWhenItIsNotAProbability:
    @pytest.mark.parametrize(
        "build",
        [
            lambda m, w: EnsembleClassifier(m, np.asarray(w, dtype=float)),
            _reduction_result,
        ],
    )
    def test_a_collapsed_constant_member_is_disclosed(self, build):
        """BEFORE: ``np.unique(predict_proba(X)) == [0.]`` with zero warnings.

        ``_ConstantClassifier`` HAS a ``predict_proba``, so it slipped past the
        ``not hasattr(clf, "predict_proba")`` test written for hard-label members
        while returning exactly 0.0 or 1.0 for every row. It is this module's own
        marker for a collapsed best response, substituted by
        ``_fit_best_response`` whenever the cost-sensitive relabeling leaves one
        class, so it is the likeliest such member of all. A probability of
        exactly 0.0 on every row is the value the module's own refusal calls
        "byte-identical to a model that looked at the features and turned
        everyone down".
        """
        with pytest.warns(UserWarning, match="collapsed constant classifiers"):
            out = build([_ConstantClassifier(0)], [1.0]).predict_proba(_X6)
        np.testing.assert_allclose(out, np.zeros(6))

    def test_weights_that_are_not_a_convex_combination_are_disclosed(self):
        """BEFORE: two members both reporting probability 1.0 came back as 0.4
        from a method named ``predict_proba``, with zero warnings.

        Measured with weights [0.2, 0.2]. The result lives on a [0, total] scale,
        so it is not a probability and must not be fed to an AUC or a calibration
        curve. The value is kept, as the sibling vote-fraction warning keeps its
        value; the disclosure is what was missing.
        """
        with pytest.warns(UserWarning, match=r"weights total 0.4, not 1"):
            out = EnsembleClassifier(
                [_ProbaStub(1.0), _ProbaStub(1.0)], np.array([0.2, 0.2])
            ).predict_proba(_X6)
        np.testing.assert_allclose(out, np.full(6, 0.4))

    def test_control_a_probabilistic_convex_ensemble_does_not_warn(self):
        """OVER-CORRECTION CONTROL. Real probabilities under weights that sum to
        1 stay silent, and the result is strictly inside (0, 1).
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = EnsembleClassifier(
                [_ProbaStub(0.3), _ProbaStub(0.7)], np.array([0.5, 0.5])
            ).predict_proba(_X6)
        assert not [str(w.message) for w in caught]
        assert 0.0 < float(out.min()) and float(out.max()) < 1.0


# ===========================================================================
# 3. reductions.ExponentiatedGradient: the neutered mitigation
# ===========================================================================


class TestACollapsedFitIsNotAFairnessPass:
    def test_a_constant_model_does_not_certify_perfect_parity(self):
        """BEFORE, on 240 rows whose group base rates were 0.90 and 0.09, with a
        base estimator that always returns the majority class:

            final_violation                0.0
            accuracy                       0.5041666666666667
            fairness_metrics               {'constraint_satisfied': True,
                                            'insufficient_data': False,
                                            'n_iterations_unmeasurable': 0, ...}
            optimization_result.converged  True
            np.unique(result.predict(X))   [0]
            warnings                       none

        A compliance certificate for a classifier that rejected all 240 rows. The
        0.0 is arithmetically real (a constant classifier has the same rate in
        every group), which is exactly why the fairness metric cannot be the
        check: neutering the mitigation IMPROVES it. The best-iterate selection
        already defends against ONE collapsed iterate by preferring the most
        accurate satisfying one; that defence is empty when every iterate is
        collapsed, because they all satisfy.
        """
        X, y, s = _two_group_data()
        with pytest.warns(UserWarning, match="predicts the single class"):
            res = ExponentiatedGradient(
                DummyClassifier(strategy="most_frequent"),
                "demographic_parity",
                max_iterations=3,
            ).fit(X, y, sensitive_attr=s)

        assert res.fairness_metrics["degenerate_constant_predictions"] is True
        assert res.fairness_metrics["n_prediction_classes"] == 1
        # The flag survives the boundary a caller actually reads.
        assert res.to_dict()["fairness_metrics"]["degenerate_constant_predictions"] is True
        # The arithmetic is untouched: the disparity of a constant model IS 0.0.
        assert res.final_violation == pytest.approx(0.0)

    def test_control_a_real_fit_is_not_flagged_degenerate(self):
        """OVER-CORRECTION CONTROL. The same data with LogisticRegression
        predicts both classes, so the flag is False, n_prediction_classes is 2
        and nothing warns about degeneracy.
        """
        X, y, s = _two_group_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ExponentiatedGradient(
                LogisticRegression(max_iter=500), "demographic_parity", max_iterations=20
            ).fit(X, y, sensitive_attr=s)
            preds = res.predict(X)
        assert not [m for m in caught if "single class" in str(m.message)]
        assert res.fairness_metrics["degenerate_constant_predictions"] is False
        assert res.fairness_metrics["n_prediction_classes"] == 2
        assert len(np.unique(preds)) == 2

    @pytest.mark.parametrize("method", ["predict", "predict_proba"])
    def test_control_an_unfitted_estimator_still_refuses(self, method):
        """CORRECT-BEHAVIOUR PIN for ExponentiatedGradient.predict /
        predict_proba: an unfitted estimator raises rather than answering. Graded
        CORRECT in this batch and pinned so it stays that way.
        """
        eg = ExponentiatedGradient(LogisticRegression(), "demographic_parity")
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(eg, method)(_X6)


# ===========================================================================
# 4. analyzer: a spread with nothing to compare, and a score with nothing scored
# ===========================================================================


class TestABaseRateSpreadWithNoPair:
    def test_one_group_is_not_zero_base_rate_disparity(self):
        """BEFORE, through ``full_analysis`` on 200 rows with a single sensitive
        value: ``fairness_analysis['base_rate_disparity'] == 0.0``, ``summary()``
        printing "base_rate_disparity: 0.0000", no warning, and no critical
        issue, in the SAME report whose baseline violation, three method
        comparisons and recommendation all correctly said the constraint could
        not be evaluated. ``max(rates) - min(rates)`` over a ONE element list is
        0.0 by arithmetic vacuity, and 0.0 is identical base rates in every
        group.

        None and not NaN, because the rendering layer already speaks this third
        state: ``adapters_training`` sets ``base_rate_measured`` from
        ``base_rate_disparity is not None`` and the SVG template gates the row on
        it, so NaN would have been drawn as a measurement.
        """
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] > 0).astype(int)
        analyzer = FairnessTrainingAnalyzer(X, y, np.array(["A"] * 200))
        with pytest.warns(UserWarning, match="base_rate_disparity NOT MEASURED"):
            report = analyzer.full_analysis(base_estimator=LogisticRegression(max_iter=500))

        assert report.fairness_analysis["base_rate_disparity"] is None
        assert "0.0000" not in report.summary().split("base_rate_disparity")[1].splitlines()[0]
        assert "not measured" in report.summary().split("base_rate_disparity")[1]
        assert "Base Rate Disparity Not Measured" in [i["type"] for i in report.critical_issues]
        assert any("base rate disparity could not be computed" in a for a in report.action_items)

    def test_control_two_groups_still_measure_the_spread(self):
        """OVER-CORRECTION CONTROL. Two groups with base rates 0.90 and 0.09 give
        a measured disparity above 0.2, which raises the Base Rate Disparity
        issue at severity medium, and nothing warns.
        """
        X, y, s = _two_group_data()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis = analyzer._compute_fairness_analysis()
        assert not [m for m in caught if "base_rate_disparity" in str(m.message)]
        assert analysis["base_rate_disparity"] == pytest.approx(0.8083333333333333, abs=1e-9)
        issues = analyzer._identify_critical_issues({}, analysis, [])
        assert "Base Rate Disparity" in [i["type"] for i in issues]


class TestAReportWithNoBaselineInIt:
    def test_an_absent_baseline_accuracy_is_not_printed_as_zero(self):
        """BEFORE, on ``full_analysis(base_estimator=None)`` whose
        ``baseline_metrics`` is ``{}``:

            BASELINE PERFORMANCE
            Accuracy: 0.0000
            Fairness Violation: not measured (the constraint could not be evaluated)
            Constraint Satisfied: not assessed (the constraint could not be evaluated)

        A model that got every row wrong, invented on the line directly above two
        lines that refuse to invent anything. The SVG fallback canvas for this
        same report already refused it
        (test_adapters_training_empty.py::test_the_fallback_canvas_does_not_print
        _an_absent_baseline_as_zero); the text summary was the surface still
        doing it.
        """
        X, y, s = _two_group_data()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(base_estimator=None)
        assert report.baseline_metrics == {}
        accuracy_line = [ln for ln in report.summary().splitlines() if ln.startswith("Accuracy:")][
            0
        ]
        assert "0.0000" not in accuracy_line
        assert "not measured" in accuracy_line

    def test_control_a_measured_baseline_keeps_its_four_decimals(self):
        """OVER-CORRECTION CONTROL. A scored baseline still prints a number."""
        X, y, s = _two_group_data()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(
                base_estimator=LogisticRegression(max_iter=500),
                include_comparisons=False,
                include_tradeoffs=False,
            )
        accuracy_line = [ln for ln in report.summary().splitlines() if ln.startswith("Accuracy:")][
            0
        ]
        assert "not measured" not in accuracy_line
        assert float(accuracy_line.split(":")[1]) > 0.5


class TestAnRSquaredWithNoVarianceToExplain:
    @pytest.mark.parametrize("constant", [3.0, 0.1])
    def test_a_zero_variance_target_is_not_an_r2_of_minus_infinity(self, constant):
        """BEFORE, on 240 rows of a constant regression target: ``accuracy`` was
        ``nan`` when y_pred equalled y (0/0) and ``-inf`` when it was shifted,
        from a single numpy RuntimeWarning ("invalid value encountered in scalar
        divide") that names neither R^2 nor the target, and ``summary()`` printed
        "Accuracy: -inf". R^2 = 1 - MSE/Var(y) is UNDEFINED when the denominator
        is zero; a model that reproduces a constant target is not infinitely bad.

        Parametrized over TWO constants on purpose. The guard is ``np.ptp`` and
        not ``var == 0``: measured, ``np.var(np.full(240, 0.1))`` is
        1.925929944387236e-34, NOT 0.0, so a variance test would have refused
        3.0 and answered R^2 = -2.1e+34 for 0.1. Only one of these two cases
        would have failed, and it is not the round number.
        """
        rng = np.random.default_rng(1)
        X = rng.normal(size=(240, 2))
        s = np.where(X[:, 0] > 0, "A", "B")
        analyzer = FairnessTrainingAnalyzer(X, np.full(240, constant), s, task_type="regression")
        with pytest.warns(UserWarning, match="zero variance"):
            result = analyzer.evaluate_baseline(y_pred=np.full(240, constant + 2.0))
        assert math.isnan(result.accuracy)
        assert result.parameters["accuracy_measured"] is False

    def test_control_a_varying_target_still_scores(self):
        """OVER-CORRECTION CONTROL. A target with real variance gets a real R^2
        and accuracy_measured stays True.
        """
        rng = np.random.default_rng(2)
        X = rng.normal(size=(240, 2))
        y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=240)
        s = np.where(X[:, 1] > 0, "A", "B")
        analyzer = FairnessTrainingAnalyzer(X, y, s, task_type="regression")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.evaluate_baseline(y_pred=X[:, 0] * 2.0)
        assert not [m for m in caught if "zero variance" in str(m.message)]
        assert result.parameters["accuracy_measured"] is True
        assert result.accuracy > 0.9


class TestEveryMethodFailingToTrain:
    class _Exploding:
        def fit(self, *args, **kwargs):
            raise RuntimeError("boom")

        def predict(self, X):
            raise RuntimeError("boom")

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kwargs):
            return self

    def test_all_methods_failing_is_not_an_empty_list_of_findings(self):
        """BEFORE, with a base estimator whose fit raises: ``compare_methods``
        returned ``[]`` with one warning per method, and the report built from it
        carried NO critical issue, because the unevaluated-constraint branch of
        ``_identify_critical_issues`` is guarded by ``if method_comparisons``.
        Three methods that trained and could not be graded DID produce an issue,
        so the worse outcome (nothing trained at all) produced the quieter
        report, and the returned ``[]`` is what ``methods=[]`` returns too.
        """
        X, y, s = _two_group_data()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with pytest.warns(UserWarning, match="failed to train"):
            comparisons = analyzer.compare_methods(self._Exploding())
        assert comparisons == []
        assert len(analyzer.method_failures) == 3
        issues = analyzer._identify_critical_issues({}, {}, comparisons)
        assert "No Method Could Be Compared" in [i["type"] for i in issues]

    def test_control_methods_that_train_are_compared_and_ranked(self):
        """OVER-CORRECTION CONTROL for compare_methods AND the correct-behaviour
        pin for generate_recommendation and analyze_tradeoffs on measurable
        input: three methods train, all three are graded, a winner is named and
        no row of the trade-off sweep is left unevaluated.
        """
        X, y, s = _two_group_data()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparisons = analyzer.compare_methods(LogisticRegression(max_iter=300))
            tradeoffs = analyzer.analyze_tradeoffs(
                LogisticRegression(max_iter=300), lambda_values=[0.0, 0.5, 1.0]
            )
        assert analyzer.method_failures == []
        assert len(comparisons) == 3
        assert all(c.constraint_satisfied is not None for c in comparisons)
        assert tradeoffs["n_not_evaluated"] == 0
        assert len(tradeoffs["pareto_frontier"]) >= 1
        recommendation = analyzer.generate_recommendation(comparisons)
        assert recommendation.recommended_method not in ("N/A", "Unconstrained")

    def test_control_one_group_leaves_the_tradeoff_sweep_unranked(self):
        """CORRECT-BEHAVIOUR PIN for analyze_tradeoffs, graded CORRECT in this
        batch. On a single group every row's violation is None, the frontier is
        empty, best_fair is None and n_not_evaluated counts every row: no row was
        reported as Pareto optimal on a fairness axis it does not have.
        """
        rng = np.random.default_rng(5)
        X = rng.normal(size=(240, 3))
        y = (X[:, 0] > 0).astype(int)
        analyzer = FairnessTrainingAnalyzer(X, y, np.array(["A"] * 240))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tradeoffs = analyzer.analyze_tradeoffs(
                LogisticRegression(max_iter=300), lambda_values=[0.0, 1.0]
            )
        assert tradeoffs["n_not_evaluated"] == 2
        assert tradeoffs["pareto_frontier"] == []
        assert tradeoffs["best_fair"] is None
        assert all(
            r["violation"] is None and r["satisfied"] is None for r in tradeoffs["all_results"]
        )


class TestAComparisonWithNoRowsScored:
    def test_an_empty_test_set_is_not_no_accuracy_cost(self):
        """BEFORE, with ZERO test rows and both metrics dicts carrying a real
        disparity (0.30 to 0.05):

            accuracy_cost         0.0
            baseline['accuracy']  0.0
            fair['accuracy']      0.0
            verdict               "Fairness constraints reduced disparity by
                                   0.250 at no accuracy cost."

        and not one warning. ``_overall_accuracy`` answered 0.0 for an empty
        array, so BOTH sides got the same fabricated number and the difference
        between them was 0.0 as well: the sentence this whole function exists to
        produce, saying the intervention was free, computed without scoring a
        single row.
        """
        empty = np.array([])
        with pytest.warns(UserWarning, match="no accuracy was measurable"):
            out = baseline_comparison_summary(
                empty,
                empty,
                empty,
                {"g": empty},
                before_metrics={"demographic_parity_difference": 0.30},
                after_metrics={"demographic_parity_difference": 0.05},
            )
        assert out["accuracy_measured"] is False
        assert math.isnan(out["accuracy_cost"])
        assert out["n_test_rows"] == 0
        assert "no accuracy cost" not in out["verdict"]
        assert "NOT MEASURED" in out["verdict"]

    def test_control_a_scored_comparison_still_reports_a_cost(self):
        """OVER-CORRECTION CONTROL. Six scored rows: the cost is measured, the
        gain is measured and the verdict names both. The expected gain is worked
        out here: 0.3 minus 0.1 is 0.2.
        """
        y_test = np.array([0, 1, 0, 1, 1, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = baseline_comparison_summary(
                y_test,
                np.array([0, 1, 1, 1, 1, 0]),
                np.array([0, 1, 0, 0, 0, 0]),
                {"g": np.array(list("aabbab"))},
                before_metrics={"demographic_parity_difference": 0.3},
                after_metrics={"demographic_parity_difference": 0.1},
            )
        assert not [str(w.message) for w in caught]
        assert out["accuracy_measured"] is True
        assert out["n_test_rows"] == 6
        assert out["fairness_gain"] == pytest.approx(0.2)
        # Worked out by hand against y_test = [0, 1, 0, 1, 1, 0]: the baseline
        # gets rows 0, 1, 3, 4, 5 right (5 of 6) and the fair model gets rows
        # 0, 1, 2, 5 right (4 of 6), so the cost is 1/6.
        assert out["accuracy_cost"] == pytest.approx(1.0 / 6.0)
        assert "NOT MEASURED" not in out["verdict"]
