"""Wave-4 pins for the F10 in-processing analyzer/diagnostics package.

Six grades the independent auditor overturned on HEAD 9e88dd5ac, every one of
them the same class: a neutral value substituted for one that could not be
measured, then published as a measurement. Each test class names the door it
pins and the sibling door beside it, because in every one of these the guard had
been written for one door while its sibling stayed open.

Every pin is paired with a CONTROL asserting the healthy case's real number, not
merely that it did not raise: a guard that refuses everything passes every
refusal test.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.in_processing.analyzer import (
    FairnessTrainingAnalyzer,
    MethodComparison,
    baseline_comparison_summary,
)

BEFORE = {"demographic_parity_difference": 0.5}
AFTER = {"demographic_parity_difference": 0.25}


def _attr(n: int = 40) -> dict:
    half = n // 2
    return {"sex": np.array(["f"] * half + ["m"] * half)}


def _call(y_test, y_pred_baseline, y_pred_fair, attr=None, before=BEFORE, after=AFTER):
    """baseline_comparison_summary plus the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = baseline_comparison_summary(
            y_test,
            y_pred_baseline,
            y_pred_fair,
            _attr(len(np.asarray(y_test).reshape(-1))) if attr is None else attr,
            before,
            after,
        )
    return result, [str(w.message) for w in caught]


class TestF10R4AnUnlabelledTestSetIsNotAZeroAccuracyCost:
    """``baseline_comparison_summary`` counted ROWS where it needs LABELS.

    The empty-test-set guard is ``len(y_test) == 0``. Forty rows carrying nothing
    but NaN labels reproduced, to the digit, the sentence that guard's own
    comment gives as its whole justification: NaN equals nothing, so both models
    score a fabricated 0.0, the subtraction gives exactly 0.0, and
    ``_measured`` sees two finite zeros and answers True.
    """

    def test_forty_nan_labels_are_not_no_accuracy_cost(self):
        y = np.full(40, np.nan)
        pred = np.array([0, 1] * 20)
        result, messages = _call(y, pred, 1 - pred)

        assert result["accuracy_measured"] is False, (
            "40 rows of NaN labels scored NOTHING, so accuracy_measured must not be True. "
            "Got accuracy_cost={0!r}.".format(result["accuracy_cost"])
        )
        assert math.isnan(result["accuracy_cost"])
        assert "NOT MEASURED" in result["verdict"]
        assert "no accuracy cost" not in result["verdict"], (
            "the verdict SENTENCE is what a reader quotes; it may not say 'no accuracy "
            "cost' for a comparison in which no row was scored. Got: " + result["verdict"]
        )
        assert result["accuracy_not_measured_reason"], "the reason must travel with the flag"
        assert "non-finite" in result["accuracy_not_measured_reason"]
        assert messages, "a run that scored no row must warn"

    def test_the_per_group_table_is_the_second_writer_of_the_same_quantity(self):
        y = np.full(40, np.nan)
        pred = np.array([0, 1] * 20)
        result, _ = _call(y, pred, 1 - pred)

        assert all(math.isnan(v) for v in result["baseline"]["per_group_accuracy"].values())
        assert all(math.isnan(v) for v in result["fair"]["per_group_accuracy"].values())
        assert result["per_group_impact"], "the groups must still be listed, as holes"
        for entry in result["per_group_impact"]:
            assert entry["acc_measured"] is False, entry
            assert math.isnan(entry["delta"]), (
                "a delta of 0.0 reads as 'this group paid nothing for the intervention', "
                "which is a finding. Got " + repr(entry)
            )
            assert entry["not_measured_reason"]

    def test_a_partly_unlabelled_test_set_keeps_the_group_it_could_score(self):
        """NOT an over-correction: the labelled group is still measured."""
        y = np.concatenate([np.full(20, np.nan), np.array([0, 1] * 10)])
        pred = np.array([0, 1] * 20)
        result, _ = _call(y, pred, pred.copy())

        per_group = result["baseline"]["per_group_accuracy"]
        assert math.isnan(per_group["f"]), "the unlabelled half is a hole"
        assert per_group["m"] == 1.0, (
            "the labelled half was scored perfectly and that is a REAL measurement; "
            "got {0!r}".format(per_group["m"])
        )

    def test_the_graded_empty_test_set_refusal_still_fires(self):
        result, messages = _call(np.array([]), np.array([]), np.array([]), attr={})
        assert result["accuracy_measured"] is False
        assert math.isnan(result["accuracy_cost"])
        assert "NOT MEASURED" in result["verdict"]
        assert len(messages) == 1, messages

    def test_control_real_labels_report_the_real_cost(self):
        y = np.array([0, 1] * 20)
        fair = y.copy()
        fair[:4] = 1 - fair[:4]  # 4 of 40 wrong => 0.9
        result, messages = _call(y, y.copy(), fair)

        assert result["accuracy_measured"] is True
        assert result["baseline"]["accuracy"] == 1.0
        assert result["fair"]["accuracy"] == pytest.approx(0.9)
        assert result["accuracy_cost"] == pytest.approx(0.1)
        assert "0.100 accuracy cost" in result["verdict"], result["verdict"]
        assert result["accuracy_not_measured_reason"] == ""
        assert result["baseline"]["per_group_accuracy"] == {"f": 1.0, "m": 1.0}
        assert result["fair"]["per_group_accuracy"]["f"] == pytest.approx(0.8)
        assert result["fair"]["per_group_accuracy"]["m"] == 1.0
        assert all(e["acc_measured"] is True for e in result["per_group_impact"])
        assert messages == []


class TestF10R4ABroadcastIsNotAnAccuracy:
    """The sibling door into the same sentence: a column prediction.

    ``np.mean(y_pred == y_test)`` with a (40, 1) prediction against (40,) labels
    broadcasts to a 40x40 matrix whose mean is the chance rate of the class mix.
    Measured on a PERFECT prediction it published 0.5 with zero warnings, both
    overall and per group. This is the fix already carried by
    loss_functions/adversarial.py, applied to the reporting surface.
    """

    def test_a_perfect_column_prediction_scores_one(self):
        y = np.array([0, 1] * 20)
        result, messages = _call(y, y.reshape(-1, 1), y.reshape(-1, 1))

        assert result["baseline"]["accuracy"] == 1.0, (
            "a perfect prediction handed over as a (40, 1) column is still perfect; "
            "0.5 is the chance rate of the class mix. Got "
            "{0!r}".format(result["baseline"]["accuracy"])
        )
        assert result["fair"]["accuracy"] == 1.0
        assert result["baseline"]["per_group_accuracy"] == {"f": 1.0, "m": 1.0}
        assert messages == []

    def test_a_prediction_count_that_does_not_line_up_is_could_not_check(self):
        y = np.array([0, 1] * 20)
        result, messages = _call(y, y[:39], y[:39])

        assert result["accuracy_measured"] is False
        assert math.isnan(result["accuracy_cost"])
        assert "NOT MEASURED" in result["verdict"]
        assert "cannot be compared row by row" in result["accuracy_not_measured_reason"]
        assert messages


class TestF10R3ANaNAccuracyMustNotWinAMaxComparison:
    """The NaN pathology was fixed for ``fairness_violation`` and left open for
    ``accuracy`` three lines below it, in the same function.

    Every comparison against NaN is False, so ``max`` keeps whatever it saw
    first: the recommendation was decided by LIST ORDER, and the rationale
    asserted a best accuracy of nan while claiming the constraint was achieved.
    """

    @staticmethod
    def _analyzer():
        rng = np.random.default_rng(0)
        y = np.array([0] * 30 + [1] * 30)
        return FairnessTrainingAnalyzer(
            rng.normal(size=(60, 3)), y, np.array(["a"] * 30 + ["b"] * 30)
        )

    @staticmethod
    def _recommend(analyzer, comparisons):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rec = analyzer.generate_recommendation(comparisons)
        return rec, [str(w.message) for w in caught]

    @pytest.mark.parametrize("nan_first", [True, False])
    def test_the_winner_does_not_depend_on_list_order(self, nan_first):
        unmeasurable = MethodComparison("Unmeasurable", float("nan"), 0.01, True)
        real = MethodComparison("Real", 0.82, 0.02, True)
        order = [unmeasurable, real] if nan_first else [real, unmeasurable]
        rec, messages = self._recommend(self._analyzer(), order)

        assert rec.recommended_method == "Real", (
            "a method whose accuracy nobody could score must not win a max() over "
            "accuracy; got {0!r} for nan_first={1}".format(rec.recommended_method, nan_first)
        )
        assert "nan" not in rec.rationale.lower()
        assert "0.820" in rec.rationale
        assert not math.isnan(rec.expected_tradeoff["accuracy"])
        assert messages, "leaving a method out of the ranking must be disclosed"

    def test_no_measured_accuracy_at_all_names_no_winner(self):
        both_nan = [
            MethodComparison("A", float("nan"), 0.01, True),
            MethodComparison("B", float("nan"), 0.02, True),
        ]
        rec, messages = self._recommend(self._analyzer(), both_nan)

        assert rec.recommended_method == "N/A", (
            "with no measurable accuracy anywhere there is no best method to name; "
            "got " + repr(rec.recommended_method)
        )
        assert "nan" not in rec.rationale.lower()
        assert messages

    def test_a_collapsed_constant_classifier_is_not_an_achieved_constraint(self):
        collapsed = [
            MethodComparison(
                "Reductions",
                0.6916666666666667,
                0.0,
                True,
                parameters={"method": "reductions", "degenerate_constant_predictions": True},
            ),
            MethodComparison(
                "Threshold",
                0.6916666666666667,
                0.0,
                True,
                parameters={"method": "threshold", "degenerate_constant_predictions": True},
            ),
        ]
        rec, messages = self._recommend(self._analyzer(), collapsed)

        assert rec.recommended_method == "N/A", (
            "a constant classifier is trivially perfectly fair because it made no "
            "decision; recommending it at priority HIGH with 'Achieves fairness "
            "constraint' is the fabrication. Got " + repr(rec.recommended_method)
        )
        assert rec.priority != "high"
        assert "constant" in rec.rationale.lower()
        assert messages

    def test_one_collapsed_method_beside_one_real_one_recommends_the_real_one(self):
        mixed = [
            MethodComparison(
                "Collapsed",
                0.95,
                0.0,
                True,
                parameters={"degenerate_constant_predictions": True},
            ),
            MethodComparison("Real", 0.82, 0.02, True, parameters={}),
        ]
        rec, messages = self._recommend(self._analyzer(), mixed)

        assert rec.recommended_method == "Real", (
            "the collapsed method has the better accuracy AND a zero violation, so it "
            "wins on both fields; it must still be excluded. Got " + repr(rec.recommended_method)
        )
        assert "0.820" in rec.rationale
        assert messages

    def test_control_two_measured_accuracies_pick_the_higher(self):
        rec, messages = self._recommend(
            self._analyzer(),
            [
                MethodComparison("Worse", 0.82, 0.02, True, parameters={}),
                MethodComparison("Better", 0.91, 0.03, True, parameters={}),
            ],
        )
        assert rec.recommended_method == "Better"
        assert rec.priority == "high"
        assert "0.910" in rec.rationale, rec.rationale
        assert rec.expected_tradeoff["accuracy"] == pytest.approx(0.91)
        assert rec.alternative_methods == ["Worse"]
        assert messages == []

    def test_control_the_graded_violation_ranking_is_untouched(self):
        rec, _ = self._recommend(
            self._analyzer(),
            [
                MethodComparison("Unranked", 0.9, None, None, parameters={}),
                MethodComparison("Ranked", 0.7, 0.2, False, parameters={}),
            ],
        )
        assert rec.recommended_method == "Ranked"
        assert rec.priority == "medium"
        assert "0.200" in rec.rationale


class TestF10R2AFiniteAbsurdRSquaredIsNotAMeasuredScore:
    """The FINITE sibling of ``evaluate_baseline``'s zero-variance guard.

    The guard is ``np.ptp(self.y) == 0.0``, and its own comment defends np.ptp
    over a variance test by naming the number a variance test would have let
    through, R^2 = -2.1e+34. One ulp away from constant the same number is back:
    the tests are absolute, and the quantity they judge is relative. The final
    derived-flag guard only normalises values that fail ``_measured``, i.e.
    non-finite ones, so a finite -5.78e+29 sailed through as a measured R^2.
    """

    N = 60

    def _evaluate(self, y, y_pred, task="regression"):
        rng = np.random.default_rng(0)
        analyzer = FairnessTrainingAnalyzer(
            rng.normal(size=(self.N, 3)),
            y,
            np.array(["a"] * (self.N // 2) + ["b"] * (self.N // 2)),
            task_type=task,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.evaluate_baseline(y_pred=y_pred)
        return result, [str(w.message) for w in caught]

    @pytest.mark.parametrize("base,bump", [(0.1, 1e-15), (3.0, 1e-12), (1.0, 1e-14), (1e6, 1e-6)])
    def test_a_target_constant_to_within_float_noise_is_not_scored(self, base, bump):
        y = np.full(self.N, base)
        y[0] += bump
        result, messages = self._evaluate(y, np.zeros(self.N))

        assert result.parameters["accuracy_measured"] is False, (
            "a target whose spread is below the floating-point resolution of its own "
            "magnitude has no variance for R^2 to divide by; got accuracy "
            "{0!r}".format(result.accuracy)
        )
        assert math.isnan(result.accuracy), (
            "a large FINITE negative R^2 is the whole defect: it is not caught by any "
            "non-finite check and every consumer prints it. Got " + repr(result.accuracy)
        )
        assert result.parameters["accuracy_not_measured_reason"]
        assert "relative spread" in result.parameters["accuracy_not_measured_reason"]
        assert messages

    def test_the_graded_exactly_constant_refusal_still_fires(self):
        result, messages = self._evaluate(np.full(self.N, 0.1), np.full(self.N, 0.1))
        assert result.parameters["accuracy_measured"] is False
        assert math.isnan(result.accuracy)
        assert "zero variance" in result.parameters["accuracy_not_measured_reason"]
        assert len(messages) == 1, messages

    def test_control_a_real_regression_target_scores_its_real_r_squared(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(self.N, 3))
        y = 2 * X[:, 0] + rng.normal(scale=0.1, size=self.N)
        # Same generator sequence as _evaluate, so the fixture is reproducible.
        analyzer = FairnessTrainingAnalyzer(
            X, y, np.array(["a"] * 30 + ["b"] * 30), task_type="regression"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.evaluate_baseline(y_pred=2 * X[:, 0])

        assert result.parameters["accuracy_measured"] is True
        assert result.accuracy == pytest.approx(0.997, abs=5e-3), result.accuracy
        assert result.parameters["accuracy_not_measured_reason"] == ""
        assert [m for m in caught if "accuracy" in str(m.message)] == []

    def test_control_a_tight_but_real_target_is_still_measured(self):
        """OVER-CORRECTION CONTROL. A relative spread of 1e-5 is four orders
        above the cancellation floor and is a real, if tight, regression
        problem."""
        y = np.linspace(100.0, 100.001, self.N)
        result, _ = self._evaluate(y, y.copy())

        assert result.parameters["accuracy_measured"] is True, (
            "a target varying by 0.001 over a magnitude of 100 is real variation; "
            "refusing it would be the guard refusing everything"
        )
        assert result.accuracy == pytest.approx(1.0)


class TestF10R2ABroadcastIsNotAnAccuracyInEvaluateBaseline:
    """The same broadcast, on the reporting arm, above the branch dispatch.

    ``coerce_to_array`` returns an ndarray unchanged, so it does not flatten. A
    (60, 1) prediction against a (60,) target expands ``y_pred == self.y`` to a
    60x60 matrix whose mean is the chance rate of the class mix: 0.5 for a
    PERFECT model, with zero warnings. Both branches read y_pred elementwise
    against self.y, so the guard belongs above the selection between them.
    """

    N = 60

    def _evaluate(self, y, y_pred, task="classification"):
        rng = np.random.default_rng(0)
        analyzer = FairnessTrainingAnalyzer(
            rng.normal(size=(self.N, 3)),
            y,
            np.array(["a"] * (self.N // 2) + ["b"] * (self.N // 2)),
            task_type=task,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.evaluate_baseline(y_pred=y_pred)
        return result, [str(w.message) for w in caught]

    def test_a_perfect_column_prediction_scores_one(self):
        y = np.array([0] * 30 + [1] * 30)
        result, _ = self._evaluate(y, y.reshape(-1, 1))
        assert result.accuracy == 1.0, (
            "0.5 is the chance rate of the class mix, published for a perfect model. "
            "Got " + repr(result.accuracy)
        )
        assert result.parameters["accuracy_measured"] is True

    def test_the_regression_branch_broadcasts_the_same_way(self):
        """THE SIBLING BRANCH. (y_pred - self.y) ** 2 expands to 60x60 too."""
        rng = np.random.default_rng(0)
        y = 2 * rng.normal(size=self.N)
        flat, _ = self._evaluate(y, y.copy(), task="regression")
        column, _ = self._evaluate(y, y.reshape(-1, 1), task="regression")
        assert flat.accuracy == pytest.approx(1.0)
        assert column.accuracy == pytest.approx(1.0), (
            "a perfect regression prediction handed over as a column is still perfect; "
            "got " + repr(column.accuracy)
        )

    def test_a_prediction_count_that_does_not_line_up_is_refused_by_name(self):
        """A crash is not one of the three states.

        A 59-for-60 mismatch cannot be reconciled by reshaping and nothing in
        this method is computable from it: it used to escape as a bare
        ``IndexError: boolean index did not match indexed array`` out of
        constraints/base.py:775, several frames below the call.
        """
        y = np.array([0] * 30 + [1] * 30)
        with pytest.raises(ValueError, match="59 prediction\\(s\\) for 60 target row\\(s\\)"):
            self._evaluate(y, y[:59])

    def test_control_a_flat_prediction_is_unchanged(self):
        y = np.array([0] * 30 + [1] * 30)
        y_pred = y.copy()
        y_pred[:6] = 1 - y_pred[:6]  # 6 of 60 wrong => 0.9
        result, messages = self._evaluate(y, y_pred)
        assert result.accuracy == pytest.approx(0.9)
        assert result.parameters["accuracy_measured"] is True
        assert result.parameters["accuracy_not_measured_reason"] == ""
        assert [m for m in messages if "accuracy" in m] == []

    def test_control_the_graded_all_nan_target_refusal_still_fires(self):
        result, messages = self._evaluate(np.full(self.N, np.nan), np.full(self.N, np.nan))
        assert result.parameters["accuracy_measured"] is False
        assert math.isnan(result.accuracy)
        assert "non-finite" in result.parameters["accuracy_not_measured_reason"]
        assert messages


class TestF10R1TheDegeneracyOfAMethodThatSucceeded:
    """``compare_methods`` disclosed partial FAILURE and dropped the COLLAPSE.

    A method that trained happily and collapsed to a constant prediction
    satisfies every rate-based constraint by construction, because approving (or
    refusing) everybody is exactly equal treatment. ``parameters`` carried only
    "method" and "constraint_evaluated", so a comparison in which every
    mitigation collapsed rendered as a complete clean comparison, and
    ``generate_recommendation`` named one at priority HIGH.

    Second gap on the same boundary: FairClassifier computes accuracy as
    ``float(np.mean(y_pred == y))`` with no unscorable-row guard and this method
    passed it through with no ``accuracy_measured`` key, while its SIBLING arm
    ``evaluate_baseline`` was given exactly that three-state treatment.
    """

    N = 240

    @staticmethod
    def _split_base_rates(seed=0, n=240):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 3))
        s = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        y = np.where(
            s == "a", (rng.random(n) < 0.85).astype(int), (rng.random(n) < 0.53).astype(int)
        )
        return X, y, s

    def test_a_collapsed_mitigation_is_named_on_the_comparison(self):
        from sklearn.dummy import DummyClassifier

        X, y, s = self._split_base_rates()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            comparisons = analyzer.compare_methods(DummyClassifier(strategy="most_frequent"))

        assert len(comparisons) == 3
        majority = float(max(np.mean(y), 1 - np.mean(y)))
        for c in comparisons:
            # The premise of the fixture, asserted rather than assumed: every arm
            # really did collapse, which is why its fairness number is vacuous.
            assert c.accuracy == pytest.approx(majority), c.method_name
            assert c.fairness_violation == pytest.approx(0.0), c.method_name
            assert c.constraint_satisfied is True
            assert c.parameters["degenerate_constant_predictions"] is True, (
                "{0} predicted one label for all {1} rows and its 0.0 violation is "
                "vacuous; the flag is the only thing that says so. params={2!r}".format(
                    c.method_name, self.N, c.parameters
                )
            )
            assert c.parameters["n_prediction_classes"] == 1
            assert "accuracy_measured" in c.parameters, (
                "the sibling arm evaluate_baseline puts this key on the same object type"
            )
        collapse_warnings = [w for w in caught if "collapsed to a CONSTANT" in str(w.message)]
        assert len(collapse_warnings) == 3, [str(w.message)[:80] for w in caught]

    def test_the_recommendation_over_a_fully_collapsed_comparison_names_no_winner(self):
        from sklearn.dummy import DummyClassifier

        X, y, s = self._split_base_rates()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            comparisons = analyzer.compare_methods(DummyClassifier(strategy="most_frequent"))
            rec = analyzer.generate_recommendation(comparisons)

        assert rec.recommended_method == "N/A", (
            "before: 'Reductions' at priority HIGH with 'Achieves fairness constraint "
            "with best accuracy (0.692)'. Got " + repr(rec.recommended_method)
        )
        assert rec.priority != "high"
        assert "Achieves fairness constraint" not in rec.rationale
        assert "collapsed" in rec.rationale

    def test_the_flag_reads_the_prediction_the_arm_was_actually_scored_on(self):
        """NOT an over-accusation: predict() is a different quantity.

        ``FairClassifier.predict`` for method='threshold' returns the uniform
        0.5 answer and says so in its own docstring, while ``_fit_threshold``
        scored ``_threshold_optimizer.predict(y_prob, sensitive_attr)``. Judging
        that arm by an output it does not consume flagged it as collapsed while
        its own recorded violation was 0.0083, i.e. not constant at all.
        """
        from sklearn.linear_model import LogisticRegression

        X, y, s = self._split_base_rates()
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            comparisons = analyzer.compare_methods(LogisticRegression(max_iter=1000))

        by_name = {c.method_name: c for c in comparisons}
        threshold = by_name["Threshold"]
        assert threshold.fairness_violation > 0.0, "fixture premise: this arm decided"
        assert threshold.parameters["degenerate_constant_predictions"] is False, (
            "an arm whose own recorded violation is {0!r} did not predict a constant; "
            "reading clf.predict() instead of the mitigated prediction said it "
            "did".format(threshold.fairness_violation)
        )
        assert threshold.parameters["n_prediction_classes"] == 2
        # And the flag still fires on this same run for the arms that DID
        # collapse, so it is discriminating rather than off.
        assert by_name["Reductions"].parameters["degenerate_constant_predictions"] is True
        assert by_name["Reductions"].fairness_violation == pytest.approx(0.0)

    def test_control_a_healthy_comparison_is_silent_and_unchanged(self):
        """OVER-CORRECTION CONTROL: no flag, no note, no warning, real numbers."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(7)
        n = 240
        X = rng.normal(size=(n, 3))
        s = np.array(["a"] * 120 + ["b"] * 120)
        y = ((X[:, 0] + 0.6 * (s == "a")) > 0).astype(int)
        analyzer = FairnessTrainingAnalyzer(X, y, s, tolerance=0.15)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            comparisons = analyzer.compare_methods(LogisticRegression(max_iter=1000))
            rec = analyzer.generate_recommendation(comparisons)

        assert len(comparisons) == 3
        for c in comparisons:
            assert c.parameters["degenerate_constant_predictions"] is False, c.method_name
            assert c.parameters["n_prediction_classes"] == 2
            assert c.parameters["accuracy_measured"] is True
            assert c.parameters["accuracy_not_measured_reason"] == ""
            assert c.accuracy > 0.8, c.method_name
        assert rec.recommended_method == "Threshold"
        assert rec.priority == "high"
        assert rec.rationale == "Achieves fairness constraint with best accuracy (0.917)", (
            "no note may be appended to a clean run. Got " + repr(rec.rationale)
        )
        assert [str(w.message) for w in caught] == []

    def test_an_arm_scored_over_unlabelled_rows_is_not_a_measured_accuracy(self):
        """The SIBLING ARM's missing three-state treatment, reachable.

        FairClassifier computes accuracy as ``float(np.mean(y_pred == y))`` with
        no unscorable-row guard, and this method passed it through with no
        ``accuracy_measured`` key, while ``evaluate_baseline`` right above it was
        given exactly that treatment. ``nan == anything`` is False, so an
        unlabelled row entered the mean as a row the model got WRONG, and the
        result is the field ``generate_recommendation`` ranks over.
        """
        from sklearn.dummy import DummyClassifier

        rng = np.random.default_rng(3)
        n = 120
        X = rng.normal(size=(n, 3))
        s = np.array(["a"] * 60 + ["b"] * 60)
        y = (X[:, 0] > 0).astype(float)
        y[:10] = np.nan  # 10 of 120 rows carry no label

        analyzer = FairnessTrainingAnalyzer(X, y, s, tolerance=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            comparisons = analyzer.compare_methods(DummyClassifier(strategy="stratified"))

        assert comparisons, "fixture premise: at least one arm must train"
        for c in comparisons:
            assert c.parameters["accuracy_measured"] is False, (
                "{0} reported an accuracy scored over 10 unlabelled rows; before this "
                "it arrived with no accuracy_measured key at all. params={1!r}".format(
                    c.method_name, c.parameters
                )
            )
            assert math.isnan(c.accuracy)
            assert "10 of 120 row(s)" in c.parameters["accuracy_not_measured_reason"]
        assert [w for w in caught if "not a measurement" in str(w.message)]

        # And the consumer withdraws the verdict rather than ranking over NaN.
        rec = analyzer.generate_recommendation(comparisons)
        assert rec.recommended_method == "N/A"
        assert "nan" not in rec.rationale.lower()


# ===========================================================================
# diagnostics.py
# ===========================================================================

from vfairness.in_processing.diagnostics import (  # noqa: E402
    adversarial_convergence_diagnostics,
    sklearn_adversarial_debiasing,
)

FIVE_ROUND_LOSS = [0.7] * 5


def _acd(acc, loss=None, pred=None, share=0.5, degenerate=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = adversarial_convergence_diagnostics(
            FIVE_ROUND_LOSS if loss is None else loss, acc, pred, share, degenerate
        )
    return out, [str(w.message) for w in caught]


class TestF10R5NoneIsNotADeletion:
    """A round recorded as None VANISHED; the same round as NaN was disclosed.

    ``acc = [float(a) for a in (history or []) if a is not None]`` filtered the
    round out BEFORE n_rounds_unusable counted it, so a reader was handed a
    two-entry trajectory, n_rounds_usable 2 and n_rounds_unusable 0 for a
    FIVE-round run, with nothing saying a round was missing. The comment on that
    line considered this channel and dismissed it, while adversary_acc_history is
    a public List[float] argument any caller loop can put None into.
    """

    def test_none_rounds_count_exactly_as_nan_rounds(self):
        as_nan, nan_msgs = _acd([0.6, float("nan"), float("nan"), float("nan"), 0.62])
        as_none, none_msgs = _acd([0.6, None, None, None, 0.62])

        assert as_none["n_rounds_unusable"] == 3, (
            "three of five rounds carry no accuracy; before this they were dropped and "
            "the count read 0. Got " + repr(as_none["n_rounds_unusable"])
        )
        assert as_none["n_rounds_usable"] == 2
        assert len(as_none["adversary_accuracy_trajectory"]) == 5, (
            "the trajectory must keep one entry per RECORDED round, not per readable "
            "one; got " + repr(as_none["adversary_accuracy_trajectory"])
        )
        assert as_none["classification"] == as_nan["classification"]
        assert as_none["n_rounds_unusable"] == as_nan["n_rounds_unusable"]
        assert none_msgs, "the same five-round game must not be silent when written None"
        assert len(none_msgs) == len(nan_msgs)

    @pytest.mark.parametrize("absent", [None, float("nan"), "", "None"])
    def test_every_door_of_absence_arrives_at_the_same_machinery(self, absent):
        out, messages = _acd([0.6, absent, absent, absent, 0.62])
        assert out["n_rounds_unusable"] == 3, (absent, out["n_rounds_unusable"])
        assert len(out["adversary_accuracy_trajectory"]) == 5
        assert messages

    def test_control_a_clean_five_round_game_is_silent_and_unchanged(self):
        out, messages = _acd([0.6, 0.62, 0.61, 0.60, 0.62])
        assert out["classification"] == "plateau"
        assert out["oscillation_score"] == pytest.approx(0.015)
        assert out["n_rounds_usable"] == 5
        assert out["n_rounds_unusable"] == 0
        assert out["n_gapped_pairs"] == 0
        assert out["final_adversary_loss"] == 0.7
        assert out["adversary_skill"] == pytest.approx(0.24)
        assert messages == []


class TestF10R5ASuccessiveDifferenceCannotCrossAGap:
    """``oscillation_score`` is documented as the mean absolute SUCCESSIVE
    difference and it gates the classification. Compacting the window to the
    usable rounds took the difference ACROSS the gaps and called it successive:
    the count of unusable rounds was disclosed and the SHAPE claim built on top
    of the gaps was not withdrawn.
    """

    def test_a_score_is_not_computed_across_the_gaps(self):
        out, messages = _acd([0.6, None, 0.95, None, 0.6])

        assert out["oscillation_score"] is None, (
            "0.35 was the difference between rounds 1 and 3 and rounds 3 and 5, "
            "published as 'Adversary accuracy oscillates (score 0.350)'. Got "
            + repr(out["oscillation_score"])
        )
        assert out["classification"] != "oscillating"
        assert "oscillates" not in out["verdict"]
        assert out["n_gapped_pairs"] == 4
        assert messages

    def test_the_same_is_true_on_the_already_disclosed_nan_path(self):
        out, _ = _acd([0.6, float("nan"), float("nan"), float("nan"), 0.62])
        assert out["oscillation_score"] is None, (
            "0.020 was reported as a successive difference between round 1 and round 5"
        )
        assert out["classification"] == "not_assessed"
        assert "plateaued" not in out["verdict"]

    def test_control_adjacent_usable_rounds_still_score(self):
        """NOT an over-correction: a gap elsewhere does not void the pairs that
        ARE adjacent."""
        out, _ = _acd([0.6, 0.90, 0.6, float("nan"), 0.62])
        assert out["oscillation_score"] == pytest.approx(0.3), out["oscillation_score"]
        assert out["classification"] == "oscillating"
        assert out["n_gapped_pairs"] == 2
        assert out["n_rounds_unusable"] == 1

    def test_control_the_graded_one_round_refusal_still_fires(self):
        out, messages = _acd([0.6], loss=[0.62])
        assert out["oscillation_score"] is None
        assert out["classification"] == "not_assessed"
        assert out["adversary_skill"] == pytest.approx(0.2)
        assert messages


class TestF10R5TheLossSeriesIsNeverChecked:
    """``n_rounds_unusable`` counted non-finite ACCURACY only, so the loss half
    of the trajectory was returned to the reader as ``final_adversary_loss`` and
    was never tested."""

    def test_an_all_nan_loss_history_is_not_a_measured_final_loss(self):
        out, messages = _acd([0.6, 0.62], loss=[float("nan"), float("nan")])

        assert out["final_adversary_loss"] is None, (
            "nan was handed over as final_adversary_loss with n_rounds_unusable 0 and "
            "zero warnings; None is this field's own third state. Got "
            + repr(out["final_adversary_loss"])
        )
        assert out["n_loss_rounds_unusable"] == 2
        assert [m for m in messages if "usable adversary LOSS" in m]
        # The ACCURACY series is fully usable here, so its verdict stands: the
        # loss disclosure must not swallow a measurement that was made.
        assert out["classification"] == "plateau"
        assert out["n_rounds_unusable"] == 0

    def test_control_a_real_loss_history_is_reported_as_measured(self):
        out, messages = _acd([0.6, 0.62], loss=[0.71, 0.69])
        assert out["final_adversary_loss"] == pytest.approx(0.69)
        assert out["n_loss_rounds_unusable"] == 0
        assert [m for m in messages if "usable adversary LOSS" in m] == []


class TestF10R6ACollapsedPredictorIsNotDebiasingConverged:
    """The single-class TARGET is refused because it yields a constant score; a
    BINARY target whose predictor collapses to a constant score anyway produced
    the identical "Debiasing converged" sentence with insufficient_data False,
    predictor_fitted True and zero warnings. The guard was keyed on
    ``len(y_classes) < 2``, a property of the INPUT, while the quantity that
    decides it is whether the predictor OUTPUT is constant.
    """

    N = 400

    @staticmethod
    def _fixture(kind, seed=0):
        rng = np.random.default_rng(seed)
        n = 400
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        sens = (x2 > 0).astype(int)
        if kind == "signal":
            return np.column_stack([x1, x2]), ((x1 + x2) > 0).astype(int), sens
        if kind == "noise":
            return np.column_stack([x1, x2]), (rng.random(n) > 0.5).astype(int), sens
        # No signal at all: the predictor cannot do better than a constant.
        return (
            np.column_stack([np.ones(n), np.full(n, 2.0)]),
            (rng.random(n) > 0.5).astype(int),
            sens,
        )

    def _run(self, kind):
        X, y, sens = self._fixture(kind)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sklearn_adversarial_debiasing(X, y, sens, n_rounds=5)
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            verdict = adversarial_convergence_diagnostics(
                result["adversary_loss_history"],
                result["adversary_acc_history"],
                result["predictor_loss_history"],
                result["majority_share"],
                result["degenerate_constant_predictions"],
            )
        return (
            result,
            [str(w.message) for w in caught],
            verdict,
            [str(w.message) for w in caught2],
        )

    def test_a_constant_output_from_a_binary_target_is_disclosed(self):
        result, messages, _, _ = self._run("flat")

        assert len(set(result["predictions"])) == 1, "fixture premise: the model collapsed"
        assert result["predictor_fitted"] is True, (
            "a predictor WAS fitted, which is why predictor_fitted could not report this"
        )
        assert result["degenerate_constant_predictions"] is True
        assert result["n_prediction_classes"] == 1
        assert result["insufficient_data"] is True, (
            "before: insufficient_data False and not_assessed_reason '' for a model that "
            "made no decision at all"
        )
        assert "CONSTANT" in result["not_assessed_reason"].upper()
        assert messages
        # The adversary accuracy WAS measured and must survive: it is the claim
        # built on it that is withdrawn, not the measurement.
        assert all(math.isfinite(a) for a in result["adversary_acc_history"])

    def test_the_verdict_sentence_is_withdrawn_not_only_the_dict_field(self):
        """A correct dict field beside a false published sentence is still a live
        defect, because the sentence is what a person reads."""
        _, _, verdict, messages = self._run("flat")

        assert "Debiasing converged" not in verdict["verdict"], (
            "the guard's own forbidden sentence, reproduced with a binary target. Got: "
            + verdict["verdict"]
        )
        assert "no longer recoverable" not in verdict["verdict"]
        assert verdict["classification"] == "not_assessed"
        assert verdict["adversary_skill"] is None
        assert verdict["degenerate_constant_predictions"] is True
        assert messages

    def test_an_unreported_collapse_is_visible_as_could_not_check(self):
        """THREE STATES on the convergence surface: True / False / not reported."""
        out, _ = _acd([0.52] * 5, share=0.52)
        assert out["degenerate_constant_predictions"] is None, (
            "None means the caller never answered the question, which is not the same "
            "as answering 'it did not collapse'"
        )

    @pytest.mark.parametrize("kind", ["signal", "noise"])
    def test_control_a_deciding_model_is_silent_and_gets_a_real_verdict(self, kind):
        """OVER-CORRECTION CONTROL in both directions: a real finding and a real
        all-clear must both still be reachable."""
        result, messages, verdict, messages2 = self._run(kind)

        assert result["n_prediction_classes"] == 2
        assert result["degenerate_constant_predictions"] is False
        assert result["insufficient_data"] is False
        assert result["not_assessed_reason"] == ""
        assert messages == []
        assert verdict["classification"] == "diverged"
        assert verdict["adversary_skill"] > 0.3
        assert "Adversary still recovers" in verdict["verdict"]
        assert messages2 == []

    def test_control_the_graded_non_binary_refusal_still_raises(self):
        rng = np.random.default_rng(0)
        n = 120
        X = rng.normal(size=(n, 2))
        with pytest.raises(ValueError, match="3 classes"):
            sklearn_adversarial_debiasing(
                X, rng.integers(0, 3, size=n), (X[:, 1] > 0).astype(int), n_rounds=3
            )

    def test_control_the_graded_single_target_class_refusal_is_not_doubled(self):
        rng = np.random.default_rng(0)
        n = 120
        X = rng.normal(size=(n, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sklearn_adversarial_debiasing(
                X, np.zeros(n, dtype=int), (X[:, 1] > 0).astype(int), n_rounds=3
            )
        assert result["insufficient_data"] is True
        assert "single class" in result["not_assessed_reason"]
        assert result["not_assessed_reason"].count("CONSTANT") == 0, (
            "the single-class branch already says this; reporting it twice would be "
            "noise. Got " + result["not_assessed_reason"]
        )
        assert caught
