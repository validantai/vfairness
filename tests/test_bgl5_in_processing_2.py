"""BGL5 pins for the six BGL4 overturns in batch A-in_processing-2.

Every defect closed here was an auditor's ``xfail(strict=True)`` in
``tests/test_bgl4_in_processing_2.py``, reproduced by execution before anything
was changed. That file keeps the auditor's own subjects and now passes; this file
adds what a pin needs and a demonstration does not:

* the mechanism's WHOLE class of input, not only the one value that was measured
  (an object column of ``None`` as well as of ``np.nan``, a member that declares
  a label domain of {1, 2} while returning nothing but 1, a foreign constant
  member reached through three different surfaces);
* the OVER-CORRECTION CONTROL beside each one, asserting the real number a
  healthy input still gets, worked out here rather than read back from the code;
* the disclosure at the surface a caller reads (``to_dict``, the warning list).

Three mechanisms recur, and each one was a guard keyed on the wrong thing:

1. ``base._unlabelled_rows`` / ``_unscored_rows`` asked a numpy DTYPE whether a
   value was missing, so an object column of NaN answered "nothing missing".
2. ``reductions._warn_about_vote_fractions`` asked ``isinstance(clf,
   _ConstantClassifier)``, a TYPE, so our own collapsed member was disclosed and
   sklearn's ``DummyClassifier`` was not.
3. ``get_adversary_accuracy`` compared a (n,) prediction against an (n, 1)
   attribute, and the silent broadcast scored a fully recovered attribute as
   exactly chance.

Each pin below was SABOTAGED (the fix reverted in an isolated copy of ``src``,
never in the shared checkout) and confirmed red; the sabotage output is recorded
in /tmp/claude-501/bgl/fix/fix-A-in_processing-2.json.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from vfairness.in_processing.constraints.base import (
    BoundedGroupLossConstraint,
    OptimizationResult,
    _rows_with_no_value,
)
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
    ReductionResult,
    _ConstantClassifier,
)

torch = pytest.importorskip("torch", reason="the adversarial losses need PyTorch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
)


def _messages(caught: List[warnings.WarningMessage]) -> List[str]:
    return [str(w.message) for w in caught]


def _bgl_fixture():
    """The BGL3 fixture: wrong on 24 of the 30 rows of a, right on every row of b.

    With the labels intact this is a MEASURED 0.36 bounded-group-loss breach
    (group losses 0.8 and 0.0, tolerance 0.1), which is what makes every silent
    0.0 below a fabrication rather than a boring input.
    """
    groups = np.array(["a"] * 30 + ["b"] * 30)
    y_true = np.ones(60, dtype=int)
    y_pred = np.concatenate([np.array([0.1] * 24 + [0.9] * 6), np.full(30, 0.9)])
    return y_pred, y_true, groups


_X6 = np.arange(12).reshape(6, 2).astype(float)


# ===========================================================================
# 1. base: the missing-value count is taken from the VALUES, not the dtype
#    (BoundedGroupLossConstraint.compute_violation / signed_constraint_value)
# ===========================================================================


class TestAnObjectColumnOfNothingIsNotAMeasurement:
    @pytest.mark.parametrize("blank", [np.nan, None], ids=["nan", "None"])
    def test_an_object_label_column_of_blanks_withdraws_the_verdict(self, blank):
        """MEASURED BEFORE on both blanks, same 60 predictions in every row::

            int labels            -> violation 0.36, is_satisfied False,
                                     could_not_evaluate False
            object([nan] * 60)    -> violation 0.0,  is_satisfied TRUE,
                                     could_not_evaluate False, n_unlabelled 0,
                                     warnings []
            object([None] * 60)   -> identical

        AFTER: violation 0.0 (the magnitude is deliberately left as computed),
        is_satisfied False, could_not_evaluate True, n_unlabelled_rows 60 and one
        warning naming '60 of 60 row(s) have no usable label'.

        ``_unlabelled_rows`` opened with ``if arr.dtype.kind not in "fc":
        return 0``, so the whole unlabelled-dataset refusal was keyed on a numpy
        dtype, and an object column is what pandas hands over for any mixed or
        all-missing column.
        """
        y_pred, _y_true, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)
        blanks = np.array([blank] * 60, dtype=object)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, blanks, groups)

        assert v.details["n_unlabelled_rows"] == 60
        assert v.could_not_evaluate is True
        assert v.is_satisfied is False
        assert v.details["insufficient_data"] is True
        assert any("60 of 60 row(s) have no usable label" in m for m in _messages(caught))
        # The disclosure has to survive the boundary a consumer reads.
        assert v.to_dict()["could_not_evaluate"] is True
        assert v.to_dict()["is_satisfied"] is False
        # And the verdict channel above it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert constraint.is_satisfied(y_pred, blanks, groups) is None

    def test_the_signed_value_carries_the_warning_for_an_object_column(self):
        """MEASURED BEFORE: ``signed_constraint_value(..., 'a')`` over an object
        column of NaN returned 0.0, no push and no violation, for the group the
        model is wrong about 80% of the time, with warnings [].

        AFTER: still exactly 0.0 (a NaN here would poison the
        exponentiated-gradient multiplier, which is why the magnitude is not the
        fix) and the unlabelled-row warning now reaches this caller.
        """
        y_pred, _y_true, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)
        blanks = np.array([np.nan] * 60, dtype=object)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signed = constraint.signed_constraint_value(y_pred, blanks, groups, "a")

        assert signed == pytest.approx(0.0)
        assert any("have no usable label" in m for m in _messages(caught)), _messages(caught)

    def test_an_object_prediction_column_of_nan_is_not_a_measured_pass(self):
        """The same hole on the PREDICTION side, measured with the labels intact::

            BEFORE: n_unscored_rows 0,  is_satisfied True,  warnings []
            AFTER:  n_unscored_rows 60, is_satisfied False, 1 warning

        An unscored row binarises to a confident rejection through
        ``y_pred >= 0.5``, so this is the same fabricated certificate reached from
        the other argument.
        """
        _y_pred, y_true, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)
        blanks = np.array([np.nan] * 60, dtype=object)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(blanks, y_true, groups)

        assert v.details["n_unscored_rows"] == 60
        assert v.could_not_evaluate is True
        assert v.is_satisfied is False
        assert any("no usable score" in m for m in _messages(caught))

    def test_control_the_real_breach_is_still_measured_in_silence(self):
        """OVER-CORRECTION CONTROL, with the number stated rather than read back:
        the model is wrong on 24 of 30 rows of a (loss 0.8) and right on all of b
        (loss 0.0), the overall loss is 0.4 and the relative bound is 1.1 * 0.4 =
        0.44, so the violation is 0.8 - 0.44 = 0.36. That number must survive the
        fix exactly, with could_not_evaluate False and no warning.
        """
        y_pred, y_true, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, y_true, groups)
            signed = constraint.signed_constraint_value(y_pred, y_true, groups, "a")

        assert _messages(caught) == []
        assert float(v.overall_violation) == pytest.approx(0.36, abs=1e-9)
        assert signed == pytest.approx(0.36, abs=1e-9)
        assert v.could_not_evaluate is False
        assert v.details["n_unlabelled_rows"] == 0
        assert v.details["n_unscored_rows"] == 0

    def test_control_an_object_column_of_real_values_counts_nothing_missing(self):
        """OVER-CORRECTION CONTROL for the mechanism itself, and the pinned
        behaviour of ``test_control_string_labels_are_not_counted_as_unlabelled``:
        a non-numeric column holding REAL values has nothing missing in it.

        Measured: an object column of 'yes'/'no' over the same two groups
        measures a violation of 0.225 with 0 unlabelled rows, 0 unscored rows and
        no warning, before and after. ``"yes"`` is a value; ``np.nan`` is not.
        """
        groups = np.array(["a"] * 30 + ["b"] * 30)
        y_true = np.array(["yes"] * 60, dtype=object)
        y_pred = np.array(["yes"] * 45 + ["no"] * 15, dtype=object)
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, y_true, groups)

        assert _messages(caught) == []
        assert v.details["n_unlabelled_rows"] == 0
        assert v.details["n_unscored_rows"] == 0
        assert float(v.overall_violation) == pytest.approx(0.225, abs=1e-9)
        assert v.could_not_evaluate is False

    def test_control_the_counter_itself_reads_values_not_dtypes(self):
        """The helper, directly, so the pin names the mechanism and not only its
        consequence. Measured for each input, before -> after:

            float64 [nan, 1.0]        1 -> 1      (unchanged)
            object  [nan, 1.0]        0 -> 1
            object  [None, 1]         0 -> 1
            object  ['yes', 'no']     0 -> 0      (must not move)
            int64   [0, 1]            0 -> 0      (nothing to count)
            bool    [True, False]     0 -> 0
            pandas Int64 with pd.NA   1 -> 1      (pandas casts to float64)
        """
        import pandas as pd

        assert _rows_with_no_value(np.array([np.nan, 1.0])) == 1
        assert _rows_with_no_value(np.array([np.nan, 1.0], dtype=object)) == 1
        assert _rows_with_no_value(np.array([None, 1], dtype=object)) == 1
        assert _rows_with_no_value(np.array(["yes", "no"], dtype=object)) == 0
        assert _rows_with_no_value(np.array(["yes", "no"])) == 0
        assert _rows_with_no_value(np.array([0, 1])) == 0
        assert _rows_with_no_value(np.array([True, False])) == 0
        assert _rows_with_no_value(np.asarray(pd.Series([1, pd.NA], dtype="Int64"))) == 1
        # pd.NA in an OBJECT column: it answers `!=` with another NA rather than
        # with a bool, so a plain `v != v` test raises instead of counting it.
        assert _rows_with_no_value(np.array([pd.NA, 1], dtype=object)) == 1
        assert (
            _rows_with_no_value(np.array([np.datetime64("NaT"), np.datetime64("2026-01-01")])) == 1
        )


# ===========================================================================
# 2. reductions: predict_proba discloses the VALUES it returned, whatever the
#    member's class is (EnsembleClassifier / ReductionResult / EG.predict_proba)
# ===========================================================================


def _reduction_result(members, weights):
    return ReductionResult(
        classifiers=list(members),
        weights=np.asarray(weights, dtype=float),
        optimization_result=OptimizationResult(
            converged=True, n_iterations=1, final_violation=0.0, best_gap=0.0
        ),
        final_violation=0.0,
        accuracy=1.0,
    )


def _foreign_constant_member():
    """DummyClassifier(strategy='most_frequent'): predict_proba is exactly 0/1.

    Not a ``_ConstantClassifier``, so the old isinstance-keyed disclosure could
    not see it, and it is the very estimator the collapsed-fit fixture uses.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = np.array([0] * 24 + [1] * 6)
    return DummyClassifier(strategy="most_frequent").fit(X, y), X


class TestAScoreColumnOfHardValuesSaysSo:
    @pytest.mark.parametrize(
        "build",
        [
            lambda m, w: EnsembleClassifier(m, np.asarray(w, dtype=float)),
            _reduction_result,
        ],
        ids=["EnsembleClassifier", "ReductionResult"],
    )
    def test_a_foreign_constant_member_is_disclosed(self, build):
        """MEASURED BEFORE, on 30 rows, both surfaces::

            [_ConstantClassifier(0)]                      -> [0.]  1 warning
            [DummyClassifier('most_frequent')]            -> [0.]  0 warnings

        Byte-identical columns, and the disclosure depended on which class had
        produced them, because it asked ``isinstance(clf, _ConstantClassifier)``.
        AFTER: both -> [0.] with a warning naming the exactly-0/1 column.
        """
        member, X = _foreign_constant_member()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = build([member], [1.0]).predict_proba(X)

        assert np.unique(out).tolist() == [0.0]
        assert any("cannot rank one row above another" in m for m in _messages(caught)), _messages(
            caught
        )

    def test_our_own_constant_member_is_still_disclosed(self):
        """The type-keyed warning it used to rely on is kept beside the new
        value-keyed one, so this member is named twice for two different reasons:
        what it IS (a collapsed constant classifier) and what it RETURNED (a
        column pinned to the 0/1 endpoints). Measured: 2 warnings, unique [0.].

        The second assertion matches 'cannot rank one row above another' and NOT
        'exactly 0.0 or 1.0', because the OLD type-keyed message contains that
        phrase too: the first version of this test stayed GREEN under the S2
        sabotage, which is a test that cannot disagree rather than a passing one.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = EnsembleClassifier([_ConstantClassifier(0)], np.array([1.0])).predict_proba(_X6)

        messages = _messages(caught)
        assert np.unique(out).tolist() == [0.0]
        assert any("collapsed constant classifiers" in m for m in messages), messages
        assert any("cannot rank one row above another" in m for m in messages), messages

    def test_the_fitted_estimator_discloses_a_collapsed_fit_at_predict_proba(self):
        """MEASURED BEFORE, 120 rows, DummyClassifier('most_frequent') as the base
        estimator, the fit-time warning suppressed to stand for a caller who was
        not there for it: ``np.unique(eg.predict_proba(X)) == [0.]`` with ZERO
        warnings at the call. The disclosure existed only in the fit warning and
        in fairness_metrics.

        AFTER: the same values with two warnings at the call, one naming
        fairness_metrics['degenerate_constant_predictions'] and one naming the
        hard 0/1 column.
        """
        eg, X = _collapsed_fit()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = eg.predict_proba(X)

        messages = _messages(caught)
        assert np.unique(proba).size == 1
        assert any("predicts a single class for every row" in m for m in messages), messages
        assert eg.get_result().fairness_metrics["degenerate_constant_predictions"] is True

    def test_a_collapsed_fit_whose_values_look_ordinary_is_disclosed_too(self):
        """THE VALUE TEST ALONE WOULD BE A GUARD THAT ONLY WORKS FOR THE FIXTURE.

        ``DummyClassifier(strategy='prior')`` returns a constant 0.2 rather than
        0.0, which is a perfectly ordinary looking probability, and the fit that
        produced it was just as collapsed. Measured on the same 120 rows: BEFORE 0
        warnings at the call; AFTER 1, from the flag rather than from the values.
        """
        eg, X = _collapsed_fit(strategy="prior")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = eg.predict_proba(X)

        messages = _messages(caught)
        assert np.unique(proba).size == 1
        assert float(np.unique(proba)[0]) not in (0.0, 1.0)
        assert any("predicts a single class for every row" in m for m in messages), messages

    def test_control_a_real_probability_column_is_silent_even_when_constant(self):
        """OVER-CORRECTION CONTROL, and the reason the test is not 'is the column
        constant'. Worked out here: 0.25*0.2 + 0.75*0.8 == 0.65 and
        0.5*0.3 + 0.5*0.7 == 0.5. Both are constant across rows and both are real
        weighted probabilities, so both must stay silent. A guard on constancy
        would have reddened two existing over-correction controls in
        tests/test_bgl3_in_processing_1.py.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            skewed = EnsembleClassifier(
                [_ProbaStub(0.2), _ProbaStub(0.8)], np.array([0.25, 0.75])
            ).predict_proba(_X6)
            even = EnsembleClassifier(
                [_ProbaStub(0.3), _ProbaStub(0.7)], np.array([0.5, 0.5])
            ).predict_proba(_X6)

        assert _messages(caught) == []
        np.testing.assert_allclose(skewed, np.full(6, 0.65))
        np.testing.assert_allclose(even, np.full(6, 0.5))

    def test_control_a_real_fit_still_answers_predict_proba_in_silence(self):
        """OVER-CORRECTION CONTROL through the fitted surface: LogisticRegression
        on the same 120 rows predicts both classes, so nothing about degeneracy
        warns and the column really separates rows (measured: more than 100
        distinct values over 120 rows, strictly inside (0, 1)).
        """
        from sklearn.linear_model import LogisticRegression

        X, y, A = _two_group_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg = ExponentiatedGradient(
                LogisticRegression(max_iter=500), "demographic_parity", max_iterations=5
            )
            eg.fit(X, y, sensitive_attr=A)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = eg.predict_proba(X)

        assert _messages(caught) == []
        assert np.unique(proba).size > 100
        assert 0.0 < float(proba.min()) and float(proba.max()) < 1.0


# ===========================================================================
# 3. reductions: the LABEL DOMAIN side of the vacuous vote
#    (EnsembleClassifier.predict / ReductionResult.predict)
# ===========================================================================


class _FixedLabel:
    """A member whose predict returns one label, outside the [0, 1] vote domain."""

    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, X):
        return np.full(len(X), self.label)


class _ProbaStub:
    """A member with a constant probability, and the hard label that follows."""

    def __init__(self, p: float) -> None:
        self.p = float(p)

    def predict(self, X):
        n = len(X)
        return np.full(n, int(self.p >= 0.5))

    def predict_proba(self, X):
        n = len(X)
        col = np.full(n, self.p)
        return np.column_stack([1.0 - col, col])


class _DeclaresTwo:
    """Fitted on y in {1, 2} and answering 1 here: the case the VALUES cannot see.

    ``classes_`` is how sklearn records the label domain, and this is what
    ``DummyClassifier(strategy='most_frequent').fit(X, y)`` with y in {1, 2}
    looks like from the outside.
    """

    classes_ = np.array([1, 2])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


@pytest.mark.parametrize(
    "build",
    [
        lambda m, w: EnsembleClassifier(m, np.asarray(w, dtype=float)),
        _reduction_result,
    ],
    ids=["EnsembleClassifier", "ReductionResult"],
)
class TestAVoteTheLabelDomainDecides:
    @pytest.mark.parametrize(
        "members",
        [
            [_FixedLabel(2), _FixedLabel(2)],
            [_FixedLabel(1), _FixedLabel(2)],
            [_FixedLabel(-1), _FixedLabel(-1)],
        ],
        ids=["both-2", "1-and-2", "both-minus-1"],
    )
    def test_labels_outside_the_vote_domain_are_refused(self, build, members):
        """MEASURED BEFORE, six rows, weights [0.5, 0.5] which total exactly 1 and
        therefore clear every other guard::

            both members answer 2      -> [1 1 1 1 1 1], 0 warnings
            members answer 1 and 2     -> [1 1 1 1 1 1], 0 warnings

        With no 0 in the label set the weighted sum can never fall BELOW 0.5, so
        ``sum >= 0.5`` is 1 for every row of every possible X: the mirror image of
        the all-zeros array ``_refuse_a_vote_that_cannot_reach_the_threshold``
        exists to refuse, decided before any classifier is consulted. That guard
        reads the WEIGHTS only and never checked its own premise, 'a weighted vote
        of labels in [0, 1]'.

        AFTER: ValueError naming the observed labels. The -1 case is refused for
        the same reason in the other direction: it is not on the vote's scale.
        """
        with pytest.raises(ValueError, match=r"outside the \[0, 1\] label domain"):
            build(members, [0.5, 0.5]).predict(_X6)

    def test_a_member_that_only_declares_a_foreign_domain_is_refused(self, build):
        """The half the returned VALUES cannot see. A member fitted on y in {1, 2}
        that answers 1 for every row of this X is indistinguishable, from its
        output alone, from an honest 0/1 model that says yes to everybody.
        Measured BEFORE: [1 1 1 1 1 1] with 0 warnings. AFTER: ValueError naming
        the declared class 2.0 from ``classes_``.
        """
        with pytest.raises(ValueError, match="classes_"):
            build([_DeclaresTwo(), _DeclaresTwo()], [0.5, 0.5]).predict(_X6)

    def test_control_a_zero_one_ensemble_still_votes_in_silence(self, build):
        """OVER-CORRECTION CONTROL, arithmetic first: members answering 0 and 1
        under weights [0.25, 0.75] give 0.25*0 + 0.75*1 == 0.75, which is at or
        above the threshold, so the label is 1 for every row and nothing is
        refused. This is the case the refusal must NOT touch: an all-ones output
        from labels inside the domain is the model's own measured decision, and
        the all-zeros case with honest weights is not refused either.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = build([_ProbaStub(0.2), _ProbaStub(0.8)], [0.25, 0.75]).predict(_X6)

        assert _messages(caught) == []
        np.testing.assert_array_equal(labels, np.full(6, 1, dtype=int))

    def test_control_an_all_zero_and_an_all_one_zero_one_ensemble_both_answer(self, build):
        """OVER-CORRECTION CONTROL, both endpoints. Members that all answer 0
        (weights summing to 1) give [0]*6 and members that all answer 1 give
        [1]*6, both in silence: a collapsed MODEL is disclosed by
        fairness_metrics['degenerate_constant_predictions'] at fit time, not by
        refusing a prediction call.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zeros = build([_FixedLabel(0), _FixedLabel(0)], [0.5, 0.5]).predict(_X6)
            ones = build([_FixedLabel(1), _FixedLabel(1)], [0.5, 0.5]).predict(_X6)

        assert _messages(caught) == []
        np.testing.assert_array_equal(zeros, np.zeros(6, dtype=int))
        np.testing.assert_array_equal(ones, np.ones(6, dtype=int))

    def test_control_a_fitted_zero_one_member_is_not_refused(self, build):
        """OVER-CORRECTION CONTROL with a real sklearn estimator, so the
        ``classes_`` arm cannot start refusing ordinary fits: a DummyClassifier
        fitted on y in {0, 1} declares classes_ [0, 1] and predicts without a
        word. Measured: all six rows 0 (the majority class), zero warnings.
        """
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 2))
        y = np.array([0] * 24 + [1] * 6)
        member = DummyClassifier(strategy="most_frequent").fit(X, y)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = build([member], [1.0]).predict(_X6)

        assert _messages(caught) == []
        np.testing.assert_array_equal(labels, np.zeros(6, dtype=int))


# ===========================================================================
# 4. adversarial: a broadcast is not an accuracy
#    (AdversarialDebiasingLoss.get_adversary_accuracy)
# ===========================================================================


def _trained_adversary(n=40):
    """A perfectly separable batch, adversary trained 50 steps: a FULL leak."""
    flat = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    y_pred = torch.where(flat == 1.0, 0.9, 0.1)
    torch.manual_seed(0)
    loss_fn = AdversarialDebiasingLoss()
    for _ in range(50):
        loss_fn.update_adversary(y_pred, flat)
    return loss_fn, y_pred, flat


class TestABroadcastIsNotAnAccuracy:
    def test_a_column_vector_attribute_scores_the_leak_it_is(self):
        """MEASURED BEFORE, the SAME values in both calls, on the fixture above::

            flat (40,)    -> 1.0   warnings []
            column (40,1) -> 0.5   warnings []

        0.5 is chance on this scale, which reads as 'no leakage detected', from a
        fully recovered attribute, in silence. The mechanism is
        ``(pred_labels == sensitive_attr)`` with pred_labels (n,) and the
        attribute (n, 1): numpy/torch broadcast that to an (n, n) matrix of every
        prediction against every value, and its mean is the group mix's chance
        rate. Every guard in the chain passes it: integer grid, two distinct
        values, finite, adversary trained.

        AFTER: 1.0 both ways, with no warning, because the comparison is row
        against row.
        """
        loss_fn, y_pred, flat = _trained_adversary()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            as_flat = loss_fn.get_adversary_accuracy(y_pred, flat)
            as_column = loss_fn.get_adversary_accuracy(y_pred, flat.reshape(-1, 1))

        assert _messages(caught) == []
        assert as_flat == pytest.approx(1.0)
        assert as_column == pytest.approx(1.0)

    def test_a_row_count_that_does_not_line_up_is_a_could_not_check(self):
        """The third state, not a quieter version of either other one. 40
        predictions against 20 attribute values cannot be compared row by row, and
        the broadcast mean (which is what the old code returned for any compatible
        shape) is the chance rate of the group mix. Measured AFTER: NaN with a
        warning naming both counts.
        """
        loss_fn, y_pred, flat = _trained_adversary()
        # Both group values are kept in the short attribute, so this reaches the
        # shape check rather than the single-group refusal above it.
        short = torch.cat([flat[:10], flat[-10:]])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc = loss_fn.get_adversary_accuracy(y_pred, short)

        assert acc != acc, "a shape that cannot be compared is not an accuracy"
        assert any("cannot be compared row by row" in m for m in _messages(caught)), _messages(
            caught
        )

    def test_control_a_non_leaking_attribute_still_scores_below_one(self):
        """OVER-CORRECTION CONTROL: the fix must not turn every call into 1.0. An
        attribute that is INDEPENDENT of the predictions (alternating groups over
        the same two prediction values) is measured, not refused, and the number
        is genuinely poor. Measured after the fix: 0.5 for the flat attribute and
        the identical 0.5 for the same values as a column, both silent, which is
        the reading the broadcast used to hand back for a perfect leak.
        """
        n = 40
        alternating = torch.tensor([float(i % 2) for i in range(n)])
        y_pred = torch.where(torch.arange(n) < n // 2, 0.9, 0.1).float()
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss()
        for _ in range(50):
            loss_fn.update_adversary(y_pred, alternating)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            as_flat = loss_fn.get_adversary_accuracy(y_pred, alternating)
            as_column = loss_fn.get_adversary_accuracy(y_pred, alternating.reshape(-1, 1))

        assert _messages(caught) == []
        assert as_flat == pytest.approx(0.5)
        assert as_column == pytest.approx(as_flat)
        assert as_flat < 1.0

    def test_control_the_existing_could_not_checks_still_fire(self):
        """OVER-CORRECTION CONTROL for the guard chain above the comparison, which
        the new shape check must not displace: a constant attribute, a non-integer
        attribute and a NaN in the predictions each still return NaN with their own
        warning rather than reaching the accuracy at all.
        """
        loss_fn, y_pred, flat = _trained_adversary()
        cases = {
            "nothing for the adversary to tell apart": torch.zeros(40),
            "not integer class labels": torch.tensor([0.3, 0.7] * 20),
        }
        for expected, attr in cases.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                acc = loss_fn.get_adversary_accuracy(y_pred, attr)
            assert acc != acc
            assert any(expected in m for m in _messages(caught)), _messages(caught)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc = loss_fn.get_adversary_accuracy(
                y_pred.clone().index_fill_(0, torch.tensor([0]), float("nan")), flat
            )
        assert acc != acc
        assert any("non-finite values" in m for m in _messages(caught)), _messages(caught)


# ===========================================================================
# Shared fixtures for the reductions pins
# ===========================================================================


def _two_group_data(n=120):
    """120 rows, two groups with different base rates, a separable feature."""
    rng = np.random.default_rng(0)
    A = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + (A == "a") * 1.5 > 0).astype(int)
    return X, y, A


def _collapsed_fit(strategy="most_frequent"):
    """A fit whose every iterate is a constant classifier, warnings suppressed.

    Suppressed on purpose: the point of the pin is what a caller who was NOT
    there for the fit can see when they ask the fitted object for a probability.
    """
    X, y, A = _two_group_data()
    eg = ExponentiatedGradient(
        DummyClassifier(strategy=strategy), "demographic_parity", max_iterations=3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg.fit(X, y, sensitive_attr=A)
    return eg, X
