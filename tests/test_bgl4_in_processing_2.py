"""BGL4 audit of batch A-in_processing-2: the overturns, as executable evidence.

Each test below asserts the CORRECT behaviour. Every one was written by an auditor
as ``xfail(strict=True)``, recording a defect that was live at the time, and every
one now PASSES: BGL5 (2026-09-27) fixed all six, so the markers were removed and
the assertions kept exactly as the auditor wrote them. The measured before/after
for each sits in the test's own docstring, taken verbatim from the auditor's
``reason`` and extended with the reading after the fix.

The subjects are unchanged. Where a fix is also pinned against regression with a
sabotage record and an over-correction control, that lives in
``tests/test_bgl5_in_processing_2.py``.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pytest

from vfairness.in_processing.constraints.base import BoundedGroupLossConstraint
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
)

torch = pytest.importorskip("torch", reason="the adversarial losses need PyTorch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
)


def _messages(caught: List[warnings.WarningMessage]) -> List[str]:
    return [str(w.message) for w in caught]


def _bgl_fixture():
    """The BGL3 fixture: wrong on 24 of the 30 rows of a, right on every row of b."""
    groups = np.array(["a"] * 30 + ["b"] * 30)
    y_true = np.ones(60, dtype=int)
    y_pred = np.concatenate([np.array([0.1] * 24 + [0.9] * 6), np.full(30, 0.9)])
    return y_pred, y_true, groups


# ===========================================================================
# 1. BoundedGroupLossConstraint: the unlabelled-dataset certificate survives
#    for an OBJECT dtype label column (grades 0 and 1)
# ===========================================================================


def test_an_object_dtype_label_column_of_nan_is_not_a_compliance_certificate():
    """BGL4 overturn, CLOSED by BGL5: _unlabelled_rows returned 0 for any dtype
    whose kind is not 'f' or 'c', so an object-dtype column of np.nan (what pandas
    hands over for a mixed or all-missing column) counted 0 unlabelled rows.

    BEFORE: is_satisfied=True, overall_violation=0.0, n_unlabelled_rows=0,
    could_not_evaluate=False, warnings=[], on the same predictions that are a
    measured 0.36 breach with an int label column.

    AFTER: is_satisfied=False, overall_violation=0.0 (the magnitude is left as
    computed, by the file's own _warn_unlabelled policy), n_unlabelled_rows=60,
    could_not_evaluate=True, 1 warning naming '60 of 60 row(s) have no usable
    label'. The count is taken from the VALUES now, by
    base._rows_with_no_value."""
    y_pred, _y_true, groups = _bgl_fixture()
    constraint = BoundedGroupLossConstraint(tolerance=0.1)
    unlabelled = np.array([np.nan] * 60, dtype=object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = constraint.compute_violation(y_pred, unlabelled, groups)

    assert v.could_not_evaluate is True, (
        f"object-dtype NaN labels read as a MEASURED verdict: "
        f"is_satisfied={v.is_satisfied!r}, overall_violation={v.overall_violation!r}, "
        f"n_unlabelled_rows={v.details['n_unlabelled_rows']!r}, "
        f"warnings={_messages(caught)!r}"
    )
    assert v.is_satisfied is False
    assert v.details["n_unlabelled_rows"] == 60


def test_the_signed_value_warns_for_object_dtype_labels_too():
    """BGL4 overturn, CLOSED by BGL5, the sibling of the row above:
    signed_constraint_value routes through compute_violation, so with an
    object-dtype label column it answered 0.0 (no violation) for the group the
    model is wrong about 80% of the time, and no warning reached this caller
    either.

    BEFORE: signed_constraint_value(y_pred, object-NaN labels, groups, 'a') = 0.0
    with warnings [].

    AFTER: still 0.0, deliberately (a NaN would poison the exponentiated-gradient
    multiplier), now with the 'no usable label' warning beside it and the
    ConstraintViolation it derives from carrying could_not_evaluate True."""
    y_pred, _y_true, groups = _bgl_fixture()
    constraint = BoundedGroupLossConstraint(tolerance=0.1)
    unlabelled = np.array([np.nan] * 60, dtype=object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        signed = constraint.signed_constraint_value(y_pred, unlabelled, groups, "a")

    assert any("no usable label" in m for m in _messages(caught)), (
        f"signed_constraint_value(..., 'a') = {signed!r} in silence over a dataset "
        f"with no usable label in it"
    )


# ===========================================================================
# 2. EnsembleClassifier.predict_proba: the collapsed-member disclosure is keyed
#    on the TYPE, so a FOREIGN constant member reproduces the defect (grade 6)
# ===========================================================================


def _constant_foreign_member():
    """A majority-class DummyClassifier: predict_proba is exactly 0.0/1.0 per row."""
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = np.array([0] * 24 + [1] * 6)
    return DummyClassifier(strategy="most_frequent").fit(X, y), X


def test_a_foreign_constant_member_is_disclosed_by_predict_proba():
    """BGL4 overturn, CLOSED by BGL5: _warn_about_vote_fractions tested
    ``isinstance(clf, _ConstantClassifier)``, a TYPE, instead of the VALUES the
    member returns.

    BEFORE: EnsembleClassifier([_ConstantClassifier(0)], [1.0]).predict_proba
    -> [0.] WITH the warning, while the same call with
    DummyClassifier(strategy='most_frequent') -> [0.] with ZERO warnings.

    AFTER: both -> [0.] with a warning. The new disclosure,
    reductions._warn_about_a_score_column_of_hard_values, reads the returned
    column, so it cannot be defeated by a member class it has never heard of."""
    member, X = _constant_foreign_member()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = EnsembleClassifier([member], np.array([1.0])).predict_proba(X)

    assert np.unique(out).tolist() == [0.0]  # the degenerate value, for the record
    assert any("HARD" in m or "collapsed" in m for m in _messages(caught)), (
        "a probability of exactly 0.0 on every row came back from predict_proba "
        "with no disclosure at all"
    )


def test_exponentiated_gradient_predict_proba_discloses_a_collapsed_fit():
    """BGL4 overturn, CLOSED by BGL5: ExponentiatedGradient.predict_proba itself
    never disclosed a collapsed foreign base estimator. The fit-time warning was
    the only channel, so a caller holding the fitted object read exactly 0.0 for
    every row as a probability. The two tests named as evidence for this unit
    execute 0 of its 4 body lines (measured under coverage).

    BEFORE: np.unique(eg.predict_proba(X)) == [0.] with 0 warnings at the call.

    AFTER: the same values with 2 warnings at the call, one from this method
    naming fairness_metrics['degenerate_constant_predictions'] and one from the
    delegate naming the hard 0/1 column."""
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(0)
    n = 120
    A = np.array(["a"] * 60 + ["b"] * 60)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + (A == "a") * 1.5 > 0).astype(int)

    eg = ExponentiatedGradient(
        DummyClassifier(strategy="most_frequent"), "demographic_parity", max_iterations=3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg.fit(X, y, sensitive_attr=A)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = eg.predict_proba(X)

    assert np.unique(proba).size == 1  # the degenerate value, for the record
    assert _messages(caught), (
        "predict_proba returned a single constant pseudo-probability for every row "
        "and said nothing at the call"
    )


# ===========================================================================
# 3. EnsembleClassifier.predict: the threshold guard covers only the all-zeros
#    side of the vacuity (grade 5)
# ===========================================================================


class _FixedLabel:
    """A member whose predict returns one label, outside the [0, 1] vote domain."""

    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, X):
        return np.full(len(X), self.label)


def test_a_vote_that_can_never_fall_below_the_threshold_is_refused_too():
    """BGL4 overturn, CLOSED by BGL5:
    _refuse_a_vote_that_cannot_reach_the_threshold reads the WEIGHTS only, and its
    premise is 'a weighted vote of labels in [0, 1]'. With members whose labels are
    {1, 2} (or any set with no 0 in it) the sum can never fall below 0.5, so predict
    returns 1 for every row for EVERY possible X and every member output.

    BEFORE: weights [0.5, 0.5] over two members that both answer 2 -> [1 1 1 1 1 1],
    zero warnings, the mirror image of the all-zeros array the weight guard was
    written to refuse.

    AFTER: ValueError from reductions._refuse_a_vote_the_label_domain_decides,
    naming the observed label 2.0 and the [0, 1] domain the vote is defined on, with
    no warning emitted."""
    X = np.arange(12).reshape(6, 2).astype(float)
    ensemble = EnsembleClassifier([_FixedLabel(2), _FixedLabel(2)], np.array([0.5, 0.5]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            ensemble.predict(X)
    assert not _messages(caught)


# ===========================================================================
# 4. get_adversary_accuracy: a column-vector attribute broadcasts (grade 10)
# ===========================================================================


def test_a_column_vector_attribute_is_not_scored_as_chance_leakage():
    """BGL4 overturn, CLOSED by BGL5: the accuracy is
    ``(pred_labels == sensitive_attr).float().mean()``, and pred_labels is (n,). An
    (n, 1) attribute, which this class accepts everywhere else, broadcasts to an
    (n, n) comparison matrix.

    BEFORE, on a fully recovered attribute: 1.0 with a flat attribute and exactly
    0.5, chance, with the same values as a column vector, zero warnings. 0.5 reads
    as 'no leakage detected' on this scale.

    AFTER: 1.0 both ways, because the comparison is made row against row on
    reshape(-1) of each side, and a row COUNT that does not line up returns NaN
    with a warning instead of the mean of a broadcast."""
    n = 40
    flat = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    y_pred = torch.where(flat == 1.0, 0.9, 0.1)

    torch.manual_seed(0)
    loss_fn = AdversarialDebiasingLoss()
    for _ in range(50):
        loss_fn.update_adversary(y_pred, flat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        as_flat = loss_fn.get_adversary_accuracy(y_pred, flat)
        as_column = loss_fn.get_adversary_accuracy(y_pred, flat.reshape(-1, 1))

    assert as_flat == pytest.approx(1.0), "the fixture must be a measurable leak"
    assert as_column == pytest.approx(as_flat) or _messages(caught), (
        f"the same attribute as a column vector scored {as_column!r} against "
        f"{as_flat!r}, in silence"
    )
