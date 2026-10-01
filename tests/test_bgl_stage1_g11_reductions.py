"""BGL Stage 1, group g11: constraint_grid_search must not certify a fairness
constraint it could not evaluate.

Finding (critical, reproduced at the public API):
``GridSearch.fit`` never checked that a fairness COMPARISON was possible for the
data it was handed. It dispatched to ``constraint.compute_violation`` and
reported whatever came back. On a sensitive attribute with a SINGLE level,
``DemographicParityConstraint`` returned ``max(rates) - min(rates)`` over a
one-element list -- 0.0 by arithmetic vacuity, not by comparison -- and this is
what a caller read back::

    final_violation=0.0, fairness_metrics={'constraint_satisfied': True,
    'insufficient_data': False, 'n_candidates_unmeasurable': 0}, converged=True,
    and no warning

which is byte-identical to a genuine two-group measured PASS. GridSearch's own
three-state channel (``insufficient_data``, which fires on a NaN) could never
trip, because no NaN ever arrived.

The fix under test lives in ``GridSearch.fit`` itself
(src/vfairness/in_processing/constraints/reductions.py): a guard ABOVE the
per-candidate dispatch. With fewer than two groups there is no pair to compare,
so every candidate's violation is recorded as NaN and the existing
insufficient_data / n_candidates_unmeasurable channel reports could-not-check.
It sits above the dispatch on purpose: ``fit`` accepts ANY
``BaseFairnessConstraint``, including a caller's own subclass, so fixing one
constraint only moves the fabrication to the sibling nobody has fixed yet.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.in_processing import GridSearch
from vfairness.in_processing.constraints.base import (
    ConstraintViolation,
    DemographicParityConstraint,
)


class _ShippedDemographicParity(DemographicParityConstraint):
    """``DemographicParityConstraint.compute_violation`` as it shipped.

    Reproduces the released arithmetic verbatim (``max(rates) - min(rates)``,
    no ``insufficient_data`` in ``details``) so this file pins GridSearch's OWN
    guard rather than a sibling module's. ``GridSearch(constraint=...)`` accepts
    any ``BaseFairnessConstraint``, so this is a supported public input, and it
    is the exact object the defect was reported against.
    """

    def compute_violation(self, y_pred, y_true, sensitive_attr):
        y_pred = np.asarray(y_pred)
        sensitive_attr = np.asarray(sensitive_attr)
        y_bin = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
        gm = GroupManager(sensitive_attr)
        overall = float(np.mean(y_bin))
        rates = {g: float(np.mean(y_bin[gm.get_mask(g)])) for g in gm.groups}
        max_violation = max(rates.values()) - min(rates.values())
        return ConstraintViolation(
            constraint_type="demographic_parity",
            overall_violation=max_violation,
            group_violations={g: r - overall for g, r in rates.items()},
            is_satisfied=bool(max_violation <= self.tolerance),
            tolerance=self.tolerance,
            details={"group_rates": rates, "overall_rate": overall},
        )


def _one_group_data(n=800, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 1).astype(int)
    return X, y, np.array(["A"] * n)


def _two_group_biased_data(n=800, seed=11):
    """Two groups with a large, real positive-rate gap."""
    rng = np.random.default_rng(seed)
    in_a = rng.random(n) < 0.5
    X = np.column_stack([rng.normal(size=n) + 3.0 * in_a, rng.normal(size=n)])
    y = ((X[:, 0] + rng.normal(scale=0.3, size=n)) > 1).astype(int)
    return X, y, np.where(in_a, "A", "B")


def _fit(constraint, X, y, s, n_lambda_values=5):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = GridSearch(
            LogisticRegression(max_iter=300),
            constraint,
            n_lambda_values=n_lambda_values,
        ).fit(X, y, sensitive_attr=s)
    return result, [str(w.message) for w in caught]


def _assert_could_not_check(result, messages):
    """The THIRD state: not a pass, not a measured breach."""
    assert np.isnan(result.final_violation), (
        f"a fairness violation nobody could measure was reported as {result.final_violation!r}"
    )
    assert np.isnan(result.optimization_result.final_violation)
    assert result.fairness_metrics["insufficient_data"] is True
    assert result.fairness_metrics["constraint_satisfied"] is False
    assert result.fairness_metrics["n_candidates_unmeasurable"] == 5
    assert result.optimization_result.converged is False, (
        "a sweep that never measured the constraint cannot have converged on it"
    )
    assert any("1 group(s)" in m and "compares groups" in m for m in messages), (
        f"the refusal must name its reason; warnings were {messages}"
    )


# ---------------------------------------------------------------------------
# THREE STATES at the public entry point
# ---------------------------------------------------------------------------


def test_one_group_is_could_not_check_not_a_pass():
    """GridSearch.fit on a single-level sensitive attribute."""
    X, y, s = _one_group_data()
    result, messages = _fit(DemographicParityConstraint(tolerance=0.05), X, y, s)
    _assert_could_not_check(result, messages)


def test_the_refusal_is_gridsearchs_own_and_not_the_constraints():
    """Guard ABOVE the dispatch.

    Handed the constraint exactly as it shipped -- which answers 0.0 /
    is_satisfied=True for one group and carries no insufficient_data flag --
    GridSearch must still refuse. Without its own guard it forwards that
    fabricated PASS: final_violation=0.0, constraint_satisfied=True,
    insufficient_data=False, converged=True, no warning.
    """
    X, y, s = _one_group_data()
    fabricating = _ShippedDemographicParity(tolerance=0.05)

    # The premise: this constraint really does fabricate a satisfied 0.0 here.
    raw = fabricating.compute_violation(np.zeros(len(y), dtype=int), y, s)
    assert raw.overall_violation == 0.0 and raw.is_satisfied is True

    result, messages = _fit(fabricating, X, y, s)
    _assert_could_not_check(result, messages)

    # A fail STATE, not a crash: the model is still fitted and still predicts,
    # it just carries no fairness verdict.
    assert len(result.predict(X)) == len(y)


# ---------------------------------------------------------------------------
# CONTROL: healthy data still measures, and BOTH verdicts stay reachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance, satisfied", [(0.9, True), (0.05, False)])
def test_control_two_groups_still_get_a_real_measurement(tolerance, satisfied):
    X, y, s = _two_group_biased_data()
    result, messages = _fit(DemographicParityConstraint(tolerance=tolerance), X, y, s)

    assert np.isfinite(result.final_violation)
    assert result.final_violation > 0.0
    assert result.fairness_metrics["insufficient_data"] is False
    assert result.fairness_metrics["n_candidates_unmeasurable"] == 0
    assert result.fairness_metrics["constraint_satisfied"] is satisfied
    assert result.optimization_result.converged is satisfied
    assert not any("compares groups" in m for m in messages), (
        f"the guard fired on measurable two-group data: {messages}"
    )

    # The reported number is the constraint's own measurement on the
    # predictions the returned model actually makes.
    measured = DemographicParityConstraint(tolerance=tolerance).compute_violation(
        result.predict(X), y, s
    )
    assert result.final_violation == pytest.approx(measured.overall_violation, abs=1e-12)


def test_control_guard_does_not_fire_for_a_custom_constraint_with_two_groups():
    """The guard keys on the DATA, not on the constraint class: a caller's own
    constraint is measured as usual once a pair of groups exists."""
    X, y, s = _two_group_biased_data()
    result, messages = _fit(_ShippedDemographicParity(tolerance=0.05), X, y, s)

    assert np.isfinite(result.final_violation)
    assert result.fairness_metrics["insufficient_data"] is False
    assert not any("compares groups" in m for m in messages)
