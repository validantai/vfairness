"""Beta Go-Live Stage 1, group g01: three states in the in-processing constraints.

Five critical findings, all rooted in
``src/vfairness/in_processing/constraints/base.py``, all reproduced at the
PUBLIC entry before the fix:

    1. ``FairnessTrainingAnalyzer.evaluate_baseline`` on a single-group
       sensitive attribute returned ``fairness_violation=0.0``,
       ``constraint_satisfied=True``, ``constraint_evaluated=True``.
    2. ``DemographicParityConstraint.is_satisfied`` returned ``np.True_`` on
       single-group data -- the strongest possible two-state answer -- and a
       flat ``False`` for the siblings that DID record could-not-evaluate.
    3. ``DemographicParityConstraint.compute_violation`` returned
       ``overall_violation=0.0``, ``is_satisfied=True``,
       ``could_not_evaluate=False`` and no ``insufficient_data`` flag: byte
       identical to a genuine two-group measured perfect parity.
    4. ``BoundedGroupLossConstraint.compute_violation`` returned
       ``overall_violation=0``, ``is_satisfied=True`` for a single group whose
       classifier was wrong on EVERY row (the bound is relative, so the
       predicate is the tautology ``L <= (1 + tol) * L``), and clamped an
       undefined group loss to a compliant 0 because ``max(0, nan - bound)``
       is 0 in Python.
    5. ``ExponentiatedGradient.fit`` issued a compliance certificate --
       ``final_violation=0.0``, ``converged=True``,
       ``constraint_satisfied=True``, ``insufficient_data=False``, zero
       warnings -- for a fit in which no pair of groups existed to compare.

Every class below carries a CONTROL: healthy multi-group data must still
measure, and measure the same numbers as before. A fix that makes everything
refuse is a worse defect than the one it replaces.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vfairness.in_processing import ExponentiatedGradient, FairnessTrainingAnalyzer
from vfairness.in_processing.constraints import (
    BoundedGroupLossConstraint,
    DemographicParityConstraint,
    FalsePositiveRateParityConstraint,
)


def _is_nan(value) -> bool:
    """NaN is the only value that is not equal to itself."""
    return value != value


# ---------------------------------------------------------------------------
# Shared fixtures-as-functions (module-local on purpose: twelve other agents
# are editing this checkout, so nothing shared is touched).
# ---------------------------------------------------------------------------


def _single_group(n: int = 30):
    """Predictions, labels and a sensitive attribute with ONE unique value."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, n), rng.integers(0, 2, n), np.array(["a"] * n)


def _two_groups_breach(n: int = 30):
    """Two groups, a measurable and large demographic-parity gap.

    Group 'a' is predicted positive on every row, group 'b' on none, so the
    spread is exactly 1.0 and nothing about it is a sampling accident.
    """
    sens = np.array(["a", "b"] * (n // 2))
    y_pred = np.where(sens == "a", 1, 0)
    return y_pred, y_pred, sens


def _two_groups_parity(n: int = 40):
    """Two groups with an IDENTICAL positive rate: a measured 0.0, not a
    vacuous one. This is the control that must stay distinguishable from the
    refusal, and before the fix it was byte-identical to it."""
    sens = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    half = n // 2
    block = np.array([1, 1, 0, 0] * (half // 4))
    y_pred = np.concatenate([block, block])
    return y_pred, y_pred, sens


# ---------------------------------------------------------------------------
# Finding 3 (root cause): DemographicParityConstraint.compute_violation
# ---------------------------------------------------------------------------


class TestDemographicParityComputeViolationThreeStates:
    def test_single_group_is_not_measured(self):
        """BEFORE: 0.0 / True / False, no insufficient_data key, no warning."""
        y_pred, y_true, sens = _single_group()
        v = DemographicParityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert _is_nan(v.overall_violation), (
            f"a one-element max-minus-min is not a measurement; got {v.overall_violation!r}"
        )
        assert v.could_not_evaluate is True
        assert v.details["insufficient_data"] is True
        # Fail closed: never a PASS for something nobody measured.
        assert bool(v.is_satisfied) is False

    def test_the_dict_a_report_reads_carries_the_third_state(self):
        """A consumer that only ever sees the dict (JSON, report payload) must
        still be able to tell a measured breach from an unevaluable one."""
        y_pred, y_true, sens = _single_group()
        d = (
            DemographicParityConstraint(tolerance=0.05)
            .compute_violation(y_pred, y_true, sens)
            .to_dict()
        )

        assert d["could_not_evaluate"] is True
        assert _is_nan(d["overall_violation"])
        assert d["details"]["insufficient_data"] is True

    def test_a_measured_zero_is_distinguishable_from_a_vacuous_one(self):
        """CONTROL, and the point of the whole finding. Two groups with an
        identical positive rate is a MEASURED perfect parity; before the fix it
        returned exactly the same three fields as the single-group case."""
        y_pred, y_true, sens = _two_groups_parity()
        v = DemographicParityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert v.overall_violation == pytest.approx(0.0)
        assert bool(v.is_satisfied) is True
        assert v.could_not_evaluate is False
        assert v.details["insufficient_data"] is False

    def test_a_measured_breach_still_measures(self):
        """CONTROL: the healthy path must not move a single number."""
        y_pred, y_true, sens = _two_groups_breach()
        v = DemographicParityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert v.overall_violation == pytest.approx(1.0)
        assert bool(v.is_satisfied) is False
        assert v.could_not_evaluate is False


# ---------------------------------------------------------------------------
# Finding 2: DemographicParityConstraint.is_satisfied (the inherited method)
# ---------------------------------------------------------------------------


class TestConstraintIsSatisfiedThreeStates:
    def test_single_group_is_none_not_true(self):
        """BEFORE: np.True_ -- a could-not-check collapsed into a PASS."""
        y_pred, y_true, sens = _single_group()
        assert (
            DemographicParityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, sens) is None
        )

    def test_a_sibling_that_records_the_flag_is_also_none_not_false(self):
        """The other half of the same defect: this method dropped the third
        state even for the constraints that DID record it, answering a flat
        False -- a measured breach -- for an unmeasurable constraint.
        BEFORE: False."""
        y_pred, y_true, sens = _single_group()
        assert (
            FalsePositiveRateParityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, sens)
            is None
        )

    def test_measured_pass_and_measured_breach_are_still_true_and_false(self):
        """CONTROL. None must be reserved for not-measured; the two measured
        verdicts keep their exact booleans."""
        y_pred, y_true, sens = _two_groups_parity()
        assert (
            DemographicParityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, sens) is True
        )

        y_pred, y_true, sens = _two_groups_breach()
        assert (
            DemographicParityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, sens) is False
        )


# ---------------------------------------------------------------------------
# Finding 4: BoundedGroupLossConstraint.compute_violation
# ---------------------------------------------------------------------------


def _nan_loss_for_group_b(sens):
    """A loss function that is undefined (NaN) for group 'b'.

    Keyed on the slice LENGTH, which is how the group slice arrives at
    ``loss_fn``; the whole-array call (len == len(sens)) stays measured, so
    ``overall_loss`` is defined and the refusal can only come from the group.
    """
    n_b = int(np.sum(sens == "b"))

    def loss_fn(y_pred, y_true):
        if len(y_true) == n_b:
            return float("nan")
        return float(np.mean(y_pred != y_true))

    return loss_fn


class TestBoundedGroupLossThreeStates:
    def test_single_group_cannot_be_violated_so_it_must_refuse(self):
        """BEFORE: overall_violation=0, is_satisfied=True, could_not_evaluate
        False -- for 60 rows, one group, a classifier wrong on EVERY row. The
        bound is relative, (1 + tol) * overall_loss, so with one group the
        predicate is the tautology L <= 1.1 * L."""
        rng = np.random.default_rng(0)
        n = 60
        y = rng.integers(0, 2, n)
        v = BoundedGroupLossConstraint(tolerance=0.1).compute_violation(
            1 - y, y, np.array(["a"] * n)
        )

        # The classifier really is wrong on every row, so this is not a quiet
        # data accident: the refusal is structural.
        assert v.details["group_losses"]["a"] == pytest.approx(1.0)
        assert _is_nan(v.overall_violation)
        assert v.could_not_evaluate is True
        assert v.details["insufficient_data"] is True
        assert bool(v.is_satisfied) is False

    def test_an_undefined_group_loss_is_not_a_compliant_group(self):
        """BEFORE: group_violations={'a': 0, 'b': 0} and overall 0/True,
        because ``max(0, nan - bound)`` returns 0 in Python (builtin max keeps
        its FIRST argument when the comparison is False, and every comparison
        against NaN is False)."""
        rng = np.random.default_rng(0)
        n = 60
        y = rng.integers(0, 2, n)
        sens = np.array(["a"] * 50 + ["b"] * 10)
        v = BoundedGroupLossConstraint(
            tolerance=0.1, loss_fn=_nan_loss_for_group_b(sens)
        ).compute_violation(1 - y, y, sens)

        assert _is_nan(v.details["group_losses"]["b"]), "fixture must make b undefined"
        assert _is_nan(v.group_violations["b"]), "an unmeasurable group is not compliant"
        assert _is_nan(v.overall_violation)
        assert v.could_not_evaluate is True
        assert bool(v.is_satisfied) is False

    def test_two_groups_one_genuinely_worse_still_measures(self):
        """CONTROL: the exact number recorded before the fix."""
        rng = np.random.default_rng(0)
        n = 60
        y = rng.integers(0, 2, n)
        y_pred = np.concatenate([y[:30], 1 - y[30:]])
        sens = np.array(["a"] * 30 + ["b"] * 30)
        v = BoundedGroupLossConstraint(tolerance=0.1).compute_violation(y_pred, y, sens)

        assert v.overall_violation == pytest.approx(0.44999999999999996)
        assert bool(v.is_satisfied) is False
        assert v.could_not_evaluate is False
        assert v.details["insufficient_data"] is False
        assert v.group_violations["a"] == pytest.approx(0.0)

    def test_a_measured_satisfied_bound_is_still_satisfied(self):
        """CONTROL on the other side: two groups with equal loss must PASS, so
        the fail-closed direction has not swallowed the pass state."""
        rng = np.random.default_rng(1)
        n = 60
        y = rng.integers(0, 2, n)
        y_pred = y.copy()
        y_pred[::10] = 1 - y_pred[::10]  # same error rate in both halves
        sens = np.array(["a"] * 30 + ["b"] * 30)
        v = BoundedGroupLossConstraint(tolerance=0.1).compute_violation(y_pred, y, sens)

        assert not _is_nan(v.overall_violation)
        assert bool(v.is_satisfied) is True
        assert v.could_not_evaluate is False


# ---------------------------------------------------------------------------
# Finding 1: FairnessTrainingAnalyzer.evaluate_baseline
# ---------------------------------------------------------------------------


class TestFairnessTrainingAnalyzerBaselineThreeStates:
    def test_single_group_baseline_reports_not_evaluated(self):
        """BEFORE: fairness_violation=0.0, constraint_satisfied=True,
        constraint_evaluated=True, no warning."""
        rng = np.random.default_rng(0)
        n = 30
        X = rng.normal(size=(n, 3))
        y = rng.integers(0, 2, n)
        res = FairnessTrainingAnalyzer(
            X=X,
            y=y,
            sensitive_attr=np.array(["a"] * n),
            fairness_constraint="demographic_parity",
            tolerance=0.05,
        ).evaluate_baseline(y_pred=y)

        assert res.fairness_violation is None
        assert res.constraint_satisfied is None
        assert res.parameters["constraint_evaluated"] is False

        d = res.to_dict()
        assert d["fairness_violation"] is None
        assert d["constraint_satisfied"] is None
        assert d["parameters"]["constraint_evaluated"] is False
        # The accuracy beside it was always measurable and must survive.
        assert d["accuracy"] == pytest.approx(1.0)

    def test_two_group_baseline_still_reports_a_measurement(self):
        """CONTROL: a real breach must still arrive as numbers, not None."""
        rng = np.random.default_rng(0)
        n = 40
        X = rng.normal(size=(n, 3))
        sens = np.array(["a", "b"] * (n // 2))
        y = np.where(sens == "a", 1, 0)
        res = FairnessTrainingAnalyzer(
            X=X,
            y=y,
            sensitive_attr=sens,
            fairness_constraint="demographic_parity",
            tolerance=0.05,
        ).evaluate_baseline(y_pred=y)

        assert res.fairness_violation == pytest.approx(1.0)
        assert res.constraint_satisfied is False
        assert res.parameters["constraint_evaluated"] is True


# ---------------------------------------------------------------------------
# Finding 5: ExponentiatedGradient.fit
# ---------------------------------------------------------------------------


class TestExponentiatedGradientThreeStates:
    def test_single_group_fit_issues_no_compliance_certificate(self):
        """BEFORE: final_violation=0.0, converged=True,
        constraint_satisfied=True, insufficient_data=False, zero warnings."""
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ExponentiatedGradient(
                LogisticRegression(max_iter=500),
                constraint="demographic_parity",
                epsilon=0.05,
                max_iterations=10,
            ).fit(X, y, sensitive_attr=np.array(["A"] * n))

        assert _is_nan(res.final_violation)
        assert bool(res.optimization_result.converged) is False
        assert bool(res.fairness_metrics["constraint_satisfied"]) is False
        assert res.fairness_metrics["insufficient_data"] is True
        assert res.fairness_metrics["n_iterations_unmeasurable"] == 10
        assert any("NOT selected for fairness" in str(c.message) for c in caught), (
            f"a fit that could not measure fairness must say so; got {[str(c.message) for c in caught]}"
        )

    def test_two_group_fit_still_measures_and_can_converge(self):
        """CONTROL: on healthy two-group data the reduction must still report a
        real number and a real verdict, and must not warn about unmeasurable
        iterates."""
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, 4))
        sens = np.array(["A", "B"] * (n // 2))
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ExponentiatedGradient(
                LogisticRegression(max_iter=500),
                constraint="demographic_parity",
                epsilon=0.05,
                max_iterations=10,
            ).fit(X, y, sensitive_attr=sens)

        assert not _is_nan(res.final_violation)
        assert res.final_violation >= 0.0
        assert res.fairness_metrics["insufficient_data"] is False
        assert res.fairness_metrics["n_iterations_unmeasurable"] == 0
        assert isinstance(bool(res.fairness_metrics["constraint_satisfied"]), bool)
        assert not any("NOT selected for fairness" in str(c.message) for c in caught)
