"""G10 grading pins: in_processing.constraints (base records + reductions).

CONSTRAINTS ARE A MITIGATION, WHICH INVERTS THE FAILURE MODE. A neutered
mitigation reports SUCCESS, not failure: a constraint that no longer binds
produces a BETTER fairness number, because approving everybody is trivially
equal. So nothing here is judged by the fairness number it reports; each pin asks
whether the constraint still binds and whether the rule still separates anybody.

One defect pinned:

D7 ``ThresholdOptimizer`` PUBLISHED A MEASURED PASS FOR A RULE THAT SEPARATES
   NOBODY. A rate-based constraint is satisfied exactly when the decision column
   is constant, so accepting everybody (or rejecting everybody) gives a true 0.0
   violation that grades nothing. The class's own docstring quotes that feedback
   for an all-NaN score column, and a CONSTANT FINITE score column walked past it
   because the unscored-row count only sees non-finite values. Worse, measured on
   a real disparity (group a scored 0.60-0.95, group b 0.05-0.40) the coordinate
   search reached perfect demographic parity by REJECTING EVERY APPLICANT and
   reported final_violation_ 0.0, constraint_satisfied_ True,
   insufficient_data_ False and zero warnings.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vfairness.exceptions import ConfigurationError
from vfairness.in_processing.constraints.base import (
    BaseFairnessConstraint,
    BoundedGroupLossConstraint,
    ConstraintViolation,
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    OptimizationResult,
)
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
    LagrangianState,
    ReductionResult,
    ThresholdOptimizer,
)

N = 200


def _two_groups():
    return np.array(["a"] * (N // 2) + ["b"] * (N // 2))


def _labels(seed: int = 0):
    return np.random.default_rng(seed).integers(0, 2, N)


def _fit(y_prob, y_true, sens, tolerance=0.05, grid_size=25):
    optimizer = ThresholdOptimizer(
        constraint=DemographicParityConstraint(tolerance=tolerance), grid_size=grid_size
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.fit(y_prob, y_true, sensitive_attr=sens)
    return optimizer, [str(w.message) for w in caught]


class _Stub:
    """An ensemble member that always answers the same label."""

    def __init__(self, label):
        self.label = label
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        return np.full(len(X), self.label)


# ------------------------------------------------- D7: a rule that separates nobody


@pytest.mark.parametrize("constant", [0.7, 0.0, 1.0, 0.5, 0.49])
def test_a_constant_score_column_is_disclosed_as_a_vacuous_pass(constant):
    """D7. Every threshold gives the same partition, so the search could not rank
    one candidate above another on fairness and the 0.0 grades nothing."""
    sens, y_true = _two_groups(), _labels()
    optimizer, messages = _fit(np.full(N, constant), y_true, sens)

    assert optimizer.degenerate_constant_predictions_ is True
    assert np.unique(optimizer.predict(np.full(N, constant), sens)).size == 1
    assert any("VACUOUS rather than good" in m for m in messages), messages
    assert any("separates nobody" in m for m in messages)
    # The number is kept, because it is a real measurement of the rule chosen.
    assert optimizer.final_violation_ == pytest.approx(0.0)
    # And the could-not-check state is NOT borrowed for this: the constraint
    # genuinely is met, vacuously, which is a different thing from unmeasured.
    assert optimizer.insufficient_data_ is False
    assert optimizer.constraint_satisfied_ is True


def test_a_search_that_reaches_parity_by_rejecting_everybody_is_disclosed():
    """D7's sharpest shape: a REAL disparity, mitigated to a perfect 0.0 by a rule
    that turns nobody down differently because it turns everybody down."""
    sens = _two_groups()
    # DETERMINISTIC and fully separated: group "a" scores in [0.60, 0.95] and
    # group "b" in [0.05, 0.40], with no overlap. Every threshold in the middle
    # of the grid accepts all of "a" and none of "b" (a disparity of 1.0), so
    # the ONLY feasible joint assignments are the ones that reject everybody.
    half = N // 2
    y_prob = np.concatenate([np.linspace(0.60, 0.95, half), np.linspace(0.05, 0.40, half)])
    # tolerance=0.0 asks for EXACT parity, which on a fully separated frame only
    # the two collapsed assignments can give, so the reproduction is deterministic
    # rather than dependent on a seed. (Measured the same day with the default
    # 0.05 tolerance on random draws from the same two ranges: the search
    # collapsed to reject-all there too, with final_violation_ 0.0,
    # constraint_satisfied_ True, insufficient_data_ False and ZERO warnings. That
    # run is not pinned because at 0.05 the grid can also stop one row short of
    # the collapse, and a pin must not depend on which.)
    y_true = _labels()
    optimizer, messages = _fit(y_prob, y_true, sens, tolerance=0.0)

    assert optimizer.get_thresholds()["a"] > max(y_prob), "the search moved above every score"
    predictions = optimizer.predict(y_prob, sens)
    assert predictions.mean() == 0.0, "the rule rejects every applicant"
    assert optimizer.degenerate_constant_predictions_ is True
    assert optimizer.final_violation_ == pytest.approx(0.0)
    assert any("VACUOUS rather than good" in m for m in messages), messages
    assert any("REJECTS every row" in m or "ACCEPTS every row" in m for m in messages)


def test_control_a_real_mitigation_is_not_flagged_as_degenerate():
    """CONTROL for D7. A flag that fires on every fit would pass every pin above
    and destroy the class."""
    sens = _two_groups()
    rng = np.random.default_rng(3)
    y_prob = np.clip(rng.normal(np.where(sens == "a", 0.65, 0.45), 0.15, N), 0.01, 0.99)
    y_true = (y_prob + rng.normal(0, 0.2, N) > 0.55).astype(int)
    optimizer, messages = _fit(y_prob, y_true, sens)

    predictions = optimizer.predict(y_prob, sens)
    assert 0.0 < predictions.mean() < 1.0, "both decisions must survive"
    assert optimizer.degenerate_constant_predictions_ is False
    assert not any("VACUOUS" in m for m in messages), messages
    assert optimizer.constraint_satisfied_ is True
    assert math.isfinite(optimizer.final_violation_)


def test_the_degeneracy_flag_is_separate_from_the_could_not_check_flag():
    """Two different states, and collapsing them would lose what each says."""
    sens, y_true = _two_groups(), _labels()
    unscored, _ = _fit(np.full(N, np.nan), y_true, sens)
    assert unscored.insufficient_data_ is True
    assert unscored.constraint_satisfied_ is None, "never False for something unmeasured"
    assert unscored.n_unscored_rows_ == N

    single, _ = _fit(np.random.default_rng(0).uniform(size=N), y_true, np.array(["a"] * N))
    assert single.insufficient_data_ is True
    assert single.constraint_satisfied_ is None
    assert not math.isfinite(single.final_violation_)


def test_the_flags_are_none_before_fit_and_get_thresholds_refuses():
    optimizer = ThresholdOptimizer(constraint=DemographicParityConstraint())
    assert optimizer.degenerate_constant_predictions_ is None
    assert optimizer.constraint_satisfied_ is None and optimizer.insufficient_data_ is None
    assert optimizer.final_violation_ is None and optimizer.n_unscored_rows_ is None
    with pytest.raises(RuntimeError, match="not fitted"):
        optimizer.get_thresholds()
    with pytest.raises(RuntimeError, match="not fitted"):
        optimizer.predict(np.array([0.5]), np.array(["a"]))


def test_predict_refuses_an_unscored_row_and_an_unseen_group():
    sens = _two_groups()
    rng = np.random.default_rng(1)
    optimizer, _ = _fit(rng.uniform(0.1, 0.9, N), _labels(), sens)
    with pytest.raises(ValueError):
        optimizer.predict(np.array([0.5, np.nan]), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="no fitted threshold for group"):
        optimizer.predict(np.array([0.5]), np.array(["c"]))


# ------------------------------------------------------- the bound must still bind


@pytest.mark.parametrize(
    "tolerance", [float("nan"), float("inf"), -float("inf"), -0.1, True, False, "0.05", None]
)
def test_a_tolerance_that_is_not_a_bound_is_refused_above_the_subclass_selection(tolerance):
    """A BOUND THAT CANNOT BE BREACHED SILENCES THE OPTIMISER. Every comparison
    against NaN is False, so an unknowable bound read as 'no violation'; this is
    checked once, above the dispatch, so sabotaging it reddens every subclass."""
    for cls in (
        DemographicParityConstraint,
        EqualizedOddsConstraint,
        EqualOpportunity := __import__(
            "vfairness.in_processing.constraints.base", fromlist=["x"]
        ).EqualOpportunityConstraint,
        BoundedGroupLossConstraint,
    ):
        with pytest.raises(ConfigurationError):
            cls(tolerance=tolerance)
    assert EqualOpportunity is not None


def test_control_a_real_tolerance_including_zero_is_accepted():
    for tolerance in (0.0, 0.001, 0.05, 0.5, 1.0, 10.0):
        assert DemographicParityConstraint(tolerance=tolerance).tolerance == tolerance


def test_a_vacuous_relative_bound_is_refused_by_range_not_by_finiteness():
    """The rule is a property of the bound AND the metric's range together. The
    0-1 loss lies in [0, 1], so a bound at or above 1.0 cannot be exceeded."""
    y_true = np.array([0] * 30 + [1] * 30)
    y_pred = np.array([1] * 30 + [0] * 30)  # wrong on every row of group "a"
    sens = np.array(["a"] * 30 + ["b"] * 30)
    y_pred = np.where(sens == "a", 1 - y_true, y_true)

    measured = BoundedGroupLossConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
    assert measured.overall_violation > 0 and measured.is_satisfied is False

    with pytest.warns(UserWarning):
        vacuous = BoundedGroupLossConstraint(tolerance=60.0).compute_violation(y_pred, y_true, sens)
    assert vacuous.could_not_evaluate is True, "a bound of 24.4 on a loss capped at 1.0"
    assert vacuous.is_satisfied is False, "fail-closed, and could_not_evaluate tells them apart"


def test_the_bound_is_only_refused_against_the_reachable_range():
    """DO NOT OVER-CORRECT. On the fixture the module's own docstring names,
    group "a" at a 0-1 loss of 0.8 and group "b" at 0.0, the overall loss is 0.4
    and tolerance=1.0 puts the bound at 0.80: a real boundary result that group
    "a" meets exactly, and a group at 0.9 would still breach. Vacuity is decided
    against the loss's REACHABLE range, not against the size of the tolerance."""
    y_true = np.zeros(60, dtype=int)
    sens = np.array(["a"] * 30 + ["b"] * 30)
    y_pred = np.zeros(60, dtype=int)
    y_pred[:24] = 1  # group "a" wrong on 24 of 30 rows -> loss 0.8; "b" -> 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = BoundedGroupLossConstraint(tolerance=1.0).compute_violation(y_pred, y_true, sens)
    assert result.could_not_evaluate is False, "a bound of 0.80 is not vacuous on [0, 1]"
    assert result.details["group_losses"]["a"] == pytest.approx(0.8)
    assert result.details["group_losses"]["b"] == pytest.approx(0.0)


def test_a_bound_type_this_class_does_not_implement_is_refused_not_relabelled():
    with pytest.raises(NotImplementedError, match="does not implement the 'ratio' bound"):
        DemographicParityConstraint(bound_type="ratio")
    with pytest.raises(ConfigurationError):
        DemographicParityConstraint(bound_type="nonsense")


def test_the_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        BaseFairnessConstraint()


def test_is_satisfied_has_three_states_at_the_public_entry():
    y_true = _labels(5)
    y_pred = y_true.copy()
    sens = _two_groups()
    constraint = DemographicParityConstraint(tolerance=0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert constraint.is_satisfied(y_pred, y_true, np.array(["a"] * N)) is None
    biased = np.where(sens == "a", 1, 0)
    assert constraint.is_satisfied(biased, y_true, sens) is False
    assert constraint.is_satisfied(np.zeros(N, dtype=int), y_true, sens) is True


# ---------------------------------------------------------------------- the records


def test_constraint_violation_reads_three_states_and_carries_them_into_the_dict():
    not_measured = ConstraintViolation(
        "demographic_parity", float("nan"), {}, False, 0.05, {"insufficient_data": True}
    )
    breach = ConstraintViolation("demographic_parity", 0.31, {"a": 0.31}, False, 0.05, {})
    passing = ConstraintViolation("demographic_parity", 0.01, {"a": 0.01}, True, 0.05, {})

    assert not_measured.could_not_evaluate is True
    assert breach.could_not_evaluate is False and passing.could_not_evaluate is False
    for violation in (not_measured, breach, passing):
        assert violation.to_dict()["could_not_evaluate"] is violation.could_not_evaluate
        assert set(violation.to_dict()) == set(violation.__dataclass_fields__) | {
            "could_not_evaluate"
        }, "a field added later must not be dropped by a hand-listed rebuild"


def test_the_flag_is_read_without_a_get_default_that_a_present_none_defeats():
    """``.get(key, default)`` returns the STORED value, not the default, when the
    key is present holding None, and None would then read as measured."""
    with_none = ConstraintViolation("x", float("nan"), {}, False, 0.05, {"insufficient_data": None})
    assert with_none.could_not_evaluate is True, "the NaN fallback must answer it"
    finite_with_none = ConstraintViolation("x", 0.2, {}, False, 0.05, {"insufficient_data": None})
    assert finite_with_none.could_not_evaluate is False
    assert ConstraintViolation("x", None, {}, False, 0.05, {}).could_not_evaluate is True


def test_optimization_result_is_a_record_and_its_dict_carries_every_field():
    result = OptimizationResult(converged=True, n_iterations=3, final_violation=0.02, best_gap=0.01)
    payload = result.to_dict()
    assert set(payload) == set(result.__dataclass_fields__)
    assert payload["final_weights"] is None
    assert json.loads(json.dumps(payload))["final_violation"] == 0.02
    weighted = OptimizationResult(True, 1, 0.0, 0.0, final_weights=np.array([0.25, 0.75]))
    assert weighted.to_dict()["final_weights"] == [0.25, 0.75]
    # Defaults are per-instance, so one result's history cannot leak into another.
    OptimizationResult(True, 1, 0.0, 0.0).loss_history.append(1.0)
    assert OptimizationResult(True, 1, 0.0, 0.0).loss_history == []


def test_lagrangian_state_records_what_it_is_given_and_nothing_else():
    state = LagrangianState(
        lambda_={"a": 0.5, "b": -0.25},
        classifier_weights=[0.3, 0.7],
        best_classifier_idx=1,
        iteration=3,
    )
    assert state.lambda_ == {"a": 0.5, "b": -0.25}
    assert state.classifier_weights == [0.3, 0.7]
    assert state.best_classifier_idx == 1 and state.iteration == 3
    assert not hasattr(state, "to_dict"), "no serialiser here, so none can drop a field"
    # Every field is required: none of them can be silently defaulted to a
    # neutral value by a caller who forgot it.
    with pytest.raises(TypeError):
        LagrangianState(lambda_={}, classifier_weights=[], best_classifier_idx=0)


# ------------------------------------------------------------- ensemble degeneracy


def test_an_ensemble_that_would_decide_nothing_is_refused():
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match="no ensemble member carries a weight above"):
        EnsembleClassifier([_Stub(0), _Stub(1)], np.array([0.0, 0.0])).predict(X)
    with pytest.raises(ValueError, match="below the 0.5 decision threshold"):
        EnsembleClassifier([_Stub(1)], np.array([0.2])).predict(X)


def test_a_vote_on_labels_outside_the_zero_one_domain_is_refused():
    """With no 0 in the label set the sum can never fall below the threshold, so
    every row is predicted 1 whatever the features say."""

    class OffDomain:
        classes_ = np.array([1, 2])

        def predict(self, X):
            return np.full(len(X), 2)

    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match=r"outside the \[0, 1\] label domain"):
        EnsembleClassifier([OffDomain()], np.array([1.0])).predict(X)


def test_control_an_honest_ensemble_still_votes():
    X = np.random.default_rng(0).normal(size=(20, 2))
    predictions = EnsembleClassifier([_Stub(0), _Stub(1)], np.array([0.25, 0.75])).predict(X)
    assert np.unique(predictions).tolist() == [1], "0.25*0 + 0.75*1 = 0.75 >= 0.5"
    flipped = EnsembleClassifier([_Stub(0), _Stub(1)], np.array([0.75, 0.25])).predict(X)
    assert np.unique(flipped).tolist() == [0], "the vote must still depend on the weights"


def test_a_score_column_of_hard_values_is_disclosed_as_not_a_probability():
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.warns(UserWarning, match="exactly 0.0 or 1.0"):
        probabilities = EnsembleClassifier([_Stub(0)], np.array([1.0])).predict_proba(X)
    assert np.unique(probabilities).tolist() == [0.0]


# ------------------------------------------------------------------ a real fit


def test_an_unevaluable_fit_issues_no_compliance_certificate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 3))
    y = (X[:, 0] > 0).astype(int)
    gradient = ExponentiatedGradient(
        base_estimator=LogisticRegression(max_iter=200),
        constraint=DemographicParityConstraint(tolerance=0.05),
        max_iterations=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gradient.fit(X, y, sensitive_attr=np.array(["a"] * N))
    result = gradient.get_result()
    assert isinstance(result, ReductionResult)
    assert not math.isfinite(result.final_violation)
    assert result.optimization_result.converged is False
    assert result.fairness_metrics["insufficient_data"] is True
    assert set(result.to_dict()) == {
        "n_classifiers",
        "weights",
        "optimization_result",
        "final_violation",
        "accuracy",
        "fairness_metrics",
    }


def test_control_a_two_group_fit_still_measures_a_real_violation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 3))
    y = (X[:, 0] > 0).astype(int)
    sens = np.where(X[:, 1] > 0, "a", "b")
    gradient = ExponentiatedGradient(
        base_estimator=LogisticRegression(max_iter=200),
        constraint=DemographicParityConstraint(tolerance=0.05),
        max_iterations=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gradient.fit(X, y, sensitive_attr=sens)
    result = gradient.get_result()
    assert math.isfinite(result.final_violation) and result.final_violation > 0
    assert result.fairness_metrics["insufficient_data"] is False
    assert result.fairness_metrics["degenerate_constant_predictions"] is False
