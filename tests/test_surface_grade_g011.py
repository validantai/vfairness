"""Surface grading g011: the in-processing fairness constraints.

Batch g011 covers every ``compute_violation``, ``signed_constraint_value``,
``is_satisfied`` and ``to_dict`` in
``src/vfairness/in_processing/constraints/base.py``.

TWO DEFECTS WERE PROVED BY EXECUTION HERE AND FIXED. Both are the same class:
a value nobody measured, reported in the shape of a measurement.

1.  A DEVIATION FROM THE MEAN OF ONE RATE. The per-group deviation guard
    counted GROUPS, not DEFINED rates. With three groups of which only one has
    any positive labels, the centre IS that group's own rate, so its deviation
    was ``r - r`` = exactly 0.0 while its two siblings were honestly NaN.
    Measured before the fix on the 40-row fixture below::

        EqualOpportunityConstraint(0.05).compute_violation(...)
            .group_violations           -> {'a': 0.0, 'b': nan, 'c': nan}
        EqualOpportunityConstraint(0.05).signed_constraint_value(..., 'a')
                                        -> 0.0, with NO warning at all

    That 0.0 is byte-identical to a measured perfect parity, and
    ``wrappers/sklearn_wrappers.py:285`` spreads ``group_violations`` straight
    into ``FairClassifierResult.fairness_metrics``, where a reader sees
    ``a: 0.0`` sitting beside ``insufficient_data: True``.
    ``FalsePositiveRateParityConstraint`` carried the mirror image, and
    ``EqualizedOddsConstraint`` carried a worse variant: its vacuous 0.0 was
    AVERAGED with the group's real deviation on the other arm, halving a
    measured 0.0333 to 0.0167. A fabricated zero there does not sit beside the
    measurement, it dilutes it.

    ``EqualOpportunityLoss`` already applied this rule the right way round:
    ``if len(tpr_rates) < 2`` over the MEASURABLE rates rather than over the
    groups, at ``in_processing/loss_functions/fairness_losses.py:621``, reason
    "fewer_than_two_groups_with_positive_labels". The siblings disagreed and
    the permissive one was the constraint.

2.  A PREDICTION VECTOR WITH NO SCORES CERTIFIED AS PERFECTLY FAIR. Every
    ``compute_violation`` binarises a float prediction with ``y_pred >= 0.5``,
    and under IEEE 754 that is False for NaN, so an unscored row was written 0,
    a confident REJECT (and ``+inf`` 1, a confident ACCEPT). Measured before the
    fix on 30 rows in two groups whose every prediction was NaN, with warnings
    recorded::

        DemographicParityConstraint(0.05).compute_violation(...)
            -> overall_violation=0.0, is_satisfied=True,
               group_rates={'a': 0.0, 'b': 0.0}, insufficient_data=False
        DemographicParityConstraint(0.05).is_satisfied(...)   -> True

    and the same flat PASS from all five constraints, with zero warnings. The
    rule now applied is the one ``reductions._count_unscored_rows`` already
    used for ``ThresholdOptimizer``; this file was never swept. The MAGNITUDE
    is deliberately left as computed and only the VERDICT is withdrawn, which
    is the same narrow line drawn in ``ThresholdOptimizer.fit``: blanking a
    mostly-real disparity because a handful of rows were unscored would be an
    over-correction.

EVERY CLASS BELOW CARRIES A CONTROL computed independently of the source, from
the fixture's own counts. A fix that makes everything refuse is a worse defect
than the one it replaces.
"""

import warnings

import numpy as np
import pytest

from vfairness.in_processing.constraints.base import (
    BoundedGroupLossConstraint,
    ConstraintViolation,
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FalsePositiveRateParityConstraint,
    OptimizationResult,
)


def _is_nan(value) -> bool:
    """NaN is the only value that is not equal to itself."""
    return value != value


def _run(fn):
    """Call fn with warnings ENABLED and return (result, [warnings])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, list(caught)


# ---------------------------------------------------------------------------
# Fixtures-as-functions, module-local on purpose: this checkout is shared with
# other grading sessions and nothing outside this file is touched.
# ---------------------------------------------------------------------------


def _healthy():
    """40 rows, two groups, a real and hand-countable disparity on every arm.

    group a: 10 positives (9 predicted 1)   -> TPR 0.9
             10 negatives (3 predicted 1)   -> FPR 0.3, 12/20 selected -> 0.60
    group b: 10 positives (5 predicted 1)   -> TPR 0.5
             10 negatives (1 predicted 1)   -> FPR 0.1,  6/20 selected -> 0.30

    So, computed here and not read off the code: TPR gap 0.4, FPR gap 0.2,
    selection-rate gap 0.3, 0-1 losses 4/20 and 6/20.
    """
    y_true = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    sens = np.array(["a"] * 20 + ["b"] * 20)
    y_pred = np.array([1] * 9 + [0] * 1 + [1] * 3 + [0] * 7 + [1] * 5 + [0] * 5 + [1] * 1 + [0] * 9)
    return y_pred, y_true, sens


def _one_measurable_tpr():
    """40 rows, THREE groups, positive labels in group 'a' ONLY.

    TPR_a = 8/10 = 0.8 and there is no second TPR anywhere to compare it with.
    The FPR arm stays fully measurable (0.2 | 0.5 | 0.0), which is what lets
    the equalized-odds dilution be seen separately from a refusal.
    """
    y_true = np.array([1] * 10 + [0] * 10 + [0] * 10 + [0] * 10)
    sens = np.array(["a"] * 20 + ["b"] * 10 + ["c"] * 10)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [0] * 10)
    return y_pred, y_true, sens


def _one_measurable_fpr():
    """The mirror: negative labels in group 'a' ONLY, so FPR_a = 0.2 is the
    only false-positive rate in the data."""
    y_true = np.array([1] * 10 + [0] * 10 + [1] * 10 + [1] * 10)
    sens = np.array(["a"] * 20 + ["b"] * 10 + ["c"] * 10)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [0] * 10)
    return y_pred, y_true, sens


ALL_CONSTRAINTS = [
    ("demographic_parity", DemographicParityConstraint),
    ("equalized_odds", EqualizedOddsConstraint),
    ("equal_opportunity", EqualOpportunityConstraint),
    ("fpr_parity", FalsePositiveRateParityConstraint),
    ("bounded_group_loss", BoundedGroupLossConstraint),
]


# ---------------------------------------------------------------------------
# DEFECT 1: a deviation from the mean of ONE rate
# ---------------------------------------------------------------------------


class TestADeviationNeedsAPeer:
    def test_equal_opportunity_refuses_the_only_measurable_group(self):
        """BEFORE: group_violations == {'a': 0.0, 'b': nan, 'c': nan}."""
        y_pred, y_true, sens = _one_measurable_tpr()
        v = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert v.details["group_tprs"]["a"] == pytest.approx(0.8), "fixture must measure a's TPR"
        assert _is_nan(v.details["group_tprs"]["b"])
        assert _is_nan(v.group_violations["a"]), (
            "a deviation from the mean of ONE rate is not a small deviation, it is "
            f"not a deviation; got {v.group_violations['a']!r}"
        )
        assert _is_nan(v.group_violations["b"])
        assert v.could_not_evaluate is True

    def test_equal_opportunity_signed_value_says_it_is_steering_not_measuring(self):
        """BEFORE: 0.0 with NO warning, identical to a measured perfect parity."""
        y_pred, y_true, sens = _one_measurable_tpr()
        c = EqualOpportunityConstraint(tolerance=0.05)
        value, caught = _run(lambda: c.signed_constraint_value(y_pred, y_true, sens, "a"))

        assert value == 0.0, "0.0 is the documented no-push steering value"
        assert any(
            issubclass(w.category, RuntimeWarning) and "UNMEASURABLE" in str(w.message)
            for w in caught
        ), f"the 0.0 must say it is not a measurement; got {[str(w.message) for w in caught]}"

    def test_fpr_parity_refuses_the_only_measurable_group(self):
        """The mirror image. BEFORE: {'a': 0.0, 'b': nan, 'c': nan}."""
        y_pred, y_true, sens = _one_measurable_fpr()
        c = FalsePositiveRateParityConstraint(tolerance=0.05)
        v = c.compute_violation(y_pred, y_true, sens)

        assert v.details["group_fprs"]["a"] == pytest.approx(0.2), "fixture must measure a's FPR"
        assert _is_nan(v.group_violations["a"])
        assert v.could_not_evaluate is True

        value, caught = _run(lambda: c.signed_constraint_value(y_pred, y_true, sens, "a"))
        assert value == 0.0
        assert any("UNMEASURABLE" in str(w.message) for w in caught)

    def test_equalized_odds_does_not_dilute_a_real_deviation_with_a_vacuous_zero(self):
        """BEFORE: 0.0167, the mean of a's REAL FPR deviation and a vacuous 0.0
        standing in for a TPR arm with nothing to compare.

        Computed here from the fixture: the three FPRs are 0.2, 0.5 and 0.0, so
        the centre is 0.7/3 and a's deviation is |0.2 - 0.7/3| = 1/30.
        """
        y_pred, y_true, sens = _one_measurable_tpr()
        v = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert _is_nan(v.details["tpr_violation"]), "fixture must leave the TPR arm unmeasurable"
        assert v.details["fpr_violation"] == pytest.approx(0.5)
        assert v.group_violations["a"] == pytest.approx(abs(0.2 - 0.7 / 3)), (
            "the only measurable deviation must be reported at its measured size, "
            f"not averaged with a zero nobody measured; got {v.group_violations['a']!r}"
        )
        assert v.could_not_evaluate is True

    def test_equalized_odds_refuses_when_neither_arm_has_a_peer(self):
        """Two rows, one per group: a's TPR is the only TPR and b's FPR is the
        only FPR, so the pair was never compared on any shared arm.
        BEFORE: group_violations == {'a': 0.0, 'b': 0.0}, a perfect-parity
        reading."""
        sens = np.array(["a", "b"])
        v = EqualizedOddsConstraint(tolerance=0.05).compute_violation(
            np.array([1, 1]), np.array([1, 0]), sens
        )

        assert _is_nan(v.group_violations["a"])
        assert _is_nan(v.group_violations["b"])
        assert v.could_not_evaluate is True

    # --- CONTROLS: the healthy path must not move a single number ----------

    def test_control_every_per_group_deviation_is_still_measured_exactly(self):
        y_pred, y_true, sens = _healthy()

        eo = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
        # centre = (0.9 + 0.5) / 2 = 0.7
        assert eo.group_violations["a"] == pytest.approx(0.2)
        assert eo.group_violations["b"] == pytest.approx(-0.2)
        assert eo.overall_violation == pytest.approx(0.4)
        assert eo.could_not_evaluate is False

        fp = FalsePositiveRateParityConstraint(tolerance=0.05).compute_violation(
            y_pred, y_true, sens
        )
        # centre = (0.3 + 0.1) / 2 = 0.2
        assert fp.group_violations["a"] == pytest.approx(0.1)
        assert fp.group_violations["b"] == pytest.approx(-0.1)
        assert fp.overall_violation == pytest.approx(0.2)

        eod = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
        # mean(|0.9 - 0.7|, |0.3 - 0.2|) = mean(0.2, 0.1) = 0.15 for both groups
        assert eod.group_violations["a"] == pytest.approx(0.15)
        assert eod.group_violations["b"] == pytest.approx(0.15)
        assert eod.overall_violation == pytest.approx(0.4)  # max over both arms

        dp = DemographicParityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
        assert dp.overall_violation == pytest.approx(0.3)
        assert dp.group_violations["a"] == pytest.approx(0.15)

        bgl = BoundedGroupLossConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
        # losses 0.2 | 0.3, overall 0.25, bound 1.05 * 0.25 = 0.2625
        assert bgl.overall_violation == pytest.approx(0.3 - 0.2625)
        assert bgl.group_violations["b"] == pytest.approx(0.3 - 0.2625)

    def test_control_the_signed_values_still_carry_size_and_sign(self):
        y_pred, y_true, sens = _healthy()
        for cls, expected_a in [
            (DemographicParityConstraint, 0.15),
            (EqualOpportunityConstraint, 0.2),
            (FalsePositiveRateParityConstraint, 0.1),
            (EqualizedOddsConstraint, 0.2),  # the larger of the TPR/FPR gaps
        ]:
            c = cls(tolerance=0.05)
            value_a, caught_a = _run(
                lambda c=c: c.signed_constraint_value(y_pred, y_true, sens, "a")
            )
            value_b, _ = _run(lambda c=c: c.signed_constraint_value(y_pred, y_true, sens, "b"))
            assert value_a == pytest.approx(expected_a), cls.__name__
            assert value_b == pytest.approx(-expected_a), cls.__name__
            assert not any("UNMEASURABLE" in str(w.message) for w in caught_a), (
                f"{cls.__name__} must not cry could-not-check over measurable data"
            )

    def test_control_a_third_group_with_two_measurable_peers_still_measures(self):
        """The narrow line: one unmeasurable group among THREE must not make
        the other two refuse. Their spread is still a real comparison."""
        # a and b have positives (TPR 0.8 and 0.4); c has none.
        y_true = np.array([1] * 10 + [1] * 10 + [0] * 10)
        sens = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        y_pred = np.array([1] * 8 + [0] * 2 + [1] * 4 + [0] * 6 + [0] * 10)
        v = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

        assert v.details["group_tprs"]["a"] == pytest.approx(0.8)
        assert v.details["group_tprs"]["b"] == pytest.approx(0.4)
        # centre = (0.8 + 0.4) / 2 = 0.6
        assert v.group_violations["a"] == pytest.approx(0.2)
        assert v.group_violations["b"] == pytest.approx(-0.2)
        assert _is_nan(v.group_violations["c"]), "c really has no TPR"
        # The headline still refuses, because c's rate could be anything.
        assert _is_nan(v.overall_violation)


# ---------------------------------------------------------------------------
# DEFECT 2: rows with no usable score
# ---------------------------------------------------------------------------


class TestAPredictionWithNoScoreIsNotAPass:
    @pytest.mark.parametrize("name,cls", ALL_CONSTRAINTS)
    def test_all_nan_predictions_get_no_verdict(self, name, cls):
        """BEFORE: overall_violation=0.0, is_satisfied=True,
        insufficient_data=False and is_satisfied() -> True, from all five."""
        sens = np.array(["a"] * 15 + ["b"] * 15)
        y_true = np.array([1] * 8 + [0] * 7 + [1] * 8 + [0] * 7)
        y_pred = np.full(30, np.nan)

        c = cls(tolerance=0.05)
        v, caught = _run(lambda: c.compute_violation(y_pred, y_true, sens))

        assert v.details["n_unscored_rows"] == 30
        assert v.could_not_evaluate is True, name
        assert bool(v.is_satisfied) is False, f"{name} must never PASS on scores nobody produced"
        assert any("no usable score" in str(w.message) for w in caught), (
            f"{name} gave no warning: {[str(w.message) for w in caught]}"
        )

        verdict, _ = _run(lambda: c.is_satisfied(y_pred, y_true, sens))
        assert verdict is None, f"{name}.is_satisfied returned {verdict!r}"

    def test_a_single_infinite_score_withdraws_the_verdict_but_keeps_the_number(self):
        """+inf lands above every threshold and is written 1, a confident
        ACCEPT. The measured part of the disparity is kept, deliberately, and
        only the verdict is withdrawn."""
        y_pred, y_true, sens = _healthy()
        y_pred = y_pred.astype(float)
        y_pred[0] = np.inf  # already a 1, so the rates do not move at all

        c = DemographicParityConstraint(tolerance=0.05)
        v, caught = _run(lambda: c.compute_violation(y_pred, y_true, sens))

        assert v.details["n_unscored_rows"] == 1
        assert v.overall_violation == pytest.approx(0.3), (
            "the magnitude must survive; blanking it would delete a real finding"
        )
        assert v.could_not_evaluate is True
        assert any("no usable score" in str(w.message) for w in caught)

    @pytest.mark.parametrize("name,cls", ALL_CONSTRAINTS)
    def test_control_finite_scores_report_zero_unscored_rows_and_keep_measuring(self, name, cls):
        y_pred, y_true, sens = _healthy()
        c = cls(tolerance=0.05)
        v, caught = _run(lambda: c.compute_violation(y_pred.astype(float), y_true, sens))

        assert v.details["n_unscored_rows"] == 0, name
        assert v.could_not_evaluate is False, name
        assert not _is_nan(v.overall_violation), name
        assert not any("no usable score" in str(w.message) for w in caught), name

    def test_control_probabilities_are_still_thresholded_the_same_way(self):
        """Floats are the normal case for this API and must not be disturbed:
        0.9/0.6 are accepted, 0.4/0.1 rejected, so a's rate is 1.0 and b's 0.0.
        """
        sens = np.array(["a", "a", "b", "b"])
        y_prob = np.array([0.9, 0.6, 0.4, 0.1])
        y_true = np.array([1, 1, 1, 1])
        c = DemographicParityConstraint(tolerance=0.05)
        v, caught = _run(lambda: c.compute_violation(y_prob, y_true, sens))

        assert v.details["group_rates"]["a"] == pytest.approx(1.0)
        assert v.details["group_rates"]["b"] == pytest.approx(0.0)
        assert v.overall_violation == pytest.approx(1.0)
        assert v.could_not_evaluate is False
        assert caught == []


# ---------------------------------------------------------------------------
# is_satisfied: the inherited three-state verdict
# ---------------------------------------------------------------------------


class TestIsSatisfiedHasThreeStates:
    @pytest.mark.parametrize("name,cls", ALL_CONSTRAINTS)
    def test_a_single_group_gets_no_verdict(self, name, cls):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 30)
        y_pred = rng.integers(0, 2, 30)
        verdict, _ = _run(
            lambda: cls(tolerance=0.05).is_satisfied(y_pred, y_true, np.array(["a"] * 30))
        )
        assert verdict is None, f"{name} answered {verdict!r} where no pair of groups exists"

    def test_control_a_measured_breach_is_false_and_a_measured_pass_is_true(self):
        y_pred, y_true, sens = _healthy()
        assert (
            EqualOpportunityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, sens) is False
        )
        # The same data read with a tolerance wider than the 0.4 gap.
        assert EqualOpportunityConstraint(tolerance=0.5).is_satisfied(y_pred, y_true, sens) is True


# ---------------------------------------------------------------------------
# The two to_dict converters
# ---------------------------------------------------------------------------


class TestTheDictsAReportReads:
    def test_constraint_violation_dict_carries_the_third_state(self):
        y_pred, y_true, sens = _one_measurable_tpr()
        d = (
            EqualOpportunityConstraint(tolerance=0.05)
            .compute_violation(y_pred, y_true, sens)
            .to_dict()
        )

        assert d["could_not_evaluate"] is True
        assert _is_nan(d["overall_violation"])
        assert d["details"]["insufficient_data"] is True
        assert _is_nan(d["group_violations"]["a"])
        # is_satisfied is the fail-closed steering field and must never read as
        # a PASS here; could_not_evaluate is what tells it from a real breach.
        assert d["is_satisfied"] is False

    def test_control_a_measured_dict_carries_the_numbers(self):
        y_pred, y_true, sens = _healthy()
        d = (
            EqualOpportunityConstraint(tolerance=0.05)
            .compute_violation(y_pred, y_true, sens)
            .to_dict()
        )
        assert d["could_not_evaluate"] is False
        assert d["overall_violation"] == pytest.approx(0.4)
        assert d["group_violations"]["a"] == pytest.approx(0.2)
        assert d["details"]["n_unscored_rows"] == 0

    def test_constraint_violation_dict_is_a_copy_of_the_fields_and_invents_nothing(self):
        """Executed check that ``to_dict`` is a converter: hand it values no
        computation produced and they must come back unchanged."""
        v = ConstraintViolation(
            constraint_type="handmade",
            overall_violation=0.42,
            group_violations={"z": -1.5},
            is_satisfied=True,
            tolerance=0.07,
            details={"anything": "at all"},
        )
        assert v.to_dict() == {
            "constraint_type": "handmade",
            "overall_violation": 0.42,
            "group_violations": {"z": -1.5},
            "is_satisfied": True,
            "tolerance": 0.07,
            "details": {"anything": "at all"},
            "could_not_evaluate": False,
        }

    def test_optimization_result_dict_passes_a_nan_through_rather_than_tidying_it(self):
        r = OptimizationResult(
            converged=False,
            n_iterations=3,
            final_violation=float("nan"),
            best_gap=float("nan"),
            final_weights=np.array([0.25, 0.75]),
        )
        d = r.to_dict()

        assert _is_nan(d["final_violation"]), "a NaN violation must not become a 0.0"
        assert _is_nan(d["best_gap"])
        assert d["converged"] is False
        assert d["final_weights"] == [0.25, 0.75]
        assert d["n_iterations"] == 3

    def test_control_optimization_result_dict_keeps_a_measured_run_exactly(self):
        r = OptimizationResult(
            converged=True,
            n_iterations=5,
            final_violation=0.12,
            best_gap=0.01,
            loss_history=[0.4, 0.3],
            metadata={"eta": 2.0},
        )
        assert r.to_dict() == {
            "converged": True,
            "n_iterations": 5,
            "final_violation": 0.12,
            "best_gap": 0.01,
            "lambda_history": [],
            "loss_history": [0.4, 0.3],
            "violation_history": [],
            "final_weights": None,
            "metadata": {"eta": 2.0},
        }


# ---------------------------------------------------------------------------
# Bounded group loss: the one-sided value, and the group that does not exist
# ---------------------------------------------------------------------------


class TestBoundedGroupLossSignedValue:
    def test_a_group_inside_the_bound_is_a_measured_zero_and_says_nothing(self):
        """0.0 here is the clamped one-sided constraint value for a group whose
        loss really is under the bound (0.2 against a bound of 0.2625), and it
        must be distinguishable from the unmeasurable 0.0 below."""
        y_pred, y_true, sens = _healthy()
        c = BoundedGroupLossConstraint(tolerance=0.05)
        value, caught = _run(lambda: c.signed_constraint_value(y_pred, y_true, sens, "a"))

        assert value == pytest.approx(0.0)
        assert caught == [], "a measured pass must not warn"

    def test_an_unmeasurable_group_returns_the_same_zero_but_announces_it(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 30)
        c = BoundedGroupLossConstraint(tolerance=0.05)
        value, caught = _run(
            lambda: c.signed_constraint_value(1 - y_true, y_true, np.array(["a"] * 30), "a")
        )

        assert value == 0.0
        assert any("UNMEASURABLE" in str(w.message) for w in caught), (
            "a classifier wrong on every row in the only group must not read as compliant"
        )

    @pytest.mark.parametrize("name,cls", ALL_CONSTRAINTS)
    def test_a_group_that_does_not_exist_raises_rather_than_answering_zero(self, name, cls):
        y_pred, y_true, sens = _healthy()
        with pytest.raises(KeyError):
            cls(tolerance=0.05).signed_constraint_value(y_pred, y_true, sens, "no_such_group")


# ---------------------------------------------------------------------------
# EqualizedOddsConstraint.signed_constraint_value: it does NOT route through
# _steering_value, so it needs its own three-state pins.
# ---------------------------------------------------------------------------


class TestEqualizedOddsSignedValueThreeStates:
    def test_a_single_group_gets_a_warning_with_its_zero(self):
        """With one group the centre IS this group's own rate, so both gaps are
        0.0 by arithmetic vacuity. 0.0 is the deliberate no-push steering value
        and it must announce that it is not a measured perfect parity."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 30)
        y_pred = rng.integers(0, 2, 30)
        c = EqualizedOddsConstraint(tolerance=0.05)
        value, caught = _run(
            lambda: c.signed_constraint_value(y_pred, y_true, np.array(["a"] * 30), "a")
        )

        assert value == 0.0
        assert any("UNMEASURABLE" in str(w.message) for w in caught), (
            f"got {[str(w.message) for w in caught]}"
        )

    def test_an_arm_whose_centre_holds_one_rate_is_not_a_measured_gap(self):
        """The case a finite-centre test cannot see. Group 'b' carries a label
        that is neither 1 nor 0 (an unknown/unresolved outcome, which this API
        does not reject), so NEITHER of b's rates is defined and both of a's
        arms are centred on a's own rate alone. Both gaps are then `r - r`,
        exactly 0.0 for every possible input, while the centres themselves are
        perfectly finite numbers, so ``overall_tpr == overall_tpr`` says
        'measured' and is wrong.

        BEFORE: 0.0 returned in silence, byte-identical to a measured perfect
        parity between two groups.
        """
        y_true = np.array([1] * 5 + [0] * 5 + [2] * 10)
        sens = np.array(["a"] * 10 + ["b"] * 10)
        y_pred = np.array([1] * 4 + [0] * 1 + [1] * 2 + [0] * 3 + [1] * 5 + [0] * 5)

        c = EqualizedOddsConstraint(tolerance=0.05)
        v = c.compute_violation(y_pred, y_true, sens)
        assert v.details["group_tprs"]["a"] == pytest.approx(0.8), "fixture: a's TPR is measured"
        assert _is_nan(v.details["group_tprs"]["b"]), "fixture: b has no positive labels"
        assert _is_nan(v.details["group_fprs"]["b"]), "fixture: b has no negative labels"

        value, caught = _run(lambda: c.signed_constraint_value(y_pred, y_true, sens, "a"))
        assert value == 0.0
        assert any("UNMEASURABLE" in str(w.message) for w in caught), (
            f"a gap against nothing must not read as a measured zero; "
            f"got {value!r} with {[str(w.message) for w in caught]}"
        )
        assert _is_nan(v.group_violations["a"])

    def test_control_both_arms_measured_returns_the_larger_signed_gap(self):
        y_pred, y_true, sens = _healthy()
        c = EqualizedOddsConstraint(tolerance=0.05)
        # TPR gap for a is +0.2, FPR gap +0.1, so the TPR gap wins on magnitude.
        value, caught = _run(lambda: c.signed_constraint_value(y_pred, y_true, sens, "a"))
        assert value == pytest.approx(0.2)
        assert caught == []


# ---------------------------------------------------------------------------
# The same defect at the surface a REPORT reads
# ---------------------------------------------------------------------------


class TestTheReportingSurface:
    """``FairnessTrainingAnalyzer.evaluate_baseline`` is the public reader of
    ``compute_violation``. Measured on the pristine file before the fix, with
    warnings recorded::

        FairnessTrainingAnalyzer(...).evaluate_baseline(
            y_pred=np.full(30, np.nan)).to_dict()
        -> fairness_violation=0.0, constraint_satisfied=True,
           constraint_evaluated=True, zero warnings

    a compliance certificate for a prediction vector that carried no scores at
    all. This is why the count is taken in the constraint and not left to each
    caller.
    """

    @staticmethod
    def _baseline(y_pred):
        from vfairness.in_processing import FairnessTrainingAnalyzer

        rng = np.random.default_rng(0)
        sens = np.array(["a"] * 15 + ["b"] * 15)
        y_true = np.array([1] * 8 + [0] * 7 + [1] * 8 + [0] * 7)
        analyzer = FairnessTrainingAnalyzer(
            X=rng.normal(size=(30, 3)),
            y=y_true,
            sensitive_attr=sens,
            fairness_constraint="demographic_parity",
            tolerance=0.05,
        )
        return _run(lambda: analyzer.evaluate_baseline(y_pred=y_pred).to_dict())

    def test_an_unscored_baseline_is_not_a_clean_baseline(self):
        d, caught = self._baseline(np.full(30, np.nan))

        assert d["fairness_violation"] is None
        assert d["constraint_satisfied"] is None
        assert d["parameters"]["constraint_evaluated"] is False
        assert any("no usable score" in str(w.message) for w in caught)

    def test_control_a_scored_baseline_still_reports_its_measurement(self):
        # group a predicted positive on every row, group b on none: a measured
        # selection-rate gap of exactly 1.0.
        d, caught = self._baseline(np.array([1.0] * 15 + [0.0] * 15))

        assert d["fairness_violation"] == pytest.approx(1.0)
        assert d["constraint_satisfied"] is False
        assert d["parameters"]["constraint_evaluated"] is True
        assert not any("no usable score" in str(w.message) for w in caught)
