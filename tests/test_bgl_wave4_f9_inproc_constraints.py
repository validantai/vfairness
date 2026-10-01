"""Wave-4 pins for the F9 in-processing constraints and calibrators package.

Six grades the independent auditor overturned on HEAD 9e88dd5ac, all of the same
class: a neutral value substituted for one that could not be measured, then
published as a measurement. Every test class names the door it pins and the
SIBLING door beside it, because in each of these the guard had been written for
one door while its sibling stayed open.

These are the MITIGATIONS, the code a user runs to make a model fairer, so the
failure mode is inverted and worth restating: A NEUTERED MITIGATION REPORTS
SUCCESS, NOT FAILURE. Approving everybody is trivially equal, so a mitigation
that has been silently switched off shows a BETTER parity number than one that
works. None of these pins therefore reads the intervention's own fairness metric;
each one reads whether the intervention still BINDS: whether the bound can be
breached, whether the optimiser is still pushed, whether the vote is still a
majority, whether the loss still sums over any bins.

Every pin is paired with a CONTROL asserting the healthy case's REAL number, not
merely that it did not raise: a guard that refuses everything passes every
refusal test.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.exceptions import ConfigurationError
from vfairness.in_processing.constraints.base import (
    BoundedGroupLossConstraint,
    DemographicParityConstraint,
    OptimizationResult,
)
from vfairness.in_processing.constraints.reductions import (
    EnsembleClassifier,
    ExponentiatedGradient,
    ReductionResult,
)

# ---------------------------------------------------------------------------
# The bounded-group-loss fixture, worked out here rather than read back from the
# code. 60 rows, every label 1. Group "a" is wrong on 24 of its 30 rows, so its
# 0-1 loss is 0.8; group "b" is right on all 30, so its loss is 0.0. The
# population loss is 24/60 == 0.4, and the relative bound is
# (1 + tolerance) * 0.4. At the default-ish tolerance 0.05 the bound is 0.42 and
# group "a" breaches it by 0.8 - 0.42 == 0.38. That is a REAL breach and it is
# what every control below asserts.
# ---------------------------------------------------------------------------
_Y_TRUE = np.ones(60, dtype=int)
_Y_PRED = np.ones(60, dtype=int)
_Y_PRED[:24] = 0
_SA = np.array(["a"] * 30 + ["b"] * 30)


def _violation(tolerance, **kwargs):
    """compute_violation plus the warnings it raised."""
    constraint = BoundedGroupLossConstraint(tolerance=tolerance, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = constraint.compute_violation(_Y_PRED, _Y_TRUE, _SA)
    return result, [str(w.message) for w in caught]


class TestAVacuousBoundIsNotAMeasuredPass:
    """F9 row 2: BoundedGroupLossConstraint.compute_violation.

    The object-dtype label/prediction door was closed. The public ``tolerance``
    argument was a second, unguarded door into the same exactly-0.0 PASS, and
    nothing anywhere tested the BOUND.
    """

    @pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), True, "0.05"])
    def test_a_bound_that_is_not_a_measurement_is_refused_at_construction(self, tolerance):
        """NaN is the worst value of the set: it is the sentinel this file uses
        for 'could not evaluate', and because every comparison against NaN is
        False while Python's max returns its FIRST argument then,
        ``max(0.0, nan - bound)`` was exactly 0.0. The guard is above the
        subclass selection, so it cannot be reached with the object already
        misconfigured."""
        with pytest.raises(ConfigurationError, match="not a finite number"):
            BoundedGroupLossConstraint(tolerance=tolerance)

    def test_a_negative_maximum_cannot_be_satisfied_so_it_is_refused(self):
        """The mirror of the vacuous bound: every violation here is a magnitude,
        so a negative MAXIMUM allowed violation can only ever fail."""
        with pytest.raises(ConfigurationError, match="negative"):
            BoundedGroupLossConstraint(tolerance=-0.5)

    def test_the_guard_is_above_the_subclass_selection(self):
        """A sibling constraint class shares the same unchecked tolerance, so the
        refusal has to live above the dispatch rather than in one
        compute_violation."""
        with pytest.raises(ConfigurationError, match="not a finite number"):
            DemographicParityConstraint(tolerance=float("nan"))

    def test_a_finite_vacuous_bound_withdraws_the_verdict(self):
        """THE DOOR A GUARD TESTING ONLY isinf LEAVES OPEN. tolerance=60 puts the
        bound at 24.4 on a 0-1 loss that cannot exceed 1.0, so the comparison
        that detects a breach is False whatever the data. It used to publish
        overall_violation 0.0, is_satisfied True, insufficient_data False and
        zero warnings for the measured 0.38 breach above."""
        result, messages = _violation(60.0)
        assert math.isnan(result.overall_violation)
        assert result.is_satisfied is False
        assert result.details["insufficient_data"] is True
        assert result.details["bound"] == pytest.approx(24.4)
        # The reader-visible third state, not only a return value nobody renders.
        reason = result.details["unusable_bound_reason"]
        assert reason is not None
        assert "24.4000" in reason and "1.0000" in reason
        assert len(messages) == 1, messages
        assert "cannot grade this metric" in messages[0]

    def test_the_guard_is_above_the_max_so_both_channels_agree(self):
        """PUT THE GUARD ABOVE THE DISPATCH. The per-group channel and the
        population channel are two comparisons against the same bound, and the
        per-group one is what the OPTIMISER reads. A check in either branch alone
        would leave the other publishing a measured 0.0."""
        result, _ = _violation(60.0)
        assert math.isnan(result.overall_violation)
        for group in ("a", "b"):
            assert math.isnan(result.group_violations[group]), group

    def test_control_a_real_breach_is_measured_exactly_as_before(self):
        """OVER-CORRECTION CONTROL with the REAL number. 0.8 - 1.05*0.4 == 0.38,
        worked out from the fixture, and it must survive untouched."""
        result, messages = _violation(0.05)
        assert result.overall_violation == pytest.approx(0.38)
        assert result.group_violations["a"] == pytest.approx(0.38)
        assert result.is_satisfied is False
        assert result.details["insufficient_data"] is False
        assert result.details["unusable_bound_reason"] is None
        assert messages == []

    def test_control_a_tolerance_of_one_is_a_real_boundary_not_a_vacuous_bound(self):
        """THE NO-OVER-ACCUSATION CONTROL, and the reason vacuity is decided
        against the loss's REACHABLE range rather than against the size of the
        tolerance. tolerance=1.0 puts the bound at 2*0.4 == 0.8, which group "a"
        meets EXACTLY, so the 0.0 here is a measured boundary result: a group at
        0.9 would still breach it. Accusing this case would be the same error
        class as fabricating."""
        result, messages = _violation(1.0)
        assert result.details["bound"] == pytest.approx(0.8)
        assert result.overall_violation == pytest.approx(0.0)
        assert result.is_satisfied is True
        assert result.details["insufficient_data"] is False
        assert result.details["unusable_bound_reason"] is None
        assert messages == []

    def test_control_a_loss_with_no_declared_range_is_a_could_not_check(self):
        """A caller-supplied loss_fn has no declared range, so nothing can be
        shown about reachability and the bound stays exactly as usable as it was
        configured. Refusing here would be the over-correction: a constraint that
        refuses everything passes every refusal test."""
        result, messages = _violation(60.0, loss_fn=lambda p, t: float(np.mean(p != t)))
        assert result.overall_violation == pytest.approx(0.0)
        assert result.details["unusable_bound_reason"] is None
        assert messages == []


class TestAVacuousBoundDoesNotSilenceTheOptimiser:
    """F9 row 3: BoundedGroupLossConstraint.signed_constraint_value.

    The same bound, one layer further in, where it stops being a report and
    becomes the steering push a mitigation applies. The auditor's finding was
    that ``_steering_value`` could not help, because it was handed the
    already-clamped 0.0 rather than a NaN, so its three-state ladder never fired.
    """

    def _signed(self, tolerance, group="a"):
        constraint = BoundedGroupLossConstraint(tolerance=tolerance)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = constraint.signed_constraint_value(_Y_PRED, _Y_TRUE, _SA, group)
        return value, [str(w.message) for w in caught]

    def test_a_vacuous_bound_no_longer_returns_a_silent_zero_push(self):
        """The float stays 0.0 on purpose: it is consumed by the exponentiated
        gradient update and by _signed_gaps, where a NaN would poison every
        multiplier. What was missing is that 0.0 meant NO PUSH rather than NO
        GAP, and nothing said so. The three-state ladder in _steering_value now
        fires, because the value it is handed is a NaN."""
        value, messages = self._signed(60.0)
        assert value == 0.0
        assert any("cannot grade this metric" in m for m in messages), messages
        assert any("UNMEASURABLE" in m and "STEERING value" in m for m in messages), messages

    def test_the_measurement_channel_carries_the_nan_not_the_push(self):
        """The docstring's own claim: this float is a push and the
        ConstraintViolation is the measurement. Pin both halves."""
        constraint = BoundedGroupLossConstraint(tolerance=60.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            violation = constraint.compute_violation(_Y_PRED, _Y_TRUE, _SA)
        assert math.isnan(violation.group_violations["a"])
        assert violation.details["insufficient_data"] is True

    def test_control_the_real_steering_push_is_unchanged(self):
        """OVER-CORRECTION CONTROL with the REAL number: the push for the
        breached group at a usable tolerance is still 0.38, in silence."""
        value, messages = self._signed(0.05)
        assert value == pytest.approx(0.38)
        assert messages == []

    def test_control_a_tolerance_of_one_still_pushes_its_measured_zero(self):
        """The boundary case pushes a measured 0.0 and says nothing, which is the
        state a vacuous bound must no longer be confused with."""
        value, messages = self._signed(1.0)
        assert value == pytest.approx(0.0)
        assert messages == []


# ---------------------------------------------------------------------------
# Ensemble fixtures. Six rows, and members whose labels are strictly in {0, 1}
# so the label-domain guard is satisfied and only the weight axis is under test.
# ---------------------------------------------------------------------------
_X6 = np.zeros((6, 2))


class _HardMember:
    """A member that returns one fixed label in the {0, 1} vote domain."""

    def __init__(self, label):
        self.label = int(label)

    def predict(self, X):
        return np.full(len(X), self.label, dtype=int)


class _ProbaMember:
    """A member returning a caller-chosen score column."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return (self.values >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.values


def _as_reduction_result(members, weights):
    return ReductionResult(
        classifiers=list(members),
        weights=np.asarray(weights, dtype=float),
        optimization_result=OptimizationResult(
            converged=True, n_iterations=1, final_violation=0.0, best_gap=0.0
        ),
        final_violation=0.0,
        accuracy=1.0,
    )


@pytest.fixture(params=["EnsembleClassifier", "ReductionResult"])
def build(request):
    """Both entry points into the same vote, because this is one of two."""
    if request.param == "EnsembleClassifier":
        return lambda members, weights: EnsembleClassifier(
            list(members), np.asarray(weights, dtype=float)
        )
    return _as_reduction_result


class TestAVoteTheWeightScaleDecides:
    """F9 row 4: EnsembleClassifier.predict.

    The label-domain door is genuinely closed. The weight NORMALISATION door was
    guarded on ONE side only: a total below 0.5 raised, and a total above 1
    silently INVERTED the verdict for every row. A one-sided guard on a two-sided
    quantity.
    """

    # One member says 1 and four say 0, so the correct weighted-majority verdict
    # is 0 for every row. Worked out here: 0.2*1 + 4*(0.2*0) == 0.2, below 0.5.
    MINORITY = [_HardMember(1)] + [_HardMember(0)] * 4

    @pytest.mark.parametrize("weights", [[1.0] * 5, [5.0] * 5, [0.6] * 5])
    def test_an_or_gate_vote_is_refused(self, build, weights):
        """At an unnormalised scale the 0.5 threshold stops being the majority
        point and one member carries the row whatever the others say. Measured
        before this guard: weights [0.2]*5 gave the correct [0] and [1.0]*5 gave
        [1] for EVERY row, with identical members, identical features and zero
        warnings of any kind."""
        with pytest.raises(ValueError, match="OR gate"):
            build(self.MINORITY, weights).predict(_X6)

    def test_the_refusal_names_the_share_that_decided_it(self, build):
        with pytest.raises(ValueError) as excinfo:
            build(self.MINORITY, [1.0] * 5).predict(_X6)
        message = str(excinfo.value)
        assert "total 5" in message
        assert "20.0%" in message, message

    def test_control_the_normalised_minority_verdict_is_the_real_one(self, build):
        """OVER-CORRECTION CONTROL with the REAL verdict: one member in five
        saying 1 must come back as 0 for every row, in silence."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = build(self.MINORITY, [0.2] * 5).predict(_X6)
        np.testing.assert_array_equal(labels, np.zeros(6, dtype=int))
        assert [str(w.message) for w in caught] == []

    def test_control_a_single_member_at_full_weight_is_untouched(self, build):
        """A total of exactly 1 with one member: the largest weight reaches the
        threshold AND holds 100% of the weight, so it is a majority, not an OR
        gate. This is the case the refusal must never touch."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = build([_HardMember(1)], [1.0]).predict(_X6)
        np.testing.assert_array_equal(labels, np.ones(6, dtype=int))
        assert [str(w.message) for w in caught] == []

    def test_control_hand_rounded_weights_keep_their_verdict_and_are_disclosed(self, build):
        """THE NO-OVER-ACCUSATION CONTROL. Three members at 0.34 total 1.02, a
        caller rounding 1/3 to two decimals. No single member reaches 0.5 from
        0.34, so the verdict is the correct one and refusing it would break a
        working ensemble over two hundredths. It is disclosed instead, on the same
        1e-6 criterion predict_proba already applies to the identical total."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = build([_HardMember(1)] * 3, [0.34] * 3).predict(_X6)
        np.testing.assert_array_equal(labels, np.ones(6, dtype=int))
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1, messages
        assert "total 1.02" in messages[0]

    def test_control_the_lower_side_of_the_same_guard_still_raises(self, build):
        """The sibling refusal this one mirrors must be untouched."""
        with pytest.raises(ValueError, match="below the 0.5 decision threshold"):
            build([_HardMember(1)], [0.2]).predict(_X6)

    def test_an_infinite_weight_is_refused_on_the_weight_value_axis(self, build):
        """THE THIRD DOOR ON THAT AXIS. The NaN and negative branches of
        _refuse_a_silent_ensemble were both written for a weight that is not a
        usable number, and +inf is neither: inf > 1e-8 is True, so it contributes
        and then dominates the sum absolutely. Measured: predict returned 1 for
        every row, the opposite of the correct 1-of-5 minority verdict. The
        scale-based guard above CANNOT catch it, because its inversion test
        compares the largest weight against total / 2 and `inf < inf` is False,
        so it is refused by the guard all four entry points already share."""
        with pytest.raises(ValueError, match="infinite"):
            build(self.MINORITY, [float("inf")] + [1.0] * 4).predict(_X6)

    def test_an_infinite_weight_is_refused_on_the_score_path_too(self, build):
        """All four entry points, from the one shared guard."""
        with pytest.raises(ValueError, match="infinite"):
            build([_ProbaMember([0.3] * 6)], [float("inf")]).predict_proba(_X6)


class TestAPartlyUnusableScoreColumn:
    """F9 row 5: EnsembleClassifier.predict_proba.

    The TOTALLY unusable score column was disclosed and the PARTIALLY unusable
    one was silent, so nine of ten rows carrying no score at all was published
    with LESS disclosure than one row of hard 0.0 gets.
    """

    def _proba(self, build, values):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = build([_ProbaMember(values)], [1.0]).predict_proba(np.zeros((len(values), 2)))
        return out, [str(w.message) for w in caught]

    @pytest.mark.parametrize(
        "values, n_unusable",
        [
            ([np.nan] * 5 + [0.3, 0.4, 0.5, 0.6, 0.7], 5),
            ([np.nan] * 9 + [0.42], 9),
            ([np.inf] * 5 + [0.3, 0.4, 0.5, 0.6, 0.7], 5),
            ([np.nan, np.nan, 0.48, 0.62, 0.76, 0.9], 2),
        ],
    )
    def test_a_partly_unusable_column_says_how_many_rows_carry_no_score(
        self, build, values, n_unusable
    ):
        """Every threshold comparison against NaN is False, so an unscored row
        silently becomes a confident REJECTION, which is the same sentence this
        module uses to justify the guard it already had."""
        out, messages = self._proba(build, values)
        assert int(np.isnan(out).sum() + np.isinf(out).sum()) == n_unusable
        assert len(messages) == 1, messages
        assert (
            f"{n_unusable} of {len(values)} returned value(s) carry no usable score"
            in (messages[0])
        )

    def test_control_a_constant_interior_probability_stays_silent(self, build):
        """OVER-CORRECTION CONTROL with the REAL values. A constant 0.3 is a real
        probability: constancy alone is a property of the members' own output, and
        this carve-out is documented on the guard beside it."""
        out, messages = self._proba(build, [0.3] * 10)
        np.testing.assert_allclose(out, np.full(10, 0.3))
        assert messages == []

    def test_control_an_honest_spread_stays_silent(self, build):
        out, messages = self._proba(build, [0.1, 0.25, 0.4, 0.55, 0.7, 0.95])
        np.testing.assert_allclose(out, [0.1, 0.25, 0.4, 0.55, 0.7, 0.95])
        assert messages == []

    def test_control_the_totally_unusable_column_keeps_its_own_disclosure(self, build):
        """The arm that was already there must not be displaced by the new one."""
        out, messages = self._proba(build, [np.nan] * 10)
        assert int(np.isnan(out).sum()) == 10
        assert len(messages) == 1, messages
        assert "not one of 10 returned value(s) is finite" in messages[0]


# ---------------------------------------------------------------------------
# The fitted ExponentiatedGradient surface.
# ---------------------------------------------------------------------------


def _split_rate_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    A = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    y = np.concatenate(
        [
            (rng.random(n // 2) < 0.25).astype(int),
            (rng.random(n // 2) < 0.75).astype(int),
        ]
    )
    return X, y, A


class _CollapsedEstimator:
    """A base estimator that predicts one class for every row, standing for the
    collapsed best response ``_fit_best_response`` substitutes."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class _PartlyUnscoredEstimator:
    """Honest on most rows and unscored on the first three."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        labels = np.zeros(len(X), dtype=int)
        labels[len(X) // 2 :] = 1
        return labels

    def predict_proba(self, X):
        p = np.full(len(X), 0.6)
        p[:3] = np.nan
        return np.column_stack([1.0 - p, p])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def _fit(estimator, X, y, A, max_iterations=3):
    eg = ExponentiatedGradient(
        estimator,
        DemographicParityConstraint(tolerance=0.05),
        max_iterations=max_iterations,
    )
    # Suppressed deliberately: this stands for a caller holding a fitted
    # estimator, or one restored from a stored result, who was not there for the
    # fit-time warnings. That caller is the whole subject of this class.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg.fit(X, y, sensitive_attr=A)
    return eg


def _call(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught]


class TestACollapsedFitIsDisclosedOnTheLabelPathToo:
    """F9 row 6: ExponentiatedGradient.predict_proba and its two siblings.

    The collapsed-fit flag branch on ``predict_proba`` works and was verified by
    execution. The column that method RETURNS could still be part NaN with zero
    disclosure, and the sibling entry point ``predict``, which is the method a
    decision is taken FROM, carried no collapse disclosure at all.
    """

    def test_the_collapsed_label_path_now_discloses(self):
        """Measured before: EG.predict returned 1 for all 240 rows with ZERO
        warnings, beside constraint_satisfied True, insufficient_data False and
        final_violation 0.0. A model that accepted everybody, reporting success."""
        X, y, A = _split_rate_data()
        eg = _fit(_CollapsedEstimator(), X, y, A)
        assert eg.get_result().fairness_metrics["degenerate_constant_predictions"] is True

        labels, messages = _call(lambda: eg.predict(X))
        assert np.unique(labels).size == 1
        assert any("single class for every row" in m for m in messages), messages
        assert any("degenerate_constant_predictions" in m for m in messages), messages

    def test_the_disclosure_reaches_the_delegate_entry_point_as_well(self):
        """ONE OF TWO ENTRY POINTS. It lives on ReductionResult.predict, which is
        what ExponentiatedGradient.predict delegates to, so both are covered from
        one place and a result restored from a stored payload is covered too."""
        X, y, A = _split_rate_data()
        result = _fit(_CollapsedEstimator(), X, y, A).get_result()
        labels, messages = _call(lambda: result.predict(X))
        assert np.unique(labels).size == 1
        assert any("single class for every row" in m for m in messages), messages

    def test_a_partly_unscored_column_through_this_method_is_disclosed(self):
        """THE SIBLING DOOR. The flag branch reads only
        degenerate_constant_predictions, and the delegate it then calls tested
        only all-non-finite or all-{0,1}, so a PARTLY unmeasurable column passed
        both. Measured before: 3 of 60 values NaN with the flag correctly False
        and ZERO warnings."""
        X, y, A = _split_rate_data(n=60, seed=1)
        eg = _fit(_PartlyUnscoredEstimator(), X, y, A)
        assert eg.get_result().fairness_metrics["degenerate_constant_predictions"] is False

        scores, messages = _call(lambda: eg.predict_proba(X))
        assert int(np.isnan(scores).sum()) > 0
        assert any("carry no usable score" in m for m in messages), messages

    def test_control_a_healthy_fit_answers_in_silence_with_real_numbers(self):
        """OVER-CORRECTION CONTROL, and the one that matters most here: a real
        fitted pipeline must be completely silent on BOTH entry points, separate
        more than one class, and carry convex weights."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(3)
        n = 300
        X = rng.normal(size=(n, 4))
        A = np.array(["a"] * 150 + ["b"] * 150)
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        eg = _fit(LogisticRegression(max_iter=300), X, y, A, max_iterations=4)

        assert eg.get_result().weights.sum() == pytest.approx(1.0)

        labels, label_messages = _call(lambda: eg.predict(X))
        scores, score_messages = _call(lambda: eg.predict_proba(X))
        assert np.unique(labels).size == 2
        assert np.unique(scores).size > 2
        assert label_messages == []
        assert score_messages == []


# ---------------------------------------------------------------------------
# The trainable calibrator. torch is optional in this package, so the whole class
# skips rather than failing when it is absent.
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from vfairness.in_processing.calibrators.group_calibrators import (  # noqa: E402
    TrainableGroupCalibrator,
)


def _calibration_batch():
    """A DETERMINISTIC batch, built without any RNG so the control's real number
    is stable across torch versions. 60 rows in two groups of 30. The logits
    sweep a wide range so sigmoid spans the bins, and the labels are deliberately
    anti-correlated with the scores on the first group, which is what gives the
    batch a real, non-zero calibration error to measure."""
    logits = torch.linspace(-4.0, 4.0, 60).reshape(60, 1)
    labels = torch.zeros(60)
    labels[:30] = 1.0  # the low-score half carries the positives: miscalibrated
    group_ids = torch.tensor([0] * 30 + [1] * 30)
    return logits, labels, group_ids


def _loss(weight=0.1, **kwargs):
    calibrator = TrainableGroupCalibrator(n_groups=2, calibration_loss_weight=weight)
    logits, labels, group_ids = _calibration_batch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = calibrator.calibration_loss(logits, labels, group_ids, **kwargs)
    return float(value.item()), [str(w.message) for w in caught]


class TestACalibrationLossWithNothingSummedIntoIt:
    """F9 row 1: TrainableGroupCalibrator.calibration_loss.

    The min_group_size=0 door was closed with a floor of one usable row. TWO
    sibling public arguments into the identical exactly-0.0 fabrication were still
    open, and 0.0 on this scale is PERFECT calibration by the docstring's own
    words.
    """

    # --- SIBLING DOOR A: the public, documented n_bins argument ---------------

    def test_zero_bins_is_refused_rather_than_returning_perfect_calibration(self):
        """``for i in range(0)`` enters no bin, so ece stayed at its requires_grad
        seed of exactly 0.0 while the group was still appended to `measured`, and
        the method returned weight * 0.0 / len(measured). Measured before:
        n_bins=0 returned 0.0 with ZERO warnings for a batch whose real loss is
        the control's number below."""
        with pytest.raises(ValueError, match=r"n_bins must be >= 1"):
            _loss(n_bins=0)

    def test_a_negative_bin_count_is_a_stated_refusal_not_a_torch_traceback(self):
        """Measured before: RuntimeError 'number of steps must be non-negative'
        out of torch.linspace, naming neither this argument nor this class, i.e.
        not even a refusal."""
        with pytest.raises(ValueError, match=r"n_bins must be >= 1"):
            _loss(n_bins=-5)

    def test_a_non_integer_bin_count_names_the_argument(self):
        with pytest.raises(TypeError, match="n_bins must be an int"):
            _loss(n_bins=2.5)

    def test_an_unmeasurable_floor_is_refused_because_it_is_no_floor_at_all(self):
        """`n_usable < nan` is False for every group, so a non-finite
        min_group_size silently applied NO floor while reading as a threshold."""
        with pytest.raises(ValueError, match="min_group_size"):
            _loss(min_group_size=float("nan"))

    # --- SIBLING DOOR B: the constructor's lambda ----------------------------

    @pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), True])
    def test_a_degenerate_loss_weight_is_refused_at_construction(self, weight):
        """The regulariser-whose-lambda-is-zero degeneracy. Measured before, on a
        batch with a real loss: weight=0.0 returned exactly 0.0, weight=-1.0
        returned a NEGATIVE loss on a scale whose floor is 0, and weight=nan
        forged this method's own documented refusal sentinel, all with ZERO
        warnings. The guard is in __init__, above both callers of
        calibration_loss."""
        with pytest.raises(ValueError, match="calibration_loss_weight"):
            TrainableGroupCalibrator(n_groups=2, calibration_loss_weight=weight)

    def test_the_refusal_points_at_the_documented_way_to_switch_the_term_off(self):
        with pytest.raises(ValueError, match="include_calibration=False"):
            TrainableGroupCalibrator(n_groups=2, calibration_loss_weight=0.0)

    # --- CONTROLS ------------------------------------------------------------

    def test_control_the_healthy_batch_measures_a_real_nonzero_loss(self):
        """OVER-CORRECTION CONTROL with the REAL number. The default n_bins=10 on
        this deliberately miscalibrated batch must return a strictly positive loss
        in silence, and it must be the SAME number after the guards as before
        them. Not compared for equality with 0.0 on a computed statistic: it is
        asserted to be a real, finite, clearly non-zero measurement."""
        value, messages = _loss(n_bins=10)
        assert math.isfinite(value)
        assert value > 0.01, value
        assert messages == []
        # The weight really is doing work: ten times the lambda is ten times the
        # reported loss, which is what a weight of 0.0 destroyed.
        scaled, _ = _loss(weight=1.0, n_bins=10)
        assert scaled == pytest.approx(value * 10.0, rel=1e-5)

    def test_control_one_bin_is_coarse_but_still_a_real_measurement(self):
        """THE NO-OVER-ACCUSATION CONTROL. n_bins=1 is not vacuous: a single bin
        still compares the mean confidence against the mean accuracy, so it
        grades something and must not be refused with n_bins=0."""
        value, messages = _loss(n_bins=1)
        assert math.isfinite(value)
        assert value > 0.01, value
        assert messages == []

    def test_control_the_all_nan_batch_still_refuses_through_its_own_door(self):
        """This was the auditor's proof that the n_bins door was a DIFFERENT one:
        the NaN arm refuses on its own and must keep doing so."""
        calibrator = TrainableGroupCalibrator(n_groups=2)
        _, labels, group_ids = _calibration_batch()
        nan_logits = torch.full((60, 1), float("nan"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = calibrator.calibration_loss(nan_logits, labels, group_ids)
        assert math.isnan(float(value.item()))
        messages = [str(w.message) for w in caught]
        assert any("no group reached min_group_size" in m for m in messages), messages

    def test_control_the_min_group_size_floor_is_unchanged(self):
        """The previously fixed door must still behave: a real floor on a healthy
        batch measures, and the floor above every group refuses."""
        measured, messages = _loss(min_group_size=10)
        assert measured > 0.01 and messages == []
        refused, refusal_messages = _loss(min_group_size=1000)
        assert math.isnan(refused)
        assert any("no group reached min_group_size" in m for m in refusal_messages)
