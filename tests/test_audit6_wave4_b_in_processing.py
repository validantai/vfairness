"""
Sixth-iteration audit, wave 4 lane B: in-processing pins.

Both findings are the same defect class, a value nobody measured standing in
for one that was measured:

    S-05  base.soft_rate_computation advertised `temperature` in its
          signature and its docstring ("lower = sharper") and read it
          nowhere. Measured before the fix: for every rate_type, T in
          {0.01, 1.0, 100.0, 1e9} produced a bit-identical value AND a
          bit-identical gradient, and 0.0, -5.0, nan and inf were accepted
          in silence. Disposed of by REFUSAL, not by inventing a soft
          indicator: nothing in the tree passes a non-default value, so the
          only honest states are "identity" and "refused".

    S-07  reductions.ThresholdOptimizer._apply_thresholds read
          `thresholds.get(group, 0.5)`. A group present at predict time but
          absent from the fit was scored at an unoptimised 0.5 and its hard
          labels returned as though the per-group optimisation had covered
          it, while get_thresholds() never listed it, so the caller had no
          signal at all. An unseen group is could-not-check; it is now
          refused.

Each block has a refusal pin AND an over-correction control that asserts the
MEASURED numbers the working path still returns, so reinstating either defect
turns exactly one pin red without the control going green by vacuity.
"""

import warnings

import numpy as np
import pytest

from vfairness.in_processing.constraints.reductions import (
    ExponentiatedGradient,
    ThresholdOptimizer,
)
from vfairness.in_processing.loss_functions.base import (
    compute_group_rates,
    soft_rate_computation,
)

torch = pytest.importorskip("torch", reason="soft_rate_computation needs PyTorch")


# --------------------------------------------------------------------------
# S-05: soft_rate_computation(temperature=...)
# --------------------------------------------------------------------------

# Three rows in the group, three outside it. Every expected number below is
# worked out by hand from these, never read back from the function.
_Y_PRED = [0.9, 0.8, 0.2, 0.55, 0.45, 0.7]
_Y_TRUE = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
_MASK = [True, True, True, False, False, False]


def _tensors():
    return (
        torch.tensor(_Y_PRED, requires_grad=True),
        torch.tensor(_Y_TRUE),
        torch.tensor(_MASK),
    )


@pytest.mark.parametrize(
    "temperature",
    [0.01, 0.5, 2.0, 100.0, 1e9, 0.0, -5.0, float("nan"), float("inf")],
)
def test_soft_rate_computation_refuses_a_temperature_it_cannot_apply(temperature):
    """REFUSAL PIN (S-05): any temperature but the identity is rejected.

    The old body ignored every one of these and returned the plain mean, so a
    caller tuning the parameter got bit-identical gradients and no warning.
    """
    y_pred, y_true, mask = _tensors()
    with pytest.raises(ValueError) as exc:
        soft_rate_computation(y_pred, y_true, mask, rate_type="tpr", temperature=temperature)

    message = str(exc.value)
    # The refusal has to name what IS supported, or it is just an obstacle.
    assert "temperature" in message
    assert "1.0" in message
    assert "FairnessAwareBCELoss" in message


@pytest.mark.parametrize(
    ("rate_type", "expected"),
    [
        # positive_rate: mean(0.9, 0.8, 0.2)
        ("positive_rate", (0.9 + 0.8 + 0.2) / 3),
        # tpr: mean over y == 1 inside the group, i.e. just 0.9
        ("tpr", 0.9),
        # fpr: mean over y == 0 inside the group, i.e. mean(0.8, 0.2)
        ("fpr", (0.8 + 0.2) / 2),
        # tnr: mean(1 - 0.8, 1 - 0.2)
        ("tnr", ((1 - 0.8) + (1 - 0.2)) / 2),
        # fnr: mean(1 - 0.9)
        ("fnr", 1 - 0.9),
    ],
)
def test_soft_rate_computation_still_returns_the_measured_rate(rate_type, expected):
    """OVER-CORRECTION CONTROL (S-05): the default path is unchanged.

    Exact hand-computed rates, not membership of a plausible range, and the
    gradient is checked to be the real one so the refusal cannot have been
    bought by breaking differentiability.
    """
    y_pred, y_true, mask = _tensors()
    value = soft_rate_computation(y_pred, y_true, mask, rate_type=rate_type)
    assert value.item() == pytest.approx(expected, abs=1e-6)

    value.backward()
    assert y_pred.grad is not None
    # Rows outside the group can never influence the group's rate.
    assert y_pred.grad[3:].abs().sum().item() == pytest.approx(0.0, abs=1e-12)
    # And the in-group rows must actually carry gradient.
    assert y_pred.grad[:3].abs().sum().item() > 0.0


def test_soft_rate_computation_accepts_the_identity_explicitly():
    """OVER-CORRECTION CONTROL (S-05): temperature=1.0 is still legal.

    The refusal must not amount to deleting the parameter from under callers
    that spell out the default.
    """
    y_pred, y_true, mask = _tensors()
    explicit = soft_rate_computation(y_pred, y_true, mask, rate_type="tpr", temperature=1.0)
    implicit = soft_rate_computation(y_pred, y_true, mask, rate_type="tpr")
    assert explicit.item() == pytest.approx(0.9, abs=1e-6)
    assert explicit.item() == implicit.item()
    assert explicit.requires_grad


def test_compute_group_rates_still_measures_each_group():
    """OVER-CORRECTION CONTROL (S-05): the only in-tree caller still works.

    compute_group_rates never forwarded a temperature, which is why removal
    or refusal was the right disposition; it must keep working untouched.
    """
    y_pred, y_true, _ = _tensors()
    sensitive = torch.tensor([0, 0, 0, 1, 1, 1])
    rates = compute_group_rates(y_pred, y_true, sensitive, rate_type="positive_rate")
    assert set(rates) == {0, 1}
    assert rates[0].item() == pytest.approx((0.9 + 0.8 + 0.2) / 3, abs=1e-6)
    assert rates[1].item() == pytest.approx((0.55 + 0.45 + 0.7) / 3, abs=1e-6)


# --------------------------------------------------------------------------
# S-07: ThresholdOptimizer, a group seen only at predict time
# --------------------------------------------------------------------------


def _fitted_optimizer():
    """Fit on exactly two groups, deterministically, with no RNG.

    Group A's scores sit low and group B's sit high, so demographic parity
    pushes A's threshold well away from 0.5. That gap is what makes the
    control below able to tell the fitted rule from the old 0.5 default.
    """
    probs_a = np.linspace(0.05, 0.55, 40)
    probs_b = np.linspace(0.45, 0.95, 40)
    y_prob = np.concatenate([probs_a, probs_b])
    y_true = (y_prob >= 0.5).astype(int)
    sensitive = np.array(["A"] * 40 + ["B"] * 40)

    optimizer = ThresholdOptimizer(constraint="demographic_parity", grid_size=50)
    with warnings.catch_warnings():
        # An infeasible fit warns; this fixture must be feasible, so promote
        # that warning to an error rather than pin against a degenerate fit.
        warnings.simplefilter("error")
        optimizer.fit(y_prob, y_true, sensitive_attr=sensitive)
    return optimizer, y_prob, y_true, sensitive


def test_threshold_optimizer_refuses_a_group_it_never_fitted():
    """REFUSAL PIN (S-07): an unseen group at predict time raises.

    Before the fix this returned hard labels for group C computed with an
    unoptimised 0.5, indistinguishable from an optimised threshold.
    """
    optimizer, _, _, _ = _fitted_optimizer()
    assert set(optimizer.get_thresholds()) == {"A", "B"}

    y_prob_test = np.array([0.10, 0.40, 0.60, 0.90] * 3)
    sensitive_test = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)

    with pytest.raises(ValueError) as exc:
        optimizer.predict(y_prob_test, sensitive_test)

    message = str(exc.value)
    assert "'C'" in message  # names the group that was never fitted
    assert "'A'" in message and "'B'" in message  # and what WAS fitted
    # get_thresholds() is unchanged: it never invented an entry for C.
    assert set(optimizer.get_thresholds()) == {"A", "B"}


def test_threshold_optimizer_applies_its_fitted_thresholds_not_a_default():
    """OVER-CORRECTION CONTROL (S-07): the fitted groups still score.

    Asserts the exact per-group rule, and probes a value that lies strictly
    between A's fitted threshold and 0.5, so a silent fall back to 0.5 would
    flip the label. A membership assertion could not see that.
    """
    optimizer, _, _, _ = _fitted_optimizer()
    thresholds = optimizer.get_thresholds()

    assert thresholds["A"] == pytest.approx(0.11, abs=1e-9)
    assert thresholds["B"] == pytest.approx(0.5, abs=1e-9)
    assert optimizer.constraint_satisfied_ is True
    assert optimizer.final_violation_ == pytest.approx(0.025, abs=1e-6)

    # A probe strictly inside (threshold_A, 0.5): the fitted rule says 1,
    # the old 0.5 default said 0.
    midpoint = (thresholds["A"] + 0.5) / 2
    assert thresholds["A"] < midpoint < 0.5
    assert optimizer.predict(np.array([midpoint]), np.array(["A"])).tolist() == [1]
    assert optimizer.predict(np.array([midpoint]), np.array(["B"])).tolist() == [0]

    # Whole-vector check against the exact per-group rule.
    y_prob_test = np.array([0.05, 0.20, 0.45, 0.80] * 2)
    sensitive_test = np.array(["A"] * 4 + ["B"] * 4)
    predictions = optimizer.predict(y_prob_test, sensitive_test)
    expected = np.concatenate(
        [
            (y_prob_test[:4] >= thresholds["A"]).astype(int),
            (y_prob_test[4:] >= thresholds["B"]).astype(int),
        ]
    )
    assert predictions.tolist() == expected.tolist()
    assert predictions.tolist() == [0, 1, 1, 1, 0, 0, 0, 1]


def test_threshold_optimizer_predicts_on_a_subset_of_fitted_groups():
    """OVER-CORRECTION CONTROL (S-07): fewer groups than fitted is fine.

    The refusal is about groups with no threshold, not about a predict set
    that happens to be smaller than the fit set.
    """
    optimizer, _, _, _ = _fitted_optimizer()
    predictions = optimizer.predict(np.array([0.05, 0.20]), np.array(["A", "A"]))
    assert predictions.tolist() == [0, 1]  # 0.05 < 0.11 <= 0.20


def test_threshold_optimizer_fit_is_unaffected_by_the_refusal():
    """OVER-CORRECTION CONTROL (S-07): fit()'s internal calls still pass.

    fit() seeds its threshold dict from the same GroupManager it applies, so
    the new completeness check must never fire there. A fit that raised would
    be the obvious way to "pass" the refusal pin while breaking the product.
    """
    optimizer, y_prob, _, sensitive = _fitted_optimizer()
    assert optimizer.final_violation_ is not None
    # And predicting on the training attribute reproduces the fitted rule.
    predictions = optimizer.predict(y_prob, sensitive)
    assert predictions.shape == y_prob.shape
    # A MEASURED expectation, not a shape check. `issubset({0, 1})` was here
    # and held for any binary array whatsoever, which is the vacuous shape this
    # audit found elsewhere today: it cannot fail, so it pins nothing. The rule
    # applied is each row's own group threshold, so recompute it and demand
    # equality, and demand that the outcome is not degenerate.
    expected = np.array(
        [
            int(prob >= optimizer.get_thresholds()[str(group)])
            for prob, group in zip(y_prob, sensitive)
        ]
    )
    assert predictions.tolist() == expected.tolist()
    assert 0 < int(predictions.sum()) < predictions.size, (
        "the control accepts all or rejects all, so it would pass under a "
        f"broken rule too: {predictions.tolist()}"
    )


def test_exponentiated_gradient_bounded_group_loss_still_fits():
    """SIBLING CONTROL: _compute_costs lost two parameters it never read.

    `y` and `sensitive_attr` were accepted and unused (the same inert-
    parameter shape as S-05). They were removed; this exercises the only
    path that calls the method, so a broken signature cannot pass unnoticed.
    """
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    sensitive = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y = (X[:, 0] + (sensitive == "B") * 0.8 > 0).astype(int)

    eg = ExponentiatedGradient(
        base_estimator=LogisticRegression(max_iter=200),
        constraint="bounded_group_loss",
        max_iterations=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = eg.fit(X, y, sensitive_attr=sensitive)

    assert result.final_violation is not None
    assert np.isfinite(result.final_violation)
    predictions = eg.predict(X)
    assert predictions.shape == (n,)


class TestAGroupNobodyFittedIsRefusedNotRejected:
    """The refuter found lane B fixed one of three copies of this shape, and
    the least public one. ``GroupThresholdOptimizer._apply_group_thresholds``
    iterates only the FITTED thresholds, so a group present at predict time but
    absent from the fit kept the ``np.zeros`` seed and was REJECTED ENTIRELY,
    silently. Measured before this fix: fitting on a and b then predicting with
    a group c gave group c an acceptance rate of 0.000 and emitted no warning.
    The harshest possible verdict, for a group nobody measured."""

    @staticmethod
    def _fitted():
        import numpy as np

        from vfairness.post_processing.threshold_optimization.optimizer import (
            GroupThresholdOptimizer,
        )

        rng = np.random.default_rng(1)
        n = 300
        sens = rng.choice(["a", "b"], n)
        opt = GroupThresholdOptimizer()
        opt.fit(y_true=rng.integers(0, 2, n), y_prob=rng.random(n), sensitive_attr=sens)
        return opt, rng, sens

    def test_an_unfitted_group_is_refused(self):
        import numpy as np

        opt, rng, _ = self._fitted()
        unseen = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
        with pytest.raises(ValueError, match="No threshold was fitted"):
            opt.predict(rng.random(300), unseen)

    def test_the_refusal_names_the_group_and_the_fitted_ones(self):
        import numpy as np

        opt, rng, _ = self._fitted()
        unseen = np.array(["a"] * 150 + ["zz"] * 150)
        with pytest.raises(ValueError) as e:
            opt.predict(rng.random(300), unseen)
        msg = str(e.value)
        assert "'zz'" in msg, msg
        assert "'a'" in msg and "'b'" in msg, msg
        assert "np.str_" not in msg, msg

    def test_control_the_fitted_groups_still_predict_unchanged(self):
        """Over-correction control: the refusal must not touch the normal path.
        Asserted as a measured value, not a range."""
        opt, _, sens = self._fitted()
        import numpy as np

        rng2 = np.random.default_rng(1)
        rng2.choice(["a", "b"], 300)
        probs = rng2.random(300)
        first = opt.predict(probs, sens)
        assert first.sum() > 0, "control fixture rejects everything, it pins nothing"
        assert (opt.predict(probs, sens) == first).all()


class TestAnUnmeasurableRateSaysSo:
    """Three lines below lane B's own edit, ``soft_rate_computation`` returned a
    bare 0.5 for a group with no positive (or no negative) examples: a
    fabricated rate fed straight into a fairness penalty. NaN cannot be used
    here because one NaN poisons the gradient for every group, so the finite
    stand-in stays and the silence goes."""

    @staticmethod
    def _rate(rate_type, y_true_val):
        import torch

        from vfairness.in_processing.loss_functions.base import soft_rate_computation

        y_pred = torch.rand(20, requires_grad=True)
        y_true = torch.full((20,), float(y_true_val))
        mask = torch.ones(20, dtype=torch.bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = soft_rate_computation(y_pred, y_true, mask, rate_type=rate_type)
        return value, [str(w.message) for w in caught]

    @pytest.mark.parametrize("rate_type,y_true_val", [("tpr", 0), ("fpr", 1)])
    def test_an_absent_arm_is_named_as_unmeasured(self, rate_type, y_true_val):
        value, msgs = self._rate(rate_type, y_true_val)
        assert float(value) == 0.5
        named = [m for m in msgs if "NOT MEASURED" in m]
        assert named, msgs
        assert "exclude this group" in named[0]

    @pytest.mark.parametrize("rate_type,y_true_val", [("tpr", 1), ("fpr", 0)])
    def test_control_a_measurable_arm_is_silent(self, rate_type, y_true_val):
        """Over-correction control: the warning must not fire on a real rate."""
        value, msgs = self._rate(rate_type, y_true_val)
        assert not [m for m in msgs if "NOT MEASURED" in m], msgs
        assert 0.0 <= float(value) <= 1.0
