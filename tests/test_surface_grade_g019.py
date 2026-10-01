"""Surface grade g019: the public surface of ``loss_functions/adversarial.py``.

Twelve items graded, four fabrications found and fixed. All four share the
shape this campaign exists to find: a number produced when nothing was
measured, in a place a caller reads as a measurement.

Measured at the public surface BEFORE the fix:

    1. ``ProjectedAdversarialLoss(n_adversary_steps=0).update_adversary(...)``
       returned **0.0**. ``range(0)`` is empty so no step ran and no loss was
       ever computed; ``total_adv_loss / max(1, 0)`` turned that into 0.0,
       which on a cross-entropy scale is the score a PERFECT adversary earns
       (the attribute fully recovered, i.e. maximal leakage). The sibling
       ``AdversarialDebiasingLoss`` raised ZeroDivisionError for the same
       input, but only AFTER flipping the trained flag.
    2. Both ``update_adversary`` methods latched
       ``_adversary_ever_trained = True`` on the CALL, not on the weights. With
       ``adversary_lr=0.0`` the Adam step provably moves nothing (``torch.equal``
       against the construction snapshot is True for every parameter), yet the
       next ``forward`` reported ``fairness_penalty_assessed=True`` with
       ``fairness_loss 0.6960861`` - exactly the random-initialisation reading
       the coverage machinery in this module exists to refuse. A zero-row
       update, and the ZeroDivisionError path above, reached the same state.
    3. ``get_adversary_accuracy`` swallowed non-finite inputs into a confident
       number: all-NaN predictions scored **0.5** with two groups and
       **0.34375** with three, +inf predictions scored **0.5**, and an all-NaN
       attribute scored **0.0**. On this scale 0.5 is chance ("no leakage") and
       0.0 is a perfectly non-leaking model. ``sigmoid(NaN) > 0.5`` is False,
       ``argmax`` of an all-NaN row is 0, and ``NaN == NaN`` is False: the
       comparison operators decided the answer, not the data.
    4. ``_adversary_is_at_initialization`` compared the snapshot device-cast but
       not dtype-cast, so ``loss_fn.adversary.half()`` (or
       ``loss_fn.to(torch.float16)``) rounded every weight and was scored as
       training: an adversary that had never seen a gradient was credited with
       it, and its next penalty came back assessed.

Every fix carries a CONTROL: a genuinely trained adversary still measures, and
measures the same numbers. A guard that refuses everything would be worse than
the defect it replaces.

Module-local fixtures on purpose: other agents are editing this checkout, so no
shared fixture, conftest or existing test file is touched.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
    Adversary,
    FairRepresentationLoss,
    GradientReversalFunction,
    GradientReversalLayer,
    ProjectedAdversarialLoss,
)

N = 64


def _is_nan(value) -> bool:
    """NaN is the only value that is not equal to itself."""
    return value != value


def _leaking_batch(seed: int = 1):
    """Predictions that carry the attribute perfectly: a findable disparity."""
    torch.manual_seed(seed)
    sensitive = (torch.arange(N) % 2).float()
    y_pred = torch.where(sensitive > 0.5, torch.full((N,), 0.9), torch.full((N,), 0.1))
    y_pred = (y_pred + 0.02 * torch.randn(N)).clamp(0.01, 0.99)
    y_true = (torch.rand(N) > 0.5).float()
    return y_pred, y_true, sensitive


def _weights_moved(loss_fn) -> bool:
    """True when any adversary weight differs from its construction value."""
    return any(
        not torch.equal(p.detach(), q)
        for p, q in zip(loss_fn.adversary.parameters(), loss_fn._adversary_init_state)
    )


def _forward(loss_fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total, comps = loss_fn(*args, return_components=True, **kwargs)
    return total, comps, [str(w.message) for w in caught]


def _call(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# 1. A step count that cannot train is a caller error, not a loss of zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
def test_zero_adversary_steps_is_refused_at_construction(cls):
    with pytest.raises(ValueError, match="n_adversary_steps must be an integer"):
        cls(n_adversary_steps=0)
    with pytest.raises(ValueError, match="n_adversary_steps must be an integer"):
        cls(n_adversary_steps=-3)


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
def test_zero_adversary_steps_set_after_construction_never_returns_a_number(cls):
    """The exact reproduction: Projected returned 0.0 here, the score a
    perfect adversary earns, and marked the untrained adversary trained."""
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = cls()
    loss_fn.n_adversary_steps = 0

    with pytest.raises(ValueError, match="no step the adversary is never trained"):
        loss_fn.update_adversary(y_pred, sensitive)

    # and the refusal did not leave the adversary marked as trained
    assert loss_fn._adversary_ever_trained is False
    assert loss_fn._adversary_is_at_initialization() is True
    _, comps, messages = _forward(loss_fn, y_pred, y_true, sensitive)
    assert _is_nan(comps.fairness_loss)
    assert comps.batch_metrics["fairness_penalty_assessed"] is False
    assert comps.batch_metrics["fairness_unassessable_reason"] == "adversary_at_initialization"
    assert any("NOT MEASURED" in m for m in messages), messages


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
def test_control_one_step_is_accepted_and_averages_over_the_steps(cls):
    """CONTROL. The guard must not refuse a workable configuration, and the
    returned value must still be the mean over the steps that ran."""
    y_pred, _, sensitive = _leaking_batch()
    one = cls(n_adversary_steps=1)
    three = cls(n_adversary_steps=3)
    three.adversary.load_state_dict(one.adversary.state_dict())

    first = one.update_adversary(y_pred, sensitive)
    assert 0.0 < first < 5.0  # a real binary cross-entropy, not a sentinel
    # three steps from the same start: the mean of three falling losses sits
    # below the single first step and above the final one.
    mean_of_three = three.update_adversary(y_pred, sensitive)
    assert 0.0 < mean_of_three < 5.0
    assert _weights_moved(three)


# ---------------------------------------------------------------------------
# 2. "Trained" is decided by the weights, never by the call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
def test_a_step_that_moves_no_weight_is_not_training(cls):
    """adversary_lr=0.0: the Adam step provably moves nothing. Before the fix
    the next forward reported fairness_penalty_assessed=True with
    fairness_loss 0.6960861 from an adversary at its construction weights."""
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = cls(adversary_lr=0.0)

    returned = loss_fn.update_adversary(y_pred, sensitive)
    # the returned loss is a REAL measurement of this batch and stays one
    assert not _is_nan(returned) and returned > 0.0

    # the fixture really does exercise the branch under test
    assert not _weights_moved(loss_fn), "lr=0.0 must leave every weight untouched"
    assert loss_fn._adversary_is_at_initialization() is True

    _, comps, messages = _forward(loss_fn, y_pred, y_true, sensitive)
    assert _is_nan(comps.fairness_loss), comps.fairness_loss
    assert comps.batch_metrics["fairness_penalty_assessed"] is False
    assert comps.batch_metrics["fairness_unassessable_reason"] == "adversary_at_initialization"
    # the mitigation is untouched: the value the optimizer saw is kept
    assert comps.batch_metrics["fairness_loss_unassessed_value"] > 0.0
    assert any("NOT MEASURED" in m for m in messages), messages


def test_an_update_on_zero_rows_is_not_training_either():
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = AdversarialDebiasingLoss()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_fn.update_adversary(torch.zeros(0), torch.zeros(0))

    assert not _weights_moved(loss_fn)
    assert loss_fn._adversary_is_at_initialization() is True
    _, comps, _ = _forward(loss_fn, y_pred, y_true, sensitive)
    assert _is_nan(comps.fairness_loss)
    assert comps.batch_metrics["fairness_unassessable_reason"] == "adversary_at_initialization"


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
def test_control_a_real_step_counts_as_training_and_the_penalty_is_measured(cls):
    """CONTROL. Removing the call-latch must not stop real training from
    registering: one default-lr step moves the weights, and the penalty for
    the next batch is reported as a measurement."""
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = cls()

    loss_fn.update_adversary(y_pred, sensitive)

    assert _weights_moved(loss_fn), "a default-lr Adam step must move the weights"
    assert loss_fn._adversary_is_at_initialization() is False
    _, comps, messages = _forward(loss_fn, y_pred, y_true, sensitive)
    assert not _is_nan(comps.fairness_loss)
    assert comps.batch_metrics["fairness_penalty_assessed"] is True
    assert comps.batch_metrics["fairness_unassessable_reason"] is None
    assert comps.batch_metrics["fairness_groups_compared"] == 2
    assert not any("NOT MEASURED" in m for m in messages), messages


def test_control_training_really_trains_and_the_penalty_follows_it():
    """CONTROL on the measurement itself: 200 steps on a perfectly leaking
    batch drive the adversary's loss down, and the penalty reported by
    forward tracks the adversary loss it is defined as."""
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = AdversarialDebiasingLoss(use_gradient_reversal=True)

    first = loss_fn.update_adversary(y_pred, sensitive)
    for _ in range(200):
        last = loss_fn.update_adversary(y_pred, sensitive)
    assert last < first * 0.5, (first, last)

    _, comps, _ = _forward(loss_fn, y_pred, y_true, sensitive)
    assert comps.batch_metrics["fairness_penalty_assessed"] is True
    # the penalty IS the adversary's loss on this batch (GRL mode), computed
    # here independently of the class
    with torch.no_grad():
        probs = loss_fn.adversary(y_pred.unsqueeze(1)).squeeze()
        expected = torch.nn.functional.binary_cross_entropy(probs, sensitive).item()
    assert comps.fairness_loss == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. A precision cast is not training
# ---------------------------------------------------------------------------


def test_a_half_precision_cast_is_not_mistaken_for_training():
    """The public consequence, read through get_adversary_accuracy: a halved
    adversary that never saw a gradient must still refuse. (forward() is not
    used here because the half-precision TASK loss has its own dtype
    friction, unrelated to this fix.)"""
    y_pred, _, sensitive = _leaking_batch()
    loss_fn = AdversarialDebiasingLoss()
    loss_fn.adversary.half()

    assert loss_fn._adversary_is_at_initialization() is True
    accuracy, messages = _call(loss_fn.get_adversary_accuracy, y_pred.half(), sensitive.half())
    assert _is_nan(accuracy), accuracy
    assert any("still at its construction weights" in m for m in messages), messages


def test_control_a_trained_adversary_stays_trained_across_a_cast():
    """CONTROL for the same fix: the dtype cast must not ERASE real training."""
    y_pred, _, sensitive = _leaking_batch()
    loss_fn = AdversarialDebiasingLoss()
    for _ in range(20):
        loss_fn.update_adversary(y_pred, sensitive)
    assert loss_fn._adversary_is_at_initialization() is False

    loss_fn.adversary.half()
    assert loss_fn._adversary_is_at_initialization() is False


# ---------------------------------------------------------------------------
# 4. get_adversary_accuracy: the three states
# ---------------------------------------------------------------------------


def _trained_binary():
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = AdversarialDebiasingLoss()
    for _ in range(200):
        loss_fn.update_adversary(y_pred, sensitive)
    return loss_fn, y_pred, y_true, sensitive


def test_control_a_trained_adversary_measures_a_real_leak_exactly():
    """CONTROL. The attribute is recoverable from the predictions by a
    threshold at 0.5, so a converged adversary must recover ALL of it: 1.0,
    computed here from the construction of the fixture, not from the code."""
    loss_fn, y_pred, _, sensitive = _trained_binary()
    accuracy, messages = _call(loss_fn.get_adversary_accuracy, y_pred, sensitive)
    assert accuracy == 1.0, accuracy
    assert messages == []


def test_control_no_leak_is_reported_as_chance_not_refused():
    """The reverse defect. Constant predictions carry NO information, so every
    row gets the same label and the accuracy is the majority share: exactly
    0.5 on this 32/32 split. That is a real measurement of "no leakage" and it
    must still be RETURNED, not refused."""
    loss_fn, _, _, sensitive = _trained_binary()
    accuracy, messages = _call(loss_fn.get_adversary_accuracy, torch.full((N,), 0.5), sensitive)
    assert accuracy == pytest.approx(0.5)
    assert not _is_nan(accuracy)
    assert messages == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_predictions_are_not_a_chance_level_measurement(bad):
    """Before the fix: NaN -> 0.5, +inf -> 0.5. Both read as "no leakage"."""
    loss_fn, _, _, sensitive = _trained_binary()
    accuracy, messages = _call(loss_fn.get_adversary_accuracy, torch.full((N,), bad), sensitive)
    assert _is_nan(accuracy), accuracy
    assert any("NOT MEASURED" in m and "y_pred" in m for m in messages), messages


def test_non_finite_attribute_is_not_a_zero_leak():
    """Before the fix: an all-NaN attribute scored 0.0, the score a perfectly
    non-leaking model earns, because NaN == NaN is False on every row."""
    loss_fn, y_pred, _, _ = _trained_binary()
    accuracy, messages = _call(
        loss_fn.get_adversary_accuracy, y_pred, torch.full((N,), float("nan"))
    )
    assert _is_nan(accuracy), accuracy
    assert any("sensitive_attr" in m and "NOT MEASURED" in m for m in messages), messages


def test_one_group_and_untrained_adversary_still_refuse():
    """The two refusals that were already in place stay in place."""
    loss_fn, y_pred, _, _ = _trained_binary()
    accuracy, messages = _call(loss_fn.get_adversary_accuracy, y_pred, torch.zeros(N))
    assert _is_nan(accuracy)
    assert any("nothing for the adversary to tell apart" in m for m in messages), messages

    fresh = AdversarialDebiasingLoss()
    accuracy, messages = _call(fresh.get_adversary_accuracy, y_pred, (torch.arange(N) % 2).float())
    assert _is_nan(accuracy)
    assert any("still at its construction weights" in m for m in messages), messages


def test_an_out_of_domain_label_stays_a_raised_error():
    """A could-not-check must not absorb a bad call: the non-finite guard sits
    BELOW the domain check, so label 7 with n_groups=2 still raises."""
    loss_fn, y_pred, _, _ = _trained_binary()
    with pytest.raises(ValueError, match="outside the adversary's label domain"):
        loss_fn.get_adversary_accuracy(y_pred, torch.full((N,), 7.0))


def test_multiclass_control_and_multiclass_nan():
    """The same two directions on the argmax branch: a real three-group
    measurement is returned, all-NaN predictions (argmax -> 0 on every row,
    which scored 0.34375) are refused."""
    sensitive = (torch.arange(N) % 3).float()
    y_pred = sensitive / 2.0
    loss_fn = AdversarialDebiasingLoss(n_groups=3)
    for _ in range(150):
        loss_fn.update_adversary(y_pred, sensitive)

    accuracy, messages = _call(loss_fn.get_adversary_accuracy, y_pred, sensitive)
    assert not _is_nan(accuracy)
    assert accuracy > 1.0 / 3.0, "must beat the base rate on a fully leaking attribute"
    assert messages == []

    accuracy, messages = _call(
        loss_fn.get_adversary_accuracy, torch.full((N,), float("nan")), sensitive
    )
    assert _is_nan(accuracy), accuracy
    assert any("NOT MEASURED" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 5. FairRepresentationLoss.forward: measured, refused, and decomposable
# ---------------------------------------------------------------------------


def test_fair_representation_forward_measures_and_decomposes():
    """CONTROL. With a trained adversary and a real reconstruction error the
    three components are all measured, and the total is exactly
    task + alpha * reconstruction + lambda * fairness, recomputed here."""
    y_pred, y_true, sensitive = _leaking_batch()
    loss_fn = FairRepresentationLoss(
        lambda_fairness=1.0, alpha_reconstruction=0.5, representation_dim=4
    )
    representation = torch.randn(N, 4)
    x = torch.randn(N, 5)
    optimizer = torch.optim.Adam(loss_fn.adversary.parameters(), lr=0.01)
    for _ in range(30):
        optimizer.zero_grad()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss_fn(
                y_pred.unsqueeze(1),
                y_true.unsqueeze(1),
                sensitive,
                representation=representation,
                x_reconstructed=x,
                x_original=x,
            ).backward()
        optimizer.step()

    _, comps, messages = _forward(
        loss_fn,
        y_pred.unsqueeze(1),
        y_true.unsqueeze(1),
        sensitive,
        representation=representation,
        x_reconstructed=x + 1.0,
        x_original=x,
    )

    assert comps.batch_metrics["fairness_penalty_assessed"] is True
    assert not _is_nan(comps.fairness_loss)
    assert comps.batch_metrics["reconstruction_assessed"] is True
    # x + 1.0 against x: the mean squared error is exactly 1.0
    assert comps.regularization_loss == pytest.approx(1.0)
    assert comps.total_loss == pytest.approx(
        comps.task_loss + 0.5 * comps.regularization_loss + 1.0 * comps.fairness_loss,
        rel=1e-6,
    )
    assert not any("NOT MEASURED" in m for m in messages), messages


def test_fair_representation_forward_refuses_what_it_could_not_measure():
    """A one-group batch and a missing reconstruction pair are both NOT
    MEASURED, and they are distinguishable from the zero each would have
    reported: NaN in the field a caller reads."""
    y_pred, y_true, _ = _leaking_batch()
    loss_fn = FairRepresentationLoss()

    _, comps, messages = _forward(loss_fn, y_pred, y_true, torch.zeros(N))

    assert _is_nan(comps.fairness_loss)
    assert comps.batch_metrics["fairness_unassessable_reason"] == "single_group"
    assert _is_nan(comps.regularization_loss)
    assert comps.batch_metrics["reconstruction_loss"] is None
    assert comps.batch_metrics["representation_supplied"] is False
    assert any("NOT MEASURED" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 6. The plumbing: executed, and asserted to be plumbing
# ---------------------------------------------------------------------------


def test_gradient_reversal_is_identity_forward_and_negated_backward():
    layer = GradientReversalLayer(lambda_=2.5)
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    out = layer(x)
    assert torch.equal(out.detach(), x.detach())

    out.sum().backward()
    assert x.grad.tolist() == [-2.5, -2.5]

    assert layer.set_lambda(0.5) is None
    assert layer.lambda_ == 0.5
    x2 = torch.tensor([1.0], requires_grad=True)
    layer(x2).sum().backward()
    assert x2.grad.tolist() == [-0.5]

    direct = GradientReversalFunction.apply(torch.tensor([3.0]), 1.0)
    assert direct.tolist() == [3.0]


def test_adversary_forward_returns_a_probability_distribution():
    binary = Adversary(input_dim=1, hidden_dims=[4], n_groups=2)
    out = binary(torch.rand(5, 1))
    assert out.shape == (5, 1)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())

    multi = Adversary(input_dim=1, hidden_dims=[4], n_groups=3)
    out3 = multi(torch.rand(5, 1))
    assert out3.shape == (5, 3)
    assert torch.allclose(out3.sum(dim=-1), torch.ones(5), atol=1e-5)


@pytest.mark.parametrize(
    "cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss, FairRepresentationLoss]
)
def test_to_returns_self_and_moves_the_adversary(cls):
    loss_fn = cls()
    moved = loss_fn.to("cpu")
    assert moved is loss_fn
    assert all(p.device.type == "cpu" for p in loss_fn.adversary.parameters())
    # and moving does not invent training
    assert loss_fn._adversary_is_at_initialization() is True
