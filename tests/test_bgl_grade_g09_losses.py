"""BGL grade wave, batch G09: vfairness.in_processing.loss_functions
(base + adversarial).

THESE UNITS ARE MITIGATIONS, so the failure mode is inverted: a mitigation that
has been switched off, or turned around, reports a BETTER fairness number than
one that is working, because a model that is no longer being steered stops
being penalised for anything. None of the pins below reads a fairness metric as
its evidence. Each one reads the INTERVENTION's own parameter: is the reversal
still reversing, is the penalty still multiplied in, is the epoch the warmup is
compared against a number at all.

Every control asserts the healthy case's REAL number, derived from the batch
rather than quoted, so a guard that refuses everything cannot pass this file.
"""

import math
import warnings

import pytest

torch = pytest.importorskip("torch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
    Adversary,
    GradientReversalFunction,
    GradientReversalLayer,
    ProjectedAdversarialLoss,
)
from vfairness.in_processing.loss_functions.base import (  # noqa: E402
    BaseFairnessLoss,
    LossComponents,
    TrainingMetrics,
)
from vfairness.in_processing.loss_functions.fairness_losses import (  # noqa: E402
    DemographicParityLoss,
)


# A batch with a REAL demographic-parity disparity, so "the penalty was
# applied" and "the penalty was switched off" are distinguishable numbers.
def _batch():
    torch.manual_seed(7)
    y_pred = torch.tensor([0.9, 0.8, 0.85, 0.75, 0.3, 0.2, 0.25, 0.15])
    y_true = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sensitive = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return y_pred, y_true, sensitive


def _expected_disparity(y_pred, sensitive):
    """The soft demographic-parity penalty, recomputed from the batch itself.

    DemographicParityLoss documents and computes the MEAN absolute deviation of
    each group's positive rate from the overall rate, so this reimplements that
    definition rather than quoting the number it produced today.
    """
    overall = y_pred.mean()
    deviations = [(y_pred[sensitive == g].mean() - overall).abs() for g in torch.unique(sensitive)]
    return float(sum(deviations) / len(deviations))


def test_the_batch_really_carries_a_disparity():
    """The control that makes every assertion below able to disagree."""
    y_pred, _, sensitive = _batch()
    assert _expected_disparity(y_pred, sensitive) > 0.25


# ---------------------------------------------------------------------------
# LossComponents / TrainingMetrics: containers. Do they mint a value?
# ---------------------------------------------------------------------------


def test_loss_components_carries_what_it_was_given_and_invents_nothing():
    comps = LossComponents(total_loss=0.4566, task_loss=0.3567, fairness_loss=0.1999)
    assert comps.to_dict() == {
        "total_loss": 0.4566,
        "task_loss": 0.3567,
        "fairness_loss": 0.1999,
        "regularization_loss": 0.0,
        "auxiliary_losses": {},
        "batch_metrics": {},
    }
    # to_dict covers every field: a field added later and not serialised is the
    # boundary defect this repo has hit before.
    assert set(LossComponents.__dataclass_fields__) == set(comps.to_dict())
    # A NaN stays a NaN. It must NOT become 0.0, which is the score a perfect
    # predictor earns.
    nan_comps = LossComponents(
        total_loss=float("nan"), task_loss=float("nan"), fairness_loss=float("nan")
    )
    out = nan_comps.to_dict()
    assert all(math.isnan(out[k]) for k in ("total_loss", "task_loss", "fairness_loss"))
    # The mutable defaults are per-instance, not shared class state.
    first, second = LossComponents(1.0, 1.0, 1.0), LossComponents(1.0, 1.0, 1.0)
    first.auxiliary_losses["adv"] = 0.5
    first.batch_metrics["effective_lambda"] = 0.1
    assert second.auxiliary_losses == {} and second.batch_metrics == {}


def test_training_metrics_carries_what_it_was_given_and_invents_nothing():
    metrics = TrainingMetrics(
        epoch=3, avg_total_loss=0.4566, avg_task_loss=0.3567, avg_fairness_loss=0.1999
    )
    assert metrics.to_dict() == {
        "epoch": 3,
        "avg_total_loss": 0.4566,
        "avg_task_loss": 0.3567,
        "avg_fairness_loss": 0.1999,
        "fairness_metrics": {},
        "convergence_info": {},
    }
    assert set(TrainingMetrics.__dataclass_fields__) == set(metrics.to_dict())
    nan_metrics = TrainingMetrics(
        epoch=0,
        avg_total_loss=float("nan"),
        avg_task_loss=float("nan"),
        avg_fairness_loss=float("nan"),
    )
    assert math.isnan(nan_metrics.to_dict()["avg_task_loss"])
    assert nan_metrics.to_dict()["epoch"] == 0
    first, second = TrainingMetrics(0, 1.0, 1.0, 1.0), TrainingMetrics(0, 1.0, 1.0, 1.0)
    first.fairness_metrics["dp"] = 0.2
    first.convergence_info["converged"] = True
    assert second.fairness_metrics == {} and second.convergence_info == {}


def test_end_epoch_publishes_the_epoch_set_epoch_wrote():
    """The two units are connected: set_epoch's value is what gets published."""
    y_pred, y_true, sensitive = _batch()
    loss_fn = DemographicParityLoss(lambda_fairness=0.5)
    loss_fn.set_epoch(4)
    loss_fn(y_pred, y_true, sensitive)
    metrics = loss_fn.end_epoch()
    assert isinstance(metrics, TrainingMetrics)
    assert metrics.epoch == 4 and metrics.to_dict()["epoch"] == 4
    assert not math.isnan(metrics.avg_task_loss)


# ---------------------------------------------------------------------------
# BaseFairnessLoss: the abstract class that was not abstract
# ---------------------------------------------------------------------------


def test_the_abstract_base_refuses_to_be_instantiated():
    """@abstractmethod is enforced by ABCMeta, and nn.Module is not an ABC.

    Before the fix, BaseFairnessLoss(lambda_fairness=0.5) constructed,
    _compute_fairness_penalty returned None, and forward() died with
    "unsupported operand type(s) for *: 'float' and 'NoneType'" one layer away
    from the mistake.
    """
    # Derived, not quoted: the class really does still carry an abstract stub,
    # and Python really does still not enforce it here.
    assert BaseFairnessLoss._compute_fairness_penalty.__isabstractmethod__ is True
    assert not getattr(BaseFairnessLoss, "__abstractmethods__", ())
    with pytest.raises(TypeError, match="abstract"):
        BaseFairnessLoss(lambda_fairness=0.5)
    with pytest.raises(TypeError, match="abstract"):
        BaseFairnessLoss()

    # An external subclass that forgets the penalty AND does not override
    # forward is the realistic mistake, and it is refused at construction.
    class _ForgotEverything(BaseFairnessLoss):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _ForgotEverything()


def test_a_subclass_that_overrides_forward_is_allowed_and_the_stub_raises():
    """The refusal is narrowed to the case it can prove.

    A subclass that inherits the base without the penalty but overrides
    forward() is a legitimate use (tests/test_bgl2_base_and_entry_points does
    exactly that, so the base end_epoch aggregation can be executed at all),
    and it can never reach the broken arithmetic. Refusing it would break a
    working use to close a failure that cannot occur in it. The other door is
    closed at the point of use: the stub now RAISES instead of returning None.
    """
    from vfairness.in_processing.loss_functions.adversarial import (
        _AdversarialCoverageLoss,
    )
    from vfairness.in_processing.loss_functions.fairness_losses import (
        _CoverageTrackingLoss,
    )

    class _OverridesForward(BaseFairnessLoss):
        def forward(self, *args, **kwargs):
            return torch.tensor(1.0)

    allowed = _OverridesForward(track_metrics=True)
    assert allowed.end_epoch() is None  # the base aggregation still reachable
    # ...and the unimplemented penalty is now a NAMED error, not None.
    with pytest.raises(NotImplementedError, match="_compute_fairness_penalty"):
        allowed._compute_fairness_penalty(*_batch())

    # The two private mixins override forward too, so they construct; the stub
    # is what answers for them.
    for cls in (_CoverageTrackingLoss, _AdversarialCoverageLoss):
        assert "_compute_fairness_penalty" not in cls.__dict__
        assert cls.forward is not BaseFairnessLoss.forward
        with pytest.raises(NotImplementedError, match="_compute_fairness_penalty"):
            cls()._compute_fairness_penalty(*_batch())


def test_a_concrete_loss_still_constructs_and_measures_its_real_numbers():
    """The control. A guard that refused every construction would pass the two
    tests above and destroy the package."""
    y_pred, y_true, sensitive = _batch()
    loss_fn = DemographicParityLoss(lambda_fairness=0.5)
    total, comps = loss_fn(y_pred, y_true, sensitive, return_components=True)
    disparity = _expected_disparity(y_pred, sensitive)
    assert comps.fairness_loss == pytest.approx(disparity, abs=1e-5)
    assert comps.batch_metrics["effective_lambda"] == 0.5
    assert comps.batch_metrics["fairness_penalty_applied"] is True
    # The penalty is really IN the total loss, not merely reported beside it.
    assert float(total.detach()) == pytest.approx(comps.task_loss + 0.5 * disparity, abs=1e-5)
    assert float(total.detach()) > comps.task_loss


# ---------------------------------------------------------------------------
# BaseFairnessLoss.set_epoch: the setter that decides whether the mitigation
# is applied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epoch", [-1, -100, -0.5])
def test_a_negative_epoch_is_refused_because_it_switches_the_penalty_off(epoch):
    """warmup_epochs=0 means NO warmup, and a negative epoch is inside it.

    Measured before the fix with warmup_epochs=0 and lambda_fairness=0.5:
    effective_lambda 0.0 and fairness_penalty_applied False, explained only by
    a warning reading "epoch -1 of a 0-epoch warmup".
    """
    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=0)
    with pytest.raises(ValueError, match="at least 0"):
        loss_fn.set_epoch(epoch)
    # The refusal did not corrupt the state it refused to set.
    assert loss_fn._current_epoch == 0


@pytest.mark.parametrize("epoch", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_epoch_is_refused_because_it_skips_the_warmup_silently(epoch):
    """`epoch < warmup_epochs` is False for NaN and +inf, so a configured
    warmup was skipped with zero warnings and TrainingMetrics.epoch then
    carried the NaN."""
    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5)
    with pytest.raises(ValueError, match="finite"):
        loss_fn.set_epoch(epoch)
    assert loss_fn._current_epoch == 0


@pytest.mark.parametrize("epoch", [None, [1], {}, object()])
def test_an_uncomparable_epoch_is_refused_here_not_inside_the_loss(epoch):
    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5)
    with pytest.raises(ValueError, match="must be a number"):
        loss_fn.set_epoch(epoch)
    assert loss_fn._current_epoch == 0


def test_a_numeric_string_epoch_is_read_rather_than_refused():
    """Documented deliberately: float("3") succeeds, so "3" is coerced to the
    epoch the caller meant. It is comparable and whole, so it is not one of the
    two dangerous shapes above, and refusing it would be over-correction."""
    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5)
    loss_fn.set_epoch("3")
    assert loss_fn._current_epoch == 3 and isinstance(loss_fn._current_epoch, int)
    # The hostile spellings of a string are still refused.
    for bad in ("nan", "inf", "-1", "2.7"):
        with pytest.raises(ValueError):
            loss_fn.set_epoch(bad)


def test_a_fractional_epoch_is_refused():
    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5)
    with pytest.raises(ValueError, match="whole number"):
        loss_fn.set_epoch(2.7)


def test_set_epoch_still_schedules_the_warmup_it_exists_for():
    """The control, in both directions, on the value the optimizer sees."""
    y_pred, y_true, sensitive = _batch()
    disparity = _expected_disparity(y_pred, sensitive)

    loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=3)
    # Inside the warmup: the penalty is off, AND it says so.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_fn.set_epoch(1)
        during, comps = loss_fn(y_pred, y_true, sensitive, return_components=True)
    assert loss_fn._current_epoch == 1
    assert comps.batch_metrics["effective_lambda"] == 0.0
    assert comps.batch_metrics["fairness_penalty_applied"] is False
    assert comps.batch_metrics["fairness_penalty_not_applied_reason"] == "warmup"
    assert float(during.detach()) == pytest.approx(comps.task_loss, abs=1e-6)
    assert any("warmup" in str(w.message) for w in caught), [str(w.message) for w in caught]

    # Out of the warmup: the full penalty, and no suppression warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_fn.set_epoch(3)
        after, comps = loss_fn(y_pred, y_true, sensitive, return_components=True)
    assert comps.batch_metrics["effective_lambda"] == 0.5
    assert comps.batch_metrics["fairness_penalty_applied"] is True
    assert float(after.detach()) == pytest.approx(comps.task_loss + 0.5 * disparity, abs=1e-5)
    assert float(after.detach()) > float(during.detach())
    assert not [w for w in caught if "penalty" in str(w.message)]
    # numpy / torch scalar epochs are accepted, not refused as "not an int".
    loss_fn.set_epoch(torch.tensor(4))
    assert loss_fn._current_epoch == 4
    loss_fn.set_epoch(5.0)
    assert loss_fn._current_epoch == 5 and isinstance(loss_fn._current_epoch, int)


# ---------------------------------------------------------------------------
# GradientReversalFunction / GradientReversalLayer: is the reversal reversing?
# ---------------------------------------------------------------------------


def test_the_reversal_is_identity_forward_and_negated_backward():
    x = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
    layer = GradientReversalLayer(lambda_=2.5)
    out = layer(x)
    assert torch.equal(out.detach(), x.detach())
    out.sum().backward()
    # Derived from lambda_, not quoted: the gradient of sum() is 1 per element.
    assert x.grad.tolist() == [-2.5, -2.5, -2.5]

    # Applied directly, with a non-unit upstream gradient.
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)
    direct = GradientReversalFunction.apply(x2, 3.0)
    assert direct.tolist() == [1.0, 2.0]
    (direct * torch.tensor([2.0, 5.0])).sum().backward()
    assert x2.grad.tolist() == [-6.0, -15.0]

    # The forward really is a COPY, so a later in-place write on the output
    # cannot reach back into x.
    x3 = torch.tensor([4.0])
    assert GradientReversalFunction.apply(x3, 1.0) is not x3


@pytest.mark.parametrize("lambda_", [-1.0, -0.5, -1e-9])
def test_a_negative_reversal_scale_is_refused_because_it_un_reverses(lambda_):
    """Measured before the fix: lambda_=-1.0 gave grad [+1.0, +1.0], i.e. the
    adversary's gradient passed THROUGH, so descent was paid for making the
    protected attribute more predictable. Zero warnings."""
    with pytest.raises(ValueError, match="at least 0"):
        GradientReversalLayer(lambda_=lambda_)
    layer = GradientReversalLayer(lambda_=1.0)
    with pytest.raises(ValueError, match="at least 0"):
        layer.set_lambda(lambda_)
    # The refusal left the working scale in place rather than half-applying it.
    assert layer.lambda_ == 1.0
    x = torch.tensor([1.0], requires_grad=True)
    layer(x).sum().backward()
    assert x.grad.tolist() == [-1.0]


@pytest.mark.parametrize("lambda_", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_reversal_scale_is_refused(lambda_):
    with pytest.raises(ValueError, match="finite"):
        GradientReversalLayer(lambda_=lambda_)
    with pytest.raises(ValueError, match="finite"):
        GradientReversalLayer(lambda_=1.0).set_lambda(lambda_)


@pytest.mark.parametrize("lambda_", ["1.0x", None, [1.0]])
def test_an_unreadable_reversal_scale_is_refused(lambda_):
    with pytest.raises(ValueError, match="must be a number"):
        GradientReversalLayer(lambda_=lambda_)


def test_a_zero_reversal_scale_is_allowed_and_says_the_arm_is_off():
    """0.0 is a legal no-reversal baseline, so it is disclosed, not refused.
    Silence is what made it dangerous: the surrounding loss keeps reporting the
    adversary loss it measured."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = GradientReversalLayer(lambda_=0.0)
    assert layer.lambda_ == 0.0
    assert any("OFF" in str(w.message) for w in caught), [str(w.message) for w in caught]
    x = torch.tensor([1.0], requires_grad=True)
    layer(x).sum().backward()
    assert x.grad.tolist() == [0.0] or x.grad.tolist() == [-0.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GradientReversalLayer(lambda_=1.0).set_lambda(0.0)
    assert any("OFF" in str(w.message) for w in caught)


def test_a_usable_scale_is_accepted_from_numpy_and_torch_scalars():
    """The control for the float() read: isinstance(v, (int, float)) would
    refuse both of these while reading as caution."""
    np = pytest.importorskip("numpy")
    assert GradientReversalLayer(lambda_=np.float32(2.0)).lambda_ == 2.0
    assert GradientReversalLayer(lambda_=torch.tensor(3.0)).lambda_ == 3.0
    assert GradientReversalLayer(lambda_=np.int64(4)).lambda_ == 4.0


def test_the_guard_reaches_the_caller_that_supplies_the_scale():
    """ProjectedAdversarialLoss hands projection_strength straight to the layer.

    The two other classes in the module pin the scale at 1.0 on purpose (a
    recorded lambda-squared defect), so this is the only public door onto it,
    and it accepted -1.0 and nan before the fix.
    """
    for hostile in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ProjectedAdversarialLoss(projection_strength=hostile)
    # Control: the healthy construction still works and still reverses.
    healthy = ProjectedAdversarialLoss(projection_strength=1.5)
    assert healthy.gradient_reversal.lambda_ == 1.5
    x = torch.tensor([1.0], requires_grad=True)
    healthy.gradient_reversal(x).sum().backward()
    assert x.grad.tolist() == [-1.5]
    # And the classes that pin it at 1.0 are unaffected.
    assert AdversarialDebiasingLoss().gradient_reversal.lambda_ == 1.0


# ---------------------------------------------------------------------------
# Adversary: a network that could not be wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_groups", [1, 0, -1])
def test_an_adversary_with_fewer_than_two_groups_is_refused(n_groups):
    """n_groups=1 built a softmax over ONE logit. Measured on 20 rows before
    the fix: every output exactly 1.0, cross-entropy against the only label
    0.0, argmax accuracy 1.0. Those are the numbers a PERFECT adversary earns,
    from arithmetic that could not have returned anything else."""
    with pytest.raises(ValueError, match="at least 2"):
        Adversary(input_dim=1, hidden_dims=[4], n_groups=n_groups)
    # ...and the refusal reaches the loss that builds one.
    with pytest.raises(ValueError, match="at least 2"):
        AdversarialDebiasingLoss(n_groups=n_groups)


def test_the_degeneracy_the_refusal_is_about_is_real():
    """Derived, not asserted from memory: a softmax over one logit IS constant.

    If torch ever changed this, the refusal above would still be correct but
    this reasoning would need rewriting, so it is checked rather than quoted.
    """
    logits = torch.randn(20, 1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    assert bool((probs == 1.0).all())


def test_an_unknown_activation_is_refused_not_silently_replaced_with_relu():
    """`activations.get(activation, nn.ReLU())` built a ReLU network for
    activation='banana' with zero warnings. The adversary's activation decides
    how much leakage it can detect."""
    with pytest.raises(ValueError, match="unknown activation"):
        Adversary(input_dim=1, hidden_dims=[4], n_groups=2, activation="banana")
    with pytest.raises(ValueError, match="unknown activation"):
        Adversary(input_dim=1, hidden_dims=[4], n_groups=2, activation="ReLU")
    # Control: every name the class documents is still accepted, and the layer
    # it builds is the one that was asked for.
    expected = {
        "relu": "ReLU",
        "leaky_relu": "LeakyReLU",
        "elu": "ELU",
        "tanh": "Tanh",
    }
    for name, layer_type in expected.items():
        built = Adversary(input_dim=1, hidden_dims=[4], n_groups=2, activation=name)
        assert [type(m).__name__ for m in built.network][1] == layer_type


@pytest.mark.parametrize("dropout", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_dropout_is_refused_rather_than_silently_dropped(dropout):
    """`if dropout > 0` is False for NaN, so dropout=nan built a network with
    no Dropout layer and said nothing."""
    with pytest.raises(ValueError, match="finite"):
        Adversary(input_dim=1, hidden_dims=[4], n_groups=2, dropout=dropout)


def test_the_adversary_still_builds_and_predicts_a_distribution():
    """The control. Every refusal above would also pass for a constructor that
    refused everything."""
    binary = Adversary(input_dim=1, hidden_dims=[8, 4], n_groups=2)
    out = binary(torch.rand(6, 1))
    assert out.shape == (6, 1)
    assert bool(((out > 0.0) & (out < 1.0)).all())
    assert binary.output_activation == "sigmoid"
    assert float(out.detach().std()) > 0.0  # not a constant

    multi = Adversary(input_dim=3, hidden_dims=[8], n_groups=4, dropout=0.25)
    out4 = multi(torch.rand(6, 3))
    assert out4.shape == (6, 4)
    assert torch.allclose(out4.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert multi.output_activation == "softmax"
    # A softmax over FOUR logits is not constant, which is the whole point of
    # refusing one.
    assert float(out4.detach().std()) > 0.0
    assert "Dropout" in [type(m).__name__ for m in multi.network]
    # A dropout of exactly 0.0 (the default) adds no layer, as before.
    assert "Dropout" not in [
        type(m).__name__ for m in Adversary(input_dim=1, hidden_dims=[4], n_groups=2).network
    ]
