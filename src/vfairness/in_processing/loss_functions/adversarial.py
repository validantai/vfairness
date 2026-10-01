"""
Adversarial Debiasing Loss Functions.

This module implements adversarial training approaches for fairness, where
an adversary tries to predict the sensitive attribute from model predictions,
and the main model is trained to produce predictions that don't leak sensitive
attribute information.

Key Concepts:
    - **Predictor Network**: The main model predicting the target variable
    - **Adversary Network**: A discriminator trying to recover sensitive attributes
    - **Gradient Reversal**: Technique to train predictor to fool the adversary

Loss Functions Implemented:
    1. AdversarialDebiasingLoss: Standard adversarial debiasing approach
    2. FairAdversarialLoss: Improved version with gradient reversal
    3. ProjectedAdversarialLoss: Projects gradients for stability

Mathematical Formulation:
    L_total = L_task(ŷ, y) - λ * L_adversary(â, a)

    where:
    - L_task: prediction loss
    - L_adversary: adversary's loss predicting sensitive attribute
    - λ: trade-off parameter (negative sign creates adversarial training)

References:
    - Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning
    - Madras et al. (2018): Learning Adversarially Fair and Transferable Representations
    - Beutel et al. (2017): Data Decisions and Theoretical Implications for Fairness
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .base import (
    TORCH_AVAILABLE,
    BaseLossType,
    LossComponents,
    check_torch_available,
)
from .fairness_losses import _CoverageTrackingLoss

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function


class _AdversarialCoverageLoss(_CoverageTrackingLoss):
    """Coverage tracking plus "has this adversary ever been trained?".

    An adversarial fairness penalty is the adversary's own loss, so it is a
    measurement of leakage only to the extent the adversary can predict the
    attribute. At random initialisation it cannot, and the number it produces
    is a property of the initialisation rather than of the data: measured on
    ONE constant-attribute batch across five seeds, fairness_loss came back as
    0.6256, 0.7104, 0.7444, 0.6986, 0.6956, a spread of 0.119 straddling the
    healthy two-group value of 0.6950.

    The adversary counts as trained once ANY of its weights differ from the
    values it was constructed with. That covers both documented routes
    (``update_adversary()``, and a caller stepping an optimizer over these
    parameters after ``loss.backward()``), and it is latched, so the tensor
    comparison stops running once training has started.
    """

    _adversary_ever_trained: bool = False
    # Every subclass builds one; declared here so the helpers below type-check
    # against nn.Module rather than nn.Module's Tensor-or-Module __getattr__.
    adversary: "nn.Module"
    _adversary_init_state: List["torch.Tensor"]

    def _init_adversary_tracking(self) -> None:
        """Snapshot the adversary's initial weights. Call after building it."""
        self._adversary_ever_trained = False
        self._adversary_init_state = [p.detach().clone() for p in self.adversary.parameters()]

    def _adversary_weights_not_finite(self) -> int:
        """How many of the adversary's parameter tensors hold a NaN or an inf.

        A BROKEN ADVERSARY IS NOT A NON-LEAKING MODEL (BGL wave 4, 2026-09-30).
        Every guard in this file checked the INPUTS handed to the adversary and
        none checked the adversary itself, so a network whose every weight is NaN
        went on publishing the same chance rate those guards exist to refuse. It
        is reachable through this library's own public API with no hand-mutation
        of anything: the learning rate is a constructor argument, and a large one
        diverges the network to NaN inside ``update_adversary``.

        Measured on AdversarialDebiasingLoss(n_groups=3), 39 rows in three
        groups, 50 calls to update_adversary and then get_adversary_accuracy:

            adversary_lr=0.01  weights finite, last loss 0.5518559813499451
                -> accuracy 1.0,                 0 warnings   (the control)
            adversary_lr=1e8   weights finite, last loss 1.5514447689056396
                -> accuracy 0.0,                 0 warnings   (a real number:
                   the network is trained and wrong)
            adversary_lr=1e14  weights NOT FINITE, last loss nan for fifty
                consecutive steps
                -> accuracy 0.3333333432674808,  0 warnings

        One third is chance for three groups, published as a leakage
        measurement out of a network with no readable parameter at all, and this
        file's own comment calls that reading "no leakage detected".
        ``_adversary_is_at_initialization()`` cannot catch it either, and says so
        in its own record: the NaN weights DIFFER from the snapshot, so the
        network counts as TRAINED.
        """
        return sum(
            1 for p in self.adversary.parameters() if not bool(torch.isfinite(p.detach()).all())
        )

    def _report_adversary_step_loss(self, mean_loss: float) -> float:
        """The mean adversary loss over the steps that ran, and warn if it broke.

        ``update_adversary`` is the entry point that DESTROYS the adversary when
        the step diverges, and it returned the damage as an ordinary float. On
        AdversarialDebiasingLoss(n_groups=3, adversary_lr=1e14) it returned nan
        for FIFTY consecutive calls without one warning, and the network it left
        behind then published a chance leakage rate as a measurement (see
        :meth:`_adversary_weights_not_finite`). The loss is still returned
        unchanged, because nan IS what the step produced; what was missing is
        anybody saying that the adversary is now unusable.
        """
        if mean_loss != mean_loss or mean_loss in (float("inf"), float("-inf")):
            n_broken = self._adversary_weights_not_finite()
            warnings.warn(
                f"{type(self).__name__}.update_adversary: the adversary training "
                f"step produced a non-finite loss ({mean_loss!r}) and "
                f"{n_broken} of the adversary's parameter tensor(s) now hold NaN "
                f"or inf. The network is broken and every leakage number read "
                f"from it after this point is NOT A MEASUREMENT, even though it "
                f"counts as trained (its weights differ from the construction "
                f"snapshot). Lower adversary_lr, or rebuild the loss.",
                UserWarning,
                stacklevel=2,
            )
        return mean_loss

    def _adversary_is_at_initialization(self) -> bool:
        """True while every adversary weight still holds its initial value.

        With no snapshot to compare against (an adversary built outside
        ``_init_adversary_tracking``) the loop below is empty and the answer
        is True, the could-not-check direction: an unverified adversary is
        reported as untrained rather than credited with training nobody saw.
        """
        if self._adversary_ever_trained:
            return False
        snapshot = getattr(self, "_adversary_init_state", None) or ()
        for param, initial in zip(self.adversary.parameters(), snapshot):
            # The snapshot is cast to the parameter's CURRENT dtype as well as
            # its device, because a precision change is not training. Before
            # that cast, a caller doing ``loss_fn.adversary.half()`` (or
            # ``.to(torch.float16)``) rounded every weight, ``torch.equal``
            # reported a difference, and an adversary that had never seen a
            # gradient was credited with training: the next penalty came back
            # fairness_penalty_assessed=True while the network still held its
            # construction weights.
            baseline = initial.to(device=param.device, dtype=param.dtype)
            if param.shape != baseline.shape or not bool(torch.equal(param.detach(), baseline)):
                self._adversary_ever_trained = True
                return False
        return True

    def _check_adversary_steps(self) -> None:
        """Raise when the configuration asks for zero adversary training steps.

        ``range(0)`` is empty, so the training loop body never runs: nothing is
        trained and no adversary loss is ever computed.
        :class:`ProjectedAdversarialLoss` then returned ``0.0 / max(1, 0)``,
        i.e. 0.0, which on this scale is the score a PERFECT adversary earns
        (cross-entropy zero, the attribute fully recovered), and
        :class:`AdversarialDebiasingLoss` raised ZeroDivisionError only AFTER
        marking the adversary trained, so a caller that swallowed the error was
        left with an untrained adversary reporting its initialisation as a
        measurement. A configuration that cannot train is a caller error, not a
        loss of zero.
        """
        steps = getattr(self, "n_adversary_steps", 1)
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(
                f"{type(self).__name__}: n_adversary_steps must be an integer "
                f"of at least 1, got {steps!r}. With no step the adversary is "
                f"never trained and no adversary loss exists; returning 0.0 "
                f"would be the value a perfect adversary earns."
            )

    def _check_sensitive_domain(self, sensitive_attr: "torch.Tensor") -> None:
        """Raise on attribute values the adversary has no output unit for.

        A label outside ``[0, n_groups - 1]`` is a CALLER ERROR, and it used
        to be loud: ``F.binary_cross_entropy`` raised RuntimeError and
        ``F.cross_entropy`` raised IndexError. The coverage guard now sits in
        front of both, so without this check a constant ``7.0`` with
        ``n_groups=2`` is swallowed into the ordinary ``single_group``
        refusal and the caller never learns the label was nonsense. A
        could-not-check must not absorb a bad call.
        """
        n_groups = getattr(self, "n_groups", None)
        if not isinstance(n_groups, int) or n_groups < 2:
            return
        if sensitive_attr.numel() == 0:
            return
        low = float(sensitive_attr.min().item())
        high = float(sensitive_attr.max().item())
        if low < 0.0 or high > float(n_groups - 1):
            raise ValueError(
                f"{type(self).__name__}: sensitive_attr holds values outside the "
                f"adversary's label domain [0, {n_groups - 1}] (observed min "
                f"{low}, max {high}). The adversary was built with "
                f"n_groups={n_groups} and has no output for these labels, so no "
                f"leakage can be measured against them. Pass integer group "
                f"labels in range, or construct the loss with the right n_groups."
            )

    def _check_adversary_training_inputs(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> int:
        """Guard an adversary training step. Returns the row count.

        A NON-FINITE INPUT DESTROYS THE ADVERSARY AND IS THEN REPORTED AS
        TRAINING. With ``n_groups=2`` torch itself refuses a NaN
        (``binary_cross_entropy`` validates its input range), but the
        multiclass path has no such validation, so one NaN prediction in 30
        rows used to run all the way through. Measured before this guard on
        ``AdversarialDebiasingLoss(n_groups=3)``, with no warning anywhere:

            update_adversary -> returned nan, every adversary weight NaN,
            _adversary_is_at_initialization() False (the NaN weights differ
            from the snapshot, so the network counted as TRAINED), and the
            next forward on a HEALTHY batch reported fairness_loss nan with
            fairness_penalty_assessed=True and reason None, while
            get_adversary_accuracy returned 0.3333 because argmax of an
            all-NaN row is 0.

        That is the worst reading in the file: a permanently broken adversary
        credited with training, publishing a NaN as a measured penalty. The
        step is refused before it can run rather than after.

        The label-domain check runs here too. ``update_adversary`` is the other
        entry point that feeds the adversary, and with ``n_groups=2``
        ``binary_cross_entropy`` does NOT validate its target, so an
        out-of-domain label trained the network toward a value it has no
        output for and the error only surfaced at the next ``forward``.
        """
        self._check_sensitive_domain(sensitive_attr)

        non_finite = []
        if not bool(torch.isfinite(y_pred).all()):
            non_finite.append("y_pred")
        if not bool(torch.isfinite(sensitive_attr).all()):
            non_finite.append("sensitive_attr")
        if non_finite:
            raise ValueError(
                f"{type(self).__name__}.update_adversary: "
                f"{' and '.join(non_finite)} holds non-finite values (NaN or "
                f"inf). A training step on them writes NaN into every adversary "
                f"weight, after which the network is permanently broken AND "
                f"counts as trained, so its loss is reported as a measured "
                f"leakage. Drop or impute the unreadable rows first."
            )

        # atleast_1d, not `... else 1`: see the same line in
        # BaseFairnessLoss._task_loss_coverage. A 0-dim prediction is one row,
        # and saying so with the torch primitive leaves no literal that reads
        # (or scans) as a neutral fallback.
        n_rows = int(torch.atleast_1d(y_pred).shape[0])
        if n_rows == 0:
            warnings.warn(
                f"{type(self).__name__}.update_adversary: the batch has 0 rows, so "
                f"no adversary loss exists and nothing is trained. The returned "
                f"value is NaN (NOT MEASURED), not the 0.0 a perfect adversary "
                f"earns; the adversary stays at its construction weights and the "
                f"next penalty is reported as unmeasurable.",
                UserWarning,
                stacklevel=3,
            )
        return n_rows

    def _mark_adversarial_coverage(
        self,
        sensitive_attr: "torch.Tensor",
        **detail: Any,
    ) -> bool:
        """Record the coverage of an adversarial penalty. True if measurable.

        THREE things make the penalty unmeasurable: a sensitive attribute with
        one distinct value (the adversary's target is constant, so there is no
        leakage to detect), an adversary still at its random initialisation, and
        an adversary whose weights are not finite, which is the door
        :meth:`_adversary_weights_not_finite` records. The third is checked here,
        above the branch selection, because all three classes in this module ask
        this one helper before choosing between a gradient-reversal arm and an
        alternating arm, and a guard inside either arm would leave the other one
        publishing the same NaN.

        Before it, on AdversarialDebiasingLoss(n_groups=3) after fifty
        update_adversary steps at adversary_lr=1e14 (every weight NaN), a forward
        on a HEALTHY batch reported fairness_loss nan with
        fairness_penalty_assessed True and fairness_unassessable_reason None. The
        comment on :meth:`_check_adversary_training_inputs` recorded exactly that
        state in 2026-09 and the fix that followed guarded the update_adversary
        INPUTS only.

        The label-domain check runs FIRST, above every branch and above the
        refusal itself: a bad call must stay a raised error rather than be
        reported as an unmeasurable batch.
        """
        self._check_sensitive_domain(sensitive_attr)
        n_groups = int(torch.unique(sensitive_attr).numel())
        untrained = self._adversary_is_at_initialization()
        n_broken = self._adversary_weights_not_finite()
        shared: Dict[str, Any] = {
            "fairness_groups_total": n_groups,
            "adversary_at_initialization": untrained,
            "adversary_weights_not_finite": n_broken,
            **detail,
        }
        measurable = True
        if n_broken:
            measurable = False
            self._mark_fairness_unassessable(
                "adversary_not_finite",
                f"{n_broken} of the adversary's parameter tensor(s) hold NaN or "
                f"inf, so the network cannot predict anything and its loss is not "
                f"a leakage measurement. A diverged adversary reads as a model "
                f"that leaks nothing, which is the opposite of what it is: "
                f"nothing was measured. Rebuild the loss, or train the adversary "
                f"with a learning rate that does not diverge.",
                fairness_groups_compared=0,
                **shared,
            )
        elif n_groups < 2:
            measurable = False
            self._mark_fairness_unassessable(
                "single_group",
                f"the sensitive attribute has {n_groups} distinct value(s) in this "
                f"batch, so the adversary has nothing to distinguish and its loss "
                f"measures no leakage.",
                fairness_groups_compared=0,
                **shared,
            )
        elif untrained:
            measurable = False
            self._mark_fairness_unassessable(
                "adversary_at_initialization",
                "the adversary is still at its construction weights, so its loss "
                "reports the random initialisation rather than any leakage. Train "
                "it with update_adversary() on detached predictions (or step an "
                "optimizer over its parameters) before reading this penalty.",
                fairness_groups_compared=n_groups,
                **shared,
            )
        else:
            self._mark_fairness_assessed(fairness_groups_compared=n_groups, **shared)
        return measurable

    def _penalty_has_no_quantity(self, sensitive_attr: "torch.Tensor") -> bool:
        """True when the penalty tensor must be a gradient-free zero.

        Two of the three refusals above mean the adversary's loss is not a
        quantity at all: a single-group batch gives the adversary nothing to tell
        apart, and an adversary with non-finite weights returns NaN, which is not
        a smaller penalty but an absent one. A NaN reaching ``total_loss`` is not
        neutral either: it poisons the gradient of every parameter at once, and
        it compares False against every threshold it is later tested against, so
        it SUPPRESSES a finding rather than raising one.

        The untrained-adversary refusal is deliberately NOT in here and keeps its
        real number: the alternating scheme has to start somewhere, and there it
        is the REPORTED measurement that is withheld.
        """
        return (
            int(torch.unique(sensitive_attr).numel()) < 2
            or self._adversary_weights_not_finite() > 0
        )


if TORCH_AVAILABLE:

    class GradientReversalFunction(Function):
        """
        Gradient Reversal Layer for adversarial training.

        During forward pass, acts as identity.
        During backward pass, negates the gradient (multiplies by -λ).
        """

        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None

    class GradientReversalLayer(nn.Module):
        """
        Module wrapper for gradient reversal.

        Args:
            lambda_: Scale of the REVERSED gradient. Must be finite and at
                least 0. 0.0 is allowed as a deliberate no-reversal baseline
                and is disclosed with a warning, because it switches the
                adversarial arm off while every loss around it keeps reporting
                the leakage it measured.

        A NEGATIVE SCALE DOES NOT REVERSE THE GRADIENT, IT PASSES IT THROUGH
        (BGL grade wave, 2026-09-30). ``GradientReversalFunction.backward``
        returns ``-lambda_ * grad_output``, so the SIGN of ``lambda_`` decides
        whether the encoder is steered away from the adversary or TOWARD it.
        Measured on ``torch.tensor([1.0, 2.0])`` with an upstream gradient of 1:

            lambda_ = 1.0  -> grad [-1.0, -1.0]   reversed, the intervention
            lambda_ = 2.5  -> grad [-2.5, -2.5]   reversed and scaled
            lambda_ = 0.0  -> grad [-0.0, -0.0]   the arm is OFF, 0 warnings
            lambda_ = -1.0 -> grad [ 1.0,  1.0]   NOT reversed: gradient
                              descent is now PAID for making the protected
                              attribute MORE predictable from the
                              representation, i.e. for increasing leakage
            lambda_ = -0.5 -> grad [ 0.5,  0.5]   the same, half strength
            lambda_ = nan  -> grad [ nan,  nan]   poisons every parameter at
                              once, and nan compares False against every
                              threshold it is later tested against, so it
                              SUPPRESSES a finding rather than raising one
            lambda_ = inf  -> grad [-inf, -inf]

        and none of the four hostile values produced a single warning. The
        value is caller-supplied through a documented public parameter:
        ``ProjectedAdversarialLoss(projection_strength=...)`` is handed
        straight to this constructor (the two other classes in this module
        pin it at 1.0 on purpose, see their comments), so
        ``projection_strength=-1.0`` inverted the whole intervention and
        ``projection_strength=nan`` destroyed it, both in silence.

        This is the same defect, in the same direction, as the
        ``lambda_fairness`` guard in :class:`BaseFairnessLoss` (which measured
        a total loss of 0.2567 against a task loss of 0.3567 at
        ``lambda_fairness=-0.5``) and as ``_check_adversary_steps`` in this
        module. The contract is deliberately identical to
        ``lambda_fairness``: refuse non-finite and negative, allow 0.0 and say
        so out loud. A NEUTERED MITIGATION REPORTS A BETTER FAIRNESS NUMBER,
        NOT A WORSE ONE, which is why silence here is the dangerous answer.
        """

        def __init__(self, lambda_: float = 1.0):
            super().__init__()
            self.lambda_ = self._checked_lambda(lambda_)

        @staticmethod
        def _checked_lambda(lambda_):
            """Refuse a reversal scale that inverts or destroys the reversal.

            Read through ``float()`` rather than ``isinstance(v, (int, float))``:
            that test REJECTS ``np.float32``/``np.float64`` and a 0-dim torch
            scalar, which would discard a real caller value while reading as
            caution.
            """
            try:
                value = float(lambda_)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"GradientReversalLayer: lambda_ must be a number, got "
                    f"{lambda_!r}. It scales the REVERSED gradient, so a value "
                    f"that cannot be read as a number leaves the adversarial "
                    f"arm in an unknown state."
                ) from exc
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(
                    f"GradientReversalLayer: lambda_ must be finite, got "
                    f"{lambda_!r}. The backward pass returns -lambda_ * grad, so "
                    f"a non-finite scale writes NaN or inf into every upstream "
                    f"parameter gradient while the surrounding loss keeps "
                    f"reporting the leakage it measured."
                )
            if value < 0:
                raise ValueError(
                    f"GradientReversalLayer: lambda_ must be at least 0, got "
                    f"{lambda_!r}. The backward pass returns -lambda_ * grad, so a "
                    f"negative scale does not reverse the adversary's gradient, it "
                    f"PASSES IT THROUGH: minimising the loss then trains the "
                    f"representation to make the protected attribute MORE "
                    f"predictable. Pass 0.0 for a deliberate no-reversal baseline."
                )
            if value == 0:
                warnings.warn(
                    "GradientReversalLayer: lambda_ is 0.0, so the reversed "
                    "gradient is multiplied out and the adversarial arm is OFF. "
                    "This is a legal no-reversal baseline; it is named here "
                    "because the surrounding loss still reports the adversary "
                    "loss it measured, which reads as an intervention that was "
                    "applied. A switched-off mitigation makes the fairness "
                    "numbers LOOK BETTER, not worse, because the model is no "
                    "longer being steered.",
                    UserWarning,
                    stacklevel=3,
                )
            return value

        def forward(self, x):
            return GradientReversalFunction.apply(x, self.lambda_)

        def set_lambda(self, lambda_: float):
            """Re-scale the reversal. Same contract as the constructor.

            Guarded here as well as in ``__init__`` because this is the other
            door onto the same field: a caller that constructs with 1.0 and
            then calls ``set_lambda(-1.0)`` inverted the intervention exactly
            as the constructor did.
            """
            self.lambda_ = self._checked_lambda(lambda_)

    class Adversary(nn.Module):
        """
        Adversary network for predicting sensitive attributes.

        This network attempts to predict the sensitive attribute from
        the main model's predictions or representations.

        Args:
            input_dim: Dimension of input (predictions or features)
            hidden_dims: List of hidden layer dimensions
            n_groups: Number of sensitive attribute groups
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            dropout: Dropout probability

        Example:
            >>> adversary = Adversary(input_dim=1, hidden_dims=[32, 16], n_groups=2)
            >>> sensitive_pred = adversary(model_predictions)

        Raises:
            ValueError: for ``n_groups < 2``, an unknown ``activation``, or a
                non-finite ``dropout``. See the comments on each check: all
                three used to be accepted in silence, and the first of them
                built a network whose output was the PERFECT-adversary score
                and could not be anything else.
        """

        #: The activation names this class implements. An exact subscript, not a
        #: defaulted get: see the comment in __init__.
        _ACTIVATIONS = ("relu", "leaky_relu", "elu", "tanh")

        def __init__(
            self,
            input_dim: int = 1,
            hidden_dims: Optional[List[int]] = None,
            n_groups: int = 2,
            activation: str = "relu",
            dropout: float = 0.0,
        ):
            super().__init__()

            # A SINGLE-OUTPUT SOFTMAX IS A CONSTANT 1.0, NOT A PREDICTION
            # (BGL grade wave, 2026-09-30). n_groups=1 took the else branch
            # below, built nn.Linear(prev_dim, 1) and applied softmax over a
            # length-1 dimension, which is identically 1.0 whatever the weights
            # are. Measured on 20 rows, Adversary(input_dim=1, hidden_dims=[4],
            # n_groups=1): every output exactly 1.0, cross-entropy against the
            # only available label 0.0, argmax accuracy 1.0. That is the score a
            # PERFECT adversary earns, i.e. total leakage, produced by
            # arithmetic that could not have returned anything else, and
            # AdversarialDebiasingLoss(n_groups=1) constructed it with zero
            # warnings. n_groups=0 built a (N, 0) output instead.
            #
            # Refused rather than disclosed because n_groups is the caller's own
            # argument and not a property of the data: with fewer than two
            # groups there is no attribute for an adversary to distinguish, so
            # there is nothing to measure and nothing to degrade to. The
            # single-group DATA case is a different question and is already
            # answered three-state by _mark_adversarial_coverage.
            try:
                n_groups_value = int(n_groups)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Adversary: n_groups must be an integer, got {n_groups!r}."
                ) from exc
            if n_groups_value < 2:
                raise ValueError(
                    f"Adversary: n_groups must be at least 2, got {n_groups!r}. An "
                    f"adversary predicts WHICH group a row belongs to, so with "
                    f"fewer than two there is nothing to predict: n_groups=1 "
                    f"builds a softmax over one logit, whose output is exactly "
                    f"1.0 for every row, giving a cross-entropy of 0.0 and an "
                    f"accuracy of 1.0. Those are the numbers a perfect adversary "
                    f"earns and they would be reported as total leakage measured "
                    f"from a network that cannot be wrong."
                )
            n_groups = n_groups_value

            if hidden_dims is None:
                hidden_dims = [32, 16]

            layers: List[nn.Module] = []
            prev_dim = input_dim

            activations = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.2),
                "elu": nn.ELU(),
                "tanh": nn.Tanh(),
            }

            # AN UNKNOWN ACTIVATION WAS SILENTLY SUBSTITUTED WITH ReLU. The
            # lookup was `activations.get(activation, nn.ReLU())`, so
            # Adversary(activation="banana") built a ReLU network with no
            # warning of any kind: measured 2026-09-30, layers ['Linear',
            # 'ReLU', 'Linear'] and 0 warnings. The adversary's activation
            # decides how much leakage it can detect, so this is a silently
            # different measurement, not a cosmetic substitution. analyzer.py
            # in post_processing carries the identical carve-out for a mistyped
            # objective ("the lookup below is an exact subscript, not a
            # defaulted get") for the same reason.
            if activation not in activations:
                raise ValueError(
                    f"Adversary: unknown activation {activation!r}. Supported: "
                    f"{sorted(activations)}. It was previously substituted with "
                    f"'relu' in silence, which changes how much leakage the "
                    f"adversary can detect while the caller believes it "
                    f"configured something else."
                )

            # `if dropout > 0` IS FALSE FOR NaN, so dropout=nan omitted the
            # Dropout layer entirely and said nothing (measured: layers
            # ['Linear', 'ReLU', 'Linear'], 0 warnings). nan defeats a
            # threshold guard silently in whichever direction the comparison
            # happens to run; nn.Dropout itself validates the [0, 1] range but
            # never sees the value.
            try:
                dropout_value = float(dropout)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Adversary: dropout must be a number, got {dropout!r}.") from exc
            if dropout_value != dropout_value or dropout_value in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(
                    f"Adversary: dropout must be finite, got {dropout!r}. "
                    f"`dropout > 0` is False for NaN, so a non-finite value "
                    f"silently built a network with no dropout at all."
                )
            dropout = dropout_value

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activations[activation])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            if n_groups == 2:
                layers.append(nn.Linear(prev_dim, 1))
                self.output_activation = "sigmoid"
            else:
                layers.append(nn.Linear(prev_dim, n_groups))
                self.output_activation = "softmax"

            self.network = nn.Sequential(*layers)
            self.n_groups = n_groups

        def forward(self, x):
            logits = self.network(x)

            if self.output_activation == "sigmoid":
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)


class AdversarialDebiasingLoss(_AdversarialCoverageLoss):
    """
    Adversarial Debiasing Loss Function.

    Implements adversarial training for fairness where an adversary
    attempts to predict the sensitive attribute from model predictions,
    and the main model is trained to prevent this leakage.

    The training objective for the main model is:
        min L_task - λ * L_adversary

    The adversary is trained separately to maximize its prediction accuracy.

    Args:
        lambda_fairness: Weight for adversarial penalty (default: 0.1)
        adversary_hidden_dims: Hidden layer sizes for adversary
        n_groups: Number of sensitive attribute groups
        adversary_lr: Learning rate for adversary (if trained separately)
        n_adversary_steps: Number of adversary updates per main update
        use_gradient_reversal: Whether to use gradient reversal layer
        base_loss: Base task loss type
        track_metrics: Whether to track training metrics

    Example:
        >>> # Create loss with embedded adversary
        >>> loss_fn = AdversarialDebiasingLoss(
        ...     lambda_fairness=1.0,
        ...     adversary_hidden_dims=[64, 32],
        ...     n_groups=2
        ... )
        >>>
        >>> # In training loop:
        >>> for batch in dataloader:
        ...     x, y, sensitive = batch
        ...     y_pred = model(x)
        ...
        ...     # Main model update
        ...     loss = loss_fn(y_pred, y, sensitive)
        ...     loss.backward()
        ...     main_optimizer.step()
        ...
        ...     # Adversary update (required in BOTH modes)
        ...     loss_fn.update_adversary(y_pred.detach(), sensitive)

    Note:
        The adversary is trained ONLY by calling ``update_adversary()`` on
        detached predictions, in both modes. With use_gradient_reversal=True
        the shared backward pass also computes correctly-signed gradients on
        the adversary's weights, but nothing steps them unless you step
        ``adversary_optimizer`` yourself after ``loss.backward()``; the
        earlier docstring claim that a single backward pass trains both
        networks left the adversary at random initialization. Meaningful
        debiasing typically needs lambda_fairness near 1.0.

    .. note::
        Two defects were fixed here (2026-08-22 audit). (1) With
        use_gradient_reversal=False the penalty was returned as +adv_loss
        while ``BaseFairnessLoss.forward`` computes task + lambda * penalty,
        so gradient descent MINIMIZED the adversary's loss through y_pred,
        training the predictor to help the adversary and measurably
        amplifying bias; the penalty is now the negated adversary loss
        against a frozen adversary, matching min L_task - lambda *
        L_adversary. (2) The gradient reversal layer was constructed with
        scale lambda_fairness while forward multiplies by lambda again, so
        the predictor saw lambda squared (default 0.1 became an effective
        0.01); the reversal scale is now fixed at 1.0 so lambda applies
        exactly once.

    References:
        - Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: adversarial_debiasing. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        adversary_hidden_dims: Optional[List[int]] = None,
        n_groups: int = 2,
        adversary_lr: float = 0.001,
        n_adversary_steps: int = 1,
        use_gradient_reversal: bool = True,
        base_loss: Union[str, BaseLossType] = BaseLossType.BINARY_CROSS_ENTROPY,
        reduction: Literal["mean", "sum", "none"] = "mean",
        track_metrics: bool = True,
        warmup_epochs: int = 0,
    ):
        super().__init__(
            lambda_fairness=lambda_fairness,
            base_loss=base_loss,
            reduction=reduction,
            track_metrics=track_metrics,
            warmup_epochs=warmup_epochs,
        )

        check_torch_available()

        if adversary_hidden_dims is None:
            adversary_hidden_dims = [32, 16]

        self.n_groups = n_groups
        self.adversary_lr = adversary_lr
        self.n_adversary_steps = n_adversary_steps
        self._check_adversary_steps()
        self.use_gradient_reversal = use_gradient_reversal

        self.adversary = Adversary(
            input_dim=1,  # Takes prediction as input
            hidden_dims=adversary_hidden_dims,
            n_groups=n_groups,
        )

        # Gradient reversal layer (optional).
        # Reversal scale is fixed at 1.0: BaseFairnessLoss.forward already
        # multiplies the penalty by lambda_fairness, so scaling the reversed
        # gradient by lambda here as well applied lambda TWICE (lambda^2;
        # the default 0.1 became an effective 0.01, nullifying the
        # intervention). Lambda must be applied exactly once, in forward.
        if use_gradient_reversal:
            self.gradient_reversal = GradientReversalLayer(lambda_=1.0)

        self.adversary_optimizer = torch.optim.Adam(
            self.adversary.parameters(),
            lr=adversary_lr,
        )

        # Baseline for "has this adversary ever been trained?" (see
        # _AdversarialCoverageLoss). Taken last, once the network exists.
        self._init_adversary_tracking()

    def _adversary_loss(
        self,
        sensitive_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Adversary's prediction loss for the sensitive attribute."""
        if self.n_groups == 2:
            return F.binary_cross_entropy(
                sensitive_pred.squeeze(), sensitive_attr.float(), reduction="mean"
            )
        return F.cross_entropy(sensitive_pred, sensitive_attr.long(), reduction="mean")

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute adversarial fairness penalty.

        The coverage is recorded ABOVE the branch selection, because both
        branches share the preconditions: a sensitive attribute with two
        distinct values to tell apart, and an adversary that has actually
        been trained. When neither holds the steering tensor is a
        gradient-free zero, so nothing is added to the total loss; when only
        the adversary is untrained the real penalty is kept (the alternating
        scheme has to start somewhere) and it is the REPORTED measurement
        that is withheld.
        """
        check_torch_available()

        measurable = self._mark_adversarial_coverage(
            sensitive_attr,
            use_gradient_reversal=self.use_gradient_reversal,
        )
        if not measurable and self._penalty_has_no_quantity(sensitive_attr):
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        if y_pred.dim() == 1:
            adv_input = y_pred.unsqueeze(1)
        else:
            adv_input = y_pred

        if self.use_gradient_reversal:
            # Reversal: the adversary's own weights receive the normal
            # gradient (correctly signed for training it, if the caller
            # steps adversary_optimizer), while the gradient reaching the
            # predictor is flipped, pushing it to make its outputs
            # uninformative about the attribute.
            adv_input = self.gradient_reversal(adv_input)
            sensitive_pred = self.adversary(adv_input)
            return self._adversary_loss(sensitive_pred, sensitive_attr)

        # Alternating scheme (Zhang et al. 2018): the predictor updates
        # against a FIXED adversary, so block gradients into the adversary's
        # weights and return the NEGATED loss. forward's task + lambda *
        # penalty then realizes task - lambda * adv_loss: gradient descent
        # makes y_pred LESS informative about the attribute. The previous
        # code returned +adv_loss here, so descent trained the predictor to
        # HELP the adversary (sign inversion, 2026-08-22 audit). The
        # adversary itself is trained only via update_adversary().
        requires = [p.requires_grad for p in self.adversary.parameters()]
        for p in self.adversary.parameters():
            p.requires_grad_(False)
        try:
            sensitive_pred = self.adversary(adv_input)
            adv_loss = self._adversary_loss(sensitive_pred, sensitive_attr)
        finally:
            for p, r in zip(self.adversary.parameters(), requires):
                p.requires_grad_(r)

        return -adv_loss

    def update_adversary(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> float:
        """
        Update adversary network (for non-gradient-reversal training).

        Args:
            y_pred: Detached model predictions
            sensitive_attr: True sensitive attributes

        Returns:
            Mean adversary loss over the steps that ran.

        Raises:
            ValueError: if ``n_adversary_steps`` is below 1, so no step could
                run and no adversary loss exists to average; or if the batch
                holds non-finite values or out-of-domain labels, either of
                which corrupts the adversary while marking it trained (see
                :meth:`_check_adversary_training_inputs`).
        """
        check_torch_available()
        self._check_adversary_steps()
        self._check_adversary_training_inputs(y_pred, sensitive_attr)

        total_adv_loss = 0.0

        for _ in range(self.n_adversary_steps):
            self.adversary_optimizer.zero_grad()

            if y_pred.dim() == 1:
                adv_input = y_pred.unsqueeze(1)
            else:
                adv_input = y_pred

            sensitive_pred = self.adversary(adv_input)

            # Compute loss (adversary wants to maximize accuracy)
            if self.n_groups == 2:
                adv_loss = F.binary_cross_entropy(
                    sensitive_pred.squeeze(),
                    sensitive_attr.float(),
                )
            else:
                adv_loss = F.cross_entropy(
                    sensitive_pred,
                    sensitive_attr.long(),
                )

            adv_loss.backward()
            self.adversary_optimizer.step()

            total_adv_loss += adv_loss.item()

        # "Has this adversary been trained?" is NOT latched here. It is decided
        # by the WEIGHTS: with adversary_lr=0.0, or a zero gradient, or a batch
        # with no rows, the optimizer step moves nothing. Latching on the call
        # alone reported the random initialisation as a measurement; measured
        # with adversary_lr=0.0, one update_adversary() call, weights provably
        # identical to the construction values: fairness_penalty_assessed came
        # back True with fairness_loss 0.6960861. Leave it to
        # _adversary_is_at_initialization(), which compares the tensors and
        # latches itself the moment any weight differs.

        return self._report_adversary_step_loss(total_adv_loss / self.n_adversary_steps)

    def get_adversary_accuracy(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> float:
        """
        Get adversary's prediction accuracy.

        Higher accuracy means more sensitive information is leaked.

        The two preconditions of :meth:`_compute_fairness_penalty` hold here
        for the same reason, so they are checked here too. With a CONSTANT
        sensitive attribute the number is decided by which side of 0.5 the
        network happens to sit on, not by the data: measured on one untrained
        adversary and one batch of predictions, an all-zero attribute scored
        1.0 ("everything leaks") and an all-one attribute scored 0.0
        ("nothing leaks") from the SAME network. With an adversary still at
        its construction weights the accuracy reports the initialisation. A
        third input makes it undefined in the same way: a NaN or inf in either
        tensor, which the ``>`` / ``argmax`` / ``==`` comparisons swallow into
        a confident-looking 0.0, 0.5 or 1.0. None of the three is a leakage
        measurement, so none is returned as one.

        The attribute is compared row against row, in either shape: a column
        vector is read as the n values it holds, and a row COUNT that does not
        line up with the predictions is a fourth could-not-check rather than a
        broadcast (see the comment at the comparison for what the broadcast
        used to score).

        Args:
            y_pred: Model predictions
            sensitive_attr: True sensitive attributes

        Returns:
            Adversary accuracy (0 to 1) when it was measurable, else
            ``float("nan")`` with a ``UserWarning`` naming the reason. NaN is
            the could-not-check state; it is deliberately not 0.0, which on
            this scale is the score a perfectly non-leaking model earns.
        """
        check_torch_available()

        # Preconditions ABOVE the computation, in the same order the coverage
        # helper uses them. This does NOT write self._fairness_coverage: that
        # field belongs to the last penalty, and reading the accuracy must not
        # overwrite what the last forward() recorded.
        self._check_sensitive_domain(sensitive_attr)

        # Non-finite values, checked before the group count, because a NaN
        # reaches the comparison silently and DECIDES it: sigmoid(NaN) > 0.5 is
        # False, argmax of an all-NaN row is 0, and NaN == NaN is False.
        # Measured on a trained adversary before this guard: all-NaN
        # predictions scored 0.5 with two groups and 0.34375 with three, +inf
        # predictions scored 0.5, and an all-NaN attribute scored 0.0. Every
        # one of those reads as a leakage measurement (0.5 is chance, 0.0 is a
        # perfectly non-leaking model) and none of them measured anything.
        non_finite = []
        if not bool(torch.isfinite(y_pred).all()):
            non_finite.append("y_pred")
        if not bool(torch.isfinite(sensitive_attr).all()):
            non_finite.append("sensitive_attr")
        if non_finite:
            warnings.warn(
                f"{type(self).__name__}.get_adversary_accuracy: "
                f"{' and '.join(non_finite)} holds non-finite values (NaN or "
                f"inf), so the adversary's prediction cannot be compared with "
                f"the attribute and no leakage can be measured. Returning NaN "
                f"(NOT MEASURED) rather than the 0.0 or 0.5 the comparison "
                f"silently produces from them.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        n_groups_present = int(torch.unique(sensitive_attr).numel())
        if n_groups_present < 2:
            warnings.warn(
                f"{type(self).__name__}.get_adversary_accuracy: the sensitive "
                f"attribute has {n_groups_present} distinct value(s) in this "
                f"batch, so there is nothing for the adversary to tell apart and "
                f"its accuracy measures no leakage. Returning NaN (NOT MEASURED) "
                f"rather than a confident 0.0 or 1.0.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        # A COMPARISON THAT CANNOT BE TRUE IS NOT A MEASUREMENT OF ZERO.
        #
        # The accuracy below is `pred_labels == sensitive_attr`, and
        # pred_labels are class INDICES: 0/1 from the sigmoid threshold, or an
        # argmax. Against an attribute that is not on the integer label grid
        # the equality is False for every row no matter what the network
        # learned, so the result is exactly 0.0, and this method's own
        # docstring says "Higher accuracy means more sensitive information is
        # leaked", making 0.0 the strongest all-clear on the scale.
        #
        # Measured before this guard, with n_groups=2 and an attribute of
        # 0.3/0.7 (inside the label domain, so _check_sensitive_domain lets it
        # through) after 50 update_adversary steps on a perfectly separating
        # batch: accuracy 0.0 with zero warnings, where the same fixture with
        # 0.0/1.0 labels scores 1.0. The adversary recovered the attribute
        # completely and the number said nothing leaked.
        #
        # Soft targets are a legal input to the adversary's own BCE training,
        # so this is a could-not-check on the accuracy rather than a raised
        # error on the call.
        if not bool(torch.equal(sensitive_attr, torch.round(sensitive_attr.float()))):
            warnings.warn(
                f"{type(self).__name__}.get_adversary_accuracy: sensitive_attr "
                f"holds values that are not integer class labels (observed e.g. "
                f"{float(sensitive_attr.reshape(-1)[0].item())}), and the "
                f"accuracy is an equality against predicted class indices, so it "
                f"would be 0.0 for every row whatever the adversary learned. "
                f"Returning NaN (NOT MEASURED) rather than that 0.0, which on "
                f"this scale means no leakage at all. Pass the integer group "
                f"labels to measure leakage.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        if self._adversary_is_at_initialization():
            warnings.warn(
                f"{type(self).__name__}.get_adversary_accuracy: the adversary is "
                f"still at its construction weights, so its accuracy reports the "
                f"random initialisation rather than any leakage. Train it with "
                f"update_adversary() on detached predictions (or step an "
                f"optimizer over its parameters) before reading this number. "
                f"Returning NaN (NOT MEASURED).",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        with torch.no_grad():
            if y_pred.dim() == 1:
                adv_input = y_pred.unsqueeze(1)
            else:
                adv_input = y_pred

            sensitive_pred = self.adversary(adv_input)

            # THE ADVERSARY'S OWN PREDICTION, NOT ITS INPUTS (BGL wave 4,
            # 2026-09-30). The two guards at the top of this method test y_pred
            # and sensitive_attr, and their comment gives the reason: "a NaN
            # reaches the comparison silently and DECIDES it: sigmoid(NaN) > 0.5
            # is False, argmax of an all-NaN row is 0, and NaN == NaN is False".
            # A non-finite ADVERSARY does exactly the same thing from the other
            # side, and was tested nowhere. Measured on a trained adversary over
            # a perfectly separable 40-row batch (accuracy 1.0, 0 warnings):
            #
            #   every adversary WEIGHT multiplied by NaN  -> accuracy 0.5,
            #       0 warnings, adversary output non-finite. 0.5 is chance on
            #       this scale, and this file calls that "no leakage detected".
            #   AdversarialDebiasingLoss(n_groups=3, adversary_lr=1e14), fifty
            #   of the library's own update_adversary calls -> accuracy
            #       0.3333333432674808, 0 warnings, every weight NaN.
            #
            # Keyed on the OUTPUT and placed ABOVE the n_groups dispatch: both
            # branches below share the precondition, the sigmoid arm and the
            # argmax arm swallow a NaN in different directions, and finite
            # weights can still produce a non-finite output (a large weight
            # sends a logit to inf and softmax turns inf - inf into NaN), which
            # a parameter check alone would miss.
            if not bool(torch.isfinite(sensitive_pred).all()):
                n_broken = self._adversary_weights_not_finite()
                warnings.warn(
                    f"{type(self).__name__}.get_adversary_accuracy: the adversary "
                    f"produced non-finite predictions "
                    f"({n_broken} of its parameter tensor(s) hold NaN or inf), so "
                    f"there is nothing to compare with the attribute and no "
                    f"leakage can be measured. Returning NaN (NOT MEASURED) "
                    f"rather than the chance rate the comparison silently "
                    f"produces from them, which on this scale reads as no "
                    f"leakage at all. The adversary has diverged; rebuild the "
                    f"loss or train it with a learning rate that does not.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")

            if self.n_groups == 2:
                pred_labels = (sensitive_pred.squeeze() > 0.5).long()
            else:
                pred_labels = sensitive_pred.argmax(dim=1)

            # A BROADCAST IS NOT AN ACCURACY.
            #
            # `pred_labels` is one label per row, shape (n,). The attribute is
            # accepted in either shape all over this class (the adversary input
            # path branches on `dim()` explicitly, and the penalty's BCE takes
            # a column target), so an (n, 1) attribute arrives here routinely.
            # `(n,) == (n, 1)` does NOT compare row against row: it broadcasts
            # to an (n, n) matrix of every prediction against every attribute
            # value, and the mean of that matrix is the chance rate of the
            # group mix, not an accuracy.
            #
            # Measured on the perfectly separable 40-row fixture, adversary
            # trained 50 steps, THE SAME VALUES in both calls:
            #
            #   flat (40,)    -> 1.0   (before and after)
            #   column (40,1) -> 0.5 with zero warnings   BEFORE
            #                 -> 1.0 with zero warnings   AFTER
            #
            # 0.5 is chance on this scale, which reads as "no leakage detected",
            # returned from a fully recovered attribute in silence. Every guard
            # above passes it: the values are on the integer grid, two distinct
            # values are present, both tensors are finite and the adversary is
            # trained. Flattening is not a coercion of somebody's data, it is
            # the row-wise comparison the docstring already promises; a row
            # COUNT that does not line up is a different matter and cannot be
            # reconciled, so it is the could-not-check below.
            pred_flat = pred_labels.reshape(-1)
            attr_flat = sensitive_attr.reshape(-1)
            if pred_flat.numel() != attr_flat.numel():
                warnings.warn(
                    f"{type(self).__name__}.get_adversary_accuracy: the adversary "
                    f"produced {pred_flat.numel()} label(s) for "
                    f"{attr_flat.numel()} attribute value(s), so they cannot be "
                    f"compared row by row. Returning NaN (NOT MEASURED) rather "
                    f"than the mean of a broadcast comparison, which is the "
                    f"chance rate of the group mix and reads as a leakage "
                    f"measurement. Pass one attribute value per prediction row.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")

            accuracy = (pred_flat == attr_flat).float().mean().item()

        return accuracy

    def to(self, device):
        """Move loss function to device."""
        super().to(device)
        self.adversary = self.adversary.to(device)
        return self


class ProjectedAdversarialLoss(_AdversarialCoverageLoss):
    """
    Scaled gradient-reversal adversarial loss.

    The adversary tries to predict the sensitive attribute from the model's
    predictions; a gradient-reversal layer (scaled by ``projection_strength``)
    sits between the predictions and the adversary, so a single backward pass
    (a) trains the adversary to predict the attribute and (b) pushes the
    predictor in the opposite direction: to make its outputs uninformative
    about the attribute. This is the DANN-style reversal update, i.e. the
    task gradient minus ``projection_strength`` times the adversary gradient.

    .. note::
        The previous implementation returned ``-adv_loss`` with no gradient
        reversal and no adversary optimizer: gradient descent then UN-trained
        the adversary (maximising its error), after which the "penalty"
        carried no fairness signal at all.

    Args:
        lambda_fairness: Weight of the adversarial penalty in the total loss
        projection_strength: Scale of the reversed adversary gradient
        adversary_hidden_dims: Hidden layer sizes for adversary
        n_groups: Number of sensitive attribute groups
        adversary_lr: Learning rate for the optional separate adversary steps
        n_adversary_steps: Adversary updates per ``update_adversary`` call
        base_loss: Base task loss type
        track_metrics: Whether to track metrics

    Example:
        >>> loss_fn = ProjectedAdversarialLoss(
        ...     lambda_fairness=0.5,
        ...     projection_strength=1.0
        ... )
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)
        >>> loss_fn.update_adversary(y_pred.detach(), sensitive)  # optional

    References:
        - Ganin & Lempitsky (2015): Unsupervised Domain Adaptation by
          Backpropagation (gradient reversal)
        - Madras et al. (2018): Learning Adversarially Fair Representations

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: projected_adversarial. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        projection_strength: float = 1.0,
        adversary_hidden_dims: Optional[List[int]] = None,
        n_groups: int = 2,
        adversary_lr: float = 0.001,
        n_adversary_steps: int = 1,
        base_loss: Union[str, BaseLossType] = BaseLossType.BINARY_CROSS_ENTROPY,
        reduction: Literal["mean", "sum", "none"] = "mean",
        track_metrics: bool = True,
        warmup_epochs: int = 0,
    ):
        super().__init__(
            lambda_fairness=lambda_fairness,
            base_loss=base_loss,
            reduction=reduction,
            track_metrics=track_metrics,
            warmup_epochs=warmup_epochs,
        )

        check_torch_available()

        if adversary_hidden_dims is None:
            adversary_hidden_dims = [32, 16]

        self.n_groups = n_groups
        self.projection_strength = projection_strength
        self.n_adversary_steps = n_adversary_steps
        self._check_adversary_steps()

        self.adversary = Adversary(
            input_dim=1,
            hidden_dims=adversary_hidden_dims,
            n_groups=n_groups,
        )

        # Reversal layer: forwards identity, backwards flips (and scales) the
        # gradient flowing into the predictor.
        self.gradient_reversal = GradientReversalLayer(lambda_=projection_strength)

        # Separate optimizer so callers can also strengthen the adversary on
        # detached predictions (mirrors AdversarialDebiasingLoss).
        self.adversary_optimizer = torch.optim.Adam(
            self.adversary.parameters(),
            lr=adversary_lr,
        )

        self._init_adversary_tracking()

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Adversarial penalty through the scaled gradient-reversal layer."""
        check_torch_available()

        # Same preconditions as AdversarialDebiasingLoss: two distinct
        # attribute values, and an adversary that has been trained.
        measurable = self._mark_adversarial_coverage(
            sensitive_attr,
            projection_strength=self.projection_strength,
        )
        if not measurable and self._penalty_has_no_quantity(sensitive_attr):
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        if y_pred.dim() == 1:
            adv_input = y_pred.unsqueeze(1)
        else:
            adv_input = y_pred

        # Reversal: the adversary's own weights receive the normal gradient
        # (it keeps learning to predict the attribute), while the gradient
        # reaching the predictor is flipped and scaled, pushing it to make
        # its outputs uninformative about the attribute.
        adv_input = self.gradient_reversal(adv_input)
        sensitive_pred = self.adversary(adv_input)

        if self.n_groups == 2:
            adv_loss = F.binary_cross_entropy(
                sensitive_pred.squeeze(), sensitive_attr.float(), reduction="mean"
            )
        else:
            adv_loss = F.cross_entropy(sensitive_pred, sensitive_attr.long(), reduction="mean")

        return adv_loss

    def update_adversary(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> float:
        """Extra adversary training steps on detached predictions.

        Args:
            y_pred: Detached model predictions
            sensitive_attr: True sensitive attributes

        Returns:
            Mean adversary loss over the steps that ran.

        Raises:
            ValueError: if ``n_adversary_steps`` is below 1, so no step could
                run and no adversary loss exists to average; or if the batch
                holds non-finite values or out-of-domain labels, either of
                which corrupts the adversary while marking it trained (see
                :meth:`_check_adversary_training_inputs`).
        """
        check_torch_available()
        self._check_adversary_steps()
        self._check_adversary_training_inputs(y_pred, sensitive_attr)

        total_adv_loss = 0.0

        for _ in range(self.n_adversary_steps):
            self.adversary_optimizer.zero_grad()

            if y_pred.dim() == 1:
                adv_input = y_pred.unsqueeze(1)
            else:
                adv_input = y_pred

            sensitive_pred = self.adversary(adv_input)

            if self.n_groups == 2:
                adv_loss = F.binary_cross_entropy(
                    sensitive_pred.squeeze(),
                    sensitive_attr.float(),
                )
            else:
                adv_loss = F.cross_entropy(
                    sensitive_pred,
                    sensitive_attr.long(),
                )

            adv_loss.backward()
            self.adversary_optimizer.step()

            total_adv_loss += adv_loss.item()

        # Not latched here, and no max(1, ...) on the denominator: see the same
        # two points in AdversarialDebiasingLoss.update_adversary. The guard
        # above makes the step count at least 1, so the division is safe; the
        # old max(1, 0) turned a loop that never ran into a returned 0.0, the
        # score a perfect adversary earns, and marked the untrained adversary
        # trained on the way out (measured: n_adversary_steps=0 returned 0.0,
        # after which forward() reported fairness_penalty_assessed=True with
        # fairness_loss 0.7254388).

        return self._report_adversary_step_loss(total_adv_loss / self.n_adversary_steps)

    def to(self, device):
        """Move loss function to device."""
        super().to(device)
        self.adversary = self.adversary.to(device)
        return self


class FairRepresentationLoss(_AdversarialCoverageLoss):
    """
    Loss for learning fair representations.

    This loss encourages the model to learn representations that are
    predictive of the target but not predictive of the sensitive attribute.
    It combines reconstruction loss, task loss, and fairness penalty.

    Components:
        L = L_task + α * L_reconstruction - λ * L_adversary

    Args:
        lambda_fairness: Weight for adversarial penalty
        alpha_reconstruction: Weight for reconstruction loss
        adversary_hidden_dims: Hidden layer sizes for adversary
        n_groups: Number of sensitive attribute groups
        representation_dim: Dimension of fair representation (optional)
        base_loss: Base task loss type
        track_metrics: Whether to track metrics

    Example:
        >>> loss_fn = FairRepresentationLoss(
        ...     lambda_fairness=1.0,
        ...     alpha_reconstruction=0.5
        ... )
        >>> # The adversary trains only if YOUR optimizer covers its
        >>> # parameters; there is no update_adversary() on this class.
        >>> opt = torch.optim.Adam(
        ...     list(model.parameters()) + list(loss_fn.adversary.parameters())
        ... )

    .. note::
        Unlike :class:`AdversarialDebiasingLoss` this class has no
        ``update_adversary()`` and no ``adversary_optimizer``. Its adversary
        is trained ONLY by an optimizer the caller built over
        ``loss_fn.adversary.parameters()``; the gradient-reversal layer
        computes correctly-signed gradients for those weights during
        ``loss.backward()``, but nothing steps them for you. If your optimizer
        does not cover them the adversary never leaves its construction
        weights, and every batch is reported as NOT MEASURED
        (``fairness_loss`` NaN, reason ``adversary_at_initialization``) for as
        long as training runs. With the parameters included it measures from
        the batch after the first step onward.

    References:
        - Zemel et al. (2013): Learning Fair Representations
        - Louizos et al. (2016): The Variational Fair Autoencoder

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: fair_representation_loss. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 1.0,
        alpha_reconstruction: float = 0.5,
        adversary_hidden_dims: Optional[List[int]] = None,
        n_groups: int = 2,
        representation_dim: Optional[int] = None,
        base_loss: Union[str, BaseLossType] = BaseLossType.BINARY_CROSS_ENTROPY,
        reduction: Literal["mean", "sum", "none"] = "mean",
        track_metrics: bool = True,
        warmup_epochs: int = 0,
    ):
        super().__init__(
            lambda_fairness=lambda_fairness,
            base_loss=base_loss,
            reduction=reduction,
            track_metrics=track_metrics,
            warmup_epochs=warmup_epochs,
        )

        check_torch_available()

        if adversary_hidden_dims is None:
            adversary_hidden_dims = [32, 16]

        self.n_groups = n_groups
        self.alpha_reconstruction = alpha_reconstruction
        self.representation_dim = representation_dim

        input_dim = representation_dim if representation_dim else 1
        self.adversary = Adversary(
            input_dim=input_dim,
            hidden_dims=adversary_hidden_dims,
            n_groups=n_groups,
        )

        # Gradient reversal. Scale fixed at 1.0: this class's forward already
        # multiplies the penalty by effective_lambda, so scaling the reversed
        # gradient by lambda here as well applied lambda TWICE (lambda^2) -
        # the same double-scaling defect fixed in AdversarialDebiasingLoss
        # (2026-08-22 audit). Lambda must be applied exactly once.
        self.gradient_reversal = GradientReversalLayer(lambda_=1.0)

        self._init_adversary_tracking()
        self._warned_representation_substituted = False

    def forward(  # type: ignore[override]  # extends base forward with keyword-only representation/reconstruction args (PyTorch forward is intentionally not LSP-constrained)
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        representation: Optional["torch.Tensor"] = None,
        x_reconstructed: Optional["torch.Tensor"] = None,
        x_original: Optional["torch.Tensor"] = None,
        sample_weight: Optional["torch.Tensor"] = None,
        return_components: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", LossComponents]]:
        """
        Compute fair representation loss.

        Args:
            y_pred: Task predictions
            y_true: True labels
            sensitive_attr: Sensitive attribute values
            representation: Learned representation (for adversary input)
            x_reconstructed: Reconstructed input (for reconstruction loss)
            x_original: Original input (for reconstruction loss)
            sample_weight: Optional sample weights
            return_components: Whether to return components

        Returns:
            Total loss, optionally with components
        """
        check_torch_available()

        # This forward replaces _CoverageTrackingLoss.forward rather than
        # calling it, so it has to clear the previous batch's coverage itself.
        # Without this line a batch whose penalty recorded nothing would be
        # stamped with the LAST batch's verdict, and a stale
        # fairness_penalty_assessed=True is the same fabrication one layer up.
        self._fairness_coverage = None

        # Half a reconstruction pair is a caller error, not a reconstruction
        # of zero error: before this check, passing only x_reconstructed
        # silently produced regularization_loss 0.0, the value a PERFECT
        # reconstruction earns.
        if (x_reconstructed is None) != (x_original is None):
            missing = "x_original" if x_original is None else "x_reconstructed"
            raise ValueError(
                f"x_reconstructed and x_original must be supplied together; "
                f"{missing} is missing. With only one of them no reconstruction "
                f"error exists, and reporting 0.0 would be the score a perfect "
                f"reconstruction earns."
            )

        if representation is None and not self._warned_representation_substituted:
            self._warned_representation_substituted = True
            warnings.warn(
                "FairRepresentationLoss: no `representation` was supplied, so the "
                "adversary is being run on the task predictions instead. The "
                "penalty then measures leakage from the OUTPUT, not from the "
                "learned representation this loss is named for. Pass "
                "representation=... to measure what the docstring describes.",
                UserWarning,
                stacklevel=2,
            )

        task_loss = self._compute_task_loss(y_pred, y_true, sample_weight)

        # This forward builds its own LossComponents instead of going through
        # BaseFairnessLoss.forward, so the task-loss coverage has to be taken
        # here too, or the disclosure exists in the base and not in the one
        # class that overrides it. Measured before this call, with every
        # sample_weight zero and a trained adversary: task_loss 0.0 with no
        # warning, i.e. a perfect fit reported from zero rows, next to a
        # correctly refused fairness term.
        n_rows, n_used = self._task_loss_coverage(y_pred, sample_weight)

        fairness_loss = self._compute_fairness_penalty(
            y_pred if representation is None else representation,
            y_true,
            sensitive_attr,
        )

        reconstruction_loss = torch.tensor(0.0, device=y_pred.device)
        reconstruction_assessed = False
        if x_reconstructed is not None and x_original is not None:
            reconstruction_loss = F.mse_loss(x_reconstructed, x_original)
            reconstruction_assessed = True

        effective_lambda = self._get_effective_lambda()

        total_loss = (
            task_loss
            + self.alpha_reconstruction * reconstruction_loss
            + effective_lambda * fairness_loss
        )

        if return_components or self.track_metrics:
            components = LossComponents(
                total_loss=total_loss.item(),
                task_loss=task_loss.item(),
                fairness_loss=fairness_loss.item(),
                # NOT 0.0 when no reconstruction pair was supplied: 0.0 is
                # exactly what a PERFECT reconstruction produces, and the two
                # cases were byte identical in every reconstruction field.
                regularization_loss=(
                    reconstruction_loss.item() if reconstruction_assessed else float("nan")
                ),
                batch_metrics={
                    "effective_lambda": effective_lambda,
                    "reconstruction_loss": (
                        reconstruction_loss.item() if reconstruction_assessed else None
                    ),
                    "reconstruction_assessed": reconstruction_assessed,
                    "representation_supplied": representation is not None,
                },
            )
            if not reconstruction_assessed:
                warnings.warn(
                    "FairRepresentationLoss: no reconstruction pair was supplied, so "
                    "the reconstruction term is NOT MEASURED. It is reported as NaN "
                    "in LossComponents.regularization_loss and as None in "
                    "batch_metrics['reconstruction_loss'], not as 0.0, which is the "
                    "value a perfect reconstruction earns. Pass x_reconstructed and "
                    "x_original to measure it.",
                    UserWarning,
                    stacklevel=2,
                )
            self._record_fairness_coverage(components)
            self._record_task_coverage(components, n_rows, n_used)
            if self.track_metrics:
                self._batch_history.append(components)

        if return_components:
            return total_loss, components
        return total_loss

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute adversarial fairness penalty on representation.

        ``y_pred`` here is the representation when the caller supplied one,
        and the task predictions when they did not (forward says so in a
        warning).
        """
        check_torch_available()

        measurable = self._mark_adversarial_coverage(sensitive_attr)
        if not measurable and self._penalty_has_no_quantity(sensitive_attr):
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        if y_pred.dim() == 1:
            adv_input = y_pred.unsqueeze(1)
        else:
            adv_input = y_pred

        # Apply gradient reversal
        adv_input = self.gradient_reversal(adv_input)

        sensitive_pred = self.adversary(adv_input)

        if self.n_groups == 2:
            adv_loss = F.binary_cross_entropy(
                sensitive_pred.squeeze(), sensitive_attr.float(), reduction="mean"
            )
        else:
            adv_loss = F.cross_entropy(sensitive_pred, sensitive_attr.long(), reduction="mean")

        return adv_loss

    def to(self, device):
        """Move loss function to device."""
        super().to(device)
        self.adversary = self.adversary.to(device)
        return self
