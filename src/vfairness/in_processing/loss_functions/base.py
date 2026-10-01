"""
Base classes for Fairness-Aware Loss Functions.

This module provides abstract base classes and common utilities for implementing
fairness-aware loss functions that can be used in PyTorch training loops.

The loss functions in this module enable fairness interventions directly during
model training by modifying the optimization objective to include fairness penalties.

Key Concepts:
    - **Task Loss**: The primary prediction loss (e.g., BCE, CE, MSE)
    - **Fairness Penalty**: Additional term penalizing fairness violations
    - **Lambda (λ)**: Trade-off parameter between accuracy and fairness
    - **Differentiable Approximations**: Soft versions of non-differentiable metrics

References:
    - Zafar et al. (2017): Fairness Constraints: Mechanisms for Fair Classification
    - Zhang et al. (2018): Mitigating Unwanted Biases with Adversarial Learning
    - Donini et al. (2018): Empirical Risk Minimization Under Fairness Constraints
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
"""

import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

# Try to import PyTorch - make it optional
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy modules for runtime use when torch is not available.
    class _TorchNNStub:  # torch-absence stub
        class Module:
            pass

    nn = _TorchNNStub  # type: ignore[assignment]  # runtime fallback stands in for torch.nn

# Base class for nn.Module-derived types. At runtime this is the real
# nn.Module when torch is present, else `object`. For type checking we always
# treat it as nn.Module so mypy resolves method overrides against a real base.
if TYPE_CHECKING:
    from torch.nn import Module as _ModuleBase
else:
    _ModuleBase = nn.Module if TORCH_AVAILABLE else object


class FairnessMetricType(Enum):
    """Enumeration of fairness metrics for loss computation."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


class BaseLossType(Enum):
    """Enumeration of base loss functions."""

    BINARY_CROSS_ENTROPY = "bce"
    CROSS_ENTROPY = "ce"
    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae"
    HINGE = "hinge"
    FOCAL = "focal"


@dataclass
class LossComponents:
    """
    Container for loss function components.

    Attributes:
        total_loss: Combined loss value
        task_loss: Primary prediction loss
        fairness_loss: Fairness penalty term
        regularization_loss: Optional regularization term
        auxiliary_losses: Additional loss components (e.g., adversarial)
        batch_metrics: Metrics computed for the batch
    """

    total_loss: float
    task_loss: float
    fairness_loss: float
    regularization_loss: float = 0.0
    auxiliary_losses: Dict[str, float] = field(default_factory=dict)
    batch_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_loss": self.total_loss,
            "task_loss": self.task_loss,
            "fairness_loss": self.fairness_loss,
            "regularization_loss": self.regularization_loss,
            "auxiliary_losses": self.auxiliary_losses,
            "batch_metrics": self.batch_metrics,
        }


@dataclass
class TrainingMetrics:
    """
    Aggregated metrics from a training epoch.

    Attributes:
        epoch: Epoch number
        avg_total_loss: Average total loss over batches
        avg_task_loss: Average task loss
        avg_fairness_loss: Average fairness penalty
        fairness_metrics: Dictionary of fairness metric values
        convergence_info: Information about convergence
    """

    epoch: int
    avg_total_loss: float
    avg_task_loss: float
    avg_fairness_loss: float
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    convergence_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "avg_total_loss": self.avg_total_loss,
            "avg_task_loss": self.avg_task_loss,
            "avg_fairness_loss": self.avg_fairness_loss,
            "fairness_metrics": self.fairness_metrics,
            "convergence_info": self.convergence_info,
        }


def check_torch_available():
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for fairness-aware loss functions. Install with: pip install torch"
        )


def soft_rate_computation(
    y_pred: "torch.Tensor",
    y_true: "torch.Tensor",
    group_mask: "torch.Tensor",
    rate_type: Literal["positive_rate", "tpr", "fpr", "tnr", "fnr"] = "positive_rate",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """
    Compute soft (differentiable) versions of rate metrics.

    This function provides differentiable approximations to binary metrics
    like TPR, FPR, etc., enabling gradient-based optimization.

    Args:
        y_pred: Predicted probabilities (after sigmoid)
        y_true: True binary labels
        group_mask: Boolean mask for the group
        rate_type: Type of rate to compute
        temperature: Accepted only as 1.0, the identity. This function does NOT
            soften its inputs: every rate below is the plain mean of the
            probabilities handed in, already smooth because they come from a
            sigmoid. Any other value is refused rather than silently ignored.
            To sharpen or flatten the predictions a fairness penalty sees, use
            FairnessAwareBCELoss(temperature=...), which re-scales them on the
            logit scale before any group statistic is computed.

    Returns:
        Soft rate value (differentiable)

    Raises:
        ValueError: If temperature is anything other than 1.0.
    """
    check_torch_available()

    # S-05 (2026-09-09): `temperature` was documented as "Temperature for
    # softening (lower = sharper)" and read by nothing. Measured before the
    # fix: for every rate_type, T in {0.01, 1.0, 100.0, 1e9} returned a
    # bit-identical value AND a bit-identical gradient, and 0.0, -5.0, nan
    # and inf were all accepted in silence. Nothing passes a non-default
    # value (compute_group_rates, the only in-tree caller, does not forward
    # it), so rather than invent a soft indicator here under release pressure
    # the parameter is refused for anything but the identity. The tempering
    # that IS implemented lives in fairness_losses._temper_predictions and is
    # applied once by FairnessAwareBCELoss before every group statistic (F14).
    if temperature != 1.0:
        raise ValueError(
            "soft_rate_computation does not implement temperature softening; got "
            f"temperature={temperature!r}. Only temperature=1.0 (the identity) is "
            "supported: every rate returned here is the plain mean of the "
            "probabilities passed in, so any other value would be advertised and "
            "not applied. To sharpen or flatten the predictions a fairness penalty "
            "sees, use FairnessAwareBCELoss(temperature=...), which re-scales them "
            "on the logit scale before the group statistics are taken."
        )

    y_pred_g = y_pred[group_mask]
    y_true_g = y_true[group_mask]

    if len(y_pred_g) == 0:
        # THE FOURTH UNMEASURED BRANCH, and the only one that was silent.
        # A mask selecting no row leaves NOTHING to average, so there is no
        # rate here at all, not even an arm of one. Measured before this fix,
        # with an all-False mask over four rows, for every rate_type:
        # tensor(0., requires_grad=True) and zero warnings. On these scales
        # 0.0 is the clean end for two of the five ("fpr" 0.0 is a group that
        # never draws a false positive, "fnr" 0.0 never misses a positive),
        # and for "positive_rate" it is the extreme that manufactures a gap
        # against any other group: compute_group_rates fed exactly that 0.0
        # into a demographic parity penalty for a group with no rows.
        # Same treatment as the three branches below, for the same reason: NaN
        # would poison the gradient of every group at once, so the finite
        # stand-in stays and the SILENCE goes.
        warnings.warn(
            "soft_rate_computation: the group mask selects 0 of "
            f"{int(group_mask.numel())} row(s), so this group's rate is NOT "
            "MEASURED at all, not even from an arm of one. Returning 0.5 as a "
            "finite stand-in to keep the gradient defined; exclude this group "
            "from any disparity rather than comparing against it.",
            UserWarning,
            stacklevel=2,
        )
        return torch.tensor(0.5, device=y_pred.device, requires_grad=True)

    if rate_type == "positive_rate":
        # Soft positive prediction rate (demographic parity)
        return y_pred_g.mean()

    elif rate_type == "tpr":
        # True Positive Rate: P(ŷ=1|y=1)
        positives_mask = y_true_g == 1
        if positives_mask.sum() == 0:
            # UNMEASURED, and 0.5 is a stand-in, not a rate. NaN cannot be
            # returned here: this value feeds a differentiable penalty, and one
            # NaN poisons the gradient for every group at once. So the finite
            # stand-in stays and the SILENCE goes, because a caller averaging
            # this into a disparity is comparing against a number nobody
            # measured.
            warnings.warn(
                "soft_rate_computation: this group has no positive-label examples, so "
                "its rate is NOT MEASURED. Returning 0.5 as a finite stand-in "
                "to keep the gradient defined; exclude this group from any "
                "disparity rather than comparing against it.",
                UserWarning,
                stacklevel=2,
            )
            return torch.tensor(0.5, device=y_pred.device, requires_grad=True)
        return y_pred_g[positives_mask].mean()

    elif rate_type == "fpr":
        # False Positive Rate: P(ŷ=1|y=0)
        negatives_mask = y_true_g == 0
        if negatives_mask.sum() == 0:
            # UNMEASURED, and 0.5 is a stand-in, not a rate. NaN cannot be
            # returned here: this value feeds a differentiable penalty, and one
            # NaN poisons the gradient for every group at once. So the finite
            # stand-in stays and the SILENCE goes, because a caller averaging
            # this into a disparity is comparing against a number nobody
            # measured.
            warnings.warn(
                "soft_rate_computation: this group has no negative-label examples, so "
                "its rate is NOT MEASURED. Returning 0.5 as a finite stand-in "
                "to keep the gradient defined; exclude this group from any "
                "disparity rather than comparing against it.",
                UserWarning,
                stacklevel=2,
            )
            return torch.tensor(0.5, device=y_pred.device, requires_grad=True)
        return y_pred_g[negatives_mask].mean()

    elif rate_type == "tnr":
        # True Negative Rate: P(ŷ=0|y=0)
        negatives_mask = y_true_g == 0
        if negatives_mask.sum() == 0:
            # UNMEASURED, and 0.5 is a stand-in, not a rate. NaN cannot be
            # returned here: this value feeds a differentiable penalty, and one
            # NaN poisons the gradient for every group at once. So the finite
            # stand-in stays and the SILENCE goes, because a caller averaging
            # this into a disparity is comparing against a number nobody
            # measured.
            warnings.warn(
                "soft_rate_computation: this group has no negative-label examples, so "
                "its rate is NOT MEASURED. Returning 0.5 as a finite stand-in "
                "to keep the gradient defined; exclude this group from any "
                "disparity rather than comparing against it.",
                UserWarning,
                stacklevel=2,
            )
            return torch.tensor(0.5, device=y_pred.device, requires_grad=True)
        return (1 - y_pred_g[negatives_mask]).mean()

    elif rate_type == "fnr":
        # False Negative Rate: P(ŷ=0|y=1)
        positives_mask = y_true_g == 1
        if positives_mask.sum() == 0:
            # UNMEASURED, and 0.5 is a stand-in, not a rate. NaN cannot be
            # returned here: this value feeds a differentiable penalty, and one
            # NaN poisons the gradient for every group at once. So the finite
            # stand-in stays and the SILENCE goes, because a caller averaging
            # this into a disparity is comparing against a number nobody
            # measured.
            warnings.warn(
                "soft_rate_computation: this group has no positive-label examples, so "
                "its rate is NOT MEASURED. Returning 0.5 as a finite stand-in "
                "to keep the gradient defined; exclude this group from any "
                "disparity rather than comparing against it.",
                UserWarning,
                stacklevel=2,
            )
            return torch.tensor(0.5, device=y_pred.device, requires_grad=True)
        return (1 - y_pred_g[positives_mask]).mean()

    else:
        raise ValueError(f"Unknown rate_type: {rate_type}")


class BaseFairnessLoss(_ModuleBase):
    """
    Abstract base class for fairness-aware loss functions.

    This class provides the common interface and utilities for implementing
    fairness-aware loss functions that combine a primary task loss with
    fairness penalty terms.

    All subclasses must implement:
        - _compute_fairness_penalty(): Calculate the fairness violation term

    The total loss is computed as:
        L_total = L_task + λ * L_fairness + L_reg

    Args:
        lambda_fairness: Trade-off parameter for fairness penalty (0 to 1+)
        base_loss: Type of base loss function
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        track_metrics: Whether to track detailed metrics during training
        warmup_epochs: Number of epochs before applying fairness penalty

    Example:
        >>> loss_fn = DemographicParityLoss(lambda_fairness=0.1)
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)
        >>> loss.backward()

    References:
        - Zafar et al. (2017): Fairness Constraints
    """

    # Class-level defaults so both read cleanly through nn.Module.__getattr__
    # before the first forward, the same reason _CoverageTrackingLoss declares
    # _fairness_coverage that way.
    _task_loss_unassessed_reason: Optional[str] = None
    _fairness_lambda_note: Optional[Tuple[str, str]] = None

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        base_loss: Union[str, BaseLossType] = BaseLossType.BINARY_CROSS_ENTROPY,
        reduction: Literal["mean", "sum", "none"] = "mean",
        track_metrics: bool = True,
        warmup_epochs: int = 0,
    ):
        # Fail loudly at construction when torch is missing. Without this,
        # the loss constructs fine and only dies at first call with an
        # obscure "'X' object is not callable" (no nn.Module __call__).
        check_torch_available()

        # @abstractmethod IS DECORATIVE ON THIS CLASS (BGL grade wave,
        # 2026-09-30). Abstractness is enforced by ABCMeta, and the base here is
        # torch.nn.Module, whose metaclass is plain `type`, so
        # BaseFairnessLoss.__abstractmethods__ is EMPTY and the docstring's "All
        # subclasses must implement" was never checked. Measured:
        #
        #   BaseFairnessLoss(lambda_fairness=0.5) -> constructed
        #   ._compute_fairness_penalty(...)       -> None
        #   forward(...)  -> TypeError: unsupported operand type(s) for *:
        #                    'float' and 'NoneType'
        #
        # and the same TypeError at lambda_fairness=0.0: a cryptic failure inside
        # the loss arithmetic, one layer away from the constructor that was the
        # actual mistake.
        #
        # THE REFUSAL IS NARROWED TO THE CASE IT CAN PROVE, and deliberately so.
        # The first version of it refused any class whose RESOLVED
        # _compute_fairness_penalty was still the abstract stub, and that
        # reddened four existing controls: tests/test_bgl2_base_and_entry_points
        # instantiates a _DirectLoss that inherits this base without the penalty
        # ON PURPOSE, overriding forward() so it never reaches the arithmetic,
        # because that subclass is the only way to execute the base end_epoch
        # aggregation at all. Refusing it would have broken a legitimate use to
        # close a failure that cannot occur in it. So the precondition checked
        # here is the one that actually leads to the TypeError: the penalty is
        # unimplemented AND forward() is this class's own, which is the external
        # implementer's realistic mistake. The other door, a subclass that
        # overrides forward and still calls the penalty, is closed at the point
        # of use by the stub itself, which now raises instead of returning None.
        #
        # Checked here rather than by giving the class an ABCMeta metaclass: this
        # file's torch-absence fallback makes the base `object`, and a metaclass
        # change would alter the class of every subclass.
        if (
            getattr(
                getattr(type(self), "_compute_fairness_penalty", None),
                "__isabstractmethod__",
                False,
            )
            and getattr(type(self), "forward", None) is BaseFairnessLoss.forward
        ):
            raise TypeError(
                f"{type(self).__name__} is abstract and cannot be instantiated: it "
                f"implements neither _compute_fairness_penalty nor forward(), so "
                f"forward() would multiply lambda_fairness by None. Use a concrete "
                f"loss (for example DemographicParityLoss or EqualizedOddsLoss), or "
                f"implement _compute_fairness_penalty in your subclass."
            )

        super().__init__()

        # A NEGATIVE OR UNREADABLE LAMBDA INVERTS THE INTERVENTION (BGL wave 4,
        # 2026-09-30). ``forward`` computes ``task_loss + effective_lambda *
        # fairness_loss``, so the sign of lambda decides whether the fairness
        # penalty is paid or REWARDED. The documented domain is "0 to 1+" and
        # nothing checked it. Measured on a 40-row batch with a real 0.2
        # demographic-parity disparity, DemographicParityLoss:
        #
        #   lambda_fairness=0.5  -> total 0.4566749632358551, ABOVE the task
        #                           loss 0.356675: the disparity is paid for
        #   lambda_fairness=-0.5 -> total 0.25667497515678406, BELOW the task
        #                           loss, so gradient descent MAXIMISES the
        #                           disparity, and LossComponents still
        #                           reported fairness_loss 0.19999998807907104
        #                           as a measured penalty with zero warnings
        #   lambda_fairness=nan  -> total nan, zero warnings, and nan compares
        #                           False against every threshold it is later
        #                           tested against, so it SUPPRESSES a finding
        #                           rather than raising one
        #
        # A mitigation that rewards what it exists to reduce is a caller error,
        # not a smaller penalty, so it is refused here rather than run. Zero is
        # allowed (a deliberate no-penalty baseline) and disclosed per batch by
        # _get_effective_lambda instead.
        if lambda_fairness != lambda_fairness or lambda_fairness in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError(
                f"{type(self).__name__}: lambda_fairness must be a finite number "
                f"(got {lambda_fairness!r}). It multiplies the fairness penalty "
                f"into the total loss, so a non-finite value makes the whole "
                f"training objective unreadable while the penalty is still "
                f"reported as measured."
            )
        if lambda_fairness < 0:
            raise ValueError(
                f"{type(self).__name__}: lambda_fairness must be at least 0 (got "
                f"{lambda_fairness!r}); the documented domain is '0 to 1+'. The "
                f"total loss is task_loss + lambda_fairness * fairness_loss, so a "
                f"negative value does not reduce unfairness, it PAYS the optimizer "
                f"for it: minimising the total then maximises the fairness "
                f"penalty. Pass 0.0 for a deliberate no-penalty baseline."
            )

        self.lambda_fairness = lambda_fairness
        self.base_loss = BaseLossType(base_loss) if isinstance(base_loss, str) else base_loss
        self.reduction = reduction
        self.track_metrics = track_metrics
        self.warmup_epochs = warmup_epochs

        # Training state
        self._current_epoch = 0
        self._batch_history: List[LossComponents] = []
        self._epoch_metrics: List[TrainingMetrics] = []

    @abstractmethod
    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute the fairness penalty term.

        This method must be implemented by subclasses to define
        the specific fairness constraint being enforced.

        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            sensitive_attr: Sensitive attribute values

        Returns:
            Fairness penalty (scalar tensor)

        Raises:
            NotImplementedError: always. The body used to be ``pass``, which
                RETURNS None, and @abstractmethod is not enforced on this class
                (see the comment in ``__init__``). ``forward`` then evaluated
                ``task_loss + effective_lambda * None`` and the caller saw
                "unsupported operand type(s) for *: 'float' and 'NoneType'"
                from inside the loss arithmetic. Raising names the missing
                implementation instead, and it covers the door the constructor
                refusal deliberately leaves open: a subclass that overrides
                forward and still reaches the penalty.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _compute_fairness_penalty, "
            f"which every fairness-aware loss must: it is the fairness term that "
            f"lambda_fairness is multiplied by in forward()."
        )

    def _compute_task_loss(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sample_weight: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute the primary task loss."""
        check_torch_available()

        if self.base_loss == BaseLossType.BINARY_CROSS_ENTROPY:
            loss = F.binary_cross_entropy(y_pred, y_true.float(), reduction="none")
        elif self.base_loss == BaseLossType.CROSS_ENTROPY:
            loss = F.cross_entropy(y_pred, y_true.long(), reduction="none")
        elif self.base_loss == BaseLossType.MEAN_SQUARED_ERROR:
            loss = F.mse_loss(y_pred, y_true.float(), reduction="none")
        elif self.base_loss == BaseLossType.MEAN_ABSOLUTE_ERROR:
            loss = F.l1_loss(y_pred, y_true.float(), reduction="none")
        elif self.base_loss == BaseLossType.HINGE:
            # Convert to -1/+1 labels for hinge loss
            y_true_hinge = 2 * y_true.float() - 1
            y_pred_hinge = 2 * y_pred - 1
            loss = F.relu(1 - y_true_hinge * y_pred_hinge)
        elif self.base_loss == BaseLossType.FOCAL:
            # Focal loss with gamma=2
            gamma = 2.0
            bce = F.binary_cross_entropy(y_pred, y_true.float(), reduction="none")
            pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
            loss = ((1 - pt) ** gamma) * bce
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    @staticmethod
    def _weights_are_usable(weights: "torch.Tensor") -> "torch.Tensor":
        """Element-wise: which weights can carry their row INTO the task loss.

        A weight is usable when it is finite, strictly positive, and at least
        the working dtype's own resolution. The test this replaces was
        ``(weights != 0) & torch.isfinite(weights)``, which asks whether a
        weight is zero or unreadable and never whether it is USABLE, so two
        more spellings of "this row was not weighted" walked through it as
        measurements. Measured 2026-09-30 (BGL wave 4) on the four-row batch of
        this file's own docstring, DemographicParityLoss(lambda_fairness=0.1):

            sample_weight=torch.tensor(1e-30)
              before -> task_loss 1.6425203722938138e-31, task_rows_used 4 of
                        4, zero warnings. The docstring below says "A task loss
                        of 0.0 is the score a PERFECT predictor earns", and
                        1.6e-31 is that 0.0 to every threshold it will be read
                        against, produced from a batch whose rows were each
                        multiplied by a number 23 orders of magnitude below
                        float32's resolution
              after  -> task_loss nan, task_rows_used 0, one warning

        ``torch.finfo(dtype).eps`` and not a literal: the floor is the dtype's
        own resolution, i.e. the point below which multiplying an O(1) task
        loss by the weight can no longer be told apart from multiplying it by
        zero. That is a data-scaled tolerance rather than an exact comparison
        with zero, which on a continuous quantity is the fabrication itself.
        The controls are unchanged: torch.ones(1), torch.tensor(2.0) and
        torch.full((4,), 0.01) all still measure their real numbers in silence.
        """
        if weights.is_floating_point():
            floor = torch.finfo(weights.dtype).eps
        else:
            # An integer weight cannot be fractional, so its own resolution is
            # 1; finfo() raises on integer dtypes.
            floor = 1
        return torch.isfinite(weights) & (weights >= floor)

    def _task_loss_coverage(
        self,
        y_pred: "torch.Tensor",
        sample_weight: Optional["torch.Tensor"] = None,
    ) -> Tuple[int, int]:
        """(rows in the batch, rows that could enter the task loss), and warn at 0.

        A task loss of 0.0 is the score a PERFECT predictor earns, and two
        inputs produce it from no rows at all. Measured before this check, on
        a four-row batch, both silent:

            sample_weight=torch.zeros(4) -> task_loss 0.0, total_loss 0.025
            reduction="sum" on an empty batch -> task_loss 0.0, total_loss 0.025

        and three such batches in a row gave ``end_epoch`` an
        ``avg_task_loss`` of 0.0, a completed epoch over nothing. The empty
        batch under the default reduction="mean" was already NaN, which is the
        honest state; "sum" reduces an empty tensor to 0.0 instead.

        A THIRD input did it too, and survived the first fix because the count
        was read from the weight tensor's shape rather than from its effect:
        ``sample_weight=torch.zeros(1)`` and ``sample_weight=torch.tensor(0.0)``
        are broadcast over all four rows by torch and still reported
        ``task_loss`` 0.0 with ``task_rows_used`` 4 in silence. See the comment
        on the ``numel() == 1`` branch below for the measurement.

        A FOURTH AND FIFTH survived both fixes, because the test was
        ``(weights != 0) & torch.isfinite(weights)``: it asks whether a weight
        is zero or unreadable and never whether it is USABLE. A NEGATIVE weight
        is finite and nonzero, so every row counted as alive while the sign of
        the objective was flipped, and a weight vector whose negatives cancel
        its positives reported a PERFECT fit from a model wrong on every row.
        A weight below the dtype's own resolution did the same through an exact
        comparison with zero on a continuous quantity. See the comment above the
        ``numel()`` dispatch and :meth:`_weights_are_usable`.
        """
        # atleast_1d rather than `... if y_pred.dim() > 0 else 1`: a 0-dim
        # prediction IS one row, and writing that as a fallback literal would both
        # READ like a neutral default and be flagged as one. This repository runs a
        # static sweep for neutral literals produced under an emptiness guard, keyed
        # on the enclosing name, and a Fairness* class returning a bare 1 there is
        # exactly the shape it looks for. The torch primitive says the same thing with
        # no literal to mistake.
        n_rows = int(torch.atleast_1d(y_pred).shape[0])
        n_used = n_rows
        reason: Optional[str] = None
        cause = "an empty batch, or every sample_weight zero or non-finite"
        if n_rows == 0:
            n_used = 0
            reason = "empty_batch"
            cause = f"the batch holds {n_rows} row(s)"
        elif sample_weight is not None:
            weights = sample_weight.detach()
            # ABOVE THE numel() DISPATCH, because all three spellings of the
            # weight tensor below share this precondition and a guard inside one
            # of them leaves the same fabrication live in its siblings.
            #
            # A NEGATIVE WEIGHT DOES NOT WEIGH A ROW LESS, IT INVERTS IT (BGL
            # wave 4, 2026-09-30). The task loss is minimised, so a row entering
            # it with a negative weight is a row the optimizer is PAID to get
            # wrong, and a weight vector whose negatives cancel its positives
            # reports a perfect fit from a model that is wrong on every row.
            # Measured on the four-row batch of this file's own docstring
            # (y_pred [0.9, 0.8, 0.2, 0.1], DemographicParityLoss,
            # lambda_fairness=0.1), whose honest task loss is
            # 0.16425204277038574:
            #
            #   sample_weight=torch.tensor(-1.)          -> task_loss
            #       -0.16425204277038574, task_rows_used 4 of 4, ZERO warnings.
            #       Minimising -0.164 maximises the prediction error on all
            #       four rows.
            #   sample_weight=torch.full((4,), -1.)      -> the same
            #   sample_weight=torch.tensor([1., 1., -1., -1.])
            #       -> task_loss 5.587935447692871e-09, task_rows_used 4 of 4,
            #       ZERO warnings: the two inverted rows cancel the two honest
            #       ones, and this docstring calls 0.0 the score a PERFECT
            #       predictor earns, so a model wrong by the same amount on
            #       both halves published a perfect fit.
            #
            # The whole batch is reported NOT MEASURED rather than the negative
            # rows alone, because the reduction has already averaged them
            # together: what reaches the optimizer is one number with the
            # inverted rows inside it, and no subset of it is a measurement of
            # fit. This library has already shipped eight reversed
            # fit(y_true, y_prob) calls of exactly this character.
            n_negative = int((torch.isfinite(weights) & (weights < 0)).sum())
            if weights.numel() == 0:
                n_used = 0
                reason = "empty_sample_weight"
                cause = "sample_weight holds no values"
            elif n_negative:
                n_used = 0
                reason = "negative_sample_weight"
                cause = (
                    f"{n_negative} sample_weight value(s) are negative, which does "
                    f"not weigh a row down but inverts it: the task loss is "
                    f"MINIMISED, so those rows are ones the optimizer is paid to "
                    f"get wrong, and their contribution cancels the honest rows "
                    f"instead of adding to them"
                )
            elif weights.numel() == 1:
                # A ONE-ELEMENT WEIGHT IS BROADCAST OVER EVERY ROW (BGL4 audit,
                # 2026-09-27). `weights.numel() % n_rows == 0` below is a test on
                # the weight's SHAPE, and 1 % 4 is 1, so a scalar weight fell
                # through it with n_used left at n_rows although torch had
                # multiplied every row of the loss by that single number.
                # Measured on the four-row batch of this file's own docstring
                # (y_pred [0.9, 0.8, 0.2, 0.1], lambda_fairness=0.1):
                #
                #   sample_weight=torch.zeros(4)  -> task_loss nan, used 0, warned
                #   sample_weight=torch.zeros(1)  -> task_loss 0.0, used 4, SILENT
                #   sample_weight=torch.tensor(0.) -> task_loss 0.0, used 4, SILENT
                #
                # so the SAME batch, every row of it multiplied by zero, was
                # refused in one spelling of the weight and reported as a
                # measured perfect fit in the other two. Three such batches gave
                # end_epoch an avg_task_loss of 0.0. After: all three spellings
                # report task_loss nan with task_rows_used 0 and the warning, and
                # the epoch averages to nan. The controls are unchanged:
                # torch.ones(1) and torch.tensor(2.0) still measure
                # 0.16425204277038574 and 0.3285040855407715 silently.
                # Its own name, not the `alive` tensor of the branch below: one
                # element decides EVERY row here, so this is a single bool.
                broadcast_alive = bool(self._weights_are_usable(weights).all())
                n_used = n_rows if broadcast_alive else 0
                if not broadcast_alive:
                    reason = "no_usable_sample_weight"
                    cause = (
                        f"the single sample_weight value broadcast over every row "
                        f"is zero, non-finite, or below the resolution of "
                        f"{weights.dtype}"
                    )
            elif weights.numel() % n_rows == 0:
                # Read per ROW so a column-shaped weight tensor is judged the
                # same way. A weight that does not divide by the row count
                # cannot be read per row, and guessing there would be worse
                # than saying nothing, so the count is left at n_rows.
                per_row = weights.reshape(n_rows, -1)
                alive = self._weights_are_usable(per_row).any(dim=1)
                n_used = int(alive.sum().item())
                if n_used == 0:
                    reason = "no_usable_sample_weight"
                    cause = (
                        f"every sample_weight value is zero, non-finite, or below "
                        f"the resolution of {weights.dtype}"
                    )

        if n_used == 0:
            # Recorded for _record_task_coverage to stamp: WHICH of the causes
            # it was is the difference between a batch nobody weighted and a
            # batch whose weights inverted it, and a reader of
            # LossComponents.batch_metrics cannot see the warning.
            self._task_loss_unassessed_reason = reason
            warnings.warn(
                f"{type(self).__name__}: 0 of {n_rows} row(s) entered the task "
                f"loss ({cause}), so the task loss is NOT MEASURED. It is reported "
                f"as NaN in LossComponents.task_loss, not as the 0.0 the "
                f"reduction produces from no rows, which is the value a perfect "
                f"predictor earns; the finite value the optimizer saw is kept in "
                f"batch_metrics['task_loss_unassessed_value'].",
                UserWarning,
                stacklevel=3,
            )
        else:
            self._task_loss_unassessed_reason = None
        return n_rows, n_used

    @staticmethod
    def _report_scalar(value: "torch.Tensor") -> float:
        """One float for :class:`LossComponents`, whatever the reduction left.

        ``LossComponents`` fields are typed ``float`` and were filled with
        ``.item()``, which torch refuses for any tensor that is not exactly one
        element. ``reduction="none"`` is a DOCUMENTED option of this class
        (``Literal["mean", "sum", "none"]``), and it leaves the task loss with
        one value per row. Measured 2026-09-27 (BGL4 audit) on the four-row
        batch above, through a direct subclass with lambda_fairness=0.1 and
        ``reduction="none"``:

            before -> RuntimeError: a Tensor with 4 elements cannot be
                      converted to Scalar, raised from forward() whenever
                      track_metrics is on (the default) or components were
                      asked for, so the option could not be used at all
            after  -> task_loss 0.16425204277038574, the mean of the four
                      per-row losses, with batch_metrics["reduction"] = "none"
                      beside it so the aggregation is not mistaken for a
                      reduction the caller chose

        The returned TENSOR is untouched and still carries one value per row:
        this affects only the recorded summary. An unreduced EMPTY batch has no
        value to summarise and gets NaN, the same could-not-check the coverage
        counter reports for it, rather than a 0.0 that would read as a perfect
        fit over no rows.
        """
        detached = value.detach()
        if detached.numel() == 1:
            return float(detached.reshape(()))
        if detached.numel() == 0:
            return float("nan")
        return float(detached.mean())

    def _record_task_coverage(
        self,
        components: LossComponents,
        n_rows: int,
        n_used: int,
    ) -> LossComponents:
        """Stamp the task-loss coverage onto ``components``, in place.

        ``total_loss`` deliberately stays finite: like
        ``regularization_value`` in the regularizers, it describes the tensor
        the optimizer really saw, and that is a true statement about the step.
        It is ``task_loss``, the field read as a MEASUREMENT of fit, that
        becomes NaN.
        """
        components.batch_metrics["task_rows_total"] = n_rows
        components.batch_metrics["task_rows_used"] = n_used
        components.batch_metrics["task_loss_assessed"] = n_used > 0
        if n_used == 0:
            components.batch_metrics["task_loss_unassessed_value"] = components.task_loss
            # WHICH could-not-check it was, where a reader of batch_metrics can
            # see it. "Nobody weighted this batch" and "the weights inverted the
            # objective" are the same NaN and are not the same finding.
            components.batch_metrics["task_loss_unassessed_reason"] = (
                self._task_loss_unassessed_reason
            )
            components.task_loss = float("nan")
        return components

    def _record_lambda_application(self, components: LossComponents) -> LossComponents:
        """Stamp whether the fairness penalty was actually APPLIED, in place.

        ``fairness_loss`` is the penalty that was MEASURED; it says nothing
        about whether the optimizer ever paid it. ``total_loss`` is
        ``task_loss + effective_lambda * fairness_loss``, so an effective lambda
        of 0.0 leaves a measured disparity sitting in the record while the
        intervention is off. See :meth:`_get_effective_lambda` for the
        measurement, and note that a neutered mitigation reports a BETTER
        fairness number, not a worse one.
        """
        effective_lambda = components.batch_metrics.get("effective_lambda")
        applied = bool(effective_lambda)
        components.batch_metrics["fairness_penalty_applied"] = applied
        note = self._fairness_lambda_note
        components.batch_metrics["fairness_penalty_not_applied_reason"] = (
            None if applied else (note[0] if note else "effective_lambda_zero")
        )
        # float(), not isinstance(v, (int, float)): that test REJECTS np.float32
        # and np.float64, so it would discard a real lambda while reading as
        # caution. Nothing else here can arrive in this field.
        contribution = 0.0
        if applied and effective_lambda is not None:
            contribution = float(effective_lambda) * components.fairness_loss
        components.batch_metrics["fairness_penalty_contribution"] = contribution
        return components

    def _get_effective_lambda(self) -> float:
        """Get effective lambda considering warmup.

        As documented on `warmup_epochs` ("Number of epochs before applying
        fairness penalty"), the fairness penalty is fully OFF while
        _current_epoch < warmup_epochs, and the full lambda applies from the
        first post-warmup epoch. The previous linear ramp applied a partial
        penalty from epoch 0, contradicting the documented contract.

        A MITIGATION THAT IS SWITCHED OFF IS NOT A MITIGATION OF ZERO SIZE
        (BGL wave 4, 2026-09-30). Returning 0.0 here removes the fairness term
        from ``total_loss`` entirely, and every caller still receives
        ``LossComponents.fairness_loss`` holding the disparity it measured, as
        though it were being paid for. ``set_epoch()`` is a SEPARATE manual
        call, so a caller who configures ``warmup_epochs`` and never makes it
        trains the whole run with the penalty off. Measured on a 40-row batch
        with a real 0.2 demographic-parity disparity,
        DemographicParityLoss(lambda_fairness=0.5):

            warmup_epochs=0                     -> total 0.456675, task
                0.356675, fairness_loss 0.200000, effective_lambda 0.5
            warmup_epochs=5, set_epoch NEVER called -> total 0.356675, i.e.
                EXACTLY the task loss, fairness_loss still 0.200000,
                effective_lambda 0.0, ZERO warnings
            lambda_fairness=0.0                 -> the same, ZERO warnings
            track_metrics=False, plain tensor call during warmup -> total
                0.356675, zero warnings and NO components at all

        The warning is raised HERE, in the one function every loss in this
        package asks for its lambda, rather than in any single ``forward``:
        there are four ``forward`` implementations and a guard in one of them
        leaves the others silent. It is also outside the ``if return_components
        or self.track_metrics`` block for the reason recorded on the task-loss
        coverage: the returned TENSOR carries the suppression whether or not the
        caller asked for components or switched tracking off.
        """
        if self._current_epoch < self.warmup_epochs:
            self._fairness_lambda_note = (
                "warmup",
                f"epoch {self._current_epoch} of a {self.warmup_epochs}-epoch "
                f"warmup, so the fairness penalty is multiplied by 0.0 and the "
                f"intervention is OFF for this batch. set_epoch() is a separate "
                f"call: if it is never made, the penalty stays off for the whole "
                f"run while LossComponents.fairness_loss keeps reporting the "
                f"disparity it measured",
            )
            self._warn_lambda_suppressed()
            return 0.0
        if not self.lambda_fairness:
            self._fairness_lambda_note = (
                "lambda_fairness_zero",
                "lambda_fairness is 0.0, so the fairness penalty is multiplied "
                "out of the total loss and the intervention is OFF. This is a "
                "legal no-penalty baseline; it is named here because "
                "LossComponents.fairness_loss still reports the disparity that "
                "was measured, which reads as a penalty that was applied",
            )
            self._warn_lambda_suppressed()
            return 0.0
        self._fairness_lambda_note = None
        return self.lambda_fairness

    def _warn_lambda_suppressed(self) -> None:
        """Say out loud that this batch's fairness penalty was not applied."""
        note = self._fairness_lambda_note
        if note is None:
            return
        warnings.warn(
            f"{type(self).__name__}: {note[1]}. batch_metrics["
            f"'fairness_penalty_applied'] is False and "
            f"batch_metrics['fairness_penalty_contribution'] is 0.0 for this "
            f"batch; a fairness penalty that is switched off makes the fairness "
            f"numbers LOOK BETTER, not worse, because the model is no longer "
            f"being steered.",
            UserWarning,
            stacklevel=3,
        )

    def forward(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        sample_weight: Optional["torch.Tensor"] = None,
        return_components: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", LossComponents]]:
        """
        Compute the total fairness-aware loss.

        Args:
            y_pred: Predicted probabilities (after sigmoid for binary)
            y_true: True labels
            sensitive_attr: Sensitive attribute values
            sample_weight: Optional per-sample weights
            return_components: Whether to return detailed components

        Returns:
            Total loss tensor, optionally with LossComponents
        """
        check_torch_available()

        task_loss = self._compute_task_loss(y_pred, y_true, sample_weight)

        # Counted and warned about outside the components block, because the
        # returned TENSOR carries the same 0.0 whether or not the caller asked
        # for components or switched tracking off.
        n_rows, n_used = self._task_loss_coverage(y_pred, sample_weight)

        fairness_loss = self._compute_fairness_penalty(y_pred, y_true, sensitive_attr)

        effective_lambda = self._get_effective_lambda()

        total_loss = task_loss + effective_lambda * fairness_loss

        if return_components or self.track_metrics:
            components = LossComponents(
                # _report_scalar and not .item(): see its docstring for the
                # measured RuntimeError that reduction="none" raised here.
                total_loss=self._report_scalar(total_loss),
                task_loss=self._report_scalar(task_loss),
                fairness_loss=self._report_scalar(fairness_loss),
                batch_metrics={
                    "effective_lambda": effective_lambda,
                    "epoch": self._current_epoch,
                    # Recorded because it decides whether the three floats above
                    # are the value the optimizer saw or a mean over the rows.
                    "reduction": self.reduction,
                },
            )
            self._record_task_coverage(components, n_rows, n_used)
            self._record_lambda_application(components)
            if self.track_metrics:
                self._batch_history.append(components)

        if return_components:
            return total_loss, components
        return total_loss

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for warmup scheduling.

        THIS SETTER DECIDES WHETHER THE FAIRNESS PENALTY IS APPLIED AT ALL, and
        it used to accept anything (BGL grade wave, 2026-09-30). The value it
        writes is read by ``_get_effective_lambda`` as ``self._current_epoch <
        self.warmup_epochs`` and is published as ``TrainingMetrics.epoch``.
        Measured on a 40-row batch with DemographicParityLoss(
        lambda_fairness=0.5), the four out-of-domain values it took:

            warmup_epochs=0, set_epoch(-1)   -> effective_lambda 0.0,
                fairness_penalty_applied False, i.e. the intervention is OFF
                although NO warmup was configured, explained by a warning
                reading "epoch -1 of a 0-epoch warmup"
            warmup_epochs=0, set_epoch(-100) -> the same
            warmup_epochs=5, set_epoch(nan)  -> effective_lambda 0.5,
                fairness_penalty_applied True, ZERO warnings: `nan < 5` is
                False, so the configured warmup was silently SKIPPED, and
                TrainingMetrics.epoch then carried nan, which compares False
                against every threshold it is later tested against
            warmup_epochs=5, set_epoch(inf)  -> the same skip
            set_epoch('3') / set_epoch(None) -> TypeError raised later, from
                inside _get_effective_lambda during forward, after
                _current_epoch had already been overwritten

        So one direction switched the mitigation off without a warmup and the
        other switched the warmup off without a word, from the same
        unvalidated assignment. The domain is refused here rather than
        disclosed, because an epoch number is the caller's own bookkeeping and
        not a property of the data: there is nothing to measure and nothing to
        degrade to.

        Read through ``float()`` rather than ``isinstance(epoch, int)``: that
        test REJECTS ``np.int64`` and a 0-dim torch scalar, which every
        ordinary training loop produces, so it would refuse real callers while
        reading as caution.
        """
        try:
            value = float(epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{type(self).__name__}.set_epoch: epoch must be a number, got "
                f"{epoch!r}. It is compared against warmup_epochs to decide "
                f"whether the fairness penalty is applied, so a value that "
                f"cannot be compared raises from inside the loss instead."
            ) from exc
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"{type(self).__name__}.set_epoch: epoch must be finite, got "
                f"{epoch!r}. The warmup test is `epoch < warmup_epochs`, which "
                f"is False for NaN and for +inf, so a non-finite epoch SKIPS a "
                f"configured warmup in silence and is then published as "
                f"TrainingMetrics.epoch."
            )
        if value < 0:
            raise ValueError(
                f"{type(self).__name__}.set_epoch: epoch must be at least 0, got "
                f"{epoch!r}. A negative epoch is inside every warmup window, "
                f"including warmup_epochs=0 where no warmup was configured, so "
                f"it switches the fairness penalty OFF while "
                f"LossComponents.fairness_loss keeps reporting the disparity it "
                f"measured. A switched-off mitigation makes the fairness numbers "
                f"LOOK BETTER, not worse."
            )
        if value != int(value):
            raise ValueError(
                f"{type(self).__name__}.set_epoch: epoch must be a whole number, "
                f"got {epoch!r}. It counts completed epochs and is published as "
                f"TrainingMetrics.epoch."
            )
        self._current_epoch = int(value)

    def end_epoch(self) -> Optional[TrainingMetrics]:
        """
        End the current epoch and compute aggregated metrics.

        Returns:
            TrainingMetrics for the epoch if tracking is enabled
        """
        if not self.track_metrics or not self._batch_history:
            return None

        # Aggregate batch metrics
        n_batches = len(self._batch_history)
        avg_total = sum(b.total_loss for b in self._batch_history) / n_batches
        avg_task = sum(b.task_loss for b in self._batch_history) / n_batches
        avg_fairness = sum(b.fairness_loss for b in self._batch_history) / n_batches

        metrics = TrainingMetrics(
            epoch=self._current_epoch,
            avg_total_loss=avg_total,
            avg_task_loss=avg_task,
            avg_fairness_loss=avg_fairness,
        )

        self._epoch_metrics.append(metrics)
        self._batch_history = []

        return metrics

    def get_training_history(self) -> List[TrainingMetrics]:
        """Get the training history across epochs."""
        return self._epoch_metrics.copy()

    def reset_history(self) -> None:
        """Reset training history."""
        self._batch_history = []
        self._epoch_metrics = []
        self._current_epoch = 0


def create_group_masks(
    sensitive_attr: "torch.Tensor",
) -> Dict[Any, "torch.Tensor"]:
    """
    Create boolean masks for each group in the sensitive attribute.

    A row whose attribute value is NaN belongs to no group: NaN is not equal
    to itself, so ``sensitive_attr == nan`` selects nothing. Such rows are
    EXCLUDED from the returned masks and the exclusion is named in a
    UserWarning, because the masks are what every caller in this package
    iterates to build a group comparison, and a comparison that covers fewer
    rows than the batch is a partial measurement.

    Args:
        sensitive_attr: Tensor of sensitive attribute values

    Returns:
        Dictionary mapping group values to boolean masks. Every mask in it has
        at least one row.
    """
    check_torch_available()

    unique_groups = torch.unique(sensitive_attr)
    masks = {}
    # A GROUP KEY WITH NO ROWS UNDER IT IS NOT A GROUP.
    #
    # torch.unique keeps NaN, and NaN != NaN, so a single unreadable attribute
    # value used to add a third key whose mask was all-False. Measured before
    # this fix on sensitive_attr [0.0, 1.0, nan, 1.0]:
    #   create_group_masks -> {0.0: [T,F,F,F], 1.0: [F,T,F,T],
    #                          nan: [F,F,F,F]}   and zero warnings
    #   compute_group_rates -> {0.0: 0.899, 1.0: 0.5, nan: 0.0}
    # so a demographic parity penalty compared a fabricated 0.0 rate for a
    # group holding no rows, and the row that carried the NaN sat in no mask
    # at all while still counting toward the population mean it was compared
    # against.
    unmatched = 0
    for group in unique_groups:
        mask = sensitive_attr == group
        if not bool(mask.any()):
            unmatched += 1
            continue
        masks[group.item()] = mask

    if unmatched:
        n_rows = int(sensitive_attr.numel())
        if masks:
            covered = int(torch.stack(list(masks.values())).any(dim=0).sum().item())
        else:
            covered = 0
        warnings.warn(
            f"create_group_masks: {n_rows - covered} of {n_rows} row(s) carry an "
            f"attribute value that matches no group (NaN is not equal to "
            f"itself), so they are in NONE of the {len(masks)} mask(s) returned "
            f"and are NOT MEASURED by any group comparison built from them. "
            f"Any disparity computed from these masks covers {covered} of "
            f"{n_rows} row(s). Drop or impute the unreadable rows before "
            f"comparing groups.",
            UserWarning,
            stacklevel=2,
        )
    return masks


def compute_group_rates(
    y_pred: "torch.Tensor",
    y_true: "torch.Tensor",
    sensitive_attr: "torch.Tensor",
    rate_type: Literal["positive_rate", "tpr", "fpr", "tnr", "fnr"] = "positive_rate",
) -> Dict[Any, "torch.Tensor"]:
    """
    Compute soft rates for each demographic group.

    Only groups that HAVE rows get a rate. A row whose attribute value is
    unreadable (NaN) is in no group, and ``create_group_masks`` names how many
    such rows there were in a UserWarning rather than returning a group with
    an empty mask, whose rate used to come back as a fabricated 0.0.

    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        sensitive_attr: Sensitive attribute values
        rate_type: Type of rate to compute

    Returns:
        Dictionary mapping groups to their rate values
    """
    check_torch_available()

    masks = create_group_masks(sensitive_attr)
    rates = {}
    for group, mask in masks.items():
        rates[group] = soft_rate_computation(y_pred, y_true, mask, rate_type=rate_type)
    return rates
