"""
Fairness-Aware Regularization Techniques.

This module provides regularization techniques that can be added to any
differentiable loss function to encourage fairness during training.
These regularizers penalize statistical dependence between model predictions
and sensitive attributes.

Regularizers Implemented:
    1. StatisticalParityRegularizer: Penalizes correlation with sensitive attr
    2. ConditionalIndependenceRegularizer: Enforces conditional independence
    3. GroupFairnessRegularizer: General group fairness penalty
    4. HilbertSchmidtRegularizer: HSIC-based independence regularizer
    5. MutualInformationRegularizer: MI-based regularizer
    6. CorrelationPenalty: Simple correlation-based penalty

Mathematical Background:
    Regularization adds a penalty term to the loss:
        L_total = L_task + λ * R(ŷ, a)

    where R measures dependence between predictions ŷ and sensitive attribute a.

References:
    - Kamishima et al. (2012): Fairness-Aware Classifier with Prejudice Remover
    - Zafar et al. (2017): Fairness Beyond Disparate Treatment
    - Gretton et al. (2005): Measuring Statistical Dependence with HSIC
    - Louizos et al. (2016): The Variational Fair Autoencoder
"""

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from vfairness._not_assessed import warn_not_assessed

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401  # availability probe

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class _TorchNNStub:  # torch-absence stub
        class Module:
            pass

    nn = _TorchNNStub  # type: ignore[assignment]  # runtime fallback stands in for torch.nn

# Base class for nn.Module-derived types. Real nn.Module when torch is present,
# else `object`. For type checking we always treat it as nn.Module.
if TYPE_CHECKING:
    from torch.nn import Module as _ModuleBase
else:
    _ModuleBase = nn.Module if TORCH_AVAILABLE else object


def check_torch_available():
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for regularizers. Install with: pip install torch")


class RegularizerType(Enum):
    """Types of fairness regularizers."""

    STATISTICAL_PARITY = "statistical_parity"
    CONDITIONAL_INDEPENDENCE = "conditional_independence"
    GROUP_FAIRNESS = "group_fairness"
    HSIC = "hsic"
    MUTUAL_INFORMATION = "mutual_information"
    CORRELATION = "correlation"


@dataclass
class RegularizerMetrics:
    """
    Metrics from regularizer computation.

    THREE STATES, NEVER TWO. Read ``dependence_measure`` and ``measured``
    together, not one at a time:

        dependence_measure = 0.031, measured = True    measured, some dependence
        dependence_measure = 0.0,   measured = True    measured, independent
        dependence_measure = NaN,   measured = False   NOT MEASURED

    The third row is the one to code against. Every regularizer in this module
    scores dependence on a scale where **0.0 is the clean end**: HSIC "is zero
    iff the variables are independent", a Pearson correlation of 0.0 is exactly
    "predictions carry no information about the sensitive attribute", and a
    group-parity penalty of 0.0 is "every group agrees with the population".
    Until 2026-09-16 each of these classes returned ``RegularizerMetrics(0.0,
    0.0)`` on inputs where nothing could be estimated at all (a batch of fewer
    than two rows, a sensitive attribute with a single value, a constant
    prediction vector, a conditioning set with no rows in it), so the best
    possible reading on the scale was handed to the caller as a finding. No
    warning was emitted and, in the single-group case, ``group_penalties`` was
    even populated with an agreeing ``{'g0': 0.0}``, so there was no tell at
    all.

    ``regularization_value`` deliberately stays 0.0 in that third state: it
    describes the penalty tensor that really was added to the loss, and a zero
    contribution is a true statement about the optimisation step. It is
    ``dependence_measure``, the field documented as a MEASUREMENT, that becomes
    NaN.

    Consumers that average, rank or plot a run's history must filter on
    ``measured``; a NaN will otherwise poison an aggregate, which is the loud
    failure and is on purpose.

    Attributes:
        regularization_value: The regularization penalty actually applied.
        dependence_measure: Measure of dependence between predictions and
            sensitive attr, or NaN when it could not be estimated.
        group_penalties: Per-group penalty contributions. May be ``{}`` both
            because no group was measurable and because the regularizer does
            not report per-group values (HSIC, CorrelationPenalty), so it is
            NOT a reliable tell on its own; ``measured`` is.
        metadata: Additional metadata. Carries ``not_assessed`` naming the
            reason whenever ``measured`` is False.
        measured: False when nothing was estimated. Never collapse this into a
            0.0 dependence.
    """

    regularization_value: float
    dependence_measure: float
    group_penalties: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    measured: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regularization_value": self.regularization_value,
            "dependence_measure": self.dependence_measure,
            "group_penalties": self.group_penalties,
            "metadata": self.metadata,
            # The third state travels with the object, so a consumer that only
            # ever sees the dict (JSON, a logged history, a plot payload) can
            # still tell "no dependence" from "no measurement".
            "measured": self.measured,
        }


def _not_measured_metrics(
    reason: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> RegularizerMetrics:
    """The one shape a regularizer in this module reports when it measured nothing.

    ``regularization_value`` is 0.0 because a zero-valued penalty really was
    added to the loss; ``dependence_measure`` is NaN because no dependence was
    estimated. Built here rather than at each of the six call sites so the two
    ends of the refusal cannot drift apart.
    """
    metadata: Dict[str, Any] = {"not_assessed": reason}
    if extra:
        metadata.update(extra)
    return RegularizerMetrics(
        regularization_value=0.0,
        dependence_measure=float("nan"),
        group_penalties={},
        metadata=metadata,
        measured=False,
    )


def _readable_rows(
    y_pred: "torch.Tensor",
    *others: Optional["torch.Tensor"],
) -> Tuple[int, int]:
    """(rows in the batch, rows finite in EVERY tensor passed).

    Pass ONLY the tensors the caller actually consumes. ``y_true`` is a
    parameter of every ``forward`` in this module and is read by two of the
    six, so handing it to this helper from a regularizer that ignores it would
    refuse a batch over a column it never looks at.

    The row count comes from ``y_pred``, and each tensor is reshaped to
    (n_rows, -1) so a column-shaped input is judged per ROW. A tensor whose
    size does not divide by the row count cannot be read per row and is
    skipped rather than guessed at.
    """
    n_rows = int(y_pred.shape[0]) if y_pred.dim() > 0 else 1
    if n_rows == 0:
        return 0, 0
    usable = torch.ones(n_rows, dtype=torch.bool, device=y_pred.device)
    for tensor in (y_pred, *others):
        # An EMPTY tensor beside a non-empty batch (a caller passing y_true=[]
        # with 200 rows) says nothing per row, and reshaping it to (n_rows, -1)
        # raises, so it is skipped: the branch that handles a missing label
        # column refuses on its own further down.
        if tensor is None or tensor.numel() < n_rows or tensor.numel() % n_rows != 0:
            continue
        usable = usable & torch.isfinite(tensor.float()).reshape(n_rows, -1).all(dim=1)
    return n_rows, int(usable.sum().item())


def _has_spread(values: "torch.Tensor") -> bool:
    """True when ``values`` holds at least two DISTINCT finite values.

    Tested on the RAW vector, never on an accumulated statistic. The sum of
    squared deviations of a constant float vector is not exactly zero:
    measured, ``torch.full((30,), 0.7)`` centred and squared sums to 4.3e-10,
    so the obvious guard ``(centred ** 2).sum() == 0`` would MISS the constant
    prediction vector it exists to catch and let an undefined 0/0 Pearson
    through as a correlation of 0.0.
    """
    finite = values[torch.isfinite(values)]
    if finite.numel() < 2:
        return False
    return bool(torch.unique(finite).numel() >= 2)


def _warn_on_degenerate_strength(strength: Any, owner: str) -> None:
    """Say so when the PENALTY ITSELF has been switched off or reversed.

    A REGULARIZER IS A MITIGATION, so its failure mode is inverted: a penalty
    that has stopped working produces a BETTER fairness number, not a worse one,
    because the model is simply left as it was (or pushed the other way) while
    the ``dependence_measure`` this module reports keeps measuring correctly and
    keeps saying ``measured=True``. Every guard elsewhere in this file is aimed
    at the MEASUREMENT; this one is aimed at the INTERVENTION'S OWN PARAMETER,
    which is the only place the degeneracy is visible.

    MEASURED on this tree, 2026-09-30, on 200 rows whose predictions depend on
    the sensitive attribute by construction (rates 0.2 and 0.8), taking
    ``d(penalty)/d(y_pred)`` for each of the four concrete regularizers::

        strength   penalty    |grad|     what the optimiser is told
        0.1        +0.029447  0.100000   reduce the dependence      (correct)
        0.0         0.000000  0.000000   nothing at all
        -0.5       -0.147234  0.500000   INCREASE the dependence

    All three were accepted in silence, and so were ``nan``, ``inf``, ``True``
    and the string ``"0.1"``. The ``0.0`` row is a training run that reports a
    fairness regularizer in its configuration and applies none: the loss is
    unchanged, the gradient is exactly zero, and ``RegularizerMetrics`` still
    reports a real dependence with ``measured=True``, so nothing downstream can
    tell it from a run that was regularised and converged. The ``-0.5`` row is
    worse: the gradient is reversed, so the optimiser is being pushed toward
    MORE dependence on the protected attribute under the name of a fairness
    penalty.

    Warnings rather than exceptions, deliberately. A negative or zero strength
    is a legitimate experiment (an ablation, a sign study), and refusing to
    construct the object would break that; what it may not do is happen quietly.
    Placed in ``BaseRegularizer.__init__``, which all five concrete classes
    reach through ``super().__init__(strength=..., track_metrics=...)``, so a
    sixth regularizer inherits the check instead of having to remember it.
    """
    if isinstance(strength, bool):
        warnings.warn(
            f"{owner}(strength={strength!r}): a bool is not a regularization strength. "
            f"True is 1.0 in Python, so this run applies a full-weight penalty, and False "
            f"is 0.0, which applies none at all and contributes no gradient.",
            UserWarning,
            stacklevel=3,
        )
        return
    import numbers

    # ``numbers.Real`` rather than ``isinstance(x, float)``: the numpy scalars
    # register with it and multiply a tensor perfectly well, so testing for the
    # Python type would refuse a legitimate ``np.float32``. What it excludes is
    # exactly what cannot scale a tensor: ``str`` (``float("0.1")`` succeeds, so
    # a convertibility test would pass it, and ``tensor * "0.1"`` then raises
    # deep inside forward) and ``Decimal``.
    if not isinstance(strength, numbers.Real):
        warnings.warn(
            f"{owner}(strength={strength!r}): the strength is a {type(strength).__name__}, "
            f"not a real number, so it cannot scale the penalty tensor and the multiplication "
            f"inside forward() will raise. Pass a float. A numeric STRING is the trap here: "
            f"float() accepts it, so a convertibility check passes and the tensor arithmetic "
            f"still does not.",
            UserWarning,
            stacklevel=3,
        )
        return
    try:
        value = float(strength)
    except (TypeError, ValueError):  # pragma: no cover - Real that will not cast
        warnings.warn(
            f"{owner}(strength={strength!r}): the strength is not a number, so the penalty "
            f"cannot be scaled by it and the multiplication will fail inside forward(). "
            f"Pass a float.",
            UserWarning,
            stacklevel=3,
        )
        return
    if math.isnan(value) or math.isinf(value):
        warnings.warn(
            f"{owner}(strength={strength!r}): a non-finite strength makes the penalty "
            f"non-finite, so every gradient computed from the total loss is non-finite too "
            f"and the optimiser updates nothing usable. This is not a stronger penalty.",
            UserWarning,
            stacklevel=3,
        )
        return
    if value == 0.0:
        warnings.warn(
            f"{owner}(strength=0.0): the penalty is identically zero and its gradient with "
            f"respect to the predictions is exactly zero, so THIS REGULARIZER APPLIES NO "
            f"MITIGATION. The reported dependence_measure is still a real measurement and "
            f"still says measured=True, so a run configured this way is indistinguishable "
            f"downstream from a regularised one. If the ablation is intended, say so where "
            f"the run is recorded.",
            UserWarning,
            stacklevel=3,
        )
        return
    if value < 0.0:
        warnings.warn(
            f"{owner}(strength={value!r}): a negative strength REVERSES the penalty's "
            f"gradient, so minimising the total loss now MAXIMISES the dependence between "
            f"the predictions and the sensitive attribute. Measured on this repo, "
            f"strength=-0.5 gave a penalty of -0.147 with |grad| 0.500 pointing the other "
            f"way. That is an anti-fairness term carrying a fairness name.",
            UserWarning,
            stacklevel=3,
        )


class BaseRegularizer(_ModuleBase):
    """
    Abstract base class for fairness regularizers.

    All regularizers compute a penalty term that measures and penalizes
    the statistical dependence between model outputs and sensitive attributes.

    Args:
        strength: Regularization strength (lambda)
        track_metrics: Whether to track detailed metrics

    Usage:
        >>> regularizer = StatisticalParityRegularizer(strength=0.1)
        >>> penalty = regularizer(y_pred, sensitive_attr)
        >>> total_loss = task_loss + penalty
    """

    def __init__(
        self,
        strength: float = 0.1,
        track_metrics: bool = True,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        _warn_on_degenerate_strength(strength, type(self).__name__)
        self.strength = strength
        self.track_metrics = track_metrics
        self._history: List[RegularizerMetrics] = []

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """
        Compute regularization penalty.

        Args:
            y_pred: Predicted probabilities or logits
            sensitive_attr: Sensitive attribute values
            y_true: Optional true labels (for conditional regularizers)
            return_metrics: Whether to return detailed metrics

        Returns:
            Regularization penalty, optionally with metrics
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_history(self) -> List[RegularizerMetrics]:
        """Get regularization history."""
        return self._history.copy()

    def reset_history(self):
        """Reset history."""
        self._history = []


class StatisticalParityRegularizer(BaseRegularizer):
    """
    Statistical Parity (Demographic Parity) Regularizer.

    Penalizes the difference in mean predictions between demographic groups,
    encouraging the model to produce similar prediction distributions regardless
    of group membership.

    Mathematical formulation:
        R = Σ_g |E[ŷ|G=g] - E[ŷ]|

    Args:
        strength: Regularization strength
        reduction: How to aggregate group penalties ('mean', 'max', 'sum')
        track_metrics: Whether to track metrics

    Example:
        >>> regularizer = StatisticalParityRegularizer(strength=0.1)
        >>> penalty = regularizer(y_pred, sensitive_attr)
        >>> loss = bce_loss(y_pred, y_true) + penalty

    Note:
        This regularizer enforces demographic parity but may conflict with
        accuracy when base rates differ between groups.

    References:
        - Zafar et al. (2017): Fairness Beyond Disparate Treatment

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: statistical_parity_regularizer. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        strength: float = 0.1,
        reduction: Literal["mean", "max", "sum"] = "mean",
        track_metrics: bool = True,
    ):
        super().__init__(strength=strength, track_metrics=track_metrics)
        self.reduction = reduction

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """Compute statistical parity regularization."""
        check_torch_available()

        # A ROW THAT CANNOT BE READ CANNOT BE FOLDED INTO A GROUP MEAN.
        #
        # The same guard GroupFairnessRegularizer.forward already carries, and
        # the reason it carries it: a NaN dependence_measure with
        # measured=True contradicts the RegularizerMetrics contract at the top
        # of this module ("NaN and measured=False means NOT MEASURED"), so a
        # consumer that filters a run history on `measured` keeps the NaN and
        # the aggregate it poisons. Measured on this class before this guard,
        # every one with measured=True and zero warnings, on 40 rows and two
        # healthy groups whose real parity gap is 0.30:
        #
        #   one NaN prediction    -> dependence_measure nan,
        #        group_penalties {0.0: nan, 1.0: nan}
        #   one inf prediction    -> nan, with {0.0: nan, 1.0: inf}
        #   one NaN ATTRIBUTE row -> nan, and group_penalties carried a THIRD
        #        fabricated key: {0.0: 0.3, 1.0: 0.3, nan: nan}, because
        #        torch.unique keeps NaN and `sensitive_attr == nan` selects no
        #        row, so the mean of an empty slice became that group's
        #        "penalty".
        #
        # y_true is deliberately NOT checked: this regularizer never reads it.
        n_rows, n_usable = _readable_rows(y_pred, sensitive_attr)
        if n_usable < n_rows:
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "StatisticalParityRegularizer.forward",
                measured=n_usable,
                total=n_rows,
                unit="rows were finite in both y_pred and the attribute",
                requirement="every row must be readable before a group mean is taken",
                reporting="dependence_measure=nan and measured=False",
                instead_of=(
                    "a NaN reported as measured, or a gap over whichever rows "
                    "happened to be readable"
                ),
            )
            metrics = _not_measured_metrics(
                "non_finite_rows",
                extra={"n_rows": n_rows, "n_usable": n_usable},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        unique_groups = torch.unique(sensitive_attr)

        if len(unique_groups) < 2:
            # Same shape as the four filed siblings in this module, found while
            # fixing them: RegularizerMetrics(0.0, 0.0) on single-group input.
            # A statistical parity gap of 0.0 means "every group's mean
            # prediction equals the population mean", and with one group there
            # is no gap to measure. The penalty stays a differentiable 0.0.
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            n_groups = int(len(unique_groups))
            warn_not_assessed(
                "StatisticalParityRegularizer.forward",
                measured=n_groups,
                total=n_groups,
                unit="groups were present in the sensitive attribute",
                requirement="a statistical parity gap needs at least 2",
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means every group agrees",
            )
            metrics = _not_measured_metrics(
                "fewer_than_two_groups",
                extra={"n_groups": n_groups},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        overall_mean = y_pred.mean()

        group_penalties = {}
        penalties = []

        for group in unique_groups:
            mask = sensitive_attr == group
            group_mean = y_pred[mask].mean()
            group_penalty = torch.abs(group_mean - overall_mean)
            group_penalties[group.item()] = group_penalty.item()
            penalties.append(group_penalty)

        penalties_tensor = torch.stack(penalties)
        if self.reduction == "mean":
            total_penalty = penalties_tensor.mean()
        elif self.reduction == "max":
            total_penalty = penalties_tensor.max()
        else:
            total_penalty = penalties_tensor.sum()

        regularization = self.strength * total_penalty

        if self.track_metrics or return_metrics:
            metrics = RegularizerMetrics(
                regularization_value=regularization.item(),
                dependence_measure=total_penalty.item(),
                group_penalties=group_penalties,
                metadata={"overall_mean": overall_mean.item()},
            )
            if self.track_metrics:
                self._history.append(metrics)

        if return_metrics:
            return regularization, metrics
        return regularization


class ConditionalIndependenceRegularizer(BaseRegularizer):
    """
    Conditional Independence Regularizer.

    Enforces conditional independence between predictions and sensitive
    attributes given the true label: ŷ ⊥ a | y

    This corresponds to equalized odds when enforced strictly.

    Mathematical formulation:
        R = Σ_y Σ_g |E[ŷ|G=g,Y=y] - E[ŷ|Y=y]|

    Args:
        strength: Regularization strength
        conditional_on: Must be 'label'. forward() conditions on y_true
            and nothing else. 'prediction' (sufficiency, y independent of
            a given the prediction) and 'both' are not implemented and are
            refused with NotImplementedError.
        reduction: How to aggregate penalties
        track_metrics: Whether to track metrics

    Example:
        >>> regularizer = ConditionalIndependenceRegularizer(strength=0.1)
        >>> penalty = regularizer(y_pred, sensitive_attr, y_true)
        >>> loss = bce_loss(y_pred, y_true) + penalty

    Note:
        Requires y_true to be provided since we're conditioning on the label.

    References:
        - Hardt et al. (2016): Equality of Opportunity

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: conditional_independence_regularizer. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """

    IMPLEMENTED_CONDITIONING = ("label",)

    def __init__(
        self,
        strength: float = 0.1,
        conditional_on: Literal["label"] = "label",
        reduction: Literal["mean", "max", "sum"] = "mean",
        track_metrics: bool = True,
    ):
        super().__init__(strength=strength, track_metrics=track_metrics)
        # F5 (2026-09-09): `conditional_on` was stored and never read;
        # forward() always conditioned on y_true. Measured: identical
        # penalty (0.00104) for 'label', 'prediction', 'both' and 'garbage'.
        # Conditioning on the prediction would need the predictions binned
        # or thresholded, which is a different regularizer, so the option
        # is refused rather than aliased.
        if conditional_on in ("prediction", "both"):
            raise NotImplementedError(
                f"conditional_on={conditional_on!r} is not implemented; this "
                f"regularizer conditions on the true label only. Pass "
                f"conditional_on='label'."
            )
        if conditional_on not in self.IMPLEMENTED_CONDITIONING:
            raise ValueError(
                f"Unknown conditional_on {conditional_on!r}. Implemented: "
                f"{self.IMPLEMENTED_CONDITIONING}."
            )
        self.conditional_on = conditional_on
        self.reduction = reduction

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """Compute conditional independence regularization."""
        check_torch_available()

        # AN UNREADABLE ROW IS DROPPED FROM EVERY CELL AND FROM NO MEAN.
        #
        # This one produced a FINITE, clean-looking number, which is why it
        # outlived the NaN cases in the two siblings. Measured before this
        # guard on 40 rows, two groups, two labels, whose honest dependence is
        # 0.30, all with measured=True, metadata {} and zero warnings:
        #
        #   one NaN in the ATTRIBUTE -> dependence_measure 0.30000001192092866,
        #        BYTE IDENTICAL to the healthy answer. The NaN row matched no
        #        group (NaN != NaN), so it entered no (group, label) cell, yet
        #        it stayed inside `y_pred[label_mask].mean()`, the stratum mean
        #        every cell is compared against.
        #   one NaN in y_true        -> 0.30000001192092866 again, with the
        #        per-cell values shifted (0.3158 / 0.2842): the row fell out of
        #        every stratum silently, because `label_mask.sum() == 0` for
        #        the NaN "label" and the loop just continues.
        #   one NaN prediction       -> nan with measured=True, which breaks
        #        the module contract a consumer filters on.
        #
        # y_true is checked only when it is actually consumed: with y_true=None
        # this method falls back to statistical parity and never reads it.
        conditioning = y_true
        n_rows_read, n_usable = _readable_rows(y_pred, sensitive_attr)
        if conditioning is not None:
            _rows, n_with_label = _readable_rows(y_pred, sensitive_attr, conditioning)
            # A label column NOTHING can be read from is refused three branches
            # below, BY NAME ("no_populated_group_label_cell", with the cell
            # count), and that is the more specific answer, so it is left
            # there. This guard exists for the case those branches cannot see:
            # a label column that is mostly readable, where the survivors
            # produce a finite number that looks like a complete measurement.
            if n_with_label > 0:
                n_usable = n_with_label
        if n_usable < n_rows_read:
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "ConditionalIndependenceRegularizer.forward",
                measured=n_usable,
                total=n_rows_read,
                unit=(
                    "rows were finite in y_pred, the attribute and the label"
                    if conditioning is not None
                    else "rows were finite in both y_pred and the attribute"
                ),
                requirement="every row must be readable before it is placed in a stratum",
                reporting="dependence_measure=nan and measured=False",
                instead_of=(
                    "a gap over whichever rows happened to be readable, which here was "
                    "numerically indistinguishable from the healthy answer"
                ),
            )
            metrics = _not_measured_metrics(
                "non_finite_rows",
                extra={"n_rows": n_rows_read, "n_usable": n_usable},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        # THE GUARD SITS ABOVE THE y_true BRANCH.
        #
        # Conditional independence is a comparison BETWEEN groups within a
        # label stratum, so both branches below share the precondition that at
        # least two groups exist. Guarding only the conditional branch left the
        # same fabrication live on the unconditional fallback, which reaches
        # StatisticalParityRegularizer and its own single-group return.
        #
        # Measured before this fix, with one group and 200 rows:
        # {'regularization_value': 0.0, 'dependence_measure': 0.0,
        # 'group_penalties': {}, 'metadata': {}} and zero warnings, byte
        # identical for an empty batch, while healthy two-group data returns
        # 0.296 through that same dependence_measure field. 0.0 on that scale
        # means "predictions are perfectly conditionally independent of the
        # protected attribute" -- the clean end -- asserted where no comparison
        # existed to make.
        n_rows = int(y_pred.shape[0]) if y_pred.dim() > 0 else 1
        n_groups = int(torch.unique(sensitive_attr).numel())
        if n_rows == 0 or n_groups < 2:
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            reason = "no_rows" if n_rows == 0 else "single_group"
            warn_not_assessed(
                "ConditionalIndependenceRegularizer.forward",
                measured=n_groups,
                total=n_groups,
                unit=f"groups were present across {n_rows} row(s)",
                requirement="a conditional independence comparison needs at least 2",
                reporting="dependence_measure=nan and measured=False",
                instead_of=("0.0, which on this scale means proven conditional independence"),
            )
            metrics = _not_measured_metrics(
                reason,
                extra={"n_groups": n_groups, "n_rows": n_rows},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        if y_true is None:
            warnings.warn(
                "y_true not provided for conditional independence regularizer. "
                "Falling back to unconditional (statistical parity). The returned "
                "dependence_measure is a STATISTICAL PARITY gap, not a conditional "
                "independence one; see metadata['fallback']."
            )
            fallback = StatisticalParityRegularizer(
                strength=self.strength,
                reduction=self.reduction,
                track_metrics=False,
            )(y_pred, sensitive_attr, return_metrics=True)
            # StatisticalParityRegularizer cannot tell the caller which
            # regularizer produced the number it returns, so the substitution
            # is named here, on the object the caller actually receives.
            fallback_penalty, fallback_metrics = fallback  # type: ignore[misc]
            fallback_metrics.metadata["fallback"] = "statistical_parity_no_y_true"
            if self.track_metrics:
                self._history.append(fallback_metrics)
            if return_metrics:
                return fallback_penalty, fallback_metrics
            return fallback_penalty

        unique_groups = torch.unique(sensitive_attr)
        unique_labels = torch.unique(y_true)

        group_penalties = {}
        all_penalties = []
        excluded_strata: List[str] = []

        for label in unique_labels:
            label_mask = y_true == label

            if label_mask.sum() == 0:
                continue

            label_mean = y_pred[label_mask].mean()

            stratum_cells = []
            for group in unique_groups:
                group_label_mask = label_mask & (sensitive_attr == group)

                if group_label_mask.sum() == 0:
                    continue

                group_label_mean = y_pred[group_label_mask].mean()
                penalty = torch.abs(group_label_mean - label_mean)

                key = f"g{group.item()}_y{label.item()}"
                stratum_cells.append((key, penalty))

            # THE SAME `_comparable` RULE, PER LABEL STRATUM.
            #
            # Each cell is |E[y_pred | G=g, Y=y] - E[y_pred | Y=y]|, and the
            # stratum mean is taken over the SAME rows, so when exactly one
            # group occupies a stratum that group IS the stratum: the cell is
            # 0.0 by arithmetic vacuity, not by comparison. Measured before
            # this fix on a perfectly confounded input (group 0 holds every
            # y=0 row at y_pred 0.95, group 1 holds every y=1 row at 0.05,
            # the most extreme conditional dependence the data can carry):
            #
            #   dependence_measure 0.0, measured True, metadata {},
            #   group_penalties {'g0_y0': 0.0, 'g1_y1': 0.0}, warnings []
            #
            # 0.0 on this scale reads as proven conditional independence. The
            # vacuous cells also DILUTE a real finding when some strata are
            # comparable and others are not: mean over [0.4, 0.4, 0.0] is
            # 0.2667 where the measured strata alone give 0.4, and a diluted
            # penalty is the direction that reads as fairer.
            if len(stratum_cells) < 2:
                excluded_strata.extend(key for key, _value in stratum_cells)
                continue

            for key, value in stratum_cells:
                group_penalties[key] = value.item()
                all_penalties.append(value)

        if not all_penalties:
            # The SIBLING of the guard at the top of this method, reached when
            # two or more groups exist but nothing below it can be compared:
            # either no (group, label) cell has any rows at all (possible when
            # y_true holds only values that produce empty strata), or every
            # populated stratum holds a single group. Same fabrication, three
            # lines apart: a 0.0 here was also reported as a measured
            # conditional independence.
            total_penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            regularization = self.strength * total_penalty
            if excluded_strata:
                reason = "no_comparable_group_pair_in_any_stratum"
                unit = "label strata held two or more groups to compare"
                requirement = (
                    "conditional independence needs two groups inside the same label stratum"
                )
            else:
                reason = "no_populated_group_label_cell"
                unit = "(group, label) cells had any rows in them"
                requirement = "a conditional comparison needs at least one populated cell"
            warn_not_assessed(
                "ConditionalIndependenceRegularizer.forward",
                measured=0,
                total=int(len(unique_groups) * len(unique_labels)),
                unit=unit,
                requirement=requirement,
                reporting="dependence_measure=nan and measured=False",
                instead_of=("0.0, which on this scale means proven conditional independence"),
            )
            metrics = _not_measured_metrics(
                reason,
                extra={
                    "n_groups": int(len(unique_groups)),
                    "n_labels": int(len(unique_labels)),
                    "excluded_strata": list(excluded_strata),
                },
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return regularization, metrics
            return regularization

        penalties_tensor = torch.stack(all_penalties)
        if self.reduction == "mean":
            total_penalty = penalties_tensor.mean()
        elif self.reduction == "max":
            total_penalty = penalties_tensor.max()
        else:
            total_penalty = penalties_tensor.sum()

        regularization = self.strength * total_penalty

        metadata: Dict[str, Any] = {}
        if excluded_strata:
            # The survivors still give a usable gradient, so this is disclosed
            # rather than refused: refusing every batch that happens to contain
            # one single group stratum would neuter the regularizer, which is a
            # worse defect than the one being fixed.
            metadata["excluded_strata"] = list(excluded_strata)
            metadata["partial_coverage"] = True
            warnings.warn(
                f"ConditionalIndependenceRegularizer.forward: "
                f"{len(excluded_strata)} label stratum cell(s) {list(excluded_strata)} "
                f"held only one group, so no between group comparison existed there "
                f"and they were excluded. The reported dependence_measure covers the "
                f"comparable strata only: it is a PARTIAL measurement of the "
                f"conditional dependence, not the whole of it. See "
                f"metadata['excluded_strata'].",
                UserWarning,
                stacklevel=3,
            )

        if self.track_metrics or return_metrics:
            metrics = RegularizerMetrics(
                regularization_value=regularization.item(),
                dependence_measure=total_penalty.item(),
                group_penalties=group_penalties,
                metadata=metadata,
            )
            if self.track_metrics:
                self._history.append(metrics)

        if return_metrics:
            return regularization, metrics
        return regularization


class GroupFairnessRegularizer(BaseRegularizer):
    """
    General Group Fairness Regularizer.

    Provides a flexible framework for penalizing various group fairness
    violations through configurable fairness metrics.

    Supported metrics:
        - 'dp': Demographic parity (equal positive rates)
        - 'eo': Equalized odds (equal TPR and FPR)
        - 'eop': Equal opportunity (equal TPR only)
        - 'fpr': False positive rate parity
        - 'ppv': Positive predictive value parity

    Args:
        strength: Regularization strength
        fairness_metric: Which metric to enforce
        reduction: How to aggregate penalties
        track_metrics: Whether to track metrics

    Example:
        >>> regularizer = GroupFairnessRegularizer(
        ...     strength=0.1,
        ...     fairness_metric='eo'
        ... )
        >>> penalty = regularizer(y_pred, sensitive_attr, y_true)

    References:
        - Hardt et al. (2016): Equality of Opportunity
        - Zafar et al. (2017): Fairness Constraints

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger rows: fairness_regularization, group_fairness_regularizer. See
    docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    SUPPORTED_METRICS = ("dp", "eo", "eop", "fpr", "ppv")

    def __init__(
        self,
        strength: float = 0.1,
        fairness_metric: Literal["dp", "eo", "eop", "fpr", "ppv"] = "dp",
        reduction: Literal["mean", "max", "sum"] = "mean",
        track_metrics: bool = True,
    ):
        super().__init__(strength=strength, track_metrics=track_metrics)
        self.fairness_metric = fairness_metric
        self.reduction = reduction

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """Compute group fairness regularization."""
        check_torch_available()

        # AN UNKNOWN METRIC NAME IS AN ERROR, NOT A COULD NOT CHECK.
        #
        # The ValueError at the bottom of the dispatch is only reached by
        # inputs that get past the guards below it, so once those guards
        # started returning early a typo'd metric on degenerate data came back
        # as a polite refusal naming the typo as the metric
        # (metadata['fairness_metric'] = 'garbage', measured=False) instead of
        # raising. Nothing about the caller's data was wrong there; their code
        # was, and a refusal hides that.
        if self.fairness_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown fairness metric: {self.fairness_metric}. "
                f"Supported: {self.SUPPORTED_METRICS}."
            )

        # A ROW THAT CANNOT ENTER A RATE CANNOT BE FOLDED INTO ONE.
        #
        # Measured before this guard, with one NaN prediction in 100 rows and
        # two healthy groups, all with measured=True and zero warnings:
        #
        #   'dp'  -> dependence_measure nan, group_penalties {'g0.0': nan, ...}
        #   'eop' -> dependence_measure nan
        #   'fpr' -> dependence_measure 0.2000, a clean finite number whose
        #            value depends on which arm the unusable row landed in
        #
        # A NaN dependence_measure with measured=True contradicts this
        # module's own contract ("NaN and measured=False means NOT MEASURED"),
        # so a consumer filtering a history on `measured` still gets the NaN
        # and the aggregate it poisons. The finite one is worse: it reads as a
        # complete measurement of a batch that was not completely read.
        # THE LABEL IS THE THIRD AXIS, and it was missing from this guard.
        #
        # Four of the five metrics condition on y_true, so an unreadable label
        # is exactly "a row that cannot enter a rate". Measured after the two
        # checks above were in place, on 40 rows, two groups and one NaN in
        # y_true, all with measured=True and zero warnings:
        #
        #   'eo'  -> dependence_measure 0.30000001192092866, the healthy value,
        #            with the fpr arm silently shifted to 0.3158 / 0.2842
        #   'fpr' -> the same silent shift
        #   'ppv' -> nan with measured=True, because overall_ppv multiplies
        #            y_true straight into its numerator
        #
        # 'dp' does not read y_true, so it is not judged on it: passing the
        # label there would refuse a batch over a column the metric ignores.
        # With y_true=None the four label metrics fall back to demographic
        # parity and do not read it either.
        label_is_read = self.fairness_metric != "dp" and y_true is not None
        n_rows, n_usable = _readable_rows(
            y_pred,
            sensitive_attr,
            y_true if label_is_read else None,
        )
        if n_usable < n_rows:
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "GroupFairnessRegularizer.forward",
                measured=n_usable,
                total=n_rows,
                unit=(
                    f"rows were finite in y_pred, the attribute and the label "
                    f"({self.fairness_metric!r})"
                    if label_is_read
                    else f"rows were finite in both y_pred and the attribute "
                    f"({self.fairness_metric!r})"
                ),
                requirement="every row must be readable before a rate is computed",
                reporting="dependence_measure=nan and measured=False",
                instead_of="a rate computed over whichever rows happened to be readable",
            )
            metrics = _not_measured_metrics(
                "non_finite_rows",
                extra={
                    "fairness_metric": self.fairness_metric,
                    "n_rows": n_rows,
                    "n_usable": n_usable,
                },
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        # THE GUARD SITS ABOVE THE DISPATCH, not inside one branch.
        #
        # All five metrics below are BETWEEN-GROUP comparisons, so all five
        # share the precondition that at least two groups are present. Fixing
        # only the branch that happened to be filed would have left the
        # identical fabrication live in the other four, and the dp branch is
        # the WORST of them: with a single group, group_mean == overall_mean
        # exactly, so penalties is [0.0] rather than empty and
        # group_penalties comes back populated as {'g0': 0.0}. That reads as
        # "the one group I looked at agrees with the population", a measured
        # finding, and it is the case where the empty-dict tell the other
        # degenerate inputs leave does not exist.
        n_groups = int(torch.unique(sensitive_attr).numel())
        if n_groups < 2:
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "GroupFairnessRegularizer.forward",
                measured=n_groups,
                total=n_groups,
                unit=f"groups were present for the {self.fairness_metric!r} comparison",
                requirement="a between-group parity penalty needs at least 2",
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means every group agrees",
            )
            metrics = _not_measured_metrics(
                "fewer_than_two_groups",
                extra={
                    "fairness_metric": self.fairness_metric,
                    "n_groups": n_groups,
                    "n_rows": n_rows,
                },
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        if self.fairness_metric == "dp":
            return self._demographic_parity(y_pred, sensitive_attr, return_metrics)
        elif self.fairness_metric == "eo":
            return self._equalized_odds(y_pred, sensitive_attr, y_true, return_metrics)
        elif self.fairness_metric == "eop":
            return self._equal_opportunity(y_pred, sensitive_attr, y_true, return_metrics)
        elif self.fairness_metric == "fpr":
            return self._fpr_parity(y_pred, sensitive_attr, y_true, return_metrics)
        elif self.fairness_metric == "ppv":
            return self._ppv_parity(y_pred, sensitive_attr, y_true, return_metrics)
        else:
            raise ValueError(f"Unknown fairness metric: {self.fairness_metric}")

    @staticmethod
    def _comparable(arm):
        """Keep an arm's cells only if TWO OR MORE groups had a defined rate in it.

        The third instance of the same shape in this class, and the one that is
        easiest to miss because the arm is not empty. Each penalty here is
        ``|group_rate - overall_rate|``, and ``overall_rate`` is computed over
        the SAME conditioning set, so when exactly one group has a defined rate
        that group IS the population: the penalty is 0.0 by arithmetic vacuity,
        not by comparison. Measured before this fix on two groups where only g0
        had a positive label, ``'eop'`` reported ``dependence_measure=0.0``
        with ``group_penalties={'g0': 0.0}`` -- TPR parity established between
        one group and itself.

        Worse, that vacuous 0.0 was then AVERAGED IN. For ``'eo'`` with a
        measurable FPR arm and an unmeasurable TPR arm, the reduction ran over
        [0.0, fpr_0, fpr_1], so the fabricated cell pulled the reported
        disparity DOWN, in the direction that reads as fairer. A neutered
        measurement reports the better number, never the worse one.

        Returns (kept_cells, excluded_keys).
        """
        if len(arm) >= 2:
            return arm, []
        return [], [key for key, _value in arm]

    def _demographic_parity(self, y_pred, sensitive_attr, return_metrics, fallback=None):
        """Demographic parity penalty.

        ``fallback`` names the metric this DP number is standing in for, when
        it was reached from one of the label conditioned metrics with no
        y_true. It is recorded on the metrics object rather than left implicit;
        see :meth:`_demographic_parity_fallback`.
        """
        unique_groups = torch.unique(sensitive_attr)
        overall_mean = y_pred.mean()

        penalties = []
        group_penalties = {}

        for group in unique_groups:
            mask = sensitive_attr == group
            group_mean = y_pred[mask].mean()
            penalty = torch.abs(group_mean - overall_mean)
            penalties.append(penalty)
            group_penalties[f"g{group.item()}"] = penalty.item()

        return self._aggregate(
            penalties,
            group_penalties,
            return_metrics,
            y_pred.device,
            fallback=fallback,
        )

    def _demographic_parity_fallback(self, y_pred, sensitive_attr, return_metrics):
        """The ONE place a label conditioned metric substitutes DP for itself.

        ``'eo'``, ``'eop'``, ``'fpr'`` and ``'ppv'`` are all defined on the
        true label, so without y_true none of them can be computed and all four
        fell back to demographic parity. The number that came back was the DP
        gap, and it was labelled with the REQUESTED metric and nothing else.
        Measured before this change on 100 rows and two groups, with
        y_true=None, for every one of the four:

            dependence_measure 0.19999995827674866, measured True,
            metadata {'fairness_metric': 'eop'}, warnings []

        byte identical to the ``'dp'`` answer on the same input. A reader (or a
        run history, or a report) then has a column of equal opportunity gaps
        that are not equal opportunity gaps, with no way to tell.

        The substitution is kept rather than refused, because a DP penalty is a
        usable gradient and refusing would break an unlabelled training loop.
        What changes is that it says so: ``metadata['fallback']`` and
        ``metadata['measured_metric']`` name what was actually computed, and a
        UserWarning says it once at the call.
        """
        warnings.warn(
            f"GroupFairnessRegularizer.forward: fairness_metric="
            f"{self.fairness_metric!r} is defined on the true label and y_true "
            f"was not provided. Falling back to demographic parity. The returned "
            f"dependence_measure is a DEMOGRAPHIC PARITY gap, not a "
            f"{self.fairness_metric!r} one; see metadata['fallback'].",
            UserWarning,
            stacklevel=3,
        )
        return self._demographic_parity(
            y_pred,
            sensitive_attr,
            return_metrics,
            fallback=f"demographic_parity_no_y_true_for_{self.fairness_metric}",
        )

    def _equalized_odds(self, y_pred, sensitive_attr, y_true, return_metrics):
        """Equalized odds penalty (TPR + FPR parity)."""
        if y_true is None:
            return self._demographic_parity_fallback(y_pred, sensitive_attr, return_metrics)

        unique_groups = torch.unique(sensitive_attr)
        penalties = []
        group_penalties = {}

        positives = y_true == 1
        negatives = y_true == 0
        # NaN, not 0.5. The else branch here IS evaluated (raising from it
        # proves so: an input with no positive label anywhere reaches it). What
        # is true, and what the earlier "unreachable by construction" comment
        # got wrong, is that the value never reaches a PENALTY: if `positives`
        # is all-False then every `mask & positives` is all-False too, so no
        # TPR cell is ever compared against this stand-in and _aggregate
        # refuses on an empty penalty list. A reader who trusted the stronger
        # claim would delete the NaN as dead code. 0.5 was a fabricated rate
        # sitting one refactor away from a penalty, and it carried no device,
        # which would raise on CUDA the moment it did reach one.
        _unmeasured = torch.tensor(float("nan"), device=y_pred.device)
        overall_tpr = y_pred[positives].mean() if positives.sum() > 0 else _unmeasured
        overall_fpr = y_pred[negatives].mean() if negatives.sum() > 0 else _unmeasured

        excluded = []
        tpr_arm = []
        fpr_arm = []
        for group in unique_groups:
            mask = sensitive_attr == group

            group_pos = mask & positives
            if group_pos.sum() > 0:
                group_tpr = y_pred[group_pos].mean()
                tpr_arm.append((f"g{group.item()}_tpr", torch.abs(group_tpr - overall_tpr)))
            else:
                excluded.append(f"g{group.item()}_tpr")

            group_neg = mask & negatives
            if group_neg.sum() > 0:
                group_fpr = y_pred[group_neg].mean()
                fpr_arm.append((f"g{group.item()}_fpr", torch.abs(group_fpr - overall_fpr)))
            else:
                excluded.append(f"g{group.item()}_fpr")

        # Equalized odds has TWO arms, and each is its own parity comparison,
        # so each is kept or dropped on its own count of defined rates.
        for arm in (tpr_arm, fpr_arm):
            kept, arm_excluded = self._comparable(arm)
            excluded.extend(arm_excluded)
            for key, value in kept:
                penalties.append(value)
                group_penalties[key] = value.item()

        return self._aggregate(penalties, group_penalties, return_metrics, y_pred.device, excluded)

    def _equal_opportunity(self, y_pred, sensitive_attr, y_true, return_metrics):
        """Equal opportunity penalty (TPR parity only)."""
        if y_true is None:
            return self._demographic_parity_fallback(y_pred, sensitive_attr, return_metrics)

        unique_groups = torch.unique(sensitive_attr)
        penalties = []
        group_penalties = {}

        positives = y_true == 1
        # NaN, not 0.5: see the note in _equalized_odds. Evaluated when no row
        # carries a positive label, but never compared against a group cell,
        # and no longer a fabricated rate if it ever is.
        overall_tpr = (
            y_pred[positives].mean()
            if positives.sum() > 0
            else torch.tensor(float("nan"), device=y_pred.device)
        )

        excluded = []
        arm = []
        for group in unique_groups:
            mask = sensitive_attr == group
            group_pos = mask & positives

            if group_pos.sum() > 0:
                group_tpr = y_pred[group_pos].mean()
                arm.append((f"g{group.item()}", torch.abs(group_tpr - overall_tpr)))
            else:
                excluded.append(f"g{group.item()}")

        kept, arm_excluded = self._comparable(arm)
        excluded.extend(arm_excluded)
        for key, value in kept:
            penalties.append(value)
            group_penalties[key] = value.item()

        return self._aggregate(penalties, group_penalties, return_metrics, y_pred.device, excluded)

    def _fpr_parity(self, y_pred, sensitive_attr, y_true, return_metrics):
        """FPR parity penalty."""
        if y_true is None:
            return self._demographic_parity_fallback(y_pred, sensitive_attr, return_metrics)

        unique_groups = torch.unique(sensitive_attr)
        penalties = []
        group_penalties = {}

        negatives = y_true == 0
        # NaN, not 0.5: see the note in _equalized_odds. Evaluated when no row
        # carries a negative label, but never compared against a group cell.
        overall_fpr = (
            y_pred[negatives].mean()
            if negatives.sum() > 0
            else torch.tensor(float("nan"), device=y_pred.device)
        )

        excluded = []
        arm = []
        for group in unique_groups:
            mask = sensitive_attr == group
            group_neg = mask & negatives

            if group_neg.sum() > 0:
                group_fpr = y_pred[group_neg].mean()
                arm.append((f"g{group.item()}", torch.abs(group_fpr - overall_fpr)))
            else:
                excluded.append(f"g{group.item()}")

        kept, arm_excluded = self._comparable(arm)
        excluded.extend(arm_excluded)
        for key, value in kept:
            penalties.append(value)
            group_penalties[key] = value.item()

        return self._aggregate(penalties, group_penalties, return_metrics, y_pred.device, excluded)

    def _ppv_parity(self, y_pred, sensitive_attr, y_true, return_metrics):
        """PPV (precision) parity penalty."""
        if y_true is None:
            return self._demographic_parity_fallback(y_pred, sensitive_attr, return_metrics)

        unique_groups = torch.unique(sensitive_attr)
        penalties = []
        group_penalties = {}

        # Soft precision: sum(p * y) / sum(p)
        overall_ppv = (y_pred * y_true.float()).sum() / (y_pred.sum() + 1e-8)

        excluded = []
        arm = []
        for group in unique_groups:
            mask = sensitive_attr == group
            y_pred_g = y_pred[mask]
            y_true_g = y_true[mask].float()

            if y_pred_g.sum() > 0:
                group_ppv = (y_pred_g * y_true_g).sum() / (y_pred_g.sum() + 1e-8)
                arm.append((f"g{group.item()}", torch.abs(group_ppv - overall_ppv)))
            else:
                excluded.append(f"g{group.item()}")

        kept, arm_excluded = self._comparable(arm)
        excluded.extend(arm_excluded)
        for key, value in kept:
            penalties.append(value)
            group_penalties[key] = value.item()

        return self._aggregate(penalties, group_penalties, return_metrics, y_pred.device, excluded)

    def _aggregate(
        self,
        penalties,
        group_penalties,
        return_metrics,
        device,
        excluded_groups=None,
        fallback=None,
    ):
        """Aggregate penalties.

        The second half of the two-part guard. ``forward`` refuses above the
        metric dispatch when fewer than two groups exist; this refuses when two
        or more groups exist but the CONDITIONING SET is empty, so no per-group
        rate was defined and ``penalties`` is empty. Measured before this fix,
        with two groups and no positive label anywhere, ``'eop'`` returned
        ``{'regularization_value': 0.0, 'dependence_measure': 0.0,
        'group_penalties': {}, 'metadata': {}}`` and zero warnings -- a TPR
        parity of 0.0 over a TPR that does not exist for either group.

        ``excluded_groups`` names the groups that HAD no defined rate while
        others did. The penalty over the survivors is still a usable gradient,
        but it is a LOWER BOUND on the real disparity rather than the
        disparity, so it is disclosed in metadata instead of being left
        invisible. It is not refused: refusing every batch in which one group
        lacks a positive label would neuter the regularizer, which is a worse
        defect than the one being fixed.
        """
        if not penalties:
            total = torch.tensor(0.0, device=device, requires_grad=True)
            regularization = self.strength * total
            excluded = list(excluded_groups or [])
            warn_not_assessed(
                "GroupFairnessRegularizer.forward",
                measured=0,
                total=len(excluded),
                unit=f"cells had a defined {self.fairness_metric!r} rate to compare",
                requirement="a parity penalty needs two groups defined in the same arm",
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means every group agrees",
            )
            metrics = _not_measured_metrics(
                "no_comparable_group_pair",
                extra={
                    "fairness_metric": self.fairness_metric,
                    "excluded_groups": excluded,
                },
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return regularization, metrics
            return regularization

        penalties_tensor = torch.stack(penalties)
        if self.reduction == "mean":
            total = penalties_tensor.mean()
        elif self.reduction == "max":
            total = penalties_tensor.max()
        else:
            total = penalties_tensor.sum()

        regularization = self.strength * total

        metadata: Dict[str, Any] = {"fairness_metric": self.fairness_metric}
        if fallback is not None:
            # fairness_metric stays the REQUESTED name, because that is what the
            # caller asked for and what a filter will look for; these two say
            # what was actually computed under it.
            metadata["fallback"] = fallback
            metadata["measured_metric"] = "dp"
        if excluded_groups:
            metadata["excluded_groups"] = list(excluded_groups)
            metadata["partial_coverage"] = True
            warnings.warn(
                f"GroupFairnessRegularizer.forward: {len(excluded_groups)} group(s) "
                f"{list(excluded_groups)} had no defined {self.fairness_metric!r} rate "
                f"and were excluded. The reported dependence_measure is a LOWER BOUND "
                f"on the disparity across all groups, not the disparity. See "
                f"metadata['excluded_groups'].",
                UserWarning,
                stacklevel=3,
            )

        if self.track_metrics or return_metrics:
            metrics = RegularizerMetrics(
                regularization_value=regularization.item(),
                dependence_measure=total.item(),
                group_penalties=group_penalties,
                metadata=metadata,
            )
            if self.track_metrics:
                self._history.append(metrics)

        if return_metrics:
            return regularization, metrics
        return regularization


class HilbertSchmidtRegularizer(BaseRegularizer):
    """
    Hilbert-Schmidt Independence Criterion (HSIC) Regularizer.

    Uses HSIC to measure and penalize the statistical dependence between
    model predictions and sensitive attributes. HSIC is a kernel-based
    measure of dependence that is zero iff the variables are independent.

    Mathematical formulation:
        HSIC(X, Y) = tr(KXHKYH) / (n-1)²

        where K are kernel matrices and H is the centering matrix.

    Args:
        strength: Regularization strength
        kernel: Kernel type ('rbf', 'linear')
        sigma: RBF kernel bandwidth (if using RBF)
        track_metrics: Whether to track metrics

    Example:
        >>> regularizer = HilbertSchmidtRegularizer(
        ...     strength=0.1,
        ...     kernel='rbf',
        ...     sigma=1.0
        ... )
        >>> penalty = regularizer(y_pred, sensitive_attr)

    References:
        - Gretton et al. (2005): Measuring Statistical Dependence with HSIC
        - Song et al. (2012): Feature Selection via Dependence Maximization

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: hilbert_schmidt_regularizer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        strength: float = 0.1,
        kernel: Literal["rbf", "linear"] = "rbf",
        sigma: float = 1.0,
        track_metrics: bool = True,
    ):
        super().__init__(strength=strength, track_metrics=track_metrics)
        # A BANDWIDTH OF ZERO IS A CALLER ERROR, NOT AN UNMEASURABLE BATCH.
        #
        # The RBF kernel below divides by 2 * sigma ** 2, so sigma=0.0 makes
        # every kernel entry NaN and the statistic NaN. Measured before this
        # check on 40 readable rows with two groups, where sigma=1.0 reports
        # 0.018390340730547905:
        #
        #   sigma=0.0  -> dependence_measure nan with measured=True and zero
        #                 warnings, which is the one shape the
        #                 RegularizerMetrics contract at the top of this module
        #                 says must never be reported (a consumer filtering a
        #                 history on `measured` keeps the NaN)
        #   sigma=-1.0 -> 0.018390340730547905, BYTE IDENTICAL to sigma=1.0,
        #                 because only sigma ** 2 is read, so a nonsense
        #                 bandwidth was accepted and silently squared away
        #
        # Same decision, same reason, as ``sigma`` in
        # operations.monitoring.drift.detect_drift_mmd, which refuses a
        # non-positive or non-finite bandwidth at the call.
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                f"HilbertSchmidtRegularizer: sigma must be a finite positive kernel "
                f"bandwidth, got {sigma!r}. sigma=0 makes every RBF kernel entry NaN "
                f"and the HSIC NaN, and a negative sigma is read only through "
                f"sigma**2, so it silently behaves as its absolute value."
            )
        self.kernel = kernel
        self.sigma = sigma

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """Compute HSIC regularization."""
        check_torch_available()

        # A 0-dimensional tensor has no rows to count, so it is treated as one
        # sample and refused by the guard below. Before 2026-09-16 it raised
        # IndexError from `y_pred.shape[0]`; a refusal naming n_samples=1 says
        # the same thing without the caller having to read a traceback.
        n = int(y_pred.shape[0]) if y_pred.dim() > 0 else 1
        if n < 2:
            # The PENALTY stays a 0.0 grad-enabled tensor: a short final batch
            # (DataLoader with drop_last=False) must not break backprop, and a
            # zero contribution is the right optimisation behaviour when there
            # is nothing to push on.
            #
            # The MEASURE must not stay 0.0. HSIC over fewer than two samples
            # has no estimate at all -- the centering matrix H has no
            # off-diagonal structure and the (n-1)^2 denominator is 0 or
            # undefined -- while this class's own docstring says HSIC "is zero
            # iff the variables are independent". Returning
            # RegularizerMetrics(0.0, 0.0) therefore handed the caller the
            # cleanest possible finding on the scale, sourced from no data.
            # Measured before this fix: n=0 and n=1 both returned
            # {'regularization_value': 0.0, 'dependence_measure': 0.0,
            # 'group_penalties': {}, 'metadata': {}} with zero warnings, and
            # the only difference from a healthy n=100 result was an empty
            # metadata dict that nothing documented.
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "HilbertSchmidtRegularizer.forward",
                measured=0,
                total=n,
                unit="rows in the batch could enter an HSIC estimate",
                requirement="the statistic needs at least 2",
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means proven independence",
            )
            metrics = _not_measured_metrics(
                "n_samples_lt_2",
                extra={"kernel": self.kernel, "sigma": self.sigma, "n_samples": n},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        # THE OTHER THREE DEGENERACIES, ported from CorrelationPenalty in this
        # same module, which already refused all three.
        #
        # n >= 2 is necessary and not sufficient. HSIC is a dependence measure
        # between TWO varying quantities, and this class's own docstring says
        # it "is zero iff the variables are independent", so a 0.0 it returns
        # is read as proven independence. Measured on this file before this
        # change, all with measured=True, empty warnings and 100 rows:
        #
        #   sensitive attribute constant  -> 1.6e-14   (no groups to compare)
        #   predictions constant          -> 6.3e-14   (no output to explain)
        #   one NaN row in y_pred         -> nan       with measured=True,
        #       which directly contradicts the RegularizerMetrics contract
        #       above ("NaN and measured=False means NOT MEASURED") and so
        #       breaks any consumer that filters a history on `measured`.
        #
        # The first two are the module's documented single-group and
        # constant-prediction cases: the kernel of a constant vector is
        # constant, its centred form is the zero matrix, and the product is
        # exactly 0 for arithmetic reasons rather than from evidence about the
        # model. The spread test is on the RAW values (max against min per
        # column, the ptp form), never on an accumulated statistic, because the
        # centred sum of squares of a constant float vector is 4.3e-10 and not
        # 0.0.
        y_pred_flat = y_pred.view(-1, 1) if y_pred.dim() == 1 else y_pred
        sensitive_flat = sensitive_attr.view(-1, 1).float()

        sensitive_values = sensitive_flat.reshape(-1)
        finite_rows = torch.isfinite(y_pred_flat).all(dim=1) & torch.isfinite(sensitive_values)
        n_usable = int(finite_rows.sum().item())

        not_assessed: Optional[str] = None
        if n_usable < n:
            not_assessed = "non_finite_rows"
        elif bool((y_pred_flat.max(dim=0).values == y_pred_flat.min(dim=0).values).all().item()):
            not_assessed = "constant_predictions"
        elif not _has_spread(sensitive_values):
            not_assessed = "single_group"

        if not_assessed is not None:
            # The penalty stays a differentiable zero for the same reason as
            # the n < 2 branch above: training must survive a degenerate batch.
            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "HilbertSchmidtRegularizer.forward",
                measured=n_usable,
                total=n,
                unit=f"rows could enter an HSIC estimate ({not_assessed})",
                requirement=(
                    "the statistic needs finite rows and spread in BOTH the "
                    "predictions and the attribute"
                ),
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means proven independence",
            )
            metrics = _not_measured_metrics(
                not_assessed,
                extra={
                    "kernel": self.kernel,
                    "sigma": self.sigma,
                    "n_samples": n,
                    "n_usable": n_usable,
                },
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return penalty, metrics
            return penalty

        K_pred = self._compute_kernel(y_pred_flat)
        K_sens = self._compute_kernel(sensitive_flat)

        # Centering matrix H = I - (1/n) * 1*1^T
        H = torch.eye(n, device=y_pred.device) - torch.ones(n, n, device=y_pred.device) / n

        # HSIC = tr(K_pred @ H @ K_sens @ H) / (n-1)^2
        # Use matrix trace trick: tr(AB) = sum(A * B^T)
        HK_pred = H @ K_pred
        HK_sens = H @ K_sens

        hsic = (HK_pred * HK_sens.t()).sum() / ((n - 1) ** 2)

        regularization = self.strength * hsic

        if self.track_metrics or return_metrics:
            metrics = RegularizerMetrics(
                regularization_value=regularization.item(),
                dependence_measure=hsic.item(),
                metadata={"kernel": self.kernel, "sigma": self.sigma},
            )
            if self.track_metrics:
                self._history.append(metrics)

        if return_metrics:
            return regularization, metrics
        return regularization

    def _compute_kernel(self, X: "torch.Tensor") -> "torch.Tensor":
        """Compute kernel matrix."""
        if self.kernel == "linear":
            return X @ X.t()
        elif self.kernel == "rbf":
            # RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
            dist = torch.cdist(X, X, p=2) ** 2
            return torch.exp(-dist / (2 * self.sigma**2))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


class CorrelationPenalty(BaseRegularizer):
    """
    Simple Correlation-Based Penalty.

    Penalizes the Pearson correlation between predictions and sensitive
    attribute. This is a simple and interpretable regularizer.

    Mathematical formulation:
        R = |corr(ŷ, a)|

    Args:
        strength: Regularization strength
        squared: Whether to use squared correlation
        track_metrics: Whether to track metrics

    Example:
        >>> regularizer = CorrelationPenalty(strength=0.1)
        >>> penalty = regularizer(y_pred, sensitive_attr)

    Note:
        This only captures linear dependence. For nonlinear relationships,
        consider HilbertSchmidtRegularizer.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: correlation_penalty. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        strength: float = 0.1,
        squared: bool = False,
        track_metrics: bool = True,
    ):
        super().__init__(strength=strength, track_metrics=track_metrics)
        self.squared = squared

    def forward(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_true: Optional["torch.Tensor"] = None,
        return_metrics: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", RegularizerMetrics]]:
        """Compute correlation penalty."""
        check_torch_available()

        y_pred_flat = y_pred.view(-1)
        sensitive_flat = sensitive_attr.view(-1).float()

        # A Pearson correlation is UNDEFINED, not zero, when either vector has
        # no spread: its denominator is a product of standard deviations and
        # the ratio is 0/0. The `+ 1e-8` below turned that 0/0 into exactly
        # 0.0 -- the value that MEANS "predictions carry no information about
        # the sensitive attribute" -- and wrote it to dependence_measure and
        # into the tracked history with no warning and no marker. Measured
        # before this fix, all silently: 0 rows -> 0.0; a single group
        # (sensitive vector constant) -> 0.0; a constant prediction vector ->
        # -4.3e-10; every label the same -> 0.0. A healthy perfectly
        # correlated batch returns 1.0 through the same field.
        #
        # The test is on the RAW vectors. `(y_pred_centered ** 2).sum() == 0`
        # is the guard that looks right and is not: for torch.full((30,), 0.7)
        # that sum is 4.3e-10, so the constant-prediction case would walk
        # straight past it.
        n_rows = int(y_pred_flat.numel())
        usable = torch.isfinite(y_pred_flat) & torch.isfinite(sensitive_flat)
        n_usable = int(usable.sum().item())

        not_assessed: Optional[str] = None
        if n_usable < 2:
            not_assessed = "fewer_than_two_finite_rows"
        elif n_usable < n_rows:
            not_assessed = "non_finite_rows"
        elif not _has_spread(y_pred_flat):
            not_assessed = "constant_predictions"
        elif not _has_spread(sensitive_flat):
            not_assessed = "single_group"

        if not_assessed is not None:
            # The penalty stays a differentiable zero so training is
            # unaffected: there is genuinely nothing to push on.
            regularization = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            warn_not_assessed(
                "CorrelationPenalty.forward",
                measured=n_usable,
                total=n_rows,
                unit=f"rows could enter a Pearson correlation ({not_assessed})",
                requirement="it needs at least 2 rows and spread in BOTH vectors",
                reporting="dependence_measure=nan and measured=False",
                instead_of="0.0, which on this scale means proven independence",
            )
            metrics = _not_measured_metrics(
                not_assessed,
                extra={"squared": self.squared, "n_rows": n_rows, "n_usable": n_usable},
            )
            if self.track_metrics:
                self._history.append(metrics)
            if return_metrics:
                return regularization, metrics
            return regularization

        y_pred_centered = y_pred_flat - y_pred_flat.mean()
        sensitive_centered = sensitive_flat - sensitive_flat.mean()

        numerator = (y_pred_centered * sensitive_centered).sum()
        denominator = torch.sqrt((y_pred_centered**2).sum() * (sensitive_centered**2).sum() + 1e-8)
        correlation = numerator / denominator

        if self.squared:
            penalty = correlation**2
        else:
            penalty = torch.abs(correlation)

        regularization = self.strength * penalty

        if self.track_metrics or return_metrics:
            metrics = RegularizerMetrics(
                regularization_value=regularization.item(),
                dependence_measure=correlation.item(),
                metadata={"squared": self.squared},
            )
            if self.track_metrics:
                self._history.append(metrics)

        if return_metrics:
            return regularization, metrics
        return regularization


def create_regularizer(
    regularizer_type: Union[str, RegularizerType],
    strength: float = 0.1,
    **kwargs,
) -> BaseRegularizer:
    """
    Factory function to create fairness regularizers.

    Args:
        regularizer_type: Type of regularizer
        strength: Regularization strength
        **kwargs: Additional regularizer-specific arguments

    Returns:
        Configured regularizer

    Example:
        >>> regularizer = create_regularizer('statistical_parity', strength=0.1)
    """
    if isinstance(regularizer_type, str):
        regularizer_type = RegularizerType(regularizer_type)

    regularizer_classes = {
        RegularizerType.STATISTICAL_PARITY: StatisticalParityRegularizer,
        RegularizerType.CONDITIONAL_INDEPENDENCE: ConditionalIndependenceRegularizer,
        RegularizerType.GROUP_FAIRNESS: GroupFairnessRegularizer,
        RegularizerType.HSIC: HilbertSchmidtRegularizer,
        RegularizerType.CORRELATION: CorrelationPenalty,
    }

    # ADVERTISED BUT NOT IMPLEMENTED IS ITS OWN ANSWER (2026-09-25).
    #
    # RegularizerType.MUTUAL_INFORMATION is a member of the PUBLIC enum with no
    # entry in the table above, so ``create_regularizer(RegularizerType.
    # MUTUAL_INFORMATION)`` raised "Unknown regularizer type". A caller who picked
    # the member out of the library's own enum reads that as "you mistyped it" and
    # goes looking for the typo, when the truth is that this release does not
    # implement it. Say which of the two it is; the enum member is deliberately
    # left in place so existing imports keep working and the gap stays visible
    # rather than being quietly deleted.
    if regularizer_type is RegularizerType.MUTUAL_INFORMATION:
        raise NotImplementedError(
            "RegularizerType.MUTUAL_INFORMATION is declared on the public enum but "
            "is NOT IMPLEMENTED in this release: there is no differentiable mutual "
            "information estimator behind it, and none is being substituted. Use "
            "RegularizerType.HSIC for a kernel dependence penalty over the same "
            f"relationship, or one of {[k.value for k in regularizer_classes]}."
        )

    if regularizer_type not in regularizer_classes:
        raise ValueError(f"Unknown regularizer type: {regularizer_type}")

    return regularizer_classes[regularizer_type](strength=strength, **kwargs)
