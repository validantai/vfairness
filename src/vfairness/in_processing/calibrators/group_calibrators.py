"""
Group-Specific Calibrators for Training-Time Calibration.

This module provides calibration methods that can be integrated into
the training process to ensure group-specific probability calibration.
These calibrators learn to correct miscalibration patterns across
demographic groups during model training.

Calibrators Implemented:
    1. TrainableGroupCalibrator: PyTorch module for differentiable calibration
    2. TemperatureScalingCalibrator: Group-specific temperature scaling
    3. PlattScalingCalibrator: Group-specific Platt scaling (logistic)
    4. BetaCalibrator: Group-specific beta calibration
    5. FocalCalibrator: Group-specific focal calibration

Key Concept:
    Unlike post-processing calibrators that are fitted after training,
    these calibrators are integrated into the training loop and can
    be optimized jointly with the main model.

References:
    - Platt (2000): Probabilities for SV Machines
    - Guo et al. (2017): On Calibration of Modern Neural Networks
    - Kull et al. (2019): Beyond Temperature Scaling
    - Pleiss et al. (2017): On Fairness and Calibration
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Union

import numpy as np

from vfairness._triage import is_measured

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401  # availability probe

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class nn:  # type: ignore[no-redef]  # noqa: N801  # torch-absence stub
        class Module:
            pass


# Base class for the calibrator nn.Modules. At runtime this is torch's
# nn.Module when torch is installed and plain ``object`` otherwise, so the
# calibrators degrade gracefully. For type-checking we bind it to torch's
# nn.Module: torch ships no stubs here (ignore_missing_imports), so this is
# effectively ``Any`` and gives the subclasses the dynamic nn.Module surface
# (register_buffer, __call__, Parameter handling) that they use at runtime.
# A concrete base also stops mypy rejecting the conditional expression as an
# "Invalid base class".
if TYPE_CHECKING:
    from typing import TypeAlias

    import torch.nn as _torch_nn

    _CalibratorBase: TypeAlias = _torch_nn.Module
else:
    _CalibratorBase = nn.Module if TORCH_AVAILABLE else object


def check_torch_available():
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for trainable calibrators. Install with: pip install torch"
        )


def _finite_rows(values: "torch.Tensor") -> "torch.Tensor":
    """Per-ROW finiteness as a flat 1-D mask, whatever the trailing shape.

    A calibration row is usable only if every number in it is finite, so the
    reduction is ``all`` over the trailing dimensions rather than ``any``:
    (n, 1) logits from ``nn.Linear(k, 1)`` and (n,) logits both come back as
    (n,), which is what makes the mask safe to combine with a group mask
    instead of broadcasting against it.

    The row count is taken from ``shape[0]`` and the reduction is skipped when
    there is nothing to reduce. ``reshape(n, -1)`` is REFUSED by torch on a
    tensor of zero elements, because -1 is then ambiguous, and this helper is
    called on the caller's raw logits. Measured 2026-09-27 (BGL4 audit), an
    empty batch through ``TrainableGroupCalibrator.calibration_loss``:

        calibration_loss(torch.zeros(0, 1), torch.zeros(0), torch.zeros(0))
          before -> RuntimeError: cannot reshape tensor of 0 elements into
                    shape [0, -1] because the unspecified dimension size -1
                    can be any value and is ambiguous, raised from the reshape
                    below
          after  -> tensor(nan) + "no group reached min_group_size=10"

    The same call with FLAT (0,) logits returned that NaN before the fix too,
    so the crash was reachable only through the trailing dimension, and the
    docstring of calibration_loss names "an empty batch" as a NaN case. It
    propagated to CalibrationAwareTrainer.train_step and
    .fine_tune_calibration, which raised the same RuntimeError.
    """
    ok = torch.isfinite(values)
    if ok.dim() > 1:
        n_rows = int(ok.shape[0])
        if ok.numel() == 0:
            # No trailing element to reduce. `all` over an empty dimension is
            # True by definition, which is also the right answer here: a row
            # holding no number carries no NON-finite number either, and when
            # n_rows is 0 the mask is simply empty.
            ok = torch.ones(n_rows, dtype=torch.bool, device=ok.device)
        else:
            ok = ok.reshape(n_rows, -1).all(dim=1)
    return ok.reshape(-1)


def _warn_unknown_group_ids(group_ids: "torch.Tensor", n_groups: int, name: str) -> None:
    """Warn when samples carry group ids outside range(n_groups).

    Such samples have no learned calibration parameters. All calibrators
    pass them through UNCALIBRATED (instead of the old behaviour of silently
    zeroing their logits, which destroyed the model's output for those
    samples). The warning makes the mismatch visible so callers can fix
    their group encoding or increase n_groups.
    """
    unknown = (group_ids < 0) | (group_ids >= n_groups)
    if bool(unknown.any()):
        warnings.warn(
            f"{name}: {int(unknown.sum())} sample(s) have group ids outside "
            f"range({n_groups}); they are passed through uncalibrated. "
            "Check the group encoding or increase n_groups."
        )


class CalibrationMethodType(Enum):
    """Types of calibration methods."""

    TEMPERATURE = "temperature"
    PLATT = "platt"
    BETA = "beta"
    FOCAL = "focal"
    HISTOGRAM = "histogram"


@dataclass
class CalibrationState:
    """
    State of calibration parameters.

    Attributes:
        method: Calibration method type
        parameters: Method-specific parameters per group
        group_ece: ECE per group (pre-calibration)
        group_ece_post: ECE per group (post-calibration)
        global_ece: Global ECE, or NaN when it was never computed.
            No caller populates it today.
    """

    method: str
    # Per-group parameters. Scalar-parameter methods (temperature, focal)
    # store {group: value}; multi-parameter methods (platt, beta) store
    # {group: {param_name: value}}. Mapping (covariant in its value type) lets
    # either concrete dict shape be assigned; the field is read-only in use.
    parameters: Mapping[str, Union[float, Dict[str, float]]]
    group_ece: Dict[str, float] = field(default_factory=dict)
    group_ece_post: Dict[str, float] = field(default_factory=dict)
    # NaN, not 0.0. 0.0 ECE is PERFECT calibration, and get_calibration_state
    # (the only construction site in the codebase) passes only `method` and
    # `parameters`, so every state object ever produced serialised
    # "global_ece": 0.0 through to_dict. That is a perfect calibration score
    # attached to a measurement nobody took. The empty dicts above are honest by
    # contrast: no entries claims nothing.
    global_ece: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "parameters": self.parameters,
            "group_ece": self.group_ece,
            "group_ece_post": self.group_ece_post,
            "global_ece": self.global_ece,
        }


class TemperatureScalingCalibrator(_CalibratorBase):
    """
    Group-Specific Temperature Scaling Calibrator.

    Temperature scaling divides logits by a learned temperature parameter T:
        calibrated_prob = sigmoid(logit / T)

    This calibrator learns separate temperature parameters for each group,
    enabling group-specific calibration during training.

    Args:
        n_groups: Number of demographic groups
        init_temperature: Initial temperature value
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        learnable: Whether temperature is learnable (trainable)

    Example:
        >>> calibrator = TemperatureScalingCalibrator(n_groups=2)
        >>> logits = model(x)
        >>> calibrated_logits = calibrator(logits, group_ids)
        >>> calibrated_probs = torch.sigmoid(calibrated_logits)

    References:
        - Guo et al. (2017): On Calibration of Modern Neural Networks

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: temperature_scaling_calibrator. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_groups: int = 2,
        init_temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_temperature: float = 10.0,
        learnable: bool = True,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        check_torch_available()

        self.n_groups = n_groups
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        # Initialize temperature parameters (one per group)
        if learnable:
            # Use log-temperature for unconstrained optimization
            init_log_temp = np.log(init_temperature)
            self.log_temperatures = nn.Parameter(torch.full((n_groups,), init_log_temp))
        else:
            self.register_buffer(
                "log_temperatures", torch.full((n_groups,), np.log(init_temperature))
            )

    @property
    def temperatures(self) -> "torch.Tensor":
        """Get actual temperature values."""
        temps = torch.exp(self.log_temperatures)
        return torch.clamp(temps, self.min_temperature, self.max_temperature)

    def forward(
        self,
        logits: "torch.Tensor",
        group_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply group-specific temperature scaling.

        Args:
            logits: Uncalibrated logits
            group_ids: Group identifier for each sample

        Returns:
            Calibrated logits
        """
        check_torch_available()

        temperatures = self.temperatures

        # Start from a pass-through copy so samples with unknown group ids
        # keep their original logits instead of being zeroed.
        _warn_unknown_group_ids(group_ids, self.n_groups, self.__class__.__name__)
        calibrated_logits = logits.clone()

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.sum() > 0:
                calibrated_logits[mask] = logits[mask] / temperatures[g]

        return calibrated_logits

    def get_parameters(self) -> Dict[str, float]:
        """Get temperature parameters per group."""
        temps = self.temperatures.detach().cpu().numpy()
        return {f"group_{i}": float(temps[i]) for i in range(self.n_groups)}


class PlattScalingCalibrator(_CalibratorBase):
    """
    Group-Specific Platt Scaling Calibrator.

    Platt scaling applies a logistic transformation to logits:
        calibrated_prob = sigmoid(a * logit + b)

    This calibrator learns separate (a, b) parameters for each group.

    Args:
        n_groups: Number of demographic groups
        init_a: Initial 'a' parameter value
        init_b: Initial 'b' parameter value
        learnable: Whether parameters are learnable

    Example:
        >>> calibrator = PlattScalingCalibrator(n_groups=2)
        >>> logits = model(x)
        >>> calibrated_logits = calibrator(logits, group_ids)

    References:
        - Platt (2000): Probabilistic Outputs for SVMs

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: platt_scaling_calibrator. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_groups: int = 2,
        init_a: float = 1.0,
        init_b: float = 0.0,
        learnable: bool = True,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        check_torch_available()

        self.n_groups = n_groups

        if learnable:
            self.a = nn.Parameter(torch.full((n_groups,), init_a))
            self.b = nn.Parameter(torch.full((n_groups,), init_b))
        else:
            self.register_buffer("a", torch.full((n_groups,), init_a))
            self.register_buffer("b", torch.full((n_groups,), init_b))

    def forward(
        self,
        logits: "torch.Tensor",
        group_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply group-specific Platt scaling.

        Args:
            logits: Uncalibrated logits
            group_ids: Group identifier for each sample

        Returns:
            Calibrated logits
        """
        check_torch_available()

        # Pass-through copy: unknown group ids stay uncalibrated, not zeroed.
        _warn_unknown_group_ids(group_ids, self.n_groups, self.__class__.__name__)
        calibrated_logits = logits.clone()

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.sum() > 0:
                calibrated_logits[mask] = self.a[g] * logits[mask] + self.b[g]

        return calibrated_logits

    def get_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get Platt parameters per group."""
        a_vals = self.a.detach().cpu().numpy()
        b_vals = self.b.detach().cpu().numpy()
        return {
            f"group_{i}": {"a": float(a_vals[i]), "b": float(b_vals[i])}
            for i in range(self.n_groups)
        }


class BetaCalibrator(_CalibratorBase):
    """
    Group-Specific Beta Calibration.

    Beta calibration (Kull et al. 2017) fits a logistic model on the
    log-transformed probability, giving a three-parameter family derived
    from the beta distribution likelihood ratio:
        calibrated_prob = sigmoid(c * ln(p) - e * ln(1 - p) + d)

    where p = sigmoid(logit). This forward returns the calibrated LOGIT
        z = c * ln(p) - e * ln(1 - p) + d
    so it composes with the other calibrators. With c = e = 1 and d = 0
    the transform is the identity (ln(p) - ln(1-p) is the logit), which is
    the initialisation. Unlike Platt scaling (a * logit + b) the two slope
    parameters c and e let the map bend asymmetrically near 0 and 1.

    Args:
        n_groups: Number of demographic groups
        init_c: Initial 'c' parameter
        init_d: Initial 'd' parameter
        init_e: Initial 'e' parameter
        learnable: Whether parameters are learnable

    Example:
        >>> calibrator = BetaCalibrator(n_groups=2)
        >>> logits = model(x)
        >>> calibrated_logits = calibrator(logits, group_ids)

    References:
        - Kull et al. (2017): Beta Calibration

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: in_processing_beta_calibrator. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_groups: int = 2,
        init_c: float = 1.0,
        init_d: float = 0.0,
        init_e: float = 1.0,
        learnable: bool = True,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        check_torch_available()

        self.n_groups = n_groups

        if learnable:
            self.c = nn.Parameter(torch.full((n_groups,), init_c))
            self.d = nn.Parameter(torch.full((n_groups,), init_d))
            self.e = nn.Parameter(torch.full((n_groups,), init_e))
        else:
            self.register_buffer("c", torch.full((n_groups,), init_c))
            self.register_buffer("d", torch.full((n_groups,), init_d))
            self.register_buffer("e", torch.full((n_groups,), init_e))

    def forward(
        self,
        logits: "torch.Tensor",
        group_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply group-specific beta calibration.

        Args:
            logits: Uncalibrated logits
            group_ids: Group identifier for each sample

        Returns:
            Calibrated logits
        """
        check_torch_available()

        # Pass-through copy: unknown group ids stay uncalibrated, not zeroed.
        _warn_unknown_group_ids(group_ids, self.n_groups, self.__class__.__name__)
        calibrated_logits = logits.clone()

        # Kull et al. (2017) beta calibration on p = sigmoid(logit):
        #   z = c * ln(p) - e * ln(1 - p) + d
        # Clamp p away from {0, 1} so the logs stay finite.
        eps = 1e-7
        probs = torch.sigmoid(logits).clamp(eps, 1 - eps)
        log_p = torch.log(probs)
        log_1mp = torch.log(1 - probs)

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.sum() > 0:
                calibrated_logits[mask] = (
                    self.c[g] * log_p[mask] - self.e[g] * log_1mp[mask] + self.d[g]
                )

        return calibrated_logits

    def get_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get beta calibration parameters per group."""
        c_vals = self.c.detach().cpu().numpy()
        d_vals = self.d.detach().cpu().numpy()
        e_vals = self.e.detach().cpu().numpy()
        return {
            f"group_{i}": {"c": float(c_vals[i]), "d": float(d_vals[i]), "e": float(e_vals[i])}
            for i in range(self.n_groups)
        }


class FocalCalibrator(_CalibratorBase):
    """
    Group-Specific Focal Calibration.

    Focal calibration applies different focusing parameters per group
    to help calibrate predictions, especially near decision boundaries.

    Args:
        n_groups: Number of demographic groups
        init_gamma: Initial focal gamma parameter
        learnable: Whether gamma is learnable

    Example:
        >>> calibrator = FocalCalibrator(n_groups=2)
        >>> probs = torch.sigmoid(model(x))
        >>> calibrated_probs = calibrator(probs, group_ids)

    References:
        - Lin et al. (2017): Focal Loss for Dense Object Detection
        - Mukhoti et al. (2020): Calibrating Deep Neural Networks Using Focal Loss

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: focal_calibrator. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_groups: int = 2,
        init_gamma: float = 0.0,
        learnable: bool = True,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        check_torch_available()

        self.n_groups = n_groups

        if learnable:
            self.gamma = nn.Parameter(torch.full((n_groups,), init_gamma))
        else:
            self.register_buffer("gamma", torch.full((n_groups,), init_gamma))

    def forward(
        self,
        probs: "torch.Tensor",
        group_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply group-specific focal calibration.

        Args:
            probs: Uncalibrated probabilities
            group_ids: Group identifier for each sample

        Returns:
            Calibrated probabilities
        """
        check_torch_available()

        # Pass-through copy: unknown group ids stay uncalibrated, not zeroed.
        _warn_unknown_group_ids(group_ids, self.n_groups, self.__class__.__name__)
        calibrated_probs = probs.clone()

        # gamma is an unconstrained learnable parameter, so gradient descent can
        # and does drive it past -1: measured 2026-09-17, plain SGD on all-ones
        # labels reached gamma = -1.11 in 20 steps. The exponent 1 + gamma is
        # then <= 0, torch.pow returns >= 1 for every input, and the clamp below
        # flattens the whole group to 1.0 - 1e-8. Every sample in that group
        # leaves as the same near-certain positive, carrying nothing from the
        # model, and a downstream parity check reads that as perfect agreement.
        # Nothing said so. Name the groups where it happens.
        degenerate = [
            g
            for g in range(self.n_groups)
            if bool((group_ids == g).any()) and float(self.gamma[g].detach()) <= -1.0
        ]
        if degenerate:
            gammas = {g: round(float(self.gamma[g].detach()), 4) for g in degenerate}
            warnings.warn(
                f"{self.__class__.__name__}: group(s) {degenerate} have gamma <= -1 "
                f"({gammas}), so the exponent 1 + gamma is <= 0 and every probability "
                "in them saturates to 1.0. Those outputs are NOT calibrated "
                "probabilities and carry no information from the model; constrain "
                "gamma or reduce the learning rate."
            )

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.sum() > 0:
                p = probs[mask]
                gamma = self.gamma[g]
                # Apply focal-style recalibration
                # p_cal = p^(1 + gamma) for positive class
                calibrated_probs[mask] = torch.pow(p, 1 + gamma)

        # Renormalize to ensure valid probabilities
        calibrated_probs = torch.clamp(calibrated_probs, 1e-8, 1 - 1e-8)

        return calibrated_probs

    def get_parameters(self) -> Dict[str, float]:
        """Get focal parameters per group."""
        gamma_vals = self.gamma.detach().cpu().numpy()
        return {f"group_{i}": float(gamma_vals[i]) for i in range(self.n_groups)}


class TrainableGroupCalibrator(_CalibratorBase):
    """
    Unified Trainable Group Calibrator.

    This is a comprehensive calibrator that can use different calibration
    methods and be trained jointly with a model or used for fine-tuning.

    Supports:
        - Temperature scaling
        - Platt scaling
        - Beta calibration
        - Focal calibration

    Args:
        n_groups: Number of demographic groups
        method: Calibration method to use
        learnable: Whether calibration parameters are learnable
        calibration_loss_weight: Weight for calibration loss in training. Must be
            a finite POSITIVE number, because it multiplies the value
            :meth:`calibration_loss` returns: at 0 that value is exactly 0.0,
            which on this scale is perfect calibration, and below 0 the objective
            is inverted so training maximises miscalibration. To train without a
            calibration term pass ``include_calibration=False`` to
            :meth:`CalibrationAwareTrainer.train_step`, which reports NaN and says
            so, rather than weighting the measurement to nothing.

    Example:
        >>> calibrator = TrainableGroupCalibrator(
        ...     n_groups=2,
        ...     method='temperature'
        ... )
        >>>
        >>> # Training loop
        >>> logits = model(x)
        >>> calibrated_logits = calibrator(logits, group_ids)
        >>> task_loss = F.cross_entropy(calibrated_logits, y)
        >>> cal_loss = calibrator.calibration_loss(calibrated_logits, y, group_ids)
        >>> total_loss = task_loss + cal_loss

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: trainable_group_calibrator. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        n_groups: int = 2,
        method: Union[str, CalibrationMethodType] = "temperature",
        learnable: bool = True,
        calibration_loss_weight: float = 0.1,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        check_torch_available()

        self.n_groups = n_groups
        self.method = CalibrationMethodType(method) if isinstance(method, str) else method
        # A REGULARISER WHOSE LAMBDA IS ZERO IS A MEASUREMENT SWITCHED OFF
        # (F9 wave 4, 2026-09-30). `calibration_loss` ends with
        # `self.calibration_loss_weight * total_loss / len(measured)`, and this
        # value was stored with no check of any kind. The guard is HERE rather
        # than at the multiplication because the weight is a property of the
        # object, so one check above both callers of calibration_loss
        # (CalibrationAwareTrainer.train_step and fine_tune_calibration) closes
        # it for each of them; a check inside calibration_loss would have to be
        # repeated and could be reached with the object already misconfigured.
        #
        # Measured on 60 finite logits in two groups of 30, seed 0, whose real
        # calibration loss is a healthy 1.9988 at the default weight of 0.1:
        #
        #   calibration_loss_weight=0.0  -> exactly 0.0 with ZERO warnings
        #   calibration_loss_weight=-1.0 -> -19.9878 with ZERO warnings
        #   calibration_loss_weight=nan  -> nan with ZERO warnings
        #
        # 0.0 on this scale is PERFECT calibration by this method's own
        # docstring, so a switched-off regulariser reported the best score
        # attainable for a batch that needed correcting. train_step already
        # recorded exactly this reasoning for its own `include_calibration=False`
        # path ("a training log could not tell a calibration term that was
        # switched off from one that had nothing left to correct"), and
        # include_calibration is the documented way to switch the term off.
        # A NEGATIVE weight is worse than off: it INVERTS the objective, so
        # gradient descent maximises miscalibration, and the negative number it
        # reports reads as better than perfect on a scale whose floor is 0.
        # A NaN weight FORGES this method's documented refusal sentinel: NaN is
        # what calibration_loss returns when no group could be measured, and
        # train_step drops the term on seeing it, so a caller could not tell a
        # calibrator that refused from one whose weight nobody set.
        if not is_measured(calibration_loss_weight):
            raise ValueError(
                f"calibration_loss_weight={calibration_loss_weight!r} is not a finite "
                f"number. NaN is this calibrator's REFUSAL sentinel, returned when no "
                f"group could be measured, so a non-finite weight forges a refusal that "
                f"never happened and CalibrationAwareTrainer.train_step silently drops "
                f"the calibration term on seeing it. Pass a positive weight."
            )
        if calibration_loss_weight <= 0:
            raise ValueError(
                f"calibration_loss_weight={calibration_loss_weight!r} is not positive, so "
                f"calibration_loss cannot report a calibration measurement: at 0 it "
                f"returns exactly 0.0, which on this scale is PERFECT calibration, and "
                f"below 0 it inverts the objective so training maximises miscalibration "
                f"while reporting a number below the scale's floor. To train without a "
                f"calibration term pass include_calibration=False to "
                f"CalibrationAwareTrainer.train_step, which reports NaN and says so, "
                f"rather than weighting the measurement to nothing."
            )
        self.calibration_loss_weight = calibration_loss_weight

        # The attribute is any of the four calibrators, not the first branch's
        # concrete type (mypy infers that) and not the torch base (whose
        # __getattr__ hides get_parameters when torch is typed).
        self.calibrator: Union[
            TemperatureScalingCalibrator, PlattScalingCalibrator, BetaCalibrator, FocalCalibrator
        ]
        if self.method == CalibrationMethodType.TEMPERATURE:
            self.calibrator = TemperatureScalingCalibrator(n_groups=n_groups, learnable=learnable)
        elif self.method == CalibrationMethodType.PLATT:
            self.calibrator = PlattScalingCalibrator(n_groups=n_groups, learnable=learnable)
        elif self.method == CalibrationMethodType.BETA:
            self.calibrator = BetaCalibrator(n_groups=n_groups, learnable=learnable)
        elif self.method == CalibrationMethodType.FOCAL:
            self.calibrator = FocalCalibrator(n_groups=n_groups, learnable=learnable)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self._history: List[CalibrationState] = []

    def forward(
        self,
        logits: "torch.Tensor",
        group_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply calibration.

        Args:
            logits: Model logits or probabilities (depending on method)
            group_ids: Group identifier for each sample

        Returns:
            Calibrated outputs
        """
        return self.calibrator(logits, group_ids)

    def calibration_loss(
        self,
        logits: "torch.Tensor",
        y_true: "torch.Tensor",
        group_ids: "torch.Tensor",
        n_bins: int = 10,
        min_group_size: int = 10,
    ) -> "torch.Tensor":
        """
        Compute differentiable calibration loss.

        Uses expected calibration error (ECE) approximation that is
        differentiable for training.

        Args:
            logits: Model logits
            y_true: True labels
            group_ids: Group identifiers
            n_bins: Number of bins for ECE computation. Must be >= 1: with no
                bins the sum that IS the expected calibration error is never
                entered, so the method used to return exactly 0.0, which on this
                scale is perfect calibration, for a batch it never measured.
                Refused rather than returned.
            min_group_size: Smallest group that can carry a calibration
                estimate. Groups below it are excluded from the average and
                named in a warning. They are NOT folded in as zero
                calibration error, which is what silently happened before.

        Returns:
            Differentiable calibration loss, averaged over the groups that
            were actually measured, or NaN when no group in the batch reached
            ``min_group_size``.

            NaN is the refusal. On this scale 0.0 is PERFECT calibration, and
            the loop below used to return exactly 0.0 for a batch in which the
            statistic was never computed for a single group: an empty batch, a
            batch smaller than ``min_group_size``, or group ids that match no
            group. A caller adding that to a task loss, logging it, or reading
            it back out of :meth:`CalibrationAwareTrainer.train_step` could not
            tell it from a model that needed no calibration at all.

            A row whose logit or label is non-finite carries no calibration
            estimate either and is excluded from its group, counted in a
            warning. It is not averaged in as zero error; see the comment on
            the ``usable`` mask below for what that measured before and after.
        """
        check_torch_available()

        # A BIN COUNT BELOW 1 MEANS THE SUM THAT IS THE ECE IS NEVER ENTERED
        # (F9 wave 4, 2026-09-30). The loop below is `for i in range(n_bins)`
        # over boundaries from `torch.linspace(0, 1, n_bins + 1)`. With n_bins=0
        # that range is empty, so `ece` stays at its seed of exactly 0.0, the
        # group is still appended to `measured`, and the method returns
        # `weight * 0.0 / len(measured)`. Measured on 60 finite logits in two
        # groups of 30, seed 0, whose real loss at the default n_bins is 1.9988:
        #
        #   n_bins=10 -> 1.9988  (healthy control)
        #   n_bins=1  -> 1.2635  (coarse but real: one bin still compares the
        #                         mean confidence against the mean accuracy)
        #   n_bins=0  -> exactly 0.0 with ZERO warnings
        #   n_bins=-5 -> RuntimeError "number of steps must be non-negative"
        #                out of torch.linspace, naming neither this argument nor
        #                this class, i.e. not even a refusal
        #
        # 0.0 on this scale is PERFECT calibration by this method's own
        # docstring, so one public documented argument turned the whole
        # measurement into the best score attainable. The all-NaN batch still
        # refused at n_bins=0, which proves that door was a DIFFERENT one: the
        # min_group_size floor above could not see this at all.
        #
        # THE GUARD SITS ABOVE THE GROUP LOOP, not inside it, because n_bins is a
        # precondition of every group's estimate and of the `measured` list the
        # average is taken over. Same two checks the sibling surface already
        # applies to its own n_bins in
        # post_processing.calibration.methods.HistogramBinning.
        if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
            raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}")
        if n_bins < 1:
            raise ValueError(
                f"n_bins must be >= 1, got {n_bins}. With no bins the sum that IS the "
                f"expected calibration error is never entered, so this method returned "
                f"exactly 0.0, which on this scale is PERFECT calibration, for a batch "
                f"whose error it never looked at."
            )
        # `min_group_size` is compared as `n_usable_g < min_group_size`, and every
        # comparison against NaN is False, so a non-finite floor silently became
        # NO floor: every group with at least one usable row was measured, and the
        # threshold the caller asked for was never applied. The one-usable-row
        # floor beside it would still catch the empty group, so this cannot
        # fabricate a 0.0 on its own; it is refused because a floor nobody can
        # evaluate is not the floor the caller configured.
        if not is_measured(min_group_size):
            raise ValueError(
                f"min_group_size={min_group_size!r} is not a finite number, so the "
                f"`n_usable_rows < min_group_size` test is False for every group and no "
                f"floor is applied at all. Pass a nonnegative integer."
            )

        probs = torch.sigmoid(logits)

        # A ROW WITH NO USABLE NUMBER IS NOT A ROW WITH ZERO CALIBRATION ERROR
        # (BGL3, 2026-09-27). Every bin below is entered only when
        # `in_bin.sum() > 1e-8`, and under IEEE 754 that comparison is False
        # for NaN, so a group holding a single non-finite logit or label skipped
        # EVERY bin, left `ece` at its seed of exactly 0.0, and was still
        # appended to `measured`. Measured on this code before the mask, 60 rows
        # in two groups of 30, min_group_size=10, seed 0:
        #
        #   all 60 logits NaN  -> tensor(0.)     and NOT ONE WARNING
        #   1 of 60 logits NaN -> tensor(0.0091) against a true 0.0247
        #                         (group 1 alone measures 0.0182, so group 0's
        #                          real error was replaced by 0.0 and averaged in)
        #
        # 0.0 on this scale is PERFECT calibration, so a model that had diverged
        # to NaN reported the best calibration score attainable, and one
        # unscored row out of sixty cut the reported loss by 63%. Non-finite
        # rows are excluded here instead, which puts the group below
        # min_group_size when there is nothing left to estimate from and so
        # reaches the NaN refusal the docstring promises. Nothing changes for an
        # all-finite batch: the mask is then all True.
        y_true_f = y_true.float()
        usable = _finite_rows(probs) & _finite_rows(y_true_f)
        n_unusable = int((~usable).sum())

        # Compute soft ECE (differentiable approximation)
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        measured: List[int] = []
        skipped: List[int] = []

        for g in range(self.n_groups):
            group_mask = group_ids == g
            # Reshaped to the GROUP mask's own shape rather than broadcast
            # against it. `group_ids` is (n,) while `logits` is commonly (n, 1)
            # (nn.Linear(k, 1)), and an (n,) mask ANDed with an (n, 1) one
            # broadcasts to (n, n), which indexes the probabilities with a
            # square mask. That is how the first version of this guard broke the
            # two control tests in tests/test_surface_grade_g006.py with
            # "IndexError: The shape of the mask [40, 40] ... does not match
            # [40, 1]". Keeping the mask's shape identical to what the loop used
            # before means every downstream slice behaves exactly as it did.
            mask = group_mask & usable.reshape(group_mask.shape)
            n_usable_g = int(mask.sum())
            # A GROUP WITH NO USABLE ROW HAS NO ESTIMATE AT ANY THRESHOLD
            # (BGL4 audit, 2026-09-27). `< min_group_size` alone cannot refuse
            # it, because 0 < 0 is False, so the caller-supplied and documented
            # min_group_size=0 made the NaN refusal unreachable: the group was
            # appended to `measured` and contributed its seed of exactly 0.0.
            # Measured on 60 all-NaN logits in two groups of 30, seed 0:
            #
            #   min_group_size=0  -> 0.0 with 1 warning (the excluded-rows one)
            #   min_group_size=10 -> nan with 2 warnings
            #
            # 0.0 on this scale is PERFECT calibration, so one public argument
            # turned the fixed refusal back into the fabrication it replaced.
            # After: min_group_size=0 returns nan with the same 2 warnings.
            # The floor of one usable row is deliberately NOT configurable;
            # min_group_size still governs everything above it, so
            # min_group_size=0 on a healthy batch is unchanged (60 finite rows,
            # two groups of 30: 2.160198926925659 before and after).
            if n_usable_g == 0 or n_usable_g < min_group_size:
                skipped.append(g)
                continue

            probs_g = probs[mask]
            y_g = y_true_f[mask]

            # Soft binning using temperature-weighted contributions
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

            ece = torch.tensor(0.0, device=logits.device, requires_grad=True)

            for i in range(n_bins):
                lower = bin_boundaries[i]
                upper = bin_boundaries[i + 1]

                # Soft membership (differentiable)
                in_bin = torch.sigmoid((probs_g - lower) * 10) * torch.sigmoid(
                    (upper - probs_g) * 10
                )

                if in_bin.sum() > 1e-8:
                    bin_count = in_bin.sum()
                    bin_conf = (in_bin * probs_g).sum() / (bin_count + 1e-8)
                    bin_acc = (in_bin * y_g).sum() / (bin_count + 1e-8)
                    ece = ece + bin_count * torch.abs(bin_conf - bin_acc) / probs_g.shape[0]

            measured.append(g)
            total_loss = total_loss + ece

        name = f"{type(self).__name__}.calibration_loss"
        # A ROW IN NO GROUP IS NOT A ROW THIS LOSS COVERED (BGL4 audit,
        # 2026-09-27). The loop above iterates range(self.n_groups), so a row
        # carrying an id outside that range enters no group mask and no bin, and
        # the returned number describes only the rows that did. Measured on 60
        # rows, seed 0, where 30 carry group id 7 with n_groups=2:
        #
        #   before -> 2.135953664779663, byte-identical to the loss over the
        #             in-range 30 rows ALONE, with the only warning naming
        #             "group(s) [1] are below min_group_size" (true, but it
        #             names the empty group, never the 30 uncovered rows)
        #   after  -> the same 2.135953664779663, now with "30 of 60 row(s)
        #             carry a group id outside range(2) ... covers 30 of 60"
        #
        # The VALUE is not corrected, because the rows genuinely have no
        # calibration parameters to be scored against; what was missing is that
        # a caller could not tell this loss covered half the batch.
        # _warn_unknown_group_ids in this file says "passed through
        # uncalibrated", which is true of the four forward() paths and false
        # here, so the coverage is worded for this surface instead.
        out_of_range = (group_ids < 0) | (group_ids >= self.n_groups)
        n_out_of_range = int(out_of_range.sum())
        if n_out_of_range:
            warnings.warn(
                f"{name}: {n_out_of_range} of {int(group_ids.numel())} row(s) carry a "
                f"group id outside range({self.n_groups}), so they are in no group mask "
                f"and entered no bin. The returned loss covers "
                f"{int(group_ids.numel()) - n_out_of_range} of {int(group_ids.numel())} "
                f"row(s); it is not a calibration statement about the excluded ones. "
                "Check the group encoding or increase n_groups."
            )
        if n_unusable:
            # Said out loud even when the batch is still measurable: the
            # returned loss is then an average over FEWER rows than the caller
            # handed in, and the count is the only way to tell.
            warnings.warn(
                f"{name}: {n_unusable} of {int(usable.numel())} row(s) carry no "
                f"usable number (non-finite logit or label) and were excluded. "
                f"They are NOT folded in as zero calibration error, which is what "
                f"silently happened before: a non-finite value makes every soft bin "
                f"test False, so the whole group's error read as exactly 0.0."
            )
        if not measured:
            # Usable sizes, not raw group sizes: a warning saying
            # "group sizes {0: 30, 1: 30}" beside a refusal would read as a bug
            # in the refusal rather than as an unusable batch.
            sizes = {
                g: int(((group_ids == g) & usable.reshape((group_ids == g).shape)).sum())
                for g in range(self.n_groups)
            }
            warnings.warn(
                f"{name}: no group reached min_group_size={min_group_size} in this "
                f"batch (usable group sizes {sizes}, {n_unusable} row(s) excluded as "
                f"non-finite), so no calibration error was measured. "
                "Returning NaN; 0.0 on this scale would read as perfect calibration."
            )
            return torch.tensor(float("nan"), device=logits.device)

        if skipped:
            warnings.warn(
                f"{name}: group(s) {skipped} are below min_group_size="
                f"{min_group_size} and carry no calibration estimate. The loss is "
                f"averaged over the {len(measured)} group(s) actually measured, not "
                f"over all {self.n_groups}; a skipped group is not zero error."
            )

        return self.calibration_loss_weight * total_loss / len(measured)

    def get_calibration_state(self) -> CalibrationState:
        """Get current calibration state."""
        params = self.calibrator.get_parameters()
        return CalibrationState(
            method=self.method.value,
            parameters=params,
        )

    def freeze(self):
        """Freeze calibration parameters."""
        for param in self.calibrator.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze calibration parameters."""
        for param in self.calibrator.parameters():
            param.requires_grad = True


class CalibrationAwareTrainer:
    """
    Helper class for training with calibration awareness.

    This class provides utilities for training models with integrated
    group-specific calibration.

    Args:
        model: The main model to train
        calibrator: Group calibrator to use
        calibration_epochs: Number of epochs for calibration fine-tuning, read
            by :meth:`fine_tune_calibration`. Keyword-only, see below.

    There is NO calibration schedule on this class, and no parameter that
    pretends there is. F23, 2026-09-10: `calibration_frequency` was written
    once in this constructor and read nowhere. Measured over 8 optimisation
    steps with a fixed seed, frequency 1, 7 and 99 produced identical loss
    traces (final total_loss 0.6608137488 for all three) and bit-identical
    model weights, and a `__getattribute__` spy recorded 0 reads of
    `self.calibration_frequency`. Nothing here counts steps, so the schedule it
    named could not have been honoured; `train_step` updates the calibrator on
    EVERY call it is given. It was REMOVED rather than implemented, because
    honouring it would mean this class owning a training loop it deliberately
    does not own. `calibration_epochs` is keyword-only so that a third
    positional argument, which used to be the frequency, is a loud TypeError
    instead of silently landing on the epoch count.

    Cadence is YOURS: call :meth:`fine_tune_calibration` from your own loop as
    often as you want it to run.

    Example:
        >>> model = YourModel()
        >>> calibrator = TrainableGroupCalibrator(n_groups=2)
        >>> trainer = CalibrationAwareTrainer(model, calibrator)
        >>> # One optimisation step. There is no `train()` loop on this class:
        >>> # it steps the model you already own, so YOUR loop stays yours.
        >>> stats = trainer.train_step(x, y, group_ids, optimizer, task_loss_fn)
        >>> # Then, at whatever cadence your own loop decides:
        >>> ece = trainer.fine_tune_calibration(val_loader, calibrator_optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        calibrator: TrainableGroupCalibrator,
        *,
        calibration_epochs: int = 5,
    ):
        check_torch_available()

        self.model = model
        self.calibrator = calibrator
        self.calibration_epochs = calibration_epochs

    def train_step(
        self,
        x: "torch.Tensor",
        y: "torch.Tensor",
        group_ids: "torch.Tensor",
        optimizer: "torch.optim.Optimizer",
        task_loss_fn: nn.Module,
        include_calibration: bool = True,
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            x: Input features
            y: True labels
            group_ids: Group identifiers
            optimizer: Optimizer for model + calibrator
            task_loss_fn: Task loss function
            include_calibration: Whether to include calibration loss

        Returns:
            Dictionary of loss values. ``calibration_loss`` is NaN, never 0.0,
            whenever no calibration error was measured on this batch: either
            ``include_calibration=False``, or the calibrator refused because no
            group reached its minimum size. Both used to report 0.0, which on
            this scale is a perfectly calibrated batch, so a training log could
            not tell a calibration term that was switched off from one that had
            nothing left to correct.
        """
        optimizer.zero_grad()

        logits = self.model(x)
        calibrated_logits = self.calibrator(logits, group_ids)

        task_loss = task_loss_fn(calibrated_logits, y)

        cal_loss_value = float("nan")
        total_loss = task_loss
        if include_calibration:
            cal_loss = self.calibrator.calibration_loss(calibrated_logits, y, group_ids)
            cal_loss_value = float(cal_loss.item())
            if not bool(torch.isnan(cal_loss)):
                total_loss = task_loss + cal_loss
            # else: the calibrator refused (it warned why). The term is left
            # out of the objective rather than poisoning every gradient with
            # NaN, and the returned dict reports NaN so the omission is
            # visible to the caller.

        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "calibration_loss": cal_loss_value,
        }

    def fine_tune_calibration(
        self,
        val_loader: Any,
        calibrator_optimizer: "torch.optim.Optimizer",
    ) -> float:
        """
        Fine-tune only the calibration parameters on validation data.

        Args:
            val_loader: Validation data loader
            calibrator_optimizer: Optimizer for calibrator only

        Returns:
            Mean calibration loss over the batches that were actually
            measured, or NaN when none were.

            NaN is the refusal, and it replaces a 0.0 that meant three
            different things at once. An empty ``val_loader``,
            ``calibration_epochs=0`` and batches too small for the calibrator
            to estimate anything all fell through to ``else 0.0``, and 0.0 on
            this scale is a perfectly calibrated model. The class docstring
            binds this return to the name ``ece``, so that zero was being read
            as a measured calibration error nobody had computed.
        """
        self.model.eval()
        self.calibrator.unfreeze()

        total_cal_loss = 0.0
        n_batches = 0
        n_unmeasured = 0

        for epoch in range(self.calibration_epochs):
            for batch in val_loader:
                x, y, group_ids = batch

                calibrator_optimizer.zero_grad()

                with torch.no_grad():
                    logits = self.model(x)

                calibrated_logits = self.calibrator(logits, group_ids)
                cal_loss = self.calibrator.calibration_loss(calibrated_logits, y, group_ids)

                if bool(torch.isnan(cal_loss)):
                    # The calibrator refused for this batch and said why. It
                    # carries no gradient and must not be averaged in: a
                    # backward pass here would push NaN into every calibration
                    # parameter.
                    n_unmeasured += 1
                    continue

                cal_loss.backward()
                calibrator_optimizer.step()

                total_cal_loss += cal_loss.item()
                n_batches += 1

        if n_batches == 0:
            warnings.warn(
                "CalibrationAwareTrainer.fine_tune_calibration: nothing was measured "
                f"({self.calibration_epochs} epoch(s), {n_unmeasured} batch(es) the "
                "calibrator refused, 0 measured), so no calibration parameter was "
                "updated. Returning NaN; 0.0 would read as perfect calibration."
            )
            return float("nan")

        if n_unmeasured:
            warnings.warn(
                f"CalibrationAwareTrainer.fine_tune_calibration: {n_unmeasured} "
                f"batch(es) carried no calibration estimate and were skipped; the "
                f"returned loss is the mean over the {n_batches} measured batch(es)."
            )

        return total_cal_loss / n_batches


def create_group_calibrator(
    n_groups: int,
    method: Union[str, CalibrationMethodType] = "temperature",
    learnable: bool = True,
    **kwargs,
) -> TrainableGroupCalibrator:
    """
    Factory function to create group calibrators.

    Args:
        n_groups: Number of demographic groups
        method: Calibration method
        learnable: Whether parameters are trainable
        **kwargs: Additional arguments

    Returns:
        Configured group calibrator
    """
    return TrainableGroupCalibrator(
        n_groups=n_groups,
        method=method,
        learnable=learnable,
        **kwargs,
    )
