"""
Fairness-Aware Loss Functions for PyTorch.

This module implements a comprehensive suite of fairness-aware loss functions
that incorporate penalties for fairness violations directly into the training
objective. These losses enable model training that optimizes for both prediction
accuracy and fairness constraints simultaneously.

Loss Functions Implemented:
    1. FairnessAwareBCELoss: General-purpose fairness-aware binary cross-entropy
    2. DemographicParityLoss: Penalizes differences in positive prediction rates
    3. EqualizedOddsLoss: Penalizes differences in TPR and FPR across groups
    4. EqualOpportunityLoss: Penalizes differences in TPR (true positive rate)
    5. PredictiveParityLoss: Penalizes differences in precision across groups
    6. CalibrationLoss: Penalizes differences in calibration across groups

Mathematical Formulations:

    Demographic Parity:
        L_fairness = Σ|P(ŷ=1|G=g) - P(ŷ=1)|

    Equalized Odds:
        L_fairness = Σ|TPR_g - TPR| + Σ|FPR_g - FPR|

    Equal Opportunity:
        L_fairness = Σ|TPR_g - TPR|

References:
    - Hardt et al. (2016): Equality of Opportunity in Supervised Learning
    - Zafar et al. (2017): Fairness Constraints: Mechanisms for Fair Classification
    - Chouldechova (2017): Fair Prediction with Disparate Impact
    - Kleinberg et al. (2017): Inherent Trade-Offs in the Fair Determination of Risk Scores
"""

import math
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from .base import (
    TORCH_AVAILABLE,
    BaseFairnessLoss,
    BaseLossType,
    FairnessMetricType,
    LossComponents,
    TrainingMetrics,
    check_torch_available,
    compute_group_rates,
    create_group_masks,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


# The band FairnessAwareBCELoss.temperature is allowed to take.
#
# The floor is not cosmetic. `logit / T` is evaluated in the tensor's dtype,
# so a temperature that rounds to 0.0 there divides by zero: the forward loss
# stays finite and plausible while EVERY gradient entry becomes NaN. Measured
# in float32 (2026-09-09): T = 1e-45 gives finite gradients, T = 1e-46 gives
# 200 NaNs out of 200 and one SGD step turns the weight into NaN. Nothing
# usable is refused by stopping well above that, because the penalty is
# already inert far higher up: below about T = 1e-4 its gradient is exactly
# zero for every sample, so it contributes a constant and trains nothing.
TEMPERATURE_MIN = 1e-6


#: The metrics ``FairnessAwareBCELoss._compute_fairness_penalty`` actually
#: dispatches on. ``FairnessMetricType`` carries two more
#: (INDIVIDUAL_FAIRNESS, COUNTERFACTUAL_FAIRNESS) that this class has no branch
#: for, so they are refused in ``__init__`` rather than reaching a dispatch
#: whose fall-through the degenerate-input guards above it can pre-empt.
_BCE_SUPPORTED_METRICS = frozenset(
    {
        FairnessMetricType.DEMOGRAPHIC_PARITY,
        FairnessMetricType.EQUALIZED_ODDS,
        FairnessMetricType.EQUAL_OPPORTUNITY,
        FairnessMetricType.PREDICTIVE_PARITY,
        FairnessMetricType.CALIBRATION,
    }
)


def _temper_predictions(y_pred: "torch.Tensor", temperature: float) -> "torch.Tensor":
    """Re-scale probabilities on the logit scale: sigmoid(logit(p) / T).

    T = 1 is the identity, T -> 0 approaches the hard 0/1 decision at 0.5,
    T -> inf flattens everything to 0.5. The input is clamped away from 0
    and 1 so the logit is finite; the operation stays differentiable.

    The clamp comes from the tensor's OWN dtype rather than a hardcoded
    1e-7, because 1.0 - 1e-7 rounds to exactly 1.0 in float16 and bfloat16.
    There the upper "bound" bounded nothing, a prediction of 1.0 reached
    log(p / (1 - p)) = inf, and the gradient came back NaN. With
    torch.finfo(dtype).eps the sentence above is true in every float dtype
    (checked by execution in float16, bfloat16, float32 and float64).
    """
    eps = torch.finfo(y_pred.dtype).eps
    p = y_pred.clamp(eps, 1.0 - eps)
    return torch.sigmoid(torch.log(p / (1.0 - p)) / temperature)


# ---------------------------------------------------------------------------
# Three states for a fairness penalty: measured / partially measured / not
# measured at all.
#
# BGL Stage 2 (2026-09-16). Every loss in this package returns a finite 0.0
# tensor when the comparison it exists to make cannot be made: one group in
# the batch, no positive labels, no counterfactual, an adversary still at its
# random initialisation. The TENSOR must stay a finite zero, because it feeds
# a backward pass and one NaN poisons every gradient in the graph. What was
# wrong is what the loss SAYS about that zero: base.forward copied it into
# LossComponents.fairness_loss, where 0.0 is the value a perfectly fair model
# earns, and end_epoch averaged it into TrainingMetrics.avg_fairness_loss, so
# an unmeasurable batch silently pulled the epoch average towards "fairer"
# (measured: 9 real batches at 0.514286 plus 1 unmeasurable batch reported
# 0.462857).
#
# So the steering value and the reported measurement are separated here:
#   - the tensor handed to the optimizer is unchanged;
#   - LossComponents.fairness_loss becomes float("nan") when nothing was
#     measured, and batch_metrics carries WHY;
#   - end_epoch averages only the batches that measured something and reports
#     the unassessable count and reasons in convergence_info;
#   - a UserWarning names the reason at the batch where it happened.
# LossComponents lives in base.py and is shared with other subsystems, so the
# disclosure uses its existing free-form batch_metrics dict rather than new
# fields.
# ---------------------------------------------------------------------------

#: batch_metrics key holding True (measured), False (nothing measurable) or
#: None (the loss did not record coverage at all).
COVERAGE_KEY = "fairness_penalty_assessed"

#: batch_metrics key holding the reason string when COVERAGE_KEY is not True.
COVERAGE_REASON_KEY = "fairness_unassessable_reason"


class _CoverageTrackingLoss(BaseFairnessLoss):
    """Base class that records whether each batch's fairness penalty was measured.

    Subclasses call :meth:`_mark_fairness_assessed` or
    :meth:`_mark_fairness_unassessable` from inside
    ``_compute_fairness_penalty`` (or from their own ``forward``, via
    :meth:`_record_fairness_coverage`). A subclass that records nothing is
    reported as ``fairness_penalty_assessed=None``, the could-not-check
    state, rather than silently passing its zero off as a measurement.
    """

    # Class-level default so the attribute reads cleanly through
    # nn.Module.__getattr__ before the first forward.
    _fairness_coverage: Optional[Dict[str, Any]] = None

    def _mark_fairness_assessed(self, **detail: Any) -> None:
        """Record that the penalty for this batch IS a measurement."""
        coverage: Dict[str, Any] = {COVERAGE_KEY: True, COVERAGE_REASON_KEY: None}
        coverage.update(detail)
        self._fairness_coverage = coverage

    def _mark_fairness_unassessable(self, reason: str, message: str, **detail: Any) -> None:
        """Record that nothing was measured, and say so out loud.

        Args:
            reason: Short machine-readable reason, e.g. ``"single_group"``.
            message: Sentence for the warning, naming what was missing.
            detail: Extra keys copied into ``batch_metrics``.
        """
        coverage: Dict[str, Any] = {COVERAGE_KEY: False, COVERAGE_REASON_KEY: reason}
        coverage.update(detail)
        self._fairness_coverage = coverage
        warnings.warn(
            f"{type(self).__name__}: {message} The fairness penalty for this batch "
            f"is NOT MEASURED (reason: {reason}). LossComponents.fairness_loss is "
            f"NaN and the batch is left out of the epoch average; the finite value "
            f"the optimizer actually saw is kept in "
            f"batch_metrics['fairness_loss_unassessed_value'].",
            UserWarning,
            stacklevel=3,
        )

    def _warn_partial_fairness_coverage(self, message: str) -> None:
        """Warn that only PART of the comparison was possible."""
        warnings.warn(
            f"{type(self).__name__}: {message} The reported penalty is a lower "
            f"bound over what could be compared, not the full criterion; see "
            f"LossComponents.batch_metrics for the coverage.",
            UserWarning,
            stacklevel=3,
        )

    def _record_fairness_coverage(self, components: LossComponents) -> LossComponents:
        """Copy the coverage of the last penalty onto ``components``, in place."""
        coverage = self._fairness_coverage
        if coverage is None:
            coverage = {COVERAGE_KEY: None, COVERAGE_REASON_KEY: "coverage_not_recorded"}
        components.batch_metrics.update(coverage)
        if coverage[COVERAGE_KEY] is not True:
            # Never report an unmeasured penalty as a number: 0.0 here is the
            # score a perfectly fair model earns.
            components.batch_metrics["fairness_loss_unassessed_value"] = components.fairness_loss
            components.fairness_loss = float("nan")
        return components

    def forward(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        sample_weight: Optional["torch.Tensor"] = None,
        return_components: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", LossComponents]]:
        """As :meth:`BaseFairnessLoss.forward`, plus the coverage disclosure."""
        self._fairness_coverage = None
        want_components = return_components or self.track_metrics
        result = super().forward(
            y_pred,
            y_true,
            sensitive_attr,
            sample_weight,
            return_components=want_components,
        )
        if not want_components:
            return cast("torch.Tensor", result)
        total_loss, components = cast(Tuple["torch.Tensor", LossComponents], result)
        self._record_fairness_coverage(components)
        if return_components:
            return total_loss, components
        return total_loss

    def end_epoch(self) -> Optional[TrainingMetrics]:
        """Aggregate the epoch, averaging fairness over MEASURED batches only.

        ``avg_fairness_loss`` is NaN when no batch in the epoch could be
        measured, and ``convergence_info`` always carries the counts, so a
        reader can tell a fair epoch from an unmeasurable one.
        """
        if not self.track_metrics or not self._batch_history:
            return None

        history = list(self._batch_history)
        n_batches = len(history)
        assessed = [b for b in history if b.batch_metrics.get(COVERAGE_KEY) is True]

        reasons: Dict[str, int] = {}
        for batch in history:
            if batch.batch_metrics.get(COVERAGE_KEY) is not True:
                # `or` rather than a .get default: the key is PRESENT holding
                # None on a batch recorded by code that never marked coverage.
                reason = batch.batch_metrics.get(COVERAGE_REASON_KEY) or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1

        avg_total = sum(b.total_loss for b in history) / n_batches
        avg_task = sum(b.task_loss for b in history) / n_batches
        if assessed:
            avg_fairness = sum(b.fairness_loss for b in assessed) / len(assessed)
        else:
            avg_fairness = float("nan")

        if reasons:
            warnings.warn(
                f"{type(self).__name__}.end_epoch: {len(assessed)} of {n_batches} "
                f"batches measured a fairness penalty; {n_batches - len(assessed)} "
                f"could not be measured ({reasons}). avg_fairness_loss is the mean "
                f"over the measured batches only"
                + (" and is NaN because there were none." if not assessed else "."),
                UserWarning,
                stacklevel=2,
            )

        metrics = TrainingMetrics(
            epoch=self._current_epoch,
            avg_total_loss=avg_total,
            avg_task_loss=avg_task,
            avg_fairness_loss=avg_fairness,
            convergence_info={
                "n_batches": n_batches,
                "n_batches_fairness_assessed": len(assessed),
                "n_batches_fairness_unassessable": n_batches - len(assessed),
                "fairness_unassessable_reasons": reasons,
            },
        )

        self._epoch_metrics.append(metrics)
        self._batch_history = []

        return metrics


def _measurable_group_rates(
    y_pred: "torch.Tensor",
    y_true: "torch.Tensor",
    sensitive_attr: "torch.Tensor",
    label: float,
) -> Tuple[Dict[Any, "torch.Tensor"], List[Any]]:
    """Per-group mean prediction over rows with ``y_true == label``.

    Returns the rates of the groups that HAVE such rows, and the list of
    groups that do not. A group with no row at that label has no rate; the
    package's own ``soft_rate_computation`` says so explicitly ("its rate is
    NOT MEASURED") before handing back a 0.5 stand-in, and comparing against
    that stand-in is what this function exists to avoid.
    """
    rates: Dict[Any, "torch.Tensor"] = {}
    unmeasurable: List[Any] = []
    for group, mask in create_group_masks(sensitive_attr).items():
        rows = mask & (y_true == label)
        if bool(rows.any()):
            rates[group] = y_pred[rows].mean()
        else:
            unmeasurable.append(group)
    return rates, unmeasurable


def _rate_disparity(
    rates: Dict[Any, "torch.Tensor"],
    overall: "torch.Tensor",
) -> "torch.Tensor":
    """Mean absolute deviation of the group rates from ``overall``."""
    penalty = torch.zeros((), device=overall.device, dtype=overall.dtype)
    for rate in rates.values():
        penalty = penalty + torch.abs(rate - overall)
    return penalty / len(rates)


class FairnessAwareBCELoss(_CoverageTrackingLoss):
    """
    General-purpose fairness-aware binary cross-entropy loss.

    This loss function combines standard binary cross-entropy with a
    configurable fairness penalty. It supports multiple fairness metrics
    and can be customized for different fairness objectives.

    The total loss is:
        L = BCE(y_pred, y_true) + λ * L_fairness(y_pred, sensitive_attr)

    Args:
        lambda_fairness: Trade-off parameter (default: 0.1)
        fairness_metric: Which fairness metric to optimize
        reduction: Loss reduction method ('mean', 'sum', 'none')
        track_metrics: Whether to track training metrics
        warmup_epochs: Epochs before applying fairness penalty
        temperature: Sharpness of the predictions the fairness penalty sees.
            Before any group statistic is computed the probabilities are
            re-scaled on the logit scale, p_T = sigmoid(logit(p) / T):
            T = 1 leaves them untouched (the default and the previous
            behaviour), T < 1 pushes them towards hard 0/1 decisions so the
            soft rates approach the hard rates, T > 1 flattens them towards
            0.5. Applies to every fairness_metric; the BCE term always uses
            the raw predictions. Must be > 0, finite, and at least
            TEMPERATURE_MIN (1e-6); anything else is refused at construction.
            Usable band: roughly 0.1 to 10. Outside it the parameter stops
            being a knob, in both directions and silently: below about
            T = 1e-4 the penalty saturates and its gradient is exactly zero
            for every sample, so it adds a constant and trains nothing, and
            at large T every prediction is 0.5, so the penalty tends to 0
            (predictive_parity is the exception: it tends to the group
            base-rate disparity, not to 0, by construction).

    Example:
        >>> loss_fn = FairnessAwareBCELoss(
        ...     lambda_fairness=0.1,
        ...     fairness_metric='demographic_parity'
        ... )
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)
        >>> loss.backward()

    Comparison with Fairlearn:
        Unlike Fairlearn's DemographicParityDifference which computes metrics
        post-hoc, this loss function provides differentiable fairness penalties
        that can be directly optimized during training with gradient descent.

    References:
        - Zafar et al. (2017): Fairness Constraints

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

    Ledger row: fairness_aware_bce. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        fairness_metric: Union[str, FairnessMetricType] = FairnessMetricType.DEMOGRAPHIC_PARITY,
        reduction: Literal["mean", "sum", "none"] = "mean",
        track_metrics: bool = True,
        warmup_epochs: int = 0,
        temperature: float = 1.0,
    ):
        super().__init__(
            lambda_fairness=lambda_fairness,
            base_loss=BaseLossType.BINARY_CROSS_ENTROPY,
            reduction=reduction,
            track_metrics=track_metrics,
            warmup_epochs=warmup_epochs,
        )
        self.fairness_metric = (
            FairnessMetricType(fairness_metric)
            if isinstance(fairness_metric, str)
            else fairness_metric
        )
        # BGL-S2B (2026-09-17). FairnessMetricType has seven members and this
        # class implements five; INDIVIDUAL_FAIRNESS and COUNTERFACTUAL_FAIRNESS
        # have no branch in _compute_fairness_penalty. That used to surface as
        # the `raise ValueError("Unknown fairness metric: ...")` at the end of
        # the dispatch, but the single-group guard now returns ABOVE it, so a
        # misconfigured loss reported a 'single_group' could-not-check on any
        # one-group batch and handed back a finite total loss. Measured:
        # FairnessAwareBCELoss(fairness_metric="individual_fairness") on a
        # 20-row single-group batch returned 0.8980 with only the single_group
        # warning, and raised on the very next batch that happened to have two
        # groups. A configuration error must be raised where it is made, and it
        # must never be able to hide behind a could-not-check.
        if self.fairness_metric not in _BCE_SUPPORTED_METRICS:
            supported = ", ".join(sorted(m.value for m in _BCE_SUPPORTED_METRICS))
            raise ValueError(
                f"FairnessAwareBCELoss does not implement fairness_metric="
                f"{self.fairness_metric.value!r}. Supported: {supported}."
            )
        # F14 (2026-09-09): `temperature` was stored and never used. Measured:
        # identical loss (0.8217) for 0.01, 1.0 and 100.0. The intended
        # recipient, base.soft_rate_computation, accepts a temperature and
        # ignores it too, so passing it through would have changed nothing;
        # the tempering is applied here, once, before every group statistic
        # (see _temper_predictions). T = 1 skips it, byte-identical to before.
        # The guard covers the whole arithmetic, not just the sign: a merely
        # positive temperature is not enough. inf was accepted and disabled
        # the fairness penalty entirely (penalty exactly 0.0, zero gradient,
        # i.e. plain BCE with nothing saying so), and anything that underflows
        # the compute dtype turned every gradient into NaN behind a forward
        # loss that still looked right. See TEMPERATURE_MIN above.
        if not (math.isfinite(temperature) and temperature >= TEMPERATURE_MIN):
            raise ValueError(
                f"temperature must be > 0, finite, and at least {TEMPERATURE_MIN:g}; "
                f"got {temperature!r}. Supported range: [{TEMPERATURE_MIN:g}, inf), "
                "usable band roughly 0.1 to 10. Smaller values underflow the "
                "compute dtype and make every gradient NaN, and a non-finite "
                "temperature silently removes the fairness penalty."
            )
        self.temperature = float(temperature)

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute fairness penalty based on configured metric."""
        check_torch_available()

        if self.temperature != 1.0:
            y_pred = _temper_predictions(y_pred, self.temperature)

        # All five metrics compare groups against each other, so the
        # precondition is checked ABOVE the branch selection rather than
        # once per branch: fixing one site otherwise just moves the defect
        # to its sibling.
        n_groups = int(torch.unique(sensitive_attr).numel())
        if n_groups < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"the sensitive attribute has {n_groups} distinct value(s) in this "
                f"batch, so no group can be compared with another.",
                fairness_groups_total=n_groups,
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        if self.fairness_metric == FairnessMetricType.DEMOGRAPHIC_PARITY:
            return self._demographic_parity_penalty(y_pred, sensitive_attr)
        elif self.fairness_metric == FairnessMetricType.EQUALIZED_ODDS:
            return self._equalized_odds_penalty(y_pred, y_true, sensitive_attr)
        elif self.fairness_metric == FairnessMetricType.EQUAL_OPPORTUNITY:
            return self._equal_opportunity_penalty(y_pred, y_true, sensitive_attr)
        elif self.fairness_metric == FairnessMetricType.PREDICTIVE_PARITY:
            return self._predictive_parity_penalty(y_pred, y_true, sensitive_attr)
        elif self.fairness_metric == FairnessMetricType.CALIBRATION:
            return self._calibration_penalty(y_pred, y_true, sensitive_attr)
        else:
            raise ValueError(f"Unknown fairness metric: {self.fairness_metric}")

    def _demographic_parity_penalty(
        self,
        y_pred: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute demographic parity penalty.

        Penalizes differences in positive prediction rates across groups.
        """
        group_rates = compute_group_rates(
            y_pred, torch.zeros_like(y_pred), sensitive_attr, "positive_rate"
        )

        if len(group_rates) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(group_rates)} group is present, so no positive rate can "
                f"be compared with another.",
                fairness_groups_total=len(group_rates),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        overall_rate = y_pred.mean()

        penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        for group, rate in group_rates.items():
            penalty = penalty + torch.abs(rate - overall_rate)

        self._mark_fairness_assessed(
            fairness_groups_total=len(group_rates),
            fairness_groups_compared=len(group_rates),
        )
        return penalty / len(group_rates)

    def _equalized_odds_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute equalized odds penalty.

        Penalizes differences in both TPR and FPR across groups.

        Each arm is averaged over the groups that HAVE rows at that label.
        The previous version took every group's rate from
        ``compute_group_rates``, which hands back a 0.5 stand-in (and says so
        in a warning) for a group with no positives or no negatives, and then
        divided both arms by the full group count. A group that could not be
        compared therefore contributed a difference against a number nobody
        measured, and an arm nobody could measure at all halved the arm that
        WAS measured.
        """
        n_groups = int(torch.unique(sensitive_attr).numel())
        if n_groups < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {n_groups} group is present, so no rate can be compared with another.",
                fairness_groups_total=n_groups,
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        tpr_rates, no_positives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 1.0)
        fpr_rates, no_negatives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 0.0)

        arms: Dict[str, "torch.Tensor"] = {}
        if len(tpr_rates) >= 2:
            arms["tpr"] = _rate_disparity(tpr_rates, y_pred[y_true == 1].mean())
        if len(fpr_rates) >= 2:
            arms["fpr"] = _rate_disparity(fpr_rates, y_pred[y_true == 0].mean())

        detail: Dict[str, Any] = {
            "fairness_groups_total": n_groups,
            "fairness_groups_compared": max(len(tpr_rates), len(fpr_rates)),
            "equalized_odds_arms_compared": sorted(arms),
            "equalized_odds_tpr_groups_compared": len(tpr_rates),
            "equalized_odds_fpr_groups_compared": len(fpr_rates),
            "equalized_odds_tpr_disparity": (float(arms["tpr"].item()) if "tpr" in arms else None),
            "equalized_odds_fpr_disparity": (float(arms["fpr"].item()) if "fpr" in arms else None),
            "fairness_unmeasurable_groups": {"tpr": no_positives, "fpr": no_negatives},
        }

        if len(arms) < 2:
            # Equalized odds is BOTH arms. Averaging only the arm that could
            # be compared would report the whole criterion as satisfied on
            # the strength of half of it: the arm that WAS measured is kept
            # in batch_metrics rather than passed off as the penalty.
            missing = "TPR" if "tpr" not in arms else "FPR"
            self._mark_fairness_unassessable(
                "tpr_arm_not_comparable" if "tpr" not in arms else "fpr_arm_not_comparable",
                f"the {missing} arm has fewer than two comparable groups "
                f"({len(tpr_rates)} with positive labels, {len(fpr_rates)} with "
                f"negative labels), so equalized odds is only half defined here.",
                **detail,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        penalty = sum(arms.values()) / len(arms)

        if no_positives or no_negatives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_positives} have no positive labels and groups "
                f"{no_negatives} have no negative labels in this batch, so they "
                f"were left out of their arm."
            )
        self._mark_fairness_assessed(**detail)
        return cast("torch.Tensor", penalty)

    def _equal_opportunity_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute equal opportunity penalty.

        Penalizes differences in TPR (true positive rate) across groups.

        Only the groups that HAVE positive-label rows are compared, and at
        least two of them are needed: with one, ``overall_tpr`` is that same
        group's rate, so the penalty is |t - t| = 0 by construction, which is
        the score a perfectly TPR-fair model earns.
        """
        n_groups = int(torch.unique(sensitive_attr).numel())
        tpr_rates, no_positives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 1.0)

        if len(tpr_rates) < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_groups_with_positive_labels",
                f"{len(tpr_rates)} of {n_groups} group(s) have positive labels in "
                f"this batch, so no true positive rate can be compared with another.",
                fairness_groups_total=n_groups,
                fairness_groups_compared=len(tpr_rates),
                fairness_unmeasurable_groups=no_positives,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        penalty = _rate_disparity(tpr_rates, y_pred[y_true == 1].mean())

        detail: Dict[str, Any] = {
            "fairness_groups_total": n_groups,
            "fairness_groups_compared": len(tpr_rates),
            "fairness_unmeasurable_groups": no_positives,
        }
        if no_positives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_positives} have no positive labels in this batch, so "
                f"they were left out of the TPR comparison."
            )
        self._mark_fairness_assessed(**detail)
        return penalty

    def _predictive_parity_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute predictive parity (precision parity) penalty.

        Penalizes differences in precision (PPV) across groups.
        Uses soft approximation since hard precision is non-differentiable.

        Not a defect, do not "fix" it: unlike the other four metrics this
        penalty does NOT go to zero as temperature grows. Precision here is
        E[y * p] / E[p], so when the tempering flattens p to 0.5 it becomes
        mean(y) per group, and the penalty converges to the group base-rate
        disparity. Measured 2026-09-09 with base rates {0: 0.54286,
        1: 0.54737}: the penalty settles at 0.002256, exactly the disparity
        those base rates predict.
        """
        masks = create_group_masks(sensitive_attr)

        # Precision = TP / (TP + FP) ≈ E[y * ŷ] / E[ŷ]. A group whose
        # predictions are all exactly zero has no predicted positives, so it
        # has no precision; the old code substituted 0.5 there and compared
        # against it.
        precisions = {}
        no_predicted_positives = []
        for group, mask in masks.items():
            y_pred_g = y_pred[mask]
            y_true_g = y_true[mask].float()
            if y_pred_g.sum() > 0:
                precisions[group] = (y_pred_g * y_true_g).sum() / (y_pred_g.sum() + 1e-8)
            else:
                no_predicted_positives.append(group)

        if len(precisions) < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_groups_with_predicted_positives",
                f"{len(precisions)} of {len(masks)} group(s) have any predicted "
                f"positive mass, so no precision can be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=len(precisions),
                fairness_unmeasurable_groups=no_predicted_positives,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        overall_precision = (y_pred * y_true.float()).sum() / (y_pred.sum() + 1e-8)

        penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        for group, prec in precisions.items():
            penalty = penalty + torch.abs(prec - overall_precision)

        detail: Dict[str, Any] = {
            "fairness_groups_total": len(masks),
            "fairness_groups_compared": len(precisions),
            "fairness_unmeasurable_groups": no_predicted_positives,
        }
        if no_predicted_positives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_predicted_positives} have no predicted positive mass, "
                f"so they were left out of the precision comparison."
            )
        self._mark_fairness_assessed(**detail)
        return penalty / len(precisions)

    def _calibration_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute calibration parity penalty.

        Penalizes differences in calibration error across groups.
        """
        masks = create_group_masks(sensitive_attr)

        # Using soft approximation: |E[y|ŷ] - ŷ| averaged over predictions.
        # An empty group cannot occur (the masks come from the values that ARE
        # present), so every group here has a calibration error; the old
        # zero-filled else branch was unreachable and is gone.
        cal_errors = {}
        for group, mask in masks.items():
            y_pred_g = y_pred[mask]
            y_true_g = y_true[mask].float()
            cal_errors[group] = torch.abs(y_pred_g - y_true_g).mean()

        if len(cal_errors) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(cal_errors)} group is present, so no calibration error "
                f"can be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=len(cal_errors),
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        overall_cal = torch.abs(y_pred - y_true.float()).mean()

        penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        for group, cal in cal_errors.items():
            penalty = penalty + torch.abs(cal - overall_cal)

        self._mark_fairness_assessed(
            fairness_groups_total=len(masks),
            fairness_groups_compared=len(cal_errors),
        )
        return penalty / len(cal_errors)


class DemographicParityLoss(_CoverageTrackingLoss):
    """
    Loss function enforcing demographic parity.

    Demographic parity requires that the positive prediction rate is
    equal across all demographic groups:
        P(ŷ=1|G=a) = P(ŷ=1|G=b) for all groups a, b

    The penalty is computed as the sum of absolute differences between
    group-specific positive rates and the overall positive rate.

    Mathematical formulation:
        L_DP = (1/|G|) * Σ_g |P(ŷ=1|G=g) - P(ŷ=1)|

    Args:
        lambda_fairness: Trade-off parameter (default: 0.1)
        base_loss: Base loss type (default: BCE)
        reduction: How to reduce loss
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = DemographicParityLoss(lambda_fairness=0.2)
        >>> # In training loop:
        >>> loss = loss_fn(model(x), y, sensitive_attr)
        >>> loss.backward()

    Note:
        Demographic parity may conflict with accuracy when base rates differ
        between groups. Consider using equalized odds if the goal is to
        equalize error rates rather than prediction rates.

    References:
        - Dwork et al. (2012): Fairness through Awareness
        - Zafar et al. (2017): Fairness Constraints

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

    Ledger row: demographic_parity_loss. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
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

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute demographic parity penalty."""
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so no positive rate can be "
                f"compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        overall_rate = y_pred.mean()
        group_rates = []

        for group, mask in masks.items():
            group_rate = y_pred[mask].mean()
            group_rates.append(group_rate)

        penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        for rate in group_rates:
            penalty = penalty + torch.abs(rate - overall_rate)

        self._mark_fairness_assessed(
            fairness_groups_total=len(masks),
            fairness_groups_compared=len(group_rates),
        )
        return penalty / len(group_rates)


class EqualizedOddsLoss(_CoverageTrackingLoss):
    """
    Loss function enforcing equalized odds.

    Equalized odds requires that both the true positive rate (TPR) and
    false positive rate (FPR) are equal across all demographic groups:
        P(ŷ=1|y=1,G=a) = P(ŷ=1|y=1,G=b)  (TPR equality)
        P(ŷ=1|y=0,G=a) = P(ŷ=1|y=0,G=b)  (FPR equality)

    Mathematical formulation:
        L_EO = (1/2|G|) * [Σ_g |TPR_g - TPR| + Σ_g |FPR_g - FPR|]

    Args:
        lambda_fairness: Trade-off parameter (default: 0.1)
        tpr_weight: Weight for TPR component (default: 1.0)
        fpr_weight: Weight for FPR component (default: 1.0)
        base_loss: Base loss type (default: BCE)
        reduction: How to reduce loss
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = EqualizedOddsLoss(
        ...     lambda_fairness=0.15,
        ...     tpr_weight=1.0,
        ...     fpr_weight=1.0
        ... )
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)

    Note:
        If only TPR equality is desired (for positive class protection),
        use EqualOpportunityLoss instead.

    References:
        - Hardt et al. (2016): Equality of Opportunity in Supervised Learning

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: equalized_odds_loss. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        tpr_weight: float = 1.0,
        fpr_weight: float = 1.0,
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
        # BGL-S2B (2026-09-17). A NEGATIVE ARM WEIGHT SILENTLY DELETED HALF THE
        # CRITERION AND THE RESULT WAS STILL REPORTED AS MEASURED.
        #
        # The arm selection below is `weights[arm] > 0`, which reads a negative
        # weight as "not required" exactly like a zero. Measured at the public
        # entry on 40 rows with both arms comparable: tpr_weight=-1,
        # fpr_weight=1 returned fairness_loss=0.045965 with
        # fairness_penalty_assessed=True, no warning, and
        # equalized_odds_arms_compared=['fpr', 'tpr'] -- the FPR arm alone,
        # published as equalized odds. Both weights negative was worse still: it
        # refused with reason 'zero_arm_weights', a message about weights that
        # are not zero.
        #
        # A negative weight is a configuration error, not a way to exclude an
        # arm (that is what 0.0 is for), so it is refused where it is made. The
        # both-zero case keeps its runtime refusal below: nothing is weighed
        # there, which is a fact about this loss's configuration that the caller
        # may have arrived at deliberately.
        for name, value in (("tpr_weight", tpr_weight), ("fpr_weight", fpr_weight)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(
                    f"{name} must be finite and >= 0; got {value!r}. Equalized odds is "
                    f"BOTH arms: a negative weight drops its arm from the average while "
                    f"the result still reads as a full measurement. Use 0.0 to exclude "
                    f"an arm deliberately."
                )
        self.tpr_weight = float(tpr_weight)
        self.fpr_weight = float(fpr_weight)

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute equalized odds penalty.

        Each arm is averaged over the groups that HAVE rows at that label, and
        an arm with fewer than two such groups is left out of the weighted
        average entirely instead of contributing a 0.0.

        The old code added nothing for a group it could not compare and still
        divided by the FULL group count across BOTH arms, so an arm nobody
        could measure halved the arm that was measured: a batch whose labels
        were all 1 (no FPR anywhere) reported 0.2 for a per-group TPR
        disparity of 0.4. On data where every group has both labels the value
        is unchanged, because n_contributing == n_groups in both arms.

        The unreachable ``else torch.tensor(0.5)`` defaults for overall_tpr /
        overall_fpr are gone with it: when no row carries that label, no group
        contributes to that arm, so the stand-in was never read.
        """
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so no rate can be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        tpr_rates, no_positives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 1.0)
        fpr_rates, no_negatives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 0.0)

        arms: Dict[str, "torch.Tensor"] = {}
        weights = {"tpr": self.tpr_weight, "fpr": self.fpr_weight}
        if len(tpr_rates) >= 2:
            arms["tpr"] = _rate_disparity(tpr_rates, y_pred[y_true == 1].mean())
        if len(fpr_rates) >= 2:
            arms["fpr"] = _rate_disparity(fpr_rates, y_pred[y_true == 0].mean())

        detail: Dict[str, Any] = {
            "fairness_groups_total": len(masks),
            "fairness_groups_compared": max(len(tpr_rates), len(fpr_rates)),
            "equalized_odds_arms_compared": sorted(arms),
            "equalized_odds_tpr_groups_compared": len(tpr_rates),
            "equalized_odds_fpr_groups_compared": len(fpr_rates),
            "equalized_odds_tpr_disparity": (float(arms["tpr"].item()) if "tpr" in arms else None),
            "equalized_odds_fpr_disparity": (float(arms["fpr"].item()) if "fpr" in arms else None),
            "fairness_unmeasurable_groups": {"tpr": no_positives, "fpr": no_negatives},
        }

        # An arm the caller has weighted to zero is not required; every other
        # arm is. Equalized odds is BOTH rates, so reporting the criterion
        # from the single arm that happened to be comparable would present
        # half a measurement as the whole one. The measured arm stays visible
        # in batch_metrics instead.
        required = [arm for arm in ("tpr", "fpr") if weights[arm] > 0]
        missing = [arm for arm in required if arm not in arms]
        if not required:
            self._mark_fairness_unassessable(
                "zero_arm_weights",
                f"both arms are weighted zero (tpr_weight={self.tpr_weight}, "
                f"fpr_weight={self.fpr_weight}), so nothing is weighed.",
                **detail,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        if missing:
            self._mark_fairness_unassessable(
                "_and_".join(f"{arm}_arm_not_comparable" for arm in missing),
                f"the {' and '.join(arm.upper() for arm in missing)} arm(s) have "
                f"fewer than two comparable groups ({len(tpr_rates)} with positive "
                f"labels, {len(fpr_rates)} with negative labels), so equalized odds "
                f"is not fully defined here.",
                **detail,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        weight_total = sum(weights[arm] for arm in required)
        weighted_penalty = sum(weights[arm] * arms[arm] for arm in required) / weight_total

        if no_positives or no_negatives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_positives} have no positive labels and groups "
                f"{no_negatives} have no negative labels in this batch, so they "
                f"were left out of their arm."
            )
        self._mark_fairness_assessed(**detail)

        return cast("torch.Tensor", weighted_penalty)


class EqualOpportunityLoss(_CoverageTrackingLoss):
    """
    Loss function enforcing equal opportunity.

    Equal opportunity is a relaxed version of equalized odds that only
    requires the true positive rate (TPR) to be equal across groups:
        P(ŷ=1|y=1,G=a) = P(ŷ=1|y=1,G=b) for all groups a, b

    This criterion focuses on ensuring that qualified individuals from
    all groups have an equal chance of being identified as positive.

    Mathematical formulation:
        L_EOp = (1/|G|) * Σ_g |TPR_g - TPR|

    Args:
        lambda_fairness: Trade-off parameter (default: 0.1)
        base_loss: Base loss type (default: BCE)
        reduction: How to reduce loss
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = EqualOpportunityLoss(lambda_fairness=0.1)
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)

    Use Case:
        Equal opportunity is appropriate when the positive class represents
        a desirable outcome (e.g., loan approval, job offer) and we want
        to ensure qualified individuals from all groups have equal chances.

    References:
        - Hardt et al. (2016): Equality of Opportunity in Supervised Learning

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: equal_opportunity_loss. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
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

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute equal opportunity (TPR parity) penalty.

        Two groups with positive-label rows are required, not one. With a
        single measurable group ``overall_tpr`` is that group's own rate, so
        the penalty is |t - t| = 0 by construction: the old code returned that
        0.0 for a batch whose second group had no positives at all, which is
        byte identical to what a perfectly TPR-fair model earns. The old
        ``valid_groups == 0`` guard could never fire, because the function had
        already returned when no row carried a positive label.
        """
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so no true positive rate can "
                f"be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        tpr_rates, no_positives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 1.0)

        if len(tpr_rates) < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_groups_with_positive_labels",
                f"{len(tpr_rates)} of {len(masks)} group(s) have positive labels in "
                f"this batch, so no true positive rate can be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=len(tpr_rates),
                fairness_unmeasurable_groups=no_positives,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        penalty = _rate_disparity(tpr_rates, y_pred[y_true == 1].mean())

        detail: Dict[str, Any] = {
            "fairness_groups_total": len(masks),
            "fairness_groups_compared": len(tpr_rates),
            "fairness_unmeasurable_groups": no_positives,
        }
        if no_positives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_positives} have no positive labels in this batch, so "
                f"they were left out of the TPR comparison and the penalty is the "
                f"disparity among the remaining {len(tpr_rates)}."
            )
        self._mark_fairness_assessed(**detail)

        return penalty


class FalsePositiveRateParityLoss(_CoverageTrackingLoss):
    """
    Loss function enforcing false positive rate parity.

    FPR parity requires that the false positive rate is equal across groups:
        P(ŷ=1|y=0,G=a) = P(ŷ=1|y=0,G=b) for all groups a, b

    This is the complement of equal opportunity, focusing on protecting
    individuals who should receive negative predictions.

    Mathematical formulation:
        L_FPR = (1/|G|) * Σ_g |FPR_g - FPR|

    Args:
        lambda_fairness: Trade-off parameter (default: 0.1)
        base_loss: Base loss type (default: BCE)
        reduction: How to reduce loss
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = FalsePositiveRateParityLoss(lambda_fairness=0.1)
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)

    Use Case:
        FPR parity is appropriate when false positives have significant
        negative consequences (e.g., criminal justice risk scores) and
        we want to ensure groups aren't disproportionately affected.

    References:
        - Chouldechova (2017): Fair Prediction with Disparate Impact

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: false_positive_rate_parity_loss. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
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

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute FPR parity penalty.

        Two groups with negative-label rows are required, not one: with a
        single measurable group ``overall_fpr`` is that group's own rate and
        the penalty is |f - f| = 0 by construction. Measured before this
        guard existed: a batch with one group, a batch with no negative
        labels and a batch where only one group had negatives all reported
        0.0, and end_epoch averaged them in as data (three batches, one real
        0.300, reported 0.100).
        """
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so no false positive rate can "
                f"be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        fpr_rates, no_negatives = _measurable_group_rates(y_pred, y_true, sensitive_attr, 0.0)

        if len(fpr_rates) < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_groups_with_negative_labels",
                f"{len(fpr_rates)} of {len(masks)} group(s) have negative labels in "
                f"this batch, so no false positive rate can be compared with another.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=len(fpr_rates),
                fairness_unmeasurable_groups=no_negatives,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        penalty = _rate_disparity(fpr_rates, y_pred[y_true == 0].mean())

        detail: Dict[str, Any] = {
            "fairness_groups_total": len(masks),
            "fairness_groups_compared": len(fpr_rates),
            "fairness_unmeasurable_groups": no_negatives,
        }
        if no_negatives:
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"groups {no_negatives} have no negative labels in this batch, so "
                f"they were left out of the FPR comparison and the penalty is the "
                f"disparity among the remaining {len(fpr_rates)}."
            )
        self._mark_fairness_assessed(**detail)

        return penalty


class BoundedGroupLoss(_CoverageTrackingLoss):
    """
    Loss function with bounded group-specific losses.

    This loss ensures that the task loss for any demographic group does
    not exceed a specified bound relative to other groups, implementing
    a form of minimax fairness.

    Mathematical formulation:
        L = max_g L_task(g) + λ * Var(L_task(g))

    Args:
        lambda_fairness: Weight for group variance penalty
        max_ratio: Maximum allowed ratio between group losses
        base_loss: Base loss type
        reduction: How to reduce loss
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = BoundedGroupLoss(
        ...     lambda_fairness=0.1,
        ...     max_ratio=1.5
        ... )
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr)

    References:
        - Hashimoto et al. (2018): Fairness Without Demographics
        - Mohri et al. (2019): Agnostic Federated Learning

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

    Ledger row: bounded_group_loss. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        max_ratio: float = 1.5,
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
        self.max_ratio = max_ratio

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute bounded group loss penalty.

        Both degenerate branches report could-not-check rather than 0.0: the
        between-group variance and the max/min ratio do not exist for a single
        group, and 0.0 is the value that means "the group losses are perfectly
        equal". Measured before the fix: a single-group batch and a two-group
        batch whose losses were genuinely equal produced byte identical
        LossComponents.
        """
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so neither the between-group "
                f"variance nor the max/min loss ratio exists.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        group_losses = []
        for group, mask in masks.items():
            y_pred_g = y_pred[mask]
            y_true_g = y_true[mask]

            if len(y_pred_g) > 0:
                group_loss = F.binary_cross_entropy(y_pred_g, y_true_g.float(), reduction="mean")
                group_losses.append(group_loss)

        if len(group_losses) < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_group_losses",
                f"{len(group_losses)} of {len(masks)} group(s) have rows to compute a "
                f"loss on, so neither the between-group variance nor the max/min "
                f"ratio exists.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=len(group_losses),
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        self._mark_fairness_assessed(
            fairness_groups_total=len(masks),
            fairness_groups_compared=len(group_losses),
        )

        group_losses_tensor = torch.stack(group_losses)

        variance_penalty = group_losses_tensor.var()

        # Additional penalty if max/min ratio exceeds bound
        max_loss = group_losses_tensor.max()
        min_loss = group_losses_tensor.min()
        ratio = max_loss / (min_loss + 1e-8)

        ratio_penalty = F.relu(ratio - self.max_ratio)

        return variance_penalty + ratio_penalty


def create_fairness_loss(
    fairness_type: Union[str, FairnessMetricType],
    lambda_fairness: float = 0.1,
    **kwargs,
) -> BaseFairnessLoss:
    """
    Factory function to create fairness-aware loss functions.

    Args:
        fairness_type: Type of fairness constraint
        lambda_fairness: Trade-off parameter
        **kwargs: Additional arguments for the specific loss

    Returns:
        Configured fairness-aware loss function

    Example:
        >>> loss_fn = create_fairness_loss(
        ...     'demographic_parity',
        ...     lambda_fairness=0.1
        ... )

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: create_fairness_loss. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    if isinstance(fairness_type, str):
        fairness_type = FairnessMetricType(fairness_type)

    # Imported lazily to avoid import-order coupling with the counterfactual
    # module (which also imports from .base).
    from .counterfactual import CounterfactualFairnessLoss, IndividualFairnessLoss

    loss_classes = {
        FairnessMetricType.DEMOGRAPHIC_PARITY: DemographicParityLoss,
        FairnessMetricType.EQUALIZED_ODDS: EqualizedOddsLoss,
        FairnessMetricType.EQUAL_OPPORTUNITY: EqualOpportunityLoss,
        # These two used to fall through to FairnessAwareBCELoss, whose
        # penalty dispatch does not know them, so the factory returned a
        # loss that crashed with ValueError at the first call.
        FairnessMetricType.INDIVIDUAL_FAIRNESS: IndividualFairnessLoss,
        FairnessMetricType.COUNTERFACTUAL_FAIRNESS: CounterfactualFairnessLoss,
    }

    if fairness_type not in loss_classes:
        # PREDICTIVE_PARITY and CALIBRATION are handled by the
        # general-purpose loss, which implements penalties for them.
        return FairnessAwareBCELoss(
            lambda_fairness=lambda_fairness,
            fairness_metric=fairness_type,
            **kwargs,
        )

    return loss_classes[fairness_type](lambda_fairness=lambda_fairness, **kwargs)
