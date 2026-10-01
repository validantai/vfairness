"""
Counterfactual Fairness Loss Functions.

This module implements loss functions based on counterfactual fairness,
which requires that a model's predictions would remain the same if an
individual had belonged to a different demographic group.

Key Concept:
    A prediction is counterfactually fair if:
        ŷ(x, a) = ŷ(x', a')
    where x' is the counterfactual version of x with sensitive attribute a'.

Loss Functions Implemented:
    1. CounterfactualFairnessLoss: Penalizes counterfactual prediction differences
    2. IndividualFairnessLoss: Penalizes different predictions for similar individuals
    3. CausalFairnessLoss: Incorporates causal structure for fairness

Mathematical Formulation:
    L_CF = E[|ŷ(x) - ŷ(x_cf)|²]

    where x_cf is the counterfactual observation.

References:
    - Kusner et al. (2017): Counterfactual Fairness
    - Chiappa (2019): Path-Specific Counterfactual Fairness
    - Dwork et al. (2012): Fairness through Awareness (individual fairness)
"""

from typing import Callable, List, Literal, Optional, Tuple, Union

from .base import (
    TORCH_AVAILABLE,
    BaseLossType,
    LossComponents,
    check_torch_available,
    create_group_masks,
)
from .fairness_losses import _CoverageTrackingLoss

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


class CounterfactualFairnessLoss(_CoverageTrackingLoss):
    """
    Counterfactual Fairness Loss Function.

    Enforces counterfactual fairness by penalizing differences between
    predictions for an individual and their counterfactual version
    (what prediction would be if they belonged to a different group).

    Counterfactual Generation Strategies:
        1. 'group_mean': Replace features with group means
        2. 'group_swap': Use matched individual from other group
        3. 'adversarial': Generate counterfactuals adversarially
        4. 'custom': Use provided counterfactual generator

    Mathematical formulation:
        L_CF = (1/n) Σᵢ |f(xᵢ) - f(xᵢ_cf)|²

    Args:
        lambda_fairness: Trade-off parameter for counterfactual penalty
        counterfactual_strategy: How to generate counterfactuals
        counterfactual_generator: Custom generator function (if strategy='custom')
        distance_metric: How to measure prediction differences ('l2', 'l1', 'kl')
        base_loss: Base task loss type
        track_metrics: Whether to track metrics
        warmup_epochs: Warmup period

    Example:
        >>> loss_fn = CounterfactualFairnessLoss(
        ...     lambda_fairness=0.1,
        ...     counterfactual_strategy='group_mean'
        ... )
        >>>
        >>> # Need to provide both original and counterfactual predictions
        >>> loss = loss_fn(
        ...     y_pred, y_true, sensitive_attr,
        ...     y_pred_counterfactual=y_pred_cf
        ... )

    Note:
        Generating meaningful counterfactuals requires domain knowledge.
        The 'group_mean' strategy is a simple approximation; for accurate
        counterfactuals, consider causal modeling approaches.

    References:
        - Kusner et al. (2017): Counterfactual Fairness
        - Chiappa (2019): Path-Specific Counterfactual Fairness

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

    Ledger row: counterfactual_fairness. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        counterfactual_strategy: Literal[
            "group_mean", "group_swap", "adversarial", "custom"
        ] = "group_mean",
        counterfactual_generator: Optional[Callable] = None,
        distance_metric: Literal["l2", "l1", "kl"] = "l2",
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

        self.counterfactual_strategy = counterfactual_strategy
        self.counterfactual_generator = counterfactual_generator
        self.distance_metric = distance_metric

        if counterfactual_strategy == "custom" and counterfactual_generator is None:
            raise ValueError("counterfactual_generator must be provided when strategy='custom'")

    def forward(  # type: ignore[override]  # extends base forward with keyword-only counterfactual args (PyTorch forward is intentionally not LSP-constrained)
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_pred_counterfactual: Optional["torch.Tensor"] = None,
        features: Optional["torch.Tensor"] = None,
        model: "Optional[nn.Module]" = None,
        sample_weight: Optional["torch.Tensor"] = None,
        return_components: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", LossComponents]]:
        """
        Compute counterfactual fairness loss.

        Args:
            y_pred: Predicted probabilities for original inputs
            y_true: True labels
            sensitive_attr: Sensitive attribute values
            y_pred_counterfactual: Pre-computed counterfactual predictions. If
                they are bit-identical to ``y_pred`` the batch is reported as
                NOT MEASURED, not as a perfect score: see the comment on the
                guard below. If SOME rows are bit-identical, those rows are left
                out of the mean and the coverage is reported as partial
                (``batch_metrics['fairness_penalty_partial']``,
                ``fairness_rows_compared`` of ``fairness_rows_total``), because a
                row nobody could move is an absent comparison rather than a
                distance of zero.
            features: Input features (needed if counterfactuals not provided)
            model: Model to generate counterfactual predictions
            sample_weight: Optional sample weights
            return_components: Whether to return components

        Returns:
            Total loss, optionally with components. ``LossComponents
            .fairness_loss`` is NaN for every batch in which no counterfactual
            distance could be measured, with the reason in
            ``batch_metrics['fairness_unassessable_reason']``.
        """
        check_torch_available()

        self._fairness_coverage = None

        task_loss = self._compute_task_loss(y_pred, y_true, sample_weight)

        zero = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        n_groups = int(torch.unique(sensitive_attr).numel())

        # PUT THE GUARD ABOVE THE DISPATCH (BGL3, 2026-09-27). The vacuity
        # "the counterfactual is the factual" was refused in the GENERATED arm
        # below and nowhere else, so the SUPPLIED arm walked straight past it
        # into _compute_fairness_penalty, which marks the batch ASSESSED.
        # Measured on 8 rows, two groups, lambda_fairness=1.0:
        #
        #   generated, 1 group   -> fairness_loss=nan, reason 'single_group',
        #                           one warning
        #   supplied y_pred_cf = y_pred.clone()
        #                        -> fairness_loss=0.0,
        #                           fairness_penalty_assessed=True,
        #                           NOT ONE WARNING
        #
        # 0.0 is the best attainable counterfactual-fairness score and it came
        # from comparing a prediction against itself. This is the arm a caller
        # with its own generator uses, and a generator that cannot move anybody
        # (a single-group batch, a swap with an empty peer group, a bug) returns
        # the factual, so the fabrication is reachable without any malice.
        #
        # torch.equal, not a tolerance: bit-identical is the only state that
        # cannot be a measurement, and it is the same test the generated arm
        # applies to features_cf. It is genuinely ambiguous, because a model
        # that ignores the sensitive feature entirely would also score exactly
        # 0.0, and that ambiguity IS the could-not-check: nothing here can tell
        # the two apart, so neither is asserted. The zero the optimizer sees is
        # unchanged (a zero push is right under either reading) and the number
        # survives in batch_metrics['fairness_loss_unassessed_value'].
        #
        # AND THE SAME COMPARISON ROW BY ROW (BGL5, 2026-09-27). torch.equal is
        # all-or-nothing, so the guard above fired only at 100 percent identity,
        # and a generator that could move ONE row of eight walked past it. Each
        # row it could not move hands back the factual, contributes a distance of
        # exactly 0.0, the best attainable score, and is then averaged in as if
        # it had been compared.
        #
        # Measured on 8 rows, two groups, lambda_fairness 1.0, a counterfactual
        # identical to the factual except row 0 moved by 0.8:
        #   before -> fairness_loss 0.07999999076128006 (the single compared
        #             row's squared distance 0.64, divided by all 8 rows),
        #             fairness_penalty_assessed True,
        #             fairness_unassessable_reason None, no
        #             'fairness_penalty_partial' key, warnings NONE
        #   after  -> fairness_loss 0.6399999260902405 over the 1 row that was
        #             compared, fairness_penalty_partial True,
        #             fairness_rows_compared 1 of fairness_rows_total 8, and a
        #             partial-coverage warning
        # The mixin already carried this vocabulary: _warn_partial_fairness_
        # coverage plus a fairness_penalty_partial flag, used at six sites in
        # fairness_losses.py. This loss used neither.
        #
        # AND ABOVE THE SUPPLIED/GENERATED DISPATCH TOO (BGL wave 4,
        # 2026-09-30). Both refusals above were keyed on the SHAPE matching, and
        # the comment they carried said a mismatch "leaves the mask None and the
        # whole batch is compared, exactly as before". That was the door: a
        # counterfactual holding the same n values as an (n, 1) COLUMN matched
        # neither guard, because torch.equal is False for different shapes, and
        # the penalty fell through to an unmasked F.mse_loss that BROADCAST (n,)
        # against (n, 1) into an n x n matrix of every prediction against every
        # other. Measured on 8 rows, two groups, lambda_fairness=1.0,
        # torch.manual_seed(0), y_pred=torch.rand(8):
        #
        #   FLAT counterfactual = y_pred + 0.3   -> fairness_loss
        #       0.09000000357627869, i.e. exactly 0.3 squared, 0 warnings
        #   THE SAME +0.3 as an (8, 1) COLUMN    -> 0.2361672818660736,
        #       2.6x the true value
        #   IDENTICAL VALUES as an (8, 1) COLUMN -> a MEASURED 0.14616726338863373
        #       where the flat spelling of the same eight numbers correctly
        #       refuses with NaN; fairness_penalty_partial, fairness_rows_compared
        #       and fairness_unassessable_reason were all None
        #   IDENTICAL VALUES as a (1, 8) ROW     -> 0.0, the best attainable
        #       counterfactual-fairness score, from a prediction compared with
        #       itself, fairness_penalty_assessed True
        #
        # 0.146 is the mean of the 8x8 broadcast matrix: a pairwise prediction
        # SPREAD published as a counterfactual distance, entering total_loss as a
        # real gradient. The only signal was torch's own broadcasting
        # UserWarning, which is not a disclosure from this unit and sets none of
        # its coverage fields.
        #
        # The GENERATED arm had the same two doors and neither guard at all: a
        # model whose last layer is Linear(d, 1) returns (n, 1), and a model that
        # ignores the changed features returns the factual unchanged. Measured on
        # the same fixture through the library's own group_mean strategy:
        #
        #   model returning (n,)   -> fairness_loss 0.009047199971973896
        #   the same model returning (n, 1) -> 0.009015798568725586, broadcast
        #   a model that ignores its features -> 0.0, assessed True, 0 warnings
        #
        # So the alignment and the bit-identical refusal are now ONE shared step
        # that both arms go through, rather than an inline block in one of them.
        if y_pred_counterfactual is not None:
            fairness_loss = self._penalty_from_counterfactual_predictions(
                y_pred,
                y_true,
                sensitive_attr,
                y_pred_counterfactual,
                n_groups=n_groups,
                source="supplied",
            )
        else:
            if features is None or model is None:
                self._mark_fairness_unassessable(
                    "no_counterfactual_supplied",
                    "counterfactual predictions were not provided for individual "
                    "fairness computation and cannot be generated without both "
                    "`features` and `model`.",
                    fairness_groups_total=n_groups,
                    counterfactual_strategy=self.counterfactual_strategy,
                )
                fairness_loss = zero
            else:
                features_cf = self._generate_counterfactuals(features, sensitive_attr)
                # A counterfactual that IS the factual measures nothing: the
                # model returns the same predictions and the MSE is exactly
                # 0.0, the best attainable counterfactual-fairness score,
                # obtained by asserting that the counterfactual of x is x.
                if features_cf is None or bool(torch.equal(features_cf, features)):
                    self._mark_fairness_unassessable(
                        ("single_group" if n_groups < 2 else "counterfactual_equals_factual"),
                        f"strategy {self.counterfactual_strategy!r} produced no "
                        f"counterfactual different from the original features "
                        f"({n_groups} distinct sensitive value(s) in this batch), so "
                        f"there is nothing to compare the predictions against.",
                        fairness_groups_total=n_groups,
                        counterfactual_strategy=self.counterfactual_strategy,
                    )
                    fairness_loss = zero
                else:
                    # THE SAME SHARED GUARD AS THE SUPPLIED ARM, not a second
                    # copy: aligning the shape and refusing a counterfactual
                    # nothing moved are preconditions of the penalty, not of one
                    # way of obtaining the counterfactual.
                    fairness_loss = self._penalty_from_counterfactual_predictions(
                        y_pred,
                        y_true,
                        sensitive_attr,
                        model(features_cf),
                        n_groups=n_groups,
                        source="generated",
                    )

        effective_lambda = self._get_effective_lambda()

        total_loss = task_loss + effective_lambda * fairness_loss

        if return_components or self.track_metrics:
            components = LossComponents(
                total_loss=total_loss.item(),
                task_loss=task_loss.item(),
                fairness_loss=fairness_loss.item(),
                batch_metrics={
                    "effective_lambda": effective_lambda,
                    "counterfactual_strategy": self.counterfactual_strategy,
                },
            )
            self._record_fairness_coverage(components)
            self._record_lambda_application(components)
            if self.track_metrics:
                self._batch_history.append(components)

        if return_components:
            return total_loss, components
        return total_loss

    def _penalty_from_counterfactual_predictions(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_pred_counterfactual: "torch.Tensor",
        n_groups: int,
        source: str,
    ) -> "torch.Tensor":
        """The counterfactual penalty, once predictions exist, from EITHER arm.

        Three things have to be true before the distance between the factual and
        the counterfactual predictions is a measurement, and all three are
        preconditions of the PENALTY rather than of one way of obtaining a
        counterfactual, so they live here and both arms of ``forward`` go through
        them:

        1. the two vectors line up ROW FOR ROW. A counterfactual holding the same
           n values in a different shape used to match neither refusal (a shape
           mismatch made ``torch.equal`` False and left the row mask None), and
           the penalty was then computed on a BROADCAST. See the comment in
           ``forward`` for the numbers.
        2. they are not bit-identical over every row, which measures nothing.
        3. the rows that DID move are the ones averaged, because a row nobody
           could move is an absent comparison and not a distance of zero.

        ``source`` is the word for the warning, "supplied" or "generated", so a
        reader can tell which arm produced the counterfactual that could not be
        read.
        """
        zero = torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        aligned = self._align_counterfactual_predictions(y_pred, y_pred_counterfactual)
        if aligned is None:
            self._mark_fairness_unassessable(
                "counterfactual_row_count_mismatch",
                f"the {source} counterfactual predictions hold "
                f"{int(y_pred_counterfactual.numel())} value(s) for "
                f"{int(y_pred.numel())} factual prediction(s), so they cannot be "
                f"compared row by row. The distance is NOT MEASURED rather than "
                f"taken over a broadcast, which pairs every prediction with every "
                f"other and reports the resulting spread as a counterfactual "
                f"distance. Pass one counterfactual prediction per row.",
                fairness_groups_total=n_groups,
                counterfactual_strategy=self.counterfactual_strategy,
                fairness_rows_total=int(torch.atleast_1d(y_pred).shape[0]),
                fairness_rows_compared=0,
            )
            return zero
        y_pred_counterfactual = aligned

        rows_unmoved: Optional["torch.Tensor"] = None
        if y_pred.dim() >= 1:
            rows_unmoved = (y_pred_counterfactual == y_pred).reshape(y_pred.shape[0], -1).all(dim=1)

        if bool(torch.equal(y_pred_counterfactual, y_pred)):
            self._mark_fairness_unassessable(
                "counterfactual_equals_factual",
                f"the {source} counterfactual predictions are bit-identical to the "
                f"factual ones over all {y_pred.numel()} row(s) "
                f"({n_groups} distinct sensitive value(s) in this batch), so nothing "
                f"was compared against anything. A model that is exactly invariant "
                f"would look the same from here, which is why this is reported as "
                f"not measured rather than as a perfect score.",
                fairness_groups_total=n_groups,
                counterfactual_strategy=self.counterfactual_strategy,
            )
            return zero

        return self._compute_fairness_penalty(
            y_pred,
            y_true,
            sensitive_attr,
            y_pred_counterfactual=y_pred_counterfactual,
            compare_mask=(
                None if rows_unmoved is None or not bool(rows_unmoved.any()) else ~rows_unmoved
            ),
        )

    @staticmethod
    def _align_counterfactual_predictions(
        y_pred: "torch.Tensor",
        y_pred_counterfactual: "torch.Tensor",
    ) -> Optional["torch.Tensor"]:
        """The counterfactual read row-for-row against ``y_pred``, or None.

        A counterfactual carrying one value per prediction in a different LAYOUT
        (the ``(n, 1)`` column a ``Linear(d, 1)`` head returns, or a ``(1, n)``
        row) is the same n numbers in the same order, so it is reshaped and
        compared row against row. That is not a coercion of somebody's data, it
        is the comparison this loss documents, and it is the same reconciliation
        ``AdversarialDebiasingLoss.get_adversary_accuracy`` applies to an
        ``(n, 1)`` sensitive attribute for the same reason.

        A different element COUNT cannot be reconciled and returns None, which
        ``_penalty_from_counterfactual_predictions`` reports as a
        could-not-check. Never a broadcast: ``(n,)`` against ``(n, 1)`` expands
        to an ``n x n`` matrix of every prediction against every other, whose
        mean is a pairwise prediction spread rather than a counterfactual
        distance.
        """
        if y_pred_counterfactual.shape == y_pred.shape:
            return y_pred_counterfactual
        if y_pred_counterfactual.numel() != y_pred.numel():
            return None
        return y_pred_counterfactual.reshape(y_pred.shape)

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        y_pred_counterfactual: Optional["torch.Tensor"] = None,
        compare_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute counterfactual fairness penalty.

        Args:
            compare_mask: Rows whose counterfactual prediction actually MOVED.
                The rows it excludes handed back the factual unchanged, so their
                distance of 0.0 is an absence of a comparison and not a perfect
                score; the penalty is the mean over the rows that WERE compared
                and the coverage is reported as partial. None compares the whole
                batch, which is what a fully moved counterfactual gets.
        """
        check_torch_available()

        if y_pred_counterfactual is None:
            self._mark_fairness_unassessable(
                "no_counterfactual_supplied",
                "no counterfactual predictions were supplied, so no counterfactual "
                "distance exists.",
                counterfactual_strategy=self.counterfactual_strategy,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        detail: dict = {
            "counterfactual_strategy": self.counterfactual_strategy,
            "counterfactual_distance_metric": self.distance_metric,
        }

        if compare_mask is not None:
            n_rows_total = int(compare_mask.numel())
            n_rows_compared = int(compare_mask.sum().item())
            detail["fairness_rows_total"] = n_rows_total
            detail["fairness_rows_compared"] = n_rows_compared
            if n_rows_compared < 1:
                # Cannot be reached through forward (the bit-identical guard
                # above the dispatch catches a batch nothing moved) and kept
                # anyway: a count of zero comparisons must never become a score.
                self._mark_fairness_unassessable(
                    "counterfactual_equals_factual",
                    f"none of the {n_rows_total} row(s) has a counterfactual prediction "
                    f"different from its factual one, so nothing was compared.",
                    **detail,
                )
                return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            if n_rows_compared < n_rows_total:
                detail["fairness_penalty_partial"] = True
                self._warn_partial_fairness_coverage(
                    f"{n_rows_total - n_rows_compared} of {n_rows_total} row(s) have a "
                    f"counterfactual prediction bit-identical to the factual one, so "
                    f"nothing was compared for them and they are left out of the mean "
                    f"rather than counted as a distance of 0.0, the best attainable "
                    f"counterfactual-fairness score."
                )
            y_pred = y_pred[compare_mask]
            y_pred_counterfactual = y_pred_counterfactual[compare_mask]

        if self.distance_metric == "kl":
            # A Bernoulli KL needs PROBABILITIES on both sides. Outside [0, 1]
            # the quantity does not exist (log of a non-positive number), which
            # is a could-not-check and not a penalty of any size.
            probs = (
                bool(torch.all((y_pred >= 0.0) & (y_pred <= 1.0)))
                and bool(torch.all((y_pred_counterfactual >= 0.0) & (y_pred_counterfactual <= 1.0)))
                and bool(torch.isfinite(y_pred).all())
                and bool(torch.isfinite(y_pred_counterfactual).all())
            )
            if not probs:
                self._mark_fairness_unassessable(
                    "kl_needs_probabilities",
                    "distance_metric='kl' reads both prediction vectors as Bernoulli "
                    "probabilities, and at least one value is outside [0, 1] or is not "
                    "finite, so the divergence does not exist. Pass probabilities (a "
                    "sigmoid output), or use distance_metric='l2'.",
                    **detail,
                )
                return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        self._mark_fairness_assessed(**detail)

        if self.distance_metric == "l2":
            penalty = F.mse_loss(y_pred, y_pred_counterfactual)
        elif self.distance_metric == "l1":
            penalty = F.l1_loss(y_pred, y_pred_counterfactual)
        elif self.distance_metric == "kl":
            # THE FULL BERNOULLI KL, both terms (BGL5, 2026-09-27). This was
            # F.kl_div(log(y_pred + eps), y_pred_counterfactual,
            # reduction="batchmean"), which sums only the p*log(p/q) term over
            # per-row scalars. That is not a divergence between two Bernoulli
            # distributions, it has no lower bound at 0, and it went NEGATIVE
            # whenever the counterfactual predictions sat below the factual ones,
            # which PAYS the model for counterfactual unfairness.
            #
            # Measured on 8 rows, two groups, lambda_fairness 1.0,
            # y_pred = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1] and
            # y_pred_counterfactual = y_pred * 0.5:
            #   before -> fairness_loss -0.1732867956161499,
            #             fairness_penalty_assessed True, total_loss
            #             0.12571436166763306 BELOW task_loss
            #             0.29900115728378296, warnings NONE. The documented
            #             quantity is L_CF = E[|y - y_cf|^2] >= 0, and this
            #             file's own comment calls 0.0 "the best attainable
            #             counterfactual-fairness score", so the run reported a
            #             score better than the best attainable one. The same
            #             input under l2 gives +0.08124999701976776.
            #   after  -> a non-negative divergence, 0 exactly when the two
            #             vectors agree, and the total loss can no longer fall
            #             below the task loss.
            eps = 1e-8
            p = y_pred_counterfactual.clamp(eps, 1.0 - eps)
            q = y_pred.clamp(eps, 1.0 - eps)
            penalty = (
                p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q))
            ).mean()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return penalty

    def _generate_counterfactuals(
        self,
        features: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> Optional["torch.Tensor"]:
        """
        Generate counterfactual features based on strategy.

        Args:
            features: Original input features
            sensitive_attr: Sensitive attribute values

        Returns:
            Counterfactual features, or None when this batch has no second
            group to move anybody to. The three built-in strategies used to
            return ``features.clone()`` there, which is not a counterfactual:
            the model then returns the same predictions and the penalty is
            exactly 0.0, the best attainable score. A custom generator is
            still called (it may define a counterfactual without a second
            group in the batch); forward checks its output against the
            factual instead.
        """
        check_torch_available()

        if self.counterfactual_strategy == "custom":
            # __init__ raises unless a generator is supplied for strategy='custom'.
            assert self.counterfactual_generator is not None
            return self.counterfactual_generator(features, sensitive_attr)

        # Shared precondition of all three built-in strategies, checked ABOVE
        # the branch selection so that fixing one does not leave the same
        # clone-the-original fallback live in its siblings.
        if len(create_group_masks(sensitive_attr)) < 2:
            return None

        if self.counterfactual_strategy == "group_mean":
            # Replace with means from other group
            return self._group_mean_counterfactual(features, sensitive_attr)

        elif self.counterfactual_strategy == "group_swap":
            # Swap with matched individual from other group
            return self._group_swap_counterfactual(features, sensitive_attr)

        elif self.counterfactual_strategy == "adversarial":
            # Minimal perturbation to change group membership
            return self._adversarial_counterfactual(features, sensitive_attr)

        else:
            raise ValueError(f"Unknown strategy: {self.counterfactual_strategy}")

    def _group_mean_counterfactual(
        self,
        features: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> Optional["torch.Tensor"]:
        """Generate counterfactuals using group means, or None if impossible."""
        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            return None

        group_means = {}
        for group, mask in masks.items():
            group_means[group] = features[mask].mean(dim=0)

        # For each sample, replace with the mean of the "other" group
        features_cf = features.clone()
        groups = list(masks.keys())

        for i, group in enumerate(groups):
            mask = masks[group]
            # Use mean from the other group (or first other group if >2)
            other_group = groups[(i + 1) % len(groups)]
            # expand() takes ints; mask.sum() is a 0-dim tensor that only
            # works through __index__ (mypy call-overload, 2026-09-09).
            # Measured before/after on a fixed input: identical
            # counterfactuals (sha 4b10e5b9...) and loss (0.86247289).
            n_in_group = int(mask.sum().item())
            features_cf[mask] = group_means[other_group].unsqueeze(0).expand(n_in_group, -1)

        return features_cf

    def _group_swap_counterfactual(
        self,
        features: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> Optional["torch.Tensor"]:
        """Swap with similar individuals from another group, or None."""
        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            return None

        features_cf = features.clone()
        groups = list(masks.keys())

        for i, group in enumerate(groups):
            mask = masks[group]
            other_group = groups[(i + 1) % len(groups)]
            other_mask = masks[other_group]

            # Find nearest neighbor in other group for each member
            group_features = features[mask]
            other_features = features[other_mask]

            # Shape: [n_group, n_other]
            distances = torch.cdist(group_features, other_features)

            nearest_indices = distances.argmin(dim=1)

            features_cf[mask] = other_features[nearest_indices]

        return features_cf

    def _adversarial_counterfactual(
        self,
        features: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> Optional["torch.Tensor"]:
        """Minimal adversarial perturbation counterfactuals, or None."""
        # Simple version: add noise proportional to group difference
        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            return None

        groups = list(masks.keys())
        mean_0 = features[masks[groups[0]]].mean(dim=0)
        mean_1 = features[masks[groups[1]]].mean(dim=0)

        direction = mean_1 - mean_0
        direction = direction / (direction.norm() + 1e-8)

        features_cf = features.clone()

        for group in groups:
            mask = masks[group]
            # Move towards the other group's mean
            sign = -1 if group == groups[0] else 1
            features_cf[mask] = features[mask] + sign * 0.5 * direction.unsqueeze(0)

        return features_cf


class IndividualFairnessLoss(_CoverageTrackingLoss):
    """
    Individual Fairness Loss Function.

    Enforces individual fairness by requiring that similar individuals
    receive similar predictions, regardless of their group membership.

    Mathematical formulation:
        L_IF = (1/n²) Σᵢⱼ d_Y(f(xᵢ), f(xⱼ)) - L * d_X(xᵢ, xⱼ)

    where d_X is the similarity metric in feature space and d_Y is
    the similarity metric in prediction space.

    Args:
        lambda_fairness: Trade-off parameter
        similarity_metric: How to measure feature similarity ('euclidean', 'cosine', 'custom')
        lipschitz_constant: Maximum allowed ratio of prediction to feature distance
        n_neighbors: Number of neighbors to compare (for efficiency)
        base_loss: Base task loss type
        track_metrics: Whether to track metrics

    Example:
        >>> loss_fn = IndividualFairnessLoss(
        ...     lambda_fairness=0.1,
        ...     similarity_metric='cosine',
        ...     n_neighbors=10
        ... )
        >>> loss = loss_fn(y_pred, y_true, sensitive_attr, features=x)

    Note:
        Computing pairwise similarities is O(n²), which can be expensive.
        Use n_neighbors to limit comparisons to nearest neighbors.

    References:
        - Dwork et al. (2012): Fairness through Awareness
        - Yurochkin et al. (2020): Training Individually Fair ML Models

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: individual_fairness. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        similarity_metric: Literal["euclidean", "cosine", "custom"] = "euclidean",
        lipschitz_constant: float = 1.0,
        n_neighbors: Optional[int] = None,
        custom_similarity_fn: Optional[Callable] = None,
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

        # n_neighbors is a COUNT of comparisons, and both branches of the
        # penalty divide by it. Unvalidated, n_neighbors=0 divided 0 by 0 and
        # put a NaN into the steering tensor (total loss NaN, poisoning
        # training), and n_neighbors=-1 reported fairness_loss=-0.0 with
        # fairness_penalty_assessed=True and n_pairs_compared=-8 on a batch
        # whose true violation was 0.514286: a perfect-fairness score on a
        # comparison that never ran. A count below one is a caller error, so
        # it is refused here rather than silently turned into a measurement.
        if n_neighbors is not None and n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be at least 1 (got {n_neighbors}); it is the "
                f"number of neighbours each individual is compared against. With "
                f"fewer than one, no pair is compared and the penalty is not a "
                f"measurement. Pass n_neighbors=None to compare all pairs."
            )

        self.similarity_metric = similarity_metric
        self.lipschitz_constant = lipschitz_constant
        self.n_neighbors = n_neighbors
        self.custom_similarity_fn = custom_similarity_fn

        if similarity_metric == "custom" and custom_similarity_fn is None:
            raise ValueError(
                "custom_similarity_fn must be provided when similarity_metric='custom'"
            )

    def forward(  # type: ignore[override]  # extends base forward with keyword-only `features` arg (PyTorch forward is intentionally not LSP-constrained)
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
        features: Optional["torch.Tensor"] = None,
        sample_weight: Optional["torch.Tensor"] = None,
        return_components: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", LossComponents]]:
        """
        Compute individual fairness loss.

        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            sensitive_attr: Sensitive attribute values
            features: Input features for similarity computation
            sample_weight: Optional sample weights
            return_components: Whether to return components

        Returns:
            Total loss, optionally with components
        """
        check_torch_available()

        self._fairness_coverage = None

        task_loss = self._compute_task_loss(y_pred, y_true, sample_weight)

        fairness_loss = self._compute_fairness_penalty(
            y_pred, y_true, sensitive_attr, features=features
        )

        effective_lambda = self._get_effective_lambda()

        total_loss = task_loss + effective_lambda * fairness_loss

        if return_components or self.track_metrics:
            components = LossComponents(
                total_loss=total_loss.item(),
                task_loss=task_loss.item(),
                fairness_loss=fairness_loss.item(),
                batch_metrics={"effective_lambda": effective_lambda},
            )
            self._record_fairness_coverage(components)
            self._record_lambda_application(components)
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
        features: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute individual fairness penalty.

        Every refusal reports could-not-check rather than 0.0, which on this
        scale is PERFECT individual fairness. Measured on eight identical
        individuals scored 0.95 four times and 0.05 four times: with features
        the violation is 0.514286, and without them the old code reported
        0.0. The n < 2 branch was silent as well, and an epoch of nine
        measured batches plus one unmeasurable one reported 0.462857 against
        a true 0.514286.

        The fourth refusal, added 2026-09-27, is a feature distance that does
        not exist: under ``similarity_metric='cosine'`` a zero-norm row has
        similarity 0/0, and ``F.normalize`` turned that into the MAXIMUM
        distance, which exempted every pair touching it from the Lipschitz
        check. Fewer than two rows with a distance is
        ``'feature_similarity_undefined'``; some rows without one is partial
        coverage, and the penalty is the mean over the pairs that have a
        distance, with ``n_pairs_compared`` reduced to that count.
        """
        check_torch_available()

        if features is None:
            self._mark_fairness_unassessable(
                "features_not_provided",
                "features were not provided for individual fairness computation, so "
                "no two individuals can be compared for similarity.",
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        n = features.shape[0]

        # Pairwise fairness needs at least two samples. With n=1 the upper
        # triangle mask is empty and mean() over it returned NaN, poisoning
        # the training loss. The zero TENSOR stays for that reason; what
        # changes is that the batch is no longer RECORDED as a measurement.
        if n < 2:
            self._mark_fairness_unassessable(
                "fewer_than_two_samples",
                f"the batch holds {n} sample(s), so there is no pair of individuals to compare.",
                n_samples=n,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # The pair count is computed ABOVE the branch selection because both
        # branches divide by it, i.e. they share the precondition "at least
        # one pair was actually compared". __init__ refuses n_neighbors < 1,
        # but it is a plain attribute a caller can reassign after
        # construction, so the batch-level guard stays as well: a count below
        # one is could-not-check, never a fairness score of zero.
        n_pairs_planned = (
            n * self.n_neighbors
            if self.n_neighbors is not None and self.n_neighbors < n
            else n * (n - 1) // 2
        )
        if n_pairs_planned < 1:
            self._mark_fairness_unassessable(
                "no_pairs_compared",
                f"n_neighbors={self.n_neighbors} over {n} sample(s) compares "
                f"{n_pairs_planned} pair(s), so no two individuals were compared "
                f"and no Lipschitz violation could be observed.",
                n_samples=n,
                n_pairs_compared=n_pairs_planned,
                similarity_metric=self.similarity_metric,
            )
            # Gradient-free zero, like the other two refusals: nothing
            # measured must reach the optimizer, and a NaN here would poison
            # the total loss rather than disclose the gap.
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # A ROW WITH NO DIRECTION HAS NO COSINE SIMILARITY (BGL5, 2026-09-27).
        # The cosine of a zero vector is 0/0. F.normalize divides by
        # max(||x||, 1e-12), so a zero row comes back as the zero vector,
        # cosine_sim is 0 and the distance becomes 1 - 0 = 1, the MAXIMUM: two
        # identical individuals are recorded as maximally dissimilar. With
        # lipschitz_constant 1.0 and predictions in [0, 1] no prediction gap can
        # then exceed the allowance, so every pair touching such a row is
        # exempted from the Lipschitz check in silence.
        #
        # Measured on the control fixture of this very class (eight individuals
        # scored 0.95 four times and 0.05 four times), similarity_metric='cosine',
        # which is the option the class docstring's own Example passes:
        #   features=torch.zeros(8, 3)
        #     before -> fairness_loss 0.0 (the best attainable score),
        #               fairness_penalty_assessed True,
        #               fairness_unassessable_reason None, n_pairs_compared 28,
        #               warnings NONE
        #     after  -> NaN, reason 'feature_similarity_undefined', a warning,
        #               and the 0.0 the optimizer saw kept in
        #               fairness_loss_unassessed_value
        #   features=torch.ones(8, 3) with row 0 zeroed (ONE undefined row)
        #     before -> 0.38571426272392273, assessed True, n_pairs_compared 28,
        #               warnings NONE: its seven pairs had silently dropped out
        #     after  -> 0.5142857 over the 21 pairs that exist, partial coverage
        #               flagged, n_pairs_compared 21 of 28
        # The same individuals with nonzero features measure 0.514285683631897
        # under cosine AND euclidean, so the zero rows did not change anybody;
        # they removed the comparison.
        row_has_distance: Optional["torch.Tensor"] = None
        if self.similarity_metric == "euclidean":
            feature_dists = torch.cdist(features, features)
        elif self.similarity_metric == "cosine":
            # 1e-12 is F.normalize's own eps, i.e. exactly the norm below which
            # it stops normalising and starts scaling by a constant.
            #
            # A NON-FINITE norm also answers False here, because `nan > 1e-12` is
            # False, and that is the direction this comparison must fail in: the
            # row is EXCLUDED from the comparison and counted into the disclosure
            # below, never admitted with a fabricated distance. The message says
            # "zero-norm or non-finite" so it does not misdescribe the NaN row it
            # is reporting. The shape (a NaN answering a comparison) is the one a
            # fabricated measurement has, and it was examined for that reason; here
            # the NaN is read as an ABSENCE and disclosed, which is correct. Stated
            # here rather than by citing the audit record, because that record is
            # internal and a published file must not point a reader at a document
            # they cannot open.
            row_has_distance = features.norm(dim=1) > 1e-12
            # Normalize and compute cosine distance
            features_norm = F.normalize(features, dim=1)
            cosine_sim = torch.mm(features_norm, features_norm.t())
            feature_dists = 1 - cosine_sim
        elif self.similarity_metric == "custom":
            # __init__ raises unless a similarity fn is supplied for metric='custom'.
            assert self.custom_similarity_fn is not None
            feature_dists = self.custom_similarity_fn(features)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # AND NOW ABOVE THE DISPATCH, FOR EVERY METRIC (BGL wave 4, 2026-09-30).
        # The row coverage above was computed INSIDE the cosine branch, and the
        # comment on `pair_has_distance` below asserted that None means "nothing
        # is undefined, which is every euclidean and custom call". That premise
        # was false: torch.cdist over a non-finite row returns NaN distances, and
        # a custom similarity function can return anything, so an unreadable
        # feature matrix IS undefined on those arms and was simply not looked at.
        # Measured on this class's own control fixture (eight individuals scored
        # 0.95 four times and 0.05 four times), lambda_fairness=1.0:
        #
        #   cosine,    one NaN row (guarded)  -> 0.5142857432365417 over 21
        #       pairs, n_rows_without_feature_distance 1, one warning
        #   euclidean, one NaN row            -> fairness_loss NAN,
        #       fairness_penalty_assessed TRUE, n_pairs_compared 28 (all of
        #       them claimed as compared), no reason, no warning, total_loss NAN
        #   euclidean, ALL features NaN       -> identical
        #   custom fn returning an all-NaN matrix -> identical
        #
        # That is worse than the cosine defect that was fixed, because
        # fairness_penalty_assessed=True is a POSITIVE assertion that the penalty
        # was measured while the value is NaN, and because a NaN in the total loss
        # is not neutral: it compares False against every threshold, so it
        # SUPPRESSES a finding rather than raising one, and it poisons the
        # gradient of the whole batch.
        #
        # Keyed on the distance MATRIX, which is the one thing all three arms
        # produce, rather than on any arm's own inputs. The cosine norm test
        # stays and is combined with it: a zero-norm row has a FINITE distance
        # of 1 that is a fabrication, so neither test subsumes the other.
        #
        # PAIR by pair, not row by row: one unreadable row makes its whole row
        # AND its whole column of the matrix NaN, so "every distance in this row
        # is finite" would condemn all n rows for one bad one and throw away the
        # pairs that ARE defined. A row counts as having a distance when it has
        # at least one comparable partner, which is what the coverage counts and
        # the Lipschitz sum need.
        usable_pairs = torch.isfinite(feature_dists)
        if row_has_distance is not None:
            usable_pairs = usable_pairs & (
                row_has_distance.unsqueeze(1) & row_has_distance.unsqueeze(0)
            )
        self_pair = torch.eye(n, dtype=torch.bool, device=feature_dists.device)
        row_has_distance = (usable_pairs & ~self_pair).any(dim=1)
        # The diagonal is kept TRUE in the mask handed downstream: the
        # n_neighbors branch finds self at rank 0 by its zero distance and drops
        # it with indices[:, 1:], so masking the diagonal out would make it drop
        # a real neighbour instead.
        usable_pairs = usable_pairs | self_pair

        n_rows_without_distance = int((~row_has_distance).sum().item())
        if int(row_has_distance.sum().item()) < 2:
            self._mark_fairness_unassessable(
                "feature_similarity_undefined",
                f"{n_rows_without_distance} of {n} row(s) have no usable feature "
                f"distance under similarity_metric={self.similarity_metric!r} (a "
                f"zero-norm vector, whose cosine similarity is 0/0, or a "
                f"non-finite distance, which is what torch.cdist returns for an "
                f"unreadable row), so fewer than two "
                f"individuals have a feature distance and no pair can be compared. "
                f"A zero-norm row is NOT maximally dissimilar from everybody, which "
                f"is what 1 - 0 would record, and that exempts every pair touching "
                f"it from the Lipschitz check.",
                n_samples=n,
                n_pairs_compared=0,
                n_rows_without_feature_distance=n_rows_without_distance,
                similarity_metric=self.similarity_metric,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        # Nothing undefined anywhere leaves the downstream masks None, so the
        # healthy path is bit-for-bit what it was. `all()` over the pair matrix
        # and not over the row counts: an isolated NaN from a custom similarity
        # function leaves every row with a partner while still poisoning the
        # pairs it sits in.
        all_pairs_usable = bool(usable_pairs.all())
        if all_pairs_usable:
            row_has_distance = None

        y_pred_expanded = y_pred.unsqueeze(1)
        pred_dists = torch.abs(y_pred_expanded - y_pred_expanded.t())

        # Individual fairness penalty: predictions should be similar
        # for similar individuals (within Lipschitz bound)
        # Penalty for: pred_dist > L * feature_dist

        # A pair is comparable only when BOTH of its rows have a distance. None
        # when nothing is undefined, which is every call on a feature matrix
        # whose distances are all finite and, under cosine, all of nonzero norm,
        # so that path is untouched. It used to say "which is every euclidean and
        # custom call", and that premise was the defect: those two arms produce
        # undefined distances too, they were just never looked at.
        pair_has_distance: Optional["torch.Tensor"] = None
        if row_has_distance is not None:
            # The PAIR mask computed above, not one rebuilt from the row counts:
            # a matrix whose rows all have some partner can still hold individual
            # undefined pairs, and rebuilding from rows would readmit them.
            pair_has_distance = usable_pairs

        if self.n_neighbors is not None and self.n_neighbors < n:
            # Only consider nearest neighbors. An undefined row is pushed out of
            # every neighbourhood rather than ranked as the most distant one.
            searchable = feature_dists
            if pair_has_distance is not None:
                searchable = feature_dists.masked_fill(~pair_has_distance, float("inf"))
            _, indices = searchable.topk(self.n_neighbors + 1, dim=1, largest=False)
            # Remove self (index 0)
            indices = indices[:, 1:]

            penalty = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            n_pairs_compared = 0
            for i in range(n):
                if row_has_distance is not None and not bool(row_has_distance[i]):
                    continue
                for j_idx in indices[i]:
                    j = j_idx.item()
                    if pair_has_distance is not None and not bool(pair_has_distance[i, j]):
                        continue
                    allowed_dist = self.lipschitz_constant * feature_dists[i, j]
                    violation = F.relu(pred_dists[i, j] - allowed_dist)
                    penalty = penalty + violation
                    n_pairs_compared += 1

            if n_pairs_compared < 1:
                self._mark_fairness_unassessable(
                    "no_pairs_compared",
                    f"none of the {n_pairs_planned} planned neighbour pair(s) had a "
                    f"defined feature distance, so no Lipschitz violation could be "
                    f"observed.",
                    n_samples=n,
                    n_pairs_compared=0,
                    n_rows_without_feature_distance=n_rows_without_distance,
                    similarity_metric=self.similarity_metric,
                )
                return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            # Divide by the pairs that were actually summed, not by the pairs a
            # full neighbourhood would have held. With every distance defined the
            # two are the same number (n * n_neighbors), so a healthy batch is
            # unchanged.
            penalty = penalty / n_pairs_compared
        else:
            # Full pairwise computation
            allowed_dists = self.lipschitz_constant * feature_dists
            violations = F.relu(pred_dists - allowed_dists)

            # Only upper triangle to avoid double counting
            mask = torch.triu(torch.ones_like(violations), diagonal=1).bool()
            if pair_has_distance is not None:
                mask = mask & pair_has_distance
            n_pairs_compared = int(mask.sum().item())
            penalty = violations[mask].mean()

        detail: dict = {
            "n_samples": n,
            "n_pairs_compared": n_pairs_compared,
            "similarity_metric": self.similarity_metric,
        }
        # `row_has_distance is not None` and not `if n_rows_without_distance`:
        # an isolated undefined PAIR leaves every row with some partner, so the
        # row count is 0 while pairs were still dropped, and that is exactly the
        # partial coverage this warning exists to name.
        if row_has_distance is not None:
            detail["n_rows_without_feature_distance"] = n_rows_without_distance
            detail["fairness_penalty_partial"] = True
            self._warn_partial_fairness_coverage(
                f"{n_rows_without_distance} of {n} row(s) have no usable feature "
                f"distance at all under "
                f"similarity_metric={self.similarity_metric!r} (a zero-norm vector, "
                f"whose cosine similarity is 0/0, or a non-finite distance, which is "
                f"what torch.cdist returns for an unreadable row), so the "
                f"{n_pairs_planned - n_pairs_compared} pair(s) touching them were left "
                f"out of the criterion instead of being recorded at the maximum distance "
                f"of 1, which exempts them from the Lipschitz check. The penalty below "
                f"is over the {n_pairs_compared} pair(s) that have a distance."
            )

        # THE LAST DOOR, ON THE ONE EXIT BOTH BRANCHES REACH (BGL wave 4,
        # 2026-09-30). _mark_fairness_assessed is a POSITIVE assertion that the
        # number below was measured, and a non-finite number is never a
        # measurement: it enters total_loss, poisons the gradient of the whole
        # batch, and then compares False against every threshold it is tested
        # against, so it SUPPRESSES a finding instead of raising one. The row
        # coverage above catches the cause this was found through (an unreadable
        # feature matrix); this catches the state itself, whichever branch or
        # metric produced it, so a future arm cannot reintroduce the same
        # publication.
        if not bool(torch.isfinite(penalty).all()):
            self._mark_fairness_unassessable(
                "penalty_not_finite",
                f"the individual-fairness penalty over {n_pairs_compared} pair(s) "
                f"came out non-finite under "
                f"similarity_metric={self.similarity_metric!r}, so it is not a "
                f"measurement of anything. It is NOT reported as a number, because "
                f"a NaN reaching the total loss poisons the gradient for the whole "
                f"batch and answers False to every threshold it is later compared "
                f"against, which hides the problem instead of showing it.",
                **detail,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        self._mark_fairness_assessed(**detail)

        return penalty


class CausalFairnessLoss(_CoverageTrackingLoss):
    """
    Causal Fairness Loss Function.

    Implements fairness constraints based on causal reasoning, ensuring
    that predictions are not causally influenced by sensitive attributes
    through unfair pathways.

    Only one criterion is implemented:

        'total_effect': the penalty is the absolute difference in mean
        prediction between the two groups, i.e. the total (average) effect
        of the sensitive attribute on the prediction. It needs no causal
        graph and no mediators.

    'direct_effect' and 'path_specific' are named here because the
    literature defines them (Chiappa 2019; Nabi and Shpitser 2018), and
    they are REFUSED with NotImplementedError: computing them requires
    intervening on mediator features, which this loss does not do. For the
    same reason `mediator_indices` is refused when non-empty; there is no
    code path that reads it.

    Args:
        lambda_fairness: Trade-off parameter
        causal_criterion: Must be 'total_effect' (see above)
        mediator_indices: Must be None or empty (see above)
        base_loss: Base task loss type
        track_metrics: Whether to track metrics

    Example:
        >>> loss_fn = CausalFairnessLoss(lambda_fairness=0.2)

    References:
        - Kusner et al. (2017): Counterfactual Fairness
        - Chiappa (2019): Path-Specific Counterfactual Fairness
        - Nabi & Shpitser (2018): Fair Inference on Outcomes

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

    Ledger row: causal_fairness. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    #: Criteria the literature names; only the first is implemented.
    KNOWN_CRITERIA = ("total_effect", "direct_effect", "path_specific")
    IMPLEMENTED_CRITERIA = ("total_effect",)

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        causal_criterion: Literal["total_effect"] = "total_effect",
        mediator_indices: Optional[List[int]] = None,
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

        # F4 (2026-09-09): all three criterion branches returned the same
        # total-effect penalty and mediator_indices was stored and never
        # read. Measured: identical loss (0.8128) for 'total_effect',
        # 'direct_effect' and 'path_specific', and for mediators [0, 1, 2]
        # vs [7]. The old warning fired only when mediators were ABSENT, so
        # a caller who supplied them was told nothing. A criterion that
        # silently computes a different quantity is refused, not accepted.
        if causal_criterion in ("direct_effect", "path_specific"):
            raise NotImplementedError(
                f"causal_criterion={causal_criterion!r} is not implemented: it "
                f"requires intervening on mediator features, which this loss "
                f"does not do. Only 'total_effect' is implemented."
            )
        if causal_criterion not in self.IMPLEMENTED_CRITERIA:
            raise ValueError(
                f"Unknown causal_criterion {causal_criterion!r}. Implemented: "
                f"{self.IMPLEMENTED_CRITERIA}."
            )
        if mediator_indices:
            raise NotImplementedError(
                f"mediator_indices={list(mediator_indices)!r} was given, but no "
                f"implemented criterion reads mediators ('total_effect' needs "
                f"none). Pass mediator_indices=None."
            )

        self.causal_criterion = causal_criterion
        self.mediator_indices: List[int] = []

    def _compute_fairness_penalty(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        sensitive_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute causal fairness penalty.

        The total effect is the spread between the group with the highest and
        the group with the lowest mean prediction, over ALL groups present.
        It used to be ``|mean_preds[0] - mean_preds[1]|``, the first two
        groups of the sensitive attribute and no others: three groups with
        mean predictions 0.5 / 0.5 / 1.0 reported a causal effect of exactly
        0.0 while the third sat at the prediction ceiling, a maximal
        disparity reported as none on data that was fully measurable. With
        two groups max - min IS |m0 - m1|, so nothing changes there.
        """
        check_torch_available()

        masks = create_group_masks(sensitive_attr)

        if len(masks) < 2:
            self._mark_fairness_unassessable(
                "single_group",
                f"only {len(masks)} group is present, so there is no second group "
                f"to compare mean predictions against and no causal effect exists.",
                fairness_groups_total=len(masks),
                fairness_groups_compared=0,
            )
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Compute average causal effect as difference in mean predictions
        groups = list(masks.keys())
        mean_preds = []

        for group in groups:
            mask = masks[group]
            mean_preds.append(y_pred[mask].mean())

        # Causal effect: the widest difference in expected outcomes between
        # any two groups (max pairwise spread), not just the first two.
        stacked = torch.stack(mean_preds)
        causal_effect = stacked.max() - stacked.min()

        self._mark_fairness_assessed(
            fairness_groups_total=len(masks),
            fairness_groups_compared=len(masks),
            causal_criterion=self.causal_criterion,
        )

        # Only 'total_effect' reaches here; the constructor refuses the rest
        # (F4). No other branch exists, so none can silently alias this one.
        if self.causal_criterion != "total_effect":
            raise ValueError(f"Unknown causal criterion: {self.causal_criterion}")
        return causal_effect
