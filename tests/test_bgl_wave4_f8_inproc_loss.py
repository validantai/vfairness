"""BGL wave 4, package F8: the in-processing loss functions.

Four overturned grades, all of the same shape: a guard written for one door
while a sibling door into the same fabrication stayed open.

    base.BaseFairnessLoss.forward
        the sample-weight guard tested only ``!= 0`` and ``isfinite``, so a
        NEGATIVE weight inverted the objective and a CANCELLING weight vector
        reported a perfect fit, both with task_rows_used 4 of 4 and no warning;
        and the FAIRNESS half of the same sum could be multiplied by exactly 0.0
        with no disclosure outside the optional components block

    counterfactual.CounterfactualFairnessLoss.forward
        both refusal guards were keyed on the counterfactual SHAPE matching, so
        a counterfactual supplied as an (n, 1) column skipped both and the
        penalty was computed on a BROADCAST

    counterfactual.IndividualFairnessLoss.forward
        the row-coverage guard lived inside the ``cosine`` branch only, so under
        ``euclidean`` and ``custom`` an unreadable feature matrix returned
        fairness_loss NaN with fairness_penalty_assessed TRUE

    adversarial.AdversarialDebiasingLoss.get_adversary_accuracy
        the two INPUT tensors were checked for non-finite values and the
        ADVERSARY itself was not, so a NaN network published the identical
        silent chance rate the guard exists to refuse

A NEUTERED MITIGATION REPORTS SUCCESS, NOT FAILURE, so none of these is judged
by the fairness number it produces. Each pin is paired with a control that
asserts the healthy case's REAL number, because a guard that refuses everything
passes every refusal test.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
)
from vfairness.in_processing.loss_functions.counterfactual import (  # noqa: E402
    CounterfactualFairnessLoss,
    IndividualFairnessLoss,
)
from vfairness.in_processing.loss_functions.fairness_losses import (  # noqa: E402
    DemographicParityLoss,
)

# This file's own four-row fixture, the one the base module's docstring uses.
Y_PRED = torch.tensor([0.9, 0.8, 0.2, 0.1])
Y_TRUE = torch.tensor([1.0, 1.0, 0.0, 0.0])
SENSITIVE = torch.tensor([0.0, 0.0, 1.0, 1.0])

# The honest task loss of that batch, measured.
HONEST_TASK_LOSS = 0.16425204277038574


def _call(loss_fn, *args, **kwargs):
    """Call a loss and hand back (total, components, warning messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total, components = loss_fn(*args, return_components=True, **kwargs)
    return total, components, [str(w.message) for w in caught]


class TestASampleWeightThatIsNotUsableIsNotASmallerWeight:
    """base.BaseFairnessLoss.forward, the weight half."""

    @pytest.mark.parametrize(
        "spelling,weights",
        [
            ("0-dim scalar", torch.tensor(-1.0)),
            ("full row vector", torch.full((4,), -1.0)),
            ("column shaped", torch.full((4, 1), -1.0)),
        ],
    )
    def test_a_negative_weight_is_not_a_measured_fit(self, spelling, weights):
        # Minimising a task loss of -0.164 MAXIMISES the prediction error on
        # every row. Before this guard: task_loss -0.16425204277038574 with
        # task_rows_used 4 of 4 and zero warnings, in every spelling above.
        _, components, messages = _call(
            DemographicParityLoss(lambda_fairness=0.1),
            Y_PRED,
            Y_TRUE,
            SENSITIVE,
            sample_weight=weights,
        )
        assert components.task_loss != components.task_loss, (
            f"{spelling}: a negative sample weight inverts the task loss, so it "
            f"is NOT a measurement of fit; got {components.task_loss!r}"
        )
        assert components.batch_metrics["task_rows_used"] == 0
        assert components.batch_metrics["task_rows_total"] == 4
        assert components.batch_metrics["task_loss_assessed"] is False
        assert components.batch_metrics["task_loss_unassessed_reason"] == ("negative_sample_weight")
        # The finite value the optimizer really saw is kept, not hidden.
        assert components.batch_metrics["task_loss_unassessed_value"] == pytest.approx(
            -HONEST_TASK_LOSS
        )
        assert any("negative" in m for m in messages), messages

    def test_a_cancelling_weight_vector_is_not_a_perfect_fit(self):
        # The two negative rows cancel the two positive ones. Before this
        # guard: task_loss 5.587935447692871e-09 with task_rows_used 4 of 4
        # and zero warnings, i.e. the score a PERFECT predictor earns, from a
        # model wrong by the same amount on both halves.
        _, components, messages = _call(
            DemographicParityLoss(lambda_fairness=0.1),
            Y_PRED,
            Y_TRUE,
            SENSITIVE,
            sample_weight=torch.tensor([1.0, 1.0, -1.0, -1.0]),
        )
        assert components.task_loss != components.task_loss, components.task_loss
        assert components.batch_metrics["task_rows_used"] == 0
        assert components.batch_metrics["task_loss_unassessed_reason"] == ("negative_sample_weight")
        # Specifically NOT the near-zero number it used to publish.
        assert components.batch_metrics["task_loss_unassessed_value"] == pytest.approx(
            0.0, abs=1e-6
        )
        assert messages

    def test_a_weight_below_the_dtype_resolution_is_not_a_perfect_fit(self):
        # Before: task_loss 1.6425203722938138e-31, task_rows_used 4 of 4,
        # zero warnings. The old test was `weights != 0`, an exact comparison
        # with zero on a continuous quantity.
        _, components, messages = _call(
            DemographicParityLoss(lambda_fairness=0.1),
            Y_PRED,
            Y_TRUE,
            SENSITIVE,
            sample_weight=torch.tensor(1e-30),
        )
        assert components.task_loss != components.task_loss, components.task_loss
        assert components.batch_metrics["task_rows_used"] == 0
        assert components.batch_metrics["task_loss_unassessed_reason"] == (
            "no_usable_sample_weight"
        )
        assert messages

    @pytest.mark.parametrize(
        "spelling,weights,expected",
        [
            ("no weight at all", None, HONEST_TASK_LOSS),
            ("ones(4)", torch.ones(4), HONEST_TASK_LOSS),
            ("ones(1) broadcast", torch.ones(1), HONEST_TASK_LOSS),
            ("tensor(2.0) broadcast", torch.tensor(2.0), 0.3285040855407715),
            ("full((4,), 0.01)", torch.full((4,), 0.01), 0.0016425203066319227),
            ("integer ones", torch.ones(4, dtype=torch.int64), HONEST_TASK_LOSS),
        ],
    )
    def test_the_control_still_measures_its_real_number_in_silence(
        self, spelling, weights, expected
    ):
        """THE CONTROL. A guard that refuses everything passes every refusal."""
        _, components, messages = _call(
            DemographicParityLoss(lambda_fairness=0.1),
            Y_PRED,
            Y_TRUE,
            SENSITIVE,
            sample_weight=weights,
        )
        assert components.task_loss == pytest.approx(expected, rel=1e-6), spelling
        assert components.batch_metrics["task_rows_used"] == 4, spelling
        assert components.batch_metrics["task_loss_assessed"] is True
        assert messages == [], f"{spelling} must be silent, got {messages}"

    @pytest.mark.parametrize("bad", [-0.5, -1e-9, float("nan"), float("inf")])
    def test_a_negative_or_unreadable_lambda_is_refused_at_construction(self, bad):
        # lambda_fairness=-0.5 measured total 0.25667497515678406 BELOW the task
        # loss 0.356675 on a 40-row batch with a real 0.2 disparity: gradient
        # descent was being PAID for the disparity, silently.
        with pytest.raises(ValueError, match="lambda_fairness"):
            DemographicParityLoss(lambda_fairness=bad)

    def test_the_control_lambda_values_still_construct(self):
        """THE CONTROL for the constructor guard: 0.0 and 0.5 are both legal."""
        assert DemographicParityLoss(lambda_fairness=0.0).lambda_fairness == 0.0
        assert DemographicParityLoss(lambda_fairness=0.5).lambda_fairness == 0.5


class TestAFairnessPenaltyThatIsSwitchedOffIsDisclosed:
    """base.BaseFairnessLoss.forward, the lambda half.

    A 40-row batch with a real demographic-parity disparity. The mitigation
    being off makes the fairness numbers look BETTER, not worse, so the only
    thing that can catch it is a disclosure of whether it was applied.
    """

    SENS = torch.cat([torch.zeros(20), torch.ones(20)])
    PRED = torch.cat([torch.full((20,), 0.7), torch.full((20,), 0.3)])
    TRUE = torch.cat([torch.ones(20), torch.zeros(20)])
    MEASURED_DISPARITY = 0.20000001

    def test_a_warmup_never_advanced_trains_with_the_penalty_off_out_loud(self):
        # set_epoch() is a separate manual call. Before: total_loss equalled
        # the task loss EXACTLY while LossComponents.fairness_loss still
        # reported 0.200000 as though it were applied, with zero warnings.
        loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5)
        _, components, messages = _call(loss_fn, self.PRED, self.TRUE, self.SENS)
        assert components.batch_metrics["effective_lambda"] == 0.0
        assert components.batch_metrics["fairness_penalty_applied"] is False
        assert components.batch_metrics["fairness_penalty_not_applied_reason"] == "warmup"
        assert components.batch_metrics["fairness_penalty_contribution"] == 0.0
        # The penalty WAS measured; what was not done is the mitigation.
        assert components.fairness_loss == pytest.approx(self.MEASURED_DISPARITY, rel=1e-5)
        assert components.total_loss == pytest.approx(components.task_loss)
        assert any("warmup" in m for m in messages), messages

    def test_lambda_zero_is_named_rather_than_left_looking_like_a_penalty(self):
        loss_fn = DemographicParityLoss(lambda_fairness=0.0)
        _, components, messages = _call(loss_fn, self.PRED, self.TRUE, self.SENS)
        assert components.batch_metrics["fairness_penalty_applied"] is False
        assert components.batch_metrics["fairness_penalty_not_applied_reason"] == (
            "lambda_fairness_zero"
        )
        assert any("lambda_fairness is 0.0" in m for m in messages), messages

    def test_the_suppression_is_disclosed_with_tracking_off_and_no_components(self):
        """Outside the components block, for the reason the task coverage is.

        The returned TENSOR carries the suppression whether or not the caller
        asked for components or switched tracking off, so the disclosure cannot
        live inside ``if return_components or self.track_metrics``.
        """
        loss_fn = DemographicParityLoss(lambda_fairness=0.5, warmup_epochs=5, track_metrics=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            total = loss_fn(self.PRED, self.TRUE, self.SENS)
        assert float(total.detach()) == pytest.approx(0.35667497)
        assert any("warmup" in str(w.message) for w in caught), [str(w.message) for w in caught]

    def test_the_control_applies_the_penalty_and_says_so_in_silence(self):
        """THE CONTROL: the real applied contribution, measured."""
        loss_fn = DemographicParityLoss(lambda_fairness=0.5)
        _, components, messages = _call(loss_fn, self.PRED, self.TRUE, self.SENS)
        assert components.batch_metrics["effective_lambda"] == 0.5
        assert components.batch_metrics["fairness_penalty_applied"] is True
        assert components.batch_metrics["fairness_penalty_not_applied_reason"] is None
        assert components.batch_metrics["fairness_penalty_contribution"] == pytest.approx(
            0.5 * self.MEASURED_DISPARITY, rel=1e-5
        )
        assert components.total_loss == pytest.approx(0.45667496, rel=1e-6)
        assert components.total_loss > components.task_loss, (
            "a fairness penalty that is applied RAISES the total loss; a "
            "negative lambda would lower it"
        )
        assert messages == [], messages


class TestACounterfactualInADifferentSHAPEIsNotABroadcast:
    """counterfactual.CounterfactualFairnessLoss.forward.

    Both refusals used to be keyed on the counterfactual SHAPE matching, so an
    (n, 1) column skipped both and the penalty was taken over an n x n broadcast
    of every prediction against every other.
    """

    Y_TRUE = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    SENS = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    # The flat +0.3 move, measured: exactly 0.3 squared.
    TRUE_PENALTY_FOR_A_03_MOVE = 0.09000000357627869

    @staticmethod
    def _y_pred():
        torch.manual_seed(0)
        return torch.rand(8)

    @pytest.mark.parametrize("shape", [(8, 1), (1, 8)])
    def test_identical_values_in_another_layout_are_still_not_a_measurement(self, shape):
        # Before: the (8, 1) spelling published a MEASURED 0.14616726338863373
        # (the mean of the 8x8 broadcast matrix) and the (1, 8) spelling
        # published 0.0, the best attainable counterfactual-fairness score, both
        # with fairness_penalty_assessed True and no coverage fields set, where
        # the flat spelling of the same eight numbers correctly refuses.
        y_pred = self._y_pred()
        _, components, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=y_pred.clone().reshape(shape),
        )
        assert components.fairness_loss != components.fairness_loss, (
            f"{shape}: a counterfactual holding bit-for-bit the same values as "
            f"the factual measures nothing; got {components.fairness_loss!r}"
        )
        assert components.batch_metrics["fairness_penalty_assessed"] is False
        assert components.batch_metrics["fairness_unassessable_reason"] == (
            "counterfactual_equals_factual"
        )
        assert any("bit-identical" in m for m in messages), messages

    def test_a_real_move_in_another_layout_measures_the_same_number_as_flat(self):
        # Before: the (8, 1) spelling reported 0.2361672818660736, 2.6x the true
        # value, because F.mse_loss broadcast (8,) against (8, 1).
        y_pred = self._y_pred()
        flat = y_pred + 0.3
        _, flat_components, _ = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=flat,
        )
        _, column_components, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=flat.reshape(8, 1),
        )
        assert flat_components.fairness_loss == pytest.approx(self.TRUE_PENALTY_FOR_A_03_MOVE)
        assert column_components.fairness_loss == pytest.approx(self.TRUE_PENALTY_FOR_A_03_MOVE), (
            "the column spelling must measure the same 0.3 squared, not a broadcast"
        )
        assert messages == [], messages

    @pytest.mark.parametrize(
        "bad,n_values",
        [(torch.rand(4), 4), (torch.rand(8, 2), 16), (torch.tensor(0.5), 1)],
    )
    def test_a_counterfactual_with_another_row_count_is_a_could_not_check(self, bad, n_values):
        """A count that cannot be reconciled is refused, never broadcast."""
        y_pred = self._y_pred()
        total, components, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=bad,
        )
        assert components.fairness_loss != components.fairness_loss
        assert components.batch_metrics["fairness_unassessable_reason"] == (
            "counterfactual_row_count_mismatch"
        )
        assert components.batch_metrics["fairness_rows_compared"] == 0
        # The steering tensor stays finite: a NaN there would poison the whole
        # batch's gradient and answer False to every later threshold.
        assert float(total.detach()) == pytest.approx(1.2159898281097412)
        assert any(f"{n_values} value(s)" in m for m in messages), messages

    def test_the_generated_arm_goes_through_the_same_guard(self):
        """PUT THE GUARD ABOVE THE DISPATCH: supplied and generated both.

        A model whose head is Linear(d, 1) returns (n, 1), and a model that
        ignores the changed features returns the factual unchanged. Before, the
        generated arm had neither guard: the column spelling was broadcast
        (0.009015798568725586 against a true 0.009047199971973896) and the
        feature-ignoring model published 0.0, assessed True, in silence.
        """
        import torch.nn as nn

        class FlatModel(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x.sum(dim=1))

        class ColumnModel(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x.sum(dim=1, keepdim=True))

        class FeatureIgnoringModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0],), 0.5)

        features = torch.arange(16.0).reshape(8, 2)
        results = {}
        for name, model in [
            ("flat", FlatModel()),
            ("column", ColumnModel()),
            ("ignores features", FeatureIgnoringModel()),
        ]:
            y_pred = model(features).reshape(-1).detach()
            _, components, messages = _call(
                CounterfactualFairnessLoss(
                    lambda_fairness=1.0, counterfactual_strategy="group_mean"
                ),
                y_pred,
                self.Y_TRUE,
                self.SENS,
                features=features,
                model=model,
            )
            results[name] = (components, messages)

        flat_value = results["flat"][0].fairness_loss
        assert flat_value == pytest.approx(0.009047199971973896)
        assert results["column"][0].fairness_loss == pytest.approx(flat_value), (
            "the generated counterfactual must be read row for row whatever shape the model returns"
        )
        ignoring, messages = results["ignores features"]
        assert ignoring.fairness_loss != ignoring.fairness_loss, ignoring.fairness_loss
        assert ignoring.batch_metrics["fairness_unassessable_reason"] == (
            "counterfactual_equals_factual"
        )
        assert any("generated" in m for m in messages), messages

    def test_the_controls_that_already_worked_still_work(self):
        """THE CONTROL: the flat spellings, unchanged, with their real numbers."""
        y_pred = self._y_pred()

        # A fully moved counterfactual: measured, silent.
        _, moved, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=y_pred + 0.3,
        )
        assert moved.fairness_loss == pytest.approx(self.TRUE_PENALTY_FOR_A_03_MOVE)
        assert moved.batch_metrics["fairness_penalty_assessed"] is True
        assert messages == []

        # Bit-identical, flat: the guard that was already there.
        _, identical, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=y_pred.clone(),
        )
        assert identical.fairness_loss != identical.fairness_loss
        assert identical.batch_metrics["fairness_unassessable_reason"] == (
            "counterfactual_equals_factual"
        )

        # One row of eight moved: the partial row mask that was already there.
        one_moved = y_pred.clone()
        one_moved[0] = one_moved[0] + 0.4
        _, partial, messages = _call(
            CounterfactualFairnessLoss(lambda_fairness=1.0),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            y_pred_counterfactual=one_moved,
        )
        assert partial.fairness_loss == pytest.approx(0.16, abs=1e-6)
        assert partial.batch_metrics["fairness_rows_compared"] == 1
        assert partial.batch_metrics["fairness_penalty_partial"] is True
        assert any("bit-identical to the factual one" in m for m in messages), messages


class TestTheRowCoverageGuardCoversEveryMetricNotOnlyCosine:
    """counterfactual.IndividualFairnessLoss.forward.

    The row-coverage guard lived inside the ``cosine`` branch. Under
    ``euclidean`` and ``custom`` an unreadable feature matrix returned
    fairness_loss NaN with fairness_penalty_assessed TRUE, all 28 pairs claimed
    as compared, no reason and no warning, and the NaN propagated into
    total_loss.
    """

    # This class's own control fixture: eight individuals scored 0.95 four times
    # and 0.05 four times, on tightly spaced features so a real Lipschitz
    # violation exists under euclidean as well as cosine.
    Y_PRED = torch.tensor([0.95] * 4 + [0.05] * 4)
    Y_TRUE = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    SENS = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    FEATURES = torch.ones(8, 3) * torch.tensor(
        [[1.0], [1.01], [1.02], [1.03], [1.04], [1.05], [1.06], [1.07]]
    )
    EUCLIDEAN_TRUTH = 0.47469601035118103
    COSINE_TRUTH = 0.514285683631897
    TASK_LOSS = 0.05129329860210419

    @classmethod
    def _with_nan_row(cls):
        features = cls.FEATURES.clone()
        features[0] = float("nan")
        return features

    @staticmethod
    def _cdist(features):
        return torch.cdist(features, features)

    @staticmethod
    def _all_nan(features):
        return torch.full((features.shape[0], features.shape[0]), float("nan"))

    @pytest.mark.parametrize(
        "branch,kwargs,features",
        [
            ("euclidean", {"similarity_metric": "euclidean"}, "nan_row"),
            (
                "euclidean n_neighbors",
                {"similarity_metric": "euclidean", "n_neighbors": 2},
                "nan_row",
            ),
            (
                "custom",
                {"similarity_metric": "custom", "custom_similarity_fn": _cdist.__func__},
                "nan_row",
            ),
        ],
    )
    def test_an_unreadable_row_is_partial_coverage_on_every_branch(self, branch, kwargs, features):
        # Before, on every one of these: fairness_loss NAN with
        # fairness_penalty_assessed TRUE, n_pairs_compared 28 (all of them
        # claimed), n_rows_without_feature_distance None, no reason, ZERO
        # warnings, and total_loss NAN reaching the optimizer.
        total, components, messages = _call(
            IndividualFairnessLoss(lambda_fairness=1.0, **kwargs),
            self.Y_PRED,
            self.Y_TRUE,
            self.SENS,
            features=self._with_nan_row(),
        )
        assert components.fairness_loss == components.fairness_loss, (
            f"{branch}: the 21 pairs among the seven readable rows ARE defined, "
            f"so this is partial coverage and not a refusal"
        )
        assert components.batch_metrics["n_rows_without_feature_distance"] == 1, branch
        assert components.batch_metrics["fairness_penalty_partial"] is True, branch
        assert components.batch_metrics["n_pairs_compared"] < 28, branch
        assert float(total.detach()) == float(total.detach()), (
            f"{branch}: total_loss must stay finite; a NaN there poisons the "
            f"gradient and answers False to every threshold"
        )
        assert any("no usable feature distance" in m for m in messages), messages

    @pytest.mark.parametrize(
        "branch,kwargs,features",
        [
            ("euclidean all NaN", {"similarity_metric": "euclidean"}, "all_nan"),
            (
                "custom all-NaN matrix",
                {"similarity_metric": "custom", "custom_similarity_fn": _all_nan.__func__},
                "healthy",
            ),
        ],
    )
    def test_a_wholly_unreadable_matrix_is_a_could_not_check_on_every_branch(
        self, branch, kwargs, features
    ):
        feature_matrix = (
            torch.full((8, 3), float("nan")) if features == "all_nan" else self.FEATURES
        )
        total, components, messages = _call(
            IndividualFairnessLoss(lambda_fairness=1.0, **kwargs),
            self.Y_PRED,
            self.Y_TRUE,
            self.SENS,
            features=feature_matrix,
        )
        assert components.fairness_loss != components.fairness_loss, branch
        assert components.batch_metrics["fairness_penalty_assessed"] is False, branch
        assert components.batch_metrics["fairness_unassessable_reason"] == (
            "feature_similarity_undefined"
        ), branch
        assert components.batch_metrics["n_pairs_compared"] == 0, branch
        # The NaN is refused rather than pushed into the training objective.
        assert float(total.detach()) == pytest.approx(self.TASK_LOSS), branch
        assert messages, branch

    def test_an_isolated_undefined_pair_is_dropped_and_disclosed(self):
        """Not a whole row: a single unreadable PAIR leaves every row a partner."""

        def one_nan_pair(features):
            dists = torch.cdist(features, features)
            dists[2, 5] = float("nan")
            dists[5, 2] = float("nan")
            return dists

        _, components, messages = _call(
            IndividualFairnessLoss(
                lambda_fairness=1.0,
                similarity_metric="custom",
                custom_similarity_fn=one_nan_pair,
            ),
            self.Y_PRED,
            self.Y_TRUE,
            self.SENS,
            features=self.FEATURES,
        )
        assert components.batch_metrics["n_pairs_compared"] == 27
        assert components.batch_metrics["n_rows_without_feature_distance"] == 0
        assert components.batch_metrics["fairness_penalty_partial"] is True
        assert components.fairness_loss == components.fairness_loss
        assert messages

    @pytest.mark.parametrize(
        "branch,kwargs,expected",
        [
            ("euclidean", {"similarity_metric": "euclidean"}, EUCLIDEAN_TRUTH),
            ("cosine", {"similarity_metric": "cosine"}, COSINE_TRUTH),
            (
                "custom cdist",
                {"similarity_metric": "custom", "custom_similarity_fn": _cdist.__func__},
                EUCLIDEAN_TRUTH,
            ),
        ],
    )
    def test_the_healthy_control_still_measures_its_real_number_in_silence(
        self, branch, kwargs, expected
    ):
        """THE CONTROL on all three branches, with the REAL violation."""
        _, components, messages = _call(
            IndividualFairnessLoss(lambda_fairness=1.0, **kwargs),
            self.Y_PRED,
            self.Y_TRUE,
            self.SENS,
            features=self.FEATURES,
        )
        assert components.fairness_loss == pytest.approx(expected, rel=1e-6), branch
        assert components.batch_metrics["n_pairs_compared"] == 28, branch
        assert components.batch_metrics["fairness_penalty_assessed"] is True, branch
        assert components.batch_metrics.get("fairness_penalty_partial") is None, branch
        assert messages == [], f"{branch}: {messages}"

    def test_the_cosine_guard_that_already_worked_is_not_made_stricter(self):
        """THE CONTROL for the guard this one sits beside.

        A single zero-norm row must still be PARTIAL coverage over the 21 pairs
        that exist, with the real 0.5142857432365417, and not a refusal.
        """
        features = self.FEATURES.clone()
        features[0] = 0.0
        _, components, messages = _call(
            IndividualFairnessLoss(lambda_fairness=1.0, similarity_metric="cosine"),
            self.Y_PRED,
            self.Y_TRUE,
            self.SENS,
            features=features,
        )
        assert components.fairness_loss == pytest.approx(0.5142857432365417)
        assert components.batch_metrics["n_pairs_compared"] == 21
        assert components.batch_metrics["n_rows_without_feature_distance"] == 1
        assert components.batch_metrics["fairness_penalty_assessed"] is True
        assert len(messages) == 1, messages

    @pytest.mark.parametrize("metric", ["euclidean", "cosine"])
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_a_non_finite_penalty_is_never_marked_assessed(self, metric, bad_value):
        """The catch-all on the one exit BOTH branches reach.

        The row coverage above closes the cause this was found through (an
        unreadable feature matrix). This closes the STATE: a non-finite penalty
        is never a measurement whatever produced it, and marking it assessed is
        a positive assertion that it was measured. Reached here through an
        unreadable PREDICTION, which the row coverage cannot see.
        """
        y_pred = torch.cat([torch.tensor([bad_value]), self.Y_PRED[1:]])
        _, components, messages = _call(
            IndividualFairnessLoss(lambda_fairness=1.0, similarity_metric=metric, base_loss="mse"),
            y_pred,
            self.Y_TRUE,
            self.SENS,
            features=self.FEATURES,
        )
        assert components.batch_metrics["fairness_penalty_assessed"] is False, metric
        assert components.batch_metrics["fairness_unassessable_reason"] == ("penalty_not_finite"), (
            metric
        )
        assert any("non-finite" in m for m in messages), messages


class TestABrokenAdversaryIsNotANonLeakingModel:
    """adversarial.AdversarialDebiasingLoss.get_adversary_accuracy.

    The two guards at the top of that method test y_pred and sensitive_attr for
    non-finite values. The ADVERSARY itself was tested nowhere, and a NaN network
    produces the identical silent chance rate those guards exist to refuse.
    Reachable through the library's own public API: adversary_lr is a constructor
    argument, and a large one diverges the network inside update_adversary.
    """

    @staticmethod
    def _separable_binary():
        sens = torch.cat([torch.zeros(20), torch.ones(20)])
        y_pred = torch.cat([torch.full((20,), -3.0), torch.full((20,), 3.0)])
        return y_pred, sens

    @staticmethod
    def _three_groups():
        sens = torch.cat([torch.zeros(13), torch.ones(13), torch.full((13,), 2.0)])
        y_pred = torch.cat(
            [torch.full((13,), -3.0), torch.full((13,), 0.0), torch.full((13,), 3.0)]
        )
        y_true = torch.cat([torch.ones(13), torch.zeros(13), torch.ones(13)])
        return y_pred, y_true, sens

    def _trained_binary(self, lr=0.05):
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss(n_groups=2, adversary_lr=lr)
        y_pred, sens = self._separable_binary()
        for _ in range(50):
            loss_fn.update_adversary(y_pred.detach(), sens)
        return loss_fn, y_pred, sens

    def test_the_trained_control_measures_full_leakage_in_silence(self):
        """THE CONTROL: a perfectly separable batch leaks completely, 1.0."""
        loss_fn, y_pred, sens = self._trained_binary()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            accuracy = loss_fn.get_adversary_accuracy(y_pred, sens)
        assert accuracy == 1.0
        assert caught == []
        # And in the column spelling too, which is the guard beside this one.
        assert loss_fn.get_adversary_accuracy(y_pred, sens.reshape(-1, 1)) == 1.0

    @pytest.mark.parametrize("multiplier", [float("nan"), 1e30])
    def test_a_non_finite_adversary_is_not_a_chance_leakage_rate(self, multiplier):
        # Before: every weight multiplied by NaN -> accuracy 0.5, zero warnings,
        # which is chance on this scale and which this file's own comment calls
        # "no leakage detected". The 1e30 case is the one a PARAMETER check alone
        # would miss: every weight stays finite and the adversary's OUTPUT is
        # not, because a large logit sends softmax through inf - inf.
        loss_fn, y_pred, sens = self._trained_binary()
        assert loss_fn.get_adversary_accuracy(y_pred, sens) == 1.0, "control first"
        with torch.no_grad():
            for parameter in loss_fn.adversary.parameters():
                parameter.mul_(multiplier)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            accuracy = loss_fn.get_adversary_accuracy(y_pred, sens)
        assert accuracy != accuracy, f"a broken adversary measures no leakage; got {accuracy!r}"
        assert any("non-finite predictions" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]

    def test_the_divergence_is_reachable_through_the_public_api_alone(self):
        """No hand-mutation of anything: only constructor arguments and
        update_adversary, the library's own documented training call."""
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss(n_groups=3, adversary_lr=1e14)
        y_pred, _, sens = self._three_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step_losses = [loss_fn.update_adversary(y_pred.detach(), sens) for _ in range(50)]
            accuracy = loss_fn.get_adversary_accuracy(y_pred, sens)
        assert step_losses[-1] != step_losses[-1], "the fixture must diverge"
        assert loss_fn._adversary_weights_not_finite() > 0
        # Before: 0.3333333432674808, chance for three groups, published as a
        # leakage measurement out of a network with no readable parameter, after
        # fifty consecutive nan steps without one warning.
        assert accuracy != accuracy, accuracy
        messages = [str(w.message) for w in caught]
        assert any("non-finite loss" in m for m in messages), messages
        assert any("non-finite predictions" in m for m in messages), messages

    def test_a_trained_but_wrong_adversary_keeps_its_real_number(self):
        """THE CONTROL that keeps this from over-accusing.

        adversary_lr=1e8 leaves every weight FINITE: the network is genuinely
        trained and genuinely wrong, and 0.0 is its real accuracy. That must stay
        a measurement, not become a refusal.
        """
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss(n_groups=3, adversary_lr=1e8)
        y_pred, _, sens = self._three_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(50):
                loss_fn.update_adversary(y_pred.detach(), sens)
            accuracy = loss_fn.get_adversary_accuracy(y_pred, sens)
        assert loss_fn._adversary_weights_not_finite() == 0
        assert accuracy == accuracy, "a finite adversary's accuracy IS measured"
        assert accuracy == pytest.approx(0.0)
        assert [str(w.message) for w in caught] == []

    @pytest.mark.parametrize("use_gradient_reversal", [True, False])
    def test_the_penalty_refuses_on_every_arm_when_the_adversary_is_broken(
        self, use_gradient_reversal
    ):
        """The sibling door: the PENALTY published the same NaN as assessed.

        The coverage helper runs above the reversal/alternating selection, so one
        guard covers both arms. Before: fairness_loss nan with
        fairness_penalty_assessed True and reason None, and the NaN went into
        total_loss.
        """
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss(
            n_groups=3,
            adversary_lr=1e14,
            lambda_fairness=0.5,
            use_gradient_reversal=use_gradient_reversal,
        )
        y_pred, y_true, sens = self._three_groups()
        y_pred = torch.sigmoid(y_pred)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(50):
                loss_fn.update_adversary(y_pred.detach(), sens)
            total, components = loss_fn(y_pred, y_true, sens, return_components=True)
        assert components.fairness_loss != components.fairness_loss
        assert components.batch_metrics["fairness_penalty_assessed"] is False
        assert components.batch_metrics["fairness_unassessable_reason"] == ("adversary_not_finite")
        assert components.batch_metrics["adversary_weights_not_finite"] > 0
        # The steering tensor stays finite: a NaN there poisons every parameter's
        # gradient at once and answers False to every later threshold.
        assert float(total.detach()) == float(total.detach())
        assert any("cannot predict anything" in str(w.message) for w in caught)

    @pytest.mark.parametrize("use_gradient_reversal", [True, False])
    def test_the_control_penalty_on_both_arms_is_a_real_number(self, use_gradient_reversal):
        """THE CONTROL: the documented signs, measured.

        The reversal arm returns the adversary's loss, the alternating arm the
        NEGATED loss, and both must still be reported as measured.
        """
        torch.manual_seed(0)
        loss_fn = AdversarialDebiasingLoss(
            n_groups=3,
            adversary_lr=0.01,
            lambda_fairness=0.5,
            use_gradient_reversal=use_gradient_reversal,
        )
        y_pred, y_true, sens = self._three_groups()
        y_pred = torch.sigmoid(y_pred)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(50):
                loss_fn.update_adversary(y_pred.detach(), sens)
            _, components = loss_fn(y_pred, y_true, sens, return_components=True)
        assert components.batch_metrics["fairness_penalty_assessed"] is True
        assert components.batch_metrics["adversary_weights_not_finite"] == 0
        assert components.fairness_loss == components.fairness_loss
        assert abs(components.fairness_loss) > 0.1, components.fairness_loss
        if use_gradient_reversal:
            assert components.fairness_loss > 0
        else:
            assert components.fairness_loss < 0
        assert [str(w.message) for w in caught] == []
