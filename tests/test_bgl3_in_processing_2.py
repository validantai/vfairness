"""BGL3 batch in_processing-2: does an in-processing mitigation refuse honestly?

Thirteen units across three files (the adversarial losses, the loss-function
base, the six fairness regularizers), each executed on input where the quantity
it reports genuinely does not exist. Every number in the docstrings below was
measured by running the unit BEFORE the fix in the same session, not read off
the source.

Eight defects were proved and fixed. The shape they share is the one this
library keeps finding: the value that reads as the clean answer, handed over
where nothing was measured.

    base.soft_rate_computation          an empty group mask returned 0.0, in
                                        silence, for every rate_type
    base.create_group_masks             a NaN attribute value became a GROUP
                                        with an all-False mask, and the row
                                        carrying it was in no mask at all
    base.compute_group_rates            which made that group's rate 0.0
    base.BaseFairnessLoss.forward       task_loss 0.0, the score a perfect
                                        predictor earns, from zero rows
    adversarial.FairRepresentationLoss  the same, in the one forward that
                     .forward           does not call the base one
    adversarial.AdversarialDebiasingLoss one NaN prediction wrote NaN into
                     .update_adversary   every adversary weight and the
    adversarial.ProjectedAdversarialLoss network then counted as TRAINED
                     .update_adversary
    adversarial.AdversarialDebiasingLoss accuracy 0.0 ("nothing leaks") from a
                     .get_adversary_     comparison that could not be true
                     accuracy
    regularizers.StatisticalParity       NaN dependence with measured=True
    regularizers.ConditionalIndependence a FINITE gap, byte identical to the
                                         healthy one, over rows it could not
                                         read
    regularizers.GroupFairness           the same, on the label axis its own
                                         guard did not cover
    regularizers.HilbertSchmidt          correct on every DATA fixture, and
                                         sigma=0.0 published a NaN as a
                                         measured dependence

CorrelationPenalty.forward was already correct on every fixture and is pinned
here so it stays that way, as are two refusals that already worked: the
untrained-adversary one inside FairRepresentationLoss.forward and the four
data-axis ones in HilbertSchmidtRegularizer.forward.

Every test here was sabotage-checked: the fix was re-broken, the test observed
going red, and the fix restored. Controls are marked CONTROL and exist because
a unit that refuses everything is as wrong as one that answers everything.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Tuple

import pytest

torch = pytest.importorskip("torch", reason="the in-processing mitigations need PyTorch")

from vfairness.in_processing.loss_functions.adversarial import (  # noqa: E402
    AdversarialDebiasingLoss,
    FairRepresentationLoss,
    ProjectedAdversarialLoss,
)
from vfairness.in_processing.loss_functions.base import (  # noqa: E402
    BaseFairnessLoss,
    compute_group_rates,
    create_group_masks,
    soft_rate_computation,
)
from vfairness.in_processing.regularizers.fairness_regularizers import (  # noqa: E402
    ConditionalIndependenceRegularizer,
    CorrelationPenalty,
    GroupFairnessRegularizer,
    HilbertSchmidtRegularizer,
    StatisticalParityRegularizer,
)

RATE_TYPES = ("positive_rate", "tpr", "fpr", "tnr", "fnr")


def _capture(fn, *args, **kwargs) -> Tuple[Any, List[str]]:
    """Run ``fn`` and return (result, every warning message it emitted)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _not_measured(messages: List[str]) -> List[str]:
    """The refusal messages among ``messages``.

    Two phrasings are in use: ``NOT MEASURED`` in the hand-written warnings and
    ``could not check`` in the one emitted by
    :func:`vfairness._not_assessed.warn_not_assessed`. Matching only the first
    made every regularizer test here fail against a refusal that was working,
    which is worth keeping visible in this helper.
    """
    return [m for m in messages if "NOT MEASURED" in m or "could not check" in m]


def _two_groups(n: int = 40):
    """A batch whose honest statistical parity gap is 0.30 and whose attribute
    is perfectly recoverable from the predictions."""
    sensitive = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    y_pred = torch.where(sensitive == 1, 0.8, 0.2)
    quarter = n // 4
    y_true = torch.cat(
        [torch.zeros(quarter), torch.ones(quarter), torch.zeros(quarter), torch.ones(quarter)]
    )
    return y_pred, y_true, sensitive


# ---------------------------------------------------------------------------
# base.soft_rate_computation
# ---------------------------------------------------------------------------


class TestAGroupWithNoRowsHasNoRate:
    """DEFECT. An all-False group mask returned ``tensor(0., requires_grad=True)``
    for EVERY rate_type with zero warnings, measured on a four-row batch. On
    two of the five scales 0.0 is the best possible score ("fpr" 0.0 is a group
    that never draws a false positive), and on "positive_rate" it is the
    extreme that manufactures a gap of 0.9 against a group sitting at 0.9.
    Now: 0.5, the finite stand-in the three sibling branches in the same
    function already use, plus a warning naming 0 of 4 rows."""

    y_pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])

    @pytest.mark.parametrize("rate_type", RATE_TYPES)
    def test_an_empty_group_mask_is_named_as_unmeasured(self, rate_type):
        value, messages = _capture(
            soft_rate_computation,
            self.y_pred,
            self.y_true,
            torch.zeros(4, dtype=torch.bool),
            rate_type=rate_type,
        )

        assert float(value) != 0.0, (
            "0.0 for a group with no rows is the clean end of the fpr and fnr "
            "scales and manufactures a gap on the others"
        )
        assert float(value) == 0.5
        named = _not_measured(messages)
        assert named, messages
        assert "0 of 4" in named[0], named[0]
        # the gradient must survive: this value feeds a differentiable penalty
        assert value.requires_grad

    @pytest.mark.parametrize(
        "rate_type,expected",
        [("positive_rate", 0.5), ("tpr", 0.85), ("fpr", 0.15)],
    )
    def test_control_a_populated_mask_measures_and_stays_silent(self, rate_type, expected):
        """CONTROL. The refusal must not spread to a group that HAS rows."""
        value, messages = _capture(
            soft_rate_computation,
            self.y_pred,
            self.y_true,
            torch.ones(4, dtype=torch.bool),
            rate_type=rate_type,
        )

        assert float(value) == pytest.approx(expected, abs=1e-6)
        assert not _not_measured(messages), messages


# ---------------------------------------------------------------------------
# base.create_group_masks and base.compute_group_rates
# ---------------------------------------------------------------------------


class TestAnUnreadableAttributeIsNotAGroup:
    """DEFECT. Measured on sensitive_attr [0.0, 1.0, nan, 1.0], in silence:

        create_group_masks  -> {0.0: [T,F,F,F], 1.0: [F,T,F,T], nan: [F,F,F,F]}
        compute_group_rates -> {0.0: 0.8999999761581421, 1.0: 0.5, nan: 0.0}

    torch.unique keeps NaN and NaN is not equal to itself, so one unreadable
    value became a third group holding no rows, whose "positive rate" was 0.0.
    That 0.0 goes straight into a demographic parity penalty. The row itself
    was in no mask while still counting toward the population mean it was
    compared against."""

    attr = torch.tensor([0.0, 1.0, float("nan"), 1.0])
    y_pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])

    def test_a_nan_attribute_value_is_not_returned_as_a_group(self):
        masks, messages = _capture(create_group_masks, self.attr)

        assert not any(isinstance(k, float) and math.isnan(k) for k in masks), masks
        assert sorted(masks) == [0.0, 1.0]
        assert all(bool(mask.any()) for mask in masks.values()), (
            "a mask with no rows is not a group"
        )
        named = [m for m in messages if "1 of 4 row(s)" in m]
        assert named, messages
        assert "NOT MEASURED" in named[0]

    def test_the_rate_of_a_group_with_no_rows_is_not_reported_as_zero(self):
        rates, messages = _capture(compute_group_rates, self.y_pred, self.y_true, self.attr)

        assert not any(isinstance(k, float) and math.isnan(k) for k in rates), rates
        assert 0.0 not in [float(v) for v in rates.values()]
        assert float(rates[0.0]) == pytest.approx(0.9, abs=1e-6)
        assert [m for m in messages if "1 of 4 row(s)" in m], messages

    def test_control_a_readable_attribute_is_measured_in_silence(self):
        """CONTROL. Two clean groups must produce two masks, two rates and no
        warning at all."""
        attr = torch.tensor([0.0, 0.0, 1.0, 1.0])
        masks, mask_messages = _capture(create_group_masks, attr)
        rates, rate_messages = _capture(compute_group_rates, self.y_pred, self.y_true, attr)

        assert sorted(masks) == [0.0, 1.0]
        assert not mask_messages, mask_messages
        assert float(rates[0.0]) == pytest.approx(0.85, abs=1e-6)
        assert float(rates[1.0]) == pytest.approx(0.15, abs=1e-6)
        assert not rate_messages, rate_messages


# ---------------------------------------------------------------------------
# base.BaseFairnessLoss.forward
# ---------------------------------------------------------------------------


class _ToyLoss(BaseFairnessLoss):
    """A direct subclass, which is what an external implementer writes. Every
    loss the package exports goes through _CoverageTrackingLoss instead, so
    this is the only way to execute the base forward."""

    def _compute_fairness_penalty(self, y_pred, y_true, sensitive_attr):
        return torch.tensor(0.25, requires_grad=True)


class TestATaskLossOverNoRowsIsNotAPerfectFit:
    """DEFECT. A task loss of 0.0 is the score a PERFECT predictor earns, and
    two inputs produced it from no rows at all. Measured on a four-row batch,
    both silent:

        sample_weight=torch.zeros(4) -> task_loss 0.0,  total_loss 0.025
        reduction="sum", empty batch -> task_loss 0.0,  total_loss 0.025

    and three weighted-out batches in a row gave end_epoch an avg_task_loss of
    0.0: a completed epoch over nothing. Now task_loss is NaN with the finite
    value kept in batch_metrics['task_loss_unassessed_value'] and a warning
    naming how many rows entered."""

    def _components(self, loss, y_pred, y_true, sensitive, **kwargs):
        (_total, components), messages = _capture(
            loss.forward, y_pred, y_true, sensitive, return_components=True, **kwargs
        )
        return components, messages

    def test_every_sample_weight_zero_is_not_a_measured_task_loss(self):
        y_pred, y_true, sensitive = _two_groups(4)
        y_pred = y_pred.clone().requires_grad_(True)

        components, messages = self._components(
            _ToyLoss(), y_pred, y_true, sensitive, sample_weight=torch.zeros(4)
        )

        assert math.isnan(components.task_loss)
        assert components.batch_metrics["task_loss_assessed"] is False
        assert components.batch_metrics["task_rows_used"] == 0
        assert components.batch_metrics["task_rows_total"] == 4
        assert components.batch_metrics["task_loss_unassessed_value"] == 0.0
        assert [m for m in messages if "0 of 4 row(s) entered the task loss" in m], messages
        # the optimisation step really taken is still described truthfully
        assert math.isfinite(components.total_loss)

    def test_a_summed_empty_batch_is_not_a_measured_task_loss(self):
        """reduction="sum" reduces an empty tensor to 0.0 rather than NaN, so
        this is the one empty-batch path that had no tell at all."""
        empty = torch.zeros(0, requires_grad=True)

        components, messages = self._components(
            _ToyLoss(reduction="sum"), empty, torch.zeros(0), torch.zeros(0)
        )

        assert math.isnan(components.task_loss)
        assert components.batch_metrics["task_loss_assessed"] is False
        assert components.batch_metrics["task_rows_total"] == 0
        assert [m for m in messages if "0 of 0 row(s) entered the task loss" in m], messages

    def test_an_epoch_of_unusable_batches_does_not_average_to_a_perfect_fit(self):
        y_pred, y_true, sensitive = _two_groups(4)
        loss = _ToyLoss(reduction="sum")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                loss.forward(y_pred, y_true, sensitive, sample_weight=torch.zeros(4))
            metrics = loss.end_epoch()

        assert metrics is not None
        assert math.isnan(metrics.avg_task_loss), (
            "0.0 averaged over three batches that read no rows is a completed epoch over nothing"
        )

    @pytest.mark.parametrize("weights", [None, torch.ones(4), torch.tensor([1.0, 0.0, 1.0, 1.0])])
    def test_control_rows_that_do_enter_are_measured_and_silent(self, weights):
        """CONTROL. Unweighted, fully weighted and PARTIALLY weighted batches
        must all still report a measured task loss: refusing a batch because
        one row was weighted out would neuter every weighted training loop."""
        y_pred, y_true, sensitive = _two_groups(4)

        components, messages = self._components(
            _ToyLoss(), y_pred, y_true, sensitive, sample_weight=weights
        )

        assert math.isfinite(components.task_loss)
        assert components.task_loss > 0.0
        assert components.batch_metrics["task_loss_assessed"] is True
        assert not [m for m in messages if "entered the task loss" in m], messages


# ---------------------------------------------------------------------------
# adversarial.AdversarialDebiasingLoss.get_adversary_accuracy
# ---------------------------------------------------------------------------


def _train_adversary(loss_fn, y_pred, sensitive, steps=50):
    for _ in range(steps):
        loss_fn.update_adversary(y_pred, sensitive)
    assert loss_fn._adversary_is_at_initialization() is False, "the fixture did not train it"


class TestAnAccuracyThatCouldNotBeTrueIsNotZeroLeakage:
    """DEFECT. get_adversary_accuracy compares predicted class INDICES with the
    attribute, so against an attribute that is not on the integer label grid
    the equality is False for every row whatever the adversary learned.

    Measured with n_groups=2, an attribute of 0.3/0.7 (inside the label domain,
    so the existing domain check let it through) and 50 training steps on a
    perfectly separating batch: accuracy 0.0 with zero warnings, where the SAME
    fixture with 0.0/1.0 labels scores 1.0. This method's own docstring says
    "Higher accuracy means more sensitive information is leaked", so 0.0 is the
    strongest all-clear on the scale, reported from a fully recovered
    attribute."""

    @staticmethod
    def _fixture(values):
        n = 40
        sensitive = torch.cat([torch.full((n // 2,), values[0]), torch.full((n // 2,), values[1])])
        y_pred = torch.where(sensitive == values[1], 0.9, 0.1)
        return y_pred, sensitive

    def test_a_non_integer_attribute_is_refused_rather_than_scored_zero(self):
        torch.manual_seed(0)
        y_pred, sensitive = self._fixture((0.3, 0.7))
        loss_fn = AdversarialDebiasingLoss()
        _train_adversary(loss_fn, y_pred, sensitive)

        accuracy, messages = _capture(loss_fn.get_adversary_accuracy, y_pred, sensitive)

        assert math.isnan(accuracy), accuracy
        named = _not_measured(messages)
        assert named, messages
        assert "integer class labels" in named[0]

    def test_control_the_same_fixture_with_integer_labels_measures_the_leak(self):
        """CONTROL, and the proof that the refused case was measurable data:
        identical predictions, labels on the grid, accuracy 1.0."""
        torch.manual_seed(0)
        y_pred, sensitive = self._fixture((0.0, 1.0))
        loss_fn = AdversarialDebiasingLoss()
        _train_adversary(loss_fn, y_pred, sensitive)

        accuracy, messages = _capture(loss_fn.get_adversary_accuracy, y_pred, sensitive)

        assert accuracy == pytest.approx(1.0)
        assert not _not_measured(messages), messages


# ---------------------------------------------------------------------------
# adversarial.*.update_adversary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [AdversarialDebiasingLoss, ProjectedAdversarialLoss])
class TestATrainingStepOnUnreadableRowsIsRefused:
    """DEFECT. Measured on AdversarialDebiasingLoss(n_groups=3) with one NaN
    prediction in 30 rows (the multiclass path: cross_entropy does not validate
    its input, unlike the binary path where torch itself raises), with no
    warning anywhere:

        update_adversary          -> returned nan, EVERY adversary weight NaN
        _adversary_is_at_initialization() -> False, so the destroyed network
                                             counted as TRAINED
        the next forward on a HEALTHY batch -> fairness_loss nan with
            fairness_penalty_assessed=True and reason None
        get_adversary_accuracy    -> 0.3333333432674408, because argmax of an
                                     all-NaN row is 0

    A permanently broken adversary, credited with training, publishing a NaN as
    a measured penalty. The step is now refused before it can run."""

    @staticmethod
    def _batch(n=30):
        sensitive = torch.cat([torch.zeros(n // 3), torch.ones(n // 3), torch.full((n // 3,), 2.0)])
        y_pred = sensitive / 2.0
        return y_pred, sensitive

    def test_a_non_finite_prediction_never_reaches_the_optimizer(self, cls):
        loss_fn = cls(n_groups=3)
        y_pred, sensitive = self._batch()
        poisoned = y_pred.clone()
        poisoned[0] = float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            loss_fn.update_adversary(poisoned, sensitive)

        assert all(bool(torch.isfinite(p).all()) for p in loss_fn.adversary.parameters())
        assert loss_fn._adversary_is_at_initialization() is True

    def test_a_non_finite_attribute_never_reaches_the_optimizer(self, cls):
        loss_fn = cls(n_groups=3)
        y_pred, sensitive = self._batch()
        poisoned = sensitive.clone()
        poisoned[0] = float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            loss_fn.update_adversary(y_pred, poisoned)

        assert loss_fn._adversary_is_at_initialization() is True

    def test_an_out_of_domain_label_is_refused_at_the_training_entry_point(self, cls):
        """The binary path's BCE does NOT validate its TARGET, so a label the
        adversary has no output for used to train the network toward it and the
        error only surfaced at the next forward."""
        loss_fn = cls(n_groups=2)
        y_pred = torch.rand(20)

        with pytest.raises(ValueError, match="label domain"):
            loss_fn.update_adversary(y_pred, torch.full((20,), 7.0))

        assert loss_fn._adversary_is_at_initialization() is True

    def test_a_batch_with_no_rows_says_so_instead_of_returning_a_bare_nan(self, cls):
        """The value was already NaN, which is the honest state; what was
        missing was any statement of why. 0.0 here would be the score a perfect
        adversary earns."""
        loss_fn = cls()

        returned, messages = _capture(loss_fn.update_adversary, torch.zeros(0), torch.zeros(0))

        assert math.isnan(returned)
        named = _not_measured(messages)
        assert named, messages
        assert "0 rows" in named[0]
        assert loss_fn._adversary_is_at_initialization() is True

    def test_control_a_readable_batch_trains_and_returns_its_loss(self, cls):
        """CONTROL. The guard must not refuse ordinary training: one step on a
        clean batch moves the weights and returns a finite adversary loss."""
        loss_fn = cls(n_groups=3)
        y_pred, sensitive = self._batch()

        returned, messages = _capture(loss_fn.update_adversary, y_pred, sensitive)

        assert math.isfinite(returned) and returned > 0.0
        assert loss_fn._adversary_is_at_initialization() is False
        assert not messages, messages


# ---------------------------------------------------------------------------
# adversarial.FairRepresentationLoss.forward
# ---------------------------------------------------------------------------


class TestFairRepresentationForwardReportsWhatItRead:
    """DEFECT plus two correct refusals. This forward builds its own
    LossComponents rather than calling BaseFairnessLoss.forward, so the
    task-coverage disclosure had to be taken here as well.

    Measured with every sample_weight zero and a trained adversary: task_loss
    0.0 (a perfect fit over zero rows) with no warning, next to a correctly
    measured fairness term. With reduction="sum" on an empty batch: task_loss
    0.0 AND total_loss 0.0.

    Already correct, and pinned so it stays: an adversary still at its
    construction weights gives fairness_loss NaN with reason
    'adversary_at_initialization' and the finite 0.6981682777404785 the
    optimizer saw kept in batch_metrics."""

    @staticmethod
    def _trained(loss_fn, representation, sensitive, steps=60):
        optimizer = torch.optim.Adam(loss_fn.adversary.parameters(), lr=0.01)
        for _ in range(steps):
            optimizer.zero_grad()
            predicted = loss_fn.adversary(representation.detach())
            loss = torch.nn.functional.binary_cross_entropy(predicted.squeeze(), sensitive.float())
            loss.backward()
            optimizer.step()

    def test_a_fully_weighted_out_batch_is_not_a_perfect_fit(self):
        loss_fn = FairRepresentationLoss()
        y_pred, y_true, sensitive = _two_groups()
        representation = torch.where(sensitive == 1, 0.8, 0.2).unsqueeze(1).requires_grad_(True)
        self._trained(loss_fn, representation, sensitive)

        (_total, components), messages = _capture(
            loss_fn.forward,
            y_pred,
            y_true,
            sensitive,
            representation=representation,
            sample_weight=torch.zeros(40),
            return_components=True,
        )

        assert math.isnan(components.task_loss)
        assert components.batch_metrics["task_loss_assessed"] is False
        assert components.batch_metrics["task_rows_used"] == 0
        assert components.batch_metrics["task_loss_unassessed_value"] == 0.0
        assert [m for m in messages if "0 of 40 row(s) entered the task loss" in m], messages
        # the fairness half was measurable here and must stay measured
        assert components.batch_metrics["fairness_penalty_assessed"] is True

    def test_an_untrained_adversary_is_still_refused_and_the_task_half_is_not(self):
        """CORRECT ALREADY. Both halves report independently: the penalty is
        withheld, the task loss is a real measurement of 40 rows."""
        loss_fn = FairRepresentationLoss()
        y_pred, y_true, sensitive = _two_groups()

        (_total, components), messages = _capture(
            loss_fn.forward,
            y_pred,
            y_true,
            sensitive,
            representation=y_pred.detach().unsqueeze(1),
            return_components=True,
        )

        assert math.isnan(components.fairness_loss)
        assert components.batch_metrics["fairness_unassessable_reason"] == (
            "adversary_at_initialization"
        )
        assert components.batch_metrics["fairness_loss_unassessed_value"] > 0.0
        assert math.isfinite(components.task_loss)
        assert components.batch_metrics["task_loss_assessed"] is True
        assert _not_measured(messages), messages

    def test_control_a_full_batch_measures_every_term(self):
        """CONTROL. Representation, reconstruction pair, trained adversary and
        forty readable rows: three measured terms and no refusal."""
        loss_fn = FairRepresentationLoss()
        y_pred, y_true, sensitive = _two_groups()
        representation = torch.where(sensitive == 1, 0.8, 0.2).unsqueeze(1).requires_grad_(True)
        self._trained(loss_fn, representation, sensitive)
        original = torch.rand(40, 3)

        (_total, components), messages = _capture(
            loss_fn.forward,
            y_pred,
            y_true,
            sensitive,
            representation=representation,
            x_reconstructed=original + 0.01,
            x_original=original,
            return_components=True,
        )

        assert math.isfinite(components.task_loss)
        assert math.isfinite(components.fairness_loss)
        assert math.isfinite(components.regularization_loss)
        assert components.batch_metrics["task_loss_assessed"] is True
        assert components.batch_metrics["fairness_penalty_assessed"] is True
        assert components.batch_metrics["reconstruction_assessed"] is True
        assert not messages, messages


# ---------------------------------------------------------------------------
# regularizers: the row axis and the label axis
# ---------------------------------------------------------------------------


def _regularize(regularizer, y_pred, sensitive, y_true=None):
    (penalty, metrics), messages = _capture(
        regularizer, y_pred, sensitive, y_true, return_metrics=True
    )
    return penalty, metrics, messages


class TestStatisticalParityRefusesRowsItCannotRead:
    """DEFECT. Measured on 40 rows and two groups whose honest gap is 0.30,
    every case with measured=True and zero warnings:

        one NaN prediction    -> dependence_measure nan, group_penalties
                                 {0.0: nan, 1.0: nan}
        one inf prediction    -> nan, with {0.0: nan, 1.0: inf}
        one NaN attribute row -> nan, and a THIRD fabricated key:
                                 {0.0: 0.3, 1.0: 0.3, nan: nan}

    A NaN with measured=True contradicts this module's own contract, so a
    consumer filtering a run history on `measured` keeps the NaN and the
    aggregate it poisons."""

    @pytest.mark.parametrize("axis", ["prediction", "attribute"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_an_unreadable_row_is_not_a_measured_gap(self, axis, bad):
        y_pred, _y_true, sensitive = _two_groups()
        if axis == "prediction":
            y_pred = y_pred.clone()
            y_pred[0] = bad
        else:
            sensitive = sensitive.clone()
            sensitive[0] = bad

        _penalty, metrics, messages = _regularize(
            StatisticalParityRegularizer(strength=0.1), y_pred, sensitive
        )

        assert metrics.measured is False
        assert math.isnan(metrics.dependence_measure)
        assert metrics.metadata["not_assessed"] == "non_finite_rows"
        assert metrics.metadata["n_usable"] == 39
        assert metrics.group_penalties == {}
        assert _not_measured(messages), messages

    def test_control_a_readable_batch_still_measures_the_gap(self):
        """CONTROL. The honest 0.30 must survive the guard."""
        y_pred, _y_true, sensitive = _two_groups()

        _penalty, metrics, messages = _regularize(
            StatisticalParityRegularizer(strength=0.1), y_pred, sensitive
        )

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(0.3, abs=1e-5)
        assert not _not_measured(messages), messages


class TestConditionalIndependenceRefusesRowsItCannotRead:
    """DEFECT, and the one that produced a FINITE clean number rather than a
    NaN. Measured on 40 rows, two groups, two labels, honest dependence 0.30,
    all with measured=True, metadata {} and zero warnings:

        one NaN in the ATTRIBUTE -> dependence_measure 0.30000001192092896,
            BYTE IDENTICAL to the healthy answer. The row matched no group
            (NaN != NaN) so it entered no cell, yet it stayed inside the
            stratum mean every cell is compared against.
        one NaN in y_true        -> 0.30000001192092896 again, with the cells
            shifted to 0.31578952 / 0.28421050: a row nobody could read moved
            the number and nothing said so.
        one NaN prediction       -> nan with measured=True.
    """

    @pytest.mark.parametrize("axis", ["prediction", "attribute", "label"])
    def test_one_unreadable_row_is_not_a_measured_dependence(self, axis):
        y_pred, y_true, sensitive = _two_groups()
        if axis == "prediction":
            y_pred = y_pred.clone()
            y_pred[0] = float("nan")
        elif axis == "attribute":
            sensitive = sensitive.clone()
            sensitive[0] = float("nan")
        else:
            y_true = y_true.clone()
            y_true[0] = float("nan")

        _penalty, metrics, messages = _regularize(
            ConditionalIndependenceRegularizer(strength=0.1), y_pred, sensitive, y_true
        )

        assert metrics.measured is False
        assert math.isnan(metrics.dependence_measure)
        assert metrics.metadata["not_assessed"] == "non_finite_rows"
        assert metrics.metadata["n_usable"] == 39
        assert _not_measured(messages), messages

    def test_a_label_column_nothing_can_be_read_from_keeps_its_own_reason(self):
        """The new guard defers to the more specific branch below it: with a
        label column of pure NaN, two groups are present and no (group, label)
        cell has any rows, which is named 'no_populated_group_label_cell' with
        the cell count. Pinned here so the deferral is deliberate and stays a
        refusal either way."""
        torch.manual_seed(0)
        y_pred = torch.rand(200, requires_grad=True)
        sensitive = (torch.rand(200) > 0.5).long()

        _penalty, metrics, messages = _regularize(
            ConditionalIndependenceRegularizer(strength=1.0),
            y_pred,
            sensitive,
            torch.full((200,), float("nan")),
        )

        assert metrics.measured is False
        assert math.isnan(metrics.dependence_measure)
        assert metrics.metadata["not_assessed"] == "no_populated_group_label_cell"
        assert _not_measured(messages), messages

    def test_control_a_readable_batch_still_measures_the_dependence(self):
        """CONTROL. Same fixture, no unreadable row: the 0.30 the defective
        runs were imitating."""
        y_pred, y_true, sensitive = _two_groups()

        _penalty, metrics, messages = _regularize(
            ConditionalIndependenceRegularizer(strength=0.1), y_pred, sensitive, y_true
        )

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(0.3, abs=1e-5)
        assert not _not_measured(messages), messages


class TestGroupFairnessRefusesAnUnreadableLabel:
    """DEFECT on the third axis. The existing guard counted rows finite in
    y_pred and in the attribute, and four of the five metrics also condition on
    y_true. Measured with one NaN in y_true, 40 rows, two groups, all with
    measured=True and zero warnings:

        'eo'  -> dependence_measure 0.30000001192092896, the healthy value,
                 with the fpr arm silently shifted to 0.31578952 / 0.28421050
        'fpr' -> the same silent shift
        'eop' -> 0.30000001192092896
        'ppv' -> nan with measured=True, because overall_ppv multiplies y_true
                 straight into its numerator
    """

    @pytest.mark.parametrize("metric", ["eo", "eop", "fpr", "ppv"])
    def test_an_unreadable_label_is_not_folded_into_a_rate(self, metric):
        y_pred, y_true, sensitive = _two_groups()
        y_true = y_true.clone()
        y_true[0] = float("nan")

        _penalty, metrics, messages = _regularize(
            GroupFairnessRegularizer(strength=0.1, fairness_metric=metric),
            y_pred,
            sensitive,
            y_true,
        )

        assert metrics.measured is False
        assert math.isnan(metrics.dependence_measure)
        assert metrics.metadata["not_assessed"] == "non_finite_rows"
        assert metrics.metadata["n_usable"] == 39
        assert _not_measured(messages), messages

    def test_the_metric_that_ignores_the_label_is_not_refused_over_it(self):
        """OVER-CORRECTION CONTROL, and the rule it encodes: a unit may only be
        judged on input it consumes. 'dp' never reads y_true, so an unreadable
        label must NOT stop it measuring."""
        y_pred, y_true, sensitive = _two_groups()
        y_true = y_true.clone()
        y_true[0] = float("nan")

        _penalty, metrics, messages = _regularize(
            GroupFairnessRegularizer(strength=0.1, fairness_metric="dp"),
            y_pred,
            sensitive,
            y_true,
        )

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(0.3, abs=1e-5)
        assert not _not_measured(messages), messages

    @pytest.mark.parametrize("metric", ["eo", "eop", "fpr"])
    def test_control_a_readable_batch_still_measures_each_metric(self, metric):
        """CONTROL."""
        y_pred, y_true, sensitive = _two_groups()

        _penalty, metrics, messages = _regularize(
            GroupFairnessRegularizer(strength=0.1, fairness_metric=metric),
            y_pred,
            sensitive,
            y_true,
        )

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(0.3, abs=1e-5)
        assert not _not_measured(messages), messages


class TestTheTwoRegularizersThatWereAlreadyHonest:
    """CORRECT, pinned so they stay. Both were executed on the full set of
    degeneracies and refused every one with dependence_measure NaN,
    measured=False and a warning, while measuring the healthy batch:
    HSIC 0.017045702785253525 and a Pearson correlation of 1.0 on the
    perfectly leaking fixture."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: HilbertSchmidtRegularizer(strength=0.1),
            lambda: CorrelationPenalty(strength=0.1),
        ],
    )
    @pytest.mark.parametrize(
        "case",
        ["non_finite_row", "single_group", "constant_predictions", "one_row"],
    )
    def test_each_degeneracy_is_refused_not_scored_zero(self, make, case):
        y_pred, _y_true, sensitive = _two_groups()
        if case == "non_finite_row":
            y_pred = y_pred.clone()
            y_pred[0] = float("nan")
        elif case == "single_group":
            sensitive = torch.zeros(40)
        elif case == "constant_predictions":
            y_pred = torch.full((40,), 0.7)
        else:
            y_pred, sensitive = y_pred[:1], sensitive[:1]

        _penalty, metrics, messages = _regularize(make(), y_pred, sensitive)

        assert metrics.measured is False
        assert math.isnan(metrics.dependence_measure), (
            "0.0 on a dependence scale is proven independence, the clean end"
        )
        assert metrics.metadata["not_assessed"]
        assert _not_measured(messages), messages

    @pytest.mark.parametrize(
        "make,expected",
        [
            (lambda: HilbertSchmidtRegularizer(strength=0.1), 0.017045702785253525),
            (lambda: CorrelationPenalty(strength=0.1), 1.0),
        ],
    )
    def test_control_the_healthy_batch_is_measured(self, make, expected):
        """CONTROL."""
        y_pred, _y_true, sensitive = _two_groups()

        _penalty, metrics, messages = _regularize(make(), y_pred, sensitive)

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(expected, abs=1e-6)
        assert not _not_measured(messages), messages

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
    def test_hsic_refuses_a_bandwidth_that_has_no_kernel(self, sigma):
        """DEFECT on the PARAMETER axis rather than the data axis, which is why
        it outlived the degeneracy guards above. The RBF kernel divides by
        2 * sigma ** 2. Measured on 40 readable rows and two groups, where
        sigma=1.0 reports 0.018390340730547905:

            sigma=0.0  -> dependence_measure nan with measured=True and zero
                          warnings, the one shape this module's contract says
                          must never be reported
            sigma=-1.0 -> 0.018390340730547905, BYTE IDENTICAL to sigma=1.0,
                          since only sigma ** 2 is read

        Refused at construction, as drift.detect_drift_mmd already refuses the
        same parameter for the same kernel."""
        with pytest.raises(ValueError, match="finite positive kernel bandwidth"):
            HilbertSchmidtRegularizer(strength=0.1, sigma=sigma)

    def test_control_a_usable_bandwidth_is_accepted_and_changes_the_statistic(self):
        """CONTROL. The guard must refuse only the unusable values, and sigma
        must still DO something: two positive bandwidths give two different
        HSIC values on the same batch."""
        y_pred, _y_true, sensitive = _two_groups()

        wide = HilbertSchmidtRegularizer(strength=0.1, sigma=1.0)
        narrow = HilbertSchmidtRegularizer(strength=0.1, sigma=0.25)
        _p, wide_metrics, _w = _regularize(wide, y_pred, sensitive)
        _p, narrow_metrics, _n = _regularize(narrow, y_pred, sensitive)

        assert wide_metrics.measured is True and narrow_metrics.measured is True
        assert math.isfinite(wide_metrics.dependence_measure)
        assert wide_metrics.dependence_measure != narrow_metrics.dependence_measure

    def test_control_neither_is_refused_over_a_label_it_never_reads(self):
        """OVER-CORRECTION CONTROL. Both take y_true in their signature and
        neither reads it, so an unreadable label must not refuse them."""
        y_pred, y_true, sensitive = _two_groups()
        y_true = y_true.clone()
        y_true[0] = float("nan")

        for make in (
            lambda: HilbertSchmidtRegularizer(strength=0.1),
            lambda: CorrelationPenalty(strength=0.1),
        ):
            _penalty, metrics, messages = _regularize(make(), y_pred, sensitive, y_true)
            assert metrics.measured is True
            assert not _not_measured(messages), messages
