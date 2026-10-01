"""BGL3 batch in_processing-3: does an in-processing mitigation refuse honestly?

Ten units across four files (the group calibrators, the constraint base, the
counterfactual losses, the sklearn wrappers), each executed on input where the
quantity it reports genuinely does not exist. Every number in the docstrings
below was measured by RUNNING the unit before the fix in the same session, not
read off the source.

Eight defects were proved and fixed:

    TrainableGroupCalibrator            60 NaN logits in two groups of 30
                 .calibration_loss     returned tensor(0.), PERFECT
                                       calibration, in silence; one NaN row out
                                       of 60 returned 0.0091 against a true
                                       0.0247, because that group's whole error
                                       was replaced by exactly 0.0
    CalibrationAwareTrainer.train_step  reported calibration_loss 0.0 for a
                                       model that had diverged to NaN
    CalibrationAwareTrainer             returned 0.0 as the mean calibration
                 .fine_tune_calibration loss over one batch it could not
                                       measure, having updated no parameter
    BoundedGroupLossConstraint          a measured 0.36 breach became
                 .compute_violation    overall_violation 0.0, is_satisfied
                                       True, could_not_evaluate False, no
                                       warning, once the labels were blanked
    BoundedGroupLossConstraint          answered 0.0, no violation, for the
                 .signed_constraint_value group it was 80% wrong about
    CounterfactualFairnessLoss.forward  a supplied counterfactual identical to
                                       the factual scored 0.0 with
                                       fairness_penalty_assessed True, while
                                       the generated arm refuses that exact
                                       vacuity
    FairClassifier.predict              wrote three unscored rows as confident
                                       REJECTIONS, while the sibling method
                                       predict_with_sensitive_attr refuses the
                                       identical input
    FairRegressor.fit                   silently dropped 120 sample weights for
                                       a base estimator that takes none, then
                                       measured the mean-parity offsets off the
                                       unweighted model

Two units were already correct and are pinned here so they stay that way:
DemographicParityConstraint.compute_violation and IndividualFairnessLoss
.forward.

Every test here was sabotage-checked: the fix was re-broken, the test observed
going red, and the fix restored. Controls are marked CONTROL and exist because a
unit that refuses everything is as wrong as one that answers everything.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from vfairness.in_processing.constraints.base import (
    BoundedGroupLossConstraint,
    DemographicParityConstraint,
)
from vfairness.in_processing.wrappers.sklearn_wrappers import FairClassifier, FairRegressor

torch = pytest.importorskip("torch", reason="the trainable calibrators need PyTorch")

import torch.nn as nn  # noqa: E402

from vfairness.in_processing.calibrators.group_calibrators import (  # noqa: E402
    CalibrationAwareTrainer,
    TrainableGroupCalibrator,
)
from vfairness.in_processing.loss_functions.counterfactual import (  # noqa: E402
    CounterfactualFairnessLoss,
    IndividualFairnessLoss,
)

# ===========================================================================
# Fixtures
# ===========================================================================


def _calibration_batch(
    n: int = 60, seed: int = 0
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """60 rows in two groups of 30, comfortably above min_group_size=10."""
    torch.manual_seed(seed)
    logits = torch.randn(n)
    y = (torch.rand(n) > 0.5).float()
    gids = torch.tensor([0] * (n // 2) + [1] * (n - n // 2))
    return logits, y, gids


class _Diverged(nn.Module):
    """A model whose forward has gone to NaN, the ordinary way a training run
    stops being measurable partway through."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.lin(x).squeeze(-1) * float("nan")


class _Healthy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.lin(x).squeeze(-1)


def _trainer(model_cls: Any, seed: int = 1) -> Tuple[CalibrationAwareTrainer, Any, Any]:
    torch.manual_seed(seed)
    model = model_cls()
    calibrator = TrainableGroupCalibrator(n_groups=2)
    trainer = CalibrationAwareTrainer(model, calibrator, calibration_epochs=1)
    opt = torch.optim.SGD(list(model.parameters()) + list(calibrator.parameters()), lr=0.01)
    return trainer, opt, nn.BCEWithLogitsLoss()


def _bgl_fixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A model right on every row of b and wrong on 24 of the 30 rows of a.

    Group losses 0.8 and 0.0, so the relative bound (1 + 0.1) * 0.4 = 0.44 is
    breached by 0.36. This is the disparity the label blanking erases.
    """
    groups = np.array(["a"] * 30 + ["b"] * 30)
    y_true = np.ones(60, dtype=int)
    y_pred = np.concatenate([np.array([0.1] * 24 + [0.9] * 6), np.full(30, 0.9)])
    return y_pred, y_true, groups


def _messages(caught: List[warnings.WarningMessage]) -> List[str]:
    return [str(w.message) for w in caught]


# ===========================================================================
# 1. TrainableGroupCalibrator.calibration_loss
# ===========================================================================


class TestCalibrationLossDoesNotScoreAnUnscorableBatch:
    def test_a_batch_of_nan_logits_is_refused_not_scored_zero(self):
        """MEASURED BEFORE THE FIX: 60 NaN logits in two groups of 30, with
        min_group_size=10, returned ``tensor(0.)`` and NOT ONE WARNING. On this
        scale 0.0 is perfect calibration, so a model that had diverged to NaN
        earned the best calibration score the metric can give. The mechanism:
        every soft bin is entered only when ``in_bin.sum() > 1e-8``, which is
        False for NaN, so the group skipped every bin, kept its seed of exactly
        0.0 and was still counted as measured."""
        cal = TrainableGroupCalibrator(n_groups=2)
        _, y, gids = _calibration_batch()
        nan_logits = torch.full((60,), float("nan"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(nan_logits, y, gids)

        assert torch.isnan(out).item(), f"a refusal must not be a number, got {out}"
        msgs = _messages(caught)
        assert any("60 of 60 row(s) carry no usable number" in m for m in msgs), msgs
        assert any("no calibration error was measured" in m for m in msgs), msgs

    def test_a_batch_of_nan_labels_is_refused_too(self):
        """MEASURED BEFORE THE FIX: 60 finite logits with every LABEL NaN
        returned ``tensor(nan)`` already, by arithmetic accident (bin_acc went
        NaN), but in silence. The refusal is now deliberate and named, so it
        survives a change to the binning."""
        cal = TrainableGroupCalibrator(n_groups=2)
        logits, _, gids = _calibration_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(logits, torch.full((60,), float("nan")), gids)

        assert torch.isnan(out).item()
        assert any("carry no usable number" in m for m in _messages(caught))

    def test_one_unscored_row_does_not_zero_its_whole_group(self):
        """MEASURED BEFORE THE FIX, the quiet half of the same defect. One NaN
        logit out of 60 returned 0.0091 where the clean batch measures 0.0247,
        because group 0's real error was replaced by exactly 0.0 and averaged
        in beside group 1's 0.0182 (0.0182 / 2 = 0.0091). One unscored row out
        of sixty cut the reported calibration error by 63%, silently.

        AFTER: 0.0236, the honest measurement over the 59 usable rows, plus a
        warning naming the excluded one."""
        cal = TrainableGroupCalibrator(n_groups=2)
        logits, y, gids = _calibration_batch()
        clean = float(cal.calibration_loss(logits, y, gids).item())

        partial = logits.clone()
        partial[0] = float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = float(cal.calibration_loss(partial, y, gids).item())

        assert any("1 of 60 row(s) carry no usable number" in m for m in _messages(caught))
        # The group-zeroing value was 0.0091 and the true one 0.0247; land near
        # the truth, not near the fabrication.
        assert abs(out - clean) < 0.2 * clean, (out, clean)
        assert out > 0.02, out

    def test_control_a_clean_batch_still_measures_and_stays_silent(self):
        """CONTROL. The refusal must not eat the measurement: the same 60 rows
        with no NaN measure 0.0247 with zero warnings, before and after."""
        cal = TrainableGroupCalibrator(n_groups=2)
        logits, y, gids = _calibration_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(logits, y, gids)

        assert _messages(caught) == []
        assert float(out.item()) == pytest.approx(0.0247, abs=5e-4)

    def test_control_the_loss_still_carries_a_gradient(self):
        """CONTROL. Masking rows must not detach the graph: the loss still
        moves the calibration parameters."""
        cal = TrainableGroupCalibrator(n_groups=2, method="temperature")
        logits, y, gids = _calibration_batch()
        before = cal.calibrator.log_temperatures.detach().clone()
        opt = torch.optim.SGD(cal.parameters(), lr=1.0)
        opt.zero_grad()
        cal.calibration_loss(cal(logits, gids), y, gids).backward()
        opt.step()
        after = cal.calibrator.log_temperatures.detach()
        assert not torch.equal(before, after), "the loss carried no gradient"


# ===========================================================================
# 2. CalibrationAwareTrainer.train_step
# ===========================================================================


class TestTrainStepDoesNotReportPerfectCalibrationForADivergedModel:
    def test_a_diverged_model_reports_nan_calibration_loss(self):
        """MEASURED BEFORE THE FIX: with a model whose forward returns NaN,
        ``train_step`` returned
        ``{'total_loss': nan, 'task_loss': nan, 'calibration_loss': 0.0}``.
        The two NaNs are loud; the 0.0 is the fabrication, and it is the field
        a training log plots as the calibration curve."""
        trainer, opt, loss_fn = _trainer(_Diverged)
        x = torch.randn(60, 4)
        _, y, gids = _calibration_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = trainer.train_step(x, y, gids, opt, loss_fn)

        assert np.isnan(stats["calibration_loss"]), stats
        assert any("carry no usable number" in m for m in _messages(caught))

    def test_control_a_healthy_model_still_reports_a_measured_number(self):
        """CONTROL. Same seed and batch with a finite model:
        calibration_loss 0.013412627391517162, before and after."""
        trainer, opt, loss_fn = _trainer(_Healthy)
        x = torch.randn(60, 4)
        _, y, gids = _calibration_batch()
        stats = trainer.train_step(x, y, gids, opt, loss_fn)
        assert np.isfinite(stats["calibration_loss"]), stats
        assert stats["calibration_loss"] > 0.0


# ===========================================================================
# 3. CalibrationAwareTrainer.fine_tune_calibration
# ===========================================================================


class _Loader:
    def __init__(self, batches: List[Tuple[Any, Any, Any]]) -> None:
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class TestFineTuneCalibrationDoesNotCompleteAFitItCouldNotRun:
    def test_a_loader_of_unusable_batches_returns_nan_and_says_nothing_moved(self):
        """MEASURED BEFORE THE FIX: one batch through a diverged model returned
        ``0.0`` as the mean calibration loss, with ZERO warnings, and the
        calibrator's log_temperatures were still [0.0, 0.0] afterwards. A
        completed fine-tune reporting perfect calibration, having updated no
        parameter and measured no batch."""
        trainer, _, _ = _trainer(_Diverged)
        copt = torch.optim.SGD(trainer.calibrator.parameters(), lr=0.5)
        before = trainer.calibrator.calibrator.log_temperatures.detach().clone()
        x = torch.randn(60, 4)
        _, y, gids = _calibration_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = trainer.fine_tune_calibration(_Loader([(x, y, gids)]), copt)

        assert np.isnan(out), out
        msgs = _messages(caught)
        assert any("nothing was measured" in m for m in msgs), msgs
        after = trainer.calibrator.calibrator.log_temperatures.detach()
        assert torch.equal(before, after), "a refused batch was back-propagated anyway"

    def test_control_a_usable_loader_still_measures_and_trains(self):
        """CONTROL. The same loader through a finite model returns
        0.013412627391517162 and moves the calibration parameters."""
        trainer, _, _ = _trainer(_Healthy)
        copt = torch.optim.SGD(trainer.calibrator.parameters(), lr=0.5)
        before = trainer.calibrator.calibrator.log_temperatures.detach().clone()
        x = torch.randn(60, 4)
        _, y, gids = _calibration_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = trainer.fine_tune_calibration(_Loader([(x, y, gids)]), copt)

        assert _messages(caught) == []
        assert np.isfinite(out) and out > 0.0, out
        after = trainer.calibrator.calibrator.log_temperatures.detach()
        assert not torch.equal(before, after), "fine-tuning changed nothing"


# ===========================================================================
# 4. BoundedGroupLossConstraint.compute_violation / .signed_constraint_value
# ===========================================================================


class TestAnUnlabelledDatasetIsNotACompliantOne:
    def test_blanking_every_label_no_longer_certifies_the_model(self):
        """MEASURED BEFORE THE FIX, on the same 60 predictions in both rows::

            labels intact -> overall_violation 0.36, is_satisfied False,
                             could_not_evaluate False
            labels blanked-> overall_violation 0.0,  is_satisfied TRUE,
                             could_not_evaluate False, ZERO warnings

        A compliance certificate over a dataset with no labels, byte-identical
        to a measured pass. The mechanism is that ``y_pred != nan`` is True for
        every prediction, so each group's 0-1 loss is exactly 1.0, and this
        bound is RELATIVE: max(0, 1.0 - 1.1 * 1.0) is 0.0."""
        y_pred, _, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, np.full(60, np.nan), groups)

        assert v.could_not_evaluate is True
        assert v.is_satisfied is False, "an unlabelled dataset must not read as a PASS"
        assert v.details["insufficient_data"] is True
        assert v.details["n_unlabelled_rows"] == 60
        assert v.to_dict()["could_not_evaluate"] is True
        assert any("60 of 60 row(s) have no usable label" in m for m in _messages(caught))
        # The group losses are still reported, so a reader can see WHY.
        assert v.details["group_losses"] == {"a": 1.0, "b": 1.0}

    def test_is_satisfied_answers_none_when_half_the_labels_are_missing(self):
        """MEASURED BEFORE THE FIX: with 30 of 60 labels NaN the violation read
        0.13 against a true 0.36 and ``is_satisfied()`` answered False, a
        MEASURED breach of the wrong size. The magnitude is deliberately left
        as computed (blanking a mostly-real disparity is the over-correction
        this campaign warns about) and only the verdict is withdrawn."""
        y_pred, y_true, groups = _bgl_fixture()
        semi = y_true.astype(float)
        semi[::2] = np.nan
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verdict = constraint.is_satisfied(y_pred, semi, groups)
            v = constraint.compute_violation(y_pred, semi, groups)

        assert verdict is None, "part measurement and part fabrication is not a verdict"
        assert v.details["n_unlabelled_rows"] == 30
        assert float(v.overall_violation) == pytest.approx(0.13, abs=1e-9)

    def test_the_signed_value_no_longer_reports_no_violation_in_silence(self):
        """MEASURED BEFORE THE FIX: ``signed_constraint_value(..., 'a')`` on the
        unlabelled dataset returned 0.0, no push, no violation, for the group
        the model was wrong about 80% of the time, and emitted no warning at
        all. The steering value stays 0.0 by design (a NaN would poison the
        exponentiated-gradient multiplier), so the disclosure is what had to
        change: the unlabelled-row warning now reaches this caller too."""
        y_pred, _, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signed = constraint.signed_constraint_value(y_pred, np.full(60, np.nan), groups, "a")

        assert any("have no usable label" in m for m in _messages(caught)), _messages(caught)
        # And the measurement channel, which is the object it derived from.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = constraint.compute_violation(y_pred, np.full(60, np.nan), groups)
        assert v.could_not_evaluate is True
        assert signed == pytest.approx(0.0)

    def test_control_the_real_breach_is_still_measured_in_silence(self):
        """CONTROL. Labels intact: overall_violation 0.36, is_satisfied False,
        could_not_evaluate False, n_unlabelled_rows 0 and no warning. If this
        goes red the refusal has eaten a real measurement."""
        y_pred, y_true, groups = _bgl_fixture()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, y_true, groups)

        assert _messages(caught) == []
        assert float(v.overall_violation) == pytest.approx(0.36, abs=1e-9)
        assert v.is_satisfied is False
        assert v.could_not_evaluate is False
        assert v.details["n_unlabelled_rows"] == 0

    def test_control_string_labels_are_not_counted_as_unlabelled(self):
        """CONTROL. ``np.isfinite`` cannot read an object or string array, so a
        non-numeric label column must count as 0 unlabelled rows rather than
        raising or refusing everything."""
        groups = np.array(["a"] * 30 + ["b"] * 30)
        y_true = np.array(["yes"] * 60, dtype=object)
        y_pred = np.array(["yes"] * 45 + ["no"] * 15, dtype=object)
        constraint = BoundedGroupLossConstraint(tolerance=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, y_true, groups)

        assert v.details["n_unlabelled_rows"] == 0
        assert not any("no usable label" in m for m in _messages(caught))


# ===========================================================================
# 5. DemographicParityConstraint.compute_violation (ALREADY CORRECT)
# ===========================================================================


class TestDemographicParityWasAlreadyHonest:
    """Judged CORRECT by execution on inputs the existing pins do not cover.
    These hold it there."""

    def test_unscored_predictions_withdraw_the_verdict(self):
        """MEASURED: 60 NaN predictions in two groups return
        overall_violation 0.0 (every NaN binarises to a rejection, so the rates
        really are equal at 0.0) but with insufficient_data True,
        could_not_evaluate True, is_satisfied False and a warning naming 60 of
        60 rows. The 0.0 is disclosed as fabrication rather than passed off as
        parity."""
        groups = np.array(["a"] * 30 + ["b"] * 30)
        constraint = DemographicParityConstraint(tolerance=0.05)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(np.full(60, np.nan), np.ones(60, int), groups)

        assert v.could_not_evaluate is True
        assert v.is_satisfied is False
        assert v.details["n_unscored_rows"] == 60
        assert any("no usable score" in m for m in _messages(caught))

    def test_a_partially_unscored_batch_keeps_its_magnitude_and_loses_its_verdict(self):
        """MEASURED: three groups, c entirely unscored, a at rate 1.0 and b at
        0.0. overall_violation 1.0 is a real lower bound and is reported;
        could_not_evaluate is True and n_unscored_rows is 20."""
        groups = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
        y_pred = np.concatenate([np.full(20, 0.9), np.full(20, 0.1), np.full(20, np.nan)])
        constraint = DemographicParityConstraint(tolerance=0.05)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = constraint.compute_violation(y_pred, np.ones(60, int), groups)

        assert v.details["n_unscored_rows"] == 20
        assert v.could_not_evaluate is True
        assert float(v.overall_violation) == pytest.approx(1.0)

    def test_a_single_group_is_not_perfect_parity(self):
        """MEASURED: one sensitive value returns overall_violation NaN,
        is_satisfied False and could_not_evaluate True. There is no pair of
        groups, so there is no disparity to be zero."""
        constraint = DemographicParityConstraint(tolerance=0.05)
        rng = np.random.default_rng(0)
        y_pred = rng.random(60)
        v = constraint.compute_violation(y_pred, np.ones(60, int), np.array(["a"] * 60))
        assert v.overall_violation != v.overall_violation
        assert v.could_not_evaluate is True
        assert constraint.is_satisfied(y_pred, np.ones(60, int), np.array(["a"] * 60)) is None

    def test_control_a_constant_predictor_is_measured_as_equal_not_refused(self):
        """CONTROL, and the trap this batch names. A model that predicts one
        class for everybody IS trivially equal across groups, so 0.0 here is a
        TRUE measurement and must not be turned into a refusal: overall
        violation 0.0, is_satisfied True, could_not_evaluate False, no
        warning. The degeneracy of such a model is a property of the model, not
        something this metric may invent a could-not-check for."""
        groups = np.array(["a"] * 30 + ["b"] * 30)
        constraint = DemographicParityConstraint(tolerance=0.05)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(
                np.zeros(60), np.random.default_rng(0).integers(0, 2, 60), groups
            )

        assert _messages(caught) == []
        assert float(v.overall_violation) == pytest.approx(0.0)
        assert v.is_satisfied is True
        assert v.could_not_evaluate is False

    def test_control_a_real_disparity_is_still_measured(self):
        """CONTROL. Rates 1.0 and 0.0 give overall_violation 1.0, is_satisfied
        False, could_not_evaluate False, no warning."""
        groups = np.array(["a"] * 30 + ["b"] * 30)
        y_pred = np.concatenate([np.full(30, 0.9), np.full(30, 0.1)])
        constraint = DemographicParityConstraint(tolerance=0.05)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = constraint.compute_violation(y_pred, np.ones(60, int), groups)

        assert _messages(caught) == []
        assert float(v.overall_violation) == pytest.approx(1.0)
        assert v.could_not_evaluate is False


# ===========================================================================
# 6. CounterfactualFairnessLoss.forward
# ===========================================================================


class _Sigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.lin(x)).squeeze(-1)


def _cf_batch(seed: int = 0):
    torch.manual_seed(seed)
    feats = torch.randn(8, 3)
    model = _Sigmoid()
    y_pred = model(feats)
    y_true = torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0])
    two = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return feats, model, y_pred, y_true, two


def _coverage(components: Any) -> Dict[str, Any]:
    return components.batch_metrics


class TestACounterfactualThatIsTheFactualMeasuresNothing:
    def test_a_supplied_counterfactual_equal_to_the_factual_is_not_a_perfect_score(self):
        """MEASURED BEFORE THE FIX, 8 rows, two groups, lambda_fairness 1.0:

            generated arm, 1 group -> fairness_loss nan, reason
                                      'single_group', one warning
            supplied y_pred_cf = y_pred.clone()
                                   -> fairness_loss 0.0,
                                      fairness_penalty_assessed True,
                                      NOT ONE WARNING

        0.0 is the best attainable counterfactual-fairness score and it came
        from comparing a prediction against itself. The guard existed, one
        branch away, in the arm that GENERATES the counterfactual."""
        _, _, y_pred, y_true, two = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, two, y_pred_counterfactual=y_pred.clone(), return_components=True
            )

        metrics = _coverage(components)
        assert np.isnan(components.fairness_loss), components.fairness_loss
        assert metrics["fairness_penalty_assessed"] is False
        assert metrics["fairness_unassessable_reason"] == "counterfactual_equals_factual"
        assert metrics["fairness_loss_unassessed_value"] == pytest.approx(0.0)
        assert any("bit-identical to the factual" in m for m in _messages(caught))

    def test_the_same_tensor_passed_back_is_caught_too(self):
        """The most literal form: the caller hands the factual predictions in as
        their own counterfactual. Before, 0.0 and assessed."""
        _, _, y_pred, y_true, two = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, two, y_pred_counterfactual=y_pred, return_components=True
            )
        assert np.isnan(components.fairness_loss)
        assert any("bit-identical to the factual" in m for m in _messages(caught))

    def test_control_a_genuinely_different_counterfactual_is_still_measured(self):
        """CONTROL. y_pred + 0.2 gives an l2 penalty of 0.04 exactly, with
        fairness_penalty_assessed True and no warning, before and after."""
        _, _, y_pred, y_true, two = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, two, y_pred_counterfactual=y_pred + 0.2, return_components=True
            )

        assert _messages(caught) == []
        assert components.fairness_loss == pytest.approx(0.04, abs=1e-6)
        assert _coverage(components)["fairness_penalty_assessed"] is True

    def test_control_the_generated_arm_is_unchanged(self):
        """CONTROL. Features plus model on two groups still measures
        0.006912893150001764 with fairness_penalty_assessed True. The guard sits
        above the dispatch and must not intercept this arm."""
        feats, model, y_pred, y_true, two = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)

        _, components = loss_fn(
            y_pred, y_true, two, features=feats, model=model, return_components=True
        )

        assert _coverage(components)["fairness_penalty_assessed"] is True
        assert components.fairness_loss == pytest.approx(0.0069128931, abs=1e-7)

    def test_control_the_generated_arm_still_refuses_a_single_group(self):
        """CONTROL. The refusal that already existed: one sensitive value, so
        group_mean can produce no counterfactual. Reason 'single_group', not the
        new one."""
        feats, model, y_pred, y_true, _ = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)
        one = torch.zeros(8, dtype=torch.long)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, one, features=feats, model=model, return_components=True
            )

        assert np.isnan(components.fairness_loss)
        assert _coverage(components)["fairness_unassessable_reason"] == "single_group"
        assert any("produced no counterfactual" in m for m in _messages(caught))


# ===========================================================================
# 7. IndividualFairnessLoss.forward (ALREADY CORRECT)
# ===========================================================================


class TestIndividualFairnessWasAlreadyHonest:
    """Judged CORRECT by execution. Recorded BGL-B semi-proven, i.e. observed
    and unprotected; these are the pins."""

    def test_no_features_is_not_perfect_individual_fairness(self):
        """MEASURED: without features, fairness_loss is NaN with reason
        'features_not_provided' and a warning, and the finite 0.0 the optimizer
        saw is kept in fairness_loss_unassessed_value."""
        _, _, y_pred, y_true, two = _cf_batch()
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(y_pred, y_true, two, return_components=True)

        assert np.isnan(components.fairness_loss)
        metrics = _coverage(components)
        assert metrics["fairness_penalty_assessed"] is False
        assert metrics["fairness_unassessable_reason"] == "features_not_provided"
        assert metrics["fairness_loss_unassessed_value"] == pytest.approx(0.0)
        assert any("no two individuals can be compared" in m for m in _messages(caught))

    def test_a_batch_of_one_has_no_pair_and_says_so(self):
        """MEASURED: n=1 gives fairness_loss NaN, reason
        'fewer_than_two_samples', n_samples 1, and a warning."""
        feats, _, y_pred, y_true, two = _cf_batch()
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred[:1], y_true[:1], two[:1], features=feats[:1], return_components=True
            )

        assert np.isnan(components.fairness_loss)
        assert _coverage(components)["fairness_unassessable_reason"] == "fewer_than_two_samples"
        assert any("no pair of individuals" in m for m in _messages(caught))

    def test_n_neighbors_below_one_is_refused_at_construction(self):
        """MEASURED: n_neighbors=0 used to divide 0 by 0 and n_neighbors=-1
        reported fairness_loss -0.0 with n_pairs_compared -8 on a batch whose
        true violation was 0.514286. Both are refused now."""
        with pytest.raises(ValueError, match="n_neighbors must be at least 1"):
            IndividualFairnessLoss(n_neighbors=0)
        with pytest.raises(ValueError, match="n_neighbors must be at least 1"):
            IndividualFairnessLoss(n_neighbors=-1)

    def test_control_the_lipschitz_violation_is_measured_exactly(self):
        """CONTROL, against an independent hand computation. Eight IDENTICAL
        individuals scored 0.95 four times and 0.05 four times: all feature
        distances are 0, so every allowed distance is 0 and the penalty is the
        mean prediction gap over the 28 upper-triangle pairs. 16 cross pairs at
        0.9 and 12 same pairs at 0.0 give 14.4 / 28 = 0.5142857."""
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0)
        feats = torch.ones(8, 3)
        y_pred = torch.tensor([0.95] * 4 + [0.05] * 4)
        y_true = torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0])
        two = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(y_pred, y_true, two, features=feats, return_components=True)

        assert _messages(caught) == []
        assert components.fairness_loss == pytest.approx(16 * 0.9 / 28, abs=1e-5)
        metrics = _coverage(components)
        assert metrics["fairness_penalty_assessed"] is True
        assert metrics["n_pairs_compared"] == 28


# ===========================================================================
# 8. FairClassifier.predict
# ===========================================================================


class _PartlyUnscored(LogisticRegression):
    """A base estimator that has no score for some rows.

    The wrapper's whole premise is that it works with ANY estimator, so its
    scores are an input ``predict`` genuinely consumes. ``unscored_rows`` is set
    AFTER fitting so the fit itself is clean and only the predict-time scores
    are missing.
    """

    unscored_rows: Tuple[int, ...] = ()

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        for i in self.unscored_rows:
            proba[i, :] = np.nan
        return proba


def _classification_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 3))
    groups = np.array(["a"] * 60 + ["b"] * 60)
    logit = X[:, 0] + 1.5 * (groups == "a")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, groups


class TestPredictDoesNotFabricateADecisionForAnUnscoredRow:
    def test_an_unscored_row_is_refused_not_rejected(self):
        """MEASURED BEFORE THE FIX, 120 rows, method='threshold', a base
        estimator whose predict_proba returned NaN for rows 0, 1 and 61::

            clf.predict(X)                        -> those rows came back
                                                     0, 0, 0, confident
                                                     REJECTIONS, and the only
                                                     warning was about
                                                     sensitive_attr
            clf.predict_with_sensitive_attr(X, g) -> ValueError, "3 of 120
                                                     row(s) have no usable
                                                     score ... first at index
                                                     [0, 1, 61]"

        One class, two predict methods, opposite answers to the same row. An
        int label array has no value that can mean could-not-check, so this
        refuses, exactly as the sibling does."""
        X, y, groups = _classification_data()
        clf = FairClassifier(base_estimator=_PartlyUnscored(max_iter=200), method="threshold")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=groups)
        clf._inner_model.unscored_rows = (0, 1, 61)

        with pytest.raises(ValueError, match=r"3 of 120 row\(s\) have no usable score"):
            clf.predict(X)
        # The sibling refuses the identical input, which is the point.
        with pytest.raises(ValueError, match="no usable score"):
            clf.predict_with_sensitive_attr(X, groups)

    def test_control_a_fully_scored_batch_still_predicts_and_still_discloses(self):
        """CONTROL. With every row scored, predict() returns labels, warns that
        the group thresholds could not be applied, and differs from the
        mitigated answer on 3 of 120 rows (thresholds a=0.4555, b=0.5049)."""
        X, y, groups = _classification_data()
        clf = FairClassifier(base_estimator=LogisticRegression(max_iter=200), method="threshold")
        clf.fit(X, y, sensitive_attr=groups)

        with pytest.warns(UserWarning, match="Threshold method requires sensitive_attr"):
            plain = clf.predict(X)
        adjusted = clf.predict_with_sensitive_attr(X, groups)
        assert set(np.unique(plain)) <= {0, 1}
        assert int(np.sum(plain != adjusted)) > 0, "the two answers really are different models"

    @pytest.mark.parametrize("method", ["reductions", "grid_search"])
    def test_control_the_other_methods_are_untouched(self, method):
        """CONTROL. The refusal is scoped to the threshold branch, which is the
        only one that thresholds a score here; the other two delegate to a
        fitted ensemble and must stay silent."""
        X, y, groups = _classification_data()
        clf = FairClassifier(base_estimator=LogisticRegression(max_iter=200), method=method)
        clf.fit(X, y, sensitive_attr=groups)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = clf.predict(X)
        assert _messages(caught) == []
        assert out.shape == (120,)


# ===========================================================================
# 9. FairRegressor.fit
# ===========================================================================


def _regression_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 3))
    groups = np.array(["a"] * 60 + ["b"] * 60)
    y = X[:, 0] * 2 + (groups == "a") * 3.0 + rng.normal(scale=0.1, size=n)
    weights = np.where(groups == "a", 5.0, 1.0)
    return X, y, groups, weights


class TestFitSaysWhenTheWeightedFitNeverHappened:
    def test_dropped_sample_weights_are_named(self):
        """MEASURED BEFORE THE FIX, 120 rows, weights 5.0 on a and 1.0 on b,
        tolerance 0.1::

            LinearRegression   no weights   -> {'a':  0.26505074, ...}
                               with weights -> {'a':  0.28920397, ...}
            KNeighborsRegressor (fit takes no sample_weight)
                               no weights   -> {'a': -0.11278072, ...}
                               with weights -> {'a': -0.11278072, ...}
                                               BYTE-IDENTICAL, zero warnings

        So the weights demonstrably move the offsets where they are honoured,
        and where they are not the caller was told nothing: fit() returned self,
        is_fitted_ True, and group_offsets_ were measured off a model the caller
        did not ask for."""
        X, y, groups, weights = _regression_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            weighted = FairRegressor(
                base_estimator=KNeighborsRegressor(n_neighbors=3), tolerance=0.1
            ).fit(X, y, sensitive_attr=groups, sample_weight=weights)

        msgs = _messages(caught)
        assert any("accepts no sample_weight" in m for m in msgs), msgs
        assert any("fitted UNWEIGHTED" in m for m in msgs), msgs

        unweighted = FairRegressor(
            base_estimator=KNeighborsRegressor(n_neighbors=3), tolerance=0.1
        ).fit(X, y, sensitive_attr=groups)
        assert weighted.group_offsets_ == unweighted.group_offsets_, (
            "if these ever differ the estimator did honour the weights and the warning is wrong"
        )

    def test_control_no_warning_when_no_weights_were_passed(self):
        """CONTROL. The retry exists for estimators that take no sample_weight
        at all, and dropping a None costs nothing: that path must stay silent."""
        X, y, groups, _ = _regression_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FairRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)).fit(
                X, y, sensitive_attr=groups
            )
        assert _messages(caught) == []

    def test_control_weights_are_honoured_and_silent_where_the_estimator_takes_them(self):
        """CONTROL. LinearRegression takes sample_weight, so the offsets MUST
        move (0.26505074 to 0.28920397) and nothing may be warned about."""
        X, y, groups, weights = _regression_data()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            weighted = FairRegressor(base_estimator=LinearRegression(), tolerance=0.1).fit(
                X, y, sensitive_attr=groups, sample_weight=weights
            )
        unweighted = FairRegressor(base_estimator=LinearRegression(), tolerance=0.1).fit(
            X, y, sensitive_attr=groups
        )

        assert _messages(caught) == []
        assert weighted.group_offsets_["a"] == pytest.approx(0.28920397, abs=1e-6)
        assert unweighted.group_offsets_["a"] == pytest.approx(0.26505074, abs=1e-6)
        assert abs(weighted.group_offsets_["a"] - unweighted.group_offsets_["a"]) > 1e-4

    def test_control_the_offsets_still_close_the_mean_gap(self):
        """CONTROL. The mitigation must still work: after fitting, the adjusted
        group mean predictions sit within `tolerance` of each other."""
        X, y, groups, _ = _regression_data()
        reg = FairRegressor(base_estimator=LinearRegression(), tolerance=0.1).fit(
            X, y, sensitive_attr=groups
        )
        adjusted = reg.predict_with_sensitive_attr(X, groups)
        means = [float(adjusted[groups == g].mean()) for g in ("a", "b")]
        assert abs(means[0] - means[1]) <= 0.1 + 1e-9, means
