"""Surface grade g006: in_processing/calibrators/group_calibrators.py.

Every pin here was written after running the item and reading what it really
returned. Four fabrications were found by execution on 2026-09-17, all of them
the same shape: a 0.0 handed back from a calibration statistic that was never
computed, on a scale where 0.0 is PERFECT calibration.

  G006-1  TrainableGroupCalibrator.calibration_loss skipped every group below
          10 samples and then returned its 0.0 accumulator. An empty batch, a
          batch of four rows, or group ids matching no group all came back as
          0.0 with no warning. Measured before: calibration_loss on 4 rows
          -> 0.0; on zero rows -> 0.0. Now NaN with a UserWarning naming the
          group sizes.

  G006-2  The same loop averaged over self.n_groups even when only some groups
          were measured, so a skipped group was folded in as zero calibration
          error. Measured before: 40 rows all in group 0, group 1 empty
          -> 0.012718; now 0.025436, the group-0 estimate itself, plus a
          warning naming the skipped group.

  G006-3  CalibrationAwareTrainer.train_step reported "calibration_loss": 0.0
          both when include_calibration=False and when the calibrator had
          measured nothing at all. A training log could not tell a calibration
          term that was switched off from a batch that needed no correction.
          Now NaN in both cases, and the NaN is kept OUT of the objective so a
          refusal cannot poison the gradients.

  G006-4  CalibrationAwareTrainer.fine_tune_calibration ended in
          "total_cal_loss / n_batches if n_batches > 0 else 0.0". An empty
          val_loader, calibration_epochs=0, and batches too small to measure
          all returned 0.0. The class docstring binds this return to the name
          "ece". Measured before: fine_tune_calibration([], opt) -> 0.0. Now
          NaN with a UserWarning.

And one defect that is not a fabricated number but a destroyed one:

  G006-5  FocalCalibrator.forward applies p ** (1 + gamma) with gamma an
          unconstrained learnable parameter. Plain SGD on all-ones labels
          reached gamma = -1.11 in 20 steps (measured 2026-09-17), after which
          the exponent is <= 0, torch.pow returns >= 1 for every input, and the
          clamp flattens the entire group to 1.0. Every sample leaves as the
          same near-certain positive, a downstream parity check reads perfect
          agreement, and nothing said a word. The transform still runs (that is
          a product decision, not a grading one) but it now names the groups.

Each refusal pin is paired with a healthy-data control that asserts a real
value is still measured, computed independently from the documented formula
rather than copied from the code. A detector that refuses everything passes
every degenerate-input test while finding nothing.
"""

import warnings

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard dep of this module
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

pytestmark = needs_torch


def _mod():
    from vfairness.in_processing.calibrators import group_calibrators

    return group_calibrators


# ---------------------------------------------------------------------------
# Independent reference for the soft ECE, re-derived from the docstring
# formula in numpy. The controls below compare against THIS, never against
# whatever the implementation happens to return.
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _reference_soft_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        in_bin = _sigmoid((probs - lower) * 10) * _sigmoid((upper - probs) * 10)
        if in_bin.sum() > 1e-8:
            bin_count = in_bin.sum()
            bin_conf = (in_bin * probs).sum() / (bin_count + 1e-8)
            bin_acc = (in_bin * y).sum() / (bin_count + 1e-8)
            ece += bin_count * abs(bin_conf - bin_acc) / probs.shape[0]
    return float(ece)


def _two_group_batch(n_per_group: int = 100, seed: int = 11):
    """A batch with a real, findable calibration disparity.

    Group 0's logits are inflated threefold against labels drawn from the
    UNinflated probability, so group 0 is genuinely overconfident. Group 1 is
    drawn honestly. There is something real here to measure.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    base = rng.normal(size=n)
    group_ids = np.arange(n) % 2
    logits = np.where(group_ids == 0, base * 3.0, base)
    y = (rng.random(n) < _sigmoid(base)).astype(float)
    return (
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(group_ids, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# G006-1 / G006-2: calibration_loss
# ---------------------------------------------------------------------------


class TestCalibrationLossRefusesWhenNothingIsMeasurable:
    def test_a_batch_too_small_for_any_group_returns_nan_not_zero(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        logits = torch.tensor([3.0, -3.0, 2.0, -2.0])
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        gids = torch.tensor([0, 1, 0, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(logits, y, gids)

        assert torch.isnan(out).item(), f"expected a refusal, got {out.item()!r}"
        msgs = [str(w.message) for w in caught]
        assert any("no group reached min_group_size" in m for m in msgs), msgs
        # The reason must be legible without reading the source: the sizes are
        # in the message.
        assert any("{0: 2, 1: 2}" in m for m in msgs), msgs

    def test_zero_rows_returns_nan_not_zero(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        empty = torch.zeros(0)
        with pytest.warns(UserWarning, match="no calibration error was measured"):
            out = cal.calibration_loss(empty, empty, torch.zeros(0, dtype=torch.long))
        assert torch.isnan(out).item()

    def test_group_ids_that_match_no_group_return_nan_not_zero(self):
        """40 rows, all carrying a group id the calibrator has never heard of.

        Plenty of data, nothing measurable. 0.0 here was the worst of the set:
        a full batch reporting perfect calibration.
        """
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        logits, y, _ = _two_group_batch(n_per_group=20)
        gids = torch.full((logits.shape[0],), 7, dtype=torch.long)
        with pytest.warns(UserWarning, match="no calibration error was measured"):
            out = cal.calibration_loss(logits, y, gids)
        assert torch.isnan(out).item()

    def test_a_skipped_group_is_not_counted_as_zero_error(self):
        """G006-2. One measurable group, one below the minimum.

        The returned loss must be the measured group's estimate, not that
        estimate halved by averaging in a group nobody could measure.
        """
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        logits, y, gids = _two_group_batch(n_per_group=20)
        # Shrink group 1 to 3 rows by moving the rest out of range entirely.
        g1 = (gids == 1).nonzero(as_tuple=True)[0]
        keep = g1[:3]
        logits = torch.cat([logits[gids == 0], logits[keep]])
        y = torch.cat([y[gids == 0], y[keep]])
        gids = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(3, dtype=torch.long)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(logits, y, gids)

        p0 = _sigmoid(logits[:20].numpy().astype(np.float64))
        expected = 0.1 * _reference_soft_ece(p0, y[:20].numpy().astype(np.float64))
        assert float(out.item()) == pytest.approx(expected, rel=1e-4)
        msgs = [str(w.message) for w in caught]
        assert any("below min_group_size" in m and "[1]" in m for m in msgs), msgs

    def test_min_group_size_is_visible_in_the_signature_and_honoured(self):
        """The threshold that decides a refusal was a hardcoded 10 buried in
        the loop. It is now a named parameter, so a caller can see what made
        their batch unmeasurable and lower it deliberately."""
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        logits = torch.tensor([3.0, -3.0, 2.0, -2.0])
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        gids = torch.tensor([0, 1, 0, 1])
        out = cal.calibration_loss(logits, y, gids, min_group_size=2)
        assert torch.isfinite(out).item()
        assert float(out.item()) > 0.0


class TestCalibrationLossStillMeasuresControl:
    """The control. If these go red, the refusal above has eaten real evidence."""

    def test_healthy_two_group_batch_returns_the_independently_computed_value(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        logits, y, gids = _two_group_batch()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.calibration_loss(logits, y, gids)

        assert [str(w.message) for w in caught] == []
        lg = logits.numpy().astype(np.float64)
        yy = y.numpy().astype(np.float64)
        gg = gids.numpy()
        ref = sum(_reference_soft_ece(_sigmoid(lg[gg == g]), yy[gg == g]) for g in (0, 1))
        expected = 0.1 * ref / 2
        assert float(out.item()) == pytest.approx(expected, rel=1e-4)
        assert float(out.item()) > 0.0

    def test_the_loss_is_still_differentiable_and_moves_the_parameters(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2, method="temperature")
        logits, y, gids = _two_group_batch()
        before = cal.calibrator.log_temperatures.detach().clone()
        opt = torch.optim.SGD(cal.parameters(), lr=1.0)
        opt.zero_grad()
        # The loss only depends on the calibration parameters through the
        # calibrated logits, which is how train_step and fine_tune_calibration
        # both call it.
        loss = cal.calibration_loss(cal(logits, gids), y, gids)
        loss.backward()
        opt.step()
        after = cal.calibrator.log_temperatures.detach()
        assert torch.isfinite(after).all().item()
        assert not torch.equal(before, after), "the loss carried no gradient"


# ---------------------------------------------------------------------------
# G006-3: train_step
# ---------------------------------------------------------------------------


def _trainer(gc, n_groups: int = 2, seed: int = 7):
    torch.manual_seed(seed)
    model = nn.Linear(4, 1)
    calibrator = gc.TrainableGroupCalibrator(n_groups=n_groups)
    trainer = gc.CalibrationAwareTrainer(model, calibrator)
    opt = torch.optim.SGD(list(model.parameters()) + list(calibrator.parameters()), lr=0.1)
    return trainer, opt, nn.BCEWithLogitsLoss()


def _xy(n: int = 40, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=gen)
    y = (torch.rand(n, 1, generator=gen) < 0.4).float()
    gids = torch.arange(n) % 2
    return x, y, gids


class TestTrainStepReportsWhatItDidNotMeasure:
    def test_calibration_off_reports_nan_not_zero(self):
        gc = _mod()
        trainer, opt, loss_fn = _trainer(gc)
        x, y, gids = _xy()
        stats = trainer.train_step(x, y, gids, opt, loss_fn, include_calibration=False)
        assert np.isnan(stats["calibration_loss"]), stats
        # and the task loss is untouched by the change
        assert stats["total_loss"] == pytest.approx(stats["task_loss"])
        assert np.isfinite(stats["task_loss"])

    def test_an_unmeasurable_batch_reports_nan_and_does_not_poison_the_step(self):
        """The caller asked FOR calibration here and the calibrator could not
        deliver. The dict says so, and the NaN stays out of the objective: a
        NaN loss would have driven every model weight to NaN in one step."""
        gc = _mod()
        trainer, opt, loss_fn = _trainer(gc)
        x, y, gids = _xy(n=4)
        with pytest.warns(UserWarning, match="no calibration error was measured"):
            stats = trainer.train_step(x, y, gids, opt, loss_fn)
        assert np.isnan(stats["calibration_loss"]), stats
        assert np.isfinite(stats["total_loss"]), stats
        assert stats["total_loss"] == pytest.approx(stats["task_loss"])
        for p in trainer.model.parameters():
            assert torch.isfinite(p).all().item(), "a refusal reached the weights"
        for p in trainer.calibrator.parameters():
            assert torch.isfinite(p).all().item(), "a refusal reached the calibrator"

    def test_control_a_measurable_batch_still_reports_a_real_number(self):
        gc = _mod()
        trainer, opt, loss_fn = _trainer(gc)
        x, y, gids = _xy(n=40)
        stats = trainer.train_step(x, y, gids, opt, loss_fn)
        assert np.isfinite(stats["calibration_loss"])
        assert stats["calibration_loss"] > 0.0
        assert stats["total_loss"] == pytest.approx(
            stats["task_loss"] + stats["calibration_loss"], rel=1e-5
        )


# ---------------------------------------------------------------------------
# G006-4: fine_tune_calibration
# ---------------------------------------------------------------------------


class TestFineTuneCalibrationRefusesWhenItMeasuredNothing:
    def test_an_empty_loader_returns_nan_not_zero(self):
        gc = _mod()
        trainer, _, _ = _trainer(gc)
        copt = torch.optim.SGD(trainer.calibrator.parameters(), lr=0.1)
        with pytest.warns(UserWarning, match="nothing was measured"):
            out = trainer.fine_tune_calibration([], copt)
        assert np.isnan(out), out

    def test_zero_epochs_returns_nan_not_zero(self):
        gc = _mod()
        torch.manual_seed(7)
        model = nn.Linear(4, 1)
        calibrator = gc.TrainableGroupCalibrator(n_groups=2)
        trainer = gc.CalibrationAwareTrainer(model, calibrator, calibration_epochs=0)
        copt = torch.optim.SGD(calibrator.parameters(), lr=0.1)
        x, y, gids = _xy(n=40)
        with pytest.warns(UserWarning, match="nothing was measured"):
            out = trainer.fine_tune_calibration([(x, y, gids)], copt)
        assert np.isnan(out), out

    def test_batches_too_small_return_nan_and_leave_the_parameters_alone(self):
        gc = _mod()
        trainer, _, _ = _trainer(gc)
        copt = torch.optim.SGD(trainer.calibrator.parameters(), lr=0.1)
        before = trainer.calibrator.calibrator.log_temperatures.detach().clone()
        x, y, gids = _xy(n=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = trainer.fine_tune_calibration([(x, y, gids)], copt)
        assert np.isnan(out), out
        after = trainer.calibrator.calibrator.log_temperatures.detach()
        assert torch.equal(before, after), "a refused batch was back-propagated anyway"
        assert torch.isfinite(after).all().item()
        assert any("nothing was measured" in str(w.message) for w in caught)

    def test_control_a_real_loader_returns_a_measured_mean_and_trains(self):
        gc = _mod()
        trainer, _, _ = _trainer(gc)
        copt = torch.optim.SGD(trainer.calibrator.parameters(), lr=0.5)
        before = trainer.calibrator.calibrator.log_temperatures.detach().clone()
        x, y, gids = _xy(n=60)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = trainer.fine_tune_calibration([(x, y, gids)], copt)
        assert np.isfinite(out), out
        assert out > 0.0
        assert [str(w.message) for w in caught] == []
        after = trainer.calibrator.calibrator.log_temperatures.detach()
        assert not torch.equal(before, after), "fine-tuning changed nothing"


# ---------------------------------------------------------------------------
# G006-5: FocalCalibrator saturation
# ---------------------------------------------------------------------------


class TestFocalCalibratorNamesItsOwnCollapse:
    def test_gamma_below_minus_one_is_named(self):
        gc = _mod()
        cal = gc.FocalCalibrator(n_groups=2)
        with torch.no_grad():
            cal.gamma[0] = -1.5
        probs = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95])
        gids = torch.zeros(5, dtype=torch.long)
        with pytest.warns(UserWarning, match="saturates to 1.0"):
            out = cal(probs, gids)
        # This is what the caller gets: five different probabilities in, one
        # value out. It is only tolerable because it is now announced.
        assert float(np.ptp(out.detach().numpy())) == 0.0
        assert out[0].item() == pytest.approx(1.0, abs=1e-6)

    def test_a_group_absent_from_the_batch_does_not_raise_the_alarm(self):
        """The warning must describe THIS batch. Group 1 is degenerate but
        contributes no sample, so there is nothing to say about it."""
        gc = _mod()
        cal = gc.FocalCalibrator(n_groups=2)
        with torch.no_grad():
            cal.gamma[1] = -3.0
        probs = torch.tensor([0.2, 0.4, 0.6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal(probs, torch.zeros(3, dtype=torch.long))
        assert [str(w.message) for w in caught] == []
        assert torch.allclose(out, probs, atol=1e-6)

    def test_control_a_healthy_gamma_transforms_and_says_nothing(self):
        gc = _mod()
        cal = gc.FocalCalibrator(n_groups=1)
        with torch.no_grad():
            cal.gamma[0] = 1.0
        probs = torch.tensor([0.1, 0.5, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal(probs, torch.zeros(3, dtype=torch.long))
        assert [str(w.message) for w in caught] == []
        # p ** (1 + 1) = p squared, computed independently.
        assert out.detach().numpy() == pytest.approx(np.array([0.01, 0.25, 0.81]), abs=1e-6)


# ---------------------------------------------------------------------------
# CalibrationState / get_calibration_state: the ECE nobody computed
# ---------------------------------------------------------------------------


class TestCalibrationStateClaimsNothingItDidNotMeasure:
    def test_global_ece_is_nan_not_a_perfect_zero(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        state = cal.get_calibration_state()
        d = state.to_dict()
        assert np.isnan(d["global_ece"]), d
        # Empty, not populated with zeros: no entry claims nothing.
        assert d["group_ece"] == {}
        assert d["group_ece_post"] == {}

    def test_control_to_dict_carries_the_real_fitted_parameters(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2, method="temperature")
        with torch.no_grad():
            cal.calibrator.log_temperatures[0] = float(np.log(2.0))
            cal.calibrator.log_temperatures[1] = float(np.log(0.5))
        d = cal.get_calibration_state().to_dict()
        assert d["method"] == "temperature"
        assert d["parameters"]["group_0"] == pytest.approx(2.0, rel=1e-5)
        assert d["parameters"]["group_1"] == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# The four forwards and their get_parameters, as controls. These exist so that
# a later "refuse everything" regression in this file cannot pass unnoticed.
# ---------------------------------------------------------------------------


class TestCalibratorTransformsStillTransform:
    def test_temperature_divides_by_the_learned_temperature(self):
        gc = _mod()
        cal = gc.TemperatureScalingCalibrator(n_groups=2)
        with torch.no_grad():
            cal.log_temperatures[0] = float(np.log(2.0))
            cal.log_temperatures[1] = float(np.log(4.0))
        logits = torch.tensor([4.0, 8.0])
        out = cal(logits, torch.tensor([0, 1]))
        assert out.detach().numpy() == pytest.approx(np.array([2.0, 2.0]), abs=1e-5)
        assert cal.get_parameters() == pytest.approx({"group_0": 2.0, "group_1": 4.0}, rel=1e-5)

    def test_platt_applies_the_learned_affine_map(self):
        gc = _mod()
        cal = gc.PlattScalingCalibrator(n_groups=1)
        with torch.no_grad():
            cal.a[0] = 3.0
            cal.b[0] = -1.0
        out = cal(torch.tensor([2.0, -2.0]), torch.zeros(2, dtype=torch.long))
        assert out.detach().numpy() == pytest.approx(np.array([5.0, -7.0]), abs=1e-5)
        assert cal.get_parameters()["group_0"] == pytest.approx({"a": 3.0, "b": -1.0})

    def test_beta_matches_the_kull_formula(self):
        gc = _mod()
        cal = gc.BetaCalibrator(n_groups=1)
        with torch.no_grad():
            cal.c[0] = 2.0
            cal.d[0] = 0.1
            cal.e[0] = 0.5
        logits = torch.tensor([1.0, -2.0])
        out = cal(logits, torch.zeros(2, dtype=torch.long))
        p = _sigmoid(logits.numpy().astype(np.float64))
        expected = 2.0 * np.log(p) - 0.5 * np.log(1 - p) + 0.1
        assert out.detach().numpy() == pytest.approx(expected, rel=1e-5)
        assert cal.get_parameters()["group_0"] == pytest.approx({"c": 2.0, "d": 0.1, "e": 0.5})

    def test_focal_get_parameters_reports_the_real_gamma(self):
        gc = _mod()
        cal = gc.FocalCalibrator(n_groups=2)
        with torch.no_grad():
            cal.gamma[0] = 0.75
            cal.gamma[1] = -0.25
        assert cal.get_parameters() == pytest.approx({"group_0": 0.75, "group_1": -0.25}, rel=1e-6)

    def test_the_unified_forward_delegates_to_the_chosen_method(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=1, method="platt")
        with torch.no_grad():
            cal.calibrator.a[0] = 2.0
            cal.calibrator.b[0] = 1.0
        out = cal(torch.tensor([3.0]), torch.zeros(1, dtype=torch.long))
        assert out.item() == pytest.approx(7.0, abs=1e-5)


class TestFactoryAndFreezing:
    def test_create_group_calibrator_builds_the_named_method(self):
        gc = _mod()
        cal = gc.create_group_calibrator(n_groups=3, method="beta")
        assert isinstance(cal, gc.TrainableGroupCalibrator)
        assert isinstance(cal.calibrator, gc.BetaCalibrator)
        assert cal.n_groups == 3
        assert len(cal.calibrator.get_parameters()) == 3

    def test_create_group_calibrator_refuses_an_unknown_method(self):
        gc = _mod()
        with pytest.raises(ValueError, match="not a valid CalibrationMethodType"):
            gc.create_group_calibrator(n_groups=2, method="definitely-not-a-method")

    def test_freeze_and_unfreeze_flip_requires_grad(self):
        gc = _mod()
        cal = gc.TrainableGroupCalibrator(n_groups=2)
        cal.freeze()
        assert [p.requires_grad for p in cal.parameters()] == [False]
        cal.unfreeze()
        assert [p.requires_grad for p in cal.parameters()] == [True]

    def test_check_torch_available_returns_none_when_torch_is_here(self):
        gc = _mod()
        assert gc.check_torch_available() is None
