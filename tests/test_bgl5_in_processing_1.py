"""BGL5 pins for batch A-in_processing-1: the six grades the BGL4 audit overturned.

Every test here was written AFTER the fix it guards, was confirmed to go red with
the defect put back (the sabotage is named in the docstring of each class), and is
paired with an over-correction control that asserts the healthy input still gets
its real number, by value.

The six rows:

  G06  compare_methods           partial-failure disclosure reached no surface
  G07  evaluate_baseline         a target it cannot read is not a measured score
  G09  generate_recommendation   evidence re-pointed: the graded test was a
                                 CONTROL that cannot fail when the refusal breaks
  G11  baseline_comparison_summary  a supplied accuracy coerced before checked
  G14  calibration_loss          REGRESSION caused by the BGL3 fix, plus a
                                 refusal unreachable at min_group_size=0
  G21  BaseFairnessLoss.forward  the coverage guard keyed on the weight's SHAPE
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression

from vfairness.in_processing.analyzer import (
    FairnessTrainingAnalyzer,
    MethodComparison,
    _unscorable_rows,
    baseline_comparison_summary,
)
from vfairness.in_processing.calibrators.group_calibrators import (
    CalibrationAwareTrainer,
    TrainableGroupCalibrator,
)
from vfairness.in_processing.loss_functions.base import BaseFairnessLoss
from vfairness.in_processing.wrappers import FairClassifier

_N = 240


def _two_group(seed: int = 11):
    """The 240-row two-group classification fixture the BGL4 audit used."""
    rng = np.random.default_rng(seed)
    s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
    x0 = rng.normal(loc=np.where(s == "a", 1.2, -1.2), scale=0.7)
    X = np.column_stack([x0, rng.normal(size=_N)])
    y = (x0 + rng.normal(scale=0.4, size=_N) > 0).astype(int)
    return X, y, s


def _calibration_batch():
    """60 finite rows in two groups of 30, seeded so the loss is reproducible."""
    torch.manual_seed(0)
    calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
    logits = torch.randn(60, 1)
    y = (torch.arange(60) % 2).float()
    groups = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    return calibrator, logits, y, groups


# ===========================================================================
# G14, TrainableGroupCalibrator.calibration_loss.
#
# SABOTAGE (run 2026-09-27):
#   S-A  _finite_rows: `if ok.dim() > 1:` body restored to the single
#        `ok.reshape(ok.shape[0], -1).all(dim=1)` -> the three empty-batch tests
#        failed with "RuntimeError: cannot reshape tensor of 0 elements into
#        shape [0, -1]"; restored, green.
#   S-B  the refusal narrowed back to `if n_usable_g < min_group_size:` ->
#        test_min_group_size_zero_does_not_score_a_diverged_batch_as_perfect
#        failed with "returned 0.0 for a batch with no usable row"; restored,
#        green.
#   S-C  the out-of-range warning deleted -> test_rows_in_no_group_are_disclosed
#        failed with "DID NOT WARN"; restored, green.
# ===========================================================================


class TestG14TheRefusalIsReachableOnEveryInputItsDocstringNames:
    def test_an_empty_batch_is_refused_and_not_a_traceback(self):
        """The docstring names "an empty batch" as a case the NaN refusal covers.

        BEFORE (the regression the BGL3 fix itself introduced): the fix's own
        ``_finite_rows`` reshaped a (0, 1) tensor to (0, -1), which torch refuses,
        so the empty batch RAISED "RuntimeError: cannot reshape tensor of 0
        elements into shape [0, -1]" from group_calibrators.py:87. The same call
        with flat (0,) logits returned NaN, before the fix and after it, so the
        crash was reachable only through the trailing dimension.

        AFTER: tensor(nan) with the "no group reached min_group_size=10" warning,
        which is what the docstring promises.
        """
        calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
        for logits in (torch.zeros(0, 1), torch.zeros(0)):
            with pytest.warns(UserWarning, match="no group reached min_group_size"):
                value = float(
                    calibrator.calibration_loss(
                        logits, torch.zeros(0), torch.zeros(0, dtype=torch.long)
                    )
                )
            assert math.isnan(value), "returned " + repr(value) + " for an empty batch"

    def test_min_group_size_zero_does_not_score_a_diverged_batch_as_perfect(self):
        """min_group_size is public and documented, and 0 disabled the refusal.

        BEFORE: the refusal read ``int(mask.sum()) < min_group_size`` and 0 is not
        < 0, so a group with no usable row was counted as measured and kept its
        seed of exactly 0.0. 60 all-NaN logits in two groups of 30 returned 0.0
        at min_group_size=0 (one warning, about the excluded rows) against nan at
        the default min_group_size=10 (two warnings). 0.0 on this scale is PERFECT
        calibration.

        AFTER: nan with two warnings at BOTH thresholds.
        """
        calibrator, _, y, groups = _calibration_batch()
        logits = torch.full((60, 1), float("nan"))
        for min_group_size in (0, 10):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = float(
                    calibrator.calibration_loss(
                        logits, y, groups, min_group_size=min_group_size
                    ).detach()
                )
            assert math.isnan(value), (
                "min_group_size=" + repr(min_group_size) + " returned " + repr(value)
            )
            assert len(caught) == 2, [str(w.message)[:60] for w in caught]

    def test_rows_in_no_group_are_disclosed_as_uncovered(self):
        """A row whose group id is out of range entered no bin and no warning.

        BEFORE: 60 rows of which 30 carried group id 7 with n_groups=2 returned
        2.135953664779663, byte-identical to the loss over the in-range 30 rows
        ALONE, and the only warning named "group(s) [1] are below
        min_group_size", which is true of the empty group and says nothing about
        the 30 rows the number does not cover.

        AFTER: the same 2.135953664779663, now with "30 of 60 row(s) carry a group
        id outside range(2) ... The returned loss covers 30 of 60 row(s)". The
        value is deliberately unchanged: those rows have no calibration parameters
        to be scored against, so the defect was the silence, not the arithmetic.
        """
        calibrator, logits, y, _ = _calibration_batch()
        groups = torch.cat(
            [torch.zeros(30, dtype=torch.long), torch.full((30,), 7, dtype=torch.long)]
        )
        with pytest.warns(UserWarning, match=r"30 of 60 row\(s\) carry a group id outside"):
            mixed = float(calibrator.calibration_loss(logits, y, groups).detach())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            in_range_only = float(
                calibrator.calibration_loss(logits[:30], y[:30], groups[:30]).detach()
            )
        assert mixed == pytest.approx(2.135953664779663)
        assert mixed == pytest.approx(in_range_only), (
            "the warning is the whole fix: the number covers half the batch"
        )

    def test_the_trainer_surfaces_refuse_an_empty_batch_too(self):
        """Same root cause one layer up: both promise NaN, both raised.

        BEFORE: ``train_step`` and ``fine_tune_calibration`` on an empty batch
        both raised the same "cannot reshape tensor of 0 elements into shape
        [0, -1]" out of the calibrator.

        AFTER: train_step returns {'total_loss': nan, 'task_loss': nan,
        'calibration_loss': nan} and fine_tune_calibration returns nan.
        """
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
        trainer = CalibrationAwareTrainer(model=model, calibrator=calibrator, calibration_epochs=1)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(calibrator.parameters()), lr=0.01
        )
        empty = (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, dtype=torch.long))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = trainer.train_step(*empty, optimizer, nn.BCEWithLogitsLoss())
            assert math.isnan(out["calibration_loss"]), out
            assert math.isnan(trainer.fine_tune_calibration([empty], optimizer))

    def test_control_a_healthy_batch_still_measures_and_stays_silent(self):
        """OVER-CORRECTION CONTROL, by value.

        60 finite rows in two groups of 30 (seed 0) measure
        2.160198926925659 with ZERO warnings, at the default min_group_size AND
        at min_group_size=0, which is the argument the refusal above was widened
        for. The number is identical before and after the fix.
        """
        for min_group_size in (10, 0):
            calibrator, logits, y, groups = _calibration_batch()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = calibrator.calibration_loss(
                    logits, y, groups, min_group_size=min_group_size
                )
            assert float(value.detach()) == pytest.approx(2.160198926925659)
            assert not [str(w.message) for w in caught]
            assert value.requires_grad, "the loss must still carry a gradient"

    def test_control_the_trainer_still_trains_on_a_usable_loader(self):
        """OVER-CORRECTION CONTROL for the two trainer surfaces, by value.

        train_step over 60 usable rows reports calibration_loss
        0.014672358520328999 in silence, and fine_tune_calibration reports
        0.014611907303333282 and MOVES the calibrator's log_temperatures.
        """
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
        trainer = CalibrationAwareTrainer(model=model, calibrator=calibrator, calibration_epochs=1)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(calibrator.parameters()), lr=0.01
        )
        X = torch.randn(60, 3)
        y = (torch.arange(60) % 2).float().unsqueeze(1)
        groups = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = trainer.train_step(X, y, groups, optimizer, nn.BCEWithLogitsLoss())
            before = calibrator.calibrator.log_temperatures.detach().clone()
            mean_loss = trainer.fine_tune_calibration([(X, y, groups)], optimizer)
            after = calibrator.calibrator.log_temperatures.detach().clone()
        assert out["calibration_loss"] == pytest.approx(0.014672358520328999)
        assert mean_loss == pytest.approx(0.014611907303333282)
        assert not torch.equal(before, after), "a fine-tune that measured must also move"
        assert not [str(w.message) for w in caught]


# ===========================================================================
# G21, BaseFairnessLoss.forward.
#
# SABOTAGE (run 2026-09-27):
#   S-D  the `elif weights.numel() == 1:` branch of _task_loss_coverage deleted
#        -> 3 failed (both broadcast spellings and the epoch roll-up) with
#        "task_loss 0.0 with task_rows_used 4"; restored, green.
#   S-E  _report_scalar's numel()-aware body replaced by `return float(
#        value.detach().reshape(()))` -> the reduction="none" test failed with
#        "RuntimeError: shape '[]' is invalid for input of size 4"; restored,
#        green.
# ===========================================================================


class _Dummy(BaseFairnessLoss):
    """The documented extension point: a direct subclass with a fixed penalty."""

    def _compute_fairness_penalty(self, y_pred, y_true, sensitive_attr):
        return torch.tensor(0.25)


def _four_rows():
    return (
        torch.tensor([0.9, 0.8, 0.2, 0.1], requires_grad=True),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )


class TestG21AScalarZeroWeightIsNotAMeasuredTaskLoss:
    @pytest.mark.parametrize(
        "weight",
        [torch.zeros(1), torch.tensor(0.0), torch.tensor([float("nan")])],
        ids=["numel-1-zero", "0-dim-zero", "numel-1-nan"],
    )
    def test_a_broadcast_dead_weight_is_not_a_perfect_fit(self, weight):
        """The guard was keyed on the weight's SHAPE rather than on its effect.

        BEFORE, on the four-row batch: ``sample_weight=torch.zeros(4)`` was
        correctly refused (task_loss nan, task_rows_used 0, one warning), while
        ``torch.zeros(1)`` and ``torch.tensor(0.0)`` reported task_loss 0.0 with
        task_rows_used 4, task_loss_assessed True and NO warning, because
        ``weights.numel() % n_rows == 0`` is False for 1 % 4 and the count was
        left at n_rows. torch broadcasts a one-element weight over every row, so
        every row really was multiplied by zero. A non-finite scalar weight
        (nan) was reported as 4 rows used as well.

        AFTER: all three report task_loss nan with task_rows_used 0,
        task_loss_assessed False and the "0 of 4 row(s) entered the task loss"
        warning.
        """
        y_pred, y_true, sensitive = _four_rows()
        loss = _Dummy(lambda_fairness=0.1)
        with pytest.warns(UserWarning, match=r"0 of 4 row\(s\) entered the task"):
            _, components = loss(
                y_pred, y_true, sensitive, sample_weight=weight, return_components=True
            )
        assert components.batch_metrics["task_loss_assessed"] is False, (
            "task_loss "
            + repr(components.task_loss)
            + " with task_rows_used "
            + repr(components.batch_metrics["task_rows_used"])
        )
        assert components.batch_metrics["task_rows_used"] == 0
        assert math.isnan(components.task_loss)

    def test_an_epoch_of_scalar_weighted_out_batches_is_not_a_completed_fit(self):
        """The epoch this rolls up into.

        BEFORE: three batches every row of which was weighted out through a
        scalar averaged to avg_task_loss 0.0, the "completed epoch over nothing"
        that ``_task_loss_coverage``'s own docstring names.

        AFTER: avg_task_loss nan, while avg_total_loss stays 0.02500000037252903,
        which is the step the optimizer really took.
        """
        y_pred, y_true, sensitive = _four_rows()
        loss = _Dummy(lambda_fairness=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                loss(y_pred, y_true, sensitive, sample_weight=torch.zeros(1))
            metrics = loss.end_epoch()
        assert metrics is not None
        assert math.isnan(metrics.avg_task_loss), repr(metrics.avg_task_loss)
        assert metrics.avg_total_loss == pytest.approx(0.02500000037252903)

    def test_the_documented_reduction_none_records_the_batch_instead_of_raising(self):
        """``reduction="none"`` is in this class's own Literal and could not run.

        BEFORE: ``LossComponents(total_loss=total_loss.item(), ...)`` raised
        "RuntimeError: a Tensor with 4 elements cannot be converted to Scalar"
        whenever track_metrics was on, which is the default, so the option was
        unusable.

        AFTER: task_loss 0.16425204277038574, the mean of the four per-row
        losses, with batch_metrics["reduction"] == "none" beside it so the
        aggregation cannot be mistaken for a reduction the caller asked for. The
        returned tensor still carries one value per row.
        """
        y_pred, y_true, sensitive = _four_rows()
        loss = _Dummy(lambda_fairness=0.1, reduction="none")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            total, components = loss(y_pred, y_true, sensitive, return_components=True)
        assert total.numel() == 4, "the returned tensor must stay unreduced"
        assert components.task_loss == pytest.approx(0.16425204277038574)
        assert components.batch_metrics["reduction"] == "none"
        assert components.batch_metrics["task_loss_assessed"] is True
        assert not [str(w.message) for w in caught]

    @pytest.mark.parametrize(
        "weight,expected,rows_used",
        [
            (None, 0.16425204277038574, 4),
            (torch.ones(4), 0.16425204277038574, 4),
            (torch.ones(1), 0.16425204277038574, 4),
            (torch.tensor(2.0), 0.3285040855407715, 4),
            (torch.tensor([1.0, 0.0, 1.0, 1.0]), 0.10846614837646484, 3),
        ],
        ids=["none", "ones-4", "ones-1-broadcast", "scalar-2", "one-row-weighted-out"],
    )
    def test_control_a_live_weight_still_measures_its_real_number(
        self, weight, expected, rows_used
    ):
        """OVER-CORRECTION CONTROL, by value.

        A one-element weight is refused only when it is dead. ``torch.ones(1)``
        broadcasts a LIVE weight over the same four rows and must still measure
        0.16425204277038574, and ``torch.tensor(2.0)`` must still measure double
        it. A partially weighted batch is still measured, over torch's own
        weighted-loss convention (divide by the full row count).
        """
        y_pred, y_true, sensitive = _four_rows()
        loss = _Dummy(lambda_fairness=0.1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss(
                y_pred, y_true, sensitive, sample_weight=weight, return_components=True
            )
        assert components.task_loss == pytest.approx(expected)
        assert components.batch_metrics["task_rows_used"] == rows_used
        assert components.batch_metrics["task_loss_assessed"] is True
        assert not [str(w.message) for w in caught]


# ===========================================================================
# G07, FairnessTrainingAnalyzer.evaluate_baseline.
#
# SABOTAGE (run 2026-09-27):
#   S-F  `if n_unscorable or self.n_samples == 0:` replaced with `if False:` ->
#        4 failed (the two unreadable-target tests, the renderer test and the
#        summary assertion), with "accuracy is nan and the flag beside it says
#        measured" and "accuracy came back 0.0 with accuracy_measured True";
#        restored, green.
#   S-G  full_analysis's `"accuracy": _as_measured(baseline.accuracy)` restored
#        to `baseline.accuracy` -> the renderer test failed with
#        "training_analysis_report_to_svg puts 'accuracy nan' in the accessible
#        description"; restored, green.
# ===========================================================================


class TestG07AnUnreadableTargetIsNotAMeasuredAccuracy:
    def test_a_regression_target_carrying_a_nan_is_not_a_measured_r2(self):
        """The zero-variance guard could not fire on a NaN-bearing target.

        BEFORE: ``np.ptp`` of a NaN-bearing array is nan and ``nan == 0.0`` is
        False; ``np.var`` is nan and ``nan <= 0.0`` is False, so neither half of
        the guard could fire. y = np.arange(240) with y[3] = nan returned
        accuracy nan with parameters['accuracy_measured'] True and NO warning.

        AFTER: accuracy nan with accuracy_measured False and the warning naming
        "1 of 240 row(s)".
        """
        X, _, s = _two_group()
        y = np.arange(_N, dtype=float)
        y[3] = np.nan
        analyzer = FairnessTrainingAnalyzer(X, y, s, task_type="regression")
        with pytest.warns(UserWarning, match=r"accuracy NOT MEASURED \(1 of 240 row"):
            result = analyzer.evaluate_baseline(y_pred=np.arange(_N, dtype=float))
        assert result.parameters["accuracy_measured"] is False, (
            "accuracy is " + repr(result.accuracy) + " and the flag beside it says measured"
        )
        assert math.isnan(result.accuracy)

    def test_a_classification_target_of_all_nan_is_not_a_measured_zero(self):
        """The classification branch wrote accuracy_measured = True as a literal.

        BEFORE: y and y_pred both all NaN returned accuracy 0.0 with
        parameters['accuracy_measured'] True, because ``nan == nan`` is False on
        every row, so 240 unscorable rows read as a model that got every row
        wrong. The flag could not be False on this path at all.

        AFTER: accuracy nan with accuracy_measured False and the warning naming
        "240 of 240 row(s)".
        """
        X, _, s = _two_group()
        analyzer = FairnessTrainingAnalyzer(X, np.full(_N, np.nan), s)
        with pytest.warns(UserWarning, match=r"accuracy NOT MEASURED \(240 of 240 row"):
            result = analyzer.evaluate_baseline(y_pred=np.full(_N, np.nan))
        assert result.parameters["accuracy_measured"] is False, (
            "accuracy came back " + repr(result.accuracy) + " with accuracy_measured True"
        )
        assert math.isnan(result.accuracy)

    def test_the_undefined_r2_is_not_drawn_as_a_number_a_reader_can_read(self):
        """The disclosure has to survive full_analysis to reach a reader.

        BEFORE: full_analysis rebuilt baseline_metrics from three fields and
        dropped parameters['accuracy_measured'], and NaN is not a state the
        rendering layer speaks: ``_baseline_state`` derives baseline_measured
        from ``accuracy is not None``. Both canvases drew <text>nan</text> and a
        <desc> reading "Baseline accuracy nan, violation 0.000 (constraint
        satisfied) ... (severity: LOW)", while summary() on the SAME report
        printed "Accuracy: not measured".

        AFTER: baseline_metrics['accuracy'] is None with accuracy_measured False
        and the reason beside it, neither canvas draws a number, and neither
        <desc> says "accuracy nan".
        """
        from vfairness.rendering.adapters_training import (
            training_analysis_report_to_svg,
            training_report_to_svg,
        )

        rng = np.random.default_rng(3)
        X = rng.normal(size=(_N, 2))
        s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
        analyzer = FairnessTrainingAnalyzer(X, np.full(_N, 3.0), s, task_type="regression")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.full_analysis(base_estimator=LinearRegression())

        assert report.baseline_metrics["accuracy"] is None
        assert report.baseline_metrics["accuracy_measured"] is False
        assert "zero variance" in report.baseline_metrics["accuracy_not_measured_reason"]
        accuracy_line = [ln for ln in report.summary().splitlines() if ln.startswith("Accuracy:")][
            0
        ]
        assert "not measured" in accuracy_line and "nan" not in accuracy_line

        for render in (training_analysis_report_to_svg, training_report_to_svg):
            svg = render(report)
            drawn = [t.strip().lower() for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]
            assert "nan" not in drawn, (
                render.__name__ + " draws a bare 'nan' where a not-measured state belongs"
            )
            desc = re.search(r"<desc>(.*?)</desc>", svg, re.S)
            assert desc is not None and "accuracy nan" not in desc.group(1).lower(), (
                render.__name__ + " puts 'accuracy nan' in the accessible description"
            )

    def test_an_overflowing_r2_is_not_a_measured_minus_infinity(self):
        """The same defect class the hardcoded flag belonged to, one input further.

        BEFORE: the flag was asserted beside the value, so any OTHER arithmetic
        that produced a non-finite score kept it. 240 finite rows with
        y_pred = 1e200 overflow the squared error to inf (numpy says "overflow
        encountered in square"), so R^2 = 1 - inf/Var(y) came out -inf with
        parameters['accuracy_measured'] True, and -inf compares worse than every
        threshold a consumer can set: a measured, infinitely bad model, from a
        prediction that is merely large.

        AFTER: accuracy nan with accuracy_measured False and the reason "the
        score computed as np.float64(-inf), which is not a finite measurement".
        The flag is derived from the value now, so it cannot disagree with it.
        """
        rng = np.random.default_rng(7)
        X = rng.normal(size=(_N, 2))
        s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
        y = rng.normal(size=_N)
        y_pred = np.full(_N, 1e200)
        assert np.isfinite(y).all() and np.isfinite(y_pred).all(), (
            "fixture precondition: every row is readable, so the row guard must NOT fire"
        )
        analyzer = FairnessTrainingAnalyzer(X, y, s, task_type="regression")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.evaluate_baseline(y_pred=y_pred)
        assert result.parameters["accuracy_measured"] is False, repr(result.accuracy)
        assert math.isnan(result.accuracy)
        assert "not a finite measurement" in result.parameters["accuracy_not_measured_reason"]
        assert [w for w in caught if "accuracy NOT MEASURED" in str(w.message)]

    def test_control_a_readable_target_still_gets_its_real_score(self):
        """OVER-CORRECTION CONTROL, by value, on both task types.

        Classification on the 240-row two-group fixture measures
        0.9041666666666667 and regression on y = 2x + noise measures
        0.9795765608867427, both with accuracy_measured True and no warning. The
        zero-variance refusal the earlier wave added is untouched: a constant
        target is still NaN with accuracy_measured False.
        """
        X, y, s = _two_group()
        model = LogisticRegression(max_iter=300).fit(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classification = FairnessTrainingAnalyzer(X, y, s).evaluate_baseline(model=model)
        assert classification.accuracy == pytest.approx(0.9041666666666667)
        assert classification.parameters["accuracy_measured"] is True
        assert not [w for w in caught if "accuracy" in str(w.message)]

        rng = np.random.default_rng(5)
        x_reg = rng.normal(size=(_N, 1))
        y_reg = 2 * x_reg[:, 0] + rng.normal(scale=0.3, size=_N)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regression = FairnessTrainingAnalyzer(
                x_reg, y_reg, s, task_type="regression"
            ).evaluate_baseline(y_pred=2 * x_reg[:, 0])
        assert regression.accuracy == pytest.approx(0.9795765608867427)
        assert regression.parameters["accuracy_measured"] is True
        assert not [w for w in caught if "accuracy" in str(w.message)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            constant = FairnessTrainingAnalyzer(
                X, np.full(_N, 0.1), s, task_type="regression"
            ).evaluate_baseline(y_pred=np.full(_N, 2.1))
        assert constant.parameters["accuracy_measured"] is False
        assert math.isnan(constant.accuracy)

    def test_control_non_numeric_labels_are_not_counted_as_unscorable(self):
        """OVER-CORRECTION CONTROL on the guard's own predicate.

        A target of "approved"/"denied" has no finiteness to test. Refusing every
        array this cannot cast to float would refuse every string-labelled
        classification problem in the library, so ``_unscorable_rows`` passes
        over such an array and returns 0.
        """
        labels = np.array(["approved", "denied"] * 3)
        assert _unscorable_rows(labels, labels) == 0
        assert _unscorable_rows(np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])) == 0
        assert _unscorable_rows(np.array([0.0, np.nan]), np.array([1.0, 1.0])) == 1
        assert _unscorable_rows(np.array([0.0, 1.0]), np.array([np.inf, 1.0])) == 1


# ===========================================================================
# G06, FairnessTrainingAnalyzer.compare_methods.
#
# SABOTAGE (run 2026-09-27):
#   S-H  `if self.method_failures and method_comparisons:` in
#        _identify_critical_issues replaced with `if False:`, and the
#        `if self.failed_methods:` block of summary() replaced with
#        `if False:` -> test_two_of_three_trainings_crashing_reaches_the_report
#        failed on the missing issue type; restored, green.
#   S-I  full_analysis's `failed_methods=list(self.method_failures)` replaced
#        with `failed_methods=[]` -> the same test failed with the report
#        carrying no record; restored, green.
# ===========================================================================


def _run_with_failing_methods(crashing):
    """full_analysis with FairClassifier.fit raising for the named methods."""
    X, y, s = _two_group()
    original = FairClassifier.fit

    def flaky(self, *args, **kwargs):
        if self.method in crashing:
            raise RuntimeError("out of memory while fitting")
        return original(self, *args, **kwargs)

    FairClassifier.fit = flaky
    try:
        analyzer = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = analyzer.full_analysis(base_estimator=LogisticRegression(max_iter=300))
        messages = [str(w.message) for w in caught]
    finally:
        FairClassifier.fit = original
    return analyzer, report, messages


class TestG06APartiallyFailedComparisonLeavesNoRecord:
    def test_two_of_three_trainings_crashing_reaches_the_report(self):
        """Both disclosures were gated on a TOTAL failure.

        BEFORE, with FairClassifier.fit raising for threshold and grid_search
        only: method_failures held both, the report held one comparison, the
        issue list was ['High Violation', 'Base Rate Disparity'] (identical to
        the control run in which nothing crashed), the report JSON carried no
        failure record, summary() carried none, and the recommendation named
        Reductions at priority high. FairnessTrainingReport had no field for
        them.

        AFTER: report.failed_methods names both, summary() carries a "METHODS
        THAT COULD NOT BE COMPARED" block naming each with its error, the issue
        list carries "Method(s) Failed To Train" at severity 'unmeasured', the
        action items carry a FIX line, and compare_methods warns "2 of 3".
        """
        analyzer, report, messages = _run_with_failing_methods(("threshold", "grid_search"))

        assert [f["method"] for f in analyzer.method_failures] == ["threshold", "grid_search"]
        assert len(report.method_comparisons) == 1, "fixture precondition"

        assert [f["method"] for f in report.failed_methods] == ["threshold", "grid_search"]
        assert report.to_dict()["failed_methods"] == report.failed_methods
        issues = {i["type"]: i for i in report.critical_issues}
        assert "Method(s) Failed To Train" in issues
        assert issues["Method(s) Failed To Train"]["severity"] == "unmeasured"
        summary = report.summary()
        assert "METHODS THAT COULD NOT BE COMPARED" in summary
        assert "grid_search: RuntimeError: out of memory while fitting" in summary
        assert [item for item in report.action_items if item.startswith("FIX:")]
        assert [m for m in messages if "2 of 3 requested method(s) failed to train" in m]

    def test_a_total_failure_is_still_disclosed_the_way_it_was(self):
        """The half the BGL3 wave fixed must not regress on the way through.

        All three methods crashing still returns no comparison, raises "No
        Method Could Be Compared" and warns "all 3 requested method(s) failed
        to train".
        """
        analyzer, report, messages = _run_with_failing_methods(
            ("reductions", "threshold", "grid_search")
        )
        assert report.method_comparisons == []
        assert len(analyzer.method_failures) == 3
        assert "No Method Could Be Compared" in [i["type"] for i in report.critical_issues]
        assert "Method(s) Failed To Train" not in [i["type"] for i in report.critical_issues]
        assert [m for m in messages if "all 3 requested method(s) failed to train" in m]

    def test_control_a_clean_run_carries_no_failure_record(self):
        """OVER-CORRECTION CONTROL, and the discrimination the audit's own
        assertion lacked: searching the report for the word "fail" now matches
        the empty ``"failed_methods": []`` key on EVERY run, so the control
        asserts the absence of the issue, the block and the warning instead.
        """
        analyzer, report, messages = _run_with_failing_methods(())
        assert analyzer.method_failures == []
        assert report.failed_methods == []
        assert len(report.method_comparisons) == 3
        assert "Method(s) Failed To Train" not in [i["type"] for i in report.critical_issues]
        assert "METHODS THAT COULD NOT BE COMPARED" not in report.summary()
        assert not [item for item in report.action_items if item.startswith("FIX:")]
        assert not [m for m in messages if "failed to train" in m]
        assert report.recommendation.recommended_method not in ("N/A", "Unconstrained")


# ===========================================================================
# G11, baseline_comparison_summary.
#
# SABOTAGE (run 2026-09-27):
#   S-J  _supplied_accuracy's `value = _as_measured(supplied)` replaced with
#        `value = float(supplied)` (the pre-fix coercion) -> 6 of the 7
#        parametrized refusals failed ("cost 0.30000000000000004, verdict
#        '... at a 0.300 accuracy cost.'") and the None case failed with the
#        original TypeError; restored, green.
# ===========================================================================


_Y_TEST = np.array([0, 1, 0, 1, 1, 0])
_PRED_BASELINE = np.array([0, 1, 1, 1, 1, 0])
_PRED_FAIR = np.array([0, 1, 0, 0, 0, 0])
_SENS = {"g": np.array(list("aabbab"))}


def _compare(accuracy_value):
    return baseline_comparison_summary(
        _Y_TEST,
        _PRED_BASELINE,
        _PRED_FAIR,
        _SENS,
        before_metrics={"accuracy": accuracy_value, "demographic_parity_difference": 0.3},
        after_metrics={"accuracy": 0.7, "demographic_parity_difference": 0.1},
    )


class TestG11ASuppliedAccuracyIsCheckedBeforeItIsCoerced:
    @pytest.mark.parametrize(
        "supplied",
        [True, False, "0.9", None, float("nan"), float("inf"), float("-inf")],
        ids=["True", "False", "str", "None", "nan", "inf", "-inf"],
    )
    def test_an_unmeasured_supplied_accuracy_is_refused_not_coerced(self, supplied):
        """``float(...)`` ran before ``_measured`` ever saw the value.

        BEFORE, on six scored rows with a 0.3 to 0.1 disparity drop:
        accuracy=True gave accuracy_measured True, accuracy_cost
        0.30000000000000004 and the headline verdict "Fairness constraints
        reduced disparity by 0.200 at a 0.300 accuracy cost." with no warning;
        accuracy='0.9' gave a cost of 0.2000 the same way; accuracy=None raised
        "TypeError: float() argument must be a string or a real number, not
        'NoneType'", and None is this library's own not-measured sentinel.
        ``summary()`` on FairnessTrainingReport refused all of these on the same
        input through the same ``_measured``, so the two surfaces disagreed.

        AFTER: accuracy_measured False, accuracy_cost NaN, the verdict saying
        "at an accuracy cost that was NOT MEASURED", and a warning naming the
        value.
        """
        with pytest.warns(UserWarning, match="is not a measurement"):
            out = _compare(supplied)
        assert out["accuracy_measured"] is False, (
            "cost " + repr(out["accuracy_cost"]) + ", verdict " + repr(out["verdict"])
        )
        assert math.isnan(out["accuracy_cost"])
        assert "NOT MEASURED" in out["verdict"]
        # The fairness half is measurable on this input and must still be
        # reported: refusing the accuracy must not refuse the disparity.
        assert out["fairness_gain"] == pytest.approx(0.2)
        assert out["fairness_improved"] is True

    @pytest.mark.parametrize(
        "supplied,cost",
        [(0.9, 0.20000000000000007), (np.float32(0.9), 0.19999997615814213), (0.0, -0.7)],
        ids=["float", "np.float32", "measured-zero"],
    )
    def test_control_a_measured_supplied_accuracy_still_costs_what_it_costs(self, supplied, cost):
        """OVER-CORRECTION CONTROL, by value.

        0.9 against the fair model's 0.7 is a cost of 0.20000000000000007; a
        numpy float32 is a measurement and is accepted; and a baseline accuracy
        that really came out 0.0 is a MEASUREMENT, reported as a 0.700 accuracy
        GAIN rather than refused.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _compare(supplied)
        assert out["accuracy_measured"] is True
        assert out["accuracy_cost"] == pytest.approx(cost)
        assert "NOT MEASURED" not in out["verdict"]
        assert not [w for w in caught if "not a measurement" in str(w.message)]

    def test_control_no_accuracy_key_still_scores_the_predictions(self):
        """OVER-CORRECTION CONTROL on the other route into the same field.

        With no 'accuracy' key the function scores the predictions itself: the
        baseline gets 5 of 6 rows right and the fair model 4 of 6, so the cost is
        1/6 and nothing is refused.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = baseline_comparison_summary(
                _Y_TEST,
                _PRED_BASELINE,
                _PRED_FAIR,
                _SENS,
                before_metrics={"demographic_parity_difference": 0.3},
                after_metrics={"demographic_parity_difference": 0.1},
            )
        assert out["accuracy_measured"] is True
        assert out["baseline"]["accuracy"] == pytest.approx(5.0 / 6.0)
        assert out["fair"]["accuracy"] == pytest.approx(4.0 / 6.0)
        assert out["accuracy_cost"] == pytest.approx(1.0 / 6.0)
        assert not [str(w.message) for w in caught]


# ===========================================================================
# G09, FairnessTrainingAnalyzer.generate_recommendation. NO CODE CHANGE.
#
# The behaviour was graded correctly; the EVIDENCE was a control. The named test,
# tests/test_bgl3_in_processing_1.py::TestEveryMethodFailingToTrain::
# test_control_methods_that_train_are_compared_and_ranked, executes only the
# measured path (body lines 885, 891-894, 901-903 of the unit) and never the
# refusal at 936-963, so no change confined to the refusal can redden it.
#
# SABOTAGE (run 2026-09-27), the one the row never had:
#   S-K  the `elif comparisons:` branch's `recommended_method="N/A",
#        priority="low"` replaced with `recommended_method=comparisons[0]
#        .method_name, priority="medium"`.
#          the NAMED test stayed GREEN (1 passed)
#          tests/test_readiness4_constraints.py::TestRecommendation::
#            test_no_winner_is_named_when_nothing_was_evaluated went RED
#            ("AssertionError: assert 'Reductions' == 'N/A'")
#          the two pins below went RED
#        restored, green.
# ===========================================================================


class TestG09TheRefusalIsPinnedByATestThatCanFail:
    def test_no_winner_is_named_when_no_method_reported_a_constraint(self):
        """Three methods TRAINED and none was graded: 'N/A', not a winner.

        'N/A' is the sentinel the rendering layer reads as "no advice was given"
        (adapters_training._recommendation_state), and the rationale names the
        count that was not ranked. Reaching this branch is the whole point:
        ``min(comparisons, key=fairness_violation)`` over NaN violations used to
        return the FIRST method with the rationale "Minimizes constraint
        violation (nan)".
        """
        analyzer = FairnessTrainingAnalyzer(*_two_group())
        comparisons = [
            MethodComparison(
                method_name=name,
                accuracy=0.88,
                fairness_violation=None,
                constraint_satisfied=None,
            )
            for name in ("Reductions", "Threshold", "Grid_Search")
        ]
        recommendation = analyzer.generate_recommendation(comparisons)
        assert recommendation.recommended_method == "N/A"
        assert recommendation.priority == "low"
        assert "3 method(s) reported a constraint result" in recommendation.rationale
        assert recommendation.alternative_methods == []

    def test_an_empty_comparison_list_is_unconstrained_and_not_a_winner(self):
        """The other refusal branch: nothing was compared at all."""
        analyzer = FairnessTrainingAnalyzer(*_two_group())
        recommendation = analyzer.generate_recommendation([])
        assert recommendation.recommended_method == "Unconstrained"
        assert recommendation.priority == "low"
        assert recommendation.rationale == "No fairness methods evaluated"

    def test_control_a_graded_comparison_still_names_a_winner(self):
        """OVER-CORRECTION CONTROL, by value. A measured comparison must still
        rank: the satisfying method with the best accuracy wins at priority
        high, and the ungraded note is absent when every method was graded.
        """
        analyzer = FairnessTrainingAnalyzer(*_two_group())
        comparisons = [
            MethodComparison("Reductions", 0.81, 0.02, True),
            MethodComparison("Threshold", 0.879, 0.01, True),
            MethodComparison("Grid_Search", 0.75, 0.30, False),
        ]
        recommendation = analyzer.generate_recommendation(comparisons)
        assert recommendation.recommended_method == "Threshold"
        assert recommendation.priority == "high"
        assert "0.879" in recommendation.rationale
        assert "not ranked" not in recommendation.rationale
        assert recommendation.alternative_methods == ["Reductions"]
