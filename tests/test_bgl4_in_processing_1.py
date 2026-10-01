"""BGL4 audit of batch A-in_processing-1: the overturns, as executable evidence.

Written by the AUDITOR, not by the wave that produced the grades. Every test
below asserts the behaviour the grade CLAIMS and was marked ``xfail(strict=True)``
because the behaviour was not there.

CLOSED 2026-09-27 (BGL5, batch A-in_processing-1). All six defects are fixed, so
every marker below is GONE and every assertion now passes as written: the
assertion was kept and the marker removed, never the other way round. The
auditor's reason text has been moved into each test's docstring as the measured
BEFORE, so the evidence of what the code used to do survives the fix. Two
mechanisms changed where the fix chose a different honest state from the one the
auditor guessed at, and each says so in its own docstring:

  * the undefined R^2 now leaves ``full_analysis`` as ``None`` rather than NaN,
    because None is the sentinel the rendering layer already speaks
  * searching the report for the word "fail" no longer discriminates, since the
    new ``"failed_methods": []`` key matches on a clean run too, so the partial
    failure is asserted by the issue type and the named methods instead

The pins written WITH the fixes, each sabotage-verified, are in
tests/test_bgl5_in_processing_1.py.

Each class names the grade it refutes and the input class the graded pin never
tried.
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

from vfairness.in_processing.analyzer import FairnessTrainingAnalyzer
from vfairness.in_processing.calibrators.group_calibrators import (
    CalibrationAwareTrainer,
    TrainableGroupCalibrator,
)
from vfairness.in_processing.loss_functions.base import BaseFairnessLoss
from vfairness.in_processing.wrappers import FairClassifier

_N = 240


def _two_group(seed: int = 11):
    rng = np.random.default_rng(seed)
    s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
    x0 = rng.normal(loc=np.where(s == "a", 1.2, -1.2), scale=0.7)
    X = np.column_stack([x0, rng.normal(size=_N)])
    y = (x0 + rng.normal(scale=0.4, size=_N) > 0).astype(int)
    return X, y, s


# ===========================================================================
# Grade 7, evaluate_baseline, claimed PROVEN.
# The guard is keyed on np.ptp / np.var of the target, and BOTH are NaN when the
# target carries a NaN, so the refusal cannot fire on the second unmeasurable
# input. accuracy comes back NaN wearing accuracy_measured=True, in silence.
# ===========================================================================


class TestG07AnUnreadableTargetIsNotAMeasuredAccuracy:
    def test_a_regression_target_carrying_a_nan_is_not_a_measured_r2(self):
        """BEFORE (BGL4 audit): np.ptp(nan-bearing y) is nan and nan == 0.0 is
        False, so the zero-variance guard cannot fire; accuracy is NaN with
        parameters['accuracy_measured'] True and no warning.

        AFTER (BGL5): accuracy NaN with accuracy_measured False and the warning
        "accuracy NOT MEASURED (1 of 240 row(s) carry a non-finite target or
        prediction ...)", from a row count taken before either branch.
        """
        X, _, s = _two_group()
        y = np.arange(_N, dtype=float)
        y[3] = np.nan
        an = FairnessTrainingAnalyzer(X, y, s, task_type="regression")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = an.evaluate_baseline(y_pred=np.arange(_N, dtype=float))
        assert result.parameters["accuracy_measured"] is False, (
            "accuracy is " + repr(result.accuracy) + " and the flag beside it says measured"
        )
        assert [w for w in caught if "accuracy" in str(w.message)]

    def test_a_classification_target_of_all_nan_is_not_a_measured_zero(self):
        """BEFORE (BGL4 audit): the classification branch hardcodes
        accuracy_measured=True, so 240 unscorable rows come back as a measured
        accuracy of 0.0, the score of a model that got every row wrong.

        AFTER (BGL5): accuracy NaN with accuracy_measured False. The flag is
        derived from the row count now and is no longer a literal on this path.
        """
        X, _, s = _two_group()
        y = np.full(_N, np.nan)
        an = FairnessTrainingAnalyzer(X, y, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = an.evaluate_baseline(y_pred=np.full(_N, np.nan))
        assert result.parameters["accuracy_measured"] is False, (
            "accuracy came back " + repr(result.accuracy) + " with accuracy_measured True"
        )

    def test_the_undefined_r2_is_not_drawn_as_a_number_a_reader_can_read(self):
        """BEFORE (BGL4 audit): the NaN the fix reports does not survive to
        either renderer. full_analysis copies three fields out of the baseline
        MethodComparison and drops parameters['accuracy_measured'], and
        adapters_training._baseline_state then derives baseline_measured from
        ``accuracy is not None``, so NaN is drawn as a measurement.

        AFTER (BGL5): full_analysis carries the state THROUGH, as the None the
        rendering layer already speaks, plus accuracy_measured and the reason.
        The precondition below therefore asserts None where the auditor's version
        asserted NaN: that is the MECHANISM changing, not the subject. The subject
        is the two canvases, and they are asserted unchanged.
        """
        from vfairness.rendering.adapters_training import (
            training_analysis_report_to_svg,
            training_report_to_svg,
        )

        rng = np.random.default_rng(3)
        X = rng.normal(size=(_N, 2))
        s = np.array(["a"] * (_N // 2) + ["b"] * (_N // 2))
        an = FairnessTrainingAnalyzer(X, np.full(_N, 3.0), s, task_type="regression")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = an.full_analysis(base_estimator=LinearRegression())
        assert report.baseline_metrics["accuracy"] is None, "fixture precondition"
        assert report.baseline_metrics["accuracy_measured"] is False

        for render in (training_analysis_report_to_svg, training_report_to_svg):
            svg = render(report)
            drawn = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
            assert "nan" not in [t.strip().lower() for t in drawn], (
                render.__name__ + " draws a bare 'nan' where a not-measured state belongs"
            )
            desc = re.search(r"<desc>(.*?)</desc>", svg, re.S)
            assert desc is not None and "accuracy nan" not in desc.group(1).lower(), (
                render.__name__ + " puts 'accuracy nan' in the accessible description"
            )


# ===========================================================================
# Grade 6, compare_methods, claimed PROVEN.
# The fix discloses only when EVERY method failed: both the summary warning and
# the critical issue are guarded by `not comparisons` / `not method_comparisons`,
# and FairnessTrainingReport has no field for the failures at all.
# ===========================================================================


class TestG06APartiallyFailedComparisonLeavesNoRecord:
    def test_two_of_three_trainings_crashing_reaches_the_report(self):
        """BEFORE (BGL4 audit): with 2 of 3 methods crashing, the report carries
        one comparison, no critical issue and no failure record, and the
        recommendation names a winner at priority high. The disclosure is gated on
        ``not comparisons``, so only a TOTAL failure is reported.

        AFTER (BGL5): FairnessTrainingReport.failed_methods carries both, summary()
        prints a "METHODS THAT COULD NOT BE COMPARED" block, and
        _identify_critical_issues raises "Method(s) Failed To Train".

        The auditor's assertion was ``"fail" in report.summary() +
        repr(report.to_dict())``, and that no longer DISCRIMINATES: the new
        ``"failed_methods": []`` key puts the word "fail" in a clean run's JSON
        too, so it would pass with the defect back. The subject is kept and the
        mechanism sharpened to the issue type and the failed method names, which
        a clean run does not have (asserted in
        tests/test_bgl5_in_processing_1.py::
        TestG06APartiallyFailedComparisonLeavesNoRecord::
        test_control_a_clean_run_carries_no_failure_record).
        """
        X, y, s = _two_group()
        original = FairClassifier.fit

        def flaky(self, *args, **kwargs):
            if self.method in ("threshold", "grid_search"):
                raise RuntimeError("out of memory while fitting")
            return original(self, *args, **kwargs)

        FairClassifier.fit = flaky
        try:
            an = FairnessTrainingAnalyzer(X, y, s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = an.full_analysis(base_estimator=LogisticRegression(max_iter=300))
        finally:
            FairClassifier.fit = original

        assert len(an.method_failures) == 2, "fixture precondition"
        blob = report.summary() + repr(report.to_dict())
        assert "Method(s) Failed To Train" in [i["type"] for i in report.critical_issues], (
            "the report of a run in which 2 of 3 trainings crashed is "
            "indistinguishable from one where only 1 method was requested; "
            "issues were " + repr([i["type"] for i in report.critical_issues])
        )
        for method in ("threshold", "grid_search"):
            assert method in blob, (
                "the crashed method " + method + " is named nowhere in the report"
            )
        assert [f["method"] for f in report.failed_methods] == ["threshold", "grid_search"]


# ===========================================================================
# Grade 14, TrainableGroupCalibrator.calibration_loss, claimed PROVEN.
# Two inputs the five pinned tests never tried.
# ===========================================================================


class TestG14TheRefusalIsReachableOnEveryInputItsDocstringNames:
    def test_min_group_size_zero_does_not_score_a_diverged_batch_as_perfect(self):
        """BEFORE (BGL4 audit): the refusal is ``int(mask.sum()) < min_group_size``
        and 0 usable rows is not < 0, so min_group_size=0 makes the NaN refusal
        unreachable and an all-NaN batch scores 0.0, which on this scale is
        perfect calibration.

        AFTER (BGL5): the refusal also fires on zero usable rows, at any
        threshold, and this returns NaN with the same two warnings the default
        min_group_size=10 produced.
        """
        torch.manual_seed(0)
        calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
        logits = torch.full((60, 1), float("nan"))
        y = (torch.arange(60) % 2).float()
        groups = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(calibrator.calibration_loss(logits, y, groups, min_group_size=0))
        assert math.isnan(value), "returned " + repr(value) + " for a batch with no usable row"

    def test_an_empty_batch_is_refused_and_not_a_traceback(self):
        """BEFORE (BGL4 audit): the docstring names 'an empty batch' as a case the
        NaN refusal covers. The BGL3 fix's own _finite_rows reshapes (0, 1) to
        (0, -1), which torch refuses, so the empty batch now RAISES. Removing the
        usable mask restores the documented NaN, so the fix caused this.

        AFTER (BGL5): _finite_rows skips the reduction when there is no element to
        reduce, so the documented NaN is back, and it is back WITH the non-finite
        row mask rather than instead of it.
        """
        calibrator = TrainableGroupCalibrator(n_groups=2, method="temperature")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(
                calibrator.calibration_loss(
                    torch.zeros(0, 1), torch.zeros(0), torch.zeros(0, dtype=torch.long)
                )
            )
        assert math.isnan(value)

    def test_the_trainer_surfaces_refuse_an_empty_batch_too(self):
        """BEFORE (BGL4 audit), same root cause one layer up: train_step and
        fine_tune_calibration promise NaN when the calibrator could not measure,
        and both raise RuntimeError on an empty batch instead.

        AFTER (BGL5): both return NaN, from the one fix in _finite_rows.
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
            assert math.isnan(out["calibration_loss"])
            assert math.isnan(trainer.fine_tune_calibration([empty], optimizer))


# ===========================================================================
# Grade 21, BaseFairnessLoss.forward, claimed PROVEN.
# _task_loss_coverage reads the weights per ROW only when
# `weights.numel() % n_rows == 0`. A scalar weight has numel 1, 1 % 4 is 1, so
# the count is left at n_rows and a batch every row of which was multiplied by
# zero is reported as a MEASURED task loss of 0.0.
# ===========================================================================


class _Dummy(BaseFairnessLoss):
    def _compute_fairness_penalty(self, y_pred, y_true, sensitive_attr):
        return torch.tensor(0.25)


class TestG21AScalarZeroWeightIsNotAMeasuredTaskLoss:
    @pytest.mark.parametrize(
        "weight", [torch.zeros(1), torch.tensor(0.0)], ids=["numel-1", "0-dim"]
    )
    def test_a_broadcast_zero_weight_is_not_a_perfect_fit(self, weight):
        """BEFORE (BGL4 audit): a broadcast scalar sample_weight of 0.0 leaves
        task_rows_used at the full row count, so task_loss 0.0 is reported with
        task_loss_assessed True and no warning; the pinned torch.zeros(4) case is
        correct, so the guard is keyed on the weight's SHAPE, not on its effect.

        AFTER (BGL5): a one-element weight is judged by its EFFECT, so both
        spellings report task_loss NaN with task_rows_used 0 and the warning, and
        a live one-element weight still measures (see the control in
        tests/test_bgl5_in_processing_1.py).
        """
        y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sensitive = torch.tensor([0.0, 0.0, 1.0, 1.0])
        loss = _Dummy(lambda_fairness=0.1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss(
                y_pred, y_true, sensitive, sample_weight=weight, return_components=True
            )
        assert components.batch_metrics["task_loss_assessed"] is False, (
            "task_loss "
            + repr(components.task_loss)
            + " with task_rows_used "
            + repr(components.batch_metrics["task_rows_used"])
            + " and warnings "
            + repr([str(w.message)[:60] for w in caught])
        )
        assert math.isnan(components.task_loss)

    def test_an_epoch_of_scalar_weighted_out_batches_is_not_a_completed_fit(self):
        """BEFORE (BGL4 audit), the epoch this rolls up into: three batches every
        row of which was weighted out average to avg_task_loss 0.0, the 'completed
        epoch over nothing' _task_loss_coverage's own docstring names.

        AFTER (BGL5): avg_task_loss NaN, while avg_total_loss stays finite because
        it describes the steps the optimizer really took.
        """
        y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sensitive = torch.tensor([0.0, 0.0, 1.0, 1.0])
        loss = _Dummy(lambda_fairness=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                loss(y_pred, y_true, sensitive, sample_weight=torch.zeros(1))
            metrics = loss.end_epoch()
        assert metrics is not None
        assert math.isnan(metrics.avg_task_loss), "avg_task_loss came back " + repr(
            metrics.avg_task_loss
        )


# ===========================================================================
# Grade 11, baseline_comparison_summary, claimed PROVEN.
# The empty-test-set half is genuinely fixed. The OTHER way an accuracy arrives,
# `before_metrics["accuracy"]`, is read as `float(...)` BEFORE `_measured` sees
# it, so the module's own bool guard ("a bool is not a magnitude: True would
# print as 1.0000") cannot fire, and None, the library's own not-measured
# sentinel, raises instead of being refused.
# ===========================================================================


class TestG11ASuppliedAccuracyIsCheckedBeforeItIsCoerced:
    def test_a_bool_accuracy_is_not_a_measured_cost(self):
        """BEFORE (BGL4 audit): float(True) is 1.0, so a bool in the accuracy slot
        is laundered past _measured and the headline verdict reads 'at a 0.300
        accuracy cost'. summary() refuses the same value, so the two surfaces
        disagree on one input.

        AFTER (BGL5): the supplied value goes through _as_measured BEFORE any
        coercion, so the bool is refused with a warning naming it, and the cost is
        NaN.
        """
        from vfairness.in_processing.analyzer import baseline_comparison_summary

        y_test = np.array([0, 1, 0, 1, 1, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = baseline_comparison_summary(
                y_test,
                np.array([0, 1, 1, 1, 1, 0]),
                np.array([0, 1, 0, 0, 0, 0]),
                {"g": np.array(list("aabbab"))},
                before_metrics={"accuracy": True, "demographic_parity_difference": 0.3},
                after_metrics={"accuracy": 0.7, "demographic_parity_difference": 0.1},
            )
        assert out["accuracy_measured"] is False, (
            "cost " + repr(out["accuracy_cost"]) + ", verdict " + repr(out["verdict"])
        )
        assert [w for w in caught if "accuracy" in str(w.message)]

    def test_a_none_accuracy_is_refused_and_not_a_traceback(self):
        """BEFORE (BGL4 audit): None is this library's not-measured sentinel and
        float(None) raises TypeError, so the one value most likely to arrive in an
        unmeasured accuracy slot is a traceback rather than the third state.

        AFTER (BGL5): None is refused like every other non-measurement, and the
        function returns accuracy_measured False with a NaN cost.
        """
        from vfairness.in_processing.analyzer import baseline_comparison_summary

        y_test = np.array([0, 1, 0, 1, 1, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = baseline_comparison_summary(
                y_test,
                np.array([0, 1, 1, 1, 1, 0]),
                np.array([0, 1, 0, 0, 0, 0]),
                {"g": np.array(list("aabbab"))},
                before_metrics={"accuracy": None, "demographic_parity_difference": 0.3},
                after_metrics={"accuracy": 0.7, "demographic_parity_difference": 0.1},
            )
        assert out["accuracy_measured"] is False
        assert math.isnan(out["accuracy_cost"])
