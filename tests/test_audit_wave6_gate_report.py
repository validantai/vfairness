"""Wave-6 audit pins: gate-report unit.

Findings fixed (docs/audits/second-iteration-audit-2026-08-22.md):

1. ModelFairnessGate approved deployment when a configured metric was NaN
   (evaluate, evaluate_from_metrics, evaluate_hierarchical): NaN failed every
   comparison, so `passed` stayed True and the gate returned APPROVED. A NaN
   metric now fails closed into an explicit insufficient-evidence outcome.
2. ModelFairnessGate._compute_default_metrics silently dropped groups with an
   undefined rate (empty denominator) and reported a 0.0 gap over the
   survivors, so a candidate whose protected group was unmeasurable sailed
   through the gate. An undefined per-group rate now yields a NaN metric,
   which the gate refuses via fix 1.
3. regression_fairness_report certified fairness_score 1.0 with a clean
   "4/4 metrics within thresholds" summary on degenerate single-valid-group
   data; the assessable guard existed only in the classification report. It
   now mirrors the classification guard (assessment["assessable"] plus a
   NOT ASSESSABLE summary).
4. calibration_difference returned 0.0 (rendered PASS) when fewer than two
   groups had a computable ECE, while auroc_parity on the same data said
   NOT_ASSESSABLE. It now returns NaN, mirroring auroc_parity (no existing
   test pins 0.0 for the single-qualifying-group case of this metric).

Negative cases (the exact scenarios that used to produce the wrong result)
come first, then does-not-overcorrect cases.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    auroc_parity,
    calibration_difference,
)
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)
from vfairness.operations.cicd.gate import GateStatus, ModelFairnessGate


def _dp_gate(**kwargs):
    return ModelFairnessGate(
        metrics=["demographic_parity_difference"],
        thresholds={"demographic_parity_difference": 0.1},
        **kwargs,
    )


def _balanced_two_groups(n=50):
    y = np.array([1, 0] * n)
    prot = np.array(["A"] * n + ["B"] * n)
    return y, prot


# ── Finding 1 negative cases: NaN metric must never be APPROVED ─────────────


class TestGateNaNFailsClosed:
    def test_evaluate_from_metrics_nan_is_blocked(self):
        # The recorded reproduction: a pre-computed NaN metric used to return
        # GateStatus.APPROVED with zero warnings and zero blocking reasons.
        decision = _dp_gate().evaluate_from_metrics({"demographic_parity_difference": float("nan")})
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert any("insufficient evidence" in r for r in decision.blocking_reasons)
        # The metric evaluation is present and explicitly not passed.
        (ev,) = decision.metric_evaluations
        assert ev.passed is False
        assert "could not be measured" in ev.message

    def test_evaluate_with_nan_compute_fn_is_blocked(self):
        y, prot = _balanced_two_groups()
        gate = _dp_gate(
            compute_metrics_fn=lambda yt, yp, pa: {"demographic_parity_difference": float("nan")}
        )
        decision = gate.evaluate(y, y, prot)
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert any("insufficient evidence" in r for r in decision.blocking_reasons)

    def test_evaluate_hierarchical_with_nan_is_blocked(self):
        y, prot = _balanced_two_groups()
        gate = _dp_gate(
            compute_metrics_fn=lambda yt, yp, pa: {"demographic_parity_difference": float("nan")}
        )
        decision = gate.evaluate_hierarchical(y, y, {"g": prot})
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED

    def test_numpy_float32_nan_also_fails_closed(self):
        decision = _dp_gate().evaluate_from_metrics(
            {"demographic_parity_difference": np.float32("nan")}
        )
        assert decision.approved is False

    def test_nonblocking_nan_downgrades_to_conditional_with_warning(self):
        # Non-blocking failures do not block by contract; a non-blocking NaN
        # must still never yield a clean APPROVED. It surfaces as CONDITIONAL
        # with an explicit could-not-check warning.
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
            blocking_metrics=[],
        )
        decision = gate.evaluate_from_metrics({"demographic_parity_difference": float("nan")})
        assert decision.status is GateStatus.CONDITIONAL
        assert any("could not be measured" in w for w in decision.warnings)


# ── Finding 2 negative case: undefined group rate must not read 0.0 ─────────


class TestGateDefaultMetricsUndefinedRate:
    @staticmethod
    def _undefined_fpr_data():
        # Group A: TP=25 FN=25 FP=25 TN=25 (FPR 0.5, TPR 0.5).
        # Group B: all y_true=1 (FPR undefined), TPR 0.5 equal to A's, so the
        # only disparity signal is the unmeasurable FPR leg.
        yA_true = np.array([1] * 50 + [0] * 50)
        yA_pred = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        yB_true = np.array([1] * 100)
        yB_pred = np.array([1] * 50 + [0] * 50)
        y_true = np.concatenate([yA_true, yB_true])
        y_pred = np.concatenate([yA_pred, yB_pred])
        prot = np.array(["A"] * 100 + ["B"] * 100)
        return y_true, y_pred, prot

    def test_undefined_fpr_yields_nan_not_zero(self):
        y_true, y_pred, prot = self._undefined_fpr_data()
        gate = ModelFairnessGate.__new__(ModelFairnessGate)  # method needs no state
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert np.isnan(metrics["false_positive_rate_difference"])

    def test_gate_refuses_approval_on_unmeasurable_fpr(self):
        # End-to-end through the real decision path: this exact input used to
        # return approved=True with false_positive_rate_difference == 0.0.
        y_true, y_pred, prot = self._undefined_fpr_data()
        gate = ModelFairnessGate(
            metrics=["false_positive_rate_difference"],
            thresholds={"false_positive_rate_difference": 0.1},
        )
        decision = gate.evaluate(y_true, y_pred, prot)
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert any("insufficient evidence" in r for r in decision.blocking_reasons)

    def test_undefined_tpr_leg_yields_nan_equalized_odds(self):
        # Group B has no positives: the TPR leg is undefined.
        y_true = np.concatenate([np.array([1] * 50 + [0] * 50), np.zeros(100, dtype=int)])
        y_pred = np.concatenate([np.array([1] * 25 + [0] * 75), np.array([1] * 30 + [0] * 70)])
        prot = np.array(["A"] * 100 + ["B"] * 100)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert np.isnan(metrics["equalized_odds_difference"])


# ── Finding 3 negative case: regression report on single-group data ─────────


class TestRegressionReportDegenerateGuard:
    @staticmethod
    def _single_valid_group_report():
        rng = np.random.default_rng(7)
        n_big, n_tiny = 100, 5
        y_true = rng.normal(10, 2, n_big + n_tiny)
        y_pred = y_true + rng.normal(0, 0.5, n_big + n_tiny)
        grp = np.array(["big"] * n_big + ["tiny"] * n_tiny)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return regression_fairness_report(y_true, y_pred, grp)

    def test_single_valid_group_is_marked_not_assessable(self):
        # This exact scenario used to certify fairness_score 1.0 with the
        # summary "4/4 metrics within thresholds" and no assessable key.
        report = self._single_valid_group_report()
        a = report["assessment"]
        assert a["assessable"] is False
        assert a["summary"].startswith("NOT ASSESSABLE")
        assert "do not certify fairness" in a["summary"]

    def test_excluded_group_still_surfaced_when_not_assessable(self):
        a = self._single_valid_group_report()["assessment"]
        assert [g["group"] for g in a["insufficient_evidence_groups"]] == ["tiny"]


# ── Finding 4 negative case: calibration_difference on one valid group ──────


class TestCalibrationDifferenceUnmeasurable:
    @staticmethod
    def _single_valid_group_data():
        rng = np.random.default_rng(11)
        n_a, n_b = 200, 10  # B below default min_group_size=30
        y_true = np.concatenate([(rng.random(n_a) < 0.1).astype(int), rng.integers(0, 2, n_b)])
        y_prob = np.concatenate([np.full(n_a, 0.9), rng.random(n_b)])
        y_pred = (y_prob >= 0.5).astype(int)
        attr = np.array(["A"] * n_a + ["B"] * n_b)
        return y_true, y_pred, attr, y_prob

    def test_single_qualifying_group_returns_nan(self):
        # Group A is badly miscalibrated (p=0.9 vs event rate 0.1) and is the
        # only group above min_group_size: the between-group difference is
        # unmeasurable. This used to return 0.0, rendering PASS while
        # auroc_parity on the same data said NOT_ASSESSABLE.
        y_true, y_pred, attr, y_prob = self._single_valid_group_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = calibration_difference(y_true, y_pred, attr, y_prob)
            auroc = auroc_parity(y_true, y_prob, attr)
        assert np.isnan(value)
        assert np.isnan(auroc)  # the two siblings now agree

    def test_report_routes_it_to_not_assessable(self):
        y_true, y_pred, attr, y_prob = self._single_valid_group_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_fairness_report(y_true, y_pred, attr, y_prob)
        a = report["assessment"]
        routed = {m["metric"] for m in a["not_assessable_metrics"]}
        assert "calibration_difference" in routed
        assert "calibration_difference" not in {m["metric"] for m in a["passed_metrics"]}


# ── Does-not-overcorrect: well-defined data keeps its old behaviour ─────────


class TestNoOvercorrection:
    def test_gate_still_approves_fair_model(self):
        y, prot = _balanced_two_groups()
        decision = _dp_gate().evaluate(y, y, prot)  # identical rates: gap 0.0
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED

    def test_gate_still_blocks_real_gap(self):
        y_true = np.array([1, 0] * 50)
        y_pred = np.array([1] * 50 + [0] * 50)  # A always 1, B always 0
        prot = np.array(["A"] * 50 + ["B"] * 50)
        decision = _dp_gate().evaluate(y_true, y_pred, prot)
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED

    def test_default_metrics_unchanged_when_all_rates_defined(self):
        # The wave-1 three-group pin: A/B identical, C never selected.
        prot = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
        y_pred = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [0] * 40)
        y_true = np.array([1, 0] * 60)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert metrics["demographic_parity_difference"] == pytest.approx(0.5)

    def test_regression_report_two_valid_groups_still_assessable(self):
        rng = np.random.default_rng(3)
        n = 60
        y_true = rng.normal(10, 2, 2 * n)
        y_pred = y_true + rng.normal(0, 0.5, 2 * n)
        grp = np.array(["a"] * n + ["b"] * n)
        report = regression_fairness_report(y_true, y_pred, grp)
        a = report["assessment"]
        assert a["assessable"] is True
        assert "metrics within thresholds" in a["summary"]
        assert not a["summary"].startswith("NOT ASSESSABLE")

    def test_calibration_difference_two_groups_still_measures(self):
        rng = np.random.default_rng(5)
        n = 200
        # A well calibrated, B badly miscalibrated: a real, finite ECE gap.
        y_true = np.concatenate(
            [(rng.random(n) < 0.5).astype(int), (rng.random(n) < 0.1).astype(int)]
        )
        y_prob = np.concatenate([np.full(n, 0.5), np.full(n, 0.9)])
        y_pred = (y_prob >= 0.5).astype(int)
        attr = np.array(["A"] * n + ["B"] * n)
        value = calibration_difference(y_true, y_pred, attr, y_prob)
        assert np.isfinite(value)
        assert value > 0.3  # B's ECE is ~0.8, A's is ~0.0
