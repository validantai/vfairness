"""Wave-6 release-pass pins: the undefined-rate / could-not-check handling that
the gate.py fix (wave-6) left un-mirrored in its two sibling call sites, plus the
gate's silent approval of an unevaluable configured metric.

Findings (second-iteration-audit-2026-08-22, unit operations-cicd-mon-rep, and the
gate-report unit's REPORTED-NOT-FIXED note):
- testing.py FairnessTestSuite mapped a NaN metric to FAILED with the misleading
  message "nan exceeds threshold"; a could-not-measure must be SKIPPED (three
  states, never two), and its _compute_default_metrics silently dropped a group
  with an undefined rate.
- monitor.py FairnessMonitor carried the same silent-drop copy.
- gate.py approved deployment when a configured metric was MISSING from the dict
  (only warned + continued), so a blocking metric that could not be computed still
  let the gate APPROVE.
"""

import math

import numpy as np

from vfairness.operations.cicd.gate import GateConfig, GateStatus, ModelFairnessGate
from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig
from vfairness.operations.cicd.testing import FairnessTestSuite, TestStatus


def _one_group_no_positives(n=120):
    """Two groups; group B has no positives (y_true all 0), so any TPR-based
    metric is undefined for B."""
    y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.zeros(n, dtype=int)])
    y_pred = np.concatenate([np.array([1] * (n // 2) + [0] * (n // 2)), np.zeros(n, dtype=int)])
    attr = np.array(["A"] * n + ["B"] * n)
    return y_true, y_pred, attr


class TestTestingSuiteNaN:
    def test_nan_metric_is_skipped_not_failed(self):
        y_true, y_pred, attr = _one_group_no_positives()
        suite = FairnessTestSuite(
            protected_attributes=["group"],
            metrics=["equalized_odds_difference"],
            thresholds={"equalized_odds_difference": 0.1},
        )
        results = suite.test_predictions(y_true, y_pred, attr, raise_on_failure=False)
        eo = [r for r in results if r.metric_name == "equalized_odds_difference"]
        assert eo, "metric should appear in results"
        # Could-not-measure is SKIPPED, never FAILED with a "nan exceeds" message.
        assert eo[0].status == TestStatus.SKIPPED
        assert "nan exceeds" not in (eo[0].message or "").lower()

    def test_compute_default_reports_nan_not_dropped_group(self):
        y_true, y_pred, attr = _one_group_no_positives()
        suite = FairnessTestSuite(
            protected_attributes=["group"], metrics=["equalized_odds_difference"], thresholds={}
        )
        m = suite._compute_default_metrics(y_true, y_pred, attr)
        assert "equalized_odds_difference" in m
        assert math.isnan(m["equalized_odds_difference"])

    def test_measurable_metric_still_evaluated(self):
        # Does-not-overcorrect: a fully measurable failing metric still FAILS.
        n = 100
        y_true = np.array([1, 0] * n)
        y_pred = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        attr = np.array(["A"] * n + ["B"] * n)
        suite = FairnessTestSuite(
            protected_attributes=["group"],
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        results = suite.test_predictions(y_true, y_pred, attr, raise_on_failure=False)
        dp = [r for r in results if r.metric_name == "demographic_parity_difference"][0]
        assert dp.status == TestStatus.FAILED


class TestMonitorNaN:
    def test_undefined_tpr_reports_nan_not_zero(self):
        y_true, y_pred, attr = _one_group_no_positives()
        mon = BiasMonitor(config=MonitorConfig(min_samples_for_alert=1))
        m = mon._compute_default_metrics(y_true, y_pred, attr)
        assert math.isnan(m.get("equalized_odds_difference", float("nan")))
        # log_batch must not crash on the NaN metric.
        mon.log_batch(y_pred, y_true, attr)


class TestGateMissingAndNaN:
    def test_blocking_metric_that_cannot_be_computed_blocks(self):
        # Single group -> _compute_default_metrics returns {} -> the configured
        # blocking metric is missing. The gate must NOT approve.
        y_true = np.array([1, 0] * 60)
        y_pred = np.array([1, 0] * 60)
        attr = np.array(["A"] * 120)
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.1},
            )
        )
        decision = gate.evaluate(y_true, y_pred, attr)
        assert not decision.approved
        assert decision.status == GateStatus.BLOCKED

    def test_nan_blocking_metric_blocks(self):
        y_true, y_pred, attr = _one_group_no_positives()
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=["equalized_odds_difference"],
                thresholds={"equalized_odds_difference": 0.1},
            )
        )
        decision = gate.evaluate(y_true, y_pred, attr)
        assert not decision.approved

    def test_clean_measurable_pass_still_approves(self):
        # Does-not-overcorrect: a fair, fully measurable model is still APPROVED.
        rng = np.random.default_rng(0)
        n = 400
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        attr = np.array(["A", "B"] * (n // 2))
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.2},
            )
        )
        decision = gate.evaluate(y_true, y_pred, attr)
        assert decision.approved
