"""READINESS-6: the deployment-blocking surfaces that said PASS for something
they never checked.

Nine defects, each reproduced by execution before it was fixed, each one making
a gate answer APPROVED / PASS / OK / exit 0 for a comparison that never
happened.

Every defect is pinned TWICE, and the second half is not optional:

- a REFUSAL test, which fails if the fail-open behaviour comes back;
- an OVER-CORRECTION CONTROL, which fails if the fix starts refusing work it
  should approve.

A gate that blocks every deployment is as useless as one that approved
everything, and two of these fixes are one careless edit away from that: a
falsiness test instead of ``is None`` would silently disable a legitimate
zero-tolerance threshold of 0.0, and a ``>= 0`` degradation guard would block
every gate that supplies a baseline on floating-point noise.

Label arrays are built with ``dtype=object``. A numpy ``'<U5'`` string array
truncates silently, which is its own way of comparing groups that do not exist.
"""

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError as MetricsFairnessAssertionError,
)
from vfairness.evaluation.vfairness_metrics.integrations import assert_fairness
from vfairness.mcp import tools as mcp_tools
from vfairness.operations.cicd.gate import (
    GateConfig,
    GateStatus,
    ModelFairnessGate,
)
from vfairness.operations.cicd.monitor import BiasMonitor, DriftType, MonitorConfig
from vfairness.operations.cicd.precommit import check_fairness_config
from vfairness.operations.cicd.quality_report import build_quality_report
from vfairness.operations.cicd.testing import (
    FairnessAssertionError,
    parametrize_fairness,
)
from vfairness.operations.cicd.validator import (
    DataBiasValidator,
    DataValidationConfig,
)

# Fixtures


def _rate_data(spec):
    """Build labels with an exact per-group TPR and FPR.

    ``spec`` is ``[(group, tpr, fpr), ...]``; each group gets 50 positives and
    50 negatives, which is above the gate's ``min_group_size`` of 30 so the
    small-sample refusal cannot be what a test is measuring.
    """
    y_true, y_pred, protected = [], [], []
    for group, tpr, fpr in spec:
        for i in range(50):
            y_true.append(1)
            y_pred.append(1 if i < round(tpr * 50) else 0)
            protected.append(group)
        for i in range(50):
            y_true.append(0)
            y_pred.append(1 if i < round(fpr * 50) else 0)
            protected.append(group)
    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(protected, dtype=object),
    )


def _grossly_unfair():
    """TPR 1.00 vs 0.10 and FPR 0.00 vs 0.90.

    equalized_odds_difference is 0.90. demographic_parity_difference is 0.00,
    because both groups are selected at 50 percent overall: the ONLY metric that
    can catch this model is the one the docstring example left unbounded.
    """
    return _rate_data([("a", 1.00, 0.00), ("b", 0.10, 0.90)])


def _genuinely_fair():
    return _rate_data([("a", 0.80, 0.20), ("b", 0.80, 0.20)])


# Defect 1: a metric with no threshold was never compared, and `passed = True`
# was left standing


class TestGateUnconfiguredThreshold:
    def test_unbounded_metric_is_refused_not_approved(self):
        """The configuration copied verbatim from ModelFairnessGate's own class
        docstring, as it stood: two metrics, one threshold. It returned
        approved=True with equalized_odds_difference at 0.9000 written into the
        markdown audit trail as "Pass"."""
        y_true, y_pred, protected = _grossly_unfair()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference", "equalized_odds_difference"],
            thresholds={"demographic_parity_difference": 0.1},
            require_improvement=True,
        )
        decision = gate.evaluate(y_true, y_pred, protected)

        eo = [
            e for e in decision.metric_evaluations if e.metric_name == "equalized_odds_difference"
        ][0]
        assert eo.value == pytest.approx(0.9)
        assert eo.threshold is None
        assert not eo.passed
        assert "no threshold" in eo.message

        assert not decision.approved
        assert decision.status is GateStatus.BLOCKED
        assert gate.create_github_check(decision)["conclusion"] != "success"

    @pytest.mark.parametrize(
        "thresholds",
        [
            pytest.param({}, id="empty-thresholds"),
            pytest.param({"equalized_odds_difference": None}, id="key-present-holding-None"),
            pytest.param({"predictive_parity_difference": 0.05}, id="threshold-names-other-metric"),
        ],
    )
    def test_every_shape_of_absent_threshold_is_refused(self, thresholds):
        """`.get(key, default)` does not fire when the key is PRESENT holding
        None, so all three of these reached the comparison with threshold=None
        and were approved."""
        y_true, y_pred, protected = _grossly_unfair()
        gate = ModelFairnessGate(metrics=["equalized_odds_difference"], thresholds=thresholds)
        decision = gate.evaluate(y_true, y_pred, protected)
        assert not decision.approved
        assert decision.status is GateStatus.BLOCKED

    def test_evaluate_from_metrics_refuses_too(self):
        """The second entry point had the identical hole; a fix at one call site
        only is how this class of defect has survived three audits."""
        gate = ModelFairnessGate(metrics=["equalized_odds_difference"], thresholds={})
        decision = gate.evaluate_from_metrics({"equalized_odds_difference": 0.9})
        assert not decision.approved
        assert decision.status is GateStatus.BLOCKED
        assert gate.create_github_check(decision)["conclusion"] != "success"

    def test_non_blocking_unbounded_metric_downgrades_rather_than_blocking(self):
        """Could-not-check follows the same blocking/non-blocking routing as the
        NaN and absent-metric cases: CONDITIONAL with a warning, never
        APPROVED."""
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference", "equalized_odds_difference"],
            thresholds={"demographic_parity_difference": 0.1},
            blocking_metrics=["demographic_parity_difference"],
        )
        decision = gate.evaluate_from_metrics(
            {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.9}
        )
        assert decision.approved
        assert decision.status is GateStatus.CONDITIONAL
        assert any("no threshold" in w for w in decision.warnings)

    def test_the_constructor_does_not_invent_a_config_the_caller_refused(self):
        """The same falsiness bug one level up. `metrics or [...]` and
        `thresholds or {...}` could not tell "said nothing" from "said NOTHING
        is configured". Measured: thresholds={} became
        {'demographic_parity_difference': 0.1} and APPROVED a value of 0.05
        against a bound nobody set, and metrics=[] became the default
        one-metric list, so evaluate()'s no-metrics guard could never fire
        through this constructor."""
        gate = ModelFairnessGate(metrics=["demographic_parity_difference"], thresholds={})
        assert gate.config.thresholds == {}
        decision = gate.evaluate_from_metrics({"demographic_parity_difference": 0.05})
        assert not decision.approved
        assert decision.metric_evaluations[0].threshold is None

        empty = ModelFairnessGate(metrics=[], thresholds={"demographic_parity_difference": 0.1})
        assert empty.config.metrics == []
        assert not empty.evaluate_from_metrics({}).approved

    def test_the_hierarchical_path_reports_rather_than_crashes(self):
        """A None threshold reached `relax_threshold` and raised TypeError
        ("unsupported operand type(s) for *: 'NoneType' and 'float'"), so the
        intersectional entry point aborted on the config the flat one now
        reports honestly."""
        rng = np.random.default_rng(3)
        n = 600
        gender = np.array(rng.choice(["F", "M"], n), dtype=object)
        race = np.array(rng.choice(["B", "W"], n), dtype=object)
        y_true = rng.integers(0, 2, n)
        y_pred = (
            ((gender == "M") & (rng.random(n) < 0.8)) | ((gender == "F") & (rng.random(n) < 0.3))
        ).astype(int)

        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": None},
        )
        decision = gate.evaluate_hierarchical(y_true, y_pred, {"gender": gender, "race": race})
        assert not decision.approved
        assert decision.status is GateStatus.BLOCKED

    # OVER-CORRECTION CONTROLS

    def test_the_hierarchical_path_still_approves_a_fair_model(self):
        rng = np.random.default_rng(5)
        n = 600
        gender = np.array(rng.choice(["F", "M"], n), dtype=object)
        race = np.array(rng.choice(["B", "W"], n), dtype=object)
        y_true = rng.integers(0, 2, n)
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )

        unfair = (
            ((gender == "M") & (rng.random(n) < 0.8)) | ((gender == "F") & (rng.random(n) < 0.3))
        ).astype(int)
        assert not gate.evaluate_hierarchical(
            y_true, unfair, {"gender": gender, "race": race}
        ).approved

        fair = (rng.random(n) < 0.5).astype(int)
        assert gate.evaluate_hierarchical(y_true, fair, {"gender": gender, "race": race}).approved

    def test_a_caller_who_passes_nothing_still_gets_the_documented_defaults(self):
        """`is None` must not become "always empty": the no-argument
        constructor keeps its documented default metric and bound."""
        gate = ModelFairnessGate()
        assert gate.config.metrics == ["demographic_parity_difference"]
        assert gate.config.thresholds == {"demographic_parity_difference": 0.1}
        assert gate.evaluate_from_metrics({"demographic_parity_difference": 0.01}).approved

    def test_fair_model_with_every_threshold_configured_is_still_approved(self):
        y_true, y_pred, protected = _genuinely_fair()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference", "equalized_odds_difference"],
            thresholds={
                "demographic_parity_difference": 0.1,
                "equalized_odds_difference": 0.1,
            },
        )
        decision = gate.evaluate(y_true, y_pred, protected)
        assert decision.approved
        assert decision.status is GateStatus.APPROVED
        assert gate.create_github_check(decision)["conclusion"] == "success"
        assert all(e.passed for e in decision.metric_evaluations)

    def test_threshold_of_zero_is_still_enforced_not_treated_as_absent(self):
        """`if not threshold` would make 0.0 behave exactly like an absent
        bound. A zero-tolerance policy is a real policy: it must still FAIL a
        non-zero value and still PASS a zero one."""
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.0},
        )

        breached = gate.evaluate_from_metrics({"demographic_parity_difference": 0.02})
        assert not breached.approved
        # It failed the COMPARISON, not the could-not-check branch.
        assert "exceeds threshold" in breached.blocking_reasons[0]
        assert breached.metric_evaluations[0].threshold == 0.0

        met = gate.evaluate_from_metrics({"demographic_parity_difference": 0.0})
        assert met.approved
        assert met.status is GateStatus.APPROVED


# Defect 2: allow_degradation_margin defaulted to 0.0 behind a `> 0` guard, so
# the default AND the strictest-looking setting switched the check off


def _degradation_decision(margin, value, baseline):
    config = GateConfig(
        metrics=["demographic_parity_difference"],
        thresholds={"demographic_parity_difference": 0.1},
        allow_degradation_margin=margin,
    )
    gate = ModelFairnessGate(config=config)
    return gate.evaluate_from_metrics(
        {"demographic_parity_difference": value},
        baseline_metrics={"demographic_parity_difference": baseline},
    )


class TestGateDegradationMargin:
    def test_zero_margin_blocks_a_nine_fold_degradation(self):
        """Measured before the fix: margin=0.0 -> approved=True, margin=0.001 ->
        blocked. The parameter was non-monotonic, and its own docstring
        ("Maximum allowed degradation from baseline") was false at its
        default."""
        decision = _degradation_decision(0.0, value=0.09, baseline=0.01)
        assert not decision.approved
        assert any("degraded" in r for r in decision.blocking_reasons)

    def test_margin_is_monotonic(self):
        """Tighter must never be more permissive than looser. This is the
        property the `> 0` guard broke."""
        blocked = [
            margin
            for margin in (0.0, 0.001, 0.01, 0.05)
            if not _degradation_decision(margin, value=0.09, baseline=0.01).approved
        ]
        assert blocked == [0.0, 0.001, 0.01, 0.05]
        # And a margin wider than the degradation still allows it through.
        assert _degradation_decision(0.5, value=0.09, baseline=0.01).approved

    def test_default_is_none_and_names_the_off_state(self):
        assert GateConfig().allow_degradation_margin is None
        assert GateConfig().to_dict()["allow_degradation_margin"] is None

    # OVER-CORRECTION CONTROLS

    def test_unchanged_metric_still_passes_at_zero_margin(self):
        """`degradation > margin` must stay STRICT. A `>=` here would block
        every gate whose metric did not move at all."""
        decision = _degradation_decision(0.0, value=0.05, baseline=0.05)
        assert decision.approved
        assert decision.status is GateStatus.APPROVED

    def test_improved_metric_still_passes_at_zero_margin(self):
        decision = _degradation_decision(0.0, value=0.02, baseline=0.05)
        assert decision.approved

    def test_default_config_does_not_start_blocking_on_a_baseline(self):
        """The default must not become zero-tolerance. A gate that supplies a
        baseline and never asked for a degradation bound keeps working."""
        decision = _degradation_decision(None, value=0.09, baseline=0.01)
        assert decision.approved


# Defect 3: assert_fairness rewrote a None threshold to 0.1


def _ratio_data():
    """demographic_parity_ratio 0.107, far below the four-fifths floor of
    0.80, and comfortably above a fabricated 0.1."""
    protected = np.array(["a"] * 400 + ["b"] * 400, dtype=object)
    y_pred = np.array([1] * 300 + [0] * 100 + [1] * 32 + [0] * 368)
    y_true = np.array([1] * 200 + [0] * 200 + [1] * 200 + [0] * 200)
    return y_true, y_pred, protected


class TestAssertFairnessNoneThreshold:
    def test_none_threshold_is_not_rewritten_to_a_default(self):
        """Measured: thresholds={'demographic_parity_ratio': None} returned
        normally at a ratio of 0.107, because 0.107 clears an invented floor of
        0.1. The four-fifths floor is 0.80."""
        y_true, y_pred, protected = _ratio_data()
        with pytest.raises(MetricsFairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                protected,
                metrics=["demographic_parity_ratio"],
                thresholds={"demographic_parity_ratio": None},
            )
        failed = excinfo.value.failed_metrics["demographic_parity_ratio"]
        assert failed["not_measurable"] is True
        assert "no threshold" in failed["reason"]

    def test_a_bound_that_cannot_be_breached_is_still_could_not_check(self):
        """The neighbouring guard, pinned here so the None fix cannot be
        'simplified' into re-enabling a degenerate floor of 0.0."""
        y_true, y_pred, protected = _ratio_data()
        with pytest.raises(MetricsFairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                protected,
                metrics=["demographic_parity_ratio"],
                thresholds={"demographic_parity_ratio": 0.0},
            )
        assert excinfo.value.failed_metrics["demographic_parity_ratio"]["not_measurable"] is True

    # OVER-CORRECTION CONTROLS

    def test_documented_defaults_still_apply_when_thresholds_is_none(self):
        """`thresholds=None` must keep meaning "use the documented defaults",
        not "refuse everything"."""
        y_true, y_pred, protected = _ratio_data()
        with pytest.raises(MetricsFairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                protected,
                metrics=["demographic_parity_ratio"],
                thresholds=None,
            )
        failed = excinfo.value.failed_metrics["demographic_parity_ratio"]
        # A MEASURED breach against the documented 0.8 floor, not a
        # could-not-check.
        assert failed["not_measurable"] is False
        assert failed["threshold"] == 0.8

    def test_a_fair_model_still_passes(self):
        protected = np.array(["a"] * 400 + ["b"] * 400, dtype=object)
        y_true = np.array(([1] * 200 + [0] * 200) * 2)
        y_pred = np.array(([1] * 180 + [0] * 20 + [1] * 20 + [0] * 180) * 2)
        values = assert_fairness(
            y_true,
            y_pred,
            protected,
            metrics=["demographic_parity_ratio"],
            thresholds={"demographic_parity_ratio": 0.8},
        )
        assert values["demographic_parity_ratio"] == pytest.approx(1.0)

    def test_a_fair_model_passes_on_the_documented_defaults(self):
        protected = np.array(["a"] * 400 + ["b"] * 400, dtype=object)
        y_true = np.array(([1] * 200 + [0] * 200) * 2)
        y_pred = np.array(([1] * 180 + [0] * 20 + [1] * 20 + [0] * 180) * 2)
        values = assert_fairness(y_true, y_pred, protected)
        assert values["demographic_parity_difference"] == pytest.approx(0.0)


# Defect 4: parametrize_fairness replaced the four-fifths FLOOR of 0.80 with a
# CAP of 0.10 for any metric the caller had not configured


class TestParametrizeFairnessFabricatedThreshold:
    def test_unconfigured_metric_never_reaches_the_test_body(self):
        """Measured: with thresholds={}, a disparate_impact_ratio of 0.11 was
        handed threshold=0.1, satisfied `value >= threshold`, and was recorded
        as passed=True."""
        called = []

        @parametrize_fairness(
            protected_attributes=["gender"],
            metrics=["disparate_impact_ratio"],
            thresholds={},
        )
        def scenario(protected_attribute, metric, threshold):
            called.append((metric, threshold))
            return 0.11 >= threshold

        with pytest.raises(FairnessAssertionError) as excinfo:
            scenario()

        assert called == []
        failed = excinfo.value.details["failed_tests"]
        assert failed[0]["threshold"] is None
        assert "no threshold" in failed[0]["error"]
        # The aggregate error must not print a fabricated measurement either.
        assert math.isnan(excinfo.value.actual_value)

    # OVER-CORRECTION CONTROLS

    def test_configured_threshold_still_runs_and_can_pass(self):
        called = []

        @parametrize_fairness(
            protected_attributes=["gender", "race"],
            metrics=["disparate_impact_ratio"],
            thresholds={"disparate_impact_ratio": 0.8},
        )
        def scenario(protected_attribute, metric, threshold):
            called.append((protected_attribute, metric, threshold))
            return 0.95 >= threshold

        results = scenario()
        assert [r["passed"] for r in results] == [True, True]
        assert called == [
            ("gender", "disparate_impact_ratio", 0.8),
            ("race", "disparate_impact_ratio", 0.8),
        ]

    def test_threshold_of_zero_still_reaches_the_test_body(self):
        """`if not threshold` here would silently drop a zero-tolerance
        scenario."""
        called = []

        @parametrize_fairness(
            protected_attributes=["gender"],
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.0},
        )
        def scenario(protected_attribute, metric, threshold):
            called.append(threshold)
            return True

        scenario()
        assert called == [0.0]


# Defect 5: the MCP tool hardwired min_group_size=2 where the library default
# is 30, and it ships as the console script vfairness-mcp


def _two_person_group_frame():
    return pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 2,
            "y": [1] * 50 + [0] * 50 + [1, 0],
            "p": [1] * 40 + [0] * 60 + [1, 0],
        }
    )


class TestMcpMeasureFairnessMinGroupSize:
    def test_a_two_person_group_is_not_silently_measured(self):
        """Measured before the fix: a clean demographic_parity_difference of
        0.100 with 0 warnings, over a group of two people. The library default
        on the identical input reports nan with 6 small-sample warnings."""
        frame = _two_person_group_frame()
        with pytest.warns(UserWarning):
            out = mcp_tools.measure_fairness(frame, ["g"], "p", "y")
        # jsonify turns the NaN into null: not a number anyone can act on.
        assert out["per_attribute"]["g"]["metrics"]["demographic_parity_difference"] is None
        assert "demographic parity difference" not in out["summary"]

    def test_default_matches_the_library_and_the_sibling_tool(self):
        import inspect

        from vfairness import FairnessAnalyzer

        library_default = (
            inspect.signature(FairnessAnalyzer.__init__).parameters["min_group_size"].default
        )
        tool_default = (
            inspect.signature(mcp_tools.measure_fairness).parameters["min_group_size"].default
        )
        sibling_default = (
            inspect.signature(mcp_tools.analyze_intersectional).parameters["min_group_size"].default
        )
        assert tool_default == library_default == sibling_default == 30

    # OVER-CORRECTION CONTROLS

    def test_an_adequately_sized_dataset_still_measures(self):
        frame = pd.DataFrame(
            {
                "g": ["a"] * 200 + ["b"] * 200,
                "y": ([1] * 100 + [0] * 100) * 2,
                "p": [1] * 90 + [0] * 110 + [1] * 60 + [0] * 140,
            }
        )
        out = mcp_tools.measure_fairness(frame, ["g"], "p", "y")
        value = out["per_attribute"]["g"]["metrics"]["demographic_parity_difference"]
        assert value == pytest.approx(0.15)
        assert "demographic parity difference" in out["summary"]

    def test_a_caller_can_still_opt_into_a_smaller_minimum(self):
        frame = _two_person_group_frame()
        out = mcp_tools.measure_fairness(frame, ["g"], "p", "y", min_group_size=2)
        assert out["per_attribute"]["g"]["metrics"]["demographic_parity_difference"] == (
            pytest.approx(0.1)
        )


# Defect 6: `nan > threshold` is False, so the else branch wrote "not drifting"
# for a metric that was never measured


def _monitor(baseline):
    config = MonitorConfig(
        drift_threshold=0.05,
        metrics_to_monitor=["demographic_parity_difference", "equalized_odds_difference"],
        min_samples_for_alert=1,
    )
    return BiasMonitor(baseline_metrics=baseline, config=config)


def _unmeasurable_batch():
    """Group 'b' has no positive labels, so its TPR is undefined and
    equalized_odds_difference is NaN by design."""
    protected = np.array(["a"] * 100 + ["b"] * 100, dtype=object)
    y_true = np.array([1] * 100 + [0] * 100)
    y_pred = np.array([1] * 100 + [1] * 50 + [0] * 50)
    return y_pred, y_true, protected


class TestMonitorUnmeasurableDrift:
    def test_unmeasurable_metric_is_not_reported_as_stable(self):
        """Measured before the fix: computed metrics
        {'demographic_parity_difference': 0.5, 'equalized_odds_difference':
        nan} against a matching baseline gave drift_detected() False and
        exported `fairness_drift_equalized_odds_difference 0`."""
        monitor = _monitor(
            {"demographic_parity_difference": 0.5, "equalized_odds_difference": 0.02}
        )
        result = monitor.log_batch(*_unmeasurable_batch())

        assert math.isnan(result.metrics["equalized_odds_difference"])
        assert result.could_not_check == ["equalized_odds_difference"]
        assert result.drift_detected
        assert "COULD NOT CHECK" in repr(result)

        # Three states, never two.
        assert monitor.get_drift_status() == {
            "demographic_parity_difference": False,
            "equalized_odds_difference": None,
        }
        assert monitor.unmeasurable_metrics() == ["equalized_odds_difference"]
        assert monitor.drift_detected()

    def test_prometheus_never_exports_a_zero_for_an_unmeasured_metric(self):
        monitor = _monitor(
            {"demographic_parity_difference": 0.5, "equalized_odds_difference": 0.02}
        )
        monitor.log_batch(*_unmeasurable_batch())
        exported = monitor.to_prometheus_metrics().splitlines()
        assert "fairness_drift_equalized_odds_difference NaN" in exported
        assert "fairness_drift_equalized_odds_difference 0" not in exported
        assert "fairness_monitor_unmeasurable_metrics 1" in exported

    def test_the_documented_alert_pattern_does_not_resolve_into_silence(self):
        """`if monitor.drift_detected(): monitor.trigger_alert()` is the class
        docstring's own example; it must not return None for the one state that
        most needs an operator."""
        monitor = _monitor(
            {"demographic_parity_difference": 0.5, "equalized_odds_difference": 0.02}
        )
        monitor.log_batch(*_unmeasurable_batch())
        assert monitor.drift_detected()
        alert = monitor.trigger_alert()
        assert alert is not None
        assert alert.drift_type is DriftType.NOT_MEASURABLE
        assert alert.severity.value == "warning"

    # OVER-CORRECTION CONTROLS

    def test_a_fully_measured_stable_batch_is_still_green(self):
        protected = np.array(["a"] * 100 + ["b"] * 100, dtype=object)
        y_true = np.array(([1] * 50 + [0] * 50) * 2)
        y_pred = np.array(([1] * 50 + [0] * 50) * 2)
        monitor = _monitor({"demographic_parity_difference": 0.0, "equalized_odds_difference": 0.0})
        result = monitor.log_batch(y_pred, y_true, protected)

        assert result.could_not_check == []
        assert not result.drift_detected
        assert repr(result).count("OK") == 1
        assert not monitor.drift_detected()
        assert monitor.unmeasurable_metrics() == []
        assert monitor.trigger_alert() is None
        exported = monitor.to_prometheus_metrics()
        assert "fairness_drift_equalized_odds_difference 0" in exported
        assert "fairness_monitor_unmeasurable_metrics 0" in exported

    def test_a_real_measured_drift_is_still_reported_as_drift(self):
        """Could-not-check must not swallow the case the monitor exists for."""
        protected = np.array(["a"] * 100 + ["b"] * 100, dtype=object)
        y_true = np.array(([1] * 50 + [0] * 50) * 2)
        y_pred = np.array([1] * 90 + [0] * 10 + [1] * 10 + [0] * 90)
        monitor = _monitor({"demographic_parity_difference": 0.0, "equalized_odds_difference": 0.0})
        monitor.log_batch(y_pred, y_true, protected)

        assert monitor.unmeasurable_metrics() == []
        assert monitor.get_drift_status()["demographic_parity_difference"] is True
        alert = monitor.trigger_alert()
        assert alert is not None
        assert alert.drift_type is DriftType.METRIC_DRIFT


# Defect 7: DataValidationResult.passed stayed True with checks_run == []


def _tiny_frame():
    return pd.DataFrame(
        {"gender": ["M"] * 10 + ["F"] * 2, "approved": [1] * 6 + [0] * 6},
    )


def _clean_frame():
    rng = np.random.default_rng(7)
    n = 600
    return pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "score": rng.normal(0, 1, n),
            "age": rng.integers(20, 70, n),
            "approved": rng.integers(0, 2, n),
        }
    )


_NO_CHECKS = DataValidationConfig(
    check_representation=False,
    check_outcome_disparity=False,
    check_missing_patterns=False,
    check_label_quality=False,
    check_data_hygiene=False,
)


class TestValidatorNoCheckRan:
    def test_a_run_that_checked_nothing_is_not_a_pass(self):
        """The object used to contradict itself: summary "neither a pass nor a
        failure", __repr__ "PASSED", passed True. CI reads the boolean."""
        result = DataBiasValidator(protected_attributes=["gender"], config=_NO_CHECKS).validate(
            _tiny_frame(), outcome_column="approved"
        )
        assert result.checks_run == []
        assert result.execution_coverage() == "none"
        assert not result.passed
        assert result.to_dict()["passed"] is False
        assert "COULD NOT CHECK" in repr(result)
        assert "PASSED" not in repr(result)

    def test_the_ci_artifact_does_not_report_an_empty_green_suite(self):
        """`tests="0" failures="0"` is painted green by every CI UI, so the
        JUnit export needs a failing case of its own for a run that examined
        nothing."""
        result = DataBiasValidator(protected_attributes=["gender"], config=_NO_CHECKS).validate(
            _tiny_frame(), outcome_column="approved"
        )
        xml = result.to_junit_xml()
        assert 'tests="1" failures="1"' in xml
        assert "validation_coverage" in xml
        # Well-formed: the CI consumer has to be able to parse it.
        ET.fromstring(xml)

    # OVER-CORRECTION CONTROLS

    def test_could_not_check_is_not_collapsed_into_a_measured_failure(self):
        """The first version of this fix injected an ERROR issue to make the
        boolean move, and the SVG banner flipped from a grey UNKNOWN to a red
        FAIL. That is a different lie about the same run: `issues` must stay
        empty so the reader-facing verdict stays could-not-check."""
        from vfairness.rendering.adapters_validation import data_validation_to_svg

        result = DataBiasValidator(protected_attributes=["gender"], config=_NO_CHECKS).validate(
            _tiny_frame(), outcome_column="approved"
        )
        assert result.issues == []
        assert result.errors == []

        svg = data_validation_to_svg(result)
        assert ">UNKNOWN<" in svg
        assert ">PASS<" not in svg
        assert ">FAIL<" not in svg

    def test_a_real_failure_with_no_coverage_record_still_reads_fail(self):
        """The mirror: an issue raised before any check ran is a MEASURED
        failure and must not be softened into could-not-check."""
        from vfairness.rendering.adapters_validation import data_validation_to_svg

        result = DataBiasValidator(protected_attributes=["not_a_column"]).validate(_clean_frame())
        assert result.checks_run == []
        assert not result.passed
        assert ">FAIL<" in data_validation_to_svg(result)

    def test_a_clean_dataset_with_full_coverage_still_passes(self):
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _clean_frame(), outcome_column="approved"
        )
        assert result.execution_coverage() == "complete"
        assert result.passed
        assert "PASSED" in repr(result)
        assert result.issues == []

    def test_partial_coverage_still_grades_over_what_ran(self):
        """Only the "nothing ran" case fails closed. A validator that refused
        every partial run would block every legitimate one."""
        config = DataValidationConfig(
            check_representation=False,
            check_outcome_disparity=False,
            check_missing_patterns=True,
            check_label_quality=False,
            check_data_hygiene=False,
        )
        result = DataBiasValidator(protected_attributes=["gender"], config=config).validate(
            _clean_frame(), outcome_column="approved"
        )
        assert result.execution_coverage() == "partial"
        assert result.passed
        assert "not run" in result.summary

    def test_an_unrecorded_result_is_still_unrecorded_not_failed(self):
        """A result built by hand, or by a version before checks_run existed,
        records nothing and must not be read as either verdict."""
        from vfairness.operations.cicd.validator import DataValidationResult

        result = DataValidationResult(passed=True, issues=[], metrics={}, summary="built by hand")
        assert result.execution_coverage() == "unrecorded"
        assert result.passed
        assert "PASSED" in repr(result)


# Defect 8: an empty stub validator manufactured three green PASS tiles


def _identifier_only_frame():
    """Six rows, three of them duplicates, one constant column, and the only
    chosen attribute is an identifier, so no attribute survives preparation."""
    frame = pd.DataFrame(
        {
            "applicant_id": [f"id{i}" for i in range(6)],
            "dead": [1] * 6,
            "approved": [1, 0, 1, 0, 1, 0],
        }
    )
    return pd.concat([frame.iloc[:3], frame.iloc[:3]], ignore_index=True)


class TestQualityReportStubValidator:
    def test_no_green_tile_is_emitted_when_the_validator_never_ran(self):
        """Measured before the fix, on this exact frame: three "pass" tiles
        reading "Every group ... is large enough", "No column has a problematic
        level of blank values" and "Enough rows, no duplicate rows, no dead
        columns" -- while the real validator on the same frame reports
        insufficient_total_samples, duplicate_rows (50.0%) and
        constant_columns."""
        report = build_quality_report(
            _identifier_only_frame(), ["applicant_id"], outcome_column="approved"
        )
        by_key = {c["key"]: c for c in report["checks"]}
        for key in ("vf_representation", "vf_missing", "vf_hygiene"):
            assert by_key[key]["status"] == "warn"
            assert "could not run" in by_key[key]["detail"]
        assert not any(c["status"] == "pass" for c in report["checks"])

    def test_the_claims_that_were_made_are_demonstrably_false(self):
        """The evidence half: the validator, given a usable attribute on the
        same frame, finds the very problems the green tiles denied."""
        frame = _identifier_only_frame()
        frame["gender"] = ["M", "F"] * 3
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            frame, outcome_column="approved"
        )
        found = {i.issue_type for i in result.issues}
        assert "insufficient_total_samples" in found
        assert "constant_columns" in found

    # OVER-CORRECTION CONTROL

    def test_a_clean_frame_with_a_real_attribute_still_shows_pass_tiles(self):
        report = build_quality_report(_clean_frame(), ["gender"], outcome_column="approved")
        by_key = {c["key"]: c for c in report["checks"]}
        assert by_key["vf_representation"]["status"] == "pass"
        assert by_key["vf_missing"]["status"] == "pass"
        assert by_key["vf_hygiene"]["status"] == "pass"
        assert report["tone"] == "pass"


# Defect 9: the pre-commit config checker passed a config that gates nothing


def _write_config(tmp_path, payload, name="fairness_gate.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


class TestPrecommitConfigChecker:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(
                {
                    "metrics": [
                        "demographic_parity_difference",
                        "equalized_odds_difference",
                    ],
                    "thresholds": {"demographic_parity_difference": 0.1},
                },
                id="metric-with-no-bound",
            ),
            pytest.param(
                {"metrics": ["demographic_parity_difference"], "thresholds": {}},
                id="no-bounds-at-all",
            ),
            pytest.param({"metrics": [], "thresholds": {}}, id="gates-nothing"),
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": True},
                },
                id="bool-is-a-subclass-of-int",
            ),
            pytest.param(
                {
                    "metrics": ["equalized_odds_difference"],
                    "thresholds": {"demographic_parity_difference": 0.1},
                },
                id="bound-names-an-ungated-metric",
            ),
            pytest.param(
                {
                    "metrics": "demographic_parity_difference",
                    "thresholds": {"demographic_parity_difference": 0.1},
                },
                id="metrics-not-a-list",
            ),
        ],
    )
    def test_a_config_that_gates_nothing_is_refused(self, tmp_path, payload):
        assert check_fairness_config([_write_config(tmp_path, payload)]) == 1

    # OVER-CORRECTION CONTROLS

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 0.1},
                },
                id="one-metric-one-bound",
            ),
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 0.0},
                },
                id="zero-tolerance-is-a-real-policy",
            ),
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference", "disparate_impact_ratio"],
                    "thresholds": {
                        "demographic_parity_difference": 0.1,
                        "disparate_impact_ratio": 0.8,
                    },
                },
                id="two-metrics-two-bounds",
            ),
        ],
    )
    def test_a_real_config_still_passes(self, tmp_path, payload):
        assert check_fairness_config([_write_config(tmp_path, payload)]) == 0

    def test_a_file_that_is_not_a_fairness_config_is_left_alone(self, tmp_path):
        path = tmp_path / "tsconfig.json"
        path.write_text(json.dumps({"compilerOptions": {}}))
        assert check_fairness_config([str(path)]) == 0
