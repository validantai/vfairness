"""Wave-4 pins for the top-level release gate: it must never pass a check it did
not perform.

Four fail-open defects in ``evaluation/vfairness_metrics/integrations.py``, all
reproduced by execution before the fix:

1. A requested metric the analyzer never produced was WARNED about and SKIPPED,
   and the gate then passed. Executed on a model whose disparate impact is 0.00
   (group B is never selected): ``assert_fairness(metrics=['disparate_impact'])``
   returned ``{}`` and did not raise. A user who asks for a check, does not get
   it, and is told everything is fine holds a false certificate.
2. ``ci_must_exclude_zero=True`` together with ``include_ci=False`` silently
   dropped the requested significance check and passed.
3. Both the gate and the training callback kept a private two-state direction
   rule (``endswith("_ratio")`` else ``abs(value) > threshold``) with no
   could-not-check state. Executed: ``disparate_impact`` at 0.10 and
   ``worst_group_accuracy`` at 0.10 against a 0.80 floor both PASSED there, while
   the shared ``_metric_direction.check_threshold`` calls both FAIL.
4. The confidence-interval ``point_estimate`` was logged to MLflow and W&B with a
   bare ``float()`` and no finiteness guard, while every neighbouring line skips
   a NaN. That writes a NaN into an experiment tracker as if it were a
   measurement.

Every test below is a NEGATIVE case: the gate must REFUSE. The controls named
``*_still_passes`` are the over-correction half, and they matter just as much:
a gate that refuses everything is not a gate.
"""

import math
import sys
import types
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError,
    assert_fairness,
    create_fairness_callback,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)

# Data builders


def _maximal_disparate_impact_data(n=200):
    """Two large groups: A is always selected, B never. Disparate impact 0.00,
    the worst value the metric can take."""
    sens = np.array(["A"] * n + ["B"] * n)
    y_true = np.array([1, 0] * n)
    y_pred = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return y_true, y_pred, sens


def _measured_fair_data(n=400):
    """Two large groups, identical perfect predictions: genuinely fair and fully
    measurable. The over-correction control."""
    sens = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_true = np.array([1, 0] * (n // 2))
    y_pred = y_true.copy()
    return y_true, y_pred, sens


def _patch_results(monkeypatch, results):
    """Make FairnessAnalyzer.compute_all_metrics return exactly ``results``.

    Lets a single named metric be driven to a chosen value, which is how the
    direction rule is tested without hunting for data that happens to produce it.
    """
    from vfairness.evaluation.vfairness_metrics import analyzer as analyzer_mod

    def fake_compute_all_metrics(self, **kwargs):
        return dict(results)

    monkeypatch.setattr(
        analyzer_mod.FairnessAnalyzer, "compute_all_metrics", fake_compute_all_metrics
    )


# Defect 1: a requested metric that was never produced


class TestRequestedButUnavailableMetricFailsTheGate:
    def test_metric_the_analyzer_never_produced_raises(self):
        """THE acceptance repro. Before the fix this returned {} and did not raise."""
        y_true, y_pred, sens = _maximal_disparate_impact_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(
                    y_true,
                    y_pred,
                    sens,
                    metrics=["disparate_impact"],
                    thresholds={"disparate_impact": 0.8},
                )

        err = excinfo.value
        text = str(err)
        # The metric is named, so the user knows which requested check is missing.
        assert "disparate_impact" in text
        assert "NOT AVAILABLE" in text
        failure = err.failed_metrics["disparate_impact"]
        # Three states, never two: could-not-check, not a measured violation.
        assert failure["not_measurable"] is True
        assert failure["value"] is None

    def test_error_says_which_metrics_were_available(self):
        """The reason must say WHY it was unavailable, not merely that it was."""
        y_true, y_pred, sens = _maximal_disparate_impact_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(y_true, y_pred, sens, metrics=["disparate_impact"])
        reason = excinfo.value.failed_metrics["disparate_impact"]["reason"]
        assert "demographic_parity_difference" in reason

    def test_result_of_an_unusable_type_also_raises(self, monkeypatch):
        """A produced result that is neither a number nor a MetricResult was
        ``continue``d over, which is the same silent skip one step later."""
        _patch_results(monkeypatch, {"demographic_parity_difference": "not a number"})
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.1},
            )
        assert excinfo.value.failed_metrics["demographic_parity_difference"]["not_measurable"]

    def test_produced_metrics_still_pass(self):
        """Over-correction control: metrics the analyzer does produce, on fair
        data, must still pass."""
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=["demographic_parity_difference", "equal_opportunity_difference"],
        )
        assert set(values) == {
            "demographic_parity_difference",
            "equal_opportunity_difference",
        }
        for name, value in values.items():
            assert math.isfinite(value), f"{name} should be measurable here"


# Defect 2: a requested significance check that cannot run


class TestContradictoryCIConfigurationIsRefused:
    def test_ci_must_exclude_zero_without_include_ci_raises(self):
        """Before the fix the requested CI check was dropped and the gate passed."""
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(ValueError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                include_ci=False,
                ci_must_exclude_zero=True,
            )
        text = str(excinfo.value)
        assert "ci_must_exclude_zero" in text
        assert "include_ci" in text

    def test_refusal_happens_even_on_data_that_would_otherwise_fail(self):
        """The configuration is refused before any verdict is reached, so the
        contradiction cannot hide behind an unrelated failure."""
        y_true, y_pred, sens = _maximal_disparate_impact_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="ci_must_exclude_zero"):
                assert_fairness(
                    y_true,
                    y_pred,
                    sens,
                    metrics=["demographic_parity_difference"],
                    thresholds={"demographic_parity_difference": 0.1},
                    ci_must_exclude_zero=True,
                )

    def test_ci_off_and_check_not_requested_still_passes(self):
        """Over-correction control: the ordinary no-CI call is untouched."""
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=["demographic_parity_difference"],
            include_ci=False,
            ci_must_exclude_zero=False,
        )
        assert math.isfinite(values["demographic_parity_difference"])

    def test_ci_on_with_significant_interval_still_passes(self, monkeypatch):
        """Over-correction control: the properly configured CI check still runs
        and still passes on an interval that excludes zero."""
        from vfairness.evaluation.vfairness_metrics import analyzer as analyzer_mod

        _patch_results(
            monkeypatch,
            {
                "demographic_parity_difference": analyzer_mod.MetricResult(
                    metric_name="demographic_parity_difference",
                    value=0.01,
                    confidence_interval=(0.005, 0.02),
                )
            },
        )
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=["demographic_parity_difference"],
            include_ci=True,
            ci_must_exclude_zero=True,
            n_bootstrap=10,
        )
        assert values["demographic_parity_difference"] == pytest.approx(0.01)


# Defect 3: the direction rule, shared rather than re-guessed


class TestGateUsesTheSharedDirectionRule:
    """The private rule read direction from a ``_ratio`` SUFFIX alone. Two whole
    families are higher-is-better without carrying that suffix, and both were
    graded as violation magnitudes, i.e. inverted."""

    @pytest.mark.parametrize(
        "metric_name",
        ["disparate_impact", "worst_group_accuracy"],
    )
    def test_higher_is_better_metric_below_its_floor_fails(self, monkeypatch, metric_name):
        _patch_results(monkeypatch, {metric_name: 0.10})
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=[metric_name],
                thresholds={metric_name: 0.80},
            )
        failure = excinfo.value.failed_metrics[metric_name]
        # A measured violation, NOT a could-not-check: the value was compared.
        assert failure["not_measurable"] is False
        assert "minimum" in failure["reason"]

    @pytest.mark.parametrize(
        "metric_name",
        ["disparate_impact", "worst_group_accuracy"],
    )
    def test_higher_is_better_metric_above_its_floor_still_passes(self, monkeypatch, metric_name):
        """Over-correction control for the direction fix."""
        _patch_results(monkeypatch, {metric_name: 0.95})
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=[metric_name],
            thresholds={metric_name: 0.80},
        )
        assert values[metric_name] == pytest.approx(0.95)

    def test_unknown_direction_metric_is_could_not_check(self, monkeypatch):
        """calibration_slope is deliberately absent from both direction tables:
        its target is 1.0, so neither direction is 'better'. The gate must say so
        rather than pick one."""
        _patch_results(monkeypatch, {"calibration_slope": 0.5})
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["calibration_slope"],
                thresholds={"calibration_slope": 0.1},
            )
        failure = excinfo.value.failed_metrics["calibration_slope"]
        assert failure["not_measurable"] is True
        assert "NOT MEASURABLE" in failure["reason"]

    def test_lower_is_better_violation_still_fails_as_measured(self, monkeypatch):
        """Over-correction control on the other side: a difference metric over
        its bound is still a MEASURED failure, not a could-not-check."""
        _patch_results(monkeypatch, {"demographic_parity_difference": 0.9})
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.1},
            )
        failure = excinfo.value.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is False


class TestCallbackUsesTheSharedDirectionRule:
    @pytest.mark.parametrize(
        "metric_name",
        ["disparate_impact", "worst_group_accuracy"],
    )
    def test_higher_is_better_metric_below_its_floor_raises(self, monkeypatch, metric_name):
        _patch_results(monkeypatch, {metric_name: 0.10})
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group", thresholds={metric_name: 0.80}, fail_on_violation=True
        )
        with pytest.raises(ValueError, match="minimum"):
            callback(y_true, y_pred, sens)

    @pytest.mark.parametrize(
        "metric_name",
        ["disparate_impact", "worst_group_accuracy"],
    )
    def test_higher_is_better_metric_above_its_floor_still_passes(self, monkeypatch, metric_name):
        _patch_results(monkeypatch, {metric_name: 0.95})
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group", thresholds={metric_name: 0.80}, fail_on_violation=True
        )
        assert callback(y_true, y_pred, sens)[metric_name] == pytest.approx(0.95)

    def test_threshold_for_a_metric_that_was_never_produced_raises(self):
        """Defect 1 in the callback: a threshold naming a metric the analyzer
        does not produce was skipped by ``if metric_name in results``, so a
        training loop guarded on disparate impact was guarded on nothing."""
        y_true, y_pred, sens = _maximal_disparate_impact_data()
        callback = create_fairness_callback(
            "group", thresholds={"disparate_impact": 0.8}, fail_on_violation=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="NOT AVAILABLE"):
                callback(y_true, y_pred, sens)

    def test_thresholds_for_produced_metrics_still_pass(self):
        """Over-correction control: a threshold on a metric that IS produced,
        satisfied by fair data, must not raise."""
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        assert math.isfinite(callback(y_true, y_pred, sens)["demographic_parity_difference"])

    def test_unknown_direction_metric_raises_as_could_not_check(self, monkeypatch):
        _patch_results(monkeypatch, {"calibration_slope": 0.5})
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group", thresholds={"calibration_slope": 0.1}, fail_on_violation=True
        )
        with pytest.raises(ValueError, match="could not be checked"):
            callback(y_true, y_pred, sens)


# Defect 4: a NaN point estimate must not reach an experiment tracker


class _FakeMlflow(types.ModuleType):
    """Records every call, and keeps NaNs so the test can catch one arriving."""

    def __init__(self):
        super().__init__("mlflow")
        self.metrics = {}
        self.params = {}
        self.tags = {}

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, name):  # pragma: no cover - artifacts are off
        raise AssertionError("log_artifacts=False in these tests")


class _FakeWandbRun:
    def __init__(self):
        self.summary = {}


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = _FakeWandbRun()
        self.logged = {}

    def log(self, data):
        self.logged.update(data)

    def log_artifact(self, artifact):  # pragma: no cover - artifacts are off
        raise AssertionError("log_artifacts=False in these tests")


@pytest.fixture
def fake_mlflow(monkeypatch):
    module = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


@pytest.fixture
def fake_wandb(monkeypatch):
    module = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _report_with_ci(point_estimate):
    return {
        "task_type": "classification",
        "metrics": {},
        "metrics_with_ci": {
            "demographic_parity_difference": {
                "point_estimate": point_estimate,
                "lower_bound": float("nan"),
                "upper_bound": float("nan"),
            }
        },
        "group_stats": {},
        "assessment": {},
        "data_info": {"n_samples": 120, "n_groups": 2},
    }


class TestTrackersRefuseANaNPointEstimate:
    def test_mlflow_does_not_log_a_nan_point_estimate(self, fake_mlflow):
        logged = log_fairness_to_mlflow(_report_with_ci(float("nan")), log_artifacts=False)
        key = "fairness.demographic_parity_difference.value"
        assert key not in fake_mlflow.metrics
        assert key not in logged
        nans = [v for v in fake_mlflow.metrics.values() if isinstance(v, float) and math.isnan(v)]
        assert not nans, f"a NaN reached the tracker: {fake_mlflow.metrics}"

    def test_mlflow_does_not_log_a_none_point_estimate(self, fake_mlflow):
        """float(None) raised TypeError rather than skipping the value."""
        logged = log_fairness_to_mlflow(_report_with_ci(None), log_artifacts=False)
        assert "fairness.demographic_parity_difference.value" not in logged

    def test_mlflow_still_logs_a_measured_point_estimate(self, fake_mlflow):
        """Over-correction control."""
        log_fairness_to_mlflow(_report_with_ci(0.25), log_artifacts=False)
        assert fake_mlflow.metrics["fairness.demographic_parity_difference.value"] == 0.25

    def test_wandb_does_not_log_a_nan_point_estimate(self, fake_wandb):
        logged = log_fairness_to_wandb(_report_with_ci(float("nan")), log_artifacts=False)
        key = "fairness/demographic_parity_difference/value"
        assert key not in fake_wandb.logged
        assert key not in logged
        nans = [v for v in fake_wandb.logged.values() if isinstance(v, float) and math.isnan(v)]
        assert not nans, f"a NaN reached the tracker: {fake_wandb.logged}"

    def test_wandb_does_not_log_a_none_point_estimate(self, fake_wandb):
        logged = log_fairness_to_wandb(_report_with_ci(None), log_artifacts=False)
        assert "fairness/demographic_parity_difference/value" not in logged

    def test_wandb_still_logs_a_measured_point_estimate(self, fake_wandb):
        """Over-correction control."""
        log_fairness_to_wandb(_report_with_ci(0.25), log_artifacts=False)
        assert fake_wandb.logged["fairness/demographic_parity_difference/value"] == 0.25
