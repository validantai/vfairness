"""Fail-closed behaviour for a NOT ASSESSABLE run (fairness_score is None).

Covers three defects found by the acceptance verifiers:

1. ``log_fairness_to_mlflow`` / ``log_fairness_to_wandb`` crashed with
   ``TypeError: float() argument must be ... not 'NoneType'`` on a run whose
   ``assessment["fairness_score"]`` is None. Neither may recover by inventing a
   number: 0.0 and 1.0 are both indistinguishable on a dashboard from a measured
   verdict. The score must be OMITTED and an explicit status logged instead.
2. ``GroupManager.get_valid_groups`` warned that "Metrics will return default
   values (0.0 for differences, 1.0 for ratios)", which is false since the
   sentinels were removed, and is the sentence that made the old false-parity
   result look deliberate.
3. ``_warn_dropped_groups`` said "The metric is computed over the remaining
   groups only" even when exactly ONE group survived, where nothing is computed
   at all and the metric is NaN.

Neither mlflow nor wandb is installed in this environment, so the two logging
paths are exercised against recording stubs injected into ``sys.modules``. The
real SDK calls are NOT exercised here; only the arguments vfairness passes.
"""

import sys
import types
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
)
from vfairness.evaluation.vfairness_metrics.integrations import (
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)


def _report(fairness_score):
    """Minimal report dict shaped like classification_fairness_report output."""
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": float("nan")},
        "group_stats": {},
        "assessment": {
            "fairness_score": fairness_score,
            "assessable": fairness_score is not None,
            "passed_metrics": [],
            "failed_metrics": [],
            "not_assessable_metrics": [],
            "summary": "NOT ASSESSABLE: nothing here certifies fairness.",
        },
        "data_info": {"n_samples": 120, "n_groups": 2},
    }


class _FakeMlflow(types.ModuleType):
    """Records the calls vfairness makes, so no real MLflow run is needed."""

    def __init__(self):
        super().__init__("mlflow")
        self.metrics = {}
        self.params = {}
        self.tags = {}

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        # The real SDK requires a float here; None or NaN must never reach it.
        assert isinstance(value, float), f"log_metric({key!r}) got {value!r}"
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


# --------------------------------------------------------------------------
# Defect 1: None fairness_score must not crash and must not become a number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("score", [None, float("nan")])
def test_mlflow_not_assessable_logs_no_score(fake_mlflow, score):
    logged = log_fairness_to_mlflow(_report(score), log_artifacts=False)

    assert "fairness.fairness_score" not in fake_mlflow.metrics
    assert "fairness.fairness_score" not in logged
    # No fabricated value anywhere in the numeric channel.
    assert 0.0 not in fake_mlflow.metrics.values()
    assert 1.0 not in fake_mlflow.metrics.values()
    # The absence is stated, not implied.
    assert fake_mlflow.tags["fairness.fairness_score_status"] == "not_assessable"
    assert logged["fairness.fairness_score_status"] == "not_assessable"


def test_mlflow_assessed_score_still_logged(fake_mlflow):
    logged = log_fairness_to_mlflow(_report(0.75), log_artifacts=False)

    assert fake_mlflow.metrics["fairness.fairness_score"] == 0.75
    assert logged["fairness.fairness_score"] == 0.75
    assert fake_mlflow.tags["fairness.fairness_score_status"] == "assessed"


@pytest.mark.parametrize("score", [None, float("nan")])
def test_wandb_not_assessable_logs_no_score(fake_wandb, score):
    logged = log_fairness_to_wandb(_report(score), log_artifacts=False)

    assert "fairness/fairness_score" not in fake_wandb.logged
    assert "fairness/fairness_score" not in logged
    numeric = [v for v in fake_wandb.logged.values() if isinstance(v, (int, float))]
    assert 0.0 not in numeric
    assert 1.0 not in numeric
    assert fake_wandb.logged["fairness/fairness_score_status"] == "not_assessable"
    assert fake_wandb.run.summary["fairness/fairness_score_status"] == "not_assessable"
    assert logged["fairness/fairness_score_status"] == "not_assessable"


def test_wandb_assessed_score_still_logged(fake_wandb):
    logged = log_fairness_to_wandb(_report(0.5), log_artifacts=False)

    assert fake_wandb.logged["fairness/fairness_score"] == 0.5
    assert logged["fairness/fairness_score"] == 0.5
    assert fake_wandb.run.summary["fairness/fairness_score_status"] == "assessed"


def test_mlflow_still_rejects_a_non_report(fake_mlflow):
    """Negative case: the type guard is not weakened by the None handling."""
    with pytest.raises(TypeError):
        log_fairness_to_mlflow("not a report", log_artifacts=False)


# --------------------------------------------------------------------------
# Defect 2: the all-groups-dropped warning must not promise sentinels
# --------------------------------------------------------------------------


def test_all_groups_dropped_warning_says_nan_not_defaults():
    gm = GroupManager(np.array(["a"] * 5 + ["b"] * 4), min_group_size=30)

    with pytest.warns(UserWarning) as record:
        assert gm.get_valid_groups() == []

    text = str(record[0].message)
    assert "NaN" in text
    assert "not assessable" in text
    # The retired promise must be gone: it is what made a 0.0 read as measured.
    assert "default values" not in text
    assert "0.0 for differences" not in text
    assert "1.0 for ratios" not in text


# --------------------------------------------------------------------------
# Defect 3: the partial-drop warning must be honest with a single survivor
# --------------------------------------------------------------------------


def _single_survivor_data():
    """100 samples in group A (never selected), 5 in group B (always selected).

    With min_group_size=30 only A survives, so no pair remains to compare.
    """
    sensitive = np.array(["A"] * 100 + ["B"] * 5)
    y_pred = np.array([0] * 100 + [1] * 5)
    y_true = np.array([0] * 100 + [1] * 5)
    return y_true, y_pred, sensitive


def test_single_survivor_warning_does_not_claim_a_computation():
    y_true, y_pred, sensitive = _single_survivor_data()

    with pytest.warns(UserWarning) as record:
        value = demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=30)

    assert np.isnan(value), "one surviving group cannot yield a measured difference"

    texts = [str(w.message) for w in record]
    dropped = [t for t in texts if "Excluding 1 group(s)" in t]
    assert dropped, f"expected the dropped-group warning, got {texts}"
    text = dropped[0]
    assert "no pair to compare" in text
    assert "NaN" in text
    assert "not assessable" in text
    # The false claim: nothing is computed over the survivor.
    assert "computed over the remaining groups only" not in text


def test_two_survivors_still_report_a_real_computation():
    """Guard against over-correcting: with 2+ survivors the metric IS computed."""
    sensitive = np.array(["A"] * 60 + ["B"] * 60 + ["C"] * 5)
    y_pred = np.array([0] * 60 + [1] * 60 + [1] * 5)
    y_true = np.array([0] * 60 + [1] * 60 + [1] * 5)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        value = demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=30)

    assert value == pytest.approx(1.0)
    texts = [str(w.message) for w in record if "Excluding 1 group(s)" in str(w.message)]
    assert texts, "expected the dropped-group warning for group C"
    assert "computed over the remaining groups only" in texts[0]
    assert "no pair to compare" not in texts[0]
