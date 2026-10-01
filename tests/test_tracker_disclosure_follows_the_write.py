"""The disclosure on an experiment-tracker run must follow the WRITE.

``_unmeasured_note`` is the single root both ``log_fairness_to_mlflow`` and
``log_fairness_to_wandb`` use to say what they could not log, and its own docstring
states the invariant: "the note follows the write rather than the dict's key set".
On 2026-09-29 that was measured at both writers and was false in two ways.

1. ``include_group_stats=False``, a first-class parameter of both writers. On a
   report with TWO FULLY MEASURED groups:

       group series written:                 []
       fairness.n_group_stats_not_measured = 0
       fairness.group_stats_not_measured   = "[]"
       warnings:                             []

   A panel reader saw zero bars and every disclosure field saying nothing had been
   left out, which is worse than the one-bar case the note exists for. Suppressed
   is its OWN state: the values were measured and the caller chose not to log
   them, so calling them "not measured" would be the mirror defect.

2. ``if not isinstance(ci_data, dict): continue``, and the same test for the effect
   sizes and the per-group stats, dropped the disclosure together with the write.
   With ``metrics_with_ci = {'demographic_parity_difference': (0.42, 0.31, 0.53)}``
   and ``effect_sizes = {'F_vs_M': (-0.81,)}``:

       ci/effect series logged:              []
       n_intervals_not_measured    = 0,  intervals_not_measured    = "[]"
       n_effect_sizes_not_measured = 0,  effect_sizes_not_measured = "[]"
       warnings:                             []

   Four interval series and one effect-size series written as nothing, with all
   four disclosure pairs reading clean.

BOTH writers are asserted for BOTH defects, never one inferred from the other: the
two are separate capabilities with separate pins, and the shared root is exactly
what makes a half-fix look complete.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from vfairness.evaluation.vfairness_metrics.integrations import (
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)


class _FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.metrics: dict = {}
        self.params: dict = {}
        self.tags: dict = {}

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, name):  # pragma: no cover - artifacts are off here
        raise AssertionError("log_artifacts=False in these tests")


class _FakeWandbRun:
    def __init__(self):
        self.summary: dict = {}


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = _FakeWandbRun()
        self.logged: dict = {}

    def log(self, data):
        self.logged.update(data)

    def log_artifact(self, artifact):  # pragma: no cover - artifacts are off here
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


def _report_with_measured_groups():
    """Two groups, every per-group statistic a real measurement."""
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.21},
        "group_stats": {
            "M": {"size": 150, "positive_rate": 0.62, "tpr": 0.78},
            "F": {"size": 150, "positive_rate": 0.41, "tpr": 0.55},
        },
        "assessment": {"fairness_score": 0.7},
        "data_info": {"n_samples": 300, "n_groups": 2},
    }


def _report_with_tuple_shaped_entries():
    """The (value, lower, upper) tuple shape, passed straight through by a caller."""
    return {
        "task_type": "classification",
        "metrics": {},
        "metrics_with_ci": {"demographic_parity_difference": (0.42, 0.31, 0.53)},
        "effect_sizes": {"F_vs_M": (-0.81,)},
        "group_stats": {},
        "assessment": {},
        "data_info": {"n_samples": 300, "n_groups": 2},
    }


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


# ── 1. include_group_stats=False ──────────────────────────────────────────────


def test_mlflow_says_the_group_panel_was_suppressed_rather_than_clean(fake_mlflow):
    log_fairness_to_mlflow(
        _report_with_measured_groups(), log_artifacts=False, include_group_stats=False
    )
    params = fake_mlflow.params
    assert not [k for k in fake_mlflow.metrics if ".group." in k], "the flag stopped working"
    assert params["fairness.group_stats_status"] == "suppressed_by_caller"
    assert params["fairness.n_group_stats_suppressed"] == 6, params[
        "fairness.group_stats_suppressed"
    ]
    assert "M.tpr" in params["fairness.group_stats_suppressed"]
    # NOT folded into not_measured: those values were measured.
    assert params["fairness.n_group_stats_not_measured"] == 0
    assert "not measured" not in params["fairness.group_stats_status"]


def test_wandb_says_the_group_panel_was_suppressed_rather_than_clean(fake_wandb):
    log_fairness_to_wandb(
        _report_with_measured_groups(), log_artifacts=False, include_group_stats=False
    )
    summary = fake_wandb.run.summary
    assert not [k for k in fake_wandb.logged if "/group/" in k], "the flag stopped working"
    assert summary["fairness/group_stats_status"] == "suppressed_by_caller"
    assert summary["fairness/n_group_stats_suppressed"] == 6
    assert "M/tpr" in summary["fairness/group_stats_suppressed"]
    assert summary["fairness/n_group_stats_not_measured"] == 0


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_control_a_logged_group_panel_says_logged_and_writes_its_bars(
    backend, fake_mlflow, fake_wandb
):
    """OVER-CORRECTION CONTROL, with the real numbers asserted: with the flag left
    at its default every series is written and the status says so."""
    report = _report_with_measured_groups()
    if backend == "mlflow":
        log_fairness_to_mlflow(report, log_artifacts=False)
        assert fake_mlflow.metrics["fairness.group.M.tpr"] == 0.78
        assert fake_mlflow.metrics["fairness.group.F.tpr"] == 0.55
        assert fake_mlflow.params["fairness.group_stats_status"] == "logged"
        assert fake_mlflow.params["fairness.n_group_stats_suppressed"] == 0
        assert fake_mlflow.params["fairness.group_stats_suppressed"] == "[]"
    else:
        log_fairness_to_wandb(report, log_artifacts=False)
        assert fake_wandb.logged["fairness/group/M/tpr"] == 0.78
        assert fake_wandb.logged["fairness/group/F/tpr"] == 0.55
        assert fake_wandb.run.summary["fairness/group_stats_status"] == "logged"
        assert fake_wandb.run.summary["fairness/n_group_stats_suppressed"] == 0


# ── 2. an entry whose shape these loggers cannot read ─────────────────────────


def test_mlflow_names_a_tuple_shaped_interval_it_wrote_as_nothing(fake_mlflow):
    _logged, msgs = _caught(
        lambda: log_fairness_to_mlflow(_report_with_tuple_shaped_entries(), log_artifacts=False)
    )
    params = fake_mlflow.params
    assert not [k for k in fake_mlflow.metrics if "ci_" in k or "effect_size" in k]
    assert params["fairness.n_intervals_not_measured"] == 4, params[
        "fairness.intervals_not_measured"
    ]
    for suffix in ("value", "ci_lower", "ci_upper", "std_error"):
        assert (
            f"demographic_parity_difference.{suffix}" in params["fairness.intervals_not_measured"]
        )
    assert params["fairness.n_effect_sizes_not_measured"] == 1
    assert "F_vs_M" in params["fairness.effect_sizes_not_measured"]
    assert params["fairness.n_unreadable_entries"] == 2
    assert (
        "metrics_with_ci.demographic_parity_difference (tuple)"
        in params["fairness.unreadable_entries"]
    )
    assert any("not in a shape these loggers can read" in m for m in msgs), msgs


def test_wandb_names_a_tuple_shaped_interval_it_wrote_as_nothing(fake_wandb):
    _logged, msgs = _caught(
        lambda: log_fairness_to_wandb(_report_with_tuple_shaped_entries(), log_artifacts=False)
    )
    summary = fake_wandb.run.summary
    assert not [k for k in fake_wandb.logged if "ci_" in k or "effect_size" in k]
    assert summary["fairness/n_intervals_not_measured"] == 4
    assert "demographic_parity_difference/ci_lower" in summary["fairness/intervals_not_measured"]
    assert summary["fairness/n_effect_sizes_not_measured"] == 1
    assert summary["fairness/n_unreadable_entries"] == 2
    assert "effect_sizes/F_vs_M (tuple)" in summary["fairness/unreadable_entries"]
    assert any("not in a shape these loggers can read" in m for m in msgs), msgs


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_an_unreadable_group_entry_is_named_too(backend, fake_mlflow, fake_wandb):
    """The third copy of the same shape test, on the per-group block."""
    report = _report_with_measured_groups()
    report["group_stats"]["X"] = (0.5, 0.6)
    logger = log_fairness_to_mlflow if backend == "mlflow" else log_fairness_to_wandb
    _logged, msgs = _caught(lambda: logger(report, log_artifacts=False))
    fields = fake_mlflow.params if backend == "mlflow" else fake_wandb.run.summary
    key = "fairness.unreadable_entries" if backend == "mlflow" else "fairness/unreadable_entries"
    assert "group_stats" in fields[key] and "X" in fields[key], fields[key]
    assert any("not in a shape these loggers can read" in m for m in msgs)


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_control_a_readable_report_reports_no_unreadable_entry_and_no_warning(
    backend, fake_mlflow, fake_wandb
):
    """OVER-CORRECTION CONTROL: the new disclosure is emitted on every run, so on a
    well-shaped report it reads 0 and "[]" rather than being absent, and the shape
    warning must NOT fire."""
    report = _report_with_measured_groups()
    report["metrics_with_ci"] = {
        "demographic_parity_difference": {
            "point_estimate": 0.42,
            "lower_bound": 0.31,
            "upper_bound": 0.53,
            "standard_error": 0.06,
        }
    }
    report["effect_sizes"] = {"F_vs_M": {"cohens_d_positive_rate": -0.81}}
    logger = log_fairness_to_mlflow if backend == "mlflow" else log_fairness_to_wandb
    _logged, msgs = _caught(lambda: logger(report, log_artifacts=False))
    if backend == "mlflow":
        fields, written = fake_mlflow.params, fake_mlflow.metrics
        assert written["fairness.demographic_parity_difference.ci_lower"] == 0.31
        assert written["fairness.effect_size.F_vs_M.cohens_d"] == -0.81
        assert fields["fairness.n_unreadable_entries"] == 0
        assert fields["fairness.unreadable_entries"] == "[]"
        assert fields["fairness.n_intervals_not_measured"] == 0
    else:
        fields, written = fake_wandb.run.summary, fake_wandb.logged
        assert written["fairness/demographic_parity_difference/ci_lower"] == 0.31
        assert written["fairness/effect_size/F_vs_M/cohens_d"] == -0.81
        assert fields["fairness/n_unreadable_entries"] == 0
        assert fields["fairness/n_intervals_not_measured"] == 0
    assert not [m for m in msgs if "shape" in m], msgs
