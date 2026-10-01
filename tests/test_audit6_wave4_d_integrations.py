"""Sixth-iteration audit, wave 4 lane D: the inert parameters in integrations.py.

Five documented parameters were accepted and never read. Each was reproduced by
execution against real data before anything was changed, and each is disposed of
individually here.

S-08, ``auto_log_fairness``:

* ``metrics`` was documented as the list of fairness metrics to compute and was
  ignored: asking for one metric put SIX metric series on the tracker. IMPLEMENTED
  as a filter over the report the analyzer produced, because the report already
  keys its metrics by name.
* ``thresholds`` was documented as "logged but not enforced by default" and was
  not even LOGGED, so the one thing the docstring promised did not happen.
  IMPLEMENTED: recorded as run parameters, next to an explicit
  ``thresholds_enforced = "false"`` marker, because a threshold sitting beside a
  value looks like a verdict and no verdict was made.
* ``protected_attr_column`` was documented as "column name if returning a
  DataFrame" and the wrapper has no DataFrame return path at all. REMOVED, so
  passing it raises TypeError instead of being silently discarded.

S-12, ``create_fairness_callback``:

* ``sensitive_attr_column`` is the REQUIRED first positional argument and was
  read nowhere: a callback built for "gender" and handed the "age_band" column
  reported age disparity under a gender label. IMPLEMENTED as a check, not a
  selection: three states, matched / mismatched (refused) / no name to check
  against (a bare numpy array, which proceeds and is never recorded as verified).
  It must NOT select a column out of a DataFrame, because the analyzer treats a
  multi-column frame as an intersectional attribute.
* ``metrics`` was ignored; the callback always returned everything
  ``compute_all_metrics`` produced. IMPLEMENTED as a filter, with a requested
  metric the analyzer never produced refused rather than answered with the others.

Two siblings of the same shape, found in the same file and fixed here:

* the confidence-interval BOUNDS in both logging paths used a bare ``np.isnan``
  while the point estimate beside them used ``_is_measured``. Executed: a None
  bound raised TypeError, and an INFINITE bound (an interval that was never
  estimated) was written to the tracker as ci_lower=-inf / ci_upper=inf.
* every MLflow run was tagged ``library_version = "1.1.0"``, a literal no release
  ever carried (the package is 0.1.0), so a run could not be reproduced from its
  own audit trail.

Neither mlflow nor wandb is exercised for real: recording stubs are injected into
``sys.modules``, so what these tests pin is exactly the arguments vfairness passes.
"""

import inspect
import math
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.integrations import (
    auto_log_fairness,
    create_fairness_callback,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)

# --------------------------------------------------------------------------
# Data. Deterministic and hand-built, so every assertion below names the exact
# number the analyzer produces rather than a range it might fall in.
#
# 200 rows of "M" predicted positive 160 times (rate 0.80) and 200 rows of "F"
# predicted positive 40 times (rate 0.20):
#   demographic_parity_difference = 0.60   demographic_parity_ratio      = 0.25
#   equal_opportunity_difference  = 0.60   equalized_odds_difference     = 0.60
#   predictive_parity_difference  = 0.00   fairness_score                = 0.20
# --------------------------------------------------------------------------

DPD = 0.6000000000000001
DP_RATIO = 0.25
EQ_OPP = 0.6000000000000001
EQ_ODDS = 0.6000000000000001
PRED_PARITY = 0.0
FAIRNESS_SCORE = 0.2
ALL_METRIC_NAMES = (
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "predictive_parity_difference",
)


def _unfair_data():
    n = 200
    sens = np.array(["M"] * n + ["F"] * n)
    y_true = np.array(([1, 0] * (n // 2)) * 2)
    y_pred = np.concatenate([np.array([1] * 160 + [0] * 40), np.array([1] * 40 + [0] * 160)])
    return y_true, y_pred, sens


def _fair_data(n=400):
    """Two large groups with identical, perfect predictions."""
    sens = np.array(["M"] * (n // 2) + ["F"] * (n // 2))
    y_true = np.array([1, 0] * (n // 2))
    return y_true, y_true.copy(), sens


def _value(result):
    """Point estimate of a metric result, which may be a MetricResult."""
    return getattr(result, "value", result)


# --------------------------------------------------------------------------
# Recording stubs. mlflow is installed in this environment and wandb is not;
# both are replaced so no real run is needed and nothing is uploaded.
# --------------------------------------------------------------------------


class _FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.metrics = {}
        self.params = {}
        self.tags = {}
        self.artifacts = []

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        # The real SDK takes a float. Anything unmeasurable reaching a metric
        # series is the defect this file is partly about, so it is caught here
        # rather than recorded and asserted about later.
        assert isinstance(value, float) and math.isfinite(value), f"log_metric({key!r}, {value!r})"
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, name):
        self.artifacts.append((path, name))


class _FakeWandbArtifact:
    def add_file(self, path, name=None):
        pass


class _FakeWandbRun:
    def __init__(self):
        self.summary = {}


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = _FakeWandbRun()
        self.logged = {}
        self.artifacts = []

    def log(self, data):
        self.logged.update(data)

    def Artifact(self, name, type=None):  # noqa: N802 - mirrors the wandb SDK name
        return _FakeWandbArtifact()

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


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


def _decorated(**kwargs):
    y_true, y_pred, sens = _unfair_data()

    @auto_log_fairness(**kwargs)
    def evaluate():
        return y_pred, y_true, sens

    return evaluate


def _metric_series(recorded, prefix, separator):
    """Top-level metric keys, i.e. not the per-group statistics."""
    group_marker = f"{separator}group{separator}"
    return {
        key: value
        for key, value in recorded.items()
        if key.startswith(f"{prefix}{separator}") and group_marker not in key
    }


# --------------------------------------------------------------------------
# S-08a: `metrics` was documented and ignored
# --------------------------------------------------------------------------


class TestAutoLogMetricsSelection:
    def test_requested_single_metric_is_the_only_series_logged(self, fake_mlflow):
        """REFUSAL PIN. Asking for one metric used to log all six."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(backend="mlflow", metrics=["demographic_parity_difference"], prefix="p")()

        series = _metric_series(fake_mlflow.metrics, "p", ".")
        assert series["p.demographic_parity_difference"] == pytest.approx(DPD)
        for unrequested in (
            "p.demographic_parity_ratio",
            "p.equal_opportunity_difference",
            "p.equalized_odds_difference",
            "p.predictive_parity_difference",
        ):
            assert unrequested not in series, f"{unrequested} was logged and was not requested"

    def test_requested_metric_the_analyzer_never_produced_is_not_logged(self, fake_mlflow):
        """REFUSAL PIN. Nothing is substituted for a metric that does not exist
        here, and the caller is told by name which request went unlogged."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _decorated(backend="mlflow", metrics=["disparate_impact"], prefix="r")()

        series = _metric_series(fake_mlflow.metrics, "r", ".")
        assert "r.disparate_impact" not in series
        assert not [key for key in series if "disparate" in key]
        messages = [str(w.message) for w in caught]
        assert any("disparate_impact" in m and "not produced" in m for m in messages), (
            f"the dropped request was not named in a warning: {messages}"
        )

    def test_metrics_none_still_logs_every_measured_series(self, fake_mlflow):
        """OVER-CORRECTION CONTROL. The default path must log exactly what it
        logged before the filter existed, at the values it measured."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(backend="mlflow", prefix="q")()

        assert fake_mlflow.metrics["q.demographic_parity_difference"] == pytest.approx(DPD)
        assert fake_mlflow.metrics["q.demographic_parity_ratio"] == pytest.approx(DP_RATIO)
        assert fake_mlflow.metrics["q.equal_opportunity_difference"] == pytest.approx(EQ_OPP)
        assert fake_mlflow.metrics["q.equalized_odds_difference"] == pytest.approx(EQ_ODDS)
        assert fake_mlflow.metrics["q.predictive_parity_difference"] == pytest.approx(PRED_PARITY)
        assert fake_mlflow.metrics["q.fairness_score"] == pytest.approx(FAIRNESS_SCORE)
        # The per-run summary is not a metric series and is logged either way.
        assert fake_mlflow.metrics["q.group.M.positive_rate"] == pytest.approx(0.8)
        assert fake_mlflow.metrics["q.group.F.positive_rate"] == pytest.approx(0.2)
        assert fake_mlflow.params["q.n_samples"] == 400

    def test_filtering_does_not_change_the_measured_value(self, fake_mlflow):
        """OVER-CORRECTION CONTROL. Selecting a metric must not perturb it."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(backend="mlflow", metrics=["demographic_parity_ratio"], prefix="s")()

        assert fake_mlflow.metrics["s.demographic_parity_ratio"] == pytest.approx(DP_RATIO)


# --------------------------------------------------------------------------
# S-08b: `thresholds` documented as logged, and not logged
# --------------------------------------------------------------------------


class TestAutoLogThresholdsAreRecorded:
    def test_thresholds_reach_the_mlflow_run(self, fake_mlflow):
        """REFUSAL PIN. Nothing recorded the thresholds at all before this."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(
                backend="mlflow",
                thresholds={
                    "demographic_parity_difference": 0.05,
                    "equalized_odds_difference": 0.1,
                },
                prefix="p",
            )()

        assert fake_mlflow.params["p.threshold.demographic_parity_difference"] == 0.05
        assert fake_mlflow.params["p.threshold.equalized_odds_difference"] == 0.1

    def test_recorded_thresholds_say_they_were_not_enforced(self, fake_mlflow):
        """REFUSAL PIN. A threshold beside a measured value reads as a verdict.
        Nothing here compared them, so the run must say so."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(
                backend="mlflow",
                thresholds={"demographic_parity_difference": 0.05},
                prefix="p",
            )()

        assert fake_mlflow.tags["p.thresholds_enforced"] == "false"
        # 0.60 breaches 0.05 by a wide margin and the decorator still returns:
        # it is a logger, not a gate, and the marker is what keeps that honest.
        assert fake_mlflow.metrics["p.demographic_parity_difference"] == pytest.approx(DPD)

    def test_a_threshold_is_never_logged_as_a_metric_series(self, fake_mlflow):
        """REFUSAL PIN. A metric series is what a dashboard plots as something
        this run measured. A threshold is configuration the caller supplied."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(
                backend="mlflow",
                thresholds={"demographic_parity_difference": 0.05},
                prefix="p",
            )()

        assert not [key for key in fake_mlflow.metrics if "threshold" in key]
        assert 0.05 not in fake_mlflow.metrics.values()

    def test_no_thresholds_records_no_enforcement_marker(self, fake_mlflow):
        """OVER-CORRECTION CONTROL. A caller who quoted no thresholds gets no
        threshold keys and no marker, and still gets the measurements."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(backend="mlflow", prefix="q")()

        assert not [key for key in fake_mlflow.params if "threshold" in key]
        assert "q.thresholds_enforced" not in fake_mlflow.tags
        assert fake_mlflow.metrics["q.demographic_parity_difference"] == pytest.approx(DPD)

    def test_thresholds_reach_the_wandb_run_summary(self, fake_wandb):
        """REFUSAL PIN, W&B path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _decorated(
                backend="wandb",
                thresholds={"demographic_parity_difference": 0.05},
                prefix="w",
            )()

        assert fake_wandb.run.summary["w/threshold/demographic_parity_difference"] == 0.05
        assert fake_wandb.run.summary["w/thresholds_enforced"] == "false"
        assert fake_wandb.logged["w/demographic_parity_difference"] == pytest.approx(DPD)


# --------------------------------------------------------------------------
# S-08c: `protected_attr_column` described a return path that does not exist
# --------------------------------------------------------------------------


class TestAutoLogProtectedAttrColumnIsGone:
    def test_the_parameter_no_longer_exists(self):
        """REFUSAL PIN. It documented a DataFrame return path the wrapper does
        not have, so it is removed rather than accepted and discarded."""
        assert "protected_attr_column" not in inspect.signature(auto_log_fairness).parameters

    def test_passing_it_is_refused_loudly(self):
        """REFUSAL PIN. Silently ignoring it is what the audit found."""
        with pytest.raises(TypeError, match="protected_attr_column"):
            auto_log_fairness(backend="mlflow", protected_attr_column="gender")

    def test_the_docstring_no_longer_promises_a_dataframe_return_path(self):
        """The result must not carry a claim the code does not honour."""
        doc = auto_log_fairness.__doc__ or ""
        assert "Column name if returning a DataFrame" not in doc

    def test_the_supported_return_shapes_still_work(self, fake_mlflow):
        """OVER-CORRECTION CONTROL. Both documented shapes still log."""
        y_true, y_pred, sens = _unfair_data()

        @auto_log_fairness(backend="mlflow", prefix="t")
        def as_dict():
            return {"y_pred": y_pred, "y_true": y_true, "sensitive_attr": sens}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            as_dict()

        assert fake_mlflow.metrics["t.demographic_parity_difference"] == pytest.approx(DPD)


# --------------------------------------------------------------------------
# S-12a: `sensitive_attr_column`, a required positional read nowhere
# --------------------------------------------------------------------------


class TestCallbackChecksTheColumnItNames:
    def test_a_differently_named_column_is_refused(self):
        """REFUSAL PIN. A callback built for 'gender' and handed 'age_band' used
        to report age disparity under a gender label."""
        y_true, y_pred, sens = _unfair_data()
        age_band = pd.Series(np.where(sens == "M", "young", "old"), name="age_band")
        callback = create_fairness_callback("gender")

        with pytest.raises(ValueError, match="monitors 'gender'.*passed the column 'age_band'"):
            callback(y_true, y_pred, age_band)

    def test_a_dataframe_without_that_column_is_refused(self):
        """REFUSAL PIN."""
        y_true, y_pred, sens = _unfair_data()
        frame = pd.DataFrame({"age_band": np.where(sens == "M", "young", "old")})
        callback = create_fairness_callback("gender")

        with pytest.raises(ValueError, match="none of them is 'gender'"):
            callback(y_true, y_pred, frame)

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_a_name_that_can_never_match_is_refused_at_construction(self, bad):
        """REFUSAL PIN. An empty name would make the check silently inert."""
        with pytest.raises(ValueError, match="non-empty column name"):
            create_fairness_callback(bad)

    def test_the_matching_column_is_accepted_and_measured(self):
        """OVER-CORRECTION CONTROL. The named column still computes its real
        answer, unchanged by the check."""
        y_true, y_pred, sens = _unfair_data()
        gender = pd.Series(sens, name="gender")
        results = create_fairness_callback("gender")(y_true, y_pred, gender)

        assert _value(results["demographic_parity_difference"]) == pytest.approx(DPD)
        assert _value(results["demographic_parity_ratio"]) == pytest.approx(DP_RATIO)

    def test_an_unnamed_array_is_could_not_check_and_still_runs(self):
        """OVER-CORRECTION CONTROL, and the third state. A numpy array carries
        no column name. That is not a mismatch and must not be refused, and it
        must not be recorded anywhere as a verified column either."""
        y_true, y_pred, sens = _unfair_data()
        results = create_fairness_callback("gender")(y_true, y_pred, sens)

        assert _value(results["demographic_parity_difference"]) == pytest.approx(DPD)
        assert not [key for key in results if "column" in key or "verified" in key]

    def test_a_dataframe_holding_the_column_is_not_narrowed_to_it(self):
        """REFUSAL PIN against the wrong fix. Selecting the named column out of a
        multi-column frame would silently drop the intersectional analysis the
        caller asked for: over gender alone the disparity measures 0.60, over
        gender x age_band it measures 1.00."""
        y_true, y_pred, sens = _unfair_data()
        age_band = np.array((["young"] * 100 + ["old"] * 100) * 2)
        frame = pd.DataFrame({"gender": sens, "age_band": age_band})

        results = create_fairness_callback("gender")(y_true, y_pred, frame)

        assert _value(results["demographic_parity_difference"]) == pytest.approx(1.0)
        assert sorted(FairnessAnalyzer(y_true, y_pred, frame).groups) == [
            "F_old",
            "F_young",
            "M_old",
            "M_young",
        ]


# --------------------------------------------------------------------------
# S-12b: `metrics`, accepted and ignored
# --------------------------------------------------------------------------


class TestCallbackMetricsSelection:
    def test_only_the_requested_metrics_are_returned(self):
        """REFUSAL PIN. Asking for one metric used to return all five."""
        y_true, y_pred, sens = _unfair_data()
        callback = create_fairness_callback("gender", metrics=["demographic_parity_ratio"])

        results = callback(y_true, y_pred, sens)

        assert list(results) == ["demographic_parity_ratio"]
        assert _value(results["demographic_parity_ratio"]) == pytest.approx(DP_RATIO)

    def test_a_requested_metric_that_does_not_exist_here_is_refused(self):
        """REFUSAL PIN. Returning the other metrics answers a question nobody
        asked; the caller wanted disparate impact and it was never computed."""
        y_true, y_pred, sens = _unfair_data()
        callback = create_fairness_callback("gender", metrics=["disparate_impact"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="disparate_impact.*not produced"):
                callback(y_true, y_pred, sens)

    def test_a_threshold_outside_the_metric_list_is_refused_at_construction(self):
        """REFUSAL PIN. The gate and the monitored list must not disagree, and
        the contradiction is caught before the first epoch runs."""
        with pytest.raises(ValueError, match="which metrics=.* excludes"):
            create_fairness_callback(
                "gender",
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_ratio": 0.8},
                fail_on_violation=True,
            )

    def test_metrics_none_still_returns_every_metric(self):
        """OVER-CORRECTION CONTROL, at measured values."""
        y_true, y_pred, sens = _unfair_data()
        results = create_fairness_callback("gender")(y_true, y_pred, sens)

        assert sorted(results) == sorted(ALL_METRIC_NAMES)
        assert _value(results["demographic_parity_difference"]) == pytest.approx(DPD)
        assert _value(results["equal_opportunity_difference"]) == pytest.approx(EQ_OPP)
        assert _value(results["equalized_odds_difference"]) == pytest.approx(EQ_ODDS)
        assert _value(results["predictive_parity_difference"]) == pytest.approx(PRED_PARITY)

    def test_the_gate_still_fires_on_a_measured_violation(self):
        """OVER-CORRECTION CONTROL for the pre-existing fail-closed gate: the
        filter must not smuggle a violation past it. 0.60 against 0.10."""
        y_true, y_pred, sens = _unfair_data()
        callback = create_fairness_callback(
            "gender",
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )

        with pytest.raises(ValueError, match="Fairness violation"):
            callback(y_true, y_pred, sens)

    def test_the_gate_still_passes_measured_fair_data(self):
        """OVER-CORRECTION CONTROL. Perfect, fully measured predictions."""
        y_true, y_pred, sens = _fair_data()
        callback = create_fairness_callback(
            "gender",
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )

        results = callback(y_true, y_pred, sens)

        assert _value(results["demographic_parity_difference"]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Siblings found in the same file
# --------------------------------------------------------------------------


def _ci_report(ci_data):
    return {
        "task_type": "classification",
        "metrics": {},
        "metrics_with_ci": {"demographic_parity_difference": ci_data},
        "group_stats": {},
        "assessment": {},
        "data_info": {},
    }


class TestConfidenceIntervalBoundsAreMeasuredOrAbsent:
    def test_an_infinite_bound_never_reaches_mlflow(self, fake_mlflow):
        """REFUSAL PIN. An infinite interval was never estimated, and it used to
        be written as ci_lower=-inf / ci_upper=inf, which a panel plots."""
        logged = log_fairness_to_mlflow(
            _ci_report(
                {
                    "point_estimate": 0.2,
                    "lower_bound": float("-inf"),
                    "upper_bound": float("inf"),
                    "standard_error": float("inf"),
                }
            ),
            log_artifacts=False,
        )

        assert "fairness.demographic_parity_difference.ci_lower" not in logged
        assert "fairness.demographic_parity_difference.ci_upper" not in logged
        assert "fairness.demographic_parity_difference.std_error" not in logged
        assert not [
            value
            for value in fake_mlflow.metrics.values()
            if isinstance(value, float) and not math.isfinite(value)
        ]

    def test_a_none_bound_does_not_crash_the_run(self, fake_mlflow):
        """REFUSAL PIN. np.isnan(None) raised TypeError and took the whole log
        call with it, which inside auto_log_fairness became a swallowed warning
        and no logging at all."""
        logged = log_fairness_to_mlflow(
            _ci_report({"point_estimate": 0.2, "lower_bound": None, "upper_bound": 0.31}),
            log_artifacts=False,
        )

        assert "fairness.demographic_parity_difference.ci_lower" not in logged
        assert logged["fairness.demographic_parity_difference.ci_upper"] == pytest.approx(0.31)

    def test_measured_bounds_are_still_logged(self, fake_mlflow):
        """OVER-CORRECTION CONTROL, at measured values."""
        log_fairness_to_mlflow(
            _ci_report(
                {
                    "point_estimate": 0.21,
                    "lower_bound": 0.11,
                    "upper_bound": 0.31,
                    "standard_error": 0.05,
                }
            ),
            log_artifacts=False,
        )

        assert fake_mlflow.metrics["fairness.demographic_parity_difference.value"] == pytest.approx(
            0.21
        )
        assert fake_mlflow.metrics[
            "fairness.demographic_parity_difference.ci_lower"
        ] == pytest.approx(0.11)
        assert fake_mlflow.metrics[
            "fairness.demographic_parity_difference.ci_upper"
        ] == pytest.approx(0.31)
        assert fake_mlflow.metrics[
            "fairness.demographic_parity_difference.std_error"
        ] == pytest.approx(0.05)

    def test_wandb_bounds_get_the_same_guard(self, fake_wandb):
        """REFUSAL PIN plus control, W&B path."""
        log_fairness_to_wandb(
            _ci_report({"point_estimate": 0.2, "lower_bound": None, "upper_bound": float("inf")}),
            log_artifacts=False,
        )
        assert "fairness/demographic_parity_difference/ci_lower" not in fake_wandb.logged
        assert "fairness/demographic_parity_difference/ci_upper" not in fake_wandb.logged

        fake_wandb.logged.clear()
        log_fairness_to_wandb(
            _ci_report({"point_estimate": 0.21, "lower_bound": 0.11, "upper_bound": 0.31}),
            log_artifacts=False,
        )
        assert fake_wandb.logged[
            "fairness/demographic_parity_difference/ci_lower"
        ] == pytest.approx(0.11)
        assert fake_wandb.logged[
            "fairness/demographic_parity_difference/ci_upper"
        ] == pytest.approx(0.31)


class TestLoggedLibraryVersionIsTheRealOne:
    def test_the_run_is_tagged_with_the_installed_version(self, fake_mlflow):
        """REFUSAL PIN. Every run was tagged '1.1.0', a version no release ever
        carried, so a run could not be reproduced from its own audit trail."""
        log_fairness_to_mlflow(_ci_report({"point_estimate": 0.2}), log_artifacts=False)

        assert fake_mlflow.tags["fairness.library_version"] == vfairness.__version__
        assert fake_mlflow.tags["fairness.library_version"] != "1.1.0"
