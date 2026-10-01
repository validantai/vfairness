"""BGL-5 batch A-evaluation-3: the eleven overturned rows, closed and pinned.

Every test here asserts the CORRECTED behaviour, so it goes red if the defect
comes back. Each defect's old, measured output is quoted in the docstring of the
test that pins it, so the record of what was wrong survives the fix. The
demonstration file tests/test_bgl4_evaluation_3.py holds the same subjects with
the assertions inverted to the fix.

Every pin in this file was SABOTAGED: the defect was put back, the named test was
confirmed red, and the source was restored and diffed byte-identical. The
over-correction controls are in the same file on purpose: a fix that refuses
everything passes every refusal test and destroys the library, so each defect's
healthy case is asserted here with its actual measured number.
"""

import math
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    classify_column_roles,
    rank_fairness_issues,
)
from vfairness.evaluation.vfairness_metrics.explainer import (
    FairExplAIner,
    explain_fairness_report,
)
from vfairness.evaluation.vfairness_metrics.integrations import (
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)
from vfairness.evaluation.vfairness_metrics.regression import compute_regression_effect_sizes
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report
from vfairness.evaluation.vfairness_metrics.robustness import (
    permutation_test_equal_opportunity,
    robust_fairness_comparison,
)

# Recording stubs, the same shape as the BGL-3 and BGL-4 fixtures so the
# comparison with the measured "before" is direct.


class _FakeMlflow(types.ModuleType):
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

    def log_artifact(self, path, name):
        pass


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

    def Artifact(self, name, type):  # noqa: N802, mirrors the W&B SDK
        return None

    def log_artifact(self, artifact):
        pass


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


# ---------------------------------------------------------------------------
# DEFECT A. `_could_not_check_reason` tested `isinstance(value, bool)`, and
# numpy's boolean is NOT a Python bool. Three overturned rows: explain_metric,
# explain_report, explain_fairness_report.
# ---------------------------------------------------------------------------


def test_explain_metric_refuses_a_numpy_boolean():
    """A np.bool_ metric value is refused, in both directions.

    BEFORE (measured 2026-09-27 at the public entry):
        explain_metric("demographic_parity_difference", np.False_)
          -> severity "info", "Excellent! The difference of 0.0000 indicates
             near-perfect fairness across groups."
        explain_metric("demographic_parity_difference", np.True_)
          -> severity "critical", "Critical. The difference of 1.0000 ..."
    `isinstance(np.True_, bool)` is False and `np.isnan(np.True_)` is also
    False, so the guard written for the Python bool was bypassed by the boolean
    type every numpy and pandas comparison returns. The same class already
    refused np.bool_ by name in `_finite_or_none`, so one class disagreed with
    itself about one input.
    """
    ex = FairExplAIner()

    # The premise, so this test cannot pass for the wrong reason.
    assert isinstance(np.True_, bool) is False

    for value in (np.False_, np.True_, False, True):
        card = ex.explain_metric("demographic_parity_difference", value)
        assert card.severity == "could_not_check", f"{value!r} was graded"
        assert "COULD NOT CHECK" in card.evaluation
        assert "Excellent!" not in card.evaluation
        assert "NOT MEASURED" in card.recommendation


def test_explain_metric_still_grades_a_real_number():
    """OVER-CORRECTION CONTROL, with the actual values asserted.

    A measured 0.0 is a MEASUREMENT of perfect parity and must not be swept into
    the third state with the flags.
    """
    ex = FairExplAIner()

    perfect = ex.explain_metric("demographic_parity_difference", 0.0)
    assert perfect.severity == "info"
    assert "The difference of 0.0000 indicates near-perfect fairness" in perfect.evaluation

    breach = ex.explain_metric("demographic_parity_difference", 0.42)
    assert breach.severity == "critical"
    assert "The difference of 0.4200" in breach.evaluation

    small = ex.explain_metric("demographic_parity_difference", 0.01)
    assert small.severity == "info"


def test_explain_report_keeps_a_could_not_check_card_for_a_numpy_boolean(recwarn):
    """The card, the summary line and the warning all fire for a np.bool_.

    BEFORE, through the public wrapper on
    {"demographic_parity_difference": np.False_, "equalized_odds_difference": 0.42}:
        cards: both keys present, but dp severity "info" with "Excellent! The
        difference of 0.0000 indicates near-perfect fairness ...", the
        recommendation written for a passing metric, recwarn EMPTY and no
        "NOT GRADED" line in the summary.
    The warning's trigger is `_could_not_check_reason(...) == "unmeasurable"`,
    which answered None for a numpy boolean, so the disclosure could not fire
    even though the card loop reached every key.
    """
    report = {
        "metrics": {"demographic_parity_difference": np.False_, "equalized_odds_difference": 0.42},
        "group_stats": {},
        "thresholds_used": {},
    }
    out = explain_fairness_report(report)

    card = out["metrics"]["demographic_parity_difference"]
    assert card["severity"] == "could_not_check"
    assert "COULD NOT CHECK" in card["evaluation"]
    assert "Excellent!" not in card["evaluation"]
    # The measured metric beside it is still graded.
    assert out["metrics"]["equalized_odds_difference"]["severity"] == "critical"
    # The disclosure reaches the two surfaces a person reads.
    assert "NOT GRADED" in out["summary"]
    named = [w for w in recwarn.list if "not numbers" in str(w.message)]
    assert named, "no warning named the ungraded metric"
    assert "demographic_parity_difference=np.False_" in str(named[0].message) or (
        "demographic_parity_difference" in str(named[0].message)
    )


def test_explain_report_is_silent_on_a_real_report():
    """OVER-CORRECTION CONTROL. A real report grades every card and warns nothing.

    Run under `simplefilter("error")` so any new warning fails the test, which is
    what catches a fix that starts refusing measured numbers.
    """
    rng = np.random.default_rng(12)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, gender)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = explain_fairness_report(report, include_statistical=False)

    severities = {name: card["severity"] for name, card in out["metrics"].items()}
    assert severities, "the healthy report produced no cards at all"
    assert "could_not_check" not in severities.values()
    assert "NOT GRADED" not in out["summary"]


# ---------------------------------------------------------------------------
# DEFECT B. The infinity carve-out in `_effect_size_unmeasured_reason` was
# argued for risk_ratio and odds_ratio and applied to every effect type.
# Overturned row: explainer.FairExplAIner.explain_effect_size.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_explain_effect_size_refuses_an_infinite_cohens_d(value):
    """An infinite standardised mean difference is an empty denominator.

    BEFORE (measured at the public entry):
        explain_effect_size("cohens_d", float("inf"), "A", "B")
          -> severity "high", "Comparing A vs B: Large effect size - the
             difference is very substantial."
        the same for -inf.
    The carve-out ignored `effect_type` entirely, although its own comment
    justifies it only for the two ratios, which reach inf from a table that WAS
    measured. Cohen's d reaches inf only by dividing by a zero pooled standard
    deviation, i.e. exactly the case the predicate says it rejects.
    """
    card = FairExplAIner().explain_effect_size("cohens_d", value, "A", "B")

    assert card.severity == "could_not_check"
    assert "COULD NOT CHECK" in card.evaluation
    assert "Large effect size" not in card.evaluation
    assert "requires action" not in card.recommendation
    # Called directly, the ladder refuses it too: these methods are public paths.
    assert FairExplAIner()._interpret_cohens_d(value)[1] == "could_not_check"


def test_the_infinity_carve_out_still_holds_for_the_two_ratios():
    """OVER-CORRECTION CONTROL. Total exclusion is a finding, not a refusal.

    An infinite risk or odds ratio comes from a measured table in which one arm
    received no positive outcomes at all. Folding it into could-not-check would
    delete the strongest disparate-impact reading a pair can produce, which is
    the whole reason the carve-out exists.
    """
    ex = FairExplAIner()

    rr = ex.explain_effect_size("risk_ratio", float("inf"), "A", "B")
    assert rr.severity == "critical"
    assert "Total exclusion" in rr.evaluation

    odds = ex.explain_effect_size("odds_ratio", float("inf"), "A", "B")
    assert odds.severity == "critical"
    assert "Total exclusion" in odds.evaluation

    # ...and a measured effect size is still graded on its magnitude.
    measured = ex.explain_effect_size("cohens_d", 0.9, "A", "B")
    assert measured.severity == "high"
    assert "Large effect size" in measured.evaluation
    negligible = ex.explain_effect_size("cohens_d", 0.0, "A", "B")
    assert negligible.severity == "info"
    assert "Negligible effect size" in negligible.evaluation


# ---------------------------------------------------------------------------
# DEFECT C. regression.compute_regression_effect_sizes graded a NaN Cohen's d.
# Overturned row: regression.compute_regression_effect_sizes.
# ---------------------------------------------------------------------------


def test_regression_effect_sizes_refuse_to_interpret_an_unmeasured_d():
    """`abs(nan) < 0.2` is False, so NaN used to fall out of the ladder as "large".

    BEFORE (measured on 80 rows, two groups of 40 whose predictions are each
    CONSTANT at 130 and 100, where the library's own `cohens_d` returns NaN and
    warns that "the standardising denominator is zero"):
        {'A_vs_B': {'cohens_d_predictions': nan, 'cohens_d_residuals': nan,
                    'interpretation': 'large', 'mean_pred_diff': 30.0, ...}}
    "large" is the strongest verdict that field can carry and it was measured by
    nothing. The classification twin of this ladder was fixed for exactly this on
    2026-09-16; the regression copy never got the guard.
    """
    n = 40
    y_true = np.concatenate([np.full(n, 100.0), np.full(n, 100.0)])
    y_pred = np.concatenate([np.full(n, 130.0), np.full(n, 100.0)])
    groups = np.array(["A"] * n + ["B"] * n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_regression_effect_sizes(y_true, y_pred, groups)

    pair = out["A_vs_B"]
    assert math.isnan(pair["cohens_d_predictions"])
    assert pair["interpretation"] == "not interpretable (effect size is nan, not a measured value)"
    assert pair["interpretation"] not in ("negligible", "small", "medium", "large")
    # The genuinely measured fields beside it are untouched.
    assert pair["mean_pred_diff"] == pytest.approx(30.0)
    assert pair["group1_size"] == 40 and pair["group2_size"] == 40


def test_regression_effect_sizes_still_band_a_measured_d():
    """OVER-CORRECTION CONTROL, asserted against the actual measured value.

    80 rows with real variance in both groups: Cohen's d for the predictions is
    0.224833 and the band is "small". Nothing warns.
    """
    rng = np.random.default_rng(1)
    y_true = rng.normal(50, 10, 80)
    y_pred = y_true + rng.normal(0, 2, 80)
    groups = np.array(["A"] * 40 + ["B"] * 40)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = compute_regression_effect_sizes(y_true, y_pred, groups)

    pair = out["A_vs_B"]
    assert pair["cohens_d_predictions"] == pytest.approx(0.224833, abs=1e-6)
    assert pair["interpretation"] == "small"


# ---------------------------------------------------------------------------
# DEFECT D. The tracker disclosure covered `metrics` and `group_stats` and not
# `metrics_with_ci` or `effect_sizes`; and `_logged_as_metric` rejected every
# numpy scalar, so a real measurement was dropped AND falsely disclosed as
# unmeasured. Overturned rows: log_fairness_to_mlflow, log_fairness_to_wandb.
# ---------------------------------------------------------------------------


def _report_whose_interval_failed():
    """The exact shape compute_metric_with_ci returns from "Bootstrap failed":
    a MEASURED point estimate and an interval that was never estimated."""
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.42},
        "metrics_with_ci": {
            "demographic_parity_difference": {
                "point_estimate": 0.42,
                "lower_bound": float("nan"),
                "upper_bound": float("nan"),
                "standard_error": float("nan"),
                "method": "stratified_bootstrap",
                "metadata": {"warning": "Bootstrap failed"},
            }
        },
        "effect_sizes": {"F_vs_M": {"cohens_d_positive_rate": float("nan")}},
        "group_stats": {"M": {"size": 150, "tpr": 0.78}, "F": {"size": 150, "tpr": 0.61}},
        "assessment": {"fairness_score": 0.5},
        "data_info": {"n_samples": 300, "n_groups": 2, "valid_groups": ["F", "M"]},
    }


def _measured_report():
    """Every metric, interval, effect size and group statistic measured."""
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.42},
        "metrics_with_ci": {
            "demographic_parity_difference": {
                "point_estimate": 0.42,
                "lower_bound": 0.31,
                "upper_bound": 0.53,
                "standard_error": 0.056,
                "method": "stratified_bootstrap_fold_debiased",
            }
        },
        "effect_sizes": {"F_vs_M": {"cohens_d_positive_rate": -0.81}},
        "group_stats": {"M": {"size": 150, "tpr": 0.78}, "F": {"size": 150, "tpr": 0.61}},
        "assessment": {"fairness_score": 0.5},
        "data_info": {"n_samples": 300, "n_groups": 2, "valid_groups": ["F", "M"]},
    }


def test_mlflow_logger_names_the_interval_it_could_not_log(fake_mlflow):
    """An unestimated interval is logged as nothing and is now NAMED.

    BEFORE (measured on the report above):
        logged  fairness.demographic_parity_difference.value = 0.42
        absent  ...ci_lower, ...ci_upper, ...std_error,
                fairness.effect_size.F_vs_M.cohens_d
        and     n_metrics_not_measured 0, n_group_stats_not_measured 0,
                with no param or tag naming the interval at all
    `_unmeasured_note` read report["metrics"] and report["group_stats"] only, so
    the two other series both trackers write kept the whole of the defect the
    metric series had just been fixed for.
    """
    logged = log_fairness_to_mlflow(_report_whose_interval_failed(), log_artifacts=False)
    params = fake_mlflow.params

    # The write is unchanged: an unmeasurable series is still logged as nothing.
    assert "fairness.demographic_parity_difference.value" in fake_mlflow.metrics
    assert "fairness.demographic_parity_difference.ci_lower" not in fake_mlflow.metrics
    assert "fairness.effect_size.F_vs_M.cohens_d" not in fake_mlflow.metrics
    # ...and it is named.
    assert params["fairness.n_intervals_not_measured"] == 3
    for suffix in ("ci_lower", "ci_upper", "std_error"):
        assert (
            f"demographic_parity_difference.{suffix}" in params["fairness.intervals_not_measured"]
        )
    assert params["fairness.n_effect_sizes_not_measured"] == 1
    assert params["fairness.effect_sizes_not_measured"] == "['F_vs_M.cohens_d']"
    # Reachable in process, for a caller that reads only the return value.
    assert logged["fairness.n_intervals_not_measured"] == 3


def test_wandb_logger_names_the_interval_it_could_not_log(fake_wandb):
    """The W&B path had the identical gap, on the surface a reader scans.

    BEFORE: run.summary held n_metrics_not_measured 0 and
    n_group_stats_not_measured 0 and nothing else, while
    fairness/demographic_parity_difference/ci_lower and /ci_upper were absent.
    """
    logged = log_fairness_to_wandb(_report_whose_interval_failed(), log_artifacts=False)
    summary = fake_wandb.run.summary

    assert "fairness/demographic_parity_difference/ci_lower" not in fake_wandb.logged
    assert summary["fairness/n_intervals_not_measured"] == 3
    assert "demographic_parity_difference/ci_lower" in summary["fairness/intervals_not_measured"]
    assert summary["fairness/n_effect_sizes_not_measured"] == 1
    assert summary["fairness/effect_sizes_not_measured"] == "['F_vs_M/cohens_d']"
    assert logged["fairness/n_intervals_not_measured"] == 3


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_logger_interval_disclosure_reads_zero_when_everything_was_measured(
    backend, fake_mlflow, fake_wandb
):
    """OVER-CORRECTION CONTROL on both trackers, with the values asserted.

    The fields are emitted on EVERY run, so on a fully measured report they read
    0 and "[]" rather than being absent, and every series is written.
    """
    report = _measured_report()
    if backend == "mlflow":
        log_fairness_to_mlflow(report, log_artifacts=False)
        params = fake_mlflow.params
        assert params["fairness.n_intervals_not_measured"] == 0
        assert params["fairness.intervals_not_measured"] == "[]"
        assert params["fairness.n_effect_sizes_not_measured"] == 0
        assert params["fairness.effect_sizes_not_measured"] == "[]"
        assert fake_mlflow.metrics["fairness.demographic_parity_difference.ci_lower"] == 0.31
        assert fake_mlflow.metrics["fairness.demographic_parity_difference.ci_upper"] == 0.53
        assert fake_mlflow.metrics["fairness.effect_size.F_vs_M.cohens_d"] == -0.81
    else:
        log_fairness_to_wandb(report, log_artifacts=False)
        summary = fake_wandb.run.summary
        assert summary["fairness/n_intervals_not_measured"] == 0
        assert summary["fairness/intervals_not_measured"] == "[]"
        assert summary["fairness/n_effect_sizes_not_measured"] == 0
        assert fake_wandb.logged["fairness/demographic_parity_difference/ci_lower"] == 0.31
        assert fake_wandb.logged["fairness/effect_size/F_vs_M/cohens_d"] == -0.81


def test_a_numpy_typed_measurement_is_logged_and_not_called_unmeasured(fake_mlflow):
    """A real measurement carried in a numpy scalar was dropped AND disclosed.

    BEFORE, with demographic_parity_difference = np.float32(0.42) and the group
    size = np.int64(150) (the types a numpy-backed report carries):
        neither was logged, and
        metrics_not_measured     = ['demographic_parity_difference']
        group_stats_not_measured = ['M.size']
    a REAL measurement reported as a could-not-check, which is the fabricated
    verdict running backwards. `isinstance(np.float32(0.42), (int, float))` is
    False; np.float64 is a float subclass, which is why real reports hid it.
    """
    report = _report_whose_interval_failed()
    report["metrics"] = {"demographic_parity_difference": np.float32(0.42)}
    report["group_stats"] = {"M": {"size": np.int64(150)}}
    report["metrics_with_ci"] = {}
    report["effect_sizes"] = {}

    log_fairness_to_mlflow(report, log_artifacts=False)

    assert fake_mlflow.metrics["fairness.demographic_parity_difference"] == pytest.approx(0.42)
    assert fake_mlflow.metrics["fairness.group.M.size"] == 150.0
    assert fake_mlflow.params["fairness.metrics_not_measured"] == "[]"
    assert fake_mlflow.params["fairness.n_metrics_not_measured"] == 0
    assert fake_mlflow.params["fairness.group_stats_not_measured"] == "[]"


# ---------------------------------------------------------------------------
# DEFECT E. permutation_test_equal_opportunity dropped a group that has no
# positive label and published the surviving comparison as the verdict.
# Overturned row: robustness.permutation_test_equal_opportunity.
# ---------------------------------------------------------------------------


def _three_groups_one_without_positives():
    rng = np.random.default_rng(2)
    y_true = np.concatenate([np.ones(100), np.ones(100), np.zeros(100)])
    y_pred = np.concatenate(
        [
            (rng.random(100) < 0.9).astype(int),
            (rng.random(100) < 0.3).astype(int),
            (rng.random(100) < 0.5).astype(int),
        ]
    )
    groups = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    return y_true, y_pred, groups


def test_equal_opportunity_permutation_names_the_group_it_omitted():
    """Three groups, one with no positive label: the omission is now on the result.

    BEFORE (measured on 300 rows in three groups of 100 where group c holds no
    positive label at all):
        {'observed_statistic': 0.600, 'p_value': 0.001996,
         'significant_at_05': True, 'n_permutations': 500, 'design_note': ''}
        warnings NONE, and PermutationTestResult had no field that could name c.
    A significant equal-opportunity verdict was published for a three-group audit
    one of whose groups was entirely outside it. The sibling
    test_equalized_odds_chi_square was fixed for this shape on the same day.
    """
    y_true, y_pred, groups = _three_groups_one_without_positives()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_equal_opportunity(
            y_true, y_pred, groups, n_permutations=500, random_state=1
        )

    assert result.groups_omitted == ("c",)
    assert result.to_dict()["groups_omitted"] == ["c"]
    named = [
        str(w.message)
        for w in caught
        if "outside this comparison" in str(w.message) and "c" in str(w.message)
    ]
    assert named, [str(w.message) for w in caught]
    assert "y_true == 1" in named[0]
    # The measurement is NOT withdrawn by its caveat: a real finding stands.
    assert result.observed_statistic == pytest.approx(0.6, abs=1e-9)
    assert result.p_value == pytest.approx(0.001996, abs=1e-5)
    assert result.significant_at_05 is True


def test_equal_opportunity_permutation_omits_nothing_on_a_complete_audit():
    """OVER-CORRECTION CONTROL. Two groups, both with positives, a planted gap.

    Measured: observed_statistic 0.7 over 100 rows per group, p 0.0005,
    significant_at_05 True, groups_omitted () and no omission warning.
    """
    rng = np.random.default_rng(3)
    y_true = np.ones(200)
    y_pred = np.concatenate(
        [(rng.random(100) < 0.9).astype(int), (rng.random(100) < 0.3).astype(int)]
    )
    groups = np.array(["a"] * 100 + ["b"] * 100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_equal_opportunity(
            y_true, y_pred, groups, n_permutations=2000, random_state=1
        )

    assert result.groups_omitted == ()
    assert result.observed_statistic == pytest.approx(0.7, abs=1e-9)
    assert result.significant_at_05 is True
    assert [str(w.message) for w in caught if "outside this comparison" in str(w.message)] == []


# ---------------------------------------------------------------------------
# DEFECT F. robust_fairness_comparison took a builtin max/min over per-group
# values that can be NaN, so a pair with one unmeasurable side answered 0.0,
# which its own docstring calls PERFECT PARITY, and which of the two possible
# answers you got depended on the alphabetical order of the labels.
# Overturned row: robustness.robust_fairness_comparison.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "unmeasurable"),
    [(("a", "z"), "z"), (("m", "b"), "b")],
)
def test_robust_fairness_comparison_refuses_a_pair_with_an_unmeasurable_side(labels, unmeasurable):
    """`max([3.0, nan])` is 3.0 and `min([3.0, nan])` is 3.0, so the gap collapsed.

    BEFORE (measured on 100 rows, group 'a' with a real error of 3.0 and group
    'z' holding no prediction at all):
        labels a, z -> {'standard_disparity': 0.0, 'robust_disparity': 0.0,
                        'n_groups_compared': 2}
        the same data with the unmeasurable group renamed so it sorts FIRST
                    -> {'standard_disparity': nan, 'robust_disparity': nan}
    The published disparity depended on the spelling of the labels and one of
    the two answers was 0.0. Both orderings now refuse identically, and the
    unmeasurable group is named. The repo bans this idiom by name in
    discovery._cramers_v ("np.maximum, NOT the builtin max").
    """
    n = 50
    first, second = labels
    y_true = np.full(2 * n, 10.0)
    y_pred = np.concatenate([np.full(n, 7.0), np.full(n, np.nan)])
    groups = np.array([first] * n + [second] * n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = robust_fairness_comparison(y_true, y_pred, groups)

    assert math.isnan(out["standard_disparity"])
    assert math.isnan(out["robust_disparity"])
    assert out["standard_disparity"] != 0.0
    assert out["conclusion_changed"] is None
    assert out["n_groups_compared"] == 1
    assert out["groups_not_measured"] == [unmeasurable]
    assert [w for w in caught if "no measurable error" in str(w.message)], [
        str(w.message) for w in caught
    ]


def test_robust_fairness_comparison_keeps_a_measured_disparity():
    """OVER-CORRECTION CONTROL, with the independently computed expectation.

    Group a's absolute errors are all 1.0 and group b's all 3.0, so the standard
    and robust disparities are both exactly 2.0, conclusion_changed is a measured
    False, two groups were compared and nothing warns.
    """
    n = 50
    y_true = np.zeros(2 * n)
    y_pred = np.concatenate([np.full(n, 1.0), np.full(n, 3.0)])
    groups = np.array(["a"] * n + ["b"] * n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = robust_fairness_comparison(y_true, y_pred, groups, metric="mae")

    assert out["standard_disparity"] == pytest.approx(2.0)
    assert out["robust_disparity"] == pytest.approx(2.0)
    assert out["conclusion_changed"] is False
    assert out["n_groups_compared"] == 2
    assert out["groups_not_measured"] == []
    assert not [str(w.message) for w in caught], [str(w.message) for w in caught]


def test_robust_fairness_comparison_keeps_the_gap_it_can_still_measure():
    """A third group that cannot be measured does not delete the real pair.

    Two measurable groups at 1.0 and 3.0 plus one group with no prediction at
    all: the 2.0 disparity is kept (it is a real lower bound on the attribute's
    gap) and the group outside it is named, rather than either being dropped.
    """
    n = 50
    y_true = np.zeros(3 * n)
    y_pred = np.concatenate([np.full(n, 1.0), np.full(n, 3.0), np.full(n, np.nan)])
    groups = np.array(["a"] * n + ["b"] * n + ["q"] * n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = robust_fairness_comparison(y_true, y_pred, groups, metric="mae")

    assert out["standard_disparity"] == pytest.approx(2.0)
    assert out["n_groups_compared"] == 2
    assert out["groups_not_measured"] == ["q"]
    assert [w for w in caught if "OUTSIDE both disparities" in str(w.message)]


# ---------------------------------------------------------------------------
# DEFECT G. rank_fairness_issues' summary max_disparity was a maximum over the
# VIOLATIONS with a 0.0 default, so a fully measured run with nothing over
# threshold published the number a perfectly fair run gets.
# Overturned row: discovery.rank_fairness_issues.
# ---------------------------------------------------------------------------


def _measured_but_passing_run():
    rng = np.random.default_rng(12)
    gender = np.array(["M"] * 1000 + ["F"] * 1000)
    y_pred = np.concatenate(
        [(rng.random(1000) < 0.500).astype(int), (rng.random(1000) < 0.495).astype(int)]
    )
    y_true = rng.integers(0, 2, 2000)
    df = pd.DataFrame({"gender": gender, "salary": rng.normal(50000, 5000, 2000)})
    return df, y_pred, y_true


def test_rank_fairness_issues_publishes_the_disparity_it_measured():
    """A measured gap is no longer summarised as the perfectly-fair 0.0.

    BEFORE (measured on 2000 rows with selection rates 0.503 and 0.478, a
    demographic-parity gap of 0.025 and nothing over any threshold):
        {'n_attributes_checked': 1, 'n_violations': 0, 'max_disparity': 0.0,
         'n_not_assessable': 0}
        recommendations ['No significant fairness violations detected ...']
    scan_fairness_violations measures each disparity and keeps it only when it
    exceeds the threshold, so every passing gap was computed and discarded and
    the summary published the number the code's own comment reserves for a
    perfect run. AFTER: the largest of the four disparities this run measured,
    0.034365109260870885 (the equal-opportunity gap; the demographic-parity one
    is the 0.025 above).
    """
    df, y_pred, y_true = _measured_but_passing_run()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rank_fairness_issues(
            df, y_pred, y_true, protected_columns=["gender"], auto_detect=False
        )

    summary = out["summary"]
    assert summary["n_violations"] == 0
    assert summary["n_not_assessable"] == 0
    assert summary["n_disparities_measured"] == 4
    assert summary["max_disparity"] == pytest.approx(0.034365109260870885, abs=1e-12)
    assert summary["max_disparity"] != 0.0
    # The measured demographic-parity gap is one of the values it is taken over.
    measured_gap = abs(y_pred[:1000].mean() - y_pred[1000:].mean())
    assert measured_gap == pytest.approx(0.025, abs=1e-9)
    assert summary["max_disparity"] >= measured_gap


def test_rank_fairness_issues_still_publishes_a_measured_zero():
    """OVER-CORRECTION CONTROL. A genuinely identical run measures 0.0, not NaN.

    Both groups get the same 500 of 1000 selected and y_true equals y_pred, so
    every one of the four disparities is exactly 0.0: a MEASUREMENT of perfect
    parity, which must not be turned into a could-not-check by a fix aimed at the
    fabricated 0.0.
    """
    gender = np.array(["M"] * 1000 + ["F"] * 1000)
    per_group = np.r_[np.ones(500, int), np.zeros(500, int)]
    y_pred = np.concatenate([per_group, per_group])
    y_true = y_pred.copy()
    df = pd.DataFrame({"gender": gender, "salary": np.linspace(1.0, 2.0, 2000)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rank_fairness_issues(
            df, y_pred, y_true, protected_columns=["gender"], auto_detect=False
        )

    summary = out["summary"]
    assert summary["max_disparity"] == 0.0
    assert not math.isnan(summary["max_disparity"])
    assert summary["n_disparities_measured"] == 4
    assert summary["n_not_assessable"] == 0
    assert out["recommendations"] == [
        "No significant fairness violations detected. Continue monitoring."
    ]


def test_rank_fairness_issues_still_refuses_when_nothing_was_assessable():
    """OVER-CORRECTION CONTROL in the other direction: the NaN state survives.

    280 rows in one group and 20 in another below min_group_size=30, so the
    attribute is not scanned at all: max_disparity stays NaN (could not check)
    and the all-clear is still withheld.
    """
    race = np.array(["w"] * 280 + ["x"] * 20)
    y_pred = np.concatenate([np.ones(280, int), np.zeros(20, int)])
    y_true = np.random.default_rng(0).integers(0, 2, 300)
    df = pd.DataFrame({"race": race})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rank_fairness_issues(
            df, y_pred, y_true, protected_columns=["race"], auto_detect=False
        )

    summary = out["summary"]
    assert math.isnan(summary["max_disparity"])
    assert summary["n_not_assessable"] >= 1
    assert "COULD NOT CHECK" in " ".join(out["recommendations"])


# ---------------------------------------------------------------------------
# DEFECT H. classify_column_roles dropped a declared protected column that is
# not in the frame, silently, and fell back to auto-detection.
# Overturned row: discovery.classify_column_roles.
# ---------------------------------------------------------------------------


def _frame_without_ethnicity():
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "gender": np.array(["M"] * 200 + ["F"] * 200),
            "salary": rng.normal(50000, 5000, 400),
        }
    )


def test_classify_column_roles_names_a_declared_column_absent_from_the_frame():
    """A misspelt or renamed protected column is named instead of discarded.

    BEFORE (measured on 400 rows holding only gender and salary):
        classify_column_roles(df, declared_protected=['ethnicity'])
          -> n_not_assessable 0, not_assessable [], mismatches [], refuse False,
             protected ['gender'] ("auto-detected protected (demographic)"),
             warnings NONE
    The declaration was dropped by a list comprehension, `has_declared`
    collapsed to False and the function fell back to auto-detection, so an
    attribute nobody had looked at bought a clean bill of health. This is the
    same shape the file records as a defect in scan_fairness_violations.
    """
    df = _frame_without_ethnicity()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roles = classify_column_roles(df, declared_protected=["ethnicity"])

    assert roles["n_not_assessable"] == 1
    assert "ethnicity" in roles["not_assessable"][0]
    assert "no such column" in roles["not_assessable"][0]
    # The mismatch channel is what the compliance report turns into a finding.
    assert [m["column"] for m in roles["mismatches"]] == ["ethnicity"]
    assert roles["mismatches"][0]["inferred"] == "absent"
    assert "NOT assessed" in roles["mismatches"][0]["detail"]
    assert [str(w.message) for w in caught if "ethnicity" in str(w.message)]


def test_classify_column_roles_is_silent_for_a_declaration_it_can_honour():
    """OVER-CORRECTION CONTROL. A declared column that EXISTS is screened quietly.

    The same frame, declaring `gender`, which is there: nothing is
    not-assessable, there is no mismatch, gender is classified protected and no
    warning is raised.
    """
    df = _frame_without_ethnicity()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roles = classify_column_roles(df, declared_protected=["gender"])

    assert roles["n_not_assessable"] == 0
    assert roles["not_assessable"] == []
    assert roles["mismatches"] == []
    assert roles["protected"] == ["gender"]
    assert roles["refuse"] is False
    assert [str(w.message) for w in caught if "classify_column_roles" in str(w.message)] == []
