"""BGL-4 audit batch A-evaluation-3: the overturns, demonstrated by execution.

EVERY TEST IN THIS FILE WAS WRITTEN AGAINST A DEFECT THAT WAS LIVE, and each one
has now been INVERTED, in place, to assert the corrected behaviour. The subject
and the docstring of each test are unchanged: the measured "before" quoted in
them is the record of what the defect was, which is why they are inverted rather
than deleted. All eight defects were closed in BGL-5 (2026-09-27); the fixes are
pinned again, with their over-correction controls, in
tests/test_bgl5_evaluation_3.py.

Audited 2026-09-27 against the BGL-3 grades that call these units PROVEN. The
grade rows named a test and a sabotage for each; both hold. What follows is the
SECOND input class the named test never tried.
"""

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
from vfairness.evaluation.vfairness_metrics.robustness import (
    permutation_test_equal_opportunity,
    robust_fairness_comparison,
)

# Recording stubs, same shape as the BGL-3 fixtures so the comparison is direct.


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


# DEFECT A. explainer._could_not_check_reason tests `isinstance(value, bool)`,
# and numpy's boolean is NOT a Python bool. Grade row:
# explainer.FairExplAIner.explain_metric, PROVEN, sabotage S3.


def test_explain_metric_still_grades_a_numpy_boolean():
    """A np.bool_ flag is graded as a fairness measurement, in both directions.

    `isinstance(np.True_, bool)` is False and `np.isnan(np.True_)` is False, so
    the guard added for `False` is bypassed by the boolean type that every numpy
    or pandas comparison produces. The same class refuses np.bool_ by name in
    `_finite_or_none` (the interval and effect-size surfaces), so one class still
    disagrees with itself about one input.
    """
    ex = FairExplAIner()

    assert isinstance(np.True_, bool) is False

    # Python bool: refused, as the graded fix said.
    assert ex.explain_metric("demographic_parity_difference", False).severity == "could_not_check"
    assert ex.explain_metric("demographic_parity_difference", True).severity == "could_not_check"

    # INVERTED (BGL-5): the numpy bool is refused too. `_could_not_check_reason`
    # now tests `isinstance(value, (bool, np.bool_))`, so the all-clear that
    # np.False_ used to receive ("Excellent! ... near-perfect fairness") is gone.
    clear = ex.explain_metric("demographic_parity_difference", np.False_)
    assert clear.severity == "could_not_check"
    assert "Excellent!" not in clear.evaluation
    alarm = ex.explain_metric("demographic_parity_difference", np.True_)
    assert alarm.severity == "could_not_check"


def test_explain_report_grades_a_numpy_boolean_card_and_warns_about_nothing(recwarn):
    """The COULD NOT CHECK card and the "NOT GRADED" warning both miss np.bool_.

    The graded fix routes a non-number to a could-not-check card and warns. Its
    trigger is `_could_not_check_reason(...) == "unmeasurable" and not
    isinstance(value, (int, float))`, and a np.bool_ fails the first clause, so
    the card is graded and no warning is raised. Reached through the public
    convenience wrapper, which is the entry point the docs show.
    """
    report = {
        "metrics": {"demographic_parity_difference": np.False_},
        "group_stats": {},
        "thresholds_used": {},
    }
    out = explain_fairness_report(report)

    # INVERTED (BGL-5): the card is a could-not-check, the summary carries the
    # NOT GRADED line and the warning names the value. Before the fix this card
    # read severity "info" with "Excellent! ..." and recwarn was empty.
    card = out["metrics"]["demographic_parity_difference"]
    assert card["severity"] == "could_not_check"
    assert "Excellent!" not in card["evaluation"]
    assert [w for w in recwarn.list if "not numbers" in str(w.message)]
    assert "NOT GRADED" in out["summary"]


# DEFECT B. The infinity carve-out in `_effect_size_unmeasured_reason` is
# justified for risk_ratio and odds_ratio and applied to every effect type.
# Grade row: explainer.FairExplAIner.explain_effect_size, PROVEN, sabotage S15b.


def test_explain_effect_size_calls_an_infinite_cohens_d_large():
    """An infinite Cohen's d is a division by an empty denominator, not an effect.

    The carve-out's own comment says `is_measured` rejects inf "because most
    producers reach it by dividing by an empty denominator" and exempts the two
    ratios, which reach inf from a measured table. It does not look at
    `effect_type`, so an infinite standardised mean difference, which can only
    come from a zero pooled standard deviation, is graded as the largest
    magnitude the ladder has.
    """
    ex = FairExplAIner()
    # INVERTED (BGL-5): the carve-out is scoped to the two ratios, so an infinite
    # Cohen's d is a could-not-check. Before the fix both directions returned
    # severity "high" with "Large effect size - the difference is very
    # substantial."
    for value in (float("inf"), float("-inf")):
        card = ex.explain_effect_size("cohens_d", value, "A", "B")
        assert card.severity == "could_not_check"
        assert "Large effect size" not in card.evaluation
    # The NaN it always refused, for contrast.
    assert ex.explain_effect_size("cohens_d", float("nan"), "A", "B").severity == "could_not_check"
    # ...and the carve-out still holds where it was argued for.
    assert ex.explain_effect_size("risk_ratio", float("inf"), "A", "B").severity == "critical"


# DEFECT C. regression.compute_regression_effect_sizes publishes an
# `interpretation` graded off a NaN Cohen's d. Grade row: PROVEN, sabotage
# "deleted the warnings.warn block".


def test_regression_effect_sizes_interpret_an_unmeasured_d_as_large():
    """`abs(nan) < 0.2` is False all the way down the ladder, so NaN reads "large".

    Two groups whose predictions are each CONSTANT and differ in mean: the
    library's own `cohens_d` returns NaN there, deliberately and with a warning
    ("the standardising denominator is zero"). The local `interpret()` in this
    function has no NaN branch, so the pair is published with
    `cohens_d_predictions: nan` beside `interpretation: 'large'`, which is the
    strongest verdict that field can carry and was measured by nothing.
    """
    n = 40
    y_true = np.concatenate([np.full(n, 100.0), np.full(n, 100.0)])
    y_pred = np.concatenate([np.full(n, 130.0), np.full(n, 100.0)])
    groups = np.array(["A"] * n + ["B"] * n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_regression_effect_sizes(y_true, y_pred, groups)

    # INVERTED (BGL-5): the ladder grades only a finite value, so the pair is
    # published as not interpretable. Before the fix `interpretation` read
    # "large" beside `cohens_d_predictions: nan`.
    pair = out["A_vs_B"]
    assert np.isnan(pair["cohens_d_predictions"])
    assert pair["interpretation"] == "not interpretable (effect size is nan, not a measured value)"
    assert pair["interpretation"] != "large"


# DEFECT D. The tracker disclosure covers `metrics` and `group_stats` and not
# `metrics_with_ci` or `effect_sizes`. Grade rows: integrations.
# log_fairness_to_mlflow / log_fairness_to_wandb, PROVEN, sabotages S8/S9/S10.


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


def test_mlflow_logger_never_names_the_interval_it_could_not_log(fake_mlflow):
    """An unestimated interval is logged as nothing and named nowhere.

    ci_lower, ci_upper, std_error and the effect size are all absent, exactly as
    the fixed metric series used to be, while every disclosure field on the run
    reads 0 / "[]". `_unmeasured_note` reads report["metrics"] and
    report["group_stats"] only.
    """
    log_fairness_to_mlflow(_report_whose_interval_failed(), log_artifacts=False)

    assert "fairness.demographic_parity_difference.value" in fake_mlflow.metrics
    assert "fairness.demographic_parity_difference.ci_lower" not in fake_mlflow.metrics
    assert "fairness.demographic_parity_difference.ci_upper" not in fake_mlflow.metrics
    assert "fairness.effect_size.F_vs_M.cohens_d" not in fake_mlflow.metrics
    # INVERTED (BGL-5): the write is unchanged (an unmeasurable series is still
    # logged as nothing) and `_unmeasured_note` now names it. Before the fix
    # these four params did not exist and no param mentioned the interval.
    assert fake_mlflow.params["fairness.n_intervals_not_measured"] == 3
    assert (
        "demographic_parity_difference.ci_lower"
        in (fake_mlflow.params["fairness.intervals_not_measured"])
    )
    assert fake_mlflow.params["fairness.n_effect_sizes_not_measured"] == 1
    assert fake_mlflow.params["fairness.effect_sizes_not_measured"] == "['F_vs_M.cohens_d']"


def test_wandb_logger_never_names_the_interval_it_could_not_log(fake_wandb):
    """The W&B path has the identical gap, on the surface a reader scans."""
    log_fairness_to_wandb(_report_whose_interval_failed(), log_artifacts=False)

    # INVERTED (BGL-5): the interval is still logged as nothing and the run
    # summary now names it. Before the fix the summary held only the metric and
    # group-stat fields, both reading 0 and "[]".
    assert "fairness/demographic_parity_difference/ci_lower" not in fake_wandb.logged
    assert fake_wandb.run.summary["fairness/n_intervals_not_measured"] == 3
    assert (
        "demographic_parity_difference/ci_lower"
        in (fake_wandb.run.summary["fairness/intervals_not_measured"])
    )
    assert fake_wandb.run.summary["fairness/n_effect_sizes_not_measured"] == 1


# DEFECT E. permutation_test_equal_opportunity drops a group that has no
# positive label and reports the surviving comparison as the verdict. Grade row:
# PROVEN, sabotage S15. Its sibling `test_equalized_odds_chi_square` was fixed
# for exactly this on the same day, in the same file, with `groups_omitted`.


def test_equal_opportunity_permutation_omits_a_group_in_silence():
    """Three groups, one with no positive label: a significant TPR gap over two.

    `eo_diff` collects a TPR only for groups that have a positive label and
    refuses only when FEWER THAN TWO remain. With two of three remaining it
    returns a real max-min gap, and PermutationTestResult has no field that
    could name the third group. No warning is raised.
    """
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_equal_opportunity(
            y_true, y_pred, groups, n_permutations=500, random_state=1
        )

    # INVERTED (BGL-5): the verdict is kept, because it is real for the groups it
    # covers, and the omitted group is named on the result and in a warning.
    # Before the fix PermutationTestResult had no such field and nothing warned.
    assert result.significant_at_05 is True
    assert result.observed_statistic > 0.5
    assert result.groups_omitted == ("c",)
    assert [
        str(w.message)
        for w in caught
        if "c" in str(w.message) and "outside this comparison" in str(w.message)
    ]


# DEFECT F. robust_fairness_comparison uses the BUILTIN max/min over values that
# can be NaN, so 0.0 (PERFECT PARITY) comes back for a pair in which one side
# was never measured, and which of the two answers you get depends on the
# alphabetical order of the group labels. Grade row: PROVEN, sabotages S1/S20.


def test_robust_fairness_comparison_calls_an_unmeasured_pair_perfect_parity():
    """`max([3.0, nan])` is 3.0 and `min([3.0, nan])` is 3.0, so the gap is 0.0.

    One group's errors are all NaN (a missing prediction column), the other's are
    a measured 3.0. The docstring of this very function says "0.0 IS PERFECT
    PARITY AND IT WAS THE ANSWER FOR NO COMPARISON AT ALL"; it is the answer
    again here. The repo bans this idiom by name in discovery._cramers_v ("np.
    maximum, NOT the builtin max: max(0.0, nan) is 0.0").
    """
    n = 50
    y_true = np.full(2 * n, 10.0)
    # "a" is measurable, "z" holds no prediction at all, and "z" sorts second.
    y_pred = np.concatenate([np.full(n, 7.0), np.full(n, np.nan)])
    groups = np.array(["a"] * n + ["z"] * n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = robust_fairness_comparison(y_true, y_pred, groups)

    # INVERTED (BGL-5): the unmeasurable group is excluded by name instead of
    # being skipped by a comparison against NaN, so one measurable group is not a
    # pair and both disparities refuse. Before the fix this ordering answered
    # 0.0 / 0.0 with n_groups_compared 2.
    assert out["n_groups_compared"] == 1
    assert out["groups_not_measured"] == ["z"]
    assert np.isnan(out["group_metrics"]["z"].standard_value)
    assert np.isnan(out["standard_disparity"])
    assert np.isnan(out["robust_disparity"])

    # The label spelling no longer changes the answer: sorting the unmeasurable
    # group FIRST gives the identical refusal, where before the two orderings
    # disagreed (nan against 0.0) on the same data.
    groups_swapped = np.array(["m"] * n + ["b"] * n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swapped = robust_fairness_comparison(y_true, y_pred, groups_swapped)
    assert np.isnan(swapped["standard_disparity"])
    assert swapped["n_groups_compared"] == out["n_groups_compared"]


# DEFECT G. rank_fairness_issues' summary max_disparity is a maximum over the
# VIOLATIONS, defaulting to 0.0, so a measured run with nothing over threshold
# publishes the number a perfectly fair run gets. Grade row: PROVEN, sabotage S12.


def test_rank_fairness_issues_publishes_zero_max_disparity_for_a_measured_gap():
    """A measured 0.025 selection-rate gap is summarised as max_disparity 0.0.

    scan_fairness_violations computes each disparity and keeps only the ones over
    threshold, so with no breach the maximum is taken over an empty sequence and
    the default 0.0 applies (the NaN default is conditioned on there being
    something in not_assessable). The code's own comment says 0.0 "is the value a
    perfectly fair run gets"; this run is not perfectly fair and nothing here was
    unmeasurable.
    """
    rng = np.random.default_rng(12)
    gender = np.array(["M"] * 1000 + ["F"] * 1000)
    y_pred = np.concatenate(
        [(rng.random(1000) < 0.500).astype(int), (rng.random(1000) < 0.495).astype(int)]
    )
    y_true = rng.integers(0, 2, 2000)
    df = pd.DataFrame({"gender": gender, "salary": rng.normal(50000, 5000, 2000)})

    measured_gap = abs(y_pred[:1000].mean() - y_pred[1000:].mean())
    assert measured_gap > 0.02

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rank_fairness_issues(
            df, y_pred, y_true, protected_columns=["gender"], auto_detect=False
        )

    # INVERTED (BGL-5): max_disparity is the largest disparity MEASURED when
    # nothing was unassessable, so this run publishes 0.0343651... instead of the
    # 0.0 the code's own comment reserves for a perfectly fair run.
    summary = out["summary"]
    assert summary["n_violations"] == 0
    assert summary["n_not_assessable"] == 0
    assert summary["max_disparity"] != 0.0
    assert summary["max_disparity"] == pytest.approx(0.034365109260870885, abs=1e-12)
    assert summary["max_disparity"] >= measured_gap


# DEFECT H. classify_column_roles skips a declared protected target that is not
# in the frame with a bare `continue`, the same silent skip the repo fixed in
# scan_fairness_violations ten days earlier. Grade row: PROVEN, sabotages S1/S2/S20.


def test_classify_column_roles_drops_a_declared_column_absent_from_the_frame():
    """A misspelt or renamed protected column is never screened and never named.

    `declared_protected=['ethnicity']` on a frame without it: the proxy loop's
    `if prot not in df.columns: continue` is silent, n_not_assessable is 0,
    not_assessable is empty, mismatches is empty, refuse is False and no warning
    is raised, while the function quietly falls back to auto-detection and
    reports `gender` as protected instead. Compare the comment at
    discovery.py:1909, for the identical pattern: "A misspelt or renamed column
    bought a clean bill of health for an attribute nobody had looked at."
    """
    rng = np.random.default_rng(12)
    df = pd.DataFrame(
        {
            "gender": np.array(["M"] * 200 + ["F"] * 200),
            "salary": rng.normal(50000, 5000, 400),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roles = classify_column_roles(df, declared_protected=["ethnicity"])

    # INVERTED (BGL-5): the declaration is recorded instead of discarded. Before
    # the fix all of these were empty, refuse was False, and `ethnicity` appeared
    # nowhere in the result. `refuse` is deliberately STILL False: it aborts a
    # whole Pulse run before any battery, so one misspelt name among several
    # correct ones must not destroy an otherwise measurable analysis.
    assert roles["n_not_assessable"] == 1
    assert "ethnicity" in roles["not_assessable"][0]
    assert [m["column"] for m in roles["mismatches"]] == ["ethnicity"]
    assert roles["refuse"] is False
    assert "ethnicity" in repr(roles)
    assert [str(w.message) for w in caught if "ethnicity" in str(w.message)]
