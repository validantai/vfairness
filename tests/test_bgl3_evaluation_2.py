"""BGL3 batch evaluation-2: does each unit refuse honestly when nothing is measurable?

Fourteen units across discovery.py, explainer.py and integrations.py were probed
by EXECUTION on inputs where the quantity they report genuinely does not exist,
and again on measurable input so that a unit which refuses everything cannot be
mistaken for one that refuses honestly.

Six defects were proved and fixed; the rest already refused correctly and are
pinned here so they cannot regress. Every docstring below states what was
measured BEFORE the fix, in the numbers the probe printed.
"""

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    ProxyScanIncompleteWarning,
    association_strength,
    classify_column_roles,
    identify_proxy_features,
    rank_fairness_issues,
)
from vfairness.evaluation.vfairness_metrics.explainer import (
    FairExplAIner,
    explain_fairness_report,
)
from vfairness.evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError,
    assert_fairness,
    auto_log_fairness,
    create_fairness_callback,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report

# Fixtures and recording stubs


class _FakeMlflow(types.ModuleType):
    """Records what vfairness sends, so no real tracking backend is needed."""

    def __init__(self):
        super().__init__("mlflow")
        self.metrics = {}
        self.params = {}
        self.tags = {}
        self.artifacts = []

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        assert isinstance(value, float), f"log_metric({key!r}) got {value!r}"
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, name):
        self.artifacts.append(name)


class _FakeArtifact:
    def __init__(self, name):
        self.name = name
        self.files = []

    def add_file(self, path, name):
        self.files.append(name)


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

    def Artifact(self, name, type):  # noqa: N802, this mirrors the W&B SDK's class name
        # auto_log_fairness leaves log_artifacts at its default True, so the stub
        # has to accept the artifact round trip or the decorator path dies inside
        # the tracker and every assertion after it becomes vacuous.
        return _FakeArtifact(name)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact.name)


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


def _biased_run(n=400, seed=5):
    """A blatant disparity: men approved ~85 percent, women ~20 percent.

    demographic_parity_difference is 0.695 here, so any gate that does not refuse
    this run is failing open rather than measuring something benign.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n)
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(gender == "M", rng.random(n) < 0.85, rng.random(n) < 0.2).astype(int)
    return y_true, y_pred, gender


def _fair_run(n=400, seed=7):
    """Predictions drawn independently of the group, so every metric is small."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n)
    y_true = rng.integers(0, 2, n)
    y_pred = (rng.random(n) < 0.5).astype(int)
    return y_true, y_pred, gender


def _mixed_report():
    """A REAL report in which some metrics are measured and others cannot be.

    Group F receives no positive LABELS, so its TPR is 0/0 and the TPR-based and
    AUROC-based metrics are NaN, while demographic parity is measured normally.
    Produced by the library's own public entry point, not hand-built.
    """
    rng = np.random.default_rng(11)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    y_true = np.where(gender == "F", 0, rng.integers(0, 2, n))
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
    y_prob = np.where(y_pred == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, gender, y_prob=y_prob)


def _measurable_report():
    """A report in which every metric and every per-group statistic is measured."""
    rng = np.random.default_rng(12)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, gender)


# DEFECT 1: classify_column_roles published no could-not-check channel at all


def test_classify_column_roles_names_what_the_proxy_screen_could_not_read():
    """The typology gate dropped `identify_proxy_features(...).not_assessable`.

    Measured 2026-09-27 on 200 rows whose DECLARED protected attribute `gender`
    holds ONE level, so Cramer's V and the correlation ratio are both 0/0 and no
    column can be screened for proxy risk at all:

        identify_proxy_features warned "2 of 2 candidate columns could NOT be
        assessed against 'gender'"
        classify_column_roles returned proxy_candidates ['home_region'],
        refuse False, refuse_reason '', and its dict held NO field naming the
        gap, so under suppressed warnings a screen that measured nothing was
        indistinguishable from a clean one.

    `identify_proxy_features` carries `not_assessable` on its result precisely so
    the incompleteness travels with it, and `rank_fairness_issues` consumes it.
    This was the second consumer and it dropped it.
    """
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame(
        {
            "gender": ["F"] * n,
            "home_region": rng.choice(["north", "south"], n),
            "sales": rng.normal(100, 20, n),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = classify_column_roles(df, declared_protected=["gender"])

    assert result["n_not_assessable"] == 2, result["not_assessable"]
    named = " ".join(result["not_assessable"])
    assert "home_region" in named and "sales" in named
    assert "proxy scan" in named
    # The one proxy candidate came from a NAME hint, with no correlation behind
    # it, which is exactly why the empty correlation screen had to be disclosed.
    roles = {r["column"]: r for r in result["roles"]}
    assert roles["home_region"]["evidence"].get("correlation") is None


def test_classify_column_roles_says_when_the_correlation_screen_never_ran():
    """With no protected target the correlation screen does not run at all.

    Before the fix that produced the same output as a screen that ran and found
    nothing: proxy_candidates [] (or name hints only) with no field saying the
    correlation pass had been skipped entirely.
    """
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {"tenure_months": rng.integers(1, 90, n), "sales_volume": rng.normal(100, 20, n)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = classify_column_roles(df)

    assert result["n_not_assessable"] >= 1
    assert any("not run at all" in entry for entry in result["not_assessable"])


def test_classify_column_roles_reports_nothing_unassessable_on_a_readable_frame():
    """Over-correction control: the field must be EMPTY when everything was read.

    A gate that always says "could not check" is as useless as one that never
    does. Here the ZIP column is a perfect proxy for gender and is found by
    correlation (association 0.99999), so nothing is unassessable.
    """
    rng = np.random.default_rng(4)
    n = 200
    gender = rng.choice(["F", "M"], n)
    df = pd.DataFrame(
        {
            "gender": gender,
            "zip": np.where(gender == "F", "90210", "10001"),
            "sales": rng.normal(100, 20, n),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = classify_column_roles(df, declared_protected=["gender"])

    assert result["not_assessable"] == []
    assert result["n_not_assessable"] == 0
    assert "zip" in result["proxy_candidates"]
    roles = {r["column"]: r for r in result["roles"]}
    assert roles["zip"]["evidence"].get("correlation") is not None


# DEFECT 2: explain_metric graded a boolean as if it were a measurement


@pytest.mark.parametrize("flag", [False, True])
def test_explain_metric_refuses_a_boolean_instead_of_grading_it(flag):
    """A bool is the one non-number np.isnan accepts, and it was graded.

    Measured 2026-09-27 at the public entry, the second line being the dangerous
    one because it is the strongest all-clear the metric can give:

        explain_metric("demographic_parity_difference", True)
          -> severity 'critical', "Critical. The difference of 1.0000 indicates
             severe disparity ..."
        explain_metric("demographic_parity_difference", False)
          -> severity 'info', "Excellent! The difference of 0.0000 indicates
             near-perfect fairness across groups.", with the recommendation
             written for a PASSING metric beside it.

    The same class already refuses a bool by name in `_finite_or_none` ("a bool
    is not a measurement (True would coerce to 1.0)"), which decides this for the
    interval and effect-size surfaces, so one class disagreed with itself about
    one input.
    """
    explanation = FairExplAIner().explain_metric("demographic_parity_difference", flag)

    assert explanation.severity == "could_not_check"
    assert "COULD NOT CHECK" in explanation.evaluation
    assert "Excellent" not in explanation.evaluation
    assert "NOT MEASURED" in explanation.recommendation


def test_explain_metric_still_grades_real_numbers():
    """Over-correction control for the bool refusal: numbers are still graded."""
    explainer = FairExplAIner()
    assert explainer.explain_metric("demographic_parity_difference", 0.01).severity == "info"
    assert explainer.explain_metric("demographic_parity_difference", 0.42).severity == "critical"
    assert explainer.explain_metric("demographic_parity_difference", 0.0).severity == "info"


# DEFECT 3: explain_report dropped a metric it could not read


def test_explain_report_keeps_a_could_not_check_card_for_an_unreadable_metric():
    """A non-numeric metric value vanished from the explanation set entirely.

    Measured 2026-09-27:

        explain_report({"metrics": {"demographic_parity_difference": None,
                                    "equalized_odds_difference": 0.42}, ...})
          -> explanations["metrics"] held ONE key, 'equalized_odds_difference'.
             The None metric had no card, no mention in the summary and no
             warning, so "never measured" and "does not exist" were one output.

    A None arrives whenever a report has been through a strict JSON encoder,
    which has no NaN and writes null.
    """
    report = {
        "metrics": {"demographic_parity_difference": None, "equalized_odds_difference": 0.42},
        "assessment": {
            "fairness_score": 0.5,
            "passed_metrics": [],
            "failed_metrics": [{"metric": "equalized_odds_difference"}],
            "assessable": True,
        },
    }
    with pytest.warns(UserWarning, match="not numbers"):
        explanations = explain_fairness_report(report)

    cards = explanations["metrics"]
    assert set(cards) == {"demographic_parity_difference", "equalized_odds_difference"}
    assert cards["demographic_parity_difference"]["severity"] == "could_not_check"
    assert cards["equalized_odds_difference"]["severity"] == "critical"
    # The summary counts the ungraded metric instead of losing it.
    assert "NOT GRADED: 1 metric(s)" in explanations["summary"]


def test_explain_report_is_silent_when_every_metric_is_readable():
    """Over-correction control: no warning and no could-not-check on real data."""
    report = _measurable_report()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        explanations = explain_fairness_report(report, include_statistical=False)

    severities = {name: card["severity"] for name, card in explanations["metrics"].items()}
    assert severities, "the healthy report produced no cards at all"
    assert "could_not_check" not in severities.values()


# DEFECT 4: assert_fairness certified a run it never examined


def test_assert_fairness_refuses_an_empty_metric_list():
    """Zero checks read as a pass in the function whose job is to stop a build.

    Measured 2026-09-27 on 400 rows where men are approved ~85 percent of the
    time and women ~20 percent (demographic_parity_difference 0.695):

        assert_fairness(y_true, y_pred, gender, metrics=[]) -> {} and NO raise.

    The loop cannot fail on a list it never enters. This is the sibling of the
    C-05 hole already fixed in operations/cicd/gate.py, where
    `evaluate(metrics=[])` returned approved=True with "APPROVED - All fairness
    requirements met".
    """
    y_true, y_pred, gender = _biased_run()
    with pytest.raises(ValueError, match="ZERO checks"):
        assert_fairness(y_true, y_pred, gender, metrics=[])


def test_assert_fairness_still_measures_and_still_passes_a_fair_run():
    """Over-correction control, both directions.

    The unfair run must FAIL on a measured breach (not on a could-not-check), and
    a fair run must PASS, or the refusal above would be indistinguishable from a
    gate that refuses everything.
    """
    y_true, y_pred, gender = _biased_run()
    with pytest.raises(FairnessAssertionError) as excinfo:
        assert_fairness(y_true, y_pred, gender, metrics=["demographic_parity_difference"])
    failure = excinfo.value.failed_metrics["demographic_parity_difference"]
    assert failure["not_measurable"] is False
    assert failure["value"] > 0.5

    y_true, y_pred, gender = _fair_run()
    values = assert_fairness(
        y_true,
        y_pred,
        gender,
        metrics=["demographic_parity_difference"],
        thresholds={"demographic_parity_difference": 0.2},
    )
    assert abs(values["demographic_parity_difference"]) < 0.2


# DEFECT 5: create_fairness_callback presented a gate that could never fire


@pytest.mark.parametrize("thresholds", [None, {}])
def test_create_fairness_callback_refuses_a_gate_with_no_thresholds(thresholds):
    """Every refusal sits behind `if fail_on_violation and thresholds:`.

    Measured 2026-09-27 on the same 400-row run:

        create_fairness_callback("gender", fail_on_violation=True)(...)
          -> returned demographic_parity_difference 0.6951 and did NOT raise,
             so training continued through a maximal violation.

    A falsy `thresholds` switched off the threshold loop AND the size-gate
    refusal at its end, leaving a callback that reads as a gate and can never
    stop anything.
    """
    with pytest.raises(ValueError, match="needs thresholds"):
        create_fairness_callback("gender", thresholds=thresholds, fail_on_violation=True)


def test_create_fairness_callback_refuses_an_empty_metric_list():
    """metrics=[] narrowed the results to {}, which reads as "nothing to report".

    Measured 2026-09-27: create_fairness_callback("gender", metrics=[]) returned
    a callback that answered {} for the 0.695 run, every epoch.
    """
    with pytest.raises(ValueError, match="monitor nothing"):
        create_fairness_callback("gender", metrics=[])


def test_create_fairness_callback_still_gates_and_still_permits():
    """Over-correction control: the configured gate refuses the unfair run only."""
    y_true, y_pred, gender = _biased_run()
    callback = create_fairness_callback(
        "gender",
        thresholds={"demographic_parity_difference": 0.1},
        fail_on_violation=True,
    )
    with pytest.raises(ValueError, match="Fairness violation"):
        callback(y_true, y_pred, pd.Series(gender, name="gender"))

    y_true, y_pred, gender = _fair_run()
    results = callback(y_true, y_pred, pd.Series(gender, name="gender"))
    assert abs(results["demographic_parity_difference"]) < 0.1


# DEFECT 6: the trackers logged the series that worked and never named the rest


def test_wandb_logger_names_the_metrics_and_group_stats_it_could_not_measure(fake_wandb):
    """An unmeasurable series was logged as nothing, and nothing said so.

    Measured 2026-09-27 on a 300-row `classification_fairness_report` where group
    F has no positive labels:

        logged   fairness/group/M/tpr = 0.7826
        absent   fairness/group/F/tpr
        absent   fairness/equalized_odds_difference,
                 fairness/equal_opportunity_difference, fairness/auroc_parity
        while    fairness/fairness_score_status = "assessed"
                 fairness/n_groups_excluded = 0

    A reader comparing TPR across groups saw ONE bar and every disclosure field
    on the run said nothing had been left out.
    """
    report = _mixed_report()
    logged = log_fairness_to_wandb(report, log_artifacts=False)

    summary = fake_wandb.run.summary
    assert summary["fairness/n_metrics_not_measured"] == 3
    for name in ("equalized_odds_difference", "equal_opportunity_difference", "auroc_parity"):
        assert name in summary["fairness/metrics_not_measured"]
    assert summary["fairness/n_group_stats_not_measured"] == 1
    assert "F/tpr" in summary["fairness/group_stats_not_measured"]
    # The measured half is untouched, so this is a disclosure, not a refusal.
    assert fake_wandb.logged["fairness/group/M/tpr"] == pytest.approx(0.7826, abs=1e-3)
    assert "fairness/group/F/tpr" not in fake_wandb.logged
    # Reachable in process, for a caller that reads only the return value.
    assert logged["fairness/n_group_stats_not_measured"] == 1


def test_mlflow_logger_names_the_metrics_and_group_stats_it_could_not_measure(fake_mlflow):
    """The MLflow path had the identical gap and gets the identical disclosure."""
    report = _mixed_report()
    logged = log_fairness_to_mlflow(report, log_artifacts=False)

    params = fake_mlflow.params
    assert params["fairness.n_metrics_not_measured"] == 3
    assert "auroc_parity" in params["fairness.metrics_not_measured"]
    assert params["fairness.n_group_stats_not_measured"] == 1
    assert "F.tpr" in params["fairness.group_stats_not_measured"]
    assert "fairness.group.M.tpr" in fake_mlflow.metrics
    assert "fairness.group.F.tpr" not in fake_mlflow.metrics
    assert logged["fairness.n_metrics_not_measured"] == 3


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_logger_disclosure_is_empty_when_everything_was_measured(backend, fake_mlflow, fake_wandb):
    """Over-correction control on BOTH trackers.

    The field is emitted on every run, never only on failure, so its absence can
    never be ambiguous between "nothing was missed" and "this version did not
    record it". On a fully measured report it must read zero, not be missing.
    """
    report = _measurable_report()
    if backend == "mlflow":
        log_fairness_to_mlflow(report, log_artifacts=False)
        assert fake_mlflow.params["fairness.n_metrics_not_measured"] == 0
        assert fake_mlflow.params["fairness.metrics_not_measured"] == "[]"
        assert fake_mlflow.params["fairness.n_group_stats_not_measured"] == 0
        assert "fairness.group.F.tpr" in fake_mlflow.metrics
    else:
        log_fairness_to_wandb(report, log_artifacts=False)
        assert fake_wandb.run.summary["fairness/n_metrics_not_measured"] == 0
        assert fake_wandb.run.summary["fairness/group_stats_not_measured"] == "[]"
        assert "fairness/group/F/tpr" in fake_wandb.logged


def test_auto_log_fairness_carries_the_disclosure_through_the_decorator(fake_wandb):
    """The decorator delegates to the logger, so the disclosure must arrive too.

    auto_log_fairness itself refuses honestly already (it warns and substitutes
    nothing for a metric the analyzer did not produce), but before the logger fix
    a decorated training run recorded only the series that worked.
    """
    rng = np.random.default_rng(11)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    y_true = np.where(gender == "F", 0, rng.integers(0, 2, n))
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)

    @auto_log_fairness(backend="wandb")
    def train():
        return y_pred, y_true, gender

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train()

    # The decorator swallows a tracker failure into a warning, so confirm the
    # report actually landed before reading the disclosure off the summary.
    assert fake_wandb.artifacts == ["fairness_report"]
    summary = fake_wandb.run.summary
    assert summary["fairness/n_group_stats_not_measured"] >= 1
    assert "F/tpr" in summary["fairness/group_stats_not_measured"]


def test_auto_log_fairness_warns_and_substitutes_nothing_for_a_missing_metric(fake_wandb):
    """CORRECT, pinned: a requested metric the analyzer never produced.

    Executed: a decorator asked for 'disparate_impact' logged nothing for it and
    warned "requested metric(s) disparate_impact were not produced by the
    analyzer for this run, so nothing is logged for them (no value is
    substituted)". No stand-in number reached the tracker.
    """
    y_true, y_pred, gender = _fair_run(n=200)

    @auto_log_fairness(backend="wandb", metrics=["disparate_impact"])
    def train():
        return y_pred, y_true, gender

    with pytest.warns(UserWarning, match="were not produced by the analyzer"):
        train()

    # The run really did reach the tracker (so the absence below is a decision,
    # not a crash), and the requested name is simply not there.
    assert fake_wandb.artifacts == ["fairness_report"]
    assert not [key for key in fake_wandb.logged if key.endswith("disparate_impact")]
    assert "fairness/fairness_score_status" in fake_wandb.logged


# CORRECT, pinned: the units that already refused honestly


@pytest.mark.parametrize(
    "tag,left,right",
    [
        ("constant numeric side", "const", "numeric"),
        ("single level nominal side", "one_level", "nominal"),
        ("too few rows", "short", "short"),
        ("unique per row nominal", "unique", "nominal"),
        ("list valued objects", "json", "nominal"),
    ],
)
def test_association_strength_returns_nan_never_zero_when_undefined(tag, left, right):
    """CORRECT, pinned. Executed on each degenerate pair: nan plus a warning.

    0.0 on this scale is "X does not track Y at all", the strongest reassurance a
    proxy screen can give. Measured: every case below returns nan and names the
    pair. The unique-per-row case is the one that used to score 0.0 for a column
    that determines gender perfectly.
    """
    rng = np.random.default_rng(1)
    n = 200
    columns = {
        "const": pd.Series(np.full(n, 5.0), name="const_fee"),
        "numeric": pd.Series(rng.normal(0, 1, n), name="salary"),
        "one_level": pd.Series(["F"] * n, name="gender"),
        "nominal": pd.Series(rng.choice(["F", "M"], n), name="gender"),
        "unique": pd.Series([f"r{i}" for i in range(n)], name="customer_ref"),
        "json": pd.Series([[1, 2]] * n, name="region_json"),
        "short": pd.Series(rng.normal(0, 1, 5), name="salary"),
    }
    with pytest.warns((ProxyScanIncompleteWarning, UserWarning)):
        value = association_strength(columns[left], columns[right])

    assert np.isnan(value), f"{tag} returned {value!r}"


def test_association_strength_measures_a_real_association():
    """Over-correction control for the refusals above."""
    rng = np.random.default_rng(1)
    n = 200
    x = pd.Series(rng.normal(0, 1, n), name="x")
    y = pd.Series(x * 0.9 + rng.normal(0, 0.4, n), name="y")
    assert association_strength(x, y) == pytest.approx(0.91, abs=0.05)

    gender = rng.choice(["F", "M"], n)
    zip_code = pd.Series(np.where(gender == "F", "90210", "10001"), name="zip")
    assert association_strength(zip_code, pd.Series(gender, name="gender")) > 0.99


def test_identify_proxy_features_carries_the_columns_it_could_not_assess():
    """CORRECT, pinned. An empty proxy list is not "no proxies exist".

    Executed on 50 rows whose protected attribute holds ONE gender: the result is
    an empty list, and the incompleteness travels on the result object rather
    than only on the warning channel.
    """
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"gender": ["F"] * 50, "salary": rng.normal(60000, 5000, 50)})
    with pytest.warns((ProxyScanIncompleteWarning, UserWarning)):
        proxies = identify_proxy_features(df, "gender")

    assert list(proxies) == []
    assert getattr(proxies, "not_assessable", []) != []
    assert "salary" in " ".join(proxies.not_assessable)


def test_identify_proxy_features_finds_a_real_proxy():
    """Over-correction control: a perfect ZIP proxy is still reported as high risk."""
    rng = np.random.default_rng(1)
    n = 200
    gender = rng.choice(["F", "M"], n)
    df = pd.DataFrame({"gender": gender, "zip": np.where(gender == "F", "90210", "10001")})
    proxies = identify_proxy_features(df, "gender")

    assert [p["column"] for p in proxies] == ["zip"]
    assert proxies[0]["risk_level"] == "high"
    assert getattr(proxies, "not_assessable", []) == []


def test_rank_fairness_issues_reports_nan_max_disparity_not_zero():
    """CORRECT, pinned. A maximum over an empty violation set is not zero.

    Executed on 200 rows holding a single gender: summary['max_disparity'] is
    NaN, n_attributes_checked is 0 against n_attributes_requested 1, and the
    recommendations lead with COULD NOT CHECK. 0.0 there is the value a
    perfectly fair run earns.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"gender": ["F"] * n, "tenure_months": rng.integers(1, 90, n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rank_fairness_issues(df, rng.integers(0, 2, n))

    summary = result["summary"]
    assert np.isnan(summary["max_disparity"])
    assert summary["n_attributes_checked"] == 0
    assert summary["n_attributes_requested"] == 1
    assert summary["n_not_assessable"] == 1
    assert any("COULD NOT CHECK" in rec for rec in result["recommendations"])


def test_rank_fairness_issues_measures_a_real_disparity():
    """Over-correction control: a measured disparity is still a number."""
    rng = np.random.default_rng(0)
    n = 200
    gender = np.where(rng.random(n) < 0.5, "M", "F")
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({"gender": gender, "tenure_months": rng.integers(1, 90, n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rank_fairness_issues(df, y_pred)

    assert result["summary"]["max_disparity"] > 0.3
    assert result["summary"]["n_violations"] >= 1


@pytest.mark.parametrize(
    "point,lower,upper",
    [
        (float("nan"), float("nan"), float("nan")),
        (0.4, None, None),
        (0.4, False, True),
        (0.4, -np.inf, np.inf),
    ],
)
def test_explain_confidence_interval_refuses_an_interval_that_was_never_built(point, lower, upper):
    """CORRECT, pinned. Every comparison against NaN is False.

    Before this guard, `nan <= 0 <= nan` being False sent the method down the
    else branch and it announced "The interval excludes zero, suggesting the
    observed disparity is statistically significant" at severity 'info', for an
    interval that does not exist. Executed on all four shapes above: severity
    'could_not_check' and no significance claim.
    """
    explanation = FairExplAIner().explain_confidence_interval(
        "demographic_parity_difference", point, lower, upper
    )

    assert explanation.severity == "could_not_check"
    assert "COULD NOT CHECK" in explanation.evaluation
    assert "statistically significant" not in explanation.evaluation


def test_explain_confidence_interval_still_reads_a_real_interval():
    """Over-correction control."""
    explanation = FairExplAIner().explain_confidence_interval(
        "demographic_parity_difference", 0.42, 0.31, 0.53
    )
    assert explanation.severity == "info"
    assert "excludes zero" in explanation.evaluation


@pytest.mark.parametrize("value", [float("nan"), None, True])
def test_explain_effect_size_refuses_an_effect_that_was_never_computed(value):
    """CORRECT, pinned, and the guard sits ABOVE the dispatch.

    All four interpretation branches grade on an abs()/comparison ladder, and
    every comparison against NaN is False, so an uncomputed effect size fell
    through to the most alarming rung: measured on 80 real rows where nobody was
    selected in either group, risk_ratio came back severity 'high' with "one
    group is nanx less likely" and "The effect is large and requires action."
    """
    explanation = FairExplAIner().explain_effect_size("risk_ratio", value, "A", "B")

    assert explanation.severity == "could_not_check"
    assert "COULD NOT CHECK" in explanation.evaluation


def test_explain_effect_size_keeps_total_exclusion_as_a_real_finding():
    """An INFINITE ratio must NOT be swept into could-not-check.

    One arm receiving no positive outcomes at all is the strongest disparate
    impact reading there is; calling it unmeasurable would delete a real finding.
    Executed: severity 'critical' with "Total exclusion" in the prose.
    """
    explainer = FairExplAIner()
    unbounded = explainer.explain_effect_size("risk_ratio", float("inf"), "A", "B")
    assert unbounded.severity == "critical"
    assert "Total exclusion" in unbounded.evaluation

    measured = explainer.explain_effect_size("cohens_d", 0.9, "A", "B")
    assert measured.severity == "high"
    assert "Large effect size" in measured.evaluation
