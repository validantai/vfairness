"""Wave-4 pins for the evaluation.vfairness_metrics units whose grade was overturned.

Every test here is a DEFECT-SIDE pin: it fails if the neutral value that was
being published in place of an unmeasurable one comes back. Each pin is paired
with a control asserting the healthy case's REAL number or rendering, because a
guard that refuses everything passes every refusal test.

The units, and the sibling door that was still open in each after the earlier
wave fixed the door the judgement named:

* ``visualization.plot_confidence_intervals`` and its dashboard twin
  ``_add_ci_panel``: the same ``point_estimate`` key read with a fabricating
  ``0`` default in the acceptable-region loop and in the whole twin.
* ``integrations.log_fairness_to_mlflow`` / ``log_fairness_to_wandb``: measured
  series the writers drop and no disclosure field names.
* ``robustness.permutation_test_equal_opportunity``: the permutation null was
  computed over rows the observed statistic cannot see.
* ``discovery.classify_column_roles``: one column declared as BOTH target and
  prediction reached no completeness channel.
* ``explanation_diagnostics.multi_seed_adversarial_probe``: the discriminating
  background-spread pin that was swapped out for two that cannot fail.
"""

import hashlib
import importlib.util
import io
import warnings

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
    plt = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
    import matplotlib.pyplot as plt
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)
needs_plotly = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="needs the optional [dashboard] extra (plotly)",
)

from vfairness.evaluation.vfairness_metrics import visualization as viz  # noqa: E402

# ───────────────────────────────────────────────────────────────────────────
# visualization.plot_confidence_intervals
# ───────────────────────────────────────────────────────────────────────────


def _ci_report(entry, threshold=0.1):
    return {
        "metrics_with_ci": {"demographic_parity_difference": entry},
        "thresholds_used": {"demographic_parity_difference": threshold},
    }


def _draw_ci(entry, threshold=0.1):
    """Render one row and return (region extents, annotation texts, warnings)."""
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            viz.plot_confidence_intervals(_ci_report(entry, threshold), ax=ax)
        bands = [
            (round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6))
            for p in ax.patches
            if p.get_gid() == viz.ACCEPTABLE_REGION_GID
        ]
        texts = [t.get_text() for t in ax.texts]
        messages = [str(c.message) for c in caught if issubclass(c.category, UserWarning)]
        return bands, texts, messages
    finally:
        plt.close(fig)


def _png_sha(entry, threshold=0.1):
    fig, ax = plt.subplots(figsize=(6, 2))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viz.plot_confidence_intervals(_ci_report(entry, threshold), ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        return hashlib.sha256(buf.getvalue()).hexdigest()
    finally:
        plt.close(fig)


@needs_matplotlib
def test_ci_chart_draws_no_acceptable_region_for_an_absent_point_estimate():
    """An ABSENT point estimate must not be shaded green as compliant.

    The acceptable-region loop read ``get("point_estimate", 0)``, so a row
    carrying only a lower bound got a band spanning -0.100..0.100, byte-identical
    in extent to a row measured at 0.420, under the label "Acceptable range".
    """
    bands, texts, _ = _draw_ci({"lower_bound": 0.02})
    assert bands == [], bands
    assert any(viz.COULD_NOT_CHECK_TEXT in t for t in texts), texts


@pytest.mark.parametrize(
    "entry",
    [
        {"lower_bound": 0.02},
        {"point_estimate": None, "lower_bound": 0.02},
        {"point_estimate": float("nan"), "lower_bound": 0.02},
    ],
    ids=["absent", "none", "nan"],
)
@needs_matplotlib
def test_ci_chart_refuses_every_spelling_of_an_absent_point_estimate(entry):
    """Absence has more than one spelling and two of the three were already shut."""
    bands, texts, _ = _draw_ci(entry)
    assert bands == [], (entry, bands)
    assert any("the value was not measured" in t for t in texts), texts


@needs_matplotlib
def test_ci_chart_does_not_raise_on_a_tuple_shaped_interval():
    """``compute_metric_with_ci`` returns a tuple, and a caller passes it straight through.

    Line 2742 was the only unguarded reader of the entry, so it raised
    ``AttributeError: 'tuple' object has no attribute 'get'`` while both
    neighbouring loops shape-guarded the identical value.
    """
    bands, texts, _ = _draw_ci((0.42, 0.31, 0.53))
    assert bands == [], bands
    assert any(viz.NOT_MEASURED_TICK in t for t in texts), texts


@needs_matplotlib
def test_ci_chart_does_not_invent_a_zero_width_interval_from_a_lone_point_estimate():
    """An absent BOUND defaulted to the point estimate, printing "[0.420, 0.420]"."""
    _bands, texts, _ = _draw_ci({"point_estimate": 0.42})
    label = [t for t in texts if t.startswith("0.420")]
    assert label, texts
    assert label[0] == f"0.420\n[{viz.NOT_MEASURED_TICK}, {viz.NOT_MEASURED_TICK}]", label


@needs_matplotlib
def test_control_ci_chart_still_renders_the_real_numbers_and_the_real_region():
    """The healthy case keeps its band and its three real numbers."""
    bands, texts, messages = _draw_ci(
        {"point_estimate": 0.42, "lower_bound": 0.31, "upper_bound": 0.53}
    )
    assert bands == [(-0.1, 0.1)], bands
    assert "0.420\n[0.310, 0.530]" in texts, texts
    assert messages == [], messages


@needs_matplotlib
def test_control_three_ci_states_render_three_different_images():
    """The RENDERED artifact must differ, not only the returned data.

    A pin asserting a dict would be green for a renderer that draws all three
    states identically, which is a defect found elsewhere in this campaign.
    """
    compliant = _png_sha({"point_estimate": 0.02, "lower_bound": 0.0, "upper_bound": 0.05})
    violating = _png_sha({"point_estimate": 0.42, "lower_bound": 0.31, "upper_bound": 0.53})
    could_not_check = _png_sha({"lower_bound": 0.02})
    assert len({compliant, violating, could_not_check}) == 3, (
        compliant,
        violating,
        could_not_check,
    )


# ───────────────────────────────────────────────────────────────────────────
# visualization._add_ci_panel: the dashboard twin left behind
# ───────────────────────────────────────────────────────────────────────────


def _ci_panel(entry):
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        viz._add_ci_panel(fig, _ci_report(entry), viz._get_palette("modern"), 1, 1)
    messages = [str(c.message) for c in caught if issubclass(c.category, UserWarning)]
    return fig, messages


@needs_plotly
def test_ci_dashboard_panel_places_no_marker_for_an_absent_point_estimate():
    """The twin put a diamond at x = 0, which on a disparity axis is perfect parity.

    Its hover read "Point Estimate: 0.0000" and "95% CI: [0.0200, 0.0000]", an
    inverted zero-width interval, with no warning anywhere.
    """
    fig, messages = _ci_panel({"lower_bound": 0.02})
    assert list(fig.data) == [], [t.x for t in fig.data]
    annotations = [a.text for a in fig.layout.annotations]
    assert any(viz.NOT_MEASURED_TICK in a for a in annotations), annotations
    assert any("not a value at zero" in m for m in messages), messages


@needs_plotly
def test_control_ci_dashboard_panel_still_draws_the_real_point_and_interval():
    fig, messages = _ci_panel({"point_estimate": 0.42, "lower_bound": 0.31, "upper_bound": 0.53})
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.x == (0.42,), trace.x
    assert trace.error_x.array == (pytest.approx(0.11),), trace.error_x.array
    assert trace.error_x.arrayminus == (pytest.approx(0.11),), trace.error_x.arrayminus
    assert "Point Estimate: 0.4200" in trace.hovertemplate
    assert "95% CI: [0.3100, 0.5300]" in trace.hovertemplate
    assert messages == [], messages


def test_ci_field_reads_absence_as_absence_and_a_measurement_as_itself():
    """The one reader both surfaces route through."""
    assert viz._ci_field({"point_estimate": 0.42}, "point_estimate") == 0.42
    assert not viz._is_measured(viz._ci_field({}, "point_estimate"))
    assert not viz._is_measured(viz._ci_field({"point_estimate": None}, "point_estimate"))
    assert not viz._is_measured(viz._ci_field({"point_estimate": float("nan")}, "point_estimate"))
    assert not viz._is_measured(viz._ci_field((0.42, 0.31, 0.53), "point_estimate"))
    # np.float32 is a real measurement; the repo's own is_measured helper says so.
    assert viz._ci_field({"point_estimate": np.float32(0.25)}, "point_estimate") == np.float32(0.25)


# ───────────────────────────────────────────────────────────────────────────
# integrations.log_fairness_to_mlflow / log_fairness_to_wandb
# ───────────────────────────────────────────────────────────────────────────


import sys  # noqa: E402
import types  # noqa: E402

from vfairness.evaluation.vfairness_metrics import classification  # noqa: E402
from vfairness.evaluation.vfairness_metrics.integrations import (  # noqa: E402
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


def _real_producer_effect_sizes():
    """What ``compute_effect_sizes`` actually returns, not a hand-written subset.

    The defect was invisible to every hand-written fixture, because the fixtures
    carried only the two fields the writers happened to support.
    """
    rng = np.random.default_rng(7)
    n = 200
    attr = np.array(["A"] * 100 + ["B"] * 100)
    y = rng.integers(0, 2, n)
    p = rng.integers(0, 2, n)
    return classification.compute_effect_sizes(y, p, attr)


def _tracker_report(effect_sizes=None, ci=None):
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.21},
        "metrics_with_ci": {"demographic_parity_difference": ci} if ci else {},
        "effect_sizes": effect_sizes or {},
        "group_stats": {},
        "assessment": {"fairness_score": 0.7},
        "data_info": {"n_samples": 200, "n_groups": 2},
    }


def test_mlflow_writes_the_odds_ratio_and_cohens_h_the_producer_measured(fake_mlflow):
    """Two measured effect sizes reached the tracker as nothing, named nowhere.

    ``compute_effect_sizes`` returns four effect sizes per pair; both writers and
    the shared disclosure hardcoded two of them, so a measured odds ratio with
    its interval and a measured Cohen's h were dropped with
    ``n_effect_sizes_not_measured = 0`` and no warning.
    """
    effects = _real_producer_effect_sizes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_fairness_to_mlflow(_tracker_report(effects), log_artifacts=False)
    written = sorted(k for k in fake_mlflow.metrics if ".effect_size." in k)
    assert written == [
        "fairness.effect_size.A_vs_B.cohens_d",
        "fairness.effect_size.A_vs_B.cohens_h",
        "fairness.effect_size.A_vs_B.odds_ratio",
        "fairness.effect_size.A_vs_B.risk_ratio",
    ], written
    # The REAL numbers, not merely presence.
    assert fake_mlflow.metrics["fairness.effect_size.A_vs_B.odds_ratio"] == pytest.approx(
        effects["A_vs_B"]["odds_ratio"][0]
    )
    assert fake_mlflow.metrics["fairness.effect_size.A_vs_B.cohens_h"] == pytest.approx(
        effects["A_vs_B"]["cohens_h_positive_rate"]
    )
    assert fake_mlflow.params["fairness.n_effect_sizes_not_measured"] == 0
    assert [str(c.message) for c in caught] == []


def test_wandb_writes_the_odds_ratio_and_cohens_h_the_producer_measured(fake_wandb):
    effects = _real_producer_effect_sizes()
    log_fairness_to_wandb(_tracker_report(effects), log_artifacts=False)
    written = sorted(k for k in fake_wandb.logged if "/effect_size/" in k)
    assert written == [
        "fairness/effect_size/A_vs_B/cohens_d",
        "fairness/effect_size/A_vs_B/cohens_h",
        "fairness/effect_size/A_vs_B/odds_ratio",
        "fairness/effect_size/A_vs_B/risk_ratio",
    ], written
    assert fake_wandb.logged["fairness/effect_size/A_vs_B/odds_ratio"] == pytest.approx(
        effects["A_vs_B"]["odds_ratio"][0]
    )
    assert fake_wandb.run.summary["fairness/n_effect_sizes_not_measured"] == 0


@pytest.mark.parametrize("backend", ["mlflow", "wandb"])
def test_an_unmeasured_odds_ratio_and_cohens_h_are_named_not_silent(
    backend, fake_mlflow, fake_wandb
):
    """The other direction: absent is named, so the disclosure follows the write."""
    nan = float("nan")
    effects = {
        "A_vs_B": {
            "cohens_d_positive_rate": 0.08,
            "cohens_h_positive_rate": nan,
            "risk_ratio": (1.07, 0.83, 1.39),
            "odds_ratio": (nan, nan, nan),
        }
    }
    report = _tracker_report(effects)
    if backend == "mlflow":
        log_fairness_to_mlflow(report, log_artifacts=False)
        fields, sep = fake_mlflow.params, "."
        written = sorted(k for k in fake_mlflow.metrics if ".effect_size." in k)
    else:
        log_fairness_to_wandb(report, log_artifacts=False)
        fields, sep = fake_wandb.run.summary, "/"
        written = sorted(k for k in fake_wandb.logged if "/effect_size/" in k)
    assert fields[f"fairness{sep}n_effect_sizes_not_measured"] == 2, fields[
        f"fairness{sep}effect_sizes_not_measured"
    ]
    named = fields[f"fairness{sep}effect_sizes_not_measured"]
    assert f"A_vs_B{sep}cohens_h" in named, named
    assert f"A_vs_B{sep}odds_ratio" in named, named
    # And the two that WERE measured are still written: not an over-correction.
    assert len(written) == 2, written


def test_wandb_writes_the_measured_standard_error_its_disclosure_names(fake_wandb):
    """The arm without its complement.

    ``_CI_SERIES`` is shared with the disclosure so that a series named as
    unmeasured is exactly a series not written, and this writer had no
    ``standard_error`` branch at all: a measured 0.06 was absent from the panel
    with ``n_intervals_not_measured = 0`` and no warning.
    """
    ci = {
        "point_estimate": 0.42,
        "lower_bound": 0.31,
        "upper_bound": 0.53,
        "standard_error": 0.06,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_fairness_to_wandb(_tracker_report(ci=ci), log_artifacts=False)
    key = "fairness/demographic_parity_difference/std_error"
    assert key in fake_wandb.logged, sorted(fake_wandb.logged)
    assert fake_wandb.logged[key] == pytest.approx(0.06)
    assert fake_wandb.run.summary["fairness/n_intervals_not_measured"] == 0
    assert [str(c.message) for c in caught] == []


def test_wandb_names_an_unmeasured_standard_error_and_writes_no_key_for_it(fake_wandb):
    ci = {
        "point_estimate": 0.42,
        "lower_bound": 0.31,
        "upper_bound": 0.53,
        "standard_error": float("nan"),
    }
    log_fairness_to_wandb(_tracker_report(ci=ci), log_artifacts=False)
    assert "fairness/demographic_parity_difference/std_error" not in fake_wandb.logged
    summary = fake_wandb.run.summary
    assert summary["fairness/n_intervals_not_measured"] == 1
    assert "demographic_parity_difference/std_error" in summary["fairness/intervals_not_measured"]


def test_control_both_trackers_write_the_same_four_interval_series(fake_mlflow, fake_wandb):
    """The two panels must not disagree about which interval series exist."""
    ci = {
        "point_estimate": 0.42,
        "lower_bound": 0.31,
        "upper_bound": 0.53,
        "standard_error": 0.06,
    }
    log_fairness_to_mlflow(_tracker_report(ci=ci), log_artifacts=False)
    log_fairness_to_wandb(_tracker_report(ci=ci), log_artifacts=False)
    ml = sorted(
        k.rsplit(".", 1)[-1]
        for k in fake_mlflow.metrics
        if k.startswith("fairness.demographic_parity_difference.")
    )
    wb = sorted(
        k.rsplit("/", 1)[-1]
        for k in fake_wandb.logged
        if k.startswith("fairness/demographic_parity_difference/")
    )
    assert ml == wb == ["ci_lower", "ci_upper", "std_error", "value"], (ml, wb)


# ───────────────────────────────────────────────────────────────────────────
# robustness.permutation_test_equal_opportunity
# ───────────────────────────────────────────────────────────────────────────


from vfairness.evaluation.vfairness_metrics.robustness import (  # noqa: E402
    permutation_test_equal_opportunity,
)


def _eo_design(unlabelled_are_positive, third_label=None):
    """206 rows: two readable groups of 53 plus a 100-row block.

    The READABLE design is identical in both cases: six positive rows, group 0 at
    TPR 3/3 and group 1 at TPR 1/3, so the observed statistic is 2/3 either way.
    The only thing that changes is whether the 100 rows the statistic cannot
    attribute carry positive labels.
    """
    y_true = np.zeros(206)
    y_pred = np.zeros(206)
    tail = [float("nan")] * 100 if third_label is None else [third_label] * 100
    attr = np.array([0.0] * 53 + [1.0] * 53 + tail)
    y_true[0:3] = 1
    y_pred[0:3] = 1
    y_true[53:56] = 1
    y_pred[53] = 1
    if unlabelled_are_positive:
        y_true[106:206] = 1
        y_pred[106:156] = 1
    return y_true, y_pred, attr


def _run_eo(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_equal_opportunity(*args, **kwargs)
    return result, [str(c.message) for c in caught]


def test_eo_permutation_verdict_does_not_depend_on_rows_the_statistic_cannot_read():
    """The null was drawn over a different population than the statistic.

    ``eo_diff`` uses ``attr == g``, so a row with an unmatchable label is
    invisible to the observed statistic, but the permutation shuffled the WHOLE
    label array and moved real labels onto those rows. Measured before the fix,
    n_permutations=2000, random_state=1: the SAME observed 0.6667 gave
    p = 0.000500 with significant_at_05 True when the 100 unlabelled rows were
    positive and p = 0.322059 with significant_at_05 False when they were not.
    """
    hot, _ = _run_eo(*_eo_design(True), n_permutations=2000, random_state=1)
    cold, _ = _run_eo(*_eo_design(False), n_permutations=2000, random_state=1)
    assert hot.observed_statistic == pytest.approx(2 / 3)
    assert cold.observed_statistic == pytest.approx(2 / 3)
    # The readable design is identical, so the null must be too.
    assert hot.p_value == pytest.approx(cold.p_value), (hot.p_value, cold.p_value)
    assert hot.significant_at_05 == cold.significant_at_05
    assert np.mean(hot.null_distribution) == pytest.approx(
        np.mean(cold.null_distribution), abs=1e-12
    )
    assert hot.groups_omitted == ("nan",)


def test_eo_permutation_says_it_held_the_unattributable_rows_fixed():
    """Could-not-check has to be visible where a reader looks."""
    _result, messages = _run_eo(*_eo_design(True), n_permutations=200, random_state=1)
    assert any("held" in m and "FIXED" in m for m in messages), messages
    assert any("100 of 206 row(s)" in m for m in messages), messages


def test_eo_permutation_design_floor_counts_only_attributable_positive_rows():
    """``np.sum(y_true == 1)`` counted the positives in the unmatchable block.

    The floor it feeds is a LOWER bound, so an inflated row count only ever makes
    the design look MORE powerful. Before: n_design_rows described 106 rows where
    the statistic reads 6, the effective-design floor did not bind at all, and
    min_attainable_p_value was the resample floor 1/(B+1) = 4.998e-04.
    """
    result, _ = _run_eo(*_eo_design(True), n_permutations=2000, random_state=1)
    resample_floor = 1.0 / 2001
    assert result.min_attainable_p_value > resample_floor * 10, (
        result.min_attainable_p_value,
        resample_floor,
    )


def test_control_a_real_third_label_is_unaffected():
    """Over-correction control: a group that IS a group keeps its verdict."""
    result, messages = _run_eo(
        *_eo_design(True, third_label=2.0), n_permutations=2000, random_state=1
    )
    assert result.observed_statistic == pytest.approx(2 / 3)
    assert result.p_value == pytest.approx(1.0 / 2001)
    assert result.significant_at_05 is True
    assert result.groups_omitted == ()
    assert not [m for m in messages if "held" in m], messages


def test_control_an_ordinary_clean_design_returns_its_real_seeded_p_value():
    """The unattributable-free path is untouched, asserted by its REAL number."""
    rng = np.random.default_rng(3)
    attr = np.array(["A"] * 100 + ["B"] * 100)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    y_pred[(attr == "A") & (y_true == 1)] = 1
    result, messages = _run_eo(y_true, y_pred, attr, n_permutations=500, random_state=11)
    assert result.observed_statistic == pytest.approx(0.4339622641509434)
    assert result.p_value == pytest.approx(0.001996007984031936)
    assert result.significant_at_05 is True
    assert result.groups_omitted == ()
    assert messages == [], messages


# ───────────────────────────────────────────────────────────────────────────
# discovery.classify_column_roles
# ───────────────────────────────────────────────────────────────────────────


import pandas as pd  # noqa: E402

from vfairness.evaluation.vfairness_metrics.discovery import (  # noqa: E402
    classify_column_roles,
)


def _roles_frame():
    rng = np.random.default_rng(5)
    n = 400
    return pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "age": rng.integers(20, 65, n),
            "label": rng.integers(0, 2, n),
            "y_hat": rng.integers(0, 2, n),
        }
    )


def _classify(**kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = classify_column_roles(_roles_frame(), **kwargs)
    messages = [str(c.message) for c in caught if issubclass(c.category, UserWarning)]
    return schema, messages


def test_a_column_declared_as_both_target_and_prediction_reaches_a_completeness_channel():
    """One column cannot be both the answer key and the model's decision.

    The declaration-versus-declaration collision resolved in silence: the answer
    key won, ``model_output`` came back EMPTY, and not_assessable, mismatches,
    refuse_reason and the warning channel were all clean. Three live Pulse
    consumers read that empty list as a fact about the data.
    """
    schema, messages = _classify(
        declared_protected=["gender"], declared_target="y_hat", declared_prediction="y_hat"
    )
    # The resolution is deliberately unchanged: only the silence is closed.
    assert schema["model_output"] == []
    assert "y_hat" in schema["oracle_columns"]
    assert schema["n_not_assessable"] == 1, schema["not_assessable"]
    assert "declared as BOTH the target and the model output" in schema["not_assessable"][0]
    collision = [
        m for m in schema["mismatches"] if m["column"] == "y_hat" and m["inferred"] == "oracle"
    ]
    assert collision, schema["mismatches"]
    assert "CONTRADICT" in collision[0]["detail"]
    assert any("declared as BOTH declared_target and declared_prediction" in m for m in messages), (
        messages
    )


def test_control_distinct_target_and_prediction_declarations_are_untouched():
    """Over-correction control: the healthy case keeps its real classification."""
    schema, messages = _classify(
        declared_protected=["gender"], declared_target="label", declared_prediction="y_hat"
    )
    assert schema["model_output"] == ["y_hat"]
    assert schema["oracle_columns"] == ["label"]
    assert schema["n_not_assessable"] == 0, schema["not_assessable"]
    assert not [m for m in schema["mismatches"] if m["column"] == "y_hat"], schema["mismatches"]
    assert schema["refuse"] is False
    assert messages == [], messages


# ───────────────────────────────────────────────────────────────────────────
# explanation_diagnostics.multi_seed_adversarial_probe
# ───────────────────────────────────────────────────────────────────────────


from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (  # noqa: E402
    _background_cannot_be_perturbed,
    multi_seed_adversarial_probe,
)


def _probe_world():
    """A real 400-row background, a scaffolded model and a clean one."""
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(400, 3))

    def scaffolded(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        on_manifold = np.any(np.all(np.isclose(arr[:, None, :], bg[None, :, :]), axis=2), axis=1)
        p = np.where(on_manifold, 0.1, 0.9)
        return np.column_stack([1 - p, p])

    def clean(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        p = np.full(len(arr), 0.5)
        return np.column_stack([1 - p, p])

    return bg, bg[0], scaffolded, clean


def _probe(model, bg, x, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = multi_seed_adversarial_probe(
            predict_fn=model, x=x, background=bg, n_perturbations=60, n_seeds=4, **kwargs
        )
    return result, [str(c.message) for c in caught]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_probe_refuses_a_threshold_no_gap_can_exceed(threshold):
    """`measured > threshold` is False for every gap, so flag False reads as clean.

    The twin in ``xai/diagnostics/adversarial.py`` refuses this at the top of its
    body and its own comment said this copy carried the defect. Measured before
    the fix on a scaffolded model: flag False, confidence 1.0 (this function's own
    words for maximum certainty the explainer is clean) over a MEASURED mean_gap
    of 0.8, with zero warnings.
    """
    bg, x, scaffolded, _clean = _probe_world()
    result, messages = _probe(scaffolded, bg, x, threshold=threshold)
    assert result.flag is None, result
    assert np.isnan(result.confidence) and np.isnan(result.mean_gap)
    assert "COULD NOT CHECK" in result.reason
    assert messages, "a refusal nobody can see is not a disclosure"


def test_probe_refuses_a_negative_threshold_every_gap_exceeds():
    """A probe gap is an absolute difference, so `gap > -0.5` cannot be False.

    Measured before the fix with threshold=-0.5: the CLEAN flat model came back
    flag True, confidence 1.0, fired_fraction 1.0 over a mean_gap of 0.0, an
    unfalsifiable positive finding. A bound that tests only `isfinite` leaves the
    finite vacuous one open.
    """
    bg, x, _scaffolded, clean = _probe_world()
    result, messages = _probe(clean, bg, x, threshold=-0.5)
    assert result.flag is None, result
    assert "never below zero" in result.not_run_because, result.not_run_because
    assert messages


def test_control_the_probe_still_measures_both_verdicts_with_a_usable_threshold():
    """Over-correction control, with the REAL numbers on both sides."""
    bg, x, scaffolded, clean = _probe_world()
    hot, hot_messages = _probe(scaffolded, bg, x, threshold=0.3)
    assert hot.flag is True
    assert hot.confidence == pytest.approx(1.0)
    assert hot.mean_gap == pytest.approx(0.8)
    assert hot.fired_fraction == pytest.approx(1.0)
    assert hot_messages == [], hot_messages

    cold, cold_messages = _probe(clean, bg, x, threshold=0.3)
    assert cold.flag is False
    assert cold.confidence == pytest.approx(1.0)
    assert cold.mean_gap == pytest.approx(0.0)
    assert cold.fired_fraction == pytest.approx(0.0)
    assert cold_messages == [], cold_messages

    # A strict-but-usable threshold of exactly 0.0 is still a real bound.
    zero, _ = _probe(scaffolded, bg, x, threshold=0.0)
    assert zero.flag is True
    assert zero.mean_gap == pytest.approx(0.8)


def test_the_background_spread_floor_is_relative_not_an_equality_with_zero():
    """The discriminating pin the graded row's named evidence did not carry.

    ``tests/test_bgl3_evaluation_4.py`` and ``tests/test_bgl5_evaluation_4.py``
    both stay fully green when ``spread > floor`` is reverted to ``spread > 0.0``
    (measured: 58 passed with the defect live), because their nearest fixture uses
    a FULLY frozen background, which the exact test also refuses. A background one
    part in 1e12 from constant is the input that separates the two rules: its
    per-column spread is not exactly zero, so the exact test lets the probe run,
    while ``sigma = background.std(axis=0) + 1e-9`` still collapses the cloud onto
    x and the gap is 0.0 for any model.
    """
    row = np.array([1.12573022, 0.86789514, 1.64042265])
    frozen = np.tile(row, (50, 1))
    jittered = frozen + np.tile(np.array([1e-12, -1e-12, 1e-12]), (50, 1)) * np.arange(50).reshape(
        -1, 1
    )
    # Not exactly constant: the exact test would call this perturbable.
    assert np.all(np.ptp(jittered, axis=0) > 0.0), np.ptp(jittered, axis=0)
    assert _background_cannot_be_perturbed(jittered) is not None
    # And the probe refuses it rather than publishing a gap of 0.0 as clean.
    bg, _x, scaffolded, _clean = _probe_world()
    result, messages = _probe(scaffolded, jittered, row, threshold=0.3)
    assert result.flag is None, result
    assert messages
    # Control: the real background is still perturbable.
    assert _background_cannot_be_perturbed(bg) is None
