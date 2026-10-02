"""G07 grading pins: vfairness.preprocessing.feature_engineering.

Five defects were measured here on 2026-09-30, all of one shape: a value that
could not be measured rendered as the mark a reader takes for an all-clear.
Every one has its healthy control beside it, because a guard that refuses
everything passes every refusal test and destroys the unit.

1. AN UNMEASURED PROXY DRAWN AS A ZERO BAR. ``plot_proxy_risk_chart`` sorted by
   ``abs(x.correlation)`` and drew ``abs(nan)``, and matplotlib DRAWS NOTHING
   for a NaN bar. Measured on income=0.85 CRITICAL, zipcode=nan NEGLIGIBLE,
   age=0.12 LOW: widths [0.85, nan, 0.12], so zipcode arrived as a y-tick label
   beside empty space at the left end of an axis labelled "Absolute
   Correlation", which is where a measured 0.00 sits; its annotation was the
   string 'nan' placed at x=nan, so it was not rendered either; zero warnings.

2. AN UNMEASURED PROXY COUNTED INTO A RISK BAND. ``plot_risk_distribution``
   counted every row into its ``risk_level``, and an ungraded row's band is the
   grader's fall-through NEGLIGIBLE. Same three rows: "NEGLIGIBLE (1) 33.3%"
   over a centre reading "3 features", the string 'not_assessed' nowhere in the
   figure, no warning.

3. AN UNMEASURED CELL BLANK ON A HEATMAP. ``plot_correlation_heatmap`` over a
   matrix whose every cell is NaN rendered a titled heatmap, zero outlined
   cells, and a colour bar reading "Absolute Correlation (>= 0.3 outlined)",
   i.e. no pair reaches the proxy threshold, silently, while the SAME matrix
   through ``get_high_correlations(0.3)`` returned ``complete=False``,
   ``pairs_not_measured=[...]`` and a loud warning.

4. A BOOL GRADED AS A PERFECT CORRELATION. ``np.isfinite(True)`` is True, and
   the module's two predicates disagreed: ``_headline_association_was_measured``
   accepted a bool that ``_finite_or_nan`` in the same file refuses, and
   rejected a numeric string that it accepts. One row with
   ``correlation=True`` drew a bar of width 1 annotated "1.00" and reached the
   dashboard's pie, bar chart and Top Concerns as GRADED with no NOT GRADED
   block. ``TransformationResult.correlation_reduction`` held the third copy of
   the same rule: ``correlation_before={'race': True}`` /
   ``correlation_after={'race': False}`` published ``{'race': 1.0}``, which
   ``plot_feature_transformation_effect`` prints as "race: 100.0%", the complete
   elimination of a proxy correlation out of two flags; and a numeric string
   raised TypeError there while the chart drew it.

5. PARTIAL LOSS PASSING WHERE TOTAL LOSS IS REFUSED.
   ``SyntheticResampler.get_resampled_data`` refuses when NO feature column is
   numeric. With one of two text-typed (an ordinary ``pd.read_csv`` of a column
   holding a single "N/A"), it balanced 60 rows to 80 with
   ``fit_result.warnings == []``, no metric and no Python warning, while every
   synthetic row's value for that column was a verbatim copy of a real one and
   the neighbour search ran in one dimension instead of two.

The rest of the file executes the containers in this batch to establish that
none of them quietly mints a value.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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

from vfairness._not_assessed import NOT_ASSESSED  # noqa: E402
from vfairness.preprocessing.feature_engineering import visualization as viz  # noqa: E402
from vfairness.preprocessing.feature_engineering.analyzer import (  # noqa: E402
    FeatureAnalysisReport,
    ProxyScreenFindings,
)
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    CorrelationResult,
    CorrelationType,
    FeatureCorrelationMatrix,
    HighCorrelationResult,
    ProxyChainResult,
    ProxyRiskLevel,
    ProxyScreenResult,
    ProxyType,
    ProxyVariableResult,
)
from vfairness.preprocessing.feature_engineering.data_balancing import (  # noqa: E402
    SyntheticResampler,
)
from vfairness.preprocessing.feature_engineering.transformers import (  # noqa: E402
    BaseFeatureTransformer,
    FeatureImportanceResult,
    TransformationResult,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


def _proxy(feature, correlation, level, *, attr="race", evidence=None):
    return ProxyVariableResult(
        feature=feature,
        protected_attribute=attr,
        correlation=correlation,
        correlation_type="pearson",
        risk_level=level,
        proxy_type=ProxyType.UNCLASSIFIED,
        evidence=evidence if evidence is not None else {},
    )


def _text(ax) -> str:
    parts = [t.get_text() for t in ax.texts]
    parts += [lab.get_text() for lab in ax.get_xticklabels() + ax.get_yticklabels()]
    parts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    return " ".join(p for p in parts if p)


def _widths(ax):
    return [p.get_width() for p in ax.patches if hasattr(p, "get_width")]


# === 1 + 2. an unmeasured proxy on the two charts =========================


THREE = [
    ("income", 0.85, ProxyRiskLevel.CRITICAL),
    ("zipcode", float("nan"), ProxyRiskLevel.NEGLIGIBLE),
    ("age", 0.12, ProxyRiskLevel.LOW),
]


@needs_matplotlib
def test_the_risk_chart_draws_no_bar_for_an_unmeasured_proxy_and_labels_it():
    """DEFECT CASE. The bar was already absent; nothing said why."""
    rows = [_proxy(*r) for r in THREE]
    with pytest.warns(UserWarning, match="could not check"):
        ax = viz.plot_proxy_risk_chart(rows)
    widths = _widths(ax)
    assert sum(1 for w in widths if np.isnan(w)) == 1
    # The one row without a measurement must SAY so, because no bar exists to
    # carry the information.
    assert "not measured" in _text(ax)
    assert "nan" not in [t.get_text() for t in ax.texts]
    # Every row still reaches the chart: omitting it is a different claim.
    for feature, _, _ in THREE:
        assert feature in _text(ax)


@needs_matplotlib
def test_the_unmeasured_row_is_never_ranked_among_the_measured_ones():
    """A NaN sort key is not an ordering; the measured ones keep theirs."""
    rows = [_proxy(*r) for r in THREE]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = viz.plot_proxy_risk_chart(rows)
    labels = [lab.get_text().split("\n")[0] for lab in ax.get_yticklabels()]
    measured = [(f, c) for f, c, _ in THREE if not math.isnan(c)]
    expected = [f for f, _ in sorted(measured, key=lambda t: -t[1])]
    assert labels[: len(expected)] == expected
    assert labels[len(expected) :] == ["zipcode"]


@needs_matplotlib
def test_a_chart_where_nothing_was_measured_says_so_on_every_row():
    """The whole-screen case: the old figure showed every feature at zero."""
    rows = [_proxy("income", float("nan"), ProxyRiskLevel.NEGLIGIBLE)] + [
        _proxy("zipcode", float("nan"), ProxyRiskLevel.NEGLIGIBLE)
    ]
    with pytest.warns(UserWarning, match="only 0 of 2"):
        ax = viz.plot_proxy_risk_chart(rows)
    assert all(np.isnan(w) for w in _widths(ax))
    assert [t.get_text() for t in ax.texts] == ["not measured", "not measured"]


@needs_matplotlib
def test_the_risk_distribution_gives_the_unmeasured_rows_their_own_wedge():
    """DEFECT CASE. They were counted into the NEGLIGIBLE band."""
    rows = [_proxy(*r) for r in THREE]
    with pytest.warns(UserWarning, match="could not check"):
        ax = viz.plot_risk_distribution(rows)
    rendered = _text(ax)
    assert NOT_ASSESSED.upper() in rendered
    assert "NEGLIGIBLE" not in rendered, (
        "the unmeasured row was still counted into the grader's fall-through band"
    )
    # The denominator every percentage is taken over is stated, split.
    assert "2 graded" in rendered


@pytest.mark.parametrize("plot", ["plot_proxy_risk_chart", "plot_risk_distribution"])
@needs_matplotlib
def test_the_producers_explicit_not_assessed_status_is_honoured(plot):
    """The second door: a FINITE float beside a status saying it was not assessed.

    Before the fix this drew a full 0.9 bar and a graded wedge.
    """
    rows = [
        _proxy("income", 0.85, ProxyRiskLevel.CRITICAL),
        _proxy(
            "zipcode",
            0.9,
            ProxyRiskLevel.NEGLIGIBLE,
            evidence={"correlation_measures": {"headline_measure_status": NOT_ASSESSED}},
        ),
    ]
    with pytest.warns(UserWarning, match="only 1 of 2"):
        ax = getattr(viz, plot)(rows)
    assert ax is not None
    if plot == "plot_proxy_risk_chart":
        assert sum(1 for w in _widths(ax) if np.isnan(w)) == 1
    else:
        assert NOT_ASSESSED.upper() in _text(ax)


@needs_matplotlib
def test_a_fully_measured_run_of_both_charts_says_nothing():
    """OVER-CORRECTION CONTROL, with the real numbers.

    A partition that classed everything as unmeasured would pass every test
    above and destroy both charts.
    """
    measured = [("zipcode", 0.82, ProxyRiskLevel.HIGH), ("age", 0.12, ProxyRiskLevel.LOW)]
    rows = [_proxy(*r) for r in measured]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bar = viz.plot_proxy_risk_chart(rows)
        pie = viz.plot_risk_distribution(rows)
    assert [round(w, 4) for w in _widths(bar)] == [0.82, 0.12]
    assert "not measured" not in _text(bar)
    assert NOT_ASSESSED.upper() not in _text(pie)
    assert "HIGH" in _text(pie) and "LOW" in _text(pie)
    assert "graded" not in _text(pie), "the split centre is only for a partial run"


# === 3. the heatmap ========================================================


def _matrix(values, features, attrs):
    frame = pd.DataFrame(values, index=features, columns=attrs)
    return FeatureCorrelationMatrix(
        correlations=frame,
        pvalues=pd.DataFrame(
            np.full((len(features), len(attrs)), 0.01), index=features, columns=attrs
        ),
        feature_names=list(features),
        protected_attributes=list(attrs),
        method="pearson",
    )


@needs_matplotlib
def test_the_heatmap_counts_the_cells_it_could_not_measure():
    """DEFECT CASE, and the picture must agree with get_high_correlations."""
    m = _matrix([[0.85, np.nan], [np.nan, 0.10]], ["income", "zipcode"], ["race", "gender"])
    with pytest.warns(UserWarning, match="could not check"):
        ax = viz.plot_correlation_heatmap(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = m.get_high_correlations(viz.CORRELATION_THRESHOLD_MEDIUM)
    n_unmeasured = len(expected.pairs_not_measured)
    assert n_unmeasured == 2
    assert "NOT MEASURED" in ax.get_xlabel()
    assert f"{n_unmeasured} of 4" in ax.get_xlabel(), (
        f"the coverage line and get_high_correlations disagree: {ax.get_xlabel()!r}"
    )
    assert "never a zero" in ax.get_xlabel()


@needs_matplotlib
def test_a_heatmap_over_a_matrix_measured_nowhere_does_not_read_as_an_all_clear():
    m = _matrix([[np.nan], [np.nan]], ["income", "zipcode"], ["race"])
    with pytest.warns(UserWarning, match="only 0 of 2"):
        ax = viz.plot_correlation_heatmap(m)
    assert "2 of 2 cell(s) NOT MEASURED" in ax.get_xlabel()


@needs_matplotlib
def test_a_fully_measured_heatmap_carries_no_coverage_line():
    """OVER-CORRECTION CONTROL. The cells and the threshold outlines are intact."""
    m = _matrix([[0.85, 0.20], [0.40, 0.10]], ["income", "zipcode"], ["race", "gender"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = viz.plot_correlation_heatmap(m)
    assert ax.get_xlabel() == "Protected Attributes"
    outlined = [p for p in ax.patches if not p.get_fill()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = m.get_high_correlations(viz.CORRELATION_THRESHOLD_MEDIUM)
    assert len(outlined) == len(expected) == 2


# === 4. the three copies of "is this a measurement" ========================


@pytest.mark.parametrize(
    "value,measured",
    [
        (0.85, True),
        (np.float32(0.85), True),
        (0.0, True),
        ("0.85", True),  # a serialised measurement, kept on purpose
        (True, False),  # DEFECT CASE: np.isfinite(True) is 1.0
        (False, False),
        (np.bool_(True), False),  # the half that gets missed
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        ("abc", False),
    ],
)
def test_the_headline_predicate_agrees_with_finite_or_nan(value, measured):
    row = _proxy("f", value, ProxyRiskLevel.NEGLIGIBLE)
    assert viz._headline_association_was_measured(row) is measured
    assert (not math.isnan(viz._finite_or_nan(value))) is measured, (
        "the two predicates in this module disagree again"
    )


@needs_matplotlib
def test_a_boolean_correlation_is_not_drawn_as_a_perfect_proxy():
    """DEFECT CASE. This drew a bar of width 1 annotated '1.00'."""
    with pytest.warns(UserWarning, match="only 0 of 1"):
        ax = viz.plot_proxy_risk_chart([_proxy("f", True, ProxyRiskLevel.NEGLIGIBLE)])
    assert all(np.isnan(w) for w in _widths(ax))
    assert [t.get_text() for t in ax.texts] == ["not measured"]


@needs_matplotlib
def test_a_numeric_string_correlation_is_still_drawn():
    """OVER-CORRECTION CONTROL. This used to raise TypeError on abs('0.85')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = viz.plot_proxy_risk_chart([_proxy("f", "0.85", ProxyRiskLevel.CRITICAL)])
    assert [round(w, 4) for w in _widths(ax)] == [0.85]


def _result(before, after):
    return TransformationResult(
        method="m",
        n_features_original=1,
        n_features_transformed=1,
        n_samples=1,
        correlation_before=before,
        correlation_after=after,
    )


def test_correlation_reduction_excludes_a_pair_of_flags():
    """DEFECT CASE. This published {'race': 1.0}, printed as "race: 100.0%"."""
    assert _result({"race": True}, {"race": False}).correlation_reduction == {}
    assert _result({"race": np.bool_(True)}, {"race": np.bool_(False)}).correlation_reduction == {}


def test_correlation_reduction_reads_a_serialised_pair_rather_than_crashing():
    """DEFECT CASE. np.isfinite('0.8') raised, so the property died."""
    got = _result({"race": "0.8"}, {"race": "0.2"}).correlation_reduction
    assert set(got) == {"race"}
    assert got["race"] == pytest.approx((0.8 - 0.2) / 0.8)


def test_correlation_reduction_refuses_to_call_an_introduced_correlation_no_change():
    """DEFECT CASE. before=0.0 after=0.9 reported a 0.0 reduction."""
    with pytest.warns(UserWarning, match="INTRODUCED"):
        assert _result({"race": 0.0}, {"race": 0.9}).correlation_reduction == {}


def test_correlation_reduction_still_reports_the_real_numbers():
    """OVER-CORRECTION CONTROL, derived rather than quoted."""
    before, after = 0.8, 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _result({"race": before, "z": 0.0}, {"race": after, "z": 0.0})
        red = got.correlation_reduction
    assert red["race"] == pytest.approx((before - after) / before)
    assert red["z"] == 0.0, "nothing before and nothing after is a real 0.0"
    # And the established exclusions still hold.
    assert _result({"race": 0.8}, {}).correlation_reduction == {}
    assert _result({"race": float("nan")}, {"race": 0.1}).correlation_reduction == {}


# === 5. the resampler's partial loss ======================================


def _frame(n=60):
    rs = np.random.RandomState(0)
    return (
        pd.DataFrame(
            {
                "g": ["m"] * (2 * n // 3) + ["f"] * (n // 3),
                "f0": rs.normal(size=n),
                "f1": rs.normal(size=n),
            }
        ),
        np.array([0, 1] * (n // 2)),
    )


def _resample(df, y, method="smote"):
    sampler = SyntheticResampler(protected_attributes=["g"], method=method)
    sampler.fit(df, y)
    out = sampler.get_resampled_data(df, y)
    return sampler, out


@pytest.mark.parametrize("method", ["smote", "adasyn", "tomek"])
def test_a_feature_column_of_numbers_stored_as_text_is_disclosed(method):
    """DEFECT CASE, on every method, because the guard is above the dispatch."""
    df, y = _frame()
    df["f1"] = df["f1"].astype(str)
    with pytest.warns(UserWarning, match="numbers stored as TEXT"):
        sampler, (out, _) = _resample(df, y, method)
    joined = " ".join(sampler.fit_result.warnings)
    assert "f1" in joined
    # On the metrics channel too: a warning can be filtered, a number cannot.
    assert sampler.fit_result.fit_metrics["feature_columns_numbers_stored_as_text"] == ["f1"]


def test_one_stray_token_types_the_whole_column_as_text_and_is_disclosed():
    """This is how it arrives: a plain pd.read_csv of a column holding an 'N/A'."""
    df, y = _frame()
    df["f1"] = df["f1"].astype(object)
    df.loc[0, "f1"] = "N/A"
    with pytest.warns(UserWarning, match="numbers stored as TEXT"):
        sampler, _ = _resample(df, y)
    assert sampler.fit_result.fit_metrics["feature_columns_numbers_stored_as_text"] == ["f1"]


def test_the_disclosed_column_really_was_copied_rather_than_interpolated():
    """The claim in the warning, verified against the returned frame."""
    df, y = _frame()
    df["f1"] = df["f1"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, (out, _) = _resample(df, y)
    synthetic = out.iloc[len(df) :]
    assert len(synthetic) > 0
    assert set(synthetic["f1"]) <= set(df["f1"]), "f1 was interpolated after all"
    assert not set(synthetic["f0"]) <= set(df["f0"]), "f0 was NOT interpolated"


@pytest.mark.parametrize("method", ["smote", "tomek"])
def test_a_genuine_category_column_is_not_reported(method):
    """OVER-CORRECTION CONTROL. A warning on every run is a warning on none.

    Copying a category verbatim is the only honest thing to do with it.
    """
    df, y = _frame()
    df["f1"] = ["red", "blue"] * (len(df) // 2)
    sampler, (out, _) = _resample(df, y, method)
    assert sampler.fit_result.warnings == []
    assert "feature_columns_numbers_stored_as_text" not in sampler.fit_result.fit_metrics


def test_a_fully_numeric_resample_still_balances_and_stays_silent():
    """OVER-CORRECTION CONTROL with the real counts."""
    df, y = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler, (out, out_y) = _resample(df, y)
    assert len(out) > len(df)
    assert len(set(out["g"].value_counts().to_dict().values())) == 1
    assert len(out_y) == len(out)
    assert sampler.fit_result.warnings == []


def test_total_loss_is_still_refused_and_the_frame_comes_back_unchanged():
    """STANDING CONTROL for the guard the partial one was modelled on."""
    df, y = _frame()
    df["f0"] = df["f0"].astype(str)
    df["f1"] = df["f1"].astype(str)
    sampler, (out, out_y) = _resample(df, y)
    assert len(out) == len(df)
    joined = " ".join(sampler.fit_result.warnings)
    assert "no synthesis was performed" in joined
    assert "returned unchanged" in joined


# === 6. the transformation charts: standing controls ======================


@needs_matplotlib
def test_an_unmeasured_after_correlation_is_labelled_not_a_zero_bar():
    with pytest.warns(UserWarning, match="no after-transformation correlation"):
        ax = viz.plot_transformation_comparison({"race": 0.8}, {})
    heights = [p.get_height() for p in ax.patches if hasattr(p, "get_height")]
    assert any(np.isnan(h) for h in heights), "an unmeasured 'After' was drawn as a bar"
    assert "not measured" in _text(ax)


@needs_matplotlib
def test_no_attribute_at_all_is_not_a_removed_correlation():
    with pytest.warns(UserWarning, match="no attribute was compared"):
        ax = viz.plot_transformation_comparison({}, {})
    assert "No attribute was compared" in _text(ax)


@needs_matplotlib
def test_the_transformation_effect_panel_says_when_there_is_nothing_to_compare():
    fig = viz.plot_feature_transformation_effect(_result({}, {}))
    rendered = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "not measured" in rendered


@needs_matplotlib
def test_the_transformation_effect_panel_prints_the_real_reduction():
    """OVER-CORRECTION CONTROL: the percentage is derived, not quoted."""
    before, after = 0.8, 0.2
    result = _result({"race": before}, {"race": after})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = viz.plot_feature_transformation_effect(result)
    rendered = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert f"{(before - after) / before:.1%}" in rendered


# === 7. the containers: nothing minted ====================================


def test_correlation_result_keeps_an_unmeasured_correlation_unmeasured():
    r = CorrelationResult(
        feature="income",
        protected_attribute="race",
        correlation=float("nan"),
        correlation_type=CorrelationType.PEARSON,
    )
    d = r.to_dict()
    assert math.isnan(d["correlation"]), "a NaN correlation was coerced on the way out"
    # The optional statistics default to None, NOT to 0.0 / 1.0, which are the
    # readings "no association" and "not significant".
    assert d["pvalue"] is None and d["mutual_information"] is None and d["cramers_v"] is None
    assert d["sample_size"] == 0
    assert d["correlation_type"] == "pearson"


def test_proxy_variable_result_to_dict_is_json_safe_and_mints_nothing():
    row = _proxy("zipcode", float("nan"), ProxyRiskLevel.NEGLIGIBLE)
    d = row.to_dict()
    assert math.isnan(d["correlation"])
    assert d["risk_level"] == "negligible" and d["proxy_type"] == "unclassified"
    assert d["affected_groups"] == [] and d["recommendations"] == [] and d["evidence"] == {}
    assert d["pvalue"] is None and d["cramers_v"] is None
    json.dumps(d)  # must not raise
    # The mutable defaults must not be shared between instances.
    other = _proxy("a", 0.5, ProxyRiskLevel.HIGH)
    other.affected_groups.append("X")
    assert row.affected_groups == []


@pytest.mark.parametrize(
    "kwargs,complete",
    [
        ({}, True),
        ({"pairs_not_measured": [("a", "b")]}, False),
        ({"not_compared_reason": "threshold cannot fire"}, False),
    ],
)
def test_high_correlation_result_is_a_list_that_also_carries_its_coverage(kwargs, complete):
    got = HighCorrelationResult([], **kwargs)
    assert isinstance(got, list) and len(got) == 0
    assert got.complete is complete, (
        "an empty selection read complete; complete coverage of nothing is the "
        "reassuring half of a two-state answer"
    )


@pytest.mark.parametrize(
    "kwargs,complete",
    [
        ({}, True),
        ({"screens_not_run": [{"feature": "f"}]}, False),
        ({"not_screened_reason": "no pair requested"}, False),
    ],
)
def test_proxy_screen_result_is_a_list_that_also_carries_its_coverage(kwargs, complete):
    got = ProxyScreenResult([], **kwargs)
    assert isinstance(got, list)
    assert got.complete is complete


@pytest.mark.parametrize(
    "kwargs,complete",
    [
        ({}, True),
        ({"screens_not_run": [{}]}, False),
        ({"ungraded_screens": [_proxy("f", float("nan"), ProxyRiskLevel.NEGLIGIBLE)]}, False),
        ({"not_screened_reason": "no pair requested"}, False),
    ],
)
def test_proxy_screen_findings_keeps_every_term_its_base_class_has(kwargs, complete):
    """The copy-constructor defect: this override rebuilt the predicate field by
    field and dropped the term the base class had just gained."""
    got = ProxyScreenFindings([], **kwargs)
    assert isinstance(got, ProxyScreenResult) and isinstance(got, list)
    assert got.complete is complete


@pytest.mark.parametrize(
    "kwargs,complete,phrase",
    [
        ({}, True, ""),
        ({"pairs_not_computed": [("a", "b", "too few rows")]}, False, "COULD NOT CHECK (partial)"),
        ({"protected_attribute_present": False}, False, "COULD NOT CHECK"),
        ({"depths_not_searched": [3]}, False, "not implemented"),
    ],
)
def test_proxy_chain_result_carries_its_coverage_and_a_sentence(kwargs, complete, phrase):
    got = ProxyChainResult([], **kwargs)
    assert isinstance(got, list)
    assert got.complete is complete
    assert phrase in got.coverage_note
    if complete:
        assert got.coverage_note == "", "a complete search must not carry a caveat"


def test_proxy_chain_result_to_dict_survives_the_serialisation_a_bare_list_does_not():
    got = ProxyChainResult(
        [{"chain": ["zip", "income", "race"]}],
        pairs_not_computed=[("a", "b", "too few rows")],
        depths_not_searched=[3],
    )
    # The defect the method exists for: the list subclass flattens.
    assert json.loads(json.dumps(got)) == [{"chain": ["zip", "income", "race"]}]
    d = got.to_dict()
    assert d["complete"] is False
    assert d["n_pairs_not_computed"] == 1
    assert d["depths_not_searched"] == [3]
    assert "COULD NOT CHECK" in d["coverage_note"]
    assert json.loads(json.dumps(d))["n_chains"] == 1


def test_feature_correlation_matrix_to_dict_keeps_an_unmeasured_cell_unmeasured():
    m = _matrix([[0.8, np.nan]], ["income"], ["race", "gender"])
    d = m.to_dict()
    assert d["correlations"]["race"]["income"] == pytest.approx(0.8)
    assert math.isnan(d["correlations"]["gender"]["income"]), (
        "an unmeasured cell was coerced to a number by to_dict"
    )
    assert d["feature_names"] == ["income"]
    assert d["protected_attributes"] == ["race", "gender"]
    assert d["method"] == "pearson"


@pytest.mark.parametrize(
    "kwargs,complete,not_assessed",
    [
        ({}, True, 0),
        ({"screens_not_run": [{"feature": "f"}]}, False, 1),
        (
            {"ungraded_proxy_screens": [_proxy("f", float("nan"), ProxyRiskLevel.NEGLIGIBLE)]},
            False,
            1,
        ),
        ({"not_screened_reason": "no pair requested"}, False, 0),
    ],
)
def test_the_report_publishes_its_coverage_and_a_third_risk_state(kwargs, complete, not_assessed):
    report = FeatureAnalysisReport(
        n_features=0,
        n_protected_attributes=0,
        n_samples=0,
        proxy_variables=[],
        correlation_matrix=None,
        high_risk_features=[],
        recommendations=[],
        **kwargs,
    )
    assert report.proxy_screen_complete is complete
    summary = report.risk_summary
    assert NOT_ASSESSED in summary, "the third state must always be present in the dict"
    assert summary[NOT_ASSESSED] == not_assessed
    # And the coverage crosses the serialisation boundary a reader consumes.
    d = report.to_dict()
    assert d["proxy_screen_complete"] is complete
    assert "screens_not_run" in d and "ungraded_proxy_screens" in d
    assert json.loads(report.to_json())["proxy_screen_complete"] is complete


def test_feature_importance_result_defaults_nothing_and_round_trips():
    r = FeatureImportanceResult(
        feature="income",
        predictive_importance=float("nan"),
        fairness_impact=float("nan"),
        correlation_with_protected={},
        recommendation="keep",
        rationale="",
    )
    d = r.to_dict()
    assert math.isnan(d["predictive_importance"]) and math.isnan(d["fairness_impact"])
    assert d["correlation_with_protected"] == {}
    assert set(d) == {
        "feature",
        "predictive_importance",
        "fairness_impact",
        "correlation_with_protected",
        "recommendation",
        "rationale",
    }


def test_the_base_transformer_cannot_be_instantiated_and_refuses_an_unfitted_transform():
    assert BaseFeatureTransformer.__abstractmethods__ == frozenset({"fit", "transform"})
    with pytest.raises(TypeError, match="abstract"):
        BaseFeatureTransformer()

    class Concrete(BaseFeatureTransformer):
        def fit(self, X, y=None, protected_attributes=None):
            self.is_fitted = True
            return self

        def transform(self, X):
            self._check_is_fitted()
            return X

    fresh = Concrete(["g"])
    assert fresh.is_fitted is False
    assert fresh.fit_result is None, "an unfitted transformer must hold no result"
    with pytest.raises(Exception):
        fresh.transform(pd.DataFrame({"g": ["a"], "f": [1.0]}))
    df = pd.DataFrame({"g": ["a", "b"], "f": [1.0, 2.0]})
    assert fresh.fit(df).transform(df) is df
    # fit_transform must go through fit, not silently pass data through.
    assert Concrete(["g"]).fit_transform(df) is not None
