"""The plotting functions the getting-started page tells a new user to call.

WHY THESE. Half of the library calls shown on the getting-started and
sample-assessment pages had never been verified, and this family is the largest
unverified cluster on that path: `plot_fairness_metrics`, `create_fairness_dashboard`
and their siblings. A first user's first chart should not come from code nobody has
run, and until 2026-09-25 no test called any of them.

WHAT THE MEASUREMENT FOUND, and it is good news that needed recording rather than
fixing. On a report where nothing is comparable (one group), every one of these
charts draws "NOT ASSESSABLE: no metric could be measured" and prints "not measured"
in place of each value. That is the three-state rule done properly on a matplotlib
canvas, and it was entirely unpinned: one refactor from becoming the fabricated
all-clear that `bias_audit_to_svg` was caught printing in this same wave.

`plot_confidence_intervals` and `plot_effect_sizes` REFUSE a report built without
`include_ci=True`, returning None and naming the fix. That is a refusal about the
INPUT, not about the data, and the pair of tests below holds both halves: it must
refuse the report that lacks the data and it must draw the one that has it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vfairness.evaluation.vfairness_metrics import visualization as VZ  # noqa: E402
from vfairness.evaluation.vfairness_metrics.report import (  # noqa: E402
    classification_fairness_report,
)

# Every chart on the documented path that takes a plain report. THREE OF THEM
# RETURN PLOTLY FIGURES, not matplotlib: plot_metrics_radar,
# plot_group_disparity_heatmap and create_fairness_dashboard. The first version of
# this file asserted `fig.axes` and failed on all three, which is a finding about
# the test rather than the library, and worth recording because a reader of this
# family would otherwise assume one backend.
CHARTS = [
    "plot_fairness_metrics",
    "plot_group_comparison",
    "plot_metrics_radar",
    "plot_fairness_report",
    "plot_group_disparity_heatmap",
    "create_fairness_dashboard",
]

# plot_group_comparison is EXCLUDED from the could-not-measure rule, deliberately.
# It plots the observed positive rate PER GROUP, and for a single group that rate is
# a real measurement ("Positive Rate by Group: a, 51.5%"), not a disparity verdict.
# Demanding a refusal there would be the over-correction: it would suppress a number
# that was measured. What it must not do is assert a DISPARITY, which it does not.
RATES_NOT_VERDICTS = {"plot_group_comparison"}

# plot_group_disparity_heatmap RAISES for a single group rather than drawing, which is
# the defect this file found and fixed: it used to draw a 1x1 matrix titled "Pairwise
# Positive Rate Difference" whose only cell read 0.000, the canvas for perfect parity,
# computed from no pair at all. It is handled by its own pair of tests below.
REFUSES_BY_RAISING = {"plot_group_disparity_heatmap"}
CI_CHARTS = ["plot_confidence_intervals", "plot_effect_sizes"]


def _report(*, single_group: bool = False, include_ci: bool = False):
    rng = np.random.default_rng(5)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    pred = y.copy()
    groups = np.array(["a"] * n) if single_group else np.array(["a"] * 100 + ["b"] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return classification_fairness_report(y, pred, groups, include_ci=include_ci)
        except TypeError:
            return classification_fairness_report(y, pred, groups)


def _has_content(obj) -> bool:
    """Does this figure carry anything, whichever backend drew it?"""
    fig = getattr(obj, "figure", obj)
    if getattr(fig, "axes", None):
        return True
    return bool(getattr(fig, "data", None))  # plotly


def _drawn(obj) -> str:
    """Every string the reader can see, across matplotlib axes and plotly traces."""
    fig = getattr(obj, "figure", obj)
    parts: list[str] = []
    for ax in getattr(fig, "axes", []) or []:
        parts += [t.get_text() for t in ax.texts]
        parts.append(ax.get_title())
        parts += [lbl.get_text() for lbl in ax.get_xticklabels()]
        parts += [lbl.get_text() for lbl in ax.get_yticklabels()]
    parts += [t.get_text() for t in getattr(fig, "texts", []) or []]
    if not parts and getattr(fig, "data", None) is not None:
        # plotly: the reader sees the layout title, the annotations and the trace
        # text, and none of that lives on an axes object.
        parts.append(str(getattr(getattr(fig, "layout", None), "title", "") or ""))
        for ann in getattr(getattr(fig, "layout", None), "annotations", []) or []:
            parts.append(str(getattr(ann, "text", "") or ""))
        for trace in fig.data:
            for attr in ("name", "text", "hovertext"):
                val = getattr(trace, attr, None)
                if isinstance(val, str):
                    parts.append(val)
                elif val is not None:
                    parts += [str(v) for v in val]
    return " | ".join(p for p in parts if p and str(p).strip())


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("name", CHARTS)
def test_every_documented_chart_renders_a_real_object(name):
    """B1 in its plainest form: the getting-started page points at these and nothing
    had ever executed them."""
    out = getattr(VZ, name)(_report())
    assert out is not None, f"{name} returned nothing for a measurable report"
    assert _has_content(out), f"{name} produced an empty figure"


@pytest.mark.parametrize(
    "name", [c for c in CHARTS if c not in RATES_NOT_VERDICTS | REFUSES_BY_RAISING]
)
def test_a_chart_over_one_group_says_it_could_not_measure(name):
    """One group means no disparity is comparable. The canvas must say so."""
    text = _drawn(getattr(VZ, name)(_report(single_group=True)))
    if not text.strip():
        pytest.skip(f"{name} draws its content through a path this reader cannot reach")
    assert "NOT ASSESSABLE" in text.upper() or "NOT MEASURED" in text.upper(), (
        f"{name} drew a chart over one group with no could-not-measure state: {text[:200]}"
    )


@pytest.mark.parametrize("name", CHARTS)
def test_control_a_two_group_chart_does_not_cry_could_not_measure(name):
    """The over-correction half. A comparable report must keep its numbers."""
    text = _drawn(getattr(VZ, name)(_report()))
    if not text.strip():
        pytest.skip(f"{name} draws its content through a path this reader cannot reach")
    assert "NOT ASSESSABLE" not in text.upper(), f"{name} refused a two-group report: {text[:200]}"


@pytest.mark.parametrize("name", CI_CHARTS)
def test_a_ci_chart_refuses_a_report_without_intervals_and_names_the_fix(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = getattr(VZ, name)(_report())
    assert out is None, f"{name} drew a chart from a report carrying no interval data"
    messages = " ".join(str(w.message) for w in caught)
    assert "include_ci=True" in messages, (
        f"{name} refused without saying how to get the data: {messages[:160]}"
    )


@pytest.mark.parametrize("name", CI_CHARTS)
def test_control_a_ci_chart_draws_when_the_report_carries_intervals(name):
    """Without this, the refusal above is satisfied by a function that never draws."""
    report = _report(include_ci=True)
    out = getattr(VZ, name)(report)
    if out is None:
        pytest.skip("this build's report does not carry interval data even with include_ci")
    assert _has_content(out), f"{name} drew an empty figure"


def test_saving_the_plots_writes_files_that_are_not_empty(tmp_path):
    """A zero-byte chart is the worst of the three states: it cannot even be read as
    could-not-check. `method_comparison_to_svg([])` used to do exactly that."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        VZ.save_fairness_plots(_report(), output_dir=str(tmp_path), prefix="t", format="png")
    written = sorted(tmp_path.glob("*.png"))
    assert written, "save_fairness_plots wrote no files at all"
    for path in written:
        assert path.stat().st_size > 0, f"{path.name} is a 0-byte chart"


def test_a_chart_that_plots_rates_keeps_them_for_one_group():
    """The other side of RATES_NOT_VERDICTS, so the exemption is not a loophole.

    plot_group_comparison is exempt from the refusal rule because it reports a
    measured per-group rate. That exemption is only defensible if it really does
    report the rate and really does not assert a disparity, so both are asserted.
    """
    text = _drawn(getattr(VZ, "plot_group_comparison")(_report(single_group=True)))
    assert "%" in text, f"the measured rate disappeared: {text[:160]}"
    lowered = text.lower()
    for verdict in ("disparity", "fair", "unfair", "pass", "compliant"):
        assert verdict not in lowered, (
            f"a one-group rate chart asserted the verdict {verdict!r}: {text[:160]}"
        )


def test_a_pairwise_heatmap_refuses_a_report_with_no_pair():
    """The defect this file found. Measured before the fix: a single-group report drew
    a heatmap titled "Pairwise Positive Rate Difference" with one cell reading 0.000
    and no warning, which is a pairwise disparity of zero over zero pairs."""
    with pytest.raises(ValueError) as excinfo:
        VZ.plot_group_disparity_heatmap(_report(single_group=True))
    message = str(excinfo.value)
    assert "two groups" in message, f"the refusal does not say what is missing: {message}"
    assert "0.000" in message or "no pair" in message, (
        "the refusal does not explain why a one-group matrix is misleading"
    )


def test_control_a_pairwise_heatmap_still_draws_a_real_comparison():
    """Without this, the refusal above is satisfied by a function that never draws."""
    fig = VZ.plot_group_disparity_heatmap(_report())
    matrix = fig.data[0].z
    assert len(matrix) == 2, f"a two-group report gave a {len(matrix)}x matrix"
    off_diagonal = [matrix[0][1], matrix[1][0]]
    assert any(abs(v) > 0 for v in off_diagonal), (
        f"the off-diagonal carries the finding and it is all zero: {matrix}"
    )


# ── 2026-09-29: three of these charts published a neutral value, and two of them
# ── died on a report their own producer emits ─────────────────────────────────
#
# Everything below was reached through the public API first and only then reduced
# to the smallest dict that reproduces it. The before-state of each is in the
# docstring of the function it pins.

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer  # noqa: E402


def _analyzer_report(*, single_group=False, regression_no_variance=False, real_gap=False):
    """A report from FairnessAnalyzer, the class the quickstart leads with."""
    if regression_no_variance:
        # Zero within-group prediction variance: the producer writes
        # cohens_d_predictions = nan and calls it "not interpretable".
        y = np.array([10.0] * 60 + [20.0] * 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return FairnessAnalyzer(
                y, y, np.array(["a"] * 60 + ["b"] * 60), task_type="regression"
            ).get_report(include_ci=True)
    if real_gap:
        # Both groups carry both labels and group b is never selected: every
        # interval IS measurable and the parity difference is 0.5 exactly.
        y = np.array([1] * 30 + [0] * 30 + [1] * 30 + [0] * 30)
        pred = np.array([1] * 30 + [0] * 30 + [0] * 30 + [0] * 30)
        groups = np.array(["a"] * 60 + ["b"] * 60)
    else:
        rng = np.random.default_rng(5)
        y = (rng.random(120) < 0.5).astype(int)
        pred = y.copy()
        groups = np.array(["a"] * 120) if single_group else np.array(["a"] * 60 + ["b"] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y, pred, groups).get_report(include_ci=True)


def _gids(ax) -> list[str]:
    out = []
    for artist in list(ax.texts) + list(ax.patches):
        gid = artist.get_gid()
        if gid:
            out.append(gid)
    return out


def test_the_ci_chart_marks_an_unmeasurable_report_instead_of_raising():
    """BEFORE: ValueError: Axis limits cannot be NaN or Inf, from ax.set_xlim, on a
    report built only from the public API (one group, include_ci=True), whose three
    interval fields are all nan with metadata.warning 'Bootstrap failed'."""
    report = _analyzer_report(single_group=True)
    entry = report["metrics_with_ci"]["demographic_parity_difference"]
    assert all(np.isnan(entry[k]) for k in ("point_estimate", "lower_bound", "upper_bound")), entry

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_confidence_intervals(report)
    assert ax is not None, "the chart refused a report its own producer emits"
    text = _drawn(ax)
    assert "not measured" in text, f"an absent interval was not marked: {text[:200]}"
    assert "NOT ASSESSABLE" in text.upper()
    assert "nan" not in text.lower(), f"the reader was shown raw nan: {text[:200]}"
    assert any("not measured" in str(w.message) for w in caught), "it marked but never warned"
    assert "vfairness-could-not-check" in _gids(ax)


def test_an_unmeasured_row_gets_no_acceptable_region():
    """An acceptable region asserts the value was checked and found compliant.
    _acceptable_region's docstring claimed an unmeasured value got none; only the
    threshold and direction legs were real, so a nan value against a 0.10 threshold
    was shaded green under "Acceptable range" on this chart and on the bar chart."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = VZ.plot_confidence_intervals(_analyzer_report(single_group=True))
    assert VZ.ACCEPTABLE_REGION_GID not in _gids(ax), (
        "a row with no interval at all was given an acceptable region"
    )


def test_control_the_ci_chart_still_draws_the_intervals_it_can_measure():
    """OVER-CORRECTION CONTROL, with the number recomputed here: group a is selected
    30/60 and group b 0/60, so the parity difference is 0.5 and its interval is real."""
    report = _analyzer_report(real_gap=True)
    assert report["metrics"]["demographic_parity_difference"] == pytest.approx(0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_confidence_intervals(report)
    text = _drawn(ax)
    assert "0.500" in text, f"a measured interval lost its number: {text[:200]}"
    assert "NOT ASSESSABLE" not in text.upper()
    assert "not measured" not in text
    assert VZ.ACCEPTABLE_REGION_GID in _gids(ax), "the compliant band vanished with the fix"
    assert not [w for w in caught if "not measured" in str(w.message)]


def test_the_effect_size_chart_survives_a_report_its_own_producer_emits():
    """BEFORE: ValueError: max() iterable argument is empty, from
    `max(abs(d) for d in cohens_d if not np.isnan(d))`, on a regression report with
    zero within-group prediction variance, where the producer itself writes
    cohens_d_predictions = nan and the interpretation "not interpretable"."""
    report = _analyzer_report(regression_no_variance=True)
    effects = report["effect_sizes"]["a_vs_b"]
    assert np.isnan(effects["cohens_d_predictions"]), effects
    assert "not interpretable" in effects["interpretation"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_effect_sizes(report)
    assert ax is not None
    text = _drawn(ax)
    assert "not measured" in text, f"the unmeasured pair was not marked: {text[:200]}"
    assert "NOT ASSESSABLE" in text.upper(), "no pair was measurable and the chart did not say so"
    assert any("no measured effect size" in str(w.message) for w in caught)


def test_the_effect_size_chart_reads_a_none_as_unmeasured_not_as_an_error():
    """A key PRESENT holding None is what this library's own JSON writer produces
    for a non-finite number. It raised TypeError: ufunc 'isnan' not supported."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_effect_sizes({"effect_sizes": {"a_vs_b": {"cohens_d_positive_rate": None}}})
    assert ax is not None
    assert "not measured" in _drawn(ax)
    assert any("no measured effect size" in str(w.message) for w in caught)


def test_an_unmeasured_pair_is_marked_beside_a_measured_one():
    """It rendered as a y-tick label with no bar and no mark, on a chart whose zero
    line means "no effect". CONTROL in the same test: the measured pair keeps its
    real number, so this is not a function that refuses everything."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_effect_sizes(
            {
                "effect_sizes": {
                    "a_vs_b": {"cohens_d_positive_rate": 1.4, "interpretation": "large"},
                    "a_vs_c": {"cohens_d_positive_rate": float("nan")},
                }
            }
        )
    text = _drawn(ax)
    assert "d = 1.40" in text, f"the measured pair lost its number: {text}"
    assert "not measured" in text, f"the unmeasured pair is still a blank row: {text}"
    assert "NOT ASSESSABLE" not in text.upper(), "one pair WAS measured"
    assert any("1 of 2 pair(s)" in str(w.message) for w in caught)


def test_an_empty_effect_size_block_is_not_blamed_on_include_ci():
    """A single-group report built WITH include_ci=True carries effect_sizes present
    and empty, and the one old message told that reader to pass include_ci=True."""
    report = _analyzer_report(single_group=True)
    assert report["effect_sizes"] == {}, "the producer changed shape"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = VZ.plot_effect_sizes(report)
    assert out is None
    messages = " ".join(str(w.message) for w in caught)
    assert "no pair of groups was comparable" in messages, messages
    assert "include_ci=True" not in messages, (
        "include_ci WAS True on this report, so naming it misdiagnoses the refusal"
    )


def test_a_group_with_no_recorded_size_is_not_drawn_as_n_zero():
    """BEFORE: `stats.get("size", 0)` on the line under one that correctly used
    np.nan, so group b was drawn as n=0 beside a 30.0% rate that zero samples
    cannot produce, with no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_group_comparison(
            {
                "group_stats": {
                    "a": {"positive_rate": 0.50, "size": 200},
                    "b": {"positive_rate": 0.30},
                }
            }
        )
    text = _drawn(ax)
    assert "n=200" in text, "the recorded size was lost"
    assert "n=0" not in text, f"an unrecorded sample count was drawn as zero: {text}"
    assert f"n {VZ.NOT_MEASURED_TICK}" in text
    assert "30.0%" in text, "the measured rate was suppressed"
    assert any("no recorded sample size" in str(w.message) for w in caught)


def test_an_average_over_one_group_says_how_many_it_covered():
    """Reached through the library's own producer: get_group_metrics reports tpr as
    nan for a group with no positive labels (tp+fn == 0). The chart then drew both
    groups on the axis, ONE bar, no warning, and "Avg: 0.667" spanning both, which
    was group a alone because np.nanmean shrank its own denominator."""
    from vfairness.evaluation.vfairness_metrics.classification import get_group_metrics

    y = np.array([1] * 60 + [0] * 60 + [0] * 120)
    pred = np.array([1] * 40 + [0] * 20 + [0] * 60 + [1] * 30 + [0] * 90)
    groups = np.array(["a"] * 120 + ["b"] * 120)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = get_group_metrics(y, pred, groups)
    assert np.isnan(stats["b"]["tpr"]), "the producer no longer refuses this group"
    assert stats["a"]["tpr"] == pytest.approx(40 / 60), stats["a"]["tpr"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_group_comparison({"group_stats": stats}, metric="tpr")
    text = _drawn(ax)
    assert "0.667" in text, "the one measured rate was lost"
    assert "1 of 2 groups" in text, f"the average does not say what it averaged over: {text}"
    assert "not measured" in text, f"the group with no bar is unmarked: {text}"
    assert any("no measured tpr" in str(w.message) for w in caught)


def test_control_a_fully_measured_group_chart_keeps_its_plain_average():
    """OVER-CORRECTION CONTROL. When every group was measured, nothing is marked and
    the average carries no group count, recomputed here rather than read off."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = VZ.plot_group_comparison(
            {
                "group_stats": {
                    "a": {"positive_rate": 0.50, "size": 200},
                    "b": {"positive_rate": 0.30, "size": 100},
                }
            }
        )
    text = _drawn(ax)
    assert "Avg: 40.0%" in text, f"the real average of 0.50 and 0.30 is 0.40: {text}"
    assert "of 2 groups" not in text, "a complete average claimed to be partial"
    assert "not measured" not in text
    assert "n=200" in text and "n=100" in text
    assert not caught, f"a fully measured chart warned: {[str(w.message) for w in caught]}"


def test_the_dashboard_twins_of_both_charts_carry_the_same_fix():
    """create_fairness_dashboard has plotly twins of the two charts above, and they
    were worse: _add_groups_panel defaulted a MISSING METRIC to 0 and drew it as a
    measured "0.0%" bar, and _add_effects_panel defaulted a missing Cohen's d to 0
    and drew "d = 0.00", the canvas for a measured absence of effect."""
    report = {
        "metrics": {"demographic_parity_difference": 0.2},
        "thresholds_used": {"demographic_parity_difference": 0.1},
        "group_stats": {"a": {"positive_rate": 0.5, "size": 100}, "b": {"size": 100}},
        "effect_sizes": {"a_vs_b": {"interpretation": "negligible"}},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = VZ.create_fairness_dashboard(report)
    text = _drawn(fig)
    # Compared label BY LABEL, never as a substring: "0.0%" is inside "50.0%", and a
    # substring test here would accuse the measured rate of being the fabricated one.
    labels = [part.strip() for part in text.split("|")]
    assert "50.0%" in labels, f"the measured rate was lost: {labels}"
    assert "0.0%" not in labels, f"a missing metric was drawn as a measured zero rate: {labels}"
    assert "d = 0.00" not in labels, f"a missing effect size was drawn as d = 0.00: {labels}"
    assert text.count("not measured") >= 2, f"only one of the two panels was fixed: {text[:300]}"
    assert "1 of 2 groups" in text, "the panel average does not say what it covered"
    messages = " ".join(str(w.message) for w in caught)
    assert "no measured positive_rate" in messages
    assert "no measured effect size" in messages


def test_a_metric_holding_none_does_not_kill_any_of_the_three_charts():
    """The dtype sibling across the family: nan, None and an absent key must land in
    the same state. matplotlib cannot draw a None, so ax.bar/barh raised TypeError
    for plot_fairness_metrics, plot_group_comparison and plot_effect_sizes alike."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bars = VZ.plot_fairness_metrics(
            {
                "metrics": {
                    "demographic_parity_difference": None,
                    "equalized_odds_difference": 0.2,
                },
                "thresholds_used": {
                    "demographic_parity_difference": 0.1,
                    "equalized_odds_difference": 0.1,
                },
            }
        )
        groups = VZ.plot_group_comparison(
            {
                "group_stats": {
                    "a": {"positive_rate": None, "size": 10},
                    "b": {"positive_rate": 0.3, "size": 20},
                }
            }
        )
    for name, ax in (("plot_fairness_metrics", bars), ("plot_group_comparison", groups)):
        text = _drawn(ax)
        assert "not measured" in text, f"{name} drew a None without marking it: {text[:200]}"
    assert "0.200" in _drawn(bars), "the measured metric beside the None was lost"
    assert "30.0%" in _drawn(groups), "the measured rate beside the None was lost"
