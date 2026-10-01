"""A chart may not draw a value it was never given.

WHY A RENDERER NEEDS ITS OWN GATE. Every check upstream of here asks whether a
number is honest. A renderer can be handed a perfectly honest NaN and still draw a
bar at zero, print "0.000" beside a label, rank a group it could not measure, or
put a confident title over an empty frame. The number was right the whole way and
the only surface a reader actually looks at is wrong, so no producer-side test can
see it. Two defects found by running this:

  1. ``plot_group_calibration`` dropped any group with fewer than ten rows on a bare
     ``continue``. No curve, no legend entry, no warning. On 57 rows in group A
     beside 3 in group B, a chart titled "Calibration by Group" showed exactly one
     group, so the question "is calibration equal across groups?" was answered by
     removing the group that could have made it unequal.
  2. ``group_calibration_to_svg``, which shares no code with that chart, required
     only ONE row. On 59 rows in A beside a single row in B it printed
     "B  ECE = 0.010", named B the BEST CALIBRATED GROUP, reported
     "Disparity: 0.244" over "2 groups analyzed", and recommended applying
     group-specific calibration. From one observation. The sample size appeared
     nowhere on the canvas, and its own docstring already promised the opposite.

BOTH HALVES ARE ASSERTED, and that is the point. A renderer that printed "not
measured" unconditionally would pass the first half of this file and fail the
second. So the disclosure has to depend on the data, which is the only version of
it worth having.

WHAT IS READ. Only the contents of ``<text>`` nodes, and for a figure only titles,
labels, annotations and legend entries. Never the whole SVG document: an earlier
version searched it for ``n/?a\\b``, which matches the FONT NAME "Verdana", so
twenty-one renderers read as disclosing when the match was a typeface. Never tick
labels either: an empty matplotlib axes still carries "0.0", so reading those as
values accused four renderers of plotting zeros when they had plotted nothing.
"""

from __future__ import annotations

import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vfairness._not_assessed import MIN_ROWS_PER_GROUP_FOR_CALIBRATION  # noqa: E402

# Phrases a reader would understand as "this was not measured". "n/a" is absent by
# design; see the module docstring.
MARKER = re.compile(
    r"not measured|not assessed|not assessable|no data|not available|unavailable|"
    r"could not|cannot compare|insufficient|no group|nothing to|not computed|"
    r"not enough|not applicable|no result|excluded|no attribute|was compared|"
    r"were compared|no comparison|no sample|too few",
    re.I,
)
TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


def visible_text(svg: str) -> list[str]:
    out = []
    for raw in TEXT_NODE.findall(svg):
        t = TAG.sub("", raw).strip()
        if t:
            out.append(t)
    return out


def figure_text(ax_or_fig) -> list[str]:
    """Titles, axis labels, annotations and legend entries. NOT tick labels."""
    fig = ax_or_fig if hasattr(ax_or_fig, "savefig") else ax_or_fig.get_figure()
    out = []
    for ax in fig.get_axes():
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            out += [t.get_text() for t in leg.get_texts()]
    return [s.strip() for s in out if isinstance(s, str) and s.strip()]


def said(labels: list[str]) -> list[str]:
    return sorted({m.group(0).lower() for t in labels for m in [MARKER.search(t)] if m})


# A protected attribute where the second group is real but too small to measure.
# MIN_ROWS_PER_GROUP_FOR_CALIBRATION is imported rather than written as 10, so
# raising the threshold cannot silently turn this fixture into a measurable case.
_SMALL = MIN_ROWS_PER_GROUP_FOR_CALIBRATION - 1
Y_TRUE = np.array([0, 1] * 30)
Y_PROB = np.linspace(0.01, 0.99, 60)
G_ONE_TOO_SMALL = np.array(["A"] * (60 - _SMALL) + ["B"] * _SMALL)
G_BOTH_FINE = np.array(["A"] * 30 + ["B"] * 30)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- unmeasurable in


def test_svg_group_calibration_names_the_group_it_could_not_measure():
    from vfairness.rendering.adapters_calibration import group_calibration_to_svg

    labels = visible_text(group_calibration_to_svg(Y_TRUE, Y_PROB, G_ONE_TOO_SMALL))
    assert said(labels), (
        f"group B has {_SMALL} rows, too few to calibrate, and the canvas says nothing "
        f"about it: {labels[:20]}"
    )
    joined = " ".join(labels)
    assert "N/A" in joined or "NOT ASSESSABLE" in joined.upper(), (
        "a disparity needs two measured groups, and only one was measured, so the "
        f"disparity must not be a number: {labels[:20]}"
    )
    assert not re.search(r"Best group:\s*B\b", joined), (
        "a group whose calibration error was not measured cannot be the best "
        f"calibrated: {joined[:300]}"
    )
    # The other half of the same property: the guard withdraws the COMPARISON, not
    # the MEASUREMENT. B's own ECE is arithmetically real and stays on the canvas,
    # now beside the sample size that makes it readable. Deleting it was the first,
    # wrong fix, and it broke two existing tests whose subject was exactly this.
    assert re.search(r"n = %d" % _SMALL, joined), (
        f"group B's sample size is what makes its ECE readable, and it is not shown: {labels[:24]}"
    )
    assert "too few to compare" in joined, (
        f"the canvas must say why B was left out of the comparison: {labels[:24]}"
    )


def test_chart_group_calibration_names_the_group_it_could_not_measure():
    from vfairness.post_processing.calibration.visualization import plot_group_calibration

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = plot_group_calibration(Y_TRUE, Y_PROB, G_ONE_TOO_SMALL)
    labels = figure_text(ax)
    assert said(labels), f"the figure never says a group was left out: {labels}"
    assert any("B" in str(w.message) for w in caught), (
        "the caller is not told which group was dropped"
    )


def test_transformation_comparison_does_not_present_an_empty_frame_as_success():
    from vfairness.preprocessing.feature_engineering.visualization import (
        plot_transformation_comparison,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = plot_transformation_comparison({}, {})
    labels = figure_text(ax)
    assert said(labels), (
        "a chart titled 'Correlation Before vs After Transformation' with no bars on "
        f"it reads as 'no correlation remains', which is the best possible news, for a "
        f"comparison never made: {labels}"
    )
    assert caught, "nothing warned the caller that no attribute was compared"


@pytest.mark.parametrize(
    "module,name,args",
    [
        (
            "vfairness.rendering.adapters_calibration",
            "reliability_diagram_to_svg",
            (np.array([]), np.array([])),
        ),
        (
            "vfairness.rendering.adapters_calibration",
            "pareto_frontier_to_svg",
            (np.array([]), np.array([])),
        ),
        ("vfairness.rendering.adapters_fairness", "metrics_bar_chart_to_svg", ({},)),
        ("vfairness.rendering.adapters_fairness", "group_comparison_to_svg", ({},)),
        ("vfairness.rendering.adapters_fairness", "radar_chart_to_svg", ({},)),
        ("vfairness.rendering.adapters_fairness", "effect_sizes_to_svg", ({},)),
        ("vfairness.rendering.adapters_fairness", "confidence_intervals_to_svg", ({},)),
        ("vfairness.rendering.adapters", "fairness_report_to_svg", ({},)),
        ("vfairness.rendering.adapters_post_processing", "fairness_detailed_report_to_svg", ({},)),
        ("vfairness.rendering.adapters_post_processing", "threshold_optimization_to_svg", ({},)),
        ("vfairness.rendering.adapters_workflow", "report_card_to_svg", ()),
        ("vfairness.rendering.adapters_workflow", "hierarchical_gate_to_svg", ()),
        ("vfairness.rendering.adapters_validation", "data_validation_to_svg", ()),
        ("vfairness.rendering.adapters_robustness", "robustness_testing_to_svg", ()),
        ("vfairness.rendering.adapters_ranking", "ranking_fairness_to_svg", ()),
        ("vfairness.rendering.adapters_regression", "regression_fairness_to_svg", ()),
        ("vfairness.rendering.adapters_reporting", "reporting_dashboard_to_svg", ()),
        ("vfairness.rendering.adapters_discovery", "auto_discovery_to_svg", ()),
        ("vfairness.rendering.adapters_experimentation", "experiment_results_to_svg", (None,)),
        ("vfairness.rendering.adapters_experimentation", "power_analysis_to_svg", (None,)),
        ("vfairness.rendering.adapters_experimentation", "causal_decomposition_to_svg", (None,)),
        (
            "vfairness.rendering.adapters_experimentation",
            "experiment_recommendation_to_svg",
            (None,),
        ),
        ("vfairness.rendering.adapters", "cicd_pipeline_to_svg", ()),
    ],
)
def test_a_renderer_given_nothing_says_so_on_the_canvas(module, name, args):
    """Each of these accepts the empty input its own signature advertises."""
    mod = __import__(module, fromlist=[name])
    fn = getattr(mod, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = fn(*args)
    labels = visible_text(svg)
    assert labels, f"{name} rendered no readable text at all"
    assert said(labels), (
        f"{name} was given nothing to measure and put no such statement on the "
        f"canvas, so a reader cannot tell an all-clear from an absence of "
        f"evidence: {labels[:24]}"
    )


# -------------------------------------------------------------- measurable in too
#
# Without these, a renderer that printed "not measured" unconditionally would pass
# everything above. The disclosure has to be a function of the data.


def test_svg_group_calibration_stays_quiet_when_both_groups_are_measurable():
    from vfairness.rendering.adapters_calibration import group_calibration_to_svg

    labels = visible_text(group_calibration_to_svg(Y_TRUE, Y_PROB, G_BOTH_FINE))
    joined = " ".join(labels)
    assert "NOT ASSESSABLE" not in joined.upper(), (
        f"both groups have 30 rows and were measured, so nothing is unassessable: {joined[:300]}"
    )
    assert re.search(r"\b2 groups analyzed\b", joined), (
        f"two measured groups must be reported as two: {joined[:300]}"
    )


def test_chart_group_calibration_stays_quiet_when_both_groups_are_measurable():
    from vfairness.post_processing.calibration.visualization import plot_group_calibration

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = plot_group_calibration(Y_TRUE, Y_PROB, G_BOTH_FINE)
    assert not said(figure_text(ax)), (
        "a chart that says 'not measured' for data it measured is crying wolf, and a "
        "reader who sees it every time stops reading it"
    )
    assert not [w for w in caught if "no calibration curve was drawn" in str(w.message)]


def test_transformation_comparison_stays_quiet_on_real_correlations():
    from vfairness.preprocessing.feature_engineering.visualization import (
        plot_transformation_comparison,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = plot_transformation_comparison({"race": 0.8}, {"race": 0.1})
    labels = figure_text(ax)
    assert not said(labels), f"both values were measured: {labels}"
    assert "0.800" in labels and "0.100" in labels, f"the measured values are not drawn: {labels}"
    assert not caught, f"nothing should warn here: {[str(w.message) for w in caught]}"


def test_reliability_diagram_prints_a_refusal_not_a_zero_for_an_unmeasurable_ece():
    """ECE is the one number this chart writes directly onto the canvas."""
    from vfairness.post_processing.calibration.visualization import plot_reliability_diagram

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        empty = figure_text(plot_reliability_diagram(np.array([]), np.array([])))
        real = figure_text(plot_reliability_diagram(Y_TRUE, Y_PROB))
    assert any("nan" in t.lower() for t in empty if "ECE" in t), (
        f"no samples means no calibration error, and 'ECE = 0.000' would be the "
        f"strongest possible claim of perfect calibration: {empty}"
    )
    assert any(re.search(r"ECE = 0\.\d\d\d", t) for t in real), (
        f"a measurable ECE must still be printed as a number: {real}"
    )


def test_the_two_group_calibration_surfaces_use_the_same_threshold():
    """They drew the same comparison with thresholds of 10 and 1, and disagreed.

    TWO CHECKS, because each one alone is defeated by something different.

    The first reads the COMPARISON, not the module. An earlier version asserted only
    that the name appeared somewhere in the source, and the ``import`` line satisfies
    that, so replacing ``n_group < MIN_ROWS_PER_GROUP_FOR_CALIBRATION`` with
    ``n_group < 10`` left the guard green. It was a guard that could not fail, and
    thirty seconds of sabotage was the only thing that showed it.

    The second is behavioural and does not care how the number is written: the two
    surfaces must AGREE at the boundary. That is the property actually worth having,
    and it keeps holding through a rename. It cannot catch a hardcoded 10 while the
    shared value is still 10, which is precisely why the textual check stays.
    """
    import inspect
    import re as _re

    from vfairness.post_processing.calibration import visualization as chart
    from vfairness.rendering import adapters_calibration as svg

    for mod, pattern in (
        (chart, r"n_group\s*[<>]=?\s*_MIN_SAMPLES_PER_GROUP_CURVE"),
        (svg, r"n_group\s*[<>]=?\s*MIN_ROWS_PER_GROUP_FOR_CALIBRATION"),
    ):
        assert _re.search(pattern, inspect.getsource(mod)), (
            f"{mod.__name__} does not compare the group size against the shared "
            f"minimum by name, so its threshold can drift from the other surface's "
            f"again. If the variable was renamed, update this pattern; if the number "
            f"was written by hand, that is the defect."
        )
    assert chart._MIN_SAMPLES_PER_GROUP_CURVE == MIN_ROWS_PER_GROUP_FOR_CALIBRATION


@pytest.mark.parametrize(
    "n_small,both_measured",
    [(MIN_ROWS_PER_GROUP_FOR_CALIBRATION - 1, False), (MIN_ROWS_PER_GROUP_FOR_CALIBRATION, True)],
)
def test_both_surfaces_agree_at_the_threshold_boundary(n_small, both_measured):
    """One row below the line, and exactly on it. Both surfaces, same answer.

    This is the property the shared constant exists to protect. It is derived from
    the constant rather than written as 9 and 10, so raising the threshold moves the
    fixture with it and a surface left behind on the old number fails here.
    """
    from vfairness.post_processing.calibration.visualization import plot_group_calibration
    from vfairness.rendering.adapters_calibration import group_calibration_to_svg

    n = 60
    y_true = np.array([0, 1] * (n // 2))
    y_prob = np.linspace(0.01, 0.99, n)
    groups = np.array(["A"] * (n - n_small) + ["B"] * n_small)

    svg_labels = " ".join(visible_text(group_calibration_to_svg(y_true, y_prob, groups)))
    svg_measured_both = "NOT ASSESSABLE" not in svg_labels.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = plot_group_calibration(y_true, y_prob, groups)
    chart_dropped = any("no calibration curve was drawn" in str(w.message) for w in caught)
    chart_measured_both = not chart_dropped and not said(figure_text(ax))

    assert svg_measured_both is both_measured, (
        f"the SVG surface {'measured' if svg_measured_both else 'refused'} a group of "
        f"{n_small} rows; expected {'measured' if both_measured else 'refused'}"
    )
    assert chart_measured_both is both_measured, (
        f"the chart surface {'measured' if chart_measured_both else 'refused'} a group "
        f"of {n_small} rows; expected {'measured' if both_measured else 'refused'}"
    )
