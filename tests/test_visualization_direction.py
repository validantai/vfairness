"""Wave 4: the direction bug in chart GEOMETRY, and the family selections that hid it.

Three waves fixed one bug class: a metric's better-direction guessed from its
NAME, so a maximal violation grades as a pass. Waves 1 to 3 found it in a gate
verdict, in a shared helper and in a rendering adapter, all of them decisions
expressed in COLOUR or in TEXT. Acceptance found the fourth instance in the one
place none of those pins could see: the SHAPE of the acceptable region drawn on
a chart.

``plot_fairness_metrics`` drew ONE hardcoded band, ``ax.axhspan(-0.1, 0.1)``,
labelled "Acceptable range" and stretched across EVERY bar on the axis. For the
ratio family that band is not merely wrong, it is inverted:
``demographic_parity_ratio = 0.000`` (the protected group is NEVER selected, the
maximal four-fifths violation) lands INSIDE it and reads as the only compliant
bar on the chart, while a healthy 0.946 sits far outside it. Both metrics are
produced by ``classification_fairness_report`` on ordinary input, so this was
live on real output. The same hardcoded geometry was on the confidence-interval
charts.

Alongside it, two family selections chose which metrics exist by looking for a
token inside the name:

* the radar plotted ``{k: v for k, v in metrics.items() if "difference" in k}``,
  which silently drops every lower-is-better metric whose name carries no
  "difference" token (multicalibration, auroc_parity, net_benefit_parity,
  brier_score, integrated_calibration_index) while the chart reads as complete,
  and whose empty-result fallback then plotted the ratio family on a gap axis,
  putting a 0.000 four-fifths violation on the fully-fair outer rim;
* the bar and CI colours keyed on an ``endswith("_ratio")`` copy of the rule
  that ``_metric_direction`` owns.

Every assertion below is on rendered GEOMETRY (patch extents in data
coordinates, polar radii) or on text the reader actually sees, never on a colour
and never on the source. Three states everywhere: an acceptable region, a
measured breach, or could-not-check with no region at all.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from vfairness.evaluation.vfairness_metrics.visualization import (  # noqa: E402
    plot_confidence_intervals,
    plot_fairness_metrics,
)

# The gid a could-not-check marker carries. It is NOT an acceptable region, so
# the helper below excludes it; every other non-bar patch on the axes is treated
# as a region that tells the reader "this value is fine", which is exactly how a
# reader reads shading behind a bar.
_COULD_NOT_CHECK_GID = "vfairness-could-not-check"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _region_patches(ax):
    """Every patch on ``ax`` that shades an acceptable region.

    Bars (and their shadow copies) live in ``ax.containers`` and are excluded.
    A patch explicitly marked as a could-not-check marker is excluded too: it
    says the opposite of "acceptable". Anything else shading the plot area is
    counted, so a single full-width band is caught whether it was drawn by
    ``axhspan`` or by a rectangle.
    """
    bar_ids = {id(p) for container in ax.containers for p in container}
    out = []
    for patch in ax.patches:
        if id(patch) in bar_ids:
            continue
        if (patch.get_gid() or "").startswith(_COULD_NOT_CHECK_GID):
            continue
        out.append(patch)
    return out


def _extent_in_data(ax, patch):
    """``(x0, x1, y0, y1)`` of ``patch`` in DATA coordinates.

    Goes through the patch's own transform, so a blended axes/data transform
    (what ``axhspan`` and ``axvspan`` use) and a plain data-space rectangle are
    measured the same way: what the reader sees on the axis.
    """
    verts = patch.get_transform().transform_path(patch.get_path()).vertices
    verts = ax.transData.inverted().transform(verts)
    xs, ys = verts[:, 0], verts[:, 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _inside_an_acceptable_region(ax, x, y):
    """True when the point ``(x, y)`` falls inside a shaded acceptable region."""
    for patch in _region_patches(ax):
        x0, x1, y0, y1 = _extent_in_data(ax, patch)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _texts(ax):
    return [t.get_text() for t in ax.texts]


# --- 1. the fourth instance: geometry, on the bar chart ----------------------


_MIXED_REPORT_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "demographic_parity_ratio": 0.80,
}


def _mixed_report(ratio_value):
    """A difference metric and a ratio metric, exactly as a real report has them."""
    return {
        "metrics": {
            "demographic_parity_difference": 0.02,
            "demographic_parity_ratio": ratio_value,
        },
        "thresholds_used": dict(_MIXED_REPORT_THRESHOLDS),
    }


class TestAcceptableRegionIsPerMetricAndDirectionAware:
    def test_a_maximal_ratio_violation_is_not_drawn_inside_an_acceptable_region(self):
        """0.000 on a four-fifths ratio is the worst possible result.

        Under the single hardcoded +/-0.1 band it was the ONLY bar on the chart
        sitting inside the shading, i.e. the chart certified the maximal
        violation and nothing else.
        """
        ax = plot_fairness_metrics(_mixed_report(0.0))
        assert not _inside_an_acceptable_region(ax, x=1, y=0.0), (
            "demographic_parity_ratio = 0.000 (the protected group is never "
            "selected) is drawn inside an acceptable region; the chart reads it "
            "as compliant"
        )

    def test_a_healthy_ratio_is_drawn_inside_its_acceptable_region(self):
        """Over-correction control: 0.946 passes the four-fifths rule easily.

        Removing the band for ratios would also make this test pass, so it is
        asserted in the same class: the fix has to put the region in the RIGHT
        place, not delete it.
        """
        ax = plot_fairness_metrics(_mixed_report(0.946))
        assert _inside_an_acceptable_region(ax, x=1, y=0.946), (
            "a ratio of 0.946 comfortably clears the 0.80 minimum, yet no "
            "acceptable region covers it"
        )

    def test_the_difference_metric_keeps_its_own_symmetric_region(self):
        """The lower-is-better family must not lose its band to the fix."""
        ax = plot_fairness_metrics(_mixed_report(0.946))
        assert _inside_an_acceptable_region(ax, x=0, y=0.02)
        assert not _inside_an_acceptable_region(ax, x=0, y=0.42), (
            "a 0.42 gap against a 0.10 threshold is a measured breach and must "
            "not be inside the acceptable region"
        )

    def test_the_ratio_region_does_not_bleed_onto_the_difference_metric(self):
        """A band belongs to ONE bar. The ratio's minimum is not the gap's."""
        ax = plot_fairness_metrics(_mixed_report(0.946))
        assert not _inside_an_acceptable_region(ax, x=0, y=0.9), (
            "the ratio metric's acceptable region covers the difference "
            "metric's bar, where 0.9 is a nine-fold breach of a 0.10 threshold"
        )

    def test_an_unknown_direction_metric_gets_no_region_and_says_so(self):
        """Could-not-check is the third state: no band, and stated on the chart."""
        report = {
            "metrics": {"some_bespoke_client_metric": 0.42},
            "thresholds_used": {"some_bespoke_client_metric": 0.10},
        }
        ax = plot_fairness_metrics(report)
        assert not _inside_an_acceptable_region(ax, x=0, y=0.42)
        assert not _inside_an_acceptable_region(ax, x=0, y=0.0), (
            "a metric with no known better-direction was given an acceptable "
            "region anyway; the reader cannot tell it was never checked"
        )
        assert any("could not check" in t.lower() for t in _texts(ax)), (
            "the chart must say the metric could not be checked, in words the "
            f"reader sees; texts were {_texts(ax)!r}"
        )

    def test_a_missing_threshold_is_could_not_check_not_a_default_band(self):
        """No threshold in the report means nothing was compared."""
        report = {
            "metrics": {"demographic_parity_difference": 0.02},
            "thresholds_used": {"demographic_parity_difference": float("nan")},
        }
        ax = plot_fairness_metrics(report)
        assert not _inside_an_acceptable_region(ax, x=0, y=0.02)
        assert any("could not check" in t.lower() for t in _texts(ax))


class TestBarColourUsesTheThreeStates:
    def _facecolor(self, report):
        ax = plot_fairness_metrics(report)
        return tuple(round(c, 4) for c in ax.patches[0].get_facecolor()[:3])

    def _palette_rgb(self, key):
        import matplotlib.colors as mc

        from vfairness.evaluation.vfairness_metrics.visualization import _get_palette

        return tuple(round(c, 4) for c in mc.to_rgb(_get_palette("academic")[key]))

    def test_an_unknown_direction_metric_is_neither_green_nor_red(self):
        """It was painted danger red: a violation this run never established."""
        report = {
            "metrics": {"some_bespoke_client_metric": 0.42},
            "thresholds_used": {"some_bespoke_client_metric": 0.10},
        }
        colour = self._facecolor(report)
        assert colour != self._palette_rgb("success")
        assert colour != self._palette_rgb("danger"), (
            "an unchecked metric is painted as a measured failure; three states, never two"
        )
        assert colour == self._palette_rgb("neutral")

    def test_a_measured_breach_is_still_red(self):
        """Negative control: the third state must not swallow real failures."""
        report = {
            "metrics": {"demographic_parity_difference": 0.42},
            "thresholds_used": {"demographic_parity_difference": 0.10},
        }
        assert self._facecolor(report) == self._palette_rgb("danger")

    def test_a_maximal_ratio_violation_is_red(self):
        report = {
            "metrics": {"demographic_parity_ratio": 0.0},
            "thresholds_used": {"demographic_parity_ratio": 0.80},
        }
        assert self._facecolor(report) == self._palette_rgb("danger")


# --- 2. the same geometry on the confidence-interval chart ------------------


def _ci_report(ratio_pe):
    return {
        "metrics_with_ci": {
            "demographic_parity_difference": {
                "point_estimate": 0.02,
                "lower_bound": 0.00,
                "upper_bound": 0.04,
            },
            "demographic_parity_ratio": {
                "point_estimate": ratio_pe,
                "lower_bound": max(0.0, ratio_pe - 0.03),
                "upper_bound": ratio_pe + 0.03,
            },
        },
        "thresholds_used": dict(_MIXED_REPORT_THRESHOLDS),
    }


class TestConfidenceIntervalRegionIsDirectionAware:
    def test_a_maximal_ratio_violation_is_not_inside_the_acceptable_region(self):
        ax = plot_confidence_intervals(_ci_report(0.0))
        # Row 1 is the ratio metric (y positions are the row indices).
        assert not _inside_an_acceptable_region(ax, x=0.0, y=1), (
            "a four-fifths ratio of 0.000 sits inside the acceptable range shading on the CI chart"
        )

    def test_a_healthy_ratio_is_inside_the_acceptable_region(self):
        ax = plot_confidence_intervals(_ci_report(0.946))
        assert _inside_an_acceptable_region(ax, x=0.946, y=1)

    def test_the_difference_row_keeps_its_symmetric_region(self):
        ax = plot_confidence_intervals(_ci_report(0.946))
        assert _inside_an_acceptable_region(ax, x=0.02, y=0)
        assert not _inside_an_acceptable_region(ax, x=0.42, y=0)


# --- 3. the radar's family selection ----------------------------------------


def _radar(report, **kwargs):
    pytest.importorskip("plotly")
    from vfairness.evaluation.vfairness_metrics.visualization import plot_metrics_radar

    return plot_metrics_radar(report, **kwargs)


def _radar_points(fig):
    """``{axis label: radius}`` for the metric trace (the closing point drops out).

    Empty when the figure has no metric trace at all, which is what a radar
    that could place nothing looks like.
    """
    if len(fig.data) < 2:
        return {}
    trace = fig.data[1]
    return dict(zip(list(trace.theta), list(trace.r)))


def _figure_texts(fig):
    texts = [a.text for a in fig.layout.annotations if a.text]
    if fig.layout.title and fig.layout.title.text:
        texts.append(fig.layout.title.text)
    return texts


class TestRadarSelectsByDirectionNotBySubstring:
    def test_a_lower_is_better_metric_without_the_difference_token_is_plotted(self):
        """multicalibration IS a violation magnitude. It was silently dropped.

        ``"difference" in k`` keeps demographic_parity_difference and discards
        multicalibration, auroc_parity, net_benefit_parity, brier_score and
        integrated_calibration_index, and the chart reads as complete.
        """
        report = {
            "metrics": {
                "demographic_parity_difference": 0.02,
                "multicalibration": 0.30,
            },
            "thresholds_used": {
                "demographic_parity_difference": 0.10,
                "multicalibration": 0.03,
            },
        }
        labels = " | ".join(_radar_points(_radar(report)).keys()).lower()
        assert "multicalibration" in labels, (
            "multicalibration is a lower-is-better calibration violation and is "
            f"missing from the radar; axes were {labels!r}"
        )

    def test_a_dropped_metric_is_never_dropped_silently(self):
        """Whatever cannot be placed must be NAMED on the chart."""
        report = {
            "metrics": {
                "demographic_parity_difference": 0.02,
                "some_bespoke_client_metric": 0.42,
            },
            "thresholds_used": {
                "demographic_parity_difference": 0.10,
                "some_bespoke_client_metric": 0.10,
            },
        }
        fig = _radar(report)
        assert "some_bespoke_client_metric" not in _radar_points(fig), (
            "a metric with no known better-direction cannot be placed on a "
            "fairness axis; plotting it asserts a direction nobody resolved"
        )
        stated = " ".join(_figure_texts(fig)).lower()
        assert "some bespoke client metric" in stated or "some_bespoke_client_metric" in stated, (
            f"the excluded metric is not named anywhere on the figure: {stated!r}"
        )
        assert "could not check" in stated

    def test_a_maximal_ratio_violation_is_not_plotted_as_fair(self):
        """The empty-result fallback put the ratio family on the gap axis.

        ``abs(0.0) <= 0.80`` reads as a zero gap, so the worst possible
        four-fifths result was drawn on the fully-fair outer rim.
        """
        report = {
            "metrics": {"disparate_impact_ratio": 0.0},
            "thresholds_used": {"disparate_impact_ratio": 0.80},
        }
        fig = _radar(report)
        points = _radar_points(fig)
        stated = " ".join(_figure_texts(fig)).lower()
        if points:
            assert max(points.values()) < 0.5, (
                "the maximal four-fifths violation is plotted at or beyond the "
                f"pass ring, i.e. as fair: {points}"
            )
        else:
            assert "disparate impact" in stated or "disparate_impact" in stated

    def test_a_healthy_ratio_is_plotted_as_fair(self):
        """Over-correction control for the ratio branch."""
        report = {
            "metrics": {"disparate_impact_ratio": 0.95},
            "thresholds_used": {"disparate_impact_ratio": 0.80},
        }
        points = _radar_points(_radar(report))
        assert points, "a measurable, passing ratio must be on the radar"
        assert min(points.values()) > 0.5

    def test_a_radar_with_nothing_placeable_says_so_instead_of_drawing_one(self):
        """Every metric unresolvable: the figure states it, and plots nothing."""
        report = {
            "metrics": {"some_bespoke_client_metric": 0.42},
            "thresholds_used": {"some_bespoke_client_metric": 0.10},
        }
        fig = _radar(report)
        assert not _radar_points(fig)
        stated = " ".join(_figure_texts(fig)).lower()
        assert "not assessable" in stated
        assert "some bespoke client metric" in stated

    def test_the_un_normalized_radar_states_which_rule_each_axis_is_under(self):
        """Raw values share one radial axis, so the axis has to name its rule.

        Inside the threshold ring is compliant for a gap and a violation for a
        ratio; without the rule on the label the same radius means both.
        """
        report = {
            "metrics": {
                "demographic_parity_difference": 0.02,
                "disparate_impact_ratio": 0.95,
            },
            "thresholds_used": {
                "demographic_parity_difference": 0.10,
                "disparate_impact_ratio": 0.80,
            },
        }
        labels = " | ".join(_radar_points(_radar(report, normalize=False)).keys()).lower()
        assert "max 0.1" in labels
        assert "min 0.8" in labels

    def test_an_unmeasured_metric_is_not_plotted_at_the_fair_rim(self):
        """NaN is not a zero gap."""
        report = {
            "metrics": {
                "demographic_parity_difference": 0.02,
                "equalized_odds_difference": float("nan"),
            },
            "thresholds_used": {
                "demographic_parity_difference": 0.10,
                "equalized_odds_difference": 0.10,
            },
        }
        fig = _radar(report)
        points = _radar_points(fig)
        assert "Equalized Odds" not in points
        radii = [r for r in points.values() if r is not None]
        assert all(np.isfinite(r) for r in radii)
        stated = " ".join(_figure_texts(fig)).lower()
        assert "equalized odds" in stated


# --- 4. the plotly dashboard panel ------------------------------------------


class TestDashboardPanelFailsClosed:
    def _panel(self, report):
        pytest.importorskip("plotly")
        from plotly.subplots import make_subplots

        from vfairness.evaluation.vfairness_metrics.visualization import (
            _add_metrics_panel,
            _get_palette,
        )

        fig = make_subplots(rows=1, cols=1)
        _add_metrics_panel(fig, report, _get_palette("modern"), 1, 1)
        return fig

    def test_an_unknown_direction_metric_gets_no_acceptable_band(self):
        fig = self._panel(
            {
                "metrics": {"some_bespoke_client_metric": 0.42},
                "thresholds_used": {"some_bespoke_client_metric": 0.10},
            }
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert not rects, (
            "the panel drew a +/-threshold acceptable band for a metric whose "
            "better-direction was never resolved"
        )
        stated = " ".join(a.text for a in fig.layout.annotations if a.text).lower()
        assert "could not check" in stated

    def test_a_known_metric_still_gets_its_band(self):
        """Negative control: the fail-closed path must not eat the real bands."""
        fig = self._panel(
            {
                "metrics": {"demographic_parity_difference": 0.02},
                "thresholds_used": {"demographic_parity_difference": 0.10},
            }
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 1
        assert (rects[0].y0, rects[0].y1) == (-0.10, 0.10)
