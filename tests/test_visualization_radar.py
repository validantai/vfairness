"""Fairness-axis semantics of the interactive Plotly radar (plot_metrics_radar).

The radar plots each metric as a fairness score: 1.0 (fully fair) at the outer
rim, 0.0 (unfair) at the centre, with every threshold mapped to a single 0.5
reference ring. This matches radar_chart_to_svg, so the interactive and SVG
radars read the same way (fair outward, failures caved inward).
"""

import pytest

pytest.importorskip("plotly")

from vfairness.evaluation.vfairness_metrics.visualization import plot_metrics_radar


def _report():
    return {
        "metrics": {
            "demographic_parity_difference": 0.02,  # fair -> outward
            "equalized_odds_difference": 0.30,  # 2x threshold -> centre (fail)
        },
        "thresholds_used": {
            "demographic_parity_difference": 0.10,
            "equalized_odds_difference": 0.15,
        },
    }


def test_normalized_radar_plots_fairness_outward():
    fig = plot_metrics_radar(_report(), normalize=True)

    # trace 0 = pass-threshold ring, trace 1 = metric fairness
    ring = list(fig.data[0].r)
    fairness = list(fig.data[1].r)

    # Every metric's threshold maps to the same 0.5 reference ring.
    assert ring and all(abs(r - 0.5) < 1e-9 for r in ring)
    # The radial axis is the fixed 0..1 fairness axis (fair at the rim).
    assert tuple(fig.layout.polar.radialaxis.range) == (0, 1.0)
    # r is closed (first point repeated); index 0 = fair, index 1 = failing.
    assert fairness[0] > 0.5, "fair metric should sit outside the pass ring"
    assert fairness[1] < 0.5, "failing metric should cave inside the pass ring"
    # Legend wording matches the SVG radar.
    assert fig.data[0].name == "Pass threshold"
    assert fig.data[1].name == "Metric fairness"
