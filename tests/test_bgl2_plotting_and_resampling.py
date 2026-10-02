"""The charts and the resamplers nothing had executed.

A chart is the surface where an unmeasured value does the most damage, because a
reader takes in a heatmap or a bar in one glance and never sees the dataframe behind
it. A cell drawn as 0.00 says "measured, no relationship". The same cell left blank
says "not measured". Those are different claims and they must look different.

These units had no assertion behind them: three correlation charts, three
calibration charts, the six lazy chart wrappers exported at the top level of
``vfairness`` (each a separate function with its own body, not a re-export), and the
two resampling transforms.

WHAT WAS CONFIRMED, and it is the reason the correlation fix in
``test_bgl2_analysis_surfaces.py`` matters visually. With a zero-variance feature in
the frame, ``plot_feature_correlation_matrix`` renders that feature's entire row and
column as MASKED cells, five of nine on a three-by-three matrix, so the chart shows a
blank stripe rather than a band of confident zeros. Before the fix the underlying
value for the three-level case was a measured 0.0, which would have painted that
stripe in the "no relationship" colour.

The resamplers refuse to transform before ``fit`` with a RuntimeError naming the
missing call, rather than passing the frame through unchanged, which would have
silently produced un-resampled data under a resampler's name.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

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

import vfairness  # noqa: E402
from vfairness.post_processing.calibration import visualization as calibration_plots  # noqa: E402
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer  # noqa: E402
from vfairness.preprocessing.feature_engineering import (
    visualization as correlation_plots,  # noqa: E402
)
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    compute_feature_correlations,
)
from vfairness.preprocessing.feature_engineering.data_balancing import (  # noqa: E402
    CounterfactualAugmenter,
    SyntheticResampler,
)

N = 400


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


def _rng():
    return np.random.default_rng(9)


def _frame(constant: bool = False) -> pd.DataFrame:
    rng = _rng()
    frame = pd.DataFrame(
        {
            "g": rng.choice(["a", "b", "c"], N),
            "age": rng.integers(18, 80, N),
            "inc": rng.random(N) * 1e5,
        }
    )
    return frame.assign(const=1.0) if constant else frame


# ---------------------------------------------------------------------------
# Correlation charts
# ---------------------------------------------------------------------------


@needs_matplotlib
def test_a_feature_that_could_not_be_measured_is_a_hole_in_the_heatmap():
    """Not a zero. A zero is painted as "no relationship" and read as a finding."""
    columns = ["const", "age", "inc"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes = correlation_plots.plot_feature_correlation_matrix(
            _frame(constant=True), feature_columns=columns
        )

    meshes = [c for c in axes.collections if hasattr(c, "get_array")]
    assert meshes, "the heatmap drew no mesh at all"
    grid = np.ma.asarray(meshes[0].get_array())
    assert np.ma.count_masked(grid) > 0, (
        "every cell carries a number, including the constant feature whose "
        "correlation is undefined; that stripe would read as a measured zero"
    )

    labels = [t.get_text() for t in axes.get_xticklabels()]
    assert "const" in labels, (
        "the unmeasurable feature was dropped from the axis entirely, so a reader "
        "cannot tell it was considered and could not be measured"
    )


@needs_matplotlib
def test_control_a_frame_without_constants_has_no_holes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes = correlation_plots.plot_feature_correlation_matrix(
            _frame(), feature_columns=["age", "inc"]
        )
    grid = np.ma.asarray([c for c in axes.collections if hasattr(c, "get_array")][0].get_array())
    assert np.ma.count_masked(grid) == 0


@needs_matplotlib
def test_the_correlation_bars_chart_draws_one_bar_per_feature():
    frame = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(
            frame, protected_attributes=["g"], feature_columns=["age", "inc"]
        )
        axes = correlation_plots.plot_correlation_bars(matrix, "g")

    assert axes is not None
    assert len(axes.patches) >= 2, "a feature in the matrix got no bar"


@needs_matplotlib
def test_no_proxy_chains_draws_no_chart_and_says_so():
    """An empty bar chart titled "Indirect Proxy Chains" would read as a clean
    result. The function declines to draw one and warns instead."""
    with pytest.warns(UserWarning, match="No proxy chains"):
        assert correlation_plots.plot_proxy_chains([]) is None


@needs_matplotlib
def test_proxy_chains_are_drawn_when_there_are_some():
    chains = [
        {"chain": ["zip", "income", "race"], "indirect_correlation": 0.41},
        {"chain": ["name", "ethnicity"], "indirect_correlation": 0.22},
    ]
    axes = correlation_plots.plot_proxy_chains(chains)
    assert axes is not None
    assert len(axes.patches) == len(chains)


# ---------------------------------------------------------------------------
# Calibration charts, including the aliases re-exported at the top level
# ---------------------------------------------------------------------------


def _calibration_inputs():
    rng = np.random.default_rng(6)
    n = 600
    groups = rng.choice(["a", "b"], n)
    probabilities = np.clip(
        np.where(groups == "a", rng.random(n) * 0.6, rng.random(n) * 0.6 + 0.3), 0.01, 0.99
    )
    return rng.binomial(1, probabilities), probabilities, groups


def _analyzer():
    y_true, y_prob, groups = _calibration_inputs()
    return CalibrationAnalyzer(y_true, y_prob, groups)


@needs_matplotlib
def test_the_disparity_chart_renders_from_a_real_disparity_result():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = calibration_plots.plot_calibration_disparity(_analyzer().analyze_disparity())
    assert figure is not None
    assert figure.axes, "the figure carries no axes"


@needs_matplotlib
def test_the_tradeoff_chart_renders_from_a_real_tradeoff_result():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes = calibration_plots.plot_tradeoff_curve(_analyzer().analyze_tradeoffs())
    assert axes is not None


@needs_matplotlib
def test_the_comparison_chart_draws_a_curve_per_named_method():
    y_true, y_prob, _groups = _calibration_inputs()
    rng = np.random.default_rng(2)
    results = {
        "uncalibrated": (y_true, y_prob),
        "isotonic": (y_true, np.clip(y_prob + rng.normal(0, 0.05, len(y_prob)), 0.01, 0.99)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes = calibration_plots.plot_calibration_comparison(results, y_true)

    assert axes is not None
    # One line per method, plus the diagonal reference.
    assert len(axes.get_lines()) >= len(results)


@needs_matplotlib
def test_the_group_calibration_chart_renders():
    y_true, y_prob, groups = _calibration_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert calibration_plots.plot_group_calibration(y_true, y_prob, groups) is not None


@needs_matplotlib
def test_the_dashboard_renders_a_multi_panel_figure():
    y_true, y_prob, groups = _calibration_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = calibration_plots.create_calibration_dashboard(y_true, y_prob, groups)
    assert len(figure.axes) > 1, "a dashboard with one panel is not a dashboard"


# The top-level names are NOT re-exports. Each is a thin wrapper in __init__.py that
# imports the module function lazily and forwards *args/**kwargs, so each has its own
# body, its own line numbers, and its own way to go wrong: a wrapper that dropped a
# keyword, or pointed at the wrong module function, would be invisible to every test
# of the function it wraps. So they are executed here, and their forwarding is checked.
LAZY_CHART_WRAPPERS = [
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_calibration_disparity",
    "plot_group_calibration",
    "plot_tradeoff_curve",
    "create_calibration_dashboard",
]


@pytest.mark.parametrize("name", LAZY_CHART_WRAPPERS)
def test_the_top_level_wrapper_forwards_every_argument_untouched(
    name, monkeypatch: pytest.MonkeyPatch
):
    """A wrapper that swallowed a keyword would still return a chart, so identity of
    the ARGUMENTS is the thing to assert."""
    seen: list[tuple] = []

    def _recorder(*args, **kwargs):
        seen.append((args, kwargs))
        return "forwarded"

    monkeypatch.setattr(calibration_plots, name, _recorder)

    result = getattr(vfairness, name)(1, 2, n_bins=7, title="t")

    assert result == "forwarded", f"vfairness.{name} does not call {name} at all"
    assert seen == [((1, 2), {"n_bins": 7, "title": "t"})]


@needs_matplotlib
def test_the_top_level_wrappers_reach_the_real_charts():
    """Control for the test above: with nothing patched, the wrappers must actually
    produce charts. A recorder proves forwarding, not that the target works."""
    y_true, y_prob, groups = _calibration_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert vfairness.plot_group_calibration(y_true, y_prob, groups) is not None
        assert vfairness.create_calibration_dashboard(y_true, y_prob, groups) is not None
        assert (
            vfairness.plot_calibration_comparison({"uncalibrated": (y_true, y_prob)}, y_true)
            is not None
        )


# ---------------------------------------------------------------------------
# Resampling transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [CounterfactualAugmenter, SyntheticResampler])
def test_transform_before_fit_is_refused_rather_than_passing_data_through(cls):
    """Returning the input unchanged would hand back un-resampled data under a
    resampler's name, and the caller would never know."""
    frame = _frame()[["age", "inc", "g"]]
    transformer = cls(protected_attributes=["g"])

    with pytest.raises(RuntimeError, match="not fitted"):
        transformer.transform(frame)


@pytest.mark.parametrize("cls", [CounterfactualAugmenter, SyntheticResampler])
def test_transform_after_fit_returns_a_frame_of_the_same_length(cls):
    """The sklearn contract: transform preserves the row index so it can sit in a
    pipeline. The resampled set is reached through get_resampled_data."""
    rng = _rng()
    frame = _frame()[["age", "inc", "g"]]
    labels = rng.integers(0, 2, len(frame))
    transformer = cls(protected_attributes=["g"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformer.fit(frame, labels)
        out = transformer.transform(frame)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(frame)
    assert "g" not in out.columns, (
        "the protected attribute survived the transform, so a model fitted on this "
        "output would train on it directly"
    )
