"""
Visualization Module for Feature Engineering Analysis.

This module provides visualization tools for understanding feature-fairness
relationships, proxy variables, and transformation effects.

Key Visualizations:
    1. Correlation heatmaps between features and protected attributes
    2. Proxy variable risk charts
    3. Before/after transformation comparisons
    4. Intersectional analysis plots
    5. Feature importance for fairness charts

All visualizations support both static (matplotlib) and interactive modes.
"""

import importlib.util
import warnings
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from ..._not_assessed import NOT_ASSESSED, warn_not_assessed
from .correlation import (
    CORRELATION_THRESHOLD_MEDIUM,
    FeatureCorrelationMatrix,
    ProxyVariableResult,
    compute_pearson_correlation_matrix,
)
from .transformers import TransformationResult
from .transformers import finite_or_nan as _finite_or_nan  # noqa: F401 -- public under this name

# matplotlib and seaborn are the optional [viz] extra and are imported LAZILY,
# on the first chart call (see _load_backends), never at `import vfairness`.
# This module used to import pyplot and seaborn at module level, and the package
# imports this module, so `import vfairness` paid for the whole plotting stack
# and inherited its import-time warnings. Measured at the declared matplotlib
# floor (3.6.0) with pyparsing 3.3: matplotlib calls the deprecated
# setParseAction while importing, so `python -W error -c "import vfairness"`
# aborted (tests/test_audit_wave4_llm.py::TestPlaceholderWarningsLazy).
# matplotlib 3.6.0, 3.6.3, 3.7.0, 3.7.1, 3.8.0, 3.9.0 and 3.10.0 all did the
# same and 3.10.7 did not, so the alternative was a floor near 3.10.7 for a
# library that draws nothing on import.
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_SEABORN = importlib.util.find_spec("seaborn") is not None

mpatches: Any = None
plt: Any = None
colormaps: Any = None
LinearSegmentedColormap: Any = None
sns: Any = None


def _load_backends() -> None:
    """Import the plotting stack into this module's globals, once."""
    global mpatches, plt, colormaps, LinearSegmentedColormap, sns
    if plt is not None:
        return
    try:
        import matplotlib.patches as _mpatches
        import matplotlib.pyplot as _plt
        from matplotlib import colormaps as _colormaps
        from matplotlib import colors as _mcolors
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc
    mpatches, plt, colormaps = _mpatches, _plt, _colormaps
    LinearSegmentedColormap = _mcolors.LinearSegmentedColormap
    if HAS_SEABORN:
        try:
            import seaborn as _sns

            sns = _sns
        except ImportError:
            sns = None


# COLOR SCHEMES

RISK_COLORS = {
    "critical": "#D32F2F",  # Red
    "high": "#F57C00",  # Orange
    "medium": "#FBC02D",  # Yellow
    "low": "#4CAF50",  # Green
    "negligible": "#9E9E9E",  # Gray
}

#: Colour for the could-not-check state on these charts. Deliberately NOT
#: ``RISK_COLORS["negligible"]``: negligible is a MEASURED all-clear and this is
#: the absence of a measurement, and sharing one colour is how the two readings
#: became one. Blue-grey, so it reads as "no verdict" rather than as a band on
#: the red-to-green risk scale.
NOT_ASSESSED_COLOR = "#455A64"

FAIRNESS_CMAP_COLORS = [
    (0.0, "#4CAF50"),  # Green for low correlation
    (0.3, "#FBC02D"),  # Yellow for medium
    (0.5, "#F57C00"),  # Orange for high
    (1.0, "#D32F2F"),  # Red for very high
]


def _check_matplotlib():
    """Check that matplotlib is available, and import it on first use."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        )
    _load_backends()


def _create_fairness_cmap():
    """Create a colormap for fairness visualizations."""
    _check_matplotlib()
    colors = ["#4CAF50", "#8BC34A", "#CDDC39", "#FFC107", "#FF9800", "#F44336"]
    return LinearSegmentedColormap.from_list("fairness", colors)


#: Outline colour for heatmap cells at or above the proxy threshold (R6-5).
THRESHOLD_MARKER_COLOR = "#212121"


def _recentered_at_zero(cmap: Any) -> Any:
    """The colormap ``sns.heatmap(center=0, vmin=0, vmax=1)`` used to build.

    seaborn 0.13 recenters with ``Colormap.set_bad``, which matplotlib 3.11
    flags with a PendingDeprecationWarning on every heatmap. The same colours
    are built here with ``with_extremes`` (matplotlib >= 3.4), and seaborn is
    called with ``center=None`` so it does not recenter again. Centred at 0 over
    [0, 1], seaborn keeps the upper half of the map: 256 samples from 0.5 to 1.
    """
    from matplotlib.colors import Colormap, ListedColormap

    base = cmap if isinstance(cmap, Colormap) else colormaps[cmap]
    bad = base(np.ma.masked_invalid([np.nan]))[0]
    extremes: Dict[str, Any] = {"bad": bad}
    under, over = base(-np.inf), base(np.inf)
    if np.any(under != base(0)):
        extremes["under"] = under
    if np.any(over != base(base.N - 1)):
        extremes["over"] = over
    return ListedColormap(base(np.linspace(0.5, 1.0, 256))).with_extremes(**extremes)


def _mark_cells_over_threshold(ax, corr_df, threshold: float, origin: float) -> int:
    """Outline every heatmap cell whose |correlation| reaches ``threshold``.

    R6-5 (2026-09-10): ``plot_correlation_heatmap(threshold_lines=...)`` was
    documented as "Whether to show threshold markers" and read nowhere.
    Measured: True and False rendered byte-identical PNGs (same sha256, on a
    harness where flipping ``annotate`` DID change the bytes), so the audit's
    own "which pairs cross the line" signal was missing from the picture while
    the option said it was there.

    On a heatmap neither axis carries correlation (both are categorical), so
    the honest threshold marker is per CELL, matching
    ``FeatureCorrelationMatrix.get_high_correlations(threshold)`` rather than an
    axis line at a coordinate that means nothing. ``origin`` is -0.5 for
    ``imshow`` (cell centres on integer coordinates) and 0.0 for the seaborn /
    pcolormesh grid (cell ``(i, j)`` spans ``[j, j+1] x [i, i+1]``); getting it
    wrong offsets every box by half a cell, which is why the pin checks the
    rectangle coordinates and not just their count.

    Returns the number of cells marked, so a caller (and the pin) can compare it
    against ``get_high_correlations``.
    """
    _check_matplotlib()
    values = np.abs(corr_df.values)
    marked = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value) or value < threshold:
                continue
            ax.add_patch(
                mpatches.Rectangle(
                    (j + origin, i + origin),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=THRESHOLD_MARKER_COLOR,
                    linewidth=2.0,
                    zorder=5,
                )
            )
            marked += 1
    return marked


# CORRELATION VISUALIZATIONS


def plot_correlation_heatmap(
    correlation_matrix: FeatureCorrelationMatrix,
    *,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn_r",
    annotate: bool = True,
    threshold_lines: bool = True,
    title: str = "Feature-Protected Attribute Correlations",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot correlation heatmap between features and protected attributes.

    Args:
        correlation_matrix: FeatureCorrelationMatrix from analysis
        figsize: Figure size (width, height)
        cmap: Colormap name
        annotate: Whether to show correlation values
        threshold_lines: Whether to show threshold markers. True (the default)
            outlines every cell whose absolute correlation reaches
            CORRELATION_THRESHOLD_MEDIUM (0.3, the same threshold
            FeatureCorrelationMatrix.get_high_correlations defaults to) and
            says so on the colour bar. Inert until R6-5 (2026-09-10).
        title: Plot title
        save_path: Path to save the figure
        ax: Existing matplotlib axes to use

    Returns:
        matplotlib axes object

    Example:
        >>> matrix = analyzer.get_correlation_matrix()
        >>> plot_correlation_heatmap(matrix)
    """
    _check_matplotlib()

    corr_df = correlation_matrix.correlations.astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cbar_label = "Absolute Correlation"
    if threshold_lines:
        cbar_label = f"{cbar_label} (>= {CORRELATION_THRESHOLD_MEDIUM:g} outlined)"

    # Use seaborn if available for nicer heatmap
    if HAS_SEABORN:
        sns.heatmap(
            corr_df.abs(),
            annot=annotate,
            fmt=".2f",
            cmap=_recentered_at_zero(cmap),
            center=None,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": cbar_label},
            linewidths=0.5,
        )
        # pcolormesh grid: cell (i, j) spans [j, j+1] x [i, i+1].
        cell_origin = 0.0
    else:
        im = ax.imshow(corr_df.abs().values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_yticklabels(corr_df.index)
        plt.colorbar(im, ax=ax, label=cbar_label)
        # imshow: cell centres sit ON the integer coordinates.
        cell_origin = -0.5

        if annotate:
            for i in range(len(corr_df.index)):
                for j in range(len(corr_df.columns)):
                    val = corr_df.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{abs(val):.2f}", ha="center", va="center", fontsize=8)

    if threshold_lines:
        # R6-5: the documented markers, drawn. Same threshold the correlation
        # module flags a proxy at, so the picture and get_high_correlations()
        # cannot disagree.
        _mark_cells_over_threshold(ax, corr_df, CORRELATION_THRESHOLD_MEDIUM, cell_origin)

    ax.set_title(title, fontsize=14, fontweight="bold")
    xlabel = "Protected Attributes"

    # THREE STATES, NEVER TWO. G07 2026-09-30.
    #
    # A non-finite cell is never outlined (``_mark_cells_over_threshold`` skips
    # NaN, correctly) and is rendered blank by both branches, so a pair that
    # COULD NOT BE CORRELATED left the picture looking exactly like a pair
    # correlated and found weak. Measured on a 2x1 matrix whose every cell is
    # NaN: a titled heatmap, zero cells outlined, and a colour bar reading
    # "Absolute Correlation (>= 0.3 outlined)", which states that no pair
    # reaches the proxy threshold, with no warning anywhere. The SAME matrix
    # through ``get_high_correlations(0.3)`` returns ``complete=False``,
    # ``pairs_not_measured=[('zipcode','race'), ...]`` and a loud warning: the
    # number knew and the picture, which is the surface a reader looks at, did
    # not. The coverage goes under the axis rather than into a new patch,
    # because the unfilled rectangles on these axes are the threshold markers
    # and a second kind would make the two indistinguishable.
    values = np.abs(corr_df.values.astype(float))
    n_cells = int(values.size)
    n_unmeasured = int(np.count_nonzero(~np.isfinite(values)))
    if n_unmeasured:
        xlabel = f"{xlabel}\n{n_unmeasured} of {n_cells} cell(s) NOT MEASURED (blank, never a zero)"
        warn_not_assessed(
            "plot_correlation_heatmap",
            measured=n_cells - n_unmeasured,
            total=n_cells,
            unit="feature/attribute cell(s) carried a finite correlation",
            requirement="a colour and a threshold outline each need one",
            reporting=f"a blank cell and a coverage line naming {n_unmeasured} of them",
            instead_of=(
                "a blank indistinguishable from a cell measured below the "
                f"{CORRELATION_THRESHOLD_MEDIUM:g} threshold"
            ),
            stacklevel=2,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Features", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_feature_correlation_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    min_periods: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    annotate: bool = False,
    title: str = "Pearson Feature Correlation Matrix",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot a Pearson correlation matrix across numeric features.

    Args:
        df: DataFrame to analyze
        feature_columns: Columns to include (numeric-only; auto-detect if None)
        min_periods: Minimum overlapping observations required per pair
        figsize: Figure size (width, height)
        cmap: Colormap name
        annotate: Whether to show correlation values
        title: Plot title
        save_path: Path to save the figure
        ax: Existing matplotlib axes to use

    Returns:
        matplotlib axes object

    Example:
        >>> ax = plot_feature_correlation_matrix(df)
    """
    _check_matplotlib()

    # return_pvalues defaults to False, so this always returns a DataFrame.
    corr_df = cast(
        pd.DataFrame,
        compute_pearson_correlation_matrix(
            df,
            feature_columns=feature_columns,
            min_periods=min_periods,
        ),
    ).astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            corr_df,
            annot=annotate,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Pearson Correlation"},
            linewidths=0.5,
        )
    else:
        im = ax.imshow(corr_df.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_yticklabels(corr_df.index)
        plt.colorbar(im, ax=ax, label="Pearson Correlation")

        if annotate:
            for i in range(len(corr_df.index)):
                for j in range(len(corr_df.columns)):
                    val = corr_df.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_correlation_bars(
    correlation_matrix: FeatureCorrelationMatrix,
    protected_attribute: str,
    *,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    threshold: float = 0.3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot bar chart of feature correlations with a protected attribute.

    Args:
        correlation_matrix: FeatureCorrelationMatrix from analysis
        protected_attribute: Name of protected attribute to show
        top_n: Number of top features to display
        figsize: Figure size
        threshold: Threshold line to display
        title: Plot title (auto-generated if None)
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object, or ``None`` when not one feature correlation with
        ``protected_attribute`` could be measured, in which case a warning names
        the gap instead of an empty chart being returned. A feature whose
        correlation is NaN keeps a labelled row marked "not measured" and gets no
        bar, because a feature silently absent from the chart reads as a feature
        with no correlation.
    """
    _check_matplotlib()

    if protected_attribute not in correlation_matrix.protected_attributes:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found")

    # BGL5 2026-09-27. `dropna()` DELETED the unmeasured features from the chart.
    # A bar chart titled "Feature Correlations with <attribute>" with a feature
    # simply missing reads as that feature carrying no proxy risk, which is the
    # opposite of "we could not measure it". Measured before, on a 5-row frame
    # (below MIN_SAMPLE_SIZE=10) whose whole gender column was NaN: bars drawn 0,
    # y tick labels [], texts on the axes [], title "Feature Correlations with
    # gender", legend ['Threshold (0.3)'] and NOT ONE warning, i.e. a clean-looking
    # empty chart. On partial coverage (300 rows, income measured at 0.999166, age
    # constant so NaN): bars 1, ylabels ['income'], texts [], no warning, so the
    # feature that was never measured was silently absent. After: the unmeasured
    # features keep a labelled row carrying "not measured (could not check)" and no
    # bar, a warning names them, and a column with nothing measured declines to
    # draw at all, as its sibling plot_proxy_chains already did. The healthy
    # 400-row control still draws its 2 bars with no warning.
    corr_column = correlation_matrix.correlations[protected_attribute].abs()
    corr_series = corr_column.dropna().sort_values(ascending=True).tail(top_n)
    not_measured = [str(f) for f in corr_column.index[corr_column.isna()]]

    if not_measured:
        warnings.warn(
            f"plot_correlation_bars: {len(not_measured)} of {len(corr_column)} "
            f"feature correlation(s) with '{protected_attribute}' could not be "
            f"measured on this data and have NO bar on the chart: "
            f"{not_measured[:10]}"
            f"{' (and more)' if len(not_measured) > 10 else ''}. They are drawn as "
            f"labelled rows marked 'not measured', because a missing bar reads as no "
            f"correlation and this is a could-not-check.",
            UserWarning,
            stacklevel=2,
        )

    if len(corr_series) == 0:
        # Nothing to draw. An axes carrying only a threshold line and a title is
        # the shape a reader mistakes for "no feature correlates with this
        # attribute", so decline, exactly like plot_proxy_chains on empty input.
        warnings.warn(
            f"plot_correlation_bars: not one feature correlation with "
            f"'{protected_attribute}' could be measured "
            f"({len(not_measured)} of {len(corr_column)} feature(s) unmeasured), so "
            f"no chart is drawn. An empty bar chart under this title would read as "
            f"no correlation found.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ["#D32F2F" if v >= threshold else "#4CAF50" for v in corr_series]

    # The unmeasured rows sit at the BOTTOM of the axes (y ascends), where the
    # weakest measured correlation would be, so they are never mistaken for the
    # strongest. They get a tick label and a text, and no bar.
    shown_not_measured = not_measured[:top_n]
    n_gap = len(shown_not_measured)

    ax.barh(range(n_gap, n_gap + len(corr_series)), corr_series.values, color=colors)

    for i, feature_name in enumerate(shown_not_measured):
        ax.text(
            0.02,
            i,
            "not measured (could not check)",
            va="center",
            fontsize=9,
            style="italic",
            color="#616161",
        )
    if len(not_measured) > n_gap:
        ax.text(
            0.02,
            n_gap + len(corr_series) - 0.5,
            f"(and {len(not_measured) - n_gap} more features not measured)",
            va="center",
            fontsize=9,
            style="italic",
            color="#616161",
        )

    ax.set_yticks(range(n_gap + len(corr_series)))
    ax.set_yticklabels(
        [f"{f} (not measured)" for f in shown_not_measured] + list(corr_series.index)
    )
    ax.set_xlabel("Absolute Correlation", fontsize=12)
    ax.set_xlim(0, 1)

    ax.axvline(
        x=threshold, color="#FF5722", linestyle="--", linewidth=2, label=f"Threshold ({threshold})"
    )

    if title is None:
        title = f"Feature Correlations with {protected_attribute}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# PROXY VARIABLE VISUALIZATIONS


def plot_proxy_risk_chart(
    proxy_variables: List[ProxyVariableResult],
    *,
    figsize: Tuple[int, int] = (12, 8),
    max_features: int = 15,
    title: str = "Proxy Variable Risk Assessment",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot chart showing proxy variables by risk level.

    Args:
        proxy_variables: List of ProxyVariableResult from analysis
        figsize: Figure size
        max_features: Maximum number of features to display
        title: Plot title
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object

    Example:
        >>> proxies = analyzer.analyze_proxies()
        >>> plot_proxy_risk_chart(proxies)
    """
    _check_matplotlib()

    if not proxy_variables:
        warnings.warn("No proxy variables to plot")
        return None

    # THREE STATES, NEVER TWO. G07 2026-09-30.
    #
    # A row whose HEADLINE association could not be measured still carries a
    # ``risk_level``, because the grader has a fall-through band, and
    # ``abs(nan)`` is ``nan``: MATPLOTLIB DRAWS NOTHING FOR A NaN BAR. Measured
    # on three rows income=0.85 CRITICAL, zipcode=nan NEGLIGIBLE, age=0.12 LOW:
    # bar widths [0.85, nan, 0.12], so `zipcode` arrived as its y-tick label
    # beside empty space at the left end of an axis labelled "Absolute
    # Correlation", which is where a measured 0.00 also sits. Its annotation was
    # the literal string 'nan' positioned at x=nan, so it was not rendered
    # either, and nothing warned. With EVERY row NaN the figure was a titled,
    # legended risk chart on which no feature had any correlation at all: the
    # best possible news, from a screen that measured nothing.
    #
    # The sort made it worse. The key was ``abs(x.correlation)`` and every
    # comparison against NaN is False, so an unmeasured row landed wherever the
    # sort left it and then competed for the ``max_features`` slots with rows
    # that had real findings.
    #
    # ``_headline_association_was_measured`` is this module's existing predicate
    # for exactly this question, and ``create_analysis_dashboard`` already
    # partitions on it before calling this function and names the ungraded rows
    # in a NOT GRADED block. The two other public entries onto the same rows did
    # not, so the fix is two functions wide.
    measured = [p for p in proxy_variables if _headline_association_was_measured(p)]
    unmeasured = [p for p in proxy_variables if not _headline_association_was_measured(p)]

    ranked = sorted(measured, key=lambda p: _headline_association(p), reverse=True)[:max_features]
    # Unmeasured rows are kept and are NEVER mixed into the ranking: dropping
    # them would be a different claim (that the screen looked and found nothing
    # there), and ordering them by a NaN key is not an ordering. Stable, by name.
    shown_unmeasured = sorted(
        unmeasured, key=lambda p: (str(p.feature), str(p.protected_attribute))
    )[:max_features]
    proxies = ranked + shown_unmeasured

    if unmeasured:
        warn_not_assessed(
            "plot_proxy_risk_chart",
            measured=len(measured),
            total=len(proxy_variables),
            unit="proxy row(s) carried a measured headline association",
            requirement="a bar length needs one",
            reporting=(
                f"no bar and a 'not measured' label for {len(shown_unmeasured)} row(s)"
                + (
                    f" ({len(unmeasured) - len(shown_unmeasured)} further unmeasured row(s) "
                    f"are past max_features={max_features} and are not on the chart)"
                    if len(unmeasured) > len(shown_unmeasured)
                    else ""
                )
            ),
            instead_of="a zero-length bar indistinguishable from a measured 0.00",
            stacklevel=2,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    features = [p.feature for p in proxies]
    # One source for the drawn value and for the state (see _headline_association).
    correlations = [_headline_association(p) for p in proxies]
    colors = [
        RISK_COLORS[p.risk_level.value] if np.isfinite(v) else NOT_ASSESSED_COLOR
        for p, v in zip(proxies, correlations)
    ]
    labels = [f"{p.protected_attribute}" for p in proxies]

    y_pos = range(len(features))
    bars = ax.barh(y_pos, correlations, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{f}\n({label})" for f, label in zip(features, labels)], fontsize=9)

    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        if np.isnan(corr):
            # No bar exists for NaN, so this label is the row's ONLY trace, and
            # it has to say which of the two blanks this is. Same idiom, and the
            # same reason, as _annotate_bar below.
            ax.text(
                0.02,
                i,
                "not measured",
                va="center",
                fontsize=9,
                style="italic",
                color=NOT_ASSESSED_COLOR,
            )
            continue
        ax.text(corr + 0.02, i, f"{corr:.2f}", va="center", fontsize=9)

    ax.set_xlabel("Absolute Correlation", fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=RISK_COLORS[level], label=level.upper())
        for level in ["critical", "high", "medium", "low"]
    ]
    if unmeasured:
        legend_patches.append(mpatches.Patch(color=NOT_ASSESSED_COLOR, label="NOT MEASURED"))
    ax.legend(handles=legend_patches, loc="lower right", title="Risk Level")

    ax.invert_yaxis()  # Highest at top
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_risk_distribution(
    proxy_variables: List[ProxyVariableResult],
    *,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Proxy Variable Risk Distribution",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot pie/donut chart of risk level distribution.

    Args:
        proxy_variables: List of ProxyVariableResult
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object
    """
    _check_matplotlib()

    if not proxy_variables:
        warnings.warn("No proxy variables to plot")
        return None

    # THREE STATES, NEVER TWO. G07 2026-09-30. This counted every row into its
    # ``risk_level`` band, and an ungraded row's band is the grader's
    # fall-through, NEGLIGIBLE, whose action is "Minimal risk. No action
    # needed.". Measured on three rows income=0.85 CRITICAL, zipcode=nan
    # NEGLIGIBLE, age=0.12 LOW: wedges "CRITICAL (1) 33.3%", "LOW (1) 33.3%",
    # "NEGLIGIBLE (1) 33.3%" over a centre reading "3 features", with the string
    # 'not_assessed' nowhere in the figure and no warning. The unmeasured row
    # both inflated the denominator every percentage is taken over and
    # contributed a third of the pie to the most reassuring band.
    #
    # ``FeatureAnalysisReport.risk_summary`` has ALWAYS carried a
    # ``not_assessed`` key for exactly this, and ``create_analysis_dashboard``
    # already partitions the rows before calling this function; this public
    # entry onto the same rows did not.
    graded = [p for p in proxy_variables if _headline_association_was_measured(p)]
    ungraded = [p for p in proxy_variables if not _headline_association_was_measured(p)]

    risk_counts: Dict[str, int] = {}
    for proxy in graded:
        level = proxy.risk_level.value
        risk_counts[level] = risk_counts.get(level, 0) + 1
    if ungraded:
        risk_counts[NOT_ASSESSED] = len(ungraded)
        warn_not_assessed(
            "plot_risk_distribution",
            measured=len(graded),
            total=len(proxy_variables),
            unit="proxy row(s) carried a measured headline association",
            requirement="a risk band needs one",
            reporting=f"a separate '{NOT_ASSESSED.upper()}' wedge of {len(ungraded)} row(s)",
            instead_of="counting them into the risk_level band the grader fell through to",
            stacklevel=2,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    levels = ["critical", "high", "medium", "low", "negligible", NOT_ASSESSED]
    sizes = [risk_counts.get(level, 0) for level in levels]
    colors = [RISK_COLORS.get(level, NOT_ASSESSED_COLOR) for level in levels]
    labels = [f"{level.upper()}\n({risk_counts.get(level, 0)})" for level in levels]

    non_zero = [
        (level, s, c, lb) for level, s, c, lb in zip(levels, sizes, colors, labels) if s > 0
    ]
    if not non_zero:
        return ax

    nz_levels, nz_sizes, nz_colors, nz_labels = zip(*non_zero)

    # ax.pie returns (wedges, texts) or (wedges, texts, autotexts) depending on
    # autopct; nothing below uses them, so they are not unpacked (mypy, 2026-09-09).
    ax.pie(
        nz_sizes,
        labels=nz_labels,
        colors=nz_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white"),
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    total = sum(sizes)
    centre = f"{total}\nfeatures"
    if ungraded:
        # The denominator every percentage above is taken over, split, because
        # "3 features" over a pie one third of which was never measured reads as
        # three graded features.
        centre = f"{total}\nfeatures\n{len(graded)} graded"
    ax.text(0, 0, centre, ha="center", va="center", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# TRANSFORMATION COMPARISON VISUALIZATIONS


# ``_finite_or_nan`` (READINESS-6, 2026-09-10) now lives in ``.transformers``
# and is imported at the top of this module under this same name. It MOVED,
# rather than being copied, because G07 2026-09-30 found
# ``TransformationResult.correlation_reduction`` deciding the identical question
# with ``np.isfinite`` and disagreeing with this rule in both directions, over
# the very two dicts ``plot_transformation_comparison`` below renders. See that
# property. The name is kept here because ``tests/test_readiness6_flags.py``
# imports it from this module.


def _annotate_bar(ax: Any, bar: Any, value: float) -> None:
    """Annotate a before/after bar with its value, or with "not measured" when
    the value is NaN (no bar is drawn for NaN, so the label is the only trace)."""
    x_mid = bar.get_x() + bar.get_width() / 2
    if np.isnan(value):
        ax.annotate(
            "not measured",
            xy=(x_mid, 0),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#9E9E9E",
            rotation=90,
        )
        return
    ax.annotate(
        f"{value:.3f}",
        xy=(x_mid, value),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def plot_transformation_comparison(
    before_correlations: Dict[str, float],
    after_correlations: Dict[str, float],
    *,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Correlation Before vs After Transformation",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot comparison of correlations before and after transformation.

    Args:
        before_correlations: Dict of attribute -> correlation before
        after_correlations: Dict of attribute -> correlation after
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object

    Example:
        >>> result = transformer.fit_result
        >>> plot_transformation_comparison(
        ...     result.correlation_before,
        ...     result.correlation_after
        ... )
    """
    _check_matplotlib()

    attributes = list(before_correlations.keys())
    # Three states, never two (audit-6 lane 2, 2026-09-09). An attribute that is
    # absent from after_correlations (or carries a non-finite value) was never
    # measured after the transformation: FeatureSuppressor.fit and
    # ResidualTransformer.fit used to leave correlation_after empty, and the old
    # `after_correlations.get(a, 0)` then drew a green 0.000 "After" bar, the
    # strongest possible claim that the proxy correlation was eliminated, for a
    # value nobody computed. Measured before the fix: before={'race': 0.8},
    # after={} rendered bar heights [0.8, 0] annotated '0.800' / '0.000' with no
    # warning. Now the unmeasured slot is NaN (matplotlib draws no bar for NaN),
    # its annotation reads "not measured", and the caller is warned by name.
    before = [_finite_or_nan(before_correlations[a]) for a in attributes]
    after = [_finite_or_nan(after_correlations.get(a)) for a in attributes]
    unmeasured_after = [a for a, v in zip(attributes, after) if np.isnan(v)]

    # NO ATTRIBUTES AT ALL is a third case, and it used to fall straight through
    # the per-attribute handling above. With before={} the loop body never runs, so
    # unmeasured_after is empty, nothing warns, and the figure renders a titled,
    # legended, axed chart with zero bars on it. "Correlation Before vs After
    # Transformation" over an empty frame reads as the best possible news, that no
    # correlation remains, when in fact no correlation was ever measured. The
    # annotation below is the only thing standing between those two readings.
    if not attributes:
        warnings.warn(
            "plot_transformation_comparison: before_correlations is empty, so no "
            "attribute was compared and the chart carries no bars. This is NOT "
            "evidence that the transformation removed the correlation.",
            stacklevel=2,
        )
    if unmeasured_after:
        warnings.warn(
            "plot_transformation_comparison: no after-transformation correlation was "
            f"measured for {unmeasured_after}; no 'After' bar is drawn for them "
            "(a blank, never a zero)."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(attributes))
    width = 0.35

    bars1 = ax.bar(x - width / 2, before, width, label="Before", color="#F44336", alpha=0.8)
    bars2 = ax.bar(x + width / 2, after, width, label="After", color="#4CAF50", alpha=0.8)

    ax.set_xlabel("Protected Attribute", fontsize=12)
    ax.set_ylabel("Average Absolute Correlation", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attributes)
    ax.legend()

    if not attributes:
        ax.text(
            0.5,
            0.5,
            "No attribute was compared:\nbefore-transformation correlations are empty",
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="center",
            color="#8A6D3B",
        )

    for bar, value in zip(list(bars1) + list(bars2), before + after):
        _annotate_bar(ax, bar, value)

    finite = [v for v in before + after if not np.isnan(v)]
    top = max(finite) if finite else 0.0
    ax.set_ylim(0, top * 1.2 if top > 0 else 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_feature_transformation_effect(
    transformation_result: TransformationResult,
    *,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Feature Transformation Effect",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot comprehensive view of transformation effects.

    Creates a multi-panel figure showing:
    - Correlation reduction per attribute
    - Features modified/removed
    - Overall statistics

    Args:
        transformation_result: TransformationResult from transformation
        figsize: Figure size
        title: Main title
        save_path: Path to save the figure

    Returns:
        matplotlib figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Correlation comparison
    if transformation_result.correlation_before and transformation_result.correlation_after:
        plot_transformation_comparison(
            transformation_result.correlation_before,
            transformation_result.correlation_after,
            ax=axes[0],
            title="Correlation Reduction",
        )
    else:
        # audit-6 lane 2 (2026-09-09): an unlabelled empty panel reads as "nothing
        # to show"; say that the after-measurement does not exist instead.
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "Correlation before/after not measured\nfor this transformation",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#9E9E9E",
        )

    # Panel 2: Summary statistics
    ax = axes[1]
    ax.axis("off")

    summary_text = [
        f"Method: {transformation_result.method}",
        f"Original features: {transformation_result.n_features_original}",
        f"Transformed features: {transformation_result.n_features_transformed}",
        f"Samples: {transformation_result.n_samples:,}",
        "",
    ]

    if transformation_result.features_removed:
        summary_text.append(f"Features removed: {len(transformation_result.features_removed)}")
        for f in transformation_result.features_removed[:5]:
            summary_text.append(f"  - {f}")
        if len(transformation_result.features_removed) > 5:
            summary_text.append(f"  ... and {len(transformation_result.features_removed) - 5} more")

    if transformation_result.correlation_reduction:
        summary_text.append("")
        summary_text.append("Correlation reduction:")
        for attr, reduction in transformation_result.correlation_reduction.items():
            summary_text.append(f"  {attr}: {reduction:.1%}")

    ax.text(
        0.1,
        0.9,
        "\n".join(summary_text),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# INTERSECTIONAL VISUALIZATIONS


def plot_intersectional_heatmap(
    intersectional_results: Dict[str, Any],
    feature: str,
    *,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot heatmap of feature values across intersectional groups.

    Args:
        intersectional_results: Results from analyze_intersectional_correlations.
            Its ``groups_not_assessed`` and ``coverage`` are read, so pass the
            result itself rather than only its ``within_group_correlations``.
        feature: Feature to visualize
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object, or ``None`` when no group carries a mean for
        ``feature``. An intersectional group that exists in the data and was never
        examined keeps a labelled slot marked "not examined" and gets no bar; when
        NO group was examined the warning says so rather than blaming the feature
        name.
    """
    _check_matplotlib()

    within_group = intersectional_results.get("within_group_correlations", {})
    # BGL5 2026-09-27. This function read NEITHER `coverage` NOR
    # `groups_not_assessed` off a result that carries both, and no test executed it
    # at all (coverage of tests/test_bgl2_plotting_and_resampling.py over its lines
    # was []). Measured before, driven by real analyze_intersectional_correlations
    # results: on PARTIAL coverage (one intersectional cell shrunk to 3 rows
    # against min_group_size 30) the result said coverage 'partial' with
    # groups_not_assessed [{'group': 'f_b', 'n': 3, ...}] and the chart was bars 3,
    # xlabels ['f_a', 'm_b', 'm_a'], texts [], warnings [], under the title 'Mean
    # "score" Across Intersectional Groups': the cell that was never examined was
    # simply missing from a picture presenting itself as the groups. On NOT
    # ASSESSED coverage it returned None warning "Feature 'score' not found in
    # intersectional results", which names the wrong cause: the feature IS in the
    # frame, and no group reached the minimum. After: the unexamined cell keeps a
    # labelled slot marked 'not examined' with no bar plus a warning, and the
    # not-assessed case says no group reached min_group_size. The complete-coverage
    # control still draws its 4 bars with no warning.
    coverage = intersectional_results.get("coverage")
    groups_not_assessed = list(intersectional_results.get("groups_not_assessed", []) or [])

    groups = []
    means = []

    for group, features in within_group.items():
        if feature in features:
            groups.append(group)
            means.append(features[feature]["mean"])

    not_examined = [str(entry.get("group")) for entry in groups_not_assessed]

    if not groups:
        if not within_group and groups_not_assessed:
            warnings.warn(
                f"plot_intersectional_heatmap: no chart for '{feature}' because not "
                f"one of the {len(groups_not_assessed)} intersectional group(s) was "
                f"examined ({not_examined[:5]}"
                f"{' and more' if len(not_examined) > 5 else ''}; coverage "
                f"{coverage!r}), so the feature has no per-group mean. This is a "
                f"could-not-check about the groups, NOT a missing or misnamed "
                f"feature. See result['groups_not_assessed'].",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(f"Feature '{feature}' not found in intersectional results")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # The colormap registry is the typed access path; `plt.cm.RdYlGn` is a
    # dynamically added attribute mypy cannot see (2026-09-09).
    colors = colormaps["RdYlGn"](np.linspace(0.2, 0.8, len(groups)))
    ax.bar(range(len(groups)), means, color=colors)

    if not_examined:
        # A labelled slot with no bar, to the right of the examined groups: the
        # cell exists in the data and was never looked at, and a reader of the
        # picture has to be able to see that without reading the result dict.
        warnings.warn(
            f"plot_intersectional_heatmap: {len(not_examined)} intersectional "
            f"group(s) were never examined and have NO bar for '{feature}': "
            f"{not_examined[:5]}{' and more' if len(not_examined) > 5 else ''} "
            f"(coverage {coverage!r}). They are drawn as labelled slots marked 'not "
            f"examined', because a cell missing from this chart reads as a cell that "
            f"agrees with the rest.",
            UserWarning,
            stacklevel=2,
        )
        # A drawn mean can itself be NaN, and a text anchored at a NaN y is
        # dropped silently by matplotlib, which would delete the disclosure.
        _finite_means = [m for m in means if np.isfinite(m)]
        base = min(_finite_means) if _finite_means else 0.0
        for offset, group_name in enumerate(not_examined):
            ax.text(
                len(groups) + offset,
                base,
                "not examined",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
                color="#616161",
            )

    ax.set_xticks(range(len(groups) + len(not_examined)))
    ax.set_xticklabels(
        list(groups) + [f"{g} (not examined)" for g in not_examined],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Mean Value", fontsize=12)

    if title is None:
        title = f'Mean "{feature}" Across Intersectional Groups'
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# DASHBOARD VISUALIZATION


def _headline_association_was_measured(proxy: ProxyVariableResult) -> bool:
    """True only when this row's HEADLINE association is a number.

    A row whose headline measure could not be computed still carries a
    ``risk_level``, and that level is the grader's fall-through band, not a
    finding. Both of the producer's records are read, the explicit status first,
    because a renderer that trusts only the float would miss a status the producer
    set for a reason of its own.

    G07 2026-09-30: the numeric half was ``bool(np.isfinite(proxy.correlation))``,
    and it DISAGREED WITH ``_finite_or_nan`` in this same module, in both
    directions, with the weaker predicate sitting on the reader-facing surface.

    * ``np.isfinite(True)`` is True, so a BOOLEAN correlation read as measured.
      Measured through ``plot_proxy_risk_chart``: one row with
      ``correlation=True`` drew a bar of width 1 annotated "1.00" on an axis
      labelled "Absolute Correlation", a PERFECT proxy manufactured out of a
      flag, and ``create_analysis_dashboard`` classed the same row as GRADED, so
      it reached the pie chart ("NEGLIGIBLE (1) 100.0%"), the bar chart and Top
      Concerns with no NOT GRADED block anywhere in the figure. ``np.bool_`` is
      the half that gets missed, because it is not a Python ``bool`` and comes
      straight out of a DataFrame column.
    * ``np.isfinite("0.85")`` raises, so a NUMERIC STRING read as NOT measured,
      discarding a real measurement that had merely been through JSON or CSV,
      and ``plot_proxy_risk_chart`` then died on ``abs("0.85")`` with a
      TypeError rather than drawing it.

    ``_finite_or_nan`` already decides exactly this question, was hardened for
    the bool under READINESS-6, and keeps the numeric string on purpose. Reusing
    it is what makes the three surfaces agree instead of each holding its own
    answer. ``np.isfinite`` on a ``numpy.float32`` is still True through it, so
    a 32-bit correlation is still not mistaken for a missing one.
    """
    measures = (getattr(proxy, "evidence", None) or {}).get("correlation_measures") or {}
    if measures.get("headline_measure_status") == NOT_ASSESSED:
        return False
    return not np.isnan(_finite_or_nan(getattr(proxy, "correlation", None)))


def _headline_association(proxy: ProxyVariableResult) -> float:
    """The row's headline absolute association, or NaN when it was not measured.

    One place, so the value a chart DRAWS and the predicate that decides whether
    it may be drawn cannot disagree. ``abs`` is applied after the coercion, not
    to the raw field, because ``abs`` on the string a CSV hands over raises and
    ``abs`` on a bool returns 1.
    """
    if not _headline_association_was_measured(proxy):
        return float("nan")
    return abs(_finite_or_nan(proxy.correlation))


def create_analysis_dashboard(
    correlation_matrix: FeatureCorrelationMatrix,
    proxy_variables: List[ProxyVariableResult],
    *,
    figsize: Tuple[int, int] = (16, 12),
    title: str = "Feature Engineering Analysis Dashboard",
    save_path: Optional[str] = None,
) -> Any:
    """
    Create comprehensive dashboard with multiple visualizations.

    Creates a multi-panel dashboard with:
    - Correlation heatmap
    - Proxy risk chart
    - Risk distribution
    - Top recommendations

    The summary panel states the COVERAGE as well as the findings: how many
    feature/attribute correlations could not be computed, and, when
    ``proxy_variables`` is a ``ProxyScreenResult``, which pairs were never
    screened. A blank panel and an empty finding list mean "clean" only when
    that coverage is complete, and the panel now says which of the two it is.

    A screened pair whose HEADLINE association could not be measured is kept out
    of the risk pie, the risk bar chart and Top Concerns, and named under NOT
    GRADED instead. Its ``risk_level`` is the grader's fall-through band for a
    value that is not a number, so drawing it as a risk slice would publish a
    could-not-check as a measurement.

    Args:
        correlation_matrix: FeatureCorrelationMatrix
        proxy_variables: List of ProxyVariableResult. Pass the
            ``ProxyScreenResult`` that ``identify_proxy_variables`` returns
            rather than a plain list copy of it, so the panel can report the
            pairs that were never screened.
        figsize: Figure size
        title: Dashboard title
        save_path: Path to save the figure

    Returns:
        matplotlib figure object

    Example:
        >>> report = analyzer.full_analysis()
        >>> create_analysis_dashboard(
        ...     report.correlation_matrix,
        ...     report.proxy_variables
        ... )
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # BGL5 2026-09-27. THE THIRD COVERAGE RECORD, the one this panel dropped.
    # `identify_proxy_variables` returns a row for a pair whose headline
    # association could NOT be measured, deliberately, so the rest of its evidence
    # survives, and it marks it in three places: correlation nan,
    # evidence['correlation_measures']['headline_measure_status'] == 'not_assessed',
    # and a recommendation reading "NOT ASSESSED ... The risk level of 'negligible'
    # is the grader's fall-through band for a value that is not a number; it is NOT
    # a finding of low risk." Measured before, on 300 rows where income determines
    # gender and age is constant: the dashboard rendered 'CRITICAL (1) 50.0% /
    # NEGLIGIBLE (1) 50.0%' in the risk pie, 'nan' as a bar value and 'Top
    # Concerns: income (critical), age (negligible)', BESIDE its own sentence "NOT
    # MEASURED: 1 of 2 ... Their absence below is a could-not-check", while that
    # pair was not absent below at all, and the string 'not_assessed' appeared
    # nowhere in the figure. After: the ungraded pair is kept out of the pie, the
    # bar chart and Top Concerns (so the sentence is true), and a NOT GRADED block
    # names it. The 300-row single-proxy control still reads 'CRITICAL: 1' and
    # 'Features analyzed: 1 of 1' with no coverage line.
    graded: List[ProxyVariableResult] = []
    ungraded: List[ProxyVariableResult] = []
    for proxy in proxy_variables:
        if _headline_association_was_measured(proxy):
            graded.append(proxy)
        else:
            ungraded.append(proxy)

    # Panel 1: Correlation heatmap (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if correlation_matrix is not None:
        plot_correlation_heatmap(correlation_matrix, ax=ax1, title="Feature-Attribute Correlations")

    # Panel 2: Risk distribution (pie chart)
    ax2 = fig.add_subplot(gs[0, 2])
    if graded:
        plot_risk_distribution(graded, ax=ax2, title="Risk Distribution")

    # Panel 3: Proxy risk chart (spans full width)
    ax3 = fig.add_subplot(gs[1, :2])
    if graded:
        plot_proxy_risk_chart(graded, ax=ax3, title="Top Proxy Variables by Risk")

    # Panel 4: Summary text
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")

    risk_counts: Dict[str, int] = {}
    for proxy in graded:
        level = proxy.risk_level.value
        risk_counts[level] = risk_counts.get(level, 0) + 1

    # BGL3 2026-09-27. Both inputs carry a coverage record and this panel read
    # neither, so the dashboard was the surface where the disclosure died.
    # Measured on a 5-row frame (below MIN_SAMPLE_SIZE=10 and below the proxy
    # screen's min_sample_size=100): `correlations` was NaN for its only cell
    # and `identify_proxy_variables` returned a ProxyScreenResult whose
    # `screens_not_run` named the pair it never looked at and whose construction
    # raised a UserWarning. The panel rendered "Total features analyzed: 1 /
    # Protected attributes: 1 / Proxy Variables Found:" with nothing under the
    # heading, which is byte-identical to a clean frame of a million rows. The
    # correct measurement existed one call up and no reader of the picture could
    # see it.
    n_cells = 0
    n_measured_cells = 0
    features_measured = 0
    # Three states, not two. A matrix whose declared `protected_attributes` are
    # not columns of its own `correlations` frame (a hand-built or older one)
    # cannot be read for coverage at all, and that is not the same as its cells
    # being unmeasured. Saying "NOT MEASURED" about it would be as wrong as
    # saying nothing.
    coverage_readable = correlation_matrix is not None
    if correlation_matrix is not None:
        for feature in correlation_matrix.feature_names:
            row_measured = 0
            for attr in correlation_matrix.protected_attributes:
                n_cells += 1
                try:
                    value = correlation_matrix.correlations.loc[feature, attr]
                    finite = bool(np.isfinite(value))
                except (KeyError, IndexError, TypeError, ValueError):
                    coverage_readable = False
                    continue
                if finite:
                    row_measured += 1
            n_measured_cells += row_measured
            if row_measured:
                features_measured += 1
    screens_not_run = list(getattr(proxy_variables, "screens_not_run", []) or [])

    if not correlation_matrix:
        features_line = "Features analyzed: N/A"
    elif coverage_readable:
        features_line = (
            f"Features analyzed: {features_measured} of {len(correlation_matrix.feature_names)}"
        )
    else:
        features_line = f"Features in matrix: {len(correlation_matrix.feature_names)}"

    summary = [
        "ANALYSIS SUMMARY",
        "=" * 30,
        features_line,
        f"Protected attributes: {len(correlation_matrix.protected_attributes) if correlation_matrix else 'N/A'}",
    ]

    if correlation_matrix is not None and not coverage_readable:
        summary += [
            "",
            "COVERAGE UNKNOWN: this matrix does",
            "not hold a cell for every declared",
            "feature/attribute pair, so how much",
            "was measured cannot be read from it.",
        ]
    elif n_cells and n_measured_cells < n_cells:
        summary += [
            "",
            f"NOT MEASURED: {n_cells - n_measured_cells} of {n_cells}",
            "feature/attribute correlations could",
            "not be computed on this data. Their",
            "absence below is a could-not-check,",
            "not a finding of no correlation.",
        ]

    if ungraded:
        summary += [
            "",
            f"NOT GRADED: {len(ungraded)} screened pair(s)",
            "carry no measurable association, so",
            "they are absent from the risk chart",
            "and the concerns below. Their risk",
            "band is the grader's fall-through,",
            "not a finding of low risk:",
        ]
        for proxy in ungraded[:3]:
            summary.append(f"  {proxy.protected_attribute} ~ {proxy.feature}")
        if len(ungraded) > 3:
            summary.append(f"  (and {len(ungraded) - 3} more)")

    if screens_not_run:
        summary += [
            "",
            f"NOT SCREENED: {len(screens_not_run)} pair(s) were",
            "never examined for proxy risk:",
        ]
        for entry in screens_not_run[:3]:
            _feat = entry.get("feature") or "(whole attribute)"
            summary.append(f"  {entry.get('protected_attribute')} ~ {_feat}")
        if len(screens_not_run) > 3:
            summary.append(f"  (and {len(screens_not_run) - 3} more)")

    summary += [
        "",
        "Proxy Variables Found:",
    ]

    for level in ["critical", "high", "medium", "low"]:
        count = risk_counts.get(level, 0)
        if count > 0:
            summary.append(f"  {level.upper()}: {count}")

    if not graded:
        # The heading with nothing under it was read as "none". Say which of the
        # four it is, because an empty screen and a clean screen look the same.
        if not coverage_readable:
            summary.append("  none found, coverage unknown")
        elif ungraded:
            summary.append("  none graded, on unmeasurable")
            summary.append("  associations")
        elif screens_not_run or (n_cells and n_measured_cells < n_cells):
            summary.append("  none found, on incomplete coverage")
        elif correlation_matrix is not None and n_cells == 0:
            # 0 of 0 is not a clean audit: no pair existed to examine.
            summary.append("  none found, nothing was examined")
        else:
            summary.append("  none found")

    if graded:
        summary.extend(
            [
                "",
                "Top Concerns:",
            ]
        )
        for proxy in graded[:3]:
            summary.append(f"  • {proxy.feature}")
            summary.append(f"    ({proxy.risk_level.value})")

    ax4.text(
        0.05,
        0.95,
        "\n".join(summary),
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8),
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# PROXY CHAIN VISUALIZATION


def plot_proxy_chains(
    proxy_chains: List[Dict[str, Any]],
    *,
    max_chains: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Indirect Proxy Chains",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize proxy chain relationships.

    Shows indirect proxy paths: Feature -> Intermediate -> Protected Attribute

    Args:
        proxy_chains: List of proxy chain dictionaries
        max_chains: Maximum chains to display
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        ax: Existing axes to use

    Returns:
        matplotlib axes object
    """
    _check_matplotlib()

    if not proxy_chains:
        warnings.warn("No proxy chains to plot")
        return None

    chains = proxy_chains[:max_chains]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    labels = []
    values = []
    colors = []

    for chain in chains:
        chain_str = " → ".join(chain["chain"])
        labels.append(chain_str)
        values.append(chain["indirect_correlation"])

        # Color based on strength
        if chain["indirect_correlation"] >= 0.3:
            colors.append("#D32F2F")
        elif chain["indirect_correlation"] >= 0.2:
            colors.append("#F57C00")
        else:
            colors.append("#FBC02D")

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Indirect Correlation (Product of Chain)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9
        )

    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
