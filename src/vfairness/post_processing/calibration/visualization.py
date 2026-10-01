"""
Calibration Visualization for vfairness.

This module provides visualization tools for calibration analysis,
including reliability diagrams, calibration comparison plots, and
trade-off visualizations.

Available Plots:
    - Reliability diagrams (calibration curves)
    - Group-specific calibration comparison
    - Calibration disparity heatmaps
    - Trade-off frontier plots
    - Before/after calibration comparison

References:
    - Kumar, A., et al. (2019). Verified Uncertainty Calibration. NeurIPS.
    - Niculescu-Mizil, A. & Caruana, R. (2005). Predicting Good Probabilities
      with Supervised Learning. ICML.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd

# _MIN_SAMPLES_PER_GROUP_CURVE is the fewest samples in a group before a per-group
# calibration curve means anything. It is IMPORTED, not restated: this module and
# rendering.adapters_calibration draw the same comparison and carried different
# thresholds, 10 here against 1 there. See the constant's own comment for what that
# produced.
from vfairness._not_assessed import (
    MIN_ROWS_PER_GROUP_FOR_CALIBRATION as _MIN_SAMPLES_PER_GROUP_CURVE,
)

# is_measured is IMPORTED, not restated. It is the module whose docstring says it
# exists so "was this actually measured?" is answered once, and it is the version
# that does NOT classify a finite np.float32 as unmeasured. A local
# ``isinstance(v, (int, float))`` here would discard real numpy measurements while
# reading as caution.
from vfairness._triage import is_measured as _is_measured

# GroupManager._level_mask is REUSED, not restated: it is the one place in this
# library that knows ``attr == nan`` selects nobody, and it carries the
# measurement for why. See _unique_levels / the mask in plot_group_calibration.
from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)


# Lazy import for visualization libraries
def _get_matplotlib():
    """Lazy import matplotlib."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        return plt, mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for calibration visualization. "
            "Install with: pip install matplotlib"
        )


def _is_missing_level(value: Any) -> bool:
    """Whether one level of a protected attribute IS the absence of a value.

    Absence reaches a grouping through at least six doors (``None``, float NaN,
    the blank string, ``pd.NA``, ``pd.NaT`` and the literal string "None"), and
    ``str(x)`` on any of them MINTS a demographic group name. ``pd.isna``
    answers four; the two string spellings are named here because they are what
    an ordinary CSV read gives you. G03, 2026-09-30.
    """
    if isinstance(value, str):
        return value.strip() in {"", "None", "nan", "NaN", "<NA>", "NaT"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _unique_levels(attr: np.ndarray) -> List[Any]:
    """The distinct levels of a protected attribute, without sorting them.

    ``np.unique`` SORTS, and on an object array holding a float NaN beside
    strings the sort raises ``TypeError: '<' not supported between instances of
    'float' and 'str'``, which is a bare crash on a perfectly ordinary column
    with one missing cell. ``pd.unique`` preserves first-appearance order and
    compares nothing. Sorted afterwards only when every level is sortable, so
    the legend order and therefore the rendered bytes are unchanged for the
    homogeneous attributes that were already working. G03, 2026-09-30.
    """
    levels = list(pd.unique(np.asarray(attr).ravel()))
    try:
        return sorted(levels)
    except TypeError:
        return levels


def _get_colors(n_colors: int) -> List[str]:
    """Get a list of distinct colors."""
    base_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    else:
        # Cycle through colors if we need more
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]


def plot_reliability_diagram(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Reliability Diagram",
    ax: Optional[Any] = None,
    show_histogram: bool = True,
    show_ece: bool = True,
    color: str = "#1f77b4",
    figsize: Tuple[int, int] = (8, 6),
) -> Any:
    """
    Plot a reliability diagram (calibration curve).

    A reliability diagram shows the relationship between predicted
    probabilities and observed frequencies. A perfectly calibrated
    model would have all points on the diagonal.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration computation
        strategy: Binning strategy ('uniform' or 'quantile')
        title: Plot title
        ax: Matplotlib axes (created if None)
        show_histogram: Show histogram of predictions below
        show_ece: Show ECE value on plot
        color: Color for the calibration curve
        figsize: Figure size

    Returns:
        Matplotlib axes object

    Example:
        >>> ax = plot_reliability_diagram(y_true, y_prob)
        >>> plt.show()
    """
    plt, mpatches = _get_matplotlib()

    from .metrics import calibration_curve

    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")

    # Public `strategy` is typed as str for caller convenience; calibration_curve
    # accepts the 'uniform'/'quantile' literal. Behaviour is unchanged.
    curve = calibration_curve(
        y_true, y_prob, n_bins, cast(Literal["uniform", "quantile"], strategy)
    )

    if ax is None:
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
            )
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None
    else:
        ax1 = ax
        ax2 = None

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")

    ax1.plot(
        curve.prob_pred,
        curve.prob_true,
        "o-",
        color=color,
        linewidth=2,
        markersize=8,
        label="Model calibration",
    )

    # Fill the gap between perfect and actual
    ax1.fill_between(curve.prob_pred, curve.prob_pred, curve.prob_true, alpha=0.2, color=color)

    if show_ece:
        ece = curve.overall_ece
        # THE SENTINEL SPELLING IS DELIBERATE, and it is pinned. G03
        # (2026-09-30) proposed rewording this to "ECE = N/A (not measured)",
        # for consistency with the SVG twin ``rendering.adapters_calibration``,
        # and WITHDREW it: the wording reddened
        # tests/test_renderer_honesty.py::
        # test_reliability_diagram_prints_a_refusal_not_a_zero_for_an_unmeasurable_ece,
        # which asserts that the caption on zero rows contains "nan". That test
        # is a control, not an accident, and the refusal it protects is already
        # here: ``calibration_curve`` answers ``overall_ece`` NaN on zero rows
        # (never the 0.0 that would read as perfect calibration) and this caption
        # prints "ECE = nan", which is visible and is not a number a reader can
        # mistake for a measurement. A guard that reddens an existing control is
        # the wrong guard; the reasoning is left here instead of the change.
        ax1.text(
            0.05,
            0.95,
            f"ECE = {ece:.3f}",
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    if show_histogram and ax2 is not None:
        ax2.hist(y_prob, bins=n_bins, range=(0, 1), color=color, alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Predicted Probability", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

    return ax1


def plot_calibration_comparison(
    results: Dict[str, Tuple[ArrayLike, ArrayLike]],
    y_true: ArrayLike,
    n_bins: int = 10,
    title: str = "Calibration Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """
    Compare calibration curves for multiple models or conditions.

    Args:
        results: Dict mapping names to (y_prob_before, y_prob_after) tuples
                 or just y_prob arrays
        y_true: True binary labels (same for all)
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib axes object

    Example:
        >>> results = {
        ...     'Before': y_prob_original,
        ...     'After Platt': y_prob_platt,
        ...     'After Isotonic': y_prob_isotonic
        ... }
        >>> ax = plot_calibration_comparison(results, y_true)
    """
    plt, _ = _get_matplotlib()
    from .metrics import calibration_curve

    y_true = coerce_to_array(y_true, "y_true")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect")

    colors = _get_colors(len(results))

    for (name, probs), color in zip(results.items(), colors):
        if isinstance(probs, tuple):
            # (before, after) tuple
            prob_values = probs[1]  # Use after
        else:
            prob_values = probs

        prob_array = coerce_to_array(prob_values, name)
        curve = calibration_curve(y_true, prob_array, n_bins)

        ax.plot(
            curve.prob_pred,
            curve.prob_true,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{name} (ECE={curve.overall_ece:.3f})",
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return ax


def plot_group_calibration(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    title: str = "Calibration by Group",
    figsize: Tuple[int, int] = (10, 6),
    show_overall: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot calibration curves for each demographic group.

    This visualization helps identify calibration disparities across
    groups, which may indicate unfair probability interpretation.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size
        show_overall: Whether to show overall calibration
        ax: Optional matplotlib axes to plot on

    Returns:
        Matplotlib axes object

    Example:
        >>> ax = plot_group_calibration(y_true, y_prob, gender)
        >>> plt.show()
    """
    plt, _ = _get_matplotlib()
    from .metrics import calibration_curve

    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect")

    if show_overall:
        overall_curve = calibration_curve(y_true, y_prob, n_bins)
        # Left as the bare ``.3f``, i.e. "nan" for an unmeasurable ECE, to match
        # plot_reliability_diagram's caption. See the withdrawn rewording there:
        # the sentinel spelling is what the renderer-honesty control asserts, and
        # having the two sibling charts spell the same refusal two different ways
        # would be worse than either spelling. G03, 2026-09-30.
        ax.plot(
            overall_curve.prob_pred,
            overall_curve.prob_true,
            "o-",
            color="gray",
            linewidth=2,
            markersize=6,
            alpha=0.5,
            label=f"Overall (ECE={overall_curve.overall_ece:.3f})",
        )

    # A LEVEL WITH A MISSING VALUE HOLDS THE ROWS IT HOLDS, and is named as
    # missing rather than as too small.
    #
    # G03 (2026-09-30). This was ``np.unique(protected_attr)`` with
    # ``protected_attr == group`` for the mask, and ``attr == nan`` is False for
    # EVERY row, so the missing level's mask selected nobody. MEASURED before
    # this fix on a float attribute of 100 zeros and 100 NaNs, the chart carried
    # "Not measured (n<10): nan (n=0)" and warned "fewer than 10 samples" about a
    # level holding HALF the dataset, while those 100 rows still sat inside the
    # pooled "Overall" curve. ``GroupManager._level_mask`` in
    # evaluation/vfairness_metrics/_grouping.py was fixed for exactly this
    # ("0 against 10 for ten missing rows") and its rule is reused here rather
    # than restated, which is also what makes the two surfaces agree on the
    # count. An object-dtype attribute holding a float NaN beside strings raised
    # a bare ``TypeError: '<' not supported between instances of 'float' and
    # 'str'`` out of ``np.unique``'s sort; the levels are taken without sorting
    # mixed types now.
    unique_groups = _unique_levels(protected_attr)
    colors = _get_colors(len(unique_groups))
    missing_levels: List[Tuple[Any, int]] = []

    # A group too small to calibrate is NAMED ON THE CHART, not dropped from it.
    #
    # This was `if np.sum(mask) < 10: continue`, a bare continue, and the group then
    # left no trace anywhere: no curve, no legend entry, no warning. Measured before
    # the fix on 57 rows in group A beside 3 rows in group B, the chart titled
    # "Calibration by Group" rendered a legend of exactly "Perfect", "Overall" and
    # "A (n=57)". A reader comparing groups saw ONE group and had no way to learn
    # that a second existed, so the plot answered "is calibration equal across
    # groups?" by quietly removing the group that could have made it unequal. That
    # is the strongest possible all-clear, drawn for a comparison never made.
    #
    # CalibrationMetricResult.to_dict already carries n_groups_compared for exactly
    # this reason. The chart is the surface most readers actually meet, and it was
    # the one surface not saying it.
    too_small: List[Tuple[Any, int]] = []
    for group, color in zip(unique_groups, colors):
        mask = GroupManager._level_mask(protected_attr, group)
        n_group = int(np.sum(mask))
        if _is_missing_level(group):
            # Not a protected class anybody chose, so no curve is drawn for it,
            # and its own count is stated rather than the 0 that `== nan` gave.
            missing_levels.append((group, n_group))
            continue
        if n_group < _MIN_SAMPLES_PER_GROUP_CURVE:
            too_small.append((group, n_group))
            continue

        group_curve = calibration_curve(y_true[mask], y_prob[mask], n_bins)

        ax.plot(
            group_curve.prob_pred,
            group_curve.prob_true,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{group} (ECE={group_curve.overall_ece:.3f}, n={n_group})",
        )

    if too_small:
        named = ", ".join(f"{g} (n={n})" for g, n in too_small)
        warnings.warn(
            f"plot_group_calibration: no calibration curve was drawn for {named}; "
            f"fewer than {_MIN_SAMPLES_PER_GROUP_CURVE} samples means the curve was "
            f"NOT MEASURED for that group, which is not the same as the group being "
            f"well calibrated. The chart names them so the omission is visible.",
            stacklevel=2,
        )
        ax.text(
            0.02,
            0.98,
            f"Not measured (n<{_MIN_SAMPLES_PER_GROUP_CURVE}): {named}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B", alpha=0.9),
        )

    if missing_levels:
        named_missing = ", ".join(f"{str(g) or repr(g)} (n={n})" for g, n in missing_levels)
        warnings.warn(
            f"plot_group_calibration: {len(missing_levels)} level(s) of the protected "
            f"attribute ARE a missing value, not a group: {named_missing}. No curve is "
            f"drawn for them, their rows are still inside the pooled 'Overall' curve, "
            f"and their row count is stated: `attr == nan` selects nobody, so this "
            f"used to be reported as a level of n=0 that was 'too small'.",
            stacklevel=2,
        )
        ax.text(
            0.02,
            0.86 if too_small else 0.98,
            f"Missing attribute (not a group): {named_missing}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B", alpha=0.9),
        )

    if len(unique_groups) and len(too_small) + len(missing_levels) == len(unique_groups):
        # Every group fell out, so the only curves on the canvas are the diagonal
        # and the pooled one. Without this the title still reads "Calibration by
        # Group" over a chart containing no group at all.
        ax.text(
            0.5,
            0.5,
            "No group could be calibrated:\nevery group has too few samples",
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="center",
            color="#8A6D3B",
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return ax


def plot_calibration_disparity(
    disparity_result,
    title: str = "Calibration Disparity Analysis",
    figsize: Tuple[int, int] = (12, 5),
) -> Any:
    """
    Plot calibration disparity analysis results.

    Creates a multi-panel visualization showing ECE, MCE, and Brier
    scores across groups.

    A group that ``min_group_size`` kept out of the comparison is named on the
    chart and in the suptitle rather than dropped from it, and a result with
    fewer than two compared groups is marked "not measured": the figure may not
    read as an all-clear for a comparison that was never made. See the measured
    before-and-after in the body.

    A group that CLEARED the size gate and still has no computable calibration
    error is marked the same way, on every panel it appears on, and no bar is
    drawn for it: in particular its ``group_mce``, which arrives as a fabricated
    0.0 rather than a NaN, is not drawn, because a real bar of height 0.0 reads as
    perfect worst-case calibration for a group nobody measured.

    Args:
        disparity_result: CalibrationDisparityResult from calibration_disparity()
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> from vfairness.post_processing.calibration import calibration_disparity
        >>> result = calibration_disparity(y_true, y_prob, race)
        >>> fig = plot_calibration_disparity(result)
        >>> plt.show()
    """
    plt, _ = _get_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    groups = list(disparity_result.group_ece.keys())
    colors = _get_colors(len(groups))

    # A GROUP THE COMPARISON LEFT OUT IS NAMED ON THIS CHART, not dropped from it.
    #
    # BGL5 (2026-09-27). ``CalibrationDisparityResult.excluded_groups`` is the
    # record of the groups ``min_group_size`` kept out of every figure drawn
    # here, and this function read only ``group_ece``, so the omitted group left
    # no trace: no bar, no tick, no legend entry, no note. MEASURED on 150 'a' +
    # 150 'b' + 25 'c' at the default gate of 30, with 'c' scored 0.97 against an
    # outcome of 0 (its own ECE is 0.97), every text on the figure was:
    #
    #   TITLE:Expected Calibration Error, TICK:a, TICK:b,
    #   LEG:Good calibration threshold, TITLE:Maximum Calibration Error,
    #   TICK:a, TICK:b, TITLE:Brier Score, TICK:a, TICK:b
    #
    # under a suptitle reading "Calibration Disparity Analysis", i.e. a chart
    # whose own title promises the comparison answered it by removing the group
    # that would have made it unequal. Two siblings in this module were fixed for
    # exactly this and are mirrored here: ``plot_group_calibration`` adds a
    # "Not measured (n<10): ..." box, and ``create_calibration_dashboard``
    # annotates "2 of 3 groups; c excluded". After the fix the same figure also
    # carries "Not measured (below min_group_size): c" and the suptitle says
    # "2 of 3 groups compared; c excluded".
    excluded = [str(g) for g in (getattr(disparity_result, "excluded_groups", None) or [])]

    # A GROUP ON THE AXIS WHOSE CALIBRATION WAS NEVER MEASURED IS MARKED TOO, and
    # its MCE bar is NOT drawn at the fabricated 0.0 the result carries.
    #
    # BGL5 wave 4 (2026-09-30). The guard above keys on ``excluded_groups``, which
    # is a COUNT gate: it holds the groups ``min_group_size`` kept out. It cannot
    # see the sibling door, a group that CLEARS the size gate and whose calibration
    # error still could not be computed, because ``groups`` is built from the KEYS
    # of ``group_ece`` and never looks at its VALUES. MEASURED before this fix, on
    # 400 rows in two 200-row groups with group B carrying no ground truth at all,
    # at the default gate of 30:
    #
    #   group_ece        {'A': 0.3006, 'B': nan}
    #   group_mce        {'A': 0.4248, 'B': 0.0}
    #   group_brier      {'A': 0.0927, 'B': nan}
    #   excluded_groups  []
    #   DRAWN TEXTS      ... 'A', 'B' on all three panels, suptitle
    #                    "Calibration Disparity Analysis", no mark of any kind
    #   warnings         []
    #
    # Two things make this WORSE than the excluded-group case the guard above
    # covers. B is NAMED on the axis, so a reader sees a labelled bar rather than
    # an absence, and its NaN bar draws nothing, which reads as zero error. And
    # ``group_mce`` for B is a FABRICATED 0.0, not a NaN, so panel 2 drew a real
    # bar of height 0.0 for a group nobody could measure, i.e. PERFECT worst-case
    # calibration: the neutral value substituted for the unmeasurable one, drawn
    # to the reader as the best possible result.
    #
    # THE GUARD SITS ABOVE THE DISPATCH because all three panels share the
    # precondition. Inside one panel it would only move the fabrication to the
    # other two. ECE is the group-level evidence test: when a group's ECE is not a
    # real finite number, nothing about that group's calibration was measured, so
    # its MCE and Brier values are not drawn either, whatever they happen to hold.
    not_measured = [g for g in groups if not _is_measured(disparity_result.group_ece.get(g))]
    not_measured_set = set(not_measured)

    def _drawable(group: Any, value: Any) -> float:
        """The bar height to draw, or NaN when there is no measurement to draw."""
        if group in not_measured_set or not _is_measured(value):
            return float("nan")
        return float(value)

    def _mark_not_measured(ax: Any, values: List[float]) -> None:
        """Name every position on this panel that carries no measurement.

        A NaN bar draws NOTHING, which is visually identical to a measured 0.0
        sitting on the axis. The annotation is what makes the two distinguishable
        to a reader and to a test reading the rendered figure.
        """
        for position, value in enumerate(values):
            if np.isfinite(value):
                continue
            ax.annotate(
                "not measured",
                xy=(position, 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
                color="#8A6D3B",
                rotation=90,
            )

    ax1 = axes[0]
    ece_values = [_drawable(g, disparity_result.group_ece.get(g)) for g in groups]
    ax1.bar(groups, ece_values, color=colors, alpha=0.8, edgecolor="black")
    _mark_not_measured(ax1, ece_values)
    ax1.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="Good calibration threshold")
    ax1.set_ylabel("ECE", fontsize=11)
    ax1.set_title("Expected Calibration Error", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    mce_values = [_drawable(g, disparity_result.group_mce.get(g)) for g in groups]
    ax2.bar(groups, mce_values, color=colors, alpha=0.8, edgecolor="black")
    _mark_not_measured(ax2, mce_values)
    ax2.set_ylabel("MCE", fontsize=11)
    ax2.set_title("Maximum Calibration Error", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)

    ax3 = axes[2]
    brier_values = [_drawable(g, disparity_result.group_brier.get(g)) for g in groups]
    ax3.bar(groups, brier_values, color=colors, alpha=0.8, edgecolor="black")
    _mark_not_measured(ax3, brier_values)
    ax3.set_ylabel("Brier Score", fontsize=11)
    ax3.set_title("Brier Score", fontsize=12)
    ax3.tick_params(axis="x", rotation=45)

    if excluded:
        named = ", ".join(excluded)
        warnings.warn(
            f"plot_calibration_disparity: no bar was drawn for {named}; those groups "
            f"fell below min_group_size, so their calibration was NOT MEASURED and the "
            f"spread on this chart is a lower bound over the "
            f"{len(groups)} group(s) that were compared. The chart names them so the "
            f"omission is visible.",
            stacklevel=2,
        )
        ax1.text(
            0.02,
            0.98,
            f"Not measured (below min_group_size): {named}",
            transform=ax1.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B", alpha=0.9),
        )

    if not_measured:
        named_unmeasured = ", ".join(str(g) for g in not_measured)
        warnings.warn(
            f"plot_calibration_disparity: {named_unmeasured} cleared min_group_size but "
            f"NO calibration error could be computed for it, so no bar is drawn for it on "
            f"any panel and its group_mce of "
            f"{[disparity_result.group_mce.get(g) for g in not_measured]} is NOT drawn: an "
            f"MCE of 0.0 for a group nobody could measure would read as perfect worst-case "
            f"calibration. The spread on this chart is a lower bound over the "
            f"{len(groups) - len(not_measured)} group(s) that carry a measurement.",
            stacklevel=2,
        )
        ax1.text(
            0.02,
            0.02 if excluded else 0.98,
            f"Calibration not measured: {named_unmeasured}",
            transform=ax1.transAxes,
            fontsize=8,
            va="bottom" if excluded else "top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B", alpha=0.9),
        )

    # A ONE-group result has nothing to compare, so ece_disparity is NaN and
    # has_significant_disparity is None. Measured before this: the three panels
    # drew one bar each with no mark of any kind, under the same "Calibration
    # Disparity Analysis" suptitle, which is a disparity chart for a disparity
    # nobody measured. It now carries "Disparity not measured: a comparison needs
    # at least two groups" across the middle of the first panel.
    #
    # BGL5 wave 4: the test counts groups that carry a MEASUREMENT, not rows on the
    # axis. Two groups of which one has no computable calibration error is the same
    # comparison as one group, and the bare ``len(groups) < 2`` count could not see
    # it, so the figure read as a made comparison.
    n_measured_groups = len(groups) - len(not_measured)
    if n_measured_groups < 2:
        ax1.text(
            0.5,
            0.5,
            "Disparity not measured:\na comparison needs at least two groups",
            transform=ax1.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            color="#8A6D3B",
        )

    # The suptitle is the one line every reader of this figure reads, and it
    # promised a comparison across the groups. It states its coverage now.
    #
    # BGL5 wave 4: a group on the axis with no computable calibration error is not
    # a compared group either, so it is subtracted from the numerator and named in
    # the same breath as the size-excluded ones. Before this the suptitle of the
    # measured 400-row case above said nothing at all.
    if excluded or not_measured:
        omitted = excluded + [f"{g} (not measured)" for g in not_measured]
        fig.suptitle(
            f"{title} ({n_measured_groups} of {len(groups) + len(excluded)} groups compared; "
            f"{', '.join(omitted)} excluded)",
            fontsize=14,
            fontweight="bold",
        )
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_tradeoff_curve(
    tradeoff_result,
    title: str = "Calibration-Fairness Trade-off",
    figsize: Tuple[int, int] = (10, 6),
    show_current: bool = True,
    show_pareto: bool = True,
) -> Any:
    """
    Plot the calibration-fairness trade-off frontier.

    Visualizes the Pareto frontier showing achievable combinations
    of calibration quality and fairness.

    Args:
        tradeoff_result: TradeoffAnalysisResult from analyze_calibration_fairness_tradeoff()
        title: Plot title
        figsize: Figure size
        show_current: Highlight current model position
        show_pareto: Show Pareto frontier line

    Returns:
        Matplotlib axes object

    Example:
        >>> from vfairness.post_processing.calibration import analyze_calibration_fairness_tradeoff
        >>> result = analyze_calibration_fairness_tradeoff(y_true, y_prob, gender)
        >>> ax = plot_tradeoff_curve(result)
        >>> plt.show()
    """
    plt, _ = _get_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    thresholds = []
    calibration_errors = []
    fairness_violations = []

    for threshold, metrics in tradeoff_result.metrics_at_thresholds.items():
        thresholds.append(threshold)
        calibration_errors.append(metrics["ece"])
        fairness_violations.append(metrics["fairness_violation"])

    scatter = ax.scatter(
        calibration_errors,
        fairness_violations,
        c=thresholds,
        cmap="viridis",
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Decision Threshold", fontsize=11)

    # Plot Pareto frontier
    if show_pareto and tradeoff_result.pareto_points:
        pareto_cal = [p.calibration_error for p in tradeoff_result.pareto_points]
        pareto_fair = [p.fairness_violation for p in tradeoff_result.pareto_points]

        ax.plot(pareto_cal, pareto_fair, "r-", linewidth=2, alpha=0.7, label="Pareto Frontier")
        ax.scatter(
            pareto_cal,
            pareto_fair,
            color="red",
            s=120,
            marker="*",
            zorder=5,
            label="Pareto Optimal",
        )

    if show_current:
        ax.scatter(
            [tradeoff_result.current_point.calibration_error],
            [tradeoff_result.current_point.fairness_violation],
            color="green",
            s=200,
            marker="X",
            zorder=6,
            label="Current (t=0.5)",
            edgecolors="black",
            linewidths=2,
        )

    ax.set_xlabel("Expected Calibration Error", fontsize=12)
    ax.set_ylabel("Fairness Violation", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    severity = tradeoff_result.tradeoff_severity
    ax.text(
        0.02,
        0.98,
        f"Trade-off Severity: {severity.upper()}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return ax


def plot_pareto_frontier(
    calibration_errors: ArrayLike,
    fairness_violations: ArrayLike,
    labels: Optional[List[str]] = None,
    x_label: str = "Calibration Error",
    y_label: str = "Fairness Violation",
    title: str = "Pareto Frontier: Calibration vs. Fairness",
    figsize: Tuple[int, int] = (10, 7),
    highlight_frontier: bool = True,
    shade_dominated: bool = True,
    frontier_color: str = "#d62728",
    frontier_linewidth: float = 2.5,
    point_color: str = "#1f77b4",
    dominated_color: str = "#e0e0e0",
    show_annotations: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot a classical Pareto frontier visualization.

    A Pareto frontier shows the set of non-dominated points where improving
    one objective necessarily worsens another. Points on the frontier represent
    optimal trade-offs between calibration error and fairness violation.

    A point with a non-finite coordinate CANNOT BE RANKED: no dominance
    comparison against it can be decided, so it is drawn neither as Pareto
    optimal nor as dominated, it is named on the chart and in a warning, and it
    does not enter the axis limits. Leaving it off the frontier is not a finding
    that it is dominated. See the body for the measurement.

    Args:
        calibration_errors: Array of calibration error values (e.g., ECE)
        fairness_violations: Array of fairness violation values (e.g., FPR disparity)
        labels: Optional labels for each point (e.g., threshold values)
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        figsize: Figure size
        highlight_frontier: Whether to highlight the Pareto frontier with a thick line
        shade_dominated: Whether to shade the dominated region
        frontier_color: Color for the Pareto frontier line
        frontier_linewidth: Line width for the frontier
        point_color: Color for non-frontier (dominated) points. R6-4
            (2026-09-10): this was documented and inert; the dominated
            scatter was hardcoded to "gray", so "#1f77b4", "#00ff00" and
            "magenta" rendered byte-identical PNGs (same sha256, on a
            harness where frontier_color did change the bytes, so the
            harness was not the reason). It is read now, which also means
            the DEFAULT picture changed: dominated points are the module's
            blue #1f77b4 rather than gray, i.e. the colour the signature
            has always promised. Pass point_color="gray" for the old look.
        dominated_color: Color for shading the dominated region (the shaded
            area, not the points)
        show_annotations: Whether to show point labels/annotations
        ax: Optional matplotlib axes to plot on

    Returns:
        Matplotlib axes object

    Example:
        >>> # From trade-off analysis
        >>> cal_errors = [0.05, 0.08, 0.10, 0.12, 0.15]
        >>> fair_violations = [0.20, 0.12, 0.08, 0.06, 0.05]
        >>> ax = plot_pareto_frontier(cal_errors, fair_violations)
        >>> plt.show()

        >>> # With threshold labels
        >>> thresholds = ['t=0.3', 't=0.4', 't=0.5', 't=0.6', 't=0.7']
        >>> ax = plot_pareto_frontier(cal_errors, fair_violations, labels=thresholds)
    """
    plt, mpatches = _get_matplotlib()

    calibration_errors = coerce_to_array(calibration_errors, "calibration_errors")
    fairness_violations = coerce_to_array(fairness_violations, "fairness_violations")

    check_consistent_length(calibration_errors, fairness_violations)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # A POINT WITH A NON-FINITE COORDINATE CANNOT BE RANKED, so it is not drawn
    # as Pareto optimal.
    #
    # G03 (2026-09-30). This function re-implements, inline, the dominance test
    # that ``tradeoffs.ParetoPoint.dominates`` and
    # ``tradeoffs.compute_pareto_frontier`` were fixed for on 2026-09-27, and it
    # kept the defect. "Nothing dominates it" is what puts a point on the
    # frontier, and every comparison against a NaN is False, so a point whose
    # fairness violation was never measured was nothing's inferior and went into
    # the red "Pareto Optimal" scatter with a black edge, the strongest mark on
    # the chart. The sibling's own docstring records the input that produces it:
    # 240 rows where one group held no negative labels, so all 17 swept
    # thresholds carried ``fairness_violation`` NaN.
    #
    # MEASURED here before this fix, on five points with the fifth fairness
    # violation set to NaN: the "Pareto Optimal" collection was built including
    # that point, and the call then died at the very END with matplotlib's bare
    # ``ValueError: Axis limits cannot be NaN or Inf`` from ``set_ylim``, because
    # ``fairness_violations.max()`` propagates the NaN. That message names an
    # axis, not an unmeasured trade-off point, so a caller cannot tell it from a
    # matplotlib problem; and it is incidental, which is the real hazard, because
    # any future change to how the limits are computed turns the crash back into
    # a silently mismarked frontier.
    n_points = len(calibration_errors)
    rankable = np.isfinite(calibration_errors) & np.isfinite(fairness_violations)
    is_pareto = rankable.copy()

    for i in range(n_points):
        if not rankable[i]:
            continue
        for j in range(n_points):
            if i != j and rankable[j]:
                # Point j dominates point i if j is <= in both and < in at least one
                if (
                    calibration_errors[j] <= calibration_errors[i]
                    and fairness_violations[j] <= fairness_violations[i]
                    and (
                        calibration_errors[j] < calibration_errors[i]
                        or fairness_violations[j] < fairness_violations[i]
                    )
                ):
                    is_pareto[i] = False
                    break

    n_unrankable = int(np.count_nonzero(~rankable))
    if n_unrankable:
        warnings.warn(
            f"plot_pareto_frontier: {n_unrankable} of {n_points} point(s) have a "
            f"non-finite calibration error or fairness violation, so no dominance "
            f"comparison against them can be decided. They are NOT drawn on the "
            f"frontier and NOT drawn as dominated, and their omission is named on the "
            f"chart: leaving them out is not a finding that they are dominated."
            + (
                " NOTHING could be ranked, so the frontier is empty for want of "
                "measurements, not because no trade-off point is optimal."
                if not np.any(rankable)
                else ""
            ),
            stacklevel=2,
        )

    # Only the finite coordinates decide the axes, the shading and the margins.
    # Every reduction below (`min`, `max`) propagates a NaN, which is where the
    # bare matplotlib ValueError came from.
    finite_cal = calibration_errors[rankable]
    finite_fair = fairness_violations[rankable]

    pareto_cal = calibration_errors[is_pareto]
    pareto_fair = fairness_violations[is_pareto]

    # Sort Pareto points by calibration error for proper line drawing
    sort_idx = np.argsort(pareto_cal)
    pareto_cal_sorted = pareto_cal[sort_idx]
    pareto_fair_sorted = pareto_fair[sort_idx]

    if shade_dominated and len(pareto_cal_sorted) > 0:
        # The dominated region is everything "above and to the right" of the frontier
        max_cal = max(finite_cal.max() * 1.1, pareto_cal_sorted[-1] * 1.2)
        max_fair = max(finite_fair.max() * 1.1, pareto_fair_sorted[0] * 1.2)

        # Start from top-left, go along frontier, then to corners
        polygon_x = [0]
        polygon_y = [max_fair]

        # Add points along the frontier (staircase pattern for proper dominated region)
        for i in range(len(pareto_cal_sorted)):
            polygon_x.append(pareto_cal_sorted[i])
            polygon_y.append(max_fair if i == 0 else pareto_fair_sorted[i - 1])
            polygon_x.append(pareto_cal_sorted[i])
            polygon_y.append(pareto_fair_sorted[i])

        # Close the polygon
        polygon_x.extend([max_cal, max_cal, 0])
        polygon_y.extend([pareto_fair_sorted[-1], max_fair, max_fair])

        ax.fill(
            polygon_x,
            polygon_y,
            color=dominated_color,
            alpha=0.4,
            label="Dominated Region",
            zorder=1,
        )

    # ``~is_pareto`` would sweep the UNRANKABLE points in here, and "Dominated
    # Points" is a finding: it says something else is better in both objectives,
    # which is precisely what could not be decided about them. G03, 2026-09-30.
    dominated_mask = rankable & ~is_pareto
    if np.any(dominated_mask):
        ax.scatter(
            calibration_errors[dominated_mask],
            fairness_violations[dominated_mask],
            # R6-4: the caller's point_color, not a hardcoded "gray".
            c=point_color,
            alpha=0.5,
            s=60,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
            label="Dominated Points",
        )

    # Plot Pareto optimal points
    ax.scatter(
        pareto_cal,
        pareto_fair,
        c=frontier_color,
        s=100,
        edgecolors="black",
        linewidths=1.5,
        zorder=4,
        label="Pareto Optimal",
    )

    # Draw the Pareto frontier line
    if highlight_frontier and len(pareto_cal_sorted) > 1:
        ax.plot(
            pareto_cal_sorted,
            pareto_fair_sorted,
            color=frontier_color,
            linewidth=frontier_linewidth,
            linestyle="-",
            marker="",
            zorder=3,
            label="Pareto Frontier",
        )

    if show_annotations and labels is not None:
        pareto_indices = np.where(is_pareto)[0]
        for idx in pareto_indices:
            ax.annotate(
                labels[idx],
                (calibration_errors[idx], fairness_violations[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # The unrankable points are NAMED ON THE CHART, not merely left off it. A
    # missing marker is invisible, and this chart's whole claim is that the red
    # points are the optimal trade-offs available. G03, 2026-09-30.
    if n_unrankable:
        ax.text(
            0.02,
            0.98,
            f"Not ranked ({n_unrankable} of {n_points}): a non-finite\n"
            f"coordinate cannot be compared, so these points are\n"
            f"neither optimal nor dominated",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B", alpha=0.9),
        )

    if len(finite_cal):
        x_margin = (finite_cal.max() - finite_cal.min()) * 0.1 or 0.01
        y_margin = (finite_fair.max() - finite_fair.min()) * 0.1 or 0.01
        ax.set_xlim(max(0, finite_cal.min() - x_margin), finite_cal.max() + x_margin)
        ax.set_ylim(max(0, finite_fair.min() - y_margin), finite_fair.max() + y_margin)
    else:
        # Not one point could be ranked, so there are no data limits to set and
        # the default unit square is left in place under an explicit statement.
        # Before this the call died on matplotlib's "Axis limits cannot be NaN
        # or Inf", which names an axis rather than the missing measurements.
        ax.text(
            0.5,
            0.5,
            "No trade-off point could be ranked:\nevery point has a non-finite coordinate",
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="center",
            color="#8A6D3B",
        )

    ax.legend(loc="upper right", fontsize=10)

    ax.text(
        0.02,
        0.02,
        "Lower-left is better\n(less error, less violation)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        alpha=0.7,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return ax


def create_calibration_dashboard(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    calibrated_probs: Optional[ArrayLike] = None,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (16, 12),
) -> Any:
    """
    Create a comprehensive calibration analysis dashboard.

    Generates a multi-panel figure with:
    - Reliability diagram (before calibration)
    - Reliability diagram (after calibration, if provided)
    - Group-specific calibration curves
    - Calibration metrics comparison

    Args:
        y_true: True binary labels
        y_prob: Original predicted probabilities
        protected_attr: Protected attribute for group analysis
        calibrated_probs: Calibrated probabilities (optional)
        n_bins: Number of bins
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> from vfairness.post_processing.calibration import GroupCalibrator
        >>> calibrator = GroupCalibrator(method='isotonic')
        >>> calibrator.fit(y_true, y_prob, gender)
        >>> calibrated = calibrator.transform(y_prob, gender)
        >>> fig = create_calibration_dashboard(
        ...     y_true, y_prob, gender, calibrated
        ... )
        >>> plt.show()
    """
    plt, _ = _get_matplotlib()

    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")

    has_calibrated = calibrated_probs is not None
    if has_calibrated:
        calibrated_probs = coerce_to_array(calibrated_probs, "calibrated_probs")

    if has_calibrated:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
        gs = fig.add_gridspec(1, 3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[0, 2])
        ax2, ax4 = None, None

    # Plot 1: Original reliability diagram
    plot_reliability_diagram(
        y_true,
        y_prob,
        n_bins,
        title="Original Calibration",
        ax=ax1,
        show_histogram=False,
        color="#d62728",
    )

    # Plot 2: Calibrated reliability diagram (if available)
    if has_calibrated and ax2 is not None:
        plot_reliability_diagram(
            y_true,
            calibrated_probs,
            n_bins,
            title="After Calibration",
            ax=ax2,
            show_histogram=False,
            color="#2ca02c",
        )

    # Plot 3: Group calibration (original)
    plot_group_calibration(
        y_true,
        y_prob,
        protected_attr,
        n_bins,
        title="Original: By Group",
        ax=ax3,
        show_overall=True,
    )
    legend3 = ax3.get_legend()
    if legend3 is not None:
        for text in legend3.get_texts():
            text.set_fontsize(8)

    # Plot 4: Group calibration (calibrated, if available)
    if has_calibrated and ax4 is not None:
        plot_group_calibration(
            y_true,
            calibrated_probs,
            protected_attr,
            n_bins,
            title="Calibrated: By Group",
            ax=ax4,
            show_overall=True,
        )
        legend4 = ax4.get_legend()
        if legend4 is not None:
            for text in legend4.get_texts():
                text.set_fontsize(8)

    # Plot 5: Metrics summary
    from .metrics import expected_calibration_error

    ece_before = expected_calibration_error(y_true, y_prob, protected_attr)

    metrics_labels = ["Overall ECE", "ECE Disparity"]
    before_values = [ece_before.overall_value, ece_before.max_group_disparity]

    # THREE STATES ON THE CHART, NOT TWO.
    #
    # BGL-final-d00 (2026-09-17). `max_group_disparity` is NaN when fewer than
    # two groups carry a measured ECE (one group only, or a minority dropped by
    # min_group_size), and a NaN bar draws exactly the same thing a measured 0.0
    # draws: nothing at all, comfortably under the red 0.05 target line. So the
    # reader of the dashboard saw PERFECT PARITY for a comparison nobody made,
    # which is the defect `metrics.py` removed from the value and `to_dict()`
    # discloses, stopping one layer short of the surface a person actually
    # looks at. Each unmeasured bar is now named on the axes it is missing from.
    def _mark_unmeasured(positions, values, series: str) -> None:
        for pos, value in zip(positions, values):
            if np.isfinite(value):
                continue
            ax5.annotate(
                f"{series}not measured" if series else "not measured",
                xy=(pos, 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
                color="#7f7f7f",
                rotation=90,
            )

    # A DISPARITY OVER A SUBSET OF THE GROUPS IS NOT THE DISPARITY.
    #
    # BGL3-PP3 (2026-09-27). The guard above is for the bar that could not be
    # measured at all. This is its sibling one step along: when SOME groups carry
    # a measured ECE and others were dropped (min_group_size, 30 by default,
    # while a per-group CURVE needs only 10), `max_group_disparity` is the
    # max-min spread over what is left, which is a LOWER BOUND, and a finite bar
    # draws it as the whole disparity. Measured on 135 rows with groups m=60,
    # f=60, x=15:
    #
    #   the By Group panels drew all THREE curves, x visibly the worst
    #     (legend: m ECE 0.095, f ECE 0.130, x ECE 0.182)
    #   the ECE Disparity bar read 0.0351, UNDER the red 0.05 target line,
    #     because x was excluded from it (n_groups_compared 2 of 3)
    #
    # so the one group that puts the disparity OVER target (0.182 - 0.095 =
    # 0.087) was named in the legend of the panel beside a bar saying the model
    # is inside it, with nothing on the chart connecting the two. The exclusion
    # is in metadata['excluded_groups'] and in a warning at build time; the chart
    # is the surface a reader meets.
    disparity_index = metrics_labels.index("ECE Disparity")

    def _mark_partial(position, value, result, series: str) -> None:
        excluded = list((getattr(result, "metadata", None) or {}).get("excluded_groups") or [])
        if not excluded or not np.isfinite(value):
            return
        n_measured = len(getattr(result, "group_values", None) or {})
        ax5.annotate(
            f"{series}{n_measured} of {n_measured + len(excluded)} groups; "
            f"{', '.join(str(g) for g in excluded)} excluded",
            xy=(position, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            style="italic",
            color="#8A6D3B",
            rotation=90,
        )

    if has_calibrated:
        ece_after = expected_calibration_error(y_true, calibrated_probs, protected_attr)
        after_values = [ece_after.overall_value, ece_after.max_group_disparity]

        x = np.arange(len(metrics_labels))
        width = 0.35

        ax5.bar(x - width / 2, before_values, width, label="Before", color="#d62728", alpha=0.8)
        ax5.bar(x + width / 2, after_values, width, label="After", color="#2ca02c", alpha=0.8)
        # Without these the grouped bars carry bare numeric ticks, so neither
        # bar is identifiable and the annotation above has nothing to attach to
        # in the reader's eye.
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_labels)
        _mark_unmeasured(x - width / 2, before_values, "before: ")
        _mark_unmeasured(x + width / 2, after_values, "after: ")
        _mark_partial(
            x[disparity_index] - width / 2, before_values[disparity_index], ece_before, "before: "
        )
        _mark_partial(
            x[disparity_index] + width / 2, after_values[disparity_index], ece_after, "after: "
        )
        ax5.set_title("Calibration Improvement", fontsize=12)
        ax5.legend()
    else:
        ax5.bar(metrics_labels, before_values, color="#1f77b4", alpha=0.8)
        _mark_unmeasured(np.arange(len(metrics_labels)), before_values, "")
        _mark_partial(disparity_index, before_values[disparity_index], ece_before, "")
        ax5.set_title("Calibration Metrics", fontsize=12)

    ax5.set_ylabel("Value", fontsize=11)
    ax5.tick_params(axis="x", rotation=15)
    ax5.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="Target")
    ax5.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Calibration Analysis Dashboard", fontsize=16, fontweight="bold", y=1.02)

    return fig
