"""
Visualization functions for fairness audit reports.

This module provides professional, publication-ready visualizations of
fairness metrics, group comparisons, and statistical validation results.

Supports:
- Plotly (recommended): Modern, interactive visualizations
- Matplotlib: Static, publication-quality figures

Example:
    >>> from vfairness import FairnessAnalyzer
    >>> from vfairness.visualization import plot_fairness_report, create_fairness_dashboard
    >>>
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
    >>> report = analyzer.get_report(include_ci=True)
    >>>
    >>> # Interactive Plotly dashboard
    >>> fig = create_fairness_dashboard(report)
    >>> fig.show()
    >>>
    >>> # Static matplotlib report
    >>> plot_fairness_report(report, style='academic')
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..._triage import is_measured
from ._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)

# Professional Color Palettes - Carefully Designed for Data Visualization
#
# Design principles:
# - Colors optimized for readability and accessibility (WCAG AA compliant)
# - Harmonious color combinations using color theory
# - Distinct hues for categorical data
# - Semantic colors (success=green, danger=red, warning=amber)
# - Consistent saturation and lightness levels
#
# Each palette includes:
# - primary: Main brand/accent color
# - success/warning/danger: Semantic status colors
# - info: Informational/neutral accent
# - neutral: Muted color for secondary elements
# - background/grid/text: Base layout colors
# - groups: 8 distinct colors for categorical data
# - accent1/accent2: Additional accent colors
# - gradient: Tuple of colors for continuous scales

PALETTES = {
    # Academic: Clean, professional, suitable for papers and presentations
    # Based on a sophisticated slate-blue palette with muted tones
    "academic": {
        "primary": "#1e3a5f",  # Deep navy blue
        "secondary": "#3d5a80",  # Medium slate blue
        "success": "#2a9d8f",  # Teal green
        "warning": "#e9c46a",  # Warm gold
        "danger": "#e76f51",  # Coral red
        "info": "#457b9d",  # Steel blue
        "neutral": "#6c757d",  # Neutral gray
        "light": "#adb5bd",  # Light gray
        "background": "#ffffff",
        "background_alt": "#f8f9fa",
        "grid": "#e9ecef",
        "text": "#212529",
        "text_muted": "#6c757d",
        "accent1": "#264653",  # Dark teal
        "accent2": "#f4a261",  # Sandy orange
        "groups": [
            "#457b9d",  # Steel blue
            "#2a9d8f",  # Teal
            "#e9c46a",  # Gold
            "#e76f51",  # Coral
            "#264653",  # Dark teal
            "#8ab17d",  # Sage green
            "#a8dadc",  # Light blue
            "#f4a261",  # Sandy orange
        ],
        "gradient": ("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"),
    },
    # Business: Corporate, trustworthy, Bloomberg/McKinsey-inspired
    # Deep blues and grays with strategic accent colors
    "business": {
        "primary": "#0d1b2a",  # Very dark blue
        "secondary": "#1b3a4b",  # Dark blue
        "success": "#2d6a4f",  # Forest green
        "warning": "#f77f00",  # Orange
        "danger": "#d62828",  # Strong red
        "info": "#003566",  # Navy blue
        "neutral": "#495057",  # Dark gray
        "light": "#adb5bd",  # Light gray
        "background": "#ffffff",
        "background_alt": "#f8f9fa",
        "grid": "#dee2e6",
        "text": "#1a1a2e",
        "text_muted": "#6c757d",
        "accent1": "#003049",  # Prussian blue
        "accent2": "#fcbf49",  # Sunflower yellow
        "groups": [
            "#003566",  # Navy
            "#2d6a4f",  # Forest green
            "#f77f00",  # Orange
            "#d62828",  # Red
            "#003049",  # Prussian blue
            "#40916c",  # Medium green
            "#fcbf49",  # Yellow
            "#6c757d",  # Gray
        ],
        "gradient": ("#003049", "#003566", "#0077b6", "#00b4d8", "#90e0ef"),
    },
    # Modern: Fresh, tech-forward, inspired by modern design systems
    # Vibrant yet balanced colors with good contrast
    "modern": {
        "primary": "#5e60ce",  # Vibrant purple
        "secondary": "#7b2cbf",  # Deep purple
        "success": "#06d6a0",  # Mint green
        "warning": "#ffd60a",  # Bright yellow
        "danger": "#ef476f",  # Pink-red
        "info": "#4cc9f0",  # Cyan
        "neutral": "#64748b",  # Slate gray
        "light": "#cbd5e1",  # Light slate
        "background": "#ffffff",
        "background_alt": "#f8fafc",
        "grid": "#e2e8f0",
        "text": "#0f172a",
        "text_muted": "#64748b",
        "accent1": "#7209b7",  # Vivid purple
        "accent2": "#3a0ca3",  # Deep indigo
        "groups": [
            "#5e60ce",  # Purple
            "#06d6a0",  # Mint
            "#4cc9f0",  # Cyan
            "#ef476f",  # Pink-red
            "#ffd60a",  # Yellow
            "#7209b7",  # Vivid purple
            "#ff9f1c",  # Orange
            "#2ec4b6",  # Teal
        ],
        "gradient": ("#3a0ca3", "#5e60ce", "#7b2cbf", "#9d4edd", "#c77dff"),
    },
    # Dark: Elegant dark theme for dashboards and presentations
    # High contrast with carefully selected accent colors
    "dark": {
        "primary": "#818cf8",  # Light indigo
        "secondary": "#a78bfa",  # Light purple
        "success": "#4ade80",  # Light green
        "warning": "#fbbf24",  # Amber
        "danger": "#f87171",  # Light coral
        "info": "#38bdf8",  # Light blue
        "neutral": "#94a3b8",  # Light slate
        "light": "#64748b",  # Slate
        "background": "#0f172a",  # Very dark blue
        "background_alt": "#1e293b",  # Dark slate
        "grid": "#334155",  # Slate 700
        "text": "#f1f5f9",  # Very light slate
        "text_muted": "#94a3b8",  # Light slate
        "accent1": "#c084fc",  # Light violet
        "accent2": "#22d3ee",  # Cyan
        "groups": [
            "#818cf8",  # Indigo
            "#4ade80",  # Green
            "#38bdf8",  # Blue
            "#f87171",  # Red
            "#fbbf24",  # Amber
            "#a78bfa",  # Purple
            "#22d3ee",  # Cyan
            "#fb7185",  # Pink
        ],
        "gradient": ("#312e81", "#4338ca", "#6366f1", "#818cf8", "#a5b4fc"),
    },
    # Nature: Earthy, organic, calming tones
    # Inspired by natural landscapes
    "nature": {
        "primary": "#2d6a4f",  # Forest green
        "secondary": "#40916c",  # Medium green
        "success": "#52b788",  # Light green
        "warning": "#dda15e",  # Tan
        "danger": "#bc4749",  # Brick red
        "info": "#5fa8d3",  # Sky blue
        "neutral": "#6b705c",  # Olive gray
        "light": "#a5a58d",  # Sage
        "background": "#fefae0",  # Cream
        "background_alt": "#f5f0e1",
        "grid": "#e9e5d6",
        "text": "#283618",  # Dark green
        "text_muted": "#6b705c",
        "accent1": "#606c38",  # Olive
        "accent2": "#dda15e",  # Tan
        "groups": [
            "#2d6a4f",  # Forest
            "#52b788",  # Light green
            "#5fa8d3",  # Sky blue
            "#bc4749",  # Brick
            "#dda15e",  # Tan
            "#606c38",  # Olive
            "#a5a58d",  # Sage
            "#40916c",  # Medium green
        ],
        "gradient": ("#283618", "#2d6a4f", "#40916c", "#52b788", "#74c69d"),
    },
    # Colorblind-safe: Optimized for deuteranopia and protanopia
    # Uses blue-orange diverging scheme
    "accessible": {
        "primary": "#0077b6",  # Strong blue
        "secondary": "#023e8a",  # Dark blue
        "success": "#0096c7",  # Cyan
        "warning": "#f48c06",  # Orange
        "danger": "#d00000",  # Pure red
        "info": "#48cae4",  # Light cyan
        "neutral": "#6c757d",  # Gray
        "light": "#adb5bd",  # Light gray
        "background": "#ffffff",
        "background_alt": "#f8f9fa",
        "grid": "#e9ecef",
        "text": "#212529",
        "text_muted": "#6c757d",
        "accent1": "#03045e",  # Very dark blue
        "accent2": "#e85d04",  # Deep orange
        "groups": [
            "#0077b6",  # Blue
            "#f48c06",  # Orange
            "#48cae4",  # Light cyan
            "#dc2f02",  # Red-orange
            "#023e8a",  # Dark blue
            "#ffba08",  # Yellow
            "#6c757d",  # Gray
            "#0096c7",  # Medium cyan
        ],
        "gradient": ("#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"),
    },
}

StyleType = Literal["academic", "business", "modern", "dark", "nature", "accessible"]

#: The palette an unknown style falls back to. Named once, because the fallback
#: is stated in three places (the warning, the returned palette and the label a
#: figure carries) and a figure that names a different one than it draws is the
#: defect this constant exists to prevent.
_FALLBACK_STYLE = "modern"


# Dependency Checking


def _check_plotly():
    """Check if Plotly is available."""
    try:
        import plotly.express as px  # noqa: F401  # availability probe
        import plotly.graph_objects as go  # noqa: F401  # availability probe

        return True
    except ImportError:
        return False


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401  # availability probe

        return True
    except ImportError:
        return False


def _get_palette(style: StyleType = "modern") -> Dict[str, Any]:
    """Get color palette for the specified style, saying so if it is not one.

    BGL3 evaluation-4, 2026-09-27. The fallback was silent, and every figure in
    this module titles itself from the style the CALLER asked for, not from the
    palette it drew. Measured before this warning:
    ``preview_palette("colorblind_safe_v2")`` returned a figure titled
    "Colorblind_Safe_V2 Color Palette" whose swatches were the modern palette,
    byte-identical to ``preview_palette("modern")``, with no warning anywhere.
    A reader checking which colours a named style uses was shown another
    style's, under the name they asked about, and a typo'd "accessible" is the
    case that costs something.

    The warning sits here rather than in ``preview_palette`` because every
    style-taking function in this module resolves through this one line, and
    fixing the one caller that was probed would move the silence next door.
    """
    if style not in PALETTES:
        warnings.warn(
            f"_get_palette: {style!r} is not a known visualization style "
            f"(known: {sorted(PALETTES)}). Falling back to {_FALLBACK_STYLE!r}, so this "
            f"figure shows the {_FALLBACK_STYLE.upper()} palette while any title built "
            f"from the requested name will still say {style!r}. Nothing here reports the "
            f"colours of {style!r}, which does not exist.",
            UserWarning,
            # Point at the caller that chose the style, not at this helper.
            stacklevel=3,
        )
    return PALETTES.get(style, PALETTES[_FALLBACK_STYLE])


def _palette_drawn(style: StyleType) -> str:
    """The name of the palette :func:`_get_palette` ACTUALLY returns for ``style``.

    One line rather than two literals, so the fallback name in the warning, in
    the returned palette and in any figure label can never drift apart.
    """
    return str(style) if style in PALETTES else _FALLBACK_STYLE


def _palette_preview_labels(style: StyleType) -> Tuple[str, str]:
    """``(headline, disclosure)`` for a figure whose SUBJECT is a palette.

    BGL5 A-evaluation-1, 2026-09-27. The warning added to :func:`_get_palette`
    was a real disclosure in a channel the artifact does not carry. Measured
    after that fix and before this one: ``preview_palette("colorblind_safe_v2")``
    warned, and returned a figure titled
    ``'<b>Colorblind_Safe_V2 Color Palette</b>'`` over the MODERN swatches whose
    whole content contained none of "not a known" (False), "falling back"
    (False), "modern palette" (False) or "does not exist" (False). A saved PNG,
    an exported HTML and a notebook with filters set all carry the title and none
    of them carry the warning, so the artifact still reported the colours of a
    style that does not exist. Now the headline names the palette the swatches
    ARE and the disclosure travels in the figure itself:
    ``("Modern Color Palette", "requested 'colorblind_safe_v2', which is not a
    known style: these are the MODERN colours, so nothing here shows
    'colorblind_safe_v2'")``.

    A known style returns its OWN name with an empty disclosure, e.g.
    ``("Academic Color Palette", "")``, which is character for character what the
    title said before, so the six documented styles are untouched. The two parts
    are returned separately because each branch of :func:`preview_palette` has
    its own markup for a second line.
    """
    drawn = _palette_drawn(style)
    headline = f"{drawn.title()} Color Palette"
    if drawn == str(style):
        return headline, ""
    return headline, (
        f"requested {str(style)!r}, which is not a known style: these are the "
        f"{drawn.upper()} colours, so nothing here shows {str(style)!r}"
    )


# Could-not-check rendering
#
# Wording used wherever a figure has to show that nothing was measured. Kept in
# one place so the dashboard, the metrics chart and the full report cannot drift
# into telling a reader three different things about the same absent result.
NOT_ASSESSABLE_TEXT = "NOT ASSESSABLE"
NOT_ASSESSABLE_REASON = "No metric could be measured: this figure reports nothing about fairness."
NOT_MEASURED_TICK = "n/a"
COULD_NOT_CHECK_TEXT = "could not check"

#: gid stamped on every matplotlib artist that marks a could-not-check metric.
#: It is the OPPOSITE of an acceptable region, and a reader (or a test reading
#: the rendered geometry) has to be able to tell the two apart by more than
#: their colour.
COULD_NOT_CHECK_GID = "vfairness-could-not-check"
#: gid stamped on every acceptable-region shading artist.
ACCEPTABLE_REGION_GID = "vfairness-acceptable-region"


def _is_measured(value: Any) -> bool:
    """True only for a real, finite number.

    A metric that is None or NaN was NOT computed. Treating it as a number is
    how a could-not-check result gets drawn as a verdict.

    READINESS-6, 2026-09-10. Delegated to ``_triage.is_measured``, which is the
    module whose docstring says it exists so this question is answered once.
    Before this there were six implementations of it and they disagreed on FIVE
    of nine inputs, including the canonical one classifying a finite
    ``np.float32`` as NOT measured. The local reasoning above is why THIS call
    site cares; the rule itself lives in one place.
    """
    return is_measured(value)


def _ci_field(ci_data: Any, key: str) -> Any:
    """Read ONE field out of one ``metrics_with_ci`` entry, absence spelled as absence.

    Returns the value only when it is a real, finite number, and ``np.nan``
    otherwise. ``np.nan`` is this module's spelling of "not measured": every
    consumer already routes through :func:`_is_measured`, so an absent field
    stays absent all the way to the mark the reader sees.

    THIS EXISTS BECAUSE THE SAME KEY WAS READ WITH THREE DIFFERENT DEFAULTS IN
    ONE UNIT, and the fabricating ones outlived the fix. Measured 2026-09-30 on
    ``plot_confidence_intervals``: the range loop and the drawing loop read
    ``get("point_estimate", np.nan)`` behind an ``isinstance`` guard, while the
    acceptable-region loop twenty lines between them read
    ``get("point_estimate", 0)`` with no guard at all. The consequences of that
    one default, all three reproduced by execution:

    * a row whose point estimate is ABSENT was shaded green under the label
      "Acceptable range", in an extent byte-identical to a row measured at
      0.420, while the row label beside it read ``n/a``. The same absent value
      spelled ``None`` or ``NaN`` was correctly refused, so two of the three
      doors into the same fabrication were shut and the third was open;
    * a tuple-shaped entry ``(0.42, 0.31, 0.53)``, which is the shape
      ``compute_metric_with_ci`` returns and a caller passes straight through,
      raised ``AttributeError: 'tuple' object has no attribute 'get'`` from that
      line although both neighbouring loops shape-guarded it;
    * ``lower = ci_data.get("lower_bound", pe)`` in the drawing loop defaulted an
      absent BOUND to the point estimate, so a report carrying only
      ``point_estimate = 0.42`` printed the 95% interval ``[0.420, 0.420]``, a
      perfect precision nobody computed, with no warning.

    The same three defaults were live in the dashboard twin ``_add_ci_panel``,
    where an absent point estimate put a diamond at x = 0 (perfect parity on a
    disparity axis) with hover text "Point Estimate: 0.0000" and the inverted
    zero-width interval "[0.0200, 0.0000]".

    One reader, so the defaults cannot drift again.
    """
    if not isinstance(ci_data, dict):
        return np.nan
    value = ci_data.get(key, np.nan)
    return value if _is_measured(value) else np.nan


def _bar_state_color(metric_name: str, value: Any, threshold: Any, palette: Dict[str, Any]) -> str:
    """Colour for one metric's bar or marker: pass, fail, or could-not-check.

    Three states, never two. The direction comes from
    :func:`._metric_direction.check_threshold`, never from a local name test
    here: every previous copy of the rule in this module was an
    ``endswith("_ratio")`` duplicate of what that helper owns, and duplicated
    rules drift (the cali-BRATIO-n incident, CLAUDE.md).

    A value that was never measured, a threshold that was never set, and a
    metric whose better-direction is unknown all land on the neutral colour.
    Painting any of them green reports a pass this run never earned; painting
    them red reports a violation it never observed.
    """
    if not _is_measured(value) or not _is_measured(threshold):
        return palette.get("neutral", palette["text"])
    outcome, _reason = check_threshold(metric_name, float(value), float(threshold))
    if outcome is ThresholdOutcome.PASS:
        return palette["success"]
    if outcome is ThresholdOutcome.FAIL:
        return palette["danger"]
    return palette.get("neutral", palette["text"])


def _acceptable_region(
    metric_name: str, value: Any, threshold: Any
) -> Tuple[Optional[Tuple[float, float]], Tuple[float, ...], str]:
    """Where the acceptable region for ONE metric lies, in that metric's direction.

    Returns ``(band, marker_values, could_not_check_reason)``:

    * ``band`` is the ``(low, high)`` interval a compliant value falls in, or
      ``None`` when there is no region to draw;
    * ``marker_values`` are the threshold positions to draw a rule line at;
    * the reason is empty when the region is real, and otherwise says, in the
      words the figure prints, why nothing could be checked.

    THE FOURTH INSTANCE OF THE DIRECTION BUG LIVED HERE, IN GEOMETRY.
    ``plot_fairness_metrics`` drew ONE hardcoded ``ax.axhspan(-0.1, 0.1)``
    labelled "Acceptable range" across EVERY metric on the axis, and the
    confidence-interval charts drew the same band vertically. For the ratio
    family that region is inverted: ``demographic_parity_ratio = 0.000``, the
    maximal four-fifths violation, falls INSIDE +/-0.1 and reads as the only
    compliant bar on the chart, while a healthy 0.946 sits far outside it. Both
    metrics come out of ``classification_fairness_report`` on ordinary input,
    so the chart was inverting real results. Colour-based and text-based pins
    could not see it because the defect was the SHAPE.

    A region is therefore per metric, and its direction is resolved by
    :func:`._metric_direction.metric_direction`. An unknown direction, an
    unusable threshold and an unmeasured value get NO region at all: an
    acceptable region is an assertion that the value was checked and found
    compliant, and none of the three was checked.

    THE THIRD OF THOSE THREE WAS ONLY IN THE DOCSTRING until 2026-09-29. The
    threshold test and the unknown-direction test were real, but the value was
    tested nowhere: ``LOWER_IS_BETTER`` returned ``(-bound, bound)`` whatever the
    value was, so a metric whose value is NaN was shaded green under the label
    "Acceptable range" on both the bar chart and the interval chart. Measured on
    the single-group report from the public producer, whose demographic parity
    difference is NaN against a 0.10 threshold. The test now sits ABOVE the
    direction dispatch, where the other two already were, because a value that was
    never measured cannot be inside any region regardless of direction.
    """
    if not _is_measured(threshold):
        return None, (), f"{COULD_NOT_CHECK_TEXT}: no usable threshold"
    if not _is_measured(value):
        return None, (), f"{COULD_NOT_CHECK_TEXT}: the value was not measured"
    bound = float(threshold)
    direction = metric_direction(metric_name)

    if direction is MetricDirection.HIGHER_IS_BETTER:
        # A parity ratio passes at or ABOVE its minimum (the four-fifths rule),
        # so the region runs upward from the bound, never symmetrically around
        # zero. Parity itself is 1.0, and the region is extended past it when a
        # value overshoots so the shading never stops short of the bar it grades.
        # The value is known to be measured here: the guard above the dispatch
        # returned already if it was not.
        top = max(1.0, bound, float(value))
        return (bound, top), (bound,), ""

    if direction is MetricDirection.LOWER_IS_BETTER:
        # A violation magnitude passes at or below its bound, in either sign.
        return (-bound, bound), (bound, -bound), ""

    return None, (), f"{COULD_NOT_CHECK_TEXT}: no known better-direction"


def _ci_state_color(
    metric_name: str,
    point: Any,
    lower: Any,
    upper: Any,
    threshold: Any,
    palette: Dict[str, Any],
) -> str:
    """Colour for one confidence interval, in that metric's direction.

    Both CI charts used to test ``name.endswith("_ratio")`` themselves and then
    paint every ratio the same informational blue, which says nothing at all
    about whether the interval clears the four-fifths minimum. The direction
    comes from the shared helper, and a ratio is now graded like anything else:
    the whole interval below the minimum is a failure, an interval straddling
    it is the caution state, and an interval entirely at or above it passes.
    """
    values = (point, lower, upper)
    if not all(_is_measured(v) for v in values) or not _is_measured(threshold):
        return palette.get("neutral", palette["text"])

    bound = float(threshold)
    direction = metric_direction(metric_name)

    if direction is MetricDirection.HIGHER_IS_BETTER:
        if float(upper) < bound:
            return palette["danger"]
        if float(lower) < bound:
            return palette["warning"]
        return palette["success"]

    if direction is MetricDirection.LOWER_IS_BETTER:
        if float(upper) < -bound or float(lower) > bound:
            return palette["danger"]
        if float(lower) > 0 or float(upper) < 0:
            # Statistically distinguishable from zero, but inside the bound.
            return palette["warning"]
        return palette["success"]

    return palette.get("neutral", palette["text"])


def _score_display(report: Dict[str, Any], palette: Dict[str, Any]) -> Tuple[str, str, bool]:
    """Return ``(score_text, color, is_assessed)`` for the overall fairness score.

    ``score_text`` is either a formatted percentage or ``"NOT ASSESSABLE"``.
    Callers compose their own surrounding wording; every caller must branch on
    ``is_assessed`` rather than printing the text into a sentence that only
    makes sense for a number.

    CRITICAL (fail closed, and do NOT substitute a number): ``fairness_score``
    is Optional. It is None when the run was NOT ASSESSABLE, i.e. no metric
    could be measured. ``.get("fairness_score", 0)`` does not protect anything
    here, because the key EXISTS with the value None, so the default never fires
    and the next comparison raised
    ``TypeError: '>=' not supported between instances of 'NoneType' and 'float'``
    in all three entry points.

    Substituting a number would be worse than the crash it replaces. A chart
    reading "Overall Score: 0%" is a measured total failure to everyone who
    looks at it, and 100% is a clean certificate. Neither was measured. So the
    figure gets the third state, in words, in the place the score would have
    been.
    """
    assessment = report.get("assessment", {})
    score = assessment.get("fairness_score") if isinstance(assessment, dict) else None

    # The explicit None test is what a type checker can follow; _is_measured is
    # the one that also rejects NaN, a bool and a non-number.
    if score is None or not _is_measured(score):
        return NOT_ASSESSABLE_TEXT, palette.get("neutral", palette["text"]), False

    # _is_measured has established a real, finite number; the local float is
    # what tells a type checker so. Do NOT replace the guard above with a
    # coercion: None here means the run was NOT ASSESSABLE, and float(None)
    # would raise where the docstring above says the figure must show the third
    # state instead.
    measured_score = float(score)
    if measured_score >= 0.7:
        color = palette["success"]
    elif measured_score >= 0.5:
        color = palette["warning"]
    else:
        color = palette["danger"]
    return f"{measured_score:.0%}", color, True


# Plotly Visualizations (Modern, Interactive)


def create_fairness_dashboard(
    report: Dict[str, Any],
    style: StyleType = "modern",
    height: int = 800,
    show_annotations: bool = True,
) -> Any:
    """
    Create an interactive fairness dashboard using Plotly.

    This creates a professional, interactive multi-panel dashboard suitable
    for business presentations and academic publications.

    Args:
        report: Report dict from classification_fairness_report or regression_fairness_report
        style: Visual style ('academic', 'business', 'modern', 'dark')
        height: Total height in pixels
        show_annotations: Whether to show value annotations

    Returns:
        Plotly Figure object

    Example:
        >>> report = classification_fairness_report(y_true, y_pred, gender, include_ci=True)
        >>> fig = create_fairness_dashboard(report, style='business')
        >>> fig.show()
        >>> fig.write_html('fairness_report.html')
    """
    if not _check_plotly():
        raise ImportError("Interactive dashboards require plotly. Install with: pip install plotly")

    from plotly.subplots import make_subplots

    palette = _get_palette(style)
    has_ci = "metrics_with_ci" in report
    has_effects = "effect_sizes" in report and report["effect_sizes"]

    # Determine layout
    n_rows = 2
    n_cols = 2

    subplot_titles = ["Fairness Metrics", "Group Comparison"]
    if has_effects:
        subplot_titles.append("Effect Sizes")
    if has_ci:
        subplot_titles.append("Confidence Intervals")

    # Pad titles if needed
    while len(subplot_titles) < 4:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Panel 1: Fairness Metrics
    _add_metrics_panel(fig, report, palette, row=1, col=1, show_annotations=show_annotations)

    # Panel 2: Group Comparison
    _add_groups_panel(fig, report, palette, row=1, col=2, show_annotations=show_annotations)

    # Panel 3: Effect Sizes or empty
    if has_effects:
        _add_effects_panel(fig, report, palette, row=2, col=1)

    # Panel 4: Confidence Intervals or empty
    if has_ci:
        _add_ci_panel(fig, report, palette, row=2, col=2)

    # Overall styling with improved design
    task_type = report.get("task_type", "Classification").title()
    is_dark = style == "dark"

    # Score line, or an explicit could-not-check state. See _score_display: the
    # score is Optional and this used to raise TypeError on None.
    score_text, score_color, score_is_assessed = _score_display(report, palette)
    score_line = (
        f"Overall Score: {score_text}"
        if score_is_assessed
        else f"{score_text}: no metric could be measured"
    )

    fig.update_layout(
        title={
            "text": (
                f"<b>{task_type} Fairness Audit Report</b><br>"
                f"<span style='color:{score_color};font-size:14px'>{score_line}</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 22,
                "color": palette["text"],
                "family": "Inter, Segoe UI, Arial, sans-serif",
            },
        },
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=palette.get("text_muted", palette["text"])),
            bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.8)",
            bordercolor=palette["grid"],
            borderwidth=1,
        ),
        plot_bgcolor=palette.get("background_alt", palette["background"]),
        paper_bgcolor=palette["background"],
        font={
            "color": palette["text"],
            "family": "Inter, Segoe UI, SF Pro Display, Arial, sans-serif",
            "size": 12,
        },
        margin=dict(t=110, b=90, l=60, r=60),
        hoverlabel=dict(
            bgcolor=palette.get("background_alt", palette["background"]),
            font_size=12,
            font_family="Inter, Arial, sans-serif",
            bordercolor=palette["grid"],
        ),
    )

    # Style all axes with improved appearance
    for i in range(1, 5):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=palette["grid"],
            griddash="dot" if is_dark else "solid",
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor=palette.get("light", palette["neutral"]),
            linecolor=palette.get("light", palette["neutral"]),
            linewidth=1,
            tickfont=dict(size=10, color=palette.get("text_muted", palette["text"])),
            title_font=dict(size=11, color=palette["text"]),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=palette["grid"],
            griddash="dot" if is_dark else "solid",
            linecolor=palette.get("light", palette["neutral"]),
            linewidth=1,
            tickfont=dict(size=10, color=palette.get("text_muted", palette["text"])),
            title_font=dict(size=11, color=palette["text"]),
            row=row,
            col=col,
        )

    return fig


def _add_metrics_panel(fig, report, palette, row, col, show_annotations=True):
    """Add fairness metrics bar chart to dashboard with improved styling."""
    import plotly.graph_objects as go

    metrics = report.get("metrics", {})
    thresholds = report.get("thresholds_used", {})
    metrics_with_ci = report.get("metrics_with_ci", {})

    if not metrics:
        return

    metric_names = list(metrics.keys())
    values = [metrics[m] for m in metric_names]

    # Better formatted display names
    display_names = []
    for m in metric_names:
        name = m.replace("_difference", "").replace("_ratio", " Ratio")
        name = name.replace("_", " ").title()
        display_names.append(name)

    # Determine colors based on pass/fail with opacity
    colors = []
    line_colors = []
    for m, v in zip(metric_names, values):
        threshold = thresholds.get(m, 0.1)
        # Three states, never two. A non-finite metric was NOT MEASURED, so it
        # is neither a pass nor a violation: painting it danger red would report
        # a failure this run never observed, exactly as painting it green would
        # report a pass. It gets the neutral colour and an "n/a" label. The
        # direction now comes from the shared helper (see _bar_state_color), so
        # this site cannot drift from the one the gate and the report use.
        base_color = _bar_state_color(m, v, threshold, palette)
        colors.append(base_color)
        line_colors.append(base_color)

    # Add bar chart with improved styling
    fig.add_trace(
        go.Bar(
            x=display_names,
            y=values,
            marker_color=colors,
            marker_line_color=line_colors,
            marker_line_width=1.5,
            marker_opacity=0.85,
            text=[f"{v:.3f}" if _is_measured(v) else NOT_MEASURED_TICK for v in values]
            if show_annotations
            else None,
            textposition="outside",
            textfont=dict(size=11, color=palette["text"]),
            name="Metrics",
            showlegend=False,
            hovertemplate=("<b>%{x}</b><br>Value: %{y:.4f}<br><extra></extra>"),
        ),
        row=row,
        col=col,
    )

    # A non-finite metric draws no bar at all, and an empty slot on a bar chart
    # reads as a measured zero disparity. Mark every unmeasured metric in the
    # place its bar would have been, so the gap is stated rather than implied.
    for i, v in enumerate(values):
        if not _is_measured(v):
            fig.add_annotation(
                x=display_names[i],
                y=0,
                text=f"<b>{NOT_MEASURED_TICK}</b><br>not measured",
                showarrow=False,
                font=dict(size=10, color=palette.get("neutral", palette["text"])),
                row=row,
                col=col,
            )

    # Add error bars if CI available with improved styling
    if metrics_with_ci:
        ci_lower = []
        ci_upper = []
        for m in metric_names:
            if m in metrics_with_ci:
                ci_data = metrics_with_ci[m]
                ci_lower.append(metrics[m] - ci_data.get("lower_bound", metrics[m]))
                ci_upper.append(ci_data.get("upper_bound", metrics[m]) - metrics[m])
            else:
                ci_lower.append(0)
                ci_upper.append(0)

        fig.add_trace(
            go.Scatter(
                x=display_names,
                y=values,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=ci_upper,
                    arrayminus=ci_lower,
                    color=palette.get("text_muted", palette["neutral"]),
                    thickness=2,
                    width=5,
                ),
                mode="markers",
                marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    # Shaded acceptable region and threshold markers PER METRIC, drawn from
    # the actual thresholds dict (defaulted where missing). The old single
    # hardcoded +/-0.1 band contradicted per-metric thresholds: it painted a
    # 0.08 equal-opportunity gap (threshold 0.05, FAIL) as acceptable and
    # made the 0.80-minimum DP ratio look like a violation. Categorical bar
    # axes accept numeric coordinates as category indices, so each metric
    # gets a band exactly as wide as its bar and as tall as its own
    # threshold. Which way the band opens is _acceptable_region's answer, and a
    # metric it cannot grade gets NO band, only the stated reason.
    for i, m in enumerate(metric_names):
        threshold = thresholds.get(m, 0.1)
        x0, x1 = i - 0.4, i + 0.4
        # Direction from the shared helper, never from a local suffix test.
        band, marker_ys, reason = _acceptable_region(m, values[i], threshold)
        if band is None:
            # No region, and the gap is stated rather than left to be read as
            # "nothing to worry about here".
            fig.add_annotation(
                x=i,
                y=0,
                yanchor="top",
                text=f"<b>{COULD_NOT_CHECK_TEXT}</b><br>{reason.split(': ', 1)[-1]}",
                showarrow=False,
                font=dict(size=9, color=palette.get("neutral", palette["text"])),
                row=row,
                col=col,
            )
            continue
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=band[0],
            y1=band[1],
            fillcolor=palette["success"],
            opacity=0.08,
            line_width=0,
            row=row,
            col=col,
        )
        for ly in marker_ys:
            fig.add_shape(
                type="line",
                x0=x0,
                x1=x1,
                y0=ly,
                y1=ly,
                line=dict(dash="dash", color=palette["warning"], width=1.5),
                opacity=0.7,
                row=row,
                col=col,
            )

    fig.update_yaxes(title_text="Metric Value", row=row, col=col)


def _add_groups_panel(fig, report, palette, row, col, show_annotations=True):
    """Add group comparison panel to dashboard with improved styling."""
    import plotly.graph_objects as go

    group_stats = report.get("group_stats", {})
    if not group_stats:
        return

    groups = list(group_stats.keys())

    # Determine which metric to show
    first_group = list(group_stats.values())[0]
    if "positive_rate" in first_group:
        metric = "positive_rate"
        metric_label = "Positive Rate"
    elif "mae" in first_group:
        metric = "mae"
        metric_label = "MAE"
    else:
        return

    # np.nan, never 0, for BOTH: this is the plotly twin of
    # plot_group_comparison, and it carried the same defect one step worse. A
    # missing metric defaulted to 0 and was drawn as a measured "0.0%" bar, and a
    # missing size was reported as "Sample Size: 0" in the hover.
    values = [group_stats[g].get(metric, np.nan) for g in groups]
    sizes = [group_stats[g].get("size", np.nan) for g in groups]
    measured = [float(v) for v in values if _is_measured(v)]
    unmeasured_groups = [g for g, v in zip(groups, values) if not _is_measured(v)]
    if unmeasured_groups:
        warnings.warn(
            f"create_fairness_dashboard: {len(unmeasured_groups)} group(s) have no measured "
            f"{metric} ({', '.join(str(g) for g in unmeasured_groups)}), so they carry a "
            f"'{NOT_MEASURED_TICK} not measured' label and no bar. An absent bar is not a zero "
            "rate, and the average covers only the groups that were measured.",
            UserWarning,
            stacklevel=2,
        )

    # Use improved group colors
    group_colors = palette["groups"][: len(groups)]

    fig.add_trace(
        go.Bar(
            x=groups,
            y=values,
            marker_color=group_colors,
            marker_line_color=group_colors,
            marker_line_width=1.5,
            marker_opacity=0.85,
            text=[
                (f"{v:.1%}" if metric == "positive_rate" else f"{v:.3f}")
                if _is_measured(v)
                else f"{NOT_MEASURED_TICK} not measured"
                for v in values
            ]
            if show_annotations
            else None,
            textposition="outside",
            textfont=dict(size=11, color=palette["text"]),
            name="Groups",
            showlegend=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{metric_label}: %{{y:.4f}}<br>"
                "Sample Size: %{customdata:,}<br>"
                "<extra></extra>"
            ),
            customdata=sizes,
        ),
        row=row,
        col=col,
    )

    # Add overall average line with improved styling. Over the MEASURED groups,
    # and the label says how many that was: an average whose denominator quietly
    # shrank is a claim about groups it never covered.
    if measured:
        avg = float(np.mean(measured))
        avg_text = f"Avg: {avg:.3f}" if metric != "positive_rate" else f"Avg: {avg:.1%}"
        if len(measured) < len(values):
            avg_text += f" ({len(measured)} of {len(values)} groups)"
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color=palette.get("accent1", palette["neutral"]),
            line_width=1.5,
            opacity=0.6,
            annotation_text=avg_text,
            annotation_position="right",
            annotation_font=dict(size=10, color=palette.get("text_muted", palette["text"])),
            row=row,
            col=col,
        )
    else:
        # No line at all rather than a line at NaN, and the reason in words.
        fig.add_annotation(
            text=f"{NOT_ASSESSABLE_TEXT}: no group's {metric} was measured, so there is "
            "no average to draw",
            showarrow=False,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            font=dict(size=10, color=palette.get("neutral", palette["text"])),
            row=row,
            col=col,
        )

    fig.update_yaxes(title_text=metric_label, row=row, col=col)
    fig.update_xaxes(title_text="Group", row=row, col=col)


def _add_effects_panel(fig, report, palette, row, col):
    """Add effect sizes panel to dashboard with improved styling."""
    import plotly.graph_objects as go

    effect_sizes = report.get("effect_sizes", {})
    if not effect_sizes:
        return

    pairs = list(effect_sizes.keys())
    cohens_d = []
    interpretations = []

    for pair in pairs:
        effects = effect_sizes[pair]
        if isinstance(effects, dict):
            # np.nan, never 0: this was `.get(..., 0)`, so a pair carrying neither
            # key was drawn as "d = 0.00" on the dashboard, which is the canvas for
            # a measured absence of effect. It is the plotly twin of
            # plot_effect_sizes and had the same defect.
            d = effects.get("cohens_d_positive_rate", effects.get("cohens_d_predictions", np.nan))
            interpretation = effects.get("interpretation", "unknown")
        else:
            d = np.nan
            interpretation = "unknown"
        cohens_d.append(d)
        interpretations.append(interpretation)

    measured_d = [abs(float(d)) for d in cohens_d if _is_measured(d)]
    unmeasured_pairs = [p for p, d in zip(pairs, cohens_d) if not _is_measured(d)]
    if unmeasured_pairs:
        warnings.warn(
            f"create_fairness_dashboard: {len(unmeasured_pairs)} of {len(pairs)} pair(s) have no "
            f"measured effect size ({', '.join(str(p) for p in unmeasured_pairs)}), labelled "
            f"'{NOT_MEASURED_TICK} not measured' with no bar rather than d = 0.00.",
            UserWarning,
            stacklevel=2,
        )

    display_pairs = [p.replace("_vs_", " vs ").replace("_", " ") for p in pairs]

    # Improved color mapping using palette
    color_map = {
        "negligible": palette["success"],
        "negligible effect": palette["success"],
        "small": palette["warning"],
        "small effect": palette["warning"],
        "medium": palette.get("accent2", "#E67E22"),
        "medium effect": palette.get("accent2", "#E67E22"),
        "large": palette["danger"],
        "large effect": palette["danger"],
    }
    # An unmeasured row takes the neutral colour whatever its interpretation
    # string says: a green "negligible" band under a row nobody measured is the
    # same claim as the 0.00 bar.
    colors = [
        color_map.get(interp, palette["neutral"]) if _is_measured(d) else palette["neutral"]
        for interp, d in zip(interpretations, cohens_d)
    ]

    fig.add_trace(
        go.Bar(
            x=cohens_d,
            y=display_pairs,
            orientation="h",
            marker_color=colors,
            marker_line_color=colors,
            marker_line_width=1.5,
            marker_opacity=0.85,
            text=[
                f"d = {float(d):.2f}" if _is_measured(d) else f"{NOT_MEASURED_TICK} not measured"
                for d in cohens_d
            ],
            textposition="outside",
            textfont=dict(size=10, color=palette["text"]),
            name="Effect Sizes",
            showlegend=False,
            hovertemplate=("<b>%{y}</b><br>Cohen's d: %{x:.3f}<br><extra></extra>"),
        ),
        row=row,
        col=col,
    )

    # Add shaded regions for effect size interpretation. Over the MEASURED values:
    # `max(abs(d) for d in cohens_d)` returns NaN as soon as one value is NaN, and
    # a NaN axis range is not a range at all.
    x_max = (max(measured_d) * 1.3) if measured_d else 1.0
    x_max = max(x_max, 1.0)

    # Negligible region
    fig.add_vrect(
        x0=-0.2, x1=0.2, fillcolor=palette["success"], opacity=0.08, line_width=0, row=row, col=col
    )

    # Add subtle threshold lines
    for threshold in [0.2, 0.5, 0.8]:
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=palette.get("light", palette["grid"]),
            line_width=1,
            opacity=0.6,
            row=row,
            col=col,
        )
        fig.add_vline(
            x=-threshold,
            line_dash="dot",
            line_color=palette.get("light", palette["grid"]),
            line_width=1,
            opacity=0.6,
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="Cohen's d (Effect Size)", row=row, col=col)


def _add_ci_panel(fig, report, palette, row, col):
    """Add confidence intervals panel to dashboard with improved styling."""
    import plotly.graph_objects as go

    metrics_with_ci = report.get("metrics_with_ci", {})
    thresholds = report.get("thresholds_used", {})

    if not metrics_with_ci:
        return

    metric_names = list(metrics_with_ci.keys())
    # Better formatted display names
    display_names = []
    for m in metric_names:
        name = m.replace("_difference", "").replace("_ratio", " Ratio")
        name = name.replace("_", " ").title()
        display_names.append(name)

    # Read every number through _ci_field. This panel is the dashboard twin of
    # plot_confidence_intervals and it kept all three of that unit's fabricating
    # defaults after the unit itself was fixed: an absent point estimate put a
    # diamond at x = 0 with hover "Point Estimate: 0.0000" and the inverted
    # zero-width interval "[0.0200, 0.0000]", and no warning anywhere.
    unmeasured_rows = []
    for i, (name, display_name) in enumerate(zip(metric_names, display_names)):
        ci_data = metrics_with_ci[name]
        pe = _ci_field(ci_data, "point_estimate")
        lower = _ci_field(ci_data, "lower_bound")
        upper = _ci_field(ci_data, "upper_bound")
        threshold = thresholds.get(name, 0.1)

        if not _is_measured(pe):
            # No marker at all. There is no honest x for a value that does not
            # exist, and on a disparity axis the old zero was perfect parity.
            unmeasured_rows.append(str(name))
            reason = ""
            metadata = ci_data.get("metadata") if isinstance(ci_data, dict) else None
            if isinstance(metadata, dict) and metadata.get("warning"):
                reason = f" ({metadata['warning']})"
            fig.add_annotation(
                x=0,
                y=display_name,
                text=f"<b>{NOT_MEASURED_TICK}</b> not measured{reason}",
                showarrow=False,
                font=dict(size=9, color=palette.get("neutral", palette["text"])),
                row=row,
                col=col,
            )
            continue

        # Colour by the CI's position in THIS metric's direction (shared helper).
        color = _ci_state_color(name, pe, lower, upper, threshold, palette)

        def _fmt(value: Any) -> str:
            return f"{float(value):.4f}" if _is_measured(value) else NOT_MEASURED_TICK

        fig.add_trace(
            go.Scatter(
                x=[pe],
                y=[display_name],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    # An absent bound draws no whisker rather than one of length
                    # zero: a zero-length whisker reads as a measured interval of
                    # perfect precision.
                    array=[(float(upper) - float(pe)) if _is_measured(upper) else np.nan],
                    arrayminus=[(float(pe) - float(lower)) if _is_measured(lower) else np.nan],
                    color=color,
                    thickness=3,
                    width=8,
                ),
                mode="markers",
                marker=dict(
                    size=14, color=color, symbol="diamond", line=dict(width=2, color="white")
                ),
                name=display_name,
                showlegend=False,
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    f"Point Estimate: {_fmt(pe)}<br>"
                    f"95% CI: [{_fmt(lower)}, {_fmt(upper)}]<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    if unmeasured_rows:
        warnings.warn(
            f"_add_ci_panel: {len(unmeasured_rows)} of {len(metric_names)} metric(s) have no "
            f"measured point estimate ({', '.join(unmeasured_rows)}), so their row carries a "
            f"'{NOT_MEASURED_TICK} not measured' mark and no marker. An absent marker is not a "
            "value at zero.",
            UserWarning,
            stacklevel=2,
        )

    # Acceptable region PER ROW, in each metric's own direction. This was one
    # hardcoded vertical band from -0.1 to +0.1 across every row, the same
    # inverted geometry as the bar chart: a four-fifths ratio of 0.000 fell
    # inside it and read as compliant. Categorical axes take numeric
    # coordinates as row indices, so each row gets its own band.
    for i, name in enumerate(metric_names):
        threshold = thresholds.get(name, 0.1)
        # Through _ci_field: the zero default here shaded the compliance band
        # green for a point estimate nobody measured.
        pe = _ci_field(metrics_with_ci[name], "point_estimate")
        band, marker_xs, reason = _acceptable_region(name, pe, threshold)
        y0, y1 = i - 0.4, i + 0.4
        if band is None:
            fig.add_annotation(
                x=pe if _is_measured(pe) else 0,
                y=i,
                yshift=-18,
                text=f"<b>{COULD_NOT_CHECK_TEXT}</b>: {reason.split(': ', 1)[-1]}",
                showarrow=False,
                font=dict(size=9, color=palette.get("neutral", palette["text"])),
                row=row,
                col=col,
            )
            continue
        fig.add_shape(
            type="rect",
            x0=band[0],
            x1=band[1],
            y0=y0,
            y1=y1,
            fillcolor=palette["success"],
            opacity=0.1,
            line_width=0,
            row=row,
            col=col,
        )
        for lx in marker_xs:
            fig.add_shape(
                type="line",
                x0=lx,
                x1=lx,
                y0=y0,
                y1=y1,
                line=dict(dash="dash", color=palette["warning"], width=1),
                opacity=0.5,
                row=row,
                col=col,
            )

    # Zero line
    fig.add_vline(
        x=0,
        line_color=palette.get("text_muted", palette["neutral"]),
        line_width=1,
        opacity=0.4,
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text="Metric Value (95% CI)", row=row, col=col)


def _radar_label(metric_name: str) -> str:
    """Axis label for one metric on the radar (display only, never a direction test)."""
    return str(metric_name).replace("_", " ").replace("difference", "").strip().title()


def _excluded_note(excluded: Sequence[str]) -> str:
    """The line naming every metric the figure could NOT place, and why."""
    return "Not plotted: " + "; ".join(excluded)


def _empty_radar(go: Any, palette: Dict[str, Any], style: StyleType, excluded: List[str]) -> Any:
    """A radar with no axes, stating why, instead of a radar drawn from leftovers."""
    fig = go.Figure()
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.0]),
            bgcolor=palette.get("background_alt", palette["background"]),
        ),
        title={
            "text": f"<b>Fairness Metrics Overview</b><br>"
            f"<span style='font-size:13px'>{NOT_ASSESSABLE_TEXT}: {NOT_ASSESSABLE_REASON}</span>",
            "x": 0.5,
            "font": {"size": 18, "color": palette["text"], "family": "Inter, Segoe UI, Arial"},
        },
        paper_bgcolor=palette["background"],
        font={"color": palette["text"], "family": "Inter, Segoe UI, Arial, sans-serif"},
        margin=dict(t=100, b=80, l=40, r=40),
    )
    if excluded:
        fig.add_annotation(
            text=_excluded_note(excluded),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.12,
            showarrow=False,
            align="center",
            font=dict(size=10, color=palette.get("text_muted", palette["text"])),
        )
    return fig


def plot_metrics_radar(
    report: Dict[str, Any], style: StyleType = "modern", normalize: bool = True, fill: bool = True
) -> Any:
    """
    Create a radar/spider chart of fairness metrics.

    Excellent for comparing multiple metrics at a glance in presentations.

    Args:
        report: Report dict from fairness report function
        style: Visual style
        normalize: Normalize metrics to 0-1 scale
        fill: Fill the radar area

    Returns:
        Plotly Figure object
    """
    if not _check_plotly():
        raise ImportError("Radar plots require plotly. Install with: pip install plotly")

    import plotly.graph_objects as go

    palette = _get_palette(style)
    metrics = report.get("metrics", {})
    thresholds = report.get("thresholds_used", {})

    if not metrics:
        raise ValueError("No metrics found in report")

    # Which metrics can go on the fairness axis, and WHY any of them cannot.
    #
    # This was ``{k: v for k, v in metrics.items() if "difference" in k}``: a
    # substring family selection, with a fallback to the whole dict when it
    # matched nothing. Both halves were wrong.
    #
    # The filter silently DROPPED every lower-is-better metric whose name
    # carries no "difference" token (multicalibration, auroc_parity,
    # net_benefit_parity, brier_score, integrated_calibration_index), so the
    # radar plotted a subset while reading as a complete picture of the audit.
    # The fallback then plotted the RATIO family on a gap axis, where
    # ``abs(0.000) <= 0.80`` reads as a zero gap: the maximal four-fifths
    # violation was drawn on the fully-fair outer rim.
    #
    # Selection is by DIRECTION now, resolved by _metric_direction, and
    # anything that cannot be placed is NAMED under the chart rather than
    # disappearing from it.
    names: List[str] = []
    values: List[float] = []
    threshold_values: List[float] = []
    directions: List[MetricDirection] = []
    excluded: List[str] = []

    for name, value in metrics.items():
        label = _radar_label(name)
        direction = metric_direction(name)
        threshold = thresholds.get(name, 0.1)
        if direction is MetricDirection.UNKNOWN:
            excluded.append(f"{label} ({COULD_NOT_CHECK_TEXT}: no known better-direction)")
            continue
        if not _is_measured(value):
            excluded.append(f"{label} ({COULD_NOT_CHECK_TEXT}: not measured)")
            continue
        if not _is_measured(threshold):
            excluded.append(f"{label} ({COULD_NOT_CHECK_TEXT}: no usable threshold)")
            continue
        names.append(name)
        values.append(float(value))
        threshold_values.append(float(threshold))
        directions.append(direction)

    if not names:
        # Nothing could be placed on a fairness axis. An empty radar with the
        # reasons on it is the honest figure; a radar drawn from whatever was
        # left is how the ratio family ended up on the fair rim.
        return _empty_radar(go, palette, style, excluded)

    display_names = [_radar_label(n) for n in names]

    # Normalize if requested
    if normalize:
        # Fairness axis: 1.0 = fair (outer rim), 0.0 = unfair (centre); every
        # metric's threshold maps to the same reference ring. Matches
        # radar_chart_to_svg so the interactive and SVG radars read the same way:
        # a fair model reaches outward into a round shape, failures cave inward.
        ring = 0.5

        def _fairness_score(v: float, t: float, direction: MetricDirection) -> float:
            if direction is MetricDirection.HIGHER_IS_BETTER:
                # A parity ratio: the threshold is a MINIMUM, perfect parity is
                # 1.0, and 0.0 is the maximal violation. Mapping it through the
                # gap branch below is what put a 0.000 four-fifths ratio on the
                # fair rim.
                if v >= t:
                    span = max(1.0 - t, 1e-9)
                    return min(1.0, ring + ((v - t) / span) * (1.0 - ring))
                return max(0.0, ring * (v / max(t, 1e-9)))
            gap = abs(v)
            t = max(t, 1e-9)
            if gap <= t:
                return 1.0 - (gap / t) * (1.0 - ring)
            return max(0.0, ring * (1.0 - (gap - t) / t))

        values = [_fairness_score(v, t, d) for v, t, d in zip(values, threshold_values, directions)]
        threshold_values = [ring] * len(names)
    else:
        # Raw values on one radial axis carry no shared meaning across
        # directions: for a gap, inside the threshold ring is compliant; for a
        # ratio it is the violation. Say which rule each axis is under.
        display_names = [
            f"{label} ({'min' if d is MetricDirection.HIGHER_IS_BETTER else 'max'} {t:g})"
            for label, d, t in zip(display_names, directions, threshold_values)
        ]

    # Close the radar
    values = values + [values[0]]
    threshold_values = threshold_values + [threshold_values[0]]
    display_names = display_names + [display_names[0]]

    fig = go.Figure()

    # Add threshold reference
    fig.add_trace(
        go.Scatterpolar(
            r=threshold_values,
            theta=display_names,
            fill="toself" if fill else None,
            fillcolor=f"rgba({int(palette['danger'][1:3], 16)}, {int(palette['danger'][3:5], 16)}, {int(palette['danger'][5:7], 16)}, 0.1)",
            line=dict(color=palette["danger"], width=2, dash="dash"),
            name="Pass threshold",
        )
    )

    # Add actual values
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=display_names,
            fill="toself" if fill else None,
            fillcolor=f"rgba({int(palette['info'][1:3], 16)}, {int(palette['info'][3:5], 16)}, {int(palette['info'][5:7], 16)}, 0.3)",
            line=dict(color=palette["info"], width=3),
            marker=dict(size=8, color=palette["info"]),
            name="Metric fairness",
        )
    )

    is_dark = style == "dark"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0]
                if normalize
                else [0, max(max(values), max(threshold_values)) * 1.15],
                gridcolor=palette["grid"],
                linecolor=palette.get("light", palette["neutral"]),
                tickfont=dict(size=10, color=palette.get("text_muted", palette["text"])),
            ),
            angularaxis=dict(
                gridcolor=palette["grid"],
                linecolor=palette.get("light", palette["neutral"]),
                tickfont=dict(size=11, color=palette["text"]),
            ),
            bgcolor=palette.get("background_alt", palette["background"]),
        ),
        showlegend=True,
        legend=dict(
            x=0.92,
            y=0.98,
            bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)",
            bordercolor=palette["grid"],
            borderwidth=1,
            font=dict(size=11, color=palette["text"]),
        ),
        title={
            "text": "<b>Fairness Metrics Overview</b>",
            "x": 0.5,
            "font": {"size": 18, "color": palette["text"], "family": "Inter, Segoe UI, Arial"},
        },
        paper_bgcolor=palette["background"],
        font={"color": palette["text"], "family": "Inter, Segoe UI, Arial, sans-serif"},
        margin=dict(t=80, b=80 if excluded else 40, l=40, r=40),
    )

    # Whatever could not be placed is named on the figure. A metric that simply
    # vanishes from the chart is indistinguishable from one that passed.
    if excluded:
        fig.add_annotation(
            text=_excluded_note(excluded),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.12,
            showarrow=False,
            align="center",
            font=dict(size=10, color=palette.get("text_muted", palette["text"])),
        )

    return fig


def plot_group_disparity_heatmap(report: Dict[str, Any], style: StyleType = "modern") -> Any:
    """
    Create a heatmap showing pairwise group disparities.

    Args:
        report: Report dict from fairness report function
        style: Visual style

    Returns:
        Plotly Figure object
    """
    if not _check_plotly():
        raise ImportError("Heatmaps require plotly. Install with: pip install plotly")

    import plotly.graph_objects as go

    palette = _get_palette(style)
    group_stats = report.get("group_stats", {})

    if not group_stats:
        raise ValueError("No group statistics found in report")

    groups = list(group_stats.keys())
    n = len(groups)

    # A PAIRWISE chart needs a PAIR (2026-09-25). With one group the matrix is 1x1,
    # its only cell is the group against itself, and that cell is 0.000 by
    # construction. Measured before this guard: a single-group report produced a
    # heatmap titled "Pairwise Positive Rate Difference" whose one cell read "0.000",
    # with no warning. A reader sees a pairwise disparity of zero, which is the
    # canvas for perfect parity, computed from no pair at all.
    #
    # The DIAGONAL is always zero for the same reason, and that is harmless in a
    # matrix of two or more groups because the off-diagonals carry the finding. In a
    # 1x1 matrix the diagonal is the entire chart.
    #
    # Raising rather than drawing an empty canvas: this function already raises for a
    # report with no group_stats and for one with no suitable metric, so a caller
    # handles the exception path, and a chart that cannot be about anything is better
    # not returned than returned looking like a measurement. The message says what to
    # do about it.
    if n < 2:
        raise ValueError(
            f"plot_group_disparity_heatmap needs at least two groups to compare and "
            f"this report has {n} ({', '.join(map(str, groups)) or 'none'}). A "
            f"single-group matrix has one cell, the group against itself, and it is "
            f"0.000 by construction: drawing it would show a pairwise disparity of "
            f"zero where there is no pair. Supply a protected attribute with at least "
            f"two groups carrying enough rows to compare."
        )

    # Get positive rate or MAE
    first_group = list(group_stats.values())[0]
    if "positive_rate" in first_group:
        metric = "positive_rate"
        metric_label = "Positive Rate Difference"
    elif "mae" in first_group:
        metric = "mae"
        metric_label = "MAE Difference"
    else:
        raise ValueError("No suitable metric found for heatmap")

    values = [group_stats[g].get(metric, 0) for g in groups]

    # Create pairwise difference matrix
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = values[i] - values[j]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=groups,
            y=groups,
            colorscale=[
                [0, palette["danger"]],
                [0.5, palette["background"]],
                [1, palette["success"]],
            ],
            zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="%{y} vs %{x}<br>Difference: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"<b>Pairwise {metric_label}</b>",
            "x": 0.5,
            "font": {"size": 18, "color": palette["text"], "family": "Inter, Segoe UI, Arial"},
        },
        xaxis_title="Group",
        yaxis_title="Group",
        paper_bgcolor=palette["background"],
        plot_bgcolor=palette.get("background_alt", palette["background"]),
        font={"color": palette["text"], "family": "Inter, Segoe UI, Arial, sans-serif"},
        xaxis=dict(
            tickfont=dict(size=11, color=palette["text"]),
            title_font=dict(size=12, color=palette["text"]),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=palette["text"]),
            title_font=dict(size=12, color=palette["text"]),
        ),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    return fig


# Matplotlib Visualizations (Static, Publication-Ready)


def _setup_matplotlib_style(style: StyleType = "academic"):
    """Configure matplotlib for professional, polished output."""
    import matplotlib.pyplot as plt

    palette = _get_palette(style)

    # Determine if dark theme
    is_dark = style == "dark"

    # Set comprehensive style parameters for polished appearance
    plt.rcParams.update(
        {
            # Figure styling
            "figure.facecolor": palette["background"],
            "figure.edgecolor": palette["background"],
            "figure.dpi": 100,
            # Axes styling
            "axes.facecolor": palette["background"],
            "axes.edgecolor": palette.get("light", palette["neutral"]),
            "axes.linewidth": 0.8,
            "axes.labelcolor": palette["text"],
            "axes.titlecolor": palette["text"],
            "axes.labelpad": 8,
            "axes.titlepad": 12,
            # Text colors
            "text.color": palette["text"],
            "xtick.color": palette.get("text_muted", palette["text"]),
            "ytick.color": palette.get("text_muted", palette["text"]),
            # Grid styling - subtle and clean
            "grid.color": palette["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.5 if not is_dark else 0.4,
            "grid.alpha": 0.7 if not is_dark else 0.5,
            # Font configuration - modern, readable fonts
            "font.family": "sans-serif",
            "font.sans-serif": [
                "SF Pro Display",
                "Segoe UI",
                "Roboto",
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "font.weight": "normal",
            # Title and label sizes - clear hierarchy
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            # Tick sizes
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            # Legend styling
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": palette.get("background_alt", palette["background"]),
            "legend.edgecolor": palette["grid"],
            "legend.borderpad": 0.6,
            "legend.labelspacing": 0.5,
            "legend.handlelength": 1.5,
            "legend.handleheight": 0.7,
            # Figure title
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            # Spine configuration - clean, minimal look
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # Grid and axis positioning
            "axes.grid": True,
            "axes.axisbelow": True,
            # Savefig defaults
            "savefig.facecolor": palette["background"],
            "savefig.edgecolor": palette["background"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            # Lines and markers
            "lines.linewidth": 2,
            "lines.markersize": 7,
            # Patches (bars, etc.)
            "patch.linewidth": 1,
            "patch.edgecolor": palette.get("light", palette["neutral"]),
        }
    )

    return palette


def plot_fairness_metrics(
    report: Dict[str, Any],
    figsize: Tuple[int, int] = (11, 6),
    style: StyleType = "academic",
    show_thresholds: bool = True,
    show_ci: bool = True,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot fairness metrics as a professional bar chart.

    Args:
        report: Report dict from fairness report function
        figsize: Figure size as (width, height)
        style: Visual style ('academic', 'business', 'modern', 'dark', 'nature', 'accessible')
        show_thresholds: Whether to show threshold lines
        show_ci: Whether to show confidence intervals
        title: Custom title for the plot
        ax: Matplotlib axis to plot on

    Returns:
        Matplotlib axis object
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib. Install with: pip install matplotlib")

    import matplotlib.pyplot as plt

    palette = _setup_matplotlib_style(style)
    is_dark = style == "dark"

    metrics = report.get("metrics", {})
    metrics_with_ci = report.get("metrics_with_ci", {})
    thresholds = report.get("thresholds_used", {})

    if not metrics:
        warnings.warn("No metrics found in report")
        return None

    metric_names = list(metrics.keys())
    values = [metrics[m] for m in metric_names]

    # Get CI data
    ci_lower, ci_upper = [], []
    has_ci = show_ci and metrics_with_ci
    if has_ci:
        for m in metric_names:
            if m in metrics_with_ci:
                ci_data = metrics_with_ci[m]
                ci_lower.append(ci_data.get("lower_bound", np.nan))
                ci_upper.append(ci_data.get("upper_bound", np.nan))
            else:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Determine colors with subtle alpha variation
    colors = []
    edge_colors = []
    for m, v in zip(metric_names, values):
        threshold = thresholds.get(m, 0.1)
        # Three states, never two. A non-finite metric was NOT MEASURED: red
        # would report a violation this run never observed, green a pass it
        # never earned. Neutral colour, and an "n/a" mark added below. The
        # direction is the shared helper's answer (see _bar_state_color), not a
        # local copy of the rule.
        bar_color = _bar_state_color(m, v, threshold, palette)
        colors.append(bar_color)
        edge_colors.append(bar_color)

    # Plot bars with rounded edges effect (using alpha gradient)
    x_pos = np.arange(len(metric_names))
    bar_width = 0.65

    # matplotlib cannot draw a None, and `values` comes straight out of the report:
    # a metric holding None, which is what a strict JSON encoder writes for a NaN,
    # raised "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'"
    # here and killed the chart that is this family's reference for marking an
    # unmeasured metric. Everything else in this function already routes that
    # value through _is_measured; only the canvas needed one dtype.
    plot_values = [float(v) if _is_measured(v) else np.nan for v in values]

    bars = ax.bar(
        x_pos,
        plot_values,
        color=colors,
        edgecolor=[c for c in edge_colors],
        linewidth=1.5,
        alpha=0.85,
        width=bar_width,
        zorder=3,
    )

    # Add subtle shadow effect for depth
    if not is_dark:
        ax.bar(
            x_pos + 0.02,
            [v * 0.98 for v in plot_values],
            color="#00000010",
            width=bar_width,
            zorder=2,
        )

    # Add error bars for CI with improved styling
    if has_ci:
        # plot_values, not values: a None here raised TypeError on the subtraction
        # below, and an error bar cannot be drawn around a value nobody measured.
        yerr_lower = [
            v - low if _is_measured(low) and _is_measured(v) else 0
            for v, low in zip(plot_values, ci_lower)
        ]
        yerr_upper = [
            u - v if _is_measured(u) and _is_measured(v) else 0
            for v, u in zip(plot_values, ci_upper)
        ]
        ax.errorbar(
            x_pos,
            plot_values,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            color=palette.get("text_muted", palette["neutral"]),
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            zorder=4,
        )

    # Acceptable region, PER METRIC and DIRECTION-AWARE.
    #
    # THE FOURTH INSTANCE OF THE DIRECTION BUG LIVED HERE. This was one
    # hardcoded band, ax.axhspan(-0.1, 0.1), labelled "Acceptable range" and
    # stretched across EVERY bar on the axis, whatever each bar measured. It
    # contradicted the report's own per-metric thresholds, and for the ratio
    # family it was INVERTED: demographic_parity_ratio = 0.000, the maximal
    # four-fifths violation, fell inside the band and read as the only
    # compliant bar on the chart, while a healthy 0.946 sat far outside it.
    # classification_fairness_report emits both metrics on ordinary input, so
    # the chart was inverting real audits. Colour-based and text-based pins
    # could not see it because the whole defect was the SHAPE.
    #
    # Each metric now gets its own region, opening in its own direction (see
    # _acceptable_region, which resolves that through _metric_direction), and a
    # metric that cannot be graded gets NO region and a stated reason instead.
    # DO NOT collapse these back into a single axis-wide band.
    legend_handles: List[Any] = []
    rule_labels: Dict[int, str] = {}
    if show_thresholds:
        from matplotlib.patches import Patch, Rectangle

        half = bar_width / 2 + 0.06
        drew_region = False
        unchecked: List[Tuple[int, str]] = []

        for i, m in enumerate(metric_names):
            threshold = thresholds.get(m, 0.1)
            band, marker_ys, reason = _acceptable_region(m, values[i], threshold)
            if band is None:
                unchecked.append((i, reason))
                continue
            drew_region = True
            region = Rectangle(
                (float(x_pos[i]) - half, band[0]),
                2 * half,
                band[1] - band[0],
                facecolor=palette["success"],
                edgecolor="none",
                alpha=0.12,
                zorder=1,
            )
            region.set_gid(ACCEPTABLE_REGION_GID)
            ax.add_patch(region)
            for ly in marker_ys:
                ax.plot(
                    [float(x_pos[i]) - half, float(x_pos[i]) + half],
                    [ly, ly],
                    color=palette["warning"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                )
            # The rule in words, under the bar it applies to (see the tick
            # labels below), so a reader never has to infer which side of the
            # line is the compliant one. It goes on the axis rather than in the
            # plot area because a label floating at the band edge lands on top
            # of whichever bar breaches it, i.e. exactly the bar that most needs
            # reading.
            rule = (
                "at least" if metric_direction(m) is MetricDirection.HIGHER_IS_BETTER else "at most"
            )
            rule_labels[i] = f"{rule} {float(threshold):g}"

        for i, reason in unchecked:
            # No band at all, and the reason where the band would have been.
            #
            # ONE MARK PER BAR. This whole block is for a bar that could not be
            # GRADED, and since 2026-09-29 an unmeasured VALUE reaches it too
            # (_acceptable_region refuses a region for one, as its docstring always
            # said it did). Such a bar already gets the "n/a not measured" mark
            # below, so annotating the reason here as well put TWO differently
            # worded could-not-check marks under the same bar, and
            # test_every_unmeasured_metric_is_marked_and_no_nan_is_drawn caught it
            # by counting marks against unmeasured metrics. The reason text
            # belongs to the case this block was written for: a value that WAS
            # measured and still could not be graded (no usable threshold, no
            # known direction).
            if _is_measured(values[i]):
                value = float(values[i])
                marker = Rectangle(
                    (float(x_pos[i]) - bar_width / 2, min(0.0, value)),
                    bar_width,
                    abs(value),
                    facecolor="none",
                    edgecolor=palette.get("neutral", palette["text"]),
                    hatch="//",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=4,
                )
                marker.set_gid(COULD_NOT_CHECK_GID)
                ax.add_patch(marker)
                ax.annotate(
                    f"{COULD_NOT_CHECK_TEXT}\n{reason.split(': ', 1)[-1]}",
                    xy=(float(x_pos[i]), 0.0),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=palette.get("neutral", palette["text"]),
                    zorder=5,
                )

        if drew_region:
            legend_handles.append(
                Patch(
                    facecolor=palette["success"],
                    alpha=0.2,
                    edgecolor="none",
                    label="Acceptable range (this metric)",
                )
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=palette["warning"],
                    linestyle="--",
                    linewidth=1.5,
                    label="Threshold",
                )
            )
        if unchecked:
            legend_handles.append(
                Patch(
                    facecolor="none",
                    hatch="//",
                    edgecolor=palette.get("neutral", palette["text"]),
                    label=f"{COULD_NOT_CHECK_TEXT.capitalize()} (no range applies)",
                )
            )

    # Format x-axis labels with better readability
    display_names = []
    for i, m in enumerate(metric_names):
        # More readable label formatting
        name = m.replace("_difference", "").replace("_ratio", " Ratio")
        name = name.replace("_", " ").title()
        # Word wrap long names
        if len(name) > 15:
            words = name.split()
            mid = len(words) // 2
            name = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        # Each metric carries its OWN rule, because each has its own region.
        if i in rule_labels:
            name = f"{name}\n({rule_labels[i]})"
        display_names.append(name)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.axhline(
        y=0,
        color=palette.get("light", palette["neutral"]),
        linestyle="-",
        alpha=0.5,
        linewidth=0.8,
        zorder=1,
    )

    # Title with better formatting
    task_type = report.get("task_type", "Classification")
    # Score line, or an explicit could-not-check state. See _score_display: the
    # score is Optional and this used to raise TypeError on None.
    score_text, score_color, score_is_assessed = _score_display(report, palette)
    score_line = (
        f"Overall Score: {score_text}"
        if score_is_assessed
        else f"{score_text}: no metric could be measured"
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=15)
    else:
        ax.set_title(
            f"{task_type.title()} Fairness Metrics", fontsize=14, fontweight="semibold", pad=15
        )
        # Add score as a subtitle
        ax.text(
            0.5,
            1.02,
            score_line,
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="bottom",
            color=score_color,
            fontweight="medium",
        )

    # A non-finite metric draws no bar at all, and an empty slot on a bar chart
    # reads as a measured zero disparity. Mark it where its bar would have been.
    for xi, val in zip(x_pos, values):
        if not _is_measured(val):
            ax.annotate(
                f"{NOT_MEASURED_TICK}\nnot measured",
                xy=(float(xi), 0.0),
                ha="center",
                va="center",
                fontsize=9,
                color=palette.get("neutral", palette["text"]),
                zorder=5,
            )

    # Add value labels with background for readability
    for bar, val in zip(bars, values):
        if not _is_measured(val):
            # No number to print: the "not measured" mark above stands instead.
            continue
        height = bar.get_height()
        label_y = height + 0.015 if height >= 0 else height - 0.015
        va = "bottom" if height >= 0 else "top"

        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, label_y),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="semibold",
            color=palette["text"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=palette.get("background_alt", palette["background"]),
                edgecolor="none",
                alpha=0.8,
            )
            if not is_dark
            else None,
        )

    # Improved legend. Built from explicit handles: the bands are per metric
    # now, so labelling one of them would name a range that only applies to a
    # single bar.
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=9)

    # Add subtle padding to y-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

    plt.tight_layout()
    return ax


def plot_group_comparison(
    report: Dict[str, Any],
    metric: str = "positive_rate",
    figsize: Tuple[int, int] = (9, 6),
    style: StyleType = "academic",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot per-group metric comparison with professional styling.

    Args:
        report: Report dict from fairness report function
        metric: Group metric to plot
        figsize: Figure size
        style: Visual style
        title: Custom title
        ax: Matplotlib axis

    Returns:
        Matplotlib axis object

    TWO UNDISCLOSED NEUTRAL VALUES WERE DRAWN ON THIS CHART, measured 2026-09-29:

    1. ``stats.get("size", 0)`` on the line under one that correctly used
       ``np.nan`` for the metric. A group whose ``size`` key is absent was drawn
       as ``n=0`` beside a 30.0% rate, which zero samples cannot produce:
           {'a': {'positive_rate': 0.50, 'size': 200},
            'b': {'positive_rate': 0.30}}
         -> labels ['30.0%', '50.0%', 'Avg: 40.0%', 'n=0', 'n=200'], warnings []
    2. The average said nothing about what it averaged over. Reached through the
       library's own producer: ``get_group_metrics`` reports ``tpr: nan`` for a
       group with no positive labels (tp+fn == 0, its honest could-not-check), and
       ``plot_group_comparison(rep, metric='tpr')`` then drew BOTH groups on the
       axis, ONE bar, no warning, and a dashed line labelled ``Avg: 0.667``
       spanning both groups which was group a alone, because ``np.nanmean``
       shrank the denominator in silence.

    So an absent size is ``n/a`` and not ``n=0``, a group whose value could not be
    measured gets the family's "n/a not measured" mark where its bar would have
    been (an absent bar and a zero bar must not look the same), the average names
    how many groups it covers whenever that is fewer than all of them, and either
    gap warns. ``_is_measured`` replaces ``np.isnan`` throughout, because a group
    stat holding None, this library's own JSON convention for a non-finite number,
    raised TypeError from ``np.isnan``.
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib")

    import matplotlib.pyplot as plt

    palette = _setup_matplotlib_style(style)
    is_dark = style == "dark"
    group_stats = report.get("group_stats", {})

    if not group_stats:
        warnings.warn("No group statistics found in report")
        return None

    groups = list(group_stats.keys())
    values = []
    sizes = []
    for g in groups:
        stats = group_stats[g]
        values.append(stats.get(metric, np.nan) if isinstance(stats, dict) else np.nan)
        # np.nan, never 0, and for the same reason the line above uses it: a
        # sample count that was never recorded is not a count of zero.
        sizes.append(stats.get("size", np.nan) if isinstance(stats, dict) else np.nan)

    measured = [float(v) for v in values if _is_measured(v)]
    if not measured:
        warnings.warn(f"Metric '{metric}' not found in group stats")
        return None

    unmeasured_values = [g for g, v in zip(groups, values) if not _is_measured(v)]
    unmeasured_sizes = [g for g, s in zip(groups, sizes) if not _is_measured(s)]
    if unmeasured_values or unmeasured_sizes:
        parts = []
        if unmeasured_values:
            parts.append(
                f"{len(unmeasured_values)} group(s) have no measured {metric} "
                f"({', '.join(str(g) for g in unmeasured_values)}), so they are on the axis "
                f"with a '{NOT_MEASURED_TICK} not measured' mark and no bar, and the average "
                f"covers the remaining {len(measured)}"
            )
        if unmeasured_sizes:
            parts.append(
                f"{len(unmeasured_sizes)} group(s) have no recorded sample size "
                f"({', '.join(str(g) for g in unmeasured_sizes)}), shown as "
                f"'n {NOT_MEASURED_TICK}' rather than n=0"
            )
        warnings.warn(
            "plot_group_comparison: " + "; ".join(parts) + ". This is could not check.",
            UserWarning,
            stacklevel=2,
        )

    # The span the value labels are offset by, over the MEASURED values only:
    # `max(values) - min(values)` with a NaN in the list returns NaN whenever the
    # NaN comes first (Python's max keeps its running best against a NaN
    # comparison), and an annotation placed at a NaN y is drawn nowhere at all.
    # So a real, measured number's label vanished depending on group ORDER.
    span = (max(measured) - min(measured)) if measured else 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Use carefully selected group colors
    colors = palette["groups"][: len(groups)]
    bar_width = 0.55

    # matplotlib cannot draw a None: `ax.bar` raised "TypeError: unsupported
    # operand type(s) for +: 'int' and 'NoneType'" on the None a strict JSON
    # encoder writes for a NaN, so the whole chart died on a value this function
    # is otherwise ready to mark as unmeasured. One dtype reaches the canvas.
    plot_values = [float(v) if _is_measured(v) else np.nan for v in values]

    bars = ax.bar(
        groups,
        plot_values,
        color=colors,
        edgecolor=[c for c in colors],  # Match edge to fill
        linewidth=1.5,
        alpha=0.85,
        width=bar_width,
        zorder=3,
    )

    # Add subtle shadow for depth (light themes only)
    if not is_dark:
        for i, (g, v) in enumerate(zip(groups, values)):
            if _is_measured(v):
                ax.bar(i + 0.015, float(v) * 0.98, color="#00000008", width=bar_width, zorder=2)

    # Add value and size labels with improved styling
    for i, (bar, val, size) in enumerate(zip(bars, values, sizes)):
        bar_center = bar.get_x() + bar.get_width() / 2
        if _is_measured(val):
            # Value label with background
            label_y = float(val) + span * 0.03
            ax.annotate(
                f"{val:.3f}" if metric != "positive_rate" else f"{val:.1%}",
                xy=(bar_center, label_y),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color=palette["text"],
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=palette.get("background_alt", palette["background"]),
                    edgecolor="none",
                    alpha=0.85,
                )
                if not is_dark
                else None,
            )
        else:
            # No bar is drawn for a value that was not measured, and an empty slot
            # on a bar chart reads as a measured zero. Mark it where the bar would
            # have been, in the same words the rest of this module uses.
            mark = ax.annotate(
                f"{NOT_MEASURED_TICK}\nnot measured",
                xy=(bar_center, 0.0),
                ha="center",
                va="center",
                fontsize=9,
                color=palette.get("neutral", palette["text"]),
                zorder=5,
            )
            mark.set_gid(COULD_NOT_CHECK_GID)

        # Size label below the bar, drawn for EVERY group: a group with no bar
        # still has a sample count worth reading, and an absent count says so
        # rather than reading as zero samples.
        if _is_measured(size):
            size_label = f"n={int(size):,}"
        else:
            size_label = f"n {NOT_MEASURED_TICK}"
        ax.annotate(
            size_label,
            xy=(bar_center, 0),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color=palette.get("text_muted", palette["neutral"]),
            style="italic",
        )

    # Add average line with improved styling. Over the MEASURED groups, and the
    # label says how many that was: np.nanmean quietly shrinks its own
    # denominator, so a mean of one group out of two was labelled "Avg" with
    # nothing to say it spanned a single bar.
    avg = float(np.nanmean(measured))
    ax.axhline(
        y=avg,
        color=palette.get("accent1", palette["neutral"]),
        linestyle="--",
        linewidth=1.8,
        alpha=0.6,
        zorder=2,
    )
    # Add average label
    avg_label = f"Avg: {avg:.3f}" if metric != "positive_rate" else f"Avg: {avg:.1%}"
    if len(measured) < len(values):
        avg_label += f" ({len(measured)} of {len(values)} groups)"
    ax.annotate(
        avg_label,
        xy=(len(groups) - 0.5, avg),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        color=palette.get("accent1", palette["neutral"]),
        fontweight="medium",
    )

    # Axis labels
    metric_label = metric.replace("_", " ").title()
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_xlabel("Group", fontsize=11)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=15)
    else:
        ax.set_title(f"{metric_label} by Group", fontsize=14, fontweight="semibold", pad=15)

    # Adjust y-axis to show labels
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.12)

    plt.tight_layout()
    return ax


def plot_effect_sizes(
    report: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 6),
    style: StyleType = "academic",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot effect sizes for pairwise group comparisons.

    Args:
        report: Report dict with effect_sizes
        figsize: Figure size
        style: Visual style
        title: Custom title
        ax: Matplotlib axis

    Returns:
        Matplotlib axis object

    THIS CHART CRASHED ON A REPORT ITS OWN PRODUCER EMITS, measured 2026-09-29:

        rep = FairnessAnalyzer(yt, yp, ['a']*60 + ['b']*60,
                               task_type='regression').get_report(include_ci=True)
        # yt == yp == 10.0 on group a and 20.0 on group b: zero within-group
        # prediction variance, so the producer writes
        #   cohens_d_predictions: nan
        #   interpretation: 'not interpretable (effect size is nan, not a measured value)'
        plot_effect_sizes(rep)
        -> ValueError: max() iterable argument is empty

    from ``x_max = max(abs(d) for d in cohens_d if not np.isnan(d)) * 1.3``:
    filtering the NaNs out emptied the generator. The producer computes that state
    on purpose, so the consumer died on a value the library deliberately refuses
    to interpret. Two more, same family: a key PRESENT holding None (this
    library's own JSON convention for a non-finite number) raised
    ``TypeError: ufunc 'isnan' not supported``, and an unmeasured pair beside a
    measured one rendered as a y-tick label with NO bar and no "not measured"
    mark, unlike ``plot_fairness_metrics`` which writes one.

    An unmeasured pair now gets the family's mark and is named in a warning, a
    report where NO pair is measurable draws the NOT ASSESSABLE headline instead
    of raising, and the axis limit comes from the measured values or from a
    default when there are none.
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = _setup_matplotlib_style(style)
    effect_sizes = report.get("effect_sizes", {})

    if not effect_sizes:
        # Two different states, and the old single message misdiagnosed one of
        # them: a single-group report built WITH include_ci=True carries
        # effect_sizes as an empty dict, and telling that reader to pass
        # include_ci=True names a setting they already used.
        if "effect_sizes" in report:
            warnings.warn(
                "No effect size could be computed for this report: effect_sizes is present "
                "and empty, which means no pair of groups was comparable (an effect size "
                "needs two). This is could not check, not a negligible effect.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn("No effect sizes in report. Use include_ci=True.")
        return None

    pairs = list(effect_sizes.keys())
    # Any, not Optional[float]: these are raw report values (a float, NaN, the
    # None a strict JSON encoder writes, or something else entirely). Every
    # float() below is reached only behind `_is_measured`, which is what makes
    # it safe; the annotation records that the guard, not the type, decides.
    cohens_d: List[Any] = []
    interpretations = []

    for pair in pairs:
        effects = effect_sizes[pair]
        if isinstance(effects, dict):
            d = effects.get("cohens_d_positive_rate", effects.get("cohens_d_predictions", np.nan))
            interpretation = effects.get("interpretation", "unknown")
        else:
            # Not the mapping this chart reads. Marked unmeasured and named in the
            # warning below rather than raising AttributeError on .get.
            d = np.nan
            interpretation = "unknown"
        cohens_d.append(d)
        interpretations.append(interpretation)

    # `_is_measured`, not `not np.isnan(d)`: np.isnan raises TypeError on a None,
    # and None is what a strict JSON encoder writes for a NaN.
    measured_d = [abs(float(d)) for d in cohens_d if _is_measured(d)]
    unmeasured_pairs = [p for p, d in zip(pairs, cohens_d) if not _is_measured(d)]
    if unmeasured_pairs:
        warnings.warn(
            f"plot_effect_sizes: {len(unmeasured_pairs)} of {len(pairs)} pair(s) have no "
            f"measured effect size ({', '.join(str(p) for p in unmeasured_pairs)}), so they are "
            f"drawn with a '{NOT_MEASURED_TICK} not measured' mark and no bar. An absent bar is "
            "not a negligible effect.",
            UserWarning,
            stacklevel=2,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Improved color mapping with consistent palette colors
    color_map = {
        "negligible": palette["success"],
        "negligible effect": palette["success"],
        "small": palette["warning"],
        "small effect": palette["warning"],
        "medium": palette.get("accent2", "#E67E22"),
        "medium effect": palette.get("accent2", "#E67E22"),
        "large": palette["danger"],
        "large effect": palette["danger"],
    }
    # An unmeasured row takes the neutral colour whatever its interpretation string
    # says: a green "negligible" under a row nobody measured is the same claim as a
    # bar at zero.
    colors = [
        color_map.get(interp, palette["neutral"]) if _is_measured(d) else palette["neutral"]
        for interp, d in zip(interpretations, cohens_d)
    ]

    display_pairs = [p.replace("_vs_", " vs ").replace("_", " ") for p in pairs]
    y_pos = np.arange(len(pairs))
    bar_height = 0.55

    # Add background bands for effect size ranges. The default holds the axis up
    # when NOTHING was measured, which is the state this line used to raise on.
    x_max = (max(measured_d) * 1.3) if measured_d else 1.0
    x_max = max(x_max, 1.0)

    # Shaded regions for effect size interpretation
    ax.axvspan(-0.2, 0.2, color=palette["success"], alpha=0.08, zorder=1)
    ax.axvspan(0.2, 0.5, color=palette["warning"], alpha=0.06, zorder=1)
    ax.axvspan(-0.5, -0.2, color=palette["warning"], alpha=0.06, zorder=1)
    ax.axvspan(0.5, 0.8, color=palette.get("accent2", "#E67E22"), alpha=0.05, zorder=1)
    ax.axvspan(-0.8, -0.5, color=palette.get("accent2", "#E67E22"), alpha=0.05, zorder=1)
    ax.axvspan(0.8, x_max, color=palette["danger"], alpha=0.04, zorder=1)
    ax.axvspan(-x_max, -0.8, color=palette["danger"], alpha=0.04, zorder=1)

    # matplotlib cannot draw a None: `barh` raised
    # "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'" on the
    # None a strict JSON encoder writes for a NaN. One dtype reaches the canvas.
    plot_d = [float(d) if _is_measured(d) else np.nan for d in cohens_d]

    bars = ax.barh(
        y_pos,
        plot_d,
        color=colors,
        edgecolor=[c for c in colors],
        linewidth=1.2,
        alpha=0.85,
        height=bar_height,
        zorder=3,
    )

    # Add value labels
    for i, (bar, d_val) in enumerate(zip(bars, cohens_d)):
        row_y = bar.get_y() + bar.get_height() / 2
        if _is_measured(d_val):
            d_val = float(d_val)
            label_x = d_val + 0.05 if d_val >= 0 else d_val - 0.05
            ha = "left" if d_val >= 0 else "right"
            ax.annotate(
                f"d = {d_val:.2f}",
                xy=(label_x, row_y),
                ha=ha,
                va="center",
                fontsize=9,
                fontweight="medium",
                color=palette["text"],
            )
        else:
            # A row with no bar at all sat on the axis as a bare tick label, which
            # on a chart whose zero line means "no effect" reads as a measured
            # absence of effect. The mark goes where the bar would have started.
            mark = ax.annotate(
                f"{NOT_MEASURED_TICK} not measured",
                xy=(0.0, row_y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color=palette.get("neutral", palette["text"]),
                zorder=5,
            )
            mark.set_gid(COULD_NOT_CHECK_GID)

    # Add subtle threshold lines
    for threshold in [0.2, 0.5, 0.8]:
        ax.axvline(
            x=threshold,
            color=palette.get("light", palette["neutral"]),
            linestyle=":",
            alpha=0.6,
            linewidth=1,
            zorder=2,
        )
        ax.axvline(
            x=-threshold,
            color=palette.get("light", palette["neutral"]),
            linestyle=":",
            alpha=0.6,
            linewidth=1,
            zorder=2,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_pairs, fontsize=10)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
    ax.axvline(
        x=0,
        color=palette.get("text_muted", palette["neutral"]),
        linestyle="-",
        alpha=0.4,
        linewidth=1.2,
        zorder=2,
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=15)
    else:
        ax.set_title(
            "Effect Sizes (Pairwise Comparisons)", fontsize=14, fontweight="semibold", pad=15
        )

    if not measured_d:
        # The headline the rest of this module uses when a figure reports nothing.
        # Without it a reader sees the interpretation bands, the legend and the
        # zero line, all of which describe effect sizes that were never measured.
        ax.annotate(
            f"{NOT_ASSESSABLE_TEXT}: no pair's effect size could be measured, so this "
            "figure reports nothing about the size of any difference",
            xy=(0.5, 0.97),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            color=palette.get("neutral", palette["text"]),
            zorder=6,
        )

    # Improved legend with clear interpretation
    legend_elements = [
        Patch(
            facecolor=palette["success"],
            alpha=0.85,
            edgecolor=palette["success"],
            linewidth=1.2,
            label="Negligible (|d| < 0.2)",
        ),
        Patch(
            facecolor=palette["warning"],
            alpha=0.85,
            edgecolor=palette["warning"],
            linewidth=1.2,
            label="Small (0.2 ≤ |d| < 0.5)",
        ),
        Patch(
            facecolor=palette.get("accent2", "#E67E22"),
            alpha=0.85,
            edgecolor=palette.get("accent2", "#E67E22"),
            linewidth=1.2,
            label="Medium (0.5 ≤ |d| < 0.8)",
        ),
        Patch(
            facecolor=palette["danger"],
            alpha=0.85,
            edgecolor=palette["danger"],
            linewidth=1.2,
            label="Large (|d| ≥ 0.8)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
        edgecolor=palette["grid"],
    )

    # Set x limits
    ax.set_xlim(-x_max, x_max)

    plt.tight_layout()
    return ax


def plot_confidence_intervals(
    report: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 6),
    style: StyleType = "academic",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot confidence intervals for fairness metrics.

    Args:
        report: Report dict with CI data
        figsize: Figure size
        style: Visual style
        title: Custom title
        ax: Matplotlib axis

    Returns:
        Matplotlib axis object

    THIS RAISED ON A REPORT ITS OWN PUBLIC PRODUCER EMITS, measured 2026-09-29:

        rep = FairnessAnalyzer(y, y, np.array(['a'] * 60)).get_report(include_ci=True)
        rep['metrics_with_ci']['demographic_parity_difference']
          -> {'point_estimate': nan, 'lower_bound': nan, 'upper_bound': nan, ...,
              'metadata': {'warning': 'Bootstrap failed'}}
        plot_confidence_intervals(rep)
          -> ValueError: Axis limits cannot be NaN or Inf   (at ax.set_xlim)

    One group means no disparity is comparable, so the bootstrap has nothing to
    resample and the producer says so in three NaNs and a warning. Both documented
    siblings handle exactly this: ``plot_fairness_metrics`` draws
    "NOT ASSESSABLE: no metric could be measured" with five "n/a not measured"
    marks, and ``plot_effect_sizes`` refuses with a warning naming the fix.

    Now: the axis range is built from the MEASURED values only (an absent bound was
    also defaulting to 0, a readable interval endpoint nobody measured), a row with
    no interval carries the family's "n/a not measured" mark plus the producer's
    own reason instead of a "nan\\n[nan, nan]" label, a partially measured row
    prints "n/a" for the parts that are missing, and a report where nothing could
    be measured gets the NOT ASSESSABLE headline rather than an exception.
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle as MplRectangle

    palette = _setup_matplotlib_style(style)
    is_dark = style == "dark"
    metrics_with_ci = report.get("metrics_with_ci", {})
    thresholds = report.get("thresholds_used", {})

    if not metrics_with_ci:
        # Two states, and the single old message misdiagnosed one: a report built
        # WITH include_ci=True can carry metrics_with_ci present and empty, and
        # telling that reader to pass include_ci=True names a setting they used.
        if "metrics_with_ci" in report:
            warnings.warn(
                "No interval could be computed for this report: metrics_with_ci is present "
                "and empty. This is could not check, and nothing here says any metric is "
                "precise or significant.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn("No CI data in report. Use include_ci=True.")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    metric_names = list(metrics_with_ci.keys())
    # Better label formatting
    display_names = []
    for m in metric_names:
        name = m.replace("_difference", "").replace("_ratio", " Ratio")
        name = name.replace("_", " ").title()
        display_names.append(name)

    y_pos = np.arange(len(metric_names))

    # Calculate x range for proper scaling.
    # MEASURED VALUES ONLY, and the defaults are NaN rather than 0. Two defects
    # met on these lines. (1) A single-group report from the public producer
    # carries point_estimate, lower_bound and upper_bound all NaN with
    # metadata.warning 'Bootstrap failed', so `max(abs(v) for v in all_values)`
    # was NaN and `ax.set_xlim` below raised
    # "ValueError: Axis limits cannot be NaN or Inf" (measured 2026-09-29 at what
    # was then line 2541), while both sibling charts marked the same input
    # honestly. (2) `.get("lower_bound", 0)` made an ABSENT bound a real, readable
    # interval endpoint at zero, which is a measurement nobody took.
    all_values = []
    for name in metric_names:
        ci_data = metrics_with_ci[name]
        for key in ("lower_bound", "upper_bound", "point_estimate"):
            candidate = _ci_field(ci_data, key)
            if _is_measured(candidate):
                all_values.append(float(candidate))
    for name in metric_names:
        # The thresholds belong in the range too: a four-fifths minimum of 0.80
        # off the edge of the axis is an acceptable region the reader cannot see.
        bound = thresholds.get(name, 0.1)
        if _is_measured(bound):
            all_values.append(float(bound))
    measured_rows = [
        name
        for name in metric_names
        if _is_measured(_ci_field(metrics_with_ci[name], "point_estimate"))
    ]
    x_max = (max(abs(v) for v in all_values) * 1.3) if all_values else 0.15
    x_max = max(x_max, 0.15)

    unmeasured_rows = [name for name in metric_names if name not in measured_rows]
    if unmeasured_rows:
        warnings.warn(
            f"plot_confidence_intervals: {len(unmeasured_rows)} of {len(metric_names)} metric(s) "
            f"have no measured interval ({', '.join(str(n) for n in unmeasured_rows)}), so their "
            f"row carries a '{NOT_MEASURED_TICK} not measured' mark and no interval. An absent "
            "interval is not a narrow one.",
            UserWarning,
            stacklevel=2,
        )

    # Acceptable region PER ROW, in each metric's own direction. This was one
    # axvspan from -0.1 to +0.1 across the whole axis, the same inverted
    # geometry as the bar chart: it put a four-fifths ratio of 0.000 inside the
    # acceptable shading and a healthy 0.946 outside it. See _acceptable_region.
    could_not_check_rows = False
    for i, name in enumerate(metric_names):
        threshold = thresholds.get(name, 0.1)
        # Through _ci_field, never `.get("point_estimate", 0)`: a zero default
        # made an ABSENT point estimate a measured compliant one and drew the
        # green "Acceptable range" band for it. See _ci_field.
        pe = _ci_field(metrics_with_ci[name], "point_estimate")
        band, marker_xs, reason = _acceptable_region(name, pe, threshold)
        if band is None:
            could_not_check_rows = True
            ax.annotate(
                f"{COULD_NOT_CHECK_TEXT}: {reason.split(': ', 1)[-1]}",
                xy=(-x_max, i),
                xytext=(4, -12),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color=palette.get("neutral", palette["text"]),
                zorder=5,
            )
            continue
        region = MplRectangle(
            (band[0], i - 0.4),
            band[1] - band[0],
            0.8,
            facecolor=palette["success"],
            edgecolor="none",
            alpha=0.12,
            zorder=1,
        )
        region.set_gid(ACCEPTABLE_REGION_GID)
        ax.add_patch(region)
        for lx in marker_xs:
            ax.plot(
                [lx, lx],
                [i - 0.4, i + 0.4],
                color=palette["warning"],
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

    # Plot each metric's CI
    for i, (name, display_name) in enumerate(zip(metric_names, display_names)):
        ci_data = metrics_with_ci[name] if isinstance(metrics_with_ci[name], dict) else {}
        # np.nan, not 0: an absent point estimate was drawn as a diamond AT ZERO,
        # which on a disparity axis is the marker for perfect parity.
        # And NOT `.get("lower_bound", pe)`: defaulting an absent BOUND to the
        # point estimate printed "[0.420, 0.420]" for a report that carried only
        # a point estimate, which is a perfect precision nobody measured. Each
        # of the three numbers is now measured or absent on its own. See
        # _ci_field.
        pe = _ci_field(ci_data, "point_estimate")
        lower = _ci_field(ci_data, "lower_bound")
        upper = _ci_field(ci_data, "upper_bound")
        threshold = thresholds.get(name, 0.1)

        if not any(_is_measured(v) for v in (pe, lower, upper)):
            # Nothing to draw, and the old code drew it anyway: the label read
            # "nan\n[nan, nan]" and the line and diamond were silently absent, so
            # the row looked like a chart still loading rather than a measurement
            # that does not exist.
            reason = ""
            metadata = ci_data.get("metadata")
            if isinstance(metadata, dict) and metadata.get("warning"):
                reason = f" ({metadata['warning']})"
            mark = ax.annotate(
                f"{NOT_MEASURED_TICK} not measured{reason}",
                xy=(0.0, i),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=palette.get("neutral", palette["text"]),
                zorder=5,
            )
            mark.set_gid(COULD_NOT_CHECK_GID)
            continue

        # Colour by where the interval sits in THIS metric's direction, from
        # the shared helper. The ratio family used to be painted a flat
        # informational blue, which says nothing about the four-fifths rule.
        color = _ci_state_color(name, pe, lower, upper, threshold, palette)

        # Draw CI line with improved styling
        ax.plot(
            [lower, upper],
            [i, i],
            color=color,
            linewidth=3,
            solid_capstyle="round",
            alpha=0.8,
            zorder=3,
        )

        # Draw point estimate marker
        ax.scatter(
            [pe],
            [i],
            color=color,
            s=100,
            marker="D",
            edgecolors="white" if is_dark else palette["background"],
            linewidths=2,
            zorder=4,
        )

        # Add CI range label. Each of the three numbers is printed only if it was
        # measured: a row with a real point estimate and no bounds used to read
        # "0.500\n[nan, nan]", and the anchor was NaN so the label was not drawn at
        # all. "n/a" is this module's word for it everywhere else.
        def _fmt(value: Any) -> str:
            return f"{float(value):.3f}" if _is_measured(value) else NOT_MEASURED_TICK

        anchor = next(
            (float(v) for v in (upper, pe, lower) if _is_measured(v)),
            0.0,
        )
        label_x = anchor + 0.02
        ax.annotate(
            f"{_fmt(pe)}\n[{_fmt(lower)}, {_fmt(upper)}]",
            xy=(label_x, i),
            ha="left",
            va="center",
            fontsize=8,
            color=palette.get("text_muted", palette["text"]),
            linespacing=1.2,
        )

    # Zero line
    ax.axvline(
        x=0,
        color=palette.get("text_muted", palette["neutral"]),
        linestyle="-",
        alpha=0.4,
        linewidth=1,
        zorder=2,
    )

    # Threshold rules are drawn per row above, alongside the region they bound.
    # Axis-wide +/-0.1 lines here would reinstate the same claim the band made.

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("Metric Value (95% CI)", fontsize=11)

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=15)
    else:
        ax.set_title(
            "Fairness Metrics with Confidence Intervals", fontsize=14, fontweight="semibold", pad=15
        )

    if not measured_rows:
        # The same headline the other charts in this module put on a figure that
        # reports nothing, and the state this function used to answer by raising.
        ax.annotate(
            f"{NOT_ASSESSABLE_TEXT}: no interval could be measured, so this figure "
            "reports nothing about precision or significance",
            xy=(0.5, 0.97),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            color=palette.get("neutral", palette["text"]),
            zorder=6,
        )

    # Improved legend
    legend_elements = [
        Patch(
            facecolor=palette["success"],
            alpha=0.15,
            edgecolor="none",
            label="Acceptable range (per metric)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=palette["success"],
            linewidth=3,
            marker="D",
            markersize=7,
            label="Pass (CI in range)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=palette["warning"],
            linewidth=3,
            marker="D",
            markersize=7,
            label="Caution (significant)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=palette["danger"],
            linewidth=3,
            marker="D",
            markersize=7,
            label="Fail (CI outside range)",
        ),
    ]
    if could_not_check_rows:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=palette.get("neutral", palette["text"]),
                linewidth=3,
                marker="D",
                markersize=7,
                label=f"{COULD_NOT_CHECK_TEXT.capitalize()} (no range applies)",
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
        edgecolor=palette["grid"],
    )

    # Set x limits with padding for labels
    ax.set_xlim(-x_max, x_max + 0.15)

    plt.tight_layout()
    return ax


def plot_fairness_report(
    report: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    style: StyleType = "academic",
    save_path: Optional[str] = None,
    axes: Optional[List[Any]] = None,
) -> Any:
    """
    Generate a comprehensive fairness report visualization.

    Creates a professional multi-panel figure suitable for academic papers
    and business presentations.

    Args:
        report: Report dict from fairness report function
        figsize: Overall figure size
        style: Visual style ('academic', 'business', 'modern', 'dark')
        save_path: If provided, save figure to this path
        axes: Optional list of matplotlib axes for subplot integration

    Returns:
        Matplotlib figure object
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib")

    import matplotlib.pyplot as plt

    palette = _setup_matplotlib_style(style)

    has_ci = "metrics_with_ci" in report
    has_effects = "effect_sizes" in report and report["effect_sizes"]

    n_panels = 2
    if has_ci:
        n_panels += 1
    if has_effects:
        n_panels += 1

    # Create figure
    if axes is not None:
        if len(axes) < 2:
            raise ValueError("Must provide at least 2 axes")
        fig = axes[0].get_figure()
        plot_axes = axes
    else:
        fig = plt.figure(figsize=figsize)
        if n_panels == 2:
            plot_axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
        elif n_panels == 3:
            plot_axes = [fig.add_subplot(2, 2, i + 1) for i in range(3)]
        else:
            plot_axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]

    # Plot panels
    plot_fairness_metrics(report, ax=plot_axes[0], style=style, show_ci=has_ci)

    group_stats = report.get("group_stats", {})
    if group_stats:
        first_group = list(group_stats.values())[0]
        metric = "positive_rate" if "positive_rate" in first_group else "mae"
        plot_group_comparison(report, metric=metric, ax=plot_axes[1], style=style)

    panel_idx = 2
    if has_effects and panel_idx < len(plot_axes):
        plot_effect_sizes(report, ax=plot_axes[panel_idx], style=style)
        panel_idx += 1

    if has_ci and panel_idx < len(plot_axes):
        plot_confidence_intervals(report, ax=plot_axes[panel_idx], style=style)

    # Overall title. See _score_display: the score is Optional and this used to
    # raise TypeError on None. A not-assessable run says so in the title rather
    # than borrowing a number it never measured.
    task_type = report.get("task_type", "Classification")
    score_text, _score_color, score_is_assessed = _score_display(report, palette)
    if score_is_assessed:
        suptitle = f"{task_type.title()} Fairness Audit Report (Score: {score_text})"
    else:
        suptitle = f"{task_type.title()} Fairness Audit Report ({score_text})"
    fig.suptitle(
        suptitle,
        fontsize=16,
        fontweight="bold",
        y=1.02,
        color=palette["text"] if score_is_assessed else palette.get("neutral", palette["text"]),
    )
    if not score_is_assessed:
        # State the reason under the title. A reader who sees only "NOT
        # ASSESSABLE" should not have to guess whether the tool broke or the
        # data could not support a comparison.
        fig.text(
            0.5,
            0.985,
            NOT_ASSESSABLE_REASON,
            ha="center",
            va="top",
            fontsize=10,
            color=palette.get("neutral", palette["text"]),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=palette["background"])
        print(f"Saved fairness report to: {save_path}")

    return fig


def save_fairness_plots(
    report: Dict[str, Any],
    output_dir: str = ".",
    prefix: str = "fairness",
    format: str = "png",
    style: StyleType = "academic",
    dpi: int = 300,
) -> List[str]:
    """
    Save individual fairness plots to files with professional quality.

    Args:
        report: Report dict from fairness report function
        output_dir: Directory to save plots
        prefix: Filename prefix
        format: Image format ('png', 'pdf', 'svg')
        style: Visual style
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    if not _check_matplotlib():
        raise ImportError("Visualization requires matplotlib")

    import matplotlib.pyplot as plt

    palette = _get_palette(style)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Metrics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fairness_metrics(report, ax=ax, style=style)
    path = output_path / f"{prefix}_metrics.{format}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=palette["background"])
    saved_files.append(str(path))
    plt.close(fig)

    # Group comparison
    group_stats = report.get("group_stats", {})
    if group_stats:
        fig, ax = plt.subplots(figsize=(8, 6))
        first_group = list(group_stats.values())[0]
        metric = "positive_rate" if "positive_rate" in first_group else "mae"
        plot_group_comparison(report, metric=metric, ax=ax, style=style)
        path = output_path / f"{prefix}_groups.{format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=palette["background"])
        saved_files.append(str(path))
        plt.close(fig)

    # Effect sizes
    if "effect_sizes" in report and report["effect_sizes"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_effect_sizes(report, ax=ax, style=style)
        path = output_path / f"{prefix}_effect_sizes.{format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=palette["background"])
        saved_files.append(str(path))
        plt.close(fig)

    # CI plot
    if "metrics_with_ci" in report:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_confidence_intervals(report, ax=ax, style=style)
        path = output_path / f"{prefix}_confidence_intervals.{format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=palette["background"])
        saved_files.append(str(path))
        plt.close(fig)

    # Full report
    fig = plot_fairness_report(report, figsize=(14, 10), style=style)
    path = output_path / f"{prefix}_full_report.{format}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=palette["background"])
    saved_files.append(str(path))
    plt.close(fig)

    return saved_files


# Utility Functions


def get_available_styles() -> List[str]:
    """Return list of available visualization styles."""
    return list(PALETTES.keys())


def preview_palette(style: StyleType = "modern") -> Any:
    """
    Display a comprehensive preview of the color palette.

    Args:
        style: Style to preview

    Returns:
        Matplotlib figure or Plotly figure
    """
    palette = _get_palette(style)
    is_dark = style == "dark"
    # The palette the swatches below ACTUALLY are, and the sentence a reader needs
    # when that is not the one they asked for. See _palette_preview_labels: the
    # title used to be built from the REQUESTED name over another palette's
    # colours, with the only disclosure in a warnings.warn that a saved PNG does
    # not carry.
    headline, substitution = _palette_preview_labels(style)

    if _check_plotly():
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Main semantic colors
        main_colors = ["primary", "secondary", "success", "warning", "danger", "info", "neutral"]
        main_values = [palette.get(c, "#ccc") for c in main_colors]

        # Group colors
        group_colors = palette.get("groups", [])

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Semantic Colors", "Group Colors"],
            vertical_spacing=0.25,
            row_heights=[0.5, 0.5],
        )

        # Semantic colors
        fig.add_trace(
            go.Bar(
                x=main_colors,
                y=[1] * len(main_colors),
                marker_color=main_values,
                marker_line_color=main_values,
                marker_line_width=2,
                text=[f"<b>{c}</b><br>{v}" for c, v in zip(main_colors, main_values)],
                textposition="inside",
                textfont=dict(
                    color="white"
                    if is_dark
                    else ["white" if i < 4 else "black" for i in range(len(main_colors))],
                    size=10,
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Group colors
        fig.add_trace(
            go.Bar(
                x=[f"Group {i + 1}" for i in range(len(group_colors))],
                y=[1] * len(group_colors),
                marker_color=group_colors,
                marker_line_color=group_colors,
                marker_line_width=2,
                text=group_colors,
                textposition="inside",
                textfont=dict(color="white", size=9),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{headline}</b>"
                + (
                    f"<br><span style='font-size:12px'>{substitution}</span>"
                    if substitution
                    else ""
                ),
                x=0.5,
                font=dict(size=18, color=palette["text"]),
            ),
            height=400,
            paper_bgcolor=palette["background"],
            plot_bgcolor=palette.get("background_alt", palette["background"]),
            font=dict(color=palette["text"]),
            showlegend=False,
        )

        fig.update_yaxes(visible=False)
        fig.update_xaxes(tickfont=dict(size=10))

        return fig

    elif _check_matplotlib():
        import matplotlib.pyplot as plt

        # Main semantic colors
        main_colors = ["primary", "secondary", "success", "warning", "danger", "info", "neutral"]
        main_values = [palette.get(c, "#ccc") for c in main_colors]

        # Group colors
        group_colors = palette.get("groups", [])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

        # Semantic colors
        bars1 = ax1.bar(
            range(len(main_colors)),
            [1] * len(main_colors),
            color=main_values,
            edgecolor="white",
            linewidth=2,
        )
        ax1.set_xticks(range(len(main_colors)))
        ax1.set_xticklabels(main_colors, fontsize=10, fontweight="medium")
        ax1.set_ylim(0, 1.2)
        ax1.set_xlim(-0.6, len(main_colors) - 0.4)
        ax1.axis("off")
        ax1.set_title("Semantic Colors", fontsize=12, fontweight="semibold", pad=10)

        # Add color codes on bars
        for bar, val in zip(bars1, main_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                val,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # Group colors
        ax2.bar(
            range(len(group_colors)),
            [1] * len(group_colors),
            color=group_colors,
            edgecolor="white",
            linewidth=2,
        )
        ax2.set_xticks(range(len(group_colors)))
        ax2.set_xticklabels(
            [f"G{i + 1}" for i in range(len(group_colors))], fontsize=10, fontweight="medium"
        )
        ax2.set_ylim(0, 1.2)
        ax2.set_xlim(-0.6, len(group_colors) - 0.4)
        ax2.axis("off")
        ax2.set_title("Group Colors", fontsize=12, fontweight="semibold", pad=10)

        fig.suptitle(headline, fontsize=16, fontweight="bold", y=1.02)
        if substitution:
            # Under the title, on the canvas, for the same reason
            # plot_fairness_report states its NOT ASSESSABLE reason there: a
            # reader of the saved file has no warning to read.
            fig.text(
                0.5,
                0.985,
                substitution,
                ha="center",
                va="top",
                fontsize=9,
                color=palette.get("neutral", palette["text"]),
            )
        plt.tight_layout()
        return fig

    else:
        print(f"\n{'=' * 50}")
        print(f"Palette: {headline}")
        if substitution:
            # The text branch is an artifact too: it is what a headless run
            # prints, and it named the requested style over another one's colours
            # exactly as the figure did.
            print(substitution)
        print("=" * 50)
        print("\nSemantic Colors:")
        for name in ["primary", "secondary", "success", "warning", "danger", "info", "neutral"]:
            print(f"  {name:12}: {palette.get(name, 'N/A')}")
        print("\nGroup Colors:")
        for i, color in enumerate(palette.get("groups", [])):
            print(f"  Group {i + 1:2}: {color}")
        return None
