"""
Unit 3: Performance Dashboards, FairnessDashboard
===================================================

Plotly-based interactive visualization engine for fairness metrics.

The ``FairnessDashboard`` reads from a :class:`MetricsStore` and produces
interactive ``plotly.graph_objects.Figure`` instances that support hover,
zoom, selection, and drill-down directly in Jupyter notebooks or exported
as standalone HTML files.

When Plotly is not installed, every method degrades gracefully to ``None``
with an informative warning.

The dashboard implements **progressive disclosure** through three view tiers:

- **Tier 1 (Executive):** Health score gauge + 3 KPI indicator cards.
- **Tier 2 (Operational):** 2×2 grid with trend chart, disparity bars,
  alert timeline, and NLG annotations.
- **Tier 3 (Technical):** All plots + data tables + intersectional heatmaps.

References
----------
Microsoft Fairlearn Dashboard; Google What-If Tool; Plotly documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..._triage import is_measured
from ...evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)
from .store import HealthScore, MetricsStore

# Optional Plotly import

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]


def _check_plotly():
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for FairnessDashboard.\n"
            "Install it with: pip install plotly>=5.0  "
            "or: pip install vfairness[dashboard]"
        )


# Enums & Config


class TimeWindow(Enum):
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"
    LAST_90D = "90d"
    CUSTOM = "custom"


_WINDOW_TD = {
    TimeWindow.LAST_24H: timedelta(hours=24),
    TimeWindow.LAST_7D: timedelta(days=7),
    TimeWindow.LAST_30D: timedelta(days=30),
    TimeWindow.LAST_90D: timedelta(days=90),
}


@dataclass
class DashboardConfig:
    """Configuration for :class:`FairnessDashboard`."""

    default_time_window: TimeWindow = TimeWindow.LAST_30D
    height: int = 500
    width: int = 900
    template: str = "plotly_white"
    color_pass: str = "#059669"  # emerald-600
    color_warn: str = "#f59e0b"  # amber-500
    color_fail: str = "#dc2626"  # red-600
    color_neutral: str = "#3b82f6"  # blue-500
    #: Could-not-check. Slate-400, the same neutral the SVG adapters use for a
    #: value that was never graded. It is deliberately neither the green of a
    #: measured pass nor the red of a measured failure.
    color_unknown: str = "#94a3b8"  # slate-400

    #: The bound a HIGHER-is-better metric (the ratio family) is graded
    #: against, and the amber band below it. 0.80 is the EEOC four-fifths rule
    #: (UGESP, 29 CFR 1607.4(D)); it is a required MINIMUM, so a ratio at or
    #: above it passes.
    ratio_pass_bound: float = 0.80
    ratio_warn_bound: float = 0.60
    #: The bound a LOWER-is-better metric (a violation magnitude: a gap, a
    #: difference, a disparity) is graded against, and the amber band above it.
    #: 0.10 is this repo's house default for a parity gap; it matches
    #: ``rendering.adapters_fairness._DEFAULT_THRESHOLDS``, which grades
    #: demographic_parity_difference / equal_opportunity_difference at 0.10.
    gap_pass_bound: float = 0.10
    gap_warn_bound: float = 0.20


# Direction-aware grading
#
# CRITICAL, three states, never two. Every colour and every reference line on
# this page used to be drawn from ONE hardcoded ladder, ``v >= 0.8 -> green``,
# annotated "80% Rule". That ladder is the EEOC four-fifths rule, and the
# four-fifths rule is a required MINIMUM on a RATIO. ``metric`` is a free
# parameter here, and ``create_operational_view`` defaults it to the store's
# FIRST metric name, which in this codebase is commonly "demographic_parity", a
# violation magnitude where LOWER is better.
#
# Measured on this repo, 2026-09-10, before this block existed: a store holding
# demographic_parity 0.95 for gender_female and 0.02 for gender_male rendered
# ``marker_color = ('#059669', '#dc2626')`` from ``create_disparity_comparison``
# with a dashed reference line at y=0.8 captioned "80% Rule", i.e. the near-total
# parity gap was painted PASS GREEN and the near-perfect one FAIL RED, under a
# legend naming a real employment-discrimination standard.
# ``check_threshold("demographic_parity", 0.95, 0.8)`` answers FAIL and
# ``(..., 0.02, 0.8)`` answers PASS. ``create_operational_view()`` on the same
# store rendered ('#059669', '#f59e0b'). This reaches the live Dash app, the
# standalone HTML from :meth:`FairnessDashboard.to_html`, and every report figure
# that embeds these views.
#
# The direction is resolved by
# :mod:`vfairness.evaluation.vfairness_metrics._metric_direction` and by nothing
# written here: that module owns the precedence table and it is pinned by
# ``tests/test_metric_direction.py``. Do not add a local name test, however
# small; a second copy of that rule is what this bug class is made of.


def _grading_bounds(
    metric: str, config: DashboardConfig
) -> Tuple[Optional[float], Optional[float]]:
    """The (pass, warn) bounds for *metric*, or ``(None, None)`` when ungraded.

    ``(None, None)`` means the metric's better-direction could not be resolved,
    so there is no bound to draw and nothing on the chart may be coloured as a
    verdict.
    """
    direction = metric_direction(metric)
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return (config.ratio_pass_bound, config.ratio_warn_bound)
    if direction is MetricDirection.LOWER_IS_BETTER:
        return (config.gap_pass_bound, config.gap_warn_bound)
    return (None, None)


def _reference_line(metric: str, config: DashboardConfig) -> Optional[Tuple[float, str]]:
    """The (y, caption) of the reference line for *metric*, or ``None``.

    The caption names the bound it actually draws. "80% Rule" over a
    demographic-parity gap was not merely the wrong colour, it was the wrong
    STANDARD: it told the reader a four-fifths selection-rate ratio had been
    applied to a quantity that is not a ratio.
    """
    direction = metric_direction(metric)
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return (config.ratio_pass_bound, f"80% Rule (minimum {config.ratio_pass_bound:.2f})")
    if direction is MetricDirection.LOWER_IS_BETTER:
        return (config.gap_pass_bound, f"Max gap {config.gap_pass_bound:.2f}")
    return None


def _value_color(metric: str, value: float, config: DashboardConfig) -> str:
    """Traffic-light colour for one value, IN THAT METRIC'S OWN DIRECTION.

    Four outcomes, not three: pass green, warn amber, fail red, and the neutral
    slate for could-not-check. Could-not-check covers an unresolvable direction
    AND an unmeasurable value: ``nan >= 0.8`` is False, so a metric that was
    never measured used to fall through to the fail red, which is a verdict.
    """
    pass_bound, warn_bound = _grading_bounds(metric, config)
    if pass_bound is None or warn_bound is None:
        return config.color_unknown
    outcome, _ = check_threshold(metric, value, pass_bound)
    if outcome is ThresholdOutcome.PASS:
        return config.color_pass
    if outcome is ThresholdOutcome.COULD_NOT_CHECK:
        return config.color_unknown
    # Measured breach of the pass bound. The warn band is the SAME comparison
    # against the more permissive bound, so it inherits the direction too.
    warn_outcome, _ = check_threshold(metric, value, warn_bound)
    if warn_outcome is ThresholdOutcome.PASS:
        return config.color_warn
    if warn_outcome is ThresholdOutcome.COULD_NOT_CHECK:
        return config.color_unknown
    return config.color_fail


def _value_colors(metric: str, values, config: DashboardConfig) -> List[str]:
    """:func:`_value_color` over a sequence, preserving order."""
    return [_value_color(metric, float(v), config) for v in values]


def _heatmap_scale(metric: str) -> Tuple[str, bool]:
    """The (colorscale, reversescale) a heatmap of *metric* may use.

    ``colorscale="RdYlGn"`` maps the MINIMUM of the data to dark red and the
    MAXIMUM to dark green. That is correct only for the ratio family. On a
    lower-is-better metric it inverts the whole chart: measured on this repo,
    the scale resolved to ``rgb(165,0,38)`` at 0.0 and ``rgb(0,104,55)`` at 1.0,
    so a zero gap (perfect parity) was the darkest RED cell on the grid and the
    largest gap the darkest GREEN one.

    A metric with no resolvable direction gets a single-hue ramp that encodes
    magnitude and claims nothing about good or bad, because painting a green end
    on an ungraded quantity IS a claim.
    """
    direction = metric_direction(metric)
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return ("RdYlGn", False)
    if direction is MetricDirection.LOWER_IS_BETTER:
        return ("RdYlGn", True)
    return ("Greys", False)


def _drift_coverage_note(store: MetricsStore, hs: HealthScore) -> Optional[str]:
    """The sentence a composite score needs when its drift weight is unmeasured.

    BGL3 operations-1, 2026-09-27. ``store._drift_stability`` returns a
    hardcoded ``100.0`` for an EMPTY drift table, and that number is weighted 20
    percent into the composite this page draws in a colour-banded gauge.
    ``ReportGenerator._health_with_drift_coverage`` (reports.py) already refuses
    to publish it unqualified on the document surface, and the chart surface had
    no equivalent. Measured before this change, on a store with four clean
    determined metric records and ZERO drift rows, which is the default state of
    every MetricsStore: ``create_health_score_gauge`` rendered ``value=100.0``
    captioned "Fairness health score is 100/100 (healthy) with a stable trend",
    and the word drift appeared nowhere on the figure.

    The existence question is asked of the WHOLE store, not of the scored
    window, deliberately: a store with no drift row at all cannot have one in
    any window, so this can never raise a false could-not-check on a caller that
    scored a different window from the one drawn. A store that holds drift rows
    outside the scored window says nothing here rather than risk that.

    Returns ``None`` when the component WAS measured.

    ABSENT FROM ``components`` IS THE UNMEASURED STATE, and that inverted on
    2026-09-27. This function used to return None for an absent key, on the reasoning
    that absence meant the component had "already been withheld one level up" and was
    therefore disclosed elsewhere. That was true while ``compute_health_score`` ALWAYS
    published ``drift_stability``, carrying a hardcoded 100.0 for an empty drift table;
    the only way to spot the fabrication from out here was to see that 100.0 sitting in
    the components.

    The producer now drops the key precisely when drift was not measured, and
    renormalises the composite over the two components that were. So absence became the
    signal rather than the exemption, and this early return made the note go silent in
    exactly the case it exists for: 11 tests across three files went red on that, each
    asserting the sentence a reader sees. The store's own ``explanation`` says it too,
    but a figure is not read through the store's explanation.
    """
    if "drift_stability" not in hs.components:
        # BGL grade-1 G08, 2026-09-30. The tail of this sentence is a positive
        # claim about the score above it ("this score covers metric compliance
        # and alert frequency"), and on a store where the score was WITHHELD
        # there is no score and nothing was covered. Measured on an empty
        # MetricsStore: score=None, status='not_assessed', components={}, and
        # this note was the ONLY sentence on the operational figure, so the one
        # caption a reader got asserted a coverage that did not exist.
        if not is_measured(hs.score):
            return (
                "Drift stability: NOT ASSESSED, and neither was anything else: no "
                "health score was computed for this window, so there is no composite "
                "for a drift weight to be absent from. It is not a drift stability of "
                "100 and not one of 0."
            )
        return (
            "Drift stability: NOT ASSESSED. No drift test was run, so the drift "
            "component is absent from this score, which covers metric compliance and "
            "alert frequency only. It is not a drift stability of 100 and not one of 0."
        )
    try:
        drift_df = store.get_drift_history()
        n_drift = 0 if drift_df is None else int(len(drift_df))
    except Exception as exc:  # the coverage question itself failed
        return (
            f"Drift stability: COULD NOT CHECK. The drift history could not be read "
            f"({exc}), so it is unknown whether the drift weight inside the score above "
            f"rests on any measurement."
        )
    if n_drift:
        return None
    return (
        "Drift stability: NOT ASSESSED. No drift test was run, so the 20 percent drift "
        "weight inside the score above carries a number nobody computed. It is not a "
        "drift stability of 100 and not one of 0."
    )


def _withheld_score_note(hs: HealthScore) -> Optional[str]:
    """The sentence a health gauge needs when the score behind it was WITHHELD.

    ``None`` when the score IS a measurement, so a graded figure says nothing
    extra.

    BGL grade-1 G08, 2026-09-30. ``compute_health_score`` withholds the
    composite (``score=None``, ``status='not_assessed'``) whenever the window
    holds no gradable metric record, and puts the reason in ``explanation``.
    Three of the four surfaces that draw the gauge carried that reason;
    ``create_operational_view`` and ``create_technical_view`` did not. Measured
    on an EMPTY MetricsStore before this change: both figures rendered an
    Indicator with ``value=None`` inside a red/amber/green banded dial titled
    "Health Score", neither figure contained the word "assessed" anywhere, and
    the store's own sentence ("No metric records were found in the evaluation
    window, so the health score could not be computed. This is not a score of
    100 and certifies nothing.") appeared on neither. An empty dial with no
    caption is not a refusal a reader can read; it is indistinguishable from a
    chart that failed to draw.

    The producer's own words are used verbatim rather than re-derived, so the
    figure and ``HealthScore.to_dict`` can never disagree about why.
    """
    if is_measured(hs.score):
        return None
    return (
        "Fairness Health Score: NOT ASSESSED. The gauge above is EMPTY because no "
        f"score was computed, not because the score is zero. {hs.explanation}"
    )


def _severity_counts(alerts: Sequence[dict]) -> Dict[str, int]:
    """Count alerts per severity, keeping an UNSCORED alert out of LOW.

    BGL3 operations-1, 2026-09-27. ``a.get("severity", "LOW")`` invented a
    severity in two different ways, and LOW is the calmest bucket on this chart,
    drawn in the pass green. Measured on a store holding one dict-shaped alert
    with no ``severity`` key (the shape :meth:`MetricsStore.ingest_alert`
    documents and accepts) plus one real CRITICAL: the chart rendered
    ``y=(1, 0, 1)`` with the unscored alert counted as LOW and painted
    ``#059669``, while the store's own record for that same alert carried
    ``metadata={'severity': 'UNSCORED'}``. The default also does NOT fire when
    the key is PRESENT holding ``None``, which produced a fourth bar labelled
    ``None`` against a three-entry colour list.

    "UNSCORED" is the spelling
    :class:`~vfairness.operations.monitoring.alerts.AlertPayload` already uses
    for a severity that could not be computed, so this reads that rule rather
    than inventing a second one. A severity this chart does not recognise keeps
    its own label: it IS a determination, just not one of ours.
    """
    counts: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "LOW": 0, "UNSCORED": 0}
    for a in alerts:
        sev = a.get("severity") or "UNSCORED"
        key = str(sev)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _severity_colors(counts: Dict[str, int], config: DashboardConfig) -> List[str]:
    """One colour per severity bar, in the order the counts are drawn.

    The colour list used to be three literals against a bar count that grows
    with whatever severities the alerts carry, so a fourth bar was drawn in
    plotly's default hue. UNSCORED and any unrecognised severity get the
    could-not-check slate, because neither is a graded severity.
    """
    graded = {
        "CRITICAL": config.color_fail,
        "HIGH": config.color_warn,
        "LOW": config.color_pass,
    }
    return [graded.get(k, config.color_unknown) for k in counts]


def _alert_evidence(
    store: MetricsStore,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[int, int]:
    """How many metric records in this window were compared to a threshold.

    Returns ``(n_records, n_determined)``. A record with ``alert_determined``
    False could not have raised an alert, so it cannot support a zero alert
    count either: ``n_determined == 0`` with no alerts present is "nothing was
    examined", not "nothing fired".

    BGL5 A-operations-1, 2026-09-27. This lived inline in
    :meth:`FairnessDashboard.create_alert_summary` and the Tier-3 panel drawing
    the same bars had no copy of it, so the same store answered two ways.
    Measured on an EMPTY MetricsStore before this change::

        create_alert_summary  colors ['#94a3b8'] * 4, annotation
                              'NOT AN ALL-CLEAR: no metric record in this
                              window was compared to a threshold (0 record(s)
                              present) ...'
        create_technical_view colors ['#dc2626', '#f59e0b', '#059669',
                              '#94a3b8'], annotations = the six subplot
                              titles only

    so the LOW bar was painted PASS GREEN over a store that was never given
    anything. One resolver, both surfaces, so they cannot disagree again.

    ``apply_privacy=False`` because only the determinations and the row count
    are read here, never a value, and the k-anonymity pass would warn about
    values these charts never show (compute_health_score reads the frame the
    same way for the same reason).
    """
    records = store.get_metrics(start_time=start, end_time=end, apply_privacy=False)
    n_records = int(len(records))
    if n_records and "alert_determined" in records.columns:
        n_determined = int(records["alert_determined"].fillna(False).astype(bool).sum())
    else:
        n_determined = n_records
    return n_records, n_determined


def _drift_point_colors(drift_df, config: DashboardConfig) -> Tuple[List[str], int]:
    """One marker colour per drift row, plus how many could not be checked.

    Three states, in this order, because the verdict and the magnitude are two
    different measurements and either can be missing:

    - ``drift_detected is None``: the monitor REFUSED the window. Slate.
    - ``drift_detected`` truthy: drift was found. Fail red, whatever the
      magnitude says, because that verdict WAS made.
    - ``drift_detected`` False and the score unmeasured: a clean verdict resting
      on a magnitude nobody computed. Slate, and counted as unchecked.
    - otherwise: measured, no drift. Pass green.

    BGL5 A-operations-1, 2026-09-27. The expression this replaces was
    ``color_unknown if d is None else (color_fail if d else color_pass)`` and
    ``n_unknown`` counted only ``d is None``, so the score itself was never
    consulted. Measured before this change, one drift row with
    ``overall_drift_score=nan`` and ``drift_detected=False``::

        y [nan]  colors ['#059669']  annotations ['Alert threshold']

    a PASS GREEN marker over a magnitude nobody measured, with no coverage
    sentence. After: colors ['#94a3b8'] and the coverage annotation fires. The
    determined rows are untouched: False with a measured 0.05 stays '#059669'
    and True stays '#dc2626', which is the over-correction control.

    A drift frame with no ``drift_detected`` column at all has made no verdict
    for any row, so every point is slate rather than green.
    """
    scores = list(drift_df.get("overall_drift_score", []))
    verdicts = list(drift_df.get("drift_detected", [None] * len(scores)))
    colors: List[str] = []
    n_unknown = 0
    for index, detected in enumerate(verdicts):
        score = scores[index] if index < len(scores) else float("nan")
        if detected is None:
            colors.append(config.color_unknown)
            n_unknown += 1
        elif detected:
            colors.append(config.color_fail)
        elif is_measured(score):
            colors.append(config.color_pass)
        else:
            colors.append(config.color_unknown)
            n_unknown += 1
    return colors, n_unknown


class FairnessDashboard:
    """Plotly-based interactive fairness dashboard.

    Parameters
    ----------
    store : MetricsStore
        The unified data layer backing all visualizations.
    config : DashboardConfig, optional
        Visual styling and layout configuration.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: fairness_dashboard. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        store: MetricsStore,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        self.store = store
        self.config = config or DashboardConfig()

    # Time helpers

    def _resolve_window(
        self,
        tw: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> Tuple[datetime, datetime]:
        if isinstance(tw, tuple):
            return tw
        tw = tw or self.config.default_time_window
        # G08: window bound in the store's tz base, see MetricsStore.window_now
        end = self.store.window_now()
        start = end - _WINDOW_TD.get(tw, timedelta(days=30))
        return (start, end)

    # Individual Plot Methods

    def create_health_score_gauge(
        self,
        health_score: Optional[HealthScore] = None,
    ) -> "go.Figure":
        """Single-number Fairness Health Score gauge with traffic-light color."""
        _check_plotly()
        hs = health_score or self.store.compute_health_score()
        c = self.config

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=hs.score,
                title={"text": "Fairness Health Score", "font": {"size": 18}},
                delta={"reference": 80, "increasing": {"color": c.color_pass}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#d1fae5"},
                    ],
                    "threshold": {
                        "line": {"color": c.color_fail, "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        annotations = [
            {
                "text": hs.explanation,
                "showarrow": False,
                "x": 0.5,
                "y": -0.15,
                "font": {"size": 11, "color": "#64748b"},
                "xref": "paper",
                "yref": "paper",
            }
        ]
        # The number in the middle of this gauge is the Tier-1 figure. If part
        # of its weight was never measured, that has to be on the same picture.
        coverage = _drift_coverage_note(self.store, hs)
        if coverage:
            annotations.append(
                {
                    "text": coverage,
                    "showarrow": False,
                    "x": 0.5,
                    "y": -0.32,
                    "font": {"size": 10, "color": c.color_unknown},
                    "xref": "paper",
                    "yref": "paper",
                }
            )
        fig.update_layout(
            height=300,
            width=400,
            template=c.template,
            annotations=annotations,
        )
        return fig

    def create_trend_analysis(
        self,
        metric: str,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
        groups: Optional[List[str]] = None,
    ) -> "go.Figure":
        """Line chart of a metric over time, with group breakdown and alert markers."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        df = self.store.get_metrics(start_time=start, end_time=end, metrics=[metric])

        # BGL3 operations-1, 2026-09-27. A requested group with no record in the
        # window was filtered away in silence, so the chart drew whatever
        # remained and read as a complete comparison. Measured before this
        # change, on a store holding gender_female and gender_male and a call
        # asking for ['gender_female', 'gender_nonbinary']: ONE trace,
        # gender_female, and no annotation anywhere on the figure. A reader
        # comparing two groups was shown one and told nothing.
        missing_groups: List[str] = []
        if groups:
            available = set(df["group"].astype(str).unique())
            missing_groups = [g for g in groups if str(g) not in available]
            df = df[df["group"].isin(groups) | (df["group"] == "overall")]

        fig = go.Figure()
        for grp, gdf in df.groupby("group"):
            fig.add_trace(
                go.Scatter(
                    x=gdf["timestamp"],
                    y=gdf["value"],
                    mode="lines+markers",
                    name=str(grp),
                    marker={"size": 5},
                )
            )
            # Alert markers.
            #
            # READINESS-6, 2026-09-10. `gdf[gdf["alert"]]` reads two states from
            # a column that has three. The store records `alert=None` when the
            # alert state could not be determined and carries the fact in a
            # separate `alert_determined` column, which this mask ignored, so an
            # undetermined point was drawn as an ordinary unmarked point:
            # visually identical to a measured "no alert". Mark them, in the
            # could-not-check colour and with their own legend entry, so the
            # chart distinguishes "we looked and it was fine" from "we could not
            # look".
            if "alert_determined" in gdf.columns:
                determined = gdf["alert_determined"].fillna(False).astype(bool)
            else:
                determined = gdf["alert"].notna()
            alert_flag = gdf["alert"].fillna(False).astype(bool)
            alerts = gdf[determined & alert_flag]
            undetermined = gdf[~determined]
            if len(undetermined) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=undetermined["timestamp"],
                        y=undetermined["value"],
                        mode="markers",
                        name=f"{grp} (not determined)",
                        marker={
                            "color": c.color_unknown,
                            "size": 10,
                            "symbol": "circle-open",
                            "line": {"width": 2},
                        },
                        showlegend=True,
                    )
                )
            if len(alerts) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=alerts["timestamp"],
                        y=alerts["value"],
                        mode="markers",
                        name=f"{grp} (alert)",
                        marker={"color": c.color_fail, "size": 10, "symbol": "x"},
                        showlegend=False,
                    )
                )

        if missing_groups:
            fig.add_annotation(
                text=(
                    f"NOT SHOWN: {len(missing_groups)} requested group(s) have no record of "
                    f"'{metric}' in this window ({', '.join(str(g) for g in missing_groups)}). "
                    f"They are absent from this comparison, not equal to the lines drawn."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=1.10,
                showarrow=False,
                font={"size": 11, "color": c.color_unknown},
            )
        if df.empty:
            # An empty line chart is not a trend. Say which metric and window
            # produced nothing, the way create_disparity_comparison does.
            fig.add_annotation(
                text=(
                    f"No record of '{metric}' in this window, so no trend was computed. "
                    f"This is not a flat trend."
                ),
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                font={"size": 12, "color": c.color_unknown},
            )

        fig.update_layout(
            title=f"Trend Analysis: {metric.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=metric.replace("_", " ").title(),
            height=c.height,
            width=c.width,
            template=c.template,
            hovermode="x unified",
        )
        return fig

    def create_disparity_comparison(
        self,
        metric: str,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> "go.Figure":
        """Bar chart comparing latest metric values across groups with 80% rule line."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        df = self.store.get_metrics(start_time=start, end_time=end, metrics=[metric])

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        latest = df.groupby("group")["value"].mean().reset_index()
        latest = latest[latest["group"] != "overall"]

        # Direction-aware. See the _value_color / _reference_line block above for
        # what the hardcoded ``v >= 0.8`` ladder and the "80% Rule" caption that
        # used to stand here produced on a demographic_parity store.
        colors = _value_colors(metric, latest["value"], c)

        fig = go.Figure(
            go.Bar(
                x=latest["group"],
                y=latest["value"],
                marker_color=colors,
                text=latest["value"].round(3),
                textposition="auto",
            )
        )

        # Reference bound, named for the standard it actually applies. Drawn only
        # when there IS one: an unresolvable direction has no bound, and drawing
        # a line anyway asserts a comparison nobody made.
        reference = _reference_line(metric, c)
        if reference is not None:
            y_ref, caption = reference
            fig.add_hline(
                y=y_ref,
                line_dash="dash",
                line_color=c.color_warn,
                annotation_text=caption,
                annotation_position="top left",
            )
        else:
            fig.add_annotation(
                text=(
                    f"NOT GRADED: no known better-direction for '{metric}' "
                    f"(is a lower value better or worse?), so no threshold was "
                    f"applied and no bar is coloured as a verdict."
                ),
                showarrow=False,
                x=0.5,
                y=1.06,
                xref="paper",
                yref="paper",
                font={"size": 10, "color": c.color_unknown},
            )

        fig.update_layout(
            title=f"Disparity Comparison: {metric.replace('_', ' ').title()}",
            xaxis_title="Group",
            yaxis_title=metric.replace("_", " ").title(),
            height=c.height,
            width=c.width,
            template=c.template,
        )
        return fig

    def create_intersectional_heatmap(
        self,
        metric: str,
        attribute_1: str,
        attribute_2: str,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> "go.Figure":
        """Heatmap of metric values for intersections of two attributes."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        df = self.store.get_metrics(start_time=start, end_time=end, metrics=[metric])

        # Axis membership, anchored on the "<attribute>_" PREFIX and excluding
        # the INTERSECTION groups. ``g.startswith(attribute_1)`` also matched
        # every composite name, so with four intersectional groups ingested the
        # y axis of the rendered chart read
        # ['F', 'F_race_black', 'F_race_white', 'M', 'M_race_black',
        #  'M_race_white']: the intersections were listed as if they were levels
        # of attribute_1, next to the levels they are made of.
        prefix_a = f"{attribute_1.rstrip('_')}_"
        prefix_b = f"{attribute_2.rstrip('_')}_"
        unique_groups = list(df["group"].unique())
        groups_a = [g for g in unique_groups if g.startswith(prefix_a) and prefix_b not in g]
        groups_b = [g for g in unique_groups if g.startswith(prefix_b) and prefix_a not in g]

        colorscale, reversescale = _heatmap_scale(metric)

        n_missing_cells = 0
        n_cells = 0
        fallback_reason = ""
        if not groups_a or not groups_b:
            # Build from available groups
            all_groups = sorted(df["group"].unique())
            matrix_data = df.groupby("group")["value"].mean()
            # BGL3 operations-1, 2026-09-27. This branch renders a DIFFERENT
            # chart from the one that was asked for: one column of marginal
            # per-group means, not an intersection of two attributes. It did so
            # in silence, under the title "Intersectional Analysis: <metric>"
            # and a y axis labelled with the second attribute. Measured before
            # this change, on a store holding only gender_male and gender_female
            # and a call for ("gender", "age"): z=([0.9], [0.02]) coloured on the
            # RdYlGn ramp, x=('demographic_parity_difference',), xaxis title
            # 'age', and NO annotation. A reader was shown verdict colours for
            # an intersectional analysis that was never computed.
            missing_axes = []
            if not groups_a:
                missing_axes.append(attribute_1)
            if not groups_b:
                missing_axes.append(attribute_2)
            # Worded for what the axis search actually failed to find: a
            # STANDALONE level of the attribute. A store holding only composite
            # groups has no levels either, and saying "no group belongs to
            # gender" would be false there.
            fallback_reason = (
                f"NOT AN INTERSECTIONAL ANALYSIS: this window holds no standalone level of "
                f"{' or '.join(missing_axes)} (a composite group such as "
                f"'gender_male_race_black' is not a level), so no grid of '{attribute_1}' "
                f"against '{attribute_2}' could be built. The cells below are MARGINAL "
                f"per-group values, one per group, and say nothing about any intersection."
            )
            fig = go.Figure(
                go.Heatmap(
                    z=[[matrix_data.get(g, np.nan)] for g in all_groups],
                    y=all_groups,
                    x=[metric],
                    colorscale=colorscale,
                    reversescale=reversescale,
                    text=[[f"{matrix_data.get(g, np.nan):.3f}"] for g in all_groups],
                    texttemplate="%{text}",
                )
            )
        else:
            # Build proper 2D matrix
            #
            # ``vals.append(val)`` sat OUTSIDE BOTH loops, so the list held ONE
            # element and ``zip(rows, cols, vals)`` truncated to it. Measured on
            # this repo with four real intersectional values ingested: a 6x2 grid
            # with 0 of 12 cells populated, every z entry NaN, rendering an empty
            # chart that said nothing while claiming to be an intersectional
            # analysis. Appending inside the inner loop is what makes rows, cols
            # and vals the three parallel arrays the zip below already assumes.
            rows, cols, vals = [], [], []
            for ga in groups_a:
                for gb in groups_b:
                    intersection = f"{ga}_{gb}"
                    val = df[df["group"] == intersection]["value"].mean()
                    rows.append(ga.removeprefix(prefix_a))
                    cols.append(gb.removeprefix(prefix_b))
                    # 0.0 is the BEST colour on a disparity heatmap, so an
                    # unmeasured cell was painted as the healthiest one on the
                    # chart. NaN leaves the cell blank, which is what "not
                    # measured" looks like, and ``.mean()`` of an empty
                    # selection already yields NaN rather than 0.
                    vals.append(val)

            row_labels = sorted(set(rows))
            col_labels = sorted(set(cols))
            z = np.full((len(row_labels), len(col_labels)), np.nan)
            for r, cl, v in zip(rows, cols, vals):
                z[row_labels.index(r), col_labels.index(cl)] = v

            # A blank cell is the right rendering for an unmeasured intersection
            # and it is not self-explanatory, exactly as the grey markers on the
            # drift timeline were not. Measured before this change, on a store
            # holding the four marginal groups and NO intersections: a 2x2 grid
            # with 0 of 4 cells populated, every cell reading "nan", and nothing
            # on the figure saying how much of the grid was measured.
            n_cells = int(z.size)
            n_missing_cells = int(np.count_nonzero(np.isnan(z)))

            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=col_labels,
                    y=row_labels,
                    colorscale=colorscale,
                    reversescale=reversescale,
                    text=np.round(z, 3).astype(str),
                    texttemplate="%{text}",
                )
            )

        if fallback_reason:
            fig.add_annotation(
                text=fallback_reason,
                showarrow=False,
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                font={"size": 10, "color": c.color_unknown},
            )
        elif n_missing_cells:
            fig.add_annotation(
                text=(
                    f"{n_missing_cells} of {n_cells} intersection(s) have no record in this "
                    f"window and are left blank. A blank cell is not a measurement of zero."
                ),
                showarrow=False,
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                font={"size": 10, "color": c.color_unknown},
            )

        if metric_direction(metric) is MetricDirection.UNKNOWN:
            fig.add_annotation(
                text=(
                    f"NOT GRADED: no known better-direction for '{metric}', so the "
                    f"colour ramp encodes magnitude only and marks no cell good or bad."
                ),
                showarrow=False,
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                font={"size": 10, "color": c.color_unknown},
            )

        fig.update_layout(
            title=f"Intersectional Analysis: {metric.replace('_', ' ').title()}",
            xaxis_title=attribute_2,
            yaxis_title=attribute_1,
            height=c.height,
            width=c.width,
            template=c.template,
        )
        return fig

    def create_drift_timeline(
        self,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> "go.Figure":
        """Timeline of drift scores with alert severity coloring."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        drift_df = self.store.get_drift_history(start_time=start, end_time=end)

        fig = go.Figure()
        if not drift_df.empty:
            # READINESS-6, 2026-09-10. `color_fail if d else color_pass` is two
            # states over a value that has three. `drift_detected` is
            # Optional[bool]: the monitor returns None for a metric it could not
            # examine, which the READINESS-5 wave made explicit upstream, and
            # None is falsy, so a could-not-check point was painted the SAME
            # green as a measured "no drift". Measured on this chart before the
            # fix: a measured False and a None both rendered #059669, so a
            # reader could not tell a clean window from an unexamined one.
            # BGL5 A-operations-1, 2026-09-27: the three-state expression moved
            # into _drift_point_colors and grew the missing half. It read the
            # VERDICT only, so a determined False over an unmeasured
            # overall_drift_score was painted pass green: measured, one row with
            # score=nan and drift_detected=False gave y=[nan] colors=['#059669']
            # and no coverage annotation, and now gives ['#94a3b8'] with the
            # annotation. The technical view's drift panel reads the same
            # resolver, so the two cannot colour one row two ways.
            colors, n_unknown = _drift_point_colors(drift_df, c)
            fig.add_trace(
                go.Scatter(
                    x=drift_df["timestamp"],
                    y=drift_df["overall_drift_score"],
                    mode="lines+markers",
                    marker={"color": colors, "size": 8},
                    name="Drift Score",
                )
            )
            fig.add_hline(
                y=0.3,
                line_dash="dash",
                line_color=c.color_warn,
                annotation_text="Alert threshold",
            )
            if n_unknown:
                # A neutral colour is not self-explanatory. Name the state.
                fig.add_annotation(
                    text=(
                        f"{n_unknown} of {len(drift_df)} window(s) could not be checked "
                        f"and are drawn in grey. That is not 'no drift'."
                    ),
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.08,
                    showarrow=False,
                    font={"size": 11, "color": c.color_unknown},
                )
        else:
            # BGL5 A-operations-1, 2026-09-27. An EMPTY drift table is the
            # DEFAULT state of every MetricsStore and it was the one unmeasurable
            # input this chart said nothing about. Measured before this change,
            # create_drift_timeline() on a store with no drift row:
            #
            #   traces 0, annotations [], title 'Drift Score Timeline'
            #
            # an empty frame under a title that promises a drift timeline, which
            # reads as a flat, quiet period. The two sibling charts in this same
            # file already answer the same shape of absence: create_trend_analysis
            # says "no trend was computed. This is not a flat trend." and
            # create_alert_summary says "NOT AN ALL-CLEAR". After: 0 traces and
            # this sentence. A window that HAS rows is untouched.
            fig.add_annotation(
                text=(
                    "No drift test was run in this window, so no drift timeline was "
                    "computed. An empty chart here is an absence of measurement, not "
                    "an absence of drift."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                font={"size": 11, "color": c.color_unknown},
            )

        fig.update_layout(
            title="Drift Score Timeline",
            xaxis_title="Time",
            yaxis_title="Drift Score",
            height=c.height,
            width=c.width,
            template=c.template,
        )
        return fig

    def create_alert_summary(
        self,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> "go.Figure":
        """Stacked bar chart of alerts by severity."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        alerts = self.store.get_alerts(start_time=start, end_time=end)

        severity_counts = _severity_counts(alerts)

        # BGL3 operations-1, 2026-09-27. Three zero counts were the output for
        # "monitored, nothing fired" AND for "nothing was ever ingested".
        # Measured before this change: an EMPTY MetricsStore and a store holding
        # four measured clean records both rendered y=(0, 0, 0) with
        # marker_color=('#dc2626', '#f59e0b', '#059669') and NO annotation, byte
        # for byte, while the store itself warned on the same call that "the
        # empty result means nothing has been ingested. It is NOT a finding that
        # no alert fired." That warning died at the chart.
        #
        # The evidence question is how many metric records in this window were
        # compared to a threshold at all: a record with alert_determined False
        # could not have raised an alert, so it cannot support a zero either.
        # The counting moved into _alert_evidence on 2026-09-27, unchanged, so
        # the Tier-3 panel drawing these same bars reads it from one place
        # instead of painting PASS GREEN over a store nobody fed.
        n_records, n_determined = _alert_evidence(self.store, start, end)
        n_undetermined = n_records - n_determined

        colors = _severity_colors(severity_counts, c)
        no_evidence = n_determined == 0 and not alerts
        if no_evidence:
            # Not a verdict palette over a window nobody examined.
            colors = [c.color_unknown] * len(severity_counts)

        fig = go.Figure(
            go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=colors,
                text=list(severity_counts.values()),
                textposition="auto",
            )
        )
        if no_evidence:
            fig.add_annotation(
                text=(
                    f"NOT AN ALL-CLEAR: no metric record in this window was compared to "
                    f"a threshold ({n_records} record(s) present), so these zeros mean "
                    f"nothing was examined, not that no alert fired."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=1.12,
                showarrow=False,
                font={"size": 11, "color": c.color_unknown},
            )
        elif n_undetermined:
            fig.add_annotation(
                text=(
                    f"{n_undetermined} of {n_records} metric record(s) in this window were "
                    f"never compared to a threshold, so they could not raise an alert and "
                    f"are counted in no bar here."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=1.12,
                showarrow=False,
                font={"size": 11, "color": c.color_unknown},
            )
        fig.update_layout(
            title="Alert Summary by Severity",
            xaxis_title="Severity",
            yaxis_title="Count",
            height=400,
            width=500,
            template=c.template,
        )
        return fig

    def create_subgroup_sunburst(
        self,
        metric: str,
        time_window: Union[TimeWindow, Tuple[datetime, datetime], None] = None,
    ) -> "go.Figure":
        """Sunburst chart for hierarchical subgroup drill-down."""
        _check_plotly()
        start, end = self._resolve_window(time_window)
        c = self.config
        df = self.store.get_metrics(start_time=start, end_time=end, metrics=[metric])

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data", showarrow=False)
            return fig

        group_means = df.groupby("group")["value"].mean().reset_index()
        group_means = group_means[group_means["group"] != "overall"]

        # Build sunburst: root → attribute → group
        labels, parents, values = ["Fairness"], [""], [0]
        for _, row in group_means.iterrows():
            parts = str(row["group"]).split("_", 1)
            attr = parts[0] if len(parts) > 1 else "groups"

            if attr not in labels:
                labels.append(attr)
                parents.append("Fairness")
                values.append(0)

            labels.append(row["group"])
            parents.append(attr)
            values.append(round(row["value"], 4))

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total" if all(v >= 0 for v in values) else "remainder",
                # Direction-aware, for the same reason as the heatmaps. Measured
                # 2026-09-10: this scale is currently INERT, because
                # ``marker.colors`` is never set, so plotly falls back to the
                # qualitative palette. It is routed through the shared resolver
                # anyway, so that whoever wires numeric colours in here does not
                # inherit a hardcoded ramp that reads the largest gap as the
                # greenest wedge.
                marker={
                    "colorscale": _heatmap_scale(metric)[0],
                    "reversescale": _heatmap_scale(metric)[1],
                },
            )
        )
        fig.update_layout(
            title=f"Subgroup Drill-Down: {metric.replace('_', ' ').title()}",
            height=c.height,
            width=c.width,
            template=c.template,
        )
        return fig

    # Tier-Based Composite Views

    def create_executive_view(self) -> "go.Figure":
        """Tier 1: Health gauge + 3 KPI indicator cards."""
        _check_plotly()
        c = self.config
        hs = self.store.compute_health_score()

        fig = make_subplots(
            rows=1,
            cols=4,
            specs=[[{"type": "indicator"}] * 4],
            column_widths=[0.4, 0.2, 0.2, 0.2],
        )

        # Health gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=hs.score,
                title={"text": "Health Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#d1fae5"},
                    ],
                },
            ),
            row=1,
            col=1,
        )

        # KPI 1: Metric compliance
        #
        # BGL3 operations-1, 2026-09-27. `.get(name, 0)` is the variable-key
        # neutral default this function already removed from the Trend KPI two
        # cards along, left standing on the one card that shows a percentage. A
        # withheld health score carries NO components at all: store.py returns
        # `components={}` on three separate could-not-check paths. Measured on an
        # empty MetricsStore before this change: `value=0` with
        # `delta={'reference': 80}`, i.e. a large 0 labelled Metric Compliance
        # and a -80 delta, printed for a quantity nobody computed, BESIDE a
        # health gauge that correctly rendered `value=None`.
        #
        # tests/test_bgl2_reporting_and_compliance.py states the house rule this
        # restores: "an indicator reading 0 is a measurement. None is the absence
        # of one." Rendered like the unknown trend below: no delta, because a
        # delta asserts a comparison that was never made.
        #
        # float() plus isfinite, never isinstance(v, (int, float)): the store
        # hands back np.float64 for metric_compliance and that test REJECTS it,
        # which would discard a real measurement and report a could-not-check.
        raw_compliance = hs.components.get("metric_compliance")
        compliance: Optional[float]
        try:
            compliance = None if raw_compliance is None else float(raw_compliance)
        except (TypeError, ValueError):
            compliance = None
        if compliance is None or not np.isfinite(compliance):
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=float("nan"),
                    title={"text": "Metric Compliance<br><sub>not assessed</sub>"},
                    number={
                        "font": {"color": c.color_unknown},
                        "prefix": "not assessed",
                        "valueformat": "",
                    },
                ),
                row=1,
                col=2,
            )
        else:
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=compliance,
                    title={"text": "Metric Compliance"},
                    delta={"reference": 80},
                ),
                row=1,
                col=2,
            )

        # KPI 2: Alerts
        #
        # READINESS-6, 2026-09-10. "0 Active Alerts" in PASS green is a claim
        # that nothing is wrong. HealthScore already carries `n_not_assessable`,
        # the count of metrics that could not be assessed at all, and this KPI
        # ignored it: a window where every metric was unassessable rendered
        # exactly the same reassuring green zero as a window that was measured
        # and clean. Zero alerts out of nothing examined is not an all-clear.
        n_unassessable = getattr(hs, "n_not_assessable", 0) or 0
        # BGL3 operations-1, 2026-09-27. The guard above keyed ONLY on
        # n_not_assessable, and the store does not set that count on the path
        # where NOTHING was ingested: an empty MetricsStore answers
        # score=None, status='not_assessed', n_metrics=0, n_not_assessable=0.
        # Measured before this change, on an empty store and on a store holding
        # two measured clean records: BOTH rendered value=0 with
        # number.font.color '#059669', byte for byte. The plainest instance of
        # the defect this KPI was fixed for walked straight through its own
        # guard. A withheld score is the third state too, so it is read here.
        no_evidence = hs.score is None or hs.status == "not_assessed" or hs.n_metrics == 0
        if hs.n_alerts > 0:
            alert_color = c.color_fail
        elif n_unassessable > 0 or no_evidence:
            alert_color = c.color_unknown
        else:
            alert_color = c.color_pass
        if n_unassessable:
            alert_title = f"Active Alerts<br><sub>{n_unassessable} metric(s) not assessable</sub>"
        elif no_evidence:
            alert_title = "Active Alerts<br><sub>no metric record was assessed</sub>"
        else:
            alert_title = "Active Alerts"
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=hs.n_alerts,
                title={"text": alert_title},
                number={"font": {"color": alert_color}},
            ),
            row=1,
            col=3,
        )

        # KPI 3: Trend
        # READINESS-6, 2026-09-10. `.get(hs.trend, 0)` is the variable-key
        # neutral default, and 0 is not neutral here: it is the value of
        # "stable". The store answers three states and returns the string
        # "unknown" when it could not compute a trend (READINESS-5 made that
        # explicit), and this map turned "unknown" into the SAME number and the
        # SAME zero delta as a measured "stable". Only the suffix text differed,
        # and an indicator is read as a number.
        #
        # An unknown trend now renders no number and no delta, because a delta
        # asserts a comparison that was never made, and it is drawn in the
        # could-not-check colour with the reason on its face.
        trend_val = {"improving": 1, "stable": 0, "degrading": -1}.get(hs.trend)
        if trend_val is None:
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=float("nan"),
                    title={"text": "Trend"},
                    number={
                        "font": {"color": c.color_unknown},
                        "prefix": "not determined",
                        "valueformat": "",
                    },
                ),
                row=1,
                col=4,
            )
        else:
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=trend_val,
                    title={"text": "Trend"},
                    delta={"reference": 0},
                    number={"suffix": f" ({hs.trend})"},
                ),
                row=1,
                col=4,
            )

        exec_annotations = [
            {
                "text": hs.explanation,
                "showarrow": False,
                "x": 0.5,
                "y": -0.2,
                "font": {"size": 11, "color": "#64748b"},
                "xref": "paper",
                "yref": "paper",
            }
        ]
        coverage = _drift_coverage_note(self.store, hs)
        if coverage:
            exec_annotations.append(
                {
                    "text": coverage,
                    "showarrow": False,
                    "x": 0.5,
                    "y": -0.36,
                    "font": {"size": 10, "color": c.color_unknown},
                    "xref": "paper",
                    "yref": "paper",
                }
            )
        fig.update_layout(
            height=300,
            width=c.width,
            template=c.template,
            title_text="Executive Fairness Overview",
            annotations=exec_annotations,
        )
        return fig

    def create_operational_view(
        self,
        metrics: Optional[List[str]] = None,
    ) -> "go.Figure":
        """Tier 2: 2×2 grid of health, trend, disparity and alerts."""
        _check_plotly()
        c = self.config
        hs = self.store.compute_health_score()
        metric_names = metrics or self.store.get_metric_names()
        primary_metric = metric_names[0] if metric_names else "demographic_parity"

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=[
                "Health Score",
                f"Trend: {primary_metric}",
                "Group Disparity",
                "Alert Timeline",
            ],
        )

        # (1,1) Health gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=hs.score,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#d1fae5"},
                    ],
                },
            ),
            row=1,
            col=1,
        )

        # (1,2) Trend
        start, end = self._resolve_window()
        df = self.store.get_metrics(
            start_time=start,
            end_time=end,
            metrics=[primary_metric],
        )
        for grp, gdf in df.groupby("group"):
            fig.add_trace(
                go.Scatter(
                    x=gdf["timestamp"],
                    y=gdf["value"],
                    mode="lines",
                    name=str(grp),
                ),
                row=1,
                col=2,
            )

        # (2,1) Disparity bars
        if not df.empty:
            means = df.groupby("group")["value"].mean().reset_index()
            means = means[means["group"] != "overall"]
            # Direction-aware, and the SAME resolver the standalone disparity
            # chart uses, so the operational tile and the drill-down can never
            # colour one value two ways. The two-state ``v >= 0.8 -> green else
            # amber`` ladder that stood here painted demographic_parity 0.95
            # (a near-total gap) '#059669' and 0.02 (near perfect) '#f59e0b'.
            # ``primary_metric`` is the store's FIRST metric name, so this tile
            # picks its metric up off the data and cannot assume a family.
            fig.add_trace(
                go.Bar(
                    x=means["group"],
                    y=means["value"],
                    marker_color=_value_colors(primary_metric, means["value"], c),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            reference = _reference_line(primary_metric, c)
            if reference is not None:
                y_ref, caption = reference
                fig.add_hline(
                    y=y_ref,
                    line_dash="dash",
                    line_color=c.color_warn,
                    annotation_text=caption,
                    annotation_position="top left",
                    annotation_font_size=9,
                    row=2,
                    col=1,
                    # This grid holds an Indicator in (1,1), and plotly's
                    # default emptiness probe reads ``xaxis`` off every trace in
                    # the figure, which an Indicator does not have. Measured:
                    # PlotlyKeyError "Invalid property specified for object of
                    # type plotly.graph_objs.Indicator: 'xaxis'", i.e. the whole
                    # operational view raised instead of rendering. The probe is
                    # only an optimisation; the target subplot is non-empty here
                    # by construction, inside ``if not df.empty``.
                    exclude_empty_subplots=False,
                )

        # (2,2) Drift timeline
        drift_df = self.store.get_drift_history(start_time=start, end_time=end)
        if not drift_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=drift_df["timestamp"],
                    y=drift_df["overall_drift_score"],
                    mode="lines+markers",
                    name="Drift Score",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        # Same disclosure as the standalone gauge, on the tile that draws the
        # same composite. add_annotation rather than update_layout(annotations=)
        # because this grid's subplot titles ARE layout annotations and
        # assigning the list would delete them.
        coverage = _drift_coverage_note(self.store, hs)
        if coverage:
            fig.add_annotation(
                text=coverage,
                showarrow=False,
                x=0,
                y=-0.08,
                xref="paper",
                yref="paper",
                font={"size": 10, "color": c.color_unknown},
            )
        # BGL grade-1 G08. Why the (1,1) gauge is blank, on the figure. See
        # _withheld_score_note: this view drew value=None with no caption at all.
        withheld = _withheld_score_note(hs)
        if withheld:
            fig.add_annotation(
                text=withheld,
                showarrow=False,
                x=0,
                y=-0.11,
                xref="paper",
                yref="paper",
                align="left",
                font={"size": 10, "color": c.color_unknown},
            )

        fig.update_layout(
            height=c.height * 2,
            width=c.width,
            template=c.template,
            title_text="Operational Fairness Dashboard",
            showlegend=True,
        )
        return fig

    def create_technical_view(
        self,
        metrics: Optional[List[str]] = None,
    ) -> "go.Figure":
        """Tier 3: Comprehensive dashboard with all available plots."""
        _check_plotly()
        c = self.config
        metric_names = metrics or self.store.get_metric_names()
        primary = metric_names[0] if metric_names else "demographic_parity"

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=[
                "Health Score",
                f"Trend: {primary}",
                "Group Disparity",
                "Alert Summary",
                "Drift Timeline",
                "Metric Correlation",
            ],
        )

        hs = self.store.compute_health_score()
        start, end = self._resolve_window()

        # (1,1) Health gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=hs.score,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#d1fae5"},
                    ],
                },
            ),
            row=1,
            col=1,
        )

        # (1,2) Trend
        df = self.store.get_metrics(
            start_time=start,
            end_time=end,
            metrics=[primary],
        )
        for grp, gdf in df.groupby("group"):
            fig.add_trace(
                go.Scatter(
                    x=gdf["timestamp"],
                    y=gdf["value"],
                    mode="lines",
                    name=str(grp),
                ),
                row=1,
                col=2,
            )

        # (2,1) Disparity
        if not df.empty:
            means = df.groupby("group")["value"].mean().reset_index()
            means = means[means["group"] != "overall"]
            if not means.empty:
                fig.add_trace(
                    go.Bar(
                        x=means["group"],
                        y=means["value"],
                        marker_color=c.color_neutral,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # (2,2) Alert summary
        #
        # The SAME resolver create_alert_summary uses, so the tile and the
        # drill-down cannot bucket one alert two ways. This copy of the
        # `.get("severity", "LOW")` default counted an unscored alert as LOW and
        # painted it pass green exactly as the standalone chart did.
        #
        # BGL5 A-operations-1, 2026-09-27: the COUNTS were shared and the
        # EVIDENCE question was not, so this panel still painted a verdict
        # palette over a store nobody had fed. Measured on an EMPTY MetricsStore
        # before this change: x=['CRITICAL','HIGH','LOW','UNSCORED'] y=[0,0,0,0]
        # marker_color=['#dc2626','#f59e0b','#059669','#94a3b8'] and the only
        # annotations on the figure were its six subplot titles, while
        # create_alert_summary on the identical store neutralised all four bars
        # and carried "NOT AN ALL-CLEAR ...". After: all four slate, and that
        # sentence on this figure too, added with add_annotation so the subplot
        # titles survive (update_layout(annotations=) would delete them).
        alerts = self.store.get_alerts(start_time=start, end_time=end)
        sev_counts = _severity_counts(alerts)
        n_records, n_determined = _alert_evidence(self.store, start, end)
        n_undetermined = n_records - n_determined
        sev_colors = _severity_colors(sev_counts, c)
        no_alert_evidence = n_determined == 0 and not alerts
        if no_alert_evidence:
            sev_colors = [c.color_unknown] * len(sev_counts)
        fig.add_trace(
            go.Bar(
                x=list(sev_counts.keys()),
                y=list(sev_counts.values()),
                marker_color=sev_colors,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        if no_alert_evidence:
            fig.add_annotation(
                text=(
                    f"NOT AN ALL-CLEAR: no metric record in this window was compared to "
                    f"a threshold ({n_records} record(s) present), so these zeros mean "
                    f"nothing was examined, not that no alert fired."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.04,
                showarrow=False,
                font={"size": 10, "color": c.color_unknown},
            )
        elif n_undetermined:
            fig.add_annotation(
                text=(
                    f"{n_undetermined} of {n_records} metric record(s) in this window were "
                    f"never compared to a threshold, so they could not raise an alert and "
                    f"are counted in no bar here."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.04,
                showarrow=False,
                font={"size": 10, "color": c.color_unknown},
            )

        # (3,1) Drift timeline
        drift_df = self.store.get_drift_history(start_time=start, end_time=end)
        if not drift_df.empty:
            # The same three-state marker colours the standalone timeline got in
            # READINESS-6 and extended here in BGL5: this panel drew the scatter
            # with NO marker colours at all, so a refused window and a measured
            # clean one were the same dot.
            drift_colors, _n_unknown_drift = _drift_point_colors(drift_df, c)
            fig.add_trace(
                go.Scatter(
                    x=drift_df["timestamp"],
                    y=drift_df["overall_drift_score"],
                    mode="lines+markers",
                    marker={"color": drift_colors, "size": 8},
                    name="Drift",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

        # (3,2) Metric correlation heatmap
        #
        # BGL7 A-operations-1-w2, 2026-09-29. THE ROW COUNT IS THE SIBLING OF THE
        # COLUMN COUNT, and only the column count was guarded. `pivot.shape[1] > 1`
        # asks HOW MANY METRICS there are; nothing asked how many OBSERVATIONS the
        # coefficient rests on. Any two points are perfectly collinear, so a Pearson
        # r over two observations is exactly +1 or -1 by construction whatever the
        # numbers are, and is not a measurement of association at all. Measured
        # before this change on a store holding two metrics at TWO timestamps
        # (ingested with group_size_col so the privacy layer releases the values),
        # create_technical_view():
        #   heatmaps 1, z [[1.0, -1.0], [-1.0, 1.0]], labels ['dp_diff', 'eo_diff']
        #   annotations mentioning correlation / points / observations:
        #       ['Metric Correlation']   <- the subplot title, and nothing else
        # so a perfect -1.0000 anticorrelation was painted at the extreme of a
        # colorscale pinned zmin=-1 / zmax=1, from two points, with nothing on the
        # figure saying how many points it used. n=3 gave the same +-1.0000.
        #
        # This is the same class as the alert-evidence and drift-coverage halves of
        # this very figure: a value that could not be measured, published as a
        # measurement. THREE STATES, and the could-not-check is annotated where a
        # reader looks, because a withheld panel and an empty window look identical
        # on a grid of six subplots.
        all_df = self.store.get_metrics(start_time=start, end_time=end)
        corr_note = ""
        if not all_df.empty and len(metric_names) > 1:
            pivot = all_df.pivot_table(
                index="timestamp",
                columns="metric_name",
                values="value",
                aggfunc="mean",
            ).dropna(axis=1, how="all")
            if pivot.shape[1] > 1:
                # Imported here rather than at module scope: it is the ONE statement
                # of this floor in the package (`_PEARSON_MIN_OVERLAP` in the same
                # module records the hard collinearity bound, and
                # `bias_detection.statistical._MIN_COMPARABLE_ROWS` the sibling
                # screen's copy), and a second copy of one rule is the cali-BRATIO-n
                # shape recorded in CLAUDE.md. Module-scope would add ~3s to every
                # import of this dashboard for a single integer.
                from ...preprocessing.feature_engineering.correlation import MIN_SAMPLE_SIZE

                n_obs = int(pivot.shape[0])
                if n_obs < MIN_SAMPLE_SIZE:
                    # WITHHELD. No matrix is painted, because a +-1.0000 at the
                    # extreme of a pinned colorscale is the strongest claim this
                    # figure can make and it would rest on nothing.
                    corr_note = (
                        f"Metric Correlation: NOT COMPUTED. Only {n_obs} observation(s) "
                        f"over the {pivot.shape[1]} metrics in this window, below the "
                        f"{MIN_SAMPLE_SIZE} this panel needs. Any two points are perfectly "
                        f"collinear, so a Pearson r over so few is arithmetic and not a "
                        f"measurement of association. This is a could-not-check, not an "
                        f"absence of correlation."
                    )
                else:
                    # `min_periods` so a PAIR with too little overlap comes out NaN
                    # (rendered blank) instead of borrowing the frame's row count.
                    corr = pivot.corr(min_periods=MIN_SAMPLE_SIZE)
                    n_pairs = pivot.shape[1] * (pivot.shape[1] - 1) // 2
                    # The UPPER TRIANGLE only, one cell per pair. Counting every NaN
                    # and halving it is wrong as soon as a metric does not vary,
                    # because its DIAGONAL cell is NaN too and has no pair.
                    _corr_values = np.asarray(corr.values, dtype=float)
                    n_ungraded = int(
                        np.isnan(_corr_values)[np.triu_indices_from(_corr_values, k=1)].sum()
                    )
                    fig.add_trace(
                        go.Heatmap(
                            z=corr.values,
                            x=corr.columns.tolist(),
                            y=corr.index.tolist(),
                            colorscale="RdBu",
                            zmin=-1,
                            zmax=1,
                            showscale=False,
                        ),
                        row=3,
                        col=2,
                    )
                    # The sample size travels WITH the matrix. A reader cannot tell
                    # a coefficient over 200 observations from one over 10 by
                    # looking at the colour.
                    corr_note = f"Metric Correlation over {n_obs} observation(s)."
                    if n_ungraded:
                        corr_note += (
                            f" {n_ungraded} of {n_pairs} metric pair(s) are BLANK: fewer "
                            f"than {MIN_SAMPLE_SIZE} overlapping observations, or a metric "
                            f"that does not vary. Blank is could-not-check, not zero "
                            f"correlation."
                        )

        # The SAME disclosure the standalone gauge, the executive view and the
        # operational view all carry for this composite. This was its THIRD
        # rendering and the only one without it. Measured before this change, on
        # a store with four determined clean metric records and ZERO drift rows:
        # the (1,1) gauge rendered value=100.0, the annotations were the six
        # subplot titles, and the word drift appeared nowhere on the figure,
        # while the other three surfaces said "Drift stability: NOT ASSESSED ..."
        # for that identical store. add_annotation, not update_layout, because
        # this grid's subplot titles ARE layout annotations.
        coverage = _drift_coverage_note(self.store, hs)
        if coverage:
            fig.add_annotation(
                text=coverage,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.08,
                showarrow=False,
                font={"size": 10, "color": c.color_unknown},
            )
        # BGL7 A-operations-1-w2. The sample size the (3,2) panel rests on, or the
        # reason it is empty. Beside the drift note and for the same reason: the
        # figure has to say what it could not measure, not only what it could.
        if corr_note:
            fig.add_annotation(
                text=corr_note,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.11,
                showarrow=False,
                font={"size": 10, "color": c.color_unknown},
            )
        # BGL grade-1 G08. Why the (1,1) gauge is blank, on the figure. This was
        # the SECOND rendering of the same omission (see _withheld_score_note):
        # both this view and the operational one drew value=None with no caption,
        # while the standalone gauge and the executive view both carried the
        # store's reason.
        withheld = _withheld_score_note(hs)
        if withheld:
            fig.add_annotation(
                text=withheld,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.14,
                showarrow=False,
                align="left",
                font={"size": 10, "color": c.color_unknown},
            )

        fig.update_layout(
            height=c.height * 3,
            width=c.width,
            template=c.template,
            title_text="Technical Fairness Dashboard",
            showlegend=True,
        )
        return fig

    # Export

    def to_html(
        self,
        figure: Optional["go.Figure"] = None,
        tier: Optional[str] = None,
        include_js: bool = True,
    ) -> str:
        """Export a figure (or auto-generate from tier) as standalone HTML.

        Parameters
        ----------
        figure : go.Figure, optional
            Specific figure to export.  If ``None``, generates from *tier*.
        tier : str, optional
            ``"executive"``, ``"operational"``, or ``"technical"``.
        include_js : bool
            Embed plotly.js for offline viewing (default ``True``).
        """
        _check_plotly()
        if figure is None:
            tier = tier or "operational"
            dispatch = {
                "executive": self.create_executive_view,
                "operational": self.create_operational_view,
                "technical": self.create_technical_view,
            }
            figure = dispatch[tier]()
        return figure.to_html(include_plotlyjs=include_js, full_html=True)

    def get_explanation(self) -> str:
        """Generate an NLG explanation of the current dashboard state."""
        hs = self.store.compute_health_score()
        return hs.explanation
