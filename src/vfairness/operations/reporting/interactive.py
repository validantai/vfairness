"""
Unit 3: Performance Dashboards: InteractiveDashboard
=====================================================

Maximum-interactivity layer for fairness monitoring.

The ``InteractiveDashboard`` provides two modes of operation:

1. **Dash application** (when ``dash`` is installed): A full server-side
   application with sidebar filters (metric selector, date range, group
   checklist), a what-if threshold slider, and auto-updating Plotly charts.
   Launch with ``run(port=8050)``.

2. **Standalone HTML** (no server required): Pre-renders all plot states as
   JSON and uses embedded JavaScript with ``Plotly.react()`` to swap traces
   on dropdown/button changes.  Output a self-contained ``.html`` file that
   works offline in any modern browser.

Privacy guardrails:

- Groups smaller than ``min_group_display_size`` are suppressed from all
  interactive views.
- Confidence intervals are shown on all trend lines to convey uncertainty.

References
----------
Plotly Dash documentation; Google What-If Tool; Module 4 Unit 3 curriculum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from vfairness._triage import is_measured, unmeasurable_reason
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    metric_direction,
)

from .dashboard import DashboardConfig, FairnessDashboard, TimeWindow, _check_plotly
from .store import MetricsStore

# Optional Dash import

_DASH_AVAILABLE = False
try:
    import dash
    import plotly.graph_objects as go
    from dash import Input, Output, State, dcc, html  # noqa: F401  # availability probe
    from plotly.subplots import make_subplots

    _DASH_AVAILABLE = True
except ImportError:
    dash = None  # type: ignore[assignment]
    dcc = None  # type: ignore[assignment]
    html = None  # type: ignore[assignment]

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots  # noqa: F401  # availability probe

    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]


def _check_dash():
    if not _DASH_AVAILABLE:
        raise ImportError(
            "Dash is required for the interactive server.\n"
            "Install it with: pip install dash>=2.0  "
            "or: pip install vfairness[interactive]"
        )


# Config


@dataclass
class InteractiveConfig:
    """Configuration for :class:`InteractiveDashboard`.

    Parameters
    ----------
    min_group_display_size : int
        Suppress groups with fewer members from all views (default 10).
    show_confidence_intervals : bool
        Show 95% CI ribbons on trend lines (default True).
    default_metric : str, optional
        Pre-select this metric on launch.
    default_time_window : TimeWindow
        Initial time filter.
    plotly_cdn : str
        Fallback CDN URL for plotly.js in standalone HTML; used only if the
        local plotly.js bundle cannot be embedded inline.
    dash_debug : bool
        Start Dash in debug mode.
    """

    min_group_display_size: int = 10
    show_confidence_intervals: bool = True
    default_metric: Optional[str] = None
    default_time_window: TimeWindow = TimeWindow.LAST_30D
    plotly_cdn: str = "https://cdn.plot.ly/plotly-2.30.0.min.js"
    dash_debug: bool = False


# What-If Simulation


def _simulatable_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a metric window into the rows a threshold simulation may use.

    R-2, consumer side. ``alert`` answers ONE question: did an alert FIRE.
    ``bool(None)`` is False, so a record nobody ever compared to a threshold (a
    FairnessMonitor custom metric, a per-group positive rate) arrived here
    indistinguishable from a compared-and-clean one. It was counted as "no
    alert today" in the baseline AND re-graded against the proposed threshold,
    so the simulation projected alerts for records the alert rule never
    evaluates, and reported a graded percentage built on them. Measured
    2026-09-10: two graded clean rows plus three never-compared ones at 0.10 to
    0.15 reported ``projected_alerts: 3`` and ``groups_impacted:
    ["never_compared"]`` when no graded row was anywhere near the threshold.

    The store already ships the column that separates the two
    (``alert_determined``, see ``MetricsStore.get_metrics``) and already keeps
    undetermined records out of the health score's compliance mean, so this
    reads the evidence the producer wrote rather than inventing a second rule.

    Two ways a row cannot be simulated, matching ``store._compliance_rows``:
    no alert determination at all, or a value that is not a real number (H-10,
    NaN), which ``value < threshold`` silently answers False for, i.e. counts
    as "would not alert".

    Frames built before ``alert_determined`` existed carry no determinations to
    read; every row there is taken as graded, exactly as before.

    Returns ``(simulatable, left_out)``.
    """
    keep = pd.Series(True, index=df.index)
    if "alert_determined" in df.columns:
        keep &= df["alert_determined"].astype(bool)
    if "value" in df.columns:
        keep &= df["value"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))
    return df[keep], df[~keep]


def simulate_threshold_change(
    store: MetricsStore,
    metric_name: str,
    new_threshold: float,
    time_window: timedelta = timedelta(days=30),
) -> Dict[str, Any]:
    """Estimate the impact of changing a metric's alert threshold.

    Only records the alert rule actually graded can be re-graded against a
    proposed threshold (see :func:`_simulatable_rows`). Returns a dict with:

    - ``current_alerts``: number of alerts under current behaviour, or ``None``
      when the window holds no record of the metric at all, and when no record
      in the window was simulatable
    - ``projected_alerts``: estimated alerts under ``new_threshold``, or
      ``None`` on the same could-not-check
    - ``groups_impacted``: groups whose mean value falls below the new
      threshold, or ``None`` on the same could-not-check (an empty list would
      read as "no group is impacted", which is a measurement)
    - ``change_abs``: absolute change in alert count (projected - current), or
      ``None`` on the same could-not-check
    - ``change_pct``: percentage change in alert count, or ``None`` when
      ``current_alerts`` is 0 and alerts appear (a percentage of zero is
      undefined; reporting 0.0 there would hide a real increase) and on the
      same could-not-check
    - ``records_simulated`` / ``records_not_simulated`` /
      ``not_simulated_groups``: how much of the window the counts above rest
      on, and which groups were left out entirely

    This is a *simulation*: it re-evaluates historical data against the
    proposed threshold without changing the store's state.

    ``new_threshold`` must itself be a real finite number. It is the one input
    the caller supplies directly and it was the one input never checked; see the
    guard below for the measurement.
    """
    # BGL5 AUDIT, 2026-09-27. Three could-not-check branches, a metric-direction
    # contract and a per-row grading check, and no test of the BOUND the whole
    # projection is made against. Measured on a window holding 2 real alerts out
    # of 3 graded records (which at the sane threshold 0.5 reports current 2,
    # projected 2, groups_impacted ['b'], change 0, 0.0%):
    #
    #   new_threshold = nan  -> current_alerts 2, projected_alerts 0,
    #                           groups_impacted [], change_abs -2,
    #                           change_pct -100.0, records_simulated 3,
    #                           not_simulated_reason '', no warning
    #   new_threshold = inf  -> the same
    #   new_threshold = -inf -> projected_alerts 3, groups_impacted ['a','b'],
    #                           change_abs +1, change_pct +50.0
    #
    # Every ``value > nan`` is False, so NO record was compared to anything, and
    # an operator using this to CHOOSE a threshold is told the change removes
    # both alerts, a 100 percent reduction. groups_impacted came back [], which
    # this function's own docstring says "would read as no group is impacted,
    # which is a measurement". After this guard all three return every count as
    # None with the reason naming the value, and the sane 0.5 case is unchanged
    # (current 2, projected 2, groups_impacted ['b'], records_simulated 3).
    #
    # ``is_measured`` is the library's canonical rule and also refuses None, a
    # bool and a string, none of which is a bound either.
    if not is_measured(new_threshold):
        return {
            "metric": metric_name,
            "new_threshold": new_threshold,
            "current_alerts": None,
            "projected_alerts": None,
            "groups_impacted": None,
            "change_abs": None,
            "change_pct": None,
            "records_simulated": 0,
            "records_not_simulated": 0,
            "not_simulated_groups": [],
            "direction": metric_direction(metric_name).value,
            "not_simulated_reason": (
                "COULD NOT SIMULATE: the proposed threshold is {0!r} "
                "({1}), which no value can be compared against, so no record was "
                "re-graded and no count below rests on a comparison. This is not a "
                "finding that the threshold change would remove every alert; supply a "
                "real finite bound.".format(new_threshold, unmeasurable_reason(new_threshold))
            ),
        }

    # G08: window bound in the store's tz base, see MetricsStore.window_now
    now = store.window_now()
    start = now - time_window
    df = store.get_metrics(
        start_time=start,
        end_time=now,
        metrics=[metric_name],
        apply_privacy=False,
    )

    if df.empty:
        # COULD NOT CHECK, and this is the COMMONEST way to reach this
        # function: a fresh store, a window that predates the data, or a typo
        # in metric_name all land here. It used to answer "0 current alerts,
        # 0 projected, no group impacted, 0.0% change", a fully quantified
        # what-if built on nothing, and byte-identical for all three causes.
        # Decisive case measured 2026-09-16: a store holding 24 graded
        # readings of which 12 ARE alerts, queried on a window that misses
        # them, reported "Alerts 0 -> 0, change 0 (0.0%)", which is flatly
        # false of that store.
        #
        # The same doctrine is stated twice inside this function already: the
        # docstring says an empty groups_impacted "would read as no group is
        # impacted, which is a measurement", and the simulatable.empty branch
        # below says zeros there "would be graded as a measurement". This
        # branch now returns that same shape, direction included, so a metric
        # whose direction is unknown is not silently re-labelled by an empty
        # window.
        return {
            "metric": metric_name,
            "new_threshold": new_threshold,
            "current_alerts": None,
            "projected_alerts": None,
            "groups_impacted": None,
            "change_abs": None,
            "change_pct": None,
            "records_simulated": 0,
            "records_not_simulated": 0,
            "not_simulated_groups": [],
            "direction": metric_direction(metric_name).value,
            "not_simulated_reason": (
                "COULD NOT CHECK: no record of metric {0!r} exists in the requested "
                "window, so there is no baseline to re-grade against a proposed "
                "threshold. Nothing was simulated. This is not a finding that the "
                "threshold change would have no effect; check the metric name and "
                "the time window.".format(metric_name)
            ),
        }

    simulatable, left_out = _simulatable_rows(df)
    not_simulated_groups: List[str] = (
        sorted({str(g) for g in left_out["group"]}) if "group" in left_out.columns else []
    )

    if simulatable.empty:
        # COULD NOT CHECK. Every record in this window is one no alert rule
        # ever graded (or one whose value could not be computed), so there is
        # no baseline for a proposed threshold to be compared against. Zero
        # current alerts and N projected ones would be counts of rows the
        # alert rule does not evaluate, and the percentage built on them would
        # be graded as a measurement. The counts are absent instead, and the
        # two record counts below say how large the gap is.
        return {
            "metric": metric_name,
            "new_threshold": new_threshold,
            "current_alerts": None,
            "projected_alerts": None,
            "groups_impacted": None,
            "change_abs": None,
            "change_pct": None,
            "records_simulated": 0,
            "records_not_simulated": int(len(left_out)),
            "not_simulated_groups": not_simulated_groups,
            "direction": metric_direction(metric_name).value,
            "not_simulated_reason": (
                "COULD NOT CHECK: every record of metric {0!r} in this window is one "
                "no alert rule ever graded, so there is no baseline to re-grade "
                "against a proposed threshold. Nothing was simulated.".format(metric_name)
            ),
        }

    # READINESS-5, 2026-09-10. Which way does this metric have to move to be
    # better? This function used to answer "down, always", projecting an alert
    # for every value BELOW the proposed threshold. That is right for the ratio
    # family (disparate_impact, the four-fifths rule) and exactly backwards for
    # every violation magnitude, which is most fairness metrics: a demographic
    # parity DIFFERENCE of 0.03 is excellent, and it was projected as an alert
    # against any threshold above 0.03. Measured on five real monitor windows
    # of demographic_parity at 0.02 to 0.08, the shipped code projected 5 of 5
    # alerting at thresholds 0.70, 0.80 and 0.90 alike, where the metric's own
    # rule projects 0, and the projection did not move with the threshold at
    # all. This is a tool for CHOOSING a threshold, so an inverted projection
    # sends the operator the wrong way.
    #
    # _metric_direction is the library's single answer to that question and it
    # fails closed by contract, so an unrecognised metric is refused here
    # rather than simulated in a guessed direction. That module's own docstring
    # records what guessing costs: the cali-BRATIO-n substring match let a
    # large miscalibration read as a PASS in production for weeks.
    direction = metric_direction(metric_name)
    if direction is MetricDirection.UNKNOWN:
        return {
            "metric": metric_name,
            "new_threshold": new_threshold,
            "current_alerts": None,
            "projected_alerts": None,
            "groups_impacted": None,
            "change_abs": None,
            "change_pct": None,
            "records_simulated": 0,
            "records_not_simulated": int(len(df)),
            "not_simulated_groups": not_simulated_groups,
            "direction": MetricDirection.UNKNOWN.value,
            "not_simulated_reason": (
                "COULD NOT SIMULATE: it is not known whether {0!r} passes ABOVE "
                "or BELOW its threshold, so projecting which records would alert "
                "would mean guessing the direction. Guessing it wrong inverts "
                "every count in this result.".format(metric_name)
            ),
        }

    current_alerts = int(simulatable["alert"].sum()) if "alert" in simulatable.columns else 0

    # Simulate: a violation magnitude alerts ABOVE its threshold, a parity
    # ratio BELOW it.
    if direction is MetricDirection.HIGHER_IS_BETTER:
        breaches = simulatable["value"] < new_threshold
    else:
        breaches = simulatable["value"] > new_threshold
    projected_alerts = int(breaches.sum())

    # Which groups would be impacted?
    group_means = simulatable.groupby("group")["value"].mean()
    if direction is MetricDirection.HIGHER_IS_BETTER:
        groups_impacted = group_means[group_means < new_threshold].index.tolist()
    else:
        groups_impacted = group_means[group_means > new_threshold].index.tolist()
    groups_impacted = [g for g in groups_impacted if g != "overall"]

    change_abs = projected_alerts - current_alerts
    # A percent change is undefined when the baseline is 0 alerts. Reporting
    # 0.0% there hid real increases (0 -> N alerts showed as "0.0% change"),
    # so the percent is None in that case and change_abs carries the truth.
    if current_alerts > 0:
        change_pct = round((change_abs / current_alerts) * 100, 1)
    elif projected_alerts == 0:
        change_pct = 0.0
    else:
        change_pct = None

    return {
        "metric": metric_name,
        "new_threshold": new_threshold,
        "current_alerts": current_alerts,
        "projected_alerts": projected_alerts,
        "groups_impacted": groups_impacted,
        "change_abs": change_abs,
        "change_pct": change_pct,
        "direction": direction.value,
        "not_simulated_reason": "",
        "records_simulated": int(len(simulatable)),
        "records_not_simulated": int(len(left_out)),
        "not_simulated_groups": not_simulated_groups,
    }


class InteractiveDashboard:
    """Maximum-interactivity fairness dashboard.

    Supports two output modes:

    1. **Dash server**: ``run(port=8050)`` launches a full interactive app.
    2. **Standalone HTML**: ``to_standalone_html()`` produces a self-contained
       HTML file with JavaScript-driven interactivity.

    Parameters
    ----------
    store : MetricsStore
        The unified data layer backing all views.
    dashboard : FairnessDashboard, optional
        If provided, re-uses its Plotly figures; otherwise creates a new one.
    config : InteractiveConfig, optional
        Interactive-specific settings.

    Examples
    --------
    >>> idash = InteractiveDashboard(store)
    >>> idash.run(port=8050)       # Dash server
    >>> html = idash.to_standalone_html()
    >>> idash.save_html("dashboard.html")

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

    Ledger row: interactive_dashboard. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        store: MetricsStore,
        dashboard: Optional[FairnessDashboard] = None,
        config: Optional[InteractiveConfig] = None,
    ) -> None:
        self.store = store
        self.config = config or InteractiveConfig()
        self.dashboard = dashboard or FairnessDashboard(
            store,
            DashboardConfig(),
        )
        self._app: Optional["dash.Dash"] = None

    # Privacy filter

    def _filter_small_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove groups with size below min_group_display_size."""
        if "group_size" not in df.columns:
            return df
        mask = (df["group_size"] >= self.config.min_group_display_size) | (df["group"] == "overall")
        return df[mask].copy()

    # What-If

    def simulate_threshold_change(
        self,
        metric_name: str,
        new_threshold: float,
        time_window: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper around the module-level simulation function.

        It forwards the module function's answer unchanged, including its
        refusals, so the BGL5 guard on ``new_threshold`` (a nan bound used to
        produce "both alerts removed, a 100 percent reduction" here too, verified
        identical through this wrapper) applies at this surface as well.
        """
        tw = time_window or timedelta(days=30)
        return simulate_threshold_change(self.store, metric_name, new_threshold, tw)

    # Dash Application

    def create_dash_app(self) -> "dash.Dash":
        """Build and return a Dash application instance.

        The app layout includes:
        - **Sidebar**: metric dropdown, date picker, group checklist,
          what-if threshold slider.
        - **Main area**: health gauge, trend chart, disparity bars,
          alert timeline.

        Returns
        -------
        dash.Dash
            The configured Dash application (not yet started).
        """
        _check_dash()

        app = dash.Dash(
            __name__,
            title="vfairness: Fairness Dashboard",
            suppress_callback_exceptions=True,
        )

        # Available data
        metric_names = self.store.get_metric_names() or ["demographic_parity"]
        groups = self.store.get_groups() or ["overall"]
        default_metric = self.config.default_metric or (
            metric_names[0] if metric_names else "demographic_parity"
        )

        app.layout = html.Div(
            style={"fontFamily": "'Inter','Segoe UI',system-ui,sans-serif", "display": "flex"},
            children=[
                # Sidebar
                html.Div(
                    style={
                        "width": "280px",
                        "padding": "20px",
                        "backgroundColor": "#f8fafc",
                        "borderRight": "1px solid #e2e8f0",
                        "minHeight": "100vh",
                    },
                    children=[
                        html.H3("Filters", style={"color": "#1e293b", "marginTop": 0}),
                        html.Label("Metric", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                        dcc.Dropdown(
                            id="metric-selector",
                            options=[{"label": m, "value": m} for m in metric_names],
                            value=default_metric,
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        html.Label(
                            "Time Window", style={"fontWeight": "600", "fontSize": "0.85rem"}
                        ),
                        dcc.Dropdown(
                            id="time-window",
                            options=[
                                {"label": "Last 24 hours", "value": "24h"},
                                {"label": "Last 7 days", "value": "7d"},
                                {"label": "Last 30 days", "value": "30d"},
                                {"label": "Last 90 days", "value": "90d"},
                            ],
                            value="30d",
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        html.Label("Groups", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                        dcc.Checklist(
                            id="group-selector",
                            options=[{"label": g, "value": g} for g in groups],
                            value=groups[:5],  # show first 5 by default
                            style={"marginBottom": "16px", "fontSize": "0.85rem"},
                        ),
                        html.Hr(),
                        html.H3("What-If Analysis", style={"color": "#1e293b"}),
                        html.Label("Threshold", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="threshold-slider",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            value=0.8,
                            marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11, 2)},
                        ),
                        html.Div(
                            id="whatif-result",
                            style={
                                "marginTop": "12px",
                                "padding": "12px",
                                "backgroundColor": "#fff",
                                "borderRadius": "8px",
                                "border": "1px solid #e2e8f0",
                                "fontSize": "0.8rem",
                            },
                        ),
                    ],
                ),
                # Main content
                html.Div(
                    style={"flex": 1, "padding": "20px"},
                    children=[
                        html.H1(
                            "Fairness Interactive Dashboard",
                            style={"color": "#1e293b", "fontSize": "1.4rem"},
                        ),
                        html.Div(
                            id="health-banner",
                            style={
                                "textAlign": "center",
                                "padding": "16px",
                                "marginBottom": "20px",
                            },
                        ),
                        # Charts grid
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "16px",
                            },
                            children=[
                                dcc.Graph(id="health-gauge", style={"height": "350px"}),
                                dcc.Graph(id="trend-chart", style={"height": "350px"}),
                                dcc.Graph(id="disparity-chart", style={"height": "350px"}),
                                dcc.Graph(id="alert-chart", style={"height": "350px"}),
                            ],
                        ),
                        # Drift timeline (full width)
                        dcc.Graph(id="drift-chart", style={"marginTop": "16px"}),
                    ],
                ),
            ],
        )

        # Callbacks
        store_ref = self.store
        dash_ref = self.dashboard

        @app.callback(
            [
                Output("health-banner", "children"),
                Output("health-gauge", "figure"),
                Output("trend-chart", "figure"),
                Output("disparity-chart", "figure"),
                Output("alert-chart", "figure"),
                Output("drift-chart", "figure"),
            ],
            [
                Input("metric-selector", "value"),
                Input("time-window", "value"),
                Input("group-selector", "value"),
            ],
        )
        def update_charts(metric, time_window_val, selected_groups):
            tw_map = {
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90),
            }
            tw = tw_map.get(time_window_val, timedelta(days=30))
            # G08: window bound in the store's tz base, see MetricsStore.window_now
            end = store_ref.window_now()
            start = end - tw
            tw_tuple = (start, end)

            # Health
            hs = store_ref.compute_health_score(time_window=tw)
            status_color = {"green": "#059669", "yellow": "#f59e0b", "red": "#dc2626"}.get(
                hs.status, "#64748b"
            )
            # C-04, consumer side. ``compute_health_score`` answers score=None /
            # status="not_assessed" when nothing in the window could be assessed,
            # and this badge formatted the score with a float spec, so the
            # callback raised TypeError instead of rendering anything.
            # ``_narrate_health`` in reports.py already says it in the reader's
            # words for every tier; the same third state is said here.
            # BGL grade-1 G08, 2026-09-30: `is_measured` rather than `is None`,
            # the same change as the standalone badge below. `nan is None` is
            # False and `f"{nan:.0f}"` is "nan", so a score that is not a
            # measurement rendered "nan/100" in whichever status colour the
            # HealthScore happened to carry.
            score_text = (
                "not assessed"
                if not is_measured(hs.score) or hs.status == "not_assessed"
                else f"{hs.score:.0f}/100"
            )
            banner = html.Div(
                [
                    html.Span(
                        score_text,
                        style={
                            "display": "inline-block",
                            "padding": "6px 18px",
                            "borderRadius": "20px",
                            "fontWeight": "700",
                            "fontSize": "1.3rem",
                            "color": "#fff",
                            "backgroundColor": status_color,
                        },
                    ),
                    html.P(hs.explanation, style={"color": "#64748b", "marginTop": "8px"}),
                ]
            )

            # Figures
            try:
                gauge_fig = dash_ref.create_health_score_gauge(hs)
            except Exception:
                gauge_fig = go.Figure()

            try:
                trend_fig = dash_ref.create_trend_analysis(
                    metric,
                    time_window=tw_tuple,
                    groups=selected_groups,
                )
            except Exception:
                trend_fig = go.Figure()

            try:
                disp_fig = dash_ref.create_disparity_comparison(metric, time_window=tw_tuple)
            except Exception:
                disp_fig = go.Figure()

            try:
                alert_fig = dash_ref.create_alert_summary(time_window=tw_tuple)
            except Exception:
                alert_fig = go.Figure()

            try:
                drift_fig = dash_ref.create_drift_timeline(time_window=tw_tuple)
            except Exception:
                drift_fig = go.Figure()

            return banner, gauge_fig, trend_fig, disp_fig, alert_fig, drift_fig

        @app.callback(
            Output("whatif-result", "children"),
            [
                Input("threshold-slider", "value"),
                Input("metric-selector", "value"),
            ],
        )
        def update_whatif(threshold, metric):
            result = simulate_threshold_change(store_ref, metric, threshold)
            skipped = int(result.get("records_not_simulated") or 0)
            skipped_groups = result.get("not_simulated_groups") or []
            skipped_note = ""
            if skipped:
                skipped_note = f"Not simulated: {skipped} record(s) no alert rule graded"
                if skipped_groups:
                    skipped_note += f" ({', '.join(skipped_groups)})"
            delta = result["change_abs"]
            if delta is None:
                # COULD NOT CHECK, rendered as such. Nothing in the window was
                # graded against a threshold, so there is no baseline. This
                # branch also used to be unreachable-by-crash: the f-string
                # below formatted `change_pct` with `:+.1f` and compared it to
                # 0, and the producer already returns None for it whenever the
                # baseline is 0 alerts, so the callback raised TypeError
                # ("unsupported format string passed to NoneType.__format__")
                # instead of rendering anything.
                #
                # BGL5, 2026-09-27: the producer's OWN reason is rendered when it
                # has one. This sentence was hardcoded, so the refusal added for
                # a proposed threshold that is not a number would have been shown
                # to the operator as "no record in this window was compared to a
                # threshold", which is a statement about the data and false of
                # that case. A correct refusal the consumer relabels is still a
                # wrong answer on the screen.
                detail = str(result.get("not_simulated_reason") or "").strip() or (
                    "no record in this window was compared to a threshold, so there "
                    "is no baseline to simulate against."
                )
                # The producer shouts its own state ("COULD NOT CHECK: ...");
                # drop that prefix so the rendered line names the state once.
                for shouted in ("COULD NOT CHECK:", "COULD NOT SIMULATE:"):
                    if detail.upper().startswith(shouted):
                        detail = detail[len(shouted) :].strip()
                        break
                return html.Div(
                    [
                        html.P(
                            f"Could not check: {detail}",
                            style={"color": "#b45309", "fontWeight": "600"},
                        ),
                        html.P(skipped_note),
                    ]
                )
            pct = result["change_pct"]
            pct_text = "n/a: no current alerts" if pct is None else f"{pct:+.1f}%"
            children = [
                html.P(f"Current alerts: {result['current_alerts']}"),
                html.P(f"Projected alerts: {result['projected_alerts']}"),
                html.P(
                    f"Change: {delta:+d} alert(s) ({pct_text})",
                    style={"color": "#dc2626" if delta > 0 else "#059669"},
                ),
                html.P(f"Groups impacted: {', '.join(result['groups_impacted']) or 'none'}"),
            ]
            if skipped_note:
                children.append(html.P(skipped_note, style={"color": "#b45309"}))
            return html.Div(children)

        self._app = app
        return app

    def run(self, port: int = 8050, host: str = "127.0.0.1", **kwargs) -> None:
        """Launch the Dash server.

        Parameters
        ----------
        port : int
            Port to serve on (default 8050).
        host : str
            Host address (default localhost).
        """
        if self._app is None:
            self._app = self.create_dash_app()
        self._app.run(
            host=host,
            port=port,
            debug=self.config.dash_debug,
            **kwargs,
        )

    # Standalone HTML

    def to_standalone_html(
        self,
        include_whatif: bool = True,
    ) -> str:
        """Generate a self-contained interactive HTML file.

        Uses embedded Plotly.js and JavaScript to switch between metrics,
        time windows, and views without a server.

        Parameters
        ----------
        include_whatif : bool
            Include the what-if threshold panel (default True).

        Returns
        -------
        str
            Complete HTML document.
        """
        _check_plotly()

        # Embed the full plotly.js bundle so the "self-contained" export
        # actually renders offline. A CDN <script src> tag would leave the
        # saved file blank without network access; the configured CDN URL
        # is kept only as a last-resort fallback if the local bundle
        # cannot be loaded.
        try:
            from plotly.offline import get_plotlyjs

            plotly_js_tag = f"<script>{get_plotlyjs()}</script>"
        except Exception:
            plotly_js_tag = f'<script src="{self.config.plotly_cdn}"></script>'

        metric_names = self.store.get_metric_names() or ["demographic_parity"]
        groups = self.store.get_groups() or []
        hs = self.store.compute_health_score()

        # EVERY FIGURE THAT FAILED TO BUILD, AND WHY. BGL grade-1 G08,
        # 2026-09-30. All six builders below were wrapped in
        # `except Exception: "{}"`, and the page's own `renderPlot` returns at
        # `if (!figData || !figData.data) return;`, so a builder that CRASHED
        # left an empty white panel: byte-identical to a panel with nothing to
        # plot, and sitting beside a health badge that still read "87/100".
        # Measured before this change, with create_trend_analysis and
        # create_health_score_gauge raising RuntimeError('trend builder
        # exploded') on a store holding four clean determined records: the page
        # came back at 4,884,285 characters, `var gaugeData = {};`, and the
        # words "exploded", "could not render" and "failed" appeared nowhere in
        # it. ReportGenerator._render_html already had exactly this fix, pinned
        # by test_a_chart_that_crashed_is_not_rendered_as_a_report_without_charts
        # ("COULD NOT RENDER ... not because there was nothing to plot"); this
        # was the second rendering of the same page and it had neither the fix
        # nor the pin. The panel is still left blank (there is no figure to put
        # there) and the reason is now ON the page.
        figure_failures: List[Tuple[str, str]] = []

        def _figure_json(label: str, build) -> str:
            try:
                return build().to_json()
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                figure_failures.append((label, f"{type(exc).__name__}: {str(exc)[:200]}"))
                return "{}"

        # Pre-render figures for each metric
        plot_data: Dict[str, Dict[str, str]] = {}
        for metric in metric_names:
            plot_data[metric] = {
                "trend": _figure_json(
                    f"Trend: {metric}",
                    lambda m=metric: self.dashboard.create_trend_analysis(m),
                ),
                "disparity": _figure_json(
                    f"Group disparity: {metric}",
                    lambda m=metric: self.dashboard.create_disparity_comparison(m),
                ),
            }

        # Health gauge (metric-independent)
        gauge_json = _figure_json(
            "Fairness Health Score gauge",
            lambda: self.dashboard.create_health_score_gauge(hs),
        )

        # Alert summary
        alert_json = _figure_json("Alert summary", self.dashboard.create_alert_summary)

        # Drift timeline
        drift_json = _figure_json("Drift timeline", self.dashboard.create_drift_timeline)

        # Pre-compute what-if data for default metric
        whatif_data = {}
        if include_whatif:
            default_metric = metric_names[0]
            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                result = simulate_threshold_change(self.store, default_metric, t)
                whatif_data[str(t)] = result

        status_color = {"green": "#059669", "yellow": "#f59e0b", "red": "#dc2626"}.get(
            hs.status, "#64748b"
        )
        # C-04, consumer side. Same third state as the Dash banner: an
        # unassessed window has score None, and the badge below formatted it
        # with `:.0f`, so `to_standalone_html` raised TypeError on exactly the
        # store the what-if panel is most likely to be pointed at. An absent
        # score is not a low score and not a high one.
        #
        # BGL grade-1 G08, 2026-09-30: `is_measured` rather than `is None`.
        # `None` is one of the doors a missing score comes through and NaN is
        # another; measured before this change on a HealthScore carrying
        # score=nan with status='green', the badge rendered "nan/100" in the
        # pass colour, because `nan is None` is False and `f"{nan:.0f}"` is
        # "nan". The library's canonical rule answers the same question for
        # every door at once, and a real finite score is untouched by it.
        score_badge = (
            "not assessed"
            if not is_measured(hs.score) or hs.status == "not_assessed"
            else f"{hs.score:.0f}/100"
        )

        # Build HTML
        metric_options = "".join(
            f'<option value="{m}" {"selected" if i == 0 else ""}>{m}</option>'
            for i, m in enumerate(metric_names)
        )

        # BGL grade-1 G08. The panels that are blank BECAUSE A BUILDER FAILED,
        # named on the page, in the wording ReportGenerator._render_html already
        # uses for the same failure on the document surface.
        failures_section = ""
        if figure_failures:
            items = "".join(
                f"<li><strong>{label}</strong>: {reason}</li>" for label, reason in figure_failures
            )
            failures_section = (
                '<div style="margin-bottom:20px;padding:12px 16px;background:#fffbeb;'
                "border:1px solid #fcd34d;border-radius:10px;color:#92400e;"
                'font-size:0.85rem;">'
                "<p><strong>COULD NOT RENDER</strong> "
                f"{len(figure_failures)} figure(s) on this page. Those panels are blank "
                "because their chart failed to build, not because there was nothing to "
                "plot, and this page carries no measurement for them:</p>"
                f'<ul style="margin:8px 0 0 20px;">{items}</ul>'
                "</div>"
            )

        # The store's OWN metric count, not the length of the selector list.
        # `metric_names` falls back to ["demographic_parity"] so the dropdown is
        # never empty, and that placeholder was then counted here: measured on an
        # empty store, the sidebar read "Data: 0 records | Metrics: 1 | Groups: 0"
        # for a store holding no metric at all. BGL grade-1 G08.
        n_metrics_held = len(self.store.get_metric_names() or [])
        metrics_line = (
            f"{n_metrics_held}"
            if n_metrics_held
            else "0 (the selector shows a placeholder name; this store holds none)"
        )

        whatif_section = ""
        if include_whatif:
            whatif_section = """
            <div style="margin-top:24px;padding:16px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;">
              <h3 style="margin-top:0;">What-If Threshold Analysis</h3>
              <label style="font-weight:600;font-size:0.85rem;">Threshold:
                <span id="threshold-display">0.80</span>
              </label>
              <input type="range" id="threshold-slider" min="0.5" max="0.9" step="0.1" value="0.8"
                     style="width:100%;margin:8px 0;" oninput="updateWhatIf(this.value)">
              <div id="whatif-result" style="margin-top:8px;font-size:0.85rem;color:#334155;"></div>
            </div>
            """

        html_content = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>vfairness: Interactive Fairness Dashboard</title>
{plotly_js_tag}
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter','Segoe UI',system-ui,sans-serif; background: #fafafa; color: #1e293b; }}
  .wrapper {{ display: flex; min-height: 100vh; }}
  .sidebar {{ width: 280px; padding: 20px; background: #f8fafc; border-right: 1px solid #e2e8f0; }}
  .main {{ flex: 1; padding: 24px; overflow-y: auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 16px; }}
  h3 {{ font-size: 1rem; color: #1e293b; margin-bottom: 12px; }}
  label {{ display: block; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; color: #334155; }}
  select {{ width: 100%; padding: 8px; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 16px; font-size: 0.85rem; }}
  .health-badge {{
    display: inline-block; padding: 8px 24px; border-radius: 24px;
    font-weight: 700; font-size: 1.5rem; color: #fff; background: {status_color};
  }}
  .health-text {{ color: #64748b; margin-top: 8px; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }}
  .chart-box {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .view-tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .view-tab {{
    padding: 6px 16px; border-radius: 20px; border: 1px solid #e2e8f0;
    background: #fff; cursor: pointer; font-size: 0.85rem; font-weight: 500;
  }}
  .view-tab.active {{ background: #1e293b; color: #fff; border-color: #1e293b; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="sidebar">
    <h3 style="margin-top:0;">Filters</h3>
    <label>Metric</label>
    <select id="metric-select" onchange="switchMetric(this.value)">
      {metric_options}
    </select>

    <label>View</label>
    <div class="view-tabs">
      <span class="view-tab active" onclick="switchView('operational',this)">Operational</span>
      <span class="view-tab" onclick="switchView('executive',this)">Executive</span>
      <span class="view-tab" onclick="switchView('technical',this)">Technical</span>
    </div>

    {whatif_section}

    <div style="margin-top:24px;padding:12px;background:#fff;border-radius:8px;border:1px solid #e2e8f0;">
      <p style="font-size:0.75rem;color:#94a3b8;">
        Data: {len(self.store)} records |
        Metrics: {metrics_line} |
        Groups: {len(groups)}
      </p>
    </div>
  </div>

  <div class="main">
    <h1>Fairness Interactive Dashboard</h1>

    {failures_section}

    <div style="text-align:center;margin-bottom:20px;">
      <span class="health-badge">{score_badge}</span>
      <p class="health-text">{hs.explanation}</p>
    </div>

    <div class="grid">
      <div class="chart-box" id="gauge-container"></div>
      <div class="chart-box" id="trend-container"></div>
      <div class="chart-box" id="disparity-container"></div>
      <div class="chart-box" id="alert-container"></div>
      <div class="chart-box full-width" id="drift-container"></div>
    </div>
  </div>
</div>

<script>
// Pre-rendered data
var plotData = {json.dumps(plot_data, default=str)};
var gaugeData = {gauge_json};
var alertData = {alert_json};
var driftData = {drift_json};
var whatifData = {json.dumps(whatif_data, default=str)};

var currentMetric = "{metric_names[0]}";

function renderPlot(containerId, figData) {{
  if (!figData || !figData.data) return;
  var config = {{responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d']}};
  Plotly.react(containerId, figData.data, figData.layout || {{}}, config);
}}

function switchMetric(metric) {{
  currentMetric = metric;
  var mdata = plotData[metric] || {{}};

  // Trend
  try {{
    var trendFig = JSON.parse(mdata.trend || '{{}}');
    renderPlot('trend-container', trendFig);
  }} catch(e) {{}}

  // Disparity
  try {{
    var dispFig = JSON.parse(mdata.disparity || '{{}}');
    renderPlot('disparity-container', dispFig);
  }} catch(e) {{}}
}}

function switchView(view, el) {{
  // Update tab styling
  document.querySelectorAll('.view-tab').forEach(function(t) {{ t.classList.remove('active'); }});
  if (el) el.classList.add('active');

  // Show/hide chart containers based on view
  var gauge = document.getElementById('gauge-container');
  var trend = document.getElementById('trend-container');
  var disp = document.getElementById('disparity-container');
  var alert = document.getElementById('alert-container');
  var drift = document.getElementById('drift-container');

  if (view === 'executive') {{
    gauge.style.display = 'block';
    trend.style.display = 'none';
    disp.style.display = 'none';
    alert.style.display = 'none';
    drift.style.display = 'none';
    gauge.style.gridColumn = '1 / -1';
  }} else if (view === 'technical') {{
    gauge.style.display = 'block';
    trend.style.display = 'block';
    disp.style.display = 'block';
    alert.style.display = 'block';
    drift.style.display = 'block';
    gauge.style.gridColumn = '';
  }} else {{
    // operational
    gauge.style.display = 'block';
    trend.style.display = 'block';
    disp.style.display = 'block';
    alert.style.display = 'block';
    drift.style.display = 'block';
    gauge.style.gridColumn = '';
  }}

  // Trigger resize
  window.dispatchEvent(new Event('resize'));
}}

function updateWhatIf(value) {{
  var display = document.getElementById('threshold-display');
  var result = document.getElementById('whatif-result');
  if (display) display.textContent = parseFloat(value).toFixed(2);

  // Normalise the slider value to the pre-computed 0.1 grid keys so a
  // browser float artefact (e.g. "0.7000000000000001") still hits data.
  var key = String(Math.round(parseFloat(value) * 10) / 10);
  var data = whatifData[key];
  if (data && result && data.change_abs === null) {{
    // COULD NOT CHECK: no record in the window carried an alert determination
    // and a real value, so there is no baseline. `null - null` is 0 in JS, so
    // the old delta fallback below rendered this as "Change: 0 alert(s)", a
    // measurement, and then threw on data.groups_impacted.length (null).
    result.innerHTML =
      '<p style="color:#b45309;font-weight:600;">Could not check: no record in this window ' +
      'was compared to a threshold, so there is no baseline to simulate against.</p>' +
      '<p>Not simulated: ' + data.records_not_simulated + ' record(s)' +
      ((data.not_simulated_groups && data.not_simulated_groups.length > 0)
        ? ' (' + data.not_simulated_groups.join(', ') + ')' : '') + '</p>';
    return;
  }}
  if (data && result) {{
    var delta = (typeof data.change_abs === 'number')
      ? data.change_abs
      : (data.projected_alerts - data.current_alerts);
    var changeColor = delta > 0 ? '#dc2626' : '#059669';
    // change_pct is null when there are 0 current alerts (percent of zero
    // is undefined); the absolute delta is always shown.
    var pctText = (data.change_pct === null || data.change_pct === undefined)
      ? 'n/a: no current alerts'
      : ((data.change_pct > 0 ? '+' : '') + data.change_pct + '%');
    result.innerHTML =
      '<p>Current alerts: ' + data.current_alerts + '</p>' +
      '<p>Projected alerts: ' + data.projected_alerts + '</p>' +
      '<p style="color:' + changeColor + '">Change: ' + (delta > 0 ? '+' : '') + delta + ' alert(s) (' + pctText + ')</p>' +
      '<p>Groups impacted: ' + (data.groups_impacted.length > 0 ? data.groups_impacted.join(', ') : 'none') + '</p>' +
      ((data.records_not_simulated > 0)
        ? '<p style="color:#b45309;">Not simulated: ' + data.records_not_simulated +
          ' record(s) no alert rule graded</p>'
        : '');
  }} else if (result) {{
    result.innerHTML = '<p style="color:#94a3b8;">No pre-computed data for this threshold.</p>';
  }}
}}

// Initial render
document.addEventListener('DOMContentLoaded', function() {{
  renderPlot('gauge-container', gaugeData);
  renderPlot('alert-container', alertData);
  renderPlot('drift-container', driftData);
  switchMetric(currentMetric);
  updateWhatIf('0.8');
}});
</script>
</body>
</html>
"""
        return html_content

    def save_html(self, path: str, include_whatif: bool = True) -> str:
        """Render the standalone HTML and write it to a file.

        Parameters
        ----------
        path : str
            File path (e.g. ``"dashboard.html"``).
        include_whatif : bool
            Include the what-if panel.

        Returns
        -------
        str
            The absolute path of the written file.
        """
        content = self.to_standalone_html(include_whatif=include_whatif)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # Summary

    def get_capabilities(self) -> Dict[str, bool]:
        """Report which features are available in the current environment."""
        return {
            "dash_server": _DASH_AVAILABLE,
            "plotly_charts": _PLOTLY_AVAILABLE,
            "standalone_html": _PLOTLY_AVAILABLE,
            # Reachable through EITHER delivery surface, and through neither when
            # both are missing. This was a hardcoded True, which reported a
            # capability as available in an environment where nothing could deliver
            # it: with no dash and no plotly, a caller reading
            # {"dash_server": False, "standalone_html": False,
            #  "whatif_simulation": True} has been told yes about a feature with no
            # remaining route to it. The threshold sweep itself is pure Python, which
            # is what made True look true, but a capability nobody can reach is not
            # a capability.
            "whatif_simulation": _DASH_AVAILABLE or _PLOTLY_AVAILABLE,
        }

    def __repr__(self) -> str:
        caps = self.get_capabilities()
        features = [k for k, v in caps.items() if v]
        return f"InteractiveDashboard(features={features})"
