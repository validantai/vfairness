"""
Unit 3: Performance Dashboards: ReportGenerator
=================================================

Automated, event-driven, multi-tier fairness report generation.

The ``ReportGenerator`` produces stakeholder-specific reports in HTML,
Markdown, and JSON formats.  It integrates:

- **MetricsStore** for data
- **FairnessDashboard** for Plotly chart embedding (optional)
- **Jinja2** for template-based rendering
- **Natural Language Generation (NLG)** for accessible explanations

Reports can be triggered manually or by events such as threshold breaches,
drift detection, or scheduled intervals.

Three audience tiers follow the **progressive disclosure** principle:

- **Executive** (Tier 1): Single health score, traffic-light, 1-paragraph
  summary.
- **Operational** (Tier 2): Key charts, insight bullets, filtered tables.
- **Technical** (Tier 3): Full statistics, model versions, confusion matrices.

References
----------
Module 4, Part 4, Unit 3: Performance Dashboards and Reporting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..._triage import is_flag, is_measured, unmeasurable_reason
from ...branding import branding_enabled
from .store import HealthScore, MetricsStore, PrivacyLevel

# Enums & Config


class ReportTier(Enum):
    """Audience tier for report generation."""

    EXECUTIVE = 1
    OPERATIONAL = 2
    TECHNICAL = 3


class OutputFormat(Enum):
    """Report output format."""

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class ReportConfig:
    """Configuration for :class:`ReportGenerator`."""

    default_tier: ReportTier = ReportTier.OPERATIONAL
    default_format: OutputFormat = OutputFormat.HTML
    include_charts: bool = True
    include_recommendations: bool = True
    time_window: timedelta = timedelta(days=30)
    custom_title: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class GeneratedReport:
    """Result of report generation.

    Attributes
    ----------
    content : str
        The rendered report body (HTML, Markdown, or JSON).
    format : OutputFormat
        Which format was used.
    tier : ReportTier
        Audience tier of the report.
    timestamp : datetime
        When the report was generated.
    title : str
        Report title.
    health_score : HealthScore, optional
        Snapshot of the composite health score.
    sections : list[dict]
        Report sections with title and content.
    recommendations : list[str]
        Actionable recommendations.
    metadata : dict
        Additional metadata (model version, time range, etc.).
    """

    content: str
    format: OutputFormat
    tier: ReportTier
    timestamp: datetime
    title: str = "Fairness Report"
    health_score: Optional[HealthScore] = None
    sections: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write the report content to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "format": self.format.value,
            "tier": self.tier.name,
            "timestamp": str(self.timestamp),
            "health_score": self.health_score.to_dict() if self.health_score else None,
            "sections": self.sections,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


# NLG Helpers


def _format_score(score: "Optional[float]", places: int = 0) -> str:
    """Render a health score, or say plainly that there is not one.

    C-04. Three render paths formatted `hs.score` with a float spec and one
    indexed a three-key colour map, so an unassessed score raised TypeError and
    KeyError rather than printing anything. "n/a" is deliberately not a number:
    a reader scanning for a figure must not find one where none was computed.
    """
    return "not assessed" if score is None else f"{score:.{places}f}"


def _format_stat(value: Any, places: int = 4) -> str:
    """Render a statistic, or say plainly that there is not one.

    H-11, reader side. A NaN sentinel handed to a float spec prints ``nan``,
    and a key defaulted to ``0`` prints a confident ``0.0000`` for a number
    nobody computed; a drift score of 0.0000 is the most reassuring figure this
    report can show. Both read as measurements. Three states: a real number is
    printed, an absent one says so, and neither is a zero.

    A FLAG is not a statistic. BGL5 A-operations-2, 2026-09-27: measured before
    this line, ``_format_stat(True)`` returned ``'1.0000'``, so a breach report
    handed ``True`` printed ``Current Value: 1.0000`` for a value nobody
    measured; it now returns ``'not measured'``. ``is_flag`` rather than the full
    :func:`vfairness._triage.is_measured` because this renderer deliberately
    accepts a numeric STRING (its rows arrive from JSON and CSV, where "0.5" is a
    measurement that was serialised), which is the split that function's own
    docstring describes.
    """
    if value is None or is_flag(value):
        return "not measured"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "not measured"
    if pd.isna(f):
        return "not measured"
    return f"{f:.{places}f}"


def _is_comparable(value: Any) -> bool:
    """Whether *value* is a number a threshold comparison can be made against.

    Every comparison against NaN is False in BOTH directions, so a NaN value
    does not breach a threshold and does not satisfy one either: it is
    could-not-check. Not ``isinstance(v, (int, float))``, which rejects
    ``np.float32`` and ``decimal.Decimal`` and would discard a real measurement:
    :func:`vfairness._triage.is_measured` is the library's one answer to this
    question and it accepts both.

    BGL5 A-operations-2, 2026-09-27. This was ``math.isfinite(float(value))``,
    and ``float(True)`` is ``1.0``, so a FLAG passed. Measured before the change,
    ``generate_threshold_breach_report('demographic_parity_difference', True,
    0.10)`` published "Current Value: 1.0000 | Threshold: 0.1000 | Breach
    Magnitude: 900.0% beyond threshold", the recommendation "The
    demographic_parity_difference metric has breached its 0.10 threshold." and
    JSON ``"current_value": true, "breach_pct": 900.0, "comparison_made": true``.
    After: "Breach Magnitude: COULD NOT CHECK ...", the COULD NOT CHECK
    recommendation and ``"comparison_made": false``. ``is_measured`` rejects a
    bool for exactly this reason and says so in its own rule 2.

    It also settles a disagreement between this predicate and the arithmetic it
    guards. ``float("0.45")`` is finite, so the numeric STRING "0.45" passed, and
    the subtraction that follows uses the RAW value: measured the same day,
    ``generate_threshold_breach_report('dp', '0.45', 0.10)`` raised
    ``TypeError: unsupported operand type(s) for -: 'str' and 'float'`` out of the
    public method. A value this guard accepts must be one the comparison can
    make, and now is.
    """
    return is_measured(value)


def _affected_groups_word(groups: Any) -> str:
    """Name the groups an alert names, or say plainly that it names none.

    BGL3 operations-1, 2026-09-27. ``d.get('affected_groups', [])`` printed
    "Affected Groups: []" on a CRITICAL alert page. An empty list is the ONLY
    thing the producer can supply when nothing identified the groups:
    ``FairnessAlertPrioritizer.create_alert`` builds it as
    ``affected_groups or drift_event.get("affected_groups", [])``, so "nobody
    recorded who is affected" and "we checked and nobody is affected" arrive here
    as the same value, and "[]" is the calmer of the two readings. Measured
    before this change on a dict-shaped alert with no affected_groups key:
    "Affected Groups: []" under a "Severity: CRITICAL" line.

    No group is ever invented: a recorded list prints exactly as recorded.
    """
    if groups is None:
        return "not recorded"
    try:
        items = [str(g) for g in groups if g is not None and str(g) != ""]
    except TypeError:
        return str(groups)
    if not items:
        return (
            "not recorded (this alert carries no affected group, which is not a "
            "finding that none is affected)"
        )
    return ", ".join(items)


def _severity_word(severity: Any) -> str:
    """Name an alert's severity, or say plainly that nothing scored it.

    BGL5 A-operations-2, 2026-09-27. One fix reached the section BODY and not the
    TITLE. ``d.get('severity', 'ALERT')`` stood at three sites in
    :meth:`ReportGenerator.generate_alert_report` (the HTML title, the markdown
    title and ``GeneratedReport.title``), and ``.get`` with a default does NOT
    fire for a key that is PRESENT holding ``None``. Measured on
    ``{'severity': None, 'metric_name': ..., 'message': ...}``:
    ``GeneratedReport.title == 'Alert Report: None'``, the markdown rendered
    ``# Alert Report: None`` and the HTML ``<title>Alert Report: None</title>``,
    while the body of the SAME object correctly read
    ``Severity: UNSCORED (not determined)``. After: every one of those four
    surfaces reads ``UNSCORED (not determined)``, from this one function, so the
    next fix cannot reach three sites out of four again.

    "None" is a reader's word for no, not for "nobody scored it", which is the
    argument :func:`_drift_verdict_word` below already makes about the same
    mistake on the same page. An ABSENT key is the same could-not-check as a null
    one and gets the same word: the previous ``'ALERT'`` default asserted that
    there IS an alert-worthy severity here, which is exactly what was not
    determined.

    A recorded severity is never altered: it prints as recorded.
    """
    if severity is None or str(severity).strip() == "":
        return "UNSCORED (not determined)"
    return str(severity)


def _drift_verdict_word(detected: Any) -> str:
    """Say what the drift verdict is, including when there is not one.

    H-11, reader side. ``MultiscaleDriftResult.drift_detected`` is Optional:
    ``None`` means not one scale could be computed, so no drift test ran. It
    used to be interpolated straight into the summary, which printed
    "Drift Detected: None" beside an overall score of ``nan``. That is a
    reader's word for "no", not for "nobody looked".
    """
    if detected is None:
        return "COULD NOT CHECK (no scale was analysed, so no drift test ran)"
    return str(bool(detected))


def _assessed_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Split metric rows into the ones a threshold test graded, and name the rest.

    R-2, reader side, 2026-09-09. ``alert`` answers ONE question: did an alert
    FIRE. ``bool(None)`` is False, so a record nobody ever compared to a
    threshold arrived here indistinguishable from a clean one, and the
    executive narrative called it "within acceptable limits". The store already
    ships the column that separates them (``alert_determined``, see
    ``MetricsStore.get_metrics``) and already excludes such records from the
    health score's compliance mean, so this reads the same evidence the
    producer read rather than inventing a second rule.

    Two ways a row carries no measurement, matching ``store._compliance_rows``:
    no alert determination at all, or a value that could not be computed (H-10,
    NaN). A value withheld by k-anonymity is NOT one of them: it is NaN in this
    frame for privacy, but the threshold comparison behind it ran on the true
    value, so the row stays graded and ``privacy_level`` says why the number is
    absent. Calling every suppressed row could-not-check would raise a false
    alarm on every privacy-protected report.

    Frames built before ``alert_determined`` existed carry no determinations to
    read; every row there is taken as graded, exactly as before.

    Returns ``(graded_rows, unassessed_names)``: the rows that may be narrated
    as passing or failing, and the sorted metric names of the rest.
    """
    if "alert_determined" not in df.columns:
        return df, []
    graded = df["alert_determined"].astype(bool)
    if "value" in df.columns:
        if "privacy_level" in df.columns:
            # Both levels that WITHHOLD a value, not just the one. READINESS-5,
            # 2026-09-10 added UNKNOWN_SIZE for a record whose group size was
            # never recorded, and it withholds the number for the same reason
            # this branch exists: the threshold comparison behind the row ran
            # on the true value, so the row is still graded and privacy_level
            # says why the number is absent. Matching only SUPPRESSED here
            # would have turned every one of those rows into a false
            # could-not-check the moment the new level started being used.
            withheld_levels = {
                PrivacyLevel.SUPPRESSED.value,
                PrivacyLevel.UNKNOWN_SIZE.value,
            }
            suppressed = df["privacy_level"].astype(str).isin(withheld_levels)
        else:
            suppressed = pd.Series(False, index=df.index)
        # The library's canonical predicate, not a sixth private copy of it.
        # BGL5 A-operations-2, 2026-09-27: this was
        # ``isinstance(v, (int, float)) and not pd.isna(v)``, which is False for
        # ``decimal.Decimal``. It is also False for ``np.float32`` as a value, but
        # NOT at this call site, and the difference was found by sabotage rather
        # than by reading: measured 2026-09-27, ``pd.Series([np.float32(0.91)]*3)
        # .apply(lambda v: type(v).__name__)`` answers ['float', 'float', 'float'],
        # because Series.apply boxes a float dtype's elements as PYTHON floats,
        # while ``s.iloc[0]`` is a np.float32. So the numpy half was already
        # passing here by accident of pandas, and an OBJECT column (which is what
        # a Decimal column is) hands the predicate the value itself. Measured on a store
        # holding 3 records with ``value=Decimal("0.91")`` and ``alert=True``
        # (a real, finite, BREACHING measurement, the shape a DB NUMERIC column
        # and ``json.loads(parse_float=Decimal)`` deliver): ``_assessed_rows``
        # returned 0 graded rows and named the metric as
        # ``['demographic_parity_difference']``, so the report said "3 of 3
        # measurement(s) were never compared to a threshold, or could not be
        # measured at all" about three rows that were both. After: 3 graded rows
        # and no name. That is the READINESS-6 defect running BACKWARDS, a
        # measurement reported as a could-not-check, which reads as caution while
        # discarding the evidence; ``_triage.is_measured`` names Decimal
        # explicitly for this reason. It also drops ``inf``, which no threshold
        # comparison can grade (``inf > threshold`` is True for EVERY threshold),
        # into the ungraded half where it belongs.
        measurable = df["value"].apply(is_measured)
        graded = graded & (measurable | suppressed)
    names: List[str] = []
    if "metric_name" in df.columns:
        names = sorted(set(df.loc[~graded, "metric_name"].astype(str).tolist()))
    return df[graded], names


def _narrate_health(hs: HealthScore, tier: ReportTier) -> str:
    """Generate health-score narrative adapted to the audience tier."""
    # C-04. `status` gained a third value, "not_assessed", when the store holds
    # no metric records for the window. Every tier below formats `hs.score` with
    # a float spec and the executive tier indexes a three-key colour map, so an
    # unassessed score raised KeyError here rather than printing anything. Said
    # once, in the reader's own words, for every tier: an absent score is not a
    # low score and certainly not a high one.
    if hs.score is None or hs.status == "not_assessed":
        # READINESS-5, 2026-09-10. This sentence used to hardcode ONE reason,
        # "no metric records were found in the evaluation window", while
        # compute_health_score withholds the score for at least three
        # different ones: an empty window, a window whose every record carries
        # no threshold comparison, and a window whose every drift test the
        # detector refused. The score object carries the true reason in
        # `explanation` and this narrator was discarding it.
        #
        # Measured that day: a store holding TWENTY records, none of them ever
        # compared to a threshold, was reported to the reader as "no metric
        # records were found". Those are different problems with different
        # fixes (a pipeline that is not feeding the store, versus one that is
        # feeding it values with no verdicts), and the report named the wrong
        # one. The reason is now the one the score itself gives.
        reason = (hs.explanation or "").strip() or (
            f"No reason was recorded ({hs.n_metrics} metric record(s), "
            f"{hs.n_not_assessable} not assessable)."
        )
        # The store's own explanations already end with the "not a score of
        # 100" clause, so adding it unconditionally said it twice. It is added
        # only when the reason does not already carry it, because a withheld
        # score MUST NOT reach a reader without it.
        disclaimer = (
            ""
            if "not a score" in reason.lower()
            else (
                " This is not a score of 0 and not a score of 100. Nothing here "
                "certifies anything about the system."
            )
        )
        return (
            f"Fairness health could NOT be assessed. {reason} "
            f"({hs.n_alerts} alert(s) recorded in the window.){disclaimer}"
        )

    if tier == ReportTier.EXECUTIVE:
        status_emoji = {"green": "Green", "yellow": "Yellow", "red": "Red"}[hs.status]
        return (
            f"Overall fairness status: {status_emoji} ({hs.score:.0f}/100). "
            f"Trend: {hs.trend}. "
            f"{hs.n_alerts} alert{'s' if hs.n_alerts != 1 else ''} in the evaluation window."
        )
    elif tier == ReportTier.OPERATIONAL:
        return (
            f"The fairness health score is {hs.score:.1f}/100 ({hs.status}), "
            f"composed of metric compliance "
            f"({_format_score(hs.components.get('metric_compliance'), 0)}), "
            f"alert frequency ({_format_score(hs.components.get('alert_frequency'), 0)}), and "
            f"drift stability ({_format_score(hs.components.get('drift_stability'), 0)}). "
            f"The trend is {hs.trend} (slope: {hs.trend_slope:+.4f}/day)."
        )
    else:
        return (
            f"Composite Health Score: {_format_score(hs.score, 2)}/100.00 | "
            f"Status: {hs.status.upper()} | Trend: {hs.trend} "
            f"(slope={hs.trend_slope:+.6f}/day)\n"
            # `.get(name, 0)` is the wrong guard twice over: the default does
            # not fire when the key is PRESENT holding None, and when the key
            # is genuinely absent a 0 is the most alarming figure this line
            # can print for something nobody measured. _format_score says
            # "not assessed" for both.
            f"Components: "
            f"metric_compliance={_format_score(hs.components.get('metric_compliance'), 2)}, "
            f"alert_frequency={_format_score(hs.components.get('alert_frequency'), 2)}, "
            f"drift_stability={_format_score(hs.components.get('drift_stability'), 2)}\n"
            f"Evaluation window: {hs.n_metrics} metric records, {hs.n_alerts} alerts"
        )


def _narrate_metrics(df: pd.DataFrame, tier: ReportTier) -> str:
    """Generate a narrative about the current metrics state."""
    if df.empty:
        return "No metric data available for the selected time window."

    # R-2, reader side. Every count below used to be taken over EVERY row, and
    # the only question asked of a row was whether an alert fired. A record
    # nobody compared to a threshold has no fired alert, so it landed in the
    # clean majority and the executive tier reported "All N tracked fairness
    # metrics are within acceptable limits" over metrics that were never
    # assessed, on the same page whose header said the health score could NOT
    # be assessed. The graded rows are separated first, and every rate below is
    # taken over THEM; what was not graded is named rather than averaged in.
    graded, unassessed_names = _assessed_rows(df)
    n_total = len(df)
    n_graded = len(graded)
    n_unassessed = n_total - n_graded
    n_alerts = int(graded["alert"].astype(bool).sum()) if "alert" in graded.columns else 0
    alert_pct = (n_alerts / n_graded * 100) if n_graded > 0 else None

    metrics = df["metric_name"].nunique()
    graded_metrics = graded["metric_name"].nunique() if n_graded > 0 else 0
    groups = df[df["group"] != "overall"]["group"].nunique()

    # Said in the reader's own words, in the wording ``adapters_reporting``
    # already prints on the SVG surface for the same records, so the two
    # documents about one run cannot disagree.
    unassessed_clause = ""
    if n_unassessed > 0:
        unassessed_clause = (
            f" COULD NOT CHECK: {n_unassessed} of {n_total} measurement(s) were never "
            f"compared to a threshold, or could not be measured at all "
            f"({', '.join(unassessed_names)}). They are neither passing nor failing "
            f"here, and this report does not certify them."
        )

    if tier == ReportTier.EXECUTIVE:
        if n_graded == 0:
            return (
                f"COULD NOT CHECK: not one of the {n_total} measurement(s) in this "
                f"window was compared to a threshold, or could be measured at all "
                f"({', '.join(unassessed_names)}). Nothing here is an all-clear."
            )
        if alert_pct == 0:
            if n_unassessed > 0:
                return (
                    f"All {graded_metrics} assessed fairness metrics are within "
                    f"acceptable limits.{unassessed_clause}"
                )
            return f"All {metrics} tracked fairness metrics are within acceptable limits."
        return (
            f"{n_alerts} of {n_graded} measurements ({alert_pct:.0f}%) "
            f"triggered alerts.{unassessed_clause}"
        )
    elif tier == ReportTier.OPERATIONAL:
        lines = [f"Tracking {metrics} fairness metrics across {groups} demographic groups."]
        if n_alerts > 0:
            # "Most frequently alerting" must be chosen by how often each
            # metric alerts, not by the highest mean value (a single
            # high-valued alert used to beat a metric alerting many times),
            # and the reported count is that metric's, not the global total.
            alert_counts = graded[graded["alert"]].groupby("metric_name").size()
            if not alert_counts.empty:
                worst_name = alert_counts.idxmax()
                worst_count = int(alert_counts.max())
                scope = "assessed" if n_unassessed > 0 else "all"
                lines.append(
                    f"Most frequently alerting metric: {worst_name} "
                    f"({worst_count} of {n_alerts} alerts; "
                    f"{alert_pct:.1f}% of {scope} measurements alerted)."
                )
        if unassessed_clause:
            lines.append(unassessed_clause.strip())
        return " ".join(lines)
    else:
        pct = "n/a" if alert_pct is None else f"{alert_pct:.2f}%"
        line = (
            f"Total records: {n_total} | Unique metrics: {metrics} | "
            f"Groups: {groups} | Alerts: {n_alerts} ({pct})"
        )
        if n_unassessed > 0:
            line += (
                f" | COULD NOT CHECK: {n_unassessed} record(s) never compared to a "
                f"threshold or not measurable ({', '.join(unassessed_names)}); the "
                f"alert rate is over the {n_graded} assessed record(s)"
            )
        return line


def _generate_recommendations(
    hs: HealthScore,
    df: pd.DataFrame,
    alerts: List[Dict],
    drift_not_assessed: bool = False,
) -> List[str]:
    """Generate actionable recommendations based on current state."""
    recs: List[str] = []

    if hs.status == "red":
        recs.append(
            "URGENT: Fairness health score is critical. Convene the fairness "
            "review board and consider pausing model updates until resolved."
        )
    elif hs.status == "yellow":
        recs.append(
            "Fairness health score is marginal. Schedule a review of the "
            "most-impacted metrics within the next sprint cycle."
        )
    elif hs.status == "not_assessed" or hs.score is None:
        # C-04 / R-2, reader side. `status` has THREE values and the two tested
        # above are the two bad ones, so "not_assessed" fell through every
        # branch to the empty-recommendations all-clear at the foot of this
        # function: an executive summary reading "All systems nominal" for a
        # window in which nothing was graded, printed under a header that
        # correctly said the score could NOT be assessed. A withheld score is
        # not a passing score, and it must produce an action, not silence.
        recs.append(
            "COULD NOT CHECK: no fairness health score was computed for this "
            "window, so this report neither passes nor fails the system and "
            "certifies nothing. Restore the metric records, or the threshold "
            "comparisons behind them, before reading it as an all-clear."
        )

    if hs.trend == "degrading":
        recs.append(
            "Fairness metrics show a degrading trend. Investigate whether "
            "recent data distribution shifts or model updates are responsible."
        )

    critical_alerts = [a for a in alerts if a.get("severity") == "CRITICAL"]
    if critical_alerts:
        recs.append(
            f"{len(critical_alerts)} CRITICAL alert(s) are unresolved. "
            "These require immediate investigation and acknowledgment."
        )

    # Check for alerting metrics
    if not df.empty and "alert" in df.columns:
        alerting_metrics = df[df["alert"]]["metric_name"].unique()
        if len(alerting_metrics) > 0:
            recs.append(
                f"Metrics currently in alert: {', '.join(alerting_metrics)}. "
                "Consider threshold optimization or fairness-aware retraining."
            )

    # R-2, reader side. Read off the SAME frame the Metrics Overview narrative
    # is read off, so the two sections of one report name the same records. An
    # ungraded record produces an action here, which is also what keeps the
    # all-clear below from firing over a window that was only partly assessed.
    if not df.empty:
        _graded, unassessed_names = _assessed_rows(df)
        n_unassessed = len(df) - len(_graded)
        if n_unassessed > 0:
            recs.append(
                f"COULD NOT CHECK: {n_unassessed} of {len(df)} metric record(s) were "
                f"never compared to a threshold, or could not be measured at all "
                f"({', '.join(unassessed_names)}). Give them a threshold, or a usable "
                "value, before this report can speak for them."
            )

    if not recs:
        recs.append(
            "All systems nominal. Continue routine monitoring and maintain "
            "the current alert threshold configuration."
        )

    # BGL S2b, 2026-09-17. The all-clear above was the LAST line of a page whose
    # own Coverage section says "**Drift stability: NOT ASSESSED.**": a reader
    # who skips to the recommendations is told everything was checked and
    # everything was fine. The caveat is APPENDED rather than substituted, and
    # it comes after the fall-through rather than before it, because a genuinely
    # clean window must still be able to say so in its own words: suppressing
    # the all-clear here hedged a fully clean two-metric report and broke the
    # over-correction control in tests/test_readiness_reports.py that exists to
    # catch exactly that. What is wrong is an UNQUALIFIED all-clear, not the
    # all-clear. Scope, not silence.
    if drift_not_assessed:
        recs.append(
            "Scope: drift stability was not assessed in this window because no "
            "drift test ran, so it is excluded from the sub-scores above and "
            "anything said here covers the metric and alert components only. "
            "Run a drift check to close that gap."
        )

    return recs


# HTML / Markdown Templates

_STATUS_COLORS = {"green": "#059669", "yellow": "#f59e0b", "red": "#dc2626"}

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: 'Inter','Segoe UI',system-ui,sans-serif; margin:0; padding:24px; background:#fafafa; color:#1e293b; }}
  .container {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; border-bottom:2px solid #e2e8f0; padding-bottom:8px; }}
  h2 {{ font-size:1.15rem; color:#334155; margin-top:24px; }}
  .health-badge {{ display:inline-block; padding:6px 18px; border-radius:20px;
                   font-weight:700; font-size:1.3rem; color:#fff; background:{status_color}; }}
  .kpi-row {{ display:flex; gap:16px; margin:16px 0; }}
  .kpi {{ flex:1; background:#fff; border:1px solid #e2e8f0; border-radius:10px;
          padding:16px; text-align:center; }}
  .kpi-value {{ font-size:1.4rem; font-weight:700; }}
  .kpi-label {{ font-size:0.8rem; color:#64748b; margin-top:4px; }}
  .narrative {{ background:#fff; border-left:4px solid {status_color};
               padding:12px 16px; margin:16px 0; border-radius:0 8px 8px 0; }}
  .section {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px;
              padding:16px; margin:16px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ text-align:left; border-bottom:2px solid #e2e8f0; padding:8px; color:#64748b; }}
  td {{ border-bottom:1px solid #f1f5f9; padding:8px; }}
  .rec {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
          padding:12px; margin:8px 0; }}
  .rec-critical {{ background:#fef2f2; border-color:#fecaca; }}
  .footer {{ text-align:center; color:#94a3b8; font-size:0.75rem; margin-top:32px; }}
  .chart-container {{ margin:16px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p style="color:#64748b;font-size:0.85rem;">
    Generated: {timestamp} | Tier: {tier} | Window: {window}
    {model_version_html}
  </p>

  <div style="text-align:center;margin:20px 0;">
    <span class="health-badge">{health_score}/100</span>
    <p style="margin-top:8px;color:#64748b;">Fairness Health Score: {status}</p>
  </div>

  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-value">{metric_compliance}</div>
      <div class="kpi-label">Metric Compliance</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">{alert_frequency}</div>
      <div class="kpi-label">Alert Score</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">{drift_stability}</div>
      <div class="kpi-label">Drift Stability</div>
    </div>
  </div>

  <div class="narrative">{narrative}</div>

  {sections_html}

  {recommendations_html}

  {charts_html}

  <div class="footer">
    {brand_footer}{timestamp}
  </div>
</div>
</body>
</html>
"""


class ReportGenerator:
    """Automated multi-tier, multi-format fairness report generator.

    Parameters
    ----------
    store : MetricsStore
        Unified data layer.
    dashboard : FairnessDashboard, optional
        For embedding Plotly charts in HTML reports.
    config : ReportConfig, optional
        Generation defaults.

    Examples
    --------
    >>> gen = ReportGenerator(store)
    >>> report = gen.generate_executive_report()
    >>> report.save("executive_report.html")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: report_generation. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        store: MetricsStore,
        dashboard=None,
        config: Optional[ReportConfig] = None,
    ) -> None:
        self.store = store
        self.dashboard = dashboard
        self.config = config or ReportConfig()

    # Coverage

    def _health_with_drift_coverage(
        self, tw: timedelta
    ) -> "Tuple[HealthScore, Optional[Dict[str, str]]]":
        """The health score, with an UNMEASURED drift component withheld.

        ``_drift_stability`` (store.py) returns a hardcoded ``100.0`` for an
        EMPTY drift table, on the reasoning that "a system that ran no drift
        test at all has no drift rows either". That reasoning does not survive
        this surface. The component grades a MEAN over drift rows; the mean of
        zero rows is not 100, it is undefined, and 100.0 is then weighted 20
        percent into a colour-banded composite an executive reads. Its sibling
        ``alert_score = max(0.0, 100.0 - n_alerts * 10)`` is a function of a
        COUNT and genuinely IS defined at zero, so the two are not analogous.

        Measured 2026-09-16 through the public API, on a store with 12 clean
        determined metric records and ZERO drift records (the default state of
        every MetricsStore): "## Health Score: 100/100 (GREEN)" over the row
        "| Drift Stability | 100.0 |", and that row was the ONLY line in the
        whole report containing the word drift. Nothing said no drift test ran.

        The substitution itself lives in ``store.py``, which this group does
        not own, so it is corrected HERE, at the surface a person reads: the
        key is dropped (absent is this module's existing token for a component
        that was not assessed, per the C-04 comments in both renderers, and
        ``_format_score`` prints "not assessed" for it), and a section saying
        so in words is added to the report. The composite ``score`` itself
        still carries the unmeasured 20 percent; the note says that too.

        Returns ``(health_score, section_or_None)``.
        """
        hs = self.store.compute_health_score(time_window=tw)
        if "drift_stability" not in hs.components:
            # ABSENT IS THE UNMEASURED STATE, inverted 2026-09-27. See the long note in
            # dashboard._drift_coverage_note: this used to return None because absence
            # meant "withheld one level up", which held while the producer always
            # published a hardcoded 100.0 here. The producer now drops the key when
            # drift was not measured AND renormalises the composite over the two
            # components that were, so the early return silenced this section in the one
            # case it is for.
            #
            # AND THERE IS NO SECOND READING TO COMPARE ANY MORE. The comparison built
            # below contrasts a measured-only figure against a composite that included
            # the unmeasured drift weight. Once the producer renormalises, hs.score IS
            # the measured-only figure, so that sentence would tell a reader the score
            # above "includes the unmeasured drift weight" when it does not. One
            # reading, stated once.
            return hs, {
                "title": "Coverage",
                "content": (
                    "**Drift stability: NOT ASSESSED.** This report withholds the "
                    "drift stability sub-score because no drift test was run in this "
                    "window, so there is nothing to grade. The sub-score is absent from "
                    "the composite above rather than defaulted. It is not a "
                    "drift stability of 100, and it is not one of 0. The health score "
                    "above is computed over the two components that WERE measured, "
                    "metric compliance and alert frequency, renormalised to their own "
                    "weights."
                ),
            }

        # G08: window bound in the store's tz base, see MetricsStore.window_now
        now = self.store.window_now()
        try:
            drift_df = self.store.get_drift_history(start_time=now - tw, end_time=now)
            n_drift = 0 if drift_df is None else int(len(drift_df))
            reason = (
                "no drift test was run in this window, so there is nothing to grade"
                if n_drift == 0
                else ""
            )
        except Exception as exc:  # the coverage question itself failed
            n_drift, reason = 0, f"the drift history could not be read ({exc})"

        if not reason:
            return hs, None

        components = {k: v for k, v in hs.components.items() if k != "drift_stability"}

        # BGL S2b, 2026-09-17. Saying "the drift weight carries a number nobody
        # computed" is true but unquantified, and the number it qualifies is
        # printed two inches higher in 48-point type beside a colour. Measured
        # on a store with every metric breaching, no ingested alerts and zero
        # drift rows: the page published "## Health Score: 50/100 (YELLOW)" from
        # 0.50*0.0 + 0.30*100.0 + 0.20*100.0, while the two components that WERE
        # measured renormalise to 37.5, which is RED. The unmeasured component
        # did not merely blur the score, it moved the band. So the note now
        # carries the measured-only figure and its band explicitly.
        #
        # The composite in store.py is NOT changed here: it is asserted by name
        # and by value in several other test lanes, and re-weighting it is a
        # store.py change that has to land with those. What this surface can do
        # without guessing is show the reader both numbers and which one rests
        # on evidence.
        measured_only: Optional[float] = None
        measured_band = ""
        weights = {"metric_compliance": 0.50, "alert_frequency": 0.30}
        # NOT `isinstance(v, (int, float))`: compute_health_score hands back
        # np.float64 for metric_compliance, which that test REJECTS, so the
        # canonical-looking predicate would have discarded a real measurement
        # and reported a could-not-check. float() plus a finite test asks the
        # question that actually matters.
        usable: Dict[str, float] = {}
        for k, v in components.items():
            if k not in weights:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                usable[k] = fv
        total_w = sum(weights[k] for k in usable)
        if usable and total_w > 0:
            measured_only = round(sum(weights[k] * usable[k] for k in usable) / total_w, 1)
            measured_band = (
                "GREEN" if measured_only >= 80 else "YELLOW" if measured_only >= 50 else "RED"
            )
        if measured_only is None:
            comparison = (
                "No component of the composite was measured on evidence in this "
                "window, so the score above rests on defaults alone."
            )
        elif hs.score is None:
            comparison = (
                f"Over the components that WERE measured, and only those, the score "
                f"is {measured_only} ({measured_band})."
            )
        else:
            same = measured_band == (
                "GREEN" if hs.score >= 80 else "YELLOW" if hs.score >= 50 else "RED"
            )
            comparison = (
                f"Over the components that WERE measured, and only those, the score is "
                f"{measured_only} ({measured_band}); the {hs.score} above includes the "
                f"unmeasured drift weight. "
                + (
                    "Both readings fall in the same band."
                    if same
                    else "THE TWO READINGS FALL IN DIFFERENT BANDS, so the colour above "
                    "is a consequence of the component nobody measured."
                )
            )

        note = (
            "**Drift stability: NOT ASSESSED.** This report withholds the drift "
            f"stability sub-score because {reason}. It is not a drift stability of "
            "100, and it is not one of 0: no drift measurement exists for this "
            "window. Treat the composite health score above as resting on the two "
            "components that were measured; the drift weight inside it carries a "
            "number nobody computed. " + comparison
        )
        return (
            replace(hs, components=components),
            {"title": "Coverage", "content": note},
        )

    # Core Generation

    def generate(
        self,
        tier: Optional[ReportTier] = None,
        output_format: Optional[OutputFormat] = None,
        time_window: Optional[timedelta] = None,
    ) -> GeneratedReport:
        """Generate a report.  Dispatches to tier-specific methods."""
        tier = tier or self.config.default_tier
        fmt = output_format or self.config.default_format
        dispatch = {
            ReportTier.EXECUTIVE: self.generate_executive_report,
            ReportTier.OPERATIONAL: self.generate_operational_report,
            ReportTier.TECHNICAL: self.generate_technical_report,
        }
        return dispatch[tier](output_format=fmt, time_window=time_window)

    def generate_executive_report(
        self,
        output_format: OutputFormat = OutputFormat.HTML,
        time_window: Optional[timedelta] = None,
    ) -> GeneratedReport:
        """Tier 1: Health score + traffic light + 1-paragraph summary."""
        tw = time_window or self.config.time_window
        hs, drift_section = self._health_with_drift_coverage(tw)
        # G08: window bound in the store's tz base, see MetricsStore.window_now
        now = self.store.window_now()
        start = now - tw

        df = self.store.get_metrics(start_time=start, end_time=now)
        alerts = self.store.get_alerts(start_time=start, end_time=now)
        narrative = _narrate_health(hs, ReportTier.EXECUTIVE)
        metrics_narrative = _narrate_metrics(df, ReportTier.EXECUTIVE)
        recs = _generate_recommendations(hs, df, alerts, drift_section is not None)

        sections = [
            {"title": "Summary", "content": narrative},
            {"title": "Metrics Overview", "content": metrics_narrative},
        ]
        if drift_section is not None:
            sections.append(drift_section)

        if output_format == OutputFormat.HTML:
            content = self._render_html(hs, sections, recs, ReportTier.EXECUTIVE, tw)
        elif output_format == OutputFormat.MARKDOWN:
            content = self._render_markdown(hs, sections, recs, ReportTier.EXECUTIVE, tw)
        else:
            content = self._render_json(hs, sections, recs, ReportTier.EXECUTIVE, tw)

        return GeneratedReport(
            content=content,
            format=output_format,
            tier=ReportTier.EXECUTIVE,
            timestamp=now,
            title=self.config.custom_title or "Executive Fairness Report",
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    def generate_operational_report(
        self,
        output_format: OutputFormat = OutputFormat.HTML,
        time_window: Optional[timedelta] = None,
    ) -> GeneratedReport:
        """Tier 2: Charts + insight bullets + metric tables."""
        tw = time_window or self.config.time_window
        hs, drift_section = self._health_with_drift_coverage(tw)
        # G08: window bound in the store's tz base, see MetricsStore.window_now
        now = self.store.window_now()
        start = now - tw

        df = self.store.get_metrics(start_time=start, end_time=now)
        alerts = self.store.get_alerts(start_time=start, end_time=now)

        narrative = _narrate_health(hs, ReportTier.OPERATIONAL)
        metrics_narrative = _narrate_metrics(df, ReportTier.OPERATIONAL)
        recs = _generate_recommendations(hs, df, alerts, drift_section is not None)

        # Build metric summary table
        metric_table = ""
        if not df.empty:
            aggs: Dict[str, Any] = {
                "mean": ("value", "mean"),
                "std": ("value", "std"),
                "n_alerts": ("alert", "sum"),
                "n_records": ("value", "count"),
            }
            # R-2, reader side. `n_alerts` 0 beside `n_records` 3 is read as
            # three clean measurements, and for a metric nobody ever compared
            # to a threshold the zero is arithmetic, not evidence. Where the
            # frame records which rows were graded, the ungraded count gets its
            # own column instead of being folded into that zero.
            if "alert_determined" in df.columns:
                aggs["n_not_assessed"] = (
                    "alert_determined",
                    lambda s: int((~s.astype(bool)).sum()),
                )
            summary = df.groupby("metric_name").agg(**aggs).round(4).reset_index()
            metric_table = summary.to_html(index=False, classes="metric-table")

        sections = [
            {"title": "Health Assessment", "content": narrative},
            {"title": "Metrics Overview", "content": metrics_narrative},
            {"title": "Metric Statistics", "content": metric_table},
        ]

        # Alert details
        if alerts:
            alert_lines = []
            for a in alerts[:10]:
                # `.get(key, 0)` printed "score 0.0" for an alert payload
                # that carries no priority score, and 0.0 is this list's
                # calmest number; it also raises TypeError when the key is
                # PRESENT holding None. Both are could-not-check. R-2.
                alert_lines.append(
                    f"[{a.get('severity', '?')}] {a.get('metric_name', '?')}: "
                    f"score {_format_stat(a.get('priority_score'), 1)}"
                )
            sections.append(
                {
                    "title": f"Recent Alerts ({len(alerts)} total)",
                    "content": "\n".join(alert_lines),
                }
            )

        if drift_section is not None:
            sections.append(drift_section)

        if output_format == OutputFormat.HTML:
            content = self._render_html(hs, sections, recs, ReportTier.OPERATIONAL, tw)
        elif output_format == OutputFormat.MARKDOWN:
            content = self._render_markdown(hs, sections, recs, ReportTier.OPERATIONAL, tw)
        else:
            content = self._render_json(hs, sections, recs, ReportTier.OPERATIONAL, tw)

        return GeneratedReport(
            content=content,
            format=output_format,
            tier=ReportTier.OPERATIONAL,
            timestamp=now,
            title=self.config.custom_title or "Operational Fairness Report",
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    def generate_technical_report(
        self,
        output_format: OutputFormat = OutputFormat.HTML,
        time_window: Optional[timedelta] = None,
    ) -> GeneratedReport:
        """Tier 3: Full statistics + model versions + drift decomposition."""
        tw = time_window or self.config.time_window
        hs, drift_section = self._health_with_drift_coverage(tw)
        # G08: window bound in the store's tz base, see MetricsStore.window_now
        now = self.store.window_now()
        start = now - tw

        df = self.store.get_metrics(start_time=start, end_time=now)
        alerts = self.store.get_alerts(start_time=start, end_time=now)
        drift_df = self.store.get_drift_history(start_time=start, end_time=now)

        narrative = _narrate_health(hs, ReportTier.TECHNICAL)
        metrics_narrative = _narrate_metrics(df, ReportTier.TECHNICAL)
        recs = _generate_recommendations(hs, df, alerts, drift_section is not None)

        sections = [
            {"title": "Health Score Decomposition", "content": narrative},
            {"title": "Metrics Summary", "content": metrics_narrative},
        ]

        # Full metric table
        if not df.empty:
            full_aggs: Dict[str, Any] = {
                "mean": ("value", "mean"),
                "std": ("value", "std"),
                "min": ("value", "min"),
                "max": ("value", "max"),
                "n_alerts": ("alert", "sum"),
                "n_records": ("value", "count"),
            }
            # Same reason as the Tier-2 table above (R-2, reader side): a zero
            # alert count over rows nobody graded is not a clean row.
            if "alert_determined" in df.columns:
                full_aggs["n_not_assessed"] = (
                    "alert_determined",
                    lambda s: int((~s.astype(bool)).sum()),
                )
            full_summary = (
                df.groupby(["metric_name", "group"]).agg(**full_aggs).round(4).reset_index()
            )
            sections.append(
                {
                    "title": "Per-Group Metric Breakdown",
                    "content": full_summary.to_html(index=False)
                    if output_format == OutputFormat.HTML
                    else full_summary.to_string(),
                }
            )

        # Drift history
        if not drift_df.empty:
            sections.append(
                {
                    "title": "Drift Detection History",
                    "content": drift_df.to_html(index=False)
                    if output_format == OutputFormat.HTML
                    else drift_df.to_string(),
                }
            )

        # Model versions
        if self.store._models:
            model_info = "\n".join(f"- {mid}: {meta}" for mid, meta in self.store._models.items())
            sections.append({"title": "Registered Models", "content": model_info})

        summary = self.store.get_summary()
        sections.append(
            {
                "title": "Store Statistics",
                "content": json.dumps(summary, indent=2, default=str),
            }
        )

        if drift_section is not None:
            sections.append(drift_section)

        if output_format == OutputFormat.HTML:
            content = self._render_html(hs, sections, recs, ReportTier.TECHNICAL, tw)
        elif output_format == OutputFormat.MARKDOWN:
            content = self._render_markdown(hs, sections, recs, ReportTier.TECHNICAL, tw)
        else:
            content = self._render_json(hs, sections, recs, ReportTier.TECHNICAL, tw)

        return GeneratedReport(
            content=content,
            format=output_format,
            tier=ReportTier.TECHNICAL,
            timestamp=now,
            title=self.config.custom_title or "Technical Fairness Report",
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    # Event-Driven Reports

    def generate_alert_report(
        self,
        alert,
        output_format: OutputFormat = OutputFormat.HTML,
    ) -> GeneratedReport:
        """Generate a report triggered by a specific alert event."""
        d = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        # BGL S2b, 2026-09-17. This called compute_health_score() DIRECTLY and
        # so skipped the drift-coverage correction the three tiered reports get.
        # Measured on a store with 12 determined metric records and zero drift
        # rows: this page rendered "| Drift Stability | 100.0 |" with no
        # coverage note, while generate_executive_report on the SAME store
        # rendered "not assessed". Same helper, same window.
        hs, drift_section = self._health_with_drift_coverage(timedelta(days=7))
        now = datetime.now()
        # ONE reading of the severity for all four surfaces this method writes:
        # the section body, the HTML <title>, the markdown H1 and
        # GeneratedReport.title. See :func:`_severity_word`: the body was fixed
        # and the three title sites, carrying the identical
        # ``d.get('severity', 'ALERT')`` idiom, were not, so a null severity was
        # rendered "Alert Report: None" on the most prominent surface on the page
        # while the body two lines down said "UNSCORED (not determined)".
        severity_word = _severity_word(d.get("severity"))
        alert_title = f"Alert Report: {severity_word}"

        sections = [
            {
                "title": "Alert Details",
                # A drift score of 0.0000 for a payload that carries none is
                # the single most reassuring figure on this page, and it was
                # printed by a `.get(key, 0)` default. Unmeasured says so. R-2.
                "content": (
                    f"Severity: {severity_word}\n"
                    f"Metric: {d.get('metric_name', '?')}\n"
                    f"Priority Score: {_format_stat(d.get('priority_score'), 2)}\n"
                    f"Affected Groups: {_affected_groups_word(d.get('affected_groups'))}\n"
                    f"Drift Score: {_format_stat(d.get('drift_score'))}\n"
                    f"Message: {d.get('message', '')}"
                ),
            },
            {"title": "Current Health", "content": hs.explanation},
        ]
        if drift_section is not None:
            sections.append(drift_section)
        recs = [
            "Investigate the root cause of this alert immediately.",
            "Check for recent data pipeline changes or model updates.",
            "Consider applying fairness-aware interventions if confirmed.",
        ]

        if output_format == OutputFormat.HTML:
            content = self._render_html(
                hs,
                sections,
                recs,
                ReportTier.OPERATIONAL,
                self.config.time_window,
                title=alert_title,
            )
        elif output_format == OutputFormat.MARKDOWN:
            content = self._render_markdown(
                hs,
                sections,
                recs,
                ReportTier.OPERATIONAL,
                self.config.time_window,
                title=alert_title,
            )
        else:
            content = json.dumps(
                {
                    "alert": d,
                    "health_score": hs.to_dict(),
                    "sections": sections,
                    "recommendations": recs,
                },
                indent=2,
                default=str,
            )

        return GeneratedReport(
            content=content,
            format=output_format,
            tier=ReportTier.OPERATIONAL,
            timestamp=now,
            title=alert_title,
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    def generate_drift_report(
        self,
        drift_result,
        output_format: OutputFormat = OutputFormat.HTML,
    ) -> GeneratedReport:
        """Generate a report triggered by a drift detection event."""
        # BGL S2b, 2026-09-17. Same skipped correction as generate_alert_report,
        # and on this page it reads worst of all: a report whose entire subject
        # is drift printed a drift stability of 100.0 that no drift row backs.
        hs, drift_section = self._health_with_drift_coverage(timedelta(days=7))
        now = datetime.now()

        sections = [
            {
                "title": "Drift Detection Summary",
                "content": (
                    f"Metric: {drift_result.metric}\n"
                    f"Overall Drift Score: {_format_stat(drift_result.overall_drift_score)}\n"
                    f"Drift Detected: {_drift_verdict_word(drift_result.drift_detected)}\n"
                    f"MMD Score: {getattr(drift_result, 'mmd_score', 'N/A')}"
                ),
            },
        ]

        if drift_result.worst_scale:
            ws = drift_result.worst_scale
            sections.append(
                {
                    "title": "Worst Scale",
                    "content": (
                        f"Scale: {ws.scale}\n"
                        f"KS Statistic: {ws.ks_statistic:.4f}\n"
                        f"p-value: {ws.p_value:.4f}\n"
                        f"Mean Shift: {ws.mean_shift:+.4f}"
                    ),
                }
            )

        if drift_section is not None:
            sections.append(drift_section)

        recs = []
        detected = drift_result.drift_detected
        if detected is None:
            # H-11, reader side. `if drift_result.drift_detected:` is a
            # two-state test and None is falsy, so a result for which NO scale
            # could be computed (no drift test ran at all, overall score NaN)
            # took the else branch and was reported as routine stability.
            # `drift_report_to_svg` renders COULD NOT CHECK for the same object;
            # this is that sentence, so the two surfaces about one result agree.
            recs.append(
                "COULD NOT CHECK: no temporal scale was analysed, so the overall "
                "drift score is the aggregate of nothing and no scale was compared "
                "to a reference. This is not a finding of stability. Collect a "
                "longer series and re-run the drift check."
            )
        elif detected:
            recs.append("Drift confirmed. Investigate data distribution changes.")
            recs.append("Consider retraining the model with recent data.")
        else:
            recs.append("No significant drift detected. Continue routine monitoring.")

        if output_format == OutputFormat.MARKDOWN:
            content = self._render_markdown(
                hs,
                sections,
                recs,
                ReportTier.TECHNICAL,
                self.config.time_window,
                title="Drift Detection Report",
            )
        elif output_format == OutputFormat.JSON:
            content = json.dumps(
                {
                    "drift_result": drift_result.to_dict()
                    if hasattr(drift_result, "to_dict")
                    else {},
                    "health_score": hs.to_dict(),
                    "sections": sections,
                    "recommendations": recs,
                },
                indent=2,
                default=str,
            )
        else:
            content = self._render_html(
                hs,
                sections,
                recs,
                ReportTier.TECHNICAL,
                self.config.time_window,
                title="Drift Detection Report",
            )

        return GeneratedReport(
            content=content,
            format=output_format,
            tier=ReportTier.TECHNICAL,
            timestamp=now,
            title="Drift Detection Report",
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    def generate_threshold_breach_report(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        output_format: OutputFormat = OutputFormat.HTML,
    ) -> GeneratedReport:
        """Generate a report when a metric crosses its threshold."""
        # BGL S2b, 2026-09-17. Same skipped correction as generate_alert_report.
        hs, drift_section = self._health_with_drift_coverage(timedelta(days=7))
        now = datetime.now()
        # BGL3 operations-1, 2026-09-27. The breach was asserted over ANY value
        # this method was handed, including one that does not exist. Measured
        # before this change on
        # ``generate_threshold_breach_report(metric, float("nan"), 0.10)``, which
        # is what a metric that could not be computed hands it: the page
        # published "Current Value: nan", "Breach Magnitude: nan% beyond
        # threshold", and the recommendation "The demographic_parity_difference
        # metric has breached its 0.10 threshold." Not one comparison was made,
        # because every comparison against NaN is False in both directions. This
        # is the repo's worst defect shape (a gate writing a verdict for a gap it
        # never compared) pointing at a breach instead of a pass.
        #
        # A threshold of 0.0 is NOT this case: it is finite, the comparison is
        # real, and only the PERCENTAGE beyond it is undefined, which the R-2
        # wording below already handles and which its test pins.
        comparable = _is_comparable(current_value) and _is_comparable(threshold)

        # Past this point the predicate above has established that both values are
        # real, finite numbers, so bind them as plain floats and do the arithmetic
        # on THOSE, exactly as operations/cicd/monitor.py binds
        # baseline_measured / current_measured after its own guard. The raw values
        # were being subtracted, and a value the predicate accepts is not
        # necessarily one Python can subtract from a float: measured 2026-09-27,
        # ``Decimal("0.45") - 0.10`` raised "TypeError: unsupported operand type(s)
        # for -: 'decimal.Decimal' and 'float'" out of this public method, which is
        # the identical crash the numeric string "0.45" produced before the
        # predicate was corrected. _triage names Decimal as a real measurement
        # arriving from a database NUMERIC column, so refusing it is not the
        # answer either: it is measured, and it now measures 350.0%.
        current_measured = float(current_value) if comparable else None
        threshold_measured = float(threshold) if comparable else None

        # A threshold of 0 has no "percent beyond" it, and the old `else 0`
        # printed "0.0% beyond threshold" for a breach of any size: the calmest
        # possible reading of a page whose entire subject is a breach. The
        # magnitude is withheld and said in words instead. R-2.
        breach_pct: Optional[float] = (
            abs(current_measured - threshold_measured) / threshold_measured * 100
            if current_measured is not None and threshold_measured
            else None
        )
        current_value_word = _format_stat(current_value)
        if not comparable:
            # WHY it could not be checked, not only that it could not. The three
            # non-measurements reach this method by different routes and an
            # operator fixes them differently: a NaN is a computation with nothing
            # to work with, a flag is a caller passing the wrong field, and a
            # string is a value that was never parsed. BGL5, 2026-09-27.
            reason = unmeasurable_reason(current_value) or unmeasurable_reason(threshold)
            # And the same reason on the VALUE line, so the page does not make two
            # statements about one value that disagree. Measured 2026-09-27 with
            # the numeric string "0.45", which _format_stat accepts on purpose
            # (its rows arrive from JSON and CSV): the page printed "Current
            # Value: 0.4500" directly above "COULD NOT CHECK: ... not a number
            # (str)". It now reads "Current Value: not measured (not a number
            # (str))". The NaN case still reads "Current Value: not measured",
            # which its own pin asserts.
            current_value_word = f"not measured ({reason})"
            breach_magnitude = (
                f"COULD NOT CHECK: the current value or the threshold is not a finite "
                f"number ({reason}), so no comparison was made and no breach was "
                f"established"
            )
        elif breach_pct is None:
            breach_magnitude = (
                "not measurable (the threshold is 0, so a percentage beyond it is undefined)"
            )
        else:
            breach_magnitude = f"{breach_pct:.1f}% beyond threshold"

        sections = [
            {
                "title": "Threshold Breach",
                "content": (
                    f"Metric: {metric_name}\n"
                    f"Current Value: {current_value_word}\n"
                    f"Threshold: {_format_stat(threshold)}\n"
                    f"Breach Magnitude: {breach_magnitude}"
                ),
            },
            {"title": "Current Health", "content": hs.explanation},
        ]
        if drift_section is not None:
            sections.append(drift_section)
        if comparable:
            recs = [
                f"The {metric_name} metric has breached its {threshold:.2f} threshold.",
                "Immediate investigation is recommended.",
                "Consider threshold optimization or post-processing fairness interventions.",
            ]
        else:
            recs = [
                f"COULD NOT CHECK: this report was handed a {metric_name} value of "
                f"{current_value_word} against a threshold of "
                f"{_format_stat(threshold)}, so no comparison was made. It does not "
                f"establish a breach and it does not clear the metric.",
                "Investigate why the metric could not be measured, then re-run this check.",
            ]

        fmt = output_format
        if fmt == OutputFormat.MARKDOWN:
            content = self._render_markdown(
                hs,
                sections,
                recs,
                ReportTier.OPERATIONAL,
                self.config.time_window,
                title="Threshold Breach Report",
            )
        elif fmt == OutputFormat.JSON:
            content = json.dumps(
                {
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "threshold": threshold,
                    "breach_pct": breach_pct,
                    # The machine-readable half of the same three states: a
                    # consumer must be able to tell "breached by this much" from
                    # "nothing was compared" without parsing prose.
                    "comparison_made": comparable,
                    "health_score": hs.to_dict(),
                    "recommendations": recs,
                },
                indent=2,
                default=str,
            )
        else:
            content = self._render_html(
                hs,
                sections,
                recs,
                ReportTier.OPERATIONAL,
                self.config.time_window,
                title="Threshold Breach Report",
            )

        return GeneratedReport(
            content=content,
            format=fmt,
            tier=ReportTier.OPERATIONAL,
            timestamp=now,
            title="Threshold Breach Report",
            health_score=hs,
            sections=sections,
            recommendations=recs,
        )

    # Rendering

    def _render_html(
        self,
        hs: HealthScore,
        sections: List[Dict],
        recs: List[str],
        tier: ReportTier,
        tw: timedelta,
        title: Optional[str] = None,
    ) -> str:
        title = title or self.config.custom_title or f"{tier.name.title()} Fairness Report"

        sections_html = ""
        for sec in sections:
            content = sec["content"].replace("\n", "<br>") if sec["content"] else ""
            sections_html += f'<div class="section"><h2>{sec["title"]}</h2><p>{content}</p></div>'

        recs_html = ""
        if recs:
            recs_html = "<h2>Recommendations</h2>"
            for r in recs:
                cls = "rec-critical" if "URGENT" in r or "CRITICAL" in r else ""
                recs_html += f'<div class="rec {cls}">{r}</div>'

        # Try embedding Plotly charts
        charts_html = ""
        if self.config.include_charts and self.dashboard is not None:
            try:
                if tier == ReportTier.EXECUTIVE:
                    fig = self.dashboard.create_executive_view()
                elif tier == ReportTier.OPERATIONAL:
                    fig = self.dashboard.create_operational_view()
                else:
                    fig = self.dashboard.create_technical_view()
                charts_html = (
                    '<div class="chart-container">'
                    + fig.to_html(include_plotlyjs="cdn", full_html=False)
                    + "</div>"
                )
            except Exception as exc:
                # BGL3 operations-1, 2026-09-27. `except Exception: charts_html
                # = ""` made a chart that CRASHED identical to a report that was
                # never given a dashboard: both render a page with no figures and
                # no explanation. The charts carry the traffic-light colours and
                # the disparity bars, so losing them silently loses evidence, and
                # this is not hypothetical: dashboard.create_operational_view
                # raised PlotlyKeyError on any grid holding an Indicator until
                # the exclude_empty_subplots fix (see the comment there).
                # Measured before this change, with a dashboard whose
                # create_executive_view raises: the HTML contained no figure and
                # no mention of a failure anywhere in 3637 characters.
                charts_html = (
                    '<div class="section"><h2>Charts</h2><p>COULD NOT RENDER: the figures '
                    f"for this report failed to build ({type(exc).__name__}: "
                    f"{str(exc)[:200]}). They are missing because of that failure, not "
                    "because there was nothing to plot.</p></div>"
                )

        mv_html = ""
        if self.config.model_version:
            mv_html = f" | Model: {self.config.model_version}"

        # The timestamp is report metadata, not branding, so it survives when
        # the validant.ai mark is switched off (VB-COM-2).
        brand_footer = "vfairness Fairness Report | validant.ai | " if branding_enabled() else ""

        return _HTML_TEMPLATE.format(
            title=title,
            brand_footer=brand_footer,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            tier=tier.name,
            window=f"{tw.days} days",
            model_version_html=mv_html,
            health_score=_format_score(hs.score),
            status=hs.status.upper(),
            status_color=_STATUS_COLORS.get(hs.status, "#64748b"),
            # C-04, reader side. A not-assessed HealthScore carries NO
            # components, and `.get(name, 0)` turned each absent sub-score into
            # a large "0" in a KPI card labelled Metric Compliance: a measured
            # total failure, printed for something nobody measured. Same
            # helper, same words, as the headline score above.
            metric_compliance=_format_score(hs.components.get("metric_compliance"), 0),
            alert_frequency=_format_score(hs.components.get("alert_frequency"), 0),
            drift_stability=_format_score(hs.components.get("drift_stability"), 0),
            narrative=_narrate_health(hs, tier),
            sections_html=sections_html,
            recommendations_html=recs_html,
            charts_html=charts_html,
        )

    def _render_markdown(
        self,
        hs: HealthScore,
        sections: List[Dict],
        recs: List[str],
        tier: ReportTier,
        tw: timedelta,
        title: Optional[str] = None,
    ) -> str:
        title = title or f"{tier.name.title()} Fairness Report"
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"**Tier:** {tier.name} | **Window:** {tw.days} days",
            "",
            f"## Health Score: {_format_score(hs.score)}"
            + ("" if hs.score is None else "/100")
            + f" ({hs.status.upper()})",
            "",
            _narrate_health(hs, tier),
            "",
            "| Component | Score |",
            "|-----------|-------|",
            # C-04, reader side: see the KPI cards in `_render_html`. An
            # absent sub-score printed as 0.0 in this table, under a header
            # that said the score could NOT be assessed.
            f"| Metric Compliance | {_format_score(hs.components.get('metric_compliance'), 1)} |",
            f"| Alert Frequency | {_format_score(hs.components.get('alert_frequency'), 1)} |",
            f"| Drift Stability | {_format_score(hs.components.get('drift_stability'), 1)} |",
            "",
        ]

        for sec in sections:
            lines.append(f"## {sec['title']}")
            lines.append("")
            lines.append(sec["content"] or "_No data._")
            lines.append("")

        if recs:
            lines.append("## Recommendations")
            lines.append("")
            for r in recs:
                lines.append(f"- {r}")
            lines.append("")

        if branding_enabled():
            lines.append("---")
            lines.append("*vfairness Fairness Report | validant.ai*")
        return "\n".join(lines)

    def _render_json(
        self,
        hs: HealthScore,
        sections: List[Dict],
        recs: List[str],
        tier: ReportTier,
        tw: timedelta,
    ) -> str:
        return json.dumps(
            {
                "title": f"{tier.name.title()} Fairness Report",
                "timestamp": datetime.now().isoformat(),
                "tier": tier.name,
                "time_window_days": tw.days,
                "health_score": hs.to_dict(),
                "sections": sections,
                "recommendations": recs,
            },
            indent=2,
            default=str,
        )
