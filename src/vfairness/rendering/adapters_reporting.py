"""
Adapter for the multi-tier, event-driven reporting dashboard SVG template.

Transforms ReportGenerator / MetricsStore output into a flat dictionary
suitable for the ``reporting_dashboard.svg`` template.

Templates:
    reporting_dashboard: Comprehensive multi-tier event-driven report
                          with health score, KPIs, metrics table, threshold
                          breach details, alert summary, NLG narrative,
                          and recommendations.
"""

from datetime import datetime
from typing import Optional

from ..evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    is_ratio_metric,
    metric_direction,
    vacuous_bound_reason,
)
from .engine import render_svg


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# Color palette
_STATUS_COLORS = {
    "green": ("#059669", "#ecfdf5", "GREEN"),
    "yellow": ("#f59e0b", "#fffbeb", "YELLOW"),
    "red": ("#dc2626", "#fef2f2", "RED"),
}

#: How severe each headline status is, so a state can only ever be escalated.
#: The dashboard's badge, its health-score colour and the accessible ``<desc>``
#: (built by ``rendering.explain._fr_reporting``, which maps GREEN to severity
#: INFO) are all read from ONE label, so whatever this resolves to is what every
#: reader gets, sighted or not.
_STATUS_SEVERITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def _fmt_ts(ts) -> str:
    """Normalise any timestamp to '%Y-%m-%d %H:%M'."""
    if ts is None:
        return _now()
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return _now()


#: Sentinel for "the report did not carry this key at all". Distinct from None,
#: which some producers write deliberately, and from 0, which is a measurement.
_MISSING = object()

#: Slate, the could-not-check tone, and its wash. Same value and same reason as
#: ``adapters_validation._UNKNOWN``: unknown is neither green nor red, because
#: either of those reads as a verdict that was reached.
_UNKNOWN = "#64748b"
_UNKNOWN_BG = "#f1f5f9"

#: Printed in the tier strip's "N groups / N records" cell when the report did
#: not carry the count. "0 groups" and "0 records" are measurements, and a
#: dashboard that prints them for a report that never counted anything is
#: describing an empty dataset that was never observed.
_UNCOUNTED = "n/a"


def _as_number(raw):
    """Read a reported number, or None when the report did not carry one.

    A bool is refused on purpose: ``True`` is an ``int`` in Python and would
    otherwise become the health score 1. NaN is refused too, because every
    comparison against it is False and it renders as the string "nan".
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    return num


#: Rendered in the metric name cell of a row whose threshold was never applied.
#: It leads the name so it survives the template's truncate_text(28).
_UNCHECKED_PREFIX = "[UNCHECKED] "
#: Rendered in the threshold cell of the same row, in place of a bound that was
#: not used. Printing the number there would imply the comparison happened.
_UNCHECKED_THRESHOLD = "NOT CHECKED"

#: An explicit ``metric_type`` on a report row, mapped onto a direction. This
#: can RESOLVE a direction the shared helper cannot infer from the name, but it
#: can never OVERRIDE one it can: see :func:`_row_direction`.
_DECLARED_DIRECTIONS = {
    "ratio": MetricDirection.HIGHER_IS_BETTER,
    "difference": MetricDirection.LOWER_IS_BETTER,
    "diff": MetricDirection.LOWER_IS_BETTER,
    "gap": MetricDirection.LOWER_IS_BETTER,
}


def _is_ratio_metric(name: str) -> bool:
    """True for ratio-type fairness metrics, where HIGHER is better.

    Thin delegation to the single owner of metric direction,
    :func:`vfairness.evaluation.vfairness_metrics._metric_direction.is_ratio_metric`.

    There is deliberately NO name test in this module any more. Until 2026-08-27
    this function guessed the direction locally and ended with a substring test
    for the disparate-impact token, so ``disparate_impact_difference`` (a
    violation magnitude) was graded as a four-fifths ratio: a 0.45 gap against a
    0.10 bound rendered ``breached=False`` and the #059669 GREEN swatch, i.e. a
    maximal violation was painted as a pass on the canvas an auditor reads. That
    is the third appearance of one bug class, after the gate and the metrics
    helper, and every appearance was a SECOND COPY of rules that already existed.
    Do not reintroduce a local copy here, however small: extend
    ``_metric_direction`` instead, where the whole precedence table is pinned.
    """
    return is_ratio_metric(name)


def _row_direction(metric_name, declared_type=None) -> MetricDirection:
    """Resolve one report row's better-direction, failing closed on conflict.

    The name is resolved ONLY by the shared helper. An explicit ``metric_type``
    on the row may fill in a direction the helper returns as UNKNOWN, because
    the producer of the row knows things the name does not. It may not contradict
    a direction the helper DID resolve: a row named ``demographic_parity_difference``
    but labelled ``"ratio"`` is contradictory evidence, and either reading can
    certify a violation as a pass, so the answer is UNKNOWN and the caller fails
    closed.
    """
    resolved = metric_direction(metric_name)
    declared = _DECLARED_DIRECTIONS.get(str(declared_type or "").strip().lower())
    if declared is None:
        return resolved
    if resolved is MetricDirection.UNKNOWN:
        return declared
    if declared is not resolved:
        return MetricDirection.UNKNOWN
    return resolved


def _row_outcome(metric_name, value, threshold, direction: MetricDirection) -> ThresholdOutcome:
    """Verdict for one row: PASS, FAIL or COULD_NOT_CHECK. Never two states.

    :func:`..evaluation.vfairness_metrics._metric_direction.check_threshold` is
    the authority whenever it can speak; it also routes NaN values and NaN
    thresholds to COULD_NOT_CHECK, which an unguarded ``value > threshold``
    reports as "not breached".

    A bound that CANNOT BE BREACHED grades nothing, and answering PASS for it is
    a green certificate issued by a comparison that never happened. On a
    higher-is-better metric the threshold is a required MINIMUM and the ratio
    family is non-negative, so a minimum of 0.0 is satisfied by every possible
    value including the worst one: ``demographic_parity_ratio = 0.00`` (the
    protected group is never selected at all) against a 0.0 minimum rendered the
    #059669 swatch and a PASS badge on this dashboard. Wave 3 closed exactly this
    hole in ``rendering/adapters_fairness._metric_state`` and recorded there that
    it is still open in the shared ``check_threshold``; this module inherited it
    by asking that helper. The MIRROR case is not degenerate and must keep
    working: 0.0 on a lower-is-better metric is a real zero-tolerance policy that
    ``abs(value)`` can exceed.
    """
    if direction is MetricDirection.UNKNOWN:
        return ThresholdOutcome.COULD_NOT_CHECK
    # THE VACUOUS-BOUND RULE, ABOVE THE DIRECTION DISPATCH, AND NOT REINVENTED.
    # G13 2026-09-30. The block below it covered only the higher-is-better half
    # (a required minimum at or under zero), and the docstring above describes
    # only that half, so EVERY VACUOUS MAXIMUM was still graded. Measured on
    # this tree::
    #
    #     _row_outcome('demographic_parity_difference', 0.01, 2.0, LOWER) -> PASS
    #     vacuous_bound_reason(...) -> 'a maximum of 2.0000 on a metric whose
    #         values lie in [0.0000, 1.0000], where the largest possible
    #         magnitude is 1.0000, so no result can exceed it'
    #
    # A green PASS badge and the #059669 "safe" swatch for a requirement no data
    # can breach, on a multi-tier dashboard built to be exported. It is the same
    # row that `gate._metric_row_state` grades UNMEASURED through this same
    # canonical function, so the dashboard and the gate disagreed about one
    # metric. `vacuous_bound_reason` is a property of the bound AND the metric's
    # range together, which is why no local test on the number alone can stand
    # in for it: it returns None for a real zero-tolerance 0.0 maximum and for a
    # metric whose range is unknown, so nothing gradable is refused.
    #
    # The higher-is-better block below is kept as defence in depth even though
    # this line already covers `minimum <= 0.0` (measured: a 0.0 minimum on
    # disparate_impact_ratio returns a reason here). It states the rule where a
    # reader of this module looks for it, and it does not depend on the shared
    # helper keeping that behaviour. Sabotaging THIS line alone still reddens
    # the vacuous-maximum pins.
    if vacuous_bound_reason(metric_name, threshold, BoundRole.THRESHOLD):
        return ThresholdOutcome.COULD_NOT_CHECK
    if direction is MetricDirection.HIGHER_IS_BETTER:
        try:
            minimum = float(threshold)
        except (TypeError, ValueError):
            return ThresholdOutcome.COULD_NOT_CHECK
        if minimum != minimum or minimum <= 0.0:  # NaN, or a floor nothing can miss
            return ThresholdOutcome.COULD_NOT_CHECK
    outcome, _reason = check_threshold(metric_name, value, threshold)
    if outcome is not ThresholdOutcome.COULD_NOT_CHECK:
        return outcome
    # The shared helper could not read the direction off the NAME, but the row
    # declared it. Re-run the same comparison in the declared direction; an
    # unmeasurable value or bound still fails closed.
    if value != value or threshold != threshold:  # NaN test
        return ThresholdOutcome.COULD_NOT_CHECK
    hib = direction is MetricDirection.HIGHER_IS_BETTER
    if _is_breached(value, threshold, higher_is_better=hib):
        return ThresholdOutcome.FAIL
    return ThresholdOutcome.PASS


def _fmt_num(x) -> str:
    """Format a cell value, without raising on a value that is not a number."""
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _is_breached(value: float, threshold: float, *, higher_is_better: bool = False) -> bool:
    """Direction-aware breach test, with the SAME semantics as ``check_threshold``.

    A lower-is-better metric is a violation MAGNITUDE, so it is compared on
    ``abs(value)``: a demographic_parity_difference of -0.45 against a 0.10
    bound is a 0.45 violation, and the unsigned test reported it as no breach.
    """
    if higher_is_better:
        return value < threshold
    return abs(value) > threshold


def _breach_pct(value: float, threshold: float, *, higher_is_better: bool = False) -> float:
    """Compute how far a value breaches its threshold as a percentage."""
    if threshold == 0:
        return 0.0
    delta = (threshold - value) if higher_is_better else (abs(value) - threshold)
    return max(0.0, round(delta / abs(threshold) * 100, 1))


def _margin_pct(value: float, threshold: float, *, higher_is_better: bool = False) -> float:
    """Compute how close a passing value is to its threshold (% margin)."""
    if threshold == 0:
        return 100.0
    delta = (value - threshold) if higher_is_better else (threshold - abs(value))
    return max(0.0, round(delta / abs(threshold) * 100, 1))


def _metric_color(value: float, threshold: float, *, higher_is_better: bool = False) -> str:
    """Return color based on value vs. threshold."""
    if _is_breached(value, threshold, higher_is_better=higher_is_better):
        return "#dc2626"  # red, breached
    margin = _margin_pct(value, threshold, higher_is_better=higher_is_better)
    if margin <= 20:
        return "#f59e0b"  # amber, close to breach
    return "#059669"  # green, safe


def _severity_from_breach(breach_pct: float) -> str:
    """Map breach percentage to severity level."""
    if breach_pct >= 50:
        return "critical"
    if breach_pct >= 20:
        return "high"
    return "medium"


# Public Adapter


def reporting_dashboard_to_svg(
    report=None,
    *,
    title: str = "Multi-Tier Fairness Report Dashboard",
    tier: str = "operational",
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a comprehensive multi-tier, event-driven fairness reporting
    dashboard as an SVG.

    Parameters
    ----------
    report : GeneratedReport or dict, optional
        A ``GeneratedReport`` object from ``ReportGenerator``, or a plain
        dictionary with the required keys.  If None, renders a
        could-not-check canvas that states plainly that nothing was assessed
        (unless *example* is set).
    title : str
        Dashboard title.
    tier : str
        Which demo tier ``example=True`` renders: ``'executive'``,
        ``'operational'``, or ``'technical'``.  Ignored when *report* is
        provided (the tier is read from the report dict instead).
    example : bool
        Render the built-in demonstration report for *tier*, watermarked
        EXAMPLE on the canvas.  Off by default and never reached implicitly:
        see the note below.
    explanation : str
        Optional educational explanation text.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27 a missing
    *report* silently dispatched to ``_demo_operational`` and this function
    returned a complete, confident YELLOW analysis of a model called
    ``credit_risk_v3``: a 72/100 health score, three KPI scores, a metric grid,
    a threshold-breach row, three alerts and five recommendations, none of it
    measured from anything the caller supplied. An SVG is an export format: it
    leaves the building, an auditor reads it, and it outlives the version that
    produced it, so a fabricated analysis is the most damaging artifact this
    module can emit. A caller could not tell that output apart from a real one.
    Missing input is COULD NOT CHECK, and it is stated on the canvas. The demo
    fixtures survive only behind the explicit ``example=True``, watermarked.
    """
    if report is not None:
        return _from_report(report, title=title, explanation=explanation, save_path=save_path)
    if not example:
        return _not_assessed(title=title, explanation=explanation, save_path=save_path)
    dispatch = {
        "executive": _demo_executive,
        "operational": _demo_operational,
        "technical": _demo_technical,
    }
    fn = dispatch.get(tier.lower(), _demo_operational)
    return fn(title=title, explanation=explanation, save_path=save_path)


def _not_assessed(*, title, explanation, save_path):
    """The third state: a dashboard for a run that measured nothing.

    Every field that could read as a finding is withheld rather than defaulted.
    ``not_assessable`` drives both the template's single "Nothing was assessed"
    panel and (through ``rendering.explain._not_assessable``) the accessible
    ``<desc>`` and the action line, so the canvas and the screen reader tell one
    story. There is deliberately no score, no status colour beyond slate, no
    KPI and no recommendation: a reader must not be able to take a number off
    this canvas.
    """
    reason = (
        "No report was supplied to reporting_dashboard_to_svg, so no metric was "
        "evaluated and no threshold was applied on this run."
    )
    data = {
        "title": title,
        "subtitle": "Nothing was assessed: no report was supplied",
        "timestamp": _now(),
        "model_name": "",
        "model_version": "",
        "not_assessable": True,
        "not_assessable_reason": reason,
        "status_color": "#64748b",
        "status_bg": "#f1f5f9",
        "status_label": "NOT CHECKED",
        "report_tier": "",
        "metrics": [],
        "breaches": [],
        "alerts": [],
        "n_metrics": 0,
        "n_active_alerts": 0,
        "n_critical_alerts": 0,
        "narrative": (
            f"COULD NOT CHECK. {reason} This dashboard is not a pass and not a "
            f"failure: it certifies nothing about any model. Pass a GeneratedReport "
            f"(or a compatible dict) to obtain an assessment."
        ),
        "recommendations": [],
        "explanation": explanation,
    }
    return _render_and_save(data, save_path=save_path)


def _from_report(report, *, title, explanation, save_path):
    """Build template data from a GeneratedReport or compatible dict."""
    if hasattr(report, "to_dict"):
        d = report.to_dict()
    elif isinstance(report, dict):
        d = report
    else:
        d = {}
    if not isinstance(d, dict):
        d = {}

    hs = d.get("health_score") or {}
    if hasattr(hs, "to_dict"):
        hs = hs.to_dict()
    if not isinstance(hs, dict):
        hs = {}

    # CRITICAL, do not restore the defaults these four lines used to carry
    # (score 0, status "yellow", trend "stable", slope 0.0). ``GeneratedReport``
    # declares ``health_score`` Optional and its ``to_dict`` writes None when it
    # is unset, so ANY report generated without a health score walked straight
    # through those defaults and this dashboard printed a complete scorecard for
    # it: "0 /100" under a YELLOW badge, "TREND Stable", "slope +0.0000",
    # "METRIC COMPLIANCE 0", "ALERT SCORE 0", "DRIFT STABILITY 0" (all three in
    # the #dc2626 failing red), "ACTIVE ALERTS 0" in pass green, and
    # "0 metrics, 0 groups, 0 records". Not one of those numbers was measured
    # from anything. The empty-input guard added when the demo fallback was
    # removed only ever caught a literal ``None`` report, so a non-None report
    # with nothing in it bypassed it entirely. Reproduced 2026-08-27, fixed
    # 2026-08-28: a score that is not a number is NOT RECORDED, and an
    # unrecorded health score withholds the whole scorecard (see
    # ``scorecard_measured`` below) instead of defaulting into one.
    score = _as_number(hs.get("score"))
    health_recorded = score is not None
    status = hs.get("status")
    trend = hs.get("trend")
    trend_slope = _as_number(hs.get("trend_slope"))
    components = hs.get("components") or {}
    if not isinstance(components, dict):
        components = {}

    # THE FOURTH DEFAULT, closed 2026-08-28. Three of the four the note above
    # names were removed; this one was reinstated ONE LINE LATER by the lookup's
    # own fallback, ``_STATUS_COLORS.get(status, _STATUS_COLORS["yellow"])``,
    # which is the removed ``hs.get("status", "yellow")`` wearing a different
    # hat. It graded a YELLOW headline badge for a band that was never reported
    # (None), was empty, or was spelled in any case but lower ("RED", "Red",
    # " red"), and the SAME label drives the accessible <desc> severity through
    # ``explain._fr_reporting``, so a health score of 31 arriving as status "RED"
    # rendered YELLOW / MEDIUM on both surfaces at once: a DOWNGRADE of a red
    # report, not merely an invented one. The band is a measurement like the
    # score; it gets the same three states.
    #
    # Case and surrounding whitespace are normalised rather than refused,
    # because "RED" and "red" are the same claim by any reader; an unrecognised
    # WORD is not, and takes the could-not-check arm.
    band = _STATUS_COLORS.get(str(status).strip().lower()) if status is not None else None
    if health_recorded and band is not None:
        status_color, status_bg, status_label = band
    else:
        status_color, status_bg, status_label = _UNKNOWN, _UNKNOWN_BG, "NOT CHECKED"
    tier = d.get("tier", "OPERATIONAL")

    # Build metrics list
    metrics_data = []
    breaches_data = []
    unchecked_names = []
    sections = d.get("sections") or []
    raw_metrics = d.get("metrics", [])

    if isinstance(raw_metrics, list):
        for m in raw_metrics:
            if not isinstance(m, dict):
                continue
            name = m.get("name", m.get("metric_name", ""))
            # _MISSING, not 0 and not 0.1. A row that reported no value used to
            # be graded against a threshold of 0.1 as though it had measured
            # exactly 0.0000, which on a lower-is-better metric is a PERFECT
            # result: the row rendered the #059669 swatch, a PASS badge and
            # "all pass" in the groups cell, for a metric nobody computed. A row
            # that reported no threshold was graded against an invented 0.1
            # bound, which can fabricate a breach just as easily as an
            # all-clear. Absent is not zero, and neither one may be graded.
            value = m.get("value", _MISSING)
            threshold = m.get("threshold", _MISSING)
            # The direction is resolved by the shared helper, never guessed
            # here. Three outcomes, never two: a row whose threshold could not
            # be applied must not be able to reach the green swatch.
            direction = _row_direction(name, m.get("metric_type"))
            if value is _MISSING or threshold is _MISSING:
                outcome = ThresholdOutcome.COULD_NOT_CHECK
            else:
                outcome = _row_outcome(name, value, threshold, direction)
            # RESIDUE CLOSED, 2026-08-28 (wave 15). It read, correctly:
            # ``reporting_dashboard.svg`` draws this cell as
            # ``{% if m.n_affected_groups > 0 %}... affected{% else %}all pass``,
            # a two-state test on a bare number, so a row that never counted its
            # affected groups still printed "all pass" beside the [UNCHECKED]
            # name and the NOT CHECKED threshold this adapter does put there.
            # "all pass" is a group-level all-clear, and it was the LAST green
            # claim left on an unchecked row.
            #
            # The template now has the third and fourth branch (could_not_check,
            # and a count that was never reported), so the default goes: an
            # absent count stays None and prints "not reported". A verdict needs
            # three states, and a COUNT needs the same, because 0 is a
            # measurement here as much as anywhere else.
            n_affected = _as_number(m.get("n_affected_groups"))
            n_affected = None if n_affected is None else int(n_affected)

            if outcome is ThresholdOutcome.COULD_NOT_CHECK:
                unchecked_names.append(str(name))
                metrics_data.append(
                    {
                        "name": _UNCHECKED_PREFIX + str(name),
                        "display_value": _fmt_num(value),
                        "display_threshold": _UNCHECKED_THRESHOLD,
                        # Amber, not the #059669 green: this row asserts nothing.
                        "color": "#f59e0b",
                        "breached": False,
                        "warning": True,
                        "breach_pct": 0,
                        # The BREACH cell is
                        # `{% elif m.warning %}{{ m.margin_pct }}% margin`, a
                        # plain interpolation with no arithmetic behind it, so a
                        # word is safe here. A 0 printed "0% margin", which says
                        # this metric sits EXACTLY on its threshold: a precise
                        # measurement, for a row whose value was never read. The
                        # `warning` flag cannot be dropped to reach the cell's
                        # "n/a" branch, because the same flag drives the WARN
                        # badge and dropping it would badge the row PASS.
                        "margin_pct": _UNCOUNTED,
                        "n_affected_groups": n_affected,
                        "could_not_check": True,
                    }
                )
                continue

            hib = direction is MetricDirection.HIGHER_IS_BETTER
            breached = outcome is ThresholdOutcome.FAIL
            bp = _breach_pct(value, threshold, higher_is_better=hib) if breached else 0
            mp = _margin_pct(value, threshold, higher_is_better=hib) if not breached else 0
            if breached and n_affected is not None:
                # A measured breach affects at least one group, so a REPORTED
                # count of 0 beside it is contradictory and is raised to 1. An
                # ABSENT count is not raised: inventing "1 affected" for a
                # producer that never counted is a scope claim about who was
                # harmed, and the breach panel below already says "affected
                # groups not reported" for exactly this row.
                n_affected = max(n_affected, 1)

            metrics_data.append(
                {
                    "name": name,
                    "display_value": _fmt_num(value),
                    "display_threshold": _fmt_num(threshold),
                    "color": _metric_color(value, threshold, higher_is_better=hib),
                    "breached": breached,
                    "warning": (not breached and mp <= 20),
                    "breach_pct": bp,
                    "margin_pct": mp,
                    "n_affected_groups": n_affected,
                    "could_not_check": False,
                }
            )

            if breached:
                # "all groups" was the default here, so a breach whose scope the
                # producer never reported named EVERY group as affected. That is
                # the widest possible claim about who was harmed, invented from
                # an absent key, and it is the sentence a reader acts on first.
                raw_groups = m.get("affected_groups")
                groups_str = (
                    ", ".join(str(g) for g in raw_groups)
                    if isinstance(raw_groups, (list, tuple)) and raw_groups
                    else "affected groups not reported"
                )
                breaches_data.append(
                    {
                        "metric_name": name,
                        "display_value": _fmt_num(value),
                        "display_threshold": _fmt_num(threshold),
                        "breach_pct": int(bp),
                        "bar_w": min(80, int(bp * 0.8)),
                        "severity": _severity_from_breach(bp),
                        "affected_groups": groups_str,
                        "insight": m.get("insight", ""),
                    }
                )

    # Build alerts list
    alerts_data = []
    raw_alerts = d.get("alerts", [])
    for a in raw_alerts[:5]:
        if isinstance(a, dict):
            alerts_data.append(
                {
                    "severity": a.get("severity", "INFO"),
                    "metric_name": a.get("metric_name", ""),
                    "message": a.get("message", ""),
                    "time_ago": a.get("time_ago", ""),
                }
            )

    # Build recommendations list
    recs_data = []
    raw_recs = d.get("recommendations", [])
    for i, r in enumerate(raw_recs):
        if isinstance(r, str):
            priority = (
                "critical"
                if "URGENT" in r or "CRITICAL" in r
                else ("high" if "immediate" in r.lower() else "normal")
            )
            recs_data.append({"text": r, "priority": priority})
        elif isinstance(r, dict):
            recs_data.append({"text": r.get("text", ""), "priority": r.get("priority", "normal")})

    # Build narrative
    narrative = ""
    for sec in sections:
        if isinstance(sec, dict) and sec.get("title", "").lower() in (
            "summary",
            "health assessment",
            "narrative",
        ):
            narrative = sec.get("content", "")
            break
    if not narrative:
        narrative = hs.get("explanation", "")

    # The withheld band, said in words. A slate NOT CHECKED chip is the right
    # colour but it is not a sentence, and this module's own rule is that an
    # unmeasured thing is named where a reader sees it (see the KPI note below,
    # and the unchecked-metric note directly under this one). Phrased about the
    # INPUT, because the breach rule below may still raise the headline to RED
    # from a breach measured here, and that is a different claim from the band
    # the producer supplied.
    if health_recorded and band is None:
        reported = "nothing" if status is None else f"{str(status)[:40]!r}"
        note = (
            f"NOT REPORTED: the health record carries a score but no recognised status "
            f"band (it reported {reported}; the bands are GREEN, YELLOW and RED), so "
            f"this dashboard read none from it. The headline reads NOT CHECKED unless a "
            f"breach measured here raises it; a YELLOW badge would be a band nobody "
            f"assigned."
        )
        narrative = f"{narrative} {note}".strip() if narrative else note

    # An unchecked metric must be stated where a reader sees it, and it must not
    # sit silently inside a GREEN all-clear about the whole model. The health
    # score was computed upstream from metrics we have just established were
    # never compared to their bounds, so it cannot certify them.
    if unchecked_names:
        note = (
            f"COULD NOT CHECK: {len(unchecked_names)} metric(s) were not compared to "
            f"their thresholds because the direction of a better value could not be "
            f"resolved, or the value was not measurable: "
            f"{', '.join(unchecked_names)}. Those metrics are neither passing nor "
            f"failing here, and this report does not certify them."
        )
        narrative = f"{narrative} {note}".strip() if narrative else note
        if status_label == "GREEN":
            status_color, status_bg, status_label = _STATUS_COLORS["yellow"]

    # A MEASURED breach is a measured failure, and it outranks whatever headline
    # status the producer computed upstream. The health score arrives from a
    # separate calculation, so it will report GREEN (severity INFO in the
    # accessible <desc>) beside a row this adapter has just graded BREACH, and
    # the reader who cannot see the grid gets the more confident sentence of the
    # two. The adapter already withdraws GREEN for a could-not-check row above,
    # so it owns this decision; applying it to one of the two cases and not the
    # other is what made the <desc> and the badge disagree.
    if breaches_data and _STATUS_SEVERITY.get(status_label, 1) < _STATUS_SEVERITY["RED"]:
        breached_names = [str(b["metric_name"]) for b in breaches_data]
        note = (
            f"BREACH: {len(breached_names)} metric(s) exceeded their threshold on this "
            f"run: {', '.join(breached_names)}. The health score supplied with this "
            f"report reads {status_label}, which does not account for them, so the "
            f"headline status is raised to RED here."
        )
        narrative = f"{narrative} {note}".strip() if narrative else note
        status_color, status_bg, status_label = _STATUS_COLORS["red"]

    # The three KPI cards are drawn from the health record's component
    # sub-scores. An absent sub-score is WITHHELD from the card (see
    # ``kpi_fields`` below) and NAMED here, on the canvas, next to the headline:
    # an unmeasured card must not be readable as a measured failure, and a
    # silently blank card is not a statement either.
    _KPI_CARDS = (
        ("metric_compliance", "kpi_compliance", "metric compliance"),
        ("alert_frequency", "kpi_alert", "alert score"),
        ("drift_stability", "kpi_drift", "drift stability"),
    )
    # None means the component was not reported, and it is carried as None all
    # the way to the template so the card can say so. ``_as_number`` is what
    # separates an absent sub-score from a MEASURED 0: the old
    # ``int(... or 0)`` collapsed the two, so a genuine zero and a missing key
    # rendered the identical red 0/100.
    kpi_fields = {}
    missing_kpis = []
    for key, var, label in _KPI_CARDS:
        num = _as_number(components.get(key))
        kpi_fields[var] = None if num is None else int(num)
        kpi_fields[f"{var}_measured"] = num is not None
        if num is None:
            missing_kpis.append(label)
    if health_recorded and missing_kpis:
        note = (
            f"NOT MEASURED: {len(missing_kpis)} of {len(_KPI_CARDS)} KPI sub-score(s) "
            f"were not reported with this health score ({', '.join(missing_kpis)}). "
            # KEEP the phrase "an absence, not a failing score" intact:
            # tests/test_rowlevel_validator_reporting.py pins it, and it is the
            # sentence that tells a reader the blank card is not a zero. Only
            # the CAUSE clause changed, because the template can withhold the
            # card now and the old clause ("the template has no unmeasured
            # state") became false the moment it could.
            # No quotation marks in this sentence: the template escapes them to
            # &#39; / &#34; in the canvas text, which is correct XML and reads as
            # noise in an exported text layer.
            f"Those cards are drawn as not measured rather than as a number: a 0 "
            f"printed there would be an absence, not a failing score."
        )
        narrative = f"{narrative} {note}".strip() if narrative else note

    # A report that recorded no health score has no scorecard to draw. Every
    # cell of that scorecard is derived from the health record (score, band,
    # trend, slope and the three KPI components) or from counts the report did
    # not carry, so drawing it means inventing all of it, which is exactly what
    # this function used to do. The verdict stack is withheld as ONE unit
    # because the template couples it as one unit, and whatever WAS measured is
    # carried into the narrative instead of being dropped: see
    # ``_scorecard_withheld``.
    if not health_recorded:
        return _scorecard_withheld(
            title=title,
            timestamp=_fmt_ts(d.get("timestamp")),
            model_name=d.get("model_name", ""),
            model_version=d.get("model_version", ""),
            metrics_data=metrics_data,
            breaches_data=breaches_data,
            alerts_data=alerts_data,
            unchecked_names=unchecked_names,
            narrative=narrative,
            recs_data=recs_data,
            explanation=explanation,
            save_path=save_path,
        )

    data = {
        "title": title,
        "subtitle": d.get("subtitle", f"{tier} report: event-driven multi-tier fairness analysis"),
        "timestamp": _fmt_ts(d.get("timestamp")),
        "model_name": d.get("model_name", ""),
        "model_version": d.get("model_version", ""),
        "health_score": f"{score:.0f}",
        "health_score_raw": min(score, 100),
        "status_color": status_color,
        "status_bg": status_bg,
        "status_label": status_label,
        "trend": trend,
        "event_trigger": d.get("event_trigger", "Scheduled"),
        "event_detail": d.get("event_detail", "Periodic report generation"),
        "time_window": d.get("time_window", "30 days"),
        "report_tier": tier,
        # CLOSED 2026-08-28, and it took BOTH halves. A health record carrying
        # no ``components`` used to print "0" in all three KPI cards, and the
        # template colours each of them by ``{% if kpi_compliance >= 70 %}``, so
        # an absent sub-score rendered as a FAILING 0/100 in #dc2626: a
        # fabricated BREACH, not a fabricated all-clear, and the kind that sends
        # someone after a compliance problem that was never measured.
        #
        # DO NOT reinstate ``int(... or 0)`` here. The withholding branch was
        # added to ``reporting_dashboard.svg`` in one wave while this dict went
        # on coercing to 0 and never sent the flag, so ``default(true)`` chose
        # the measured arm and the three red cards survived a fix that had
        # "landed" in the template. A flag nobody sets renders exactly like no
        # flag at all, which is why this pair has to move together.
        # ``_as_number`` also stops a non-numeric component taking the whole
        # render down inside ``int()``.
        **kpi_fields,
        "n_active_alerts": d.get("n_active_alerts", len(alerts_data)),
        "n_critical_alerts": d.get(
            "n_critical_alerts", sum(1 for a in alerts_data if a.get("severity") == "CRITICAL")
        ),
        # The slope line is `{% if trend_slope is defined %}slope: {{ … }}/day`,
        # so the ONLY way to withhold it is to leave the key out of this dict:
        # None is "defined" and printed the literal word None beside /day, and
        # the old 0.0 default printed "+0.0000/day", a measured-looking flat
        # trend for a report that reported no slope at all.
        **({"trend_slope": f"{trend_slope:+.4f}"} if trend_slope is not None else {}),
        # The GRADED count, not the row count. ``explain._fr_reporting`` writes
        # this key as the DENOMINATOR of the accessible <desc>'s headline,
        # "N breach(es) of M", so an unchecked row landing in M made the screen
        # reader say "0 breach(es) of 2" about a run that compared one metric to
        # its bound and never read the other. An ungraded row belongs to no
        # count that implies it was measured; it is named in the narrative and
        # carries its own [UNCHECKED] row instead.
        "n_metrics": d.get("n_metrics", len(metrics_data) - len(unchecked_names)),
        # "n/a", not 0. The tier strip reads "N metrics, N groups, N records",
        # and "0 groups, 0 records" describes a dataset that was observed and
        # found to be empty. A report that did not carry the counts observed
        # nothing of the sort. These two cells are plain interpolations in the
        # template, with no comparison behind them, so a word is safe here where
        # it would break the KPI cards.
        "n_groups": d.get("n_groups", _UNCOUNTED),
        "n_records": d.get("n_records", _UNCOUNTED),
        "metrics": metrics_data,
        "breaches": breaches_data,
        "alerts": alerts_data,
        "narrative": narrative,
        "recommendations": recs_data,
        "explanation": explanation,
    }

    svg = render_svg("reporting_dashboard", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _scorecard_withheld(
    *,
    title,
    timestamp,
    model_name,
    model_version,
    metrics_data,
    breaches_data,
    alerts_data,
    unchecked_names,
    narrative,
    recs_data,
    explanation,
    save_path,
):
    """The third state for a report that carried no health score.

    Every cell of the verdict stack is withheld rather than defaulted: no score,
    no band, no trend, no KPI, no "0 groups, 0 records". The stack goes as one
    unit because ``reporting_dashboard.svg`` couples it as one unit, and because
    the parts that CAN be defaulted from the adapter (score, band) sit beside
    parts that cannot (the trend word, the three KPI colours), so keeping the
    panel would have left fabricated cells on the canvas either way.

    What WAS measured is not dropped with it. Metric rows the report did carry
    are graded here as usual and their tally is written into the narrative, in
    words, so a breach can never go silent just because the headline could not
    be established. That is the headline rule applied in both directions: grade
    over the graded subset, and never let an ungraded row into a count.

    RESIDUE, template-blocked, wave 13 (2026-08-28), both in
    ``reporting_dashboard.svg``:

    * the panel's fixed second line reads "No report was supplied", which is
      true for ``reporting_dashboard_to_svg(None)`` and overstated here, where a
      report WAS supplied and carried no health score. The precise statement is
      in the subtitle directly above it, in the narrative below it and in the
      accessible ``<desc>``; the panel text needs to come from
      ``not_assessable_reason`` instead of being hard-coded.
    * the ``not_assessable`` branch re-computes ``health_y``, ``nlg_y``,
      ``rec_y`` and ``total_h`` but NOT ``breaches_y`` or ``alerts_y``, so a
      non-empty breach or alert panel would be drawn below the shortened canvas
      and silently clipped. They are passed empty for that reason ONLY, and
      their content is written into the narrative instead. Do not "restore" them
      here without fixing the two y-offsets first.
    """
    n_graded = len(metrics_data) - len(unchecked_names)
    n_breached = len(breaches_data)
    reason = (
        "The report supplied to reporting_dashboard_to_svg carries no health score, so no "
        "fairness health was computed, no status band was assigned, no trend was measured "
        "and no KPI sub-score was reported on this run."
    )

    lead = f"COULD NOT CHECK. {reason}"
    if metrics_data:
        # Only the graded rows are counted here. An unchecked row is named, and
        # named separately, but it enters neither the numerator nor the
        # denominator of anything that implies it was measured.
        lead += (
            f" {n_graded} metric(s) on this report WERE compared to a threshold: "
            f"{n_graded - n_breached} within bound, {n_breached} breached."
        )
        if breaches_data:
            lead += " BREACH: " + "; ".join(
                f"{b['metric_name']} {b['display_value']} against {b['display_threshold']} "
                f"({b['affected_groups']})"
                for b in breaches_data
            )
        if unchecked_names:
            lead += (
                f" {len(unchecked_names)} further metric(s) could not be checked at all "
                f"and are neither passing nor failing: {', '.join(unchecked_names)}."
            )
    else:
        lead += " No metric row was supplied either, so nothing at all was graded."
    if alerts_data:
        lead += f" {len(alerts_data)} alert(s) accompanied the report: " + "; ".join(
            f"[{a['severity']}] {a['metric_name']} {a['message']}".strip() for a in alerts_data
        )
    lead += (
        " This dashboard is not a pass and not a failure: it certifies nothing about the "
        "model as a whole."
    )

    data = {
        "title": title,
        "subtitle": "No health score was reported: the scorecard is withheld, not zero",
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "not_assessable": True,
        "not_assessable_reason": reason,
        "status_color": _UNKNOWN,
        "status_bg": _UNKNOWN_BG,
        "status_label": "NOT CHECKED",
        "report_tier": "",
        "metrics": [],
        "breaches": [],
        "alerts": [],
        "n_metrics": 0,
        "n_active_alerts": 0,
        "n_critical_alerts": 0,
        "narrative": f"{lead} {narrative}".strip() if narrative else lead,
        "recommendations": recs_data,
        "explanation": explanation,
    }
    return _render_and_save(data, save_path=save_path)


def _render_and_save(data, *, save_path):
    """Shared render + optional save."""
    svg = render_svg("reporting_dashboard", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# Executive Tier Demo: green, healthy, board-level summary


def _demo_executive(*, title, explanation, save_path):
    """Tier 1 demo: green status, all metrics passing, minimal detail."""
    data = {
        "title": title or "Executive Fairness Summary",
        # DO NOT REMOVE. Every number below is invented. This flag is what puts
        # the EXAMPLE band and watermark on the canvas, and it is the only thing
        # that stops this render being read as a finding about a real model.
        "is_example": True,
        "subtitle": "EXECUTIVE report: weekly board-level fairness overview",
        "timestamp": _now(),
        "model_name": "credit_risk_v3",
        "model_version": "3.2.1",
        "health_score": "91",
        "health_score_raw": 91,
        "status_color": "#059669",
        "status_bg": "#ecfdf5",
        "status_label": "GREEN",
        "trend": "stable",
        "trend_slope": "+0.0008",
        "event_trigger": "Scheduled",
        "event_detail": "Weekly executive summary",
        "time_window": "7 days",
        "report_tier": "EXECUTIVE",
        "kpi_compliance": 95,
        "kpi_alert": 90,
        "kpi_drift": 88,
        "n_active_alerts": 0,
        "n_critical_alerts": 0,
        "n_metrics": 3,
        "n_groups": 3,
        "n_records": 8400,
        "metrics": [
            {
                "name": "demographic_parity_diff",
                "display_value": "0.0420",
                "display_threshold": "0.1000",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 58,
                "n_affected_groups": 0,
            },
            {
                "name": "equal_opportunity_diff",
                "display_value": "0.0650",
                "display_threshold": "0.1000",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 35,
                "n_affected_groups": 0,
            },
            {
                "name": "equalized_odds_diff",
                "display_value": "0.0710",
                "display_threshold": "0.1500",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 53,
                "n_affected_groups": 0,
            },
        ],
        "breaches": [],
        "alerts": [],
        "narrative": (
            "Overall fairness status: Green (91/100). All three tracked fairness metrics "
            "are within acceptable limits. Trend: stable. 0 alerts in the evaluation window. "
            "The model continues to meet fairness requirements across all demographic groups."
        ),
        "recommendations": [
            {
                "text": "All systems nominal. Continue routine monitoring and maintain current alert threshold configuration.",
                "priority": "normal",
            },
            {
                "text": "Next scheduled review: weekly executive summary in 7 days.",
                "priority": "normal",
            },
        ],
        "explanation": explanation,
    }
    return _render_and_save(data, save_path=save_path)


# Operational Tier Demo: yellow, one breach, daily engineer view


def _demo_operational(*, title, explanation, save_path):
    """Tier 2 demo: yellow status, one threshold breach, alerts active."""
    data = {
        "title": title or "Operational Fairness Dashboard",
        # DO NOT REMOVE. See the note in _demo_executive: every number below is
        # invented, and this flag is what marks the canvas as an example.
        "is_example": True,
        "subtitle": "OPERATIONAL report: event-driven multi-tier fairness analysis",
        "timestamp": _now(),
        "model_name": "credit_risk_v3",
        "model_version": "3.2.1",
        "health_score": "72",
        "health_score_raw": 72,
        "status_color": "#f59e0b",
        "status_bg": "#fffbeb",
        "status_label": "YELLOW",
        "trend": "degrading",
        "trend_slope": "-0.0032",
        "event_trigger": "Threshold Breach",
        "event_detail": "equal_opportunity_diff > 0.10",
        "time_window": "30 days",
        "report_tier": "OPERATIONAL",
        "kpi_compliance": 68,
        "kpi_alert": 75,
        "kpi_drift": 82,
        "n_active_alerts": 3,
        "n_critical_alerts": 1,
        "n_metrics": 5,
        "n_groups": 3,
        "n_records": 4250,
        "metrics": [
            {
                "name": "demographic_parity_diff",
                "display_value": "0.0800",
                "display_threshold": "0.1000",
                "color": "#059669",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": 20,
                "n_affected_groups": 0,
            },
            {
                "name": "equal_opportunity_diff",
                "display_value": "0.1200",
                "display_threshold": "0.1000",
                "color": "#dc2626",
                "breached": True,
                "warning": False,
                "breach_pct": 20,
                "margin_pct": 0,
                "n_affected_groups": 2,
            },
            {
                "name": "equalized_odds_diff",
                "display_value": "0.1500",
                "display_threshold": "0.1500",
                "color": "#f59e0b",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": 0,
                "n_affected_groups": 0,
            },
            {
                "name": "predictive_parity_diff",
                "display_value": "0.0600",
                "display_threshold": "0.1000",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 40,
                "n_affected_groups": 0,
            },
            {
                "name": "calibration_diff",
                "display_value": "0.0400",
                "display_threshold": "0.0500",
                "color": "#f59e0b",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": 20,
                "n_affected_groups": 0,
            },
        ],
        "breaches": [
            {
                "metric_name": "equal_opportunity_diff",
                "display_value": "0.1200",
                "display_threshold": "0.1000",
                "breach_pct": 20,
                "bar_w": 16,
                "severity": "medium",
                "affected_groups": "Female, Other",
                "insight": "TPR gap 12pp",
            },
        ],
        "alerts": [
            {
                "severity": "CRITICAL",
                "metric_name": "equal_opportunity_diff",
                "message": "Threshold breached for 3 consecutive windows",
                "time_ago": "2h ago",
            },
            {
                "severity": "WARNING",
                "metric_name": "calibration_diff",
                "message": "Approaching threshold (80% margin consumed)",
                "time_ago": "6h ago",
            },
            {
                "severity": "WARNING",
                "metric_name": "demographic_parity_diff",
                "message": "Degrading trend detected (slope -0.003/day)",
                "time_ago": "1d ago",
            },
        ],
        "narrative": (
            "The fairness health score is 72/100 (yellow), composed of metric compliance (68), "
            "alert frequency (75), and drift stability (82). The trend is degrading "
            "(slope: -0.0032/day). 1 of 5 metrics is currently in breach: equal_opportunity_diff "
            "(0.12) exceeds its 0.10 threshold by 20%. Female and Other groups are most affected. "
            "Immediate investigation is recommended before the degrading trend impacts additional metrics."
        ),
        "recommendations": [
            {
                "text": "URGENT: equal_opportunity_diff has breached its threshold for 3 consecutive windows. Convene fairness review.",
                "priority": "critical",
            },
            {
                "text": "Investigate root cause: Female TPR (73%) vs. Male TPR (85%), a 12pp gap driving the breach.",
                "priority": "high",
            },
            {
                "text": "Consider threshold optimization or post-processing calibration to equalize true positive rates.",
                "priority": "normal",
            },
            {
                "text": "Schedule retraining with fairness constraints if post-processing is insufficient.",
                "priority": "normal",
            },
            {
                "text": "Monitor calibration_diff closely: at 80% of threshold with degrading trend.",
                "priority": "normal",
            },
        ],
        "explanation": explanation,
    }
    return _render_and_save(data, save_path=save_path)


# Technical Tier Demo: red, critical, full audit-level detail


def _demo_technical(*, title, explanation, save_path):
    """Tier 3 demo: red status, multiple breaches, full audit detail."""
    data = {
        "title": title or "Technical Fairness Audit Report",
        # DO NOT REMOVE. See the note in _demo_executive: every number below is
        # invented, and this flag is what marks the canvas as an example.
        "is_example": True,
        "subtitle": "TECHNICAL report: full statistical audit with drift decomposition",
        "timestamp": _now(),
        "model_name": "credit_risk_v3",
        "model_version": "3.2.1",
        "health_score": "38",
        "health_score_raw": 38,
        "status_color": "#dc2626",
        "status_bg": "#fef2f2",
        "status_label": "RED",
        "trend": "degrading",
        "trend_slope": "-0.0071",
        "event_trigger": "Critical Alert",
        "event_detail": "3 metrics in breach, drift confirmed",
        "time_window": "90 days",
        "report_tier": "TECHNICAL",
        "kpi_compliance": 32,
        "kpi_alert": 28,
        "kpi_drift": 55,
        "n_active_alerts": 7,
        "n_critical_alerts": 3,
        "n_metrics": 7,
        "n_groups": 4,
        "n_records": 12800,
        "metrics": [
            {
                "name": "demographic_parity_diff",
                "display_value": "0.1400",
                "display_threshold": "0.1000",
                "color": "#dc2626",
                "breached": True,
                "warning": False,
                "breach_pct": 40,
                "margin_pct": 0,
                "n_affected_groups": 3,
            },
            {
                "name": "equal_opportunity_diff",
                "display_value": "0.1800",
                "display_threshold": "0.1000",
                "color": "#dc2626",
                "breached": True,
                "warning": False,
                "breach_pct": 80,
                "margin_pct": 0,
                "n_affected_groups": 3,
            },
            {
                "name": "equalized_odds_diff",
                "display_value": "0.2200",
                "display_threshold": "0.1500",
                "color": "#dc2626",
                "breached": True,
                "warning": False,
                "breach_pct": 47,
                "margin_pct": 0,
                "n_affected_groups": 4,
            },
            {
                "name": "predictive_parity_diff",
                "display_value": "0.0900",
                "display_threshold": "0.1000",
                "color": "#f59e0b",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": 10,
                "n_affected_groups": 0,
            },
            {
                "name": "calibration_diff",
                "display_value": "0.0480",
                "display_threshold": "0.0500",
                "color": "#f59e0b",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": 4,
                "n_affected_groups": 0,
            },
            {
                "name": "treatment_equality_ratio",
                "display_value": "0.6200",
                "display_threshold": "0.8000",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 22,
                "n_affected_groups": 0,
            },
            {
                "name": "balance_positive_class",
                "display_value": "0.0350",
                "display_threshold": "0.0500",
                "color": "#059669",
                "breached": False,
                "warning": False,
                "breach_pct": 0,
                "margin_pct": 30,
                "n_affected_groups": 0,
            },
        ],
        "breaches": [
            {
                "metric_name": "equal_opportunity_diff",
                "display_value": "0.1800",
                "display_threshold": "0.1000",
                "breach_pct": 80,
                "bar_w": 64,
                "severity": "critical",
                "affected_groups": "Female, Non-binary, Age 60+",
                "insight": "TPR gap 18pp",
            },
            {
                "metric_name": "equalized_odds_diff",
                "display_value": "0.2200",
                "display_threshold": "0.1500",
                "breach_pct": 47,
                "bar_w": 38,
                "severity": "high",
                "affected_groups": "All 4 demographic groups",
                "insight": "FPR+TPR combined",
            },
            {
                "metric_name": "demographic_parity_diff",
                "display_value": "0.1400",
                "display_threshold": "0.1000",
                "breach_pct": 40,
                "bar_w": 32,
                "severity": "high",
                "affected_groups": "Female, Non-binary, Age 60+",
                "insight": "Selection rate gap",
            },
        ],
        "alerts": [
            {
                "severity": "CRITICAL",
                "metric_name": "equal_opportunity_diff",
                "message": "80% above threshold, 5 consecutive windows",
                "time_ago": "35m ago",
            },
            {
                "severity": "CRITICAL",
                "metric_name": "equalized_odds_diff",
                "message": "Crossed threshold after drift event on day 47",
                "time_ago": "1h ago",
            },
            {
                "severity": "CRITICAL",
                "metric_name": "demographic_parity_diff",
                "message": "New breach, warning since day 62",
                "time_ago": "3h ago",
            },
            {
                "severity": "WARNING",
                "metric_name": "predictive_parity_diff",
                "message": "10% margin remaining, degrading slope -0.004/day",
                "time_ago": "6h ago",
            },
            {
                "severity": "WARNING",
                "metric_name": "calibration_diff",
                "message": "4% margin, likely to breach within 2 days",
                "time_ago": "12h ago",
            },
        ],
        "narrative": (
            "Composite Health Score: 38.00/100.00 | Status: RED | Trend: degrading "
            "(slope=-0.0071/day). Components: metric_compliance=32.00, alert_frequency=28.00, "
            "drift_stability=55.00. Evaluation window: 7 metric records, 7 alerts. "
            "3 of 7 metrics are in active breach. equal_opportunity_diff is the most critical "
            "at 80% above threshold (0.18 vs. 0.10), affecting Female, Non-binary, and Age 60+ groups. "
            "A confirmed distribution drift event on day 47 (KS=0.22, p=0.003, mean_shift=+0.08) "
            "coincides with the breach onset. 2 additional metrics (predictive_parity_diff, "
            "calibration_diff) are within 10% of their thresholds and projected to breach "
            "within 2 to 5 days at current degradation rates."
        ),
        "recommendations": [
            {
                "text": "URGENT: Fairness health score is critical (38/100). Convene the fairness review board and consider pausing model updates until resolved.",
                "priority": "critical",
            },
            {
                "text": "CRITICAL: 3 metrics in active breach. equal_opportunity_diff at 80% above threshold requires immediate root-cause investigation.",
                "priority": "critical",
            },
            {
                "text": "Investigate confirmed drift event on day 47 (KS=0.22, p=0.003). Check upstream data pipeline for schema or distribution changes.",
                "priority": "high",
            },
            {
                "text": "Apply emergency post-processing: group-specific threshold optimization can reduce equal_opportunity_diff by ~60% with <1pp accuracy cost.",
                "priority": "high",
            },
            {
                "text": "Schedule fairness-constrained retraining on updated data. Current model was trained on pre-drift distribution.",
                "priority": "normal",
            },
            {
                "text": "Monitor predictive_parity_diff and calibration_diff hourly: both projected to breach within 2 to 5 days.",
                "priority": "normal",
            },
        ],
        "explanation": explanation,
    }
    return _render_and_save(data, save_path=save_path)
