"""
Adapters that transform monitoring & drift-detection result objects
into flat dictionaries suitable for SVG template rendering.

Each adapter:
    1. Accepts a result object (dataclass or list of dataclasses).
    2. Extracts and pre-computes all pixel-level values the SVG needs.
    data["explanation"] = explanation
    3. Calls ``render_svg()`` and optionally writes to disk.
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .._bands import DRIFT_BAND_THRESHOLDS, _is_nan
from .engine import render_svg

logger = logging.getLogger(__name__)

# Neutral slate, the muted tone these templates already use for labels. It is
# the "we could not measure this" colour: deliberately neither the green that
# reads as a pass nor the red that reads as a finding.
NOT_MEASURED_COLOR = "#94a3b8"
NOT_MEASURED_BG = "#f1f5f9"
NOT_MEASURED_LABEL = "NOT MEASURED"


def _mark_not_assessable(data: Dict[str, Any], reason: str, headline: str) -> None:
    """Arm the third state on one chart's data dict: could-not-check.

    CRITICAL, three states, never two. A count of zero is not a finding of zero.
    ``alert_timeline_to_svg([])`` rendered "0 of 0 windows raised alerts
    (0 clean)" beside TOTAL EVENTS 0 / WITH ALERTS 0 / CLEAN 0 at severity INFO,
    which is the canvas for a monitoring period that was audited and came back
    clean. Nothing had been monitored at all. The same shape appeared on the live
    dashboard (STATUS OK, 0 alerts, on 0 metrics) and on the drift report
    (STABLE, overall score 0.000, across 0 scales).

    Setting both keys here means the canvas and the accessible ``<desc>`` cannot
    disagree: the template swaps its verdict stack for the shared
    ``_could_not_check.svg`` panel, and ``rendering.explain._not_assessable``
    reads the SAME flag to replace the finding sentence and the action line.
    Setting only one of the two is what let a chart say COULD NOT CHECK in one
    place and print a verdict in the other.
    """
    data["not_assessable"] = True
    data["not_assessable_reason"] = reason
    data["na_headline"] = headline


class _NotMeasured:
    """A statistic the row never reported, safe to hand to a template comparison.

    CRITICAL, do not replace this with None or with 0.0. ``drift_report.svg``
    compares two of the per-scale statistics INLINE, in the template:
    ``{{ '#dc2626' if s.p_value < 0.05 else '#475569' }}`` and
    ``{% if s.mean_shift >= 0 %}``. A ``None`` reaches those lines as
    "'<' not supported between instances of 'NoneType' and 'float'", which
    ``render_svg`` re-raises as a ValueError, so ONE scale that did not report a
    p-value took the entire drift report down instead of rendering the other
    scales that were measured perfectly well. Reproduced 2026-08-27; a partial
    row must answer could-not-check, never end the process.

    Every comparison against this object answers False, which is the correct "no
    claim in either direction" for a number that does not exist: the p-value
    cell gets the neutral slate rather than the significant red, and the mean
    shift is printed without an invented ``+`` sign. ``float()`` raises, so the
    engine's ``f3``/``f4`` filters fall to their "N/A" branch and the cell says
    so in words rather than printing 0.0000 or the string "nan".
    """

    __slots__ = ()

    def __lt__(self, other) -> bool:
        return False

    def __le__(self, other) -> bool:
        return False

    def __gt__(self, other) -> bool:
        return False

    def __ge__(self, other) -> bool:
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, _NotMeasured)

    def __ne__(self, other) -> bool:
        return not isinstance(other, _NotMeasured)

    def __hash__(self) -> int:
        return hash(("_NotMeasured",))

    def __bool__(self) -> bool:
        # A statistic that was never measured is not evidence of anything, so it
        # must not satisfy an ``{% if %}`` that gates a finding.
        return False

    def __float__(self):
        raise TypeError("this statistic was never measured")

    def __str__(self) -> str:
        return "not measured"

    def __repr__(self) -> str:
        return "NOT_MEASURED"


#: The single instance. Identity comparison is how this module tests for it,
#: because ``==`` is deliberately False against every real number.
NOT_MEASURED = _NotMeasured()


def _measured(value: Any) -> Optional[float]:
    """Read a reported statistic, or None when the row did not report one.

    A bool is refused: ``True`` is an ``int`` in Python and would become the
    drift score 1.0, the most alarming value on the chart. NaN is refused
    because it is the sentinel the upstream detector writes for "not computed",
    and because every comparison against it silently answers False.

    READINESS-6, 2026-09-10, two holes in the sentence above.

    ``isinstance(value, bool)`` is False for ``np.bool_``, which is not a Python
    bool, so a boolean column read out of a DataFrame produced precisely the 1.0
    this docstring says it refuses.

    INFINITY was not refused at all, and it is worse than the NaN beside it.
    ``_clamp01`` pins ``inf`` to 1.0, so an unmeasurable drift score drew a
    FULL-WIDTH bar, the most alarming geometry the dashboard can produce, as a
    measured reading. ``-inf`` clamps to 0.0 and drew a zero-width bar, which is
    the geometry this module chose to MEAN "not measured", so that half was
    wrong in the opposite direction: a fabricated all-clear that looks identical
    to an honest blank. Both are now None, which is the state ``_clamp01`` and
    the status chip beside it already handle correctly.

    A numeric STRING is still accepted, deliberately. Rows reach this adapter
    from JSON and CSV, where "0.5" is a recorded measurement that happens to
    have been serialised, and refusing it would discard real evidence. That is
    the opposite trade-off from :func:`compliance._as_float`, which refuses
    strings because its input is a caller's free-text field; the contracts
    genuinely differ, and this paragraph is here so the difference is a decision
    rather than an oversight.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):  # NaN, +inf, -inf
        return None
    return num


def _clamp01(value: Optional[float]) -> float:
    """Clamp to [0, 1], mapping an unmeasurable value to 0.0.

    ``max(0.0, min(1.0, nan))`` is 1.0 in CPython: ``nan < 1.0`` is False so
    ``min`` returns 1.0, and ``max(0.0, 1.0)`` is 1.0. A NaN drift score
    therefore drew a FULL-WIDTH bar, the most alarming-looking geometry the
    dashboard can produce, painted in the most reassuring colour. Zero width is
    the honest geometry for a number that does not exist; the status chip beside
    it is what carries the "not measured" meaning. None arrives here from
    ``_measured`` for a row that reported no score at all, and means the same
    thing as the NaN: zero width, and the chip beside it says why.
    """
    if value is None or _is_nan(value):
        return 0.0
    return max(0.0, min(1.0, value))


# colour helpers


def _metric_color(value: float, threshold: float = 0.8) -> str:
    """Green when above threshold (fair), red when below."""
    if value >= threshold:
        return "#059669"  # emerald-600
    if value >= threshold * 0.85:
        return "#f59e0b"  # amber-500
    return "#dc2626"  # red-600


def _drift_color(score: Optional[float]) -> str:
    """Map a 0-1 drift score to a colour.

    An unmeasurable score is neutral grey, never green. Every ``>=`` test
    against NaN is False, so without this guard the function fell through to
    the emerald return and a could-not-measure drift statistic was painted as
    the calmest state the dashboard has. Thresholds come from
    ``_bands.DRIFT_BAND_THRESHOLDS`` so the colour and the explainer severity
    (``_bands.drift_severity``) cannot drift apart. None means the same as the
    NaN and takes the same branch: a scale that reported no score is neither
    stable nor drifting, and must not be painted as either.
    """
    if score is None or _is_nan(score):
        return NOT_MEASURED_COLOR
    if score >= DRIFT_BAND_THRESHOLDS[1]:
        return "#dc2626"
    if score >= DRIFT_BAND_THRESHOLDS[0]:
        return "#f59e0b"
    return "#059669"


def _trend_slope_color(slope: float) -> str:
    """Colour a trend slope: red for strongly changing, green for stable.

    Carries the same NaN guard as ``_drift_color``: ``abs(nan)`` is NaN and
    every comparison against it is False, so an uncomputable slope fell through
    to green and read as "stable".
    """
    if _is_nan(slope):
        return NOT_MEASURED_COLOR
    mag = abs(slope)
    if mag >= 0.01:
        return "#dc2626"
    if mag >= 0.005:
        return "#f59e0b"
    return "#059669"


def _window_excluded_groups(wm: Any) -> List[str]:
    """Display names of the strata ``min_samples`` kept OUT of every rate drawn.

    BGL5 (2026-09-27). ``WindowMetrics.excluded_groups`` is the tracker's own
    record of them ("the groups the ``min_samples`` floor kept OUT of every
    aggregate over that column", R-3), and ``_group_positive_rates`` says in its
    docstring that "groups below min_samples are absent from the result. That
    absence is invisible to a caller looking only at this dict". Until now no
    renderer in this module read the key (``grep -rn excluded_groups
    src/vfairness/rendering/`` answered adapters_fairness.py and adapters.py
    only), so the Group Positive Rates table presented the surviving groups as
    though they were the groups. MEASURED on a real two-column monitor run,
    ``excluded_groups={'group_gender': {'Other': 7}}`` while the canvas drew
    "Group Positive Rates Female 69.7% Male 71.0%" and the string 'Other'
    appeared nowhere in the canvas text or in the metadata JSON.

    The shape is ``{attribute: {group: row_count}}``; the display form matches
    ``adapters_fairness._excluded_groups`` ("name (n=7)") with the attribute
    named too, because one window can drop a stratum of one column and not of
    another.
    """
    raw = getattr(wm, "excluded_groups", None)
    if not isinstance(raw, dict):
        return []
    names: List[str] = []
    for attr, dropped in sorted(raw.items()):
        if not isinstance(dropped, dict):
            continue
        for group, count in sorted(dropped.items()):
            names.append(f"{group} (n={count}, {attr})")
    return names


# 1.  WindowMetrics  →  monitoring_dashboard.svg


def monitoring_dashboard_to_svg(
    window_metrics,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a FairnessMonitor WindowMetrics snapshot as a live dashboard.

    Parameters
    ----------
    window_metrics : WindowMetrics
        The result of ``FairnessMonitor.update_and_check()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    wm = window_metrics

    # CRITICAL, a partial window must not take the render down. Every one of
    # these four records reached the code below as an attribute and was then
    # iterated or tested directly, so a window carrying None for any of them
    # (a deserialised snapshot with a null field, a partially built one) raised
    # AttributeError or "argument of type 'NoneType' is not iterable" and the
    # whole dashboard was lost. A crash is not the third state. An absent record
    # is read here as "this window holds no such record", which is exactly the
    # case the per-row branches below already answer on the canvas.
    _raw_metrics = getattr(wm, "metrics", None)
    metrics_in = _raw_metrics if isinstance(_raw_metrics, dict) else {}
    _raw_alerts = getattr(wm, "alerts", None)
    alerts_in = _raw_alerts if isinstance(_raw_alerts, dict) else {}
    _raw_rates = getattr(wm, "group_rates", None)
    rates_in = _raw_rates if isinstance(_raw_rates, dict) else {}
    _raw_mmd = getattr(wm, "mmd_scores", None)
    mmd_in = _raw_mmd if isinstance(_raw_mmd, dict) else {}

    # -- timestamp --
    ts = getattr(wm, "timestamp", None)
    # The header slot is "{{ timestamp_date }} {{ timestamp_time }}", so the
    # absent case fills the date and blanks the time: one honest phrase rather
    # than the same placeholder twice. It used to be an em dash in each slot.
    try:
        ts_date = ts.strftime("%Y-%m-%d") if ts else "not recorded"
        ts_time = ts.strftime("%H:%M:%S") if ts else ""
    except (AttributeError, TypeError, ValueError):
        ts_date, ts_time = "not recorded", ""

    # -- metrics rows --
    bar_max = 80  # px
    metrics_list: List[Dict[str, Any]] = []
    for name, value in sorted(metrics_in.items()):
        # Three states PER ROW, never two. ``alerts.get(name, False)`` defaulted
        # the GUARDRAIL RESULT for a metric that carries no entry in ``alerts``,
        # which is a metric that was computed but never compared to a threshold.
        # The row then took the emerald OK badge and the emerald bar, so "no
        # threshold was configured for this metric" and "this metric is inside
        # its threshold" were the same picture. A default is not a measurement.
        # An entry recorded as None is a guardrail slot that exists and holds no
        # result, which is the same silence as no entry at all. `bool(None)` is
        # False, so it used to take the emerald OK pill: an all-clear from an
        # unrecorded result.
        checked = name in alerts_in and alerts_in[name] is not None
        alert = bool(alerts_in[name]) if checked else False
        # The VALUE is its own state, separate from the guardrail result beside
        # it: `abs(value)` raised TypeError on a metric the window recorded
        # without a number, and `f"{value:.3f}"` raised on the very next line.
        # A metric that reported no reading gets no number and no bar, and its
        # guardrail result (if one was recorded) still stands.
        measured = _measured(value)
        clamped = _clamp01(abs(measured) if measured is not None else None)
        if not checked or measured is None:
            color = NOT_MEASURED_COLOR
        elif alert:
            color = "#dc2626"
        else:
            color = "#059669"
        metrics_list.append(
            {
                "name": name,
                # "n/a", not "not measured": this cell is drawn at x=310 and the
                # bar starts at x=380, so a longer phrase at font-size 12 runs
                # under the bar and is unreadable. It is the same spelling the
                # MMD cell below already uses for a statistic that reported
                # nothing, in the same neutral slate. Measured by rasterising.
                "display_value": f"{measured:.3f}" if measured is not None else "n/a",
                "bar_w": max(4, int(clamped * bar_max)),
                "color": color,
                "alert": alert,
                "checked": checked,
                "status_label": "ALERT" if alert else ("OK" if checked else "NOT CHECKED"),
            }
        )

    # -- group positive rates --
    groups_list: List[Dict[str, Any]] = []
    rate_bar_max = 100
    for attr_name, group_dict in sorted(rates_in.items()):
        if not isinstance(group_dict, dict):
            continue
        for group_label, rate in sorted(group_dict.items()):
            # `f"{rate:.1%}"` raised TypeError for a group whose positive rate
            # the window did not record, and a rate of 0.0% is a measurement (no
            # member of this group was predicted positive), so the absence
            # cannot be printed as one.
            measured_rate = _measured(rate)
            clamped = _clamp01(measured_rate)
            groups_list.append(
                {
                    "label": f"{group_label}",
                    "display_rate": (
                        f"{measured_rate:.1%}" if measured_rate is not None else "not measured"
                    ),
                    "bar_w": max(4, int(clamped * rate_bar_max)),
                    "color": "#3b82f6" if measured_rate is not None else NOT_MEASURED_COLOR,
                }
            )

    # A stratum kept out of every rate above is NAMED IN THE TABLE, not dropped
    # from it, the same way post_processing.calibration.visualization's
    # plot_group_calibration names a group too small to draw a curve for. It
    # reuses the not-measured row this table already builds for a rate the window
    # recorded without a number, so the reader meets it where the rates are.
    excluded_display = _window_excluded_groups(wm)
    excluded_rows: List[Dict[str, Any]] = []
    for name in excluded_display:
        excluded_rows.append(
            {
                "label": name,
                "display_rate": "not measured",
                "bar_w": 4,
                "color": NOT_MEASURED_COLOR,
            }
        )

    # -- MMD scores --
    mmd_bar_max = 400
    mmd_list: List[Dict[str, Any]] = []
    for attr, score in sorted(mmd_in.items()):
        # `_measured` refuses None, NaN and anything non-numeric alike, so the
        # single `score is None` test below carries what `_is_nan(score)` used
        # to: an MMD score the window recorded without a number cannot raise
        # here, and it cannot be compared to the 0.1 shift threshold either.
        score = _measured(score)
        clamped = _clamp01(score)
        # Three states, never two. The template used to decide the chip itself
        # with `{% if s.score >= 0.1 %}`, and NaN fails that test exactly as a
        # tiny score does, so an unmeasurable statistic rendered as STABLE. The
        # verdict is computed here, where NaN can be named, and the template
        # only draws what it is given: this module's contract is to
        # "pre-compute all pixel-level values the SVG needs".
        if score is None:
            status, status_color, status_bg = (
                NOT_MEASURED_LABEL,
                NOT_MEASURED_COLOR,
                NOT_MEASURED_BG,
            )
        elif score >= 0.1:
            status, status_color, status_bg = "SHIFT", "#92400e", "#fef3c7"
        else:
            status, status_color, status_bg = "STABLE", "#059669", "#ecfdf5"
        mmd_list.append(
            {
                "attribute": attr,
                "score": score,
                "display_score": "n/a" if score is None else f"{score:.4f}",
                "bar_w": max(4, int(clamped * mmd_bar_max)),
                "color": _drift_color(score),
                "status": status,
                "status_color": status_color,
                "status_bg": status_bg,
            }
        )

    n_alerts = sum(1 for a in alerts_in.values() if a)

    # Three states on the HEADLINE badge, never two. The template's
    # `{% if any_alert %}` sends every falsy value to the emerald OK branch, and
    # a window that recorded no overall verdict is falsy, so STATUS OK in
    # 26px green was printed for a window that never said whether anything
    # fired. A recorded True or False still bands normally.
    #
    # A window that recorded no verdict of its own is still read from its
    # guardrail results when it has any: one alert in `alerts` IS a firing, and
    # withholding the badge over a metric that visibly says ALERT would be the
    # opposite fabrication.
    _raw_any_alert = getattr(wm, "any_alert", None)
    if isinstance(_raw_any_alert, bool):
        any_alert, any_alert_recorded = _raw_any_alert, True
    elif n_alerts:
        any_alert, any_alert_recorded = True, True
    elif any(name in alerts_in and alerts_in[name] is not None for name in metrics_in):
        any_alert, any_alert_recorded = False, True
    else:
        any_alert, any_alert_recorded = False, False

    _sample_count = getattr(wm, "sample_count", None)
    data = {
        "title": "FairnessMonitor: Live Dashboard",
        "any_alert": any_alert,
        "any_alert_recorded": any_alert_recorded,
        "n_alerts": n_alerts,
        "n_metrics": len(metrics_in),
        # "N metrics monitored" counted every metric that was COMPUTED, and a
        # metric with no entry in wm.alerts was never compared to a guardrail.
        # Kept as a separate key rather than narrowing n_metrics, because
        # explain._fr_monitoring reads n_metrics as the population size.
        "n_checked": sum(
            1 for name in metrics_in if name in alerts_in and alerts_in[name] is not None
        ),
        # Printed straight onto the canvas as "{{ sample_count }} samples in
        # window", so a None would reach the reader as the literal "None".
        "sample_count": _sample_count if _sample_count is not None else "an unreported number of",
        "timestamp_date": ts_date,
        "timestamp_time": ts_time,
        "metrics": metrics_list,
        # The excluded rows lead, because this list is TRUNCATED at eight and a
        # truncation that can drop the disclosure is not a formatting choice.
        # Measured: 2 measured groups + 1 excluded renders 3 rows either way.
        "groups": (excluded_rows + groups_list)[:8],
        # Read by explain._excluded_clause, which build_explanation appends to
        # the finding for EVERY chart and which floors the severity with it. That
        # is the machine-readable half of the same disclosure, for the reader who
        # meets the <desc> rather than the canvas.
        "excluded_groups": excluded_display,
        "mmd_scores": mmd_list,
    }

    # The window carried no metric, no group rate and no drift statistic, so
    # nothing was watched. The STATUS OK badge and "0 alerts" are then a claim
    # about a guardrail that was never applied. See _mark_not_assessable.
    if not metrics_list and not groups_list and not mmd_list:
        _mark_not_assessable(
            data,
            "This monitoring window carried no fairness metric, no group rate and no "
            "drift statistic, so no guardrail was applied and no alert could fire.",
            "Nothing was monitored",
        )

    data["explanation"] = explanation
    svg = render_svg("monitoring_dashboard", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  MultiscaleDriftResult  →  drift_report.svg


def drift_report_to_svg(
    drift_result,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a MultiscaleDriftResult as a multi-scale drift report.

    Parameters
    ----------
    drift_result : MultiscaleDriftResult
        The result of ``FairnessDriftDetector.check_drift()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    dr = drift_result
    score_bar_max = 120  # px

    scales_list: List[Dict[str, Any]] = []
    #: Scales that reported a drift score, and scales that reported nothing but
    #: their name. A scale in the second set enters no count and no colour that
    #: implies it was measured.
    n_scored = 0
    partial_scales: List[str] = []
    _raw_scales = getattr(dr, "scales", None)
    for scale_name, sr in (_raw_scales if isinstance(_raw_scales, dict) else {}).items():
        label = str(scale_name).replace("_", " ").title()
        # PER ROW, three states, never two. Every one of these six statistics is
        # optional in practice: a scale can compute a KS statistic and no
        # p-value, or a mean shift over a reference window that was itself
        # empty. Each is read on its own, so one absent number withholds ONE
        # cell instead of grading the whole row from defaults, and (before this
        # change) instead of killing the render for every other scale.
        score = _measured(getattr(sr, "drift_score", None))
        ks_stat = _measured(getattr(sr, "ks_statistic", None))
        p_value = _measured(getattr(sr, "p_value", None))
        mean_shift = _measured(getattr(sr, "mean_shift", None))
        ref_mean = _measured(getattr(sr, "reference_mean", None))
        cur_mean = _measured(getattr(sr, "current_mean", None))
        detected_raw = getattr(sr, "drift_detected", None)
        detected = detected_raw if isinstance(detected_raw, bool) else None

        unmeasured = [
            field
            for field, val in (
                ("drift score", score),
                ("KS statistic", ks_stat),
                ("p-value", p_value),
                ("mean shift", mean_shift),
                ("reference mean", ref_mean),
                ("current mean", cur_mean),
                ("drift verdict", detected),
            )
            if val is None
        ]
        if score is None:
            partial_scales.append(label)
        else:
            n_scored += 1
            if unmeasured:
                partial_scales.append(label)

        clamped = _clamp01(score)
        scales_list.append(
            {
                "scale_name": label,
                # None where the row reported nothing: the f3/f4 filters print
                # "N/A" for it, and 0.000 in a drift cell is the single most
                # reassuring number the chart can show.
                "drift_score": score,
                "ks_stat": ks_stat,
                "ref_mean": ref_mean,
                "cur_mean": cur_mean,
                # NOT_MEASURED, not None, for exactly the two the template
                # compares inline. See the _NotMeasured docstring: None here is
                # what raised ValueError and took down the whole report.
                "p_value": p_value if p_value is not None else NOT_MEASURED,
                "mean_shift": mean_shift if mean_shift is not None else NOT_MEASURED,
                # CLOSED 2026-08-28 (was RESIDUE, template-blocked, wave 13).
                # The status cell was `{% if s.drift_detected %}DRIFT{% else %}
                # STABLE`, a two-state test, so a scale that recorded no verdict
                # took the green STABLE badge: the most reassuring state the
                # chart has, awarded for silence. Neither branch is honest for an
                # unrecorded verdict, so drift_report.svg now reads
                # `verdict_recorded` FIRST and draws a slate NOT RECORDED pill.
                # Both keys are load-bearing; do not collapse them back into one.
                "drift_detected": bool(detected),
                "verdict_recorded": detected is not None,
                "score_bar_w": max(4, int(clamped * score_bar_max)),
                "color": _drift_color(score),
                "unmeasured_fields": unmeasured,
            }
        )

    overall_score = _measured(getattr(dr, "overall_drift_score", None))
    _raw_overall_detected = getattr(dr, "drift_detected", None)
    # Three states on the HEADLINE badge, never two. `{% if drift_detected %}`
    # sends every falsy value to the green STABLE branch, and None is falsy, so
    # a result object that recorded no overall verdict was crowned with the
    # calmest word this chart has. The badge now reads `overall_verdict_recorded`
    # first; keep both keys.
    overall_detected = _raw_overall_detected if isinstance(_raw_overall_detected, bool) else None
    # Read ONCE, through getattr, and reused by the two could-not-check reasons
    # below. Those two f-strings still said ``dr.metric``, so a result object
    # that carries no metric name raised AttributeError inside the branch whose
    # whole job is to explain that nothing was measured.
    metric_raw = str(getattr(dr, "metric", "") or "") or "this metric"
    metric_label = str(getattr(dr, "metric", "") or "")
    # The subtitle is the one line beside the headline badge, so the count of
    # scales that did NOT fully report belongs there rather than only in the
    # accessible <desc>. Headline rule: band over the graded subset, and state
    # the ungraded count where the reader of the verdict sees it.
    if partial_scales:
        metric_label = (
            f"{metric_label} ({len(partial_scales)} of {len(scales_list)} "
            f"scale(s) did not fully report)"
        )

    data = {
        "title": "FairnessDriftDetector: Drift Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "drift_detected": bool(overall_detected),
        "overall_verdict_recorded": overall_detected is not None,
        "overall_drift_score": overall_score,
        "overall_color": _drift_color(overall_score),
        "metric_name": metric_label,
        "scales": scales_list,
        "n_scored_scales": n_scored,
        "n_partial_scales": len(partial_scales),
    }

    # ``overall_drift_score`` is documented as the MAXIMUM drift_score across
    # all scales (operations.monitoring.drift.MultiscaleDriftResult), so with no
    # scale analysed it is a sentinel, not a measurement. Rendering it as 0.000
    # under a green STABLE badge reports that drift was looked for and not
    # found. The same is true of a report whose scales all exist but none of
    # which produced a drift score: the rows are there, nothing was measured,
    # and the maximum over nothing is still not a reading. See
    # _mark_not_assessable.
    if not scales_list:
        _mark_not_assessable(
            data,
            f"No temporal scale was analysed for '{metric_raw}', so the overall drift "
            f"score is the aggregate of nothing and no scale was compared to a reference.",
            "No scale was analysed",
        )
    elif n_scored == 0:
        _mark_not_assessable(
            data,
            f"None of the {len(scales_list)} temporal scale(s) for '{metric_raw}' reported a "
            f"drift score, so the overall score is the aggregate of nothing and no scale "
            f"was compared to a reference.",
            "No scale reported a drift score",
        )

    data["explanation"] = explanation
    svg = render_svg("drift_report", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  List[WindowMetrics]  →  alert_timeline.svg


#: The alert-timeline row cell is drawn through ``truncate_text(35)``, and a
#: metric key in this library is routinely 31 characters
#: ("demographic_parity_group_gender"), so the cell can carry the NAMES only
#: sometimes. This picks the names when they fit and falls back to the count
#: when they do not, so the cell never silently loses the end of a name to the
#: ellipsis; the <desc> carries the full list either way.
_TIMELINE_CELL_CHARS = 35


def _fit_names(names: List[str], count: int) -> str:
    """Name the uncompared metrics in the row cell, or count them if they cannot fit."""
    joined = ", ".join(names)
    if names and len(joined) <= _TIMELINE_CELL_CHARS:
        return joined
    if len(names) > 1 and len(f"{names[0]} +{len(names) - 1}") <= _TIMELINE_CELL_CHARS:
        return f"{names[0]} +{len(names) - 1}"
    return f"{count} metric(s) never checked"


def alert_timeline_to_svg(
    history: list,
    *,
    max_entries: int = 12,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a FairnessMonitor alert history as a vertical timeline.

    Parameters
    ----------
    history : list[WindowMetrics]
        A chronological list of monitoring snapshots.
    max_entries : int
        Maximum number of rows to display (default 12).
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # A history that is absent altogether is not an empty history, but neither
    # is it a crash: ``history[-max_entries:]`` raised TypeError on None and
    # took the caller's process down. It lands in the same empty-history branch
    # below, which says on the canvas that no window was checked.
    recent = list(history)[-max_entries:] if history else []

    entries: List[Dict[str, Any]] = []
    n_alerted = 0
    n_unmonitored = 0
    # BGL5 (2026-09-27). ``checked = len(alerts) > 0`` below is a WINDOW-level
    # two-state test over a PER-METRIC record, so a window that applied a
    # guardrail to SOME of its metrics entered n_clean whole. MEASURED end to end
    # through FairnessMonitor(config=FairnessMonitorConfig()).update_and_check()
    # on a frame with two protected columns, one fully measured and one carrying
    # an 'Other' stratum of 7 rows below the default min_samples=30:
    #   metrics : 4 keys, two of them NaN
    #   alerts  : 2 keys, both False
    # and alert_timeline_to_svg([wm]) rendered "1 monitoring events / 0 with
    # alerts / 1 clean", TOTAL EVENTS 1 / WITH ALERTS 0 / CLEAN 1, the row's
    # STATUS pill "OK" in emerald, and published "0 of 1 recent monitoring
    # windows raised alerts (1 clean)." at severity info: the same artifact a
    # window watched end to end produces. monitoring_dashboard_to_svg, reading
    # the SAME object, already got it right ("2 of 4 metrics checked against a
    # guardrail", severity medium), so the split it keeps between "computed" and
    # "compared to a guardrail" is the split read here. After the fix the same
    # input renders CLEAN 0, the row reads "2 of 4 checked" with no OK pill, and
    # the finding carries "2 metric(s) were never compared to a guardrail and are
    # not covered by this finding." at severity medium.
    n_partial = 0
    n_partial_clean = 0
    n_metrics_unchecked = 0
    # The strata min_samples kept out of every rate behind these windows, unioned
    # over the history in the order they were met. This timeline has no rates
    # table to name them in, so they travel to the reader through
    # explain._excluded_clause, which build_explanation appends to the finding.
    # See _window_excluded_groups for the measurement.
    excluded_display: List[str] = []
    # Every metric name this history left uncompared, in the order met, so the
    # accessible layer can NAME them instead of counting them.
    unchecked_names: List[str] = []
    for wm in recent:
        # CRITICAL, do not go back to ``wm.alerts.items()``. A window whose
        # alerts record is absent or None raised AttributeError there, so ONE
        # partial window took down the whole timeline, including the eleven
        # fully recorded windows beside it, and the caller got a traceback
        # instead of a chart. A crash is not the third state: it cannot be read,
        # it certifies nothing, and it teaches nobody which window was short.
        # An absent record is exactly the case ``checked`` already answers, so
        # such a window renders NOT MONITORED, the state this chart has carried
        # since the last wave, and enters no tally that implies it was watched.
        raw_alerts = getattr(wm, "alerts", None)
        alerts = raw_alerts if isinstance(raw_alerts, dict) else {}
        alerted_names = [k for k, v in alerts.items() if v]
        # CRITICAL, three states PER ROW, never two. ``any_alert`` was
        # ``len(alerted_names) > 0``, which is a two-state test, and it put every
        # window that did not raise an alert into the same green OK badge and the
        # same ``n_clean`` tally. ``WindowMetrics.alerts`` is documented as
        # "whether each metric triggered an alert", so it is the record of which
        # guardrails were APPLIED: an empty dict means none were, and a window
        # where nothing was checked cannot have raised an alert. Counting it as
        # clean reports a period as watched and healthy when nothing watched it.
        checked = len(alerts) > 0
        # Per METRIC, the same read monitoring_dashboard_to_svg makes: a metric
        # present in ``metrics`` with no non-None entry in ``alerts`` was never
        # compared to anything (WindowMetrics documents that absence as
        # could-not-check and forbids defaulting it to False). A window whose
        # ``metrics`` record is absent altogether cannot be split this way, and
        # falls back to the window-level test, which is what the empty and
        # control fixtures in tests/test_adapters_zero_and_empty.py supply.
        raw_metrics = getattr(wm, "metrics", None)
        metrics_w = raw_metrics if isinstance(raw_metrics, dict) else {}
        n_checked_w = sum(1 for name in metrics_w if name in alerts and alerts[name] is not None)
        # THE NAMES, not only the count. ``WindowMetrics.uncompared_metrics``
        # already supplies exactly this list and nothing on this canvas or in
        # its <desc> was reading it, so a reader was told "2 metric(s) never
        # checked" and could not tell WHICH. That distinction is the whole
        # difference between a metric the window genuinely could not compute
        # (disparate_impact with no denominator, say) and a REQUESTED metric
        # whose name has a one-letter typo, which produces zero warnings at
        # ingest and, until now, zero identifiable ones at reporting.
        #
        # Derived from the same two records the count is derived from rather
        # than read off the property, so the name list and the number cannot
        # disagree, and so a window object that predates the property (or a
        # SimpleNamespace fixture) still yields both.
        unchecked_w = [
            str(name) for name in metrics_w if not (name in alerts and alerts[name] is not None)
        ]
        n_unchecked_w = len(unchecked_w)
        for name in unchecked_w:
            if name not in unchecked_names:
                unchecked_names.append(name)
        n_metrics_unchecked += n_unchecked_w
        for name in _window_excluded_groups(wm):
            if name not in excluded_display:
                excluded_display.append(name)
        fully_checked = checked and not n_unchecked_w
        any_alert = len(alerted_names) > 0
        # Partial coverage is a statement about the EVIDENCE, so it is counted
        # whatever the window's verdict: a window that raised an alert AND never
        # compared two of its metrics to anything is a real finding over part of
        # the evidence, and the part it did not cover has to be said too. The
        # clean TALLY is a separate question, below: only a partial window that
        # did NOT alert was previously landing in n_clean.
        partly_checked = checked and bool(n_unchecked_w)
        if partly_checked:
            n_partial += 1
        if any_alert:
            n_alerted += 1
        elif not checked:
            n_unmonitored += 1
        elif partly_checked:
            # Neither alerted nor clean: part of this window was compared to a
            # guardrail and part of it was not, so it belongs to no tally that
            # implies the window was watched. It used to fall into n_clean.
            n_partial_clean += 1

        # ``strftime`` on anything that is not a datetime raised here too, and a
        # timestamp is not a measurement, so an unusable one is printed as the
        # phrase this row already uses for an absent one rather than killing the
        # render.
        ts = getattr(wm, "timestamp", None)
        try:
            ts_display = ts.strftime("%Y-%m-%d %H:%M") if ts else "n/a"
        except (AttributeError, TypeError, ValueError):
            ts_display = "n/a"
        # ``{{ e.sample_count }}`` is printed straight onto the canvas, so a
        # None reaches the reader as the literal string "None". A window that
        # did not report its size says so.
        sample_count = getattr(wm, "sample_count", None)
        entries.append(
            {
                "timestamp": ts_display,
                "sample_count": sample_count if sample_count is not None else "not reported",
                # None, not 0, for a window that applied no guardrail: "0 alerts"
                # is the same cell a genuinely clean window prints. A PARTIAL
                # window prints its coverage instead of a bare count, for the
                # same reason: "0" there is the clean cell.
                "n_alerts": len(alerted_names) if fully_checked else None,
                "n_alerts_display": (
                    str(len(alerted_names))
                    if fully_checked
                    else (
                        f"{n_checked_w} of {len(metrics_w)} checked" if checked else "not monitored"
                    )
                ),
                "flagged_names": (
                    ", ".join(alerted_names)
                    if alerted_names
                    else (
                        "n/a"
                        if fully_checked
                        # The NAMES when they fit the 35-character cell, and
                        # the count when they do not: a metric key here is
                        # routinely 31 characters, so the cell cannot promise
                        # to carry them and the <desc> is where they always
                        # land (see `unchecked_metric_names` below).
                        else (
                            _fit_names(unchecked_w, n_unchecked_w)
                            if checked
                            else "no metric was checked"
                        )
                    )
                ),
                "any_alert": any_alert,
                # The OK pill is a VERDICT about a window that was watched, so
                # it is available only to a window every one of whose metrics
                # was compared to a guardrail. A partial window takes the same
                # third state an unmonitored one takes: no pill, slate italic.
                "checked": fully_checked,
                "status_label": (
                    "ALERT"
                    if any_alert
                    else ("OK" if fully_checked else ("PARTIAL" if checked else "NOT MONITORED"))
                ),
            }
        )

    data = {
        "title": "FairnessMonitor: Alert Timeline",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_total": len(recent),
        "n_alerted": n_alerted,
        # A window that monitored nothing is neither alerted nor clean, so it
        # belongs to neither tally. It used to fall into this one by subtraction,
        # and so did a window monitored only in PART (see the note above the
        # loop): CLEAN read 1 for a window two of whose four metrics were never
        # compared to anything.
        "n_clean": len(recent) - n_alerted - n_unmonitored - n_partial_clean,
        "n_unmonitored": n_unmonitored,
        # Windows compared to a guardrail in part, and the metric count behind
        # them. Both are READ by explain._fr_alert_timeline, which names them in
        # the finding and floors the severity with them; the canvas states the
        # same shortfall per ROW ("2 of 4 checked", "PARTIAL"). No third tally
        # tile is claimed for them: a partly checked window is neither alerted
        # nor clean, and CLEAN no longer takes it by subtraction.
        "n_partial": n_partial,
        "n_unchecked_metrics": n_metrics_unchecked,
        # READ by explain._fr_alert_timeline, which names them in the finding.
        # The count alone cannot tell a metric that could not be computed from a
        # requested metric whose name is misspelled.
        "unchecked_metric_names": unchecked_names,
        "excluded_groups": excluded_display,
        "entries": entries,
    }

    # No window in the history means no window was ever checked, so "0 clean out
    # of 0" is not a clean bill of health. The same is true of a history whose
    # windows all applied a guardrail to nothing: the rows exist, but not one of
    # them measured anything, so the page as a whole certifies nothing.
    # See _mark_not_assessable.
    if not recent:
        _mark_not_assessable(
            data,
            "The monitoring history is empty, so no window was checked against a "
            "guardrail: there is no clean period here and no alerting period either.",
            "No monitoring window was checked",
        )
    elif n_unmonitored == len(recent):
        _mark_not_assessable(
            data,
            f"None of the {len(recent)} window(s) in this history applied a fairness "
            "guardrail to any metric, so no alert could fire and no window is clean.",
            "No window checked any metric",
        )

    data["explanation"] = explanation
    svg = render_svg("alert_timeline", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4.  TemporalFairnessAnalyzer  →  temporal_analysis.svg


def temporal_analysis_to_svg(
    analyzer,
    metric_names: Optional[List[str]] = None,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a TemporalFairnessAnalyzer as a trend & pattern report.

    Parameters
    ----------
    analyzer : TemporalFairnessAnalyzer
        An already-populated temporal analyzer instance.
    metric_names : list[str], optional
        Metrics to include.  If None, uses the first 3 available.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # Discover available metric names
    #
    # BGL S2, 2026-09-16. This looked for a "metric_name" COLUMN, which is the
    # shape of a long frame. ``TemporalFairnessAnalyzer.update_daily_metrics``
    # builds a WIDE one: ``to_dataframe()`` returns
    # ``['demographic_parity_difference', 'date']``, so the column never exists,
    # ``metric_names`` was unconditionally ``[]``, and the one-argument call
    # published in the API reference rendered:
    #
    #   "COULD NOT CHECK: nothing on this chart was assessed ... No metric
    #    history was available" (severity: MEDIUM), with a NOT CHECKED chip
    #
    # on an analyzer holding 60 days of history whose detect_trend() was
    # ('increasing', +0.0019) and whose detect_weekly_degradation() was
    # (True, 0.358). The SAME analyzer with metric_names passed by hand
    # rendered "1 showing weekly degradation" at severity HIGH. This is the
    # could-not-check running BACKWARDS: the chart-level guard is right, and a
    # real HIGH finding was starved into it. ``get_tracked_metrics()`` is the
    # analyzer's own answer to "what do you hold", so ask that first and keep
    # the column probe for any long-format analyzer that still has one.
    #
    # BGL S2b, 2026-09-17. Discovery no longer stops at three. The ``[:3]``
    # that used to sit on ``get_tracked_metrics()`` narrowed the set BEFORE
    # anything was examined, so an analyzer holding a real HIGH weekly
    # degradation on its fifth metric rendered "Tracked 3 metrics; 0 showing
    # weekly degradation across 3 trend checks" - a clean bill of health, on a
    # page where the pre-fix code at least refused. Measured on five metrics
    # with only the fifth degrading (detect_weekly_degradation -> (True,
    # 0.45)): the finding was truncated out of the report entirely.
    #
    # The CARDS still stop at three, because the template lays them out three
    # across a 680px canvas and a fourth is drawn off the edge. The trend and
    # degradation tables are stacked rows whose heights are computed from the
    # list lengths, so those cover EVERY discovered metric and no finding can
    # be truncated away. ``n_metrics_discovered`` / ``n_metrics_not_carded``
    # travel on the data dict so the gap between the two is a number a reader
    # can see rather than an assumption.
    if metric_names is None:
        tracked = getattr(analyzer, "get_tracked_metrics", None)
        discovered: List[str] = []
        if callable(tracked):
            # A duck-typed analyzer may raise here, or answer something that
            # is not a sequence of names. Its failure must not take down the
            # public entry: fall through to the long-format probe below, which
            # is the same could-not-discover path an analyzer without the
            # method takes.
            try:
                discovered = [str(name) for name in tracked()]
            except Exception:
                logger.warning(
                    "temporal_analysis_to_svg: get_tracked_metrics() failed; "
                    "falling back to the long-format metric_name column.",
                    exc_info=True,
                )
                discovered = []
        if not discovered:
            df = analyzer.to_dataframe()
            if "metric_name" in df.columns:
                discovered = list(df["metric_name"].unique())
        metric_names = discovered

    n_metrics_discovered = len(metric_names)

    # -- metric summary cards (up to 3) --
    #
    # CRITICAL, per ROW. Until 2026-08-28 every statistic here came out of a
    # `.get` default: mean, std, min and max defaulted to 0.0, the trend to
    # "stable" and the slope to 0.0. ``get_metric_summary`` returns an EMPTY
    # dict for a metric the analyzer never tracked, so a name that was asked for
    # and never recorded rendered a card carrying five 0.000s, a +0.0000 slope
    # and a GREEN "STABLE" chip. Executed on this repo, one genuinely tracked
    # metric beside the untracked name "equalized_odds" produced exactly that
    # card, and the subtitle counted it: "2 metrics - temporal trend and pattern
    # analysis". Stable at zero is the calmest reading this page has, and it was
    # produced by absence.
    #
    # A metric with no history reported nothing: no statistic, no trend
    # direction, no slope, no colour, and no place in the tracked count.
    summaries: List[Dict[str, Any]] = []
    n_untracked = 0
    for name in metric_names[:3]:
        s = analyzer.get_metric_summary(name) or {}
        # An empty summary is the analyzer saying it holds no column for this
        # metric at all, which is a different claim from a metric whose stats
        # came out flat.
        tracked = bool(s)
        if not tracked:
            n_untracked += 1
        mean_val = s.get("mean")
        std_val = s.get("std")
        min_val = s.get("min")
        max_val = s.get("max")
        trend_dir = s.get("trend_direction") if tracked else None
        slope = s.get("trend_slope")

        # BGL5 (2026-09-27). "tracked" is not the same question as "a trend was
        # fitted", and this card read the statistic straight off
        # get_metric_summary. ``detect_trend`` answers ("not_assessed", nan) for a
        # metric the analyzer HOLDS but cannot fit a line through, and warns while
        # doing it ("has too little data to fit a trend (3 row(s), 5 needed), so
        # none was measured. Returning 'not_assessed', not 'stable'"). MEASURED on
        # a real TemporalFairnessAnalyzer with three daily readings
        # (get_metric_summary -> {'mean': 0.06, 'std': 0.01, 'min': 0.05,
        # 'max': 0.07, 'trend_direction': 'not_assessed', 'trend_slope': nan,
        # 'n_days': 3}): the card drew the chip STABLE in the pass green (#059669,
        # #41ba1b after the skin), "Slope/day +nan" because NaN is not None, and a
        # green range bar, while the trends TABLE below it read "slope: N/A/day /
        # 3 days / not_assessed" and the degradation table read "NOT CHECKED".
        # A genuinely flat 30-day metric produced a byte-identical chip, so the
        # calmest reading this card has was produced by absence. The adapter's own
        # comment at the trends loop says the slope "is now written as None, not
        # NaN, for an unfitted row": that was done for the table and the counts,
        # never for the card. Same test, on the same quantity, here:
        #   Slope/day  "+nan"   -> "not measured", in the neutral slate
        #   range bar  green    -> the neutral slate
        # The chip's own label and colour are computed inside
        # templates/temporal_analysis.svg, whose {% else %} arm maps every
        # direction other than increasing/decreasing to STABLE green; it reads
        # `trend_fitted` / `trend_label` / `trend_color` from here once that
        # template can be changed, and until then the chip is the one element on
        # the card that still says STABLE for an unfitted trend.
        trend_fitted = (
            bool(tracked) and trend_dir not in (None, "not_assessed") and not _is_nan(slope)
        )
        slope_measured = slope is not None and not _is_nan(slope)

        # Normalise range bar (200px wide)
        min_x = 0
        range_w = max(4, 200)
        range_color = "#3b82f6"
        if not tracked or not trend_fitted:
            # Neutral slate, never the green that reads as a steady metric.
            range_color = NOT_MEASURED_COLOR
        elif trend_dir == "increasing":
            range_color = "#dc2626"
        elif trend_dir == "decreasing":
            range_color = "#3b82f6"
        else:
            range_color = "#059669"

        summaries.append(
            {
                "name": name.replace("_", " ").title(),
                "tracked": tracked,
                "mean": mean_val,
                "std": std_val,
                "min_val": min_val,
                "max_val": max_val,
                "trend": trend_dir,
                # Whether a line was actually fitted, kept apart from "tracked"
                # because the statistics above ARE measured for an unfitted
                # metric and withholding them would be the reverse fabrication.
                "trend_fitted": trend_fitted,
                "trend_label": (
                    "NOT TRACKED"
                    if not tracked
                    else (
                        "NOT FITTED"
                        if not trend_fitted
                        else {
                            "increasing": "INCREASING",
                            "decreasing": "DECREASING",
                        }.get(str(trend_dir), "STABLE")
                    )
                ),
                "trend_color": (
                    NOT_MEASURED_COLOR
                    if not trend_fitted
                    else {"increasing": "#dc2626", "decreasing": "#3b82f6"}.get(
                        str(trend_dir), "#059669"
                    )
                ),
                "slope_display": f"{slope:+.4f}" if slope_measured else "not measured",
                "slope_color": (
                    _trend_slope_color(slope)
                    if slope_measured and slope is not None
                    else NOT_MEASURED_COLOR
                ),
                "min_x": min_x,
                "range_w": range_w,
                "range_color": range_color,
            }
        )

    # -- weekly pattern for first metric --
    weekly_data: List[Dict[str, Any]] = []
    weekly_metric = ""
    if metric_names:
        weekly_metric = metric_names[0].replace("_", " ").title()
        try:
            pattern = analyzer.detect_seasonal_pattern(metric_names[0], period=7)
            if pattern:
                max_bar = 90  # px
                vals = list(pattern.values())
                p_max = max(vals) if vals else 1.0
                p_min = min(vals) if vals else 0.0
                p_range = p_max - p_min if p_max > p_min else 0.01
                for slot, val in pattern.items():
                    normed = (val - p_min) / p_range
                    bar_h = max(4, int(normed * max_bar))
                    weekly_data.append(
                        {
                            "value": val,
                            "bar_h": bar_h,
                            "color": "#3b82f6",
                        }
                    )
        except Exception:
            pass

    # -- trends for all metrics --
    #
    # ``detect_trend`` answers ("stable", 0.0) for a metric it holds no column
    # for, and the table drew that as a green "stable" badge over "slope:
    # 0.0000/day, 0 days". A trend nobody fitted is not a flat trend, so a row
    # with no history at all is ungraded here: no direction, no slope, no chip
    # colour.
    # BGL S2b, 2026-09-17, second half of the same defect: "tracked" was used
    # as a proxy for "a trend was fitted", and they are different questions.
    # ``detect_trend`` answers ("not_assessed", nan) for a metric it HOLDS but
    # cannot fit a line through (fewer than five rows, or fewer than three
    # non-null values), so a metric with three days of history counted into
    # "across N trend checks" as a check that happened. Measured on one metric
    # with three days: direction "not_assessed", slope nan, and the sentence
    # read "Tracked 1 metrics; 0 showing weekly degradation across 1 trend
    # checks". The slope is now written as None, not NaN, for an unfitted row:
    # None is the value explain._fr_temporal._trend_graded withdraws on, and
    # "N/A" is what the template prints for it instead of "nan".
    trends: List[Dict[str, Any]] = []
    n_trends_graded = 0
    for name in metric_names:
        try:
            s = analyzer.get_metric_summary(name) or {}
            tracked = bool(s)
            direction, slope = analyzer.detect_trend(name) if tracked else (None, None)
            fitted = tracked and direction != "not_assessed" and not _is_nan(slope)
            if fitted:
                n_trends_graded += 1
            trends.append(
                {
                    "name": name.replace("_", " ").title(),
                    "tracked": tracked,
                    # Kept verbatim so the row says "not_assessed" rather than
                    # inventing a direction; the COUNT is what changes.
                    "direction": direction,
                    "fitted": fitted,
                    "slope": slope if fitted else None,
                    "n_days": s.get("n_days"),
                }
            )
        except Exception:
            pass

    # -- degradation checks --
    #
    # ``detect_weekly_degradation`` returns (False, nan) when it REFUSES to run:
    # fewer than fourteen days of history, or no column for the metric at all.
    # False is the branch that paints the green STABLE chip, so a check that
    # declined to run certified the metric as steady, with "worst day mean: nan"
    # printed beside it. The NaN is the tell, and it is the signal read here:
    # a real weekday mean is never NaN.
    #
    # This loop runs over EVERY tracked metric name, so it cannot assume a
    # family, and it deliberately passes no ``higher_is_better``: the analyzer
    # resolves the direction from the metric name and refuses to grade a name it
    # cannot place. Until 2026-09-10 that parameter DEFAULTED to False here, i.e.
    # every tracked metric was graded as a disparity magnitude. Measured on this
    # repo through this very function, on a ``disparate_impact_ratio`` where
    # HIGHER is better: a week at 0.30 with Fridays collapsing to 0.10 rendered
    # "DEGRADED / worst day mean: 0.300" (0.300 is the BEST weekday); a week at
    # 0.85 with Fridays at PERFECT 1.00 rendered "DEGRADED / worst day mean:
    # 1.000"; and a week at 0.15 with Fridays at 0.10, in which every day fails
    # the four-fifths rule, rendered "STABLE / worst day mean: 0.150". A metric
    # whose direction cannot be resolved now answers (None, nan) and lands in the
    # ungraded branch below, which is what NaN has always meant on this chart.
    degradations: List[Dict[str, Any]] = []
    n_degradations_ran = 0
    for name in metric_names:
        try:
            degraded, worst_day = analyzer.detect_weekly_degradation(name)
            ran = not _is_nan(worst_day)
            if ran:
                n_degradations_ran += 1
            degradations.append(
                {
                    "name": name.replace("_", " ").title(),
                    "ran": ran,
                    "degraded": bool(degraded) if ran else False,
                    "worst_day": worst_day if ran else None,
                }
            )
        except Exception:
            pass

    data: Dict[str, Any] = {
        "title": "TemporalFairnessAnalyzer: Trend Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metric_summaries": summaries,
        # The subtitle counts the cards, and a count reads as a measurement, so
        # the untracked ones are named in it rather than left to be assumed
        # measured. This is the decided headline rule (c): the ungraded count
        # goes on the CANVAS beside the headline, not only into the <desc>.
        "n_untracked": n_untracked,
        # Discovery is no longer capped, but the CARDS still are (three across
        # a 680px canvas). These two say how many metrics were found and how
        # many got no card, so the difference between the three cards and the
        # longer trend/degradation tables is stated rather than inferred.
        # Every discovered metric IS examined for a trend and for weekly
        # degradation, so nothing can be graded out of sight.
        "n_metrics_discovered": n_metrics_discovered,
        "n_metrics_not_carded": max(0, n_metrics_discovered - len(summaries)),
        "weekly_pattern": weekly_data[:7],
        "weekly_metric": weekly_metric,
        "trends": trends,
        "degradations": degradations,
    }

    # "Tracked 0 metrics; 0 showing weekly degradation across 0 trend checks"
    # reads as a period that was watched and stayed healthy. No metric was
    # tracked, so no trend was fitted and no degradation could be found. Note
    # the summaries test as well as metric_names: every per-metric call above is
    # wrapped in a bare except, so a name can survive with nothing computed for
    # it. See _mark_not_assessable.
    #
    # The tests are on MEASURED rows, not on the row lists: a run whose every
    # named metric was untracked builds three full-length lists out of nothing,
    # and the chart-level guard has to fire for it just as it does for the empty
    # analyzer. Otherwise the page's only remaining claim is the honest per-row
    # one, under a headline that still says a period was analysed.
    n_tracked = len(summaries) - n_untracked
    if not n_tracked and not n_trends_graded and not n_degradations_ran:
        _mark_not_assessable(
            data,
            "No metric history was available, so no trend was fitted, no seasonal "
            "pattern was extracted and no weekly degradation check was run.",
            "No metric was tracked over time",
        )

    data["explanation"] = explanation
    svg = render_svg("temporal_analysis", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
