"""
Adapters for fairness-metrics visualization SVG templates.

Each adapter transforms a FairnessAnalyzer report dict (from
``analyzer.get_report()``) into the flat data dict expected by
the corresponding Jinja2 SVG template.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .._triage import is_flag
from ..evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    is_ratio_metric,
    metric_direction,
)
from .engine import render_svg

# Re-usable constants

# Common suffixes to strip for shorter labels
_SUFFIX_STRIP = ["_difference", "_ratio", "_score", "_index"]
_WORD_ABBREV = {
    "demographic": "Demo.",
    "prediction": "Pred.",
    "predictive": "Pred.",
    "individual": "Indiv.",
    "coefficient": "Coeff.",
    "equalized": "Equal.",
    "calibration": "Calib.",
    "opportunity": "Opp.",
    "treatment": "Treat.",
    "between": "Betw.",
    "negative": "Neg.",
    "positive": "Pos.",
    "smoothed": "Smooth.",
    "balance": "Bal.",
}


def _shorten_metric_name(key: str, max_len: int = 16) -> str:
    """Convert a snake_case metric key into a short readable label."""
    name = key
    for suffix in _SUFFIX_STRIP:
        if name.endswith(suffix) and len(name) > len(suffix) + 3:
            name = name[: -len(suffix)]
            break
    words = name.split("_")
    label = " ".join(w.capitalize() for w in words)
    if len(label) <= max_len:
        return label
    # Apply abbreviations to longest words first
    words_titled = [w.capitalize() for w in words]
    for i, w in enumerate(words):
        if w in _WORD_ABBREV:
            words_titled[i] = _WORD_ABBREV[w]
    label = " ".join(words_titled)
    if len(label) > max_len:
        label = label[:max_len].rstrip() + "."
    return label


_METRIC_LABELS = {
    "demographic_parity_difference": "Demo. Parity",
    "demographic_parity_ratio": "Disparate Impact",
    "equal_opportunity_difference": "Equal Opp.",
    "equalized_odds_difference": "Equalized Odds",
    "predictive_parity_difference": "Pred. Parity",
    "calibration_difference": "Calibration",
    "mae_parity_difference": "MAE Parity",
    "rmse_parity_difference": "RMSE Parity",
    "mean_prediction_difference": "Mean Pred Diff",
    "treatment_equality_ratio": "Treatment Eq.",
    "balance_positive_class": "Balance Pos.",
    "balance_negative_class": "Balance Neg.",
    "theil_index": "Theil Index",
    "between_group_theil": "Between Theil",
    "coefficient_of_variation": "Coeff. of Var.",
    "consistency_score": "Consistency",
    "smoothed_edf": "Smoothed EDF",
    "individual_fairness_score": "Indiv. Fairness",
}

_DEFAULT_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "demographic_parity_ratio": 0.80,
    "equal_opportunity_difference": 0.10,
    "equalized_odds_difference": 0.15,
    "predictive_parity_difference": 0.10,
    "calibration_difference": 0.05,
    "mae_parity_difference": 0.10,
    "rmse_parity_difference": 0.10,
    "mean_prediction_difference": 0.10,
}

_GROUP_COLORS = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

# ── Three states, never two ─────────────────────────────────────────────────
# assessed-pass / assessed-fail / could-not-check. The third state is NOT a
# shade of the other two, so it never borrows their palette: not the green of a
# measured pass, not the red of a measured failure. An SVG is an export format,
# it leaves the building and outlives the run that produced it, so a chart that
# measured nothing must SAY it measured nothing on its own canvas.
COULD_NOT_CHECK = "could_not_check"
PASS = "pass"
FAIL = "fail"

_UNKNOWN_COLOR = "#64748b"  # slate-500
_UNKNOWN_BG = "#f1f5f9"  # slate-100
_UNKNOWN_BORDER = "#cbd5e1"
_UNKNOWN_ICON = "·"
_UNKNOWN_STATUS = "NO DATA"
NOT_ASSESSABLE_TEXT = "NOT ASSESSABLE"


def _to_finite_float(value: Any) -> Optional[float]:
    """Coerce to a real, finite float, or None. NaN, inf and None all give None.

    READINESS-6, 2026-09-10. A BOOL is refused, which the sentence above always
    claimed and the code never did: ``float(True)`` is 1.0, a finite number that
    passes every check here. numpy's ``np.bool_`` is the half that gets missed,
    because it is not a Python bool, so a boolean column read out of a DataFrame
    walked straight through. A numeric STRING is still accepted, deliberately:
    these rows arrive from JSON and CSV where "0.5" is a real measurement that
    was serialised, and refusing it would discard evidence.
    """
    if is_flag(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_finite(value: Any) -> bool:
    """True only for a real, finite number. NaN, inf and None are all False."""
    return _to_finite_float(value) is not None


# Advance width in px, at font-size 10, of every character the subtitle line
# carries. MEASURED on 2026-08-27 by rasterising the template's own font stack
# with rsvg-convert and reading the ink extent of 30 repeats of each character,
# then rounding up to 0.05. It is not an em fraction guessed from the font size.
_SUBTITLE_CHAR_PX: Dict[str, float] = {
    c: w
    for group, w in (
        ("ijl", 2.45),
        ("I", 2.70),
        (" ", 2.85),
        (".,:", 2.90),
        ("'", 3.00),
        (";", 3.05),
        ("t", 3.45),
        ("/", 3.65),
        ("()[]{}", 3.70),
        ("f", 3.75),
        ("r", 3.95),
        ("1", 4.10),
        ("-", 4.60),
        ("s", 5.30),
        ("xkz", 5.55),
        ("aLJ7cvy", 5.80),
        ("enFhu5", 5.95),
        ("oE", 6.05),
        ("2bdpqg38", 6.20),
        ("69Z", 6.30),
        ("0PSRT4", 6.50),
        ("B=", 6.65),
        ("KYXAV", 6.90),
        ("DCHUGN", 7.55),
        ("OQ", 7.65),
        ("w", 8.20),
        ("m", 8.80),
        ("M", 9.05),
        ("%W", 9.90),
    )
    for c in group
}
# A character not in the table (an ellipsis, an accented letter, a rune) is
# charged the widest measured advance, so an unknown glyph can only ever be
# OVER-budgeted and never overflow the line.
_SUBTITLE_CHAR_FALLBACK_PX = 9.90
# The table lands within about 2 percent of the rasterised width; the margin
# absorbs that residual so an estimate is never an underrun.
_SUBTITLE_WIDTH_SAFETY = 1.03
# x=36 to the 680px canvas edge, the line every subtitle is drawn on.
_SUBTITLE_LINE_PX = 644.0


def _subtitle_px(text: str) -> float:
    """Estimated rendered width of *text* on the 10px subtitle line."""
    return _SUBTITLE_WIDTH_SAFETY * sum(
        _SUBTITLE_CHAR_PX.get(c, _SUBTITLE_CHAR_FALLBACK_PX) for c in str(text)
    )


def _fit_subtitle(text: str, max_px: float = _SUBTITLE_LINE_PX) -> str:
    """Trim a subtitle to what actually fits the 644px subtitle line at 10px.

    A NOT ASSESSABLE reason is long, and a subtitle that runs off the right edge
    of the canvas hides the very thing it exists to say. The full text always
    remains in the explanation panel, which wraps.

    The budget is WIDTH, not a character count. A count is the wrong unit twice
    over: 116 characters of this prose measure about 560px and threw away 80px
    of a line that was there all along, which silently truncated the healthy
    radar subtitle at "dots outside it…" and dropped the sentence telling the
    reader what the dots and labels mean; and 116 characters of capitals or
    "W" measure well over 644px and overflow the canvas the count was meant to
    protect. Both readings were wrong in the direction that hides text.
    """
    text = str(text)
    if _subtitle_px(text) <= max_px:
        return text
    ellipsis = "…"
    budget = max_px - _subtitle_px(ellipsis)
    cut = ""
    used = 0.0
    for ch in text:
        w = _subtitle_px(ch)
        if used + w > budget:
            break
        cut += ch
        used += w
    cut = cut.rstrip()
    space = cut.rfind(" ")
    if space > 0 and _subtitle_px(cut[:space]) > budget / 2:
        cut = cut[:space]
    return cut + ellipsis


def _excluded_groups(report_dict: dict) -> List[str]:
    """Display names of groups dropped BEFORE any comparison ran.

    A group below ``min_group_size`` is removed from every disparity metric, so
    it is precisely the group the chart says nothing about. Naming it on the
    canvas is the difference between "we checked and it was fine" and "this
    stratum was never looked at".

    THREE MECHANISMS, NOT TWO (2026-09-30). ``groups_dropped`` is a THIRD way a
    group leaves the comparison and it is not a smaller version of the other two:
    the group was not too small and was not short of evidence, it lost EVERY row
    to missing-data cleaning and is absent from the report altogether, so it
    appears in neither ``insufficient_evidence_groups`` nor ``invalid_groups``.
    The evaluation analyser started publishing it (with ``n_groups_before`` and
    ``n_groups_after``) in the same wave that fixed its staleness, and this
    function read the other two keys only.

    Executed end to end on the real engine, 40 'a' + 40 'b' + 40 'c' with every
    one of c's predictions missing and ``missing_strategy='exclude'``:
    ``data_info`` correctly said ``n_groups_before 3, n_groups_after 2,
    groups_dropped ['c']`` and ``_excluded_groups`` returned ``[]``, so
    ``radar_chart_to_svg`` published ``<desc>Fairness Radar: Overall fairness
    status: Fair. (severity: INFO)`` and ``disparity_heatmap_to_svg``
    ``<desc>Disparity Heatmap: Low disparity across groups across 2 group(s) and
    5 metric(s). (severity: INFO)``. A perfect all-clear at the benign severity,
    produced by losing the only group that could have made it unequal, and its
    name was nowhere on any of the six canvases. That is the incident this
    function exists to prevent, through the door the producer had just named.

    UNION, not a third fallback. The ``invalid_groups`` arm stays a fallback
    under ``insufficient_evidence_groups`` because those two describe the SAME
    population at different richness. ``groups_dropped`` describes a different
    population, so it is added, de-duplicated by name so a group that somehow
    appears in both is still named once.
    """
    assessment = report_dict.get("assessment") or {}
    info = report_dict.get("data_info") or {}
    sizes = info.get("group_sizes") or {}

    names: List[str] = []
    seen: set = set()

    def _add(label: str, key: Any) -> None:
        if key in seen:
            return
        seen.add(key)
        names.append(label)

    for entry in assessment.get("insufficient_evidence_groups") or []:
        if isinstance(entry, dict) and entry.get("group") is not None:
            n = entry.get("n")
            group = entry["group"]
            _add(f"{group} (n={n})" if n is not None else str(group), str(group))
    if not names:
        for name in info.get("invalid_groups") or []:
            n = sizes.get(name)
            _add(f"{name} (n={n})" if n is not None else str(name), str(name))
    # "0 rows left" rather than "(n=0)": n=0 is the shape the other two arms use
    # for a MEASURED size, and this group has no size in this report at all. The
    # phrase is kept short because the subtitle it lands in is trimmed to 644px.
    for name in info.get("groups_dropped") or []:
        if name is None:
            continue
        _add(f"{name} (0 rows left)", str(name))
    return names


def report_assessability(report_dict: dict) -> Dict[str, Any]:
    """Resolve a report's assessability into the third state the charts render.

    Returns ``{"assessable": bool, "reason": str, "excluded": [str]}``.

    ``assessment["assessable"]`` is the engine's own flag and wins outright. A
    report from a producer that predates the flag still carries the engine's
    other "could not check" marker, ``fairness_score is None``, so that is
    honoured too. A bare data dict with no assessment block at all claims
    nothing either way and is treated as assessable: inventing a failure for a
    hand-built dict would be the mirror-image lie of inventing a pass.
    """
    assessment = report_dict.get("assessment")
    excluded = _excluded_groups(report_dict)

    assessable = True
    if isinstance(assessment, dict):
        flag = assessment.get("assessable")
        if flag is not None:
            assessable = bool(flag)
        elif "fairness_score" in assessment and assessment.get("fairness_score") is None:
            assessable = False

    if assessable:
        return {"assessable": True, "reason": "", "excluded": excluded}

    info = report_dict.get("data_info") or {}
    valid = info.get("valid_groups")
    parts = [f"{NOT_ASSESSABLE_TEXT}:"]
    if isinstance(valid, list):
        parts.append(
            f"{len(valid)} comparable group(s) after filtering (2 are needed), "
            f"so no between-group comparison was performed."
        )
    else:
        parts.append("no between-group comparison was performed on this data.")
    if excluded:
        parts.append(f"Excluded: {', '.join(excluded)}.")
    parts.append("This chart certifies nothing.")
    return {"assessable": False, "reason": " ".join(parts), "excluded": excluded}


def _subtitle_with_exclusions(subtitle: str, assess: Dict[str, Any]) -> str:
    """Name the groups dropped before any comparison, then fit the line.

    Returns the fitted subtitle unchanged when nothing was dropped, so a healthy
    chart's canvas stays byte-identical.

    WHY EVERY SITE GOES THROUGH ONE WRAPPER. ``report_assessability`` resolves
    the dropped groups and every adapter here already passes them on as
    ``excluded_groups``, which NO template reads (grep the templates: zero
    hits). So the only surface that ever named them was the
    ``not_assessable_reason``, i.e. only on a run that measured nothing at all.
    Measured on a 130-row report, A=60 and B=60 selected identically while C=10
    was never selected at all and fell below min_group_size=30: the radar drew
    "✓ Fair", the bar chart "PASSED 5 / 5" and "FAIR RATE 100%", the heatmap
    "✓ Low disparity across groups", the effect-size tiles "|d|=0.00
    (Negligible)", and five of the six canvases here mentioned C nowhere at
    all. A perfect all-clear, produced by removing the only group that could
    have made it unequal. Wiring the clause in per adapter is what left five of the six
    sites out, so the wrapper sits above the choice instead.

    The clause goes FIRST for the reason the sites around it already give: the
    line is trimmed to 644px and the generic legend prose is the half that can
    afford to be cut, so an appended caveat would be the part that disappears
    (the radar's own legend already measures 595px of the budget).

    At most three names are spelled out and the rest are counted, because the
    clause has to fit on the line it is protecting. It is skipped when the
    subtitle already names the first dropped group, so a site that discloses in
    its own words (group_comparison) is not made to say it twice.
    """
    excluded = assess.get("excluded") or []
    if excluded and str(excluded[0]) not in subtitle:
        shown = ", ".join(str(g) for g in excluded[:3])
        if len(excluded) > 3:
            shown += f" and {len(excluded) - 3} more"
        subtitle = f"Not shown (insufficient evidence): {shown}. {subtitle}"
    return _fit_subtitle(subtitle)


def _metric_state(key: str, value: Any, threshold: Any) -> Tuple[str, str]:
    """Grade one metric against one threshold into one of THREE states.

    Returns ``(state, reason)`` where state is :data:`PASS`, :data:`FAIL` or
    :data:`COULD_NOT_CHECK`.

    Two traps live here, both of which used to render green.

    1. ``threshold is None`` means no bound was configured. The old
       ``passed = abs(value) <= threshold if threshold else True`` answered
       "pass" for that, i.e. an unchecked metric was drawn as a fair one.
    2. ``if threshold`` is FALSY for a threshold of ``0.0``, a zero-tolerance
       policy. Verified against the engine: with every threshold set to 0.0 the
       report returned fairness_score 0.0 with all five metrics FAIL while the
       SVG rendered four of five as FAIR. The test is ``is not None``, never
       truthiness.

    3. A threshold that CANNOT BE BREACHED grades nothing, and answering "pass"
       for it is the same lie in a third costume. On a higher-is-better metric
       the threshold is a required MINIMUM, and ratios are non-negative, so a
       minimum of 0.0 is satisfied by every possible value including the worst
       one: ``_metric_state('demographic_parity_ratio', 0.00, 0.0)`` returned
       ``('pass', '')``, and both the bar chart and the dashboard rendered a
       green FAIR badge for a protected group that is never selected at all.
       The MIRROR case is not degenerate and must keep working: 0.0 on a
       lower-is-better metric is a real zero-tolerance policy, and ``abs(value)``
       can exceed it (trap 2 above).

    Direction, NaN and unknown-metric handling are delegated to the shared
    ``_metric_direction.check_threshold`` so this renderer cannot drift away
    from the deployment gate's answer for the same metric.
    """
    if not _is_finite(value):
        return (
            COULD_NOT_CHECK,
            f"{key} could not be measured on this data (undefined/NaN)",
        )
    if threshold is None:
        return (
            COULD_NOT_CHECK,
            f"{key} has no configured threshold, so it was not checked",
        )
    if not _is_finite(threshold):
        return (COULD_NOT_CHECK, f"{key} has a non-numeric threshold, so it was not checked")

    # The degenerate bound, before the comparison that cannot fail it. The same
    # hole is still open in the shared check_threshold (and therefore in the
    # deployment gate), which is why the guard is applied here rather than
    # assumed upstream.
    if metric_direction(key) is MetricDirection.HIGHER_IS_BETTER and float(threshold) <= 0.0:
        return (
            COULD_NOT_CHECK,
            f"{key} has a required minimum of {float(threshold):.4f}, which no value "
            f"can fall below, so the bound was never applied and nothing was graded",
        )

    outcome, message = check_threshold(key, float(value), float(threshold))
    if outcome is ThresholdOutcome.PASS:
        return (PASS, "")
    if outcome is ThresholdOutcome.FAIL:
        return (FAIL, message)
    return (COULD_NOT_CHECK, message)


def _state_style(state: str) -> Dict[str, str]:
    """Badge colours/labels for a metric state. Could-not-check is never green."""
    if state == PASS:
        return {
            "status": "FAIR",
            "color": "#059669",
            "bg": "#d1fae5",
            "icon": "✓",
        }
    if state == FAIL:
        return {
            "status": "FAIL",
            "color": "#dc2626",
            "bg": "#fee2e2",
            "icon": "✗",
        }
    return {
        "status": _UNKNOWN_STATUS,
        "color": _UNKNOWN_COLOR,
        "bg": _UNKNOWN_BG,
        "icon": _UNKNOWN_ICON,
    }


def _disparity_magnitude(key: str, value: Any) -> Optional[float]:
    """Distance from parity for one metric, on the common 0 == fair scale.

    Returns ``None`` when the value cannot be placed on that scale at all, and a
    caller MUST treat ``None`` as could-not-check rather than as 0.0.

    The direction itself is never decided here: it is read from the shared
    :func:`..evaluation.vfairness_metrics._metric_direction.metric_direction`,
    so this renderer cannot drift away from the deployment gate's answer for the
    same metric name.

    * lower-is-better (a violation magnitude): the magnitude IS ``abs(value)``.
    * a parity ratio (higher is better, 1.0 == parity): the symmetric shortfall
      from parity, ``1 - min(r, 1/r)``. ``r <= 0`` means the protected group was
      never selected, the worst outcome the metric can express, so it maps to a
      full 1.0 gap. Banding ``abs(r)`` instead put that WORST case in the
      "< 0.1" bucket and painted a green "Low disparity across groups" badge,
      while perfect parity (1.0) was banded as high.
    * anything else, including a higher-is-better metric that is not a ratio
      (a performance floor such as ``worst_group_accuracy``, whose distance from
      a floor is not a between-group disparity): ``None``. Fail closed.
    """
    v = _to_finite_float(value)
    if v is None:
        return None
    direction = metric_direction(key)
    if direction is MetricDirection.LOWER_IS_BETTER:
        return abs(v)
    if direction is MetricDirection.HIGHER_IS_BETTER and is_ratio_metric(key):
        if v <= 0:
            return 1.0
        return 1.0 - min(v, 1.0 / v)
    return None


# Radar fairness axis: every metric is plotted as a fairness score in [0, 1] where
# 1.0 is fully fair (the outer rim) and 0.0 is maximally unfair (the centre), so a
# fair model draws a large round shape and an unfair one caves inward toward the
# centre. Each metric's own threshold maps to this single fraction, so the pass
# line is one clean reference ring instead of a per-axis jagged polygon. A point
# beyond the ring passes; a point inside it fails.
_THRESHOLD_RING_FRACTION = 0.5


def _fairness_score_from_gap(gap: float, threshold: float) -> float:
    """Map a non-negative distance-from-fair ``gap`` to a radius fraction in [0, 1].

    ``gap == 0`` (perfectly fair) maps to 1.0 (the rim); ``gap == threshold`` maps
    to the threshold ring; ``gap >= 2 * threshold`` maps to 0.0 (the centre). The
    two segments are linear, so the threshold always lands on the same ring for
    every metric regardless of its natural scale.
    """
    t = max(threshold, 1e-6)
    if gap <= t:
        # Rim (1.0) down to the threshold ring.
        return 1.0 - (gap / t) * (1.0 - _THRESHOLD_RING_FRACTION)
    # Threshold ring down to the centre (0.0 at twice the threshold), clamped.
    return max(0.0, _THRESHOLD_RING_FRACTION * (1.0 - (gap - t) / t))


def _fairness_radius_fraction(key: str, value: float, threshold: float) -> Optional[float]:
    """Fairness radius fraction for a metric value, honouring metric direction.

    Returns ``None`` when the metric cannot be placed on this axis at all, and a
    caller MUST leave it off the chart rather than plot it anywhere: on this
    canvas the radius MEANS fairness, so both ends of it are a verdict.

    Difference metrics are lower-is-better (0.0 == parity), so the gap is
    ``abs(value)``. Parity ratios are higher-is-better with 1.0 == parity, so
    the gap is the symmetric shortfall from parity ``1 - min(r, 1/r)`` and the
    threshold (a lower bound such as the four-fifths 0.80) is converted to the
    matching gap ``1 - threshold``.

    The gap itself comes from :func:`_disparity_magnitude`, i.e. from the shared
    ``_metric_direction`` helper. This function used to test
    ``key.endswith("_ratio")`` on its own. That is an EXACT suffix test, so it
    is not the cali-BRATIO-n substring bug, but it is the same class one level
    down: it misses ``disparate_impact``, which IS a parity ratio and does not
    end in "_ratio". A ``disparate_impact`` of 0.00 (the protected group is
    never selected) therefore took the difference branch, produced a gap of
    ``abs(0.0) = 0.0``, and was plotted at radius fraction 1.0, the outer rim,
    which on this chart means perfectly fair.
    """
    gap = _disparity_magnitude(key, value)
    if gap is None:
        return None
    if is_ratio_metric(key):
        # The bound is a floor (0.80), so the tolerated GAP is 1 - 0.80.
        return _fairness_score_from_gap(gap, 1.0 - threshold)
    return _fairness_score_from_gap(gap, threshold)


# ── Absence in a protected attribute, before it becomes a group name ────────
#
# ONE DEFINITION FOR THE WHOLE RENDERING PACKAGE. It lives beside
# ``_excluded_groups`` and ``report_assessability`` because this module already
# owns the vocabulary for "a group this chart says nothing about", and
# ``adapters.py`` and ``adapters_calibration.py`` import it from here rather
# than restating it: three copies of "this level is the absence of a value" is
# exactly the shape that drifts, and the ROW FLOOR for two of the same charts
# already drifted once (ten rows against one), which is why
# MIN_ROWS_PER_GROUP_FOR_CALIBRATION now lives in a third module.
#
# THE TWIN. ``post_processing/calibration/visualization.py`` is the matplotlib
# rendering of the group-calibration chart and it grew these two helpers on
# 2026-09-30 (G03). The rendering package cannot import them: the
# ``vfairness.post_processing`` package imports matplotlib at package-import
# time, and the whole point of the SVG renderer is that it needs jinja2 and
# nothing else. So the definitions are restated here and
# ``test_bgl_grade1_g11_render_monitor_regularizers.py::
# test_the_svg_and_matplotlib_twins_agree_about_absence`` asserts the two agree
# level for level, so they cannot drift apart silently.
#
# The set of literals is the same one for the same reason: the two string
# spellings are what an ordinary CSV read gives you, and ``pd.isna`` does not
# see them.
_ABSENT_LEVEL_LITERALS = frozenset({"", "None", "nan", "NaN", "<NA>", "NaT"})


def _is_missing_level(value: Any) -> bool:
    """Whether one level of a protected attribute IS the absence of a value.

    Absence reaches a grouping through at least six doors (``None``, float NaN,
    the blank string, ``pd.NA``, ``pd.NaT`` and the literal string "None"), and
    ``str(x)`` on any of them MINTS a demographic group name.

    MEASURED on this tree, 2026-09-30, before this guard existed. With 200 rows
    labelled "m" and 200 rows whose protected attribute was ``None``,
    ``group_calibration_to_svg`` published "Calibration disparity 0.012 across 2
    groups; worst m vs best None" at severity MEDIUM: a demographic group named
    after the absence of a demographic label, ranked as the best calibrated of
    the two. With ``pd.NA`` instead the chart did not render at all
    (``TypeError: boolean value of NA is ambiguous``), and with a float NaN or a
    ``None`` beside a string it died in ``sorted()``: three different outcomes
    for one ordinary column with a missing cell.
    """
    if isinstance(value, str):
        return value.strip() in _ABSENT_LEVEL_LITERALS
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (TypeError, ValueError, ImportError):
        return False


def _present_levels(attr) -> List[Any]:
    """The distinct PRESENT levels of a protected attribute.

    ``np.unique`` and ``sorted(set(...))`` both SORT, and on an object array
    holding a float NaN or a ``None`` beside strings the sort raises
    ``TypeError: '<' not supported between instances of 'str' and 'NoneType'``,
    a bare crash on a column with one missing cell. ``pd.unique`` preserves
    first-appearance order and compares nothing; the result is sorted afterwards
    only when every level is sortable, so the legend order and therefore the
    rendered bytes are unchanged for the homogeneous attributes that already
    worked.
    """
    import numpy as np
    import pandas as pd

    levels = [
        lv for lv in pd.unique(np.asarray(attr, dtype=object).ravel()) if not _is_missing_level(lv)
    ]
    try:
        return sorted(levels)
    except TypeError:
        return levels


# 1.  Radar / Spider Chart


def radar_chart_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """Render a radar chart of fairness metrics as SVG.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``FairnessAnalyzer.get_report()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    metrics = report_dict.get("metrics", {})
    thresholds = report_dict.get("thresholds_used", _DEFAULT_THRESHOLDS)
    assess = report_assessability(report_dict)

    # Build axes list. A non-finite value (NaN: the metric was never computed)
    # has no radius, so it cannot be a vertex. Plotting it as frac 0.0 would put
    # it at the centre, which on this chart MEANS "maximally unfair", the
    # opposite lie to the one this fix removes.
    #
    # A metric with no KNOWN BETTER-DIRECTION is refused for the mirror reason:
    # both ends of this radius are verdicts, so there is no honest place to put
    # a value when we cannot tell which end is good. It is named in the subtitle
    # instead of being drawn.
    #
    # A metric with NO CONFIGURED THRESHOLD is refused for a third reason, and
    # this is the site the earlier waves walked past. The radius on this chart
    # is computed from the gap RELATIVE TO THE THRESHOLD (`_fairness_score_from_gap`
    # maps gap == threshold onto the pass ring), so the threshold decides where
    # the dot lands and therefore whether the reader sees a pass or a failure.
    # The old `_DEFAULT_THRESHOLDS.get(key, 0.5)` invented a 0.5 bound for every
    # metric this library has no default for, which is a tolerance nobody set:
    # a theil_index of 0.45 was plotted just inside the pass ring against a
    # bound that does not exist, and a consistency_score would have been graded
    # against the same borrowed number. `_metric_state`, two hundred lines
    # below in this same file, has answered COULD_NOT_CHECK for exactly this
    # input since wave 3; the radar kept the default. It is the same rule here:
    # no bound, no plotted verdict, and the metric is NAMED in the subtitle.
    n_unmeasured = sum(1 for k, v in metrics.items() if v is not None and not _is_finite(v))
    items: List[Tuple[str, Any, float]] = []
    undirected: List[str] = []
    unbounded: List[str] = []
    for key, value in metrics.items():
        if not _is_finite(value):
            continue
        thresh = _to_finite_float(thresholds.get(key, _DEFAULT_THRESHOLDS.get(key)))
        if thresh is None:
            unbounded.append(str(key))
            continue
        frac = _fairness_radius_fraction(key, value, thresh)
        if frac is None:
            undirected.append(str(key))
        else:
            items.append((key, value, frac))
    plottable = bool(items) and assess["assessable"]
    n = len(items)
    if n == 0:
        items = [("no_data", 0.0, 0.0)]
        n = 1

    # Fairness axis (fixed 0..1): the radius encodes a per-metric fairness score,
    # so the scale no longer depends on the raw values and every metric shares one
    # axis. Four faint concentric grid rings give depth; the meaningful reference
    # is the single threshold ring drawn at _THRESHOLD_RING_FRACTION (see the
    # template). Grid rings are unlabelled so their radii are not misread as raw
    # metric values, which the dots carry instead.
    rings = [{"fraction": ri / 4, "label": ""} for ri in range(1, 5)]

    cx, cy, r_max = 340, 362, 140
    axes = []
    data_dots = []
    data_pts = []

    for i, (key, value, frac) in enumerate(items):
        angle_deg = -90 + i * (360 / n)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Fairness radius (computed above, 1.0 = fair rim / 0.0 = unfair centre),
        # honouring the metric's direction. A fair model therefore reaches
        # outward (round shape); a failing metric caves toward the centre.

        # Spoke endpoint
        spoke_x = cx + int(r_max * cos_a)
        spoke_y = cy + int(r_max * sin_a)

        # Label position (beyond spoke with breathing room)
        label_x = cx + int((r_max + 36) * cos_a)
        label_y = cy + int((r_max + 36) * sin_a)

        # Text-anchor based on angle: left side = end, right side = start, top/bottom = middle
        if cos_a < -0.3:
            label_anchor = "end"
        elif cos_a > 0.3:
            label_anchor = "start"
        else:
            label_anchor = "middle"

        # Vertical nudge: push top labels up, bottom labels down
        if sin_a < -0.3:
            label_y -= 4
        elif sin_a > 0.3:
            label_y += 4

        # Data point position
        data_r = frac * r_max
        pt_x = cx + int(data_r * cos_a)
        pt_y = cy + int(data_r * sin_a)

        label = _METRIC_LABELS.get(key, key.replace("_", " ").title())

        axes.append(
            {
                "label": label[:18],
                "spoke_x": spoke_x,
                "spoke_y": spoke_y,
                "label_x": label_x,
                "label_y": label_y,
                "label_anchor": label_anchor,
            }
        )
        # Anchor for value labels (same logic as axis labels)
        if cos_a < -0.3:
            value_anchor = "end"
        elif cos_a > 0.3:
            value_anchor = "start"
        else:
            value_anchor = "middle"

        data_dots.append(
            {
                "x": pt_x,
                "y": pt_y,
                "value": value,
                "value_x": pt_x + int(16 * cos_a),
                "value_y": pt_y + int(16 * sin_a),
                "value_anchor": value_anchor,
            }
        )
        data_pts.append(f"{pt_x},{pt_y}")

    # Determine overall status. Bands 0.8 / 0.5 match the detailed fairness
    # report's FAIR (>=80) / MARGINAL (>=50) / UNFAIR scale, so the radar and
    # the report can never give contradictory verdicts for the same input
    # (previously 0.7/0.4 here disagreed for scores in [0.7, 0.8) and
    # [0.4, 0.5)).
    assessment = report_dict.get("assessment", {})
    # NO DEFAULT, in either the lookup or the coercion. 0.5 is not a neutral
    # placeholder here: it lands in the `>= 0.5` band and painted the amber
    # "Marginal" badge, a graded verdict about the whole model, for a report
    # that carried no fairness score at all and for one whose score was a
    # string. The badge is the single most-quoted thing on this canvas. A score
    # that was not reported now leaves `score` None, which takes the
    # could-not-check arm below, exactly as an unplottable chart does.
    score = _to_finite_float(assessment.get("fairness_score"))
    # Accept both scales: sibling adapters (detailed report, post-processing)
    # normalise via `raw * 100 if raw <= 1 else raw`; the radar works in 0-1,
    # so a 0-100 score (> 1) is scaled down by 100. Without this, a score of
    # 45/100 read as 45 >= 0.8 and always rendered "Fair".
    if score is not None and score > 1:
        score = score / 100.0
    if not plottable or score is None:
        # Third state. A run that compared no groups has no fairness status at
        # all, so it gets neither the green Fair badge nor the red Unfair one.
        status_color, status_bg, status_icon, status_text = (
            _UNKNOWN_COLOR,
            _UNKNOWN_BG,
            _UNKNOWN_ICON,
            NOT_ASSESSABLE_TEXT,
        )
    elif score >= 0.8:
        if undirected:
            # The green badge is a claim about the WHOLE chart, and a metric
            # that could not be placed on the axis is not on the chart at all.
            # Same rule as the heatmap's all-clear: one unbanded metric
            # withdraws it. A measured MARGINAL or UNFAIR verdict below is left
            # alone, because nothing about an unplotted metric makes a measured
            # breach less true.
            status_color, status_bg, status_icon, status_text = (
                _UNKNOWN_COLOR,
                _UNKNOWN_BG,
                _UNKNOWN_ICON,
                NOT_ASSESSABLE_TEXT,
            )
        else:
            status_color, status_bg, status_icon, status_text = "#059669", "#d1fae5", "✓", "Fair"
    elif score >= 0.5:
        status_color, status_bg, status_icon, status_text = "#f59e0b", "#fef3c7", "⚠", "Marginal"
    else:
        status_color, status_bg, status_icon, status_text = "#dc2626", "#fee2e2", "✗", "Unfair"

    polygon_color = status_color

    # Measured at 595px on the 644px subtitle line, so the sentence that tells
    # the reader what the dots and the labels mean survives to the canvas. The
    # longer phrasing this replaces measured 639px, which fit HEAD's untrimmed
    # line by 5px and nothing else: any honest width budget refuses it.
    subtitle = (
        "Distance from centre = fairness (rim fair, centre unfair). "
        "Dashed ring = threshold; dots outside it pass. Labels show raw values."
    )
    if not plottable:
        subtitle = assess["reason"] or (
            f"{NOT_ASSESSABLE_TEXT}: no metric on this chart could be measured, "
            f"so no polygon is drawn."
        )
        if n_unmeasured:
            subtitle += f" {n_unmeasured} metric(s) undefined."
    elif n_unmeasured:
        subtitle += f" {n_unmeasured} metric(s) could not be measured and are not plotted."
    if undirected:
        # Named, not silently dropped: an axis missing without explanation reads
        # as a chart that covered everything it was given.
        subtitle = (
            f"Not plotted (no known better-direction): {', '.join(undirected[:3])}. {subtitle}"
        )
    if unbounded:
        # Same rule, different cause. The radius is measured AGAINST the
        # threshold, so a metric with no configured bound has no honest radius,
        # and it is named rather than plotted against an invented 0.5.
        subtitle = f"Not plotted (no configured threshold): {', '.join(unbounded[:3])}. {subtitle}"
    if plottable and score is None:
        # The badge is withheld while the polygon stands. Say which of the two
        # happened, on the canvas, or the slate badge reads as a rendering fault.
        subtitle = f"Overall status withheld: this report carries no fairness score. {subtitle}"

    data: Dict[str, Any] = {
        "title": _attr_title("Fairness Metrics: Radar Chart", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "rings": rings,
        "axes": axes,
        # Suppressed, not zeroed: with nothing measurable there is no polygon and
        # no dot, so the reader cannot mistake an unmeasured run for a shape.
        "data_dots": data_dots if plottable else [],
        "data_points": " ".join(data_pts) if plottable else "",
        "not_assessable": not plottable,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "n_unmeasured": n_unmeasured,
        # The two populations this chart does NOT plot, carried out so a caller
        # or a test can count them without parsing the subtitle prose.
        "unbounded_metrics": unbounded,
        "n_unbounded": len(unbounded),
        "undirected_metrics": undirected,
        "threshold_ring_fraction": _THRESHOLD_RING_FRACTION,
        "polygon_color": polygon_color,
        "status_color": status_color,
        "status_bg": status_bg,
        "status_icon": status_icon,
        "status_text": status_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("radar_chart", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  Disparity Heatmap


# A cell is banded by how far its group falls SHORT of the column's reference
# group, on the four-fifths scale the rest of this module already speaks: a
# group receiving 80 percent of the reference rate is the last one that is not
# painted as a breach.
_CELL_SHORTFALL_GREEN = 0.10
_CELL_SHORTFALL_AMBER = 0.20


def _heatmap_cell_style(
    value: Any, col_min: Any = 0.0, col_max: Any = 1.0, neutral: bool = False
) -> Tuple[str, str]:
    """Return bg and text colours for one heatmap cell.

    The colour is the group's DISTANCE FROM THE REFERENCE group of its column,
    ``(reference - value) / reference`` where the reference is the best-served
    group (``col_max``), banded on the four-fifths scale. It is NOT the raw
    magnitude of the rate.

    That distinction is the whole fix. These columns hold per-group RATES
    (selection rate, TPR, precision), and a rate is not a disparity: the old
    code normalised the raw rate within the column and commented "Invert so that
    higher rates are red (higher disparity risk)", so on a Selection Rate column
    the group receiving 0 percent of the positive outcomes was painted GREEN and
    the group receiving 54 percent RED. Wave 2 fixed the summary badge above
    these cells and left the cells themselves inverted, which is worse than
    either state alone: the badge and the grid contradicted each other.

    Two inputs are refused rather than coloured, because neither can produce a
    distance-from-reference at all:

    * ``neutral``, set by the caller when there is no comparison to make (an
      unassessable report, or a single surviving group). The measured rate is
      still shown; only the colour claim is withdrawn.
    * a non-finite value or reference. A NaN rate used to reach the gradient,
      print the literal "nan" on the grid, and turn its column's min/max into
      NaN, which silently repainted every OTHER cell in that column too.

    A reference of 0 is refused for the same reason: with nobody selected there
    is no scale to be short of, so there is nothing to band.

    ``col_min`` is kept for call-signature compatibility and no longer takes
    part: the band is anchored on the reference, never on the column's low end.

    The three value parameters are annotated ``Any``, not ``float``, because
    refusing an absent or non-finite input is the FUNCTION'S JOB: the caller
    passes ``Optional[float]`` straight from ``_to_finite_float`` and a column
    whose rates were all unmeasurable has ``(None, None)`` for its range. A
    ``float`` annotation here contradicted the body, hid the ``None`` path from
    the reader, and was three real mypy arg-type errors at the call site.
    """
    if neutral:
        return _UNKNOWN_BG, _UNKNOWN_COLOR
    v = _to_finite_float(value)
    reference = _to_finite_float(col_max)
    if v is None or reference is None or reference <= 0:
        return _UNKNOWN_BG, _UNKNOWN_COLOR
    shortfall = max(0.0, min(1.0, (reference - v) / reference))
    if shortfall <= _CELL_SHORTFALL_GREEN:
        return "#d1fae5", "#065f46"  # green: at or near the reference
    if shortfall <= _CELL_SHORTFALL_AMBER:
        return "#fef9c3", "#713f12"  # amber: down to the four-fifths floor
    return "#fecaca", "#991b1b"  # red: below the four-fifths floor


# Maps metric function names to the per-group stat key that drives them
_METRIC_TO_GROUP_STAT: Dict[str, str] = {
    "demographic_parity_difference": "positive_rate",
    "demographic_parity_ratio": "positive_rate",
    "equal_opportunity_difference": "tpr",
    "equalized_odds_difference": "tpr",
    "predictive_parity_difference": "precision",
    "calibration_difference": "precision",
    "treatment_equality_ratio": "positive_rate",
}


def _attr_title(base: str, report_dict: dict) -> str:
    """Append the protected attribute name to an SVG title if available."""
    attr = report_dict.get("protected_attribute", "")
    if attr:
        return f"{base} [{attr.replace('_', ' ').title()}]"
    return base


def _is_numeric_label(label) -> bool:
    """Check if a group label looks like a bare number (e.g. 0, 1, 2.0)."""
    try:
        float(str(label))
        return True
    except (ValueError, TypeError):
        return False


def _heatmap_not_assessable_text(
    assess: Dict[str, Any],
    n_unmeasured: int,
    undirected: Optional[List[str]] = None,
) -> str:
    """One-line summary-badge text for a heatmap that could not band a disparity.

    The template renders ``summary_icon`` in front of this, so no icon here.

    ``undirected`` names the metrics whose better-direction the shared helper
    could not resolve. Those are unbanded evidence: they cannot be folded into
    a green all-clear about the whole grid, so they are named instead.
    """
    text = f"{NOT_ASSESSABLE_TEXT}: no disparity was banded"
    if assess["excluded"]:
        text += f"; excluded: {', '.join(assess['excluded'])}"
    elif undirected:
        text += f"; no known better-direction: {', '.join(undirected[:3])}"
    elif n_unmeasured > 0:
        text += f"; {n_unmeasured} metric(s) undefined"
    return text


def _disparity_heatmap_not_assessable(
    report_dict: dict,
    *,
    n_groups: int,
    n_metrics: int,
    explanation: Optional[str],
    save_path: Optional[str],
) -> str:
    """The third state for a grid with no row or no column to draw.

    Every field that could read as a finding is withheld: no cell, no colour
    gradient, no summary band. ``not_assessable`` is the same flag the banded
    path already sets, so the template's could-not-check branch and
    ``rendering.explain._not_assessable`` both fire from one signal and the
    canvas cannot disagree with the accessible ``<desc>``.
    """
    if n_groups == 0 and n_metrics == 0:
        reason = (
            "The report carried no group and no measured fairness metric, so there "
            "was no rate to compare and no disparity to band."
        )
        headline = "No group and no metric to compare"
    elif n_groups == 0:
        reason = (
            f"The report carried {n_metrics} metric(s) but no group, so no per-group "
            f"rate could be placed on the grid and no groups were compared."
        )
        headline = "No group to compare"
    else:
        reason = (
            f"The report carried {n_groups} group(s) but no measured fairness metric, "
            f"so there was no column to band and no disparity was computed."
        )
        headline = "No measured metric to band"

    data = {
        "title": _attr_title("Fairness Metrics: Disparity Heatmap", report_dict),
        # The full reason is repeated in the panel and in the explanation, both
        # of which wrap; the subtitle line does not, and one that runs off the
        # right edge hides the very thing it exists to say.
        "subtitle": _fit_subtitle(f"{NOT_ASSESSABLE_TEXT}: {reason}"),
        "not_assessable": True,
        "not_assessable_reason": reason,
        "na_headline": headline,
        "excluded_groups": [],
        "rows": [],
        "col_labels": [],
        "col_sublabels": [],
        "n_rows": 0,
        "n_cols": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "explanation": explanation,
    }
    svg = render_svg("disparity_heatmap", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def disparity_heatmap_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """Render a group x metric disparity heatmap as SVG.

    Each column shows a per-group rate (e.g. selection rate, TPR) that
    underlies the corresponding disparity metric. Colour intensity
    reflects the spread within each column, making group-level
    differences immediately visible.

    Notes
    -----
    CRITICAL, do not restore the early ``return ""``. Until 2026-08-27 a report
    with no group, or with no measured metric, returned a zero-length string,
    and a caller that passed ``save_path`` got a 0-byte .svg on disk and no
    error: the file opens as a broken image and the pipeline that wrote it
    reports success. A silent empty artifact is worse than either verdict,
    because it cannot even be read as could-not-check. The grid now renders the
    could-not-check state it already knows how to draw (``not_assessable``,
    which the template and ``rendering.explain`` both key on) instead of
    vanishing.
    """
    metrics = report_dict.get("metrics", {})
    group_stats = report_dict.get("group_stats", {})
    assess = report_assessability(report_dict)

    metric_keys = [k for k, v in metrics.items() if v is not None]
    raw_group_names = list(group_stats.keys())
    if not raw_group_names or not metric_keys:
        return _disparity_heatmap_not_assessable(
            report_dict,
            n_groups=len(raw_group_names),
            n_metrics=len(metric_keys),
            explanation=explanation,
            save_path=save_path,
        )

    # Make purely-numeric group labels descriptive by prefixing with attribute name
    attr_name = report_dict.get("protected_attribute", "")
    all_numeric = all(_is_numeric_label(g) for g in raw_group_names)
    if all_numeric and attr_name:
        short_attr = attr_name.replace("_", " ").title()
        group_names = [f"{short_attr} = {g}" for g in raw_group_names]
    else:
        group_names = [str(g) for g in raw_group_names]

    # Resolve which per-group stat to show for each metric column
    stat_keys = []
    display_keys = []  # metric keys we can actually visualise per-group
    for k in metric_keys:
        base = k.replace("_" + k.rsplit("_", 1)[-1], "") if "_" in k else k
        stat = _METRIC_TO_GROUP_STAT.get(k) or _METRIC_TO_GROUP_STAT.get(base)
        if stat is None:
            stat = "positive_rate"  # sensible default
        stat_keys.append(stat)
        display_keys.append(k)

    # Gather per-group values for each column, then find each column's reference
    # (best-served) group. An absent or unmeasurable rate stays None all the way
    # to the canvas: coercing it to 0.0 printed "0.00" for a rate nobody
    # measured, which on a Selection Rate column reads as "this group was never
    # selected", and it dragged the column's range down with it.
    col_values: List[List[Optional[float]]] = [[] for _ in display_keys]
    for group in raw_group_names:
        stats = group_stats[group]
        for ci, sk in enumerate(stat_keys):
            val = stats.get(sk, stats.get("positive_rate", stats.get("mean_prediction", None)))
            col_values[ci].append(_to_finite_float(val))

    col_ranges: List[Tuple[Optional[float], Optional[float]]] = []
    for cv in col_values:
        measured = [v for v in cv if v is not None]
        col_ranges.append((min(measured), max(measured)) if measured else (None, None))

    # Build rows with per-group rate values. The colour gradient is a claim about
    # SPREAD between groups, so it is withdrawn when there is no comparison to
    # make; the rates themselves are real and stay on the grid.
    neutral_cells = (not assess["assessable"]) or len(raw_group_names) < 2
    rows = []
    for gi, group in enumerate(group_names):
        cells = []
        for ci in range(len(display_keys)):
            val = col_values[ci][gi]
            cmin, cmax = col_ranges[ci]
            bg, text_color = _heatmap_cell_style(val, cmin, cmax, neutral=neutral_cells)
            cells.append({"value": val, "bg": bg, "text_color": text_color})
        rows.append({"label": str(group)[:18], "cells": cells})

    # Column labels: show the per-group stat name beneath the metric name
    col_labels = []
    for k, sk in zip(display_keys, stat_keys):
        metric_label = _METRIC_LABELS.get(k, _shorten_metric_name(k))
        col_labels.append(metric_label)

    # Summary. Only MEASURED disparities may band this badge: max() over NaN
    # returns NaN, and every `nan < x` test is False, so an all-NaN report fell
    # straight through to "High disparity detected", a measured-sounding
    # verdict on a comparison that never ran.
    #
    # And the magnitude is the distance from PARITY, not abs(value): banding
    # abs() treated every metric as lower-is-better, so a demographic_parity
    # RATIO of 0.00 (the protected group is never selected, the worst outcome
    # the metric can express) landed in the "< 0.1" bucket and rendered the
    # green "Low disparity across groups" badge, while 1.00 (perfect parity)
    # rendered "High disparity detected". Direction comes from the shared
    # _metric_direction helper, never from a local name test.
    measured = [v for v in metrics.values() if _is_finite(v)]
    magnitudes: List[float] = []
    undirected: List[str] = []
    for _k, _v in metrics.items():
        if not _is_finite(_v):
            continue
        _mag = _disparity_magnitude(_k, _v)
        if _mag is None:
            undirected.append(str(_k))
        else:
            magnitudes.append(_mag)
    max_disparity = max(magnitudes, default=0.0)
    # A green all-clear is a claim about the WHOLE grid, so it requires that
    # every finite metric on it was actually banded. One metric with no known
    # better-direction is enough to withdraw it: we cannot tell whether its
    # value is the good end or the bad end of its own scale.
    unbanded = (
        (not assess["assessable"]) or (not magnitudes) or (max_disparity < 0.1 and bool(undirected))
    )
    if unbanded:
        # Third state: neither the green all-clear nor the red alarm.
        s_bg, s_border, s_color, s_icon, s_text = (
            _UNKNOWN_BG,
            _UNKNOWN_BORDER,
            _UNKNOWN_COLOR,
            _UNKNOWN_ICON,
            _heatmap_not_assessable_text(
                assess, len(metric_keys) - len(measured), undirected=undirected
            ),
        )
    elif max_disparity < 0.1:
        s_bg, s_border, s_color, s_icon, s_text = (
            "#d1fae5",
            "#a7f3d0",
            "#065f46",
            "✓",
            "Low disparity across groups",
        )
    elif max_disparity < 0.2:
        s_bg, s_border, s_color, s_icon, s_text = (
            "#fef3c7",
            "#fde68a",
            "#92400e",
            "⚠",
            "Moderate disparity detected",
        )
    else:
        s_bg, s_border, s_color, s_icon, s_text = (
            "#fee2e2",
            "#fecaca",
            "#991b1b",
            "✗",
            "High disparity detected",
        )

    # Build stat label subtitle for each column (e.g. "Sel. Rate" under "Demo. Parity")
    _STAT_LABELS = {
        "positive_rate": "Sel. Rate",
        "tpr": "TPR",
        "fpr": "FPR",
        "precision": "Precision",
        "accuracy": "Accuracy",
        "base_rate": "Base Rate",
        "mean_prediction": "Mean Pred.",
    }
    col_sublabels = [_STAT_LABELS.get(sk, sk.replace("_", " ").title()[:12]) for sk in stat_keys]

    subtitle = (
        "Per-group rates underlying each fairness metric. Colour shows spread within each column."
    )
    if unbanded:
        subtitle = assess["reason"] or (
            f"{NOT_ASSESSABLE_TEXT}: no fairness metric on this grid could be banded"
            + (
                f" ({', '.join(undirected[:3])} has no known better-direction)."
                if undirected
                else "."
            )
        )
    elif undirected:
        # The band survives (something measurable did breach), but the grid must
        # still say which columns it does NOT cover. The caveat goes FIRST: the
        # subtitle is trimmed to what fits the canvas, and the generic sentence
        # is the half that can afford to be cut.
        subtitle = (
            f"Not banded (no known better-direction): {', '.join(undirected[:3])}. {subtitle}"
        )

    data = {
        "title": _attr_title("Fairness Metrics: Disparity Heatmap", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "not_assessable": unbanded,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "rows": rows,
        "col_labels": col_labels,
        "col_sublabels": col_sublabels,
        "n_rows": len(rows),
        "n_cols": len(col_labels),
        "summary_bg": s_bg,
        "summary_border": s_border,
        "summary_color": s_color,
        "summary_icon": s_icon,
        "summary_text": s_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("disparity_heatmap", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  Metrics Bar Chart


def metrics_bar_chart_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """Render fairness metric values as a bar chart SVG.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``FairnessAnalyzer.get_report()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    metrics = report_dict.get("metrics", {})
    thresholds = report_dict.get("thresholds_used", _DEFAULT_THRESHOLDS)
    assess = report_assessability(report_dict)
    bar_area_w = 450

    bars = []
    n_passed = 0
    n_assessed = 0
    n_unknown = 0
    for key, value in metrics.items():
        if value is None:
            continue
        label = _METRIC_LABELS.get(key, key.replace("_", " ").title())
        threshold = thresholds.get(key, _DEFAULT_THRESHOLDS.get(key))
        # Which scale the bar is drawn on is a direction question, so it is
        # asked of the shared helper and never of the name here. A local
        # `endswith("_ratio")` also answered False for `disparate_impact`, which
        # IS a ratio, so its bar was drawn on the difference scale (x4) beside a
        # threshold marker placed on the other one.
        is_ratio = is_ratio_metric(key)

        state, reason = _metric_state(key, value, threshold)
        # An unassessable REPORT overrides a per-metric pass: with no comparable
        # pair of groups the number is vacuous even when it is finite.
        if not assess["assessable"]:
            state = COULD_NOT_CHECK
            reason = assess["reason"]

        # ONE coercion, and the arithmetic below reads the COERCED number.
        # `_is_finite` accepts a numeric string on purpose (its docstring says
        # why: these rows arrive from JSON and CSV, where "0.5" is a real
        # measurement that was serialised, and refusing it discards evidence).
        # The arithmetic then used the RAW value, so the guard coerced while the
        # arithmetic did not. Measured 2026-09-30 on a real report whose metrics
        # dict had been stringified: `abs("0.5")` raised
        # ``TypeError: bad operand type for abs(): 'str'`` and took the whole
        # chart down, while `radar_chart_to_svg` and `group_comparison_to_svg`
        # (which coerce through `_to_finite_float` first) rendered the identical
        # report correctly. `"0.5" * 450` is worse than a raise: it is a 1350
        # character string, and it only failed later, in `max()`.
        num = _to_finite_float(value)
        thresh_num = _to_finite_float(threshold)
        if num is None:
            # No bar and no threshold marker: a value that does not exist must
            # not be drawn as a length a reader can compare.
            bar_w = 0
            display = "n/a"
            thresh_x = None
        elif is_ratio:
            bar_w = int(min(bar_area_w, max(4, num * bar_area_w)))
            display = f"{num:.2f}"
            thresh_x = int(thresh_num * bar_area_w) if thresh_num is not None else None
        else:
            bar_w = int(min(bar_area_w, max(4, abs(num) * bar_area_w * 4)))
            display = f"{num:.3f}"
            thresh_x = int(thresh_num * bar_area_w * 4) if thresh_num is not None else None

        if state == COULD_NOT_CHECK:
            # The legend calls this line "the fairness threshold", so drawing it
            # beside an unbanded bar shows a comparison that never ran. The
            # threshold is still printed in the row's reason text.
            thresh_x = None

        style = _state_style(state)
        if state == PASS:
            n_passed += 1
            n_assessed += 1
        elif state == FAIL:
            n_assessed += 1
        else:
            n_unknown += 1

        bars.append(
            {
                "label": label,
                "bar_w": bar_w,
                "display_value": display,
                "threshold_x": thresh_x,
                "color": style["color"],
                "status": style["status"],
                "status_bg": style["bg"],
                "status_color": style["color"],
                "status_icon": style["icon"],
                "state": state,
                "reason": reason,
            }
        )

    # PASSED and FAIR RATE describe a fraction of CHECKED metrics. With nothing
    # checked there is no fraction, and "0 / 0" plus "0%" reads as a measured
    # total failure just as "5 / 5" plus "100%" read as a measured pass. The
    # tiles are SUPPRESSED in that case, not zeroed.
    not_assessable = (not assess["assessable"]) or n_assessed == 0
    subtitle = "Dashed red line = fairness threshold. Green = fair, red = unfair."
    if not_assessable:
        subtitle = assess["reason"] or (
            f"{NOT_ASSESSABLE_TEXT}: not one metric could be checked, so no pass rate is reported."
        )
    elif n_unknown:
        subtitle += f" {n_unknown} metric(s) could not be checked and are excluded."

    data = {
        "title": _attr_title("Fairness Metrics: Bar Chart", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "bars": bars[:6],
        "n_passed": n_passed,
        "n_total": n_assessed,
        "n_unknown": n_unknown,
        "not_assessable": not_assessable,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("metrics_bar_chart", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4.  Group Comparison


def group_comparison_to_svg(
    report_dict: dict,
    *,
    metric: str = "positive_rate",
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render per-group metric comparison as SVG bars.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``FairnessAnalyzer.get_report()``.
    metric : str
        The group stat key to compare (e.g. ``'positive_rate'``).
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    CRITICAL, do not restore the ``, 0)`` default on the rate lookup. A group
    that reported no rate was drawn at 0, entered the max-min gap as its
    extreme, and was named as the gap's best-served party, so this chart
    rendered "Max disparity 0.970 between ZZ_BARE and Male" at severity HIGH for
    two groups that are 0.06 apart. A fabricated breach costs the same as a
    fabricated all-clear: someone is sent after a violation that is not there,
    and the tool loses its credit when they find that out.
    """
    group_stats = report_dict.get("group_stats", {})
    task_type = report_dict.get("task_type", "classification")
    assess = report_assessability(report_dict)

    bars = []
    values = []
    graded_pairs: List[Tuple[str, float]] = []
    ungraded_groups: List[str] = []
    for i, (group, stats) in enumerate(group_stats.items()):
        # An absent or unmeasurable rate stays None all the way to the canvas.
        # The precedent is disparity_heatmap above (see the note at its
        # col_values loop): coercing it to 0 printed "0%" for a rate nobody
        # measured, which on a selection-rate chart reads as "this group is
        # never selected", the worst finding the chart can express.
        #
        # Here the same default did something worse still. The 0 entered the
        # max-min gap as its extreme AND the sort that names the gap's operands,
        # so a report with two groups 0.06 apart rendered "Max disparity 0.970
        # between ZZ_BARE and Male" at severity HIGH: a breach that does not
        # exist, naming as its worst-served party a group that reported nothing
        # at all. Someone is sent to investigate that, finds nothing, and stops
        # believing the next chart. Absent is not zero, in either direction.
        raw_rate = stats.get(metric, stats.get("positive_rate", stats.get("mean_prediction", None)))
        rate = _to_finite_float(raw_rate)
        # The engine emits the per-group row count as "size" (see
        # evaluation/vfairness_metrics/classification.py::group_statistics).
        # Reading only "count"/"n" missed on every real report, so a group with
        # 300 rows was labelled "n=0" on the bar, which reads as an empty
        # stratum: the exact thing an auditor looks for. The older keys stay as
        # fallbacks for hand-built and legacy report shapes. A row that carries
        # none of the three reported no count, and "n=0" is a count.
        n = stats.get("size", stats.get("count", stats.get("n", None)))
        if rate is None:
            display = None
            ungraded_groups.append(str(group))
        else:
            values.append(rate)
            graded_pairs.append((str(group), rate))
            if task_type == "regression":
                display = f"{rate:.3f}"
            else:
                display = f"{int(rate * 100)}%"
        bars.append(
            {
                "group": str(group)[:20],
                "rate": rate,
                "n": n,
                "display_value": display,
                "color": _GROUP_COLORS[i % len(_GROUP_COLORS)],
            }
        )

    overall_rate = sum(values) / len(values) if values else None
    if task_type == "regression":
        overall_display = f"{overall_rate:.3f}" if overall_rate is not None else "N/A"
    else:
        overall_display = f"{int(overall_rate * 100)}%" if overall_rate is not None else "N/A"

    max_disp = max(values, default=0) - min(values, default=0) if values else None
    # A disparity needs TWO comparable groups. With one surviving group the gap
    # is 0.0 by construction, which the bands read as "Low", a green all-clear
    # for a comparison that never happened, with the dropped group's name
    # nowhere on the canvas. Suppress the number instead of banding it.
    if not assess["assessable"] or len(values) < 2:
        max_disp = None
    if max_disp is None:
        # No group data: a neutral state, never a red 'High' verdict about
        # a chart that has nothing on it.
        d_color, d_bg, d_icon, d_label = "#64748b", "#f1f5f9", "·", "No data"
    else:
        # Band from the value as it is DISPLAYED (the template shows it at
        # 3 decimals): banding the raw value let 0.0999 render as '0.100'
        # right next to a 'Low' label.
        shown_disp = round(max_disp, 3)
        if shown_disp < 0.1:
            d_color, d_bg, d_icon, d_label = "#059669", "#d1fae5", "✓", "Low"
        elif shown_disp < 0.2:
            d_color, d_bg, d_icon, d_label = "#f59e0b", "#fef3c7", "⚠", "Moderate"
        else:
            d_color, d_bg, d_icon, d_label = "#dc2626", "#fee2e2", "✗", "High"

    # The gap's operands are named from the MEASURED rows only, and they are
    # ranked by the rate the bars actually show. The old sort ran over every
    # group with a `.get(metric, 0)` key, so a group with no rate sorted to the
    # front and was printed as the best-served party of a disparity it took no
    # part in.
    ordered = sorted(graded_pairs, key=lambda pair: pair[1])
    best_group = ordered[0][0][:12] if ordered else ""
    worst_group = ordered[-1][0][:12] if ordered else ""

    metric_label = metric.replace("_", " ").title()

    subtitle = f"Comparing {metric_label} across demographic groups."
    if not assess["assessable"] or len(values) < 2:
        if assess["reason"]:
            subtitle = assess["reason"]
        elif len(values) < 2:
            subtitle = (
                f"{NOT_ASSESSABLE_TEXT}: fewer than two groups are shown, so no "
                f"between-group disparity was measured."
            )
    elif assess["excluded"]:
        subtitle += f" Not shown (insufficient evidence): {', '.join(assess['excluded'])}."
    # The gap is banded over the groups that WERE measured, so a partly measured
    # run still gives a usable verdict on the part that ran. What it may never
    # do is read as an all-clear over the whole attribute, so the count of
    # unmeasured groups goes on the canvas beside the badge, not only here.
    if ungraded_groups:
        subtitle += (
            f" {len(ungraded_groups)} group(s) reported no {metric_label.lower()} "
            f"and are not in the gap: {', '.join(ungraded_groups[:4])}."
        )

    data = {
        "title": _attr_title("Fairness Metrics: Group Comparison", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "not_assessable": not assess["assessable"] or len(values) < 2,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "metric_label": metric_label,
        "bars": bars[:6],
        "n_ungraded_groups": len(ungraded_groups),
        "ungraded_groups": ungraded_groups,
        "overall_rate": overall_rate,
        "overall_display": overall_display,
        "max_disparity": max_disp,
        "disparity_color": d_color,
        "disparity_bg": d_bg,
        "disparity_icon": d_icon,
        "disparity_label": d_label,
        "best_group": best_group,
        "worst_group": worst_group,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("group_comparison", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 5.  Effect Sizes


def _effect_interpretation(d: float):
    """Map |d| to Cohen's interpretation and styling."""
    ad = abs(d)
    if ad < 0.2:
        return "Negligible", "#059669", "#d1fae5"
    if ad < 0.5:
        return "Small", "#3b82f6", "#dbeafe"
    if ad < 0.8:
        return "Medium", "#f59e0b", "#fef3c7"
    return "Large", "#dc2626", "#fee2e2"


def effect_sizes_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """Render effect sizes (Cohen's d) as a lollipop chart SVG.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``FairnessAnalyzer.get_report()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    metrics = report_dict.get("metrics", {})
    assess = report_assessability(report_dict)
    px_per_unit = 120

    effects = []
    abs_vals = []
    n_unmeasured = 0
    ungraded: List[str] = []
    for key, value in metrics.items():
        if value is None:
            continue
        # Cohen's d bands the magnitude of a DIFFERENCE, so only a
        # lower-is-better violation magnitude belongs on this chart. The
        # direction is asked of the shared helper: a local
        # `key.endswith("_ratio")` excluded the "_ratio" family but let
        # `disparate_impact` through, and |0.00| < 0.2 badged the worst outcome
        # that metric can express as a green "Negligible" effect.
        if metric_direction(key) is not MetricDirection.LOWER_IS_BETTER:
            ungraded.append(str(key))
            continue
        num = _to_finite_float(value)
        if num is None:
            # A lollipop needs a position. |nan| < 0.2 is False and so is every
            # other band test, so a NaN effect fell through to "Large" and drew
            # a red badge for a quantity that was never computed.
            n_unmeasured += 1
            continue
        label = _METRIC_LABELS.get(key, key.replace("_", " ").title())
        # The BANDING and the pixel offset read the coerced number, not the raw
        # one. `_is_finite` accepts a numeric string deliberately, so the old
        # `_effect_interpretation(value)` reached `abs("0.5")` and raised
        # TypeError on any report that had been through JSON or CSV. Measured
        # 2026-09-30: the identical report rendered fine through the radar.
        interp, color, badge_bg = _effect_interpretation(num)
        px_offset = int(max(-200, min(200, num * px_per_unit)))

        effects.append(
            {
                "label": label,
                "value": num,
                "label_short": label[:2].upper(),
                "px_offset": px_offset,
                "color": color,
                "interpretation": interp,
                "badge_bg": badge_bg,
                "badge_color": color,
            }
        )
        abs_vals.append(abs(num))

    not_assessable = (not assess["assessable"]) or not abs_vals
    if not_assessable:
        # Third state: the badges report no magnitude at all rather than a
        # banded one. `|f3` renders None as "N/A", so nothing numeric is shown.
        max_eff = None
        avg_eff = None
        max_color = _UNKNOWN_COLOR
        avg_color = _UNKNOWN_COLOR
        max_interp = NOT_ASSESSABLE_TEXT
    else:
        max_eff = max(abs_vals, default=0)
        avg_eff = sum(abs_vals) / len(abs_vals) if abs_vals else 0
        max_interp, max_color, _ = _effect_interpretation(max_eff)
        _, avg_color, _ = _effect_interpretation(avg_eff)

    subtitle = "Negligible: |d| < 0.2 · Small: < 0.5 · Medium: < 0.8 · Large: ≥ 0.8"
    if not_assessable:
        subtitle = assess["reason"] or (
            f"{NOT_ASSESSABLE_TEXT}: no effect size could be computed on this data."
        )
        if n_unmeasured:
            subtitle += f" {n_unmeasured} metric(s) undefined."
    elif n_unmeasured:
        subtitle += f" · {n_unmeasured} metric(s) undefined and not plotted"
    if ungraded:
        # Named on the canvas. A metric silently absent from the chart reads as
        # a metric with nothing to report.
        subtitle = f"Not an effect size: {', '.join(ungraded[:3])}. {subtitle}"

    data = {
        "title": _attr_title("Fairness Metrics: Effect Sizes (Cohen's d)", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "effects": effects[:6],
        "max_effect": max_eff,
        "max_effect_color": max_color,
        "max_effect_interpretation": max_interp,
        "avg_effect": avg_eff,
        "avg_effect_color": avg_color,
        "not_assessable": not_assessable,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "n_unmeasured": n_unmeasured,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("effect_sizes", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 6.  Confidence Intervals (Forest Plot)


def _computed_interval(key: str, ci_sources: List[dict]) -> Optional[Dict[str, Any]]:
    """Return the COMPUTED interval for ``key``, or None. Never invents one.

    The report emits bootstrap intervals under ``metrics_with_ci`` keyed
    ``lower_bound`` / ``upper_bound``. This adapter used to read a
    ``confidence_intervals`` key with ``lower`` / ``upper`` that NOTHING in the
    library has ever emitted, so on genuine output every lookup missed and the
    caller silently received ``value +/- 0.02`` drawn under a heading reading
    "Confidence Intervals". A real stratified-bootstrap band of [0.530, 0.695]
    rendered as [0.585, 0.625] with no marker separating it from a computed one,
    and the significance badge read "4 significant" on data the library's own
    bootstrap called NOT significant.

    Both key spellings are accepted (``confidence_intervals`` is still a
    legitimate way for a CALLER to pass a real interval in), but a missing or
    non-finite bound returns None, which the caller must render as "CI not
    computed". There is deliberately no fallback branch here to fall into.
    """
    for source in ci_sources:
        ci_info = source.get(key)
        if not isinstance(ci_info, dict):
            continue
        low = _to_finite_float(ci_info.get("lower_bound", ci_info.get("lower")))
        high = _to_finite_float(ci_info.get("upper_bound", ci_info.get("upper")))
        if low is None or high is None:
            continue
        result: Dict[str, Any] = {"low": low, "high": high}
        # The engine's own significance verdict, when it computed one.
        if "is_significant" in ci_info:
            result["is_significant"] = ci_info["is_significant"]
        return result
    return None


def confidence_intervals_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """Render confidence intervals as a forest plot SVG.

    Only COMPUTED intervals are drawn. A metric without one is shown as a bare
    point estimate labelled "CI not computed" and is excluded from the
    significance count, which is suppressed entirely when no metric has an
    interval.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``FairnessAnalyzer.get_report()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    metrics = report_dict.get("metrics", {})
    assess = report_assessability(report_dict)
    ci_sources = [
        report_dict.get("metrics_with_ci") or {},
        report_dict.get("confidence_intervals") or {},
    ]

    plot_left = 200
    plot_w = 420  # px

    # Determine scale range over MEASURED values only: a NaN reaches int() and
    # raises "cannot convert float NaN to integer", so an unmeasurable metric
    # used to take the whole chart down instead of being reported as unmeasured.
    # COERCED, not merely filtered. `_is_finite` says yes to a numeric string
    # on purpose, so the old `[v for v in metrics.values() if _is_finite(v)]`
    # put the raw ``"0.5"`` into the list and the very next line,
    # ``min(all_vals + [0])``, raised ``TypeError: '<' not supported between
    # instances of 'int' and 'str'``. Measured 2026-09-30 on a stringified real
    # report: this chart and two of its five siblings died while the radar and
    # the group comparison rendered the same input.
    all_vals = [_to_finite_float(v) for v in metrics.values() if _is_finite(v)]
    if not all_vals:
        all_vals = [0]
    val_min = min(all_vals + [0])
    val_max = max(all_vals + [0])
    for source in ci_sources:
        for key in metrics:
            interval = _computed_interval(key, [source])
            if interval:
                val_min = min(val_min, interval["low"])
                val_max = max(val_max, interval["high"])

    margin = max(0.05, (val_max - val_min) * 0.15)
    scale_min = val_min - margin
    scale_max = val_max + margin
    scale_range = scale_max - scale_min if scale_max != scale_min else 1

    def val_to_px(v):
        return int(plot_left + ((v - scale_min) / scale_range) * plot_w)

    # Build tick labels
    n_ticks = 6
    tick_labels = []
    x_ticks_vals = []
    for i in range(n_ticks):
        v = scale_min + i * scale_range / (n_ticks - 1)
        tick_labels.append(f"{v:.2f}")
        x_ticks_vals.append(v)

    intervals = []
    n_significant = 0
    n_with_ci = 0
    n_unmeasured = 0
    n_undecided = 0
    for key, value in metrics.items():
        if value is None:
            continue
        num = _to_finite_float(value)
        if num is None:
            # No point estimate means no diamond and no whisker: there is
            # nothing to place on the axis.
            n_unmeasured += 1
            continue
        label = _METRIC_LABELS.get(key, key.replace("_", " ").title())
        interval = _computed_interval(key, ci_sources)

        if interval is None:
            # Third state. The point estimate is real, the interval is not
            # available, and NOTHING is synthesised to fill the gap.
            intervals.append(
                {
                    "label": label,
                    "value": num,
                    "ci_low": None,
                    "ci_high": None,
                    "ci_computed": False,
                    "ci_left_px": val_to_px(num),
                    "ci_right_px": val_to_px(num),
                    "point_px": val_to_px(num),
                    "color": _UNKNOWN_COLOR,
                }
            )
            continue

        n_with_ci += 1
        ci_low = interval["low"]
        ci_high = interval["high"]

        # Determine if significant. Which test applies is a DIRECTION question,
        # so it goes to the shared helper rather than to a local name test: a
        # ratio is significant when its interval sits outside the fair band, a
        # difference when it excludes 0. An exact `endswith("_ratio")` here also
        # sent `disparate_impact` (a ratio) down the difference branch.
        significant: Optional[bool]
        if is_ratio_metric(key):
            significant = ci_high < 0.8 or ci_low > 1.2
        elif "is_significant" in interval:
            # The engine already answered this for its own bootstrap interval;
            # re-deriving it here is how the two surfaces drift apart.
            significant = bool(interval["is_significant"])
        elif metric_direction(key) is MetricDirection.LOWER_IS_BETTER:
            significant = ci_low > 0 or ci_high < 0
        else:
            # Fail closed. With no known better-direction and no engine verdict
            # there is no test to run, and "excludes 0" is not one: it answered
            # False for an interval straddling zero, which fed the green "No
            # significant disparity" all-clear below.
            significant = None

        if significant is None:
            n_undecided += 1
            color = _UNKNOWN_COLOR
        elif significant:
            n_significant += 1
            color = "#dc2626"
        else:
            color = "#3b82f6"

        intervals.append(
            {
                "label": label,
                "value": num,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_computed": True,
                "ci_left_px": val_to_px(ci_low),
                "ci_right_px": val_to_px(ci_high),
                "point_px": val_to_px(num),
                "color": color,
            }
        )

    # Zero line position
    threshold_x = val_to_px(0) if scale_min <= 0 <= scale_max else None

    # The significance badge is a claim ABOUT confidence intervals. With no
    # computed interval there is no such claim to make, so the badge is
    # suppressed rather than reporting "0 / n" (which reads as a measured
    # all-clear) over bands that were never computed.
    not_assessable = (not assess["assessable"]) or n_with_ci == 0
    if not_assessable:
        s_bg, s_color, s_icon = _UNKNOWN_BG, _UNKNOWN_COLOR, _UNKNOWN_ICON
        if not assess["assessable"]:
            s_text = f"{NOT_ASSESSABLE_TEXT}: no comparison was performed"
        else:
            s_text = "CI not computed: no significance claim is made"
    elif n_significant == 0 and n_undecided:
        # The green all-clear speaks for every interval on the plot, so one
        # interval that could not be tested withdraws it (the same rule the
        # heatmap badge follows).
        s_bg, s_color, s_icon = _UNKNOWN_BG, _UNKNOWN_COLOR, _UNKNOWN_ICON
        s_text = f"{n_undecided} interval(s) could not be tested: no all-clear is claimed"
    elif n_significant == 0:
        s_bg, s_color, s_icon, s_text = "#d1fae5", "#059669", "✓", "No significant disparity"
    else:
        s_bg, s_color, s_icon, s_text = "#fee2e2", "#dc2626", "✗", f"{n_significant} significant"

    subtitle = "Diamond = point estimate. Whiskers = confidence interval bounds."
    if not_assessable:
        subtitle = assess["reason"] or (
            "No confidence interval was computed for any metric on this report, "
            "so no interval is drawn and no significance is claimed."
        )
        if n_unmeasured:
            subtitle += f" {n_unmeasured} metric(s) undefined."
    elif n_unmeasured or len(intervals) > n_with_ci:
        subtitle += (
            f" {len(intervals) - n_with_ci + n_unmeasured} metric(s) without a "
            f"computed interval are excluded from the significance count."
        )

    data = {
        "title": _attr_title("Fairness Metrics: Confidence Intervals", report_dict),
        "subtitle": _subtitle_with_exclusions(subtitle, assess),
        "intervals": intervals[:6],
        "n_ticks": n_ticks,
        "tick_labels": tick_labels,
        "x_axis_label": "Metric Value",
        "threshold_x": threshold_x,
        "n_significant": n_significant,
        "n_total": n_with_ci,
        "n_ci_computed": n_with_ci,
        "not_assessable": not_assessable,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "n_unmeasured": n_unmeasured,
        "summary_bg": s_bg,
        "summary_color": s_color,
        "summary_icon": s_icon,
        "summary_text": s_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("confidence_intervals", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
