"""
SVG Rendering Adapter for Auto-Discovery / Intersectional Analysis Module.

Transforms detect_protected_attributes, scan_fairness_violations, and
GroupAdvantage results into flat dicts for ``auto_discovery.svg``.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .adapters_fairness import NOT_ASSESSABLE_TEXT
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

_PASS = "#059669"
_FAIL = "#dc2626"
_WARN = "#f59e0b"
_INFO = "#3b82f6"
_GROUP_COLORS = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

_SEVERITY_COLORS = {"high": _FAIL, "medium": _WARN, "low": _INFO}

# Slate, the could-not-check tone. Same value and same reason as
# ``adapters_validation._UNKNOWN`` and the NaN branch of ``engine._risk_color``:
# a cell that reported nothing is neither green nor amber nor red, because every
# one of those reads to a reviewer as something that was measured.
_UNKNOWN = "#64748b"

#: What a cell prints when the row did not report that field. Deliberately a
#: phrase and not a dash, a zero or "n/a": a reader must be told that the number
#: is ABSENT, not handed a glyph they can mistake for a measurement.
_WITHHELD_TEXT = "not reported"

#: The only severities this page can colour and count. Anything else, including
#: nothing at all, is ungraded.
_SEVERITY_LEVELS = ("high", "medium", "low")


def _number_or_none(raw: Any) -> Optional[float]:
    """A finite number the row actually reported, or None.

    CRITICAL, per row this time. Until 2026-08-27 the violation loop read
    ``float(d.get("value", 0))`` and ``float(d.get("threshold", 0))``, so a
    violation carrying nothing but an attribute name rendered VALUE 0.000 beside
    THRESHOLD 0.00 in the table and invited the reader to compare them. Both
    numbers were the adapter's own default. A default is not a measurement, and
    absent and zero are different: 0.000 against a 0.00 threshold is a specific,
    checkable claim about a model, and nothing in the input made it.

    Booleans are refused even though ``float(True)`` is 1.0, and NaN/inf are
    refused because they are sentinels rather than values.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _severity_or_none(raw: Any) -> Optional[str]:
    """One of high/medium/low as reported, or None when the row graded nothing.

    A severity is a JUDGEMENT, so defaulting one invents a finding. The old
    ``d.get("severity", "medium")`` gave every unlabelled violation an amber
    MEDIUM badge, and ``d.get("severity", "low")`` gave every unlabelled group a
    blue LOW one: both of those are verdicts, printed in a verdict colour, about
    a row that reported no verdict at all.

    An unrecognised severity is ungraded too, not silently coloured as the
    mildest thing it resembles.
    """
    if raw is None:
        return None
    if hasattr(raw, "value"):  # a ValidationSeverity-style enum
        raw = raw.value
    name = str(raw).strip().lower()
    return name if name in _SEVERITY_LEVELS else None


def _count_or_none(raw: Any) -> Optional[int]:
    """A reported group size, or None. ``size`` used to default to 0.

    "n=0" is a claim that the group was looked at and found empty, which is a
    finding; a group that reported no size made no such claim.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


#  auto_discovery.svg


def auto_discovery_to_svg(
    candidates: Optional[List] = None,
    violations: Optional[List] = None,
    group_advantages: Optional[List] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render auto-discovery scanner results as an SVG dashboard.

    Parameters
    ----------
    candidates : list of dicts
        Discovered protected attribute candidates, each with
        name, confidence, category.
    violations : list of dicts
        Fairness violations, each with attribute, metric, value,
        threshold, severity.
    group_advantages : list of GroupAdvantage or dicts
    example : bool
        Render the built-in demonstration fixture, watermarked EXAMPLE on the
        canvas. Off by default and never reached implicitly: see the note below.
    explanation : str
    save_path : str, optional

    Returns
    -------
    str  SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27 a call carrying
    nothing at all dispatched to ``_demo`` and this function returned a complete,
    confident scan of a dataset that was never opened: three named candidate
    attributes with confidences, four violations of which two were high
    severity, six named intersectional groups with sample sizes and positive
    rates, and the recommendation "2 high-severity violations, immediate review
    needed". None of it came from anything the caller supplied, and a caller
    could not tell that output apart from a real scan. An SVG is an export
    format: it leaves the building, an auditor reads it, and it outlives the
    version that produced it.

    A scan that never ran is COULD NOT CHECK, and it is stated on the canvas.
    That is the same trap ``identify_proxy_features`` fell into when it reported
    "no proxies" from a scan that could not look at all: a sentinel is not a
    measurement, and a count of zero is not a finding of zero. The demo fixture
    survives only behind the explicit ``example=True``, watermarked.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    if candidates is None and violations is None and group_advantages is None:
        if example:
            return _demo(explanation=explanation, save_path=save_path)
        return _not_scanned(explanation=explanation, save_path=save_path)

    # Candidates
    # A row that reported nothing gets no number, no badge, no colour and no
    # bar. The confidence bar is the clearest case: a missing confidence used to
    # default to 0, and `max(4, ...)` then drew a 4px stub in a confident hue,
    # so "no confidence was reported" and "the scanner is 2% confident" were the
    # same picture.
    cands: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates or []):
        d = c if isinstance(c, dict) else _obj_to_dict(c)
        conf = _number_or_none(d.get("confidence"))
        cands.append(
            {
                # `str(None)` MINTS a name: `.get(key, default)` does not fire for a
                # key PRESENT holding None, so a candidate whose `name` field defaults
                # to None was listed as the attribute "None". G13 2026-09-30.
                "name": str(d.get("name") or d.get("attribute") or f"attr_{i}"),
                "confidence": conf,
                "confidence_reported": conf is not None,
                "confidence_display": f"{conf:.2f}" if conf is not None else _WITHHELD_TEXT,
                "bar_w": max(4, int(conf * 160)) if conf is not None else 0,
                "color": _GROUP_COLORS[i % len(_GROUP_COLORS)] if conf is not None else _UNKNOWN,
                "category": str(d.get("category") or "unknown"),
            }
        )

    # Violations
    viols: List[Dict[str, Any]] = []
    for v in violations or []:
        d = v if isinstance(v, dict) else _obj_to_dict(v)
        sev = _severity_or_none(d.get("severity"))
        value = _number_or_none(d.get("value"))
        threshold = _number_or_none(d.get("threshold"))
        viols.append(
            {
                "attribute": str(d.get("attribute") or "unspecified"),
                "metric": str(d.get("metric") or "unspecified"),
                "value": value,
                "value_reported": value is not None,
                "value_display": f"{value:.3f}" if value is not None else _WITHHELD_TEXT,
                "threshold": threshold,
                "threshold_reported": threshold is not None,
                "threshold_display": f"{threshold:.2f}"
                if threshold is not None
                else _WITHHELD_TEXT,
                # "" and not "medium": every downstream test here is an equality
                # against a named level, so an ungraded row can match none of
                # them rather than quietly matching the one it was defaulted to.
                "severity": sev or "",
                "graded": sev is not None,
                "severity_label": sev.upper() if sev else "NOT GRADED",
                "color": _SEVERITY_COLORS[sev] if sev is not None else _UNKNOWN,
            }
        )

    # Group advantages
    groups: List[Dict[str, Any]] = []
    for i, ga in enumerate(group_advantages or []):
        d = ga if isinstance(ga, dict) else _obj_to_dict(ga)
        rate = _number_or_none(d.get("positive_rate"))
        size = _count_or_none(d.get("size"))
        # 1.0 was the worst default on this canvas: "ratio=1.00" is the precise
        # statement that the group sits exactly at the overall rate, which is
        # parity, which is the single most reassuring number the page can print.
        rel = _number_or_none(d.get("relative_to_overall"))
        sev = _severity_or_none(d.get("severity"))
        groups.append(
            {
                "group": str(d.get("group", f"group_{i}")),
                "positive_rate": rate,
                "positive_rate_reported": rate is not None,
                "positive_rate_display": f"{rate:.3f}" if rate is not None else _WITHHELD_TEXT,
                "size": size,
                "relative_to_overall": rel,
                "meta_display": (
                    f"n={size if size is not None else _WITHHELD_TEXT} · "
                    f"ratio={f'{rel:.2f}' if rel is not None else _WITHHELD_TEXT}"
                ),
                "severity": sev or "",
                "graded": sev is not None,
                "severity_label": sev.upper() if sev else "NOT GRADED",
                "color": _SEVERITY_COLORS[sev] if sev is not None else _UNKNOWN,
                "bar_w": max(4, int(rate * 200)) if rate is not None else 0,
            }
        )

    recs: List[str] = []
    if cands:
        recs.append(f"{len(cands)} candidate attribute(s) discovered, verify relevance.")
    high_viols = [v for v in viols if v["severity"] == "high"]
    if high_viols:
        recs.append(f"{len(high_viols)} high-severity violation(s), immediate review needed.")
    # An ungraded row has no place in a count that implies it was assessed, and
    # the caveat goes ABOVE the graded tally so it cannot be the line that falls
    # off the end of the four-line block.
    ungraded_viols = [v for v in viols if not v["graded"]]
    if ungraded_viols:
        recs.append(
            f"{len(ungraded_viols)} violation(s) reported no severity and are not graded here; "
            "their seriousness is unknown."
        )
    unmeasured_viols = [v for v in viols if not (v["value_reported"] and v["threshold_reported"])]
    if unmeasured_viols:
        recs.append(
            f"{len(unmeasured_viols)} violation(s) reported no value or no threshold, "
            "so nothing was compared against a limit."
        )
    # Every recorded violation reaches this block, not just the high-severity
    # ones. Reacting to "high" alone let a medium-severity breach at four times
    # its threshold sit in the table while the summary line underneath it read
    # "No significant fairness issues auto-discovered".
    #
    # Counted over the GRADED rows only. "N further violation(s) recorded below
    # high severity" is a statement about severity, and an ungraded row does not
    # support it: it is not below high severity, it is unplaced.
    other_viols = len(viols) - len(high_viols) - len(ungraded_viols)
    if other_viols:
        recs.append(
            f"{other_viols} further violation(s) recorded below high severity, "
            "check each value against its threshold."
        )
    if groups:
        # None is not a ratio. A group that reported no relative_to_overall used
        # to arrive here defaulted to exactly 1.0, which sits in neither bucket,
        # so it silently made the "no advantaged/disadvantaged group" case look
        # measured. Compare only the groups that reported the number.
        rated = [g for g in groups if g["relative_to_overall"] is not None]
        adv = [g for g in rated if g["relative_to_overall"] > 1.2]
        dis = [g for g in rated if g["relative_to_overall"] < 0.8]
        if adv and dis:
            recs.append(f"{len(adv)} advantaged and {len(dis)} disadvantaged group(s) identified.")
        unrated = len(groups) - len(rated)
        if unrated:
            recs.append(
                f"{unrated} group(s) reported no rate relative to the overall population "
                "and were not compared."
            )

    # Could-not-check: a scan that was never run is not a scan that came back
    # clean. None means the caller passed no result for that part (the scanner
    # did not run, or could not look); an empty list means it ran and found
    # nothing, which is the only thing that earns the reassuring line below.
    not_scanned = [
        name
        for name, supplied in (
            ("attributes", candidates),
            ("violations", violations),
            ("groups", group_advantages),
        )
        if supplied is None
    ]
    if not_scanned:
        # The disclosure must never be the line that falls off the end of the
        # four-line block, so the findings yield to it.
        recs = recs[:3] + [
            "Not scanned: "
            + ", ".join(not_scanned)
            + ". A clean result cannot be reported for what was not scanned."
        ]

    # Zero of zero is not a clean bill of health. When the scan came back with
    # no candidate attribute, no violation AND no group, it examined nothing:
    # there was no attribute to test and no group to compare, so "no
    # significant issues" is a claim about work that never happened. It is the
    # same trap identify_proxy_features fell into when it reported "no proxies"
    # from a scan that could not look at all, and the same shape as a headline
    # that printed EQUITABLE because 0 == 0.
    examined_nothing = not cands and not viols and not groups
    if not recs:
        if examined_nothing:
            # One line, and it has to fit the 644-wide card without wrapping:
            # the recommendation block draws each entry as a single <text>.
            recs.append(
                f"{NOT_ASSESSABLE_TEXT}: nothing was examined, 0 attributes, "
                "0 violations, 0 groups; no issue can be reported as absent."
            )
        else:
            recs.append("No significant fairness issues auto-discovered.")

    # The violations badge needs the third state too. n_high > 0 alone cannot
    # tell "scanned and found none" from "did not scan", and both of those
    # rendered the same green zero.
    scan_state = "could_not_check" if (violations is None or examined_nothing) else "scanned"

    # Each of the three panels needs that third state too, not just the badge.
    # An empty panel is SILENT, and silence next to a heading reads as "the scan
    # looked here and found nothing". None means the scanner did not run for
    # that part at all, so the template writes a different sentence for each.
    cand_state = "not_scanned" if candidates is None else "scanned"
    viol_state = "not_scanned" if violations is None else "scanned"
    group_state = "not_scanned" if group_advantages is None else "scanned"

    template_data = {
        "title": "Auto-Discovery Scanner",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": f"{len(cands)} attributes · {len(viols)} violations · {len(groups)} groups",
        "candidates": cands[:6],
        "violations": viols[:8],
        "groups": groups[:8],
        "n_violations": len(viols),
        "n_high": len([v for v in viols if v["severity"] == "high"]),
        # The badge sub-line said "{{ n_high }} high severity" and nothing else,
        # so a table of rows that reported no severity at all printed
        # "0 high severity" in emerald. That is a clean bill of health issued
        # over rows nobody graded. See the third badge branch in the template.
        "n_ungraded": len(ungraded_viols),
        "scan_state": scan_state,
        "cand_state": cand_state,
        "viol_state": viol_state,
        "group_state": group_state,
        "recommendations": recs[:4],
        "explanation": explanation,
    }
    if examined_nothing:
        # Swaps the curated "prioritise the high-severity violations" action
        # for the one that says outright that this chart certifies nothing.
        template_data["not_assessable"] = True
        template_data["not_assessable_reason"] = "The scan examined no attribute and no group."

    return _render_and_save("auto_discovery", template_data, save_path)


def _not_scanned(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """The third state: a scanner page for a run where the scanner never ran.

    Every field that could read as a finding is withheld rather than defaulted.
    ``scan_state`` keeps the violations badge on NOT SCANNED, the three per-panel
    states put a sentence in each empty panel, and ``not_assessable`` drives the
    accessible ``<desc>`` and the action line through
    ``rendering.explain.build_explanation``, so the canvas and the screen reader
    tell one story.

    The subtitle deliberately does not read "0 attributes, 0 violations, 0
    groups". Those three zeroes are the same line a genuinely clean scan prints,
    and nothing here was examined.
    """
    reason = (
        "No candidate attribute, no violation and no group profile was supplied to "
        "auto_discovery_to_svg, so nothing was scanned on this run."
    )
    template_data = {
        "title": "Auto-Discovery Scanner",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": "Nothing was scanned: no scanner result was supplied",
        "candidates": [],
        "violations": [],
        "groups": [],
        "n_violations": 0,
        "n_high": 0,
        "n_ungraded": 0,
        "scan_state": "could_not_check",
        "cand_state": "not_scanned",
        "viol_state": "not_scanned",
        "group_state": "not_scanned",
        "not_assessable": True,
        "not_assessable_reason": reason,
        "recommendations": [
            "COULD NOT CHECK: nothing was scanned, so no attribute and no group was examined.",
            "This page is not a clean bill of health and not a finding: it certifies nothing.",
            "Pass detect_protected_attributes, scan_fairness_violations or group results to scan.",
        ],
        "explanation": explanation,
    }
    return _render_and_save("auto_discovery", template_data, save_path)


def _obj_to_dict(obj) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {}


def _render_and_save(template: str, data: dict, save_path: Optional[str]) -> str:
    try:
        svg = render_svg(template, data)
        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)
        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise


def _demo(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """Build template data with demo values. Reachable only via example=True."""
    # Every row below carries the same *_display / *_reported keys the live path
    # computes, because the template reads only those. A fixture row that omits
    # them renders a blank cell where a number belongs, which is the one thing a
    # withheld-value design must never look like.
    candidates = [
        {
            "name": "gender",
            "confidence": 0.95,
            "confidence_reported": True,
            "confidence_display": "0.95",
            "bar_w": 152,
            "color": "#3b82f6",
            "category": "demographic",
        },
        {
            "name": "race",
            "confidence": 0.88,
            "confidence_reported": True,
            "confidence_display": "0.88",
            "bar_w": 141,
            "color": "#ec4899",
            "category": "demographic",
        },
        {
            "name": "age_bucket",
            "confidence": 0.72,
            "confidence_reported": True,
            "confidence_display": "0.72",
            "bar_w": 115,
            "color": "#f59e0b",
            "category": "inferred",
        },
    ]
    violations = [
        {
            "attribute": "gender",
            "metric": "demographic_parity",
            "value": 0.15,
            "value_reported": True,
            "value_display": "0.150",
            "threshold": 0.10,
            "threshold_reported": True,
            "threshold_display": "0.10",
            "severity": "high",
            "graded": True,
            "severity_label": "HIGH",
            "color": _FAIL,
        },
        {
            "attribute": "race",
            "metric": "equalized_odds",
            "value": 0.12,
            "value_reported": True,
            "value_display": "0.120",
            "threshold": 0.10,
            "threshold_reported": True,
            "threshold_display": "0.10",
            "severity": "medium",
            "graded": True,
            "severity_label": "MEDIUM",
            "color": _WARN,
        },
        {
            "attribute": "age_bucket",
            "metric": "calibration_diff",
            "value": 0.08,
            "value_reported": True,
            "value_display": "0.080",
            "threshold": 0.10,
            "threshold_reported": True,
            "threshold_display": "0.10",
            "severity": "low",
            "graded": True,
            "severity_label": "LOW",
            "color": _INFO,
        },
        {
            "attribute": "gender × race",
            "metric": "demographic_parity",
            "value": 0.22,
            "value_reported": True,
            "value_display": "0.220",
            "threshold": 0.10,
            "threshold_reported": True,
            "threshold_display": "0.10",
            "severity": "high",
            "graded": True,
            "severity_label": "HIGH",
            "color": _FAIL,
        },
    ]
    groups = [
        {
            "group": "male × white",
            "positive_rate": 0.45,
            "positive_rate_reported": True,
            "positive_rate_display": "0.450",
            "size": 1200,
            "relative_to_overall": 1.32,
            "meta_display": "n=1200 · ratio=1.32",
            "severity": "high",
            "graded": True,
            "severity_label": "HIGH",
            "color": _FAIL,
            "bar_w": 90,
        },
        {
            "group": "female × white",
            "positive_rate": 0.38,
            "positive_rate_reported": True,
            "positive_rate_display": "0.380",
            "size": 980,
            "relative_to_overall": 1.12,
            "meta_display": "n=980 · ratio=1.12",
            "severity": "low",
            "graded": True,
            "severity_label": "LOW",
            "color": _INFO,
            "bar_w": 76,
        },
        {
            "group": "male × minority",
            "positive_rate": 0.32,
            "positive_rate_reported": True,
            "positive_rate_display": "0.320",
            "size": 750,
            "relative_to_overall": 0.94,
            "meta_display": "n=750 · ratio=0.94",
            "severity": "low",
            "graded": True,
            "severity_label": "LOW",
            "color": _INFO,
            "bar_w": 64,
        },
        {
            "group": "female × minority",
            "positive_rate": 0.22,
            "positive_rate_reported": True,
            "positive_rate_display": "0.220",
            "size": 620,
            "relative_to_overall": 0.65,
            "meta_display": "n=620 · ratio=0.65",
            "severity": "high",
            "graded": True,
            "severity_label": "HIGH",
            "color": _FAIL,
            "bar_w": 44,
        },
        {
            "group": "non-binary × white",
            "positive_rate": 0.35,
            "positive_rate_reported": True,
            "positive_rate_display": "0.350",
            "size": 180,
            "relative_to_overall": 1.03,
            "meta_display": "n=180 · ratio=1.03",
            "severity": "low",
            "graded": True,
            "severity_label": "LOW",
            "color": _INFO,
            "bar_w": 70,
        },
        {
            "group": "non-binary × minority",
            "positive_rate": 0.28,
            "positive_rate_reported": True,
            "positive_rate_display": "0.280",
            "size": 120,
            "relative_to_overall": 0.82,
            "meta_display": "n=120 · ratio=0.82",
            "severity": "medium",
            "graded": True,
            "severity_label": "MEDIUM",
            "color": _WARN,
            "bar_w": 56,
        },
    ]
    recs = [
        "3 candidate attributes discovered, verify relevance for your use case.",
        "2 high-severity violations, immediate review needed.",
        "1 advantaged and 1 disadvantaged group identified, investigate the disparity.",
    ]

    template_data = {
        "title": "Auto-Discovery Scanner",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # DO NOT REMOVE. Every attribute, violation and group below is invented.
        # This flag is what puts the EXAMPLE band and watermark on the canvas and
        # the EXAMPLE ONLY marker at the head of the accessible description, and
        # it is the only thing that stops this render being read as a scan of a
        # real dataset.
        "is_example": True,
        "subtitle": "3 attributes · 4 violations · 6 groups",
        "candidates": candidates,
        "violations": violations,
        "groups": groups,
        "n_violations": 4,
        "n_high": 2,
        "n_ungraded": 0,
        "scan_state": "scanned",
        "cand_state": "scanned",
        "viol_state": "scanned",
        "group_state": "scanned",
        "recommendations": recs,
        "explanation": explanation,
    }

    return _render_and_save("auto_discovery", template_data, save_path)
