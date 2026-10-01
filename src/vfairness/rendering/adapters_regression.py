"""
SVG Rendering Adapter for Regression Fairness Module.

Transforms get_group_metrics(), compute_regression_effect_sizes(), and
parity functions into flat dicts for ``regression_fairness.svg``.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .._triage import is_flag
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

_PASS = "#059669"
_FAIL = "#dc2626"
_WARN = "#f59e0b"
_GROUP_COLORS = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

# Three states, never two: assessed-pass / assessed-fail / could-not-check.
#
# OPEN DEFECT, closed 2026-08-27: the headline was `n_pass == n_total`, and a
# report where NOTHING was measured satisfies 0 == 0, so it printed a green
# EQUITABLE. A metric supplied as None or NaN was dropped from the badge list
# entirely, which took it out of the DENOMINATOR too, so three measured metrics
# plus one unmeasurable one read as "3/3 metrics pass, EQUITABLE" and the reader
# never learned a metric was missing. Same shape as register findings #8 and
# #11. The could-not-check state is decided HERE and handed to the template;
# DO NOT reduce it back to the two-way comparison.
NOT_ASSESSABLE_TEXT = "NOT ASSESSABLE"
_UNKNOWN_LABEL = "NO DATA"
_UNKNOWN_COLOR = "#64748b"  # slate-500: neither the pass green nor the fail red
_UNKNOWN_BG = "#f1f5f9"  # slate-100


def _finite(value: Any) -> Optional[float]:
    """Coerce to a real, finite float, or None. NaN, inf, None and junk give None.

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
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _pf(val: float, thr: float) -> str:
    return "PASS" if abs(val) <= thr else "FAIL"


def _pf_color(val: float, thr: float) -> str:
    return _PASS if abs(val) <= thr else _FAIL


#  regression_fairness.svg


def regression_fairness_to_svg(
    group_metrics: Optional[Dict] = None,
    disparities: Optional[Dict] = None,
    effect_sizes: Optional[List] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render regression fairness results as an SVG dashboard.

    Parameters
    ----------
    group_metrics : dict
        Per-group metrics from get_group_metrics(). Keys are group names,
        values are dicts with mae, rmse, r2, mean_residual, etc.
    disparities : dict
        Disparity values with keys like mae_parity, rmse_parity,
        mean_pred_diff, r2_parity. A key that is absent was never part of the
        report; a key present with None or NaN was requested and could not be
        measured, so it is shown as NO DATA, kept in the denominator, and the
        headline is withheld (NOT ASSESSABLE) rather than reading EQUITABLE.
    effect_sizes : list of dicts
        Effect sizes between group pairs from compute_regression_effect_sizes().
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
    no data at all dispatched to ``_demo`` and this function returned a
    complete, confident regression assessment: four named disparity metrics over
    three named groups, a ``2/4 metrics pass`` rate, a BIASED headline, per-group
    MAE/RMSE/R-squared figures, three Cohen's d effect sizes and the line "the
    model under-predicts for the Female group". None of it was measured from
    anything the caller supplied, and a caller could not tell that output apart
    from a real evaluation. An SVG is an export format: it leaves the building,
    an auditor reads it, and it outlives the version that produced it.

    Missing input is now routed through the SAME could-not-check path that
    ``regression_fairness_to_svg({}, {}, [])`` already took, which prints NOT
    ASSESSABLE on the badge, withholds the rate and says on the canvas that
    nothing was measured. Deliberately not a second could-not-check canvas of
    its own: two codepaths saying the same thing is how one of them drifts. The
    demo fixture survives only behind the explicit ``example=True``, watermarked.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    if example and group_metrics is None and disparities is None:
        return _demo(explanation=explanation, save_path=save_path)

    # Disparity badges
    disp = disparities or {}
    badges: List[Dict[str, Any]] = []
    badge_defs = [
        ("MAE Parity", "mae_parity", 0.10),
        ("RMSE Parity", "rmse_parity", 0.10),
        ("Mean Pred Diff", "mean_pred_diff", 0.10),
        ("R² Parity", "r2_parity", 0.10),
    ]
    for label, key, thr in badge_defs:
        if key not in disp:
            # Not supplied at all: this metric was never part of the report.
            continue
        val = _finite(disp[key])
        if val is None:
            # Supplied but unmeasurable (None, NaN, or a value that is not a
            # number). It stays on the chart and in the denominator, because a
            # metric that quietly leaves the count turns "1 of 2 could not be
            # checked" into a clean "1/1 metrics pass".
            badges.append(
                {
                    "name": label,
                    "value": None,  # the |f3 filter prints None as "N/A"
                    "threshold": thr,
                    "label": _UNKNOWN_LABEL,
                    "color": _UNKNOWN_COLOR,
                    "bg": _UNKNOWN_BG,
                }
            )
            continue
        badges.append(
            {
                "name": label,
                "value": val,
                "threshold": thr,
                "label": _pf(val, thr),
                "color": _pf_color(val, thr),
                "bg": "#ecfdf5" if abs(val) <= thr else "#fef2f2",
            }
        )

    # Per-group table
    #
    # Three states per ROW, never two. The chart badge above already withholds
    # the verdict when a disparity metric could not be checked, but until
    # 2026-08-27 every cell of this table defaulted its own value to 0: a group
    # supplied as `{"Female": {}}`, carrying nothing but its name, was tabulated
    # as N 0, MAE 0.000, RMSE 0.000, R-squared 0.000, MEAN RES. +0.0000 and
    # STD RES. 0.000, and drew a residual bar labelled "over-predict". Six
    # invented measurements and a plotted point for a group that reported none.
    # An honest NOT ASSESSABLE headline makes that WORSE, not better: the badge
    # lends the fabricated rows its credibility. `_finite` is the same coercion
    # the disparity badges use, so a value that is absent, None, NaN, inf or not
    # a number stays None all the way to the template, where `|f3` prints N/A
    # and the cell is greyed. NEVER restore a numeric default here: a default is
    # not a measurement, and absent and zero are different.
    groups: List[Dict[str, Any]] = []
    if group_metrics:
        for i, (name, gm) in enumerate(sorted(group_metrics.items())):
            residual = _finite(gm.get("mean_residual"))
            size = _finite(gm.get("size", gm.get("n")))
            groups.append(
                {
                    "name": str(name),
                    "n": int(size) if size is not None else None,
                    "mae": _finite(gm.get("mae")),
                    "rmse": _finite(gm.get("rmse")),
                    "r2": _finite(gm.get("r2")),
                    "mean_residual": residual,
                    "std_residual": _finite(gm.get("std_residual")),
                    "color": _GROUP_COLORS[i % len(_GROUP_COLORS)],
                    # No residual measured means no bar and no direction: a
                    # zero-width bar at the axis still reads as "measured, and
                    # it came out neutral".
                    "residual_bar_w": None
                    if residual is None
                    else int(min(80, abs(residual) * 400)),
                    "residual_dir": None
                    if residual is None
                    else ("right" if residual >= 0 else "left"),
                }
            )

    # Effect sizes (lollipop)
    #
    # Three states PER ROW, never two. This line was
    # `float(d.get("cohens_d", d.get("effect_size", 0)))`, and it failed in both
    # directions at once. A pair explicitly reporting `cohens_d=None` raised
    # TypeError out of `float(None)` and took the ENTIRE report down, including
    # every group and disparity metric that had been measured perfectly well; a
    # crash is not the third state. And a pair that simply carried no effect
    # size at all was plotted as a measured d = 0.00: a dot painted ON the zero
    # axis, in the PASS green, which is the strongest statement this panel can
    # make ("these two groups' predictions do not differ at all") issued for a
    # pair nobody compared. `_finite` is the same coercion the disparity badges
    # and the group table use. NEVER restore a numeric default here.
    effects: List[Dict[str, Any]] = []
    for es in effect_sizes or []:
        d: Dict[str, Any] = es if isinstance(es, dict) else _obj_to_dict(es)
        cohen = _finite(
            d.get("cohens_d") if d.get("cohens_d") is not None else d.get("effect_size")
        )
        effects.append(
            {
                # `str(None)` MINTS a name: a `.get` default does not fire for a key
                # PRESENT holding None, which is what a dataclass field defaulting to
                # None gives, so the panel listed an "None" pair. G13 2026-09-30.
                "pair": str(d.get("pair") or d.get("groups") or "unnamed pair"),
                "cohens_d": cohen,
                "measured": cohen is not None,
                # No dot and no band for a pair that reported no effect size:
                # every x on this axis is a magnitude, and slate is not one.
                "dot_x": None
                if cohen is None
                else int(380 + min(200, abs(cohen) * 240) * (1 if cohen >= 0 else -1)),
                "color": _UNKNOWN_COLOR
                if cohen is None
                else (_FAIL if abs(cohen) >= 0.5 else (_WARN if abs(cohen) >= 0.2 else _PASS)),
            }
        )
    effects_unmeasured = [e for e in effects if not e["measured"]]

    n_pass = sum(1 for b in badges if b["label"] == "PASS")
    failing = [b for b in badges if b["label"] == "FAIL"]
    unchecked = [b for b in badges if b["label"] == _UNKNOWN_LABEL]

    # The headline. A measured breach is a finding whatever else is missing, so
    # FAIL still wins. EQUITABLE is only claimable when every metric on the
    # chart was actually compared to its threshold and passed: one unchecked
    # metric withdraws the all-clear without asserting a breach that was never
    # measured either.
    if failing:
        not_assessable = False
        not_assessable_reason = ""
    elif not badges:
        not_assessable = True
        not_assessable_reason = "no disparity metric was supplied, so nothing was measured."
    elif unchecked:
        not_assessable = True
        not_assessable_reason = (
            f"{len(unchecked)} of {len(badges)} disparity metric(s) could not be checked, "
            f"so no all-clear is claimed."
        )
    else:
        not_assessable = False
        not_assessable_reason = ""

    recs: List[str] = []
    if failing:
        recs.append(f"{len(failing)} disparity metric(s) exceed threshold; review model equity.")
    if unchecked:
        recs.append(
            f"{len(unchecked)} disparity metric(s) could not be checked "
            f"({', '.join(b['name'] for b in unchecked)}); this report certifies nothing "
            f"about them."
        )
    if not badges:
        recs.append(
            "No disparity metric was measured, so this report does not establish "
            "prediction equity either way."
        )
    if any(e["measured"] and abs(e["cohens_d"]) >= 0.5 for e in effects):
        recs.append("Large effect size detected between groups; investigate prediction gaps.")
    # A pair that reported no effect size was not compared, and the silence has
    # to be said out loud for the same reason the residual one below is: without
    # it the fall-through prints "All regression fairness checks pass" over a
    # panel whose rows measured nothing.
    if effects_unmeasured:
        recs.append(
            f"{len(effects_unmeasured)} group pair(s) reported no effect size "
            f"({', '.join(e['pair'] for e in effects_unmeasured[:4])}); the size of the "
            f"prediction gap between them was not measured."
        )
    residual_measured = [g for g in groups if g["mean_residual"] is not None]
    residual_missing = [g for g in groups if g["mean_residual"] is None]
    if any(abs(g["mean_residual"]) > 0.05 for g in residual_measured):
        recs.append(
            "Residual bias present: the model systematically over-predicts or "
            "under-predicts for some groups."
        )
    # A group that reported no mean residual was not screened for residual bias,
    # and the silence must be said out loud. It also keeps the clean-sheet line
    # below off a page whose rows measured nothing: the fall-through fires
    # whenever no other recommendation applies, so an all-defaults table used to
    # end on "All regression fairness checks pass; prediction equity is
    # maintained." with not one residual actually checked.
    if residual_missing:
        recs.append(
            f"{len(residual_missing)} group(s) reported no mean residual "
            f"({', '.join(g['name'] for g in residual_missing[:4])}); residual bias was "
            f"not checked for them."
        )
    if not recs:
        recs.append("All regression fairness checks pass; prediction equity is maintained.")

    subtitle = f"{len(badges)} metrics · {len(groups)} groups"
    if not_assessable:
        subtitle = f"{NOT_ASSESSABLE_TEXT}: {not_assessable_reason}"
    # Headline rule (c): the ungraded counts belong ON THE CANVAS, in the
    # headline band, not only down in the Recommendations panel. This line sits
    # at y=122 beside the verdict badge, which is where a reader decides how
    # much of the page was actually measured. Kept to short clauses: the run is
    # unwrapped and the badge starts at x=520.
    silences = []
    if effects_unmeasured:
        silences.append(f"{len(effects_unmeasured)} pair(s) not compared")
    if residual_missing:
        silences.append(f"{len(residual_missing)} group(s) with no residual")
    if silences:
        subtitle = f"{subtitle} · {'; '.join(silences)}"

    template_data = {
        "title": "Regression Fairness Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "not_assessable": not_assessable,
        "not_assessable_reason": not_assessable_reason,
        "badges": badges[:4],
        "groups": groups[:8],
        "effects": effects[:6],
        "n_pass": n_pass,
        "n_total": len(badges),
        "n_effects_unmeasured": len(effects_unmeasured),
        "n_residual_missing": len(residual_missing),
        "recommendations": recs[:4],
        "explanation": explanation,
    }

    return _render_and_save("regression_fairness", template_data, save_path)


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
    badges = [
        {
            "name": "MAE Parity",
            "value": 0.12,
            "threshold": 0.10,
            "label": "FAIL",
            "color": _FAIL,
            "bg": "#fef2f2",
        },
        {
            "name": "RMSE Parity",
            "value": 0.08,
            "threshold": 0.10,
            "label": "PASS",
            "color": _PASS,
            "bg": "#ecfdf5",
        },
        {
            "name": "Mean Pred Diff",
            "value": 0.15,
            "threshold": 0.10,
            "label": "FAIL",
            "color": _FAIL,
            "bg": "#fef2f2",
        },
        {
            "name": "R² Parity",
            "value": 0.06,
            "threshold": 0.10,
            "label": "PASS",
            "color": _PASS,
            "bg": "#ecfdf5",
        },
    ]
    groups = [
        {
            "name": "Male",
            "n": 1200,
            "mae": 0.142,
            "rmse": 0.185,
            "r2": 0.823,
            "mean_residual": 0.032,
            "std_residual": 0.178,
            "color": "#3b82f6",
            "residual_bar_w": 13,
            "residual_dir": "right",
        },
        {
            "name": "Female",
            "n": 980,
            "mae": 0.168,
            "rmse": 0.212,
            "r2": 0.791,
            "mean_residual": -0.045,
            "std_residual": 0.201,
            "color": "#ec4899",
            "residual_bar_w": 18,
            "residual_dir": "left",
        },
        {
            "name": "Other",
            "n": 320,
            "mae": 0.155,
            "rmse": 0.198,
            "r2": 0.808,
            "mean_residual": 0.012,
            "std_residual": 0.190,
            "color": "#f59e0b",
            "residual_bar_w": 5,
            "residual_dir": "right",
        },
    ]
    effects = [
        {"pair": "Male vs Female", "cohens_d": 0.35, "dot_x": 464, "color": _WARN},
        {"pair": "Male vs Other", "cohens_d": 0.15, "dot_x": 416, "color": _PASS},
        {"pair": "Female vs Other", "cohens_d": -0.22, "dot_x": 327, "color": _WARN},
    ]
    recs = [
        "2 disparity metrics exceed threshold; review model equity.",
        "Residual bias present: the model under-predicts for the Female group.",
        "Cohen's d between Male/Female is medium (0.35); investigate.",
    ]

    template_data = {
        "title": "Regression Fairness Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": "4 metrics · 3 groups",
        # DO NOT REMOVE. Every metric, group and count below is invented. This
        # flag is what puts the EXAMPLE band and watermark on the canvas and the
        # EXAMPLE ONLY marker at the head of the accessible description, and it
        # is the only thing that stops this render being read as a finding about
        # a real model.
        "is_example": True,
        # The demo shows a measured, failing report: two metrics breach their
        # threshold, so the headline is a finding and not a withheld verdict.
        "not_assessable": False,
        "not_assessable_reason": "",
        "badges": badges,
        "groups": groups,
        "effects": effects,
        "n_pass": 2,
        "n_total": 4,
        "recommendations": recs,
        "explanation": explanation,
    }

    return _render_and_save("regression_fairness", template_data, save_path)
