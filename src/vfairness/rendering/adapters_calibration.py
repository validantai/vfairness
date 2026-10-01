"""
Adapters for calibration visualization SVG templates.

Each adapter transforms calibration data (arrays, reports, results)
into the flat data dict expected by the corresponding Jinja2 SVG template.
"""

import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from vfairness._not_assessed import MIN_ROWS_PER_GROUP_FOR_CALIBRATION

from .adapters_fairness import _is_missing_level, _present_levels
from .engine import render_svg

_GROUP_COLORS = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

# Three states, never two: assessed-pass / assessed-fail / could-not-check.
# The third state is NOT a shade of the other two, so it never borrows their
# palette. A calibration chart is an export: it leaves the building and outlives
# the run that produced it, so one that measured nothing must SAY so on its own
# canvas.
#
# OPEN DEFECT, closed 2026-08-27: with no data at all, or with probabilities
# that enter no bin (every comparison against NaN is False), the ECE loop never
# ran, `ece` stayed at its 0.0 seed and the badge read "Well Calibrated,
# ECE = 0.000". That is a certificate of calibration issued on no measurement.
# Same root shape as register finding #11 in adapters_monitoring. DO NOT drop
# the guard in _reliability_verdict below.
#
# The same defect survived in the two siblings until 2026-08-27, and both are
# closed here. group_calibration_to_svg issued a green "Well Calibrated Across
# Groups" with ECE 0.000 for all-NaN probabilities and for zero groups, and
# calibration_disparity_to_svg printed a green "NO SIGNIFICANT DISPARITY / All
# groups calibrated within acceptable bounds" beside a 99.90x DISPARITY RATIO
# and two N/A group names. A sentinel is not a measurement, and a comparison
# needs at least two measured groups before it exists at all. DO NOT reinstate
# a 0.0 seed, a 99.9 fallback or an unconditional green branch in either.
NOT_ASSESSABLE_TEXT = "NOT ASSESSABLE"
_UNKNOWN_COLOR = "#64748b"  # slate-500: neither the pass green nor the fail red
_UNKNOWN_BG = "#f1f5f9"  # slate-100
_UNKNOWN_ICON = "·"


# 1.  Reliability Diagram


def _compute_calibration_bins(y_true, y_prob, n_bins=10):
    """Compute calibration curve bins from arrays."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() > 0:
            mean_pred = y_prob[mask].mean()
            frac_pos = y_true[mask].mean()
            count = int(mask.sum())
            # A bin whose coordinates are not finite cannot be placed on the
            # axis, so it is not on the chart at all. Plotting it used to raise
            # "cannot convert float NaN to integer" deep inside val_to_y; the
            # bin is dropped here instead, and the ECE loops (which keep their
            # own masks) still see the NaN and withdraw the verdict.
            if not (math.isfinite(mean_pred) and math.isfinite(frac_pos)):
                continue
            bins.append(
                {
                    "center": float(mean_pred),
                    "frac_pos": float(frac_pos),
                    "count": count,
                }
            )
    return bins


def reliability_diagram_to_svg(
    y_true,
    y_prob,
    *,
    n_bins: int = 10,
    show_histogram: bool = True,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render a reliability diagram as SVG.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of calibration bins.
    show_histogram : bool
        Whether to show prediction histogram.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    Three states, never two. When no prediction can be placed in a bin (no
    data at all, or probabilities that are NaN or outside [0, 1]), or the error
    comes out undefined, the chart renders NOT ASSESSABLE with ECE and MCE as
    "N/A" instead of certifying a calibration that was never measured.
    """
    import numpy as np

    bins = _compute_calibration_bins(y_true, y_prob, n_bins)

    # Chart geometry
    chart_left, chart_right = 90, 520
    chart_top, chart_bottom = 80, 370
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    def val_to_x(v):
        return int(chart_left + v * chart_w)

    def val_to_y(v):
        return int(chart_bottom - v * chart_h)

    # Build curve points
    curve_pts = []
    curve_dots = []
    for b in bins:
        px = val_to_x(b["center"])
        py = val_to_y(b["frac_pos"])
        curve_pts.append(f"{px},{py}")
        curve_dots.append({"x": px, "y": py})

    # Histogram
    histogram_bins = []
    if show_histogram and bins:
        max_count = max(b["count"] for b in bins)
        for b in bins:
            histogram_bins.append(
                {
                    "center": b["center"],
                    "height": b["count"] / max_count if max_count > 0 else 0,
                }
            )

    # Compute ECE
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    ece = 0.0
    mce = 0.0
    n_total = len(y_true_arr)
    # Rows that entered a bin. A row whose probability is NaN, or outside
    # [0, 1], enters none of them, and it is the count of rows that WERE
    # measured, not the count supplied, that says whether anything was
    # measured at all. Weighting by n_total instead would divide the error by
    # rows that contributed nothing to it, which shrinks ECE toward zero, i.e.
    # toward the green band, in exact proportion to how much data was
    # unusable.
    n_binned = 0
    weighted_gap = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (y_prob_arr >= bin_edges[i]) & (y_prob_arr < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob_arr == bin_edges[i + 1])
        count = int(mask.sum())
        if count > 0:
            bin_acc = y_true_arr[mask].mean()
            bin_conf = y_prob_arr[mask].mean()
            diff = abs(bin_acc - bin_conf)
            n_binned += count
            weighted_gap += count * float(diff)
            mce = max(mce, diff)
    if n_binned > 0:
        ece = weighted_gap / n_binned
    mce = float(mce)

    # Could-not-check, decided here and not in the template: the template can
    # only draw what it is given, and `ece < 0.05` is True for the 0.0 seed of
    # a run that measured nothing exactly as it is for a genuinely calibrated
    # model.
    unbinned = n_total - n_binned
    assessable = n_binned > 0 and math.isfinite(ece) and math.isfinite(mce)

    subtitle = "Points on the diagonal indicate perfect calibration."
    not_assessable_reason = ""
    if not assessable:
        if n_total == 0:
            not_assessable_reason = "no data was supplied, so nothing was measured."
        elif n_binned == 0:
            not_assessable_reason = "no probability could be binned, so nothing was measured."
        else:
            not_assessable_reason = "the calibration error came out undefined on this data."
        subtitle = f"{NOT_ASSESSABLE_TEXT}: {not_assessable_reason}"
    elif unbinned > 0:
        subtitle = f"{unbinned} of {n_total} prediction(s) could not be binned and are excluded."

    # THE SAMPLE SIZE, WHEN IT IS TOO SMALL TO CARRY THE BADGE.
    #
    # MEASURED on this tree, 2026-09-30, with ONE row (y_true=[1], y_prob=[0.7]):
    # this chart published the badge "Poorly Calibrated", "ECE = 0.300" and the
    # accessible desc "Model is Poorly Calibrated (ECE 0.300, MCE 0.300)" at
    # severity HIGH. A calibration verdict about a model, from one observation,
    # with the sample size nowhere on the canvas. `assessable = n_binned > 0` is
    # the whole floor, so any single binned row buys the graded badge.
    #
    # WITHHOLDING THE BADGE BELOW THAT FLOOR WAS TRIED AND WITHDRAWN. Refusing
    # to grade under MIN_ROWS_PER_GROUP_FOR_CALIBRATION reddened an existing
    # control, ``test_a_partly_unusable_input_is_measured_on_what_survived_and
    # _says_so``, whose fixture is six rows with four surviving and which
    # asserts both that "ECE = " is on the canvas and that the verdict is NOT
    # "NOT ASSESSABLE". That control is a deliberate over-correction guard from
    # an earlier wave: it exists to prove the NaN-binning guard does not refuse
    # an input that has some usable rows. A guard that reddens an existing
    # control is the wrong guard, so it is not here.
    #
    # WHAT REMAINS OPEN, for whoever owns the calibration surfaces: whether a
    # whole-model calibration verdict needs a sample floor at all, and if so at
    # what n. MIN_ROWS_PER_GROUP_FOR_CALIBRATION is a per-GROUP comparability
    # floor, and reusing it for a single-model verdict is a different judgement
    # than the one its docstring records. Until that is decided, the honest half
    # that conflicts with nothing is shipped: the COUNT goes on the canvas, so
    # the badge is no longer a verdict presented as though it rested on data.
    # Added only when the count is below the floor, which is how the healthy
    # canvas stays byte-identical.
    small_sample = assessable and n_binned < MIN_ROWS_PER_GROUP_FOR_CALIBRATION
    if small_sample:
        subtitle = (
            f"From {n_binned} binned prediction(s), fewer than "
            f"{MIN_ROWS_PER_GROUP_FOR_CALIBRATION}: read the verdict as provisional. {subtitle}"
        )

    # Styling
    if not assessable:
        curve_color = _UNKNOWN_COLOR
        ece_color = _UNKNOWN_COLOR
    elif ece < 0.05:
        curve_color = "#059669"
        ece_color = "#059669"
    elif ece < 0.1:
        curve_color = "#3b82f6"
        ece_color = "#f59e0b"
    else:
        curve_color = "#dc2626"
        ece_color = "#dc2626"

    if not assessable:
        mce_color = _UNKNOWN_COLOR
    else:
        mce_color = "#dc2626" if mce > 0.15 else "#f59e0b" if mce > 0.08 else "#059669"

    if not assessable:
        v_bg, v_color, v_icon, v_text, v_detail = (
            _UNKNOWN_BG,
            _UNKNOWN_COLOR,
            _UNKNOWN_ICON,
            NOT_ASSESSABLE_TEXT,
            "ECE not measurable",
        )
    elif ece < 0.05:
        v_bg, v_color, v_icon, v_text, v_detail = (
            "#d1fae5",
            "#065f46",
            "✓",
            "Well Calibrated",
            f"ECE = {ece:.3f}",
        )
    elif ece < 0.1:
        v_bg, v_color, v_icon, v_text, v_detail = (
            "#fef3c7",
            "#92400e",
            "⚠",
            "Moderate",
            f"ECE = {ece:.3f}",
        )
    else:
        v_bg, v_color, v_icon, v_text, v_detail = (
            "#fee2e2",
            "#991b1b",
            "✗",
            "Poorly Calibrated",
            f"ECE = {ece:.3f}",
        )

    data = {
        "title": "Reliability Diagram",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "curve_points": " ".join(curve_pts),
        "curve_dots": curve_dots,
        "curve_color": curve_color,
        "show_histogram": show_histogram,
        "histogram_bins": histogram_bins,
        # None, not 0.0: the |f3 filter prints None as "N/A", and a printed
        # 0.000 is read as a measured zero by every reader of the chart.
        "ece": ece if assessable else None,
        "mce": mce if assessable else None,
        # Carried out so a caller or a test can read the denominator behind the
        # badge without parsing the subtitle prose. See the small_sample note.
        "n_binned": n_binned,
        "small_sample": small_sample,
        "min_rows_to_grade": MIN_ROWS_PER_GROUP_FOR_CALIBRATION,
        "ece_color": ece_color,
        "mce_color": mce_color,
        "not_assessable": not assessable,
        "not_assessable_reason": not_assessable_reason,
        "verdict_bg": v_bg,
        "verdict_color": v_color,
        "verdict_icon": v_icon,
        "verdict_text": v_text,
        "verdict_detail": v_detail,
    }

    data["explanation"] = explanation
    svg = render_svg("reliability_diagram", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  Group Calibration


def group_calibration_to_svg(
    y_true,
    y_prob,
    protected_attr,
    *,
    n_bins: int = 10,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render group calibration comparison as SVG.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    protected_attr : array-like
        Protected attribute group labels.
    n_bins : int
        Number of calibration bins.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    Three states, never two. A group whose probabilities enter no bin has no
    measurable ECE and renders as "N/A" rather than as 0.000, and a disparity
    between groups does not exist until two groups have been measured, so with
    fewer than two the chart renders NOT ASSESSABLE instead of certifying that
    the groups agree.
    """
    import numpy as np

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    protected_attr = np.asarray(protected_attr)

    # PRESENT levels only, and the rows whose group label is absent are counted
    # so the canvas can say they are in no curve. See _present_levels: a group
    # minted from missing data was ranked "best calibrated" here, and the same
    # column with pd.NA in it took the whole chart down instead.
    unique_groups = _present_levels(protected_attr)
    _flat = np.asarray(protected_attr, dtype=object).ravel()
    _present = [not _is_missing_level(v) for v in _flat]
    n_missing_attr = int(len(_present) - sum(_present))

    # Chart geometry
    cl, cr, ct, cb = 80, 420, 80, 390
    cw, ch = cr - cl, cb - ct

    def val_to_x(v):
        return int(cl + v * cw)

    def val_to_y(v):
        return int(cb - v * ch)

    groups: List[Dict[str, Any]] = []
    eces: List[float] = []

    for gi, group in enumerate(unique_groups[:5]):
        # Elementwise, and never `protected_attr == group`: on an object column
        # holding pd.NA that expression yields pd.NA cells and `mask.sum()`
        # raises "boolean value of NA is ambiguous". The present-mask is
        # consulted first, so an absent cell can never be counted into a group.
        # Identical to the old expression, row for row, on a homogeneous column.
        mask = np.array([p and (v == group) for p, v in zip(_present, _flat)], dtype=bool)
        n_group = int(mask.sum())
        if n_group == 0:
            continue

        bins_data = _compute_calibration_bins(y_true[mask], y_prob[mask], n_bins)
        color = _GROUP_COLORS[gi % len(_GROUP_COLORS)]

        # Curve points
        pts = []
        dots = []
        for b in bins_data:
            px = val_to_x(b["center"])
            py = val_to_y(b["frac_pos"])
            pts.append(f"{px},{py}")
            dots.append({"x": px, "y": py})

        # Compute group ECE over the rows of this group that entered a bin.
        # Weighting by the group's supplied size instead would divide its error
        # by rows that contributed nothing to it, shrinking the group's ECE
        # toward the green band in exact proportion to how much of its data was
        # unusable. Identical arithmetic on healthy data, where every row bins.
        bin_edges = np.linspace(0, 1, n_bins + 1)
        n_binned_group = 0
        weighted_gap = 0.0
        for i in range(n_bins):
            bmask = mask & (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                bmask = bmask | (mask & (y_prob == bin_edges[i + 1]))
            count = int(bmask.sum())
            if count > 0:
                gap = abs(y_true[bmask].mean() - y_prob[bmask].mean())
                n_binned_group += count
                weighted_gap += count * float(gap)

        # None, never 0.0: a group whose rows entered no bin (all-NaN scores,
        # scores outside [0, 1]) was not measured, and a printed 0.000 is read
        # as a measured zero by every reader of the chart.
        ece: Optional[float] = weighted_gap / n_binned_group if n_binned_group > 0 else None
        if ece is not None and not math.isfinite(ece):
            ece = None

        # MEASURED and COMPARABLE are two different questions, and conflating them
        # is what produced the defect here AND, briefly, the wrong fix for it.
        #
        # THE DEFECT. A group with one row enters one bin, so n_binned_group is 1 and
        # the arithmetic above yields a real-looking ECE from a single observation.
        # Everything downstream then treated it as fully measured. Measured on 59
        # rows in group A beside ONE row in group B, this chart printed
        # "B  ECE = 0.010", named B the BEST CALIBRATED GROUP, reported
        # "Disparity: 0.244" across "2 groups analyzed", and recommended applying
        # group-specific calibration. The sample size appeared nowhere.
        #
        # THE WRONG FIX, which was tried first, was to set that ECE to None. It made
        # the ranking honest and it also deleted a real measurement: a four-row
        # group's own ECE is arithmetically correct, and two existing tests rely on
        # exactly that, one for the binned-rows weighting and one whose whole subject
        # is that "the guard withdraws the comparison, not the measurement".
        #
        # So the group KEEPS its ECE and is shown WITH ITS SAMPLE SIZE, which is the
        # disclosure that was actually missing, and it is excluded only from the
        # comparative claims: the spread, and best/worst. Those are the two places
        # where a small group's noise stops being a number about itself and becomes a
        # statement about another group.
        comparable = ece is not None and n_group >= MIN_ROWS_PER_GROUP_FOR_CALIBRATION
        if comparable and ece is not None:
            eces.append(ece)

        groups.append(
            {
                "label": str(group)[:14],
                "color": color,
                "ece": ece,
                "n": n_group,
                "comparable": comparable,
                "curve_points": " ".join(pts),
                "dots": dots,
            }
        )

    # Overall ECE, weighted the same way and over the same rows.
    bin_edges = np.linspace(0, 1, n_bins + 1)
    overall_binned = 0
    overall_gap = 0.0
    for i in range(n_bins):
        bmask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bmask = bmask | (y_prob == bin_edges[i + 1])
        count = int(bmask.sum())
        if count > 0:
            gap = abs(y_true[bmask].mean() - y_prob[bmask].mean())
            overall_binned += count
            overall_gap += count * float(gap)
    overall_ece: Optional[float] = overall_gap / overall_binned if overall_binned > 0 else None
    if overall_ece is not None and not math.isfinite(overall_ece):
        overall_ece = None

    # From `eces`, which now holds only the comparable groups, so "Max group ECE"
    # in the summary cannot be a number from a group too small to compare.
    max_ece: Optional[float] = max(eces) if eces else None

    # Could-not-check, decided here and not in the template, because the
    # template can only draw what it is given and `not has_disparity` is True
    # for a run that compared nothing exactly as it is for genuinely consistent
    # groups. A spread between groups does not exist until TWO groups have been
    # measured: one measured group, or none, is not agreement.
    measured = [g for g in groups if g["ece"] is not None]
    # Eligible for the comparison: measured AND large enough that the number says
    # something about the group rather than about four rows of it.
    comparables = [g for g in groups if g["comparable"]]
    ece_disparity: Optional[float] = max(eces) - min(eces) if len(eces) >= 2 else None
    assessable = ece_disparity is not None
    # Narrow on the value, not on the bool above: a plain bool carries no type
    # information, so a checker cannot see that ece_disparity is a float here.
    has_disparity = bool(ece_disparity is not None and ece_disparity > 0.02)

    unmeasured = len(groups) - len(measured)
    subtitle = "Each curve shows calibration for one demographic group."
    not_assessable_reason = ""
    if not assessable:
        # Kept short on purpose: this reason goes into the subtitle, which runs
        # under the stats badge at x=400 and is CLIPPED, not wrapped, past
        # roughly seventy characters. The finding panel below carries the same
        # sentence at full width, so the canvas never depends on the clip.
        if not groups:
            not_assessable_reason = "no group was supplied, so nothing was measured."
        elif not measured:
            not_assessable_reason = "no group's calibration error could be measured."
        elif len(measured) == 1:
            not_assessable_reason = "only one group's calibration error could be measured."
        else:
            # Several groups measured, fewer than two big enough to be compared. The
            # distinction matters to a reader: the numbers on the canvas are real,
            # and it is the RANKING between them that is withheld.
            not_assessable_reason = (
                f"only {len(comparables)} group(s) had at least "
                f"{MIN_ROWS_PER_GROUP_FOR_CALIBRATION} rows, too few to compare."
            )
        subtitle = f"{NOT_ASSESSABLE_TEXT}: {not_assessable_reason}"
    elif unmeasured > 0:
        subtitle = f"{unmeasured} of {len(groups)} group(s) could not be measured and are excluded."
    # Rows with NO group label are in no curve at all, and a curve set that
    # silently covers three quarters of the data reads as covering all of it.
    # The clause goes FIRST because this subtitle is CLIPPED past roughly
    # seventy characters (see the note above), so an appended caveat is the half
    # that disappears. Skipped entirely when nothing is missing, which is how
    # the healthy canvas stays byte-identical.
    if n_missing_attr:
        subtitle = f"{n_missing_attr} row(s) have no group label and are in no curve. {subtitle}"

    # Find best/worst, among the groups that were actually measured. Indexing
    # `groups` by a position in `eces` is only valid while every group has an
    # ECE, which is exactly what stopped being true above. "Best" and "worst"
    # are comparative, so they need the same two measured groups the spread
    # does: one group is not the best of anything.
    # From `comparables`, never `measured`: a group too small to compare cannot be
    # the best or the worst of anything, and naming it either is the claim that
    # started this.
    best_group = min(comparables, key=lambda g: g["ece"])["label"] if assessable else "N/A"
    worst_group = max(comparables, key=lambda g: g["ece"])["label"] if assessable else "N/A"

    if overall_ece is None:
        overall_ece_color = _UNKNOWN_COLOR
    else:
        overall_ece_color = (
            "#059669" if overall_ece < 0.05 else "#f59e0b" if overall_ece < 0.1 else "#dc2626"
        )
    if max_ece is None:
        max_ece_color = _UNKNOWN_COLOR
    else:
        max_ece_color = "#059669" if max_ece < 0.05 else "#f59e0b" if max_ece < 0.1 else "#dc2626"
    if not assessable:
        disparity_color = _UNKNOWN_COLOR
    else:
        disparity_color = "#dc2626" if has_disparity else "#059669"

    data = {
        "title": "Calibration by Group",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "groups": groups,
        "n_groups": len(groups),
        "n_measured_groups": len(measured),
        "n_comparable_groups": len(comparables),
        # Carried out so a caller or a test can count the rows with no group
        # label without parsing the subtitle prose.
        "n_missing_group_rows": n_missing_attr,
        "min_rows_to_compare": MIN_ROWS_PER_GROUP_FOR_CALIBRATION,
        "overall_ece": overall_ece,
        "overall_ece_color": overall_ece_color,
        "max_ece": max_ece,
        "max_ece_color": max_ece_color,
        "ece_disparity": ece_disparity,
        "disparity_color": disparity_color,
        "has_disparity": has_disparity,
        "not_assessable": not assessable,
        "not_assessable_text": NOT_ASSESSABLE_TEXT,
        "not_assessable_reason": not_assessable_reason,
        "unknown_color": _UNKNOWN_COLOR,
        "best_group": best_group,
        "worst_group": worst_group,
        "recommendation": "Apply group-specific calibration"
        if has_disparity
        else "Measure every group before drawing a conclusion"
        if not assessable
        else "No action needed",
    }

    data["explanation"] = explanation
    svg = render_svg("group_calibration", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  Calibration Disparity Analysis


def calibration_disparity_to_svg(
    disparity_result,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render calibration disparity analysis as SVG.

    Parameters
    ----------
    disparity_result : DisparityResult or dict
        Calibration disparity analysis result.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    Three states, never two. A disparity between groups does not exist until
    two groups have been measured, so with fewer than two the page renders NOT
    ASSESSABLE and prints "N/A" for the spread, the ratio and the best and
    worst ECE, rather than a green "no significant disparity" over a sentinel.
    """
    # Support both objects and dicts. `overall_ece` defaults to None, never to
    # 0: a caller that supplied no overall error did not measure a perfect one.
    #
    # The significance VERDICT gets the same three states, and for the same
    # reason. `.get("has_significant_disparity", False)` was the last defaulted
    # value on this path, and the recommendation panel keys on it directly: an
    # absent flag fell through to the green "NO SIGNIFICANT DISPARITY / All
    # groups calibrated within acceptable bounds", so a caller who supplied
    # group errors and no verdict was handed a clean bill of health nobody
    # issued. The gallery fixture is exactly that shape, and the canvas
    # contradicted itself in the process: a 0.037 spread in the alarm band in
    # the ECE DISPARITY tile at the top, an unqualified all-clear underneath.
    # A verdict is reported or it is missing. A reported False is a verdict and
    # still renders green.
    if isinstance(disparity_result, dict):
        group_metrics = disparity_result.get("group_metrics", {})
        has_sig_raw = disparity_result.get("has_significant_disparity")
        overall_ece = disparity_result.get("overall_ece", None)
        strategy = disparity_result.get("strategy", "N/A")
        recommendation = disparity_result.get("recommendation", "")
    else:
        group_metrics = getattr(disparity_result, "group_metrics", {})
        has_sig_raw = getattr(disparity_result, "has_significant_disparity", None)
        overall_ece = getattr(disparity_result, "overall_ece", None)
        strategy = getattr(disparity_result, "strategy", "N/A")
        recommendation = getattr(disparity_result, "recommendation", "")

    # `.get(key, {})` and `getattr(obj, key, {})` only fall back when the key is
    # ABSENT. A key that is PRESENT holding the wrong shape sails straight past
    # both, and `group_metrics.items()` below then raises
    # "'list' object has no attribute 'items'", which is how this surfaced: an
    # intermittent AttributeError from a calibration task, reported by the
    # session running the deployed consumer on 2026-09-09 and reproduced here
    # with {"group_metrics": []}. Same family as the `.get(key, default)` trap
    # already recorded in this repo, one shape along.
    #
    # An unrecognised shape is COULD NOT CHECK, so it is named and then handed
    # on as empty. That is not a silent swallow: with no measured groups the
    # `assessable` gate below is False, so the panel renders "not assessable"
    # rather than the green all-clear the comment above describes.
    if not isinstance(group_metrics, dict):
        warnings.warn(
            "calibration_disparity_to_svg: group_metrics arrived as "
            f"{type(group_metrics).__name__}, not a mapping of group to metrics, "
            "so per-group calibration was NOT MEASURED and the panel reports it "
            "as not assessable. Every in-library producer emits a dict; check "
            "whatever built this result.",
            UserWarning,
            stacklevel=2,
        )
        group_metrics = {}

    has_sig_known = has_sig_raw is not None
    has_sig = bool(has_sig_raw)

    try:
        overall_ece = float(overall_ece) if overall_ece is not None else None
    except (TypeError, ValueError):
        overall_ece = None
    if overall_ece is not None and not math.isfinite(overall_ece):
        overall_ece = None

    # Build group list. A group's metrics dict may lack 'ece' entirely; that
    # group renders as N/A (the |f3 filter shows None as "N/A") instead of
    # crashing, and is excluded from the disparity statistics.
    # A GROUP KEY THAT IS THE ABSENCE OF A GROUP is dropped before it becomes a
    # name. `str(group)[:14]` below mints one out of any of absence's six doors,
    # and this panel then RANKS it. Measured 2026-09-30 with
    # {None: {"ece": 0.02}, "b": {"ece": 0.09}}: the desc read "ECE disparity
    # 0.070 (4.5x); worst b vs best None" at severity HIGH, and with pd.NA
    # instead, "worst b vs best <NA>". The recommendation panel then said
    # "Recalibrate the worst-calibrated group toward parity", i.e. toward the
    # calibration of the rows whose group was not recorded. The sibling
    # group_calibration_to_svg closed the same six doors in the same wave; this
    # is the copy that reads a mapping instead of a column.
    n_missing_group_keys = sum(1 for k in group_metrics if _is_missing_level(k))
    group_metrics = {k: v for k, v in group_metrics.items() if not _is_missing_level(k)}

    groups: List[Dict[str, Any]] = []
    for gi, (group, metrics) in enumerate(group_metrics.items()):
        raw = metrics.get("ece") if isinstance(metrics, dict) else metrics
        try:
            ece = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            ece = None
        # NaN parses as a float but is not a measurement, and it prints as the
        # literal "nan" through the |f3 filter, which reads as a value.
        if ece is not None and not math.isfinite(ece):
            ece = None
        groups.append(
            {
                "label": str(group)[:14],
                "ece": ece,
                "ece_scaled": min(1.0, ece * 10)
                if ece is not None
                else 0.0,  # scale for bar display
                "color": _GROUP_COLORS[gi % len(_GROUP_COLORS)],
            }
        )

    with_ece = [g for g in groups if g["ece"] is not None]
    eces = [g["ece"] for g in with_ece]

    # Could-not-check, decided here and not in the template. Every quantity
    # below used to have a fallback that looked like a measurement: the spread
    # fell back to 0, the best and worst ECE to 0.000 beside a group named
    # "N/A", and the ratio to a 99.9 sentinel that the template printed as
    # "99.90x" in alarm red. A spread needs TWO measured groups to exist, and
    # the ratio additionally needs a non-zero denominator, which a genuinely
    # perfect best group does not provide.
    # "Best" and "worst" are comparative and need the same two measured groups
    # the spread does: naming the single supplied group both the best and the
    # worst calibrated, under a green and a red stripe, compares it to nothing.
    assessable = len(eces) >= 2
    best_ece: Optional[float] = min(eces) if assessable else None
    worst_ece: Optional[float] = max(eces) if assessable else None
    best_group = min(with_ece, key=lambda g: g["ece"])["label"] if assessable else "N/A"
    worst_group = max(with_ece, key=lambda g: g["ece"])["label"] if assessable else "N/A"

    # Derived from the Optionals directly rather than through `assessable`, so a
    # type checker can follow the narrowing. Same values, same three states.
    ece_disparity: Optional[float] = (
        worst_ece - best_ece if worst_ece is not None and best_ece is not None else None
    )
    disparity_ratio: Optional[float] = (
        worst_ece / best_ece
        if worst_ece is not None and best_ece is not None and best_ece > 0
        else None
    )

    if ece_disparity is None:
        ece_d_color = _UNKNOWN_COLOR
    else:
        ece_d_color = (
            "#dc2626" if ece_disparity > 0.03 else "#f59e0b" if ece_disparity > 0.01 else "#059669"
        )
    if disparity_ratio is None:
        ratio_color = _UNKNOWN_COLOR
        # "N/A", not "N/Ax": the unit belongs to a number, and there is none.
        ratio_text = "N/A"
    else:
        ratio_color = (
            "#dc2626" if disparity_ratio > 3 else "#f59e0b" if disparity_ratio > 1.5 else "#059669"
        )
        # The 99.9 cap is a display limit, so a capped value says so instead of
        # printing a ratio the data did not establish.
        ratio_text = f"{disparity_ratio:.2f}x" if disparity_ratio <= 99.9 else "over 99.9x"

    overall_ece_color = _UNKNOWN_COLOR if overall_ece is None else "#0f172a"
    best_color = "#059669" if best_ece is not None else _UNKNOWN_COLOR
    worst_color = "#dc2626" if worst_ece is not None else _UNKNOWN_COLOR

    # A group that reported no calibration error is not a group that was
    # compared, so the number of them belongs on the canvas in the headline
    # band, next to the title, and not only in the panel two thirds of the way
    # down. group_calibration_to_svg has carried this line since its own fix;
    # this sibling never got it, which is how "two measured groups plus one
    # silent one" could reach an unqualified all-clear with nothing above the
    # fold saying a group had been left out.
    ungraded_groups = len(groups) - len(with_ece)
    subtitle = "Compares calibration error across demographic groups."
    not_assessable_reason = ""
    if not assessable:
        # Kept short for the same reason as in group_calibration_to_svg above:
        # the subtitle is clipped, not wrapped, and the panel repeats it in
        # full, so nothing on the canvas depends on the clip.
        if not groups:
            not_assessable_reason = "no group was supplied, so nothing was compared."
        elif not with_ece:
            not_assessable_reason = "no group's calibration error was supplied."
        else:
            not_assessable_reason = "only one group's calibration error was supplied."
        subtitle = f"{NOT_ASSESSABLE_TEXT}: {not_assessable_reason}"
    elif ungraded_groups > 0:
        # Kept under about seventy characters, like the reasons above it. The
        # subtitle runs from x=36 at font-size 10 and the headline-band badge
        # that carries the same count starts at x=404, so a longer sentence
        # would run underneath it. The badge states HOW MANY, this states WHY.
        subtitle = f"{ungraded_groups} of {len(groups)} group(s) reported no calibration error."
    # The dropped no-name keys, named where a reader looks. Silently removing
    # them would be its own omission: the rows are still in the model's traffic,
    # they are just not a demographic group. Prepended, because the subtitle is
    # clipped and the generic legend prose is the half that can afford to go.
    if n_missing_group_keys:
        subtitle = (
            f"{n_missing_group_keys} key(s) carried no group label and are not compared. {subtitle}"
        )

    data = {
        "title": "Calibration Disparity Analysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "groups": groups[:5],
        "n_groups": len(groups),
        "n_measured_groups": len(with_ece),
        "overall_ece": overall_ece,
        "overall_ece_color": overall_ece_color,
        "ece_disparity": ece_disparity,
        "ece_disparity_color": ece_d_color,
        "best_group": best_group,
        "best_ece": best_ece,
        "best_color": best_color,
        "worst_group": worst_group,
        "worst_ece": worst_ece,
        "worst_color": worst_color,
        "disparity_ratio": disparity_ratio,
        "disparity_ratio_text": ratio_text,
        "ratio_color": ratio_color,
        "not_assessable": not assessable,
        "not_assessable_text": NOT_ASSESSABLE_TEXT,
        "not_assessable_reason": not_assessable_reason,
        "unknown_color": _UNKNOWN_COLOR,
        "has_significant_disparity": has_sig,
        # The third state of the verdict. The template needs both: `has_sig`
        # alone cannot tell "the caller reported no disparity" from "the caller
        # reported nothing", and those are the green branch and a slate one.
        "has_significant_disparity_known": has_sig_known,
        "n_ungraded_groups": ungraded_groups,
        # Carried out so a caller or a test can count them without parsing prose.
        "n_missing_group_keys": n_missing_group_keys,
        "recommendation": str(recommendation)[:44],
        "strategy": str(strategy),
    }

    data["explanation"] = explanation
    svg = render_svg("calibration_disparity", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4.  Pareto Frontier


def pareto_frontier_to_svg(
    calibration_errors,
    fairness_violations,
    *,
    labels: Optional[List[str]] = None,
    x_label: str = "Calibration Error",
    y_label: str = "Fairness Violation",
    x_higher_is_better: bool = False,
    explanation: Optional[str] = None,
    baseline_index: Optional[int] = None,
    fairness_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render a Pareto frontier scatter plot as SVG.

    Parameters
    ----------
    calibration_errors : array-like
        X-axis values (calibration errors).
    fairness_violations : array-like
        Y-axis values (fairness violations).
    labels : list of str, optional
        Labels for each point.
    x_label : str
        X axis label.
    y_label : str
        Y axis label.
    baseline_index : int, optional
        Index of the baseline point (rendered distinctly).
    fairness_threshold : float, optional
        Horizontal threshold line on the Y axis.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Raises
    ------
    ValueError
        If the two coordinate arrays (or *labels*) disagree in length. Each
        point needs both of its coordinates, and nothing here can say which
        values were meant to pair, so there is no honest chart to draw and no
        could-not-check canvas either: this is contradictory input, not absent
        input.

    Notes
    -----
    Three states, never two. With no configuration to plot, or with none whose
    coordinates are finite, the chart renders NOT ASSESSABLE and withholds the
    point count, the frontier count and the best trade-off, rather than
    reporting "0 configurations | 0 Pareto-optimal" as though a search had run
    and come back empty. Until 2026-08-27 the empty case did not get that far:
    ``pareto_frontier_to_svg([], [])`` died inside numpy on "zero-size array to
    reduction operation minimum which has no identity", which names nothing a
    caller can act on. Non-finite points are dropped the way
    ``_compute_calibration_bins`` drops unplaceable bins, and the subtitle says
    how many, because a NaN coordinate cannot be placed on an axis and used to
    raise "cannot convert float NaN to integer" deep inside the pixel mapping.
    """
    import numpy as np

    cal_err_all = np.atleast_1d(np.asarray(calibration_errors, dtype=float))
    fair_viol_all = np.atleast_1d(np.asarray(fairness_violations, dtype=float))
    if cal_err_all.shape[0] != fair_viol_all.shape[0]:
        raise ValueError(
            f"pareto_frontier_to_svg() got {cal_err_all.shape[0]} calibration_errors and "
            f"{fair_viol_all.shape[0]} fairness_violations. Every point needs both of its "
            "coordinates and there is no way to tell which values were meant to pair, so "
            "no chart is drawn."
        )

    if labels is None:
        labels = [f"M{i + 1}" for i in range(cal_err_all.shape[0])]
    elif len(labels) != cal_err_all.shape[0]:
        raise ValueError(
            f"pareto_frontier_to_svg() got {len(labels)} labels for "
            f"{cal_err_all.shape[0]} point(s). Labelling by position needs one label per "
            "point; a short list silently mislabels the frontier."
        )

    # A point whose coordinates are not finite cannot be placed on either axis,
    # so it is not on the chart at all. It is dropped here, with its label and
    # with the baseline marker re-pointed at the row it still refers to, and
    # the subtitle reports the exclusion. Identical arithmetic on healthy data,
    # where every point is finite and `kept` is the full range.
    finite = np.isfinite(cal_err_all) & np.isfinite(fair_viol_all)
    kept = [i for i, ok in enumerate(finite) if ok]
    n_supplied = int(cal_err_all.shape[0])
    cal_err = cal_err_all[finite]
    fair_viol = fair_viol_all[finite]
    labels = [labels[i] for i in kept]
    n_dropped = n_supplied - len(kept)
    if baseline_index is not None:
        baseline_index = kept.index(baseline_index) if baseline_index in kept else None

    # Could-not-check, decided here and not in the template: the template can
    # only draw what it is given, and "0 configurations | 0 Pareto-optimal"
    # reads as the result of a search that found nothing rather than as a
    # search that never happened. A count of zero is not a finding of zero.
    if cal_err.size == 0:
        if n_supplied == 0:
            reason = "no configuration was supplied, so no trade-off was evaluated."
        else:
            reason = (
                f"none of the {n_supplied} supplied configuration(s) has finite coordinates, "
                "so none could be placed on the axes."
            )
        return _pareto_not_assessable(
            reason,
            x_label=x_label,
            y_label=y_label,
            explanation=explanation,
            save_path=save_path,
        )

    # Chart geometry (must match template: cl=80, cr=690, ct=30, cb=390)
    cl, cr, ct, cb = 80, 690, 30, 390
    cw, ch = cr - cl, cb - ct

    # Compute scale (include threshold in Y range if provided)
    x_min = max(0, float(cal_err.min()) - 0.01)
    x_max = float(cal_err.max()) * 1.15 + 0.01
    y_vals_max = float(fair_viol.max())
    if fairness_threshold is not None:
        y_vals_max = max(y_vals_max, fairness_threshold)
    y_min = max(0, float(fair_viol.min()) - 0.01)
    y_max = y_vals_max * 1.15 + 0.01

    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1

    def x_to_px(v):
        return int(cl + ((v - x_min) / x_range) * cw)

    def y_to_px(v):
        return int(cb - ((v - y_min) / y_range) * ch)

    # Find Pareto-optimal points (non-dominated)
    n = len(cal_err)
    is_frontier = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if x_higher_is_better:
                    # Higher X (accuracy) is better, lower Y (violation) is better
                    x_better = cal_err[j] >= cal_err[i]
                    x_strict = cal_err[j] > cal_err[i]
                else:
                    # Lower X (error) is better, lower Y (violation) is better
                    x_better = cal_err[j] <= cal_err[i]
                    x_strict = cal_err[j] < cal_err[i]
                y_better = fair_viol[j] <= fair_viol[i]
                y_strict = fair_viol[j] < fair_viol[i]
                if x_better and y_better and (x_strict or y_strict):
                    is_frontier[i] = False
                    break

    # Format values for display
    def _fmt_tick(v):
        """Format tick: use percentage if values are 0-1 range, else short decimal."""
        if x_max <= 1.5 and y_max <= 1.5:
            return f"{v * 100:.1f}%"
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"

    # Build points with smart labeling:
    # - Always label baseline and frontier points
    # - Only label dominated points if they're far enough from others
    # - Collision avoidance pushes labels up/down alternately
    points = []
    frontier_pts = []
    used_label_positions: List[Tuple[int, int]] = []  # track (px, py) of placed labels
    LABEL_MIN_DISTANCE = 26  # minimum pixels between label positions

    for i in range(n):
        px = x_to_px(cal_err[i])
        py = y_to_px(fair_viol[i])

        is_bl = (i == baseline_index) if baseline_index is not None else False
        is_fr = is_frontier[i]

        # Decide whether to show label: always for baseline/frontier,
        # for dominated only if sparse enough (fewer than 8 points)
        show_label = is_bl or is_fr or n <= 7

        # Collision avoidance: alternate push direction
        label_y_offset = 4
        if show_label:
            attempts = 0
            direction = -1  # start pushing up
            while attempts < 10:
                collision = False
                for ux, uy in used_label_positions:
                    if abs(px - ux) < 70 and abs(py + label_y_offset - uy) < LABEL_MIN_DISTANCE:
                        collision = True
                        break
                if not collision:
                    break
                label_y_offset += direction * LABEL_MIN_DISTANCE
                direction = -direction if direction < 0 else -(direction + 1)
                attempts += 1

            used_label_positions.append((px + 10, py + label_y_offset))

        points.append(
            {
                "x": px,
                "y": py,
                "label": labels[i] if show_label else "",
                "label_y_offset": label_y_offset,
                "is_frontier": is_fr,
                "is_baseline": is_bl,
                "x_val": _fmt_tick(cal_err[i]),
                "y_val": _fmt_tick(fair_viol[i]),
                "show_values": is_bl or is_fr,
            }
        )
        if is_fr:
            frontier_pts.append((cal_err[i], fair_viol[i], px, py))

    # Sort frontier by x for line
    frontier_pts.sort(key=lambda t: t[0])
    frontier_points = " ".join(f"{p[2]},{p[3]}" for p in frontier_pts)

    n_frontier = sum(is_frontier)

    # Ticks
    x_ticks = []
    y_ticks = []
    for i in range(6):
        xv = x_min + i * x_range / 5
        yv = y_min + i * y_range / 5
        x_ticks.append({"px": x_to_px(xv), "label": _fmt_tick(xv)})
        y_ticks.append({"px": y_to_px(yv), "label": _fmt_tick(yv)})

    # Best label
    best_label = ""
    if frontier_pts:
        # Pick the one closest to origin
        best_idx = min(
            range(len(frontier_pts)),
            key=lambda i: frontier_pts[i][0] ** 2 + frontier_pts[i][1] ** 2,
        )
        # Find matching label
        best_cal = frontier_pts[best_idx][0]
        best_fair = frontier_pts[best_idx][1]
        for i in range(n):
            if cal_err[i] == best_cal and fair_viol[i] == best_fair:
                best_label = labels[i]
                break

    # Threshold line pixel position
    threshold_px = None
    threshold_label = None
    if fairness_threshold is not None and y_min <= fairness_threshold <= y_max:
        threshold_px = y_to_px(fairness_threshold)
        threshold_label = _fmt_tick(fairness_threshold)

    subtitle = "Red = Pareto-optimal. Blue = dominated configurations."
    if n_dropped:
        subtitle = (
            f"{n_dropped} of {n_supplied} configuration(s) have non-finite coordinates "
            "and are excluded."
        )

    data = {
        "title": f"Pareto Frontier: {x_label} vs. {y_label}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "x_label": x_label,
        "y_label": y_label,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "points": points,
        "frontier_points": frontier_points,
        "dominated_region": "",  # simplified
        "current_point": None,
        "n_points": n,
        "n_frontier": n_frontier,
        # CARRIED OUT, not only written into the subtitle prose. `n_points` is
        # the count that SURVIVED, so "2 of 2 configurations are Pareto-optimal"
        # in the accessible <desc> reads as full coverage of everything the
        # caller supplied. Measured 2026-09-30 with three configurations, one of
        # them non-finite: the canvas said "1 of 3 configuration(s) have
        # non-finite coordinates and are excluded" and the <desc> said "2 of 2
        # configurations are Pareto-optimal; best trade-off: a. (severity:
        # INFO)". The <desc> and the <metadata> are the whole artifact for a
        # screen-reader user and for anything that parses the SVG, so the
        # partial run had to reach _fr_pareto, which could not see it.
        #
        # TOTAL loss was already refused (see _pareto_not_assessable); it was
        # the PARTIAL case that passed.
        "n_supplied": n_supplied,
        "n_dropped": n_dropped,
        "best_label": best_label,
        "threshold_px": threshold_px,
        "threshold_label": threshold_label,
        "not_assessable": False,
        "explanation": explanation,
    }
    svg = render_svg("pareto_frontier", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _pareto_not_assessable(
    reason: str,
    *,
    x_label: str,
    y_label: str,
    explanation: Optional[str],
    save_path: Optional[str],
) -> str:
    """The third state: an axis pair with nothing plottable on it.

    Every count is omitted rather than set to 0. The template's badges and
    summary key on ``not_assessable`` and print "N/A" in slate, so the canvas
    neither recommends a configuration nor reports that a search came back
    empty, and ``not_assessable`` carries the same fact into the accessible
    ``<desc>`` through ``rendering.explain._not_assessable``.
    """
    data: Dict[str, Any] = {
        "title": f"Pareto Frontier: {x_label} vs. {y_label}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": f"{NOT_ASSESSABLE_TEXT}: {reason}",
        "x_label": x_label,
        "y_label": y_label,
        "x_ticks": [],
        "y_ticks": [],
        "points": [],
        "frontier_points": "",
        "dominated_region": "",
        "current_point": None,
        "n_points": None,
        "n_frontier": None,
        "best_label": "",
        "threshold_px": None,
        "threshold_label": None,
        "not_assessable": True,
        "not_assessable_text": NOT_ASSESSABLE_TEXT,
        "not_assessable_reason": reason,
        "unknown_color": _UNKNOWN_COLOR,
        "unknown_bg": _UNKNOWN_BG,
        "explanation": explanation,
    }
    svg = render_svg("pareto_frontier", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
