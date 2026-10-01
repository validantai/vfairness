"""
Adapters for feature-engineering visualization SVG templates.

Each adapter transforms feature-engineering data objects into the flat
data dict expected by the corresponding Jinja2 SVG template.
"""

import math
from datetime import datetime
from typing import Any, Dict, Optional

from .._triage import is_flag
from .engine import render_svg


def _mark_not_assessable(data: Dict[str, Any], reason: str, headline: str) -> None:
    """Arm the third state on one chart's data dict: could-not-check.

    CRITICAL, three states, never two. A count of zero is not a finding of zero,
    and a sentinel is not a measurement. Every adapter in this module scanned a
    collection and then reported the RESULT of the scan even when the collection
    was empty: ``transformation_comparison_to_svg({}, {})`` printed the verdict
    word "Minimal" beside AVG REDUCTION 0% and FEATURES IMPROVED 0 / 0 across
    zero features, ``intersectional_disparity_to_svg({})`` printed a banded
    "DISPARITY: MEDIUM" and a 0.0pp gap "between N/A and N/A" across zero
    subgroups, and both correlation charts printed the reassuring phrase "No
    high correlations detected" for a scan across zero features.

    Setting all three keys here means the canvas and the accessible ``<desc>``
    cannot disagree: the template swaps its verdict stack for the shared
    ``_could_not_check.svg`` panel, and ``rendering.explain._not_assessable``
    reads the SAME flag to replace the finding sentence and the action line.
    Setting only one of the two flags is what let a chart say COULD NOT CHECK in
    one place and print a verdict in the other.
    """
    data["not_assessable"] = True
    data["not_assessable_reason"] = reason
    data["na_headline"] = headline


def _finite_number(value: Any) -> Optional[float]:
    """Coerce any reported quantity to a real, finite float, or None.

    CRITICAL, per ROW as well as per chart. None, NaN, inf and anything that is
    not a number all give None, which means "this row reported nothing" and is
    NOT the number zero. Every adapter in this module reached its numbers through
    a ``.get(key, 0)`` default, so a row that carried only its own name was
    written onto the canvas as a measured zero: a feature with no AFTER
    correlation scored a PERFECT 100% reduction, an intersection cell that was
    never populated scored 0.00 and became the minimum operand of the headline
    disparity, and a proxy candidate with no correlation scored r=0.00 and was
    cleared as LOW risk. NEVER reintroduce a numeric default on any of those
    paths: absent and zero are different claims, and only the second one is a
    measurement.

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


# 1.  Correlation Heatmap


def _finite_corr(value: Any) -> Optional[float]:
    """Coerce a correlation to a real, finite float, or None.

    None, NaN, inf and anything that is not a number all give None, which is
    "this pair was not computed" and NOT the number zero. Kept separate from the
    cell styling so that no caller can accidentally style an absence.
    """
    return _finite_number(value)


def _corr_cell_style(value: float):
    """Return bg + text colour for a correlation cell."""
    v = abs(value)
    if v >= 0.7:
        return ("#fecaca" if value > 0 else "#bfdbfe"), "#991b1b" if value > 0 else "#1e3a8a"
    if v >= 0.5:
        return ("#fed7aa" if value > 0 else "#93c5fd"), "#78350f" if value > 0 else "#1e40af"
    if v >= 0.3:
        return ("#fef9c3" if value > 0 else "#dbeafe"), "#713f12" if value > 0 else "#1e40af"
    return "#f8fafc", "#64748b"


def correlation_heatmap_to_svg(
    correlation_matrix,
    *,
    threshold: float = 0.3,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render a feature × protected-attribute correlation heatmap as SVG.

    Parameters
    ----------
    correlation_matrix : FeatureCorrelationMatrix | DataFrame | dict-of-dicts
        The correlation matrix from feature engineering analysis. Supported
        shapes, all normalised by ``_resolve_matrix_data``:
        - ``FeatureCorrelationMatrix`` (``preprocessing.feature_engineering.
          correlation``): ``.correlations`` DataFrame, rows ``.feature_names``,
          columns ``.protected_attributes``
        - an older object exposing ``.correlations`` as a dict-of-dicts with
          ``.features`` and ``.protected_attributes``
        - a pandas DataFrame, rows = features, columns = protected attributes
        - a dict-of-dicts ``{feature: {attribute: float, ...}, ...}``
        - ``None``, which is the absence of a matrix and renders could-not-check
    threshold : float
        Correlation threshold for proxy variable detection.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # CRITICAL, do not reintroduce `correlation_matrix.features`. The real
    # FeatureCorrelationMatrix this docstring names exposes `.feature_names`,
    # never `.features`, so that attribute access raised AttributeError for
    # EVERY real matrix, populated as well as empty: this adapter had never once
    # rendered the type it advertises, and the comment that used to sit here
    # asserted the opposite. `_resolve_matrix_data` is this module's existing
    # normaliser and already reads both attribute names, correlations held as a
    # DataFrame or as a dict-of-dicts, a bare DataFrame, a bare dict-of-dicts,
    # and None. Reusing it fixes the advertised shape without trading away the
    # older `.features` one, which the gallery fixture still uses.
    correlations, features, protected_attrs, _method = _resolve_matrix_data(correlation_matrix)
    features = features[:10]  # cap for display
    protected_attrs = protected_attrs[:6]

    rows = []
    n_high_corr = 0
    n_uncomputed = 0
    for feat in features:
        # A cell's value is Optional now: None is "not computed", which is a
        # different thing from any float, zero included.
        cells: list[Dict[str, Any]] = []
        for attr in protected_attrs:
            # `_resolve_matrix_data` has already flattened a DataFrame to a
            # dict-of-dicts keyed row-then-column, so the .get path is the live
            # one; the .loc branch stays as a guard for any future normaliser
            # result that is still frame-like.
            try:
                raw = (
                    correlations.loc[feat, attr]
                    if hasattr(correlations, "loc")
                    else correlations.get(feat, {}).get(attr, None)
                )
            except (KeyError, TypeError):
                raw = None
            val = _finite_corr(raw)
            # CRITICAL, three states per CELL, never two. This guard used to read
            # `if val != val: val = 0.0`, and the .get default was 0.0 as well, so
            # a correlation that could not be computed (a constant column, a
            # zero-variance feature, a pair the matrix simply never carried) was
            # rewritten as a measured 0.00, painted in the neutral cell style,
            # counted into "0 HIGH CORRELATIONS" and cleared by "No high
            # correlations detected". That is a proxy-risk all-clear issued for a
            # pair nobody ever tested. NEVER coerce an absent or non-finite
            # correlation to a number here: absent and zero are different, and
            # only a computed cell may enter the count.
            if val is None:
                n_uncomputed += 1
                # `|f2` prints None as N/A; slate is the could-not-check colour
                # used across this surface, not a value on the correlation scale.
                cells.append({"value": None, "bg": "#f1f5f9", "text_color": "#94a3b8"})
                continue
            bg, text_color = _corr_cell_style(val)
            if abs(val) >= threshold:
                n_high_corr += 1
            cells.append({"value": val, "bg": bg, "text_color": text_color})
        rows.append({"label": str(feat)[:22], "cells": cells})

    n_cells = len(features) * len(protected_attrs)
    n_computed = n_cells - n_uncomputed

    subtitle = (
        f"Correlations between features and protected attributes. Threshold: |r| ≥ {threshold:.2f}."
    )
    # The emerald "No high correlations detected" pill fires on n_high_corr == 0,
    # and a cell that was never computed cannot raise that count. Naming the
    # uncomputed cells in the subtitle keeps the all-clear from being read as
    # covering pairs that were never tested against the threshold. Kept terse on
    # purpose: this subtitle is a single unwrapped <text> run 644px wide, and a
    # longer sentence runs off the right edge of the canvas.
    if n_uncomputed:
        subtitle += f" {n_uncomputed} of {n_cells} pair(s) not computed, so not tested."

    data = {
        "title": "Feature to Protected Attribute Correlations",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "rows": rows,
        "col_labels": [str(a)[:12] for a in protected_attrs],
        "n_rows": len(rows),
        "n_cols": len(protected_attrs),
        "threshold": threshold,
        "n_high_corr": n_high_corr,
        "n_features": len(features),
        "risk_color": "#dc2626" if n_high_corr > 2 else "#f59e0b" if n_high_corr > 0 else "#059669",
    }

    # A scan over no feature, or over no protected attribute, computed no
    # correlation at all. "No high correlations detected" in emerald is then a
    # proxy-risk all-clear issued by a scan that never ran. See
    # _mark_not_assessable.
    if not rows or not protected_attrs:
        _mark_not_assessable(
            data,
            "The matrix carried no feature or no protected attribute, so no "
            "correlation was computed and no proxy relationship was looked for.",
            "No correlation was computed",
        )
    elif not n_computed:
        # A grid full of cells none of which is a number is the same absence
        # wearing a table: every pair failed to compute, so nothing was tested
        # against the threshold and the pill would still clear the feature set.
        _mark_not_assessable(
            data,
            f"None of the {n_cells} feature-attribute pair(s) carried a finite "
            "correlation, so no pair was computed and none was tested against the "
            f"|r| >= {threshold:.2f} threshold.",
            "No correlation was computed",
        )

    data["explanation"] = explanation
    svg = render_svg("correlation_heatmap", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  Proxy Risk Assessment


# Slate, the shared could-not-check colour on this surface. It is deliberately
# neither the emerald of LOW nor the red of HIGH: a candidate nobody rated is
# not a cleared candidate.
_NOT_RATED_STYLE = ("#64748b", "#f1f5f9", "NOT RATED")


def _risk_level_style(level_str: str):
    """Map risk level to colour scheme, or to NOT RATED when there is no level.

    'critical' keeps its own label row: collapsing it into HIGH understated
    the most severe proxies in the badge and the summary counts.

    CRITICAL, three states, never two. The final ``return`` used to be the LOW
    style, so ANY unrecognised level fell into the green all-clear: a
    ProxyVariableResult carrying ``risk_level=None`` stringified to "none" and
    was rendered "LOW" with an emerald badge and counted into "LOW: n". That is
    the strongest statement this chart can make about a feature ("this is not a
    proxy for a protected attribute") issued for a feature nobody rated, and
    proxy risk is exactly the question where a false clear does the damage.
    An unrecognised level now returns NOT RATED and the caller keeps the row off
    every risk count.
    """
    level = level_str.lower()
    if level == "critical":
        return "#b91c1c", "#fee2e2", "CRITICAL"
    if level == "high":
        return "#dc2626", "#fee2e2", "HIGH"
    if level in ("medium", "moderate"):
        return "#f59e0b", "#fef3c7", "MEDIUM"
    if level == "low":
        return "#059669", "#d1fae5", "LOW"
    return _NOT_RATED_STYLE


def proxy_risk_to_svg(
    proxy_variables: list,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render proxy variable risk assessment as SVG.

    Parameters
    ----------
    proxy_variables : list of ProxyVariableResult
        List of proxy variable analysis results.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    features = []
    unrated_features = []
    risk_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # An absent candidate list is the absence of a scan, and `None[:8]` raised
    # TypeError. It belongs in the "no candidate feature was supplied" branch
    # at the foot of this function, which says so on the canvas.
    for pv in (proxy_variables or [])[:8]:
        # getattr rather than attribute access: a candidate that carries only its
        # own name is missing data, not a programming error, and it must reach
        # the NOT RATED row instead of raising out of the render.
        raw_level = getattr(pv, "risk_level", None)
        # An enum RiskLevel gives its .value; a plain string gives itself; a
        # missing level gives "None", which _risk_level_style now reads as
        # unrecognised rather than as LOW.
        risk_str = str(getattr(raw_level, "value", raw_level))
        risk_color, risk_bg, risk_label = _risk_level_style(risk_str)
        name = str(getattr(pv, "feature", ""))[:22]
        protected_attr = str(getattr(pv, "protected_attribute", "") or "")[:14]

        # An unrated candidate is held in its OWN list, not in `features`. The
        # canvas still shows the row, so the reader sees the feature was looked
        # at and not scored, but it stays out of every risk count and out of the
        # denominator of the accessible finding ("n high-risk proxy variable(s)
        # among N features", explain._fr_proxy_risk, which reads len(features)).
        # An unrated row in that denominator says the feature was assessed and
        # came back clear.
        if risk_label == _NOT_RATED_STYLE[2]:
            unrated_features.append({"name": name, "protected_attr": protected_attr})
            continue

        risk_levels[risk_label] = risk_levels.get(risk_label, 0) + 1

        # The magnitude and the confidence are separate measurements from the
        # level, and either can be absent while the level stands. Absent stays
        # None all the way to the canvas: `abs(None) -> 0` printed "r=0.00" and
        # drew a full-width emerald bar, which reads as "measured, and as far
        # from a proxy as this feature could be", and the 0.5 confidence default
        # drew a half-full bar for a confidence nobody reported.
        raw_corr = _finite_number(getattr(pv, "correlation", None))
        correlation = abs(raw_corr) if raw_corr is not None else None
        confidence = _finite_number(getattr(pv, "confidence_score", None))

        # Colour for correlation bar. Slate when there is no correlation to
        # band: the bar colours are a scale of proxy strength, and slate is not
        # a point on it.
        if correlation is None:
            corr_color = "#94a3b8"
        elif correlation >= 0.7:
            corr_color = "#dc2626"
        elif correlation >= 0.5:
            corr_color = "#f59e0b"
        elif correlation >= 0.3:
            corr_color = "#3b82f6"
        else:
            corr_color = "#059669"

        features.append(
            {
                "name": name,
                "protected_attr": protected_attr,
                "correlation": min(1.0, correlation) if correlation is not None else None,
                "raw_correlation": raw_corr,
                "confidence": min(1.0, confidence) if confidence is not None else None,
                "corr_color": corr_color,
                "conf_color": "#8b5cf6",
                "risk_level": risk_label,
                "risk_color": risk_color,
                "risk_bg": risk_bg,
            }
        )

    # Each badge carries its own x. The template used to advance a `badge_x`
    # inside its {% for %}, and a Jinja assignment does not survive the loop
    # iteration, so EVERY badge was drawn at x=36 and the last one painted (LOW,
    # last in this list) covered the others: a scan with one CRITICAL and two
    # LOW findings showed a single emerald "LOW: 2" pill. The markup carried
    # "CRITICAL: 1" the whole time, which is why a substring test never caught
    # it. Positions are computed here, where they can be tested.
    risk_counts = []
    badge_x = 36
    for label, count in [
        ("CRITICAL", risk_levels.get("CRITICAL", 0)),
        ("HIGH", risk_levels.get("HIGH", 0)),
        ("MEDIUM", risk_levels.get("MEDIUM", 0)),
        ("LOW", risk_levels.get("LOW", 0)),
    ]:
        if count > 0:
            color, bg, _ = _risk_level_style(label)
            width = max(70, count * 30 + 60)
            risk_counts.append(
                {
                    "label": label,
                    "count": count,
                    "color": color,
                    "bg": bg,
                    "x": badge_x,
                    "width": width,
                }
            )
            badge_x += width + 10

    # The ungraded count sits ON THE CANVAS beside the graded ones, not only in
    # the description: a reader who sees CRITICAL/HIGH/MEDIUM/LOW pills and no
    # fifth pill reads the four as covering every candidate.
    if unrated_features:
        n_unrated = len(unrated_features)
        risk_counts.append(
            {
                "label": _NOT_RATED_STYLE[2],
                "count": n_unrated,
                "color": _NOT_RATED_STYLE[0],
                "bg": _NOT_RATED_STYLE[1],
                "x": badge_x,
                "width": max(96, n_unrated * 30 + 86),
            }
        )

    data: Dict[str, Any] = {
        "title": "Proxy Variable Risk Assessment",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": "Features ranked by proxy risk. Correlation = |r| with protected attribute.",
        "features": features,
        "unrated_features": unrated_features,
        "n_unrated": len(unrated_features),
        # Row height is shared by both blocks, so the layout is driven by the
        # total number of rows the table draws.
        "n_table_rows": len(features) + len(unrated_features),
        "risk_counts": risk_counts,
    }

    # No feature was rated, so the empty risk-count row reads as "nothing scored
    # CRITICAL, HIGH, MEDIUM or even LOW", which is the strongest possible
    # all-clear a proxy scan can give, from a scan that examined nothing. See
    # _mark_not_assessable. Candidates that arrived WITHOUT a risk level land
    # here too: a table of nothing but NOT RATED rows has graded no feature, so
    # there is no graded subset to band and the whole chart is could-not-check.
    if not features:
        if unrated_features:
            _mark_not_assessable(
                data,
                f"None of the {len(unrated_features)} candidate feature(s) carried a proxy "
                "risk level, so no feature was rated and no feature was cleared.",
                "No feature was rated",
            )
        else:
            _mark_not_assessable(
                data,
                "No candidate feature was supplied, so no correlation with a protected "
                "attribute was measured and no feature was rated for proxy risk.",
                "No feature was rated",
            )

    data["explanation"] = explanation
    svg = render_svg("proxy_risk", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  Transformation Comparison (Before vs After)


def transformation_comparison_to_svg(
    before_correlations: Dict[str, float],
    after_correlations: Dict[str, float],
    *,
    threshold: float = 0.3,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render before/after correlation comparison as SVG.

    Parameters
    ----------
    before_correlations : dict
        Feature name → absolute correlation before transformation.
    after_correlations : dict
        Feature name → absolute correlation after transformation.
    threshold : float
        Proxy correlation threshold.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    features = []
    reductions = []
    n_improved = 0
    n_unmeasured = 0
    n_ratio_ungraded = 0

    # Either half absent altogether is the absence of that scan, not a
    # programming error: `.keys()` and `.get` raised AttributeError on None and
    # took the render down, where the "no feature correlation was supplied"
    # branch below already describes the input.
    before_correlations = before_correlations if isinstance(before_correlations, dict) else {}
    after_correlations = after_correlations if isinstance(after_correlations, dict) else {}

    for feat_name in list(before_correlations.keys())[:6]:
        # CRITICAL, per row. Both halves of the pair were read through a
        # `.get(name, 0)` default and then through abs(), so a feature the AFTER
        # scan never produced was scored as after=0.00, which is a correlation of
        # exactly zero: the reduction came out at (before - 0) / before = 1.0 and
        # the row rendered the PERFECT result 100%, in green, at the top of the
        # column. It then raised AVG REDUCTION, took a place in FEATURES IMPROVED
        # n / n, and pushed the summary band towards "Effective". The single
        # most flattering number this chart can print was being produced by the
        # ABSENCE of the measurement it reports. Neither half may default now:
        # an unmeasured pair keeps reduction None, draws no bar, and is counted
        # nowhere. Never restore a numeric default on either line.
        before = _finite_number(before_correlations.get(feat_name))
        after = _finite_number(after_correlations.get(feat_name))
        if before is None or after is None:
            n_unmeasured += 1
            features.append(
                {
                    "name": str(feat_name)[:22],
                    "measured": False,
                    "before": None,
                    "after": None,
                    "reduction": None,
                }
            )
            continue

        before = abs(before)
        after = abs(after)
        # A feature that started at |r| = 0 has no proportional reduction to
        # report: (0 - after) / 0 is undefined, and the old `else 0` wrote that
        # undefined ratio into the canvas as a measured 0% AND into the mean
        # that decides the summary band, dragging the verdict towards the red
        # "Minimal". A fabricated breach is no safer than a fabricated
        # all-clear. Both endpoints WERE measured, so the row keeps its bars;
        # only the ratio is withheld, and it enters no average.
        if before > 0:
            reduction = (before - after) / before
            reductions.append(reduction)
        else:
            reduction = None
            n_ratio_ungraded += 1
        if after < before:
            n_improved += 1

        features.append(
            {
                "name": str(feat_name)[:22],
                "measured": True,
                "before": min(1.0, before),
                "after": min(1.0, after),
                "reduction": reduction,
            }
        )

    # The average, the band and the improved count are all over the MEASURED
    # subset, so a run that compared four pairs out of five still gives a useful
    # verdict on the four. What the subset may not do is speak for the fifth:
    # n_total below is the size of the graded subset, never len(features), and
    # the ungraded count is put on the canvas next to the band.
    n_measured = len(reductions)
    avg_reduction = sum(reductions) / n_measured if reductions else 0

    if avg_reduction >= 0.3:
        s_bg, s_color, s_icon, s_text = "#d1fae5", "#059669", "✓", "Effective"
    elif avg_reduction >= 0.1:
        s_bg, s_color, s_icon, s_text = "#fef3c7", "#f59e0b", "⚠", "Moderate"
    else:
        s_bg, s_color, s_icon, s_text = "#fee2e2", "#dc2626", "✗", "Minimal"

    # Derive the average's colour from the SAME thresholds as the summary band
    # above (>= 0.3 effective/green, >= 0.1 moderate/amber, else red): the old
    # 0.2 / 0 thresholds could show a green average beside an amber band.
    avg_color = s_color

    data = {
        "title": "Correlation Before vs After Transformation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": "Red = before, Green = after. Percentage shows correlation reduction.",
        "features": features,
        "threshold": threshold,
        "avg_reduction": avg_reduction,
        "avg_color": avg_color,
        "n_improved": n_improved,
        "n_total": n_measured,
        # Both are ungraded, for different reasons, and both are named on the
        # canvas beside the band: n_unmeasured never had a pair to compare,
        # n_ratio_ungraded had one whose "before" was zero so no proportional
        # reduction exists.
        "n_unmeasured": n_unmeasured,
        "n_ratio_ungraded": n_ratio_ungraded,
        "summary_bg": s_bg,
        "summary_color": s_color,
        "summary_icon": s_icon,
        "summary_text": s_text,
    }

    # No feature was compared, so no correlation was reduced and none failed to
    # be. The band is derived from avg_reduction, and the empty mean falls to 0,
    # which lands in the last branch: a transformation nobody ran was graded
    # "Minimal" in red at desc severity HIGH. See _mark_not_assessable.
    #
    # A table of rows that all lack one half of their pair is the same absence
    # wearing a table: nothing was compared, so there is no graded subset to
    # band and the whole chart is could-not-check.
    if not features:
        _mark_not_assessable(
            data,
            "No feature correlation was supplied, so no before/after pair was compared "
            "and the effect of the transformation was never measured.",
            "No feature was compared",
        )
    elif not n_measured:
        _mark_not_assessable(
            data,
            (
                f"None of the {len(features)} feature(s) carried both a before and an "
                "after correlation, so no pair was compared and no reduction was measured."
                if n_unmeasured
                else f"Every one of the {len(features)} feature(s) started at zero "
                "correlation, so no proportional reduction is defined for any of them "
                "and the effect of the transformation was never sized."
            ),
            "No feature was compared",
        )

    data["explanation"] = explanation
    svg = render_svg("transformation_comparison", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4.  Intersectional Analysis Heatmap


def _intersection_label(y_label: Any, x_label: Any, *, multi_col: bool) -> str:
    """Name the intersection a headline operand came from.

    A cell on this grid is a pair, so on a grid with more than one column the
    row label alone does not identify it: two cells in the same row would both
    print as that row's name. The single-column case keeps the bare row label,
    which is the whole of the identity there.
    """
    y = str(y_label)[:12]
    if not multi_col:
        return y
    return f"{y} · {str(x_label)[:10]}"


def _intersect_cell_style(value: float, global_max: float):
    """Return bg and text colour for an intersectional analysis cell."""
    ratio = value / global_max if global_max > 0 else 0
    if ratio >= 0.8:
        return "#fecaca", "#991b1b", "#b91c1c"
    if ratio >= 0.6:
        return "#fed7aa", "#78350f", "#92400e"
    if ratio >= 0.4:
        return "#fde68a", "#713f12", "#92400e"
    if ratio >= 0.2:
        return "#d1fae5", "#065f46", "#065f46"
    return "#ecfdf5", "#065f46", "#047857"


def intersectional_analysis_to_svg(
    intersectional_results: Dict[str, Any],
    feature: str,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render intersectional analysis heatmap as SVG.

    Parameters
    ----------
    intersectional_results : dict
        Intersectional analysis results dict.
    feature : str
        The feature being analysed.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # Extract the structure
    # Expected: intersectional_results has 'matrix' (2D dict), 'x_attr', 'y_attr'
    #
    # An absent result, an absent matrix, or a row that is not a mapping are all
    # missing data rather than programming errors, and each of them raised
    # (AttributeError on `.get`, TypeError on iterating None) and took the whole
    # render down. They belong in the could-not-check branches at the foot of
    # this function, which say on the canvas that no intersection was populated.
    intersectional_results = (
        intersectional_results if isinstance(intersectional_results, dict) else {}
    )
    matrix = intersectional_results.get("matrix", {})
    matrix = matrix if isinstance(matrix, dict) else {}
    x_attr = intersectional_results.get("x_attr", "Attribute 1")
    y_attr = intersectional_results.get("y_attr", "Attribute 2")
    counts = intersectional_results.get("counts", {})
    counts = counts if isinstance(counts, dict) else {}

    def _row(mapping, key) -> Dict[str, Any]:
        """One row of a 2D dict, or an empty row when it is not a mapping."""
        row = mapping.get(key)
        return row if isinstance(row, dict) else {}

    y_labels = list(matrix.keys())[:6]
    x_labels = []
    for yl in y_labels:
        for xl in _row(matrix, yl):
            if xl not in x_labels:
                x_labels.append(xl)
    x_labels = x_labels[:6]

    # Find global max over the cells that were actually populated. A cell the
    # matrix never carried is None here, not 0: the `.get(xl, 0)` default it
    # replaces made an absent intersection the darkest-green, best-outcome cell
    # on the grid and dragged the colour scale with it.
    all_vals = []
    for yl in y_labels:
        for xl in x_labels:
            v = _finite_number(_row(matrix, yl).get(xl))
            if v is not None:
                all_vals.append(abs(v))
    global_max = max(all_vals) if all_vals else 1

    rows = []
    measured_cells = []  # (value, y_label, x_label) for the populated cells only
    n_unmeasured = 0
    for yl in y_labels:
        cells = []
        for xl in x_labels:
            val = _finite_number(_row(matrix, yl).get(xl))
            raw_n = _row(counts, yl).get(xl)
            if val is None:
                # Slate, and no n: an unpopulated cell has no count either, and
                # "n=0" reads as a measured empty stratum rather than as silence.
                n_unmeasured += 1
                cells.append(
                    {
                        "value": None,
                        "n": raw_n,
                        "bg": "#f1f5f9",
                        "text_color": "#94a3b8",
                        "sub_color": "#94a3b8",
                    }
                )
                continue
            n = raw_n if raw_n is not None else 0
            bg, text_color, sub_color = _intersect_cell_style(abs(val), global_max)
            cells.append(
                {
                    "value": val,
                    "n": n,
                    "bg": bg,
                    "text_color": text_color,
                    "sub_color": sub_color,
                }
            )
            measured_cells.append((val, yl, xl))
        rows.append({"label": str(yl)[:16], "cells": cells})

    # True spread across ALL POPULATED intersections (max - min). The previous
    # scan compared only consecutive cells in iteration order, understating the
    # disparity whenever the extreme cells were not adjacent.
    #
    # CRITICAL, an operand that was never measured may not name a breach. With
    # the absent cells defaulted to 0 the minimum of this scan was routinely a
    # cell nobody populated, so the headline read "MAX DISPARITY 0.720 between
    # White and ZZ_BARE" in red at desc severity HIGH: a reader is sent after a
    # violation involving a group the run never measured, and when they work out
    # why, the tool is what they stop trusting. adapters_fairness's
    # disparity_heatmap closed the same defect the same way, over measured
    # values only.
    #
    # The operands are the cells that actually hold the extremes, too. They used
    # to be y_labels[0] and y_labels[-1], the FIRST and LAST row of the grid,
    # which name the true extremes only by coincidence: on the gallery example
    # the maximum sits in row 1 and the minimum in row 2 of 4, and the canvas
    # named rows 1 and 4.
    if len(measured_cells) >= 2:
        hi = max(measured_cells, key=lambda c: c[0])
        lo = min(measured_cells, key=lambda c: c[0])
        max_disparity = hi[0] - lo[0]
        max_group_1 = _intersection_label(hi[1], hi[2], multi_col=len(x_labels) > 1)
        max_group_2 = _intersection_label(lo[1], lo[2], multi_col=len(x_labels) > 1)
    else:
        max_disparity = 0
        max_group_1 = ""
        max_group_2 = ""

    # Max color
    if max_disparity < 0.1:
        max_color = "#059669"
    elif max_disparity < 0.2:
        max_color = "#f59e0b"
    else:
        max_color = "#dc2626"

    data = {
        "title": "Intersectional Analysis Heatmap",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": f"Feature '{feature}' analysed across intersecting protected attributes.",
        "feature_name": feature,
        "x_attr": str(x_attr)[:20],
        "y_attr": str(y_attr)[:20],
        "col_labels": [str(x)[:10] for x in x_labels],
        "rows": rows,
        "n_rows": len(rows),
        "n_cols": len(x_labels),
        "max_disparity": max_disparity,
        "max_color": max_color,
        "max_group_1": max_group_1,
        "max_group_2": max_group_2,
        "n_unmeasured": n_unmeasured,
        "n_measured": len(measured_cells),
        "n_cells": len(y_labels) * len(x_labels),
    }

    # No cell means no intersection was populated, so max(...) - min(...) was
    # never taken. The 0 fallback rendered as MAX DISPARITY 0.000 in emerald,
    # which is the canvas for perfect parity across every intersection. See
    # _mark_not_assessable.
    #
    # One populated cell is the same absence with a number on it: a spread needs
    # two operands, and max - min over a single cell is 0 by arithmetic, not by
    # measurement. This chart's whole verdict is that spread, so a grid that
    # cannot produce one is could-not-check rather than a partially graded run.
    if not measured_cells:
        _mark_not_assessable(
            data,
            "No intersection cell was supplied, so no outcome rate was compared and "
            "no spread between intersections was measured.",
            "No intersection was populated",
        )
    elif len(measured_cells) < 2:
        _mark_not_assessable(
            data,
            f"Only 1 of the {len(y_labels) * len(x_labels)} intersection cell(s) carried an "
            "outcome rate, so there was no second intersection to compare it with and no "
            "spread was measured.",
            "No pair of intersections was compared",
        )

    data["explanation"] = explanation
    svg = render_svg("intersectional_analysis", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4b. Intersectional Disparity, Ranked Bar Chart

_SEVERITY_COLORS = {
    "critical": "#b91c1c",
    "high": "#dc2626",
    "medium": "#d97706",
    "low": "#64748b",
    "info": "#64748b",
}

_SEVERITY_DOT_COLORS = {
    "critical": "#dc2626",
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#94a3b8",
    "info": "#94a3b8",
}

_SEVERITY_BAR_COLORS = {
    "critical": "#b91c1c",
    "high": "#dc2626",
    "medium": "#d97706",
    "low": "#64748b",
    "info": "#94a3b8",
}

_INSIGHT_STYLES = {
    "critical": ("#b91c1c", "#fee2e2"),
    "high": ("#dc2626", "#fef2f2"),
    "medium": ("#d97706", "#fffbeb"),
    "low": ("#64748b", "#f8fafc"),
    "info": ("#3b82f6", "#eff6ff"),
}


def intersectional_disparity_to_svg(
    result: Dict[str, Any],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render intersectional disparity ranked-bar chart as SVG.

    Accepts an IntersectionalAnalysisResult-shaped dict with:
    - privilegedGroup, disadvantagedGroup (group objects with positiveRate, etc.)
    - allGroups: list of group objects sorted by positiveRate
    - maxDisparity: float (0-1)
    - disparitySeverity: str
    - insights: list of strings
    - findings: list of finding objects with severity

    Returns SVG markup string.
    """
    # An absent result is the absence of an analysis, not a programming error:
    # `result.get` raised AttributeError on None and took the caller's process
    # down, where the could-not-check branch at the foot of this function
    # already describes exactly that input.
    result = result if isinstance(result, dict) else {}
    priv = result.get("privilegedGroup") or result.get("privileged_group") or {}
    disadv = result.get("disadvantagedGroup") or result.get("disadvantaged_group") or {}
    all_groups = result.get("allGroups") or result.get("all_groups") or []
    # CRITICAL, the headline gap is a MEASUREMENT. `or 0` made a missing
    # maxDisparity into a measured 0.0pp, which is the canvas for perfect parity
    # between the two named subgroups, and a maxDisparity supplied as the STRING
    # "0.3" reached `max_disp * 100` as Python string repetition and printed 300
    # characters of "0.3" across the headline. _finite_number gives None for
    # every one of those, and None routes to the could-not-check branch below.
    max_disp = _finite_number(
        result.get("maxDisparity")
        if result.get("maxDisparity") is not None
        else result.get("max_disparity")
    )
    # The band beside the gap is a JUDGEMENT, not a restatement of the gap, and
    # nobody made it here unless the caller reported one. The `or "medium"`
    # default printed a banded "DISPARITY: MEDIUM" for a run whose severity was
    # never assessed, which manufactures a finding out of a missing field.
    raw_severity = result.get("disparitySeverity") or result.get("disparity_severity")
    severity_graded = str(raw_severity or "").lower() in _SEVERITY_COLORS
    severity = str(raw_severity).lower() if severity_graded else "not graded"
    insights_raw = result.get("insights") or []
    findings = result.get("findings") or []

    def _rate_of(g):
        """The subgroup's positive rate as a measurement, or None. NEVER 0.

        CRITICAL, per ROW. This value is the bar, the ranking key and the number
        in the RATE column, and it used to be reached through
        ``g.get("positiveRate", g.get("positive_rate", 0))``. A subgroup that
        carried only its own name therefore rendered a measured 0.0% positive
        rate, sorted to the BOTTOM of a ranking that is read as "most
        disadvantaged last", and sat there as the apparent worst-affected group
        in the chart. A rate that really came out 0.0 still ranks and still
        prints, because the test below is ``is None`` and not truthiness.
        """
        return _finite_number(
            g.get("positiveRate", g.get("positive_rate")) if isinstance(g, dict) else None
        )

    # Only a subgroup with a measured rate can be ranked. The rest keep their
    # place on the canvas but not in the ordering, so nothing is sorted by a
    # value that was invented for it.
    ranked = [g for g in all_groups if _rate_of(g) is not None]
    unranked = [g for g in all_groups if _rate_of(g) is None]
    sorted_groups = sorted(ranked, key=lambda g: _rate_of(g), reverse=True)

    # Build max rate for bar scaling, over the graded rows only.
    max_rate = max((_rate_of(g) for g in sorted_groups), default=0.01)
    if max_rate <= 0:
        max_rate = 0.01

    # Map severity for each group
    def _group_severity(g):
        """The reported severity, or None. NEVER a default of "low".

        A severity is a judgement about a subgroup, and defaulting one
        manufactures a finding: every subgroup that reported no severity was
        coloured and dotted as a LOW-severity row, which is an all-clear that
        nobody made. An unrecognised label is ungraded for the same reason, so a
        typo cannot be read as a grade either.
        """
        s = g.get("severity") if isinstance(g, dict) else None
        s = str(s).lower() if s is not None else None
        return s if s in _SEVERITY_COLORS else None

    groups = []
    for i, g in enumerate(sorted_groups):
        rate = _rate_of(g)
        # The ground-truth rate used to default to the PREDICTED rate, which
        # draws the grey ground-truth bar exactly underneath the coloured
        # prediction bar and reads as "prediction matches reality exactly", the
        # strongest all-clear this row can give, for a row that reported no
        # ground truth at all.
        gt = _finite_number(g.get("groundTruthRate", g.get("ground_truth_rate")))
        delta = _finite_number(g.get("predictionDelta", g.get("prediction_delta")))
        n = _finite_number(g.get("size", g.get("n")))
        sev = _group_severity(g)
        groups.append(
            {
                "rank": i + 1,
                "name": str(g.get("group", ""))[:24],
                "rate": rate,
                "ground_truth": gt,
                # percentage points, withheld when no delta was reported
                "delta_pp": delta * 100 if delta is not None else None,
                "n": int(n) if n is not None else None,
                "severity": sev,
                "severity_graded": sev is not None,
                # Slate-200 is not a point on the severity scale; it is the
                # absence of one, and it is paired with a hollow dot so the row
                # cannot be mistaken for a graded LOW.
                "bar_color": _SEVERITY_BAR_COLORS.get(sev, "#cbd5e1"),
                "dot_color": _SEVERITY_DOT_COLORS.get(sev, "#cbd5e1"),
            }
        )

    # Subgroups with no positive rate. They are listed, so the reader sees the
    # subgroup was named and never measured, and they hold no rank, no bar, no
    # severity colour and no place in the subtitle count or in max_rate.
    ungraded_groups = []
    for g in unranked:
        if isinstance(g, dict):
            n = _finite_number(g.get("size", g.get("n")))
            ungraded_groups.append(
                {"name": str(g.get("group", ""))[:24], "n": int(n) if n is not None else None}
            )
        else:
            ungraded_groups.append({"name": str(g)[:24], "n": None})

    # Build insight cards
    insight_cards = []
    # Derive severity for insights from findings. A finding that is not a dict
    # carries no severity to read, and `f.get` raised AttributeError on it; the
    # neutral "info" style is this card's could-not-check colour, so an
    # unreadable finding lands there rather than ending the render.
    finding_severities = [
        (f.get("severity", "info") if isinstance(f, dict) else "info") for f in findings
    ]
    for i, text in enumerate(insights_raw[:6]):  # cap at 6
        sev = finding_severities[i] if i < len(finding_severities) else "info"
        sev = str(sev).lower() if str(sev).lower() in _INSIGHT_STYLES else "info"
        color, bg = _INSIGHT_STYLES[sev]
        insight_cards.append({"text": str(text), "color": color, "bg": bg})

    # Slate when the severity was never graded. #d97706 is amber, the MEDIUM
    # colour, so the old default painted the cards, the connector and the pill
    # in the colour of a mid-severity finding as well as labelling one.
    sev_color = _SEVERITY_COLORS.get(severity, "#94a3b8")

    def _card(grp):
        """One comparison card, with every cell withheld unless it was reported.

        ``grp.get("positiveRate", grp.get("positive_rate", 0))`` printed a
        measured 0.0% positive rate for a named subgroup that carried no rate,
        and the ACTUAL field defaulted to the PREDICTED one, so the card read
        "0.0% predicted | 0.0% actual" and claimed perfect agreement with a
        ground truth nobody supplied. ``n`` defaulted to 0, which is a sample
        size, and a sample size of zero is a claim about the data.
        """
        if not isinstance(grp, dict):
            return {"group": "N/A", "rate": None, "predicted": None, "actual": None, "n": None}
        rate = _finite_number(grp.get("positiveRate", grp.get("positive_rate")))
        actual = _finite_number(grp.get("groundTruthRate", grp.get("ground_truth_rate")))
        n = _finite_number(grp.get("size", grp.get("n")))
        return {
            "group": str(grp.get("group", "N/A"))[:28],
            "rate": rate,
            "predicted": rate,
            "actual": actual,
            "n": int(n) if n is not None else None,
        }

    _single_attr_gt = _finite_number(
        result.get("maxGroundTruthDisparity")
        if result.get("maxGroundTruthDisparity") is not None
        else result.get("max_ground_truth_disparity")
    )

    # Build template data
    data = {
        "title": "Intersectional Disparity Analysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Part (d): the count is over the GRADED subgroups only. It used to
        # count every row supplied, including rows whose 0.0% rate was this
        # adapter's own default, so "7 subgroups analysed" covered subgroups
        # that were never analysed. The ungraded ones are named on the canvas
        # beside this line (part (c)); they are kept out of this sentence only
        # so that a long list cannot push the subtitle past the 680 canvas.
        "subtitle": (
            "Compound disparities across intersecting protected attributes. "
            f"{len(sorted_groups)} subgroups analysed."
        ),
        "privileged": _card(priv),
        "disadvantaged": _card(disadv),
        # percentage points, withheld when no gap was reported
        "disparity_pp": max_disp * 100 if max_disp is not None else None,
        "severity": severity.upper(),
        "severity_graded": severity_graded,
        "severity_color": sev_color,
        # The single-attribute comparison figure. The old expression tested the
        # value for TRUTH before using it, so a ground-truth disparity that was
        # really measured at 0.0 (perfect single-attribute parity, the whole
        # point of the comparison) was withheld as though it had never been
        # reported, while a non-numeric value reached `* 100` and printed as
        # repeated text. `is not None` keeps a measured zero, and
        # _finite_number refuses everything that is not a real number.
        "single_attr_max_pp": (_single_attr_gt * 100) if _single_attr_gt is not None else None,
        "groups": groups,
        # Held OUT of `groups`, so an unmeasured subgroup is in no count, in no
        # ranking and in no bar scale, and cannot reach a numerator anywhere.
        "ungraded_groups": ungraded_groups,
        "n_ungraded": len(ungraded_groups),
        "max_rate": max_rate,
        "insights": insight_cards,
        "explanation": explanation,
    }

    # Three states, never two. With no subgroup, or with no privileged or
    # disadvantaged subgroup named, nothing was compared: the severity default
    # of "medium" printed a banded "DISPARITY: MEDIUM" beside "0.0pp between N/A
    # and N/A". A band is a verdict about a measured gap, and N/A beside it is
    # the same contradiction the calibration charts carried. Note the severity
    # is only trusted when a subgroup ranking actually exists; the caller's
    # `disparitySeverity` cannot certify a comparison that had no operands.
    # See _mark_not_assessable.
    #
    # The test is on the COERCED gap, not on the key's presence: a maxDisparity
    # supplied as a string, a NaN or an inf is a key that is present and a gap
    # that was never sized, and it used to satisfy this guard and then print
    # N/A under a headline claiming a measured comparison.
    has_disparity = max_disp is not None
    if not sorted_groups or not priv or not disadv or not has_disparity:
        _mark_not_assessable(
            data,
            "No intersectional subgroup pair was supplied with a measured disparity, so "
            "no privileged and disadvantaged subgroup were compared and no gap was sized.",
            "No subgroup pair was compared",
        )

    svg = render_svg("intersectional_disparity", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 5.  General Correlation Matrix (multi-method)

_METHOD_COLORS = {
    "pearson": ("#3b82f6", "#dbeafe"),  # blue
    "spearman": ("#8b5cf6", "#ede9fe"),  # violet
    "cramers_v": ("#f59e0b", "#fef3c7"),  # amber
    "mutual_information": ("#059669", "#d1fae5"),  # emerald
    "point_biserial": ("#ec4899", "#fce7f3"),  # pink
}

_METHOD_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "cramers_v": "Cramér's V",
    "mutual_information": "Mutual Info",
    "point_biserial": "Point-Biserial",
}


def _union_columns(corr) -> list:
    """Every column any row of a dict-of-dicts carries, in first-seen order.

    This was ``list(next(iter(corr.values())).keys())``, the columns of the
    FIRST row only, so a ragged matrix silently LOST every column the first row
    happened not to carry: a pair the caller really did report never reached the
    grid, was in no count, and left no trace of having been dropped. Taking the
    union puts the pair on the grid, where the per-cell rule then renders the
    rows that lack it as not computed rather than as a measured 0.00.
    """
    cols: list = []
    if not isinstance(corr, dict):
        return cols
    for row in corr.values():
        if not isinstance(row, dict):
            continue
        for c in row:
            if c not in cols:
                cols.append(c)
    return cols


def _resolve_matrix_data(data):
    """Normalise input to (dict-of-dicts, row_labels, col_labels, method).

    Accepts pandas DataFrame, dict-of-dicts, or FeatureCorrelationMatrix-like.
    ``None`` normalises to an empty scan rather than raising: it is not an
    unsupported TYPE, it is the absence of a matrix, and the caller gets a
    could-not-check canvas that says so. Every other unsupported type still
    raises, because that is a programming error rather than missing data.
    """
    if data is None:
        return {}, [], [], None

    # pandas DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            row_labels = list(data.index)
            col_labels = list(data.columns)
            corr_dict = {}
            for r in row_labels:
                corr_dict[r] = {}
                for c in col_labels:
                    corr_dict[r][c] = float(data.loc[r, c])
            return corr_dict, row_labels, col_labels, None
    except ImportError:
        pass

    # FeatureCorrelationMatrix-like object
    if hasattr(data, "correlations"):
        corr = data.correlations
        try:
            import pandas as pd

            if isinstance(corr, pd.DataFrame):
                row_labels = list(corr.index)
                col_labels = list(corr.columns)
                corr_dict = {}
                for r in row_labels:
                    corr_dict[r] = {}
                    for c in col_labels:
                        corr_dict[r][c] = float(corr.loc[r, c])
                return corr_dict, row_labels, col_labels, getattr(data, "method", None)
        except ImportError:
            pass
        # dict-of-dicts inside the object
        row_labels = (
            getattr(data, "features", None)
            or getattr(data, "feature_names", None)
            or list(corr.keys())
        )
        col_labels = getattr(data, "protected_attributes", None) or _union_columns(corr)
        return corr, row_labels, col_labels, getattr(data, "method", None)

    # Plain dict-of-dicts
    if isinstance(data, dict):
        row_labels = list(data.keys())
        col_labels = _union_columns(data)
        return data, row_labels, col_labels, None

    raise TypeError(f"Unsupported correlation data type: {type(data)}")


def _method_key(raw) -> Optional[str]:
    """Normalise one reported correlation method to a key, or None.

    None, NaN and the empty string all give None, which is "this cell did not
    record which test was run" and is NOT the library default. A method name is
    a claim about how a number was produced, so it may not be supplied by this
    module on the caller's behalf.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and not math.isfinite(raw):
        return None
    m = str(raw).strip().lower()
    if m in ("", "nan", "none"):
        return None
    return m


def _resolve_methods(methods, row_labels, col_labels):
    """Normalise methods to (dict-of-dicts, unique_methods_set).

    A cell may map to None, meaning no method was recorded for it. That is a
    third state and it is kept out of ``unique_methods``, so it can neither
    name the matrix's METHOD badge nor be counted in a legend entry.
    """
    if methods is None:
        methods = "pearson"

    if isinstance(methods, str):
        m = methods.lower()
        d = {r: {c: m for c in col_labels} for r in row_labels}
        return d, {m}

    # DataFrame
    try:
        import pandas as pd

        if isinstance(methods, pd.DataFrame):
            d, unique = {}, set()
            for r in row_labels:
                d[r] = {}
                for c in col_labels:
                    try:
                        raw = methods.loc[r, c]
                    except (KeyError, TypeError, IndexError):
                        raw = None
                    m = _method_key(raw)
                    d[r][c] = m
                    if m is not None:
                        unique.add(m)
            return d, unique
    except ImportError:
        pass

    # dict-of-dicts
    if isinstance(methods, dict):
        d, unique = {}, set()
        for r in row_labels:
            d[r] = {}
            for c in col_labels:
                # CRITICAL, a cell whose method was not recorded gets None, never
                # "pearson". The old default named a statistic for every cell the
                # caller left out, so a partially reported methods map printed
                # "Pearson" in the METHOD badge and put every unrecorded cell
                # into the Pearson legend count: a claim about WHICH test was run
                # on a pair where none was reported. None flows to a slate dot,
                # no legend count, and its own "not recorded" tally.
                raw = methods.get(r, {}).get(c) if isinstance(methods.get(r), dict) else None
                m = _method_key(raw)
                d[r][c] = m
                if m is not None:
                    unique.add(m)
        return d, unique

    raise TypeError(f"Unsupported methods type: {type(methods)}")


def correlation_matrix_to_svg(
    data,
    *,
    methods=None,
    threshold: float = 0.5,
    title: str = "Feature Correlation Matrix",
    max_rows: int = 12,
    max_cols: int = 12,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """Render a general-purpose correlation matrix as SVG.

    Parameters
    ----------
    data : DataFrame | dict-of-dicts | FeatureCorrelationMatrix
        Correlation values.  Supported formats:
        - pandas DataFrame (row=feature, col=feature or attribute)
        - dict-of-dicts ``{row_label: {col_label: float, ...}, ...}``
        - ``FeatureCorrelationMatrix`` object
    methods : str | DataFrame | dict-of-dicts, optional
        Which correlation method was used per cell.
        - ``None`` or single string → uniform for all cells
        - DataFrame or dict-of-dicts matching shape → per-cell
    threshold : float
        Highlight cells with ``|r| >= threshold``.
    title : str
        Title text.
    max_rows : int
        Maximum rows to display.
    max_cols : int
        Maximum columns to display.
    explanation : str
        Explanation panel text.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # 1. Resolve data
    corr_dict, row_labels, col_labels, detected_method = _resolve_matrix_data(data)
    row_labels = row_labels[:max_rows]
    col_labels = col_labels[:max_cols]

    # 2. Resolve methods
    if methods is None and detected_method is not None:
        methods = detected_method
    methods_dict, unique_methods = _resolve_methods(methods, row_labels, col_labels)
    # `next(iter(...))` on an empty set raised StopIteration, which is not even
    # an error a caller can act on: `correlation_matrix_to_svg({}, methods={})`
    # died here before it could reach the could-not-check canvas that describes
    # exactly that input. A matrix that recorded no method anywhere says so.
    if not unique_methods:
        method_display = "not recorded"
    elif len(unique_methods) > 1:
        method_display = "Mixed"
    else:
        only = next(iter(unique_methods))
        method_display = _METHOD_LABELS.get(only, only.title())

    # 3. Build rows
    rows = []
    n_high_corr = 0
    n_uncomputed = 0
    n_pairs = 0
    n_method_unrecorded = 0
    method_counts: Dict[str, int] = {}
    for rl in row_labels:
        cells = []
        for cl in col_labels:
            is_diagonal = rl == cl
            # CRITICAL, three states per CELL, never two. This read
            # `float(corr_dict.get(rl, {}).get(cl, 0.0))`, which did two wrong
            # things at once. A pair explicitly reported as None crashed the
            # whole render on `float(None)`, so one unreportable pair cost the
            # entire matrix. And a pair the matrix never carried was written
            # onto the canvas as a measured 0.00, in the neutral cell style,
            # counted as tested against the threshold and cleared by the emerald
            # "No high correlations detected" pill. On a square matrix that also
            # made the grid contradict ITSELF: the missing (a, b) cell read 0.00
            # beside a mirror (b, a) cell reading 0.72. Absent is not zero, and
            # only a computed cell may enter n_high_corr. NEVER restore a numeric
            # default on this line.
            val = _finite_corr(
                corr_dict.get(rl, {}).get(cl) if isinstance(corr_dict, dict) else None
            )
            if not is_diagonal:
                n_pairs += 1
            method_key = methods_dict.get(rl, {}).get(cl)
            if method_key is None:
                # Slate, not a method colour: the dot legend is a scale of which
                # test was run, and "nobody said" is not a point on it.
                method_color = "#94a3b8"
                # Counted over EVERY cell, the same population `method_counts`
                # has always used, so the legend's entries add up.
                n_method_unrecorded += 1
            else:
                method_color, _ = _METHOD_COLORS.get(method_key, ("#64748b", "#f1f5f9"))
                method_counts[method_key] = method_counts.get(method_key, 0) + 1
            if val is None and is_diagonal:
                # THE DESCRIPTION MUST AGREE WITH THE CANVAS. correlation_matrix
                # .svg draws every diagonal cell as a muted "1.00" without ever
                # reading this value, because corr(x, x) = 1 is an identity and
                # not a measurement. `explain._fr_corr_matrix` counts a cell with
                # value None as "not computed", straight off these same rows, so
                # leaving the diagonal None would have the accessible text report
                # an uncomputed pair that the canvas shows as 1.00. The two must
                # not be able to disagree, and the canvas is right here.
                val = 1.0
            if val is None:
                n_uncomputed += 1
                # `|f2` prints None as N/A; slate is the could-not-check colour
                # used across this surface, not a value on the correlation scale.
                bg, text_color = "#f1f5f9", "#94a3b8"
            else:
                bg, text_color = _corr_cell_style(val)
                if abs(val) >= threshold and not is_diagonal:
                    n_high_corr += 1
            cells.append(
                {
                    "value": val,
                    "bg": bg,
                    "text_color": text_color,
                    "method": method_key,
                    "method_color": method_color,
                    "is_diagonal": is_diagonal,
                }
            )
        rows.append({"label": str(rl)[:22], "cells": cells})

    n_computed = n_pairs - n_uncomputed
    # The legend band is the only place on the canvas that names which test
    # produced which cell, so it has to render whenever ANY cell is unaccounted
    # for, not only when two named methods disagree.
    is_mixed = len(unique_methods) > 1 or bool(n_method_unrecorded)

    # 4. Legend items
    legend_items = []
    for m in sorted(unique_methods):
        color, bg = _METHOD_COLORS.get(m, ("#64748b", "#f1f5f9"))
        legend_items.append(
            {
                "label": _METHOD_LABELS.get(m, m.title()),
                "color": color,
                "bg": bg,
                "count": method_counts.get(m, 0),
            }
        )
    if n_method_unrecorded:
        legend_items.append(
            {
                "label": "Not recorded",
                "color": "#94a3b8",
                "bg": "#f1f5f9",
                "count": n_method_unrecorded,
            }
        )

    # 5. Risk colour for badge
    if n_high_corr > 4:
        risk_color = "#dc2626"
    elif n_high_corr > 0:
        risk_color = "#f59e0b"
    else:
        risk_color = "#059669"

    # 6. Template data
    n_features = len(row_labels)
    if set(row_labels) != set(col_labels):
        n_features = len(set(row_labels) | set(col_labels))

    # The emerald "No high correlations detected" pill fires on n_high_corr == 0,
    # and a pair that was never computed cannot raise that count. Naming the
    # uncomputed pairs in the subtitle keeps the all-clear from being read as
    # covering pairs that were never tested. Kept terse: this subtitle is a
    # single unwrapped <text> run at x=36 on a 680 canvas, and it sits in the
    # headline band (y=122), which is where the ungraded count belongs.
    subtitle = f"{method_display} correlations. Threshold: |r| >= {threshold:.2f}."
    if n_uncomputed:
        subtitle += f" {n_uncomputed} of {n_pairs} pair(s) not computed, so not tested."

    template_data = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "rows": rows,
        "col_labels": [str(c)[:12] for c in col_labels],
        "n_rows": len(rows),
        "n_cols": len(col_labels),
        "n_features": n_features,
        "n_high_corr": n_high_corr,
        "n_uncomputed": n_uncomputed,
        "n_computed": n_computed,
        "n_pairs": n_pairs,
        "threshold": threshold,
        "risk_color": risk_color,
        "method_display": method_display,
        "is_mixed": is_mixed,
        "legend_items": legend_items,
        "is_square": (row_labels == col_labels),
        "explanation": explanation,
    }

    # An empty (or absent) matrix produced no cell, so no pair was tested
    # against the threshold. "No high correlations detected" in emerald is a
    # redundancy and proxy-risk all-clear, and here it certifies a scan across
    # zero features. See _mark_not_assessable.
    if not rows or not col_labels:
        _mark_not_assessable(
            template_data,
            "The correlation matrix was empty, so no feature pair was computed and "
            f"none was tested against the |r| >= {threshold:.2f} threshold.",
            "No feature pair was tested",
        )
    elif not n_computed:
        # A grid whose every off-diagonal cell is blank is the same absence
        # wearing a table: a 1x1 matrix carries no pair at all, and a larger one
        # whose pairs all failed to compute tested nothing against the
        # threshold, so the emerald pill would still clear the feature set.
        _mark_not_assessable(
            template_data,
            (
                "This matrix carries no feature pair at all, only its own diagonal, "
                if not n_pairs
                else f"None of the {n_pairs} feature pair(s) in this matrix carried a "
                "finite correlation, "
            )
            + "so no pair was computed and none was tested against the "
            + f"|r| >= {threshold:.2f} threshold.",
            "No feature pair was tested",
        )

    svg = render_svg("correlation_matrix", template_data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
