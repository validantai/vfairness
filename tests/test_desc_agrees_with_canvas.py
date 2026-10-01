"""The accessible description must agree with the canvas it describes.

Every chart this library renders carries two accounts of the same run: the
CANVAS, which a sighted reader checks, and the ``<desc>`` that
``rendering.explain`` injects, which is the whole artifact for a screen-reader
user and for every downstream parser. Fourteen waves hardened the canvas per
row. The description was hardened per SITE, so it drifted: whichever of the two
surfaces is more confident is the one a reader believes, and it was repeatedly
the one the audience least able to notice gets.

Reproduced on the committed tree at 4608730, each line below is what the
``<desc>`` carried while the canvas beside it already said otherwise:

    hierarchical_gate    ->  "single-attr 1/2 metrics passed" beside a canvas
                             reading "PASS 1/1 checks, 1 not checked". The
                             template re-derives every level's denominator from
                             the rows, taking out the metrics the gate never
                             graded; the finder used the RAW adapter totals, so
                             the sentence reported a measured failure where the
                             canvas reported full marks over what it measured.
    report_card          ->  "1/3 metrics pass" for a card whose third row is
                             badged NOT GRADED in slate and prints "not
                             measured" where its value would be. This is the
                             surface that gets pasted into a pull request.
    disparity_heatmap    ->  "across 3 group(s) and 4 metric(s)" for a grid
                             drawing one of those groups entirely N/A.
    robustness_testing   ->  "Robustness ROBUST (score 0.85); None/0 subgroups
                             flagged." A Python None formatted into a fraction
                             over a zero denominator.
    tradeoff_analysis    ->  "0 of 5 configurations satisfy the constraint" at
                             severity INFO, contradicting its own headline: the
                             adapter puts the withheld count in the TITLE
                             because this template's subtitle is fixed prose.
    correlation_matrix   ->  "0 feature pair(s) exceed |r| >= 0.30" over pairs
                             the grid draws N/A, the same all-clear the
                             heatmap beside it had already had removed.
    drift_report         ->  "Drift not detected" at severity INFO for a result
                             that recorded no overall verdict, beside a badge
                             the adapter had already given a third state.
    causal_decomposition ->  "0/4 Baron-Kenny steps satisfied" at severity LOW:
                             the band assumed 4 for an absent count while the
                             text printed 0, so one sentence contradicted
                             itself.
    reporting_dashboard  ->  "0 breach(es) of 1" at severity INFO with the
                             unchecked row named only in the narrative panel at
                             the foot of the canvas.

The guards below are deliberately GENERAL rather than ten one-off assertions,
because the campaign's failure mode is fixing the site it was pointed at. Every
case is run through the same four rules, every case carries a HEALTHY TWIN whose
sentence and canvas must stay clean (the over-correction control and the proof
the fixture is not vacuous), and two static guards cover shapes no assertion
about one chart can see.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from vfairness.rendering import explain as EX
from vfairness.rendering.engine import render_svg

# The band a reader takes the headline from. Rule (c) of the headline rule:
# an ungraded count stated only at the foot of a tall canvas is a count the
# reader of the verdict never meets.
HEADLINE_BAND = 200

_TEXT_RE = re.compile(r'<text\b[^>]*\by="(-?[\d.]+)"[^>]*>(.*?)</text>', re.S)
_DESC_RE = re.compile(r"<desc>(.*?)</desc>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
# "3/4", "3 / 4", "3 of 4". Guarded on both sides so a version string or a
# decimal cannot masquerade as a fraction.
_FRACTION_RE = re.compile(r"(?<![\d.])(\d+)\s*(?:/|of)\s*(\d+)(?![\d.])")
# The fixed clause every partial-run finding ends with, from explain._NOT_COVERED.
_UNGRADED_RE = re.compile(r"(\d+)\s+[^.]*?" + re.escape(EX._NOT_COVERED))


def _plain(markup: str) -> str:
    return html.unescape(_TAG_RE.sub("", markup)).strip()


def canvas_lines(svg: str):
    """[(y, text)] for every <text> the canvas draws."""
    return [(float(y), _plain(body)) for y, body in _TEXT_RE.findall(svg)]


def canvas_text(svg: str, *, max_y: float = float("inf")) -> str:
    return " │ ".join(t for y, t in canvas_lines(svg) if y <= max_y)


def render(template: str, data: dict) -> str:
    """Render with the on-canvas Explanation panel SUPPRESSED.

    CRITICAL, do not drop the empty string. ``render_svg`` auto-generates that
    panel from ``ChartExplanation.paragraph()``, which is the SAME sentence the
    ``<desc>`` is built from, so a canvas that carries it contains the
    description's own words and every comparison below becomes circular: this
    file was written without it and rule 1 passed with the description saying
    "single-attr 1/2" beside a canvas badge reading "1/1 checks, 1 not
    checked", which is the exact defect it exists to catch. "" suppresses the
    panel and leaves the accessible layer untouched, so what remains is the
    canvas the chart draws on its own.
    """
    return render_svg(template, {**data, "explanation": ""})


def desc_of(svg: str) -> str:
    match = _DESC_RE.search(svg)
    assert match, "the render carries no <desc> at all, so it has no accessible layer"
    return html.unescape(match.group(1))


def fractions(text: str):
    return {(int(a), int(b)) for a, b in _FRACTION_RE.findall(text)}


# ───────────────────────────────────────────────────────────────────────────
# Fixtures: one partly measured run and one fully measured twin per chart.
#
# Built as template DATA DICTS and rendered through `render_svg`, which is the
# exact call every adapter makes and the call that injects the <desc>, so the
# two surfaces under test are produced together from one input. The keys mirror
# what the adapters emit; where a key is a partial signal the adapter's name for
# it is used verbatim so a rename cannot leave this file quietly passing.
# ───────────────────────────────────────────────────────────────────────────


def _gate(*, ungraded: bool):
    """A three-level gate. With *ungraded*, one single-attribute metric carries
    no verdict, so the canvas grades 1/1 and the raw adapter total is 2."""
    single = [{"name": "demographic_parity", "value": 0.03, "threshold": 0.1, "passed": True}]
    if ungraded:
        # `passed: None` explicitly, which is what adapters_workflow emits and
        # what the template's three-state row branch keys on.
        single.append(
            {"name": "equal_opportunity", "value": None, "threshold": 0.1, "passed": None}
        )
    return {
        "title": "Hierarchical Fairness Gate",
        "approved": True,
        "overall_status": "PASS",
        "single_status": "PASS",
        "intersect_status": "PASS",
        "n_overall_pass": 1,
        "n_overall_total": 1,
        "n_single_pass": 1,
        "n_single_total": 2 if ungraded else 1,
        "n_intersect_pass": 1,
        "n_intersect_total": 1,
        "min_group_size": 30,
        "warnings": [],
        "levels": [
            {
                "type": "overall",
                "name": "All data",
                "approved": True,
                "metrics": [
                    {"name": "demographic_parity", "value": 0.02, "threshold": 0.1, "passed": True}
                ],
            },
            {
                "type": "single attribute",
                "name": "gender",
                "approved": True,
                "metrics": single,
            },
            {
                "type": "intersectional",
                "name": "gender x race",
                "approved": True,
                "metrics": [
                    {"name": "equalized_odds", "value": 0.04, "threshold": 0.1, "passed": True}
                ],
            },
        ],
    }


def _card(*, ungraded: bool):
    metrics = [
        {"name": "demographic_parity", "value": 0.02, "threshold": 0.1, "passed": True},
        {"name": "equal_opportunity", "value": 0.04, "threshold": 0.1, "passed": True},
    ]
    if ungraded:
        # `passed: None` explicitly: report_card.svg splits its graded subset with
        # `rejectattr("passed", "none")`, which keeps a row whose key is simply
        # ABSENT, so the fixture uses the shape the adapter really emits.
        metrics.append({"name": "equalized_odds", "value": None, "threshold": 0.1, "passed": None})
    return {
        "title": "Fairness Report Card",
        "model_name": "credit_v3",
        "approved": True,
        "metrics": metrics,
        "blocking_reasons": [],
        "card_warnings": [],
        "has_baseline": False,
    }


def _heatmap(*, blank_row: bool):
    def cell(v):
        return {"value": v, "bg": "#ffffff", "text_color": "#0f172a"}

    rows = [
        {"label": "male", "cells": [cell(0.51), cell(0.48)]},
        {"label": "female", "cells": [cell(0.47), cell(0.44)]},
    ]
    if blank_row:
        rows.append({"label": "nonbinary", "cells": [cell(None), cell(None)]})
    return {
        "title": "Disparity Heatmap",
        "subtitle": "Per-group rates underlying each fairness metric.",
        "rows": rows,
        "col_labels": ["Demo. Parity", "Equal Opp."],
        "col_sublabels": ["Sel. Rate", "TPR"],
        "n_rows": len(rows),
        "n_cols": 2,
        "summary_bg": "#d1fae5",
        "summary_border": "#059669",
        "summary_color": "#059669",
        "summary_icon": "✓",
        "summary_text": "Low disparity across groups",
    }


def _dashboard(*, unchecked: bool):
    metrics = [
        {
            "name": "demographic_parity",
            "display_value": "0.0200",
            "display_threshold": "0.1000",
            "color": "#059669",
            "breached": False,
            "warning": False,
            "breach_pct": 0,
            "margin_pct": 80,
            "n_affected_groups": 0,
        }
    ]
    if unchecked:
        metrics.append(
            {
                "name": "[UNCHECKED] equal_opportunity",
                "display_value": "n/a",
                "display_threshold": "NOT CHECKED",
                "color": "#f59e0b",
                "breached": False,
                "warning": True,
                "breach_pct": 0,
                "margin_pct": "n/a",
                "n_affected_groups": 0,
                "could_not_check": True,
            }
        )
    return {
        "title": "Fairness Reporting Dashboard",
        "subtitle": "OPERATIONAL report",
        "timestamp": "2026-08-28 00:00",
        "model_name": "credit_v3",
        "health_score": "88",
        "health_score_raw": 88,
        "status_color": "#059669",
        "status_bg": "#ecfdf5",
        "status_label": "GREEN",
        "trend": "stable",
        "event_trigger": "Scheduled",
        "event_detail": "Periodic report",
        "time_window": "30 days",
        "report_tier": "OPERATIONAL",
        "kpi_compliance": 90,
        "kpi_alert": 95,
        "kpi_drift": 92,
        "n_active_alerts": 0,
        "n_critical_alerts": 0,
        "n_metrics": 1,
        "n_groups": 2,
        "n_records": 1000,
        "metrics": metrics,
        "breaches": [],
        "alerts": [],
        "recommendations": [],
        "narrative": "",
    }


def _radar(*, unmeasured: bool):
    return {
        "title": "Fairness Metrics: Radar Chart",
        # Deliberately the LONG subtitle the adapter builds and `_fit_subtitle`
        # trims: the caveat used to ride on the end of this line and be cut off.
        "subtitle": (
            "Distance from centre = fairness (rim fair, centre unfair). "
            "Dashed ring = threshold; dots outside it pass. Labels show raw…"
        ),
        "rings": [{"fraction": 1.0, "label": "1.0"}],
        "axes": [
            {
                "spoke_x": 480,
                "spoke_y": 362,
                "label_x": 490,
                "label_y": 362,
                "label_anchor": "start",
                "label": "Demo. Parity",
            }
        ],
        "data_dots": [{"x": 470, "y": 362, "value_x": 470, "value_y": 350, "value": 0.03}],
        "data_points": "470,362",
        "threshold_ring_fraction": 0.6,
        "polygon_color": "#f59e0b",
        "status_color": "#f59e0b",
        "status_bg": "#fef3c7",
        "status_icon": "⚠",
        "status_text": "Marginal",
        "n_unmeasured": 2 if unmeasured else 0,
    }


def _effects(*, unmeasured: bool):
    return {
        "title": "Fairness Metrics: Effect Sizes (Cohen's d)",
        "subtitle": "Negligible: |d| < 0.2 · Small: < 0.5 · Medium: < 0.8 · Large: >= 0.8",
        "effects": [
            {
                "label": "Demographic Parity",
                "label_short": "DE",
                "value": 0.05,
                "px_offset": 6,
                "color": "#059669",
                "badge_bg": "#d1fae5",
                "badge_color": "#059669",
                "interpretation": "Negligible",
            }
        ],
        "max_effect": 0.05,
        "max_effect_color": "#059669",
        "max_effect_interpretation": "Negligible",
        "avg_effect": 0.05,
        "avg_effect_color": "#059669",
        "n_unmeasured": 2 if unmeasured else 0,
    }


def _bars(*, unknown: bool):
    return {
        "title": "Fairness Metrics: Bar Chart",
        "subtitle": "Dashed red line = fairness threshold.",
        "bars": [
            {
                "label": "Demographic Parity",
                "bar_w": 40,
                "display_value": "0.020",
                "threshold_x": 120,
                "color": "#059669",
                "status": "FAIR",
                "status_bg": "#d1fae5",
                "status_color": "#059669",
                "status_icon": "✓",
                "state": "PASS",
            }
        ],
        "n_passed": 1,
        "n_total": 1,
        "n_unknown": 2 if unknown else 0,
    }


def _fairness_report(*, unmeasured_groups: bool):
    bars = [
        {"group": "male", "rate": 0.51, "n": 500, "color": "#059669"},
        {"group": "female", "rate": 0.47, "n": 500, "color": "#059669"},
    ]
    return {
        "title": "Fairness Metric Dashboard",
        "task_type": "classification",
        "fairness_score": 92,
        "n_passed": 2,
        "n_failed": 0,
        "n_unknown": 0,
        "n_unmeasured_groups": 2 if unmeasured_groups else 0,
        "cards": [
            {
                "label": "Demographic Parity",
                "key": "demographic_parity_difference",
                "is_ratio": False,
                "value": 0.02,
                "threshold": 0.1,
                "color": "#059669",
                "bg": "#d1fae5",
                "status": "FAIR",
                "passed": True,
                "state": "pass",
            }
        ],
        "bars": bars,
        "summary": "Two metrics fair, none unfair.",
    }


def _cal_disparity(*, ungraded_groups: bool):
    return {
        "title": "Calibration Disparity",
        "subtitle": "Expected calibration error by group.",
        "overall_ece": 0.03,
        "overall_ece_color": "#059669",
        "ece_disparity": 0.012,
        "ece_disparity_color": "#f59e0b",
        "disparity_ratio": 1.6,
        "disparity_ratio_text": "1.6x",
        "ratio_color": "#f59e0b",
        "groups": [
            {"label": "male", "ece": 0.02, "ece_scaled": 0.4, "color": "#059669"},
            {"label": "female", "ece": 0.032, "ece_scaled": 0.64, "color": "#f59e0b"},
        ],
        "best_group": "male",
        "best_ece": 0.02,
        "best_color": "#059669",
        "worst_group": "female",
        "worst_ece": 0.032,
        "worst_color": "#f59e0b",
        "has_significant_disparity": False,
        "has_significant_disparity_known": True,
        "n_groups": 4 if ungraded_groups else 2,
        "n_measured_groups": 2,
        "n_ungraded_groups": 2 if ungraded_groups else 0,
        "recommendation": "",
        "strategy": "",
        "unknown_color": "#64748b",
    }


def _tradeoff(*, withheld: bool):
    return {
        "title": (
            "Accuracy-Fairness Trade-off: 2 of 5 points not plotted"
            if withheld
            else "Accuracy-Fairness Trade-off Analysis"
        ),
        "x_min": 0.7,
        "x_range": 0.2,
        "y_max": 0.2,
        "points": [
            {"x": 0.2, "y": 0.8, "lambda": 0.1, "satisfied": False},
            {"x": 0.5, "y": 0.4, "lambda": 0.5, "satisfied": True},
            {"x": 0.8, "y": 0.1, "lambda": 0.9, "satisfied": True},
        ],
        "pareto_points": [{"x": 0.5, "y": 0.4}, {"x": 0.8, "y": 0.1}],
        "best_fair": None,
        "best_accurate": None,
        "n_withheld_points": 2 if withheld else 0,
        "n_supplied_points": 5 if withheld else 3,
    }


def _corr_matrix(*, uncomputed: bool):
    def cell(v):
        return {"value": v, "bg": "#ffffff", "text_color": "#0f172a", "method_dot": ""}

    rows = [
        {"label": "income", "cells": [cell(1.0), cell(0.12)]},
        {"label": "zipcode", "cells": [cell(0.12), cell(1.0) if not uncomputed else cell(None)]},
    ]
    return {
        "title": "Correlation Matrix",
        # The adapter appends the uncomputed count to this line (see
        # adapters_feature_engineering: "N of M pair(s) not computed, so not
        # tested"), and the subtitle sits at y=122, inside the headline band. The
        # fixture carries the real sentence so the rule (c) check is testing the
        # adapter's behaviour and not a caption invented here.
        "subtitle": (
            "Off-diagonal cells above the threshold flag proxy risk."
            + (" 1 of 4 pair(s) not computed, so not tested." if uncomputed else "")
        ),
        "rows": rows,
        "col_labels": ["income", "zipcode"],
        "labels": ["income", "zipcode"],
        "n_high_corr": 0,
        "n_features": 2,
        "n_cols": 2,
        "n_rows": 2,
        "threshold": 0.3,
        "method_display": "auto",
        "risk_color": "#059669",
        "legend": [],
    }


def _drift(*, partial_scales: bool):
    return {
        "title": "FairnessDriftDetector: Drift Report",
        "metric_name": "demographic_parity"
        + (" (1 of 2 scale(s) did not fully report)" if partial_scales else ""),
        "drift_detected": False,
        "overall_verdict_recorded": True,
        "overall_drift_score": 0.05,
        "overall_color": "#059669",
        "scales": [
            {
                "label": "short",
                "drift_score": 0.05,
                "drift_detected": False,
                "color": "#059669",
                "score_known": True,
                "verdict_recorded": True,
                "ref_mean": 0.51,
                "cur_mean": 0.52,
                "mean_shift": 0.01,
                "score_bar_w": 5,
                "ks_stat": 0.02,
                "p_value": 0.6,
            }
        ],
        "n_scored_scales": 1,
        "n_partial_scales": 1 if partial_scales else 0,
    }


def _causal(*, ungraded_steps: bool):
    return {
        "title": "Causal Decomposition",
        "measured": True,
        "mediator": "education",
        "prop_med_pct": 42,
        "prop_med_arc": 50,
        "proportion_mediated_known": True,
        "trend_slope_known": True,
        "recommendations": [],
        "n_steps_ok": 3,
        "n_steps_ungraded": 1 if ungraded_steps else 0,
        "total_effect": 0.12,
        "direct_effect": 0.07,
        "indirect_effect": 0.05,
        "total_known": True,
        "direct_known": True,
        "indirect_known": True,
        "direct_w": 100,
        "indirect_w": 80,
        "bk_steps": [],
        "temporal": None,
    }


def _bias_audit(*, ungraded_findings: bool):
    return {
        "title": "BiasDetector: Audit Report",
        "overall_score": 0.12,
        "overall_color": "#059669",
        "overall_bg": "#d1fae5",
        "overall_label": "MINIMAL",
        "coverage": "full",
        "n_modules_run": 4,
        "n_modules": 4,
        "n_critical": 0,
        "n_ungraded_findings": 3 if ungraded_findings else 0,
        "modules": [],
        "issues": [],
        # `na_line` is the key bias_audit.svg actually draws, at y=136, inside
        # the headline band; adapters.py builds it from the same count.
        "na_line": ("3 finding(s) reported no severity, none graded" if ungraded_findings else ""),
        "unknown_color": "#64748b",
        "unknown_bg": "#f1f5f9",
    }


class Case:
    def __init__(self, name, template, partial, healthy, must_say, other_population=()):
        self.name = name
        self.template = template
        self.partial = partial
        self.healthy = healthy
        self.must_say = must_say
        # Fractions the canvas draws that describe a DIFFERENT population from
        # anything the description counts, so a shared numerator between them is
        # a coincidence rather than a contradiction. Each one has to be argued
        # for at the case, never added to quieten a failure.
        self.other_population = set(other_population)

    def __repr__(self):  # pragma: no cover - pytest id only
        return self.name


CASES = [
    Case(
        "hierarchical_gate",
        "hierarchical_gate",
        _gate(ungraded=True),
        _gate(ungraded=False),
        ["single-attr 1/1"],
    ),
    Case("report_card", "report_card", _card(ungraded=True), _card(ungraded=False), ["2/2"]),
    Case(
        "disparity_heatmap",
        "disparity_heatmap",
        _heatmap(blank_row=True),
        _heatmap(blank_row=False),
        ["2 group(s)"],
    ),
    Case(
        "reporting_dashboard",
        "reporting_dashboard",
        _dashboard(unchecked=True),
        _dashboard(unchecked=False),
        ["breach(es) of 1"],
    ),
    Case("radar_chart", "radar_chart", _radar(unmeasured=True), _radar(unmeasured=False), []),
    Case("effect_sizes", "effect_sizes", _effects(unmeasured=True), _effects(unmeasured=False), []),
    Case("metrics_bar_chart", "metrics_bar_chart", _bars(unknown=True), _bars(unknown=False), []),
    Case(
        "fairness_report",
        "fairness_report",
        _fairness_report(unmeasured_groups=True),
        _fairness_report(unmeasured_groups=False),
        [],
    ),
    Case(
        "calibration_disparity",
        "calibration_disparity",
        _cal_disparity(ungraded_groups=True),
        _cal_disparity(ungraded_groups=False),
        [],
    ),
    Case(
        "tradeoff_analysis",
        "tradeoff_analysis",
        _tradeoff(withheld=True),
        _tradeoff(withheld=False),
        ["2 of 3"],
        # "2 of 5 points not plotted" is the WITHHELD count, which
        # adapters_training puts in the title because this template's subtitle
        # is fixed prose. It shares a numerator with "2 of 3 configurations
        # satisfy the constraint" and counts something else entirely: rule 2
        # below is what checks that this number reached the canvas at all.
        other_population={(2, 5)},
    ),
    Case(
        "correlation_matrix",
        "correlation_matrix",
        _corr_matrix(uncomputed=True),
        _corr_matrix(uncomputed=False),
        [],
    ),
    Case(
        "drift_report",
        "drift_report",
        _drift(partial_scales=True),
        _drift(partial_scales=False),
        [],
    ),
    Case(
        "causal_decomposition",
        "causal_decomposition",
        _causal(ungraded_steps=True),
        _causal(ungraded_steps=False),
        ["3/4"],
    ),
    Case(
        "bias_audit",
        "bias_audit",
        _bias_audit(ungraded_findings=True),
        _bias_audit(ungraded_findings=False),
        [],
    ),
]


@pytest.fixture(params=CASES, ids=lambda c: c.name)
def case(request):
    return request.param


# ───────────────────────────────────────────────────────────────────────────
# The four general rules. Each applies to EVERY case, so a chart cannot be
# fixed at the site somebody pointed at and left wrong everywhere else.
# ───────────────────────────────────────────────────────────────────────────


def test_no_fraction_in_the_description_contradicts_the_canvas(case):
    """Rule 1, the denominator rule, and the one that catches the whole class.

    A fraction in the ``<desc>`` describes the same population the canvas draws
    a fraction for. If the canvas prints a fraction with the same numerator, the
    denominators must be the same number: "1/3 metrics pass" beside "1/1
    checks" is one surface counting a row the other one withheld.
    """
    svg = render(case.template, case.partial)
    on_canvas = fractions(canvas_text(svg)) - case.other_population
    in_desc = fractions(desc_of(svg))
    for num, den in sorted(in_desc):
        same_numerator = {(a, b) for a, b in on_canvas if a == num}
        if not same_numerator:
            continue
        assert (num, den) in same_numerator, (
            f"{case.name}: the description says {num}/{den} while the canvas says "
            f"{sorted(same_numerator)} for the same numerator. One of the two is "
            f"counting a row the other withheld.\n"
            f"  desc:   {desc_of(svg)}\n"
            f"  canvas: {canvas_text(svg)}"
        )


# W-30 CLOSED 2026-09-07, and this set is now empty. disparity_heatmap.svg used
# to count nothing about a group row whose every cell came back N/A: the group
# was drawn, greyed, read "N/A", and never counted anywhere a reader of the
# headline could see. `explain._fr_disparity_heatmap` said "1 group(s) reported
# no value on any metric" in the <desc> while the canvas showed a green badge
# reading "Low disparity across groups", so the ACCESSIBLE layer was the more
# honest of the two, which is the wrong way round.
#
# The template now draws a COULD NOT CHECK chip in the headline band, counted
# from the same rows the grid draws. The entry was removed the moment the count
# reached the band, which is what the assertion below asks for.
#
# Keep the set, and the branch: it is how the NEXT chart in this position gets
# recorded rather than silently excused, and an empty set means the guard is
# currently excusing nothing.
_HEADLINE_BAND_GAPS: set = set()


def test_the_ungraded_count_in_the_description_is_on_the_canvas_headline(case):
    """Rule 2, part (c) of the headline rule.

    Whatever count the description gives for the rows nobody graded must be
    readable in the headline band, where the reader takes the verdict from. A
    number that appears only in the accessible layer leaves the canvas the more
    confident of the two; a number only at the foot of a tall canvas leaves the
    reader of the headline none the wiser.
    """
    svg = render(case.template, case.partial)
    counts = {int(n) for n in _UNGRADED_RE.findall(desc_of(svg))}
    assert counts, (
        f"{case.name}: the partial fixture produced no ungraded clause at all, so this "
        f"case is not exercising the rule it was written for.\n  desc: {desc_of(svg)}"
    )
    band = canvas_text(svg, max_y=HEADLINE_BAND)
    stated = all(re.search(rf"(?<![\d.]){n}(?![\d.])", band) for n in counts)
    if case.name in _HEADLINE_BAND_GAPS:
        assert not stated, (
            f"{case.name} now states its ungraded count in the headline band; remove it "
            f"from _HEADLINE_BAND_GAPS so the guard stops excusing a chart that is fixed"
        )
        return
    for n in sorted(counts):
        assert re.search(rf"(?<![\d.]){n}(?![\d.])", band), (
            f"{case.name}: the description reports {n} ungraded row(s) and no number {n} "
            f"appears on the canvas above y={HEADLINE_BAND}.\n"
            f"  desc:          {desc_of(svg)}\n"
            f"  headline band: {band}"
        )


def test_the_description_never_prints_a_withheld_value_as_a_number(case):
    """Rule 3. A withheld value is said in words or it is not said at all.

    ``None``, ``nan``, a bare ``?`` and an orphan ``%`` are all the shape of a
    number that was not there, and a reader cannot tell any of them from a
    rendering fault. "None/0 subgroups flagged" is the one this rule is named
    after.
    """
    for label, data in (("partial", case.partial), ("healthy", case.healthy)):
        text = desc_of(render(case.template, data))
        for bad in ("None", "nan", "N/A/", "/0 ", " ? ", "(? "):
            assert bad not in text, f"{case.name} ({label}): {bad!r} in the description: {text}"
        assert not re.search(r"(?<![\d.\w])%", text), (
            f"{case.name} ({label}): a percent sign with no number before it: {text}"
        )
        assert not re.search(r"(?<![\d.])0\s*(?:/|of)\s*0(?![\d.])", text), (
            f"{case.name} ({label}): a fraction over a denominator of zero: {text}"
        )


def test_a_partial_run_never_carries_the_all_clear_severity(case):
    """Rule 4, part (b) in its machine-readable form.

    ``ChartExplanation.metadata()["severity"]`` is what a report pipeline reads
    INSTEAD of the sentence, so "info" there is the same unqualified all-clear
    the prose is forbidden to write.
    """
    ce = EX.build_explanation(case.template, case.partial)
    assert ce.severity != "info", (
        f"{case.name}: a partly measured run graded at the all-clear severity.\n"
        f"  finding: {ce.finding}"
    )


# ───────────────────────────────────────────────────────────────────────────
# The over-correction control, and the proof the fixtures are not vacuous.
# ───────────────────────────────────────────────────────────────────────────


def test_the_fully_measured_twin_is_left_alone(case):
    """A healthy run says nothing about ungraded rows, on either surface."""
    svg = render(case.template, case.healthy)
    text = desc_of(svg)
    assert EX._NOT_COVERED not in text, (
        f"{case.name}: a fully measured run picked up a partial-run clause: {text}"
    )
    assert EX._COULD_NOT_CHECK not in text.upper(), (
        f"{case.name}: a fully measured run was described as could-not-check: {text}"
    )
    assert "COULD NOT CHECK" not in canvas_text(svg, max_y=HEADLINE_BAND).upper(), (
        f"{case.name}: a fully measured run drew an ungraded chip in the headline band"
    )


def test_the_partial_fixture_is_not_the_healthy_one_in_disguise(case):
    partial = desc_of(render(case.template, case.partial))
    healthy = desc_of(render(case.template, case.healthy))
    assert partial != healthy, (
        f"{case.name}: the partial run reads identically to the fully measured one, so "
        f"every assertion above is vacuous for it"
    )


# ───────────────────────────────────────────────────────────────────────────
# Static guard: the {% set %} accumulator inside a {% for %}.
#
# Jinja discards a bare `{% set x = x + ... %}` at the end of each iteration, so
# an accumulator written that way stays at its initial value and every item is
# drawn at the SAME coordinate: the last one painted covers the rest. The markup
# is correct, so a substring assertion passes straight over it, and it shipped a
# false green "LOW: 2" over a HIGH-risk proxy in the published gallery. Only the
# coordinates, or a rasteriser, can see it.
#
# Fixed by putting the accumulator on a `namespace`, which survives the loop.
# ───────────────────────────────────────────────────────────────────────────

_TEMPLATES = Path(EX.__file__).parent / "templates"

# The two shapes this wave found alive: hierarchical_gate.svg accumulated both
# its total height and its running panel y this way (four level panels at one
# coordinate, the last one covering the other three, under an APPROVED gate),
# and cicd_pipeline.svg counted its BLOCKING REASON lines this way, so a gate
# naming three reasons to refuse a model drew all three at decision_y + 48 and
# showed one. Both are fixed; nothing is excused here.


_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.S)


def _accumulator_sites(path: Path):
    """Bare `{% set x = <expr containing x> %}` sites inside a `{% for %}`.

    Jinja comments are blanked first, line count preserved: the notes this wave
    left in the templates QUOTE the defective shape (`{% set x = x + ... %}`
    inside a `{% for %}`) in order to forbid it, and a scanner that reads its
    own warning as an instance is a scanner that cannot be commented.
    """
    source = path.read_text("utf-8")
    source = _JINJA_COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), source)
    namespaces = set(re.findall(r"\{%-?\s*set\s+(\w+)\s*=\s*namespace\(", source))
    depth = 0
    hits = []
    for lineno, line in enumerate(source.splitlines(), 1):
        opens = len(re.findall(r"\{%-?\s*for\s", line))
        closes = len(re.findall(r"\{%-?\s*endfor\s*-?%\}", line))
        if depth > 0:
            for match in re.finditer(r"\{%-?\s*set\s+([\w.]+)\s*=\s*([^%]*?)-?%\}", line):
                name, expr = match.group(1), match.group(2)
                if "." in name and name.split(".")[0] in namespaces:
                    continue  # a namespace attribute: survives the loop
                if re.search(r"\b" + re.escape(name) + r"\b", expr):
                    hits.append((lineno, name))
        depth = max(0, depth + opens - closes)
    return hits


def test_no_template_accumulates_with_a_bare_set_inside_a_for():
    offenders = {}
    for path in sorted(_TEMPLATES.glob("*.svg")):
        hits = _accumulator_sites(path)
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        "a bare {% set %} accumulator inside a {% for %} is discarded by Jinja, so every "
        "item draws at the same coordinate and only the last one painted survives. The "
        "markup is correct, so a substring assertion reads straight past it. Put the "
        f"accumulator on a namespace, or use loop.index0: {offenders}"
    )


def test_the_gate_draws_every_level_at_its_own_coordinate():
    """The rendered proof of the rule above, on the chart it shipped broken.

    The published hierarchical_gate.svg showed an APPROVED gate, three green
    PASS chips and ONE of the eight metrics that substantiate them, because all
    four level panels were painted at `levels_y`. Asserted on the coordinates
    rather than on the markup, because the markup was never wrong.
    """
    data = _gate(ungraded=False)
    # Four levels, two metric rows each: the shape the gallery renders.
    base = data["levels"][0]
    data["levels"] = [dict(base, name=f"level {i}", type="single attribute") for i in range(4)]
    for lv in data["levels"]:
        lv["metrics"] = [
            {"name": "demographic_parity", "value": 0.02, "threshold": 0.1, "passed": True},
            {"name": "equalized_odds", "value": 0.03, "threshold": 0.1, "passed": True},
        ]
    svg = render("hierarchical_gate", data)

    rows = [y for y, text in canvas_lines(svg) if text in {"demographic_parity", "equalized_odds"}]
    assert len(rows) == 8, f"expected eight metric rows, drew {len(rows)}"
    assert len(set(rows)) == 8, (
        f"the four level panels share coordinates, so metric rows are painted over each "
        f"other and only the last panel is readable: {sorted(rows)}"
    )

    # The canvas must also be tall enough to hold them: `levels_h` is the other
    # half of the same accumulator, and at 0 it put the warnings panel and the
    # explanation on top of the level stack.
    height = float(re.search(r'viewBox="0 0 680 ([\d.]+)"', svg).group(1))
    assert height > max(rows), (
        f"the canvas is {height}px tall and a metric row is drawn at y={max(rows)}, so the "
        f"level stack falls outside the viewBox"
    )


# ───────────────────────────────────────────────────────────────────────────
# robustness_testing: the withheld numerator, kept out of the parametrised set
# because the second half of its fix is blocked by a test in another file.
# ───────────────────────────────────────────────────────────────────────────


def test_an_absent_subgroup_audit_is_stated_in_words_not_as_none_over_zero():
    """ "None/0 subgroups flagged" was a literal string this library emitted.

    ``subgroup`` empty means the audit examined nobody: `_g(sg, "n_flagged")`
    is None and `_n(sg, "n_analyzed")` is 0, so the withheld branch (which needs
    a non-zero denominator) was skipped and the else arm formatted a Python
    None into a fraction over zero, on the accessible layer, beside a canvas
    that draws no subgroup panel at all.
    """
    ce = EX.build_explanation(
        "robustness_testing",
        {"overall_score": 0.85, "overall_label": "ROBUST", "subgroup": {}},
    )
    assert "None" not in ce.finding, ce.finding
    assert "/0" not in ce.finding, ce.finding
    assert "no subgroup audit was reported" in ce.finding, ce.finding

    # An audit that DID examine subgroups and reported no flagged count keeps
    # its own wording, and one that reported both keeps its fraction.
    withheld = EX.build_explanation(
        "robustness_testing",
        {"overall_score": 0.85, "overall_label": "ROBUST", "subgroup": {"n_analyzed": 6}},
    )
    assert "reported no flagged count" in withheld.finding, withheld.finding
    assert withheld.severity != "info"

    measured = EX.build_explanation(
        "robustness_testing",
        {
            "overall_score": 0.85,
            "overall_label": "ROBUST",
            "subgroup": {"n_analyzed": 6, "n_flagged": 1},
        },
    )
    assert "1/6 subgroups flagged" in measured.finding, measured.finding


def test_an_absent_subgroup_audit_is_not_graded_at_the_all_clear_severity():
    """W-28, xfail(strict) cleared 2026-09-07 when the defect was fixed.

    An absent subgroup audit used to take its severity from the score band, so
    "no audit ran" graded `info`, which is what a COMPLETE audit finding nothing
    wrong grades. Measured at score 0.85 before the fix:

        subgroup {}                          -> info    (no audit ran)
        subgroup {n_analyzed: 4}             -> medium  (count withheld)
        subgroup {n_analyzed: 4, flagged: 0} -> info    (real all-clear)

    The weakest evidence of the three was graded like the strongest.
    """
    ce = EX.build_explanation(
        "robustness_testing",
        {"overall_score": 0.85, "overall_label": "ROBUST", "subgroup": {}},
    )
    assert ce.severity != "info"


def test_a_real_all_clear_subgroup_audit_still_grades_as_a_pass():
    """Over-correction control for W-28: flooring everything hides real passes."""
    ce = EX.build_explanation(
        "robustness_testing",
        {
            "overall_score": 0.85,
            "overall_label": "ROBUST",
            "subgroup": {"n_analyzed": 4, "n_flagged": 0},
        },
    )
    assert ce.severity == "info"


# ───────────────────────────────────────────────────────────────────────────
# Two-state branches: absent must not render as a verdict.
# ───────────────────────────────────────────────────────────────────────────


def test_an_unchecked_dashboard_row_draws_no_verdict_and_no_all_clear():
    """reporting_dashboard.svg tested `{% if m.breached %}...{% else %}PASS`.

    adapters_reporting recorded this as template-blocked RESIDUE and worked
    around it by sending `warning=True` for an unchecked row, because dropping
    that flag would have sent the row to the green arm; the affected-groups cell
    it could NOT work around, so a row named "[UNCHECKED]" with the threshold
    "NOT CHECKED" printed "all pass" beside itself. No value an adapter can send
    suppresses a two-state test on a bare number, so the third branch has to be
    in the template.
    """
    svg = render("reporting_dashboard", _dashboard(unchecked=True))
    cells = [t for _y, t in canvas_lines(svg)]
    # The BADGE, not the threshold cell (which also reads "NOT CHECKED"). The
    # adapter sets warning=True on this row for one reason only: with the
    # two-state test the row would otherwise reach the green arm. So WARN on
    # the canvas is the tell that the workaround is still load-bearing and the
    # third branch is gone.
    assert "WARN" not in cells, (
        "the unchecked row is badged WARN, which is the adapter's workaround for a "
        f"two-state test, not a verdict anyone measured: {cells}"
    )
    assert cells.count("NOT CHECKED") >= 2, (
        f"expected NOT CHECKED as both the threshold and the badge: {cells}"
    )
    assert "not counted" in cells, (
        "the unchecked row still prints a groups verdict; 'all pass' beside an "
        "[UNCHECKED] name is a fabricated all-clear on the canvas"
    )
    assert cells.count("PASS") == 1, (
        f"expected one PASS badge for the one graded row, drew {cells.count('PASS')}"
    )
    # And the fully graded twin keeps its badge, so this is not a blanket ban.
    healthy = [
        t for _y, t in canvas_lines(render("reporting_dashboard", _dashboard(unchecked=False)))
    ]
    assert healthy.count("PASS") == 1 and "NOT CHECKED" not in healthy


# ───────────────────────────────────────────────────────────────────────────
# experiment_results.svg: the flag that was emitted and read zero times.
#
# adapters_experimentation sets `significant` False and `significant_known`
# False for an intersection that reported neither a significance flag nor a
# p-value to derive one from. Every test on `e.significant` in the template was
# bare, so an UNKNOWN significance took the arm that means "tested and found not
# significant": the same ink in the EFFECT column, the same filled slate dot on
# the forest plot, the same solid whisker. A reader takes the grade off the
# colour before the words, so the third state existed in the data and nowhere on
# the canvas. An unread flag is a fix that never landed.
#
# Asserted on the MARKUP THAT PAINTS, and confirmed by rasterising: on the
# rendered page the three forest rows are a filled ocean dot, a filled slate dot
# and a hollow ring with a dashed whisker.
# ───────────────────────────────────────────────────────────────────────────

_CIRCLE_RE = re.compile(r"<circle\b[^>]*>")
_ATTR_RE = re.compile(r'(\w[\w-]*)="([^"]*)"')


def _attrs(tag: str) -> dict:
    return dict(_ATTR_RE.findall(tag))


def _effect_inks(svg: str):
    """The fill of every EFFECT (d) cell, heading excluded.

    The column heading sits at the same x as its cells and is the only text
    there carrying `letter-spacing`, which is how the two are told apart
    without pinning a y coordinate that any layout change would move.
    """
    return [
        _attrs(tag)["fill"]
        for tag in re.findall(r'<text x="220"[^>]*>', svg)
        if "fill=" in tag and "letter-spacing" not in tag
    ]


def _experiment(*, undecided_row: bool) -> str:
    """One significant row, one measured non-significant row, and optionally a
    row that reported no p-value and no significance flag at all."""
    from vfairness.rendering.adapters_experimentation import experiment_results_to_svg

    rows = [
        {
            "intersection": ("female", "30-40"),
            "effect": 0.02,
            "p_value": 0.01,
            "ci_lower": 0.005,
            "ci_upper": 0.035,
            "n_control": 100,
            "n_treatment": 100,
            "powered": True,
        },
        {
            "intersection": ("male", "30-40"),
            "effect": 0.01,
            "p_value": 0.40,
            "ci_lower": -0.01,
            "ci_upper": 0.03,
            "n_control": 100,
            "n_treatment": 100,
            "powered": True,
        },
    ]
    if undecided_row:
        rows.append(
            {
                "intersection": ("nonbinary", "30-40"),
                "effect": 0.015,
                "ci_lower": -0.005,
                "ci_upper": 0.035,
                "n_control": 50,
                "n_treatment": 50,
                "powered": True,
            }
        )
    return experiment_results_to_svg(
        {
            "overall_effect": 0.012,
            "overall_p_value": 0.03,
            "overall_significant": True,
            "heterogeneity_detected": False,
            "heterogeneity_p_value": 0.4,
            "intersection_effects": rows,
        }
    )


def test_an_undecided_significance_is_not_painted_as_a_decided_one():
    svg = _experiment(undecided_row=True)
    circles = [_attrs(t) for t in _CIRCLE_RE.findall(svg)]
    assert len(circles) == 3, f"expected one forest marker per row, drew {len(circles)}"

    fills = [c["fill"] for c in circles]
    assert len(set(fills)) == 3, (
        "two of the three forest markers share a fill, so an undecided significance is "
        f"painted as a decided one: {fills}"
    )
    # The undecided row is HOLLOW: its fill is the same white every other marker
    # uses as its outline, and its outline carries the colour instead. Checked
    # structurally rather than by hex, because skins.py recolours the palette.
    outline = circles[0]["stroke"]
    hollow = [c for c in circles if c["fill"] == outline]
    assert len(hollow) == 1, f"expected exactly one hollow marker, found {len(hollow)}: {circles}"
    assert hollow[0]["stroke"] != outline, "the hollow marker has no outline colour at all"
    assert hollow[0] is circles[2], "the hollow marker is not on the row that reported no p-value"

    # The whisker for that row is dashed, so the row is separable in one glance
    # and in greyscale, not only by hue. "3,2" specifically: the zero line of
    # the plot is dashed "4,3" and always was, so a bare substring test for
    # stroke-dasharray would pass on a template that changed nothing.
    assert 'stroke-dasharray="3,2"' in svg, (
        "the undecided row's confidence interval is drawn solid, exactly like a measured one"
    )

    # And the EFFECT column ink, which is the other site that read `significant`
    # bare: three rows, three distinct inks. The column HEADING also sits at
    # x=220, so it is excluded by its letter-spacing rather than by position.
    effect_inks = _effect_inks(svg)
    assert len(effect_inks) == 3 and len(set(effect_inks)) == 3, (
        f"the effect column paints an undecided significance as a decided one: {effect_inks}"
    )


def test_a_fully_decided_experiment_keeps_the_two_colours_it_had():
    """The over-correction control, rendered rather than reasoned about.

    Every row here reported a p-value, so significance was decided for all of
    them: no hollow marker, no dashed whisker, and the two established colours
    unchanged.
    """
    svg = _experiment(undecided_row=False)
    circles = [_attrs(t) for t in _CIRCLE_RE.findall(svg)]
    assert len(circles) == 2

    outline = circles[0]["stroke"]
    assert all(c["fill"] != outline for c in circles), (
        f"a fully decided experiment drew a hollow marker: {circles}"
    )
    assert 'stroke-dasharray="3,2"' not in svg, (
        "a fully decided experiment drew the undecided row's dashed whisker"
    )
    assert circles[0]["r"] == "4" and circles[1]["r"] == "3.5"
    # Two rows, two inks, and the ocean of the significant one is still ocean.
    inks = _effect_inks(svg)
    assert len(inks) == 2 and len(set(inks)) == 2, inks
    assert inks[0] == circles[0]["fill"], (
        f"the significant row's effect ink no longer matches its forest marker: {inks}"
    )


def test_a_sweep_that_satisfies_the_constraint_nowhere_is_not_graded_info():
    """`_fr_tradeoff` returned the literal severity "info" for every input.

    "0 of 5 configurations satisfy the constraint" beside a scatter of five red
    dots was read out at the all-clear grade, and
    ``ChartExplanation.metadata()["severity"]`` is what a report pipeline reads
    INSTEAD of the sentence. Checked with NOTHING withheld, so the floor a
    partial run applies cannot stand in for the band.
    """
    data = _tradeoff(withheld=False)
    data["points"] = [dict(p, satisfied=False) for p in data["points"]]
    ce = EX.build_explanation("tradeoff_analysis", data)
    assert "0 of 3" in ce.finding, ce.finding
    assert ce.severity == "high", (
        f"a sweep that met the constraint nowhere graded {ce.severity!r}: {ce.finding}"
    )
    # A sweep that meets it everywhere is still the all-clear it always was.
    allpass = _tradeoff(withheld=False)
    allpass["points"] = [dict(p, satisfied=True) for p in allpass["points"]]
    assert EX.build_explanation("tradeoff_analysis", allpass).severity == "info"
