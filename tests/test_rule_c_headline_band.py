"""Rule (c): the ungraded count belongs BESIDE THE HEADLINE, not at the foot.

THE HEADLINE RULE, decided 2026-08-28 (see the block comment above
``_ungraded_tail`` in ``vfairness/rendering/explain.py``):

  a) grade and band over the GRADED subset, so a partial run still gives a
     useful verdict on what WAS measured;
  b) never write an unqualified all-clear while anything is ungraded;
  c) state the ungraded count ON THE CANVAS, in the HEADLINE BAND;
  d) an ungraded row never enters a numerator or a denominator.

This file is part (c), and only part (c). The other three are checked against
the accessible description in ``tests/test_explain_partial_runs.py``; here the
question is WHERE on the canvas the count is drawn.

Why the position is the rule and not a detail. A reader takes the headline and
a subtitle and stops, and the reader who stops early is exactly the one this
rule protects: the count is the difference between "4 intersections were
analysed" and "4 intersections, two of them not graded", and only one of those
two sentences is true of a partly measured run. Two templates stated it only at
the foot. ``experiment_results`` named its omitted forest rows at the top of the
forest plot (y around 640 on a four-effect canvas) and its ungraded power
verdicts inside the POWER SUMMARY panel below that (y around 710), under a
subtitle that said "4 intersections" and nothing else. ``training_report`` never
stated a count at all: a method whose run reported no constraint result got a
slate NOT CHECKED chip in its own table row, at y=487 on a six-method report,
and a reader had to scroll the whole table and tally the chips themselves.

Deliberately NOT four one-off assertions. Every template that draws an ungraded
count is checked against ONE rule, from one fixture table, and a static guard
below requires a fixture for every template whose source names such a count, so
a new chart cannot arrive with its count at the foot and no test to say so.

HOW THE COUNT LINE IS FOUND, and why not by phrase. The wordings differ per
chart ("not graded", "not monitored", "could not be checked", "reported
nothing", "never tracked"), and a test that greps for a phrase zoo checks the
zoo rather than the rule: a new chart phrasing it a seventeenth way would pass
by saying nothing at all. So each fixture is rendered TWICE, once with the
ungraded count set to a distinctive value and once with it zero, and the text
elements that appear only in the first are, by construction, the lines the count
produced. That also makes the fixture self-checking: a fixture that does not
move the canvas has nothing to place, and is reported as vacuous rather than
passing.
"""

import re
from pathlib import Path

import pytest

from vfairness.rendering import engine as ENGINE
from vfairness.rendering.engine import render_svg

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


# The band. 200 is not arbitrary: every template in this library lays out a
# 52px header bar, a title at y=96 and a subtitle at y=122, and the first
# content panel starts at `header_h`, which is 140 on all of them. 200 leaves
# room for the badge row a few charts put directly under the header (the
# group-comparison disparity badge sits at y=162, the power summary's own
# caption at y=200) and stops well above any scrolled content.
_HEADLINE_BAND = 200

# The count used for the partial render. Distinctive on purpose: a "2" collides
# with sample counts, axis labels and percentages all over these canvases, and
# the test would then accept a line that never mentioned the ungraded rows.
_N = 7

_TS = "2026-08-28 10:00"


def _base(**kw):
    d = {"title": "T", "timestamp": _TS, "subtitle": "s", "explanation": None}
    d.update(kw)
    return d


_FIXTURES = {}


def _add(template, common, keys):
    """A fixture whose twins differ ONLY in *keys*, set to _N and to 0.

    Keeping every other value identical is what keeps the render diff small
    enough to be meaningful: a twin with a different number of rows would shift
    every panel below it and the diff would be the whole canvas.
    """
    _FIXTURES[template] = (
        dict(common, **{k: _N for k in keys}),
        dict(common, **{k: 0 for k in keys}),
    )


def _add_pair(template, partial, healthy):
    """For a template that counts its ungraded rows itself, in Jinja, so there
    is no data key to flip. The twins carry the same NUMBER of rows, differing
    only in whether those rows were graded."""
    _FIXTURES[template] = (partial, healthy)


_add(
    "alert_timeline",
    _base(
        windows=[
            {"label": "w1", "n_alerts": 0, "color": "#059669", "x": 60, "width": 40, "height": 20},
            {"label": "w2", "n_alerts": 0, "color": "#059669", "x": 110, "width": 40, "height": 20},
        ],
        n_total=9,
        n_alerted=0,
        n_clean=2,
        alerts=[],
        metrics=[],
        recommendations=[],
    ),
    ["n_unmonitored"],
)

_add(
    "auto_discovery",
    _base(
        candidates=[{"name": "a", "score": 0.4, "width": 80, "color": "#000", "display": "0.40"}],
        groups=[{"name": "g", "rate": 0.5, "display": "50%", "width": 60, "color": "#000"}],
        violations=[],
        n_violations=9,
        n_high=0,
        scan_state="scanned",
        cand_state="scanned",
        viol_state="scanned",
        group_state="scanned",
        recommendations=[],
    ),
    ["n_ungraded"],
)

_add(
    "fairness_report",
    _base(
        task_type="classification",
        cards=[{"name": "c", "state": "pass", "value": 0.1, "display": "0.10", "threshold": "0.2"}],
        bars=[{"group": "g", "rate": 0.5, "display": "50%", "width": 60, "color": "#000"}],
        unmeasured_groups=[],
        fairness_score=92,
        n_passed=1,
        n_failed=0,
        not_assessable=False,
        not_assessable_reason="",
        excluded_groups=[],
        summary="ok",
        explanations={},
    ),
    ["n_unknown"],
)

_add(
    "group_comparison",
    _base(
        bars=[
            {"group": "a", "rate": 0.5, "display": "50%", "width": 100, "color": "#000"},
            {"group": "b", "rate": 0.4, "display": "40%", "width": 80, "color": "#000"},
        ],
        ungraded_groups=[],
        overall_rate=0.45,
        overall_display="45%",
        max_disparity=0.1,
        disparity_color="#059669",
        disparity_bg="#d1fae5",
        disparity_icon="OK",
        disparity_label="Low",
        metric_label="Rate",
        worst_group="a",
        best_group="b",
        not_assessable=False,
    ),
    ["n_ungraded_groups"],
)

_CELL = {"value": 0.5, "n": 10, "bg": "#fff", "text_color": "#000", "sub_color": "#999"}
_add(
    "intersectional_analysis",
    _base(
        feature_name="f",
        x_attr="x",
        y_attr="y",
        col_labels=["c1"],
        rows=[
            {"label": "r1", "cells": [_CELL]},
            {"label": "r2", "cells": [dict(_CELL, value=0.4)]},
        ],
        n_rows=2,
        n_cols=1,
        max_disparity=0.1,
        max_color="#059669",
        max_group_1="a",
        max_group_2="b",
        n_measured=2,
        n_cells=9,
    ),
    ["n_unmeasured"],
)

_add(
    "method_comparison",
    _base(
        methods=[
            {
                "name": f"m{i}",
                "accuracy": 0.8,
                "violation": 0.02,
                "satisfied": True,
                "graded": True,
                "accuracy_measured": True,
                "violation_measured": True,
                "time": 1.0,
                "acc_bar_w": 60,
                "viol_bar_w": 20,
            }
            for i in range(2)
        ]
    ),
    ["n_ungraded"],
)

_add(
    "metrics_bar_chart",
    _base(
        metrics=[
            {
                "name": "m",
                "value": 0.5,
                "display": "0.50",
                "width": 100,
                "color": "#000",
                "passed": True,
                "state": "pass",
                "threshold_x": 50,
                "threshold_display": "0.2",
            }
        ],
        n_passed=1,
        n_total=1,
        pass_pct=1.0,
        not_assessable=False,
    ),
    ["n_unknown"],
)

_add(
    "power_analysis",
    _base(
        measured=True,
        rows=[
            {
                "name": "a",
                "power": 0.9,
                "power_display": "90",
                "powered": True,
                "powered_known": True,
                "power_known": True,
                "n_current": 10,
                "n_required": 10,
                "bar_w": 50,
                "color": "#000",
                "n_known": True,
                "required_known": True,
                "current_known": True,
                "n_current_known": True,
                "n_required_known": True,
            }
        ],
        n_underpowered=0,
        n_total=1,
        n_powered=1,
        n_power_denom=1,
        avg_power=0.9,
        avg_power_display="90",
        avg_power_known=True,
        power_target=0.8,
        total_current_n=10,
        total_required_n=10,
        total_required_known=True,
        n_supplied=1,
        recommendations=[],
    ),
    ["n_ungraded"],
)

# The ranking twins carry `n_supplied` as well, because the finder falls back to
# `n_supplied - n_total` when no explicit count is given: a healthy twin that left
# n_supplied at nine would claim seven ungraded metrics in the same breath as
# n_ungraded=0, and an internally inconsistent twin is not a control.
_RANKING = _base(
    badges=[
        {"name": "m1", "label": "PASS", "color": "#059669", "bg": "#d1fae5"},
        {"name": "m2", "label": "PASS", "color": "#059669", "bg": "#d1fae5"},
    ],
    groups=[
        {
            "name": "g",
            "exposure": 0.5,
            "display": "50%",
            "width": 60,
            "color": "#000",
            "avg_position": 3.0,
            "median_position": 3.0,
            "exposure_display": "50%",
            "n": 10,
        }
    ],
    n_pass=2,
    n_total=2,
    verdict={"label": "FAIR", "color": "#059669", "bg": "#d1fae5"},
    not_assessable=False,
    not_assessable_reason="",
    recommendations=[],
)
_add_pair(
    "ranking_fairness",
    dict(_RANKING, n_ungraded=_N, n_supplied=2 + _N),
    dict(_RANKING, n_ungraded=0, n_supplied=2),
)

# report_card counts its own ungraded rows in Jinja (`metrics|rejectattr(...)`),
# so its twins differ in the ROWS rather than in a key. Same row count either
# way, so the layout does not move.
_add_pair(
    "report_card",
    _base(
        model_name="m",
        approved=True,
        metrics=[
            {"name": f"x{i}", "passed": True, "value": 0.1, "threshold": 0.2} for i in range(2)
        ]
        + [{"name": f"u{i}", "passed": None, "value": None, "threshold": 0.2} for i in range(_N)],
        blocking_reasons=[],
        card_warnings=[],
    ),
    _base(
        model_name="m",
        approved=True,
        metrics=[
            {"name": f"x{i}", "passed": True, "value": 0.1, "threshold": 0.2} for i in range(2 + _N)
        ],
        blocking_reasons=[],
        card_warnings=[],
    ),
)

_add(
    "reweighting_comparison_report",
    _base(
        methods=[
            {
                "name": "m",
                "disparity": 0.05,
                "display": "0.050",
                "width": 60,
                "color": "#000",
                "graded": True,
                "improvement": 0.1,
                "fairness_improvement": 0.1,
                "accuracy_change": -0.01,
                "calibration_change": 0.0,
                "trade_off_score": 0.5,
                "accuracy_display": "0.800",
                "accuracy": 0.8,
            }
        ],
        best_method="m",
        original_disparity=0.1,
        max_trade_off=0.05,
        best_improvement=0.1,
        recommendations=[],
    ),
    ["n_ungraded"],
)

_add(
    "robustness_testing",
    _base(
        score_known=True,
        overall_score=0.9,
        overall_color="#059669",
        overall_label="PASS",
        not_assessable=False,
        not_assessable_reason="",
        nothing_tested=False,
        is_example=False,
        perm_tests=[
            {
                "method": "p",
                "p_value": 0.4,
                "graded": True,
                "stable": True,
                "color": "#059669",
                "observed": 0.05,
                "null_mean": 0.04,
                "null_std": 0.01,
                "max_deviation": 0.02,
            }
        ],
        sens_bars=[
            {
                "type": "s",
                "score": 0.9,
                "graded": True,
                "measured": True,
                "is_robust": True,
                "max_deviation": 0.02,
                "direction": "n/a",
                "bar_w": 180,
                "color": "#059669",
            }
        ],
        subgroup={"n_analyzed": 6, "n_flagged": 2, "gerrymandering": False, "worst_disparity": 0.1},
        flagged=[],
        recommendations=[],
    ),
    ["n_perm_ungraded"],
)

_add(
    "temporal_analysis",
    _base(
        metric_summaries=[
            {
                "name": "m",
                "mean": 0.5,
                "display": "0.50",
                "n_days": 30,
                "tracked": True,
                "std": 0.01,
                "latest": 0.5,
                "color": "#059669",
                "min_x": 60,
                "range_w": 200,
                "min_val": 0.4,
                "max_val": 0.6,
                "range_color": "#059669",
                "trend": "stable",
                "slope_display": "+0.0000",
                "slope_color": "#059669",
            }
        ],
        weekly_pattern=[],
        weekly_metric="m",
        trends=[
            {
                "name": "m",
                "tracked": True,
                "direction": "stable",
                "slope": 0.0,
                "n_days": 30,
                "color": "#059669",
            }
        ],
        degradations=[{"name": "m", "ran": True, "degraded": False, "worst_day": 0.5}],
        recommendations=[],
    ),
    ["n_untracked"],
)

_add(
    "transformation_comparison",
    _base(
        features=[{"name": "f", "measured": True, "before": 0.5, "after": 0.2, "reduction": 0.6}],
        threshold=0.3,
        avg_reduction=0.6,
        avg_color="#059669",
        n_improved=1,
        n_total=1,
        summary_bg="#d1fae5",
        summary_color="#059669",
        summary_icon="OK",
        summary_text="Effective",
    ),
    ["n_unmeasured"],
)

_TRAINING = _base(
    task_type="classification",
    n_samples=1000,
    n_groups=3,
    n_features=12,
    attribute_name="gender",
    baseline_accuracy=0.85,
    baseline_violation=0.04,
    baseline_satisfied=True,
    baseline_measured=True,
    violation_measured=True,
    constraint_evaluated=True,
    recommendation={
        "method": "reweighing",
        "priority": "low",
        "rationale": "ok",
        "given": True,
        "alternatives": [],
    },
    issues=[],
    actions=[],
)
_TR_GRADED = {
    "name": "g",
    "accuracy": 0.84,
    "violation": 0.03,
    "satisfied": True,
    "graded": True,
    "accuracy_measured": True,
    "violation_measured": True,
    "time": 1.2,
}
_TR_UNGRADED = {
    "name": "u",
    "accuracy": None,
    "violation": None,
    "satisfied": None,
    "graded": False,
    "accuracy_measured": False,
    "violation_measured": False,
    "time": None,
}
# training_report also counts its own rows, for the same reason as report_card:
# its adapter emits no ungraded key at all, so the template derives the count
# from `m.graded`, the same signal the table's NOT CHECKED chips are drawn from.
_add_pair(
    "training_report",
    dict(_TRAINING, methods=[_TR_GRADED] * 2 + [_TR_UNGRADED] * _N),
    dict(_TRAINING, methods=[_TR_GRADED] * (2 + _N)),
)

_EFFECT = {
    "name": "i",
    "effect": 0.1,
    "effect_display": "+0.1000",
    "effect_known": True,
    "ci": "[0.00, 0.20]",
    "p_display": "0.030",
    "significant": True,
    "significant_known": True,
    "n_display": "40 / 40",
    "dot_x": 380,
    "lo_x": 340,
    "hi_x": 420,
    "color": "#0ea5e9",
    "plotted": True,
}
_add(
    "experiment_results",
    _base(
        design_type="Independent",
        design_known=True,
        n_intersections=9,
        attribute_names="gender x race",
        nothing_measured=False,
        overall_effect="+0.1200",
        overall_effect_known=True,
        overall_p=0.03,
        overall_p_display="0.0300",
        overall_p_known=True,
        overall_significant=True,
        heterogeneity_known=True,
        heterogeneity_detected=False,
        heterogeneity_p_known=True,
        heterogeneity_p=0.4,
        heterogeneity_p_display="0.400",
        experiment_metrics_known=True,
        effects=[_EFFECT],
        forest_rows=[_EFFECT],
        n_excluded=0,
        n_powered=1,
        n_power_denom=1,
        n_power_graded=1,
        could_not_check_text="COULD NOT CHECK",
    ),
    ["n_power_ungraded", "forest_omitted"],
)


# cicd_pipeline derives its count in Jinja too (`_tests` minus `_graded_tests`),
# so the twins differ in whether a check ran, not in a key. A check the suite
# SKIPPED carries `passed=None`; the graded ones carry True or False.
_CICD = _base(
    validation={"state": "ran", "passed": True, "n_errors": 0, "n_warnings": 0, "issues": []},
    tests_state="ran",
    gate={"state": "ran", "status": "approved", "approved": True, "reasons": [], "warnings": []},
    all_passed=False,
)
_add_pair(
    "cicd_pipeline",
    dict(
        _CICD,
        tests=[
            {
                "name": "t",
                "passed": True,
                "status": "passed",
                "metric": "m",
                "value": 0.1,
                "threshold": 0.2,
            }
        ]
        + [
            {
                "name": f"s{i}",
                "passed": None,
                "status": "skipped",
                "metric": "m",
                "value": None,
                "threshold": 0.2,
            }
            for i in range(_N)
        ],
    ),
    dict(
        _CICD,
        tests=[
            {
                "name": f"t{i}",
                "passed": True,
                "status": "passed",
                "metric": "m",
                "value": 0.1,
                "threshold": 0.2,
            }
            for i in range(1 + _N)
        ],
        all_passed=True,
    ),
)

_add(
    "intersectional_disparity",
    _base(
        disparity_pp=4.0,
        severity="medium",
        severity_color="#f59e0b",
        severity_bg="#fef3c7",
        privileged={
            "group": "a",
            "rate": 0.5,
            "display": "50%",
            "n": 100,
            "predicted": 0.5,
            "actual": 0.5,
        },
        disadvantaged={
            "group": "b",
            "rate": 0.46,
            "display": "46%",
            "n": 100,
            "predicted": 0.46,
            "actual": 0.46,
        },
        subgroups=[
            {
                "group": "a",
                "rate": 0.5,
                "display": "50%",
                "n": 100,
                "width": 100,
                "color": "#000",
                "graded": True,
            }
        ],
        n_subgroups=1,
        recommendations=[],
    ),
    ["n_ungraded"],
)


# ───────────────────────────────────────────────────────────────────────────
# The five the discovery guard caught, 2026-08-28. Each of these templates was
# already drawing an ungraded count and NOTHING here placed it, which is the
# exact hole the guard above exists to find. One of them, experiment_recommendation,
# turned out to be drawing its count at y=382 rather than in the band, so it is a
# template fix and not just a missing fixture.
# ───────────────────────────────────────────────────────────────────────────

# calibration_disparity is handed its count as one adapter key, but its twins
# still cannot be a bare key flip: `n_groups` and `n_measured_groups` have to move
# with it, or the healthy control claims two measured groups out of nine while
# reporting nothing ungraded. Both twins carry the five rows the adapter's
# `groups[:5]` draws for a nine-group run, so the bar panel keeps its height and
# no panel below it moves.
#
# The subtitle is held at the adapter's NEUTRAL wording in both twins on purpose.
# The real adapter rewrites it to "7 of 9 group(s) reported no calibration error."
# on a partial run, and feeding that in here would hardcode the count into the
# fixture: the band assertion would then be satisfied by the fixture's own string
# even if the template's badge were deleted. Held identical, the only in-band line
# the count can add is one the TEMPLATE drew.
_CD = _base(
    subtitle="Compares calibration error across demographic groups.",
    overall_ece=0.012,
    overall_ece_color="#0f172a",
    ece_disparity=0.004,
    ece_disparity_color="#059669",
    disparity_ratio=1.4,
    disparity_ratio_text="1.40x",
    ratio_color="#059669",
    best_group="g0",
    best_ece=0.010,
    best_color="#059669",
    worst_group="g1",
    worst_ece=0.014,
    worst_color="#dc2626",
    # A verdict of False that was actually SUPPLIED, so both twins reach the green
    # branch and the recommendation panel is the same shape either side of the diff.
    has_significant_disparity=False,
    has_significant_disparity_known=True,
    not_assessable=False,
    not_assessable_text="NOT ASSESSABLE",
    not_assessable_reason="",
    unknown_color="#94a3b8",
    recommendation="",
    strategy="N/A",
)
_CD_MEASURED = [
    {"label": "g0", "ece": 0.010, "ece_scaled": 0.10, "color": "#0ea5e9"},
    {"label": "g1", "ece": 0.014, "ece_scaled": 0.14, "color": "#8b5cf6"},
]
# A group that reported no calibration error: `|f3` prints None as "N/A" and the
# bar collapses to its 4px minimum, which is what the adapter builds for one.
_CD_UNGRADED = [
    {"label": f"g{i}", "ece": None, "ece_scaled": 0.0, "color": "#0ea5e9"} for i in range(2, 5)
]
_CD_GRADED = [
    {"label": f"g{i}", "ece": 0.012, "ece_scaled": 0.12, "color": "#0ea5e9"} for i in range(2, 5)
]
_add_pair(
    "calibration_disparity",
    dict(
        _CD,
        groups=_CD_MEASURED + _CD_UNGRADED,
        n_groups=2 + _N,
        n_measured_groups=2,
        n_ungraded_groups=_N,
    ),
    dict(
        _CD,
        groups=_CD_MEASURED + _CD_GRADED,
        n_groups=2 + _N,
        n_measured_groups=2 + _N,
        n_ungraded_groups=0,
    ),
)

# causal_decomposition prints its count in the subtitle from the key the adapter
# computes (`len(bk_steps) - n_steps_graded`), so there is one key to flip and
# both twins share a single four-step ladder. _N exceeds the four rungs on
# purpose: the ladder is four steps by definition, the template's denominator is
# a literal 4, and a count of 2 or 3 would collide with the "n/4 satisfied"
# clause in the very same line, which is the collision _N was chosen to avoid.
_BK_STEP = {"label": "s", "satisfied": True, "step_known": True}
_add(
    "causal_decomposition",
    _base(
        measured=True,
        is_example=False,
        mediator="m",
        temporal=None,
        total_effect=0.045,
        direct_effect=0.028,
        indirect_effect=0.017,
        direct_w=174,
        indirect_w=105,
        proportion_mediated_known=True,
        prop_med_pct=38,
        prop_med_arc=68,
        bk_steps=[_BK_STEP] * 4,
        n_steps_ok=4,
        recommendations=[],
    ),
    ["n_steps_ungraded"],
)

# effect_sizes puts BOTH headline tiles over the measured subset: MAX EFFECT is a
# maximum and AVG |d| a mean, and a maximum taken over four of six metrics is not
# the maximum. The count is a chip beside those two tiles rather than a subtitle
# clause, because `_fit_subtitle` trims the 644px line and the adapter puts its
# "Not an effect size: ..." prefix FIRST, so a subtitle caveat can be trimmed off
# the canvas entirely.
_add(
    "effect_sizes",
    _base(
        effects=[
            {
                "label": "Statistical Parity Difference",
                "value": 0.12,
                "label_short": "ST",
                "px_offset": 14,
                "color": "#059669",
                "interpretation": "Negligible",
                "badge_bg": "#d1fae5",
                "badge_color": "#059669",
            }
        ],
        max_effect=0.12,
        max_effect_color="#059669",
        max_effect_interpretation="Negligible",
        avg_effect=0.12,
        avg_effect_color="#059669",
        not_assessable=False,
        not_assessable_reason="",
        excluded_groups=[],
    ),
    ["n_unmeasured"],
)

# experiment_recommendation draws its count in the arm-totals line at the FOOT
# of the EXPERIMENT METRICS panel, so its twins differ in one adapter-supplied
# key. `n_power_denom` stays at 1 in both: the template prints its own literal
# "not graded" for the Powered cell when the denominator is 0, and a healthy
# twin that tripped that would fail the over-correction control for a reason
# that has nothing to do with the count under test.
_add(
    "experiment_recommendation",
    _base(
        decision="DEPLOY_TREATMENT",
        decision_known=True,
        confidence=0.9,
        confidence_known=True,
        confidence_display="90",
        summary_text="Deploy the treatment, it improves fairness outcomes.",
        reasoning=[],
        tradeoffs=[],
        caveats=[],
        experiment_metrics_known=True,
        overall_effect="+0.1200",
        overall_effect_known=True,
        overall_p=0.03,
        overall_p_display="0.0300",
        overall_p_known=True,
        heterogeneity_known=True,
        heterogeneity_detected=False,
        n_powered=1,
        n_power_denom=1,
        n_control=40,
        n_control_known=True,
        n_treatment=40,
        n_treatment_known=True,
        design_type="Independent",
        design_known=True,
        could_not_check_text="COULD NOT CHECK",
    ),
    ["n_power_ungraded"],
)

# radar_chart draws its count in a chip BESIDE the status badge (y=169), not in
# the subtitle: `_fit_subtitle` trims the 644px line, and on a partly measured
# run the appended sentence was the part it cut. A metric that is not plotted
# leaves no visible gap either, because the polygon just closes over fewer
# spokes, so the chip is the only thing that states it.
_add(
    "radar_chart",
    _base(
        rings=[{"fraction": ri / 4, "label": ""} for ri in range(1, 5)],
        axes=[
            {
                "label": "m1",
                "spoke_x": 340,
                "spoke_y": 222,
                "label_x": 340,
                "label_y": 182,
                "label_anchor": "middle",
            }
        ],
        data_points="340,292",
        data_dots=[
            {
                "x": 340,
                "y": 292,
                "value": 0.5,
                "value_x": 340,
                "value_y": 276,
                "value_anchor": "middle",
            }
        ],
        threshold_ring_fraction=0.5,
        polygon_color="#059669",
        status_color="#059669",
        status_bg="#d1fae5",
        status_icon="OK",
        status_text="Fair",
        not_assessable=False,
        not_assessable_reason="",
        excluded_groups=[],
        undirected_metrics=[],
        unbounded_metrics=[],
        n_unbounded=0,
    ),
    ["n_unmeasured"],
)

_TEMPLATES = sorted(_FIXTURES)
_TEXT_RE = re.compile(r'<text[^>]*\by="([-0-9.]+)"[^>]*>(.*?)</text>', re.S)


def _texts(svg):
    """Every drawn text element as (y, content), whitespace normalised."""
    return [
        (float(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()) for m in _TEXT_RE.finditer(svg)
    ]


def _count_lines(template):
    """The text elements the ungraded count itself put on the canvas.

    Found by difference rather than by phrase, so a chart that words its caveat
    a new way is still checked. The count value must appear in the line: a line
    that merely SHIFTED between the two renders is not a line that states how
    many rows went ungraded.
    """
    partial, healthy = _FIXTURES[template]
    graded_texts = _texts(render_svg(template, healthy))
    added = [t for t in _texts(render_svg(template, partial)) if t not in graded_texts]
    return [t for t in added if str(_N) in t[1]]


@pytest.mark.parametrize("template", _TEMPLATES)
def test_the_ungraded_count_is_stated_in_the_headline_band(template):
    """THE RULE. One assertion, every chart that draws a count."""
    lines = _count_lines(template)

    assert lines, (
        f"{template}: the partial render is identical to the fully graded one, so this "
        f"fixture places nothing and the rule below is being checked against a canvas "
        f"that never noticed the difference"
    )

    in_band = [(y, text) for y, text in lines if y <= _HEADLINE_BAND]
    assert in_band, (
        f"{template} states its ungraded count only at the foot of the canvas "
        f"(lowest at y={min(y for y, _ in lines):.0f}, band ends at {_HEADLINE_BAND}): "
        f"{sorted(lines)[0][1]!r}. A reader who takes the headline and stops reading is "
        f"the reader this rule protects, so the count belongs beside the headline."
    )


@pytest.mark.parametrize("template", _TEMPLATES)
def test_a_fully_graded_canvas_states_no_ungraded_count(template):
    """The over-correction control, and the proof the fixture is not vacuous.

    A chart that graded everything it was given must draw exactly what it always
    drew. Kept as its own test so a sabotage that removes the whole mechanism
    leaves THIS one green: unchanged healthy output is what must not move.
    """
    _partial, healthy = _FIXTURES[template]
    svg = render_svg(template, healthy)

    for phrase in ("not graded", "not monitored", "never tracked", "not in this"):
        assert phrase not in svg, (
            f"{template} caveats a canvas that graded every row it was given ({phrase!r})"
        )


# ───────────────────────────────────────────────────────────────────────────
# The discovery guard: a template cannot draw a count with no fixture here.
#
# Without it this file checks the sixteen charts that were known on the day it
# was written, which is exactly how `_PARTIAL_SIGNAL_KEYS` in
# tests/test_explain_partial_runs.py came to certify six signals while six more
# went unread for months.
# ───────────────────────────────────────────────────────────────────────────

_TEMPLATE_DIR = Path(ENGINE.__file__).parent / "templates"

# The same narrow shape used by the sibling guard in test_explain_partial_runs:
# "un" must be followed by a word this library uses for an ABSENCE, which keeps
# `n_underpowered` out. That is a measured finding about a subgroup that was
# sized and found short, not a row nobody graded.
_COUNT_VAR_RE = re.compile(
    r"n_(?:un(?:graded|measured|known|monitored|rated|tracked|checked)|[a-z]+_ungraded)[a-z_]*"
    r"|forest_omitted"
)


def _templates_that_draw_a_count():
    found = set()
    for path in sorted(_TEMPLATE_DIR.glob("*.svg")):
        if path.name.startswith("_"):
            continue
        for match in re.finditer(r"<text\b.*?</text>", path.read_text(encoding="utf-8"), re.S):
            body = match.group(0)
            # Only an INTERPOLATED count counts: `{% if n_ungraded %}` around a
            # sentence with no number in it states a state, not a count, and the
            # difference is the whole point of the rule.
            for out in re.finditer(r"\{\{(.*?)\}\}", body, re.S):
                if _COUNT_VAR_RE.search(out.group(1)):
                    found.add(path.stem)
    return found


def test_every_template_that_draws_an_ungraded_count_has_a_fixture_here():
    drawn = _templates_that_draw_a_count()

    assert drawn, "the scan found no template drawing a count at all, so it checks nothing"

    missing = sorted(drawn - set(_FIXTURES))
    assert not missing, (
        f"these templates draw an ungraded count and no fixture in this file places it, "
        f"so nothing checks that it reaches the headline band: {missing}"
    )


def test_the_fixture_table_does_not_outlive_its_templates():
    """The other direction. A fixture for a template that no longer draws a
    count is a test that passes by rendering nothing, and it hides the day the
    count was deleted."""
    available = {p.stem for p in _TEMPLATE_DIR.glob("*.svg")}

    gone = sorted(set(_FIXTURES) - available)
    assert not gone, f"these fixtures name a template that no longer exists: {gone}"
