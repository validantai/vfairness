"""A PARTLY measured run must not be described as a fully measured one.

An earlier wave taught every finder in :mod:`vfairness.rendering.explain` the
third state for a run that measured NOTHING: a data dict carrying
``not_assessable`` yields a COULD NOT CHECK clause, and
``tests/test_explain_matches_canvas.py`` sweeps every chart for it.

The PARTIAL case was left behind, and it is the commoner one. Reproduced on the
committed tree at 39cc1e6, every line below is what the accessible ``<desc>``
carried while the canvas beside it already said otherwise:

    hierarchical_gate    ->  "Hierarchical gate APPROVED - overall 2/2,
                              single-attr 0/0, intersectional 0/0 metrics
                              passed" (severity INFO) for a gate that evaluated
                              one level of three
    monitoring_dashboard ->  "0 of 4 fairness metrics alerting" where one metric
                              of the four was compared to a guardrail
    alert_timeline       ->  "0 of 5 recent monitoring windows raised alerts"
                              where two windows applied no guardrail at all
    calibration_report   ->  "ECE disparity ? across groups", a question mark
                              where a measurement would be
    power_analysis       ->  "0 of 3 intersections are underpowered" with two
                              more supplied and never graded
    auto_discovery       ->  "3 violation(s) (0 high-severity)" over two rows
                              that reported no severity at all

In each one the canvas is the honest surface and the sentence is the confident
one, and the sentence is what a screen reader reads out.

The rule these guards encode, decided 2026-08-28 (see the block comment above
``_ungraded_tail`` in explain.py):

  a) grade and band over the GRADED subset, so a partly measured run still
     gives a useful verdict on what WAS measured;
  b) never write an unqualified all-clear while anything is ungraded, in the
     prose OR in the machine-readable severity;
  c) state the ungraded count in the finding itself;
  d) an ungraded row never enters a numerator or a denominator that implies it
     was measured.

Deliberately NOT ten one-off assertions. Each case below is checked against the
same four rules, a static guard requires every partial signal an adapter emits
to be read by some finder, and every case carries a HEALTHY TWIN whose sentence
must stay clean, which is both the over-correction control and the proof that
the fixture is not vacuous.
"""

import ast
import re
from pathlib import Path

import pytest

from vfairness.rendering import explain as EX
from vfairness.rendering.explain import (
    _COULD_NOT_CHECK,
    _FINDERS,
    _NOT_COVERED,
    build_explanation,
)

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


def _desc(svg: str) -> str:
    """The accessible one-liner, which is all a screen-reader user receives."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "the rendered SVG carries no <desc>"
    return match.group(1)


def _is_could_not_check(text) -> bool:
    return str(text or "").strip().upper().startswith(_COULD_NOT_CHECK)


class Case:
    """One partial run and the fully graded twin it must not be confused with.

    ``population`` is graded + ungraded: the number that must never appear as a
    denominator, which is rule (d) in the only form a test can check without
    re-implementing each sentence.

    ``graded_clean`` says the rows this chart DID grade all came out fine, which
    is what makes the fabricated-breach rule checkable. Opt-in rather than
    inferred: whether a chart's graded subset is clean is a fact about the
    fixture, and guessing it from the sentence would be re-implementing the
    finder inside its own test.
    """

    def __init__(self, name, template, partial, healthy, n_ungraded, population, graded_clean=None):
        self.name = name
        self.template = template
        self.partial = partial
        self.healthy = healthy
        self.n_ungraded = n_ungraded
        self.population = population
        self.graded_clean = graded_clean

    def __repr__(self):
        return self.name


# ───────────────────────────────────────────────────────────────────────────
# The representative set. Every one of these dicts is the shape the chart's own
# adapter produces for a partly measured run, and the key each reads is named in
# _PARTIAL_SIGNAL_KEYS below so the two cannot drift apart.
# ───────────────────────────────────────────────────────────────────────────

_CASES = [
    Case(
        "hierarchical_gate: two of three levels evaluated nothing",
        "hierarchical_gate",
        {
            "approved": True,
            "n_overall_pass": 2,
            "n_overall_total": 2,
            "n_single_pass": 0,
            "n_single_total": 0,
            "n_intersect_pass": 0,
            "n_intersect_total": 0,
            "warnings": [],
        },
        {
            "approved": True,
            "n_overall_pass": 2,
            "n_overall_total": 2,
            "n_single_pass": 4,
            "n_single_total": 4,
            "n_intersect_pass": 2,
            "n_intersect_total": 2,
            "warnings": [],
        },
        n_ungraded=2,
        population=3,
    ),
    Case(
        "monitoring_dashboard: three of four metrics met no guardrail",
        "monitoring_dashboard",
        {"any_alert": False, "n_alerts": 0, "n_metrics": 4, "n_checked": 1, "sample_count": 900},
        {"any_alert": False, "n_alerts": 0, "n_metrics": 4, "n_checked": 4, "sample_count": 900},
        n_ungraded=3,
        population=4,
    ),
    Case(
        "alert_timeline: two of five windows applied no guardrail",
        "alert_timeline",
        {"n_total": 5, "n_alerted": 0, "n_clean": 3, "n_unmonitored": 2},
        {"n_total": 5, "n_alerted": 0, "n_clean": 5, "n_unmonitored": 0},
        n_ungraded=2,
        population=5,
    ),
    Case(
        "auto_discovery: two of three violations reported no severity",
        "auto_discovery",
        {
            "candidates": [{"name": "zip"}],
            "n_violations": 3,
            "n_high": 0,
            "n_ungraded": 2,
            "groups": [{"group": "a"}, {"group": "b"}],
        },
        {
            "candidates": [{"name": "zip"}],
            "n_violations": 3,
            "n_high": 0,
            "n_ungraded": 0,
            "groups": [{"group": "a"}, {"group": "b"}],
        },
        n_ungraded=2,
        population=3,
    ),
    Case(
        "calibration_report: the ECE was measured, the group spread was not",
        "calibration_report",
        {
            "ece": 0.031,
            "is_well_calibrated": True,
            "has_disparity": False,
            "disparity_assessable": False,
            "disparity_reason": "only one group's calibration error was supplied.",
            "ece_disparity": None,
            "n_groups": 3,
            "n_measured_groups": 1,
        },
        {
            "ece": 0.031,
            "is_well_calibrated": True,
            "has_disparity": False,
            "disparity_assessable": True,
            "ece_disparity": 0.012,
            "n_groups": 3,
            "n_measured_groups": 3,
        },
        n_ungraded=2,
        population=3,
    ),
    Case(
        "robustness_testing: one permutation test reported no p-value",
        "robustness_testing",
        {
            "overall_score": 0.86,
            "overall_label": "PASS",
            "n_perm_ungraded": 1,
            "subgroup": {"n_flagged": 0, "n_analyzed": 6},
        },
        {
            "overall_score": 0.86,
            "overall_label": "PASS",
            "n_perm_ungraded": 0,
            "subgroup": {"n_flagged": 0, "n_analyzed": 6},
        },
        n_ungraded=1,
        population=6,
    ),
    Case(
        "fairness_report: two metrics never reached a threshold",
        "fairness_report",
        {"fairness_score": 92, "n_passed": 3, "n_failed": 0, "n_unknown": 2},
        {"fairness_score": 92, "n_passed": 5, "n_failed": 0, "n_unknown": 0},
        n_ungraded=2,
        population=5,
    ),
    Case(
        "correlation_heatmap: four of six pairs were never computed",
        "correlation_heatmap",
        {
            "n_high_corr": 0,
            "n_features": 3,
            "threshold": 0.3,
            "n_cols": 2,
            "rows": [
                {"label": "a", "cells": [{"value": 0.11}, {"value": None}]},
                {"label": "b", "cells": [{"value": None}, {"value": None}]},
                {"label": "c", "cells": [{"value": 0.05}, {"value": None}]},
            ],
        },
        {
            "n_high_corr": 0,
            "n_features": 3,
            "threshold": 0.3,
            "n_cols": 2,
            "rows": [
                {"label": "a", "cells": [{"value": 0.11}, {"value": 0.02}]},
                {"label": "b", "cells": [{"value": 0.03}, {"value": 0.01}]},
                {"label": "c", "cells": [{"value": 0.05}, {"value": 0.04}]},
            ],
        },
        n_ungraded=4,
        population=6,
    ),
    Case(
        "experiment_results: two intersections graded no power, one no effect",
        "experiment_results",
        {
            "overall_effect": "+0.0120",
            "overall_p_display": "0.0300",
            "overall_p_known": True,
            "overall_significant": True,
            "heterogeneity_known": True,
            "heterogeneity_detected": False,
            "n_intersections": 4,
            "n_power_graded": 2,
            "n_power_ungraded": 2,
            "forest_omitted": 1,
        },
        {
            "overall_effect": "+0.0120",
            "overall_p_display": "0.0300",
            "overall_p_known": True,
            "overall_significant": True,
            "heterogeneity_known": True,
            "heterogeneity_detected": False,
            "n_intersections": 4,
            "n_power_graded": 4,
            "n_power_ungraded": 0,
            "forest_omitted": 0,
        },
        n_ungraded=2,
        population=4,
    ),
    Case(
        "power_analysis: two of five intersections were never graded",
        "power_analysis",
        {
            "n_underpowered": 0,
            "n_total": 3,
            "n_supplied": 5,
            "n_ungraded": 2,
            "avg_power_display": "88",
            "avg_power_known": True,
            "power_target": "80%",
        },
        {
            "n_underpowered": 0,
            "n_total": 5,
            "n_supplied": 5,
            "n_ungraded": 0,
            "avg_power_display": "88",
            "avg_power_known": True,
            "power_target": "80%",
        },
        n_ungraded=2,
        population=5,
    ),
    Case(
        "radar_chart: one metric could not be placed on an axis",
        "radar_chart",
        {"status_text": "Fair", "n_unmeasured": 1},
        {"status_text": "Fair", "n_unmeasured": 0},
        n_ungraded=1,
        population=5,
    ),
    Case(
        "metrics_bar_chart: one metric was never compared to a threshold",
        "metrics_bar_chart",
        {"n_passed": 4, "n_total": 4, "n_unknown": 1},
        {"n_passed": 5, "n_total": 5, "n_unknown": 0},
        n_ungraded=1,
        population=5,
    ),
    Case(
        "effect_sizes: two metrics had no computable effect size",
        "effect_sizes",
        {
            "max_effect": 0.11,
            "avg_effect": 0.08,
            "max_effect_interpretation": "Negligible",
            "n_unmeasured": 2,
        },
        {
            "max_effect": 0.11,
            "avg_effect": 0.08,
            "max_effect_interpretation": "Negligible",
            "n_unmeasured": 0,
        },
        n_ungraded=2,
        population=6,
    ),
    Case(
        "confidence_intervals: two metrics came out undefined",
        "confidence_intervals",
        {"n_significant": 0, "n_total": 3, "n_ci_computed": 3, "n_unmeasured": 2},
        {"n_significant": 0, "n_total": 3, "n_ci_computed": 3, "n_unmeasured": 0},
        n_ungraded=2,
        population=5,
    ),
    Case(
        "group_calibration: two of five groups reported no error",
        "group_calibration",
        {
            "ece_disparity": 0.02,
            "max_ece": 0.03,
            "has_disparity": False,
            "n_groups": 5,
            "n_measured_groups": 3,
            "worst_group": "a",
            "best_group": "b",
        },
        {
            "ece_disparity": 0.02,
            "max_ece": 0.03,
            "has_disparity": False,
            "n_groups": 5,
            "n_measured_groups": 5,
            "worst_group": "a",
            "best_group": "b",
        },
        n_ungraded=2,
        population=5,
    ),
    Case(
        "calibration_disparity: two of five groups were never ranked",
        "calibration_disparity",
        {
            "ece_disparity": 0.005,
            "disparity_ratio": 1.2,
            "n_groups": 5,
            "n_measured_groups": 3,
            "worst_group": "a",
            "best_group": "b",
        },
        {
            "ece_disparity": 0.005,
            "disparity_ratio": 1.2,
            "n_groups": 5,
            "n_measured_groups": 5,
            "worst_group": "a",
            "best_group": "b",
        },
        n_ungraded=2,
        population=5,
    ),
    Case(
        "report_card: a verdict with no metric row behind it",
        "report_card",
        {"approved": True, "model_name": "m", "metrics": [], "blocking_reasons": []},
        {
            "approved": True,
            "model_name": "m",
            "metrics": [{"passed": True}, {"passed": True}],
            "blocking_reasons": [],
        },
        n_ungraded=2,
        population=2,
    ),
    # ── The closing wave, 2026-08-28. Every case below is a finder whose
    # ADAPTER had already learned the third state per row and whose sentence had
    # not, so the accessible layer contradicted a canvas that was corrected.
    # Each partial dict is the shape that adapter produces.
    Case(
        # `all_passed` is an AND over every test row and a SKIPPED check carries
        # passed=None, so one ungraded check turned it False and the sentence
        # read "Pipeline failed" beside a green PASS tile: a fabricated breach,
        # which is not the safer error.
        "cicd_pipeline: one fairness check was skipped, so it never ran",
        "cicd_pipeline",
        {
            "all_passed": False,
            "validation": {"state": "ran", "passed": True, "n_errors": 0, "n_warnings": 0},
            "tests": [{"passed": True}, {"passed": None}],
            "tests_state": "ran",
            "n_tests_graded": 1,
            "n_tests_ungraded": 1,
            "gate": {"state": "ran", "status": "approved", "approved": True},
        },
        {
            "all_passed": True,
            "validation": {"state": "ran", "passed": True, "n_errors": 0, "n_warnings": 0},
            "tests": [{"passed": True}],
            "tests_state": "ran",
            "n_tests_graded": 1,
            "n_tests_ungraded": 0,
            "gate": {"state": "ran", "status": "approved", "approved": True},
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    Case(
        # The worst shape of rule (d): the ungraded row was counted OUT of the
        # numerator and INTO the denominator at once, so the third method was
        # reported as one that was tested and fell short.
        "method_comparison: one method reported no constraint result",
        "method_comparison",
        {
            "methods": [
                {"satisfied": True, "graded": True},
                {"satisfied": True, "graded": True},
                {"satisfied": None, "graded": False},
            ],
            "n_ungraded": 1,
        },
        {
            "methods": [
                {"satisfied": True, "graded": True},
                {"satisfied": True, "graded": True},
            ],
            "n_ungraded": 0,
        },
        n_ungraded=1,
        population=3,
        graded_clean=True,
    ),
    Case(
        # `n_total` is len(badges), which counts the unchecked metric too. This
        # is the only path that reaches the fraction: with an unchecked metric
        # and no failure the adapter sets not_assessable instead.
        "regression_fairness: a real failure beside a metric nobody checked",
        "regression_fairness",
        {
            "n_pass": 1,
            "n_total": 3,
            "badges": [
                {"name": "a", "label": "PASS"},
                {"name": "b", "label": "FAIL"},
                {"name": "c", "label": "NO DATA"},
            ],
            "groups": [1, 2],
            "not_assessable": False,
        },
        {
            "n_pass": 1,
            "n_total": 2,
            "badges": [{"name": "a", "label": "PASS"}, {"name": "b", "label": "FAIL"}],
            "groups": [1, 2],
            "not_assessable": False,
        },
        n_ungraded=1,
        population=3,
    ),
    Case(
        # The denominator was already the graded population here, so this one is
        # rule (b): "; all pass" and severity INFO over a subset, with a metric
        # the canvas had already named as ungraded beside the headline.
        "ranking_fairness: every graded metric passes, one was never checked",
        "ranking_fairness",
        {
            "n_pass": 2,
            "n_total": 2,
            "n_ungraded": 1,
            "n_supplied": 3,
            "badges": [],
            "groups": [1],
            "not_assessable": False,
        },
        {
            "n_pass": 2,
            "n_total": 2,
            "n_ungraded": 0,
            "n_supplied": 2,
            "badges": [],
            "groups": [1],
            "not_assessable": False,
        },
        n_ungraded=1,
        population=3,
        graded_clean=True,
    ),
    Case(
        # Three counts in one sentence: an untracked metric card, a trend that
        # was never fitted, and a weekly check that REFUSED to run and carried
        # degraded=False into the numerator as a metric found steady.
        "temporal_analysis: an untracked metric, an unfitted trend, a check that declined",
        "temporal_analysis",
        {
            "metric_summaries": [1, 2, 3],
            "n_untracked": 1,
            "trends": [
                {"slope": 0.0, "tracked": True},
                {"slope": 0.0, "tracked": True},
                {"slope": None, "tracked": False},
            ],
            "degradations": [
                {"degraded": False, "ran": True},
                {"degraded": False, "ran": False},
            ],
        },
        {
            "metric_summaries": [1, 2],
            "n_untracked": 0,
            "trends": [{"slope": 0.0, "tracked": True}, {"slope": 0.0, "tracked": True}],
            "degradations": [{"degraded": False, "ran": True}],
        },
        n_ungraded=1,
        population=3,
        graded_clean=True,
    ),
    Case(
        # The canvas carries three counters and no fourth, so the adapter files
        # an ungraded issue under WARNING and names the count beside it. The
        # sentence took the merged counter and read the ungraded row out as a
        # warning the validator raised.
        "data_validation: one issue carries an unrecognised severity",
        "data_validation",
        {
            "pass_label": "PASS",
            "passed": True,
            "n_critical": 0,
            "n_warning": 2,
            "n_info": 0,
            "issues": [{"bucket": "WARNING"}, {"bucket": "UNKNOWN"}],
        },
        {
            "pass_label": "PASS",
            "passed": True,
            "n_critical": 0,
            "n_warning": 2,
            "n_info": 0,
            "issues": [{"bucket": "WARNING"}, {"bucket": "WARNING"}],
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    Case(
        # The sensitivity half of the canvas, and a withheld numerator printed
        # as itself: `n_flagged` is None when the audit reported no flagged
        # count, and the key IS present, so the sentence read "None/6".
        "robustness_testing: no flagged count, and two ungraded sensitivity checks",
        "robustness_testing",
        {
            "overall_score": 0.9,
            "overall_label": "PASS",
            "n_perm_ungraded": 0,
            "n_sens_ungraded": 2,
            "subgroup": {"n_analyzed": 6, "n_flagged": None},
        },
        {
            "overall_score": 0.9,
            "overall_label": "PASS",
            "n_perm_ungraded": 0,
            "n_sens_ungraded": 0,
            "subgroup": {"n_analyzed": 6, "n_flagged": 2},
        },
        n_ungraded=2,
        population=8,
        graded_clean=True,
    ),
    Case(
        # The denominator is the measured subset already; the all-clear was the
        # problem. "Effective ... 3/3 features improved" at severity "low" over
        # a run that compared three pairs of five.
        "transformation_comparison: two features carried no before/after pair",
        "transformation_comparison",
        {
            "avg_reduction": 0.4,
            "summary_text": "Effective",
            "n_improved": 3,
            "n_total": 3,
            "n_unmeasured": 2,
        },
        {
            "avg_reduction": 0.4,
            "summary_text": "Effective",
            "n_improved": 3,
            "n_total": 3,
            "n_unmeasured": 0,
        },
        n_ungraded=2,
        population=5,
        graded_clean=True,
    ),
    Case(
        # A group that reported no rate is not on the plot and not in the gap.
        # The adapter names it in the subtitle; the sentence read as a verdict
        # over the whole attribute at severity INFO.
        "group_comparison: two groups reported no rate at all",
        "group_comparison",
        {
            "bars": [1, 2],
            "max_disparity": 0.05,
            "disparity_label": "Low",
            "metric_label": "Rate",
            "worst_group": "a",
            "best_group": "b",
            "n_ungraded_groups": 2,
        },
        {
            "bars": [1, 2],
            "max_disparity": 0.05,
            "disparity_label": "Low",
            "metric_label": "Rate",
            "worst_group": "a",
            "best_group": "b",
            "n_ungraded_groups": 0,
        },
        n_ungraded=2,
        population=4,
        graded_clean=True,
    ),
    Case(
        # "0 high-risk proxy variable(s) among 3 features" is the strongest
        # all-clear a proxy scan can give, and a candidate with no risk level
        # was neither cleared nor flagged: it was never scored.
        "proxy_risk: two candidates carried no risk level",
        "proxy_risk",
        {
            "risk_counts": [{"label": "LOW", "count": 3}],
            "features": [1, 2, 3],
            "n_unrated": 2,
            "unrated_features": [{"name": "d"}, {"name": "e"}],
        },
        {
            "risk_counts": [{"label": "LOW", "count": 3}],
            "features": [1, 2, 3],
            "n_unrated": 0,
            "unrated_features": [],
        },
        n_ungraded=2,
        population=5,
        graded_clean=True,
    ),
    Case(
        # A spread over three of eight cells is not a spread "across gender x
        # race", which is what the sentence claimed at severity "low".
        "intersectional_analysis: three of eight cells carried no rate",
        "intersectional_analysis",
        {
            "feature_name": "f",
            "max_disparity": 0.05,
            "x_attr": "x",
            "y_attr": "y",
            "n_unmeasured": 3,
            "n_measured": 5,
            "n_cells": 8,
        },
        {
            "feature_name": "f",
            "max_disparity": 0.05,
            "x_attr": "x",
            "y_attr": "y",
            "n_unmeasured": 0,
            "n_measured": 8,
            "n_cells": 8,
        },
        n_ungraded=3,
        population=8,
        graded_clean=True,
    ),
    # ── Wave 15, 2026-08-28: the ACCESSIBLE DESCRIPTIONS. Every case below is a
    # finder whose canvas had already grown a third state and whose sentence
    # still spoke for the whole run, so the <desc> was the more confident of the
    # two surfaces and it is the one a screen-reader user gets. Eight of them
    # also close the handoff left by the last wave: a finder that had learned
    # the partial clause with no fixture exercising it here.
    Case(
        # adapters._module_score stopped grading a finding whose severity it
        # could not read, and bias_audit.svg states the count at y=136. The
        # sentence reported the score and the critical count and nothing about
        # the findings behind either.
        "bias_audit: three findings reported no severity",
        "bias_audit",
        {
            "overall_score": 0.12,
            "overall_label": "MINIMAL",
            "coverage": "full",
            "n_critical": 0,
            "n_ungraded_findings": 3,
            "score_assessable": True,
        },
        {
            "overall_score": 0.12,
            "overall_label": "MINIMAL",
            "coverage": "full",
            "n_critical": 0,
            "n_ungraded_findings": 0,
            "score_assessable": True,
        },
        n_ungraded=3,
        population=8,
        graded_clean=True,
    ),
    Case(
        # `engine._risk_label` answers "N/A" for a score that is None, and
        # adapters.bias_audit_to_svg withholds the BADGE through its own
        # `score_assessable` while leaving `not_assessable` False on purpose,
        # because the findings are real. The severity map read that "N/A"
        # through `.get(..., "info")`, so an audit with a critical issue and no
        # aggregate score was announced at the all-clear grade, with the score
        # itself printed as a bare question mark.
        "bias_audit: a critical issue and no aggregate risk score",
        "bias_audit",
        {
            "overall_score": None,
            "overall_label": "N/A",
            "coverage": "full",
            "n_critical": 1,
            "n_ungraded_findings": 0,
            "score_assessable": False,
        },
        {
            "overall_score": 0.62,
            "overall_label": "MEDIUM",
            "coverage": "full",
            "n_critical": 1,
            "n_ungraded_findings": 0,
            "score_assessable": True,
        },
        n_ungraded=1,
        population=4,
    ),
    Case(
        # causal_decomposition.svg prints "N step(s) not reported" in its
        # subtitle; `n_steps_ok` deliberately does not count an ungraded step as
        # satisfied, so "3/4 steps satisfied" over a fourth step nobody ran
        # reported a step that WAS run and failed.
        "causal_decomposition: one Baron-Kenny step reported no verdict",
        "causal_decomposition",
        {
            "mediator": "education",
            "prop_med_pct": 42,
            "proportion_mediated_known": True,
            "n_steps_ok": 3,
            "n_steps_ungraded": 1,
            "temporal": {"is_stable": True},
        },
        {
            "mediator": "education",
            "prop_med_pct": 42,
            "proportion_mediated_known": True,
            "n_steps_ok": 4,
            "n_steps_ungraded": 0,
            "temporal": {"is_stable": True},
        },
        n_ungraded=1,
        population=8,
        graded_clean=True,
    ),
    Case(
        # The gauge is the largest thing on that canvas and the template draws
        # no needle without `proportion_mediated_known`, because "a 0% gauge is
        # the finding 'the mediator carries none of the gap'". The sentence
        # printed that finding from `_g(d, "prop_med_pct", 0)`.
        "causal_decomposition: no proportion mediated was reported",
        "causal_decomposition",
        {
            "mediator": "education",
            "prop_med_pct": None,
            "proportion_mediated_known": False,
            "n_steps_ok": 4,
            "n_steps_ungraded": 0,
            "temporal": {"is_stable": True},
        },
        {
            "mediator": "education",
            "prop_med_pct": 38,
            "proportion_mediated_known": True,
            "n_steps_ok": 4,
            "n_steps_ungraded": 0,
            "temporal": {"is_stable": True},
        },
        n_ungraded=1,
        population=4,
        graded_clean=True,
    ),
    Case(
        # The same all-clear the heatmap beside it had already had removed: a
        # pair that could not be computed cannot raise `n_high_corr`, so the
        # fewer pairs were tested the cleaner the sentence read.
        "correlation_matrix: one feature pair was never computed",
        "correlation_matrix",
        {
            "n_high_corr": 0,
            "n_features": 2,
            "threshold": 0.3,
            "method_display": "auto",
            "rows": [
                {"cells": [{"value": 1.0}, {"value": 0.12}]},
                {"cells": [{"value": 0.12}, {"value": None}]},
            ],
        },
        {
            "n_high_corr": 0,
            "n_features": 2,
            "threshold": 0.3,
            "method_display": "auto",
            "rows": [
                {"cells": [{"value": 1.0}, {"value": 0.12}]},
                {"cells": [{"value": 0.12}, {"value": 1.0}]},
            ],
        },
        n_ungraded=1,
        population=4,
        graded_clean=True,
    ),
    Case(
        # "across 3 group(s) and 4 metric(s)" for a grid drawing one of those
        # groups entirely N/A. The counts are re-derived cell by cell from the
        # same rows the template draws.
        "disparity_heatmap: one group reported no value on any metric",
        "disparity_heatmap",
        {
            "summary_icon": "✓",
            "summary_text": "Low disparity across groups",
            "n_rows": 3,
            "n_cols": 2,
            "rows": [
                {"label": "male", "cells": [{"value": 0.51}, {"value": 0.48}]},
                {"label": "female", "cells": [{"value": 0.47}, {"value": 0.44}]},
                {"label": "nonbinary", "cells": [{"value": None}, {"value": None}]},
            ],
        },
        {
            "summary_icon": "✓",
            "summary_text": "Low disparity across groups",
            "n_rows": 2,
            "n_cols": 2,
            "rows": [
                {"label": "male", "cells": [{"value": 0.51}, {"value": 0.48}]},
                {"label": "female", "cells": [{"value": 0.47}, {"value": 0.44}]},
            ],
        },
        n_ungraded=1,
        population=3,
        graded_clean=True,
    ),
    Case(
        # The overall drift score is the MAXIMUM over the scales that reported
        # one, so a scale that reported nothing cannot raise it and the fewer
        # scales reported, the calmer the number read.
        "drift_report: one temporal scale did not fully report",
        "drift_report",
        {
            "metric_name": "demographic_parity",
            "drift_detected": False,
            "overall_verdict_recorded": True,
            "overall_drift_score": 0.05,
            "n_partial_scales": 1,
        },
        {
            "metric_name": "demographic_parity",
            "drift_detected": False,
            "overall_verdict_recorded": True,
            "overall_drift_score": 0.05,
            "n_partial_scales": 0,
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    Case(
        # "0 breach(es) of 1" plus a GREEN health status at severity "info" is
        # the strongest all-clear this dashboard can give, with the unchecked
        # row named only in the narrative panel at the foot of the canvas.
        "reporting_dashboard: one metric was never compared to a threshold",
        "reporting_dashboard",
        {
            "status_label": "GREEN",
            "report_tier": "OPERATIONAL",
            "health_score": "88",
            "breaches": [],
            "n_metrics": 1,
            "n_active_alerts": 0,
            "n_critical_alerts": 0,
            "metrics": [{"could_not_check": False}, {"could_not_check": True}],
        },
        {
            "status_label": "GREEN",
            "report_tier": "OPERATIONAL",
            "health_score": "88",
            "breaches": [],
            "n_metrics": 2,
            "n_active_alerts": 0,
            "n_critical_alerts": 0,
            "metrics": [{"could_not_check": False}, {"could_not_check": False}],
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    Case(
        # BEST METHOD is a recommendation, and the canvas paints it emerald only
        # over a graded row. The other half is `n_ungraded`: supplied method
        # results that reported no fairness, accuracy, calibration or trade-off
        # number at all, named in the subtitle and in no sentence.
        "reweighting_comparison_report: two supplied methods reported nothing",
        "reweighting_comparison_report",
        {
            "best_method": "inverse_propensity",
            "best_method_graded": True,
            "original_disparity": 0.02,
            "methods": [{"name": "inverse_propensity"}, {"name": "class_balance"}],
            "n_ungraded": 2,
        },
        {
            "best_method": "inverse_propensity",
            "best_method_graded": True,
            "original_disparity": 0.02,
            "methods": [{"name": "inverse_propensity"}, {"name": "class_balance"}],
            "n_ungraded": 0,
        },
        n_ungraded=2,
        population=4,
        graded_clean=True,
    ),
    Case(
        # adapters_training puts the withheld count in the TITLE of this chart,
        # in the headline band, because the subtitle is fixed prose. The
        # sentence underneath spoke for the plotted subset as though it were the
        # whole sweep.
        "tradeoff_analysis: two trade-off points were never plotted",
        "tradeoff_analysis",
        {
            "points": [{"satisfied": True}, {"satisfied": True}, {"satisfied": True}],
            "pareto_points": [{}, {}],
            "n_withheld_points": 2,
        },
        {
            "points": [{"satisfied": True}, {"satisfied": True}, {"satisfied": True}],
            "pareto_points": [{}, {}],
            "n_withheld_points": 0,
        },
        n_ungraded=2,
        population=5,
        graded_clean=True,
    ),
    Case(
        # `{% if any_alert %}` sends every falsy value to the emerald OK branch,
        # so monitoring_dashboard.svg draws NOT RECORDED in slate off
        # `any_alert_recorded` and replaces the "0 alerts" pill with "no
        # guardrail applied". This finder read only the bool.
        "monitoring_dashboard: the window recorded no alert verdict",
        "monitoring_dashboard",
        {
            "any_alert": False,
            "any_alert_recorded": False,
            "n_alerts": 0,
            "n_metrics": 4,
            "n_checked": 4,
            "sample_count": 900,
        },
        {
            "any_alert": False,
            "any_alert_recorded": True,
            "n_alerts": 0,
            "n_metrics": 4,
            "n_checked": 4,
            "sample_count": 900,
        },
        n_ungraded=1,
        population=8,
        graded_clean=True,
    ),
    Case(
        # experiment_results.svg withholds the 22px headline estimate entirely
        # without `overall_effect_known`; its own note calls that number "the
        # last fabrication left on this canvas". The sentence read out the
        # adapter's replacement words at the all-clear severity.
        "experiment_results: no overall treatment effect was estimated",
        "experiment_results",
        # The p-value IS reported here and the effect estimate is not, which is
        # the combination that made this the last fabrication on the canvas: a
        # real test beside a headline number nobody computed. Keeping the two
        # apart is also what makes this case non-vacuous, because an unknown
        # p-value writes its own could-not-check clause and would satisfy the
        # rule below whatever this finder did with the effect.
        {
            "overall_effect": "not computed",
            "overall_effect_known": False,
            "overall_p_display": "0.4000",
            "overall_p_known": True,
            "overall_significant": False,
            "heterogeneity_known": True,
            "heterogeneity_detected": False,
            "n_intersections": 4,
            "n_power_ungraded": 0,
            "forest_omitted": 0,
        },
        {
            "overall_effect": "+0.0120",
            "overall_effect_known": True,
            "overall_p_display": "0.0300",
            "overall_p_known": True,
            "overall_significant": True,
            "heterogeneity_known": True,
            "heterogeneity_detected": False,
            "n_intersections": 4,
            "n_power_ungraded": 0,
            "forest_omitted": 0,
        },
        n_ungraded=1,
        population=8,
        graded_clean=True,
    ),
    Case(
        # The regression page has three panels and this sentence read one. The
        # subtitle at y=122 already says "1 pair(s) not compared"; the <desc>
        # said "4/4 disparity metrics pass across 2 groups; all pass" at INFO.
        "regression_fairness: a pair never compared and a group never screened",
        "regression_fairness",
        {
            "n_pass": 2,
            "n_total": 2,
            "badges": [{"name": "a", "label": "PASS"}, {"name": "b", "label": "PASS"}],
            "groups": [1, 2, 3],
            "not_assessable": False,
            "n_effects_unmeasured": 1,
            "n_residual_missing": 1,
        },
        {
            "n_pass": 2,
            "n_total": 2,
            "badges": [{"name": "a", "label": "PASS"}, {"name": "b", "label": "PASS"}],
            "groups": [1, 2, 3],
            "not_assessable": False,
            "n_effects_unmeasured": 0,
            "n_residual_missing": 0,
        },
        n_ungraded=1,
        population=4,
        graded_clean=True,
    ),
    Case(
        # A group that reported no exposure is drawn without a bar, is out of
        # the max-exposure scale and out of the exposure gap, and the subtitle
        # says so. "across 3 groups" claimed the population the gap was never
        # taken over.
        "ranking_fairness: one group reported no exposure",
        "ranking_fairness",
        {
            "n_pass": 2,
            "n_total": 2,
            "n_ungraded": 0,
            "n_supplied": 2,
            "badges": [],
            "groups": [
                {"name": "a", "measured": True},
                {"name": "b", "measured": True},
                {"name": "c", "measured": False},
            ],
            "not_assessable": False,
        },
        {
            "n_pass": 2,
            "n_total": 2,
            "n_ungraded": 0,
            "n_supplied": 2,
            "badges": [],
            "groups": [
                {"name": "a", "measured": True},
                {"name": "b", "measured": True},
                {"name": "c", "measured": True},
            ],
            "not_assessable": False,
        },
        n_ungraded=1,
        population=3,
        graded_clean=True,
    ),
    Case(
        # `disparity_pp` is None when no gap was reported and `severity_graded`
        # is False when nobody banded one. The canvas draws NOT GRADED in slate
        # and prints N/A through its |f1 filter; the sentence printed a bare
        # question mark where the measurement would be.
        "intersectional_disparity: no gap and no band were reported",
        "intersectional_disparity",
        {
            "disparity_pp": None,
            "severity": "NOT GRADED",
            "severity_graded": False,
            "disadvantaged": {"group": "female x 50+"},
            "privileged": {"group": "male x 30-40"},
        },
        {
            "disparity_pp": 4.2,
            "severity": "MEDIUM",
            "severity_graded": True,
            "disadvantaged": {"group": "female x 50+"},
            "privileged": {"group": "male x 30-40"},
        },
        n_ungraded=1,
        population=6,
    ),
    Case(
        # The second silence on the transformation headline, and the one the
        # template already folds into the same NOT MEASURED pill: a feature that
        # started at |r| = 0 has both halves of its pair and still no
        # proportional reduction, so it is in neither the average nor `n_total`.
        "transformation_comparison: a feature started at zero correlation",
        "transformation_comparison",
        {
            "avg_reduction": 0.4,
            "summary_text": "Effective",
            "n_improved": 1,
            "n_total": 1,
            "n_unmeasured": 0,
            "n_ratio_ungraded": 1,
        },
        {
            "avg_reduction": 0.4,
            "summary_text": "Effective",
            "n_improved": 2,
            "n_total": 2,
            "n_unmeasured": 0,
            "n_ratio_ungraded": 0,
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    Case(
        # Added 2026-09-30 with the fix in _fr_training_report. The baseline
        # panel grades two INDEPENDENT cells, and the finder formatted both from
        # the raw values with a `0` default while the canvas beside it correctly
        # said "Accuracy and fairness violation were not measured on this run".
        # An accuracy the producer had disowned (accuracy=nan carried with
        # parameters['accuracy_measured']=False) reached the <desc> as "Baseline
        # accuracy nan" at severity LOW; an absent one reached it as "accuracy
        # ?". Both dicts below are exactly what _baseline_state emits.
        "training_report: the baseline's accuracy was never scored",
        "training_report",
        {
            "baseline_accuracy": None,
            "baseline_violation": 0.03,
            "baseline_satisfied": True,
            "baseline_accuracy_measured": False,
            "baseline_violation_measured": True,
            "baseline_accuracy_reason": "no row carried a scorable target",
            "baseline_measured": True,
            "constraint_evaluated": True,
            "recommendation": {"method": "Reweight", "priority": "low", "given": True},
            "issues": [],
            "methods": [],
        },
        {
            "baseline_accuracy": 0.82,
            "baseline_violation": 0.03,
            "baseline_satisfied": True,
            "baseline_accuracy_measured": True,
            "baseline_violation_measured": True,
            "baseline_accuracy_reason": "",
            "baseline_measured": True,
            "constraint_evaluated": True,
            "recommendation": {"method": "Reweight", "priority": "low", "given": True},
            "issues": [],
            "methods": [],
        },
        n_ungraded=1,
        population=2,
        graded_clean=True,
    ),
    # Added 2026-09-30 (G11). pareto_frontier_to_svg filters its points to the
    # ones with FINITE coordinates and then reports "n_frontier of n_points",
    # where n_points is the count that SURVIVED, so "2 of 2 configurations are
    # Pareto-optimal" read as full coverage of a set that was one larger. The
    # canvas already said "1 of 3 configuration(s) have non-finite coordinates
    # and are excluded"; the <desc> did not, at severity info. Total loss was
    # already refused by _pareto_not_assessable, so this was the PARTIAL case.
    Case(
        "pareto_frontier: one of three configurations had no finite coordinates",
        "pareto_frontier",
        {
            "n_frontier": 2,
            "n_points": 2,
            "best_label": "a",
            "n_supplied": 3,
            "n_dropped": 1,
        },
        {
            "n_frontier": 3,
            "n_points": 3,
            "best_label": "b",
            "n_supplied": 3,
            "n_dropped": 0,
        },
        n_ungraded=1,
        population=3,
    ),
]

_COVERED_TEMPLATES = {c.template for c in _CASES}


# ───────────────────────────────────────────────────────────────────────────
# The four rules, applied to every case rather than written out sixteen times.
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_the_description_names_what_was_not_graded(case):
    """Rule (c). The count reaches the sentence, not only the chart's subtitle.

    Two shapes are allowed and no third. Either the finding carries an explicit
    COULD NOT CHECK clause, which names the half of the chart that was withheld
    (the calibration report keeps its measured ECE and withholds the group
    spread in one sentence), or it states the ungraded COUNT and closes with the
    shared not-covered marker. Silence is not a shape.
    """
    finding = build_explanation(case.template, case.partial).finding
    assert finding, f"{case.template} produced no finding at all"

    if _COULD_NOT_CHECK in finding:
        return
    assert _NOT_COVERED in finding, (
        f"{case.template} described a partial run with no caveat at all: {finding!r}"
    )
    assert re.search(rf"\b{case.n_ungraded}\b", finding), (
        f"{case.template} does not say HOW MANY rows went ungraded "
        f"(expected {case.n_ungraded}): {finding!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_a_partial_run_never_carries_the_all_clear_severity(case):
    """Rule (b), in the form a machine reads.

    ``metadata()["severity"]`` is what a report pipeline, an assistant or a
    downstream dashboard consumes INSTEAD of the sentence, so a qualifying
    clause in the prose does not save an "info" here.
    """
    explanation = build_explanation(case.template, case.partial)

    assert explanation.severity != "info", (
        f"{case.template} graded a partial run as benign news: {explanation.finding!r}"
    )


# The absolute quantifiers this library writes into a finding. Short and
# reviewed by hand rather than inferred, because the point is that ANY new one
# has to be looked at: the guard below refuses a literal that is not on this
# list, so a finder cannot invent a seventeenth way to say "everything is fine"
# and slip past a phrase test that only knew sixteen.
_ALL_CLEAR_LITERALS = ("; all pass",)


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_a_partial_run_never_writes_an_unqualified_all_clear(case):
    """Rule (b) in the PROSE, which the severity check above does not cover.

    "Ranking fairness: 2/2 metrics pass across 1 groups; all pass" is true of
    the graded subset and false of the chart, and the qualifying clause that
    follows it does not unsay it: "all pass" is the phrase a reader quotes.
    Severity alone does not catch this, because the ungraded clause floors the
    severity while those two words stay on the line.
    """
    finding = build_explanation(case.template, case.partial).finding

    for phrase in _ALL_CLEAR_LITERALS:
        assert phrase not in finding, (
            f"{case.template} wrote an unqualified all-clear ({phrase!r}) over a run "
            f"with {case.n_ungraded} ungraded row(s): {finding!r}"
        )


def test_the_all_clear_list_still_covers_what_the_finders_can_write():
    """The guard on the guard above. An absolute quantifier that reaches a
    finding without being on the list is checked by nothing."""
    # Parsed, not grepped. This file and explain.py both QUOTE the sentences
    # they are about ("Ranking fairness: 6/7 metrics pass ...; all pass" is in
    # _pass_tail's docstring, describing the incident), and a raw grep reads
    # those narratives as writable output. Comments never reach the AST at all,
    # and docstrings are dropped explicitly, so what is left is the strings this
    # module can actually put in front of a reader.
    tree = ast.parse(Path(EX.__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant):
                docstrings.add(id(doc.value))

    pattern = re.compile(r"\ball (?:pass|clear|fine|good)")
    written = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and pattern.search(node.value)
    }

    unreviewed = sorted(w for w in written if w not in _ALL_CLEAR_LITERALS)
    assert not unreviewed, (
        f"explain.py can write these absolute all-clears and _ALL_CLEAR_LITERALS does "
        f"not list them, so no case checks that a partial run withholds them: {unreviewed}"
    )


@pytest.mark.parametrize("case", [c for c in _CASES if c.graded_clean], ids=repr)
def test_a_partial_run_is_never_reported_as_a_breach_it_did_not_measure(case):
    """The OTHER direction, and the one that reads as the safe error.

    A fabricated breach is exactly as wrong as a fabricated all-clear and worse
    to act on: the reader chases a failure, cannot find it, and stops believing
    the tool. The CI/CD pipeline is where this bit. ``all_passed`` is an AND
    over every test row and a SKIPPED check carries ``passed=None``, which is
    falsy, so one check that never ran made the sentence read "Pipeline failed"
    at severity HIGH beside a canvas whose tests tile was green and said "N
    passed, 0 failed, 1 not run".

    Severity is the machine-readable form: "high" and "critical" are what a
    report pipeline escalates on, and neither is claimable when every row that
    WAS graded came out clean. The unresolved unknown floors at medium, which is
    what _floor_sev exists to do.
    """
    explanation = build_explanation(case.template, case.partial)

    assert explanation.severity not in ("high", "critical"), (
        f"{case.template} reported a breach on a run whose graded rows all came out "
        f"clean, at severity {explanation.severity}: {explanation.finding!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_the_ungraded_rows_never_become_a_denominator(case):
    """Rule (d). "0 of 4 alerting" over one checked metric is the whole defect.

    The more rows went ungraded, the cleaner such a fraction reads, which is
    exactly backwards.
    """
    finding = build_explanation(case.template, case.partial).finding

    assert not re.search(rf"\bof {case.population}\b", finding), (
        f"{case.template} counted ungraded rows into a denominator of "
        f"{case.population}: {finding!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_a_withheld_number_is_never_printed_as_a_question_mark(case):
    """The calibration report's literal output: "ECE disparity ? across groups".

    ``_f(None)`` renders a bare question mark, which is neither a measurement
    nor a statement that there is none. A reader cannot act on it and a screen
    reader reads it as punctuation. Whatever could not be measured is SAID, in
    words, or it is not on the line at all.

    "None" and "nan" are the same defect in a different token, and they reach a
    sentence the same way: an f-string slot whose value was withheld. The
    robustness dashboard printed "None/6 subgroups flagged" for an audit that
    reported no flagged count, because ``_g(sg, "n_flagged", 0)`` returns the
    None that is actually stored rather than the default, the KEY being present.
    A reader cannot tell that from a rendering fault, which is the one reading
    worse than a wrong number.
    """
    finding = build_explanation(case.template, case.partial).finding

    assert "?" not in finding, (
        f"{case.template} printed a withheld value as a question mark: {finding!r}"
    )
    for token in ("None", "nan", "NaN"):
        assert not re.search(rf"(?<![A-Za-z]){token}(?![A-Za-z])", finding), (
            f"{case.template} printed a withheld value as the bare token {token!r}, which "
            f"reads as a rendering fault rather than as a measurement that was not taken: "
            f"{finding!r}"
        )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_no_fraction_is_printed_over_a_denominator_of_zero(case):
    """Rule (d) in its other shape. "single-attr 0/0 metrics passed" is not a
    level that passed, and "0/0 metrics pass" is not a pass rate: it is the
    arithmetic of a row nobody graded, printed as though it were measured."""
    finding = build_explanation(case.template, case.partial).finding

    assert not re.search(r"\b\d+/0\b", finding), (
        f"{case.template} printed a fraction over an empty denominator: {finding!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_the_fully_graded_twin_is_left_alone(case):
    """The over-correction control, on its own so it can never be collateral.

    A chart that graded everything it was given must read exactly as it always
    did: no caveat, no COULD NOT CHECK. Kept separate from the non-vacuity check
    below deliberately: a sabotage that removes the whole mechanism must leave
    THIS test green, because healthy output is exactly what must not move.
    """
    healthy = build_explanation(case.template, case.healthy).finding

    assert healthy, f"{case.template} produced no finding for a fully graded run"
    assert _NOT_COVERED not in healthy, (
        f"{case.template} caveats a run that graded everything: {healthy!r}"
    )
    assert not _is_could_not_check(healthy), (
        f"{case.template} withheld its finding from a fully graded run: {healthy!r}"
    )
    assert "?" not in healthy, (
        f"{case.template} printed a question mark on a fully graded run: {healthy!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=repr)
def test_the_partial_fixture_is_not_the_healthy_one_in_disguise(case):
    """Non-vacuity, and a guard in its own right.

    If the partial run and the fully graded run produce the SAME sentence, then
    every rule above is being checked against a chart that never noticed the
    difference, which is the original defect wearing a passing test.
    """
    partial = build_explanation(case.template, case.partial).finding
    healthy = build_explanation(case.template, case.healthy).finding

    assert partial != healthy, (
        f"{case.template}: the partial run reads identically to the fully graded one"
    )


# ───────────────────────────────────────────────────────────────────────────
# The static guard: no partial signal may be emitted and then ignored.
#
# This is the shape of the failure this file exists for. Each key below was
# ADDED to an adapter by an earlier wave, so the canvas learned the third state
# per row, and no finder was taught to read it, so the accessible layer went on
# describing a full run. A key that no finder reads is that gap, before anyone
# has to notice the sentence.
# ───────────────────────────────────────────────────────────────────────────

_PARTIAL_SIGNAL_KEYS = (
    "n_unknown",  # adapters.py / adapters_fairness.py: cards in the third state
    "n_ungraded",  # adapters_discovery.py, adapters_experimentation.py
    "n_unmeasured",  # adapters_fairness.py: radar, effect sizes, intervals
    "n_measured_groups",  # adapters.py, adapters_calibration.py
    "n_checked",  # adapters_monitoring.py: guardrails actually applied
    "n_unmonitored",  # adapters_monitoring.py: windows that checked nothing
    "n_perm_ungraded",  # adapters_robustness.py
    "n_power_ungraded",  # adapters_experimentation.py
    "forest_omitted",  # adapters_experimentation.py
    # Added 2026-08-28. Every key below was emitted by an adapter, drawn on the
    # canvas, and read by NO finder, which is precisely the gap this guard
    # exists for. The list was short because it was written from the finders
    # that had already been fixed rather than from what the adapters emit, so it
    # certified the six it knew about and stayed silent on these.
    "n_tests_ungraded",  # adapters.py: cicd checks that did not run
    "n_sens_ungraded",  # adapters_robustness.py: sensitivity checks with no score
    "n_untracked",  # adapters_monitoring.py: metrics with no history
    "n_unrated",  # adapters_feature_engineering.py: candidates with no risk level
    "n_ungraded_groups",  # adapters_fairness.py: groups that reported no rate
    "n_unmeasured_groups",  # adapters.py: group bars that were never plotted
    # Added 2026-08-28 (wave 15, core adapters). adapters._module_score stopped
    # grading a finding whose severity it could not read: risk_level=None used
    # to map through str(None) == "none" to 0.05 (NEGLIGIBLE, so absence
    # LOWERED a module's risk) and an unknown severity to a flat 0.5. Those
    # findings are now counted out of the average and stated in the bias-audit
    # headline band, and _fr_bias_audit carries the same count into the <desc>.
    "n_ungraded_findings",  # adapters.py: findings with no readable severity
    # Added 2026-08-28 (wave 15, the accessible descriptions). Each of these was
    # already drawn on its canvas and read by no finder, so the <desc> beside it
    # went on describing a fully measured run. They arrive here together with
    # the widened _SIGNAL_KEY_RE below, which is what stopped the last two of
    # them from being able to sit on a canvas unread.
    "n_steps_ungraded",  # adapters_experimentation.py: Baron-Kenny steps with no verdict
    "n_ratio_ungraded",  # adapters_feature_engineering.py: |r| = 0 before, so no ratio
    "n_effects_unmeasured",  # adapters_regression.py: group pairs never compared
    "n_residual_missing",  # adapters_regression.py: groups never screened for residual bias
    # Added 2026-09-27 (BGL5). alert_timeline_to_svg tested "was a guardrail
    # applied?" at WINDOW level over a PER-METRIC record, so a window that
    # compared two of its four metrics to a guardrail was tallied as fully clean
    # at severity info. The per-metric shortfall now travels on the data dict and
    # _fr_alert_timeline names it beside the count of partly checked windows.
    "n_unchecked_metrics",  # adapters_monitoring.py: metrics never compared to a guardrail
)

_RENDERING = Path(EX.__file__).parent


@pytest.mark.parametrize("key", _PARTIAL_SIGNAL_KEYS)
def test_every_partial_signal_an_adapter_emits_is_read_by_a_finder(key):
    emitters = sorted(
        path.name
        for path in _RENDERING.glob("adapters*.py")
        if f'"{key}"' in path.read_text(encoding="utf-8")
    )
    assert emitters, (
        f"no adapter emits {key!r} any more; drop it from _PARTIAL_SIGNAL_KEYS "
        f"rather than leaving a guard that checks nothing"
    )

    assert f'"{key}"' in Path(EX.__file__).read_text(encoding="utf-8"), (
        f"{', '.join(emitters)} put {key!r} on the canvas for a partly measured run "
        f"and no finder in explain.py reads it, so the accessible <desc> still "
        f"describes a fully measured one"
    )


# The shape of an ungraded-count key, so the guard above stops depending on a
# hand-written list. `_PARTIAL_SIGNAL_KEYS` was short for six months not because
# the adapters emitted six signals but because it was written from the finders
# that had already been fixed: six more keys were on the canvas and in no
# finder, and the guard passed the whole time. A list that only grows when
# somebody remembers to grow it is not a guard.
#
# Deliberately narrow. "un" must be followed by one of the words this library
# actually uses for an absence, which is what keeps `n_underpowered` out: that
# is a MEASURED finding about a subgroup that was sized and found short, not a
# row nobody graded, and folding the two together is the confusion this whole
# rule exists to stop.
#
# The SUFFIX half was narrower still: only `_ungraded`, so `n_ratio_ungraded`
# matched and `n_effects_unmeasured` beside it did not, and the pattern's own
# hard-coded escape hatch (`forest_omitted`) is the tell that the shape had
# already been met once and answered by name rather than by rule. The suffixes
# below are the five words the adapters use for a row nobody graded; each one is
# a word about ABSENCE, never about a measured shortfall, which is what keeps
# `n_underpowered` and `n_high` out on the same argument as before.
_SIGNAL_KEY_RE = re.compile(
    r'"(n_(?:un(?:graded|measured|known|monitored|rated|tracked|checked)|'
    r"[a-z]+_(?:ungraded|unmeasured|unrated|missing|omitted))[a-z_]*"
    r'|forest_omitted)"\s*:'
)


def test_the_signal_key_list_still_covers_every_key_the_adapters_emit():
    """The guard on the guard. A new absence key must not be able to arrive
    quietly: it is either read by a finder, or this fails and says which
    adapter put it on the canvas."""
    emitted = {}
    for path in sorted(_RENDERING.glob("adapters*.py")):
        for match in _SIGNAL_KEY_RE.finditer(path.read_text(encoding="utf-8")):
            emitted.setdefault(match.group(1), set()).add(path.name)

    assert emitted, "the key pattern matched nothing at all, so this guard checks nothing"

    missing = {k: sorted(v) for k, v in emitted.items() if k not in _PARTIAL_SIGNAL_KEYS}
    assert not missing, (
        f"these adapters emit an ungraded count that _PARTIAL_SIGNAL_KEYS does not list, "
        f"so nothing checks that a finder reads it: {missing}"
    )


def test_every_finder_with_a_partial_clause_has_a_case_here():
    """A finder that gains the clause without a fixture is not actually guarded."""
    source = Path(EX.__file__).read_text(encoding="utf-8")
    blocks = re.split(r"\ndef (_fr_\w+)\(", source)
    with_clause = {
        blocks[i]
        for i in range(1, len(blocks), 2)
        if "_ungraded_tail(" in blocks[i + 1] or "_floor_sev(" in blocks[i + 1]
    }
    covered = {_FINDERS[t].__name__ for t in _COVERED_TEMPLATES}

    assert with_clause <= covered, (
        f"these finders learned the partial-run clause but no case in _CASES "
        f"exercises them: {sorted(with_clause - covered)}"
    )


# ───────────────────────────────────────────────────────────────────────────
# End to end, through the real adapters, on the artifact that ships.
# ───────────────────────────────────────────────────────────────────────────


def test_a_power_analysis_with_an_ungraded_row_says_so_in_the_desc():
    from vfairness.rendering.adapters_experimentation import power_analysis_to_svg

    svg = power_analysis_to_svg(
        [
            {"intersection": ("a",), "power": 0.42, "is_powered": False},
            {"intersection": ("b",)},
        ]
    )
    desc = _desc(svg)

    assert "1 of 2 intersections are underpowered" not in desc, (
        "the ungraded row was counted as adequately powered"
    )
    assert "1 of 1 intersections are underpowered" in desc
    assert _NOT_COVERED in desc, desc
    assert "severity: INFO" not in desc


def test_a_discovery_scan_with_an_ungraded_violation_says_so_in_the_desc():
    from vfairness.rendering.adapters_discovery import auto_discovery_to_svg

    svg = auto_discovery_to_svg(
        candidates=[{"column": "zip", "confidence": 0.9, "attribute_type": "race"}],
        violations=[
            {
                "attribute": "race",
                "metric": "dp",
                "value": 0.31,
                "threshold": 0.1,
                "severity": "high",
            },
            {"attribute": "sex", "metric": "dp", "value": 0.05, "threshold": 0.1},
        ],
        group_advantages=[],
    )
    desc = _desc(svg)

    assert "NOT GRADED" in svg, "this input no longer has an ungraded row; test is vacuous"
    assert _NOT_COVERED in desc, desc


def test_a_correlation_heatmap_with_an_uncomputed_pair_says_so_in_the_desc():
    from vfairness.rendering.adapters_feature_engineering import correlation_heatmap_to_svg

    svg = correlation_heatmap_to_svg(
        {
            "income": {"race": 0.42, "sex": None},
            "zip": {"race": 0.05, "sex": 0.01},
        }
    )
    desc = _desc(svg)

    assert "not computed" in svg, "this input no longer has an uncomputed pair; test is vacuous"
    assert _NOT_COVERED in desc, desc
    assert "1 feature-attribute pair" in desc, desc


# ───────────────────────────────────────────────────────────────────────────
# The template that outranked a correctly-honest description.
# ───────────────────────────────────────────────────────────────────────────

_GRADED_DECISIONS = {
    "DEPLOY_TREATMENT": "DEPLOY TREATMENT",
    "KEEP_CONTROL": "KEEP CONTROL",
    "EXTEND_EXPERIMENT": "EXTEND EXPERIMENT",
    "INVESTIGATE_FURTHER": "INVESTIGATE FURTHER",
}


def _recommendation(decision):
    from vfairness.rendering.adapters_experimentation import experiment_recommendation_to_svg

    return experiment_recommendation_to_svg({"decision": decision, "confidence": 0.9})


def test_an_unrecognised_decision_is_not_badged_investigate_further():
    """The canvas used to fall through to the loudest badge it owns.

    ``experiment_recommendation_to_svg({"decision": "ship it now"})`` drew a red
    INVESTIGATE FURTHER, a recommendation this library never made, while the
    <desc> beside it correctly said COULD NOT CHECK. A fabricated breach sends
    someone after a violation that does not exist.
    """
    svg = _recommendation("ship it now")

    assert "INVESTIGATE FURTHER" not in svg
    assert "DECISION NOT RECOGNISED" in svg
    assert "SHIP_IT_NOW" in svg, "the string that was supplied is not named anywhere"
    assert _is_could_not_check(_desc(svg).split(": ", 1)[1])


def test_an_unrecognised_decision_is_drawn_in_the_neutral_tone():
    """Not red, and not green either: unrecognised is neither of the two."""
    svg = _recommendation("ship it now")
    badge = re.search(r"DECISION NOT RECOGNISED.*?</text>", svg, re.S)
    assert badge

    head = svg[: svg.index("REASONING")]
    for finding_red in ("#dc2626", "#b6573a", "#fef2f2"):
        assert finding_red not in head, f"the unrecognised decision is painted {finding_red}"
    for pass_green in ("#059669", "#ecfdf5"):
        assert pass_green not in head, f"the unrecognised decision is painted {pass_green}"


def test_the_action_line_stops_telling_the_reader_to_follow_a_recommendation():
    """The third surface. Two agreeing states and one contradicting action is
    still a canvas that reads more confidently than the evidence."""
    svg = _recommendation("ship it now")

    assert "Follow the recommendation" not in svg
    assert "certifies nothing" in svg


@pytest.mark.parametrize("decision,badge", sorted(_GRADED_DECISIONS.items()))
def test_each_decision_this_library_does_grade_keeps_its_own_badge(decision, badge):
    """Over-correction control for the template: the four real decisions are
    unchanged, INVESTIGATE_FURTHER included, which now has its own branch
    instead of arriving via the fall-through."""
    svg = _recommendation(decision)

    assert badge in svg
    assert "DECISION NOT RECOGNISED" not in svg
    assert badge.title().replace("_", " ") in _desc(svg) or badge.title() in _desc(svg)


def test_an_absent_decision_still_reads_no_recommendation_made():
    """The other third state, pinned so the new branch did not displace it."""
    from vfairness.rendering.adapters_experimentation import experiment_recommendation_to_svg

    svg = experiment_recommendation_to_svg({})

    assert "NO RECOMMENDATION MADE" in svg
    assert "DECISION NOT RECOGNISED" not in svg
    assert "INVESTIGATE FURTHER" not in svg
