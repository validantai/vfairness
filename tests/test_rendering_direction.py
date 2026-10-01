"""Release-pass pins for the rendering layer: direction, intervals, group sizes.

Wave 1 removed a family of false-FAIR defects from the adapters. The adversarial
acceptance pass then found that three of them were still live on the canvas a
human actually looks at, and every one of the three is the same class: a surface
that answers "fair" or prints a number when the honest answer is "we did not
measure that".

1. The disparity-heatmap summary badge banded ``abs(value)`` for EVERY metric,
   including the ratio family. ``demographic_parity_ratio = 0.0`` means the
   protected group was NEVER selected, the worst outcome the metric can express,
   and it rendered a green tick reading "Low disparity across groups". The band
   now runs through the shared
   ``evaluation.vfairness_metrics._metric_direction`` helper, and a metric whose
   better-direction that helper cannot resolve can no longer produce a green
   all-clear.

2. The experimentation adapter defaulted a missing confidence bound to ``0``, so
   an effect with no interval rendered "[0.0000, 0.0000]" under a "95%
   CONFIDENCE INTERVALS" heading and fed 0,0 into the forest-plot scale. That is
   an invented interval, the same fabrication class removed from the fairness
   forest plot in wave 1.

3. Both group-bar adapters read the per-group size as ``count``/``n``, but the
   engine emits ``size``. Every bar therefore printed "n=0" beside a group with
   hundreds of rows, which reads as an empty stratum.

Everything is asserted against the rendered SVG string, because the SVG is the
artifact that leaves the building.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)

from vfairness.evaluation.vfairness_metrics.report import (  # noqa: E402
    classification_fairness_report,
)
from vfairness.rendering import (  # noqa: E402
    adapters,
    adapters_experimentation,
    adapters_fairness,
)

_GREEN_BADGE = "Low disparity across groups"


def _ratio_report(value, extra_metrics=None):
    """A two-group report whose only disparity metric is a parity RATIO.

    Both groups are well populated, so the report is genuinely assessable: the
    only thing under test is how the value is banded.
    """
    metrics = {"demographic_parity_ratio": value}
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "protected_attribute": "gender",
        "metrics": metrics,
        "thresholds_used": {"demographic_parity_ratio": 0.80},
        "assessment": {"assessable": True, "fairness_score": 0.0, "summary": ""},
        "group_stats": {
            "female": {"positive_rate": 0.0, "tpr": 0.0, "size": 300},
            "male": {"positive_rate": 0.54, "tpr": 1.0, "size": 300},
        },
    }


# 1.  Heatmap summary badge: direction, not magnitude


def test_maximal_ratio_violation_is_never_a_green_heatmap_badge():
    """NEGATIVE case. 0.00 on a parity ratio is the WORST possible outcome.

    The protected group was never selected once. Banding ``abs(0.0)`` put it in
    the "< 0.1" bucket and painted the green tick.
    """
    svg = adapters_fairness.disparity_heatmap_to_svg(_ratio_report(0.0))
    assert svg, "rendered nothing"
    assert _GREEN_BADGE not in svg, "a maximal ratio violation rendered the green all-clear badge"
    assert "High disparity detected" in svg, "the maximal violation is not called out at all"


def test_ratio_at_parity_still_earns_its_green_badge():
    """Positive control: the fix is a direction test, not a blanket refusal.

    ``demographic_parity_ratio = 1.0`` IS parity and must keep the green badge,
    otherwise the ratio family has simply been inverted a second time.
    """
    svg = adapters_fairness.disparity_heatmap_to_svg(_ratio_report(1.0))
    assert _GREEN_BADGE in svg, "perfect parity lost its green badge"


def test_ratio_just_under_the_four_fifths_rule_is_not_green():
    """0.75 fails the four-fifths rule; ``abs(0.75)`` banded it as HIGH before,
    which was the right colour for the wrong reason. It must stay non-green."""
    svg = adapters_fairness.disparity_heatmap_to_svg(_ratio_report(0.75))
    assert _GREEN_BADGE not in svg


def test_unknown_direction_metric_cannot_produce_a_green_badge():
    """Fail closed. A metric whose better-direction the shared helper cannot
    resolve is unbanded evidence, so it must not sit silently inside a green
    all-clear about the whole grid."""
    report = _ratio_report(1.0, extra_metrics={"bespoke_house_metric": 0.93})
    svg = adapters_fairness.disparity_heatmap_to_svg(report)
    assert _GREEN_BADGE not in svg, "an unknown-direction metric was folded into a green all-clear"
    assert "COULD NOT CHECK" in svg or "NOT ASSESSABLE" in svg


def test_only_unknown_direction_metrics_is_could_not_check():
    report = {
        "metrics": {"bespoke_house_metric": 0.02},
        "thresholds_used": {},
        "assessment": {"assessable": True, "fairness_score": 0.0, "summary": ""},
        "group_stats": {
            "a": {"positive_rate": 0.5, "size": 200},
            "b": {"positive_rate": 0.5, "size": 200},
        },
    }
    svg = adapters_fairness.disparity_heatmap_to_svg(report)
    assert _GREEN_BADGE not in svg
    assert "NOT ASSESSABLE" in svg


def test_heatmap_badge_agrees_with_the_engine_on_a_real_run():
    """End to end on real engine output where the ratio is 0.0: the badge must
    not contradict the engine's own FAIL verdict for that metric."""
    rng = np.random.default_rng(0)
    groups = np.array(["male"] * 300 + ["female"] * 300)
    y_true = rng.integers(0, 2, size=600)
    y_pred = y_true.copy()
    y_pred[groups == "female"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups)
    assert report["metrics"]["demographic_parity_ratio"] == 0.0
    svg = adapters_fairness.disparity_heatmap_to_svg(report)
    assert _GREEN_BADGE not in svg


# 2.  A confidence interval is computed or it is absent


def _effect(**overrides):
    eff = {
        "intersection": ("female", "senior"),
        "effect_size_d": 0.42,
        "p_value": 0.03,
        "n_control": 120,
        "n_treatment": 130,
        "significant": True,
        "powered": True,
    }
    eff.update(overrides)
    return eff


def test_missing_confidence_bounds_are_not_rendered_as_a_zero_interval():
    """NEGATIVE case. An effect dict with no interval must not gain one."""
    svg = adapters_experimentation.experiment_results_to_svg(
        {"intersection_effects": [_effect()], "overall_effect": 0.42, "overall_p_value": 0.03}
    )
    assert svg, "rendered nothing"
    assert "[0.0000, 0.0000]" not in svg, "an absent interval was fabricated as [0.0000, 0.0000]"
    assert "CI not computed" in svg, "the missing interval is not labelled on the canvas"


def test_half_an_interval_is_still_no_interval():
    """One bound present and one absent is not an interval either."""
    svg = adapters_experimentation.experiment_results_to_svg(
        {"intersection_effects": [_effect(ci_lower=0.11)], "overall_effect": 0.42}
    )
    assert "[0.1100, 0.0000]" not in svg
    assert "CI not computed" in svg


def test_a_real_interval_is_still_drawn():
    """Positive control: computed bounds must survive untouched."""
    svg = adapters_experimentation.experiment_results_to_svg(
        {
            "intersection_effects": [_effect(ci_lower=0.11, ci_upper=0.73)],
            "overall_effect": 0.42,
            "overall_ci": (0.10, 0.74),
        }
    )
    assert "[0.1100, 0.7300]" in svg
    assert "CI not computed" not in svg
    assert "[0.1000, 0.7400]" in svg


def test_missing_overall_interval_is_not_rendered_as_zero_to_zero():
    svg = adapters_experimentation.experiment_results_to_svg(
        {"intersection_effects": [_effect(ci_lower=0.11, ci_upper=0.73)], "overall_effect": 0.42}
    )
    assert "95% CI [0.0000, 0.0000]" not in svg
    assert "CI not computed" in svg


# 3.  The group size the engine actually emits


def _sized_report():
    return {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "assessment": {"assessable": True, "fairness_score": 1.0, "summary": ""},
        "group_stats": {
            "female": {"positive_rate": 0.48, "size": 317},
            "male": {"positive_rate": 0.52, "size": 283},
        },
    }


def test_group_comparison_prints_the_real_group_size():
    svg = adapters_fairness.group_comparison_to_svg(_sized_report())
    assert "n=317" in svg and "n=283" in svg
    assert "n=0" not in svg, "a populated group was labelled n=0"


def test_fairness_report_dashboard_prints_the_real_group_size():
    svg = adapters.fairness_report_to_svg(_sized_report())
    assert "n=317" in svg and "n=283" in svg
    assert "n=0" not in svg


def test_group_size_on_a_real_engine_report():
    rng = np.random.default_rng(4)
    groups = np.array(["a"] * 250 + ["b"] * 150)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups)
    assert report["group_stats"]["a"]["size"] == 250
    svg = adapters_fairness.group_comparison_to_svg(report)
    assert "n=250" in svg and "n=150" in svg


def test_older_report_shapes_still_resolve_their_size():
    """The legacy ``count``/``n`` keys keep working; only the real key was added."""
    legacy = _sized_report()
    legacy["group_stats"] = {
        "female": {"positive_rate": 0.48, "count": 317},
        "male": {"positive_rate": 0.52, "n": 283},
    }
    svg = adapters_fairness.group_comparison_to_svg(legacy)
    assert "n=317" in svg and "n=283" in svg


# 4.  The detailed report template: operator, third badge, banner


def _detailed(metric_name, value, threshold, score=0.9):
    return {
        "title": "Fairness Analysis Report",
        "metrics": {metric_name: {"value": value, "threshold": threshold}},
        "group_statistics": {
            "female": {"size": 300, "positive_rate": 0.0, "tpr": 0.0, "fpr": 0.0},
            "male": {"size": 300, "positive_rate": 0.54, "tpr": 1.0, "fpr": 0.0},
        },
        "assessment": {"fairness_score": score},
    }


def _detailed_svg(*args, **kwargs):
    from vfairness.rendering import adapters_post_processing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return adapters_post_processing.fairness_detailed_report_to_svg(_detailed(*args, **kwargs))


def test_detailed_report_states_the_threshold_in_the_metrics_own_direction():
    """A row reading "Disparate Impact Ratio | 1.0000 | <= 0.80 | PASS" misstates
    the four-fifths rule: the rule is a FLOOR. The operator must agree with the
    verdict beside it."""
    svg = _detailed_svg("disparate_impact_ratio", 1.0, 0.80)
    assert "PASS" in svg
    assert "≤ 0.80" not in svg, "the four-fifths floor was rendered as a ceiling"
    assert "≥ 0.80" in svg


def test_detailed_report_keeps_the_ceiling_for_a_difference_metric():
    """Positive control: a violation magnitude really is bounded above."""
    svg = _detailed_svg("demographic_parity_difference", 0.04, 0.10)
    assert "≤ 0.10" in svg
    assert "≥ 0.10" not in svg


def test_detailed_report_can_express_a_could_not_check_row():
    """A metric with no resolvable direction must get its own badge, not the red
    FAIL badge of a measured breach."""
    svg = _detailed_svg("bespoke_house_metric", 0.42, 0.10)
    assert "NO DATA" in svg, "the template still has only PASS and FAIL badges"
    assert ">FAIL<" not in svg, "could-not-check was painted as a measured failure"


def test_detailed_report_banner_is_downgraded_when_a_metric_is_unchecked():
    """It rendered "FAIRNESS SCORE 90/100 FAIR" beside an unknown-direction
    metric. A score cannot certify metrics that were never checked."""
    svg = _detailed_svg("bespoke_house_metric", 0.42, 0.10, score=0.90)
    assert ">FAIR<" not in svg, "the FAIR banner survived beside an unchecked metric"
    assert "COULD NOT CHECK" in svg


def test_detailed_report_banner_survives_when_everything_was_checked():
    """Positive control: a fully checked, high-scoring report keeps FAIR."""
    svg = _detailed_svg("demographic_parity_difference", 0.01, 0.10, score=0.90)
    assert ">FAIR<" in svg


def test_detailed_report_template_handles_a_missing_score_without_crashing():
    """``fairness_score is None`` reached ``fairness_score >= 70`` in the
    template and blew up the render. The template must render the third state
    itself, not depend on the caller diverting around it."""
    svg = engine.render_svg(
        "fairness_detailed_report",
        {
            "title": "Fairness Analysis Report",
            "task_type": "classification",
            "attribute_name": "gender",
            "n_samples": 600,
            "n_groups": 2,
            "fairness_score": None,
            "overall_assessment": "N/A",
            "metrics": [],
            "groups": [],
            "pairwise_comparisons": [],
            "key_findings": [],
            "recommendations": [],
        },
    )
    assert "NOT ASSESSABLE" in svg
    assert "0/100" not in svg


# ── WAVE 3 ──────────────────────────────────────────────────────────────────
# Same bug class, third sighting: a surface that paints a maximal violation, or
# an absent measurement, green. Everything below is asserted against the FINAL
# rendered SVG, after the Blanco skin has recoloured it, because that is the
# file that leaves the building. The skin maps the emerald pass palette onto
# #41ba1b / #1c6d00 / #e4f3da, so asserting the unskinned hex would pass while
# the exported file was still green.

import pathlib  # noqa: E402
import re  # noqa: E402

# Blanco pass palette (post-skin). A cell, badge or dot wearing any of these is
# telling the reader "measured, and fine".
_BLANCO_PASS = ("#41ba1b", "#1c6d00", "#e4f3da", "#d6ecca", "#ecf8e8")
# Blanco neutral, the third state's own palette.
_BLANCO_NEUTRAL_BG = "#f4f3f2"

_CELL_RE = re.compile(
    r'<rect x="(\d+)" y="\d+" width="\d+" height="\d+" fill="(#[0-9a-f]{6})"/>\s*'
    r'<text[^>]*fill="(#[0-9a-f]{6})">([^<]*)</text>'
)


def _heatmap_cells(svg):
    """(bg, fg, displayed_value) for every data cell in a rendered heatmap."""
    return [(bg, fg, text) for _x, bg, fg, text in _CELL_RE.findall(svg)]


# 5.  A degenerate threshold grades nothing (defect 1)


def _degenerate_report(value, threshold, key="demographic_parity_ratio"):
    return {
        "protected_attribute": "gender",
        "metrics": {key: value},
        "thresholds_used": {key: threshold},
        "assessment": {"assessable": True, "fairness_score": 0.0, "summary": ""},
        "group_stats": {
            "female": {"positive_rate": 0.0, "tpr": 0.0, "size": 300},
            "male": {"positive_rate": 0.54, "tpr": 1.0, "size": 300},
        },
    }


def test_zero_minimum_on_a_ratio_metric_is_not_a_measured_pass():
    """NEGATIVE case. A required MINIMUM of 0.0 cannot be breached by anything.

    ``check_threshold`` answered PASS for it, so a demographic_parity_ratio of
    0.00 (the protected group is never selected once, the worst outcome the
    metric can express) earned a green FAIR badge. An unenforceable bound is not
    a pass, it is the third state.
    """
    state, reason = adapters_fairness._metric_state("demographic_parity_ratio", 0.00, 0.0)
    assert state == adapters_fairness.COULD_NOT_CHECK, (
        f"a 0.0 floor on a ratio graded {state!r} for the worst value the metric has"
    )
    assert reason, "the third state must say why on the canvas"


def test_zero_minimum_ratio_renders_no_fair_badge_end_to_end():
    report = _degenerate_report(0.00, 0.0)
    svg = adapters_fairness.metrics_bar_chart_to_svg(report)
    assert "✓ FAIR" not in svg, "a never-selected group kept its green FAIR badge"
    assert "NO DATA" in svg
    for green in _BLANCO_PASS:
        assert f'fill="{green}"' not in svg, f"the pass green {green} survived on the canvas"

    dash = adapters.fairness_report_to_svg(report)
    assert ">FAIR<" not in dash
    assert [c["state"] for c in adapters._metric_cards(report)] == ["could_not_check"]


def test_a_real_four_fifths_floor_still_grades_both_ways():
    """Positive control: the fix is a degenerate-bound test, not a refusal to
    grade ratios. A genuine 0.80 floor must still pass and still fail."""
    assert adapters_fairness._metric_state("demographic_parity_ratio", 0.95, 0.80)[0] == "pass"
    assert adapters_fairness._metric_state("demographic_parity_ratio", 0.50, 0.80)[0] == "fail"


def test_zero_tolerance_on_a_difference_metric_is_untouched():
    """Positive control for the MIRROR case, which is NOT degenerate: 0.0 on a
    lower-is-better metric is a real zero-tolerance policy that a value can
    breach, and wave 1 exists to keep it working."""
    assert adapters_fairness._metric_state("demographic_parity_difference", 0.00, 0.0)[0] == "pass"
    assert adapters_fairness._metric_state("demographic_parity_difference", 0.25, 0.0)[0] == "fail"


# 6.  Heatmap CELLS: distance from the reference, never raw magnitude (defect 2)


def test_the_never_selected_group_is_not_the_green_cell():
    """NEGATIVE case. On a Selection Rate column the group receiving 0 percent of
    the positive outcomes was painted green and the group receiving 54 percent
    red, because the gradient banded the raw rate as if it were a disparity."""
    svg = adapters_fairness.disparity_heatmap_to_svg(_degenerate_report(0.00, 0.80))
    cells = _heatmap_cells(svg)
    assert cells, "no data cells were found in the rendered heatmap"
    zero = [c for c in cells if c[2] == "0.00"]
    assert zero, f"the 0.00 rate is not on the grid: {cells}"
    for bg, fg, _v in zero:
        assert bg not in _BLANCO_PASS, "the never-selected group is still painted as a pass"
        assert fg not in _BLANCO_PASS, "the never-selected group still has pass-green text"


def test_the_reference_group_keeps_its_green_cell():
    """Positive control: the best-served group is the reference, so its distance
    from the reference really is zero. The gradient is re-anchored, not removed."""
    svg = adapters_fairness.disparity_heatmap_to_svg(_degenerate_report(0.00, 0.80))
    best = [c for c in _heatmap_cells(svg) if c[2] == "0.54"]
    assert best, "the reference rate is not on the grid"
    assert any(bg in _BLANCO_PASS for bg, _fg, _v in best), (
        "the reference group is not shown as the reference"
    )


def test_two_equal_groups_are_both_green_cells():
    """Positive control: no shortfall from the reference anywhere is genuine
    parity on that column, and must still read as such."""
    report = _degenerate_report(1.00, 0.80)
    report["group_stats"] = {
        "female": {"positive_rate": 0.50, "tpr": 0.5, "size": 300},
        "male": {"positive_rate": 0.50, "tpr": 0.5, "size": 300},
    }
    cells = _heatmap_cells(adapters_fairness.disparity_heatmap_to_svg(report))
    assert cells
    assert all(bg in _BLANCO_PASS for bg, _fg, _v in cells), f"parity lost its green: {cells}"


def test_an_unmeasurable_cell_is_could_not_check_not_a_colour():
    """NEGATIVE case. A NaN rate reached the gradient, printed the literal
    'nan' on the grid and dragged the whole column's range to NaN, which
    silently repainted every other cell in it."""
    report = _degenerate_report(0.00, 0.80)
    report["group_stats"] = {
        "female": {"positive_rate": float("nan"), "tpr": float("nan"), "size": 300},
        "male": {"positive_rate": 0.54, "tpr": 1.0, "size": 300},
    }
    svg = adapters_fairness.disparity_heatmap_to_svg(report)
    assert ">nan<" not in svg, "an unmeasured rate was printed as a number"
    cells = _heatmap_cells(svg)
    unknown = [c for c in cells if c[2] in ("N/A", "n/a")]
    assert unknown, f"the unmeasurable cell is not labelled: {cells}"
    for bg, fg, _v in unknown:
        assert bg not in _BLANCO_PASS and fg not in _BLANCO_PASS, (
            "an unmeasurable cell was painted as a measured pass"
        )
        assert bg == _BLANCO_NEUTRAL_BG, f"could-not-check borrowed another palette: {bg}"


# 7.  An absent statistical test is not a homogeneous one (defect 3)


def _experiment(**overrides):
    payload = {
        "intersection_effects": [_effect(ci_lower=0.11, ci_upper=0.73)],
        "overall_effect": 0.42,
        "overall_ci": (0.10, 0.74),
    }
    payload.update(overrides)
    return payload


def test_absent_heterogeneity_test_is_not_a_green_homogeneous_card():
    """NEGATIVE case. With neither field present the adapter defaulted to
    ``False`` / ``1.0``, so an experiment that never ran a heterogeneity test
    rendered a green HOMOGENEOUS card reading "p-value: 1.0000"."""
    svg = adapters_experimentation.experiment_results_to_svg(_experiment())
    assert svg, "rendered nothing"
    assert "HOMOGENEOUS" not in svg, "a test that never ran was reported as homogeneous"
    assert "1.0000" not in svg, "a p-value of 1.0000 was fabricated from no data"
    assert "COULD NOT CHECK" in svg


def test_absent_overall_p_value_suppresses_the_significance_badge():
    """NEGATIVE case. ``overall_p_value`` defaulted to 1.0, which is not a
    missing value, it is the most confident 'no effect' the scale can express."""
    svg = adapters_experimentation.experiment_results_to_svg(_experiment())
    assert "NOT SIGNIFICANT" not in svg, "a significance verdict was issued without a p-value"
    assert "COULD NOT CHECK" in svg


def test_experiment_prose_claims_neither_test_it_did_not_run():
    """The badge and the sentence under it are built from one state. The
    accessible <desc> read "is not significant" and omitted any heterogeneity
    caveat, both drawn from the same two defaults."""
    svg = adapters_experimentation.experiment_results_to_svg(_experiment())
    desc = re.search(r"<desc>([^<]*)</desc>", svg)
    assert desc, "no accessible <desc>"
    text = desc.group(1)
    assert "not significant" not in text, f"<desc> ruled on significance with no p-value: {text!r}"
    assert text.count("COULD NOT CHECK") == 2, f"<desc> does not carry both third states: {text!r}"

    # Positive control: with both tests reported the prose states both verdicts.
    both = adapters_experimentation.experiment_results_to_svg(
        _experiment(overall_p_value=0.42, heterogeneity_detected=False, heterogeneity_p_value=0.5)
    )
    text = re.search(r"<desc>([^<]*)</desc>", both).group(1)
    assert "The effect is not significant." in text
    assert "COULD NOT CHECK" not in text


def test_a_reported_heterogeneity_test_still_reports_its_verdict():
    """Positive control, both ways: a real test keeps its card and its p-value."""
    homo = adapters_experimentation.experiment_results_to_svg(
        _experiment(heterogeneity_detected=False, heterogeneity_p_value=0.42)
    )
    assert "HOMOGENEOUS" in homo
    assert "0.4200" in homo

    hetero = adapters_experimentation.experiment_results_to_svg(
        _experiment(heterogeneity_detected=True, heterogeneity_p_value=0.001)
    )
    assert "DETECTED" in hetero


def test_a_reported_overall_p_value_still_earns_its_badge():
    """Positive control: a measured p-value keeps the SIGNIFICANT badge."""
    svg = adapters_experimentation.experiment_results_to_svg(_experiment(overall_p_value=0.03))
    assert ">SIGNIFICANT<" in svg
    assert "0.0300" in svg

    ns = adapters_experimentation.experiment_results_to_svg(_experiment(overall_p_value=0.42))
    assert "NOT SIGNIFICANT" in ns
    assert "0.4200" in ns


# 8.  The accessible text is never the more confident one (defect 4)


def _plain_text(svg):
    """Everything a sighted reader sees, tags stripped, whitespace collapsed."""
    return " ".join(re.sub(r"<[^>]+>", " ", svg).split())


def test_detailed_report_desc_is_not_more_confident_than_its_banner():
    """NEGATIVE case. Wave 2 downgraded the banner to COULD NOT CHECK, but the
    ``<desc>`` a screen reader gets still read "Overall fairness is FAIR with a
    score of 90/100", and so did the visible Explanation paragraph."""
    svg = _detailed_svg("bespoke_house_metric", 0.42, 0.10, score=0.90)
    assert "COULD NOT CHECK" in svg, "the banner downgrade regressed; this test is now vacuous"

    desc = re.search(r"<desc>([^<]*)</desc>", svg)
    assert desc, "no accessible <desc>"
    text = desc.group(1)
    assert not re.search(r"Overall fairness is (FAIR|MARGINAL|UNFAIR)", text), (
        f"<desc> is more confident than the banner beside it: {text!r}"
    )
    assert "COULD NOT CHECK" in text, f"<desc> does not carry the third state: {text!r}"

    body = _plain_text(svg)
    assert "Overall fairness is FAIR" not in body, (
        "the visible Explanation paragraph still certifies the report as fair"
    )


def test_detailed_report_desc_still_reports_a_fully_checked_verdict():
    """Positive control: when every metric WAS checked the verdict survives, in
    the prose and in the accessible layer alike."""
    svg = _detailed_svg("demographic_parity_difference", 0.01, 0.10, score=0.90)
    desc = re.search(r"<desc>([^<]*)</desc>", svg).group(1)
    assert "Overall fairness is FAIR with a score of 90/100" in desc
    assert "COULD NOT CHECK" not in desc


# 9.  Widen the reintroduction guard: no adapter decides direction on its own


_RENDERING_SRC = pathlib.Path(adapters_fairness.__file__).parent

# The bug class is a call site answering "which way is better?" from the metric
# NAME on its own. `_metric_direction` is the single answer; every form below is
# a local re-derivation of it, and each one has already shipped a false pass:
#   * a substring test matched cali-BRATIO-n (CLAUDE.md, commit 47f1e09f8);
#   * an exact `_ratio` suffix test misses `disparate_impact`, which IS a ratio,
#     so a maximal violation (0.00) was plotted on the radar's FAIR rim and
#     badged "Negligible" on the effect-size chart.
_LOCAL_DIRECTION_TESTS = (
    (re.compile(r"""\.endswith\(\s*["']_ratio["']\s*\)"""), "an exact _ratio suffix test"),
    (re.compile(r"""["']_?ratio["']\s+in\s+\w"""), "a substring test for 'ratio'"),
    (re.compile(r"""["']disparate_impact["']\s+in\s+\w"""), "a substring test for a family name"),
)


def _docstring_lines(source):
    """Line numbers occupied by docstrings, so prose about the bug is not a hit.

    Only Expr-statement string constants count, i.e. real docstrings. A string
    used in CODE (``key.endswith("_ratio")``) is never one of these, so the
    guard keeps its teeth.
    """
    import ast

    covered = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
        return covered
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                covered.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return covered


def test_no_rendering_adapter_decides_metric_direction_on_its_own():
    """The guard the last two waves lacked.

    ``tests/test_metric_direction.py`` pins the shared helper, but nothing
    stopped a RENDERER from answering the same question locally, which is how
    the identical defect reached a third file. Direction is imported, never
    re-derived.
    """
    offences = []
    for path in sorted(_RENDERING_SRC.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        prose = _docstring_lines(source)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if lineno in prose:
                continue
            code = line.split("#", 1)[0]
            for pattern, what in _LOCAL_DIRECTION_TESTS:
                if pattern.search(code):
                    offences.append(f"{path.name}:{lineno}: {what} -> {line.strip()}")
    assert not offences, (
        "a renderer is deciding metric direction from the name again; import "
        "metric_direction/is_ratio_metric instead:\n" + "\n".join(offences)
    )


def test_radar_puts_a_maximal_ratio_violation_at_the_centre_not_the_rim():
    """NEGATIVE case, found by the guard above. ``disparate_impact`` IS a parity
    ratio but does not end in '_ratio', so the radar took the difference branch:
    a value of 0.00 (never selected) gave a gap of abs(0.0) = 0.0 and was plotted
    at radius fraction 1.0, the outer rim, which on this chart MEANS fully fair.
    """
    frac = adapters_fairness._fairness_radius_fraction("disparate_impact", 0.00, 0.80)
    assert frac is not None
    assert frac < adapters_fairness._THRESHOLD_RING_FRACTION, (
        f"a maximal violation was plotted at radius fraction {frac} (1.0 is the fair rim)"
    )
    assert adapters_fairness._fairness_radius_fraction("disparate_impact", 1.00, 0.80) == 1.0


def test_radar_refuses_to_plot_a_metric_with_no_known_direction():
    """Fail closed. A metric whose better-direction cannot be resolved has no
    place on an axis where the radius MEANS fairness."""
    assert adapters_fairness._fairness_radius_fraction("bespoke_house_metric", 0.42, 0.10) is None
    report = _degenerate_report(0.42, 0.10, key="bespoke_house_metric")
    svg = adapters_fairness.radar_chart_to_svg(report)
    assert ">Fair<" not in svg
    assert "NOT ASSESSABLE" in svg or "COULD NOT CHECK" in svg


def test_effect_sizes_does_not_band_a_parity_ratio_as_a_cohens_d():
    """NEGATIVE case. Cohen's d bands a DIFFERENCE. ``disparate_impact`` = 0.00
    is a maximal violation, and |0.00| < 0.2 badged it a green "Negligible"
    effect."""
    svg = adapters_fairness.effect_sizes_to_svg(_degenerate_report(0.00, 0.80, "disparate_impact"))
    assert "Negligible" not in svg, "a maximal ratio violation was badged as a negligible effect"


def test_effect_sizes_still_bands_a_real_difference_metric():
    """Positive control: the chart's actual subject matter is untouched."""
    svg = adapters_fairness.effect_sizes_to_svg(
        _degenerate_report(0.02, 0.10, "demographic_parity_difference")
    )
    assert "Negligible" in svg


# ── WAVE 4 ──────────────────────────────────────────────────────────────────
# Three waves fixed direction decided from a metric NAME. Acceptance then found
# the same shape in chart GEOMETRY and in the accessible layer: a subtitle that
# truncated on healthy data, a <desc> more confident than the badge beside it,
# and one more private direction rule outside the rendering package.


# 10.  The heatmap cell signature says what the body actually accepts


def test_heatmap_cell_style_is_annotated_for_the_inputs_it_handles():
    """The annotation contradicted the implementation, and mypy said so.

    ``_heatmap_cell_style`` was annotated ``(value: float, col_min: float,
    col_max: float)`` while its body coerces each one through
    ``_to_finite_float`` and RETURNS THE UNKNOWN PALETTE for a non-finite or
    absent input, which is the whole point of the function. Its only caller
    passes ``Optional[float]`` straight from that same coercion, so the file
    carried three real arg-type errors, and the reader of the signature was
    told the None path could not happen.
    """
    import inspect

    sig = inspect.signature(adapters_fairness._heatmap_cell_style)
    for param in ("value", "col_min", "col_max"):
        annotation = sig.parameters[param].annotation
        assert annotation is not float, (
            f"{param} is annotated float again, but a None reaches it from "
            "_to_finite_float on any column whose rates were unmeasurable"
        )


def test_heatmap_cell_style_really_accepts_the_absent_measurement():
    """Behavioural half of the pin: the None path the annotation denied."""
    unknown = (adapters_fairness._UNKNOWN_BG, adapters_fairness._UNKNOWN_COLOR)
    assert adapters_fairness._heatmap_cell_style(None, None, None) == unknown
    assert adapters_fairness._heatmap_cell_style(float("nan"), 0.0, 1.0) == unknown
    assert adapters_fairness._heatmap_cell_style(0.5, None, None) == unknown
    # And the measured path is untouched: at the reference, so green.
    bg, _fg = adapters_fairness._heatmap_cell_style(0.5, 0.1, 0.5)
    assert bg == "#d1fae5"


def test_mypy_agrees_the_call_site_is_clean():
    """Run the checker that reported it, on the file it reported.

    Asserting on the annotation alone is a proxy: mypy is what failed, so mypy
    is what has to pass. Skipped where mypy is not installed rather than being
    silently green.
    """
    import subprocess
    import sys

    pytest.importorskip("mypy", reason="mypy not installed in this environment")
    path = pathlib.Path(adapters_fairness.__file__)
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", str(path)],
        capture_output=True,
        text=True,
    )
    assert "_heatmap_cell_style" not in proc.stdout, (
        f"mypy still reports the heatmap call site:\n{proc.stdout}"
    )


# 11.  The subtitle line is budgeted in PIXELS, so healthy data keeps its text


_FULL_RADAR_TAIL = "dots outside it pass. Labels show raw values."


def _healthy_radar_report():
    """A fully measured, fully directed, assessable two-group report."""
    return {
        "metrics": {
            "demographic_parity_difference": 0.01,
            "equal_opportunity_difference": 0.02,
            "demographic_parity_ratio": 0.95,
            "equalized_odds_difference": 0.03,
        },
        "thresholds_used": {
            "demographic_parity_difference": 0.10,
            "equal_opportunity_difference": 0.10,
            "demographic_parity_ratio": 0.80,
            "equalized_odds_difference": 0.10,
        },
        "assessment": {"fairness_score": 92, "assessable": True},
        "data_info": {"valid_groups": ["A", "B"], "group_sizes": {"A": 500, "B": 500}},
    }


def _rendered_subtitle(svg):
    """The subtitle as the template writes it: the 10px line under the title."""
    found = re.search(r'font-size="10" fill="[^"]*">([^<]*)</text>', svg)
    assert found, "no subtitle line in the rendered chart"
    return found.group(1)


def test_healthy_radar_keeps_the_sentence_that_explains_its_own_dots():
    """NEGATIVE case. On this input every metric direction is known and every
    value is finite, so nothing is appended to the subtitle: the truncation was
    a pure loss of line width. It cut at "dots outside it…" and took with it the
    only text saying what the dots and the printed labels mean."""
    svg = adapters_fairness.radar_chart_to_svg(_healthy_radar_report())
    subtitle = _rendered_subtitle(svg)
    assert subtitle.endswith(_FULL_RADAR_TAIL), f"subtitle was cut short: {subtitle!r}"
    assert "…" not in subtitle


def test_the_subtitle_budget_is_width_and_not_a_character_count():
    """A count was wrong in both directions at once: it threw away 80px of a
    644px line for ordinary prose, and it let a line of capitals overflow the
    canvas it was meant to protect."""
    fit = adapters_fairness._fit_subtitle
    px = adapters_fairness._subtitle_px
    line = adapters_fairness._SUBTITLE_LINE_PX

    # Narrow prose that a 116-character cap would have cut.
    prose = "i" * 200
    assert fit(prose) == prose, "a line of the narrowest glyphs is nowhere near the edge"

    # Wide text a 116-character cap would have waved through.
    wide = "W" * 116
    assert px(wide) > line, "the width model no longer sees a wide line as wide"
    assert fit(wide) != wide, "116 wide characters overflow the line and were not trimmed"


def test_a_genuinely_long_subtitle_is_still_trimmed_inside_the_line():
    """Positive control for the trim itself: the NOT ASSESSABLE reason is long
    on purpose, and it must still be cut to fit rather than run off the canvas."""
    long_reason = (
        "NOT ASSESSABLE: 1 comparable group(s) after filtering (2 are needed), so no "
        "between-group comparison was performed. Excluded: Nonbinary respondents (n=3), "
        "Prefer not to say (n=2), Another long group label (n=1). This chart certifies nothing."
    )
    fitted = adapters_fairness._fit_subtitle(long_reason)
    assert fitted != long_reason and fitted.endswith("…")
    assert adapters_fairness._subtitle_px(fitted) <= adapters_fairness._SUBTITLE_LINE_PX


def test_every_subtitle_a_fairness_chart_emits_fits_its_line():
    """The invariant across the adapters, not just the radar: whatever text an
    adapter puts on the subtitle line, the line has room for it."""
    report = _healthy_radar_report()
    svgs = [
        adapters_fairness.radar_chart_to_svg(report),
        adapters_fairness.metrics_bar_chart_to_svg(report),
        adapters_fairness.effect_sizes_to_svg(report),
    ]
    for svg in svgs:
        subtitle = _rendered_subtitle(svg)
        assert adapters_fairness._subtitle_px(subtitle) <= adapters_fairness._SUBTITLE_LINE_PX, (
            f"subtitle overflows its line: {subtitle!r}"
        )


def test_the_rasterised_subtitle_stays_on_the_canvas():
    """Verify the GEOMETRY by rendering it, not by reading the source.

    The width model is calibrated against rasterised ink, so the pin is too: the
    chart is rendered to PNG and the subtitle band is measured. Skipped, never
    silently passed, where the rasteriser or PIL is missing.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("rsvg-convert") is None:
        pytest.skip("rsvg-convert not available")
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")

    svg = adapters_fairness.radar_chart_to_svg(_healthy_radar_report())
    assert _rendered_subtitle(svg).endswith(_FULL_RADAR_TAIL)

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "radar.svg"
        out = pathlib.Path(tmp) / "radar.png"
        src.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-w", "2040", str(src), "-o", str(out)],
            check=True,
            capture_output=True,
        )
        img = Image.open(out)
        scale = img.width / 680.0
        pixels = np.array(img.convert("L"))
        # The subtitle baseline is y=122 at font-size 10 (radar_chart.svg).
        band = pixels[int(112 * scale) : int(127 * scale), :]
        inked = np.where((band < 190).any(axis=0))[0]
        assert len(inked), "no subtitle ink found in the rasterised band"
        right_edge = inked.max() / scale
        assert right_edge <= 680, f"the subtitle runs off the 680px canvas at {right_edge:.1f}px"


# 12.  The dashboard's badge, headline and <desc> are one state


def _dashboard(metric, *, status="green", score=100):
    from vfairness.rendering import adapters_reporting

    return adapters_reporting.reporting_dashboard_to_svg(
        {
            "health_score": {"score": score, "status": status, "components": {}},
            "metrics": [metric],
            "tier": "OPERATIONAL",
            "sections": [],
            "alerts": [],
            "recommendations": [],
        }
    )


def _desc(svg):
    found = re.search(r"<desc>([^<]*)</desc>", svg)
    assert found, "no accessible <desc>"
    return found.group(1)


_ROW_BADGE_RE = re.compile(
    r'text-anchor="middle" font-size="8" font-weight="600" fill="[^"]*">([A-Z]+)</text>'
)


def test_dashboard_desc_is_not_more_confident_than_the_row_beside_it():
    """NEGATIVE case. The health score arrives from a separate upstream
    calculation, so it reported GREEN 100/100 next to a row this very adapter
    had just graded BREACH, and ``explain._fr_reporting`` maps GREEN to severity
    INFO. The screen-reader user therefore got the MORE confident sentence, from
    the one surface that carries no colour at all."""
    svg = _dashboard({"name": "demographic_parity_difference", "value": 0.45, "threshold": 0.10})
    assert _ROW_BADGE_RE.findall(svg) == ["BREACH"], "this input no longer breaches; test vacuous"
    assert ">GREEN<" not in svg, "the GREEN all-clear survived beside a measured breach"
    desc = _desc(svg)
    assert "(GREEN)" not in desc and "severity: INFO" not in desc, (
        f"<desc> still certifies a breached run: {desc!r}"
    )
    assert "(RED)" in desc and "severity: CRITICAL" in desc
    assert "BREACH:" in svg, "the narrative does not name what breached"
    assert "demographic_parity_difference" in svg


def test_dashboard_green_survives_a_run_with_nothing_wrong_in_it():
    """Positive control for the escalation: a measured, fully checked pass keeps
    its GREEN badge and its INFO description."""
    svg = _dashboard({"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10})
    assert _ROW_BADGE_RE.findall(svg) == ["PASS"]
    assert ">GREEN<" in svg
    desc = _desc(svg)
    assert "(GREEN)" in desc and "severity: INFO" in desc


def test_dashboard_never_softens_a_status_the_producer_already_raised():
    """The resolution only ever escalates. A producer that says RED keeps RED,
    breach or no breach."""
    svg = _dashboard(
        {"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10},
        status="red",
        score=20,
    )
    assert ">RED<" in svg and ">GREEN<" not in svg


def test_a_minimum_no_value_can_miss_is_not_a_measured_pass_on_the_dashboard():
    """NEGATIVE case wave 3 flagged and closed only in adapters_fairness.

    On a higher-is-better metric the threshold is a required MINIMUM, and ratios
    are non-negative, so a minimum of 0.0 is met by every possible value
    including the worst one. ``demographic_parity_ratio = 0.00`` means the
    protected group is never selected at all, and this dashboard rendered it as
    a green PASS because the comparison it never really made came back true.
    """
    svg = _dashboard({"name": "demographic_parity_ratio", "value": 0.0, "threshold": 0.0})
    assert _ROW_BADGE_RE.findall(svg) != ["PASS"], "a maximal violation kept its PASS badge"
    assert "[UNCHECKED]" in svg and "NOT CHECKED" in svg
    assert ">GREEN<" not in svg
    assert "COULD NOT CHECK" in svg


def test_a_real_four_fifths_floor_on_the_dashboard_still_grades_both_ways():
    """Positive control: the guard refuses only the bound that cannot be missed.
    A genuine 0.80 floor still passes above it and breaches below it.

    0.99 rather than 0.95, because the dashboard's own proximity band badges
    anything within 20 percent of its bound as WARN, and 0.95 sits 18.75 percent
    above 0.80. That band is not what is under test here.
    """
    fine = _dashboard({"name": "demographic_parity_ratio", "value": 0.99, "threshold": 0.80})
    assert _ROW_BADGE_RE.findall(fine) == ["PASS"]
    assert ">GREEN<" in fine

    bad = _dashboard({"name": "demographic_parity_ratio", "value": 0.50, "threshold": 0.80})
    assert _ROW_BADGE_RE.findall(bad) == ["BREACH"]
    assert ">RED<" in bad


def test_zero_tolerance_on_a_difference_metric_is_untouched_on_the_dashboard():
    """The MIRROR case, which is not degenerate: 0.0 on a lower-is-better metric
    is a real zero-tolerance policy that a value can and does exceed."""
    ok = _dashboard({"name": "demographic_parity_difference", "value": 0.0, "threshold": 0.0})
    assert _ROW_BADGE_RE.findall(ok) == ["PASS"]

    breach = _dashboard({"name": "demographic_parity_difference", "value": 0.02, "threshold": 0.0})
    assert _ROW_BADGE_RE.findall(breach) == ["BREACH"]


# 13.  The last private direction rule outside the rendering package


def test_no_module_in_the_direction_family_re_derives_a_direction():
    """The section-9 guard, widened past ``rendering/``.

    ``in_processing/analyzer._first_violation`` kept its own copy of the rule as
    an exact ``endswith("_ratio")`` suffix test, which is the narrow form: it
    misses ``disparate_impact`` (a four-fifths ratio that does not end in
    "_ratio") and ``worst_group_accuracy``, so a HEALTHY 0.85 ratio was reported
    as a 0.85 violation magnitude. The scan lives here because the guard it
    extends does.
    """
    from vfairness.in_processing import analyzer as in_processing_analyzer

    offences = []
    paths = sorted(_RENDERING_SRC.glob("*.py")) + [pathlib.Path(in_processing_analyzer.__file__)]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        prose = _docstring_lines(source)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if lineno in prose:
                continue
            code = line.split("#", 1)[0]
            for pattern, what in _LOCAL_DIRECTION_TESTS:
                if pattern.search(code):
                    offences.append(f"{path.name}:{lineno}: {what} -> {line.strip()}")
    assert not offences, (
        "a module is deciding metric direction from the name again; import "
        "metric_direction/is_ratio_metric instead:\n" + "\n".join(offences)
    )


def test_a_healthy_ratio_is_not_reported_as_a_violation_magnitude():
    """NEGATIVE case for the rule above, on the value it produces.

    ``disparate_impact`` IS the four-fifths ratio: 0.85 clears the rule. The
    suffix test did not recognise it, took abs(0.85) as the run's disparity, and
    fed it to the baseline-vs-fair comparison as a violation of 0.85.
    """
    from vfairness.in_processing.analyzer import _first_violation

    # The sentinel for "found no violation-type metric" became None on
    # 2026-09-08 (audit H-08): 0.0 also reads as "a violation of exactly zero",
    # and baseline_comparison_summary was rendering it as "constraints left
    # disparity unchanged" for a comparison nobody could make. The claim this
    # test exists to pin is unchanged and asserted directly below: a healthy
    # ratio must NOT come back as its own magnitude.
    assert _first_violation({"accuracy": 0.9, "disparate_impact": 0.85}) is None
    assert _first_violation({"accuracy": 0.9, "worst_group_accuracy": 0.72}) is None
    assert _first_violation({"accuracy": 0.9, "disparate_impact": 0.85}) != 0.85
    assert _first_violation({"accuracy": 0.9, "worst_group_accuracy": 0.72}) != 0.72


def test_a_real_violation_magnitude_is_still_counted():
    """Positive controls, including the cali-BRATIO-n metric a substring test
    would skip and the unresolved name that must NOT be skipped: answering 0.0
    for a disparity nobody looked at reads as "no disparity"."""
    from vfairness.in_processing.analyzer import _first_violation

    assert _first_violation({"accuracy": 0.9, "calibration_difference": 0.42}) == 0.42
    assert _first_violation({"accuracy": 0.9, "multicalibration": 0.3}) == 0.3
    assert _first_violation({"accuracy": 0.9, "theil_index": 0.31}) == 0.31
    assert _first_violation({"accuracy": 0.9, "disparate_impact_ratio": 0.85}) is None
