"""Audit wave 5: rendering robustness + template coverage.

Pins the fixes for:
  * engine boundary: malformed data (stringified numbers, Infinity, None in
    nested dicts, empty divisor lists) either renders via a template guard or
    raises a clear ValueError naming the template, never a bare
    ZeroDivisionError / OverflowError / TypeError from inside Jinja;
  * template guards: bias_audit (empty modules), cicd_pipeline (None nested
    lists), confidence_intervals (float/str n_ticks), group_comparison
    (non-finite rates through |int);
  * explain.py: fairness score 0 is data, not "missing"; causal severities
    align with Baron-Kenny step counts; robustness never displays a rounded
    score in a better band than its verdict;
  * skins.py: hex recolouring is limited to markup and <style> CSS, never
    visible text content;
  * adapters: calibration_disparity tolerates a group without 'ece'; the radar
    accepts 0-1 and 0-100 fairness scores; proxy risk keeps a CRITICAL label;
    transformation_comparison derives band and average colour from the same
    thresholds;
  * registry: intersectional_disparity and workflow_overview are registered so
    the render smoke suite covers them.

Plus behaviour tests for six major adapters: the on-canvas verdict text must
match the input data for crafted pass / warn / fail cases.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jinja2", reason="rendering requires jinja2")

from vfairness._registry import CAPABILITY_REGISTRY
from vfairness.rendering.engine import render_svg
from vfairness.rendering.explain import build_explanation
from vfairness.rendering.skins import apply_skin

# ── Engine boundary + named template guards ─────────────────────────────────


def _bias_audit_data(**over):
    d = {
        "title": "t",
        "modules": [],
        "issues": [],
        "bars": [],
        "recommendations": [],
        "n_critical": 0,
        "overall_score": 0.1,
        "overall_color": "#059669",
        "overall_bg": "#d1fae5",
        "overall_label": "LOW",
        "explanation": "",
    }
    d.update(over)
    return d


def test_bias_audit_empty_modules_renders():
    # Was: ZeroDivisionError from `608 // modules|length`.
    out = render_svg("bias_audit", _bias_audit_data())
    assert out.lstrip().startswith("<svg")


def test_cicd_pipeline_none_nested_lists_render():
    # Was: TypeError "object of type 'NoneType' has no len()" on gate.reasons.
    out = render_svg(
        "cicd_pipeline",
        {
            "title": "t",
            "tests": None,
            "validation": {"issues": None, "passed": True, "n_errors": 0, "n_warnings": 0},
            "gate": {"approved": True, "status": "ok", "reasons": None, "warnings": None},
            "explanation": "",
        },
    )
    assert out.lstrip().startswith("<svg")


@pytest.mark.parametrize("n_ticks", [5.0, "5", "not a number"])
def test_confidence_intervals_nonint_n_ticks_renders(n_ticks):
    # Was: TypeError from range(float) / range(str).
    out = render_svg(
        "confidence_intervals",
        {
            "title": "t",
            "subtitle": "s",
            "intervals": [],
            "n_ticks": n_ticks,
            "tick_labels": ["a"],
            "threshold_x": None,
            "summary_bg": "#ffffff",
            "summary_color": "#059669",
            "summary_icon": "i",
            "summary_text": "x",
            "n_significant": 0,
            "n_total": 0,
            "x_axis_label": "v",
            "explanation": "",
        },
    )
    assert out.lstrip().startswith("<svg")


@pytest.mark.parametrize("rate", [float("inf"), float("-inf"), float("nan")])
def test_group_comparison_nonfinite_rate_renders(rate):
    # Was: OverflowError "cannot convert float infinity to integer" via |int.
    out = render_svg(
        "group_comparison",
        {
            "title": "t",
            "subtitle": "s",
            "metric_label": "rate",
            "max_disparity": None,
            "overall_rate": rate,
            "bars": [
                {"group": "a", "rate": rate, "color": "#059669", "display_value": "?", "n": 3}
            ],
            "explanation": "",
        },
    )
    assert out.lstrip().startswith("<svg")


def test_stringified_number_raises_clear_valueerror():
    # A str fairness_score hits `>=` against an int inside the template; the
    # engine must surface that as a ValueError naming the template, not a
    # bare TypeError from Jinja internals.
    with pytest.raises(ValueError, match="fairness_report"):
        render_svg(
            "fairness_report",
            {
                "title": "t",
                "fairness_score": "85",
                "n_passed": 1,
                "n_failed": 0,
                "cards": [],
                "bars": [],
                "explanation": "",
            },
        )


def test_infinity_through_int_raises_clear_valueerror():
    # bias_audit applies |int to overall_score * 100; Infinity overflows and
    # must be re-raised as a ValueError naming the template.
    with pytest.raises(ValueError, match="bias_audit"):
        render_svg("bias_audit", _bias_audit_data(overall_score=float("inf")))


# ── explain.py severities and display bands ─────────────────────────────────


def test_fairness_report_zero_score_is_high_not_info():
    ce = build_explanation("fairness_report", {"fairness_score": 0, "n_passed": 0, "n_failed": 0})
    assert ce.severity == "high"


def test_fairness_report_good_score_stays_info():
    ce = build_explanation("fairness_report", {"fairness_score": 85, "n_passed": 4, "n_failed": 0})
    assert ce.severity == "info"


def test_fairness_report_missing_score_with_failures_stays_high():
    # Pre-fix behaviour preserved: unknown score + failed metrics is high.
    ce = build_explanation("fairness_report", {"n_passed": 1, "n_failed": 2})
    assert ce.severity == "high"


@pytest.mark.parametrize(
    "n_ok,expected",
    [
        (0, "high"),
        (1, "high"),
        (2, "medium"),
        (3, "medium"),
        (4, "low"),
    ],
)
def test_causal_severity_matches_baron_kenny_steps(n_ok, expected):
    ce = build_explanation(
        "causal_decomposition", {"n_steps_ok": n_ok, "mediator": "m", "prop_med_pct": 10}
    )
    assert ce.severity == expected


def test_robustness_display_stays_in_verdict_band():
    # 0.795 is below the 0.8 pass boundary; it must not display as "0.80"
    # beside a MARGINAL verdict.
    ce = build_explanation(
        "robustness_testing", {"overall_score": 0.795, "overall_label": "MARGINAL", "subgroup": {}}
    )
    assert "0.795" in ce.finding
    assert "0.80" not in ce.finding
    assert ce.severity == "medium"
    # A genuinely passing score still displays at two decimals.
    ce = build_explanation(
        "robustness_testing", {"overall_score": 0.85, "overall_label": "ROBUST", "subgroup": {}}
    )
    assert "0.85" in ce.finding
    # W-28. This asserted "info" until 2026-09-07 and pinned the defect: this
    # dict carries `subgroup: {}`, meaning NO subgroup audit ran, and "info" is
    # what a COMPLETE audit finding nothing wrong is graded. The severity is now
    # floored. The display half of this test, which is its actual purpose, is
    # untouched: 0.795 still must not print as 0.80.
    assert ce.severity == "medium"
    # The display band with a real, complete audit still grades as a pass.
    ce = build_explanation(
        "robustness_testing",
        {
            "overall_score": 0.85,
            "overall_label": "ROBUST",
            "subgroup": {"n_analyzed": 4, "n_flagged": 0},
        },
    )
    assert "0.85" in ce.finding
    assert ce.severity == "info"


# ── skins.py: recolour markup only, never visible text ──────────────────────


def test_skin_does_not_rewrite_hex_in_text_content():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text fill="#dc2626">the token #dc2626 appears in prose</text>'
        "<style>.warn{fill:#dc2626}</style>"
        "</svg>"
    )
    out = apply_skin(svg)
    # Attribute and CSS occurrences are remapped to the Blanco terracotta.
    assert 'fill="#b6573a"' in out
    assert ".warn{fill:#b6573a}" in out
    # The visible text content keeps the original token verbatim.
    assert "the token #dc2626 appears in prose" in out


def test_skin_markup_only_output_unchanged():
    # An SVG whose text carries no hex tokens recolours exactly as before
    # (verified byte-wise over all registered templates during the fix).
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<rect fill="#ef4444" stroke="#1e293b"/>'
        '<text fill="#64748b">plain label</text>'
        "</svg>"
    )
    out = apply_skin(svg)
    assert 'fill="#b6573a"' in out and 'stroke="#2e4057"' in out
    assert 'fill="#5a6a78"' in out
    assert ">plain label<" in out


# ── Adapter behaviour: on-canvas verdicts must match the input data ─────────


def test_adapter_radar_verdicts_both_scales():
    from vfairness.rendering.adapters_fairness import radar_chart_to_svg

    metrics = {"demographic_parity_difference": 0.02}
    # 0-1 scale
    assert "✓ Fair" in radar_chart_to_svg(
        {"metrics": metrics, "assessment": {"fairness_score": 0.85}}
    )
    assert "⚠ Marginal" in radar_chart_to_svg(
        {"metrics": metrics, "assessment": {"fairness_score": 0.6}}
    )
    assert "✗ Unfair" in radar_chart_to_svg(
        {"metrics": metrics, "assessment": {"fairness_score": 0.3}}
    )
    # 0-100 scale (sibling adapters emit this): same verdicts.
    assert "✓ Fair" in radar_chart_to_svg(
        {"metrics": metrics, "assessment": {"fairness_score": 85}}
    )
    assert "✗ Unfair" in radar_chart_to_svg(
        {"metrics": metrics, "assessment": {"fairness_score": 45}}
    )


def test_adapter_group_comparison_disparity_bands():
    from vfairness.rendering.adapters_fairness import group_comparison_to_svg

    def report(rate_a, rate_b):
        return {
            "group_stats": {
                "A": {"positive_rate": rate_a, "count": 50},
                "B": {"positive_rate": rate_b, "count": 50},
            }
        }

    assert "✓ Low" in group_comparison_to_svg(report(0.50, 0.48))
    assert "⚠ Moderate" in group_comparison_to_svg(report(0.50, 0.35))
    out = group_comparison_to_svg(report(0.60, 0.20))
    assert "✗ High" in out
    assert "0.400" in out  # the max disparity value itself is on canvas


def test_adapter_metrics_bar_chart_pass_fail():
    from vfairness.rendering.adapters_fairness import metrics_bar_chart_to_svg

    # 0.02 passes the 0.10 default threshold; 0.40 fails it.
    out = metrics_bar_chart_to_svg({"metrics": {"demographic_parity_difference": 0.02}})
    assert "✓ FAIR" in out and "✗ FAIL" not in out
    out = metrics_bar_chart_to_svg({"metrics": {"demographic_parity_difference": 0.40}})
    assert "✗ FAIL" in out and "✓ FAIR" not in out


def test_adapter_transformation_comparison_bands_consistent():
    import vfairness.rendering.adapters_feature_engineering as afe

    assert "Effective" in afe.transformation_comparison_to_svg(
        {"f": 1.0}, {"f": 0.4}
    )  # reduction 0.60
    assert "Moderate" in afe.transformation_comparison_to_svg(
        {"f": 1.0}, {"f": 0.85}
    )  # reduction 0.15
    assert "Minimal" in afe.transformation_comparison_to_svg(
        {"f": 1.0}, {"f": 1.0}
    )  # reduction 0.00

    # The average's colour comes from the SAME thresholds as the band: at a
    # 0.25 reduction (Moderate) both must be the amber summary colour.
    captured = {}
    orig = afe.render_svg
    afe.render_svg = lambda name, data, **kw: captured.update(data) or orig(name, data, **kw)
    try:
        afe.transformation_comparison_to_svg({"f": 0.4}, {"f": 0.3})
    finally:
        afe.render_svg = orig
    assert captured["avg_color"] == captured["summary_color"]


def test_adapter_proxy_risk_keeps_critical_label():
    from vfairness.rendering.adapters_feature_engineering import proxy_risk_to_svg

    class PV:
        def __init__(self, feature, risk_level, correlation):
            self.feature = feature
            self.risk_level = risk_level
            self.correlation = correlation
            self.protected_attribute = "gender"
            self.confidence_score = 0.9

    out = proxy_risk_to_svg(
        [
            PV("zip", "critical", 0.95),
            PV("income", "high", 0.75),
            PV("age_band", "medium", 0.45),
            PV("tenure", "low", 0.10),
        ]
    )
    # Per-feature badges and the summary count row both name CRITICAL.
    assert "CRITICAL: 1" in out
    assert "HIGH: 1" in out
    assert "MEDIUM: 1" in out
    assert "LOW: 1" in out


def test_adapter_calibration_disparity_missing_ece_renders_na():
    from vfairness.rendering.adapters_calibration import (
        calibration_disparity_to_svg,
    )

    out = calibration_disparity_to_svg(
        {
            "group_metrics": {
                "alpha": {"ece": 0.02},
                "beta": {"ece": 0.09},
                "gamma": {"brier": 0.2},  # no 'ece' key: was a TypeError crash
            },
        }
    )
    assert out.lstrip().startswith("<svg")
    assert "N/A" in out  # the ece-less group renders N/A, not garbage
    assert "beta" in out  # worst-calibrated group is still named
    assert "0.070" in out  # disparity computed over the groups WITH ece


# ── Registry coverage ────────────────────────────────────────────────────────


def test_registry_covers_intersectional_disparity_and_workflow_overview():
    registered = {t for e in CAPABILITY_REGISTRY.values() for t in e.get("svg_templates", [])}
    assert "intersectional_disparity" in registered
    assert "workflow_overview" in registered


def test_workflow_overview_adapter_renders_the_reference_set_behind_example():
    """Superseded 2026-08-27. This used to call the adapter with NO arguments.

    The built-in MLflow / W&B / pytest rows are vfairness's own reference
    integrations, not a reading of the caller's pipeline, and substituting them
    silently produced a page indistinguishable from one built from a real
    ``integrations=`` list. They now need an explicit ``example=True`` and carry
    the EXAMPLE watermark; a bare call is COULD NOT CHECK. The full guard lives
    in ``tests/test_adapters_zero_and_empty.py``.
    """
    from vfairness.rendering.adapters_workflow import workflow_overview_to_svg

    out = workflow_overview_to_svg(example=True)
    assert out.lstrip().startswith("<svg")
    assert "MLflow" in out
    assert "EXAMPLE ONLY" in out

    bare = workflow_overview_to_svg()
    assert bare.lstrip().startswith("<svg")
    assert "MLflow" not in bare
    assert "NOT CHECKED" in bare


def test_intersectional_disparity_adapter_renders_and_names_groups():
    from vfairness.rendering.adapters_feature_engineering import (
        intersectional_disparity_to_svg,
    )

    out = intersectional_disparity_to_svg(
        {
            "privilegedGroup": {"group": "white_male", "positiveRate": 0.7, "size": 100},
            "disadvantagedGroup": {"group": "black_female", "positiveRate": 0.3, "size": 80},
            "allGroups": [
                {"group": "white_male", "positiveRate": 0.7, "size": 100, "severity": "low"},
                {"group": "black_female", "positiveRate": 0.3, "size": 80, "severity": "high"},
            ],
            "maxDisparity": 0.4,
            "disparitySeverity": "high",
            "insights": ["Largest gap is 40 percentage points."],
            "findings": [{"severity": "high"}],
        }
    )
    assert out.lstrip().startswith("<svg")
    assert "white_male" in out and "black_female" in out
    assert "HIGH" in out
