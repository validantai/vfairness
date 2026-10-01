"""Row-level fabrication guards for the four CORE rendering adapters.

Scope: ``rendering/adapters.py``, ``rendering/adapters_post_processing.py``,
``rendering/adapters_fairness.py``, ``rendering/adapters_reporting.py`` and the
SVG templates they render.

The pattern under test, in one sentence: a value the report did not carry is
replaced by a default, and the DEFAULT IS THEN GRADED, counted, coloured,
sorted by or plotted as though it were a measurement. Both directions are
defects. A fabricated all-clear ("0.000 disparity", "all pass", "n=0" beside a
populated group) and a fabricated breach ("0/100 UNFAIR", "0 affected") are
equally wrong, and the second is not the safer error.

Two of the cases below live in the TEMPLATES rather than in Python, and neither
is visible to a substring assertion over the markup:

* a two-state verdict branch, where absent renders as the negative state;
* a ``{% set counter = counter + 1 %}`` inside a ``{% for %}``, which Jinja
  discards at the end of each iteration, so every item is drawn at the same
  coordinate and only the last one painted survives. Those are asserted on the
  RENDERED GEOMETRY (distinct y coordinates), which is what a rasteriser sees.
"""

import re
import warnings

import pytest

from vfairness.rendering import adapters, adapters_fairness, adapters_post_processing
from vfairness.rendering.adapters import (
    bias_audit_to_svg,
    cicd_pipeline_to_svg,
    fairness_report_to_svg,
)
from vfairness.rendering.adapters_fairness import radar_chart_to_svg
from vfairness.rendering.adapters_post_processing import (
    fairness_detailed_report_to_svg,
    threshold_optimization_to_svg,
)
from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

# Fixtures: the smallest report shapes that reproduce each site.


class _Enum:
    def __init__(self, value):
        self.value = value


class _Finding:
    """A bias-audit finding carrying only the attributes it is given."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _BiasReport:
    """A BiasAuditReport-shaped object with every attribute overridable."""

    def __init__(self, **kwargs):
        self.historical_findings = []
        self.representation_findings = []
        self.disparity_findings = []
        self.proxy_findings = []
        self.overall_risk_score = 0.0
        self.critical_issues = []
        self.recommendations = []
        self.protected_attributes = ["gender"]
        self.modules_run = list(adapters._AUDIT_MODULES)
        self.timestamp = "2026-01-02 03:04"
        for key, value in kwargs.items():
            setattr(self, key, value)

    def execution_coverage(self):
        recorded = set(self.modules_run or [])
        ran = [m for m in adapters._AUDIT_MODULES if m in recorded]
        if len(ran) == len(adapters._AUDIT_MODULES):
            return "complete"
        return "partial" if ran else "none"


class _RepresentationFinding:
    def __init__(self, distributions, ratios):
        self.attribute = "gender"
        self.group_distributions = distributions
        self.representation_ratios = ratios
        self.severity = _Enum("high")


def _text_at(svg, y, tolerance=0.51):
    """Every <text> body drawn within *tolerance* of the y coordinate."""
    out = []
    for match in re.finditer(r'<text[^>]*\by="([\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
        if abs(float(match.group(1)) - y) <= tolerance:
            out.append(match.group(2))
    return out


def _headline_text(svg, max_y=200.0):
    """All <text> drawn in the headline band, y <= max_y."""
    out = []
    for match in re.finditer(r'<text[^>]*\by="([\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
        if float(match.group(1)) <= max_y:
            out.append(match.group(2))
    return " ".join(out)


def _desc(svg):
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    return match.group(1) if match else ""


# 1. THE CRASH. A report with REAL findings and no overall risk score.


def test_bias_audit_with_findings_and_no_score_does_not_crash():
    """A partial row must never take the canvas down.

    ``not_assessable`` keyed only on the finding count, which is the wrong
    question to ask about the OVERALL RISK badge: the badge renders
    ``overall_risk_score``, a separate field. A report with findings and no
    score took the assessable branch, handed the template None, and
    ``(overall_score * 100)|int`` raised TypeError, which engine.render_svg
    re-raises as ValueError.
    """
    report = _BiasReport(
        historical_findings=[_Finding(risk_level=_Enum("high"))],
        overall_risk_score=None,
    )
    svg = bias_audit_to_svg(report)
    assert svg.startswith("<svg") or "<svg" in svg
    assert len(svg) > 500


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), "not measured"])
def test_bias_audit_withholds_an_unreported_overall_score(bad_score):
    """No number, no band label, and the findings are not swept away with it."""
    report = _BiasReport(
        historical_findings=[_Finding(risk_level=_Enum("high"))],
        overall_risk_score=bad_score,
    )
    svg = bias_audit_to_svg(report)
    headline = _headline_text(svg)
    assert "NOT ASSESSABLE" in headline
    # Neither the emerald MINIMAL band nor a percentage may stand in for it.
    assert "MINIMAL" not in headline
    assert not re.search(r">\s*\d+%\s*<", headline)
    # And the reason must not read as an all-clear over the findings.
    assert "no overall risk score" in headline.lower()
    assert "below stand" in headline
    # THE DESCRIPTION MUST AGREE WITH THE CANVAS. Withholding the badge must not
    # be done by raising `not_assessable`, which is what rendering.explain keys
    # on: that replaces the whole <desc> with "nothing on this chart was
    # assessed" while the issue table beside it lists the finding. The sighted
    # reader would see the finding and the screen-reader user would be told
    # there was none.
    desc = _desc(svg)
    assert "nothing on this chart was assessed" not in desc
    assert "bias-risk score" in desc


@pytest.mark.parametrize("good_score", [0.32, "0.32"])
def test_bias_audit_measured_score_is_untouched(good_score):
    """The control: a reported score still renders exactly as before.

    A stringified "0.32" is a reported measurement badly typed, not an absence,
    so it is coerced and graded. It used to reach ``_risk_color`` as a str and
    raise TypeError before the template ever saw it.
    """
    report = _BiasReport(
        historical_findings=[_Finding(confidence_score=0.4)],
        overall_risk_score=good_score,
    )
    svg = bias_audit_to_svg(report)
    headline = _headline_text(svg)
    assert "32%" in headline
    assert "NOT ASSESSABLE" not in headline


# 2. A finding with no readable severity is not graded into a module score.


def test_absent_risk_level_is_not_scored_as_negligible():
    """``risk_level=None`` used to map through str(None) == "none" to 0.05.

    0.05 is the NEGLIGIBLE band, so an absent severity actively pulled the
    module's risk DOWN: absence lowering a risk is the quiet direction of this
    bug, and it is the one that ships a false all-clear.
    """
    score, n_ungraded = adapters._module_score([_Finding(risk_level=None)], ran=True)
    assert score is None
    assert n_ungraded == 1


def test_unknown_severity_is_not_scored_as_medium():
    """The mirror direction: an unreadable severity used to score a flat 0.5."""
    score, n_ungraded = adapters._module_score([_Finding(severity=_Enum("weird"))], ran=True)
    assert score is None
    assert n_ungraded == 1

    score, n_ungraded = adapters._module_score([_Finding(other="x")], ran=True)
    assert score is None
    assert n_ungraded == 1


def test_ungraded_finding_does_not_dilute_a_measured_one():
    """Part (d): an ungraded row enters neither numerator nor denominator."""
    score, n_ungraded = adapters._module_score(
        [_Finding(risk_level=_Enum("high")), _Finding(risk_level=None)], ran=True
    )
    assert score == pytest.approx(0.75)  # the HIGH finding alone, not (0.75+0.05)/2
    assert n_ungraded == 1


def test_reported_none_severity_still_grades():
    """The control. RiskLevel.NONE is a severity the detector REPORTED."""
    score, n_ungraded = adapters._module_score([_Finding(risk_level=_Enum("none"))], ran=True)
    assert score == pytest.approx(0.05)
    assert n_ungraded == 0


def test_ungraded_findings_are_counted_on_the_canvas():
    """Part (c): the ungraded count goes in the headline band, y <= 200."""
    report = _BiasReport(
        historical_findings=[_Finding(risk_level=_Enum("high")), _Finding(risk_level=None)],
        overall_risk_score=0.4,
    )
    svg = bias_audit_to_svg(report)
    headline = _headline_text(svg)
    assert "reported no severity" in headline
    assert "1 finding(s)" in headline


# 3. A representation ratio nobody reported is not a benchmark of parity.


def test_absent_representation_ratio_draws_no_benchmark_bar():
    """``ratios.get(name, 1.0)`` makes benchmark == dataset share.

    Two identical bars under a "Dataset / Population benchmark" legend is a
    picture of perfect representation, for a group nobody benchmarked.
    """
    report = _BiasReport(
        representation_findings=[
            _RepresentationFinding({"Female": 0.30}, {}),
        ],
        overall_risk_score=0.4,
    )
    bars, no_benchmark = adapters._representation_bars(report)
    assert bars[0]["benchmark_pct"] is None
    assert no_benchmark == ["Female"]

    svg = bias_audit_to_svg(report)
    assert "no population benchmark was reported" in svg
    assert "% benchmark" not in svg


def test_measured_representation_ratio_still_draws_its_benchmark():
    """The control."""
    report = _BiasReport(
        representation_findings=[
            _RepresentationFinding({"Female": 0.30}, {"Female": 0.6}),
        ],
        overall_risk_score=0.4,
    )
    bars, no_benchmark = adapters._representation_bars(report)
    assert bars[0]["benchmark_pct"] == pytest.approx(0.5)
    assert no_benchmark == []


# 4. "n=0" beside a group bar is a measured count.


def test_group_bar_without_a_size_key_says_not_reported():
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.41}},
        "assessment": {"assessable": True, "fairness_score": 0.9},
    }
    bars, _unmeasured = adapters._group_bars(report)
    assert bars[0]["n"] is None
    svg = fairness_report_to_svg(report)
    assert "n not reported" in svg
    assert "n=0" not in svg


def test_group_bar_with_a_size_key_still_prints_it():
    """The control."""
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.41, "size": 1200}},
        "assessment": {"assessable": True, "fairness_score": 0.9},
    }
    svg = fairness_report_to_svg(report)
    assert "n=1200" in svg
    assert "n not reported" not in svg


# 5. An absent fairness score is not a score of 0.


def test_absent_fairness_score_is_not_rendered_as_zero_out_of_100():
    """`assessment.get("fairness_score", 0)` was a fabricated BREACH.

    0 is finite, so it passed the suppression guard untouched and the panel
    printed a confident "0/100" in failing red for a report whose own metric
    cards can all be passing.
    """
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.41, "size": 10}},
        "assessment": {"assessable": True},  # no fairness_score key at all
    }
    svg = fairness_report_to_svg(report)
    assert "NOT ASSESSABLE" in svg
    assert not re.search(r">\s*0\s*<", svg.split("Metric Table")[0])


def test_reported_fairness_score_is_untouched():
    """The control."""
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.41, "size": 10}},
        "assessment": {"assessable": True, "fairness_score": 0.88},
    }
    svg = fairness_report_to_svg(report)
    assert ">88<" in svg


# 6. TEMPLATE SHAPE: a {% set %} counter inside a {% for %}.


class _Gate:
    def __init__(self, reasons):
        self.status = _Enum("blocked")
        self.approved = False
        self.blocking_reasons = reasons
        self.warnings = []


def test_every_blocking_reason_is_drawn_at_its_own_y():
    """Jinja discards a {% set %} made inside a {% for %}.

    ``{% set line_idx = line_idx + 1 %}`` therefore left line_idx at 0 for every
    reason, so all four were painted on top of one another and only the last one
    was legible. The markup was correct in the file, so a substring assertion
    over the SVG passed straight over it: the defect is in the GEOMETRY.
    """
    reasons = [
        "demographic_parity_difference 0.24 exceeds 0.10",
        "equal_opportunity_difference 0.19 exceeds 0.10",
        "equalized_odds_difference 0.31 exceeds 0.15",
        "calibration_difference 0.09 exceeds 0.05",
    ]
    svg = cicd_pipeline_to_svg(gate_decision=_Gate(reasons))
    ys = [
        float(m.group(1))
        for m in re.finditer(r'<text[^>]*\by="([\d.]+)"[^>]*>([^<]*)</text>', svg)
        if "exceeds" in m.group(2)
    ]
    assert len(ys) == 4, f"expected 4 reason lines, drew {len(ys)}"
    assert len(set(ys)) == 4, f"reasons overlap at {sorted(ys)}"
    assert sorted(ys) == ys or sorted(ys) == sorted(set(ys))


# 7. TEMPLATE SHAPE: a two-state verdict branch on the reporting dashboard.


def _dashboard_report(metric):
    return {
        "tier": "OPERATIONAL",
        "timestamp": "2026-01-02 03:04",
        "model_name": "m",
        "model_version": "1",
        "health_score": {
            "score": 72,
            "status": "yellow",
            "trend": "stable",
            "trend_slope": 0.0,
            "components": {
                "metric_compliance": 68,
                "alert_frequency": 75,
                "drift_stability": 82,
            },
        },
        "metrics": [metric],
    }


def test_graded_row_without_an_affected_count_does_not_claim_all_pass():
    """ "all pass" is a group-level all-clear, and it was the default branch.

    A row whose producer never counted its affected groups took the {% else %}
    arm of a two-state test on a bare number, so the cell asserted that every
    group passed, from an absent key.
    """
    svg = reporting_dashboard_to_svg(
        _dashboard_report(
            {"name": "demographic_parity_difference", "value": 0.08, "threshold": 0.10}
        )
    )
    assert "not reported" in svg
    assert "all pass" not in svg


def test_reported_zero_affected_groups_still_reads_all_pass():
    """The control: a reported 0 IS a measurement."""
    svg = reporting_dashboard_to_svg(
        _dashboard_report(
            {
                "name": "demographic_parity_difference",
                "value": 0.08,
                "threshold": 0.10,
                "n_affected_groups": 0,
            }
        )
    )
    assert "all pass" in svg


def test_breach_without_a_reported_count_does_not_invent_one():
    """``max(n_affected, 1)`` on an absent count fabricates the harm's scope."""
    svg = reporting_dashboard_to_svg(
        _dashboard_report(
            {"name": "demographic_parity_difference", "value": 0.24, "threshold": 0.10}
        )
    )
    assert "1 affected" not in svg
    assert "not reported" in svg
    # The breach itself is still stated: this is not a suppression.
    assert "BREACH" in svg


# 8. The threshold-optimisation headline counts only what it evaluated.


def test_headline_counts_only_the_groups_that_were_evaluated():
    """``ThresholdResult.to_dict()`` carries group_thresholds and nothing else.

    Every real optimiser run therefore printed "N groups evaluated" for N groups
    with no rate and no size, which breaks (b), (c) and (d) of the headline rule
    on the one line a reader takes the scope of the run from.
    """
    svg = threshold_optimization_to_svg(
        {"group_thresholds": {"Female": 0.42, "Male": 0.55, "Other": 0.5}}
    )
    headline = _headline_text(svg)
    assert "0 groups evaluated" in headline
    assert "3 groups evaluated" not in headline
    assert "3 further groups reported no rate and no size" in headline


def test_headline_counts_a_partly_evaluated_run_over_the_graded_subset():
    svg = threshold_optimization_to_svg(
        {
            "group_thresholds": {"Female": 0.42, "Male": 0.55},
            "original_rates": {"Female": 0.31},
            "optimized_rates": {"Female": 0.44},
            "group_sizes": {"Female": 1200},
        }
    )
    headline = _headline_text(svg)
    assert "1 group evaluated" in headline
    assert "1 further group reported no rate and no size" in headline
    assert "Male" in headline


def test_fully_evaluated_run_headline_is_unchanged():
    """The control."""
    svg = threshold_optimization_to_svg(
        {
            "constraint_type": "equalized_odds",
            "tolerance": 0.05,
            "objective": "accuracy",
            "group_thresholds": {"Female": 0.42, "Male": 0.55},
            "original_rates": {"Female": 0.31, "Male": 0.48},
            "optimized_rates": {"Female": 0.44, "Male": 0.46},
            "group_sizes": {"Female": 1200, "Male": 1500},
        }
    )
    headline = _headline_text(svg)
    assert "2 groups evaluated" in headline
    assert "reported no rate" not in headline


# 9. The detailed report's group SIZE and pairwise DISPARITY cells.


def _detailed_report(**overrides):
    report = {
        "task_type": "classification",
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "assessment": {"fairness_score": 0.9},
        "group_stats": {"Female": {"positive_rate": 0.41, "size": 1200}},
        "data_info": {"n_samples": 2700},
    }
    report.update(overrides)
    return report


def test_group_size_absent_is_not_printed_as_zero():
    """ "0" in the SIZE column says this stratum is EMPTY."""
    svg = fairness_detailed_report_to_svg(
        _detailed_report(group_stats={"Female": {"positive_rate": 0.41}})
    )
    assert "not reported" in svg
    groups_block = svg.split("GROUP")[-1]
    assert not re.search(r'x="220"[^>]*>0<', groups_block)


def test_group_size_present_still_prints():
    """The control."""
    svg = fairness_detailed_report_to_svg(_detailed_report())
    assert ">1200<" in svg


def test_pairwise_disparity_absent_draws_no_green_zero():
    """The DISPARITY cell is banded green below 0.05 and drawn as a bar.

    A defaulted 0 is therefore the strongest all-clear the table can express,
    for a pair the report never compared.
    """
    svg = fairness_detailed_report_to_svg(
        _detailed_report(pairwise_comparisons=[{"group_a": "Female", "group_b": "Male"}])
    )
    rows = svg.split("DISPARITY")[-1]
    assert "not reported" in rows
    assert "0.000" not in rows
    # No banded bar either: the bar's width IS the value, so drawing one is the
    # same claim in a second costume. (The engine rewrites hex colours on the
    # way out, so the assertion is on the geometry, not on a palette value.)
    assert not re.search(r'<rect x="380"[^>]*height="10"', rows)


def test_pairwise_disparity_present_is_banded_as_before():
    """The control."""
    svg = fairness_detailed_report_to_svg(
        _detailed_report(
            pairwise_comparisons=[{"group_a": "Female", "group_b": "Male", "disparity": 0.043}]
        )
    )
    rows = svg.split("DISPARITY")[-1]
    assert "0.043" in rows
    assert re.search(r'<rect x="380"[^>]*height="10"', rows)
    assert "not reported" not in rows


# 10. The radar's invented threshold and its invented status.


def test_metric_with_no_configured_threshold_is_not_plotted():
    """The radius is measured AGAINST the threshold, so it IS the verdict.

    ``_DEFAULT_THRESHOLDS.get(key, 0.5)`` invented a 0.5 bound for every metric
    this library has no default for, and the dot was then drawn inside or
    outside the pass ring on the strength of it.
    """
    report = {
        "metrics": {"theil_index": 0.45},
        "thresholds_used": {},
        "group_stats": {"Female": {"positive_rate": 0.4}, "Male": {"positive_rate": 0.5}},
        "assessment": {"assessable": True, "fairness_score": 0.9},
    }
    svg = radar_chart_to_svg(report)
    assert "no configured threshold" in svg
    assert "theil_index" in svg
    # Nothing plotted, so no polygon and no dot claim a position.
    assert 'points=""' in svg or "<polygon" not in svg


def test_metric_with_a_configured_threshold_is_still_plotted():
    """The control."""
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.4}, "Male": {"positive_rate": 0.5}},
        "assessment": {"assessable": True, "fairness_score": 0.9},
    }
    svg = radar_chart_to_svg(report)
    assert "no configured threshold" not in svg
    assert "Fair" in svg


@pytest.mark.parametrize(
    "assessment", [{"assessable": True}, {"assessable": True, "fairness_score": "high"}]
)
def test_absent_fairness_score_does_not_badge_the_radar_marginal(assessment):
    """0.5 lands in the `>= 0.5` band and painted the amber Marginal badge."""
    report = {
        "metrics": {"demographic_parity_difference": 0.04},
        "thresholds_used": {"demographic_parity_difference": 0.10},
        "group_stats": {"Female": {"positive_rate": 0.4}, "Male": {"positive_rate": 0.5}},
        "assessment": assessment,
    }
    svg = radar_chart_to_svg(report)
    # The badge itself, not the boilerplate legend prose, which names all three
    # bands by design.
    badge = _headline_text(svg)
    assert "Marginal" not in badge
    assert "NOT ASSESSABLE" in badge
    assert "carries no fairness score" in svg


# 11. Nothing above may crash on a partial row, and nothing returns "".


@pytest.mark.parametrize(
    "call",
    [
        lambda: bias_audit_to_svg(_BiasReport(overall_risk_score=None)),
        lambda: fairness_report_to_svg({"group_stats": {"A": {}}, "metrics": {}}),
        lambda: fairness_detailed_report_to_svg(
            {"group_stats": {"A": {}}, "pairwise_comparisons": [{}]}
        ),
        lambda: threshold_optimization_to_svg({"group_thresholds": {"A": 0.5}}),
        lambda: radar_chart_to_svg({"metrics": {"x": 1.0}, "group_stats": {}}),
        lambda: reporting_dashboard_to_svg(_dashboard_report({"name": "x"})),
        lambda: cicd_pipeline_to_svg(),
    ],
)
def test_a_partial_row_never_crashes_and_never_returns_empty(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = call()
    assert svg, "an adapter returned an empty string instead of a canvas"
    assert "<svg" in svg


def test_the_scanner_pattern_is_gone_from_the_four_core_adapters():
    """The file-level guard: no numeric or boolean `.get(key, default)` remains.

    One survivor is allowed and named: ``_STATUS_SEVERITY.get(status_label, 1)``
    in adapters_reporting, which is a control-flow guard for an UNRECOGNISED
    status label on the escalate-to-RED path. It grades nothing, renders
    nothing, and its default can only ever let a MEASURED breach raise the
    headline, never lower it.
    """
    import ast
    import pathlib

    from vfairness.rendering import adapters_reporting

    allowed = {("adapters_reporting.py", "_STATUS_SEVERITY.get(status_label, 1)")}
    found = set()
    for module in (adapters, adapters_post_processing, adapters_fairness, adapters_reporting):
        path = pathlib.Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 2
            ):
                continue
            default = node.args[1]
            if isinstance(default, ast.Constant) and isinstance(default.value, (int, float, bool)):
                found.add((path.name, ast.unparse(node)))
    assert found <= allowed, f"new numeric-default sites: {sorted(found - allowed)}"
