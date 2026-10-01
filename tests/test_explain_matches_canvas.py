"""The explanation layer must never be more confident than the canvas it sits on.

An SVG carries its finding twice: once in the visible Explanation paragraph and
once in the accessible ``<desc>``, which is all a screen-reader user gets. Both
are built by :mod:`vfairness.rendering.explain` from the SAME data dict the
chart's badge is built from, so the two can only disagree when a finder forgets
to read the state the template branches on. Whichever surface is more confident
is the one a reader believes, so a forgotten finder is not a cosmetic mismatch:
it is a fabricated result printed beside a withheld one.

Reproduced on the committed tree at 91b8d8c, both confirmed by rasterising the
chart and reading the pixels:

    reliability_diagram, not assessable   ->  "Model is WELL CALIBRATED
                                              (ECE 0.000, MCE 0.000)."
    regression_fairness, not assessable   ->  "Regression fairness: 0/0
                                              disparity metrics pass across 0
                                              groups; all pass."

Neither number was measured. 'WELL CALIBRATED' and 0.000 are the ADAPTER
DEFAULTS that ``_g(d, key, default)`` returns for an absent key, and 0/0 is the
empty-denominator all-clear that this suite has already removed from four other
surfaces. The badge beside each read NOT ASSESSABLE the whole time.

Two guards, deliberately of different shapes:

* the named regressions, pinned individually and end to end through the real
  adapter, so the rendered artifact is checked and not a helper's return value;
* a STRUCTURAL sweep over every chart in ``_FINDERS``, which is what would have
  caught these two without anyone naming them, and which fails for the next
  chart added without the third state.

The over-correction controls matter as much: an ordinary healthy chart must read
exactly as it did before, numbers and verdict included.
"""

import math
import re

import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)
from vfairness.rendering.explain import (
    _COULD_NOT_CHECK,
    _FINDERS,
    _NOT_ASSESSABLE_ACTION,
    build_explanation,
)

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.rendering.adapters_ranking import (  # noqa: E402
    _NOT_CHECKED_LABEL,
    _headline,
    _pf,
    _pf_color,
    ranking_fairness_to_svg,
)


def _desc(svg: str) -> str:
    """The accessible one-liner, which is all a screen-reader user receives."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "the rendered SVG carries no <desc>"
    return match.group(1)


def _is_could_not_check(text) -> bool:
    return str(text or "").strip().upper().startswith(_COULD_NOT_CHECK)


# ---------------------------------------------------------------------------
# The two named regressions.
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    """A diagram that binned nothing must not certify calibration."""

    NOT_ASSESSABLE = {
        "not_assessable": True,
        "not_assessable_reason": "no data was supplied, so nothing was measured.",
        "verdict_text": "WELL CALIBRATED",
        "ece": 0,
        "mce": 0,
    }

    def test_the_finding_is_could_not_check(self):
        explanation = build_explanation("reliability_diagram", self.NOT_ASSESSABLE)

        assert _is_could_not_check(explanation.finding), explanation.finding

    def test_the_fabricated_verdict_is_gone_from_every_surface(self):
        explanation = build_explanation("reliability_diagram", self.NOT_ASSESSABLE)

        for surface in (explanation.finding, explanation.caption(), explanation.paragraph()):
            assert "WELL CALIBRATED" not in surface, surface

    def test_the_action_stops_telling_the_reader_to_recalibrate(self):
        """ "Recalibrate if ECE exceeds 0.05" implies an ECE was computed."""
        explanation = build_explanation("reliability_diagram", self.NOT_ASSESSABLE)

        assert explanation.action == _NOT_ASSESSABLE_ACTION

    def test_the_rendered_diagram_agrees_with_its_own_badge(self):
        """End to end through the real adapter, on the artifact that ships."""
        numpy = pytest.importorskip("numpy")
        from vfairness.rendering.adapters_calibration import reliability_diagram_to_svg

        svg = reliability_diagram_to_svg(numpy.array([]), numpy.array([]))

        assert "NOT ASSESSABLE" in svg, "the canvas did not withhold its verdict"
        assert _COULD_NOT_CHECK in _desc(svg)
        assert "WELL CALIBRATED" not in svg


class TestRegressionFairness:
    """A report where a disparity metric was never checked is not an all-clear."""

    NOT_ASSESSABLE = {
        "not_assessable": True,
        "not_assessable_reason": "no disparity metric was supplied, so nothing was measured.",
        "badges": [],
        "groups": [],
        "n_pass": 0,
        "n_total": 0,
    }

    def test_the_finding_is_could_not_check(self):
        explanation = build_explanation("regression_fairness", self.NOT_ASSESSABLE)

        assert _is_could_not_check(explanation.finding), explanation.finding

    def test_the_empty_denominator_all_clear_is_gone(self):
        explanation = build_explanation("regression_fairness", self.NOT_ASSESSABLE)

        assert "0/0" not in explanation.paragraph()
        assert "all pass" not in explanation.paragraph()

    def test_a_partly_checked_report_is_not_reported_as_passing(self):
        """One unchecked metric withdraws the all-clear over the whole page."""
        from vfairness.rendering.adapters_regression import regression_fairness_to_svg

        svg = regression_fairness_to_svg({}, {"mae_parity": 0.02, "rmse_parity": None})

        assert "NOT ASSESSABLE" in svg
        assert _COULD_NOT_CHECK in _desc(svg)
        assert "all pass" not in _desc(svg)


# ---------------------------------------------------------------------------
# The structural sweep: no chart may opt out, including one added tomorrow.
# ---------------------------------------------------------------------------

_ALL_TEMPLATES = sorted(_FINDERS)


@pytest.mark.parametrize("template", _ALL_TEMPLATES)
def test_every_chart_withholds_its_finding_when_it_was_not_assessable(template):
    """The guard that does not need the next gap to be named first.

    ``_fr_reliability`` and ``_fr_regression`` were left out of the sweep that
    fixed the radar, the heatmap and the metrics bar, and nothing failed. Every
    finder is swept here, so the same omission cannot pass again.
    """
    data = {
        "not_assessable": True,
        "not_assessable_reason": "nothing was measured on this run.",
    }

    explanation = build_explanation(template, data)

    assert _is_could_not_check(explanation.finding), (
        f"{template} reported '{explanation.finding}' for a run its own data dict "
        f"marks not assessable"
    )
    assert explanation.action == _NOT_ASSESSABLE_ACTION


@pytest.mark.parametrize("template", _ALL_TEMPLATES)
def test_a_not_assessable_chart_never_reports_an_all_clear_severity(template):
    """Could-not-check is not benign news, so it is never 'info'."""
    data = {"not_assessable": True, "not_assessable_reason": "nothing was measured."}

    explanation = build_explanation(template, data)

    # group_comparison is the one deliberate exception, pinned in test_explain.py:
    # an empty chart must never claim a High disparity either.
    if template == "group_comparison":
        return
    assert explanation.severity != "info", template


@pytest.mark.parametrize(
    "template",
    ["report_card", "hierarchical_gate", "threshold_optimization_report"],
)
def test_a_verdict_is_never_written_from_an_absent_key(template):
    """``_g(d, 'approved', True)`` printed APPROVED for a dict with no decision.

    An approval is the most damaging sentence this library can emit, so it is
    written only when the data dict actually carries one.
    """
    explanation = build_explanation(template, {})

    assert _is_could_not_check(explanation.finding), explanation.finding
    # None of the three affirmative verdicts may be stated. "nothing was
    # approved here" is the refusal, so the check is on the affirmative forms.
    for claim in (": approved", "APPROVED", "is feasible"):
        assert claim not in explanation.finding


# ---------------------------------------------------------------------------
# Ranking direction: the adapter and the shared resolver must not disagree.
# ---------------------------------------------------------------------------


class TestRankingDirection:
    def test_the_adapter_and_the_resolver_agree_on_a_ratio(self):
        """The reproduced defect: 0.62 against a 0.80 four-fifths floor.

        ``check_threshold`` called it FAIL ("is below the required minimum")
        while the adapter painted a green PASS for the same two numbers.
        """
        outcome, _ = check_threshold("exposure_parity_ratio", 0.62, 0.80)

        assert outcome is ThresholdOutcome.FAIL
        assert _pf(0.62, 0.80, "exposure_parity_ratio") == "FAIL"

    def test_perfect_parity_on_the_same_ratio_still_passes(self):
        """Over-correction control: the fix must not invert the OTHER way."""
        assert _pf(1.0, 0.80, "exposure_parity_ratio") == "PASS"
        assert _pf(0.95, 0.80, "exposure_parity_ratio") == "PASS"

    def test_the_ratio_failure_reaches_the_canvas_and_the_desc(self):
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity_ratio", "value": 0.62, "threshold": 0.80}],
            {"A": {"mean_exposure": 0.60, "count": 5}, "B": {"mean_exposure": 0.37, "count": 5}},
        )

        assert ">UNFAIR<" in svg
        assert ">FAIR<" not in svg
        assert ">FAIL<" in svg
        assert "0/1" in svg
        assert "failing: exposure_parity_ratio" in _desc(svg)

    @pytest.mark.parametrize(
        "name",
        [
            "exposure_parity",
            "ndcg_parity",
            "attention_weighted_rank_fairness",
            "ndkl",
            "representation_ndkl",
            "normalized_discounted_kl_divergence",
            "exposure_parity_difference",
        ],
    )
    def test_every_lower_is_better_ranking_metric_is_registered(self, name):
        """Registered in the SHARED resolver, so every surface inherits it.

        Without these an ordinary healthy value renders NOT CHECKED, which is
        honest but is still a regression on a chart that used to grade.
        """
        assert metric_direction(name) is MetricDirection.LOWER_IS_BETTER
        assert _pf(0.02, 0.10, name) == "PASS"
        assert _pf(0.40, 0.10, name) == "FAIL"

    def test_the_ratio_sibling_is_not_swept_up_with_the_bare_name(self):
        """Exact matching, never a prefix: registering the bare family name must
        not reach ``exposure_parity_ratio``, whose direction is the opposite."""
        assert metric_direction("exposure_parity_ratio") is MetricDirection.HIGHER_IS_BETTER

    def test_an_unresolvable_direction_is_neither_pass_nor_fail(self):
        """Fail closed, and say so, rather than guessing either way."""
        assert _pf(0.02, 0.10, "shiny_new_ranking_metric_2027") == _NOT_CHECKED_LABEL

    def test_an_unchecked_metric_is_never_painted_green(self):
        pass_color = _pf_color(0.02, 0.10, "exposure_parity")
        unknown_color = _pf_color(0.02, 0.10, "shiny_new_ranking_metric_2027")

        assert unknown_color != pass_color

    def test_an_unchecked_metric_withdraws_the_pages_all_clear(self):
        svg = ranking_fairness_to_svg(
            [
                {"metric_name": "exposure_parity", "value": 0.02, "threshold": 0.10},
                {"metric_name": "shiny_new_ranking_metric_2027", "value": 0.02, "threshold": 0.10},
            ],
            {"A": {"mean_exposure": 0.50, "count": 5}, "B": {"mean_exposure": 0.48, "count": 5}},
        )

        assert ">FAIR<" not in svg
        assert ">NOT ASSESSED<" in svg
        assert "All ranking fairness metrics pass" not in svg
        assert "could not be checked" in svg
        assert _COULD_NOT_CHECK in _desc(svg)

    def test_the_headline_still_reports_a_measured_breach_over_a_missing_check(self):
        """A measured FAIL is a finding whatever else is missing."""
        assert _headline(3, 1, 1)["label"] == "UNFAIR"


# ---------------------------------------------------------------------------
# Over-correction controls. Healthy input must render exactly as it did.
# ---------------------------------------------------------------------------


class TestHealthyInputIsUnchanged:
    GROUPS = {
        "A": {"mean_exposure": 0.59, "avg_position": 2.0, "count": 5},
        "B": {"mean_exposure": 0.52, "avg_position": 3.0, "count": 5},
    }

    def test_a_healthy_ranking_page_still_reads_fair(self):
        svg = ranking_fairness_to_svg(
            [
                {"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.1},
                {"metric_name": "ndcg_parity", "value": 0.01, "threshold": 0.1},
            ],
            self.GROUPS,
        )

        assert ">FAIR<" in svg
        assert ">UNFAIR<" not in svg
        assert ">NOT ASSESSED<" not in svg
        assert "2/2" in svg
        assert "All ranking fairness metrics pass" in svg
        assert _COULD_NOT_CHECK not in svg

    def test_a_healthy_ranking_explanation_still_reports_the_numbers(self):
        explanation = build_explanation(
            "ranking_fairness",
            {"n_pass": 2, "n_total": 2, "badges": [], "groups": [1, 2], "not_assessable": False},
        )

        assert explanation.finding == (
            "Ranking fairness: 2/2 metrics pass across 2 groups; all pass."
        )
        assert explanation.severity == "info"

    def test_a_healthy_reliability_explanation_still_reports_ece_and_mce(self):
        explanation = build_explanation(
            "reliability_diagram",
            {"verdict_text": "WELL CALIBRATED", "ece": 0.021, "mce": 0.08},
        )

        assert explanation.finding == "Model is WELL CALIBRATED (ECE 0.021, MCE 0.080)."
        assert explanation.severity == "low"
        assert explanation.action == "Recalibrate if ECE exceeds roughly 0.05."

    def test_a_healthy_regression_explanation_still_reports_the_pass_count(self):
        explanation = build_explanation(
            "regression_fairness",
            {"n_pass": 4, "n_total": 4, "badges": [], "groups": [1, 2], "not_assessable": False},
        )

        assert explanation.finding == (
            "Regression fairness: 4/4 disparity metrics pass across 2 groups; all pass."
        )

    def test_a_healthy_report_card_still_says_approved(self):
        explanation = build_explanation(
            "report_card",
            {"approved": True, "model_name": "m", "metrics": [{"passed": True}]},
        )

        assert "approved" in explanation.finding
        assert not _is_could_not_check(explanation.finding)

    def test_a_healthy_hierarchical_gate_still_says_approved(self):
        explanation = build_explanation("hierarchical_gate", {"approved": True, "warnings": []})

        assert "APPROVED" in explanation.finding

    def test_a_healthy_threshold_report_still_says_feasible(self):
        explanation = build_explanation(
            "threshold_optimization_report",
            {
                "is_feasible": True,
                "constraint_type": "demographic_parity",
                "fairness_improvement": {"reduction_pct": 60},
                "accuracy_change": {"change": -0.01},
            },
        )

        assert "is feasible" in explanation.finding

    def test_a_blocked_gate_is_still_reported_as_blocked(self):
        """The other direction of the same control: a real refusal survives."""
        explanation = build_explanation("hierarchical_gate", {"approved": False, "warnings": []})

        assert "BLOCKED" in explanation.finding
        assert explanation.severity == "critical"

    def test_the_ranking_demo_page_is_untouched(self):
        # example=True is now required; the fixture itself is unchanged.
        svg = ranking_fairness_to_svg(example=True)

        assert ">UNFAIR<" in svg
        assert "NDKL" in svg
        assert "2/3" in svg
        assert _NOT_CHECKED_LABEL not in svg


# ---------------------------------------------------------------------------
# House rule: no em dash in text that ships. tracker.py:914 was verdict-adjacent.
# ---------------------------------------------------------------------------


def test_the_weekly_degradation_recommendation_carries_no_em_dash():
    """The banned punctuation is removed without costing the reader the finding."""
    pd = pytest.importorskip("pandas")
    from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

    analyzer = TemporalFairnessAnalyzer(lookback_days=90)
    start = pd.Timestamp("2026-01-05")  # a Monday
    for day in range(28):
        date = start + pd.Timedelta(days=day)
        # A weekday-shaped degradation: one weekday runs far above the mean.
        spike = date.dayofweek == 4
        analyzer.update_daily_metrics(
            date, {"demographic_parity": 0.30 if spike else 0.05 + day * 0.0005}
        )

    degraded, _ = analyzer.detect_weekly_degradation("demographic_parity")
    assert degraded, "the fixture did not produce the weekly degradation being checked"

    report = analyzer.get_explanation("demographic_parity")
    text = " ".join(report.recommendations)

    assert "Weekly degradation detected" in text, text
    assert "—" not in text
    assert "--" not in text
    # Over-correction control: the punctuation went, the finding stayed.
    assert "investigate weekday patterns" in text


def test_no_em_dash_survives_in_the_modules_this_wave_owns():
    """A file-scoped repeat of the package-wide guard, so the fix cannot rot."""
    import ast
    from pathlib import Path

    import vfairness

    root = Path(vfairness.__file__).resolve().parent
    owned = [
        root / "rendering" / "explain.py",
        root / "rendering" / "adapters_ranking.py",
        root / "operations" / "monitoring" / "tracker.py",
    ]
    offenders = []
    for path in owned:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstrings.add(id(value))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            if "—" in node.value:
                offenders.append(f"{path.name}:{node.lineno}: {node.value[:60]}")

    assert not offenders, offenders


def test_a_nan_value_still_fails_closed_on_the_ranking_page():
    """Pinned by test_adapters_ranking_verdicts too; repeated here because the
    three-state badge introduced above must not quietly relax it."""
    svg = ranking_fairness_to_svg(
        [{"metric_name": "exposure_parity", "value": float("nan"), "threshold": 0.1}],
        {"A": {"mean_exposure": 0.5, "count": 5}},
    )

    assert ">UNFAIR<" in svg
    assert ">FAIR<" not in svg
    assert math.isnan(float("nan"))
