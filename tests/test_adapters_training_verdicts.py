"""Behavioural tests for :mod:`vfairness.rendering.adapters_training`.

Closes the coverage half of register finding #22 for the training adapter,
which sat at 6.5 percent of 163 statements: the lowest of the seven SVG
adapters the audit named.

The report reader's decision is "which mitigation method should I ship", so the
tests check that a method which did NOT satisfy the constraint can never be
drawn as if it had, in the template path and in the fallback path alike.

The report objects are built as light stand-ins rather than imported from
``in_processing.analyzer``: the adapter reads plain attributes, so a stand-in
exercises exactly the contract the adapter depends on and keeps this file
independent of that module's own refactors.
"""

import re
import warnings
from types import SimpleNamespace

import pytest

from vfairness.rendering import adapters_training
from vfairness.rendering.adapters_training import (
    _axis_ranges,
    _nice_ceil,
    _nice_floor,
    method_comparison_to_svg,
    tradeoff_analysis_to_svg,
    training_analysis_report_to_svg,
    training_report_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


FAIR_METHOD = SimpleNamespace(
    method_name="reweighting",
    accuracy=0.842,
    fairness_violation=0.031,
    constraint_satisfied=True,
    training_time=1.2,
)
UNFAIR_METHOD = SimpleNamespace(
    method_name="adversarial",
    accuracy=0.871,
    fairness_violation=0.180,
    constraint_satisfied=False,
    training_time=9.4,
)

TRADEOFF = {
    "all_results": [
        {"accuracy": 0.84, "violation": 0.03, "lambda": 0.5, "satisfied": True},
        {"accuracy": 0.87, "violation": 0.18, "lambda": 0.1, "satisfied": False},
    ],
    "pareto_frontier": [{"accuracy": 0.84, "violation": 0.03}],
    "best_fair": {"accuracy": 0.84, "violation": 0.03},
    "best_accurate": {"accuracy": 0.87, "violation": 0.18},
}


def _report(**overrides):
    base = dict(
        timestamp="2026-08-27T10:00:00",
        task_type="classification",
        data_info={
            "n_samples": 1000,
            "n_groups": 2,
            "n_features": 12,
            "attribute_name": "gender",
        },
        baseline_metrics={
            "accuracy": 0.88,
            "fairness_violation": 0.22,
            "constraint_satisfied": False,
        },
        method_comparisons=[FAIR_METHOD, UNFAIR_METHOD],
        recommendation=SimpleNamespace(
            recommended_method="reweighting",
            priority="high",
            rationale="Best fairness at a small accuracy cost",
            alternative_methods=["exp_gradient"],
        ),
        critical_issues=["Baseline violates the constraint"],
        action_items=["Retrain with reweighting"],
        tradeoff_analysis=TRADEOFF,
        fairness_analysis={
            "constraint_type": "demographic_parity",
            "base_rate_disparity": 0.19,
            "group_statistics": {
                "male": {"size": 600, "proportion": 0.6, "positive_rate": 0.51},
                "female": {"size": 400, "proportion": 0.4, "positive_rate": 0.32},
            },
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestAxisHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"), [(0.13, 0.15), (0.10, 0.10), (0.0, 0.0), (0.51, 0.55)]
    )
    def test_nice_ceil_rounds_up_to_the_grid(self, value, expected):
        assert _nice_ceil(value) == pytest.approx(expected)

    @pytest.mark.parametrize(("value", "expected"), [(0.13, 0.10), (0.10, 0.10), (0.99, 0.95)])
    def test_nice_floor_rounds_down_to_the_grid(self, value, expected):
        assert _nice_floor(value) == pytest.approx(expected)

    def test_axis_ranges_bracket_the_data_and_start_the_violation_axis_at_zero(self):
        x_min, x_range, y_max = _axis_ranges([0.84, 0.87], [0.03, 0.18])

        assert x_min == pytest.approx(0.80)
        assert x_min + x_range >= 0.87
        assert y_max >= 0.18
        assert y_max == pytest.approx(0.25)

    def test_a_tiny_accuracy_spread_is_widened_to_a_readable_window(self):
        x_min, x_range, _ = _axis_ranges([0.8401, 0.8402], [0.01])

        assert x_range >= 0.10 - 1e-9

    def test_empty_data_falls_back_to_the_full_unit_axes(self):
        x_min, x_range, y_max = _axis_ranges([], [])

        assert (x_min, x_range, y_max) == (0.0, 1.0, 1.0)

    def test_the_violation_axis_never_collapses_below_its_floor(self):
        _, _, y_max = _axis_ranges([0.9], [0.0])

        assert y_max == pytest.approx(0.10)


class TestTrainingReport:
    def test_each_method_is_labelled_by_whether_it_met_the_constraint(self):
        svg = training_report_to_svg(_report())

        assert "reweighting" in svg and "adversarial" in svg
        assert ">PASS<" in svg
        assert ">FAIL<" in svg
        assert ">0.8420<" in svg and ">0.1800<" in svg

    def test_a_violated_baseline_is_labelled_violated(self):
        svg = training_report_to_svg(_report())

        # "SATISFIED" is also a column header in the method table, so the chip
        # is identified by the word that only the baseline verdict can produce.
        assert ">VIOLATED<" in svg

    def test_a_satisfied_baseline_is_labelled_satisfied(self):
        svg = training_report_to_svg(
            _report(
                baseline_metrics={
                    "accuracy": 0.90,
                    "fairness_violation": 0.01,
                    "constraint_satisfied": True,
                }
            )
        )

        assert ">SATISFIED<" in svg
        assert ">VIOLATED<" not in svg

    def test_the_recommendation_and_its_alternatives_reach_the_page(self):
        svg = training_report_to_svg(_report())

        assert "Best fairness at a small accuracy cost" in svg
        assert "Alternatives: exp_gradient" in svg
        assert ">HIGH<" in svg

    def test_missing_data_info_is_shown_as_not_available_not_as_zero(self):
        svg = training_report_to_svg(_report(data_info={}))

        assert ">N/A<" in svg
        assert "sensitive_attr" in svg

    def test_a_string_issue_is_wrapped_rather_than_dropped(self):
        svg = training_report_to_svg(_report(critical_issues=["Baseline violates the constraint"]))

        assert "Baseline violates the constraint" in svg
        # W-19. This asserted ">MEDIUM<" until 2026-09-07, which pinned the defect
        # it was not written for: an issue that arrived as a bare string carries no
        # severity, and stamping it MEDIUM drew it exactly as a REPORTED medium.
        # The test exists to pin the WRAPPING (the issue must survive, not be
        # dropped), so it pins the label the wrapper actually gives it.
        assert ">UNRATED<" in svg
        assert ">MEDIUM<" not in svg, "an ungraded issue must not read as a graded one"

    def test_a_graded_issue_still_draws_its_own_severity(self):
        """Over-correction control for W-19.

        Routing everything to UNRATED would hide real severities, which is the
        same defect pointing the other way.
        """
        svg = training_report_to_svg(
            _report(
                critical_issues=[
                    {"type": "issue", "description": "Reported high", "severity": "high"}
                ]
            )
        )

        assert "Reported high" in svg
        assert ">HIGH<" in svg
        assert ">UNRATED<" not in svg

    def test_a_graded_and_an_ungraded_issue_are_told_apart(self):
        """The case a reader actually meets: both on one canvas."""
        svg = training_report_to_svg(
            _report(
                critical_issues=[
                    "arrived as a bare string",
                    {"type": "issue", "description": "arrived graded", "severity": "high"},
                ]
            )
        )

        assert ">UNRATED<" in svg and ">HIGH<" in svg

    def test_at_most_five_issues_and_five_actions_are_drawn(self):
        svg = training_report_to_svg(
            _report(
                critical_issues=[f"issue{i}" for i in range(8)],
                action_items=[f"action{i}" for i in range(8)],
            )
        )

        assert "issue4" in svg and "issue5" not in svg
        assert "action4" in svg and "action5" not in svg

    def test_the_svg_is_written_to_save_path(self, tmp_path):
        target = tmp_path / "training.svg"

        svg = training_report_to_svg(_report(), save_path=str(target))

        assert target.read_text() == svg


class TestFallbackPathStillTellsTheTruth:
    def test_a_template_failure_falls_back_without_inventing_a_pass(self, monkeypatch):
        """Negative case: the fallback marks the unsatisfied method with a cross,
        so a rendering failure cannot upgrade a failing method to a passing one."""

        def _boom(template, data):
            raise RuntimeError("template exploded")

        monkeypatch.setattr(adapters_training, "render_svg", _boom)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = training_report_to_svg(_report())

        assert any("template exploded" in str(w.message) for w in caught)
        assert "reweighting: Acc=0.842, Viol=0.031 [✓]" in svg
        assert "adversarial: Acc=0.871, Viol=0.180 [✗]" in svg
        assert "adversarial: Acc=0.871, Viol=0.180 [✓]" not in svg

    def test_the_fallback_is_still_a_well_formed_svg_document(self, monkeypatch):
        monkeypatch.setattr(
            adapters_training,
            "render_svg",
            lambda template, data: (_ for _ in ()).throw(RuntimeError("nope")),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            svg = training_analysis_report_to_svg(_report())

        assert svg.lstrip().startswith("<?xml")
        assert svg.rstrip().endswith("</svg>")
        assert "Generated by vfairness" in svg


class TestMethodComparison:
    def test_methods_are_charted_with_their_status(self):
        svg = method_comparison_to_svg([FAIR_METHOD, UNFAIR_METHOD])

        assert "2 methods evaluated" in svg
        assert ">0.842<" in svg and ">0.180<" in svg
        assert ">PASS<" in svg and ">FAIL<" in svg

    def test_an_empty_comparison_list_produces_a_could_not_check_chart(self, tmp_path):
        """Superseded 2026-08-27. This used to assert the return value was "".

        Returning "" is not "producing nothing": a caller passing ``save_path``
        wrote a 0-BYTE .svg to disk and got no exception, so the pipeline
        reported success and someone later opened a broken image with nothing
        anywhere to say why. Three states, never two, and a silent empty
        artifact is not one of them. The full guard, including the negative
        cases and the healthy-input controls, lives in
        ``tests/test_adapters_zero_and_empty.py``.
        """
        target = tmp_path / "empty.svg"
        svg = method_comparison_to_svg([], save_path=str(target))

        assert svg.rstrip().endswith("</svg>")
        assert "NOT CHECKED" in svg
        assert ">PASS<" not in svg and ">FAIL<" not in svg
        assert "methods evaluated" not in svg
        assert target.stat().st_size > 0

    def test_a_missing_jinja2_is_refused_not_answered_with_an_empty_string(self, monkeypatch):
        """REWRITTEN 2026-08-28. This test PINNED THE DEFECT.

        It asserted ``svg == ""`` plus a warning: the empty-string answer the
        audit flagged. An empty string is not the "SVG markup string" this
        function documents, a caller writing it to disk gets a zero-byte report,
        and the warning is shown once per process, so every later report in a
        batch failed in silence. The other 28 adapters already raised; all 44 do
        now.
        """
        monkeypatch.setattr(adapters_training, "JINJA2_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"vfairness\[rendering\]"):
            method_comparison_to_svg([FAIR_METHOD])


class TestTradeoffScatter:
    def test_the_axis_window_covers_every_plotted_configuration(self):
        svg = tradeoff_analysis_to_svg(TRADEOFF)

        # Axis tick labels are drawn from x_min upwards in 0.02 steps.
        assert ">0.80<" in svg
        assert ">0.90<" in svg
        assert "Fairness Violation" in svg

    def test_points_are_normalised_into_the_drawing_area(self, monkeypatch):
        captured = {}

        def _capture(template, data):
            captured.update(data)
            return "<svg></svg>"

        monkeypatch.setattr(adapters_training, "render_svg", _capture)

        tradeoff_analysis_to_svg(TRADEOFF)

        assert len(captured["points"]) == 2
        for point in captured["points"] + captured["pareto_points"]:
            assert 0.0 <= point["x"] <= 1.0
            assert 0.0 <= point["y"] <= 1.0
        # The satisfied flag must survive the transform: it is what colours the dot.
        assert [p["satisfied"] for p in captured["points"]] == [True, False]

    def test_an_empty_tradeoff_still_renders_a_labelled_empty_chart(self):
        svg = tradeoff_analysis_to_svg({})

        assert "Accuracy-Fairness Trade-off" in svg
        assert "<svg" in svg


class TestFullAnalysisReport:
    def test_group_statistics_are_tabulated_with_their_positive_rates(self):
        svg = training_analysis_report_to_svg(_report())

        assert ">male<" in svg and ">female<" in svg
        assert "51.0%" in svg and "32.0%" in svg
        assert "Base rate disparity: 0.190" in svg

    def test_the_constraint_under_test_is_named_in_the_header(self):
        svg = training_analysis_report_to_svg(_report())

        assert "demographic_parity" in svg

    def test_a_report_with_no_recommendation_object_says_not_available(self):
        svg = training_analysis_report_to_svg(_report(recommendation=None))

        assert ">N/A<" in svg
        assert "<svg" in svg

    def test_a_report_with_nothing_measured_still_renders_without_claiming_a_result(self):
        empty = _report(
            method_comparisons=[],
            tradeoff_analysis={},
            fairness_analysis={},
            critical_issues=[],
            action_items=[],
        )

        svg = training_analysis_report_to_svg(empty)

        # No method row may be drawn, so no method accuracy and no per-method
        # PASS badge can appear.
        assert ">0.8420<" not in svg
        assert re.search(r">PASS<", svg) is None
        assert ">male<" not in svg
