"""Behavioural tests for :mod:`vfairness.rendering.adapters_experimentation`.

Closes the coverage half of register finding #22 for the experimentation
adapter, which sat at 23.2 percent of 293 statements.

``experiment_results_to_svg`` carries three defensive blocks with incident
comments attached (a missing CI bound was defaulted to 0, a missing p-value to
1.0, and a missing heterogeneity verdict to False, each of which manufactured a
reassuring badge out of a test that never ran). Those blocks had no test. They
have one now, in both directions: the measured case must show the number, and
the unmeasured case must show could-not-check.

The same hardening had NOT been carried into ``experiment_recommendation_to_svg``
(a recommendation rendered with no experiment result printed "p-value 1.0000"
and a green "Heterogeneity: No"), nor into the temporal-stability caption of
``causal_decomposition_to_svg`` (an absent ``is_stable`` was captioned STABLE in
green). Both were fixed on 2026-08-27 and both are pinned below, in both
directions: the reported verdict must still read exactly as it did, and the
absent one must read could-not-check.
"""

import re
from types import SimpleNamespace

import pytest

from vfairness.rendering import adapters_experimentation
from vfairness.rendering.adapters_experimentation import (
    CI_NOT_COMPUTED,
    COULD_NOT_CHECK_TEXT,
    P_NOT_COMPUTED,
    _finite,
    _interval,
    causal_decomposition_to_svg,
    experiment_recommendation_to_svg,
    experiment_results_to_svg,
    power_analysis_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


MEASURED_RESULT = {
    "overall_effect": 0.0512,
    "overall_ci": (0.0104, 0.0920),
    "overall_p_value": 0.0137,
    "heterogeneity_detected": True,
    "heterogeneity_p_value": 0.0021,
    "n_intersections": 2,
    "metadata": {"protected_attributes": ["gender", "age_band"]},
    "intersection_effects": [
        {
            "intersection": ("female", "under_30"),
            "effect_size_d": 0.31,
            "ci_lower": 0.09,
            "ci_upper": 0.53,
            "p_value": 0.004,
            "n_control": 220,
            "n_treatment": 240,
            "significant": True,
            "powered": True,
        },
        {
            "intersection": ("male", "over_50"),
            "effect_size_d": -0.12,
            "ci_lower": -0.40,
            "ci_upper": 0.16,
            "p_value": 0.41,
            "n_control": 130,
            "n_treatment": 140,
            "significant": False,
            "powered": False,
        },
    ],
}

UNMEASURED_RESULT = {
    "overall_effect": 0.05,
    "intersection_effects": [
        {
            "intersection": ("female", "under_30"),
            "effect_size_d": 0.30,
            "n_control": 100,
            "n_treatment": 100,
        }
    ],
}


def _recommendation_lines(svg):
    """The numbered recommendation lines only.

    The rendering engine also writes an auto-generated `finding` sentence into
    the chart's `<desc>` and metadata block, so a bare substring search over the
    whole document cannot tell a recommendation from that narration.
    """
    return re.findall(r">(\d+\. [^<>]+)<", svg)


def _capture_template_data(monkeypatch, target="render_svg"):
    captured = {}

    def _capture(template, data):
        captured["template"] = template
        captured.update(data)
        return "<svg></svg>"

    monkeypatch.setattr(adapters_experimentation, target, _capture)
    return captured


class TestFiniteAndIntervalHelpers:
    @pytest.mark.parametrize(
        "value", [None, float("nan"), float("inf"), float("-inf"), "abc", object()]
    )
    def test_unusable_values_become_none(self, value):
        assert _finite(value) is None

    @pytest.mark.parametrize(("value", "expected"), [(0, 0.0), ("0.25", 0.25), (-3, -3.0)])
    def test_usable_values_are_coerced_to_float(self, value, expected):
        assert _finite(value) == expected

    def test_half_an_interval_is_not_an_interval(self):
        assert _interval(0.1, None) is None
        assert _interval(None, 0.9) is None
        assert _interval(float("nan"), 0.9) is None

    def test_a_complete_interval_is_returned_as_a_pair(self):
        assert _interval(0.1, 0.9) == (0.1, 0.9)


class TestExperimentResultsShowsWhatWasMeasured:
    def test_a_measured_experiment_prints_its_interval_and_p_value(self):
        svg = experiment_results_to_svg(MEASURED_RESULT)

        assert "95% CI [0.0104, 0.0920]" in svg
        assert "0.0137" in svg
        assert CI_NOT_COMPUTED not in svg
        assert P_NOT_COMPUTED not in svg
        assert COULD_NOT_CHECK_TEXT not in svg

    def test_both_intersections_are_tabulated_with_their_own_intervals(self):
        svg = experiment_results_to_svg(MEASURED_RESULT)

        assert "female × under_30" in svg
        assert "male × over_50" in svg
        assert "[0.0900, 0.5300]" in svg
        assert "[-0.4000, 0.1600]" in svg

    def test_a_detected_heterogeneity_is_reported_with_its_p_value(self):
        svg = experiment_results_to_svg(MEASURED_RESULT)

        assert "0.0021" in svg
        assert COULD_NOT_CHECK_TEXT not in svg

    def test_a_very_small_p_value_is_shown_as_a_bound_not_as_zero(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg({**MEASURED_RESULT, "overall_p_value": 1e-9})

        assert captured["overall_p_display"] == "<0.0001"

    def test_the_protected_attributes_are_named_in_the_header(self):
        svg = experiment_results_to_svg(MEASURED_RESULT)

        assert "gender, age_band" in svg


class TestExperimentResultsRefusesToInventEvidence:
    """Negative cases for the three documented incidents in this adapter."""

    def test_a_missing_confidence_interval_says_so_instead_of_showing_zero_to_zero(self):
        svg = experiment_results_to_svg(UNMEASURED_RESULT)

        assert CI_NOT_COMPUTED in svg
        assert "[0.0000, 0.0000]" not in svg

    def test_a_missing_p_value_says_so_instead_of_reading_as_one(self):
        svg = experiment_results_to_svg(UNMEASURED_RESULT)

        assert P_NOT_COMPUTED in svg
        assert "1.0000" not in svg

    def test_a_heterogeneity_test_that_never_ran_is_could_not_check_not_homogeneous(self):
        svg = experiment_results_to_svg(UNMEASURED_RESULT)

        assert COULD_NOT_CHECK_TEXT in svg
        assert "no heterogeneity test was reported" in svg

    def test_a_reported_homogeneous_verdict_is_distinguished_from_no_verdict(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg({**UNMEASURED_RESULT, "heterogeneity_detected": False})

        assert captured["heterogeneity_known"] is True
        assert captured["heterogeneity_detected"] is False

    def test_an_absent_verdict_is_flagged_as_unknown(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(UNMEASURED_RESULT)

        assert captured["heterogeneity_known"] is False

    def test_a_missing_bound_contributes_nothing_to_the_forest_scale(self, monkeypatch):
        """A defaulted 0 bound used to drag the axis toward the centre line and
        squash the real effects, so the scale is asserted by value."""
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {
                "overall_effect": 0.0,
                "intersection_effects": [
                    {
                        "intersection": ("a",),
                        "effect_size_d": 0.30,
                        "n_control": 1,
                        "n_treatment": 1,
                    }
                ],
            }
        )

        assert captured["forest_abs_max"] == pytest.approx(0.36)

    def test_a_present_interval_does_widen_the_forest_scale(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {
                "overall_effect": 0.0,
                "intersection_effects": [
                    {
                        "intersection": ("a",),
                        "effect_size_d": 0.30,
                        "ci_lower": -0.10,
                        "ci_upper": 0.80,
                        "n_control": 1,
                        "n_treatment": 1,
                    }
                ],
            }
        )

        assert captured["forest_abs_max"] == pytest.approx(0.96)

    def test_significance_is_derived_from_the_p_value_when_it_is_not_supplied(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {
                "overall_effect": 0.0,
                "intersection_effects": [
                    {"intersection": ("a",), "effect_size_d": 0.3, "p_value": 0.01},
                    {"intersection": ("b",), "effect_size_d": 0.3, "p_value": 0.30},
                ],
            }
        )

        assert [e["significant"] for e in captured["effects"]] == [True, False]

    def test_an_effect_without_a_p_value_is_never_called_significant(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {
                "overall_effect": 0.0,
                "intersection_effects": [{"intersection": ("a",), "effect_size_d": 0.9}],
            }
        )

        assert captured["effects"][0]["significant"] is False
        assert captured["effects"][0]["has_p"] is False


class TestExperimentResultsInputForms:
    def test_a_dataclass_result_is_unpacked_field_by_field(self):
        from dataclasses import dataclass, field

        @dataclass
        class ExperimentResult:
            overall_effect: float = 0.0512
            overall_ci: tuple = (0.0104, 0.0920)
            overall_p_value: float = 0.0137
            intersection_effects: list = field(default_factory=list)
            heterogeneity_detected: bool = True
            heterogeneity_p_value: float = 0.0021
            power_results: list = field(default_factory=list)
            n_intersections: int = 0
            n_excluded: int = 0
            design_type: str = "paired"
            metadata: dict = field(default_factory=dict)

        svg = experiment_results_to_svg(ExperimentResult())

        assert "95% CI [0.0104, 0.0920]" in svg
        assert "Paired" in svg

    def test_an_unusable_result_object_renders_a_could_not_check_report(self):
        class Unusable:
            pass

        svg = experiment_results_to_svg(Unusable())

        assert CI_NOT_COMPUTED in svg
        assert COULD_NOT_CHECK_TEXT in svg

    def test_only_twelve_intersections_are_drawn_but_all_are_counted(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {
                "overall_effect": 0.0,
                "n_intersections": 20,
                "intersection_effects": [
                    {"intersection": (f"g{i}",), "effect_size_d": 0.1} for i in range(20)
                ],
            }
        )

        assert len(captured["effects"]) == 12
        assert captured["n_intersections"] == 20

    def test_a_missing_jinja2_is_refused_not_answered_with_an_empty_string(self, monkeypatch):
        """REWRITTEN 2026-08-28. This test PINNED THE DEFECT.

        It asserted ``== ""`` and a warning, which is exactly the behaviour the
        audit flagged: 16 of the 44 public ``*_to_svg`` adapters answered a
        missing backend with an empty string while the other 28 raised. The
        docstrings document the return as "SVG markup string", so a caller doing
        ``open(p, "w").write(svg)`` wrote a zero-byte file and read it as this
        run's report; and the default warning filter shows the message once per
        process, so a batch job warned once and then produced nothing, silently,
        for every report after it. All 44 now raise.
        """
        monkeypatch.setattr(adapters_experimentation, "JINJA2_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"vfairness\[rendering\]"):
            experiment_results_to_svg(MEASURED_RESULT)


class TestPowerAnalysis:
    RESULTS = [
        {
            "intersection": ("female", "under_30"),
            "power": 0.42,
            "required_n": 800,
            "n_control": 100,
            "n_treatment": 100,
            "effect_size": 0.3,
            "is_powered": False,
        },
        {
            "intersection": ("male", "over_50"),
            "power": 0.91,
            "required_n": 300,
            "n_control": 400,
            "n_treatment": 400,
            "effect_size": 0.5,
            "is_powered": True,
        },
    ]

    def test_powered_and_underpowered_intersections_are_counted_separately(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(self.RESULTS)

        assert captured["n_powered"] == 1
        assert captured["n_underpowered"] == 1
        assert captured["n_total"] == 2

    def test_the_average_power_and_sample_totals_are_summed_across_intersections(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(self.RESULTS)

        assert captured["avg_power"] == pytest.approx(0.665)
        assert captured["total_required_n"] == "1,100"
        assert captured["total_current_n"] == "1,000"

    def test_an_underpowered_intersection_is_labelled_on_the_page(self):
        svg = power_analysis_to_svg(self.RESULTS)

        assert "NEEDS DATA" in svg
        assert "female × under_30" in svg
        assert ">42%<" in svg

    def test_power_is_judged_against_the_requested_target_when_the_flag_is_absent(
        self, monkeypatch
    ):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(
            [{"intersection": ("a",), "power": 0.85, "required_n": 10}], power_target=0.90
        )

        assert captured["intersections"][0]["is_powered"] is False

    def test_a_summary_dict_is_accepted_as_well_as_a_list(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg({"results": self.RESULTS})

        assert captured["n_total"] == 2

    def test_an_empty_analysis_reports_zero_rather_than_dividing_by_zero(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg([])

        assert captured["n_total"] == 0
        assert captured["avg_power"] == 0
        assert captured["subtitle"].startswith("0 intersections")

    def test_the_minimum_detectable_effect_is_only_shown_when_supplied(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)
        power_analysis_to_svg(self.RESULTS)
        assert captured["mde_value"] is None

        captured_with = _capture_template_data(monkeypatch)
        power_analysis_to_svg(self.RESULTS, mde=0.25)
        assert captured_with["mde_value"] == "0.2500"


class TestExperimentRecommendation:
    RECOMMENDATION = {
        "decision": "DEPLOY_TREATMENT",
        "confidence": 0.82,
        "reasoning": ["Effect is positive across every intersection"],
        "trade_offs": {"accuracy": "-0.4%"},
        "caveats": ["Only two weeks of data"],
    }

    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            ("DEPLOY_TREATMENT", "Deploy the treatment"),
            ("KEEP_CONTROL", "Keep the control"),
            ("EXTEND_EXPERIMENT", "Extend the experiment"),
            ("INVESTIGATE_FURTHER", "Investigate further"),
        ],
    )
    def test_each_decision_gets_its_own_plain_language_summary(self, decision, expected):
        svg = experiment_recommendation_to_svg({**self.RECOMMENDATION, "decision": decision})

        assert expected in svg

    def test_an_unrecognised_decision_falls_back_to_review_carefully(self):
        svg = experiment_recommendation_to_svg({**self.RECOMMENDATION, "decision": "SHIP_IT"})

        assert "Review the experiment results carefully." in svg

    def test_the_confidence_reasoning_tradeoffs_and_caveats_all_reach_the_page(self):
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION)

        assert ">82%<" in svg
        assert "Effect is positive across every intersection" in svg
        assert "accuracy" in svg and "-0.4%" in svg
        assert "Only two weeks of data" in svg

    def test_an_enum_backed_decision_is_read_through_its_value(self):
        recommendation = SimpleNamespace(
            __dataclass_fields__={},
            decision=SimpleNamespace(value="keep_control"),
            confidence=0.5,
            reasoning=["flat"],
            trade_offs={},
            caveats=[],
        )

        svg = experiment_recommendation_to_svg(recommendation)

        assert "Keep the control" in svg

    def test_experiment_metrics_are_taken_from_the_result_when_one_is_supplied(self):
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION, MEASURED_RESULT)

        assert "+0.0512" in svg
        assert "0.0137" in svg
        assert "1/2" in svg

    def test_a_recommendation_without_an_experiment_result_invents_no_p_value(self):
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION)

        assert "1.0000" not in svg

    def test_a_recommendation_without_an_experiment_result_invents_no_heterogeneity_verdict(
        self,
    ):
        """'No' under Heterogeneity is a positive finding about between-group
        consistency, and it used to be printed in green for an experiment that
        never ran the test."""
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION)

        assert "Heterogeneity" not in svg
        assert "COULD NOT CHECK" in svg

    def test_a_recommendation_without_an_experiment_result_invents_no_effect(self):
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION)

        assert "+0.0000" not in svg
        assert "No experiment result was supplied" in svg

    def test_the_decision_itself_still_renders_without_an_experiment_result(self):
        """The control: the recommendation is the point of this page, so
        suppressing the unmeasured statistics must not blank the decision."""
        svg = experiment_recommendation_to_svg(self.RECOMMENDATION)

        assert "DEPLOY TREATMENT" in svg
        assert "Deploy the treatment" in svg
        assert ">82%<" in svg
        assert "Effect is positive across every intersection" in svg

    def test_a_measured_experiment_still_prints_every_metric(self, monkeypatch):
        """The over-correction control: with a real result attached, nothing is
        hidden behind could-not-check and the values are unchanged."""
        captured = _capture_template_data(monkeypatch)

        experiment_recommendation_to_svg(self.RECOMMENDATION, MEASURED_RESULT)

        assert captured["experiment_metrics_known"] is True
        assert captured["overall_effect_known"] is True
        assert captured["overall_effect"] == "+0.0512"
        assert captured["overall_p_known"] is True
        assert captured["overall_p_display"] == "0.0137"
        assert captured["heterogeneity_known"] is True
        assert captured["heterogeneity_detected"] is True

    def test_a_result_with_an_effect_but_no_test_keeps_the_effect_and_drops_the_test(
        self, monkeypatch
    ):
        """The states are per-field: a measured effect is not suppressed just
        because the significance test is missing."""
        captured = _capture_template_data(monkeypatch)

        experiment_recommendation_to_svg(self.RECOMMENDATION, UNMEASURED_RESULT)

        assert captured["overall_effect"] == "+0.0500"
        assert captured["overall_p_known"] is False
        assert captured["overall_p_display"] == "not computed"
        assert captured["heterogeneity_known"] is False

    def test_a_reported_homogeneous_verdict_is_distinguished_from_no_verdict(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        experiment_recommendation_to_svg(
            self.RECOMMENDATION, {**UNMEASURED_RESULT, "heterogeneity_detected": False}
        )

        assert captured["heterogeneity_known"] is True
        assert captured["heterogeneity_detected"] is False


class TestCausalDecomposition:
    MOSTLY_MEDIATED = {
        "total_effect": 0.20,
        "direct_effect": 0.06,
        "indirect_effect": 0.14,
        "mediator": "education",
        "proportion_mediated": 0.70,
        "steps_satisfied": {
            "step_1": True,
            "step_2": True,
            "step_3": True,
            "step_4": False,
        },
    }

    def test_the_decomposition_is_reported_path_by_path(self):
        svg = causal_decomposition_to_svg(self.MOSTLY_MEDIATED)

        assert "Direct: 0.0600" in svg
        assert "Indirect: 0.1400" in svg
        assert "Total: 0.2000" in svg
        assert ">70%<" in svg

    def test_a_dominant_mediator_earns_the_intervention_recommendation(self):
        svg = causal_decomposition_to_svg(self.MOSTLY_MEDIATED)

        assert any("accounts for 70%" in line for line in _recommendation_lines(svg))

    def test_partial_mediation_gets_the_monitor_both_paths_recommendation(self):
        svg = causal_decomposition_to_svg({**self.MOSTLY_MEDIATED, "proportion_mediated": 0.35})

        lines = _recommendation_lines(svg)
        # The two mediation bands are exclusive, so exactly one of them may fire.
        assert any("Partial mediation (35%)" in line for line in lines)
        assert not any("accounts for" in line for line in lines)

    def test_an_incomplete_baron_kenny_ladder_limits_the_causal_claim(self):
        svg = causal_decomposition_to_svg(self.MOSTLY_MEDIATED)

        assert "Only 3/4 Baron-Kenny steps satisfied" in svg
        assert "3/4 Baron-Kenny steps satisfied" in svg

    def test_an_absent_step_verdict_counts_as_not_satisfied(self):
        """Negative case: a step nobody reported must not be credited."""
        svg = causal_decomposition_to_svg(
            {**self.MOSTLY_MEDIATED, "steps_satisfied": {}, "proportion_mediated": 0.1}
        )

        assert "Only 0/4 Baron-Kenny steps satisfied" in svg
        assert "Causal decomposition is clean" not in svg

    def test_a_complete_clean_decomposition_says_so(self):
        svg = causal_decomposition_to_svg(
            {
                **self.MOSTLY_MEDIATED,
                "proportion_mediated": 0.1,
                "steps_satisfied": {
                    "step_1": True,
                    "step_2": True,
                    "step_3": True,
                    "step_4": True,
                },
            }
        )

        lines = _recommendation_lines(svg)
        assert any("Causal decomposition is clean" in line for line in lines)
        assert not any("Baron-Kenny" in line for line in lines)

    def test_temporal_instability_is_labelled_and_recommended_on(self):
        svg = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED,
            {
                "periods": ["Q1", "Q2", "Q3"],
                "effects_over_time": [0.20, 0.15, 0.05],
                "is_stable": False,
                "trend_slope": -0.07,
            },
        )

        assert ">UNSTABLE<" in svg
        assert ">STABLE<" not in svg
        assert "Temporal instability detected" in svg

    def test_a_reported_stable_series_is_labelled_stable(self):
        svg = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED,
            {
                "periods": ["Q1", "Q2"],
                "effects_over_time": [0.20, 0.21],
                "is_stable": True,
                "trend_slope": 0.001,
            },
        )

        assert ">STABLE<" in svg
        assert "Temporal instability detected" not in svg

    def test_a_dataclass_decomposition_is_unpacked_field_by_field(self):
        from dataclasses import dataclass, field

        @dataclass
        class CausalDecomposition:
            total_effect: float = 0.20
            direct_effect: float = 0.06
            indirect_effect: float = 0.14
            mediator: str = "education"
            proportion_mediated: float = 0.70
            steps_satisfied: dict = field(default_factory=dict)

        svg = causal_decomposition_to_svg(CausalDecomposition())

        assert "Direct: 0.0600" in svg
        assert "Only 0/4 Baron-Kenny steps satisfied" in svg

    def test_a_zero_total_effect_does_not_divide_by_zero(self):
        svg = causal_decomposition_to_svg(
            {"total_effect": 0.0, "direct_effect": 0.0, "indirect_effect": 0.0}
        )

        assert "<svg" in svg
        assert "Total: 0.0000" in svg

    def test_the_demo_path_renders_a_labelled_decomposition(self):
        svg = causal_decomposition_to_svg()

        assert "<svg" in svg
        assert "Causal Decomposition Analysis" in svg

    def test_a_temporal_series_with_no_stability_verdict_is_not_labelled_stable(self):
        svg = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED, {"periods": ["Q1"], "effects_over_time": [0.2]}
        )

        assert ">STABLE<" not in svg

    def test_an_untested_series_says_could_not_check_rather_than_unstable(self):
        """The absent verdict is its own state. Reading it as False would swap
        one manufactured finding for the opposite one."""
        svg = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED, {"periods": ["Q1"], "effects_over_time": [0.2]}
        )

        assert ">COULD NOT CHECK<" in svg
        assert ">UNSTABLE<" not in svg
        assert "Temporal instability detected" not in svg
        assert "No temporal stability verdict was supplied" in svg

    def test_the_untested_trend_is_drawn_in_neither_verdict_colour(self):
        untested = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED, {"periods": ["Q1", "Q2"], "effects_over_time": [0.2, 0.19]}
        )
        stable = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED,
            {"periods": ["Q1", "Q2"], "effects_over_time": [0.2, 0.19], "is_stable": True},
        )
        unstable = causal_decomposition_to_svg(
            self.MOSTLY_MEDIATED,
            {"periods": ["Q1", "Q2"], "effects_over_time": [0.2, 0.19], "is_stable": False},
        )

        def _pill_fill(svg, label):
            found = re.search(r'fill="(#[0-9a-fA-F]{6})">' + label + "<", svg)
            assert found, f"the {label} pill did not render"
            return found.group(1)

        neutral = _pill_fill(untested, "COULD NOT CHECK")
        assert neutral != _pill_fill(stable, "STABLE")
        assert neutral != _pill_fill(unstable, "UNSTABLE")

    def test_a_reported_stable_series_is_unchanged_by_the_third_state(self):
        """The over-correction control: an explicit verdict still reads exactly
        as it did, in its own colour, with its recommendation intact."""
        stable = causal_decomposition_to_svg(
            {
                **self.MOSTLY_MEDIATED,
                "steps_satisfied": dict.fromkeys(["step_1", "step_2", "step_3", "step_4"], True),
            },
            {"periods": ["Q1", "Q2"], "effects_over_time": [0.20, 0.21], "is_stable": True},
        )

        assert ">STABLE<" in stable
        assert ">COULD NOT CHECK<" not in stable
        assert "No temporal stability verdict was supplied" not in stable
        assert any("accounts for 70%" in line for line in _recommendation_lines(stable))
