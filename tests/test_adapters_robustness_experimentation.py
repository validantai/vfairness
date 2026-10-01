"""Nothing failed and nothing ran must never render the same.

Wave 9 of the invented-analysis sweep, covering the robustness adapter and the
two remaining experimentation fabrications. Four defects are pinned here, each
in BOTH directions: the absence must read as an absence, and the measurement
must still read exactly as it did.

TIER A, the runtime demo fixtures. ``robustness_testing_to_svg(None)`` returned
the ``_demo`` fixture: verdict ROBUST, score 0.72, three named permutation tests
with p-values, three sensitivity bars, a six-subgroup audit.
``causal_decomposition_to_svg(None)`` returned ``_demo_causal``: "Partial
mediation (38%) through 'score_calibration'", direct 0.0280 / indirect 0.0170 /
total 0.0450, four of four Baron-Kenny steps satisfied. Both were reached as a
RUNTIME FALLBACK, so a caller could not tell the output from a real evaluation,
and an SVG is an export format: it leaves the building and an auditor reads it.

TIER B, the sentinels and the falsy tests. ``robustness_testing_to_svg([], [],
None)`` printed a banded MARGINAL with "score: 0.50" over "0 permutation tests"
and a Recommendations panel reading "All robustness checks passed": a midpoint
sentinel presented as a measurement. ``causal_decomposition_to_svg({})`` printed
0.0000 effects and four crosses, so a decomposition that never ran read as a
definite negative result. ``experiment_recommendation_to_svg({})`` printed the
headline INVESTIGATE FURTHER at CONFIDENCE 0% over a metrics panel that said
COULD NOT CHECK: one canvas, two contradictory states. ``power_analysis_to_svg``
treated a measured effect size of exactly 0.0 as absent through a falsy check.

The over-correction control is the class ``TestHealthyInputIsUnchanged`` at the
bottom: every healthy render in it is byte-identical to the same render against
commit 92f7bd2, verified outside pytest by rendering both trees.
"""

import re

import pytest

from vfairness.rendering.adapters_experimentation import (
    P_NOT_COMPUTED,
    causal_decomposition_to_svg,
    experiment_recommendation_to_svg,
    power_analysis_to_svg,
)
from vfairness.rendering.adapters_robustness import robustness_testing_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


COULD_NOT_CHECK = "COULD NOT CHECK"
EXAMPLE_BAND = "EXAMPLE ONLY: SYNTHETIC DEMONSTRATION DATA, NOT A REAL EVALUATION"
EXAMPLE_DESC = "EXAMPLE ONLY, synthetic demonstration data, not a real evaluation."

# The three verdict colours. A could-not-check canvas may use none of them for a
# headline, because a reader takes the grade off the colour before the words.
GREEN, AMBER, RED = "#059669", "#f59e0b", "#dc2626"

PERM_TESTS = [
    {
        "method": "demographic_parity",
        "observed_statistic": 0.042,
        "p_value": 0.312,
        "significant_at_05": False,
        "effect_direction": "positive",
    },
    {
        "method": "equalized_odds",
        "observed_statistic": 0.087,
        "p_value": 0.003,
        "significant_at_05": True,
        "effect_direction": "negative",
    },
]
SENSITIVITY = [
    {
        "perturbation_type": "label_noise_5pct",
        "robustness_score": 0.92,
        "is_robust": True,
        "max_deviation": 0.008,
    },
    {
        "perturbation_type": "feature_dropout_10pct",
        "robustness_score": 0.74,
        "is_robust": False,
        "max_deviation": 0.031,
    },
]
ALL_ROBUST = [
    {
        "perturbation_type": "label_noise_5pct",
        "robustness_score": 0.95,
        "is_robust": True,
        "max_deviation": 0.004,
    }
]
AUDIT = {
    "n_subgroups_analyzed": 6,
    "n_subgroups_flagged": 2,
    "worst_subgroup": "age_under_25 and female",
    "worst_disparity": 0.142,
    "gerrymandering_detected": False,
    "flagged_subgroups": [("age_under_25 and female", 0.142)],
}

MEDIATED = {
    "total_effect": 0.20,
    "direct_effect": 0.06,
    "indirect_effect": 0.14,
    "mediator": "education",
    "proportion_mediated": 0.70,
    "steps_satisfied": {"step_1": True, "step_2": True, "step_3": True, "step_4": True},
}
TEMPORAL = {
    "periods": ["Q1", "Q2", "Q3"],
    "effects_over_time": [0.2, 0.19, 0.21],
    "is_stable": True,
    "trend_slope": 0.001,
}

RECOMMENDATION = {
    "decision": "DEPLOY_TREATMENT",
    "confidence": 0.82,
    "reasoning": ["Effect is positive across every intersection"],
    "trade_offs": {"accuracy": "-0.4%"},
    "caveats": ["Only two weeks of data"],
}


def _desc(svg: str) -> str:
    """The accessible description, which is the whole artifact for a reader who
    cannot see the canvas and for anything that parses the SVG."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.DOTALL)
    return match.group(1) if match else ""


class TestRobustnessWithNoInputInventsNothing:
    """TIER A: the ``_demo`` fixture was the runtime fallback for ``None``."""

    def test_no_input_does_not_return_the_demo_fixture(self):
        svg = robustness_testing_to_svg(None)

        assert "0.72" not in svg
        assert "demographic_parity" not in svg
        assert "label_noise" not in svg
        assert "age<25" not in svg

    def test_no_input_renders_no_verdict_and_no_score(self):
        svg = robustness_testing_to_svg(None)

        assert ">ROBUST<" not in svg
        assert ">MARGINAL<" not in svg
        assert ">FAIL<" not in svg
        assert "score:" not in svg
        assert ">NOT SCORED<" in svg

    def test_no_input_says_so_where_a_reader_sees_it(self):
        svg = robustness_testing_to_svg(None)

        assert "Nothing was tested" in svg
        assert ">NOT CHECKED<" in svg

    def test_no_input_is_not_a_rejection_either(self):
        """Could-not-check is the third state, not a failure. A reader must not
        be able to conclude the model is fragile from a run that tested it."""
        svg = robustness_testing_to_svg(None)

        assert "This is not a pass and it is not a failure." in svg
        assert GREEN not in svg
        assert RED not in svg

    def test_the_accessible_description_certifies_nothing(self):
        desc = _desc(robustness_testing_to_svg(None))

        assert desc.startswith("Robustness Testing: " + COULD_NOT_CHECK)
        assert "MARGINAL" not in desc
        assert "0.72" not in desc

    def test_the_demo_survives_behind_an_explicit_example_flag(self):
        svg = robustness_testing_to_svg(example=True)

        assert "0.72" in svg
        assert "demographic_parity" in svg

    def test_an_example_render_is_watermarked_on_the_canvas_and_in_the_description(self):
        svg = robustness_testing_to_svg(example=True)

        assert EXAMPLE_BAND in svg
        assert _desc(svg).startswith("Robustness Testing: " + EXAMPLE_DESC)

    def test_a_real_render_carries_no_example_watermark(self):
        assert EXAMPLE_BAND not in robustness_testing_to_svg(PERM_TESTS, SENSITIVITY, AUDIT)


class TestRobustnessWithZeroChecksScoresNothing:
    """TIER B: 0.50 was the neutral SENTINEL for the no-scored-test branch, and
    the template graded it like a measurement."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: robustness_testing_to_svg([], [], None),
            lambda: robustness_testing_to_svg([], [], {}),
            lambda: robustness_testing_to_svg([], []),
        ],
        ids=["empty-lists", "empty-audit-dict", "two-empty-lists"],
    )
    def test_zero_checks_print_no_score_and_no_verdict(self, call):
        svg = call()

        assert "0.50" not in svg
        assert ">MARGINAL<" not in svg
        assert "score:" not in svg

    def test_zero_checks_do_not_report_that_every_check_passed(self):
        svg = robustness_testing_to_svg([], [], None)

        assert "All robustness checks passed" not in svg

    def test_zero_checks_state_the_absence_on_the_canvas(self):
        svg = robustness_testing_to_svg([], [], None)

        assert "Nothing was tested" in svg
        assert _desc(svg).startswith("Robustness Testing: " + COULD_NOT_CHECK)

    def test_zero_checks_render_no_empty_table_shell(self):
        """A column heading over no rows is a table that ran, reporting nothing."""
        svg = robustness_testing_to_svg([], [], None)

        assert ">METHOD<" not in svg
        assert ">P-VALUE<" not in svg
        assert ">PERTURBATION TYPE<" not in svg


class TestRobustnessWithOnlyASubgroupAudit:
    """An audit with no scored test is a partial run: the audit is real, the
    score is not. The old code split the difference at 0.50 and banded it."""

    def test_the_headline_is_not_scored_rather_than_marginal(self):
        svg = robustness_testing_to_svg([], [], AUDIT)

        assert ">NOT SCORED<" in svg
        assert ">MARGINAL<" not in svg
        assert "0.50" not in svg

    def test_the_supplied_audit_is_still_rendered_in_full(self):
        """The control for this branch: withholding the score must not withhold
        the numbers the caller did measure."""
        svg = robustness_testing_to_svg([], [], AUDIT)

        assert "Subgroup Audit" in svg
        assert "age_under_25 and female" in svg
        assert "0.142" in svg

    def test_the_canvas_says_why_there_is_no_score(self):
        svg = robustness_testing_to_svg([], [], AUDIT)

        assert "no robustness score was computed" in svg
        assert "the audit is not graded" in svg

    def test_the_description_reports_no_score_of_zero(self):
        desc = _desc(robustness_testing_to_svg([], [], AUDIT))

        assert desc.startswith("Robustness Testing: " + COULD_NOT_CHECK)
        assert "score 0.00" not in desc


class TestCausalDecompositionWithNoInputInventsNothing:
    """TIER A: ``_demo_causal`` was the runtime fallback for ``None``."""

    def test_no_input_does_not_return_the_demo_fixture(self):
        svg = causal_decomposition_to_svg()

        assert "score_calibration" not in svg
        assert "Partial mediation (38%)" not in svg
        assert "0.0280" not in svg
        assert "0.0170" not in svg
        assert "0.0450" not in svg

    def test_no_input_renders_no_effect_and_no_baron_kenny_verdict(self):
        svg = causal_decomposition_to_svg()

        assert "Direct:" not in svg
        assert "Indirect:" not in svg
        assert "Total:" not in svg
        assert "Baron-Kenny steps satisfied" not in svg
        assert "&#x2713;" not in svg
        assert "&#x2717;" not in svg

    def test_no_input_says_so_where_a_reader_sees_it(self):
        svg = causal_decomposition_to_svg()

        assert "Nothing was decomposed" in svg
        assert ">NOT CHECKED<" in svg

    def test_no_input_is_not_a_finding_of_no_mediation(self):
        svg = causal_decomposition_to_svg()

        assert "This is not a finding of no mediation: no mediation was tested." in svg

    def test_the_demo_survives_behind_an_explicit_example_flag(self):
        svg = causal_decomposition_to_svg(example=True)

        assert "score_calibration" in svg
        assert "Partial mediation (38%)" in svg

    def test_an_example_render_is_watermarked_on_the_canvas_and_in_the_description(self):
        svg = causal_decomposition_to_svg(example=True)

        assert EXAMPLE_BAND in svg
        assert _desc(svg).startswith("Causal Decomposition: " + EXAMPLE_DESC)


class TestCausalDecompositionWithAnEmptyResult:
    """TIER B: an absence rendered as a definite negative result."""

    @pytest.mark.parametrize(
        "supplied",
        [{}, {"steps_satisfied": {}}, "not a decomposition"],
        ids=["dict", "steps", "junk"],
    )
    def test_an_empty_decomposition_prints_no_zero_effects(self, supplied):
        svg = causal_decomposition_to_svg(supplied)

        assert "0.0000" not in svg
        assert "Direct:" not in svg

    def test_an_empty_decomposition_renders_no_failed_baron_kenny_steps(self):
        svg = causal_decomposition_to_svg({})

        assert "0/4" not in svg
        assert "&#x2717;" not in svg
        assert "Only 0/4 Baron-Kenny steps satisfied" not in svg

    def test_an_empty_decomposition_reports_no_mediated_proportion(self):
        svg = causal_decomposition_to_svg({})

        assert ">0%<" not in svg
        assert "Proportion Mediated" not in svg

    def test_the_description_makes_no_causal_claim(self):
        desc = _desc(causal_decomposition_to_svg({}))

        assert desc.startswith("Causal Decomposition: " + COULD_NOT_CHECK)
        assert "accounts for" not in desc

    def test_a_supplied_temporal_series_is_still_drawn(self):
        """The control for this branch: the decomposition is absent, but the
        caller measured the series, and dropping it would be its own falsehood."""
        svg = causal_decomposition_to_svg({}, TEMPORAL)

        assert "Temporal Stability" in svg
        assert ">STABLE<" in svg


class TestAZeroEffectIsAMeasurement:
    """The mirror rule. Suppressing a real zero swaps one false claim for the
    other, so every zero below must survive exactly as it is."""

    def test_a_decomposition_of_exact_zeros_still_prints_them(self):
        svg = causal_decomposition_to_svg(
            {"total_effect": 0.0, "direct_effect": 0.0, "indirect_effect": 0.0}
        )

        assert "Total: 0.0000" in svg
        assert "Direct: 0.0000" in svg
        assert "Nothing was decomposed" not in svg

    def test_a_decomposition_with_only_a_zero_proportion_is_still_a_measurement(self):
        svg = causal_decomposition_to_svg({"proportion_mediated": 0.0})

        assert "Proportion Mediated" in svg
        assert "Nothing was decomposed" not in svg

    def test_a_measured_effect_size_of_zero_is_not_reported_as_not_computed(self):
        svg = power_analysis_to_svg(
            [
                {
                    "intersection": ("female", "under_30"),
                    "power": 0.42,
                    "required_n": 800,
                    "n_control": 100,
                    "n_treatment": 100,
                    "effect_size": 0.0,
                    "is_powered": False,
                }
            ]
        )

        assert ">0.000<" in svg
        assert P_NOT_COMPUTED not in svg

    def test_a_missing_effect_size_is_still_reported_as_not_computed(self):
        svg = power_analysis_to_svg(
            [{"intersection": ("female",), "power": 0.42, "required_n": 800}]
        )

        assert P_NOT_COMPUTED in svg


class TestExperimentRecommendationWithNoDecision:
    """TIER B: one canvas, two contradictory states. The metrics panel said
    COULD NOT CHECK while the headline recommended a course of action."""

    def test_an_empty_recommendation_makes_no_recommendation(self):
        svg = experiment_recommendation_to_svg({})

        assert "INVESTIGATE FURTHER" not in svg
        assert "heterogeneous effects or anomalies detected" not in svg
        assert ">NO RECOMMENDATION MADE<" in svg

    def test_an_empty_recommendation_reports_no_confidence(self):
        svg = experiment_recommendation_to_svg({})

        assert ">0%<" not in svg
        assert ">not reported<" in svg

    def test_the_headline_and_the_metrics_panel_agree(self):
        svg = experiment_recommendation_to_svg({})

        assert COULD_NOT_CHECK in svg
        assert "recommends nothing" in svg

    def test_the_description_carries_no_decision(self):
        desc = _desc(experiment_recommendation_to_svg({}))

        assert desc.startswith("Experiment Recommendation: " + COULD_NOT_CHECK)
        assert "Investigate Further" not in desc

    def test_a_decision_with_no_confidence_keeps_the_decision(self):
        """The states are per-field: an unreported confidence must not blank the
        decision the caller did make."""
        svg = experiment_recommendation_to_svg({"decision": "KEEP_CONTROL"})

        assert "KEEP CONTROL" in svg
        assert "Keep the control" in svg
        assert ">not reported<" in svg

    def test_a_reported_zero_confidence_is_still_printed(self):
        svg = experiment_recommendation_to_svg({"decision": "KEEP_CONTROL", "confidence": 0.0})

        assert ">0%<" in svg
        assert ">not reported<" not in svg


class TestHealthyInputIsUnchanged:
    """The over-correction control.

    Every render below is byte-identical to the same render against commit
    92f7bd2, checked outside pytest by rendering both trees into files and
    diffing them. These assertions pin the parts a future edit is most likely to
    over-suppress: the verdict, the score, the effects and the all-clear.
    """

    def test_a_scored_robustness_run_still_reports_its_verdict(self):
        svg = robustness_testing_to_svg(PERM_TESTS, SENSITIVITY, AUDIT)

        assert ">MARGINAL<" in svg
        assert "score:" in svg
        assert ">0.70</tspan>" in svg
        assert ">SIGNIF.<" in svg
        assert ">STABLE<" in svg
        assert "Nothing was tested" not in svg

    def test_a_clean_robustness_run_still_reports_the_all_clear(self):
        svg = robustness_testing_to_svg([], ALL_ROBUST, None)

        assert "All robustness checks passed" in svg
        assert ">PASS<" in svg

    def test_a_permutation_only_run_is_still_scored(self):
        svg = robustness_testing_to_svg(PERM_TESTS, None, None)

        assert "score:" in svg
        assert ">NOT SCORED<" not in svg

    def test_a_measured_decomposition_still_reports_every_effect(self):
        svg = causal_decomposition_to_svg(MEDIATED, TEMPORAL)

        assert "Direct: 0.0600" in svg
        assert "Indirect: 0.1400" in svg
        assert "Total: 0.2000" in svg
        assert "4/4 Baron-Kenny steps satisfied" in svg
        assert "education" in svg
        assert ">STABLE<" in svg

    def test_a_measured_recommendation_still_reports_its_decision(self):
        svg = experiment_recommendation_to_svg(RECOMMENDATION)

        assert "DEPLOY TREATMENT" in svg
        assert "Deploy the treatment" in svg
        assert ">82%<" in svg
        assert "Effect is positive across every intersection" in svg
        assert ">NO RECOMMENDATION MADE<" not in svg

    def test_a_measured_power_analysis_still_reports_its_effect_sizes(self):
        svg = power_analysis_to_svg(
            [
                {
                    "intersection": ("male", "over_50"),
                    "power": 0.91,
                    "required_n": 300,
                    "n_control": 400,
                    "n_treatment": 400,
                    "effect_size": 0.5,
                    "is_powered": True,
                }
            ]
        )

        assert ">0.500<" in svg
        assert ">91%<" in svg
