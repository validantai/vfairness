"""The defaulted-measurement pattern in the experimentation and calibration adapters.

Scoped by FILE rather than by site: every test here covers a `.get(key, <number
or bool>)` whose default was rendered as a measurement, graded, coloured,
counted or plotted by
:mod:`vfairness.rendering.adapters_experimentation` or
:mod:`vfairness.rendering.adapters_calibration`.

Each guard is paired with a CONTROL asserting that healthy input still renders
exactly as it did, because every one of these guards can be "passed" by
suppressing the number in all cases, and a suppressed measurement is the mirror
fabrication of an invented one. A SUPPLIED ZERO is a measurement throughout: an
indirect effect of exactly 0.0, a proportion mediated of 0%, a power of 0 and a
sample count of 0 all keep rendering as themselves.
"""

import re

import pytest

from vfairness.rendering import (
    calibration_disparity_to_svg,
    causal_decomposition_to_svg,
    experiment_recommendation_to_svg,
    experiment_results_to_svg,
)

# The two verdict glyphs the Baron-Kenny panel draws, and the neutral third.
TICK = "&#x2713;"
CROSS = "&#x2717;"
NEUTRAL = "&#xb7;"

# A canvas is not allowed to state its ungraded count only at the foot of the
# page: a reader takes the headline first. Everything at or above this y is the
# headline band.
HEADLINE_BAND_Y = 200

FULL_DECOMPOSITION = {
    "total_effect": 0.20,
    "direct_effect": 0.06,
    "indirect_effect": 0.14,
    "mediator": "education",
    "proportion_mediated": 0.70,
    "steps_satisfied": {"step_1": True, "step_2": True, "step_3": True, "step_4": False},
}

FULL_EXPERIMENT = {
    "overall_effect": 0.0342,
    "overall_p_value": 0.065,
    "heterogeneity_detected": True,
    "n_intersections": 2,
    "design_type": "independent",
    "intersection_effects": [
        {"intersection": ("F", "Young"), "powered": True, "n_control": 200, "n_treatment": 210},
        {"intersection": ("M", "Old"), "powered": False, "n_control": 220, "n_treatment": 215},
    ],
}


def _headline_text(svg: str) -> str:
    """Every <text> the chart draws at or above HEADLINE_BAND_Y, concatenated."""
    parts = []
    for m in re.finditer(r'<text[^>]*\by="(-?\d+(?:\.\d+)?)"[^>]*>(.*?)</text>', svg, re.S):
        if float(m.group(1)) <= HEADLINE_BAND_Y:
            parts.append(re.sub(r"<[^>]+>", "", m.group(2)))
    return " ".join(parts)


def _recommendation_lines(svg: str) -> list:
    """The numbered rows of the Recommendations panel.

    Matched on the "N. " prefix rather than on a fill colour: the render engine
    remaps the palette, so a hard-coded #6b7280 silently matches nothing and
    every assertion over the result passes vacuously.
    """
    return re.findall(r'<text[^>]*font-size="9\.5"[^>]*>\d+\. ([^<]*)</text>', svg)


class TestBaronKennyStepsAreNotFailedByDefault:
    """`steps.get(key, False)`: a step nobody evaluated drew a red cross.

    A red cross is the strongest claim the panel can make about a path, namely
    that the mediation ladder was climbed here and it broke.
    """

    HALF_LADDER = {**FULL_DECOMPOSITION, "steps_satisfied": {"step_1": True, "step_2": True}}

    def test_an_unevaluated_step_draws_no_red_cross(self):
        svg = causal_decomposition_to_svg(self.HALF_LADDER)

        assert svg.count(CROSS) == 0
        assert svg.count(TICK) == 2
        assert svg.count(NEUTRAL) == 2
        assert svg.count(">not reported<") == 2

    def test_the_ungraded_step_count_reaches_the_headline_band(self):
        """Rule (c): stated where a reader meets it, not only at the foot."""
        svg = causal_decomposition_to_svg(self.HALF_LADDER)

        assert "2 steps not reported" in _headline_text(svg)

    def test_the_canvas_says_the_steps_were_not_tested_rather_than_failed(self):
        svg = causal_decomposition_to_svg(self.HALF_LADDER)

        assert any("not tested rather than failed" in line for line in _recommendation_lines(svg))

    def test_a_step_reported_false_still_draws_a_red_cross(self):
        """CONTROL, the other direction. Suppressing every cross would pass the
        test above while erasing a real negative finding."""
        svg = causal_decomposition_to_svg(FULL_DECOMPOSITION)

        assert svg.count(CROSS) == 1
        assert svg.count(TICK) == 3
        assert svg.count(NEUTRAL) == 0
        assert "not reported" not in _headline_text(svg)

    def test_an_unreported_step_is_never_credited_as_satisfied(self):
        """The numerator keeps its old meaning: only a reported True counts."""
        svg = causal_decomposition_to_svg(self.HALF_LADDER)

        assert "2/4 Baron-Kenny steps satisfied" in svg
        assert "Causal decomposition is clean" not in svg


class TestCausalEffectsAreGradedPerField:
    """`float(dec.get(key, 0))`: the canvas-level guard cannot see a partial row.

    A decomposition reporting only a total printed "Direct: 0.0000" and
    "Indirect: 0.0000" under EFFECT DECOMPOSITION and drew a 0% mediation gauge.
    """

    TOTAL_ONLY = {"total_effect": 0.045, "mediator": "m"}

    def test_an_unreported_effect_is_not_printed_as_zero(self):
        svg = causal_decomposition_to_svg(self.TOTAL_ONLY)

        assert "Direct: 0.0000" not in svg
        assert "Indirect: 0.0000" not in svg
        assert "Direct: not reported" in svg
        assert "Indirect: not reported" in svg
        # The one effect that WAS reported survives untouched.
        assert "Total: 0.0450" in svg

    def test_an_unreported_proportion_mediated_draws_no_gauge_reading(self):
        svg = causal_decomposition_to_svg(self.TOTAL_ONLY)

        assert ">0%<" not in svg
        assert "no proportion mediated was reported" in svg

    def test_a_supplied_zero_effect_still_renders_as_a_measurement(self):
        """CONTROL. An indirect effect of exactly 0.0 is the finding 'this
        mediator carries none of the gap', and withholding it would swap one
        false claim for another."""
        svg = causal_decomposition_to_svg(
            {
                "total_effect": 0.0,
                "direct_effect": 0.0,
                "indirect_effect": 0.0,
                "proportion_mediated": 0.0,
                "mediator": "m",
                "steps_satisfied": {
                    "step_1": False,
                    "step_2": False,
                    "step_3": False,
                    "step_4": False,
                },
            }
        )

        assert "Total: 0.0000" in svg
        assert "Direct: 0.0000" in svg
        assert "Indirect: 0.0000" in svg
        assert ">0%<" in svg
        # Not one of the four measured quantities is withheld.
        assert "not reported" not in svg

    def test_a_fully_reported_decomposition_is_untouched_by_the_guard(self):
        """CONTROL, the healthy render."""
        svg = causal_decomposition_to_svg(FULL_DECOMPOSITION)

        assert "Direct: 0.0600" in svg
        assert "Indirect: 0.1400" in svg
        assert "Total: 0.2000" in svg
        assert ">70%<" in svg
        assert "not reported" not in svg

    def test_the_all_clear_is_never_unqualified_over_ungraded_effects(self):
        """Rule (b). All four steps can pass on a decomposition that estimated
        nothing, and 'Causal decomposition is clean' would then be a verdict on
        numbers nobody produced."""
        svg = causal_decomposition_to_svg(
            {
                "mediator": "m",
                "steps_satisfied": dict.fromkeys(["step_1", "step_2", "step_3", "step_4"], True),
            }
        )

        assert "Causal decomposition is clean" not in svg
        assert "incomplete, not clean" in svg

    def test_every_recommendation_fits_inside_the_panel(self):
        """The recommendation rows are drawn at x=56 in a 644-wide panel with no
        wrap and no truncate filter. The first draft of the line above ran off
        the right edge of the canvas, and only rasterising showed it: a
        half-rendered statement of absence is not a statement."""
        for dec in (
            {"mediator": "m", "steps_satisfied": dict.fromkeys(["step_1", "step_2"], True)},
            {"total_effect": 0.045, "mediator": "m"},
            FULL_DECOMPOSITION,
            {},
        ):
            for line in _recommendation_lines(causal_decomposition_to_svg(dec)):
                assert len(line) <= 130, f"{len(line)} chars will clip: {line!r}"


class TestPartialRowsNeverCrashOrBlankTheChart:
    """`float(None)` raised TypeError outside the try that wraps the render, so
    a partial row cost the caller the whole chart."""

    @pytest.mark.parametrize(
        "dec,temporal",
        [
            ({"direct_effect": 0.06, "total_effect": None}, None),
            ({"total_effect": "n/a", "direct_effect": "n/a", "mediator": "m"}, None),
            ({"total_effect": float("nan"), "mediator": "m"}, None),
            (
                {"total_effect": 0.2},
                {
                    "periods": ["Q1"],
                    "effects_over_time": [0.1],
                    "is_stable": True,
                    "trend_slope": None,
                },
            ),
        ],
    )
    def test_a_partial_row_still_produces_a_chart(self, dec, temporal):
        svg = causal_decomposition_to_svg(dec, temporal)

        assert svg, "returned an empty string instead of a chart"
        assert "<svg" in svg

    def test_an_absent_excluded_count_does_not_blank_the_experiment_chart(self):
        """`None > 0` raises inside Jinja, and the adapter's except swallowed it
        into an empty string."""
        svg = experiment_results_to_svg({"overall_effect": 0.03, "n_excluded": None})

        assert svg
        assert "<svg" in svg


class TestTemporalTrendSlope:
    """`float(td.get("trend_slope", 0))`: a flat trend is a positive finding
    about temporal consistency, and it was printed for a series carrying no
    slope."""

    SERIES = {"periods": ["Q1", "Q2"], "effects_over_time": [0.2, 0.2], "is_stable": True}

    def test_an_unreported_slope_is_not_printed_as_zero(self):
        svg = causal_decomposition_to_svg(FULL_DECOMPOSITION, self.SERIES)

        assert "trend slope: 0.0000" not in svg
        assert "trend slope: not reported" in svg

    def test_a_supplied_zero_slope_still_renders_as_a_measurement(self):
        """CONTROL. A measured slope of exactly 0.0 is a real flat trend."""
        svg = causal_decomposition_to_svg(FULL_DECOMPOSITION, {**self.SERIES, "trend_slope": 0.0})

        assert "trend slope: 0.0000" in svg

    def test_a_supplied_slope_is_untouched(self):
        """CONTROL."""
        svg = causal_decomposition_to_svg(
            FULL_DECOMPOSITION, {**self.SERIES, "trend_slope": -0.0700}
        )

        assert "trend slope: -0.0700" in svg


class TestRecommendationExperimentCounts:
    """The block that sat directly beside the hardened one: `.get("powered",
    False)`, `.get("n_control", 0)`, `.get("n_treatment", 0)` and
    `.get("n_intersections", 0)` were still counted off the same rows."""

    REC = {"decision": "KEEP_CONTROL", "confidence": 0.8}

    def test_an_experiment_result_with_no_rows_reports_no_power_fraction(self):
        svg = experiment_recommendation_to_svg(self.REC, {"overall_effect": 0.03})

        assert "0/0" not in svg
        assert "not graded" in svg

    def test_arms_nobody_counted_are_not_two_empty_arms(self):
        svg = experiment_recommendation_to_svg(self.REC, {"overall_effect": 0.03})

        assert "0 control" not in svg
        assert "control samples not reported" in svg
        assert "treatment samples not reported" in svg

    def test_rows_that_reported_no_arm_count_do_not_sum_to_zero_samples(self):
        """The PER ROW case, which the test above cannot reach: with no rows at
        all the loop never runs, so it exercises the initial state and not the
        guard inside the loop. Sabotaging that guard left the test above green,
        which is how this gap was found. Here the rows exist and report a power
        verdict and nothing else."""
        svg = experiment_recommendation_to_svg(
            self.REC,
            {
                "overall_effect": 0.03,
                "n_intersections": 2,
                "intersection_effects": [
                    {"intersection": ("A",), "powered": True},
                    {"intersection": ("B",), "powered": False},
                ],
            },
        )

        assert "0 control" not in svg
        assert "0 treatment samples" not in svg
        assert "control samples not reported" in svg
        assert "treatment samples not reported" in svg
        # The verdict the rows DID report is still counted.
        assert "1/2" in svg

    def test_a_design_nobody_reported_is_not_named(self):
        svg = experiment_recommendation_to_svg(self.REC, {"overall_effect": 0.03})

        assert "Independent" not in svg
        assert "intersection count not reported" in svg

    def test_the_power_denominator_is_the_graded_population(self):
        """ "1/2 powered" over a row carrying no power verdict says that second
        subgroup was sized and found short."""
        svg = experiment_recommendation_to_svg(
            self.REC,
            {
                "overall_effect": 0.03,
                "n_intersections": 2,
                "intersection_effects": [
                    {"intersection": ("A",), "powered": True},
                    {"intersection": ("B",)},
                ],
            },
        )

        assert "1/2" not in svg
        assert "1/1" in svg
        assert "1 intersection not graded for power" in svg

    def test_a_supplied_zero_arm_count_still_renders(self):
        """CONTROL. A reported zero is a measurement: that arm was empty."""
        svg = experiment_recommendation_to_svg(
            self.REC,
            {
                "overall_effect": 0.03,
                "n_intersections": 1,
                "intersection_effects": [
                    {"intersection": ("A",), "powered": False, "n_control": 0, "n_treatment": 0}
                ],
            },
        )

        assert "0 control · 0 treatment samples" in svg
        assert "0/1" in svg

    def test_a_complete_experiment_result_is_untouched_by_the_guard(self):
        """CONTROL, the healthy render: the fraction, the arms and the design
        name are exactly what they were before the guard existed."""
        svg = experiment_recommendation_to_svg(self.REC, FULL_EXPERIMENT)

        assert "1/2" in svg
        assert "420 control · 425 treatment samples" in svg
        assert "Independent design · 2 intersections" in svg
        assert "not reported" not in svg
        assert "not graded" not in svg


class TestCalibrationDisparityVerdict:
    """`disparity_result.get("has_significant_disparity", False)`: the
    recommendation panel keys on this flag directly, so an absent one fell
    through to the green all-clear."""

    MEASURED = {"Group A": {"ece": 0.035}, "Group B": {"ece": 0.072}, "Group C": {"ece": 0.045}}

    def test_an_absent_verdict_is_not_an_all_clear(self):
        """The gallery fixture is exactly this shape, and the canvas
        contradicted itself: a 0.037 spread in the alarm band at the top, an
        unqualified all-clear underneath."""
        svg = calibration_disparity_to_svg({"group_metrics": self.MEASURED})

        assert "NO SIGNIFICANT DISPARITY" not in svg
        assert "All groups calibrated within acceptable bounds" not in svg
        assert "DISPARITY VERDICT NOT SUPPLIED" in svg
        # The measured spread survives; only the grading of it is withdrawn.
        assert "Spread measured at 0.037 across 3 group(s), not graded." in svg

    def test_an_absent_verdict_is_not_a_fabricated_breach_either(self):
        """Both directions. Grading the spread against a threshold the caller
        never chose would trade one fabrication for the opposite one."""
        svg = calibration_disparity_to_svg({"group_metrics": self.MEASURED})

        assert ">SIGNIFICANT DISPARITY<" not in svg

    def test_a_reported_false_verdict_still_renders_the_all_clear(self):
        """CONTROL. A reported False is a verdict."""
        svg = calibration_disparity_to_svg(
            {"group_metrics": self.MEASURED, "has_significant_disparity": False}
        )

        assert "NO SIGNIFICANT DISPARITY" in svg
        assert "All groups calibrated within acceptable bounds." in svg
        assert "DISPARITY VERDICT NOT SUPPLIED" not in svg

    def test_a_reported_true_verdict_still_renders_the_warning(self):
        """CONTROL, the other direction."""
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.02}, "B": {"ece": 0.19}},
                "has_significant_disparity": True,
                "recommendation": "Recalibrate group B",
                "strategy": "platt",
            }
        )

        assert ">SIGNIFICANT DISPARITY<" in svg
        assert "NO SIGNIFICANT DISPARITY" not in svg


class TestCalibrationDisparityAllClearIsQualified:
    """Rule (b): never an unqualified all-clear while anything is ungraded.

    `assessable` was true on a COUNT alone (two measured groups), so two
    measured groups plus one silent one certified every group.
    """

    PARTLY_MEASURED = {
        "group_metrics": {"Group A": {"ece": 0.035}, "Group B": {"ece": 0.072}, "Group C": {}},
        "has_significant_disparity": False,
    }

    def test_a_silent_group_is_not_covered_by_the_all_clear(self):
        svg = calibration_disparity_to_svg(self.PARTLY_MEASURED)

        assert "All groups calibrated within acceptable bounds" not in svg
        assert "The 2 measured group(s) are calibrated within acceptable bounds." in svg
        assert "were not covered by this verdict" in svg

    def test_the_ungraded_group_count_reaches_the_headline_band(self):
        """Rule (c). It was stated only in the panel at the foot of the canvas,
        and only on two of that panel's branches."""
        headline = _headline_text(calibration_disparity_to_svg(self.PARTLY_MEASURED))

        assert "1 of 3 group(s) ungraded" in headline
        assert "reported no calibration error" in headline

    def test_a_fully_measured_run_carries_no_ungraded_notice(self):
        """CONTROL: the healthy canvas keeps its ordinary subtitle and says
        nothing about coverage, because there is nothing to say."""
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.01}, "B": {"ece": 0.02}, "C": {"ece": 0.015}},
                "has_significant_disparity": False,
                "overall_ece": 0.015,
            }
        )
        headline = _headline_text(svg)

        assert "Compares calibration error across demographic groups." in headline
        assert "ungraded" not in headline
        assert "COULD NOT CHECK" not in svg
        assert "All groups calibrated within acceptable bounds." in svg
