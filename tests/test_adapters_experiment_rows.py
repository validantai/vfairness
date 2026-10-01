"""Per-ROW and per-CELL guards for the two experimentation and post-processing
canvases that still fabricated at that level.

Defect 1, the last confirmed fabrication in this library. The chart-level state
of ``experiment_results_to_svg`` was fixed in an earlier wave, but the
PER-INTERSECTION row was not. An intersection dict that reported nothing
rendered "+0.0000" in the EFFECT column, "0" under N, a full UNDERPOWERED badge
under STATUS, and was PLOTTED as a point on the forest plot's zero line, which
is the single most specific claim that plot can make about a subgroup. It was
then folded into "1 / 2 intersections adequately powered", so the canvas also
said the row had been graded for power and found short. A row that reported
nothing is not a row of zeros, and it is certainly not a row measured as
underpowered.

Defect 2, library defaults presented as the caller's own configuration.
``threshold_optimization_to_svg({})`` printed CONSTRAINT demographic_parity,
TOLERANCE 0.05 and OBJECTIVE accuracy beside panels that honestly read NOT
MEASURED and NOT ESTABLISHED, so a reader took the three cells for a record of
the run. It was never only the empty dict: ``ThresholdResult.to_dict()`` carries
none of those keys, so a REAL optimisation configured with equalized_odds at
tolerance 0.10 rendered "demographic_parity" and "0.05" as its own setup. The
same run took ORIG RATE 50.0%, OPT RATE 50.0% and SIZE 0 per group row from the
same kind of default.

Every test here is paired with a positive control, because a guard that cannot
tell a measured zero from an absence has swapped one false claim for another: a
supplied effect of +0.0000, a supplied sample count of 0, a supplied
``powered: False`` and a supplied group size of 0 are all measurements and must
still render as numbers.
"""

import re

import pytest

from vfairness.rendering.adapters_experimentation import (
    COULD_NOT_CHECK_TEXT,
    P_NOT_COMPUTED,
    experiment_results_to_svg,
)
from vfairness.rendering.adapters_post_processing import threshold_optimization_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


# ── fixtures ────────────────────────────────────────────────────────────────

MEASURED_ROW = {
    "intersection": ("Female", "40-59"),
    "effect_size_d": 0.31,
    "ci_lower": 0.11,
    "ci_upper": 0.51,
    "p_value": 0.004,
    "n_control": 300,
    "n_treatment": 310,
    "significant": True,
    "powered": True,
}

#: The row at the centre of defect 1: an intersection that carried its name and
#: nothing else. It is a shape the analysis really produces, for a cell too
#: small to estimate anything in.
SILENT_ROW = {"intersection": ("Non-binary", "18-24")}

#: A MEASURED null row. The experiment ran on this subgroup, moved it by exactly
#: nothing, and found it underpowered on real (tiny) arms. Every number here is
#: a finding and must survive the fix untouched.
MEASURED_ZERO_ROW = {
    "intersection": ("Other", "60+"),
    "effect_size_d": 0.0,
    "ci_lower": 0.0,
    "ci_upper": 0.0,
    "p_value": 0.99,
    "n_control": 0,
    "n_treatment": 0,
    "significant": False,
    "powered": False,
}


def _experiment(rows, **extra):
    data = {
        "overall_effect": 0.042,
        "overall_ci": [0.01, 0.07],
        "overall_p_value": 0.003,
        "heterogeneity_detected": False,
        "heterogeneity_p_value": 0.42,
        "design_type": "independent",
        "intersection_effects": list(rows),
    }
    data.update(extra)
    return data


def _table_row(svg, name):
    """The markup of one table row, from its name cell to the next row's.

    Bounded by the effects table, never running on into the forest plot: the
    plot's "0" axis label would otherwise satisfy an assertion about the N cell.
    """
    table = svg[svg.index("INTERSECTION") : svg.index("FOREST PLOT")]
    start = table.index(f">{name}<")
    # Row name cells are the only text anchored at x="56" after the header.
    nxt = table.find('<text x="56" ', start + 1)
    return table[start : nxt if nxt > 0 else len(table)]


def _forest_points(svg):
    """The x coordinates of the plotted point estimates."""
    plot = svg[svg.index("FOREST PLOT") :]
    return [int(x) for x in re.findall(r'<circle cx="(\d+)"', plot)]


# ── defect 1: the per-intersection row ──────────────────────────────────────


class TestSilentRowIsNotAZeroRow:
    def test_a_row_that_reported_nothing_prints_no_effect(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ROW, SILENT_ROW]))

        row = _table_row(svg, "Non-binary × 18-24")
        assert "+0.0000" not in row, "an effect nobody estimated was printed as exactly zero"
        assert P_NOT_COMPUTED in row

    def test_a_row_that_reported_nothing_is_not_badged_underpowered(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ROW, SILENT_ROW]))

        row = _table_row(svg, "Non-binary × 18-24")
        assert "UNDERPOWERED" not in row, (
            "a subgroup that reported no power verdict was called underpowered, "
            "which says it was sized and found short of data"
        )
        assert COULD_NOT_CHECK_TEXT in row

    def test_a_row_that_reported_no_arm_is_not_a_sample_of_zero(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ROW, SILENT_ROW]))

        row = _table_row(svg, "Non-binary × 18-24")
        assert ">0<" not in row, "an unreported sample size was printed as a measured 0"
        assert "not reported" in row

    def test_a_row_that_reported_nothing_is_not_plotted_on_the_zero_line(self):
        """The forest plot's centre line is the no-difference position, so a
        marker drawn there states an effect of exactly zero."""
        one_row = experiment_results_to_svg(_experiment([MEASURED_ROW]))
        with_silent = experiment_results_to_svg(_experiment([MEASURED_ROW, SILENT_ROW]))

        assert len(_forest_points(with_silent)) == 1
        assert _forest_points(with_silent) == _forest_points(one_row)
        assert "1 row not plotted: no effect estimate was reported" in with_silent

    def test_a_row_that_reported_nothing_is_not_counted_as_graded_for_power(self):
        """The denominator of a power verdict is the graded population. "1 / 2
        adequately powered" over an ungraded row says that row is not."""
        svg = experiment_results_to_svg(_experiment([MEASURED_ROW, SILENT_ROW], n_intersections=2))

        assert "1 / 2" not in svg
        assert "1 / 1" in svg
        assert "1 further intersection not graded: no power verdict" in svg

    def test_a_row_that_reported_nothing_does_not_total_the_arms_to_zero(self):
        """Only the rows that reported an arm are summed, and a canvas where no
        row reported one says so rather than printing two empty arms."""
        svg = experiment_results_to_svg(_experiment([SILENT_ROW]))

        metadata = svg[svg.index("N CONTROL") :]
        assert ">0<" not in metadata
        assert metadata.count("not reported") >= 2

    def test_a_canvas_whose_only_row_reported_nothing_is_a_could_not_check(self):
        svg = experiment_results_to_svg({"intersection_effects": [SILENT_ROW]})

        assert "No effect estimate and no statistical test were supplied" in svg
        assert "+0.0000" not in svg


class TestMeasuredRowsAreUntouched:
    """Positive controls: a supplied zero is a measurement, in every cell."""

    def test_a_measured_zero_effect_still_prints_as_a_number(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ZERO_ROW]))

        row = _table_row(svg, "Other × 60+")
        assert "+0.0000" in row
        assert P_NOT_COMPUTED not in row

    def test_a_measured_zero_sample_and_a_measured_underpowered_verdict_survive(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ZERO_ROW]))

        row = _table_row(svg, "Other × 60+")
        assert ">0<" in row, "a reported sample size of 0 was withheld as an absence"
        assert "UNDERPOWERED" in row, "a reported powered=False verdict was withheld"

    def test_a_measured_row_is_still_plotted_and_still_counted(self):
        svg = experiment_results_to_svg(
            _experiment([MEASURED_ROW, MEASURED_ZERO_ROW], n_intersections=2)
        )

        assert len(_forest_points(svg)) == 2
        assert "not plotted" not in svg
        assert "1 / 2" in svg
        assert "not graded" not in svg

    def test_a_fully_reported_canvas_carries_no_could_not_check_row(self):
        svg = experiment_results_to_svg(_experiment([MEASURED_ROW, MEASURED_ZERO_ROW]))

        table = svg[svg.index("INTERSECTION") : svg.index("FOREST PLOT")]
        assert COULD_NOT_CHECK_TEXT not in table
        assert "not reported" not in table


# ── defect 2: library defaults as the caller's configuration ────────────────


FULL_THRESHOLD_RESULT = {
    "constraint_type": "equalized_odds",
    "tolerance": 0.10,
    "objective": "f1",
    "groups": [
        {
            "name": "Male",
            "threshold": 0.55,
            "original_rate": 0.72,
            "optimized_rate": 0.63,
            "size": 2750,
        }
    ],
    "original_disparity": 0.14,
    "optimized_disparity": 0.01,
    "original_accuracy": 0.852,
    "optimized_accuracy": 0.841,
    "is_feasible": True,
}


class TestThresholdConfigurationIsNotInvented:
    def test_an_empty_result_states_no_constraint_tolerance_or_objective(self):
        svg = threshold_optimization_to_svg({})

        assert "demographic_parity" not in svg, "a constraint nobody chose was named as the run's"
        assert ">0.05<" not in svg
        assert ">accuracy<" not in svg
        assert svg.count("not supplied") >= 3

    def test_the_canvas_says_the_configuration_was_not_supplied(self):
        """Not only in the accessible description: a reader of the canvas has to
        be told, next to the cells that would otherwise read as a record."""
        svg = threshold_optimization_to_svg({})

        assert "no constraint, tolerance or objective was supplied" in svg

    def test_a_real_result_object_carrying_only_thresholds_invents_no_setup(self):
        """``ThresholdResult.to_dict()`` carries no constraint, tolerance or
        objective, so this is what every real optimiser run rendered."""

        class RealResult:
            def to_dict(self):
                return {
                    "global_threshold": None,
                    "group_thresholds": {"A": 0.73, "B": 0.49},
                    "is_feasible": True,
                }

        svg = threshold_optimization_to_svg(RealResult())

        assert "demographic_parity" not in svg
        assert ">0.05<" not in svg
        # The thresholds it DID report are still on the canvas.
        assert "0.730" in svg and "0.490" in svg

    def test_a_group_row_with_no_rate_or_size_is_not_a_half_rate_of_no_members(self):
        class RealResult:
            def to_dict(self):
                return {"group_thresholds": {"A": 0.73}, "is_feasible": True}

        svg = threshold_optimization_to_svg(RealResult())

        assert "50.0%" not in svg, "0.5 is the library's placeholder, not a measured rate"
        assert "N/A" in svg
        assert "not reported" in svg


class TestThresholdConfigurationPositiveControl:
    def test_a_supplied_configuration_prints_exactly_what_was_supplied(self):
        svg = threshold_optimization_to_svg(FULL_THRESHOLD_RESULT)

        assert ">equalized_odds<" in svg
        assert ">0.10<" in svg
        assert ">f1<" in svg
        assert "not supplied" not in svg
        assert "no constraint, tolerance or objective was supplied" not in svg

    def test_a_supplied_tolerance_of_zero_is_a_configuration_not_an_absence(self):
        svg = threshold_optimization_to_svg({**FULL_THRESHOLD_RESULT, "tolerance": 0.0})

        assert ">0.00<" in svg
        assert "not supplied" not in svg

    def test_a_supplied_group_size_of_zero_still_prints_as_zero(self):
        svg = threshold_optimization_to_svg(
            {
                "constraint_type": "demographic_parity",
                "group_thresholds": {"A": 0.5},
                "original_rates": {"A": 0.0},
                "optimized_rates": {"A": 0.0},
                "group_sizes": {"A": 0},
                "is_feasible": True,
            }
        )

        assert "0.0%" in svg
        assert ">0<" in svg
        assert "not reported" not in svg
