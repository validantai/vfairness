"""Verdict tests for :mod:`vfairness.rendering.adapters_calibration`.

Closes the coverage half of register finding #22 for the calibration adapter,
which sat at 10.7 percent of 279 statements: the four public renderers were
never executed by any test, only imported.

Assertions are made against the RENDERED SVG, never against the module's colour
constants. The rendering engine applies a palette transform, so a source hex is
a proxy for what the reader sees; the visible words are not. This follows the
lesson recorded when register finding #11 was closed in the sibling monitoring
adapter.
"""

import re

import numpy as np
import pytest

from vfairness.rendering.adapters_calibration import (
    calibration_disparity_to_svg,
    group_calibration_to_svg,
    pareto_frontier_to_svg,
    reliability_diagram_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

# Four bands, because there are three states and the measured one splits into
# three severities: assessed (Well Calibrated / Moderate / Poorly Calibrated)
# and could-not-check (NOT ASSESSABLE). The helper below still insists that ONE
# of them is on the canvas, so a blank badge fails the same as a wrong one.
_VERDICT = re.compile(r">(Well Calibrated|Moderate|Poorly Calibrated|NOT ASSESSABLE)<")

# Spelled out rather than imported from the adapter: what is being pinned is
# the words a reader sees on the exported canvas, and an assertion against the
# module's own constant would pass even if the constant were renamed to
# something meaningless.
NOT_ASSESSABLE = "NOT ASSESSABLE"
_ECE = re.compile(r">ECE = ([\d.]+)<")


def _verdict(svg: str) -> str:
    found = _VERDICT.findall(svg)
    assert found, "no calibration verdict was rendered at all"
    return found[0]


def _printed_ece(svg: str) -> float:
    found = _ECE.findall(svg)
    assert found, "no ECE value was rendered"
    return float(found[0])


class TestReliabilityDiagramVerdict:
    """The verdict band is what a reader takes away, so each band is pinned."""

    def test_a_perfectly_calibrated_model_reads_as_well_calibrated(self):
        # Half the subjects are positive and every score is 0.5, so the single
        # occupied bin has confidence 0.5 and accuracy 0.5: ECE is exactly 0.
        labels = np.array([1] * 50 + [0] * 50)
        probs = np.full(100, 0.5)

        svg = reliability_diagram_to_svg(labels, probs)

        assert _verdict(svg) == "Well Calibrated"
        assert _printed_ece(svg) == pytest.approx(0.0, abs=1e-3)

    def test_a_moderately_miscalibrated_model_reads_as_moderate(self):
        # One bin: confidence 0.60, accuracy 0.53, so ECE is exactly 0.07,
        # which sits in the [0.05, 0.10) band.
        labels = np.array([1] * 53 + [0] * 47)
        probs = np.full(100, 0.6)

        svg = reliability_diagram_to_svg(labels, probs)

        assert _verdict(svg) == "Moderate"
        assert _printed_ece(svg) == pytest.approx(0.07, abs=1e-3)

    def test_a_maximal_violation_is_never_rendered_as_calibrated(self):
        """Negative case: a model that is confidently wrong every single time."""
        labels = np.array([0] * 50 + [1] * 50)
        probs = np.array([0.99] * 50 + [0.01] * 50)

        svg = reliability_diagram_to_svg(labels, probs)

        assert _verdict(svg) == "Poorly Calibrated"
        assert _printed_ece(svg) == pytest.approx(0.99, abs=1e-3)
        assert "Well Calibrated" not in svg

    def test_the_printed_ece_is_the_binned_expectation_not_a_placeholder(self):
        rng = np.random.default_rng(11)
        probs = rng.uniform(0.0, 1.0, 400)
        labels = (rng.uniform(size=400) < probs).astype(int)

        svg = reliability_diagram_to_svg(labels, probs, n_bins=10)

        # Independent recomputation of the same definition.
        edges = np.linspace(0, 1, 11)
        expected = 0.0
        for i in range(10):
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
            if i == 9:
                mask = mask | (probs == edges[i + 1])
            if mask.sum():
                expected += mask.sum() / 400 * abs(labels[mask].mean() - probs[mask].mean())

        assert _printed_ece(svg) == pytest.approx(expected, abs=5e-4)
        assert expected > 0.0, "fixture degenerated to a trivial zero ECE"

    def test_the_bin_count_actually_changes_the_measurement(self):
        rng = np.random.default_rng(3)
        probs = rng.uniform(0.0, 1.0, 300)
        labels = (rng.uniform(size=300) < probs**2).astype(int)

        coarse = _printed_ece(reliability_diagram_to_svg(labels, probs, n_bins=2))
        fine = _printed_ece(reliability_diagram_to_svg(labels, probs, n_bins=20))

        assert coarse != fine

    def test_the_svg_is_written_to_save_path(self, tmp_path):
        target = tmp_path / "reliability.svg"

        svg = reliability_diagram_to_svg(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), save_path=str(target)
        )

        assert target.read_text() == svg
        assert svg.lstrip().startswith("<svg") or "<svg" in svg[:400]

    def test_no_data_is_not_rendered_as_a_calibrated_model(self):
        """Was an OPEN DEFECT until 2026-08-27: no data entered no bin, ECE
        stayed at its 0.0 seed and the badge read 'Well Calibrated, ECE =
        0.000'. Nothing was measured, so the state is could-not-check."""
        svg = reliability_diagram_to_svg([], [])

        assert _verdict(svg) != "Well Calibrated"
        assert _verdict(svg) == "NOT ASSESSABLE"
        # No number may be printed at all: a 0.000 reads as a measured zero.
        assert "ECE = 0.000" not in svg
        assert ">0.000<" not in svg
        assert ">N/A<" in svg
        assert "no data was supplied" in svg

    def test_unmeasurable_probabilities_are_not_rendered_as_calibrated(self):
        """Was an OPEN DEFECT until 2026-08-27: every comparison against NaN is
        False, so all-NaN probabilities entered no bin, ECE stayed 0.0 and the
        page reported 'Well Calibrated'. The NaN-reads-as-green hole closed in
        adapters_monitoring for register finding #11."""
        svg = reliability_diagram_to_svg(np.array([0, 1, 0, 1]), np.full(4, np.nan))

        assert _verdict(svg) != "Well Calibrated"
        assert _verdict(svg) == "NOT ASSESSABLE"
        assert "ECE = 0.000" not in svg
        assert ">0.000<" not in svg
        assert "no probability could be binned" in svg

    def test_unmeasurable_labels_are_not_rendered_as_calibrated(self):
        """The mirror case: the probabilities bin, but the outcome they are
        compared against is NaN, so the gap is undefined. It used to raise
        'cannot convert float NaN to integer' from inside the curve builder."""
        svg = reliability_diagram_to_svg(
            np.array([0.0, np.nan, 1.0, 1.0]), np.array([0.2, 0.3, 0.8, 0.9])
        )

        assert _verdict(svg) == "NOT ASSESSABLE"
        assert "ECE = " not in svg

    def test_a_partly_unusable_input_is_measured_on_what_survived_and_says_so(self):
        """Rows that enter no bin are excluded from the weighting, not counted
        as agreeing with the model: dividing by the supplied row count shrinks
        ECE toward the green band in proportion to how much data was unusable.
        """
        labels = np.array([1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, np.nan, np.nan, 0.85, 0.05])

        svg = reliability_diagram_to_svg(labels, probs)

        # Four rows survive: gaps 0.10, 0.15 and 0.075 twice, over four rows.
        assert _printed_ece(svg) == pytest.approx((0.10 + 0.15 + 2 * 0.075) / 4, abs=5e-4)
        assert "2 of 6 prediction(s) could not be binned" in svg
        assert _verdict(svg) != "NOT ASSESSABLE"

    def test_ordinary_input_is_untouched_by_the_could_not_check_guard(self):
        """Over-correction control. A library that renders could-not-check for
        everything passes every failure test above and is useless."""
        # Ten bins, each holding 100 subjects whose observed rate is exactly
        # the confidence they were given: honest data, no sampling noise.
        probs, labels = [], []
        for i in range(10):
            p = 0.05 + 0.1 * i
            k = round(p * 100)
            probs.extend([p] * 100)
            labels.extend([1] * k + [0] * (100 - k))
        probs, labels = np.array(probs), np.array(labels)

        svg = reliability_diagram_to_svg(labels, probs)

        assert _verdict(svg) == "Well Calibrated"
        assert "NOT ASSESSABLE" not in svg
        assert "could not be binned" not in svg
        assert "Points on the diagonal indicate perfect calibration." in svg
        assert ">N/A<" not in svg


class TestGroupCalibration:
    def test_the_worst_calibrated_group_is_named(self):
        rng = np.random.default_rng(1)
        groups = np.array(["A"] * 300 + ["B"] * 300)
        probs = rng.uniform(0.05, 0.95, 600)
        labels = np.empty(600, dtype=int)
        labels[:300] = (rng.uniform(size=300) < probs[:300]).astype(int)
        labels[300:] = 1  # group B is always positive whatever it was told

        svg = group_calibration_to_svg(labels, probs, groups)

        assert "Calibration Disparity Detected" in svg
        assert "Best group: A" in svg
        assert "Worst group: B" in svg

    def test_two_identically_calibrated_groups_show_no_disparity(self):
        groups = np.array(["A"] * 50 + ["B"] * 50)
        labels = np.array(([1] * 25 + [0] * 25) * 2)
        probs = np.full(100, 0.5)

        svg = group_calibration_to_svg(labels, probs, groups)

        assert "Well Calibrated Across Groups" in svg
        assert "Calibration Disparity Detected" not in svg

    def test_every_present_group_is_counted_in_the_summary(self):
        groups = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        probs = np.full(60, 0.5)
        labels = np.array(([1] * 10 + [0] * 10) * 3)

        svg = group_calibration_to_svg(labels, probs, groups)

        assert "3 groups analyzed" in svg

    def test_unmeasurable_probabilities_are_not_rendered_as_calibrated_groups(self):
        """Was an OPEN DEFECT until 2026-08-27, and the same hole wave 7 closed
        one function up: every comparison against NaN is False, so no row
        entered a bin, every group ECE stayed at its 0.0 seed, the spread came
        out 0.000 and the panel certified 'Well Calibrated Across Groups'."""
        svg = group_calibration_to_svg(
            np.array([0, 1, 0, 1]), np.full(4, np.nan), np.array(["m", "m", "f", "f"])
        )

        assert "Well Calibrated Across Groups" not in svg
        assert "Calibration Disparity Detected" not in svg
        assert NOT_ASSESSABLE in svg
        # No number may be printed at all: a 0.000 reads as a measured zero.
        assert ">0.000<" not in svg
        assert "ECE = 0.000" not in svg
        assert ">N/A<" in svg
        assert "calibration error could be measured" in svg

    def test_no_groups_at_all_is_not_rendered_as_calibrated_groups(self):
        """A scan of zero groups compared nothing, so it certifies nothing."""
        svg = group_calibration_to_svg(np.array([]), np.array([]), np.array([]))

        assert "Well Calibrated Across Groups" not in svg
        assert NOT_ASSESSABLE in svg
        assert ">0.000<" not in svg
        assert "no group was supplied" in svg

    def test_one_measured_group_is_not_a_comparison(self):
        """A spread between groups needs two measured groups to exist. One
        group is not agreement, and it is not the best of anything either."""
        svg = group_calibration_to_svg(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.5, 0.5, np.nan, np.nan]),
            np.array(["a", "a", "a", "a", "b", "b"]),
        )

        assert "Well Calibrated Across Groups" not in svg
        assert NOT_ASSESSABLE in svg
        assert "only one group" in svg
        # The one group that WAS measured still reports its own number: the
        # guard withdraws the comparison, not the measurement.
        assert "ECE = 0.000" in svg
        assert "1 of 2 group(s) had a measurable calibration error." in svg

    def test_an_unmeasurable_group_is_excluded_and_the_exclusion_is_visible(self):
        """The mixed case, which is the one an auditor is most likely to be
        handed: two honest groups and one whose scores are all NaN. The two
        keep their verdict, the third reads N/A, and the canvas says so."""
        probs, labels, attrs = [], [], []
        for group in ("A", "C"):
            for i in range(10):
                p = 0.05 + 0.1 * i
                k = round(p * 100)
                probs.extend([p] * 100)
                labels.extend([1] * k + [0] * (100 - k))
                attrs.extend([group] * 100)
        probs.extend([np.nan] * 20)
        labels.extend([1] * 10 + [0] * 10)
        attrs.extend(["B"] * 20)

        svg = group_calibration_to_svg(np.array(labels), np.array(probs), np.array(attrs))

        assert "Well Calibrated Across Groups" in svg
        assert "ECE = N/A" in svg
        assert "1 of 3 group(s) could not be measured and are excluded." in svg

    def test_a_group_ece_is_weighted_by_the_rows_that_binned(self):
        """Dividing a group's error by its SUPPLIED size instead of its binned
        size shrinks that group's ECE toward the green band in proportion to
        how much of its data was unusable, which is backwards."""
        # Group A: four usable rows with gaps 0.10, 0.15, 0.075, 0.075, plus
        # two rows that enter no bin at all.
        labels = np.array([1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.85, 0.05, np.nan, np.nan])
        attrs = np.array(["A"] * 6)

        svg = group_calibration_to_svg(labels, probs, attrs)

        expected = (0.10 + 0.15 + 2 * 0.075) / 4
        assert f"ECE = {expected:.3f}" in svg
        # Weighting by the six supplied rows would print this smaller number.
        assert f"ECE = {(0.10 + 0.15 + 2 * 0.075) / 6:.3f}" not in svg

    def test_ordinary_multi_group_input_is_untouched_by_the_guard(self):
        """Over-correction control. A library that renders could-not-check for
        everything passes every failure test above and is useless."""
        probs, labels, attrs = [], [], []
        for group in ("north", "south", "east"):
            for i in range(10):
                p = 0.05 + 0.1 * i
                k = round(p * 100)
                probs.extend([p] * 100)
                labels.extend([1] * k + [0] * (100 - k))
                attrs.extend([group] * 100)

        svg = group_calibration_to_svg(np.array(labels), np.array(probs), np.array(attrs))

        assert "Well Calibrated Across Groups" in svg
        assert NOT_ASSESSABLE not in svg
        assert "N/A" not in svg
        assert "could not be measured" not in svg
        assert "Each curve shows calibration for one demographic group." in svg
        assert "3 groups analyzed" in svg


class TestCalibrationDisparity:
    def test_a_flagged_disparity_names_the_worst_group_and_the_remedy(self):
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.02}, "B": {"ece": 0.19}},
                "has_significant_disparity": True,
                "overall_ece": 0.10,
                "strategy": "platt",
                "recommendation": "Recalibrate group B",
            }
        )

        assert "SIGNIFICANT DISPARITY" in svg
        assert "NO SIGNIFICANT DISPARITY" not in svg
        assert "Recalibrate group B" in svg
        assert "Strategy: platt" in svg
        # worst ECE 0.19 against best 0.02: the printed spread must be the gap.
        assert "0.170" in svg

    def test_an_object_result_is_read_through_attributes(self):
        class DisparityResult:
            group_metrics = {"A": {"ece": 0.01}, "B": {"ece": 0.02}}
            has_significant_disparity = False
            overall_ece = 0.015
            strategy = "isotonic"
            recommendation = "No action needed"

        svg = calibration_disparity_to_svg(DisparityResult())

        assert "NO SIGNIFICANT DISPARITY" in svg

    def test_a_group_with_no_ece_renders_as_not_available_and_is_left_out_of_the_spread(
        self,
    ):
        """Defensive behaviour documented in the adapter: an unmeasured group
        must not be silently scored, and must not widen the disparity either.

        Both fixtures gained an `overall_ece` on 2026-08-27. Without one the
        control half now renders "N/A" in the OVERALL ECE stat, which is the
        point of the fix (the default used to be 0, so a result that measured
        no overall error printed a perfect one). Supplying it keeps the
        unmeasured group C as the only source of an N/A, which is what this
        test is about.
        """
        with_gap = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.02}, "B": {"ece": 0.05}, "C": {}},
                "overall_ece": 0.035,
            }
        )
        without_c = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.02}, "B": {"ece": 0.05}},
                "overall_ece": 0.035,
            }
        )

        assert "N/A" in with_gap
        assert "N/A" not in without_c
        # 0.05 minus 0.02, unaffected by the unmeasured group C.
        assert "0.030" in with_gap
        assert "0.030" in without_c

    def test_a_non_numeric_ece_does_not_crash_the_report(self):
        svg = calibration_disparity_to_svg(
            {"group_metrics": {"A": {"ece": "not a number"}, "B": {"ece": 0.04}}}
        )

        assert "N/A" in svg
        assert "<svg" in svg

    def test_an_empty_result_is_not_rendered_as_no_significant_disparity(self):
        """Was an OPEN DEFECT until 2026-08-27, and the clearest of the set:
        the canvas showed a 99.90x DISPARITY RATIO sentinel, two N/A group
        names and ECE 0.000 for both, under a green 'NO SIGNIFICANT DISPARITY /
        All groups calibrated within acceptable bounds'. A sentinel is not a
        measurement, and nothing was compared."""
        svg = calibration_disparity_to_svg({})

        assert "NO SIGNIFICANT DISPARITY" not in svg
        assert "All groups calibrated within acceptable bounds" not in svg
        assert NOT_ASSESSABLE in svg
        assert "no group was supplied" in svg
        # Neither the sentinel nor a fabricated zero may reach the canvas.
        assert "99.90x" not in svg
        assert ">0.000<" not in svg
        assert "ECE = 0.000" not in svg
        assert ">N/A<" in svg

    def test_the_ratio_of_an_unmeasured_comparison_is_never_a_number(self):
        """The ratio had two ways to become 99.9: the inf sentinel for a zero
        denominator, and the display cap. Both printed as '99.90x' in alarm
        red, so a reader could not tell a missing value from a real one."""
        svg = calibration_disparity_to_svg({"group_metrics": {}})

        assert "99.90x" not in svg
        assert "99.9x" not in svg
        assert "N/Ax" not in svg  # the unit belongs to a number, and there is none

    def test_one_supplied_group_is_not_a_comparison(self):
        """Best and worst are comparative. Naming the single supplied group
        both the best and the worst calibrated compares it to nothing."""
        svg = calibration_disparity_to_svg(
            {"group_metrics": {"A": {"ece": 0.03}}, "overall_ece": 0.03}
        )

        assert "NO SIGNIFICANT DISPARITY" not in svg
        assert NOT_ASSESSABLE in svg
        assert "only one group" in svg
        # The group's own measured error survives; only the ranking is withdrawn.
        assert ">0.030<" in svg
        assert "ECE = 0.030" not in svg
        assert "1 of 1 group(s) had a supplied calibration error." in svg

    def test_an_undefined_ratio_is_not_printed_as_a_number(self):
        """A genuinely perfect best group makes the ratio a division by zero.
        That is could-not-check for the RATIO only: the spread is measured, and
        the disparity finding must still be reported."""
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.0}, "B": {"ece": 0.08}},
                "has_significant_disparity": True,
                "overall_ece": 0.04,
                "strategy": "platt",
                "recommendation": "Recalibrate B",
            }
        )

        assert "SIGNIFICANT DISPARITY" in svg
        assert "99.90x" not in svg
        assert "0.080" in svg  # the spread WAS measured
        assert ">N/A<" in svg  # the ratio was not

    def test_a_ratio_past_the_display_cap_says_so_instead_of_printing_the_cap(self):
        """5000x shown as '99.90x' is a number the data did not establish."""
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.0001}, "B": {"ece": 0.5}},
                "has_significant_disparity": True,
                "overall_ece": 0.25,
                "strategy": "platt",
                "recommendation": "Recalibrate B",
            }
        )

        assert "over 99.9x" in svg
        assert "99.90x" not in svg

    def test_a_caller_asserted_disparity_is_not_suppressed_by_the_guard(self):
        """The guard exists to stop fabricated REASSURANCE. A caller that
        reports a disparity without the numbers still gets its warning; only
        the numbers read N/A."""
        svg = calibration_disparity_to_svg(
            {
                "has_significant_disparity": True,
                "recommendation": "Recalibrate group B",
                "strategy": "platt",
            }
        )

        assert "SIGNIFICANT DISPARITY" in svg
        assert "NO SIGNIFICANT DISPARITY" not in svg
        assert "Recalibrate group B" in svg
        assert ">0.000<" not in svg
        assert ">N/A<" in svg

    def test_a_missing_overall_ece_is_not_printed_as_zero(self):
        """The default was 0, so a result that never measured an overall error
        rendered a perfect one in the headline stat."""
        svg = calibration_disparity_to_svg(
            {"group_metrics": {"A": {"ece": 0.02}, "B": {"ece": 0.05}}}
        )

        assert ">0.000<" not in svg
        assert ">N/A<" in svg
        # The spread between the two supplied groups is still measured.
        assert "0.030" in svg

    def test_a_nan_group_ece_is_not_printed_as_a_value(self):
        """float('nan') parses, then prints as the literal 'nan', which reads
        as a value and sorts unpredictably into the best/worst ranking."""
        svg = calibration_disparity_to_svg(
            {"group_metrics": {"A": {"ece": float("nan")}, "B": {"ece": 0.04}}}
        )

        assert "nan" not in svg
        assert ">N/A<" in svg

    def test_ordinary_disparity_input_is_untouched_by_the_guard(self):
        """Over-correction control for the second adapter."""
        svg = calibration_disparity_to_svg(
            {
                "group_metrics": {"A": {"ece": 0.01}, "B": {"ece": 0.02}, "C": {"ece": 0.015}},
                "has_significant_disparity": False,
                "overall_ece": 0.015,
                "strategy": "isotonic",
                "recommendation": "No action needed",
            }
        )

        assert "NO SIGNIFICANT DISPARITY" in svg
        assert "All groups calibrated within acceptable bounds." in svg
        assert NOT_ASSESSABLE not in svg
        assert "N/A" not in svg
        assert "Compares calibration error across demographic groups." in svg
        assert "2.00x" in svg  # 0.02 over 0.01
        assert "0.010" in svg  # the measured spread


class TestParetoFrontier:
    def test_dominated_configurations_are_excluded_from_the_frontier(self):
        # m4 is worse than every other point on both axes.
        svg = pareto_frontier_to_svg(
            [0.10, 0.20, 0.05, 0.30],
            [0.30, 0.10, 0.40, 0.50],
            labels=["m1", "m2", "m3", "m4"],
        )

        assert "4 configurations  |  3 Pareto-optimal" in svg

    def test_the_best_tradeoff_is_the_frontier_point_closest_to_the_origin(self):
        svg = pareto_frontier_to_svg(
            [0.10, 0.20, 0.05, 0.30],
            [0.30, 0.10, 0.40, 0.50],
            labels=["m1", "m2", "m3", "m4"],
        )

        assert "best trade-off: m2" in svg

    def test_a_single_configuration_is_its_own_frontier(self):
        svg = pareto_frontier_to_svg([0.2], [0.2], labels=["only"])

        assert "1 configurations  |  1 Pareto-optimal" in svg

    def test_flipping_the_x_direction_changes_which_points_dominate(self):
        lower_better = pareto_frontier_to_svg(
            [0.10, 0.90], [0.20, 0.20], labels=["cheap", "costly"]
        )
        higher_better = pareto_frontier_to_svg(
            [0.10, 0.90], [0.20, 0.20], labels=["cheap", "costly"], x_higher_is_better=True
        )

        assert "best trade-off: cheap" in lower_better
        assert "best trade-off: costly" in higher_better

    def test_the_fairness_threshold_line_is_labelled_when_it_is_in_range(self):
        svg = pareto_frontier_to_svg(
            [0.10, 0.20], [0.05, 0.30], labels=["a", "b"], fairness_threshold=0.15
        )

        assert "ABOVE THRESHOLD" in svg
        assert "PASS ZONE" in svg
        # With no threshold given, no pass/fail zone may be implied at all.
        assert "PASS ZONE" not in pareto_frontier_to_svg(
            [0.10, 0.20], [0.05, 0.30], labels=["a", "b"]
        )
