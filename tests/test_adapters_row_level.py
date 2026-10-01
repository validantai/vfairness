"""A ROW that reported nothing must not be graded, coloured or counted.

Wave 12 of the invented-analysis sweep. The eleven waves before it closed the
CHART-level hole: every public ``*_to_svg`` adapter now renders an explicit
could-not-check state when the WHOLE input is empty. That is not the same as
being closed per row, and these two adapters were the proof.

DEFECT 1, ranking, the worst of the campaign. ``adapters_ranking`` read the two
numbers it grades out of ``d.get("value", 0)`` and ``d.get("threshold", 0.1)``,
so a row carrying ONLY ``metric_name`` was graded against an invented pair.
Executed on the tree at 4d1da0a, ``ranking_fairness_to_svg([{"metric_name":
"ndkl"}], {})`` rendered a green PASS badge reading 0.000, a green OVERALL FAIR
headline, "1/1 metrics pass", and the sentence "All ranking fairness metrics
pass their configured thresholds". Confirmed identically for
``exposure_parity_difference`` and ``ndcg_parity_difference``. For a
higher-is-better name the same defaults produced a fabricated FAIL and an UNFAIR
headline, which is not the safe direction either: it names a metric that was
never measured and it gets acted on.

DEFECT 2, robustness. ``adapters_robustness`` derived significance from
``d.get("p_value", 1)`` and printed the same default in the table, so a test
that reported no p-value rendered "P-VALUE 1.0000" beside a green STABLE badge.
1.0 is the most reassuring p-value the table can print and it was a SENTINEL.
Worse, that invented stability was averaged into the overall score: the single
row ``{"method": "demographic_parity"}`` rendered ROBUSTNESS PASS at score 1.00.

The rule pinned here, per row: a row that reported nothing gets no number, no
badge, no colour, and no place in any count or headline that implies it was
measured. A default is not a measurement, a sentinel is not a measurement, and
absent and zero are different.

The over-correction control is ``TestHealthyRowsAreUnchanged`` at the bottom.
Beyond it, every healthy render of both adapters was rasterised against the
committed tree at 4d1da0a and compared byte for byte, outside pytest so that the
``pythonpath = ["src"]`` setting in pyproject could not silently load the new
code for both halves; the build actually loaded was asserted each way by a
marker unique to it. Sixteen of eighteen controls were byte-identical PNGs. The
two that moved: the robustness subgroup-only canvas differs in template
WHITESPACE only (its PNG is byte-identical), and the ranking NOT CHECKED badge
label, which the old build painted PAST the right edge of its own badge, is now
set from the right edge inside it.
"""

import re

import pytest

from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg
from vfairness.rendering.adapters_robustness import robustness_testing_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


COULD_NOT_CHECK = "COULD NOT CHECK"
ALL_PASS = "All ranking fairness metrics pass their configured thresholds."
ALL_ROBUST = "All robustness checks passed"

GROUPS = {
    "A": {
        "mean_exposure": 0.59,
        "avg_position": 2.0,
        "median_position": 2,
        "min_position": 0,
        "max_position": 4,
        "count": 5,
    },
    "B": {
        "mean_exposure": 0.52,
        "avg_position": 3.0,
        "median_position": 3,
        "min_position": 1,
        "max_position": 6,
        "count": 5,
    },
}

# The three ranking names reproduced by hand on the committed tree, plus the
# ratio, whose direction is the opposite and whose fabricated verdict was a FAIL.
LOWER_IS_BETTER = ["ndkl", "exposure_parity_difference", "ndcg_parity_difference"]

AUDIT = {
    "n_subgroups_analyzed": 6,
    "n_subgroups_flagged": 1,
    "worst_subgroup": "age_under_25 and female",
    "worst_disparity": 0.142,
    "gerrymandering_detected": False,
    "flagged_subgroups": [{"name": "age_under_25 and female", "disparity": 0.142}],
}
PERM_GRADED = [
    {
        "method": "equalized_odds",
        "observed_statistic": 0.087,
        "p_value": 0.003,
        "significant_at_05": True,
        "effect_direction": "negative",
    },
    {
        "method": "demographic_parity",
        "observed_statistic": 0.042,
        "p_value": 0.312,
        "effect_direction": "positive",
    },
]
SENS = [
    {
        "perturbation_type": "label_noise_5%",
        "robustness_score": 0.92,
        "is_robust": True,
        "max_deviation": 0.008,
    }
]


def _desc(svg: str) -> str:
    """The accessible one-liner, which is all a screen-reader user receives."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "the rendered SVG carries no <desc>"
    return match.group(1)


def _drawn(svg: str) -> list:
    """Every drawn text node, so a badge label is never matched inside prose."""
    return [t.strip() for t in re.findall(r">([^<>]+)<", svg) if t.strip()]


# ---------------------------------------------------------------------------
# DEFECT 1: the ranking row graded from a defaulted value and threshold.
# ---------------------------------------------------------------------------


class TestARankingRowThatReportedNothing:
    @pytest.mark.parametrize("name", LOWER_IS_BETTER)
    def test_a_row_carrying_only_its_name_is_not_a_pass(self, name):
        """The reproduced defect: PASS, FAIR and "all pass" out of one bare name."""
        svg = ranking_fairness_to_svg([{"metric_name": name}], GROUPS)

        assert "PASS" not in _drawn(svg)
        assert ">FAIR<" not in svg
        assert ALL_PASS not in svg

    def test_a_higher_is_better_row_carrying_only_its_name_is_not_a_fail(self):
        """The same defaults fabricated a FAIL for the ratio family, which names
        a breach that was never measured. Neither direction may be invented."""
        svg = ranking_fairness_to_svg([{"metric_name": "exposure_parity_ratio"}], GROUPS)

        assert "FAIL" not in _drawn(svg)
        assert ">UNFAIR<" not in svg

    @pytest.mark.parametrize("name", LOWER_IS_BETTER)
    def test_the_row_prints_no_number(self, name):
        """0.000 is the defaulted value, drawn in the verdict colour."""
        svg = ranking_fairness_to_svg([{"metric_name": name}], GROUPS)

        assert "0.000" not in _drawn(svg)
        assert "no value reported" in svg

    def test_the_row_is_not_painted_in_either_verdict_colour(self):
        """A reader takes the grade off the colour before the words.

        Read out of the rendered document rather than compared to a literal hex:
        the skin recolours every fill on the way out.
        """
        passing = ranking_fairness_to_svg(
            [{"metric_name": "ndkl", "value": 0.02, "threshold": 0.10}], GROUPS
        )
        failing = ranking_fairness_to_svg(
            [{"metric_name": "ndkl", "value": 0.42, "threshold": 0.10}], GROUPS
        )
        green = re.search(r'fill="(#[0-9a-fA-F]{6})">PASS<', passing)
        red = re.search(r'fill="(#[0-9a-fA-F]{6})">FAIL<', failing)
        assert green and red, "the graded controls did not render a badge to compare against"

        svg = ranking_fairness_to_svg([{"metric_name": "ndkl"}], GROUPS)
        neutral = re.search(r'fill="(#[0-9a-fA-F]{6})">NOT MEASURED<', svg)
        assert neutral, "the ungraded badge did not render"
        assert neutral.group(1) not in (green.group(1), red.group(1))

    def test_the_row_takes_no_place_in_the_pass_rate(self):
        """One graded pass beside one ungraded row is not two out of two."""
        svg = ranking_fairness_to_svg(
            [
                {"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.10},
                {"metric_name": "ndkl"},
            ],
            GROUPS,
        )

        assert "2/2" not in svg
        assert "1/2" not in svg
        assert ">NOT ASSESSED<" in svg
        assert "1 of 2 metric(s) not graded" in svg
        assert ALL_PASS not in svg

    def test_a_value_with_no_threshold_is_not_graded_against_an_invented_bound(self):
        """0.1 was the defaulted threshold, and it is nobody's decision."""
        svg = ranking_fairness_to_svg([{"metric_name": "ndkl", "value": 0.02}], GROUPS)

        assert "PASS" not in _drawn(svg)
        assert ">FAIR<" not in svg

    def test_the_row_is_named_where_a_reader_looks(self):
        svg = ranking_fairness_to_svg([{"metric_name": "ndkl"}], GROUPS)

        assert "1 metric(s) reported no value to grade (ndkl)" in svg
        assert "certifies nothing about them" in svg

    def test_a_reported_verdict_with_no_value_still_withholds_the_number(self):
        """The two flags are independent. A caller-supplied verdict is graded,
        and the number beside it is still withheld rather than defaulted to
        0.000: the metric module reported a decision, not a measurement."""
        svg = ranking_fairness_to_svg([{"metric_name": "exposure_parity", "is_fair": True}], GROUPS)

        assert "no value reported" in svg
        assert "0.000" not in _drawn(svg)

    def test_the_accessible_description_certifies_nothing(self):
        """The <desc> is the whole canvas for a screen-reader user, and it used
        to read "Ranking fairness: 1/1 metrics pass ...; all pass"."""
        desc = _desc(ranking_fairness_to_svg([{"metric_name": "ndkl"}], GROUPS))

        assert COULD_NOT_CHECK in desc
        assert "1/1 metrics pass" not in desc
        assert "all pass" not in desc


# ---------------------------------------------------------------------------
# DEFECT 2: the permutation row whose significance came from a defaulted 1.0.
# ---------------------------------------------------------------------------


class TestAPermutationRowThatReportedNothing:
    def test_the_row_prints_no_p_value(self):
        """1.0000 was the defaulted p-value: the most reassuring one there is."""
        svg = robustness_testing_to_svg([{"method": "demographic_parity"}], [])

        assert "1.0000" not in svg
        assert "not reported" in svg

    def test_the_row_is_not_reported_as_stable(self):
        svg = robustness_testing_to_svg([{"method": "demographic_parity"}], [])

        assert "STABLE" not in _drawn(svg)
        assert "SIGNIF." not in _drawn(svg)
        assert "NOT CHECKED" in _drawn(svg)

    def test_the_row_does_not_score_a_run_that_measured_nothing(self):
        """The invented stability was averaged into the headline: one bare row
        rendered ROBUSTNESS PASS at score 1.00."""
        svg = robustness_testing_to_svg([{"method": "demographic_parity"}], [])

        assert ">NOT SCORED<" in svg
        assert "PASS" not in _drawn(svg)
        assert "score:" not in svg
        assert "1.00" not in _drawn(svg)

    def test_the_row_is_kept_out_of_the_stability_count(self):
        """One significant test beside one ungraded row is a permutation
        component of 0, not the 0.5 that counting the ungraded row as a pass
        produced. The old build rendered MARGINAL here; the graded evidence says
        FAIL."""
        svg = robustness_testing_to_svg(
            [
                {"method": "equalized_odds", "observed_statistic": 0.09, "p_value": 0.003},
                {"method": "calibration_diff"},
            ],
            [],
        )

        assert "0.50" not in svg
        assert ">MARGINAL<" not in svg
        assert ">FAIL<" in svg

    def test_the_row_is_named_and_the_all_clear_is_withheld(self):
        svg = robustness_testing_to_svg(
            [
                {"method": "equalized_odds", "observed_statistic": 0.01, "p_value": 0.7},
                {"method": "calibration_diff"},
            ],
            SENS,
        )

        assert "1 permutation test(s) reported no p-value (calibration_diff)" in svg
        assert "excluded from the robustness score" in svg
        assert ALL_ROBUST not in svg

    def test_the_row_is_still_shown_and_counted_as_supplied(self):
        """Withholding the grade must not hide the row: it was supplied, and the
        count in the subtitle says how many of the supplied tests were graded."""
        svg = robustness_testing_to_svg([{"method": "calibration_diff"}], [])

        assert "calibration_diff" in _drawn(svg)
        assert "1 permutation tests (1 not graded)" in svg

    def test_a_reported_significance_with_no_p_value_still_withholds_the_number(self):
        """A verdict is not a p-value. The row grades on the verdict it reported
        and the cell stays empty rather than printing the defaulted 1.0000."""
        svg = robustness_testing_to_svg(
            [{"method": "dp", "observed_statistic": 0.01, "significant_at_05": False}], []
        )

        assert "1.0000" not in svg
        assert "not reported" in svg

    def test_the_accessible_description_certifies_nothing(self):
        desc = _desc(robustness_testing_to_svg([{"method": "demographic_parity"}], []))

        assert COULD_NOT_CHECK in desc
        assert "score 1.00" not in desc
        assert "score 0.00" not in desc


# ---------------------------------------------------------------------------
# Over-correction controls. A row that DID report is graded exactly as before.
# ---------------------------------------------------------------------------


class TestHealthyRowsAreUnchanged:
    def test_a_measured_ranking_pass_is_still_a_pass(self):
        svg = ranking_fairness_to_svg(
            [
                {"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.1},
                {"metric_name": "ndcg_parity", "value": 0.01, "threshold": 0.1},
            ],
            GROUPS,
        )

        assert ">FAIR<" in svg
        assert "2/2" in svg
        assert ALL_PASS in svg
        assert "no value reported" not in svg

    def test_a_measured_ranking_breach_is_still_a_breach(self):
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity", "value": 0.27, "threshold": 0.1}], GROUPS
        )

        assert ">UNFAIR<" in svg
        assert "0.270" in _drawn(svg)

    def test_the_module_verdict_still_wins_when_the_caller_supplies_it(self):
        """Pinned in test_adapters_ranking_verdicts.py as well: is_fair beats the
        adapter's own comparison, and the row-level fix must not re-decide it."""
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity", "value": 0.01, "threshold": 0.1, "is_fair": False}],
            GROUPS,
        )

        assert ">UNFAIR<" in svg
        assert "0/1" in svg

    def test_a_reported_verdict_with_no_value_keeps_the_verdict(self):
        """A verdict is a measurement even when the number beside it is missing.

        The withholding half of this case is asserted in
        ``TestARankingRowThatReportedNothing`` above; this control asserts only
        that the verdict itself survives, and it passes on the committed tree.
        """
        svg = ranking_fairness_to_svg([{"metric_name": "exposure_parity", "is_fair": True}], GROUPS)

        assert "PASS" in _drawn(svg)
        assert ">FAIR<" in svg

    def test_an_unmeasurable_ranking_value_still_fails_closed(self):
        """A NaN VALUE was reported, and this page has always failed closed on
        it. It is not the same as no value at all."""
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity", "value": float("nan"), "threshold": 0.1}], GROUPS
        )

        assert ">UNFAIR<" in svg
        assert "0/1" in svg

    def test_a_measured_ranking_zero_is_still_a_measurement(self):
        """Absent and zero are different: an exact 0.0 is a real reading."""
        svg = ranking_fairness_to_svg(
            [{"metric_name": "ndkl", "value": 0.0, "threshold": 0.1}], GROUPS
        )

        assert "PASS" in _drawn(svg)
        assert "0.000" in _drawn(svg)
        assert "no value reported" not in svg

    def test_a_full_robustness_run_still_reports_its_verdict(self):
        svg = robustness_testing_to_svg(PERM_GRADED, SENS, AUDIT)

        assert ">NOT SCORED<" not in svg
        assert "score:" in svg
        assert "0.3120" in svg and "0.0030" in svg
        assert "STABLE" in _drawn(svg) and "SIGNIF." in _drawn(svg)
        assert "not reported" not in svg

    def test_a_clean_robustness_run_still_reports_the_all_clear(self):
        svg = robustness_testing_to_svg(
            [{"method": "dp", "observed_statistic": 0.01, "p_value": 0.7}],
            [
                {
                    "perturbation_type": "noise",
                    "robustness_score": 0.95,
                    "is_robust": True,
                    "max_deviation": 0.001,
                }
            ],
        )

        assert ALL_ROBUST in svg
        assert "STABLE" in _drawn(svg)

    def test_a_reported_significance_with_no_p_value_still_grades(self):
        """The test's own verdict wins, exactly as is_fair does on the ranking
        canvas.

        The withholding half is asserted in
        ``TestAPermutationRowThatReportedNothing`` above; this control asserts
        only that the verdict survives, and it passes on the committed tree.
        """
        svg = robustness_testing_to_svg(
            [{"method": "dp", "observed_statistic": 0.01, "significant_at_05": False}], []
        )

        assert "STABLE" in _drawn(svg)
        assert "NOT CHECKED" not in _drawn(svg)
        assert "score:" in svg

    def test_a_measured_p_value_of_zero_is_still_a_measurement(self):
        svg = robustness_testing_to_svg(
            [{"method": "dp", "observed_statistic": 0.4, "p_value": 0.0}], []
        )

        assert "0.0000" in _drawn(svg)
        assert "SIGNIF." in _drawn(svg)
        assert "NOT CHECKED" not in _drawn(svg)

    def test_the_subgroup_only_canvas_is_unchanged(self):
        """Pinned in test_adapters_robustness_experimentation.py: this branch has
        its own wording, and the row-level fix must not reword it."""
        svg = robustness_testing_to_svg([], [], AUDIT)

        assert ">NOT SCORED<" in svg
        assert "No permutation test and no sensitivity check was supplied" in svg
        assert "no robustness score was computed" in svg
        assert "the audit is not graded" in svg

    def test_the_nothing_tested_canvas_is_unchanged(self):
        svg = robustness_testing_to_svg([], [], None)

        assert "Nothing was tested" in svg
        assert ">METHOD<" not in svg
