"""Per-ROW honesty for the robustness, monitoring, training and ranking canvases.

The chart-level rule ("an adapter handed the emptiest input it accepts renders an
explicit could-not-check state") was closed for all 44 public ``*_to_svg``
adapters by an earlier campaign. This file pins the same rule ONE ROW AT A TIME,
for the sites recorded in ``docs/audits/row-level-fabrication-register-2026-08-27.md``
under this agent's four adapters.

The rule
--------
A row that reported nothing gets no number, no badge, no colour, no point on a
plot, and no place in any count or headline that implies it was measured. A
default is not a measurement. A sentinel is not a measurement. Absent and zero
are different claims.

Both directions are wrong, and both are pinned below:

* a fabricated ALL-CLEAR tells a reader a group was checked and cleared when it
  was not (the robustness subgroup panel, the temporal trend cards);
* a fabricated BREACH sends someone after a violation that does not exist and
  discredits the tool when they work out why (the robustness sensitivity bars,
  both training method tables).

The decided headline rule
-------------------------
Two adapters had drifted into two different conventions for a partially measured
run, so it was decided:

a. grade the headline over the GRADED subset, so a partially-measured run still
   gives a useful verdict on what WAS measured;
b. never render an unqualified all-clear while anything is ungraded;
c. state the ungraded count prominently ON THE CANVAS, next to the headline;
d. an ungraded row never enters a numerator or a denominator that implies
   measurement.

Every class below carries at least one CONTROL: fully measured input must render
exactly as it always did. Beyond these string assertions, each case was also
rasterised with ``rsvg-convert`` and read, and the fully measured renders of all
four adapters were diffed against the committed tree (39cc1e6) at the pixel
level.
"""

import re

import pandas as pd
import pytest

from vfairness.in_processing.analyzer import FairnessTrainingReport, MethodComparison
from vfairness.operations.monitoring import TemporalFairnessAnalyzer
from vfairness.rendering.adapters_monitoring import temporal_analysis_to_svg
from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg
from vfairness.rendering.adapters_robustness import robustness_testing_to_svg
from vfairness.rendering.adapters_training import (
    method_comparison_to_svg,
    training_report_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


# One measured permutation test, so the CHART carries real data in every case
# below and only the row under test reports nothing. That is the whole point:
# a green chart must not certify a row nobody measured.
MEASURED_PERM = {
    "method": "demographic_parity",
    "observed_statistic": 0.042,
    "p_value": 0.312,
    "effect_direction": "positive",
}
MEASURED_SENS = {
    "perturbation_type": "label_noise_5%",
    "robustness_score": 0.92,
    "max_deviation": 0.008,
    "is_robust": True,
}


def _headline(svg: str) -> str:
    """The ROBUSTNESS verdict word off the canvas."""
    m = re.search(r'font-weight="700" fill="#[0-9a-fA-F]{6}">(PASS|MARGINAL|FAIL|NOT SCORED)<', svg)
    assert m, "the robustness headline did not render"
    return m.group(1)


def _score(svg: str):
    """The printed robustness score, or None when the canvas withheld it."""
    m = re.search(r"score: <tspan[^>]*>([\d.]+)", svg)
    return float(m.group(1)) if m else None


class TestSensitivityRowThatReportedNothing:
    """A perturbation with no ``robustness_score``: a FABRICATED BREACH.

    ``d.get("robustness_score", 0)`` handed the row the WORST score the scale
    has, so it drew a red WEAK badge at 0.00 and then that invented weakness was
    averaged into the 0.6-weighted sensitivity component.
    """

    def test_it_gets_no_number_and_no_bar(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[MEASURED_SENS, {"perturbation_type": "feature_dropout_10%"}],
        )

        # The measured row keeps its number; the bare row prints no number at all.
        assert ">0.92<" in svg
        assert ">0.00<" not in svg
        assert ">no score<" in svg
        assert "max dev: not reported" in svg

    def test_it_gets_no_weak_badge(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[MEASURED_SENS, {"perturbation_type": "feature_dropout_10%"}],
        )

        assert ">WEAK<" not in svg
        assert ">NOT CHECKED<" in svg
        assert ">ROBUST<" in svg  # the measured row still carries its verdict

    def test_it_never_drags_the_overall_score(self):
        """The whole reason this one matters: it moved the HEADLINE."""
        measured_only = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], sensitivity_results=[MEASURED_SENS]
        )
        with_bare_row = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[MEASURED_SENS, {"perturbation_type": "feature_dropout_10%"}],
        )

        assert _score(measured_only) == _score(with_bare_row)
        assert _headline(measured_only) == _headline(with_bare_row) == "PASS"

    def test_two_bare_rows_do_not_manufacture_a_failure(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[
                MEASURED_SENS,
                {"perturbation_type": "feature_dropout_10%"},
                {"perturbation_type": "sample_bootstrap"},
            ],
        )

        assert _headline(svg) != "FAIL"
        assert "Some perturbation types show low robustness" not in svg

    def test_a_reported_verdict_still_grades_without_a_score(self):
        """``is_robust`` is a measurement even with no number beside it.

        Same precedence as ``significant_at_05`` on the permutation table and
        ``is_fair`` on the ranking canvas: the check's own verdict wins.
        """
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[{"perturbation_type": "feature_dropout_10%", "is_robust": False}],
        )

        assert ">WEAK<" in svg
        assert ">no score<" in svg  # graded, but not measured
        assert "Some perturbation types show low robustness" in svg

    def test_the_ungraded_count_is_on_the_canvas(self):
        """Decided headline rule (c)."""
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[MEASURED_SENS, {"perturbation_type": "feature_dropout_10%"}],
        )

        assert "2 sensitivity checks (1 not graded)" in svg

    def test_it_forbids_the_all_clear(self):
        """Decided headline rule (b): a check that did not run did not pass."""
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[MEASURED_SENS, {"perturbation_type": "feature_dropout_10%"}],
        )

        assert "All robustness checks passed" not in svg
        assert "feature_dropout_10%" in svg
        assert "not graded" in svg

    # CONTROL
    def test_fully_measured_sensitivity_rows_are_unchanged(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            sensitivity_results=[
                MEASURED_SENS,
                {
                    "perturbation_type": "feature_dropout_10%",
                    "robustness_score": 0.31,
                    "max_deviation": 0.19,
                    "is_robust": False,
                },
            ],
        )

        assert ">ROBUST<" in svg and ">WEAK<" in svg
        assert ">NOT CHECKED<" not in svg
        assert ">0.92<" in svg and ">0.31<" in svg
        assert "max dev: 0.008" in svg and "max dev: 0.190" in svg
        assert "not graded" not in svg
        assert "Some perturbation types show low robustness" in svg


class TestSubgroupAuditThatReportedOnlyItsCount:
    """A FABRICATED ALL-CLEAR: "0 flagged", worst disparity 0.000, no risk chip."""

    BARE = {"n_subgroups_analyzed": 6}
    FULL = {
        "n_subgroups_analyzed": 6,
        "n_subgroups_flagged": 0,
        "worst_subgroup": "age<25",
        "worst_disparity": 0.02,
        "gerrymandering_detected": False,
    }

    def test_it_does_not_render_zero_flagged(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], subgroup_audit=self.BARE
        )

        assert "0 flagged" not in svg
        assert "flagged count not reported" in svg

    def test_it_does_not_render_a_zero_worst_disparity(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], subgroup_audit=self.BARE
        )

        assert "(0.000)" not in svg
        assert "worst disparity not reported" in svg

    def test_an_unrun_gerrymandering_check_says_so(self):
        """The ABSENCE of the red chip reads as "checked and clear"."""
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], subgroup_audit=self.BARE
        )

        assert "GERRYMANDERING NOT CHECKED" in svg
        assert "GERRYMANDERING RISK" not in svg

    def test_the_recommendations_refuse_the_all_clear(self):
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], subgroup_audit=self.BARE
        )

        assert "All robustness checks passed" not in svg
        assert "reported no flagged count" in svg

    def test_an_enumerated_flag_list_is_itself_a_reported_count(self):
        """An audit that listed its flagged subgroups DID report the count."""
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM],
            subgroup_audit={
                "n_subgroups_analyzed": 6,
                "flagged_subgroups": [("age<25", 0.14)],
                "worst_subgroup": "age<25",
                "worst_disparity": 0.14,
                "gerrymandering_detected": False,
            },
        )

        assert "1 flagged" in svg
        assert "flagged count not reported" not in svg

    # CONTROL
    def test_a_genuinely_clean_audit_still_reads_clean(self):
        """A measured zero is a measurement and keeps its honest silence."""
        svg = robustness_testing_to_svg(
            permutation_results=[MEASURED_PERM], subgroup_audit=self.FULL
        )

        assert "6 analysed · 0 flagged · worst: age&lt;25 (0.020)" in svg
        assert "not reported" not in svg
        assert "GERRYMANDERING" not in svg
        assert "All robustness checks passed" in svg


class TestTemporalMetricNeverTracked:
    """A FABRICATED ALL-CLEAR: five 0.000s under a green STABLE chip.

    ``get_metric_summary`` returns ``{}`` for a metric the analyzer holds no
    column for, and every statistic came out of a ``.get`` default on it.
    """

    @staticmethod
    def _analyzer(days: int = 21) -> TemporalFairnessAnalyzer:
        a = TemporalFairnessAnalyzer(lookback_days=90)
        for i in range(days):
            a.update_daily_metrics(
                pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                {"demographic_parity": 0.05 + i * 0.002},
            )
        return a

    def test_it_gets_no_statistics(self):
        svg = temporal_analysis_to_svg(
            self._analyzer(), metric_names=["demographic_parity", "equalized_odds"]
        )

        assert "min 0.000" not in svg
        assert "max 0.000" not in svg
        assert "+0.0000" not in svg
        assert "No history was recorded for this metric" in svg

    def test_it_gets_no_stable_chip(self):
        svg = temporal_analysis_to_svg(
            self._analyzer(), metric_names=["demographic_parity", "equalized_odds"]
        )

        assert ">NOT TRACKED<" in svg
        # The tracked metric keeps its own real verdict.
        assert ">INCREASING<" in svg
        assert ">STABLE<" not in svg

    def test_its_trend_row_is_not_a_fitted_trend(self):
        """``detect_trend`` answers ("stable", 0.0) for a metric it does not hold."""
        svg = temporal_analysis_to_svg(
            self._analyzer(), metric_names=["demographic_parity", "equalized_odds"]
        )

        assert "0.0000/day" not in svg
        assert "0 days" not in svg
        assert "no history recorded, so no trend was fitted" in svg
        assert ">stable</text>" not in svg

    def test_a_degradation_check_that_did_not_run_is_not_a_stable_verdict(self):
        """``detect_weekly_degradation`` returns (False, nan) when it REFUSES to run."""
        svg = temporal_analysis_to_svg(
            self._analyzer(), metric_names=["demographic_parity", "equalized_odds"]
        )

        assert "worst day mean: nan" not in svg
        assert "the weekly check did not run" in svg
        assert ">NOT CHECKED<" in svg

    def test_the_subtitle_names_the_untracked_metric(self):
        """Decided headline rule (c)."""
        svg = temporal_analysis_to_svg(
            self._analyzer(), metric_names=["demographic_parity", "equalized_odds"]
        )

        assert "2 metrics (1 never tracked)" in svg

    def test_every_metric_untracked_is_could_not_check(self):
        """Three full-length row lists, none of them measuring anything."""
        svg = temporal_analysis_to_svg(self._analyzer(), metric_names=["never_a", "never_b"])

        assert "No metric history was available for trend analysis" in svg
        assert "COULD NOT CHECK" in svg or "NOT CHECKED" in svg

    # CONTROL
    def test_a_tracked_metric_keeps_every_statistic(self):
        svg = temporal_analysis_to_svg(self._analyzer(), metric_names=["demographic_parity"])

        assert "min 0.050" in svg and "max 0.090" in svg
        assert "+0.0020" in svg
        assert ">INCREASING<" in svg
        assert "never tracked" not in svg
        assert "no history recorded" not in svg


def _mc(name, accuracy, violation, satisfied, time=1.0):
    return MethodComparison(
        method_name=name,
        accuracy=accuracy,
        fairness_violation=violation,
        constraint_satisfied=satisfied,
        training_time=time,
    )


def _report(methods):
    return FairnessTrainingReport(
        timestamp="2026-08-27T12:00:00",
        task_type="classification",
        data_info={"n_samples": 1000, "n_groups": 2, "n_features": 8, "attribute_name": "sex"},
        baseline_metrics={
            "accuracy": 0.83,
            "fairness_violation": 0.04,
            "constraint_satisfied": True,
        },
        fairness_analysis={},
        method_comparisons=methods,
        recommendation=None,
        tradeoff_analysis={},
        critical_issues=[],
        action_items=[],
    )


PASSING_METHOD = _mc("reweighting", 0.81, 0.03, True)
FAILING_METHOD = _mc("adversarial", 0.79, 0.12, False)
UNGRADED_METHOD = _mc("adversarial", 0.79, 0.06, None)


class TestTrainingMethodWithNoConstraintResult:
    """A FABRICATED BREACH on both training canvases.

    Every verdict branch was ``{% if m.satisfied %} ... {% else %} FAIL``, so a
    ``None`` took the FAIL arm: red row stripe, red violation number, red chip.
    """

    def test_the_training_report_does_not_draw_it_as_a_failure(self):
        svg = training_report_to_svg(_report([PASSING_METHOD, UNGRADED_METHOD]))

        assert ">PASS<" in svg
        assert ">FAIL<" not in svg
        assert ">NOT CHECKED<" in svg

    def test_the_method_comparison_does_not_draw_it_as_a_failure(self):
        """Both canvases read the same field, so both had the same defect."""
        svg = method_comparison_to_svg([PASSING_METHOD, UNGRADED_METHOD])

        assert ">PASS<" in svg
        assert ">FAIL<" not in svg
        assert ">NOT CHECKED<" in svg

    def test_the_ungraded_count_is_on_the_canvas(self):
        """Decided headline rule (c)."""
        svg = method_comparison_to_svg([PASSING_METHOD, UNGRADED_METHOD])

        assert "2 methods evaluated (1 constraint not evaluated)" in svg

    def test_every_method_ungraded_is_could_not_check(self):
        svg = method_comparison_to_svg([UNGRADED_METHOD, _mc("postproc", 0.7, 0.2, None)])

        assert "No training method was evaluated" in svg
        assert ">PASS<" not in svg and ">FAIL<" not in svg
        assert "reported a constraint result" in svg

    # CONTROLS
    def test_a_measured_failing_method_still_fails(self):
        """The point of the fix is the DIFFERENCE between None and False."""
        report_svg = training_report_to_svg(_report([PASSING_METHOD, FAILING_METHOD]))
        compare_svg = method_comparison_to_svg([PASSING_METHOD, FAILING_METHOD])

        for svg in (report_svg, compare_svg):
            assert ">FAIL<" in svg
            assert ">PASS<" in svg
            assert ">NOT CHECKED<" not in svg
            assert "not evaluated" not in svg

    def test_a_measured_zero_violation_still_prints(self):
        """``is not None``, never truthiness: 0.0 is a measurement."""
        svg = method_comparison_to_svg([_mc("perfect", 0.9, 0.0, True)])

        assert ">0.000<" in svg
        assert "not measured" not in svg


class TestMethodComparisonNoLongerReturnsSilentlyEmpty:
    """The silent-empty-return class, one row at a time.

    A method with no accuracy made ``method_comparison.svg`` raise TypeError
    inside its bar-width arithmetic, and the adapter answered with a
    zero-length string and a UserWarning. A zero-length SVG is the worst of the
    three states: it cannot even be read as could-not-check, and a caller that
    writes it gets a 0-byte file that opens as a broken image.
    """

    def test_a_row_with_no_accuracy_no_longer_empties_the_chart(self):
        svg = method_comparison_to_svg([PASSING_METHOD, _mc("postproc", None, 0.09, True)])

        assert svg, "the whole chart was returned empty for one unmeasured cell"
        assert svg.lstrip().startswith(("<svg", "<?xml", "\n"))
        assert "<svg" in svg
        # The measured row survives intact; only the unmeasured CELL is withheld.
        assert ">0.810<" in svg
        assert "not measured" in svg

    def test_it_raises_no_warning_on_that_row(self, recwarn):
        method_comparison_to_svg([PASSING_METHOD, _mc("postproc", None, 0.09, True)])

        assert [w for w in recwarn.list if "SVG rendering failed" in str(w.message)] == []

    def test_a_row_with_no_violation_no_longer_empties_the_chart(self):
        """The violation bar multiplies its number too, so it had the same hole."""
        svg = method_comparison_to_svg([PASSING_METHOD, _mc("postproc", 0.77, None, True)])

        assert svg, "the whole chart was returned empty for one unmeasured cell"
        assert ">0.770<" in svg
        assert "not measured" in svg

    def test_the_file_it_writes_is_not_zero_bytes(self, tmp_path):
        path = tmp_path / "method_comparison.svg"
        svg = method_comparison_to_svg(
            [PASSING_METHOD, _mc("postproc", None, None, True)], save_path=str(path)
        )

        assert path.stat().st_size > 0
        assert path.read_text() == svg


class TestRankingCountsOverTheGradedSubset:
    """Decided headline rule (a) and (d), where the two conventions met.

    Wave 12 fixed the row STATE on this canvas and left the counts alone, so a
    page with one measured breach and six rows that reported nothing printed
    "UNFAIR 0/7 metrics pass": six metrics in the DENOMINATOR of a rate about
    passing a threshold none of them was ever compared to.
    """

    GROUPS = {"A": {"mean_exposure": 0.59, "count": 5}, "B": {"mean_exposure": 0.32, "count": 5}}
    MEASURED_FAIL = {"metric_name": "exposure_parity", "value": 0.27, "threshold": 0.1}
    MEASURED_PASS = {"metric_name": "ndcg_parity", "value": 0.01, "threshold": 0.1}

    @staticmethod
    def _ratio(svg: str) -> str:
        found = re.findall(r">(\d+/\d+)<", svg)
        assert found, "the metrics-pass ratio was not rendered"
        return found[0]

    def test_ungraded_rows_leave_the_denominator(self):
        bare = [{"metric_name": f"m{i}"} for i in range(6)]

        svg = ranking_fairness_to_svg([self.MEASURED_FAIL] + bare, self.GROUPS)

        assert self._ratio(svg) == "0/1"
        assert "0/7" not in svg

    def test_a_measured_breach_still_wins_the_headline(self):
        """Rule (a): the verdict is graded over what WAS measured."""
        bare = [{"metric_name": f"m{i}"} for i in range(6)]

        svg = ranking_fairness_to_svg([self.MEASURED_FAIL] + bare, self.GROUPS)

        assert ">UNFAIR<" in svg

    def test_the_ungraded_count_is_on_the_canvas(self):
        """Rule (c): beside the headline, not only in the <desc>."""
        bare = [{"metric_name": f"m{i}"} for i in range(6)]

        svg = ranking_fairness_to_svg([self.MEASURED_FAIL] + bare, self.GROUPS)

        assert "6 of 7 metrics not graded" in svg

    def test_a_partially_measured_page_never_claims_the_all_clear(self):
        """Rule (b): FAIR is a claim over the whole population."""
        svg = ranking_fairness_to_svg([self.MEASURED_PASS, {"metric_name": "ndkl"}], self.GROUPS)

        assert ">FAIR<" not in svg
        assert "All ranking fairness metrics pass" not in svg
        assert "1 of 2 metrics not graded" in svg

    # CONTROLS
    def test_a_fully_graded_page_keeps_its_rate(self):
        svg = ranking_fairness_to_svg([self.MEASURED_PASS, self.MEASURED_FAIL], self.GROUPS)

        assert self._ratio(svg) == "1/2"
        assert "not graded" not in svg

    def test_a_fully_passing_page_still_reads_fair(self):
        svg = ranking_fairness_to_svg(
            [
                self.MEASURED_PASS,
                {"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.1},
            ],
            self.GROUPS,
        )

        assert ">FAIR<" in svg
        assert self._ratio(svg) == "2/2"
        assert "All ranking fairness metrics pass" in svg
        assert "not graded" not in svg
