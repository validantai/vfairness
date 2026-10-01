"""Row-level guards for fabricated BREACHES and for ungraded rows inside counts.

The chart-level campaign closed the fabricated all-clear. This file pins the
mirror-image defect and the counting defect that survived it, on the five
surfaces where a row that reported nothing was being turned into a finding:

* ``group_comparison_to_svg`` (adapters_fairness) rendered "Max disparity 0.970
  between ZZ_BARE and Male" at severity HIGH for two groups that are 0.06 apart.
  A group with no rate defaulted to 0, which became the extreme of the max-min
  gap AND sorted to the front of the best/worst ranking, so the chart named a
  group that reported nothing as the best-served party of a breach that does not
  exist.
* ``cicd_pipeline_to_svg`` (adapters) turned a check whose status was None into
  a red FAIL, counted it into "1 passed, 2 failed", and withheld the pipeline
  all-clear on it. A gate that blocks on a check which never ran is the same
  class of defect as a gate that approves one, and it is the kind a reader
  eventually catches: they chase the failure, find no such breach, and stop
  believing the report.
* the ``fairness_report`` group panel (adapters._group_bars) plotted a group with
  no rate at 0 and labelled it "0.0%" beside "n=0", which on a positive-rate
  panel reads as a group the model never selects.
* ``templates/hierarchical_gate.svg`` graded each level and each metric row
  PASS-or-else-FAIL, so ``approved is None`` and ``passed is None`` were painted
  red, and the level badge counted the ungraded row into its denominator
  ("1/2 metrics" beside a level where one metric was measured).
* ``templates/report_card.svg`` rendered "PASS RATE 1/2" beside DEPLOYMENT
  APPROVED and badged the ungraded row a blocking FAIL, on the artifact that is
  pasted into a pull request.

Two directions, one rule. A row that reported nothing gets no number, no badge,
no colour, no point on a plot, and no place in any count or headline that implies
it was measured. A default is not a measurement, a sentinel is not a
measurement, and absent and zero are different claims.

THE HEADLINE RULE these tests encode, decided 2026-08-28 because two adapters had
drifted apart on it:

a. The grade and the band are taken over the GRADED subset, so a partly measured
   run still gives a useful verdict on what WAS measured.
b. An unqualified all-clear ("ALL PASS", "all N metrics pass") is forbidden while
   anything is ungraded.
c. The ungraded count is stated ON THE CANVAS next to the headline, not only in
   the accessible description.
d. An ungraded row enters neither a numerator nor a denominator.

Every control here is a negative case: a genuine failure must still render as a
failure, in the failure red, or the fix has traded one lie for another.
"""

import re

import pytest

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.rendering.adapters import (  # noqa: E402
    cicd_pipeline_to_svg,
    fairness_report_to_svg,
)
from vfairness.rendering.adapters_fairness import (  # noqa: E402
    group_comparison_to_svg,
)
from vfairness.rendering.adapters_workflow import (  # noqa: E402
    hierarchical_gate_to_svg,
    report_card_to_svg,
)
from vfairness.rendering.skins import BLANCO_MAP  # noqa: E402


def _skinned(hex_color: str) -> str:
    """The colour as it survives the Blanco skin, which rewrites the palette.

    Asserting on the raw template colour passes vacuously: the skin has already
    rewritten it by the time the markup exists.
    """
    return BLANCO_MAP.get(hex_color, hex_color)


_FAIL_FILL = _skinned("#dc2626")
_FAIL_BG = _skinned("#fef2f2")
_PASS_FILL = _skinned("#059669")
_UNCHECKED_FILL = _skinned("#64748b")

# Every value bar on both group panels is drawn at this opacity and nothing else
# is, so counting it counts the rows that were given a position on the axis.
_BAR_MARK = 'opacity="0.75"'


def _canvas_only(svg: str) -> str:
    """The drawn canvas, with the accessibility layer stripped.

    ``<desc>``/``<metadata>`` are built by ``rendering.explain`` from the same
    data dict but by a separate finder, so a claim there is a different defect
    from a claim on the canvas and must not silently satisfy a canvas test.
    """
    svg = re.sub(r"<desc\b.*?</desc>", "", svg, flags=re.S)
    svg = re.sub(r"<metadata\b.*?</metadata>", "", svg, flags=re.S)
    return re.sub(r"<title\b.*?</title>", "", svg, flags=re.S)


# The explanation paragraph is drawn ON the canvas by rendering.explain, which
# is a SEPARATE set of finders reading the same data dict (see
# TestExplanationLayerHonoursTheThirdState at the bottom). Suppressing it with
# explanation="" keeps these tests measuring the chart's own markup rather than
# that finder's sentence, so a claim in one layer cannot silently satisfy a test
# about the other. Both layers honour the rule now; they are still pinned apart.
NO_EXPL = ""


# Fixtures


def _report(*, bare_group=False, gap=0.06):
    """A healthy two-group report, optionally with one group that reported nothing."""
    groups = {
        "Male": {"positive_rate": 0.970, "size": 480},
        "Female": {"positive_rate": round(0.970 - gap, 3), "size": 460},
    }
    if bare_group:
        groups["ZZ_BARE"] = {}
    return {
        "protected_attribute": "gender",
        "task_type": "classification",
        "group_stats": groups,
        "metrics": {"demographic_parity_difference": gap},
        "assessment": {"assessable": True, "fairness_score": 0.9, "summary": "Groups compared."},
    }


class _Status:
    def __init__(self, value):
        self.value = value


class _Test:
    def __init__(self, name, status, value=0.04, threshold=0.10):
        self.test_name = name
        self.status = status
        self.metric_name = "demographic_parity_difference"
        self.metric_value = value
        self.threshold = threshold


class _Validation:
    passed = True
    errors = []
    warnings = []


class _Gate:
    status = _Status("approved")
    approved = True
    blocking_reasons = []
    warnings = []


class _Eval:
    def __init__(self, name, value, threshold, passed, is_blocking=True, baseline=None):
        self.metric_name = name
        self.value = value
        self.threshold = threshold
        self.passed = passed
        self.is_blocking = is_blocking
        self.baseline_value = baseline


class _LevelDecision:
    def __init__(self, approved, evaluations):
        self.approved = approved
        self.metric_evaluations = evaluations


class _Decision:
    def __init__(self, approved, level_results):
        self.approved = approved
        self.level_results = level_results
        self.small_sample_warnings = []
        self.hierarchical_config = None


class _CardDecision:
    def __init__(self, approved, evaluations, status="approved", blocking=()):
        self.approved = approved
        self.metric_evaluations = evaluations
        self.status = _Status(status)
        self.blocking_reasons = list(blocking)
        self.warnings = []


def _graded_card(*, ungraded):
    metrics = [_Eval("demographic_parity_difference", 0.042, 0.10, True)]
    if ungraded:
        metrics.append(_Eval("equalized_odds_difference", None, 0.10, None))
    return _CardDecision(True, metrics)


# 1.  group_comparison: the fabricated breach


class TestGroupComparisonFabricatedBreach:
    """A group that reported no rate is not one end of a disparity."""

    def test_the_gap_is_not_measured_from_a_group_that_reported_nothing(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True), explanation=NO_EXPL)
        )

        assert "0.970" not in canvas, "the absent group's 0 default became the disparity"
        assert "High disparity" not in canvas
        assert "0.060" in canvas, "the gap between the two MEASURED groups is still reported"
        assert "Low disparity" in canvas

    def test_the_unmeasured_group_is_not_named_as_the_breach_operand(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True), explanation=NO_EXPL)
        )

        assert "between ZZ_BARE" not in canvas
        assert "and ZZ_BARE" not in canvas
        assert "between Female and Male" in canvas

    def test_the_unmeasured_row_gets_no_bar_no_number_and_no_count(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True), explanation=NO_EXPL)
        )

        assert canvas.count(_BAR_MARK) == 2, "a row that reported nothing was given a bar"
        assert "0%" not in canvas
        assert "n=0" not in canvas
        assert "not measured" in canvas
        assert "n not reported" in canvas

    def test_the_ungraded_count_is_on_the_canvas_beside_the_headline(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True), explanation=NO_EXPL)
        )

        # Headline rule (c): next to the badge, not only in the description.
        assert "1 group not measured" in canvas
        assert "ZZ_BARE" in canvas, "the group that was skipped must still be named"

    def test_the_average_line_excludes_the_group_that_reported_nothing(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True), explanation=NO_EXPL)
        )

        # (0.97 + 0.91) / 2 = 0.94. Including a fabricated 0 gave 62%.
        assert "Avg: 94%" in canvas
        assert "Avg: 62%" not in canvas

    # Controls

    def test_a_healthy_two_group_report_is_unchanged(self):
        canvas = _canvas_only(group_comparison_to_svg(_report(), explanation=NO_EXPL))

        assert canvas.count(_BAR_MARK) == 2
        assert "Low disparity" in canvas and "0.060" in canvas
        assert "between Female and Male" in canvas
        assert "not measured" not in canvas
        assert "not in this gap" not in canvas

    def test_a_real_high_disparity_is_still_reported_as_high(self):
        canvas = _canvas_only(group_comparison_to_svg(_report(gap=0.31), explanation=NO_EXPL))

        assert "High disparity" in canvas
        assert "0.310" in canvas
        assert _FAIL_FILL in canvas, "a measured breach must keep the failure red"

    def test_a_real_high_disparity_survives_an_unmeasured_group_alongside_it(self):
        canvas = _canvas_only(
            group_comparison_to_svg(_report(bare_group=True, gap=0.31), explanation=NO_EXPL)
        )

        # Headline rule (a): the graded subset still produces its verdict.
        assert "High disparity" in canvas and "0.310" in canvas
        # Headline rule (c): and it is qualified on the canvas.
        assert "1 group not measured" in canvas

    def test_a_report_where_nothing_reported_a_rate_bands_no_gap_at_all(self):
        report = _report()
        report["group_stats"] = {"Male": {}, "Female": {}}
        canvas = _canvas_only(group_comparison_to_svg(report, explanation=NO_EXPL))

        assert "NOT ASSESSABLE" in canvas
        assert "disparity:" not in canvas
        assert canvas.count(_BAR_MARK) == 0
        assert canvas.count(">· not measured</text>") == 2
        assert "2 groups not measured, not in this gap" in canvas

    def test_a_genuine_zero_rate_is_still_plotted_as_a_measurement(self):
        report = _report()
        report["group_stats"]["Female"] = {"positive_rate": 0.0, "size": 460}
        canvas = _canvas_only(group_comparison_to_svg(report, explanation=NO_EXPL))

        # 0.0 was MEASURED here. Suppressing it would be the mirror-image defect.
        assert canvas.count(_BAR_MARK) == 2
        assert "not measured" not in canvas
        assert "High disparity" in canvas


# 2.  cicd_pipeline: the fabricated failure


class TestCicdPipelineUngradedCheck:
    """A check whose status never arrived did not fail; it did not run."""

    def _svg(self):
        return _canvas_only(
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[
                    _Test("demographic parity", _Status("passed")),
                    _Test("equalized odds", None),
                    _Test("equal opportunity", _Status("skipped"), value=None, threshold=None),
                ],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        )

    def test_a_check_with_no_status_is_not_badged_as_a_failure(self):
        canvas = self._svg()

        assert ">FAIL<" not in canvas, "a check that never ran was reported as failing"
        assert canvas.count(">NOT RUN<") == 2
        assert _FAIL_BG not in canvas, "the failure row wash was painted for an ungraded row"

    def test_the_ungraded_checks_are_not_counted_as_failures(self):
        canvas = self._svg()

        assert "1 passed, 2 failed" not in canvas
        assert "1 passed, 0 failed, 2 not run" in canvas

    def test_the_executed_count_counts_only_the_checks_that_ran(self):
        canvas = self._svg()

        assert "1 test executed, 2 not executed" in canvas
        assert "3 tests executed" not in canvas

    def test_the_tile_reports_a_pass_without_claiming_all_of_them(self):
        canvas = self._svg()

        # Headline rule (a): the graded check passed, so the tile is green.
        assert ">PASS<" in canvas
        # Headline rule (b): but two checks were never graded, so not "ALL PASS".
        assert "ALL PASS" not in canvas
        assert "FAILURES" not in canvas

    def test_the_pipeline_all_clear_is_withheld_while_a_check_is_ungraded(self):
        from vfairness.rendering import adapters

        captured = {}
        original = adapters.render_svg

        def _spy(template_name, data, *args, **kwargs):
            captured.update(data)
            return original(template_name, data, *args, **kwargs)

        adapters.render_svg = _spy
        try:
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[
                    _Test("demographic parity", _Status("passed")),
                    _Test("equalized odds", None),
                ],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        finally:
            adapters.render_svg = original

        assert captured["all_passed"] is False
        assert captured["n_tests_ungraded"] == 1
        assert captured["n_tests_graded"] == 1

    # Controls

    def test_a_check_that_really_failed_is_still_a_failure(self):
        canvas = _canvas_only(
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[
                    _Test("demographic parity", _Status("passed")),
                    _Test("equalized odds", _Status("failed"), value=0.31),
                ],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        )

        assert ">FAIL<" in canvas
        assert "FAILURES" in canvas
        assert "1 passed, 1 failed" in canvas
        assert _FAIL_FILL in canvas

    def test_an_errored_check_still_blocks(self):
        canvas = _canvas_only(
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[_Test("equalized odds", _Status("error"), value=None)],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        )

        # An errored check RAN and the run broke. That is a fact about this
        # pipeline, not an absence, so it is deliberately not the third state.
        assert ">FAIL<" in canvas
        assert ">NOT RUN<" not in canvas

    def test_an_all_passing_suite_is_unchanged(self):
        canvas = _canvas_only(
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[
                    _Test("demographic parity", _Status("passed")),
                    _Test("equalized odds", _Status("passed")),
                ],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        )

        assert "ALL PASS" in canvas
        assert "2 passed, 0 failed" in canvas
        assert "2 tests executed" in canvas
        assert "not run" not in canvas

    def test_a_suite_whose_every_check_is_ungraded_reports_that_nothing_ran(self):
        canvas = _canvas_only(
            cicd_pipeline_to_svg(
                validation_result=_Validation(),
                test_results=[_Test("equalized odds", None)],
                gate_decision=_Gate(),
                explanation=NO_EXPL,
            )
        )

        assert ">NOT RUN<" in canvas
        assert "no test executed" in canvas
        assert "0 passed" not in canvas, "zero of zero is not a measurement"


# 3.  fairness_report group panel


class TestFairnessReportGroupPanel:
    """A group with no rate is not plotted at zero, and is not lost either."""

    def test_a_group_with_no_rate_is_not_drawn_as_never_selected(self):
        canvas = _canvas_only(fairness_report_to_svg(_report(bare_group=True), explanation=NO_EXPL))

        assert canvas.count(_BAR_MARK) == 2, "a group that reported nothing was given a bar"
        assert "0.0%" not in canvas
        assert "n=0" not in canvas

    def test_the_group_that_could_not_be_plotted_is_named_on_the_canvas(self):
        canvas = _canvas_only(fairness_report_to_svg(_report(bare_group=True), explanation=NO_EXPL))

        # A silently dropped stratum is its own fabrication: the reader is left
        # believing every group was looked at.
        assert "ZZ_BARE" in canvas
        assert "no rate reported" in canvas

    # Controls

    def test_a_healthy_report_plots_every_group(self):
        canvas = _canvas_only(fairness_report_to_svg(_report(), explanation=NO_EXPL))

        assert canvas.count(_BAR_MARK) == 2
        assert "97.0%" in canvas and "91.0%" in canvas
        assert "no rate reported" not in canvas

    def test_a_genuine_zero_rate_group_is_still_plotted(self):
        report = _report()
        report["group_stats"]["Female"] = {"positive_rate": 0.0, "size": 460}
        canvas = _canvas_only(fairness_report_to_svg(report, explanation=NO_EXPL))

        assert canvas.count(_BAR_MARK) == 2
        assert "0.0%" in canvas
        assert "no rate reported" not in canvas


# 4.  hierarchical_gate: levels and metric rows


class TestHierarchicalGateUngradedRows:
    """PASS or else FAIL swallowed the level and the metric nobody graded."""

    def _svg(self):
        return _canvas_only(
            hierarchical_gate_to_svg(
                _Decision(
                    True,
                    {
                        "overall": _LevelDecision(
                            None,
                            [
                                _Eval("demographic_parity_difference", 0.04, 0.10, True),
                                _Eval("equalized_odds_difference", None, 0.10, None),
                            ],
                        )
                    },
                ),
                explanation=NO_EXPL,
            )
        )

    def test_an_ungraded_metric_row_is_not_badged_as_a_failure(self):
        canvas = self._svg()

        assert ">FAIL<" not in canvas, "a metric nobody graded was reported as failing"
        assert ">NOT CHECKED<" in canvas
        assert _FAIL_BG not in canvas

    def test_an_ungraded_metric_row_gets_no_number(self):
        canvas = self._svg()

        assert "not measured" in canvas
        assert "N/A" not in canvas, "N/A in the failure red reads as a lost measurement"

    def test_a_level_with_no_verdict_is_not_painted_as_blocked(self):
        canvas = self._svg()

        # The level panel's own badge and its left stripe.
        assert _FAIL_FILL not in canvas
        assert _UNCHECKED_FILL in canvas

    def test_the_ungraded_row_is_out_of_the_level_denominator(self):
        canvas = self._svg()

        assert "1/2 metrics" not in canvas, "an ungraded row took a place in the denominator"
        assert "1/1 metrics, 1 not checked" in canvas

    # Controls

    def test_a_level_that_was_evaluated_and_failed_is_still_a_failure(self):
        canvas = _canvas_only(
            hierarchical_gate_to_svg(
                _Decision(
                    False,
                    {
                        "overall": _LevelDecision(
                            False, [_Eval("demographic_parity_difference", 0.31, 0.10, False)]
                        )
                    },
                ),
                explanation=NO_EXPL,
            )
        )

        assert ">FAIL<" in canvas
        assert "0/1 metrics" in canvas
        assert "BLOCKED" in canvas
        assert _FAIL_FILL in canvas

    def test_a_fully_graded_gate_is_unchanged(self):
        canvas = _canvas_only(
            hierarchical_gate_to_svg(
                _Decision(
                    True,
                    {
                        "overall": _LevelDecision(
                            True, [_Eval("demographic_parity_difference", 0.04, 0.10, True)]
                        ),
                        "attr:gender": _LevelDecision(
                            True, [_Eval("equalized_odds_difference", 0.06, 0.10, True)]
                        ),
                    },
                ),
                explanation=NO_EXPL,
            )
        )

        assert "1/1 metrics" in canvas and "1/1 checks" in canvas
        assert "not checked" not in canvas.split("INTERSECTIONAL", 1)[0]
        assert ">PASS<" in canvas
        assert "APPROVED" in canvas

    def test_a_level_with_no_metric_at_all_still_reads_not_checked(self):
        canvas = _canvas_only(
            hierarchical_gate_to_svg(
                _Decision(True, {"overall": _LevelDecision(True, [])}), explanation=NO_EXPL
            )
        )

        summary = canvas.split("Overall (all groups)", 1)[0]
        assert ">FAIL<" not in summary
        assert summary.count(">NOT CHECKED<") == 3
        assert "0/0 metrics" not in summary


# 5.  report_card: the pass rate beside the approval


class TestReportCardPassRateDenominator:
    """A denominator is a claim that that many checks happened."""

    def _svg(self):
        return _canvas_only(
            report_card_to_svg(
                decision=_graded_card(ungraded=True),
                model_name="risk_model",
                explanation=NO_EXPL,
            )
        )

    def test_an_ungraded_metric_has_no_place_in_the_pass_rate(self):
        canvas = self._svg()

        assert "1/2" not in canvas, "PASS RATE 1/2 was printed beside DEPLOYMENT APPROVED"
        assert ">1/1<" in canvas

    def test_the_ungraded_count_is_stated_beside_the_pass_rate(self):
        canvas = self._svg()

        assert "1 metric not graded, not in this rate" in canvas
        assert "1 evaluated" in canvas and "of 2 listed" in canvas
        assert "2 evaluated" not in canvas

    def test_the_ungraded_row_is_not_badged_as_a_blocking_failure(self):
        canvas = self._svg()

        assert ">FAIL<" not in canvas
        assert ">NOT GRADED<" in canvas
        assert _FAIL_FILL not in canvas
        assert "not measured" in canvas

    def test_the_recommendation_says_what_the_verdict_does_not_cover(self):
        canvas = self._svg()

        # Headline rule (b): an approval next to an ungraded row must not read
        # as an approval of the whole card.
        assert "were not graded" in canvas
        assert "says nothing about the rest" in canvas

    # Controls

    def test_a_fully_graded_approved_card_is_unchanged(self):
        canvas = _canvas_only(
            report_card_to_svg(
                decision=_graded_card(ungraded=False),
                model_name="risk_model",
                explanation=NO_EXPL,
            )
        )

        assert ">1/1<" in canvas
        assert "1 evaluated" in canvas
        assert "of 1 listed" not in canvas
        assert "not graded" not in canvas
        assert "APPROVED" in canvas

    def test_a_card_with_a_real_blocking_failure_is_unchanged(self):
        canvas = _canvas_only(
            report_card_to_svg(
                decision=_CardDecision(
                    False,
                    [
                        _Eval("demographic_parity_difference", 0.42, 0.10, False),
                        _Eval("equalized_odds_difference", 0.065, 0.10, True),
                    ],
                    status="blocked",
                    blocking=["demographic_parity_difference 0.42 exceeds 0.10"],
                ),
                model_name="risk_model",
                explanation=NO_EXPL,
            )
        )

        assert ">FAIL<" in canvas
        assert ">1/2<" in canvas, "both metrics WERE graded here"
        assert "BLOCKED" in canvas
        assert _FAIL_FILL in canvas

    def test_a_card_with_no_decision_still_reads_not_checked(self):
        canvas = _canvas_only(report_card_to_svg(explanation=NO_EXPL))

        assert "NOT CHECKED" in canvas
        assert "not measured" in canvas
        assert "APPROVED" not in canvas


# 6.  The accessibility layer agrees with the canvas


class TestExplanationLayerHonoursTheThirdState:
    """CLOSED, and these are the pins that keep it closed.

    The paragraph and the accessible ``<desc>`` are written by a separate set of
    finders in ``rendering/explain.py``. While those finders were still
    two-state, a screen-reader user was told the old story by the same SVG whose
    picture told the new one, and the three tests below carried
    ``xfail(strict=False)`` markers recording that.

    WHY THE MARKERS ARE GONE (2026-08-28). Waves 14 and 15 fixed all three
    defects (register entries N-02, N-03, N-04), and nobody cleared the markers.
    A non-strict xfail over a passing test is inert in BOTH directions: it does
    not fail now that the defect is gone, and it will not fail if the defect
    comes back. Measured: with ``explain._fr_cicd``'s third-state branch
    reinstated as the old two-state one, this whole file stayed GREEN, and a
    DIFFERENT file (``test_explain_partial_runs.py``) was the only thing that
    noticed. The file that claimed to record the defect contributed nothing.

    That is the N-22 entry in ``docs/audits/row-level-fabrication-register-
    2026-08-27.md``, and it was worse than a dead marker: this class was named
    ``TestExplanationLayerStillLags``, its docstring said the finders "are still
    two-state", and each marker's ``reason`` described a live fabrication. All
    of that is false today, and ``docs/audits/`` is on the export EXCLUDES list,
    so the public repo would have shipped the three misleading claims without the
    register that explains them.

    As ordinary tests they are real pins: reinstate any of the three defects and
    the corresponding test goes red here, which also settles N-02, N-03 and N-04
    as PINNED rather than merely fixed.

    DEFERRED, and recorded here so it is not merely mentioned: the structural
    half of N-22 is ``xfail_strict = true`` under ``[tool.pytest.ini_options]``
    in ``pyproject.toml``, which is what stops the NEXT non-strict marker from
    outliving its defect the same way. That file was outside this change's scope
    (2026-08-28, audit lane F5/F10). Gating condition: none, it is a one-line
    setting; the whole tree currently has exactly one xfail left
    (``tests/test_desc_agrees_with_canvas.py``, W-28) and it is already strict,
    so the setting can be turned on without changing any other result.
    """

    def test_the_cicd_description_does_not_call_a_partial_run_a_failure(self):
        """N-02: ``_fr_cicd`` read only ``all_passed``, which wave 13 made False
        whenever a check is ungraded, so the ``<desc>`` said "Pipeline failed"
        for a pipeline where every check that ran passed and the gate approved.
        """
        svg = cicd_pipeline_to_svg(
            validation_result=_Validation(),
            test_results=[
                _Test("demographic parity", _Status("passed")),
                _Test("equalized odds", None),
            ],
            gate_decision=_Gate(),
        )

        assert "Pipeline failed" not in svg

    def test_the_report_card_description_uses_the_graded_denominator(self):
        """N-03: ``_fr_report_card`` wrote ``N/len(metrics) metrics pass``, so
        the ungraded row wave 13 took out of the CANVAS denominator was still in
        the ``<desc>`` denominator: canvas 1/1, description 1/2.
        """
        svg = report_card_to_svg(decision=_graded_card(ungraded=True), model_name="risk_model")

        assert "1/2 metrics pass" not in svg

    def test_the_hierarchical_description_uses_the_graded_denominator(self):
        """N-04: the same defect one level up. ``_fr_hierarchical`` wrote
        ``n_overall_pass/n_overall_total`` straight from the adapter, and that
        total still counted the ungraded row: canvas "1/1 metrics, 1 not
        checked", ``<desc>`` "overall 1/2".
        """
        svg = hierarchical_gate_to_svg(
            _Decision(
                True,
                {
                    "overall": _LevelDecision(
                        None,
                        [
                            _Eval("demographic_parity_difference", 0.04, 0.10, True),
                            _Eval("equalized_odds_difference", None, 0.10, None),
                        ],
                    )
                },
            )
        )

        assert "overall 1/2" not in svg
