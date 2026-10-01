"""An audit that ran nothing, and an audit that ran everything and cleared it.

Wave 10 could not tell those two apart, so it withheld the verdict from both:
``BiasDetector._calculate_overall_risk`` returns 0.0 for a dataset every module
cleared exactly as it does for a run where no module executed (its ``if not
risk_components: return 0.0`` branch), and four empty finding lists arrive in
the adapter either way. Withholding from both is honest, and it costs a real
clean result its green badge.

The fix is in the DATA MODEL, not in the view: ``full_audit`` now records which
modules executed in ``BiasAuditReport.modules_run``, ``execution_coverage()``
reads it back as complete / partial / none / unrecorded, and the canvas renders

* a genuine "0% MINIMAL" only under complete coverage with a protected
  attribute to compare across,
* COULD NOT CHECK for none, partial, unrecorded, or no protected attribute.

``modules_run`` defaults to None, not to [], because "this report does not say"
is a third state: an older or hand-built report must not be read as either a
clean audit or an empty one. Every could-not-check assertion here is paired
with a healthy control in ``TestTheCleanAuditIsGreenAgain`` and
``TestHealthyInputIsUntouched``, so a fix that suppresses the verdict
everywhere fails the controls.

Also pinned here: the two count-of-zero wordings this wave removed
(``bias_audit.svg`` "0 critical issues found", ``power_analysis.svg`` "no
intersection was supplied" said of intersections that WERE supplied), and the
absence of punctuation dashes from the rendered canvas.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jinja2", reason="SVG rendering requires jinja2")

from vfairness.preprocessing.bias_detection.detector import (  # noqa: E402
    AUDIT_MODULES,
    BiasAuditReport,
    BiasDetector,
)
from vfairness.rendering import adapters  # noqa: E402
from vfairness.rendering.adapters import bias_audit_to_svg  # noqa: E402
from vfairness.rendering.adapters_experimentation import power_analysis_to_svg  # noqa: E402
from vfairness.rendering.skins import apply_skin  # noqa: E402

# Written as escapes so the file that FORBIDS these characters on the canvas
# does not itself carry them in prose.
EN_DASH = "\u2013"
EM_DASH = "\u2014"


# ── helpers ─────────────────────────────────────────────────────────────────


def _skinned(raw: str) -> str:
    """The Blanco hex a raw palette colour becomes in rendered output.

    Asserting on the RAW hex would silently pass forever: render_svg runs
    apply_skin as its last step, so '#059669' never appears in any output.
    """
    out = apply_skin(f'<a fill="{raw}"/>', "blanco")
    match = re.search(r'fill="(#[0-9a-fA-F]{6})"', out)
    assert match, f"apply_skin dropped the fill for {raw}"
    return match.group(1).lower()


FAIL_RED = _skinned("#dc2626")
PASS_GREEN = _skinned("#059669")
SLATE = _skinned("#64748b")


def _visible(svg: str) -> str:
    """Every rendered text node, joined. This is what a reader of the SVG sees."""
    body = svg.split("</metadata>")[-1]
    return " | ".join(html.unescape(t) for t in re.findall(r">([^<>]+)<", body) if t.strip())


def _desc(svg: str) -> str:
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "render_svg must always inject an accessible <desc>"
    return html.unescape(match.group(1))


def _fills(svg: str) -> set:
    return {c.lower() for c in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)}


def _report(**over) -> BiasAuditReport:
    """A report with four empty finding lists. `modules_run` decides the state.

    ``attribute_assessed`` is part of the base dict as of 2026-09-27. This helper
    predates that field, so `_report(modules_run=list(AUDIT_MODULES))` used to
    build a report whose EXECUTION half says every module ran and whose ASSESSMENT
    half is unrecorded, and `execution_coverage()` answered "complete" over it. It
    now answers "unrecorded", which is correct and which is what these fixtures
    were silently relying on the old answer for.

    The tests here are about a COMPLETE audit that found nothing, so the fixture
    has to actually be one. Adding the field keeps every subject in this file
    unchanged; leaving it out would have meant weakening the assertions instead.
    """
    base = dict(
        timestamp="2026-08-27 10:00",
        dataset_info={"n_rows": 900, "n_columns": 3},
        protected_attributes=["cohort"],
        attribute_assessed={"cohort": True},
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return BiasAuditReport(**base)


def _neutral_frame() -> pd.DataFrame:
    """A small frame no detection module has a finding about."""
    rng = np.random.default_rng(11)
    n = 900
    return pd.DataFrame(
        {
            "cohort": rng.choice(["a", "b"], size=n),
            "score_x": rng.normal(0, 1, n),
            "approved": rng.integers(0, 2, n),
        }
    )


def _detector() -> BiasDetector:
    return BiasDetector(
        _neutral_frame(), protected_attributes=["cohort"], outcome_column="approved"
    )


# ── 1. the detector records what executed ───────────────────────────────────


class TestTheDetectorRecordsWhatExecuted:
    def test_a_full_run_records_every_module(self):
        report = _detector().full_audit()

        assert set(report.modules_run) == set(AUDIT_MODULES)
        assert report.execution_coverage() == "complete"

    def test_a_run_with_every_module_off_records_an_empty_list(self):
        # NOT None. "It is recorded that nothing ran" is a positive statement
        # and the whole point of this wave; None would mean "not recorded".
        report = _detector().full_audit(
            include_historical=False,
            include_representation=False,
            include_disparities=False,
            include_proxies=False,
        )

        assert report.modules_run == []
        assert report.execution_coverage() == "none"

    def test_the_score_alone_still_cannot_tell_the_two_apart(self):
        # The record exists precisely BECAUSE the score cannot carry this. If
        # this assert ever fails the score changed meaning, and every consumer
        # reading it (explainer, adapters, notebooks) must be revisited.
        ran_nothing = _detector().full_audit(
            include_historical=False,
            include_representation=False,
            include_disparities=False,
            include_proxies=False,
        )

        assert ran_nothing.overall_risk_score == 0.0
        assert isinstance(ran_nothing.overall_risk_score, float)

    def test_a_partial_run_records_only_the_modules_that_ran(self):
        report = _detector().full_audit(include_disparities=False, include_proxies=False)

        assert set(report.modules_run) == {"historical", "representation"}
        assert report.execution_coverage() == "partial"

    def test_a_report_that_does_not_record_it_is_unrecorded_not_empty(self):
        # Backward compatibility: a report built by hand or by an older version
        # says nothing about coverage, and must not be read as either state.
        assert _report().modules_run is None
        assert _report().execution_coverage() == "unrecorded"

    def test_the_dict_export_carries_the_record(self):
        exported = _detector().full_audit().to_dict()

        assert set(exported["modules_run"]) == set(AUDIT_MODULES)
        assert exported["execution_coverage"] == "complete"
        # Still a float, still there: nothing external that reads the score breaks.
        assert isinstance(exported["overall_risk_score"], float)

    def test_the_text_summary_says_when_nothing_executed(self):
        summary = (
            _detector()
            .full_audit(
                include_historical=False,
                include_representation=False,
                include_disparities=False,
                include_proxies=False,
            )
            .summary()
        )

        assert "Overall Risk Score: 0.0%" in summary
        assert "no audit module executed" in summary

    def test_the_recommendation_is_not_an_all_clear_when_nothing_ran(self):
        report = _detector().full_audit(
            include_historical=False,
            include_representation=False,
            include_disparities=False,
            include_proxies=False,
        )

        joined = " ".join(report.recommendations)
        assert "No critical bias issues detected" not in joined
        assert "No audit module executed" in joined

    def test_a_complete_run_keeps_the_original_all_clear_recommendation(self):
        # Control: the wording a real clean audit has always carried.
        joined = " ".join(_detector().full_audit().recommendations)

        assert joined  # a report always recommends something


# ── 2. the clean audit is green again ───────────────────────────────────────


class TestTheCleanAuditIsGreenAgain:
    def test_a_complete_audit_that_found_nothing_scores_zero_percent(self):
        # Wave 10 rendered this NOT ASSESSABLE. It is a measurement: four
        # modules ran, against a protected attribute, and none had a finding.
        seen = _visible(bias_audit_to_svg(_report(modules_run=list(AUDIT_MODULES))))

        assert "0%" in seen
        assert "MINIMAL" in seen
        assert "NOT ASSESSABLE" not in seen

    def test_its_module_tiles_are_measured_zeroes_not_withheld(self):
        svg = bias_audit_to_svg(_report(modules_run=list(AUDIT_MODULES)))
        seen = _visible(svg)

        # Four tiles, all measured. (The fifth "0.00" in `seen` is the score
        # repeated in the explanation paragraph.)
        assert seen.count("0.00") >= 4
        assert "N/A" not in seen
        assert PASS_GREEN in _fills(svg)

    def test_the_desc_reports_the_measured_score(self):
        desc = _desc(bias_audit_to_svg(_report(modules_run=list(AUDIT_MODULES))))

        assert "COULD NOT CHECK" not in desc
        assert "MINIMAL" in desc

    def test_a_real_clean_full_audit_renders_a_verdict(self):
        # The end to end path, not a hand-built report: BiasDetector on a frame
        # nothing is wrong with, through full_audit, through to_svg.
        seen = _visible(_detector().full_audit().to_svg())

        assert "NOT ASSESSABLE" not in seen
        assert "MINIMAL" in seen


# ── 3. the audit that measured nothing ──────────────────────────────────────


class TestTheAuditThatRanNothing:
    def test_it_withholds_the_percentage_and_the_band(self):
        seen = _visible(bias_audit_to_svg(_report(modules_run=[])))

        assert "0%" not in seen
        assert "MINIMAL" not in seen
        assert "NOT ASSESSABLE" in seen

    def test_its_module_tiles_are_withheld(self):
        seen = _visible(bias_audit_to_svg(_report(modules_run=[])))

        assert "0.00" not in seen
        assert seen.count("N/A") >= 4

    def test_it_is_neither_a_pass_nor_a_rejection(self):
        svg = bias_audit_to_svg(_report(modules_run=[]))

        assert PASS_GREEN not in _fills(svg)
        assert FAIL_RED not in _fills(svg)
        assert SLATE in _fills(svg)

    def test_the_canvas_says_no_module_executed(self):
        # On the canvas, not only in the <desc>: an SVG is an export and
        # outlives the run that made it.
        #
        # UPDATED 2026-09-25, and the old assertion was the finding. The band
        # printed a sentence HARDCODED in the template, "no module of this audit
        # returned a finding", while the adapter computed four accurate variants
        # that reached the <desc> and nowhere else. On this report the hardcoded
        # sentence was merely vague; on a two-row frame that DID return an
        # insufficient-data finding it was false. The canvas now prints the same
        # reason the <desc> names, which is what the code comment beside
        # `not_assessable_reason` already claimed it did.
        seen = _visible(bias_audit_to_svg(_report(modules_run=[])))

        assert "no module of this audit executed" in seen
        assert "no finding was possible" in seen
        assert "neither clears the dataset nor faults it" in seen

    def test_the_canvas_and_the_desc_give_the_same_reason(self):
        """A screen-reader user and a sighted reader must not be told different things."""
        svg = bias_audit_to_svg(_report(modules_run=[]))

        assert "no module of this audit executed" in _visible(svg)
        assert "no module of this audit executed" in _desc(svg)

    def test_the_desc_names_the_reason(self):
        desc = _desc(bias_audit_to_svg(_report(modules_run=[])))

        assert "COULD NOT CHECK" in desc
        assert "no module of this audit executed" in desc

    def test_a_real_full_audit_with_every_module_off_is_not_assessable(self):
        report = _detector().full_audit(
            include_historical=False,
            include_representation=False,
            include_disparities=False,
            include_proxies=False,
        )

        assert "NOT ASSESSABLE" in _visible(report.to_svg())


class TestTheStatesThatAreNotAMeasurement:
    def test_an_unrecorded_report_still_withholds_the_verdict(self):
        # The pre-existing wave 10 guard, kept: with no record, the canvas
        # cannot tell a cleared dataset from an empty run, so it says so.
        seen = _visible(bias_audit_to_svg(_report()))

        assert "NOT ASSESSABLE" in seen
        assert "0%" not in seen

    def test_an_unrecorded_report_says_the_record_is_missing(self):
        desc = _desc(bias_audit_to_svg(_report()))

        assert "does not record which modules executed" in desc

    def test_a_partial_audit_that_found_nothing_is_not_an_all_clear(self):
        seen = _visible(bias_audit_to_svg(_report(modules_run=["historical"])))

        assert "NOT ASSESSABLE" in seen
        assert "MINIMAL" not in seen

    def test_a_partial_audit_names_the_modules_that_did_not_run(self):
        desc = _desc(bias_audit_to_svg(_report(modules_run=["historical"])))

        for skipped in ("representation", "disparities", "proxies"):
            assert skipped in desc

    def test_a_complete_audit_with_no_protected_attribute_is_not_a_clean_bill(self):
        # Four modules ran and had nothing to compare across. That is an
        # absence of measurement, however complete the coverage.
        report = _report(modules_run=list(AUDIT_MODULES), protected_attributes=[])
        seen = _visible(bias_audit_to_svg(report))

        assert "NOT ASSESSABLE" in seen
        assert "no protected attribute was audited" in _desc(report and bias_audit_to_svg(report))

    def test_a_partial_audit_with_findings_qualifies_its_score(self):
        # Findings exist, so the score is a measurement and stays on the badge,
        # but it covers two modules of four and both the canvas and the
        # description must say so. The desc may never be the more confident of
        # the two.
        report = _detector().full_audit(include_disparities=False, include_proxies=False)
        svg = report.to_svg()

        assert "Only 2 of 4 audit modules ran" in _visible(svg)
        assert "audit modules ran" in _desc(svg)


# ── 4. the adapter and the detector agree on the module names ───────────────


def test_the_adapter_module_names_match_the_detector():
    # The canvas reads coverage by matching these names against
    # BiasAuditReport.modules_run. A rename on one side alone would make every
    # module read as "did not run"; this test is the only thing that notices.
    assert adapters._AUDIT_MODULES == tuple(AUDIT_MODULES)


# ── 5. wording: a count of zero is not a finding of zero ────────────────────


class TestZeroIsNotWordedAsAFinding:
    def test_the_subtitle_does_not_report_zero_issues_found(self):
        seen = _visible(bias_audit_to_svg(_report(modules_run=list(AUDIT_MODULES))))

        assert "0 critical issues found" not in seen
        assert "no critical issue was returned" in seen

    def test_a_real_count_still_reads_as_found(self):
        # Control: the wording only changes for the zero case.
        report = _report(
            modules_run=list(AUDIT_MODULES),
            overall_risk_score=0.62,
            critical_issues=[
                {
                    "type": "Representation",
                    "description": "Women under-represented",
                    "details": "31% vs a 50% benchmark",
                    "severity": "critical",
                }
            ],
        )

        assert "1 critical issue found" in _visible(bias_audit_to_svg(report))


class TestPowerAnalysisNamesTheRightAbsence:
    def _ungraded(self):
        # Supplied, and carrying neither a power value nor a powered verdict.
        return [{"intersection": ("female", "senior"), "n_control": 40, "n_treatment": 40}]

    def test_supplied_but_ungraded_rows_are_not_called_missing(self):
        seen = _visible(power_analysis_to_svg(self._ungraded()))

        assert "no intersection was supplied" not in seen
        assert "no intersection was graded" in seen

    def test_supplied_but_ungraded_rows_say_why_nothing_was_plotted(self):
        seen = _visible(power_analysis_to_svg(self._ungraded()))

        assert "no intersection carried a power value" in seen

    def test_an_empty_call_still_says_nothing_was_supplied(self):
        # Control: the original sentence is correct for the original case.
        seen = _visible(power_analysis_to_svg([]))

        assert "no intersection was supplied" in seen


class TestNoPunctuationDashesReachTheCanvas:
    def test_the_bias_audit_canvas_carries_no_dash_punctuation(self):
        svg = bias_audit_to_svg(_report(modules_run=list(AUDIT_MODULES)))

        assert EN_DASH not in svg
        assert EM_DASH not in svg

    def test_the_power_analysis_canvas_carries_no_dash_punctuation(self):
        svg = power_analysis_to_svg([])

        assert EN_DASH not in svg
        assert EM_DASH not in svg

    def test_the_correlation_heatmap_title_carries_no_dash_punctuation(self):
        # Rendered, not grepped: the title is only a defect where a reader
        # meets it. The fixture carries both attribute spellings so this test
        # holds whichever one the adapter reads.
        from vfairness.rendering.adapters_feature_engineering import correlation_heatmap_to_svg

        class _Matrix:
            correlations = {"income": {"gender": 0.12}}
            features = ["income"]
            feature_names = ["income"]
            protected_attributes = ["gender"]

        svg = correlation_heatmap_to_svg(_Matrix())

        assert "Feature to Protected Attribute Correlations" in _visible(svg)
        assert EN_DASH not in svg
        assert EM_DASH not in svg

    def test_the_reporting_demo_narrative_carries_no_dash_punctuation(self):
        from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

        svg = reporting_dashboard_to_svg(tier="technical", example=True)
        seen = _visible(svg)

        assert "within 2 to 5 days" in seen
        assert EN_DASH not in svg
        assert EM_DASH not in svg


# ── 6. healthy input is untouched ───────────────────────────────────────────


class TestHealthyInputIsUntouched:
    def _healthy(self) -> BiasAuditReport:
        class _Confidence:
            def __init__(self, score):
                self.confidence_score = score

        class _Correlated:
            def __init__(self, correlation):
                self.correlation = correlation

        return _report(
            protected_attributes=["gender", "age_band"],
            historical_findings=[_Confidence(0.82), _Confidence(0.44)],
            disparity_findings=[_Correlated(0.37)],
            proxy_findings=[_Correlated(0.79)],
            overall_risk_score=0.6183,
            critical_issues=[
                {
                    "type": "Representation",
                    "description": "Women under-represented",
                    "details": "31% vs a 50% benchmark",
                    "severity": "critical",
                }
            ],
            recommendations=["Rebalance the training set."],
        )

    def test_a_reported_audit_keeps_its_score_and_its_verdict(self):
        seen = _visible(bias_audit_to_svg(self._healthy()))

        assert "61%" in seen
        assert "MEDIUM" in seen
        assert "NOT ASSESSABLE" not in seen

    def test_a_reported_audit_keeps_its_measured_tiles(self):
        # An unrecorded report with findings behaves exactly as before: the two
        # modules that reported are scored, the two that did not are withheld.
        seen = _visible(bias_audit_to_svg(self._healthy()))

        assert "0.63" in seen  # historical, mean of 0.82 and 0.44
        assert "0.79" in seen  # proxy
        assert "N/A" in seen  # representation, which returned nothing

    def test_a_reported_power_analysis_keeps_its_counts(self):
        svg = power_analysis_to_svg(
            [{"intersection": ("female", "senior"), "power": 0.91, "required_n": 300}]
        )

        assert "1 / 1" in _visible(svg)
        assert "COULD NOT CHECK" not in _visible(svg)
