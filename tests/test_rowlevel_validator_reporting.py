"""Row-level fabrication, wave 13: the validator record, the dashboard scorecard
and the two adapters that crashed instead of answering.

THE RULE, PER ROW. A row that reported nothing gets no number, no badge, no
colour, no point on a plot, and no place in any count or headline that implies it
was measured. A default is not a measurement. A sentinel is not a measurement.
Absent and zero are different claims.

BOTH DIRECTIONS. A fabricated ALL-CLEAR tells a reader a group was checked and
cleared when it was not. A fabricated BREACH sends someone after a violation that
does not exist and discredits the tool when they work out why. Both appear below.

THE HEADLINE RULE, as decided for this wave and applied here:
  a) grade and band over the GRADED subset, so a partially measured run still
     gives a useful verdict on what WAS measured;
  b) never render an unqualified all-clear while anything is ungraded;
  c) state the ungraded count ON THE CANVAS, next to the headline, not only in
     the accessible ``<desc>``;
  d) an ungraded row never enters a numerator or a denominator that implies
     measurement.

The four sites, all reproduced by execution on 2026-08-27 before being fixed:

1. ``operations.cicd.validator.DataValidationResult`` recorded ``passed`` and
   nothing about which checks produced it. ``DataBiasValidator.validate`` gates
   all five checks on config flags, so a config with every flag off returned
   ``passed=True``, ``issues=[]`` and "Validation passed - no bias issues
   detected", and ``data_validation_to_svg`` painted a green PASS reading "The
   data is ready for fairness analysis". The adapter-side reader had already
   landed in a previous wave; the dataclass half had not, so the false PASS
   still shipped.
2. ``reporting_dashboard_to_svg`` fabricated its ENTIRE scorecard from a
   non-None report with nothing in it: health 0/100 under a YELLOW badge, TREND
   Stable, slope +0.0000, three KPI cards at a failing 0/100, ACTIVE ALERTS 0 in
   pass green, and "0 metrics, 0 groups, 0 records". The empty-input guard added
   when the demo fallback was removed only ever caught a literal ``None``.
3. Per row on the same dashboard: a metric with no ``value`` was graded against
   an invented 0.1 threshold as though it had measured exactly 0.0000, which on
   a lower-is-better metric is a perfect score, so it rendered the pass swatch
   and a PASS badge.
4. ``drift_report_to_svg`` raised ValueError for a scale with a None p-value,
   because the template compares that value inline. One scale that did not
   report a p-value took down the whole report, including the scales that were
   measured perfectly well.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering adapters require jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)

from vfairness.operations.cicd.validator import (  # noqa: E402
    VALIDATION_CHECKS,
    DataBiasValidator,
    DataValidationConfig,
    DataValidationResult,
)
from vfairness.rendering.adapters_monitoring import drift_report_to_svg  # noqa: E402
from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg  # noqa: E402
from vfairness.rendering.adapters_validation import data_validation_to_svg  # noqa: E402

_TEXT_NODE = re.compile(r">([^<>]+)<")


def canvas_text(svg: str) -> str:
    """Every drawn text node, joined, with runs of whitespace collapsed.

    Read the canvas, never the data dict: a reader sees the rendered page, and a
    dict key that says could-not-check while the page prints a verdict is the
    exact failure this file exists to catch.

    The collapse matters. These templates wordwrap a narrative into one ``<text>``
    per line, and the newlines BETWEEN those elements are text nodes too, so a
    sentence the reader sees as one line arrives here as
    ``"0 within \\n bound"``. Collapsing reconstructs the sentence a human reads.
    """
    return re.sub(r"\s+", " ", " ".join(_TEXT_NODE.findall(svg))).strip()


def _clean_frame() -> pd.DataFrame:
    """A dataset with nothing wrong with it, for the positive controls.

    Balanced groups, equal outcome rates, no missing values, no duplicates and
    comfortably above ``min_total_samples``, so a fully enabled validator run
    over it is a genuine, complete pass. Without that the "a real pass survives"
    controls would be vacuous.
    """
    n = 600
    return pd.DataFrame(
        {
            "gender": ["M", "F"] * (n // 2),
            "region": (["north"] * (n // 4) + ["south"] * (n // 4)) * 2,
            "score": [i / n for i in range(n)],
            "approved": [1, 0, 0, 1] * (n // 4),
        }
    )


# 1.  The validator records which checks ran, and the page reads it


class TestValidatorExecutionCoverage:
    def test_a_run_with_every_check_disabled_is_not_a_pass(self):
        """THE DEFECT, end to end, on the surface a reader actually sees."""
        config = DataValidationConfig(
            check_representation=False,
            check_outcome_disparity=False,
            check_missing_patterns=False,
            check_label_quality=False,
            check_data_hygiene=False,
        )
        result = DataBiasValidator(["gender"], config=config).validate(
            _clean_frame(), outcome_column="approved"
        )

        assert result.checks_run == [], "a run that executed nothing did not record that"
        assert result.execution_coverage() == "none"
        assert "no bias issues detected" not in result.summary

        svg = data_validation_to_svg(result)
        drawn = canvas_text(svg)
        assert ">PASS<" not in svg, "a validation that checked nothing still badged PASS"
        assert ">UNKNOWN<" in svg
        assert "The data is ready for fairness analysis" not in drawn

    def test_the_counters_are_withheld_rather_than_printed_as_zero(self):
        """ "0 critical, 0 warn, 0 info" is the line a clean validation prints."""
        config = DataValidationConfig(
            check_representation=False,
            check_outcome_disparity=False,
            check_missing_patterns=False,
            check_label_quality=False,
            check_data_hygiene=False,
        )
        result = DataBiasValidator(["gender"], config=config).validate(_clean_frame())

        drawn = canvas_text(data_validation_to_svg(result))
        assert "0 critical" not in drawn
        assert "no issue was counted" in drawn

    def test_a_partial_run_names_the_checks_it_did_not_make(self):
        """Headline rule (a) and (c): band over what ran, count what did not.

        With no outcome column there is nothing for the disparity and label
        checks to read, so three of five run. The verdict still stands over
        those three, and the two that did not run are counted where the reader
        of the verdict sees them.
        """
        result = DataBiasValidator(["gender"]).validate(_clean_frame())

        assert result.execution_coverage() == "partial"
        assert "outcome_disparity" in result.summary and "label_quality" in result.summary
        assert "3 of 5 checks" in result.summary

        drawn = canvas_text(data_validation_to_svg(result))
        assert "Only some validation checks ran" in drawn

    def test_a_recorded_failure_survives_a_no_check_record(self):
        """Withholding a measured failure is as wrong as inventing a pass.

        A missing protected attribute is caught before any check runs, so the
        result records ``checks_run == []`` AND a critical issue. The issue was
        raised by something, so the FAIL stands.
        """
        result = DataBiasValidator(["not_a_column"]).validate(_clean_frame())

        assert result.checks_run == []
        assert result.passed is False
        assert ">FAIL<" in data_validation_to_svg(result)

    # POSITIVE CONTROLS

    def test_a_complete_clean_run_still_renders_a_real_green_pass(self):
        """The fix must not turn a genuine all-clear into a shrug."""
        result = DataBiasValidator(["gender"]).validate(_clean_frame(), outcome_column="approved")

        assert set(result.checks_run) == set(VALIDATION_CHECKS)
        assert result.execution_coverage() == "complete"
        assert result.passed is True
        assert result.summary == "Validation passed - no bias issues detected"

        svg = data_validation_to_svg(result)
        assert ">PASS<" in svg
        assert "The data is ready for fairness analysis" in canvas_text(svg)
        assert "Only some validation checks ran" not in canvas_text(svg)

    def test_a_result_that_does_not_record_coverage_is_unchanged(self):
        """Backward compatibility, and it is deliberate.

        A ``DataValidationResult`` built by hand, or unpickled from a version
        before ``checks_run`` existed, answers "unrecorded". Unrecorded grants no
        verdict and refuses none: turning every PASS on the install base into
        COULD NOT CHECK would be a false alarm rather than a fix.
        """
        result = DataValidationResult(
            passed=True,
            issues=[],
            metrics={"n_samples": 120},
            summary="Validation passed - no bias issues detected",
        )

        assert result.checks_run is None
        assert result.execution_coverage() == "unrecorded"
        assert ">PASS<" in data_validation_to_svg(result)

    def test_the_record_travels_with_the_serialised_result(self):
        """A consumer that only ever sees ``to_dict()`` must be able to tell the
        two apart too, and ``passed`` cannot."""
        ran = DataBiasValidator(["gender"]).validate(_clean_frame(), outcome_column="approved")
        exported = ran.to_dict()
        assert exported["execution_coverage"] == "complete"
        assert set(exported["checks_run"]) == set(VALIDATION_CHECKS)

        bare = DataValidationResult(passed=True, issues=[], metrics={}, summary="s").to_dict()
        assert bare["checks_run"] is None
        assert bare["execution_coverage"] == "unrecorded"


# 2.  The dashboard scorecard is withheld, not defaulted


def _report(**over):
    """A dashboard report dict, healthy unless a test breaks one field."""
    base = {
        "tier": "OPERATIONAL",
        "health_score": {
            "score": 91,
            "status": "green",
            "components": {
                "metric_compliance": 95,
                "alert_frequency": 90,
                "drift_stability": 88,
            },
            "trend": "stable",
            "trend_slope": 0.0008,
        },
        "metrics": [{"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10}],
        "alerts": [],
        "recommendations": [],
        "sections": [],
        "n_groups": 3,
        "n_records": 8400,
    }
    base.update(over)
    return base


class _ReportObject:
    """A GeneratedReport with no health score, as ``to_dict`` really writes it.

    ``GeneratedReport.health_score`` is Optional and its ``to_dict`` writes None
    for it, which is how a non-None report reached the defaults.
    """

    def to_dict(self):
        return {
            "title": "Fairness Report",
            "tier": "OPERATIONAL",
            "timestamp": "2026-08-27 10:00:00",
            "health_score": None,
            "sections": [],
            "recommendations": [],
            "metadata": {},
        }


class TestDashboardScorecard:
    def test_a_report_with_nothing_in_it_draws_no_scorecard(self):
        """THE DEFECT. Every cell below was invented from a default."""
        drawn = canvas_text(reporting_dashboard_to_svg(_ReportObject()))

        assert "FAIRNESS HEALTH SCORE" not in drawn
        assert "/100" not in drawn, "a score denominator was drawn with no score"
        for band in ("GREEN", "YELLOW", "RED"):
            assert band not in drawn, f"a {band} status band was drawn from no health score"
        assert "Stable" not in drawn, "a TREND was drawn for a report that measured none"
        assert "METRIC COMPLIANCE" not in drawn and "DRIFT STABILITY" not in drawn
        assert "ACTIVE ALERTS" not in drawn
        assert not re.search(r"\d+ groups", drawn), "a group count was drawn from no count"
        assert "NOT CHECKED" in drawn

    def test_the_reason_is_stated_on_the_canvas_not_only_in_the_desc(self):
        drawn = canvas_text(reporting_dashboard_to_svg(_ReportObject()))
        assert "the scorecard is withheld, not zero" in drawn
        assert "carries no health score" in drawn
        assert "certifies nothing" in drawn

    def test_a_measured_breach_is_not_lost_with_the_withheld_scorecard(self):
        """Withholding a measured failure is as wrong as inventing a pass.

        The scorecard goes because no health score was recorded. The metric row
        beside it WAS measured and WAS breached, so the breach is named in
        words, and the tally counts only the rows that were graded.
        """
        drawn = canvas_text(
            reporting_dashboard_to_svg(
                _report(
                    health_score=None,
                    metrics=[
                        {
                            "name": "demographic_parity_difference",
                            "value": 0.240,
                            "threshold": 0.10,
                        }
                    ],
                )
            )
        )

        assert "/100" not in drawn
        assert "BREACH:" in drawn
        assert "demographic_parity_difference" in drawn
        assert "0 within bound, 1 breached" in drawn

    def test_a_row_with_no_value_is_never_graded(self):
        """FABRICATED ALL-CLEAR, per row. An absent value was read as 0.0000,
        which on a lower-is-better metric is a perfect result."""
        svg = reporting_dashboard_to_svg(
            _report(
                metrics=[
                    {"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10},
                    {"name": "equal_opportunity_difference", "threshold": 0.10},
                ]
            )
        )
        drawn = canvas_text(svg)

        assert "[UNCHECKED]" in drawn
        assert "0.0000" not in drawn, "an unmeasured row was graded as a perfect 0.0000"
        assert "NOT CHECKED" in drawn
        assert "0% margin" not in drawn, "an unmeasured row was placed exactly on its threshold"
        assert ">GREEN<" not in svg, "the all-clear survived beside a row nobody measured"
        assert "COULD NOT CHECK" in drawn

    def test_a_row_with_no_threshold_is_never_graded(self):
        """FABRICATED BREACH, the other direction. The old default bound of 0.1
        could convict a value just as easily as it could clear one."""
        svg = reporting_dashboard_to_svg(
            _report(metrics=[{"name": "demographic_parity_difference", "value": 0.45}])
        )
        drawn = canvas_text(svg)

        assert "[UNCHECKED]" in drawn
        assert "THRESHOLD BREACH REPORT" not in drawn, "a breach was declared against no bound"
        assert "0.1000" not in drawn, "an invented threshold was printed as the bound applied"

    def test_an_ungraded_row_enters_no_count_that_implies_measurement(self):
        """Headline rule (d). ``explain._fr_reporting`` writes n_metrics as the
        denominator of "N breach(es) of M" in the accessible description."""
        svg = reporting_dashboard_to_svg(
            _report(
                metrics=[
                    {"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10},
                    {"name": "equal_opportunity_difference", "threshold": 0.10},
                ]
            )
        )
        desc = re.search(r"<desc>([^<]*)</desc>", svg)
        assert desc, "no accessible <desc>"
        assert "of 2" not in desc.group(1), (
            f"an ungraded row entered the breach denominator: {desc.group(1)!r}"
        )
        assert "of 1" in desc.group(1)

    def test_an_unreported_breach_scope_does_not_name_every_group(self):
        """FABRICATED BREACH SCOPE. The default was the literal string
        "all groups", the widest possible claim about who was harmed."""
        drawn = canvas_text(
            reporting_dashboard_to_svg(
                _report(
                    metrics=[
                        {
                            "name": "demographic_parity_difference",
                            "value": 0.240,
                            "threshold": 0.10,
                        }
                    ]
                )
            )
        )
        assert "THRESHOLD BREACH REPORT" in drawn
        assert "all groups" not in drawn
        assert "affected groups not reported" in drawn

    def test_an_unreported_slope_is_not_drawn_as_a_flat_trend(self):
        hs = dict(_report()["health_score"])
        hs.pop("trend_slope")
        drawn = canvas_text(reporting_dashboard_to_svg(_report(health_score=hs)))

        assert "slope:" not in drawn
        assert "+0.0000" not in drawn

    def test_unreported_kpi_subscores_are_named_rather_than_read_as_failures(self):
        """FABRICATED FAILURE. An absent sub-score printed 0/100 in the failing
        red, which sends someone after a compliance problem nobody measured. The
        template has no unmeasured state for those cards, so the canvas says so
        in words instead."""
        hs = dict(_report()["health_score"])
        hs["components"] = {}
        drawn = canvas_text(reporting_dashboard_to_svg(_report(health_score=hs)))

        assert "NOT MEASURED: 3 of 3 KPI sub-score(s) were not reported" in drawn
        assert "an absence, not a failing score" in drawn

    # POSITIVE CONTROLS

    def test_a_fully_reported_dashboard_is_unchanged(self):
        svg = reporting_dashboard_to_svg(_report())
        drawn = canvas_text(svg)

        assert ">GREEN<" in svg
        assert "91" in drawn and "/100" in drawn
        assert "3 groups" in drawn and "8400 records" in drawn
        assert "slope: +0.0008/day" in drawn
        assert "NOT CHECKED" not in drawn
        assert "COULD NOT CHECK" not in drawn

    def test_a_measured_breach_still_reads_as_a_breach(self):
        svg = reporting_dashboard_to_svg(
            _report(
                metrics=[
                    {
                        "name": "demographic_parity_difference",
                        "value": 0.240,
                        "threshold": 0.10,
                        "affected_groups": ["female"],
                    }
                ]
            )
        )
        drawn = canvas_text(svg)
        assert "THRESHOLD BREACH REPORT" in drawn
        assert "female" in drawn
        assert ">RED<" in svg


# 3.  A partial row answers could-not-check instead of ending the process


def _scale(**over):
    base = dict(
        drift_score=0.42,
        ks_statistic=0.31,
        p_value=0.004,
        mean_shift=0.12,
        reference_mean=0.5,
        current_mean=0.62,
        drift_detected=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _drift(scales, **over):
    base = dict(
        drift_detected=True,
        overall_drift_score=0.42,
        metric="demographic_parity_difference",
    )
    base.update(over)
    return SimpleNamespace(scales=scales, **base)


class TestDriftPartialRow:
    def test_a_scale_with_no_p_value_no_longer_takes_the_report_down(self):
        """THE DEFECT. ``drift_report.svg`` compares the p-value inline, so a
        None arrived at the template as a TypeError that ``render_svg``
        re-raised, and ONE incomplete scale destroyed the whole report."""
        svg = drift_report_to_svg(
            _drift(
                {
                    "daily": _scale(),
                    "weekly": _scale(
                        drift_score=0.10,
                        ks_statistic=0.05,
                        p_value=None,
                        mean_shift=0.01,
                        current_mean=0.51,
                        drift_detected=False,
                    ),
                }
            )
        )
        drawn = canvas_text(svg)

        assert svg, "the report rendered nothing"
        assert "N/A" in drawn, "the unmeasured p-value was not stated as unmeasured"
        assert "0.0000" not in drawn, "an unmeasured p-value was printed as a measured zero"
        # Headline rule (a): the scale that WAS measured keeps its numbers.
        assert "0.0040" in drawn and "0.420" in drawn
        # Headline rule (c): the ungraded count sits beside the headline badge.
        assert "1 of 2 scale(s) did not fully report" in drawn

    def test_a_scale_that_reported_nothing_gets_no_drift_number(self):
        svg = drift_report_to_svg(
            _drift(
                {
                    "daily": _scale(),
                    "weekly": _scale(
                        drift_score=None,
                        ks_statistic=None,
                        p_value=None,
                        mean_shift=None,
                        reference_mean=None,
                        current_mean=None,
                        drift_detected=None,
                    ),
                }
            )
        )
        drawn = canvas_text(svg)

        assert "0.000" not in drawn, "an unmeasured scale was scored 0.000"
        assert "ref=N/A / cur=N/A" in drawn
        assert "1 of 2 scale(s) did not fully report" in drawn

    def test_a_report_whose_scales_all_reported_nothing_grades_none_of_them(self):
        """The rows exist, nothing was measured, and the maximum over nothing is
        still not a reading."""
        svg = drift_report_to_svg(
            _drift(
                {
                    "daily": _scale(
                        drift_score=None,
                        ks_statistic=None,
                        p_value=None,
                        mean_shift=None,
                        reference_mean=None,
                        current_mean=None,
                        drift_detected=None,
                    )
                },
                drift_detected=None,
                overall_drift_score=0.0,
            )
        )
        drawn = canvas_text(svg)

        assert "STABLE" not in drawn, "an unmeasured drift report badged STABLE"
        assert "DETECTED" not in drawn
        assert "No scale reported a drift score" in drawn

    def test_a_non_numeric_statistic_is_refused_rather_than_rendered(self):
        """A stringified number is not a measurement either, and NaN is the
        sentinel the detector itself writes for "not computed"."""
        svg = drift_report_to_svg(
            _drift({"daily": _scale(), "weekly": _scale(p_value=float("nan"))})
        )
        drawn = canvas_text(svg)
        assert "nan" not in drawn.lower().replace("n/a", "")
        assert "1 of 2 scale(s) did not fully report" in drawn

    # POSITIVE CONTROLS

    def test_a_fully_measured_drift_report_is_unchanged(self):
        drawn = canvas_text(drift_report_to_svg(_drift({"daily": _scale()})))

        assert "DETECTED" in drawn
        assert "0.0040" in drawn and "+0.1200" in drawn and "0.420" in drawn
        assert "ref=0.500 / cur=0.620" in drawn
        assert "N/A" not in drawn
        assert "did not fully report" not in drawn

    def test_a_measured_stable_scale_keeps_its_stable_badge(self):
        drawn = canvas_text(
            drift_report_to_svg(
                _drift(
                    {
                        "daily": _scale(
                            drift_score=0.02,
                            ks_statistic=0.01,
                            p_value=0.8,
                            mean_shift=0.001,
                            current_mean=0.501,
                            drift_detected=False,
                        )
                    },
                    drift_detected=False,
                    overall_drift_score=0.02,
                )
            )
        )
        assert "STABLE" in drawn
        assert "0.8000" in drawn
        assert "did not fully report" not in drawn
