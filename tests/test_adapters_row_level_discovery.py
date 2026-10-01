"""Per-ROW could-not-check, for the three adapters that manufactured rows.

The chart-level rule is already closed: every public ``*_to_svg`` adapter says
so when the WHOLE input is empty. It was not closed per row. An adapter whose
chart-level state is correct could still fabricate a complete, confident row
from a name alone, and a fabricated row is worse than a fabricated chart:
it sits inside an otherwise real report, in the same table as measured rows, and
nothing on the canvas separates the two.

THE RULE, per row: a row that reported nothing gets no number, no badge, no
colour, no point on a plot, and no place in any count or headline that implies
it was measured. A default is not a measurement. A sentinel is not a
measurement. Absent and zero are different.

Every class below pairs the withheld case with a HEALTHY CONTROL that must keep
rendering exactly what it rendered before. A guard that also silences real
measurements is not a fix, it is a second false report pointing the other way.
"""

import datetime as dt
import re

import pytest

from vfairness.rendering.adapters_discovery import auto_discovery_to_svg
from vfairness.rendering.adapters_monitoring import (
    alert_timeline_to_svg,
    monitoring_dashboard_to_svg,
)
from vfairness.rendering.adapters_validation import data_validation_to_svg


def _texts(svg: str) -> str:
    """The whole canvas as a string, for substring assertions."""
    return svg


class TestAutoDiscoveryViolationRows:
    """A violation carrying only an attribute name is not a finding.

    Reproduced 2026-08-27: ``auto_discovery_to_svg(violations=[{"attribute":
    "gender"}])`` rendered VALUE 0.000, THRESHOLD 0.00 and an amber MEDIUM badge
    under a green "0 high severity" summary. Every one of those numbers and the
    severity judgement came from the adapter's own defaults.
    """

    def test_a_named_violation_reports_no_value_and_no_threshold(self):
        svg = auto_discovery_to_svg(violations=[{"attribute": "gender"}])

        # The exact defaults that used to be printed.
        assert ">0.000<" not in svg, "a value of 0.000 was invented for a row that reported none"
        assert ">0.00<" not in svg, "a threshold of 0.00 was invented for a row that reported none"
        assert svg.count("not reported") >= 2

    def test_a_named_violation_is_not_graded(self):
        svg = auto_discovery_to_svg(violations=[{"attribute": "gender"}])

        assert ">MEDIUM<" not in svg, "a severity judgement was invented for an ungraded row"
        assert ">NOT GRADED<" in svg

    def test_an_ungraded_row_does_not_produce_a_green_zero_high_all_clear(self):
        svg = auto_discovery_to_svg(violations=[{"attribute": "gender"}])

        # The emerald badge fill is the all-clear. It must not be reachable with
        # rows nobody graded.
        assert 'fill="#059669">1</text>' not in svg
        assert ">1 not graded<" in svg

    def test_an_ungraded_row_is_not_counted_as_below_high_severity(self):
        svg = auto_discovery_to_svg(violations=[{"attribute": "gender"}])

        assert "further violation(s) recorded below high severity" not in svg
        assert "reported no severity and are not graded here" in svg

    def test_an_unrecognised_severity_is_ungraded_not_silently_downgraded(self):
        svg = auto_discovery_to_svg(
            violations=[{"attribute": "gender", "metric": "dp", "severity": "catastrophic"}]
        )

        assert ">NOT GRADED<" in svg
        assert ">CATASTROPHIC<" not in svg

    @pytest.mark.parametrize("sentinel", [float("nan"), float("inf")])
    def test_a_sentinel_value_is_not_a_measurement(self, sentinel):
        svg = auto_discovery_to_svg(
            violations=[
                {"attribute": "gender", "metric": "dp", "value": sentinel, "threshold": 0.1}
            ]
        )

        assert "nan" not in svg.lower().replace("finance", "")
        assert "not reported" in svg

    # HEALTHY CONTROL

    def test_a_fully_reported_violation_still_renders_every_number_and_its_badge(self):
        svg = auto_discovery_to_svg(
            violations=[
                {
                    "attribute": "gender",
                    "metric": "demographic_parity",
                    "value": 0.15,
                    "threshold": 0.10,
                    "severity": "high",
                }
            ]
        )

        assert ">0.150<" in svg
        assert ">0.10<" in svg
        assert ">HIGH<" in svg
        assert "not reported" not in svg
        assert "NOT GRADED" not in svg
        assert "1 high-severity violation(s), immediate review needed." in svg


class TestAutoDiscoveryCandidateRows:
    """A candidate with no confidence gets no bar and no number."""

    def test_a_candidate_without_confidence_gets_no_number(self):
        svg = auto_discovery_to_svg(candidates=[{"name": "zipcode"}])

        assert ">0.00<" not in svg
        assert "not reported" in svg

    def test_a_candidate_without_confidence_gets_no_bar(self):
        svg = auto_discovery_to_svg(candidates=[{"name": "zipcode"}])

        # The 4px stub `max(4, int(0 * 160))` used to draw in a confident hue,
        # so "nothing reported" looked like "2% confident".
        assert 'width="4" height="10"' not in svg

    # HEALTHY CONTROL

    def test_a_reported_confidence_still_draws_its_bar_and_number(self):
        svg = auto_discovery_to_svg(
            candidates=[{"name": "gender", "confidence": 0.95, "category": "demographic"}]
        )

        assert ">0.95<" in svg
        assert 'width="152" height="10"' in svg
        assert "not reported" not in svg


class TestAutoDiscoveryGroupRows:
    """A group that reported nothing is not a group sitting exactly at parity."""

    def test_a_bare_group_does_not_claim_parity(self):
        svg = auto_discovery_to_svg(group_advantages=[{"group": "non-binary x minority"}])

        # ratio=1.00 is the single most reassuring number this panel can print,
        # and it was the adapter's default for a group that reported nothing.
        assert "ratio=1.00" not in svg
        assert "n=0" not in svg
        assert "ratio=not reported" in svg

    def test_a_bare_group_is_not_graded_low(self):
        svg = auto_discovery_to_svg(group_advantages=[{"group": "non-binary x minority"}])

        assert ">LOW<" not in svg
        assert ">NOT GRADED<" in svg

    def test_an_unrated_group_is_disclosed_rather_than_compared(self):
        svg = auto_discovery_to_svg(group_advantages=[{"group": "non-binary x minority"}])

        assert "reported no rate relative to the overall population" in svg

    # HEALTHY CONTROL

    def test_a_fully_reported_group_still_renders_its_figures(self):
        svg = auto_discovery_to_svg(
            group_advantages=[
                {
                    "group": "male x white",
                    "positive_rate": 0.45,
                    "size": 1200,
                    "relative_to_overall": 1.32,
                    "severity": "high",
                }
            ]
        )

        assert "n=1200" in svg
        assert "ratio=1.32" in svg
        assert ">0.450<" in svg
        assert ">HIGH<" in svg
        assert "not reported" not in svg


class _Window:
    """The subset of WindowMetrics the timeline and dashboard adapters read."""

    def __init__(self, alerts, metrics=None, sample_count=0, timestamp=None):
        self.alerts = alerts
        self.metrics = metrics or {}
        self.group_rates = {}
        self.mmd_scores = {}
        self.sample_count = sample_count
        self.timestamp = timestamp or dt.datetime(2026, 8, 27, 9, 0)

    @property
    def any_alert(self):
        return any(self.alerts.values())


class TestAlertTimelineWindowRows:
    """A window that monitored nothing is not a clean window.

    ``WindowMetrics.alerts`` is the record of which guardrails were APPLIED, so
    an empty dict means none were. ``any_alert = len(alerted_names) > 0`` is a
    two-state test and it put such a window in the emerald OK badge and the
    ``n_clean`` tally.
    """

    def test_an_unmonitored_window_is_not_counted_clean(self):
        svg = alert_timeline_to_svg(
            [
                _Window({"demographic_parity": False}, {"demographic_parity": 0.02}, 500),
                _Window({}, {}, 0),
            ]
        )

        assert "2 monitoring events · 0 with alerts · 1 clean · 1 not monitored" in svg

    def test_an_unmonitored_window_is_not_badged_ok(self):
        # Two windows, because a history where NOTHING was monitored collapses
        # to the whole-chart could-not-check panel and draws no table at all
        # (covered separately below). The row-level state has to hold when the
        # unmonitored window sits beside a real one.
        svg = alert_timeline_to_svg(
            [
                _Window({"demographic_parity": False}, {"demographic_parity": 0.02}, 500),
                _Window({}, {}, 0),
            ]
        )

        assert svg.count(">OK<") == 1, "the unmonitored window kept the OK badge"
        assert ">NOT MONITORED<" in svg
        assert "no metric was checked" in svg

    def test_an_unmonitored_window_prints_no_alert_count(self):
        svg = alert_timeline_to_svg(
            [
                _Window({"demographic_parity": True}, {"demographic_parity": 0.31}, 480),
                _Window({}, {}, 0),
            ]
        )

        # Read the ALERTS column of the table (x="290") rather than searching
        # the whole canvas: a bare ">0</text>" also matches the CLEAN stat tile,
        # and a colour literal would be worse still because the skin remaps
        # every hex in these templates. The alerting window keeps its 1; the
        # unmonitored one has no count to print.
        cells = re.findall(r'<text x="290"[^>]*>(.*?)</text>', svg)
        assert cells == ["ALERTS", "1", "not monitored"], cells

    def test_a_history_where_nothing_was_monitored_certifies_nothing(self):
        svg = alert_timeline_to_svg([_Window({}, {}, 0), _Window({}, {}, 0)])

        assert "COULD NOT CHECK" in svg.upper()
        assert ">2</text>" not in svg or "not monitored" in svg

    # HEALTHY CONTROL

    def test_a_monitored_clean_window_is_still_ok_and_still_counted_clean(self):
        svg = alert_timeline_to_svg(
            [
                _Window(
                    {"demographic_parity": False, "equal_opportunity": False},
                    {"demographic_parity": 0.02},
                    500,
                )
            ]
        )

        assert ">OK<" in svg
        assert "1 monitoring events · 0 with alerts · 1 clean" in svg
        assert "not monitored" not in svg
        # The clean window still prints its real zero alert count.
        assert re.findall(r'<text x="290"[^>]*>(.*?)</text>', svg) == ["ALERTS", "0"]

    def test_a_window_that_raised_an_alert_is_unchanged(self):
        svg = alert_timeline_to_svg(
            [_Window({"demographic_parity": True}, {"demographic_parity": 0.31}, 480)]
        )

        assert ">ALERT<" in svg
        assert "demographic_parity" in svg
        assert "1 monitoring events · 1 with alerts · 0 clean" in svg


class TestMonitoringDashboardMetricRows:
    """A metric that was computed but never compared to a threshold is not OK."""

    def test_an_unchecked_metric_is_not_badged_ok(self):
        svg = monitoring_dashboard_to_svg(
            _Window({"demographic_parity": False}, {"demographic_parity": 0.02, "tpr_gap": 0.31})
        )

        assert ">NOT CHECKED<" in svg
        # Two ">OK<" survive: the window-level STATUS tile in the header (no
        # alert fired, which is true) and the ONE checked metric's row badge.
        # Before the fix there were three, the extra one belonging to a metric
        # nothing had compared to a threshold.
        assert svg.count(">OK<") == 2

    def test_an_unchecked_metric_is_not_counted_as_monitored_in_the_headline(self):
        svg = monitoring_dashboard_to_svg(
            _Window({"demographic_parity": False}, {"demographic_parity": 0.02, "tpr_gap": 0.31})
        )

        assert "2 metrics monitored" not in svg
        assert "1 of 2 metrics checked against a guardrail" in svg

    # HEALTHY CONTROL

    def test_every_checked_metric_keeps_its_verdict(self):
        svg = monitoring_dashboard_to_svg(
            _Window(
                {"demographic_parity": False, "tpr_gap": True},
                {"demographic_parity": 0.02, "tpr_gap": 0.31},
                500,
            )
        )

        assert ">OK<" in svg
        assert ">ALERT<" in svg
        assert "NOT CHECKED" not in svg
        assert "2 metrics monitored" in svg


class _ValidationResult:
    """A DataValidationResult once the source records which checks ran.

    ROOT CAUSE, and only half fixed. The real
    ``operations.cicd.validator.DataValidationResult`` records ``passed`` but
    NOT which checks produced it, so a validate() run with every check disabled
    returns passed=True and this page renders a green PASS reading "The data is
    ready for fairness analysis" (reproduced 2026-08-27). The remaining half is
    a ``checks_run`` field plus ``execution_coverage()`` on that dataclass,
    mirroring ``BiasAuditReport.modules_run``. These tests pin the READER so the
    source change is a drop-in, and
    ``test_a_result_that_does_not_record_coverage_is_unchanged`` documents
    exactly what still ships wrong until it lands.
    """

    def __init__(self, coverage, passed=True, issues=()):
        self._coverage = coverage
        self._passed = passed
        self._issues = list(issues)

    def execution_coverage(self):
        return self._coverage

    def to_dict(self):
        return {
            "passed": self._passed,
            "issues": self._issues,
            "metrics": {"n_samples": 120},
            "summary": "Validation passed - no bias issues detected",
        }


class TestDataValidationCoverage:
    def test_a_run_that_checked_nothing_is_not_a_pass(self):
        svg = data_validation_to_svg(_ValidationResult("none"))

        assert ">PASS<" not in svg
        assert ">UNKNOWN<" in svg
        assert "The data is ready for fairness analysis" not in svg
        assert "records that no validation check ran" in svg

    def test_a_run_that_checked_nothing_withholds_the_counters(self):
        svg = data_validation_to_svg(_ValidationResult("none"))

        # "0 critical, 0 warn, 0 info" is the same line a clean validation
        # prints, and this run counted nothing because it checked nothing.
        assert "0 critical" not in svg
        assert "no issue was counted" in svg

    def test_a_recorded_failure_survives_a_none_coverage_record(self):
        svg = data_validation_to_svg(
            _ValidationResult(
                "none",
                passed=False,
                issues=[
                    {
                        "severity": "CRITICAL",
                        "issue_type": "imbalanced_groups",
                        "message": "group too small",
                    }
                ],
            )
        )

        # An issue was raised by something, so the failure stands. Withholding a
        # FAIL is as wrong as inventing a PASS.
        assert ">FAIL<" in svg

    def test_a_partial_run_says_what_it_does_not_cover(self):
        svg = data_validation_to_svg(_ValidationResult("partial"))

        assert ">PASS<" in svg
        assert "Only some validation checks ran" in svg

    def test_checks_run_is_read_off_a_plain_dict_too(self):
        svg = data_validation_to_svg(
            {
                "passed": True,
                "issues": [],
                "metrics": {},
                "summary": "Validation passed",
                "checks_run": [],
            }
        )

        assert ">UNKNOWN<" in svg
        assert ">PASS<" not in svg

    # HEALTHY CONTROL

    def test_a_complete_run_still_renders_a_real_green_pass(self):
        svg = data_validation_to_svg(_ValidationResult("complete"))

        assert ">PASS<" in svg
        assert "All validation checks passed. The data is ready for fairness analysis." in svg
        assert "Only some validation checks ran" not in svg

    def test_a_result_that_does_not_record_coverage_is_unchanged(self):
        """OPEN: this is the false PASS the source fix must close.

        A real ``DataValidationResult`` answers "unrecorded", and unrecorded
        deliberately changes nothing: it neither grants a verdict nor refuses
        one. Turning every existing PASS into COULD NOT CHECK would be a false
        alarm across the whole install base rather than a fix. When
        ``checks_run`` lands on the dataclass, a no-check run stops answering
        "unrecorded", this assertion's premise disappears, and the test above
        for coverage "none" takes over.
        """
        from vfairness.operations.cicd.validator import DataValidationResult

        result = DataValidationResult(
            passed=True,
            issues=[],
            metrics={"n_samples": 120},
            summary="Validation passed - no bias issues detected",
        )

        svg = data_validation_to_svg(result)

        assert ">PASS<" in svg
