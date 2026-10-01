"""G05: the CI/CD release-gate surfaces, on input where the thing they publish
does not exist.

Five defects, all of one shape: the object had THREE OR FOUR states and the
surface a reader (or a CI runner) actually consumes had TWO.

1. ``GateDecision.to_markdown_report`` / ``IntersectionalGateDecision
   .to_markdown_report`` / ``FairnessReportCard`` mapped the row's verdict
   straight off ``MetricEvaluation.passed``, which cannot say whether the
   comparison HAPPENED. Two doors, both measured on HEAD:

     evaluate(y, y, ['a'] * 200), one group, threshold 0.1
         | demographic_parity_difference | nan | 0.1000 | X Fail |
     evaluate on a GENUINELY FAIR model, threshold 2.0 (range [0, 1])
         | demographic_parity_difference | 0.0100 | 2.0000 | X Fail |

   A metric nobody could compute, and a measured 0.01 gap against a bound no
   data can breach, both published as measured breaches. A ``value`` of None
   did not merely mislead: ``f"{None:.4f}"`` raised TypeError, so
   ``create_github_check`` could not build its payload at all.

2. ``MetricEvaluation.message``, written at every one of the gate's fail-closed
   branches, was rendered by no surface: the page-level reason lists are not
   keyed to a metric, and the hierarchical report printed the identical sentence
   once per level with nothing saying which.

3. ``FairnessAssertionError``, which IS this module's refusal channel, raised
   ValueError when the value it was handed was text-typed, which is what an
   ordinary CSV read gives you. ValueError is not an AssertionError, so the
   refusal escaped ``parametrize_fairness``'s own ``except
   FairnessAssertionError`` handler.

4. ``FairnessTestSuite.to_junit_xml`` published ``tests="0" failures="0"`` for a
   suite that ran nothing, while ``get_summary()`` on the same object answered
   ``no_tests_run``.

5. ``DataValidationResult.to_junit_xml`` had the ``validation_coverage`` guard
   for ONE of its four coverage states, so ``partial`` and ``unrecorded``
   (the DEFAULT) still published the empty green suite its own docstring names.

Every refusal below is paired with a control asserting the healthy case's REAL
number, because a surface that refuses everything passes every refusal test and
is just as useless as one that approves everything.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateDecision,
    GateStatus,
    MetricEvaluation,
    ModelFairnessGate,
)
from vfairness.operations.cicd.monitor import (
    AlertSeverity,
    BiasMonitor,
    DriftAlert,
    DriftType,
)
from vfairness.operations.cicd.testing import (
    FairnessAssertionError,
    FairnessTestResult,
    FairnessTestSuite,
    _refuse_unless_measured,
)
from vfairness.operations.cicd.testing import TestStatus as Status
from vfairness.operations.cicd.validator import (
    VALIDATION_CHECKS,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

METRIC = "demographic_parity_difference"
N = 200


def _fair_arrays():
    """Two groups of 100 and a model that is right about everybody."""
    y = np.random.default_rng(7).integers(0, 2, N)
    protected = np.array(["a"] * (N // 2) + ["b"] * (N // 2))
    return y, y.copy(), protected


def _gate(threshold: float = 0.1) -> ModelFairnessGate:
    return ModelFairnessGate(metrics=[METRIC], thresholds={METRIC: threshold})


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _row(report: str, metric: str = METRIC) -> str:
    rows = [ln for ln in report.splitlines() if ln.startswith(f"| {metric} ")]
    assert len(rows) >= 1, f"no metric row for {metric} in:\n{report}"
    return rows[0]


# ===========================================================================
# 1a. THE VALUE DOOR. A metric nobody could compute.
# ===========================================================================


class TestAnUncomputableMetricIsNotAMeasuredBreach:
    def _one_group(self):
        y_true, y_pred, _ = _fair_arrays()
        return y_true, y_pred, np.array(["a"] * N)

    def test_the_flat_report_row_says_could_not_check(self):
        decision = _quiet(_gate().evaluate, *self._one_group())
        row = _row(decision.to_markdown_report())

        assert "Could not check" in row, row
        assert "Fail" not in row, row
        assert "Pass" not in row, row
        # And the row names WHY there is no number, rather than printing one.
        assert "not measured" in row, row
        # The decision is still a refusal: the fix moves the row's label, never
        # the verdict.
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED

    def test_the_github_check_text_is_the_same_page(self):
        """create_github_check publishes to_markdown_report as output.text, so
        this is the surface branch protection shows a reviewer."""
        decision = _quiet(_gate().evaluate, *self._one_group())
        payload = ModelFairnessGate().create_github_check(decision)

        assert payload["conclusion"] == "failure"
        assert "Could not check" in payload["output"]["text"]

    def test_the_hierarchical_report_and_the_card_agree_with_it(self):
        y_true, y_pred, one_group = self._one_group()
        decision = _quiet(_gate().evaluate_hierarchical, y_true, y_pred, {"gender": one_group})
        report = decision.to_markdown_report()
        card = FairnessReportCard(decision, model_name="m").to_markdown()

        assert "Could not check" in _row(report)
        assert "COULD NOT BE CHECKED" in card.upper()
        # The card's "Failures" count is a count of comparisons, and there were
        # none to count.
        assert "could not be checked" in card

    def test_a_none_value_does_not_crash_the_report_or_the_payload(self):
        """``f"{None:.4f}"`` raised TypeError here while to_dict() on the same
        object serialised it happily, so the CI payload could not be built at
        all. MetricEvaluation.value is a required positional field and
        FairnessTestResult's equivalent DEFAULTS to None."""
        decision = GateDecision(
            approved=False,
            status=GateStatus.BLOCKED,
            metric_evaluations=[MetricEvaluation(METRIC, None, 0.1, False)],
        )

        row = _row(decision.to_markdown_report())
        assert "Could not check" in row
        assert "no value was recorded" in row
        assert (
            "Could not check"
            in (ModelFairnessGate().create_github_check(decision)["output"]["text"])
        )


# ===========================================================================
# 1b. THE BOUND DOOR. A bound no data in the metric's range can breach.
# ===========================================================================


class TestAVacuousBoundIsNotAMeasuredBreach:
    def test_a_fair_model_is_not_reported_as_failing_an_unbreachable_bound(self):
        """The 0.01-against-2.0 row. The bound is vacuous because
        demographic_parity_difference lies in [0, 1], which is a property of the
        bound AND the range together: 2.0 is finite, so an isinf test leaves it
        wide open."""
        decision = _quiet(_gate(threshold=2.0).evaluate, *_fair_arrays())
        row = _row(decision.to_markdown_report())
        measured = decision.metric_evaluations[0].value

        # The measurement is real and is still published as a number.
        assert 0.0 <= measured <= 1.0
        assert f"{measured:.4f}" in row
        # What is refused is the VERDICT, not the value.
        assert "Could not check" in row, row
        assert "Fail" not in row, row

    def test_a_metric_with_no_threshold_configured_is_not_a_failure_either(self):
        """The gate's own message for it: "never compared against anything"."""
        gate = ModelFairnessGate(metrics=[METRIC], thresholds={})
        decision = _quiet(gate.evaluate, *_fair_arrays())
        row = _row(decision.to_markdown_report())

        assert "N/A" in row
        assert "Could not check" in row, row
        assert "Fail" not in row, row

    def test_control_a_usable_bound_still_grades_both_ways(self):
        """THE OVER-CORRECTION CONTROL. A guard that calls every bound unusable
        passes every test above and destroys the gate."""
        y_true, y_pred, protected = _fair_arrays()
        clean = _gate().evaluate(y_true, y_pred, protected)
        clean_row = _row(clean.to_markdown_report())

        assert clean.approved is True
        assert "Pass" in clean_row
        assert "Could not check" not in clean_row
        assert f"{clean.metric_evaluations[0].value:.4f}" in clean_row
        assert ModelFairnessGate().create_github_check(clean)["conclusion"] == "success"

        # A REAL breach: every 'a' selected, no 'b' selected, so the gap is the
        # largest a rate gap can be.
        biased = np.where(protected == "a", 1, 0)
        breach = _gate().evaluate(biased, biased, protected)
        breach_row = _row(breach.to_markdown_report())

        assert breach.approved is False
        assert "Fail" in breach_row
        assert "Could not check" not in breach_row
        assert f"{breach.metric_evaluations[0].value:.4f}" in breach_row


# ===========================================================================
# 1c. THE THIRD DOOR. Refused by something that is not this bound.
# ===========================================================================


class TestARowRefusedByAnotherCheckDoesNotContradictItsOwnNumbers:
    """``passed`` is one boolean carrying the outcome of up to six checks, so
    "not passed" was rendered as "over its threshold" for all of them. Each
    case below has a MEASURED value INSIDE its threshold and is refused by a
    baseline or a margin that could not be checked."""

    @pytest.mark.parametrize(
        "kwargs, baseline",
        [
            ({"require_improvement": True}, {METRIC: float("nan")}),
            ({"require_improvement": True}, {}),
            ({"require_improvement": True}, {METRIC: None}),
            (
                {"require_improvement": True, "improvement_margin": float("nan")},
                {METRIC: 0.05},
            ),
            ({"allow_degradation_margin": float("nan")}, {METRIC: 0.05}),
        ],
    )
    def test_the_row_is_not_labelled_a_threshold_failure(self, kwargs, baseline):
        from vfairness.operations.cicd.gate import GateConfig

        gate = ModelFairnessGate(
            config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.1}, **kwargs)
        )
        decision = _quiet(gate.evaluate, *_fair_arrays(), baseline_metrics=baseline)
        row = _row(decision.to_markdown_report())
        value = decision.metric_evaluations[0].value

        # The premise: the value IS measured and IS inside its bound. Derived,
        # not quoted, so the assertion cannot go stale.
        assert abs(value) <= 0.1, f"fixture error: {value} is outside the bound"
        assert f"{value:.4f}" in row
        # The refusal stands, and it is not attributed to the bound the row met.
        assert decision.approved is False
        assert "Refused (not by this bound)" in row, row
        assert "Fail" not in row, row
        assert "Pass" not in row, row
        # And the reason is beside it, keyed to this metric.
        assert "### Per-metric notes" in decision.to_markdown_report()

    def test_control_a_baseline_requirement_that_was_met_still_passes(self):
        """THE OVER-CORRECTION CONTROL for this door: a REAL, measured, met
        improvement requirement must still read Pass with its real number."""
        from vfairness.operations.cicd.gate import GateConfig

        gate = ModelFairnessGate(
            config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.1}, require_improvement=True)
        )
        decision = gate.evaluate(*_fair_arrays(), baseline_metrics={METRIC: 0.9})
        row = _row(decision.to_markdown_report())

        assert decision.approved is True
        assert "Pass" in row
        assert "Refused" not in row
        assert f"{decision.metric_evaluations[0].value:.4f}" in row

    def test_control_a_measured_breach_beside_an_unchecked_baseline_is_still_a_fail(
        self,
    ):
        """The other half: when the bound IS breached, the label must stay Fail
        even though a second check could not run. The refusal that names the
        bound is the more precise one."""
        from vfairness.operations.cicd.gate import GateConfig

        _, _, protected = _fair_arrays()
        biased = np.where(protected == "a", 1, 0)
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.1}, require_improvement=True)
        )
        decision = _quiet(
            gate.evaluate, biased, biased, protected, baseline_metrics={METRIC: float("nan")}
        )
        row = _row(decision.to_markdown_report())

        assert "Fail" in row, row
        assert "Refused" not in row, row


# ===========================================================================
# 2. The per-metric reason, keyed to the metric it belongs to.
# ===========================================================================


class TestEachRowCarriesItsOwnReason:
    def test_the_reason_is_keyed_to_the_metric_not_only_to_the_page(self):
        gate = ModelFairnessGate(
            metrics=[METRIC, "auroc_parity"],
            thresholds={METRIC: 0.1, "auroc_parity": 0.1},
        )
        decision = _quiet(gate.evaluate, *_fair_arrays())
        report = decision.to_markdown_report()

        assert "### Per-metric notes" in report
        # auroc_parity is not one of the five evaluate() can compute, so it is
        # the could-not-check half; METRIC is measured and passes.
        note_lines = [ln for ln in report.splitlines() if ln.startswith("- **")]
        noted = {ln.split("**")[1] for ln in note_lines}
        assert "auroc_parity" in noted
        assert METRIC not in noted, "a row that passed has no reason to state"

    def test_control_a_clean_decision_has_no_notes_section(self):
        decision = _gate().evaluate(*_fair_arrays())
        assert "Per-metric notes" not in decision.to_markdown_report()

    def test_the_hierarchical_report_keys_its_reason_to_the_level(self):
        y_true, y_pred, _ = _fair_arrays()
        decision = _quiet(
            _gate().evaluate_hierarchical, y_true, y_pred, {"gender": np.array(["a"] * N)}
        )
        report = decision.to_markdown_report()

        # One notes block per level, each under its own level heading, where the
        # page-level Blocking Issues list printed the same sentence twice with
        # nothing saying which level it belonged to.
        assert report.count("### Per-metric notes") == len(decision.level_results)


# ===========================================================================
# 3. The refusal channel cannot itself raise.
# ===========================================================================


class TestTheRefusalChannelSurvivesWhatItIsHanded:
    def test_a_text_typed_metric_still_raises_the_fairness_error(self):
        """Fully reachable: a compute_metrics_fn reading a metrics CSV hands
        this module strings. Measured before: ValueError("Unknown format code
        'f' for object of type 'str'"), for which isinstance(e, AssertionError)
        is False, so parametrize_fairness's own `except FairnessAssertionError`
        never saw it and the per-case failure was never recorded."""
        y_true, y_pred, protected = _fair_arrays()
        suite = FairnessTestSuite(
            protected_attributes=["g"],
            metrics=[METRIC],
            thresholds={METRIC: 0.1},
            compute_metrics_fn=lambda a, b, c: {METRIC: "0.30"},
        )
        results = suite.test_predictions(y_true, y_pred, protected, raise_on_failure=False)
        assert [r.status for r in results] == [Status.SKIPPED], results

        with pytest.raises(FairnessAssertionError) as excinfo:
            _refuse_unless_measured(results, METRIC, 0.1)
        assert isinstance(excinfo.value, AssertionError)
        assert "NOT MEASURABLE" in str(excinfo.value)
        assert "not measured (not a number (str))" in str(excinfo.value)

    @pytest.mark.parametrize(
        "actual, threshold",
        [(None, 0.1), (0.3, None), (None, None), (float("nan"), 0.1), ("0.3", 0.1)],
    )
    def test_constructing_it_never_raises(self, actual, threshold):
        text = str(FairnessAssertionError("m", "dpd", actual, threshold))
        assert "Actual:" in text and "Threshold:" in text

    def test_control_a_measured_breach_still_prints_both_numbers(self):
        """The wording for a real comparison is unchanged, to four places."""
        text = str(FairnessAssertionError("breach", "dpd", 0.3, 0.1))
        assert "  Actual: 0.3000" in text
        assert "  Threshold: 0.1000" in text
        assert "not measured" not in text


# ===========================================================================
# 3b. The monitor's alert line, which is what lands in a log.
# ===========================================================================


class TestTheAlertLineSaysWhichStateItIs:
    def _monitor(self):
        monitor = BiasMonitor(baseline_metrics={METRIC: 0.02}, drift_threshold=0.05)
        monitor.config.min_samples_for_alert = 10
        return monitor

    def test_a_could_not_check_alert_does_not_print_as_a_drift_number(self):
        """BiasMonitor._create_alert passes drift_magnitude=float("nan") for the
        NOT_MEASURABLE state BY DESIGN, and this repr printed it as
        "drift=nan" with drift_type -- the field that holds the answer -- absent
        from the line altogether."""
        monitor = self._monitor()
        y_true, y_pred, _ = _fair_arrays()
        result = _quiet(
            monitor.log_batch, y_pred, y_true, np.array(["a"] * N), batch_id="one_group"
        )

        assert result.could_not_check == [METRIC]
        assert len(result.alerts) == 1
        # The batch record's own three states, on the same window: the third one
        # is named, not collapsed into "OK" or into "DRIFT DETECTED".
        assert "COULD NOT CHECK" in repr(result)
        assert "OK" not in repr(result)
        line = repr(result.alerts[0])
        assert "not_measurable" in line, line
        assert "NOT MEASURED" in line, line
        assert "drift=nan" not in line, line

    def test_control_a_measured_batch_still_reads_ok(self):
        """The over-correction control for the record's repr: a batch where every
        monitored metric WAS measured and did not drift must still read OK."""
        y_true, y_pred, protected = _fair_arrays()
        # The baseline is DERIVED from the same window, so "no drift" is a real
        # zero rather than a number quoted from today's fixture.
        probe = BiasMonitor(baseline_metrics={METRIC: 0.0})
        probe.config.min_samples_for_alert = 10
        measured = _quiet(probe.log_batch, y_pred, y_true, protected).metrics[METRIC]

        monitor = BiasMonitor(baseline_metrics={METRIC: measured}, drift_threshold=0.05)
        monitor.config.min_samples_for_alert = 10
        result = _quiet(monitor.log_batch, y_pred, y_true, protected, batch_id="clean")

        assert result.could_not_check == []
        assert result.drift_detected is False
        assert "OK" in repr(result)
        assert "COULD NOT CHECK" not in repr(result)
        assert result.metrics[METRIC] == pytest.approx(measured)

    def test_a_none_magnitude_prints_instead_of_raising(self):
        alert = DriftAlert(
            alert_id="a",
            severity=AlertSeverity.INFO,
            drift_type=DriftType.METRIC_DRIFT,
            metric_name=METRIC,
            baseline_value=None,
            current_value=None,
            drift_magnitude=None,
        )
        assert "no value was recorded" in repr(alert)

    def test_control_a_measured_drift_still_prints_its_magnitude(self):
        """The over-correction control: a real drift must still read as a
        number, to four places, with its severity."""
        alert = DriftAlert(
            alert_id="a",
            severity=AlertSeverity.CRITICAL,
            drift_type=DriftType.METRIC_DRIFT,
            metric_name=METRIC,
            baseline_value=0.05,
            current_value=0.40,
            drift_magnitude=0.35,
        )
        line = repr(alert)
        assert "drift=0.3500" in line
        assert "critical" in line
        assert "NOT MEASURED" not in line


# ===========================================================================
# 4. The fairness suite's own CI artifact.
# ===========================================================================


def _suite_with(statuses) -> FairnessTestSuite:
    suite = FairnessTestSuite(protected_attributes=["g"])
    suite.results = [
        FairnessTestResult(
            test_name=f"t{i}",
            status=status,
            metric_name=METRIC,
            actual_value=0.02,
            threshold=0.1,
            message="m",
        )
        for i, status in enumerate(statuses)
    ]
    return suite


class TestTheSuiteJunitArtifactCarriesTheSuiteState:
    def test_a_suite_that_ran_nothing_is_not_an_empty_green_file(self):
        """Reached by the misconfiguration parametrize_fairness already fails
        closed on: a suite naming no protected attribute."""
        suite = FairnessTestSuite(protected_attributes=[], metrics=[METRIC])
        assert suite.get_summary()["status"] == "no_tests_run"

        root = ET.fromstring(suite.to_junit_xml())
        assert root.get("tests") == "1"
        assert root.get("failures") == "1"
        failure = root.find('.//testcase[@name="fairness_suite_coverage"]/failure')
        assert failure is not None
        assert "not a pass" in failure.get("message")

    def test_a_run_where_nothing_could_be_measured_is_skipped_not_silent(self):
        suite = _suite_with([Status.SKIPPED, Status.SKIPPED])
        assert suite.get_summary()["status"] == "incomplete"

        root = ET.fromstring(suite.to_junit_xml())
        coverage = root.find('.//testcase[@name="fairness_suite_coverage"]/skipped')
        assert coverage is not None
        assert "incomplete" in coverage.get("message")
        # Counted, so the suite attributes and the testcases agree.
        assert root.get("skipped") == "3"
        assert root.get("failures") == "0"

    def test_control_a_measured_pass_is_still_a_clean_artifact(self):
        suite = _suite_with([Status.PASSED, Status.PASSED])
        root = ET.fromstring(suite.to_junit_xml())

        assert root.get("failures") == "0"
        assert root.get("errors") == "0"
        assert root.get("skipped") == "0"
        assert root.find('.//testcase[@name="fairness_suite_coverage"]/failure') is None
        assert root.find('.//testcase[@name="fairness_suite_coverage"]/skipped') is None
        body = root.find('.//testcase[@name="fairness_suite_coverage"]/system-out').text
        assert "2 of 2" in body

    def test_control_a_measured_failure_is_still_counted_as_one(self):
        root = ET.fromstring(_suite_with([Status.FAILED]).to_junit_xml())
        assert root.get("failures") == "1"
        assert root.find('.//testcase[@name="fairness_suite_coverage"]/failure') is None

    def test_the_coverage_case_is_last_so_the_first_testcase_is_a_result(self):
        root = ET.fromstring(_suite_with([Status.FAILED]).to_junit_xml())
        assert root.find("testcase").get("name") == "t0"


# ===========================================================================
# 5. The validator's CI artifact: four coverage states, not one.
# ===========================================================================


def _validation(checks_run, issues=()):
    return DataValidationResult(
        passed=True, issues=list(issues), metrics={}, summary="s", checks_run=checks_run
    )


class TestTheValidatorJunitArtifactAnswersForItsCoverage:
    def test_an_unrecorded_coverage_is_not_an_empty_green_suite(self):
        """checks_run=None is this dataclass's DEFAULT, so this is every result
        built anywhere other than validate()."""
        result = _validation(None)
        assert result.execution_coverage() == "unrecorded"

        root = ET.fromstring(result.to_junit_xml())
        assert root.get("tests") == "1"
        assert root.get("skipped") == "1"
        skipped = root.find('.//testcase[@name="validation_coverage"]/skipped')
        assert skipped is not None
        assert "could-not-check" in skipped.get("message")

    def test_a_partial_run_names_what_it_did_not_examine(self):
        result = _validation(["representation"])
        assert result.execution_coverage() == "partial"

        skipped = ET.fromstring(result.to_junit_xml()).find(
            './/testcase[@name="validation_coverage"]/skipped'
        )
        assert skipped is not None
        message = skipped.get("message")
        assert "representation ran" in message
        for check in VALIDATION_CHECKS:
            if check != "representation":
                assert check in message, check

    def test_control_a_run_that_checked_nothing_is_still_a_failure(self):
        """Unchanged behaviour, asserted so the wider fix cannot soften it: a
        could-not-check is a SKIP, a recorded no-check-ran is a FAILURE."""
        root = ET.fromstring(_validation([]).to_junit_xml())
        assert root.get("failures") == "1"
        assert root.find('.//testcase[@name="validation_coverage"]/failure') is not None

    def test_control_a_complete_clean_run_says_it_measured(self):
        result = _validation(list(VALIDATION_CHECKS))
        root = ET.fromstring(result.to_junit_xml())

        assert root.get("failures") == "0"
        assert root.get("skipped") == "0"
        body = root.find('.//testcase[@name="validation_coverage"]/system-out').text
        assert "Every validation check ran" in body

    def test_control_a_measured_error_is_still_counted_and_still_parses(self):
        issue = ValidationIssue(
            issue_type='bad<type>&"q"',
            severity=ValidationSeverity.ERROR,
            message='msg with <angle> & "quotes"',
            details={"k": "<v>"},
        )
        root = ET.fromstring(_validation(list(VALIDATION_CHECKS), [issue]).to_junit_xml())

        assert root.get("failures") == "1"
        assert root.get("tests") == "2"
        # The escaping contract from test_junit_xml_escaping, unchanged, and the
        # issue is still the FIRST testcase.
        first = root.find("testcase")
        assert first.get("name") == 'bad<type>&"q"'
        assert first.find("failure").get("message") == 'msg with <angle> & "quotes"'


# ===========================================================================
# 6. Every serialiser carries every field it declares.
# ===========================================================================
#
# "A copy constructor drops every later field" is how summary_decision was lost
# from IntersectionalGateDecision.to_dict once already, and a three-state
# disclosure that never reaches the dict is a could-not-check no consumer can
# read. One pin over all thirteen dataclasses in this package, so a field added
# to any of them has to be carried or explain itself.


def _every_dataclass():
    import dataclasses

    from vfairness.operations.cicd.gate import (
        GateConfig,
        HierarchicalGateConfig,
        SmallSampleWarning,
    )
    from vfairness.operations.cicd.gate import IntersectionalGateDecision as Hier
    from vfairness.operations.cicd.monitor import MonitorConfig, MonitoringResult
    from vfairness.operations.cicd.validator import DataValidationConfig

    instances = [
        GateConfig(),
        HierarchicalGateConfig(),
        SmallSampleWarning("A", 5, 30),
        MetricEvaluation("m", 0.1, 0.2, True),
        GateDecision(True, GateStatus.APPROVED, []),
        Hier(True, GateStatus.APPROVED),
        MonitorConfig(),
        DriftAlert("i", AlertSeverity.INFO, DriftType.METRIC_DRIFT, "m", 0.1, 0.2, 0.1),
        MonitoringResult("b", {}, False, [], 0),
        FairnessTestResult("t", Status.PASSED, "m"),
        DataValidationConfig(),
        DataValidationResult(True, [], {}, "s"),
        ValidationIssue("t", ValidationSeverity.INFO, "m"),
    ]
    return [(o, {f.name for f in dataclasses.fields(o)}) for o in instances]


@pytest.mark.parametrize("instance, declared", _every_dataclass())
def test_to_dict_carries_every_declared_field(instance, declared):
    keys = set(instance.to_dict())
    assert declared <= keys, f"{type(instance).__name__}.to_dict drops {sorted(declared - keys)}"


def test_every_to_dict_survives_a_json_round_trip():
    """A dict a CI integration cannot serialise is a report nobody receives."""
    import json

    for instance, _ in _every_dataclass():
        json.dumps(instance.to_dict(), default=str)
