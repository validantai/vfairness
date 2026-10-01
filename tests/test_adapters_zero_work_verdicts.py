"""Zero executed work must never render as a green all-clear.

Three adapters presented a complete, confident analysis for input that
established nothing:

* ``cicd_pipeline_to_svg()`` rendered "DATA VALIDATION PASS / 0 errors, 0
  warnings", "FAIRNESS TESTS ALL PASS / 0 passed, 0 failed" and "All validation
  checks passed successfully" for a pipeline in which no stage ran.
* ``threshold_optimization_to_svg({})`` rendered a green FEASIBLE badge, a
  "100% reduction" scale and "ACCURACY CHANGE +0.00%" across 0 groups.
* ``auto_discovery_to_svg([], [], [])`` rendered "No significant fairness issues
  auto-discovered" for a scan that examined no attribute and no group.

The shape is the same in all three: zero of zero reads as complete success, the
same defect as a regression headline that printed EQUITABLE because 0 == 0, and
as ``identify_proxy_features`` reporting "no proxies" from a scan that could not
look at all. An SVG is an export format; it is handed to auditors and regulators
and it outlives the run that produced it, so "nothing failed" and "nothing ran"
must never render the same.

Every could-not-check test here is paired with a POSITIVE CONTROL asserting that
a genuine all-pass run keeps its green all-pass. The assertions deliberately
target marks the adapters and their own templates put on the canvas, not the
prose that :mod:`vfairness.rendering.explain` composes into the explanation
panel.
"""

from enum import Enum
from types import SimpleNamespace

import pytest

from vfairness.rendering.adapters import cicd_pipeline_to_svg
from vfairness.rendering.adapters_discovery import auto_discovery_to_svg
from vfairness.rendering.adapters_post_processing import threshold_optimization_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


# ── CI/CD pipeline fixtures ─────────────────────────────────────────────────


class _Status(Enum):
    passed = "passed"
    failed = "failed"


class _GateStatus(Enum):
    approved = "approved"
    rejected = "rejected"


def _test_result(name, status, value, threshold):
    return SimpleNamespace(
        test_name=name,
        status=status,
        metric_name="DP",
        metric_value=value,
        threshold=threshold,
    )


CLEAN_VALIDATION = SimpleNamespace(passed=True, errors=[], warnings=[])
PASSING_TESTS = [
    _test_result("test_demographic_parity[gender]", _Status.passed, 0.08, 0.10),
    _test_result("test_equal_opportunity[gender]", _Status.passed, 0.06, 0.10),
]
FAILING_TESTS = PASSING_TESTS + [
    _test_result("test_equalized_odds[gender]", _Status.failed, 0.22, 0.15)
]
APPROVING_GATE = SimpleNamespace(
    status=_GateStatus.approved, approved=True, blocking_reasons=[], warnings=[]
)
BLOCKING_GATE = SimpleNamespace(
    status=_GateStatus.rejected,
    approved=False,
    blocking_reasons=["Equalized odds exceeds threshold (0.22 > 0.15)"],
    warnings=[],
)


class TestCicdPipelineThatExecutedNothing:
    def test_an_unrun_validator_is_not_reported_as_a_data_validation_pass(self):
        svg = cicd_pipeline_to_svg()

        assert ">PASS<" not in svg
        assert "0 errors, 0 warnings" not in svg
        assert "NOT RUN" in svg
        assert "no result supplied" in svg

    def test_zero_executed_tests_are_not_reported_as_all_pass(self):
        svg = cicd_pipeline_to_svg()

        assert "ALL PASS" not in svg
        assert "0 passed, 0 failed" not in svg
        assert "no test executed" in svg

    def test_the_validation_panel_does_not_claim_a_clean_dataset(self):
        svg = cicd_pipeline_to_svg()

        assert "All validation checks passed successfully." not in svg
        assert "Data validation did not run" in svg
        assert "nothing here reports a clean dataset" in svg

    def test_an_absent_gate_decision_neither_approves_nor_blocks(self):
        svg = cicd_pipeline_to_svg()

        assert "APPROVED" not in svg
        assert "No gate decision was supplied" in svg
        assert "neither approves nor blocks the model" in svg

    def test_a_partial_pipeline_still_names_the_stages_that_did_not_run(self):
        """Tests alone are not a gate report: the other two stages are disclosed."""
        svg = cicd_pipeline_to_svg(test_results=PASSING_TESTS)

        assert "ALL PASS" in svg  # the stage that DID run keeps its verdict
        assert "Data validation did not run" in svg
        assert "No gate decision was supplied" in svg

    def test_nothing_ran_and_nothing_failed_do_not_render_the_same(self):
        nothing_ran = cicd_pipeline_to_svg()
        nothing_failed = cicd_pipeline_to_svg(
            validation_result=CLEAN_VALIDATION,
            test_results=PASSING_TESTS,
            gate_decision=APPROVING_GATE,
        )

        assert nothing_ran != nothing_failed
        assert "ALL PASS" in nothing_failed and "ALL PASS" not in nothing_ran


class TestCicdPipelinePositiveControl:
    """A genuine all-pass run must keep every green mark it earned."""

    def test_a_fully_executed_passing_run_still_renders_its_green_all_pass(self):
        svg = cicd_pipeline_to_svg(
            validation_result=CLEAN_VALIDATION,
            test_results=PASSING_TESTS,
            gate_decision=APPROVING_GATE,
        )

        assert ">PASS<" in svg
        assert "0 errors, 0 warnings" in svg
        assert "ALL PASS" in svg
        assert "2 passed, 0 failed" in svg
        assert "All validation checks passed successfully." in svg
        assert "APPROVED" in svg
        assert "NOT RUN" not in svg

    def test_a_failing_test_still_blocks_and_still_reads_as_a_failure(self):
        svg = cicd_pipeline_to_svg(
            validation_result=CLEAN_VALIDATION,
            test_results=FAILING_TESTS,
            gate_decision=BLOCKING_GATE,
        )

        assert "FAILURES" in svg
        assert "2 passed, 1 failed" in svg
        assert "BLOCKED" in svg
        assert "Equalized odds exceeds threshold" in svg
        assert "NOT RUN" not in svg


# ── Threshold optimization ──────────────────────────────────────────────────


FULL_THRESHOLD_RESULT = {
    "constraint_type": "demographic_parity",
    "tolerance": 0.05,
    "groups": [
        {
            "name": "Male",
            "threshold": 0.55,
            "original_rate": 0.72,
            "optimized_rate": 0.63,
            "size": 2750,
        },
        {
            "name": "Female",
            "threshold": 0.42,
            "original_rate": 0.58,
            "optimized_rate": 0.64,
            "size": 2000,
        },
    ],
    "original_disparity": 0.14,
    "optimized_disparity": 0.01,
    "original_accuracy": 0.852,
    "optimized_accuracy": 0.841,
    "is_feasible": True,
}


class TestThresholdOptimizationWithNoResult:
    def test_an_empty_result_does_not_render_a_green_feasible(self):
        svg = threshold_optimization_to_svg({})

        assert ">FEASIBLE<" not in svg
        assert ">INFEASIBLE<" not in svg
        assert "NOT ESTABLISHED" in svg

    def test_an_unmeasured_disparity_is_not_a_disparity_of_zero(self):
        svg = threshold_optimization_to_svg({})

        assert ">0.000<" not in svg
        assert "100% reduction" not in svg
        assert "no disparity figure was supplied" in svg
        assert "NOT MEASURED" in svg

    def test_an_uncomputed_accuracy_cost_is_not_a_cost_of_zero(self):
        svg = threshold_optimization_to_svg({})

        assert "+0.00%" not in svg
        assert "NOT MEASURED" in svg

    def test_the_recommendation_does_not_tell_the_reader_to_deploy(self):
        svg = threshold_optimization_to_svg({})

        assert "Use optimized group-specific thresholds" not in svg
        # The panel word-wraps into one <text> per line, so assert on a phrase
        # that stays inside a single line.
        assert "No threshold optimisation result was supplied" in svg
        assert "does not recommend deploying" in svg

    def test_a_result_object_that_carries_nothing_is_treated_the_same(self):
        class EmptyResult:
            def to_dict(self):
                return {}

        svg = threshold_optimization_to_svg(EmptyResult())

        assert "NOT ESTABLISHED" in svg
        assert ">FEASIBLE<" not in svg

    def test_nothing_optimised_and_nothing_to_optimise_do_not_render_the_same(self):
        nothing_ran = threshold_optimization_to_svg({})
        measured = threshold_optimization_to_svg(FULL_THRESHOLD_RESULT)

        assert nothing_ran != measured
        assert "NOT ESTABLISHED" in nothing_ran
        assert "NOT ESTABLISHED" not in measured


class TestThresholdOptimizationPositiveControl:
    def test_a_measured_optimisation_keeps_its_feasible_verdict_and_its_numbers(self):
        svg = threshold_optimization_to_svg(FULL_THRESHOLD_RESULT)

        assert ">FEASIBLE<" in svg
        assert "0.140" in svg and "0.010" in svg
        assert "92.9%" in svg
        assert "100% reduction" in svg
        assert "-1.10%" in svg
        assert "NOT MEASURED" not in svg

    def test_a_measured_zero_disparity_still_reports_a_measured_zero(self):
        """The fix must not swallow a genuine 0: supplied is supplied."""
        svg = threshold_optimization_to_svg(
            {
                "original_disparity": 0.0,
                "optimized_disparity": 0.0,
                "original_accuracy": 0.9,
                "optimized_accuracy": 0.9,
                "is_feasible": True,
            }
        )

        assert ">FEASIBLE<" in svg
        assert "0.0%" in svg
        assert "NOT MEASURED" not in svg

    def test_an_infeasible_result_still_renders_infeasible(self):
        svg = threshold_optimization_to_svg({**FULL_THRESHOLD_RESULT, "is_feasible": False})

        assert ">INFEASIBLE<" in svg
        assert "NOT ESTABLISHED" not in svg


# ── Auto-discovery ──────────────────────────────────────────────────────────


DISCOVERY_CANDIDATES = [{"name": "zipcode", "confidence": 0.82, "category": "proxy"}]
DISCOVERY_VIOLATION = {
    "attribute": "gender",
    "metric": "demographic_parity",
    "value": 0.15,
    "threshold": 0.10,
    "severity": "high",
}
PROFILED_GROUPS = [
    {"group": "a", "positive_rate": 0.40, "size": 500, "relative_to_overall": 1.05},
    {"group": "b", "positive_rate": 0.39, "size": 480, "relative_to_overall": 0.97},
]


class TestAutoDiscoveryThatExaminedNothing:
    def test_a_scan_that_examined_nothing_is_not_a_clean_scan(self):
        svg = auto_discovery_to_svg([], [], [])

        assert "No significant fairness issues auto-discovered." not in svg
        assert "nothing was examined" in svg
        assert "no issue can be reported as absent" in svg

    def test_the_violations_band_carries_the_third_state(self):
        svg = auto_discovery_to_svg([], [], [])

        assert "NOT SCANNED" in svg
        assert "0 high severity" not in svg

    def test_a_violation_scan_that_did_not_run_is_not_a_green_zero(self):
        """Attributes were found, but nothing was scanned for violations."""
        svg = auto_discovery_to_svg(DISCOVERY_CANDIDATES, None, None)

        assert "NOT SCANNED" in svg
        assert "0 high severity" not in svg

    def test_nothing_examined_and_nothing_found_do_not_render_the_same(self):
        examined_nothing = auto_discovery_to_svg([], [], [])
        found_nothing = auto_discovery_to_svg([], [], PROFILED_GROUPS)

        assert examined_nothing != found_nothing
        assert "NOT SCANNED" in examined_nothing
        assert "NOT SCANNED" not in found_nothing


class TestAutoDiscoveryPositiveControl:
    def test_a_scan_that_profiled_groups_and_found_nothing_still_says_so(self):
        svg = auto_discovery_to_svg([], [], PROFILED_GROUPS)

        assert "No significant fairness issues auto-discovered." in svg
        assert "0 high severity" in svg
        assert "NOT SCANNED" not in svg

    def test_a_recorded_violation_still_renders_its_red_count(self):
        svg = auto_discovery_to_svg(DISCOVERY_CANDIDATES, [DISCOVERY_VIOLATION], PROFILED_GROUPS)

        assert "NOT SCANNED" not in svg
        assert "1 high severity" in svg
        assert "1 high-severity violation(s)" in svg
