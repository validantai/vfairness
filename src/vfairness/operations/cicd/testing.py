"""
Fairness Test Suite for pytest Integration.

This module provides pytest-compatible testing utilities for fairness validation,
enabling test-driven bias prevention in ML workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

import numpy as np
import pandas as pd

from vfairness._triage import is_measured, unmeasurable_reason
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)


class TestStatus(Enum):
    """Status of a fairness test."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class FairnessTestResult:
    """Result of a single fairness test.

    Attributes:
        test_name: Name of the test.
        status: Test status (passed, failed, skipped, error).
        metric_name: Name of the fairness metric tested.
        actual_value: Computed metric value.
        threshold: Threshold the metric was compared against.
        message: Human-readable result message.
        details: Additional test details.
    """

    test_name: str
    status: TestStatus
    metric_name: str
    actual_value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if the test passed."""
        return self.status == TestStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "metric_name": self.metric_name,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"FairnessTestResult({self.test_name}: {self.status.value})"


def _number_or_reason(value: Any) -> str:
    """A measured number formatted, or one clause naming why there is none.

    G05, 2026-09-30. Both header lines of :class:`FairnessAssertionError` were
    ``f"{value:.4f}"``, and this class IS the refusal channel: every path in this
    module that declines to certify something raises it. ``None`` raised
    ``TypeError: unsupported format string passed to NoneType.__format__`` and a
    TEXT-typed value raised ``ValueError: Unknown format code 'f' for object of
    type 'str'``, so constructing the refusal destroyed it.

    Neither is a hypothetical. ``FairnessTestResult.actual_value`` and
    ``.threshold`` are ``Optional[float]`` and DEFAULT to None, and a
    ``compute_metrics_fn`` reading a metrics CSV hands this module strings,
    which is what an ordinary ``read_csv`` gives you. Measured on a
    ``compute_metrics_fn`` returning ``{"demographic_parity_difference":
    "0.30"}``: the suite correctly routed it to SKIPPED with
    ``actual_value='0.30'``, and ``_refuse_unless_measured`` on that result
    raised ``ValueError``, for which ``isinstance(e, AssertionError)`` is
    **False**. ``parametrize_fairness`` catches ``FairnessAssertionError`` to
    record each case as failed, so the refusal did not merely print badly: it
    escaped the handler that logs it and the per-case failure was never
    recorded.

    The four call sites in this module each work around this by hand
    (``float("nan") if x is None else x`` three times, two bare ``assert x is
    not None``, which ``python -O`` removes). Those are left in place as
    defence in depth; the rule now lives in the class so a caller that does not
    know about them cannot lose a refusal.
    """
    if is_measured(value):
        return f"{float(value):.4f}"
    return f"not measured ({unmeasurable_reason(value)})"


class FairnessAssertionError(AssertionError):
    """Exception raised when a fairness assertion fails.

    This integrates with pytest to provide detailed failure information.

    Constructing it can never raise: see :func:`_number_or_reason` for the
    measured before-state, in which a text-typed metric turned a refusal into a
    ``ValueError`` that is not an ``AssertionError`` at all.
    """

    def __init__(
        self,
        message: str,
        metric_name: str,
        actual_value: float,
        threshold: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.metric_name = metric_name
        self.actual_value = actual_value
        self.threshold = threshold
        self.details = details or {}

        full_message = (
            f"{message}\n"
            f"  Metric: {metric_name}\n"
            f"  Actual: {_number_or_reason(actual_value)}\n"
            f"  Threshold: {_number_or_reason(threshold)}"
        )
        super().__init__(full_message)


def _attr(value: Any) -> str:
    """Quote an arbitrary value for use as an XML attribute.

    Returns the value WITH its surrounding quotes, because quoteattr picks the
    quote character that does not appear in the payload. A metric name or an
    assertion message is caller-controlled text and routinely contains `&`,
    `<`, `>` or a quote, and interpolating it raw produced XML that no CI
    consumer could parse. Measured 2026-09-07 on a message reading
    ``gap 0.3 > 0.1 for group "A" & <B>``: ElementTree refused the document with
    "not well-formed (invalid token)". The quote is the sharper half, since it
    ends the attribute early and lets text become markup.
    """
    return _xml_quoteattr("" if value is None else str(value))


def _text(value: Any) -> str:
    """Escape an arbitrary value for use as XML character data."""
    return _xml_escape("" if value is None else str(value))


class FairnessTestSuite:
    """pytest-compatible test suite for fairness validation.

    This class provides a structured way to define and run fairness tests
    that integrate seamlessly with pytest and CI/CD pipelines.

    Attributes:
        protected_attributes: List of protected attribute column names.
        metrics: List of fairness metrics to test.
        thresholds: Threshold values for each metric.
        results: List of test results (populated after running tests).

    Example (in conftest.py):
        >>> @pytest.fixture
        ... def fairness_suite():
        ...     return FairnessTestSuite(
        ...         protected_attributes=['gender'],
        ...         metrics=['demographic_parity_difference'],
        ...         thresholds={'demographic_parity_difference': 0.1}
        ...     )

    Example (in test_fairness.py):
        >>> def test_model_fairness(fairness_suite, trained_model, test_data):
        ...     fairness_suite.test_model(trained_model, test_data)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: fairness_test_suite. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: List[str],
        metrics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        compute_metrics_fn: Optional[Callable] = None,
    ):
        """Initialize the FairnessTestSuite.

        Args:
            protected_attributes: List of protected attribute column names.
            metrics: List of fairness metrics to test.
            thresholds: Dictionary mapping metric names to threshold values.
            compute_metrics_fn: Optional custom function to compute metrics.
        """
        self.protected_attributes = protected_attributes
        self.metrics = metrics or ["demographic_parity_difference"]
        self.thresholds = thresholds or {"demographic_parity_difference": 0.1}
        self.compute_metrics_fn = compute_metrics_fn
        self.results: List[FairnessTestResult] = []

    def test_model(
        self,
        model: Any,
        test_data: Union[pd.DataFrame, tuple],
        target_column: Optional[str] = None,
        raise_on_failure: bool = True,
    ) -> List[FairnessTestResult]:
        """Test a model for fairness.

        Args:
            model: Trained model with a predict method.
            test_data: DataFrame or (X, y) tuple of test data.
            target_column: Name of target column (if using DataFrame).
            raise_on_failure: Whether to raise exception on test failure.

        Returns:
            List of FairnessTestResult objects.

        Raises:
            FairnessAssertionError: If any test fails and raise_on_failure is True.
        """
        # Extract data
        if isinstance(test_data, pd.DataFrame):
            if target_column is None:
                raise ValueError("target_column required when test_data is DataFrame")
            y_true = test_data[target_column].values
            # Drop target and protected attribute columns before prediction
            cols_to_drop = [target_column] + [
                a for a in self.protected_attributes if a in test_data.columns
            ]
            X = test_data.drop(columns=cols_to_drop)
            df = test_data
        else:
            X, y_true = test_data
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                df["_target"] = y_true
            else:
                raise ValueError("X must be a DataFrame when using tuple input")

        # Get predictions
        y_pred = model.predict(X)

        # Run tests for each protected attribute
        self.results = []

        for attr in self.protected_attributes:
            if attr not in df.columns:
                self.results.append(
                    FairnessTestResult(
                        test_name=f"fairness_{attr}",
                        status=TestStatus.ERROR,
                        metric_name="N/A",
                        message=f"Protected attribute '{attr}' not found in data",
                    )
                )
                continue

            protected_values = df[attr].values

            # Compute and test each metric
            attr_results = self._test_metrics(
                y_true, y_pred, protected_values, attr, raise_on_failure
            )
            self.results.extend(attr_results)

        return self.results

    def test_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        attr_name: str = "protected_attribute",
        raise_on_failure: bool = True,
    ) -> List[FairnessTestResult]:
        """Test predictions for fairness.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            protected_attr: Protected attribute values.
            attr_name: Name of the protected attribute for reporting.
            raise_on_failure: Whether to raise exception on test failure.

        Returns:
            List of FairnessTestResult objects.
        """
        self.results = self._test_metrics(
            y_true, y_pred, protected_attr, attr_name, raise_on_failure
        )
        return self.results

    def _test_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        attr_name: str,
        raise_on_failure: bool,
    ) -> List[FairnessTestResult]:
        """Test all configured metrics."""
        results = []

        # Compute metrics
        if self.compute_metrics_fn:
            computed = self.compute_metrics_fn(y_true, y_pred, protected_attr)
        else:
            computed = self._compute_default_metrics(y_true, y_pred, protected_attr)

        for metric_name in self.metrics:
            test_name = f"test_{metric_name}_{attr_name}"

            if metric_name not in computed:
                results.append(
                    FairnessTestResult(
                        test_name=test_name,
                        status=TestStatus.SKIPPED,
                        metric_name=metric_name,
                        message=f"Could not compute metric '{metric_name}'",
                    )
                )
                continue

            value = computed[metric_name]
            threshold = self.thresholds.get(metric_name)

            # A NaN metric could not be measured on this data (undefined group
            # rate, degenerate data, or a custom metric reporting insufficient
            # evidence). abs(nan) <= threshold is False, so it used to fall into
            # the FAILED branch with the misleading message "nan exceeds
            # threshold". A could-not-measure is neither a PASS nor a FAIL:
            # route it to SKIPPED with an honest message (three states, never
            # two), mirroring the gate's fail-closed NaN handling.
            # Infinity too, added READINESS-6: it does not fall into the
            # FAILED branch the way NaN did, it SATISFIES a higher-is-better
            # bound and reports a PASSING test over a metric nobody measured.
            # With every layer NaN-only this suite reported status=PASSED with
            # the message "disparate_impact_ratio = inf (threshold: 0.8000)".
            #
            # Defence in depth, not the carrier: measured by sabotage,
            # reinstating the NaN-only test here changes nothing on its own,
            # because check_threshold now answers COULD_NOT_CHECK for a
            # non-finite value and this loop routes that to SKIPPED. Kept so the
            # two paths cannot drift apart.
            if not is_measured(value):
                results.append(
                    FairnessTestResult(
                        test_name=test_name,
                        status=TestStatus.SKIPPED,
                        metric_name=metric_name,
                        actual_value=value,
                        threshold=threshold,
                        message=(
                            f"{metric_name} could not be measured on this data "
                            f"({unmeasurable_reason(value)})"
                        ),
                        details={"protected_attribute": attr_name},
                    )
                )
                continue

            if threshold is None:
                results.append(
                    FairnessTestResult(
                        test_name=test_name,
                        status=TestStatus.SKIPPED,
                        metric_name=metric_name,
                        actual_value=value,
                        message=f"No threshold defined for '{metric_name}'",
                    )
                )
                continue

            # Test against threshold, in THIS metric's direction. The old
            # `abs(value) <= threshold` treated every metric as lower-is-better,
            # so the whole ratio family was inverted: a disparate_impact_ratio of
            # 0.50 (a clear four-fifths violation) satisfied 0.50 <= 0.80 and was
            # reported as PASSED, while 1.00 (perfect parity) was reported FAILED.
            outcome, threshold_message = check_threshold(metric_name, value, threshold)

            if outcome is ThresholdOutcome.COULD_NOT_CHECK:
                # Could-not-check is neither PASS nor FAIL (three states, never
                # two): a metric whose better-direction is unknown cannot be
                # graded against a threshold at all. SKIPPED with an honest
                # message, the same route this suite already takes for a NaN
                # value; get_summary() reports a run with no real PASS as
                # 'incomplete' rather than green.
                results.append(
                    FairnessTestResult(
                        test_name=test_name,
                        status=TestStatus.SKIPPED,
                        metric_name=metric_name,
                        actual_value=value,
                        threshold=threshold,
                        message=threshold_message,
                        details={"protected_attribute": attr_name},
                    )
                )
                continue

            passed = outcome is ThresholdOutcome.PASS

            if passed:
                results.append(
                    FairnessTestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        metric_name=metric_name,
                        actual_value=value,
                        threshold=threshold,
                        message=f"{metric_name} = {value:.4f} (threshold: {threshold:.4f})",
                    )
                )
            else:
                result = FairnessTestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    metric_name=metric_name,
                    actual_value=value,
                    threshold=threshold,
                    # Direction-accurate wording: a ratio breach is BELOW its
                    # minimum, so the old fixed "exceeds threshold" text described
                    # the opposite of what happened.
                    message=threshold_message,
                    details={"protected_attribute": attr_name},
                )
                results.append(result)

                if raise_on_failure:
                    raise FairnessAssertionError(
                        f"Fairness test failed for {attr_name}",
                        metric_name=metric_name,
                        actual_value=value,
                        threshold=threshold,
                        details={"protected_attribute": attr_name},
                    )

        return results

    def _compute_default_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """Compute default fairness metrics."""
        metrics: Dict[str, float] = {}
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return metrics

        # Worst-case (max - min) gaps across ALL groups. Truncating to the
        # first two groups (previous behaviour) let a fairness assertion PASS
        # while a third or later group was grossly disadvantaged. A group with
        # an undefined rate (empty denominator) makes the worst-case gap itself
        # unmeasurable, so it is reported as NaN and _test_metrics routes it to
        # SKIPPED, never a silently-dropped group or a false 0.0 (mirror of the
        # gate's fail-closed NaN handling and the fd9746b metric fix).
        #
        # BGL5 A-operations-3, 2026-09-27. The paragraph above was TRUE ONLY WHEN
        # THE UNMEASURABLE GROUP SORTED FIRST. Builtin max()/min() do not
        # propagate NaN: they keep the accumulator whenever the comparison
        # against NaN answers False, so a NaN anywhere but the first position
        # vanishes and the gap over the SURVIVING groups was published as the
        # worst-case gap across ALL groups. Measured on three groups of 10 rows
        # with every prediction for group C NaN (the shape a model that cannot
        # score a group's rows produces):
        #   before  max(dp_rates) - min(dp_rates) -> 0.0, so test_predictions
        #           returned [('passed', 0.0)], get_summary() said 'passed',
        #           assert_fairness returned None and the fairness_test
        #           decorator raised nothing: four green gates over a group
        #           nobody measured.
        #   after   float(np.max(...)) - float(np.min(...)) -> nan, which
        #           _test_metrics routes to SKIPPED and both gate entry points
        #           refuse via _refuse_unless_measured.
        # The same data with the NaN group FIRST already answered nan, which is
        # why every existing pin passed. np.max/np.min propagate NaN; this is the
        # mechanism already pinned for two other modules by
        # tests/test_readiness6_names.py::test_python_max_really_does_swallow_nan
        # ("pinned so nobody 'simplifies' np.maximum back to max"). Healthy input
        # is untouched: two groups at rates 0.0 and 1.0 still measure 1.0.
        dp_rates = [float(np.mean(y_pred[protected_attr == g])) for g in groups]
        metrics["demographic_parity_difference"] = float(np.max(dp_rates)) - float(np.min(dp_rates))

        # Equalized odds (worst-case TPR gap): a group with no positives has an
        # undefined TPR -> NaN for the whole gap.
        tpr_rates = []
        for g in groups:
            pos = (protected_attr == g) & (y_true == 1)
            tpr_rates.append(float(np.mean(y_pred[pos])) if np.sum(pos) > 0 else float("nan"))
        if any(r != r for r in tpr_rates):
            metrics["equalized_odds_difference"] = float("nan")
        elif len(tpr_rates) >= 2:
            metrics["equalized_odds_difference"] = max(tpr_rates) - min(tpr_rates)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        if not self.results:
            return {"status": "no_tests_run", "results": []}

        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

        # Three states, never two. A batch with a failure or error is 'failed'; a
        # batch where nothing could be measured (every metric SKIPPED, e.g. an
        # undefined group rate) is 'incomplete' (could-not-verify), NOT 'passed' --
        # collapsing an all-unmeasurable run into a green pass is the same fail-open
        # the deployment gate refuses. Only a run with at least one real PASS and no
        # failures is 'passed'.
        if failed > 0 or errors > 0:
            status = "failed"
        elif passed == 0 and skipped > 0:
            status = "incomplete"
        else:
            status = "passed"

        return {
            "status": status,
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "results": [r.to_dict() for r in self.results],
        }

    def to_junit_xml(self) -> str:
        """Export results in JUnit XML format.

        Carries the SUITE-LEVEL state as a testcase of its own, because the XML
        is read by a machine that cannot call :meth:`get_summary`.

        G05, 2026-09-30. ``get_summary()`` has three states on purpose and its
        own comment says why ("collapsing an all-unmeasurable run into a green
        pass is the same fail-open the deployment gate refuses"); this exporter
        published none of them. Measured on a suite built with
        ``protected_attributes=[]`` (the exact misconfiguration
        ``parametrize_fairness`` fails closed on, recorded at its own call site),
        against a model and 200 rows::

            test_model(...) -> []
            get_summary()['status'] -> 'no_tests_run'
            to_junit_xml() -> tests="0" failures="0" errors="0" skipped="0"

        which every CI UI paints green, and which is byte-identical to a suite
        that measured nothing because there was nothing wrong. It is the same
        false all-clear ``DataValidationResult.to_junit_xml`` grew a
        ``validation_coverage`` case for, and the rule this module already states
        in :func:`_refuse_unless_measured`: "An EMPTY result list is the same
        refusal: no result is not a pass."

        A run where every result was a could-not-check is SKIPPED rather than a
        failure, because it did not measure a problem either. Collapsing it into
        the failure would contradict ``get_summary``'s third state, which is the
        over-correction this library refuses in the other direction.
        """
        total = len(self.results) + 1  # +1 for the suite_coverage case below
        failures = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

        suite_status = self.get_summary()["status"]
        if suite_status == "no_tests_run":
            failures += 1
        elif suite_status == "incomplete":
            skipped += 1

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="FairnessTests" tests="{total}" failures="{failures}" errors="{errors}" skipped="{skipped}">',
        ]

        for result in self.results:
            lines.append(
                f'  <testcase name={_attr(result.test_name)} classname="FairnessTestSuite">'
            )

            if result.status == TestStatus.FAILED:
                lines.append(f"    <failure message={_attr(result.message)}>")
                lines.append(f"      Metric: {_text(result.metric_name)}")
                lines.append(f"      Actual: {_text(result.actual_value)}")
                lines.append(f"      Threshold: {_text(result.threshold)}")
                lines.append("    </failure>")
            elif result.status == TestStatus.ERROR:
                lines.append(f"    <error message={_attr(result.message)}/>")
            elif result.status == TestStatus.SKIPPED:
                lines.append(f"    <skipped message={_attr(result.message)}/>")

            lines.append("  </testcase>")

        # LAST, so a consumer reading the first testcase still reads the first
        # RESULT.
        lines.append('  <testcase name="fairness_suite_coverage" classname="FairnessTestSuite">')
        if suite_status == "no_tests_run":
            lines.append(
                "    <failure message="
                + _xml_quoteattr(
                    "No fairness test ran, so nothing was compared against any "
                    "threshold. An empty result list is not a pass. Check that the "
                    "suite names at least one protected attribute present in the data."
                )
                + "/>"
            )
        elif suite_status == "incomplete":
            lines.append(
                "    <skipped message="
                + _xml_quoteattr(
                    f"Not one of the {len(self.results)} fairness test(s) produced a "
                    f"measured comparison: every result was a could-not-check, so this "
                    f"run is incomplete and the absence of failures below is not a pass."
                )
                + "/>"
            )
        else:
            lines.append(
                f"    <system-out>{_text(suite_status)}: "
                f"{sum(1 for r in self.results if r.status == TestStatus.PASSED)} of "
                f"{len(self.results)} fairness test(s) were measured comparisons that "
                f"passed.</system-out>"
            )
        lines.append("  </testcase>")

        lines.append("</testsuite>")
        return "\n".join(lines)


def _refuse_unless_measured(
    results: List[FairnessTestResult],
    metric: str,
    threshold: float,
    message: Optional[str] = None,
) -> None:
    """Raise unless every result is a real comparison against a threshold.

    Three states, never two: a measured breach FAILS (the caller's own business),
    a metric that could not be measured also fails, loudly and with a different
    message, and neither is ever a pass. An EMPTY result list is the same
    refusal: no result is not a pass, it means the suite produced nothing to
    judge.

    Shared by :func:`assert_fairness` and the :func:`fairness_test` decorator so
    the two gate entry points CANNOT DRIFT APART. They had drifted: the wording
    and the fail-closed rule below were written for ``assert_fairness`` on
    2026-09-07 and the decorator never got either, so the identical degenerate
    input raised on one path and returned green on the other.
    """
    if not results:
        raise FairnessAssertionError(
            message
            or (
                f"NOT MEASURABLE: the fairness suite returned no result for "
                f"'{metric}', so it was never compared against threshold "
                f"{threshold} (fail closed)."
            ),
            metric_name=metric,
            actual_value=float("nan"),
            threshold=threshold,
        )

    for result in results:
        # Anything that is not an explicit PASS is a refusal. SKIPPED is the
        # live case and it arrives from two directions indistinguishable here:
        # a metric undefined on this data, and a metric NAME that does not
        # exist. Both returned None and passed.
        if result.status in (TestStatus.SKIPPED, TestStatus.ERROR):
            raise FairnessAssertionError(
                message
                or (
                    f"NOT MEASURABLE: '{result.metric_name}' came back "
                    f"{result.status.name} ({result.message or 'no reason given'}), "
                    f"so it was never compared against threshold {threshold} "
                    f"(fail closed). If the name is a typo, fix the name; if the "
                    f"metric is undefined on this data, that is a real result and "
                    f"the gate cannot approve on it."
                ),
                metric_name=result.metric_name,
                actual_value=(
                    result.actual_value if result.actual_value is not None else float("nan")
                ),
                threshold=result.threshold if result.threshold is not None else threshold,
            )


def fairness_test(
    metric: str,
    threshold: float,
    protected_attribute: str,
):
    """Decorator to mark a test function as a fairness test.

    This decorator integrates with pytest to provide rich fairness test reporting.

    Args:
        metric: Name of the fairness metric to test.
        threshold: Maximum acceptable value for the metric.
        protected_attribute: Name of the protected attribute.

    Example:
        >>> @fairness_test(
        ...     metric='demographic_parity_difference',
        ...     threshold=0.1,
        ...     protected_attribute='gender'
        ... )
        ... def test_loan_model_fairness(model, test_data):
        ...     y_pred = model.predict(test_data.drop('target', axis=1))
        ...     return test_data['target'], y_pred, test_data['gender']
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get y_true, y_pred, protected from the test function
            result = func(*args, **kwargs)

            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError(
                    "Fairness test function must return (y_true, y_pred, protected_attr)"
                )

            y_true, y_pred, protected_attr = result

            # Compute metric
            suite = FairnessTestSuite(
                protected_attributes=[protected_attribute],
                metrics=[metric],
                thresholds={metric: threshold},
            )

            suite.test_predictions(
                y_true=np.asarray(y_true),
                y_pred=np.asarray(y_pred),
                protected_attr=np.asarray(protected_attr),
                attr_name=protected_attribute,
                raise_on_failure=True,
            )

            # FAIL CLOSED, the half ``raise_on_failure=True`` above does not
            # cover. pytest judges the decorated test purely by whether it
            # raises, so a SKIPPED result made the test GREEN over a metric that
            # was never compared to anything. Measured 2026-09-27 (BGL3
            # operations-2) on this decorator, both cases returning a single
            # SKIPPED result and NO exception, so the test passed:
            #   * one group in protected_attr, so no between-group comparison
            #     exists: "Could not compute metric
            #     'demographic_parity_difference'", actual_value None;
            #   * metric='demogrpahic_parity_difference', A TYPO IN THE METRIC
            #     NAME, on a healthy two-group frame.
            # The typo case is the sharper one: it is a green fairness gate over
            # a metric that does not exist. ``assert_fairness`` has refused both
            # since 2026-09-07 and says so in its own docstring; this decorator
            # is the same gate with the same inputs and had none of it.
            _refuse_unless_measured(suite.results, metric, threshold)

            return suite.results

        # Mark for pytest discovery
        setattr(wrapper, "_fairness_test", True)
        setattr(wrapper, "_fairness_metric", metric)
        setattr(wrapper, "_fairness_threshold", threshold)
        setattr(wrapper, "_fairness_protected_attribute", protected_attribute)

        return wrapper

    return decorator


def parametrize_fairness(
    protected_attributes: List[str],
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
):
    """Decorator to parametrize a test across multiple fairness scenarios.

    This creates multiple test cases, one for each combination of
    protected attribute and metric.

    Args:
        protected_attributes: List of protected attribute column names. At least
            one is required: an empty list produces no test case at all, and a
            run that examined nothing fails closed rather than returning an
            empty, green result.
        metrics: List of metrics to test (default: demographic_parity_difference).
        thresholds: The bound each metric is tested against, in that metric's
            own direction. EVERY metric listed in ``metrics`` needs an entry:
            a metric with no threshold is could-not-check and is recorded as a
            FAILED case without calling the test body, because there is no
            bound to test it against. It is never given an invented one.

    Example:
        >>> @parametrize_fairness(
        ...     protected_attributes=['gender', 'race'],
        ...     metrics=['demographic_parity_difference', 'equalized_odds_difference'],
        ...     thresholds={'demographic_parity_difference': 0.1, 'equalized_odds_difference': 0.1}
        ... )
        ... def test_model_fairness(model, test_data, protected_attribute, metric, threshold):
        ...     # Test implementation
        ...     pass
    """
    metrics = metrics or ["demographic_parity_difference"]
    thresholds = thresholds or {"demographic_parity_difference": 0.1}

    def decorator(func: Callable):
        # Store parametrization info for pytest collection
        setattr(
            func,
            "_fairness_parametrize",
            {
                "protected_attributes": protected_attributes,
                "metrics": metrics,
                "thresholds": thresholds,
            },
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # FAIL CLOSED ON AN EMPTY PARAMETRIZATION. The two loops below
            # produce one case per (attribute, metric) pair, so an empty
            # ``protected_attributes`` runs the test body ZERO times, collects no
            # failure and returns ``[]``. Measured 2026-09-27 (BGL3
            # operations-2) with protected_attributes=[] and a real metric and
            # threshold configured: the body was called 0 times, the wrapper
            # returned [], nothing was raised, and the pytest test went GREEN. An
            # empty result list makes "no fairness problem was found" and
            # "nothing was examined" the same output, which is the collapse this
            # library refuses everywhere else. A parametrization that names no
            # protected attribute is a misconfiguration, and a misconfigured gate
            # must not report a pass it did not earn.
            if not protected_attributes or not metrics:
                raise FairnessAssertionError(
                    f"NOT MEASURABLE: this parametrization covers "
                    f"{len(protected_attributes)} protected attribute(s) and "
                    f"{len(metrics)} metric(s), so the test body was never run and "
                    f"nothing was compared against any threshold. An empty result is "
                    f"not a pass. Name at least one protected attribute and one metric.",
                    metric_name=metrics[0] if metrics else "none configured",
                    actual_value=float("nan"),
                    threshold=float("nan"),
                )
            results = []
            for attr in protected_attributes:
                for metric in metrics:
                    # CRITICAL (fail closed): this was `thresholds.get(metric,
                    # 0.1)`, which handed the test body a bound nobody
                    # configured. 0.1 is a cap-shaped number, so for the whole
                    # ratio family it replaced the four-fifths FLOOR of 0.80
                    # with a CAP of 0.10. Measured 2026-09-10 with
                    # metrics=['disparate_impact_ratio'] and thresholds={}: the
                    # decorator passed threshold=0.1 to the test, a
                    # disparate-impact ratio of 0.11 satisfied `value >=
                    # threshold`, and the run was recorded passed=True. The
                    # honest floor would have failed it by a factor of seven.
                    #
                    # A metric with no configured threshold was never compared
                    # to anything: could-not-check, which FAILS here as it does
                    # on every other gate in this library. `is None`, never
                    # falsiness, so a deliberate 0.0 is still handed through and
                    # still enforced by the test body.
                    threshold = thresholds.get(metric)
                    if threshold is None:
                        results.append(
                            {
                                "protected_attribute": attr,
                                "metric": metric,
                                "threshold": None,
                                "error": (
                                    f"no threshold is configured for '{metric}', so nothing "
                                    f"was compared; the test fails closed rather than running "
                                    f"against an invented bound. Add it to `thresholds`, or "
                                    f"remove it from `metrics`."
                                ),
                                "passed": False,
                            }
                        )
                        continue
                    try:
                        result = func(
                            *args,
                            protected_attribute=attr,
                            metric=metric,
                            threshold=threshold,
                            **kwargs,
                        )
                        results.append(
                            {
                                "protected_attribute": attr,
                                "metric": metric,
                                "threshold": threshold,
                                "result": result,
                                "passed": True,
                            }
                        )
                    except FairnessAssertionError as e:
                        results.append(
                            {
                                "protected_attribute": attr,
                                "metric": metric,
                                "threshold": threshold,
                                "error": str(e),
                                "passed": False,
                            }
                        )

            # Raise if any failed
            failed = [r for r in results if not r["passed"]]
            if failed:
                # NaN, not 0, and NaN for an absent threshold: this is an
                # aggregate over `len(failed)` failures and it has no single
                # measured value of its own. "Actual: 0.0000" printed a
                # fabricated measurement into the failure header (the same
                # convention the SKIPPED path at the bottom of this module
                # already uses), and a None threshold from the fail-closed
                # branch above would crash the `:.4f` formatting outright. Every
                # per-test value stays in details['failed_tests'].
                first_threshold = failed[0]["threshold"]
                raise FairnessAssertionError(
                    f"{len(failed)} fairness test(s) failed",
                    metric_name=failed[0]["metric"],
                    actual_value=float("nan"),
                    threshold=(float("nan") if first_threshold is None else first_threshold),
                    details={"failed_tests": failed},
                )

            return results

        return wrapper

    return decorator


def assert_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    metric: str = "demographic_parity_difference",
    threshold: float = 0.1,
    message: Optional[str] = None,
) -> None:
    """Assert that predictions meet fairness requirements.

    This is a simple assertion function for use in tests.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        metric: Fairness metric to check.
        threshold: Maximum acceptable value.
        message: Optional custom error message.

    Raises:
        FairnessAssertionError: If the fairness requirement is not met, OR if
            it could not be checked at all. Three states, never two: a metric
            measured and over its threshold FAILS, and a metric that could not
            be measured also fails, loudly and with a different message. It
            never passes.

            This is a release gate, so could-not-check must never land on the
            green side. Until 2026-09-07 it raised only on TestStatus.FAILED
            and let every other status through. Both cases that reach here are
            SKIPPED with a value of None: a metric undefined on the data
            (equal_opportunity_difference where no group has a positive label),
            and A METRIC NAME THAT DOES NOT EXIST. A typo in the metric name
            made the gate pass. So did an empty result list.

    Example:
        >>> def test_model_fairness():
        ...     y_pred = model.predict(X_test)
        ...     assert_fairness(
        ...         y_test, y_pred, test_data['gender'],
        ...         metric='demographic_parity_difference',
        ...         threshold=0.1
        ...     )
    """
    suite = FairnessTestSuite(
        protected_attributes=["attr"],
        metrics=[metric],
        thresholds={metric: threshold},
    )

    results = suite.test_predictions(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        protected_attr=np.asarray(protected_attr),
        attr_name="protected_attribute",
        raise_on_failure=False,
    )

    # FAIL CLOSED ON AN EMPTY RESULT, and on any result that is not a real
    # comparison. No result is not a pass: it means the suite produced nothing to
    # judge, and returning quietly would let a gate approve a model nobody
    # checked. The rule and its wording live in ``_refuse_unless_measured`` so
    # the ``fairness_test`` decorator, which is the same gate reached another
    # way, cannot drift away from them again.
    _refuse_unless_measured(results, metric, threshold, message)

    for result in results:
        if result.status == TestStatus.FAILED:
            error_message = message or f"Fairness assertion failed: {result.message}"
            # A FAILED result is always constructed with concrete
            # actual_value and threshold floats (see test_predictions), so
            # neither can be None here.
            assert result.actual_value is not None
            assert result.threshold is not None
            raise FairnessAssertionError(
                error_message,
                metric_name=result.metric_name,
                actual_value=result.actual_value,
                threshold=result.threshold,
            )
