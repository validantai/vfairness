"""Beta Go-Live stage 1, group g06 (CI/CD): three states, never two.

Two capabilities, one defect each, both reproduced at the PUBLIC entry point
before anything was changed:

``bias_monitor`` (``BiasMonitor.log_batch``)
    A batch in which the monitored metric was never computed was reported as a
    clean batch. Measured 2026-09-11 with a single-group second batch:
    ``MonitoringResult(b2: OK, 0 alerts)``, ``could_not_check []``,
    ``drift_detected False``, ``get_drift_status()
    {'demographic_parity_difference': False}`` -- which that method's own
    docstring defines as "measured against the baseline and within the
    threshold" -- and ``to_prometheus_metrics()`` exporting
    ``fairness_demographic_parity_difference 0.8999999999999999``, the PREVIOUS
    batch's value re-served as the current one, beside
    ``fairness_drift_demographic_parity_difference 0`` and
    ``fairness_monitor_unmeasurable_metrics 0``.

``data_bias_validation`` (``DataBiasValidator.validate``)
    An attribute with one level was a PASS. Measured 2026-09-11 on 200 rows
    with a single gender level: ``passed=True``, ``checks_run`` including
    ``'outcome_disparity'``, and ``metrics['outcome_disparity']['gender']``
    holding a one-entry ``outcome_rates`` with NO ``outcome_ratio`` key and no
    issue of any kind. The same branch inverted on unreadable outcomes: two
    groups whose rates are both NaN fail ``min_rate > 0`` exactly as 0.0 does,
    so an unmeasurable attribute was written up as a CRITICAL "zero positive
    outcome rate" finding whose ``zero_rate_groups`` list was EMPTY.

Every assertion below is made on what a caller gets back, never on a private
helper, and each defect is paired with a control proving healthy data is still
measured.
"""

import math

import numpy as np
import pandas as pd
import pytest

from vfairness import BiasMonitor
from vfairness.operations.cicd import DataBiasValidator
from vfairness.operations.cicd.monitor import DriftType, MonitorConfig
from vfairness.operations.cicd.validator import DataValidationConfig, ValidationSeverity

# ---------------------------------------------------------------------------
# bias_monitor: BiasMonitor.log_batch
# ---------------------------------------------------------------------------

BASELINE = {"demographic_parity_difference": 0.9}


def _two_group_batch():
    """A 0.90 demographic-parity gap across two groups: fully measurable."""
    protected = np.array(["A"] * 100 + ["B"] * 100)
    y_pred = np.concatenate([np.array([1] * 95 + [0] * 5), np.array([1] * 5 + [0] * 95)])
    y_true = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] * 50 + [0] * 50)])
    return y_pred, y_true, protected


def _single_group_batch():
    """The same rows, one group. No gap EXISTS to be measured, so
    ``_compute_default_metrics`` returns {} and nothing is computed."""
    y_pred, y_true, _ = _two_group_batch()
    return y_pred, y_true, np.array(["A"] * 200)


def _monitor(baseline=BASELINE):
    return BiasMonitor(baseline_metrics=baseline)


class TestMonitorUnmeasuredBatchIsNotAnOkBatch:
    def test_a_batch_that_measured_nothing_is_could_not_check(self):
        monitor = _monitor()
        first = monitor.log_batch(*_two_group_batch(), batch_id="b1")
        assert first.could_not_check == [], "the measurable batch must stay measured"

        result = monitor.log_batch(*_single_group_batch(), batch_id="b2_single_group")

        # Before: could_not_check [], drift_detected False, repr "b2: OK".
        assert result.could_not_check == ["demographic_parity_difference"]
        assert result.drift_detected is True
        assert "COULD NOT CHECK" in repr(result)
        assert result.to_dict()["could_not_check"] == ["demographic_parity_difference"]

        # Three states, never two, on the monitor-level surface.
        assert monitor.get_drift_status() == {"demographic_parity_difference": None}
        assert monitor.unmeasurable_metrics() == ["demographic_parity_difference"]
        assert monitor.drift_detected() is True
        assert monitor.get_summary()["drift_detected"] is True

    def test_the_unmeasured_batch_does_not_re_serve_the_last_measured_value(self):
        """The stale half of the same defect: `get_current_metrics()` answered
        0.8999999999999999 for a batch in which the gap was never computed, and
        the Prometheus gauge exported that number as the current one."""
        monitor = _monitor()
        monitor.log_batch(*_two_group_batch(), batch_id="b1")
        monitor.log_batch(*_single_group_batch(), batch_id="b2_single_group")

        current = monitor.get_current_metrics()["demographic_parity_difference"]
        assert current is not None
        assert math.isnan(current), f"the previous batch's value was re-served: {current!r}"

        exported = monitor.to_prometheus_metrics().splitlines()
        assert "fairness_demographic_parity_difference NaN" in exported
        assert "fairness_demographic_parity_difference 0.8999999999999999" not in exported
        assert "fairness_drift_demographic_parity_difference NaN" in exported
        assert "fairness_drift_demographic_parity_difference 0" not in exported
        assert "fairness_monitor_unmeasurable_metrics 1" in exported

    def test_the_documented_alert_pattern_does_not_resolve_into_silence(self):
        """`if monitor.drift_detected(): monitor.trigger_alert()` is the class
        docstring's own example."""
        monitor = _monitor()
        monitor.log_batch(*_two_group_batch(), batch_id="b1")
        monitor.log_batch(*_single_group_batch(), batch_id="b2_single_group")

        assert monitor.drift_detected()
        alert = monitor.trigger_alert()
        assert alert is not None
        assert alert.drift_type is DriftType.NOT_MEASURABLE

    def test_an_empty_batch_is_could_not_check_too(self):
        """A zero-row batch behaved identically to the single-group one."""
        monitor = _monitor()
        monitor.log_batch(*_two_group_batch(), batch_id="b1")
        result = monitor.log_batch(
            y_pred=np.array([]), y_true=np.array([]), protected_attr=np.array([]), batch_id="empty"
        )
        assert result.could_not_check == ["demographic_parity_difference"]
        assert monitor.get_drift_status()["demographic_parity_difference"] is None

    def test_a_monitored_metric_with_no_baseline_is_could_not_check(self):
        """The sibling door. Nothing is compared, so `get_drift_status()` stayed
        EMPTY and `drift_detected()` answered False over `any([])` -- the same
        all-clear, reached with an absent entry instead of a False one. A 0.90
        gap read "OK, 0 alerts"."""
        monitor = BiasMonitor()  # no baseline at all
        result = monitor.log_batch(*_two_group_batch(), batch_id="no_baseline")

        assert result.could_not_check == ["demographic_parity_difference"]
        assert monitor.get_drift_status() == {"demographic_parity_difference": None}
        assert monitor.drift_detected() is True

    def test_a_custom_metrics_function_that_omits_the_metric_is_could_not_check(self):
        """The guard sits ABOVE the dispatch, so a caller-supplied compute
        function that omits a monitored metric cannot slip through either."""
        monitor = BiasMonitor(
            baseline_metrics=BASELINE,
            compute_metrics_fn=lambda y_true, y_pred, attr: {"something_else": 0.1},
            config=MonitorConfig(min_samples_for_alert=1),
        )
        result = monitor.log_batch(*_two_group_batch(), batch_id="omitted")
        assert result.could_not_check == ["demographic_parity_difference"]
        assert monitor.get_drift_status()["demographic_parity_difference"] is None

    # ---------------- CONTROLS: healthy data is still measured --------------

    def test_control_a_measured_stable_batch_is_still_green(self):
        monitor = _monitor()
        monitor.log_batch(*_two_group_batch(), batch_id="c1")
        result = monitor.log_batch(*_two_group_batch(), batch_id="c2")

        assert result.could_not_check == []
        assert result.drift_detected is False
        assert "OK" in repr(result)
        assert result.metrics["demographic_parity_difference"] == pytest.approx(0.9)
        assert monitor.get_drift_status() == {"demographic_parity_difference": False}
        assert monitor.unmeasurable_metrics() == []
        assert monitor.drift_detected() is False
        assert monitor.trigger_alert() is None

        exported = monitor.to_prometheus_metrics().splitlines()
        assert "fairness_drift_demographic_parity_difference 0" in exported
        assert "fairness_monitor_unmeasurable_metrics 0" in exported
        assert any(
            line.startswith("fairness_demographic_parity_difference 0.89") for line in exported
        ), exported

    def test_control_a_real_measured_drift_is_still_reported_as_drift(self):
        """Could-not-check must not swallow the case the monitor exists for."""
        monitor = BiasMonitor(baseline_metrics={"demographic_parity_difference": 0.0})
        monitor.log_batch(*_two_group_batch(), batch_id="drifted")

        assert monitor.unmeasurable_metrics() == []
        assert monitor.get_drift_status()["demographic_parity_difference"] is True
        alert = monitor.trigger_alert()
        assert alert is not None
        assert alert.drift_type is DriftType.METRIC_DRIFT
        assert alert.drift_magnitude == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# data_bias_validation: DataBiasValidator.validate
# ---------------------------------------------------------------------------


def _single_level_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"gender": ["F"] * n, "score": rng.normal(size=n), "hired": rng.integers(0, 2, n)}
    )


def _unreadable_outcome_frame(n: int = 200) -> pd.DataFrame:
    """Two groups, and an outcome column that is not binary, so every group
    rate is NaN: no rate is DEFINED, which is not a rate of zero."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "gender": ["F"] * (n // 2) + ["M"] * (n // 2),
            "score": rng.normal(size=n),
            "hired": ["maybe", "perhaps", "dunno", "later"] * (n // 4),
        }
    )


def _clean_frame(n: int = 600) -> pd.DataFrame:
    """Healthy: two well-sized groups, no duplicates, no constant column, a
    readable binary outcome whose rates are close enough to pass."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "score": rng.normal(0, 1, n),
            "age": rng.integers(20, 70, n),
            "approved": rng.integers(0, 2, n),
        }
    )


def _disparate_frame(n: int = 600) -> pd.DataFrame:
    """Healthy enough to measure, and genuinely unfair: 10% vs 90%."""
    rng = np.random.default_rng(3)
    half = n // 2
    return pd.DataFrame(
        {
            "gender": ["F"] * half + ["M"] * half,
            "score": rng.normal(0, 1, n),
            "age": rng.integers(20, 70, n),
            "approved": [1] * (half // 10)
            + [0] * (half - half // 10)
            + [1] * (half - half // 10)
            + [0] * (half // 10),
        }
    )


def _zero_rate_frame(n: int = 600) -> pd.DataFrame:
    """One group with a genuinely MEASURED zero positive rate."""
    rng = np.random.default_rng(5)
    half = n // 2
    return pd.DataFrame(
        {
            "gender": ["F"] * half + ["M"] * half,
            "score": rng.normal(0, 1, n),
            "age": rng.integers(20, 70, n),
            "approved": [0] * half + ([1] * (half // 2) + [0] * (half - half // 2)),
        }
    )


def _issue_types(result):
    return [i.issue_type for i in result.issues]


def _disparity(result, attr="gender"):
    return result.metrics["outcome_disparity"][attr]


class TestValidatorOutcomeDisparityNotMeasurable:
    def test_a_single_level_attribute_is_not_a_pass(self):
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _single_level_frame(), outcome_column="hired"
        )

        # Before: passed True, no disparity issue, no outcome_ratio key at all.
        assert result.passed is False
        assert "disparity_not_measurable" in _issue_types(result)
        issue = next(i for i in result.issues if i.issue_type == "disparity_not_measurable")
        assert issue.severity is ValidationSeverity.ERROR
        assert "could not be measured" in issue.message

        # The machine-readable half: a recorded third state, not a missing key.
        assert _disparity(result)["outcome_ratio"] is None
        assert _disparity(result)["disparity_measured"] is False

    def test_the_refusal_does_not_depend_on_an_incidental_hygiene_warning(self):
        """The decisive variant. With data hygiene off, the only accidental hint
        (a WARNING naming gender as a constant column) disappears, and before
        the fix the run was passed=True with ZERO issues."""
        config = DataValidationConfig(check_data_hygiene=False)
        result = DataBiasValidator(protected_attributes=["gender"], config=config).validate(
            _single_level_frame(), outcome_column="hired"
        )
        assert "constant_columns" not in _issue_types(result)
        assert result.passed is False
        assert "disparity_not_measurable" in _issue_types(result)
        assert "no bias issues detected" not in result.summary

    def test_undefined_rates_are_not_written_up_as_a_measured_zero(self):
        """The mirror direction: a fabricated measurement claim in the ALARMING
        direction. Both group rates NaN gave a CRITICAL 'zero positive outcome
        rate' with an EMPTY zero_rate_groups list and outcome_ratio inf."""
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _unreadable_outcome_frame(), outcome_column="hired"
        )

        assert "zero_outcome_rate" not in _issue_types(result)
        assert "disparity_not_measurable" in _issue_types(result)
        assert _disparity(result)["outcome_ratio"] is None
        assert _disparity(result)["disparity_measured"] is False
        assert _disparity(result)["groups_without_defined_rate"] == ["F", "M"]
        assert result.passed is False

    # ---------------- CONTROLS: healthy data is still measured --------------

    def test_control_a_clean_two_group_dataset_still_passes(self):
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _clean_frame(), outcome_column="approved"
        )
        assert result.passed is True
        assert result.issues == []
        assert result.execution_coverage() == "complete"
        assert _disparity(result)["disparity_measured"] is True
        assert _disparity(result)["outcome_ratio"] == pytest.approx(1.0, abs=0.3)

    def test_control_a_real_disparity_is_still_measured_and_reported(self):
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _disparate_frame(), outcome_column="approved"
        )
        assert result.passed is False
        assert "outcome_disparity" in _issue_types(result)
        assert "disparity_not_measurable" not in _issue_types(result)
        assert _disparity(result)["disparity_measured"] is True
        assert _disparity(result)["outcome_ratio"] == pytest.approx(9.0)

    def test_control_a_measured_zero_rate_is_still_a_critical_finding(self):
        """Separating "no defined rate" from "a rate of zero" must not lose the
        real zero-rate alarm, which names the group it measured."""
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _zero_rate_frame(), outcome_column="approved"
        )
        issue = next(i for i in result.issues if i.issue_type == "zero_outcome_rate")
        assert issue.severity is ValidationSeverity.CRITICAL
        assert issue.details["zero_rate_groups"] == ["F"]
        assert issue.affected_groups == ["F"]
        assert _disparity(result)["disparity_measured"] is True
        assert math.isinf(_disparity(result)["outcome_ratio"])
        assert result.passed is False


# ---------------------------------------------------------------------------
# Beta Go-Live Stage 1 audit, 2026-09-11.
#
# The fix above introduced a regression of the worst kind for this campaign: by
# routing "fewer than two defined rates" into a could-not-check `continue`, it
# SWALLOWED the CRITICAL zero_outcome_rate finding that the pre-fix code
# correctly raised when exactly one group had a defined rate and that rate was a
# genuinely measured 0.0. Removing a fabricated disparity took a real finding
# with it. "No member of this group received a single positive outcome" is a fact
# about the rows that WERE read; it does not need a comparison group.
# ---------------------------------------------------------------------------

import numpy as _np
import pandas as _pd

from vfairness.operations.cicd.validator import DataBiasValidator as _Validator


def _issue_types(result) -> set:
    return {i.issue_type for i in result.issues}


def _one_defined_zero_rate() -> "_pd.DataFrame":
    """'roma' has a MEASURED zero rate; 'other' has no readable outcome at all."""
    return _pd.DataFrame(
        {
            "gender": ["roma"] * 60 + ["other"] * 60,
            "approved": [0] * 60 + [_np.nan] * 60,
        }
    )


def test_a_measured_zero_rate_survives_the_could_not_check_path() -> None:
    result = _Validator(protected_attributes=["gender"]).validate(
        _one_defined_zero_rate(), outcome_column="approved"
    )
    types = _issue_types(result)
    assert "zero_outcome_rate" in types, (
        "the CRITICAL measured-zero finding was swallowed by the not-measurable branch"
    )
    zero = next(i for i in result.issues if i.issue_type == "zero_outcome_rate")
    assert zero.severity.value == "critical"
    assert zero.affected_groups == ["roma"], "the finding must name the group it measured"


def test_both_states_are_reported_side_by_side() -> None:
    """The two facts are independent and BOTH must reach the reader: the disparity
    could not be measured, AND a group that was measured had no positive outcome.
    Reporting only one of them is the two-state failure in either direction."""
    result = _Validator(protected_attributes=["gender"]).validate(
        _one_defined_zero_rate(), outcome_column="approved"
    )
    types = _issue_types(result)
    assert {"disparity_not_measurable", "zero_outcome_rate"} <= types
    assert result.passed is False


def test_control_a_nonzero_single_rate_invents_no_zero_finding() -> None:
    """Over-correction control. One defined rate that is NOT zero must raise the
    could-not-check and nothing else."""
    df = _pd.DataFrame(
        {
            "gender": ["roma"] * 60 + ["other"] * 60,
            "approved": [1] * 30 + [0] * 30 + [_np.nan] * 60,
        }
    )
    types = _issue_types(
        _Validator(protected_attributes=["gender"]).validate(df, outcome_column="approved")
    )
    assert "disparity_not_measurable" in types
    assert "zero_outcome_rate" not in types


def test_control_the_two_group_path_still_raises_it() -> None:
    """The branch that always worked must keep working."""
    df = _pd.DataFrame(
        {
            "gender": ["roma"] * 60 + ["other"] * 60,
            "approved": [0] * 60 + [1] * 30 + [0] * 30,
        }
    )
    result = _Validator(protected_attributes=["gender"]).validate(df, outcome_column="approved")
    assert "zero_outcome_rate" in _issue_types(result)
    assert "disparity_not_measurable" not in _issue_types(result)
