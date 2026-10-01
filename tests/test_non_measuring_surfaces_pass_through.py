"""The five non-measuring open defects, examined one at a time. None reproduces.

These are the units the release gate counts under B3: open defects in code that
renders, converts or prints rather than returning a fairness number. B3 asks for a
severity on each, and assigning one by guesswork would be the defect this library
exists to remove, so each was executed on the input its defect describes instead.
All five turned out to be honest, and four of the five were already honest in ways
worth pinning because a renderer IS able to put a fabricated verdict in front of a
reader: `BiasAuditReport.to_svg`, fixed earlier in this same wave, rendered
"OVERALL RISK 0% MINIMAL" over an audit that assessed nothing.

What was measured, 2026-09-25:

  build_drift_event          propagates a NaN drift score as NaN and warns. It does
                             not invent a drift verdict. Recorded as "fabrication
                             open"; that record was stale.
  OptimizationResult.to_dict passes NaN through faithfully.
  build_recheck              three-state and exemplary: an undetermined LL144 screen
                             SAYS the statute test did not happen rather than
                             claiming no statute applies.
  to_prometheus_metrics      on a fresh monitor emits counters only, all genuinely
                             zero, and no fairness metric values.
  ThresholdResult.summary    prints "Global Threshold: nan" and "Feasible: False".

So nothing is changed here. What was missing is the evidence, and a renderer with
no pin is one refactor away from the to_svg defect.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness.in_processing.constraints.base import OptimizationResult
from vfairness.operations.cicd.monitor import BiasMonitor
from vfairness.operations.monitoring.alerts import FairnessAlertPrioritizer
from vfairness.operations.pulse.regulatory import build_recheck
from vfairness.post_processing.threshold_optimization.optimizer import ThresholdResult

NAN = float("nan")


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        return fn(*a, **k), [str(w.message) for w in rec]


def test_a_drift_event_built_from_an_unmeasured_score_carries_the_nan():
    event, _msgs = _caught(
        FairnessAlertPrioritizer.build_drift_event, "demographic_parity", ["a", "b"], NAN, NAN
    )
    assert math.isnan(event["drift_score"]), (
        f"a drift event was built with drift_score={event['drift_score']!r} from an "
        "unmeasured score"
    )
    assert math.isnan(event["mean_shift"])


def test_control_a_drift_event_from_real_numbers_keeps_them():
    event, _msgs = _caught(
        FairnessAlertPrioritizer.build_drift_event, "demographic_parity", ["a", "b"], 0.42, 0.15
    )
    assert event["drift_score"] == pytest.approx(0.42)
    assert event["mean_shift"] == pytest.approx(0.15)
    assert event["affected_groups"] == ["a", "b"]


def test_an_optimisation_result_does_not_launder_a_nan_into_a_number():
    result = OptimizationResult(converged=False, n_iterations=0, final_violation=NAN, best_gap=NAN)
    as_dict = result.to_dict()
    assert math.isnan(as_dict["final_violation"]) and math.isnan(as_dict["best_gap"])
    assert as_dict["converged"] is False


def test_control_an_optimisation_result_reports_what_it_was_given():
    as_dict = OptimizationResult(
        converged=True, n_iterations=17, final_violation=0.004, best_gap=0.012
    ).to_dict()
    assert as_dict["n_iterations"] == 17
    assert as_dict["final_violation"] == pytest.approx(0.004)


def test_an_undetermined_statute_screen_is_not_reported_as_no_statute():
    """The three-state property, and the one with legal consequences."""
    out, _msgs = _caught(build_recheck, {}, None)
    basis = out["basis"]
    assert "could not be" in basis and "NOT a finding that none applies" in basis
    assert "No annual audit statute matched" not in basis


@pytest.mark.parametrize(
    "applicable,expect",
    [(True, "requires a bias audit"), (False, "No annual audit statute matched")],
)
def test_control_a_determined_statute_screen_states_its_finding(applicable, expect):
    out, _msgs = _caught(build_recheck, {}, applicable)
    assert expect in out["basis"]
    assert "NOT a finding" not in out["basis"]


def test_a_monitor_with_no_batches_exports_counters_not_fairness_values():
    text, _msgs = _caught(BiasMonitor().to_prometheus_metrics)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines, "the exporter produced nothing at all"
    for line in lines:
        name = line.split()[0]
        assert name.endswith(("_total", "_metrics")), (
            f"a fresh monitor exported {name!r}, which is not a counter: a scraped 0 "
            "for a fairness metric nobody measured feeds dashboards and alerts"
        )


def test_an_infeasible_threshold_summary_shows_the_nan_rather_than_a_number():
    result = ThresholdResult(
        global_threshold=NAN,
        group_thresholds={},
        constraint_violations={},
        performance_metrics={},
        is_feasible=False,
        optimization_details={},
    )
    text = result.summary()
    assert "nan" in text.lower(), "an unset global threshold was summarised as a number"
    assert "False" in text


def test_control_a_feasible_threshold_summary_shows_its_threshold():
    result = ThresholdResult(
        global_threshold=0.63,
        group_thresholds={"a": 0.6, "b": 0.66},
        constraint_violations={},
        performance_metrics={"accuracy": 0.81},
        is_feasible=True,
        optimization_details={},
    )
    text = result.summary()
    assert "0.63" in text and "True" in text
    assert "nan" not in text.lower()
