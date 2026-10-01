"""READINESS-6: seven guards refused NaN and let INFINITY through.

Every one of them carried a comment explaining, at length and correctly, why a
value nobody measured must never become a verdict. Each then tested for NaN
alone. NaN and infinity are opposite halves of one hole: NaN LOSES every
comparison, so it slipped through as a silent non-breach; infinity WINS them,
so on a higher-is-better metric it arrived as an outright PASS.

The worst of it, measured on this repo before the fix::

    ModelFairnessGate(metrics=["disparate_impact_ratio"],
                      thresholds={"disparate_impact_ratio": 0.8})
      .evaluate_from_metrics({"disparate_impact_ratio": float("inf")})

    status=APPROVED  approved=True  passed=True  message=''
    github conclusion='success'

Byte-identical to the genuine pass at 0.95, while the NaN beside it failed
closed and explained itself. ``disparate_impact_ratio`` is ``inf`` exactly when
the denominator group has zero selections, i.e. a group that is NEVER SELECTED
AT ALL. The four-fifths rule shipped that green.

This file pins all seven, and pins the over-correction controls beside them: a
real measurement must still grade, and each predicate that deliberately keeps a
different contract must keep it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)
from vfairness.operations.cicd.gate import GateStatus, ModelFairnessGate
from vfairness.operations.cicd.monitor import _is_finite
from vfairness.operations.cicd.testing import FairnessTestSuite
from vfairness.operations.cicd.testing import TestStatus as _TestStatus
from vfairness.operations.reporting.compliance import (
    _as_float,
    _as_measured_number,
    _as_severity,
)
from vfairness.rendering.adapters_monitoring import _clamp01, _measured

INF = float("inf")
NEG_INF = float("-inf")
NAN = float("nan")

# Every way a non-measurement reaches a grader, and what it must never become.
NOT_MEASUREMENTS = [
    ("positive infinity", INF),
    ("negative infinity", NEG_INF),
    ("nan", NAN),
    ("numpy infinity", np.float64("inf")),
    ("numpy nan", np.float64("nan")),
]


# ---------------------------------------------------------------- check_threshold


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_an_ungradeable_value_is_could_not_check_not_pass(label, value):
    """The higher-is-better direction is where infinity did its damage: it does
    not merely fail to breach the bound, it SATISFIES it."""
    outcome, message = check_threshold("disparate_impact_ratio", value, 0.8)
    assert outcome is ThresholdOutcome.COULD_NOT_CHECK, f"{label}: {outcome} {message}"
    assert message, "a could-not-check must say why; an empty message reads as a pass"


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_the_lower_is_better_direction_too(label, value):
    outcome, message = check_threshold("demographic_parity_difference", value, 0.1)
    assert outcome is ThresholdOutcome.COULD_NOT_CHECK, f"{label}: {outcome} {message}"


@pytest.mark.parametrize("label,threshold", NOT_MEASUREMENTS)
def test_an_ungradeable_bound_grades_nothing(label, threshold):
    """A bound of ``inf`` on a higher-is-better metric cannot be SATISFIED, the
    mirror of the degenerate bound this function already guarded. Before the
    fix a PERFECT ratio of 1.0 was reported as a FAIL against it."""
    outcome, message = check_threshold("disparate_impact_ratio", 1.0, threshold)
    assert outcome is ThresholdOutcome.COULD_NOT_CHECK, f"{label}: {outcome} {message}"


def test_real_values_still_grade():
    """OVER-CORRECTION CONTROL. The fix must not turn measurements into
    could-not-checks; that is the other way to lose a finding."""
    assert check_threshold("disparate_impact_ratio", 0.95, 0.8)[0] is ThresholdOutcome.PASS
    assert check_threshold("disparate_impact_ratio", 0.55, 0.8)[0] is ThresholdOutcome.FAIL
    assert check_threshold("demographic_parity_difference", 0.05, 0.1)[0] is ThresholdOutcome.PASS
    assert check_threshold("demographic_parity_difference", 0.50, 0.1)[0] is ThresholdOutcome.FAIL
    # numpy scalars are what the metric functions actually return.
    assert (
        check_threshold("disparate_impact_ratio", np.float64(0.95), 0.8)[0] is ThresholdOutcome.PASS
    )
    assert (
        check_threshold("disparate_impact_ratio", np.float32(0.55), 0.8)[0] is ThresholdOutcome.FAIL
    )


# ---------------------------------------------------------------- the gate


def _gate():
    return ModelFairnessGate(
        metrics=["disparate_impact_ratio"],
        thresholds={"disparate_impact_ratio": 0.8},  # the four-fifths rule
    )


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_the_gate_never_approves_a_metric_nobody_measured(label, value):
    decision = _gate().evaluate_from_metrics({"disparate_impact_ratio": value})
    assert decision.approved is False, f"{label} was APPROVED"
    assert decision.status is not GateStatus.APPROVED, f"{label}: {decision.status}"
    evaluation = decision.metric_evaluations[0]
    assert evaluation.passed is False
    assert evaluation.message, "an approved-looking empty message on an unmeasured metric"
    assert "could not be measured" in evaluation.message


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_the_github_check_says_failure_not_success(label, value):
    """The surface a reviewer actually sees on the pull request."""
    gate = _gate()
    decision = gate.evaluate_from_metrics({"disparate_impact_ratio": value})
    check = gate.create_github_check(decision)
    assert check["conclusion"] != "success", f"{label} wrote a green check"


def test_the_gate_still_approves_a_real_pass_and_blocks_a_real_breach():
    """OVER-CORRECTION CONTROL. A gate that blocks everything is not a fix."""
    gate = _gate()
    good = gate.evaluate_from_metrics({"disparate_impact_ratio": 0.95})
    assert good.approved is True and good.status is GateStatus.APPROVED
    assert gate.create_github_check(good)["conclusion"] == "success"

    bad = gate.evaluate_from_metrics({"disparate_impact_ratio": 0.55})
    assert bad.approved is False
    assert "below the required minimum" in bad.metric_evaluations[0].message


def test_the_other_gate_path_computes_and_refuses_the_same_way():
    """`evaluate()` carried its OWN copy of the NaN-only guard, so driving
    `evaluate_from_metrics()` alone would leave half the defect pinned."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))

    gate = ModelFairnessGate(
        metrics=["disparate_impact_ratio"],
        thresholds={"disparate_impact_ratio": 0.8},
        compute_metrics_fn=lambda *a, **k: {"disparate_impact_ratio": INF},
    )
    decision = gate.evaluate(y_true, y_pred, groups)
    assert decision.approved is False, "evaluate() approved an infinite metric"
    message = decision.metric_evaluations[0].message
    assert "could not be measured" in message
    # "fails closed" comes only from evaluate()'s OWN pre-check. Without this
    # clause the assertion above is satisfied by check_threshold downstream, and
    # reverting this path's guard is a GREEN sabotage: see the note on
    # test_the_gate_says_it_failed_closed_not_just_that_it_could_not_measure.
    assert "fails closed" in message, "evaluate()'s own pre-check no longer runs"


# ---------------------------------------------------------------- the test suite


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_a_fairness_test_over_an_unmeasurable_metric_is_skipped(label, value):
    """Not PASSED, and not FAILED either: three states, never two."""
    n = 100
    suite = FairnessTestSuite(
        protected_attributes=["group"],
        metrics=["disparate_impact_ratio"],
        thresholds={"disparate_impact_ratio": 0.8},
        compute_metrics_fn=lambda *a, **k: {"disparate_impact_ratio": value},
    )
    results = suite.test_predictions(
        y_true=np.zeros(n, dtype=int),
        y_pred=np.zeros(n, dtype=int),
        protected_attr=np.array(["a"] * 50 + ["b"] * 50),
        attr_name="group",
        raise_on_failure=False,
    )
    assert results, "no test ran, so this pins nothing"
    for r in results:
        assert r.status is _TestStatus.SKIPPED, f"{label}: status={r.status} msg={r.message}"
        assert "could not be measured" in (r.message or "")


# ---------------------------------------------------------------- compliance


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_an_unmeasurable_risk_score_is_not_graded_on_the_1_to_5_scale(label, value):
    """`_clamp` pins inf to the TOP of the scale and -inf to the BOTTOM, so the
    register read either "maximum severity" or "all clear" from a value nobody
    measured. Before the fix: _as_severity(inf) == 5, _as_severity(-inf) == 1."""
    assert _as_measured_number(value) is None, label
    assert _as_severity(value) is None, f"{label} graded to {_as_severity(value)!r}"


def test_a_numpy_bool_is_not_a_severity():
    """np.bool_ is not a Python bool, so it walked past the bool clause."""
    assert _as_float(np.bool_(True)) is None
    assert _as_severity(np.bool_(True)) is None


def test_real_severities_still_grade():
    """OVER-CORRECTION CONTROL."""
    assert _as_severity(4.2) == 4
    assert _as_severity(1) == 1
    assert _as_severity(np.float64(3.0)) == 3
    assert _as_float(np.float64(0.42)) == pytest.approx(0.42)


# ---------------------------------------------------------------- the dashboard


@pytest.mark.parametrize("label,value", NOT_MEASUREMENTS)
def test_an_unmeasurable_drift_score_draws_no_bar(label, value):
    """`_clamp01(inf)` is 1.0, a FULL-WIDTH bar: the most alarming geometry the
    dashboard can draw, from a number that does not exist. `_clamp01(-inf)` is
    0.0, which is the geometry this module chose to MEAN "not measured", so that
    half was a fabricated all-clear that looks identical to an honest blank."""
    assert _measured(value) is None, label
    assert _clamp01(_measured(value)) == 0.0, label


def test_a_numpy_bool_is_not_a_drift_score():
    assert _measured(np.bool_(True)) is None
    assert _measured(np.bool_(False)) is None


def test_the_dashboard_still_reads_real_values_including_serialised_ones():
    """OVER-CORRECTION CONTROL, and the deliberate contract difference.

    A numeric STRING is still accepted here on purpose: these rows arrive from
    JSON and CSV, where "0.5" is a real measurement that was serialised.
    `compliance._as_float` refuses strings for the opposite and equally correct
    reason, that its input is a caller's free-text field. Pinned so the
    divergence is a decision rather than an oversight.
    """
    assert _measured(0.42) == pytest.approx(0.42)
    assert _measured(np.float64(0.42)) == pytest.approx(0.42)
    assert _measured("0.5") == pytest.approx(0.5)
    assert _as_float("0.5") is None, "compliance keeps the opposite contract"


# ---------------------------------------------------------------- the monitor


def test_a_list_is_not_a_finite_number_at_any_length():
    """`bool(np.isfinite(value))` answered True for a ONE-element list and False
    for a two-element one, because numpy maps over sequences and the
    ambiguous-truth ValueError was caught and read as "not finite"."""
    assert _is_finite([1]) is False
    assert _is_finite([1, 2]) is False
    assert _is_finite(np.array([0.4])) is False
    assert _is_finite(np.array([0.4, 0.5])) is False


def test_the_monitor_refuses_flags_and_non_numbers():
    assert _is_finite(True) is False
    assert _is_finite(np.bool_(True)) is False
    assert _is_finite(None) is False
    assert _is_finite("0.5") is False
    assert _is_finite(INF) is False
    assert _is_finite(NAN) is False


def test_the_monitor_still_accepts_what_the_metrics_actually_return():
    """OVER-CORRECTION CONTROL. Metric functions return numpy scalars, and
    refusing those would silently drop every real measurement."""
    assert _is_finite(0.5) is True
    assert _is_finite(np.float64(0.4)) is True
    assert _is_finite(np.float32(0.4)) is True
    assert _is_finite(np.int64(1)) is True


# ---------------------------------------------------------- which layer carries it


def test_the_gate_says_it_failed_closed_not_just_that_it_could_not_measure():
    """The gate's OWN pre-check, pinned by the only thing it observably adds.

    Measured by sabotage: reinstating the NaN-only test in `gate.py` changes no
    verdict, because `check_threshold` now answers COULD_NOT_CHECK for a
    non-finite value and the gate fails closed on that outcome. `check_threshold`
    is the carrier; the pre-check is defence in depth.

    What the pre-check does still contribute is the SENTENCE. "could not be
    measured" tells an operator the metric is missing. "the gate fails closed
    rather than approving an unmeasured metric" tells them the block was a
    deliberate policy rather than an error, which is the difference between
    investigating the data and investigating the gate. Pinned here so the
    pre-check is not silently dropped as dead code by someone who notices, quite
    correctly, that removing it changes no verdict.
    """
    decision = _gate().evaluate_from_metrics({"disparate_impact_ratio": INF})
    message = decision.metric_evaluations[0].message
    assert "could not be measured" in message
    assert "fails closed" in message, (
        "the gate's own pre-check no longer runs; check_threshold is covering "
        "for it. That is not wrong, but it is a different message to the "
        "operator, so make it a decision rather than a silent regression."
    )


def test_the_whole_stack_reproduces_the_original_defect_when_it_is_all_reinstated():
    """A record of what the defect looked like, kept as executable prose.

    This does not sabotage anything; it asserts the CURRENT behaviour at each of
    the three observable surfaces the defect passed through, so that a future
    reader can see exactly which readings changed. With every layer NaN-only,
    the same three calls produced:

        check_threshold("disparate_impact_ratio", inf, 0.8) -> (PASS, "")
        gate.evaluate_from_metrics(...)  -> APPROVED, approved=True, github success
        suite.test_predictions(...)      -> PASSED, "disparate_impact_ratio = inf"
    """
    outcome, message = check_threshold("disparate_impact_ratio", INF, 0.8)
    assert (outcome, bool(message)) == (ThresholdOutcome.COULD_NOT_CHECK, True)

    gate = _gate()
    decision = gate.evaluate_from_metrics({"disparate_impact_ratio": INF})
    assert decision.status is GateStatus.BLOCKED
    assert gate.create_github_check(decision)["conclusion"] == "failure"

    suite = FairnessTestSuite(
        protected_attributes=["group"],
        metrics=["disparate_impact_ratio"],
        thresholds={"disparate_impact_ratio": 0.8},
        compute_metrics_fn=lambda *a, **k: {"disparate_impact_ratio": INF},
    )
    result = suite.test_predictions(
        y_true=np.zeros(100, dtype=int),
        y_pred=np.zeros(100, dtype=int),
        protected_attr=np.array(["a"] * 50 + ["b"] * 50),
        attr_name="group",
        raise_on_failure=False,
    )[0]
    assert result.status is _TestStatus.SKIPPED
    assert "= inf" not in (result.message or ""), "the passing-test message is back"


def test_the_reason_distinguishes_the_two_sentinels():
    """`unmeasurable_reason` exists so the message names WHICH non-measurement
    arrived, because an operator fixes them differently.

    NaN means the computation had nothing to work with: look at the input data,
    the group sizes, the label column. Infinity means a division by zero, which
    on the ratio family means a group with no selections at all: the data is
    there and the model is excluding a group outright. Collapsing both into "not
    measured" would send the reader to the wrong place.
    """
    from vfairness._triage import unmeasurable_reason

    assert unmeasurable_reason(NAN) == "NaN: insufficient evidence"
    assert unmeasurable_reason(INF) == "infinite: no comparison can grade it"
    assert unmeasurable_reason(NEG_INF) == "infinite: no comparison can grade it"
    assert unmeasurable_reason(None) == "no value was recorded"
    assert unmeasurable_reason(True) == "a yes/no flag, not a measurement"
    assert unmeasurable_reason(np.bool_(True)) == "a yes/no flag, not a measurement"
    assert unmeasurable_reason("0.5") == "not a number (str)"
    # A measurement has no reason to give.
    assert unmeasurable_reason(0.42) == ""
    assert unmeasurable_reason(np.float32(0.42)) == ""

    # And the reason actually reaches the operator-facing message, differently
    # for each sentinel. Without this the two collapse and nobody notices.
    nan_msg = check_threshold("disparate_impact_ratio", NAN, 0.8)[1]
    inf_msg = check_threshold("disparate_impact_ratio", INF, 0.8)[1]
    assert "NaN" in nan_msg and "infinite" not in nan_msg
    assert "infinite" in inf_msg and "NaN" not in inf_msg
