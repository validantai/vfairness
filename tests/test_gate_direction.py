"""Deployment gate: metric direction, and fail-closed on what cannot be checked.

Two release blockers are pinned here.

**Blocker 1, the gate treated every metric as lower-is-better.** ``gate.py`` used
a bare ``abs(value) > threshold`` commented "for difference metrics, lower is
better", with no direction handling at all. Under the four-fifths rule
(``disparate_impact_ratio >= 0.80``) that inverted every case, measured against
the code as it stood:

    DI = 0.00 (protected group NEVER selected) -> APPROVED   (must BLOCK)
    DI = 0.50 (clear violation)                -> APPROVED   (must BLOCK)
    DI = 0.85 (compliant)                      -> BLOCKED    (must APPROVE)
    DI = 1.00 (PERFECT PARITY)                 -> BLOCKED    (must APPROVE)

and it wrote ``| disparate_impact_ratio | 0.0000 | 0.8000 | Pass |`` into the
markdown report kept as the audit trail.

**Blocker 2, ``evaluate_from_metrics`` approved when a required metric was
absent.** ``evaluate()`` had already been fixed to fail closed on a configured
metric that is missing from the metrics dict; ``evaluate_from_metrics()`` had
not, so ``evaluate_from_metrics({})`` returned ``approved=True`` and
``create_github_check`` reported ``conclusion='success'`` for a required metric
that was never measured.

Every direction case below is driven in BOTH directions: the ratio family must
stop passing when it is unfair, and the difference family must not be inverted
in the process. The unknown-metric cases prove the fail-closed property, which
is the one that must never be weakened.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
    relax_threshold,
)
from vfairness.operations.cicd.gate import (
    HierarchicalGateConfig,
    ModelFairnessGate,
)

DI = "disparate_impact_ratio"
DPD = "demographic_parity_difference"
FOUR_FIFTHS = 0.80


def _ratio_gate(threshold=FOUR_FIFTHS, blocking=True):
    return ModelFairnessGate(
        metrics=[DI],
        thresholds={DI: threshold},
        blocking_metrics=[DI] if blocking else [],
    )


# ---------------------------------------------------------------------------
# The full inverted table, on evaluate_from_metrics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,should_approve",
    [
        (0.00, False),  # protected group never selected: the worst possible case
        (0.50, False),  # clear four-fifths violation
        (0.85, True),  # compliant
        (1.00, True),  # perfect parity
    ],
)
def test_disparate_impact_ratio_direction_from_metrics(value, should_approve):
    decision = _ratio_gate().evaluate_from_metrics({DI: value})
    assert decision.approved is should_approve, (
        f"DI={value:.2f} must {'APPROVE' if should_approve else 'BLOCK'}"
    )
    assert decision.status.value == ("approved" if should_approve else "blocked")


@pytest.mark.parametrize(
    "value,should_approve",
    [(0.00, False), (0.50, False), (0.85, True), (1.00, True)],
)
def test_disparate_impact_ratio_direction_from_evaluate(value, should_approve):
    """Same table through evaluate(), which is the documented CI/CD entry point."""
    gate = ModelFairnessGate(
        metrics=[DI],
        thresholds={DI: FOUR_FIFTHS},
        blocking_metrics=[DI],
        compute_metrics_fn=lambda y_true, y_pred, prot: {DI: value},
    )
    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 1] * 50)
    prot = np.array(["A", "B"] * 50)
    decision = gate.evaluate(y_true, y_pred, prot)
    assert decision.approved is should_approve


def test_zero_ratio_is_not_reported_as_pass_in_the_audit_trail():
    """The markdown report is kept as the audit trail; it said 'Pass' for DI=0.00."""
    decision = _ratio_gate().evaluate_from_metrics({DI: 0.0})
    report = decision.to_markdown_report()
    assert "| disparate_impact_ratio | 0.0000 | 0.8000 | ✅ Pass |" not in report
    assert "Fail" in report
    assert "below the required minimum" in report


def test_zero_ratio_fails_the_github_check():
    decision = _ratio_gate().evaluate_from_metrics({DI: 0.0})
    check = ModelFairnessGate().create_github_check(decision)
    assert check["conclusion"] == "failure"


# ---------------------------------------------------------------------------
# Control: difference metrics must NOT be inverted by the fix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,should_approve",
    [
        (0.00, True),  # perfect parity for a difference metric is ZERO
        (0.05, True),  # within the 0.10 threshold
        (0.30, False),  # exceeds it
        (-0.30, False),  # magnitude, so the sign must not rescue it
    ],
)
def test_difference_metric_direction_not_inverted(value, should_approve):
    gate = ModelFairnessGate(
        metrics=[DPD],
        thresholds={DPD: 0.10},
        blocking_metrics=[DPD],
    )
    assert gate.evaluate_from_metrics({DPD: value}).approved is should_approve


# ---------------------------------------------------------------------------
# Fail closed: a metric whose direction cannot be determined
# ---------------------------------------------------------------------------
def test_unknown_metric_direction_blocks_rather_than_approving():
    """The single most important property: unknown direction is never a pass.

    0.05 against a 0.10 threshold would have been a comfortable PASS under the
    old lower-is-better assumption. We cannot know that assumption holds for a
    name we do not recognise, so the gate must refuse instead of guessing.
    """
    gate = ModelFairnessGate(
        metrics=["mystery_metric"],
        thresholds={"mystery_metric": 0.10},
        blocking_metrics=["mystery_metric"],
    )
    decision = gate.evaluate_from_metrics({"mystery_metric": 0.05})
    assert decision.approved is False
    assert decision.status.value == "blocked"
    assert any("no known better-direction" in r for r in decision.blocking_reasons)
    assert decision.metric_evaluations[0].passed is False


def test_unknown_metric_direction_non_blocking_is_conditional_not_clean():
    gate = ModelFairnessGate(
        metrics=["mystery_metric"],
        thresholds={"mystery_metric": 0.10},
        blocking_metrics=[],
    )
    decision = gate.evaluate_from_metrics({"mystery_metric": 0.05})
    assert decision.approved is True
    assert decision.status.value == "conditional"
    assert any("no known better-direction" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# Blocker 2: a required metric that is simply absent
# ---------------------------------------------------------------------------
def test_absent_required_metric_blocks_deployment():
    decision = _ratio_gate().evaluate_from_metrics({})
    assert decision.approved is False
    assert decision.status.value == "blocked"
    assert any("was not provided" in r for r in decision.blocking_reasons)


def test_absent_required_metric_fails_the_github_check():
    decision = _ratio_gate().evaluate_from_metrics({})
    assert ModelFairnessGate().create_github_check(decision)["conclusion"] == "failure"


def test_absent_metric_appears_in_the_report_and_never_as_a_pass():
    """SUBJECT UNCHANGED: a required metric that was never supplied has to be
    VISIBLE in the audit trail and must never read as a pass. Only the mechanism
    moved (G05, 2026-09-30).

    This asserted ``"Fail" in to_markdown_report()``, which was the row label the
    renderer produced at the time. That label was itself the defect: the row
    mapped straight off ``passed``, so a metric nobody could compute and a
    measured breach rendered byte-identically (``| ... | nan | 0.8000 | Fail |``),
    and the gate's own message for this case says "fails closed rather than
    approving an unevaluated metric", not that anything exceeded anything. The
    row now states the could-not-check and the decision is still BLOCKED, so the
    fail-closed property this file exists to protect is asserted directly below
    rather than through a word that has changed meaning.
    """
    decision = _ratio_gate().evaluate_from_metrics({})
    assert len(decision.metric_evaluations) == 1
    ev = decision.metric_evaluations[0]
    assert ev.passed is False and ev.metric_name == DI

    report = decision.to_markdown_report()
    assert DI in report
    assert "BLOCKED" in report
    assert "Could not check" in report
    assert "Pass" not in report
    # And the reason, in the words of the gate, on the same page.
    assert "was not provided" in report


def test_absent_non_blocking_metric_is_conditional_not_approved_clean():
    gate = _ratio_gate(blocking=False)
    decision = gate.evaluate_from_metrics({})
    assert decision.approved is True
    assert decision.status.value == "conditional"
    assert any("was not provided" in w for w in decision.warnings)


def test_absent_metric_does_not_mask_a_supplied_one():
    gate = ModelFairnessGate(
        metrics=[DI, DPD],
        thresholds={DI: FOUR_FIFTHS, DPD: 0.10},
        blocking_metrics=[DI, DPD],
    )
    decision = gate.evaluate_from_metrics({DPD: 0.01})
    assert decision.approved is False
    assert [e.metric_name for e in decision.metric_evaluations] == [DI, DPD]
    assert decision.metric_evaluations[1].passed is True


# ---------------------------------------------------------------------------
# Intersectional relaxation must be permissive, not stricter
# ---------------------------------------------------------------------------
def test_relax_threshold_moves_each_direction_the_permissive_way():
    # A 1.2x relaxation on the four-fifths floor must LOWER it, not raise it to
    # 0.96 (which is stricter than the base check it relaxes).
    assert relax_threshold(DI, 0.80, 1.2) == pytest.approx(0.80 / 1.2)
    assert relax_threshold(DI, 0.80, 1.2) < 0.80
    # A difference metric relaxes by allowing a LARGER gap.
    assert relax_threshold(DPD, 0.10, 1.2) == pytest.approx(0.12)
    # Unknown direction: leave it exactly where it was rather than move it the
    # wrong way.
    assert relax_threshold("mystery_metric", 0.10, 1.2) == 0.10


def test_hierarchical_intersection_threshold_relaxes_a_ratio_downward():
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    race = np.where(rng.random(n) < 0.5, "B", "W")

    gate = ModelFairnessGate(
        metrics=[DI],
        thresholds={DI: FOUR_FIFTHS},
        blocking_metrics=[DI],
        compute_metrics_fn=lambda y_true, y_pred, prot: {DI: 0.70},
    )
    hconfig = HierarchicalGateConfig(
        check_overall=False,
        check_single_attributes=False,
        min_group_size=1,
    )
    decision = gate.evaluate_hierarchical(
        y_true, y_pred, {"gender": gender, "race": race}, hierarchical_config=hconfig
    )
    level = decision.level_results["intersection:gender_x_race"]
    seen = [e.threshold for e in level.metric_evaluations]
    assert seen == [pytest.approx(0.80 / 1.2)]
    # 0.70 clears the RELAXED intersectional floor (0.667) while breaching the
    # base 0.80 floor. Under the blind 1.2x multiplier the floor became 0.96 and
    # the "relaxed" level was stricter than the base level.
    assert level.approved is True


# ---------------------------------------------------------------------------
# The shared resolver itself
# ---------------------------------------------------------------------------
def test_direction_resolver_core_cases():
    assert metric_direction(DI) is MetricDirection.HIGHER_IS_BETTER
    assert metric_direction("demographic_parity_ratio") is MetricDirection.HIGHER_IS_BETTER
    assert metric_direction("worst_group_accuracy") is MetricDirection.HIGHER_IS_BETTER
    assert metric_direction(DPD) is MetricDirection.LOWER_IS_BETTER
    assert metric_direction("equalized_odds_difference") is MetricDirection.LOWER_IS_BETTER
    assert metric_direction("mystery_metric") is MetricDirection.UNKNOWN
    assert metric_direction("") is MetricDirection.UNKNOWN


def test_direction_resolver_is_not_fooled_by_the_calibration_substring():
    """The cali-BRATIO-n trap: "ratio" is a substring of "calibration"."""
    for name in (
        "calibration_difference",
        "multicalibration",
        "integrated_calibration_index",
        "calibration_disparity",
    ):
        assert metric_direction(name) is MetricDirection.LOWER_IS_BETTER, name


def test_check_threshold_states():
    assert check_threshold(DI, 1.0, 0.8)[0] is ThresholdOutcome.PASS
    assert check_threshold(DI, 0.0, 0.8)[0] is ThresholdOutcome.FAIL
    assert check_threshold(DPD, 0.05, 0.1)[0] is ThresholdOutcome.PASS
    assert check_threshold(DPD, 0.5, 0.1)[0] is ThresholdOutcome.FAIL
    # Three states, never two: unmeasurable and undeterminable are their own
    # outcome, and neither of them is PASS.
    assert check_threshold(DPD, float("nan"), 0.1)[0] is ThresholdOutcome.COULD_NOT_CHECK
    assert check_threshold(DPD, 0.05, float("nan"))[0] is ThresholdOutcome.COULD_NOT_CHECK
    assert check_threshold("mystery_metric", 0.05, 0.1)[0] is ThresholdOutcome.COULD_NOT_CHECK


def test_check_threshold_never_reports_pass_for_a_could_not_check():
    for outcome, _msg in (
        check_threshold("mystery_metric", v, 0.1) for v in (-1.0, 0.0, 0.05, 5.0)
    ):
        assert outcome is not ThresholdOutcome.PASS


# ---------------------------------------------------------------------------
# Pulse metric tone (pipeline._metric_tone was never executed by any test)
# ---------------------------------------------------------------------------
def test_pulse_metric_tone_gap_direction():
    from vfairness.operations.pulse.pipeline import _metric_tone

    # A gap below its threshold passes; one whose whole CI is above it is critical.
    assert _metric_tone(0.05, 0.02, 0.08, 0.10, "dp") == "pass"
    assert _metric_tone(0.15, 0.05, 0.25, 0.10, "dp") == "warn"
    assert _metric_tone(0.30, 0.20, 0.40, 0.10, "dp") == "critical"


def test_pulse_metric_tone_ratio_direction_is_not_inverted():
    from vfairness.operations.pulse.pipeline import _metric_tone

    # A ratio breaches BELOW its threshold. The bare `value > threshold` scored
    # 0.0 (never selects the protected group) as "pass" and 1.0 as "critical".
    assert _metric_tone(0.0, 0.0, 0.0, 0.80, DI) == "critical"
    assert _metric_tone(0.50, 0.40, 0.60, 0.80, DI) == "critical"
    assert _metric_tone(0.75, 0.60, 0.90, 0.80, DI) == "warn"
    assert _metric_tone(1.0, 0.95, 1.05, 0.80, DI) == "pass"


def test_pulse_metric_tone_unknown_metric_is_not_pass():
    from vfairness.operations.pulse.pipeline import _metric_tone

    assert _metric_tone(0.02, 0.01, 0.03, 0.10, "mystery_metric") == "unknown"
