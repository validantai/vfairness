"""VF-5: the four-fifths rule must be REACHABLE through ModelFairnessGate.evaluate().

The finding: ``_compute_default_metrics`` emitted only four ``*_difference`` gaps
(demographic parity, equalized odds, false positive rate, predictive parity), so a
user who configured a ``disparate_impact_ratio`` threshold against the raw-data
entry point named a metric the gate could never compute. It came back as
could-not-check and BLOCKED every deployment, fair or unfair alike, unless the
caller supplied a custom ``compute_metrics_fn``. That is fail-closed and therefore
safe, but it made the one statistic a regulator actually names (adverse impact
below 0.80) unenforceable through the public API.

The tests below therefore check three separate things, because "BLOCKED" on its own
proves nothing here: the pre-fix gate blocked everything.

  1. the ratio is COMPUTED (a finite number, equal to the canonical library
     function, not a reimplementation),
  2. the verdict is CORRECT in both directions (an unfair dataset blocks ON THE
     THRESHOLD, a fair one is approved), and
  3. an input on which the ratio genuinely cannot be measured still fails CLOSED.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import disparate_impact_ratio
from vfairness.operations.cicd.gate import GateStatus, ModelFairnessGate

FOUR_FIFTHS = "disparate_impact_ratio"


def _two_group_data(sel_a: int, sel_b: int, n: int = 100):
    """Two equal-sized groups with ``sel_a`` / ``sel_b`` positive predictions.

    ``y_true`` alternates so both classes are present in each group, which keeps
    every other default gap metric measurable and the label vector binary.
    """
    prot = np.array(["A"] * n + ["B"] * n)
    y_pred = np.concatenate(
        [
            np.array([1] * sel_a + [0] * (n - sel_a)),
            np.array([1] * sel_b + [0] * (n - sel_b)),
        ]
    )
    y_true = np.array([1, 0] * n)
    return y_true, y_pred, prot


def _four_fifths_gate() -> ModelFairnessGate:
    return ModelFairnessGate(
        metrics=[FOUR_FIFTHS],
        thresholds={FOUR_FIFTHS: 0.80},
    )


def _evaluation(decision, metric_name: str):
    for ev in decision.metric_evaluations:
        if ev.metric_name == metric_name:
            return ev
    raise AssertionError(f"{metric_name} absent from the decision: {decision.to_dict()}")


class TestFourFifthsIsComputable:
    def test_default_metrics_emit_the_ratio(self):
        # The defect itself: the key was simply not in the dict.
        y_true, y_pred, prot = _two_group_data(sel_a=80, sel_b=20)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)  # method needs no state
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert FOUR_FIFTHS in metrics
        assert np.isfinite(metrics[FOUR_FIFTHS])

    def test_value_equals_the_canonical_library_function(self):
        # Proves the gate REUSES the library implementation rather than
        # restating the maths, so the two surfaces cannot drift apart.
        y_true, y_pred, prot = _two_group_data(sel_a=80, sel_b=20)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        expected = disparate_impact_ratio(y_true, y_pred, prot, min_group_size=1)
        assert metrics[FOUR_FIFTHS] == pytest.approx(expected)
        assert metrics[FOUR_FIFTHS] == pytest.approx(0.25)  # 0.20 / 0.80

    def test_every_group_is_compared_not_only_the_largest(self):
        # The gate's convention is that EVERY group present in the data is
        # compared and none is dropped for being small (HierarchicalGateConfig
        # .min_group_size only warns). A tiny, never-selected third group is the
        # disparity, and it must not be filtered out of the ratio.
        prot = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 12)
        y_pred = np.concatenate(
            [np.array([1] * 50 + [0] * 50), np.array([1] * 50 + [0] * 50), np.zeros(12, dtype=int)]
        )
        y_true = np.array([1, 0] * 106)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert metrics[FOUR_FIFTHS] == pytest.approx(0.0)  # C is never selected


class TestFourFifthsVerdictIsCorrectInBothDirections:
    def test_unfair_dataset_is_blocked_on_the_threshold(self):
        # Selection rates 0.80 vs 0.20 -> ratio 0.25, far below the 0.80 floor.
        # Asserting BLOCKED alone would pass on the BROKEN gate too (it blocked
        # everything), so the value and the REASON are asserted as well.
        y_true, y_pred, prot = _two_group_data(sel_a=80, sel_b=20)
        decision = _four_fifths_gate().evaluate(y_true, y_pred, prot)

        ev = _evaluation(decision, FOUR_FIFTHS)
        assert np.isfinite(ev.value), "the ratio must be measured, not could-not-check"
        assert ev.value == pytest.approx(0.25)
        assert ev.passed is False
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        # Blocked because the measured ratio breaches the floor, NOT because the
        # metric was missing or unmeasurable.
        joined = " ".join(decision.blocking_reasons)
        assert "could not be computed" not in joined
        assert "could not be measured" not in joined

    def test_fair_dataset_is_approved(self):
        # Selection rates 0.50 vs 0.45 -> ratio 0.90, above the 0.80 floor.
        # This is the case the broken gate got WRONG: it blocked a fair model.
        y_true, y_pred, prot = _two_group_data(sel_a=50, sel_b=45)
        decision = _four_fifths_gate().evaluate(y_true, y_pred, prot)

        ev = _evaluation(decision, FOUR_FIFTHS)
        assert ev.value == pytest.approx(0.90)
        assert ev.passed is True
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.blocking_reasons == []

    def test_a_ratio_is_not_graded_as_a_difference(self):
        # A ratio breaches BELOW its threshold. Graded as a difference
        # ("lower is better"), 0.25 would read as a comfortable PASS and 0.90
        # as a FAIL, i.e. the verdict exactly inverted.
        y_true, unfair, prot = _two_group_data(sel_a=80, sel_b=20)
        _, fair, _ = _two_group_data(sel_a=50, sel_b=45)
        gate = _four_fifths_gate()
        assert gate.evaluate(y_true, unfair, prot).approved is False
        assert gate.evaluate(y_true, fair, prot).approved is True


class TestUnmeasurableStillFailsClosed:
    def test_single_group_cannot_be_measured_and_blocks(self):
        # One group: there is no pair to compare, so a four-fifths reading would
        # certify no adverse impact on a test that never ran.
        n = 100
        prot = np.array(["A"] * n)
        y_pred = np.array([1] * 50 + [0] * 50)
        y_true = np.array([1, 0] * 50)
        decision = _four_fifths_gate().evaluate(y_true, y_pred, prot)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        ev = _evaluation(decision, FOUR_FIFTHS)
        assert ev.passed is False
        assert not np.isfinite(ev.value)

    def test_non_binary_labels_fail_closed_instead_of_raising(self):
        # The canonical implementation validates its inputs and refuses labels
        # the four raw-numpy gaps tolerate. A deployment gate must neither crash
        # nor omit the metric: it must report could-not-check and BLOCK.
        prot = np.array(["A"] * 100 + ["B"] * 100)
        y_pred = np.array([2] * 50 + [0] * 50 + [1] * 50 + [0] * 50)  # 2 is not binary
        y_true = np.array([1, 0] * 100)
        decision = _four_fifths_gate().evaluate(y_true, y_pred, prot)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        ev = _evaluation(decision, FOUR_FIFTHS)
        assert not np.isfinite(ev.value)


class TestNoOvercorrection:
    def test_the_four_difference_gaps_are_unchanged(self):
        # Purely additive: the metrics the gate already computed keep their
        # values and their names.
        y_true, y_pred, prot = _two_group_data(sel_a=80, sel_b=20)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        metrics = gate._compute_default_metrics(y_true, y_pred, prot)
        assert metrics["demographic_parity_difference"] == pytest.approx(0.60)
        for name in (
            "equalized_odds_difference",
            "false_positive_rate_difference",
            "predictive_parity_difference",
        ):
            assert name in metrics

    def test_fewer_than_two_groups_still_returns_no_metrics(self):
        prot = np.array(["A"] * 20)
        gate = ModelFairnessGate.__new__(ModelFairnessGate)
        assert (
            gate._compute_default_metrics(np.array([1, 0] * 10), np.array([1, 0] * 10), prot) == {}
        )


class TestGateDocumentsWhatItCanCompute:
    def test_the_docstring_names_the_computable_metrics(self):
        # A user configuring a threshold must be able to read which metric names
        # evaluate() can actually measure, rather than discovering by deployment
        # that one of them is unreachable.
        text = (ModelFairnessGate.__doc__ or "") + (ModelFairnessGate.evaluate.__doc__ or "")
        for name in (
            "demographic_parity_difference",
            "equalized_odds_difference",
            "false_positive_rate_difference",
            "predictive_parity_difference",
            FOUR_FIFTHS,
        ):
            assert name in text, f"{name} is not documented on the gate"
        assert "compute_metrics_fn" in text
