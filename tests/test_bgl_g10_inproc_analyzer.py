"""G10 grading pins: in_processing.analyzer records (the three report dataclasses).

One defect pinned:

D10 A MINTED VALUE AND A CORRECT REFUSAL ON ONE LINE. ``FairnessTrainingReport``
    already refuses to invent a BASELINE accuracy: this method's own docstring
    records that the baseline ACCURACY line "was left behind by that fix and
    still read ``.get('accuracy', 0):.4f``", and it was corrected there. The
    PER-METHOD line in the METHOD COMPARISON block was never reached by the same
    fix and kept ``Acc={comp.accuracy:.4f}``, while ``compare_methods``
    deliberately sets accuracy to NaN when it could not be measured, recording
    the reason in ``parameters['accuracy_not_measured_reason']``. Measured:

        Exponentiated_Gradient: Acc=nan, Violation=not measured (the constraint
        could not be evaluated) [?]

    and the recorded reason appeared nowhere in the whole report, so a reader
    could not tell "the method reported a non-finite accuracy" from "12 of 240
    rows could not be scored".

Every pin is paired with a control asserting the measured case's real number,
including a measured accuracy of exactly 0.0, which must never read as absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vfairness.in_processing.analyzer import (
    FairnessTrainingReport,
    MethodComparison,
    TrainingRecommendation,
)

REASON = "12 of 240 row(s) carry a non-finite target or prediction"


def _report(comparisons, **kwargs):
    defaults = dict(
        timestamp="2026-09-30T00:00:00+00:00",
        data_info={"n_samples": 240, "n_groups": 3},
        task_type="classification",
        baseline_metrics={},
        fairness_analysis={},
        method_comparisons=comparisons,
        recommendation=TrainingRecommendation("Exponentiated_Gradient", "high", "rationale"),
        tradeoff_analysis={},
        critical_issues=[],
        action_items=[],
    )
    defaults.update(kwargs)
    return FairnessTrainingReport(**defaults)


def _method_block(report):
    return report.summary().split("METHOD COMPARISON")[1].split("RECOMMENDATION")[0]


# ---------------------------------------------------------- D10: the per-method line


@pytest.mark.parametrize("unmeasured", [float("nan"), float("inf"), -float("inf"), None, True])
def test_a_method_accuracy_that_is_not_a_measurement_is_not_printed_as_a_value(unmeasured):
    """``True`` is in here because a bool is not a magnitude: it would print as
    1.0000, a perfect accuracy from a flag."""
    report = _report([MethodComparison("Exponentiated_Gradient", unmeasured, None, None, 1.0, {})])
    block = _method_block(report)
    assert "nan" not in block and "inf" not in block, block
    assert "1.0000" not in block
    assert "Acc=not measured" in block


def test_the_recorded_reason_reaches_the_reader():
    """D10's other half: the producer took the trouble to record WHY, and the text
    surface dropped it, so two different causes read identically."""
    report = _report(
        [
            MethodComparison(
                "Exponentiated_Gradient",
                float("nan"),
                None,
                None,
                1.0,
                {"accuracy_measured": False, "accuracy_not_measured_reason": REASON},
            )
        ]
    )
    block = _method_block(report)
    assert REASON in block, block
    assert "accuracy not measured:" in block


def test_a_missing_reason_is_not_invented():
    report = _report([MethodComparison("No_Reason", float("nan"), 0.2, False, 1.0, {})])
    block = _method_block(report)
    assert "Acc=not measured" in block
    assert "accuracy not measured:" not in block, "no reason was recorded, so none is printed"
    assert "Violation=0.2000" in block, "the measured half of the line is untouched"


def test_control_a_measured_accuracy_including_exactly_zero_still_prints():
    """CONTROL for D10. A guard that refuses every accuracy passes every pin above
    and destroys the block."""
    report = _report(
        [
            MethodComparison("Baseline", 0.8234, 0.31, False, 0.5, {}),
            MethodComparison("Zero_Acc", 0.0, 0.0, True, 0.5, {}),
            MethodComparison("Perfect", 1.0, 0.0, True, 0.5, {}),
        ]
    )
    block = _method_block(report)
    assert "Baseline: Acc=0.8234, Violation=0.3100 [✗]" in block
    assert "Zero_Acc: Acc=0.0000, Violation=0.0000 [✓]" in block
    assert "Perfect: Acc=1.0000" in block
    assert "not measured" not in block


def test_the_two_halves_of_the_line_refuse_independently():
    """A measured accuracy beside an unevaluated constraint, and the reverse, must
    each keep the half that WAS measured."""
    report = _report(
        [
            MethodComparison("Acc_Only", 0.75, None, None, 0.5, {}),
            MethodComparison("Viol_Only", float("nan"), 0.12, False, 0.5, {}),
        ]
    )
    block = _method_block(report)
    assert "Acc_Only: Acc=0.7500, Violation=not measured" in block
    assert (
        "Viol_Only: Acc=not measured (no finite accuracy was reported), Violation=0.1200" in block
    )


def test_three_glyphs_never_two_and_the_ungraded_count_is_shown():
    report = _report(
        [
            MethodComparison("Met", 0.8, 0.01, True, 0.1, {}),
            MethodComparison("Breach", 0.8, 0.31, False, 0.1, {}),
            MethodComparison("Ungraded", 0.8, None, None, 0.1, {}),
        ]
    )
    block = _method_block(report)
    assert "[✓]" in block and "[✗]" in block and "[?]" in block
    assert "1 of 3 method(s) reported no constraint result" in block
    assert "neither cleared nor in breach" in block


# ------------------------------------------------------------------- the records


def test_method_comparison_three_states_and_a_dict_that_drops_nothing():
    unevaluated = MethodComparison("m", 0.8, None, None)
    breach = MethodComparison("m", 0.8, 0.31, False)
    met = MethodComparison("m", 0.8, 0.01, True)
    assert unevaluated.fairness_violation is None and unevaluated.constraint_satisfied is None
    assert breach.constraint_satisfied is False and met.constraint_satisfied is True
    for record in (unevaluated, breach, met):
        assert set(record.to_dict()) == set(record.__dataclass_fields__)
        assert record.to_dict()["constraint_satisfied"] is record.constraint_satisfied
    # NaN must never be the not-evaluated state here: it is not None, so it would
    # pass every `is not None` gate in the rendering layer and be drawn as a
    # measurement. The record's own type allows it, so the pin is on the reading.
    assert MethodComparison("m", 0.8, float("nan"), None).fairness_violation is not None
    MethodComparison("m", 0.8, None, None).parameters["leak"] = 1
    assert MethodComparison("m", 0.8, None, None).parameters == {}


def test_training_recommendation_carries_optional_tradeoffs_without_substituting_zero():
    recommendation = TrainingRecommendation(
        recommended_method="Exponentiated_Gradient",
        priority="high",
        rationale="rationale",
        alternative_methods=["Adversarial"],
        expected_tradeoff={"accuracy": 0.81, "fairness_violation": None},
        implementation_notes="notes",
    )
    payload = recommendation.to_dict()
    assert set(payload) == set(recommendation.__dataclass_fields__)
    assert payload["expected_tradeoff"]["fairness_violation"] is None, "never a substituted 0.0"
    assert payload["expected_tradeoff"]["accuracy"] == pytest.approx(0.81)
    # An empty dict is "no method was recommended at all", a different claim.
    assert TrainingRecommendation("none", "low", "r").expected_tradeoff == {}
    TrainingRecommendation("n", "low", "r").alternative_methods.append("leak")
    TrainingRecommendation("n", "low", "r").expected_tradeoff["leak"] = 1.0
    assert TrainingRecommendation("n", "low", "r").alternative_methods == []
    assert TrainingRecommendation("n", "low", "r").expected_tradeoff == {}


def test_a_report_with_no_baseline_does_not_print_a_zero_accuracy():
    """The half of this defect that WAS already fixed, pinned so it stays fixed."""
    report = _report([], baseline_metrics={})
    text = report.summary()
    assert "Accuracy: 0.0000" not in text
    assert "Accuracy: not measured (no baseline was scored)" in text
    assert "Fairness Violation: not measured (the constraint could not be evaluated)" in text
    assert "Constraint Satisfied: not assessed (the constraint could not be evaluated)" in text


def test_a_baseline_reason_is_preferred_over_the_generic_phrase():
    report = _report(
        [],
        baseline_metrics={
            "accuracy": float("nan"),
            "accuracy_not_measured_reason": "the regression R^2 is undefined",
            "fairness_violation": 0.0,
            "constraint_satisfied": True,
        },
    )
    text = report.summary()
    assert "Accuracy: not measured (the regression R^2 is undefined)" in text
    # CONTROL on the same report: a measured 0.0 violation is a real result.
    assert "Fairness Violation: 0.0000" in text
    assert "Constraint Satisfied: True" in text


def test_a_training_that_crashed_is_not_a_method_that_was_not_asked_for():
    report = _report(
        [MethodComparison("Baseline", 0.8, 0.01, True, 0.1, {})],
        failed_methods=[{"method": "adversarial", "error": "ValueError: no groups"}],
    )
    text = report.summary()
    assert "METHODS THAT COULD NOT BE COMPARED" in text
    assert "adversarial: ValueError: no groups" in text
    assert "neither cleared nor found in breach" in text
    # And a clean run must not print that block.
    assert (
        "METHODS THAT COULD NOT BE COMPARED"
        not in _report([MethodComparison("Baseline", 0.8, 0.01, True, 0.1, {})]).summary()
    )


def test_report_defaults_are_per_instance_and_the_fairness_block_has_three_states():
    assert _report([]).failed_methods == [] and _report([]).metadata == {}
    _report([]).failed_methods.append({"method": "leak"})
    assert _report([]).failed_methods == []

    text = _report(
        [],
        fairness_analysis={
            "demographic_parity": 0.25,
            "base_rate_disparity": None,
            "is_significant": True,
            "worst_group": "C",
        },
    ).summary()
    assert "demographic_parity: 0.2500" in text
    assert "base_rate_disparity: not measured" in text
    assert "is_significant: True" in text, "a bool is not a magnitude"
    assert "1.0000" not in text.split("FAIRNESS ANALYSIS")[1].split("=" * 70)[0]
    assert "worst_group: C" in text
    assert math.isfinite(np.float64(1.0))  # keeps the imports honest
