"""The CI/CD gate must not certify a comparison it could not measure.

``ModelFairnessGate`` is the highest-consequence surface the package ships and
it is reachable from a bare ``pip install`` with no extras: it is the object a
CI system asks "may this model deploy".

**The finding, measured 2026-09-10 from a clean-room wheel install** (group ``a``
n=100 at selection rate 0.40, group ``b`` n=5 at 0.40, which is 2 of 5 people)::

    ModelFairnessGate(metrics=[...], thresholds={...}).evaluate(...)
        approved = True, status = APPROVED, warnings = []
        computed_metrics = {'demographic_parity_difference': 0.0,
                            'disparate_impact_ratio': 1.0}
        markdown audit trail: "| demographic_parity_difference | 0.0000 | ... | Pass |"
        create_github_check -> conclusion "success"

A five-person group produced the textbook perfect-fairness pair and a green
check. Every other surface in the SAME library refused that comparison on the
same data::

    demographic_parity_difference          -> nan
    report['assessment']['assessable']      -> False
    report['assessment']['fairness_score']  -> None
    assert_fairness                         -> FairnessAssertionError
          "NOT MEASURABLE: value is nan, so the metric certifies nothing"

The sweep of the minority-group size, gate against metric layer::

    n_b=1  -> approved False, DP 0.4000            | metric nan
    n_b=2  -> approved True,  DP 0.1000, DI 0.8000 | metric nan
    n_b=5  -> approved True,  DP 0.0000, DI 1.0000 | metric nan
    n_b=29 -> approved True,  DP 0.0138            | metric nan
    n_b=30 -> approved True,  DP 0.0000            | metric 0.0

n_b=1 blocked only by accident: with one person the gap happens to be large.
The gate got safer as the group got smaller, which is the wrong way round.

**What is NOT the fix.** ``min_group_size=1`` inside ``_compute_default_metrics``
is deliberate and documented: silently DROPPING a small group is the failure the
metric layer's own convention comment warns about, and it is how a 25-person
group that was never selected once reported perfect parity. Every group is still
compared and every measured number is still reported. What the gate withholds
now is the VERDICT, routed to the same could-not-check outcome the NaN and
absent-metric branches already used: a blocking metric refuses approval, a
non-blocking one downgrades to CONDITIONAL. Three states, and APPROVED is not
one of them for an uncertifiable comparison.

Two siblings of the same shape, in the same file, are pinned here too:

* ``evaluate()`` with ``metrics=[]`` returned approved=True and "APPROVED - All
  fairness requirements met". The identical C-05 guard already stood in
  ``evaluate_from_metrics`` and ``evaluate_hierarchical``; the raw-data entry
  point was missed.
* ``evaluate_hierarchical`` collected ``SmallSampleWarning`` objects into a field
  nothing read, and returned "APPROVED - 2/2 levels passed, 1 small-sample
  warning(s)": a verdict that contradicts its own summary line.

Asserting BLOCKED on its own would prove nothing (a gate that blocks everything
is not a gate), so every refusal pin below is paired with an over-correction
control that asserts MEASURED values on adequately sized groups.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
)
from vfairness.operations.cicd.gate import (
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    ModelFairnessGate,
)

DPD = "demographic_parity_difference"
DI = "disparate_impact_ratio"


def _groups(n_a: int, n_b: int, rate_a: float = 0.40, rate_b: float = 0.40):
    """Two groups at the given selection rates.

    ``y_true`` equals ``y_pred`` so the labels are binary and every default gap
    is measurable; the disparity under test is the selection-rate one.
    The label array is ``dtype=object`` on purpose: a numpy ``<U1`` array would
    truncate longer group names and quietly void the fixture.
    """
    k_a, k_b = int(round(rate_a * n_a)), int(round(rate_b * n_b))
    y_pred = np.concatenate(
        [
            np.array([1] * k_a + [0] * (n_a - k_a)),
            np.array([1] * k_b + [0] * (n_b - k_b)),
        ]
    )
    attr = np.array(["group_a"] * n_a + ["group_b"] * n_b, dtype=object)
    return y_pred.copy(), y_pred, attr


def _gate(**kwargs) -> ModelFairnessGate:
    return ModelFairnessGate(
        metrics=[DPD, DI],
        thresholds={DPD: 0.1, DI: 0.8},
        **kwargs,
    )


def _value(decision, metric_name: str) -> float:
    for ev in decision.metric_evaluations:
        if ev.metric_name == metric_name:
            return ev.value
    raise AssertionError(f"{metric_name} absent from {decision.to_dict()}")


def _metric_layer(y_true, y_pred, attr) -> float:
    """The library's own answer for the same comparison, warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(demographic_parity_difference(y_true, y_pred, attr))


# REFUSAL PINS
class TestTheGateDoesNotCertifyAFivePersonGroup:
    def test_the_recorded_reproduction_no_longer_approves(self):
        """The exact measured case: n=100 at 0.40 against 5 people at 0.40."""
        y_true, y_pred, attr = _groups(100, 5)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert decision.approved is False, (
            "a comparison drawn across a five-person group was certified for deployment"
        )
        assert decision.status is GateStatus.BLOCKED

    def test_the_measured_numbers_are_still_reported(self):
        """The fix withholds the VERDICT, it does not hide or mangle the numbers.

        This is the half that stops a future "fix" from returning NaN here and
        calling the group unmeasured: it WAS measured, on five people.
        """
        y_true, y_pred, attr = _groups(100, 5)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert _value(decision, DPD) == pytest.approx(0.0)
        assert _value(decision, DI) == pytest.approx(1.0)
        assert decision.metadata["computed_metrics"][DPD] == pytest.approx(0.0)
        assert decision.metadata["computed_metrics"][DI] == pytest.approx(1.0)

    def test_the_group_and_its_size_are_named(self):
        y_true, y_pred, attr = _groups(100, 5)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [
            ("group_b", 5)
        ]
        assert decision.small_sample_warnings[0].minimum_recommended == 30
        joined = " ".join(decision.blocking_reasons)
        assert "could-not-check" in joined
        assert "minimum group size of 30" in joined
        assert "'group_b' n=5" in joined

    def test_the_audit_trail_no_longer_says_pass(self):
        """The markdown report is kept as the record of the decision."""
        y_true, y_pred, attr = _groups(100, 5)
        report = _gate().evaluate(y_true, y_pred, attr).to_markdown_report()

        assert "**Status**: BLOCKED" in report
        assert "| demographic_parity_difference | 0.0000 | 0.1000 | ✅ Pass |" not in report
        assert "| disparate_impact_ratio | 1.0000 | 0.8000 | ✅ Pass |" not in report
        # The reader has to be able to see WHY, not only that it was refused.
        assert "## Small-Sample Warnings" in report
        assert "- **group_b**: 5 samples (min 30)" in report

    def test_the_github_check_is_not_green(self):
        y_true, y_pred, attr = _groups(100, 5)
        gate = _gate()
        decision = gate.evaluate(y_true, y_pred, attr)
        assert gate.create_github_check(decision)["conclusion"] != "success"

    @pytest.mark.parametrize("n_b", [1, 2, 5, 29])
    def test_the_whole_undersized_sweep_refuses(self, n_b):
        """Every row of the measured sweep below the minimum, not just n_b=5."""
        y_true, y_pred, attr = _groups(100, n_b)
        decision = _gate().evaluate(y_true, y_pred, attr)
        assert decision.approved is False, f"n_b={n_b} was approved"

    @pytest.mark.parametrize("n_b", [2, 5, 29])
    def test_the_gate_does_not_certify_what_the_metric_layer_refuses(self, n_b):
        """Cross-surface pin, both halves MEASURED on the same arrays.

        The divergence is the finding: the metric function reported NOT
        MEASURABLE while the gate reported APPROVED for the same comparison.
        """
        y_true, y_pred, attr = _groups(100, n_b)
        metric = _metric_layer(y_true, y_pred, attr)
        assert metric != metric, f"fixture drifted: the metric layer measured {metric}"
        assert _gate().evaluate(y_true, y_pred, attr).approved is False

    def test_a_non_blocking_metric_downgrades_instead_of_approving(self):
        """Non-blocking failures do not block by contract, but they must never
        yield a clean APPROVED either. Same routing the NaN branch already uses."""
        y_true, y_pred, attr = _groups(100, 5)
        decision = _gate(blocking_metrics=[]).evaluate(y_true, y_pred, attr)

        assert decision.status is GateStatus.CONDITIONAL
        assert decision.status is not GateStatus.APPROVED
        assert any("could-not-check" in w for w in decision.warnings)
        assert _value(decision, DPD) == pytest.approx(0.0)


class TestSiblingsOfTheSameShapeInTheSameFile:
    def test_evaluate_with_no_metrics_configured_does_not_approve(self):
        """C-05 at the third call site. "Every requirement was met" is not the
        same claim as "there were no requirements", and evaluate() reported the
        first for the second while its two sibling entry points already guarded it."""
        y_true, y_pred, attr = _groups(100, 100)
        gate = ModelFairnessGate(config=GateConfig(metrics=[], thresholds={}))
        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is False, "a gate checking nothing approved the model"
        assert decision.status is GateStatus.BLOCKED
        assert any("nothing was checked" in r for r in decision.blocking_reasons)
        assert gate.create_github_check(decision)["conclusion"] != "success"

    def test_hierarchical_small_sample_warnings_now_reach_the_verdict(self):
        """Measured before the fix: approved=True with the summary
        "APPROVED - 2/2 levels passed, 1 small-sample warning(s)"."""
        y_true, y_pred, attr = _groups(100, 5)
        decision = _gate().evaluate_hierarchical(y_true, y_pred, {"g": attr})

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [
            ("group_b", 5)
        ]

    def test_the_hierarchy_minimum_governs_the_level_checks(self):
        """A hierarchy declaring min_group_size=60 must not approve a 40-person
        group at a level and then warn about it in the same object."""
        y_true, y_pred, attr = _groups(100, 40)
        hconfig = HierarchicalGateConfig(min_group_size=60, check_intersections=False)
        decision = _gate().evaluate_hierarchical(
            y_true, y_pred, {"g": attr}, hierarchical_config=hconfig
        )

        assert decision.small_sample_warnings, "the 40-person group was not flagged"
        assert decision.approved is False

    def test_evaluate_from_metrics_says_it_cannot_see_group_sizes(self):
        """Honest empty: that entry point is handed numbers, not labels, so an
        empty warning list there means NOT CHECKED and the docstring says so."""
        decision = _gate().evaluate_from_metrics({DPD: 0.0, DI: 1.0})
        assert decision.small_sample_warnings == []
        assert "cannot see" in ModelFairnessGate.evaluate_from_metrics.__doc__
        assert "NOT CHECKED" in ModelFairnessGate.evaluate_from_metrics.__doc__


# OVER-CORRECTION CONTROLS: measured values, both directions
class TestTheGateStillWorks:
    def test_a_fair_model_on_adequate_groups_is_still_approved(self):
        """n_b=30, the first row of the sweep the metric layer also measures.

        Asserted as MEASURED values, not just approved=True: the same 0.0 / 1.0
        pair the five-person case produced, and here it means something.
        """
        y_true, y_pred, attr = _groups(100, 30)
        gate = _gate()
        decision = gate.evaluate(y_true, y_pred, attr)

        assert _metric_layer(y_true, y_pred, attr) == pytest.approx(0.0)
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert _value(decision, DPD) == pytest.approx(0.0)
        assert _value(decision, DI) == pytest.approx(1.0)
        assert decision.blocking_reasons == []
        assert decision.warnings == []
        assert decision.small_sample_warnings == []
        assert gate.create_github_check(decision)["conclusion"] == "success"

    def test_a_comfortably_fair_model_on_large_groups_is_still_approved(self):
        """Selection rates 0.50 against 0.45: DP gap 0.05, ratio 0.90."""
        y_true, y_pred, attr = _groups(200, 200, rate_a=0.50, rate_b=0.45)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert _value(decision, DPD) == pytest.approx(0.05)
        assert _value(decision, DI) == pytest.approx(0.90)
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED

    def test_an_unfair_model_on_adequate_groups_still_blocks_on_the_threshold(self):
        """Selection rates 0.80 against 0.20: DP gap 0.60, ratio 0.25.

        It must block for the RIGHT reason: the measured breach, not a
        small-sample refusal borrowed from the fix above.
        """
        y_true, y_pred, attr = _groups(100, 100, rate_a=0.80, rate_b=0.20)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert _value(decision, DPD) == pytest.approx(0.60)
        assert _value(decision, DI) == pytest.approx(0.25)
        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert decision.small_sample_warnings == []
        joined = " ".join(decision.blocking_reasons)
        assert "exceeds threshold" in joined
        assert "below the required minimum" in joined
        assert "minimum group size" not in joined

    def test_an_unfair_model_blocks_at_exactly_the_minimum_group_size(self):
        """The boundary in the other direction: n_b=30 is certifiable, and an
        unfair model there is caught on its number rather than waved through."""
        y_true, y_pred, attr = _groups(100, 30, rate_a=0.80, rate_b=0.20)
        decision = _gate().evaluate(y_true, y_pred, attr)

        assert decision.small_sample_warnings == []
        assert _value(decision, DPD) == pytest.approx(0.60)
        assert decision.approved is False

    def test_a_hierarchical_run_on_large_groups_is_still_approved(self):
        rng = np.random.default_rng(7)
        n = 2000
        gender = rng.choice(np.array(["Male", "Female"], dtype=object), size=n)
        race = rng.choice(np.array(["White", "Black"], dtype=object), size=n)
        y_true = rng.integers(0, 2, size=n)
        y_pred = y_true.copy()

        decision = ModelFairnessGate(metrics=[DPD], thresholds={DPD: 0.1}).evaluate_hierarchical(
            y_true, y_pred, {"gender": gender, "race": race}
        )

        assert decision.small_sample_warnings == []
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert set(decision.level_results) == {
            "overall",
            "attr:gender",
            "attr:race",
            "intersection:gender_x_race",
        }
