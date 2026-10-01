"""BGL3 batch agents_multi-2: does each unit refuse honestly when nothing can
be measured?

Thirteen units across ``vfairness.agents`` and ``vfairness.multi_agent`` were
run on inputs where the quantity they report does not exist, and on healthy
inputs beside them. Eleven already refused honestly and are pinned here so a
refactor cannot quietly take the refusal away. Two fabricated a verdict and were
fixed:

1. ``ActionBiasAnalyzer.analyze_delegation`` reported a significance verdict its
   test could not have reached, in BOTH directions. Measured 2026-09-27:

   * 3 group-A tasks all routed to "junior" against 3 group-B tasks all routed
     to "senior" (total segregation, the strongest routing bias the method can
     express) returned ``is_significant=False`` per target and overall, with no
     warning. Fisher's floor at 3 against 3 is 0.1, so no arrangement of that
     data could have said True.
   * 12 group-A tasks over two routes against 1 group-B task on a third
     returned ``overall_p_value=0.0015`` and ``is_significant=True`` from the
     asymptotic chi-square, on a table whose smallest EXPECTED cell count is
     0.077 and whose exact conditional p is 0.0769. A fabricated FINDING.
   * A single observed target returned ``overall_chi2=0.0`` and
     ``overall_p_value=1.0``: the p of a test that cannot run at 0 degrees of
     freedom, and 1.0 is the strongest "no difference" that field has.

2. ``ToolBiasAuditor.compute_selection_disparity`` (and so
   ``analyze_tool_calls``) reported ``is_significant=False`` for TOTAL EXCLUSION
   at 3 calls per group, where the smallest attainable p is 0.1. Measured
   2026-09-27: ``disparity_ratio 0.0`` (a true measurement of total exclusion)
   sat beside a False that was not a reading about those tools at all.

Three further fabrications of the same shape were found in code paths the public
entries cannot reach today, and are fixed for the reason
``DelegationRoutingAuditor._cramers_v`` gives for its own unreachable refusal
(the helper is reachable on its own, and a fabricated 0.0 is invisible while a
NaN is not): ``_per_route_disparity`` returned 0.0 per route for a one-group
table, ``detect_coalitions`` returned ``[]`` for an empty matrix in silence, and
``record_sample`` accepted an agent set introduced after a first sample that had
none, which diverged the per-agent arrays from ``sample_groups`` until numpy
raised an IndexError naming neither.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness._not_assessed import NOT_ASSESSED
from vfairness.agents import action_bias as action_bias_module
from vfairness.agents.action_bias import ActionBiasAnalyzer
from vfairness.agents.tool_bias import ToolBiasAuditor
from vfairness.multi_agent import delegation as delegation_module
from vfairness.multi_agent.collusion import AdversarialCollusionDetector
from vfairness.multi_agent.compositionality import CompositionalityAnalyzer
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.multi_agent.emergent import EmergentBiasDetector
from vfairness.multi_agent.groupthink import GroupthinkDetector
from vfairness.multi_agent.harness import MultiAgentRunHarness
from vfairness.multi_agent.negotiation import NegotiationFairnessTracker

NAN = float("nan")


def _messages(caught) -> str:
    return " || ".join(str(w.message) for w in caught)


def _capture(fn):
    """Run ``fn`` and return ``(result, joined warning messages)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, _messages(caught)


# =========================================================================
# 1. ActionBiasAnalyzer.analyze_delegation
# =========================================================================


class TestDelegationVerdictsThatCouldNotHaveBeenReached:
    def test_total_segregation_at_three_per_group_is_not_reported_as_no_bias(self):
        """Before: per_target junior and senior each came back p=0.1,
        is_significant=False, overall_chi2=2.667, overall_p_value=0.1025,
        is_significant=False, with no warning at all, for routing in which every
        group-A task went to one agent and every group-B task to another."""
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_delegation(["junior"] * 3, ["senior"] * 3)
        )
        for target in ("junior", "senior"):
            row = result["per_target"][target]
            assert row["is_significant"] is None, row
            assert row["detectable"] is False, row
            assert row["min_attainable_p"] == pytest.approx(0.1), row
        # The MEASUREMENT survives the refusal: the rates are still there.
        assert result["per_target"]["junior"]["rate_a"] == 1.0
        assert result["per_target"]["junior"]["rate_b"] == 0.0
        assert result["is_significant"] is None
        assert result["detectable"] is False
        assert "NOT ASSESSED" in msg
        assert "NOT False" in msg

    def test_a_sparse_table_does_not_publish_an_invalid_asymptotic_finding(self):
        """Before: overall_chi2=13.0, overall_p_value=0.0015034391929775713,
        is_significant=True, on a 2x3 table whose smallest expected cell count is
        0.0769 and whose exact conditional p is 0.0769. The asymptotic
        chi-square does not merely lose power there, it reports the wrong p."""
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_delegation(
                ["junior"] * 6 + ["senior"] * 6, ["external"]
            )
        )
        assert result["overall_p_value"] is None, result
        assert result["is_significant"] is None, result
        assert result["test_used"] is None
        # The descriptive statistic is unchanged; only the verdict is withheld.
        assert result["overall_chi2"] == pytest.approx(13.0)
        assert "0.0769" in msg
        assert "DelegationRoutingAuditor.analyze" in msg

    def test_one_observed_target_is_not_a_measured_absence_of_routing_bias(self):
        """Before: overall_chi2=0.0 and overall_p_value=1.0 for a 2x1 table,
        which has 0 degrees of freedom and no p at all; is_significant=False.
        The per-target Fisher test IS valid here (both rates are 1.0 over 5
        calls each, exact p=1.0, floor 0.0079) and must keep reporting.

        The refusal is None and NOT nan on every field. This dict is published
        verbatim as `actionDistribution` by the pulse agent probe, and the agent
        lane's run_pulse return is not passed through orchestrator._jsonify: with
        nan here, json.dumps(run_pulse(...), allow_nan=False) raised ValueError
        on 2026-09-27, and that nan was the only non-finite float in the payload.
        """
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_delegation(["approve"] * 5, ["approve"] * 5)
        )
        assert result["overall_chi2"] is None, result
        assert result["overall_p_value"] is None
        assert result["is_significant"] is None
        assert "only one routing target" in msg
        approve = result["per_target"]["approve"]
        assert approve["is_significant"] is False, approve
        assert approve["detectable"] is True, approve
        # No non-finite float may leave this method: null is the JSON boundary's
        # value for an unmeasured quantity, nan is refused by strict JSON.
        assert not any(
            isinstance(v, float) and not math.isfinite(v)
            for v in (result["overall_chi2"], result["overall_p_value"], result["min_attainable_p"])
            if v is not None
        ), result

    def test_control_a_real_routing_disparity_at_scale_is_still_reported(self):
        """OVER-CORRECTION CONTROL. 80/20 against 30/70 over 100 decisions per
        group must stay a finding, with no could-not-check anywhere."""
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_delegation(
                ["approve"] * 80 + ["manual"] * 20, ["approve"] * 30 + ["manual"] * 70
            )
        )
        assert result["is_significant"] is True
        assert result["test_used"] == "chi2"
        assert result["overall_p_value"] < 1e-9
        assert result["per_target"]["approve"]["is_significant"] is True
        assert "NOT ASSESSED" not in msg
        assert "COULD NOT CHECK" not in msg

    def test_control_identical_routing_at_scale_is_a_measured_not_significant(self):
        """OVER-CORRECTION CONTROL, the other side: a design WITH power that
        finds nothing must report False, not None. Without this the fix would be
        indistinguishable from a unit that refuses everything."""
        routing = ["approve"] * 80 + ["manual"] * 20
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_delegation(routing, list(routing))
        )
        assert result["is_significant"] is False, result
        assert result["detectable"] is True
        assert result["per_target"]["approve"]["is_significant"] is False
        assert "NOT ASSESSED" not in msg

    def test_the_cochran_threshold_is_not_written_twice_with_two_values(self):
        """action_bias and multi_agent.delegation each name the minimum expected
        cell count. vfairness._not_assessed documents what happens when two
        copies of one number drift apart (a chart requiring ten rows beside an
        SVG requiring one), so the copies are pinned equal here."""
        assert action_bias_module.MIN_EXPECTED_CELL == delegation_module.MIN_EXPECTED_CELL


# =========================================================================
# 2. ToolBiasAuditor
# =========================================================================


class TestToolSelectionVerdictsThatCouldNotHaveBeenReached:
    def test_total_exclusion_at_three_calls_per_group_is_not_reported_as_no_bias(self):
        """Before: approve rate 1.0 vs 0.0 and escalate 0.0 vs 1.0, each with
        p=0.1 and is_significant=False, in silence. 0.1 is the smallest p 3
        against 3 can produce, so the False was not a reading about these
        tools."""
        result, msg = _capture(
            lambda: ToolBiasAuditor().compute_selection_disparity(["approve"] * 3, ["escalate"] * 3)
        )
        for tool in ("approve", "escalate"):
            row = result[tool]
            assert row["is_significant"] is None, row
            assert row["detectable"] is False, row
            assert row["min_attainable_p"] == pytest.approx(0.1), row
        assert result["approve"]["rate_a"] == 1.0
        assert result["approve"]["rate_b"] == 0.0
        assert "NOT ASSESSED" in msg
        assert "NOT False" in msg

    def test_the_typed_result_carries_the_refusal_too(self):
        """A correct measurement no reader can see is the same defect one layer
        up: analyze_tool_calls is the surface most callers use, and before the
        fix its ToolBiasResult.is_significant was False beside a
        disparity_ratio of 0.0, which is total exclusion."""
        results, msg = _capture(
            lambda: ToolBiasAuditor().analyze_tool_calls(
                [{"tool": "approve"}] * 3, [{"tool": "escalate"}] * 3
            )
        )
        by_tool = {r.tool_name: r for r in results}
        assert by_tool["approve"].is_significant is None
        assert by_tool["escalate"].is_significant is None
        # The real measurement is untouched.
        assert by_tool["approve"].disparity_ratio == 0.0
        assert by_tool["approve"].invocation_rate_a == 1.0
        assert "NOT ASSESSED" in msg
        # It survives serialisation, which is how the pulse report reads it.
        assert by_tool["approve"].to_dict()["is_significant"] is None

    def test_control_a_real_tool_disparity_at_scale_is_still_flagged(self):
        """OVER-CORRECTION CONTROL. 0.8 vs 0.3 over 100 calls per group."""
        results, msg = _capture(
            lambda: ToolBiasAuditor().analyze_tool_calls(
                [{"tool": "approve"}] * 80 + [{"tool": "review"}] * 20,
                [{"tool": "approve"}] * 30 + [{"tool": "review"}] * 70,
            )
        )
        by_tool = {r.tool_name: r for r in results}
        assert by_tool["approve"].is_significant is True
        assert by_tool["approve"].p_value < 1e-9
        assert "NOT ASSESSED" not in msg

    def test_control_identical_tool_traces_at_scale_are_a_measured_false(self):
        """OVER-CORRECTION CONTROL, the other side: a design with power that
        finds nothing reports False, never None."""
        traces = [{"tool": "approve"}] * 80 + [{"tool": "review"}] * 20
        results, msg = _capture(lambda: ToolBiasAuditor().analyze_tool_calls(traces, list(traces)))
        for r in results:
            assert r.is_significant is False, r
        assert "NOT ASSESSED" not in msg


# =========================================================================
# 3. MultiAgentRunHarness
# =========================================================================


class TestHarnessDisclosesTheGroupItCouldNotMeasure:
    @staticmethod
    def _nan_group_harness() -> MultiAgentRunHarness:
        h = MultiAgentRunHarness()
        for i in range(10):
            g = i % 2
            h.record_sample(
                group=g,
                component_outputs={"a": NAN if g else 0.5},
                system_output=NAN if g else 0.5,
            )
        return h

    def test_system_bias_now_says_which_group_went_unmeasured(self):
        """Before: system_bias() returned nan and warned about NOTHING, so a
        caller comparing it (`nan > threshold` is False, which reads as a gap
        under the bar) or formatting it could not tell that one of the two
        recorded groups was never measured. The value was already right."""
        value, msg = _capture(self._nan_group_harness().system_bias)
        assert math.isnan(value)
        assert "only 1 of 2 recorded groups had a finite mean" in msg
        assert "could not check" in msg
        assert "MultiAgentRunHarness.system_bias" in msg

    def test_the_disclosure_sits_above_both_accessors(self):
        """as_compositionality_input and system_bias share this helper. A
        warning added to one of them would leave the other silent, which is the
        shape of the DO NOT REMOVE note already in harness.py."""
        out, msg = _capture(self._nan_group_harness().as_compositionality_input)
        assert math.isnan(out["a"])
        assert "as_compositionality_input[a]" in msg
        assert "only 1 of 2 recorded groups had a finite mean" in msg

    def test_control_a_healthy_binary_run_measures_and_stays_silent(self):
        """OVER-CORRECTION CONTROL: no NaN, no warning, exact values."""
        h = MultiAgentRunHarness()
        for i in range(20):
            g = i % 2
            h.record_sample(
                group=g, component_outputs={"a": 0.5 + 0.1 * g}, system_output=0.4 + 0.3 * g
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb = h.system_bias()
            comp = h.as_compositionality_input()
        assert sb == pytest.approx(0.3)
        assert comp["a"] == pytest.approx(0.1)
        assert [w for w in caught if w.category is UserWarning] == []

    def test_record_sample_refuses_an_agent_set_introduced_after_an_empty_one(self):
        """Before: a first sample with component_outputs={} left that dict
        empty, so the agent-set check was skipped forever after. Measured
        2026-09-27: recording {} then {"a": 0.9} was ACCEPTED, leaving
        n_samples=2 against one recorded value for "a", and
        as_compositionality_input then raised a numpy IndexError ("boolean index
        did not match indexed array along axis 0; size of axis is 1 but size of
        corresponding boolean axis is 2") naming neither the sample nor the
        agent."""
        h = MultiAgentRunHarness()
        h.record_sample(group=0, component_outputs={}, system_output=0.1)
        with pytest.raises(ValueError, match="Inconsistent agent set"):
            h.record_sample(group=1, component_outputs={"a": 0.9}, system_output=0.9)

    def test_control_a_consistent_agent_set_is_still_accepted(self):
        """OVER-CORRECTION CONTROL: the guard must not refuse a normal run, nor
        a run that records no components at all on EVERY sample."""
        h = MultiAgentRunHarness()
        for i in range(4):
            h.record_sample(group=i % 2, component_outputs={"a": 0.5}, system_output=0.5)
        assert h.trace.n_samples == 4
        assert len(h.trace.component_outputs["a"]) == 4

        bare = MultiAgentRunHarness()
        for i in range(4):
            bare.record_sample(group=i % 2, component_outputs={}, system_output=0.5)
        assert bare.trace.n_samples == 4
        assert bare.trace.component_outputs == {}


# =========================================================================
# 4. GroupthinkDetector.detect_coalitions
# =========================================================================


class TestCoalitionsOnAMatrixThatExaminedNothing:
    def test_an_empty_matrix_is_not_a_finding_of_independence(self):
        """Before: detect_coalitions(np.zeros((0, 0))) returned [] with no
        warning, byte-identical to the output for a matrix in which every pair
        WAS compared and none agreed."""
        out, msg = _capture(lambda: GroupthinkDetector().detect_coalitions(np.zeros((0, 0))))
        assert out == []
        assert "0 agents" in msg
        assert "could not check" in msg

    def test_control_a_measured_matrix_is_still_grouped_silently(self):
        """OVER-CORRECTION CONTROL: two agents above threshold, one below."""
        matrix = np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]])
        out, msg = _capture(lambda: GroupthinkDetector().detect_coalitions(matrix))
        assert out == [{0, 1}, {2}]
        assert msg == ""

    def test_the_one_pair_being_undefined_still_refuses_both_agents(self):
        """Already correct before this batch (BGL-S2b, 2026-09-17) and pinned
        here: [[1.0, nan], [nan, 1.0]] used to return [{0}, {1}], two
        measured-looking independents, from a matrix whose only real pair was
        undefined."""
        out, msg = _capture(
            lambda: GroupthinkDetector().detect_coalitions(np.array([[1.0, NAN], [NAN, 1.0]]))
        )
        assert out == []
        assert "COULD NOT BE CHECKED" in msg


# =========================================================================
# 5. DelegationRoutingAuditor
# =========================================================================


class TestDelegationRoutingAuditor:
    def test_per_route_disparity_refuses_a_table_with_one_group(self):
        """Before: _per_route_disparity({'junior': {'A': 3}, 'senior':
        {'A': 1}}, ['A']) returned {'junior': 0.0, 'senior': 0.0} silently, and
        0.0 on that scale means both groups reach the route at exactly the same
        rate. analyze() refuses fewer than two groups with a ValueError so this
        never fires from the public entry; it is pinned for the reason
        _cramers_v gives for its own unreachable refusal."""
        auditor = DelegationRoutingAuditor()
        out, msg = _capture(
            lambda: auditor._per_route_disparity({"junior": {"A": 3}, "senior": {"A": 1}}, ["A"])
        )
        assert all(math.isnan(v) for v in out.values()), out
        assert "could not check" in msg
        assert "not 0.0" in msg

    def test_control_the_public_entry_still_measures_per_route_disparity(self):
        """OVER-CORRECTION CONTROL: the reachable path is unchanged."""
        result, _ = _capture(
            lambda: DelegationRoutingAuditor().analyze(
                ["junior"] * 80 + ["senior"] * 20 + ["junior"] * 30 + ["senior"] * 70,
                ["A"] * 100 + ["B"] * 100,
            )
        )
        assert result.per_route_disparity["junior"] == pytest.approx(0.5)
        assert result.is_significant is True

    def test_a_design_without_power_refuses_the_routing_verdict(self):
        """Already correct (2026-09-10) and pinned: 3 decisions per group,
        100 percent segregated, gives Cramer's V = 1.0 with Fisher's floor at
        0.1, so is_significant must be None and never False."""
        result, msg = _capture(
            lambda: DelegationRoutingAuditor().analyze(
                ["junior"] * 3 + ["senior"] * 3, ["A"] * 3 + ["B"] * 3
            )
        )
        assert result.is_significant is None
        assert result.cramers_v == pytest.approx(1.0)
        assert result.detectable is False
        assert "NOT False" in msg


# =========================================================================
# 6. The eight refusals that were already correct.
#
# Each was executed on 2026-09-27 on an input where the quantity does not exist
# and on a measurable input beside it. They are pinned because an observed
# refusal that nothing holds in place is the BGL-B state: a refactor can take it
# away with every gate staying green.
# =========================================================================


class TestRefusalsAlreadyCorrect:
    def test_action_bias_outcomes_refuses_significance_on_one_value_per_group(self):
        """Measured 2026-09-27: p_value=nan, is_significant=None,
        effect_size=nan, and the disparity of 0.6 still reported. `nan < alpha`
        is False, so the earlier code read as a test that found nothing."""
        result, msg = _capture(
            lambda: ActionBiasAnalyzer().analyze_outcomes(
                [{"action": "x", "score": 0.9}], [{"action": "x", "score": 0.3}], "score"
            )
        )
        assert result.is_significant is None
        assert math.isnan(result.p_value)
        assert result.disparity == pytest.approx(0.6)
        assert "not 1.0 and False" in msg

    def test_action_bias_outcomes_control(self):
        rng = np.random.default_rng(1)
        a = [{"action": "approve", "score": float(v)} for v in rng.normal(0.7, 0.05, 40)]
        b = [{"action": "approve", "score": float(v)} for v in rng.normal(0.5, 0.05, 40)]
        result, _ = _capture(lambda: ActionBiasAnalyzer().analyze_outcomes(a, b, "score"))
        assert result.is_significant is True
        assert result.p_value < 0.001

    def test_collusion_refuses_a_non_finite_permutation_null(self):
        """Measured 2026-09-27 with every post-interaction output NaN:
        p_value=nan, is_significant=None, is_collusion=None. Before that fix the
        same input gave p=0.0196 and is_significant=True, byte-identical to the
        genuine-collusion control."""
        groups = np.array([0] * 40 + [1] * 40)
        pre = {"a": np.random.default_rng(7).random(80)}
        post = {"a": np.full(80, NAN)}
        result, msg = _capture(lambda: AdversarialCollusionDetector().analyze(pre, post, groups))
        assert result.is_collusion is None
        assert result.is_significant is None
        assert math.isnan(result.p_value)
        assert "could not check" in msg

    def test_collusion_control_measures_a_real_amplification(self):
        groups = np.array([0] * 40 + [1] * 40)
        rng = np.random.default_rng(7)
        pre = {"a": rng.random(80) * 0.1 + 0.5, "b": rng.random(80) * 0.1 + 0.5}
        post = {k: v.copy() for k, v in pre.items()}
        for arr in post.values():
            arr[groups == 1] += 0.5
        result, _ = _capture(lambda: AdversarialCollusionDetector().analyze(pre, post, groups))
        assert result.is_collusion is True
        assert result.collusion_score == pytest.approx(0.5, abs=1e-9)

    def test_compositionality_refuses_an_unmeasurable_component(self):
        """Measured 2026-09-27 with one NaN component against system_bias=0.9:
        scenario='not_assessed', divergence=nan. Before that fix every `>` and
        `<` against NaN was False and control fell through to
        scenario='consistent': the system behaves in line with its parts."""
        result, msg = _capture(
            lambda: CompositionalityAnalyzer().analyze({"a": NAN, "b": 0.03}, 0.9)
        )
        assert result.scenario == "not_assessed"
        # The literal on the result must BE the library's one name for this
        # state; two spellings of it are how the state stops being searchable.
        assert result.scenario == NOT_ASSESSED
        assert math.isnan(result.divergence)
        assert "NOT 'consistent'" in msg

    def test_compositionality_control(self):
        result, msg = _capture(
            lambda: CompositionalityAnalyzer().analyze({"a": 0.05, "b": 0.03}, 0.15)
        )
        assert result.scenario == "amplification"
        assert msg == ""

    def test_emergent_refuses_when_there_is_nothing_to_compare_against(self):
        """Measured 2026-09-27 on a perfectly separating system with NO
        components: is_emergent=None, amplification_factor=nan. Before that fix
        `max(...) if component_biases else 0.0` made the comparison value zero,
        which is the strongest possible evidence of emergence, and the same call
        returned amplification 1e6 with is_emergent=True and p_value=0.0."""
        groups = np.array([0, 1] * 100)
        system = np.where(groups == 1, 1.0, 0.0)
        result, msg = _capture(lambda: EmergentBiasDetector().analyze({}, system, groups))
        assert result.is_emergent is None
        assert math.isnan(result.amplification_factor)
        assert result.system_bias == pytest.approx(1.0)
        assert "could not check" in msg

    def test_emergent_refuses_when_no_sample_carries_a_system_measurement(self):
        """Measured 2026-09-27 with all 200 system outputs NaN: system_bias=nan,
        is_emergent=None, and the count of excluded samples reported."""
        groups = np.array([0, 1] * 100)
        result, msg = _capture(
            lambda: EmergentBiasDetector().analyze(
                {"a": np.full(200, 0.5)}, np.full(200, NAN), groups
            )
        )
        assert result.is_emergent is None
        assert result.n_samples_unmeasurable == 200
        assert "200 of 200" in msg

    def test_emergent_control(self):
        groups = np.array([0, 1] * 100)
        result, _ = _capture(
            lambda: EmergentBiasDetector().analyze(
                {
                    "a": np.where(groups == 1, 0.55, 0.5),
                    "b": np.where(groups == 1, 0.54, 0.5),
                },
                np.where(groups == 1, 1.0, 0.0),
                groups,
            )
        )
        assert result.is_emergent is True
        assert result.amplification_factor == pytest.approx(20.0, rel=1e-6)

    def test_groupthink_refuses_a_convergence_series_it_could_not_measure(self):
        """Measured 2026-09-27 on three rounds of two agents whose output
        vectors are all zeros (cosine undefined): trajectory [nan, nan, nan],
        has_groupthink=None, echo_chamber_score=nan, is_significant=None, and
        both agents named in unplaced_agents. 0.0 for that cosine used to put
        the MINIMUM of the convergence scale, a measurement of maximal
        disagreement, on a pair nobody could compare."""
        result, msg = _capture(
            lambda: GroupthinkDetector().analyze_convergence(
                [{"a": [0.0, 0.0], "b": [0.0, 0.0]}] * 3
            )
        )
        assert result.has_groupthink is None
        assert result.is_significant is None
        assert math.isnan(result.echo_chamber_score)
        assert result.coalition_structure == []
        assert result.unplaced_agents == ["a", "b"]
        assert result.coalitions_detectable is None
        assert "NOT False" in msg

    def test_groupthink_control_detects_real_convergence(self):
        rounds = [
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            {"a": [0.9, 0.2], "b": [0.2, 0.9]},
            {"a": [0.7, 0.5], "b": [0.5, 0.7]},
            {"a": [0.6, 0.6], "b": [0.58, 0.62]},
            {"a": [0.6, 0.6], "b": [0.6, 0.6]},
        ]
        result, _ = _capture(lambda: GroupthinkDetector().analyze_convergence(rounds))
        assert result.has_groupthink is True
        assert result.echo_chamber_score == pytest.approx(1.0)
        assert result.coalition_structure == [["a", "b"]]

    def test_negotiation_refuses_a_trend_on_too_few_measurable_turns(self):
        """Measured 2026-09-27 on [0.1, nan, 0.4] against a flat zero:
        trend='not_assessed', is_widening=None, tau=nan, and every summary
        statistic nan. Before that fix `tau = 0.0 if isnan(tau)` and
        `p_value = 1.0 if isnan(p_value)` reported trend='stable' for a gap that
        was doubling.

        The mechanism is asserted, not only the verdict. Sabotaging the
        `n_measured < 3` branch alone left this test GREEN on 2026-09-27, because
        the detectability guard further down ALSO refuses (Kendall's floor on 2
        turns is 1.0) and its warning also ends in "NOT 'stable'". Two guards
        over one output is defence in depth and good; a pin that cannot tell
        which of them fired is not a pin on either. tau=nan and
        mean_disparity=nan belong to this branch only: the other one reports
        tau=1.0 and mean_disparity=0.25 over the two turns that remained.
        """
        result, msg = _capture(
            lambda: NegotiationFairnessTracker().analyze([0.1, NAN, 0.4], [0.0, 0.0, 0.0])
        )
        assert result.trend == NOT_ASSESSED
        assert result.is_widening is None
        assert result.n_turns_measured == 2
        assert math.isnan(result.mann_kendall_tau), result
        assert math.isnan(result.mean_disparity), result
        assert math.isnan(result.max_disparity), result
        assert result.trend_detectable is None, result
        assert "Kendall's tau needs at least 3" in msg
        assert "NOT 'stable'" in msg

    def test_negotiation_control_detects_a_widening_gap(self):
        result, _ = _capture(
            lambda: NegotiationFairnessTracker().analyze(
                [100, 95, 90, 85, 80, 75, 70, 65], [100, 99, 98, 97, 96, 95, 94, 93]
            )
        )
        assert result.trend == "widening"
        assert result.is_widening is True
        assert result.trend_detectable is True
