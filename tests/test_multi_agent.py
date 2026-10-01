"""Tests for vfairness.multi_agent module."""

import numpy as np
import pytest

from vfairness.multi_agent.compositionality import (
    CompositionalityAnalyzer,
    CompositionalityResult,
)
from vfairness.multi_agent.emergent import EmergentBiasDetector, EmergentBiasResult
from vfairness.multi_agent.groupthink import GroupthinkDetector, GroupthinkResult


class TestCompositionalityAnalyzer:
    def test_amplification(self):
        analyzer = CompositionalityAnalyzer()
        result = analyzer.analyze({"agent_1": 0.1, "agent_2": 0.15}, system_bias=0.4)
        assert isinstance(result, CompositionalityResult)
        assert result.scenario == "amplification"
        assert result.divergence > 0

    def test_reduction(self):
        analyzer = CompositionalityAnalyzer()
        result = analyzer.analyze({"agent_1": 0.3, "agent_2": 0.4}, system_bias=0.05)
        assert result.scenario == "reduction"

    def test_consistent(self):
        analyzer = CompositionalityAnalyzer()
        result = analyzer.analyze({"agent_1": 0.1}, system_bias=0.1, threshold=0.05)
        assert result.scenario == "consistent"

    def test_novel_emergence(self):
        analyzer = CompositionalityAnalyzer()
        result = analyzer.analyze(
            {"agent_1": 0.01, "agent_2": 0.02}, system_bias=0.3, threshold=0.05
        )
        assert result.scenario == "novel_emergence"

    def test_empty_components(self):
        analyzer = CompositionalityAnalyzer()
        with pytest.raises(ValueError, match="At least one"):
            analyzer.analyze({}, system_bias=0.1)


class TestGroupthinkDetector:
    def test_convergence_detected(self):
        detector = GroupthinkDetector()
        outputs = [
            {"agent_a": [0.8, 0.2], "agent_b": [0.2, 0.8]},  # round 0: divergent
            {"agent_a": [0.6, 0.4], "agent_b": [0.4, 0.6]},  # round 1: closer
            {"agent_a": [0.51, 0.49], "agent_b": [0.50, 0.50]},  # round 2: nearly identical
        ]
        result = detector.analyze_convergence(outputs)
        assert isinstance(result, GroupthinkResult)
        assert len(result.convergence_trajectory) == 3
        # Convergence should increase over rounds
        assert result.convergence_trajectory[-1] > result.convergence_trajectory[0]

    def test_detect_coalitions(self):
        detector = GroupthinkDetector()
        # Two agents agree strongly, third disagrees
        agreement = np.array(
            [
                [1.0, 0.95, 0.3],
                [0.95, 1.0, 0.3],
                [0.3, 0.3, 1.0],
            ]
        )
        coalitions = detector.detect_coalitions(agreement, threshold=0.9)
        assert len(coalitions) == 2  # {0,1} and {2}

    def test_too_few_rounds(self):
        detector = GroupthinkDetector()
        with pytest.raises(ValueError, match="At least 2 rounds"):
            detector.analyze_convergence([{"a": [1.0]}])


class TestEmergentBiasDetector:
    def test_emergent_bias(self):
        rng = np.random.RandomState(42)
        detector = EmergentBiasDetector()
        groups = np.array([0] * 50 + [1] * 50)
        # Components show little bias
        comp = {
            "a": rng.normal(0.5, 0.1, 100),
            "b": rng.normal(0.5, 0.1, 100),
        }
        # System shows clear group-dependent bias
        sys_out = np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])
        result = detector.analyze(comp, sys_out, groups)
        assert isinstance(result, EmergentBiasResult)
        assert result.is_emergent is True
        assert result.amplification_factor > 1

    def test_no_emergent_bias(self):
        rng = np.random.RandomState(42)
        detector = EmergentBiasDetector()
        groups = np.array([0] * 50 + [1] * 50)
        # Both components and system show similar bias
        comp = {"a": np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])}
        sys_out = np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])
        result = detector.analyze(comp, sys_out, groups)
        assert result.is_emergent is False
        assert result.amplification_factor <= 1.5

    def test_inconsistent_lengths(self):
        detector = EmergentBiasDetector()
        groups = np.array([0, 1, 0, 1])
        comp = {"a": np.array([1.0, 2.0])}  # wrong length
        sys_out = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="must match"):
            detector.analyze(comp, sys_out, groups)

    def test_single_group_raises(self):
        detector = EmergentBiasDetector()
        groups = np.array([0, 0, 0])
        comp = {"a": np.array([1.0, 2.0, 3.0])}
        sys_out = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2 unique"):
            detector.analyze(comp, sys_out, groups)


# ─────────────────────────────────────────────────────────────────────────
# AdversarialCollusionDetector
# ─────────────────────────────────────────────────────────────────────────
from vfairness.multi_agent.collusion import (
    AdversarialCollusionDetector,
    CollusionResult,
)


class TestAdversarialCollusionDetector:
    def _make_groups(self, n=80):
        return np.array([0] * (n // 2) + [1] * (n // 2))

    def test_no_collusion_when_pre_equals_post(self):
        detector = AdversarialCollusionDetector(n_permutations=200, random_seed=7)
        rng = np.random.default_rng(0)
        groups = self._make_groups()
        # Identical pre and post: collusion score ~ 0, not significant.
        pre = {"a": rng.normal(size=80), "b": rng.normal(size=80)}
        post = {"a": pre["a"].copy(), "b": pre["b"].copy()}
        result = detector.analyze(pre, post, groups)
        assert isinstance(result, CollusionResult)
        assert abs(result.collusion_score) < 0.05
        assert not result.is_collusion

    def test_collusion_when_post_bias_inflated(self):
        detector = AdversarialCollusionDetector(n_permutations=300, random_seed=11)
        rng = np.random.default_rng(1)
        groups = self._make_groups(100)
        pre = {
            "a": rng.normal(size=100),
            "b": rng.normal(size=100),
        }
        # Post-interaction: both agents agree to favor group 0 strongly.
        post = {
            "a": rng.normal(size=100) + 1.5 * (groups == 0),
            "b": rng.normal(size=100) + 1.5 * (groups == 0),
        }
        result = detector.analyze(pre, post, groups)
        assert result.collusion_score > 0.5
        assert result.is_collusion
        assert result.p_value < 0.05

    def test_mismatched_agent_sets_raise(self):
        detector = AdversarialCollusionDetector(n_permutations=100)
        groups = self._make_groups(40)
        pre = {"a": np.zeros(40)}
        post = {"a": np.zeros(40), "b": np.zeros(40)}
        with pytest.raises(ValueError, match="same agents"):
            detector.analyze(pre, post, groups)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            AdversarialCollusionDetector(alpha=1.5)

    def test_p_value_never_exactly_zero(self):
        """Regression: Phipson & Smyth (2010) (1+k)/(1+n) correction."""
        detector = AdversarialCollusionDetector(n_permutations=200, random_seed=1)
        rng = np.random.default_rng(2)
        groups = self._make_groups(120)
        # Extreme separation: every null permutation should produce a
        # smaller stat than observed, so the raw count is 0/200 → must
        # be corrected to 1/201, not reported as 0.0.
        pre = {"a": rng.normal(size=120), "b": rng.normal(size=120)}
        post = {
            "a": rng.normal(size=120) + 5.0 * (groups == 0),
            "b": rng.normal(size=120) + 5.0 * (groups == 0),
        }
        result = detector.analyze(pre, post, groups)
        assert result.p_value > 0.0
        assert result.p_value <= 1.0 / 201.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────
# DelegationRoutingAuditor
# ─────────────────────────────────────────────────────────────────────────
from vfairness.multi_agent.delegation import (
    DelegationResult,
    DelegationRoutingAuditor,
)


class TestDelegationRoutingAuditor:
    def test_balanced_routing(self):
        # Same route distribution per group → not significant.
        routes = (["junior", "senior"] * 20) + (["junior", "senior"] * 20)
        demographics = (["A"] * 40) + (["B"] * 40)
        result = DelegationRoutingAuditor().analyze(routes, demographics)
        assert isinstance(result, DelegationResult)
        assert not result.is_significant
        assert result.cramers_v < 0.2

    def test_biased_routing_detected(self):
        # Group A routed entirely to "senior", group B entirely to "junior".
        routes = (["senior"] * 40) + (["junior"] * 40)
        demographics = (["A"] * 40) + (["B"] * 40)
        result = DelegationRoutingAuditor().analyze(routes, demographics)
        assert result.is_significant
        assert result.p_value < 0.001
        # Regression: V must be 1.0 (within float tolerance) for perfect
        # association. Using Yates-corrected chi-square would give 0.975.
        assert result.cramers_v > 0.99
        assert result.test_used in ("fisher", "chi2")

    def test_per_route_disparity_nonzero(self):
        routes = (["senior"] * 30 + ["junior"] * 10) + (["senior"] * 10 + ["junior"] * 30)
        demographics = (["A"] * 40) + (["B"] * 40)
        result = DelegationRoutingAuditor().analyze(routes, demographics)
        # Senior route should show a per-route invocation gap.
        assert result.per_route_disparity["senior"] > 0.4

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            DelegationRoutingAuditor().analyze(["x", "y"], ["A"])


# ─────────────────────────────────────────────────────────────────────────
# NegotiationFairnessTracker
# ─────────────────────────────────────────────────────────────────────────
from vfairness.multi_agent.negotiation import (
    NegotiationFairnessTracker,
    NegotiationResult,
)


class TestNegotiationFairnessTracker:
    def test_stable_when_parallel(self):
        # Parallel decreases: per-turn disparity is constant.
        a = [100, 95, 90, 85, 80]
        b = [102, 97, 92, 87, 82]
        result = NegotiationFairnessTracker().analyze(a, b)
        assert isinstance(result, NegotiationResult)
        assert result.trend == "stable"

    def test_widening_disparity_detected(self):
        # Group A concedes harder each turn while group B holds firm.
        a = [100, 90, 80, 70, 60, 50, 40, 30]
        b = [100, 99, 98, 97, 96, 95, 94, 93]
        result = NegotiationFairnessTracker().analyze(a, b)
        assert result.is_widening
        assert result.mann_kendall_tau < 0  # disparity = a - b decreases over turns
        # final - initial should be strongly negative (disparity moved against A).
        assert result.final_minus_initial < -30

    def test_requires_at_least_3_turns(self):
        with pytest.raises(ValueError, match="3 turns"):
            NegotiationFairnessTracker().analyze([1, 2], [1, 2])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of turns"):
            NegotiationFairnessTracker().analyze([1, 2, 3, 4], [1, 2, 3])


# ─────────────────────────────────────────────────────────────────────────
# MultiAgentRunHarness
# ─────────────────────────────────────────────────────────────────────────
from vfairness.multi_agent.harness import HarnessTrace, MultiAgentRunHarness


class TestMultiAgentRunHarness:
    def test_record_and_aggregate(self):
        with MultiAgentRunHarness() as h:
            for i in range(20):
                h.record_sample(
                    group=i % 2,
                    component_outputs={"a": float(i), "b": float(i) * 2},
                    system_output=float(i) * 3,
                )
        assert isinstance(h.trace, HarnessTrace)
        assert h.trace.n_samples == 20
        comp, sysout, groups = h.as_emergent_inputs()
        assert set(comp) == {"a", "b"}
        assert len(sysout) == 20
        assert len(groups) == 20

    def test_consistent_agent_set_enforced(self):
        h = MultiAgentRunHarness()
        h.record_sample(group=0, component_outputs={"a": 1.0, "b": 2.0}, system_output=1.0)
        with pytest.raises(ValueError, match="Inconsistent agent set"):
            h.record_sample(group=1, component_outputs={"a": 1.0}, system_output=1.0)

    def test_routing_and_negotiation_capture(self):
        h = MultiAgentRunHarness()
        for r, d in [("x", "A"), ("y", "B"), ("x", "A"), ("y", "B")]:
            h.record_routing(r, d)
        for _ in range(5):
            h.record_turn(1.0, 2.0)
        routes, demos = h.as_delegation_inputs()
        assert routes == ["x", "y", "x", "y"]
        assert demos == ["A", "B", "A", "B"]
        a, b = h.as_negotiation_inputs()
        assert len(a) == len(b) == 5

    def test_collusion_requires_pre_post(self):
        h = MultiAgentRunHarness()
        h.record_sample(group=0, component_outputs={"a": 1.0}, system_output=1.0)
        with pytest.raises(ValueError, match="pre_interaction and post_interaction"):
            h.as_collusion_inputs()

    def test_pre_interaction_agent_set_must_match(self):
        """Regression: silent array-length divergence when pre/post agent
        set differs from component_outputs."""
        h = MultiAgentRunHarness()
        with pytest.raises(ValueError, match="pre_interaction must cover"):
            h.record_sample(
                group=0,
                component_outputs={"a": 1.0, "b": 2.0},
                system_output=1.0,
                pre_interaction={"a": 1.0},  # missing 'b'
                post_interaction={"a": 1.0, "b": 2.0},
            )

    def test_cannot_start_pre_interaction_partway(self):
        """Regression: introducing pre_interaction after prior samples
        without it would diverge array lengths."""
        h = MultiAgentRunHarness()
        h.record_sample(group=0, component_outputs={"a": 1.0}, system_output=1.0)
        with pytest.raises(ValueError, match="cannot introduce it partway"):
            h.record_sample(
                group=1,
                component_outputs={"a": 2.0},
                system_output=2.0,
                pre_interaction={"a": 1.5},
                post_interaction={"a": 2.5},
            )

    def test_cannot_stop_pre_interaction_partway(self):
        """Regression: dropping pre_interaction after providing it on
        earlier samples diverges array lengths."""
        h = MultiAgentRunHarness()
        h.record_sample(
            group=0,
            component_outputs={"a": 1.0},
            system_output=1.0,
            pre_interaction={"a": 1.5},
            post_interaction={"a": 2.5},
        )
        with pytest.raises(ValueError, match="once started"):
            h.record_sample(
                group=1,
                component_outputs={"a": 2.0},
                system_output=2.0,
            )
