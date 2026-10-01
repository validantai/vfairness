"""Tests for vfairness.agents module."""

import numpy as np
import pytest

from vfairness.agents.action_bias import ActionBiasAnalyzer, ActionBiasResult
from vfairness.agents.correspondence import CorrespondenceResult, CorrespondenceTester
from vfairness.agents.pipeline_tracker import PipelineTracker, StageResult
from vfairness.agents.temporal import TemporalTracker, TrajectoryResult
from vfairness.agents.tool_bias import ToolBiasAuditor, ToolBiasResult


class TestCorrespondenceTester:
    def test_four_fifths_rule_pass(self):
        tester = CorrespondenceTester()
        result = tester.four_fifths_rule(0.85, 0.80)
        # Exact ratio, not just the pass/fail band: 0.80 / 0.85 = 16/17.
        assert result["ratio"] == pytest.approx(0.80 / 0.85)
        assert result["adverse_impact"] is False
        assert result["favored_group"] == "group_a"

    def test_four_fifths_rule_fail(self):
        tester = CorrespondenceTester()
        result = tester.four_fifths_rule(0.90, 0.50)
        assert result["adverse_impact"] is True

    def test_analyze_numeric_outcomes(self):
        tester = CorrespondenceTester()
        rng = np.random.RandomState(42)
        outcomes_a = rng.normal(100, 10, 50).tolist()
        outcomes_b = rng.normal(80, 10, 50).tolist()
        result = tester.analyze_outcomes(outcomes_a, outcomes_b)
        assert isinstance(result, CorrespondenceResult)
        assert bool(result.is_significant) is True  # Clear difference

    def test_analyze_binary_outcomes(self):
        tester = CorrespondenceTester()
        outcomes_a = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
        outcomes_b = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        result = tester.analyze_outcomes(outcomes_a, outcomes_b)
        # Exact detection outcome: 8/10 vs 2/10 callback rates, a 0.6 gap,
        # significant at alpha=0.05 (Fisher exact p ~ 0.023).
        assert result.outcome_a == pytest.approx(0.8)
        assert result.outcome_b == pytest.approx(0.2)
        assert result.disparity_metric == pytest.approx(0.6)
        assert bool(result.is_significant) is True
        assert result.p_value < 0.05

    def test_create_paired_artifacts(self):
        tester = CorrespondenceTester()
        artifacts = tester.create_paired_artifacts(
            base_artifact={"name": "PLACEHOLDER", "experience": 5},
            demographic_field="name",
            values=["Alice", "Bob"],
        )
        assert len(artifacts) == 2
        assert artifacts[0]["name"] == "Alice"
        assert artifacts[1]["name"] == "Bob"
        assert "_artifact_id" in artifacts[0]

    def test_create_paired_artifacts_too_few(self):
        tester = CorrespondenceTester()
        with pytest.raises(ValueError, match="At least 2"):
            tester.create_paired_artifacts(
                base_artifact={"name": "X"},
                demographic_field="name",
                values=["Alice"],
            )

    def test_analyze_outcomes_empty(self):
        tester = CorrespondenceTester()
        with pytest.raises(ValueError, match="must not be empty"):
            tester.analyze_outcomes([], [1, 2, 3])


class TestToolBiasAuditor:
    def test_analyze_tool_calls_detects_planted_bias(self):
        # n=100 per group so the 0.8 vs 0.3 approval-rate gap is decisively
        # significant, then verify the detection OUTCOME, not just shapes.
        auditor = ToolBiasAuditor()
        traces_a = [{"tool": "approve"}] * 80 + [{"tool": "review"}] * 20
        traces_b = [{"tool": "approve"}] * 30 + [{"tool": "review"}] * 70
        results = auditor.analyze_tool_calls(traces_a, traces_b)
        assert len(results) == 2
        assert all(isinstance(r, ToolBiasResult) for r in results)
        by_tool = {r.tool_name: r for r in results}
        approve = by_tool["approve"]
        assert approve.invocation_rate_a == pytest.approx(0.8)
        assert approve.invocation_rate_b == pytest.approx(0.3)
        assert approve.disparity_ratio == pytest.approx(0.3 / 0.8)
        assert bool(approve.is_significant) is True
        assert approve.p_value < 0.001
        review = by_tool["review"]
        assert review.invocation_rate_a == pytest.approx(0.2)
        assert review.invocation_rate_b == pytest.approx(0.7)
        assert bool(review.is_significant) is True

    def test_analyze_tool_calls_no_false_positive_on_identical_traces(self):
        # Unbiased contrast: identical distributions must NOT be flagged.
        auditor = ToolBiasAuditor()
        traces = [{"tool": "approve"}] * 80 + [{"tool": "review"}] * 20
        results = auditor.analyze_tool_calls(traces, list(traces))
        for r in results:
            assert r.invocation_rate_a == pytest.approx(r.invocation_rate_b)
            assert r.disparity_ratio == pytest.approx(1.0)
            # `is False`, not `bool(...) is False`. At 100 calls per side the test
            # HAS the power to detect a difference, so a real False is available and
            # is the only right answer. The bool() form also accepts None, which
            # would let a could-not-check pass as a clean bill here, and that is the
            # one thing this control exists to rule out.
            assert r.is_significant is False

    def test_compute_selection_disparity(self):
        auditor = ToolBiasAuditor()
        tools_a = ["search", "search", "compute"]
        tools_b = ["compute", "compute", "compute"]
        result = auditor.compute_selection_disparity(tools_a, tools_b)
        # Exact per-tool rates: search 2/3 vs 0, compute 1/3 vs 1.
        assert result["search"]["rate_a"] == pytest.approx(2 / 3)
        assert result["search"]["rate_b"] == pytest.approx(0.0)
        assert result["compute"]["rate_a"] == pytest.approx(1 / 3)
        assert result["compute"]["rate_b"] == pytest.approx(1.0)
        # n=3 per group cannot reach significance, and that is a statement about
        # the DESIGN, not about these tools. Fisher's smallest attainable p at 3
        # against 3 is 0.1, so no data of this shape could ever clear 0.05.
        #
        # This asserted `is False` until 2026-09-27, with the comment "a flag here
        # would be a false positive from the small-sample path". The first half was
        # right and the second was the defect: a test with no power does not get to
        # return the clean answer. False here says "we looked and these tools are
        # fine", over a comparison that could not have come out any other way, and
        # the rates beside it (2/3 against 0.0) are the most extreme separation the
        # sample allows. So None, which is neither a flag nor a clean bill, and the
        # rates above still stand as real measurements.
        assert result["search"]["is_significant"] is None
        assert result["compute"]["is_significant"] is None

    def test_empty_traces(self):
        auditor = ToolBiasAuditor()
        with pytest.raises(ValueError, match="must not be empty"):
            auditor.analyze_tool_calls([], [{"tool": "x"}])


class TestPipelineTracker:
    def test_compute_cumulative(self):
        tracker = PipelineTracker(["retrieval", "reasoning", "action"])
        rng = np.random.RandomState(42)
        tracker.record_stage("retrieval", rng.normal(0.5, 0.1, 30), rng.normal(0.6, 0.1, 30))
        tracker.record_stage("reasoning", rng.normal(0.5, 0.1, 30), rng.normal(0.55, 0.1, 30))
        tracker.record_stage("action", rng.normal(0.5, 0.1, 30), rng.normal(0.7, 0.1, 30))
        results = tracker.compute_cumulative()
        assert len(results) == 3
        assert all(isinstance(r, StageResult) for r in results)
        # All three stages plant a positive B-shift, so absolute disparity
        # accumulates: cumulative bias must be monotone nondecreasing and
        # end well above the first stage alone.
        cum = [r.cumulative_bias for r in results]
        assert cum == sorted(cum)
        assert cum[-1] > cum[0] > 0
        # The final stage plants the largest shift (0.2 vs 0.1 / 0.05).
        assert results[-1].bias_metrics["abs_disparity"] == max(
            r.bias_metrics["abs_disparity"] for r in results
        )

    def test_identify_bias_source(self):
        tracker = PipelineTracker(["retrieval", "reasoning"])
        rng = np.random.RandomState(42)
        tracker.record_stage("retrieval", rng.normal(0.5, 0.1, 50), rng.normal(0.9, 0.1, 50))
        tracker.record_stage("reasoning", rng.normal(0.5, 0.1, 50), rng.normal(0.51, 0.1, 50))
        source = tracker.identify_bias_source()
        assert source == "retrieval"

    def test_unknown_stage(self):
        tracker = PipelineTracker(["retrieval"])
        with pytest.raises(ValueError, match="Unknown stage"):
            tracker.record_stage("nonexistent", np.array([1.0]), np.array([2.0]))

    def test_no_data_raises(self):
        tracker = PipelineTracker(["retrieval"])
        with pytest.raises(RuntimeError, match="No stage data"):
            tracker.identify_bias_source()

    def test_empty_stages(self):
        with pytest.raises(ValueError, match="At least one stage"):
            PipelineTracker([])


class TestTemporalTracker:
    def test_detect_drift(self):
        tracker = TemporalTracker()
        rng = np.random.RandomState(42)
        for turn in range(10):
            shift = turn * 0.05
            tracker.record_turn(turn, rng.normal(0.5, 0.1, 20), rng.normal(0.5 + shift, 0.1, 20))
        assert tracker.detect_drift(threshold=0.1) is True

    def test_no_drift(self):
        tracker = TemporalTracker()
        rng = np.random.RandomState(42)
        for turn in range(5):
            tracker.record_turn(turn, rng.normal(0.5, 0.1, 20), rng.normal(0.5, 0.1, 20))
        assert tracker.detect_drift(threshold=0.3) is False

    def test_compute_trajectory(self):
        tracker = TemporalTracker()
        tracker.record_turn(0, [0.8, 0.7], [0.6, 0.5])
        tracker.record_turn(1, [0.9, 0.8], [0.5, 0.4])
        trajectory = tracker.compute_trajectory()
        assert len(trajectory) == 2
        assert all(isinstance(t, TrajectoryResult) for t in trajectory)
        assert trajectory[0].cumulative_drift == 0.0

    def test_detect_feedback_loop(self):
        tracker = TemporalTracker()
        for turn in range(6):
            shift = turn * 0.1
            tracker.record_turn(turn, [0.5], [0.5 + shift])
        result = tracker.detect_feedback_loop()
        # A strictly monotone disparity ramp IS a feedback loop: Kendall
        # tau = 1 over 6 turns, p ~ 0.0028. Key presence alone proved nothing.
        assert result["has_feedback_loop"] is True
        assert result["trend_direction"] == "increasing"
        assert result["trend_strength"] == pytest.approx(1.0)
        assert result["p_value"] < 0.05

    def test_detect_feedback_loop_stable_disparity_is_not_flagged(self):
        # Unbiased contrast: constant disparity must not read as a loop.
        tracker = TemporalTracker()
        for turn in range(6):
            tracker.record_turn(turn, [0.5], [0.5])
        result = tracker.detect_feedback_loop()
        assert result["has_feedback_loop"] is False
        assert result["trend_direction"] == "stable"

    def test_empty_trajectory(self):
        import warnings

        tracker = TemporalTracker()
        trajectory = tracker.compute_trajectory()
        assert trajectory == []
        # Was `is False`. Drift is a change BETWEEN observations, so an empty
        # trajectory has none to measure, and False is the verdict "no drift
        # exceeds the threshold". The mechanical sweep flagged the shape; this
        # test had pinned it.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert tracker.detect_drift() is None
        assert caught


class TestActionBiasAnalyzer:
    def test_analyze_outcomes(self):
        analyzer = ActionBiasAnalyzer()
        actions_a = [{"action": "approve", "score": s} for s in [0.9, 0.85, 0.88, 0.92]]
        actions_b = [{"action": "review", "score": s} for s in [0.5, 0.55, 0.48, 0.52]]
        result = analyzer.analyze_outcomes(actions_a, actions_b, "score")
        assert isinstance(result, ActionBiasResult)
        assert result.disparity > 0
        assert result.outcome_a > result.outcome_b

    def test_analyze_delegation_detects_planted_bias(self):
        # n=100 per group so 0.8 vs 0.3 approval routing is decisively
        # significant; verify detection outcomes, not just key presence.
        analyzer = ActionBiasAnalyzer()
        routing_a = ["approve"] * 80 + ["manual_review"] * 20
        routing_b = ["approve"] * 30 + ["manual_review"] * 70
        result = analyzer.analyze_delegation(routing_a, routing_b)
        approve = result["per_target"]["approve"]
        assert approve["rate_a"] == pytest.approx(0.8)
        assert approve["rate_b"] == pytest.approx(0.3)
        assert approve["is_significant"] is True
        assert result["is_significant"] is True
        assert result["overall_p_value"] < 0.001

    def test_analyze_delegation_no_false_positive_on_identical_routing(self):
        analyzer = ActionBiasAnalyzer()
        routing = ["approve"] * 80 + ["manual_review"] * 20
        result = analyzer.analyze_delegation(routing, list(routing))
        assert result["is_significant"] is False
        approve = result["per_target"]["approve"]
        assert approve["rate_a"] == pytest.approx(approve["rate_b"])

    def test_empty_actions(self):
        analyzer = ActionBiasAnalyzer()
        with pytest.raises(ValueError, match="must not be empty"):
            analyzer.analyze_outcomes([], [{"action": "x", "score": 1}], "score")
