"""BGL-3 batch agents_multi-1: does each unit refuse honestly when nothing can
be measured?

Fourteen units across agents/correspondence.py, agents/pipeline_tracker.py,
agents/rag_bias.py and agents/temporal.py were executed on inputs where the
quantity they claim to report does not exist, and on healthy input beside it.
Three sites needed a change and are pinned first; the rest already refused
correctly and are pinned so the refusals cannot be quietly removed.

Every number in the docstrings below was measured by running the unit, before
and after, on 2026-09-27.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.agents.correspondence import CorrespondenceTester
from vfairness.agents.pipeline_tracker import PipelineTracker
from vfairness.agents.rag_bias import RAGBiasAnalyzer
from vfairness.agents.temporal import TemporalTracker


def _caught(fn):
    """Run fn and return (value, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


# ===========================================================================
# 1. correspondence.create_paired_artifacts: a paired design with no contrast
# ===========================================================================


def test_paired_artifacts_refuse_a_design_that_varies_nothing():
    """BEFORE (2026-09-27): create_paired_artifacts(values=["John Smith",
    "John Smith"]) returned 2 artifacts whose 'name' was 'John Smith' in both,
    with no warning and no error, because the guard counted entries rather than
    DISTINCT values. A correspondence study built on that pair varies nothing,
    so a 0.0 disparity from analyze_outcomes is byte for byte what measured
    equal treatment returns.

    AFTER: ValueError naming the single value supplied.
    """
    tester = CorrespondenceTester()
    with pytest.raises(ValueError, match="At least 2 DISTINCT"):
        tester.create_paired_artifacts(
            base_artifact={"name": "PLACEHOLDER", "experience": 5},
            demographic_field="name",
            values=["John Smith", "John Smith"],
        )


def test_paired_artifacts_control_a_real_contrast_is_still_built():
    """OVER-CORRECTION CONTROL. The documented example still returns 2
    artifacts differing only on 'name', with experience 5 on both."""
    tester = CorrespondenceTester()
    artifacts, messages = _caught(
        lambda: tester.create_paired_artifacts(
            base_artifact={"name": "PLACEHOLDER", "experience": 5},
            demographic_field="name",
            values=["John Smith", "Jamal Washington"],
        )
    )
    assert [a["name"] for a in artifacts] == ["John Smith", "Jamal Washington"]
    assert {a["experience"] for a in artifacts} == {5}
    assert artifacts[0]["_artifact_id"] != artifacts[1]["_artifact_id"]
    assert messages == []


def test_paired_artifacts_control_replicate_arms_are_still_allowed():
    """The refusal is deliberately narrow: only a TOTAL absence of contrast is
    unmeasurable. Repeated values beside a distinct one are replicate arms, a
    normal part of the resume-audit method, and ["A", "A", "B"] still returns
    3 artifacts."""
    tester = CorrespondenceTester()
    artifacts = tester.create_paired_artifacts({"name": "P"}, "name", ["A", "A", "B"])
    assert [a["name"] for a in artifacts] == ["A", "A", "B"]


def test_paired_artifacts_control_the_older_guard_is_intact():
    """A single value still raises the original message, which
    tests/test_agents.py::test_create_paired_artifacts_too_few matches."""
    tester = CorrespondenceTester()
    with pytest.raises(ValueError, match="At least 2 demographic values"):
        tester.create_paired_artifacts({"name": "P"}, "name", ["A"])


# ===========================================================================
# 2. pipeline_tracker: a ranking over the stages that happened to be recorded
# ===========================================================================


def _partial_pipeline():
    """Four configured stages, one recorded."""
    rng = np.random.default_rng(3)
    tracker = PipelineTracker(["retrieval", "reasoning", "tool_selection", "action"])
    tracker.record_stage("retrieval", rng.normal(0.8, 0.1, 40), rng.normal(0.4, 0.1, 40))
    return tracker


def test_bias_source_says_which_configured_stages_were_never_measured():
    """BEFORE (2026-09-27): PipelineTracker(["retrieval", "reasoning",
    "tool_selection", "action"]) with only 'retrieval' recorded returned
    'retrieval' from identify_bias_source with ZERO warnings, so "the stage
    contributing most to overall bias" rested on 1 of 4 declared stages and
    said nothing about the 3 nobody measured.

    AFTER: the same 'retrieval' (it IS the largest contributor among the
    recorded stages) plus a warning naming reasoning, tool_selection and
    action, and stating 3 of 4 and 1 recorded.
    """
    source, messages = _caught(lambda: _partial_pipeline().identify_bias_source())
    assert source == "retrieval"
    joined = " ".join(messages)
    assert "3 of 4 configured stage(s) were never recorded" in joined
    assert "reasoning, tool_selection, action" in joined
    assert "not a finding that they carry no bias" in joined


def test_compute_cumulative_says_the_same_thing_to_its_own_caller():
    """The disclosure lives where the data lives, so a caller that reads the
    per-stage rows directly gets it too. BEFORE: one StageResult returned with
    no warning of any kind."""
    rows, messages = _caught(lambda: _partial_pipeline().compute_cumulative())
    assert [r.stage_name for r in rows] == ["retrieval"]
    assert any("never recorded and are ABSENT" in m for m in messages)


def test_bias_source_control_a_complete_pipeline_warns_about_nothing():
    """OVER-CORRECTION CONTROL. Every configured stage recorded: 'action'
    (means 0.90 vs 0.20, abs_disparity ~0.70) is named over 'retrieval'
    (~0.05), and nothing warns. This is the assertion that would break if the
    new warning fired on a complete pipeline, which is the shape
    tests/test_bgl_stage2_s2g07.py already pins with `== []`.
    """
    rng = np.random.default_rng(11)
    tracker = PipelineTracker(["retrieval", "action"])
    tracker.record_stage("retrieval", rng.normal(0.55, 0.1, 40), rng.normal(0.50, 0.1, 40))
    tracker.record_stage("action", rng.normal(0.90, 0.05, 40), rng.normal(0.20, 0.05, 40))
    source, messages = _caught(tracker.identify_bias_source)
    assert source == "action"
    assert messages == []


def test_record_stage_refuses_an_empty_arm_and_an_unknown_stage():
    """CORRECT already. Two observations cannot be compared against none, and a
    stage outside the declared pipeline is a caller error, so both raise rather
    than record a row that would later read as a measurement."""
    tracker = PipelineTracker(["a"])
    with pytest.raises(ValueError, match="must be non-empty"):
        tracker.record_stage("a", np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match="Unknown stage"):
        tracker.record_stage("b", np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_an_untestable_stage_keeps_its_nan_p_value_and_is_disclosed():
    """CORRECT already, re-pinned because the ranking above depends on it. One
    observation per arm: p_value is nan, NOT 1.0, the disparity of 1.0 is still
    reported beside n_a=1 and n_b=1, and the warning names the stage."""
    tracker = PipelineTracker(["thin", "thick"])
    rng = np.random.default_rng(5)
    tracker.record_stage("thin", np.array([1.0]), np.array([0.0]))
    tracker.record_stage("thick", rng.normal(0.6, 0.1, 40), rng.normal(0.4, 0.1, 40))
    rows, messages = _caught(tracker.compute_cumulative)
    thin = {r.stage_name: r for r in rows}["thin"]
    assert math.isnan(thin.bias_metrics["p_value"]), "a test that did not run must not report 1.0"
    assert thin.bias_metrics["abs_disparity"] == 1.0
    assert (thin.bias_metrics["n_a"], thin.bias_metrics["n_b"]) == (1, 1)
    assert any("p_value is NaN, NOT 1.0" in m for m in messages)


# ===========================================================================
# 3. temporal.compute_trajectory: honest nans, no counts
# ===========================================================================


def _ramp(n=6, blank_at=None):
    tracker = TemporalTracker()
    for turn in range(n):
        if blank_at is not None and turn == blank_at:
            tracker.record_turn(turn, [float("nan")], [0.0])
        else:
            tracker.record_turn(turn, [0.5 + 0.1 * turn], [0.5])
    return tracker


def test_trajectory_says_how_many_turns_could_not_be_measured():
    """BEFORE (2026-09-27): a six-turn ramp with turn 2 blanked returned
    values [0.0, 0.1, nan, 0.3, 0.4, 0.5] and cumulative_drift
    [0.0, 0.1, nan, nan, nan, nan] with ZERO warnings, while
    detect_feedback_loop on the same tracker said "1 of 6 recorded turns had a
    non-finite disparity and were excluded". The values were already honest;
    the counts were the part no reader could reconstruct.

    AFTER: identical values plus a warning stating 1 of 6.
    """
    trajectory, messages = _caught(lambda: _ramp(blank_at=2).compute_trajectory())
    assert [r.turn_number for r in trajectory] == [0, 1, 2, 3, 4, 5]
    assert math.isnan(trajectory[2].value)
    assert trajectory[3].value == pytest.approx(0.3)
    assert trajectory[1].cumulative_drift == pytest.approx(0.1)
    assert math.isnan(trajectory[5].cumulative_drift), "a total with a hole in it is not a total"
    joined = " ".join(messages)
    assert "1 of 6 recorded turns" in joined
    assert "not a finding of no drift" in joined


def test_trajectory_control_a_measured_ramp_warns_about_nothing():
    """OVER-CORRECTION CONTROL. The same ramp with nothing blanked: values
    0.0 to 0.5 in steps of 0.1, cumulative_drift 0.5 at the last turn, and no
    warning."""
    trajectory, messages = _caught(lambda: _ramp().compute_trajectory())
    assert [round(r.value, 6) for r in trajectory] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert trajectory[-1].cumulative_drift == pytest.approx(0.5)
    assert messages == []


def test_record_turn_refuses_an_empty_arm_and_a_negative_turn():
    """CORRECT already. Both raise, so no turn can be recorded that carries no
    comparison."""
    tracker = TemporalTracker()
    with pytest.raises(ValueError, match="must be non-empty"):
        tracker.record_turn(0, [], [1.0])
    with pytest.raises(ValueError, match="turn must be >= 0"):
        tracker.record_turn(-1, [1.0], [1.0])


# ===========================================================================
# 4. refusals that were already correct, pinned so they cannot be removed
# ===========================================================================


def test_four_fifths_refuses_when_nobody_was_selected():
    """CORRECT already. four_fifths_rule(0.0, 0.0) returns ratio=nan and
    adverse_impact=None with a warning, never ratio=1.0 with
    adverse_impact=False, which is what MEASURED parity (0.5, 0.5) returns."""
    result, messages = _caught(lambda: CorrespondenceTester().four_fifths_rule(0.0, 0.0))
    assert math.isnan(result["ratio"])
    assert result["adverse_impact"] is None
    assert any("could not check" in m for m in messages)


def test_four_fifths_control_still_measures_a_real_adverse_impact():
    """OVER-CORRECTION CONTROL. 0.90 vs 0.50 is a ratio of 0.5555... and a
    True adverse-impact finding, with no warning; 0.85 vs 0.80 is 0.9411...
    and False."""
    hit, hit_messages = _caught(lambda: CorrespondenceTester().four_fifths_rule(0.90, 0.50))
    assert hit["ratio"] == pytest.approx(0.50 / 0.90)
    assert hit["adverse_impact"] is True
    assert hit_messages == []
    miss, _ = _caught(lambda: CorrespondenceTester().four_fifths_rule(0.85, 0.80))
    assert miss["adverse_impact"] is False


def test_analyze_outcomes_excludes_and_counts_an_unmeasurable_outcome():
    """CORRECT already. 40 callbacks against 0, with one None in group B:
    disparity stays 1.0, p stays 1.86e-23, is_significant stays True,
    n_unmeasurable is 1 and sample_size reports the 39 that REMAINED, not the
    40 supplied. The defect this closed reported disparity=nan with
    is_significant=False and sample_size 40."""
    result, messages = _caught(
        lambda: CorrespondenceTester().analyze_outcomes([1.0] * 40, [0.0] * 39 + [None])
    )
    assert result.disparity_metric == pytest.approx(1.0)
    assert result.is_significant is True
    assert result.p_value < 1e-20
    assert result.n_unmeasurable == 1
    assert result.sample_size == 39
    assert any("EXCLUDED" in m for m in messages)


def test_analyze_outcomes_refuses_when_a_group_has_nothing_left():
    """CORRECT already. Every outcome in group B unmeasurable: it raises rather
    than compare 40 observations against nothing."""
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="empty after filtering"):
            CorrespondenceTester().analyze_outcomes([1.0] * 40, [None] * 40)


def test_analyze_outcomes_control_a_clean_run_reports_no_exclusions():
    """OVER-CORRECTION CONTROL. 40 vs 0 with everything measured: disparity
    1.0, p=1.86e-23, is_significant True, n_unmeasurable 0, no warning."""
    result, messages = _caught(
        lambda: CorrespondenceTester().analyze_outcomes([1.0] * 40, [0.0] * 40)
    )
    assert result.disparity_metric == pytest.approx(1.0)
    assert result.is_significant is True
    assert result.n_unmeasurable == 0
    assert result.sample_size == 40
    assert messages == []


def test_rag_retrieval_and_output_refuse_when_nothing_exists():
    """CORRECT already. Both empty retrieval sets return nan, not the 0.0 that
    means IDENTICAL retrieval; three blank completions per group return nan,
    not the 0.0 that means identical outputs. Measured before those fixes: both
    returned exactly 0.0, indistinguishable from a genuinely identical pair."""
    analyzer = RAGBiasAnalyzer()
    retrieval, retrieval_messages = _caught(lambda: analyzer.analyze_retrieval([], []))
    assert math.isnan(retrieval)
    assert any("not measured" in m for m in retrieval_messages)

    blank, blank_messages = _caught(lambda: analyzer.analyze_output(["", "", ""], ["", "", ""]))
    assert math.isnan(blank)
    assert any("empty or whitespace" in m for m in blank_messages)

    whitespace, _ = _caught(lambda: analyzer.analyze_output(["   "] * 3, ["\n"] * 3))
    assert math.isnan(whitespace)


def test_rag_control_real_retrieval_and_real_outputs_are_measured():
    """OVER-CORRECTION CONTROL. Disjoint document sets score 1.0 and identical
    ones 0.0; one group answered while the other was not is the finding and
    keeps its 1.0; two different answers score 0.9375 on the trigram metric."""
    analyzer = RAGBiasAnalyzer()
    docs_a = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
    docs_b = [{"id": "d4"}, {"id": "d5"}, {"id": "d6"}]
    assert analyzer.analyze_retrieval(docs_a, docs_b) == pytest.approx(1.0)
    assert analyzer.analyze_retrieval(docs_a, list(docs_a)) == pytest.approx(0.0)
    assert analyzer.analyze_output(["", "", ""], ["approved"]) == pytest.approx(1.0)
    divergent = analyzer.analyze_output(["approved, credit fine"], ["denied, high risk"])
    assert divergent > 0.9


def test_rag_full_analysis_reports_none_not_false_for_an_unmeasured_stage():
    """CORRECT already. A run that retrieved nothing and generated nothing
    reports is_retrieval_biased=None and is_output_biased=None with both
    disparities nan. Measured before that fix: both flags were False, a clean
    bill of health for a run that produced no evidence at all."""
    result, messages = _caught(lambda: RAGBiasAnalyzer().full_analysis([], [], [], [], [], []))
    assert result.is_retrieval_biased is None
    assert result.is_output_biased is None
    assert math.isnan(result.retrieval_disparity)
    assert math.isnan(result.output_disparity)
    assert sum("could not check" in m for m in messages) == 2


def test_rag_full_analysis_control_grades_a_biased_and_a_clean_run():
    """OVER-CORRECTION CONTROL. Disjoint retrieval with divergent outputs
    reports True/True; identical retrieval with identical outputs reports
    False/False and disparities of 0.0, so the flags are still real verdicts
    in both directions."""
    analyzer = RAGBiasAnalyzer()
    docs_a = [{"id": "d1"}, {"id": "d2"}]
    docs_b = [{"id": "d3"}, {"id": "d4"}]
    biased, biased_messages = _caught(
        lambda: analyzer.full_analysis(
            ["q"], ["q"], docs_a, docs_b, ["approved, credit fine"], ["denied, high risk"]
        )
    )
    assert biased.is_retrieval_biased is True
    assert biased.is_output_biased is True
    assert biased_messages == []

    clean, _ = _caught(
        lambda: analyzer.full_analysis(
            ["q"], ["q"], docs_a, list(docs_a), ["same answer"], ["same answer"]
        )
    )
    assert clean.is_retrieval_biased is False
    assert clean.is_output_biased is False
    assert clean.retrieval_disparity == pytest.approx(0.0)
    assert clean.output_disparity == pytest.approx(0.0)


def test_temporal_detectors_refuse_below_three_measurable_turns():
    """CORRECT already, pinned for this batch. Two turns of total disparity:
    detect_feedback_loop returns has_feedback_loop=None with
    trend_direction='not_assessed' and a nan tau and p, detect_drift_cusum
    returns has_drift=None with a nan max_cusum, detect_drift_ewma returns
    has_drift=None with a nan centre line, and each warns "only 2 of 2". The
    defect they closed answered False with trend_direction='stable', tau 0.0
    and p 1.0, the signature of a trend test that ran and found nothing."""
    feedback, feedback_messages = _caught(lambda: _ramp(2).detect_feedback_loop())
    assert feedback["has_feedback_loop"] is None
    assert feedback["trend_direction"] == "not_assessed"
    assert math.isnan(feedback["trend_strength"]) and math.isnan(feedback["p_value"])
    assert feedback["n_turns_measured"] == 2
    assert any("only 2 of 2" in m for m in feedback_messages)

    cusum, cusum_messages = _caught(lambda: _ramp(2).detect_drift_cusum())
    assert cusum["has_drift"] is None
    assert math.isnan(cusum["max_cusum"])
    assert any("NOT False" in m for m in cusum_messages)

    ewma, ewma_messages = _caught(lambda: _ramp(2).detect_drift_ewma())
    assert ewma["has_drift"] is None
    assert math.isnan(ewma["center_line"])
    assert any("could not check" in m for m in ewma_messages)


def test_temporal_detectors_refuse_when_every_turn_is_unmeasurable():
    """CORRECT already. Five recorded turns, none with a finite disparity: all
    three detectors report None and each says "only 0 of 5"."""
    tracker = TemporalTracker()
    for turn in range(5):
        tracker.record_turn(turn, [float("nan")], [0.0])
    for name in ("detect_feedback_loop", "detect_drift_cusum", "detect_drift_ewma"):
        result, messages = _caught(getattr(tracker, name))
        key = "has_feedback_loop" if name == "detect_feedback_loop" else "has_drift"
        assert result[key] is None, name
        assert result["n_turns_unmeasurable"] == 5, name
        assert any("only 0 of 5" in m for m in messages), name


def test_temporal_control_a_widening_ramp_is_still_detected():
    """OVER-CORRECTION CONTROL. Eight turns of strictly widening disparity
    (0.0 to 0.7): detect_feedback_loop reports has_feedback_loop=True,
    trend_direction='increasing', tau 0.99999, p 4.96e-05 with no warning, and
    a flat six-turn trajectory reports False with trend_direction='stable' and
    a MEASURED tau of 0.0. Without this the refusals above could be satisfied
    by a detector that refuses everything."""
    loop, loop_messages = _caught(lambda: _ramp(8).detect_feedback_loop())
    assert loop["has_feedback_loop"] is True
    assert loop["trend_direction"] == "increasing"
    assert loop["p_value"] < 0.05
    assert loop["n_turns_measured"] == 8
    assert loop_messages == []

    flat_tracker = TemporalTracker()
    for turn in range(6):
        flat_tracker.record_turn(turn, [0.7], [0.4])
    flat, flat_messages = _caught(flat_tracker.detect_feedback_loop)
    assert flat["has_feedback_loop"] is False
    assert flat["trend_direction"] == "stable"
    assert flat["n_turns_measured"] == 6
    assert flat_messages == []

    cusum, _ = _caught(lambda: _ramp(12).detect_drift_cusum())
    assert cusum["has_drift"] is False
    assert math.isfinite(cusum["max_cusum"]) and cusum["max_cusum"] > 0
