"""BGL-5 pins for the six grades the BGL-4 audit OVERTURNED in batches
A-agents_multi-1 and A-agents_multi-2.

Every test here was written beside the fix, executed against it, and then
SABOTAGED: the defect was reintroduced in the source, the named test was shown
to go red, and the source was restored and compared byte for byte with ``diff``.
The sabotage output is recorded in /tmp/claude-501/bgl/fix/fix-A-agents_multi-1
.json and fix-A-agents_multi-2.json.

Each unit gets BOTH directions, because a refusal test on its own is satisfied by
a detector that refuses everything (that is precisely how
``detect_drift_ewma``'s previous pin stayed green while the method published
drift on a series that never moved):

1. ``TemporalTracker.detect_drift_ewma`` published has_drift=True with EVERY
   turn as a drift point on a CONSTANT disparity series, because
   ``np.std(ddof=1)`` is 0 there, so the control limits had zero width, and the
   recursion landed one ulp off the centre. Its sibling detect_drift_cusum
   guards exactly that with ``sigma <= 1e-12``; ewma had no such branch, and the
   verdict reached the published pulse payload as driftDetected TRUE.
2. ``TemporalTracker.detect_drift`` answered the verdict False from a
   non-finite threshold and True from a negative one: the BGL-S2 fix guarded the
   DATA side of ``abs(final - initial) > threshold`` and not the threshold side.
3. ``GroupthinkDetector.analyze_convergence`` answered a two-state bool at 2
   rounds, its documented minimum, where the same shape at 3 rounds is refused
   with None, so total agreement (echo 1.0) and total diversity (echo 0.0) came
   back identical.
4. ``GroupthinkDetector.detect_coalitions`` accepted a threshold no agreement on
   its [0, 1] scale can clear and returned one singleton per agent, byte for
   byte what a measured absence of coalitions returns.
5. ``CompositionalityAnalyzer.analyze`` validated aggregation_method and not
   ``threshold``, so a non-finite tolerance reported 'consistent' on a
   divergence of 0.9 and a negative one reported 'amplification' on a divergence
   of exactly 0.0.
6. ``EmergentBiasDetector.analyze`` checked ``len(unique_groups) < 2`` and never
   more than two, so a three-group run whose whole bias sat in the third
   reported system_bias 0.0 and is_emergent False while claiming all 180 samples
   were measured. Its second named refusal was also unpinned: the test named as
   evidence for the system-bias guard stayed green when that guard was DELETED,
   because the branch below it refuses with the same fields. The last test in
   this file pins the DIAGNOSIS only that branch produces.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.agents.temporal import TemporalTracker
from vfairness.multi_agent.compositionality import CompositionalityAnalyzer
from vfairness.multi_agent.emergent import EmergentBiasDetector
from vfairness.multi_agent.groupthink import GroupthinkDetector

NAN = float("nan")
INF = float("inf")


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _tracker(series):
    """A tracker whose per-turn disparity is exactly each element of series."""
    tracker = TemporalTracker()
    for turn, value in enumerate(series):
        tracker.record_turn(turn, [value], [0.0])
    return tracker


# =========================================================================
# 1. TemporalTracker.detect_drift_ewma
# =========================================================================


class TestEwmaOnAChartWithNoWidth:
    def test_a_constant_trajectory_is_a_measured_absence_of_drift(self):
        """Before (2026-09-27): five turns of an identical 0.005 disparity gave
        has_drift True, drift_points [0, 1, 2, 3, 4], ewma_values
        [0.005000000000000001] x 5, center_line == upper_limit == lower_limit
        == 0.005, n_turns_unmeasurable 0 and ZERO warnings, while the same
        tracker's cumulative_drift was [0.0] x 5 and detect_drift_cusum said
        False. After: has_drift False, drift_points [], the EWMA sitting exactly
        on the centre line."""
        tracker = _tracker([0.005] * 5)
        assert [r.value for r in tracker.compute_trajectory()] == [0.005] * 5

        ewma, messages = _caught(tracker.detect_drift_ewma)
        assert ewma["has_drift"] is False
        assert ewma["drift_points"] == []
        assert ewma["center_line"] == pytest.approx(0.005)
        assert ewma["upper_limit"] == ewma["lower_limit"] == ewma["center_line"]
        assert ewma["ewma_values"] == [ewma["center_line"]] * 5
        assert ewma["n_turns_measured"] == 5
        assert ewma["n_turns_unmeasurable"] == 0
        assert messages == []

        # The same answer its sibling gives, which is the point: one input, one
        # reading, from the two detectors the payload publishes side by side.
        cusum, _ = _caught(tracker.detect_drift_cusum)
        assert cusum["has_drift"] is False
        assert cusum["max_cusum"] == 0.0

    def test_a_constant_rate_gap_is_not_drift_either(self):
        """The same defect on an ordinary fairness input rather than a small
        number: 9 of 10 approvals against 0 of 10, identical in all five
        windows. Before: has_drift True with drift_points [0, 1, 2, 3, 4] at a
        centre of 0.9. After: False, with the gap itself still reported as
        0.9, because a gap that never moved is a constant gap and not a drift.
        """
        tracker = TemporalTracker()
        for turn in range(5):
            tracker.record_turn(turn, [1.0] * 9 + [0.0], [0.0] * 10)
        ewma, messages = _caught(tracker.detect_drift_ewma)
        assert ewma["has_drift"] is False
        assert ewma["drift_points"] == []
        assert ewma["center_line"] == pytest.approx(0.9)
        assert messages == []

    def test_control_a_real_excursion_is_still_flagged_with_its_turn_number(self):
        """OVER-CORRECTION CONTROL, the one the previous pin lacked: forcing
        this method to refuse (or to answer False for) every input left that pin
        at '3 passed'. Fifteen turns at 0.05 followed by three at 0.9: the chart
        is drawn, the EWMA leaves the upper limit, and the RECORDED TURN NUMBER
        17 is reported. Exact measured values, so a change of mechanism moves
        them."""
        ewma, messages = _caught(lambda: _tracker([0.05] * 15 + [0.9] * 3).detect_drift_ewma())
        assert ewma["has_drift"] is True
        assert ewma["drift_points"] == [17]
        assert ewma["center_line"] == pytest.approx(0.19166666666666665)
        assert ewma["upper_limit"] == pytest.approx(0.6289880587800643)
        assert ewma["lower_limit"] == pytest.approx(-0.2456547254467309)
        assert ewma["ewma_values"][-1] == pytest.approx(0.6482444852941432)
        assert ewma["n_turns_measured"] == 18
        assert messages == []

    def test_control_a_varying_trajectory_inside_its_limits_is_a_measured_false(self):
        """OVER-CORRECTION CONTROL, the other side: a series that MOVES but
        stays in control must keep its real chart, with limits of non-zero
        width. Without this, returning the centre line for everything would
        pass the constant-series test above."""
        ewma, messages = _caught(
            lambda: _tracker([0.30, 0.31, 0.29, 0.30, 0.31, 0.29]).detect_drift_ewma()
        )
        assert ewma["has_drift"] is False
        assert ewma["drift_points"] == []
        assert ewma["center_line"] == pytest.approx(0.3)
        assert ewma["upper_limit"] == pytest.approx(0.312)
        assert ewma["lower_limit"] == pytest.approx(0.288)
        assert ewma["upper_limit"] > ewma["lower_limit"]
        assert ewma["ewma_values"][1] == pytest.approx(0.30333333333333334)
        assert messages == []

    def test_the_refusal_below_three_turns_still_refuses(self):
        """The branch the previous pin DID cover, kept: two recorded turns have
        no centre line and no limits, so the answer is None with a nan chart and
        a warning, never False."""
        ewma, messages = _caught(lambda: _tracker([0.0, 0.1]).detect_drift_ewma())
        assert ewma["has_drift"] is None
        assert math.isnan(ewma["center_line"])
        assert math.isnan(ewma["upper_limit"]) and math.isnan(ewma["lower_limit"])
        assert ewma["ewma_values"] == []
        assert any("only 2 of 2" in m for m in messages)

    def test_the_constant_branch_still_discloses_the_turns_it_excluded(self):
        """The new branch returns EARLY, so the exclusion disclosure above it has
        to stay above it. Five recorded turns, two of them unmeasurable, the
        other three an identical 0.005: has_drift False on the three that
        remain, WITH the warning and the counts, not a silent flat chart."""
        ewma, messages = _caught(
            lambda: _tracker([0.005, NAN, 0.005, NAN, 0.005]).detect_drift_ewma()
        )
        assert ewma["has_drift"] is False
        assert ewma["n_turns_measured"] == 3
        assert ewma["n_turns_unmeasurable"] == 2
        assert ewma["ewma_values"] == [pytest.approx(0.005)] * 3
        assert any("2 of 5 recorded turns had a non-finite metric value" in m for m in messages)

    def test_the_published_pulse_payload_no_longer_reports_drift(self):
        """CONSUMER. operations/pulse/agent_probe._temporal_section sets
        driftDetected True when ANY detector says True, so the EWMA false
        positive was published. Before: perWindow [0.9] x 5 with
        ewma {'hasDrift': True, 'driftPoints': [0, 1, 2, 3, 4]}, driftDetected
        TRUE and detectorsFailed empty. After: driftDetected False, no finding,
        and the 0.9 gap still reported window by window."""
        from vfairness.operations.pulse.agent_probe import _temporal_section

        rows = []
        stamp = 0
        for _window in range(5):
            for i in range(10):
                rows.append(
                    {
                        "timestamp": stamp,
                        "group": "A",
                        "action": "approve" if i < 9 else "escalate",
                    }
                )
                stamp += 1
            for _i in range(10):
                rows.append({"timestamp": stamp, "group": "B", "action": "escalate"})
                stamp += 1
        frame = pd.DataFrame(rows)

        (section, findings), _ = _caught(
            lambda: _temporal_section(frame, frame["group"], ["A", "B"], "action", [])
        )
        assert [w["disparity"] for w in section["perWindow"]] == [0.9] * 5
        assert section["ewma"] == {"hasDrift": False, "driftPoints": []}
        assert section["driftDetected"] is False
        assert section["detectorsFailed"] == []
        assert [f for f in findings if f["type"] == "agent_temporal_drift"] == []


# =========================================================================
# 2. TemporalTracker.detect_drift
# =========================================================================


class TestDetectDriftThresholdIsAnOperandToo:
    RAMP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.parametrize("threshold", [NAN, INF, -1.0])
    def test_a_threshold_that_cannot_decide_is_a_could_not_check(self, threshold):
        """Before (2026-09-27) on this ramp, whose first and last measurable
        disparities are 0.0 and 0.5: detect_drift(nan) returned False with ZERO
        warnings, detect_drift(inf) the same, and detect_drift(-1.0) returned
        True for the ramp AND for a flat trajectory, because abs(...) cannot be
        below a negative number. After: None with a warning naming the
        threshold."""
        verdict, messages = _caught(lambda: _tracker(self.RAMP).detect_drift(threshold))
        assert verdict is None
        assert any(f"threshold={threshold!r}" in m for m in messages)
        assert any("NOT False" in m and "NOT True" in m for m in messages)

    def test_control_a_measured_threshold_still_measures_both_verdicts(self):
        """OVER-CORRECTION CONTROL. The same ramp at 0.1 is True and at 0.6 is
        False, both silently, and a flat trajectory at 0.3 is a measured False.
        A threshold of exactly 0 is a legal magnitude and still measures."""
        ramp = _tracker(self.RAMP)
        assert _caught(lambda: ramp.detect_drift(0.1)) == (True, [])
        assert _caught(lambda: ramp.detect_drift(0.6)) == (False, [])
        assert _caught(lambda: ramp.detect_drift(0.0)) == (True, [])

        flat = _tracker([0.3] * 6)
        assert _caught(lambda: flat.detect_drift(0.3)) == (False, [])

    def test_the_data_side_refusal_is_untouched(self):
        """The BGL-S2 guard this fix sits beside: one measurable turn is still
        None, and an excluded turn is still counted in a warning."""
        verdict, messages = _caught(lambda: _tracker([0.4, NAN, NAN]).detect_drift(0.1))
        assert verdict is None
        assert any("only 1 of 3" in m for m in messages)

        verdict, messages = _caught(
            lambda: _tracker([0.0, 0.1, 0.2, 0.3, 0.4, NAN]).detect_drift(0.1)
        )
        assert verdict is True
        assert any("1 of 6" in m for m in messages)


# =========================================================================
# 3. GroupthinkDetector.analyze_convergence at 2 rounds
# =========================================================================


class TestTwoRoundsIsNotATrendTest:
    IDENTICAL = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    ORTHOGONAL = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    def test_total_agreement_and_total_diversity_are_no_longer_the_same_verdict(self):
        """Before (2026-09-27): two rounds of TOTAL agreement gave
        echo_chamber_score 1.0, the strongest echo chamber on this scale, with
        has_groupthink FALSE and no warning naming that field, and two rounds of
        TOTAL diversity gave echo_chamber_score 0.0 with the same FALSE. The
        identical agreement data at 3 rounds returns None with 'NOT False'.
        After: None in both, each with a warning, and the echo scores still
        measured and still different."""
        agreed, agreed_messages = _caught(
            lambda: GroupthinkDetector().analyze_convergence([self.IDENTICAL] * 2)
        )
        diverse, diverse_messages = _caught(
            lambda: GroupthinkDetector().analyze_convergence([self.ORTHOGONAL] * 2)
        )
        assert agreed.echo_chamber_score == pytest.approx(1.0)
        assert diverse.echo_chamber_score == pytest.approx(0.0)
        assert agreed.has_groupthink is None
        assert diverse.has_groupthink is None
        for messages in (agreed_messages, diverse_messages):
            assert any("no groupthink verdict was reached" in m for m in messages)
            assert any("NOT False and NOT True" in m for m in messages)
        # The reason travels on the result, not only in a warning, and reaches
        # the consumer: _groupthink_section copies trend_note into couldNotCheck.
        assert agreed.trend_detectable is None
        assert "NOT A TEST" in agreed.trend_note
        assert agreed.to_dict()["has_groupthink"] is None

    def test_a_two_round_rise_is_described_and_not_graded(self):
        """The direction that used to answer True. A jump from total diversity
        to total agreement in two rounds is real movement, so it is stated (the
        warning carries 'from 0 to 1'), but at 3 rounds the same perfect rise is
        already refused for want of power (the Kendall floor is 0.3333 against a
        TREND_ALPHA of 0.1), so publishing True from TWO rounds made the
        shortest series the most confident one."""
        result, messages = _caught(
            lambda: GroupthinkDetector().analyze_convergence([self.ORTHOGONAL, self.IDENTICAL])
        )
        assert result.convergence_trajectory == [pytest.approx(0.0), pytest.approx(1.0)]
        assert result.has_groupthink is None
        assert any("from 0 to 1" in m for m in messages)
        assert result.coalition_structure == [["a", "b"]]

    def test_control_a_real_convergence_over_five_rounds_is_still_true(self):
        """OVER-CORRECTION CONTROL. The batch's own converging fixture: the
        trend test runs, trend_detectable is True and the verdict is a measured
        True with an echo chamber of 1.0."""
        rounds = [
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            {"a": [0.9, 0.2], "b": [0.2, 0.9]},
            {"a": [0.7, 0.5], "b": [0.5, 0.7]},
            {"a": [0.6, 0.6], "b": [0.58, 0.62]},
            {"a": [0.6, 0.6], "b": [0.6, 0.6]},
        ]
        result, _ = _caught(lambda: GroupthinkDetector().analyze_convergence(rounds))
        assert result.has_groupthink is True
        assert result.trend_detectable is True
        assert result.echo_chamber_score == pytest.approx(1.0)
        assert result.convergence_trajectory[1] == pytest.approx(0.4235294117647059)
        assert result.coalition_structure == [["a", "b"]]

    def test_control_a_real_divergence_over_five_rounds_is_still_a_measured_false(self):
        """OVER-CORRECTION CONTROL, the False direction, which is the one a
        blanket None would destroy. The same five rounds reversed: the agents
        move apart, the trend test runs, and has_groupthink is a measured
        False."""
        rounds = [
            {"a": [0.6, 0.6], "b": [0.6, 0.6]},
            {"a": [0.6, 0.6], "b": [0.58, 0.62]},
            {"a": [0.7, 0.5], "b": [0.5, 0.7]},
            {"a": [0.9, 0.2], "b": [0.2, 0.9]},
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        ]
        result, _ = _caught(lambda: GroupthinkDetector().analyze_convergence(rounds))
        assert result.has_groupthink is False
        assert result.trend_detectable is True
        assert result.echo_chamber_score == pytest.approx(0.0)
        assert result.coalition_structure == [["a"], ["b"]]


# =========================================================================
# 4. GroupthinkDetector.detect_coalitions
# =========================================================================


class TestCoalitionThresholdOutsideTheAgreementScale:
    MATRIX = np.array([[1.0, 0.95], [0.95, 1.0]])

    @pytest.mark.parametrize("threshold", [1.5, NAN, INF, -0.5])
    def test_a_threshold_outside_the_agreement_scale_is_refused(self, threshold):
        """Before (2026-09-27): this fully measured matrix at threshold=1.5
        returned [{0}, {1}] with no warning, as did threshold=nan, byte for byte
        what a measured absence of coalitions returns, while the same matrix at
        the default 0.9 returns [{0, 1}]. After: ValueError, the way the sibling
        detectors reject an out-of-range alpha.

        RENAMED 2026-09-29. It was test_a_threshold_no_pair_could_clear_is_refused,
        which is true of 1.5, nan and inf and FALSE of -0.5: every pair clears -0.5,
        and it is refused for being outside the scale, not for being unclearable. The
        assertion was right the whole time; only the name gave a reason that is untrue
        of one of its own parameters, which is the same defect the product message
        carried and BGL6-f06 fixed in src/vfairness/multi_agent/groupthink.py."""
        with pytest.raises(ValueError, match=r"finite agreement score in \[0, 1\]"):
            GroupthinkDetector().detect_coalitions(self.MATRIX, threshold=threshold)

    def test_control_every_in_scale_threshold_still_groups_the_same_matrix(self):
        """OVER-CORRECTION CONTROL with the exact groupings. 1.0 is IN scale and
        attainable (identical vectors have cosine 1.0), so it is accepted and
        returns two singletons as a MEASURED answer: 0.95 really is below
        1.0."""
        detector = GroupthinkDetector()
        assert _caught(lambda: detector.detect_coalitions(self.MATRIX)) == ([{0, 1}], [])
        assert _caught(lambda: detector.detect_coalitions(self.MATRIX, threshold=0.5)) == (
            [{0, 1}],
            [],
        )
        assert _caught(lambda: detector.detect_coalitions(self.MATRIX, threshold=0.0)) == (
            [{0, 1}],
            [],
        )
        assert _caught(lambda: detector.detect_coalitions(self.MATRIX, threshold=1.0)) == (
            [{0}, {1}],
            [],
        )
        three = np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]])
        assert _caught(lambda: detector.detect_coalitions(three)) == ([{0, 1}, {2}], [])

    def test_the_matrix_side_refusals_are_untouched(self):
        """The two guards this fix sits beside: an empty matrix and an undefined
        pair still answer with their own disclosures."""
        out, messages = _caught(lambda: GroupthinkDetector().detect_coalitions(np.zeros((0, 0))))
        assert out == []
        assert any("0 agents" in m and "could not check" in m for m in messages)

        out, messages = _caught(
            lambda: GroupthinkDetector().detect_coalitions(np.array([[1.0, NAN], [NAN, 1.0]]))
        )
        assert out == []
        assert any("COULD NOT BE CHECKED" in m for m in messages)


# =========================================================================
# 5. CompositionalityAnalyzer.analyze
# =========================================================================


class TestCompositionalityTolerance:
    @pytest.mark.parametrize(
        "components, system, threshold",
        [
            ({"a": 0.0, "b": 0.0}, 0.9, NAN),
            ({"a": 0.0, "b": 0.0}, 1e9, INF),
            ({"a": 0.05, "b": 0.05}, 0.05, -5.0),
            ({"a": 0.05, "b": 0.05}, 0.05, 0.0),
        ],
    )
    def test_a_tolerance_that_cannot_classify_is_refused(self, components, system, threshold):
        """Before (2026-09-27), verbatim: threshold=nan gave scenario
        'consistent' with divergence 0.9 and no warning, where the default gives
        'novel_emergence'; threshold=inf gave 'consistent' on a system bias of
        1e9 and made the other three scenarios unreachable for ANY data;
        threshold=-5.0 gave 'amplification' on a divergence of EXACTLY 0.0. A
        threshold of 0 is refused for the same vacuity reason, because
        abs(v) < 0 makes 'novel_emergence' unreachable. After: ValueError."""
        with pytest.raises(ValueError, match="finite tolerance greater than 0"):
            CompositionalityAnalyzer().analyze(components, system, threshold=threshold)

    def test_control_all_four_scenarios_are_still_reachable_and_measured(self):
        """OVER-CORRECTION CONTROL with the real divergences. These are the
        inputs the bad thresholds overwrote, at the default tolerance."""
        analyzer = CompositionalityAnalyzer()
        for components, system, scenario, divergence in (
            ({"a": 0.0, "b": 0.0}, 0.9, "novel_emergence", 0.9),
            ({"a": 0.05, "b": 0.03}, 0.15, "amplification", 0.1),
            ({"a": 0.30}, 0.30, "consistent", 0.0),
            ({"a": 0.3, "b": 0.4}, 0.05, "reduction", -0.35),
        ):
            result, messages = _caught(lambda c=components, s=system: analyzer.analyze(c, s))
            assert result.scenario == scenario, (components, system)
            assert result.divergence == pytest.approx(divergence), (components, system)
            assert messages == []

    def test_the_data_side_refusal_is_untouched(self):
        """The guard this fix sits beside: an unmeasurable component is still
        'not_assessed' with a nan divergence and its warning."""
        result, messages = _caught(
            lambda: CompositionalityAnalyzer().analyze({"a": NAN, "b": 0.03}, 0.9)
        )
        assert result.scenario == "not_assessed"
        assert math.isnan(result.divergence)
        assert any("NOT 'consistent'" in m for m in messages)


# =========================================================================
# 6. EmergentBiasDetector.analyze
# =========================================================================


class TestEmergentNeedsExactlyTwoGroups:
    def test_a_third_group_is_refused_not_dropped(self):
        """Before (2026-09-27) on 180 samples in 3 groups with the whole bias in
        group 2 (system mean 1.0 against group 0's 0.0, component gap 0.40):
        system_bias 0.0, max_component_bias 0.01, is_emergent FALSE, p_value
        1.0, n_samples_unmeasurable 0 and metadata n_samples_measured 180, with
        NO warning, while 60 samples entered neither mask. After: ValueError
        naming the three labels."""
        groups = np.array([0, 1, 2] * 60)
        component = np.where(groups == 2, 0.90, np.where(groups == 1, 0.51, 0.50))
        system = np.where(groups == 2, 1.0, 0.0)
        with pytest.raises(ValueError, match="exactly 2 unique values") as excinfo:
            EmergentBiasDetector().analyze({"a": component}, system, groups)
        assert "[0, 1, 2]" in str(excinfo.value)

    def test_a_nan_group_label_is_refused_too(self):
        """np.unique sorts NaN last, so a sample whose GROUP is unmeasurable
        matched neither mask. Before: is_emergent True over a claimed 200
        measured samples, two of which were never compared, with no warning.
        After: the same guard refuses and names nan."""
        groups = np.array([0.0, 1.0] * 99 + [NAN, NAN])
        binary = np.nan_to_num(groups)
        with pytest.raises(ValueError, match="exactly 2 unique values") as excinfo:
            EmergentBiasDetector().analyze(
                {"a": np.where(binary == 1, 0.55, 0.50)},
                np.where(binary == 1, 1.0, 0.0),
                groups,
            )
        assert "nan" in str(excinfo.value)

    def test_control_the_two_group_run_still_measures_its_real_amplification(self):
        """OVER-CORRECTION CONTROL with the exact numbers: 200 samples, two
        components at gaps of 0.05 and 0.04, a system gap of 1.0."""
        groups = np.array([0, 1] * 100)
        result, messages = _caught(
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
        assert result.system_bias == pytest.approx(1.0)
        assert result.max_component_bias == pytest.approx(0.05)
        assert result.amplification_factor == pytest.approx(20.0, rel=1e-6)
        assert result.n_samples_unmeasurable == 0
        assert result.metadata.parameters["n_samples_measured"] == 200
        assert messages == []

    def test_control_the_pair_carrying_the_bias_is_still_measured_on_its_own(self):
        """OVER-CORRECTION CONTROL for the refusal above: the caller's remedy
        (one pair at a time) reaches the finding the silent truncation hid.
        Groups 0 and 2 of that same three-group frame: system gap 1.0 against a
        component gap of 0.40, amplification 2.5, is_emergent True."""
        groups = np.array([0, 1, 2] * 60)
        component = np.where(groups == 2, 0.90, np.where(groups == 1, 0.51, 0.50))
        system = np.where(groups == 2, 1.0, 0.0)
        pair = groups != 1
        result, messages = _caught(
            lambda: EmergentBiasDetector().analyze(
                {"a": component[pair]}, system[pair], groups[pair]
            )
        )
        assert result.is_emergent is True
        assert result.system_bias == pytest.approx(1.0)
        assert result.max_component_bias == pytest.approx(0.40)
        assert result.amplification_factor == pytest.approx(2.5)
        assert result.metadata.parameters["n_samples_measured"] == 120
        assert messages == []

    def test_the_harness_seam_is_stated_and_the_two_consumers_still_differ(self):
        """CONSUMER of the refusal. MultiAgentRunHarness.record_sample documents
        that more than two group labels are ACCEPTED, and as_emergent_inputs
        hands the labels through as recorded, so the seam between the two
        dispositions has to be visible: the harness's own accessors measure
        every group and disclose the non-binary run (worst-case pairwise gap
        1.0, with a warning), and the binary detector refuses. Measured on a
        90-sample, three-group run whose whole system gap sits in group 2."""
        from vfairness.multi_agent.harness import MultiAgentRunHarness

        harness = MultiAgentRunHarness()
        with harness:
            for i in range(90):
                group = i % 3
                harness.record_sample(
                    group=group,
                    component_outputs={"a": 0.5 + 0.01 * group},
                    system_output=float(group == 2),
                )

        gap, messages = _caught(harness.system_bias)
        assert gap == pytest.approx(1.0)
        assert any("3 distinct group labels were recorded" in m for m in messages)

        components, system, groups = harness.as_emergent_inputs()
        assert sorted(np.unique(groups).tolist()) == [0, 1, 2]
        with pytest.raises(ValueError, match="exactly 2 unique values"):
            EmergentBiasDetector().analyze(components, system, groups)
        # The seam is stated where a caller reads it, not only here.
        assert "raises\nValueError for a third label" in (
            MultiAgentRunHarness.as_emergent_inputs.__doc__ or ""
        )

    def test_emergent_system_bias_refusal_says_the_system_outputs_are_the_cause(self):
        """THE PIN THE AUDIT FOUND UNPINNED, re-aimed. The existing test
        test_emergent_refuses_when_no_sample_carries_a_system_measurement
        asserts is_emergent None, n_samples_unmeasurable 200 and '200 of 200',
        and every one of those three is produced by the n_measured == 0 branch
        BELOW as well, so DELETING the `if not math.isfinite(system_bias):`
        branch left it green. What only that branch produces is the DIAGNOSIS:
        it names the system outputs and the groups that remain. The branch below
        blames the components instead ('none of the 1 components had a finite
        bias'), which is wrong here: the component is a perfectly ordinary
        constant 0.5 and it is the SYSTEM that could not be measured."""
        groups = np.array([0, 1] * 100)
        result, messages = _caught(
            lambda: EmergentBiasDetector().analyze(
                {"a": np.full(200, 0.5)}, np.full(200, NAN), groups
            )
        )
        assert result.is_emergent is None
        assert result.is_significant is None
        assert result.n_samples_unmeasurable == 200
        joined = " || ".join(messages)
        assert "system bias could not be measured" in joined
        assert "0 group(s) remain after excluding 200 of 200" in joined
        assert "not an absence of emergent bias" in joined
        # The other branch's diagnosis must NOT be the one a reader gets here.
        assert "components had a finite bias" not in joined

    def test_the_no_component_refusal_keeps_its_own_diagnosis(self):
        """The companion half: with no components at all the OTHER branch fires,
        and it must name the components rather than the system outputs. Pinning
        both sentences is what makes either branch's deletion visible."""
        groups = np.array([0, 1] * 100)
        result, messages = _caught(
            lambda: EmergentBiasDetector().analyze({}, np.where(groups == 1, 1.0, 0.0), groups)
        )
        assert result.is_emergent is None
        assert result.system_bias == pytest.approx(1.0)
        assert math.isnan(result.amplification_factor)
        joined = " || ".join(messages)
        assert "no components were supplied" in joined
        assert "system bias could not be measured" not in joined
