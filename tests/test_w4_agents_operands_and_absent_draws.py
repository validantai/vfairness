"""Wave-4 pins for the three agent/multi-agent grades the tier-1 audit OVERTURNED.

Each of the three was overturned on a SIBLING DOOR beside a fix that itself
holds, so every section here pins the sibling and keeps a control that asserts
the healthy case's REAL number. Written beside the fix, executed against it, and
then SABOTAGED: the defect was reintroduced at its source line, the named test
was shown to go red, and the source was restored and compared byte for byte with
``diff -q`` against ``git show HEAD:<path>``. The counts are recorded in
/tmp/claude-501/bgl/fix/fix-T1-F11-agents.json.

1. ``TemporalTracker.detect_drift_ewma`` had its DATA side guarded (sigma <=
   1e-12, a chart with no width) and both CALLER-SUPPLIED operands open:
   sigma_limit=nan published has_drift=False, the verdict "no drift left the
   control limits", from `ewma_val > nan` / `ewma_val < nan`, which are both
   False for every turn; sigma_limit=-3.0 INVERTED the limits and flagged every
   turn of a clean chart; span=nan made every EWMA value nan and published the
   same False; span=0 and span=-1 raised ZeroDivisionError. Its two siblings on
   the same class carried the identical defect through their own operands
   (detect_drift_cusum threshold / drift_limit, detect_feedback_loop alpha) and
   are fixed and pinned here in the same change.

2. ``MultiAgentRunHarness.record_routing`` refused three shapes of "absent
   rather than written down" (None, a float nan, a blank string) and accepted
   pd.NA, pd.NaT, np.datetime64('NaT'), np.ma.masked and the literal strings
   'None' and 'nan', each of which ``str()`` turns into a demographic group that
   does not exist. The incident the method's own docstring documents was
   reproduced verbatim with pd.NA in place of None: cramers_v 1.0, odds_ratio
   inf, p 1.0825e-05, is_significant True, zero warnings, against a group that
   is not a group.

3. ``GroupthinkDetector.analyze_convergence`` guarded the unmeasurable OBSERVED
   convergence and left the unmeasurable permutation DRAWS voting: a nan draw is
   never >= the observed value, so every one of them stayed in the denominator
   as a draw that did NOT reach it, which is evidence FOR significance. 173 of
   500 draws unmeasurable published p=0.039920 is_significant=True where the
   p over the measurable draws alone is 0.060976, above the method's alpha of
   0.05. Its sibling EmergentBiasDetector.analyze already skipped, counted,
   warned and published the same quantity.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness._not_assessed import NOT_ASSESSED
from vfairness.agents.temporal import TemporalTracker
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.multi_agent.groupthink import GroupthinkDetector
from vfairness.multi_agent.harness import MultiAgentRunHarness, _label_not_recorded

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


# The auditor's own fixture: eight turns ramping from parity to a 0.84 gap.
# The default detect_drift_ewma call draws limits 0.025640 .. 0.814360 on it and
# reports no drift; detect_drift_cusum reports max_cusum 1.7732683535398865 and
# no drift; detect_feedback_loop reports increasing with p = 4.9603e-05.
RAMP = [round(i * 0.12, 10) for i in range(8)]


# =========================================================================
# 1. TemporalTracker.detect_drift_ewma and its two siblings on the same class
# =========================================================================


class TestTheChartParametersAreOperandsToo:
    @pytest.mark.parametrize("sigma_limit", [NAN, INF, -3.0, 0.0])
    def test_a_control_limit_width_that_cannot_decide_is_a_could_not_check(self, sigma_limit):
        """Before (2026-09-30) on this ramp: sigma_limit=nan and inf both
        published has_drift=False, the verdict 'no drift left the control
        limits', from comparisons that are False for every value; -3.0 inverted
        the limits (upper 0.0256 BELOW lower 0.8144) and flagged all eight
        turns; 0.0 flagged all eight through zero-width limits. All four with
        ZERO warnings."""
        result, messages = _caught(
            lambda: _tracker(RAMP).detect_drift_ewma(sigma_limit=sigma_limit)
        )
        assert result["has_drift"] is None
        assert result["drift_points"] == []
        assert math.isnan(result["upper_limit"])
        assert math.isnan(result["lower_limit"])
        assert math.isnan(result["center_line"])
        assert any(f"sigma_limit={sigma_limit!r}" in m for m in messages)
        assert any("NOT False" in m and "NOT True" in m for m in messages)

    @pytest.mark.parametrize("span", [NAN, INF, 0, -1, 0.5])
    def test_a_smoothing_span_that_cannot_smooth_is_a_could_not_check(self, span):
        """span=nan made every ewma value nan and published has_drift=False with
        zero warnings; span=0 and span=-1 raised ZeroDivisionError out of
        lam / (2 - lam)."""
        result, messages = _caught(lambda: _tracker(RAMP).detect_drift_ewma(span=span))
        assert result["has_drift"] is None
        assert result["ewma_values"] == []
        assert math.isnan(result["center_line"])
        assert any(f"span={span!r}" in m for m in messages)

    def test_the_operand_guard_sits_above_the_dispatch(self):
        """A bad operand is refused on a two-turn tracker (the too-few-turns
        branch), on a CONSTANT tracker (the zero-width-chart branch, which does
        not read sigma_limit at all) and on the ramp (the chart branch). A guard
        inside one branch would let the other two answer for it."""
        for series in ([0.0, 0.1], [0.3] * 6, RAMP):
            result, messages = _caught(
                lambda s=series: _tracker(s).detect_drift_ewma(sigma_limit=NAN)
            )
            assert result["has_drift"] is None
            assert any("sigma_limit=nan" in m for m in messages)

    def test_control_the_chart_still_measures_both_verdicts_with_its_real_numbers(self):
        """OVER-CORRECTION CONTROL. The default call on this ramp is a silent
        measured False with the limits the auditor printed, a real excursion is
        still flagged at its own turn number, and a non-default but legal
        sigma_limit and span still measure."""
        ewma, messages = _caught(lambda: _tracker(RAMP).detect_drift_ewma())
        assert messages == []
        assert ewma["has_drift"] is False
        assert ewma["drift_points"] == []
        assert ewma["upper_limit"] == pytest.approx(0.8143602414037197)
        assert ewma["lower_limit"] == pytest.approx(0.025639758596280482)
        assert ewma["center_line"] == pytest.approx(0.42)

        excursion, messages = _caught(lambda: _tracker([0.05] * 15 + [0.9] * 3).detect_drift_ewma())
        assert messages == []
        assert excursion["has_drift"] is True
        assert excursion["drift_points"] == [17]

        tight, messages = _caught(lambda: _tracker(RAMP).detect_drift_ewma(span=2, sigma_limit=0.5))
        assert messages == []
        assert tight["has_drift"] is True
        assert tight["n_turns_measured"] == 8

    def test_control_the_data_side_refusals_are_untouched(self):
        """The two guards this one sits above still answer for themselves: a
        two-turn tracker is None for want of a chart, and a CONSTANT trajectory
        is still a MEASURED False with all three chart lines on the centre."""
        short, messages = _caught(lambda: _tracker([0.0, 0.1]).detect_drift_ewma())
        assert short["has_drift"] is None
        assert any("2 of 2" in m for m in messages)

        flat, messages = _caught(lambda: _tracker([0.005] * 5).detect_drift_ewma())
        assert messages == []
        assert flat["has_drift"] is False
        assert flat["drift_points"] == []
        assert flat["center_line"] == pytest.approx(0.005)
        assert flat["upper_limit"] == flat["center_line"] == flat["lower_limit"]


class TestTheCusumParametersAreOperandsToo:
    @pytest.mark.parametrize("threshold", [NAN, INF, -1.0])
    def test_a_slack_that_cannot_accumulate_is_a_could_not_check(self, threshold):
        """Before: threshold=nan and inf published has_drift=False AND
        max_cusum=0.0, the reading of a chart that accumulated nothing, with
        zero warnings; threshold=-1.0 signalled drift at turn 2 of this ramp
        because a negative slack adds to the statistic every turn whatever the
        data does."""
        result, messages = _caught(lambda: _tracker(RAMP).detect_drift_cusum(threshold=threshold))
        assert result["has_drift"] is None
        assert result["drift_point"] is None
        assert math.isnan(result["max_cusum"])
        assert any(f"threshold={threshold!r}" in m for m in messages)

    @pytest.mark.parametrize("drift_limit", [NAN, INF, -1.0, 0.0])
    def test_a_decision_interval_that_cannot_decide_is_a_could_not_check(self, drift_limit):
        """Before: drift_limit=nan published has_drift=False with a max_cusum of
        1.77 in the same dict (`cusum > nan` is False at every turn) and
        drift_limit=0.0 / -1.0 signalled at turn 0 for the same ramp."""
        result, messages = _caught(
            lambda: _tracker(RAMP).detect_drift_cusum(drift_limit=drift_limit)
        )
        assert result["has_drift"] is None
        assert math.isnan(result["max_cusum"])
        assert any(f"drift_limit={drift_limit!r}" in m for m in messages)

    def test_control_the_cusum_still_measures_both_verdicts_with_its_real_numbers(self):
        """OVER-CORRECTION CONTROL. A slack of exactly 0 is legal and measures;
        the default call on this ramp is a silent measured False carrying the
        max_cusum the auditor printed; a tight interval still finds the shift."""
        default, messages = _caught(lambda: _tracker(RAMP).detect_drift_cusum())
        assert messages == []
        assert default["has_drift"] is False
        assert default["max_cusum"] == pytest.approx(1.7732683535398865)

        no_slack, messages = _caught(lambda: _tracker(RAMP).detect_drift_cusum(threshold=0.0))
        assert messages == []
        assert no_slack["has_drift"] is False
        assert no_slack["max_cusum"] == pytest.approx(3.4914862437758787)

        tight, messages = _caught(
            lambda: _tracker(RAMP).detect_drift_cusum(threshold=0.1, drift_limit=1.0)
        )
        assert messages == []
        assert tight["has_drift"] is True
        assert tight["drift_point"] == 0
        assert tight["max_cusum"] == pytest.approx(3.0914862437758783)

    def test_control_the_data_side_refusals_are_untouched(self):
        short, messages = _caught(lambda: _tracker([0.0, 0.1]).detect_drift_cusum())
        assert short["has_drift"] is None
        assert math.isnan(short["max_cusum"])
        assert any("2 of 2" in m for m in messages)

        flat, messages = _caught(lambda: _tracker([0.005] * 5).detect_drift_cusum())
        assert messages == []
        assert flat["has_drift"] is False
        assert flat["max_cusum"] == 0.0


class TestTheTrendAlphaIsAnOperandToo:
    @pytest.mark.parametrize("alpha", [NAN, INF, -1.0, 0.0, 1.0, 2.0])
    def test_an_alpha_that_cannot_decide_is_a_could_not_check(self, alpha):
        """Before, on a ramp whose default call reports increasing with
        p=4.96e-05: alpha=nan published has_feedback_loop=False with
        trend_direction 'stable' and that same p printed beside it, from
        `p_value < nan`; alpha<=0 could never be significant and alpha>=1 made
        every positive tau a feedback loop. All with zero warnings."""
        result, messages = _caught(lambda: _tracker(RAMP).detect_feedback_loop(alpha=alpha))
        assert result["has_feedback_loop"] is None
        assert result["trend_direction"] == NOT_ASSESSED
        assert math.isnan(result["trend_strength"])
        assert math.isnan(result["p_value"])
        assert any(f"alpha={alpha!r}" in m for m in messages)

    def test_control_a_real_alpha_still_measures_both_verdicts(self):
        """OVER-CORRECTION CONTROL. The ramp is still an increasing feedback
        loop at the default alpha with its real tau and p, a flat trajectory is
        still a measured False, and a legal non-default alpha still decides."""
        loop, messages = _caught(lambda: _tracker(RAMP).detect_feedback_loop())
        assert messages == []
        assert loop["has_feedback_loop"] is True
        assert loop["trend_direction"] == "increasing"
        assert loop["trend_strength"] == pytest.approx(1.0)
        assert loop["p_value"] == pytest.approx(4.960317460317462e-05, rel=1e-6)

        flat, messages = _caught(lambda: _tracker([0.3] * 8).detect_feedback_loop())
        assert messages == []
        assert flat["has_feedback_loop"] is False
        assert flat["trend_direction"] == "stable"

        strict, messages = _caught(lambda: _tracker(RAMP).detect_feedback_loop(alpha=1e-6))
        assert messages == []
        assert strict["has_feedback_loop"] is False
        assert strict["trend_direction"] == "stable"
        assert strict["p_value"] == pytest.approx(4.960317460317462e-05, rel=1e-6)

    def test_control_the_data_side_refusal_is_untouched(self):
        short, messages = _caught(lambda: _tracker([0.0, 0.1]).detect_feedback_loop())
        assert short["has_feedback_loop"] is None
        assert short["trend_direction"] == NOT_ASSESSED
        assert any("2 of 2" in m for m in messages)


# =========================================================================
# 2. MultiAgentRunHarness.record_routing: absence has six doors
# =========================================================================

# Every scalar a dataframe uses for a missing cell, plus the strings str() prints
# for them. The first five were already refused by the wave-2 predicate; the rest
# were ACCEPTED and stored as a label.
_ALREADY_REFUSED = [None, NAN, np.float64("nan"), "", "   "]
_WAVE4_DOORS = [
    pd.NA,
    pd.NaT,
    np.datetime64("NaT"),
    np.ma.masked,
    np.float32("nan"),
    "None",
    "nan",
    "<NA>",
    "NaT",
    "--",
    " None ",
]


class TestAbsenceHasSixDoorsNotThree:
    @pytest.mark.parametrize("absent", _WAVE4_DOORS + _ALREADY_REFUSED)
    def test_a_demographic_that_was_not_recorded_is_refused(self, absent):
        """Before (2026-09-30): pd.NA stored the group '<NA>', pd.NaT and
        np.datetime64('NaT') stored 'NaT', np.ma.masked stored '--', and the
        literal strings 'None' and 'nan' stored themselves, all with
        _label_not_recorded returning False."""
        harness = MultiAgentRunHarness()
        with pytest.raises(ValueError):
            harness.record_routing("agent_x", absent)
        assert harness.trace.routing_demographics == []
        assert harness.trace.routing_decisions == []

    @pytest.mark.parametrize("absent", _WAVE4_DOORS + _ALREADY_REFUSED)
    def test_a_route_that_was_not_recorded_is_refused(self, absent):
        """BOTH ARMS share the one predicate, so the route side has to be pinned
        too: record_routing(pd.NA, 'a') stored the route '<NA>'."""
        harness = MultiAgentRunHarness()
        with pytest.raises(ValueError):
            harness.record_routing(absent, "a")
        assert harness.trace.routing_decisions == []

    @pytest.mark.parametrize("absent", _WAVE4_DOORS + _ALREADY_REFUSED)
    def test_the_predicate_itself_says_not_recorded(self, absent):
        assert _label_not_recorded(absent) is True

    def test_the_message_names_the_rendering_the_caller_would_have_got(self):
        """A caller who passed pd.NA has to read '<NA>' to recognise the column
        it came out of. The wave-2 message said "'None' (or '')" for every
        shape, which sends them looking for the wrong string."""
        harness = MultiAgentRunHarness()
        with pytest.raises(ValueError) as excinfo:
            harness.record_routing("agent_x", pd.NA)
        assert "'<NA>'" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            harness.record_routing("agent_x", np.ma.masked)
        assert "'--'" in str(excinfo.value)

    def test_the_maximal_false_finding_can_no_longer_be_built_through_pd_na(self):
        """The incident this method's docstring documents, reproduced verbatim
        with pd.NA in place of None: ten record_routing('agent_x', pd.NA)
        against ten record_routing('agent_y', 'a') gave groups ['<NA>', 'a'],
        cramers_v 1.0, odds_ratio inf, p 1.082508822446903e-05,
        is_significant True, detectable True and ZERO warnings, a maximal
        significant finding of routing discrimination against a group that is
        not a group. It now cannot be recorded at all."""
        harness = MultiAgentRunHarness()
        with pytest.raises(ValueError):
            for _ in range(10):
                harness.record_routing("agent_x", pd.NA)
        for _ in range(10):
            harness.record_routing("agent_y", "a")
        routes, demographics = harness.as_delegation_inputs()
        assert set(demographics) == {"a"}
        assert "<NA>" not in demographics
        assert set(routes) == {"agent_y"}
        # One route and one group, so there is no contrast left to publish a
        # finding about, and the auditor says so rather than inventing one.
        with pytest.raises(ValueError, match="2 distinct routes"):
            DelegationRoutingAuditor().analyze(routes, demographics)

    @pytest.mark.parametrize(
        "label",
        [
            0,
            0.0,
            False,
            True,
            1,
            np.int64(1),
            np.float64(0.5),
            "a",
            "NA",
            "N/A",
            "null",
            "NULL",
            "unknown",
            "missing",
            "0",
            "None of the above",
            "nanotech",
            "--x",
        ],
    )
    def test_control_a_real_label_is_still_recorded_and_still_stringified(self, label):
        """OVER-CORRECTION CONTROL, and the reason _ABSENCE_RENDERINGS is an
        EXACT-match set rather than a substring or a fuzzy one. 'NA' is a region,
        'null' and 'missing' and 'unknown' are labels a caller can mean, 'None of
        the above' is a survey category, 'nanotech' contains 'nan' and '--x'
        contains '--'. All of them are still recorded, and 0 / 0.0 / False are
        still recorded because they are falsy, not absent."""
        harness = MultiAgentRunHarness()
        harness.record_routing("r", label)
        assert harness.trace.routing_demographics == [str(label)]
        assert _label_not_recorded(label) is False

    def test_control_a_real_routing_disparity_is_still_found_with_its_real_numbers(self):
        """OVER-CORRECTION CONTROL. The same ten-against-ten shape with BOTH
        groups labelled still reports the maximal finding, which is correct
        there: the discrimination is real and both groups exist."""
        harness = MultiAgentRunHarness()
        for _ in range(10):
            harness.record_routing("agent_x", "a")
        for _ in range(10):
            harness.record_routing("agent_y", "b")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = DelegationRoutingAuditor().analyze(*harness.as_delegation_inputs())
        published = result.to_dict()
        assert sorted(published["groups"]) == ["a", "b"]
        assert published["cramers_v"] == pytest.approx(1.0)
        assert published["p_value"] == pytest.approx(1.082508822446903e-05)
        assert published["is_significant"] is True
        assert published["detectable"] is True
        assert caught == []


# =========================================================================
# 3. GroupthinkDetector.analyze_convergence: an unmeasurable draw is not a vote
# =========================================================================


def _silent_rounds(n_rounds, n_silent):
    """The auditor's fixture: agent a holds a ZERO-LENGTH output vector in the
    first ``n_silent`` rounds, so the cosine against b is undefined there and
    _cosine_similarity correctly answers nan. The final round is total agreement,
    so the OBSERVED convergence is measurable (1.0) and the guard above the loop
    does not fire; only the permutation DRAWS that happen to place a silent round
    last are unmeasurable."""
    return [
        {
            "a": [0.0, 0.0] if i < n_silent else [1.0, 0.0],
            "b": [1.0, 0.0] if i == n_rounds - 1 else [0.0, 1.0],
        }
        for i in range(n_rounds)
    ]


class TestAnUnmeasurableDrawIsNotAVoteAgainst:
    # (n_rounds, n_silent, measured, unmeasurable, honest p as published now,
    #  p as published BEFORE with the nan draws in the denominator)
    CASES = [
        (15, 5, 327, 173, 0.06097560975609756, 0.039920159680638724),
        (16, 5, 338, 162, 0.07079646017699115, 0.047904191616766467),
        (18, 6, 308, 192, 0.0744336569579288, 0.045908183632734525),
    ]

    @pytest.mark.parametrize("n_rounds,n_silent,measured,unmeasurable,honest,before", CASES)
    def test_the_p_value_rests_only_on_the_draws_that_were_measured(
        self, n_rounds, n_silent, measured, unmeasurable, honest, before
    ):
        """Before (2026-09-30): a nan draw is never >= the observed convergence,
        so every unmeasurable draw stayed in the denominator (n_permutations + 1)
        as a draw that did NOT reach it, which is evidence FOR significance. All
        three shapes published is_significant=True below the 0.05 alpha; the p
        over the measurable draws alone is above it in all three."""
        result, messages = _caught(
            lambda: GroupthinkDetector().analyze_convergence(_silent_rounds(n_rounds, n_silent))
        )
        assert result.n_permutations_measured == measured
        assert result.n_permutations_unmeasurable == unmeasurable
        assert result.n_permutations_measured + result.n_permutations_unmeasurable == 500
        assert result.p_value == pytest.approx(honest)
        # The number it used to publish is strictly below alpha; the honest one is
        # strictly above it, so the verdict really did turn on the nan draws.
        assert before < 0.05 < honest
        assert result.is_significant is None
        assert result.convergence_detectable is False
        assert any(
            f"{unmeasurable} of 500 permutation draws" in m and str(measured) in m for m in messages
        )

    @pytest.mark.parametrize("n_rounds,n_silent,measured,unmeasurable,honest,before", CASES)
    def test_the_loss_is_visible_where_a_reader_looks_and_not_only_in_a_warning(
        self, n_rounds, n_silent, measured, unmeasurable, honest, before
    ):
        """to_dict() had NO field whose name contained 'unmeasur', metadata still
        said n_permutations=500 as though all 500 had voted, and convergence_note
        was the empty string. A correct count nobody renders is not a
        disclosure."""
        result = GroupthinkDetector().analyze_convergence(_silent_rounds(n_rounds, n_silent))
        published = result.to_dict()
        assert published["n_permutations_unmeasurable"] == unmeasurable
        assert published["n_permutations_measured"] == measured
        assert result.metadata.parameters["n_permutations"] == 500
        assert result.metadata.parameters["n_permutations_unmeasurable"] == unmeasurable
        assert result.metadata.parameters["n_permutations_measured"] == measured
        assert "PARTIAL COVERAGE" in result.convergence_note
        assert f"{unmeasurable} of 500" in result.convergence_note

    @pytest.mark.parametrize("n_rounds,n_silent,measured,unmeasurable,honest,before", CASES)
    def test_the_detectability_floor_uses_the_same_denominator_as_the_p_value(
        self, n_rounds, n_silent, measured, unmeasurable, honest, before
    ):
        """min_attainable is the FLOOR of the p-value estimator, so its
        denominator has to be the one the p-value divides by. Left at
        1 + n_permutations it would understate the floor of an estimator that can
        only reach 1 / (measured + 1), and detectability() would certify a design
        whose smallest attainable p-value is above alpha."""
        result = GroupthinkDetector().analyze_convergence(_silent_rounds(n_rounds, n_silent))
        floor = result.metadata.parameters["min_attainable_p_permutation"]
        assert floor == pytest.approx(honest)
        assert floor * (measured + 1) == pytest.approx(round(floor * (measured + 1)))

    def test_total_loss_of_the_draws_is_refused_not_given_the_smallest_p_on_the_scale(self):
        """PARTIAL loss is disclosed; TOTAL loss is refused. With every round but
        the last holding a zero-length vector for BOTH agents, not one of the 500
        draws is measurable while the observed round is. Before: p_value
        0.001996007984031936, which is 1/(500 + 1), the SMALLEST this estimator
        can produce, published as a number out of a test where nothing was
        compared, with a note saying only that the floor 'was not computable'."""
        n_rounds = 14
        rounds = [{"a": [0.0, 0.0], "b": [0.0, 0.0]} for _ in range(n_rounds - 1)]
        rounds.append({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        result, messages = _caught(lambda: GroupthinkDetector().analyze_convergence(rounds))
        assert result.n_permutations_measured == 0
        assert result.n_permutations_unmeasurable == 500
        assert math.isnan(result.p_value)
        assert result.is_significant is None
        assert result.convergence_detectable is None
        assert "all 500 permutation draws were unmeasurable" in result.convergence_note
        assert any("not one of 500 permutation draws" in m for m in messages)

    def test_control_a_real_convergence_is_still_measured_and_still_significant(self):
        """OVER-CORRECTION CONTROL 1. A guard that refuses everything passes every
        refusal test. The five-round rising series still MEASURES
        has_groupthink=True with its real echo score and its real p, and no draw
        is lost, so nothing is disclosed that did not happen."""
        rising = [
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            {"a": [1.0, 0.0], "b": [0.3, 1.0]},
            {"a": [1.0, 0.0], "b": [0.6, 1.0]},
            {"a": [1.0, 0.0], "b": [1.0, 0.6]},
            {"a": [1.0, 0.0], "b": [1.0, 0.05]},
        ]
        result = GroupthinkDetector().analyze_convergence(rising)
        assert result.has_groupthink is True
        assert result.echo_chamber_score == pytest.approx(0.9987523388778446)
        assert result.p_value == pytest.approx(0.18762475049900199)
        assert result.n_permutations_measured == 500
        assert result.n_permutations_unmeasurable == 0
        assert "PARTIAL COVERAGE" not in result.convergence_note

    def test_control_a_real_finding_survives_a_small_draw_loss(self):
        """OVER-CORRECTION CONTROL 2, the one that matters most: excluding the
        unmeasurable draws must not suppress a finding that is really there. Over
        30 rounds with agent a silent in exactly ONE of them, 16 of 500 draws are
        lost, the design keeps its power and is_significant is still True. The
        p-value moves from 0.031936127744510975 (16/501) to
        0.032989690721649485 (16/485), which is the same numerator over the
        draws that existed, and the PARTIAL COVERAGE sentence is the only thing
        on the result that would otherwise mention the loss, because
        detectability() leaves its note EMPTY for a design with power."""
        n_rounds = 30
        rounds = [
            {
                "a": [0.0, 0.0] if i == 3 else [1.0, 0.0],
                "b": [i / (n_rounds - 1.0), 1.0 - i / (n_rounds - 1.0)],
            }
            for i in range(n_rounds)
        ]
        result = GroupthinkDetector().analyze_convergence(rounds)
        assert result.n_permutations_unmeasurable == 16
        assert result.n_permutations_measured == 484
        assert result.p_value == pytest.approx(0.032989690721649485)
        assert result.is_significant is True
        assert result.convergence_detectable is True
        assert result.convergence_note.startswith("PARTIAL COVERAGE")

    def test_control_the_observed_value_guard_is_untouched(self):
        """The guard this fix sits beside: an unmeasurable OBSERVED convergence
        is still refused with p nan, is_significant None, zero draws and its own
        diagnosis."""
        rounds = [{"a": [1.0, 0.0], "b": [0.0, 1.0]}] * 3 + [{"a": [0.0, 0.0], "b": [1.0, 0.0]}]
        result, messages = _caught(lambda: GroupthinkDetector().analyze_convergence(rounds))
        assert math.isnan(result.p_value)
        assert result.is_significant is None
        assert result.n_permutations_measured == 0
        assert "the observed convergence was not measurable" in result.convergence_note
        assert any("observed convergence could not" in m for m in messages)


class TestEveryRoundMustCarryTheSameAgents:
    """The sibling door in the same method. `all_agents` is taken from round 0
    alone, so the permutation null is built over round 0's agents while the
    observed convergence and the agreement matrix are built over each round's
    own agents. The precondition was documented in the Args and enforced
    nowhere."""

    LATE_ARRIVAL = [
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        {"a": [1.0, 0.0], "b": [0.5, 1.0], "c": [1.0, 0.0]},
        {"a": [1.0, 0.0], "b": [1.0, 0.1], "c": [1.0, 0.0]},
        {"a": [1.0, 0.0], "b": [1.0, 0.05], "c": [1.0, 0.0]},
    ]
    EARLY_EXIT = [
        {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]},
        {"a": [1.0, 0.0], "b": [0.5, 1.0]},
        {"a": [1.0, 0.0], "b": [1.0, 0.1]},
    ]

    def test_an_agent_that_appears_only_after_round_zero_is_refused(self):
        """Before: has_groupthink True, echo_chamber_score 0.999168225918563 over
        THREE pairs, is_significant True, convergence_detectable True and
        p_value 0.001996007984031936, which is 1/(500 + 1), the SMALLEST p this
        estimator can produce, reached because not one TWO-agent draw could match
        a THREE-agent observation. metadata reported n_agents=2 while
        coalition_structure reported [['a', 'b', 'c']], with ZERO warnings."""
        with pytest.raises(ValueError, match="Every round must carry the same agents"):
            GroupthinkDetector().analyze_convergence(self.LATE_ARRIVAL)

    def test_an_agent_that_disappears_after_round_zero_is_refused_with_a_diagnosis(self):
        """The other direction used to raise a bare KeyError: 'c' from inside the
        permutation loop, naming neither the round nor the requirement. Both
        directions now name the round, the two agent sets and what to do."""
        with pytest.raises(ValueError) as excinfo:
            GroupthinkDetector().analyze_convergence(self.EARLY_EXIT)
        message = str(excinfo.value)
        assert "round 0 has ['a', 'b', 'c']" in message
        assert "round 1 has ['a', 'b']" in message
        assert "missing ['c']" in message

    def test_the_refusal_names_the_extra_agent_and_the_round_it_appears_in(self):
        with pytest.raises(ValueError) as excinfo:
            GroupthinkDetector().analyze_convergence(self.LATE_ARRIVAL)
        message = str(excinfo.value)
        assert "round 1 has ['a', 'b', 'c']" in message
        assert "extra ['c']" in message

    def test_control_a_consistent_three_agent_run_is_still_measured(self):
        """OVER-CORRECTION CONTROL. The same interaction with c present in EVERY
        round, including round 0, still measures: the guard is about consistency,
        not about the number of agents."""
        consistent = [
            {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]},
            {"a": [1.0, 0.0], "b": [0.5, 1.0], "c": [1.0, 0.0]},
            {"a": [1.0, 0.0], "b": [1.0, 0.1], "c": [1.0, 0.0]},
            {"a": [1.0, 0.0], "b": [1.0, 0.05], "c": [1.0, 0.0]},
        ]
        result = GroupthinkDetector().analyze_convergence(consistent)
        assert result.metadata.parameters["n_agents"] == 3
        assert result.coalition_structure == [["a", "b", "c"]]
        assert result.echo_chamber_score == pytest.approx(0.999168225918563)
        assert result.n_permutations_measured == 500
        assert result.n_permutations_unmeasurable == 0
