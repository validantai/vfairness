"""Batch g014: ``vfairness.operations.monitoring.alerts``.

Every item in the batch was executed on healthy input carrying a real,
findable disparity and on the degenerate inputs where its claim stops being
defined. Two defects were found by execution and fixed here; a third is
disclosed but NOT closed and is described below so the next wave can finish it.

DEFECT 1, FIXED: ``calculate_priority`` refused a drift score it HAD.
    The boost read its value with ``isinstance(raw_drift, (int, float))``.
    ``np.float64`` is a subclass of ``float``; ``np.float32``, ``np.float16``
    and ``np.int64`` are not. Measured on HEAD, one drift event, the dtype of a
    measured 0.82 the only difference::

        0.82             -> (11.17, 'CRITICAL')  pagerduty, 0 warnings
        np.float32(0.82) -> (nan,   'UNSCORED')  slack triage, warned that
                            "the drift this alert is ABOUT was never measured"

    while the same value reached the payload as ``drift_score=0.82`` and the
    message printed "Drift score: 0.820" inside the sentence saying the
    severity could not be computed. A real measurement thrown away, a real
    CRITICAL de-escalated, and one payload contradicting itself: the refusal
    defect running backwards, which costs more than the original. The fix
    delegates to ``vfairness._triage.is_measured``, the library's single
    answer, rather than adding a sixth private copy of the predicate.

DEFECT 2, FIXED: ``is_alert_warranted`` guarded one operand of two.
    ``set_threshold`` refuses a non-finite threshold, and says in its own
    docstring that "the guard belongs on both operands". The constructor
    bypasses ``set_threshold``, so ``initial_threshold`` never meets it.
    Measured on HEAD::

        mgr = AdaptiveThresholdManager(initial_threshold=float("nan"))
        mgr.is_alert_warranted("k", 0.94) -> False, 0 warnings
        mgr.is_alert_warranted("k", 0.99) -> False, 0 warnings

    Every comparison against NaN is False, so the loudest drift this gate can
    receive was answered "nobody is paged", silently, for every key.

DEFECT 3, CLOSED in BGL5 A-operations-1 on 2026-09-27 (it was OPEN, disclosed
    only, when this file was written; the closure is recorded at the end of this
    entry): ``build_drift_event`` manufactured the four
    context factors. The triage called it a converter. It is the thing that
    decides the severity. ``calculate_priority`` was hardened twice (R-8, BGL
    S2) so an ABSENT factor is excluded and named rather than counted; this
    helper defeats both by making the absence PRESENT::

        calculate_priority({"drift_score": 0.5, "mean_shift": 0.1})
            -> (nan, 'UNSCORED')  hand triage, 1 warning
        calculate_priority(build_drift_event("dp", ["B"], 0.5, 0.1))
            -> (5.1, 'HIGH')      slack, 0 warnings

    5.1 cleared the 5.0 HIGH band on substitutes alone. The fix applied on
    2026-09-27 omits an unsupplied factor from the returned dict, which routes
    it into the exclusion path already written for it, so the second line above
    is now ``(nan, 'UNSCORED')`` as well and both routes agree. A fully supplied
    event is unchanged: all four present still score (11.67, 'CRITICAL'), a
    partially supplied one scores on what was measured, and four supplied zeros
    are kept as the measurements they are, (0.5, 'LOW') with no warning. The
    test below pinned the DISCLOSURE rather than the fabricated 5.1; its second
    half now pins the absence instead of recording it as open.

Every test here asserts the third state AND a control showing a real value is
still measured exactly; the expected numbers are computed from the weights by
hand in the assertions, never copied from a run.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.operations.monitoring import (
    AdaptiveThresholdManager,
    FairnessAlertPrioritizer,
)

# regulatory_risk 1.0*3.5 + population_impact 0.8*2.0 + drift_velocity 0.9*2.5
# + historical_discrimination 1.0*3.0 = 3.5 + 1.6 + 2.25 + 3.0 = 10.35.
_CONTEXT_ONLY = 3.5 + 1.6 + 2.25 + 3.0
_FULL_EVENT = {
    "regulatory_risk": 1.0,
    "population_impact": 0.8,
    "drift_velocity": 0.9,
    "historical_discrimination": 1.0,
    "metric_name": "equalized_odds",
    "affected_groups": ["Black", "Female"],
    "mean_shift": 0.07,
}


def _catch(fn, *args, **kwargs):
    """Run *fn* with warnings ENABLED, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Defect 1: a measured drift score, in a dtype isinstance did not recognise
# ---------------------------------------------------------------------------


class TestAMeasuredDriftScoreIsNotRefusedForItsDtype:
    @pytest.mark.parametrize(
        "value",
        [0.82, np.float64(0.82), np.float32(0.82), np.float16(0.82)],
        ids=["float", "np.float64", "np.float32", "np.float16"],
    )
    def test_every_numeric_dtype_of_one_measured_score_is_scored(self, value):
        prioritizer = FairnessAlertPrioritizer()
        (score, severity), caught = _catch(
            prioritizer.calculate_priority, {**_FULL_EVENT, "drift_score": value}
        )

        assert not math.isnan(score), f"a measured {type(value).__name__} was refused"
        assert severity == "CRITICAL"
        # 10.35 from the four weighted factors + the drift magnitude itself.
        # float16 carries ~3 decimal digits, so the tolerance is the dtype's.
        assert score == pytest.approx(_CONTEXT_ONLY + 0.82, abs=1e-2)
        assert not any("never measured" in w for w in caught), caught

    def test_an_integer_dtype_score_is_scored_too(self):
        prioritizer = FairnessAlertPrioritizer()
        (score, severity), caught = _catch(
            prioritizer.calculate_priority, {**_FULL_EVENT, "drift_score": np.int64(1)}
        )
        assert score == pytest.approx(_CONTEXT_ONLY + 1.0)
        assert severity == "CRITICAL"
        assert caught == []

    def test_the_payload_no_longer_contradicts_itself(self):
        """The end-to-end shape of the defect: the message printed the measured
        drift score in the same sentence that said the severity could not be
        computed, and routed a CRITICAL to hand triage."""
        prioritizer = FairnessAlertPrioritizer()
        payload, caught = _catch(
            prioritizer.create_alert, {**_FULL_EVENT, "drift_score": np.float32(0.82)}
        )

        assert payload.severity == "CRITICAL"
        assert payload.routing["channel"] == "pagerduty"
        assert payload.drift_score == pytest.approx(0.82, abs=1e-6)
        assert payload.priority_score == pytest.approx(_CONTEXT_ONLY + 0.82, abs=1e-6)
        assert "NOT MEASURED" not in payload.message
        assert "0.820" in payload.message
        assert caught == []

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), None, "0.9", True, np.bool_(True), np.array(0.82)],
        ids=["nan", "inf", "none", "string", "bool", "np.bool_", "0d-array"],
    )
    def test_control_an_unmeasurable_drift_score_is_still_refused(self, value):
        """Over-correction control. Widening the dtypes accepted must not make
        an UNMEASURABLE score readable.

        The rule is ``vfairness._triage.is_measured``, the library's single
        answer to "did this actually get measured?", so these refusals are its
        refusals: NaN and the infinities are could-not-check; a string is not
        parsed, because reading a number out of text invents one; a bool is a
        flag and ``float(True) == 1.0`` would manufacture a drift magnitude out
        of it (this one used to be accepted as +1.0 here); and ``numbers.Real``
        excludes ndarray, so a 0-d array is refused as well."""
        prioritizer = FairnessAlertPrioritizer()
        (score, severity), caught = _catch(
            prioritizer.calculate_priority, {**_FULL_EVENT, "drift_score": value}
        )
        assert math.isnan(score)
        assert severity == "UNSCORED"
        assert any("never measured" in w for w in caught), caught

    def test_control_an_absent_drift_score_still_reads_as_a_floor(self):
        score, severity = None, None
        prioritizer = FairnessAlertPrioritizer()
        (score, severity), caught = _catch(prioritizer.calculate_priority, _FULL_EVENT)
        assert score == pytest.approx(_CONTEXT_ONLY)
        assert severity == "CRITICAL"
        assert any("FLOOR" in w for w in caught), caught

    def test_control_a_measured_zero_drift_score_is_a_real_zero(self):
        prioritizer = FairnessAlertPrioritizer()
        calm = {k: 0.0 for k in _FULL_EVENT if k in FairnessAlertPrioritizer._DEFAULT_WEIGHTS}
        (score, severity), caught = _catch(
            prioritizer.calculate_priority, {**calm, "drift_score": np.float32(0.0)}
        )
        assert score == 0.0 and severity == "LOW"
        assert caught == []


# ---------------------------------------------------------------------------
# Defect 2: the gate compared against a threshold nobody could use
# ---------------------------------------------------------------------------


class TestBothOperandsOfTheAlertGate:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_an_unusable_threshold_is_could_not_check_not_an_all_clear(self, bad):
        manager = AdaptiveThresholdManager(initial_threshold=bad)
        verdict, caught = _catch(manager.is_alert_warranted, "demographic_parity", 0.99)

        assert verdict is None, f"could-not-check collapsed into {verdict!r}"
        assert verdict is not False
        assert any("COULD NOT CHECK" in w for w in caught), caught
        assert any("threshold" in w for w in caught), caught

    def test_the_gate_refuses_whichever_operand_is_missing(self):
        manager = AdaptiveThresholdManager(initial_threshold=float("nan"))
        both, caught = _catch(manager.is_alert_warranted, "k", float("nan"))
        assert both is None
        assert any("COULD NOT CHECK" in w for w in caught), caught

    def test_control_a_finite_threshold_still_grades_both_ways(self):
        """Over-correction control: the gate must still answer, and answer
        correctly, whenever both operands are real. 0.7 is the documented
        default; the boundary is >=."""
        manager = AdaptiveThresholdManager()
        assert manager.get_threshold("k") == 0.7

        above, w_above = _catch(manager.is_alert_warranted, "k", 0.95)
        below, w_below = _catch(manager.is_alert_warranted, "k", 0.10)
        zero, w_zero = _catch(manager.is_alert_warranted, "k", 0.0)
        exact, w_exact = _catch(manager.is_alert_warranted, "k", 0.7)
        numpy_above, w_numpy = _catch(manager.is_alert_warranted, "k", np.float32(0.95))

        assert above is True and w_above == []
        assert below is False and w_below == []
        # A MEASURED 0.0 is a real calm reading, not a refusal.
        assert zero is False and w_zero == []
        assert exact is True and w_exact == []
        assert numpy_above is True and w_numpy == []

    def test_control_a_threshold_set_by_hand_still_moves_the_verdict(self):
        manager = AdaptiveThresholdManager()
        manager.set_threshold("k", 0.9)
        assert manager.get_threshold("k") == 0.9
        assert _catch(manager.is_alert_warranted, "k", 0.85)[0] is False
        assert _catch(manager.is_alert_warranted, "k", 0.95)[0] is True

        with pytest.raises(ValueError):
            manager.set_threshold("k", float("nan"))
        # Refused means NOT stored: the key keeps the threshold it had.
        assert manager.get_threshold("k") == 0.9


# ---------------------------------------------------------------------------
# Defect 3, OPEN: the helper that manufactures the severity inputs
# ---------------------------------------------------------------------------


class TestTheSubstitutedContextIsNamedWhereItHappens:
    def test_every_unsupplied_context_factor_is_named(self):
        event, caught = _catch(FairnessAlertPrioritizer.build_drift_event, "dp", ["B"], 0.5, 0.1)
        assert len(caught) == 1
        for factor in FairnessAlertPrioritizer._DEFAULT_WEIGHTS:
            assert factor in caught[0], (factor, caught)
        assert "ABSENT from the returned event" in caught[0]
        # BGL5 A-operations-1, 2026-09-27: the substitution itself is now CLOSED,
        # so this half moves from recording the open defect to pinning the fix.
        # The subject is unchanged (the unsupplied context is named where it
        # happens); what changed is that the event no longer CARRIES it.
        for factor in FairnessAlertPrioritizer._DEFAULT_WEIGHTS:
            assert factor not in event, (
                f"{factor} was never supplied and nobody measured it, yet the event "
                f"carries {event.get(factor)!r}, which calculate_priority cannot tell "
                f"from a measurement"
            )
        assert FairnessAlertPrioritizer().calculate_priority(event)[1] == "UNSCORED"

    def test_only_the_unsupplied_ones_are_named(self):
        _, caught = _catch(
            FairnessAlertPrioritizer.build_drift_event,
            "dp",
            ["B"],
            0.5,
            0.1,
            regulatory_risk=1.0,
            historical_discrimination=1.0,
        )
        assert len(caught) == 1
        assert "population_impact" in caught[0] and "drift_velocity" in caught[0]
        assert "regulatory_risk" not in caught[0]
        assert "historical_discrimination" not in caught[0]

    def test_control_a_fully_supplied_event_is_silent_and_unchanged(self):
        event, caught = _catch(
            FairnessAlertPrioritizer.build_drift_event,
            "equalized_odds",
            ["Black", "Female"],
            0.82,
            0.07,
            regulatory_risk=1.0,
            population_impact=0.8,
            drift_velocity=0.9,
            historical_discrimination=1.0,
            intersectional=True,
        )
        assert caught == []
        assert event["regulatory_risk"] == 1.0
        assert event["intersectional"] is True

        (score, severity), w = _catch(FairnessAlertPrioritizer().calculate_priority, event)
        # 10.35 context + 0.82 drift + 0.5 intersectional boost.
        assert score == pytest.approx(_CONTEXT_ONLY + 0.82 + 0.5)
        assert severity == "CRITICAL"
        assert w == []

    def test_control_a_measured_zero_is_not_read_as_unsupplied(self):
        """0.0 is a real answer to "is this group historically discriminated
        against" and must not be swept into the substituted set."""
        event, caught = _catch(
            FairnessAlertPrioritizer.build_drift_event,
            "dp",
            ["B"],
            0.5,
            0.1,
            regulatory_risk=0.0,
            population_impact=0.0,
            drift_velocity=0.0,
            historical_discrimination=0.0,
        )
        assert caught == []
        assert event["regulatory_risk"] == 0.0


# ---------------------------------------------------------------------------
# The routing decision: an ungraded alert never takes a graded route
# ---------------------------------------------------------------------------


class TestRoutingNeverBorrowsAGrade:
    def test_unscored_does_not_go_where_measured_calm_alerts_go(self):
        prioritizer = FairnessAlertPrioritizer()
        low = prioritizer.route_alert("LOW")
        unscored, caught = _catch(prioritizer.route_alert, "UNSCORED")

        assert low["channel"] == "jira"
        assert unscored["channel"] != low["channel"]
        assert "triage" in unscored
        assert caught == []

    def test_a_label_the_map_does_not_know_is_not_routed_as_low(self):
        prioritizer = FairnessAlertPrioritizer()
        routing, caught = _catch(prioritizer.route_alert, "BOGUS")
        assert routing["channel"] != "jira"
        assert routing == prioritizer.route_alert("UNSCORED")
        assert any("NOT to the" in w for w in caught), caught

    def test_a_custom_map_without_unscored_refuses_rather_than_guessing(self):
        prioritizer = FairnessAlertPrioritizer(
            routing_map={"CRITICAL": {"channel": "pd"}, "LOW": {"channel": "jira"}}
        )
        routing, caught = _catch(prioritizer.route_alert, "UNSCORED")
        assert routing["channel"] == "unrouted"
        assert "triage" in routing
        assert any("NO" in w for w in caught), caught

    def test_control_each_measured_band_still_reaches_its_own_channel(self):
        prioritizer = FairnessAlertPrioritizer()
        assert prioritizer.route_alert("CRITICAL")["channel"] == "pagerduty"
        assert prioritizer.route_alert("HIGH")["channel"] == "slack"
        assert prioritizer.route_alert("LOW")["channel"] == "jira"


# ---------------------------------------------------------------------------
# The log and its summary: empty because nothing happened, or because nothing
# is being kept
# ---------------------------------------------------------------------------


class TestTheAlertLogSaysWhyItIsEmpty:
    def test_retention_off_is_not_evidence_that_no_alert_was_raised(self):
        prioritizer = FairnessAlertPrioritizer(log_alerts=False)
        _catch(prioritizer.create_alert, {**_FULL_EVENT, "drift_score": 0.82})

        summary, w_summary = _catch(prioritizer.get_alert_summary)
        log, w_log = _catch(prioritizer.get_alert_log)

        assert summary["total_alerts"] == 0 and log == []
        assert any("log_alerts=False" in w for w in w_summary), w_summary
        assert any("NOT evidence" in w for w in w_summary), w_summary
        assert any("log_alerts=False" in w for w in w_log), w_log

    def test_control_an_empty_log_that_is_being_kept_says_nothing(self):
        prioritizer = FairnessAlertPrioritizer()
        summary, w_summary = _catch(prioritizer.get_alert_summary)
        log, w_log = _catch(prioritizer.get_alert_log)
        assert summary == {
            "total_alerts": 0,
            "by_severity": {},
            "acknowledged": 0,
            "unacknowledged": 0,
        }
        assert log == [] and w_summary == [] and w_log == []

    def test_an_ungraded_alert_is_counted_in_its_own_band(self):
        """An UNSCORED alert must never be counted as, or filtered into, a
        measured band: the summary is where a reader counts criticals."""
        prioritizer = FairnessAlertPrioritizer()
        _catch(prioritizer.create_alert, {**_FULL_EVENT, "drift_score": 0.82})
        _catch(prioritizer.create_alert, {"metric_name": "dp"})

        summary, _ = _catch(prioritizer.get_alert_summary)
        assert summary["total_alerts"] == 2
        assert summary["by_severity"] == {"CRITICAL": 1, "UNSCORED": 1}
        assert summary["by_severity"].get("LOW", 0) == 0
        assert len(prioritizer.get_alert_log(severity="LOW")) == 0
        assert len(prioritizer.get_alert_log(severity="UNSCORED")) == 1

    def test_control_acknowledgement_counts_are_real(self):
        prioritizer = FairnessAlertPrioritizer()
        first, _ = _catch(prioritizer.create_alert, {**_FULL_EVENT, "drift_score": 0.82})
        _catch(prioritizer.create_alert, {**_FULL_EVENT, "drift_score": 0.9})
        first.acknowledge("retraining scheduled")

        summary, _ = _catch(prioritizer.get_alert_summary)
        assert summary == {
            "total_alerts": 2,
            "by_severity": {"CRITICAL": 2},
            "acknowledged": 1,
            "unacknowledged": 1,
        }
        assert prioritizer.get_alert_log(acknowledged=True) == [first]
        assert first.resolution == "retraining scheduled"

        prioritizer.clear_log()
        assert prioritizer.get_alert_log() == []
        assert _catch(prioritizer.get_alert_summary)[0]["total_alerts"] == 0


# ---------------------------------------------------------------------------
# The payload a consumer serialises
# ---------------------------------------------------------------------------


class TestToDictKeepsTheThirdState:
    def test_an_unmeasured_field_stays_unmeasured_through_serialisation(self):
        prioritizer = FairnessAlertPrioritizer()
        payload, _ = _catch(prioritizer.create_alert, _FULL_EVENT)
        data = payload.to_dict()

        assert math.isnan(data["drift_score"]), "an unmeasured drift score became a number"
        assert math.isnan(data["mean_shift"]) is False  # mean_shift WAS supplied
        assert data["mean_shift"] == pytest.approx(0.07)
        assert "NOT MEASURED" in data["message"]

    def test_an_ungraded_alert_serialises_as_ungraded(self):
        prioritizer = FairnessAlertPrioritizer()
        payload, _ = _catch(prioritizer.create_alert, {"metric_name": "dp"})
        data = payload.to_dict()
        assert data["severity"] == "UNSCORED"
        assert math.isnan(data["priority_score"])
        assert "COULD NOT be computed" in data["message"]

    def test_control_measured_values_round_trip_exactly(self):
        prioritizer = FairnessAlertPrioritizer()
        payload, _ = _catch(prioritizer.create_alert, {**_FULL_EVENT, "drift_score": 0.123456789})
        data = payload.to_dict()
        assert data["drift_score"] == pytest.approx(0.123457, abs=1e-6)
        assert data["mean_shift"] == pytest.approx(0.07)
        assert data["severity"] == "CRITICAL"
        assert data["acknowledged"] is False


# ---------------------------------------------------------------------------
# The feedback loop: a backlog is not a verdict
# ---------------------------------------------------------------------------


class TestFeedbackCountsOnlyWhatAPersonSaid:
    def test_an_untriaged_backlog_moves_nothing_and_rates_nothing(self):
        manager = AdaptiveThresholdManager()
        for _ in range(30):
            value, caught = _catch(manager.update_from_feedback, "k", None)

        assert value == 0.7, "untriaged alerts moved the threshold"
        assert any("not a false positive" in w for w in caught), caught

        stats = manager.get_feedback_stats("k")
        assert stats["n_alerts"] == 30
        assert stats["n_untriaged"] == 30
        assert stats["n_false_positive"] == 0
        assert stats["false_positive_rate"] is None, (
            "a backlog nobody has read reported as a measured rate"
        )

    def test_control_real_feedback_still_moves_the_threshold_both_ways(self):
        """Expected values computed from the documented rule, not from a run:
        one confirmed false positive is a rate of 1.0 > 0.30, so the threshold
        rises by one learning step, 0.7 * 1.01. One confirmed-valid alert is a
        rate of 0.0 < 0.05, so it falls, 0.7 * 0.99."""
        up = AdaptiveThresholdManager()
        assert _catch(up.update_from_feedback, "k", False)[0] == pytest.approx(0.7 * 1.01)

        down = AdaptiveThresholdManager()
        assert _catch(down.update_from_feedback, "k", True)[0] == pytest.approx(0.7 * 0.99)

    def test_control_a_mixed_history_rates_over_the_triaged_only(self):
        manager = AdaptiveThresholdManager()
        for verdict in (True, None, False, None):
            _catch(manager.update_from_feedback, "k", verdict)

        stats = manager.get_feedback_stats("k")
        assert stats["n_alerts"] == 4
        assert stats["n_valid"] == 1
        assert stats["n_false_positive"] == 1
        assert stats["n_untriaged"] == 2
        assert stats["false_positive_rate"] == pytest.approx(0.5)

    def test_thresholds_and_reset_report_what_is_stored(self):
        manager = AdaptiveThresholdManager()
        assert manager.get_all_thresholds() == {}
        manager.set_threshold("a", 0.4)
        manager.set_threshold("b", 0.8)
        assert manager.get_all_thresholds() == {"a": 0.4, "b": 0.8}

        # A copy, not the live dict.
        manager.get_all_thresholds()["a"] = 99.0
        assert manager.get_threshold("a") == 0.4

        manager.reset("a")
        assert manager.get_all_thresholds() == {"b": 0.8}
        # A key with no stored threshold is at the initial one, not at zero.
        assert manager.get_threshold("a") == 0.7
        manager.reset()
        assert manager.get_all_thresholds() == {}

    def test_set_threshold_clamps_a_finite_value_into_the_band(self):
        manager = AdaptiveThresholdManager()
        manager.set_threshold("a", 5.0)
        assert manager.get_threshold("a") == 0.95
        manager.set_threshold("a", -1.0)
        assert manager.get_threshold("a") == 0.1
