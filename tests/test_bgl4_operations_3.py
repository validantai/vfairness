"""BGL4 AUDIT EXHIBITS, batch A-operations-3 (2026-09-27), INVERTED 2026-09-27.

Every assertion below started as a record of what the code did TODAY, and each
one was a DEFECT the audit overturned a grade on. They were written in the
affirmative so the suite stayed green while the evidence stayed executable, with
the instruction: when a defect is fixed, INVERT the assertion in the same change.

BGL5 fixed all seven root causes, so every assertion that recorded a defect is
now inverted to the corrected behaviour. Each test keeps its subject, its
docstring and the measured before-value, so the exhibit still reads as evidence:
what changed is the direction. The refusals and their over-correction controls
live in tests/test_bgl5_operations_3.py; this file is the audit's own record that
the defect was real and is gone.

ONE defect is NOT closed here and its exhibit still records the open state:
orchestrator.py:6761 still flattens the LL144 screen with ``bool(...)``, and
orchestrator.py is outside this batch's files (see the needs_another_batch entry
in /tmp/claude-501/bgl/fix/fix-A-operations-3.json).

Read with /tmp/claude-501/bgl/results/audit-A-operations-3.json, which carries the
commands and the verbatim output these exhibits were distilled from.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.operations.cicd.testing import (  # noqa: F401
    FairnessAssertionError,
    FairnessTestSuite,
    assert_fairness,
    fairness_test,
)
from vfairness.operations.cicd.testing import (
    TestStatus as _TestStatus,
)
from vfairness.operations.cicd.validator import DataBiasValidator, DataValidationResult
from vfairness.operations.pulse.regulatory import (
    _ll144_screen_state,
    build_recheck,
    empty_exports,
)

_N = 10
_THREE_GROUPS = np.array(["A"] * _N + ["B"] * _N + ["C"] * _N)
_Y_TRUE = np.array([1, 0] * (3 * _N // 2))
# group C's predicted selection rate is UNDEFINED (every prediction is NaN).
_Y_PRED_C_UNMEASURABLE = np.concatenate([np.zeros(_N), np.zeros(_N), np.full(_N, np.nan)])


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*a, **k)
    return value, [str(w.message) for w in rec]


# ===========================================================================
# EXHIBIT 1. builtin max()/min() swallow a NaN that is not first, so a group
# whose rate could not be measured is dropped and the remaining gap is
# published as the worst-case gap across ALL groups.
#
# _compute_default_metrics says the opposite in its own comment: "A group with
# an undefined rate (empty denominator) makes the worst-case gap itself
# unmeasurable, so it is reported as NaN and _test_metrics routes it to
# SKIPPED, never a silently-dropped group or a false 0.0."
#
# The mechanism is already pinned elsewhere in this repository, for two other
# modules, by tests/test_readiness6_names.py::test_python_max_really_does_swallow_nan
# ("pinned so nobody 'simplifies' np.maximum back to max").
# ===========================================================================


def test_exhibit1_the_mechanism():
    """max() keeps its accumulator when the comparison against NaN is False, so a
    NaN that is not the first element vanishes. np.max propagates it."""
    assert max([0.0, 0.5, float("nan")]) == 0.5
    assert min([0.0, 0.5, float("nan")]) == 0.0
    assert math.isnan(float(np.max([0.0, 0.5, float("nan")])))


def test_exhibit1_a_dropped_group_is_published_as_a_measured_zero_gap():
    """FIXED, assertion INVERTED. Honest answer: NaN, routed to SKIPPED, as the
    first-group case always was. Measured before the fix: 0.0.

    Group C's rate is undefined, so no worst-case gap over three groups exists.
    """
    suite = FairnessTestSuite(protected_attributes=["g"])
    computed = suite._compute_default_metrics(_Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS)
    assert math.isnan(computed["demographic_parity_difference"])  # was a measured 0.0


def test_exhibit1_it_is_position_dependent():
    """CONTROL that isolated the mechanism: the identical data with the
    unmeasurable group FIRST always refused, which is why the pins passed. It
    still refuses, and after the fix the LAST-group case agrees with it, so the
    answer no longer depends on where the unmeasurable group sorts."""
    y_pred_first = np.concatenate([np.full(_N, np.nan), np.zeros(_N), np.zeros(_N)])
    suite = FairnessTestSuite(protected_attributes=["g"])
    computed = suite._compute_default_metrics(_Y_TRUE, y_pred_first, _THREE_GROUPS)
    assert math.isnan(computed["demographic_parity_difference"])
    last = suite._compute_default_metrics(_Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS)
    assert math.isnan(last["demographic_parity_difference"])


def test_exhibit1_test_predictions_reports_it_as_a_passing_comparison():
    """FIXED, assertions INVERTED, on
    vfairness.operations.cicd.testing.FairnessTestSuite.test_predictions (graded
    PROVEN). Honest answer: SKIPPED with a reason. Measured before the fix:
    [('passed', 0.0)], get_summary() 'passed', warnings []."""
    suite = FairnessTestSuite(protected_attributes=["g"])
    results, _warned = _caught(
        suite.test_predictions,
        _Y_TRUE,
        _Y_PRED_C_UNMEASURABLE,
        _THREE_GROUPS,
        "g",
        raise_on_failure=False,
    )
    assert [r.status for r in results] == [_TestStatus.SKIPPED]  # was PASSED
    assert math.isnan(results[0].actual_value)  # was 0.0
    assert "could not be measured" in results[0].message
    assert suite.get_summary()["status"] == "incomplete"  # was 'passed'


def test_exhibit1_the_gate_entry_points_go_green():
    """FIXED, assertions INVERTED, on assert_fairness and on the fairness_test
    decorator (both graded PROVEN). Honest answer: both refuse, exactly as they
    do for one group. Measured before the fix: assert_fairness returned None, and
    the decorated function returned [('passed', 0.0)] without raising, so pytest
    recorded a green fairness gate."""
    with pytest.raises(FairnessAssertionError, match="NOT MEASURABLE"):
        assert_fairness(_Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS)

    @fairness_test(metric="demographic_parity_difference", threshold=0.1, protected_attribute="g")
    def probe():
        return _Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS

    with pytest.raises(FairnessAssertionError, match="NOT MEASURABLE"):
        probe()  # used to raise nothing at all


def test_exhibit1_test_model_publishes_the_same_zero():
    """FIXED, assertions INVERTED, on FairnessTestSuite.test_model (graded
    PROVEN). A model that cannot score one group's rows is the realistic shape of
    this input. Measured before the fix: [('passed', 0.0)]."""
    import pandas as pd

    class _NaNForGroupC:
        def predict(self, X):
            out = np.zeros(len(X))
            out[np.asarray(X["marker"]) == 2] = np.nan
            return out

    frame = pd.DataFrame(
        {
            "g": _THREE_GROUPS,
            "marker": [0] * _N + [1] * _N + [2] * _N,
            "y": _Y_TRUE,
        }
    )
    results = FairnessTestSuite(protected_attributes=["g"]).test_model(
        _NaNForGroupC(), frame, target_column="y", raise_on_failure=False
    )
    assert [r.status for r in results] == [_TestStatus.SKIPPED]  # was PASSED
    assert math.isnan(results[0].actual_value)  # was 0.0


# ===========================================================================
# EXHIBIT 2. build_recheck's third state could not be reached by any caller.
# Both production call sites passed a plain bool (orchestrator.py wraps the screen
# in bool(), regulatory.py hardcoded False), so `ll144_applicable is None` was
# reachable only from tests, and the collapse the branch was written for
# ("the export stage collapsed") arrived as False.
#
# FIXED IN PART. regulatory.py's own call site now passes the three-state value
# from _ll144_screen_state, a collapsed block is marked `screenRan: False` /
# `applicabilityDetermined: False`, and a non-bool marker is read as undetermined
# rather than as the statute applying. The FIRST test below still records the one
# half that is NOT closed: orchestrator.py:6761 flattens with bool() and that file
# is outside this batch.
# ===========================================================================


def test_exhibit2_the_collapse_path_publishes_a_legal_finding_nobody_made():
    """STILL OPEN in orchestrator.py, which is outside this batch's files. The
    flattening expression is orchestrator.py's, verbatim:

        bool((regulatory_exports.get("ll144") or {}).get("applicable"))

    with regulatory_exports = empty_exports(...), the collapsed block. bool()
    cannot carry a third state, so that call site still turns a collapse into the
    determined finding. Recorded here so the deferral stays executable.

    What IS fixed: the collapsed block now SAYS the screen did not run, and
    _ll144_screen_state reads it, so the one-line change at that call site is
    `_ll144_screen_state(regulatory_exports)` in place of the bool().
    """
    collapsed, _ = _caught(empty_exports, "The regulatory export stage could not be computed.")
    as_the_caller_passes_it = bool((collapsed.get("ll144") or {}).get("applicable"))
    assert as_the_caller_passes_it is False

    basis = build_recheck({}, as_the_caller_passes_it)["basis"]
    assert "No annual audit statute matched this run" in basis  # the open half

    # The fix this file's batch DID land: the block says the screen never ran, and
    # the helper the call site should use answers the third state.
    assert _ll144_screen_state(collapsed) is None
    assert "could not be determined" in build_recheck({}, _ll144_screen_state(collapsed))["basis"]
    assert "could not be determined" in build_recheck({}, None)["basis"]


def test_exhibit2_any_non_bool_marker_reads_as_the_statute_applying():
    """FIXED, assertion INVERTED, same unit: `is None` used to be the only route
    to the third state, so every other spelling of could-not-measure was truthy
    and asserted that LL144 applies. Measured before the fix: both markers below
    produced 'NYC Local Law 144 requires a bias audit ...' and a 12-month
    window."""
    for marker in (float("nan"), "unknown"):
        out, warned = _caught(build_recheck, {}, marker)
        assert "NYC Local Law 144 requires" not in out["basis"]
        assert "could not be determined" in out["basis"]
        assert any("not a statute screen verdict" in w for w in warned)


# ===========================================================================
# EXHIBIT 3. DataBiasValidator.get_explanation read only `passed` and `issues`,
# never `checks_run` / execution_coverage(), so the four-state coverage record
# that DataValidationResult carries was dropped at the surface a person reads.
#
# FIXED: get_explanation now amends the report the explainer returns whenever the
# coverage is not 'complete'. The deeper fix belongs in
# explainer._explain_validation_result, which is outside this batch (handed back
# as needs_another_batch), so FairnessExplainer.explain called DIRECTLY on a
# validation result still drops it; every caller of get_explanation is covered.
# ===========================================================================


def test_exhibit3_an_unrecorded_coverage_is_explained_as_a_clean_bill():
    """FIXED, assertions INVERTED, on
    vfairness.operations.cicd.validator.DataBiasValidator.get_explanation (graded
    SEMI-PROVEN, "already honest"). Honest answer: the report says the coverage is
    unrecorded, as repr()/to_dict()/to_junit_xml() all do.

    checks_run=None is the DEFAULT, so this is every DataValidationResult built
    anywhere other than validate(). Measured before the fix: a 1150 character
    report saying 'Validation PASSED. 0 issue(s) found' and 'All checks passed;
    proceed to training with confidence', in which the words 'could not',
    'unrecorded', 'coverage' and 'no check' appeared NOWHERE.
    """
    result = DataValidationResult(
        passed=True, issues=[], metrics={}, summary="built elsewhere", checks_run=None
    )
    assert result.execution_coverage() == "unrecorded"

    text = str(DataBiasValidator(protected_attributes=["g"]).get_explanation(result))
    assert "Validation PASSED" in text
    assert "proceed to training with confidence" not in text  # was present
    for token in ("could not", "unrecorded", "coverage"):
        assert token in text.lower(), token  # every one was absent before


def test_exhibit3_a_run_that_executed_nothing_is_explained_as_a_measured_failure():
    """FIXED, assertions INVERTED, same unit and the other direction: coverage
    'none' was rendered FAILED/HIGH with 0 issues and 'Fix all error-level
    issues', which the same report says cannot happen ("FAIL: at least one
    error-level issue detected"). The boolean stays False, deliberately (that is
    the CI fail-closed decision of 2026-09-10); what changes is that the report
    now says no check ran and drops the instruction about issues it does not have.
    """
    result = DataValidationResult(
        passed=False,
        issues=[],
        metrics={},
        summary="No validation check ran, so nothing was checked",
        checks_run=[],
    )
    assert result.execution_coverage() == "none"
    assert "COULD NOT CHECK" in repr(result)

    text = str(DataBiasValidator(protected_attributes=["g"]).get_explanation(result))
    assert "Validation FAILED. 0 issue(s) found" in text
    assert "Fix all error-level issues" not in text  # was present
    assert "NO validation check ran" in text
    assert "could not check" in text.lower()  # was absent entirely


# ===========================================================================
# EXHIBIT 4. validate_batch dropped frames in silence, because zip() truncates.
# validate() itself RAISES for the analogous misconfiguration ("A scope that names
# a missing column would check nothing while claiming to have checked it").
#
# FIXED: a name list that is not one distinct name per dataframe is refused with
# ValueError, held to the same standard as validate(), and an empty batch warns.
# ===========================================================================


@pytest.mark.parametrize(
    "names,refusal",
    [(["only_one"], r"1 name\(s\) for 3 dataframe\(s\)"), (["same", "same"], "duplicate name")],
    ids=["fewer-names-than-frames", "duplicate-names"],
)
def test_exhibit4_frames_are_dropped_from_the_batch_without_a_word(names, refusal):
    """FIXED, assertions INVERTED, on
    vfairness.operations.cicd.validator.DataBiasValidator.validate_batch (graded
    SEMI-PROVEN). Honest answer: refuse, or warn and name the frames that were
    never validated; it refuses.

    Measured before the fix: 3 frames with 1 name returned 1 result and 0
    warnings, with 2 frames never validated; 2 frames with a duplicate name
    returned 1 result and 0 warnings, the first frame's result overwritten."""
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], 120), "y": rng.integers(0, 2, 120)})
    frames = [frame] * (3 if len(names) == 1 else 2)

    with pytest.raises(ValueError, match=refusal):
        DataBiasValidator(protected_attributes=["g"]).validate_batch(
            frames, outcome_column="y", names=names
        )


# ===========================================================================
# EXHIBIT 5. compute_health_score excluded the drift component when nothing
# measured it (the graded 2026-09-27 fix) and scored the ALERT component 100.0 on
# a store whose alert channel was never fed, undisclosed. get_alerts stayed silent
# there too, because its guard asked whether ANYTHING was ingested.
#
# FIXED: the alert component is excluded and renormalised on the same rule as
# drift, and get_alerts answers the second state.
# ===========================================================================


def test_exhibit5_an_unfed_alert_channel_is_scored_as_perfect():
    """FIXED, assertions INVERTED, on MetricsStore.compute_health_score and
    .get_alerts (both graded PROVEN). Honest answer, by the rule the same function
    applies to drift two lines up: exclude alert_frequency and renormalise, or say
    it was not measured. Both, as it turns out.

    Every metric record in the window carries alert=True, so the store knew a
    breach had occurred while the alert component reported a spotless 100.0.
    Measured before the fix: get_alerts [] with no warning, score 37.5 with
    components {'metric_compliance': 0.0, 'alert_frequency': 100.0}.
    """
    from datetime import datetime, timedelta

    from vfairness.operations.reporting.store import (
        MetricsStore,
        MetricsStoreConfig,
        StoredMetricRecord,
    )

    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    now = datetime.now()
    for i in range(2):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(hours=i),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.95,
                group="gender",
                group_size=500,
                alert=True,
            )
        )
    assert not store._alert_records, "the fixture must never have fed the alert channel"

    alerts, alert_warnings = _caught(store.get_alerts)
    assert alerts == []
    assert any("never received an alert record" in w for w in alert_warnings)  # was []
    assert any("2 of 2 metric record(s)" in w for w in alert_warnings)

    health, warned = _caught(store.compute_health_score)
    assert health.components["metric_compliance"] == pytest.approx(0.0)
    assert "alert_frequency" not in health.components  # was a perfect 100.0
    assert health.score == pytest.approx(0.0)  # was 37.5, lifted by the unmeasured 30%
    assert any("alert frequency could not be measured" in w for w in warned)
    assert "Alert frequency was NOT measured" in health.explanation


# ===========================================================================
# EXHIBIT 6. ingest_window_metrics disclosed the substituted group_size in a
# WARNING only. The record kept privacy_level 'exact', and PrivacyLevel has an
# UNKNOWN_SIZE member for exactly this state.
#
# FIXED: the record carries group_size_measured=False and the k-anonymity tier
# reads unknown_size, so the caveat travels with the row and the round trip
# through ingest_dataframe no longer re-ingests it as a verified measurement. The
# VALUE is still released: what was fabricated was the guarantee, not the rate.
# ===========================================================================


def test_exhibit6_the_k_anonymity_caveat_does_not_travel_with_the_record():
    """FIXED, assertions INVERTED, on MetricsStore.ingest_window_metrics (graded
    PROVEN). Honest answer: classify those rows UNKNOWN_SIZE, which needs nothing
    from another file, rather than 'exact' plus a warning nobody downstream
    receives. Measured before the fix: both per-group rows read privacy_level
    'exact' with group_size 500, the query warned nothing, and no column said the
    size was the window's."""
    from datetime import datetime

    from vfairness.operations.monitoring.tracker import WindowMetrics
    from vfairness.operations.reporting.store import (
        MetricsStore,
        MetricsStoreConfig,
        PrivacyLevel,
    )

    snapshot = WindowMetrics(
        batch_id="b1",
        timestamp=datetime.now(),
        sample_count=500,
        metrics={"demographic_parity_gender": 0.04},
        group_rates={"gender": {"F": 0.31, "M": 0.33}},
        alerts={"demographic_parity_gender": False},
        mmd_scores={"gender": 0.02},
        excluded_groups={},
    )
    store = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    _, ingest_warnings = _caught(store.ingest_window_metrics, snapshot)
    assert any("NOT a verified k-anonymity result" in w for w in ingest_warnings)

    frame, query_warnings = _caught(store.get_metrics)
    rates = frame[frame["metric_name"] == "positive_rate"]
    assert set(rates["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}  # was EXACT
    assert set(rates["group_size"]) == {500}  # still the window's, see the docstring
    assert set(rates["group_size_measured"]) == {False}  # the column did not exist
    assert any("measured for a DIFFERENT population" in w for w in query_warnings)  # was []
    # The values are still released; the guarantee is what is withheld.
    assert rates["value"].notna().all()
