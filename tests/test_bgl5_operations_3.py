"""BGL5 PINS, batch A-operations-3 (2026-09-27).

Every test here pins a defect an independent audit OVERTURNED a grade on, with
the over-correction control beside it: the refusal must fire on the degenerate
input AND the healthy input must still get its real measured number.

Seven root causes, thirteen overturned rows:

1. ``cicd/testing.py`` ``max(dp_rates) - min(dp_rates)``: builtin max/min do not
   propagate NaN, so a group whose rate could not be measured was dropped and
   the gap over the survivors was published as the worst-case gap over ALL
   groups. Four rows: test_model, test_predictions, assert_fairness,
   fairness_test.
2. ``cicd/validator.py`` ``get_explanation``: the four-state execution coverage
   was dropped at the surface a person reads.
3. ``cicd/validator.py`` ``validate_batch``: zip() truncated, so frames were
   dropped from the batch report in silence.
4. ``experimentation/experiment.py`` ``to_dataframe`` (both entry points): an
   empty frame with no columns for a run in which every intersection was
   excluded, in silence.
5. ``experimentation/experiment.py`` ``assign_clusters``: a clamp that could not
   fire on any of 2002000 inputs, beside a delivered treatment fraction that
   still differed from the requested one in silence.
6. ``pulse/regulatory.py`` ``build_recheck``: the third state was unreachable
   from either production call site, and every non-bool marker read as the
   statute applying.
7. ``reporting/store.py``: an alert component scored 100.0 on a channel nobody
   fed, and the k-anonymity caveat on a per-group rate lived only in an
   ingest-time warning.

Read with /tmp/claude-501/bgl/results/audit-A-operations-3.json for the
auditor's commands and verbatim output, and with tests/test_bgl4_operations_3.py,
whose exhibits were inverted in the same change that landed these fixes.
"""

from __future__ import annotations

import inspect
import math
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.testing import (
    FairnessAssertionError,
    FairnessTestSuite,
    assert_fairness,
    fairness_test,
)
from vfairness.operations.cicd.testing import (
    TestStatus as _TestStatus,  # aliased: pytest tries to COLLECT a name starting with Test
)
from vfairness.operations.cicd.validator import (
    VALIDATION_CHECKS,
    DataBiasValidator,
    DataValidationResult,
)
from vfairness.operations.experimentation.experiment import (
    INTERSECTION_COLUMNS,
    ExperimentConfig,
    ExperimentResult,
    FairnessExperiment,
    IntersectionEffect,
    assign_clusters,
)
from vfairness.operations.pulse.regulatory import (
    _ll144_screen_state,
    attach_lightweight_regulatory,
    build_ll144,
    build_recheck,
    empty_exports,
)
from vfairness.operations.reporting.store import (
    MetricsStore,
    MetricsStoreConfig,
    PrivacyLevel,
    StoredMetricRecord,
)


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*a, **k)
    return value, [str(w.message) for w in rec]


def _code(obj) -> str:
    """Source of *obj* with the comment lines removed.

    A measured before-and-after comment quotes the defective expression
    verbatim, so a source assertion that greps the whole function would match
    the comment and never the code.
    """
    return "\n".join(
        line for line in inspect.getsource(obj).splitlines() if not line.lstrip().startswith("#")
    )


# ===========================================================================
# ROOT CAUSE 1. A NaN that is not first vanishes from builtin max()/min().
# Closes 4 rows: FairnessTestSuite.test_model, .test_predictions,
# assert_fairness and the fairness_test decorator.
# ===========================================================================

_N = 10
_THREE_GROUPS = np.array(["A"] * _N + ["B"] * _N + ["C"] * _N)
_Y_TRUE = np.array([1, 0] * (3 * _N // 2))
# group C's predicted selection rate is UNDEFINED (every prediction is NaN).
_Y_PRED_C_UNMEASURABLE = np.concatenate([np.zeros(_N), np.zeros(_N), np.full(_N, np.nan)])
# Three measurable groups, every rate 0.2, so the worst-case gap is a MEASURED
# 0.0 rather than an unmeasurable one: the control for every refusal below.
_Y_PRED_ALL_MEASURABLE = np.array(
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    + [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    + [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    dtype=float,
)


def test_the_mechanism_is_pinned_at_this_call_site_too():
    """Builtin max/min keep the accumulator when the comparison against NaN is
    False. Already pinned for two other modules by
    tests/test_readiness6_names.py::test_python_max_really_does_swallow_nan;
    cicd/testing.py was the call site that pin did not cover."""
    assert max([0.0, 0.5, float("nan")]) == 0.5
    assert min([0.0, 0.5, float("nan")]) == 0.0
    assert math.isnan(float(np.max([0.0, 0.5, float("nan")])))
    assert math.isnan(float(np.min([0.0, 0.5, float("nan")])))

    code = _code(FairnessTestSuite._compute_default_metrics)
    assert re.search(r"(?<!np\.)max\(dp_rates\)", code) is None, (
        "the builtin max is back over dp_rates, so a NaN rate that is not first "
        "is silently dropped again"
    )
    assert "np.max(dp_rates)" in code and "np.min(dp_rates)" in code


def test_a_group_whose_rate_is_undefined_makes_the_gap_unmeasurable():
    """REFUSAL PIN. Measured before: 0.0, a worst-case gap over three groups
    computed after dropping the one group nobody could measure."""
    computed = FairnessTestSuite(protected_attributes=["g"])._compute_default_metrics(
        _Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS
    )
    assert math.isnan(computed["demographic_parity_difference"])


def test_it_is_no_longer_position_dependent():
    """The identical data with the unmeasurable group FIRST already answered NaN,
    which is why every pre-existing pin passed. Both orders now agree."""
    first = np.concatenate([np.full(_N, np.nan), np.zeros(_N), np.zeros(_N)])
    suite = FairnessTestSuite(protected_attributes=["g"])
    assert math.isnan(
        suite._compute_default_metrics(_Y_TRUE, first, _THREE_GROUPS)[
            "demographic_parity_difference"
        ]
    )
    assert math.isnan(
        suite._compute_default_metrics(_Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS)[
            "demographic_parity_difference"
        ]
    )


def test_test_predictions_refuses_rather_than_passing_the_dropped_group():
    """REFUSAL PIN on FairnessTestSuite.test_predictions. Measured before:
    [('passed', 0.0)] with get_summary() 'passed' and no warning."""
    suite = FairnessTestSuite(protected_attributes=["g"])
    results = suite.test_predictions(
        _Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS, "g", raise_on_failure=False
    )
    assert [r.status for r in results] == [_TestStatus.SKIPPED]
    assert math.isnan(results[0].actual_value)
    assert "could not be measured" in results[0].message
    assert suite.get_summary()["status"] == "incomplete"


def test_test_model_refuses_a_model_that_cannot_score_a_group():
    """REFUSAL PIN on FairnessTestSuite.test_model, the realistic shape of this
    input. Measured before: [('passed', 0.0)]."""

    class _NaNForGroupC:
        def predict(self, X):
            out = np.zeros(len(X))
            out[np.asarray(X["marker"]) == 2] = np.nan
            return out

    frame = pd.DataFrame(
        {"g": _THREE_GROUPS, "marker": [0] * _N + [1] * _N + [2] * _N, "y": _Y_TRUE}
    )
    results = FairnessTestSuite(protected_attributes=["g"]).test_model(
        _NaNForGroupC(), frame, target_column="y", raise_on_failure=False
    )
    assert [r.status for r in results] == [_TestStatus.SKIPPED]
    assert math.isnan(results[0].actual_value)


def test_both_gate_entry_points_fail_closed_on_the_dropped_group():
    """REFUSAL PIN on assert_fairness and on the fairness_test decorator.
    Measured before: assert_fairness returned None and the decorated test raised
    nothing, so pytest recorded a green fairness gate."""
    with pytest.raises(FairnessAssertionError, match="NOT MEASURABLE"):
        assert_fairness(_Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS)

    @fairness_test(metric="demographic_parity_difference", threshold=0.1, protected_attribute="g")
    def probe():
        return _Y_TRUE, _Y_PRED_C_UNMEASURABLE, _THREE_GROUPS

    with pytest.raises(FairnessAssertionError, match="NOT MEASURABLE"):
        probe()


def test_control_three_measurable_groups_still_get_their_real_number():
    """OVER-CORRECTION CONTROL. The same three-group shape with every rate
    measurable: rates 0.2, 0.2, 0.2, so the worst-case gap is a MEASURED 0.0 and
    every surface reports it as a comparison that happened."""
    suite = FairnessTestSuite(protected_attributes=["g"])
    computed = suite._compute_default_metrics(_Y_TRUE, _Y_PRED_ALL_MEASURABLE, _THREE_GROUPS)
    assert computed["demographic_parity_difference"] == pytest.approx(0.0)

    results = suite.test_predictions(
        _Y_TRUE, _Y_PRED_ALL_MEASURABLE, _THREE_GROUPS, "g", raise_on_failure=False
    )
    assert [r.status for r in results] == [_TestStatus.PASSED]
    assert results[0].actual_value == pytest.approx(0.0)
    assert suite.get_summary()["status"] == "passed"
    assert assert_fairness(_Y_TRUE, _Y_PRED_ALL_MEASURABLE, _THREE_GROUPS) is None


def test_control_a_measured_breach_is_still_a_failure_not_a_refusal():
    """OVER-CORRECTION CONTROL, the other direction. Two measurable groups at
    rates 0.2 and 0.4 give a measured gap of 0.2, which breaches the 0.1 bound
    and must FAIL rather than come back as could-not-check."""
    groups = np.array(["A"] * _N + ["B"] * _N)
    y_true = np.array([1, 0] * _N)
    y_pred = np.array([1, 0, 0, 0, 0] * 2 + [1, 1, 0, 0, 0] * 2, dtype=float)
    suite = FairnessTestSuite(protected_attributes=["g"])
    computed = suite._compute_default_metrics(y_true, y_pred, groups)
    assert computed["demographic_parity_difference"] == pytest.approx(0.2)

    results = suite.test_predictions(y_true, y_pred, groups, "g", raise_on_failure=False)
    assert [r.status for r in results] == [_TestStatus.FAILED]
    assert results[0].actual_value == pytest.approx(0.2)


# ===========================================================================
# ROOT CAUSE 2. get_explanation dropped the four-state execution coverage.
# ===========================================================================


def test_an_unrecorded_coverage_is_no_longer_explained_as_a_clean_bill():
    """REFUSAL PIN. checks_run defaults to None, so this is every
    DataValidationResult built anywhere other than validate(). Measured before:
    a 1150 character report reading 'Validation PASSED. 0 issue(s) found' and
    'All checks passed; proceed to training with confidence', in which the words
    'could not', 'unrecorded' and 'coverage' appeared NOWHERE."""
    result = DataValidationResult(
        passed=True, issues=[], metrics={}, summary="built elsewhere", checks_run=None
    )
    assert result.execution_coverage() == "unrecorded"

    report = DataBiasValidator(protected_attributes=["g"]).get_explanation(result)
    text = str(report)
    assert "COULD NOT CHECK" in text
    assert "unrecorded" in text
    assert "proceed to training with confidence" not in text
    assert "Validation Coverage" in text
    assert any("Record which validation checks ran" in r for r in report.recommendations)
    assert report.severity == "medium", "an unrecorded coverage read as 'info'"


def test_a_run_that_executed_nothing_says_so_instead_of_naming_issues_to_fix():
    """REFUSAL PIN, the other direction. Coverage 'none' rendered FAILED/HIGH
    with 0 issues and 'Fix all error-level issues', which the same report's
    interpretation guide says cannot happen."""
    result = DataValidationResult(
        passed=False,
        issues=[],
        metrics={},
        summary="No validation check ran, so nothing was checked",
        checks_run=[],
    )
    assert result.execution_coverage() == "none"

    report = DataBiasValidator(protected_attributes=["g"]).get_explanation(result)
    text = str(report)
    assert "NO validation check ran" in text
    assert "COULD NOT CHECK" in text
    assert "Fix all error-level issues" not in text
    assert any("Enable at least one check" in r for r in report.recommendations)


def test_a_partial_coverage_names_the_checks_that_did_not_run():
    """The third state of the same record: a subset ran, so the verdict covers
    that subset and the report says which checks it does not cover."""
    result = DataValidationResult(
        passed=True,
        issues=[],
        metrics={},
        summary="partial",
        checks_run=[VALIDATION_CHECKS[0]],
    )
    assert result.execution_coverage() == "partial"
    text = str(DataBiasValidator(protected_attributes=["g"]).get_explanation(result))
    assert f"Only 1 of {len(VALIDATION_CHECKS)} validation checks ran" in text
    for name in VALIDATION_CHECKS[1:]:
        assert name in text


def test_control_a_complete_run_is_explained_exactly_as_before():
    """OVER-CORRECTION CONTROL. A real five-check validate() run is 'complete',
    and its explanation must be byte-identical to the pre-fix one: 1150
    characters, severity INFO, no coverage block, and the confident
    recommendation intact."""
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame(
        {
            "g": np.repeat(["M", "F"], n // 2),
            "y": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
        }
    )
    validator = DataBiasValidator(protected_attributes=["g"])
    result, _ = _caught(validator.validate, frame, outcome_column="y")
    assert result.execution_coverage() == "complete"

    text = str(validator.get_explanation(result))
    assert len(text) == 1150, len(text)
    assert "[INFO]" in text
    assert "Validation Coverage" not in text
    assert "All checks passed; proceed to training with confidence." in text
    assert "COULD NOT CHECK" not in text


# ===========================================================================
# ROOT CAUSE 3. validate_batch dropped frames in silence, because zip()
# truncates. validate() RAISES for the analogous misconfiguration.
# ===========================================================================


@pytest.mark.parametrize(
    "n_frames,names,expect",
    [(3, ["only_one"], "1 name(s) for 3 dataframe(s)"), (1, ["a", "b"], "2 name(s) for 1")],
    ids=["fewer-names-than-frames", "more-names-than-frames"],
)
def test_a_name_list_that_does_not_match_the_batch_is_refused(n_frames, names, expect):
    """REFUSAL PIN. Measured before: 3 frames and 1 name returned
    {'only_one': ...}, one result, zero warnings, with 2 frames never
    validated."""
    frame = _batch_frame()
    with pytest.raises(ValueError, match=re.escape(expect)):
        DataBiasValidator(protected_attributes=["g"]).validate_batch(
            [frame] * n_frames, outcome_column="y", names=names
        )


def test_duplicate_names_are_refused_rather_than_overwriting_a_result():
    """REFUSAL PIN. Measured before: keys ['same'], 1 result for 2 frames, zero
    warnings, and the FIRST frame's result silently lost."""
    frame = _batch_frame()
    with pytest.raises(ValueError, match="duplicate name"):
        DataBiasValidator(protected_attributes=["g"]).validate_batch(
            [frame, frame], outcome_column="y", names=["same", "same"]
        )


def test_an_empty_batch_says_it_validated_nothing():
    """Measured before: {} with zero warnings, which reads as a batch of clean
    datasets."""
    results, warned = _caught(
        DataBiasValidator(protected_attributes=["g"]).validate_batch, [], outcome_column="y"
    )
    assert results == {}
    assert any("validated 0 dataset(s)" in w for w in warned)


def test_control_a_well_formed_batch_still_validates_every_frame_silently():
    """OVER-CORRECTION CONTROL. One name per frame, and no names at all, both
    still return one result per dataframe with no warning."""
    frame = _batch_frame()
    named, warned = _caught(
        DataBiasValidator(protected_attributes=["g"]).validate_batch,
        [frame, frame.head(60)],
        outcome_column="y",
        names=["first", "second"],
    )
    assert set(named) == {"first", "second"}
    assert warned == []

    auto, auto_warned = _caught(
        DataBiasValidator(protected_attributes=["g"]).validate_batch,
        [frame, frame.head(60)],
        outcome_column="y",
    )
    assert set(auto) == {"dataset_0", "dataset_1"}
    assert auto_warned == []


def _batch_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": rng.choice(["a", "b"], 120), "y": rng.integers(0, 2, 120)})


# ===========================================================================
# ROOT CAUSE 4. to_dataframe returned an empty (0, 0) frame in silence for a
# run in which every intersection was EXCLUDED. Closes 2 rows.
# ===========================================================================


def _all_excluded_experiment() -> FairnessExperiment:
    def arm(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame(
            {"gender": ["F"] * 5 + ["M"] * 5, "approved": r.integers(0, 2, 10).astype(float)}
        )

    return FairnessExperiment(
        control_data=arm(1),
        treatment_data=arm(2),
        protected_attributes=["gender"],
        outcome_column="approved",
        config=ExperimentConfig(min_group_size=30, n_bootstrap=50),
    )


def test_the_tidy_frame_says_why_it_is_empty():
    """REFUSAL PIN on ExperimentResult.to_dataframe. Measured before:
    empty=True shape=(0, 0) columns=[] warnings=[] attrs={}, although the same
    object reported n_excluded=2."""
    exp = _all_excluded_experiment()
    result, _ = _caught(exp.run_full_analysis)
    assert result.n_intersections == 0 and result.n_excluded == 2

    frame, warned = _caught(result.to_dataframe)
    assert frame.empty
    assert list(frame.columns) == list(INTERSECTION_COLUMNS), "an empty frame with no schema"
    assert any("EXCLUDED" in w and "could-not-check" in w for w in warned)
    assert frame.attrs["coverage"] == {
        "n_intersections": 0,
        "n_excluded": 2,
        "measured": False,
        "notes": frame.attrs["coverage"]["notes"],
    }
    assert frame.attrs["coverage"]["measured"] is False


def test_the_experiment_entry_point_discloses_it_too():
    """REFUSAL PIN on FairnessExperiment.to_dataframe, which delegates. Measured
    before: empty=True shape=(0, 0) with no warning."""
    exp = _all_excluded_experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp.run_full_analysis()
    frame, warned = _caught(exp.to_dataframe)
    assert list(frame.columns) == list(INTERSECTION_COLUMNS)
    assert any("EXCLUDED" in w for w in warned)


def test_an_experiment_that_analysed_nothing_at_all_is_a_different_sentence():
    """The other empty state: no intersection was analysed and none excluded."""
    result = ExperimentResult(
        overall_effect=0.0,
        overall_ci=(0.0, 0.0),
        overall_p_value=1.0,
        intersection_effects=[],
        heterogeneity_detected=None,
        heterogeneity_p_value=1.0,
    )
    frame, warned = _caught(result.to_dataframe)
    assert frame.empty
    assert any("no intersection was analysed at all" in w for w in warned)


def test_the_empty_schema_cannot_drift_from_the_row_schema():
    """INTERSECTION_COLUMNS is what the empty frame carries, so it has to be the
    same key order IntersectionEffect.to_dict produces for a real row."""
    effect = IntersectionEffect(
        intersection=("F",),
        control_mean=0.1,
        treatment_mean=0.2,
        effect=0.1,
        ci_lower=0.0,
        ci_upper=0.2,
        p_value=0.04,
        effect_size_d=0.3,
        n_control=40,
        n_treatment=40,
    )
    assert list(effect.to_dict()) == list(INTERSECTION_COLUMNS)


def test_control_a_run_with_effects_is_unchanged_and_silent():
    """OVER-CORRECTION CONTROL. Two analysed intersections still give a (2, 12)
    frame with no warning, and the coverage record says it was measured."""

    def arm(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "gender": ["F"] * 200 + ["M"] * 200,
                "approved": r.integers(0, 2, 400).astype(float),
            }
        )

    exp = FairnessExperiment(
        control_data=arm(3),
        treatment_data=arm(4),
        protected_attributes=["gender"],
        outcome_column="approved",
        config=ExperimentConfig(min_group_size=30, n_bootstrap=50),
    )
    result, _ = _caught(exp.run_full_analysis)
    frame, warned = _caught(result.to_dataframe)
    assert frame.shape == (2, 12)
    assert list(frame.columns) == list(INTERSECTION_COLUMNS)
    assert warned == []
    assert frame.attrs["coverage"]["measured"] is True
    assert frame.attrs["coverage"]["n_excluded"] == 0


# ===========================================================================
# ROOT CAUSE 5. assign_clusters' clamp was dead code, and the delivered
# treatment fraction still deviated from the requested one in silence.
# ===========================================================================


def test_the_dead_clamp_is_gone_and_was_measured_dead():
    """The clamp `min(n - 1, n_treat)` could not change the answer on any input.
    Swept here as the auditor swept it: n in 1..500 against fraction 0.000..1.000
    in steps of 0.002, with the original expression beside the current one."""

    def original(n, f):
        n_treat = max(1, int(n * f))
        return min(n - 1, n_treat) if n >= 2 and f < 1.0 else n_treat

    def unclamped(n, f):
        return max(1, int(n * f))

    changes = sum(
        1
        for n in range(1, 501)
        for f in (i / 500 for i in range(501))
        if f > 0.0 and original(n, f) != unclamped(n, f)
    )
    assert changes == 0, f"the clamp fires on {changes} input(s), so it is reachable after all"

    code = _code(assign_clusters)
    assert "min(n - 1, n_treat)" not in code, (
        "a guard that cannot fire on any input is back; it reads as protection"
    )
    # The two warnings ARE load bearing and must not go with it.
    assert "COULD NOT BE PERFORMED" in code and "SINGLE ARM" in code


def test_a_canary_ramp_that_cannot_be_delivered_says_what_it_delivered():
    """REFUSAL PIN. Measured before: 20 clusters at treatment_fraction=0.01 put 1
    cluster in treatment, a delivered fraction of 0.050, with warnings=[] and
    nothing in the returned frame. A 1 percent ramp shipped 5 percent."""
    frame = pd.DataFrame({"cluster": np.repeat(np.arange(20), 20), "x": 1})
    out, warned = _caught(
        assign_clusters, frame, cluster_column="cluster", treatment_fraction=0.01, random_state=1
    )
    assert out.attrs["assignment"]["delivered_treatment_fraction"] == pytest.approx(0.05)
    assert out.attrs["assignment"]["requested_treatment_fraction"] == pytest.approx(0.01)
    assert any("DELIVERED treatment fraction is 0.050" in w and "not the 0.01" in w for w in warned)


def test_a_requested_zero_rollout_is_honoured_rather_than_inventing_an_arm():
    """REFUSAL PIN. Measured before: treatment_fraction=0.0 over 20 clusters
    delivered 1 treated cluster (max(1, 0)), a fraction of 0.050 nobody asked
    for, silently. Mirror of the >= 1.0 decision: honoured and disclosed."""
    frame = pd.DataFrame({"cluster": np.repeat(np.arange(20), 20), "x": 1})
    out, warned = _caught(
        assign_clusters, frame, cluster_column="cluster", treatment_fraction=0.0, random_state=1
    )
    assert set(out["treatment"]) == {0}
    assert out.attrs["assignment"]["n_clusters_treated"] == 0
    assert any("SINGLE ARM" in w and "every row is control" in w for w in warned)


def test_rows_with_no_cluster_id_are_named_rather_than_written_into_an_arm():
    """REFUSAL PIN. Measured before: 40 of 400 rows with a NaN cluster id came
    back treatment=0, i.e. CONTROL, with no warning at all, in both the plain
    and the stratified path."""
    plain = pd.DataFrame({"cluster": [0.0] * 180 + [1.0] * 180 + [np.nan] * 40, "x": 1})
    out, warned = _caught(
        assign_clusters, plain, cluster_column="cluster", treatment_fraction=0.5, random_state=1
    )
    assert out.attrs["assignment"]["n_rows_missing_cluster_id"] == 40
    assert any("40 of 400 row(s) carry NO cluster id" in w for w in warned)

    stratified = plain.assign(region=["n"] * 200 + ["s"] * 200)
    out2, warned2 = _caught(
        assign_clusters,
        stratified,
        cluster_column="cluster",
        treatment_fraction=0.5,
        stratify_by=["region"],
        random_state=1,
    )
    assert out2.attrs["assignment"]["n_rows_not_assignable"] == 40
    assert any("written as CONTROL (treatment=0) without being" in w for w in warned2)


def test_control_a_deliverable_fraction_is_delivered_and_stays_silent():
    """OVER-CORRECTION CONTROL. 20 clusters at 0.5 still deliver exactly 10 of
    20, and two clusters at 0.5 still fill both arms, both without a word: a
    warning on every assignment would be the over-correction."""
    frame = pd.DataFrame({"cluster": np.repeat(np.arange(20), 20), "x": 1})
    out, warned = _caught(
        assign_clusters, frame, cluster_column="cluster", treatment_fraction=0.5, random_state=1
    )
    assert out.attrs["assignment"]["delivered_treatment_fraction"] == pytest.approx(0.5)
    assert out.groupby("cluster")["treatment"].first().sum() == 10
    assert warned == []

    two = pd.DataFrame({"cluster": np.repeat(np.arange(2), 20), "x": 1})
    out2, warned2 = _caught(
        assign_clusters, two, cluster_column="cluster", treatment_fraction=0.5, random_state=1
    )
    assert set(out2["treatment"]) == {0, 1}
    assert warned2 == []


def test_control_a_deliberate_full_rollout_is_still_honoured():
    """OVER-CORRECTION CONTROL. treatment_fraction >= 1.0 remains a rollout, not
    a mistake: every cluster treated, and the single-arm warning says there is
    nothing to compare. The deviation warning must NOT fire for it."""
    frame = pd.DataFrame({"cluster": np.repeat(np.arange(4), 20), "x": 1})
    out, warned = _caught(
        assign_clusters, frame, cluster_column="cluster", treatment_fraction=1.0, random_state=1
    )
    assert set(out["treatment"]) == {1}
    assert any("SINGLE ARM" in w for w in warned)
    assert not any("DELIVERED treatment fraction" in w for w in warned)


# ===========================================================================
# ROOT CAUSE 6. build_recheck's third state was unreachable from either
# production call site, and only `is None` reached it.
# ===========================================================================


def test_the_probe_attachment_no_longer_publishes_a_statute_finding():
    """REFUSAL PIN on the only production call site inside this module, which
    hardcoded False. Measured before: attach_lightweight_regulatory wrote basis
    'No annual audit statute matched this run', a finding about the law that
    nothing on this pathway screened for."""
    result = {"data": {}}
    out, _ = _caught(attach_lightweight_regulatory, result, {"domain": "hiring"})
    basis = out["data"]["recheck"]["basis"]
    assert "could not be determined" in basis
    assert "NOT a finding that none applies" in basis
    assert "No annual audit statute matched" not in basis


@pytest.mark.parametrize(
    "marker", [float("nan"), "unknown", 1, 0, object()], ids=lambda m: type(m).__name__
)
def test_every_non_bool_marker_is_undetermined_rather_than_applying(marker):
    """REFUSAL PIN. Measured before: float('nan') and 'unknown' both published
    'NYC Local Law 144 requires a bias audit ...' with a 12-month window,
    because every non-None spelling of could-not-measure is truthy."""
    out, warned = _caught(build_recheck, {}, marker)
    assert "could not be determined" in out["basis"]
    assert "NYC Local Law 144 requires" not in out["basis"]
    assert any("not a statute screen verdict" in w for w in warned)


def test_a_collapsed_exports_block_reads_as_an_unrun_screen():
    """The orchestrator flattens the screen with
    `bool((exports.get("ll144") or {}).get("applicable"))`, which cannot express
    the third state. _ll144_screen_state can, and says None for the collapse."""
    collapsed, _ = _caught(empty_exports, "The regulatory export stage could not be computed.")
    assert bool((collapsed.get("ll144") or {}).get("applicable")) is False, "fixture drifted"
    assert _ll144_screen_state(collapsed) is None
    assert collapsed["ll144"]["applicabilityDetermined"] is False
    assert collapsed["screenRan"] is False


def test_control_a_determined_screen_still_states_its_finding_either_way():
    """OVER-CORRECTION CONTROL. A screen that RAN keeps both verdicts, from the
    real builder, and _ll144_screen_state passes them through unchanged."""
    df = pd.DataFrame({"gender": ["M", "F"] * 5, "y": [1, 0] * 5})
    y_pred = np.array([1.0, 0.0] * 5)
    applies = build_ll144(
        df, ["gender"], y_pred, df, ["gender"], "hiring", "New York City", score=None
    )
    not_applicable = build_ll144(
        df, ["gender"], y_pred, df, ["gender"], "credit scoring", "Germany", score=None
    )
    assert _ll144_screen_state({"ll144": applies}) is True
    assert _ll144_screen_state({"ll144": not_applicable}) is False

    yes, yes_warned = _caught(build_recheck, {}, _ll144_screen_state({"ll144": applies}))
    no, no_warned = _caught(build_recheck, {}, _ll144_screen_state({"ll144": not_applicable}))
    assert "NYC Local Law 144 requires a bias audit" in yes["basis"]
    assert "No annual audit statute matched this run" in no["basis"]
    assert yes_warned == [] and no_warned == []


def test_control_the_two_measured_windows_are_exactly_12_and_6_months():
    """OVER-CORRECTION CONTROL. The windows are what a compliance reader acts
    on, so they are asserted against the returned assessedAt, not assumed."""
    from vfairness.operations.pulse.regulatory import _add_months

    for applicable, months in ((True, 12), (False, 6), (None, 6)):
        out = build_recheck({"domain": "hiring"}, applicable)
        assessed = datetime.fromisoformat(out["assessedAt"].replace("Z", "+00:00"))
        until = datetime.fromisoformat(out["validUntil"].replace("Z", "+00:00"))
        assert until == _add_months(assessed, months), applicable


# ===========================================================================
# ROOT CAUSE 7a. The alert component scored 100.0 on a channel nobody fed,
# and get_alerts stayed silent there. Closes 2 rows.
# ===========================================================================


def _store_with(n_breach: int, n_clean: int, *, alerts: int = 0, drift: Optional[List[Any]] = None):
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    now = datetime.now()
    i = 0
    for _ in range(n_breach):
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
        i += 1
    for _ in range(n_clean):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(hours=i),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.01,
                group="gender",
                group_size=500,
                alert=False,
            )
        )
        i += 1
    for _ in range(alerts):
        store._alert_records.append({"timestamp": now, "severity": "HIGH", "priority_score": 0.9})
    for j, verdict in enumerate(drift or []):
        store._drift_records.append(
            {
                "timestamp": now - timedelta(hours=j),
                "metric": "demographic_parity",
                "overall_drift_score": 0.1,
                "drift_detected": verdict,
            }
        )
    return store


def test_a_store_whose_alert_channel_was_never_fed_says_so():
    """REFUSAL PIN on MetricsStore.get_alerts. The 2026-09-25 guard asked whether
    ANYTHING had been ingested, so the common case was silent. Measured before:
    a store holding two metric records that BOTH carry alert=True returned []
    with warnings=[]."""
    alerts, warned = _caught(_store_with(2, 0).get_alerts)
    assert alerts == []
    assert any("never received an alert record" in w for w in warned)
    assert any("2 of 2 metric record(s)" in w for w in warned)


def test_the_unmeasured_alert_component_no_longer_lifts_the_score():
    """REFUSAL PIN on MetricsStore.compute_health_score. Measured before: one
    breaching record in two, no alert record and no drift row gave

        score 37.5 with components {'metric_compliance': 0.0,
                                    'alert_frequency': 100.0}

    so 30 points of the composite were a constant for a component nothing had
    measured, and the store contradicted itself: every record breaching beside a
    spotless alert component."""
    health, warned = _caught(_store_with(1, 1).compute_health_score)
    assert health.components == {"metric_compliance": pytest.approx(0.0)}
    assert "alert_frequency" not in health.components
    assert health.score == pytest.approx(0.0)
    assert health.status == "red"
    assert any("never received an alert record" in w and "EXCLUDED" in w for w in warned)
    assert "Alert frequency was NOT measured" in health.explanation


def test_the_no_evidence_branch_publishes_no_alert_component_either():
    """The same defect in the branch that exists to refuse it: with every metric
    record unmeasurable the function withholds the score, and its `components`
    still carried alert_frequency 100.0 for an unfed channel."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    store._records.append(
        StoredMetricRecord(
            timestamp=datetime.now(),
            source="FairnessMonitor",
            metric_name="custom_metric",
            value=float("nan"),
            group="gender",
            group_size=500,
            alert=None,
        )
    )
    health, _ = _caught(store.compute_health_score)
    assert health.score is None and health.status == "not_assessed"
    assert health.components == {}
    assert "never received an alert record" in health.explanation


def test_control_a_fed_alert_channel_keeps_the_documented_composite():
    """OVER-CORRECTION CONTROL, and the one that matters most: a store with all
    three kinds of evidence still scores the documented 50/30/20 composite.

    Four clean metric records (metric_compliance 100.0), one alert record
    (alert_frequency 90.0) and two drift verdicts, one of them drifting
    (drift_stability 50.0):

        0.50 * 100 + 0.30 * 90 + 0.20 * 50 = 87.0
    """
    store = _store_with(0, 4, alerts=1, drift=[True, False])
    health, warned = _caught(store.compute_health_score)
    assert health.components["metric_compliance"] == pytest.approx(100.0)
    assert health.components["alert_frequency"] == pytest.approx(90.0)
    assert health.components["drift_stability"] == pytest.approx(50.0)
    assert health.score == pytest.approx(87.0)
    assert health.status == "green"
    assert not any("never received an alert record" in w for w in warned)
    assert "Alert frequency was NOT measured" not in health.explanation


def test_control_a_fed_channel_with_no_alert_in_the_window_stays_silent():
    """OVER-CORRECTION CONTROL. An alert channel that WAS connected and simply
    has no alert in this query's window is a measured absence, so get_alerts
    returns [] without a caveat."""
    store = _store_with(0, 2, alerts=1)
    alerts, warned = _caught(
        store.get_alerts,
        start_time=datetime.now() + timedelta(days=1),
        end_time=datetime.now() + timedelta(days=2),
    )
    assert alerts == []
    assert warned == []


# ===========================================================================
# ROOT CAUSE 7b. The k-anonymity caveat on a per-group rate lived only in an
# ingest-time warning; the record a consumer reads claimed a verified tier.
# ===========================================================================


def _window_snapshot():
    from vfairness.operations.monitoring.tracker import WindowMetrics

    return WindowMetrics(
        batch_id="b1",
        timestamp=datetime.now(),
        sample_count=500,
        metrics={"demographic_parity_gender": 0.04},
        group_rates={"gender": {"F": 0.31, "M": 0.33}},
        alerts={"demographic_parity_gender": False},
        mmd_scores={"gender": 0.02},
        excluded_groups={},
    )


def test_a_substituted_group_size_no_longer_claims_a_verified_tier():
    """REFUSAL PIN on MetricsStore.ingest_window_metrics. Measured before, on a
    500-row window with group_rates {'gender': {'F': 0.31, 'M': 0.33}} and
    privacy enabled, get_metrics() returned

        positive_rate gender_F 0.31 group_size 500 privacy_level exact
        positive_rate gender_M 0.33 group_size 500 privacy_level exact

    with no warning at query time and nothing in the row saying the size was the
    window's. 'exact' asserts the GROUP cleared noisy_threshold; nothing checked
    the group at all."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    _, ingest_warnings = _caught(store.ingest_window_metrics, _window_snapshot())
    assert any("NOT a verified k-anonymity result" in w for w in ingest_warnings)

    frame, query_warnings = _caught(store.get_metrics)
    rates = frame[frame["metric_name"] == "positive_rate"]
    assert set(rates["privacy_level"]) == {PrivacyLevel.UNKNOWN_SIZE.value}
    assert set(rates["group_size_measured"]) == {False}
    assert any("measured for a DIFFERENT population" in w for w in query_warnings)
    # The VALUES are still released: what is withheld is the guarantee.
    assert sorted(rates["value"]) == pytest.approx([0.31, 0.33])


def test_the_window_level_rows_keep_their_own_measured_size():
    """The other half of the same record: a window-level metric and an MMD score
    ARE computed over the whole window, so its count is theirs and the tier
    stays exact. READINESS-6 restored that count deliberately."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())
    frame, _ = _caught(store.get_metrics)
    window_rows = frame[frame["metric_name"].isin(["demographic_parity", "mmd_score"])]
    assert set(window_rows["group_size_measured"]) == {True}
    assert set(window_rows["privacy_level"]) == {PrivacyLevel.EXACT.value}
    assert window_rows["value"].notna().all()


def test_the_round_trip_can_no_longer_launder_the_caveat():
    """ingest_dataframe refuses rows whose privacy_level is not 'exact', so with
    the tier corrected the exported per-group rates can no longer be re-ingested
    as verified measurements. Before the fix they carried 'exact' and were."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_window_metrics(_window_snapshot())
        frame = store.get_metrics()
    rates = frame[frame["metric_name"] == "positive_rate"]

    clone = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    ingested, warned = _caught(
        clone.ingest_dataframe, rates, metric_col="metric_name", group_col="group"
    )
    assert ingested == 0
    assert any("did not release" in w or "k-anonymity" in w for w in warned)


def test_control_a_real_group_size_still_releases_as_exact():
    """OVER-CORRECTION CONTROL. A row ingested with its OWN group size, large
    enough to clear noisy_threshold, is still released exactly and still says the
    size was measured: the fix must not withhold a guarantee that was checked."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=True))
    ingested, _ = _caught(
        store.ingest_dataframe,
        pd.DataFrame(
            {
                "timestamp": [datetime.now()] * 2,
                "metric": ["demographic_parity"] * 2,
                "value": [0.31, 0.33],
                "group": ["F", "M"],
                "group_size": [240, 260],
            }
        ),
        group_col="group",
        group_size_col="group_size",
    )
    assert ingested == 2
    frame, warned = _caught(store.get_metrics)
    assert set(frame["privacy_level"]) == {PrivacyLevel.EXACT.value}
    assert frame["value"].notna().all()
    assert not any("DIFFERENT population" in w for w in warned)
