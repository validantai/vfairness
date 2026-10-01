"""BGL5, batch A-preprocessing-2: the twelve overturned rows, closed and pinned.

Every test here FAILS against the source as it stood on 2026-09-27 and passes
against the repaired source, which is the opposite of its predecessor
``tests/test_bgl4_preprocessing_2.py``: that file recorded the defects by
pinning the fabricated values, and its assertions have been inverted in place.

Each defect is paired with an over-correction control that asserts a REAL
measured number on healthy input, because a fix that refuses everything passes
every refusal test and destroys the library. The controls carry the actual
values, not just "is not None".

The four detector rows (``assessment_coverage``, ``execution_coverage``,
``to_svg``, ``empty_is_not_a_measurement``) share ONE root cause and ONE fix
point: "complete" was spoken over a report that records no assessment half at
all, and every consumer, including ``rendering.adapters``, reads that word as a
licence.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.detector import (
    AUDIT_MODULES,
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_UNASSESSED,
    COVERAGE_UNRECORDED,
    MIN_ASSESSABLE_GROUP_SIZE,
    BiasAuditReport,
    BiasDetector,
)
from vfairness.preprocessing.bias_detection.statistical import (
    detect_specification_bias,
    detect_temporal_drift,
)
from vfairness.preprocessing.feature_engineering.correlation import (
    FeatureCorrelationMatrix,
    analyze_intersectional_correlations,
    compute_feature_correlations,
    compute_pearson_correlation_matrix,
    identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.data_balancing import SyntheticResampler


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


def _joined(messages) -> str:
    return " ".join(messages)


def _report(**over) -> BiasAuditReport:
    base = dict(
        timestamp="2026-09-27T00:00:00",
        dataset_info={"n_rows": 500},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return BiasAuditReport(**base)


def _visible(svg: str) -> str:
    """Every rendered text node of the canvas, which is what a reader sees."""
    return " | ".join(t.strip() for t in re.findall(r">([^<>]+)<", svg) if t.strip())


# ===========================================================================
# 1. THE FOUR IN ONE. detector.BiasAuditReport: assessment_coverage,
#    execution_coverage, to_svg, empty_is_not_a_measurement.
# ===========================================================================


def test_execution_coverage_withholds_complete_when_the_assessment_half_is_unrecorded():
    """Its own docstring defines "complete" as every module ran AND at least one
    protected attribute was actually assessed. A report that records the first
    half and not the second establishes no such thing.

    Before: ``execution_coverage()`` returned ``'complete'`` while
    ``assessment_coverage()`` said ``'unrecorded'`` in the same breath.
    """
    report = _report(modules_run=list(AUDIT_MODULES))

    assert report.assessment_coverage() == COVERAGE_UNRECORDED
    assert report.execution_coverage() == COVERAGE_UNRECORDED


def test_the_bgl_s2_observation_count_fallback_reads_min_assessable_group_size():
    """Two rows no longer buy a clean bill of health on the legacy branch.

    ``MIN_ASSESSABLE_GROUP_SIZE`` is defined in that module for this decision,
    with a comment naming the very defect ("a 2-row frame reported
    execution_coverage() complete"), and the fallback tested ``bool(n)``, i.e.
    n >= 1, without ever consulting it.

    Before: ``assessment_coverage()`` 'complete', ``execution_coverage()``
    'complete', ``empty_is_not_a_measurement()`` None,
    ``unassessable_attributes()`` [] and an SVG carrying 'MINIMAL' with no
    'NOT ASSESSABLE' anywhere.
    """
    report = _report(
        dataset_info={"n_rows": 2},
        modules_run=list(AUDIT_MODULES),
        attribute_observations={"gender": 2},
    )

    assert report.attribute_assessed is None
    assert 2 < MIN_ASSESSABLE_GROUP_SIZE
    assert report.assessment_coverage() == COVERAGE_NONE
    assert report.execution_coverage() == COVERAGE_UNASSESSED
    # The refusal names the attribute that was requested and not assessed. It
    # used to read "(none were requested)", because unassessable_attributes()
    # applied `if not n` to the same count.
    assert report.unassessable_attributes() == ["gender"]
    assert "gender" in report.empty_is_not_a_measurement()


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        ({"gender": 0}, COVERAGE_NONE),
        ({"gender": 2}, COVERAGE_NONE),
        ({"gender": 29}, COVERAGE_NONE),
        ({"gender": 30}, COVERAGE_COMPLETE),
        ({"gender": 40}, COVERAGE_COMPLETE),
        ({"gender": 500}, COVERAGE_COMPLETE),
        ({"gender": 500, "age": 2}, "partial"),
        # A free-form mapping on a hand-built report can hold anything, and an
        # unreadable count is a could-not-check, never a licence.
        ({"gender": None}, COVERAGE_NONE),
        ({"gender": "no"}, COVERAGE_NONE),
        ({"gender": -5}, COVERAGE_NONE),
        ({"gender": True}, COVERAGE_NONE),
        ({}, COVERAGE_NONE),
    ],
)
def test_the_observation_count_fallback_over_the_whole_bound(observations, expected):
    """The edge sweep of the fallback, at and either side of the bound."""
    assert _report(attribute_observations=observations).assessment_coverage() == expected


def test_empty_is_not_a_measurement_names_the_unrecorded_assessment_half():
    """The residual that was recorded rather than changed.

    ``empty_is_not_a_measurement`` returned None, whose documented meaning is
    "this IS a measurement", so ``get_critical_count()`` published 0 with no
    warning. The reason string also has to be TRUE: the report records all four
    modules, so "this report does not record which audit modules executed"
    would be the wrong sentence for this state.
    """
    report = _report(modules_run=list(AUDIT_MODULES))

    reason = report.empty_is_not_a_measurement()
    assert reason is not None
    assert "does not record whether any requested protected attribute was assessed" in reason
    count, messages = _caught(report.get_critical_count)
    assert count is None
    assert "no count of critical issues is reported" in _joined(messages)


def test_the_canvas_says_not_assessable_and_carries_a_could_not_check_token():
    """The R-11 canvas, reached through the unrecorded record.

    Before: the rendered text read 'OVERALL RISK | 0% | MINIMAL | Historical |
    0.00 | 0 patterns | ...' with no could-not-check token anywhere, because
    ``rendering.adapters`` keys on ``coverage == "complete" and bool(protected)``.
    The fix point is upstream of ``to_svg``: the same word the adapter reads.
    """
    svg, _ = _caught(_report(modules_run=list(AUDIT_MODULES)).to_svg)
    seen = _visible(svg)

    assert "NOT ASSESSABLE" in seen
    assert "COULD NOT CHECK" in svg.upper()
    assert "MINIMAL" not in seen
    assert "0%" not in seen


def test_a_legacy_two_row_report_renders_no_measured_module_tile():
    """The same canvas through the count fallback: four '0.00' tiles are gone."""
    svg, _ = _caught(
        _report(
            dataset_info={"n_rows": 2},
            modules_run=list(AUDIT_MODULES),
            attribute_observations={"gender": 2},
        ).to_svg
    )

    assert "NOT ASSESSABLE" in _visible(svg)
    assert svg.count(">0.00<") == 0


# ── over-correction controls for the four in one ────────────────────────────


def _clean_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 900
    return pd.DataFrame(
        {
            "cohort": rng.choice(["a", "b"], size=n),
            "score_x": rng.normal(0, 1, n),
            "approved": rng.integers(0, 2, n),
        }
    )


def test_control_a_real_full_audit_still_earns_its_measured_all_clear():
    """The producer records BOTH halves, so nothing about a real audit changed.

    This is the control that decides whether the fix is a refusal machine: 900
    rows, two cohorts, four modules, no finding. It must still be a measurement.
    """
    report = BiasDetector(
        _clean_frame(), protected_attributes=["cohort"], outcome_column="approved"
    ).full_audit()

    assert report.modules_run == list(AUDIT_MODULES)
    assert report.attribute_assessed == {"cohort": True}
    assert report.attribute_observations == {"cohort": 900}
    assert report.assessment_coverage() == COVERAGE_COMPLETE
    assert report.execution_coverage() == COVERAGE_COMPLETE
    assert report.empty_is_not_a_measurement() is None
    assert report.overall_risk_score == 0.0
    count, messages = _caught(report.get_critical_count)
    assert count == 0, "a complete audit that found nothing measured zero"
    assert messages == []

    seen = _visible(report.to_svg())
    assert "0%" in seen and "MINIMAL" in seen
    assert "NOT ASSESSABLE" not in seen


def test_control_an_audit_with_a_real_gap_still_reports_its_real_number():
    """The strongest control: an audit that MEASURED something keeps it.

    500 rows approving 85% of 'm' against 15% of 'f'. A fix that withholds too
    much would show up here as a withheld score or a withheld critical count.
    """
    rng = np.random.default_rng(7)
    n = 500
    # Drawn in explicit statements, never inside a dict literal: the literal
    # decides the draw ORDER, so the same seed gave a different frame and a
    # different risk score when the columns were reordered.
    g = rng.choice(["m", "f"], n)
    approved = np.where(g == "m", rng.random(n) < 0.85, rng.random(n) < 0.15).astype(int)
    score = rng.normal(0, 1, n)
    frame = pd.DataFrame({"gender": g, "score": score, "approved": approved})

    report, _ = _caught(
        BiasDetector(frame, protected_attributes=["gender"], outcome_column="approved").full_audit
    )

    assert report.execution_coverage() == COVERAGE_COMPLETE
    assert report.empty_is_not_a_measurement() is None
    assert report.overall_risk_score == pytest.approx(0.5, abs=1e-6)
    assert report.get_critical_count() == 2
    seen = _visible(report.to_svg())
    assert "50%" in seen and "MEDIUM" in seen
    assert "NOT ASSESSABLE" not in seen


def test_control_a_legacy_report_with_a_real_count_still_reads_complete():
    """The fallback still grants the word when the count clears the bound."""
    report = _report(modules_run=list(AUDIT_MODULES), attribute_observations={"gender": 500})

    assert report.assessment_coverage() == COVERAGE_COMPLETE
    assert report.execution_coverage() == COVERAGE_COMPLETE
    assert report.empty_is_not_a_measurement() is None
    assert report.unassessable_attributes() == []
    assert "MINIMAL" in _visible(report.to_svg())


def test_the_two_coverage_answers_still_cannot_contradict_each_other():
    """complete over here requires a non-refusal over there, in every state."""
    for modules in (None, [], ["historical"], list(AUDIT_MODULES)):
        for assessed in (None, {}, {"gender": True}, {"gender": False}):
            for observations in (None, {}, {"gender": 2}, {"gender": 500}):
                report = _report(
                    modules_run=modules,
                    attribute_assessed=assessed,
                    attribute_observations=observations,
                )
                if report.execution_coverage() == COVERAGE_COMPLETE:
                    assert report.assessment_coverage() in (COVERAGE_COMPLETE, "partial")


# ===========================================================================
# 2. detect_temporal_drift: fillna("missing") invented a stable group
# ===========================================================================


def _dated(values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "gender": values,
        }
    )


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("all-None", [None] * 100),
        ("all-NaN", [np.nan] * 100),
        ("empty", []),
    ],
)
def test_a_protected_column_with_no_observation_is_not_assessed_rather_than_stable(label, values):
    """``fillna("missing")`` invents ONE category, so PSI comes out 0.00.

    Before, for all three: ``{'available': True, 'severity': 'pass', 'drift':
    [{'attribute': 'gender', 'compositionPSI': 0.0, 'severity': 'pass',
    'plain': 'Across application_date, the gender mix is stable (PSI 0.00)'}],
    'summary': 'No material temporal drift in the assessed attributes over
    application_date'}`` and ``warnings == []``. "Stable" is the one verdict on
    this scale that claims the opposite of a could-not-check.
    """
    frame = (
        _dated(values)
        if values
        else pd.DataFrame(
            {"application_date": pd.to_datetime([]), "gender": pd.Series([], dtype=object)}
        )
    )

    result, messages = _caught(lambda: detect_temporal_drift(frame, ["gender"]))

    assert result["available"] is False
    assert result["drift"] == []
    assert result["attributesWithNoObservation"] == ["gender"]
    assert "no recorded value in either the earliest or the latest period" in result["reason"]
    assert "nothing was compared, so nothing is clean" in result["reason"]
    assert "severity" not in result, "an unavailable screen must not publish a verdict"
    assert "no protected attribute could be assessed" in _joined(messages)


def test_an_unobserved_attribute_beside_a_measured_one_is_named_not_dropped():
    """Partial coverage: the measured attribute keeps its number and the
    unobserved one is named where a reader will see it."""
    frame = _dated(["a"] * 50 + ["b"] * 50)
    frame["region"] = [None] * 100

    result, messages = _caught(lambda: detect_temporal_drift(frame, ["gender", "region"]))

    assert result["available"] is True
    assert [d["attribute"] for d in result["drift"]] == ["gender"]
    assert result["attributesWithNoObservation"] == ["region"]
    assert "carry no recorded value" in _joined(messages)


def test_control_temporal_drift_still_measures_a_real_shift_and_a_real_pass():
    """Over-correction control, with the actual PSI on both sides."""
    drifting, messages = _caught(
        lambda: detect_temporal_drift(
            _dated(["a"] * 80 + ["b"] * 20 + ["a"] * 20 + ["b"] * 80), ["gender"]
        )
    )
    assert drifting["available"] is True
    assert drifting["severity"] == "critical"
    assert drifting["drift"][0]["compositionPSI"] == pytest.approx(1.6636, abs=1e-4)
    assert "Temporal drift risk" in drifting["summary"]
    assert messages == []

    stable, quiet = _caught(lambda: detect_temporal_drift(_dated(["a", "b"] * 100), ["gender"]))
    assert stable["severity"] == "pass"
    assert stable["drift"][0]["compositionPSI"] == 0.0
    assert stable["attributesWithNoObservation"] == []
    assert quiet == []


def test_control_a_mostly_observed_column_is_still_measured():
    """90% observed in both cohorts is a measurement, not a refusal."""
    values = (["a", "b"] * 45 + [None] * 10) * 2
    result, messages = _caught(lambda: detect_temporal_drift(_dated(values), ["gender"]))

    assert result["available"] is True
    assert result["severity"] == "pass"
    assert result["drift"][0]["compositionPSI"] == 0.0
    assert messages == []


# ===========================================================================
# 3. detect_specification_bias: three CRITICALs nobody measured
# ===========================================================================


_SPEC_FRAME = pd.DataFrame(
    {"x": np.linspace(0, 1, 50), "cost": np.linspace(0, 1, 50), "approved": [1, 0] * 25}
)


def test_an_empty_frame_does_not_produce_a_critical_circular_evaluation_finding():
    """Two EMPTY Series compare equal, so the screen published its worst finding
    over zero rows.

    Before: ``{'available': True, 'severity': 'critical', 'findings':
    [{'kind': 'circular_evaluation', ...}], 'checksRun': ['proxy_target',
    'circular_evaluation']}`` with no warning.
    """
    empty = pd.DataFrame(
        {
            "approved": pd.Series([], dtype=float),
            "pred": pd.Series([], dtype=float),
            "x": pd.Series([], dtype=float),
        }
    )

    result, messages = _caught(
        lambda: detect_specification_bias(empty, outcome="approved", prediction="pred")
    )

    assert [f["kind"] for f in result["findings"]] == []
    assert result["severity"] == "pass"
    assert "circular_evaluation" not in result["checksRun"]
    not_run = _joined(result["checksNotRun"])
    assert "circular_evaluation: only 0 of 0 row(s)" in not_run
    assert "could not run" in _joined(messages)


def test_an_outcome_and_prediction_that_are_both_entirely_null_are_not_called_equal():
    """``astype(str)`` maps every missing value to the string 'nan', so two
    columns nobody recorded a value in compared equal over 50 rows.

    Before: severity 'critical' with a circular_evaluation finding.
    """
    frame = pd.DataFrame({"approved": [np.nan] * 50, "pred": [np.nan] * 50, "x": np.arange(50.0)})

    result, _ = _caught(
        lambda: detect_specification_bias(frame, outcome="approved", prediction="pred")
    )

    assert [f["kind"] for f in result["findings"]] == []
    assert result["severity"] == "pass"
    assert "circular_evaluation: only 0 of 50 row(s)" in _joined(result["checksNotRun"])


def test_two_rows_do_not_make_every_numeric_feature_a_critical_leak():
    """Any two points are perfectly collinear, so |r| was 1.00 by construction.

    Before: severity 'critical', findings ['target_leakage', 'target_leakage'],
    both plain sentences reading '|r|=1.00', and checksRun claiming
    'target_leakage (2 feature column(s) scanned)'.
    """
    frame = pd.DataFrame({"approved": [1, 0], "x": [5.0, 9.0], "z": [2.0, 3.0]})

    result, messages = _caught(lambda: detect_specification_bias(frame, outcome="approved"))

    assert [f["kind"] for f in result["findings"]] == []
    assert result["severity"] == "pass"
    assert not any("scanned" in c for c in result["checksRun"])
    not_run = _joined(result["checksNotRun"])
    assert "x (n=2)" in not_run and "z (n=2)" in not_run
    assert "could not run" in _joined(messages)


def test_control_the_screen_still_finds_a_proxy_target_and_a_real_leak():
    """Over-correction control: 50 rows, both findings, silent."""
    result, messages = _caught(
        lambda: detect_specification_bias(_SPEC_FRAME, outcome="cost", prediction="approved")
    )

    assert messages == []
    assert result["severity"] == "critical"
    assert sorted(f["kind"] for f in result["findings"]) == ["proxy_target", "target_leakage"]
    assert result["checksRun"] == [
        "proxy_target",
        "target_leakage (1 feature column(s) scanned)",
        "circular_evaluation",
    ]
    assert result["checksNotRun"] == []
    assert "|r|=1.00" in _joined([f["plain"] for f in result["findings"]])


def test_control_a_real_circular_evaluation_is_still_critical():
    """50 rows in which the outcome column genuinely IS the prediction column."""
    frame = pd.DataFrame({"y": [1, 0] * 25, "pred": [1, 0] * 25, "f": np.linspace(0, 3, 50)})

    result, messages = _caught(
        lambda: detect_specification_bias(frame, outcome="y", prediction="pred")
    )

    assert result["severity"] == "critical"
    assert [f["kind"] for f in result["findings"]] == ["circular_evaluation"]
    assert "circular_evaluation" in result["checksRun"]
    assert messages == []


# ===========================================================================
# 4. compute_pearson_correlation_matrix: a diagonal p-value of 0.0 for a
#    coefficient the same call refused to compute
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "frame"),
    [
        ("constant column", pd.DataFrame({"a": [5.0] * 40, "b": list(np.arange(40.0))})),
        (
            "3 non-null rows against min_periods=10",
            pd.DataFrame({"a": [1.0, 2.0, 3.0] + [np.nan] * 20, "b": list(np.arange(23.0))}),
        ),
    ],
)
def test_the_two_matrices_agree_on_a_cell_neither_could_compute(label, frame):
    """A p-value of 0.0 is the STRONGEST assertion of an association the test
    can make, and it was published on the diagonal of a column whose
    self-correlation the same call had refused.

    Before, for both frames: ``corr.loc['a','a']`` NaN and
    ``pvals.loc['a','a'] == 0.0``, so ``(pvals < 0.05).loc['a','a']`` was True
    and the docstring's own ``corr[pvals < 0.05]`` selected it.
    """
    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(frame, return_pvalues=True)
    )

    assert np.isnan(corr.loc["a", "a"])
    assert np.isnan(pvals.loc["a", "a"]), "the p-value must agree with the coefficient"
    assert not bool((pvals < 0.05).loc["a", "a"])
    assert "NOT a coefficient of 0 or a p-value of" in _joined(messages)


def test_a_constant_column_pair_is_named_by_a_vfairness_warning():
    """The docstring promises "a warning names every pair it applies to", and the
    only notice for a constant column was scipy's ConstantInputWarning, which
    names neither column."""
    frame = pd.DataFrame({"a": [5.0] * 40, "b": list(np.arange(40.0))})

    _, messages = _caught(lambda: compute_pearson_correlation_matrix(frame, return_pvalues=True))

    joined = _joined(messages)
    assert "a ~ b (n=40)" in joined
    assert "no variation" in joined


def test_control_a_real_pearson_matrix_keeps_its_numbers():
    """Over-correction control, with the measured coefficient and p-value."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    frame = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.1, 100)})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(frame, return_pvalues=True)
    )

    assert corr.loc["a", "b"] == pytest.approx(0.9988, abs=5e-4)
    assert corr.loc["a", "a"] == 1.0
    assert pvals.loc["a", "a"] == 0.0, "a populated column against itself needs no test"
    assert pvals.loc["a", "b"] < 0.05
    assert bool((pvals < 0.05).loc["a", "b"])
    assert messages == []


# ===========================================================================
# 5. analyze_intersectional_correlations
# ===========================================================================


def _inter_frame(x_second_half) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "gender": ["m"] * 40 + ["f"] * 40,
            "x": list(rng.normal(50, 5, 40)) + list(x_second_half),
            "y": list(rng.normal(10, 1, 80)),
        }
    )


def test_coverage_is_not_complete_when_a_feature_was_never_measured_in_one_group():
    """The disclosure was per GROUP and the drop is per FEATURE.

    Before: ``coverage == 'complete'``, ``groups_not_assessed == []``,
    ``warnings == []``, ``within_group_correlations`` {'m': ['x','y'], 'f':
    ['y']} and ``between_group_differences`` holding ONLY y. Feature x was
    measured in one group, was unmeasurable in the other, and vanished from the
    between-group answer with the coverage word still reading complete.
    """
    result, messages = _caught(
        lambda: analyze_intersectional_correlations(_inter_frame([np.nan] * 40), ["gender"])
    )

    assert result["coverage"] == "partial"
    assert result["groups_not_assessed"] == []
    assert [(e["feature"], e["group"]) for e in result["features_not_assessed"]] == [("x", "f")]
    assert [e["feature"] for e in result["features_not_compared"]] == ["x"]
    assert "x" not in result["between_group_differences"]
    assert "y" in result["between_group_differences"], "y was measured in both groups"
    assert "could-not-check, not a finding that the groups agree" in _joined(messages)


def test_rows_with_no_recorded_group_are_not_a_group_and_drive_no_difference():
    """The absence of a record was stringified into the group key 'None' and then
    measured against a real group.

    Before: ``intersectional_groups == ['m', 'None']``, ``coverage 'complete'``,
    ``warnings == []`` and
    ``between_group_differences['x']['max_difference'] == 31.43``, a difference
    between a group and the absence of a record. ``_fit_propensity_model`` in
    this same package refuses that input by name.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 40 + [None] * 40,
            "x": list(rng.normal(50, 5, 40)) + list(rng.normal(80, 5, 40)),
        }
    )

    result, messages = _caught(lambda: analyze_intersectional_correlations(frame, ["gender"]))

    assert result["intersectional_groups"] == ["m"]
    assert "None" not in result["group_sizes"]
    assert result["rows_without_a_recorded_group"] == 40
    assert result["between_group_differences"] == {}
    assert result["coverage"] != "complete"
    joined = _joined(messages)
    assert "have no recorded value" in joined
    assert "They are NOT a group" in joined


def test_control_two_fully_measured_groups_keep_their_real_difference():
    """Over-correction control, with the actual max_difference and CV."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 40 + ["f"] * 40,
            "x": list(rng.normal(50, 5, 40)) + list(rng.normal(80, 5, 40)),
            "y": list(rng.normal(10, 1, 80)),
        }
    )

    result, messages = _caught(lambda: analyze_intersectional_correlations(frame, ["gender"]))

    assert result["coverage"] == "complete"
    assert result["features_not_assessed"] == []
    assert result["features_not_compared"] == []
    assert result["rows_without_a_recorded_group"] == 0
    assert result["between_group_differences"]["x"]["max_difference"] == pytest.approx(
        29.4602, abs=1e-3
    )
    assert result["between_group_differences"]["x"]["coefficient_of_variation"] == pytest.approx(
        0.2283, abs=1e-3
    )
    assert messages == []


def test_control_the_negative_mean_coefficient_of_variation_is_still_measured():
    """The CV fix of the previous wave still holds: four negative group means
    produce a real CV beside a real max_difference."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 200 + ["f"] * 200,
            "race": (["w"] * 100 + ["b"] * 100) * 2,
            "score": (
                list(rng.normal(-99.94, 1, 100))
                + list(rng.normal(-99.98, 1, 100))
                + list(rng.normal(-1.0, 1, 100))
                + list(rng.normal(-1.03, 1, 100))
            ),
        }
    )

    result, messages = _caught(
        lambda: analyze_intersectional_correlations(frame, ["gender", "race"])
    )

    assert result["coverage"] == "complete"
    assert result["between_group_differences"]["score"][
        "coefficient_of_variation"
    ] == pytest.approx(0.9787, abs=1e-3)
    assert result["between_group_differences"]["score"]["max_difference"] == pytest.approx(
        98.8485, abs=1e-3
    )
    assert messages == []


# ===========================================================================
# 6. identify_proxy_variables: a requested feature that is not a column
# ===========================================================================


def _proxy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 200
    g = rng.choice(["m", "f"], n)
    return pd.DataFrame(
        {
            "gender": g,
            "height": np.where(g == "m", rng.normal(180, 5, n), rng.normal(165, 5, n)),
            "noise": rng.normal(0, 1, n),
        }
    )


@pytest.mark.parametrize(
    ("features", "expected_skipped"),
    [
        (["not_a_column"], ["not_a_column"]),
        # The misspelling is the name carrying the critical proxy.
        (["heigth", "noise"], ["heigth"]),
    ],
)
def test_a_requested_feature_that_is_not_a_column_is_recorded(features, expected_skipped):
    """``ProxyScreenResult.complete`` is documented as "True only when every
    requested pair was actually screened", and zero pairs had been screened.

    Before: 0 results, ``complete is True``, ``screens_not_run == []`` and no
    warning, while ``feature_columns=None`` on the same frame finds 'height' as
    a CRITICAL proxy at 0.819.
    """
    result, messages = _caught(
        lambda: identify_proxy_variables(_proxy_frame(), ["gender"], feature_columns=features)
    )

    assert result.complete is False
    assert [e["feature"] for e in result.screens_not_run] == expected_skipped
    assert "is not a column of this DataFrame" in _joined(
        e["reason"] for e in result.screens_not_run
    )
    assert "were NOT SCREENED" in _joined(messages)


def test_control_the_proxy_screen_still_finds_the_critical_proxy():
    """Over-correction control, with the measured correlation."""
    for features in (None, ["height"]):
        result, _ = _caught(
            lambda: identify_proxy_variables(_proxy_frame(), ["gender"], feature_columns=features)
        )
        assert result.complete is True
        assert [r.feature for r in result] == ["height"]
        assert result[0].risk_level.value == "critical"
        assert result[0].correlation == pytest.approx(0.819, abs=1e-3)


# ===========================================================================
# 7. get_high_correlations: a threshold the check cannot fire against
# ===========================================================================


def _matrix() -> FeatureCorrelationMatrix:
    return FeatureCorrelationMatrix(
        correlations=pd.DataFrame({"gender": [0.85, 0.02]}, index=["height", "noise"]),
        pvalues=pd.DataFrame({"gender": [0.001, 0.9]}, index=["height", "noise"]),
        feature_names=["height", "noise"],
        protected_attributes=["gender"],
        method="mixed",
    )


@pytest.mark.parametrize("threshold", [float("nan"), 2.0])
def test_a_threshold_that_cannot_fire_is_disclosed_not_answered_with_an_empty_list(
    threshold,
):
    """Every comparison against NaN is False, so a MEASURED 0.85 was dropped in
    exactly the silence the NaN-cell fix was written to end.

    Before, for ``threshold=float('nan')`` on a matrix holding a measured 0.85:
    ``[]`` with ``complete == True``, ``pairs_not_measured == []`` and no
    warning. A threshold above 1.0 is the same vacuum from the other side.
    """
    result, messages = _caught(lambda: _matrix().get_high_correlations(threshold=threshold))

    assert list(result) == []
    assert result.complete is False
    assert result.not_compared_reason is not None
    assert sorted(result.pairs_not_measured) == [("height", "gender"), ("noise", "gender")]
    assert "could-not-check, not a finding that no correlation is high" in _joined(messages)
    assert isinstance(result, list), "it must still BE a list for every consumer"


def test_a_matrix_with_no_pair_at_all_is_not_complete_coverage():
    """``compute_feature_correlations(frame, [])`` compared zero pairs."""
    frame = pd.DataFrame({"income": np.arange(50.0), "gender": ["m", "f"] * 25})

    vacuous, messages = _caught(lambda: compute_feature_correlations(frame, []))
    result, more = _caught(vacuous.get_high_correlations)

    assert list(result) == []
    assert result.complete is False
    assert "no pair to compare" in (result.not_compared_reason or "")
    assert "not a finding that no correlation is high" in _joined(more)


def test_a_negative_threshold_says_the_list_is_not_a_selection():
    """Every measured pair is above it, so the list is every pair the matrix has.
    The numbers are real, so this is a legibility disclosure and not a refusal."""
    result, messages = _caught(lambda: _matrix().get_high_correlations(threshold=-1.0))

    assert [row[0] for row in result] == ["height", "noise"]
    assert "is not a selection of high correlations" in _joined(messages)


def test_control_a_real_threshold_still_selects_the_measured_strong_pair():
    """Over-correction control, with the actual coefficient."""
    result, messages = _caught(lambda: _matrix().get_high_correlations(threshold=0.3))

    assert [(f, a, round(float(v), 2)) for f, a, v in result] == [("height", "gender", 0.85)]
    assert result.complete is True
    assert result.not_compared_reason is None
    assert result.pairs_not_measured == []
    assert messages == []


# ===========================================================================
# 8. SyntheticResampler.get_resampled_data: synthetic rows for a non-group
# ===========================================================================


def _resampled(frame: pd.DataFrame, y, attrs=("gender",)):
    resampler = SyntheticResampler(protected_attributes=list(attrs))
    resampler.fit(frame, y)
    out, messages = _caught(lambda: resampler.get_resampled_data(frame, y))
    return resampler, out, messages


def test_no_synthetic_row_is_invented_for_rows_with_no_recorded_group():
    """``_group_key_values`` stringifies, so the absence of a record became the
    group key 'None' and SMOTE interpolated into it.

    Before: 130 rows in, 180 rows out,
    ``value_counts(dropna=False)`` 'm 60, f 60, None 60', so 40 SYNTHETIC rows
    had been interpolated into a non-group and one third of the returned
    training set belonged to it, with ``fit_result.warnings == []`` and no
    Python warning. The sibling ``CounterfactualAugmenter`` refuses the same
    rows and says so.
    """
    rng = np.random.default_rng(9)
    n = 130
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 60 + ["f"] * 50 + [None] * 20,
            "x": rng.normal(0, 1, n),
            "z": rng.normal(3, 1, n),
        }
    )
    y = np.array(([1] * 30 + [0] * 30) + ([1] * 25 + [0] * 25) + ([1] * 10 + [0] * 10))

    resampler, (out_x, out_y), messages = _resampled(frame, y)

    # The 20 unrecorded rows come back exactly as they went in: not dropped
    # (that would destroy real rows) and not multiplied.
    assert int(out_x["gender"].isna().sum()) == 20
    assert len(out_x) == 140
    assert len(out_y) == 140
    counts = out_x["gender"].value_counts().to_dict()
    assert counts == {"m": 60, "f": 60}, "the RECORDED groups are still balanced"
    joined = _joined(resampler.fit_result.warnings)
    assert "20 of 130 row(s) have no recorded value" in joined
    assert "NO synthetic row was created for them" in joined
    assert "have no recorded value" in _joined(messages)


def test_an_entirely_unrecorded_protected_column_says_so():
    """Before: the frame came back unchanged with 'every one of the 2 (group,
    label) cell(s) already holds 30 row(s), so there was nothing to add', a
    group-balance statement about a frame in which no group was recorded."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"gender": [None] * 60, "x": rng.normal(0, 1, 60)})
    y = np.array([1] * 30 + [0] * 30)

    resampler, (out_x, _), _ = _resampled(frame, y)

    assert len(out_x) == 60
    joined = _joined(resampler.fit_result.warnings)
    assert "none of the 60 row(s) has a recorded value" in joined
    assert "returned UNCHANGED" in joined
    assert "already holds" not in joined


def test_control_a_real_resample_still_balances_and_stays_silent():
    """Over-correction control, with the actual group counts."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"gender": ["a"] * 30 + ["b"] * 6, "x": rng.normal(0, 1, 36), "z": rng.normal(3, 1, 36)}
    )
    y = np.array([1] * 15 + [0] * 15 + [1] * 3 + [0] * 3)

    resampler, (out_x, out_y), messages = _resampled(frame, y)

    assert len(out_x) == 60
    assert out_x["gender"].value_counts().to_dict() == {"a": 30, "b": 30}
    assert resampler.fit_result.warnings == []
    assert messages == []


# ===========================================================================
# 9. SyntheticResampler.fit: NOT A MEASUREMENT (grade category, no code change)
# ===========================================================================


def test_synthetic_resampler_fit_records_configuration_and_measures_nothing():
    """``fit`` reaches no verdict, so it has no measurement to substitute a clean
    value for. Its grade was corrected from PROVEN to NOT A MEASUREMENT, and no
    code changed: every capability refusal in this class is raised at the point
    of use in ``get_resampled_data``.

    Pinned so the category stays true: if a fairness quantity is ever estimated
    here, this test fails and the row needs grading again.
    """
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"gender": ["a"] * 20 + ["b"] * 20, "x": rng.normal(0, 1, 40)})
    resampler = SyntheticResampler(protected_attributes=["gender"])
    resampler.fit(frame, np.array([1, 0] * 20))

    assert resampler.is_fitted is True
    assert set(resampler.fit_result.fit_metrics) == {"method", "strategy", "k_neighbors"}
    assert resampler.fit_result.fit_metrics["method"] == "smote"
    assert resampler.fit_result.n_samples == 40
