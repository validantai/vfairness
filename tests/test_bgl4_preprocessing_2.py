"""BGL4 audit of batch A-preprocessing-2: the overturns, now CLOSED.

Every test in this file used to PASS, and that was the finding: each one pinned
the CURRENT output of a unit graded PROVEN, on an input class the named pin never
tried, and each docstring said what the honest answer would have been.

INVERTED 2026-09-27 (BGL5). All twelve overturned rows were repaired, so every
assertion that named a fabricated value has been turned round to the corrected
behaviour. The docstrings, the fixtures and the subject of each test are
unchanged, and each one still records what the output WAS, because that sentence
is the only reason a reader can tell this input class matters. Where a test's
NAME asserted the defect as a fact, the name states the corrected behaviour and
the docstring carries the former name, so the audit report's cross-references
still resolve by grep.

The repair ships its own pins in ``tests/test_bgl5_preprocessing_2.py``, with the
over-correction controls. This file is the historical record and the second,
independent guard.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vfairness.preprocessing.bias_detection.detector import AUDIT_MODULES, BiasAuditReport
from vfairness.preprocessing.bias_detection.statistical import (
    detect_specification_bias,
    detect_temporal_drift,
)
from vfairness.preprocessing.feature_engineering.correlation import (
    FeatureCorrelationMatrix,
    analyze_intersectional_correlations,
    compute_pearson_correlation_matrix,
    identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.data_balancing import SyntheticResampler


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


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


# ===========================================================================
# 1. execution_coverage / to_svg / empty_is_not_a_measurement
#    ONE root cause: "complete" is returned when the assessment half of the
#    record is UNRECORDED, and the canvas grants a measured all clear on it.
# ===========================================================================


def test_execution_coverage_withholds_complete_when_nothing_says_anything_was_assessed():
    """The honest answer is a could-not-check.

    ``execution_coverage``'s own docstring defines ``"complete"`` as "every module
    in AUDIT_MODULES ran AND at least one protected attribute was actually
    assessed". Here the second half is not recorded at all, ``unrecorded`` is
    already a member of the vocabulary, and the word returned is the one single
    word every reader in this library treats as a licence.

    INVERTED (was ``test_execution_coverage_says_complete_when_nothing_says_
    anything_was_assessed``). Before: 'unrecorded' beside 'complete',
    ``empty_is_not_a_measurement()`` None and ``get_critical_count()`` 0 with no
    warning.
    """
    report = _report(modules_run=list(AUDIT_MODULES))

    assert report.assessment_coverage() == "unrecorded"
    assert report.execution_coverage() == "unrecorded"
    assert report.empty_is_not_a_measurement() is not None
    count, messages = _caught(report.get_critical_count)
    assert count is None
    assert "no count of critical issues is reported" in " ".join(messages)


def test_the_bgl_s2_defect_is_closed_in_the_observation_count_fallback():
    """Two rows no longer buy a clean bill of health on the legacy branch.

    ``assessment_coverage``'s own docstring describes the BGL-S2 defect as
    "attribute_observations said {'gender': 2}, ``if n`` was true, and the word
    licensed four empty finding lists and a 0.0 risk score as a clean bill of
    health". The repair moved the primary test to ``attribute_assessed`` and left
    ``bool(n)`` in the fallback, which is the branch that exists for reports
    predating that record. ``MIN_ASSESSABLE_GROUP_SIZE`` is defined in this very
    module for this decision and is not consulted here.

    The named pin exercises the fallback at n=40 and n=0 only, and it reaches n=2
    solely in the company of an explicit verdict, which wins before the count is
    ever read.

    INVERTED (was ``test_the_bgl_s2_defect_survives_verbatim_in_the_observation_
    count_fallback``). Before: assessment_coverage() 'complete',
    execution_coverage() 'complete', empty_is_not_a_measurement() None,
    unassessable_attributes() [] and an SVG carrying 'MINIMAL' with no
    'NOT ASSESSABLE'. The fallback now reads the count against
    MIN_ASSESSABLE_GROUP_SIZE, which is what that constant was defined for.
    """
    report = _report(
        dataset_info={"n_rows": 2},
        modules_run=list(AUDIT_MODULES),
        attribute_observations={"gender": 2},
    )

    assert report.attribute_assessed is None
    assert report.assessment_coverage() == "none"
    assert report.execution_coverage() == "ran_but_assessed_nothing"
    assert "gender" in report.empty_is_not_a_measurement()
    assert report.unassessable_attributes() == ["gender"]
    svg, _ = _caught(report.to_svg)
    assert "NOT ASSESSABLE" in svg and "MINIMAL" not in svg


def test_the_canvas_no_longer_bands_minimal_without_a_could_not_check_anywhere():
    """The R-11 canvas defect, reached through the unrecorded record.

    ``tests/test_bias_audit_svg_coverage.py`` pins this canvas against reports
    produced by ``full_audit``, which always records both halves. A hand built or
    older report records only ``modules_run``, and the export then certifies a
    dataset nobody established anything about.

    INVERTED (was ``test_the_canvas_bands_that_report_minimal_with_no_could_not_
    check_anywhere``). Before: 'MINIMAL', four '>0.00<' tiles, no 'NOT
    ASSESSABLE' and no 'COULD NOT CHECK' token anywhere on the canvas.
    """
    svg, _ = _caught(_report(modules_run=list(AUDIT_MODULES)).to_svg)

    assert "MINIMAL" not in svg
    assert "NOT ASSESSABLE" in svg
    assert "COULD NOT CHECK" in svg.upper()
    # The residual this test recorded is CLOSED, 2026-09-29, by BGL6 F06 in
    # rendering/adapters.py (the file batch A-preprocessing-2 did not own, which
    # is why it was handed back as needs_another_batch). The tiles no longer key
    # on `assessed_nothing`; they key on a NEW `assessment_recorded`, which is
    # False for "unrecorded" as well, and is asked of the report's own
    # `assessment_coverage()` so a partial run with no assessment record is
    # withheld too. Re-measured 2026-09-29: this canvas carries ZERO '>0.00<'
    # tiles, which is what the batch's own 2-row pin always asserted.
    #
    # The assertion is INVERTED rather than deleted: it is the same question, and
    # 4 was the measured before-state.
    assert svg.count(">0.00<") == 0, (
        f"{svg.count('>0.00<')} measured-looking module tiles on a canvas whose "
        f"headline reads NOT ASSESSABLE"
    )


# ===========================================================================
# 2. detect_temporal_drift: "pass" over a column holding no observation
# ===========================================================================


def _dated(values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "gender": values,
        }
    )


def test_an_all_null_protected_column_is_not_assessed_rather_than_graded_stable():
    """``fillna("missing")`` invents one synthetic category, so PSI is 0.0.

    The three routes to "nothing was assessed" that the pin covers are an absent
    column, an empty attribute list and an identifier grade column. A column that
    is PRESENT and holds no observation is the fourth, and it is the one R-11 was
    written about.

    INVERTED (was ``test_an_all_null_protected_column_is_graded_stable_rather_
    than_not_assessed``). Before: available True, severity 'pass',
    compositionPSI 0.0, plain 'Across application_date, the gender mix is stable
    (PSI 0.00)' and no warning.
    """
    result, messages = _caught(lambda: detect_temporal_drift(_dated([None] * 100), ["gender"]))

    assert result["available"] is False
    assert "severity" not in result
    assert result["drift"] == []
    assert result["attributesWithNoObservation"] == ["gender"]
    assert "nothing was compared, so nothing is clean" in result["reason"]
    assert "no protected attribute could be assessed" in " ".join(messages)


def test_an_empty_frame_is_not_graded_stable_either():
    """Zero rows compared, and the summary was a clean bill of health.

    INVERTED (was ``test_an_empty_frame_is_graded_stable_as_well``). Before:
    available True, severity 'pass', summary 'No material temporal drift in the
    assessed attributes over application_date' and no warning.
    """
    empty = pd.DataFrame(
        {"application_date": pd.to_datetime([]), "gender": pd.Series([], dtype=object)}
    )
    result, messages = _caught(lambda: detect_temporal_drift(empty, ["gender"]))

    assert result["available"] is False
    assert "summary" not in result
    assert "no protected attribute could be assessed" in result["reason"]
    assert messages != []


# ===========================================================================
# 3. detect_specification_bias: CRITICAL findings nobody measured
# ===========================================================================


def test_an_empty_frame_no_longer_produces_a_critical_circular_evaluation_finding():
    """Two empty Series compare equal, so the screen reported the worst finding
    it has over zero rows. The honest answer is that the check could not run.

    INVERTED (was ``test_an_empty_frame_produces_a_critical_circular_evaluation_
    finding``). Before: available True, severity 'critical', findings[0]['kind']
    'circular_evaluation' and checksRun ['proxy_target', 'circular_evaluation'].
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

    assert result["available"] is True
    assert result["severity"] == "pass"
    assert result["findings"] == []
    assert "circular_evaluation" not in result["checksRun"]
    assert "circular_evaluation: only 0 of 0 row(s)" in " ".join(result["checksNotRun"])
    assert "could not run" in " ".join(messages)


def test_an_outcome_and_prediction_that_are_both_entirely_null_are_not_called_equal():
    """``astype(str)`` maps every missing value to the string 'nan', so two
    columns nobody recorded a value in were reported as the same column.

    INVERTED (was ``test_an_outcome_and_prediction_that_are_both_entirely_null_
    are_called_equal``). Before: severity 'critical' with a circular_evaluation
    finding over 50 rows in which nothing had been recorded.
    """
    frame = pd.DataFrame({"approved": [np.nan] * 50, "pred": [np.nan] * 50, "x": np.arange(50.0)})
    result, _ = _caught(
        lambda: detect_specification_bias(frame, outcome="approved", prediction="pred")
    )

    assert result["severity"] == "pass"
    assert result["findings"] == []
    assert "circular_evaluation: only 0 of 50 row(s)" in " ".join(result["checksNotRun"])


def test_two_rows_no_longer_make_every_numeric_feature_a_critical_leak():
    """Any two points are perfectly collinear, so |r| is 1.0 by construction.

    There was no minimum n on this Pearson scan, so the screen reported that it
    scanned two columns and found the answer leaking in both of them.

    INVERTED (was ``test_two_rows_make_every_numeric_feature_a_critical_leak``).
    Before: severity 'critical', findings ['target_leakage', 'target_leakage'],
    '|r|=1.00' in the first plain sentence, and checksRun carrying
    'target_leakage (2 feature column(s) scanned)'.
    """
    frame = pd.DataFrame({"approved": [1, 0], "x": [5.0, 9.0], "z": [2.0, 3.0]})
    result, _ = _caught(lambda: detect_specification_bias(frame, outcome="approved"))

    assert result["severity"] == "pass"
    assert [f["kind"] for f in result["findings"]] == []
    assert not any("scanned" in c for c in result["checksRun"])
    not_run = " ".join(result["checksNotRun"])
    assert "x (n=2)" in not_run and "z (n=2)" in not_run


# ===========================================================================
# 4. compute_pearson_correlation_matrix: a p-value of 0.0 on a diagonal cell
#    whose coefficient the same call refused to compute
# ===========================================================================


def test_the_diagonal_p_value_is_withheld_for_a_column_with_no_variation():
    """The code comment justified the 0.0 with "it is 1.0 by construction whenever
    the column has any observation at all". A constant column has observations and
    no self correlation: the coefficient matrix said NaN and the p-value matrix
    said 0.0, the strongest assertion of an association the test can make.

    INVERTED (was ``test_the_diagonal_p_value_is_zero_for_a_column_with_no_
    variation``). Before: pvals.loc['a','a'] == 0.0 beside a NaN coefficient, so
    ``(pvals < 0.05).loc['a','a']`` was True and the docstring's own
    ``corr[pvals < 0.05]`` selected it.
    """
    frame = pd.DataFrame({"a": [5.0] * 40, "b": list(np.arange(40.0))})
    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(frame, return_pvalues=True)
    )

    assert np.isnan(corr.loc["a", "a"])
    assert np.isnan(pvals.loc["a", "a"])
    assert not bool((pvals < 0.05).loc["a", "a"])
    # The docstring promises "a warning names every pair it applies to", and the
    # only notice used to be scipy's ConstantInputWarning, which names neither.
    assert "a ~ b (n=40)" in " ".join(messages)


def test_the_two_matrices_agree_for_a_column_below_min_periods():
    """Three non-null rows against the default ``min_periods=10``.

    INVERTED (was ``test_the_two_matrices_disagree_for_a_column_below_min_
    periods``). Before: a NaN coefficient beside a p-value of 0.0.
    """
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0] + [np.nan] * 20, "b": list(np.arange(23.0))})
    (corr, pvals), _ = _caught(
        lambda: compute_pearson_correlation_matrix(frame, return_pvalues=True)
    )

    assert np.isnan(corr.loc["a", "a"])
    assert np.isnan(pvals.loc["a", "a"])


# ===========================================================================
# 5. analyze_intersectional_correlations
# ===========================================================================


def test_coverage_no_longer_reads_complete_when_a_feature_was_never_measured():
    """The per GROUP disclosure was complete and the per FEATURE one did not exist.

    ``x`` is measured in group 'm' and unmeasurable in group 'f', so it is dropped
    from ``between_group_differences``.

    INVERTED (was ``test_coverage_reads_complete_while_a_feature_was_never_
    measured_in_one_group``). Before: coverage 'complete', groups_not_assessed
    [], no warning, and 'x' absent from between_group_differences with nothing
    saying so.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 40 + ["f"] * 40,
            "x": list(rng.normal(50, 5, 40)) + [np.nan] * 40,
            "y": list(rng.normal(10, 1, 80)),
        }
    )
    result, messages = _caught(lambda: analyze_intersectional_correlations(frame, ["gender"]))

    assert result["coverage"] == "partial"
    assert result["groups_not_assessed"] == []
    assert "x" in result["within_group_correlations"]["m"]
    assert "x" not in result["within_group_correlations"]["f"]
    assert "x" not in result["between_group_differences"]
    assert [(e["feature"], e["group"]) for e in result["features_not_assessed"]] == [("x", "f")]
    assert [e["feature"] for e in result["features_not_compared"]] == ["x"]
    assert "NOT measured inside an examined group" in " ".join(messages)


def test_rows_with_no_recorded_group_no_longer_become_a_group():
    """The absence of a record was stringified into a group key and then measured
    against a real group. ``_fit_propensity_model`` in this package refuses the
    same input by name; this function reported a between group difference from it.

    INVERTED (was ``test_rows_with_no_recorded_group_become_a_group_and_drive_
    the_difference``). Before: intersectional_groups ['m', 'None'], coverage
    'complete', warnings [], and
    between_group_differences['x']['max_difference'] 31.43 measured between a
    real group and the absence of a record.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "gender": ["m"] * 40 + [None] * 40,
            "x": list(rng.normal(50, 5, 40)) + list(rng.normal(80, 5, 40)),
        }
    )
    result, messages = _caught(lambda: analyze_intersectional_correlations(frame, ["gender"]))

    assert "None" not in result["intersectional_groups"]
    assert result["intersectional_groups"] == ["m"]
    assert result["coverage"] != "complete"
    assert result["rows_without_a_recorded_group"] == 40
    assert "x" not in result["between_group_differences"]
    assert "They are NOT a group" in " ".join(messages)


# ===========================================================================
# 6. identify_proxy_variables: a requested feature that is not a column
# ===========================================================================


def test_a_requested_feature_that_is_not_a_column_now_leaves_a_trace():
    """``ProxyScreenResult.complete`` is documented as "True only when every
    requested pair was actually screened". Zero pairs were screened here.

    INVERTED (was ``test_a_requested_feature_that_is_not_a_column_leaves_no_
    trace``). Before: 0 results, complete True, screens_not_run [] and no
    warning, on a frame where feature_columns=None finds 'height' CRITICAL.
    """
    rng = np.random.default_rng(5)
    n = 200
    g = rng.choice(["m", "f"], n)
    frame = pd.DataFrame(
        {"gender": g, "height": np.where(g == "m", rng.normal(180, 5, n), rng.normal(165, 5, n))}
    )
    result, messages = _caught(
        lambda: identify_proxy_variables(frame, ["gender"], feature_columns=["not_a_column"])
    )

    assert list(result) == []
    assert result.complete is False
    assert [e["feature"] for e in result.screens_not_run] == ["not_a_column"]
    assert "were NOT SCREENED" in " ".join(messages)


# ===========================================================================
# 7. get_high_correlations: a threshold that cannot fire
# ===========================================================================


def test_a_nan_threshold_no_longer_drops_a_measured_strong_pair_in_silence():
    """The sibling unit graded in the same wave (``GroupManager`` with
    ``min_group_size <= 0``) discloses a threshold that disables its check. This
    one reported an empty list and ``complete is True``.

    INVERTED (was ``test_a_nan_threshold_drops_a_measured_strong_pair_in_
    silence``). Before: [] with complete True, pairs_not_measured [] and no
    warning, over a matrix holding a MEASURED 0.85.
    """
    corr = pd.DataFrame({"gender": [0.85, 0.02]}, index=["height", "noise"])
    pvals = pd.DataFrame({"gender": [0.001, 0.9]}, index=["height", "noise"])
    matrix = FeatureCorrelationMatrix(
        correlations=corr,
        pvalues=pvals,
        feature_names=["height", "noise"],
        protected_attributes=["gender"],
        method="mixed",
    )

    measured, _ = _caught(lambda: matrix.get_high_correlations(threshold=0.3))
    assert [row[0] for row in measured] == ["height"]

    vacuous, messages = _caught(lambda: matrix.get_high_correlations(threshold=float("nan")))
    assert list(vacuous) == []
    assert vacuous.complete is False
    assert sorted(vacuous.pairs_not_measured) == [("height", "gender"), ("noise", "gender")]
    assert "is not a finite number" in (vacuous.not_compared_reason or "")
    assert "could-not-check" in " ".join(messages)


# ===========================================================================
# 8. SyntheticResampler.get_resampled_data: synthetic rows for a non group
# ===========================================================================


def test_no_synthetic_row_is_invented_for_rows_with_no_recorded_group():
    """20 rows whose protected value was never recorded were oversampled to 60.

    ``_group_key_values`` stringifies, so the absence of a record became the
    group key 'None' and SMOTE interpolated 40 new rows into it. The sibling
    ``CounterfactualAugmenter`` refuses the same rows and says so; this class had
    an EMPTY disclosure channel for it.

    INVERTED (was ``test_synthetic_rows_are_invented_for_rows_with_no_recorded_
    group``). Before: 130 rows in, 180 rows out,
    ``out_x['gender'].isna().sum()`` 60, ``fit_result.warnings == []`` and no
    Python warning.
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

    resampler = SyntheticResampler(protected_attributes=["gender"])
    resampler.fit(frame, y)
    (out_x, _), messages = _caught(lambda: resampler.get_resampled_data(frame, y))

    assert len(frame) == 130
    assert len(out_x) == 140
    assert int(out_x["gender"].isna().sum()) == 20, "returned as they came in"
    assert out_x["gender"].value_counts().to_dict() == {"m": 60, "f": 60}
    assert resampler.fit_result is not None
    assert "NO synthetic row was created for them" in " ".join(resampler.fit_result.warnings)
    assert "have no recorded value" in " ".join(messages)
