"""BGL7: the four preprocessing fixes that did NOT survive re-attack (2026-09-29).

Four BGL5 fixes were graded PROVEN, audited independently, and OVERTURNED to
DEFECT OPEN because the same defect was still live one input step away from the
shape the fix pinned. Every defect here is the same class: a NEUTRAL value
(0.0, 1.0, True, [], a PSI, a row count) substituted for one that could not be
measured, and then published as a measurement.

  * ``detect_temporal_drift``: a time column with NO VARIATION was split by file
    order and graded CRITICAL, and unreadable timestamps were IMPUTED into the
    cohorts.
  * ``SyntheticResampler.get_resampled_data``: the fix excluded rows with no
    recorded GROUP; the other half of the (group, label) cell key, an absent
    LABEL, was still synthesised into.
  * ``compute_pearson_correlation_matrix``: an inf made the two matrices
    DISAGREE (corr 1.0 beside a nan p-value) while the warning said both were
    nan, and ``min_periods <= 2`` disabled the floor silently.
  * ``identify_proxy_variables``: a screen over ZERO pairs returned
    ``complete=True`` with no disclosure.

Every test here is a PIN on the corrected behaviour. Each docstring opens with
the BEFORE state as MEASURED on the unfixed code, and each fix carries an
over-correction control that asserts the healthy case's REAL numbers, because a
refusal that fires on everything passes every refusal pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift
from vfairness.preprocessing.feature_engineering.analyzer import FeatureEngineeringAnalyzer
from vfairness.preprocessing.feature_engineering.correlation import (
    compute_pearson_correlation_matrix,
    identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.data_balancing import SMOTEResampler


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


def _joined(messages) -> str:
    return " || ".join(messages)


# ===========================================================================
# 1. detect_temporal_drift: the TIME AXIS was never checked
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "frame"),
    [
        (
            "constant date string",
            lambda: pd.DataFrame(
                {
                    "app_date": ["2024-01-01"] * 60,
                    "gender": ["m"] * 21 + ["f"] * 9 + ["m"] * 9 + ["f"] * 21,
                }
            ),
        ),
        (
            "constant numeric time",
            lambda: pd.DataFrame(
                {
                    "app_time": [7] * 60,
                    "gender": ["m"] * 21 + ["f"] * 9 + ["m"] * 9 + ["f"] * 21,
                }
            ),
        ),
    ],
)
def test_a_time_column_with_no_variation_is_a_static_snapshot_not_drift(label, frame):
    """A single-valued time column has no earliest and no latest period.

    BEFORE, measured 2026-09-29 on 60 rows all carrying ``app_date``
    '2024-01-01' with gender 21 'm' / 9 'f' / 9 'm' / 21 'f'::

        {'available': True, 'severity': 'critical',
         'summary': 'Temporal drift risk: the gender population changes over
                     app_date (PSI 0.46); a model trained on the early period
                     may degrade or bias against the later mix.',
         'drift': [{'attribute': 'gender', 'compositionPSI': 0.4621,
                    'severity': 'critical',
                    'plain': 'Across app_date, the gender mix shifted materially
                              (PSI 0.46) between the earliest and latest
                              applicants.'}]}

    and ``warnings == []``. There are no earliest and latest applicants:
    ``np.argsort`` on a constant array falls back to file order, so the two
    cohorts were rows 1-30 and 31-60 of the input and the CRITICAL finding was a
    fact about the row order of a file. The constant NUMERIC column gave the
    identical PSI 0.46 'critical'. This function's own ``available=False`` branch
    already exists to refuse a static snapshot, and a single-valued time column
    IS one.
    """
    result, messages = _caught(lambda: detect_temporal_drift(frame(), ["gender"]))

    assert result["available"] is False
    assert result["drift"] == []
    assert "severity" not in result, "an unavailable screen must not publish a verdict"
    assert "summary" not in result
    assert "no earliest and no latest period" in result["reason"]
    assert "nothing was compared over time, so nothing is clean" in result["reason"]
    assert "no earliest and no latest period" in _joined(messages)


def test_an_unreadable_timestamp_is_excluded_and_counted_not_imputed_into_a_cohort():
    """A row with no readable timestamp belongs to no period.

    BEFORE, measured 2026-09-29 on 31 readable dates (all gender 'm') beside 29
    rows whose ``app_date`` is the string 'not a date' (all gender 'f'): the 0.5
    parseability floor passes at 0.517, then ``order.fillna(order.median())``
    imputed all 29 unreadable timestamps into the middle of the ordering, and a
    COMPLETE m -> f flip was published as::

        {'available': True, 'severity': 'pass',
         'drift': [{'attribute': 'gender', 'compositionPSI': 0.0045,
                    'severity': 'pass',
                    'plain': 'Across app_date, the gender mix is stable
                              (PSI 0.00).'}]}

    with ``warnings == []`` and no key anywhere counting the rows whose
    timestamp could not be read (the returned keys were attributesNotInFrame,
    attributesWithNoObservation, attributesWithTooFewObservations only).
    """
    frame = pd.DataFrame(
        {
            "app_date": [f"2024-01-{d:02d}" for d in range(1, 32)] + ["not a date"] * 29,
            "gender": ["m"] * 31 + ["f"] * 29,
        }
    )

    result, messages = _caught(lambda: detect_temporal_drift(frame, ["gender"]))

    # The 29 rows are OUT of the ordering, counted where a reader looks, and
    # named in a warning. What is left is a measurement over the 31 rows whose
    # time could be read, which is what the verdict now covers.
    assert result["nRowsWithUnreadableTime"] == 29
    assert "29 of 60 row(s) hold no readable value in 'app_date'" in _joined(messages)
    assert "they belong to no period" in _joined(messages)
    assert "could-not-check, not evidence" in _joined(messages)


def test_the_compared_cohorts_hold_only_rows_whose_time_could_be_read():
    """The other half of the same defect: WHERE the unreadable rows went.

    The count and the warning above can be published while the rows are still
    imputed into the cohorts, so this pins the cohort membership by the number it
    changes. 40 rows carry a readable date and a genuine 80/20 -> 20/80 gender
    drift; 21 more carry 'not a date' and the gender value 'x', which exists
    nowhere else.

    BEFORE (``order.fillna(order.median())``, re-measured 2026-09-29 by restoring
    that one line): the 21 'x' rows sort to the median instant, so 11 land in the
    earliest cohort and 10 in the latest, 'x' becomes a third compared category
    and the published PSI is 1.0932. AFTER: 1.6636, which is exactly what the 40
    readable rows give when they are passed on their own, so the compared
    cohorts are those rows and no others.
    """
    dates = [str(d.date()) for d in pd.date_range("2024-01-01", periods=40, freq="D")]
    gender = ["m"] * 16 + ["f"] * 4 + ["m"] * 4 + ["f"] * 16
    frame = pd.DataFrame({"app_date": dates + ["not a date"] * 21, "gender": gender + ["x"] * 21})

    result, _ = _caught(lambda: detect_temporal_drift(frame, ["gender"]))
    readable_only, _ = _caught(
        lambda: detect_temporal_drift(
            pd.DataFrame({"app_date": dates, "gender": gender}), ["gender"]
        )
    )

    assert result["nRowsWithUnreadableTime"] == 21
    assert result["drift"][0]["compositionPSI"] == pytest.approx(1.6636, abs=5e-4)
    assert result["drift"][0]["compositionPSI"] == readable_only["drift"][0]["compositionPSI"]
    assert result["drift"][0]["compositionPSI"] != pytest.approx(1.0932, abs=5e-4)


def test_a_fully_readable_time_column_reports_zero_unreadable_rows():
    """The three-state disclosure must be able to say "none", not only "some"."""
    frame = pd.DataFrame(
        {
            "app_date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "gender": ["m", "f"] * 50,
        }
    )

    result, messages = _caught(lambda: detect_temporal_drift(frame, ["gender"]))

    assert result["available"] is True
    assert result["nRowsWithUnreadableTime"] == 0
    assert messages == []


def test_control_a_real_temporal_drift_is_still_measured_at_its_real_psi():
    """Over-correction control, with the actual PSI on both sides.

    A refusal that fires on every time column would pass both pins above.
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    drifting, messages = _caught(
        lambda: detect_temporal_drift(
            pd.DataFrame(
                {
                    "app_date": dates,
                    "gender": ["m"] * 40 + ["f"] * 10 + ["m"] * 10 + ["f"] * 40,
                }
            ),
            ["gender"],
        )
    )

    assert drifting["available"] is True
    assert drifting["severity"] == "critical"
    assert drifting["drift"][0]["compositionPSI"] == pytest.approx(1.6636, abs=5e-4)
    assert "Temporal drift risk" in drifting["summary"]
    assert messages == []

    stable, stable_messages = _caught(
        lambda: detect_temporal_drift(
            pd.DataFrame({"app_date": dates, "gender": ["m", "f"] * 50}), ["gender"]
        )
    )

    assert stable["available"] is True
    assert stable["severity"] == "pass"
    assert stable["drift"][0]["compositionPSI"] == 0.0
    assert stable_messages == []


def test_control_a_mostly_readable_time_column_is_still_measured():
    """Only 4 of 100 timestamps unreadable: the drift is still graded, at its
    real PSI, over the 96 rows that carry a time."""
    dates = [str(d.date()) for d in pd.date_range("2024-01-01", periods=100, freq="D")]
    gender = ["m"] * 40 + ["f"] * 10 + ["m"] * 10 + ["f"] * 40
    for i in (5, 20, 60, 90):
        dates[i] = "not a date"

    result, messages = _caught(
        lambda: detect_temporal_drift(
            pd.DataFrame({"app_date": dates, "gender": gender}), ["gender"]
        )
    )

    assert result["available"] is True
    assert result["severity"] == "critical"
    assert result["drift"][0]["compositionPSI"] > 1.0
    assert result["nRowsWithUnreadableTime"] == 4
    assert "4 of 100 row(s) hold no readable value" in _joined(messages)


# ===========================================================================
# 2. SyntheticResampler.get_resampled_data: the OTHER half of the cell key
# ===========================================================================


def _resample_frame(gender):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=130), "f2": rng.normal(size=130), "gender": gender})


def _resampled(gender, y):
    df = _resample_frame(gender)
    sampler = SMOTEResampler(protected_attributes=["gender"])
    (out_df, out_y), messages = _caught(
        lambda: (sampler.fit(df, y), sampler.get_resampled_data(df, y))[1]
    )
    return sampler, out_df, out_y, messages


_UNLABELLED = [
    ("None", np.array(([0, 1] * 55) + [None] * 20, dtype=object)),
    ("pd.NA", np.array(([0, 1] * 55) + [pd.NA] * 20, dtype=object)),
    ("np.nan", np.concatenate([np.array([0.0, 1.0] * 55), np.full(20, np.nan)])),
]


@pytest.mark.parametrize(("label", "y"), _UNLABELLED, ids=[k for k, _ in _UNLABELLED])
def test_no_synthetic_row_is_invented_for_a_row_whose_label_was_never_recorded(label, y):
    """An absent LABEL is not a label value, the half of the cell key nobody guarded.

    BEFORE, measured 2026-09-29 on 130 rows (gender 60 'm' / 70 'f', fully
    recorded) with ``y = np.array(([0, 1] * 55) + [None] * 20, dtype=object)``:
    130 rows in, **150 rows OUT**, output label counts ``{0: 60, 1: 60, None:
    30}``, so TEN synthetic rows were interpolated into a cell whose label was
    never recorded. Python warnings: ``[]``. The only trace treated the absence
    as a legitimate label value: "not group-balanced: 1 (group, label) cell(s)
    have no examples to resample from, so they stay empty and the group totals
    remain uneven: (m, label=None). Every cell that does exist was balanced to 30
    rows." ``pd.NA`` behaved identically (150 out, 30 ``<NA>``). With float
    ``np.nan`` each NaN is its own cell, so the same 130 rows became **720**, of
    which **600** carried no label, disclosed only as "20 (group, label) cell(s)
    have no examples" plus a duplication warning about 580 rows, neither of which
    says the label was never recorded.

    AFTER: 140 rows out, exactly the 20 unlabelled rows unchanged beside the
    recorded cells balanced to 30, and the refusal is on both channels.
    """
    sampler, out_df, out_y, messages = _resampled(["m"] * 60 + ["f"] * 70, y)

    assert len(out_df) == 140, "no row may be invented for an unrecorded label"
    assert int(pd.isna(pd.Series(out_y)).sum()) == 20, (
        "the 20 unlabelled rows come back exactly as they came in, and no more"
    )
    assert sampler.fit_result.fit_metrics["n_rows_without_a_recorded_label"] == 20
    assert "20 of 130 row(s) have no recorded label" in _joined(messages)
    assert "An absent label is not a label VALUE" in _joined(messages)
    assert "NO synthetic row was created for them" in _joined(messages)
    assert "20 of 130 row(s) have no recorded label" in _joined(sampler.fit_result.warnings)
    # And the sentence that called the absence a balanced cell is gone.
    assert "label=None" not in _joined(sampler.fit_result.warnings)
    assert "label=nan" not in _joined(sampler.fit_result.warnings)


def test_an_entirely_unlabelled_column_says_so_instead_of_naming_the_group():
    """The group-axis sentence would be wrong one axis across.

    BEFORE: an all-None ``y`` reached the (group, label) machinery and every NaN
    became its own cell. AFTER: the frame is returned UNCHANGED and the cause
    names the LABEL, not ``['gender']``, which is fully recorded here.
    """
    sampler, out_df, out_y, messages = _resampled(
        ["m"] * 60 + ["f"] * 70, np.array([None] * 130, dtype=object)
    )

    assert len(out_df) == 130
    assert "none of the 130 row(s) has a recorded label" in _joined(sampler.fit_result.warnings)
    assert "no recorded value for ['gender']" not in _joined(sampler.fit_result.warnings)
    assert "130 of 130 row(s) have no recorded label" in _joined(messages)


def test_control_the_group_axis_refusal_is_unchanged():
    """The BGL5 fix this one sits beside must still behave exactly as recorded:
    130 rows with 20 unrecorded genders become 140, not 180."""
    sampler, out_df, out_y, messages = _resampled(
        ["m"] * 60 + ["f"] * 50 + [None] * 20, np.array([0, 1] * 65)
    )

    assert len(out_df) == 140
    assert out_df["gender"].value_counts().to_dict() == {"m": 60, "f": 60}
    assert int(out_df["gender"].isna().sum()) == 20
    assert sampler.fit_result.fit_metrics["n_rows_without_a_recorded_group"] == 20
    assert "20 of 130 row(s) have no recorded value for ['gender']" in _joined(messages)
    assert "no recorded label" not in _joined(messages)


def test_control_a_healthy_resample_still_balances_and_stays_silent():
    """Over-correction control with the REAL row count.

    A refusal that fires on every row would pass every pin above: 30 rows of 'a'
    and 6 of 'b' over both labels must still come out at 60 rows, {'a': 30,
    'b': 30}, with no warning on either channel.
    """
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"f1": rng.normal(size=36), "f2": rng.normal(size=36), "gender": ["a"] * 30 + ["b"] * 6}
    )
    y = np.array([0, 1] * 18)
    sampler = SMOTEResampler(protected_attributes=["gender"])
    (out_df, out_y), messages = _caught(
        lambda: (sampler.fit(df, y), sampler.get_resampled_data(df, y))[1]
    )

    assert len(out_df) == 60
    assert out_df["gender"].value_counts().to_dict() == {"a": 30, "b": 30}
    assert sampler.fit_result.warnings == []
    assert messages == []
    assert "n_rows_without_a_recorded_label" not in sampler.fit_result.fit_metrics
    assert "n_rows_without_a_recorded_group" not in sampler.fit_result.fit_metrics


# ===========================================================================
# 3. compute_pearson_correlation_matrix: two matrices, two definitions of
#    "overlapping row", and a floor the caller could switch off
# ===========================================================================

_INF = float("inf")


def test_an_inf_pair_no_longer_makes_the_two_matrices_disagree():
    """The two matrices counted different rows, so one had a number and one did not.

    BEFORE, measured 2026-09-29 on ``pd.DataFrame({'a': [inf] +
    list(range(39)), 'b': list(range(40))})`` with ``min_periods=10`` and
    ``return_pvalues=True``: ``corr.loc['a','b'] == 1.0`` beside
    ``pval.loc['a','b'] == nan``, so AGREE was False, and the warning emitted
    for that pair read "1 pair(s) have enough overlapping rows but no defined
    coefficient, because one side has no variation: a ~ b (n=40). Both matrices
    report nan there (could not check), NOT a coefficient of 0 or a p-value of 0
    or 1." The corr cell was not nan, neither side was constant, and n was 39 and
    not 40: three false statements about a cell that HAD been measured.

    The cause was two definitions of an overlapping row. pandas' own pearson
    masks with ``np.isfinite`` and had already dropped the inf; this function's
    loop used ``.notna()``, kept it, and handed it to ``scipy.stats.pearsonr``,
    which refuses it.
    """
    df = pd.DataFrame({"a": [_INF] + list(range(39)), "b": list(range(40))})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(df, min_periods=10, return_pvalues=True)
    )

    agree = bool(np.isnan(corr.loc["a", "b"])) == bool(np.isnan(pvals.loc["a", "b"]))
    assert agree, "a coefficient may not be published without the p-value beside it"
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert pvals.loc["a", "b"] == pytest.approx(0.0)
    # And the sentence that described it is true now: it names the inf, not a
    # constant column that does not exist.
    assert "are infinite: a (1 of 40 row(s))" in _joined(messages)
    assert "no variation" not in _joined(messages)


def test_the_inf_row_is_excluded_from_the_coefficient_itself():
    """Discriminating fixture: a and b whose finite-row correlation is NOT 1.0.

    The pin above cannot tell a measurement over the 39 finite rows from any
    other number that happens to be 1.0, so this one measures a pair whose real
    coefficient is 0.9105 and checks the published value against the same 39 rows
    passed on their own.
    """
    rng = np.random.default_rng(7)
    base = np.arange(39.0)
    noisy = 2 * base + rng.normal(0, 12, 39)
    df = pd.DataFrame({"a": np.concatenate([[_INF], base]), "b": np.concatenate([[0.0], noisy])})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(df, min_periods=10, return_pvalues=True)
    )
    finite_only = pd.DataFrame({"a": base, "b": noisy}).corr(min_periods=10)

    assert corr.loc["a", "b"] == pytest.approx(0.9105, abs=5e-4)
    assert corr.loc["a", "b"] == pytest.approx(float(finite_only.loc["a", "b"]))
    assert pvals.loc["a", "b"] < 0.05
    assert "1 value(s) in 1 column(s) are infinite" in _joined(messages)


@pytest.mark.parametrize("min_periods", [2, 1, 0])
def test_min_periods_below_three_does_not_disable_the_overlap_floor(min_periods):
    """|r| is +1 or -1 for ANY two points, so a 2-row overlap is arithmetic.

    BEFORE, measured 2026-09-29 on two columns sharing exactly 2 non-null rows
    (``a = [1.0, 2.0] + [nan]*38``, ``b = [3.0, 9.0] + [nan]*38``):
    ``min_periods=10`` gave corr nan / pval nan with two warnings (correct),
    while ``min_periods=2``, ``1`` and ``0`` each gave **corr 1.0 and pval 1.0
    with ZERO warnings**. Changing the data to ``[99.0, -7.0]`` moved the
    coefficient to -1.0, which is how you can tell it is the row count talking
    and not the data, and a p-value of 1.0 is the strongest statement the test
    can make that the data give no evidence. ``min_periods`` is caller
    controlled, and nothing disclosed a value that disables the check.
    """
    df = pd.DataFrame({"a": [1.0, 2.0] + [np.nan] * 38, "b": [3.0, 9.0] + [np.nan] * 38})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(df, min_periods=min_periods, return_pvalues=True)
    )

    assert np.isnan(corr.loc["a", "b"])
    assert np.isnan(pvals.loc["a", "b"])
    assert f"min_periods={min_periods} is below 3" in _joined(messages)
    assert "a ~ b (n=2)" in _joined(messages)
    assert "NOT a coefficient of 1 or a p-value of 1" in _joined(messages)


def test_the_floor_holds_on_the_default_return_pvalues_false_path_too():
    """There is no p-value beside the coefficient there to contradict it.

    BEFORE: ``compute_pearson_correlation_matrix(df, min_periods=2)`` returned
    1.0 for the same 2-row overlap, with no warning at all.
    """
    df = pd.DataFrame({"a": [1.0, 2.0] + [np.nan] * 38, "b": [3.0, 9.0] + [np.nan] * 38})

    corr, messages = _caught(lambda: compute_pearson_correlation_matrix(df, min_periods=2))

    assert np.isnan(corr.loc["a", "b"])
    assert "min_periods=2 is below 3" in _joined(messages)


def test_control_a_real_pearson_matrix_keeps_its_numbers():
    """Over-correction control with the REAL coefficient.

    A refusal that fires on every pair, or an inf mask that eats finite values,
    would pass every pin above. 100 rows of ``b = 2a + noise`` must still be
    measured, significant, and silent.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(size=100)
    df = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.05, 100)})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(df, return_pvalues=True)
    )

    assert corr.loc["a", "b"] > 0.99
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert pvals.loc["a", "a"] == 0.0
    assert pvals.loc["a", "b"] < 0.05
    assert bool((pvals < 0.05).loc["a", "b"]) is True
    assert messages == []


def test_control_min_periods_below_three_still_measures_a_real_overlap():
    """The floor is on the OVERLAP, not on min_periods: a pair with 50
    overlapping rows must still be measured at min_periods=2."""
    rng = np.random.default_rng(13)
    a = rng.normal(size=50)
    df = pd.DataFrame({"a": a, "b": 3 * a + rng.normal(0, 0.1, 50)})

    (corr, pvals), messages = _caught(
        lambda: compute_pearson_correlation_matrix(df, min_periods=2, return_pvalues=True)
    )

    assert corr.loc["a", "b"] > 0.99
    assert pvals.loc["a", "b"] < 0.05
    assert messages == []


# ===========================================================================
# 4. identify_proxy_variables: complete=True over ZERO pairs
# ===========================================================================


def _proxy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    gender = np.array(["m"] * 100 + ["f"] * 100)
    return pd.DataFrame(
        {
            "gender": gender,
            "height": np.where(gender == "m", 178, 165) + rng.normal(0, 6, 200),
            "noise": rng.normal(size=200),
        }
    )


@pytest.mark.parametrize(
    ("label", "attributes", "features"),
    [
        ("no feature requested", ["gender"], []),
        ("no attribute requested", [], None),
        ("only the attribute itself", ["gender"], ["gender"]),
    ],
)
def test_a_screen_over_zero_pairs_is_not_complete(label, attributes, features):
    """True is the neutral value, and it was published for a screen that
    measured nothing.

    BEFORE, measured 2026-09-29 on 200 rows where height IS a critical proxy for
    gender (``feature_columns=None``: 1 result, 'critical', r=0.761,
    ``complete`` True): ``feature_columns=['not_a_column']`` gave 0 results with
    ``complete`` FALSE, ``screens_not_run`` 1 and 1 warning (the BGL5 gate
    working), while ``feature_columns=[]``, ``protected_attributes=[]`` and
    ``feature_columns=['gender']`` EACH gave 0 results, ``complete`` **True**,
    ``screens_not_run`` ``[]`` and ZERO warnings. ``ProxyScreenResult`` documents
    "complete: True only when every requested pair was actually screened" and
    implemented it as ``return not self.screens_not_run``, which cannot tell
    "every requested pair was screened" from "no pair was requested".
    """
    kwargs = {} if features is None else {"feature_columns": features}
    result, messages = _caught(
        lambda: identify_proxy_variables(_proxy_frame(), attributes, **kwargs)
    )

    assert len(result) == 0
    assert result.complete is False
    assert result.not_screened_reason is not None
    assert "0 feature/attribute pair(s) to screen" in result.not_screened_reason
    assert "not a finding that no feature is a proxy" in result.not_screened_reason
    assert "0 feature/attribute pair(s) to screen" in _joined(messages)


def test_the_zero_pair_disclosure_survives_the_analyzer_boundary():
    """A correct measurement is worth nothing at a surface that drops it.

    BEFORE, measured 2026-09-29 through the two public analyzer entries with
    ``feature_columns=[]``: ``analyze_proxies()`` returned 0 results with
    ``complete`` True and ``full_analysis()`` reported
    ``proxy_screen_complete`` True with ``risk_summary['not_assessed'] == 0``,
    because both rebuild the coverage predicate field by field and neither
    carried the new term.
    """
    df = _proxy_frame()

    empty, _ = _caught(
        lambda: FeatureEngineeringAnalyzer(df, ["gender"], feature_columns=[]).analyze_proxies()
    )
    report, _ = _caught(
        lambda: FeatureEngineeringAnalyzer(df, ["gender"], feature_columns=[]).full_analysis()
    )

    assert len(empty) == 0
    assert empty.complete is False
    assert report.proxy_screen_complete is False
    assert report.to_dict()["not_screened_reason"] is not None
    assert "0 feature/attribute pair(s) to screen" in report.not_screened_reason


def test_control_the_proxy_screen_still_finds_the_critical_proxy():
    """Over-correction control with the REAL coefficient and risk band.

    A refusal that fires on every request would pass both pins above.
    """
    df = _proxy_frame()

    auto, _ = _caught(lambda: identify_proxy_variables(df, ["gender"]))
    named, _ = _caught(lambda: identify_proxy_variables(df, ["gender"], feature_columns=["height"]))

    for result in (auto, named):
        assert len(result) == 1
        assert result.complete is True
        assert result.not_screened_reason is None
        assert result[0].feature == "height"
        assert result[0].risk_level.value == "critical"
        assert result[0].correlation == pytest.approx(0.761, abs=5e-3)

    analysed, _ = _caught(lambda: FeatureEngineeringAnalyzer(df, ["gender"]).analyze_proxies())
    assert analysed.complete is True
    assert len(analysed) == 1


def test_control_a_misspelled_feature_is_still_recorded_as_a_skipped_pair():
    """The BGL5 gate this one sits beside must still fire: a requested feature
    that is not a column is a could-not-check, not a zero-pair request."""
    result, messages = _caught(
        lambda: identify_proxy_variables(
            _proxy_frame(), ["gender"], feature_columns=["not_a_column"]
        )
    )

    assert len(result) == 0
    assert result.complete is False
    assert len(result.screens_not_run) == 1
    assert result.not_screened_reason is None, "one pair WAS requested"
    assert "is not a column of this DataFrame" in _joined(messages)
