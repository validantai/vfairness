"""BGL6 audit of batch F12: the fixes that did NOT survive re-attack.

ALL EIGHT RECORDS IN THIS FILE ARE CLOSED, as of 2026-09-29, and
``RECORDED_DEFECTS_STILL_OPEN`` is empty. Every test is now a PIN on the
corrected behaviour, and every docstring opens with "CLOSED 2026-09-29", says
what the fix was, carries the re-measured numbers and the output of the sabotage
that proved the pin can fail, and keeps the original record verbatim beneath a
"THE ORIGINAL RECORD follows." line.

Each record began life in the inverted style of ``tests/test_bgl4_*``: it asserted
the fabricated value the unit published, one input step away from the case the
BGL5 fix pinned, so that closing the defect turned the test red and forced it to
be inverted in its turn. That is what happened to all eight.

The file also carries the over-correction controls the fixes needed, because a
refusal that fires on everything passes every refusal pin: they assert the healthy
case's REAL numbers, not just its shape.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
import pytest

#: HOW MANY CLAIMS THE SECOND-ROUND AUDIT RECORDED IN THIS FILE, on 2026-09-28.
#:
#: A HISTORICAL FACT, and it must not move. The register used to take this count by
#: counting the test functions in the file, which was right on the day the audit
#: landed and wrong from the first fix onwards: inverting a witness into a pin
#: renames it, and a fix arrives with its own over-correction control, so closing
#: records made the audit look BIGGER. It had grown from 59 claims to 69 by the time
#: anybody added them up, on a page whose whole subject is not misstating what was
#: measured.
#:
#: Recovered from the audit's own baseline commit e6a5780, which is where every one
#: of these numbers comes from.
RECORDED_CLAIMS_AT_AUDIT = 11

#: THE TESTS IN THIS FILE THAT STILL RECORD AN OPEN DEFECT.
#:
#: A name is removed from this list in the SAME commit that fixes its defect and
#: inverts the test into a pin, so the two cannot drift. It is declared here rather
#: than inferred from pass/fail because a test that RECORDS a defect passes while the
#: defect is live, which is indistinguishable by execution from a pin that passes
#: because the defect is gone.
#:
#: Read by scripts/bgl6_register.py, published as counts in
#: docs/bgl6-audit-register.json and in QUALITY_AND_HARDENING.md, and checked by
#: tests/test_bgl6_register_is_honest.py, which refuses a name that is not a test
#: function in this module.
RECORDED_DEFECTS_STILL_OPEN: list[str] = []


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


def _dated(values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "gender": values,
        }
    )


# === row 1, CLOSED: detect_temporal_drift was keyed on TOTAL absence =========


def test_a_one_sided_protected_column_is_no_longer_reported_as_a_stable_mix():
    """CLOSED 2026-09-29. The guard was ``n_first_obs == 0 and n_last_obs == 0``, so
    it only fired when the column was unobserved in BOTH compared cohorts, and the
    comment above it justified that with "a column observed on ONE side only already
    lands on 'not_assessed' through the degenerate-PSI bound with a reason that says
    more".

    Measurement refuted the comment, which is what this record was: a column
    observed in 5 of the earliest 500 rows and in NONE of the latest 500 landed on
    severity 'pass' with "the gender mix is stable (PSI 0.09)", the clean top-level
    summary, ``attributesWithNoObservation`` empty and zero warnings. It never
    reached the degenerate bound, because the invented "missing" category is 99
    percent of both cohorts and hardly moves between them, so it was doing the
    measuring.

    The fix, in ``src/vfairness/preprocessing/bias_detection/statistical.py``, is
    ``or`` in place of ``and``, and the refuted claim is kept in the comment with the
    measured numbers beneath it rather than quietly deleted. The list's own wording
    ("carry no recorded value in either the earliest or the latest period") was
    already the ``or`` reading.

    Re-measured on the same 1000 rows: available False, 'gender' named in
    attributesWithNoObservation, drift [], no severity and no summary key at all
    (there is nothing to be clean about), and one warning.

    SABOTAGE (``or`` changed back to ``and``): the attribute falls through to the
    recorded-count floor instead, which is a could-not-check too but not the RIGHT
    one, so the reader is told the record was thin rather than absent.
        AssertionError: assert [] == ['gender']

    THE ORIGINAL RECORD follows.

    The guard is ``n_first_obs == 0 and n_last_obs == 0``, so it only fires
    when the column is unobserved in BOTH compared cohorts.

    The BGL5 row states that a column observed on ONE side only is deliberately
    left alone "because it already lands on 'not_assessed' through the
    degenerate-PSI bound with a reason that says more". Measured here: a column
    observed in 5 of the earliest 500 rows and in NONE of the latest 500 lands on
    severity 'pass' with "the gender mix is stable", the clean top-level summary
    and zero warnings. The invented "missing" category is doing the measuring.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift

    values = (["m"] * 5 + [None] * 495) + [None] * 500
    result, messages = _caught(lambda: detect_temporal_drift(_dated(values), ["gender"]))

    assert result["available"] is False
    assert result["attributesWithNoObservation"] == ["gender"]
    assert result["drift"] == []
    # No graded verdict of any kind, rather than a clean one.
    assert "severity" not in result
    assert "summary" not in result
    assert "no recorded value in either the earliest or the latest period" in result["reason"]
    assert "could-not-check" in result["reason"]
    assert len(messages) == 1, messages


def test_a_recorded_mix_that_flips_completely_is_no_longer_reported_as_stable():
    """CLOSED 2026-09-29. Both cohorts hold exactly ONE recorded value and they are
    different ones, so the recorded mix flips from 100 percent 'm' to 100 percent
    'f'. 998 of the 1000 rows carry no value, the zero-observation guard did not fire
    because each cohort has one, and the screen reported severity 'pass' with PSI
    0.0304 and no warning.

    Two changes in ``src/vfairness/preprocessing/bias_detection/statistical.py``, and
    the second is the root the first only covers:

    1. ``_MIN_COHORT_RECORDED`` (10, the floor ``_MIN_COMPARABLE_ROWS`` in the same
       module already uses): below it the cohort's mix is not a mix, and the
       attribute is reported in the new ``attributesWithTooFewObservations`` with
       BOTH counts, because "recorded in 1 of 500 and 1 of 500" is the finding.
    2. The composition PSI is now computed over the RECORDED values only. ``g``
       carries an invented "missing" category so the array holds no NaN, and letting
       that category into the compared distributions is what allowed it to outvote
       every real observation. An absent value is not a group, so it is not one of
       the categories either.

    Re-measured on the same 1000 rows: available False, drift [],
    attributesWithTooFewObservations {'gender': {'earliest': 1, 'latest': 1,
    'required': 10}}, and one warning. And the change is not only about the floor:
    the same frame with 20 recorded values per cohort, which CLEARS the floor, was
    measured at PSI 0.0304 / 'pass' before and PSI 27.63 / 'not_assessed' after, with
    "the earliest and latest period share only 0 of 2 observed gender value(s)".

    SABOTAGE (``ef``/``al`` pointed back at ``g[first]``/``g[last]``, so the invented
    category is back in the compared distributions, the floor left in place, on the
    20-recorded frame):
        AssertionError: assert 0.8477 == 27.631 +/- 1.0e-02

    THE ORIGINAL RECORD follows.

    Both cohorts hold exactly ONE recorded value and they are different ones,
    so the recorded mix flips from 100 percent 'm' to 100 percent 'f'.

    998 of the 1000 rows carry no value, the guard does not fire because each
    cohort has one, and the screen reports severity 'pass' with PSI 0.03.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift

    values = (["m"] + [None] * 499) + (["f"] + [None] * 499)
    result, messages = _caught(lambda: detect_temporal_drift(_dated(values), ["gender"]))

    assert result["available"] is False
    assert result["drift"] == []
    assert result["attributesWithTooFewObservations"] == {
        "gender": {"earliest": 1, "latest": 1, "required": 10}
    }
    assert "too thin to read a group mix from" in result["reason"]
    assert len(messages) == 1, messages

    # The floor is the smaller half of the fix. With 20 recorded values per cohort,
    # which clears it, the flip is now SEEN instead of being averaged away by the
    # invented category: PSI 27.63 and 'not_assessed', where it was 0.0304 and
    # 'pass'. Without this the fix would only have moved the threshold.
    thicker = (["m"] * 20 + [None] * 480) + (["f"] * 20 + [None] * 480)
    graded, graded_messages = _caught(lambda: detect_temporal_drift(_dated(thicker), ["gender"]))
    assert graded["available"] is True
    assert graded["drift"][0]["compositionPSI"] == pytest.approx(27.631, abs=1e-2)
    assert graded["drift"][0]["severity"] == "not_assessed"
    assert "share only 0 of 2 observed gender value(s)" in graded["drift"][0]["notAssessedReason"]
    assert "NOT graded" in graded["summary"]
    assert len(graded_messages) == 1, graded_messages


def test_a_well_recorded_mix_is_still_graded_with_its_real_psi():
    """OVER-CORRECTION CONTROL for the two records above. A floor that refused real
    data, or a recorded-only PSI that broke the ordinary path, would pass both pins
    while making the screen useless. Measured after the fix:

    * 1000 rows of alternating a/b, fully recorded: PSI 0.0, 'pass', the clean
      summary, both coverage fields empty, zero warnings.
    * 1000 rows drifting 80/20 to 20/80: PSI 1.6636, 'critical', the drift-risk
      summary naming gender.
    * 90 recorded and 10 unrecorded per cohort with a stable recorded mix: still PSI
      0.0 and 'pass', so dropping the invented category does not turn a partly
      recorded column into a finding.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift

    stable, quiet = _caught(lambda: detect_temporal_drift(_dated(["a", "b"] * 500), ["gender"]))
    assert stable["available"] is True
    assert stable["severity"] == "pass"
    assert stable["drift"][0]["compositionPSI"] == 0.0
    assert stable["attributesWithNoObservation"] == []
    assert stable["attributesWithTooFewObservations"] == {}
    assert "No material temporal drift" in stable["summary"]
    assert quiet == []

    moved = ["a"] * 400 + ["b"] * 100 + ["a"] * 100 + ["b"] * 400
    drifted, _ = _caught(lambda: detect_temporal_drift(_dated(moved), ["gender"]))
    assert drifted["drift"][0]["compositionPSI"] == pytest.approx(1.6636, abs=1e-3)
    assert drifted["severity"] == "critical"
    assert "gender population" in drifted["summary"]

    partly = (["a", "b"] * 45 + [None] * 10) * 2
    tolerated, tolerated_quiet = _caught(lambda: detect_temporal_drift(_dated(partly), ["gender"]))
    assert tolerated["severity"] == "pass"
    assert tolerated["drift"][0]["compositionPSI"] == 0.0
    assert tolerated_quiet == []


# === row 2, CLOSED: detect_specification_bias counted comparable rows, then
#            correlated values it had invented ===============================


def test_a_perfect_target_leak_is_no_longer_masked_by_the_median_it_imputed():
    """CLOSED 2026-09-29. The ``n_pairs >= _MIN_COMPARABLE_ROWS`` floor counts the
    rows where the feature and the outcome BOTH hold a value, and the correlation was
    then computed over ALL rows with ``x.fillna(x.median())``, so the invented values
    diluted the measurement the floor had just qualified.

    Measured before: 'leak' is IDENTICAL to 'approved' on all 25 rows the floor
    counted (|r| = 1.0 over those rows), the 15 rows where the feature was never
    recorded were filled with its median, |r| over the imputed 40 rows came out at
    0.6547, below the 0.97 bound, and the screen published severity 'pass' while
    checksRun asserted "target_leakage (1 feature column(s) scanned)". A perfect
    target leak reported as clean, by a check that said it had run.

    The fix, in ``src/vfairness/preprocessing/bias_detection/statistical.py``,
    correlates over the comparable rows only, which is what the sibling
    ``circular_evaluation`` check in the same module already did. A column with no
    spread exactly where the outcome is recorded is named in checksNotRun instead of
    producing a 0/0, tested with ``np.unique`` rather than ``std() == 0``, because a
    standard deviation is an accumulated statistic and is exactly 0.0 only for some
    values at some n.

    Re-measured on the same 40 rows: severity 'critical', one target_leakage finding
    reading "|r|=1.00 over the 25 row(s) where both hold a value", which is the number
    the floor counted and the number the sentence now names.

    SABOTAGE (``np.corrcoef(xc, yc)`` changed back to ``np.corrcoef(xv, yv)``):
        AssertionError: assert [] == ['target_leakage']

    THE ORIGINAL RECORD follows.

    The new ``n_pairs >= _MIN_COMPARABLE_ROWS`` floor counts the rows where
    the feature and the outcome BOTH hold a value, and then computes the
    correlation over ALL rows with ``x.fillna(x.median())``.

    Measured: 'leak' is IDENTICAL to 'approved' on all 25 rows the floor
    counted (|r| = 1.0 over those rows), the 15 rows where the feature was never
    recorded are filled with its median, |r| over the imputed 40 rows is 0.6547,
    and the screen publishes severity 'pass' while checksRun asserts
    "target_leakage (1 feature column(s) scanned)". The sibling circular check in
    the same fix WAS restricted to the comparable rows; this one was not.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_specification_bias

    y = np.array([0, 1] * 20, dtype=float)
    leak = y.copy()
    leak[25:] = np.nan
    frame = pd.DataFrame({"approved": y, "leak": leak})

    comparable = ~np.isnan(leak)
    assert int(comparable.sum()) == 25
    assert abs(float(np.corrcoef(leak[comparable], y[comparable])[0, 1])) == 1.0
    # The number the imputation used to publish, kept so the two are side by side.
    imputed = np.where(comparable, leak, np.nanmedian(leak))
    assert abs(float(np.corrcoef(imputed, y)[0, 1])) == pytest.approx(0.6547, abs=1e-4)

    result, _ = _caught(lambda: detect_specification_bias(frame, outcome="approved"))

    assert [f["kind"] for f in result["findings"]] == ["target_leakage"]
    assert result["severity"] == "critical"
    assert "target_leakage (1 feature column(s) scanned)" in result["checksRun"]
    # The sentence names the rows it was measured over, so |r| cannot be read as a
    # statement about the 15 rows nobody recorded.
    plain = result["findings"][0]["plain"]
    assert "|r|=1.00" in plain, plain
    assert "over the 25 row(s) where both hold a value" in plain, plain


def test_a_feature_unrelated_to_the_outcome_is_still_not_a_leak():
    """OVER-CORRECTION CONTROL for the record above. Restricting the correlation to
    the comparable rows must not turn every partly recorded column into a critical:
    a check that fired on everything would pass that pin. Measured after the fix,
    with the same 15 unrecorded rows: severity 'pass', no finding, and the scan still
    reported as RUN over 1 column, so the clean verdict is one the check earned.

    Also measured: the same perfect leak with no missing value at all is still
    critical, at |r|=1.00 over 40 rows, so the fix did not make the finding depend on
    there being a gap.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_specification_bias

    y = np.array([0, 1] * 20, dtype=float)
    noise = pd.Series(np.random.default_rng(0).normal(size=40))
    noise[25:] = np.nan
    quiet, _ = _caught(
        lambda: detect_specification_bias(
            pd.DataFrame({"approved": y, "noise": noise}), outcome="approved"
        )
    )
    assert quiet["findings"] == []
    assert quiet["severity"] == "pass"
    assert "target_leakage (1 feature column(s) scanned)" in quiet["checksRun"]

    whole, _ = _caught(
        lambda: detect_specification_bias(
            pd.DataFrame({"approved": y, "leak": y.copy()}), outcome="approved"
        )
    )
    assert [f["kind"] for f in whole["findings"]] == ["target_leakage"]
    assert "over the 40 row(s) where both hold a value" in whole["findings"][0]["plain"]


def test_a_feature_flat_exactly_where_the_outcome_is_recorded_is_named_not_scanned():
    """The third state the comparable-rows restriction introduces. A column can carry
    plenty of spread overall and be CONSTANT on exactly the rows where the outcome was
    recorded, and then there is no correlation to read, only a 0/0. Measured after the
    fix, with 'approved' recorded on the first 25 of 40 rows and 'flat' holding 5.0 on
    all of them: nothing scanned, severity 'pass' with target_leakage absent from
    checksRun entirely, and checksNotRun naming the column with its shared-row count.
    """
    from vfairness.preprocessing.bias_detection.statistical import detect_specification_bias

    approved = np.array([0, 1] * 20, dtype=float)
    approved[25:] = np.nan
    flat = np.arange(40.0)
    flat[:25] = 5.0

    result, _ = _caught(
        lambda: detect_specification_bias(
            pd.DataFrame({"approved": approved, "flat": flat}), outcome="approved"
        )
    )

    assert result["findings"] == []
    assert not any("target_leakage (" in c for c in result["checksRun"])
    said = " ".join(result["checksNotRun"])
    assert "flat (no spread over its 25 shared row(s))" in said, said


# === row 3, CLOSED 2026-09-28: the unrecorded-group guard sat BELOW the tomek
#            dispatch =======================================================


def _balancing_frame():
    rng = np.random.default_rng(0)
    g = ["m"] * 60 + ["f"] * 50 + [None] * 20
    n = len(g)
    frame = pd.DataFrame({"gender": g, "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = np.array(([0, 1] * (n // 2)) + [0] * (n % 2))
    return frame, y


def test_the_tomek_branch_no_longer_treats_no_recorded_value_as_a_protected_group():
    """FIXED 2026-09-28, and this is now the pin.

    ``get_resampled_data`` dispatches to ``_tomek`` BEFORE it computes
    ``recorded``, so for ``method='tomek'`` the absence of a record was still a
    group key and the repair the sibling path got on 2026-09-27 could not reach it.
    Measured on this 130-row frame BEFORE: group_totals_before {'None': 20, ...},
    group_totals_after {'None': 19, ...}, real rows deleted and accounted to that
    pseudo-group, fit_result.warnings == [] and no Python warning.

    AFTER, measured on the same frame: group_totals_before {'f': 50, 'm': 60},
    group_totals_after {'f': 41, 'm': 46}, n_rows_without_a_recorded_group 20 with
    19 of them surviving, and one fit_result warning that says they are NOT a
    protected group. The rows are still RETURNED: excluding them from the totals is
    the fix, deleting them would be a second and larger defect.
    """
    from vfairness.preprocessing.feature_engineering.data_balancing import SyntheticResampler

    frame, y = _balancing_frame()
    resampler = SyntheticResampler(
        protected_attributes=["gender"], method="tomek", strategy="undersample", random_state=0
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        resampler.fit(frame, y)
        out, _y_out = resampler.get_resampled_data(frame, y)

    metrics = resampler.fit_result.fit_metrics
    assert "None" not in metrics["group_totals_before"], metrics["group_totals_before"]
    assert "None" not in metrics["group_totals_after"], metrics["group_totals_after"]
    assert metrics["group_totals_before"] == {"f": 50, "m": 60}
    assert metrics["group_totals_after"] == {"f": 41, "m": 46}

    # Counted under their own key, in both directions, so a reader can see how many
    # there were and how many the boundary cleaning removed.
    assert metrics["n_rows_without_a_recorded_group"] == 20
    assert metrics["n_rows_without_a_recorded_group_after"] == 19

    # And disclosed, in words that refuse the pseudo-group reading.
    disclosures = [w for w in resampler.fit_result.warnings if "no recorded value" in w]
    assert len(disclosures) == 1, resampler.fit_result.warnings
    assert "NOT a protected group" in disclosures[0]
    assert "could-not-check" in disclosures[0]

    # The rows themselves are not deleted for being unrecorded: 24 rows were removed
    # as Tomek links out of 130, and 19 of the 20 unrecorded rows survive.
    assert len(out) == 106
    assert metrics["n_rows_removed"] == 24


def test_the_tomek_disclosure_no_longer_calls_the_absence_of_a_record_a_group():
    """FIXED 2026-09-28, and this is now the pin on the WORDING.

    BEFORE, when it spoke at all, it spoke about 'None' as a protected group:
    "removed 1 of 22 row(s) unevenly across protected groups: {'None': 2, ...}.
    Group(s) ['None'] are left with fewer than two rows." A reader was told the
    absence of a record is a group that lost rows.

    AFTER, measured on the same 22-row frame: one disclosure, "2 of 22 row(s) have
    no recorded value for ['gender'], so they are NOT a protected group and are
    excluded from the group totals", and no sentence anywhere that names 'None' as
    a group. The wording matters as much as the arithmetic: this is the surface the
    reader has.
    """
    from vfairness.preprocessing.feature_engineering.data_balancing import SyntheticResampler

    frame = pd.DataFrame(
        {
            "gender": ["m"] * 10 + ["f"] * 10 + [None] * 2,
            "f1": list(np.arange(10.0)) + list(np.arange(100.0, 110.0)) + [1000.0, 1000.1],
            "f2": [0.0] * 22,
        }
    )
    y = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0, 1])
    resampler = SyntheticResampler(
        protected_attributes=["gender"], method="tomek", strategy="undersample", random_state=0
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        resampler.fit(frame, y)
        resampler.get_resampled_data(frame, y)

    said = " ".join(resampler.fit_result.warnings)
    assert "no recorded value" in said, said
    assert "NOT a protected group" in said, said
    # The pseudo-group must not be NAMED as a group anywhere in what a reader sees.
    assert "'None'" not in said, said
    assert "{'None'" not in said, said


# === row 5, CLOSED: compare_to_benchmark dropped a group on a key collision and
#            named it nowhere ================================================


def test_an_observed_label_folded_into_a_benchmark_key_is_no_longer_dropped():
    """CLOSED 2026-09-29. ``{_group_key(k): v for k, v in actual.items()}``
    OVERWRITES on a collision, so the second spelling of a group discarded the
    first one's share instead of adding to it, and 200 of 300 rows left the
    calculation without being named anywhere.

    The fix adds ``_fold_shares`` to
    ``src/vfairness/preprocessing/bias_detection/representation.py``, which SUMS on a
    collision and returns the raw labels that were combined, applied to both the
    observed and the benchmark side. The summing half is what
    ``_analyze_single_attribute`` in the same module already did, in the loop above
    ``_canonical_group``; this gives that behaviour a name so the two entry points
    cannot drift again. The fold is also DISCLOSED, in
    ``result['labels_folded_together']`` and a warning, because
    ``actual_distribution`` keeps 'm' and 'M' apart while the ratios answer to one
    key, and a reader comparing the two has to be able to see why.

    Re-measured on the same 300 rows: representation_ratios {'m': 2.0} with gap
    +0.5, a SURPLUS for the group that holds every observed row where a deficit of
    -0.167 was published before; and the chi-squared half of the same return now
    RUNS, chi2 300.0 with p = 3.3e-67, because the 200 rows the overwrite had
    dropped are no longer missing from the coverage check.

    SABOTAGE (``folded[key] = folded.get(key, 0.0) + float(share)`` changed back to
    ``folded[key] = float(share)``):
        AssertionError: assert 0.6666666666666666 == 2.0 +/- 2.0e-06

    THE ORIGINAL RECORD follows.

    ``actual_normalized = {_group_key(k): v for k, v in actual.items()}``
    OVERWRITES on a collision instead of summing, and ``groups_seen`` is built
    from the folded keys, so the label that lost the collision cannot appear in
    ``groups_not_measured``.

    Measured on 300 rows that are ALL male, spelt 'm' (200) and 'M' (100):
    representation_ratios {'m': 0.667} with gap -0.167 (a deficit for a group
    holding every observed row), measurement_status 'measured',
    groups_not_measured [] and 'M' named nowhere, while the chi-squared half of
    the SAME return refuses because "66.7% of the rows fall in groups the
    benchmark never mentions".
    """
    from vfairness.preprocessing.bias_detection.representation import compare_to_benchmark

    df = pd.DataFrame({"gender": ["m"] * 200 + ["M"] * 100})
    result, messages = _caught(lambda: compare_to_benchmark(df, "gender", {"m": 0.5, "f": 0.5}))

    # The reader's own spellings are still reported as they were recorded.
    assert set(result["actual_distribution"]) == {"m", "M"}

    # Both spellings are one group now, and it holds every observed row: twice its
    # expected half, a surplus of +0.5, not a deficit of -0.167.
    assert dict(result["representation_ratios"])["m"] == pytest.approx(2.0)
    assert result["gaps"]["m"] == pytest.approx(0.5)
    assert result["gaps"]["m"] > 0

    # And the fold is visible, under the folded key, naming the labels combined.
    assert result["labels_folded_together"]["observed"] == {"m": ["m", "M"]}
    assert result["labels_folded_together"]["benchmark"] == {}
    said = " ".join(messages)
    assert "treated as ONE group and their shares ADDED" in said, said

    # The chi-squared half stops refusing, because no row is missing from it now.
    assert result["chi_squared_statistic"] == pytest.approx(300.0)
    assert result["pvalue"] is not None
    assert result["pvalue"] < 1e-60
    assert "groups the benchmark never mentions" not in said, said
    assert result["measurement_status"] == "measured"
    assert result["groups_not_measured"] == []


def test_a_benchmark_with_no_folded_label_is_untouched_and_unannounced():
    """OVER-CORRECTION CONTROL for the record above, with the real numbers. A fold
    that fired on every input, or a disclosure that warned on every call, would pass
    that pin and make the field worthless. Measured after the fix on 150 'm' and 150
    'f' against a 50/50 benchmark: ratios {'f': 1.0, 'm': 1.0}, gaps both 0.0,
    labels_folded_together empty on both sides, measurement_status 'measured' and
    ZERO warnings.
    """
    from vfairness.preprocessing.bias_detection.representation import compare_to_benchmark

    df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})
    result, messages = _caught(lambda: compare_to_benchmark(df, "gender", {"m": 0.5, "f": 0.5}))

    assert dict(result["representation_ratios"]) == {"f": 1.0, "m": 1.0}
    assert dict(result["gaps"]) == {"f": 0.0, "m": 0.0}
    assert result["labels_folded_together"] == {"observed": {}, "benchmark": {}}
    assert result["measurement_status"] == "measured"
    assert messages == []


# === row 6, CLOSED: a positive finite share this module could not divide by ==


def test_a_subnormal_but_positive_finite_benchmark_no_longer_publishes_an_infinity():
    """CLOSED 2026-09-29. ``_is_a_divisible_share`` tests the DIVISOR, and the
    division was then performed unchecked, so every positive finite value including
    the subnormals was admitted and the QUOTIENT could still leave the float range.

    The fix adds ``_finite_ratio`` to
    ``src/vfairness/preprocessing/bias_detection/representation.py``, which returns
    the quotient only when it is finite and None otherwise. It sits above BOTH call
    sites (``calculate_representation_ratio`` and the loop in
    ``compare_to_benchmark``) rather than being repeated in each, because they share
    the precondition and a guard written per branch is the guard one branch ends up
    missing. The two reasons are named separately in ``not_measured_reason``: "the
    benchmark is not a share" and "the benchmark is a share too small to divide this
    by" are different facts about the input.

    Re-measured on the same 300 rows with benchmark_proportion=5e-324:
    representation_ratio None, is_overrepresented None, is_underrepresented None,
    deficit_count None, measurement_status 'could_not_measure', not_measured_reason
    "benchmark_proportion=5e-324 is positive and finite, but actual_proportion=0.5
    divided by it is not a finite number, so the ratio has no value to report", and
    one warning where there were none. The same input through
    ``compare_to_benchmark`` now names the group in groups_not_measured with
    measurement_status 'partial' instead of publishing an infinite ratio for it.

    SABOTAGE (``if not math.isfinite(quotient): return None`` changed to
    ``if False: return None``):
        AssertionError: assert inf is None

    THE ORIGINAL RECORD follows.

    ``_is_a_divisible_share`` admits any value with ``0.0 < v < inf``, and the
    QUOTIENT can still be non-finite.

    Measured on 300 rows against benchmark_proportion=5e-324:
    representation_ratio inf with is_overrepresented True, measurement_status
    'measured' and zero warnings, from the function whose docstring promises None
    "when benchmark_proportion is not a positive finite number".
    """
    from vfairness.preprocessing.bias_detection.representation import (
        calculate_representation_ratio,
        compare_to_benchmark,
    )

    df = pd.DataFrame({"sex": ["m"] * 150 + ["f"] * 150})
    result, messages = _caught(lambda: calculate_representation_ratio(df, "sex", "m", 5e-324))

    assert result["representation_ratio"] is None
    assert result["is_overrepresented"] is None
    assert result["is_underrepresented"] is None
    assert result["deficit_count"] is None
    assert result["measurement_status"] == "could_not_measure"
    assert "not a finite number" in (result["not_measured_reason"] or "")
    # The observed share itself WAS measured and is still reported: the refusal is
    # about the ratio, and collapsing the whole result would lose a real number.
    assert result["actual_proportion"] == pytest.approx(0.5)
    assert len(messages) == 1, messages
    assert "UNMEASURED" in messages[0]

    # The sibling call site, guarded by the same helper.
    sibling, sibling_messages = _caught(
        lambda: compare_to_benchmark(
            pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150}),
            "gender",
            {"m": 5e-324, "f": 0.5},
        )
    )
    assert sibling["groups_not_measured"] == ["m"]
    assert "m" not in dict(sibling["representation_ratios"])
    assert sibling["measurement_status"] == "partial"
    assert "quotient is not finite" in " ".join(sibling_messages)


def test_an_ordinary_benchmark_share_still_divides_to_its_real_ratio():
    """OVER-CORRECTION CONTROL for the record above. A quotient test that refused
    anything it did not like would pass that pin while destroying the function.
    Measured after the fix on 150 'm' of 300 rows against benchmark 0.4: ratio 1.25
    exactly, deficit_count -29, is_overrepresented True, measurement_status
    'measured', not_measured_reason None and no warning.
    """
    from vfairness.preprocessing.bias_detection.representation import (
        calculate_representation_ratio,
    )

    df = pd.DataFrame({"sex": ["m"] * 150 + ["f"] * 150})
    result, messages = _caught(lambda: calculate_representation_ratio(df, "sex", "m", 0.4))

    assert result["representation_ratio"] == pytest.approx(1.25)
    assert result["deficit_count"] == -29
    assert result["is_overrepresented"] is True
    assert result["measurement_status"] == "measured"
    assert result["not_measured_reason"] is None
    assert messages == []


# === rows 7 and 8, CLOSED 2026-09-28: the probes were keyed on TOTAL background
#                   collapse ==================================================


def _scaffolded(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return (np.abs(arr[:, 1] - arr[:, 0]) < 0.5).astype(float)


def _partly_frozen_background() -> np.ndarray:
    """Columns 0 and 1, the only two the scaffold reads, are frozen; an
    irrelevant third column varies, which is all ``_unperturbable_background``
    asks for."""
    rng = np.random.RandomState(7)
    a = rng.normal(0, 1, 400)
    manifold = np.column_stack([a, a])
    frozen = np.repeat(manifold[:1], 400, axis=0)
    third = np.random.default_rng(1).normal(0, 5, (400, 1))
    return np.hstack([frozen, third])


def test_neither_probe_clears_a_scaffold_on_a_background_they_could_not_move():
    """FIXED 2026-09-28, and this is now the pin. Same defect as BGL6 F04-2,
    recorded independently here, and closed by the same change to both twins.

    ``n_moving`` counted the columns with spread and returned "" (perturbable) as
    soon as ONE had it, so partial collapse was not refused.

    BEFORE, against the same scaffold the fix's control finds at gap 0.640:
    _probe_gap 0.0, the single draw (False, ''), the multi-seed twin flag False with
    confidence 1.0, mean_gap 0.0, fired_fraction 0.0, and ZERO warnings. That output
    was byte-identical to the pre-fix output for the fully frozen background.

    AFTER: the guard names the frozen column, the single draw refuses with that
    reason, and the multi-seed twin returns flag None with NaN confidence and one
    warning. The gap itself is still 0.0, which is the point: it is 0.0 by
    construction for every model, so it can never be evidence either way.
    """
    from vfairness.xai.diagnostics.adversarial import (
        _probe_gap,
        _unperturbable_background,
        multi_seed_adversarial_probe,
        slack_adversarial_probe,
    )

    bg = _partly_frozen_background()
    cannot = _unperturbable_background(bg)
    assert cannot, "the guard still calls a partly frozen background perturbable"
    assert "no usable spread" in cannot, cannot

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gap = _probe_gap(
            predict_fn=_scaffolded, x=bg[0], background=bg, n_perturbations=200, seed=7
        )
        flag, reason = slack_adversarial_probe(predict_fn=_scaffolded, x=bg[0], background=bg)
        out = multi_seed_adversarial_probe(
            predict_fn=_scaffolded, x=bg[0], background=bg, n_seeds=4
        )

    # The gap REFUSES now. It was 0.0, and 0.0 is exactly what it is by construction
    # along a frozen column for every model, so it could never be evidence either
    # way. _probe_gap carries the same guard, so the refusal starts at the bottom.
    assert math.isnan(gap), gap
    # The single draw refuses instead of returning a clean False with no reason.
    assert flag is not False or reason, (flag, reason)
    assert "spread" in (reason or ""), reason
    # And the multi-seed twin could-not-checks rather than clearing the scaffold.
    assert out.flag is None, out.flag
    assert math.isnan(out.confidence)
    assert "COULD NOT CHECK" in (out.reason or ""), out.reason
    assert [str(w.message) for w in rec], "the probe refused and said nothing"


# === rows 9 and 10, CLOSED: the relative floor was a knife edge =============


@pytest.fixture(scope="module")
def _lime_world():
    pytest.importorskip("lime")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=800).fit(X, y)
    x = X[int(np.argmax(X[:, 2]))]
    # A jitter of 1e-8, three orders of magnitude above the 1e-12 the fix pinned
    # and still nothing but numerical noise on features living at 1.3 to 2.5.
    noisy = np.tile(x, (50, 1)) + np.random.default_rng(3).normal(scale=1e-8, size=(50, 4))
    return model, X, x, noisy


_LIME_KW = dict(subject_id="s", model_hash="m", data_hash="d", feature_names=["a", "b", "c", "d"])


def test_lime_no_longer_reports_an_all_zero_explanation_as_fully_measured(_lime_world):
    """CLOSED 2026-09-29. ``_MIN_RELATIVE_SPREAD = 1e-9`` moved the knife edge; it
    did not remove it, because the DISCLOSURE had only two states.
    ``unperturbable_features`` is a list: a column is in it (not measured) or absent
    from it, and absent is read as measured. There was no state for "varied, but
    across a window so small that the number returned is not a statement about this
    feature".

    The fix, in ``src/vfairness/xai/explainers/lime_adapter.py``, adds that third
    state: ``_background_relative_spread`` measures every examined column,
    ``_MIN_INTERPRETABLE_RELATIVE_SPREAD`` (1e-4) selects the ones whose window is
    too small, and they are published as ``barely_perturbable_features`` with the
    MEASURED numbers beside them in ``background_relative_spread``, plus one warning
    that says a near-zero contribution for those names is a could-not-check and not
    a finding of no influence. Publishing the measurement is what removes the knife
    edge: the floor now decides only whether the warning fires, and it sits four
    orders above the failing background and four below the healthy one.

    Re-measured on the same input: attributions UNCHANGED at 1.2e-14 (they are what
    lime returns, and the fix does not invent a different number),
    unperturbable_features still [] (the columns were varied, so saying otherwise
    would be false), barely_perturbable_features ['a', 'b', 'c', 'd'],
    background_relative_spread a=2.265e-08 b=4.466e-08 c=1.828e-08 d=2.466e-08, and
    exactly one warning where there were none.

    SABOTAGE (``_MIN_INTERPRETABLE_RELATIVE_SPREAD`` 1e-4 changed back to 1e-9):
        AssertionError: assert [] == ['a', 'b', 'c', 'd']

    THE ORIGINAL RECORD follows.

    ``_MIN_RELATIVE_SPREAD = 1e-9`` moved the knife edge; it did not remove it.

    Measured with a background whose relative spread is 1.8e-08 to 4.5e-08,
    against the same LogisticRegression whose prediction moves 0.9999996 to
    6.3e-07 when 'c' flips sign: every attribution comes back at 1e-15 or below,
    fidelity 0.047, ``params['unperturbable_features'] == []`` (the key a machine
    reader is told to branch on, positively asserting every feature was measured)
    and ZERO warnings.
    """
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    model, _X, x, noisy = _lime_world
    flipped = x.copy()
    flipped[2] = -flipped[2]
    assert model.predict_proba(x.reshape(1, -1))[0, 1] > 0.99
    assert model.predict_proba(flipped.reshape(1, -1))[0, 1] < 0.01

    explanation, messages = _caught(
        lambda: LimeExplainer().explain_local(
            model, x, noisy, instance_id="i", stability_reruns=2, num_samples=400, **_LIME_KW
        )
    )

    # The attributions are the SAME as before the fix. That is deliberate: they are
    # what lime returns for this background, and the defect was never that the
    # number was wrong, it was that the number was published as a measurement.
    contributions = {a.feature: abs(float(a.contribution)) for a in explanation.attributions}
    assert max(contributions.values()) < 1e-13

    # Not in the hard list, because these columns WERE varied.
    assert explanation.params["unperturbable_features"] == []
    # In the third-state list, because the window they were varied across is
    # 1.8e-08 of their own size.
    assert explanation.params["barely_perturbable_features"] == ["a", "b", "c", "d"]

    # And the measured numbers are published, so the verdict does not rest on where
    # the floor sits.
    spreads = explanation.params["background_relative_spread"]
    assert set(spreads) == {"a", "b", "c", "d"}
    assert spreads["c"] == pytest.approx(1.828e-08, rel=0.05)
    assert max(spreads.values()) == pytest.approx(4.466e-08, rel=0.05)

    said = " ".join(messages)
    assert len(messages) == 1, messages
    assert "COULD NOT CHECK" in said, said
    assert "not a finding of no influence" in said, said


def test_lime_healthy_background_is_still_measured_with_its_real_numbers(_lime_world):
    """OVER-CORRECTION CONTROL for the two records above. A disclosure that fired on
    real data would be worse than the defect, and a blanket "flag everything" would
    pass both pins. Measured after the fix on the 300-row background: c = 0.5667 and
    the dominant attribution, relative spread 1.85 / 1.79 / 1.89 / 1.76, both lists
    empty and no warning. The global path over the same background is clean on all
    three rows.

    Also measured: the whole fixture scaled by 1e-12, whose per-column spread is
    ~2e-12 in absolute terms and identical in shape to the jitter, is NOT flagged,
    because the test is relative to the column's own magnitude. That is the
    over-correction an absolute floor would have caused.
    """
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    model, X, x, _noisy = _lime_world

    explanation, messages = _caught(
        lambda: LimeExplainer().explain_local(
            model, x, X, instance_id="i", stability_reruns=2, num_samples=400, **_LIME_KW
        )
    )
    contributions = {a.feature: float(a.contribution) for a in explanation.attributions}
    assert contributions["c"] == pytest.approx(0.5667, abs=0.02)
    assert max(contributions, key=lambda k: abs(contributions[k])) == "c"
    assert explanation.params["unperturbable_features"] == []
    assert explanation.params["barely_perturbable_features"] == []
    assert min(explanation.params["background_relative_spread"].values()) > 1.0
    assert messages == []

    rows, global_messages = _caught(
        lambda: LimeExplainer().explain_global(
            model, X[:3], X, stability_reruns=2, num_samples=400, **_LIME_KW
        )
    )
    assert [ex.params["barely_perturbable_features"] for ex in rows] == [[], [], []]
    assert global_messages == []

    # Tiny units are not a collapsed background.
    scaled, scaled_messages = _caught(
        lambda: LimeExplainer().explain_local(
            model,
            x * 1e-12,
            X * 1e-12,
            instance_id="i",
            stability_reruns=2,
            num_samples=200,
            **_LIME_KW,
        )
    )
    assert scaled.params["barely_perturbable_features"] == []
    assert scaled_messages == []


def test_lime_global_no_longer_inherits_it_row_by_row(_lime_world):
    """CLOSED 2026-09-29 by the same change, which is why this record was carried
    separately: the global path is a loop over ``explain_local``, so it inherited the
    two-state disclosure once per row and three explanations came back asserting
    that every feature had been measured.

    Re-measured over the same background: still three Explanations and still a
    maximum absolute attribution of 4.5e-04 (unchanged, it is what lime returns),
    ``unperturbable_features == []`` on all three, but now
    ``barely_perturbable_features == ['a', 'b', 'c', 'd']`` on all three and THREE
    warnings where there were none, one per row, so a caller iterating the rows
    cannot read any of them as measured.

    SABOTAGE (``_MIN_INTERPRETABLE_RELATIVE_SPREAD`` 1e-4 changed back to 1e-9):
        AssertionError: assert [[], [], []] == [['a', 'b', 'c', 'd'], ...]

    THE ORIGINAL RECORD follows.

    The global path over the same background returns three Explanations with a
    maximum absolute attribution of 4.5e-04, ``unperturbable_features == []`` on
    all three and zero warnings, which is the BGL5 row's own stated BEFORE value.
    """
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    model, X, _x, noisy = _lime_world
    rows, messages = _caught(
        lambda: LimeExplainer().explain_global(
            model, X[:3], noisy, stability_reruns=2, num_samples=400, **_LIME_KW
        )
    )

    assert len(rows) == 3
    largest = max(abs(float(a.contribution)) for ex in rows for a in ex.attributions)
    assert largest == pytest.approx(4.5e-4, abs=5e-5)
    assert [ex.params["unperturbable_features"] for ex in rows] == [[], [], []]
    assert [ex.params["barely_perturbable_features"] for ex in rows] == [
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d"],
    ]
    assert len(messages) == 3, messages
    assert all("COULD NOT CHECK" in m for m in messages)


# === row 13, CLOSED: a fifth state the summary did not have =================


class _Rpc:
    def execute(self):
        class _Response:
            data = [{"msg_id": 1, "message": {"owner": "o", "job_id": "j"}}]

        return _Response()


class _Client:
    def rpc(self, *_a, **_k):
        return _Rpc()


class _Writer:
    def __init__(self):
        self._client = _Client()


def _crashing_worker(monkeypatch, runner, *, crash_every: bool):
    """A loop over three messages whose dispatch raises on all of them, or, when
    ``crash_every`` is False, on the first one only."""
    loop = runner.WorkerLoop(config=runner.WorkerConfig(poll_interval_s=0.0))
    monkeypatch.setattr(runner, "SupabaseWriter", lambda **_k: _Writer())
    seen = {"n": 0}

    def _boom(_msg):
        seen["n"] += 1
        if seen["n"] >= 3:
            loop.stop()
        if crash_every or seen["n"] == 1:
            raise RuntimeError("explainer crashed on this job")

    monkeypatch.setattr(loop, "_dispatch", _boom)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    return loop, seen


def test_the_worker_no_longer_reports_a_clean_stop_when_every_job_crashed(monkeypatch, caplog):
    """CLOSED 2026-09-29. ``run`` read the POLL counters alone to decide its
    summary, and a poll is not a tick: a run in which every dispatched job raised
    has three ANSWERED polls and zero failed ones, so it fell through to the else.

    The fix adds ``_ticks_completed`` / ``_ticks_failed`` in
    ``src/vfairness/xai/worker/runner.py``, counted around the loop's own
    ``except Exception``, and two branches above the clean-stop line: all ticks
    failed returns the new ``EXIT_WORK_ALL_FAILED`` (4, distinct from the 3 that
    means the queue was never read), some ticks failed warns and returns 0. The
    ``_polls_failed`` branch no longer returns early, or a partly answered poll
    history would have skipped the tick verdict.

    Re-measured on the same three crashing messages: polls (3, 0) unchanged, ticks
    (0 completed, 3 failed), "NOT ONE of them completed" at ERROR, no "stopped
    cleanly" anywhere, status 4.

    SABOTAGE (``return EXIT_WORK_ALL_FAILED`` changed to ``return 0``):
        AssertionError: assert 0 == 4

    THE ORIGINAL RECORD follows.

    ``run`` counts POLLS, so a run in which every dispatched job raised has
    three answered polls and zero failed ones.

    Measured: three ERROR records reading "worker loop tick failed", then the
    summary line "XAI worker stopped cleanly" and a returned status of 0, which
    is the only channel ``main`` and a supervisor read. "stopped cleanly" is the
    exact sentence the block this fix rewrote exists to prevent.
    """
    from vfairness.xai.worker import runner

    loop, seen = _crashing_worker(monkeypatch, runner, crash_every=True)
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")

    status = loop.run()

    assert seen["n"] == 3
    # The poll counters are unchanged: the queue DID answer. That is exactly why
    # they could never have caught this.
    assert (loop._polls_answered, loop._polls_failed) == (3, 0)
    assert (loop._ticks_completed, loop._ticks_failed) == (0, 3)
    assert "stopped cleanly" not in caplog.text, caplog.text
    assert "NOT ONE of them completed" in caplog.text
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert status == runner.EXIT_WORK_ALL_FAILED
    # Distinct from the never-read-the-queue status, because they are different
    # states and a supervisor has to be able to tell them apart.
    assert status != runner.EXIT_NOTHING_MEASURED


def test_a_worker_that_crashed_on_one_job_of_three_is_not_reported_as_all_failed(
    monkeypatch, caplog
):
    """OVER-CORRECTION CONTROL for the record above. A blanket "any crash is a
    failed run" would restart-loop on a single bad job, so the degraded case has to
    keep its real numbers: one failed tick, two completed, a WARNING that names
    "1 of 3", no ERROR summary and a status of 0.
    """
    from vfairness.xai.worker import runner

    loop, seen = _crashing_worker(monkeypatch, runner, crash_every=False)
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")

    status = loop.run()

    assert seen["n"] == 3
    assert (loop._ticks_completed, loop._ticks_failed) == (2, 1)
    assert "1 of 3 dispatched tick(s)" in caplog.text
    assert "NOT ONE of them completed" not in caplog.text
    assert status == 0


def test_an_idle_worker_is_still_a_clean_stop(monkeypatch, caplog):
    """OVER-CORRECTION CONTROL. An empty queue dispatches nothing, so both tick
    counters stay at 0 and the new branches must not fire at all. Measured: three
    answered polls, ticks (0, 0), "XAI worker stopped cleanly", nothing above INFO
    and a status of 0.
    """
    from vfairness.xai.worker import runner

    class _EmptyRpc:
        def execute(self):
            class _Response:
                data = []

            return _Response()

    class _EmptyWriter:
        def __init__(self):
            self._client = type("_C", (), {"rpc": lambda _s, *_a, **_k: _EmptyRpc()})()

    loop = runner.WorkerLoop(config=runner.WorkerConfig(poll_interval_s=0.0))
    monkeypatch.setattr(runner, "SupabaseWriter", lambda **_k: _EmptyWriter())
    ticks = {"n": 0}

    def _fake_sleep(_s):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            loop.stop()

    monkeypatch.setattr(runner.time, "sleep", _fake_sleep)
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")

    status = loop.run()

    assert (loop._polls_answered, loop._polls_failed) == (3, 0)
    assert (loop._ticks_completed, loop._ticks_failed) == (0, 0)
    assert "XAI worker stopped cleanly" in caplog.text
    assert status == 0
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
