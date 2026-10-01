"""BGL7 PIN, batch B4-rep-pre-w2 (2026-09-29): preprocessing/bias_detection/statistical.py.

The SIBLING of the guard tests/test_bgl7_preprocessing_2.py pinned hours earlier.
That fix refused a time column holding EXACTLY ONE distinct value
(``if len(distinct) == 1``). One distinct value more reopened the whole defect,
because ``np.array_split`` cut the cohorts by ROW POSITION and ``np.argsort`` is
stable, so a boundary landing inside a block of rows tied on the same instant was
decided by where those rows sit in the FILE.

MEASURED BEFORE THIS CHANGE (canonical interpreter, OMP_NUM_THREADS=1), 60 rows,
``app_date = ['2024-01-01'] * 59 + ['2024-06-01']``, gender 21 'm' / 9 'f' /
9 'm' / 21 'f', which is the pin's own gender arrangement::

    as given               -> available=True severity='critical' PSI=0.4621 warnings=0
       "Across app_date, the gender mix shifted materially (PSI 0.46) between
        the earliest and latest applicants."
    sample(random_state=0) -> available=True severity='pass'     PSI=0.0178 warnings=0
    sample(random_state=1) -> available=True severity='warn'     PSI=0.1622 warnings=0
    sample(random_state=2) -> available=True severity='pass'     PSI=0.0715 warnings=0
    sample(random_state=3) -> available=True severity='pass'     PSI=0.0     warnings=0

0.4621 is the identical number the single-value guard records as ITS fabricated
before-state, and the same 60 rows merely reordered moved the published verdict
across three severities. There are no "earliest and latest applicants" among 59
rows stamped with one instant.

THE FIX cuts the periods over the DISTINCT INSTANTS and reads the cohorts back in
row space by comparing times, so cohort membership is decided by a row's
timestamp and by nothing else. The dominated frame above now comes out as the
59-against-1 comparison it really is and is refused by the existing
``_MIN_COHORT_RECORDED`` floor, with both counts named.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift

# The auditor's own frame, gender arrangement included.
_GENDER = ["m"] * 21 + ["f"] * 9 + ["m"] * 9 + ["f"] * 21
_TIED = pd.DataFrame({"app_date": ["2024-01-01"] * 59 + ["2024-06-01"], "gender": _GENDER})


def _run(frame: pd.DataFrame, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = detect_temporal_drift(frame, ["gender"], time_column="app_date", **kwargs)
    return out, [str(w.message) for w in caught]


def test_fiftynine_rows_tied_on_one_instant_no_longer_publish_a_critical():
    """BEFORE: available True, severity 'critical', compositionPSI 0.4621, zero
    warnings. 59 of the 60 rows carry one instant, so the 'earliest' and 'latest'
    cohorts the finding compared were rows 1-30 and 31-60 of the FILE."""
    out, warned = _run(_TIED)

    assert out["available"] is False, out
    assert out.get("drift") == [], out.get("drift")
    assert "severity" not in out or out["severity"] != "critical"
    # The counts are the finding: 59 recorded on one side, 1 on the other.
    assert "(59, 1)" in out["reason"], out["reason"]
    assert "could-not-check" in out["reason"]
    assert out["attributesWithTooFewObservations"]["gender"] == {
        "earliest": 59,
        "latest": 1,
        "required": 10,
    }
    assert len(warned) == 1, warned
    # The fabricated number must not survive anywhere in the document.
    assert "0.46" not in repr(out), repr(out)


@pytest.mark.parametrize("random_state", [0, 1, 2, 3], ids=lambda s: f"shuffled-{s}")
def test_the_verdict_no_longer_depends_on_the_order_of_the_rows(random_state):
    """BEFORE: the same 60 rows reordered gave critical / pass / warn / pass /
    pass. Cohort membership is now a fact about a row's timestamp, so every
    permutation of one frame must produce the IDENTICAL document."""
    given, given_warned = _run(_TIED)
    shuffled, shuffled_warned = _run(_TIED.sample(frac=1.0, random_state=random_state))

    assert shuffled == given, (shuffled, given)
    assert shuffled_warned == given_warned


def test_control_a_real_drift_is_still_critical_at_its_real_psi():
    """OVER-CORRECTION CONTROL, asserting the healthy case's REAL number.

    60 rows on 60 distinct dates, the same gender arrangement. The cut falls
    between two different instants, the cohorts are 30 rows each exactly as
    before, and the measured PSI is 0.6778, graded 'critical', with no warning. A
    fix that refused whenever any instant repeated would pass every assertion
    above and delete the screen's whole purpose.
    """
    spread = pd.DataFrame(
        {
            "app_date": pd.date_range("2024-01-01", periods=60, freq="D").astype(str),
            "gender": _GENDER,
        }
    )
    out, warned = _run(spread)

    assert out["available"] is True, out
    assert len(out["drift"]) == 1, out["drift"]
    entry = out["drift"][0]
    assert entry["attribute"] == "gender"
    assert entry["compositionPSI"] == 0.6778, entry
    assert entry["severity"] == "critical", entry
    assert "shifted materially (PSI 0.68)" in entry["plain"], entry["plain"]
    assert warned == [], warned


def test_control_two_instants_with_a_real_mix_on_each_are_still_measured():
    """OVER-CORRECTION CONTROL. Two distinct instants is the SMALLEST frame this
    screen can read, and it must still read it. 30 rows on one date (21 'm' /
    9 'f') against 30 on a later date (9 'm' / 21 'f') is a genuine
    earliest-against-latest comparison: PSI 0.6778, 'critical', no warning.

    This is the case the fix must keep, and it is one row-count away from the
    refused frame above, so it also shows the refusal is about the THIN COHORT and
    not about the number of instants.
    """
    balanced = pd.DataFrame(
        {"app_date": ["2024-01-01"] * 30 + ["2024-06-01"] * 30, "gender": _GENDER}
    )
    out, warned = _run(balanced)

    assert out["available"] is True, out
    assert out["drift"][0]["compositionPSI"] == 0.6778, out["drift"]
    assert out["drift"][0]["severity"] == "critical"
    assert warned == [], warned


def test_control_more_periods_than_instants_does_not_empty_the_latest_cohort():
    """The period count is capped at the number of distinct instants. Without the
    cap, cutting 2 instants into 4 periods leaves the LAST period empty, and an
    empty latest cohort reports 'gender carries no recorded value in either the
    earliest or the latest period', which would be a false statement about a
    column recorded on every row."""
    balanced = pd.DataFrame(
        {"app_date": ["2024-01-01"] * 30 + ["2024-06-01"] * 30, "gender": _GENDER}
    )
    out, warned = _run(balanced, n_periods=4)

    assert out["available"] is True, out
    assert out.get("attributesWithNoObservation", []) == []
    assert out["drift"][0]["compositionPSI"] == 0.6778, out["drift"]
    assert warned == [], warned


def test_control_the_single_instant_refusal_is_untouched():
    """The guard this fix is the sibling of must keep working: a fully constant
    time column still reports its own reason, naming the raw value, not the thin
    cohort reason added here."""
    constant = pd.DataFrame({"app_date": ["2024-01-01"] * 60, "gender": _GENDER})
    out, warned = _run(constant)

    assert out["available"] is False
    assert out["nDistinctTimeValues"] == 1
    assert "a single distinct value" in out["reason"], out["reason"]
    assert len(warned) == 1, warned
