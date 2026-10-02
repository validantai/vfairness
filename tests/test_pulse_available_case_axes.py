"""Regression R-B1: one attribute's absence must not erase another's finding.

acad091 (2026-09-30) rightly stopped ``run_pulse`` from minting a "missing"
pseudo-group out of rows whose protected value is absent. It did so by dropping
every row that lacked ANY chosen attribute (listwise deletion), so each
attribute's reading became conditional on every OTHER attribute having been
recorded. On the bundled recruitment dataset that kept 204 of 2700 rows (only
applicants with a recorded disability) and the planted gender penalty, on a
column with no missing value at all, read ``tone=pass``: the CI recall gate
fell from 18 to 17.

These pins hold both halves at once:

* a complete attribute is read on ALL of its rows, whatever another attribute
  lacks (the regression);
* an absent value is still never a group, anywhere in the payload (acad091).
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse.orchestrator import run_pulse

_HERE = os.path.dirname(os.path.abspath(__file__))
_RECRUITMENT = os.path.join(_HERE, "fixtures", "recruitment_fairness_dataset.csv")

#: Labels ``str()`` mints out of an absence, plus the old fillna sentinel.
_PHANTOM_LABELS = {"missing", "nan", "<NA>", "NaT", "None", ""}


def _run(df, inputs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pulse(df, inputs).get("data") or {}


def _row(data, attr):
    return next(r for r in data.get("perVariable") or [] if r.get("attribute") == attr)


def _frame(n: int = 800) -> pd.DataFrame:
    """A REAL gender penalty (A ~0.6, B ~0.3) on a column with NO absence, and
    a second attribute recorded ONLY on some of group A's rows.

    The second attribute's recorded rows are all in gender group A, so the
    complete-row frame holds no group-B row at all: listwise deletion turns a
    real two-group disparity into a one-group attribute with nothing to compare.
    """
    rng = np.random.default_rng(7)
    half = n // 2
    gender = np.array(["A"] * half + ["B"] * half, dtype=object)
    pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.3, half)])
    disability = np.array([None] * n, dtype=object)
    disability[:120] = np.array(["X", "Y"] * 60, dtype=object)
    return pd.DataFrame(
        {
            "gender": gender,
            "disability": disability,
            "prediction": pred,
            "feature": rng.normal(size=n),
        }
    )


def test_a_complete_attribute_is_read_on_every_row_whatever_another_lacks():
    df = _frame()
    data = _run(df, {"protected_attributes": ["gender", "disability"]})
    pv = _row(data, "gender")

    groups = {g["label"]: (g["n"], g["rate"]) for g in pv["groups"]}
    # At acad091 this was {'A': (120, ...)}: one group, nothing compared.
    assert set(groups) == {"A", "B"}, groups
    assert groups["A"][0] == 400 and groups["B"][0] == 400, groups
    for label in ("A", "B"):
        expected = float(df.loc[df["gender"] == label, "prediction"].mean())
        assert groups[label][1] == pytest.approx(expected, abs=1e-12)
    assert pv["worstGroup"] == "B"
    assert pv["gap"] == pytest.approx(groups["A"][1] - groups["B"][1], abs=1e-12)
    assert pv["significant"] is True
    assert pv["tone"] in ("warn", "critical"), pv["tone"]

    # The same rows reach the disparity matrix, not the complete-row subset.
    dm = (data.get("disparityMatrix") or {}).get("gender") or {}
    assert dm.get("min_ratio") == pytest.approx(groups["B"][1] / groups["A"][1], abs=1e-9), dm


def test_the_partly_recorded_attribute_is_read_on_its_own_recorded_rows_only():
    df = _frame()
    data = _run(df, {"protected_attributes": ["gender", "disability"]})
    pv = _row(data, "disability")
    groups = {g["label"]: g["n"] for g in pv["groups"]}
    assert groups == {"X": 60, "Y": 60}, groups

    prep = data.get("dataPreparation") or {}
    assert prep.get("rowsExcludedNoProtectedValue") == 680
    assert prep.get("rowsExcludedNoProtectedValueByAttribute") == {"disability": 680}
    details = " ".join(str(b.get("detail")) for b in prep.get("binning") or [])
    assert "no recorded value" in details and "not a demographic group" in details


def test_no_group_is_ever_built_out_of_an_absence_anywhere_in_the_payload():
    """acad091's guarantee, asked of the WHOLE payload rather than one key."""
    df = _frame()
    for absent in (None, np.nan, pd.NA, "   "):
        holed = df.copy()
        holed["disability"] = holed["disability"].astype(object)
        holed.loc[120:, "disability"] = absent
        data = _run(holed, {"protected_attributes": ["gender", "disability"]})
        for r in data.get("perVariable") or []:
            labels = {str(g["label"]).strip() for g in r.get("groups") or []}
            assert not labels & _PHANTOM_LABELS, (repr(absent), r["attribute"], labels)
        for attr, m in (data.get("disparityMatrix") or {}).items():
            blob = json.dumps(m, default=str)
            for bad in ('"missing"', '"nan"', '"<NA>"'):
                assert bad not in blob, (repr(absent), attr, bad)
        headline = (data.get("verdict") or {}).get("headline") or ""
        assert "missing" not in headline and '""' not in headline, headline


def test_control_a_complete_frame_reads_identically_with_one_or_two_attributes():
    """No absence anywhere: the available-case view IS the complete frame."""
    df = _frame()
    df["disability"] = np.array(["X", "Y"] * 400, dtype=object)
    alone = _row(_run(df, {"protected_attributes": ["gender"]}), "gender")
    both_data = _run(df, {"protected_attributes": ["gender", "disability"]})
    both = _row(both_data, "gender")
    for key in ("groups", "gap", "fourFifthsRatio", "worstGroup", "referenceGroup"):
        assert alone[key] == both[key], key
    assert (both_data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 0


@pytest.mark.skipif(not os.path.exists(_RECRUITMENT), reason="bundled dataset absent")
def test_the_bundled_recruitment_gender_penalty_is_read_on_all_2700_rows():
    """The measured regression itself. gender has no missing value; the
    disability_status column is absent on 2386 rows once pandas reads its
    literal "None" as NaN, and religion on 864."""
    df = pd.read_csv(_RECRUITMENT).drop(
        columns=["true_qualification_score", "_bias_flags"], errors="ignore"
    )
    assert int(df["gender"].isna().sum()) == 0, "the fixture changed; this pins nothing"
    assert int(df["disability_status"].isna().sum()) > 2000, "the fixture changed"
    data = _run(
        df,
        {
            "domain": "recruitment / employment screening",
            "jurisdiction": "US",
            "source_kind": "tabular",
            "protected_attributes": ["gender", "disability_status", "religion"],
        },
    )
    pv = _row(data, "gender")
    groups = {g["label"]: g["n"] for g in pv["groups"]}
    expected = {str(k): int(v) for k, v in df["gender"].value_counts().items()}
    assert groups == expected, groups
    assert pv["worstGroup"] == "Non-binary"
    # Before the fix: 'pass' on 204 rows.
    assert pv["tone"] in ("warn", "critical"), pv["tone"]
    dis = {g["label"] for g in _row(data, "disability_status")["groups"]}
    assert dis == set(df["disability_status"].dropna().unique()), dis
