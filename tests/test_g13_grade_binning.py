"""G13 grading pins: ``vfairness.preprocessing.protected_binning``.

Binning DECIDES THE GROUPS every downstream fairness number is computed over.
If a bin edge, a bin label or the handling of an absent / impossible value is
wrong, every metric that follows measures the wrong partition and no downstream
test can see it, because the numbers are all internally consistent with the
wrong partition.

Five defects were measured here on 2026-09-30 and fixed:

1. ``_derive_age`` used ``.round()``, so a cohort within ~6 months of a band
   edge was banded ONE BAND TOO OLD. A date of birth 17.6 years ago derived
   18.0 and was published in ``18-24``: a minor in an adult band.
2. The raw-``age`` branch had no plausibility window at all (``_derive_age``
   always had one), and the outer band edges are infinite, so nothing can fall
   OUTSIDE the bands. The sentinels -999 and 999 were published as ``under_18``
   and ``65+`` with ``excluded == []`` and not one note.
3. The same branch gated on ``is_numeric_dtype``, so an ordinary ``read_csv``
   of an age column with one blank in it (dtype object) got quantile bands
   while the identical int64 column got the canonical ADEA-aligned bands, in
   one frame at once.
4. Every binning branch ends in ``.fillna("missing")``, minting a group out of
   the ABSENCE of a value, and nothing disclosed it: 48% of a frame sat in an
   eighth level while the only note said the bands were the seven canonical
   ones. Worse, that level counted as a second group, which DEFEATED the
   single-group disclosure the BGL5 pass exists to emit.
5. The blank string is the one door of absence ``nunique(dropna=True)`` counts
   as a value, so 50 rows of ``""`` came back usable as a single group named
   the empty string while the identical column of ``None`` was excluded.

Every class carries an over-correction control asserting the REAL partition of
a healthy column, because a binning that refuses everything passes every
refusal test and destroys the unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.protected_binning import (
    _AGE_BAND_EDGES,
    _AGE_BAND_LABELS,
    _MAX_PLAUSIBLE_AGE,
    PreparedProtected,
    prepare_protected_attributes,
)


def _band_of(age: float) -> str:
    """The band a COMPLETED-years age belongs in, read off the edges."""
    for label, lo, hi in zip(_AGE_BAND_LABELS, _AGE_BAND_EDGES[:-1], _AGE_BAND_EDGES[1:]):
        if lo <= age < hi:
            return label
    raise AssertionError(f"{age} falls in no band")


def _dob_years_ago(years: float) -> str:
    now = pd.Timestamp.now(tz="UTC")
    return (now - pd.Timedelta(days=years * 365.25)).strftime("%Y-%m-%d")


class TestAgeIsCompletedYearsNotRoundedYears:
    @pytest.mark.parametrize("true_age", [17.6, 17.9, 64.6, 64.9, 24.7, 34.8])
    def test_a_cohort_just_under_a_band_edge_stays_below_it(self, true_age):
        """You are 17 until your 18th birthday.

        Before: derived ``round(17.6) == 18.0`` -> band '18-24'. The expected
        band is DERIVED from the edges, never quoted, so this cannot go stale
        if the bands are ever revised.
        """
        df = pd.DataFrame({"date_of_birth": [_dob_years_ago(true_age)] * 50})

        prep = prepare_protected_attributes(df, ["date_of_birth"])

        levels = set(prep.frame["date_of_birth"].dropna().unique())
        assert levels == {_band_of(int(true_age))}, (
            f"true age {true_age} must band as completed years "
            f"({_band_of(int(true_age))}), not as the rounded-up year"
        )

    def test_control_a_cohort_in_the_middle_of_a_band_is_unmoved(self):
        """The fix must not shift anyone who was already right."""
        for true_age in (30.2, 30.5, 30.9, 47.1, 70.3):
            df = pd.DataFrame({"date_of_birth": [_dob_years_ago(true_age)] * 50})
            prep = prepare_protected_attributes(df, ["date_of_birth"])
            assert set(prep.frame["date_of_birth"].dropna().unique()) == {_band_of(int(true_age))}

    def test_control_a_real_spread_of_births_still_fills_its_real_bands(self):
        rng = np.random.default_rng(3)
        ages = rng.uniform(19.0, 78.0, 400)
        df = pd.DataFrame({"date_of_birth": [_dob_years_ago(a) for a in ages]})

        prep = prepare_protected_attributes(df, ["date_of_birth"])

        assert prep.usable == ["date_of_birth"]
        assert set(prep.frame["date_of_birth"].dropna().unique()) == {
            _band_of(int(np.floor(a))) for a in ages
        }
        assert [label for label, _d in prep.notes] == ["Binning"]


class TestAnImpossibleAgeIsNotAnAge:
    def test_sentinels_are_not_published_as_protected_groups(self):
        """Before: 30 rows of -999 published as 'under_18' and 30 rows of 999
        as '65+', with excluded == [] and notes == [one Binning note]."""
        real = list(range(20, 60))
        df = pd.DataFrame({"age": [-999] * 30 + [999] * 30 + real})

        prep = prepare_protected_attributes(df, ["age"])

        counts = prep.frame["age"].value_counts().to_dict()
        for band in ("under_18", "65+"):
            assert counts.get(band, 0) == sum(1 for a in real if _band_of(a) == band), (
                f"{band} must hold only the rows whose real age is in it"
            )
        detail = " ".join(d for label, d in prep.notes if label == "Coverage")
        assert f"outside 0-{_MAX_PLAUSIBLE_AGE}" in detail
        assert "60 of 100 rows" in detail

    def test_a_column_of_nothing_but_sentinels_yields_no_group_at_all(self):
        """Before: usable ['age'] with 200 rows in 'under_18'."""
        df = pd.DataFrame({"age": [-990 - i for i in range(20)] * 10})

        prep = prepare_protected_attributes(df, ["age"])

        assert prep.usable == []
        assert "no value after preparation" in dict(prep.excluded)["age"]

    def test_control_the_plausible_extremes_are_kept(self):
        """0 (an infant) and 120 are real ages and must survive the window."""
        df = pd.DataFrame({"age": list(range(0, 121))})

        prep = prepare_protected_attributes(df, ["age"])

        assert prep.usable == ["age"]
        assert set(prep.frame["age"].dropna().unique()) == set(_AGE_BAND_LABELS)
        assert [label for label, _d in prep.notes] == ["Binning"]


class TestAnAgeReadFromACsvIsStillAnAge:
    def test_text_and_numeric_age_get_one_canonical_partition(self):
        """Before, in one frame at once: int64 -> the canonical ADEA bands,
        the identical values as strings -> '(19.999, 29.8]' quantile bands,
        which has no under_18 edge and no 40+ edge.
        """
        ages = list(range(20, 70)) * 3
        numeric = prepare_protected_attributes(pd.DataFrame({"age": ages}), ["age"])
        text = prepare_protected_attributes(pd.DataFrame({"age": [str(a) for a in ages]}), ["age"])

        expected = {_band_of(a) for a in ages}
        assert set(numeric.frame["age"].dropna().unique()) == expected
        assert set(text.frame["age"].dropna().unique()) == expected
        assert [label for label, _d in text.notes] == ["Binning"]

    def test_an_age_column_holding_words_is_not_forced_into_bands(self):
        """The over-correction control for the coercion: 'young'/'old' are not
        years, and inventing bands for them would be the worse answer."""
        words = [
            "young",
            "middle",
            "old",
            "ancient",
            "teen",
            "adult",
            "senior",
            "child",
            "infant",
            "elder",
            "mid",
            "late",
            "veryold",
        ]
        prep = prepare_protected_attributes(pd.DataFrame({"age": words * 10}), ["age"])

        levels = set(prep.frame["age"].dropna().unique())
        assert not (levels & set(_AGE_BAND_LABELS))
        assert levels <= set(words) | {"other"}


class TestTheMissingLabelIsDisclosedAndIsNotAGroup:
    def test_half_a_frame_in_the_minted_level_is_disclosed(self):
        """Before: one Binning note naming the seven canonical bands, while
        48% of the rows sat in an eighth level called 'missing'."""
        df = pd.DataFrame({"age": list(range(20, 80)) + [np.nan] * 55})

        prep = prepare_protected_attributes(df, ["age"])

        assert (prep.frame["age"] == "missing").sum() == 55
        coverage = [d for label, d in prep.notes if label == "Coverage"]
        assert coverage, "a minted group must not be silent"
        assert "55 of 115" in coverage[0] and "not a protected group" in coverage[0]

    def test_empty_rows_no_longer_defeat_the_single_group_disclosure(self):
        """60 ZIPs from ONE area plus 30 empty rows: the rolled-up column has
        levels {'ZIP 100xx', 'missing'}, so nunique was 2 and the BGL5
        single-group note (added three days earlier) silently stopped firing.
        'missing' is not a second group."""
        df = pd.DataFrame(
            {"zipcode": [f"10{i:03d}" for i in range(60)] + [None] * 20 + [pd.NaT] * 10}
        )

        prep = prepare_protected_attributes(df, ["zipcode"])

        assert prep.usable == ["zipcode"]
        assert prep.frame["zipcode"].nunique(dropna=True) == 2
        details = [d for label, d in prep.notes if label == "Coverage"]
        assert any("single group" in d for d in details)
        assert any("not a protected group" in d for d in details)

    def test_control_a_binned_column_with_nothing_absent_says_nothing_extra(self):
        """The existing BGL5 control, restated: a healthy binned column keeps
        exactly its Binning note. If this reddens, the disclosure is firing on
        columns that have nothing to disclose."""
        rng = np.random.default_rng(1)
        zips = [f"{a}00{i:02d}" for a in (1, 2, 3) for i in range(20)]
        prep = prepare_protected_attributes(
            pd.DataFrame({"zipcode": rng.choice(zips, 400)}), ["zipcode"]
        )
        assert [label for label, _d in prep.notes] == ["Binning"]
        assert prep.frame["zipcode"].nunique(dropna=True) == 3

    def test_control_a_real_value_named_missing_is_not_treated_as_absence(self):
        """A low-cardinality column used AS-IS is never in binned_cols, so a
        genuine level spelled 'missing' stays a group and gets no note."""
        df = pd.DataFrame({"gender": (["m", "f", "missing"] * 40)})

        prep = prepare_protected_attributes(df, ["gender"])

        assert prep.usable == ["gender"]
        assert prep.notes == []
        assert prep.frame["gender"].nunique(dropna=True) == 3


class TestTheBlankStringIsTheSixthDoorOfAbsence:
    @pytest.mark.parametrize(
        "absent",
        [None, float("nan"), pd.NA, pd.NaT, "", "   "],
        ids=["None", "nan", "pdNA", "NaT", "blank", "whitespace"],
    )
    def test_a_column_holding_only_absence_is_excluded_whichever_door(self, absent):
        """Before: the four null doors were excluded and '' came back usable
        as a single group NAMED the empty string. Same absence, two answers."""
        prep = prepare_protected_attributes(pd.DataFrame({"gender": [absent] * 50}), ["gender"])

        assert prep.usable == []
        assert "no values at all" in dict(prep.excluded)["gender"]

    def test_control_a_real_two_level_attribute_is_untouched(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"gender": rng.choice(["m", "f"], 100)})

        prep = prepare_protected_attributes(df, ["gender"])

        assert prep.usable == ["gender"]
        assert prep.excluded == []
        assert prep.notes == []


class TestPreparedProtectedItself:
    def test_the_dataclass_carries_the_frame_the_groups_were_binned_into(self):
        """The unit graded is the RESULT type: it must hand back the binned
        frame, not the caller's, and every disclosure list must be present."""
        df = pd.DataFrame({"age": list(range(18, 80))})
        prep = prepare_protected_attributes(df, ["age"])

        assert isinstance(prep, PreparedProtected)
        assert list(df["age"]) == list(range(18, 80)), "the input must not be mutated"
        assert prep.frame is not df
        assert set(prep.frame["age"].dropna().unique()) == {_band_of(a) for a in range(18, 80)}
        assert isinstance(prep.usable, list) and isinstance(prep.excluded, list)
        assert isinstance(prep.notes, list)

    def test_the_defaults_are_not_shared_between_instances(self):
        a = PreparedProtected(frame=pd.DataFrame(), usable=[])
        b = PreparedProtected(frame=pd.DataFrame(), usable=[])
        a.excluded.append(("x", "y"))
        a.notes.append(("l", "d"))
        assert b.excluded == [] and b.notes == []
