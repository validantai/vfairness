"""BGL-U01: the two capabilities the census never reached, executed at last.

Both were recorded ``BGL-C UNPROVEN``: no generic fixture could call them with
data, so nobody knew whether they fabricate. Unproven is a positive statement
that nothing is known, not a clean bill, and running them found one fabrication
each.

U01-A  ``detect_historical_patterns`` reported a correlation between a
       historically loaded column and a protected attribute that was computed
       over rows where the column had NEVER BEEN OBSERVED.
       ``pd.Categorical(series).codes`` gives every missing value the integer
       sentinel ``-1``, and ``pd.isna`` on an integer code array is ``False``
       everywhere, so the overlap mask could not drop those rows. Measured at
       the public entry on 400 rows whose ``zipcode`` was recorded only for
       white applicants (among the observed rows race is constant, so there is
       no correlation to take at all, and the same helper answers ``{}`` when
       handed exactly those rows): ``evidence['protected_correlations']`` came
       back ``{'race': 0.8115}``, that number cleared the caller's 0.3 gate, and
       the reported ``confidence`` rose from 0.80 to 1.0 with nothing on any
       channel saying the column was half empty.

U01-B  ``discover_intersectional_groups`` skipped every attribute that was not a
       column of the frame with a bare ``continue``, in three separate places,
       and recorded the skip nowhere. Measured at the public entry on 1200 rows
       where the schema had renamed ``race`` to ``ethnicity`` and the caller
       still asked for ``['sex', 'race']``, with the F x black cell selected at
       3% against 65% everywhere else: the function returned ``[]`` with
       ``not_assessable == []`` and zero warnings, which is the exact shape of
       "every intersection was checked and none was disparate", and
       ``rank_fairness_issues`` reported ``n_not_assessable=0``.

Every pin below is a PAIR: the refusal, and a healthy-data control proving the
capability still measures the disparity it exists to find. A fix that only makes
the refusal green would be a different defect. The controls check the FIELD and
not only the warning, because the assessment that started this campaign ran
under ``python -W ignore``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    discover_intersectional_groups,
    rank_fairness_issues,
)
from vfairness.preprocessing.bias_detection import historical as H
from vfairness.preprocessing.bias_detection.historical import (
    _compute_protected_correlations,
    _encode_for_correlation,
    detect_historical_patterns,
)


def _messages(records, needle: str):
    return [str(r.message) for r in records if needle in str(r.message)]


# ---------------------------------------------------------------------------
# U01-A. detect_historical_patterns
# ---------------------------------------------------------------------------


def _zipcode_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """HEALTHY: zipcode is redlining linked AND genuinely tracks race.

    This is the finding the capability exists to make, and every refusal pinned
    below has to leave it intact.
    """
    rng = np.random.default_rng(seed)
    race = rng.choice(["white", "black"], size=n, p=[0.6, 0.4])
    zipcode = np.where(
        race == "black",
        rng.integers(60600, 60610, n),
        rng.integers(60000, 60010, n),
    )
    return pd.DataFrame({"zipcode": zipcode, "race": race, "widget_count": rng.integers(0, 50, n)})


def _zipcode_observed_for_one_race_only(n_per_race: int = 200) -> pd.DataFrame:
    """DEGENERATE: zipcode recorded only for white applicants.

    Among the rows where zipcode WAS observed, race is a single value, so no
    zipcode/race correlation exists to be taken. Everything else about the frame
    is ordinary: 400 rows, ten zipcodes, two races.
    """
    rng = np.random.default_rng(7)
    race = np.array(["black"] * n_per_race + ["white"] * n_per_race)
    zipcode = np.array(
        [None] * n_per_race + [f"z{v}" for v in rng.integers(0, 10, n_per_race)],
        dtype=object,
    )
    return pd.DataFrame({"zipcode": pd.Series(zipcode, dtype=object), "race": race})


class TestU01AControlTheRealCorrelationIsStillMeasured:
    """The control. Nothing below may cost the capability this finding."""

    def test_a_genuine_zipcode_race_correlation_is_measured_and_raises_confidence(self):
        df = _zipcode_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = detect_historical_patterns(df, protected_attributes=["race"])

        by_feature = {r.feature: r for r in results}
        assert "zipcode" in by_feature, sorted(by_feature)
        zipc = by_feature["zipcode"]
        # The zipcodes are disjoint by race, so r is exactly -1.
        assert zipc.evidence["protected_correlations"]["race"] == pytest.approx(-1.0)
        assert "protected_correlations_not_assessable" not in zipc.evidence
        # 0.8 name confidence + 0.2 for a correlation above the 0.3 gate.
        assert zipc.confidence == pytest.approx(1.0)
        assert zipc.risk_level is H.HistoricalRiskLevel.HIGH
        # CONTROL on the other side: an ordinary column is not a finding.
        assert "widget_count" not in by_feature
        assert not _messages(caught, "detect_historical_patterns")

    def test_an_observed_column_with_missing_rows_is_still_correlated_over_what_is_there(self):
        """The refusal must be about NOT ENOUGH OVERLAP, never about any NaN at all.

        Half the zipcodes are missing, but both races remain represented among
        the rows that survive, so the correlation is real and must be reported.
        """
        df = _zipcode_frame()
        df.loc[df.index[::2], "zipcode"] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = detect_historical_patterns(df, protected_attributes=["race"])

        zipc = {r.feature: r for r in results}["zipcode"]
        assert zipc.evidence["protected_correlations"]["race"] == pytest.approx(-1.0)
        assert "protected_correlations_not_assessable" not in zipc.evidence
        assert not _messages(caught, "detect_historical_patterns")


class TestU01ATheMissingnessPatternIsNotACorrelation:
    """Pre-fix, measured 2026-09-17 at the public entry: 0.8115 and confidence
    1.0 on a column that was never observed for half the rows."""

    def test_the_fabricated_correlation_is_gone_and_the_reason_is_on_the_evidence(self):
        df = _zipcode_observed_for_one_race_only()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = detect_historical_patterns(df, protected_attributes=["race"])

        zipc = {r.feature: r for r in results}["zipcode"]
        # THREE STATES. Not a correlation, and not a 0.0 either: named, with why.
        assert "protected_correlations" not in zipc.evidence
        reasons = zipc.evidence["protected_correlations_not_assessable"]
        assert set(reasons) == {"race"}
        assert "constant" in reasons["race"]
        assert "200" in reasons["race"], reasons["race"]
        # The confidence that the fabricated 0.8115 had inflated to 1.0.
        assert zipc.confidence == pytest.approx(0.8)
        named = _messages(caught, "no correlation with a protected attribute")
        assert named, [str(c.message) for c in caught]
        assert "absent is not the same as uncorrelated" in named[0]

    def test_the_same_helper_answers_the_same_way_on_the_observed_rows_alone(self):
        """The decisive comparison: handed only the rows where zipcode exists,
        the helper has always refused. The whole frame must not be more
        confident than its own observed subset."""
        df = _zipcode_observed_for_one_race_only()
        observed = df[df["zipcode"].notna()]
        assert dict(_compute_protected_correlations(observed, "zipcode", ["race"])) == {}
        assert dict(_compute_protected_correlations(df, "zipcode", ["race"])) == {}

    def test_missing_is_encoded_as_nan_and_never_as_the_category_minus_one(self):
        """The mechanism, pinned directly: pandas uses -1 for "no category", and
        pd.isna on an integer code array cannot see it."""
        series = pd.Series(["a", None, "b", np.nan, "a"], dtype=object)
        encoded = _encode_for_correlation(series)
        assert encoded.dtype == np.dtype(float)
        assert np.isnan(encoded[[1, 3]]).all()
        assert not np.isnan(encoded[[0, 2, 4]]).any()
        assert -1 not in encoded[~np.isnan(encoded)]
        # And the raw pandas call this replaced still behaves the old way, so
        # the pin fails if anyone puts it back.
        assert (pd.Categorical(series).codes == -1).sum() == 2
        assert not pd.isna(pd.Categorical(series).codes).any()

    def test_numeric_columns_keep_their_missing_values_missing_too(self):
        encoded = _encode_for_correlation(pd.Series([1.0, np.nan, 3.0]))
        assert np.isnan(encoded[1])
        assert encoded[0] == 1.0 and encoded[2] == 3.0


@pytest.mark.parametrize(
    "label,frame,needle",
    [
        (
            "zero rows",
            pd.DataFrame(
                {"zipcode": pd.Series([], dtype=float), "race": pd.Series([], dtype=object)}
            ),
            "only 0 of 0 row(s)",
        ),
        (
            "nine rows, one under the overlap floor",
            pd.DataFrame(
                {
                    "zipcode": list(range(60601, 60610)),
                    "race": ["black"] * 5 + ["white"] * 4,
                }
            ),
            "below the minimum of 10",
        ),
        (
            "every zipcode missing",
            pd.DataFrame({"zipcode": [np.nan] * 200, "race": ["b"] * 100 + ["w"] * 100}),
            "only 0 of 200 row(s)",
        ),
        (
            "every zipcode identical",
            pd.DataFrame({"zipcode": [60601] * 200, "race": ["b"] * 100 + ["w"] * 100}),
            "is constant over the 200 row(s)",
        ),
        (
            "the protected attribute itself is never observed",
            pd.DataFrame(
                {
                    "zipcode": list(range(200)),
                    "race": pd.Series([None] * 200, dtype=object),
                }
            ),
            "only 0 of 200 row(s)",
        ),
    ],
)
def test_u01a_every_degenerate_frame_names_its_refusal(label, frame, needle):
    """Each of these makes the correlation undefined for a different reason, and
    each used to come back either as a number or as silence."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = detect_historical_patterns(frame, protected_attributes=["race"])

    zipc = {r.feature: r for r in results}["zipcode"]
    assert "protected_correlations" not in zipc.evidence, label
    reason = zipc.evidence["protected_correlations_not_assessable"]["race"]
    assert needle in reason, (label, reason)
    assert _messages(caught, "no correlation with a protected attribute"), label


def test_u01a_a_protected_attribute_that_is_not_a_column_is_named_not_dropped():
    df = _zipcode_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = detect_historical_patterns(df, protected_attributes=["race", "ethnicity"])

    zipc = {r.feature: r for r in results}["zipcode"]
    assert zipc.evidence["protected_correlations"]["race"] == pytest.approx(-1.0)
    reasons = zipc.evidence["protected_correlations_not_assessable"]
    assert reasons == {"ethnicity": "not a column of this frame"}
    assert _messages(caught, "no correlation with a protected attribute")


# ---------------------------------------------------------------------------
# U01-B. discover_intersectional_groups
# ---------------------------------------------------------------------------


def _intersectional_frame(n: int = 1200, seed: int = 3):
    """HEALTHY: a real intersectional disparity neither attribute shows alone."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["white", "black"], n, p=[0.7, 0.3])
    rate = np.where((sex == "F") & (race == "black"), 0.03, 0.65)
    y_pred = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({"sex": sex, "race": race}), y_pred


class TestU01BControlTheIntersectionalDisparityIsStillFound:
    def test_healthy_data_reports_the_hidden_intersectional_gap_and_nothing_else(self):
        df, y_pred = _intersectional_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["sex", "race"], y_pred)

        assert isinstance(out, list), "the return type changed and callers would break"
        assert len(out) == 1
        found = out[0]
        assert found["attributes"] == ["sex", "race"]
        assert found["disadvantaged_group"] == "F_black"
        assert found["disparity"] > 0.5
        assert found["reveals_more"] is True
        assert math.isfinite(found["hidden_disparity"]) and found["hidden_disparity"] > 0.2
        assert found["single_attributes_not_assessable"] == []
        # The whole point of the channel: on clean, fully assessable input it is
        # EMPTY, so a non-empty one always means something really was skipped.
        assert out.not_assessable == []
        assert out.zero_selection_alerts == []
        assert not _messages(caught, "discover_intersectional_groups")


class TestU01BARequestedAttributeThatIsNotAColumnCannotVanish:
    """Pre-fix, measured 2026-09-17: [] with not_assessable == [] and zero
    warnings, while an F x black cell selected at 3% against 65% was never
    looked at."""

    def test_the_renamed_column_is_named_on_the_not_assessable_channel(self):
        df, y_pred = _intersectional_frame()
        df = df.rename(columns={"race": "ethnicity"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["sex", "race"], y_pred)

        assert list(out) == []
        assert out.not_assessable, "an empty result with an empty reason list reads as clean"
        joined = " | ".join(out.not_assessable)
        assert "'race'" in joined and "not a column of this frame" in joined
        assert "not the same as having no disparity" in joined
        named = _messages(caught, "are not columns of the frame")
        assert named, [str(c.message) for c in caught]
        assert "SKIPPED, not cleared" in named[0]

    def test_it_reaches_the_public_entry_as_n_not_assessable(self):
        """The number a report actually prints. rank_fairness_issues counts this
        off `not_assessable`, never off a warning."""
        df, y_pred = _intersectional_frame()
        df = df.rename(columns={"race": "ethnicity"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analysis = rank_fairness_issues(
                df, y_pred, protected_columns=["sex", "race"], auto_detect=False
            )
        summary = analysis["summary"]
        assert summary["n_not_assessable"] > 0
        assert any("race" in entry for entry in summary["not_assessable"])
        assert any(
            "not the same as having no disparity" in entry for entry in summary["not_assessable"]
        )

    def test_when_no_attribute_survives_the_result_says_nothing_was_intersected(self):
        df, y_pred = _intersectional_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["gender", "ethnicity"], y_pred)

        assert list(out) == []
        joined = " | ".join(out.not_assessable)
        assert "'gender'" in joined and "'ethnicity'" in joined
        assert "no intersectional combination was formed at all" in joined
        assert _messages(caught, "no combination could be formed")

    def test_a_single_usable_attribute_is_not_an_intersectional_all_clear(self):
        df, y_pred = _intersectional_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["sex"], y_pred)

        assert list(out) == []
        assert any("nothing was intersected" in e for e in out.not_assessable)
        assert _messages(caught, "no combination could be formed")

    def test_max_combinations_below_two_also_says_it_intersected_nothing(self):
        df, y_pred = _intersectional_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = discover_intersectional_groups(df, ["sex", "race"], y_pred, max_combinations=1)
        assert list(out) == []
        assert any("max_combinations=1" in e for e in out.not_assessable)


class TestU01BAWarningOnlyRefusalIsASilentOne:
    """Both of these were raised as warnings and recorded nowhere, so the layer
    above counted `n_not_assessable=0` for a pass that evaluated nothing."""

    def test_a_combination_that_raised_lands_on_the_not_assessable_channel(self):
        df, y_pred = _intersectional_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Mismatched lengths: every call into the layer below raises.
            out = discover_intersectional_groups(df, ["sex", "race"], y_pred[:100])

        assert list(out) == []
        joined = " | ".join(out.not_assessable)
        assert "sex x race: the combination was not evaluated at all" in joined
        assert "InvalidDataError" in joined
        assert _messages(caught, "skipped combination")

    def test_a_single_attribute_that_could_not_be_assessed_is_recorded_too(self):
        """A constant column has no disparity of its own to measure. It was
        already excluded from the hidden-disparity maximum (R-5); now it is also
        counted among the things that could not be checked."""
        rng = np.random.default_rng(0)
        n = 400
        sex = rng.choice(["M", "F"], n)
        y_pred = np.where(sex == "M", rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
        df = pd.DataFrame({"sex": sex, "site": ["hq"] * n})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["sex", "site"], y_pred)

        assert len(out) == 1
        # R-5 is untouched: 'site' is named on the result dict and left out of
        # the maximum, and 'sex' was assessed so the comparison is still real.
        assert out[0]["single_attributes_not_assessable"] == ["site"]
        assert out[0]["reveals_more"] is False
        assert any(
            e.startswith("site (single attribute): its own disparity could not be assessed")
            for e in out.not_assessable
        ), out.not_assessable
        assert _messages(caught, "single-attribute disparity could not be assessed")


# ---------------------------------------------------------------------------
# U01-A, second consumer. `check_geographic_redlining_risk` adds 0.3 to a risk
# SUM on the strength of the same correlation, so the fabricated 0.8115 was
# worth a third of that score. A correlation that could not be taken has to be
# distinguishable there too, because an absent key subtracts the same 0.3 as a
# measured weak one.
# ---------------------------------------------------------------------------


def test_u01a_redlining_risk_still_scores_a_real_segregation_pattern():
    df = _zipcode_frame()
    df["approved"] = np.where(df["race"] == "black", 0.2, 0.8)
    result = H.check_geographic_redlining_risk(
        df, "zipcode", outcome_column="approved", demographic_column="race"
    )
    assert result["demographic_correlation"]["race"] == pytest.approx(-1.0)
    assert "demographic_correlation_not_assessable" not in result
    assert any("segregation pattern" in f for f in result["findings"])
    assert result["redlining_risk_score"] >= 0.3


def test_u01a_redlining_risk_does_not_score_a_missingness_pattern():
    df = _zipcode_observed_for_one_race_only()
    df["approved"] = np.linspace(0.0, 1.0, len(df))
    result = H.check_geographic_redlining_risk(
        df, "zipcode", outcome_column="approved", demographic_column="race"
    )
    # Pre-fix the sentinel correlation of 0.8115 cleared the 0.3 gate here too,
    # invented a "segregation pattern" finding and added 0.3 to the score.
    assert "demographic_correlation" not in result
    assert not any("segregation pattern" in f for f in result["findings"])
    reason = result["demographic_correlation_not_assessable"]["race"]
    assert "constant" in reason


def test_u01a_a_duplicated_column_name_is_refused_by_name_not_silently_mismeasured():
    """`df[name]` hands back a DataFrame when the name appears twice, and
    `pd.Categorical` accepted it and produced codes of the wrong length. The old
    blanket `except Exception` turned that into an empty dict with no reason."""
    df = pd.DataFrame(np.arange(40).reshape(20, 2), columns=["zipcode", "zipcode"])
    df["race"] = ["a", "b"] * 10
    result = _compute_protected_correlations(df, "zipcode", ["race"])
    assert dict(result) == {}
    assert "more than once" in result.not_assessable["race"]
