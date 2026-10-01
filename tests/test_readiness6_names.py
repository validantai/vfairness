"""Column-name interpretation: the collisions, and the over-corrections.

Every test here is a PAIR. The refusal test pins the defect that was
measured; the control pins the genuine detection that a careless fix would
destroy. A fix that only makes the refusal green is not a fix, it is a
different defect, so the controls are not optional extras.

The measurements each test refers to were taken on 2026-09-10 against the
code as it then stood, through the public entry points wherever one exists.
"""

from __future__ import annotations

import inspect
import math
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from vfairness import _names
from vfairness._names import (
    name_token_list,
    resolve_attribute_key,
    tokens_contain,
)
from vfairness.evaluation.vfairness_metrics import discovery as D
from vfairness.preprocessing import protected_binning as B
from vfairness.preprocessing.bias_detection import historical as H
from vfairness.preprocessing.bias_detection import proxy as P
from vfairness.preprocessing.bias_detection import statistical as S
from vfairness.preprocessing.bias_detection.detector import BiasDetector
from vfairness.preprocessing.feature_engineering import correlation as C

# ---------------------------------------------------------------------------
# THE SHARED HELPER
# ---------------------------------------------------------------------------
#
# The point of this wave was not "fix six tokenisers", it was "stop having
# six". Five modules carried their own copy and a fix reached exactly one of
# them, so the same collision was live in four places for months.

_OWNING_MODULES = (P, C, B, D, S, H)


def test_every_name_based_detector_imports_the_shared_tokeniser():
    """Source-level: no module may grow its own tokenising regex again."""
    for mod in _OWNING_MODULES:
        src = inspect.getsource(mod)
        assert "from vfairness._names import" in src, (
            f"{mod.__name__} does not import the shared name helper; a local "
            "copy of the tokeniser is exactly how this class of defect recurs."
        )
        # The pre-fix regexes, in every spelling they appeared in.
        for banned in (r'r"[A-Za-z][a-z]*|[0-9]+"', r'r"[A-Za-z]+|[0-9]+"'):
            assert banned not in src, f"{mod.__name__} still carries {banned}"


def test_the_three_named_helpers_are_the_same_function():
    """Behavioural: one input, one answer, in all three wrappers."""
    for name in ("AGE", "average_spend", "customerAge", "género", "Größe", "BIRTH_DATE"):
        shared = set(name_token_list(name))
        assert P._name_tokens(name) == shared
        assert C._name_tokens(name) == shared
        assert set(B._tokens(name)) == shared


def test_shared_tokeniser_is_whole_token_and_unicode_aware():
    assert name_token_list("AVERAGE_SPEND") == ["average", "spend"]
    assert name_token_list("SUSSEX_COUNTY") == ["sussex", "county"]
    assert name_token_list("customerAge") == ["customer", "age"]
    assert name_token_list("género") == ["genero"]
    assert name_token_list("raça") == ["raca"]
    assert name_token_list("âge") == ["age"]
    assert name_token_list("état_civil") == ["etat", "civil"]
    assert name_token_list("Größe") == ["grosse"]
    # CONTROL: contiguity, so a multi-word keyword is not satisfied by two
    # unrelated tokens sitting in the same name.
    assert tokens_contain("census_tract_id", "census_tract")
    assert not tokens_contain("census_id_and_tract_of_land", "census_tract")


# ---------------------------------------------------------------------------
# DEFECT 1 -- historical.py invented seven discrimination findings
# ---------------------------------------------------------------------------

_MANUFACTURING = {
    "aroma_score": "roma",
    "award_amount": "ward",
    "valve_pressure": "alv",
    "product_name": "name",
    "singapore_office": "gap",
    "pipeline_id": "ipe",
    "iris_diameter": "iris",
}


def _frame(columns, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.random(n) for c in columns})


def test_a_manufacturing_table_produces_no_discrimination_findings():
    """Measured before: 7 findings, one of them CRITICAL 'Roma and Traveller
    Discrimination' because a column is called aroma_score."""
    results = H.detect_historical_patterns(_frame(_MANUFACTURING))
    assert results == [], [
        (r.feature, r.risk_level.value, r.pattern_type, r.evidence["matched_keyword"])
        for r in results
    ]


@pytest.mark.parametrize("column,keyword", sorted(_MANUFACTURING.items()))
def test_each_invented_match_is_refused_individually(column, keyword):
    assert not H._keyword_matches_column(column, keyword)


@pytest.mark.parametrize(
    "column,keyword",
    [
        # CONTROL: the compound spellings the reporter singled out.
        ("zipcode", "zipcode"),
        ("plz_code", "plz"),
        # CONTROL: the genuine columns each ambiguous keyword exists for.
        ("employee_name", "name"),
        ("applicant_name", "applicant_name"),
        ("iris_scan", "iris_scan"),
        ("employment_gap", "employment_gap"),
        ("census_tract", "census_tract"),
        ("block_group", "block_group"),
        ("roma_flag", "roma"),
        ("alv_status", "alv"),
        ("credit_score", "credit_score"),
    ],
)
def test_genuine_columns_still_match(column, keyword):
    assert H._keyword_matches_column(column, keyword)


def test_genuine_columns_still_produce_findings_end_to_end():
    """CONTROL: every one of these must still be found, or the fix above has
    simply switched the failure from false positives to false negatives."""
    columns = [
        "zipcode",
        "plz_code",
        "employee_name",
        "surname",
        "iris_scan",
        "employment_gap",
        "census_tract",
        "sozialhilfe",
        "roma_flag",
        "alv_status",
        "credit_score",
        "postcode",
    ]
    found = {r.feature for r in H.detect_historical_patterns(_frame(columns))}
    assert found == set(columns), f"not detected: {sorted(set(columns) - found)}"


def test_ambiguous_keywords_need_context_but_bare_keywords_still_match():
    # The bare word IS the column: a match.
    assert H._keyword_matches_column("name", "name")
    assert H._keyword_matches_column("ward", "ward")
    # Ordinary non-person vocabulary next to it: not a match.
    assert not H._keyword_matches_column("product_name", "name")
    assert not H._keyword_matches_column("file_name", "name")
    assert not H._keyword_matches_column("sales_district", "district")
    assert not H._keyword_matches_column("contract_value", "tract")
    assert not H._keyword_matches_column("blockchain_hash", "block")


# ---------------------------------------------------------------------------
# DEFECT 2 -- discovery.py reported spend / mortgage / package as demographics
# ---------------------------------------------------------------------------


def _discovery_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "average_spend": rng.random(n) * 500,
            "mortgage_balance": rng.random(n) * 1e5,
            "package_count": rng.integers(0, 40, n),
            "sussex_county": rng.choice(["a", "b", "c"], n),
            "alter": rng.integers(18, 80, n),
            "age": rng.integers(18, 80, n),
            "gender": rng.choice(["M", "F"], n),
            "race": rng.choice(["White", "Black", "Asian"], n),
            "zipcode": rng.choice(["10001", "10002", "10003"], n),
        }
    )


def _by_column(df=None):
    return {c.column: c for c in D.detect_protected_attributes(df or _discovery_frame())}


@pytest.mark.parametrize("column", ["average_spend", "mortgage_balance", "package_count"])
def test_an_average_a_mortgage_and_a_package_are_not_protected_attributes(column):
    """Measured before: each reported as a protected DEMOGRAPHIC attribute at
    0.55 confidence, above the 0.3 default threshold."""
    assert column not in _by_column()


def test_the_real_german_age_column_is_detected():
    """Measured before: 'alter' was not detected at all, while three
    innocuous columns were reported in its place."""
    found = _by_column()
    assert "alter" in found
    assert found["alter"].category == "demographic"
    assert found["alter"].confidence == pytest.approx(0.75)


@pytest.mark.parametrize(
    "column,confidence,category",
    [
        ("age", 0.75, "demographic"),
        ("gender", 1.0, "demographic"),
        ("race", 1.0, "demographic"),
    ],
)
def test_genuine_protected_columns_keep_their_old_confidence(column, confidence, category):
    """CONTROL: the confidences measured BEFORE the fix, unchanged."""
    found = _by_column()
    assert column in found
    assert found[column].confidence == pytest.approx(confidence)
    assert found[column].category == category


def test_a_compound_geographic_column_is_still_detected():
    """CONTROL: whole-token matching must not lose 'zipcode', which is in the
    catalogue as its own keyword."""
    assert "zipcode" in _by_column()


def test_sussex_county_is_geographic_and_never_matched_through_sex():
    found = _by_column()
    assert found["sussex_county"].category == "geographic"
    assert "'sex'" not in found["sussex_county"].reason


@pytest.mark.parametrize("column", ["average_spend", "mortgage_balance", "package_count"])
def test_a_numeric_average_is_not_even_admitted_as_an_age_column(column):
    """Pins the age-column GATE on its own. The name-keyword signal and the
    numeric-age signal each contributed to the 0.55 that was measured, so a
    test at the default threshold can pass while one of the two is still
    broken. At a 0.05 threshold the age gate alone is enough to report the
    column, so this fires if that gate goes back to substring matching."""
    found = {
        c.column: c for c in D.detect_protected_attributes(_discovery_frame(), min_confidence=0.05)
    }
    assert column not in found, found.get(column)


def test_age_token_detection_is_whole_token():
    assert D._AGE_TOKENS & set(name_token_list("age_group"))
    assert D._AGE_TOKENS & set(name_token_list("alter"))
    assert not D._AGE_TOKENS & set(name_token_list("average_spend"))
    assert not D._AGE_TOKENS & set(name_token_list("mortgage_balance"))


# ---------------------------------------------------------------------------
# DEFECT 3 -- accented names shattered into single letters
# ---------------------------------------------------------------------------

_PROXY_MODULES = [P, C]


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_weight_in_grams_does_not_directly_encode_gender(mod):
    """Measured before: ('peso_g', 'género') -> ProxyType.DIRECT, because
    both names produced a 'g' token."""
    assert mod._determine_proxy_type("peso_g", "género") is not mod.ProxyType.DIRECT


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_the_real_accented_direct_encoding_is_found(mod):
    """Measured before: ('genero_cliente', 'género') -> UNCLASSIFIED."""
    assert mod._determine_proxy_type("genero_cliente", "género") is mod.ProxyType.DIRECT


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
@pytest.mark.parametrize(
    "feature,attr",
    [("AVERAGE_SPEND", "AGE"), ("SUSSEX_COUNTY", "SEX"), ("package_count", "age")],
)
def test_the_ascii_collisions_stay_refused(mod, feature, attr):
    """CONTROL: the collision the docstring already promised cannot happen."""
    assert mod._determine_proxy_type(feature, attr) is not mod.ProxyType.DIRECT


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
@pytest.mark.parametrize(
    "feature,attr",
    [("gender_code", "gender"), ("customerAge", "AGE"), ("RACE_ETHNICITY", "race")],
)
def test_genuine_direct_encodings_are_still_direct(mod, feature, attr):
    """CONTROL: a Unicode-aware tokeniser must not lose the plain cases."""
    assert mod._determine_proxy_type(feature, attr) is mod.ProxyType.DIRECT


# ---------------------------------------------------------------------------
# DEFECT 4 -- renaming 'gender' to 'sex' changed the verdict
# ---------------------------------------------------------------------------


def _proxy_frame(n=600, seed=4, label="gender"):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    flipped = rng.random(n) < 0.25
    encoded = np.where(flipped, 1 - g, g).astype(float) + rng.normal(0, 0.05, n)
    return pd.DataFrame({label: pd.Series(g.astype(str), dtype=object), "name_score": encoded})


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_sex_and_gender_get_the_same_verdict_on_the_same_data(mod):
    """Measured before, same column and same 0.515 correlation:
    gender -> critical/direct/['Women', 'Non-binary individuals'];
    sex    -> high/unclassified/[]. 'sex' is the UCI Adult and COMPAS
    spelling."""
    out = {}
    for label in ("gender", "sex"):
        (result,) = mod.identify_proxy_variables(
            _proxy_frame(label=label), [label], min_sample_size=50
        )
        out[label] = (result.risk_level, result.proxy_type, tuple(result.affected_groups))
    assert out["sex"] == out["gender"]
    assert out["gender"][2] == ("Women", "Non-binary individuals")


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_ethnicity_resolves_to_race(mod):
    (result,) = mod.identify_proxy_variables(
        _proxy_frame(label="ethnicity"), ["ethnicity"], min_sample_size=50
    )
    assert result.affected_groups
    assert result.evidence["attribute_catalogue_key"] == "race"


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_an_unmapped_attribute_is_a_could_not_check_not_a_clean_result(mod):
    """The empty affected_groups list must never be printed as a fact when
    the attribute was simply never looked up."""
    (result,) = mod.identify_proxy_variables(
        _proxy_frame(label="origin_airport"), ["origin_airport"], min_sample_size=50
    )
    assert result.evidence["attribute_catalogue_key"] is None
    assert result.affected_groups == []
    assert any(r.startswith("COULD NOT CHECK") for r in result.recommendations), (
        "an unmapped attribute must SAY it could not be checked"
    )


@pytest.mark.parametrize(
    "attr,expected",
    [
        ("gender", "gender"),
        ("sex", "gender"),
        ("género", "gender"),
        ("ethnicity", "race"),
        ("ethnic_origin", "race"),
        ("disability_status", "disability"),
        ("alter", "age"),
        ("nationality", "national_origin"),
        ("country_of_origin", "national_origin"),
        # OVER-CORRECTION CONTROL: the key test must not be token OVERLAP.
        # 'national_origin' must not match every column containing 'origin'.
        ("origin", None),
        ("origin_airport", None),
        ("origin_country_code", None),
        ("average_spend", None),
        ("", None),
    ],
)
def test_attribute_resolution_is_whole_key_not_token_overlap(attr, expected):
    assert resolve_attribute_key(attr, P.KNOWN_PROXY_PATTERNS) == expected


# ---------------------------------------------------------------------------
# DEFECT 5 / 6 -- silent truncation, and "y" matched as a substring
# ---------------------------------------------------------------------------

_FAKE_OUTCOMES = [
    "salary",
    "yearly_bonus",
    "city",
    "county",
    "payment_type",
    "birthday",
    "company_type",
    "employment_years",
    "study_years",
    "display_name",
]


def _disparity_frame(n_noise, n=600, seed=7):
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    approved = np.where(sex == 1, rng.random(n) < 0.85, rng.random(n) < 0.15).astype(int)
    data = {"sex": sex}
    for i in range(n_noise):
        data[f"yield_{i:02d}"] = rng.random(n)
    data["approved"] = approved  # the REAL outcome, LAST in column order
    return pd.DataFrame(data)


@pytest.mark.parametrize("column", _FAKE_OUTCOMES)
def test_an_ordinary_column_is_not_a_model_outcome(column):
    """Measured before: all ten were auto-detected as outcomes, because "y"
    was matched as a SUBSTRING."""
    df = pd.DataFrame({c: [0.0, 1.0] for c in _FAKE_OUTCOMES + ["sex"]})
    assert column not in S._detect_outcome_columns(df, ["sex"])


@pytest.mark.parametrize("column", ["y", "target", "approved", "risk_score", "label"])
def test_a_real_outcome_column_is_still_detected(column):
    """CONTROL: including the bare "y", which is the whole point of keeping
    that keyword."""
    df = pd.DataFrame({column: [0.0, 1.0], "sex": [0, 1]})
    assert column in S._detect_outcome_columns(df, ["sex"])


def test_the_real_disparity_survives_thirty_five_decoy_columns():
    """Measured before: 0 disparities reported, while passing
    outcome_columns=['approved'] explicitly found the same gap at p=5.9e-60."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = S.analyze_statistical_disparities(_disparity_frame(35), ["sex"])
    found = {r.feature: r for r in results}
    assert "approved" in found, "the real 85%-vs-15% approval gap was never tested"
    assert found["approved"].pvalue < 1e-30
    assert found["approved"].disparity_type is S.DisparityType.OUTCOME


def test_untested_columns_are_disclosed_not_silently_dropped():
    """CONTROL for the truncation cap: it STAYS (an unbounded family size
    would hide real findings through Benjamini-Hochberg), but it must say
    what it skipped, so 'tested and clean' is distinguishable from
    'not tested'."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        S.analyze_statistical_disparities(_disparity_frame(35), ["sex"])
    messages = [str(w.message) for w in caught]
    assert any("NOT TESTED" in m for m in messages), messages
    # The cap itself is still enforced.
    detected = S._detect_outcome_columns(_disparity_frame(35), ["sex"])
    assert S._MAX_AUTO_OUTCOME_COLUMNS == 10
    assert S._MAX_AUTO_FEATURE_COLUMNS == 20
    assert len(detected) >= 1


def test_the_detectors_return_every_candidate_so_the_caller_can_disclose():
    """A helper that truncates in silence cannot be disclosed by anyone."""
    df = _disparity_frame(35)
    assert len(S._detect_feature_columns(df, ["sex"], [])) > 20


# ---------------------------------------------------------------------------
# DEFECT 7 -- protected_binning tokenised ALL-CAPS into single letters
# ---------------------------------------------------------------------------


# A dotted European date column that pandas CANNOT read with its default
# month-first guess. The first row is ambiguous (05.06), so the format is
# inferred as month-first, and every later row has a day above 12 and is
# coerced to NaT: measured, the default parse resolves 3.2% of these rows and
# the day-first parse resolves 100%. A fixture whose first row is already
# unambiguous would never exercise the retry at all.
_DOTTED_DOB = ["05.06.1970"] + [
    f"{d}.{m:02d}.19{60 + (d % 30):02d}" for d in range(13, 28) for m in (1, 2)
]


def _age_frame(seed=3):
    n = len(_DOTTED_DOB)
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 80, n)
    return pd.DataFrame(
        {
            "age": ages,
            "AGE": ages,
            "BIRTH_DATE": _DOTTED_DOB,
            "customerAge": rng.integers(18, 80, n),
            "customer_id": np.arange(n),
        }
    )


def test_identical_age_data_is_banded_identically_under_any_casing():
    """Measured before: 'age' got the canonical ADEA-aligned bands and 'AGE'
    got quantile bands like '(17.999, 33.8]', in the SAME frame."""
    df = _age_frame()
    out = B.prepare_protected_attributes(df, ["age", "AGE"])
    assert list(out.frame["age"]) == list(out.frame["AGE"])
    assert set(out.frame["AGE"].unique()) <= set(B._AGE_BAND_LABELS)


@pytest.mark.parametrize(
    "name,tokens",
    [
        ("AGE", ["age"]),
        ("DOB", ["dob"]),
        ("BIRTH_DATE", ["birth", "date"]),
        ("customerAge", ["customer", "age"]),
    ],
)
def test_upper_case_names_are_words_not_letters(name, tokens):
    assert B._tokens(name) == tokens


@pytest.mark.parametrize("name", ["DOB", "BIRTH_DATE", "dob", "date_of_birth"])
def test_a_date_of_birth_is_recognised_in_any_casing(name):
    assert B._is_dob(name)


def test_the_dotted_fixture_really_needs_the_day_first_retry():
    """Guard on the guard: if pandas ever reads this column month-first, the
    test below stops testing anything and must be given a harder column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = pd.to_datetime(pd.Series(_DOTTED_DOB), errors="coerce", utc=True)
        first = pd.to_datetime(pd.Series(_DOTTED_DOB), errors="coerce", utc=True, dayfirst=True)
    assert default.notna().mean() < 0.5
    assert first.notna().mean() == 1.0


def test_a_european_dotted_date_of_birth_becomes_age_bands():
    """Measured before: excluded with '98% distinct values, effectively an
    identifier' -- a confident explanation of the wrong thing, for a column
    an age was perfectly derivable from."""
    df = _age_frame()
    out = B.prepare_protected_attributes(df, ["BIRTH_DATE"])
    assert out.usable == ["BIRTH_DATE"]
    assert set(out.frame["BIRTH_DATE"].unique()) <= set(B._AGE_BAND_LABELS)


def test_customer_age_is_binned_not_excluded_as_an_identifier():
    """OVER-CORRECTION CONTROL. With correct tokenisation 'customerAge'
    yields BOTH 'customer' (an identifier word) and 'age', so the identifier
    escape hatch had to widen or an age column would still be thrown away."""
    assert not B._is_identifier_name("customerAge")
    out = B.prepare_protected_attributes(_age_frame(), ["customerAge"])
    assert "customerAge" in out.usable
    assert [c for c, _ in out.excluded] == []
    assert set(out.frame["customerAge"].unique()) <= set(B._AGE_BAND_LABELS)


@pytest.mark.parametrize(
    "name", ["customer_id", "user_uuid", "applicant_name", "email_address", "record_id"]
)
def test_real_identifiers_are_still_excluded(name):
    """OVER-CORRECTION CONTROL in the other direction: widening the escape
    must not turn every identifier into a fairness group."""
    assert B._is_identifier_name(name)


def test_a_us_format_date_still_parses_month_first():
    """OVER-CORRECTION CONTROL for the day-first retry: parsing everything
    day-first would be the same defect pointing the other way. This column's
    first row is ambiguous (01/02) so the format is inferred month-first, and
    the rest are month-first only: measured, the default parse resolves 100%
    and a forced day-first parse resolves 33%."""
    us = pd.Series(["01/02/1990", "12/25/1990", "11/30/1985"] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pd.to_datetime(us, errors="coerce", utc=True, dayfirst=True).notna().mean() < 0.5
    ages = B._derive_age(us)
    assert ages is not None
    assert ages.notna().all(), "the better-of-two rule flipped a US-format column"


# ---------------------------------------------------------------------------
# DEFECT 8 -- mutual information answered 0.0 for a failed computation
# ---------------------------------------------------------------------------


def test_mutual_information_refuses_rather_than_reporting_independence():
    """Measured before: two PERFECTLY DEPENDENT string series returned 0.0,
    the value that MEANS statistically independent."""
    x = np.array(["a", "b", "c", "d"] * 50, dtype=object)
    y = np.array(["A", "B", "C", "D"] * 50, dtype=object)
    with pytest.raises(Exception):
        C._compute_mutual_information(x, y)


def test_the_two_mutual_information_twins_behave_the_same_way():
    x = np.array(["a", "b", "c", "d"] * 50, dtype=object)
    y = np.array(["A", "B", "C", "D"] * 50, dtype=object)
    with pytest.raises(Exception):
        P._compute_mutual_information(x, y)


def test_mutual_information_still_measures_real_dependence():
    """CONTROL: a genuine 0 and a genuine positive must both still work."""
    rng = np.random.default_rng(0)
    a = rng.random(500)
    assert C._compute_mutual_information(a, a * 3 + 1) > 0.5
    independent = C._compute_mutual_information(a, rng.random(500))
    assert 0.0 <= independent < 0.5


# ---------------------------------------------------------------------------
# DEFECT 9 -- one blanket except returned a partial dict with no marker
# ---------------------------------------------------------------------------


def _measure_args():
    # FIXTURE CORRECTED (BGL-S2b, 2026-09-17), not the assertions below.
    # It was `np.arange(200.0)`: 200 distinct values over 200 rows, so the
    # Cramer's V contingency table was 200x2 with every cell 0 or 1, Bergsma's
    # bias correction consumed the whole table (min_corr == 0.0 exactly) and
    # the statistic was NOT IDENTIFIED. `_cramers_v_with_pvalue` used to answer
    # that with a clean 0.0 -- "no association" -- and these three tests, one
    # of which calls itself a CONTROL for the failure marker, therefore pinned
    # a fabricated measurement as evidence of a healthy run. The helper now
    # returns NaN there and names cramers_v in measures_not_computed, which is
    # correct and which made all three red.
    #
    # 20 levels over 200 rows keeps every property the three tests are about:
    # the feature is still numeric (so the eta branch still runs and can still
    # be made to fail), Pearson, Spearman and mutual information are unchanged
    # in kind, and Cramer's V is now genuinely identifiable (min_corr ~ 0.995),
    # so "a clean run carries no failure marker" is once again a statement
    # about a run in which every measure could actually be computed.
    feature = np.repeat(np.arange(20.0), 10)
    attr = np.tile([0.0, 1.0], 100)
    return feature, attr, pd.Series(feature), pd.Series(attr), True, False, True


def test_a_failed_measure_is_named_and_the_others_survive():
    """Measured before: an exception inside the eta branch silently dropped
    BOTH eta and mutual information, with nothing to say why."""
    with mock.patch.object(C, "_correlation_ratio_with_pvalue", side_effect=RuntimeError):
        results = C._compute_all_correlations(*_measure_args())
    assert results["measures_not_computed"] == ["correlation_ratio"]
    assert "mutual_information" in results
    assert "pearson" in results


def test_a_failed_mutual_information_is_omitted_never_zeroed():
    with mock.patch.object(C, "_compute_mutual_information", side_effect=RuntimeError):
        results = C._compute_all_correlations(*_measure_args())
    assert "mutual_information" not in results
    assert results["measures_not_computed"] == ["mutual_information"]


def test_a_clean_run_carries_no_failure_marker():
    """CONTROL: the marker must mean something."""
    results = C._compute_all_correlations(*_measure_args())
    assert "measures_not_computed" not in results
    assert {"pearson", "spearman", "cramers_v", "mutual_information"} <= set(results)


# ---------------------------------------------------------------------------
# DEFECT A -- untestable comparisons inflated the correction family size
# ---------------------------------------------------------------------------


def _constant_column_frame(n_const, n=1200, seed=2):
    rng = np.random.default_rng(seed)
    gender = rng.integers(0, 2, n)
    approved = np.where(gender == 1, rng.random(n) < 0.56, rng.random(n) < 0.46).astype(int)
    data = {"gender": gender, "approved": approved}
    for i in range(n_const):
        data[f"const_{i:02d}"] = np.zeros(n)
    return pd.DataFrame(data)


def _run(n_const):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = BiasDetector(
            _constant_column_frame(n_const), ["gender"], outcome_column="approved"
        ).analyze_disparities()
    return results, [str(w.message) for w in caught]


def test_constant_columns_do_not_erase_a_real_disparity():
    """Measured before, same data and same gender approval gap both times:
    0 constant columns -> 1 disparity at p=0.00465; 18 -> 0 disparities."""
    clean, _ = _run(0)
    padded, messages = _run(18)
    assert [r.feature for r in clean] == ["approved"]
    assert [r.feature for r in padded] == ["approved"]
    assert clean[0].pvalue == pytest.approx(padded[0].pvalue)
    assert any("could not be tested" in m for m in messages), messages


def test_a_non_finite_p_value_is_not_testable_not_not_significant():
    assert S._interpret_significance(float("nan")) is S.SignificanceLevel.NOT_TESTABLE
    assert S._interpret_significance(None) is S.SignificanceLevel.NOT_TESTABLE
    # CONTROL: the real grades are unchanged.
    assert S._interpret_significance(0.0001) is S.SignificanceLevel.HIGHLY_SIGNIFICANT
    assert S._interpret_significance(0.005) is S.SignificanceLevel.SIGNIFICANT
    assert S._interpret_significance(0.03) is S.SignificanceLevel.MARGINALLY_SIGNIFICANT
    assert S._interpret_significance(0.4) is S.SignificanceLevel.NOT_SIGNIFICANT


def _result(pvalue):
    return S.StatisticalDisparityResult(
        feature="f",
        protected_attribute="gender",
        disparity_type=S.DisparityType.OUTCOME,
        test_name="t",
        test_statistic=1.0,
        pvalue=pvalue,
        significance=S._interpret_significance(pvalue),
        effect_size=0.5,
        effect_size_type="Cohen's d",
        effect_interpretation=S.EffectSizeInterpretation.MEDIUM,
        confidence_interval=(0.0, 1.0),
        group_statistics={},
        privileged_group="a",
        disadvantaged_group="b",
        disparity_magnitude=0.1,
        sample_sizes={},
        recommendations=[],
    )


def test_the_correction_family_counts_only_testable_comparisons():
    """m must be 1, not 19, when 18 of the 19 could not be tested."""
    results = [_result(0.004)] + [_result(float("nan")) for _ in range(18)]
    corrected = S._apply_correction(results, "fdr_bh", 0.05)
    assert corrected[0].significance is S.SignificanceLevel.SIGNIFICANT
    assert all(r.significance is S.SignificanceLevel.NOT_TESTABLE for r in corrected[1:])


def test_the_correction_still_corrects_for_real_comparisons():
    """OVER-CORRECTION CONTROL: filtering NaNs must not disable the
    correction. With 19 genuine comparisons, p=0.004 is NOT rejected by BH at
    alpha=0.05 (0.004 > (1/19)*0.05 = 0.00263)."""
    results = [_result(0.004)] + [_result(0.9) for _ in range(18)]
    corrected = S._apply_correction(results, "fdr_bh", 0.05)
    assert corrected[0].significance is S.SignificanceLevel.NOT_SIGNIFICANT
    bonf = S._apply_correction(
        [_result(0.004)] + [_result(0.9) for _ in range(18)], "bonferroni", 0.05
    )
    assert bonf[0].significance is S.SignificanceLevel.NOT_SIGNIFICANT


def test_untestable_results_are_never_returned_as_findings():
    results = [_result(float("nan"))]
    S._apply_correction(results, "fdr_bh", 0.05)
    assert results[0].significance not in (
        S.SignificanceLevel.SIGNIFICANT,
        S.SignificanceLevel.HIGHLY_SIGNIFICANT,
    )
    assert results[0].significance in S._NON_FINDING_SIGNIFICANCE


# ---------------------------------------------------------------------------
# DEFECT B -- max(0.0, nan) collapsed a measured correlation to a clean zero
# ---------------------------------------------------------------------------


def _race_proxy_frame(spoil, n=900, seed=5):
    rng = np.random.default_rng(seed)
    race = rng.choice(["A", "B", "C"], n)
    band = np.where(
        race == "A",
        rng.normal(10, 0.3, n),
        np.where(race == "B", rng.normal(50, 0.3, n), rng.normal(90, 0.3, n)),
    )
    if spoil:
        band = band.copy()
        band[0] = np.inf
    return pd.DataFrame({"race": pd.Series(race, dtype=object), "income_band": band})


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_one_infinite_value_does_not_delete_a_critical_proxy(mod):
    """Measured before: clean -> 1 finding at r=0.9993, p=3.5e-283; with a
    single inf -> 0 findings, because max(0.0, nan) is 0.0 and the ANOVA's
    NaN p-value was then forced to a confident 1.0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = mod.identify_proxy_variables(_race_proxy_frame(False), ["race"], min_sample_size=50)
        spoiled = mod.identify_proxy_variables(
            _race_proxy_frame(True), ["race"], min_sample_size=50
        )
    assert [r.feature for r in clean] == ["income_band"]
    assert [r.feature for r in spoiled] == ["income_band"], (
        "one non-finite cell erased a critical proxy"
    )
    assert spoiled[0].risk_level is mod.ProxyRiskLevel.CRITICAL


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_a_genuinely_measured_zero_is_still_reported_as_zero(mod):
    """OVER-CORRECTION CONTROL: a real eta of 0.0 must not become NaN.

    THE FIXTURE CHANGED ON 2026-09-25, and the control's subject did not. It used
    a CONSTANT continuous variable, ``Series([5.0] * 100)``, described as "0.0 is a
    real answer when the continuous variable has no variance". That example was
    wrong: eta is ss_between / ss_total, so with no variance it is 0/0, undefined,
    and the accompanying ``pval == 1.0`` asserted a p-value for an ANOVA that
    cannot run (scipy's f_oneway returns nan for all-identical input, exactly as
    pearsonr does for a constant array). The library's own two-group path already
    returned nan for the same input, so the two arities disagreed.

    A genuinely measured zero is a different situation and it is the one worth
    protecting: the continuous variable HAS variance, and the group means happen
    to coincide, so ss_total > 0, ss_between == 0, and eta is a real, measured
    0.0. That is the fixture now. Group "a" alternates 1 and 3, group "b"
    alternates 0 and 4; both have mean 2.0 with genuine within-group spread.
    """
    values = pd.Series([1.0, 3.0] * 50 + [0.0, 4.0] * 50)
    groups = pd.Series(["a"] * 100 + ["b"] * 100)
    eta, pval = mod._correlation_ratio_with_pvalue(values, groups)
    assert eta == 0.0, "a measured zero was swallowed by the undefined-eta refusal"
    assert pval == 1.0


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_a_constant_variable_has_no_eta_to_report(mod):
    """The other half of the pair above, and the reason its fixture changed. Both
    modules must answer the same way: there are two copies of this function."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta, pval = mod._correlation_ratio_with_pvalue(
            pd.Series([5.0] * 100), pd.Series(["a", "b"] * 50)
        )
    assert math.isnan(eta), "a 0/0 eta was published as a measured 0.0"
    assert math.isnan(pval), (
        "a p-value of 1.0 claims 'definitely not significant' about an ANOVA that "
        "cannot run on constant input"
    )


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_an_uncomputable_eta_is_nan_and_never_a_confident_one_point_zero(mod):
    """Too few rows: (nan, nan), as the docstring promises."""
    eta, pval = mod._correlation_ratio_with_pvalue(pd.Series([1.0, 2.0]), pd.Series(["a", "b"]))
    assert math.isnan(eta) and math.isnan(pval)


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_an_anova_that_cannot_be_computed_is_nan_not_a_confident_one(mod):
    """Measured before: the ANOVA's NaN p-value was rewritten to 1.0, a claim
    of "definitely not significant" about a test that never ran, and the
    caller's significance gate then deleted the finding. Fifteen singleton
    groups have no within-group variance, so f_oneway returns NaN.

    EXPECTATION CORRECTED (BGL-S2b, 2026-09-17), SUBJECT UNCHANGED. This test
    also asserted ``eta == pytest.approx(1.0)`` and its docstring called that
    "perfectly measurable". It is not measurable: with one observation per
    level ``ss_between`` equals ``ss_total`` identically, so eta is exactly 1.0
    for ANY fifteen numbers at all. That is the fabricated-measurement shape
    this campaign exists to remove, and pinning it here made the defect a
    requirement. ``feature_engineering.correlation`` refused the partition from
    2026-09-16; ``bias_detection.proxy``, the copy the proxy DETECTOR calls,
    kept returning the 1.0 artefact until BGL-final d02 on 2026-09-17, when the
    same guard was applied to it. BOTH now return nan. The permissive
    disposition below is kept so this test is not what breaks if a third copy
    ever appears; the tight assertion that both copies must REFUSE lives in
    tests/test_bgl_final_d02.py. Neither module may return a CONFIDENT value
    strictly between 0 and 1, which is what a real eta on this partition would
    have to look like.

    What this test is NAMED for -- a p-value that could not be computed must be
    NaN and never a confident 1.0 -- is asserted unchanged, for both modules.
    """
    values = pd.Series(np.arange(1.0, 16.0))
    groups = pd.Series([f"g{i}" for i in range(15)])
    eta, pval = mod._correlation_ratio_with_pvalue(values, groups)
    assert math.isnan(pval), f"a p-value of {pval} was asserted for a test that did not run"
    assert not (0.0 < eta < 1.0), (
        f"eta={eta} reads as a measured association on a partition where every "
        f"observation is its own group"
    )
    assert math.isnan(eta) or eta == pytest.approx(1.0), (
        f"eta={eta} is neither the refusal nor the known 1.0 arithmetic artefact"
    )


@pytest.mark.parametrize("mod", _PROXY_MODULES, ids=["proxy", "correlation"])
def test_the_eta_helper_drops_non_finite_values_before_summing(mod):
    """CONTROL: the surviving rows must give the SAME answer as if the inf
    had never been in the column."""
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 300) + np.tile([0.0, 5.0], 150)
    groups = pd.Series(np.tile(["a", "b"], 150))
    clean_eta, _ = mod._correlation_ratio_with_pvalue(pd.Series(values), groups)
    spoiled = values.copy()
    spoiled[0] = np.inf
    spoiled_eta, _ = mod._correlation_ratio_with_pvalue(pd.Series(spoiled), groups)
    assert math.isfinite(spoiled_eta)
    assert abs(spoiled_eta - clean_eta) < 0.05


def test_python_max_really_does_swallow_nan():
    """The mechanism, pinned so nobody 'simplifies' np.maximum back to max."""
    assert max(0.0, float("nan")) == 0.0
    assert math.isnan(float(np.maximum(0.0, float("nan"))))
    for mod in _PROXY_MODULES:
        src = inspect.getsource(mod._correlation_ratio_with_pvalue)
        assert "np.maximum(0.0, ss_between)" in src
        assert "max(0.0, ss_between)" not in src.replace("np.maximum(0.0, ss_between)", "")


def test_the_shared_module_has_no_import_cycle():
    """It is imported by six modules across three sub-packages."""
    assert _names.name_tokens("age") == {"age"}
