"""
Sixth-iteration audit, wave 4 lane F: preprocessing pins.

Every finding here is the same defect class, a label nobody had evidence for
being handed out as though it were measured:

    S-11  `_determine_proxy_type(feature, protected_attr)`, duplicated in
          bias_detection/proxy.py and feature_engineering/correlation.py,
          documented `protected_attr` and read it nowhere. It classified a
          RELATIONSHIP from one side of that relationship. Measured before
          the fix: for feature "credit_score" the answer was "historical"
          for race, gender, age, disability, an empty string and the
          nonsense attribute "zzz_not_an_attribute" alike, and every feature
          matching no keyword at all ("commute_minutes", correlation 0.952
          with race) fell through to `return ProxyType.DIRECT  # Default`,
          the STRONGEST label in the enum ("Direct correlation with
          protected attribute"), on zero evidence about the attribute.
          Fixed by consulting KNOWN_PROXY_PATTERNS, which is keyed BY
          protected attribute, and by adding a third state, UNCLASSIFIED.

    S-11b (sibling, found while in the file) `_check_known_patterns` and
          `_get_affected_groups` matched an attribute with
          `attr_pattern in attr_lower or attr_lower in attr_pattern`, and
          "" is a substring of every key. Measured before the fix: a blank
          protected attribute promoted a 0.31 correlation from MEDIUM to
          CRITICAL and attached race's affected groups
          ("Black/African American", ...) to it.

    S-11c (sibling, found while in the file) correlation.py's
          `_analyze_proxy_relationship` took `attr_encoded` as a fourth
          positional argument and never read it. Measured before the fix:
          the real encoding, a length-3 array of zeros, an all-NaN array,
          None and the string "not an array" all returned the identical
          correlation 0.9686088460294854. Disposed of by REMOVAL: the body
          must encode the pairwise-aligned rows itself, so no caller could
          ever pass anything meaningful.

    S-15  `prepare_protected_attributes(df, protected, jurisdiction=None)`
          read `jurisdiction` nowhere: the word occurred exactly once in the
          function source, in the signature. Measured before the fix:
          None, "us-federal", "eu", "br" and "TOTAL NONSENSE JURISDICTION"
          returned byte-identical usable / excluded / notes. Disposed of by
          REFUSAL, not by inventing a legal ruleset: the binning here is a
          statistical operation that runs the same everywhere, and which
          attributes a jurisdiction protects is answered by
          vfairness.legal.admissibility.classify_columns(), which also needs
          a use case.

Each block carries a refusal pin AND an over-correction control asserting the
MEASURED values the working path still returns, so reinstating a defect turns
a pin red without the control passing by vacuity.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection import proxy as proxy_mod
from vfairness.preprocessing.bias_detection.proxy import _name_tokens as _proxy_name_tokens
from vfairness.preprocessing.feature_engineering import correlation as corr_mod
from vfairness.preprocessing.feature_engineering.correlation import (
    _name_tokens as _corr_name_tokens,
)
from vfairness.preprocessing.protected_binning import (
    _AGE_BAND_LABELS,
    prepare_protected_attributes,
)

# The two copies of the same function. Every S-11 pin runs against BOTH, so a
# fix that lands in one module and not the other fails here.
_MODULES = [
    pytest.param(proxy_mod, id="bias_detection.proxy"),
    pytest.param(corr_mod, id="feature_engineering.correlation"),
]


# --------------------------------------------------------------------------
# S-11: _determine_proxy_type ignored the protected attribute
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES)
def test_proxy_type_reads_the_protected_attribute(mod):
    """REFUSAL PIN: the same feature must not get the same label for every
    attribute. credit_score is a documented proxy for income, not for gender.
    """
    assert mod._determine_proxy_type("credit_score", "income") is mod.ProxyType.HISTORICAL
    assert mod._determine_proxy_type("credit_score", "gender") is mod.ProxyType.UNCLASSIFIED

    # The parameter is read at all: a feature classified against six
    # different attributes must not collapse to one answer.
    attrs = ["income", "gender", "age", "disability", "religion", "not_an_attribute"]
    answers = {a: mod._determine_proxy_type("credit_score", a) for a in attrs}
    assert len(set(answers.values())) > 1, answers


@pytest.mark.parametrize("mod", _MODULES)
def test_unclassifiable_feature_is_not_labelled_direct(mod):
    """REFUSAL PIN: no keyword and no attribute-keyed evidence must NOT
    produce DIRECT, the strongest label in the enum. Three states.
    """
    for feature in ("commute_minutes", "shoe_size", "widget_count"):
        got = mod._determine_proxy_type(feature, "race")
        assert got is mod.ProxyType.UNCLASSIFIED, (feature, got)
        assert got is not mod.ProxyType.DIRECT


@pytest.mark.parametrize("mod", _MODULES)
def test_measured_proxy_types_still_returned(mod):
    """OVER-CORRECTION CONTROL: the classifications the module CAN justify
    are unchanged. Exact enum members, never "one of the plausible ones".
    """
    assert mod._determine_proxy_type("zip_code", "race") is mod.ProxyType.HISTORICAL
    assert mod._determine_proxy_type("neighborhood", "race") is mod.ProxyType.HISTORICAL
    assert mod._determine_proxy_type("credit_score", "income") is mod.ProxyType.HISTORICAL
    assert mod._determine_proxy_type("surname", "race") is mod.ProxyType.DIRECT
    assert mod._determine_proxy_type("first_name", "gender") is mod.ProxyType.DIRECT
    # The feature names the attribute itself.
    assert mod._determine_proxy_type("gender_code", "gender") is mod.ProxyType.DIRECT
    # ... by whole token, never by substring: "age" must not match "average".
    assert mod._determine_proxy_type("average_spend", "age") is mod.ProxyType.UNCLASSIFIED


@pytest.mark.parametrize("mod", _MODULES)
def test_unclassified_is_a_real_enum_member_every_consumer_can_carry(mod):
    """The new state round-trips through the enum and through to_dict(), so
    a consumer reading `proxy_type` sees "unclassified", never "direct".
    """
    assert mod.ProxyType("unclassified") is mod.ProxyType.UNCLASSIFIED
    result = mod.ProxyVariableResult(
        feature="commute_minutes",
        protected_attribute="race",
        correlation=0.95,
        correlation_type="Correlation ratio (eta)",
        risk_level=mod.ProxyRiskLevel.CRITICAL,
        proxy_type=mod._determine_proxy_type("commute_minutes", "race"),
        mutual_information=None,
        cramers_v=None,
        pvalue=None,
        sample_size=400,
        affected_groups=[],
        recommendations=[],
    )
    assert result.to_dict()["proxy_type"] == "unclassified"


def _proxy_frame():
    """400 rows. zip_code and commute_minutes are BOTH near-perfect proxies
    for race; only zip_code is a documented pattern for that attribute.
    """
    rng = np.random.default_rng(4)
    n = 400
    race = rng.choice(["A", "B"], size=n)
    zipc = np.where(
        race == "A",
        rng.choice(["10001", "10002"], size=n),
        rng.choice(["90210", "90211"], size=n),
    )
    commute = np.where(race == "A", rng.normal(20, 3, n), rng.normal(45, 3, n))
    return pd.DataFrame({"race": race, "zip_code": zipc, "commute_minutes": commute})


def _proxy_frame_200():
    """The 200-row frame the pre-removal S-11c measurement was taken on."""
    rng = np.random.default_rng(4)
    n = 200
    race = rng.choice(["A", "B"], size=n)
    commute = np.where(race == "A", rng.normal(20, 3, n), rng.normal(45, 3, n))
    return pd.DataFrame({"race": race, "commute_minutes": commute})


@pytest.mark.parametrize("mod", _MODULES)
def test_end_to_end_unclassified_does_not_drop_the_finding(mod):
    """REFUSAL PIN + CONTROL on the public entry point.

    The measured numbers survive: a 0.97 correlation is still reported at
    CRITICAL risk. Only the KIND of the relationship becomes honest.
    """
    results = {r.feature: r for r in mod.identify_proxy_variables(_proxy_frame(), ["race"])}
    assert set(results) == {"zip_code", "commute_minutes"}

    historical = results["zip_code"]
    assert historical.proxy_type is mod.ProxyType.HISTORICAL
    assert historical.correlation == pytest.approx(0.9975, abs=0.02)
    assert historical.risk_level is mod.ProxyRiskLevel.CRITICAL

    unclassified = results["commute_minutes"]
    assert unclassified.proxy_type is mod.ProxyType.UNCLASSIFIED
    # The measurement is untouched: still found, still critical.
    assert unclassified.correlation == pytest.approx(0.9699, abs=0.02)
    assert unclassified.risk_level is mod.ProxyRiskLevel.CRITICAL
    assert unclassified.sample_size == 400


# --------------------------------------------------------------------------
# S-11b: "" matched every attribute group
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES)
def test_blank_attribute_yields_no_pattern_evidence(mod):
    """REFUSAL PIN: a blank protected attribute is could-not-check, not a
    match against every key in KNOWN_PROXY_PATTERNS.
    """
    for blank in ("", "   "):
        assert mod._check_known_patterns("credit_score", blank) is None
        assert mod._check_known_patterns("zipcode", blank) is None
        assert mod._get_affected_groups(blank) == []
        # 0.31 is a MEDIUM correlation. It used to be promoted to CRITICAL.
        assert mod._determine_risk_level(0.31, "credit_score", blank, True) is (
            mod.ProxyRiskLevel.MEDIUM
        )


@pytest.mark.parametrize("mod", _MODULES)
def test_named_attribute_still_promotes_and_still_names_groups(mod):
    """OVER-CORRECTION CONTROL: the real pattern lookup is untouched."""
    assert mod._check_known_patterns("credit_score", "income") == "high"
    assert mod._check_known_patterns("school", "income") == "medium"
    assert mod._check_known_patterns("shoe_size", "income") is None
    assert mod._get_affected_groups("race") == [
        "Black/African American",
        "Hispanic/Latino",
        "Asian",
        "Other minorities",
    ]
    # Same 0.31 correlation, this time with an attribute named: still CRITICAL.
    assert mod._determine_risk_level(0.31, "credit_score", "income", True) is (
        mod.ProxyRiskLevel.CRITICAL
    )


# --------------------------------------------------------------------------
# S-11c: correlation._analyze_proxy_relationship(attr_encoded=...)
# --------------------------------------------------------------------------


def test_analyze_proxy_relationship_no_longer_advertises_an_unread_argument():
    """REFUSAL PIN: `attr_encoded` was a 4th positional parameter that the
    body deliberately never read (it must encode the pairwise-aligned rows
    instead). Measured before removal: a wrong-length array, an all-NaN
    array, None and even the string "not an array" all produced the identical
    correlation 0.9686088460294854. Removed, so it cannot be passed at all.
    """
    signature = inspect.signature(corr_mod._analyze_proxy_relationship)
    assert "attr_encoded" not in signature.parameters

    with pytest.raises(TypeError):
        corr_mod._analyze_proxy_relationship(
            _proxy_frame(),
            "commute_minutes",
            "race",
            np.zeros(3),  # the removed argument
            correlation_threshold=0.2,
            include_mutual_information=True,
            include_known_patterns=True,
            min_sample_size=10,
            significance_level=0.05,
        )


def test_analyze_proxy_relationship_returns_the_same_measured_correlation():
    """OVER-CORRECTION CONTROL: removing the parameter changed no number.
    0.9686088460294854 is the value measured with the argument still in
    place, on this exact frame.
    """
    result = corr_mod._analyze_proxy_relationship(
        _proxy_frame_200(),
        "commute_minutes",
        "race",
        correlation_threshold=0.2,
        include_mutual_information=True,
        include_known_patterns=True,
        min_sample_size=10,
        significance_level=0.05,
    )
    assert result is not None
    assert result.correlation == pytest.approx(0.9686088460294854, abs=1e-12)
    assert result.sample_size == 200


# --------------------------------------------------------------------------
# S-15: prepare_protected_attributes(jurisdiction=...)
# --------------------------------------------------------------------------


def _binning_frame():
    rng = np.random.default_rng(11)
    n = 120
    return pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "zip_code": [f"{z:05d}" for z in rng.integers(10000, 99999, n)],
            "email": [f"p{i}@example.com" for i in range(n)],
            "gender": rng.choice(["F", "M"], n),
        }
    )


@pytest.mark.parametrize(
    "jurisdiction",
    ["us-federal", "eu", "br", "CH", "", "TOTAL NONSENSE JURISDICTION"],
)
def test_jurisdiction_is_refused_not_ignored(jurisdiction):
    """REFUSAL PIN: every non-default value is refused loudly, and the
    message names what IS supported and where the legal question belongs.
    """
    with pytest.raises(ValueError) as excinfo:
        prepare_protected_attributes(_binning_frame(), ["age", "gender"], jurisdiction)
    message = str(excinfo.value)
    assert "jurisdiction" in message
    assert repr(jurisdiction) in message
    assert "classify_columns" in message
    assert "jurisdiction=None" in message


def test_jurisdiction_none_still_performs_the_measured_binning():
    """OVER-CORRECTION CONTROL: the default path is byte-for-byte the
    behaviour that existed before the refusal was added. Measured values.
    """
    prep = prepare_protected_attributes(_binning_frame(), ["age", "zip_code", "email", "gender"])

    assert prep.usable == ["age", "zip_code", "gender"]
    assert [col for col, _ in prep.excluded] == ["email"]
    assert "personal identifier" in prep.excluded[0][1]

    # age -> the canonical bands, not quantiles.
    assert set(prep.frame["age"]) <= set(_AGE_BAND_LABELS)
    assert set(prep.frame["age"]) == {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"}

    # zip -> 3-digit prefix roll-up, never a numeric quantile band. 12 real
    # prefixes are kept and the long tail is folded into "other".
    zips = set(prep.frame["zip_code"])
    assert len(zips) == 13
    assert "other" in zips
    prefixes = zips - {"other"}
    assert len(prefixes) == 12
    assert all(z.startswith("ZIP ") and z.endswith("xx") for z in prefixes), prefixes

    # gender is low-cardinality and passes through untouched.
    assert set(prep.frame["gender"]) == {"F", "M"}

    labels = [label for label, _ in prep.notes]
    assert labels == ["Binning", "Binning"]


def test_jurisdiction_none_is_explicitly_accepted():
    """The keyword still exists and None is the supported value."""
    prep = prepare_protected_attributes(_binning_frame(), ["gender"], jurisdiction=None)
    assert prep.usable == ["gender"]


class TestNameTokensFoldsCaseBeforeMatching:
    """The first version of the S-11 fix INTRODUCED the defect it was removing.

    ``_name_tokens`` used ``[A-Za-z][a-z]*``, which splits an ALL-CAPS name into
    single LETTERS. Almost any two upper-case names then shared one, the token
    sets intersected, and ``_determine_proxy_type`` returned DIRECT, the
    strongest label in the enum, for unrelated columns. Measured before the
    repair: ``("AVERAGE_SPEND", "AGE") -> direct`` and
    ``("SUSSEX_COUNTY", "SEX") -> direct``, which are the exact two pairs the
    helper's own docstring promises cannot match.
    """

    @pytest.mark.parametrize(
        "impl",
        [_proxy_name_tokens, _corr_name_tokens],
        ids=["proxy", "correlation"],
    )
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("AVERAGE_SPEND", {"average", "spend"}),
            ("SUSSEX_COUNTY", {"county", "sussex"}),
            ("AGE", {"age"}),
            ("customerAge", {"age", "customer"}),
            ("HTTPServer", {"http", "server"}),
            ("zip_code", {"code", "zip"}),
        ],
    )
    def test_a_name_tokenises_to_its_words_in_any_casing(self, impl, name, expected):
        assert impl(name) == expected

    @pytest.mark.parametrize(
        "feature,attr",
        [("AVERAGE_SPEND", "AGE"), ("SUSSEX_COUNTY", "SEX"), ("average", "age")],
    )
    def test_sharing_only_letters_is_not_a_direct_proxy(self, feature, attr):
        """The docstring's own two examples, plus the lower-case case that
        always worked, so the pin covers both castings of the same claim."""
        assert proxy_mod._determine_proxy_type(feature, attr) is not proxy_mod.ProxyType.DIRECT

    @pytest.mark.parametrize(
        "feature,attr", [("customerAge", "age"), ("age_group", "age"), ("AGE_BRACKET", "AGE")]
    )
    def test_control_a_real_shared_word_is_still_direct(self, feature, attr):
        """Over-correction control: folding case must not stop a genuine
        whole-word match, including the ALL-CAPS spelling of one."""
        assert proxy_mod._determine_proxy_type(feature, attr) is proxy_mod.ProxyType.DIRECT

    def test_control_the_two_copies_cannot_drift(self):
        """The helper is duplicated in two modules. They were byte-identical
        before this fix and must stay in agreement, or one caller classifies
        differently from the other."""
        for name in ("AVERAGE_SPEND", "customerAge", "HTTPServer", "SUSSEX_COUNTY", "AGE_2", "a"):
            assert _proxy_name_tokens(name) == _corr_name_tokens(name), name
