"""
Automatic fairness discovery utilities for vfairness.

This module provides automatic detection and discovery features for fairness analysis:
    - Protected attribute detection
    - Proxy feature identification
    - Intersectional group auto-discovery
    - Comprehensive bias scanning

These utilities help users identify potential fairness issues without needing
to manually specify all protected attributes upfront.

Example:
    >>> from vfairness import detect_protected_attributes, scan_fairness_violations
    >>>
    >>> # Auto-detect potential protected attributes
    >>> candidates = detect_protected_attributes(df)
    >>> print(f"Found {len(candidates)} potential protected attributes")
    >>>
    >>> # Scan for fairness violations across all categorical columns
    >>> violations = scan_fairness_violations(df, y_pred)
    >>> for v in violations:
    ...     print(f"{v['attribute']}: disparity = {v['disparity']:.1%}")
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from vfairness._names import name_token_list, tokens_contain

from ..._triage import is_measured
from ...exceptions import ConfigurationError, ProtectedAttributeError


class ProxyScanIncompleteWarning(RuntimeWarning):
    """A proxy/association scan could not assess part of its input.

    Raised instead of letting an unassessable column vanish. An empty proxy
    list must not be readable as "no proxies exist" when it can also mean "we
    could not look": those are different answers and only one of them is
    reassuring.
    """


@dataclass
class ProtectedAttributeCandidate:
    """
    Container for a potential protected attribute.

    Attributes:
        column: Column name
        confidence: Confidence score (0-1) that this is a protected attribute
        reason: Why this column was identified as a potential protected attribute
        n_unique: Number of unique values
        sample_values: Sample of values in the column
        category: Detected category (demographic, geographic, etc.)
    """

    column: str
    confidence: float
    reason: str
    n_unique: int
    sample_values: List[Any]
    category: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "confidence": self.confidence,
            "reason": self.reason,
            "n_unique": self.n_unique,
            "sample_values": self.sample_values,
            "category": self.category,
        }


@dataclass
class FairnessViolation:
    """
    Container for a detected fairness violation.

    Attributes:
        attribute: Column name of the protected attribute
        metric: Fairness metric that was violated
        value: The metric value
        threshold: The threshold that was exceeded
        severity: Severity level ('low', 'medium', 'high', 'critical')
        privileged_group: Most advantaged group
        disadvantaged_group: Most disadvantaged group
        disparity: Disparity between groups
    """

    attribute: str
    metric: str
    value: float
    threshold: float
    severity: str
    privileged_group: str
    disadvantaged_group: str
    disparity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attribute": self.attribute,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "privileged_group": self.privileged_group,
            "disadvantaged_group": self.disadvantaged_group,
            "disparity": self.disparity,
        }


# PROTECTED ATTRIBUTE CATALOG
# Enhanced keyword dictionary based on comprehensive fairness catalog
# covering EU Charter Art.21, US Civil Rights, GDPR Art.9, and domain research

PROTECTED_KEYWORDS = {
    # IDENTITY & DEMOGRAPHICS
    "demographic": [
        # Gender / Sex (EU Charter Art.21, US Title VII)
        "gender",
        "sex",
        "male",
        "female",
        "man",
        "woman",
        "men",
        "women",
        "gender_identity",
        "transgender",
        "trans",
        "nonbinary",
        "non_binary",
        "cisgender",
        "genderqueer",
        "agender",
        "bigender",
        # Age (EU Employment Equality Directive, US ADEA)
        "age",
        "age_group",
        "age_range",
        "age_band",
        "age_bin",
        "birth_year",
        "birth_date",
        "dob",
        "date_of_birth",
        "birthdate",
        "yob",
        # Non-English spellings of age / date of birth. 'alter' (German for
        # age) was measured 2026-09-10 as not detected at all, while
        # average_spend was reported as a demographic attribute in its place.
        "alter",
        "edad",
        "idade",
        "leeftijd",
        "geburtsdatum",
        "geburtsjahr",
        "graduation_year",
        # Race / Ethnicity (EU Racial Equality Directive, US Title VII)
        "race",
        "racial",
        "ethnicity",
        "ethnic",
        "ethnic_origin",
        "ethnic_group",
        "skin_color",
        "skin_colour",
        "skin_tone",
        "color",
        "colour",
        "ancestry",
        "heritage",
        "background",
        "fitzpatrick",
        # Religion (EU Charter Art.21, US Title VII)
        "religion",
        "religious",
        "faith",
        "belief",
        "church",
        "mosque",
        "temple",
        "synagogue",
        "denomination",
        "spiritual",
        "atheist",
        "agnostic",
        # Sexual Orientation (EU Charter Art.21, US Title VII post-Bostock)
        "sexual_orientation",
        "orientation",
        "sexuality",
        "lgbtq",
        "lgbt",
        "gay",
        "lesbian",
        "bisexual",
        "homosexual",
        "heterosexual",
        "queer",
        # Indigenous Status (UNDRIP, national laws)
        "indigenous",
        "native",
        "tribal",
        "aboriginal",
        "first_nation",
        "first_nations",
        "tribe",
        "indian",
        "inuit",
        "metis",
        # Caste (India Constitution, emerging US protections)
        "caste",
        "dalit",
        "brahmin",
        "varna",
        "jati",
        # Birth status (EU Charter Art.21)
        "birth",
        "legitimacy",
        "birth_status",
        "born",
    ],
    # CITIZENSHIP & MIGRATION
    "citizenship_migration": [
        # Nationality (EU Charter Art.21)
        "nationality",
        "national",
        "national_origin",
        "citizen",
        "citizenship",
        "passport",
        "country_of_origin",
        "country_birth",
        "birth_country",
        # Migration status
        "migration",
        "migrant",
        "immigrant",
        "immigration",
        "immigration_status",
        "refugee",
        "asylum",
        "visa",
        "visa_status",
        "residency",
        "resident_status",
        "undocumented",
        "documented",
        "naturalized",
        "foreign_born",
        "native_born",
        # National minority (EU Charter Art.21)
        "minority",
        "national_minority",
        "minority_group",
    ],
    # FAMILY & CARE
    "family_care": [
        # Marital status
        "marital",
        "marital_status",
        "married",
        "single",
        "divorced",
        "widowed",
        "separated",
        "partnership",
        "civil_union",
        "domestic_partner",
        "spouse",
        "partner",
        "cohabiting",
        # Parental / Family status
        "parent",
        "parental",
        "parental_status",
        "family",
        "family_status",
        "children",
        "child",
        "kids",
        "dependents",
        "dependent",
        "mother",
        "father",
        "motherhood",
        "fatherhood",
        "childcare",
        "single_parent",
        # Pregnancy / Maternity (EU Pregnant Workers Directive, US PDA)
        "pregnancy",
        "pregnant",
        "maternity",
        "maternal",
        "paternity",
        "paternal",
        "parental_leave",
        "maternity_leave",
        "paternity_leave",
        # Caregiving
        "caregiver",
        "caregiving",
        "carer",
        "eldercare",
        "elder_care",
        "care_responsibilities",
        "unpaid_care",
    ],
    # HEALTH & ABILITY
    "health_ability": [
        # Disability (EU Charter Art.21, US ADA, UN CRPD)
        "disability",
        "disabled",
        "handicap",
        "handicapped",
        "impairment",
        "ability",
        "abilities",
        "accessible",
        "accessibility",
        "accommodation",
        "accommodations",
        "assistive",
        "ada",
        "special_needs",
        # Neurodivergence
        "neurodivergent",
        "neurodivergence",
        "neurotypical",
        "autism",
        "autistic",
        "adhd",
        "dyslexia",
        "dyslexic",
        "dyscalculia",
        "cognitive",
        # Mental health
        "mental_health",
        "psychiatric",
        "psychological",
        "depression",
        "anxiety",
        "ptsd",
        "bipolar",
        "schizophrenia",
        "mental_illness",
        "mental_disorder",
        # Health status (GDPR Art.9)
        "health",
        "health_status",
        "medical",
        "medical_condition",
        "chronic",
        "diagnosis",
        "diagnosed",
        "condition",
        "illness",
        "disease",
        "hiv",
        # Genetics (GDPR Art.9, US GINA)
        "genetic",
        "genetics",
        "genetic_data",
        "dna",
        "genome",
        "hereditary",
        # Biometrics (GDPR Art.9)
        "biometric",
        "biometrics",
        "fingerprint",
        "face_id",
        "facial",
        "iris",
        "retina",
        "voiceprint",
        "voice_id",
        # Body size
        "weight",
        "bmi",
        "body_mass",
        "body_size",
        "obesity",
        "obese",
        "height",
        "body_weight",
    ],
    # SOCIOECONOMIC
    "socioeconomic": [
        # Income / Wealth (EU Charter Art.21 property)
        "income",
        "salary",
        "wage",
        "wages",
        "earnings",
        "compensation",
        "wealth",
        "assets",
        "net_worth",
        "affluence",
        "poverty",
        "poor",
        "low_income",
        "high_income",
        "income_level",
        "income_band",
        "income_bracket",
        # Education
        "education",
        "education_level",
        "educational",
        "degree",
        "diploma",
        "college",
        "university",
        "school",
        "graduate",
        "undergraduate",
        "highschool",
        "high_school",
        "ged",
        "literacy",
        "literate",
        "illiterate",
        "first_gen",
        "first_generation",
        # Employment / Occupation
        "employment",
        "employed",
        "unemployed",
        "unemployment",
        "job",
        "jobs",
        "occupation",
        "occupational",
        "profession",
        "professional",
        "career",
        "work",
        "worker",
        "working",
        "labor",
        "labour",
        "workforce",
        "full_time",
        "part_time",
        "gig",
        "contractor",
        "freelance",
        "temp",
        "employment_gap",
        "employment_gaps",
        "work_history",
        # Housing
        "housing",
        "housing_status",
        "homeowner",
        "homeownership",
        "renter",
        "tenant",
        "homeless",
        "homelessness",
        "housing_insecurity",
        "shelter",
        "eviction",
        "foreclosure",
        # Social origin / class (EU Charter Art.21)
        "social_origin",
        "social_class",
        "class",
        "socioeconomic",
        "ses",
        "socioeconomic_status",
        "deprivation",
        "disadvantaged",
        "underserved",
        # Digital access
        "digital_literacy",
        "digital_access",
        "internet_access",
        "broadband",
        "tech_access",
        "digital_divide",
    ],
    # GEOGRAPHIC (often proxy for race/SES)
    "geographic": [
        "zip",
        "zipcode",
        "zip_code",
        "postal",
        "postal_code",
        "postcode",
        "region",
        "regional",
        "state",
        "province",
        "county",
        "country",
        "city",
        "urban",
        "rural",
        "suburban",
        "neighborhood",
        "neighbourhood",
        "district",
        "area",
        "zone",
        "locale",
        "location",
        "address",
        "geolocation",
        "geo",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng",
        "redline",
        "redlining",
        "census_tract",
        "block_group",
        "geohash",
        "commute",
        "commuting",
        "residence",
        "residential",
    ],
    # CULTURE & COMMUNICATION
    "culture_communication": [
        # Language (EU Charter Art.21)
        "language",
        "lang",
        "locale",
        "dialect",
        "accent",
        "speech",
        "native_language",
        "primary_language",
        "mother_tongue",
        "first_language",
        "second_language",
        "esl",
        "ell",
        "english_proficiency",
        "fluency",
        "multilingual",
        "bilingual",
        "monolingual",
        # Communication patterns
        "aave",
        "african_american_english",
        "vernacular",
        "pidgin",
        "creole",
    ],
    # BELIEF & CIVIC STATUS
    "belief_civic": [
        # Political opinion (EU Charter Art.21)
        "political",
        "politics",
        "political_opinion",
        "political_affiliation",
        "party",
        "partisan",
        "liberal",
        "conservative",
        "democrat",
        "republican",
        "voting",
        "voter",
        "vote",
        # Trade union (GDPR Art.9)
        "union",
        "trade_union",
        "union_member",
        "unionized",
        "labor_union",
        "collective_bargaining",
        # Veteran status
        "veteran",
        "military",
        "military_status",
        "armed_forces",
        "service_member",
        "army",
        "navy",
        "marines",
        "air_force",
        "coast_guard",
        "national_guard",
        "deployment",
        "combat",
        "discharge",
    ],
    # OTHER STATUS
    "other_status": [
        # Criminal / Justice involvement
        "criminal",
        "criminal_record",
        "conviction",
        "convicted",
        "arrest",
        "arrested",
        "incarceration",
        "incarcerated",
        "felony",
        "felon",
        "misdemeanor",
        "justice_involved",
        "formerly_incarcerated",
        "ex_offender",
        "background_check",
        "ban_the_box",
        # Sex life (GDPR Art.9)
        "sex_life",
        "sexual_behavior",
        "sexual_activity",
    ],
    # PROXY INDICATORS (high risk for indirect discrimination)
    "proxy_indicators": [
        # Name-based (proxy for ethnicity, religion, gender, caste)
        "name",
        "first_name",
        "last_name",
        "surname",
        "given_name",
        "family_name",
        "maiden_name",
        "patronym",
        "matronym",
        # Generic sensitive markers
        "protected",
        "sensitive",
        "demographic",
        "group",
        "subgroup",
        "cohort",
        "segment",
        "category",
        "classification",
    ],
}

# Flatten all keywords for quick lookup
ALL_PROTECTED_KEYWORDS = set()
for keywords in PROTECTED_KEYWORDS.values():
    ALL_PROTECTED_KEYWORDS.update(keywords)

# High-risk proxy features from catalog (should trigger warnings)
HIGH_RISK_PROXIES = {
    "race": [
        "zip",
        "zipcode",
        "zip_code",
        "postal",
        "postcode",
        "neighborhood",
        "surname",
        "last_name",
        "name",
        "school",
        "address",
    ],
    "gender": ["name", "first_name", "given_name", "voice", "pitch", "height"],
    "age": [
        "graduation_year",
        "years_experience",
        "experience_years",
        "tenure",
        "seniority",
        "career_length",
    ],
    "nationality": [
        "name",
        "surname",
        "accent",
        "language",
        "country_code",
        "phone_prefix",
        "passport_type",
    ],
    "socioeconomic": [
        "zip",
        "zipcode",
        "address",
        "school",
        "college",
        "device_type",
        "browser",
        "credit_score",
    ],
    "disability": [
        "speech_pattern",
        "response_time",
        "interaction_pace",
        "typing_speed",
        "error_rate",
    ],
    "religion": ["name", "surname", "holiday", "dietary", "diet_preference"],
}

# Binary value patterns indicating protected attributes
BINARY_PROTECTED_PATTERNS = [
    # Gender patterns
    {"m", "f"},
    {"male", "female"},
    {"man", "woman"},
    {"men", "women"},
    {"m", "w"},
    {"masculine", "feminine"},
    # Yes/No patterns (often disability, veteran, etc.)
    {"yes", "no"},
    {"y", "n"},
    {"true", "false"},
    {"1", "0"},
    {"t", "f"},
    {"si", "no"},
    {"oui", "non"},
    {"ja", "nein"},
    # Race patterns (problematic but detectable)
    {"white", "black"},
    {"caucasian", "african_american"},
    {"white", "non_white"},
    {"white", "minority"},
    # Citizenship patterns
    {"citizen", "non_citizen"},
    {"native", "foreign"},
    {"domestic", "international"},
    {"resident", "nonresident"},
    # Employment patterns
    {"employed", "unemployed"},
    {"full_time", "part_time"},
    # Marital patterns
    {"married", "single"},
    {"married", "unmarried"},
]

# Category-specific value patterns (multi-value)
CATEGORY_VALUE_PATTERNS = {
    "demographic": {
        # Race/ethnicity values
        "race_ethnicity": {
            "white",
            "black",
            "asian",
            "hispanic",
            "latino",
            "latina",
            "latinx",
            "african_american",
            "caucasian",
            "native_american",
            "pacific_islander",
            "multiracial",
            "mixed",
            "other",
            "aian",
            "nhpi",
            "aapi",
        },
        # Gender values
        "gender": {
            "male",
            "female",
            "non_binary",
            "nonbinary",
            "other",
            "prefer_not_to_say",
            "transgender",
            "cisgender",
            "genderqueer",
        },
        # Religion values
        "religion": {
            "christian",
            "catholic",
            "protestant",
            "jewish",
            "muslim",
            "hindu",
            "buddhist",
            "sikh",
            "atheist",
            "agnostic",
            "none",
            "other",
            "spiritual",
        },
    },
    "socioeconomic": {
        # Education levels
        "education": {
            "high_school",
            "hs",
            "ged",
            "some_college",
            "associates",
            "bachelors",
            "bachelor",
            "masters",
            "master",
            "doctorate",
            "phd",
            "md",
            "jd",
            "mba",
            "graduate",
            "postgraduate",
        },
        # Employment status
        "employment": {
            "employed",
            "unemployed",
            "self_employed",
            "retired",
            "student",
            "homemaker",
            "disabled",
            "part_time",
            "full_time",
        },
    },
    "health_ability": {
        # Disability disclosure
        "disability": {
            "yes",
            "no",
            "prefer_not_to_say",
            "not_disclosed",
            "physical",
            "mental",
            "cognitive",
            "sensory",
            "none",
        },
    },
    "family_care": {
        # Marital status values
        "marital": {
            "single",
            "married",
            "divorced",
            "widowed",
            "separated",
            "domestic_partner",
            "civil_union",
            "cohabiting",
            "engaged",
        },
    },
}


# Whole-token spellings of "this column is about a person's age". Measured
# 2026-09-10, the substring test this replaces
# (``any(kw in col_lower for kw in ["age", "birth", "dob", "yob"])``)
# reported average_spend, mortgage_balance and package_count as protected
# DEMOGRAPHIC attributes at 0.55 confidence, above the 0.3 default -- the
# identical AVERAGE_SPEND/AGE collision that bias_detection.proxy documents
# as fixed, alive in the higher-level user-facing detector. The same test
# missed the German 'alter' entirely, so the real age column was invisible
# while three innocuous ones were reported in its place.
_AGE_TOKENS = {
    "age",
    "aged",
    "ages",
    "birth",
    "birthdate",
    "birthday",
    "birthyear",
    "dob",
    "yob",
    # non-English spellings of the same attribute
    "alter",
    "edad",
    "idade",
    "leeftijd",
    "geburtsdatum",
    "geburtsjahr",
    "naissance",
    "nacimiento",
}


def detect_protected_attributes(
    df: pd.DataFrame,
    *,
    min_confidence: float = 0.3,
    exclude_columns: Optional[List[str]] = None,
    max_unique_values: int = 50,
    include_numeric: bool = False,
    include_age_columns: bool = True,
) -> List[ProtectedAttributeCandidate]:
    """
    Automatically detect potential protected attributes in a DataFrame.

    Uses enhanced heuristics based on comprehensive fairness catalog covering:
    - EU Charter Art.21 protected grounds
    - US Civil Rights legislation (Title VII, ADA, ADEA, etc.)
    - GDPR Art.9 special categories
    - Domain-specific fairness research

    Detection signals:
    - Column names matching 400+ protected attribute keywords across 9 categories
    - Data types (categorical, low-cardinality)
    - Value patterns (30+ binary patterns, category-specific value sets)
    - High-risk proxy indicators

    Args:
        df: DataFrame to analyze
        min_confidence: Minimum confidence score to include (0-1)
        exclude_columns: Columns to exclude from detection
        max_unique_values: Maximum unique values for a column to be considered
        include_numeric: Whether to include numeric columns with few unique values
        include_age_columns: Whether to include age-related numeric columns (default True)

    Returns:
        List of ProtectedAttributeCandidate objects, sorted by confidence

    Example:
        >>> candidates = detect_protected_attributes(df)
        >>> for c in candidates:
        ...     print(f"{c.column}: {c.confidence:.1%} confidence ({c.reason})")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: detect_protected_attributes. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    exclude_columns = exclude_columns or []
    candidates = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        # Get column info
        series = df[col]
        n_unique = series.nunique()
        dtype = series.dtype

        # Handle extension dtypes
        try:
            is_numeric = np.issubdtype(dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        # Special handling for age columns (numeric but protected).
        # WHOLE-TOKEN, never substring: "age" must not match "average",
        # "mortgage" or "package". See _AGE_TOKENS above for the measurement.
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        col_tokens = name_token_list(col)
        is_age_column = bool(set(col_tokens) & _AGE_TOKENS)

        # Skip if too many unique values (unless it's age)
        if n_unique > max_unique_values and not is_age_column:
            continue

        # Skip numeric unless explicitly included or it's an age column
        if is_numeric and not include_numeric:
            if not (include_age_columns and is_age_column):
                continue

        # Calculate confidence based on multiple factors
        confidence = 0.0
        reasons = []
        category = "unknown"
        proxy_risk = None

        # SIGNAL 1: Column name keyword matching (0.4-0.6 confidence).
        # Whole-token containment, never substring: the substring form made
        # 'age' match average_spend / mortgage_balance / package_count and
        # 'sex' match sussex_county. Multi-word keywords must appear
        # contiguously, so 'census_tract' does not match a frame that merely
        # has a 'census' column and a 'tract' column.
        for cat, keywords in PROTECTED_KEYWORDS.items():
            for keyword in keywords:
                if not tokens_contain(col_tokens, keyword):
                    continue
                # Higher confidence for exact matches or longer keywords
                if col_tokens == name_token_list(keyword):
                    confidence += 0.6
                    reasons.append(f"Exact column name match: '{keyword}'")
                elif len(keyword) >= 6:  # Longer keywords are more specific
                    confidence += 0.5
                    reasons.append(f"Column name contains '{keyword}'")
                else:
                    confidence += 0.4
                    reasons.append(f"Column name contains '{keyword}'")
                category = cat
                break
            if category != "unknown":
                break

        # SIGNAL 2: High-risk proxy detection (adds warning, +0.1 confidence)
        for protected_attr, proxy_keywords in HIGH_RISK_PROXIES.items():
            for proxy_kw in proxy_keywords:
                # Whole-token, same reason as SIGNAL 1: the substring form
                # flagged any column containing 'name' or 'zip' as a proxy.
                if tokens_contain(col_tokens, proxy_kw):
                    proxy_risk = protected_attr
                    if category == "unknown":
                        confidence += 0.1
                        category = "proxy_indicators"
                    reasons.append(f"High-risk proxy for '{protected_attr}'")
                    break
            if proxy_risk:
                break

        # SIGNAL 3: Value pattern matching (0.2-0.4 confidence)
        sample_values = series.dropna().unique()[:20].tolist()
        values_lower = {str(v).lower().strip() for v in sample_values}

        # Binary pattern detection (expanded patterns)
        if n_unique == 2:
            for pattern in BINARY_PROTECTED_PATTERNS:
                if values_lower == pattern or values_lower.issubset(pattern):
                    confidence += 0.35
                    reasons.append(f"Binary protected pattern: {sample_values}")
                    # Infer category from pattern
                    if pattern in [
                        {"m", "f"},
                        {"male", "female"},
                        {"man", "woman"},
                        {"men", "women"},
                        {"masculine", "feminine"},
                    ]:
                        if category == "unknown":
                            category = "demographic"
                    elif pattern in [
                        {"citizen", "non_citizen"},
                        {"native", "foreign"},
                        {"domestic", "international"},
                    ]:
                        if category == "unknown":
                            category = "citizenship_migration"
                    elif pattern in [{"employed", "unemployed"}, {"full_time", "part_time"}]:
                        if category == "unknown":
                            category = "socioeconomic"
                    elif pattern in [{"married", "single"}, {"married", "unmarried"}]:
                        if category == "unknown":
                            category = "family_care"
                    break

        # Category-specific value pattern matching (multi-value)
        for cat, pattern_groups in CATEGORY_VALUE_PATTERNS.items():
            for pattern_name, pattern_values in pattern_groups.items():
                # Check if column values overlap significantly with known patterns
                overlap = values_lower & pattern_values
                if len(overlap) >= 2 or (len(overlap) >= 1 and n_unique <= 5):
                    overlap_ratio = len(overlap) / max(len(values_lower), 1)
                    if overlap_ratio >= 0.3:  # At least 30% match
                        confidence += 0.3
                        reasons.append(f"Values match {pattern_name} pattern: {list(overlap)[:5]}")
                        if category == "unknown":
                            category = cat
                        break
            if category != "unknown" and len(reasons) > 1:
                break

        # SIGNAL 4: Cardinality-based scoring (0.1-0.2 confidence)
        if not is_numeric and 2 <= n_unique <= 10:
            confidence += 0.2
            reasons.append(f"Low-cardinality categorical ({n_unique} values)")
        elif not is_numeric and 10 < n_unique <= 25:
            confidence += 0.1
            reasons.append(f"Medium-cardinality categorical ({n_unique} values)")

        # SIGNAL 5: Data type scoring (0.05-0.1 confidence)
        if dtype == "object" or pd.api.types.is_categorical_dtype(dtype):
            confidence += 0.1
            reasons.append("Categorical data type")
        elif is_age_column and is_numeric:
            confidence += 0.15
            reasons.append("Numeric age-related column")
            if category == "unknown":
                category = "demographic"

        # SIGNAL 6: Special column name patterns (0.1-0.2 confidence)
        # Abbreviated/coded column names common in datasets
        abbrev_patterns = {
            "dem_": "demographic",
            "prot_": "proxy_indicators",
            "sens_": "proxy_indicators",
            "grp_": "proxy_indicators",
            "_grp": "proxy_indicators",
            "_cat": "proxy_indicators",
            "_cd": "proxy_indicators",  # code suffix
            "_flg": "proxy_indicators",  # flag suffix
            "_ind": "proxy_indicators",  # indicator suffix
        }
        for pat_key, pat_category in abbrev_patterns.items():
            if pat_key in col_lower:
                confidence += 0.15
                reasons.append("Column naming pattern suggests protected attribute")
                if category == "unknown":
                    category = pat_category
                break

        # FINALIZE
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)

        # Add if above threshold
        if confidence >= min_confidence:
            # Build comprehensive reason string
            reason_str = "; ".join(reasons) if reasons else "Low cardinality column"
            if proxy_risk:
                reason_str += f" [PROXY WARNING: may correlate with {proxy_risk}]"

            candidates.append(
                ProtectedAttributeCandidate(
                    column=col,
                    confidence=confidence,
                    reason=reason_str,
                    n_unique=n_unique,
                    sample_values=sample_values[:10],
                    category=category,
                )
            )

    # Sort by confidence (descending)
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def _chi2_stat(obs: np.ndarray) -> float:
    """Pearson chi-square of a contingency table (no scipy dependency)."""
    obs = obs.astype(float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    tot = obs.sum()
    if tot <= 0:
        return 0.0
    exp = row @ col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013) for two nominal series.

    The correct association measure for nominal x nominal. Pearson on
    arbitrary `Categorical.codes` (the previous behaviour) is meaningless
    for unordered categories -- a high-cardinality nominal such as a ZIP
    code that perfectly predicts race scored ~0 and the proxy was missed.

    Returns NaN, never 0.0, when the corrected statistic is NOT IDENTIFIED on
    the table it was handed. 0.0 is the strongest reassurance on this scale
    ("X does not track Y at all"), so publishing it for a table nobody could
    measure is a clean bill of health nobody earned. Three refusals, each
    decided on its own merits and each measured at the public entry first
    (2026-09-17):

    (1) AN EMPTY TABLE. Nothing was observed, so nothing was measured.

    (2) A DEGENERATE MARGIN, one level on a side, so min(r, k) - 1 is 0 and V
        is 0/0. This is the single-group case: on 50 rows whose protected
        attribute held ONE gender, ``identify_proxy_features`` returned ``[]``
        with no warning, a confident "no proxies" for a frame in which the
        proxy question cannot be asked at all. The same branch catches a
        constant feature column, which has the same 0/0.

    (3) THE BIAS CORRECTION CONSUMED THE TABLE, ``denom <= 0``, reached by any
        column with about one row per level. A unique-per-row ``customer_ref``
        that determines ``gender`` perfectly over 200 rows scored
        ``_cramers_v -> 0.0``, ``association_strength -> 0.0`` and
        ``vfairness.identify_proxy_features -> []`` with no warning anywhere.
        Bucket the high-cardinality side and measure again.

    A ``phi2corr`` clamped to 0.0 with ``denom > 0`` is NOT refused: Bergsma's
    estimator is genuinely zero for a weak association, and that is a
    measurement.
    """
    ct = pd.crosstab(a, b)
    n = int(ct.values.sum()) if ct.size else 0
    if ct.size == 0 or n <= 0:
        warnings.warn(
            f"_cramers_v: the contingency table is empty (shape {tuple(ct.shape)}, "
            f"{n} observation(s)), so no association was measured. Returning nan, "
            f"not 0.0, which reads as a measured independence.",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        warnings.warn(
            f"_cramers_v: Cramer's V is NOT DEFINED on a {ct.shape[0]}x{ct.shape[1]} "
            f"table over {n} row(s): one side holds a single level, so "
            f"min(r, k) - 1 is 0 and the statistic is 0/0. Returning nan, not 0.0, "
            f"which reads as a measured independence. A single-group protected "
            f"attribute, or a constant feature column, cannot be screened for proxy "
            f"risk at all.",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    if n <= 1:
        warnings.warn(
            f"_cramers_v: {n} observation(s) on a {ct.shape[0]}x{ct.shape[1]} table. "
            f"Bergsma's correction divides by n - 1, so nothing is defined here. "
            f"Returning nan, not 0.0, which reads as a measured independence.",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    phi2 = _chi2_stat(ct.values) / n
    r, k = ct.shape
    # np.maximum, NOT the builtin max: max(0.0, nan) is 0.0, because every
    # comparison against NaN is False and max keeps its first argument, so a
    # chi-square that did not compute would come back an exact zero. The same
    # idiom is banned by name in feature_engineering/correlation.py.
    phi2corr = float(np.maximum(0.0, phi2 - (k - 1) * (r - 1) / (n - 1)))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        warnings.warn(
            f"_cramers_v: the bias correction leaves no effective dimension "
            f"(denom={denom:.4g} <= 0) on a {r}x{k} table over {n} rows, which is "
            f"what happens when the levels are as numerous as the rows, so Cramer's "
            f"V COULD NOT BE MEASURED. Returning nan, not 0.0, which reads as a "
            f"measured independence. Bucket the high-cardinality side and measure "
            f"again.",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    v = float(np.sqrt(phi2corr / denom))
    if not np.isfinite(v):
        warnings.warn(
            f"_cramers_v: the corrected statistic came out non-finite on a {r}x{k} "
            f"table over {n} rows, so it was NOT measured. Returning nan, not 0.0, "
            f"which reads as a measured independence.",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    return v


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta in [0,1] for categorical x numeric association."""
    vals = pd.to_numeric(values, errors="coerce")
    m = ~vals.isna()
    vals = vals[m]
    cats = categories[m]
    if len(vals) < 10:
        # eta = 0.0 is "no association whatever", the exact reading a proxy
        # variable must NOT be given when nobody looked. This feeds proxy
        # detection, so a substituted zero hides the thing the function exists
        # to find.
        warnings.warn(
            f"_correlation_ratio: {len(vals)} usable row(s), below the minimum of 10, "
            f"so no association was measured. Returning nan, not 0.0, which reads as "
            f"'no association'.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    # eta is a RATIO OF VARIANCES BETWEEN GROUPS, so two degenerate groupings
    # force its value by arithmetic alone and say nothing whatever about the
    # data. Both were live at the public entry on 2026-09-17 and they fail in
    # OPPOSITE directions, which is why neither can be left to the caller.
    sizes = vals.groupby(cats.values).size()
    if len(sizes) < 2:
        # ONE GROUP. ss_between is 0 no matter what the values do, so eta is
        # exactly 0.0: "this feature does not track the groups at all",
        # published about a sample that holds a single group. Measured on 50
        # rows with one gender and a numeric salary:
        # `identify_proxy_features -> []`, no warning. A single-group sample
        # cannot answer the proxy question at all.
        warnings.warn(
            f"_correlation_ratio: the grouping holds {len(sizes)} level(s) across "
            f"{len(vals)} row(s), so eta is 0.0 by arithmetic and says nothing "
            f"about the data. Returning nan, not 0.0, which reads as a measured "
            f"independence. A single-group sample cannot be screened for proxy "
            f"risk.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    if int(sizes.max()) < 2:
        # EVERY LEVEL A SINGLETON. Each group mean IS its one observation, so
        # ss_between == ss_total and eta is exactly 1.0: a PERFECT PROXY
        # finding manufactured out of cardinality. Measured: a unique-per-row
        # `customer_ref` scored 1.0000000000000002, risk_level "high", against
        # a numeric `age`, with no warning. The twin
        # `_correlation_ratio_with_pvalue` in bias_detection/proxy.py already
        # refuses this by name; this copy did not.
        warnings.warn(
            f"_correlation_ratio: no level of the grouping holds more than one "
            f"observation ({len(sizes)} level(s) over {len(vals)} row(s)), so eta "
            f"is 1.0 by arithmetic and says nothing about the data. Returning nan, "
            f"not 1.0, which reads as a perfect association. Bucket the "
            f"high-cardinality side before reading this as a proxy finding.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    gmean = vals.mean()
    ss_total = float(((vals - gmean) ** 2).sum())
    if ss_total <= 0:
        # NOT a refusal, and deliberately so. ss_total == 0 means the numeric
        # side is CONSTANT, and no grouping can explain variation that does not
        # exist, so eta 0 is the correct answer rather than a fabricated one.
        # This exact question was reviewed on the sibling copy
        # (proxy.py:_correlation_ratio_with_pvalue) and the 0.0 was recorded as
        # legitimate there. `categories` is the grouping side by contract, so
        # the degenerate side here is always the explained variable, never the
        # protected attribute. Do not "fix" this to NaN: the two guards above
        # are the degeneracies that do need it.
        return 0.0
    ss_between = 0.0
    for _, grp in vals.groupby(cats.values):
        ss_between += len(grp) * (grp.mean() - gmean) ** 2
    return float(np.sqrt(max(0.0, ss_between) / ss_total))


def association_strength(a: pd.Series, b: pd.Series) -> float:
    """Type-aware association in [0,1] between two aligned series. ONE
    source of truth for "how strongly does X track Y" across the library:
      numeric x numeric   -> |Pearson|
      numeric x nominal    -> correlation ratio (eta)
      nominal x nominal    -> bias-corrected Cramer's V
    Pearson on pd.Categorical.codes (the legacy approach) is invalid for
    unordered categories and silently missed high-cardinality proxies.

    The return is NaN, never 0.0, whenever the chosen measure is not defined
    on the data it was handed: too little overlap, a table the Bergsma
    correction consumes, a single-level side, a constant numeric side, or a
    dtype that raises. Every one of those carries a warning naming the pair.
    Callers must test `np.isnan(...)` before comparing the result with a
    threshold: `nan >= 0.3` is False, which would turn a could-not-check back
    into a clean bill of health.
    """
    pair = pd.DataFrame({"_a": a, "_b": b}).dropna()
    if len(pair) < 10:
        # Too little overlap to measure. NaN, not 0.0: a returned zero is
        # indistinguishable from a genuine measurement of no association.
        warnings.warn(
            f"association_strength: only {len(pair)} aligned non-null rows "
            "(need 10); returning NaN, which is not a measurement of zero association",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")
    av, bv = pair["_a"], pair["_b"]

    def _num(s: pd.Series) -> bool:
        try:
            return bool(np.issubdtype(s.dtype, np.number))
        except (TypeError, AttributeError):
            return False

    an, bn = _num(av), _num(bv)
    try:
        if an and bn:
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(av.astype(float), bv.astype(float))[0, 1]
            if np.isnan(c):
                # Pearson is 0/0 when a side has zero variance. The sibling
                # branch inside `identify_proxy_features` already calls this
                # "Pearson undefined: zero variance" and refuses it; this one
                # returned 0.0, so the same constant column read as a measured
                # independence here and as could-not-check one layer up.
                warnings.warn(
                    f"association_strength: Pearson is NOT DEFINED between "
                    f"'{getattr(a, 'name', '?')}' and '{getattr(b, 'name', '?')}' "
                    f"over {len(pair)} aligned rows, because at least one side has "
                    f"zero variance. Returning NaN, not 0.0, which reads as a "
                    f"measured independence.",
                    ProxyScanIncompleteWarning,
                    stacklevel=2,
                )
                return float("nan")
            return float(abs(c))
        if (not an) and (not bn):
            return _cramers_v(av.astype("object"), bv.astype("object"))
        if an and not bn:
            return _correlation_ratio(bv.astype("object"), av)
        return _correlation_ratio(av.astype("object"), bv)
    except Exception as exc:  # noqa: BLE001 -- association is never fatal
        # Never fatal, but never silent either. Returning 0.0 here made a
        # PERFECT proxy report no association at all: a column of list-valued
        # objects (a JSON column) raises inside _cramers_v, and the same proxy
        # stored as plain strings scores 1.0. identify_proxy_features then
        # reported "no proxies found" on a dataset containing one, with no
        # warning anywhere. NaN is could-not-measure; the warning is what makes
        # it visible.
        warnings.warn(
            f"association_strength: could not measure association between "
            f"'{getattr(a, 'name', '?')}' and '{getattr(b, 'name', '?')}' "
            f"({type(exc).__name__}: {exc}); returning NaN, NOT zero association",
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )
        return float("nan")


class _ProxyScanResult(list):
    """A list of proxy features that can also say which columns it could not read.

    Same shape as `_ScanResult`, for the same reason. `identify_proxy_features`
    already collected `unassessable` and warned about it, but returned a bare
    list, so the only channel was the warning: under `python -W ignore` a scan
    that could not read a single column was indistinguishable from a scan that
    found no proxies, and `rank_fairness_issues` published
    `n_proxy_warnings: 0` as a measurement.
    """

    not_assessable: List[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.not_assessable = []


def identify_proxy_features(
    df: pd.DataFrame,
    protected_attr: str,
    *,
    correlation_threshold: float = 0.3,
    exclude_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Identify features that may be proxies for a protected attribute.

    A proxy feature is one that correlates highly with a protected attribute
    and could lead to indirect discrimination.

    Args:
        df: DataFrame containing features
        protected_attr: Name of the protected attribute column
        correlation_threshold: Minimum correlation to flag as proxy
        exclude_columns: Columns to exclude from analysis

    Returns:
        List of dicts with proxy information:
            - column: Feature name
            - correlation: Correlation with protected attribute
            - correlation_type: 'numeric' or 'categorical'
            - risk_level: 'low', 'medium', 'high'

    Example:
        >>> proxies = identify_proxy_features(df, 'gender')
        >>> for p in proxies:
        ...     print(f"{p['column']}: correlation = {p['correlation']:.2f}")
    """
    exclude_columns = exclude_columns or []
    exclude_columns.append(protected_attr)
    proxies = []
    # Columns the scan could not assess, with the reason. An empty `proxies`
    # list must not be readable as "no proxies exist" when it can also mean
    # "we could not look at that column": before this, five separate `continue`
    # statements dropped a column with no trace, and a PERFECT proxy stored as
    # list-valued objects (a JSON column) returned [] and zero warnings.
    unassessable: List[str] = []

    if protected_attr not in df.columns:
        raise ProtectedAttributeError(f"Protected attribute '{protected_attr}' not in DataFrame")

    protected_series = df[protected_attr]

    # Determine whether the protected attribute is numeric (drives the
    # type-aware association choice below).
    try:
        _is_numeric = np.issubdtype(protected_series.dtype, np.number)
    except (TypeError, AttributeError):
        _is_numeric = False

    for col in df.columns:
        if col in exclude_columns:
            continue

        try:
            feature_series = df[col]

            # Determine whether the feature is numeric (drives the
            # type-aware association choice below).
            try:
                _feat_numeric = np.issubdtype(feature_series.dtype, np.number)
            except (TypeError, AttributeError):
                _feat_numeric = False
            if not _feat_numeric:
                corr_type = "categorical"
            else:
                corr_type = "numeric"

            # Type-aware association (NaN-aligned on the raw series):
            #  numeric x numeric      -> |Pearson|        (unchanged)
            #  nominal x nominal      -> bias-corrected Cramer's V
            #  numeric x nominal      -> correlation ratio (eta)
            # All measures live in [0,1] so the existing risk thresholds
            # keep their meaning. This replaces Pearson-on-Categorical.codes
            # which was invalid for unordered categories and silently
            # missed high-cardinality proxies (e.g. ZIP -> race redlining).
            pair = pd.DataFrame({"_f": feature_series, "_p": protected_series}).dropna()
            if len(pair) < 10:
                unassessable.append(f"{col} (only {len(pair)} aligned non-null rows, need 10)")
                continue
            fser, pser = pair["_f"], pair["_p"]
            if _feat_numeric and _is_numeric:
                c = np.corrcoef(fser.astype(float), pser.astype(float))[0, 1]
                if np.isnan(c):
                    unassessable.append(f"{col} (Pearson undefined: zero variance)")
                    continue
                corr, abs_corr, corr_type = float(c), abs(float(c)), "numeric"
            elif (not _feat_numeric) and (not _is_numeric):
                abs_corr = _cramers_v(fser.astype("object"), pser.astype("object"))
                corr, corr_type = abs_corr, "categorical"
            elif _feat_numeric and (not _is_numeric):
                abs_corr = _correlation_ratio(pser.astype("object"), fser)
                corr, corr_type = abs_corr, "mixed"
            else:  # feature categorical, protected numeric
                abs_corr = _correlation_ratio(fser.astype("object"), pser)
                corr, corr_type = abs_corr, "mixed"

            if abs_corr is None or np.isnan(abs_corr):
                unassessable.append(f"{col} (association undefined on this data)")
                continue

            if abs_corr >= correlation_threshold:
                # Determine risk level
                if abs_corr >= 0.7:
                    risk = "high"
                elif abs_corr >= 0.5:
                    risk = "medium"
                else:
                    risk = "low"

                proxies.append(
                    {
                        "column": col,
                        "correlation": corr,
                        "abs_correlation": abs_corr,
                        "correlation_type": corr_type,
                        "risk_level": risk,
                    }
                )

        except Exception as exc:
            # Skip columns that can't be processed, but SAY SO. Silently
            # skipping is what let a perfect proxy read as "no proxies found".
            unassessable.append(f"{col} ({type(exc).__name__}: {exc})")
            continue

    if unassessable:
        warnings.warn(
            f"identify_proxy_features: {len(unassessable)} of "
            f"{len(df.columns) - len(exclude_columns)} candidate columns could NOT be "
            f"assessed against '{protected_attr}', so this result is incomplete. "
            "An unassessed column is not a column without proxy risk. "
            "Unassessed: " + "; ".join(unassessable),
            ProxyScanIncompleteWarning,
            stacklevel=2,
        )

    # Sort by absolute correlation (descending)
    proxies.sort(key=lambda x: x["abs_correlation"], reverse=True)
    # Still a list for every existing caller; the incompleteness now travels
    # with it instead of living only on the warning channel.
    scanned = _ProxyScanResult(proxies)
    scanned.not_assessable = [f"{protected_attr}/proxy scan: {u}" for u in unassessable]
    return scanned


def _classify_severity(disparity: float) -> str:
    """Classify disparity into severity level."""
    if disparity >= 0.20:
        return "critical"
    elif disparity >= 0.15:
        return "high"
    elif disparity >= 0.10:
        return "medium"
    else:
        return "low"


# Default metrics to evaluate when y_true is available vs not
_DEFAULT_METRICS_WITH_LABELS = [
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "predictive_parity_difference",
]

_DEFAULT_METRICS_NO_LABELS = [
    "demographic_parity_difference",
]


def _identify_violation_groups(
    metric_name: str,
    gm: Any,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify (privileged_group, disadvantaged_group) for a triggered violation
    using the RATE the metric is actually built from.

    Returns ``(None, None)`` when the pair could not be identified (fewer than
    two defined rates on the arm the metric is built from): a third state, not
    a group named "" and not a guess. The caller maps it to "unknown". The
    annotation was ``Tuple[str, str]`` until 2026-09-09, which mypy rejected
    for exactly this return (main CI red on discovery.py).

    Deriving the labels from the positive-prediction (selection) rate for EVERY
    metric inverts the attribution for error-based metrics: a group can have the
    highest selection rate yet the lowest TPR (equal opportunity) or highest FPR
    (equalized odds), so the group being harmed gets labelled privileged. The
    metric VALUE is correct regardless; only these labels, which a remediation
    team acts on, were wrong.

    Orientation follows the library's favourable-positive convention (the same
    one demographic parity already assumes: higher selection rate = privileged).
    A more favourably treated group has a HIGHER selection rate / TPR / FPR and
    a LOWER PPV (more of its positive predictions are unwarranted).

    References:
        Hardt et al. 2016 (equal opportunity = TPR gap); Chouldechova 2017
        (predictive parity = PPV gap).
    """
    from ._grouping import compute_max_difference

    def _priv_disadv(
        rates: Dict[str, float], higher_is_privileged: bool
    ) -> Tuple[Optional[str], Optional[str]]:
        defined = {g: v for g, v in rates.items() if not np.isnan(v)}
        if len(defined) < 2:
            # One spelling of "not identified" (2026-09-09): this used to
            # return ("", ""), a second, string-typed way of saying None.
            return None, None
        _, high, low = compute_max_difference(defined)
        return (high, low) if higher_is_privileged else (low, high)

    # Demographic parity is a selection-rate metric; it also covers the
    # no-labels case where only y_pred is available.
    if metric_name == "demographic_parity_difference" or y_true is None:
        return _priv_disadv(gm.compute_group_rate(y_pred == 1), True)

    if metric_name == "equal_opportunity_difference":
        tpr = gm.compute_group_rate((y_pred == 1) & (y_true == 1), (y_true == 1))
        return _priv_disadv(tpr, higher_is_privileged=True)

    if metric_name == "predictive_parity_difference":
        ppv = gm.compute_group_rate((y_pred == 1) & (y_true == 1), (y_pred == 1))
        return _priv_disadv(ppv, higher_is_privileged=False)

    if metric_name == "equalized_odds_difference":
        # Attribute to the driving arm: the larger of the TPR gap and FPR gap
        # is what equalized_odds_difference returns as its value, so the label
        # pair must come from that same arm. Higher = privileged for both arms.
        tpr = gm.compute_group_rate((y_pred == 1) & (y_true == 1), (y_true == 1))
        fpr = gm.compute_group_rate((y_pred == 1) & (y_true == 0), (y_true == 0))
        tpr_def = {g: v for g, v in tpr.items() if not np.isnan(v)}
        fpr_def = {g: v for g, v in fpr.items() if not np.isnan(v)}
        # Each arm needs two defined rates to have a gap at all. 0.0 said "no
        # gap on this arm", and the comparison below then picked a driving arm
        # out of two values neither of which was measured.
        tpr_gap = compute_max_difference(tpr_def)[0] if len(tpr_def) >= 2 else float("nan")
        fpr_gap = compute_max_difference(fpr_def)[0] if len(fpr_def) >= 2 else float("nan")
        if tpr_gap != tpr_gap and fpr_gap != fpr_gap:
            warnings.warn(
                "violation grouping: neither the TPR nor the FPR arm had two defined "
                "group rates, so no driving arm could be identified.",
                UserWarning,
                stacklevel=2,
            )
            return None, None
        # A measured arm always wins over an unmeasured one, in either order.
        if fpr_gap != fpr_gap:
            driving = tpr
        elif tpr_gap != tpr_gap:
            driving = fpr
        else:
            driving = tpr if tpr_gap >= fpr_gap else fpr
        return _priv_disadv(driving, higher_is_privileged=True)

    # Unknown metric: fall back to the selection rate rather than guess.
    return _priv_disadv(gm.compute_group_rate(y_pred == 1), True)


class _ScanResult(list):
    """A list of violations that can also say what it could not check.

    C-03. `scan_fairness_violations` returns a bare list, so there was nowhere to
    record the attributes and metrics that were skipped, and an empty list read
    as "checked, and clean". This stays a list for every existing caller and
    carries `not_assessable` for the ones that ask.
    """

    not_assessable: List[str]
    skipped_columns: List[str]
    zero_selection_alerts: List[Dict[str, Any]]
    measured_disparities: List[float]
    metrics_not_attempted: List[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.not_assessable = []
        # EVERY disparity this scan actually measured, violation or not. The list
        # of violations holds only the ones OVER threshold, so a maximum taken
        # over it is a maximum over the breaches, not over the measurements: see
        # the `max_disparity` comment in `rank_fairness_issues`, which published
        # 0.0 for a measured gap of 0.025 because nothing crossed the line.
        self.measured_disparities = []
        # The columns that were skipped ENTIRELY, as names. `rank_fairness_issues`
        # used to recover this by looking for a "/" in each `not_assessable`
        # string, which silently miscounts the moment a new attribute-level entry
        # is added (a partially-scanned column is not an unscanned one).
        self.skipped_columns = []
        # Groups the size gate removed that the model had also selected nobody
        # from. Structured, not only prose inside `not_assessable`, so a caller
        # can act on it without parsing a sentence.
        self.zero_selection_alerts = []
        # The REGISTERED metrics this scan did not attempt at all, by name. Empty
        # means every registered metric was evaluated on every scanned attribute;
        # see the block at the metric-menu selection in `scan_fairness_violations`
        # for the measured all-clear that an empty list used to cover.
        self.metrics_not_attempted = []


def scan_fairness_violations(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    *,
    protected_columns: Optional[List[str]] = None,
    auto_detect: bool = True,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.1,
    thresholds: Optional[Dict[str, float]] = None,
    min_group_size: int = 30,
) -> List[FairnessViolation]:
    """
    Scan for fairness violations across multiple metrics and protected attributes.

    Evaluates each protected attribute column against one or more fairness
    metrics and returns violations that exceed the specified thresholds.

    Args:
        df: DataFrame containing features
        y_pred: Predicted labels (binary 0/1)
        y_true: Optional true labels (required for equal_opportunity,
                equalized_odds, and predictive_parity metrics)
        protected_columns: Specific columns to check (if None, auto-detect)
        auto_detect: Whether to auto-detect protected attributes
        metrics: List of metric names to evaluate. Supported:
                 'demographic_parity_difference',
                 'equal_opportunity_difference',
                 'equalized_odds_difference',
                 'predictive_parity_difference'.
                 If None, defaults to all when y_true is provided,
                 or demographic_parity_difference only otherwise.
        threshold: Default fairness threshold for all metrics
        thresholds: Optional per-metric thresholds (overrides threshold).
                    E.g. {'demographic_parity_difference': 0.1,
                           'equal_opportunity_difference': 0.05}
        min_group_size: Minimum group size for analysis

    Returns:
        List of FairnessViolation objects, sorted by severity/disparity

    Example:
        >>> violations = scan_fairness_violations(
        ...     df, y_pred, y_true,
        ...     protected_columns=['sex', 'race'],
        ...     metrics=['demographic_parity_difference', 'equal_opportunity_difference'],
        ...     thresholds={'demographic_parity_difference': 0.1,
        ...                 'equal_opportunity_difference': 0.05}
        ... )
        >>> for v in violations:
        ...     print(f"{v.attribute} [{v.metric}]: {v.disparity:.3f} ({v.severity})")

    BGL-U00 (2026-09-17). This function HAS now been reached and run, on nine
    degenerate inputs (tests/test_bgl_final_u00.py), and it FABRICATED on three of
    them: a requested column absent from the frame, an auto-detection that found
    no candidate, and a metric that raised. Each returned violations=[] with
    not_assessable=[], skipped_columns=[] and no field a reader could tell it from
    a clean scan by, so rank_fairness_issues reported n_attributes_checked=1,
    n_not_assessable=0, max_disparity=0.0 and "No significant fairness violations
    detected. Continue monitoring." over a frame holding a real 0.60 gap. The
    other six (one group, zero rows, an all-NaN attribute, an undefined TPR, and
    both saturated-prediction frames) were already honest. Fixed at all three
    sites below; a healthy frame still reports its measured 0.60 gap, and a
    genuinely clean one still gets the all-clear.

    The machine-generated stamp below is derived from src/vfairness/_proof_status.py
    and is STALE until that ledger, its evidence file and the published tables are
    regenerated together. Read this paragraph, not the stamp, for what is known.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: auto_discovery. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    import warnings

    from ._grouping import GroupManager
    from .classification import (
        demographic_parity_difference as _dp_diff,
    )
    from .classification import (
        equal_opportunity_difference as _eo_diff,
    )
    from .classification import (
        equalized_odds_difference as _eod_diff,
    )
    from .classification import (
        predictive_parity_difference as _pp_diff,
    )

    # Map metric names to (function, requires_y_true)
    _METRIC_REGISTRY = {
        "demographic_parity_difference": (_dp_diff, False),
        "equal_opportunity_difference": (_eo_diff, True),
        "equalized_odds_difference": (_eod_diff, True),
        "predictive_parity_difference": (_pp_diff, True),
    }

    violations = []
    # Populated by the could-not-check paths below. Surfaced to the caller
    # through `rank_fairness_issues`, and as warnings here, because this function
    # returns a bare list with nowhere to put it.
    not_assessable: List[str] = []
    skipped_columns: List[str] = []
    zero_selection_alerts: List[Dict[str, Any]] = []
    # The registered metrics this scan never attempted, for the same reason
    # `not_assessable` exists: an empty violation list has to be readable as
    # "this many questions were asked" and not only as "no answer came back".
    metrics_not_attempted: List[str] = []
    # Every disparity that was MEASURED, whether or not it exceeded a threshold.
    # `violations` keeps only the ones over the line, so it cannot answer "what
    # was the largest gap we measured"; see `rank_fairness_issues`.
    measured_disparities: List[float] = []

    # Determine columns to check
    if protected_columns:
        columns_to_check = protected_columns
    elif auto_detect:
        candidates = detect_protected_attributes(df, min_confidence=0.3)
        columns_to_check = [c.column for c in candidates]
    else:
        raise ConfigurationError("Must provide protected_columns or set auto_detect=True")

    # BGL-C (2026-09-17). With `auto_detect=True` and a frame the detector finds
    # no candidate in, `columns_to_check` is empty, the loop below never runs and
    # this returned `violations=[]` with `not_assessable=[]` and no warning on
    # any channel: a scan that examined NOTHING, reported as clean. Detection is
    # a heuristic with a confidence floor, so an empty candidate list is a
    # could-not-check, not a finding of fairness.
    if not columns_to_check:
        not_assessable.append(
            "no attribute was scanned at all: "
            + (
                "auto-detection found no protected attribute candidate in the "
                f"{len(df.columns)} column(s) of this frame"
                if not protected_columns
                else "the requested protected_columns list is empty"
            )
        )
        warnings.warn(
            "scan_fairness_violations scanned NO attribute: "
            + (
                "auto-detection found no protected attribute candidate in this frame. "
                "Detection is a heuristic with a confidence floor; pass "
                "protected_columns explicitly."
                if not protected_columns
                else "the requested protected_columns list is empty."
            )
            + " An empty violation list here means nothing was checked, not that the "
            "model is fair.",
            UserWarning,
            stacklevel=2,
        )

    # Determine metrics to evaluate
    if metrics is None:
        metrics = _DEFAULT_METRICS_WITH_LABELS if y_true is not None else _DEFAULT_METRICS_NO_LABELS
        # THE METRIC SET NARROWED FROM FOUR TO ONE AND THE DISCLOSURE CHANNEL SAID
        # NOTHING.
        # BGL5 A-evaluation-3, 2026-09-29. Without y_true only demographic parity
        # can be computed, and this line silently dropped the three metrics that
        # need labels: no field, no count and no warning a reader could tell a
        # one-metric scan from a four-metric one by, so the only completeness
        # question the summary answered was about attributes. `rank_fairness_issues`
        # has no `metrics` parameter, so this default is its ONLY path, and with
        # nothing over threshold it published its all-clear. Measured before this
        # change,
        # 2000 rows in two groups whose selection rates are identical at 0.500
        # while the model is perfect for one group and fully inverted for the
        # other, i.e. an error-rate disparity of 1.0:
        #   y_true=None -> {'n_violations': 0, 'max_disparity': 0.0,
        #                   'n_disparities_measured': 1, 'n_not_assessable': 0},
        #                  not_assessable [], recommendations
        #                  ['No significant fairness violations detected.
        #                    Continue monitoring.'], warnings NONE
        #   the SAME data with the real y_true -> {'n_violations': 3,
        #                   'max_disparity': 1.0, 'n_disparities_measured': 4}
        # An all-clear was published for a run structurally incapable of seeing
        # the disparity that was there. Derived from the REGISTRY rather than from
        # the two default lists, so a fifth label-dependent metric added to the
        # registry is covered without touching this line.
        if y_true is None:
            # Its OWN channel, not `not_assessable`. That list answers "which
            # ATTRIBUTE or GROUP could not be measured", is counted into
            # `n_attributes_checked` and drives the three-case `max_disparity` in
            # `rank_fairness_issues`, where it would turn the demographic-parity
            # gap this run DID measure into a NaN and destroy a real number. The
            # incompleteness here is a different one: every attribute was scanned,
            # by a menu of one metric instead of four. Both are published, both
            # are counted, and neither is read as the other.
            metrics_not_attempted = [
                m for m, (_fn, needs_labels) in _METRIC_REGISTRY.items() if needs_labels
            ]
            if metrics_not_attempted:
                warnings.warn(
                    f"scan_fairness_violations ran WITHOUT labels: "
                    f"{', '.join(metrics_not_attempted)} "
                    f"were not attempted, so only {', '.join(metrics)} was evaluated. An "
                    f"empty violation list means no SELECTION-RATE gap was found, not that "
                    f"the model is fair: a model that is perfect for one group and inverted "
                    f"for another has identical selection rates. Pass y_true to measure the "
                    f"other metrics.",
                    UserWarning,
                    stacklevel=2,
                )

    # Validate metric names
    for m in metrics:
        if m not in _METRIC_REGISTRY:
            raise ConfigurationError(
                f"Unknown metric '{m}'. Supported: {list(_METRIC_REGISTRY.keys())}"
            )

    # Check that y_true is provided for metrics that need it
    needs_labels = [m for m in metrics if _METRIC_REGISTRY[m][1]]
    if needs_labels and y_true is None:
        raise ConfigurationError(
            f"y_true is required for metrics: {needs_labels}. "
            "Either provide y_true or remove these metrics."
        )

    y_pred = np.asarray(y_pred)
    if y_true is not None:
        y_true = np.asarray(y_true)

    for col in columns_to_check:
        if col not in df.columns:
            # BGL-C (2026-09-17). This `continue` was SILENT. Measured on 1000
            # rows with a real 0.60 selection-rate gap on `race` and
            # protected_columns=['sex'] (a column the frame does not have):
            # violations=[], not_assessable=[], skipped_columns=[], zero
            # warnings, and `rank_fairness_issues` reported
            # n_attributes_checked=1, n_not_assessable=0, max_disparity=0.0 and
            # "No significant fairness violations detected. Continue
            # monitoring." A misspelt or renamed column bought a clean bill of
            # health for an attribute nobody had looked at.
            not_assessable.append(
                f"{col}: column is not in the DataFrame "
                f"(available: {list(df.columns)[:10]}), so it was not scanned at all"
            )
            skipped_columns.append(col)
            warnings.warn(
                f"Attribute {col!r} was NOT scanned: there is no such column in the "
                "DataFrame. An empty violation list for this attribute means it was "
                "not checked, not that it is clean.",
                UserWarning,
                stacklevel=2,
            )
            continue

        sensitive_attr = df[col].values

        # C-03. An attribute with fewer than two valid groups is skipped, and
        # the skip was SILENT: the caller got an empty violation list, which is
        # the same answer as "checked, and clean". Measured 2026-09-07 with 280
        # rows in one group and 20 in another (below min_group_size=30), where the
        # small group was NEVER selected: violations=[], and the triage layer
        # above printed "No significant fairness violations detected."
        gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
        valid_groups = gm.get_valid_groups()
        if len(valid_groups) < 2:
            not_assessable.append(
                f"{col}: only {len(valid_groups)} group(s) meet "
                f"min_group_size={min_group_size}, so no comparison was possible"
            )
            skipped_columns.append(col)
            warnings.warn(
                f"Attribute {col!r} was NOT scanned: only {len(valid_groups)} group(s) "
                f"meet min_group_size={min_group_size}. An empty violation list for "
                "this attribute means it was not checked, not that it is clean.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # C-03, second half (2026-09-09). The fewer-than-two case above was
        # covered; the PARTIAL case was not. When SOME groups meet
        # min_group_size and one does not, every metric below is computed over
        # the qualifying subset and the dropped group leaves no trace at all.
        # Measured on 1000 applicants {white:500, black:300, asian:175,
        # native:25} where native was NEVER selected: the true worst
        # demographic-parity gap is 0.498 and the true four-fifths ratio 0.0,
        # yet this reported demographic_parity_difference=0.0180 PASS,
        # disparate_impact_ratio=0.9639 PASS, violations=[] and
        # not_assessable=[]. An empty `not_assessable` is a POSITIVE assertion
        # that nothing was skipped, and it was false. The metric layer does warn
        # ("Excluding 1 group(s) below min_group_size"), but the assessment that
        # found this ran under `python -W ignore`: the FIELD has to be right,
        # not only the warning.
        dropped_groups = gm.get_invalid_groups()
        if dropped_groups:
            group_sizes = gm.get_group_sizes()
            dropped_detail = []
            for name in sorted(dropped_groups, key=lambda g: -group_sizes[g]):
                never_selected = bool(np.all(y_pred[gm.get_mask(name)] != 1))
                dropped_detail.append(
                    f"{name!r} (n={group_sizes[name]}"
                    + (", NEVER SELECTED: 0 positive predictions" if never_selected else "")
                    + ")"
                )
                if never_selected:
                    zero_selection_alerts.append(
                        {
                            "scope": f"{col} (single attribute)",
                            "group": name,
                            "size": group_sizes[name],
                            "positive_count": 0,
                            "analysed": False,
                            "reason": (
                                "zero positive predictions; excluded by "
                                f"min_group_size={min_group_size} (n={group_sizes[name]}) "
                                "so it is absent from every metric for this attribute"
                            ),
                        }
                    )
            detail = ", ".join(dropped_detail)
            not_assessable.append(
                f"{col}: {len(dropped_groups)} group(s) below "
                f"min_group_size={min_group_size} were excluded ({detail}). Every metric "
                f"reported for {col!r} is computed over the {len(valid_groups)} remaining "
                "group(s) only, so its value is a partial result, not the worst gap."
            )
            warnings.warn(
                f"Attribute {col!r} was scanned over a SUBSET of its groups: {detail} "
                f"did not meet min_group_size={min_group_size} and were excluded. A "
                "metric under its threshold here does not mean the excluded group(s) "
                "were treated fairly; they were not measured.",
                UserWarning,
                stacklevel=2,
            )

        for metric_name in metrics:
            metric_fn, requires_labels = _METRIC_REGISTRY[metric_name]
            metric_threshold = thresholds.get(metric_name, threshold) if thresholds else threshold

            try:
                # All metric functions have signature:
                # (y_true, y_pred, sensitive_attr, *, min_group_size)
                y_true_arg = y_true if y_true is not None else y_pred
                disparity = metric_fn(
                    y_true_arg, y_pred, sensitive_attr, min_group_size=min_group_size
                )

                # A NaN disparity fails EVERY comparison, so `>` is False and the
                # metric silently leaves the violation list. Recorded instead: a
                # metric that could not be computed is not a metric that passed.
                if not np.isfinite(disparity):
                    not_assessable.append(
                        f"{col}/{metric_name}: metric could not be computed (returned {disparity})"
                    )
                    continue

                # Recorded BEFORE the threshold test: a measurement is a
                # measurement whether or not it breached anything, and this is
                # the only place the value exists.
                measured_disparities.append(float(disparity))

                if disparity > metric_threshold:
                    # Identify the privileged/disadvantaged pair from the RATE
                    # the triggering metric is built from. Using the selection
                    # rate for every metric inverted the attribution for
                    # error-based metrics (audit3): see
                    # _identify_violation_groups.
                    priv_group, disadv_group = _identify_violation_groups(
                        metric_name, gm, y_pred, y_true
                    )
                    # (None, None) means the pair could not be identified; it
                    # is reported as "unknown", never formatted as "None".

                    violations.append(
                        FairnessViolation(
                            attribute=col,
                            metric=metric_name,
                            value=disparity,
                            threshold=metric_threshold,
                            severity=_classify_severity(disparity),
                            privileged_group=priv_group or "unknown",
                            disadvantaged_group=disadv_group or "unknown",
                            disparity=disparity,
                        )
                    )

            except Exception as e:
                # BGL-C (2026-09-17). The warning was the ONLY channel: a metric
                # that raised left no trace in `not_assessable`, so under
                # `python -W ignore` (how the assessment that found the sibling
                # defects ran) it was indistinguishable from a metric that
                # passed. Measured on 1000 rows with a real 0.60 gap and a
                # three-class y_pred, which every metric here refuses: the scan
                # computed nothing and `rank_fairness_issues` still reported
                # n_attributes_checked=1, n_not_assessable=0, max_disparity=0.0
                # and "No significant fairness violations detected."
                not_assessable.append(f"{col}/{metric_name}: metric raised {type(e).__name__}: {e}")
                warnings.warn(
                    f"Failed to compute {metric_name} for column '{col}': {e}. "
                    "This metric is absent from the violation list because it could "
                    "not be computed, not because it passed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

    # Sort by disparity (descending)
    violations.sort(key=lambda x: x.disparity, reverse=True)
    if not_assessable:
        warnings.warn(
            f"{len(not_assessable)} check(s) could not be assessed and are absent "
            f"from the violation list: {'; '.join(not_assessable[:5])}"
            + (" ..." if len(not_assessable) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )
    # Carried on a list SUBCLASS so `rank_fairness_issues` can report it without
    # changing this function's return type. A plain list refuses attributes
    # ("no __dict__ for setting new attributes"), so the obvious version of this
    # would have discarded the data silently, which is the defect being fixed.
    result = _ScanResult(violations)
    result.metrics_not_attempted = metrics_not_attempted
    result.not_assessable = not_assessable
    result.skipped_columns = skipped_columns
    result.zero_selection_alerts = zero_selection_alerts
    result.measured_disparities = measured_disparities
    return result


class _IntersectionalResult(list):
    """A list of intersectional findings that can also say what it could not check.

    R-2 (2026-09-09). `discover_intersectional_groups` returned a bare list, so
    the cells `identify_privileged_groups` had dropped below `min_group_size`
    had nowhere to go and were discarded at the call site. An intersectional
    cell the model NEVER SELECTED therefore vanished from the whole assessment.
    This stays a list for every existing caller and carries the dropped cells
    for the ones that ask, exactly as `_ScanResult` does for the single
    attribute scan.
    """

    not_assessable: List[str]
    zero_selection_alerts: List[Dict[str, Any]]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.not_assessable = []
        self.zero_selection_alerts = []


def discover_intersectional_groups(
    df: pd.DataFrame,
    protected_attrs: List[str],
    y_pred: np.ndarray,
    *,
    max_combinations: int = 2,
    min_group_size: int = 30,
    min_disparity: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Discover meaningful intersectional group combinations.

    Generates combinations of protected attributes and identifies
    those with significant disparities.

    Args:
        df: DataFrame containing features
        protected_attrs: List of protected attribute column names
        y_pred: Predicted labels
        max_combinations: Maximum number of attributes to combine (2 or 3)
        min_group_size: Minimum samples per group
        min_disparity: Minimum disparity to report

    Returns:
        List of dicts with intersectional analysis results:
            - attributes: List of combined attributes
            - disparity: Maximum disparity found
            - privileged_group: Most advantaged intersection
            - disadvantaged_group: Most disadvantaged intersection
            - n_groups: Number of intersectional groups
            - hidden_disparity: Additional disparity revealed vs single attributes;
              nan when no single attribute in the combination could be assessed
            - reveals_more: True / False when the comparison was made, None when
              it could not be (hidden_disparity is nan)
            - single_attributes_not_assessable: attributes in the combination whose
              own disparity could not be assessed and were excluded from the
              comparison
            - cells_not_assessable: the intersectional cells dropped by
              ``min_group_size``, each with {group, size, positive_count, ...}.
              When this is non-empty, ``disparity`` is the worst gap among the
              cells that REMAINED, not the worst gap that exists.
            - disparity_over_subset: True when cells were dropped, so
              ``disparity`` is a partial aggregate; False when every cell was
              measured.
            - zero_selection_alerts: cells in this combination where the model
              selected nobody, whether or not they met ``min_group_size``.

        The returned list also carries two attributes, in the style of
        ``_ScanResult``: ``not_assessable`` (one line per dropped cell; per
        requested attribute that is not a column of ``df``; per single attribute
        whose own disparity could not be assessed; per combination that could
        not be evaluated; and one line when no combination could be formed at
        all) and
        ``zero_selection_alerts``. They are what ``rank_fairness_issues``
        reports as ``n_not_assessable``. THEY EXIST BECAUSE A DROPPED CELL IS
        NOT REPORTABLE THROUGH THE RESULT DICTS: a combination whose remaining
        cells are within ``min_disparity`` produces no dict at all, and that is
        exactly the case where the dropped cell was the finding.

    Example:
        >>> results = discover_intersectional_groups(
        ...     df, ['gender', 'race'], y_pred
        ... )
        >>> for r in results:
        ...     print(f"{r['attributes']}: {r['disparity']:.1%} disparity")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: discover_intersectional_groups. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    from itertools import combinations

    from .intersectional import identify_privileged_groups

    results = []
    y_pred = np.asarray(y_pred)

    # R-2 (2026-09-09). Every cell `identify_privileged_groups` drops below
    # min_group_size, and every zero-selection alert it raises, is collected
    # here instead of being discarded at the call site. Measured on 1200 rows,
    # race {white:1000, black:200} x sex {M,F}: the cell black_F (n=20, rate
    # 0.0) was excluded by the size gate, the layer below reported it in BOTH
    # `excluded_groups` and `zero_selection_alerts`, and this function read only
    # `max_disparity` -- so the assessment above reported n_violations=0,
    # n_not_assessable=0, not_assessable=[], max_disparity=0.0 and "No
    # significant fairness violations detected. Continue monitoring." against a
    # true cell gap of 0.53, with zero warnings on any channel.
    not_assessable: List[str] = []
    zero_alerts: List[Dict[str, Any]] = []

    def _record_dropped_cells(label: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pull the dropped cells and zero-selection alerts out of one analysis.

        Returns the excluded cells so the caller can attach them to the result
        dict as well; the strings go to the shared `not_assessable` channel.
        Direct indexing, not `.get`: both return paths of
        `identify_privileged_groups` carry these keys, and a shape change
        should be loud rather than read as "nothing was dropped".
        """
        excluded = list(analysis["excluded_groups"])
        for alert in analysis["zero_selection_alerts"]:
            zero_alerts.append({"scope": label, **alert})
        for cell in excluded:
            never_selected = cell["positive_count"] == 0
            not_assessable.append(
                f"{label}: cell {cell['group']!r} (n={cell['size']}"
                + (", NEVER SELECTED: 0 positive predictions" if never_selected else "")
                + f") was below min_group_size={min_group_size} and was excluded, so it "
                "is absent from every disparity reported for this combination"
            )
        return excluded

    # BGL-U01 (2026-09-17). An attribute that is NOT A COLUMN of the frame was
    # skipped by a bare `continue` in BOTH loops below, and a combination
    # containing one was skipped by a third. Nothing was recorded anywhere: the
    # function returned an empty list with `not_assessable == []` and no warning
    # on any channel, which is the exact shape of "every intersection was
    # checked and none was disparate". Measured at the public entry
    # (`rank_fairness_issues`) on 1200 rows where the schema had renamed 'race'
    # to 'ethnicity' and the caller still asked for ['sex', 'race'], with F x
    # black selected at 3% against 65% everywhere else: intersectional_issues=[],
    # not_assessable=[], n_not_assessable=0, zero warnings, and the 0.6 gap was
    # never looked at.
    usable_attrs = [a for a in protected_attrs if a in df.columns]
    missing_attrs = [a for a in protected_attrs if a not in df.columns]
    for attr in missing_attrs:
        not_assessable.append(
            f"{attr!r}: not a column of this frame, so neither its own disparity nor "
            "any intersection containing it was computed; it is absent from these "
            "results, which is not the same as having no disparity"
        )
    if missing_attrs:
        warnings.warn(
            "discover_intersectional_groups: requested attribute(s) "
            + ", ".join(repr(a) for a in missing_attrs)
            + " are not columns of the frame. Every intersection containing them was "
            "SKIPPED, not cleared; they are listed in not_assessable.",
            UserWarning,
            stacklevel=2,
        )

    # No pair can be formed at all: `combinations` below yields nothing and the
    # empty return would otherwise read as "intersections checked, none found".
    n_pairable = min(max_combinations, len(usable_attrs))
    if n_pairable < 2:
        not_assessable.append(
            f"no intersectional combination was formed at all: {len(usable_attrs)} "
            f"usable attribute(s) {usable_attrs} with max_combinations="
            f"{max_combinations}; an empty result here means nothing was intersected, "
            "not that no intersectional disparity exists"
        )
        warnings.warn(
            f"discover_intersectional_groups: no combination could be formed from "
            f"{usable_attrs} with max_combinations={max_combinations}. Nothing was "
            "intersected; this empty result does not say the data is clean.",
            UserWarning,
            stacklevel=2,
        )

    # First, compute single-attribute disparities.
    #
    # R-5 (audit 6, 2026-09-09). A single-attribute disparity that could not be
    # assessed used to enter the max() below as a 0.0 floor: a swallowed
    # exception became `0.0` outright, and a constant column (whose own
    # disparity is correctly nan) rode through `max(...)` on Python's nan
    # ordering. Either way `hidden = max(0, disparity - max_single)` fabricated
    # a number. Measured: attributes=['gender','site'] with a constant 'site'
    # reported disparity=0.69, hidden_disparity=0, reveals_more=False and no
    # warning from this function. This is audit finding H-01, already closed in
    # the sibling implementation (intersectional.py:590, _generate_comparison):
    # an unassessable attribute is named and left out of the maximum, and when
    # none is assessable the comparison is nan / None, never 0 / False.
    single_disparities: Dict[str, float] = {}
    single_not_assessable: Dict[str, str] = {}
    for attr in usable_attrs:
        try:
            result = identify_privileged_groups(
                y_pred, y_pred, df[attr].values, min_group_size=min_group_size
            )
            single_disparities[attr] = result["max_disparity"]
        except Exception as exc:
            # Not 0.0: an attribute that raised was not measured at all.
            single_disparities[attr] = float("nan")
            single_not_assessable[attr] = f"{type(exc).__name__}: {exc}"
        else:
            # The same discard happened here: a single attribute's own dropped
            # cells were read past on the way to `max_disparity`, so
            # `single_disparities[attr]` entered the hidden-disparity comparison
            # below as if it covered every group of that attribute.
            _record_dropped_cells(f"{attr} (single attribute)", result)
            if not is_measured(single_disparities[attr]):
                single_not_assessable[attr] = (
                    "no assessable disparity (max_disparity is not finite)"
                )
    if single_not_assessable:
        # BGL-U01. These were warned about and NOTHING ELSE. They reach a result
        # dict through `single_attributes_not_assessable`, but only when a dict
        # is produced at all, and `rank_fairness_issues` counts
        # `n_not_assessable` off this list, never off the warning. By this
        # function's own standard (the assessment that found R-2 ran under
        # `python -W ignore`) a warning-only refusal is a silent one.
        for attr, why in single_not_assessable.items():
            not_assessable.append(
                f"{attr} (single attribute): its own disparity could not be assessed "
                f"({why}), so it is excluded from every hidden-disparity comparison "
                "reported here"
            )
        warnings.warn(
            "discover_intersectional_groups: single-attribute disparity could not be "
            "assessed for "
            + "; ".join(f"'{a}' ({why})" for a, why in single_not_assessable.items())
            + ". These attributes are excluded from every hidden-disparity comparison.",
            UserWarning,
            stacklevel=2,
        )

    # Generate combinations over the attributes that ARE columns. The absent
    # ones are already named in `not_assessable` above; combining over
    # `protected_attrs` and dropping the bad combinations with a bare `continue`
    # is what made them disappear without trace.
    for r in range(2, n_pairable + 1):
        for combo in combinations(usable_attrs, r):
            try:
                # Create intersectional attribute
                intersect_df = df[list(combo)]
                intersect_attr = intersect_df.apply(
                    lambda row: "_".join(str(v) for v in row), axis=1
                ).values

                result = identify_privileged_groups(
                    y_pred, y_pred, intersect_attr, min_group_size=min_group_size
                )

                disparity = result["max_disparity"]

                # BEFORE the min_disparity gate, always. A dropped cell is at
                # its most dangerous exactly when the cells that survived look
                # calm: the combination then produces no result dict at all and
                # the cell would leave no trace anywhere.
                combo_label = " x ".join(combo)
                excluded_cells = _record_dropped_cells(combo_label, result)

                if not is_measured(disparity):
                    # Not a combination without disparity: a combination with no
                    # disparity that could be computed. It falls through the
                    # `>=` below silently because every comparison with nan is
                    # False.
                    not_assessable.append(
                        f"{combo_label}: no intersectional disparity could be computed "
                        f"(max_disparity={disparity!r}); this combination is absent from "
                        "the results, which is not the same as having no disparity"
                    )
                    continue

                if disparity >= min_disparity:
                    # Calculate hidden disparity over the ASSESSED single
                    # attributes only (R-5, see the loop above). Three states:
                    # a number when at least one single attribute was measured,
                    # nan / None when none was, never a 0.0 floor.
                    assessed = [a for a in combo if is_measured(single_disparities.get(a))]
                    unassessed = [a for a in combo if a not in assessed]
                    hidden: float
                    reveals_more: Optional[bool]
                    if assessed:
                        max_single = max(float(single_disparities[a]) for a in assessed)
                        hidden = max(0, disparity - max_single)
                        reveals_more = hidden > 0.01
                    else:
                        hidden = float("nan")
                        reveals_more = None
                        warnings.warn(
                            f"discover_intersectional_groups: no single-attribute disparity "
                            f"was assessable for {list(combo)} "
                            f"({', '.join(unassessed)}); reporting hidden_disparity=nan and "
                            f"reveals_more=None (could not check).",
                            UserWarning,
                            stacklevel=2,
                        )

                    priv = result.get("privileged_group")
                    disadv = result.get("disadvantaged_group")

                    results.append(
                        {
                            "attributes": list(combo),
                            "disparity": disparity,
                            "privileged_group": priv.group if priv else "unknown",
                            "disadvantaged_group": disadv.group if disadv else "unknown",
                            "n_groups": len(result.get("all_groups", [])),
                            "hidden_disparity": hidden,
                            "reveals_more": reveals_more,
                            "single_attributes_not_assessable": unassessed,
                            # An aggregate over a subset has to say so. With
                            # cells dropped, `disparity` is the worst gap among
                            # what remained, and the reader cannot tell that
                            # from the number alone.
                            "cells_not_assessable": excluded_cells,
                            "disparity_over_subset": bool(excluded_cells),
                            "zero_selection_alerts": list(result["zero_selection_alerts"]),
                        }
                    )

            except Exception as exc:
                # A combination that could not be evaluated is not a
                # combination without disparity. Say which one was dropped.
                # BGL-U01: on the `not_assessable` channel as well as the
                # warning one, because that is the channel the layer above
                # counts and the one that survives `-W ignore`.
                not_assessable.append(
                    f"{' x '.join(combo)}: the combination was not evaluated at all "
                    f"({type(exc).__name__}: {exc}); it is absent from these results, "
                    "which is not the same as having no disparity"
                )
                warnings.warn(
                    f"discover_intersectional_groups: skipped combination "
                    f"{list(combo)} ({type(exc).__name__}: {exc}); this result is incomplete",
                    ProxyScanIncompleteWarning,
                    stacklevel=2,
                )
                continue

    # Sort by disparity (descending)
    results.sort(key=lambda x: x["disparity"], reverse=True)

    never_selected = [a for a in zero_alerts if a["positive_count"] == 0 and not a["analysed"]]
    if never_selected:
        warnings.warn(
            "discover_intersectional_groups: "
            + "; ".join(
                f"{a['scope']} cell {a['group']!r} (n={a['size']}) was NEVER SELECTED "
                "and is below min_group_size"
                for a in never_selected
            )
            + ". A cell the model selected nobody from is the headline, not the noise "
            "floor: it is reported in not_assessable / zero_selection_alerts and is "
            "absent from every disparity number here.",
            UserWarning,
            stacklevel=2,
        )
    if not_assessable:
        warnings.warn(
            f"discover_intersectional_groups: {len(not_assessable)} cell(s)/combination(s) "
            "could not be assessed and are absent from these results: "
            + "; ".join(not_assessable[:5])
            + (" ..." if len(not_assessable) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )

    # Carried on a list SUBCLASS, exactly as `scan_fairness_violations` does, so
    # every existing caller keeps a list and `rank_fairness_issues` can report
    # what was dropped.
    out = _IntersectionalResult(results)
    out.not_assessable = not_assessable
    out.zero_selection_alerts = zero_alerts
    return out


def rank_fairness_issues(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    *,
    protected_columns: Optional[List[str]] = None,
    auto_detect: bool = True,
    include_intersectional: bool = True,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """
    Comprehensive fairness analysis with ranked issues.

    Combines auto-detection, single-attribute analysis, and intersectional
    analysis to provide a prioritized list of fairness issues.

    Args:
        df: DataFrame containing features
        y_pred: Predicted labels
        y_true: Optional true labels
        protected_columns: Specific columns to check
        auto_detect: Whether to auto-detect protected attributes
        include_intersectional: Whether to include intersectional analysis
        min_group_size: Minimum group size for analysis

    Returns:
        Dict containing:
            - detected_attributes: Auto-detected protected attributes
            - violations: List of fairness violations (sorted by severity)
            - intersectional_issues: List of intersectional disparities
            - proxy_warnings: Potential proxy features
            - summary: Overall summary statistics. ``max_disparity`` is the
              largest disparity the scan MEASURED, not the largest one that
              breached a threshold, and it is NaN when any attribute-level check
              was not assessable (the maximum over the whole scan is unknown
              then) or when nothing was measured at all.
              ``n_disparities_measured`` says how many values it is taken over.
            - recommendations: Prioritized recommendations

    Example:
        >>> analysis = rank_fairness_issues(df, y_pred)
        >>> print(f"Found {len(analysis['violations'])} violations")
        >>> for v in analysis['violations'][:5]:
        ...     print(f"  {v.attribute}: {v.disparity:.1%}")
    """
    result: Dict[str, Any] = {
        "detected_attributes": [],
        "violations": [],
        "intersectional_issues": [],
        "proxy_warnings": [],
        "summary": {},
        "recommendations": [],
    }

    # Detect protected attributes
    if auto_detect or protected_columns is None:
        candidates = detect_protected_attributes(df)
        result["detected_attributes"] = [c.to_dict() for c in candidates]
        columns_to_check = (
            [c.column for c in candidates] if not protected_columns else protected_columns
        )
    else:
        columns_to_check = protected_columns

    y_pred = np.asarray(y_pred)

    # Scan for violations
    violations = scan_fairness_violations(
        df,
        y_pred,
        y_true,
        protected_columns=columns_to_check,
        auto_detect=False,
        min_group_size=min_group_size,
    )
    result["violations"] = [v.to_dict() for v in violations]

    # Intersectional analysis
    inter_results: List[Dict[str, Any]] = _IntersectionalResult()
    if include_intersectional and len(columns_to_check) >= 2:
        inter_results = discover_intersectional_groups(
            df,
            columns_to_check[:5],  # Limit to top 5 attributes
            y_pred,
            min_group_size=min_group_size,
        )
        result["intersectional_issues"] = inter_results

    # Proxy detection for top violations
    proxy_not_assessable: List[str] = []
    for v in violations[:3]:
        try:
            proxies = identify_proxy_features(df, v.attribute, correlation_threshold=0.3)
            for p in proxies[:3]:
                result["proxy_warnings"].append({"protected_attribute": v.attribute, **p})
            # A column the proxy scan could not read is not a column without
            # proxy risk, and `n_proxy_warnings` alone cannot say which it was.
            proxy_not_assessable.extend(getattr(proxies, "not_assessable", []))
        except Exception as exc:
            warnings.warn(
                f"proxy detection skipped for '{v.attribute}' "
                f"({type(exc).__name__}: {exc}); proxy_warnings is incomplete",
                ProxyScanIncompleteWarning,
                stacklevel=2,
            )
            proxy_not_assessable.append(
                f"{v.attribute}/proxy scan: not run at all ({type(exc).__name__}: {exc})"
            )
            continue

    # Summary
    # C-03. `n_attributes_checked` counted every attribute REQUESTED, including
    # the ones skipped for having fewer than two valid groups, so a scan that
    # checked nothing still reported "1 attribute checked, 0 violations".
    scan_not_assessable = list(getattr(violations, "not_assessable", []))
    # The columns skipped ENTIRELY, by name. Recovering this by looking for a
    # "/" in each string counted a PARTIALLY scanned attribute as unscanned the
    # moment such an entry started being recorded.
    skipped_columns = list(getattr(violations, "skipped_columns", []))
    # R-2 (2026-09-09). The intersectional pass drops cells below
    # min_group_size, and those cells used to reach nobody: see
    # `discover_intersectional_groups`. They belong in the same could-not-check
    # channel as the scan's, because the reader of `not_assessable` is asking
    # one question -- "was anything left unmeasured?" -- and both answer it.
    inter_not_assessable = list(getattr(inter_results, "not_assessable", []))
    not_assessable = scan_not_assessable + inter_not_assessable + proxy_not_assessable
    # The registered metrics the scan never attempted. Kept out of
    # `not_assessable` on purpose (see the metric-menu block in
    # `scan_fairness_violations`): every attribute WAS scanned, so this must not
    # reduce `n_attributes_checked` nor turn the disparity that was measured into
    # a NaN, and it must not be silent either.
    metrics_not_attempted = list(getattr(violations, "metrics_not_attempted", []))
    # Both passes can see the same never-selected group (the intersectional pass
    # scores each single attribute on its way to the combinations), so they are
    # merged on (scope, group) rather than concatenated into a double count.
    zero_selection_alerts: List[Dict[str, Any]] = []
    seen_alerts = set()
    for alert in list(getattr(violations, "zero_selection_alerts", [])) + list(
        getattr(inter_results, "zero_selection_alerts", [])
    ):
        key = (alert["scope"], alert["group"])
        if key in seen_alerts:
            continue
        seen_alerts.add(key)
        zero_selection_alerts.append(alert)

    # THE LARGEST DISPARITY THAT WAS MEASURED, not the largest one that breached
    # a threshold.
    # BGL-5 (2026-09-27). This was `max(v.disparity for v in violations)` with a
    # 0.0 default, and `scan_fairness_violations` keeps a disparity only when it
    # exceeds its threshold, so every measured-but-passing gap was computed,
    # discarded, and then summarised as the number the comment below reserves for
    # a perfect run. Measured on 2000 rows with selection rates 0.503 and 0.478:
    #   before: {'n_attributes_checked': 1, 'n_violations': 0,
    #            'max_disparity': 0.0, 'n_not_assessable': 0}
    #   after:  the same summary with 'max_disparity': 0.0253... , the gap the
    #           scan actually measured
    # THREE cases, and the order between them is the point:
    #   1. a measured BREACH exists -> the largest breach. A real finding is
    #      never withdrawn because something else in the scan was unassessable;
    #      the repo already pins that as an over-correction control
    #      (test_bgl_final_u00: "one absent column must not suppress the real
    #      finding on the column that IS there", max_disparity 0.6 beside a
    #      non-empty not_assessable).
    #   2. no breach, and anything at all in this assessment was not assessable,
    #      or nothing was measured -> NaN, unchanged from before. The maximum
    #      over the run is then unknown, and a maximum over the subset that
    #      happened to work would read as the worst gap there is. The whole
    #      `not_assessable` channel counts here, the intersectional and proxy
    #      passes included, exactly as the old default did: the readiness suite
    #      pins NaN for a run whose only gap is an intersectional cell nobody
    #      could measure.
    #   3. no breach and nothing unassessable -> the largest gap MEASURED. This
    #      is the case that was broken, and the only one whose value changes.
    scan_measured = [d for d in getattr(violations, "measured_disparities", []) if np.isfinite(d)]
    breaches = [v.disparity for v in violations]
    if breaches:
        max_disparity = max(breaches)
    elif not_assessable or not scan_measured:
        # NaN, not 0.0, when nothing was measurable. A maximum over an empty set
        # is not zero disparity, and 0.0 is the value a perfectly fair run gets.
        max_disparity = float("nan")
    else:
        max_disparity = max(scan_measured)

    result["summary"] = {
        "n_attributes_checked": len(columns_to_check) - len(skipped_columns),
        "n_attributes_requested": len(columns_to_check),
        "n_not_assessable": len(not_assessable),
        "not_assessable": not_assessable,
        "n_violations": len(violations),
        "max_disparity": max_disparity,
        # How many disparities the maximum above is taken over, so a reader can
        # tell a measured 0.0 (every rate identical) from a NaN refusal without
        # inferring it from another field.
        "n_disparities_measured": len(scan_measured),
        "critical_violations": sum(1 for v in violations if v.severity == "critical"),
        "high_violations": sum(1 for v in violations if v.severity == "high"),
        "n_proxy_warnings": len(result["proxy_warnings"]),
        # A cell the model selected nobody from, whether or not it met
        # min_group_size. It is carried here rather than only warned about
        # because the assessment that found this ran under `python -W ignore`.
        "zero_selection_alerts": zero_selection_alerts,
        "n_zero_selection_alerts": len(zero_selection_alerts),
        # WHICH FAIRNESS QUESTIONS WERE NEVER ASKED.
        # BGL5 A-evaluation-3, 2026-09-29. Without y_true the scan evaluates
        # demographic parity alone, and nothing in this summary said so: a model
        # with a 1.0 error-rate disparity and identical selection rates got
        # n_violations 0, max_disparity 0.0 and the all-clear below. The numbers
        # are real; the SCOPE was missing. Always present, never only when
        # something was skipped, so an absent key cannot mean both "nothing was
        # missed" and "this version did not record it".
        "metrics_not_attempted": metrics_not_attempted,
        "n_metrics_not_attempted": len(metrics_not_attempted),
    }

    # Recommendations
    if result["summary"]["critical_violations"] > 0:
        result["recommendations"].append(
            "CRITICAL: Immediate action required. Review model for severe bias."
        )
    if result["summary"]["high_violations"] > 0:
        result["recommendations"].append(
            "HIGH PRIORITY: Investigate and address significant disparities."
        )
    if len(result["intersectional_issues"]) > 0:
        hidden = [i for i in result["intersectional_issues"] if i.get("reveals_more")]
        if hidden:
            result["recommendations"].append(
                f"INTERSECTIONAL: {len(hidden)} hidden disparities found. "
                "Ensure fairness interventions consider group intersections."
            )
        # R-5 (2026-09-09): reveals_more=None is "could not compare", not "nothing
        # hidden"; it must not vanish from the recommendations.
        uncompared = [i for i in result["intersectional_issues"] if i.get("reveals_more") is None]
        if uncompared:
            result["recommendations"].append(
                f"COULD NOT CHECK: {len(uncompared)} intersectional disparit"
                f"{'y' if len(uncompared) == 1 else 'ies'} could not be compared against "
                "single attributes (no single-attribute disparity was assessable)."
            )
    if result["proxy_warnings"]:
        result["recommendations"].append(
            f"PROXY RISK: {len(result['proxy_warnings'])} potential proxy features detected. "
            "Review features that correlate with protected attributes."
        )
    if zero_selection_alerts:
        # Louder than "could not check", and deliberately ahead of it: a group
        # the model selected nobody from is the starkest result an assessment
        # can produce, and it is the one the size gate hides.
        result["recommendations"].append(
            f"NEVER SELECTED: {len(zero_selection_alerts)} group(s) received zero positive "
            "predictions ("
            + "; ".join(
                f"{a['scope']} {a['group']!r}, n={a['size']}"
                + ("" if a["analysed"] else ", below min_group_size so NOT analysed")
                for a in zero_selection_alerts[:3]
            )
            + (" ..." if len(zero_selection_alerts) > 3 else "")
            + "). Investigate these before reading any disparity number as clean."
        )
    if not_assessable:
        result["recommendations"].append(
            f"COULD NOT CHECK: {len(not_assessable)} check(s) were not assessable "
            f"and are absent from the violation list ({'; '.join(not_assessable[:3])}"
            + (" ..." if len(not_assessable) > 3 else "")
            + "). An empty violation list for those is not a clean result."
        )
    if metrics_not_attempted:
        # Ahead of the all-clear and in place of it: this is the sentence that has
        # to reach a reader who would otherwise take "no violations" for "fair".
        result["recommendations"].append(
            f"NOT MEASURED: {len(metrics_not_attempted)} fairness metric(s) were never "
            f"attempted ({', '.join(metrics_not_attempted)}) because no y_true was "
            f"supplied, so only selection rates were compared. A model that is perfect "
            f"for one group and inverted for another has IDENTICAL selection rates: an "
            f"error-rate or precision disparity of any size is invisible to this run. "
            f"Supply y_true to measure them."
        )
    if (
        not violations
        and not not_assessable
        and not zero_selection_alerts
        and not metrics_not_attempted
    ):
        # The all-clear is a statement about FAIRNESS, so it is withheld when a
        # whole metric was never attempted, exactly as it is when an attribute was
        # never scanned. It is NOT withheld for a measured 0.0: that is a
        # measurement of parity and keeps its all-clear.
        result["recommendations"].append(
            "No significant fairness violations detected. Continue monitoring."
        )

    return result


# COLUMN-ROLE TYPOLOGY
#
# Before any fairness analysis runs, every column needs a ROLE so the engine
# knows what may legitimately feed a model, what is the answer key, and what
# must never enter a model at all. detect_protected_attributes() only finds
# protected columns; this classifies the WHOLE schema and is the gate behind
# the Pulse "refuse-fast" step and the Navigator profiling check (one source
# of truth for all three surfaces).

# Identity / PII: never a model feature, never a protected-group axis.
_PII_PATTERNS = [
    "first_name",
    "last_name",
    "given_name",
    "surname",
    "middle_name",
    "full_name",
    "name",
    "email",
    "e_mail",
    "phone",
    "mobile",
    "fax",
    "ssn",
    "social_security",
    "passport",
    "national_id",
    "tax_id",
    "license",
    "street",
    "address",
    "addr",
    "postal_address",
    "photo",
    "image_url",
    "picture",
    "avatar",
    "headshot",
    "linkedin",
    "twitter",
    "facebook",
    "url",
    "profile_url",
    "ip_address",
    "device_id",
    "mac_address",
    "account_number",
    # Date of birth is direct PII AND an exact age key -- must never be a
    # model feature (spec: flag DOB as "must not enter the model").
    "dob",
    "date_of_birth",
    "birthdate",
    "birth_date",
    "birthday",
    "yob",
]
_ID_PATTERNS = [
    "id",
    "uuid",
    "guid",
    "identifier",
    "ref",
    "reference",
    "record_no",
    "row_id",
    "case_id",
]
# Explicit model-output names ONLY. A bare "score" matched feature columns
# like skill_match_score / video_interview_score -- those are PREDICTORS,
# not the model's decision. The decision column is normally declared.
_MODEL_OUTPUT_PATTERNS = [
    "prediction",
    "predicted",
    "y_pred",
    "model_score",
    "model_prediction",
    "model_output",
    "predicted_label",
    "risk_score",
    "invite_decision",
]
_MODEL_OUTPUT_TOKENS = {"prediction", "predicted", "pred", "decision", "probability", "proba"}
_ORACLE_PATTERNS = [
    "label",
    "y_true",
    "target",
    "outcome",
    "ground_truth",
    "actual",
    "true_",
    "_truth",
    "qualification_score",
    "is_qualified",
    "gold",
    "oracle",
    "y",
]
# Feature names that are commonly proxies (zip, university, school...) so we
# can tag proxy_candidate cheaply even without running a correlation pass.
_PROXY_NAME_HINTS = [
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "neighborhood",
    "census_tract",
    "block_group",
    "school",
    "college",
    "university",
    "tier",
    "gap_months",
    "employment_gap",
    "minority",
    "majority",
    "device",
    "browser",
    "credit_score",
    "graduation_year",
]


@dataclass
class ColumnRole:
    """Inferred role of one column in a fairness assessment."""

    column: str
    role: str  # identity_pii | protected | proxy_candidate |
    # job_relevant | model_output | oracle | unknown
    confidence: float
    reason: str
    category: str = ""  # protected category when role == 'protected'
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "category": self.category,
            "evidence": self.evidence,
        }


def classify_column_roles(
    df: pd.DataFrame,
    *,
    declared_protected: Optional[List[str]] = None,
    declared_target: Optional[str] = None,
    declared_prediction: Optional[str] = None,
    min_protected_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Classify every column's role and surface schema risks.

    Precedence (a column gets the FIRST role that matches): oracle ->
    model_output -> identity_pii -> protected -> proxy_candidate ->
    job_relevant -> unknown. Identity/PII outranks proxy/job-relevant so a
    name is flagged "must not model" even though it also leaks gender.

    Args:
        df: dataset
        declared_protected: columns the user marked protected (trusted, but
            still verified -- a declared job-relevant column that strongly
            predicts a protected attribute is reported as a mismatch)
        declared_target / declared_prediction: user-declared oracle / model
            output columns (override name heuristics)
        min_protected_confidence: threshold for auto-detected protected cols

    Returns dict:
        roles: List[ColumnRole.to_dict()]
        pii_leakage: [col]      -- identity/PII that must never be a feature
        oracle_columns: [col]   -- the answer key; never a feature
        model_output: [col]     -- the model's decision/score
        protected: [col]
        proxy_candidates: [col]
        job_relevant: [col]
        mismatches: [{column, declared, inferred, detail}]
        not_assessable: [str]   what this call could not measure: what the
            CORRELATION proxy screen could not reach, and every declaration it
            could not honour, each entry naming the column or target and the
            reason. An empty `proxy_candidates` beside a non-empty list here is
            "we could not look", NOT "there is nothing to find", and the same
            goes for an empty `model_output`.
        n_not_assessable: int
        refuse: bool            -- a critical schema error is present
        refuse_reason: str
    """
    # A DECLARED PROTECTED COLUMN THAT IS NOT IN THE FRAME WAS DISCARDED HERE,
    # SILENTLY.
    # BGL-5 (2026-09-27). This filter drops the caller's declaration with no
    # note, no warning and no mismatch entry, and `has_declared` below then
    # collapses to False so the function falls back to auto-detection as though
    # nothing had been declared. Measured on 400 rows holding only `gender` and
    # `salary`:
    #   classify_column_roles(df, declared_protected=["ethnicity"])
    #     before -> n_not_assessable 0, not_assessable [], mismatches [],
    #               refuse False, protected ['gender'] ("auto-detected"),
    #               warnings NONE
    #     after  -> n_not_assessable 1, not_assessable ["ethnicity: declared
    #               protected but there is no such column ..."], mismatches
    #               [{'column': 'ethnicity', 'declared': 'protected',
    #                 'inferred': 'absent', 'detail': ...}] and a UserWarning
    #               naming it; `protected` is unchanged, because auto-detection
    #               is still the best available answer.
    # This is the same shape the file records as a defect at
    # scan_fairness_violations ("A misspelt or renamed column bought a clean bill
    # of health for an attribute nobody had looked at"), arriving through a list
    # comprehension instead of a `continue`.
    #
    # It deliberately does NOT set `refuse`: the refuse flag aborts a whole Pulse
    # run before any battery (operations/pulse/orchestrator.py reads
    # `schema.get("refuse")`), so one misspelt name among several correct ones
    # would destroy an analysis that is otherwise entirely measurable. The
    # mismatch is a "warn" finding in the compliance report
    # (operations/reporting/compliance.py reads `mismatches`), which is where a
    # person sees it.
    declared_requested = list(declared_protected or [])
    declared_absent = [c for c in declared_requested if c not in df.columns]
    declared_protected = [c for c in declared_requested if c in df.columns]
    decl_prot = set(declared_protected)

    # THE SIBLING ARGUMENT IN THE SAME SIGNATURE GOT NOTHING AT ALL.
    # BGL5 A-evaluation-3, 2026-09-29. The three `declared_*` arguments answered
    # an absent column three different ways: `declared_protected` on the three
    # channels above and below, `declared_target` with refuse=True (the loudest
    # channel there is: it aborts the whole Pulse run), and `declared_prediction`
    # with NOTHING. The name simply never equalled a column, so `model_output`
    # came back [] with no mismatch, no not_assessable entry, no warning and
    # refuse False, and a misspelt prediction column was indistinguishable from a
    # frame that genuinely holds no model output. Measured on 400 rows holding
    # only `gender` and `salary`, classify_column_roles(df,
    # declared_protected=['gender'], declared_prediction='y_hat'):
    #   before -> model_output [], n_not_assessable 0, not_assessable [],
    #             mismatches [], refuse False, warnings NONE
    #   after  -> the same roles, plus one not_assessable entry, one mismatch and
    #             a UserWarning naming 'y_hat'; refuse stays False
    # That empty list is READ: operations/pulse/orchestrator.py:4640 puts it in
    # the leakage-exclusion set, operations/pulse/regulatory.py:491 turns it into
    # the governance narrative, and operations/pulse/causal_skeleton.py:352 takes
    # the decision node from it or falls back to a template default. All three
    # received "this frame has no model output" as a finding about the data.
    # Deliberately NOT refuse, for the same reason as declared_protected above: a
    # misspelt prediction column must not destroy an analysis whose data checks
    # are entirely measurable. `declared_target` is the one that refuses, because
    # without the answer key nothing downstream of it can be measured at all.
    prediction_absent = bool(declared_prediction) and declared_prediction not in df.columns

    # AND THE DECLARATION-VERSUS-DECLARATION COLLISION REACHED NO CHANNEL EITHER.
    # BGL-W4, 2026-09-30. The fix above moved the explicit declaration ahead of
    # the name heuristic and left the case where the caller declares ONE column as
    # both the answer key and the model output, a flat contradiction. The oracle
    # test is written first and `declared_target` wins it, so `model_output` came
    # back EMPTY: bit for bit the same neutral value, from a third route, with
    # nothing saying why. Measured on 400 rows holding gender, age, label and
    # y_hat, classify_column_roles(df, declared_protected=['gender'],
    # declared_target='y_hat', declared_prediction='y_hat'):
    #   before -> oracle_columns ['label', 'y_hat'], model_output [],
    #             not_assessable [], n_not_assessable 0, mismatches carrying only
    #             the unrelated 'age' proxy hint, refuse False, refuse_reason '',
    #             and ZERO warnings
    #   after  -> the same resolution (the answer key still wins, which is the
    #             decision recorded at the oracle test below), plus one
    #             not_assessable entry, one mismatch naming the column and a
    #             UserWarning; refuse stays False
    # The resolution is deliberately unchanged and only the silence is closed: the
    # three live consumers of model_output (operations/pulse/orchestrator.py:4640
    # leakage exclusion, operations/pulse/regulatory.py:491 governance narrative,
    # operations/pulse/causal_skeleton.py:352 causal decision node) were reading
    # "this frame has no model output column" for a frame whose model output the
    # caller had named. Not `refuse`, for the same reason as the two cases above.
    prediction_is_target = bool(declared_prediction) and declared_prediction == declared_target

    def _name_hit(col: str, pats: List[str]) -> Optional[str]:
        cl = col.lower().replace("-", "_")
        toks = set(cl.split("_"))
        for p in pats:
            if p == cl or p in toks or (len(p) >= 4 and p in cl):
                return p
        return None

    # Auto-detected protected attributes (reuse the canonical detector).
    prot_cat: Dict[str, Tuple[float, str]] = {}
    try:
        for c in detect_protected_attributes(
            df, min_confidence=min_protected_confidence, include_numeric=True
        ):
            prot_cat[c.column] = (float(c.confidence), c.category)
    except Exception:  # noqa: BLE001 -- never crash the typology gate
        prot_cat = {}

    n_rows = max(1, len(df))
    has_declared = len(decl_prot) > 0
    roles: List[ColumnRole] = []
    buckets: Dict[str, List[str]] = {
        "identity_pii": [],
        "oracle": [],
        "model_output": [],
        "protected": [],
        "proxy_candidate": [],
        "job_relevant": [],
        "unknown": [],
    }

    # Precompute correlation-based proxy hits ONCE (was per-column = slow).
    proxy_hit: Dict[str, Dict[str, Any]] = {}
    # What the CORRELATION proxy screen could not measure. `proxy_candidates` is
    # read as the answer to "which features track a protected attribute", and an
    # empty one must not mean both "the screen found nothing" and "the screen
    # measured nothing". `identify_proxy_features` was given a `not_assessable`
    # list for exactly this reason, and `rank_fairness_issues` consumes it; this
    # second consumer dropped it on the floor, so the only channel left was the
    # warning. Measured 2026-09-27 on 200 rows whose declared `gender` holds ONE
    # level: identify_proxy_features warned "2 of 2 candidate columns could NOT
    # be assessed" and this function returned proxy_candidates ['home_region']
    # (a NAME hint, no correlation behind it), refuse False and no field naming
    # the gap at all, so under `python -W ignore` a screen that measured nothing
    # was indistinguishable from a clean one.
    not_assessable: List[str] = []
    # The declared targets that are not columns at all. Recorded before the screen
    # runs, because they are the reason it will not look at them; see the comment
    # at the top of this function for the measured before and after.
    if prediction_absent:
        not_assessable.append(
            f"{declared_prediction}: declared as the model output but there is no such "
            f"column in the DataFrame (available: {list(df.columns)[:10]}), so it was "
            f"NOT classified and model_output below is empty because the declaration "
            f"could not be honoured, not because this frame holds no model output"
        )
        warnings.warn(
            f"classify_column_roles: declared model output {declared_prediction!r} is NOT "
            f"a column in this DataFrame. `model_output` is empty because the declaration "
            f"names nothing, not because there is no model output to find: fix the name or "
            f"remove the declaration. Anything reading model_output (leakage exclusion, "
            f"the governance narrative, the causal decision node) is reading an absence "
            f"that was never measured.",
            UserWarning,
            stacklevel=2,
        )
    if prediction_is_target:
        not_assessable.append(
            f"{declared_prediction}: declared as BOTH the target and the model output, "
            f"which cannot both be true of one column. It was classified as the oracle "
            f"(the answer key wins), so model_output below is empty because the two "
            f"declarations contradict each other, not because this frame holds no model "
            f"output"
        )
        warnings.warn(
            f"classify_column_roles: {declared_prediction!r} was declared as BOTH "
            f"declared_target and declared_prediction. One column cannot be both the "
            f"answer key and the model's decision. It is classified as the oracle and "
            f"`model_output` is empty AS A RESULT OF THE CONTRADICTION, not as a finding "
            f"about the data: declare the prediction column separately, or drop one of the "
            f"two declarations. Anything reading model_output (leakage exclusion, the "
            f"governance narrative, the causal decision node) would otherwise read an "
            f"absence that was never measured.",
            UserWarning,
            stacklevel=2,
        )
    for missing in declared_absent:
        not_assessable.append(
            f"{missing}: declared protected but there is no such column in the "
            f"DataFrame (available: {list(df.columns)[:10]}), so it was NOT screened "
            f"for proxy risk and nothing in this result says anything about it"
        )
        warnings.warn(
            f"classify_column_roles: declared protected attribute {missing!r} is NOT a "
            f"column in this DataFrame, so it was not screened at all and the protected "
            f"roles below come from auto-detection instead. An empty proxy_candidates "
            f"list for it means it was never looked at, not that it is clean.",
            UserWarning,
            stacklevel=2,
        )
    try:
        targets = declared_protected or list(prot_cat)
        if not targets:
            # No protected target, so the correlation screen never ran. Said out
            # loud: the name hints below still fire, and a reader seeing a short
            # proxy_candidates list would otherwise credit a correlation pass
            # that did not happen.
            not_assessable.append(
                "correlation proxy screen: not run at all (no declared or detected "
                "protected attribute to correlate against; only name hints applied)"
            )
        # identify_proxy_features takes a SINGLE protected column name and returns
        # one dict per correlated feature (keys: column, correlation, ...), so run
        # it once per protected target and stamp the attribute it correlates with.
        # The map is keyed by the proxy feature column so the role loop below can
        # attach proxy evidence. (Previously a whole list was passed as the single
        # str argument, which always raised and left proxy detection a no-op.)
        for prot in targets:
            if prot not in df.columns:
                continue
            try:
                hits = identify_proxy_features(df, prot)
            except Exception as exc:  # one bad target never fails the typology gate
                warnings.warn(
                    f"typology proxy scan skipped protected target '{prot}' "
                    f"({type(exc).__name__}: {exc}); proxy evidence is incomplete",
                    ProxyScanIncompleteWarning,
                    stacklevel=2,
                )
                not_assessable.append(
                    f"{prot}/proxy scan: not run at all ({type(exc).__name__}: {exc})"
                )
                continue
            # A column the scan could not read is not a column without proxy
            # risk, and `proxy_candidates` alone cannot say which it was.
            not_assessable.extend(getattr(hits, "not_assessable", []))
            for hd in hits:
                f = hd.get("column")
                if f and f not in proxy_hit:
                    proxy_hit[f] = {"protected_attribute": prot, **hd}
    except Exception as exc:  # noqa: BLE001
        proxy_hit = {}
        not_assessable.append(
            f"correlation proxy screen: collapsed entirely ({type(exc).__name__}: {exc}), "
            "so no column was screened by correlation"
        )

    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        uniq_ratio = nunique / n_rows
        role = cat = ""
        conf = 0.5
        reason = ""
        ev: Dict[str, Any] = {}
        toks = set(col.lower().replace("-", "_").split("_"))

        # 1. Oracle / model output by EXPLICIT declaration or tight name.
        # A NAME HEURISTIC OUTRANKED AN EXPLICIT DECLARATION.
        # BGL5 A-evaluation-3, 2026-09-29. This function's own contract says the
        # declarations "override name heuristics", and the oracle test runs first,
        # so a declared PREDICTION column whose name happens to match an oracle
        # pattern was classified as the answer key. Measured on 400 rows holding
        # gender, salary and y_hat, classify_column_roles(df,
        # declared_protected=['gender'], declared_prediction='y_hat'):
        #   before -> y_hat oracle "name matches a ground-truth / oracle pattern",
        #             model_output [], oracle_columns ['y_hat']
        #   after  -> y_hat model_output "user-declared model output"
        # `_name_hit` matches on tokens, and 'y_hat' splits to {'y', 'hat'}, so the
        # one-character oracle pattern 'y' hit it. The empty model_output that came
        # out is the same neutral value the absent-declaration defect above
        # produced, by a different route, and it reaches the same three Pulse
        # consumers; regulatory.py:491 additionally narrates a prediction column to
        # the reader as an "oracle/answer-key column". An explicit declared_target
        # still wins, because it is tested first and a caller who declares one
        # column as both has said something contradictory that the answer key
        # should win. THAT CONTRADICTION IS NOW DISCLOSED rather than resolved in
        # silence: see `prediction_is_target` at the top of this function, which
        # puts it in not_assessable, in mismatches and in a warning, because the
        # empty model_output it leaves behind is otherwise read as a fact about
        # the data.
        if col == declared_target or (
            col != declared_prediction and _name_hit(col, _ORACLE_PATTERNS)
        ):
            role, conf = "oracle", (0.99 if col == declared_target else 0.7)
            reason = (
                "user-declared target / oracle"
                if col == declared_target
                else "name matches a ground-truth / oracle pattern"
            )
        if not role and (
            col == declared_prediction
            or _name_hit(col, _MODEL_OUTPUT_PATTERNS)
            or (toks & _MODEL_OUTPUT_TOKENS)
        ):
            role, conf = "model_output", (0.99 if col == declared_prediction else 0.75)
            reason = (
                "user-declared model output"
                if col == declared_prediction
                else "name matches a model-output pattern"
            )
        # 2. Identity / PII -- outranks protected/proxy ("must not model").
        if not role:
            pii = _name_hit(col, _PII_PATTERNS)
            is_idish = _name_hit(col, _ID_PATTERNS) and uniq_ratio > 0.9
            if pii or is_idish:
                role, conf = "identity_pii", 0.85 if pii else 0.7
                reason = (
                    f"name matches PII pattern '{pii}'" if pii else "near-unique identifier column"
                )
                ev["unique_ratio"] = round(uniq_ratio, 3)
        # 3. Protected: the user's declaration is AUTHORITATIVE. Only fall
        #    back to auto-detection when nothing was declared. Auto-detected
        #    but NOT declared columns are treated as proxy candidates and
        #    surfaced as a mismatch -- never silently reclassified protected.
        if not role and col in decl_prot:
            role, conf, cat = "protected", 0.95, prot_cat.get(col, (0.0, "user-declared"))[1]
            reason = "user-declared protected attribute"
        if not role and not has_declared and col in prot_cat:
            conf, cat = prot_cat[col]
            role, reason = "protected", f"auto-detected protected ({cat})"
        # 4. Proxy candidate: correlated with a protected attr, or a known
        #    proxy-shaped name (zip, university, tenure gap...).
        if not role:
            hint = _name_hit(col, _PROXY_NAME_HINTS)
            if col in proxy_hit:
                role, conf = "proxy_candidate", 0.65
                reason = (
                    "correlates with protected attribute "
                    f"{proxy_hit[col].get('protected_attribute', '')}"
                )
                ev["correlation"] = proxy_hit[col].get("correlation")
            elif has_declared and col in prot_cat and col not in decl_prot:
                role, conf, cat = "proxy_candidate", 0.6, prot_cat[col][1]
                reason = (
                    f"looks like a protected attribute ({cat}) but was "
                    "not declared, so it is treated as a proxy risk"
                )
            elif hint:
                role, conf = "proxy_candidate", 0.55
                reason = f"name '{hint}' is a common proxy for a protected attribute"
        # 5. Fallbacks.
        if not role:
            if nunique <= 1:
                role, conf, reason = "unknown", 0.4, "constant / empty column"
            else:
                role, conf, reason = (
                    "job_relevant",
                    0.5,
                    "no PII / protected / output signal, so it is treated as a legitimate predictor",
                )

        roles.append(
            ColumnRole(
                column=col, role=role, confidence=conf, reason=reason, category=cat, evidence=ev
            )
        )
        buckets[role].append(col)

    # Declared-vs-inferred mismatches (verify, don't just trust the upload).
    mismatches: List[Dict[str, Any]] = []
    # A declaration that names nothing in the frame is the sharpest mismatch there
    # is: what the caller declared and what the data holds cannot be reconciled at
    # all. It used to be dropped by the filter at the top of this function without
    # a trace; see the measured before and after there.
    if prediction_absent:
        mismatches.append(
            {
                "column": declared_prediction,
                "declared": "model_output",
                "inferred": "absent",
                "detail": (
                    f"'{declared_prediction}' was declared as the model output but there "
                    f"is no such column in the dataset, so model_output is empty: the "
                    f"declaration could not be honoured. Every consumer of model_output "
                    f"(leakage exclusion, the governance narrative, the causal decision "
                    f"node) is reading that empty list as a fact about the data. Fix the "
                    f"name or remove the declaration."
                ),
            }
        )
    if prediction_is_target:
        mismatches.append(
            {
                "column": declared_prediction,
                "declared": "model_output",
                "inferred": "oracle",
                "detail": (
                    f"'{declared_prediction}' was declared as BOTH the target and the "
                    f"model output. One column cannot be both the answer key and the "
                    f"model's decision, so it is classified as the oracle and "
                    f"model_output is empty BECAUSE THE TWO DECLARATIONS CONTRADICT EACH "
                    f"OTHER, not because the data holds no model output. Declare the "
                    f"prediction column separately, or drop one of the two declarations."
                ),
            }
        )
    for col in declared_absent:
        mismatches.append(
            {
                "column": col,
                "declared": "protected",
                "inferred": "absent",
                "detail": (
                    f"'{col}' was declared protected but there is no such column in "
                    f"the dataset, so it was NOT assessed: not screened for proxy "
                    f"risk, not classified, and absent from every list below. Fix the "
                    f"name or remove the declaration; an empty finding for it is not a "
                    f"clean result."
                ),
            }
        )
    for col in declared_protected:
        inf = next((r for r in roles if r.column == col), None)
        if inf and inf.role == "identity_pii":
            mismatches.append(
                {
                    "column": col,
                    "declared": "protected",
                    "inferred": "identity_pii",
                    "detail": (
                        f"'{col}' was marked protected but looks like raw "
                        "PII. Use a derived category, not the identifier."
                    ),
                }
            )
    for col, (cf, ct) in prot_cat.items():
        if col not in decl_prot and cf >= 0.6:
            r = next((x for x in roles if x.column == col), None)
            if r and r.role in ("job_relevant", "proxy_candidate"):
                mismatches.append(
                    {
                        "column": col,
                        "declared": r.role,
                        "inferred": "protected",
                        "detail": (
                            f"'{col}' was not declared protected but looks "
                            f"like a protected attribute ({ct}, "
                            f"confidence {cf:.0%}). Confirm before modelling."
                        ),
                    }
                )

    refuse = False
    refuse_reason = ""
    # Refuse-fast only on an unambiguous, unrecoverable error: a user
    # explicitly declared a protected/target column that does not exist, or
    # every column is identity/oracle (nothing legitimate to assess).
    if declared_target and declared_target not in df.columns:
        refuse = True
        refuse_reason = f"Declared target '{declared_target}' is not a column in the dataset."
    elif not buckets["job_relevant"] and not buckets["protected"]:
        refuse = True
        refuse_reason = (
            "No legitimate (job-relevant or protected) column "
            "found: the dataset is all identifiers / answer "
            "keys. Nothing can be fairly assessed."
        )

    return {
        "roles": [r.to_dict() for r in roles],
        "pii_leakage": buckets["identity_pii"],
        "oracle_columns": buckets["oracle"],
        "model_output": buckets["model_output"],
        "protected": buckets["protected"],
        "proxy_candidates": buckets["proxy_candidate"],
        "job_relevant": buckets["job_relevant"],
        "unknown": buckets["unknown"],
        "mismatches": mismatches,
        # Always present, never only when something was unmeasurable: a field
        # that appeared only on failure would make its absence ambiguous between
        # "nothing was missed" and "this version did not record it".
        "not_assessable": not_assessable,
        "n_not_assessable": len(not_assessable),
        "refuse": refuse,
        "refuse_reason": refuse_reason,
    }
