"""Canonical preparation of protected attributes for fairness analysis.

Single source of truth. Every Pulse / Navigator / CI path that needs to group
by a protected attribute MUST call :func:`prepare_protected_attributes`
instead of binning ad-hoc. Divergent local binning was the cause of repeated
incorrect output (date-of-birth treated as a group, "age" matching
"primary_language", per-person micro-groups).

Design, grounded in practice:

* **Identifiers are never a fairness group.** Date of birth, names, emails,
  record IDs, etc. are PII identifiers (HIPAA Safe Harbor: dates more precise
  than year, and any near-unique field, are identifiers). Grouping by them
  yields one tiny group per person -- statistically meaningless and a
  re-identification risk. They are *excluded with a stated reason*.
* **Age, not date of birth.** When a DOB column is given we DERIVE age and
  band it; the raw date is dropped. If age cannot be derived the column is
  excluded (not grouped raw).
* **Canonical age bands** match the library's existing
  ``representation.py`` benchmark keys so representation-bias benchmarking
  stays consistent: ``under_18, 18-24, 25-34, 35-44, 45-54, 55-64, 65+``
  (the ADEA 40+ protected boundary sits inside 35-44 / 45-54).
* **Continuous numerics** -> equal-frequency (quantile) bands, the
  defensible neutral default for skewed data.
* **Prefer an existing binned sibling** (``age`` + ``age_group`` -> use
  ``age_group``). Match is *whole-token*, never substring.
* Every transformation is **disclosed** -- never silent.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from vfairness._names import name_token_list

# At/above this many distinct values a column is too granular to be a group
# key directly (needs binning, a binned sibling, or exclusion).
_MAX_GROUP_CARDINALITY = 12
# Fraction of distinct values above which a column is identifier-grade.
_IDENTIFIER_UNIQUE_FRACTION = 0.5

_AGE_BAND_EDGES = [-np.inf, 18, 25, 35, 45, 55, 65, np.inf]
_AGE_BAND_LABELS = ["under_18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
# The outer band edges are -inf / +inf, so NO age is ever "out of range" and an
# impossible value cannot fall outside the bands: it joins a real one. The
# plausibility window is therefore the only thing standing between a sentinel
# and a protected group. Shared by `_derive_age` (which always had it) and the
# raw-`age` branch (which did not).
_MAX_PLAUSIBLE_AGE = 120

# The label every binning branch gives a row it could not place. It is the
# ABSENCE of a value, never a protected group, and it is disclosed as such by
# the pass at the end of prepare_protected_attributes().
_ABSENT_LEVEL = "missing"

# Geographic codes are PROXIES (redlining etc.), not magnitudes. They must
# never be quantile-binned -- the numeric order of a ZIP is meaningless and
# banding it destroys the very locality signal that carries the bias.
_GEO_TOKENS = {
    "zip",
    "zipcode",
    "zcta",
    "postal",
    "postcode",
    "plz",
    "fips",
    "tract",
    "censustract",
    "blockgroup",
    "county",
    "district",
    "region",
    "area",
    "areacode",
    "neighbourhood",
    "neighborhood",
    "borough",
    "ward",
    "precinct",
    "municipality",
    "city",
    "town",
    "location",
    "geo",
    "geoid",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "lng",
}

_STOP_TOKENS = {"of", "the", "a", "an", "is", "by", "per"}
_BIN_WORD_TOKENS = {
    "group",
    "groups",
    "band",
    "bands",
    "bracket",
    "brackets",
    "bin",
    "bins",
    "bucket",
    "buckets",
    "range",
    "ranges",
    "cat",
    "category",
    "categories",
    "tier",
    "tiers",
    "level",
    "levels",
    "class",
    "classes",
    "decile",
    "deciles",
    "quartile",
    "quartiles",
    "bucketed",
    "binned",
    "grouped",
}
# Protected-characteristic concept words. A column carrying one of these is
# about a protected GROUP, so it is never excluded merely because an
# identifier word ("customer", "user", "account") sits next to it.
_PROTECTED_CONCEPT_TOKENS = {
    "age",
    "aged",
    "alter",
    "edad",
    "idade",
    "birth",
    "birthyear",
    "dob",
    "yob",
    "gender",
    "genero",
    "geschlecht",
    "sex",
    "sexe",
    "sexo",
    "race",
    "racial",
    "raza",
    "raca",
    "ethnicity",
    "ethnic",
    "nationality",
    "citizenship",
    "religion",
    "religious",
    "faith",
    "disability",
    "disabled",
    "handicap",
    "marital",
    "orientation",
    "pregnancy",
    "veteran",
}

# Token sets that mark a column as a personal identifier (never a group).
_IDENTIFIER_TOKENS = {
    "id",
    "uuid",
    "guid",
    "ssn",
    "sin",
    "nino",
    "passport",
    "email",
    "mail",
    "phone",
    "mobile",
    "tel",
    "msisdn",
    "name",
    "firstname",
    "lastname",
    "surname",
    "givenname",
    "fullname",
    "username",
    "account",
    "customer",
    "user",
    "record",
    "row",
    "index",
    "address",
    "street",
    "ip",
    "mac",
}


def _tokens(name: str) -> List[str]:
    """snake/kebab/space/camelCase -> lowercase word tokens.

    Delegates to :func:`vfairness._names.name_token_list`, the single source
    of truth shared with the proxy / correlation / discovery / statistical
    detectors. This module kept its own PRE-fix ``[A-Za-z][a-z]*`` regex long
    after that collision was fixed elsewhere, and that regex splits an
    ALL-CAPS name into single LETTERS. Measured 2026-09-10:
    ``_tokens("AGE") == ['a', 'g', 'e']``, ``_tokens("DOB") == ['d','o','b']``,
    ``_tokens("BIRTH_DATE") == ['b','i','r','t','h','d','a','t','e']``.

    The consequences all landed in this file: ``"age" in _tokens(col)`` was
    False for ``AGE``, so identical age data was quantile-banded under one
    casing and put in the canonical ADEA-aligned bands under another, in the
    SAME frame -- destroying exactly the cross-run comparability those
    standard bands exist for. ``_is_dob`` was False for ``DOB`` and
    ``BIRTH_DATE``, so a date of birth fell through to the near-unique branch
    and was excluded as "98% distinct values, effectively an identifier" --
    a confident explanation of the wrong thing, for a column an age was
    perfectly derivable from.
    """
    return name_token_list(name)


def _is_dob(name: str) -> bool:
    t = set(_tokens(name))
    return (
        "dob" in t or "birthdate" in t or ("birth" in t and bool({"date", "dt", "day", "year"} & t))
    )


def _looks_dateish(name: str, series: pd.Series) -> bool:
    t = set(_tokens(name))
    if _is_dob(name) or {"date", "datetime", "timestamp"} & t:
        return True
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    # The dotted European form (31.05.1974) was missing, so a DOB column
    # whose NAME did not say "birth" was never recognised as a date at all.
    # A 4-digit year is required so a version string ("1.2.34") is not read
    # as a date.
    hit = sum(
        bool(
            re.search(
                r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{4}",
                v,
            )
        )
        for v in sample
    )
    return hit >= max(3, len(sample) // 2)


def _is_identifier_name(name: str) -> bool:
    t = set(_tokens(name))
    if t & _IDENTIFIER_TOKENS:
        # "group name" / "customer age" edge: an identifier word next to a
        # bin word or a PROTECTED CONCEPT is not an identifier. The comment
        # here always claimed the protected-concept half; only the bin-word
        # half was implemented, and with correct tokenisation that gap
        # bites: ``customerAge`` yields BOTH 'customer' (an identifier
        # token) and 'age', so an age column was excluded from fairness
        # analysis as "a personal identifier" -- measured 2026-09-10,
        # _is_identifier_name("customerAge") was True.
        return not (t & (_BIN_WORD_TOKENS | _PROTECTED_CONCEPT_TOKENS))
    return False


def _find_binned_sibling(col: str, df: pd.DataFrame) -> Optional[str]:
    """Whole-token sibling match: age -> age_group (NOT age -> language)."""
    ct = set(_tokens(col))
    if _is_dob(col) or "age" in ct:
        concept = {"age"}
    else:
        concept = {t for t in ct if t not in _STOP_TOKENS}
    if not concept:
        return None
    for other in df.columns:
        if other == col:
            continue
        ot = set(_tokens(other))
        if (
            (concept & ot)
            and (ot & _BIN_WORD_TOKENS)
            and df[other].nunique(dropna=True) <= _MAX_GROUP_CARDINALITY
        ):
            return other
    return None


def _derive_age(series: pd.Series) -> Optional[pd.Series]:
    """Years of age from a DOB column, or None if it will not parse.

    Parsed twice, month-first and day-first, keeping whichever resolves more
    rows. pandas defaults to month-first, which silently coerces every
    European day-first date with a day above 12 to NaT: measured 2026-09-10
    on 200 dotted dates ('11.05.1974', '25.10.1975', ...), the default parse
    resolved 42.5% and fell under the 50% floor, so the column was reported
    as "looks like a date of birth but an age could not be derived from it"
    and dropped. day-first resolved 100% of the same column. An
    unambiguously month-first column parses identically either way, so the
    better-of-two rule cannot flip a US-format column.
    """
    with warnings.catch_warnings():
        # pandas warns about the format it guessed; the retry below is this
        # function's answer to that guess, so the warning is noise here.
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        if dt.notna().mean() < 1.0:
            alt = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=True)
            if alt.notna().mean() > dt.notna().mean():
                dt = alt
    if dt.notna().mean() < 0.5:
        return None
    now = pd.Timestamp.now(tz="UTC")
    age = (now - dt).dt.days / 365.25
    age = age.where((age >= 0) & (age <= _MAX_PLAUSIBLE_AGE))
    if age.notna().mean() < 0.5:
        return None
    # Age is COMPLETED years: you are 17 until your 18th birthday. This used
    # to be `age.round()`, which rounds the last half-year UP, so everyone
    # within ~6 months of a band edge was banded ONE BAND TOO OLD. Measured
    # 2026-09-30: a date of birth 17.6 years ago derived 18.0 and was banded
    # "18-24" -- a MINOR published in an adult band -- and 64.6 years derived
    # 65.0 and was banded "65+". `under_18` and `65+` are the two legally
    # loaded edges of this scale (minors, ADEA/Medicare) and they are exactly
    # the two the rounding crossed, in whichever direction the cohort sat.
    return np.floor(age)


def _age_bands(age: pd.Series) -> pd.Series:
    return (
        pd.cut(age, bins=_AGE_BAND_EDGES, labels=_AGE_BAND_LABELS, right=False)
        .astype("string")
        .fillna(_ABSENT_LEVEL)
    )


def _age_years(s: pd.Series) -> Optional[pd.Series]:
    """A raw ``age`` column as plausible years, or None if it is not ages.

    Two separate defects lived at this gate, both measured 2026-09-30.

    *Text is not a different scale.* The gate was
    ``pd.api.types.is_numeric_dtype(s)``, and an ordinary ``read_csv`` of a
    file with a single blank or "n/a" in the age column yields dtype object.
    The SAME 150 ages therefore went into the canonical ADEA-aligned bands as
    int64 and into ``(19.999, 29.8]``-style quantile bands as strings -- in
    one frame at once, measured as usable ``['age', 'age_years_txt']`` with
    two incompatible partitions, and the quantile partition has no ``under_18``
    edge and no 40+ edge at all. That is precisely the cross-run
    incomparability the canonical bands exist to prevent, and the module
    docstring already records another door to it (ALL-CAPS tokenisation).

    *An impossible age is not an age.* There was no plausibility window here,
    while ``_derive_age`` has always had one. The outer band edges are
    infinite, so an out-of-range value cannot land OUTSIDE the bands; it joins
    a real one. Measured on 100 rows carrying the sentinels -999 and 999:
    thirty rows published as ``under_18`` and thirty as ``65+``, with
    ``excluded == []`` and not one note. Implausible rows are returned as NaN
    here so the caller can count and disclose them.
    """
    if not (
        pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        # datetime / timedelta / categorical: to_numeric would hand back
        # nanoseconds and band them as ages. A date column is branch 1's job.
        return None
    years = pd.to_numeric(s, errors="coerce")
    if years.notna().mean() < 0.5:
        # Not ages (an "age" column holding words, e.g. young/middle/old).
        return None
    return years


def _all_blank(s: pd.Series) -> bool:
    """True when the column carries no observation at all.

    The blank string is the sixth door of absence and the only one
    ``nunique(dropna=True)`` counts as a value: 100% ``''`` was ``nunique == 1``
    and came back usable as a single group NAMED the empty string, while the
    same column of ``None`` / ``nan`` / ``pd.NA`` / ``pd.NaT`` was excluded as
    holding no values at all. Same absence, two answers.
    """
    obs = s.dropna()
    if obs.empty:
        return True
    try:
        return bool(obs.astype("string").str.strip().eq("").all())
    except (TypeError, ValueError):  # non-string dtype: not blank
        return False


def _quantile_bands(series: pd.Series, q: int = 5) -> Optional[pd.Series]:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() < 0.5:
        return None
    try:
        binned = pd.qcut(num, q=q, duplicates="drop")
    except (ValueError, IndexError):
        try:
            binned = pd.cut(num, bins=min(q, 4), duplicates="drop")
        except (ValueError, IndexError):
            return None
    return binned.astype("string").fillna(_ABSENT_LEVEL)


def _is_geographic(col: str) -> bool:
    return bool(set(_tokens(col)) & _GEO_TOKENS)


def _zip_like(col: str, s: pd.Series) -> bool:
    """A US-ZIP-style column: name says zip/postal AND values are mostly
    4-5 char numeric codes."""
    if not (set(_tokens(col)) & {"zip", "zipcode", "zcta", "postal", "postcode", "plz"}):
        return False
    sample = s.dropna().astype("string").str.strip().head(500)
    if sample.empty:
        return False
    digits = sample.str.replace(r"\D", "", regex=True)
    return bool((digits.str.len().between(4, 5)).mean() >= 0.7)


def _zip3(s: pd.Series) -> pd.Series:
    """Aggregate ZIP to its 3-digit prefix (Sectional Center Facility) --
    the standard geographic roll-up that PRESERVES locality / redlining
    structure, unlike a meaningless numeric quantile band."""
    d = s.astype("string").str.replace(r"\D", "", regex=True).str.zfill(5)
    pref = d.str.slice(0, 3)
    return ("ZIP " + pref + "xx").where(pref.str.len() == 3).fillna(_ABSENT_LEVEL)


@dataclass
class PreparedProtected:
    """Result of :func:`prepare_protected_attributes`."""

    frame: pd.DataFrame
    usable: List[str]
    excluded: List[Tuple[str, str]] = field(default_factory=list)  # (col, reason)
    notes: List[Tuple[str, str]] = field(default_factory=list)  # (label, detail)


def prepare_protected_attributes(
    df: pd.DataFrame,
    protected: List[str],
    jurisdiction: Optional[str] = None,
) -> PreparedProtected:
    """Make every protected attribute a valid, disclosed fairness group key.

    Returns usable attribute names (post-binning / sibling substitution),
    a list of excluded columns each with a plain-language reason, and the
    binning disclosures. The returned ``frame`` has binned columns replaced
    in place so downstream group-bys are correct everywhere.

    ``jurisdiction`` is accepted ONLY as ``None``, and any other value is
    REFUSED. The binning here is deliberately jurisdiction-neutral: it is a
    statistical operation (identifiers excluded, dates of birth turned into
    canonical age bands, geographic codes rolled up, continuous numerics
    quantile-banded) and it runs identically everywhere. Which attributes a
    jurisdiction protects, restricts or forbids is a legal question answered
    by :func:`vfairness.legal.admissibility.classify_columns`, which needs a
    use case as well as a jurisdiction and is a separate step. Accepting a
    jurisdiction here and ignoring it produced output that LOOKED
    jurisdiction-aware and was not.

    Raises:
        ValueError: if ``jurisdiction`` is anything other than ``None``.
    """
    if jurisdiction is not None:
        raise ValueError(
            "prepare_protected_attributes() does not implement jurisdiction-specific "
            f"binning; it was given jurisdiction={jurisdiction!r}. What IS supported is "
            "jurisdiction-neutral preparation: identifiers excluded, date of birth -> "
            f"age bands ({', '.join(_AGE_BAND_LABELS)}), geographic codes rolled up "
            "(ZIP -> 3-digit prefix), continuous numerics quantile-banded, applied the "
            "same way in every jurisdiction. For the legal question of which attributes "
            "a jurisdiction protects or restricts, call "
            "vfairness.legal.admissibility.classify_columns(columns, use_case, "
            "jurisdiction). Pass jurisdiction=None here."
        )

    out = df.copy()
    usable: List[str] = []
    excluded: List[Tuple[str, str]] = []
    notes: List[Tuple[str, str]] = []
    # Columns that were turned into age bands (raw `age` OR age derived
    # from a date of birth). Tracked so the SAME fairness axis is never
    # counted twice even when the two columns are not byte-identical.
    age_axis_cols: set = set()
    # Columns THIS function binned. Every binning branch ends in
    # `.fillna(_ABSENT_LEVEL)`, so only for these columns is the level
    # "missing" a label we minted out of absence rather than a value the
    # data actually carried. Read by the disclosure pass at the end.
    binned_cols: set = set()
    n = max(1, len(out))

    for col in protected:
        if col not in out.columns:
            excluded.append((col, f'"{col}" is not present in the uploaded data.'))
            continue

        s = out[col]
        nunique = int(s.nunique(dropna=True))
        uniq_frac = nunique / n

        # 1. Date-of-birth / date-like -> derive age and band it.
        if _is_dob(col) or _looks_dateish(col, s):
            age = _derive_age(s)
            if age is not None:
                out[col] = _age_bands(age)
                binned_cols.add(col)
                notes.append(
                    (
                        "Binning",
                        f'"{col}" is a date (an identifier, never a fairness group). '
                        "Age was derived from it and grouped into standard bands "
                        f"({', '.join(_AGE_BAND_LABELS)}).",
                    )
                )
                usable.append(col)
                age_axis_cols.add(col)
            else:
                excluded.append(
                    (
                        col,
                        f'"{col}" looks like a date of birth but an age could not '
                        "be derived from it. Raw dates are identifiers, not a "
                        "fairness group, so it was excluded.",
                    )
                )
            continue

        # 2. Raw "age" -> canonical age bands (consistency with the
        #    representation-bias benchmark keys). Numeric OR numeric-as-text:
        #    see _age_years for why the dtype gate was the defect.
        if "age" in set(_tokens(col)) and nunique > _MAX_GROUP_CARDINALITY:
            years = _age_years(s)
            if years is not None:
                implausible = int((years.notna() & ~years.between(0, _MAX_PLAUSIBLE_AGE)).sum())
                if implausible:
                    years = years.where(years.between(0, _MAX_PLAUSIBLE_AGE))
                out[col] = _age_bands(years)
                binned_cols.add(col)
                notes.append(
                    (
                        "Binning",
                        f'"{col}" was grouped into standard age bands '
                        f"({', '.join(_AGE_BAND_LABELS)}) so groups are large enough "
                        "to assess.",
                    )
                )
                if implausible:
                    notes.append(
                        (
                            "Coverage",
                            f'{implausible} of {n} rows of "{col}" hold a value '
                            f"outside 0-{_MAX_PLAUSIBLE_AGE} years (a sentinel such as "
                            "-999 or 999, or a typo). An impossible age is not an age, "
                            "so those rows were treated as having no value rather than "
                            "banded: left unfiltered they join a real band and are "
                            "published as that protected group.",
                        )
                    )
                usable.append(col)
                age_axis_cols.add(col)
                continue

        # 3. Personal identifier by name -> never a fairness group.
        if _is_identifier_name(col):
            excluded.append(
                (
                    col,
                    f'"{col}" is a personal identifier, not a group you can '
                    "assess fairness across. Excluded.",
                )
            )
            continue

        # 4. Low-cardinality categorical / numeric -> use as-is.
        if nunique <= _MAX_GROUP_CARDINALITY:
            # BGL3 2026-09-27. `nunique(dropna=True)` is 0 for a column with no
            # observation at all, and `0 <= 12` is true, so an ENTIRELY NULL
            # protected column came back in `usable` with excluded=[] and
            # notes=[]: measured on 100 rows of gender=None, usable was
            # ['gender'] and nothing anywhere said the column held no data. It
            # yields no group to compare, which is the opposite of the "valid,
            # disclosed fairness group key" this function promises, and every
            # downstream refusal then looks like a surprise rather than a
            # consequence. Zero groups is not low cardinality.
            # G13 2026-09-30: `_all_blank` covers the SIXTH door. The other
            # five doors of absence (None, nan, pd.NA, pd.NaT and a literal
            # empty column) are all dropped by `nunique(dropna=True)`, but a
            # blank string is a VALUE to pandas, so 50 rows of "" was
            # `nunique == 1` and came back usable as a single group named the
            # empty string while the identical column of None was excluded.
            if nunique < 1 or _all_blank(s):
                excluded.append(
                    (
                        col,
                        f'"{col}" has no values at all (every row is empty or '
                        "blank), so it "
                        "yields no group to compare fairness across. Excluded.",
                    )
                )
                continue
            # One level is one group, and a fairness comparison needs two. The
            # disclosure for that is NOT here any more: it is below the whole
            # dispatch, over the POST-binning column, because this branch is one
            # of six that reach usable.append(col) and the other five can
            # manufacture a single group out of a multi-level column. See the
            # "single group" pass after the loop.
            usable.append(col)
            continue

        # 4b. Geographic code (zip, postal, tract, county, ...). NEVER
        #     quantile-bin -- the numeric order is meaningless and banding
        #     it destroys the locality signal (zip is a redlining proxy).
        #     ZIP -> 3-digit prefix (preserves locality); other geo codes
        #     -> keep the most common areas, fold the long tail to "other".
        if _is_geographic(col):
            if _zip_like(col, s):
                z = _zip3(s)
                if int(z.nunique(dropna=True)) > _MAX_GROUP_CARDINALITY:
                    top = z.value_counts().head(_MAX_GROUP_CARDINALITY).index
                    z = z.where(z.isin(top), other="other")
                out[col] = z
                binned_cols.add(col)
                notes.append(
                    (
                        "Binning",
                        f'"{col}" is a ZIP code, a geographic proxy (e.g. for '
                        "redlining), not a number to band. It was rolled up to "
                        "its 3-digit prefix (a real geographic unit) so locality "
                        "is preserved, never quantile-binned.",
                    )
                )
            else:
                ss = s.astype("string")
                top = ss.value_counts().head(_MAX_GROUP_CARDINALITY).index
                out[col] = ss.where(ss.isin(top), other="other").fillna(_ABSENT_LEVEL)
                binned_cols.add(col)
                notes.append(
                    (
                        "Binning",
                        f'"{col}" is a geographic attribute (a proxy for protected '
                        "characteristics), kept categorical: the "
                        f"{len(top)} most common areas are retained and the rest "
                        'folded into "other". It was not quantile-binned.',
                    )
                )
            usable.append(col)
            continue

        # 5. Too granular. Prefer an existing binned sibling.
        sibling = _find_binned_sibling(col, out)
        if sibling is not None:
            notes.append(
                (
                    "Binning",
                    f'"{col}" is too granular to compare directly; using the '
                    f'already-binned "{sibling}" column instead.',
                )
            )
            if sibling not in usable:
                usable.append(sibling)
            continue

        # 6. Continuous numeric -> equal-frequency bands.
        qb = _quantile_bands(s)
        if qb is not None:
            out[col] = qb
            binned_cols.add(col)
            notes.append(
                (
                    "Binning",
                    f'"{col}" had too many distinct values; grouped into 5 '
                    "equal-size (quantile) bands for analysis.",
                )
            )
            usable.append(col)
            continue

        # 7. Near-unique non-numeric with no sibling -> identifier-grade.
        if uniq_frac >= _IDENTIFIER_UNIQUE_FRACTION:
            excluded.append(
                (
                    col,
                    f'"{col}" has {uniq_frac:.0%} distinct values, effectively '
                    "an identifier. Grouping by it would make one tiny group per "
                    "row, so it was excluded.",
                )
            )
            continue

        # 8. Moderately high-cardinality categorical: keep the common levels,
        #    fold the long tail into "other".
        top = s.astype("string").value_counts().head(_MAX_GROUP_CARDINALITY).index
        out[col] = (
            s.astype("string")
            .where(s.astype("string").isin(top), other="other")
            .fillna(_ABSENT_LEVEL)
        )
        binned_cols.add(col)
        notes.append(
            (
                "Binning",
                f'"{col}" had many categories; kept the {len(top)} most common '
                'and folded the rest into "other".',
            )
        )
        usable.append(col)

    # de-dup by NAME, preserve order
    seen: set = set()
    deduped: List[str] = []
    for a in usable:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    usable = deduped

    # de-dup by VALUE: a date-of-birth column binned to age bands and a raw
    # `age` column binned to the same bands are the SAME fairness axis with
    # identical per-row values. Keeping both double-counted age in every
    # aggregate (per-variable, findings/areas/taxonomy/recommendations),
    # manufactured a spurious DOB<->age "proxy", and doubled the
    # intersectional dimension/label ("35-44 · 35-44"). Collapse columns
    # whose post-binning values are identical to ONE canonical column,
    # preferring the non-DOB / shorter name, and disclose the drop.
    def _canon_rank(name: str) -> Tuple[int, int]:
        return (1 if _is_dob(name) else 0, len(name))

    sig_owner: dict = {}
    value_dropped: set = set()
    for a in sorted(usable, key=_canon_rank):
        try:
            col = out[a].astype("string").fillna("\x00")
            sig = pd.util.hash_pandas_object(col, index=False).values.tobytes()
        except Exception:  # noqa: BLE001 -- never fatal; keep the column
            continue
        if sig in sig_owner:
            kept = sig_owner[sig]
            value_dropped.add(a)
            excluded.append(
                (
                    a,
                    f'"{a}" produces the exact same groups as "{kept}" after '
                    "binning (a date of birth and an age are the same fairness "
                    f'axis). Kept "{kept}" and excluded "{a}" so age is not '
                    "counted twice across the analysis.",
                )
            )
        else:
            sig_owner[sig] = a
    usable = [a for a in usable if a not in value_dropped]

    # Semantic age-axis collapse. The value-identity check above only
    # catches BYTE-identical columns. A recorded `age` (age at record
    # time) and an age DERIVED from date_of_birth (age as of today)
    # describe the SAME fairness axis but differ by the elapsed years,
    # so their post-binning hashes are NOT identical and the value
    # de-dup misses them -- leaving age double-counted (spurious
    # DOB<->age "proxy", doubled "35-44 · 35-44" intersections, every
    # age aggregate counted twice). This is the exact failure observed
    # on the recruitment dataset. Collapse every surviving age-band
    # column to ONE canonical column (prefer non-DOB / shorter name),
    # regardless of value identity, and disclose the drop.
    age_axis = [a for a in usable if a in age_axis_cols]
    if len(age_axis) > 1:
        keep = sorted(age_axis, key=_canon_rank)[0]
        for a in age_axis:
            if a == keep:
                continue
            excluded.append(
                (
                    a,
                    f'"{a}" and "{keep}" are the same fairness axis: both were '
                    "grouped into the same standard age bands (a date of birth "
                    f'and an age describe the same groups). Kept "{keep}" and '
                    f'excluded "{a}" so age is not counted twice across the '
                    "analysis.",
                )
            )
        usable = [a for a in usable if a not in age_axis or a == keep]

    # THE SINGLE-GROUP DISCLOSURE BELONGS ABOVE THE BINNING DISPATCH.
    # BGL5 2026-09-27. The nunique==1 Coverage note used to sit INSIDE branch 4
    # (low-cardinality, used as-is), so it could only ever describe a column that
    # arrived with one level. Branches 1, 2, 4b, 6 and 8 all reach
    # usable.append(col) without passing it, and each of them BINS, which is
    # exactly how a single group gets manufactured. Measured before, on 400 rows
    # over 60 distinct ZIP codes all inside the 100xx area: usable ['zipcode'],
    # notes [('Binning', ...)] only, and the post-binning column had nunique 1 with
    # the single level 'ZIP 100xx', returned as a valid fairness group key with
    # nothing saying no between-group comparison exists. Same on 300 rows of
    # numeric age holding 20 distinct values that all fall inside 25-34: usable
    # ['age'], Binning note only, post-binning levels ['25-34']. Both now carry the
    # Coverage note below as well as their Binning note. Read from `out`, after
    # every branch and every de-dup, so there is one place it can be missed from.
    for col in list(usable):
        try:
            final = out[col]
        except KeyError:  # a binned sibling substitution can rename the axis
            continue

        # THE "missing" LABEL IS A MINTED GROUP, AND IT WAS NEVER DISCLOSED.
        # G13 2026-09-30. Every binning branch ends in `.fillna(_ABSENT_LEVEL)`,
        # so a row the binning could not place is published under the level
        # "missing" -- a group manufactured out of the ABSENCE of a value, which
        # nobody chose and no protected characteristic names. Measured before:
        # 115 rows of `age` with 55 empty came back usable with ONE note saying
        # the bands are the seven canonical ones, while 48% of the frame sat in
        # an eighth level called "missing"; 90 ZIP rows with 30 empty likewise,
        # and 90 income rows with 30 empty. Any downstream gap between "missing"
        # and a real band measures who is UNRECORDED, and it was being reported
        # as an age / geography / income disparity.
        #
        # It is not dropped: silently deleting a third of the rows is the worse
        # answer, and the level being visible is what lets a reader see it. It
        # is disclosed, and it does NOT count as a group in the single-group
        # check below -- otherwise adding empty rows DEFEATS that check, which
        # is how 60 ZIPs from one area plus 30 empty rows lost the
        # "single group" note the BGL5 pass above exists to emit.
        real = final.dropna()
        if col in binned_cols:
            minted = int((final.astype("string") == _ABSENT_LEVEL).sum())
            if minted:
                notes.append(
                    (
                        "Coverage",
                        f'{minted} of {n} rows ({minted / n:.0%}) of "{col}" held no '
                        "value the binning could place, so they carry the level "
                        f'"{_ABSENT_LEVEL}". That is the absence of a value, not a '
                        "protected group: any gap involving it measures who is "
                        "unrecorded, not a disparity between real groups.",
                    )
                )
            real = real[real.astype("string") != _ABSENT_LEVEL]

        post_nunique = int(real.nunique())
        if post_nunique == 1:
            notes.append(
                (
                    "Coverage",
                    f'"{col}" has only one distinct value '
                    f"({real.iloc[0]!r}), so it forms a single group. No "
                    "between-group fairness comparison exists for it; any "
                    "per-group result is a description of the whole dataset.",
                )
            )
        elif post_nunique < 1:
            # Zero groups after binning: nothing survived to compare. Branch 4
            # catches this on the RAW column; a binning that empties a column
            # (every value out of range, every value unparseable) reaches here
            # instead, and a usable key with no group at all is the defect this
            # whole pass exists for.
            excluded.append(
                (
                    col,
                    f'"{col}" holds no value after preparation (every row is '
                    "empty, or held nothing the binning could place), so it "
                    "yields no group to compare fairness across. Excluded.",
                )
            )
            usable = [a for a in usable if a != col]

    return PreparedProtected(frame=out, usable=usable, excluded=excluded, notes=notes)
