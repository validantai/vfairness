"""Canonical column-name tokenisation, shared by every name-based detector.

Single source of truth. Five detectors used to carry their OWN "split a
column name into words" helper, and the fix for one collision reached only
one of them. Measured 2026-09-10, before this module existed:

* ``proxy._name_tokens`` had been fixed for the ``AVERAGE_SPEND`` /``AGE``
  collision, and ``correlation._name_tokens`` was a byte-identical copy of
  the fixed version;
* ``protected_binning._tokens`` was still the PRE-fix ``[A-Za-z][a-z]*``
  regex, so ``"AGE"`` tokenised to ``['a', 'g', 'e']`` and identical age
  data was banded two different ways depending on the column's letter case;
* ``discovery.detect_protected_attributes`` and
  ``statistical._detect_outcome_columns`` used ad-hoc ``keyword in
  col_lower`` substring lists, so ``average_spend``, ``mortgage_balance``
  and ``package_count`` were all reported as protected DEMOGRAPHIC
  attributes (the ``AVERAGE_SPEND``/``AGE`` bug again, one layer up) and
  ``salary``/``city``/``birthday`` were all reported as model "outcomes"
  because ``"y"`` is a substring of each.

Everything name-based now routes through this module, so the next collision
is fixed once rather than in five places.

Two properties the callers rely on, both of which the older helpers broke:

**Whole-token, never substring.** ``age`` must not match ``average`` and
``sex`` must not match ``sussex``. That has to hold for ANY casing. The
first version of the proxy helper split an ALL-CAPS name into single
LETTERS (``[A-Za-z][a-z]*``), so almost any two upper-case names shared a
token and the caller handed out DIRECT, the strongest label in the proxy
enum, to unrelated columns.

**Unicode.** ``re.findall(r"[A-Za-z]+|[0-9]+", ...)`` shatters accented
names: measured 2026-09-10, ``'género' -> ['g', 'nero']``, ``'raça' ->
['a', 'ra']``, ``'âge' -> ['ge']``, ``'Größe' -> ['e', 'gr']``. Both
directions of that were live: ``('peso_g', 'género')`` was classified
DIRECT (weight-in-grams "directly encodes" gender, because both produced a
``'g'`` token) while ``('genero_cliente', 'género')``, the real direct
encoding, came back UNCLASSIFIED. Tokens are therefore taken with a Unicode
word class and accent-folded through NFKD, so ``género`` and ``genero``
unify.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Set

__all__ = [
    "PROTECTED_ATTR_SYNONYMS",
    "name_token_list",
    "name_tokens",
    "resolve_attribute_key",
    "split_case",
    "tokens_contain",
]


# [^\W\d_] is "a Unicode word character that is neither a digit nor an
# underscore", i.e. a letter in ANY script -- the Unicode-safe replacement
# for the [A-Za-z] that shattered accented names.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def split_case(text: str) -> str:
    """Insert a space at every camelCase / PascalCase word boundary.

    Done BEFORE case folding, so ``customerAge`` still yields its words
    (customer, age) rather than one unmatchable blob. Implemented as a scan
    rather than a regex because ``re`` has no Unicode ``\\p{Lu}`` class:
    a character class of ``A-Z`` would silently miss ``Ä``/``Ö``/``É`` and
    reintroduce exactly the Unicode hole this module exists to close.
    """
    out: List[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if i and ch.isupper():
            prev = text[i - 1]
            nxt = text[i + 1] if i + 1 < n else ""
            # lower/digit -> Upper  (customerAge, age2Group)
            # Upper -> Upper+lower  (HTTPServer -> HTTP Server)
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                out.append(" ")
        out.append(ch)
    return "".join(out)


def _is_missing(name: object) -> bool:
    """True for ``None`` and for the NaN / NA scalars a column label can hold.

    NaN is the only value unequal to itself, which catches a Python float NaN
    and every numpy float NaN without importing numpy into this module.
    ``pd.NA`` answers that comparison with another NA rather than with a bool,
    so the ``TypeError`` raised by ``bool()`` IS the missing answer here and is
    deliberately swallowed rather than propagated.
    """
    if name is None:
        return True
    try:
        return bool(name != name)
    except (TypeError, ValueError):
        return True


def name_token_list(name: object) -> List[str]:
    """snake/kebab/space/camelCase -> ordered lowercase word tokens.

    Accent-folded, so ``género``/``genero`` and ``âge``/``age`` unify.
    ``casefold`` runs before the NFKD strip because German ``ß`` does not
    decompose under NFKD but DOES casefold to ``ss`` (``Größe`` -> grosse).

    A NAME NOBODY SUPPLIED HAS NO WORDS. Measured 2026-09-27, before the
    ``_is_missing`` guard below: the ``str(name)`` coercion turned an absent
    name into a real, matchable word token, ``None -> ['none']`` and
    ``float('nan') -> ['nan']``, and the invented token then WON a whole-token
    comparison: ``tokens_contain('none_reported', None)`` returned ``True``,
    a match asserted from a keyword that was never supplied, out of the module
    this package nominates as the single source of truth for name matching.
    An empty token list is what every caller already reads as "nothing to
    match on" (``tokens_contain`` refuses an empty needle, and
    ``resolve_attribute_key`` returns its explicit could-not-resolve ``None``).
    The STRING ``"None"`` still tokenises to ``['none']``: that is a name
    somebody wrote, and only the missing objects are refused.
    """
    if _is_missing(name):
        return []
    text = split_case(str(name))
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _TOKEN_RE.findall(folded)


def name_tokens(name: object) -> Set[str]:
    """:func:`name_token_list` as a set, for whole-token overlap tests."""
    return set(name_token_list(name))


def tokens_contain(column: object, keyword: object) -> bool:
    """True when ``keyword``'s tokens appear as WHOLE tokens in ``column``.

    Multi-token keywords must appear contiguously and in order, so
    ``census_tract`` matches ``census_tract_id`` but not a frame that merely
    happens to hold a ``census`` column and a ``tract`` column. Substring
    matching is deliberately not offered: it is the defect this module
    exists to remove.

    A KEYWORD WITH NO TOKENS IS REFUSED, never matched. There is nothing to
    look for, so ``False`` is the answer, and the ``False`` is reached without
    consulting the column. Measured 2026-09-27: while the tokeniser coerced a
    missing name with ``str()``, ``tokens_contain('none_reported', None)``
    returned ``True`` and ``tokens_contain('nan_column', float('nan'))``
    returned ``True``, both of them whole-token matches on a keyword nobody
    supplied. See :func:`name_token_list`.
    """
    kw = keyword if isinstance(keyword, list) else name_token_list(keyword)
    if not kw:
        return False
    col = column if isinstance(column, list) else name_token_list(column)
    return _is_subsequence(col, kw)


def _is_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return False
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and list(haystack[i : i + m]) == list(needle):
            return True
    return False


# ---------------------------------------------------------------------------
# Protected-attribute name resolution
# ---------------------------------------------------------------------------
#
# The proxy catalogues in preprocessing.bias_detection.proxy and
# preprocessing.feature_engineering.correlation are keyed by a canonical
# attribute name ('gender', 'race', ...). The lookup used to be
# ``key in attr_lower or attr_lower in key``, which is a substring test in
# both directions and therefore knows only the exact English word the
# catalogue happens to use. Measured 2026-09-10 on the same column and the
# same 0.515 correlation, renaming the column and changing nothing about the
# data:
#
#     attr='gender'  -> risk=critical, type=direct,
#                       affected_groups=['Women', 'Non-binary individuals']
#     attr='sex'     -> risk=high,     type=unclassified, affected_groups=[]
#
# UCI Adult and COMPAS, the two most-cited fairness datasets in the field,
# both name that column 'sex'; 'ethnicity' is the standard UK/EU spelling of
# 'race'. The empty affected_groups list was printed as a finding when the
# attribute had simply never been looked up.
#
# The fix is a synonym table plus an explicit "unresolved" answer, NOT a
# relaxation to token OVERLAP: overlap would make the catalogue key
# 'national_origin' match any attribute containing the word 'origin'. A key
# matches only as a complete, contiguous token sequence.
PROTECTED_ATTR_SYNONYMS = {
    # -> gender
    "sex": "gender",
    "sexe": "gender",
    "sexo": "gender",
    "genero": "gender",  # 'género' folds to this
    "genere": "gender",
    "genre": "gender",
    "geschlecht": "gender",
    # -> race
    "ethnicity": "race",
    "ethnic": "race",
    "ethnic_origin": "race",
    "ethnic_group": "race",
    "ethnicite": "race",  # 'ethnicité'
    "etnia": "race",
    "raza": "race",
    "raca": "race",  # 'raça'
    "rasse": "race",
    "herkunft": "race",
    # -> age
    "alter": "age",
    "edad": "age",
    "idade": "age",
    "leeftijd": "age",
    "years_old": "age",
    "age_years": "age",
    # -> disability
    "disabled": "disability",
    "handicap": "disability",
    "behinderung": "disability",
    "discapacidad": "disability",
    "invalidite": "disability",  # 'invalidité'
    # -> religion
    "religious": "religion",
    "faith": "religion",
    "confession": "religion",
    "glaube": "religion",
    "religione": "religion",
    # -> national_origin
    "nationality": "national_origin",
    "nationalitat": "national_origin",  # 'nationalität'
    "nationalitaet": "national_origin",
    "nationalite": "national_origin",  # 'nationalité'
    "nacionalidad": "national_origin",
    "nazionalita": "national_origin",
    "citizenship": "national_origin",
    "country_of_origin": "national_origin",
    "country_of_birth": "national_origin",
    "staatsangehorigkeit": "national_origin",  # 'staatsangehörigkeit'
    # -> income
    "salary": "income",
    "salario": "income",
    "wage": "income",
    "wages": "income",
    "earnings": "income",
    "einkommen": "income",
    "revenu": "income",
    "socioeconomic": "income",
    "socioeconomic_status": "income",
    "ses": "income",
}


def resolve_attribute_key(
    protected_attr: object,
    catalogue_keys: Iterable[str],
) -> Optional[str]:
    """Map a protected-attribute column name onto a catalogue key.

    Returns ``None`` when the name matches nothing, and that ``None`` is a
    COULD-NOT-CHECK: the caller must disclose it rather than reporting the
    empty lookup as "no pattern found". Matching is whole-token and, for
    multi-word keys, contiguous, so 'origin' alone never resolves to
    'national_origin'.
    """
    keys = list(catalogue_keys)
    if not keys:
        return None
    tokens = name_token_list(protected_attr)
    if not tokens:
        return None
    joined = "_".join(tokens)
    keyset = set(keys)

    if joined in keyset:
        return joined
    syn = PROTECTED_ATTR_SYNONYMS.get(joined)
    if syn in keyset:
        return syn

    # Longest first, so 'national_origin' wins over a hypothetical 'origin'.
    for key in sorted(keys, key=lambda k: -len(name_token_list(k))):
        if tokens_contain(tokens, key):
            return key
    for name, key in sorted(
        PROTECTED_ATTR_SYNONYMS.items(), key=lambda kv: -len(name_token_list(kv[0]))
    ):
        if key in keyset and tokens_contain(tokens, name):
            return key
    return None
