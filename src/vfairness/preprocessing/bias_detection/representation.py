"""
B. Representation Bias Detection

Detects underrepresentation and overrepresentation of demographic groups
by comparing dataset distributions against population benchmarks.

Key capabilities:
- Calculate representation ratios for protected groups
- Compare against census or population benchmarks
- Statistical significance testing for representation gaps
- Intersectional representation analysis
- Automatic flagging of underrepresented groups

References:
    - Buolamwini & Gebru (2018): Gender Shades
    - Mehrabi et al. (2021): A Survey on Bias and Fairness in ML
    - Suresh & Guttag (2021): A Framework for Understanding Sources of Harm
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy import stats

from ..._not_assessed import NOT_ASSESSED, warn_not_assessed


class RepresentationSeverity(Enum):
    """Severity levels for representation bias."""

    CRITICAL = "critical"  # Severe underrepresentation (<50% of expected)
    HIGH = "high"  # Significant underrepresentation (50-70% of expected)
    MEDIUM = "medium"  # Moderate underrepresentation (70-80% of expected)
    LOW = "low"  # Slight underrepresentation (80-90% of expected)
    ADEQUATE = "adequate"  # Acceptable representation (90-110% of expected)
    OVERREPRESENTED = "overrepresented"  # Potential overrepresentation (>110%)
    # NOT ASSESSED, in either of the two ways this module can fail to assess:
    # too few rows to read a proportion from, or no expected proportion to
    # compare one against (no benchmark was supplied and none of the built-in
    # ones describes the labels this column uses). It is never a statement
    # about the data being fine; ADEQUATE is that statement.
    INSUFFICIENT_DATA = "insufficient_data"


def _group_key(group: Any) -> str:
    """The one normalisation used for every group label in this module.

    Benchmark keys, dataset labels and ratio keys are all folded through this
    single function so no two of them can disagree about what "the same group"
    means.
    """
    return str(group).lower().strip()


def _count_label_matches_type_tolerant(series: "pd.Series", target_group: Any) -> int:
    """How many rows the label matches once the dtype stops mattering.

    Used only to tell a LABEL MISMATCH apart from a group that is genuinely
    absent. The string comparison the caller performs is the measurement; this is
    the evidence that the measurement asked the wrong question. Measured
    2026-09-27 on 300 rows of ``[1.0] * 150 + [0.0] * 150`` with
    ``target_group=1``: the string comparison matches 0 rows ('1' against '1.0')
    and this returns 150. Returns 0 for a label that is not a number, which is
    the case where a 0 count really is a finding.
    """
    try:
        target_number = float(target_group)
    except (TypeError, ValueError):
        return 0
    if target_number != target_number:  # a NaN target matches nothing.
        return 0
    numeric = pd.to_numeric(series, errors="coerce")
    return int((numeric == target_number).sum())


def _is_a_divisible_share(value: Any) -> bool:
    """True only for a share this module can divide by: positive and finite.

    BGL5 2026-09-27. Both consumers used a bare ``value > 0``, which is False for
    a NaN (so the group was dropped with no trace) and True for
    ``float('inf')`` (so it reached ``int(inf * n)``). Measured before:
    ``compare_to_benchmark`` on 300 observed rows against
    ``{'m': 0.5, 'f': 0.5, 'x': 0.0}`` returned ratios and gaps for m and f only,
    with ``measurement_status 'measured'`` and ``groups_not_measured []``, i.e.
    "nothing went unmeasured" about a group it had just dropped, and identically
    for -0.1 and for NaN; ``calculate_representation_ratio`` with
    ``benchmark_proportion=float('inf')`` raised ``OverflowError: cannot convert
    float infinity to integer`` instead of the None its docstring promises.

    Written with ``float()`` and self-comparison rather than ``isinstance``,
    because ``isinstance(v, (int, float))`` is False for ``numpy.float32`` and
    ``numpy.int64`` and would discard real numbers.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v:  # NaN is the one value unequal to itself.
        return False
    return 0.0 < v < float("inf")


_SHARE_SCALE_TOLERANCE = 1e-9


def _is_on_the_share_scale(value: Any) -> bool:
    """True only for a value that can BE a population share: divisible and <= 1.

    :func:`_is_a_divisible_share` closes the BELOW and NON-FINITE end of the
    scale. This closes the ABOVE-1 end, and the two are different questions.

    BGL-F5 2026-09-30. Nothing in this module tested that a CALLER-SUPPLIED
    benchmark was a share at all, and the consequence was a verdict no data could
    have changed. ``actual_proportion`` cannot exceed 1.0, so once
    ``benchmark_proportion`` passes 1.25 the test ``ratio < 0.8`` is True for
    EVERY possible dataset, and once it passes 0.8333 ``ratio > 1.2`` can never
    fire at all. Measured before, 300 rows with 'm' holding 150 of them
    (actual_proportion 0.50), against the docstring's own "Expected proportion
    (0-1)": bp=2.0 -> representation_ratio 0.25, is_underrepresented True,
    deficit_count 450, measurement_status 'measured', warnings 0, i.e. a deficit
    of 450 people in a 300-row frame; bp=100.0 -> ratio 0.005, deficit 29850;
    bp=1e9 -> ratio 5e-10, deficit 299999999850, still 'measured', still silent.

    The tolerance is a small absolute epsilon, not an equality: 1.0 itself is a
    legal share (a benchmark expecting the whole population), and a share
    arrived at by division can land a few ulps above it.
    """
    if not _is_a_divisible_share(value):
        return False
    return float(value) <= 1.0 + _SHARE_SCALE_TOLERANCE


def _is_a_readable_floor(value: Any) -> bool:
    """True when a sample-size floor is a number a row count can be compared to.

    Written with ``float()`` and self-comparison rather than ``isinstance``, for
    the reason :func:`_is_a_divisible_share` records: ``isinstance(v, (int,
    float))`` is False for ``numpy.int64`` and would discard a real floor. A NaN
    is the one value every comparison is False against, which is what makes it
    able to switch a gate off in silence; ``inf`` is readable and refuses
    correctly, and 0 or a negative is a caller deliberately grading any sample.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v  # NaN is the one value unequal to itself.


def _positive_share_mass(benchmark: Dict[str, float]) -> float:
    """Total expected mass of a benchmark, counting only its divisible shares.

    The same sum ``detect_representation_bias`` takes at representation.py's
    renormalisation site, factored out so the three entry points that consume a
    caller-supplied benchmark cannot disagree about what "sums to 1.0" means. A
    share that is not divisible (absent, zero, negative, NaN, infinite) carries no
    mass and is left to the per-group refusal, exactly as before.
    """
    return float(sum(float(v) for v in benchmark.values() if _is_a_divisible_share(v)))


def _finite_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """The quotient, or None when it is not a finite number.

    :func:`_is_a_divisible_share` tests the DIVISOR. This tests the RESULT, which
    is the thing that gets published, and the two are not the same question.

    BGL6 F12, 2026-09-29. ``0.0 < v < inf`` admits every positive finite value
    including the subnormals, and a subnormal divisor sends a perfectly ordinary
    numerator over the top of the float range. Measured before, on 300 rows with
    ``benchmark_proportion=5e-324``: ``calculate_representation_ratio`` published
    ``representation_ratio`` inf with ``is_overrepresented`` True,
    ``measurement_status 'measured'``, ``not_measured_reason`` None and ZERO
    warnings, from the function whose docstring promises None "when
    benchmark_proportion is not a positive finite number". ``compare_to_benchmark``
    divided at the same hole one screen above.

    The guard sits here, above both call sites, rather than being repeated at each:
    they share the precondition, and a guard written per branch is a guard that one
    branch will be missing. An infinite ratio is a could-not-check, never a finding
    of extreme overrepresentation, so the caller reports None for it.
    """
    if not _is_a_divisible_share(denominator):
        return None
    try:
        quotient = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(quotient):
        return None
    return quotient


def _fold_shares(shares: Dict[Any, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Group shares folded onto :func:`_group_key`, SUMMING on a collision.

    Returns the folded mapping and, for every key two or more raw labels folded
    into, the raw labels in the order they arrived.

    BGL6 F12, 2026-09-29. ``{_group_key(k): v for k, v in actual.items()}``
    OVERWRITES on a collision, so a second spelling of the same group discarded the
    first one's share instead of adding to it. Measured before, on 300 rows that are
    ALL male, spelt 'm' (200) and 'M' (100), against ``{'m': 0.5, 'f': 0.5}``:
    ``representation_ratios {'m': 0.667}`` with ``gaps {'m': -0.167}``, a DEFICIT
    for a group holding every observed row, ``measurement_status 'measured'``,
    ``groups_not_measured []`` and 'M' named nowhere, while the chi-squared half of
    the same return refused because "66.7% of the rows fall in groups the benchmark
    never mentions" (the 200 rows the overwrite had dropped).

    Summing is what ``_analyze_single_attribute`` in this module already does for
    the same reason, in the loop above ``_canonical_group``; this function is that
    behaviour given a name so the two entry points cannot drift again.
    """
    folded: Dict[str, float] = {}
    raw_labels: Dict[str, List[str]] = {}
    for label, share in shares.items():
        key = _group_key(label)
        # The RAW label, not the key: 'Male' and 'MALE' both fold to 'male', and a
        # reader who is told only 'male' twice learns nothing about which spellings
        # were combined.
        raw_labels.setdefault(key, []).append(str(label))
        folded[key] = folded.get(key, 0.0) + float(share)
    # Only the keys that really collided, so an empty dict means "no label was
    # folded into another" and not "nobody looked".
    return folded, {k: v for k, v in raw_labels.items() if len(v) > 1}


class _GroupRatios(Dict[str, float]):
    """Representation ratios that answer to the labels ``group_distributions`` uses.

    CRITICAL, do not flatten this back to a plain dict. ``group_distributions``
    carries each group label EXACTLY as it appears in the data ("Female",
    "Non-binary", or a number out of a numeric column), while a ratio can only
    be computed after the label has been folded to match a benchmark key
    (``_group_key``). Every caller that wants "the ratio for this group" reads
    the two dicts TOGETHER, and with plain dicts on both sides that join MISSES
    for any dataset whose labels are not already lower-case.

    That is not hypothetical. The render gallery's own audit lost all six of its
    benchmark bars to it: the representation panel printed "no population
    benchmark was reported for this group" six times on a report that supplied a
    benchmark for every one of them, two inches under a HIGH "Severe
    underrepresentation in gender: [female]" computed from those same ratios.
    One page, two contradictory claims, and the reasonable reading is the wrong
    one. A previous default masked it by drawing the benchmark equal to the
    dataset share, which was a fabricated all-clear; removing that default did
    not create this bug, it exposed it.

    So the normalisation lives HERE, once, rather than being re-invented as a
    ``.lower()`` at each call site and left broken for the next caller. Keys are
    STORED normalised, exactly as before, so iteration, equality, ``to_dict()``
    and JSON output are unchanged; every lookup folds the key it is handed, so
    ``ratios.get("Female")``, ``ratios.get("female")`` and ``ratios.get(0)``
    against a "0" group all answer.

    ``aliases`` extends that same fold by one step, for benchmarks that declare
    spelling variants (``m`` -> ``male``). Ratios are stored ONCE, under the
    canonical category, so the mapping stays a distribution over real groups;
    the lookup folds an alias onto it, so a dataset labelled "M"/"F" still joins
    ``group_distributions`` to ``representation_ratios`` on the labels it
    actually spells. Without that step the alias fold at the source would have
    re-created the very miss this class exists to close.
    """

    # Class-level default so an instance built without __init__ (dict subclasses
    # are rebuilt that way by copy/pickle, which then replays __setitem__) still
    # has a map to fold through instead of raising AttributeError.
    _aliases: Dict[str, str] = {}

    def __init__(self, *args, aliases: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__()
        self._aliases = {_group_key(k): _group_key(v) for k, v in (aliases or {}).items()}
        self.update(dict(*args, **kwargs))

    def _canonical(self, key) -> str:
        folded = _group_key(key)
        return self._aliases.get(folded, folded)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(self._canonical(key), value)

    def __getitem__(self, key):
        return super().__getitem__(self._canonical(key))

    def __contains__(self, key) -> bool:
        return super().__contains__(self._canonical(key))

    def get(self, key, default=None):
        return super().get(self._canonical(key), default)

    def update(self, other: Any = (), /, **kwargs: float) -> None:
        # Routed through __setitem__ so a bulk insert cannot smuggle in an
        # unfolded key that no folded lookup would then find. dict.update does
        # NOT call __setitem__ on its own, which is why this override exists.
        items = other.items() if hasattr(other, "items") else other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value


@dataclass
class RepresentationBiasResult:
    """
    Result of representation bias detection for a protected attribute.

    Attributes:
        attribute: Protected attribute column name
        group_distributions: Distribution of groups in the dataset
        representation_ratios: Ratio of actual to expected representation per group
        underrepresented_groups: Groups with representation ratio < threshold
        overrepresented_groups: Groups with representation ratio > threshold
        severity: Severity of representation bias, graded from
            ``representation_ratios`` ALONE. INSUFFICIENT_DATA means those
            ratios could not be computed, so no under/overrepresentation
            verdict was reached; ``underrepresented_groups`` and
            ``overrepresented_groups`` are then empty by construction, because
            a group cannot be below an expectation that does not exist. It does
            NOT grade ``intersectional_findings``, which are measured against
            independence from the data's own marginals and therefore stand on
            their own even when this field says INSUFFICIENT_DATA.
        chi_squared_statistic: Chi-squared test statistic
        chi_squared_pvalue: P-value from chi-squared test
        sample_size: Total sample size analyzed
        recommendations: Suggested actions to address bias
        intersectional_findings: Findings from intersectional analysis (if
            performed). Measured against the product of the data's own
            marginals, so these need no population benchmark and are NOT
            summarised by ``severity``; read the two as separate analyses.
        benchmark_source: Where the benchmark came from ('provided',
            'default_us_census', or None when none was applied)
        significance_level: The alpha the caller asked for, recorded beside the
            verdict it produced so the two cannot drift apart
        chi_squared_significant: Three-state verdict on the chi-squared p-value
            against that alpha. True or False when the test ran; None when it
            could NOT run, which means UNMEASURED and never 'not significant'.
            ``chi_squared_statistic`` and ``chi_squared_pvalue`` are None in
            exactly that case. Do not read this field with a truthiness test.
    """

    attribute: str
    group_distributions: Dict[str, float]
    representation_ratios: Dict[str, float]
    underrepresented_groups: List[Dict[str, Any]]
    overrepresented_groups: List[Dict[str, Any]]
    severity: RepresentationSeverity
    chi_squared_statistic: Optional[float]
    chi_squared_pvalue: Optional[float]
    sample_size: int
    recommendations: List[str]
    intersectional_findings: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_source: Optional[str] = None
    significance_level: float = 0.05
    chi_squared_significant: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attribute": self.attribute,
            "group_distributions": self.group_distributions,
            "representation_ratios": self.representation_ratios,
            "underrepresented_groups": self.underrepresented_groups,
            "overrepresented_groups": self.overrepresented_groups,
            "severity": self.severity.value,
            "chi_squared_statistic": self.chi_squared_statistic,
            "chi_squared_pvalue": self.chi_squared_pvalue,
            "sample_size": self.sample_size,
            "recommendations": self.recommendations,
            "intersectional_findings": self.intersectional_findings,
            "benchmark_source": self.benchmark_source,
            "significance_level": self.significance_level,
            "chi_squared_significant": self.chi_squared_significant,
        }


@dataclass(frozen=True)
class _BenchmarkScheme:
    """One population benchmark: a distribution, plus the spellings that fold onto it.

    ``distribution`` maps a CANONICAL category to its population proportion and
    MUST sum to 1.0. ``aliases`` maps a spelling variant a dataset might use
    onto one of those canonical categories, and carries no probability mass of
    its own.

    WHY THE TWO ARE SEPARATE, AND WHY THIS IS NOT A STYLE CHOICE. The shipped
    table used to hold both in one dict, giving every spelling its own copy of
    the proportion:

        "gender": {"male": 0.49, "female": 0.51, "m": 0.49, "f": 0.51,
                   "man": 0.49, "woman": 0.51}

    Consumed as a distribution that sums to 3.000, not 1.0. Three faults
    compounded from there, and all three were measured on 4900 male + 5100
    female rows, a dataset that IS the built-in benchmark to the last row:

    1. the renormalisation below divided by 3.0, so 0.49 became 0.1633;
    2. every present group then read as three times its expected share
       (ratio 3.0 for both male and female);
    3. the absent-group loop treats a benchmark key missing from the data as a
       missing demographic group, so "m", "f", "man" and "woman" were reported
       as phantom populations with deficits of 1633 and 1700 people each.

    The verdict was CRITICAL, "Severe underrepresentation detected", naming
    groups m, f and man, with chi_squared_statistic None because scipy refused
    the mismatched totals. A spelling variant is not a demographic group and
    cannot be underrepresented; it is a label for a group that is already
    counted. Keeping the two in one mapping made that distinction unexpressible,
    so it is expressed in the type instead, and _validate_default_schemes()
    below refuses at import any table that blurs it again.
    """

    name: str
    distribution: Dict[str, float]
    aliases: Dict[str, str]


# Default population benchmarks for representation analysis (US-centric,
# customize as needed). US Census 2020 approximate distributions.
#
# age_group carries TWO schemes because the old table carried two mutually
# exclusive bandings at once: the seven detailed bands sum to 1.00 on their own,
# and young/middle/senior sum to 1.00 on their own as AGGREGATES of those bands
# (young = under_18 + 18-24 = 0.31, middle = 25-34 + 35-44 + 45-54 = 0.39,
# senior = 55-64 + 65+ = 0.30). They are not aliases of each other and no single
# distribution can hold both. The scheme that matches the labels a dataset
# actually uses is selected per call by _select_default_scheme().
_DEFAULT_BENCHMARK_SCHEMES: Dict[str, Tuple[_BenchmarkScheme, ...]] = {
    "gender": (
        _BenchmarkScheme(
            name="us_census_2020_gender",
            distribution={"male": 0.49, "female": 0.51},
            aliases={
                "m": "male",
                "man": "male",
                "men": "male",
                "f": "female",
                "woman": "female",
                "women": "female",
            },
        ),
    ),
    "race": (
        _BenchmarkScheme(
            name="us_census_2020_race",
            distribution={
                "white": 0.576,
                "black": 0.121,
                "hispanic": 0.187,
                "asian": 0.059,
                "native_american": 0.007,
                "pacific_islander": 0.002,
                "multiracial": 0.028,
                "other": 0.02,
            },
            aliases={
                "african_american": "black",
                "latino": "hispanic",
                "latina": "hispanic",
                "hispanic_latino": "hispanic",
                "mixed": "multiracial",
                "two_or_more": "multiracial",
            },
        ),
    ),
    "age_group": (
        _BenchmarkScheme(
            name="us_census_2020_age_detailed",
            distribution={
                "under_18": 0.22,
                "18-24": 0.09,
                "25-34": 0.14,
                "35-44": 0.13,
                "45-54": 0.12,
                "55-64": 0.13,
                "65+": 0.17,
            },
            aliases={"elderly": "65+", "65_plus": "65+", "minor": "under_18"},
        ),
        _BenchmarkScheme(
            name="us_census_2020_age_coarse",
            distribution={"young": 0.31, "middle": 0.39, "senior": 0.30},
            aliases={"middle_aged": "middle", "elderly": "senior"},
        ),
    ),
}


def _validate_default_schemes(
    schemes_by_attribute: Dict[str, Tuple[_BenchmarkScheme, ...]],
) -> None:
    """Refuse, at import, any built-in table that is not a distribution.

    The defect this exists for shipped in a release: a table whose values summed
    to 3.000 was consumed as a probability distribution and graded a perfectly
    representative dataset CRITICAL. Nothing checked it, because the table is a
    constant and constants are assumed right. This is that check.
    """
    for attribute, schemes in schemes_by_attribute.items():
        if not schemes:
            raise ValueError(f"No benchmark scheme declared for '{attribute}'")
        for scheme in schemes:
            total = sum(scheme.distribution.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"Built-in benchmark '{scheme.name}' for '{attribute}' sums to "
                    f"{total!r}, not 1.0; a benchmark is a probability distribution."
                )
            if any(v <= 0 for v in scheme.distribution.values()):
                raise ValueError(
                    f"Built-in benchmark '{scheme.name}' for '{attribute}' has a "
                    f"non-positive proportion; every category must carry real mass."
                )
            overlap = set(scheme.aliases) & set(scheme.distribution)
            if overlap:
                raise ValueError(
                    f"Built-in benchmark '{scheme.name}' for '{attribute}' declares "
                    f"{sorted(overlap)} as BOTH a category and an alias; an alias "
                    f"carries no mass of its own."
                )
            unknown = set(scheme.aliases.values()) - set(scheme.distribution)
            if unknown:
                raise ValueError(
                    f"Built-in benchmark '{scheme.name}' for '{attribute}' aliases "
                    f"onto {sorted(unknown)}, which is not one of its categories."
                )


_validate_default_schemes(_DEFAULT_BENCHMARK_SCHEMES)


# The canonical distributions on their own, kept under the name the docstrings
# and downstream callers already use. Each one sums to 1.0. For an attribute
# with more than one scheme this is the PRIMARY scheme; the alternatives live in
# _DEFAULT_BENCHMARK_SCHEMES and are selected from the data's own labels.
DEFAULT_BENCHMARKS: Dict[str, Dict[str, float]] = {
    attribute: dict(schemes[0].distribution)
    for attribute, schemes in _DEFAULT_BENCHMARK_SCHEMES.items()
}

# The spelling variants, keyed the same way. A value here is a label a dataset
# may use; it normalises onto a category of DEFAULT_BENCHMARKS and never
# competes with one.
DEFAULT_BENCHMARK_ALIASES: Dict[str, Dict[str, str]] = {
    attribute: dict(schemes[0].aliases) for attribute, schemes in _DEFAULT_BENCHMARK_SCHEMES.items()
}


def detect_representation_bias(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    underrepresentation_threshold: float = 0.8,
    overrepresentation_threshold: float = 1.2,
    min_group_size: int = 30,
    significance_level: float = 0.05,
    include_intersectional: bool = True,
    max_intersection_depth: int = 2,
) -> List[RepresentationBiasResult]:
    """
    Detect representation bias in dataset for specified protected attributes.

    Compares the distribution of protected groups in the dataset against
    population benchmarks to identify under/overrepresentation.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute column names
        benchmarks: Dictionary of expected distributions per attribute
                   Format: {'attribute': {'group': proportion, ...}, ...}
                   If None, uses DEFAULT_BENCHMARKS where available
        underrepresentation_threshold: Ratio below which a group is underrepresented
        overrepresentation_threshold: Ratio above which a group is overrepresented
        min_group_size: Minimum group size for reliable analysis
        significance_level: Alpha for the chi-squared goodness-of-fit verdict.
            Must be strictly between 0 and 1. It is applied to the p-value and
            reported back on every result as ``significance_level``, with the
            verdict in ``chi_squared_significant`` (True / False / None, where
            None means the test could not run and is NOT "not significant").
            It does NOT move ``severity``, which is graded from the per-group
            representation ratios and not from a hypothesis test.
        include_intersectional: Whether to analyze intersectional groups
        max_intersection_depth: Maximum number of attributes to combine

    Returns:
        List of RepresentationBiasResult objects for each protected attribute

    Example:
        >>> benchmarks = {
        ...     'gender': {'male': 0.5, 'female': 0.5},
        ...     'race': {'white': 0.6, 'black': 0.13, 'asian': 0.06, 'other': 0.21}
        ... }
        >>> results = detect_representation_bias(
        ...     df, ['gender', 'race'],
        ...     benchmarks=benchmarks
        ... )
        >>> for r in results:
        ...     print(f"{r.attribute}: {r.severity.value}")
        ...     for ug in r.underrepresented_groups:
        ...         print(f"  {ug['group']}: {ug['ratio']:.1%} of expected")

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

    Ledger row: detect_representation_bias. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            f"significance_level must be strictly between 0 and 1, got "
            f"{significance_level!r}. It is the alpha applied to the "
            f"chi-squared p-value; a value outside (0, 1) cannot be one."
        )

    results = []
    benchmarks = benchmarks or {}

    for attr in protected_attributes:
        if attr not in df.columns:
            continue

        # Distinguish caller-supplied benchmarks from the built-in US-census
        # fallback: silently labelling the fallback 'provided' misled users
        # into thinking their own population figures were being applied.
        benchmark = benchmarks.get(attr)
        benchmark_source = "provided" if benchmark else None
        aliases: Optional[Dict[str, str]] = None
        if benchmark is None:
            scheme = _get_default_benchmark(attr, df[attr])
            if scheme is not None:
                benchmark = dict(scheme.distribution)
                aliases = dict(scheme.aliases)
                benchmark_source = "default_us_census"
                warnings.warn(
                    f"No benchmark provided for '{attr}'; falling back to "
                    f"the built-in approximate US Census 2020 distribution "
                    f"(benchmark_source='default_us_census', scheme "
                    f"'{scheme.name}'). Pass "
                    f"benchmarks={{'{attr}': {{...}}}} for your actual "
                    f"population.",
                    UserWarning,
                    stacklevel=2,
                )

        result = _analyze_single_attribute(
            df,
            attr,
            benchmark=benchmark,
            benchmark_aliases=aliases,
            benchmark_source=benchmark_source,
            underrepresentation_threshold=underrepresentation_threshold,
            overrepresentation_threshold=overrepresentation_threshold,
            min_group_size=min_group_size,
            significance_level=significance_level,
        )
        results.append(result)

    if include_intersectional and len(protected_attributes) >= 2:
        for depth in range(2, min(max_intersection_depth + 1, len(protected_attributes) + 1)):
            for combo in combinations(protected_attributes, depth):
                if not all(a in df.columns for a in combo):
                    continue

                intersectional_result = _analyze_intersectional(
                    df,
                    list(combo),
                    min_group_size=min_group_size,
                    underrepresentation_threshold=underrepresentation_threshold,
                )

                # Attach intersectional findings to the first attribute's result
                if intersectional_result and results:
                    for r in results:
                        if r.attribute in combo:
                            r.intersectional_findings.extend(intersectional_result)
                            break

    return results


def _analyze_single_attribute(
    df: pd.DataFrame,
    attribute: str,
    *,
    benchmark: Optional[Dict[str, float]],
    underrepresentation_threshold: float,
    overrepresentation_threshold: float,
    min_group_size: int,
    significance_level: float,
    benchmark_aliases: Optional[Dict[str, str]] = None,
    benchmark_source: Optional[str] = None,
) -> RepresentationBiasResult:
    """Analyze representation for a single attribute."""
    series = df[attribute].dropna()
    n_total = len(series)
    # THE DOORS `dropna` DOES NOT CLOSE. BGL grade-1 G08, 2026-09-30. `dropna`
    # removes None, float nan, pd.NA and pd.NaT, and every one of them also has
    # a STRING spelling that `str()` mints and that a CSV or JSON export writes:
    # 'None', 'nan', '<NA>', 'NaT', 'null' and the blank string. Those survive
    # `dropna` as ordinary labels. Measured on this repo, 300 rows of
    # White/Black/Hispanic with the first 60 race values set to the string
    # 'None': the result carried group_distributions
    # {'White': 0.463, 'Black': 0.217, 'None': 0.2, 'Hispanic': 0.12} and
    # sample_size 300, so 'None' was published as a demographic group holding a
    # fifth of the population, and every real group's share was diluted by 60
    # rows that carry no group at all (White is 0.579 among the 240 labelled
    # rows, not 0.463). The same 60 rows written as real pd.NA gave
    # sample_size 240 and three groups, which is this function's own policy.
    #
    # The shares are NOT recomputed and no row is dropped here, deliberately:
    # 'None' is a legitimate category for some protected attributes (a
    # disability status of "None"), and silently deleting a fifth of a real
    # population would be the worse error of the two. What was missing is that
    # the reader was never told, so the caveat is stated and named. The
    # chi-squared refusal that fires alongside it says only that the benchmark
    # does not mention the label; it does not say the label is not a group.
    _absence_spellings = {"", "none", "nan", "<na>", "nat", "null"}
    try:
        _label_text = series.astype("string").fillna("").str.strip().str.lower()
        _absent_labels = sorted(
            {str(raw) for raw, key in zip(series, _label_text) if key in _absence_spellings}
        )
        _n_absent = int(_label_text.isin(_absence_spellings).sum())
    except Exception:  # noqa: BLE001 -- never turn a dtype problem into a finding
        _absent_labels, _n_absent = [], 0
    if _n_absent:
        warnings.warn(
            f"{_n_absent} of {n_total} row(s) for '{attribute}' carry a label that is a "
            f"SPELLING OF ABSENCE rather than a group ({_absent_labels}); str() of an "
            "absent value produces exactly these and a CSV export writes them, so they "
            "survive dropna as ordinary labels. They are counted in sample_size and in "
            "group_distributions below, which means each of them appears as a group with "
            "a share of the population and every real group's share is diluted by them. "
            "If they mean 'missing', re-read the column with those spellings as NA "
            "before assessing representation; if they are a real category, this warning "
            "is safe to ignore.",
            UserWarning,
            stacklevel=2,
        )
    alias_map = {_group_key(k): _group_key(v) for k, v in (benchmark_aliases or {}).items()}

    # Min sample gate: with fewer rows than min_group_size the observed
    # proportions are noise (3 random rows previously yielded a confident
    # HIGH finding). Return an explicit insufficient-data outcome instead.
    if n_total < min_group_size:
        warnings.warn(
            f"Only {n_total} non-null rows for '{attribute}' "
            f"(min_group_size={min_group_size}); representation is "
            f"UNASSESSED, not adequate.",
            UserWarning,
            stacklevel=2,
        )
        return RepresentationBiasResult(
            attribute=attribute,
            group_distributions=(series.value_counts() / n_total).to_dict() if n_total else {},
            representation_ratios=_GroupRatios(aliases=alias_map),
            underrepresented_groups=[],
            overrepresented_groups=[],
            severity=RepresentationSeverity.INSUFFICIENT_DATA,
            chi_squared_statistic=None,
            chi_squared_pvalue=None,
            sample_size=n_total,
            recommendations=[
                f"Insufficient data: only {n_total} rows available for "
                f"'{attribute}' (minimum {min_group_size}). Collect more "
                f"data before drawing representation conclusions."
            ],
            benchmark_source=benchmark_source if benchmark else None,
            significance_level=significance_level,
        )

    value_counts = series.value_counts()
    group_distributions = (value_counts / n_total).to_dict()

    # Normalize group names for benchmark matching. `group_distributions` keeps
    # the labels the data actually uses, so a caller can print them; the ratios
    # below are keyed by the normalised form and are handed back in a
    # _GroupRatios, which folds any lookup key through the SAME _group_key. That
    # is what lets `ratios.get(label)` answer for a label taken straight out of
    # `group_distributions`. See the class docstring for the panel this broke.
    #
    # An alias is folded here too, and the shares of the labels that fold
    # together are SUMMED. That is what makes an alias an alias: "M" and "male"
    # are one population counted under two spellings, so they contribute one
    # share to one category. The old code both gave each spelling its own
    # benchmark mass AND, on the data side, let a second spelling of the same
    # group overwrite the first instead of adding to it.
    group_distributions_normalized: Dict[str, float] = {}
    for label, share in group_distributions.items():
        canonical = _canonical_group(label, alias_map)
        group_distributions_normalized[canonical] = (
            group_distributions_normalized.get(canonical, 0.0) + share
        )

    representation_ratios = _GroupRatios(aliases=alias_map)
    underrepresented = []
    overrepresented = []

    if benchmark:
        benchmark_normalized = {_group_key(k): v for k, v in benchmark.items()}

        # Benchmarks are population PROPORTIONS: if they do not sum to ~1,
        # renormalise (with a warning) instead of comparing raw values,
        # otherwise perfectly balanced data gets flagged with phantom
        # deficits (or real deficits get scaled away).
        bench_sum = sum(v for v in benchmark_normalized.values() if v and v > 0)
        if bench_sum > 0 and abs(bench_sum - 1.0) > 0.01:
            warnings.warn(
                f"Benchmark proportions for '{attribute}' sum to "
                f"{bench_sum:.3f}, not 1.0; renormalising.",
                UserWarning,
                stacklevel=2,
            )
            benchmark_normalized = {
                k: (v / bench_sum if v and v > 0 else v) for k, v in benchmark_normalized.items()
            }

        for group, actual_prop in group_distributions_normalized.items():
            expected_prop = benchmark_normalized.get(group)
            if expected_prop and expected_prop > 0:
                ratio = actual_prop / expected_prop
                representation_ratios[group] = ratio

                group_count = int(actual_prop * n_total)

                if ratio < underrepresentation_threshold:
                    underrepresented.append(
                        {
                            "group": group,
                            "ratio": ratio,
                            "actual_proportion": actual_prop,
                            "expected_proportion": expected_prop,
                            "count": group_count,
                            "deficit": int((expected_prop - actual_prop) * n_total),
                        }
                    )
                elif ratio > overrepresentation_threshold:
                    overrepresented.append(
                        {
                            "group": group,
                            "ratio": ratio,
                            "actual_proportion": actual_prop,
                            "expected_proportion": expected_prop,
                            "count": group_count,
                            "surplus": int((actual_prop - expected_prop) * n_total),
                        }
                    )

        # Benchmark groups entirely ABSENT from the data are the most extreme
        # underrepresentation there is, the loop above only visits groups
        # present in the data, so an all-male dataset against a 50/50
        # benchmark was rated 'adequate'. Flag every expected-but-missing
        # group with ratio 0.0 and its full deficit.
        for bench_group, expected_prop in benchmark_normalized.items():
            if not expected_prop or expected_prop <= 0:
                continue
            if bench_group not in group_distributions_normalized:
                representation_ratios[bench_group] = 0.0
                underrepresented.append(
                    {
                        "group": bench_group,
                        "ratio": 0.0,
                        "actual_proportion": 0.0,
                        "expected_proportion": expected_prop,
                        "count": 0,
                        "deficit": int(expected_prop * n_total),
                        "absent": True,
                    }
                )

        chi2_stat, chi2_pval, chi2_reason = _chi_squared_representation_test(
            group_distributions_normalized, benchmark_normalized, n_total
        )
        if chi2_reason is not None:
            warnings.warn(
                f"Chi-squared representation test did not run for "
                f"'{attribute}' ({chi2_reason}); chi_squared_statistic, "
                f"chi_squared_pvalue and chi_squared_significant are None, "
                f"which is UNMEASURED, not 'not significant'. The severity "
                f"below is graded from the per-group representation ratios "
                f"only.",
                UserWarning,
                stacklevel=3,
            )
    else:
        # Without a benchmark there is no expected proportion, so there is no
        # representation ratio, and nothing can be UNDERREPRESENTED: the word
        # means "below what was expected" and nothing was expected. This branch
        # appended findings anyway, scoring each group against 1 / len(props),
        # a uniform distribution over whatever labels the column happened to
        # contain. That denominator is not a population. It is an artefact of
        # the data's own shape, and it MOVES when the other groups are split
        # differently: a group holding exactly 5% of the rows scored ratio
        # 0.150 beside two other groups, 0.200 beside three and 0.250 beside
        # four, with its own share identical in all three. A number that
        # changes when its subject does not is not a measurement of its
        # subject.
        #
        # The object those findings landed in graded INSUFFICIENT_DATA, because
        # representation_ratios was left empty here and _determine_severity
        # reads an empty ratio map as "nothing was compared". So one result
        # said "I could not assess this" in `severity` while handing the reader
        # a named group to act on in `underrepresented_groups`, and the Pulse
        # UI renders the second without consulting the first. The entries also
        # carried neither `expected_proportion` nor `deficit`, which the
        # documented contract and every benchmarked entry do carry, so a
        # consumer reading the list uniformly raised KeyError on exactly these.
        #
        # Nothing MEASURED is dropped: every group's share stays in
        # `group_distributions` and the row count in `sample_size`. What goes is
        # the invented ratio and the verdict word attached to it. The skew is
        # said out loud instead, because silence here reads as "nothing to
        # report", and this is the module's third state, not a clean bill.
        chi2_stat, chi2_pval, chi2_reason = None, None, "no benchmark was supplied"

        props = list(group_distributions_normalized.values())
        if props and min(props) < 0.1 and max(props) > 0.3:
            small = sorted(g for g, p in group_distributions_normalized.items() if p < 0.1)
            warnings.warn(
                f"'{attribute}' is skewed: {', '.join(small)} hold under 10% of "
                f"the rows each while another group holds over 30%. No expected "
                f"proportion was available for any group, so representation is "
                f"UNBENCHMARKED: no group is reported as underrepresented and "
                f"that is NOT a finding of adequate representation. Pass "
                f"benchmarks={{'{attribute}': {{...}}}} for a verdict.",
                UserWarning,
                stacklevel=3,
            )

    severity = _determine_severity(underrepresented, representation_ratios, overrepresented)

    recommendations = _generate_representation_recommendations(
        severity, underrepresented, overrepresented
    )

    return RepresentationBiasResult(
        attribute=attribute,
        group_distributions=group_distributions,
        representation_ratios=representation_ratios,
        underrepresented_groups=underrepresented,
        overrepresented_groups=overrepresented,
        severity=severity,
        chi_squared_statistic=chi2_stat,
        chi_squared_pvalue=chi2_pval,
        sample_size=n_total,
        recommendations=recommendations,
        benchmark_source=benchmark_source if benchmark else None,
        significance_level=significance_level,
        # `is not None`, never a truthiness test. A chi-squared p-value of
        # exactly 0.0 is the STRONGEST evidence the test can give and it is
        # reached by ordinary data (it underflows the double range above
        # chi2 = 1497); `if chi2_pval` would send that to None, which is the
        # value reserved for a test that never ran. Same defect as S-01 in
        # compare_to_benchmark, one function away.
        chi_squared_significant=(chi2_pval < significance_level if chi2_pval is not None else None),
    )


def _chi_squared_representation_test(
    observed: Dict[str, float],
    expected: Dict[str, float],
    n_total: int,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Chi-squared goodness of fit, or the reason it could not be computed.

    Returns ``(statistic, pvalue, reason)``. ``reason`` is None when the test
    ran, and a short phrase when it did not; the two are never both set, so a
    caller cannot mistake a could-not-check for a measurement.

    The test spans EVERY benchmarked category, not just the ones the data
    happens to contain. A category the benchmark expects and the data does not
    have is an observed count of 0, which is exactly what a goodness-of-fit test
    is for: an all-male dataset against a 50/50 benchmark now yields a measured
    chi2 rather than nothing, so the CRITICAL that case already produced has a
    statistic behind it instead of a None.

    The old body was a bare ``except Exception: return None, None``. The one
    thing it caught in practice was scipy refusing a comparison whose observed
    and expected totals disagree, and that refusal is CORRECT: it means the
    benchmark does not describe the whole observed population. Swallowing it
    silently reported a partial comparison exactly like a complete one, and the
    same silence would have hidden a genuine programming error. The mismatch is
    now detected and NAMED before scipy is called, and anything still escaping
    is logged rather than discarded.
    """
    groups = sorted(g for g, e in expected.items() if e is not None and e > 0)
    if len(groups) < 2:
        return None, None, "fewer than two benchmark categories carry positive expected mass"

    obs_counts = [observed.get(g, 0.0) * n_total for g in groups]
    exp_counts = [expected[g] * n_total for g in groups]

    obs_total = sum(obs_counts)
    exp_total = sum(exp_counts)
    if exp_total <= 0:
        return None, None, "the benchmark carries no expected mass"

    # scipy requires sum(f_obs) == sum(f_exp) for a goodness-of-fit test.
    # They differ here exactly when part of the data falls outside the
    # benchmark's categories, and a partial comparison is not a verdict on the
    # whole population.
    if abs(obs_total - exp_total) > max(1e-6, 1e-6 * exp_total):
        uncovered = max(0.0, 1.0 - (obs_total / n_total)) if n_total else 0.0
        # THE REASON HAS TO BE THE REAL ONE. BGL-F5 2026-09-30. The totals also
        # disagree when the BENCHMARK is not a distribution, and this branch
        # attributed that to uncovered rows regardless. Measured before, 300 rows
        # 150 'm' / 150 'f' against {'m': 50.0, 'f': 50.0}: the published reason
        # was "0.0% of the rows fall in groups the benchmark never mentions, so
        # observed and expected totals cannot be compared", quoting the 0.0% it had
        # just computed while the actual cause was expected mass summing to 100. A
        # reader given that reason goes looking for a missing benchmark category
        # and finds none. Naming both quantities is what lets them tell the two
        # apart; `uncovered` near zero IS the tell.
        expected_mass = exp_total / n_total if n_total else 0.0
        if uncovered <= 1e-9:
            return (
                None,
                None,
                f"the benchmark's expected shares sum to {expected_mass:.3f}, not 1.0, so "
                f"it is not a population distribution and its expected total "
                f"({exp_total:.1f}) cannot be compared against the observed total "
                f"({obs_total:.1f}); every observed row IS covered by a benchmark category",
            )
        return (
            None,
            None,
            f"{uncovered:.1%} of the rows fall in groups the benchmark never "
            f"mentions, so observed and expected totals cannot be compared "
            f"(the benchmark's expected shares sum to {expected_mass:.3f})",
        )

    try:
        chi2, pval = stats.chisquare(obs_counts, f_exp=exp_counts)
    except Exception:
        logging.getLogger(__name__).warning(
            "chi-squared representation test failed unexpectedly; reporting "
            "could-not-check rather than a verdict",
            exc_info=True,
        )
        return None, None, "the chi-squared computation failed unexpectedly"

    return float(chi2), float(pval), None


def _canonical_group(group: Any, aliases: Dict[str, str]) -> str:
    """Fold a label to its benchmark category: normalise, then resolve the alias."""
    folded = _group_key(group)
    return aliases.get(folded, folded)


def _select_default_scheme(
    schemes: Tuple[_BenchmarkScheme, ...],
    series: "pd.Series",
) -> Optional[_BenchmarkScheme]:
    """Pick the built-in scheme whose categories the data's own labels land in.

    Coverage is measured as the SHARE OF ROWS whose label folds onto one of the
    scheme's categories, not as a count of distinct labels, so a scheme is
    chosen for describing the population rather than for matching a rare label.

    Returns None when no scheme covers a single row. That is the honest answer,
    and it matters: the absent-group loop reports every benchmark category
    missing from the data as a missing demographic group, so applying a
    benchmark whose categories the column never uses (a gender column spelled
    "0"/"1", an age column in raw years) manufactures a full slate of phantom
    populations with deficits in the thousands. No benchmark is applied, the
    caller is told, and the result says so via benchmark_source=None.
    """
    values = series.dropna()
    if values.empty:
        return None

    shares = (values.value_counts() / len(values)).to_dict()

    best: Optional[_BenchmarkScheme] = None
    best_coverage = 0.0
    for scheme in schemes:
        coverage = sum(
            share
            for label, share in shares.items()
            if _canonical_group(label, scheme.aliases) in scheme.distribution
        )
        if coverage > best_coverage + 1e-12:
            best, best_coverage = scheme, coverage

    return best


def _get_default_benchmark(
    attribute: str,
    series: "pd.Series",
) -> Optional[_BenchmarkScheme]:
    """Get the default benchmark scheme for a common attribute name, if one fits."""
    attr_lower = attribute.lower()

    for key, schemes in _DEFAULT_BENCHMARK_SCHEMES.items():
        if key in attr_lower or attr_lower in key:
            scheme = _select_default_scheme(schemes, series)
            if scheme is None:
                warnings.warn(
                    f"The built-in US Census benchmark for '{attribute}' "
                    f"describes none of the labels this column uses, so NO "
                    f"population benchmark was applied and representation is "
                    f"UNBENCHMARKED rather than adequate. Pass "
                    f"benchmarks={{'{attribute}': {{...}}}} for your actual "
                    f"population.",
                    UserWarning,
                    stacklevel=3,
                )
            return scheme

    return None


def _determine_severity(
    underrepresented: List[Dict],
    representation_ratios: Dict[str, float],
    overrepresented: Optional[List[Dict]] = None,
) -> RepresentationSeverity:
    """Determine overall severity of representation bias."""
    # No ratios at all means nothing was compared, and ADEQUATE is a verdict.
    # RepresentationSeverity.INSUFFICIENT_DATA exists in this very enum for the
    # case ("Too few rows for any verdict") and the main entry point already
    # uses it; this helper answered ADEQUATE instead. Same shape as the
    # OVERREPRESENTED value that used to be unreachable here.
    if not representation_ratios:
        return RepresentationSeverity.INSUFFICIENT_DATA

    for ug in underrepresented:
        if ug["ratio"] < 0.5:
            return RepresentationSeverity.CRITICAL

    for ug in underrepresented:
        if ug["ratio"] < 0.7:
            return RepresentationSeverity.HIGH

    for ug in underrepresented:
        if ug["ratio"] < 0.8:
            return RepresentationSeverity.MEDIUM

    if underrepresented:
        return RepresentationSeverity.LOW

    # No underrepresentation: an overrepresentation-only skew must reach
    # OVERREPRESENTED, not ADEQUATE (previously this enum value was
    # unreachable and the skew was rated fine).
    if overrepresented:
        return RepresentationSeverity.OVERREPRESENTED

    return RepresentationSeverity.ADEQUATE


def _generate_representation_recommendations(
    severity: RepresentationSeverity,
    underrepresented: List[Dict],
    overrepresented: List[Dict],
) -> List[str]:
    """Generate recommendations based on representation analysis."""
    recommendations = []

    if severity == RepresentationSeverity.CRITICAL:
        recommendations.append(
            "CRITICAL: Severe underrepresentation detected. Model predictions "
            "will likely be unreliable for underrepresented groups."
        )
        recommendations.append(
            "Consider: (1) Collecting more data for underrepresented groups, "
            "(2) Using resampling techniques, (3) Applying fairness constraints."
        )

    elif severity == RepresentationSeverity.HIGH:
        recommendations.append(
            "HIGH PRIORITY: Significant representation gap. Validate model "
            "performance separately for underrepresented groups."
        )
        recommendations.append("Consider targeted data collection or oversampling strategies.")

    elif severity == RepresentationSeverity.MEDIUM:
        recommendations.append(
            "Monitor performance for underrepresented groups and consider "
            "stratified sampling for validation."
        )

    elif severity == RepresentationSeverity.LOW:
        recommendations.append(
            "Slight representation gap detected. Continue monitoring group-level "
            "performance metrics and track representation ratios over time."
        )

    elif severity == RepresentationSeverity.ADEQUATE:
        recommendations.append(
            "Representation is within acceptable bounds. Continue monitoring "
            "representation ratios in future data refreshes to ensure stability."
        )

    elif severity == RepresentationSeverity.OVERREPRESENTED:
        recommendations.append(
            "One or more groups are overrepresented relative to the benchmark. "
            "Consider whether this skew may bias model behavior toward majority patterns."
        )

    elif severity == RepresentationSeverity.INSUFFICIENT_DATA:
        # Every other severity puts a line here, so this one returned an EMPTY
        # list and the reader saw nothing at all beside a verdict of
        # insufficient_data. An empty recommendation list is what an untroubled
        # result looks like, which is the same could-not-check being read as a
        # clean bill that this module removes everywhere else. Say it instead.
        recommendations.append(
            "Representation was NOT assessed for this attribute: no expected "
            "proportion could be applied to any group, so no representation "
            "ratio was computed. Group shares are reported as measured; the "
            "absence of an under/overrepresentation verdict is not a finding "
            "of adequate representation. Supply a population benchmark "
            "covering these groups to obtain one."
        )

    if underrepresented:
        groups = [ug["group"] for ug in underrepresented[:3]]
        recommendations.append(f"Underrepresented groups requiring attention: {', '.join(groups)}")

    if overrepresented:
        recommendations.append(
            "Overrepresented groups may dominate model behavior. Consider "
            "downsampling or weighted training."
        )

    return recommendations


def _analyze_intersectional(
    df: pd.DataFrame,
    attributes: List[str],
    *,
    min_group_size: int,
    underrepresentation_threshold: float,
) -> List[Dict[str, Any]]:
    """Analyze intersectional groups for representation bias.

    ``min_group_size`` does NOT decide whether a group is reported here.

    R-11, 2026-09-10. It used to: the loop below opened with
    ``if count < min_group_size: continue``, and in an INTERSECTIONAL analysis
    smallness IS the signal. Measured on counts white_male 332, black_male 332,
    white_female 331, black_female 5, at the default min_group_size=30::

        [{'intersection': 'white_male', 'ratio': 0.754, 'severity': 'high'}]

    and no warning at all. The same call at min_group_size=5 returned
    ``black_female`` first, ratio 0.044, graded ``critical`` by this function's
    own rule. So the default dropped the group sitting at 4.4% of its expected
    share and handed the reader a single finding naming the MAJORITY group,
    with nothing to say anything had been skipped: the exact opposite of the
    truth, presented as the whole of it.

    A representation ratio is arithmetic on two counts. It needs no minimum n
    to be valid, and it is reported for every group. What a small count does
    remove is any claim that the share GENERALISES beyond this sample, and that
    claim is withheld in its own field (``sampling_support``) rather than by
    deleting the finding. Every exclusion this function still makes is stated
    in a warning.
    """
    findings = []
    #: Groups seen, and groups whose share rests on fewer rows than
    #: *min_group_size*. Counted so the warning can state both numbers.
    n_groups = 0
    n_small = 0

    try:
        intersect_col = df[attributes].apply(lambda row: "_".join(str(v) for v in row), axis=1)

        n_total = len(intersect_col.dropna())
        value_counts = intersect_col.value_counts()

        # Calculate expected proportions (assuming independence)
        marginal_props = {}
        for attr in attributes:
            marginal_props[attr] = (df[attr].value_counts() / len(df)).to_dict()

        for group, count in value_counts.items():
            n_groups += 1
            small = count < min_group_size
            if small:
                n_small += 1

            actual_prop = count / n_total

            # Calculate expected under independence
            group_parts = str(group).split("_")
            if len(group_parts) != len(attributes):
                continue

            expected_prop = 1.0
            for attr, part in zip(attributes, group_parts):
                marginal = marginal_props.get(attr, {})
                part_prop = marginal.get(part, 0)
                if part_prop > 0:
                    expected_prop *= part_prop
                else:
                    expected_prop = 0
                    break

            if expected_prop > 0:
                ratio = actual_prop / expected_prop

                if ratio < underrepresentation_threshold or ratio < 0.5:
                    findings.append(
                        {
                            "intersection": group,
                            "attributes": attributes,
                            "ratio": ratio,
                            "actual_proportion": actual_prop,
                            "expected_proportion": expected_prop,
                            "count": count,
                            # Graded from the ratio, which is a deterministic
                            # share comparison and stands at any n.
                            "severity": "critical" if ratio < 0.5 else "high",
                            # The INFERENTIAL half, withheld separately. Three
                            # states are not needed here because the count is
                            # always known: it is "adequate" or it is the
                            # library's could-not-check word, and it never
                            # silently downgrades the severity beside it.
                            "sampling_support": NOT_ASSESSED if small else "adequate",
                            "min_group_size": int(min_group_size),
                        }
                    )

    except Exception:
        # A DEBUG log is invisible by default, and the empty list this then
        # returns is the same value the analysis returns when it ran and found
        # nothing: a crash and a clean intersectional result were reported
        # identically. Same shape as the swallowed chi-squared above, so it is
        # said out loud in the same way. The partial findings collected before
        # the failure are still returned, and the caller is told they are
        # partial rather than complete.
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)
        warnings.warn(
            f"Intersectional representation analysis for "
            f"{'/'.join(attributes)} failed part way through; "
            f"{len(findings)} finding(s) collected before the failure are "
            f"returned and the rest are UNASSESSED, not clean.",
            UserWarning,
            stacklevel=2,
        )

    if n_small:
        warn_not_assessed(
            "_analyze_intersectional",
            measured=n_groups - n_small,
            total=n_groups,
            unit=(
                f"intersectional group(s) of {'/'.join(attributes)} reached "
                f"min_group_size={min_group_size}"
            ),
            requirement=(
                "a share resting on fewer rows than that supports no claim about the "
                "population it was drawn from"
            ),
            reporting=(
                f"every group's measured share ratio, with sampling_support="
                f"'{NOT_ASSESSED}' on the {n_small} smaller one(s)"
            ),
            instead_of="dropping those groups from the findings entirely",
            stacklevel=2,
        )

    # Ascending ratio, so the WORST-represented intersections come first and the
    # cap below can only ever drop the mildest. The cap is still an exclusion,
    # and this function no longer makes any exclusion silently.
    findings.sort(key=lambda x: x.get("ratio", 1.0))
    if len(findings) > 10:
        warnings.warn(
            f"Intersectional representation analysis for {'/'.join(attributes)} "
            f"found {len(findings)} under-represented intersection(s); only the 10 "
            f"worst-represented are returned. The remaining "
            f"{len(findings) - 10} are UNREPORTED, not clean.",
            UserWarning,
            stacklevel=2,
        )

    return findings[:10]


# Alpha used by compare_to_benchmark()'s significance verdict. That function
# takes no `significance_level` argument and no caller in this module supplies
# one, so the alpha genuinely is fixed here. It is named rather than written as
# a bare 0.05 inside the verdict so the docstring, the returned
# `significance_level` key and the comparison cannot drift apart. This is NOT
# the knob detect_representation_bias() exposes: that one threads the caller's
# `significance_level` down its own path and never reaches this function.
_BENCHMARK_SIGNIFICANCE_LEVEL = 0.05


def compare_to_benchmark(
    df: pd.DataFrame,
    attribute: str,
    benchmark: Dict[str, float],
    *,
    visualize: bool = False,
) -> Dict[str, Any]:
    """
    Compare dataset distribution to a specific benchmark.

    Args:
        df: DataFrame to analyze
        attribute: Column name to analyze
        benchmark: Expected distribution {'group': proportion, ...}
        visualize: Whether to return visualization data

    Returns:
        Dictionary with comparison results. ``representation_ratios`` and
        ``gaps`` are reported only for groups whose ratio could be computed, and
        ``groups_not_measured`` names every group they leave out, whichever
        reason it was. ``measurement_status`` is three-state:

        - ``"could_not_measure"`` when no ratio exists at all, which is what the
          column carrying no non-null row produces (no group share to divide);
        - ``"partial"`` when some group has no positive finite expected share to
          divide by, whether the benchmark gives it 0, a negative, a NaN, or
          never mentions it;
        - ``"measured"`` only when every group seen got a ratio.

        A group absent from rows that WERE observed still gets a measured ratio
        of 0, which is a finding, not a could-not-check.

        ``pvalue`` and ``statistically_significant`` are three-state:

        - a float / ``True`` / ``False`` when the chi-squared test ran;
        - ``None`` for BOTH when it could not run (fewer than two benchmark
          categories carry positive expected mass, or part of the data falls in
          groups the benchmark never mentions so observed and expected totals
          cannot be compared), which is a could-not-check, not a clean result.
          The returned warning names which of those it was.

        ``statistically_significant`` compares ``pvalue`` against a FIXED alpha
        of 0.05, reported back as ``significance_level``. This function has no
        ``significance_level`` parameter, so a caller cannot change it here;
        ``detect_representation_bias(significance_level=...)`` is a separate
        entry point and does not affect this one.

    Example:
        >>> benchmark = {'male': 0.5, 'female': 0.5}
        >>> result = compare_to_benchmark(df, 'gender', benchmark)
        >>> print(f"Chi-squared p-value: {result['pvalue']:.4f}")
    """
    if attribute not in df.columns:
        return {"error": f"Column '{attribute}' not found"}

    series = df[attribute].dropna()
    n_total = len(series)

    value_counts = series.value_counts()
    actual = (value_counts / n_total).to_dict()
    # SUMMING on a collision, never overwriting: see _fold_shares for the measured
    # before-state. Two spellings of one group are one group, on both sides.
    actual_normalized, actual_folded = _fold_shares(actual)
    benchmark_normalized, benchmark_folded = _fold_shares(benchmark)
    labels_folded_together = {
        "observed": actual_folded,
        "benchmark": benchmark_folded,
    }
    if actual_folded or benchmark_folded:
        # Disclosed, because folding is a real transformation of the reader's own
        # labels: actual_distribution still shows 'm' and 'M' separately while the
        # ratios answer to one key, and a reader comparing the two has to be able to
        # see why. Not a warning about a defect, a statement of what was done.
        warnings.warn(
            f"compare_to_benchmark: for '{attribute}', label(s) differing only in case "
            f"or surrounding space were treated as ONE group and their shares ADDED: "
            f"{ {k: v for k, v in list(actual_folded.items()) + list(benchmark_folded.items())} }. "
            f"actual_distribution keeps the original spellings; representation_ratios "
            f"and gaps are keyed on the folded label. See "
            f"result['labels_folded_together'].",
            UserWarning,
            stacklevel=2,
        )

    # THE BENCHMARK MUST BE A DISTRIBUTION, not merely a mapping of divisible
    # numbers, and this was the SECOND of the three entry points that consume a
    # caller-supplied benchmark to be missing the check. BGL-F5 2026-09-30.
    # Measured before, 300 rows 150 'm' / 150 'f': benchmark {'m': 50.0, 'f': 50.0}
    # published representation_ratios {'f': 0.01, 'm': 0.01} and gaps {'f': -49.5,
    # 'm': -49.5} with measurement_status 'measured' and groups_not_measured [],
    # i.e. "nothing went unmeasured" beside a "gap" of -49.5 between two
    # proportions, a quantity bounded in [-1, 1]. {'m': 2.0, 'f': 0.5} did the same
    # with a gap of -1.5. The THIRD entry point already had the guard:
    # detect_representation_bias renormalises with a warning at the site whose
    # comment reads "otherwise perfectly balanced data gets flagged with phantom
    # deficits", and _validate_default_schemes refuses a BUILT-IN table that does
    # not sum to 1.0 at import BECAUSE, in its own words, "a table whose values
    # summed to 3.000 was consumed as a probability distribution and graded a
    # perfectly representative dataset CRITICAL. Nothing checked it". That is this
    # defect, shipped once already, and it was still open on this path.
    #
    # Renormalise rather than refuse, so the three entry points agree on the same
    # input and a percentage-scaled benchmark recovers its real answer (ratios
    # 1.0 / 1.0 here) instead of losing the comparison. The transformation is
    # disclosed: a warning, plus benchmark_renormalised_from and
    # benchmark_distribution_used on the result, because the caller's own numbers
    # are not what was divided by. The 0.01 tolerance is the sibling's.
    benchmark_mass = _positive_share_mass(benchmark_normalized)
    benchmark_renormalised_from: Optional[float] = None
    if benchmark_mass > 0 and abs(benchmark_mass - 1.0) > 0.01:
        benchmark_renormalised_from = benchmark_mass
        warnings.warn(
            f"compare_to_benchmark: benchmark proportions for '{attribute}' sum to "
            f"{benchmark_mass:.3f}, not 1.0, so they are not a population distribution; "
            f"renormalising by that sum before any ratio or gap is taken. Without this, the "
            f"'gap' between two proportions lands outside the [-1, 1] range a gap can "
            f"occupy and the ratios are not representation ratios. The caller's own numbers are "
            f"kept in benchmark_distribution; what was divided by is in "
            f"benchmark_distribution_used, and benchmark_renormalised_from carries the sum.",
            UserWarning,
            stacklevel=2,
        )
        benchmark_normalized = {
            k: (float(v) / benchmark_mass if _is_a_divisible_share(v) else v)
            for k, v in benchmark_normalized.items()
        }

    # Same join hazard as _analyze_single_attribute, same fix at the source:
    # this result carries `actual_distribution` under the data's own labels and
    # the ratios under normalised ones, so a caller reading the two together
    # needs the ratios to answer to either form.
    representation_ratios = _GroupRatios()
    gaps = _GroupRatios()

    # BGL3 2026-09-27. `actual_normalized.get(group, 0)` is the right answer for
    # a group absent from 300 OBSERVED rows (it really does hold 0% of them),
    # and the wrong answer when there are no observed rows at all: measured on
    # 50 rows of gender=None against a 50/50 benchmark, this returned
    # sample_size 0 with representation_ratios {'f': 0.0, 'm': 0.0} and gaps
    # {'f': -0.5, 'm': -0.5}, four graded numbers computed from nothing. A ratio
    # of 0 is not a small ratio, it is the severest finding this scale carries,
    # and `calculate_representation_ratio` one function below already refuses
    # the identical case with None and a warning. No share exists, so no ratio
    # and no gap is reported, and the groups they would have covered are named.
    groups_seen = sorted(set(actual_normalized.keys()) | set(benchmark_normalized.keys()))
    # BGL5 2026-09-27. `groups_not_measured` was `[] if measured else groups_seen`
    # and `measured` was `n_total > 0`, so the field promised in the comment below
    # ("the groups no ratio was reported for named either way") reported NOTHING
    # unmeasured for every input holding at least one observed row. Measured
    # before, on 300 observed rows against {'m': 0.5, 'f': 0.5, 'x': 0.0}:
    # representation_ratios {'f': 1.0, 'm': 1.0}, gaps {'f': 0.0, 'm': 0.0},
    # measurement_status 'measured', groups_not_measured [] and no warning, with
    # 'x' silently absent from both dicts because `expected_prop > 0` skipped it;
    # identical for -0.1 and for NaN, and for an OBSERVED group the benchmark
    # never mentions (140 m / 140 f / 20 nb against a two-group benchmark left
    # 'nb' out of both dicts with groups_not_measured []). After: 'x' and 'nb' are
    # named in groups_not_measured with measurement_status 'partial' and a
    # warning, while the 300-row control keeps 'measured', [] and its real ratios.
    groups_not_measured: List[str] = []
    measured = n_total > 0
    if not measured:
        warnings.warn(
            f"No non-null rows for '{attribute}', so no group share exists: the "
            f"representation ratios and gaps for {groups_seen or ['no group']} are "
            f"UNMEASURED and are not reported. That is a could-not-check, NOT a "
            f"ratio of 0, which would read as the group being wholly absent from "
            f"the population.",
            UserWarning,
            stacklevel=2,
        )
        groups_not_measured = list(groups_seen)
    else:
        for group in groups_seen:
            actual_prop = actual_normalized.get(group, 0)
            expected_prop = benchmark_normalized.get(group, 0)

            # _finite_ratio, not `_is_a_divisible_share` then divide: the divisor
            # being positive and finite does not make the QUOTIENT finite. Measured
            # before on 300 rows against a subnormal expected share of 5e-324: the
            # ratio published was inf. BGL6 F12, 2026-09-29.
            ratio = _finite_ratio(actual_prop, expected_prop)
            if ratio is not None:
                representation_ratios[group] = ratio
                gaps[group] = actual_prop - expected_prop
            else:
                # No expected share to divide by (absent from the benchmark, zero,
                # negative or NaN), or one so small that the quotient leaves the
                # float range, so this group has no ratio and no gap. Named, never
                # silently dropped.
                groups_not_measured.append(group)
        if groups_not_measured:
            warnings.warn(
                f"compare_to_benchmark: no representation ratio or gap exists for "
                f"{groups_not_measured} of '{attribute}', because the benchmark gives "
                f"each of them no expected share this can be divided by, being absent, "
                f"zero, negative, NaN or so small that the quotient is not finite "
                f"({ {g: benchmark_normalized.get(g, 'absent from the benchmark') for g in groups_not_measured} }). "
                f"Their absence from representation_ratios and gaps is a "
                f"could-not-check, NOT a ratio of 0. measurement_status is "
                f"'partial'; see result['groups_not_measured'].",
                UserWarning,
                stacklevel=2,
            )

    chi2_stat, chi2_pval, chi2_reason = _chi_squared_representation_test(
        actual_normalized, benchmark_normalized, n_total
    )

    # `if chi2_pval` was a FALSY test, not a None test. A chi-squared p-value of
    # exactly 0.0 is the STRONGEST possible evidence that the observed
    # distribution differs from the benchmark, and it underflows to a hard 0.0
    # well inside the range real data reaches (3000 male / 1 female against a
    # 50/50 benchmark gives chi2 = 2997.0, p = 0.0). Being falsy, it took the
    # else branch and reported None, which every consumer reads as "could not be
    # determined": the strongest evidence of bias and no measurement at all
    # produced the SAME output. `is not None` keeps the genuine could-not-check
    # state (the test did not run, so p is None) reporting None, and lets a
    # measured 0.0 be graded as what it is.
    if chi2_pval is None:
        warnings.warn(
            f"Chi-squared representation test did not run for '{attribute}' "
            f"({chi2_reason}); pvalue and "
            f"statistically_significant are None, which is UNMEASURED, not "
            f"'not significant'.",
            UserWarning,
            stacklevel=2,
        )

    result = {
        "attribute": attribute,
        "sample_size": n_total,
        "actual_distribution": actual,
        "benchmark_distribution": benchmark,
        # What was actually divided by, and the sum it was divided by, or None when
        # the caller's benchmark was already a distribution and nothing was
        # rescaled. None here is a measurement ("no renormalisation happened"), not
        # a silence. BGL-F5 2026-09-30.
        "benchmark_distribution_used": dict(benchmark_normalized),
        "benchmark_renormalised_from": benchmark_renormalised_from,
        "representation_ratios": representation_ratios,
        "gaps": gaps,
        "chi_squared_statistic": chi2_stat,
        "pvalue": chi2_pval,
        "statistically_significant": (
            chi2_pval < _BENCHMARK_SIGNIFICANCE_LEVEL if chi2_pval is not None else None
        ),
        "significance_level": _BENCHMARK_SIGNIFICANCE_LEVEL,
        # Three states on the surface a reader reads, beside the ratios rather
        # than inferred from `sample_size`: "measured", "partial" (some group got
        # no ratio) or "could_not_measure" (none did), with the groups no ratio
        # was reported for named in every one of the three.
        "measurement_status": (
            "could_not_measure"
            if (not measured or len(groups_not_measured) == len(groups_seen))
            else ("partial" if groups_not_measured else "measured")
        ),
        "groups_not_measured": groups_not_measured,
        # Which raw labels were treated as one group, on each side. Empty on both
        # sides means no label was folded into another, which is a measurement and
        # not a silence. BGL6 F12, 2026-09-29.
        "labels_folded_together": labels_folded_together,
    }

    if visualize:
        # Build the group list first: the previous dict literal referenced
        # result['visualization_data']['groups'] inside its own construction,
        # which raised KeyError on every visualize=True call.
        viz_groups = sorted(set(actual_normalized.keys()) | set(benchmark_normalized.keys()))
        result["visualization_data"] = {
            "groups": viz_groups,
            "actual": [actual_normalized.get(g, 0) for g in viz_groups],
            "expected": [benchmark_normalized.get(g, 0) for g in viz_groups],
        }

    return result


def calculate_representation_ratio(
    df: pd.DataFrame,
    attribute: str,
    target_group: str,
    benchmark_proportion: float,
    *,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """
    Calculate representation ratio for a specific group.

    Args:
        df: DataFrame to analyze
        attribute: Column name
        target_group: Group to analyze
        benchmark_proportion: Expected proportion (0-1)
        min_group_size: Fewest non-null rows the ratio may be computed from.
            Below it the observed proportion is noise, and this function reports
            UNMEASURED rather than a graded verdict, matching
            ``_analyze_single_attribute`` in this module, which refuses the same
            input with ``RepresentationSeverity.INSUFFICIENT_DATA``. Pass a lower
            number to grade a smaller sample deliberately.

    Returns:
        Dictionary with ratio and related metrics. ``actual_proportion``,
        ``representation_ratio``, ``deficit_count``, ``is_underrepresented`` and
        ``is_overrepresented`` are three-state: real values when the ratio could
        be computed, and ``None`` when it could not. ``None`` means UNMEASURED,
        not "adequately represented" and not "underrepresented". It is reported
        when there are no non-null rows, when there are fewer than
        ``min_group_size`` of them, when ``min_group_size`` is not a number a row
        count can be compared against (so the floor could not be applied at all),
        when ``benchmark_proportion`` is not a positive finite number, when it is
        ABOVE the 0-1 share scale (it is then not an expected proportion, and the
        ratio it produces reports underrepresentation for every possible dataset),
        and when ``target_group`` matched no row although a type-tolerant
        comparison of the same label does match some.

        ``measurement_status`` is ``"measured"`` or ``"could_not_measure"`` and
        ``not_measured_reason`` says which case it was. ``label_status`` is
        ``"matched"``, ``"absent_from_observed_rows"`` (a MEASURED zero: the
        label is a level this column could hold and no row carries it) or
        ``"did_not_match_any_level"`` (UNMEASURED: the label matched nothing as
        written), and ``observed_levels`` lists the levels the column does hold
        so a reader can see a label error rather than infer a finding from it.
    """
    if attribute not in df.columns:
        return {"error": f"Column '{attribute}' not found"}

    series = df[attribute].dropna()
    n_total = len(series)

    # Normalize for matching
    target_normalized = _group_key(target_group)
    series_normalized = series.astype(str).str.lower().str.strip()

    actual_count = int((series_normalized == target_normalized).sum())

    # Siblings of the same defect as the significance verdict above: both of
    # these divisions used to fall back to a literal 0, and 0 is then a REAL
    # ratio that gets graded. An empty column reported
    # `is_underrepresented: True` off zero rows, and a benchmark_proportion of 0
    # made a group holding HALF the dataset come back underrepresented, because
    # an undefined ratio was written down as 0.0. Neither is a measurement, so
    # neither gets a verdict.
    actual_proportion: Optional[float]
    ratio: Optional[float]
    deficit_count: Optional[int]
    not_measured_reason: Optional[str] = None

    # BGL5 2026-09-27. `actual_count == 0` carries two different meanings and this
    # function graded both the same way, with the severest value the scale has.
    # Measured before on 300 rows of a float-coded column ([1.0]*150 + [0.0]*150)
    # with target_group=1: actual_count 0, actual_proportion 0.0,
    # representation_ratio 0.0, deficit_count 150, is_underrepresented True and no
    # warning, for a group holding 150 of those 300 rows (the same call with
    # target_group=1.0 returns actual_count 150). `series.astype(str)` gives '1.0'
    # and `_group_key(1)` gives '1', so nothing matched. After: the ratio fields
    # are None, label_status 'did_not_match_any_level', and a warning naming the
    # levels the column holds. A label that matched nothing WHILE no type-tolerant
    # match exists either is still a measured 0 (0% of observed rows IS a
    # finding), now with label_status 'absent_from_observed_rows' and a warning
    # rather than silence.
    observed_levels = sorted(series_normalized.unique().tolist())[:12] if n_total else []
    label_status = "matched" if actual_count > 0 else "absent_from_observed_rows"
    if n_total > 0 and actual_count == 0:
        tolerant_count = _count_label_matches_type_tolerant(series, target_group)
        if tolerant_count > 0:
            label_status = "did_not_match_any_level"
            not_measured_reason = (
                f"the label {target_group!r} normalises to "
                f"{target_normalized!r}, which matches none of the levels "
                f"'{attribute}' holds ({observed_levels}), while a type-tolerant "
                f"comparison of the same label matches {tolerant_count} of "
                f"{n_total} row(s): the count is a label mismatch, not a "
                "measurement of the group"
            )
            warnings.warn(
                f"calculate_representation_ratio: {not_measured_reason}. Reporting "
                f"representation_ratio=None (could not check), NOT 0.0 with "
                f"is_underrepresented=True, which would read as the group being "
                f"wholly absent. See result['observed_levels'].",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"calculate_representation_ratio: no row of '{attribute}' carries "
                f"{target_group!r} (normalised {target_normalized!r}); the levels "
                f"present are {observed_levels}. The ratio of 0.0 below is a "
                f"MEASURED zero, which is the severest value this scale has, and it "
                f"is only a finding if that label is spelt the way the data spells "
                f"it.",
                UserWarning,
                stacklevel=2,
            )

    if n_total <= 0:
        warnings.warn(
            f"No non-null rows for '{attribute}'; the representation ratio for "
            f"'{target_group}' is UNMEASURED (None), not zero.",
            UserWarning,
            stacklevel=2,
        )
        actual_proportion = None
        not_measured_reason = f"'{attribute}' has no non-null row to take a share of"
    elif not _is_a_readable_floor(min_group_size):
        # THE GATE'S OWN OPERAND, BGL-F5 2026-09-30. The floor below is
        # `n_total < min_group_size`, and EVERY comparison against a NaN is False,
        # so a NaN floor did not raise it, did not refuse anything and did not say
        # a word: it silently switched the sample-size gate OFF. Measured before on
        # the same 300 rows, min_group_size=float('nan'): measurement_status
        # 'measured', representation_ratio 1.0, ZERO warnings, identical output to
        # min_group_size=30, while min_group_size=inf correctly refused. 0 and a
        # negative are NOT this case: they are a caller deliberately grading a
        # small sample, which the parameter's own docstring invites.
        warnings.warn(
            f"calculate_representation_ratio: min_group_size={min_group_size!r} is not a "
            f"number this gate can compare {n_total} row(s) against, so the sample-size "
            f"floor could not be applied at all. The representation ratio for "
            f"'{target_group}' is reported as UNMEASURED (None) rather than graded past a "
            f"gate that silently did not run. Pass a real number (0 to grade any sample "
            f"size deliberately).",
            UserWarning,
            stacklevel=2,
        )
        actual_proportion = None
        not_measured_reason = (
            f"min_group_size={min_group_size!r} is not a comparable number, so the "
            f"sample-size floor could not be applied and the ratio was not graded"
        )
    elif n_total < min_group_size:
        # The same gate `_analyze_single_attribute` applies in this module, which
        # refuses below min_group_size with "UNASSESSED, not adequate" and
        # RepresentationSeverity.INSUFFICIENT_DATA. Measured before on a ONE-row
        # frame: representation_ratio 0.0 with is_underrepresented True and no
        # warning at all, a graded verdict off one row from the entry point that
        # did not have the gate.
        warnings.warn(
            f"calculate_representation_ratio: only {n_total} non-null row(s) for "
            f"'{attribute}' (min_group_size={min_group_size}); the representation "
            f"ratio for '{target_group}' is UNASSESSED (None), not adequate and not "
            f"underrepresented. Pass min_group_size lower to grade it anyway.",
            UserWarning,
            stacklevel=2,
        )
        actual_proportion = None
        not_measured_reason = (
            f"only {n_total} non-null row(s), and the ratio needs min_group_size={min_group_size}"
        )
    elif label_status == "did_not_match_any_level":
        actual_proportion = None
    else:
        actual_proportion = actual_count / n_total

    # THE QUOTIENT, NOT THE DIVISOR, BGL6 F12, 2026-09-29. The test below was
    # `_is_a_divisible_share(benchmark_proportion)`, which admits every positive
    # finite value including the subnormals, and the division was then performed
    # unchecked. Measured before on 300 rows with benchmark_proportion=5e-324:
    # representation_ratio inf, is_overrepresented True, measurement_status
    # 'measured', not_measured_reason None and ZERO warnings, from the function
    # whose docstring promises None "when benchmark_proportion is not a positive
    # finite number". An infinite ratio is a could-not-check, and by far the
    # loudest-reading one this scale can carry: it grades as the most extreme
    # overrepresentation there is.
    #
    # AND THE SCALE, NOT ONLY THE QUOTIENT. BGL-F5 2026-09-30. `_finite_ratio`
    # answers "can I divide by this and get a finite number", which every positive
    # finite value satisfies, and nothing anywhere tested that benchmark_proportion
    # is a SHARE. Above 1.25 the published verdict stopped being a function of the
    # data at all: actual_proportion cannot exceed 1.0, so `ratio < 0.8` is True
    # for every possible dataset and `ratio > 1.2` can never fire above 0.8333. See
    # _is_on_the_share_scale for the measured before-state (bp=1e9 published
    # deficit_count 299999999850 on a 300-row frame, 'measured', zero warnings).
    benchmark_off_the_share_scale = _is_a_divisible_share(
        benchmark_proportion
    ) and not _is_on_the_share_scale(benchmark_proportion)
    ratio = (
        None
        if actual_proportion is None or benchmark_off_the_share_scale
        else _finite_ratio(actual_proportion, benchmark_proportion)
    )
    if actual_proportion is None:
        ratio = None
        deficit_count = None
    elif ratio is not None:
        deficit_count = int((benchmark_proportion - actual_proportion) * n_total)
    else:
        # `benchmark_proportion > 0` was True for float('inf'), which then reached
        # `int(inf * n)`: measured before on 300 rows, benchmark float('inf')
        # raised OverflowError: cannot convert float infinity to integer, from a
        # function whose docstring promises None for a benchmark that cannot be
        # divided by. A NaN benchmark took the other side of the same hole: it is
        # not > 0, so it landed here already, and now says so by name.
        if benchmark_off_the_share_scale:
            # FIRST, because a value above the scale IS a divisible share and would
            # otherwise be handed the subnormal branch's reason, which states a
            # different fact and would send a reader to fix the wrong thing.
            not_measured_reason = (
                f"benchmark_proportion={benchmark_proportion!r} is above the 0-1 share "
                f"scale this parameter is documented on, so it is not an expected "
                f"PROPORTION and what it produces is not a representation ratio: any "
                f"observed share divided by it is below 0.8, which reports "
                f"underrepresentation for every possible dataset"
            )
            warnings.warn(
                f"calculate_representation_ratio: {not_measured_reason}. Reported as "
                f"UNMEASURED (None) for '{target_group}', NOT as the ratio "
                f"{_finite_ratio(actual_proportion, benchmark_proportion)!r} with a deficit of "
                f"{int((float(benchmark_proportion) - float(actual_proportion)) * n_total)} "
                f"people in a {n_total}-row frame. If the benchmark is a percentage, divide "
                f"it by 100; if it is a head count, divide it by the population.",
                UserWarning,
                stacklevel=2,
            )
        elif _is_a_divisible_share(benchmark_proportion):
            # Positive and finite, and still not divisible by: named separately,
            # because "the benchmark is not a share" and "the benchmark is a share
            # too small to divide this by" are different facts about the input and a
            # reader fixing one would not fix the other.
            not_measured_reason = (
                f"benchmark_proportion={benchmark_proportion!r} is positive and finite, "
                f"but actual_proportion={actual_proportion!r} divided by it is not a "
                f"finite number, so the ratio has no value to report"
            )
            warnings.warn(
                f"calculate_representation_ratio: {not_measured_reason} for "
                f"'{target_group}'. The representation ratio is UNMEASURED and reported "
                f"as None, rather than the infinity the division produces, which grades "
                f"as the most extreme overrepresentation this scale can carry.",
                UserWarning,
                stacklevel=2,
            )
        else:
            not_measured_reason = (
                f"benchmark_proportion={benchmark_proportion!r} is not a positive "
                "finite share, so there is nothing to divide by"
            )
            # Wording preserved verbatim: three pins across BGL4, BGL5 and audit6
            # match on "not positive and finite" / "not positive". Rephrasing it
            # turned all three red without changing any behaviour.
            warnings.warn(
                f"benchmark_proportion={benchmark_proportion!r} for "
                f"'{target_group}' is not positive and finite, so the "
                f"representation ratio is undefined and reported as None rather than 0.",
                UserWarning,
                stacklevel=2,
            )
        ratio = None
        # None, matching the docstring and the ratio beside it. A "deficit"
        # measured against a non-positive benchmark is not a small deficit, it
        # is an undefined one: (0 - actual) * n is always <= 0 and reads as a
        # surplus. The docstring already promised None here and the code
        # returned a number, so the two disagreed about the same field.
        deficit_count = None

    return {
        "group": target_group,
        "actual_count": actual_count,
        "actual_proportion": actual_proportion,
        "benchmark_proportion": benchmark_proportion,
        "representation_ratio": ratio,
        "deficit_count": deficit_count,
        "is_underrepresented": None if ratio is None else ratio < 0.8,
        "is_overrepresented": None if ratio is None else ratio > 1.2,
        # Three states where a reader looks, not inferred from a None.
        "measurement_status": "measured" if ratio is not None else "could_not_measure",
        "not_measured_reason": not_measured_reason,
        "label_status": label_status,
        "observed_levels": observed_levels,
        "sample_size": n_total,
    }
