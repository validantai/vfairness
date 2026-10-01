"""
Feature Correlation Analysis for Fairness-Aware Feature Engineering.

This module provides comprehensive tools for analyzing correlations between
features and protected attributes, identifying proxy variables, and assessing
the fairness implications of feature selection.

Key Capabilities:
    1. Proxy variable identification with multiple correlation measures
    2. Feature correlation heatmaps and analysis
    3. Redundant feature detection
    4. Feature importance analysis for fairness
    5. Intersectional correlation analysis
    6. Proxy chain detection (indirect proxies)

References:
    - Barocas & Selbst (2016): Big Data's Disparate Impact
    - Datta et al. (2017): Proxy Discrimination in Data-Driven Systems
    - Kilbertus et al. (2017): Avoiding Discrimination through Causal Reasoning
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from vfairness._names import name_tokens, resolve_attribute_key
from vfairness._not_assessed import NOT_ASSESSED

# Module constants

# Minimum sample size for reliable statistical computations
MIN_SAMPLE_SIZE = 10

# Minimum sample size for proxy chain analysis (needs more samples)
CHAIN_MIN_SAMPLES = 30

# Maximum ratio of unique values to total rows (above this, likely an ID column)
MAX_CARDINALITY_RATIO = 0.5

# The fewest overlapping rows from which a Pearson coefficient can be a
# MEASUREMENT rather than arithmetic. Any two points are perfectly collinear, so
# |r| is exactly +1 or -1 for every 2-row overlap whatever the numbers are, and
# scipy's p-value for one is 1.0. `min_periods` is caller-controlled, and set to
# 2, 1 or 0 it silently turned that arithmetic into a published correlation
# (BGL7 F15, 2026-09-29). The same reasoning, and the same floor, as
# ``bias_detection.statistical._MIN_COMPARABLE_ROWS``, which the sibling screen
# there records for exactly this shape.
_PEARSON_MIN_OVERLAP = 3

# Number of bins for mutual information discretization
MI_BINS = 10

# Default correlation thresholds for risk levels
CORRELATION_THRESHOLD_CRITICAL = 0.7
CORRELATION_THRESHOLD_HIGH = 0.5
CORRELATION_THRESHOLD_MEDIUM = 0.3
CORRELATION_THRESHOLD_LOW = 0.1


class ProxyRiskLevel(Enum):
    """Risk levels for proxy variables."""

    CRITICAL = "critical"  # Very high correlation (>0.7), direct proxy
    HIGH = "high"  # High correlation (0.5-0.7)
    MEDIUM = "medium"  # Moderate correlation (0.3-0.5)
    LOW = "low"  # Weak correlation (0.1-0.3)
    NEGLIGIBLE = "negligible"  # Very weak correlation (<0.1)


class ProxyType(Enum):
    """Types of proxy relationships."""

    DIRECT = "direct"  # Direct correlation with protected attribute
    INDIRECT = "indirect"  # Correlated through intermediate variable
    INTERSECTIONAL = "intersectional"  # Proxy for intersection of attributes
    HISTORICAL = "historical"  # Proxy rooted in historical discrimination
    # The correlation is measured, but the NATURE of the relationship to THIS
    # protected attribute is not established by any evidence we hold. Never
    # collapse this into DIRECT: that is the strongest label in the enum and
    # it was previously handed out as the default (see _determine_proxy_type).
    UNCLASSIFIED = "unclassified"  # Relationship type could not be determined


class CorrelationType(Enum):
    """Types of correlation measures."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    CRAMERS_V = "cramers_v"
    MUTUAL_INFORMATION = "mutual_information"
    POINT_BISERIAL = "point_biserial"


@dataclass
class CorrelationResult:
    """
    Result of correlation analysis between a feature and protected attribute.

    Attributes:
        feature: Feature name
        protected_attribute: Protected attribute name
        correlation: Primary correlation coefficient
        correlation_type: Type of correlation measure used
        pvalue: P-value for statistical significance
        mutual_information: Mutual information score
        cramers_v: Cramér's V for categorical associations
        sample_size: Number of samples used
    """

    feature: str
    protected_attribute: str
    correlation: float
    correlation_type: CorrelationType
    pvalue: Optional[float] = None
    mutual_information: Optional[float] = None
    cramers_v: Optional[float] = None
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "protected_attribute": self.protected_attribute,
            "correlation": self.correlation,
            "correlation_type": self.correlation_type.value,
            "pvalue": self.pvalue,
            "mutual_information": self.mutual_information,
            "cramers_v": self.cramers_v,
            "sample_size": self.sample_size,
        }


@dataclass
class ProxyVariableResult:
    """
    Result of proxy variable identification.

    Attributes:
        feature: Feature identified as potential proxy
        protected_attribute: Protected attribute it correlates with
        correlation: Correlation coefficient
        correlation_type: Type of correlation measure used
        risk_level: Assessed risk level
        proxy_type: Type of proxy relationship
        mutual_information: Mutual information score (if computed)
        cramers_v: Cramér's V statistic (for categorical variables)
        pvalue: P-value for correlation significance
        sample_size: Sample size used for computation
        affected_groups: Groups that may be affected
        recommendations: Suggested actions
        evidence: Additional evidence supporting the finding
    """

    feature: str
    protected_attribute: str
    correlation: float
    correlation_type: str
    risk_level: ProxyRiskLevel
    proxy_type: ProxyType
    mutual_information: Optional[float] = None
    cramers_v: Optional[float] = None
    pvalue: Optional[float] = None
    sample_size: int = 0
    affected_groups: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "protected_attribute": self.protected_attribute,
            "correlation": self.correlation,
            "correlation_type": self.correlation_type,
            "risk_level": self.risk_level.value,
            "proxy_type": self.proxy_type.value,
            "mutual_information": self.mutual_information,
            "cramers_v": self.cramers_v,
            "pvalue": self.pvalue,
            "sample_size": self.sample_size,
            "affected_groups": self.affected_groups,
            "recommendations": self.recommendations,
            "evidence": self.evidence,
        }


class HighCorrelationResult(List[Tuple[str, str, float]]):
    """The pairs over the threshold, PLUS the pairs never measured.

    Same shape, and the same reason, as :class:`ProxyScreenResult` and
    :class:`ProxyChainResult` below: a bare ``[]`` was one token for "compared
    every pair and none reached the threshold" and for "could not compare
    them", and only the first is reassuring.

    It subclasses ``list`` on purpose so every existing consumer keeps working
    unchanged (``for f, a, v in ...``, ``len``, ``set(...)``, slicing,
    ``isinstance(x, list)``), while a caller that wants the coverage can read
    it.

    Attributes:
        pairs_not_measured: ``(feature, attribute)`` for every pair whose
            correlation is not finite, so it was never compared against the
            threshold. NOT pairs that were measured and came in under it:
            those are a finding.
        not_compared_reason: why NOTHING here was selected, when that is the
            case: a threshold the comparison cannot fire against, or a matrix
            with no pair in it at all. ``None`` when a real selection was made.
        complete: True only when every pair carried a measured correlation AND
            the selection itself was possible.
    """

    def __init__(
        self,
        results: Optional[List[Tuple[str, str, float]]] = None,
        *,
        pairs_not_measured: Optional[List[Tuple[str, str]]] = None,
        not_compared_reason: Optional[str] = None,
    ) -> None:
        super().__init__(results or [])
        self.pairs_not_measured: List[Tuple[str, str]] = list(pairs_not_measured or [])
        self.not_compared_reason: Optional[str] = not_compared_reason

    @property
    def complete(self) -> bool:
        # BGL5 2026-09-27: `not self.pairs_not_measured` alone read True over a
        # matrix with ZERO pairs in it, and True for a threshold of NaN, which
        # every comparison is False against. Both are vacuous, and complete
        # coverage of nothing is the reassuring half of a two-state answer.
        return not self.pairs_not_measured and self.not_compared_reason is None


@dataclass
class FeatureCorrelationMatrix:
    """
    Complete correlation matrix between features and protected attributes.

    Attributes:
        correlations: DataFrame of correlation values
        pvalues: DataFrame of p-values
        feature_names: List of feature names
        protected_attributes: List of protected attribute names
        method: Correlation method used
    """

    correlations: pd.DataFrame
    pvalues: pd.DataFrame
    feature_names: List[str]
    protected_attributes: List[str]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlations": self.correlations.to_dict(),
            "pvalues": self.pvalues.to_dict(),
            "feature_names": self.feature_names,
            "protected_attributes": self.protected_attributes,
            "method": self.method,
        }

    def get_high_correlations(
        self,
        threshold: float = 0.3,
    ) -> "HighCorrelationResult":
        """Get feature-attribute pairs with correlation above threshold.

        Returns a :class:`HighCorrelationResult`, a ``list`` of
        ``(feature, attribute, abs_correlation)`` triples that ALSO carries the
        pairs whose correlation was never measured. Iteration, ``len``,
        indexing and ``isinstance(x, list)`` behave exactly as before.

        A NaN cell fails ``corr >= threshold`` silently, so an unmeasured pair
        used to leave the same trace as a pair measured and found weak: none at
        all. Measured 2026-09-27 on a 5-row frame, below
        ``MIN_SAMPLE_SIZE=10``, whose only pair ``income ~ gender`` was NaN in
        ``correlations``: this returned ``[]``, byte-identical to a clean frame,
        with no warning. Same output when the protected attribute was not a
        column of the frame at all, which leaves the whole column NaN.
        """
        results = []
        not_measured: List[Tuple[str, str]] = []
        # THE THRESHOLD ITSELF, decided above the loop, because a threshold the
        # comparison cannot fire against makes every pair in the loop vacuous and
        # no per-pair test can notice. BGL5 2026-09-27, measured on a matrix
        # holding a MEASURED 0.85 for height ~ gender: threshold=0.3 returned
        # [('height','gender',0.85)]; threshold=float('nan') returned [] with
        # complete True, pairs_not_measured [] and NO warning, because every
        # comparison against NaN is False, so the strong pair was dropped in
        # exactly the silence the NaN-cell fix above was written to end.
        # threshold=2.0 is the same vacuum from the other side: no absolute
        # correlation can reach it. The sibling unit graded in the same wave
        # (GroupManager with min_group_size <= 0) already discloses a threshold
        # that disables its check.
        not_compared_reason: Optional[str] = None
        n_pairs = len(self.feature_names) * len(self.protected_attributes)
        if not _is_finite_number(threshold):
            not_compared_reason = (
                f"threshold={threshold!r} is not a finite number, and every comparison "
                f"against it is False, so none of the {n_pairs} pair(s) was selected or "
                f"rejected. This empty result is a could-not-check, not a finding that no "
                f"correlation is high."
            )
        elif float(threshold) > 1.0:
            not_compared_reason = (
                f"threshold={threshold} is above 1.0 and an absolute correlation cannot "
                f"exceed 1.0, so none of the {n_pairs} pair(s) could ever be selected. "
                f"This empty result is a could-not-check, not a finding that no "
                f"correlation is high."
            )
        elif n_pairs == 0:
            not_compared_reason = (
                f"this matrix holds {len(self.feature_names)} feature(s) and "
                f"{len(self.protected_attributes)} protected attribute(s), so there was no "
                f"pair to compare. An empty result over zero pairs is not a finding that "
                f"no correlation is high."
            )
        if not_compared_reason is None:
            for feature in self.feature_names:
                for attr in self.protected_attributes:
                    corr = abs(self.correlations.loc[feature, attr])
                    if not np.isfinite(corr):
                        not_measured.append((feature, attr))
                        continue
                    if corr >= threshold:
                        results.append((feature, attr, corr))
            if float(threshold) < 0.0:
                # Not a could-not-check: every measured pair IS above a negative
                # threshold, so the numbers are real. But the LIST is then not a
                # selection, and a caller reading it as "the high ones" is
                # reading every pair the matrix holds.
                warnings.warn(
                    f"FeatureCorrelationMatrix.get_high_correlations: threshold="
                    f"{threshold} is negative, so every one of the {len(results)} measured "
                    f"pair(s) is above it and this list is not a selection of high "
                    f"correlations.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Nothing was compared, so no pair carries a measured comparison.
            not_measured = [
                (feature, attr)
                for feature in self.feature_names
                for attr in self.protected_attributes
            ]
            warnings.warn(
                f"FeatureCorrelationMatrix.get_high_correlations: {not_compared_reason} "
                f"See result.not_compared_reason and result.complete.",
                UserWarning,
                stacklevel=2,
            )
        out = HighCorrelationResult(
            sorted(results, key=lambda x: -x[2]),
            pairs_not_measured=not_measured,
            not_compared_reason=not_compared_reason,
        )
        # `not_compared_reason is None` keeps the two disclosures apart: the
        # sentence below says the pair carries no measured correlation, which is
        # false when the coefficients are fine and it was the THRESHOLD that
        # could not fire. That case has its own sentence above.
        if not_measured and not_compared_reason is None:
            _pairs = ", ".join(f"{f} ~ {a}" for f, a in not_measured[:5])
            _more = f" (and {len(not_measured) - 5} more)" if len(not_measured) > 5 else ""
            warnings.warn(
                f"FeatureCorrelationMatrix.get_high_correlations: "
                f"{len(not_measured)} of "
                f"{len(self.feature_names) * len(self.protected_attributes)} "
                f"feature/attribute pair(s) carry no measured correlation, so they "
                f"were NOT compared against the threshold: {_pairs}{_more}. Their "
                f"absence from this list is a could-not-check, not a finding that "
                f"they are below {threshold}. See result.pairs_not_measured.",
                UserWarning,
                stacklevel=2,
            )
        return out


# Known proxy patterns

KNOWN_PROXY_PATTERNS = {
    "race": {
        "high_risk": [
            "zip",
            "zipcode",
            "zip_code",
            "postal",
            "postcode",
            "neighborhood",
            "census_tract",
            "block_group",
            "surname",
            "last_name",
            "name",
        ],
        "medium_risk": [
            "school",
            "school_name",
            "college",
            "university",
            "language",
            "native_language",
            "address",
            "city",
            "district",
        ],
        "affected_groups": [
            "Black/African American",
            "Hispanic/Latino",
            "Asian",
            "Other minorities",
        ],
    },
    "gender": {
        "high_risk": [
            "first_name",
            "name",
            "given_name",
            "title",
            "salutation",
            "prefix",
            "height",
            "weight",
        ],
        "medium_risk": [
            "occupation",
            "job_title",
            "industry",
            "major",
            "field_of_study",
            "hobbies",
            "interests",
        ],
        "affected_groups": ["Women", "Non-binary individuals"],
    },
    "age": {
        "high_risk": [
            "graduation_year",
            "grad_year",
            "years_experience",
            "experience_years",
            "tenure",
            "birth_year",
            "yob",
        ],
        "medium_risk": [
            "technology_proficiency",
            "digital_skills",
            "social_media_usage",
            "career_length",
            "seniority",
        ],
        "affected_groups": ["Older workers (40+)", "Young workers"],
    },
    "income": {
        "high_risk": [
            "zip",
            "zipcode",
            "postal",
            "neighborhood",
            "address",
            "credit_score",
            "credit_rating",
        ],
        "medium_risk": [
            "education_level",
            "degree",
            "school",
            "university",
            "car_make",
            "vehicle_type",
            "device_type",
            "browser",
        ],
        "affected_groups": ["Low-income individuals", "Working class"],
    },
}


# Correlation analysis functions


def compute_pearson_correlation_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    min_periods: int = 10,
    return_pvalues: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Compute a Pearson correlation matrix across numeric features.

    This function computes pairwise Pearson correlation coefficients between
    all specified numeric features. It is useful for:
    - Identifying multicollinearity between features
    - Detecting redundant features that could be removed
    - Understanding feature relationships before fairness analysis

    Args:
        df: DataFrame to analyze
        feature_columns: Columns to include (numeric-only; auto-detect if None)
        min_periods: Minimum overlapping observations required per pair
        return_pvalues: If True, also return a matrix of p-values for
            statistical significance testing

    Returns:
        DataFrame correlation matrix (features × features).
        If return_pvalues=True, returns tuple of (correlations, pvalues).

        Both matrices are three-state: a number is a measurement, and ``nan``
        means the pair was NOT tested, because fewer than ``min_periods`` rows
        carry both values or one side has no variation. A ``nan`` p-value is
        NOT a p-value of 1, and a warning names every pair it applies to.

        A non-finite (``inf``) value is not a measurement: it is excluded from
        every pair it appears in, the way a missing value is, and the count is
        named in a warning, because the pair is then measured on fewer rows
        than were passed. ``min_periods`` below 3 does not disable the overlap
        floor: |r| is +1 or -1 for any two points whatever they are, so a pair
        with fewer than 3 overlapping rows reports ``nan`` in both matrices and
        is named in a warning rather than published as a perfect correlation.

    Raises:
        ValueError: If no numeric feature columns are available

    Example:
        >>> # Basic usage - get correlation matrix
        >>> corr = compute_pearson_correlation_matrix(df)
        >>> print(corr.round(2))

        >>> # With p-values for significance testing
        >>> corr, pvals = compute_pearson_correlation_matrix(df, return_pvalues=True)
        >>> significant = corr[pvals < 0.05]

        >>> # Specific columns only
        >>> corr = compute_pearson_correlation_matrix(
        ...     df,
        ...     feature_columns=['income', 'age', 'education_years']
        ... )

    Note:
        For correlations between features and protected attributes, use
        `compute_feature_correlations()` instead, which handles mixed
        data types (numeric, categorical) appropriately.
    """
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        valid_columns = []
        for c in feature_columns:
            if c not in df.columns:
                continue
            if _is_numeric(df[c]):
                valid_columns.append(c)
        feature_columns = valid_columns

    if not feature_columns:
        raise ValueError("No numeric feature columns available for Pearson correlation.")

    # TWO DIFFERENT DEFINITIONS OF "OVERLAPPING ROW", one per matrix, is what made
    # the two matrices disagree. pandas' `libalgos.nancorr` masks with
    # ``np.isfinite``, so it silently EXCLUDES a non-finite value and counts only
    # the finite rows toward ``min_periods`` (verified 2026-09-29 by measurement,
    # not by reading: a = [inf, 1, 2, 3, 4, 5] against b = [99, 1, 2, 3, 4, 5]
    # gives 1.0, exactly what the five finite rows give on their own, and a frame
    # with 2 finite rows of 4 returns nan at min_periods=3). The loop below used
    # ``.notna()``, which KEEPS an inf, so it counted a row pandas had dropped and
    # handed the inf to ``scipy.stats.pearsonr``, which refuses it.
    #
    # BGL7 F15, 2026-09-29, measured on pd.DataFrame({'a': [inf] +
    # list(range(39)), 'b': list(range(40))}) with min_periods=10 and
    # return_pvalues=True: ``corr.loc['a','b']`` was 1.0 (a real measurement over
    # the 39 finite rows) while ``pval.loc['a','b']`` was nan, so the matrices
    # DISAGREED, and the warning emitted for the pair read "1 pair(s) have enough
    # overlapping rows but no defined coefficient, because one side has no
    # variation: a ~ b (n=40). Both matrices report nan there (could not check),
    # NOT a coefficient of 0 or a p-value of 0 or 1." The corr cell was not nan,
    # neither side was constant, and n was 39 and not 40: three false statements
    # about a cell that HAD been measured, which is the defect class pointing the
    # other way.
    #
    # An inf is now excluded explicitly, the way this module already treats a
    # value that was never measured, so BOTH sides read the same rows: the n that
    # is disclosed is the n that was used, the p-value exists beside its
    # coefficient, and the count of infinite values is named because the pair was
    # measured on fewer rows than the caller passed. The frame is rebuilt only when
    # something non-finite is actually present, so a clean frame takes the
    # identical path it always did.
    nonfinite: List[Tuple[str, int]] = []
    replaced: Dict[str, pd.Series] = {}
    for _col in feature_columns:
        _vals = pd.to_numeric(df[_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        _bad = np.isinf(_vals)
        _n_bad = int(_bad.sum())
        if _n_bad:
            nonfinite.append((str(_col), _n_bad))
            _vals = _vals.copy()
            _vals[_bad] = np.nan
            replaced[_col] = pd.Series(_vals, index=df.index, name=_col)
    if nonfinite:
        work = df[feature_columns].assign(**replaced)
        warnings.warn(
            f"compute_pearson_correlation_matrix: {sum(n for _, n in nonfinite)} value(s) "
            f"in {len(nonfinite)} column(s) are infinite: "
            + ", ".join(f"{c} ({n} of {len(df)} row(s))" for c, n in nonfinite[:5])
            + f"{f' (and {len(nonfinite) - 5} more)' if len(nonfinite) > 5 else ''}. An "
            "infinite value is not a measurement, so it is excluded the way an unmeasured "
            "value is: every coefficient involving those columns is computed on the rows "
            "that hold a finite value, and the row counts reported in any warning below are "
            "those rows. A pair may therefore be measured on fewer rows than this frame "
            "holds.",
            UserWarning,
            stacklevel=2,
        )
    else:
        work = df[feature_columns]

    corr_matrix = work.corr(method="pearson", min_periods=min_periods)

    # THE FLOOR ITSELF, decided above the pair loop, because a floor set below the
    # arithmetic cannot discriminate and no per-pair test can notice. |r| is +1 or
    # -1 for ANY two points, so a coefficient read off a 2-row overlap is
    # arithmetic, not a measurement, and scipy's p-value for it is 1.0, "the
    # strongest statement the test can make that the data give no evidence", which
    # is what the note below was written to stop asserting. BGL7 F15, 2026-09-29,
    # measured on two columns sharing exactly 2 non-null rows: min_periods=10 gave
    # corr nan / pval nan and two warnings (correct), while min_periods=2, 1 and 0
    # each gave corr 1.0 and pval 1.0 with ZERO warnings, and changing the data to
    # [99.0, -7.0] moved it to -1.0, which is how you can tell it is the row count
    # talking and not the data. min_periods is caller-controlled and nothing
    # disclosed a value that disables the check; the sibling convention is quoted
    # in this module already (GroupManager with min_group_size <= 0, and
    # get_high_correlations with a threshold no comparison can fire against).
    #
    # Applied on BOTH return paths, because the default one publishes the
    # coefficient with no p-value beside it to contradict it. Only computed when
    # min_periods is below the floor: above it pandas' own min_periods already
    # enforces the overlap, so a clean call does no extra work.
    below_floor: List[Tuple[str, str, int]] = []
    if min_periods < _PEARSON_MIN_OVERLAP:
        for i, col_i in enumerate(feature_columns):
            for j, col_j in enumerate(feature_columns):
                if i >= j:
                    continue
                n_overlap = int(work[[col_i, col_j]].notna().all(axis=1).sum())
                if n_overlap < _PEARSON_MIN_OVERLAP:
                    below_floor.append((str(col_i), str(col_j), n_overlap))
                    corr_matrix.loc[col_i, col_j] = np.nan
                    corr_matrix.loc[col_j, col_i] = np.nan
    if below_floor:
        _pairs = ", ".join(f"{a} ~ {b} (n={n})" for a, b, n in below_floor[:5])
        _more = f" (and {len(below_floor) - 5} more)" if len(below_floor) > 5 else ""
        warnings.warn(
            f"compute_pearson_correlation_matrix: min_periods={min_periods} is below "
            f"{_PEARSON_MIN_OVERLAP}, the fewest overlapping rows from which a Pearson "
            f"coefficient can be a measurement rather than arithmetic (|r| is +1 or -1 for "
            f"ANY two points, whatever they are), so it does not disable the floor: "
            f"{len(below_floor)} pair(s) whose overlap is below it report nan in both "
            f"matrices (could not check), NOT a coefficient of 1 or a p-value of 1: "
            f"{_pairs}{_more}.",
            UserWarning,
            stacklevel=2,
        )

    if not return_pvalues:
        return corr_matrix

    n_features = len(feature_columns)
    # NaN, not ones. A p-value of 1.0 is the STRONGEST statement the test can
    # make that the data give no evidence of an association, and this matrix
    # started out asserting it for every pair, so any pair skipped below kept
    # that assertion. Measured 2026-09-27 on two columns sharing 5 non-null
    # rows against the default min_periods=10: `corr_matrix` correctly said NaN
    # for both off-diagonal cells while `pval_matrix` said 1.0 for all four,
    # with no warning, so `corr[pvals < 0.05]` (the documented usage above)
    # silently dropped a pair nobody had tested instead of flagging it as
    # untested. NaN is the value pandas and scipy already use here for "not
    # computed", and it is what the corr matrix beside it uses.
    pval_matrix = pd.DataFrame(
        np.full((n_features, n_features), np.nan),
        index=feature_columns,
        columns=feature_columns,
    )
    # Columns whose SELF-correlation does not exist (no variation, or too few
    # overlapping rows). Collected so the warning can name them: the docstring
    # promises "a warning names every pair it applies to", and before BGL5 the
    # only list feeding that warning was the min_periods one.
    undefined_self: List[str] = []
    # The diagonal is a column against itself, which needs no test, but ONLY
    # where the coefficient beside it exists. The condition used to be "the
    # column has any observation at all", justified by the comment "it is 1.0 by
    # construction whenever the column has any observation at all". That is
    # false in two reachable cases, and BGL5 2026-09-27 measured both:
    #
    #   * a CONSTANT column (40 rows of 5.0 beside a varying b): pandas gives
    #     corr.loc['a','a'] = NaN, because a column with no variation has no
    #     self-correlation, while pval_matrix.loc['a','a'] read 0.0, the
    #     STRONGEST assertion of an association the test can make, for a cell
    #     the same call had refused to compute. `(pvals < 0.05).loc['a','a']`
    #     was True, so the docstring's own `corr[pvals < 0.05]` selected it.
    #   * a column with 3 non-null rows against min_periods=10: corr NaN,
    #     p-value 0.0, the same contradiction.
    #
    # Both now read NaN in both matrices. Asking the coefficient rather than the
    # row count is what makes the two matrices agree BY CONSTRUCTION, which is
    # what the note above claims they do; any future reason for a NaN
    # coefficient is carried across for free.
    for col in feature_columns:
        if _is_finite_number(corr_matrix.loc[col, col]):
            pval_matrix.loc[col, col] = 0.0
        else:
            undefined_self.append(str(col))

    skipped: List[Tuple[str, str, int]] = []
    # Pairs with enough overlapping rows whose coefficient is still undefined,
    # with the columns that have no variation named per pair rather than assumed.
    undefined: List[Tuple[str, str, int, List[str]]] = []
    _below_floor_pairs = {(a, b) for a, b, _ in below_floor}
    for i, col_i in enumerate(feature_columns):
        for j, col_j in enumerate(feature_columns):
            if i >= j:  # Diagonal and lower triangle
                continue
            if (str(col_i), str(col_j)) in _below_floor_pairs:
                # Already refused above, in both matrices, with its own warning.
                continue
            mask = work[[col_i, col_j]].notna().all(axis=1)
            n_overlap = int(mask.sum())
            if n_overlap >= min_periods:
                with warnings.catch_warnings():
                    # scipy's ConstantInputWarning is the only notice a caller
                    # got for a pair one side of which has no variation, and it
                    # names neither column. It is swallowed here and the pair is
                    # recorded in `undefined` instead, so the vfairness warning
                    # below names it the way the docstring promises.
                    warnings.simplefilter("ignore")
                    _, pval = stats.pearsonr(work.loc[mask, col_i], work.loc[mask, col_j])
                coef = corr_matrix.loc[col_i, col_j]
                if _is_finite_number(pval) and _is_finite_number(coef):
                    pval_matrix.loc[col_i, col_j] = pval
                    pval_matrix.loc[col_j, col_i] = pval  # Symmetric
                else:
                    # Enough overlapping rows, and one of the two does not exist.
                    # ENFORCED here, not hoped for: the comment above claims the
                    # two matrices agree BY CONSTRUCTION because the diagonal asks
                    # the coefficient, and that only ever covered "NaN coefficient
                    # -> NaN p-value". The other direction was untested and live
                    # (the inf pair above: coefficient 1.0, p-value nan), so a
                    # coefficient is no longer published for a pair whose test
                    # could not run, and the reason is MEASURED per pair instead
                    # of asserted. np.unique, not a variance against zero.
                    flat = [
                        str(col)
                        for col in (col_i, col_j)
                        if len(np.unique(work.loc[mask, col].to_numpy(dtype=float))) < 2
                    ]
                    undefined.append((col_i, col_j, n_overlap, flat))
                    corr_matrix.loc[col_i, col_j] = np.nan
                    corr_matrix.loc[col_j, col_i] = np.nan
            else:
                skipped.append((col_i, col_j, n_overlap))

    if skipped:
        _pairs = ", ".join(f"{a} ~ {b} (n={n})" for a, b, n in skipped[:5])
        _more = f" (and {len(skipped) - 5} more)" if len(skipped) > 5 else ""
        warnings.warn(
            f"compute_pearson_correlation_matrix: {len(skipped)} pair(s) have fewer "
            f"than min_periods={min_periods} overlapping non-null rows, so no "
            f"correlation and no p-value were computed for them: {_pairs}{_more}. "
            f"Both matrices report nan there (could not check), NOT a coefficient of "
            f"0 or a p-value of 1.",
            UserWarning,
            stacklevel=2,
        )
    if undefined or undefined_self:
        # The second half of the docstring's promise. BGL5 2026-09-27: a constant
        # column produced a NaN off-diagonal coefficient with NO vfairness
        # warning at all (only scipy's ConstantInputWarning, which names no
        # column) and a diagonal p-value of 0.0. Both cells are now NaN and this
        # sentence is what tells a reader which cells and why.
        parts = []
        if undefined:
            # The cause is now whatever was MEASURED on the pair's own overlapping
            # rows. "because one side has no variation" was asserted for every
            # pair in this list and was false for the inf pair, which had plenty
            # of variation on both sides. A sentence that names a cause it did not
            # check is the same defect one level up from the number it explains.
            parts.append(
                f"{len(undefined)} pair(s) have enough overlapping rows and still no "
                "coefficient and no p-value: "
                + "; ".join(
                    f"{a} ~ {b} (n={n})"
                    + (
                        f", no variation in {', '.join(flat)}"
                        if flat
                        else ", and both sides DO vary over those rows, so the reason is not "
                        "a constant column"
                    )
                    for a, b, n, flat in undefined[:5]
                )
            )
        if undefined_self:
            parts.append(
                f"{len(undefined_self)} column(s) have no self-correlation either, so "
                "their diagonal cell is nan in both matrices and NOT a p-value of 0: "
                + ", ".join(undefined_self[:5])
            )
        warnings.warn(
            "compute_pearson_correlation_matrix: " + "; ".join(parts) + ". Both matrices "
            "report nan there (could not check), NOT a coefficient of 0 or a p-value of "
            "0 or 1.",
            UserWarning,
            stacklevel=2,
        )

    return corr_matrix, pval_matrix


def compute_feature_correlations(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    feature_columns: Optional[List[str]] = None,
    method: str = "auto",
) -> FeatureCorrelationMatrix:
    """
    Compute correlation matrix between features and protected attributes.

    Uses appropriate correlation measures based on data types:
    - Numeric-Numeric: Pearson correlation
    - Numeric-Categorical: Point-biserial correlation
    - Categorical-Categorical: Cramér's V

    P-values are always computed alongside every coefficient (they are
    by-products of the same scipy calls) and returned in ``pvalues``; a NaN
    p-value marks a pair that could not be tested.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute columns
        feature_columns: Columns to include (auto-detect if None)
        method: Correlation method ('auto', 'pearson', 'spearman', 'cramers_v');
            anything else raises ValueError

    Returns:
        FeatureCorrelationMatrix with correlations and p-values

    Example:
        >>> matrix = compute_feature_correlations(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     method='auto'
        ... )
        >>> print(matrix.correlations)

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

    Ledger row: compute_feature_correlations. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    # audit-6 lane 2 (2026-09-09): the documented `include_pvalues` parameter
    # ("Whether to compute p-values") was never referenced; p-values were
    # computed and returned regardless (measured: include_pvalues=False still
    # filled `pvalues`). It is removed rather than honoured because the values
    # cost nothing extra to keep, so the honest signature is one without a
    # switch that does nothing. An unknown `method` used to fall through to an
    # all-NaN matrix with no warning; it is refused instead.
    supported_methods = ("auto", "pearson", "spearman", "cramers_v")
    if method not in supported_methods:
        raise ValueError(
            f"compute_feature_correlations method must be one of {supported_methods}, "
            f"got {method!r}"
        )

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in protected_attributes]
    else:
        feature_columns = [c for c in feature_columns if c in df.columns]

    corr_matrix = pd.DataFrame(index=feature_columns, columns=protected_attributes, dtype=float)
    pval_matrix = pd.DataFrame(index=feature_columns, columns=protected_attributes, dtype=float)

    for attr in protected_attributes:
        if attr not in df.columns:
            continue

        attr_series = df[attr].dropna()
        attr_is_numeric = _is_numeric(attr_series)

        for feature in feature_columns:
            if feature not in df.columns:
                continue

            feature_series = df[feature].dropna()
            feature_is_numeric = _is_numeric(feature_series)

            common_idx = attr_series.index.intersection(feature_series.index)
            if len(common_idx) < MIN_SAMPLE_SIZE:
                corr_matrix.loc[feature, attr] = np.nan
                pval_matrix.loc[feature, attr] = np.nan
                continue

            attr_aligned = attr_series[common_idx]
            feature_aligned = feature_series[common_idx]

            if method == "auto":
                if feature_is_numeric and attr_is_numeric:
                    corr, pval = _pearson_correlation(feature_aligned, attr_aligned)
                elif feature_is_numeric or attr_is_numeric:
                    # Exactly one side numeric: continuous vs categorical.
                    # Point-biserial (Tate 1954) is valid ONLY when the
                    # categorical side is BINARY; for a 3+ level nominal it
                    # degenerates to Pearson on arbitrary category codes and a
                    # near-perfect proxy reads as 'weak'. Use the correlation
                    # ratio (eta) for the polytomous case, reserving
                    # point-biserial for genuinely dichotomous attributes.
                    if feature_is_numeric:
                        numeric_side, cat_side = feature_aligned, attr_aligned
                    else:
                        numeric_side, cat_side = attr_aligned, feature_aligned
                    if cat_side.nunique() <= 2:
                        corr, pval = _point_biserial(feature_aligned, attr_aligned)
                    else:
                        corr, pval = _correlation_ratio_with_pvalue(numeric_side, cat_side)
                else:
                    corr, pval = _cramers_v_with_pvalue(feature_aligned, attr_aligned)
            elif method == "pearson":
                feature_encoded = _encode_for_correlation(feature_aligned)
                attr_encoded = _encode_for_correlation(attr_aligned)
                if feature_encoded is None or attr_encoded is None:
                    corr, pval = np.nan, np.nan
                else:
                    corr, pval = _pearson_correlation(feature_encoded, attr_encoded)
            elif method == "spearman":
                feature_encoded = _encode_for_correlation(feature_aligned)
                attr_encoded = _encode_for_correlation(attr_aligned)
                if feature_encoded is None or attr_encoded is None:
                    corr, pval = np.nan, np.nan
                else:
                    corr, pval = _spearman_correlation(feature_encoded, attr_encoded)
            elif method == "cramers_v":
                corr, pval = _cramers_v_with_pvalue(feature_aligned, attr_aligned)
            else:
                corr, pval = np.nan, np.nan

            corr_matrix.loc[feature, attr] = corr
            pval_matrix.loc[feature, attr] = pval

    return FeatureCorrelationMatrix(
        correlations=corr_matrix,
        pvalues=pval_matrix,
        feature_names=feature_columns,
        protected_attributes=protected_attributes,
        method=method,
    )


class ProxyScreenResult(List[ProxyVariableResult]):
    """What ``identify_proxy_variables`` FOUND, plus what it never looked at.

    A ``list`` subclass for the same reason as :class:`ProxyChainResult`: every
    existing consumer (iteration, ``len``, ``isinstance(x, list)``, slicing)
    keeps working unchanged, while a caller that needs the coverage can read
    ``screens_not_run`` and ``complete``.

    BGL-S2b (2026-09-17). The screen has four gates that end in a bare
    ``continue`` or a ``return None``: the attribute is absent, the attribute
    has too few non-null rows, the attribute or feature will not encode, and
    the feature is high-cardinality. A pair stopped by ANY of them left no
    trace, so a perfect CRITICAL proxy sitting in an identifier-shaped column
    came back as an empty list, byte-identical to a frame that is genuinely
    clean. The high-cardinality gate is the dangerous one, because an
    identifier-shaped column is exactly what a strong proxy often looks like.

    Attributes:
        screens_not_run: one dict per pair the screen never measured, with
            ``protected_attribute``, ``feature`` (None when the whole attribute
            was skipped) and a ``reason`` a reader can act on.
        not_screened_reason: why NOTHING here was screened, when that is the
            case: a request holding no feature/attribute pair at all. ``None``
            when at least one pair was requested.
        complete: True only when every requested pair was actually screened AND
            at least one pair was requested.
    """

    def __init__(
        self,
        results: Optional[List[ProxyVariableResult]] = None,
        *,
        screens_not_run: Optional[List[Dict[str, Any]]] = None,
        not_screened_reason: Optional[str] = None,
    ) -> None:
        super().__init__(results or [])
        self.screens_not_run: List[Dict[str, Any]] = list(screens_not_run or [])
        self.not_screened_reason: Optional[str] = not_screened_reason

    @property
    def complete(self) -> bool:
        # BGL7 F16 2026-09-29: `not self.screens_not_run` alone read True over a
        # screen that looked at ZERO pairs, and True is the reassuring half of a
        # two-state answer. It cannot tell "every requested pair was screened",
        # which is what the docstring above promises, from "no pair was
        # requested". The sibling :class:`HighCorrelationResult` in this module
        # was given exactly this second term in BGL5 and this class was not.
        return not self.screens_not_run and self.not_screened_reason is None


def identify_proxy_variables(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    feature_columns: Optional[List[str]] = None,
    correlation_threshold: float = 0.3,
    include_mutual_information: bool = True,
    include_known_patterns: bool = True,
    min_sample_size: int = 100,
    max_cardinality: int = 50,
    significance_level: float = 0.05,
) -> ProxyScreenResult:
    """
    Identify features that may serve as proxies for protected attributes.

    Uses multiple correlation measures and pattern matching to detect
    features that could enable indirect discrimination.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute column names
        feature_columns: Columns to test as potential proxies (auto-detect if None)
        correlation_threshold: Minimum correlation to flag as proxy
        include_mutual_information: Whether to compute mutual information
        include_known_patterns: Whether to check against known proxy patterns
        min_sample_size: Minimum sample size for reliable computation
        max_cardinality: Maximum unique values for a feature column
        significance_level: P-value threshold for statistical significance

    Returns:
        List of ProxyVariableResult objects, sorted by risk level

    Example:
        >>> results = identify_proxy_variables(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     correlation_threshold=0.3
        ... )
        >>> for r in results:
        ...     print(f"{r.feature} -> {r.protected_attribute}: {r.risk_level.value}")
    """
    results: List[ProxyVariableResult] = []
    # BGL-S2b (2026-09-17). Every gate below used to be a bare `continue`, so a
    # pair the screen NEVER LOOKED AT left no trace at all and the caller read
    # the empty result as "no proxy here". The high-cardinality gate is the one
    # that matters most: it is the gate a genuine CRITICAL proxy is most likely
    # to trip, because an identifier-shaped column is exactly what a strong
    # proxy often looks like. Each skip is now named, warned about, and carried
    # on the returned object as `screens_not_run`.
    screens_not_run: List[Dict[str, Any]] = []

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in protected_attributes]

    # THE REQUEST ITSELF, decided above the loop, because a request holding no
    # pair makes every gate in the loop vacuous and no per-pair record can notice:
    # the loop simply never runs a screen, so `screens_not_run` stays empty and
    # `complete` reads True. BGL7 F16, 2026-09-29, measured on 200 rows where
    # height IS a critical proxy for gender (feature_columns=None: 1 result,
    # 'critical', r=0.761, complete True): feature_columns=['not_a_column'] gave 0
    # results with complete FALSE, screens_not_run 1 and a warning (the gate below
    # working as intended), while feature_columns=[], protected_attributes=[] and
    # feature_columns=['gender'] (the attribute itself, which the loop skips
    # deliberately and deliberately does not record) EACH gave 0 results,
    # complete TRUE, screens_not_run [] and ZERO warnings. True is the neutral
    # value here and it was published for a screen that measured nothing.
    #
    # The sibling graded in the same wave already closes this hole with the same
    # shape: get_high_correlations carries a `not_compared_reason` for an
    # n_pairs == 0 matrix, saying "An empty result over zero pairs is not a
    # finding that no correlation is high."
    n_pairs = sum(1 for attr in protected_attributes for f in feature_columns if f != attr)
    not_screened_reason: Optional[str] = None
    if n_pairs == 0:
        not_screened_reason = (
            f"this screen was asked for {len(protected_attributes)} protected attribute(s) "
            f"and {len(feature_columns)} candidate feature(s), which is 0 feature/attribute "
            f"pair(s) to screen"
            + (
                " (the only feature(s) named are the protected attribute(s) themselves, and "
                "an attribute is not a candidate proxy for itself)"
                if feature_columns and protected_attributes
                else ""
            )
            + ". Nothing was screened, so this empty result is a could-not-check, not a "
            "finding that no feature is a proxy for a protected attribute."
        )

    for attr in protected_attributes:
        if attr not in df.columns:
            screens_not_run.append(
                {
                    "protected_attribute": attr,
                    "feature": None,
                    "reason": "column not present in the DataFrame",
                }
            )
            continue

        attr_series = df[attr].dropna()
        if len(attr_series) < min_sample_size:
            screens_not_run.append(
                {
                    "protected_attribute": attr,
                    "feature": None,
                    "reason": (
                        f"only {len(attr_series)} non-null row(s), and the screen needs "
                        f"min_sample_size={min_sample_size}"
                    ),
                }
            )
            continue

        # Encodability gate only: an attribute that cannot be encoded at all is
        # skipped. The encoding itself is NOT reused downstream (see
        # _analyze_proxy_relationship, which must encode the pairwise-aligned
        # rows instead).
        if _encode_for_correlation(attr_series) is None:
            screens_not_run.append(
                {
                    "protected_attribute": attr,
                    "feature": None,
                    "reason": "attribute could not be encoded for correlation",
                }
            )
            continue

        for feature in feature_columns:
            if feature not in df.columns:
                # The fifth gate, and the last one in this function that left no
                # trace. BGL5 2026-09-27, measured on 200 rows where 'height' IS
                # a critical proxy for gender (feature_columns=None finds it:
                # 1 result, 'critical', 0.819): feature_columns=['not_a_column']
                # returned 0 results with complete == True, screens_not_run == []
                # and no warning, and feature_columns=['heigth', 'noise'] (one
                # misspelling, one real column) did the same while the misspelled
                # name was the one carrying the proxy. ProxyScreenResult
                # documents "complete: True only when every requested pair was
                # actually screened", and zero pairs had been screened. The
                # sibling case (the ATTRIBUTE not being a column) was already
                # recorded thirty lines above; this is the same fact about the
                # other half of the pair.
                screens_not_run.append(
                    {
                        "protected_attribute": attr,
                        "feature": feature,
                        "reason": (
                            f"{feature!r} is not a column of this DataFrame, so the pair "
                            f"was never screened. A misspelled feature name is a "
                            f"could-not-check, not a finding that it is clean."
                        ),
                    }
                )
                continue
            if feature == attr:
                # Not recorded: a protected attribute is not a candidate proxy
                # for itself, so no screen was requested here. The auto-detected
                # feature list already excludes the protected attributes, so this
                # only fires when a caller named one explicitly.
                continue

            # Skip high-cardinality categorical columns (likely IDs)
            # Only apply this check to non-numeric columns
            feature_series = df[feature].dropna()
            if not _is_numeric(feature_series):
                n_unique = df[feature].nunique()
                n_rows = len(feature_series)
                if n_unique > max_cardinality or (
                    n_rows > 0 and n_unique / n_rows > MAX_CARDINALITY_RATIO
                ):
                    screens_not_run.append(
                        {
                            "protected_attribute": attr,
                            "feature": feature,
                            "reason": (
                                f"high cardinality: {n_unique} distinct value(s) over "
                                f"{n_rows} row(s) (max_cardinality={max_cardinality}, "
                                f"ratio limit {MAX_CARDINALITY_RATIO}). The pair was "
                                f"NOT screened; this is not a finding that it is clean."
                            ),
                        }
                    )
                    continue

            result = _analyze_proxy_relationship(
                df,
                feature,
                attr,
                correlation_threshold=correlation_threshold,
                include_mutual_information=include_mutual_information,
                include_known_patterns=include_known_patterns,
                min_sample_size=min_sample_size,
                significance_level=significance_level,
            )

            if result:
                results.append(result)
            else:
                # `_analyze_proxy_relationship` returns None for two quite
                # different reasons: "measured and below the reporting
                # threshold" (a finding) and "could not measure at all" (an
                # absence of evidence). Only the second is recorded here; the
                # helper reports which by way of `_proxy_screen_skipped`.
                _why = _proxy_screen_skipped(df, feature, attr, min_sample_size=min_sample_size)
                if _why is not None:
                    screens_not_run.append(
                        {
                            "protected_attribute": attr,
                            "feature": feature,
                            "reason": _why,
                        }
                    )

    risk_order = {
        ProxyRiskLevel.CRITICAL: 0,
        ProxyRiskLevel.HIGH: 1,
        ProxyRiskLevel.MEDIUM: 2,
        ProxyRiskLevel.LOW: 3,
        ProxyRiskLevel.NEGLIGIBLE: 4,
    }
    results.sort(key=lambda x: (risk_order[x.risk_level], -abs(x.correlation)))

    if screens_not_run:
        _detail = "; ".join(
            f"{e['protected_attribute']}"
            + (f" ~ {e['feature']}" if e["feature"] else "")
            + f": {e['reason']}"
            for e in screens_not_run[:5]
        )
        _more = f" (and {len(screens_not_run) - 5} more)" if len(screens_not_run) > 5 else ""
        warnings.warn(
            f"identify_proxy_variables: {len(screens_not_run)} feature/attribute "
            f"pair(s) were NOT SCREENED, so this result does not cover the whole "
            f"frame: {_detail}{_more}. An empty or short result is not evidence that "
            f"those pairs are clean. See result.screens_not_run.",
            UserWarning,
            stacklevel=2,
        )
    if not_screened_reason is not None:
        warnings.warn(
            f"identify_proxy_variables: {not_screened_reason} See "
            f"result.not_screened_reason and result.complete.",
            UserWarning,
            stacklevel=2,
        )

    return ProxyScreenResult(
        results,
        screens_not_run=screens_not_run,
        not_screened_reason=not_screened_reason,
    )


def _proxy_screen_skipped(
    df: pd.DataFrame,
    feature: str,
    protected_attr: str,
    *,
    min_sample_size: int,
) -> Optional[str]:
    """Why ``_analyze_proxy_relationship`` looked at nothing, or None.

    Returns a reason string when the pair was never measurable on these rows,
    and ``None`` when the helper DID measure and simply found nothing worth
    reporting. The distinction is the whole point: only the first is a gap in
    coverage, and calling the second one a gap would flood the disclosure with
    every clean pair in the frame and make it unreadable.
    """
    try:
        aligned = df[[feature, protected_attr]].dropna()
    except Exception:
        return f"the pair {feature!r} / {protected_attr!r} could not be aligned"
    if len(aligned) < min_sample_size:
        return (
            f"only {len(aligned)} row(s) carry both values, and the screen needs "
            f"min_sample_size={min_sample_size}"
        )
    if _encode_for_correlation(aligned[protected_attr]) is None:
        return "the protected attribute could not be encoded on the aligned rows"
    if _encode_for_correlation(aligned[feature]) is None:
        return "the feature could not be encoded for correlation"
    return None


class ProxyChainResult(List[Dict[str, Any]]):
    """The chains ``find_proxy_chains`` found, PLUS what it could not look at.

    A bare ``[]`` was the same token for "looked and found nothing" and "could
    not look", and the second case is common: below ``CHAIN_MIN_SAMPLES`` rows,
    a constant column, a column that will not encode, or a protected attribute
    that is not in the frame at all. Measured 2026-09-16 at n=25, where the
    chain zipcode -> income -> race was STRONGER (0.7947 and 0.7721) than in
    the passing n=400 control and was reported as absent, in silence.

    This subclasses ``list`` on purpose, so every existing consumer
    (``extend``, ``for ch in ...``, ``len``, ``isinstance(x, list)``,
    ``... or []``) keeps working unchanged, while a caller that wants the
    coverage can read it:

    Attributes:
        pairs_not_computed: ``(col_a, col_b, reason)`` for every pair whose
            correlation could not be computed. NOT pairs that were measured
            and came in under the threshold: those are a finding.
        protected_attribute_present: False when the named attribute is not a
            column of the frame, in which case nothing was examined at all.
        complete: True only when the attribute was present and every pair was
            computable, so an empty result really does mean "none found".
    """

    def __init__(
        self,
        chains: Optional[List[Dict[str, Any]]] = None,
        *,
        pairs_not_computed: Optional[List[Tuple[str, str, str]]] = None,
        protected_attribute_present: bool = True,
        depths_not_searched: Optional[List[int]] = None,
    ) -> None:
        super().__init__(chains or [])
        self.pairs_not_computed: List[Tuple[str, str, str]] = list(pairs_not_computed or [])
        self.protected_attribute_present = protected_attribute_present
        #: Chain depths the caller ASKED FOR that this search does not
        #: implement. Only depth 2 is implemented, so `max_chain_length=3`
        #: used to return `complete=True` over a depth-3 search that never
        #: ran. An unimplemented depth is could-not-check, exactly like an
        #: uncomputable pair, and it belongs in the same verdict.
        self.depths_not_searched: List[int] = list(depths_not_searched or [])

    def to_dict(self) -> Dict[str, Any]:
        """The result as JSON-safe data, coverage INCLUDED.

        ``ProxyChainResult`` is a ``list`` subclass, so every ordinary way of
        moving it across a boundary -- ``jsonify``, ``json.dumps``, ``result +
        []``, ``result[:n]`` -- flattens it to a bare list and silently drops
        ``pairs_not_computed``, ``depths_not_searched`` and ``complete``.
        Measured at ``vfairness.mcp.tools.detect_proxies`` on a 25-row frame:
        the payload said "Proxy analysis complete: 0 proxy chain(s)" with no
        could-not-check anywhere in it, while this object knew two pairs had
        never been correlated. A consumer that must serialise calls this.
        """
        return {
            "chains": [dict(c) for c in self],
            "n_chains": len(self),
            "complete": self.complete,
            "protected_attribute_present": self.protected_attribute_present,
            "pairs_not_computed": [list(t) for t in self.pairs_not_computed],
            "n_pairs_not_computed": len(self.pairs_not_computed),
            "depths_not_searched": list(self.depths_not_searched),
            "coverage_note": self.coverage_note,
        }

    @property
    def coverage_note(self) -> str:
        """One sentence a reader can act on, empty when the search was total."""
        if self.complete:
            return ""
        if not self.protected_attribute_present:
            return (
                "COULD NOT CHECK: the protected attribute is not a column of the "
                "frame, so no proxy chain search was performed at all. An empty "
                "result is not a finding that no chain exists."
            )
        parts = []
        if self.pairs_not_computed:
            parts.append(f"{len(self.pairs_not_computed)} column pair(s) could not be correlated")
        if self.depths_not_searched:
            parts.append(
                f"chain depth(s) {sorted(self.depths_not_searched)} were requested but are "
                f"not implemented by this search"
            )
        return (
            "COULD NOT CHECK (partial): "
            + ", and ".join(parts)
            + ". A chain through any of those would not appear here, so an empty or "
            "short result is not evidence that none exists."
        )

    @property
    def complete(self) -> bool:
        return (
            self.protected_attribute_present
            and not self.pairs_not_computed
            and not self.depths_not_searched
        )


def find_proxy_chains(
    df: pd.DataFrame,
    protected_attribute: str,
    *,
    max_chain_length: int = 2,
    correlation_threshold: float = 0.3,
) -> ProxyChainResult:
    """
    Find chains of proxy relationships (A -> B -> Protected).

    Identifies indirect proxies where a feature is correlated with another
    feature that is itself a proxy for the protected attribute.

    Args:
        df: DataFrame to analyze
        protected_attribute: Protected attribute to trace proxies for
        max_chain_length: Maximum length of proxy chains to find (currently only 2 is supported)
        correlation_threshold: Minimum correlation for chain links

    Returns:
        ProxyChainResult: a ``list`` of proxy chain dictionaries with structure

        {
            'chain': [feature, intermediate, protected_attribute],
            'correlations': [corr1, corr2],
            'indirect_correlation': product of correlations,
            'chain_length': length of chain
        }

        carrying ``pairs_not_computed``, ``protected_attribute_present`` and
        ``complete`` beside them. An EMPTY result is only a finding when
        ``complete`` is True; otherwise part of the frame was never examined.

    Example:
        >>> result = find_proxy_chains(df, 'race', correlation_threshold=0.3)
        >>> for chain in result:
        ...     print(f"Chain: {' -> '.join(chain['chain'])}")
        >>> if not result.complete:
        ...     print(result.pairs_not_computed)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: find_proxy_chains. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    chains: List[Dict[str, Any]] = []
    not_computed: List[Tuple[str, str, str]] = []
    # BGL-S2b (2026-09-17). Only depth 2 is implemented. A caller asking for
    # more (pulse/orchestrator.py passes max_chain_length=3) got the depth-2
    # answer stamped complete=True, i.e. a certificate over a search that never
    # ran. Record the unsearched depths; `complete` now refuses while any
    # remain. This is could-not-check, not a finding of no deeper chain.
    depths_not_searched = [d for d in range(3, max(3, int(max_chain_length) + 1))]

    if protected_attribute not in df.columns:
        # Nothing was examined. Answering [] here is the worst of the three
        # states: it reads as a clean result for a column that is not there.
        warnings.warn(
            f"find_proxy_chains: protected attribute {protected_attribute!r} is not a "
            f"column of the frame, so NO proxy chain search was performed. This is "
            f"COULD NOT CHECK, not a finding that the frame holds no proxy chains. "
            f"Columns present: {list(df.columns)}.",
            UserWarning,
            stacklevel=2,
        )
        return ProxyChainResult(
            [],
            pairs_not_computed=[],
            protected_attribute_present=False,
            depths_not_searched=depths_not_searched,
        )

    direct_proxies: Dict[str, float] = {}

    for col in df.columns:
        if col == protected_attribute:
            continue

        try:
            corr, reason = _safe_correlation(df, col, protected_attribute)
            if corr is None:
                # Distinguish "measured and below threshold" (a finding) from
                # "could not be measured" (an absence of evidence).
                not_computed.append((col, protected_attribute, reason))
                continue
            if abs(corr) >= correlation_threshold:
                direct_proxies[col] = abs(corr)
        except Exception as e:
            not_computed.append((col, protected_attribute, f"error: {e}"))
            warnings.warn(f"Failed to compute correlation for {col}: {e}")
            continue

    if max_chain_length >= 2:
        for proxy, proxy_corr in direct_proxies.items():
            for col in df.columns:
                if col in [protected_attribute, proxy]:
                    continue
                if col in direct_proxies:
                    continue  # Already a direct proxy

                try:
                    corr, reason = _safe_correlation(df, col, proxy)
                    if corr is None:
                        not_computed.append((col, proxy, reason))
                        continue
                    if abs(corr) >= correlation_threshold:
                        chains.append(
                            {
                                "chain": [col, proxy, protected_attribute],
                                "correlations": [abs(corr), proxy_corr],
                                "indirect_correlation": abs(corr) * proxy_corr,
                                "chain_length": 2,
                            }
                        )
                except Exception as e:
                    not_computed.append((col, proxy, f"error: {e}"))
                    warnings.warn(f"Failed to compute chain correlation for {col}->{proxy}: {e}")
                    continue

    chains.sort(key=lambda x: x["indirect_correlation"], reverse=True)

    if not_computed:
        _detail = "; ".join(f"{a} ~ {b}: {why}" for a, b, why in not_computed[:6])
        _more = f" (and {len(not_computed) - 6} more)" if len(not_computed) > 6 else ""
        warnings.warn(
            f"find_proxy_chains: {len(not_computed)} column pair(s) COULD NOT BE "
            f"CORRELATED, so this search did not cover the whole frame: {_detail}"
            f"{_more}. A chain through one of those pairs would not appear below, and "
            f"an empty or short result is not evidence that none exists. See "
            f"result.pairs_not_computed.",
            UserWarning,
            stacklevel=2,
        )

    if depths_not_searched:
        warnings.warn(
            f"find_proxy_chains: max_chain_length={max_chain_length} was requested but "
            f"only chains of length 2 are implemented, so depth(s) "
            f"{depths_not_searched} were NOT SEARCHED. result.complete is False and "
            f"result.depths_not_searched names them. A longer chain would not appear "
            f"below; this is could not check, not a finding that none exists.",
            UserWarning,
            stacklevel=2,
        )

    return ProxyChainResult(
        chains[:20],
        pairs_not_computed=not_computed,
        protected_attribute_present=True,
        depths_not_searched=depths_not_searched,
    )


def _safe_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    min_samples: int = CHAIN_MIN_SAMPLES,
) -> Tuple[Optional[float], str]:
    """
    Safely compute correlation between two columns with proper index alignment.

    Args:
        df: DataFrame containing both columns
        col1: First column name
        col2: Second column name
        min_samples: Minimum number of valid samples required

    Returns:
        ``(correlation, "")`` when the correlation was computed, or
        ``(None, reason)`` when it could not be. The REASON is the point: a
        bare None told the caller nothing about whether the pair was weak or
        unexaminable, and the caller then folded both into the same empty
        list.
    """
    aligned = df[[col1, col2]].dropna()

    if len(aligned) < min_samples:
        return None, (
            f"insufficient_samples: {len(aligned)} rows with both values present, "
            f"below the minimum of {min_samples}"
        )

    x = _encode_for_correlation(aligned[col1])
    y = _encode_for_correlation(aligned[col2])

    if x is None or y is None:
        unencodable = ", ".join(name for name, enc in ((col1, x), (col2, y)) if enc is None)
        return None, f"unencodable: {unencodable} could not be encoded for correlation"

    try:
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            return None, (
                "undefined: one side has no variation (a constant or all-missing "
                "column), so the correlation is 0/0"
            )
        return float(corr), ""
    except Exception as exc:
        return None, f"error: {exc}"


def analyze_intersectional_correlations(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    feature_columns: Optional[List[str]] = None,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """
    Analyze correlations accounting for intersectional groups.

    Computes correlations within and across intersectional subgroups
    (e.g., race AND gender combinations).

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute columns
        feature_columns: Features to analyze
        min_group_size: Minimum samples per intersectional group

    Returns:
        Dictionary with intersectional correlation analysis. Three-state on both
        halves that a reader grades the data by:

        * ``groups_not_assessed`` lists every intersectional group that exists
          in the data and was NOT examined, with its size and the reason, so an
          empty ``within_group_correlations`` can be told apart from a frame
          whose groups were all examined and all agreed.
        * ``features_not_assessed`` lists every (group, feature) pair that was
          not measured INSIDE a group that was examined, and
          ``features_not_compared`` every feature that fewer than two examined
          groups measured. Either one drops a feature out of
          ``between_group_differences``, and neither is visible in
          ``groups_not_assessed``, which is per group only.
        * ``rows_without_a_recorded_group`` counts the rows whose protected
          value was never recorded. They are NOT a group and are excluded from
          every number here.
        * ``features_not_numeric`` lists every requested feature column this
          analysis cannot read as a number, with its dtype. It is the reason
          ``features_not_assessed`` and ``features_not_compared`` can both be
          empty while nothing was measured: a feature that is not numeric never
          reaches either loop.
        * ``coverage`` is ``"complete"`` (every group examined AND every numeric
          feature measured in each of them and compared across them),
          ``"partial"`` or ``NOT_ASSESSED``. ``NOT_ASSESSED`` covers BOTH ways
          this analysis can measure nothing: no group qualified, or no requested
          feature is numeric. ``"complete"`` is never published over an empty
          ``between_group_differences``, which is what "every numeric feature
          measured" reads as when there is no numeric feature to measure. A
          non-numeric feature beside at least one numeric one does not by itself
          reduce ``coverage``, since this analysis never claimed to measure it;
          it is listed in ``features_not_numeric`` and must be read there.
        * ``coefficient_of_variation`` is ``nan``, never ``0``, when the group
          means do not admit one.

    Example:
        >>> results = analyze_intersectional_correlations(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     min_group_size=30
        ... )
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in protected_attributes]

    numeric_features = [c for c in feature_columns if _is_numeric(df[c])]
    # BGL6 AUDIT, 2026-09-29. Every requested feature this analysis cannot read
    # as a number, RECORDED. It used to be dropped here in silence, which is how
    # 'complete' came to be published over an empty comparison: with no numeric
    # feature BOTH loops below record nothing, so features_not_assessed and
    # features_not_compared stay empty and the coverage rule reads as satisfied
    # VACUOUSLY. Measured on 80 rows across two examined groups whose only
    # requested feature is categorical: coverage 'complete',
    # between_group_differences {}, features_not_assessed [],
    # features_not_compared [], zero warnings, byte-identical to a frame whose
    # every feature was measured and whose groups all agreed.
    features_not_numeric = [
        {
            "feature": str(c),
            "dtype": str(df[c].dtype) if c in df.columns else "absent",
            "reason": (
                f"{c!r} is not numeric (dtype "
                f"{str(df[c].dtype) if c in df.columns else 'absent'}), so no mean, "
                f"no within-group correlation and no between-group difference could be "
                f"computed for it. This is a could-not-check about that feature, not a "
                f"finding that the groups agree about it."
            ),
        }
        for c in feature_columns
        if c not in numeric_features
    ]

    df = df.copy()
    df["_intersect"] = df[protected_attributes].apply(
        lambda row: "_".join(str(v) for v in row.values), axis=1
    )
    # AN UNRECORDED GROUP IS NOT A GROUP. The join above runs str() over the raw
    # values, so a row whose protected value was never recorded becomes the
    # literal group key 'None' (or 'nan'), and it was then measured against the
    # real groups. BGL5 2026-09-27, measured on 40 rows of gender 'm' beside 40
    # rows of gender None, x drawn around 50 and 80: intersectional_groups
    # ['m', 'None'], coverage 'complete', warnings [] and
    # between_group_differences['x']['max_difference'] 31.43, i.e. the largest
    # reported difference in the frame was between a group and the absence of a
    # record. It now reports intersectional_groups ['m'], 40 rows without a
    # recorded group, and a warning. ``_fit_propensity_model`` in this same
    # package refuses the identical input by name (2026-09-17); this function was
    # the one that did not.
    _recorded = df[protected_attributes].notna().all(axis=1)
    n_rows_without_a_recorded_group = int((~_recorded).sum())
    if n_rows_without_a_recorded_group:
        # NaN, so value_counts drops them and no `_intersect == group` test can
        # ever match them: they are excluded from the groups AND from the means.
        df.loc[~_recorded, "_intersect"] = np.nan
        warnings.warn(
            f"analyze_intersectional_correlations: {n_rows_without_a_recorded_group} of "
            f"{len(df)} row(s) have no recorded value for one or more of "
            f"{list(protected_attributes)}. They are NOT a group, so they are excluded "
            f"from intersectional_groups, from within_group_correlations and from "
            f"between_group_differences. Their absence is a could-not-check about those "
            f"rows, not a finding that they match any group. See "
            f"result['rows_without_a_recorded_group'].",
            UserWarning,
            stacklevel=2,
        )

    group_counts = df["_intersect"].value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index.tolist()

    # BGL3 2026-09-27. Every group under `min_group_size` was dropped with no
    # trace, so a frame in which NOTHING qualified returned
    # `within_group_correlations: {}` and `between_group_differences: {}` with
    # no warning. Measured on 20 rows across four gender x race cells (sizes 2,
    # 5, 6, 7) at the default min_group_size=30: both dicts empty, output
    # byte-identical to a frame whose intersectional groups were all examined
    # and all agreed. An absence of examined groups is not a finding that the
    # groups agree.
    groups_not_assessed = [
        {
            "group": str(g),
            "n": int(group_counts[g]),
            "reason": (
                f"only {int(group_counts[g])} row(s), and the analysis needs "
                f"min_group_size={min_group_size}"
            ),
        }
        for g in group_counts.index
        if g not in set(valid_groups)
    ]
    # BGL6, 2026-09-29: ``not numeric_features`` is a could-not-check about the
    # WHOLE analysis, exactly like ``not valid_groups``. Nothing downstream can
    # measure anything, so 'complete' would be satisfied by having nothing to do.
    if not valid_groups or not numeric_features:
        coverage = NOT_ASSESSED
    elif groups_not_assessed:
        coverage = "partial"
    else:
        coverage = "complete"

    results = {
        "intersectional_groups": valid_groups,
        "group_sizes": group_counts.to_dict(),
        "within_group_correlations": {},
        "between_group_differences": {},
        "groups_not_assessed": groups_not_assessed,
        # Per (group, feature): a feature can be unmeasurable inside a group that
        # WAS examined, and that gap is invisible in groups_not_assessed, which
        # is per group only. See the note on the loop below.
        "features_not_assessed": [],
        # Features that no between-group difference could be computed for,
        # because fewer than two examined groups measured them.
        "features_not_compared": [],
        # Requested features this analysis cannot read as numbers. A third fact,
        # beside the per-group and the per-(group, feature) ones: with none of the
        # requested features numeric, NOTHING below runs and the other two lists
        # are empty for want of anything to record.
        "features_not_numeric": features_not_numeric,
        "rows_without_a_recorded_group": n_rows_without_a_recorded_group,
        "coverage": coverage,
    }

    if not numeric_features:
        # BGL6, 2026-09-29. Said out loud, because the reassuring half of the old
        # two-state answer was total silence: an empty between_group_differences
        # with coverage 'complete' and no warning at all.
        _ndetail = (
            "; ".join(f"{e['feature']} ({e['dtype']})" for e in features_not_numeric[:5])
            or "none were requested"
        )
        warnings.warn(
            f"analyze_intersectional_correlations: none of the "
            f"{len(feature_columns)} requested feature column(s) is numeric "
            f"({_ndetail}), so NOTHING was measured. within_group_correlations and "
            f"between_group_differences are empty because no feature could be read, "
            f"not because the groups agree, and coverage is "
            f"{NOT_ASSESSED!r} rather than 'complete'. See "
            f"result['features_not_numeric'].",
            UserWarning,
            stacklevel=2,
        )

    # The group-level disclosure. The condition is the old ``coverage !=
    # "complete"`` written out: it must NOT fire for the no-numeric-feature case
    # added above, because its sentence would then read "0 were NOT examined",
    # which is true and beside the point.
    if groups_not_assessed or not valid_groups:
        _detail = "; ".join(f"{e['group']} (n={e['n']})" for e in groups_not_assessed[:5])
        _more = (
            f" (and {len(groups_not_assessed) - 5} more)" if len(groups_not_assessed) > 5 else ""
        )
        warnings.warn(
            f"analyze_intersectional_correlations: {len(valid_groups)} of "
            f"{len(group_counts)} intersectional group(s) reached "
            f"min_group_size={min_group_size}, so "
            f"{len(groups_not_assessed)} were NOT examined: {_detail}{_more}. "
            f"Their absence from within_group_correlations and from "
            f"between_group_differences is a could-not-check, not a finding that "
            f"they match the rest. See result['groups_not_assessed'] and "
            f"result['coverage'].",
            UserWarning,
            stacklevel=2,
        )

    for group in valid_groups:
        group_df = df[df["_intersect"] == group]
        group_corrs = {}

        for feature in numeric_features:
            feature_vals = group_df[feature].dropna()
            if len(feature_vals) < MIN_SAMPLE_SIZE:
                # The disclosure above is per GROUP and this drop is per
                # FEATURE, so it left no trace at all. BGL5 2026-09-27, measured
                # on 40 rows of gender 'm' with x measured beside 40 rows of
                # gender 'f' whose x is entirely NaN: coverage read 'complete',
                # groups_not_assessed [], warnings [], within_group_correlations
                # {'m': ['x','y'], 'f': ['y']} and between_group_differences
                # holding ONLY y. Feature x was measured in one group, was
                # unmeasurable in the other, and vanished from the between-group
                # answer while the coverage word still said every group was
                # examined. Both facts are now recorded and coverage degrades.
                results["features_not_assessed"].append(
                    {
                        "group": str(group),
                        "feature": str(feature),
                        "n": int(len(feature_vals)),
                        "reason": (
                            f"only {len(feature_vals)} non-null row(s) of {feature!r} in "
                            f"group {group!r}, and the analysis needs "
                            f"MIN_SAMPLE_SIZE={MIN_SAMPLE_SIZE}"
                        ),
                    }
                )
                continue

            group_corrs[feature] = {
                "mean": float(feature_vals.mean()),
                "std": float(feature_vals.std()),
                "count": len(feature_vals),
            }

        results["within_group_correlations"][group] = group_corrs

    for feature in numeric_features:
        means = []
        for group in valid_groups:
            if feature in results["within_group_correlations"].get(group, {}):
                means.append(results["within_group_correlations"][group][feature]["mean"])

        if len(means) < 2 and valid_groups:
            # A feature that vanishes from between_group_differences is not a
            # feature the groups agreed about. Recorded for the same reason as
            # the per-group gap above: absence of a comparison is not agreement.
            results["features_not_compared"].append(
                {
                    "feature": str(feature),
                    "n_groups_measured": len(means),
                    "reason": (
                        f"{len(means)} of the {len(valid_groups)} examined group(s) "
                        f"measured {feature!r}, and a between-group difference needs 2. "
                        f"No difference was computed, which is a could-not-check and not "
                        f"a finding that the groups agree."
                    ),
                }
            )
        if len(means) >= 2:
            # A coefficient of variation is std / mean, which is undefined when
            # the mean is zero and meaningless in sign when it is negative. The
            # fallback was a literal 0, and 0 dispersion is PERFECT AGREEMENT
            # between the groups, the strongest all-clear this number can give.
            # Measured 2026-09-27 on 400 rows whose four intersectional groups
            # had means -99.94, -99.98, -1.00, -1.03: this reported
            # coefficient_of_variation 0 beside max_difference 98.98, so the
            # single largest between-group gap in the frame read as no variation
            # at all. Divide by the magnitude of the mean, which is what a CV on
            # a signed quantity means, and report nan when there is no magnitude
            # to divide by.
            mean_of_means = float(np.mean(means))
            if abs(mean_of_means) > 0:
                cv = float(np.std(means) / abs(mean_of_means))
            else:
                cv = float("nan")
                warnings.warn(
                    f"analyze_intersectional_correlations: the group means of "
                    f"'{feature}' average to 0, so no coefficient of variation "
                    f"exists for it. Reporting nan (could not check), NOT 0, which "
                    f"would read as the groups agreeing exactly. max_difference is "
                    f"still measured beside it.",
                    UserWarning,
                    stacklevel=2,
                )
            results["between_group_differences"][feature] = {
                "max_difference": max(means) - min(means),
                "coefficient_of_variation": cv,
            }

    # Coverage is decided ONLY here, after both loops, because the feature-level
    # gaps above are not known until they have run. Deciding it before them is
    # how 'complete' came to be published over a feature that was never measured
    # in one of the examined groups (BGL5 2026-09-27, see the loop). 'complete'
    # now means what the docstring says: every group examined AND every numeric
    # feature measured in each of them and compared across them.
    if coverage == "complete" and (
        results["features_not_assessed"] or results["features_not_compared"]
    ):
        coverage = "partial"
        results["coverage"] = coverage
    if results["features_not_assessed"] or results["features_not_compared"]:
        _fdetail = "; ".join(
            f"{e['feature']} in {e['group']} (n={e['n']})"
            for e in results["features_not_assessed"][:5]
        )
        _cdetail = "; ".join(
            f"{e['feature']} ({e['n_groups_measured']} group(s) measured it)"
            for e in results["features_not_compared"][:5]
        )
        warnings.warn(
            "analyze_intersectional_correlations: "
            + "; ".join(
                part
                for part in (
                    (
                        f"{len(results['features_not_assessed'])} feature/group pair(s) were "
                        f"NOT measured inside an examined group: {_fdetail}"
                        if results["features_not_assessed"]
                        else ""
                    ),
                    (
                        f"{len(results['features_not_compared'])} feature(s) could not be "
                        f"compared between groups: {_cdetail}"
                        if results["features_not_compared"]
                        else ""
                    ),
                )
                if part
            )
            + ". Their absence from between_group_differences is a could-not-check, not a "
            "finding that the groups agree about them. See "
            "result['features_not_assessed'], result['features_not_compared'] and "
            "result['coverage'].",
            UserWarning,
            stacklevel=2,
        )

    return results


# Helper functions


def _is_finite_number(value: Any) -> bool:
    """True only for a real, finite number.

    Coerces through ``float`` rather than testing ``isinstance(value, (int,
    float))``, because that test REJECTS ``np.float64`` / ``np.int64``, which is
    what every cell of a pandas frame actually holds; a predicate that rejects
    them would discard real measurements while reading as caution. Anything that
    will not coerce, and NaN / inf, are False.
    """
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _is_numeric(series: pd.Series) -> bool:
    """Check if series is numeric.

    Uses pandas' own dtype check, not np.issubdtype, which raises TypeError on a
    pandas extension dtype (the pandas-3 string dtype, nullable Int, Arrow) and
    would then be swallowed into a False that silently drops the column.
    """
    return bool(pd.api.types.is_numeric_dtype(series))


def _encode_for_correlation(series: pd.Series) -> Optional[np.ndarray]:
    """Encode series for correlation computation."""
    try:
        if pd.api.types.is_numeric_dtype(series):
            return series.to_numpy().astype(float)
        else:
            return pd.Categorical(series).codes.astype(float)
    except Exception:
        return None


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation with p-value."""
    try:
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < MIN_SAMPLE_SIZE:
            return np.nan, np.nan
        corr, pval = stats.pearsonr(x[mask], y[mask])
        return float(corr), float(pval)
    except Exception:
        return np.nan, np.nan


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation with p-value."""
    try:
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < MIN_SAMPLE_SIZE:
            return np.nan, np.nan
        corr, pval = stats.spearmanr(x[mask], y[mask])
        return float(corr), float(pval)
    except Exception:
        return np.nan, np.nan


def _point_biserial(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Compute point-biserial correlation (numeric vs binary)."""
    try:
        x_numeric = _is_numeric(x)

        if x_numeric:
            continuous = x.values.astype(float)
            binary = pd.Categorical(y).codes.astype(float)
        else:
            continuous = y.values.astype(float)
            binary = pd.Categorical(x).codes.astype(float)

        # For binary, use point-biserial (same as Pearson for 0/1)
        mask = ~(np.isnan(continuous) | np.isnan(binary))
        if mask.sum() < MIN_SAMPLE_SIZE:
            return np.nan, np.nan

        corr, pval = stats.pointbiserialr(binary[mask], continuous[mask])
        return float(corr), float(pval)
    except Exception:
        return np.nan, np.nan


def _cramers_v_with_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Compute Cramér's V with bias correction and chi-squared p-value.

    Returns ``(nan, nan)`` when the bias-corrected statistic is NOT IDENTIFIED,
    never ``0.0``.

    BGL-S2b (2026-09-17). Two fabricated zeros lived here, both of them the
    weakest association on the scale, which grades NEGLIGIBLE and recommends no
    action:

    (a) ``min_corr <= 0`` fell to an ``else: cramers_v = 0.0``. That branch is
        reached when Bergsma's correction has consumed the whole table: for a
        column with ONE ROW PER LEVEL against a 60-level attribute over 60 rows
        the table is 60x60 of ones, ``r_corr = k_corr = 1.0``, ``min_corr =
        0.0``, and V came back a clean, measured 0.0. There is nothing left to
        divide by; that is could-not-check, not independence. Measured at the
        public entry before this change: a unique-per-row ``zipcode`` against a
        60-level ``race`` reported ``cramers_v: 0.0`` in the same evidence dict
        as ``mutual_information: 0.582`` bits.
    (b) ``max(0, phi2 - ...)`` is the BUILTIN max, and ``max(0, nan)`` is 0
        because every comparison against NaN is False and max keeps its first
        argument. A non-finite chi-square therefore became an exact zero. The
        module bans this idiom by name in ``_correlation_ratio_with_pvalue``
        and it was still live three functions away. ``np.maximum`` propagates
        NaN, which is the state a chi-square that did not compute belongs in.

    A phi2_corr of 0.0 with ``min_corr > 0`` is NOT refused: Bergsma's
    estimator is genuinely clamped at zero for a weak association, and that is
    a measured answer.
    """
    try:
        contingency = pd.crosstab(pd.Categorical(x), pd.Categorical(y))
        if contingency.size == 0:
            return np.nan, np.nan

        chi2, pval, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.sum().sum()
        r, k = contingency.shape
        min_dim = min(r, k) - 1

        if min_dim <= 0 or n <= 1:
            return np.nan, np.nan

        # Bias-corrected Cramér's V (Bergsma 2013)
        phi2 = chi2 / n
        # np.maximum, NOT the builtin max: see (b) above.
        phi2_corr = float(np.maximum(0.0, phi2 - ((r - 1) * (k - 1)) / (n - 1)))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        min_corr = min(r_corr, k_corr) - 1

        if min_corr <= 0:
            warnings.warn(
                f"Cramér's V COULD NOT BE MEASURED on a {r}x{k} table over {n} "
                f"observation(s): the bias correction leaves no effective dimension "
                f"(min_corr={min_corr:.4g}), which happens when the levels are as "
                f"numerous as the rows. Returning nan, not 0.0, which reads as a "
                f"measured independence. Bucket the high-cardinality side first.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan

        cramers_v = float(np.sqrt(phi2_corr / min_corr))
        if not np.isfinite(cramers_v):
            return np.nan, np.nan

        return cramers_v, float(pval)
    except Exception:
        return np.nan, np.nan


def _correlation_ratio_with_pvalue(values: pd.Series, categories: pd.Series) -> Tuple[float, float]:
    """Correlation ratio eta in [0,1] for a CONTINUOUS variable across the
    levels of a CATEGORICAL grouping, with a one-way ANOVA F-test p-value.

    The correct association measure for the mixed continuous-vs-categorical
    case. Point-biserial (Tate 1954) is defined only for a DICHOTOMOUS
    categorical side; for a 3+ level nominal attribute it degenerates to
    Pearson on arbitrary category codes and a near-perfect continuous proxy
    reads as 'weak'. Cramér's V needs both sides categorical and collapses to
    0 when a continuous feature is treated as categorical. eta
    (Fisher/Pearson correlation ratio) with a one-way ANOVA p-value is the
    general measure (Datta et al. 2017; Barocas & Selbst 2016). Returns
    (nan, nan) when it cannot be computed.

    Non-finite values (inf/-inf as well as NaN) are dropped before the sums.
    Measured 2026-09-10, when only NaN was dropped: a single inf in the
    continuous column made grand_mean and ss_total non-finite, ss_between
    NaN, and then the builtin ``max(0.0, nan)`` returned 0.0 -- Python's max
    keeps its FIRST argument when the comparison is False, and every
    comparison against NaN is False. eta therefore read as a clean,
    measured 0.0 and the ANOVA's NaN p-value was forced to 1.0, so both
    gates in the caller (the correlation threshold and the significance
    test) rejected in silence: a CRITICAL race proxy at r=0.9993,
    p=3.5e-283 went from one reported finding to ZERO. The clamp now uses
    np.maximum, which propagates NaN, and an ANOVA that cannot be computed
    returns NaN rather than a confident 1.0. A genuinely measured 0.0 is a
    real answer and is still returned as one (ss_total == 0 below).

    ONE ROW PER LEVEL IS NOT A MEASUREMENT (Stage 2, s2g07, 2026-09-16). eta
    is the share of the continuous variable's variance that lies BETWEEN the
    levels, so it needs variation WITHIN a level to have anything to compare
    against. When every observation is its own group, ``ss_between`` equals
    ``ss_total`` identically and eta is sqrt(1) = 1.0 for any numbers at all.

    CONTROL NUMBERS, re-measured 2026-09-17 (the figures first written here,
    0.334933 and 0.134669, did not match the stated recipe and are corrected).
    The frame is 60 rows: ``ids = [f"cust-{i:03d}" for i in range(60)]`` (60
    singleton levels), ``city = np.tile(["Zurich", "Bern", "Geneva"], 20)`` (3
    levels), and ``age = np.random.default_rng(seed).normal(45, 12, 60)``:

        seed 0   eta(age ~ ids) refused (nan)   eta(age ~ city) = 0.095829
        seed 7   eta(age ~ ids) refused (nan)   eta(age ~ city) = 0.312773

    The genuine 3-level column still measures, and it measures DIFFERENTLY for
    the two age vectors, which is what a measurement does. Before the guard,
    both id columns read exactly 1.000000 regardless of the numbers, because
    that value is arithmetic and not data. A perfect 1.000 is the loudest cell
    in a correlation matrix a human reads or plots as a heatmap. The guard sits
    HERE, above both call sites, because ``_analyze_proxy_relationship`` reaches
    the same function by a different route and would otherwise keep the
    fabricated value.

    NOT REACHABLE FROM HERE ANY MORE (2026-09-17): the ANOVA-refusal branch at
    the foot of this function. ``stats.f_oneway`` returns a NaN p-value in
    exactly two situations, every input array of length 1 and every value in
    the whole sample identical, and both are now caught above it (the first by
    ``_largest_level < 2``, the second by ``ss_total <= 0``). The branch is kept
    because it is the correct disposition if either guard is ever narrowed, and
    because ``f_oneway`` may raise; a forced ``1.0`` there would be a claim of
    "definitely not significant" about a test that never ran, which is what
    deleted a CRITICAL race proxy at the caller's significance gate.
    """
    try:
        cat_name = getattr(categories, "name", None)
        vals = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
        cats = pd.Series(np.asarray(categories)).reset_index(drop=True)
        # np.isfinite, not ~isna: inf is NOT NaN and survived the old mask.
        mask = np.isfinite(vals.to_numpy(dtype=float))
        vals = vals[mask]
        cats = cats[mask]
        if len(vals) < MIN_SAMPLE_SIZE:
            return np.nan, np.nan

        grand_mean = vals.mean()
        ss_total = float(((vals - grand_mean) ** 2).sum())
        if ss_total <= 0:
            # ZERO VARIANCE IS NOT "NO ASSOCIATION" (2026-09-25). This branch
            # returned ``0.0, 1.0`` under the comment "no variance in the
            # continuous variable: no association". eta is ss_between / ss_total,
            # so with ss_total == 0 the statistic is 0/0: not zero, undefined. And
            # the 1.0 is a p-value for an ANOVA that never ran, which is the exact
            # claim the note at the foot of this docstring refuses ("definitely not
            # significant" about a test that never happened, the thing that deleted
            # a CRITICAL race proxy at the caller's significance gate).
            #
            # It was also internally inconsistent, and that is how it was found. A
            # constant feature against a TWO-level attribute goes to
            # ``_point_biserial``, where scipy returns nan, so the same undefined
            # quantity read as nan for two groups and as a confident
            # ``corr=0.0, pvalue=1.0`` for three. Measured 2026-09-25 at the public
            # entry ``compute_feature_correlations``.
            #
            # A constant column cannot in fact be a proxy, so nothing downstream
            # changes: nan fails every ``>= threshold`` gate exactly as 0.0 did.
            # What changes is what the matrix SAYS, from a measurement that was
            # never taken to a visible hole.
            warnings.warn(
                f"correlation ratio (eta) COULD NOT BE MEASURED against categorical "
                f"column {cat_name!r}: the continuous variable has zero variance, so "
                f"eta is 0/0 and undefined. Returning nan, not a correlation of 0.0. "
                f"A constant column carries no information and cannot be a proxy, but "
                f"that is a property of the column, not a measured association.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan

        groups = [g.to_numpy() for _, g in vals.groupby(cats.to_numpy())]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return np.nan, np.nan

        # eta is not identified without within-level variation. Two ways it
        # can be missing, both of which make the result an artefact of the
        # PARTITION rather than of the data:
        #   (a) no level holds two observations, so ss_between == ss_total
        #       identically and eta is exactly 1.0 whatever the numbers are;
        #   (b) the levels outnumber the pairs, so the between-level degrees
        #       of freedom dominate the residual ones and eta is inflated
        #       towards 1.0 by the same mechanism, just not all the way.
        # Refuse both. nan lands in the correlation matrix as a visible hole
        # and nan fails every downstream `>= threshold` gate, which is the
        # correct reading: this cell was not measured.
        _largest_level = max(len(g) for g in groups)
        _too_many_levels = len(groups) > len(vals) / 2
        if _largest_level < 2 or _too_many_levels:
            _why = (
                "no level holds more than one observation, so the correlation ratio is "
                "exactly 1.0 by arithmetic and says nothing about the data"
                if _largest_level < 2
                else (
                    f"{len(groups)} levels over {len(vals)} observations leaves too "
                    "little variation within a level for the correlation ratio to be "
                    "identified"
                )
            )
            warnings.warn(
                f"correlation ratio (eta) COULD NOT BE MEASURED against categorical "
                f"column {cat_name!r}: {_why}. Returning nan, not a correlation. "
                f"A high-cardinality identifier column is not a proxy finding; drop it "
                f"or bucket it before reading this matrix.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan

        ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
        # np.maximum, NOT the builtin max: max(0.0, nan) is 0.0, which turns
        # an uncomputable eta into a confident, clean-looking zero.
        eta = float(np.sqrt(np.maximum(0.0, ss_between) / ss_total))
        if not np.isfinite(eta):
            return np.nan, np.nan

        # An ANOVA that cannot be computed returns NaN: a forced 1.0 is a
        # measurement claim ("definitely not significant") the data does not
        # support, and it silently deleted findings at the caller's gate.
        try:
            _, pval = stats.f_oneway(*groups)
            pval = float(pval)
            if not np.isfinite(pval):
                pval = np.nan
        except Exception:
            pval = np.nan

        return eta, pval
    except Exception:
        return np.nan, np.nan


def _compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information (bits) between two arrays.

    RAISES rather than returning a number when it cannot be computed. The
    twin of this function in ``bias_detection.proxy`` has always raised, and
    its caller omits the key; this copy used to end in
    ``except Exception: return 0.0``, and 0.0 is not "unknown", it is the
    value that MEANS statistically independent. Measured 2026-09-10: two
    PERFECTLY DEPENDENT string series (``['a','b','c','d']*50`` against
    ``['A','B','C','D']*50``) came back as 0.0 -- pd.cut cannot bin strings,
    the DTypePromotionError was swallowed, and the report stated
    independence for a pair with a one-to-one mapping. The caller now
    records the failure instead of publishing a fabricated zero.
    """
    n_bins = min(MI_BINS, len(np.unique(x)), len(np.unique(y)))
    if n_bins < 2:
        # A constant column genuinely carries no information: a measured 0.
        return 0.0

    x_binned = pd.cut(x, bins=n_bins, labels=False, duplicates="drop")
    y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates="drop")

    # Handle NaN from binning
    mask = ~(pd.isna(x_binned) | pd.isna(y_binned))
    if mask.sum() < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"mutual information needs at least {MIN_SAMPLE_SIZE} paired "
            f"observations after binning; got {int(mask.sum())}"
        )

    x_binned = x_binned[mask].astype(int)
    y_binned = y_binned[mask].astype(int)

    joint = pd.crosstab(x_binned, y_binned, normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            pxy = joint.loc[i, j]
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px[i] * py[j]))

    return max(0.0, float(mi))


def _analyze_proxy_relationship(
    df: pd.DataFrame,
    feature: str,
    protected_attr: str,
    *,
    correlation_threshold: float,
    include_mutual_information: bool,
    include_known_patterns: bool,
    min_sample_size: int,
    significance_level: float = 0.05,
) -> Optional[ProxyVariableResult]:
    """Analyze potential proxy relationship with safe index alignment."""
    try:
        # Use safe alignment by working with aligned DataFrame subset
        aligned_df = df[[feature, protected_attr]].dropna()

        if len(aligned_df) < min_sample_size:
            return None

        feature_aligned = aligned_df[feature]
        attr_aligned_series = aligned_df[protected_attr]

        # Encode the attribute HERE, over the pairwise-aligned rows. The caller
        # holds an encoding of the attribute's own dropna() rows, which is a
        # DIFFERENT row set (it keeps rows where the feature is missing), so
        # reusing it would silently misalign the two vectors. It used to be
        # passed in as `attr_encoded` and then ignored for exactly this
        # reason; the parameter is gone rather than accepted and unread.
        attr_aligned = _encode_for_correlation(attr_aligned_series)
        if attr_aligned is None:
            return None

        feature_is_numeric = _is_numeric(feature_aligned)
        attr_is_numeric = _is_numeric(df[protected_attr])

        feature_encoded = _encode_for_correlation(feature_aligned)
        if feature_encoded is None:
            return None

        correlation_results = _compute_all_correlations(
            feature_encoded,
            attr_aligned,
            feature_aligned,
            attr_aligned_series,
            feature_is_numeric,
            attr_is_numeric,
            include_mutual_information,
        )

        # BGL-S2b (2026-09-17). Each of these three used to be
        # ``.get(<measure>, 0)``. The key is ABSENT precisely when the measure
        # REFUSED to be computed (the eta guard above, a Cramer's V that came
        # back NaN, a Pearson that could not run on the aligned rows), so the
        # default turned every refusal into a measured zero, which is the
        # weakest correlation on the scale and grades NEGLIGIBLE: "Minimal
        # risk. No action needed." Measured at the public entry on 60 rows with
        # a 60-level attribute: `identify_proxy_variables` returned
        # correlation=0, risk_level=NEGLIGIBLE, correlation_type="Correlation
        # ratio (eta)" and mutual_information=0.582 bits for a pair whose eta
        # had just been refused by name in a UserWarning.
        #
        # NaN, not 0: `FeatureEngineeringAnalyzer._partition_gradeable` splits
        # on `is_measured(proxy.correlation)`, and `is_measured(0)` is True, so
        # the zero walked straight through the boundary that exists to catch
        # exactly this. The record is still RETURNED, never dropped: its
        # evidence carries the other measures, and a deleted row is a finding a
        # reader can never get back.
        _refused_measure: Optional[str] = None
        if feature_is_numeric and attr_is_numeric:
            correlation = correlation_results.get("pearson", np.nan)
            correlation_type = "Pearson"
            if "pearson" not in correlation_results:
                _refused_measure = "pearson"
        elif feature_is_numeric != attr_is_numeric:
            # Continuous vs categorical: correlation ratio (eta), not
            # Cramér's V (which collapses to 0 for a continuous feature
            # treated as categorical). ANOVA p-value replaces the degenerate
            # chi-square p-value.
            correlation = correlation_results.get("correlation_ratio", np.nan)
            correlation_type = "Correlation ratio (eta)"
            if "correlation_ratio" not in correlation_results:
                _refused_measure = "correlation_ratio"
            if "eta_pvalue" in correlation_results:
                correlation_results["pvalue"] = correlation_results["eta_pvalue"]
            else:
                # eta was measured but its ANOVA p-value could not be. The
                # Pearson p sitting in results is computed on arbitrary
                # CATEGORY CODES and is not a substitute, so it is removed
                # rather than reused: the significance gate below is then
                # skipped and the finding is reported with pvalue=None
                # (could-not-check), never deleted by a p that was never
                # about eta.
                correlation_results.pop("pvalue", None)
        else:
            correlation = correlation_results.get("cramers_v", np.nan)
            correlation_type = "Cramér's V"
            if "cramers_v" not in correlation_results:
                _refused_measure = "cramers_v"
            if "chi2_pvalue" in correlation_results:
                correlation_results["pvalue"] = correlation_results["chi2_pvalue"]

        correlation = float(correlation)
        if _refused_measure is not None:
            # Name it where a consumer reads it, not only in a warning.
            _missing = list(correlation_results.get("measures_not_computed", []))
            if _refused_measure not in _missing:
                _missing.append(_refused_measure)
            correlation_results["measures_not_computed"] = _missing
            correlation_results["headline_measure_status"] = NOT_ASSESSED
            warnings.warn(
                f"identify_proxy_variables: the headline association between "
                f"{feature!r} and {protected_attr!r} is {correlation_type}, and it "
                f"COULD NOT BE MEASURED on these rows. Reporting correlation=nan "
                f"(could not check), NOT 0, which would grade NEGLIGIBLE. The row is "
                f"still returned so its other evidence survives; read "
                f"evidence['correlation_measures']['measures_not_computed'].",
                UserWarning,
                stacklevel=3,
            )

        abs_corr = abs(correlation)

        # `nan < threshold` is False, so an unmeasured pair falls THROUGH this
        # gate rather than being deleted by it, and reaches the caller carrying
        # its NaN. That is deliberate: the gate is "measured and too weak to
        # report", and a refusal is neither.
        if abs_corr < correlation_threshold:
            if include_known_patterns:
                pattern_risk = _check_known_patterns(feature, protected_attr)
                if pattern_risk is None:
                    return None
            else:
                return None

        pvalue = correlation_results.get("pvalue")
        if pvalue is not None and significance_level < 1.0:
            if pvalue >= significance_level:
                return None

        risk_level = _determine_risk_level(
            abs_corr, feature, protected_attr, include_known_patterns
        )

        proxy_type = _determine_proxy_type(feature, protected_attr)

        attr_key = _attr_catalogue_key(protected_attr)
        affected_groups = _get_affected_groups(protected_attr)

        recommendations = _generate_proxy_recommendations(
            feature, protected_attr, risk_level, abs_corr
        )
        if _refused_measure is not None:
            # BGL-final d02 (2026-09-17). The VALUE is now nan, but the GRADE
            # stamped beside it is not: `_determine_risk_level` compares with
            # `>=`, every comparison against NaN is False, and it falls through
            # to NEGLIGIBLE, whose action is "Minimal risk. No action needed."
            # A reader of `to_dict()` sees risk_level 'negligible' and an empty
            # recommendations list, which is exactly what a clean pair looks
            # like. `FeatureEngineeringAnalyzer` already routes these rows to
            # `ungraded` rather than grading them, but a direct caller of this
            # public function gets no such help, so the refusal is stated in the
            # field a human actually reads. The enum is left alone on purpose:
            # adding a member would change the meaning of every exhaustive
            # mapping over ProxyRiskLevel in the package.
            recommendations.insert(
                0,
                f"NOT ASSESSED: the headline {correlation_type} association between "
                f"{feature!r} and {protected_attr!r} could not be measured on these "
                f"rows, so correlation is nan. The risk level of "
                f"{risk_level.value!r} is the grader's fall-through band for a value "
                f"that is not a number; it is NOT a finding of low risk. See "
                f"evidence['correlation_measures']['measures_not_computed'].",
            )
        if attr_key is None:
            # Disclose the could-not-check WHERE A READER SEES IT. The empty
            # affected_groups list was previously indistinguishable from a
            # catalogue entry with no groups.
            recommendations.append(_unmapped_attribute_note(protected_attr))

        return ProxyVariableResult(
            feature=feature,
            protected_attribute=protected_attr,
            correlation=correlation,
            correlation_type=correlation_type,
            risk_level=risk_level,
            proxy_type=proxy_type,
            mutual_information=correlation_results.get("mutual_information"),
            cramers_v=correlation_results.get("cramers_v"),
            pvalue=correlation_results.get("pvalue"),
            sample_size=len(aligned_df),
            affected_groups=affected_groups,
            recommendations=recommendations,
            evidence={
                "correlation_measures": correlation_results,
                "attribute_catalogue_key": attr_key,
                "pvalue_status": (
                    "computed"
                    if correlation_results.get("pvalue") is not None
                    else "could_not_compute"
                ),
            },
        )

    except Exception as e:
        warnings.warn(f"Failed to analyze proxy relationship for {feature}: {e}")
        return None


def _compute_all_correlations(
    feature_encoded: np.ndarray,
    attr_encoded: np.ndarray,
    feature_original: pd.Series,
    attr_original: pd.Series,
    feature_is_numeric: bool,
    attr_is_numeric: bool,
    include_mutual_information: bool,
) -> Dict[str, Any]:
    """Compute multiple correlation measures.

    Each measure gets its OWN try/except and every failure is NAMED in
    ``measures_not_computed``. One blanket ``except Exception: pass`` used to
    wrap the whole block, so a failure in the middle (measured 2026-09-10 by
    raising inside the eta branch) returned a partially populated dict --
    Cramer's V and Pearson present, eta and mutual information simply
    absent -- with nothing to distinguish "not applicable here" from "the
    computation blew up". The twin in ``bias_detection.proxy`` already used
    per-measure handling; this brings the copy in line and adds the marker.
    """
    results: Dict[str, Any] = {}
    failed: List[str] = []
    log = logging.getLogger(__name__)

    try:
        mask = ~(np.isnan(feature_encoded) | np.isnan(attr_encoded))
    except Exception:
        log.debug("correlation mask could not be built", exc_info=True)
        results["measures_not_computed"] = ["pearson", "spearman", "mutual_information"]
        return results

    if mask.sum() >= MIN_SAMPLE_SIZE:
        try:
            corr, pvalue = stats.pearsonr(feature_encoded[mask], attr_encoded[mask])
            results["pearson"] = float(corr)
            results["pvalue"] = float(pvalue)
        except Exception:
            log.debug("pearson could not be computed", exc_info=True)
            failed.append("pearson")
        try:
            spearman_corr, _ = stats.spearmanr(feature_encoded[mask], attr_encoded[mask])
            results["spearman"] = float(spearman_corr)
        except Exception:
            log.debug("spearman could not be computed", exc_info=True)
            failed.append("spearman")

    try:
        cramers_v, chi2_pval = _cramers_v_with_pvalue(feature_original, attr_original)
        if not np.isnan(cramers_v):
            results["cramers_v"] = cramers_v
            results["chi2_pvalue"] = chi2_pval
        else:
            # A REFUSED measure and a measure that raised are the same thing to
            # a reader: neither was computed. Both are named. Silently omitting
            # the key left "not applicable here" and "it would not compute"
            # indistinguishable, which is the marker this function was given.
            failed.append("cramers_v")
    except Exception:
        log.debug("cramers_v could not be computed", exc_info=True)
        failed.append("cramers_v")

    # Correlation ratio (eta) for the mixed continuous-vs-categorical
    # case. Cramér's V needs BOTH sides categorical; treating a
    # continuous feature as categorical drives the bias-corrected V to 0
    # and hides a continuous proxy of a categorical protected attribute
    # (Datta et al. 2017; Barocas & Selbst 2016).
    if feature_is_numeric != attr_is_numeric:
        try:
            if feature_is_numeric:
                eta, eta_pval = _correlation_ratio_with_pvalue(feature_original, attr_original)
            else:
                eta, eta_pval = _correlation_ratio_with_pvalue(attr_original, feature_original)
            if np.isfinite(eta):
                results["correlation_ratio"] = eta
                # Only when the ANOVA actually produced a p-value; a NaN
                # here must not be published as though it were measured.
                if np.isfinite(eta_pval):
                    results["eta_pvalue"] = eta_pval
            else:
                failed.append("correlation_ratio")
        except Exception:
            log.debug("correlation ratio could not be computed", exc_info=True)
            failed.append("correlation_ratio")

    if include_mutual_information:
        try:
            results["mutual_information"] = float(
                _compute_mutual_information(feature_encoded[mask], attr_encoded[mask])
            )
        except Exception:
            # The key is OMITTED, never set to 0.0: 0.0 is the value that
            # means statistically independent.
            log.debug("mutual information could not be computed", exc_info=True)
            failed.append("mutual_information")

    if failed:
        results["measures_not_computed"] = failed

    return results


def _attr_catalogue_key(protected_attr: str) -> Optional[str]:
    """Which KNOWN_PROXY_PATTERNS entry this attribute name refers to.

    ``None`` means COULD NOT CHECK -- the attribute is not in the catalogue,
    so no attribute-keyed evidence exists for it and the caller must say so
    rather than reporting the empty lookup as a clean result. See
    :func:`vfairness._names.resolve_attribute_key` for the measurement that
    forced this: with the previous ``key in attr_lower or attr_lower in key``
    substring test, renaming a column from 'gender' to 'sex' turned the same
    0.515 correlation from critical/direct/['Women', 'Non-binary
    individuals'] into high/unclassified/[], and 'sex' is the spelling used
    by both UCI Adult and COMPAS.
    """
    return resolve_attribute_key(protected_attr, KNOWN_PROXY_PATTERNS)


def _unmapped_attribute_note(protected_attr: str) -> str:
    """The disclosure text for an attribute the catalogue does not know."""
    return (
        f"COULD NOT CHECK: '{protected_attr}' matches no entry in the known-proxy "
        f"catalogue ({', '.join(sorted(KNOWN_PROXY_PATTERNS))}), so no attribute-keyed "
        "pattern check and no affected-group lookup were performed. An empty "
        "affected_groups list here means 'not looked up', NOT 'no one is affected'."
    )


def _check_known_patterns(feature: str, protected_attr: str) -> Optional[str]:
    """Check if feature matches known proxy patterns for THIS attribute."""
    feature_lower = feature.lower()
    attr_key = _attr_catalogue_key(protected_attr)
    if attr_key is None:
        # No catalogue entry means no attribute-keyed evidence. The old
        # substring test would otherwise match EVERY group ("" in "race"),
        # which promoted a 0.31 correlation to CRITICAL on the first pattern
        # that happened to match the feature name.
        return None

    patterns = KNOWN_PROXY_PATTERNS[attr_key]
    for keyword in patterns.get("high_risk", []):
        if keyword in feature_lower:
            return "high"
    for keyword in patterns.get("medium_risk", []):
        if keyword in feature_lower:
            return "medium"

    return None


def _determine_risk_level(
    correlation: float,
    feature: str,
    protected_attr: str,
    check_patterns: bool,
) -> ProxyRiskLevel:
    """
    Determine risk level based on correlation and known patterns.

    Uses module-level constants for threshold values:
    - CRITICAL: correlation >= 0.7 or high-risk pattern with correlation >= 0.3
    - HIGH: correlation >= 0.5
    - MEDIUM: correlation >= 0.3
    - LOW: correlation >= 0.1
    - NEGLIGIBLE: correlation < 0.1
    """
    if check_patterns:
        pattern_risk = _check_known_patterns(feature, protected_attr)
        if pattern_risk == "high" and correlation >= CORRELATION_THRESHOLD_MEDIUM:
            return ProxyRiskLevel.CRITICAL

    if correlation >= CORRELATION_THRESHOLD_CRITICAL:
        return ProxyRiskLevel.CRITICAL
    elif correlation >= CORRELATION_THRESHOLD_HIGH:
        return ProxyRiskLevel.HIGH
    elif correlation >= CORRELATION_THRESHOLD_MEDIUM:
        return ProxyRiskLevel.MEDIUM
    elif correlation >= CORRELATION_THRESHOLD_LOW:
        return ProxyRiskLevel.LOW
    else:
        return ProxyRiskLevel.NEGLIGIBLE


def _name_tokens(name: str) -> set:
    """snake/kebab/space/camelCase -> lowercase word tokens.

    Delegates to :func:`vfairness._names.name_tokens`. This helper used to
    be defined here AND copied byte-identically into
    ``feature_engineering.correlation``, while three further detectors kept
    their own, older variants; a fix landed in one copy and not the others.
    The shared module is the single source of truth so that cannot recur.

    Whole-token matching, never substring: "age" must not match "average"
    and "sex" must not match "sussex". That has to hold for ANY casing, and
    the first version of this helper did not: ``[A-Za-z][a-z]*`` split an
    ALL-CAPS name into single LETTERS, so almost any two upper-case names
    shared one and the caller handed out DIRECT, the strongest label in the
    enum, to unrelated columns. Measured 2026-09-09 before that fix:
    ``_determine_proxy_type("AVERAGE_SPEND", "AGE")`` and
    ``("SUSSEX_COUNTY", "SEX")`` both returned ``direct``, which is the exact
    pair of examples the paragraph above promises cannot happen.

    It also has to hold outside ASCII, and the ``[A-Za-z]+`` replacement did
    not: measured 2026-09-10, ``'género'`` tokenised to ``['g', 'nero']`` and
    ``'peso_g'`` to ``['g', 'peso']``, so ``_determine_proxy_type("peso_g",
    "género")`` returned DIRECT -- weight in grams "directly encoding"
    gender -- while ``("genero_cliente", "género")``, the real direct
    encoding, returned UNCLASSIFIED. Tokens are now taken with a Unicode
    word class and accent-folded, so genero and género are one token.
    """
    return name_tokens(name)


def _determine_proxy_type(
    feature: str,
    protected_attr: str,
    is_indirect: bool = False,
    intersectional_attrs: Optional[List[str]] = None,
) -> ProxyType:
    """
    Determine the type of proxy relationship.

    The label describes a RELATIONSHIP, so it needs evidence about BOTH sides.
    Keywords in the feature name alone say nothing about which attribute the
    feature stands in for: `credit_score` is a historical proxy for income and
    race, and says nothing about gender. This therefore consults
    ``protected_attr`` through KNOWN_PROXY_PATTERNS, which is keyed BY
    protected attribute.

    Args:
        feature: Feature name
        protected_attr: Protected attribute name the feature is being
            classified AGAINST. Read, not decorative.
        is_indirect: Whether this proxy was detected through a chain
        intersectional_attrs: List of attributes if this is an intersectional proxy

    Returns:
        DIRECT when the feature name encodes the protected attribute itself,
        or matches a known direct-encoding pattern FOR THAT ATTRIBUTE;
        HISTORICAL when it matches a known pattern for that attribute rooted
        in historical discrimination; otherwise UNCLASSIFIED. UNCLASSIFIED is
        a "could not determine", not a low-risk verdict: the correlation and
        the risk level are reported independently of it.
    """
    if intersectional_attrs and len(intersectional_attrs) > 1:
        return ProxyType.INTERSECTIONAL

    if is_indirect:
        return ProxyType.INDIRECT

    feature_lower = feature.lower()

    # The feature names the protected attribute itself (gender -> gender_code).
    attr_tokens = _name_tokens(protected_attr)
    if attr_tokens and (attr_tokens & _name_tokens(feature)):
        return ProxyType.DIRECT

    # Evidence about THIS protected attribute: KNOWN_PROXY_PATTERNS is keyed by
    # attribute, so a hit means the feature is a documented proxy for THIS
    # attribute, not merely a suspicious-looking column name.
    if _check_known_patterns(feature, protected_attr) is None:
        return ProxyType.UNCLASSIFIED

    # Historical proxies - features that encode historical discrimination
    historical_keywords = {
        "zip",
        "zipcode",
        "zip_code",
        "postal",
        "postcode",
        "neighborhood",
        "census",
        "census_tract",
        "block_group",
        "redline",
        "redlining",
        "school",
        "school_district",
        "college",
        "university",
        "credit_score",
        "credit_rating",
        "fico",
        "property_value",
        "home_value",
        "median_income",
    }
    if any(kw in feature_lower for kw in historical_keywords):
        return ProxyType.HISTORICAL

    # Direct proxies - features that directly encode protected attributes
    direct_keywords = {
        "name",
        "first_name",
        "last_name",
        "surname",
        "given_name",
        "title",
        "salutation",
        "prefix",
        "suffix",
        "gender",
        "sex",
        "male",
        "female",
        "birth",
        "dob",
        "age",
        "birth_year",
        "ethnicity",
        "race",
        "nationality",
        "country_of_origin",
    }
    if any(kw in feature_lower for kw in direct_keywords):
        return ProxyType.DIRECT

    # Known proxy for this attribute, but of no recognised kind. The risk is
    # reported by risk_level; the KIND stays unclassified rather than
    # defaulting to DIRECT, the strongest label in the enum.
    return ProxyType.UNCLASSIFIED


def _get_affected_groups(protected_attr: str) -> List[str]:
    """Groups potentially affected by a proxy for this attribute.

    An empty list is ambiguous on its own -- it is returned both for an
    attribute the catalogue knows and has no groups for, and for one the
    catalogue never heard of. Callers disclose the second case through
    :func:`_attr_catalogue_key` / :func:`_unmapped_attribute_note`.
    """
    attr_key = _attr_catalogue_key(protected_attr)
    if attr_key is None:
        # Same trap as _check_known_patterns: "" is a substring of every key,
        # so a blank attribute used to return race's affected groups.
        return []
    return KNOWN_PROXY_PATTERNS[attr_key].get("affected_groups", [])


def _generate_proxy_recommendations(
    feature: str,
    protected_attr: str,
    risk_level: ProxyRiskLevel,
    correlation: float,
) -> List[str]:
    """Generate recommendations for handling proxy variable."""
    recommendations = []

    if risk_level == ProxyRiskLevel.CRITICAL:
        recommendations.append(
            f"CRITICAL: '{feature}' is a strong proxy for '{protected_attr}' "
            f"(correlation: {correlation:.2f}). Consider removing this feature."
        )
        recommendations.append("Apply strict fairness constraints or fair representation learning.")
    elif risk_level == ProxyRiskLevel.HIGH:
        recommendations.append(
            f"HIGH RISK: '{feature}' shows high correlation with '{protected_attr}'. "
            "Evaluate whether this feature is essential."
        )
        recommendations.append("Consider fairness-aware feature selection or regularization.")
    elif risk_level == ProxyRiskLevel.MEDIUM:
        recommendations.append(
            f"MODERATE RISK: '{feature}' has moderate correlation with '{protected_attr}'. "
            "Monitor for disparate impact."
        )
    elif risk_level == ProxyRiskLevel.LOW:
        recommendations.append(
            f"LOW RISK: '{feature}' shows weak correlation. Continue monitoring."
        )

    return recommendations
