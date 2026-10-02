"""
D. Proxy Variable Identification

Detects features that serve as proxies for protected attributes, enabling
indirect discrimination even when protected attributes are removed.

Key capabilities:
- Multiple correlation measures (Pearson, Cramér's V, mutual information)
- Risk scoring for potential proxy discrimination
- Causal proxy detection (direct vs indirect proxies)
- Proxy chain analysis (A -> B -> C where B is protected)
- Recommendations for proxy handling

References:
    - Barocas & Selbst (2016): Big Data's Disparate Impact
    - Datta et al. (2017): Proxy Discrimination in Data-Driven Systems
    - Kilbertus et al. (2017): Avoiding Discrimination through Causal Reasoning
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from vfairness._names import name_tokens, resolve_attribute_key
from vfairness._triage import is_measured

# Column names that unambiguously denote the prediction target / decision.
# Used only to INFER the outcome column when the caller does not pass one
# and does not restrict feature_columns. Kept to exact matches on purpose:
# substring heuristics (e.g. 'score', 'status') would silently drop real
# proxy candidates such as credit_score.
_OUTCOME_NAME_HINTS = {
    "outcome",
    "target",
    "label",
    "y",
    "y_true",
    "y_pred",
    "y_hat",
    "prediction",
    "predicted",
    "decision",
}


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
    mutual_information: Optional[float]
    cramers_v: Optional[float]
    pvalue: Optional[float]
    sample_size: int
    affected_groups: List[str]
    recommendations: List[str]
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


# Known high-risk proxy patterns: features commonly known to serve as proxies
# for protected attributes.
KNOWN_PROXY_PATTERNS = {
    # Race/Ethnicity proxies
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
    # Gender proxies
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
    # Age proxies
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
    # Socioeconomic proxies
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
    # Disability proxies
    "disability": {
        "high_risk": [
            "accommodation_request",
            "accommodation",
            "ada_flag",
            "accessibility",
            "medical_leave",
            "fmla",
        ],
        "medium_risk": [
            "typing_speed",
            "response_time",
            "interaction_time",
            "session_duration",
            "error_rate",
            "completion_rate",
        ],
        "affected_groups": ["People with disabilities"],
    },
    # Religion proxies
    "religion": {
        "high_risk": [
            "name",
            "surname",
            "first_name",
            "holiday_preference",
            "time_off_requests",
        ],
        "medium_risk": [
            "dietary_preference",
            "diet",
            "neighborhood",
            "community",
        ],
        "affected_groups": ["Religious minorities"],
    },
    # National origin proxies
    "national_origin": {
        "high_risk": [
            "name",
            "surname",
            "accent",
            "language_proficiency",
            "country_code",
            "phone_prefix",
        ],
        "medium_risk": [
            "education_country",
            "degree_country",
            "work_authorization",
            "visa_status",
        ],
        "affected_groups": ["Immigrants", "Foreign nationals"],
    },
}


def identify_proxy_variables(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    feature_columns: Optional[List[str]] = None,
    outcome_column: Optional[str] = None,
    correlation_threshold: float = 0.3,
    include_mutual_information: bool = True,
    include_known_patterns: bool = True,
    min_sample_size: int = 100,
    max_cardinality: int = 50,
    significance_level: float = 0.05,
) -> List[ProxyVariableResult]:
    """
    Identify features that may serve as proxies for protected attributes.

    Uses multiple correlation measures and pattern matching to detect
    features that could enable indirect discrimination.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute column names
        feature_columns: Columns to test as potential proxies (auto-detect if None)
        outcome_column: The target/decision column. It is excluded from the
            proxy candidates: the outcome correlating with a protected
            attribute is outcome disparity, not proxy leakage, and flagging
            it produced 'remove the target' as the top recommendation. When
            None and feature_columns is None, columns whose names
            unambiguously denote a target (see _OUTCOME_NAME_HINTS) are
            excluded with a warning.
        correlation_threshold: Minimum correlation to flag as proxy
        include_mutual_information: Whether to compute mutual information
        include_known_patterns: Whether to check against known proxy patterns
        min_sample_size: Minimum sample size for reliable computation
        max_cardinality: Maximum unique values for a feature column. Columns
            with more unique values (e.g., IDs, timestamps) are skipped to
            avoid inflated Cramér's V on sparse contingency tables. Also
            skips columns where unique-to-rows ratio > 0.5.
        significance_level: P-value threshold for statistical significance.
            Only features with p-value < this threshold are flagged (default
            0.05). Set to 1.0 to disable p-value gating.

    Returns:
        List of ProxyVariableResult objects, sorted by risk level

    Example:
        >>> results = identify_proxy_variables(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     correlation_threshold=0.3
        ... )
        >>> for r in results:
        ...     print(f"{r.feature} -> {r.protected_attribute}:")
        ...     print(f"  Correlation: {r.correlation:.3f}")
        ...     print(f"  Risk: {r.risk_level.value}")

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

    Ledger row: identify_proxy_variables. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    results = []

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in protected_attributes]
        # Only in auto-detect mode: infer the target column by name so the
        # audit does not scan the outcome itself as a proxy candidate. An
        # explicit feature_columns list is respected as-is.
        if outcome_column is None:
            inferred = [c for c in feature_columns if str(c).strip().lower() in _OUTCOME_NAME_HINTS]
            if inferred:
                warnings.warn(
                    f"Excluding likely outcome column(s) {inferred} from "
                    f"proxy candidates. Pass outcome_column or "
                    f"feature_columns explicitly to override.",
                    UserWarning,
                    stacklevel=2,
                )
                feature_columns = [c for c in feature_columns if c not in inferred]

    # An explicitly named outcome column is never a proxy candidate,
    # regardless of how feature_columns was supplied.
    if outcome_column is not None:
        feature_columns = [c for c in feature_columns if c != outcome_column]

    # The feature-level twin of the attribute disclosure below, checked once here
    # rather than inside the loop so it is not repeated per attribute. A caller
    # who names a column that is not in the frame gets it dropped by the
    # `feature not in df.columns` continue in the loop, which is silent, so a
    # config listing five features to screen could screen four and report the
    # result as if it had screened five. Auto-detected feature lists are taken
    # from df.columns and can never reach this. The sibling lane already refuses
    # the same mistake outright (cicd.validator.validate raises on an unknown
    # feature column); a warning is the non-breaking form of that here.
    unknown_features = [c for c in feature_columns if c not in df.columns]
    if unknown_features:
        warnings.warn(
            f"feature_columns names {len(unknown_features)} column(s) that are "
            f"not in the frame: {unknown_features[:10]}. They were NOT screened "
            f"against any protected attribute and are absent from the result "
            f"for that reason, not because they are clean.",
            UserWarning,
            stacklevel=2,
        )

    for attr in protected_attributes:
        if attr not in df.columns:
            # An attribute that is not a column of the frame was never SCREENED,
            # and this `continue` used to be completely silent. The function
            # returns a LIST, so "screened and found nothing" and "never looked"
            # are the same empty list; the only thing that can separate them is
            # this warning. Measured 2026-09-17:
            # identify_proxy_variables(df, ['gender', 'age']) on a frame whose
            # column is spelled 'sex' returned [] with no warning of any kind,
            # i.e. a clean proxy report for two attributes nothing was measured
            # about. The too-few-samples gate directly below has always said
            # UNASSESSED out loud; this sibling had not.
            warnings.warn(
                f"Protected attribute '{attr}' is not a column of the frame "
                f"(columns: {sorted(str(c) for c in df.columns)[:20]}), so NO "
                f"feature was screened against it. Proxy risk is UNASSESSED "
                f"for this attribute, not clean.",
                UserWarning,
                stacklevel=2,
            )
            continue

        attr_series = df[attr].dropna()
        if len(attr_series) < min_sample_size:
            # Loud, not silent: an empty result must not read as 'no
            # proxies'. Below this floor even a literal copy of the
            # protected attribute would go unreported.
            warnings.warn(
                f"Too few samples to assess proxies for '{attr}' "
                f"(n={len(attr_series)} < min_sample_size={min_sample_size}). "
                f"Proxy risk is UNASSESSED for this attribute, not clean.",
                UserWarning,
                stacklevel=2,
            )
            continue

        attr_encoded = _encode_for_correlation(attr_series)
        if attr_encoded is None:
            # Same silent-skip class as the missing-column branch above.
            # _encode_for_correlation returns None for any column it cannot turn
            # into numbers (an object column holding lists, for instance), and
            # every feature then goes unscreened against this attribute while
            # the caller receives an ordinary empty list.
            warnings.warn(
                f"Protected attribute '{attr}' could not be encoded for "
                f"correlation (dtype={df[attr].dtype}), so NO feature was "
                f"screened against it. Proxy risk is UNASSESSED for this "
                f"attribute, not clean.",
                UserWarning,
                stacklevel=2,
            )
            continue

        skipped_small = []
        skipped_cardinality = []
        for feature in feature_columns:
            if feature not in df.columns or feature == attr:
                continue

            # Track features skipped for lack of overlapping samples so the
            # skip is reported, not silent (mirrors the attribute-level gate).
            n_overlap = len(df[[feature, attr]].dropna())
            if n_overlap < min_sample_size:
                skipped_small.append(feature)
                continue

            # High-cardinality filter: skip ID-like CATEGORICAL columns (e.g.,
            # applicant_id, timestamps) whose high unique-value count produces
            # sparse contingency tables and inflated Cramér's V values. This
            # applies ONLY to non-numeric features: a continuous feature
            # naturally has ~n unique values, and blanket-filtering it here
            # discarded the textbook continuous proxy (income/credit-score of
            # a categorical protected attribute) before it could be measured
            # by the correlation ratio, which is not inflated by cardinality.
            if not pd.api.types.is_numeric_dtype(df[feature]):
                n_unique = df[feature].nunique()
                n_rows = len(df[feature].dropna())
                if n_unique > max_cardinality or (n_rows > 0 and n_unique / n_rows > 0.5):
                    # Collected, not dropped in silence: the last of the four
                    # skips in this function that left an unscreened feature
                    # indistinguishable from a feature measured below the
                    # threshold. The exclusion is deliberate and correct (a
                    # sparse contingency table inflates Cramér's V), but the
                    # consequence is that this column carries NO proxy verdict,
                    # and only a warning can say so. Reported once per attribute
                    # with a count, exactly as skipped_small below.
                    skipped_cardinality.append(feature)
                    continue

            result = _analyze_proxy_relationship(
                df,
                feature,
                attr,
                attr_encoded,
                correlation_threshold=correlation_threshold,
                include_mutual_information=include_mutual_information,
                include_known_patterns=include_known_patterns,
                min_sample_size=min_sample_size,
                significance_level=significance_level,
            )

            if result:
                results.append(result)

        if skipped_small:
            warnings.warn(
                f"Too few overlapping samples to assess "
                f"{len(skipped_small)} feature(s) as proxies for '{attr}' "
                f"(< min_sample_size={min_sample_size}): "
                f"{skipped_small[:10]}. These are UNASSESSED, not clean.",
                UserWarning,
                stacklevel=2,
            )

        if skipped_cardinality:
            warnings.warn(
                f"{len(skipped_cardinality)} categorical feature(s) exceeded "
                f"max_cardinality={max_cardinality} (or hold a distinct value "
                f"for more than half the rows) and were NOT screened as proxies "
                f"for '{attr}': {skipped_cardinality[:10]}. These are "
                f"UNASSESSED, not clean. Bucket a high-cardinality column, or "
                f"raise max_cardinality, to bring it into the screen.",
                UserWarning,
                stacklevel=2,
            )

    risk_order = {
        ProxyRiskLevel.CRITICAL: 0,
        ProxyRiskLevel.HIGH: 1,
        ProxyRiskLevel.MEDIUM: 2,
        ProxyRiskLevel.LOW: 3,
        ProxyRiskLevel.NEGLIGIBLE: 4,
    }
    results.sort(key=lambda x: (risk_order[x.risk_level], -abs(x.correlation)))

    return results


def _analyze_proxy_relationship(
    df: pd.DataFrame,
    feature: str,
    protected_attr: str,
    attr_encoded: np.ndarray,
    *,
    correlation_threshold: float,
    include_mutual_information: bool,
    include_known_patterns: bool,
    min_sample_size: int,
    significance_level: float = 0.05,
) -> Optional[ProxyVariableResult]:
    """Analyze potential proxy relationship between feature and protected attribute."""
    try:
        feature_series = df[feature].dropna()
        if len(feature_series) < min_sample_size:
            return None

        common_idx = feature_series.index.intersection(df[protected_attr].dropna().index)
        if len(common_idx) < min_sample_size:
            return None

        feature_aligned = feature_series[common_idx]
        attr_aligned = attr_encoded[df[protected_attr].dropna().index.get_indexer(common_idx)]

        try:
            feature_is_numeric = pd.api.types.is_numeric_dtype(feature_aligned)
        except (TypeError, AttributeError):
            feature_is_numeric = False

        try:
            attr_is_numeric = pd.api.types.is_numeric_dtype(df[protected_attr])
        except (TypeError, AttributeError):
            attr_is_numeric = False

        feature_encoded = _encode_for_correlation(feature_aligned)
        if feature_encoded is None:
            return None

        correlation_results = _compute_all_correlations(
            feature_encoded,
            attr_aligned,
            feature_aligned,
            df[protected_attr][common_idx],
            feature_is_numeric,
            attr_is_numeric,
            include_mutual_information,
        )

        if feature_is_numeric and attr_is_numeric:
            correlation = correlation_results.get("pearson", float("nan"))
            correlation_type = "Pearson"
            # pvalue already set by Pearson
        elif feature_is_numeric != attr_is_numeric:
            # Continuous vs categorical: use the correlation ratio (eta), not
            # Cramér's V. Treating a continuous feature as categorical makes
            # every distinct value its own level, so the bias-corrected
            # Cramér's V collapses to 0 and a near-perfect continuous proxy
            # of a categorical protected attribute is missed. The ANOVA-based
            # eta p-value replaces the degenerate chi-square p-value.
            correlation = correlation_results.get("correlation_ratio", float("nan"))
            correlation_type = "Correlation ratio (eta)"
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
            correlation = correlation_results.get("cramers_v", float("nan"))
            correlation_type = "Cramér's V"
            # Use chi-squared p-value for Cramér's V (more appropriate
            # than the Pearson p-value on encoded category codes)
            if "chi2_pvalue" in correlation_results:
                correlation_results["pvalue"] = correlation_results["chi2_pvalue"]

        # A correlation that could not be MEASURED is not a correlation of zero.
        # Each of the three branches above reads its statistic out of a dict that
        # only carries the key when the statistic came out finite: eta, for
        # instance, is written `if np.isfinite(eta)` and is simply absent
        # otherwise. With a `0` default an unmeasurable association scored 0.0,
        # fell below `correlation_threshold`, and the pair was dropped by the
        # `return None` below with no trace at all, so the screen reported a
        # feature it could not assess exactly as it reports a clean one. That is
        # the defect this whole campaign exists to remove, sitting inside the
        # proxy detector itself.
        if not np.isfinite(correlation):
            warnings.warn(
                f"proxy screen for '{feature}' vs '{protected_attr}': the "
                f"{correlation_type} association could not be computed on this "
                f"data, so the pair is NOT SCREENED. This is a could-not-check, "
                f"not a finding that the feature is clean.",
                UserWarning,
                stacklevel=2,
            )
            return None

        abs_corr = abs(correlation)

        if abs_corr < correlation_threshold:
            if include_known_patterns:
                pattern_risk = _check_known_patterns(feature, protected_attr)
                if pattern_risk is None:
                    return None
            else:
                return None

        # P-value gating: reject features where the correlation is not
        # statistically significant.  This prevents false positives from
        # sparse contingency tables or small-sample noise.
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

        evidence = {
            "correlation_measures": correlation_results,
            "feature_cardinality": int(feature_aligned.nunique()),
            "attr_cardinality": int(df[protected_attr][common_idx].nunique()),
            "attribute_catalogue_key": attr_key,
            "pvalue_status": (
                "computed" if correlation_results.get("pvalue") is not None else "could_not_compute"
            ),
        }
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
            sample_size=len(common_idx),
            affected_groups=affected_groups,
            recommendations=recommendations,
            evidence=evidence,
        )

    except Exception:
        return None


def _encode_for_correlation(series: pd.Series) -> Optional[np.ndarray]:
    """Encode series for correlation computation."""
    try:
        # pd.api.types, not np.issubdtype: pandas 3 string columns carry an
        # Arrow-backed extension dtype that np.issubdtype cannot interpret
        # (TypeError), which the except swallowed -- silently skipping every
        # string protected attribute and returning an empty proxy report.
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return pd.Categorical(series).codes.astype(float)
    except Exception:
        return None


def _compute_all_correlations(
    feature_encoded: np.ndarray,
    attr_encoded: np.ndarray,
    feature_original: pd.Series,
    attr_original: pd.Series,
    feature_is_numeric: bool,
    attr_is_numeric: bool,
    include_mutual_information: bool,
) -> Dict[str, float]:
    """Compute multiple correlation measures."""
    results = {}

    try:
        if feature_is_numeric or True:  # Always try Pearson on encoded data
            mask = ~(np.isnan(feature_encoded) | np.isnan(attr_encoded))
            if mask.sum() >= 10:
                # The mask drops NaN, not inf, so an inf cell reaches pearsonr.
                # Recent scipy answers (nan, nan) for that; scipy before 1.11
                # RAISES ValueError ("array must not contain infs or NaNs") from
                # its finite-checked norm. Measured at the declared scipy floor:
                # the raise escaped to the outer except below and took Cramer's V
                # and the eta block with it, so ONE inf cell erased a CRITICAL
                # race proxy (test_readiness6_names). Pearson is undefined on
                # that input either way; record it as nan and keep screening.
                try:
                    corr, pvalue = stats.pearsonr(feature_encoded[mask], attr_encoded[mask])
                except ValueError:
                    corr, pvalue = float("nan"), float("nan")
                results["pearson"] = float(corr)
                results["pvalue"] = float(pvalue)

        if mask.sum() >= 10:
            spearman_corr, _ = stats.spearmanr(feature_encoded[mask], attr_encoded[mask])
            results["spearman"] = float(spearman_corr)

        # Cramér's V: bias-corrected (Bergsma 2013)
        # The naïve formula V = sqrt(χ²/(n·min(r,c)-1)) is inflated for
        # sparse contingency tables.  The corrected version subtracts the
        # expected inflation under H₀ and clamps to zero.
        try:
            contingency = pd.crosstab(
                pd.Categorical(feature_original), pd.Categorical(attr_original)
            )
            if contingency.size > 0:
                chi2, chi2_pval, dof, _ = stats.chi2_contingency(contingency)
                n = contingency.sum().sum()
                r, k = contingency.shape
                min_dim = min(r, k) - 1
                if min_dim > 0 and n > 0:
                    # Bias-corrected ϕ² (Bergsma 2013)
                    phi2 = chi2 / n
                    phi2_corr = float(np.maximum(0.0, phi2 - ((r - 1) * (k - 1)) / (n - 1)))
                    r_corr = r - ((r - 1) ** 2) / (n - 1)
                    k_corr = k - ((k - 1) ** 2) / (n - 1)
                    min_corr = min(r_corr, k_corr) - 1
                    undefined_because = ""
                    if min_corr > 0:
                        cramers_v = np.sqrt(phi2_corr / min_corr)
                    else:
                        # THE THIRD COPY OF THIS DEFECT. `else: cramers_v = 0.0`
                        # published NO ASSOCIATION for a table the corrected
                        # statistic is not defined on: min_corr <= 0 means the
                        # bias correction consumed the whole dimension, which
                        # happens on a sparse or degenerate contingency table,
                        # and it is exactly when a small sample makes the
                        # question unanswerable. 0.0 there says the feature is
                        # clean. The guarded version in
                        # feature_engineering/correlation.py:1322 already
                        # refuses; this copy had drifted from it.
                        cramers_v = float("nan")
                        undefined_because = (
                            f"proxy screen: Cramer's V for this pair is not defined "
                            f"(bias-corrected min dimension {min_corr:.3g} <= 0 on a "
                            f"{r}x{k} table of {n} rows), so the association was NOT "
                            f"measured. This is a could-not-check, not a finding that "
                            f"the feature is clean."
                        )
                    results["cramers_v"] = float(cramers_v)
                    # Store chi-squared p-value for the contingency test.
                    # This is the correct p-value for Cramér's V and
                    # should take priority over the Pearson p-value
                    # when Cramér's V is the primary correlation measure.
                    results["chi2_pvalue"] = float(chi2_pval)
                    if "pvalue" not in results:
                        results["pvalue"] = float(chi2_pval)
                    # The warning goes out AFTER the keys are written, and the
                    # order is load-bearing. Under `-W error::UserWarning` the
                    # warn RAISES, the enclosing `except Exception` swallowed
                    # it, and `cramers_v` and `chi2_pvalue` then vanished from
                    # the dict entirely: measured 2026-09-17, the dict came
                    # back as ['pearson', 'pvalue', 'spearman']. A disappearing
                    # key is not a third state a caller can read. The twin
                    # `_cramers_v_with_pvalue` hands back (nan, nan) whatever
                    # the warning filter says, and now so does this one.
                    if undefined_because:
                        warnings.warn(undefined_because, UserWarning, stacklevel=2)
        except Exception:
            logging.getLogger(__name__).debug(
                "optional computation failed; skipping", exc_info=True
            )

        # Correlation ratio (eta) for the mixed continuous-vs-categorical
        # case. Cramér's V above needs BOTH sides categorical; a continuous
        # feature treated as categorical drives the bias-corrected V to 0 and
        # hides a continuous proxy of a categorical protected attribute
        # (Datta et al. 2017; Barocas & Selbst 2016). eta is the right
        # measure and _analyze_proxy_relationship uses it for this case.
        if feature_is_numeric != attr_is_numeric:
            try:
                if feature_is_numeric:
                    eta, eta_pval = _correlation_ratio_with_pvalue(feature_original, attr_original)
                else:
                    eta, eta_pval = _correlation_ratio_with_pvalue(attr_original, feature_original)
                if np.isfinite(eta):
                    results["correlation_ratio"] = float(eta)
                    # Only when the ANOVA actually produced a p-value; a NaN
                    # here must not be published as though it were measured.
                    if np.isfinite(eta_pval):
                        results["eta_pvalue"] = float(eta_pval)
            except Exception:
                logging.getLogger(__name__).debug(
                    "optional computation failed; skipping", exc_info=True
                )

        if include_mutual_information:
            try:
                mi = _compute_mutual_information(feature_encoded[mask], attr_encoded[mask])
                results["mutual_information"] = float(mi)
            except Exception:
                logging.getLogger(__name__).debug(
                    "optional computation failed; skipping", exc_info=True
                )

    except Exception:
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    return results


def _compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two arrays."""
    n_bins = min(10, len(np.unique(x)), len(np.unique(y)))
    if n_bins < 2:
        return 0.0

    x_binned = pd.cut(x, bins=n_bins, labels=False, duplicates="drop")
    y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates="drop")

    # Handle NaN from binning
    mask = ~(pd.isna(x_binned) | pd.isna(y_binned))
    if mask.sum() < 10:
        return 0.0

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

    return max(0, mi)


def _correlation_ratio_with_pvalue(values: pd.Series, categories: pd.Series) -> Tuple[float, float]:
    """Correlation ratio eta in [0,1] for a CONTINUOUS variable across the
    levels of a CATEGORICAL grouping, with a one-way ANOVA F-test p-value.

    This is the correct association measure for the mixed
    continuous-vs-categorical case. Cramér's V requires BOTH sides to be
    categorical; a continuous feature treated as categorical makes every
    distinct value its own level, so the Bergsma (2013) bias correction
    drives V to exactly 0 and a continuous proxy of a categorical protected
    attribute (income / credit-score proxying for race, the textbook proxy
    case) is silently missed (Datta et al. 2017; Barocas & Selbst 2016).
    eta (Fisher/Pearson correlation ratio) with a one-way ANOVA p-value
    recovers it. Returns (nan, nan) when it cannot be computed.

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

    ONE ROW PER LEVEL IS NOT A MEASUREMENT (BGL-final d02, 2026-09-17). eta is
    the share of the continuous variable's variance that lies BETWEEN the
    levels, so it needs variation WITHIN a level to have anything to compare
    against. When every observation is its own group, ``ss_between`` equals
    ``ss_total`` identically and eta is sqrt(1) = 1.0 for ANY numbers at all.

    THE GUARD WAS APPLIED TO ONE OF TWO COPIES. The twin of this function in
    ``feature_engineering.correlation`` got it on 2026-09-16; this copy did
    not, and it is the one the proxy DETECTOR calls. Measured at the public
    entry before this change, 120 rows, ``race`` holding one distinct label per
    row and ``age`` drawn from ``default_rng(0).normal(45, 12, 120)``::

        identify_proxy_variables(df, ["race"], min_sample_size=100)
          -> correlation=1.0, risk_level='critical', pvalue=None,
             "CRITICAL: 'age' is a strong proxy for 'race' (correlation: 1.00).
              Removing this feature is strongly recommended."

    A perfect CRITICAL proxy, manufactured out of the PARTITION rather than the
    data: the same 1.00 comes back for any age column at all. The caller
    already treats a non-finite correlation as NOT SCREENED and says so in a
    warning, so the refusal reaches a reader instead of being silently dropped.
    """
    try:
        cat_name = getattr(categories, "name", None)
        vals = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
        cats = pd.Series(np.asarray(categories)).reset_index(drop=True)
        # np.isfinite, not ~isna: inf is NOT NaN and survived the old mask.
        mask = np.isfinite(vals.to_numpy(dtype=float))
        vals = vals[mask]
        cats = cats[mask]
        if len(vals) < 10:
            return float("nan"), float("nan")

        grand_mean = vals.mean()
        ss_total = float(((vals - grand_mean) ** 2).sum())
        if ss_total <= 0:
            # ZERO VARIANCE IS NOT "NO ASSOCIATION" (2026-09-25). Fixed here and in
            # the second copy of this function in
            # preprocessing/feature_engineering/correlation.py, because the two
            # disagreed: the same undefined quantity returned nan there (after the
            # fix) and a confident 0.0 / 1.0 here, and a caller cannot be asked to
            # know which module answered.
            #
            # eta is ss_between / ss_total, so with ss_total == 0 it is 0/0, and the
            # 1.0 is a p-value for an ANOVA that cannot run: scipy's f_oneway returns
            # nan for all-identical input, as pearsonr does for a constant array.
            # A constant column cannot be a proxy, so nothing downstream changes
            # (nan fails every >= threshold gate exactly as 0.0 did); what changes is
            # that the matrix stops asserting a measurement nobody took.
            #
            # A GENUINELY MEASURED ZERO IS UNAFFECTED, which is the distinction the
            # over-correction control in tests/test_readiness6_names.py exists to
            # protect: when the continuous variable HAS variance and the group means
            # happen to coincide, ss_total > 0, this branch is not taken, and eta is
            # a real, measured 0.0 with p = 1.0.
            warnings.warn(
                "correlation ratio (eta) COULD NOT BE MEASURED: the continuous "
                "variable has zero variance, so eta is 0/0 and undefined. Returning "
                "nan, not a correlation of 0.0. A constant column carries no "
                "information and cannot be a proxy, but that is a property of the "
                "column, not a measured association.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        groups = [g.to_numpy() for _, g in vals.groupby(cats.to_numpy())]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return float("nan"), float("nan")

        # eta is not identified without within-level variation. Two ways it
        # can be missing, both of which make the result an artefact of the
        # PARTITION rather than of the data:
        #   (a) no level holds two observations, so ss_between == ss_total
        #       identically and eta is exactly 1.0 whatever the numbers are;
        #   (b) the levels outnumber the pairs, so the between-level degrees
        #       of freedom dominate the residual ones and eta is inflated
        #       towards 1.0 by the same mechanism, just not all the way.
        # Refuse both. nan fails every downstream `>= threshold` gate and is
        # caught by the caller's explicit `np.isfinite(correlation)` check,
        # which warns and records the pair as NOT SCREENED.
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
                f"or bucket it before reading this screen.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
        # np.maximum, NOT the builtin max: max(0.0, nan) is 0.0, which turns
        # an uncomputable eta into a confident, clean-looking zero.
        eta = float(np.sqrt(np.maximum(0.0, ss_between) / ss_total))
        if not np.isfinite(eta):
            return float("nan"), float("nan")

        # One-way ANOVA F-test: is the between-group variation significant?
        # This is the correct significance test for eta and replaces the
        # degenerate chi-square p-value of a continuous-as-categorical table
        # (which read ~0.48, non-significant, and dropped real proxies).
        # An ANOVA that cannot be computed returns NaN: a forced 1.0 is a
        # measurement claim ("definitely not significant") the data does not
        # support, and it silently deleted findings at the caller's gate.
        try:
            _, pval = stats.f_oneway(*groups)
            pval = float(pval)
            if not np.isfinite(pval):
                pval = float("nan")
        except Exception:
            pval = float("nan")

        return eta, pval
    except Exception:
        return float("nan"), float("nan")


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
    """Determine risk level based on correlation and patterns."""
    if check_patterns:
        pattern_risk = _check_known_patterns(feature, protected_attr)
        if pattern_risk == "high" and correlation >= 0.3:
            return ProxyRiskLevel.CRITICAL

    if correlation >= 0.7:
        return ProxyRiskLevel.CRITICAL
    elif correlation >= 0.5:
        return ProxyRiskLevel.HIGH
    elif correlation >= 0.3:
        return ProxyRiskLevel.MEDIUM
    elif correlation >= 0.1:
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


def _determine_proxy_type(feature: str, protected_attr: str) -> ProxyType:
    """Determine the type of the proxy relationship between two named columns.

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

    Returns:
        DIRECT when the feature name encodes the protected attribute itself,
        or matches a known direct-encoding pattern FOR THAT ATTRIBUTE;
        HISTORICAL when it matches a known pattern for that attribute rooted
        in historical discrimination; otherwise UNCLASSIFIED. UNCLASSIFIED is
        a "could not determine", not a low-risk verdict: the correlation and
        the risk level are reported independently of it.
    """
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

    historical_keywords = [
        "zip",
        "zipcode",
        "neighborhood",
        "census",
        "redline",
        "school",
        "college",
        "credit_score",
    ]
    for kw in historical_keywords:
        if kw in feature_lower:
            return ProxyType.HISTORICAL

    direct_keywords = ["name", "surname", "title", "salutation"]
    for kw in direct_keywords:
        if kw in feature_lower:
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
            f"(correlation: {correlation:.2f}). Removing this feature is strongly recommended."
        )
        recommendations.append(
            "If the feature is necessary, apply strict fairness constraints or "
            "consider fair representation learning."
        )

    elif risk_level == ProxyRiskLevel.HIGH:
        recommendations.append(
            f"HIGH RISK: '{feature}' shows high correlation with '{protected_attr}'. "
            "Evaluate whether this feature is essential for the model."
        )
        recommendations.append("Consider fairness-aware feature selection or regularization.")

    elif risk_level == ProxyRiskLevel.MEDIUM:
        recommendations.append(
            f"MODERATE RISK: '{feature}' has moderate correlation with '{protected_attr}'. "
            "Monitor for disparate impact in model predictions."
        )

    elif risk_level == ProxyRiskLevel.LOW:
        recommendations.append(
            f"LOW RISK: '{feature}' shows weak correlation. "
            "Continue monitoring but lower priority for intervention."
        )

    return recommendations


def compute_proxy_correlations(
    df: pd.DataFrame,
    feature: str,
    protected_attribute: str,
) -> Dict[str, Any]:
    """
    Compute all correlation measures between a feature and protected attribute.

    Convenience function for detailed proxy analysis of a single pair.

    Args:
        df: DataFrame to analyze
        feature: Potential proxy feature
        protected_attribute: Protected attribute

    Returns:
        Dictionary with all correlation measures and metadata
    """
    if feature not in df.columns or protected_attribute not in df.columns:
        return {"error": "Column not found"}

    feature_series = df[feature].dropna()
    attr_series = df[protected_attribute].dropna()

    common_idx = feature_series.index.intersection(attr_series.index)
    if len(common_idx) < 10:
        return {"error": "Insufficient overlapping samples"}

    feature_aligned = feature_series[common_idx]
    attr_aligned = attr_series[common_idx]

    feature_encoded = _encode_for_correlation(feature_aligned)
    attr_encoded = _encode_for_correlation(attr_aligned)

    if feature_encoded is None or attr_encoded is None:
        return {"error": "Could not encode variables"}

    try:
        feature_is_numeric = pd.api.types.is_numeric_dtype(feature_aligned)
    except (TypeError, AttributeError):
        feature_is_numeric = False

    try:
        attr_is_numeric = pd.api.types.is_numeric_dtype(attr_aligned)
    except (TypeError, AttributeError):
        attr_is_numeric = False

    correlations = _compute_all_correlations(
        feature_encoded,
        attr_encoded,
        feature_aligned,
        attr_aligned,
        feature_is_numeric,
        attr_is_numeric,
        include_mutual_information=True,
    )

    # NaN, not 0: a statistic that is ABSENT from this dict was never measured,
    # and 0.0 is the value of "no association at all". `_compute_all_correlations`
    # writes `correlation_ratio` only `if np.isfinite(eta)`, and it writes
    # `pearson` even when scipy returned NaN on a constant input, so BOTH the
    # missing key and the present-but-NaN key have to be caught: `.get(key,
    # default)` does not fire for the second. `_analyze_proxy_relationship`
    # already refuses this case behind its own np.isfinite guard; this
    # convenience wrapper was the sibling that still carried the defect.
    # Measured 2026-09-17, one categorical protected attribute with a single
    # level and a continuous feature: eta was never computed at all, and this
    # function returned {'primary_correlation': 0, 'risk_level': 'negligible'},
    # a confident clean bill for a pair nothing was measured about.
    if feature_is_numeric and attr_is_numeric:
        primary = float(correlations.get("pearson", float("nan")))
        primary_type = "Pearson"
    elif feature_is_numeric != attr_is_numeric:
        # Continuous vs categorical: correlation ratio (eta), not Cramér's V.
        primary = float(correlations.get("correlation_ratio", float("nan")))
        primary_type = "Correlation ratio (eta)"
    else:
        primary = float(correlations.get("cramers_v", float("nan")))
        primary_type = "Cramér's V"

    measured = bool(np.isfinite(primary))
    if not measured:
        warnings.warn(
            f"proxy correlation for '{feature}' vs '{protected_attribute}': the "
            f"{primary_type} association could not be computed on this data, so "
            f"the pair is NOT SCREENED. This is a could not check, not a finding "
            f"that the feature is clean.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "feature": feature,
        "protected_attribute": protected_attribute,
        "sample_size": len(common_idx),
        "primary_correlation": primary,
        "primary_correlation_type": primary_type,
        # Three states on the surface a reader actually reads, so the caller
        # never has to infer "unmeasurable" from a number that looks measured.
        "measurement_status": "measured" if measured else "could_not_measure",
        "all_correlations": correlations,
        "risk_level": (
            _determine_risk_level(abs(primary), feature, protected_attribute, True).value
            if measured
            else None
        ),
        "is_known_pattern": _check_known_patterns(feature, protected_attribute) is not None,
    }


class _ProxyChainScanResult(List[Dict[str, Any]]):
    """The chains this scan found, PLUS the pairs it could not look at.

    Same shape, and the same reason, as ``ProxyChainResult`` in
    ``feature_engineering/correlation.py``: a bare ``[]`` was one token for
    "looked and found nothing" and for "could not look", and only one of those
    is reassuring. Measured 2026-09-17 on 200 rows where a unique-per-row
    ``customer_ref`` determined ``gender`` perfectly: every link association
    came back NaN, ``corr >= correlation_threshold`` is False for a NaN, and
    this function returned ``[]`` with nothing recorded anywhere.

    It subclasses ``list`` on purpose, so every existing consumer (``for ch in
    ...``, ``len``, ``isinstance(x, list)``, ``... or []``) keeps working
    unchanged, while a caller that wants the coverage can read it.

    Attributes:
        pairs_not_computed: ``(col_a, col_b, reason)`` for every pair whose
            association could NOT be measured. Not pairs that were measured
            and came in under the threshold: those are a finding.
        protected_attribute_present: False when the named attribute is not a
            column of the frame, in which case nothing was examined at all.
    """

    pairs_not_computed: List[Tuple[str, str, str]]
    protected_attribute_present: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pairs_not_computed = []
        self.protected_attribute_present = True


#: Aligned non-null rows a link needs before its association is measured.
CHAIN_MIN_SAMPLES = 30


def find_proxy_chains(
    df: pd.DataFrame,
    protected_attribute: str,
    *,
    max_chain_length: int = 2,
    correlation_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Find chains of proxy relationships (A -> B -> Protected).

    Identifies indirect proxies where a feature is correlated with another
    feature that is itself a proxy for the protected attribute.

    Args:
        df: DataFrame to analyze
        protected_attribute: Protected attribute to trace proxies for
        max_chain_length: Maximum length of proxy chains to find
        correlation_threshold: Minimum correlation for chain links

    Returns:
        A ``_ProxyChainScanResult``: still a list of proxy chain dictionaries
        for every existing caller, carrying ``pairs_not_computed`` and
        ``protected_attribute_present`` so an empty result can be told apart
        from a scan that could not look. A link whose association is NaN is
        recorded there and never compared with the threshold: ``nan >= 0.3``
        is False, which is how a could-not-check turns itself back into a
        clean bill of health.
    """
    chains: List[Dict[str, Any]] = []
    not_computed: List[Tuple[str, str, str]] = []

    if protected_attribute not in df.columns:
        warnings.warn(
            f"find_proxy_chains: protected attribute {protected_attribute!r} is not a "
            f"column of this frame, so NO chain was examined. The empty result is a "
            f"could not check, not a finding that no proxy chain exists.",
            UserWarning,
            stacklevel=2,
        )
        empty = _ProxyChainScanResult()
        empty.protected_attribute_present = False
        return empty

    # Type-aware association (Cramer's V / eta / |Pearson| by dtype). The
    # legacy Pearson-on-Categorical.codes was invalid for unordered
    # categories and silently missed high-cardinality proxy chains
    # (e.g. zip -> school -> race). One source of truth with the
    # univariate proxy screen.
    from vfairness.evaluation.vfairness_metrics.discovery import association_strength

    direct_proxies = {}
    prot = df[protected_attribute]

    for col in df.columns:
        if col == protected_attribute:
            continue
        try:
            pair = pd.DataFrame({"c": df[col], "p": prot}).dropna()
            if len(pair) < CHAIN_MIN_SAMPLES:
                not_computed.append(
                    (
                        str(col),
                        str(protected_attribute),
                        f"only {len(pair)} aligned non-null row(s), the link needs "
                        f"{CHAIN_MIN_SAMPLES}",
                    )
                )
                continue
            corr = association_strength(pair["c"], pair["p"])
            if corr is None or not np.isfinite(corr):
                # NaN is association_strength saying it could not measure this
                # pair. Comparing it with the threshold answers False and the
                # column silently leaves the scan, so it is recorded instead.
                not_computed.append(
                    (
                        str(col),
                        str(protected_attribute),
                        "the association is not defined on this data (see the warning "
                        "association_strength raised for the pair)",
                    )
                )
                continue
            if corr >= correlation_threshold:
                direct_proxies[col] = corr
        except Exception as exc:
            not_computed.append(
                (str(col), str(protected_attribute), f"{type(exc).__name__}: {exc}")
            )
            continue

    if max_chain_length >= 2:
        for proxy, proxy_corr in direct_proxies.items():
            for col in df.columns:
                if col in [protected_attribute, proxy]:
                    continue
                if col in direct_proxies:
                    continue  # Already a direct proxy

                try:
                    pair = pd.DataFrame({"c": df[col], "p": df[proxy]}).dropna()
                    if len(pair) < CHAIN_MIN_SAMPLES:
                        not_computed.append(
                            (
                                str(col),
                                str(proxy),
                                f"only {len(pair)} aligned non-null row(s), the link "
                                f"needs {CHAIN_MIN_SAMPLES}",
                            )
                        )
                        continue
                    corr = association_strength(pair["c"], pair["p"])
                    if corr is None or not np.isfinite(corr):
                        not_computed.append(
                            (
                                str(col),
                                str(proxy),
                                "the association is not defined on this data (see the "
                                "warning association_strength raised for the pair)",
                            )
                        )
                        continue

                    if corr >= correlation_threshold:
                        chains.append(
                            {
                                "chain": [col, proxy, protected_attribute],
                                "correlations": [corr, proxy_corr],
                                "indirect_correlation": corr * proxy_corr,
                                "chain_length": 2,
                            }
                        )

                except Exception as exc:
                    not_computed.append((str(col), str(proxy), f"{type(exc).__name__}: {exc}"))
                    continue

    chains.sort(key=lambda x: x["indirect_correlation"], reverse=True)

    if not_computed:
        _detail = "; ".join(f"{a} ~ {b}: {why}" for a, b, why in not_computed[:6])
        _more = f" (and {len(not_computed) - 6} more)" if len(not_computed) > 6 else ""
        warnings.warn(
            f"find_proxy_chains: {len(not_computed)} column pair(s) COULD NOT BE "
            f"MEASURED, so this search did not cover the whole frame: {_detail}"
            f"{_more}. A chain through one of those pairs would not appear below, and "
            f"an empty or short result is not evidence that none exists. See "
            f"result.pairs_not_computed.",
            UserWarning,
            stacklevel=2,
        )

    result = _ProxyChainScanResult(chains[:20])
    result.pairs_not_computed = not_computed
    return result


# Multivariate / systemic proxy leakage.
#
# Univariate proxy detection asks "does feature X correlate with attribute A?".
# It misses *collective* leakage: features that are each only mildly
# correlated with A but together reconstruct it (e.g. zip + university tier +
# photo score jointly recovering race). The canonical test (Barocas & Selbst
# 2016; Datta et al. 2017) is: train a model on ALL non-protected features to
# predict A and measure held-out AUC. AUC well above chance means the protected
# attribute is *redundantly encoded* -- removing individual proxies will not
# protect the model. Shared by Pulse, the Navigator proxy handler, and the
# Bias-Detection module so all three report identical leakage figures.


#: Severity of a leakage test that never ran. It is not a band on the AUC
#: scale, it is the absence of one, and it is deliberately NOT a word any
#: threshold comparison produces. BGL-S2 2026-09-16: an empty frame, a 2-row
#: frame and a single-group 200-row frame all returned the identical graded
#: record ``auc=0.5, macro_auc=0.5, worst_group_auc=0.5, severity="negligible",
#: systemic_leakage=False`` with zero warnings, indistinguishable at every
#: machine-read field from the genuine no-leak control (auc 0.5080,
#: "negligible", False). Two prose fields did disclose it, and no scorecard
#: reads prose.
LEAKAGE_NOT_ASSESSED = "not_assessed"


@dataclass
class MultivariateProxyResult:
    """Systemic leakage of one protected attribute via all other features.

    Every graded field is Optional and ``None`` is the third state: the test
    did not run, so there is no AUC, no band and no leakage verdict. ``None``
    is never a measurement of no leakage; ``severity`` carries
    :data:`LEAKAGE_NOT_ASSESSED` in that case and ``method`` says which of the
    three ways the test failed to run.
    """

    protected_attribute: str
    # None when no test ran. Never 0.5: chance-level AUC is what a measured
    # no-leak attribute scores, and that is the one thing this must not be
    # confused with.
    auc: Optional[float]  # held-out CV AUC (macro OVR if multiclass)
    chance_auc: float  # 0.5 baseline reference
    severity: str  # negligible|low|medium|high|severe|not_assessed
    # None when no test ran. NOT False: False is the positive statement "this
    # attribute was tested and is not systemically leaked".
    systemic_leakage: Optional[bool]  # auc >= 0.70 (spec threshold)
    n_features_used: int
    n_samples: int
    n_classes: int
    method: str
    top_contributors: List[Dict[str, Any]]  # [{feature, importance}]
    interpretation: str
    # Default None, not 0.5. These two used to default to chance and the three
    # fallback branches never overwrote them, so to_dict() published a
    # chance-level AUC on both for a test that never ran.
    macro_auc: Optional[float] = None  # avg OVR AUC over all groups
    worst_group_auc: Optional[float] = None  # most-identifiable single group

    def was_assessed(self) -> bool:
        """Whether a leakage test actually ran for this attribute.

        The one call a consumer needs before reading ``auc`` or
        ``systemic_leakage`` as a measurement.

        BGL grade-1 G08, 2026-09-30. ``self.auc is not None`` re-derived "was
        this measured?" from one door out of six, and NaN is not that door.
        Measured on this repo before the change, on a record carrying
        ``auc=nan, severity='low', systemic_leakage=False``:
        ``was_assessed()`` returned True, so the one call the docstring tells a
        consumer to make before reading ``auc`` said the leakage test HAD run,
        and every gate downstream then read ``nan >= 0.70`` as False, i.e. as a
        clean bill of health. ``inf`` and ``True`` passed it as well
        (``float(True) == 1.0``, which is a perfect reconstruction score).
        ``is_measured`` is the library's canonical rule for the same question
        and refuses all three, so the predicate and the arithmetic below it now
        agree about what counts as a measurement.
        """
        return is_measured(self.auc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protected_attribute": self.protected_attribute,
            "auc": self.auc,
            "macro_auc": self.macro_auc,
            "worst_group_auc": self.worst_group_auc,
            "chance_auc": self.chance_auc,
            "severity": self.severity,
            "systemic_leakage": self.systemic_leakage,
            "n_features_used": self.n_features_used,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "method": self.method,
            "top_contributors": self.top_contributors,
            "interpretation": self.interpretation,
        }


def _leakage_severity(auc: float) -> str:
    """AUC -> severity. Thresholds per the Pulse spec (0.65/0.75/0.85)."""
    if auc >= 0.85:
        return "severe"
    if auc >= 0.75:
        return "high"
    if auc >= 0.65:
        return "medium"
    if auc >= 0.55:
        return "low"
    return "negligible"


def multivariate_proxy_leakage(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    max_rows: int = 5000,
    max_cardinality: int = 50,
    cv: int = 5,
    random_state: int = 0,
) -> List[MultivariateProxyResult]:
    """Systemic-leakage test: can each protected attribute be reconstructed
    from all the *non-protected* features together?

    For every attribute A in ``protected_attributes`` a gradient-boosted
    classifier is cross-validated to predict A from all other (non-protected,
    non-excluded) columns; the held-out ROC AUC quantifies redundant
    encoding. AUC >= 0.70 means dropping individual proxy columns will not
    de-bias the model -- the platform's strongest argument for transforming,
    not just removing, features.

    Args:
        df: dataset
        protected_attributes: columns to try to reconstruct
        feature_columns: predictors (default: all columns except protected /
            excluded)
        exclude_columns: columns never used as predictors (IDs, the target,
            the model decision, oracle labels -- pass these from the column
            typology so leakage is measured on legitimate features only)
        max_rows: row cap for speed (stratified head sample)
        max_cardinality: skip predictor columns with more uniques than this
            (free-text / IDs) to avoid one-hot explosion
        cv: CV folds
        random_state: determinism

    Returns:
        List[MultivariateProxyResult], one per protected attribute, sorted by
        AUC descending. Never raises -- an un-modellable attribute yields a
        negligible result with an explanatory ``interpretation``.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: multivariate_proxy_leakage. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import LabelEncoder
    except Exception:  # noqa: BLE001 -- sklearn missing: degrade, never crash
        # Degrading is right; grading is not. Every field a scorecard reads is
        # None and the severity is the not-assessed word, so this record cannot
        # be mistaken for the measured no-leak result it used to be identical
        # to.
        if protected_attributes:
            warnings.warn(
                "multivariate_proxy_leakage: scikit-learn is unavailable, so no "
                "leakage test ran for "
                f"{', '.join(str(a) for a in protected_attributes)}. Those attributes "
                "are NOT assessed; this is not a finding of no leakage.",
                UserWarning,
                stacklevel=2,
            )
        return [
            MultivariateProxyResult(
                protected_attribute=a,
                auc=None,
                chance_auc=0.5,
                severity=LEAKAGE_NOT_ASSESSED,
                systemic_leakage=None,
                n_features_used=0,
                n_samples=0,
                n_classes=0,
                method="unavailable",
                top_contributors=[],
                interpretation=(
                    "scikit-learn unavailable; leakage test skipped. This attribute is "
                    "NOT assessed for systemic leakage."
                ),
            )
            for a in protected_attributes
        ]

    protected_set = {c for c in protected_attributes}
    excluded = set(exclude_columns or []) | protected_set
    results: List[MultivariateProxyResult] = []

    work = df
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=random_state)

    for attr in protected_attributes:
        if attr not in df.columns:
            continue
        try:
            preds = feature_columns or [c for c in work.columns if c not in excluded and c != attr]
            # Drop free-text / id-like columns (cardinality guard).
            preds = [
                c
                for c in preds
                if work[c].nunique(dropna=True) <= max_cardinality
                or pd.api.types.is_numeric_dtype(work[c])
            ]
            sub = work[preds + [attr]].dropna(subset=[attr])
            # Continuous / high-cardinality targets (age, date_of_birth, income)
            # must NOT be label-encoded value-by-value: that turns every
            # distinct value into its own class, yielding hundreds of
            # singleton classes and a degenerate multiclass ROC AUC that
            # collapses toward 0.00 and is meaningless. Bin numeric targets
            # with many distinct values into quantile buckets so
            # "reconstruction" is a well-posed few-class problem.
            attr_col = sub[attr]
            if pd.api.types.is_numeric_dtype(attr_col) and attr_col.nunique(dropna=True) > 20:
                binned = pd.qcut(attr_col, q=5, duplicates="drop")
                if binned.nunique(dropna=True) < 2:
                    binned = pd.cut(attr_col, bins=5, duplicates="drop")
                y_raw = binned.astype("string").fillna("missing")
            else:
                y_raw = attr_col.astype("string").fillna("missing")
            counts = y_raw.value_counts()
            # Need >=2 classes each with enough rows for CV.
            keep = counts[counts >= max(cv, 10)].index
            sub = sub[y_raw.isin(keep)]
            y_raw = y_raw[y_raw.isin(keep)]
            if y_raw.nunique() < 2 or len(sub) < max(50, cv * 10):
                warnings.warn(
                    f"multivariate_proxy_leakage: not enough labelled rows per class to "
                    f"test reconstruction of '{attr}' ({int(len(sub))} usable row(s), "
                    f"{int(y_raw.nunique())} class(es)). It is UNASSESSED for systemic "
                    f"leakage, not clean.",
                    UserWarning,
                    stacklevel=2,
                )
                results.append(
                    MultivariateProxyResult(
                        protected_attribute=attr,
                        auc=None,
                        chance_auc=0.5,
                        severity=LEAKAGE_NOT_ASSESSED,
                        systemic_leakage=None,
                        n_features_used=len(preds),
                        n_samples=int(len(sub)),
                        n_classes=int(y_raw.nunique()),
                        method="insufficient_data",
                        top_contributors=[],
                        interpretation=(
                            "Not enough labelled rows per class to reliably test "
                            "reconstruction of this attribute. It is NOT assessed for "
                            "systemic leakage, which is not the same as not leaking."
                        ),
                    )
                )
                continue

            # One-hot encode categorical predictors (bounded cardinality) so
            # the tree can split on individual proxy CATEGORIES (e.g. an HBCU
            # university_tier, a minority-majority zip). Ordinal label-encoding
            # injects a fake order and destroys nominal-proxy signal -- it
            # under-detected categorical race proxies.
            num_cols, cat_cols = [], []
            for c in preds:
                if pd.api.types.is_numeric_dtype(sub[c]):
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
            Xnum = (
                sub[num_cols].apply(pd.to_numeric, errors="coerce")
                if num_cols
                else pd.DataFrame(index=sub.index)
            )
            # dtype=float, not get_dummies' default bool. scikit-learn 1.4 (the
            # declared floor) reads a pandas frame through the dataframe
            # interchange protocol, and pandas 2.2 cannot export a bool column
            # through it ("Conversion of boolean to Arrow C format string is not
            # implemented"), so every attribute with a categorical feature came
            # back NOT ASSESSED at the floor. 0/1 floats are the same numbers.
            Xcat = (
                pd.get_dummies(
                    sub[cat_cols].astype("string").fillna("missing"),
                    dummy_na=False,
                    dtype=float,
                )
                if cat_cols
                else pd.DataFrame(index=sub.index)
            )
            X = pd.concat([Xnum, Xcat], axis=1)
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            y = LabelEncoder().fit_transform(y_raw)
            n_classes = int(len(np.unique(y)))

            n_splits = max(2, min(cv, int(np.bincount(y).min())))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=None, learning_rate=0.12, random_state=random_state
            )
            proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
            if n_classes == 2:
                macro_auc = float(roc_auc_score(y, proba[:, 1]))
                macro_auc = max(macro_auc, 1.0 - macro_auc)
                worst_auc = macro_auc
            else:
                # Macro OVR AUC, NOT reflected. Reflecting max(a,1-a) is
                # only defensible for a single binary task; a macro-average
                # over many OVR tasks near 0.5 is genuine chance, and the
                # per-class max-of-reflected statistic is a positively
                # biased order statistic (the max of K folded noisy AUCs
                # sits well above 0.5 even with no real leakage), which
                # systematically inflated "systemic leakage" for every
                # multi-class attribute.
                macro_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
                from sklearn.preprocessing import label_binarize

                Yb = label_binarize(y, classes=list(range(n_classes)))
                # Worst identifiable group: diagnostic only, raw (not
                # reflected), and ONLY for classes with enough support for
                # the per-class AUC to be stable.
                per = []
                for k in range(n_classes):
                    pos = int(Yb[:, k].sum())
                    if pos < 50 or (len(y) - pos) < 50:
                        continue
                    try:
                        per.append(float(roc_auc_score(Yb[:, k], proba[:, k])))
                    except Exception:  # noqa: BLE001
                        continue
                worst_auc = max(per) if per else macro_auc
            # Headline AUC = the macro OVR reconstructability. The
            # worst-group value is reported separately as a diagnostic and
            # no longer inflates the headline / systemic-leakage gate.
            auc = float(macro_auc)

            # Top contributors via mutual information (fast, model-free, and
            # HistGradientBoosting exposes no feature_importances_). One-hot
            # columns are mapped back to their source feature so the user
            # sees "university_tier", not "university_tier_HBCU".
            contributors: List[Dict[str, Any]] = []
            try:
                from sklearn.feature_selection import mutual_info_classif

                mi = mutual_info_classif(X, y, discrete_features=False, random_state=random_state)
                src_of = {}
                for col in X.columns:
                    src = next((c for c in cat_cols if str(col).startswith(c + "_")), str(col))
                    src_of[col] = src
                agg: Dict[str, float] = {}
                for col, m in zip(X.columns, mi):
                    agg[src_of[col]] = agg.get(src_of[col], 0.0) + float(m)
                contributors = [
                    {"feature": k, "importance": round(v, 4)}
                    for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:8]
                    if v > 0
                ]
            except Exception:  # noqa: BLE001
                contributors = []

            sev = _leakage_severity(auc)
            leak = auc >= 0.70
            grp_note = (
                f" The most identifiable single group reaches AUC {worst_auc:.2f}."
                if n_classes > 2 and worst_auc > macro_auc + 0.05
                else ""
            )
            # Plain-language reconstructability score on a 0-100 scale so a
            # non-specialist can read it without knowing what AUC is.
            recon_pct = int(round((auc - 0.5) / 0.5 * 100))
            recon_pct = max(0, min(100, recon_pct))
            interp = (
                f"The other columns can rebuild {attr} on their own: a "
                f"reconstructability score of {recon_pct} out of 100 "
                f"(0 = the data hides {attr} completely, 100 = {attr} can be "
                f"recovered exactly; AUC {auc:.2f}).{grp_note} "
                + (
                    "Because this is high, {0} is effectively still in the data "
                    "even if you delete the {0} column: removing it will NOT "
                    "de-bias the model. The features must be transformed or the "
                    "target re-derived.".format(attr)
                    if leak
                    else f"Because this is low, {attr} is not strongly hidden in the "
                    f"other columns; handling individual proxy columns is "
                    f"likely sufficient."
                )
            )
            results.append(
                MultivariateProxyResult(
                    protected_attribute=attr,
                    auc=auc,
                    chance_auc=0.5,
                    severity=sev,
                    systemic_leakage=leak,
                    n_features_used=len(preds),
                    n_samples=int(len(sub)),
                    n_classes=n_classes,
                    method="hist_gradient_boosting",
                    top_contributors=contributors,
                    interpretation=interp,
                    macro_auc=float(macro_auc),
                    worst_group_auc=float(worst_auc),
                )
            )
        except Exception as e:  # noqa: BLE001 -- never crash the audit
            warnings.warn(
                f"multivariate_proxy_leakage: the leakage test for '{attr}' raised "
                f"{type(e).__name__}: {e}. That attribute is UNASSESSED for systemic "
                f"leakage, not clean.",
                UserWarning,
                stacklevel=2,
            )
            results.append(
                MultivariateProxyResult(
                    protected_attribute=attr,
                    auc=None,
                    chance_auc=0.5,
                    severity=LEAKAGE_NOT_ASSESSED,
                    systemic_leakage=None,
                    n_features_used=0,
                    n_samples=0,
                    n_classes=0,
                    method="error",
                    top_contributors=[],
                    interpretation=(
                        f"Leakage test could not run: {e}. This attribute is NOT "
                        f"assessed for systemic leakage."
                    ),
                )
            )

    # Rank only what was measured. Sorting an unmeasured record by a
    # substituted AUC put it in the same ordered list as real results and let a
    # reader take its position as a standing; the not-assessed ones are
    # appended after the ranking, in the order they were requested.
    measured = [r for r in results if r.auc is not None]
    unmeasured = [r for r in results if r.auc is None]
    measured.sort(key=lambda r: float(r.auc or 0.0), reverse=True)
    return measured + unmeasured
