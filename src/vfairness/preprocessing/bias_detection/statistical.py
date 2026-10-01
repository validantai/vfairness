"""
C. Statistical Disparity Analysis

Provides rigorous statistical testing for detecting significant differences
between demographic groups in outcomes, features, and data quality.

Key capabilities:
- Hypothesis testing for group differences (t-test, one-way ANOVA,
  chi-squared, Fisher's exact). The test is SELECTED FROM THE DATA, never
  named by the caller. Mann-Whitney was listed here and is not implemented
  anywhere in this module: scipy.stats.mannwhitneyu is never called.
- Effect size estimation (Cohen's d, Cramér's V, odds ratios)
- Confidence interval computation
- Multiple comparison correction (Bonferroni, FDR)
- Intersectional subgroup analysis
- Data quality disparity detection (missing rates, error rates)

References:
    - Barocas, S., Hardt, M., & Narayanan, A. (2023). Fairness and ML
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    - Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from vfairness._names import tokens_contain
from vfairness._not_assessed import NOT_ASSESSED
from vfairness.evaluation.vfairness_metrics._validation import _check_option


class DisparityType(Enum):
    """Types of statistical disparities."""

    OUTCOME = "outcome"  # Differences in outcome rates
    DISTRIBUTION = "distribution"  # Differences in feature distributions
    QUALITY = "quality"  # Differences in data quality (missing, errors)
    INTERSECTIONAL = "intersectional"  # Intersectional disparities


class SignificanceLevel(Enum):
    """Significance levels for statistical tests."""

    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.001
    SIGNIFICANT = "significant"  # p < 0.01
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.05
    NOT_SIGNIFICANT = "not_significant"  # p >= 0.05
    # Three states, not two. A test that could not run (a constant column
    # makes ttest_ind return NaN) is NOT the same claim as "tested, no
    # disparity". It used to be graded NOT_SIGNIFICANT, because every
    # comparison against NaN is False and the final `else` caught it, and it
    # then took a full slot in the multiple-comparison family size.
    NOT_TESTABLE = "not_testable"  # the test could not be computed


class EffectSizeInterpretation(Enum):
    """Interpretation of effect sizes."""

    NEGLIGIBLE = "negligible"  # |d| < 0.2
    SMALL = "small"  # 0.2 <= |d| < 0.5
    MEDIUM = "medium"  # 0.5 <= |d| < 0.8
    LARGE = "large"  # |d| >= 0.8
    # Three states, not two, for the MAGNITUDE as well as for the test. A
    # standardised effect size needs a scale to standardise BY, and when every
    # group is constant there is none: Cohen's d divides by a pooled SD of
    # zero, eta-squared by a total sum of squares of zero. Both used to answer
    # 0.0, which is the band NEGLIGIBLE, so a total 0.90-vs-0.10 separation
    # was graded "negligible" and the recommendation downgraded to "may not
    # require immediate intervention" while disparity_magnitude=0.8 sat in the
    # same object unread. UNMEASURED is not SMALL.
    NOT_MEASURABLE = "not_measurable"  # no effect size could be computed


@dataclass
class StatisticalDisparityResult:
    """
    Result of statistical disparity analysis.

    Attributes:
        feature: Feature or outcome being analyzed
        protected_attribute: Protected attribute used for grouping
        disparity_type: Type of disparity detected
        test_name: Name of statistical test used
        test_statistic: Test statistic value
        pvalue: P-value from statistical test
        significance: Significance level interpretation
        effect_size: Estimated effect size
        effect_size_type: Type of effect size (Cohen's d, Cramér's V, etc.)
        effect_interpretation: Interpretation of effect size magnitude
        confidence_interval: Confidence interval for the effect
        group_statistics: Per-group summary statistics
        privileged_group: Group with more favorable outcome/value, or None when
            which outcome is favourable could not be determined
        disadvantaged_group: Group with less favorable outcome/value, or None for
            the same reason
        positive_class_used: Outcome label the positive_rate is a rate of
        direction_not_determined_reason: Why the two group fields are None
        disparity_magnitude: Magnitude of the disparity
        sample_sizes: Sample sizes per group
        recommendations: Suggested actions
    """

    feature: str
    protected_attribute: str
    disparity_type: DisparityType
    test_name: str
    test_statistic: float
    pvalue: float
    significance: SignificanceLevel
    effect_size: float
    effect_size_type: str
    effect_interpretation: EffectSizeInterpretation
    confidence_interval: Tuple[Optional[float], Optional[float]]
    group_statistics: Dict[str, Dict[str, float]]
    # OPTIONAL BECAUSE THE DIRECTION IS NOT ALWAYS DETERMINABLE. For a binary
    # outcome whose labels are in neither truthy whitelist, which of the two
    # labels is the FAVOURABLE one cannot be read off the data, and guessing it
    # does not fail closed: it INVERTS the finding and names the favoured group
    # as the victim. None here means could-not-check, and
    # direction_not_determined_reason says so in words.
    privileged_group: Optional[str]
    disadvantaged_group: Optional[str]
    disparity_magnitude: float
    sample_sizes: Dict[str, int]
    recommendations: List[str]
    #: The label the reported positive_rate is a rate OF, so a reader can settle
    #: the direction themselves even when this code will not assert it.
    positive_class_used: Optional[str] = None
    #: Why privileged_group and disadvantaged_group are None. Present only then.
    direction_not_determined_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "protected_attribute": self.protected_attribute,
            "disparity_type": self.disparity_type.value,
            "test_name": self.test_name,
            "test_statistic": self.test_statistic,
            "pvalue": self.pvalue,
            "significance": self.significance.value,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "effect_interpretation": self.effect_interpretation.value,
            "confidence_interval": self.confidence_interval,
            "group_statistics": self.group_statistics,
            "privileged_group": self.privileged_group,
            "disadvantaged_group": self.disadvantaged_group,
            "positive_class_used": self.positive_class_used,
            "direction_not_determined_reason": self.direction_not_determined_reason,
            "disparity_magnitude": self.disparity_magnitude,
            "sample_sizes": self.sample_sizes,
            "recommendations": self.recommendations,
        }


# Significance verdicts that are NOT a reportable disparity. NOT_TESTABLE
# belongs here for the opposite reason to NOT_SIGNIFICANT: it is not a clean
# result, it is no result at all, and it is disclosed by its own warning
# rather than published as a finding.
_NON_FINDING_SIGNIFICANCE = frozenset(
    {SignificanceLevel.NOT_SIGNIFICANT, SignificanceLevel.NOT_TESTABLE}
)

# Budgets for AUTO-DETECTED columns. These are deliberately kept -- running a
# test per column on a 500-column frame inflates the multiple-comparison
# family size, and Benjamini-Hochberg then HIDES real findings -- but a
# budget that is not disclosed is indistinguishable from a clean result.
# Measured 2026-09-10: with 35 innocuous columns whose names contain a "y"
# and the real outcome last in column order, analyze_statistical_disparities
# reported 0 disparities, while passing outcome_columns=['approved']
# explicitly found the same data's 85%-vs-15% approval gap at p=5.9e-60.
_MAX_AUTO_OUTCOME_COLUMNS = 10
_MAX_AUTO_FEATURE_COLUMNS = 20


def analyze_statistical_disparities(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_columns: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    *,
    significance_level: float = 0.05,
    min_group_size: int = 30,
    include_quality_analysis: bool = True,
    include_intersectional: bool = True,
    correction_method: str = "fdr_bh",
) -> List[StatisticalDisparityResult]:
    """
    Comprehensive statistical disparity analysis across protected attributes.

    Performs hypothesis tests to identify statistically significant differences
    between demographic groups in outcomes, features, and data quality.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute column names
        outcome_columns: Columns representing outcomes to test (auto-detected if None)
        feature_columns: Columns representing features to test (auto-detected if None)
        significance_level: Alpha level for hypothesis tests
        min_group_size: Minimum group size for reliable testing
        include_quality_analysis: Whether to analyze data quality disparities
        include_intersectional: Whether to include intersectional analysis
        correction_method: Multiple comparison correction ('bonferroni', 'fdr_bh', None)

    Returns:
        List of StatisticalDisparityResult objects for significant disparities

    Example:
        >>> results = analyze_statistical_disparities(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     outcome_columns=['approved', 'score'],
        ... )
        >>> for r in results:
        ...     print(f"{r.feature} by {r.protected_attribute}:")
        ...     print(f"  Effect: {r.effect_size:.3f} ({r.effect_interpretation.value})")
        ...     print(f"  P-value: {r.pvalue:.4f}")

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

    Ledger row: analyze_statistical_disparities. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    results = []
    not_tested: List[str] = []
    # Comparisons that lost a group to min_group_size. Collected rather than
    # dropped, because [] was doing double duty for "tested and clean" and
    # "never tested at all".
    undersized: List[str] = []

    if outcome_columns is None:
        detected = _detect_outcome_columns(df, protected_attributes)
        outcome_columns = detected[:_MAX_AUTO_OUTCOME_COLUMNS]
        not_tested.extend(detected[_MAX_AUTO_OUTCOME_COLUMNS:])
    if feature_columns is None:
        detected_f = _detect_feature_columns(df, protected_attributes, outcome_columns)
        feature_columns = detected_f[:_MAX_AUTO_FEATURE_COLUMNS]
        not_tested.extend(detected_f[_MAX_AUTO_FEATURE_COLUMNS:])

    for attr in protected_attributes:
        if attr not in df.columns:
            continue

        for outcome in outcome_columns:
            if outcome not in df.columns or outcome == attr:
                continue

            result = _test_disparity(
                df,
                outcome,
                attr,
                disparity_type=DisparityType.OUTCOME,
                min_group_size=min_group_size,
                excluded=undersized,
            )
            if result:
                results.append(result)

        for feature in feature_columns:
            if feature not in df.columns or feature == attr:
                continue

            result = _test_disparity(
                df,
                feature,
                attr,
                disparity_type=DisparityType.DISTRIBUTION,
                min_group_size=min_group_size,
                excluded=undersized,
            )
            if result:
                results.append(result)

        if include_quality_analysis:
            quality_results = _analyze_quality_disparities(df, attr, min_group_size=min_group_size)
            results.extend(quality_results)

    if include_intersectional and len(protected_attributes) >= 2:
        for combo in combinations(protected_attributes, 2):
            if not all(a in df.columns for a in combo):
                continue

            for outcome in outcome_columns[:3]:  # Limit for performance
                if outcome not in df.columns:
                    continue

                inter_results = _analyze_intersectional_disparities(
                    df,
                    outcome,
                    list(combo),
                    min_group_size=min_group_size,
                )
                results.extend(inter_results)

    if correction_method and results:
        results = _apply_correction(results, correction_method, significance_level)

    significant_results = [r for r in results if r.significance not in _NON_FINDING_SIGNIFICANCE]
    # An unmeasured effect size must not silently reorder the ranking: NaN
    # compares False against everything, which makes a sort key holding one a
    # non-total order and puts elements wherever the algorithm happens to
    # leave them. Unknown magnitudes rank last among equal p-values.
    significant_results.sort(
        key=lambda x: (
            x.pvalue,
            -abs(x.effect_size) if np.isfinite(x.effect_size) else float("inf"),
        )
    )

    # Disclose what was NOT tested, so an empty list is distinguishable from
    # "tested and clean". Both of these silently deleted real findings.
    if not_tested:
        shown = ", ".join(not_tested[:10])
        more = "" if len(not_tested) <= 10 else f" (and {len(not_tested) - 10} more)"
        warnings.warn(
            f"analyze_statistical_disparities auto-detected more columns than it "
            f"tests: {len(not_tested)} column(s) were NOT TESTED and cannot be "
            f"reported as clean: {shown}{more}. The caps "
            f"({_MAX_AUTO_OUTCOME_COLUMNS} outcomes, {_MAX_AUTO_FEATURE_COLUMNS} "
            f"features) keep the multiple-comparison family small; pass "
            f"outcome_columns=/feature_columns= explicitly to test the columns "
            f"you care about.",
            UserWarning,
            stacklevel=2,
        )
    if undersized:
        shown = "; ".join(undersized[:10])
        more = "" if len(undersized) <= 10 else f" (and {len(undersized) - 10} more)"
        warnings.warn(
            f"analyze_statistical_disparities excluded {len(undersized)} comparison(s) "
            f"from testing because a group held fewer than min_group_size="
            f"{min_group_size} rows: {shown}{more}. Those groups were NOT TESTED and "
            f"their absence from the returned list is NOT evidence that they are free "
            f"of disparity. Lower min_group_size to test them, at correspondingly "
            f"lower statistical power.",
            UserWarning,
            stacklevel=2,
        )

    n_unmeasured = sum(
        1 for r in results if r.effect_interpretation is EffectSizeInterpretation.NOT_MEASURABLE
    )
    if n_unmeasured:
        warnings.warn(
            f"{n_unmeasured} of {len(results)} comparison(s) have an UNMEASURED effect "
            f"size (effect_size is NaN, effect_interpretation is not_measurable): every "
            f"group was constant, so there is no within-group spread to standardise the "
            f"difference by. This is not a small effect and must not be read as one; the "
            f"raw gap is on disparity_magnitude and group_statistics.",
            UserWarning,
            stacklevel=2,
        )

    n_untestable = sum(1 for r in results if r.significance is SignificanceLevel.NOT_TESTABLE)
    if n_untestable:
        warnings.warn(
            f"{n_untestable} of {len(results)} comparison(s) could not be tested "
            f"(a non-finite test statistic, e.g. a constant column). They are "
            f"reported as not_testable, are excluded from the multiple-comparison "
            f"family size, and are NOT evidence of an absence of disparity.",
            UserWarning,
            stacklevel=2,
        )

    return significant_results


def _test_disparity(
    df: pd.DataFrame,
    target_column: str,
    protected_attr: str,
    *,
    disparity_type: DisparityType,
    min_group_size: int,
    excluded: Optional[List[str]] = None,
) -> Optional[StatisticalDisparityResult]:
    """Test for disparity in a single column across protected groups.

    *excluded*, when given, is appended to with one line per comparison that
    lost a group to ``min_group_size``. Returning a bare ``None`` made "no
    test was run" indistinguishable from "tested and clean": measured
    2026-09-11, 200 majority rows all approved against 25 minority rows all
    rejected returned ``[]`` with no warning, and so did a genuinely clean
    100-vs-100 frame at 50%/50% -- the two return values were identical. The
    same excluded frame at ``min_group_size=10`` reports the gap at
    p=9.7e-34. The caller turns this list into a UserWarning, which is how
    this module already discloses its other untested comparisons.
    """
    try:
        groups = df.groupby(protected_attr)[target_column]
        group_data = {}
        sample_sizes = {}
        dropped: List[str] = []

        for name, group in groups:
            data = group.dropna()
            if len(data) >= min_group_size:
                group_data[str(name)] = data
                sample_sizes[str(name)] = len(data)
            else:
                dropped.append(f"{name!s} (n={len(data)})")

        if dropped and excluded is not None:
            kept = len(group_data)
            excluded.append(
                f"'{target_column}' by '{protected_attr}': "
                + ", ".join(dropped)
                + (
                    ": fewer than 2 groups remained, so NO test was run for this column"
                    if kept < 2
                    else f": excluded; the test ran on the {kept} remaining group(s) only"
                )
            )

        if len(group_data) < 2:
            return None

        target_series = df[target_column].dropna()

        try:
            is_numeric = np.issubdtype(target_series.dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        is_binary = target_series.nunique() == 2

        if is_numeric and not is_binary:
            # Continuous outcome: use t-test or ANOVA
            result = _test_continuous_disparity(
                group_data, target_column, protected_attr, disparity_type, sample_sizes
            )
        else:
            # Categorical/binary: use chi-squared or proportion test
            result = _test_categorical_disparity(
                df, target_column, protected_attr, disparity_type, sample_sizes, min_group_size
            )

        return result

    except Exception:
        return None


def _test_continuous_disparity(
    group_data: Dict[str, pd.Series],
    target_column: str,
    protected_attr: str,
    disparity_type: DisparityType,
    sample_sizes: Dict[str, int],
) -> Optional[StatisticalDisparityResult]:
    """Test disparity for continuous variables."""
    try:
        group_names = list(group_data.keys())
        group_values = [group_data[g].values for g in group_names]

        group_statistics = {}
        for name, values in zip(group_names, group_values):
            group_statistics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "n": len(values),
            }

        # Two groups: t-test, more groups: ANOVA
        if len(group_data) == 2:
            stat, pvalue = stats.ttest_ind(group_values[0], group_values[1])
            test_name = "Independent t-test"

            # Cohen's d effect size
            effect_size = _cohens_d(group_values[0], group_values[1])
            effect_type = "Cohen's d"

        else:
            stat, pvalue = stats.f_oneway(*group_values)
            test_name = "One-way ANOVA"

            # Eta-squared effect size
            effect_size = _eta_squared(group_values)
            effect_type = "Eta-squared"

        # Identify privileged/disadvantaged groups (by mean)
        means = {name: group_statistics[name]["mean"] for name in group_names}
        privileged = max(means, key=lambda name: means[name])
        disadvantaged = min(means, key=lambda name: means[name])
        disparity_magnitude = means[privileged] - means[disadvantaged]

        # Confidence interval for difference (for two groups)
        ci: Tuple[Optional[float], Optional[float]]
        if len(group_data) == 2:
            ci = _mean_difference_ci(group_values[0], group_values[1])
        else:
            ci = (None, None)

        significance = _interpret_significance(pvalue)
        effect_interpretation = _interpret_effect_size(abs(effect_size), effect_type)

        recommendations = _generate_disparity_recommendations(
            significance, effect_interpretation, disparity_type, privileged, disadvantaged
        )

        return StatisticalDisparityResult(
            feature=target_column,
            protected_attribute=protected_attr,
            disparity_type=disparity_type,
            test_name=test_name,
            test_statistic=float(stat),
            pvalue=float(pvalue),
            significance=significance,
            effect_size=float(effect_size),
            effect_size_type=effect_type,
            effect_interpretation=effect_interpretation,
            confidence_interval=ci,
            group_statistics=group_statistics,
            privileged_group=privileged,
            disadvantaged_group=disadvantaged,
            disparity_magnitude=float(disparity_magnitude),
            sample_sizes=sample_sizes,
            recommendations=recommendations,
        )

    except Exception:
        return None


def _test_categorical_disparity(
    df: pd.DataFrame,
    target_column: str,
    protected_attr: str,
    disparity_type: DisparityType,
    sample_sizes: Dict[str, int],
    min_group_size: int,
) -> Optional[StatisticalDisparityResult]:
    """Test disparity for categorical/binary variables."""
    try:
        contingency = pd.crosstab(df[protected_attr], df[target_column])

        contingency = contingency[contingency.sum(axis=1) >= min_group_size]

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None

        # Chi-squared test, with the validity guard it was missing: the
        # chi-square approximation is invalid when any expected cell count
        # is < 5 (Cochran's rule). For a 2x2 table fall back to Fisher's
        # exact test; otherwise keep chi-square but flag it low-validity.
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        test_name = "Chi-squared test"
        if np.min(expected) < 5:
            if contingency.shape == (2, 2):
                _, pvalue = stats.fisher_exact(contingency.values)
                test_name = "Fisher's exact test"
            else:
                test_name = "Chi-squared test (low validity: sparse cells)"

        # Cramér's V effect size, through the shared refusal (see
        # _cramers_v_from_chi2). The `contingency.shape < 2` guard above means
        # a degenerate table returns None from this function before reaching
        # here, so today this call always measures; it goes through the helper
        # so that the guard and the statistic can never drift apart, and so a
        # NaN would be graded not_measurable instead of negligible.
        cramers_v = _cramers_v_from_chi2(
            float(chi2),
            int(contingency.sum().sum()),
            (contingency.shape[0], contingency.shape[1]),
            "_test_categorical_disparity",
        )
        effect_type = "Cramér's V"

        # Calculate group proportions (for binary outcomes, use positive rate)
        group_statistics = {}
        # DECLARED ABOVE THE BRANCH, because the return below reads both and the
        # multi-class arm never enters the block that sets them. A NameError here
        # would be swallowed by this function's `except Exception: return None`,
        # turning every multi-class disparity into a silent absence of result.
        positive_col = None
        direction_reason: Optional[str] = None
        if contingency.shape[1] == 2:
            # Binary outcome. Resolve the POSITIVE label explicitly --
            # assuming the last column is positive silently inverts the
            # disparity sign whenever sort order puts the negative label
            # last (e.g. "denied" > "approved" alphabetically).
            _POS = {
                "1",
                "1.0",
                "true",
                "yes",
                "y",
                "approved",
                "hired",
                "selected",
                "invite",
                "invited",
                "positive",
                "pass",
                "accept",
                "accepted",
                "qualified",
                "eligible",
            }
            cols = list(contingency.columns)
            positive_col = next((c for c in cols if str(c).strip().lower() in _POS), None)
            if positive_col is None and set(str(c).strip() for c in cols) == {"0", "1"}:
                positive_col = next(c for c in cols if str(c).strip() == "1")
            # NO LAST RESORT. `positive_col = cols[-1]` used to stand here, and the
            # comment above this whitelist already says why that is wrong: sort order
            # puts the NEGATIVE label last for ordinary vocabulary. The whitelist was
            # added to fix exactly that and then left the original defect in place as
            # the fallback, so every label outside the whitelist still got it.
            #
            # Measured 2026-09-30 with labels 'admit' and 'waitlist', neither of them
            # exotic for an admissions audit: 'waitlist' sorts last, so the rate was
            # read as the waitlist rate and the result named group B as PRIVILEGED and
            # group A as DISADVANTAGED, when A was admitted 90% of the time and B 10%.
            # Zero warnings. With 'offer' and 'decline' the same code is right by luck.
            # This is worse than a fabricated magnitude: it points a real finding at
            # the wrong group, so it cannot be allowed to fall through to a guess.
            #
            # The rate is still computed, against a named reference label, because it
            # is genuinely measured and a reader who knows their own data settles the
            # direction instantly from positive_class_used. What is withheld is the
            # DIRECTION, which is the only part that is not determinable here.
            if positive_col is None:
                positive_col = cols[-1]
                direction_reason = (
                    f"neither outcome label ({', '.join(repr(str(c)) for c in cols)}) is "
                    f"recognisable as the favourable one, so which group is privileged "
                    f"cannot be read off the data. positive_rate is the rate of "
                    f"{str(positive_col)!r}; if that is the favourable outcome then the "
                    f"higher rate is the privileged group, and if it is not, the lower "
                    f"one is. Encode the outcome as 0/1 to have this decided."
                )
                warnings.warn(
                    f"_test_categorical_disparity: {direction_reason}",
                    UserWarning,
                    stacklevel=2,
                )
            for group in contingency.index:
                row = contingency.loc[group]
                total = row.sum()
                positive_rate = row[positive_col] / total if total > 0 else 0
                group_statistics[str(group)] = {
                    "positive_rate": float(positive_rate),
                    "n": int(total),
                }
        else:
            # Multi-class
            for group in contingency.index:
                row = contingency.loc[group]
                total = row.sum()
                group_statistics[str(group)] = {
                    "distribution": (row / total).to_dict() if total > 0 else {},
                    "n": int(total),
                }

        # Identify privileged/disadvantaged (for binary outcomes)
        if contingency.shape[1] == 2:
            rates = {g: group_statistics[g]["positive_rate"] for g in group_statistics}
            higher = max(rates, key=lambda g: rates[g])
            lower = min(rates, key=lambda g: rates[g])
            # THE MAGNITUDE IS DIRECTION INVARIANT and stays measured either way:
            # max minus min is the same gap whichever label is called positive, as
            # are chi2, the p-value and Cramer's V. Only WHO IS FAVOURED flips, so
            # only that is withheld. Suppressing the gap as well would delete a real
            # finding in order to avoid naming a direction.
            disparity_magnitude = rates[higher] - rates[lower]
            privileged = None if direction_reason else higher
            disadvantaged = None if direction_reason else lower
        else:
            # THE SAME DEFECT, BY DICT ORDER RATHER THAN SORT ORDER. For a
            # multi-class outcome there is no positive rate to rank groups by, so
            # these two lines named the FIRST and LAST group encountered as
            # privileged and disadvantaged. That is not a measurement of anything:
            # it is insertion order, and it reads as a finding about who is
            # harmed. Cramer's V is a strength, not a direction, and it is kept.
            privileged = None
            disadvantaged = None
            direction_reason = (
                f"the outcome has {contingency.shape[1]} classes, not two, so there "
                f"is no single favourable outcome to rank the groups by. "
                f"effect_size is Cramer's V, which measures the STRENGTH of the "
                f"association and carries no direction. Read group_statistics for "
                f"each group's distribution across the classes."
            )
            disparity_magnitude = cramers_v

        significance = _interpret_significance(pvalue)
        effect_interpretation = _interpret_effect_size(cramers_v, effect_type)

        # Confidence interval for proportion difference (for binary, two groups)
        ci: Tuple[Optional[float], Optional[float]]
        if contingency.shape[0] == 2 and contingency.shape[1] == 2:
            # Use the SAME resolved positive-class column as the point
            # estimate above: taking the last crosstab column flipped the
            # CI's sign whenever sort order put the negative label last
            # (e.g. 'denied' after 'approved').
            ci = _proportion_difference_ci(
                contingency.iloc[0][positive_col],
                contingency.iloc[0].sum(),
                contingency.iloc[1][positive_col],
                contingency.iloc[1].sum(),
            )
        else:
            ci = (None, None)

        recommendations = _generate_disparity_recommendations(
            significance, effect_interpretation, disparity_type, privileged, disadvantaged
        )

        return StatisticalDisparityResult(
            feature=target_column,
            protected_attribute=protected_attr,
            disparity_type=disparity_type,
            test_name=test_name,
            test_statistic=float(chi2),
            pvalue=float(pvalue),
            significance=significance,
            effect_size=float(cramers_v),
            effect_size_type=effect_type,
            effect_interpretation=effect_interpretation,
            confidence_interval=ci,
            group_statistics=group_statistics,
            privileged_group=privileged,
            disadvantaged_group=disadvantaged,
            disparity_magnitude=float(disparity_magnitude),
            sample_sizes=sample_sizes,
            recommendations=recommendations,
            positive_class_used=(str(positive_col) if positive_col is not None else None),
            direction_not_determined_reason=direction_reason,
        )

    except Exception:
        return None


def _analyze_quality_disparities(
    df: pd.DataFrame,
    protected_attr: str,
    *,
    min_group_size: int,
) -> List[StatisticalDisparityResult]:
    """Analyze data quality disparities (missing rates, etc.) across groups."""
    results: List[StatisticalDisparityResult] = []

    try:
        groups = df[protected_attr].unique()
        valid_groups = []

        for g in groups:
            if pd.isna(g):
                continue
            group_size = (df[protected_attr] == g).sum()
            if group_size >= min_group_size:
                valid_groups.append(str(g))

        if len(valid_groups) < 2:
            return results

        for col in df.columns:
            if col == protected_attr:
                continue

            group_missing_rates = {}
            sample_sizes = {}

            for g in valid_groups:
                mask = df[protected_attr].astype(str) == g
                group_data = df.loc[mask, col]
                missing_rate = group_data.isna().mean()
                group_missing_rates[g] = missing_rate
                sample_sizes[g] = int(mask.sum())

            # Check if there's significant variation in missing rates
            rates = list(group_missing_rates.values())
            if max(rates) - min(rates) > 0.05:  # >5% difference
                # Proportion test for missing rates
                max_group = max(group_missing_rates, key=lambda g: group_missing_rates[g])
                min_group = min(group_missing_rates, key=lambda g: group_missing_rates[g])

                # Z-test for proportion difference
                n1, n2 = sample_sizes[max_group], sample_sizes[min_group]
                p1, p2 = group_missing_rates[max_group], group_missing_rates[min_group]

                if n1 > 0 and n2 > 0:
                    pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
                    if pooled_p > 0 and pooled_p < 1:
                        se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
                        if se > 0:
                            z_stat = (p1 - p2) / se
                            pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                            significance = _interpret_significance(pvalue)

                            if significance not in _NON_FINDING_SIGNIFICANCE:
                                results.append(
                                    StatisticalDisparityResult(
                                        feature=f"{col}_missing_rate",
                                        protected_attribute=protected_attr,
                                        disparity_type=DisparityType.QUALITY,
                                        test_name="Z-test for proportions",
                                        test_statistic=float(z_stat),
                                        pvalue=float(pvalue),
                                        significance=significance,
                                        effect_size=float(p1 - p2),
                                        effect_size_type="Proportion difference",
                                        effect_interpretation=EffectSizeInterpretation.MEDIUM
                                        if abs(p1 - p2) > 0.1
                                        else EffectSizeInterpretation.SMALL,
                                        confidence_interval=(None, None),
                                        group_statistics={
                                            g: {"missing_rate": r}
                                            for g, r in group_missing_rates.items()
                                        },
                                        privileged_group=min_group,
                                        disadvantaged_group=max_group,
                                        disparity_magnitude=float(p1 - p2),
                                        sample_sizes=sample_sizes,
                                        recommendations=[
                                            f"Higher missing rate for '{max_group}' in '{col}'",
                                            "Investigate data collection process for bias",
                                            "Consider imputation strategies that don't amplify bias",
                                        ],
                                    )
                                )

    except Exception:
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    return results


def _analyze_intersectional_disparities(
    df: pd.DataFrame,
    outcome: str,
    attributes: List[str],
    *,
    min_group_size: int,
) -> List[StatisticalDisparityResult]:
    """Analyze disparities at intersectional level."""
    results: List[StatisticalDisparityResult] = []

    try:
        df_valid = df.dropna(subset=attributes + [outcome])

        intersect_col = df_valid[attributes].apply(
            lambda row: "_".join(str(v) for v in row), axis=1
        )

        group_counts = intersect_col.value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index.tolist()

        if len(valid_groups) < 2:
            return results

        mask = intersect_col.isin(valid_groups)
        df_filtered = df_valid[mask].copy()
        df_filtered["_intersect"] = intersect_col[mask]

        result = _test_disparity(
            df_filtered,
            outcome,
            "_intersect",
            disparity_type=DisparityType.INTERSECTIONAL,
            min_group_size=min_group_size,
        )

        if result:
            # Update the result to indicate intersectionality
            result.protected_attribute = f"Intersection({', '.join(attributes)})"
            results.append(result)

    except Exception:
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    return results


# Effect size calculations


def compute_effect_sizes(
    df: pd.DataFrame,
    target_column: str,
    protected_attribute: str,
) -> Dict[str, Any]:
    """
    Compute various effect sizes for group differences.

    Args:
        df: DataFrame to analyze
        target_column: Column to analyze
        protected_attribute: Grouping column

    Returns:
        Dictionary with effect sizes and interpretations
    """
    result: Dict[str, Any] = {
        "target": target_column,
        "protected_attribute": protected_attribute,
        "effect_sizes": {},
    }

    try:
        groups = df.groupby(protected_attribute)[target_column]
        present = {str(name): group.dropna().values for name, group in groups}

        # A group whose every value is unmeasured is not a group WITH DATA, and
        # the guard below already says so in words. It counted keys, not rows,
        # so an empty array satisfied it and then travelled into the effect-size
        # code as though it were an observed group. Measured 2026-09-27 on 40
        # rows of 'a' against 40 rows of 'b' whose target was entirely NaN:
        # len(group_data) was 2, the guard passed, and eta-squared came back
        # NaN graded "large" (see _interpret_eta_squared) with no disclosure
        # naming 'b'. Dropping it here is what the rest of this module does
        # (see the `dropped` list in _test_disparity), and the drop is reported
        # rather than silent.
        group_data = {name: vals for name, vals in present.items() if len(vals)}
        groups_without_data = sorted(name for name, vals in present.items() if not len(vals))
        if groups_without_data:
            result["groups_without_data"] = groups_without_data
            warnings.warn(
                f"compute_effect_sizes: group(s) {groups_without_data} have no measured "
                f"value of {target_column!r} at all, so they were NOT compared; the effect "
                f"sizes below describe the {len(group_data)} group(s) that had data. Their "
                "absence is a could-not-check, not evidence that they are alike.",
                UserWarning,
                stacklevel=2,
            )

        if len(group_data) < 2:
            result["error"] = "Fewer than 2 groups with data"
            return result

        sample = df[target_column].dropna()
        try:
            is_numeric = np.issubdtype(sample.dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        if is_numeric and sample.nunique() > 2:
            # Continuous: Cohen's d (for 2 groups) or eta-squared
            if len(group_data) == 2:
                g1, g2 = list(group_data.values())
                d = _cohens_d(g1, g2)
                result["effect_sizes"]["cohens_d"] = {
                    "value": float(d),
                    "interpretation": _interpret_effect_size(abs(d), "Cohen's d").value,
                }

            eta_sq = _eta_squared(list(group_data.values()))
            result["effect_sizes"]["eta_squared"] = {
                "value": float(eta_sq),
                "interpretation": _interpret_eta_squared(eta_sq),
            }

        else:
            # Categorical: Cramér's V
            contingency = pd.crosstab(df[protected_attribute], df[target_column])
            chi2 = stats.chi2_contingency(contingency)[0]
            # Nothing above constrains this table's shape: a constant target
            # column (every row "approved") crosstabs to r x 1 and the old
            # `else 0` answered "negligible" for it. See _cramers_v_from_chi2.
            v = _cramers_v_from_chi2(
                float(chi2),
                int(contingency.sum().sum()),
                (contingency.shape[0], contingency.shape[1]),
                "compute_effect_sizes",
            )

            result["effect_sizes"]["cramers_v"] = {
                "value": float(v),
                "interpretation": _interpret_effect_size(v, "Cramér's V").value,
            }

            # For binary outcome: odds ratio
            if contingency.shape[1] == 2 and contingency.shape[0] == 2:
                try:
                    a, b = contingency.iloc[0]
                    c, d = contingency.iloc[1]
                    odds_ratio = (a * d) / (b * c) if b * c > 0 else np.inf
                    result["effect_sizes"]["odds_ratio"] = {
                        "value": float(odds_ratio),
                        "interpretation": _interpret_odds_ratio(odds_ratio),
                    }
                except Exception:
                    logging.getLogger(__name__).debug(
                        "optional computation failed; skipping", exc_info=True
                    )

    except Exception as e:
        result["error"] = str(e)

    return result


def _cramers_v_from_chi2(chi2: float, n: int, shape: Tuple[int, int], where: str) -> float:
    """Cramer's V from a chi-square statistic, or NaN when it is undefined.

    ONE helper for what were two byte-identical copies of the same expression
    (``_test_categorical_disparity`` and :func:`compute_effect_sizes`). They
    are different functions, not one duplicated, and each carried its own
    ``else 0``; a third copy is one feature away, so the refusal lives here
    rather than at the call sites.

    V = sqrt(chi2 / (n * (min(r, c) - 1))) needs at least 2 rows AND 2 columns
    and at least one observation. A table with a single row or a single column
    gives the outcome nothing to vary AGAINST, so the association is not zero,
    it is undefined. The old ``else 0`` published 0.0, which lands in the
    NEGLIGIBLE band and reads as a clean bill of health. Measured 2026-09-17
    through the public ``compute_effect_sizes`` on 200 rows whose outcome was
    the constant "approved": the 2x1 table returned
    ``{"cramers_v": {"value": 0.0, "interpretation": "negligible"}}`` and
    raised no warning at all. NaN is graded NOT_MEASURABLE by
    :func:`_interpret_effect_size`, which is the could-not-check state this
    module already uses for Cohen's d and eta-squared.
    """
    rows, cols = int(shape[0]), int(shape[1])
    min_dim = min(rows, cols) - 1
    n_int = int(n)
    if min_dim <= 0 or n_int <= 0 or not np.isfinite(chi2):
        warnings.warn(
            f"{where}: Cramer's V is not defined for a {rows}x{cols} table of "
            f"{n_int} observation(s) with chi-square {chi2}. It needs at least "
            f"2 rows AND 2 columns, a positive count and a finite chi-square, "
            f"so the association was NOT measured. Returning NaN, graded "
            f"not_measurable. This is a could-not-check, never a finding that "
            f"the groups are alike.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")
    return float(np.sqrt(chi2 / (n_int * min_dim)))


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.

    Returns NaN, never 0.0, when there is no within-group spread to
    standardise the difference by. With both groups constant the true value
    is unbounded, not zero, and 0.0 is the NEGLIGIBLE band: measured
    2026-09-11 on 40 rows at 0.90 vs 40 rows at 0.10, Cohen's d came back
    0.0 / NEGLIGIBLE with the recommendation "Statistically significant but
    small effect. Monitor but may not require immediate intervention.", while
    the same gap with sd=0.05 noise returned 15.68 / LARGE / "Investigate
    root causes." NaN is carried through _interpret_effect_size, which grades
    it NOT_MEASURABLE rather than letting it fall into a magnitude band.

    The degeneracy is read off the RAW data with np.ptp, not off the
    accumulated variance: np.var of a constant array is exactly 0.0 only at
    some n, so ``pooled_std == 0`` is a guard that works for the fixture
    value and lets a 1e-17 residue through as a colossal "measurement".
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return float("nan")

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # THE DUPLICATE, CORRECTED 2026-09-29. `np.ptp` is exact, and the EQUALITY
    # WITH ZERO this used to be was not: it asks whether the array is constant in
    # the LAST BIT, which only a caller-supplied constant array is. An array the
    # caller COMPUTED, a residual or a difference, is constant to a few ulps and
    # walks straight past it. Measured here on `(pred + 50.0) - pred` over 40 rows,
    # peak-to-peak 2.84e-14: this returned **-6624998218327707.0 with zero
    # warnings** for two groups whose real gap is 50. The canonical copy in
    # evaluation/vfairness_metrics/_statistics.py was fixed the same day and this
    # is the same test, data-scaled so it carries no units.
    _sp1, _sp2 = float(np.ptp(group1)), float(np.ptp(group2))
    _mag = max(abs(float(np.mean(group1))), abs(float(np.mean(group2))), _sp1, _sp2)
    _atol = 1e-12 * _mag
    no_spread = _sp1 <= _atol and _sp2 <= _atol
    if no_spread or not np.isfinite(pooled_std) or pooled_std <= 0:
        return float("nan")

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Calculate eta-squared effect size for ANOVA.

    Sibling of :func:`_cohens_d` under the same dispatch in
    ``_test_continuous_disparity`` (two groups -> Cohen's d, more -> this),
    and fixed with it: fixing one branch of a dispatch only moves the
    fabrication to the other. With no variance at all to partition, the ratio
    is 0/0 and the answer is NaN / NOT_MEASURABLE, not the NEGLIGIBLE band.
    """
    all_data = np.concatenate(groups) if groups else np.asarray([])
    if all_data.size == 0:
        return float("nan")

    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_data - grand_mean) ** 2)

    # Read off the RAW data, not off the accumulated sum of squares, for the
    # same reason as _cohens_d: an accumulated statistic is exactly 0.0 only
    # at some n.
    if bool(np.ptp(all_data) == 0) or not np.isfinite(ss_total) or ss_total <= 0:
        return float("nan")

    return ss_between / ss_total


def _mean_difference_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for mean difference."""
    n1, n2 = len(group1), len(group2)
    mean_diff = np.mean(group1) - np.mean(group2)

    se = np.sqrt(np.var(group1, ddof=1) / n1 + np.var(group2, ddof=1) / n2)
    df = n1 + n2 - 2

    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * se

    return (mean_diff - margin, mean_diff + margin)


def _proportion_difference_ci(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for proportion difference."""
    p1, p2 = x1 / n1 if n1 > 0 else 0, x2 / n2 if n2 > 0 else 0
    diff = p1 - p2

    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0

    z_crit = stats.norm.ppf((1 + confidence) / 2)
    margin = z_crit * se

    return (diff - margin, diff + margin)


# Interpretation helpers


def _interpret_significance(pvalue: float) -> SignificanceLevel:
    """Interpret p-value significance.

    A non-finite p-value is NOT_TESTABLE, never NOT_SIGNIFICANT. Every
    comparison against NaN is False, so a NaN used to fall through all three
    branches into the final ``else`` and be graded "tested, no significant
    disparity" -- a measurement claim about a test that never ran.
    """
    if pvalue is None or not np.isfinite(pvalue):
        return SignificanceLevel.NOT_TESTABLE
    if pvalue < 0.001:
        return SignificanceLevel.HIGHLY_SIGNIFICANT
    elif pvalue < 0.01:
        return SignificanceLevel.SIGNIFICANT
    elif pvalue < 0.05:
        return SignificanceLevel.MARGINALLY_SIGNIFICANT
    else:
        return SignificanceLevel.NOT_SIGNIFICANT


def _interpret_effect_size(value: float, effect_type: str) -> EffectSizeInterpretation:
    """Interpret effect size magnitude.

    The non-finite guard sits ABOVE the dispatch on ``effect_type`` on
    purpose. Every band below is a chain of ``<`` comparisons and every
    comparison against NaN is False, so a NaN falls through to the final
    ``else`` of whichever branch it entered and is published as a confident
    grade: LARGE for Cohen's d, LARGE for Cramer's V, LARGE for the generic
    band. Guarding inside one branch would just move the fabricated grade to
    its sibling.
    """
    if value is None or not np.isfinite(value):
        return EffectSizeInterpretation.NOT_MEASURABLE

    if effect_type == "Cohen's d":
        if abs(value) < 0.2:
            return EffectSizeInterpretation.NEGLIGIBLE
        elif abs(value) < 0.5:
            return EffectSizeInterpretation.SMALL
        elif abs(value) < 0.8:
            return EffectSizeInterpretation.MEDIUM
        else:
            return EffectSizeInterpretation.LARGE

    elif effect_type == "Cramér's V":
        if value < 0.1:
            return EffectSizeInterpretation.NEGLIGIBLE
        elif value < 0.3:
            return EffectSizeInterpretation.SMALL
        elif value < 0.5:
            return EffectSizeInterpretation.MEDIUM
        else:
            return EffectSizeInterpretation.LARGE

    elif effect_type == "Eta-squared":
        if value < 0.01:
            return EffectSizeInterpretation.NEGLIGIBLE
        elif value < 0.06:
            return EffectSizeInterpretation.SMALL
        elif value < 0.14:
            return EffectSizeInterpretation.MEDIUM
        else:
            return EffectSizeInterpretation.LARGE

    else:
        # Generic interpretation
        if value < 0.1:
            return EffectSizeInterpretation.NEGLIGIBLE
        elif value < 0.3:
            return EffectSizeInterpretation.SMALL
        elif value < 0.5:
            return EffectSizeInterpretation.MEDIUM
        else:
            return EffectSizeInterpretation.LARGE


def _interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared value, or say it was not measurable.

    The sibling :func:`_interpret_effect_size` grew a non-finite guard above its
    dispatch for exactly this reason, and this second, string-valued
    interpretation of the SAME statistic kept none of it. Every band below is a
    ``<`` comparison and every comparison against NaN is False, so a NaN fell
    through to the final ``else``: measured 2026-09-27 through the public
    :func:`compute_effect_sizes` on 40 rows of group 'a' beside 40 rows whose
    target was entirely unmeasured, ``_eta_squared`` correctly answered NaN and
    this function published
    ``{"eta_squared": {"value": nan, "interpretation": "large"}}``, the strongest
    magnitude band on the scale, for a statistic nobody could compute. The
    sibling on the same result dict already read 'not_measurable' for its own
    NaN, so one object carried both states at once.
    """
    if eta_sq is None or not np.isfinite(eta_sq):
        return EffectSizeInterpretation.NOT_MEASURABLE.value
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def _interpret_odds_ratio(odds_ratio: float) -> str:
    """Interpret odds ratio."""
    if odds_ratio == 1:
        return "no effect"
    elif 0.67 <= odds_ratio <= 1.5:
        return "small"
    elif 0.5 <= odds_ratio <= 2.0:
        return "medium"
    else:
        return "large"


def _generate_disparity_recommendations(
    significance: SignificanceLevel,
    effect_interpretation: EffectSizeInterpretation,
    disparity_type: DisparityType,
    privileged: Optional[str],
    disadvantaged: Optional[str],
) -> List[str]:
    """Generate recommendations based on disparity analysis."""
    recommendations = []

    # NEVER f-STRING AN ABSENT GROUP NAME. Every line below interpolates these
    # two names, and `f"{None}"` is the readable word "None", so a withheld
    # direction would print as a recommendation about a group called None: the
    # absent value MINTING CONTENT, which is the defect class this module is
    # audited for. Both are None together, by construction at both call sites.
    if privileged is None or disadvantaged is None:
        recommendations.append(
            "Which group is favoured could NOT be determined, so no group is named \n"
            "here as privileged or disadvantaged. This is a could-not-check, not a \n"
            "finding that the groups are treated equally: read \n"
            "direction_not_determined_reason for why, and group_statistics with \n"
            "positive_class_used for the rates the decision would rest on."
        )
        if significance not in _NON_FINDING_SIGNIFICANCE:
            recommendations.append(
                "The association between the protected attribute and the outcome IS \n"
                "statistically significant, and disparity_magnitude is measured. Only \n"
                "its DIRECTION is unknown, so investigate: do not read an undetermined \n"
                "direction as an absence of harm."
            )
        return recommendations

    # An UNMEASURED magnitude is handled before the significance ladder, and
    # never falls into its "small effect" arm. That arm is where the defect
    # surfaced to the reader: a fabricated effect size of 0.0 on a total
    # 0.90-vs-0.10 separation produced "Statistically significant but small
    # effect. Monitor but may not require immediate intervention." Removing
    # the number must not remove the alarm, so the significant case keeps an
    # explicit investigate line and points at the raw gap that is still
    # measured and still sitting on the result.
    if effect_interpretation is EffectSizeInterpretation.NOT_MEASURABLE:
        recommendations.append(
            f"Effect size between '{privileged}' and '{disadvantaged}' could NOT be "
            f"computed (no within-group variation to standardise by), so the "
            f"MAGNITUDE of this disparity is UNMEASURED, not negligible. Read "
            f"disparity_magnitude and group_statistics for the raw gap."
        )
        if significance not in _NON_FINDING_SIGNIFICANCE:
            recommendations.append(
                f"The difference between '{privileged}' and '{disadvantaged}' is "
                f"statistically significant. Investigate root causes; do not read an "
                f"unmeasured effect size as a small one."
            )
        return recommendations

    if significance in [SignificanceLevel.HIGHLY_SIGNIFICANT, SignificanceLevel.SIGNIFICANT]:
        if effect_interpretation in [
            EffectSizeInterpretation.LARGE,
            EffectSizeInterpretation.MEDIUM,
        ]:
            recommendations.append(
                f"Significant and meaningful disparity detected between '{privileged}' "
                f"and '{disadvantaged}'. Investigate root causes."
            )

            if disparity_type == DisparityType.OUTCOME:
                recommendations.append(
                    "Consider fairness constraints or post-processing to reduce outcome disparity."
                )
            elif disparity_type == DisparityType.DISTRIBUTION:
                recommendations.append(
                    "Feature distribution differs significantly across groups. "
                    "Evaluate if this feature should be included in the model."
                )
            elif disparity_type == DisparityType.QUALITY:
                recommendations.append(
                    "Data quality differs across groups. Address data collection bias "
                    "and consider group-aware imputation strategies."
                )

        else:
            recommendations.append(
                "Statistically significant but small effect. Monitor but may not "
                "require immediate intervention."
            )

    elif significance == SignificanceLevel.MARGINALLY_SIGNIFICANT:
        recommendations.append(
            "Marginally significant disparity. Consider collecting more data or "
            "monitoring over time."
        )

    return recommendations


# Helper functions


# Column-name words that mark a model outcome / decision. Matched WHOLE-TOKEN
# (see _detect_outcome_columns), which is what makes the single letter "y"
# safe to keep: "y" is the conventional name of a target column, but as a
# SUBSTRING it is inside a great many ordinary words. Measured 2026-09-10,
# with substring matching, every one of these was reported as a model
# outcome: salary, yearly_bonus, city, county, payment_type, birthday,
# company_type, employment_years, study_years, display_name. Ten fabricated
# outcomes consumed the whole auto-detection budget, and the real outcome
# column was pushed out of it and re-tested as a "distribution" instead --
# which changes the regulatory framing of the finding downstream.
_OUTCOME_KEYWORDS = [
    "outcome",
    "result",
    "decision",
    "approved",
    "accepted",
    "rejected",
    "score",
    "rating",
    "risk",
    "label",
    "target",
    "y",
    "prediction",
    "status",
    "class",
    "category",
]


def _detect_outcome_columns(
    df: pd.DataFrame,
    protected_attributes: List[str],
) -> List[str]:
    """Auto-detect likely outcome columns (whole-token name match).

    Returns EVERY candidate. The caller applies the budget and discloses
    what it dropped; a helper that silently truncates makes "not tested"
    indistinguishable from "tested and clean".
    """
    outcomes = []

    for col in df.columns:
        if col in protected_attributes:
            continue
        if any(tokens_contain(col, keyword) for keyword in _OUTCOME_KEYWORDS):
            outcomes.append(col)

    return outcomes


def _detect_feature_columns(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_columns: List[str],
) -> List[str]:
    """Auto-detect feature columns for analysis.

    Returns EVERY candidate; the caller applies the budget and discloses it.
    """
    excluded = set(protected_attributes) | set(outcome_columns)
    features = []

    for col in df.columns:
        if col in excluded:
            continue

        # Include numeric columns with reasonable cardinality
        try:
            if np.issubdtype(df[col].dtype, np.number):
                features.append(col)
            elif df[col].nunique() < 50:
                features.append(col)
        except (TypeError, AttributeError):
            if df[col].nunique() < 50:
                features.append(col)

    # Every candidate. The caller budgets and discloses; see
    # _detect_outcome_columns.
    return features


def _apply_correction(
    results: List[StatisticalDisparityResult],
    method: str,
    alpha: float,
) -> List[StatisticalDisparityResult]:
    """Apply multiple comparison correction.

    Only TESTABLE comparisons enter the family. A comparison whose p-value is
    non-finite was never a hypothesis test, and counting it inflates m, which
    makes both Bonferroni and Benjamini-Hochberg strictly more conservative
    and deletes real findings. Measured 2026-09-10 through the public
    BiasDetector(...).analyze_disparities(), same data and same gender
    approval gap both times:

        0 constant columns -> 1 statistical disparity: [('approved', 0.00465)]
       18 constant columns -> 0 statistical disparities: []

    A constant column makes ttest_ind return NaN, every comparison against
    NaN is False, so it was graded NOT_SIGNIFICANT and still took a full slot
    in m. The reader saw "Statistical Disparities: 0" while the real finding
    had been erased by columns that could not be tested at all.
    """
    if not results:
        return results

    testable = [i for i, r in enumerate(results) if r.pvalue is not None and np.isfinite(r.pvalue)]
    for i, r in enumerate(results):
        if i not in set(testable):
            r.significance = SignificanceLevel.NOT_TESTABLE

    pvalues = [results[i].pvalue for i in testable]
    m = len(pvalues)
    if m == 0:
        return results

    if method == "bonferroni":
        adjusted_alpha = alpha / m
        for i in testable:
            if results[i].pvalue >= adjusted_alpha:
                results[i].significance = SignificanceLevel.NOT_SIGNIFICANT

    elif method == "fdr_bh":
        # Benjamini-Hochberg STEP-UP (1995): find the LARGEST rank k with
        # p_(k) <= (k/m)*alpha, then reject ALL hypotheses with rank <= k.
        # The previous code thresholded each p at its own rank with no
        # step-up max, which is not BH -- it under-rejects (loses the FDR
        # guarantee and power). This is the corrected, monotone procedure.
        order = list(np.argsort(pvalues))
        k_max = 0
        for rank, pos in enumerate(order, 1):
            if pvalues[pos] <= (rank / m) * alpha:
                k_max = rank
        rejected = {testable[pos] for pos in order[:k_max]}
        for i in testable:
            if i not in rejected:
                results[i].significance = SignificanceLevel.NOT_SIGNIFICANT

    return results


# The only test selection this function can honour. The test that runs is
# chosen from the DATA (dtype and group count) by _test_disparity, and each
# branch carries the matching effect size and confidence interval. There is
# no name-keyed selector anywhere in this module, so a caller-named test
# cannot be routed to anything.
#
# This tuple used to read 'auto', 'ttest', 'anova', 'chi2', 'mannwhitney' in
# the docstring while the parameter was never read at all: asking for
# 'mannwhitney' returned a parametric Independent t-test, and asking for
# 'ttest' on a binary column returned a Chi-squared test, both reported under
# the name of the test that actually ran. Publishing a number attributed to
# the wrong method is worse than not getting the number, so anything this
# function cannot honour now raises instead of being silently discarded.
# DO NOT widen this tuple without also implementing the routing, the matching
# effect size and the matching confidence interval for the added test.
_SUPPORTED_TEST_TYPES: Tuple[str, ...] = ("auto",)


def run_disparity_tests(
    df: pd.DataFrame,
    target: str,
    protected_attribute: str,
    *,
    test_type: str = "auto",
) -> Dict[str, Any]:
    """
    Run statistical tests for disparity between groups.

    Convenience function for running a single disparity test.

    The test is selected from the data, not by the caller: a continuous
    target gives an independent t-test (two groups) or a one-way ANOVA
    (three or more), and a binary or categorical target gives a chi-squared
    test (or Fisher's exact test on a sparse 2x2 table). The name of the
    test that actually ran is returned in ``test_name``.

    Args:
        df: DataFrame to analyze
        target: Column to test for disparity
        protected_attribute: Grouping column
        test_type: Only 'auto' is supported. Any other value, including the
            'ttest', 'anova', 'chi2' and 'mannwhitney' that earlier versions
            of this docstring advertised, raises ValueError rather than being
            ignored and answered with a different test.

    Returns:
        Dictionary with test results, including ``test_type_requested``
        (what was asked for), ``test_name`` (what actually ran, or None
        when no test could be run), ``min_group_size`` (the threshold this
        entry point applies) and ``groups_excluded``: one line per group that
        held fewer than ``min_group_size`` rows and was therefore NOT TESTED.
        An excluded group is a could-not-check about that group, never
        evidence that it is free of disparity.

    Raises:
        ConfigurationError: If ``test_type`` is anything other than 'auto'.
    """
    # The house refusal, so an unknown option raises the same type here as
    # everywhere else in the library. ConfigurationError subclasses ValueError,
    # so a caller already catching ValueError is unaffected.
    _check_option(
        test_type,
        name="test_type",
        allowed=_SUPPORTED_TEST_TYPES,
        hints={
            name: (
                "The test is selected from the data (t-test or one-way ANOVA "
                "for a continuous target, chi-squared or Fisher's exact for a "
                "categorical one) and the test that ran is reported in "
                "'test_name'. Refusing rather than running a different test "
                "under the requested name."
            )
            for name in ("ttest", "anova", "chi2", "mannwhitney")
        },
    )

    # `excluded` is the channel _test_disparity already documents, and this
    # entry point was the one caller that passed nothing, so the exclusions it
    # collected were thrown away. Measured 2026-09-27 on 125 rows in three
    # groups where the third held 5 rows and every one of them was rejected:
    # this function returned test_name='Chi-squared test', pvalue=1.0,
    # effect_size=0.0, effect_interpretation='negligible',
    # disparity_magnitude=0.0, sample_sizes={'a': 60, 'b': 60} and NO warning.
    # Group 'c' was absent from the whole payload, so the strongest all-clear
    # the scale can give was published for a comparison that never saw the only
    # group with a disparity. analyze_statistical_disparities discloses the
    # identical list; a fix at one call site of a shared helper is not a fix.
    min_group_size = 10
    excluded: List[str] = []
    result = _test_disparity(
        df,
        target,
        protected_attribute,
        disparity_type=DisparityType.OUTCOME,
        min_group_size=min_group_size,
        excluded=excluded,
    )

    if excluded:
        warnings.warn(
            f"run_disparity_tests excluded {len(excluded)} group comparison(s) from "
            f"testing because a group held fewer than min_group_size={min_group_size} "
            f"rows: {'; '.join(excluded)}. Those groups were NOT TESTED and their "
            "absence from the result is NOT evidence that they are free of disparity. "
            "Use analyze_statistical_disparities, which takes min_group_size, to test "
            "them at correspondingly lower statistical power.",
            UserWarning,
            stacklevel=2,
        )

    if result:
        payload = result.to_dict()
        payload["test_type_requested"] = test_type
        payload["min_group_size"] = min_group_size
        payload["groups_excluded"] = list(excluded)
        return payload
    else:
        # Could-not-check, distinct from both a measured result and a
        # refusal: test_name is None because no test ran.
        return {
            "error": "Could not run disparity test",
            "target": target,
            "test_type_requested": test_type,
            "test_name": None,
            "min_group_size": min_group_size,
            "groups_excluded": list(excluded),
        }


# Temporal / distribution-shift bias  (Suresh & Guttag 2021 deployment bias;
# EU AI Act Art. 15(4) feedback loops)
#
# A static snapshot cannot observe a live feedback loop, but it CAN flag the
# risk: split the rows chronologically and measure whether (a) the group
# mix and (b) the per-group selection rate drift across time. Population
# Stability Index (PSI) is the standard, dependency-light drift statistic.

# Ordered most-specific (event time) first. A generic 'date' must NOT win
# over 'application_date', and birth/DOB/age are NEVER a valid event axis
# (they are PII / the age attribute itself -- using them yields a nonsense
# "drift" that is really just age sorted by age).
_TIME_NAME_HINTS = [
    "application_date",
    "applied_date",
    "apply_date",
    "submission_date",
    "submitted_at",
    "created_at",
    "event_time",
    "event_date",
    "decision_date",
    "timestamp",
    "datetime",
    "application",
    "applied",
    "submitted",
    "created",
    "cohort",
    "period",
    "_at",
    "date",
    "time",
    "year",
    "month",
]
_TIME_NAME_BLOCK = [
    "birth",
    "dob",
    "yob",
    "born",
    "age",
    "dead",
    "death",
    "expire",
    "expiry",
    "updated",
]


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two categorical distributions
    (already normalised to proportions, same category order)."""
    e = np.clip(expected, 1e-6, None)
    a = np.clip(actual, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_severity(psi: float) -> str:
    # Industry convention: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
    # A sane PSI is ~0-1; anything above this bound is not a real drift
    # signal but the degenerate result of running PSI on a near-unique /
    # identifier-grade column (early vs late cohorts share almost no
    # categories). Never let that escalate, and never let it read as a clean
    # bill of health either: the comment here said "it is not assessable" and
    # the code returned "pass", which is the one verdict on this scale that
    # claims the opposite. Measured 2026-09-27 on 200 rows whose 20-level
    # attribute was REPLACED wholesale between the early and late cohort:
    # compositionPSI 23.0256, severity "pass", top-level severity "pass" and
    # the summary "No material temporal drift in the assessed attributes",
    # beside a plain sentence in the same entry reading "shifted materially
    # (PSI 23.03)". `not_assessed` keeps the non-escalation (the pulse
    # orchestrator only raises a finding on warn / critical) without publishing
    # a clean verdict for a statistic outside its valid range.
    if psi >= _PSI_DEGENERATE or not np.isfinite(psi):
        return NOT_ASSESSED
    if psi >= 0.25:
        return "critical"
    if psi >= 0.10:
        return "warn"
    return "pass"


# Max distinct categories for which PSI is a meaningful drift statistic.
# Above this a column is identifier-grade for drift purposes (one tiny
# group per value), so we skip it rather than emit a degenerate finding.
_PSI_MAX_CATS = 30
# PSI above this is mathematically degenerate (see _psi_severity).
_PSI_DEGENERATE = 3.0

# The fewest RECORDED values a cohort must hold before its group mix is a mix.
#
# BGL6 F12, 2026-09-29. ``detect_temporal_drift`` folded every unrecorded row into
# an invented "missing" category and then compared the result, so the composition
# PSI was dominated by a category nobody recorded. Measured on 1000 rows, 998 of
# them carrying no value, where the two compared cohorts hold exactly ONE recorded
# value each and they are DIFFERENT ones, so the recorded mix flips from 100 percent
# 'm' to 100 percent 'f': compositionPSI 0.0304, severity 'pass', the summary "No
# material temporal drift" and zero warnings. The invented category was doing the
# measuring, and the one real observation on each side was 0.2 percent of it.
#
# 10, the same floor ``_MIN_COMPARABLE_ROWS`` uses below for reading a comparison
# between two columns, and the one the sibling screens in this package already use
# for a correlation (``feature_engineering.correlation.MIN_SAMPLE_SIZE``). A mix
# read off one observation is not a mix, and the honest report is a could-not-check
# naming both counts, not a PSI computed from the padding.
_MIN_COHORT_RECORDED = 10

# The fewest rows in which BOTH of two columns hold a value before
# ``detect_specification_bias`` will read a comparison between them. Below it
# the two checks that compare columns cannot separate a finding from
# arithmetic: any two points are perfectly collinear, so a 2-row frame gives
# every numeric feature |r| = 1.00 against the outcome, and a handful of
# matching labels is a coin flip rather than evidence that the outcome column
# IS the prediction column. BGL5 2026-09-27: a 2-row frame {'approved': [1, 0],
# 'x': [5.0, 9.0], 'z': [2.0, 3.0]} published TWO critical target_leakage
# findings reading "|r|=1.00" and reported "target_leakage (2 feature column(s)
# scanned)". Over-reporting a CRITICAL nobody measured is the same defect class
# as under-reporting one, and it needs the same three states. 10 is the floor
# the sibling screens in this package already use for a correlation
# (``feature_engineering.correlation.MIN_SAMPLE_SIZE`` and the default
# ``min_periods`` of ``compute_pearson_correlation_matrix``).
_MIN_COMPARABLE_ROWS = 10

# The decision tokens ``detect_temporal_drift`` can READ in a text-typed
# prediction column. The truthy set is the one that shipped inline; the falsy set
# is new and is the whole point of the pair. BGL-F5 2026-09-30: with only a
# truthy whitelist, "not truthy" and "unreadable" were the same answer, so an
# absent value, a free-text comment and a numeric score all counted as
# rejections and moved the published selection rate. A value in NEITHER set is
# now a could-not-check, and 'none' is deliberately in the falsy set rather than
# left to fall through as unreadable, because a reader writing 'none' in a
# decision column means "not selected", not "no record".
_DECISION_TRUTHY = frozenset(
    {
        "yes",
        "true",
        "1",
        "y",
        "approved",
        "hired",
        "selected",
        "invite",
        "invited",
        "positive",
    }
)
_DECISION_FALSY = frozenset(
    {
        "no",
        "false",
        "0",
        "n",
        "denied",
        "declined",
        "rejected",
        "not hired",
        "not selected",
        "negative",
        "none",
    }
)
# The fewest READABLE predictions a cohort must hold before a selection rate is
# taken from it. The same floor the group axis uses for a mix, and for the same
# reason: a rate read off one or two recorded outcomes is not a rate, and the
# honest report is a could-not-check naming both counts.
_MIN_COHORT_PREDICTIONS = _MIN_COHORT_RECORDED


def detect_temporal_drift(
    df: pd.DataFrame,
    protected_attributes: List[str],
    *,
    time_column: Optional[str] = None,
    prediction: Optional[str] = None,
    n_periods: int = 2,
) -> Dict[str, Any]:
    """Distribution-shift / temporal-drift risk on a static file.

    Cuts the readable time axis into at most ``n_periods`` chronological
    periods and, per protected attribute, measures PSI of the group mix and the
    swing in per-group selection rate between the first and last period. Returns
    a taxonomy finding-shaped result. Never raises.

    The periods are cut over the DISTINCT INSTANTS in ``time_column``, not over
    row positions, so every row sharing an instant belongs to the same period and
    no instant appears in both the earliest and the latest cohort. The published
    verdict is therefore a fact about time and is invariant under the order of
    the rows in the file. When one instant holds most of the frame, the cohorts
    come out at their true sizes and the thin one is refused by
    ``attributesWithTooFewObservations`` rather than graded.

    Three states, never two. ``available`` is False, with a ``reason``, when
    nothing could be compared: no usable time column, a time column holding a
    single distinct value (a static snapshot has no earliest and no latest
    period), or not one of the named protected attributes qualified (missing
    from the frame, identifier-grade for PSI, or carrying no recorded value in
    either of the two compared periods). Rows whose timestamp cannot be read
    belong to no period and are excluded from every cohort, counted in
    ``nRowsWithUnreadableTime`` and named in a warning. When it is True,
    ``attributesNotInFrame``,
    ``skippedHighCardinality`` and ``attributesWithNoObservation`` name what
    was left out, and a per-attribute
    ``severity`` of ``'not_assessed'`` (with ``notAssessedReason``) means the
    composition PSI landed outside the range in which it grades drift. None of
    those is a ``'pass'``.

    The PREDICTION axis is three-state on the same terms. A row whose outcome
    cannot be read belongs to no selection rate: it is excluded rather than
    imputed to a rejection, counted in ``nRowsWithUnreadablePrediction`` and named
    in a warning. Per attribute, ``selectionRateFirst`` / ``selectionRateLast`` /
    ``selectionRateShift`` are floats when both cohorts hold at least
    ``_MIN_COHORT_PREDICTIONS`` readable outcomes and ``None`` when they do not,
    with ``selectionRateStatus`` and ``selectionRateNotMeasuredReason`` saying
    which and ``nPredictionsReadFirst`` / ``nPredictionsReadLast`` giving the
    counts. A ``None`` rate never escalates ``severity``. A text-typed prediction
    column is read: numbers spelled as text are read as scores, decision words as
    decisions, and anything that is neither is unreadable rather than a rejection.
    """
    try:
        if time_column is None:
            cand = [c for c in df.columns if not any(b in c.lower() for b in _TIME_NAME_BLOCK)]
            # Prefer the most-specific event-time hint over generic 'date'.
            for h in _TIME_NAME_HINTS:
                hit = next((c for c in cand if h == c.lower() or h in c.lower()), None)
                if hit is not None:
                    time_column = hit
                    break
        if time_column is None or time_column not in df.columns:
            return {
                "available": False,
                "reason": "No date / time column found, so temporal "
                "drift cannot be assessed (static snapshot).",
            }
        order = pd.to_datetime(df[time_column], errors="coerce")
        if order.notna().mean() < 0.5:
            order = pd.to_numeric(df[time_column], errors="coerce")
        if order.notna().mean() < 0.5:
            return {
                "available": False,
                "reason": f"'{time_column}' is not parseable as a "
                "date/number; temporal drift not assessed.",
            }
        # THE TIME AXIS ITSELF, above the per-attribute loop, because every
        # comparison below is "earliest cohort against latest cohort" and a time
        # column with NO VARIATION has neither. BGL6 F13, 2026-09-29, measured on
        # 60 rows all carrying app_date '2024-01-01', gender 21 'm' / 9 'f' / 9
        # 'm' / 21 'f': available True, severity 'critical', compositionPSI
        # 0.4621, plain "Across app_date, the gender mix shifted materially (PSI
        # 0.46) between the earliest and latest applicants", and ZERO warnings.
        # There are no earliest and latest applicants: np.argsort on a constant
        # array falls back to file order, so the two compared cohorts were just
        # rows 1-30 and 31-60 of the input and the CRITICAL finding was a fact
        # about the row order of a file. A constant numeric time column did the
        # same. This function's own available=False branch above already exists to
        # refuse a static snapshot ("No date / time column found ... static
        # snapshot"); a single-valued time column IS a static snapshot, and it
        # never checked.
        #
        # np.unique, not `std() == 0` and not an equality against an accumulated
        # statistic: np.var of a constant array is exactly 0.0 only at some n, so
        # a variance test passes for a round-numbered fixture and fails on real
        # data. Counting distinct values cannot do that.
        distinct = np.unique(order.dropna().to_numpy())
        # EXACTLY one, not `< 2`: zero distinct readable values is reachable only
        # from a 0-row frame (a non-empty column with nothing readable in it is
        # already refused by the parseability floor above), and that frame has a
        # better reason waiting for it at the end of this function, naming the
        # attribute that could not be observed. `< 2` swallowed it and two
        # existing tests were right to go red.
        if len(distinct) == 1:
            # Report the RAW value, not the coerced one: pd.to_datetime turns the
            # integer 7 into 1970-01-01T00:00:00.000000007, and a reader looking
            # for '7' in their file would not recognise it.
            raw_distinct = pd.unique(df[time_column].dropna())
            reason = (
                f"'{time_column}' holds "
                + (
                    f"a single distinct value ({raw_distinct[0]!r}) across all {len(df)} row(s)"
                    if len(raw_distinct) == 1
                    else f"{len(raw_distinct)} raw value(s) over {len(df)} row(s) that all "
                    "sort to one and the same instant"
                )
                + ", so there is no earliest and no latest period to compare: a "
                "single-valued time column IS the static snapshot this screen refuses. "
                "Splitting on it would order the rows by their position in the file, not "
                "by time. This is a could-not-check: nothing was compared over time, so "
                "nothing is clean."
            )
            warnings.warn(f"detect_temporal_drift: {reason}", UserWarning, stacklevel=2)
            return {
                "available": False,
                "reason": reason,
                "timeColumn": time_column,
                "drift": [],
                "nDistinctTimeValues": int(len(distinct)),
            }
        # ROWS WITH A READABLE TIME ONLY. `order.fillna(order.median())` IMPUTED
        # every unreadable timestamp into the middle of the ordering, which put
        # rows carrying no time at all into a chronological cohort. BGL6 F13,
        # 2026-09-29, measured on 31 readable dates (all gender 'm') beside 29
        # rows whose app_date is the string 'not a date' (all gender 'f'): the
        # 0.5 parseability floor above passes at 0.517, the 29 imputed rows land
        # in the middle bins, and a COMPLETE m -> f flip was reported as severity
        # 'pass', "the gender mix is stable (PSI 0.00)", with no warning and no
        # key counting the rows whose timestamp could not be read. A row with no
        # readable timestamp belongs to no period; it is now left out of the
        # ordering and counted where a reader looks.
        readable = np.flatnonzero(order.notna().to_numpy())
        n_unreadable = int(len(df) - len(readable))
        if n_unreadable:
            warnings.warn(
                f"detect_temporal_drift: {n_unreadable} of {len(df)} row(s) hold no "
                f"readable value in '{time_column}', so they belong to no period and are "
                f"NOT in any compared cohort; the drift below is measured on the "
                f"{len(readable)} row(s) whose time could be read. Their exclusion is a "
                f"could-not-check, not evidence that they match the rows that were "
                f"compared. See nRowsWithUnreadableTime.",
                UserWarning,
                stacklevel=2,
            )
        _t = order.to_numpy()
        idx = readable[np.argsort(_t[readable])]
        # THE COHORTS ARE A FACT ABOUT TIME, NOT ABOUT FILE ROW ORDER.
        #
        # BGL7 B4-rep-pre-w2, 2026-09-29. The guard above refuses a time column
        # with EXACTLY ONE distinct value, and one distinct value more reopened the
        # whole defect, because `np.array_split(idx, ...)` splits by POSITION and
        # `np.argsort` is stable, so a boundary falling inside a block of rows tied
        # on the same instant was decided by where those rows sit in the FILE.
        # Measured before this change, 60 rows, app_date ['2024-01-01'] * 59 +
        # ['2024-06-01'], gender 21 'm' / 9 'f' / 9 'm' / 21 'f':
        #   as given                  -> available True, severity 'critical',
        #                                compositionPSI 0.4621, ZERO warnings,
        #                                "the gender mix shifted materially (PSI
        #                                0.46) between the earliest and latest
        #                                applicants"
        #   sample(random_state=0)    -> PSI 0.0178  'pass'
        #   sample(random_state=1)    -> PSI 0.1622  'warn'
        #   sample(random_state=2)    -> PSI 0.0715  'pass'
        #   sample(random_state=3)    -> PSI 0.0     'pass'
        # The same 60 rows, merely reordered, moved the published verdict across
        # three severities, and 0.4621 is the identical fabricated number the
        # single-value guard above records as ITS before-state. There are no
        # "earliest and latest applicants" among 59 rows stamped with one instant.
        #
        # The periods are now cut over the DISTINCT INSTANTS and the cohorts are
        # read back in ROW space by comparing times, so every row sharing an
        # instant lands in the same cohort and no instant can appear in both the
        # earliest and the latest one. Two consequences, both wanted:
        #   * the verdict is invariant under row order, because cohort membership
        #     is decided by a row's timestamp and by nothing else;
        #   * when one instant dominates, the earliest and latest cohorts come out
        #     as the genuinely different instants they are (here 59 rows against
        #     1), and the existing _MIN_COHORT_RECORDED floor below refuses that
        #     as too thin to read a mix from, naming both counts. A could-not-check
        #     with the numbers on it, not a fabricated 'critical'.
        # Number of bins is capped at the number of distinct instants so a cut can
        # never produce an empty latest cohort out of a frame that has readable
        # time in it; len(distinct) == 0 (a 0-row frame) is left to fall through to
        # the reason at the end of this function, exactly as the block above says.
        _n_bins = max(2, n_periods)
        if len(distinct) >= 2:
            _n_bins = min(_n_bins, int(len(distinct)))
        _instant_cuts = np.array_split(np.arange(len(distinct)), _n_bins)
        _empty = np.empty(0, dtype=idx.dtype)
        _sorted_t = _t[idx]
        first = (
            idx[_sorted_t <= distinct[_instant_cuts[0][-1]]] if len(_instant_cuts[0]) else _empty
        )
        last = (
            idx[_sorted_t >= distinct[_instant_cuts[-1][0]]] if len(_instant_cuts[-1]) else _empty
        )
        pred = None
        # THE PREDICTION AXIS HAS ITS OWN MISSINGNESS, AND ITS OWN DTYPE TRAP.
        # BGL-F5 2026-09-30. The whole GROUP axis below is guarded three ways
        # (recorded-values-only distributions, attributesWithNoObservation on an
        # `or` test, and the _MIN_COHORT_RECORDED floor per cohort) and this
        # sibling axis had no missingness handling at all. Two live doors, both
        # measured on 200 rows over two instants with gender 50/50 in both
        # cohorts, so the composition PSI is 0.00 and the selection rate is the
        # only thing being reported:
        #
        # (1) IMPUTATION. `num.fillna(thr - 1)` put every unreadable prediction
        #     BELOW the threshold, i.e. counted it as a rejection, and the
        #     non-numeric arm did the same through `low.isin({...})`, where
        #     anything outside the truthy whitelist (an absent value included)
        #     became 0. With the decision recorded in the earliest cohort at 0.50
        #     and NEVER RECORDED in the latest: selectionRateFirst 0.5,
        #     selectionRateLast 0.0, selectionRateShift -0.5, severity ESCALATED
        #     to 'warn', "the selection rate moved -50 points", warnings [] and no
        #     key anywhere counting the 100 rows whose outcome could not be read.
        #     That 0.0 was 100 imputed negatives. Byte-for-byte the same shape as
        #     a genuine 0.50 -> 0.90 swing, so a reader could not tell a measured
        #     shift from an imputed one.
        #
        # (2) A TEXT-TYPED SCORE COLUMN, which is what an ordinary CSV or database
        #     read gives you, and `prediction` is documented as "binary 0/1 or
        #     scores" with no dtype named anywhere. `is_numeric_dtype` is a
        #     ROUTING guard: it decides which arithmetic runs, and it coerces
        #     nothing. Scores spelled '0.90' / '0.05' therefore fell to the token
        #     arm, where no score matches the truthy whitelist, so EVERY row
        #     counted as a rejection. Measured on the identical values, once as
        #     str and once as float64: str -> selectionRateFirst 0.0,
        #     selectionRateLast 0.0, selectionRateShift 0.0, severity 'pass', "the
        #     selection rate moved +0 points", ZERO warnings; float64 -> 0.5, 0.9,
        #     +0.4, severity 'warn'. A 40-point real swing read as perfect
        #     stability because of the column's dtype.
        #
        # READABILITY IS DECIDED ABOVE THE BRANCH, because both arms publish the
        # same selectionRate* keys and an imputation in either one lands in them.
        pred_observed: Optional[np.ndarray] = None
        n_pred_unreadable = 0
        pred_kind = ""
        if prediction and prediction in df.columns:
            ps = df[prediction]
            num = pd.to_numeric(ps, errors="coerce")
            low = ps.astype("string").str.strip().str.lower()
            recognised_token = low.isin(_DECISION_TRUTHY | _DECISION_FALSY)
            numeric_path = (
                ps.dtype == bool
                or pd.api.types.is_numeric_dtype(ps)
                # A text column carrying numbers and NO decision token is a score
                # column that happens to be typed as text; read it as numbers
                # rather than as a wall of rejections.
                or (bool(num.notna().any()) and not bool(recognised_token.any()))
            )
            if numeric_path:
                thr = float(np.nanmedian(num)) if num.notna().any() else 0.5
                # The imputation is GONE, not merely disclosed: an unreadable
                # prediction is excluded from the mean below, so nothing stands in
                # for it. `.fillna(thr - 1)` survives only as a way to keep the
                # comparison from raising on NaN; those positions are masked out.
                pred_observed = num.notna().to_numpy()
                pred = (
                    (num.fillna(thr - 1) >= (thr if set(num.dropna().unique()) - {0, 1} else 0.5))
                    .astype(int)
                    .to_numpy()
                )
                pred_kind = "number"
            else:
                pred = low.isin(_DECISION_TRUTHY).astype(int).to_numpy()
                # A value that is neither a truthy NOR a falsy decision token is
                # not a rejection, it is unreadable. Absence reaches here through
                # all six of its doors (None, float nan, pd.NA, pd.NaT, the blank
                # string and the literal string 'none'): `astype("string")` maps
                # the first four to <NA>, which is in neither set, and 'none' is
                # in _DECISION_FALSY, where a reader would put it.
                pred_observed = (recognised_token & low.notna()).to_numpy()
                pred_kind = "decision token"
            n_pred_unreadable = int((~pred_observed).sum())
            if n_pred_unreadable:
                warnings.warn(
                    f"detect_temporal_drift: {n_pred_unreadable} of {len(df)} row(s) hold no "
                    f"readable {pred_kind} in the prediction column '{prediction}', so they "
                    f"are in NO selection rate below; the rates are measured on the rows "
                    f"whose outcome could be read. Their exclusion is a could-not-check, "
                    f"not evidence that they were rejected. See "
                    f"nRowsWithUnreadablePrediction.",
                    UserWarning,
                    stacklevel=2,
                )

        drift = []
        skipped: List[str] = []
        absent: List[str] = []
        unobserved: List[str] = []
        # Attribute -> the recorded counts in each compared cohort and the floor they
        # missed. A dict rather than a list of names, because the counts are the whole
        # point: "recorded in 1 of 500 and 1 of 500" is the finding.
        too_few: Dict[str, Dict[str, int]] = {}
        for attr in protected_attributes:
            if attr not in df.columns:
                absent.append(str(attr))
                continue
            g = df[attr].astype("string").fillna("missing").to_numpy()
            # fillna("missing") INVENTS a category, and a column holding no
            # observation at all therefore becomes one perfectly stable group.
            # BGL5 2026-09-27, measured on a 100-row frame whose gender column
            # was entirely None: {'available': True, 'severity': 'pass',
            # 'drift': [{'attribute': 'gender', 'compositionPSI': 0.0,
            # 'severity': 'pass', 'plain': 'Across application_date, the gender
            # mix is stable (PSI 0.00)'}], 'summary': 'No material temporal
            # drift in the assessed attributes over application_date' and NO
            # warning. Identical for [np.nan]*100 and for a 0-row frame. It now
            # reports available False with a reason naming gender, plus a
            # warning, and the entry is not emitted at all.
            #
            # The bound is "at least one real observation in EACH of the two
            # cohorts being compared", because that is what a mix comparison
            # needs; it is deliberately NOT "any observation anywhere", which
            # would still let an attribute observed only in the middle periods
            # answer for the first and the last. A column observed on one side
            # only is left alone here: it already lands on 'not_assessed'
            # through the degenerate-PSI bound below, with a reason that says
            # the mix did change, and that is more than this guard could say.
            #
            # THAT LAST SENTENCE WAS WRONG, and the test `and` encoded it. BGL6 F12,
            # 2026-09-29, measured on 1000 rows whose gender is recorded in 5 of the
            # earliest 500 and in NONE of the latest 500: compositionPSI 0.0922,
            # severity 'pass', plain "the gender mix is stable (PSI 0.09)", summary
            # "No material temporal drift", attributesWithNoObservation [] and zero
            # warnings. It did not land on 'not_assessed' at all, because the
            # invented "missing" category is 99 percent of both cohorts and moves
            # hardly at all between them. The guard is now `or`: a cohort with no
            # recorded value has no mix, and the existing wording of this list
            # ("carry no recorded value in either the earliest or the latest
            # period") was already the `or` reading.
            observed = df[attr].notna().to_numpy()
            n_first_obs = int(observed[first].sum()) if len(first) else 0
            n_last_obs = int(observed[last].sum()) if len(last) else 0
            if n_first_obs == 0 or n_last_obs == 0:
                unobserved.append(str(attr))
                continue
            # And enough of them to be a mix rather than a coin flip, named with
            # both counts so a reader can see how thin the record was.
            if n_first_obs < _MIN_COHORT_RECORDED or n_last_obs < _MIN_COHORT_RECORDED:
                too_few[str(attr)] = {
                    "earliest": n_first_obs,
                    "latest": n_last_obs,
                    "required": _MIN_COHORT_RECORDED,
                }
                continue
            # RECORDED VALUES ONLY, on both sides. `g` carries the invented "missing"
            # category so the array has no NaN in it, and including that category in
            # the compared distributions let it outvote every real observation: the
            # 998-of-1000-unrecorded frame above published a PSI of 0.03 for a
            # recorded mix that had flipped from all 'm' to all 'f'. An absent value
            # is not a group, so it is not one of the categories either.
            g_first = g[first][observed[first]]
            g_last = g[last][observed[last]]
            cats = sorted(set(g_first) | set(g_last))
            # Identifier-grade for drift (one tiny group per value, e.g. raw
            # birth dates): PSI is degenerate here, not a fairness signal.
            # Skip rather than emit a false "critical".
            if len(cats) > _PSI_MAX_CATS:
                skipped.append(attr)
                continue
            ef = np.array([(g_first == c).mean() for c in cats])
            al = np.array([(g_last == c).mean() for c in cats])
            psi = _psi(ef, al)
            entry: Dict[str, Any] = {
                "attribute": attr,
                "compositionPSI": round(psi, 4),
                "severity": _psi_severity(psi),
            }
            if entry["severity"] == NOT_ASSESSED:
                # The count of categories that the early and late cohort do NOT
                # share is the readable half of a degenerate PSI, so say it
                # rather than leaving a reader with a number out of its range.
                shared = len(set(g_first) & set(g_last))
                entry["notAssessedReason"] = (
                    f"the composition PSI came out at {psi:.2f}, above the {_PSI_DEGENERATE:g} "
                    f"bound where the statistic stops being a drift measure: the earliest and "
                    f"latest period share only {shared} of {len(cats)} observed "
                    f"{attr} value(s). The mix did change; how much is NOT graded here."
                )
            if pred is not None and pred_observed is not None:
                # READABLE PREDICTIONS ONLY, on both sides, and enough of them to
                # be a rate. Three states where a reader looks: the two rates and
                # the shift are floats when they were measured and None when they
                # could not be, selectionRateStatus says which, and the counts are
                # on the entry so the reader can see how thin the record was.
                pred_ok_first = pred_observed[first] if len(first) else np.zeros(0, dtype=bool)
                pred_ok_last = pred_observed[last] if len(last) else np.zeros(0, dtype=bool)
                n_pred_first = int(pred_ok_first.sum())
                n_pred_last = int(pred_ok_last.sum())
                entry["nPredictionsReadFirst"] = n_pred_first
                entry["nPredictionsReadLast"] = n_pred_last
                if n_pred_first < _MIN_COHORT_PREDICTIONS or n_pred_last < _MIN_COHORT_PREDICTIONS:
                    entry["selectionRateFirst"] = None
                    entry["selectionRateLast"] = None
                    entry["selectionRateShift"] = None
                    entry["selectionRateStatus"] = "could_not_measure"
                    entry["selectionRateNotMeasuredReason"] = (
                        f"'{prediction}' holds a readable outcome in {n_pred_first} row(s) of "
                        f"the earliest cohort and {n_pred_last} of the latest, and a "
                        f"selection rate needs at least {_MIN_COHORT_PREDICTIONS} in EACH, so "
                        f"no rate and no shift could be taken. This is a could-not-check: it "
                        f"is NOT a rate of 0.0 and NOT evidence that the unread rows were "
                        f"rejected."
                    )
                    warnings.warn(
                        f"detect_temporal_drift: {entry['selectionRateNotMeasuredReason']} "
                        f"The selectionRate* fields for '{attr}' are None and the severity "
                        f"below was NOT escalated from them. See "
                        f"entry['selectionRateNotMeasuredReason'].",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    sr_first = float(pred[first][pred_ok_first].mean())
                    sr_last = float(pred[last][pred_ok_last].mean())
                    entry["selectionRateFirst"] = round(sr_first, 4)
                    entry["selectionRateLast"] = round(sr_last, 4)
                    entry["selectionRateShift"] = round(sr_last - sr_first, 4)
                    entry["selectionRateStatus"] = "measured"
                    # A selection-rate swing is measured on the prediction column,
                    # not on the PSI, so it still escalates an ungraded composition.
                    if abs(sr_last - sr_first) >= 0.10 and entry["severity"] in (
                        "pass",
                        NOT_ASSESSED,
                    ):
                        entry["severity"] = "warn"
            entry["plain"] = (
                f"Across {time_column}, the {attr} mix "
                + (
                    f"changed so completely that the PSI ({psi:.2f}) is outside the range in "
                    "which it grades drift, so the size of the shift is NOT assessed"
                    if entry["severity"] == NOT_ASSESSED
                    else f"shifted materially (PSI {psi:.2f}) between the earliest and latest applicants"
                    if psi >= 0.10
                    else f"is stable (PSI {psi:.2f})"
                )
                + (
                    "."
                    if pred is None
                    # `entry.get('selectionRateShift', 0)` would NOT fire here: the
                    # key is PRESENT holding None whenever the rate could not be
                    # measured, so the default never applies and the multiplication
                    # raises. The test is on the VALUE.
                    else (
                        f"; the selection rate could NOT be measured "
                        f"({entry['nPredictionsReadFirst']} readable outcome(s) in the "
                        f"earliest cohort, {entry['nPredictionsReadLast']} in the latest), "
                        f"so no shift is reported: that is a could-not-check, not a rate "
                        f"of zero."
                        if entry.get("selectionRateShift") is None
                        else (
                            f"; the selection rate moved "
                            f"{entry['selectionRateShift'] * 100:+.0f} points"
                            + (
                                f", measured on the {entry['nPredictionsReadFirst']} and "
                                f"{entry['nPredictionsReadLast']} row(s) whose outcome could "
                                f"be read (of {len(first)} and {len(last)} in the two "
                                f"cohorts)."
                                if n_pred_unreadable
                                else "."
                            )
                        )
                    )
                )
            )
            drift.append(entry)

        # THREE STATES, NEVER TWO. An empty `drift` list means NOT ONE attribute
        # was assessed, and the clean verdict below ("No material temporal drift
        # in the assessed attributes") was published for it, because [] was doing
        # double duty for "assessed and stable" and "nothing was assessed".
        # Measured 2026-09-27, all three routes into it: a protected attribute
        # that is not a column of the frame, an empty protected_attributes list,
        # and an attribute skipped as identifier-grade each returned
        # {'available': True, 'drift': [], 'severity': 'pass', 'summary': 'No
        # material temporal drift in the assessed attributes over
        # application_date'}. `available: False` with a reason is the state this
        # function already uses for the missing time column, so it is the state
        # this belongs in as well.
        if not drift:
            detail = []
            if absent:
                detail.append(f"{absent} are not column(s) of the frame")
            if skipped:
                detail.append(
                    f"{skipped} hold more than {_PSI_MAX_CATS} distinct values, which is "
                    "identifier-grade for PSI"
                )
            if unobserved:
                detail.append(
                    f"{unobserved} carry no recorded value in either the earliest or the "
                    "latest period, so there was no group mix to compare (an absent value "
                    "is not a group)"
                )
            if too_few:
                detail.append(
                    f"{ {a: (c['earliest'], c['latest']) for a, c in too_few.items()} } hold "
                    f"fewer than {_MIN_COHORT_RECORDED} recorded value(s) in one of the two "
                    "compared periods (earliest, latest), which is too thin to read a group "
                    "mix from"
                )
            if not protected_attributes:
                detail.append("no protected attribute was named")
            reason = (
                f"no protected attribute could be assessed for temporal drift over "
                f"'{time_column}': " + "; ".join(detail or ["none qualified"]) + ". This is a "
                "could-not-check: nothing was compared, so nothing is clean."
            )
            warnings.warn(f"detect_temporal_drift: {reason}", UserWarning, stacklevel=2)
            return {
                "available": False,
                "reason": reason,
                "timeColumn": time_column,
                "drift": [],
                "attributesNotInFrame": absent,
                "skippedHighCardinality": skipped,
                "attributesWithNoObservation": unobserved,
                "attributesWithTooFewObservations": too_few,
                "nRowsWithUnreadableTime": n_unreadable,
                "nRowsWithUnreadablePrediction": n_pred_unreadable,
            }

        if absent or skipped or unobserved or too_few:
            # Partial coverage. The summary honestly says "the assessed
            # attributes", but a reader has no way to know which those were
            # unless the ones left out are named where they will be seen.
            warnings.warn(
                f"detect_temporal_drift assessed {len(drift)} of "
                f"{len(protected_attributes)} named protected attribute(s) over "
                f"'{time_column}'; not assessed: "
                + "; ".join(
                    part
                    for part in (
                        f"{absent} absent from the frame" if absent else "",
                        f"{skipped} identifier-grade for PSI" if skipped else "",
                        (
                            f"{unobserved} carry no recorded value in either the earliest "
                            "or the latest period"
                            if unobserved
                            else ""
                        ),
                        (
                            f"{ {a: (c['earliest'], c['latest']) for a, c in too_few.items()} } "
                            f"hold fewer than {_MIN_COHORT_RECORDED} recorded value(s) "
                            "(earliest, latest) in one of the two compared periods"
                            if too_few
                            else ""
                        ),
                    )
                    if part
                )
                + ". Their absence from 'drift' is not evidence that they are stable.",
                UserWarning,
                stacklevel=2,
            )

        drift.sort(key=lambda d: d["compositionPSI"], reverse=True)
        # Overall severity must be the WORST severity across attributes, and
        # the summary must name that attribute. Taking drift[0] (highest PSI)
        # let a degenerate high-PSI 'pass' or a low-PSI attribute mask a
        # genuine 'critical'/'warn' elsewhere. An ungraded composition ranks
        # ABOVE 'pass' for the same reason: it must not be reported as clean,
        # and below 'warn' so it cannot mask a measured finding.
        _sev_rank = {"pass": 0, NOT_ASSESSED: 1, "warn": 2, "critical": 3}
        worst = (
            max(drift, key=lambda d: (_sev_rank.get(d["severity"], 0), d["compositionPSI"]))
            if drift
            else None
        )
        sev = worst["severity"] if worst else "pass"
        # `worst` is None only when `drift` is empty, and then `sev` is "pass", so
        # both graded branches always have it; the explicit test says so to mypy.
        if worst is not None and sev == NOT_ASSESSED:
            summary = (
                f"Temporal drift was NOT graded for {worst['attribute']} over {time_column}: "
                f"{worst['notAssessedReason']} Read this as a could-not-check, not as an "
                "absence of drift."
            )
            warnings.warn(f"detect_temporal_drift: {summary}", UserWarning, stacklevel=2)
        elif worst is not None and sev != "pass":
            summary = (
                f"Temporal drift risk: the {worst['attribute']} population "
                f"changes over {time_column} (PSI "
                f"{worst['compositionPSI']:.2f}); a model trained on the "
                f"early period may degrade or bias against the later mix."
            )
        else:
            summary = (
                "No material temporal drift in the assessed attributes over "
                f"{time_column} (static-snapshot screen only)."
            )
        return {
            "available": True,
            "timeColumn": time_column,
            "nPeriods": int(max(2, n_periods)),
            "drift": drift,
            "attributesNotInFrame": absent,
            "skippedHighCardinality": skipped,
            # Beside the other two coverage lists, never instead of them: an
            # attribute nobody recorded a value for was not compared, and its
            # absence from 'drift' must be as readable as an absent column's.
            "attributesWithNoObservation": unobserved,
            # Beside the other three, for the same reason: an attribute whose record
            # was too thin to read a mix from was not compared, and its absence from
            # 'drift' is not evidence that it is stable. BGL6 F12, 2026-09-29.
            "attributesWithTooFewObservations": too_few,
            # The ROW axis of the same disclosure: rows whose timestamp could not
            # be read are in no cohort, so a verdict here covers len(df) minus
            # this count. BGL6 F13, 2026-09-29.
            "nRowsWithUnreadableTime": n_unreadable,
            # The SIBLING row axis: rows whose PREDICTION could not be read are in
            # no selection rate, and before BGL-F5 they were imputed to a
            # rejection and moved the published rate with nothing counting them.
            "nRowsWithUnreadablePrediction": n_pred_unreadable,
            "severity": sev,
            "summary": summary,
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"Temporal drift error: {e}"}


# Specification / learning bias  (Jacobs & Wallach, FAccT 2021)
#
# The construct ("is this a good hire?") vs the measured target ("was this
# person invited?") gap. Three tractable static-data signals:
#  (1) target leakage : a legitimate feature almost equals the outcome
#      (the model is learning the answer, not the construct);
#  (2) proxy target   : the outcome column name is a documented proxy for
#      the real construct (cost↦health need: Obermeyer 2019; arrests↦crime);
#  (3) label==prediction: the outcome IS the model's own output (the
#      "evaluation" is circular).

_PROXY_TARGET_PATTERNS = {
    "cost": "healthcare cost is a documented proxy for health need and "
    "under-serves equally-sick minority patients (Obermeyer et al. "
    "2019, Science).",
    "spend": "spend/charges proxy for need, the same Obermeyer (2019) failure mode.",
    "charges": "billed charges proxy for clinical need (Obermeyer 2019).",
    "arrest": "arrests/stops are a proxy for criminality that encodes "
    "enforcement bias, not offending.",
    "stop": "police stops proxy for criminality (enforcement bias).",
    "click": "clicks/engagement are a proxy for quality/relevance and "
    "encode popularity & exposure bias.",
    "engagement": "engagement proxy for quality (popularity bias).",
    "tenure": "tenure/retention proxy for performance encodes survivorship.",
}


def detect_specification_bias(
    df: pd.DataFrame,
    *,
    outcome: Optional[str] = None,
    prediction: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct-validity / specification-bias screen. Returns a
    finding-shaped result; never raises.

    Three states, never two. The screen runs three checks and each one needs an
    input it may not have been given: the proxy-target check needs an outcome or
    prediction column NAME, the target-leakage check needs a numeric outcome
    column, and the circular-evaluation check needs both an outcome and a
    prediction. ``available`` is False, with a ``reason``, when not one check
    could run; otherwise ``checksRun`` and ``checksNotRun`` say which did, so an
    empty ``findings`` list is readable as "the checks that ran found nothing"
    rather than as a clean bill of health for a screen that never happened.
    """
    try:
        findings: List[Dict[str, Any]] = []
        # What this run actually examined. An empty findings list used to be the
        # only output, and it was identical whether three checks ran and found
        # nothing or none of them could run at all. Measured 2026-09-27, four
        # inputs on which NOTHING was examined and every one of them returned
        # {'available': True, 'findings': [], 'severity': 'pass', 'summary': 'No
        # specification / construct-validity red flag detected on this data.'}:
        # no outcome and no prediction given; an outcome naming a column the
        # frame does not have; an empty frame; and a non-numeric outcome
        # ('yes'/'no'), where the leakage scan is skipped wholesale even though a
        # feature column encoded that same decision exactly.
        checks_run: List[str] = []
        not_run: List[str] = []
        excl = set(exclude_columns or [])
        target = outcome or prediction
        if not target:
            not_run.append(
                "proxy_target: neither an outcome nor a prediction column was named, and "
                "the check reads the outcome's NAME"
            )
        elif target not in df.columns:
            not_run.append(
                f"proxy_target: {target!r} is not a column of this frame, so no outcome "
                "name was examined"
            )
        else:
            checks_run.append("proxy_target")
        if target and target in df.columns:
            tl = target.lower()
            for pat, why in _PROXY_TARGET_PATTERNS.items():
                if pat in tl:
                    findings.append(
                        {
                            "kind": "proxy_target",
                            "severity": "warn",
                            "plain": (
                                f"The outcome '{target}' looks like a "
                                f"PROXY for the real construct: {why} "
                                "Optimising it can institutionalise that "
                                "bias (Jacobs & Wallach 2021)."
                            ),
                        }
                    )
                    break
        # target leakage: a legitimate feature ~ the outcome
        if not outcome:
            not_run.append(
                "target_leakage: no outcome column was named, and a correlation needs "
                "something to correlate the features WITH"
            )
        elif outcome not in df.columns:
            not_run.append(
                f"target_leakage: {outcome!r} is not a column of this frame, so no "
                "feature was correlated against it"
            )
        if outcome and outcome in df.columns:
            y = pd.to_numeric(df[outcome], errors="coerce")
            if y.notna().mean() > 0.5:
                yv = y.fillna(y.median())
                scanned = 0
                unreadable: List[str] = []
                too_few: List[str] = []
                for c in df.columns:
                    if c == outcome or c in excl or c == prediction:
                        continue
                    x = pd.to_numeric(df[c], errors="coerce")
                    if x.notna().mean() <= 0.5:
                        unreadable.append(str(c))
                        continue
                    xv = x.fillna(x.median())
                    if xv.std() == 0 or yv.std() == 0:
                        unreadable.append(str(c))
                        continue
                    # How many rows actually carry BOTH values. There was no
                    # minimum n on this scan at all. BGL5 2026-09-27, measured
                    # on pd.DataFrame({'approved': [1, 0], 'x': [5.0, 9.0],
                    # 'z': [2.0, 3.0]}) with outcome='approved': severity
                    # 'critical', findings [target_leakage, target_leakage],
                    # both plain sentences reading "|r|=1.00", checksRun
                    # ['proxy_target', 'target_leakage (2 feature column(s)
                    # scanned)']. Two points are perfectly collinear whatever
                    # they are, so the coefficient was arithmetic and not a
                    # measurement. It now scans nothing on that frame and names
                    # both columns with their n in checksNotRun.
                    pair = (x.notna() & y.notna()).to_numpy()
                    n_pairs = int(pair.sum())
                    if n_pairs < _MIN_COMPARABLE_ROWS:
                        too_few.append(f"{c} (n={n_pairs})")
                        continue
                    # CORRELATED OVER THE COMPARABLE ROWS ONLY, BGL6 F12, 2026-09-29.
                    # The floor above counts the rows where the feature and the outcome
                    # BOTH hold a value, and the coefficient was then taken over ALL
                    # rows with x.fillna(x.median()), so the invented values diluted the
                    # measurement the floor had just qualified. Measured on 40 rows
                    # whose 'leak' column is IDENTICAL to 'approved' wherever it is
                    # recorded (25 rows, |r| = 1.0 over them) and unrecorded on the
                    # other 15: |r| over the imputed 40 rows came out at 0.6547, below
                    # the 0.97 bound, so a PERFECT target leak published severity
                    # 'pass' while checksRun asserted "target_leakage (1 feature
                    # column(s) scanned)". The sibling circular_evaluation check in the
                    # same module was already restricted to its comparable rows; this
                    # one was not.
                    xc = x.to_numpy()[pair]
                    yc = y.to_numpy()[pair]
                    # np.unique, never `std() == 0`: a standard deviation is an
                    # accumulated statistic and is exactly 0.0 only for some values at
                    # some n, while "how many distinct values are there" is a fact
                    # about the data. A column with spread overall can be constant
                    # exactly where the outcome was recorded, and then there is no
                    # correlation to read, only a 0/0.
                    if np.unique(xc).size < 2 or np.unique(yc).size < 2:
                        unreadable.append(f"{c} (no spread over its {n_pairs} shared row(s))")
                        continue
                    scanned += 1
                    r = abs(float(np.corrcoef(xc, yc)[0, 1]))
                    if r >= 0.97:
                        findings.append(
                            {
                                "kind": "target_leakage",
                                "severity": "critical",
                                "plain": (
                                    f"Feature '{c}' is almost identical to "
                                    f"the outcome '{outcome}' (|r|={r:.2f} over the "
                                    f"{n_pairs} row(s) where both hold a value): "
                                    "the model learns the answer, not "
                                    "the construct (specification bias)."
                                ),
                            }
                        )
                if scanned:
                    checks_run.append(f"target_leakage ({scanned} feature column(s) scanned)")
                    if unreadable:
                        not_run.append(
                            f"target_leakage skipped {len(unreadable)} column(s) that are not "
                            f"numeric or have no spread: {unreadable[:8]}"
                        )
                    if too_few:
                        not_run.append(
                            f"target_leakage skipped {len(too_few)} column(s) sharing fewer "
                            f"than {_MIN_COMPARABLE_ROWS} row(s) with {outcome!r}, where a "
                            f"correlation is arithmetic rather than a measurement: "
                            f"{too_few[:8]}"
                        )
                else:
                    # The original sentence named only the not-numeric reason.
                    # It has to stay true now that a column can also be skipped
                    # for having too few comparable rows: a reader told "not
                    # numeric with any spread" about a clean numeric column
                    # would go looking for the wrong thing.
                    # Not `why`: that name is already the loop variable holding each
                    # proxy pattern's explanation string earlier in this function.
                    skip_reasons: List[str] = []
                    if unreadable:
                        skip_reasons.append(f"not numeric or no spread: {unreadable[:8]}")
                    if too_few:
                        skip_reasons.append(
                            f"fewer than {_MIN_COMPARABLE_ROWS} row(s) sharing a value with "
                            f"{outcome!r}: {too_few[:8]}"
                        )
                    not_run.append(
                        f"target_leakage: no feature column could be correlated against "
                        f"{outcome!r}, so nothing was scanned ("
                        + "; ".join(skip_reasons or ["there is no feature column at all"])
                        + ")"
                    )
            else:
                # A non-numeric outcome disables this whole check, and that used
                # to be invisible. Measured 2026-09-27 on outcome='decision'
                # holding 'yes'/'no' beside a feature column that encoded the
                # same decision as 1/0: the screen returned severity 'pass'
                # having correlated nothing at all.
                not_run.append(
                    f"target_leakage: only {y.notna().mean():.0%} of {outcome!r} reads as a "
                    "number, and the check is a Pearson correlation, so no feature was "
                    "compared against the outcome. Encode the outcome numerically to run it"
                )
        # Rows in which BOTH the outcome and the prediction hold a value. None
        # by default so the finding below cannot fire on a check that did not
        # run: the guard sits ABOVE the comparison, not inside it.
        comparable = None
        n_comparable = 0
        if not (outcome and prediction):
            not_run.append(
                "circular_evaluation: it compares the outcome column against the prediction "
                "column and one of the two was not named"
            )
        elif not (outcome in df.columns and prediction in df.columns):
            not_run.append(
                f"circular_evaluation: {[c for c in (outcome, prediction) if c not in df.columns]} "
                "is not a column of this frame"
            )
        else:
            # BGL5 2026-09-27, measured twice, both CRITICAL and both fabricated:
            # (1) an EMPTY frame carrying approved/pred/x with outcome='approved'
            # and prediction='pred' returned {'available': True, 'severity':
            # 'critical', 'findings': [{'kind': 'circular_evaluation', ...}],
            # 'checksRun': ['proxy_target', 'circular_evaluation']} and no
            # warning, because Series([]).equals(Series([])) is True and ZERO
            # rows had been compared; (2) 50 rows with the outcome and the
            # prediction both entirely NaN produced the same CRITICAL, because
            # astype(str) maps every missing value to the string 'nan', so two
            # columns nobody recorded a value in compare equal. Both now report
            # the check as NOT RUN, naming the number of comparable rows, and
            # neither publishes a finding.
            comparable = (df[outcome].notna() & df[prediction].notna()).to_numpy()
            n_comparable = int(comparable.sum())
            if n_comparable < _MIN_COMPARABLE_ROWS:
                not_run.append(
                    f"circular_evaluation: only {n_comparable} of {len(df)} row(s) hold a "
                    f"value in both {outcome!r} and {prediction!r}, fewer than the "
                    f"{_MIN_COMPARABLE_ROWS} this check needs, so the two columns were not "
                    "compared. Two unrecorded values would match as the string 'nan', and a "
                    "handful of matching labels is coincidence rather than a shared column"
                )
            else:
                checks_run.append("circular_evaluation")
        if (
            n_comparable >= _MIN_COMPARABLE_ROWS
            and comparable is not None
            and outcome is not None
            and prediction is not None
            # Compared over the comparable rows ONLY, so an unrecorded value can
            # neither manufacture a match nor hide one.
            and np.array_equal(
                df.loc[comparable, outcome].astype(str).to_numpy(),
                df.loc[comparable, prediction].astype(str).to_numpy(),
            )
        ):
            findings.append(
                {
                    "kind": "circular_evaluation",
                    "severity": "critical",
                    "plain": (
                        "The outcome column equals the model's prediction "
                        "column: the system is being evaluated against "
                        "its own output (no real ground truth)."
                        # The sentence says "equals", so it has to say over
                        # WHAT when the two columns were not comparable
                        # everywhere. The rows where either value is missing
                        # were not compared and are not covered by this finding.
                        + (
                            ""
                            if n_comparable == len(df)
                            else f" (compared over the {n_comparable} of {len(df)} row(s) "
                            "holding a value in both columns; the rest were not compared.)"
                        )
                    ),
                }
            )
        if not checks_run:
            reason = (
                "no specification / construct-validity check could run on this input, so "
                "nothing was examined and nothing is clean: " + "; ".join(not_run) + "."
            )
            warnings.warn(f"detect_specification_bias: {reason}", UserWarning, stacklevel=2)
            return {
                "available": False,
                "reason": reason,
                "findings": [],
                "checksRun": [],
                "checksNotRun": not_run,
            }

        sev = (
            "critical"
            if any(f["severity"] == "critical" for f in findings)
            else "warn"
            if findings
            else "pass"
        )
        if findings:
            summary = findings[0]["plain"]
        elif not_run:
            summary = (
                f"No specification / construct-validity red flag in the {len(checks_run)} "
                f"check(s) that ran ({', '.join(checks_run)}); {len(not_run)} could not run "
                f"on this input: {'; '.join(not_run)}."
            )
            warnings.warn(
                f"detect_specification_bias ran {len(checks_run)} of its checks and could "
                f"not run {len(not_run)}: {'; '.join(not_run)}. The 'pass' covers only what "
                "ran.",
                UserWarning,
                stacklevel=2,
            )
        else:
            summary = "No specification / construct-validity red flag detected on this data."
        return {
            "available": True,
            "findings": findings,
            "severity": sev,
            "checksRun": checks_run,
            "checksNotRun": not_run,
            "summary": summary,
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"Specification screen: {e}"}
