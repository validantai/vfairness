"""
Classification fairness metrics for vfairness.

This module provides fairness metrics for binary classification tasks.

Library Comparisons:
    - AI Fairness 360 (AIF360): BinaryLabelDatasetMetric class
    - Microsoft Fairlearn: fairlearn.metrics module
    - Google What-If Tool: Fairness features in UI
    - Aequitas: Group metrics in bias audit

Statistical Validation:
    All metrics support confidence interval computation via the
    `with_ci=True` parameter or the `*_with_ci` function variants.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ...exceptions import ConfigurationError, InvalidDataError
from ._grouping import GroupManager, compute_max_difference
from ._statistics import (
    RECOMMENDED_BOOTSTRAP_SAMPLES,
    SMALL_SAMPLE_THRESHOLD,
    StatisticalResult,
    bayesian_proportion_ci,
    bootstrap_over_index,
    cohens_h,
    compute_metric_with_ci,
    odds_ratio,
    risk_ratio,
    wilson_score_interval,
)
from ._validation import (
    VALID_MISSING_STRATEGIES,
    ArrayLike,
    MissingStrategy,
    _check_option,
    validate_inputs,
)


def _warn_dropped_groups(gm: GroupManager) -> None:
    """Warn when SOME groups fall below min_group_size while others survive.

    GroupManager itself only warns when ALL groups fail the size gate, so a
    partial drop was silent and a metric could quietly ignore a small
    protected group. Mirrors the existing all-groups-dropped warning style.

    The wording differs by how many groups survive: with two or more the metric
    really is computed over them, with exactly one nothing is computed at all
    and the metric is NaN (not assessable).

    Two things the warning must carry, both added after the 2026-09-10 headline
    repro (2302 rows, a 38-level tail of 1102 applicants, 47.9 percent of the
    data, denied outright; every default metric read fair):

    * HOW MUCH DATA left the comparison. "Excluding 38 group(s)" reads like a
      rounding detail; "1102 of 2302 rows (47.9 percent)" does not, and it is
      the number that separates a five-person tail from half the dataset.
    * WHO the warning belongs to. It is raised with ``stacklevel=3``, i.e. at
      the line in the CALLER that asked for the metric, exactly as
      ``compute_max_difference`` already does. Without it every one of the 17
      call sites in this file reported the same file and line, so Python's
      default "once per location" filter collapsed a whole report's worth of
      drops into ONE stderr line pointing at this module rather than at the
      user's code. That single line was the only signal a default run gave.

    The multi-survivor branch also states the consequence for the EXCLUDED
    groups, not only for the metric: the value covers the survivors, and the
    dropped groups are could-not-check. Saying only "computed over the remaining
    groups" leaves a reader to supply the missing half themselves, and the whole
    finding is that they supply "so the rest must be fine".
    """
    # DO NOT ASK A QUESTION WHOSE ANSWER WOULD NOT BE A FINDING HERE. This helper
    # has exactly one use for get_invalid_groups: the NAMES of the groups that
    # were dropped. An empty list therefore means "none were dropped", and when
    # the threshold is 1 or less that reading is TRUE, because every group is
    # built from a level that actually occurs so no group can hold fewer than one
    # row. The caller has asked for no filtering on purpose (rerank.py:318, "so
    # every present group is included in this intervention view"), and
    # get_invalid_groups warns, correctly, that its empty answer is VACUOUS for a
    # caller that would read it as a clean bill of group sizes. Measured
    # 2026-09-27: exposure_parity_rerank(10 items, 5 in 'A' and 5 in 'B'), a
    # healthy rerank with nothing dropped anywhere, raised that vacuity warning
    # FOUR times, once per get_ranking_group_metrics call, through
    # ranking.py:785 and ranking.py:308 into this line. After: zero warnings
    # there, and the vacuity warning stays exactly as it is in _grouping.py for
    # the callers that do read the empty list as a clean bill. A real drop is
    # untouched, because a threshold of 2 or more can flag: at min_group_size=2 a
    # one-row group is still named here (see the control in
    # tests/test_bgl5_evaluation_2.py).
    if gm.min_group_size <= 1:
        return

    dropped = gm.get_invalid_groups()
    survivors = gm.get_valid_groups(warn_if_empty=False)
    if dropped and survivors:
        sizes = gm.get_group_sizes()
        dropped_sizes = {name: sizes[name] for name in dropped}
        n_dropped_rows = sum(dropped_sizes.values())
        n_rows = sum(sizes.values())
        share = (100.0 * n_dropped_rows / n_rows) if n_rows else 0.0
        if len(survivors) == 1:
            # CRITICAL: with a single survivor there is no PAIR left, so no
            # between-group metric is computed AT ALL. The old wording ("computed
            # over the remaining groups only") described a measurement that never
            # happens here, which is how a vacuous run read as a deliberate,
            # narrower assessment. The value is NaN and the run is not assessable.
            consequence = (
                f"Only 1 group ({survivors[0]!r}) meets the size gate, so there is "
                f"no pair to compare: the metric is NOT computed, it returns NaN "
                f"and the run is not assessable."
            )
        else:
            consequence = (
                "The metric is computed over the remaining groups only, so it "
                "measures nothing about the excluded group(s): they are could "
                "not check, not a measured pass."
            )
        warnings.warn(
            f"Excluding {len(dropped)} group(s) below "
            f"min_group_size={gm.min_group_size}: {dropped_sizes}. "
            f"That leaves {n_dropped_rows} of {n_rows} rows ({share:.1f} percent) "
            f"out of this metric. "
            f"{consequence} "
            f"To include them, lower min_group_size.",
            UserWarning,
            # Attribute the warning to the caller that asked for the metric, not
            # to this helper: see the docstring, one shared location made Python
            # print it once per RUN instead of once per metric.
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# CONVENTION (fail closed): a between-group metric that could not actually
# COMPARE two groups returns NaN, never the "perfect" sentinel (0.0 for a
# difference metric, 1.0 for a ratio metric). Fewer than two groups surviving
# min_group_size, or fewer than two groups having a DEFINED per-group rate,
# means the comparison NEVER HAPPENED. A 0.0 / 1.0 there is indistinguishable
# downstream from a measured perfect parity, so it certifies as FAIR exactly
# where nothing was measured. NaN routes through the report's NOT_ASSESSABLE
# path (insufficient evidence), which is the honest answer.
#
# The incident this convention exists for: with the default min_group_size=30 a
# 25-person minority group that was NEVER selected, against a majority selected
# at 56 percent, reported demographic_parity_difference 0.0 and
# demographic_parity_ratio 1.0, the exact INVERSE of the truth (0.56 and 0.0),
# and the report certified fairness_score 1.0 on it. A warning was emitted, but
# the VALUE was a false certificate and every downstream consumer inherited it.
#
# Sites below that say "unmeasurable -> NaN" mean exactly this. Do not "restore"
# a 0.0 / 1.0 return at any of them.
# ---------------------------------------------------------------------------


def demographic_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute demographic parity difference (statistical parity).

    Measures the maximum absolute difference in positive prediction rates
    across groups. A value of 0 indicates perfect demographic parity.

    Library Comparisons:
        AIF360: statistical_parity_difference()
        Fairlearn: demographic_parity_difference()
        Aequitas: ppr_disparity (as ratio)

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in positive prediction rates; NaN
        (insufficient evidence) when fewer than two groups meet
        ``min_group_size``, because no between-group comparison happened there.

    Example:
        >>> dp = demographic_parity_difference(y_true, y_pred, gender)
        >>> print(f"DP Difference: {dp:.3f}")

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

    Ledger row: demographic_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # Compute positive rate per group
    positive_rates = gm.compute_group_rate(y_pred == 1)

    # Fewer than two qualifying groups: there is no pair to compare, so the
    # comparison is unmeasurable -> NaN (see the CONVENTION note above). A 0.0
    # here read as PERFECT PARITY on data where a dropped minority group was
    # never selected at all.
    if len(positive_rates) < 2:
        return float("nan")

    max_diff, _, _ = compute_max_difference(positive_rates)
    return max_diff


def demographic_parity_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute demographic parity ratio (disparate impact ratio).

    Measures the minimum ratio of positive prediction rates across groups.
    A value of 1 indicates perfect demographic parity. Values < 0.8 often
    indicate potential disparate impact (80% rule).

    Library Comparisons:
        AIF360: disparate_impact()
        Fairlearn: demographic_parity_ratio()
        Aequitas: ppr_disparity

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Minimum ratio of positive prediction rates (min/max); NaN (insufficient
        evidence) when fewer than two groups have a defined positive rate,
        because no between-group comparison happened there, and ALSO when every
        group's selection rate is 0 (the model selects nobody), because the
        ratio is then 0/0. ``demographic_parity_difference`` correctly reports a
        measured 0.0 on that same data; a ratio has no value there, and
        answering 1.0 would certify perfect parity from an undefined quotient.
        fairlearn.demographic_parity_ratio and aif360.disparate_impact both
        return NaN for this case too.

    Example:
        >>> ratio = demographic_parity_ratio(y_true, y_pred, gender)
        >>> if ratio < 0.8:
        ...     print("Potential disparate impact detected")

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

    Ledger row: demographic_parity_ratio. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    positive_rates = gm.compute_group_rate(y_pred == 1)

    # Fewer than two qualifying groups (or fewer than two with a DEFINED rate):
    # no pair to compare, so the ratio is unmeasurable -> NaN (see the
    # CONVENTION note above). A 1.0 here read as PERFECT PARITY on data where a
    # dropped minority group was never selected at all, i.e. the exact inverse
    # of the truth.
    if len(positive_rates) < 2:
        return float("nan")

    valid_rates = [r for r in positive_rates.values() if not np.isnan(r)]
    if len(valid_rates) < 2:
        return float("nan")

    min_rate = min(valid_rates)
    max_rate = max(valid_rates)

    # Every qualifying group has a zero selection rate: the model selects
    # NOBODY. The ratio is 0/0, which is not a number, and this branch used to
    # answer 1.0 on the argument that "the comparison DID happen and found the
    # groups equal". The groups ARE equal, and the DIFFERENCE metric says so
    # correctly with its measured 0.0 - but a RATIO of two zeros is undefined,
    # not one. fairlearn.demographic_parity_ratio and aif360's disparate_impact
    # both return NaN here; we agreed with neither.
    #
    # The consequence was concrete, not philosophical. This function is what
    # ``disparate_impact_ratio`` and ``disparate_impact_ratio_with_ci`` compute,
    # and the US-hiring / lending seal gates on that CI's LOWER bound being
    # >= 0.80. On an all-reject model every bootstrap resample also returned
    # 1.0, so the interval was a degenerate [1.0, 1.0] and the four-fifths
    # adverse-impact test was recorded as PASSED on a statistic that was 0/0.
    # A model that rejects every applicant has not demonstrated no adverse
    # impact; it has produced no selection to test, which is the CONVENTION
    # note's case exactly: the comparison this metric names never happened.
    #
    # The mirror case is NOT degenerate and must keep working: one group at 0.0
    # against another that IS selected is a measured 0.0 ratio, the worst
    # possible reading, and it stays 0.0 (see the boundary test in
    # tests/test_audit_wave5_hardening.py).
    if max_rate == 0:
        return float("nan")

    return min_rate / max_rate


def equal_opportunity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute equal opportunity difference.

    Measures the maximum absolute difference in true positive rates (TPR)
    across groups. Focuses on ensuring equal benefit for positive class.

    Library Comparisons:
        AIF360: equal_opportunity_difference()
        Fairlearn: true_positive_rate (use with MetricFrame)
        Aequitas: tpr_disparity

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in TPR across groups; NaN (insufficient
        evidence) when fewer than two groups meet ``min_group_size`` or any
        qualifying group has an undefined TPR.

    Example:
        >>> eo = equal_opportunity_difference(y_true, y_pred, race)
        >>> print(f"Equal Opportunity Difference: {eo:.3f}")

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

    Ledger row: equal_opportunity_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # TPR = TP / (TP + FN) = P(pred=1 | true=1)
    tpr_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 1) & (y_true == 1), denominator_mask=(y_true == 1)
    )

    # Fewer than two qualifying groups: no pair to compare -> unmeasurable, NaN
    # (see the CONVENTION note above), never a false-parity 0.0.
    if len(tpr_rates) < 2:
        return float("nan")

    # A group with zero positive labels has an UNDEFINED TPR (NaN).
    # compute_max_difference skips NaN pairs, so this used to silently
    # return 0.0 ('perfectly fair') exactly when the metric could not be
    # assessed for that group. Return NaN loudly instead.
    undefined = [g for g, r in tpr_rates.items() if np.isnan(r)]
    if undefined:
        warnings.warn(
            f"TPR is undefined for group(s) {undefined} (no positive labels "
            f"in y_true). equal_opportunity_difference cannot be computed "
            f"across all groups; returning NaN.",
            UserWarning,
        )
        return float("nan")

    max_diff, _, _ = compute_max_difference(tpr_rates)
    return max_diff


def equalized_odds_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute equalized odds difference.

    Measures the maximum of TPR difference and FPR difference across groups.
    Ensures similar error rates for both positive and negative classes.

    Library Comparisons:
        AIF360: average_odds_difference() (average, not max)
        Fairlearn: equalized_odds_difference()
        Aequitas: fpr_disparity + tpr_disparity

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum of (TPR difference, FPR difference); NaN (insufficient evidence)
        when fewer than two groups meet ``min_group_size``, or ANY qualifying
        group has an undefined TPR or FPR. An unmeasurable group is never
        dropped and the survivors' spread reported in its place: that spread is
        a LOWER bound, not a measurement. This is the same rule
        ``equal_opportunity_difference`` applies to the TPR leg, so the two are
        assessable under exactly the same conditions.

    Example:
        >>> eod = equalized_odds_difference(y_true, y_pred, gender)
        >>> print(f"Equalized Odds Difference: {eod:.3f}")

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

    Ledger row: equalized_odds_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # TPR = P(pred=1 | true=1)
    tpr_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 1) & (y_true == 1), denominator_mask=(y_true == 1)
    )

    # FPR = P(pred=1 | true=0)
    fpr_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 1) & (y_true == 0), denominator_mask=(y_true == 0)
    )

    # A group with no positives has an UNDEFINED TPR leg; no negatives, an undefined FPR
    # leg. compute_max_difference skips NaN pairs, which would collapse an unmeasurable
    # leg to 0.0 and seal false parity (the same class 0b8bf54 fixed for NPV). With a
    # single qualifying group there is likewise no pair to compare, so that case is
    # unmeasurable too and now returns NaN (see the CONVENTION note above); it used to
    # return 0.0, which certified perfect equalized odds on data where a dropped group
    # was never compared to anything. Either way, any undefined leg across two or more
    # groups is insufficient evidence, not perfect parity.
    if len(tpr_rates) < 2:
        return float("nan")

    # DROPPING an unmeasurable group is what manufactures a LOWER BOUND, so the drop
    # itself is the defect, not merely the two-groups-collapse-to-one case. This block
    # used to filter to the groups with a defined rate and compute the spread whenever
    # TWO survived. With three or more groups that silently EXCLUDED the unmeasurable
    # group and reported the survivors' spread as if it were the whole answer:
    # A (TPR 0.90) | B (TPR 0.88) | C (no positive labels at all) returned 0.0200,
    # a confident number over A and B only, and the report filed it under
    # passed_metrics while listing equal_opportunity_difference -- literally this
    # metric's OWN TPR leg, over the SAME rates -- under not_assessable. One report
    # thereby certified equalized odds and declared its TPR leg not assessable at once.
    # The true equalized-odds gap is >= 0.0200 and unbounded above: C's TPR could be
    # anything. A lower bound presented as a measurement is not a measurement.
    #
    # The rule is now the same one equal_opportunity_difference has always applied
    # (see that function): ANY qualifying group with an undefined rate on EITHER leg
    # makes the metric unassessable, so warn and return NaN. The siblings must be
    # assessable under exactly the same conditions; divergence was the bug.
    undefined_tpr = [g for g, r in tpr_rates.items() if np.isnan(r)]
    undefined_fpr = [g for g, r in fpr_rates.items() if np.isnan(r)]
    if undefined_tpr or undefined_fpr:
        reasons = []
        if undefined_tpr:
            reasons.append(
                f"TPR is undefined for group(s) {undefined_tpr} (no positive labels in y_true)"
            )
        if undefined_fpr:
            reasons.append(
                f"FPR is undefined for group(s) {undefined_fpr} (no negative labels in y_true)"
            )
        warnings.warn(
            f"{'; '.join(reasons)}. equalized_odds_difference cannot be computed "
            f"across all groups; returning NaN.",
            UserWarning,
        )
        return float("nan")

    tpr_diff, _, _ = compute_max_difference(tpr_rates)
    fpr_diff, _, _ = compute_max_difference(fpr_rates)

    return max(tpr_diff, fpr_diff)


def predictive_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute predictive parity difference.

    Measures the maximum absolute difference in precision (PPV) across groups.
    Ensures predictions mean the same thing across groups.

    Library Comparisons:
        AIF360: No direct equivalent (compute via confusion matrix)
        Fairlearn: precision_score (use with MetricFrame)
        Aequitas: ppv_disparity

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum absolute difference in precision across groups; NaN
        (insufficient evidence) when fewer than two groups have a defined
        precision.

    Example:
        >>> pp = predictive_parity_difference(y_true, y_pred, race)
        >>> print(f"Predictive Parity Difference: {pp:.3f}")

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

    Ledger row: predictive_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # Precision = TP / (TP + FP) = P(true=1 | pred=1)
    precision_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 1) & (y_true == 1), denominator_mask=(y_pred == 1)
    )

    # A group with no positive predictions has an UNDEFINED precision (NaN). Collapsing
    # that to a 0.0 difference would seal false parity on a metric that could not be
    # computed (mirror of the 0b8bf54 NPV fix; the canonical case is a model that never
    # approves one group, where sufficiency is exactly unmeasurable). Fewer than two
    # QUALIFYING groups is the same unmeasurable case, not a 0.0 pass (see the
    # CONVENTION note above).
    #
    # The count gated here is the number of QUALIFYING groups, exactly as
    # equal_opportunity_difference counts its tpr_rates. It used to be the number of
    # groups with a DEFINED precision, and that one word made the warning below
    # unreachable in the very case the paragraph above calls canonical: with TWO
    # groups, one of which the model never predicts positive for, the defined count
    # is 1, so the function returned a bare NaN and never said which group was
    # unmeasurable or why. Measured 2026-09-17, n=200, group B never approved:
    # returned nan with zero warnings raised. The returned VALUE is unchanged on
    # every input by this reordering (NaN in each case either way); what changes is
    # that the could-not-check now names its reason.
    if len(precision_rates) < 2:
        warnings.warn(
            f"Fewer than two groups qualify for a precision comparison "
            f"(qualifying group(s): {sorted(precision_rates)}, "
            f"min_group_size={min_group_size}), so predictive parity is NOT "
            f"MEASURABLE; returning NaN. This is insufficient evidence, not a "
            f"measured parity of 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # Same rule as the siblings above. A group the model never predicts positive
    # for has no precision to compare, and reporting the remaining groups' spread
    # as "the" predictive parity understates a disparity whose worst arm is the
    # one that fell out.
    undefined = [g for g, r in precision_rates.items() if not np.isfinite(r)]
    if undefined:
        warnings.warn(
            f"Precision is undefined for group(s) {undefined} (no positive "
            f"predictions in those groups), so the predictive-parity comparison is "
            f"NOT MEASURABLE; returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    max_diff, _, _ = compute_max_difference(precision_rates)
    return max_diff


def calibration_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute calibration difference across groups.

    Measures how well probability predictions reflect true outcomes
    and how this varies across groups. Uses Expected Calibration Error (ECE).

    Library Comparisons:
        AIF360: No direct equivalent
        Fairlearn: No direct equivalent (use calibration_curve separately)
        What-If Tool: Calibration visualization available

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Maximum difference in ECE across groups; NaN (insufficient evidence)
        when fewer than two groups meet min_group_size, because a
        between-group difference cannot be measured there. The report routes
        NaN to NOT_ASSESSABLE; a 0.0 would falsely certify calibration parity.

    Example:
        >>> cal_diff = calibration_difference(y_true, y_pred, gender, y_prob)
        >>> print(f"Calibration Difference: {cal_diff:.3f}")
    """
    y_true, y_pred, sensitive_attr, y_prob, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        y_prob,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    # y_prob is a required (non-None) argument here, so validate_inputs
    # always returns it as an array (it only preserves None for None input).
    assert y_prob is not None

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    def compute_ece(probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)

        for i in range(n_bins):
            bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include 1.0 in last bin
                bin_mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probs[bin_mask])
                ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

        return ece

    ece_values = {}
    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        group_probs = y_prob[mask]
        group_labels = y_true[mask]

        if len(group_probs) >= min_group_size:
            ece_values[name] = compute_ece(group_probs, group_labels)

    # Fewer than two groups have a computable ECE: the between-group
    # calibration difference is UNMEASURABLE. Returning 0.0 read as perfect
    # calibration parity (PASS) exactly when it could not be measured, while
    # auroc_parity on the same data correctly said NOT_ASSESSABLE. NaN routes
    # the report to insufficient evidence instead. This metric led the way: the
    # single-qualifying-group case was already NaN here while the four parity
    # metrics fixed in fd9746b still pinned 0.0 for it, and
    # test_audit_wave5_hardening pinned that 0.0 as documented behaviour.
    # The whole family now agrees on NaN (see the CONVENTION note at the top of
    # this module). A per-group ECE is always defined for a nonempty group, so
    # <2 here always means fewer than two qualifying groups.
    if len(ece_values) < 2:
        return float("nan")

    max_diff, _, _ = compute_max_difference(ece_values)
    return max_diff


def get_group_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    y_prob: Optional[ArrayLike] = None,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed metrics for each group.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        y_prob: Optional predicted probabilities
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Dict mapping group names to metric dictionaries

    Example:
        >>> metrics = get_group_metrics(y_true, y_pred, gender)
        >>> for group, m in metrics.items():
        ...     print(f"{group}: TPR={m['tpr']:.3f}, FPR={m['fpr']:.3f}")
    """
    y_true, y_pred, sensitive_attr, y_prob, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        y_prob,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    results = {}

    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        tn = np.sum((yp == 0) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))

        results[name] = {
            "size": len(yt),
            "positive_rate": np.mean(yp),
            "base_rate": np.mean(yt),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "accuracy": (tp + tn) / len(yt),
        }

    return results


def fpr_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute false-positive-rate (FPR) parity difference, aka predictive equality.

    Max-min spread of FPR = P(pred=1 | true=0) across groups. This is the FPR leg of
    equalized odds surfaced as a first-class metric: the KG catalog advertised ER-003
    (FPR Parity) but the engine had no standalone function (spine audit finding 10).

    Library Comparisons:
        AIF360: false_positive_rate_difference()
        Aequitas: fpr_disparity

    Returns:
        Maximum absolute difference in FPR across groups; NaN (insufficient evidence) when
        fewer than two groups have a DEFINED FPR.

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

    Ledger row: fpr_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    fpr_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 1) & (y_true == 0), denominator_mask=(y_true == 0)
    )
    # A group with no negatives (y_true all 1) has an UNDEFINED FPR; that is insufficient
    # evidence, not 0.0 parity (mirror of the 0b8bf54 NPV fix). Fewer than two QUALIFYING
    # groups is equally unmeasurable and no longer returns 0.0 (CONVENTION note above).
    defined = {g: v for g, v in fpr_rates.items() if np.isfinite(v)}
    if len(defined) < 2:
        return float("nan")
    # ANY qualifying group with an undefined rate makes the comparison
    # unmeasurable, exactly as equal_opportunity_difference has always held and
    # as fnr_parity_difference, negative_predictive_value_difference and
    # auroc_parity were brought onto. Dropping the group and reporting the
    # survivors' spread hands back a LOWER BOUND presented as the disparity, and
    # it is monotone the wrong way: the more groups fall out, the better the
    # number reads. A group with no negative labels has no false-positive rate
    # to compare, and that is insufficient evidence, not parity.
    undefined = [g for g, r in fpr_rates.items() if not np.isfinite(r)]
    if undefined:
        warnings.warn(
            f"FPR is undefined for group(s) {undefined} (no negative labels in "
            f"those groups), so the false-positive-rate comparison is NOT MEASURABLE; "
            f"returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    max_diff, _, _ = compute_max_difference(defined)
    return max_diff


def fnr_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute false-negative-rate (FNR) parity difference.

    Max-min spread of FNR = P(pred=0 | true=1) = 1 - TPR across groups. By this identity
    FNR parity is numerically equal to equal-opportunity (TPR) difference; it is provided
    as a named metric because the KG catalog advertised ER-004 (spine audit finding 2),
    and because FNR framing (missed positives) is the salient harm in some domains.

    Library Comparisons:
        AIF360: false_negative_rate_difference()

    Returns:
        Maximum absolute difference in FNR across groups; NaN (insufficient evidence) when
        fewer than two groups meet ``min_group_size``, or ANY qualifying group has an
        undefined FNR. An unmeasurable group is never dropped and the survivors' spread
        reported in its place: that spread is a LOWER bound, not a measurement. This is
        the same rule ``equal_opportunity_difference`` applies, and since FNR = 1 - TPR
        the two must be assessable under exactly the same conditions.

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

    Ledger row: fnr_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    fnr_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 0) & (y_true == 1), denominator_mask=(y_true == 1)
    )
    # Fewer than two qualifying groups: no pair to compare -> unmeasurable, NaN
    # (see the CONVENTION note above), never a false-parity 0.0.
    if len(fnr_rates) < 2:
        return float("nan")

    # A group with no positives (y_true all 0) has an UNDEFINED FNR; that is insufficient
    # evidence, not 0.0 parity (mirror of the 0b8bf54 NPV fix).
    #
    # DROPPING that group is itself the defect, not only the case where the drop leaves a
    # single survivor. This block used to keep the groups with a defined rate and compute
    # the spread whenever TWO survived, so with three or more groups it silently EXCLUDED
    # the unmeasurable one: A (FNR 0.25) | B (FNR 0.75) | C (no positive labels at all)
    # returned 0.5000, a confident number over A and B only, while
    # equal_opportunity_difference on the SAME rows returned NaN. FNR = 1 - TPR, so one
    # report filed this metric as measured and its own identity twin as not assessable.
    # C's FNR could be anything, so 0.5000 is a lower bound, and a lower bound presented
    # as a measurement is not a measurement. It is also monotone the wrong way: the more
    # groups fall out, the narrower, and therefore the fairer, the number looks.
    undefined = [g for g, r in fnr_rates.items() if not np.isfinite(r)]
    if undefined:
        warnings.warn(
            f"FNR is undefined for group(s) {undefined} (no positive labels in "
            f"y_true). fnr_parity_difference cannot be computed across all groups; "
            f"returning NaN.",
            UserWarning,
        )
        return float("nan")

    max_diff, _, _ = compute_max_difference(fnr_rates)
    return max_diff


def accuracy_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute accuracy parity difference: max-min spread of per-group accuracy = P(pred==true).

    NOTE: accuracy parity is a PERFORMANCE metric, not a group-fairness criterion in the
    independence / separation / sufficiency sense, and it can hide FP/FN trade-offs. It is
    provided because the KG catalog advertised ER-006 (spine audit finding 11); callers
    should treat it as a performance/robustness signal, not a standalone fairness verdict.

    Returns:
        Maximum absolute difference in accuracy across groups; NaN (insufficient evidence)
        when fewer than two groups have a defined accuracy.

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

    Ledger row: accuracy_parity_difference. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    metrics = get_group_metrics(
        y_true,
        y_pred,
        sensitive_attr,
        min_group_size=min_group_size,
        missing_strategy=missing_strategy,
    )
    accs = [m["accuracy"] for m in metrics.values() if np.isfinite(m["accuracy"])]
    # Fewer than two comparable groups: unmeasurable -> NaN, never a 0.0 that reads as
    # equal accuracy for everyone (CONVENTION note above).
    if len(accs) < 2:
        return float("nan")
    return float(max(accs) - min(accs))


def worst_group_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """
    Compute worst-group accuracy: the minimum per-group accuracy (Sagawa et al. 2020).

    NOTE: a performance / robustness statistic, not a fairness-difference metric. Provided
    because the KG catalog advertised RO-001 as "Worst-Group Accuracy" while the engine
    computed only worst-subgroup disparity / selection rate (spine audit finding 3).

    A group EXCLUDED by ``min_group_size`` makes this statistic unanswerable, and
    that is the whole point of it. Measured 2026-09-07, before this guard: group
    ``a`` n=396 accuracy 1.000, group ``b`` n=4 accuracy 0.000, and
    ``worst_group_accuracy`` returned **1.000** with no warning. It reported the
    BEST group's accuracy as the worst group's, which inverts the reading of a
    Sagawa-style statistic whose entire purpose is to surface the worst-off
    subgroup. The excluded group is precisely the one most likely to be worst.

    So a drop returns NaN rather than a minimum over the survivors. The minimum
    over survivors is a real number about the survivors, but it is not the
    quantity this function is named for, and the two are indistinguishable to a
    caller reading a float.

    Returns:
        Minimum accuracy across groups, or nan when no group is valid OR when any
        group was excluded by the size gate (the excluded group could be the
        worst, so the worst-group statistic is not assessable).

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

    Ledger row: worst_group_accuracy. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    # missing_strategy applied BEFORE the size gate, not only after it. The
    # gate used to run on the RAW attribute, so records with NO protected
    # attribute formed a phantom group of their own and tripped
    # get_invalid_groups(). Measured 2026-09-09 on 40 + 40 records at
    # accuracies 1.000 and 0.500 plus 5 records with no attribute: this
    # returned NaN, "not assessable", where the answer is 0.500, and the same
    # frame with an object-dtype attribute raised a bare TypeError out of the
    # sort instead of the library's InvalidDataError for
    # missing_strategy='error'. Every other metric in this file honours the
    # strategy by calling validate_inputs first; this one now does too.
    y_true, y_pred, sensitive_attr, _, _ = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    gm = GroupManager(np.asarray(sensitive_attr), min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    # THE SECOND READER OF THE DROP LIST IN THIS FILE, and the same reading: this
    # asks "was any group dropped", and at a threshold of 1 or less nothing can
    # be, because every group is built from a level that actually occurs. The
    # answer is [] either way, so the short-circuit changes no verdict; what it
    # removes is a warning about a vacuous size check that this line never treats
    # as a clean bill of group sizes. Measured 2026-09-27 on 20 + 20 rows at
    # accuracy 1.0: worst_group_accuracy(min_group_size=1) returned 1.0 and raised
    # "GroupManager.get_invalid_groups: min_group_size is 1 ... This empty result
    # is vacuous"; after, it returns the same 1.0 in silence. min_group_size=2
    # (1.0, silent) and 30 (nan, the size gate firing) are untouched, and
    # operations/cicd/gate.py:1288 reaches this file with min_group_size=1.
    if gm.min_group_size > 1 and gm.get_invalid_groups():
        # Reported, not repaired: substituting the survivors' minimum would
        # fabricate the very quantity the caller asked to have measured.
        return float("nan")

    metrics = get_group_metrics(
        y_true,
        y_pred,
        sensitive_attr,
        min_group_size=min_group_size,
        missing_strategy="exclude",  # Already handled above
    )
    accs = [m["accuracy"] for m in metrics.values() if np.isfinite(m["accuracy"])]
    if not accs:
        return float("nan")
    return float(min(accs))


def disparate_impact_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """Disparate-impact (adverse-impact) ratio: the min/max selection-rate ratio across
    groups, i.e. the four-fifths-rule statistic (a value below 0.80 is the EEOC screening
    heuristic). This is the legally recognizable NAME for the selection-rate RATIO and is the
    independence PRIMARY for US hiring / US lending. It is computed identically to
    ``demographic_parity_ratio``; it is kept distinct from ``demographic_parity_difference``
    (a rate DIFFERENCE band) because a ratio floor and a difference band are different tests,
    and conflating them is a category error.

    Returns:
        Minimum ratio of positive-prediction rates (min/max); 1.0 = parity; NaN
        (insufficient evidence) when fewer than two groups can be compared, since a
        four-fifths reading of 1.0 there would certify no adverse impact on an
        adverse-impact test that never ran. NaN for the same reason when NO group is
        selected at all (an all-reject model): the ratio is 0/0, there is no selection
        to test for adverse impact, and a 1.0 would clear the four-fifths floor - and
        the sealed ``lower_bound >= 0.80`` gate with it - on an undefined quotient.

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

    Ledger row: disparate_impact_ratio. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    return demographic_parity_ratio(
        y_true,
        y_pred,
        sensitive_attr,
        min_group_size=min_group_size,
        missing_strategy=missing_strategy,
    )


def negative_predictive_value_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> float:
    """Negative predictive value (NPV) parity: the maximum |NPV| difference across groups,
    where NPV = P(y=0 | pred=0) = TN / (TN + FN). The sufficiency counterpart of predictive
    parity (PPV) on the NEGATIVE decision; used as a criminal-justice sufficiency diagnostic
    (a low-risk label should carry the same true low-risk probability across groups).

    Returns:
        Maximum absolute difference in NPV across groups; NaN (insufficient evidence) when
        fewer than two groups meet ``min_group_size``, or ANY qualifying group has an
        undefined NPV (no negative predictions at all). An unmeasurable group is never
        dropped and the survivors' spread reported in its place: that spread is a LOWER
        bound, not a measurement. Same rule as ``equal_opportunity_difference``.

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

    Ledger row: negative_predictive_value_difference. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # NPV = TN / (TN + FN) = P(true=0 | pred=0)
    npv_rates = gm.compute_group_rate(
        numerator_mask=(y_pred == 0) & (y_true == 0), denominator_mask=(y_pred == 0)
    )

    # Fewer than two qualifying groups: no pair to compare -> unmeasurable, NaN
    # (see the CONVENTION note above), never a false-parity 0.0.
    if len(npv_rates) < 2:
        return float("nan")

    # A group with no negative predictions has an UNDEFINED NPV (NaN). Collapsing that to a
    # 0.0 difference would seal false parity (with a deceptively tight CI) on a metric that
    # could not be computed.
    #
    # Counting only the DEFINED NPVs, the previous behaviour, fixed just the case where the
    # drop left a single survivor. With three or more groups it still dropped the
    # unmeasurable group and reported the survivors' spread as the answer: A (NPV 0.75) |
    # B (NPV 0.50) | C (never predicts the negative class) returned 0.2500 over A and B
    # only. C's NPV is unknown, so the true spread is >= 0.2500 and unbounded above; the
    # sufficiency diagnostic for the one group whose negative decisions were never
    # assessed is could not check, not a measured pass. Dropping can only NARROW the
    # spread, so the more groups fall out, the better the number reads.
    undefined = [g for g, r in npv_rates.items() if not np.isfinite(r)]
    if undefined:
        warnings.warn(
            f"NPV is undefined for group(s) {undefined} (no negative predictions in "
            f"y_pred). negative_predictive_value_difference cannot be computed across "
            f"all groups; returning NaN.",
            UserWarning,
        )
        return float("nan")

    max_diff, _, _ = compute_max_difference(npv_rates)
    return max_diff


def _recorded_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of RECORDED observations: entries that carry a decision or a score.

    A non-finite entry (NaN, None, inf) is an ABSENT observation, not a recorded negative.
    This distinction is load-bearing: ``np.nan == 1`` and ``np.nan >= threshold`` are both
    silently False under IEEE-754, so a comparison written against the raw array counts every
    unrecorded row as a recorded rejection. Metrics built that way return a confident
    "no disparity" for data in which nothing was decided at all. Filter with this mask FIRST,
    then apply the size/cell floors to what survives, so a hollowed-out group falls out and
    the metric's own insufficient-evidence escape fires.

    Non-numeric arrays (a metric handed categorical labels) cannot be tested for finiteness;
    only ``None`` is treated as absent there, and everything else counts as recorded.
    """
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return np.array([v is not None for v in values.ravel()], dtype=bool).reshape(values.shape)
    return np.isfinite(numeric)


def conditional_demographic_disparity(
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    strata: ArrayLike,
    *,
    min_group_size: int = 30,
    min_cell: int = 5,
) -> float:
    """Conditional demographic disparity (CDD): the selection-rate disparity that REMAINS
    after conditioning on a legitimate stratifier (an audited, non-proxy factor such as a
    bona-fide job requirement). It is the CJEU objective-justification mirror the EU-hiring
    profile seals on: within each stratum the maximum selection-rate difference across
    protected groups is measured, then size-weighted by stratum population and pooled. A raw
    group gap that is fully explained by the legitimate factor does NOT count against the
    model; 0 means no disparity beyond that factor.

    Args:
        y_pred: selection / positive-decision indicator (0/1).
        sensitive_attr: protected attribute.
        strata: the legitimate conditioning variable (categorical); disparity is measured
            WITHIN each level and pooled by size.
        min_group_size: minimum size for a protected group to be assessed at all.
        min_cell: minimum (stratum, group) cell size to enter that stratum's disparity.

    Returns:
        Assessed-stratum-weighted within-stratum demographic-parity difference, a float in
        [0, 1]; NaN (insufficient evidence) when no stratum has two assessable groups.

    Reference: Wachter, Mittelstadt & Russell (2021), Why Fairness Cannot Be Automated
    (conditional demographic disparity as the EU objective-justification mirror).

    DIVERGENCE FROM THAT REFERENCE, read this before registering a margin. What this
    function returns is NOT the statistic defined in the cited paper, and the two are not
    interchangeable:

    * The canonical CDD of Wachter, Mittelstadt & Russell (2021), as operationalized for
      example in AWS SageMaker Clarify, is COMPOSITION based and SIGNED. Within stratum k
      it compares a protected group's share of the REJECTED pool with its share of the
      ACCEPTED pool, ``DD_k = P(group=d | rejected, k) - P(group=d | accepted, k)``, then
      size-weights across strata. Its sign names which group is disadvantaged, and it can
      be negative.
    * This function returns the size-weighted within-stratum maximum SELECTION-RATE gap
      across protected groups: an unsigned, reference-free spread in [0, 1]. It never goes
      negative, so it names no direction, and it is a rate difference rather than a
      composition difference.

    The two answer related but different questions and diverge materially on the same data
    (differences of tens of percent have been measured). A tolerance calibrated on the CDD
    literature, or on a Clarify run, therefore does NOT transfer to this number one for
    one: register any margin against the definition implemented here. The estimand is a
    recorded open decision, not an oversight; it is deliberately not realigned because
    sealed assessment cells gate on the current statistic and changing the formula would
    alter already-sealed verdicts. Full record: ``docs/DIVERGENCES.md`` in the project
    repository, section "Conditional demographic disparity (CDD)".

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

    Ledger row: conditional_demographic_disparity. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    yp = np.asarray(y_pred)
    a = np.asarray(sensitive_attr)
    s = np.asarray(strata)
    if not (len(yp) == len(a) == len(s)):
        raise ValueError("y_pred, sensitive_attr and strata must have the same length")

    gm = GroupManager(a, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = set(gm.get_valid_groups(warn_if_empty=False))
    if len(valid_groups) < 2:
        return float("nan")

    # An UNRECORDED decision is an absent observation, not a recorded rejection. Without this
    # mask the cell rate is float(np.mean(yp[cell] == 1)) over the raw array, and nan == 1 is
    # False, so every missing decision is counted as a negative: with nothing recorded every
    # cell rate is 0.0, the within-stratum max-min gap is 0.0, assessed_weight is non-zero
    # (the cells still have ROWS, and rows are all the min_cell test counted), the NaN escape
    # below is bypassed, and this function - the CJEU objective-justification mirror the
    # EU-hiring profile SEALS on - certifies "no disparity beyond the legitimate factor" for a
    # comparison that read no decision at all. Partial missingness is worse than an undercount:
    # the direction of the reported disparity flips with which group lost its decisions.
    recorded = _recorded_mask(yp)
    n_unrecorded = int((~recorded).sum())
    if n_unrecorded:
        warnings.warn(
            f"{n_unrecorded} of {len(yp)} decisions in y_pred are unrecorded (non-finite) and "
            f"are EXCLUDED from conditional_demographic_disparity rather than counted as "
            f"rejections. Cells are held to min_cell on the RECORDED rows only, so an assessed "
            f"stratum is not the same as a fully observed one; the result is NaN if no stratum "
            f"retains two assessable groups.",
            UserWarning,
        )

    cdd = 0.0
    assessed_weight = 0.0
    partial_strata: list[str] = []
    for level in np.unique(s):
        s_mask = s == level
        n_s = int(s_mask.sum())
        if n_s == 0:
            continue
        rates = []
        dropped_here = []
        for g in valid_groups:
            cell = s_mask & (a == g) & recorded
            n_cell = int(cell.sum())
            if n_cell >= min_cell and n_cell > 0:
                rates.append(float(np.mean(yp[cell] == 1)))
            else:
                dropped_here.append(str(g))
        # BGL-D, found by the Stage 1 audit 2026-09-11. A stratum that lost ANY
        # valid group's cell is COULD-NOT-CHECK, not a measured stratum, and it
        # must not carry weight.
        #
        # The old code scored such a stratum on whichever groups survived and
        # gave it full population weight, which dilutes the result toward "fair"
        # using a comparison that never examined the missing group. Reproduced:
        # group 'a' approving at 100% against 50%, valid overall at 58 rows but
        # holding only 2 rows in the second stratum, took the metric from 0.5000
        # over the assessable stratum to 0.2804 over both. The group most likely
        # to lose a cell is the smallest one, which is the group a fairness audit
        # exists to protect, so the error runs in the unsafe direction by
        # construction. This is the rule auroc_parity in this same file already
        # applies.
        if dropped_here:
            partial_strata.append(
                f"{level} (no assessable cell for {', '.join(sorted(dropped_here))})"
            )
            continue
        if len(rates) >= 2:
            cdd += n_s * (max(rates) - min(rates))
            assessed_weight += n_s

    if partial_strata:
        warnings.warn(
            f"conditional_demographic_disparity: {len(partial_strata)} stratum/strata were "
            f"EXCLUDED because a valid group had no assessable cell in them "
            f"({'; '.join(partial_strata)}). A stratum scored on the groups that happened to "
            f"survive would dilute the result toward 'no disparity' using a comparison that "
            f"never examined the missing group, and the group most likely to lose a cell is "
            f"the smallest one. The result covers the fully assessed strata only, and is NaN "
            f"if none remain.",
            UserWarning,
            stacklevel=2,
        )
    # Renormalize by the ASSESSED-stratum population, not the full n. A stratum whose
    # disparity could not be measured (a cell below min_cell) must not dilute the result
    # toward "fair" as though it had zero disparity. If NO stratum is assessable, the metric
    # is insufficient evidence, not zero disparity, so return NaN.
    if assessed_weight == 0:
        return float("nan")
    return cdd / assessed_weight


def auroc_parity(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
) -> float:
    """AUROC parity: the maximum difference in per-group ROC AUC (discrimination). This is a
    VALIDITY gate, not a fairness-difference metric: it guards against a model that looks
    'fair' only because it is uninformative for some group (the levelling-down failure). Used
    as the EU-healthcare validity gate alongside the sealed calibration statistic. A group
    whose AUROC is undefined (a single outcome class present) makes the gate unassessable;
    it is not skipped and the rest reported in its place.

    Returns:
        Max |AUROC_g - AUROC_h| across groups; NaN (insufficient evidence) when fewer than
        two groups meet ``min_group_size``, or ANY qualifying group has an undefined
        AUROC. A 0.0 there would falsely PASS the validity gate precisely when the model
        is uninformative for a group, and a spread computed over the survivors is a LOWER
        bound on the real gap, not a measurement of it.

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

    Ledger row: auroc_parity. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    a = np.asarray(sensitive_attr)
    if not (len(yt) == len(ys) == len(a)):
        raise ValueError("y_true, y_score and sensitive_attr must have the same length")

    gm = GroupManager(a, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    qualifying = gm.get_valid_groups(warn_if_empty=False)

    # Fewer than two qualifying groups: no pair to compare -> unmeasurable, NaN
    # (see the CONVENTION note above), never a false-parity 0.0.
    if len(qualifying) < 2:
        return float("nan")

    aucs: Dict[str, float] = {}
    undefined = []
    for g in qualifying:
        m = gm.get_mask(g)
        if len(np.unique(yt[m])) < 2:
            # ROC AUC is undefined when only one outcome class is present.
            undefined.append(g)
            continue
        aucs[g] = float(roc_auc_score(yt[m], ys[m]))

    # The undefined group used to be SKIPPED and the spread reported over the rest as
    # long as two survived: A (AUROC 1.00) | B (AUROC 0.60) | C (one outcome class only)
    # returned 0.4000, a confident number over A and B while C, the group this validity
    # gate exists to protect, was never assessed at all. That is exactly the
    # levelling-down failure the gate watches for, arriving as a PASS: C's AUROC is
    # unknown, so the true gap is >= 0.4000 and unbounded above, and dropping a group can
    # only NARROW the spread, so the more groups fall out the better the gate reads.
    # An undefined AUROC is could not check, so the gate is could not check.
    if undefined:
        warnings.warn(
            f"ROC AUC is undefined for group(s) {undefined} (only one outcome class "
            f"present in y_true). auroc_parity cannot be computed across all groups; "
            f"returning NaN.",
            UserWarning,
        )
        return float("nan")

    return float(max(aucs.values()) - min(aucs.values()))


def net_benefit_parity(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> float:
    """Net-benefit (decision-curve) parity across groups at a decision threshold p_t.

    Net benefit = TP/n - (FP/n) * (p_t / (1 - p_t)), the standard decision-curve quantity
    (Vickers & Elkin 2006): it weights the harm of a false positive by the threshold odds, so
    it captures whether the model is CLINICALLY useful for each group, not merely statistically
    fair. Guards the 'fair but useless' failure the EU-healthcare tertiary watches. Parity is
    the maximum between-group net-benefit difference.

    Returns:
        Max |net_benefit_g - net_benefit_h| across groups; NaN (insufficient evidence) when
        fewer than two groups have a defined net benefit.

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

    Ledger row: net_benefit_parity. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    a = np.asarray(sensitive_attr)
    if not (len(yt) == len(yp) == len(a)):
        raise ValueError("y_true, y_prob and sensitive_attr must have the same length")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1)")

    odds = threshold / (1.0 - threshold)
    gm = GroupManager(a, min_group_size=min_group_size)
    _warn_dropped_groups(gm)

    # A group's net benefit is defined only over the subjects that were actually SCORED.
    # `nan >= threshold` is silently False, so reading the raw array makes every unscored
    # subject a confident "do not treat": tp = fp = 0, and a group nothing was scored for gets
    # a DEFINED net benefit of exactly 0.0. That is byte-identical to what a real, equal,
    # clinically useful model returns, the len(net_benefit) < 2 escape below never fires, and
    # check_threshold grades the non-measurement PASS. Hold the SCORED subset to
    # min_group_size, the same floor the group itself is held to: a 60-row group with 2 scored
    # rows is not a 60-row group for this estimand, and dropping it here is what lets the
    # insufficient-evidence escape do its job.
    scored_mask = np.isfinite(yp)
    net_benefit = {}
    unscored = {}
    for g in gm.get_valid_groups(warn_if_empty=False):
        m = gm.get_mask(g) & scored_mask
        n = int(m.sum())
        if n == 0 or n < min_group_size:
            unscored[g] = n
            continue
        pred = yp[m] >= threshold
        tp = int(((yt[m] == 1) & pred).sum())
        fp = int(((yt[m] == 0) & pred).sum())
        net_benefit[g] = tp / n - (fp / n) * odds

    if unscored:
        warnings.warn(
            f"net_benefit_parity: group(s) {sorted(unscored, key=str)} have fewer than "
            f"min_group_size={min_group_size} SCORED subjects (finite y_prob): "
            f"{ {str(g): n for g, n in sorted(unscored.items(), key=lambda kv: str(kv[0]))} }. "
            f"Their net benefit is undefined, not 0.0, so they are excluded from the parity "
            f"comparison; the result is NaN if fewer than two groups remain.",
            UserWarning,
        )

    # Fewer than two groups have a defined net benefit: the between-group difference is
    # UNMEASURABLE. Returning 0.0 would seal false parity; NaN routes to insufficient_evidence.
    if len(net_benefit) < 2:
        return float("nan")
    return float(max(net_benefit.values()) - min(net_benefit.values()))


def conditional_adverse_impact(
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    covariates: ArrayLike,
    *,
    min_group_size: int = 30,
) -> float:
    """Conditional adverse impact: the selection-rate disparity that REMAINS after
    regression-controlling for bona-fide (non-proxy) covariates.

    The group effect is estimated JOINTLY: a logistic model predicts the selection decision
    from the legitimate covariates AND a full set of group indicators, and the metric is the
    maximum pairwise difference of the groups' average marginal effects (AME). The AME of a
    group is the counterfactual mean selection probability if the whole sample were placed in
    that group with covariates held fixed; the max pairwise AME gap is the prohibited-basis
    selection effect the covariates do not explain. Fitting the covariates WITHOUT the group
    indicators and reading the between-group residual-mean gap (the previous behaviour) is
    downward-biased whenever the covariates correlate with the protected attribute (omitted-
    group-regressor attenuation), which is the realistic case and biases toward PASS. It is a
    hiring-US diagnostic, not a sealed gate.

    Args:
        y_pred: selection / positive-decision indicator (0/1).
        sensitive_attr: protected attribute.
        covariates: bona-fide predictors (1D or 2D); the disparity is measured after these are
            controlled for jointly with the group indicators.
        min_group_size: minimum size for a protected group to be assessed.

    Returns:
        Max pairwise between-group AME difference; NaN (insufficient evidence) when fewer than
        two groups are assessable or the group effect is not identified (covariates collinear
        with the group indicators, e.g. a perfect proxy).

    Reference: average marginal effect of a categorical predictor in a logistic model; the
    joint-fit fair-lending regression estimand (Ross & Yinger 2002), the binary-outcome
    counterpart of ``pricing_disparity``.

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

    Ledger row: conditional_adverse_impact. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    from sklearn.linear_model import LogisticRegression

    yp = np.asarray(y_pred, dtype=float)
    a = np.asarray(sensitive_attr)
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not (len(yp) == len(a) == len(X)):
        raise ValueError("y_pred, sensitive_attr and covariates must have the same length")

    gm = GroupManager(a, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups(warn_if_empty=False)
    # Fewer than two assessable groups: the adjusted gap is unmeasurable -> NaN (insufficient
    # evidence), never a false-fair 0.0.
    if len(valid_groups) < 2:
        return float("nan")

    # BGL-S2 (2026-09-16). MISSING is not CONSTANT, and the variation guard below
    # cannot tell them apart: np.unique on an all-NaN float array collapses to a
    # single element ([nan]), so `len(...) < 2` was True and the function returned
    # the literal 0.0, "no adverse impact", for 2000 rows in which not one
    # selection decision was recorded. The inversion was exact: PARTIAL
    # missingness was refused loudly (sklearn raises "Input y contains NaN" at the
    # fit below) while TOTAL missingness returned the cleanest number the metric
    # can produce. Finiteness is therefore checked ABOVE the variation guard, and
    # covers both cases with the same NaN the docstring already promises for
    # insufficient evidence.
    finite = np.isfinite(yp)
    if not finite.all():
        warnings.warn(
            f"conditional_adverse_impact: {int((~finite).sum())} of {len(yp)} decisions are "
            f"not finite, so they were never recorded. Returning NaN (insufficient evidence), "
            f"not 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    if len(np.unique(yp)) < 2:
        # No variation in the decision: no adverse impact to measure, and the logistic fit is
        # degenerate. The counterfactual selection probability is identical for every group.
        # Reachable only for a genuinely CONSTANT recorded decision: the finiteness check
        # above has already refused a missing one.
        return 0.0

    # Joint fit: selection ~ group indicators + covariates, no separate intercept (the full
    # one-hot block spans it), so every group's direct effect is estimated instead of leaking
    # into the covariate coefficients. Every present group gets an indicator (including groups
    # below min_group_size), but only assessable groups are compared.
    all_groups = list(gm.groups)
    D = np.column_stack([gm.get_mask(g).astype(float) for g in all_groups])
    design = np.column_stack([D, X])

    # BGL-S2 (2026-09-16), same shape on the covariate side: a non-finite covariate
    # made np.linalg.matrix_rank raise LinAlgError("SVD did not converge") out of
    # the identifiability guard, an opaque failure from a metric whose documented
    # refusal is NaN. Covariates nobody recorded are a could-not-check, so say so.
    finite_X = np.isfinite(X)
    if not finite_X.all():
        warnings.warn(
            f"conditional_adverse_impact: {int((~finite_X).sum())} of {X.size} covariate "
            f"values are not finite, so the adjusted model cannot be fitted. Returning NaN "
            f"(insufficient evidence), not 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # Guard identifiability: if the design is rank-deficient the group effect is not
    # identified (a covariate is collinear with the indicators, e.g. a perfect proxy).
    if np.linalg.matrix_rank(design) < design.shape[1]:
        warnings.warn(
            "conditional_adverse_impact: covariates are collinear with the group indicators, "
            "so the adjusted group effect is not identified; returning NaN.",
            UserWarning,
        )
        return float("nan")

    model = LogisticRegression(max_iter=1000, C=1e6, fit_intercept=False)
    model.fit(design, yp)

    # AME of each group: set the whole sample to that group (covariates fixed), average the
    # predicted selection probability. The gap between two groups' AMEs is the covariate-
    # adjusted selection difference.
    group_index = {g: i for i, g in enumerate(all_groups)}
    ames = {}
    for g in valid_groups:
        Dg = np.zeros_like(D)
        Dg[:, group_index[g]] = 1.0
        design_g = np.column_stack([Dg, X])
        ames[g] = float(np.mean(model.predict_proba(design_g)[:, 1]))
    return max(ames.values()) - min(ames.values())


# Statistical Validation Functions


def demographic_parity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute demographic parity difference with confidence interval.

    Uses stratified bootstrap resampling to maintain group proportions.
    For small samples (n < 30 per group) the 'auto'/'bayesian' path currently
    falls back to bootstrap with doubled resamples (and warns); a genuine
    Bayesian difference interval is available via
    ``_statistics.bayesian_difference_ci`` for two-group proportion gaps.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples (≥5000 recommended)
        confidence_level: Confidence level (e.g., 0.95)
        method: 'auto' (select based on sample size), 'bootstrap', or 'bayesian'
        random_state: Random seed for reproducibility

    Returns:
        StatisticalResult with point estimate and confidence/credible interval

    Example:
        >>> result = demographic_parity_difference_with_ci(
        ...     y_true, y_pred, gender, n_bootstrap=5000
        ... )
        >>> print(f"DP Diff: {result.point_estimate:.3f} "
        ...       f"95% CI [{result.lower_bound:.3f}, {result.upper_bound:.3f}]")
        >>> print(f"Interval type: {result.interval_type.value}")

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

    Ledger row: demographic_parity_difference_with_ci. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    return compute_metric_with_ci(
        demographic_parity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",  # Already handled above
    )


def predictive_parity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute predictive parity (precision / PPV gap) difference with a confidence interval.

    Closes the CI hole for the sufficiency-family metric ER-005 (spine audit finding 14):
    it previously shipped as a bare point estimate, yet it is one leg of the impossibility
    trade-off and must carry uncertainty when sealed. Uses the same stratified-bootstrap /
    small-sample routing as the other _with_ci variants via compute_metric_with_ci.

    Returns:
        StatisticalResult with point estimate and confidence/credible interval.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: predictive_parity_difference_with_ci. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    return compute_metric_with_ci(
        predictive_parity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",  # Already handled above
    )


def equalized_odds_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute equalized odds difference with confidence interval.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        method: 'auto', 'bootstrap', or 'bayesian'
        random_state: Random seed

    Returns:
        StatisticalResult with point estimate and interval

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: equalized_odds_difference_with_ci. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    return compute_metric_with_ci(
        equalized_odds_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def equal_opportunity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute equal opportunity difference with confidence interval.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        method: 'auto', 'bootstrap', or 'bayesian'
        random_state: Random seed

    Returns:
        StatisticalResult with point estimate and interval

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: equal_opportunity_difference_with_ci. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    return compute_metric_with_ci(
        equal_opportunity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def compute_effect_sizes(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, Dict]:
    """
    Compute effect sizes for pairwise group comparisons.

    Returns Cohen's d, risk ratios, and odds ratios for all pairs
    of groups, providing standardized measures of disparity magnitude.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        Dict with effect sizes for each group pair:
        {
            'group1_vs_group2': {
                'cohens_d_positive_rate': float,
                'risk_ratio': (rr, lower, upper),
                'odds_ratio': (or, lower, upper),
                'interpretation': str
            },
            ...
        }

    Example:
        >>> effects = compute_effect_sizes(y_true, y_pred, gender)
        >>> for pair, e in effects.items():
        ...     print(f"{pair}: Cohen's d = {e['cohens_d_positive_rate']:.3f}")
        ...     print(f"  Risk Ratio = {e['risk_ratio'][0]:.2f}")

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: compute_effect_sizes. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    valid_groups = gm.get_valid_groups()

    if len(valid_groups) < 2:
        return {}

    results = {}

    for i, g1 in enumerate(valid_groups):
        for g2 in valid_groups[i + 1 :]:
            mask1 = gm.get_mask(g1)
            mask2 = gm.get_mask(g2)

            pred1 = y_pred[mask1]
            pred2 = y_pred[mask2]

            # Effect size on the SELECTION-RATE difference. These are
            # Bernoulli (0/1) vectors, so Cohen's d (assumes continuous,
            # ~normal) is the wrong measure and its 0.2/0.5/0.8 bands do
            # not hold. Cohen's h (arcsine-transformed proportions) is the
            # correct effect size for a difference in rates.
            d = cohens_h(float(np.mean(pred1)), float(np.mean(pred2)))

            # Risk ratio
            events1 = int(np.sum(pred1))
            total1 = len(pred1)
            events2 = int(np.sum(pred2))
            total2 = len(pred2)

            rr, rr_lower, rr_upper = risk_ratio(events1, total1, events2, total2)
            or_val, or_lower, or_upper = odds_ratio(events1, total1, events2, total2)

            # Interpretation.
            # BGL-S2 (2026-09-16): `abs(nan) < 0.2` is False, and so is every
            # later threshold, so an effect size that could NOT be computed fell
            # through this ladder to "large effect", the most alarming verdict
            # the function can produce, about nothing at all. Grade only a
            # finite value; anything else is a could-not-check and must say so.
            if not np.isfinite(d):
                interp = f"not interpretable (effect size is {d}, not a measured value)"
            elif abs(d) < 0.2:
                interp = "negligible effect"
            elif abs(d) < 0.5:
                interp = "small effect"
            elif abs(d) < 0.8:
                interp = "medium effect"
            else:
                interp = "large effect"

            results[f"{g1}_vs_{g2}"] = {
                # Key kept for back-compat; value is now Cohen's h (correct
                # for a rate difference). `cohens_h_positive_rate` is the
                # explicitly-named field going forward.
                "cohens_d_positive_rate": d,
                "cohens_h_positive_rate": d,
                "risk_ratio": (rr, rr_lower, rr_upper),
                "odds_ratio": (or_val, or_lower, or_upper),
                "interpretation": interp,
                "group1_positive_rate": np.mean(pred1),
                "group2_positive_rate": np.mean(pred2),
                "group1_size": total1,
                "group2_size": total2,
            }

    return results


def get_group_metrics_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    confidence_level: float = 0.95,
    method: Literal["auto", "wilson", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, StatisticalResult]]:
    """
    Compute per-group metrics with confidence/credible intervals.

    Automatically selects Bayesian credible intervals for small groups
    (n < 30) and closed-form Wilson score confidence intervals for larger
    groups.

    NOTHING HERE RESAMPLES. Both intervals are closed-form (Wilson algebra,
    Beta posterior quantiles), so this function has no bootstrap and no
    randomness at all. Until audit 6 (S-16b) the signature said otherwise:
    ``method='bootstrap'`` was accepted and silently served a Wilson interval,
    and ``random_state`` was accepted and never read, so two runs with
    different seeds returned byte-identical bounds while the caller believed a
    seeded bootstrap had run. For a genuine bootstrap interval use the
    disparity functions (``demographic_parity_difference_with_ci`` and the
    other ``*_with_ci`` variants), which do resample and do honour
    ``random_state``.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        confidence_level: Confidence/credible level
        method: 'auto' (Beta-posterior credible interval below
            SMALL_SAMPLE_THRESHOLD rows, Wilson score interval at or above it),
            'wilson' (always Wilson), or 'bayesian' (always the Beta
            posterior). ``'bootstrap'`` is REFUSED rather than renamed
            silently: no resampling happens in this function.
        random_state: NOT SUPPORTED, refused when set. There is no resampling
            here to seed, so a seed cannot change any number returned.

    Returns:
        Dict mapping group names to metric StatisticalResults:
        {
            'group_name': {
                'positive_rate': StatisticalResult,
                'tpr': StatisticalResult,
                'fpr': StatisticalResult,
                ...
            },
            ...
        }
        A rate whose OWN denominator is below ``min_group_size`` (fewer than
        that many rows with ``y_true == 1`` for tpr, ``y_true == 0`` for fpr) is
        ABSENT from that group's dict rather than present with an invented
        interval, and a warning names every such omission. An absent key means
        could-not-measure; it never means the rate was fine.

    Example:
        >>> results = get_group_metrics_with_ci(y_true, y_pred, gender)
        >>> for group, metrics in results.items():
        ...     pr = metrics['positive_rate']
        ...     print(f"{group}: {pr.point_estimate:.3f} "
        ...           f"[{pr.lower_bound:.3f}, {pr.upper_bound:.3f}] "
        ...           f"({pr.interval_type.value})")
    """
    if method not in ("auto", "wilson", "bayesian"):
        extra = (
            " No bootstrap runs in this function: 'bootstrap' used to be accepted "
            "and served a closed-form Wilson interval labelled 'wilson_score', "
            "which named a method that never ran. For a bootstrap interval call "
            "demographic_parity_difference_with_ci (or another *_with_ci disparity "
            "function), which resamples and honours random_state."
            if method == "bootstrap"
            else ""
        )
        raise ConfigurationError(
            f"method must be 'auto', 'wilson' or 'bayesian', got {method!r}.{extra}"
        )
    if random_state is not None:
        raise ConfigurationError(
            f"random_state is not supported here, got {random_state!r}: every "
            f"interval this function builds is closed-form (Wilson algebra, Beta "
            f"posterior quantiles), so there is no resampling to seed and the "
            f"argument was never read. Two runs with different seeds returned "
            f"identical bounds. Pass random_state to a *_with_ci disparity "
            f"function (e.g. demographic_parity_difference_with_ci), which does "
            f"bootstrap."
        )

    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr, min_group_size=min_group_size)
    _warn_dropped_groups(gm)
    results = {}
    # group.metric labels for every rate this call could NOT build an interval
    # for, so the omission is named once at the end instead of never.
    withheld: List[str] = []

    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        yt = y_true[mask]
        yp = y_pred[mask]
        n = len(yt)

        # Determine method based on sample size
        use_method = method
        if method == "auto":
            use_method = "bayesian" if n < SMALL_SAMPLE_THRESHOLD else "wilson"

        group_metrics = {}

        # The Bayesian path keeps the posterior CREDIBLE interval but must
        # not report the posterior mean as the point estimate: with a
        # uniform prior an observed 0/20 becomes 0.045, i.e. the report
        # invents positives that were never observed. Report the OBSERVED
        # rate as the point estimate and disclose the posterior mean in
        # metadata.
        def _with_observed_rate(sr, successes, trials):
            sr.metadata = {
                **sr.metadata,
                "posterior_mean": sr.point_estimate,
                "point_estimate_source": "observed_rate",
            }
            sr.point_estimate = float(successes / trials) if trials else float("nan")
            return sr

        # Positive rate
        pos_count = int(np.sum(yp))
        if use_method == "bayesian":
            group_metrics["positive_rate"] = _with_observed_rate(
                bayesian_proportion_ci(pos_count, n, confidence_level), pos_count, n
            )
        else:
            # Wilson score interval: closed-form proportion CI for larger
            # groups (previously NaN bounds with method='point_estimate',
            # contradicting the documented bootstrap CI).
            group_metrics["positive_rate"] = wilson_score_interval(pos_count, n, confidence_level)

        # TPR (only for true positives)
        tp_fn = np.sum(yt == 1)
        if tp_fn >= min_group_size:
            tp = int(np.sum((yp == 1) & (yt == 1)))
            if use_method == "bayesian":
                group_metrics["tpr"] = _with_observed_rate(
                    bayesian_proportion_ci(tp, int(tp_fn), confidence_level), tp, int(tp_fn)
                )
            else:
                group_metrics["tpr"] = wilson_score_interval(tp, int(tp_fn), confidence_level)
        else:
            withheld.append(f"{name}.tpr ({int(tp_fn)} row(s) with y_true=1)")

        # FPR (only for true negatives)
        fp_tn = np.sum(yt == 0)
        if fp_tn >= min_group_size:
            fp = int(np.sum((yp == 1) & (yt == 0)))
            if use_method == "bayesian":
                group_metrics["fpr"] = _with_observed_rate(
                    bayesian_proportion_ci(fp, int(fp_tn), confidence_level), fp, int(fp_tn)
                )
            else:
                group_metrics["fpr"] = wilson_score_interval(fp, int(fp_tn), confidence_level)
        else:
            withheld.append(f"{name}.fpr ({int(fp_tn)} row(s) with y_true=0)")

        results[name] = group_metrics

    # SAY WHICH RATE IS MISSING AND WHY. A rate whose denominator is under the
    # gate is simply LEFT OUT of that group's dict, which is a real third state
    # (the key is absent, so nobody can read a number that was not computed) but
    # it was announced nowhere at all. Measured 2026-09-27 on 40 rows with no
    # positive label in group A and no negative label in group B: A came back
    # {positive_rate, fpr} and B came back {positive_rate, tpr}, zero warnings,
    # so a caller building a TPR table across groups gets one row and no reason
    # for the other. The sibling get_group_metrics reports the same unmeasurable
    # rate as an explicit NaN, so the two public functions disclosed it
    # differently; this closes the gap without inventing an interval.
    if withheld:
        warnings.warn(
            f"get_group_metrics_with_ci: no interval was computed for {withheld} "
            f"because the rate's own denominator is below min_group_size="
            f"{min_group_size}. Those keys are ABSENT from the result: that is "
            f"could-not-measure, not a measured rate and not a pass.",
            UserWarning,
            stacklevel=2,
        )

    return results


def selection_rate_disparity_matrix(
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    confidence_level: float = 0.95,
    missing_strategy: MissingStrategy = "exclude",
) -> Dict[str, object]:
    """Jurisdiction-NEUTRAL selection-rate disparity.

    Computes, for every group: the selection rate, its 95% confidence
    interval, n, and reliability tier; then the full group x group RATIO and
    DIFFERENCE matrices and the worst pair. This is pure descriptive
    statistics -- valid in every jurisdiction and domain.

    It deliberately does **not** apply the US EEOC four-fifths (0.80) rule or
    any other legal bright-line: the four-fifths rule is US-employment-only
    and is *not* law in the EU/UK/CH (which use proportionality + objective
    justification). The legal interpretation of these numbers is a separate,
    jurisdiction-aware overlay applied by the verdict engine -- never baked
    into the metric. (Forest-plot ready: each group carries rate + CI.)

    Args:
        y_pred: Predictions or scores, one per record.
        sensitive_attr: Protected attribute values, one per record.
        confidence_level: Two-sided level for the per-group Wilson interval
            and for the simultaneous bound.
        missing_strategy: What to do with a record that has no protected
            attribute or no prediction. Same three meanings as everywhere else
            in the library (see ``_validation.handle_missing_values``):
            'exclude' (the default) drops those records, so they appear in no
            selection rate and in no denominator; 'as_group' keeps records with
            no protected attribute under the sentinel group ``"__missing__"``
            (records with no PREDICTION are still dropped, because nothing was
            measured for them); 'error' refuses to compute anything while any
            value is missing. Anything else is refused.

    Returns dict:
        groups: [name] sorted by rate desc
        rates: {name: {rate, ci_low, ci_high, n, tier, interpretable}}
        ratio_matrix / difference_matrix: {row: {col: value}} where
            ratio = rate(row)/rate(col), difference = rate(row)-rate(col)
        reference_group: highest-rate interpretable group
        min_ratio / min_ratio_pair: worst adverse-impact ratio + (low, high),
            over the INTERPRETABLE groups only (the same rule the comment at
            that loop states: a tiny group's rate is noise and must not define
            the headline disparity). NaN when fewer than two INTERPRETABLE
            groups are present, i.e. no pair exists that this function is
            willing to read; a 1.0 there would read as "no adverse impact" for
            a test that never ran, and a 0.0 there would be the starkest
            adverse-impact reading the statistic has, taken from rows the
            function itself grades uninterpretable.
        max_difference / max_difference_pair: NaN in that same case.
        headline_basis: which pool the two headline numbers came from.
            'interpretable_groups' (two or more, the normal case),
            'no_interpretable_pair' (a pair exists but fewer than two groups
            are interpretable, so the headline is could-not-check and
            ``headline_over_all_groups`` carries what it would have been), or
            'no_pair' (fewer than two groups at all).
        headline_excluded_groups: the groups the headline did NOT use because
            they are too small to read (interpretable=False); [] when none.
            Their rates are still in ``rates`` and in both matrices.
        headline_over_all_groups: None unless headline_basis is
            'no_interpretable_pair', in which case {min_ratio, min_ratio_pair,
            max_difference, max_difference_pair, uninterpretable_groups}
            computed over EVERY group. It is disclosed, never published as the
            headline: the pairwise ratio_matrix / difference_matrix already
            carry every measured comparison.

    Raises:
        ConfigurationError: If ``missing_strategy`` is not one of the three
            accepted values.
        InvalidDataError: If ``missing_strategy='error'`` and any record has a
            missing protected attribute or a missing prediction.
        ValueError: If ``y_pred`` and ``sensitive_attr`` have different lengths,
            or ``y_pred`` holds labels that cannot be mapped to
            selected/not-selected.
    """
    import pandas as pd

    from ._statistics import _norm_ppf, group_reliability

    # FAIL CLOSED before any row is touched: an unrecognised strategy is a
    # configuration mistake, not a request for the 'exclude' default. Same
    # refusal, same error type, as every other option in the library.
    _check_option(missing_strategy, "missing_strategy", VALID_MISSING_STRATEGIES)

    # Wilson score interval -- closed-form, robust for proportions incl.
    # rates near 0/1 and small n, no special functions (the Beta-ppf path
    # collapsed to a point for large posteriors). This is the CI the Pulse
    # spec explicitly calls for.
    def _wilson(succ: int, n: int, z: float = 1.959963984540054):
        if n <= 0:
            return 0.0, 0.0
        p = succ / n
        d = 1.0 + z * z / n
        c = (p + z * z / (2 * n)) / d
        m = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return max(0.0, c - m), min(1.0, c + m)

    # Robust binary coercion -- predictions may arrive as 'Yes'/'No',
    # True/False, 0/1 or a continuous score. Do NOT route through
    # validate_inputs (it forces y_true to float and crashes on strings).
    s = pd.Series(np.asarray(y_pred).ravel())
    if s.dtype == bool:
        yp = s.astype(int).to_numpy()
    elif pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        u = set(pd.unique(num.dropna()))
        if u and u <= {0, 1}:
            yp = num.fillna(0).astype(int).to_numpy()
        elif num.dropna().nunique() <= 1:
            # Degenerate constant scores: the median equals every score, so
            # the old `>= median` rule marked 100% of EVERY group selected
            # and reported a fake zero-disparity result. No threshold can
            # separate identical scores; disclose it and select nobody.
            warnings.warn(
                "y_pred contains a single constant score value; a median "
                "threshold cannot distinguish selected from not-selected, "
                "so all rows are treated as NOT selected. Selection-rate "
                "disparity is not meaningful for constant scores.",
                UserWarning,
            )
            yp = np.zeros(len(num), dtype=int)
        else:
            thr = float(np.nanmedian(num)) if num.notna().any() else 0.5
            yp = (num.fillna(thr - 1) >= thr).astype(int).to_numpy()
    else:
        low = s.astype("string").str.strip().str.lower()
        truthy = {
            "yes",
            "true",
            "1",
            "y",
            "approved",
            "hired",
            "selected",
            "positive",
            "accept",
            "accepted",
            "pass",
            "invite",
            "invited",
        }
        falsy = {
            "no",
            "false",
            "0",
            "n",
            "rejected",
            "denied",
            "declined",
            "not selected",
            "not hired",
            "not invited",
            "negative",
            "fail",
            "failed",
            "reject",
        }
        # A label that is neither truthy nor falsy (e.g. German 'Ja'/'Nein')
        # used to be silently coerced to 'not selected', producing all-zero
        # selection rates and a fake 'no disparity' verdict. Refuse instead.
        unrecognized = sorted(set(low.dropna()) - truthy - falsy)
        if unrecognized:
            raise ValueError(
                f"y_pred contains labels that cannot be mapped to "
                f"selected/not-selected: {unrecognized}. Recode predictions "
                f"to 0/1 (or recognised English labels) before calling "
                f"selection_rate_disparity_matrix."
            )
        yp = low.isin(truthy).astype(int).to_numpy()
    sens_raw = pd.Series(np.asarray(sensitive_attr).ravel()).astype("string")
    # Mismatched lengths mean misaligned rows; truncating silently would
    # pair predictions with the wrong people. Refuse instead.
    if len(yp) != len(sens_raw):
        raise ValueError(
            f"y_pred and sensitive_attr must have the same length; "
            f"got {len(yp)} and {len(sens_raw)}."
        )
    yp = yp.astype(float)

    # missing_strategy, HONOURED rather than accepted and ignored. The line
    # above used to be fillna("missing"), which is the exact opposite of the
    # declared 'exclude' default: a record with NO protected attribute became a
    # group literally named "missing", which could take the reference_group
    # slot and one half of the worst adverse-impact pair, so the headline
    # disparity was reported about records that carry no protected attribute at
    # all. Measured 2026-09-09 on 40 A + 40 B + 40 attribute-less records:
    # reference_group 'missing', min_ratio 0.211 for the pair ('B', 'missing'),
    # while the only real comparison, A against B, is 0.667.
    # A record whose PREDICTION is missing is unmeasured in the same way: the
    # fillna in the coercion branches above turned it into not-selected and
    # left it in its group's denominator, which drags a measured rate toward
    # zero and can manufacture parity (20 unmeasured of 40 in one group read as
    # rate 0.50 against a measured 1.00, and published min_ratio 1.0).
    missing_pred = pd.isna(s).to_numpy()
    missing_sens = sens_raw.isna().to_numpy()
    if missing_strategy == "error":
        n_missing = int(np.count_nonzero(missing_pred | missing_sens))
        if n_missing:
            raise InvalidDataError(
                f"Found {n_missing} rows with missing values "
                f"({int(np.count_nonzero(missing_sens))} with no protected "
                f"attribute, {int(np.count_nonzero(missing_pred))} with no "
                f"prediction). Use missing_strategy='exclude' or 'as_group' to "
                f"handle them."
            )
    # 'as_group' is the ONLY strategy that keeps an attribute-less record, and
    # it still drops the unmeasured PREDICTIONS, exactly as
    # handle_missing_values keeps the missing attribute and excludes the
    # missing label. Written as "not as_group" rather than "== exclude" so that
    # if the refusal above is ever weakened, 'error' drops those rows instead
    # of carrying pd.NA into np.unique as a phantom group.
    drop_sens = np.zeros_like(missing_sens) if missing_strategy == "as_group" else missing_sens
    drop = missing_pred | drop_sens
    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        warnings.warn(
            f"selection_rate_disparity_matrix: excluded {n_dropped} of {len(yp)} "
            f"row(s) with missing values under "
            f"missing_strategy={missing_strategy!r} "
            f"({int(np.count_nonzero(missing_pred))} with no prediction, "
            f"{int(np.count_nonzero(drop_sens))} with no protected attribute). "
            f"Excluded rows appear in no selection rate and in no denominator.",
            UserWarning,
        )
        keep = ~drop
        yp = yp[keep]
        sens_raw = sens_raw[keep].reset_index(drop=True)
        missing_sens = missing_sens[keep]
    if missing_strategy == "as_group":
        # Same sentinel as _validation.handle_missing_values, and deliberately
        # not a plausible real category value (the old "missing" was).
        sens_raw = sens_raw.where(~missing_sens, "__missing__")
    sens = sens_raw.to_numpy()

    rates: Dict[str, Dict[str, Union[float, int, str, bool]]] = {}
    sizes: Dict[str, int] = {}
    succ_by: Dict[str, int] = {}
    for name in np.unique(sens):
        mask = sens == name
        n = int(mask.sum())
        sizes[name] = n
        succ = int(np.nansum(yp[mask] >= 0.5))
        succ_by[name] = succ
        rate = float(np.nanmean(yp[mask] >= 0.5)) if n else 0.0
        # Honor the REQUESTED confidence level (this used to cap the z-score
        # at the 95% value, so confidence_level=0.99 still produced 95% CIs).
        # Same two-sided z as _statistics.wilson_score_interval.
        z = float(_norm_ppf(1.0 - (1.0 - confidence_level) / 2.0))
        lo, hi = _wilson(succ, n, z)
        rates[name] = {"rate": rate, "ci_low": lo, "ci_high": hi, "n": n}

    tiers = group_reliability(sizes)
    for name in rates:
        t = tiers.get(name, {})
        rates[name]["tier"] = t.get("tier", "invalid")
        rates[name]["interpretable"] = t.get("interpretable", False)

    names = sorted(rates, key=lambda g: float(rates[g]["rate"]), reverse=True)
    ratio_m: Dict[str, Dict[str, float]] = {}
    diff_m: Dict[str, Dict[str, float]] = {}
    for r in names:
        ratio_m[r], diff_m[r] = {}, {}
        for c in names:
            rr, rc = float(rates[r]["rate"]), float(rates[c]["rate"])
            if rc > 0:
                ratio_m[r][c] = float(rr / rc)
            elif rr > 0:
                # A measured positive rate against a measured zero rate: the
                # ratio really is unbounded, which is a finding, not a gap.
                ratio_m[r][c] = float("inf")
            elif r == c:
                # Structural diagonal, not a comparison between two groups: a
                # group against itself is 1.0 whatever its rate.
                ratio_m[r][c] = 1.0
            else:
                # 0/0 between two DIFFERENT groups: nobody was selected on
                # either side, so the selection-rate ratio has no value. 1.0 was
                # the four-fifths all-clear for a division that never happened
                # (measured 2026-09-27, 50 A + 50 B with every prediction 0: the
                # whole ratio_matrix came back 1.0). The difference matrix still
                # carries the measured 0.0, because the RATES were measured and
                # they are genuinely equal; only their ratio is undefined.
                ratio_m[r][c] = float("nan")
            diff_m[r][c] = float(rr - rc)

    # Worst pair over INTERPRETABLE groups only (n>=10) -- a 4-person group's
    # rate is noise and must not define the headline disparity.
    interp = [g for g in names if rates[g]["interpretable"]]

    def _worst_pair(pool: list) -> Tuple[float, object, float, object, bool]:
        """Worst adverse-impact ratio and widest gap inside ONE pool of groups.

        Returns (min_ratio, min_ratio_pair, max_difference, max_difference_pair,
        nobody_selected). Every seed and every comment below is carried over
        verbatim from the single-pool version of this loop; only the pool is now
        a parameter, so the headline (interpretable groups) and the disclosure
        (every group) come from the same arithmetic.
        """
        # With fewer than two groups in the pool no pair is ever formed, so the
        # loop below cannot move these seeds. Seeding them at 1.0 / 0.0 published
        # "no adverse impact, no gap" for a comparison that never happened; seed
        # them unmeasurable instead (see the CONVENTION note at the top of this
        # module). With two or more groups the loop assigns real values, so normal
        # operation is unchanged: the seeds are only ever the answer in the
        # degenerate case.
        _comparable = len(pool) >= 2
        # SECOND WAY THE RATIO SEED CAN SURVIVE UNTOUCHED, and it is not a group
        # count. The loop below only divides when the higher group's rate is > 0,
        # so if NO group in the pool selected anybody, every candidate ratio is
        # 0/0 and the seed is the published answer. Measured 2026-09-27 on 50 A +
        # 50 B with every prediction 0: min_ratio 1.0, i.e. "no adverse impact",
        # for a model that rejected 100 percent of applicants and a division that
        # never ran. The constant-score branch above reaches the same state (it
        # selects nobody by design and says so), and still published 1.0.
        # max_difference is left at its measured 0.0 on purpose: the per-group
        # RATES were measured, and they are equal. Only the ratio is undefined.
        # Cannot be detected from min_ratio_pair being None either: with every
        # rate equal and positive (all-1 predictions) the loop runs, computes 1.0,
        # and still leaves the pair None because nothing beats the seed.
        _any_selected = any(float(rates[g]["rate"]) > 0 for g in pool)
        min_ratio, min_pair = (1.0 if (_comparable and _any_selected) else float("nan")), None
        max_diff, max_pair = (0.0 if _comparable else float("nan")), None
        for hi_g in pool:
            for lo_g in pool:
                if hi_g == lo_g:
                    continue
                rh, rl = float(rates[hi_g]["rate"]), float(rates[lo_g]["rate"])
                if rh > 0:
                    ratio = rl / rh
                    if ratio < min_ratio:
                        min_ratio, min_pair = ratio, (lo_g, hi_g)
                d = rh - rl
                if d > max_diff:
                    max_diff, max_pair = d, (hi_g, lo_g)
        return min_ratio, min_pair, max_diff, max_pair, (_comparable and not _any_selected)

    # THE POOL NO LONGER FALLS BACK TO EVERY GROUP. It used to read
    # `pool = interp if len(interp) >= 2 else names`, directly under the comment
    # above saying a tiny group's rate must not define the headline disparity, so
    # with fewer than two interpretable groups every group came back in and the
    # rule was reversed exactly where it mattered. Measured 2026-09-27 on 50 A at
    # rate 0.8 plus 2 B at rate 0.0, which is ordinary fairness-audit data:
    #
    #   min_ratio 0.0, min_ratio_pair ('B', 'A'), max_difference 0.8, warnings []
    #   rates['B'] = {'n': 2, 'tier': 'invalid', 'interpretable': False}
    #
    # and on 2 A plus 2 B, neither interpretable: min_ratio 0.0, pair ('B', 'A'),
    # max_difference 1.0, warnings []. min_ratio 0.0 is the starkest adverse-
    # impact reading this statistic has, and it was published in silence from two
    # rows the function itself grades 'invalid'.
    # Measured after: min_ratio nan, max_difference nan, headline_basis
    # 'no_interpretable_pair', a UserWarning naming the groups, and the values
    # that used to be the headline preserved under headline_over_all_groups. The
    # per-pair ratio_matrix / difference_matrix are untouched, so no measurement
    # is lost: only the HEADLINE is refused. Unchanged with two interpretable
    # groups (50 A at 0.8 plus 20 B at 0.4 still gives min_ratio 0.5 for the pair
    # ('B', 'A') and max_difference 0.4, in silence).
    min_ratio, min_pair, max_diff, max_pair, _nobody_selected = _worst_pair(interp)
    if _nobody_selected:
        warnings.warn(
            "selection_rate_disparity_matrix: no group has a positive selection "
            "rate (nobody was selected anywhere), so every selection-rate ratio is "
            "0/0 and min_ratio is NaN, not 1.0. This is NOT a finding of no "
            "adverse impact: the ratio could not be computed. The per-group rates "
            "and max_difference ARE measured.",
            UserWarning,
        )

    headline_over_all: Optional[Dict[str, object]] = None
    if len(interp) >= 2:
        headline_basis = "interpretable_groups"
    elif len(names) >= 2:
        headline_basis = "no_interpretable_pair"
        _a_min, _a_min_pair, _a_max, _a_max_pair, _ = _worst_pair(names)
        headline_over_all = {
            "min_ratio": float(_a_min),
            "min_ratio_pair": _a_min_pair,
            "max_difference": float(_a_max),
            "max_difference_pair": _a_max_pair,
            "uninterpretable_groups": [g for g in names if not rates[g]["interpretable"]],
        }
        _small = ", ".join(
            f"{g} (n={int(rates[g]['n'])}, tier={rates[g]['tier']})"
            for g in names
            if not rates[g]["interpretable"]
        )
        warnings.warn(
            f"selection_rate_disparity_matrix: fewer than two groups are "
            f"interpretable ({len(interp)} of {len(names)}; not interpretable: "
            f"{_small}), so no pair of readable selection rates exists and the "
            f"headline min_ratio / max_difference are NaN, which means could not "
            f"check. That is NOT a finding of no adverse impact, and it is not a "
            f"finding of adverse impact either: over every group, including the "
            f"uninterpretable ones, the worst ratio would have been "
            f"{_a_min:.4g} for the pair {_a_min_pair!r} (see "
            f"headline_over_all_groups, and the per-pair ratio_matrix).",
            UserWarning,
        )
    else:
        headline_basis = "no_pair"

    reference = next((g for g in names if rates[g]["interpretable"]), names[0] if names else None)
    # ELFA: distribution-free, multiple-comparison-SIMULTANEOUS bound across
    # every subgroup at once (the audit-grade statement, vs the per-group
    # Wilson CI used for the forest plot). Interpretable groups only.
    try:
        from ._statistics import simultaneous_disparity_bounds

        gc = {g: (succ_by[g], sizes[g]) for g in names if rates[g].get("interpretable")}
        simultaneous = (
            simultaneous_disparity_bounds(gc, confidence_level)
            if len(gc) >= 2
            else {"available": False, "reason": "Need >=2 interpretable groups."}
        )
    except Exception as e:  # noqa: BLE001
        simultaneous = {"available": False, "reason": str(e)}
    return {
        "groups": names,
        "rates": rates,
        "ratio_matrix": ratio_m,
        "difference_matrix": diff_m,
        "reference_group": reference,
        "min_ratio": float(min_ratio),
        "min_ratio_pair": min_pair,
        "max_difference": float(max_diff),
        "max_difference_pair": max_pair,
        "headline_basis": headline_basis,
        "headline_over_all_groups": headline_over_all,
        # WHICH GROUPS THE HEADLINE LEFT OUT, always present (2026-10-01). With two
        # or more interpretable groups the headline is computed over those alone,
        # which is right, but a group excluded as too small to read was visible
        # only by walking `rates` for interpretable=False: 300/200/98 rows plus a
        # 2-row group that selected nobody gave min_ratio 0.56 with nothing at the
        # top level saying the 2-row group existed. [] means nothing was left out.
        "headline_excluded_groups": [g for g in names if not rates[g]["interpretable"]],
        "simultaneous": simultaneous,
    }


# ---------------------------------------------------------------------------
# Confidence-interval variants for the spec-v2 SEALED gates. A sealed gate must
# pass on affirmative CI evidence (one-sided equivalence / TOST on the interval),
# never on a point estimate, so each sealed metric needs a CI. The standard-shape
# metrics delegate to the shared stratified-bootstrap / Bayesian router; the ones
# that need an extra array (strata) bootstrap over row indices.
# ---------------------------------------------------------------------------


def disparate_impact_ratio_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the disparate-impact (four-fifths) ratio. The US-hiring / lending
    seal gates on the CI LOWER bound (``lower_bound >= 0.80``, one-sided equivalence), so a noisy
    point estimate below 0.80 does not fail a model whose interval clears the floor.

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

    Ledger row: disparate_impact_ratio_with_ci. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    return compute_metric_with_ci(
        disparate_impact_ratio,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        # THE ONLY RATIO in this package's CI surface, and therefore the only
        # place the shared default of 0.0 is wrong. A ratio's parity value is 1
        # and its values are bounded in [0, 1], so an interval can never contain
        # 0: ``is_significant`` read off this result was True for EVERY input,
        # including a model with an identical selection rate in every group.
        null_value=1.0,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def fpr_parity_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the false-positive-rate parity gap (predictive equality). The
    EU-hiring / healthcare seal gates on the whole CI being within [-margin, +margin] (TOST
    equivalence), completing the sealed equalized-odds criterion on affirmative evidence.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: fpr_parity_difference_with_ci. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    return compute_metric_with_ci(
        fpr_parity_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def negative_predictive_value_difference_with_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the negative-predictive-value parity gap (a criminal-justice
    sufficiency diagnostic), via the shared stratified-bootstrap / Bayesian router.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: negative_predictive_value_difference_with_ci. See
    docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true, y_pred, sensitive_attr, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )
    return compute_metric_with_ci(
        negative_predictive_value_difference,
        y_true,
        y_pred,
        sensitive_attr,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        min_group_size=min_group_size,
        missing_strategy="exclude",
    )


def conditional_demographic_disparity_with_ci(
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    strata: ArrayLike,
    *,
    min_group_size: int = 30,
    min_cell: int = 5,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    method: Literal["percentile", "basic"] = "basic",
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the conditional demographic disparity (the EU-hiring sealed
    primary). Because the statistic also needs the strata array it bootstraps over row indices,
    stratified by the protected attribute so group sizes are preserved. The seal gates on the CI
    upper bound being within the operator-registered margin (TOST equivalence).

    Register that margin against the definition implemented here, NOT against the CDD
    literature: see the DIVERGENCE note on ``conditional_demographic_disparity``. This is an
    unsigned within-stratum selection-rate spread, not the signed composition-based statistic
    of Wachter, Mittelstadt & Russell (2021), and the two diverge materially on the same data.

    INTERVAL CONSTRUCTION (changed 2026-09-16, BGL-S2). ``method`` defaults to ``'basic'``.
    CDD is a FOLDED max-min spread: it cannot go below zero, so the bootstrap distribution of
    the statistic is shifted upward from the truth, and a percentile interval inherits that
    shift instead of correcting it. ``stratified_bootstrap_ci`` in ``_statistics.py`` has
    documented this since the 2026-08-22 audit ("Prefer 'basic' for folded/nonnegative spread
    statistics: the percentile interval of a max-min spread has 0% coverage at the parity
    null"), and this sealed primary nonetheless hard-coded the percentile interval with no way
    for a caller to override it. Measured at this entry on data at EXACT parity (identical
    selection probability in both groups within every stratum), 40 replications, nominal 95%:
    percentile covered the true 0.0 in 0/40, basic in 37/40. The percentile interval therefore
    returned ``is_significant=True`` and a 95% interval that never touched zero for a model at
    exact parity, at every sample size tried up to n=8000.

    Operational consequence: a margin registered against a percentile upper bound is NOT
    comparable to one registered against the basic upper bound. Re-derive any registered TOST
    margin for this metric, or pass ``method='percentile'`` explicitly to reproduce a sealed
    cell issued before this date.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: conditional_demographic_disparity_with_ci. See
    docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """
    yp = np.asarray(y_pred)
    a = np.asarray(sensitive_attr)
    s = np.asarray(strata)

    def stat(i: np.ndarray) -> float:
        return conditional_demographic_disparity(
            yp[i], a[i], s[i], min_group_size=min_group_size, min_cell=min_cell
        )

    return bootstrap_over_index(
        len(yp),
        stat,
        groups=a,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
    )
