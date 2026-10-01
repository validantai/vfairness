"""
Fairness report generation for vfairness.

This module provides comprehensive fairness assessment reports
with threshold-based evaluation, statistical validation, and visualization.

Statistical Validation Features:
    - Bootstrap confidence intervals for all metrics
    - Bayesian credible intervals for small groups (n < 30)
    - Multiple comparison corrections (Bonferroni, Benjamini-Hochberg)
    - Effect size calculations (Cohen's d, risk ratios)
    - Automatic method selection based on sample size
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np

from ..._methodology import METHODOLOGY_VERSION
from ._grouping import GroupManager
from ._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)
from ._statistics import (
    RECOMMENDED_BOOTSTRAP_SAMPLES,
    SMALL_SAMPLE_THRESHOLD,
    apply_multiple_testing_correction,
    proportion_z_test,
    select_method,
)
from ._validation import ArrayLike, MissingStrategy, validate_inputs
from .report_types import FairnessReport


def _provenance(*intervals: Any) -> Tuple[str, List[str]]:
    """The method that actually RAN, never the one a heuristic recommended.

    ``select_method`` returns a SIZE-BASED RECOMMENDATION and can answer
    "bayesian" or "mixed". The disparity intervals here always run a
    stratified bootstrap: ``compute_metric_with_ci`` has no Bayesian estimator
    (asking for one warns and bootstraps, see the F18 fix), so recording the
    recommendation as ``method_used`` put the name of a statistical method that
    never ran into a report a reader may cite. Measured 2026-09-09: two groups
    of 12 gave ``select_method -> 'bayesian'`` while every interval reported
    ``stratified_bootstrap_fold_debiased``.
    """
    ran = sorted({m for m in (getattr(i, "method", None) for i in intervals) if m})
    if not ran:
        return "unknown", []
    return (ran[0] if len(ran) == 1 else "mixed"), ran


def _within(value: float, threshold: float) -> bool:
    """value <= threshold, robust to float representation noise.

    0.80 - 0.70 evaluates to 0.10000000000000009, so an exact `<=` failed a
    metric sitting EXACTLY on its threshold while 0.30 - 0.20 passed; the
    verdict depended on binary representation, not fairness. Rounding the
    excess to 12 decimals removes representation noise without masking any
    real violation (fairness metrics are never meaningful at 1e-12 scale).
    """
    return round(float(value) - float(threshold), 12) <= 0.0


def _threshold_entry(metric_name: str, value: float, threshold: float) -> Dict[str, Any]:
    """Grade ONE metric against ONE threshold, in that metric's own direction.

    Both report surfaces (classification and regression) route every threshold
    verdict through here, so the two can never drift apart again. They did: the
    classification loop special-cased exactly one NAME ('demographic_parity_ratio')
    for the higher-is-better branch and the regression loop had no ratio branch at
    all, so every other higher-is-better metric was graded as if LOWER were better
    inside the report itself. A ``disparate_impact_ratio`` of 0.00, meaning the
    protected group is never selected, was recorded as PASS.

    The direction is resolved by the shared
    :func:`._metric_direction.metric_direction`, never by a local name test: an
    ad-hoc test here is exactly what produced the disagreement, and a naive
    substring test for "ratio" also matches "cali[bratio]n_difference".

    An unknown direction is COULD-NOT-CHECK, never a pass. We cannot say whether
    a smaller value is better or worse, so we cannot say the metric passed; it is
    returned as NOT_ASSESSABLE and is excluded from the fairness score rather
    than silently counted as a metric within threshold.

    Returns:
        A verdict entry with ``status`` in ``{"PASS", "FAIL", "NOT_ASSESSABLE"}``.
        NOT_ASSESSABLE entries carry ``threshold: None`` and a ``reason``, matching
        the shape of the other not-assessable records in this module.
    """
    # NaN fails every comparison, so an unguarded `value > threshold` reports
    # "not breached" and an unguarded `_within` reports FAIL, for a metric that
    # was never measured. Both loops below already route NaN out before calling
    # here; this guard keeps a DIRECT caller from getting a fairness violation
    # for a metric nobody could compute.
    if value != value or threshold != threshold:  # NaN test, also for np scalars
        return {
            "metric": metric_name,
            "value": value,
            "threshold": None,
            "status": "NOT_ASSESSABLE",
            "reason": "metric is undefined (NaN) for this data; excluded from the verdict",
        }

    direction = metric_direction(metric_name)

    if direction is MetricDirection.UNKNOWN:
        return {
            "metric": metric_name,
            "value": value,
            "threshold": None,
            "status": "NOT_ASSESSABLE",
            "reason": (
                f"could not check: no known better-direction for {metric_name} "
                f"(is a lower value better or worse?), so the bound {threshold} "
                f"could not be applied; reported as not assessable rather than "
                f"as a metric within threshold"
            ),
        }

    # A BOUND THAT CANNOT BE BREACHED GRADES NOTHING, and the shared rule owns
    # that question. READINESS-6, 2026-09-10: measured on this repo, this
    # function DISAGREED with `check_threshold` and `_fairness_verdict` on the
    # identical input, in both directions at once.
    #
    #   _threshold_entry("demographic_parity_ratio", 0.00, 0.0)   -> PASS
    #   check_threshold  (same input)                             -> COULD_NOT_CHECK
    #   _fairness_verdict(same input)                             -> insufficient_evidence
    #
    # A required MINIMUM of 0.0 is met by every possible value, the worst one
    # included, so nothing was graded. In the report that PASS was not cosmetic:
    # on a model that NEVER selects one group (ratio 0.00) it entered
    # passed_metrics and lifted fairness_score from 0.00 to 0.25.
    #
    # And the OVER-CORRECTION already present here, fixed in the same edit: a
    # NEGATIVE maximum returned FAIL, where the shared rule returns
    # COULD_NOT_CHECK. No magnitude can meet a negative bound either, so the
    # value was never compared against a bound that could hold, and a measured
    # FAIL overstates what happened. This function was STRICTER than the shared
    # rule there and MORE PERMISSIVE on the mirror case; it is neither now.
    #
    # The comparison below stays local because it must: `_within` carries the
    # float-representation tolerance (see its docstring) that a metric sitting
    # EXACTLY on its threshold depends on, which a plain `<` does not.
    outcome, message = check_threshold(metric_name, value, threshold)
    if outcome is ThresholdOutcome.COULD_NOT_CHECK:
        return {
            "metric": metric_name,
            "value": value,
            "threshold": None,
            "status": "NOT_ASSESSABLE",
            "reason": message,
        }

    if direction is MetricDirection.HIGHER_IS_BETTER:
        # Parity ratios pass at or ABOVE the bound (the four-fifths rule).
        is_pass = _within(threshold, value)
    else:
        # Violation magnitudes pass at or BELOW the bound.
        is_pass = _within(abs(value), threshold)

    return {
        "metric": metric_name,
        "value": value,
        "threshold": threshold,
        "status": "PASS" if is_pass else "FAIL",
    }


def _insufficient_evidence_groups(gm: "GroupManager") -> List[Dict[str, Any]]:
    """Surface groups excluded from the metric verdict as first-class evidence.

    Groups whose size is below ``min_group_size`` are dropped from every
    disparity metric (GroupManager only computes over valid groups). That
    drop is warned about at call time, but a downstream verdict consumer that
    reads only the machine-readable ``assessment`` block never saw it, so an
    excluded protected stratum could be treated as an implicit pass. An
    audit-evidence tool must never silently drop a stratum from the verdict.

    This routes every excluded group through the shared reliability tiers
    (invalid n<10 / underpowered 10..29, both below the default gate) and
    returns an explicit "insufficient evidence" record per excluded group so
    the verdict carries the caveat rather than swallowing it. Purely additive:
    it does not change which groups are computed, only what the report
    discloses.

    Returns:
        A list (possibly empty) of
        ``{group, n, tier, verdict: 'insufficient_evidence', reason}`` records,
        sorted by group name for deterministic output.
    """
    from ._statistics import group_reliability

    excluded = gm.get_invalid_groups()
    if not excluded:
        return []
    sizes = gm.get_group_sizes()
    tiers = group_reliability({name: sizes[name] for name in excluded})
    records: List[Dict[str, Any]] = []
    for name in sorted(excluded):
        info = tiers.get(name, {})
        records.append(
            {
                "group": name,
                "n": int(sizes[name]),
                "tier": info.get("tier", "invalid"),
                "verdict": "insufficient_evidence",
                "reason": (
                    f"n={sizes[name]} is below min_group_size="
                    f"{gm.min_group_size}; this group is excluded from the "
                    f"disparity metrics, so there is insufficient evidence to "
                    f"assess it. {info.get('note', '')}".strip()
                ),
            }
        )
    return records


def _missing_data_clause(info: Dict[str, Any]) -> str:
    """Render the missing-data provenance that must travel WITH the verdict.

    Two runs over the same data with different ``missing_strategy`` values can
    reach different verdicts. The structured provenance has always been in
    ``report["data_info"]`` (``original_size``, ``final_size``, ``n_excluded``,
    ``missing_strategy``), and that stays the machine-readable source of truth.
    What was missing is disclosure on the VERDICT surface: the summary line, which
    is what a reader sees and what most downstream consumers render, said
    "3/5 metrics within thresholds" identically whether 0 or 15 rows had been
    silently dropped. A report that excluded rows without saying so cannot be
    audited, because nothing in it explains why two runs disagree.

    The clause is emitted on EVERY report, never only when rows were excluded. If
    it appeared only on exclusion, its absence would be ambiguous: an auditor
    could not tell "nothing was excluded" from "this version did not record it",
    which is exactly the absence-of-evidence reading provenance exists to remove.

    A field that is genuinely not there prints ``unknown``, never ``0``: reporting
    zero exclusions for a run that never recorded any is a false statement about
    provenance, not a harmless default.

    Returns:
        A parenthesised clause to append to the summary. Always non-empty, and
        free of em/en-dashes (the summary is rasterised into a report panel).
    """

    def _count(key: str) -> str:
        value = info.get(key)
        return "unknown" if value is None else str(value)

    strategy = info.get("missing_strategy")
    strategy_text = "unknown" if strategy is None else repr(str(strategy))

    # THE SIZE GATE IS AN EXCLUSION TOO, and it used to be invisible here.
    # `n_excluded` counts only rows dropped for MISSING VALUES. A group removed
    # by min_group_size keeps its rows in `final_size` while contributing to no
    # metric at all, so a run could state "2302 of 2302 rows assessed, 0
    # excluded" while 1102 rows in 50 small groups had left every measurement.
    # That clause is precisely what a careful reader consults to decide whether
    # a verdict covers their population, so stating the opposite is worse than
    # staying silent. Measured 2026-09-10 on a model that denied one class
    # outright: the class was split across 50 levels of about 22 rows, every one
    # below the default floor of 30.
    #
    # Counted from `invalid_groups` and `group_sizes`, which the block already
    # carries. Unknown rather than 0 when either is absent, for the same reason
    # the counts above print unknown: a number nobody recorded is not a zero.
    invalid = info.get("invalid_groups")
    sizes = info.get("group_sizes")
    if isinstance(invalid, (list, tuple)) and isinstance(sizes, dict):
        gated_rows = sum(
            int(sizes[g]) for g in invalid if g in sizes and isinstance(sizes[g], (int, np.integer))
        )
        gated_text = f"{gated_rows} in {len(invalid)} group(s)"
    else:
        gated_text = "unknown"

    return (
        f" (data provenance: {_count('final_size')} of {_count('original_size')} "
        f"rows assessed, {_count('n_excluded')} excluded for missing values, "
        f"{gated_text} withheld by the group-size floor, "
        f"missing_strategy={strategy_text})"
    )


def _metric_proportion_p_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gm: "GroupManager",
) -> "tuple[List[str], List[float], Dict[str, List[str]]]":
    """Real two-sided p-values for the report's proportion-gap metrics.

    For each metric, the underlying per-group proportion is computed for every
    valid group and the extreme (max vs min) pair is tested with a
    two-proportion z-test. Metrics whose underlying quantity is not a
    proportion (e.g. calibration error) are not tested.

    ONE FAMILY MEMBER PER DISTINCT HYPOTHESIS. Several report metrics are
    different renderings of the SAME underlying comparison, and correcting for
    one hypothesis more than once suppresses real findings.
    ``demographic_parity_difference`` and ``demographic_parity_ratio`` are the
    gap and the ratio of the identical pair of group selection rates, so they
    were handed the identical ``_extreme_pair_p(dp)``;
    ``equalized_odds_difference`` reuses whichever of the TPR or FPR p-values
    belongs to its larger gap, so it duplicates ``equal_opportunity_difference``
    whenever the TPR gap is the larger one.

    Measured 2026-09-10 on 1000 rows (seed 85): the family entering the
    correction was m=5 while the distinct hypotheses actually tested were 3
    (selection-rate gap, TPR gap, PPV gap); the p-vector was
    ``[0.016225, 0.016225, 0.037311, 0.604939, 0.037311]``. Under Bonferroni at
    m=5 nothing was rejected (0.0811); at the honest m=3 the real
    demographic-parity finding IS rejected (0.048675). Two copies of one
    hypothesis buried it.

    Returns:
        ``(names, p_values, shared)``. ``names`` and ``p_values`` are aligned and
        hold ONE entry per distinct hypothesis, named by the first report metric
        that tests it. ``shared`` maps that representative metric name to the
        other report metrics that resolve to the same hypothesis, so the
        de-duplication is disclosed rather than silent.
    """

    def _group_props(numer_fn) -> "List[tuple[float, int]]":
        props = []
        for g in gm.get_valid_groups(warn_if_empty=False):
            mask = gm.get_mask(g)
            sel, tot = numer_fn(mask)
            if tot >= 2:
                props.append((float(np.sum(sel)) / float(tot), int(tot)))
        return props

    def _extreme_pair_p(props: "List[tuple[float, int]]") -> Optional[float]:
        if len(props) < 2:
            return None
        hi = max(props, key=lambda t: t[0])
        lo = min(props, key=lambda t: t[0])
        return proportion_z_test(hi[0], hi[1], lo[0], lo[1])

    # Per-group proportions underlying each metric family
    dp = _group_props(lambda m: (y_pred[m] == 1, int(np.sum(m))))
    tpr = _group_props(lambda m: (y_pred[m & (y_true == 1)] == 1, int(np.sum(m & (y_true == 1)))))
    fpr = _group_props(lambda m: (y_pred[m & (y_true == 0)] == 1, int(np.sum(m & (y_true == 0)))))
    ppv = _group_props(lambda m: (y_true[m & (y_pred == 1)] == 1, int(np.sum(m & (y_pred == 1)))))

    dp_p, tpr_p, fpr_p, ppv_p = (
        _extreme_pair_p(dp),
        _extreme_pair_p(tpr),
        _extreme_pair_p(fpr),
        _extreme_pair_p(ppv),
    )
    # Equalized odds: test the larger of the TPR / FPR gaps. Which one it picks
    # decides which hypothesis it duplicates, so the KEY is chosen alongside it.
    tpr_gap = (max(p for p, _ in tpr) - min(p for p, _ in tpr)) if len(tpr) >= 2 else -1
    fpr_gap = (max(p for p, _ in fpr) - min(p for p, _ in fpr)) if len(fpr) >= 2 else -1
    eo_p, eo_key = (tpr_p, "tpr_gap") if tpr_gap >= fpr_gap else (fpr_p, "fpr_gap")

    # (report metric name, the hypothesis it actually tests, its p-value)
    candidates: "List[tuple[str, str, Optional[float]]]" = [
        ("demographic_parity_difference", "selection_rate_gap", dp_p),
        ("demographic_parity_ratio", "selection_rate_gap", dp_p),
        ("equal_opportunity_difference", "tpr_gap", tpr_p),
        ("predictive_parity_difference", "ppv_gap", ppv_p),
        ("equalized_odds_difference", eo_key, eo_p),
    ]

    names: List[str] = []
    p_values: List[float] = []
    shared: Dict[str, List[str]] = {}
    representative: Dict[str, str] = {}
    for metric_name, hypothesis, p in candidates:
        if p is None:
            continue
        if hypothesis in representative:
            # Same comparison, second rendering. It is disclosed, not corrected
            # for again. See the docstring: this is the whole fix.
            shared.setdefault(representative[hypothesis], []).append(metric_name)
            continue
        representative[hypothesis] = metric_name
        names.append(metric_name)
        p_values.append(p)
    return names, p_values, shared


def classification_fairness_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    y_prob: Optional[ArrayLike] = None,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    thresholds: Optional[Dict[str, float]] = None,
    include_ci: bool = False,
    include_group_analysis: bool = False,
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    ci_method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    multiple_testing_correction: Literal["bonferroni", "fdr", "none"] = "none",
    random_state: Optional[int] = None,
) -> FairnessReport:
    """
    Generate a comprehensive fairness report for classification.

    Computes all relevant fairness metrics, evaluates them against
    thresholds, and provides a structured assessment with optional
    statistical validation.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        y_prob: Optional predicted probabilities
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        thresholds: Custom thresholds for each metric
        include_ci: Whether to compute confidence/credible intervals
        include_group_analysis: Whether to include detailed group privilege analysis
                               (identifies most/least advantaged groups)
        n_bootstrap: Number of bootstrap samples (≥5000 recommended)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        ci_method: 'auto' (select based on sample size), 'bootstrap', or 'bayesian'
        multiple_testing_correction: 'bonferroni', 'fdr', or 'none'
        random_state: Random seed for reproducibility

    Returns:
        Dict containing:
            - metrics: Computed fairness metrics (point estimates)
            - metrics_with_ci: Metrics with confidence intervals (if include_ci=True)
            - group_stats: Per-group statistics
            - group_analysis: Privilege/disadvantage analysis (if include_group_analysis=True)
            - effect_sizes: Pairwise effect size comparisons
            - assessment: Pass/fail evaluation
            - statistical_validation: Method selection and corrections info
            - data_info: Data summary

    Example:
        >>> report = classification_fairness_report(
        ...     y_true, y_pred, gender,
        ...     include_ci=True, n_bootstrap=5000
        ... )
        >>> # fairness_score is None when nothing was assessable: check before formatting.
        >>> score = report['assessment']['fairness_score']
        >>> print(f"Fairness Score: {score:.1%}" if score is not None else "Fairness Score: could not check")
        >>>
        >>> # Access confidence intervals
        >>> if 'metrics_with_ci' in report:
        ...     dp_ci = report['metrics_with_ci']['demographic_parity_difference']
        ...     print(f"DP: {dp_ci['point_estimate']:.3f} "
        ...           f"[{dp_ci['lower_bound']:.3f}, {dp_ci['upper_bound']:.3f}]")
        >>>
        >>> # Access group analysis
        >>> report = classification_fairness_report(
        ...     y_true, y_pred, gender, include_group_analysis=True
        ... )
        >>> if 'group_analysis' in report:
        ...     priv = report['group_analysis']['privileged_group']
        ...     disadv = report['group_analysis']['disadvantaged_group']
        ...     print(f"Most advantaged: {priv['group']} ({priv['positive_rate']:.1%})")
        ...     print(f"Most disadvantaged: {disadv['group']} ({disadv['positive_rate']:.1%})")

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

    Ledger row: classification_fairness_report. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    from ...post_processing.calibration import (
        integrated_calibration_index,
        multicalibration,
    )
    from .classification import (
        auroc_parity,
        calibration_difference,
        compute_effect_sizes,
        demographic_parity_difference,
        demographic_parity_difference_with_ci,
        demographic_parity_ratio,
        equal_opportunity_difference,
        equal_opportunity_difference_with_ci,
        equalized_odds_difference,
        equalized_odds_difference_with_ci,
        get_group_metrics,
        predictive_parity_difference,
    )

    # Validate inputs
    y_true_v, y_pred_v, sensitive_attr_v, y_prob_v, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        y_prob,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    # Default thresholds
    default_thresholds = {
        "demographic_parity_difference": 0.10,
        "demographic_parity_ratio": 0.80,  # 80% rule (min acceptable)
        "equalized_odds_difference": 0.10,
        "equal_opportunity_difference": 0.05,
        "predictive_parity_difference": 0.05,
        "calibration_difference": 0.05,
        "multicalibration": 0.03,
        "integrated_calibration_index": 0.03,
        "auroc_parity": 0.05,
    }
    if thresholds:
        default_thresholds.update(thresholds)

    # Compute metrics
    metrics = {
        "demographic_parity_difference": demographic_parity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "equal_opportunity_difference": equal_opportunity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "predictive_parity_difference": predictive_parity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
    }

    if y_prob_v is not None:
        metrics["calibration_difference"] = calibration_difference(
            y_true_v, y_pred_v, sensitive_attr_v, y_prob_v, min_group_size=min_group_size
        )
        # spec-v2 spine sufficiency + validity metrics (probability-only). multicalibration is
        # the EU-healthcare sealed PRIMARY (worst-subgroup calibration violation); the ICI gap is
        # the grid-free calibration disparity; auroc_parity is the discrimination validity gate.
        metrics["multicalibration"] = multicalibration(
            y_true_v, y_prob_v, sensitive_attr_v, min_group_size=min_group_size
        ).alpha
        metrics["integrated_calibration_index"] = integrated_calibration_index(
            y_true_v, y_prob_v, sensitive_attr_v, min_group_size=min_group_size
        ).ici_disparity
        metrics["auroc_parity"] = auroc_parity(
            y_true_v, y_prob_v, sensitive_attr_v, min_group_size=min_group_size
        )

    # Get group statistics
    group_stats = get_group_metrics(
        y_true_v, y_pred_v, sensitive_attr_v, y_prob_v, min_group_size=min_group_size
    )

    # Group manager for additional info
    gm = GroupManager(sensitive_attr_v, min_group_size=min_group_size)

    # Degenerate data guard: with fewer than two valid groups (or no samples) no
    # between-group comparison ever happens, so every disparity metric is vacuous
    # and must not be reported as a pass. Computed BEFORE the threshold loop so it
    # can gate the loop, rather than being a caveat bolted on afterwards that a
    # consumer reading passed_metrics would never see.
    n_valid_groups = len(gm.get_valid_groups(warn_if_empty=False))
    assessable = bool(info.get("final_size", 0) > 0 and n_valid_groups >= 2)

    # Evaluate against thresholds
    passed = []
    failed = []
    not_assessable = []

    for metric_name, value in metrics.items():
        # Nothing was comparable, so no metric may enter passed_metrics. The
        # metrics themselves return NaN in this case, but this guard does not
        # depend on that: a metric that still produced a finite number with one
        # group did not measure a DISPARITY, and routing it to "passed" is how a
        # vacuous run certified fairness_score 1.0.
        if not assessable:
            not_assessable.append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": None,
                    "status": "NOT_ASSESSABLE",
                    "reason": (
                        f"{n_valid_groups} valid group(s) after filtering (need at "
                        f"least 2); no between-group comparison was performed, so "
                        f"this metric certifies nothing"
                    ),
                }
            )
            continue

        # A metric that could not be computed (e.g. equal_opportunity_difference
        # when a group has no positive labels, so TPR is undefined) returns NaN.
        # _within(nan, t) is False, which would wrongly count it as a FAIL; route
        # it to not_assessable so it is excluded from the fairness score, never
        # reported as a fairness violation.
        if isinstance(value, float) and value != value:  # NaN
            not_assessable.append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": None,
                    "status": "NOT_ASSESSABLE",
                    "reason": "metric is undefined (NaN) for this data; excluded from the verdict",
                }
            )
            continue

        threshold = default_thresholds.get(metric_name, 0.1)

        # Direction comes from the shared resolver, never from a local name test
        # here. See _threshold_entry.
        entry = _threshold_entry(metric_name, value, threshold)

        if entry["status"] == "PASS":
            passed.append(entry)
        elif entry["status"] == "FAIL":
            failed.append(entry)
        else:
            not_assessable.append(entry)

    # Compute fairness score over ASSESSABLE metrics only (NaN/undefined metrics
    # are excluded, not counted as failures). With NO assessable metric there is
    # no fraction to report: the score is None ("could not check"), never a
    # number. It used to be 0.0 here and 1.0 in the degenerate case above, i.e.
    # the two opposite lies about the same absence of evidence. A None score is
    # the reason ``fairness_score`` is Optional in report_types.AssessmentReport;
    # consumers must not format it as a percentage without checking.
    total_metrics = len(passed) + len(failed)
    passed_count = len(passed)
    fairness_score = passed_count / total_metrics if total_metrics > 0 else None

    # Groups dropped from every metric because they are below min_group_size.
    # Surface them as an explicit "insufficient evidence" verdict so a
    # downstream consumer reading only this assessment block never mistakes a
    # silently excluded protected stratum for an implicit pass.
    insufficient_evidence = _insufficient_evidence_groups(gm)
    if not assessable:
        summary_text = (
            f"NOT ASSESSABLE: {n_valid_groups} valid group(s) after filtering "
            f"(need at least 2): disparity metrics are vacuous and do not "
            f"certify fairness."
        )
    elif total_metrics == 0:
        # Two or more groups, yet not one metric was computable. "0/0 metrics
        # within thresholds" read as a clean bill of health for a run that
        # measured nothing at all.
        summary_text = (
            "NOT ASSESSABLE: no metric could be computed on this data; "
            "nothing here certifies fairness."
        )
    else:
        summary_text = f"{passed_count}/{total_metrics} metrics within thresholds"
    if insufficient_evidence:
        summary_text += (
            f" ({len(insufficient_evidence)} group(s) with insufficient "
            f"evidence, excluded from the verdict: "
            f"{[g['group'] for g in insufficient_evidence]})"
        )
    if not_assessable:
        # The reason is per-entry (undefined/NaN, or too few comparable groups),
        # so do not hard-code one of them in the summary.
        summary_text += (
            f" ({len(not_assessable)} metric(s) not assessable, excluded from the verdict)"
        )
    # Missing-data provenance travels with the verdict, always. See
    # _missing_data_clause: the counts and the strategy stay machine-readable in
    # data_info below, and this states them where the verdict itself is read.
    summary_text += _missing_data_clause(info)

    # Build result dict
    result = {
        "task_type": "classification",
        # Stamp the methodology version here too (not only in the analyzer
        # wrapper), so a rating built by calling this report function directly
        # still records which methodology produced it (VB-DOC-7).
        "methodology_version": METHODOLOGY_VERSION,
        "metrics": metrics,
        "group_stats": group_stats,
        "assessment": {
            "fairness_score": fairness_score,
            "assessable": assessable,
            "passed_metrics": passed,
            "failed_metrics": failed,
            "not_assessable_metrics": not_assessable,
            "insufficient_evidence_groups": insufficient_evidence,
            "summary": summary_text,
        },
        "data_info": {
            **info,
            "n_samples": info["final_size"],  # Alias for convenience
            "n_groups": gm.n_groups,
            "valid_groups": gm.get_valid_groups(),
            "invalid_groups": gm.get_invalid_groups(),
            "group_sizes": gm.get_group_sizes(),
            "is_intersectional": gm.is_intersectional,
        },
        "thresholds_used": default_thresholds,
    }

    # Add statistical validation if requested
    if include_ci:
        # Determine method based on sample sizes
        group_sizes = gm.get_group_sizes()
        overall_method, per_group_methods = select_method(group_sizes)

        # Compute metrics with confidence intervals
        metrics_with_ci = {}

        dp_ci = demographic_parity_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["demographic_parity_difference"] = dp_ci.to_dict()

        eo_ci = equalized_odds_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["equalized_odds_difference"] = eo_ci.to_dict()

        eop_ci = equal_opportunity_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["equal_opportunity_difference"] = eop_ci.to_dict()

        result["metrics_with_ci"] = metrics_with_ci

        # Compute effect sizes
        effect_sizes = compute_effect_sizes(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        )
        result["effect_sizes"] = effect_sizes

        # Statistical validation info
        method_used, methods_that_ran = _provenance(dp_ci, eo_ci, eop_ci)
        result["statistical_validation"] = {
            # What RAN. `size_based_recommendation` below is what select_method
            # SUGGESTED from the group sizes, which is a different question and
            # was previously reported here as though it were the answer.
            "method_used": method_used,
            "methods_that_ran": methods_that_ran,
            "size_based_recommendation": overall_method,
            "per_group_size_recommendation": per_group_methods,
            "per_group_methods": per_group_methods,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        }

        # Multiple testing correction over REAL per-metric p-values.
        # Each metric here is a gap between group proportions, so we test the
        # extreme (max vs min rate) group pair with a two-proportion z-test.
        # The previous pseudo p-value min(1, value/threshold) was INVERTED:
        # grossly unfair metrics capped at p=1.0 (never significant) while
        # fair metrics got small p-values.
        if multiple_testing_correction != "none" and len(metrics) > 1:
            tested_names, real_p_values, shared_hypotheses = _metric_proportion_p_values(
                y_true_v, y_pred_v, gm
            )
            if len(real_p_values) > 1:
                correction_result = apply_multiple_testing_correction(
                    np.array(real_p_values), method=multiple_testing_correction
                )
                correction_dict = correction_result.to_dict()
                correction_dict["tested_metrics"] = tested_names
                # What was FOLDED IN rather than corrected for a second time.
                # Empty when no two report metrics resolved to the same
                # comparison; never absent, so its absence cannot be read as
                # "this version did not record it". See
                # _metric_proportion_p_values.
                correction_dict["metrics_sharing_a_tested_hypothesis"] = shared_hypotheses
                correction_dict["p_value_method"] = (
                    "two_proportion_z_test on the extreme (max vs min rate) group pair per "
                    "DISTINCT hypothesis; report metrics that are different renderings of "
                    "the same comparison are listed in metrics_sharing_a_tested_hypothesis "
                    "and enter the family once"
                )
                result["multiple_testing_correction"] = correction_dict

    # Add group analysis if requested
    if include_group_analysis:
        from .intersectional import identify_privileged_groups

        group_analysis = identify_privileged_groups(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            missing_strategy="exclude",  # Already validated
        )

        # Convert GroupAdvantage objects to dicts
        result["group_analysis"] = {
            "privileged_group": group_analysis["privileged_group"].to_dict()
            if group_analysis["privileged_group"]
            else None,
            "disadvantaged_group": group_analysis["disadvantaged_group"].to_dict()
            if group_analysis["disadvantaged_group"]
            else None,
            "all_groups": [g.to_dict() for g in group_analysis["all_groups"]],
            "overall_rate": group_analysis["overall_rate"],
            "max_disparity": group_analysis["max_disparity"],
            "disparity_severity": group_analysis["disparity_severity"],
        }

    return cast(FairnessReport, result)


def regression_fairness_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
    thresholds: Optional[Dict[str, float]] = None,
    include_ci: bool = False,
    n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    ci_method: Literal["auto", "bootstrap", "bayesian"] = "auto",
    multiple_testing_correction: Literal["bonferroni", "fdr", "none"] = "none",
    random_state: Optional[int] = None,
) -> FairnessReport:
    """
    Generate a comprehensive fairness report for regression.

    Computes all relevant regression fairness metrics, evaluates
    them against thresholds, and provides a structured assessment
    with optional statistical validation.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sensitive_attr: Protected attribute values
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values
        thresholds: Custom thresholds for each metric
        include_ci: Whether to compute confidence/credible intervals
        n_bootstrap: Number of bootstrap samples (≥5000 recommended)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        ci_method: 'auto' (select based on sample size), 'bootstrap', or 'bayesian'
        multiple_testing_correction: 'bonferroni', 'fdr', or 'none'
        random_state: Random seed for reproducibility

    Returns:
        Dict containing:
            - metrics: Computed fairness metrics (point estimates)
            - metrics_with_ci: Metrics with confidence intervals (if include_ci=True)
            - group_stats: Per-group statistics
            - effect_sizes: Pairwise effect size comparisons
            - assessment: Pass/fail evaluation. assessment["assessable"] is
              False when fewer than two valid groups remain after filtering
              (or no samples survive validation); the parity metrics are then
              vacuous and the summary says NOT ASSESSABLE instead of
              certifying fairness.
            - statistical_validation: Method selection info
            - data_info: Data summary

    Example:
        >>> report = regression_fairness_report(
        ...     y_true, y_pred, region,
        ...     include_ci=True, n_bootstrap=5000
        ... )
        >>> # fairness_score is None when nothing was assessable: check before formatting.
        >>> score = report['assessment']['fairness_score']
        >>> print(f"Fairness Score: {score:.1%}" if score is not None else "Fairness Score: could not check")
    """
    from .regression import (
        compute_regression_effect_sizes,
        get_group_metrics,
        mae_parity_difference,
        mae_parity_difference_with_ci,
        mean_prediction_difference,
        mean_prediction_difference_with_ci,
        r2_parity_difference,
        residual_bias,
        rmse_parity_difference,
        rmse_parity_difference_with_ci,
    )

    # Validate inputs
    y_true_v, y_pred_v, sensitive_attr_v, _, info = validate_inputs(
        y_true, y_pred, sensitive_attr, task_type="regression", missing_strategy=missing_strategy
    )

    # Compute std for relative thresholds
    y_std = np.std(y_true_v)

    # Constant y_true makes every scale-relative threshold 0.0, so ANY
    # nonzero parity gap (even 1e-9 noise) was marked FAIL. There is no
    # data scale to assess against, so those metrics are reported as
    # NOT_ASSESSABLE instead of silently failing, unless the caller
    # supplied an explicit absolute threshold.
    degenerate_scale = bool(y_std <= 1e-12)
    scale_relative_metrics = {
        "mae_parity_difference",
        "rmse_parity_difference",
        "mean_prediction_difference",
    }
    if degenerate_scale:
        warnings.warn(
            "y_true is constant (std=0): scale-relative fairness thresholds "
            "are undefined. Affected metrics are reported as NOT_ASSESSABLE; "
            "pass explicit absolute thresholds to assess them.",
            UserWarning,
        )

    # Default thresholds (relative to data scale)
    default_thresholds = {
        "mae_parity_difference": 0.15 * y_std,  # 15% of std
        "rmse_parity_difference": 0.15 * y_std,
        "mean_prediction_difference": 0.10 * y_std,
        "r2_parity_difference": 0.10,
    }
    explicit_thresholds = set(thresholds.keys()) if thresholds else set()
    if thresholds:
        default_thresholds.update(thresholds)

    # Compute metrics
    metrics = {
        "mae_parity_difference": mae_parity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "rmse_parity_difference": rmse_parity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "mean_prediction_difference": mean_prediction_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
        "r2_parity_difference": r2_parity_difference(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        ),
    }

    # Get additional info
    bias_by_group = residual_bias(
        y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
    )
    group_stats = get_group_metrics(
        y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
    )

    gm = GroupManager(sensitive_attr_v, min_group_size=min_group_size)

    # Degenerate data guard (mirror of the classification report): with fewer
    # than two valid groups (or no samples) no between-group comparison ever
    # happens, so no parity metric may be reported as a pass. Computed BEFORE the
    # threshold loop so it gates the loop rather than only annotating the summary.
    n_valid_groups = len(gm.get_valid_groups(warn_if_empty=False))
    assessable = bool(info.get("final_size", 0) > 0 and n_valid_groups >= 2)

    # Evaluate against thresholds
    passed = []
    failed = []
    not_assessable = []

    for metric_name, value in metrics.items():
        # Nothing was comparable: no metric may enter passed_metrics. See the
        # matching guard in classification_fairness_report.
        if not assessable:
            not_assessable.append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": None,
                    "status": "NOT_ASSESSABLE",
                    "reason": (
                        f"{n_valid_groups} valid group(s) after filtering (need at "
                        f"least 2); no between-group comparison was performed, so "
                        f"this metric certifies nothing"
                    ),
                }
            )
            continue

        if (
            degenerate_scale
            and metric_name in scale_relative_metrics
            and metric_name not in explicit_thresholds
        ):
            not_assessable.append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": None,
                    "status": "NOT_ASSESSABLE",
                    "reason": (
                        "y_true is constant (std=0); the scale-relative threshold is undefined"
                    ),
                }
            )
            continue

        # A metric that could not be computed (e.g. r2_parity_difference when a
        # group has constant y_true, so R2 is undefined) returns NaN.
        # _within(nan, t) is False, which would wrongly count it as a FAIL; route
        # it to not_assessable so it is excluded from the fairness score.
        if isinstance(value, float) and value != value:  # NaN
            not_assessable.append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": None,
                    "status": "NOT_ASSESSABLE",
                    "reason": "metric is undefined (NaN) for this data; excluded from the verdict",
                }
            )
            continue

        threshold = default_thresholds.get(metric_name, 0.1)

        # Same shared direction resolver as the classification loop. This site
        # previously had NO ratio branch at all, so a caller supplying a
        # higher-is-better metric here was graded backwards. See _threshold_entry.
        entry = _threshold_entry(metric_name, value, threshold)

        if entry["status"] == "PASS":
            passed.append(entry)
        elif entry["status"] == "FAIL":
            failed.append(entry)
        else:
            not_assessable.append(entry)

    # Compute fairness score over ASSESSABLE metrics only. With no assessable
    # metric the score is None ("could not check"), never a number: see the
    # matching note in classification_fairness_report.
    total_metrics = len(passed) + len(failed)
    passed_count = len(passed)
    fairness_score = passed_count / total_metrics if total_metrics > 0 else None

    # Groups dropped from every metric because they are below min_group_size.
    # Surface them as an explicit "insufficient evidence" verdict so a
    # downstream consumer reading only this assessment block never mistakes a
    # silently excluded protected stratum for an implicit pass.
    insufficient_evidence = _insufficient_evidence_groups(gm)

    if not assessable:
        summary = (
            f"NOT ASSESSABLE: {n_valid_groups} valid group(s) after filtering "
            f"(need at least 2); disparity metrics are vacuous and do not "
            f"certify fairness."
        )
    elif total_metrics == 0:
        # See the matching branch in classification_fairness_report: "0/0 metrics
        # within thresholds" read as a clean bill of health for a run that
        # measured nothing.
        summary = (
            "NOT ASSESSABLE: no metric could be computed on this data; "
            "nothing here certifies fairness."
        )
    else:
        summary = f"{passed_count}/{total_metrics} metrics within thresholds"
    if not_assessable:
        # The reason is per-entry (constant y_true, undefined metric, or too few
        # comparable groups), so do not hard-code one of them in the summary.
        summary += f" ({len(not_assessable)} metric(s) not assessable, excluded from the verdict)"
    if insufficient_evidence:
        summary += (
            f" ({len(insufficient_evidence)} group(s) with insufficient "
            f"evidence, excluded from the verdict: "
            f"{[g['group'] for g in insufficient_evidence]})"
        )
    # Missing-data provenance travels with the verdict, always (mirror of the
    # classification report). See _missing_data_clause.
    summary += _missing_data_clause(info)

    # Build result dict
    result = {
        "task_type": "regression",
        # See VB-DOC-7 note in the classification report: stamp the methodology
        # version on the direct report path too, not only via the analyzer.
        "methodology_version": METHODOLOGY_VERSION,
        "metrics": metrics,
        "residual_bias": bias_by_group,
        "group_stats": group_stats,
        "assessment": {
            "fairness_score": fairness_score,
            "assessable": assessable,
            "passed_metrics": passed,
            "failed_metrics": failed,
            "not_assessable_metrics": not_assessable,
            "insufficient_evidence_groups": insufficient_evidence,
            "summary": summary,
        },
        "data_info": {
            **info,
            "n_samples": info["final_size"],  # Alias for convenience
            "n_groups": gm.n_groups,
            "valid_groups": gm.get_valid_groups(),
            "invalid_groups": gm.get_invalid_groups(),
            "group_sizes": gm.get_group_sizes(),
            "is_intersectional": gm.is_intersectional,
            "y_std": y_std,
        },
        "thresholds_used": default_thresholds,
    }

    # Add statistical validation if requested
    if include_ci:
        # Determine method based on sample sizes
        group_sizes = gm.get_group_sizes()
        overall_method, per_group_methods = select_method(group_sizes)

        # Compute metrics with confidence intervals
        metrics_with_ci = {}

        mae_ci = mae_parity_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["mae_parity_difference"] = mae_ci.to_dict()

        rmse_ci = rmse_parity_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["rmse_parity_difference"] = rmse_ci.to_dict()

        mpd_ci = mean_prediction_difference_with_ci(
            y_true_v,
            y_pred_v,
            sensitive_attr_v,
            min_group_size=min_group_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=ci_method,
            random_state=random_state,
        )
        metrics_with_ci["mean_prediction_difference"] = mpd_ci.to_dict()

        result["metrics_with_ci"] = metrics_with_ci

        # Compute effect sizes
        effect_sizes = compute_regression_effect_sizes(
            y_true_v, y_pred_v, sensitive_attr_v, min_group_size=min_group_size
        )
        result["effect_sizes"] = effect_sizes

        # Statistical validation info
        method_used, methods_that_ran = _provenance(mae_ci, rmse_ci, mpd_ci)
        result["statistical_validation"] = {
            # What RAN. `size_based_recommendation` below is what select_method
            # SUGGESTED from the group sizes, which is a different question and
            # was previously reported here as though it were the answer.
            "method_used": method_used,
            "methods_that_ran": methods_that_ran,
            "size_based_recommendation": overall_method,
            "per_group_size_recommendation": per_group_methods,
            "per_group_methods": per_group_methods,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        }

        # Multiple testing correction: NOT performed for regression metrics.
        # The previous implementation fed min(1, value/threshold) pseudo
        # p-values into the correction, which is statistically meaningless
        # and inverted (grossly unfair metrics capped at p=1.0, never
        # significant). Regression gap metrics (MAE/RMSE/R^2 differences)
        # are not proportions, so no cheap exact test applies; rather than
        # fabricate p-values we report why the correction is unavailable.
        if multiple_testing_correction != "none" and len(metrics) > 1:
            result["multiple_testing_correction"] = {
                "available": False,
                "reason": (
                    "Regression disparity metrics have no per-metric "
                    "hypothesis test implemented; refusing to fabricate "
                    "pseudo p-values. Use bootstrap CIs (include_ci=True) "
                    "to judge uncertainty instead."
                ),
                "requested_method": multiple_testing_correction,
            }

    return cast(FairnessReport, result)


def print_report(report: Dict[str, Any], show_explanations: bool = True) -> None:
    """
    Print a formatted fairness report to console.

    Args:
        report: Report from classification_fairness_report or regression_fairness_report
        show_explanations: Whether to display FairExplAIner explanations if present

    Example:
        >>> report = classification_fairness_report(y_true, y_pred, gender)
        >>> print_report(report)

    With explanations:
        >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
        >>> report = analyzer.get_report()
        >>> print_report(report)  # Will include explanations
    """
    import sys

    if not report:
        print("No report data to display.", flush=True)
        return

    task_type = report.get("task_type", "unknown")
    assessment = report.get("assessment", {})
    metrics = report.get("metrics", {})
    data_info = report.get("data_info", {})
    explanations = report.get("explanations", {}) if show_explanations else {}

    lines = []
    lines.append("=" * 70)
    lines.append(f"VFAIRNESS REPORT - {task_type.upper()}")
    if explanations:
        lines.append("FairExplAIner Mode: ENABLED")
    lines.append("=" * 70)

    # Data summary
    lines.append("\nData Summary:")
    # Row accounting, INCLUDING the strategy that produced the exclusions. Two
    # runs over the same data with different missing_strategy values can reach
    # different verdicts, and a printed artifact naming only the count does not
    # distinguish them. An absent field prints "unknown", never 0: claiming zero
    # exclusions for a report that never recorded any is a false provenance
    # statement, not a harmless default.
    n_excluded = data_info.get("n_excluded")
    strategy = data_info.get("missing_strategy")

    # Render a PRESENT-but-None value as "unknown", not as the literal None.
    # dict.get(key, default) only substitutes when the KEY IS ABSENT, and the
    # fail-closed provenance path deliberately sets these keys TO None when the
    # row accounting cannot be reconciled. The same shape (a .get default that
    # never fires because the key exists holding None) crashed three matplotlib
    # entry points on the new Optional fairness_score, so it is spelled out here.
    def _known(value: Any) -> str:
        return "unknown" if value is None else str(value)

    final_size = data_info.get("final_size")
    original_size = data_info.get("original_size")
    lines.append(
        f"  Samples: {_known(final_size)} of {_known(original_size)} "
        f"(excluded {_known(n_excluded)} with missing "
        f"values, missing_strategy="
        f"{'unknown' if strategy is None else repr(str(strategy))})"
    )
    lines.append(
        f"  Groups: {data_info.get('n_groups', 'N/A')} "
        f"({len(data_info.get('valid_groups', []))} valid, "
        f"{len(data_info.get('invalid_groups', []))} below min size)"
    )
    lines.append(f"  Intersectional: {data_info.get('is_intersectional', False)}")

    # Metrics
    lines.append("\nFairness Metrics:")
    if metrics:
        for name, value in metrics.items():
            # Handle different value types (float, int, MetricResult, etc.)
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    lines.append(f"  {name}: N/A (insufficient data)")
                else:
                    lines.append(f"  {name}: {value:.4f}")
            elif hasattr(value, "value"):
                # Handle MetricResult objects
                lines.append(f"  {name}: {value.value:.4f}")
            elif value is None:
                lines.append(f"  {name}: N/A (null value)")
            else:
                lines.append(f"  {name}: {value}")
    else:
        lines.append("  No metrics computed (check min_group_size setting)")

    # Assessment
    lines.append("\nAssessment:")
    # fairness_score is Optional[float]: it is None on a run where no metric was
    # assessable. isinstance() is checked FIRST so None never reaches np.isnan or
    # the percent format, and the printed line names the could-not-check state
    # instead of implying a missing field.
    #
    # BGL5 A-evaluation-4, 2026-09-27. THE DEFAULT WAS THE WORST VERDICT ON THE
    # SCALE. ``assessment.get("fairness_score", 0)`` substitutes the NUMBER 0
    # when the key is ABSENT, and 0 is a real value here, so a report carrying no
    # score printed a measurement nobody made. Measured before this change:
    #     print_report({'task_type': 'classification',
    #                   'assessment': {'summary': 'no score was ever recorded'},
    #                   'metrics': {}, 'data_info': {}})
    #     before -> "  Fairness Score: 0.0%"   (0 percent: the WORST reading)
    #     after  -> "  Fairness Score: N/A (could not check; this report carries
    #                no fairness_score field)"
    # The present-but-None case was already correct and is unchanged. The rule is
    # written out fifty lines above, for the row accounting: "An absent field
    # prints 'unknown', never 0". Both in-repo producers do write the key, so
    # this needs a hand-built, adapter-built or deserialised report, and
    # print_report is a public export (vfairness.print_report) that takes a plain
    # dict and defends every other field in it.
    _ABSENT = object()
    score = assessment.get("fairness_score", _ABSENT)
    if score is _ABSENT:
        lines.append(
            "  Fairness Score: N/A (could not check; this report carries no fairness_score field)"
        )
    elif isinstance(score, (int, float)) and not isinstance(score, bool) and not np.isnan(score):
        lines.append(f"  Fairness Score: {score:.1%}")
    else:
        lines.append("  Fairness Score: N/A (could not check; no metric was assessable)")
    lines.append(f"  {assessment.get('summary', 'No summary available')}")

    # The per-metric reasons reached the reader only through the one summary
    # sentence ("N metric(s) not assessable"), so the printed artifact never said
    # WHICH metric was withheld or why. They are three-state states and this is
    # the surface a person reads.
    if assessment.get("not_assessable_metrics"):
        lines.append("\n  Not Assessable (excluded from the score, NOT a pass):")
        for m in assessment["not_assessable_metrics"]:
            if isinstance(m, dict):
                lines.append(
                    f"    - {m.get('metric', 'unknown')}: {m.get('reason', 'no reason recorded')}"
                )
            else:
                lines.append(f"    - {m}")

    if assessment.get("failed_metrics"):
        lines.append("\n  Failed Metrics:")
        for m in assessment["failed_metrics"]:
            metric_val = m.get("value", 0)
            threshold_val = m.get("threshold", 0)
            if isinstance(metric_val, (int, float)) and isinstance(threshold_val, (int, float)):
                lines.append(
                    f"    - {m.get('metric', 'unknown')}: {metric_val:.4f} (threshold: {threshold_val:.4f})"
                )
            else:
                lines.append(
                    f"    - {m.get('metric', 'unknown')}: {metric_val} (threshold: {threshold_val})"
                )

    lines.append("=" * 70)

    # Print all lines - using explicit flush to ensure output is displayed
    output = "\n".join(lines)
    print(output, flush=True)

    # Print explanations if present
    if explanations:
        print("\n")
        _print_explanations_section(explanations)

    # Force flush stdout and stderr to ensure output appears immediately
    sys.stdout.flush()
    sys.stderr.flush()


def _print_explanations_section(explanations: Dict[str, Any]) -> None:
    """
    Print the FairExplAIner explanations section.

    Args:
        explanations: Dictionary from FairExplAIner.explain_report()
    """
    import sys

    lines = []
    lines.append("=" * 70)
    lines.append("FAIREXPLAINER - Detailed Metric Explanations")
    lines.append("=" * 70)

    # Print summary
    if "summary" in explanations and explanations["summary"]:
        lines.append("\nSUMMARY:")
        lines.append(f"  {explanations['summary']}")

    # Print metric explanations
    metric_explanations = explanations.get("metrics", {})
    if metric_explanations:
        lines.append("\n" + "-" * 70)
        lines.append("METRIC DETAILS")
        lines.append("-" * 70)

        for metric_name, explanation in metric_explanations.items():
            lines.append(f"\n{metric_name.upper().replace('_', ' ')}")
            lines.append("-" * 40)

            # Value and severity.
            # `.get("severity", "info")` does NOT fire when the key is PRESENT
            # holding None, so `severity.upper()` below raised AttributeError and
            # took the whole printout with it. And a could-not-check card used to
            # fall through this map to the "i" of info, which is the state a
            # graded, genuinely benign value gets: the icon has its own "?" now,
            # and so does any severity this map has never heard of.
            value = explanation.get("value")
            severity = explanation.get("severity") or "not_stated"
            severity_icon = {
                "info": "i",
                "low": "!",
                "medium": "!!",
                "high": "!!!",
                "critical": "!!!!",
                "could_not_check": "?",
                "not_stated": "?",
            }.get(severity, "?")

            if isinstance(value, (int, float)) and not np.isnan(value):
                lines.append(
                    f"  Value: {value:.4f}  [{severity_icon}] Severity: {str(severity).upper()}"
                )
            else:
                lines.append(
                    f"  Value: {value}  [{severity_icon}] Severity: {str(severity).upper()}"
                )

            # Definition (truncated)
            definition = explanation.get("definition", "")
            if definition:
                # Wrap text to fit console
                lines.append("\n  Definition:")
                wrapped = _wrap_text(definition, 65, "    ")
                lines.extend(wrapped)

            # Evaluation
            evaluation = explanation.get("evaluation", "")
            if evaluation:
                lines.append("\n  Evaluation:")
                wrapped = _wrap_text(evaluation, 65, "    ")
                lines.extend(wrapped)

            # Recommendation
            recommendation = explanation.get("recommendation", "")
            if recommendation:
                lines.append("\n  Recommendation:")
                wrapped = _wrap_text(recommendation, 65, "    ")
                lines.extend(wrapped)

            lines.append("")

    # Print statistical explanations (condensed)
    stat_explanations = explanations.get("statistical", {})
    if stat_explanations:
        lines.append("-" * 70)
        lines.append("STATISTICAL MEASURES")
        lines.append("-" * 70)

        for stat_name, explanation in stat_explanations.items():
            evaluation = explanation.get("evaluation", "")
            lines.append(f"\n  {stat_name}:")
            if evaluation:
                wrapped = _wrap_text(evaluation, 65, "    ")
                lines.extend(wrapped)
            else:
                # The name is printed BEFORE the text is tested, so a card with
                # no evaluation text is a named gap and not an absent line: this
                # block used to skip such a card entirely, and a skipped
                # could-not-check is indistinguishable from one that passed.
                severity = explanation.get("severity") or "not stated"
                lines.append(
                    f"    (no evaluation text on this card; severity "
                    f"{str(severity).upper()}. Nothing here is a pass.)"
                )

    lines.append("\n" + "=" * 70)

    print("\n".join(lines), flush=True)
    sys.stdout.flush()


def _wrap_text(text: str, width: int, prefix: str = "") -> List[str]:
    """
    Wrap text to fit within specified width.

    Args:
        text: Text to wrap
        width: Maximum line width
        prefix: Prefix for each line

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = prefix

    for word in words:
        if len(current_line) + len(word) + 1 <= width + len(prefix):
            current_line += word + " "
        else:
            if current_line.strip():
                lines.append(current_line.rstrip())
            current_line = prefix + word + " "

    if current_line.strip():
        lines.append(current_line.rstrip())

    return lines
