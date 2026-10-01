"""
Intersectional fairness analysis for vfairness.

This module provides detailed group-level analysis to identify which
specific groups are driving fairness disparities. It answers the key question:
"We know there's a disparity of X% - but WHO is most affected?"

Features:
    - Identify most/least advantaged groups (privileged group identification)
    - Comparative disparity analysis (intersectional vs single-attribute)
    - Risk assessment with severity indicators
    - Actionable group-specific insights with relative disadvantage metrics
    - Transparency contract: every cell the engine looked at is reported,
      including those filtered by the size gate (excluded_groups) and the
      audit-grade zero_selection_alerts (cells with positive_count == 0 at
      n >= zero_selection_floor, surfaced regardless of the size gate).

Configuration parameters (callers should override per audit when needed):
    min_group_size (default 30):
        Minimum samples required for a cell to be included in the ranked
        analysis. 30 is the Turing M3 standard for reporting; smaller cells
        produce selection-rate estimates too noisy to rank against each
        other. Lower for exploratory work or when small protected groups
        matter; never let it drive the *only* signal you act on.
    low_n_warning_threshold (default = max(min_group_size, 30)):
        Cells whose n is below this threshold but at/above min_group_size
        are analysed AND flagged with low_n_warning=True on the
        corresponding GroupAdvantage. Frontends should render a confidence
        caveat on these rows.
    zero_selection_floor (default 10):
        Any cell with positive_count == 0 and n >= zero_selection_floor is
        surfaced in zero_selection_alerts, *regardless* of min_group_size.
        Rationale: for a regulator a 0/22 protected group is the
        headline, not the noise floor. Default 10 keeps single-digit
        cells out of the alert list while catching real harm.

Example:
    >>> from vfairness import intersectional_disparity_analysis
    >>> analysis = intersectional_disparity_analysis(
    ...     y_true, y_pred, demographics,
    ...     min_group_size=15,       # smaller-cell aware
    ...     zero_selection_floor=10, # surface 0/N protected cells
    ... )
    >>> print(f"Most advantaged: {analysis['privileged_group']['group']}")
    >>> print(f"Most disadvantaged: {analysis['disadvantaged_group']['group']}")
    >>> for cell in analysis['zero_selection_alerts']:
    ...     print(f"Zero invites: {cell['group']} (n={cell['size']})")
    >>> dt = analysis['data_treatment']
    >>> print(f"Analysed {dt['n_cells_included']} / {dt['n_total_cells_seen']} cells; "
    ...       f"{dt['n_cells_excluded_small']} excluded at min_group_size={dt['min_group_size']}")
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..._triage import is_measured
from ._grouping import GroupManager
from ._statistics import (
    bayesian_difference_ci,
    benjamini_hochberg_correction,
    cohens_h,
    cohens_h_interpretation,
    fisher_exact_test,
    minimum_detectable_effect,
    power_warning,
    proportion_z_test,
)
from ._validation import ArrayLike, MissingStrategy, validate_inputs

logger = logging.getLogger(__name__)


@dataclass
class GroupAdvantage:
    """
    Container for group advantage analysis results.

    Attributes:
        group: Group name/identifier
        positive_rate: Rate of positive predictions for this group
        ground_truth_rate: Rate of positive ground truth labels for this group
        false_positive_rate: FPR for this group (wrongly predicted positive among actual negatives)
        prediction_delta: positive_rate minus ground_truth_rate (over/under-prediction)
        size: Number of samples in this group
        relative_to_overall: Ratio compared to overall rate
        relative_to_best: Ratio compared to best group
        disparity_contribution: How much this group contributes to overall disparity
        severity: Risk level ('info', 'low', 'medium', 'high', 'critical')
    """

    group: str
    positive_rate: float
    size: int
    relative_to_overall: float
    relative_to_best: float
    disparity_contribution: float
    severity: str
    ground_truth_rate: float = 0.0
    # FPR for this cell, or NaN when it could NOT be computed because the cell
    # has no actual negatives (the denominator fp + tn is zero). NaN is a third
    # state and must never be read as 0.0, which on this scale is the BEST
    # possible FPR: "this group is never wrongly selected". `unmeasured` names
    # every such field on this cell.
    false_positive_rate: float = 0.0
    prediction_delta: float = 0.0
    # Confidence caveat for cells that survived the size gate but are still
    # small enough that the rate is noisy. Frontend should annotate these.
    low_n_warning: bool = False
    # BGL-S2, 2026-09-16. Names the fields on THIS cell that could not be
    # measured, with the reason, so a GUI can render could-not-check instead of
    # a number. Empty means every field above is a real measurement.
    unmeasured: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "group": self.group,
            "positive_rate": self.positive_rate,
            "ground_truth_rate": self.ground_truth_rate,
            "false_positive_rate": self.false_positive_rate,
            "prediction_delta": self.prediction_delta,
            "size": self.size,
            "relative_to_overall": self.relative_to_overall,
            "relative_to_best": self.relative_to_best,
            "disparity_contribution": self.disparity_contribution,
            "severity": self.severity,
            "low_n_warning": self.low_n_warning,
            "unmeasured": dict(self.unmeasured),
        }


def _assess_severity(disparity: float) -> str:
    """
    Assess severity level based on disparity magnitude.

    Args:
        disparity: Absolute disparity value (0 to 1)

    Returns:
        Severity level string
    """
    if disparity < 0.05:
        return "info"
    elif disparity < 0.10:
        return "low"
    elif disparity < 0.15:
        return "medium"
    elif disparity < 0.20:
        return "high"
    else:
        return "critical"


def identify_privileged_groups(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    min_group_size: int = 30,
    low_n_warning_threshold: Optional[int] = None,
    zero_selection_floor: int = 10,
    missing_strategy: MissingStrategy = "exclude",
    outcome_polarity: str = "positive_favorable",
) -> Dict[str, Any]:
    """
    Identify most and least advantaged groups with dual-lens analysis.

    Computes prediction rates, ground truth rates, false positive rates, and
    prediction deltas per subgroup. Supports outcome polarity: when
    positive=unfavorable (e.g., recidivism), the group with the LOWEST
    positive prediction rate is the most advantaged.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values (can be intersectional)
        min_group_size: Minimum samples required for a cell to be included
            in the ranked analysis. Default 30 (Turing M3 reporting standard).
            Lower (15 / 10 / 5) for exploratory or small-cell-aware audits;
            the engine still surfaces transparency about everything it dropped.
        low_n_warning_threshold: Cells with n below this threshold AND at or
            above ``min_group_size`` are analysed but tagged with
            ``low_n_warning=True``. Defaults to ``max(min_group_size, 30)``
            so callers who lower ``min_group_size`` still see a confidence
            caveat on every cell with n<30.
        zero_selection_floor: Any cell with positive_count==0 and n at or
            above this floor is surfaced in ``zero_selection_alerts``
            regardless of ``min_group_size``. Default 10. Rationale: a 0/22
            protected group is the headline, not the noise floor.
        missing_strategy: How to handle missing values
        outcome_polarity: 'positive_favorable' (e.g., loan approved) or
                         'positive_unfavorable' (e.g., recidivism predicted)

    Returns:
        Dict containing:
            - privileged_group: GroupAdvantage for most advantaged group
            - disadvantaged_group: GroupAdvantage for least advantaged group
            - all_groups: List of GroupAdvantage for all groups (sorted);
              each carries ``low_n_warning: bool``
            - overall_rate: Overall positive prediction rate
            - overall_ground_truth_rate: Overall ground truth positive rate
            - max_disparity: Maximum disparity between any two groups
            - max_ground_truth_disparity: Max disparity in ground truth rates
            - disparity_severity: Severity assessment of the disparity
            - outcome_polarity: The polarity used for this analysis
            - excluded_groups: list of cells dropped by ``min_group_size``,
              each with {group, size, positive_count, positive_rate,
              ground_truth_rate, reason}. For transparency: the GUI can say
              "we did NOT analyse these, here's why".
            - zero_selection_alerts: list of cells with positive_count==0
              and n >= ``zero_selection_floor``, surfaced regardless of the
              size gate. Each row is {group, size, positive_count,
              ground_truth_rate, analysed, reason}.
            - data_treatment: audit log of thresholds used + counts of
              cells in each treatment bucket {min_group_size,
              low_n_warning_threshold, zero_selection_floor,
              outcome_polarity, n_total_cells_seen, n_cells_included,
              n_cells_excluded_small, n_cells_warning_low_n,
              n_zero_selection_alerts, overall_rate,
              overall_ground_truth_rate}.

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

    Ledger row: identify_privileged_groups. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    # Validate inputs
    y_true_v, y_pred_v, sensitive_attr_v, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr_v, min_group_size=min_group_size)

    # Detect if y_true and y_pred are identical (no separate prediction column)
    same_column = np.array_equal(y_true_v, y_pred_v)

    # Compute overall rates
    overall_rate = np.mean(y_pred_v)
    overall_gt_rate = np.mean(y_true_v)

    # Default the low-n warning threshold to the legacy default (30) so callers
    # using a smaller min_group_size still surface a confidence caveat on every
    # cell with n < 30. Set explicitly to ``min_group_size`` (or smaller) to
    # suppress the warning.
    if low_n_warning_threshold is None:
        low_n_warning_threshold = max(min_group_size, 30)
    is_favorable = outcome_polarity != "positive_unfavorable"

    def _per_group_stats(name: str) -> Dict[str, Any]:
        mask = gm.get_mask(name)
        group_preds = y_pred_v[mask]
        group_true = y_true_v[mask]
        size = int(len(group_preds))
        rate = float(np.mean(group_preds)) if size else 0.0
        gt_rate = float(np.mean(group_true)) if size else 0.0
        fp = int(np.sum((group_preds == 1) & (group_true == 0)))
        tn = int(np.sum((group_preds == 0) & (group_true == 0)))
        # BGL-S2 (2026-09-16). A cell with NO ACTUAL NEGATIVES has an undefined
        # false positive rate: fp + tn is the denominator and it is zero. It was
        # reported as a measured 0.0, which on this scale is the BEST possible
        # value, "this group is never wrongly selected". Measured at the public
        # entry: group A (100 rows, every label positive) came back
        # false_positive_rate 0.0 next to group B's measured 0.8, with no
        # warning and nothing in the payload distinguishing the two.
        n_actual_negatives = fp + tn
        fpr = float(fp / n_actual_negatives) if n_actual_negatives > 0 else float("nan")
        return {
            "group": name,
            "positive_rate": rate,
            "ground_truth_rate": gt_rate,
            "false_positive_rate": fpr,
            "n_actual_negatives": n_actual_negatives,
            "prediction_delta": rate - gt_rate,
            "size": size,
            "positive_count": int(np.sum(group_preds == 1)),
        }

    # Compute per-group rates (dual-lens: prediction + ground truth + FPR).
    # The valid set drives the ranking; excluded cells are kept for transparency
    # so the GUI can show "we did NOT analyse these because n < threshold" and
    # so we can still raise zero-selection alerts on small protected groups.
    group_data = [_per_group_stats(name) for name in gm.get_valid_groups()]
    excluded_data = [_per_group_stats(name) for name in gm.get_invalid_groups()]

    # Transparency: zero-selection alerts + excluded cells + audit log
    # A "zero-selection cell" is a protected group where the model selected
    # nobody (positive_rate == 0) at n >= zero_selection_floor. We flag these
    # for the favorable-polarity case across BOTH the analysed set and the
    # cells excluded by the size gate, because for an auditor a 0/22 Native
    # American group is the headline, not the noise floor.
    zero_alerts: List[Dict[str, Any]] = []
    if is_favorable:
        for gd in group_data + excluded_data:
            if gd["size"] >= zero_selection_floor and gd["positive_count"] == 0:
                zero_alerts.append(
                    {
                        "group": gd["group"],
                        "size": gd["size"],
                        "positive_count": 0,
                        "ground_truth_rate": gd["ground_truth_rate"],
                        "analysed": gd in group_data,
                        "reason": (
                            "zero positives with n >= zero_selection_floor; "
                            + (
                                "included in main analysis"
                                if gd in group_data
                                else f"excluded by min_group_size={min_group_size} "
                                f"(n={gd['size']}) but still surfaced as alert"
                            )
                        ),
                    }
                )

    excluded_summary = [
        {
            "group": gd["group"],
            "size": gd["size"],
            "positive_count": gd["positive_count"],
            "positive_rate": gd["positive_rate"],
            "ground_truth_rate": gd["ground_truth_rate"],
            "reason": f"n={gd['size']} < min_group_size={min_group_size}",
        }
        for gd in sorted(excluded_data, key=lambda x: -x["size"])
    ]

    data_treatment = {
        "min_group_size": min_group_size,
        "low_n_warning_threshold": low_n_warning_threshold,
        "zero_selection_floor": zero_selection_floor,
        "outcome_polarity": outcome_polarity,
        "n_total_cells_seen": gm.n_groups,
        "n_cells_included": len(group_data),
        "n_cells_excluded_small": len(excluded_data),
        "n_cells_warning_low_n": sum(
            1 for gd in group_data if gd["size"] < low_n_warning_threshold
        ),
        "n_zero_selection_alerts": len(zero_alerts),
        "overall_rate": float(overall_rate),
        "overall_ground_truth_rate": float(overall_gt_rate),
    }

    if len(group_data) < 2:
        # A DISPARITY IS A COMPARISON. With fewer than two analysable cells there
        # is nothing to compare, and 0.0 + "info" is the most reassuring pair of
        # values this function can produce. Measured 2026-09-08 on 200 rows where
        # a 10-person group was rejected outright (positive rate 0.0 against a
        # ground-truth rate of 0.6) and fell below min_group_size=30: this
        # returned max_disparity=0.0, disparity_severity="info", while
        # excluded_groups and zero_selection_alerts sat in the SAME dict
        # describing exactly the group that was wiped out. Downstream,
        # _generate_comparison turned that 0.0 into a clean single-attribute bar
        # standing next to real ones.
        n_excluded = len(excluded_data)
        warnings.warn(
            f"identify_privileged_groups: only {len(group_data)} cell(s) met "
            f"min_group_size={min_group_size}"
            + (f" ({n_excluded} excluded as too small)" if n_excluded else "")
            + ". No disparity was computed. Reporting max_disparity=nan and "
            "disparity_severity='not_assessed' (could not check), NOT 0.0/'info'."
            + (
                " The excluded_groups and zero_selection_alerts entries below still "
                "describe those cells."
                if n_excluded
                else ""
            ),
            UserWarning,
            stacklevel=2,
        )
        return {
            "privileged_group": None,
            "disadvantaged_group": None,
            "all_groups": [],
            "overall_rate": overall_rate,
            "overall_ground_truth_rate": overall_gt_rate,
            "max_disparity": float("nan"),
            "max_ground_truth_disparity": float("nan"),
            "disparity_severity": "not_assessed",
            "outcome_polarity": outcome_polarity,
            "excluded_groups": excluded_summary,
            "zero_selection_alerts": zero_alerts,
            "data_treatment": data_treatment,
        }

    # Polarity-aware sorting:
    # positive_favorable (e.g., approved): highest rate = most advantaged (descending)
    # positive_unfavorable (e.g., recidivism): lowest rate = most advantaged (ascending)
    is_unfavorable = outcome_polarity == "positive_unfavorable"
    group_data.sort(key=lambda x: x["positive_rate"], reverse=not is_unfavorable)

    # After sorting, index 0 = most advantaged, index -1 = most disadvantaged
    best_rate = group_data[0]["positive_rate"]
    worst_rate = group_data[-1]["positive_rate"]
    max_disparity = abs(best_rate - worst_rate)

    # Ground truth disparity (for comparison: is model amplifying or reflecting reality?)
    gt_rates = [gd["ground_truth_rate"] for gd in group_data]
    max_gt_disparity = max(gt_rates) - min(gt_rates) if gt_rates else 0.0

    # Create GroupAdvantage objects
    #
    # BGL-S2 (2026-09-16). The two ratios fell back to 1.0 -- EXACTLY AT PARITY,
    # the cleanest possible reading -- whenever their denominator was zero.
    # Measured at the public entry with nobody selected at all (overall_rate 0.0):
    # every group reported relative_to_overall 1.0 and relative_to_best 1.0,
    # severity 'info', max_disparity 0.0. A ratio with an empty denominator is
    # undefined, so it is NaN here: `is_measured` (imported at the top of this
    # module) answers False for it, and NaN is also what survives the Pulse
    # normaliser, whose `pick(...) or 0.0` would turn a None back into a number.
    all_groups = []
    unmeasured_cells: Dict[str, Dict[str, str]] = {}
    for gd in group_data:
        rate = gd["positive_rate"]
        unmeasured: Dict[str, str] = {}

        if overall_rate > 0:
            relative_to_overall = rate / overall_rate
        else:
            relative_to_overall = float("nan")
            unmeasured["relative_to_overall"] = (
                "the overall selection rate is 0, so there is no ratio to take"
            )
        if best_rate > 0:
            relative_to_best = rate / best_rate
        else:
            relative_to_best = float("nan")
            unmeasured["relative_to_best"] = (
                "the best group's selection rate is 0, so there is no ratio to take"
            )
        if not is_measured(gd["false_positive_rate"]):
            unmeasured["false_positive_rate"] = (
                f"no actual negatives in this cell (n={gd['size']}, "
                f"ground_truth_rate={gd['ground_truth_rate']:.3f}), so the FPR "
                "denominator is empty"
            )
        if unmeasured:
            unmeasured_cells[str(gd["group"])] = unmeasured

        disparity_contribution = abs(rate - overall_rate)
        gap_from_best = abs(best_rate - rate)
        severity = _assess_severity(gap_from_best)

        ga = GroupAdvantage(
            group=gd["group"],
            positive_rate=rate,
            size=gd["size"],
            relative_to_overall=relative_to_overall,
            relative_to_best=relative_to_best,
            disparity_contribution=disparity_contribution,
            severity=severity,
            ground_truth_rate=gd["ground_truth_rate"],
            false_positive_rate=gd["false_positive_rate"],
            prediction_delta=gd["prediction_delta"],
            low_n_warning=gd["size"] < low_n_warning_threshold,
            unmeasured=unmeasured,
        )
        all_groups.append(ga)

    if unmeasured_cells:
        warnings.warn(
            "identify_privileged_groups: some per-cell figures could not be measured and are "
            "NaN, not 0.0 or 1.0: "
            + "; ".join(
                f"{cell} [{', '.join(fields)}]" for cell, fields in unmeasured_cells.items()
            )
            + ". See GroupAdvantage.unmeasured on each cell for the reason.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "privileged_group": all_groups[0],
        "disadvantaged_group": all_groups[-1],
        "all_groups": all_groups,
        "overall_rate": overall_rate,
        "overall_ground_truth_rate": overall_gt_rate,
        "max_disparity": max_disparity,
        "max_ground_truth_disparity": max_gt_disparity,
        "disparity_severity": _assess_severity(max_disparity),
        "outcome_polarity": outcome_polarity,
        "prediction_equals_target": same_column,
        "excluded_groups": excluded_summary,
        "zero_selection_alerts": zero_alerts,
        "data_treatment": data_treatment,
    }


def intersectional_disparity_analysis(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    single_attributes: Optional[Dict[str, ArrayLike]] = None,
    min_group_size: int = 30,
    low_n_warning_threshold: Optional[int] = None,
    zero_selection_floor: int = 10,
    missing_strategy: MissingStrategy = "exclude",
    outcome_polarity: str = "positive_favorable",
) -> Dict[str, Any]:
    """
    Comprehensive intersectional disparity analysis with dual-lens comparison.

    Computes both prediction and ground truth rates per subgroup, identifies
    where the model amplifies or mitigates societal disparities, and generates
    structured findings with severity ratings.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Intersectional protected attribute values
        single_attributes: Optional dict of single attributes for comparison
                          e.g., {'gender': gender_array, 'race': race_array}
        min_group_size: Minimum samples required for a cell to be ranked.
            Default 30 (Turing M3 standard). Lower for small-cell-aware
            audits; transparency outputs (excluded_groups,
            zero_selection_alerts) still surface everything filtered out.
        low_n_warning_threshold: Cells with n below this threshold but at or
            above ``min_group_size`` carry ``low_n_warning=True`` on their
            GroupAdvantage. Defaults to ``max(min_group_size, 30)``.
        zero_selection_floor: Cells with positive_count==0 and n at or
            above this floor are surfaced in ``zero_selection_alerts``
            regardless of ``min_group_size``. Default 10.
        missing_strategy: How to handle missing values
        outcome_polarity: 'positive_favorable' or 'positive_unfavorable'

    Returns:
        Dict containing:
            - intersectional_analysis: Full group privilege identification
              (dual-lens). Each group carries ``low_n_warning: bool``.
            - single_attribute_analyses: Analysis for each single attribute
            - comparison: Intersectional vs single-attribute comparison
            - insights: Human-readable insights
            - recommendations: Actionable recommendations
            - findings: Structured findings with types, severities, and metric values
            - excluded_groups: cells dropped by the size gate, kept for
              transparency. Same shape as identify_privileged_groups().
            - zero_selection_alerts: cells with positive_count==0 at
              n >= ``zero_selection_floor``. The audit-grade headline.
            - data_treatment: audit log of thresholds + counts in each
              treatment bucket. See identify_privileged_groups() for the
              full schema.
            - statistical_summary: BH-FDR-corrected finding counts.
            - data_info: validate_inputs() telemetry (n, n_dropped, etc.).

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

    Ledger row: intersectional_disparity_analysis. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    # Validate inputs
    y_true_v, y_pred_v, sensitive_attr_v, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    # Main intersectional analysis (now with dual-lens metrics + transparency)
    intersectional = identify_privileged_groups(
        y_true_v,
        y_pred_v,
        sensitive_attr_v,
        min_group_size=min_group_size,
        low_n_warning_threshold=low_n_warning_threshold,
        zero_selection_floor=zero_selection_floor,
        missing_strategy="exclude",
        outcome_polarity=outcome_polarity,
    )

    # Single-attribute analyses (if provided). Use a separate, larger size gate
    # for marginals (n >= min_group_size) but keep the same transparency
    # outputs so the GUI can render small-marginal warnings consistently.
    #
    # BROKEN-DATA CHECK 2, 2026-10-01. A SUPPLIED ATTRIBUTE WAS DROPPED IN
    # SILENCE AND THEN REPORTED AS NEVER SUPPLIED. The length test below compared
    # each single attribute (caller's length) against the VALIDATED rows, so the
    # moment validate_inputs excluded a single row every single attribute failed
    # it and was skipped with no trace. _generate_comparison then saw an empty
    # dict and wrote "no single_attributes were supplied". Measured on 400 rows
    # with 10 predictions NaN and single_attributes={'sex': s}: intersectional
    # disparity 0.2055 measured, single_attribute_disparities {}, hidden_disparity
    # NaN, comparison_not_run "no single_attributes were supplied". The MCP
    # intersectional tool reaches this on every frame with one missing attribute
    # value, because it marks those rows None and they are excluded. A measurable
    # comparison was destroyed and the reason given was false.
    # Now an attribute of the caller's ORIGINAL length is cut to the same rows
    # validate_inputs kept; one that fits neither length is named, not dropped.
    single_analyses = {}
    unusable_singles: Dict[str, str] = {}
    if single_attributes:
        n_input = info.get("original_size")
        keep = _kept_row_mask(y_true, y_pred, sensitive_attr, missing_strategy, len(y_true_v))
        for attr_name, attr_values in single_attributes.items():
            if len(attr_values) == len(y_true_v):
                values = attr_values
            elif len(attr_values) == n_input and keep is not None:
                values = np.asarray(attr_values)[keep]
            elif len(attr_values) == n_input:
                unusable_singles[attr_name] = (
                    f"the {len(y_true_v)} rows that survived missing-value handling could "
                    "not be located among the rows supplied, so this attribute could not be "
                    "aligned with the intersectional analysis"
                )
                continue
            else:
                unusable_singles[attr_name] = (
                    f"length {len(attr_values)} matches neither the {n_input} rows supplied "
                    f"nor the {len(y_true_v)} rows that survived missing-value handling, so "
                    "its rows could not be aligned with the intersectional analysis"
                )
                continue
            single_analyses[attr_name] = identify_privileged_groups(
                y_true_v,
                y_pred_v,
                values,
                min_group_size=min_group_size,
                low_n_warning_threshold=low_n_warning_threshold,
                zero_selection_floor=zero_selection_floor,
                missing_strategy="exclude",
                outcome_polarity=outcome_polarity,
            )

    # Generate comparison
    comparison = _generate_comparison(intersectional, single_analyses)
    if unusable_singles:
        # Named in the result that outlives the call, not only in a warning, and
        # never under the "none were supplied" wording: they were supplied.
        comparison["single_attributes_unusable"] = dict(unusable_singles)
        if not single_analyses:
            comparison["comparison_not_run"] = (
                f"{len(unusable_singles)} single attribute(s) were supplied "
                f"({', '.join(unusable_singles)}) but none could be aligned with the "
                "intersectional rows, so no comparison was made. hidden_disparity is NaN "
                "and intersectional_reveals_more is None (could not check)."
            )
        warnings.warn(
            "intersectional_disparity_analysis: single attribute(s) not used: "
            + "; ".join(f"'{k}' ({v})" for k, v in unusable_singles.items()),
            UserWarning,
            stacklevel=2,
        )

    # Generate insights
    insights = _generate_insights(intersectional, single_analyses)

    # Generate recommendations
    recommendations = _generate_recommendations(intersectional, single_analyses)

    # Generate structured findings (dual-lens: prediction vs ground truth)
    findings = generate_structured_findings(intersectional, outcome_polarity)

    # Promote the transparency triple to the top level so callers can render
    # them without spelunking into intersectional_analysis. Keys are kept on
    # the sub-dict too for backward compatibility.
    return {
        "intersectional_analysis": intersectional,
        "single_attribute_analyses": single_analyses,
        "comparison": comparison,
        "insights": insights,
        "recommendations": recommendations,
        "findings": findings,
        "excluded_groups": intersectional.get("excluded_groups", []),
        "zero_selection_alerts": intersectional.get("zero_selection_alerts", []),
        "data_treatment": intersectional.get("data_treatment", {}),
        "statistical_summary": {
            "total_findings": len(findings),
            "significant_findings": sum(
                1 for f in findings if f.get("statistically_significant", False)
            ),
            "correction_method": "benjamini_hochberg_fdr",
            "alpha": 0.05,
            "n_groups_tested": len(intersectional.get("all_groups", [])),
        },
        "data_info": info,
    }


def _kept_row_mask(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    missing_strategy: str,
    n_kept: int,
) -> Optional[np.ndarray]:
    """The rows of the caller's input that ``validate_inputs`` kept, or None.

    Mirrors its rule for a one-dimensional attribute: 'exclude' drops a row
    missing a label, a prediction or the attribute; 'as_group' drops only a row
    missing a label or a prediction. The result is checked against the number
    of rows validation actually kept, and None (could not align) is returned on
    any disagreement, so a single attribute is never matched to the wrong rows.
    """
    try:
        yt = pd.isna(np.asarray(y_true, dtype=object)).ravel()
        yp = pd.isna(np.asarray(y_pred, dtype=object)).ravel()
        if isinstance(sensitive_attr, pd.DataFrame):
            sa = sensitive_attr.isna().any(axis=1).to_numpy()
        else:
            sa = pd.isna(np.asarray(sensitive_attr, dtype=object)).ravel()
    except Exception as exc:  # noqa: BLE001 - an unreadable input is could-not-align
        # Not silent: the caller names the attribute in
        # comparison["single_attributes_unusable"], and the cause is logged here.
        logger.warning("intersectional: could not align single attributes to kept rows: %s", exc)
        return None
    if not (yt.shape == yp.shape == sa.shape):
        return None
    missing = yt | yp
    if missing_strategy != "as_group":
        missing = missing | sa
    keep = ~missing
    if int(keep.sum()) != int(n_kept):
        return None
    return keep


def _generate_comparison(
    intersectional: Dict[str, Any], single_analyses: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate comparison between intersectional and single-attribute analyses."""
    comparison = {
        "intersectional_disparity": intersectional["max_disparity"],
        "single_attribute_disparities": {},
        "intersectional_reveals_more": False,
        "hidden_disparity": 0.0,
    }

    # An attribute whose disparity could not be computed is NOT an attribute with
    # a disparity of zero. It used to enter max() as one, so a run where the only
    # analysable attribute was unassessable compared a real intersectional gap
    # against a fabricated floor of 0.0 and announced hidden disparity.
    max_single_disparity = 0.0
    not_assessable: List[str] = []
    for attr_name, analysis in single_analyses.items():
        disp = analysis["max_disparity"]
        comparison["single_attribute_disparities"][attr_name] = disp
        if is_measured(disp):
            max_single_disparity = max(max_single_disparity, float(disp))
        else:
            not_assessable.append(attr_name)

    comparison["single_attributes_not_assessable"] = not_assessable
    assessed = [a for a in single_analyses if a not in not_assessable]
    inter = intersectional["max_disparity"]
    inter_measured = is_measured(inter)

    if not single_analyses:
        # BGL-S2 (2026-09-16). This branch used to `pass`, leaving the initial
        # defaults in place: intersectional_reveals_more=False and
        # hidden_disparity=0.0. No single-attribute comparison was RUN at all,
        # so both were claims about a comparison nobody made, and they were made
        # on healthy data too: a real, measured 100% intersectional gap still
        # reported reveals_more=False, hidden_disparity=0.0 and
        # single_attributes_not_assessable=[] (an explicit claim that nothing was
        # skipped), silently. `single_attributes` is the documented default of
        # None, so no warning: the caller knows it asked for no comparison. The
        # RESULT, which outlives the call, must still say so.
        comparison["hidden_disparity"] = float("nan")
        comparison["intersectional_reveals_more"] = None
        comparison["comparison_not_run"] = (
            "no single_attributes were supplied, so the intersectional view was never "
            "compared against any single-attribute view. hidden_disparity is NaN and "
            "intersectional_reveals_more is None (could not check), not 0.0 / False."
        )
    elif not assessed or not inter_measured:
        # Nothing to subtract from, or nothing to subtract. Three states, so a
        # missing comparison never reads as "intersectional revealed nothing".
        comparison["hidden_disparity"] = float("nan")
        comparison["intersectional_reveals_more"] = None
        warnings.warn(
            "intersectional comparison: "
            + (
                "the intersectional disparity was not assessed"
                if not inter_measured
                else f"no single-attribute disparity was assessable ({', '.join(not_assessable)})"
            )
            + ". Reporting intersectional_reveals_more=None (could not check).",
            UserWarning,
            stacklevel=2,
        )
    else:
        if not_assessable:
            warnings.warn(
                f"intersectional comparison: {len(not_assessable)} attribute(s) had no "
                f"assessable disparity ({', '.join(not_assessable)}) and were left out of "
                f"the single-attribute maximum.",
                UserWarning,
                stacklevel=2,
            )
        # Check if intersectional analysis reveals larger disparity
        hidden = float(inter) - max_single_disparity
        comparison["hidden_disparity"] = max(0, hidden)
        comparison["intersectional_reveals_more"] = hidden > 0.01  # >1% hidden disparity

    return comparison


def _generate_insights(
    intersectional: Dict[str, Any], single_analyses: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate human-readable insights about the disparities."""
    insights = []

    priv = intersectional.get("privileged_group")
    disadv = intersectional.get("disadvantaged_group")

    if priv and disadv:
        # Main disparity insight
        disparity = intersectional["max_disparity"]
        insights.append(
            f"Maximum disparity of {disparity:.1%} found between "
            f"'{priv.group}' (rate: {priv.positive_rate:.1%}) and "
            f"'{disadv.group}' (rate: {disadv.positive_rate:.1%})."
        )

        # Relative disadvantage (polarity-aware)
        polarity = intersectional.get("outcome_polarity", "positive_favorable")
        if polarity == "positive_unfavorable":
            # Unfavorable: disadvantaged has HIGHEST rate (most flagged for bad outcome)
            # priv has lowest rate (least flagged)
            if priv.positive_rate > 0 and disadv.positive_rate > 0:
                multiplier = disadv.positive_rate / priv.positive_rate
                insights.append(
                    f"'{disadv.group}' is flagged for the unfavorable outcome at "
                    f"{multiplier:.1f}x the rate of '{priv.group}' "
                    f"({disadv.positive_rate:.1%} vs {priv.positive_rate:.1%})."
                )
            elif disadv.positive_rate > 0:
                insights.append(
                    f"'{disadv.group}' is flagged at {disadv.positive_rate:.1%}, while "
                    f"'{priv.group}' is essentially never flagged ({priv.positive_rate:.1%})."
                )
        else:
            # Favorable: disadvantaged has LOWEST rate (least positive outcomes)
            if priv.positive_rate > 0 and disadv.positive_rate > 0:
                ratio = disadv.positive_rate / priv.positive_rate
                multiplier = priv.positive_rate / disadv.positive_rate
                insights.append(
                    f"'{disadv.group}' receives positive predictions at "
                    f"{ratio:.1%} the rate of '{priv.group}' "
                    f"({multiplier:.1f}x disadvantage)."
                )
            elif priv.positive_rate > 0:
                insights.append(
                    f"'{disadv.group}' receives no positive predictions, while "
                    f"'{priv.group}' has a rate of {priv.positive_rate:.1%}."
                )

        # Severity assessment
        severity = intersectional["disparity_severity"]
        severity_messages = {
            "info": "This disparity is within acceptable limits.",
            "low": "This disparity is low but should be monitored.",
            "medium": "This disparity warrants investigation.",
            "high": "This disparity is significant and requires attention.",
            "critical": "This disparity is severe and requires immediate action.",
            "not_assessed": (
                "No disparity was computed: too few groups met the minimum size. "
                "This is not a finding that the disparity is small."
            ),
        }
        insights.append(f"Severity: {severity.upper()} - {severity_messages.get(severity, '')}")

    # Intersectional vs single attribute comparison
    for attr_name, analysis in single_analyses.items():
        single_disp = analysis["max_disparity"]
        inter_disp = intersectional["max_disparity"]

        if inter_disp > single_disp + 0.01:  # >1% more disparity revealed
            diff = inter_disp - single_disp
            insights.append(
                f"Intersectional analysis reveals {diff:.1%} additional disparity "
                f"not visible when analyzing {attr_name} alone "
                f"(single-attribute disparity: {single_disp:.1%})."
            )

    return insights


def _generate_recommendations(
    intersectional: Dict[str, Any], single_analyses: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate actionable recommendations based on the analysis."""
    recommendations = []

    priv = intersectional.get("privileged_group")
    disadv = intersectional.get("disadvantaged_group")
    severity = intersectional.get("disparity_severity", "info")

    if not priv or not disadv:
        return ["Insufficient data for group analysis. Consider reducing min_group_size."]

    # Severity-based recommendations
    if severity in ["high", "critical"]:
        recommendations.append(
            f"PRIORITY: Investigate model behavior for '{disadv.group}' group. "
            f"Review training data representation and feature engineering."
        )
        recommendations.append(
            "Consider applying fairness constraints during model training "
            "(e.g., demographic parity constraints, reweighting)."
        )
    elif severity == "medium":
        recommendations.append(
            f"Monitor predictions for '{disadv.group}' group. "
            f"Investigate potential bias in training data or features."
        )
    elif severity == "low":
        recommendations.append(
            "Continue monitoring fairness metrics. Current disparity levels are acceptable."
        )

    # Size-based recommendations
    if disadv.size < 100:
        recommendations.append(
            f"Note: '{disadv.group}' has only {disadv.size} samples. "
            f"Consider collecting more data for this group to validate findings."
        )

    # Intersectional-specific recommendations
    if single_analyses:
        inter_disp = intersectional["max_disparity"]
        max_single = max(a["max_disparity"] for a in single_analyses.values())

        if inter_disp > max_single + 0.05:  # >5% hidden disparity
            recommendations.append(
                "Intersectional analysis reveals significant hidden disparities. "
                "Ensure fairness interventions consider group intersections, "
                "not just individual protected attributes."
            )

    return recommendations


def _proportion_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """
    Proportion test with automatic method selection.
    Uses Fisher's exact test for small samples (n < 50), z-test otherwise.
    All functions imported from _statistics.py (vfairness library).
    """
    if n1 < 50 or n2 < 50:
        # Fisher's exact test for small samples (Agresti & Caffo, 2000)
        a = int(round(p1 * n1))
        b = n1 - a
        c = int(round(p2 * n2))
        d = n2 - c
        return fisher_exact_test(a, b, c, d)
    else:
        return proportion_z_test(p1, n1, p2, n2)


def generate_structured_findings(
    intersectional: Dict[str, Any],
    outcome_polarity: str = "positive_favorable",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Generate structured findings from dual-lens intersectional analysis.

    Includes statistical significance testing (two-proportion z-test) with
    Benjamini-Hochberg FDR correction for multiple comparisons. Only findings
    that are statistically significant after correction are marked as confirmed.

    Args:
        intersectional: Output from identify_privileged_groups (with dual-lens fields)
        outcome_polarity: 'positive_favorable' or 'positive_unfavorable'

    Returns:
        List of finding dicts with: type, severity, groups, metric_values,
        description, p_value, p_value_corrected, statistically_significant.

        When NO cell met the size gate the list is not empty: it carries a
        single ``no_cells_analysed`` entry, because an empty list cannot say
        whether every cell was examined and none showed a disparity or no cell
        was examined at all. Entries that tested no hypothesis (that one, and
        ``fpr_not_assessed``, and ``accurate_reflection``) carry ``p_value``,
        ``p_value_corrected`` and ``statistically_significant`` as None and are
        excluded from the Benjamini-Hochberg family.
    """
    findings: List[Dict[str, Any]] = []
    all_groups = intersectional.get("all_groups", [])
    if not all_groups:
        # BGL3 evaluation-4, 2026-09-27. An empty findings list was the answer to
        # BOTH "every cell was examined and none of them showed a disparity" and
        # "no cell was examined at all". The second is a could-not-check, and the
        # findings list is the boundary a report writer, the pulse envelope and
        # every JSON consumer read: the not-assessed state that
        # identify_privileged_groups puts in max_disparity=nan,
        # disparity_severity='not_assessed' and data_treatment does not travel
        # with a list.
        #
        # Measured before this branch, 20 rows in two 10-row cells at the default
        # min_group_size=30: identify_privileged_groups warns twice and reports
        # n_cells_included 0 of 2, and generate_structured_findings on its output
        # returned [] with NO warning and nothing naming the gap, which is
        # byte-identical to a clean 2000-row audit.
        #
        # Same treatment as the 'fpr_not_assessed' entry below, for the same
        # reason: the entry is not part of the multiple-comparison family (no
        # hypothesis was tested), so p_value is None.
        n_seen = 0
        treatment = intersectional.get("data_treatment") or {}
        if isinstance(treatment, dict):
            n_seen = int(treatment.get("n_total_cells_seen", 0) or 0)
        excluded = intersectional.get("excluded_groups") or []
        if not n_seen:
            n_seen = len(excluded)
        # BGL5 A-evaluation-4, 2026-09-27. THE STATE WAS RIGHT AND THE STATED
        # CAUSE WAS INVENTED. ``all_groups`` is empty whenever fewer than TWO
        # cells are analysable, not only when none is (identify_privileged_groups
        # returns all_groups=[] at ``len(group_data) < 2``), so this entry was
        # also the answer when a cell DID meet the gate, and it blamed the gate
        # anyway.
        #
        # Measured before this change, one 100-row cell at min_group_size=30:
        #     data_treatment -> n_total_cells_seen 1, n_cells_included 1,
        #                       n_cells_excluded_small 0, excluded_groups []
        #     description    -> "COULD NOT CHECK: none of the 1 cell(s) seen met
        #                       the minimum group size, so NO finding was
        #                       generated..." with metric_values
        #                       {'n_cells_analysed': 0, 'n_total_cells_seen': 1,
        #                       'n_cells_excluded_small': 0}
        # The sentence contradicted its own metric_values in the same dict:
        # nothing was excluded as small, so the one cell passed the gate. With a
        # partial drop (A 100, B 10, C 10) it said "none of the 3 cell(s) seen
        # met the minimum group size" while n_cells_excluded_small was 2.
        #
        # After: the cause comes from n_cells_included, which this function
        # already reads the same dict for, and n_cells_analysed reports it
        # instead of a hardcoded 0. The upstream warning has stated it correctly
        # all along ("only 1 cell(s) met min_group_size=30. No disparity was
        # computed"). A disclosure that misdescribes its own cause is a statement
        # nobody measured, which is what this batch rejected a sibling helper for
        # in _warn_groups_absent_from_ranking.
        n_included: Optional[int] = None
        if isinstance(treatment, dict) and treatment.get("n_cells_included") is not None:
            n_included = int(treatment["n_cells_included"])
        if n_included is None:
            cause = (
                f"this analysis reports no cell inventory, so WHY no cell was "
                f"comparable is itself not known ({n_seen} cell(s) seen, "
                f"{len(excluded)} excluded as too small)"
            )
        elif n_included == 0:
            cause = (
                f"not one cell met the size gate: none of the {n_seen} cell(s) seen "
                f"met the minimum group size ({len(excluded)} excluded as too small)"
            )
        else:
            cause = (
                f"only {n_included} of the {n_seen} cell(s) seen met the minimum "
                f"group size ({len(excluded)} excluded as too small), and a "
                f"disparity is a COMPARISON, so there was no second cell to "
                f"compare against"
            )
        warnings.warn(
            f"generate_structured_findings: {cause}, so no finding could be "
            "generated at all. The returned list carries a 'no_cells_analysed' "
            "entry: this is COULD NOT CHECK, NOT a finding that the model shows no "
            "disparity.",
            UserWarning,
            stacklevel=2,
        )
        findings.append(
            {
                "type": "no_cells_analysed",
                "severity": "info",
                # The excluded cells are dicts here (identify_privileged_groups
                # builds excluded_summary), but a hand-built analysis dict may
                # carry bare names, and refusing to name them would defeat the
                # point of the entry.
                "groups": [
                    str(g.get("group", "")) if isinstance(g, dict) else str(g) for g in excluded
                ],
                "metric_values": {
                    # The count that was MEASURED, not a hardcoded 0: it was 0
                    # for an analysis in which one cell had been analysed.
                    "n_cells_analysed": 0 if n_included is None else n_included,
                    "n_total_cells_seen": n_seen,
                    "n_cells_excluded_small": len(excluded),
                },
                "unmeasured": {
                    "all_findings": (
                        f"{cause}, so no disparity, FPR or amplification hypothesis "
                        f"was examined in this analysis"
                    )
                },
                "description": (
                    f"COULD NOT CHECK: {cause}, so NO finding was generated and "
                    "nothing here says anything about the model. An empty findings "
                    "list must not be read as 'no disparity was found'."
                ),
                "p_value": None,
                "p_value_corrected": None,
                "statistically_significant": None,
                "not_tested_reason": (
                    "no cell was analysable, so no hypothesis was tested and this "
                    "entry is not part of the multiple-comparison family"
                ),
                "n_groups_tested": 0,
            }
        )
        return findings

    is_unfavorable = outcome_polarity == "positive_unfavorable"
    outcome_word = "unfavorable" if is_unfavorable else "favorable"

    # Extract group attributes (handle both dataclass and dict)
    def _attr(g, key, default=0.0):
        return getattr(g, key, None) if hasattr(g, key) else g.get(key, default)

    # total_n counts only the INCLUDED groups (those meeting min_group_size),
    # matching each group's stated sample size. Used below for the power/MDE
    # warnings and, indirectly, the FPR baseline (which weights by negatives).
    total_n = sum(_attr(g, "size", 0) for g in all_groups)

    # Per-group over/under-prediction findings
    for g in all_groups:
        delta = _attr(g, "prediction_delta")
        gt_rate = _attr(g, "ground_truth_rate")
        pred_rate = _attr(g, "positive_rate")
        group_name = _attr(g, "group", "")
        fpr = _attr(g, "false_positive_rate")
        size = int(_attr(g, "size", 0))

        # BGL-S2B (2026-09-17). metric_values carries the cell's FPR, and that
        # value is NaN when the cell has no actual negatives. The reason lived
        # ONLY on GroupAdvantage.unmeasured, so a consumer reading the findings
        # list alone (the report writer, the pulse envelope, any JSON boundary)
        # saw a bare NaN with nothing saying it was never measured. Every
        # over/under-prediction finding now carries its own `unmeasured` map,
        # empty when all of its metric_values are real measurements.
        finding_unmeasured: Dict[str, str] = {}
        if not is_measured(fpr):
            cell_unmeasured = _attr(g, "unmeasured", {}) or {}
            finding_unmeasured["false_positive_rate"] = cell_unmeasured.get(
                "false_positive_rate",
                "no actual negatives in this cell, so the false positive rate "
                "has an empty denominator and was never measured",
            )

        if abs(delta) > 0.05:
            # Statistical test of the hypothesis this finding actually makes:
            # does the group's prediction rate differ significantly from its
            # OWN ground-truth (base) rate? That is the within-group
            # over/under-prediction the description states ("actual outcome
            # rate: gt, predicted rate: pred"). Both rates are measured on the
            # same `size` individuals, so this is a two-proportion comparison
            # of pred_rate vs gt_rate at n = size (Agresti, Categorical Data
            # Analysis). Testing pred_rate against the population selection
            # rate answered a DIFFERENT question (a between-group selection
            # disparity) and decoupled significance from the claim (audit3).
            p_val = _proportion_test(pred_rate, size, gt_rate, size)

            if delta > 0:
                direction = "over-predicted" if not is_unfavorable else "over-flagged"
                finding_type = "over_prediction"
                desc = (
                    f"'{group_name}' is {direction} by {abs(delta):.1%}. "
                    f"Actual {outcome_word} outcome rate: {gt_rate:.1%}, "
                    f"predicted rate: {pred_rate:.1%}."
                )
            else:
                direction = "under-predicted" if not is_unfavorable else "under-flagged"
                finding_type = "under_prediction"
                desc = (
                    f"'{group_name}' is {direction} by {abs(delta):.1%}. "
                    f"Actual {outcome_word} outcome rate: {gt_rate:.1%}, "
                    f"predicted rate: {pred_rate:.1%}."
                )

            findings.append(
                {
                    "type": finding_type,
                    "severity": _assess_severity(abs(delta)),
                    "groups": [group_name],
                    "metric_values": {
                        "ground_truth_rate": round(gt_rate, 4),
                        "prediction_rate": round(pred_rate, 4),
                        "delta": round(delta, 4),
                        "false_positive_rate": round(fpr, 4),
                    },
                    "unmeasured": finding_unmeasured,
                    "description": desc,
                    "p_value": round(p_val, 6),
                    "n_groups_tested": len(all_groups),
                }
            )

    # FPR disparity findings (each group vs overall FPR)
    # Test each group's FPR against the overall FPR, not max-vs-min (avoids cherry-picking bias)
    # FPR is a proportion over ACTUAL NEGATIVES, so the pooled baseline and
    # the significance test must weight each group by its negative count,
    # not its total size (a group that is 90% positive contributed 10x too
    # much weight to the baseline before). Negatives are reconstructed from
    # size * (1 - ground_truth_rate), which is exact up to float rounding.
    #
    # BGL-S2B (2026-09-17). The skip test used to read `if group_fpr is None`,
    # which is FALSE for the NaN this module now returns for a cell with no
    # actual negatives. One unmeasurable cell therefore entered the pooled
    # baseline, `overall_fpr` became NaN, every `fpr_delta` became NaN, and
    # `nan > 0.05` is False, so EVERY fpr_disparity finding in the run was
    # deleted -- including fully measured, critical ones. Measured at the public
    # entry (intersectional_disparity_analysis, 250 rows): with cells A (FPR
    # 0.900) and B (FPR 0.033) alone, 2 fpr_disparity findings, both severity
    # critical; adding one 50-row cell C whose labels are all positive (FPR
    # undefined), 0 findings, silently. A cell nobody could measure must be
    # DROPPED before the baseline and NAMED in the output, never allowed to
    # erase the cells that were measured.
    fprs_valid = []
    fpr_not_compared: List[Dict[str, Any]] = []
    for g in all_groups:
        group_fpr = _attr(g, "false_positive_rate")
        size = int(_attr(g, "size", 0))
        gt_rate = float(_attr(g, "ground_truth_rate", 0.0) or 0.0)
        if not is_measured(group_fpr):
            cell_unmeasured = _attr(g, "unmeasured", {}) or {}
            fpr_not_compared.append(
                {
                    "group": _attr(g, "group", ""),
                    "size": size,
                    "ground_truth_rate": round(gt_rate, 4),
                    "reason": cell_unmeasured.get(
                        "false_positive_rate",
                        "the false positive rate for this cell is not a finite "
                        "number, so it could not enter the comparison",
                    ),
                }
            )
            continue
        n_negatives = int(round(size * (1.0 - gt_rate)))
        fprs_valid.append((g, float(group_fpr), n_negatives))

    if fpr_not_compared:
        # The cells that WERE compared are still compared (see above); this entry
        # only says which cells were left out, so "no fpr_disparity finding for
        # group C" can never be read as "group C was checked and was fine".
        skipped_names = [str(c["group"]) for c in fpr_not_compared]
        findings.append(
            {
                "type": "fpr_not_assessed",
                "severity": "info",
                "groups": skipped_names,
                "metric_values": {
                    "n_cells_not_compared": len(fpr_not_compared),
                    "n_cells_compared": len(fprs_valid),
                },
                "description": (
                    f"The false positive rate could not be measured for "
                    f"{len(fpr_not_compared)} of {len(all_groups)} analysed cell(s) "
                    f"({', '.join(skipped_names)}), so "
                    + ("they were" if len(fpr_not_compared) > 1 else "it was")
                    + " left out of the FPR comparison entirely. "
                    + (
                        f"The comparison still ran on the remaining "
                        f"{len(fprs_valid)} measured cell(s)."
                        if len(fprs_valid) >= 2
                        else f"Only {len(fprs_valid)} cell(s) kept a measurable FPR, "
                        "so NO FPR comparison was run at all in this analysis."
                    )
                    + " This is COULD NOT CHECK for those cells, not a finding that "
                    "their FPR is in line with the rest."
                ),
                "fpr_comparison_ran": len(fprs_valid) >= 2,
                "p_value": None,
                "not_tested_reason": (
                    "no FPR was measurable for these cells, so no FPR hypothesis was "
                    "tested for them and this entry is not part of the "
                    "multiple-comparison family"
                ),
                "cells_not_compared": fpr_not_compared,
                "n_groups_tested": len(all_groups),
            }
        )
        warnings.warn(
            "generate_structured_findings: the false positive rate could not be "
            f"measured for {len(fpr_not_compared)} cell(s) "
            f"({', '.join(skipped_names)}); they were excluded from the FPR "
            f"baseline, "
            + (
                f"which was then pooled over the remaining {len(fprs_valid)} measured cell(s)"
                if len(fprs_valid) >= 2
                else f"leaving only {len(fprs_valid)} measurable cell(s), too few "
                "for any FPR comparison"
            )
            + ". See the 'fpr_not_assessed' finding.",
            UserWarning,
            stacklevel=2,
        )

    if len(fprs_valid) >= 2:
        total_negatives = sum(n for _, _, n in fprs_valid)
        overall_fpr = sum(f * n for _, f, n in fprs_valid) / max(1, total_negatives)
        for g, group_fpr, group_negatives in fprs_valid:
            fpr_delta = abs(group_fpr - overall_fpr)
            if fpr_delta > 0.05:
                group_name = _attr(g, "group", "")
                direction = "above" if group_fpr > overall_fpr else "below"
                p_val = _proportion_test(group_fpr, group_negatives, overall_fpr, total_negatives)
                findings.append(
                    {
                        "type": "fpr_disparity",
                        "severity": _assess_severity(fpr_delta),
                        "groups": [group_name],
                        "metric_values": {
                            "group_fpr": round(group_fpr, 4),
                            "overall_fpr": round(overall_fpr, 4),
                            "fpr_delta": round(fpr_delta, 4),
                        },
                        "description": (
                            f"'{group_name}' has a false positive rate of {group_fpr:.1%}, "
                            f"which is {fpr_delta:.1%} {direction} the overall FPR of {overall_fpr:.1%}. "
                            f"Among individuals who are actually negative in this group, "
                            f"{'more' if group_fpr > overall_fpr else 'fewer'} are wrongly flagged."
                        ),
                        "p_value": round(p_val, 6),
                        "n_groups_tested": len(all_groups),
                    }
                )

    # Amplified vs reflected bias finding
    pred_disparity = intersectional.get("max_disparity", 0)
    gt_disparity = intersectional.get("max_ground_truth_disparity", 0)

    if pred_disparity > 0.05 and gt_disparity > 0.01:
        amplification = pred_disparity - gt_disparity
        if amplification > 0.03:
            # Test: is the prediction disparity significantly larger than ground truth disparity?
            priv = intersectional.get("privileged_group")
            disadv = intersectional.get("disadvantaged_group")
            priv_size = int(_attr(priv, "size", 0)) if priv else 0
            disadv_size = int(_attr(disadv, "size", 0)) if disadv else 0
            priv_pred = float(_attr(priv, "positive_rate", 0)) if priv else 0
            disadv_pred = float(_attr(disadv, "positive_rate", 0)) if disadv else 0
            p_val = _proportion_test(priv_pred, priv_size, disadv_pred, disadv_size)
            findings.append(
                {
                    "type": "amplified_bias",
                    "severity": _assess_severity(amplification),
                    "groups": [],
                    "metric_values": {
                        "prediction_disparity": round(pred_disparity, 4),
                        "ground_truth_disparity": round(gt_disparity, 4),
                        "amplification": round(amplification, 4),
                    },
                    "description": (
                        f"The model amplifies existing disparities by {amplification:.1%}. "
                        f"Ground truth disparity is {gt_disparity:.1%}, but prediction disparity "
                        f"is {pred_disparity:.1%}. The algorithm makes the gap wider than reality."
                    ),
                    "p_value": round(p_val, 6),
                    "n_groups_tested": len(all_groups),
                }
            )
        elif amplification < -0.03:
            findings.append(
                {
                    "type": "accurate_reflection",
                    "severity": "info",
                    "groups": [],
                    "metric_values": {
                        "prediction_disparity": round(pred_disparity, 4),
                        "ground_truth_disparity": round(gt_disparity, 4),
                        "reduction": round(abs(amplification), 4),
                    },
                    "description": (
                        f"The model partially mitigates existing disparities. "
                        f"Ground truth disparity is {gt_disparity:.1%}, but prediction disparity "
                        f"is {pred_disparity:.1%} (reduced by {abs(amplification):.1%})."
                    ),
                    # READINESS-6, 2026-09-10. NO HYPOTHESIS WAS TESTED HERE, so
                    # this finding carries no p-value. It used to carry a
                    # hardcoded 1.0 annotated "Informational, not a disparity
                    # claim", which is exactly right about the finding and
                    # exactly wrong about the number: a neutral p is still a
                    # FAMILY MEMBER, and the correction below charges every real
                    # finding for it. Measured this day: a 7-point
                    # over-prediction on 660 people had p=0.008968, which the
                    # tests that RAN (m=5) correct to 0.044840 and report
                    # SIGNIFICANT; adding this one informational entry (m=6)
                    # corrects it to 0.053808 and reports NOT significant. A
                    # real finding was buried by an entry that makes no claim.
                    "p_value": None,
                    "not_tested_reason": (
                        "informational finding: the model MITIGATES the ground-truth "
                        "disparity, so no disparity hypothesis was tested and this "
                        "entry is not part of the multiple-comparison family"
                    ),
                    "n_groups_tested": len(all_groups),
                }
            )

    # Apply Benjamini-Hochberg FDR correction across the findings that actually
    # TESTED something. An entry with no p-value, or with a p-value that is not a
    # finite number, did not run a hypothesis: including it inflates the family
    # size m and so penalises every finding that did run. Same rule, and the same
    # reason, as `_statistics._testable`.
    # Uses vfairness library function (not inline copy)
    if findings:
        tested_idx = [
            i
            for i, f in enumerate(findings)
            if f.get("p_value") is not None and np.isfinite(float(f["p_value"]))
        ]
        for f in findings:
            # THREE STATES on the field a reader acts on. None is "no hypothesis
            # was tested here", which is neither a confirmed disparity nor a
            # cleared one, and must not be rendered as the measured False the
            # tested findings carry.
            f.setdefault("p_value_corrected", None)
            f.setdefault("statistically_significant", None)
        if tested_idx:
            raw_p_values = np.asarray([float(findings[i]["p_value"]) for i in tested_idx])
            bh_result = benjamini_hochberg_correction(raw_p_values, alpha=alpha)
            for k, i in enumerate(tested_idx):
                findings[i]["p_value_corrected"] = round(float(bh_result.adjusted_p_values[k]), 6)
                findings[i]["statistically_significant"] = bool(bh_result.rejection_mask[k])

    # Add Cohen's h effect size and power warnings
    for f in findings:
        # Every finding carries the key, so a consumer reads one place to learn
        # which of its metric_values are NOT measurements. Empty means all of
        # them are.
        f.setdefault("unmeasured", {})
        mv = f.get("metric_values", {})
        groups = f.get("groups", [])

        # Cohen's h for proportion differences
        if f["type"] in ("over_prediction", "under_prediction"):
            pred_r = mv.get("prediction_rate", 0)
            gt_r = mv.get("ground_truth_rate", 0)
            h = cohens_h(pred_r, gt_r)
            f["effect_size_h"] = round(h, 4)
            f["effect_size_interpretation"] = cohens_h_interpretation(h)
        elif f["type"] == "fpr_disparity":
            group_fpr = mv.get("group_fpr", 0)
            overall_fpr = mv.get("overall_fpr", 0)
            h = cohens_h(group_fpr, overall_fpr)
            f["effect_size_h"] = round(h, 4)
            f["effect_size_interpretation"] = cohens_h_interpretation(h)

        # Power warning for small groups
        if groups:
            group_obj = next((g for g in all_groups if _attr(g, "group") == groups[0]), None)
            if group_obj:
                gsize = int(_attr(group_obj, "size", 0))
                pw = power_warning(gsize, total_n, alpha=alpha)
                if pw:
                    f["power_warning"] = pw
                f["minimum_detectable_effect"] = round(
                    minimum_detectable_effect(gsize, total_n, alpha=alpha), 4
                )

    # Confidence intervals on disparity estimates
    # Uses bayesian_difference_ci for small groups (fast, no resampling)
    # and bootstrap for larger groups (200 iterations for speed).
    # Provides uncertainty bounds so users know how precise the estimates are.
    for f in findings:
        mv = f.get("metric_values", {})
        groups = f.get("groups", [])
        try:
            if f["type"] in ("over_prediction", "under_prediction") and groups:
                group_obj = next((g for g in all_groups if _attr(g, "group") == groups[0]), None)
                if group_obj:
                    gsize = int(_attr(group_obj, "size", 0))
                    pred_r = mv.get("prediction_rate", 0)
                    gt_r = mv.get("ground_truth_rate", 0)
                    pred_successes = int(round(pred_r * gsize))
                    gt_successes = int(round(gt_r * gsize))
                    # CI on the delta (prediction_rate - ground_truth_rate)
                    ci_result = bayesian_difference_ci(
                        pred_successes,
                        gsize,
                        gt_successes,
                        gsize,
                        confidence_level=1.0 - alpha,
                        n_samples=2000,
                    )
                    f["confidence_interval"] = {
                        "lower": round(float(ci_result.lower_bound), 4),
                        "upper": round(float(ci_result.upper_bound), 4),
                        "level": round(1.0 - alpha, 2),
                        "method": "bayesian_difference",
                    }

            elif f["type"] == "fpr_disparity" and groups:
                group_obj = next((g for g in all_groups if _attr(g, "group") == groups[0]), None)
                if group_obj:
                    gsize = int(_attr(group_obj, "size", 0))
                    group_fpr = mv.get("group_fpr", 0)
                    overall_fpr = mv.get("overall_fpr", 0)
                    g_successes = int(round(group_fpr * gsize))
                    o_successes = int(round(overall_fpr * total_n))
                    ci_result = bayesian_difference_ci(
                        g_successes,
                        gsize,
                        o_successes,
                        total_n,
                        confidence_level=1.0 - alpha,
                        n_samples=2000,
                    )
                    f["confidence_interval"] = {
                        "lower": round(float(ci_result.lower_bound), 4),
                        "upper": round(float(ci_result.upper_bound), 4),
                        "level": round(1.0 - alpha, 2),
                        "method": "bayesian_difference",
                    }

            elif f["type"] == "amplified_bias":
                # CI on the amplification (pred_disparity - gt_disparity)
                # Use simple normal approximation for the disparity difference
                pred_disp = mv.get("prediction_disparity", 0)
                gt_disp = mv.get("ground_truth_disparity", 0)
                amplification = pred_disp - gt_disp
                # Approximate SE from group sizes
                priv_obj = intersectional.get("privileged_group")
                disadv_obj = intersectional.get("disadvantaged_group")
                n_priv = int(_attr(priv_obj, "size", 50)) if priv_obj else 50
                n_disadv = int(_attr(disadv_obj, "size", 50)) if disadv_obj else 50
                se = np.sqrt(
                    pred_disp * (1 - pred_disp) * (1 / n_priv + 1 / n_disadv)
                    + gt_disp * (1 - gt_disp) * (1 / n_priv + 1 / n_disadv)
                )
                z_val = 1.96 if alpha == 0.05 else (2.576 if alpha == 0.01 else 1.645)
                f["confidence_interval"] = {
                    "lower": round(amplification - z_val * se, 4),
                    "upper": round(amplification + z_val * se, 4),
                    "level": round(1.0 - alpha, 2),
                    "method": "normal_approximation",
                }
        except Exception as exc:
            # CI computation is best-effort: never block a finding on a CI
            # failure, but surface it with a warning rather than swallowing it
            # silently. An audit tool must not hide a computation error (VB-SEC-4).
            warnings.warn(
                "Intersectional confidence-interval computation failed; the "
                f"finding is reported without a confidence interval ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )

    # Sort: significant findings first (by severity), then non-significant
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings.sort(
        key=lambda f: (
            0 if f.get("statistically_significant", False) else 1,
            severity_order.get(f["severity"], 5),
        )
    )

    return findings


def _warn_groups_absent_from_ranking(gm: GroupManager) -> None:
    """Say which groups the size gate kept OUT of a returned ranking.

    ``GroupManager`` warns only when EVERY group fails the gate, so a partial
    drop was silent, and a ranking is exactly where that silence does damage: a
    caller reads the list top to bottom, finds the worst value in it and calls
    that the worst group. The shut-out group is not the last row, it is not a
    row at all.

    Measured 2026-09-27 on 65 rows, ``A`` 60 rows selected at 100 percent beside
    ``B`` 5 rows selected at 0 percent, at the default min_group_size=30:
    ``get_group_rankings`` returned exactly one row,
    ``[{'group': 'A', 'value': 1.0, 'size': 60, 'rank': 1}]``, with ZERO
    warnings, so max-minus-min over the returned list is 0.0, i.e. perfect
    parity, over data whose only disparity is the group that is missing.

    Not ``classification._warn_dropped_groups``, which the ranking and
    regression modules share: its single-survivor branch states that "the metric
    is NOT computed, it returns NaN and the run is not assessable". That is true
    of a between-group metric and false here, and a warning that misdescribes
    its own function is worse than none. Same counts, same shape, different
    consequence clause.
    """
    dropped = gm.get_invalid_groups()
    survivors = gm.get_valid_groups(warn_if_empty=False)
    if not (dropped and survivors):
        return
    sizes = gm.get_group_sizes()
    dropped_sizes = {name: sizes[name] for name in dropped}
    n_dropped_rows = sum(dropped_sizes.values())
    n_rows = sum(sizes.values())
    share = (100.0 * n_dropped_rows / n_rows) if n_rows else 0.0
    consequence = (
        f"Only 1 group ({survivors[0]!r}) meets the gate, so the returned ranking "
        f"ranks nothing against anything: a spread taken over it is a spread over "
        f"one group."
        if len(survivors) == 1
        else "The ranking covers the surviving groups only."
    )
    warnings.warn(
        f"get_group_rankings: {len(dropped)} group(s) are ABSENT from the returned "
        f"ranking because they are below min_group_size={gm.min_group_size}: "
        f"{dropped_sizes}. That leaves {n_dropped_rows} of {n_rows} rows "
        f"({share:.1f} percent) out of the ranking entirely. {consequence} The "
        f"excluded group(s) are could not check, NOT a measured pass, and the best "
        f"and worst rows below are the best and worst of what remained. To rank "
        f"them too, lower min_group_size.",
        UserWarning,
        # Attribute this to the line that asked for the ranking, not to this
        # helper: one shared location makes Python's once-per-location filter
        # collapse a whole report's worth of drops into a single line.
        stacklevel=3,
    )


def _warn_unmeasured_ranking_cells(rankings: List[Dict[str, Any]], metric: str) -> None:
    """Say which cells are in the list WITHOUT a measured value or a rank.

    The size-gate drop is disclosed by ``_warn_groups_absent_from_ranking``; this
    is the other way a cell fails to be measured, and it was silent. A cell that
    passes the size gate but whose metric has an empty denominator (no actual
    negatives, so no FPR; no actual positives, so no TPR; nothing predicted
    positive, so no precision) has no value to rank, and the old code sorted its
    NaN to the end and numbered it anyway.

    Kept separate from the size-gate warning because the consequence is
    different: that group IS in the returned list, with its size, and only its
    position and value are withheld.
    """
    unmeasured = [r for r in rankings if r.get("rank") is None]
    if not unmeasured:
        return
    named = {str(r["group"]): r["size"] for r in unmeasured}
    n_ranked = len(rankings) - len(unmeasured)
    warnings.warn(
        f"get_group_rankings: {len(unmeasured)} group(s) are in the returned list "
        f"with NO rank and a NaN {metric}, because that metric has an empty "
        f"denominator for them: {named} (group: rows). They are could not check, NOT "
        f"a measured worst place: the ranking orders the {n_ranked} cell(s) that were "
        f"measured, and the bottom row of THAT order is the worst measured group. "
        f"Each such row carries an 'unmeasured' clause naming the empty denominator.",
        UserWarning,
        # Attribute this to the line that asked for the ranking, not to this
        # helper, for the reason _warn_groups_absent_from_ranking records.
        stacklevel=3,
    )


def get_group_rankings(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    metric: Literal["positive_rate", "tpr", "fpr", "precision"] = "positive_rate",
    min_group_size: int = 30,
    missing_strategy: MissingStrategy = "exclude",
) -> List[Dict[str, Any]]:
    """
    Get all groups ranked by a specific metric.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Protected attribute values
        metric: Metric to rank by ('positive_rate', 'tpr', 'fpr', 'precision')
        min_group_size: Minimum samples required per group
        missing_strategy: How to handle missing values

    Returns:
        List of dicts with group info, sorted by metric (descending), with cells
        whose metric is undefined (NaN) at the end.

        Groups below ``min_group_size`` are ABSENT from the list, and that
        omission is disclosed as a ``UserWarning`` naming them, their sizes and
        the share of rows they hold: a spread taken over the returned rows is a
        spread over the groups that survived the gate, not over the population.

        A cell that passes the size gate but whose metric has an EMPTY
        DENOMINATOR keeps its row (group, size, NaN value) and gets ``rank``
        None plus an ``unmeasured`` clause, never a rank number: ranks number
        the cells that were measured. That omission is disclosed too.

    Example:
        >>> rankings = get_group_rankings(y_true, y_pred, demographics, metric='tpr')
        >>> for i, group in enumerate(rankings, 1):
        ...     print(f"{i}. {group['group']}: TPR={group['value']:.1%}")
    """
    y_true_v, y_pred_v, sensitive_attr_v, _, info = validate_inputs(
        y_true,
        y_pred,
        sensitive_attr,
        task_type="classification",
        missing_strategy=missing_strategy,
    )

    gm = GroupManager(sensitive_attr_v, min_group_size=min_group_size)
    # A group below the size gate is simply ABSENT from the returned list, which
    # reads as a complete ranking of the population. Disclosed for the same
    # reason ranking.get_ranking_group_metrics discloses its own drops.
    _warn_groups_absent_from_ranking(gm)
    rankings = []

    for name in gm.get_valid_groups():
        mask = gm.get_mask(name)
        yt = y_true_v[mask]
        yp = y_pred_v[mask]

        # Compute requested metric. ``denominator`` is what the rate is taken
        # OVER, kept so an empty one can be named rather than left as a bare NaN.
        if metric == "positive_rate":
            value = np.mean(yp)
            denominator = f"{len(yp)} row(s)"
        elif metric == "tpr":
            tp = np.sum((yp == 1) & (yt == 1))
            fn = np.sum((yp == 0) & (yt == 1))
            value = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            denominator = f"{int(tp + fn)} actual positive(s) among {len(yt)} row(s)"
        elif metric == "fpr":
            fp = np.sum((yp == 1) & (yt == 0))
            tn = np.sum((yp == 0) & (yt == 0))
            value = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            denominator = f"{int(fp + tn)} actual negative(s) among {len(yt)} row(s)"
        elif metric == "precision":
            tp = np.sum((yp == 1) & (yt == 1))
            fp = np.sum((yp == 1) & (yt == 0))
            value = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            denominator = f"{int(tp + fp)} predicted positive(s) among {len(yt)} row(s)"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        entry: Dict[str, Any] = {
            "group": name,
            "value": value,
            "size": len(yt),
            "metric": metric,
        }
        if not np.isfinite(value):
            entry["unmeasured"] = (
                f"{metric} was never measured for this cell: it has {denominator}, so "
                f"the rate has an empty denominator. No rank was assigned, because a "
                f"rank is a position relative to cells that WERE measured."
            )
        rankings.append(entry)

    # Sort by value (descending), NaN values at end
    rankings.sort(
        key=lambda x: (np.isnan(x["value"]), -x["value"] if not np.isnan(x["value"]) else 0)
    )

    # Add rank. BGL5 A-evaluation-4, 2026-09-27. A CELL WHOSE METRIC WAS NEVER
    # MEASURED IS NOT LAST, IT IS NOT IN THE ORDER AT ALL. ``np.argsort``-style
    # NaN-to-the-end sorting plus ``enumerate`` handed it a rank number anyway,
    # and in a descending FPR ranking the bottom position is the one a reader
    # calls "least often wrongly flagged".
    #
    # Measured before this change, A 60 rows with every label positive (so no
    # actual negatives and no FPR exists) beside B 60 rows with a measured FPR,
    # min_group_size=30, under warnings.simplefilter("error"):
    #     metric='fpr' -> [('B', 0.5, rank 1), ('A', nan, rank 2)], ZERO
    #                     warnings, and no field on the row saying it was not
    #                     measured
    #     metric='tpr' -> [('A', 1.0, 1), ('B', nan, 2)]
    # After: rank None on that row, an ``unmeasured`` clause naming the empty
    # denominator, and a warning. The sibling module calls the old behaviour the
    # defect in so many words (ranking._undefined_order_reason: "an item whose
    # score is missing is silently ranked LAST, a position nobody measured,
    # graded as the worst one").
    n_measured = 0
    for r in rankings:
        if "unmeasured" in r:
            r["rank"] = None
            continue
        n_measured += 1
        r["rank"] = n_measured
    _warn_unmeasured_ranking_cells(rankings, metric)

    return rankings
