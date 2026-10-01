"""QUARANTINED legacy Pulse pipeline. DO NOT WIRE TO A CONSUMER.

Superseded by ``vfairness.operations.pulse.orchestrator.run_pulse``, the
single source of truth for Pulse analysis. This legacy path FABRICATES
metrics: when labels are absent it invents equalized odds (dp*0.9) and
predictive parity (dp*0.8), hardcodes a calibration gap of 0.06, and
builds its Pareto frontier from random prediction flips. Shipping those
numbers as audit output violates the platform's never-invent-scores rule.

``run_pulse_pipeline`` now raises immediately. The module body is kept
readable ONLY as reference for two pending ports (legal admissibility
classification and the SVG adapters); port them into the orchestrator,
never re-enable this entry point.

Original description (historical): "Pulse pipeline -- the real fairness
compute the React UI renders." It deliberately stayed inside numpy +
pandas (no heavy optional deps).

Pipeline order:
    1. Resolve predictions / labels / sensitive columns
    2. Sanity-check data quality (basic completeness + class balance)
    3. Pick reference vs worst group on Demographic Parity
    4. Bootstrap BCa confidence intervals for the headline metrics
    5. Intersectional grid (primary x secondary)
    6. Pareto frontier over score thresholds
    7. Heuristic bias-mechanism tagging
    8. Causal DAG (proxy detection via Cramer's V / mutual information)
    9. Recommended fairness definition based on domain
   10. Plain-language verdict + next-step copy

Every value in the returned dict is JSON-serialisable so the consumer can
POST it directly to ``submit-result``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    metric_direction,
)

PRED_COL_CANDIDATES = ("prediction", "predicted", "pred", "y_pred", "score", "decision")
LABEL_COL_CANDIDATES = ("label", "y", "y_true", "target", "outcome", "ground_truth")
SCORE_COL_CANDIDATES = ("score", "probability", "prob", "y_score")


# Result tone helpers


def _tone_for_gap(gap: float, threshold: float = 0.10) -> str:
    if gap >= threshold * 2:
        return "critical"
    if gap >= threshold:
        return "warn"
    return "pass"


# The four Pulse headline cards are all gap magnitudes (max - min between
# groups), so lower is better for every one of them. Stated per key rather than
# assumed: _metric_tone refuses to grade a key it does not know, instead of
# defaulting an unrecognised metric to the lower-is-better comparison, which is
# what inverted the ratio family everywhere else in the codebase.
_PULSE_METRIC_DIRECTIONS: Dict[str, MetricDirection] = {
    "dp": MetricDirection.LOWER_IS_BETTER,  # demographic parity gap
    "eo": MetricDirection.LOWER_IS_BETTER,  # equal opportunity gap
    "cal": MetricDirection.LOWER_IS_BETTER,  # calibration gap
    "pp": MetricDirection.LOWER_IS_BETTER,  # predictive parity gap
}


def _metric_tone(
    value: float, ci_low: float, ci_high: float, threshold: float, metric_key: str
) -> str:
    """Tone for one metric card, in that metric's own direction.

    "critical" when even the FAVOURABLE end of the confidence interval is on the
    wrong side of the threshold (so the breach survives sampling error),
    "warn" when only the point estimate is, "pass" otherwise, and "unknown" when
    the metric's better-direction cannot be determined. "unknown" is deliberately
    NOT "pass": a card whose direction we cannot resolve has not been checked.
    """
    direction = _PULSE_METRIC_DIRECTIONS.get(metric_key) or metric_direction(metric_key)

    if direction is MetricDirection.HIGHER_IS_BETTER:
        # A ratio breaches BELOW its threshold, and its favourable end is
        # ci_high, so the bare `value > threshold` used here previously reported
        # "pass" for a ratio of 0.0 and "critical" for perfect parity.
        if ci_high < threshold:
            return "critical"
        if value < threshold:
            return "warn"
        return "pass"

    if direction is MetricDirection.LOWER_IS_BETTER:
        if ci_low > threshold:
            return "critical"
        if value > threshold:
            return "warn"
        return "pass"

    return "unknown"


# Bootstrap BCa CI (lightweight, percentile fallback)


def _bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_iter: int = 600,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Returns (point estimate, ci_low, ci_high) using the percentile bootstrap."""
    rng = rng or np.random.default_rng(0)
    point = statistic(values)
    if len(values) < 5:
        return point, point, point
    samples = rng.integers(0, len(values), size=(n_iter, len(values)))
    boots = np.empty(n_iter, dtype=float)
    for i, idx in enumerate(samples):
        boots[i] = statistic(values[idx])
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def _gap_bootstrap_ci(
    pred: np.ndarray,
    group: np.ndarray,
    a_label,
    b_label,
    statistic,
    n_iter: int = 600,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap the |stat(group=a) - stat(group=b)| gap."""
    rng = rng or np.random.default_rng(0)
    a_mask = group == a_label
    b_mask = group == b_label
    a = pred[a_mask]
    b = pred[b_mask]
    if len(a) < 5 or len(b) < 5:
        gap = abs(statistic(a) - statistic(b)) if len(a) and len(b) else 0.0
        return gap, gap, gap
    boots = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        ai = a[rng.integers(0, len(a), size=len(a))]
        bi = b[rng.integers(0, len(b), size=len(b))]
        boots[i] = abs(statistic(ai) - statistic(bi))
    point = abs(statistic(a) - statistic(b))
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(low), float(high)


# Column resolution


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # partial match
    for col in df.columns:
        cl = col.lower()
        if any(cl == c or cl.endswith("_" + c) or cl.startswith(c + "_") for c in candidates):
            return col
    return None


def _coerce_binary(series: pd.Series) -> Tuple[np.ndarray, Optional[float]]:
    """Returns a numpy array of {0,1} and the threshold used (if continuous)."""
    s = series.dropna()
    if s.empty:
        return np.array([], dtype=int), None
    if s.dtype == bool:
        return s.astype(int).to_numpy(), 0.5
    unique = s.unique()
    if len(unique) <= 2:
        # map larger / "yes" / "true" / "1" -> 1
        ordered = sorted(unique, key=lambda x: str(x))
        mapping = {ordered[0]: 0, ordered[-1]: 1}
        return s.map(mapping).fillna(0).astype(int).to_numpy(), None
    # continuous score -> threshold at 0.5 or median
    threshold = 0.5 if s.between(0, 1).mean() > 0.9 else float(s.median())
    return (s.to_numpy() >= threshold).astype(int), threshold


# Auto-binning for group analysis
#
# Mirrors the Navigator's data-ingestion behaviour in the platform UI:
#   1. If the dataset carries BOTH a continuous column ("age") AND a
#      pre-binned sibling ("age_group"), drop the continuous one and use the
#      grouped column -- otherwise every distinct value becomes its own
#      subgroup and the fairness read is meaningless.
#   2. If a continuous protected attribute has NO pre-binned sibling, bin it
#      automatically (age-aware brackets when the name looks like age,
#      quantile bins otherwise) so group fairness is computable.
#
# This is analysis-time grouping, NOT mitigation: the model and the data's
# substance are untouched; binning is only how disparities are *read*.

# Same suffix/prefix vocabulary the Navigator uses to spot a pre-binned twin.
_BIN_SUFFIXES = (
    "_group",
    "_bin",
    "_binned",
    "_bucket",
    "_bracket",
    "_band",
    "_range",
    "_cat",
    "_category",
    "_tier",
    "_level",
    "_class",
)
_BIN_PREFIXES = ("binned_", "grouped_", "cat_")

# Cardinality at/above which a numeric column is treated as continuous and
# must be binned before it can serve as a group key.
_MAX_GROUP_CARDINALITY = 12


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _looks_like_age(name: str) -> bool:
    n = name.lower().replace("_", "").replace("-", "")
    return "age" in n and not any(v in n for v in ("usage", "average", "page", "agent"))


def _find_prebinned_sibling(col: str, df: pd.DataFrame) -> Optional[str]:
    """Return a categorical column that is a pre-binned version of `col`."""
    a = col.lower()
    for other in df.columns:
        if other == col:
            continue
        b = other.lower()
        # name relation: one contains the other, or bin suffix/prefix on a
        # base that matches the continuous column's name.
        base_suffix = next((b[: -len(s)] for s in _BIN_SUFFIXES if b.endswith(s)), None)
        base_prefix = next((b[len(p) :] for p in _BIN_PREFIXES if b.startswith(p)), None)
        name_related = (
            (a in b or b in a)
            or (base_suffix is not None and len(base_suffix) >= 2 and base_suffix in a)
            or (base_prefix is not None and len(base_prefix) >= 2 and base_prefix in a)
        )
        if not name_related:
            continue
        # the sibling must be lower-cardinality and not itself continuous
        if df[other].nunique(dropna=True) <= _MAX_GROUP_CARDINALITY:
            return other
    return None


def _group_series(df: pd.DataFrame, col: str) -> pd.Series:
    """A categorical Series fit to be a group key.

    Continuous/high-cardinality numerics are auto-binned (age-aware brackets
    or quantile bins). Everything else is used as-is.
    """
    s = df[col]
    if _is_numeric(s) and s.nunique(dropna=True) > _MAX_GROUP_CARDINALITY:
        clean = pd.to_numeric(s, errors="coerce")
        if _looks_like_age(col):
            edges = [-np.inf, 18, 25, 35, 45, 55, 65, np.inf]
            labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            binned = pd.cut(clean, bins=edges, labels=labels, right=False)
            return binned.astype("string").fillna("missing")
        # generic continuous -> up to 5 quantile bins with readable edges
        try:
            binned = pd.qcut(clean, q=5, duplicates="drop")
        except (ValueError, IndexError):
            try:
                binned = pd.cut(clean, bins=4, duplicates="drop")
            except (ValueError, IndexError):
                return s.astype("string").fillna("missing")
        return binned.astype("string").fillna("missing")
    return s.astype("string").fillna("missing")


def _binning_disclosure(df: pd.DataFrame, col: str) -> Optional[str]:
    """Plain-language note when `col` was auto-binned for grouping.

    Audit-grade transparency: the fairness literature (Asudeh, "Unbiased
    Binning", arXiv:2509.21785; Mary et al., ICML 2019) shows results depend
    on the discretisation of continuous protected attributes, so the choice
    must be disclosed and auditable, never silent. Returns None when no
    binning was applied.
    """
    s = df[col]
    if not (_is_numeric(s) and s.nunique(dropna=True) > _MAX_GROUP_CARDINALITY):
        return None
    if _looks_like_age(col):
        return (
            f'"{col}" is a continuous number, so it was grouped into standard '
            "age brackets (<18, 18-24, 25-34, 35-44, 45-54, 55-64, 65+) before "
            "fairness was measured. Bracket choice can affect the result; these "
            "are the conventional brackets used in employment-fairness analysis."
        )
    return (
        f'"{col}" is a continuous number, so it was grouped into 5 equal-size '
        "(equal-frequency) bands before fairness was measured. Equal-frequency "
        "binning is the recommended neutral default; a different split could "
        "shift the numbers, which is why this is stated here."
    )


def _dedupe_binned_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Drop a continuous column when a pre-binned sibling is also present.

    Keeps the grouped column so subgroups are meaningful, matching the
    Navigator's "use H groups, not H absolute numbers" rule.
    """
    keep: List[str] = []
    dropped: set = set()
    for c in cols:
        if c in dropped:
            continue
        if (
            c in df.columns
            and _is_numeric(df[c])
            and df[c].nunique(dropna=True) > _MAX_GROUP_CARDINALITY
        ):
            sibling = _find_prebinned_sibling(c, df)
            if sibling is not None:
                # prefer the binned sibling; pull it into the list if the
                # user did not already include it.
                if sibling not in keep:
                    keep.append(sibling)
                dropped.add(c)
                continue
        keep.append(c)
    return keep


# Data quality
#
# Runs as the FIRST analytic stage (before any metric). A weak dataset makes
# every downstream number untrustworthy, so Pulse states the data's fitness
# up front rather than burying caveats. Plain language, no jargon: every
# check has a status and a one-line "what this means".

# Below this many rows in a compared group, a disparity estimate is too noisy
# to act on (standard rule of thumb; also the representation.py default).
_MIN_GROUP_SIZE = 30
# Below this many total rows the bootstrap CIs are wide enough that the
# verdict should be read as indicative only.
_MIN_TOTAL_ROWS = 200


def _data_quality(
    df: pd.DataFrame,
    pred_col: str,
    label_col: Optional[str],
    primary: str,
    primary_series: pd.Series,
    pred_arr: np.ndarray,
) -> Dict[str, Any]:
    """Thin wrapper: the canonical plain-language report is built by
    vfairness.operations.cicd.quality_report so Pulse and the Navigator
    share one source of truth. Signature kept for call-site stability."""
    from vfairness.operations.cicd.quality_report import build_quality_report

    return build_quality_report(df, [primary], outcome_column=label_col)


# Pipeline


@dataclass
class PulsePipelineInputs:
    df: pd.DataFrame
    system_type: str
    domain: str
    jurisdiction: str
    artifact_label: str
    protected_attributes: List[str]


def run_pulse_pipeline(inputs: PulsePipelineInputs) -> Dict[str, Any]:
    raise RuntimeError(
        "run_pulse_pipeline is quarantined: it fabricates metrics when labels "
        "are absent (eo=dp*0.9, pp=dp*0.8, calibration=0.06, random-flip "
        "Pareto). Use vfairness.operations.pulse.orchestrator.run_pulse via "
        "handle_pulse_run instead. See CONSUMER_REGISTRATION.md."
    )
    df = inputs.df.copy()
    if df.empty:
        raise ValueError("Pulse pipeline received an empty dataset.")

    pred_col = _pick_column(df, PRED_COL_CANDIDATES)
    label_col = _pick_column(df, LABEL_COL_CANDIDATES)
    if pred_col is None:
        raise ValueError(
            "No prediction column found. Pulse needs a column named "
            "'prediction' (or pred / y_pred / score / decision)."
        )
    available_protected = [c for c in inputs.protected_attributes if c in df.columns]
    if not available_protected:
        # Pick first low-cardinality column as a fallback so the pipeline can
        # still run end-to-end; the result will flag the limitation.
        candidates = [
            c
            for c in df.columns
            if c not in (pred_col, label_col) and 2 <= df[c].nunique(dropna=True) <= 10
        ]
        if not candidates:
            raise ValueError(
                "None of the protected attributes were found in the dataset and "
                "no low-cardinality column could substitute. Pulse cannot read "
                "fairness without a group column."
            )
        available_protected = candidates[:1]

    # Auto-binning step (mirrors the Navigator): when both a continuous column
    # and its pre-binned sibling are present, keep only the grouped one.
    available_protected = _dedupe_binned_columns(df, available_protected)
    available_protected = [c for c in available_protected if c in df.columns]

    pred_arr, _ = _coerce_binary(df[pred_col])

    # Primary group is the first protected attribute that has >= 2 values
    primary = next((c for c in available_protected if df[c].nunique() >= 2), available_protected[0])
    secondary = next(
        (c for c in available_protected if c != primary and df[c].nunique() >= 2),
        None,
    )

    # Group key: continuous attributes are auto-binned so disparities are
    # readable instead of one subgroup per distinct value.
    primary_series = _group_series(df, primary)

    # First analytic stage: data quality. Computed before any metric so the
    # result can state the data's fitness up front.
    data_quality = _data_quality(df, pred_col, label_col, primary, primary_series, pred_arr)

    # Audit transparency: if the assessed attribute was auto-binned, disclose
    # the method as an informational check (results depend on bin choice).
    _bin_note = _binning_disclosure(df, primary)
    if _bin_note:
        data_quality.setdefault("checks", []).append(
            {
                "key": "binning_disclosure",
                "label": "How groups were formed",
                "status": "info",
                "detail": _bin_note,
            }
        )

    rng = np.random.default_rng(abs(hash((inputs.artifact_label, primary))) % (2**32))

    # Reference vs worst group by selection rate.
    selection_rate = pd.Series(pred_arr, index=df.index).groupby(primary_series).mean()
    if selection_rate.empty:
        raise ValueError("Could not compute selection rate: pred column has no usable values.")
    reference_label = str(selection_rate.idxmax())
    worst_label = str(selection_rate.idxmin())
    reference_rate = float(selection_rate.max())
    worst_rate = float(selection_rate.min())

    # If secondary attribute exists, refine worst_label to a (primary, secondary) pair
    worst_pair = None
    if secondary is not None:
        secondary_series = _group_series(df, secondary)
        grid = pd.crosstab([primary_series, secondary_series], pred_arr).reindex(
            columns=[0, 1], fill_value=0
        )
        rates = grid[1] / grid.sum(axis=1).replace(0, np.nan)
        rates = rates.dropna()
        if not rates.empty:
            worst_idx = rates.idxmin()
            worst_pair = (str(worst_idx[0]), str(worst_idx[1]))

    # Headline metrics with bootstrap CIs
    def selection_rate_diff(arr: np.ndarray) -> float:
        return float(arr.mean()) if len(arr) else 0.0

    dp_value, dp_lo, dp_hi = _gap_bootstrap_ci(
        pred_arr,
        primary_series.to_numpy(),
        reference_label,
        worst_label,
        selection_rate_diff,
        n_iter=500,
        rng=rng,
    )

    # Equal opportunity needs labels; degrade gracefully if absent.
    eo_value, eo_lo, eo_hi = dp_value * 0.9, dp_lo * 0.9, dp_hi * 0.9
    pp_value, pp_lo, pp_hi = dp_value * 0.8, dp_lo * 0.8, dp_hi * 0.8
    cal_value, cal_lo, cal_hi = 0.06, 0.04, 0.08
    if label_col is not None:
        label_arr, _ = _coerce_binary(df[label_col])
        if len(label_arr) == len(pred_arr):
            pos = label_arr == 1
            if pos.any():
                eo_value, eo_lo, eo_hi = _gap_bootstrap_ci(
                    pred_arr[pos],
                    primary_series.to_numpy()[pos],
                    reference_label,
                    worst_label,
                    selection_rate_diff,
                    n_iter=500,
                    rng=rng,
                )
            pred_pos = pred_arr == 1
            if pred_pos.any():
                pp_value, pp_lo, pp_hi = _gap_bootstrap_ci(
                    label_arr[pred_pos],
                    primary_series.to_numpy()[pred_pos],
                    reference_label,
                    worst_label,
                    selection_rate_diff,
                    n_iter=500,
                    rng=rng,
                )
            # Calibration: absolute group-wise base-rate gap.
            base_rates = pd.Series(label_arr).groupby(primary_series.to_numpy()).mean()
            cal_value = float(base_rates.max() - base_rates.min()) if not base_rates.empty else 0.0
            cal_lo, cal_hi = max(0.0, cal_value - 0.02), cal_value + 0.02

    metrics = [
        _metric_dict(
            "dp",
            "Demographic parity gap",
            "Difference in approval rates between groups.",
            dp_value,
            dp_lo,
            dp_hi,
            threshold=0.10,
        ),
        _metric_dict(
            "eo",
            "Equal opportunity gap",
            "Difference in correct approvals between qualified people.",
            eo_value,
            eo_lo,
            eo_hi,
            threshold=0.10,
        ),
        _metric_dict(
            "cal",
            "Calibration gap",
            "How well the score predicts the outcome across groups.",
            cal_value,
            cal_lo,
            cal_hi,
            threshold=0.08,
        ),
        _metric_dict(
            "pp",
            "Predictive parity gap",
            "When the system says yes, how often is it right per group.",
            pp_value,
            pp_lo,
            pp_hi,
            threshold=0.10,
        ),
    ]

    # Intersectional grid
    rows, cols, cells, concentration = _intersectional_grid(
        df,
        primary,
        secondary,
        pred_arr,
        reference_rate,
    )

    # Pareto frontier
    pareto_points = _pareto_frontier(
        pred_arr, primary_series, reference_label, worst_label, label_col, df
    )

    # Bias mechanism tags
    bias_findings = _bias_findings(
        df, primary, reference_label, worst_label, primary_series, pred_arr, label_col
    )

    # Causal DAG
    causal = _causal_dag(df, primary, label_col, pred_col, secondary)

    # Recommended fairness definition
    recommended = _recommended_definition(inputs.domain)

    # Legal admissibility (universal; per (use_case x jurisdiction))
    # Always runs. When no rule pack matches the (use_case, jurisdiction) pair
    # the block reports coverage="uncovered" so the UI can stay honest about
    # the gap instead of fabricating a verdict.
    try:
        from vfairness.legal import classify_columns, map_domain_to_use_case

        legal_admissibility = classify_columns(
            list(df.columns),
            map_domain_to_use_case(inputs.domain),
            inputs.jurisdiction,
        )
    except Exception as exc:
        legal_admissibility = {
            "coverage": "uncovered",
            "useCase": "generic",
            "jurisdiction": inputs.jurisdiction,
            "jurisdictionId": None,
            "findings": [],
            "summary": {"forbidden": 0, "restricted": 0, "monitoringOnly": 0, "allowed": 0},
            "unmappedColumns": [],
            "error": f"legal admissibility check unavailable: {exc}",
        }

    # Verdict
    tone = _tone_for_gap(dp_value, threshold=0.10)
    worst_group_label = (
        f"{worst_pair[0]} · {worst_pair[1]}" if worst_pair else f"{primary}={worst_label}"
    )
    reference_group_label = f"{primary}={reference_label}"
    headline = {
        "critical": f"Material fairness issue for {worst_group_label}.",
        "warn": f"Watch-level fairness gap for {worst_group_label}.",
        "pass": "This system is within fairness budget.",
    }[tone]
    summary = (
        f"Out of every 100 {worst_group_label}, {round(worst_rate * 100)} get a positive decision. "
        f"Out of every 100 {reference_group_label}, {round(reference_rate * 100)} do."
    )
    next_step = {
        "critical": (
            f"Not fit to deploy for {worst_group_label} in {inputs.domain or 'this domain'}"
            + (f" under {inputs.jurisdiction}" if inputs.jurisdiction else "")
            + ". Convert this Pulse into a full assessment to document the finding "
            "and plan remediation."
        ),
        "warn": (
            "Deployable with active monitoring. Convert this Pulse to schedule "
            "recurring checks and pin a fairness budget."
        ),
        "pass": (
            "No blocking fairness issue detected. Convert this Pulse into a "
            "tracked assessment to keep the finding on the record."
        ),
    }[tone]
    # Jurisdiction-aware legal framing so the 80% rule is never presented as
    # universal (it is US-only; EU/UK use proportionality, no fixed bar).
    next_step = f"{next_step} {_legal_framing(inputs.jurisdiction)}"

    result: Dict[str, Any] = {
        "verdict": {
            "headline": headline,
            "summary": summary,
            "worstGroupLabel": worst_group_label,
            "worstGroupRate": worst_rate,
            "referenceGroupLabel": reference_group_label,
            "referenceGroupRate": reference_rate,
            "gapPoints": dp_value,
            "ciLow": dp_lo,
            "ciHigh": dp_hi,
            "tone": tone,
        },
        "dataQuality": data_quality,
        "metrics": metrics,
        "intersectional": {
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "concentration": concentration,
        },
        "pareto": pareto_points,
        "bias": bias_findings,
        "causal": causal,
        "recommendedDefinition": recommended,
        "legalAdmissibility": legal_admissibility,
        "nextStep": next_step,
    }
    return result


def _metric_dict(
    key: str, label: str, plain: str, value: float, ci_low: float, ci_high: float, threshold: float
) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "plain": plain,
        "value": value,
        "ciLow": max(0.0, ci_low),
        "ciHigh": ci_high,
        "threshold": threshold,
        "tone": _metric_tone(value, max(0.0, ci_low), ci_high, threshold, key),
    }


def _intersectional_grid(
    df: pd.DataFrame,
    primary: str,
    secondary: Optional[str],
    pred_arr: np.ndarray,
    reference_rate: float,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], float]:
    # Same auto-binning as the headline read, so the grid groups match.
    primary_series = _group_series(df, primary)
    if secondary is None:
        # Single-row grid: primary x "overall"
        rows = sorted(primary_series.unique().tolist())
        cols = ["overall"]
        cells: List[Dict[str, Any]] = []
        worst_gap = -1.0
        worst_idx = 0
        for r in rows:
            mask = primary_series == r
            rate = float(pred_arr[mask.to_numpy()].mean()) if mask.any() else 0.0
            gap = abs(reference_rate - rate)
            cells.append({"rowLabel": r, "colLabel": "overall", "gap": gap, "worst": False})
            if gap > worst_gap:
                worst_gap = gap
                worst_idx = len(cells) - 1
        if cells:
            cells[worst_idx]["worst"] = True
        total = sum(c["gap"] for c in cells)
        return rows, cols, cells, (worst_gap / total if total > 0 else 0.0)

    secondary_series = _group_series(df, secondary)
    rows = sorted(primary_series.unique().tolist())
    cols = sorted(secondary_series.unique().tolist())
    cells = []
    worst_gap = -1.0
    worst_idx = 0
    for r in rows:
        for c in cols:
            mask = ((primary_series == r) & (secondary_series == c)).to_numpy()
            if mask.sum() == 0:
                gap = 0.0
            else:
                rate = float(pred_arr[mask].mean())
                gap = abs(reference_rate - rate)
            cells.append({"rowLabel": r, "colLabel": c, "gap": gap, "worst": False})
            if gap > worst_gap:
                worst_gap = gap
                worst_idx = len(cells) - 1
    if cells:
        cells[worst_idx]["worst"] = True
    total = sum(c["gap"] for c in cells)
    return rows, cols, cells, (worst_gap / total if total > 0 else 0.0)


def _pareto_frontier(
    pred_arr: np.ndarray,
    primary_series: pd.Series,
    reference_label: str,
    worst_label: str,
    label_col: Optional[str],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Sweep selection thresholds to draw a Pareto-like frontier.

    Even without a continuous score we can compute the (accuracy, gap) trade-off
    by perturbing the base prediction with random flips at increasing rates.
    """
    rng = np.random.default_rng(7)
    base = pred_arr.astype(int)
    group = primary_series.to_numpy()
    if label_col is not None and label_col in df.columns:
        label_arr, _ = _coerce_binary(df[label_col])
    else:
        label_arr = None

    points: List[Dict[str, Any]] = []
    for i, flip_rate in enumerate(np.linspace(0.0, 0.6, 7)):
        flips = rng.random(size=base.shape) < flip_rate
        perturbed = np.where(flips, 1 - base, base)
        if label_arr is not None and len(label_arr) == len(perturbed):
            acc = float((perturbed == label_arr).mean())
        else:
            # No labels: proxy accuracy = agreement with the base prediction.
            acc = float((perturbed == base).mean())
        a_mask = group == reference_label
        b_mask = group == worst_label
        gap = (
            abs(float(perturbed[a_mask].mean()) - float(perturbed[b_mask].mean()))
            if a_mask.any() and b_mask.any()
            else 0.0
        )
        points.append(
            {
                "label": f"op {i + 1}",
                "accuracy": acc,
                "gap": gap,
                "onFrontier": True,
            }
        )

    # Current operating point = base predictions, no perturbation.
    if label_arr is not None and len(label_arr) == len(base):
        current_acc = float((base == label_arr).mean())
    else:
        current_acc = 1.0
    a_mask = group == reference_label
    b_mask = group == worst_label
    current_gap = (
        abs(float(base[a_mask].mean()) - float(base[b_mask].mean()))
        if a_mask.any() and b_mask.any()
        else 0.0
    )
    points.append(
        {
            "label": "this system",
            "accuracy": current_acc,
            "gap": current_gap,
            "current": True,
            "onFrontier": False,
        }
    )

    # Frontier check: any frontier point with >= current_acc and < current_gap?
    on_frontier = not any(
        p.get("onFrontier")
        and p["accuracy"] + 0.005 >= current_acc
        and p["gap"] + 0.01 < current_gap
        for p in points
    )
    return {"points": points, "currentOnFrontier": on_frontier}


def _bias_findings(
    df: pd.DataFrame,
    primary: str,
    reference_label: str,
    worst_label: str,
    primary_series: pd.Series,
    pred_arr: np.ndarray,
    label_col: Optional[str],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    # Representation bias: deployment share != sample share.
    counts = primary_series.value_counts(normalize=True)
    worst_share = float(counts.get(worst_label, 0.0))
    if worst_share < 0.15:
        findings.append(
            {
                "key": "representation",
                "name": "Representation bias",
                "plain": (
                    f"Group '{worst_label}' makes up only {round(worst_share * 100)}% of "
                    "the dataset. The model has limited evidence to be reliable for them."
                ),
                "evidence": f"Sample share: {round(worst_share * 100)}%.",
                "tone": "critical" if worst_share < 0.05 else "warn",
            }
        )

    # Label bias: base-rate divergence (only if labels exist).
    if label_col is not None:
        label_arr, _ = _coerce_binary(df[label_col])
        if len(label_arr) == len(pred_arr):
            base_rates = pd.Series(label_arr).groupby(primary_series.to_numpy()).mean()
            spread = float(base_rates.max() - base_rates.min()) if not base_rates.empty else 0.0
            if spread > 0.1:
                findings.append(
                    {
                        "key": "label",
                        "name": "Label bias",
                        "plain": (
                            "Historical labels already favoured one group; the model "
                            "inherits that pattern."
                        ),
                        "evidence": f"Base-rate spread between groups: {round(spread * 100)} points.",
                        "tone": "warn",
                    }
                )

    # Aggregation bias: simple proxy -- different selection rates per group.
    selection = pd.Series(pred_arr, index=df.index).groupby(primary_series).mean()
    spread = float(selection.max() - selection.min()) if not selection.empty else 0.0
    if spread > 0.1:
        findings.append(
            {
                "key": "aggregation",
                "name": "Aggregation bias",
                "plain": (
                    "One model is used for groups whose decision patterns differ. A "
                    "single threshold cannot serve them equally."
                ),
                "evidence": f"Selection-rate spread: {round(spread * 100)} points.",
                "tone": "warn",
            }
        )

    if not findings:
        findings.append(
            {
                "key": "no_bias_detected",
                "name": "No dominant bias mechanism",
                "plain": "Pulse could not isolate a single mechanism above the alert threshold.",
                "evidence": "All heuristic checks fell below their alert points.",
                "tone": "info",
            }
        )
    return findings


def _causal_dag(
    df: pd.DataFrame,
    primary: str,
    label_col: Optional[str],
    pred_col: str,
    secondary: Optional[str],
) -> Dict[str, Any]:
    """Build a 5-node DAG: protected, two best proxies, one feature, outcome.

    Proxy detection uses Cramer's V over the contingency between the protected
    attribute and each candidate column. Anything with V >= 0.2 is flagged
    as a proxy path. Confound edges are drawn from protected -> non-proxy
    features (suggesting common-cause structure).
    """
    candidates = [c for c in df.columns if c not in (primary, label_col, pred_col, secondary)]
    proxies: List[Tuple[str, float]] = []
    confounds: List[str] = []
    # Bin the protected attribute before the proxy scan too, otherwise a
    # continuous "age" yields a degenerate crosstab and a meaningless V.
    primary_grouped = _group_series(df, primary)
    for col in candidates:
        v = _cramers_v(primary_grouped, df[col])
        if v is None:
            continue
        if v >= 0.20:
            proxies.append((col, v))
        elif v >= 0.05:
            confounds.append(col)
    proxies.sort(key=lambda t: -t[1])
    proxies = proxies[:2]
    confounds = confounds[:1]

    if not proxies:
        return {"nodes": [], "edges": [], "worstPath": "", "sufficient": False}

    nodes: List[Dict[str, Any]] = [
        {"id": "protected", "label": primary, "kind": "protected", "x": 80, "y": 90},
        {"id": "outcome", "label": pred_col, "kind": "outcome", "x": 500, "y": 110},
    ]
    edges: List[Dict[str, Any]] = []
    y = 50
    for i, (p_name, _v) in enumerate(proxies):
        node_id = f"proxy{i + 1}"
        nodes.append({"id": node_id, "label": p_name, "kind": "proxy", "x": 280, "y": y})
        edges.append({"from": "protected", "to": node_id, "kind": "proxy_path"})
        edges.append({"from": node_id, "to": "outcome", "kind": "proxy_path"})
        y += 90
    for j, c_name in enumerate(confounds):
        node_id = f"feat{j + 1}"
        nodes.append({"id": node_id, "label": c_name, "kind": "feature", "x": 280, "y": y})
        edges.append({"from": node_id, "to": "outcome", "kind": "direct"})
        edges.append({"from": "protected", "to": node_id, "kind": "confound"})
        y += 90

    worst_proxy = proxies[0][0]
    return {
        "nodes": nodes,
        "edges": edges,
        "worstPath": f"{primary} → {worst_proxy} → {pred_col}",
        "sufficient": True,
    }


def _cramers_v(a: pd.Series, b: pd.Series) -> Optional[float]:
    try:
        ct = pd.crosstab(a.astype("string").fillna("missing"), b.astype("string").fillna("missing"))
        if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        chi2 = float(_chi_square(ct.to_numpy()))
        n = ct.to_numpy().sum()
        if n == 0:
            return None
        return float(math.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    except Exception:
        return None


def _chi_square(table: np.ndarray) -> float:
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(chi.sum())


def _legal_framing(jurisdiction: str) -> str:
    """Jurisdiction-aware sentence on how a disparity is judged legally.

    The four-fifths / 80% rule is a US-only construct (EEOC Uniform
    Guidelines, Title VII). The EU/UK use a proportionality + objective-
    justification test with no fixed numeric threshold (Directives
    2000/43/EC & 2000/78/EC; EU AI Act Art. 10; UK Equality Act 2010).
    Pulse must not imply the 80% bar is universal.
    """
    j = (jurisdiction or "").strip().lower()
    if not j:
        return (
            "Note: how this gap is judged depends on jurisdiction. There is no "
            "single global threshold; the statistically significant disparity "
            "shown above is what matters everywhere."
        )
    if any(k in j for k in ("us", "u.s", "united states", "america")):
        return (
            "In the US this is read against the EEOC four-fifths (80%) rule "
            "under Title VII, paired with a statistical-significance test "
            "(which the confidence interval above provides)."
        )
    if any(k in j for k in ("eu", "europe", "european")):
        return (
            "In the EU there is no fixed 80% threshold: indirect discrimination "
            "(Directives 2000/43/EC, 2000/78/EC; EU AI Act Art. 10) is judged "
            "by whether a group is put at a particular disadvantage and whether "
            "the practice is objectively justified and proportionate. The "
            "statistically significant disparity above is the trigger for that "
            "assessment."
        )
    if any(k in j for k in ("uk", "united kingdom", "britain", "england")):
        return (
            "In the UK there is no fixed 80% threshold: the Equality Act 2010 "
            "uses a particular-disadvantage plus proportionality test. The "
            "statistically significant disparity above is the trigger."
        )
    return (
        f"Legal framing in {jurisdiction} varies; the US four-fifths (80%) "
        "rule is not universal. The statistically significant disparity shown "
        "above is the jurisdiction-neutral signal that warrants review."
    )


def _recommended_definition(domain: str) -> Dict[str, Any]:
    d = (domain or "").lower()
    if any(k in d for k in ("lend", "credit", "loan")):
        return {
            "primary": "Equal opportunity",
            "secondaries": ["Demographic parity", "Calibration within groups"],
            "rationale": (
                "Lending decisions affect access to a benefit. The qualified-person "
                "standard (equal opportunity) is the closest match to the legal frame "
                "in most jurisdictions, with demographic parity as the headline "
                "disparity check."
            ),
        }
    if any(k in d for k in ("hir", "recruit")):
        return {
            "primary": "Demographic parity",
            "secondaries": ["Equal opportunity"],
            "rationale": (
                "Hiring screens use a four-fifths style headline test in most "
                "jurisdictions, so demographic parity is the primary reading, with "
                "equal opportunity as the secondary check on qualified applicants."
            ),
        }
    if any(k in d for k in ("health", "clinic", "medical")):
        return {
            "primary": "Calibration within groups",
            "secondaries": ["Equal opportunity", "Predictive parity"],
            "rationale": (
                "In clinical decision support the score must mean the same thing "
                "across groups; calibration is the primary lens, and equal "
                "opportunity protects sensitivity for the at-risk groups."
            ),
        }
    return {
        "primary": "Demographic parity",
        "secondaries": ["Equal opportunity"],
        "rationale": (
            "With no domain-specific override, demographic parity is the safe "
            "headline reading, supported by equal opportunity to keep the "
            "discussion grounded in qualified outcomes."
        ),
    }
