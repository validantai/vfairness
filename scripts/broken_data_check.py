#!/usr/bin/env python3
"""Check 2 of the beta gate: honest on broken data.

WHAT THIS ANSWERS. For every capability the registry marks as measuring
(``pipeline_stage == "evaluation"``), does it refuse when the thing it claims to
measure does not exist in the input, and does it still measure when it does?

It is docs/GRADING.md steps 3 to 5 made executable for the whole set at once:
the healthy control, the documented broken inputs (one group, one outcome, every
score missing, zero rows, every score identical, two rows, a one-row group), and
one more that the denominators make undefined in a single group: a group with no
positive outcomes, the classic way an equal-opportunity gap invents a 0.0.

THE EXPECTED ANSWER IS COUNTED, NOT CHOSEN. Each spec names the denominators its
number needs per group (rows, actual positives, actual negatives, predicted
positives, predicted negatives, varying scores). For each input the harness
counts them, applies the capability's own ``min_group_size``, and from that alone
decides whether a measurement exists. Nobody writes "this one should refuse".

THE VERDICT.
  * measurement exists, a number came back            PASS
  * measurement exists, it refused                    OVER-REFUSES (destroys evidence)
  * no measurement exists, it refused or flagged      PASS
  * no measurement exists, a finite unflagged number  FABRICATES
A refusal is an exception, None, NaN, an empty result, or a flag carried IN THE
RETURNED OBJECT (``measured=False``, ``available=False``, an ``unmeasured_reason``,
a not-assessable list covering everything). A warning does not count: it is
invisible to every caller who suppresses warnings (GRADING.md step 5).

Per-group specs are stricter: the set of groups that carry a finite value must be
exactly the set of groups that can be measured.

Cases whose answer is genuinely debatable are listed per spec in ``skip`` with the
reason and reported as NOT JUDGED, never as passed.

Usage:
    python scripts/broken_data_check.py               # print the table
    python scripts/broken_data_check.py --json OUT    # also write the evidence
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import io
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

REFUSED = object()  # a headline extractor returns this when the result says "not measured"

# ===========================================================================
# Worlds
# ===========================================================================

N = 400


@dataclasses.dataclass
class World:
    name: str
    s: np.ndarray  # group label per row
    y: np.ndarray  # actual outcome 0/1
    p: np.ndarray  # predicted outcome 0/1 (float when it may hold NaN)
    prob: np.ndarray  # predicted probability
    yr: np.ndarray  # regression target
    yr_hat: np.ndarray  # regression prediction
    price: np.ndarray
    strata: np.ndarray

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": self.s,
                "stratum": self.strata,
                "y_true": self.y,
                "y_pred": self.p,
                "score": self.prob,
                "income": self.yr,
            }
        )


def _build(name: str) -> World:
    rng = np.random.default_rng(20261001)
    n = N
    s = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    is_a = (s == "a").astype(float)
    y = (rng.random(n) < 0.3 + 0.1 * is_a).astype(int)
    prob = np.clip(0.35 * y + rng.random(n) * 0.55 + 0.08 * is_a, 0.01, 0.99)
    p = (prob > 0.5).astype(int)
    yr = rng.normal(10, 3, n)
    yr_hat = yr + rng.normal(0.5 * is_a, 1 + is_a, n)
    price = 100 + 10 * is_a + rng.normal(0, 5, n)
    strata = np.array(["s1", "s2"] * (n // 2))

    if name == "healthy":
        pass
    elif name == "single_group":
        s = np.array(["a"] * n)
    elif name == "empty":
        k = slice(0, 0)
        s, y, p, prob, yr, yr_hat, price, strata = (
            s[k],
            y[k],
            p[k],
            prob[k],
            yr[k],
            yr_hat[k],
            price[k],
            strata[k],
        )
    elif name == "all_nan_scores":
        p = np.full(n, np.nan)
        prob = np.full(n, np.nan)
        yr_hat = np.full(n, np.nan)
        price = np.full(n, np.nan)
    elif name == "one_label_only":
        y = np.ones(n, dtype=int)
        p = np.ones(n, dtype=int)
        yr = np.full(n, 10.0)
    elif name == "constant_scores":
        prob = np.full(n, 0.7)
        p = np.ones(n, dtype=int)
        yr_hat = np.full(n, 10.0)
    elif name == "n_equals_2":
        idx = np.array([0, n - 1])
        s, y, p, prob, yr, yr_hat, price, strata = (
            s[idx],
            y[idx],
            p[idx],
            prob[idx],
            yr[idx],
            yr_hat[idx],
            price[idx],
            strata[idx],
        )
    elif name == "one_row_minority":
        s = np.array(["a"] * (n - 1) + ["b"])
    elif name == "group_without_positives":
        y = np.where(s == "b", 0, y)
    else:  # pragma: no cover
        raise ValueError(name)
    return World(name, s, y, p, prob, yr, yr_hat, price, strata)


WORLD_NAMES = (
    "healthy",
    "single_group",
    "empty",
    "all_nan_scores",
    "one_label_only",
    "constant_scores",
    "n_equals_2",
    "one_row_minority",
    "group_without_positives",
)


# ===========================================================================
# What a measurement needs, counted from the data
# ===========================================================================


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _needs_met(w: World, mask: np.ndarray, needs: Sequence[str]) -> bool:
    y = w.y[mask]
    p = np.asarray(w.p, dtype=float)[mask]
    prob = np.asarray(w.prob, dtype=float)[mask]
    for need in needs:
        if need == "pos" and not (y == 1).any():
            return False
        if need == "neg" and not (y == 0).any():
            return False
        if need == "ppos" and not (p == 1).any():
            return False
        if need == "pneg" and not (p == 0).any():
            return False
        if need == "var_prob" and not (len(_finite(prob)) and np.ptp(_finite(prob)) > 0):
            return False
        if need == "var_yr" and not (len(w.yr[mask]) and np.ptp(w.yr[mask]) > 0):
            return False
    return True


@dataclasses.dataclass
class Spec:
    capability: str  # registry name
    label: str  # unique row label
    call: Callable[[World], Any]
    headline: Callable[[Any], Any] = lambda r: r
    needs: Sequence[str] = ()
    consumes: Sequence[str] = ("p",)
    grouped_min: int = 2  # groups that must be measurable; 0 = not a group comparison
    min_group_size: int = 30
    per_group: bool = False  # headline is {group: value}
    skip: Dict[str, str] = dataclasses.field(default_factory=dict)

    def measurable_groups(self, w: World) -> List[str]:
        out = []
        for g in sorted(set(w.s.tolist())):
            mask = w.s == g
            if mask.sum() >= self.min_group_size and _needs_met(w, mask, self.needs):
                out.append(g)
        return out

    def expected(self, w: World) -> Optional[str]:
        """'measured', 'not_measured', or None (not judged)."""
        if w.name in self.skip:
            return None
        if len(w.s) == 0:
            return "not_measured"
        for c in self.consumes:
            if len(_finite(getattr(w, c))) == 0:
                return "not_measured"
        if self.grouped_min == 0:
            if w.name in ("n_equals_2", "one_row_minority"):
                return None  # group-size cases say nothing about an ungrouped number
            return (
                "measured" if _needs_met(w, np.ones(len(w.s), bool), self.needs) else "not_measured"
            )
        return "measured" if len(self.measurable_groups(w)) >= self.grouped_min else "not_measured"


# ===========================================================================
# Reading what came back
# ===========================================================================

_ABSENT_FALSE = ("measured", "available", "is_measured", "assessable", "is_assessable", "valid")
_ABSENT_TRUE = ("could_not_check", "not_assessable_all", "insufficient_data")
_ABSENT_TEXT = ("unmeasured_reason", "not_measured_reason", "refusal_reason")


def _flags_absence(obj: Any) -> bool:
    """True when the returned object itself says it did not measure."""
    places: List[Any] = [obj]
    meta = obj.get("metadata") if isinstance(obj, dict) else getattr(obj, "metadata", None)
    if isinstance(meta, dict):
        places.append(meta)
    for place in places:
        get = (
            place.get if isinstance(place, dict) else (lambda k, d=None, o=place: getattr(o, k, d))
        )
        for k in _ABSENT_FALSE:
            if get(k, None) is False:
                return True
        for k in _ABSENT_TRUE:
            if get(k, None) is True:
                return True
        for k in _ABSENT_TEXT:
            v = get(k, None)
            if isinstance(v, str) and v.strip():
                return True
    return False


def _numbers(v: Any) -> List[float]:
    if v is None or v is REFUSED:
        return []
    if isinstance(v, (bool, np.bool_)):
        return [float(v)]
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    if isinstance(v, dict):
        return [x for vv in v.values() for x in _numbers(vv)]
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [x for vv in list(np.asarray(v, dtype=object).ravel()) for x in _numbers(vv)]
    return []


def _state(result: Any, value: Any) -> str:
    if value is REFUSED or _flags_absence(result):
        return "refused"
    nums = _numbers(value)
    if not nums or not any(math.isfinite(x) for x in nums):
        return "refused"
    return "measured"


# ===========================================================================
# Running and judging
# ===========================================================================


def _run(spec: Spec, w: World) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                result = spec.call(w)
                value = spec.headline(result)
        except Exception as exc:  # noqa: BLE001 - an exception is a refusal, recorded
            return {"state": "refused", "detail": f"raised {type(exc).__name__}: {str(exc)[:140]}"}
    if spec.per_group:
        have = (
            sorted(
                str(g)
                for g, v in (value or {}).items()
                if any(math.isfinite(x) for x in _numbers(v)) and not _flags_absence(v)
            )
            if isinstance(value, dict)
            else []
        )
        return {"state": "per_group", "groups": have, "detail": f"groups with a value: {have}"}
    st = _state(result, value)
    shown = repr(value)[:140] if value is not REFUSED else "flagged not measured"
    return {"state": st, "detail": shown}


def judge(spec: Spec, w: World) -> Dict[str, Any]:
    exp = spec.expected(w)
    obs = _run(spec, w)
    row = {"world": w.name, "expected": exp, **obs}
    if exp is None:
        row["verdict"] = "NOT JUDGED"
        row["reason"] = spec.skip.get(w.name, "group-size case, ungrouped number")
        return row
    if obs["state"] == "per_group":
        want = (
            sorted(spec.measurable_groups(w)) if exp == "measured" or spec.grouped_min <= 1 else []
        )
        if len(w.s) == 0 or any(len(_finite(getattr(w, c))) == 0 for c in spec.consumes):
            want = []
        extra = sorted(set(obs["groups"]) - set(want))
        missing = sorted(set(want) - set(obs["groups"]))
        row["verdict"] = "FABRICATES" if extra else ("OVER-REFUSES" if missing else "PASS")
        row["detail"] += f" | measurable: {want}"
        return row
    if exp == "measured":
        row["verdict"] = "PASS" if obs["state"] == "measured" else "OVER-REFUSES"
    else:
        row["verdict"] = "PASS" if obs["state"] == "refused" else "FABRICATES"
    return row


# ===========================================================================
# Specs: capabilities that take tabular input
# ===========================================================================


def _m(path: str):
    return importlib.import_module("vfairness." + path)


def _dp(yp, sa):
    return _m("evaluation.vfairness_metrics.classification").demographic_parity_difference(
        yp, yp, sa
    )


def _point(r):
    return r.point_estimate


def _overall(r):
    return r.overall_value


SEL = ()
TPR = ("pos",)
FPR = ("neg",)
ODDS = ("pos", "neg")
PPV = ("ppos",)
NPV = ("pneg",)

_CONST_AUC = {
    "constant_scores": "AUROC of a constant scorer is 0.5 by the tie rule; refusing it is defensible too"
}


def tabular_specs() -> List[Spec]:
    C = _m("evaluation.vfairness_metrics.classification")
    Rg = _m("evaluation.vfairness_metrics.regression")
    M = _m("evaluation.vfairness_metrics")
    P = _m("post_processing.calibration")
    specs: List[Spec] = []

    def cls(name, needs, ci=False, grouped_min=2, skip=None):
        fn = getattr(M if ci else C, name)
        call = (
            (lambda w, f=fn: f(w.y, w.p, w.s, n_bootstrap=200, random_state=0))
            if ci
            else (lambda w, f=fn: f(w.y, w.p, w.s))
        )
        specs.append(
            Spec(
                name,
                name,
                call,
                _point if ci else (lambda r: r),
                needs=needs,
                grouped_min=grouped_min,
                skip=skip or {},
            )
        )

    for name, needs in [
        ("demographic_parity_difference", SEL),
        ("demographic_parity_ratio", SEL),
        ("disparate_impact_ratio", SEL),
        ("equal_opportunity_difference", TPR),
        ("equalized_odds_difference", ODDS),
        ("fpr_parity_difference", FPR),
        ("fnr_parity_difference", TPR),
        ("accuracy_parity_difference", SEL),
        ("predictive_parity_difference", PPV),
        ("negative_predictive_value_difference", NPV),
    ]:
        cls(name, needs)
    # A minimum over groups: defined for a single measurable group.
    cls(
        "worst_group_accuracy",
        SEL,
        grouped_min=1,
        skip={
            "one_row_minority": "a group dropped as too small may be the worst one; refusing is defensible"
        },
    )
    for name, needs in [
        ("demographic_parity_difference_with_ci", SEL),
        ("disparate_impact_ratio_with_ci", SEL),
        ("equal_opportunity_difference_with_ci", TPR),
        ("equalized_odds_difference_with_ci", ODDS),
        ("fpr_parity_difference_with_ci", FPR),
        ("predictive_parity_difference_with_ci", PPV),
        ("negative_predictive_value_difference_with_ci", NPV),
    ]:
        cls(name, needs, ci=True)

    for meth, needs in [
        ("demographic_parity_difference", SEL),
        ("equal_opportunity_difference", TPR),
        ("equalized_odds_difference", ODDS),
    ]:
        specs.append(
            Spec(
                "FairnessAnalyzer",
                f"FairnessAnalyzer.{meth}",
                lambda w, m=meth: getattr(M.FairnessAnalyzer(w.y, w.p, w.s), m)(),
                lambda r: getattr(r, "value", r),
                needs=needs,
            )
        )

    specs.append(
        Spec(
            "auroc_parity",
            "auroc_parity",
            lambda w: C.auroc_parity(w.y, w.prob, w.s),
            needs=ODDS,
            consumes=("prob",),
            skip=_CONST_AUC,
        )
    )
    specs.append(
        Spec(
            "net_benefit_parity",
            "net_benefit_parity",
            lambda w: C.net_benefit_parity(w.y, w.prob, w.s),
            consumes=("prob",),
        )
    )
    specs.append(
        Spec(
            "pricing_disparity",
            "pricing_disparity",
            lambda w: Rg.pricing_disparity(w.price, w.s),
            consumes=("price",),
        )
    )
    specs.append(
        Spec(
            "pricing_disparity_with_ci",
            "pricing_disparity_with_ci",
            lambda w: M.pricing_disparity_with_ci(w.price, w.s, n_bootstrap=200, random_state=0),
            _point,
            consumes=("price",),
        )
    )
    specs.append(
        Spec(
            "conditional_demographic_disparity",
            "conditional_demographic_disparity",
            lambda w: C.conditional_demographic_disparity(w.p, w.s, w.strata),
        )
    )
    specs.append(
        Spec(
            "conditional_demographic_disparity_with_ci",
            "conditional_demographic_disparity_with_ci",
            lambda w: M.conditional_demographic_disparity_with_ci(
                w.p, w.s, w.strata, n_bootstrap=200, random_state=0
            ),
            _point,
        )
    )
    specs.append(
        Spec(
            "conditional_adverse_impact",
            "conditional_adverse_impact",
            lambda w: C.conditional_adverse_impact(w.p, w.s, pd.DataFrame({"x": w.yr})),
        )
    )

    for name in ("mae_parity_difference", "rmse_parity_difference", "mean_prediction_difference"):
        specs.append(
            Spec(
                name,
                name,
                lambda w, f=getattr(Rg, name): f(w.yr, w.yr_hat, w.s),
                consumes=("yr_hat",),
            )
        )
    specs.append(
        Spec(
            "r2_parity_difference",
            "r2_parity_difference",
            lambda w: Rg.r2_parity_difference(w.yr, w.yr_hat, w.s),
            needs=("var_yr",),
            consumes=("yr_hat",),
        )
    )
    specs.append(
        Spec(
            "residual_bias",
            "residual_bias",
            lambda w: Rg.residual_bias(w.yr, w.yr_hat, w.s),
            consumes=("yr_hat",),
            grouped_min=1,
            per_group=True,
        )
    )

    specs.append(
        Spec(
            "compute_effect_sizes",
            "compute_effect_sizes",
            lambda w: M.compute_effect_sizes(w.y, w.p, w.s),
            lambda r: [v.get("cohens_h_positive_rate") for v in r.values()],
        )
    )
    specs.append(
        Spec(
            "comprehensive_fairness_test",
            "comprehensive_fairness_test",
            lambda w: M.comprehensive_fairness_test(
                w.y, w.p, w.s, n_permutations=200, random_state=0
            ),
            lambda r: r.get("p_values"),
            min_group_size=1,
            skip={
                "n_equals_2": "a permutation test over two rows has no power; refusing is defensible"
            },
            # Tests several metrics and lists each undefined one in not_assessable;
            # the selection-rate test needs only rows, so that is the floor.
            needs=SEL,
        )
    )
    specs.append(
        Spec(
            "classification_fairness_report",
            "classification_fairness_report",
            lambda w: M.classification_fairness_report(w.y, w.p, w.s),
            lambda r: {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in (r.get("metrics") or {}).items()
                if not (isinstance(v, dict) and _flags_absence(v))
            },
            needs=SEL,
        )
    )

    # Calibration, overall (no groups): needs only rows, except where noted.
    for name, needs, skip in [
        ("expected_calibration_error", (), {}),
        ("maximum_calibration_error", (), {}),
        ("brier_score", (), {}),
        ("calibration_slope", ("pos", "neg", "var_prob"), {}),
        ("calibration_in_the_large", ("pos", "neg"), {}),
    ]:
        specs.append(
            Spec(
                name,
                name,
                lambda w, f=getattr(P, name): f(w.y, w.prob),
                _overall,
                needs=needs,
                consumes=("prob",),
                grouped_min=0,
                skip=skip,
            )
        )
        specs.append(
            Spec(
                name,
                f"{name}[per group]",
                lambda w, f=getattr(P, name): f(w.y, w.prob, w.s),
                lambda r: r.group_values,
                needs=needs,
                consumes=("prob",),
                grouped_min=1,
                per_group=True,
                skip=skip,
            )
        )
    _ici_skip = {
        "constant_scores": "a smoother over one x value is undefined or the mean; both defensible"
    }
    specs.append(
        Spec(
            "integrated_calibration_index",
            "integrated_calibration_index",
            lambda w: P.integrated_calibration_index(w.y, w.prob),
            lambda r: r.overall_ici,
            consumes=("prob",),
            grouped_min=0,
            skip=_ici_skip,
        )
    )
    specs.append(
        Spec(
            "integrated_calibration_index",
            "integrated_calibration_index[per group]",
            lambda w: P.integrated_calibration_index(w.y, w.prob, w.s),
            lambda r: r.group_ici,
            consumes=("prob",),
            grouped_min=1,
            per_group=True,
            skip=_ici_skip,
        )
    )
    specs.append(
        Spec(
            "calibration_curve",
            "calibration_curve",
            lambda w: P.calibration_curve(w.y, w.prob),
            lambda r: r.prob_true,
            consumes=("prob",),
            grouped_min=0,
        )
    )
    specs.append(
        Spec(
            "brier_score_decomposition",
            "brier_score_decomposition",
            lambda w: P.brier_score_decomposition(w.y, w.prob),
            lambda r: r.brier_score,
            consumes=("prob",),
            grouped_min=0,
        )
    )
    specs.append(
        Spec(
            "calibration_disparity",
            "calibration_disparity",
            lambda w: P.calibration_disparity(w.y, w.prob, w.s),
            lambda r: r.ece_disparity,
            consumes=("prob",),
        )
    )
    specs.append(
        Spec(
            "group_calibration_metrics",
            "group_calibration_metrics",
            lambda w: P.group_calibration_metrics(w.y, w.prob, w.s),
            lambda r: r["ece"].group_values,
            consumes=("prob",),
            grouped_min=1,
            per_group=True,
        )
    )
    specs.append(
        Spec(
            "multicalibration",
            "multicalibration",
            lambda w: P.multicalibration(w.y, w.prob, w.s),
            lambda r: r.weighted_mean,
            consumes=("prob",),
            grouped_min=1,
        )
    )
    specs.append(
        Spec(
            "multicalibration_with_ci",
            "multicalibration_with_ci",
            lambda w: P.multicalibration_with_ci(w.y, w.prob, w.s, n_bootstrap=100, random_state=0),
            _point,
            consumes=("prob",),
            grouped_min=1,
        )
    )
    specs.append(
        Spec(
            "integrated_calibration_index_with_ci",
            "integrated_calibration_index_with_ci",
            lambda w: P.integrated_calibration_index_with_ci(
                w.y, w.prob, w.s, n_bootstrap=100, random_state=0
            ),
            _point,
            consumes=("prob",),
            grouped_min=2,  # a between-group gap
            skip={
                "constant_scores": "a smoother over one x value is undefined or the mean; both defensible"
            },
        )
    )

    # Procedures wrapped around a metric: they inherit its denominators.
    specs.append(
        Spec(
            "permutation_test",
            "permutation_test",
            lambda w: M.permutation_test(w.p, w.s, _dp, n_permutations=200, random_state=0),
            lambda r: r.p_value,
        )
    )
    specs.append(
        Spec(
            "sensitivity_analysis",
            "sensitivity_analysis",
            lambda w: M.sensitivity_analysis(w.p, w.s, _dp, n_iterations=20, random_state=0),
            lambda r: r.original_metric,
        )
    )
    specs.append(
        Spec(
            "stress_test_fairness",
            "stress_test_fairness",
            lambda w: M.stress_test_fairness(w.p, w.s, _dp, n_iterations=20, random_state=0),
            lambda r: r.get("original_metric"),
        )
    )
    specs.append(
        Spec(
            "subgroup_robustness_audit",
            "subgroup_robustness_audit",
            lambda w: M.subgroup_robustness_audit(w.p, {"g": w.s}, y_true=w.y),
            # The verdict is the headline; None is its documented could-not-check.
            lambda r: REFUSED if r.gerrymandering_detected is None else r.gerrymandering_detected,
        )
    )

    # Statistics with no groups.
    specs.append(
        Spec(
            "bootstrap_ci",
            "bootstrap_ci",
            lambda w: M.bootstrap_ci(
                np.asarray(w.yr_hat, float), np.mean, n_bootstrap=500, random_state=0
            ),
            lambda r: [r.lower_bound, r.upper_bound],
            consumes=("yr_hat",),
            grouped_min=0,
        )
    )
    specs.append(
        Spec(
            "bayesian_proportion_ci",
            "bayesian_proportion_ci",
            lambda w: M.bayesian_proportion_ci(
                int(np.nansum(np.asarray(w.p, float))),
                int(np.isfinite(np.asarray(w.p, float)).sum()),
            ),
            _point,
            grouped_min=0,
        )
    )
    for name in ("bonferroni_correction", "benjamini_hochberg_correction"):
        specs.append(
            Spec(
                name,
                name,
                lambda w, f=getattr(M, name): f(np.asarray(w.prob, float)),
                lambda r: (
                    np.asarray(r.adjusted_p_values)[np.asarray(r.tested_mask, bool)]
                    if np.asarray(r.tested_mask, bool).any()
                    else REFUSED
                ),
                consumes=("prob",),
                grouped_min=0,
            )
        )
    specs.append(
        Spec(
            "scan_fairness_violations",
            "scan_fairness_violations",
            lambda w: M.scan_fairness_violations(
                w.df, w.p, y_true=w.y, protected_columns=["group"], auto_detect=False
            ),
            # The violation count is a claim only if something was measured; the
            # returned list carries measured_disparities and not_assessable.
            lambda r: len(r) if getattr(r, "measured_disparities", None) else REFUSED,
        )
    )
    return specs


# ===========================================================================
# The scope, and the report
# ===========================================================================


# Registered as evaluation, and run to show they produce no measurement, so check
# 2 cannot apply to them (GRADING.md "Not a measurement": only after running it and
# stating what it returns). tests/test_honest_on_broken_data.py re-runs each one
# and fails if a number ever comes back, so this list cannot hide a measurement.
NOT_A_MEASUREMENT = {
    "route_explainer": (
        "takes no data, only the model type and the goal; returns a RouteDecision "
        "naming which explainer to run (e.g. primary='shap.TreeExplainer'), with no "
        "number and no verdict about any data (run 2026-10-01 for tree, linear, "
        "neural and black_box)"
    ),
}


def evaluation_capabilities() -> List[str]:
    from vfairness._registry import CAPABILITY_REGISTRY

    return sorted(
        {e["name"] for e in CAPABILITY_REGISTRY.values() if e["pipeline_stage"] == "evaluation"}
    )


def all_specs() -> List[Spec]:
    specs = tabular_specs()
    spec_dir = ROOT / "scripts" / "broken_data_specs"
    if spec_dir.is_dir():
        sys.path.insert(0, str(ROOT / "scripts"))
        for f in sorted(spec_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            mod = importlib.import_module(f"broken_data_specs.{f.stem}")
            specs.extend(mod.specs(Spec, REFUSED, _flags_absence))
    return specs


def run(specs: Optional[List[Spec]] = None) -> Dict[str, Any]:
    logging.disable(logging.CRITICAL)
    specs = specs if specs is not None else all_specs()
    worlds = {n: _build(n) for n in WORLD_NAMES}
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        rows[spec.label] = [judge(spec, worlds[n]) for n in WORLD_NAMES]
    caps: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        c = caps.setdefault(spec.capability, {"specs": [], "verdict": None})
        c["specs"].append(spec.label)
    for cap, c in caps.items():
        verdicts = [r["verdict"] for lbl in c["specs"] for r in rows[lbl]]
        if "FABRICATES" in verdicts:
            c["verdict"] = "FABRICATES"
        elif "OVER-REFUSES" in verdicts:
            c["verdict"] = "OVER-REFUSES"
        else:
            c["verdict"] = "PASS"
    scope = evaluation_capabilities()
    for name in scope:
        if name in NOT_A_MEASUREMENT and name not in caps:
            caps[name] = {
                "specs": [],
                "verdict": "NOT A MEASUREMENT",
                "reason": NOT_A_MEASUREMENT[name],
            }
        caps.setdefault(name, {"specs": [], "verdict": "NOT COVERED"})
    summary = {
        v: sum(1 for n in scope if caps[n]["verdict"] == v)
        for v in ("PASS", "FABRICATES", "OVER-REFUSES", "NOT A MEASUREMENT", "NOT COVERED")
    }
    return {
        "scope": len(scope),
        "summary": summary,
        "capabilities": {n: caps[n] for n in scope},
        "rows": rows,
        "worlds": list(WORLD_NAMES),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--only", help="comma-separated capability names")
    ap.add_argument("--failures", action="store_true", help="print only failing rows")
    a = ap.parse_args(argv)
    specs = all_specs()
    if a.only:
        keep = set(a.only.split(","))
        specs = [s for s in specs if s.capability in keep]
    res = run(specs)
    for lbl, rws in res["rows"].items():
        for r in rws:
            if a.failures and r["verdict"] in ("PASS", "NOT JUDGED"):
                continue
            print(f"{r['verdict']:12s} {lbl:48s} {r['world']:24s} {r['detail'][:110]}")
    print(json.dumps(res["summary"]), f"of {res['scope']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
