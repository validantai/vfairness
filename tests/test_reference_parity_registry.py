"""Check 1 of the beta gate: correct on normal data, against an outside reference.

WHAT THIS ANSWERS. Of the capabilities the registry marks as measuring
(``pipeline_stage == "evaluation"``, 110 on 2026-10-01), which ones compute a
number that an established, independently written library also computes, and do
the two agree on clean data? Each row below names the registry capability, the
vfairness call, the reference call and the tolerance.

WHY A SECOND FILE. ``test_cross_library_parity.py`` pins four metrics against
fairlearn and AIF360 with the min_group_size floor lowered. This file covers
every evaluation capability that HAS a reference, called with its shipped
defaults, so the parity a user gets is the parity tested.

WHAT IT CANNOT SAY. The other evaluation capabilities have no installed library
that computes the same number (LLM, agent, multi-agent, ranking and explainer
tools, and some calibration and intersectional measures). They are not in this
file, and ``test_every_evaluation_capability_is_accounted_for`` makes that list
explicit instead of letting it be silently shorter. Their correctness on normal
data is NOT established here. Check 2 (honest on broken data) is the one that
applies to them.

EVERY ROW MUST BE ABLE TO DISAGREE. Agreement within a tolerance proves nothing if
the tolerance would also accept a wrong answer, so each row also asserts that
the vfairness value moved by three tolerances (at least 0.003) is REFUSED by the
same comparison. A row that cannot fail is a row that is not checking anything.

Monte Carlo rows (permutation p-value, bootstrap interval) compare two random
procedures, so their tolerance is 0.02 at 20,000 resamples, about four standard
errors of the Monte Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest

fl = pytest.importorskip("fairlearn.metrics")
sm = pytest.importorskip("statsmodels.api")
from scipy import stats  # noqa: E402
from sklearn.calibration import calibration_curve as sk_calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_effectsize  # noqa: E402

import vfairness.evaluation.vfairness_metrics as M  # noqa: E402
import vfairness.evaluation.vfairness_metrics.classification as C  # noqa: E402
import vfairness.evaluation.vfairness_metrics.regression as R  # noqa: E402
import vfairness.post_processing.calibration as P  # noqa: E402
from vfairness._registry import CAPABILITY_REGISTRY  # noqa: E402
from vfairness.evaluation.vfairness_metrics import FairnessAnalyzer  # noqa: E402


def _fixture(seed, groups, sizes):
    """Clean data with a real, unequal disparity between every pair of groups."""
    rng = np.random.default_rng(seed)
    s = np.concatenate([[g] * n for g, n in zip(groups, sizes)])
    idx = np.array([groups.index(g) for g in s])
    y = (rng.random(len(s)) < 0.25 + 0.15 * idx).astype(int)
    prob = np.clip(0.5 * y + 0.1 * idx + rng.normal(0.2, 0.2, len(s)), 0.01, 0.99)
    yr = rng.normal(10, 3, len(s)) + idx
    return {
        "s": s,
        "y": y,
        "p": (prob > 0.5).astype(int),
        "prob": prob,
        "yr": yr,
        "yr_hat": yr + rng.normal(0, 1 + 0.5 * idx, len(s)),
    }


# Two groups, and three groups of unequal size: max-minus-min spreads and
# pairwise definitions coincide for two groups and part ways for three.
FIXTURES = {
    "two_groups_600_400": _fixture(1, ["A", "B"], [600, 400]),
    "three_groups_700_250_120": _fixture(2, ["A", "B", "C"], [700, 250, 120]),
}


def _mf(fn, y, p, s):
    return fl.MetricFrame(metrics=fn, y_true=y, y_pred=p, sensitive_features=s)


def _npv(y, p):
    tn = ((y == 0) & (p == 0)).sum()
    fn_ = ((y == 1) & (p == 0)).sum()
    return tn / (tn + fn_)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _logit(p):
    return np.log(p / (1 - p))


def _first_pair(d):
    return next(iter(M.compute_effect_sizes(d["y"], d["p"], d["s"]).values()))


def _effect_reference(d):
    g = list(dict.fromkeys(d["s"]))[:2]
    r = [d["p"][d["s"] == x].mean() for x in g]
    n = [(d["s"] == x).sum() for x in g]
    e = [d["p"][d["s"] == x].sum() for x in g]
    odds = (e[0] / (n[0] - e[0])) / (e[1] / (n[1] - e[1]))
    return [proportion_effectsize(r[0], r[1]), r[0] / r[1], odds]


_PV = np.array([0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5, 0.049])


def _fl(name):
    return lambda d: getattr(fl, name)(d["y"], d["p"], sensitive_features=d["s"])


# (registry capability name, vfairness value, reference value, reference, tolerance)
ROWS = [
    (
        "demographic_parity_difference",
        lambda d: C.demographic_parity_difference(d["y"], d["p"], d["s"]),
        _fl("demographic_parity_difference"),
        "fairlearn",
        1e-9,
    ),
    (
        "demographic_parity_ratio",
        lambda d: C.demographic_parity_ratio(d["y"], d["p"], d["s"]),
        _fl("demographic_parity_ratio"),
        "fairlearn",
        1e-9,
    ),
    (
        "disparate_impact_ratio",
        lambda d: C.disparate_impact_ratio(d["y"], d["p"], d["s"]),
        _fl("demographic_parity_ratio"),
        "fairlearn min/max selection rate",
        1e-9,
    ),
    (
        "equal_opportunity_difference",
        lambda d: C.equal_opportunity_difference(d["y"], d["p"], d["s"]),
        _fl("equal_opportunity_difference"),
        "fairlearn",
        1e-9,
    ),
    (
        "equalized_odds_difference",
        lambda d: C.equalized_odds_difference(d["y"], d["p"], d["s"]),
        _fl("equalized_odds_difference"),
        "fairlearn",
        1e-9,
    ),
    (
        "fpr_parity_difference",
        lambda d: C.fpr_parity_difference(d["y"], d["p"], d["s"]),
        _fl("false_positive_rate_difference"),
        "fairlearn",
        1e-9,
    ),
    (
        "fnr_parity_difference",
        lambda d: C.fnr_parity_difference(d["y"], d["p"], d["s"]),
        _fl("false_negative_rate_difference"),
        "fairlearn",
        1e-9,
    ),
    (
        "accuracy_parity_difference",
        lambda d: C.accuracy_parity_difference(d["y"], d["p"], d["s"]),
        lambda d: _mf(accuracy_score, d["y"], d["p"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "predictive_parity_difference",
        lambda d: C.predictive_parity_difference(d["y"], d["p"], d["s"]),
        lambda d: _mf(precision_score, d["y"], d["p"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "negative_predictive_value_difference",
        lambda d: C.negative_predictive_value_difference(d["y"], d["p"], d["s"]),
        lambda d: _mf(_npv, d["y"], d["p"], d["s"]).difference(),
        "fairlearn MetricFrame, NPV from counts",
        1e-9,
    ),
    (
        "worst_group_accuracy",
        lambda d: C.worst_group_accuracy(d["y"], d["p"], d["s"]),
        lambda d: _mf(accuracy_score, d["y"], d["p"], d["s"]).group_min(),
        "fairlearn MetricFrame group_min",
        1e-9,
    ),
    (
        "auroc_parity",
        lambda d: C.auroc_parity(d["y"], d["prob"], d["s"]),
        lambda d: _mf(roc_auc_score, d["y"], d["prob"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "mae_parity_difference",
        lambda d: R.mae_parity_difference(d["yr"], d["yr_hat"], d["s"]),
        lambda d: _mf(mean_absolute_error, d["yr"], d["yr_hat"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "rmse_parity_difference",
        lambda d: R.rmse_parity_difference(d["yr"], d["yr_hat"], d["s"]),
        lambda d: _mf(_rmse, d["yr"], d["yr_hat"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "r2_parity_difference",
        lambda d: R.r2_parity_difference(d["yr"], d["yr_hat"], d["s"]),
        lambda d: _mf(r2_score, d["yr"], d["yr_hat"], d["s"]).difference(),
        "fairlearn MetricFrame + sklearn",
        1e-9,
    ),
    (
        "mean_prediction_difference",
        lambda d: R.mean_prediction_difference(d["yr"], d["yr_hat"], d["s"]),
        lambda d: _mf(fl.mean_prediction, d["yr"], d["yr_hat"], d["s"]).difference(),
        "fairlearn mean_prediction",
        1e-9,
    ),
    (
        "demographic_parity_difference_with_ci",
        lambda d: (
            M.demographic_parity_difference_with_ci(
                d["y"], d["p"], d["s"], n_bootstrap=200
            ).point_estimate
        ),
        _fl("demographic_parity_difference"),
        "fairlearn, point estimate",
        1e-9,
    ),
    (
        "disparate_impact_ratio_with_ci",
        lambda d: (
            M.disparate_impact_ratio_with_ci(d["y"], d["p"], d["s"], n_bootstrap=200).point_estimate
        ),
        _fl("demographic_parity_ratio"),
        "fairlearn, point estimate",
        1e-9,
    ),
    (
        "equal_opportunity_difference_with_ci",
        lambda d: (
            M.equal_opportunity_difference_with_ci(
                d["y"], d["p"], d["s"], n_bootstrap=200
            ).point_estimate
        ),
        _fl("equal_opportunity_difference"),
        "fairlearn, point estimate",
        1e-9,
    ),
    (
        "equalized_odds_difference_with_ci",
        lambda d: (
            M.equalized_odds_difference_with_ci(
                d["y"], d["p"], d["s"], n_bootstrap=200
            ).point_estimate
        ),
        _fl("equalized_odds_difference"),
        "fairlearn, point estimate",
        1e-9,
    ),
    (
        "fpr_parity_difference_with_ci",
        lambda d: (
            M.fpr_parity_difference_with_ci(d["y"], d["p"], d["s"], n_bootstrap=200).point_estimate
        ),
        _fl("false_positive_rate_difference"),
        "fairlearn, point estimate",
        1e-9,
    ),
    (
        "predictive_parity_difference_with_ci",
        lambda d: (
            M.predictive_parity_difference_with_ci(
                d["y"], d["p"], d["s"], n_bootstrap=200
            ).point_estimate
        ),
        lambda d: _mf(precision_score, d["y"], d["p"], d["s"]).difference(),
        "fairlearn MetricFrame, point estimate",
        1e-9,
    ),
    (
        "negative_predictive_value_difference_with_ci",
        lambda d: (
            M.negative_predictive_value_difference_with_ci(
                d["y"], d["p"], d["s"], n_bootstrap=200
            ).point_estimate
        ),
        lambda d: _mf(_npv, d["y"], d["p"], d["s"]).difference(),
        "fairlearn MetricFrame, point estimate",
        1e-9,
    ),
    # A class: its three headline methods, each against fairlearn.
    (
        "FairnessAnalyzer",
        lambda d: [
            getattr(v, "value", v)
            for v in (
                FairnessAnalyzer(d["y"], d["p"], d["s"]).demographic_parity_difference(),
                FairnessAnalyzer(d["y"], d["p"], d["s"]).equal_opportunity_difference(),
                FairnessAnalyzer(d["y"], d["p"], d["s"]).equalized_odds_difference(),
            )
        ],
        lambda d: [
            _fl(n)(d)
            for n in (
                "demographic_parity_difference",
                "equal_opportunity_difference",
                "equalized_odds_difference",
            )
        ],
        "fairlearn, three methods",
        1e-9,
    ),
    (
        "brier_score",
        lambda d: P.brier_score(d["y"], d["prob"]).overall_value,
        lambda d: brier_score_loss(d["y"], d["prob"]),
        "sklearn brier_score_loss",
        1e-12,
    ),
    (
        "calibration_curve",
        lambda d: (lambda c: np.concatenate([c.prob_true, c.prob_pred]))(
            P.calibration_curve(d["y"], d["prob"])
        ),
        lambda d: np.concatenate(sk_calibration_curve(d["y"], d["prob"], n_bins=10)),
        "sklearn calibration_curve",
        1e-12,
    ),
    (
        "calibration_slope",
        lambda d: P.calibration_slope(d["y"], d["prob"]).overall_value,
        lambda d: sm.Logit(d["y"], sm.add_constant(_logit(d["prob"]))).fit(disp=0).params[1],
        "statsmodels logistic recalibration slope",
        1e-5,
    ),
    (
        "calibration_in_the_large",
        lambda d: P.calibration_in_the_large(d["y"], d["prob"]).metadata["signed"],
        lambda d: (
            sm.GLM(
                d["y"],
                np.ones(len(d["y"])),
                family=sm.families.Binomial(),
                offset=_logit(d["prob"]),
            )
            .fit()
            .params[0]
        ),
        "statsmodels intercept with logit offset",
        1e-5,
    ),
    (
        "bayesian_proportion_ci",
        lambda d: (lambda r: [r.point_estimate, r.lower_bound, r.upper_bound])(
            M.bayesian_proportion_ci(int(d["p"].sum()), len(d["p"]))
        ),
        lambda d: (
            lambda a, b: [a / (a + b), stats.beta.ppf(0.025, a, b), stats.beta.ppf(0.975, a, b)]
        )(d["p"].sum() + 1, len(d["p"]) - d["p"].sum() + 1),
        "scipy beta posterior, uniform prior",
        1e-9,
    ),
    (
        "bonferroni_correction",
        lambda d: (lambda r: np.r_[r.adjusted_p_values, r.rejection_mask])(
            M.bonferroni_correction(_PV)
        ),
        lambda d: (lambda r: np.r_[r[1], r[0]])(multipletests(_PV, 0.05, "bonferroni")),
        "statsmodels multipletests",
        1e-12,
    ),
    (
        "benjamini_hochberg_correction",
        lambda d: (lambda r: np.r_[r.adjusted_p_values, r.rejection_mask])(
            M.benjamini_hochberg_correction(_PV)
        ),
        lambda d: (lambda r: np.r_[r[1], r[0]])(multipletests(_PV, 0.05, "fdr_bh")),
        "statsmodels multipletests",
        1e-12,
    ),
    (
        "compute_effect_sizes",
        lambda d: (lambda e: [e["cohens_h_positive_rate"], e["risk_ratio"][0], e["odds_ratio"][0]])(
            _first_pair(d)
        ),
        _effect_reference,
        "statsmodels Cohen's h, risk and odds ratio from counts",
        1e-9,
    ),
    (
        "permutation_test",
        lambda d: (
            M.permutation_test(
                d["p"][np.isin(d["s"], ["A", "B"])],
                d["s"][np.isin(d["s"], ["A", "B"])],
                lambda yp, sa: abs(yp[sa == "A"].mean() - yp[sa == "B"].mean()),
                n_permutations=20000,
                random_state=0,
            ).p_value
        ),
        lambda d: (
            stats.permutation_test(
                (d["p"][d["s"] == "A"], d["p"][d["s"] == "B"]),
                lambda a, b: abs(a.mean() - b.mean()),
                n_resamples=20000,
                alternative="greater",
                random_state=0,
            ).pvalue
        ),
        "scipy permutation_test, Monte Carlo",
        0.02,
    ),
    (
        "bootstrap_ci",
        lambda d: (lambda r: [r.lower_bound, r.upper_bound])(
            M.bootstrap_ci(d["yr"], np.mean, n_bootstrap=20000, random_state=0)
        ),
        lambda d: (lambda r: [r.confidence_interval.low, r.confidence_interval.high])(
            stats.bootstrap(
                (d["yr"],), np.mean, n_resamples=20000, method="percentile", random_state=0
            )
        ),
        "scipy bootstrap percentile, Monte Carlo",
        0.02,
    ),
]


def _agree(a, b, tol):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (
        a.shape == b.shape
        and bool(np.all(np.isfinite(a)))
        and bool(np.allclose(a, b, atol=tol, rtol=0))
    )


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("name,ours,ref,source,tol", ROWS, ids=[r[0] for r in ROWS])
def test_matches_the_reference_on_clean_data(name, ours, ref, source, tol, fixture):
    d = FIXTURES[fixture]
    mine = ours(d)
    theirs = ref(d)
    assert _agree(mine, theirs, tol), f"{name} vs {source} on {fixture}: {mine!r} != {theirs!r}"
    # The same comparison must REFUSE a wrong answer, or this row checks nothing.
    nudged = np.asarray(mine, dtype=float) + max(3 * tol, 1e-3)
    assert not _agree(nudged, theirs, tol), f"{name}: tolerance {tol} would accept a wrong value"


def test_every_row_names_a_registered_evaluation_capability():
    evaluation = {
        e["name"] for e in CAPABILITY_REGISTRY.values() if e["pipeline_stage"] == "evaluation"
    }
    named = [r[0] for r in ROWS]
    assert len(named) == len(set(named)), "a capability is listed twice"
    assert set(named) <= evaluation, sorted(set(named) - evaluation)


# The evaluation capabilities with NO installed reference, listed by name so the
# split is a recorded decision. A capability added to the registry fails the test
# below until it is put on one side or the other.
NO_REFERENCE = {
    "ActionBiasAnalyzer",
    "AdversarialCollusionDetector",
    "AnchorsExplainer",
    "BenchmarkRunner",
    "CausalFairnessGraph",
    "CoTFaithfulnessAnalyzer",
    "CompositionalityAnalyzer",
    "CorrespondenceTester",
    "CounterfactualTester",
    "DecodingTrustRunner",
    "DelegationRoutingAuditor",
    "DiceCounterfactualExplainer",
    "EmbeddingBiasDetector",
    "EmergentBiasDetector",
    "FairExplAIner",
    "FairnessExplainer",
    "FeatureAttributionExplainer",
    "GroupthinkDetector",
    "IntegratedGradientsExplainer",
    "IntersectionalAnalyzer",
    "KernelShapExplainer",
    "LimeExplainer",
    "LinearShapExplainer",
    "NegotiationFairnessTracker",
    "NonDeterminismAnalyzer",
    "OutputAnalyzer",
    "PipelineTracker",
    "RAGBiasAnalyzer",
    "TemporalTracker",
    "TextFairnessAnalyzer",
    "ToolBiasAuditor",
    "TreeShapExplainer",
    "attention_weighted_rank_fairness",
    "bias_amplification",
    "brier_score_decomposition",
    "calibration_disparity",
    "classification_fairness_report",
    "comprehensive_fairness_test",
    "conditional_adverse_impact",
    "conditional_demographic_disparity",
    "conditional_demographic_disparity_with_ci",
    "counterfactual_fairness",
    "detect_protected_attributes",
    "diagnose_local_attribution",
    "discover_intersectional_groups",
    "expected_calibration_error",
    "exposure_parity_difference",
    "exposure_parity_ratio",
    "fairness_decomposition",
    "generate_recourse",
    "group_calibration_metrics",
    "identify_privileged_groups",
    "integrated_calibration_index",
    "integrated_calibration_index_with_ci",
    "intersectional_disparity_analysis",
    "judge_is_subject",
    "lundberg_fairness_decomposition",
    "maximum_calibration_error",
    "multicalibration",
    "multicalibration_with_ci",
    "ndkl",
    "net_benefit_parity",
    "noise_floor_from_runs",
    "normalized_discounted_kl_divergence",
    "pricing_disparity",
    "pricing_disparity_with_ci",
    "proxy_score",
    "representation_severity",
    "residual_bias",
    "route_explainer",
    "scan_fairness_violations",
    "sensitivity_analysis",
    "skew",
    "stress_test_fairness",
    "subgroup_robustness_audit",
}


def test_every_evaluation_capability_is_accounted_for():
    evaluation = {
        e["name"] for e in CAPABILITY_REGISTRY.values() if e["pipeline_stage"] == "evaluation"
    }
    checked = {r[0] for r in ROWS}
    assert not (checked & NO_REFERENCE), sorted(checked & NO_REFERENCE)
    assert evaluation == checked | NO_REFERENCE, {
        "registered but on neither list": sorted(evaluation - checked - NO_REFERENCE),
        "listed but no longer registered": sorted((checked | NO_REFERENCE) - evaluation),
    }
