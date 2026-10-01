"""Check 2 specs for the explainers and the explanation layer.

HOW A WORLD BECOMES AN EXPLAINER'S INPUT. An explainer is handed a model and
the rows to explain. The world carries the rows (income, stratum and group as
features) and the scores a model produced for them (``w.prob``) or its decisions
(``w.p``). So the model explained here is the world's own scorer, recovered from
the world: a small regressor fitted on the features against ``w.prob`` (or a
classifier against ``w.p`` for the methods that explain a decision: anchors and
DiCE). When the world has no finite score at all, the scorer the world describes
returns a missing value for every row, and that is the model handed over:
``_NanScorer``. That is the faithful reading of "every score missing" and it
reaches the library, rather than stopping at a fit that cannot run. The
tree and linear SHAP adapters read a fitted estimator's internals, so for them
the fit is the only door and an unfittable world refuses there.

WHAT A MEASUREMENT IS. An attribution is a measurement of the model, so a
constant model has attributions of exactly 0.0 and that is TRUE: ``needs=()``.
Only a quantity that is undefined for a constant model declares a need: the
faithfulness score (a removal curve over a prediction that cannot move) and a
counterfactual (a decision that cannot flip). Explainers are not group
comparisons, so ``grouped_min=0``; the decomposition and the proxy score compare
two groups and use their own floor, which is one row per group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = ["income", "stratum", "group"]
_BG = 60  # background rows; small keeps KernelSHAP and LIME fast


def _features(w) -> np.ndarray:
    return np.column_stack(
        [np.asarray(w.yr, float), (w.strata == "s2").astype(float), (w.s == "b").astype(float)]
    ).reshape(-1, 3)


class _NanScorer:
    """The world's scorer when every score it produced is missing."""

    def predict(self, A):
        return np.full(len(np.atleast_2d(A)), np.nan)

    def predict_proba(self, A):
        return np.full((len(np.atleast_2d(A)), 2), np.nan)

    def __call__(self, A):
        return self.predict(A)


def _fit_regressor(w, kind="tree"):
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    X, t = _features(w), np.asarray(w.prob, float)
    ok = np.isfinite(t)
    m = (
        LinearRegression()
        if kind == "linear"
        else DecisionTreeRegressor(max_depth=3, random_state=0)
    )
    return m.fit(X[ok], t[ok])  # raises on zero finite scores: nothing to fit


def _scorer(w):
    """The world's score function, for model-agnostic methods."""
    if not np.isfinite(np.asarray(w.prob, float)).any():
        return _NanScorer()
    return _fit_regressor(w)


def _decider(w):
    """The world's decision function (labels), for anchors and DiCE."""
    from sklearn.tree import DecisionTreeClassifier

    p = np.asarray(w.p, float)
    ok = np.isfinite(p)
    if not ok.any():
        return _NanScorer()
    return DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        _features(w)[ok], p[ok].astype(int)
    )


def _bg(w) -> np.ndarray:
    """A fixed random subsample. A stride would alias with the alternating
    stratum column and freeze it, which is a background the world does not have."""
    X = _features(w)
    if len(X) <= _BG:
        return X
    return X[np.sort(np.random.default_rng(0).choice(len(X), _BG, replace=False))]


_IDS = dict(subject_id="s", model_hash="m", data_hash="d")


def _contribs(r):
    if isinstance(r, list):
        if not r:
            return []
        return [[a.contribution for a in e.attributions] for e in r]
    return [a.contribution for a in r.attributions]


# Every explainer below is local or per row; a world with no rows has no
# instance, and n_equals_2 / one_row_minority are not judged for an ungrouped
# number (harness rule).


def _tree_shap(w):
    from vfairness.xai.explainers.shap_adapter import TreeShapExplainer

    return TreeShapExplainer().explain_global(
        _fit_regressor(w), _features(w), feature_names=NAMES, units="raw", **_IDS
    )


def _linear_shap(w):
    from vfairness.xai.explainers.shap_adapter import LinearShapExplainer

    return LinearShapExplainer().explain_local(
        _fit_regressor(w, "linear"),
        _features(w)[:1],
        _bg(w),
        instance_id="r0",
        feature_names=NAMES,
        **_IDS,
    )


def _kernel_shap(w):
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer

    return KernelShapExplainer().explain_local(
        _scorer(w).predict,
        _features(w)[:1],
        _bg(w),
        instance_id="r0",
        feature_names=NAMES,
        nsamples=64,
        **_IDS,
    )


def _lime(w):
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    return LimeExplainer().explain_local(
        _scorer(w),
        _features(w)[:1],
        _bg(w),
        instance_id="r0",
        feature_names=NAMES,
        mode="regression",
        num_samples=300,
        stability_reruns=2,
        **_IDS,
    )


def _ig(w):
    import torch

    from vfairness.xai.explainers.ig_adapter import IntegratedGradientsExplainer

    X, t = _features(w), np.asarray(w.prob, float)
    ok = np.isfinite(t)
    # The world's scorer as a linear torch net: least squares on the finite
    # rows, every weight NaN when no score was observed (the _NanScorer case).
    if ok.any():
        A = np.column_stack([X[ok], np.ones(ok.sum())])
        coef = np.linalg.lstsq(A, t[ok], rcond=None)[0]
    else:
        coef = np.full(X.shape[1] + 1, np.nan)
    net = torch.nn.Linear(3, 1)
    with torch.no_grad():
        net.weight.copy_(torch.tensor(coef[:3], dtype=torch.float32).reshape(1, 3))
        net.bias.copy_(torch.tensor(coef[3:], dtype=torch.float32))
    return IntegratedGradientsExplainer().explain_local(
        net, X[:1], _bg(w), instance_id="r0", feature_names=NAMES, target=0, n_steps=16, **_IDS
    )


def _anchors(w):
    from vfairness.xai.explainers.anchors_adapter import AnchorsExplainer

    return AnchorsExplainer().explain_local(
        _decider(w), _features(w)[:1], _bg(w), instance_id="r0", feature_names=NAMES, **_IDS
    )


def _dice(w):
    from vfairness.xai.explainers.dice_adapter import DiceCounterfactualExplainer

    X = _features(w)
    df = pd.DataFrame(X, columns=NAMES)
    df["y_pred"] = np.asarray(w.p, float)
    return DiceCounterfactualExplainer().explain_local(
        _decider(w),
        X[:1],
        instance_id="r0",
        feature_names=NAMES,
        dataframe=df,
        outcome_name="y_pred",
        total_CFs=2,
        **_IDS,
    )


def _shap_matrix(w) -> np.ndarray:
    import shap

    m = _fit_regressor(w)
    return np.asarray(shap.TreeExplainer(m).shap_values(_features(w)), float).reshape(-1, 3)


def _lundberg(w):
    from vfairness.xai.decomposition import lundberg_fairness_decomposition

    return lundberg_fairness_decomposition(
        shap_values=_shap_matrix(w),
        group_labels=np.asarray(w.s),
        feature_names=NAMES,
        metric="demographic_parity",
        protected_attribute="group",
        subject_id="s",
        audit_artifact_id="a",
    )


def _per_feature_disparity(w) -> dict:
    """Between-group mean difference of each SHAP column, as the decomposition
    computes it; NaN for a group with no rows, which is what that mean is."""
    phi = _shap_matrix(w)
    s = np.asarray(w.s)
    a, b = (sorted(set(s.tolist())) + [None, None])[:2]
    out = {}
    for j, n in enumerate(NAMES):
        ga, gb = phi[s == a, j], phi[s == b, j]
        out[n] = float(ga.mean() - gb.mean()) if len(ga) and len(gb) else float("nan")
    return out


def _proxy(w):
    from vfairness.xai.decomposition import proxy_score

    return proxy_score(_per_feature_disparity(w))


def _occlusion(predict, x, bg):
    base = float(np.asarray(predict(x.reshape(1, -1))).ravel()[0])
    mu = bg.mean(axis=0)
    out = []
    for j in range(x.size):
        z = x.copy()
        z[j] = mu[j]
        out.append(base - float(np.asarray(predict(z.reshape(1, -1))).ravel()[0]))
    return np.asarray(out)


def _diagnose(w):
    from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
        diagnose_local_attribution,
    )

    # A linear scorer: its exact attribution is coef * (x - mean), so no two
    # features tie by accident of a tree leaf, and a tie is the removal curve's
    # own could-not-check (see removal_curve_auc).
    full = (
        _fit_regressor(w, "linear")
        if np.isfinite(np.asarray(w.prob, float)).any()
        else _NanScorer()
    ).predict
    x_full = _features(w)[0]
    bg_full = _bg(w)
    # The probe's own documented remedy for a background column with no spread
    # (a world where every row shares a group, or an income): "drop the constant
    # columns from both x and the background before probing". A faithful caller
    # does that, so this one does, and the model still sees every column.
    keep = np.array(
        [np.ptp(c[np.isfinite(c)]) > 0 if np.isfinite(c).any() else False for c in bg_full.T]
    )
    if not keep.any():
        keep[:] = True
    fill = bg_full[0].copy()

    def f(A):
        A = np.atleast_2d(A)
        Z = np.tile(fill, (len(A), 1))
        Z[:, keep] = A
        return full(Z)

    x, bg = x_full[keep], bg_full[:, keep]
    return diagnose_local_attribution(f, x, _occlusion(f, x, bg), bg, n_seeds=3)


def _fae(w):
    from vfairness.evaluation.vfairness_metrics.attribution import FeatureAttributionExplainer

    return FeatureAttributionExplainer(_scorer(w).predict, NAMES)


def _cal_explanation(w):
    from vfairness.explainer import FairnessExplainer
    from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer

    rep = CalibrationAnalyzer(w.y, w.prob, w.s).full_analysis()
    return FairnessExplainer.explain(rep)


def _disparity_verdict(REFUSED):
    def read(r):
        cards = [
            e for e in r.explanations if e.metric_name == "Calibration Disparity Across Groups"
        ]
        if cards:
            c = cards[0]
            if c.severity not in ("medium",) or not str(c.value).startswith("Significant"):
                return REFUSED
            return 1.0
        if "No significant calibration disparity." in r.summary:
            return 0.0
        return REFUSED

    return read


def specs(Spec, REFUSED, flags_absence) -> list:
    def attribution_spec(cap, label, call, consumes=("prob",)):
        return Spec(cap, label, call, _contribs, consumes=consumes, grouped_min=0)

    out = [
        attribution_spec("TreeShapExplainer", "TreeShapExplainer.explain_global", _tree_shap),
        attribution_spec("LinearShapExplainer", "LinearShapExplainer.explain_local", _linear_shap),
        attribution_spec("KernelShapExplainer", "KernelShapExplainer.explain_local", _kernel_shap),
        attribution_spec("LimeExplainer", "LimeExplainer.explain_local", _lime),
        attribution_spec(
            "IntegratedGradientsExplainer", "IntegratedGradientsExplainer.explain_local", _ig
        ),
        # An anchor is a rule with a measured precision and coverage. A constant
        # decision has the empty rule at precision 1.0: that IS its anchor.
        Spec(
            "AnchorsExplainer",
            "AnchorsExplainer.explain_local",
            _anchors,
            lambda r: [r.params.get("precision"), r.params.get("coverage")],
            consumes=("p",),
            grouped_min=0,
        ),
        # A counterfactual needs a decision that can flip: both decisions present.
        Spec(
            "DiceCounterfactualExplainer",
            "DiceCounterfactualExplainer.explain_local",
            _dice,
            lambda r: [c.proximity for c in r.counterfactuals] or REFUSED,
            needs=("ppos", "pneg"),
            consumes=("p",),
            grouped_min=0,
        ),
        Spec(
            "lundberg_fairness_decomposition",
            "lundberg_fairness_decomposition",
            _lundberg,
            lambda r: [r.total_disparity, *r.per_feature.values()],
            consumes=("prob",),
            min_group_size=1,
        ),
        Spec(
            "proxy_score",
            "proxy_score",
            _proxy,
            lambda r: r,
            consumes=("prob",),
            min_group_size=1,
        ),
        # Faithfulness is a removal curve over the prediction: undefined when
        # the prediction cannot move.
        Spec(
            "diagnose_local_attribution",
            "diagnose_local_attribution.faithfulness",
            _diagnose,
            lambda r: REFUSED if r["faithfulness"] is None else r["faithfulness"],
            needs=("var_prob",),
            consumes=("prob",),
            grouped_min=0,
        ),
        Spec(
            "diagnose_local_attribution",
            "diagnose_local_attribution.adversarial_flag",
            _diagnose,
            lambda r: REFUSED if r["adversarial_flag"] is None else r["adversarial_flag"],
            consumes=("prob",),
            grouped_min=0,
        ),
        Spec(
            "FeatureAttributionExplainer",
            "FeatureAttributionExplainer.global_importance",
            lambda w: _fae(w).global_importance(_features(w), n_repeats=3, random_state=0),
            lambda r: [c.importance for c in r.contributions],
            consumes=("prob",),
            grouped_min=0,
        ),
        Spec(
            "FeatureAttributionExplainer",
            "FeatureAttributionExplainer.explain_decision",
            lambda w: _fae(w).explain_decision(_features(w)[:1], _bg(w)),
            lambda r: [c.signed_value for c in r.contributions],
            consumes=("prob",),
            grouped_min=0,
        ),
        # The explanation layer over a real report: every graded card is a
        # reading; a could-not-check card is not.
        Spec(
            "FairExplAIner",
            "FairExplAIner.explain_report",
            lambda w: _fair_explainer_report(w),
            lambda r: {
                k: v.get("value")
                for k, v in r["metrics"].items()
                if v.get("severity") != "could_not_check"
            },
        ),
        Spec(
            "FairnessExplainer",
            "FairnessExplainer.explain[CalibrationReport].ece",
            _cal_explanation,
            lambda r: (
                [e.value for e in r.explanations if e.metric_name.startswith("Expected")] or REFUSED
            ),
            consumes=("prob",),
            grouped_min=0,
        ),
        Spec(
            "FairnessExplainer",
            "FairnessExplainer.explain[CalibrationReport].disparity",
            _cal_explanation,
            _disparity_verdict(REFUSED),
            consumes=("prob",),
        ),
    ]
    return out


def _fair_explainer_report(w):
    from vfairness.evaluation.vfairness_metrics import (
        FairExplAIner,
        classification_fairness_report,
    )

    return FairExplAIner().explain_report(classification_fairness_report(w.y, w.p, w.s))
