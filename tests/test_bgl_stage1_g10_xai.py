"""
Beta Go-Live Stage 1, group g10 (xai): two critical defects, pinned at the
PUBLIC entry point.

G10-1  ``lundberg_fairness_decomposition`` (xai/decomposition/shap_fairness.py)
       A single non-finite SHAP column made every feature's proxy score NaN.
       ``score >= proxy_threshold`` is False for NaN, so ``flagged_proxies``
       came back ``[]`` -- an affirmative "no proxy features detected" -- for a
       proxy whose disparity was measured at exactly the value that gets it
       flagged in the healthy run. ``assert_identity()``, the designated
       fail-loudly gate, did not fire either (residual NaN, ``NaN > 1e-6`` is
       False), so ``to_db_row()`` wrote the row. Now refused, loudly, naming
       the columns and the row counts: could-not-check, never "clean".

G10-2  ``LimeExplainer.explain_global`` (xai/explainers/lime_adapter.py)
       In regression mode -- the DEFAULT for any model without
       ``predict_proba``, including a classifier served as a bare callable --
       every attribution came back with its sign INVERTED, because
       ``next(iter(exp.as_map()))`` selects lime's negated copy of the weights.
       A feature that drove the prediction up was reported as driving it down.

Each defect gets its refusal/correctness assertion AND a control proving
healthy data still measures. Written for BGL Stage 1; do not merge into the
existing xai test modules (twelve other agents share this checkout).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vfairness.xai.decomposition.shap_fairness import lundberg_fairness_decomposition

# G10-1  lundberg_fairness_decomposition: a NaN column is a could-not-check


def _proxy_shap(n: int = 2000, seed: int = 0):
    """SHAP matrix with a planted proxy: 'zipcode' carries the whole disparity."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    groups = np.where(g == 1, "A", "B")
    shap = np.column_stack(
        [
            rng.normal(0, 0.1, n),  # income: no group skew
            np.where(g == 1, 5.0, -6.0) + rng.normal(0, 0.1, n),  # zipcode: the proxy
        ]
    )
    return shap, groups


_KW = dict(
    feature_names=["income", "zipcode"],
    metric="demographic_parity",
    protected_attribute="group",
    subject_id="subj-1",
    audit_artifact_id="art-1",
)


def test_healthy_decomposition_still_measures_and_flags_the_proxy() -> None:
    """CONTROL. The fix must not make the decomposition refuse healthy data."""
    shap, groups = _proxy_shap()
    d = lundberg_fairness_decomposition(shap_values=shap, group_labels=groups, **_KW)

    assert d.flagged_proxies == ["zipcode"]
    assert d.per_feature["zipcode"] == pytest.approx(10.999887145702433, abs=1e-9)
    assert d.total_disparity == pytest.approx(11.004456686268458, abs=1e-9)
    assert d.proxy_scores["zipcode"] > 0.99
    assert d.identity_residual <= 1e-6
    row = d.to_db_row("auth0|probe")
    assert math.isfinite(row["total_disparity"])
    assert all(math.isfinite(v) for v in row["per_feature"].values())
    assert all(math.isfinite(v) for v in row["proxy_scores"].values())


def test_one_nan_shap_column_refuses_instead_of_reporting_no_proxies() -> None:
    """2% missing SHAP on ONE feature must not silently clear the proxy flags.

    The zipcode column here is byte-identical to the one the control measures
    at 10.999887145702433 and flags. Before the fix the public entry returned
    flagged_proxies=[] for it, and to_db_row() wrote that row.
    """
    shap, groups = _proxy_shap()
    degenerate = shap.copy()
    degenerate[:40, 0] = np.nan  # 40 of 2000 income SHAP values missing

    with pytest.raises(ValueError) as excinfo:
        lundberg_fairness_decomposition(shap_values=degenerate, group_labels=groups, **_KW)

    message = str(excinfo.value)
    # The refusal must name what could not be measured, and how much of it.
    assert "non-finite" in message
    assert "income" in message
    assert "40 of 2000" in message
    # ...and must not be mistaken for a clean result.
    assert "COULD NOT BE MEASURED" in message


def test_all_nan_shap_refuses() -> None:
    shap, groups = _proxy_shap(n=400)
    with pytest.raises(ValueError, match="non-finite"):
        lundberg_fairness_decomposition(
            shap_values=np.full_like(shap, np.nan), group_labels=groups, **_KW
        )


def test_inf_shap_column_refuses_too() -> None:
    """+inf poisons the same arithmetic (inf - inf = nan); same refusal."""
    shap, groups = _proxy_shap(n=400)
    degenerate = shap.copy()
    degenerate[3, 1] = np.inf
    with pytest.raises(ValueError) as excinfo:
        lundberg_fairness_decomposition(shap_values=degenerate, group_labels=groups, **_KW)
    assert "zipcode" in str(excinfo.value)


def test_refusal_is_not_blanket_it_is_about_the_bad_column() -> None:
    """Three states, told apart at the entry: measured / refused / refused-for-a-named-reason.

    Same fixture, same call, one changed cell: one returns a measurement, the
    other raises. A caller can tell them apart without reading the source.
    """
    shap, groups = _proxy_shap(n=600)
    measured = lundberg_fairness_decomposition(shap_values=shap, group_labels=groups, **_KW)
    assert measured.flagged_proxies == ["zipcode"]

    unmeasurable = shap.copy()
    unmeasurable[0, 0] = np.nan
    with pytest.raises(ValueError):
        lundberg_fairness_decomposition(shap_values=unmeasurable, group_labels=groups, **_KW)


# G10-2  LimeExplainer: regression attributions must not be sign-inverted


class _LinearRegressor:
    """f(x) = 5*x0 - 2*x1. Exposes only .predict, so run_mode -> 'regression'."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return 5.0 * X[:, 0] - 2.0 * X[:, 1]


def _sigmoid_endpoint(X):
    """p(approve) = sigmoid(2*income - 1*debt). income UP -> approval UP."""
    X = np.asarray(X, dtype=float)
    return 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - 1.0 * X[:, 1])))


class _ProbaClassifier:
    def predict_proba(self, X):
        p = _sigmoid_endpoint(X)
        return np.column_stack([1.0 - p, p])


def _explain(model, x, background, names):
    from vfairness.xai.explainers import LimeExplainer

    return LimeExplainer().explain_global(
        model,
        x,
        background,
        subject_id="s",
        model_hash="h",
        data_hash="d",
        feature_names=names,
        num_samples=1000,
        stability_reruns=2,
    )[0]


def test_regression_attributions_match_the_model_and_lime_itself() -> None:
    pytest.importorskip("lime")
    import lime.lime_tabular

    background = np.random.default_rng(11).uniform(-2, 2, (400, 2))
    x = np.array([[1.5, 0.2]])
    model = _LinearRegressor()

    exp = _explain(model, x, background, ["f0", "f1"])
    contrib = {a.feature: a.contribution for a in exp.attributions}

    assert exp.params["mode"] == "regression"
    # Sign must follow the coefficients: +5 on f0, -2 on f1.
    assert contrib["f0"] > 0.0, contrib
    assert contrib["f1"] < 0.0, contrib

    # ...and must agree with lime's OWN reading of the same explanation
    # (as_list uses dummy_label; the adapter used to disagree with it).
    raw = lime.lime_tabular.LimeTabularExplainer(
        background,
        feature_names=["f0", "f1"],
        mode="regression",
        discretize_continuous=True,
        kernel_width=None,
        random_state=7,
    ).explain_instance(
        x.flatten(),
        lambda a: np.asarray(model.predict(a)).reshape(-1),
        num_features=2,
        num_samples=1000,
    )
    lime_weights = sorted(w for _, w in raw.as_list())
    assert sorted(contrib.values()) == pytest.approx(lime_weights, abs=1e-9)

    # The local surrogate is additive: base + sum(attributions) == its prediction.
    # Under the inversion this was base - sum(attributions).
    assert exp.base_value + sum(contrib.values()) == pytest.approx(exp.prediction, abs=1e-6)


def test_same_classifier_agrees_through_the_callable_and_proba_paths() -> None:
    """CONTROL + the decisive cross-check.

    One model, two ways of serving it. A bare callable has no predict_proba, so
    it lands in regression mode; the same function behind predict_proba lands
    in classification mode. The attributions must agree, and both must say
    income drove the approval UP.
    """
    pytest.importorskip("lime")

    background = np.random.default_rng(0).uniform(-2, 2, (400, 2))
    x = np.array([[1.5, 0.2]])
    names = ["income", "debt"]

    as_callable = _explain(_sigmoid_endpoint, x, background, names)
    as_classifier = _explain(_ProbaClassifier(), x, background, names)

    assert as_callable.params["mode"] == "regression"
    assert as_classifier.params["mode"] == "classification"

    reg = {a.feature: a.contribution for a in as_callable.attributions}
    clf = {a.feature: a.contribution for a in as_classifier.attributions}

    assert reg["income"] > 0.0, reg
    assert reg["debt"] < 0.0, reg
    assert reg["income"] == pytest.approx(clf["income"], abs=1e-9)
    assert reg["debt"] == pytest.approx(clf["debt"], abs=1e-9)

    # Healthy data still measures: fidelity/stability are real numbers, not
    # stand-ins, and the additive identity closes on the regression path.
    for exp in (as_callable, as_classifier):
        assert exp.fidelity is not None and math.isfinite(exp.fidelity)
        assert exp.stability is not None and math.isfinite(exp.stability)
        assert math.isfinite(exp.base_value)
    assert as_callable.base_value + sum(reg.values()) == pytest.approx(
        as_callable.prediction, abs=1e-6
    )
