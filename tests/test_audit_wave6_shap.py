"""Wave-6 audit pins: SHAP class-consistency on the modern (shap >= 0.45) convention.

Finding (explainer-xai, CRITICAL): _class_consistent only handled the legacy
list-of-arrays convention. shap >= 0.45 returns a single
(n_samples, n_features, n_outputs) ndarray, so the class-selection branch was
dead code: attributions stayed interleaved across classes (feature j's class-1
value landed on feature j+1's name, the back half of the features was silently
dropped), base_value came from class 0, and explain_global crashed with a
TypeError.

The stub-module tests pin the adapter code path on any machine; the
importorskip tests prove additivity (base + sum(attributions) equals the
class-1 probability) against whatever real shap version is installed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vfairness.xai.explainers.shap_adapter import (
    KernelShapExplainer,
    TreeShapExplainer,
    _class_consistent,
)

# ── fixtures: a stub shap module speaking the MODERN (>= 0.45) convention ───

N_FEATURES = 4
FEATURE_NAMES = ["age", "income", "tenure", "score"]
# One sample, 4 features, 2 classes; class axis LAST (modern convention).
SV_3D = np.array(
    [
        [
            [0.10, -0.10],
            [0.20, -0.20],
            [0.05, -0.05],
            [0.30, -0.30],
        ]
    ]
)
EV_2 = np.array([0.7, 0.3])


def _install_stub_shap(monkeypatch: pytest.MonkeyPatch, shap_values, expected_value) -> None:
    stub = types.ModuleType("shap")
    stub.__version__ = "0.52-stub"

    class _Explainer:
        def __init__(self, model, background=None):
            self.expected_value = expected_value

        def shap_values(self, X, **kwargs):
            return shap_values

    stub.TreeExplainer = _Explainer
    stub.KernelExplainer = _Explainer
    monkeypatch.setitem(sys.modules, "shap", stub)


# ── negative case FIRST: the exact scenario that used to be wrong ───────────


def test_modern_3d_local_selects_class1_values_names_and_base(monkeypatch):
    """Used to return interleaved values under shifted names with a class-0 base."""
    _install_stub_shap(monkeypatch, SV_3D, EV_2)
    exp = TreeShapExplainer().explain_local(
        model=object(),
        x=np.zeros(N_FEATURES),
        instance_id="i0",
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
    )
    # All 4 features present (the bug dropped the back half).
    assert [a.feature for a in exp.attributions] == FEATURE_NAMES
    # Each name carries ITS OWN class-1 value (the bug put age's class-1
    # value on income's name).
    got = {a.feature: a.contribution for a in exp.attributions}
    assert got == pytest.approx({"age": -0.10, "income": -0.20, "tenure": -0.05, "score": -0.30})
    # Base value from class 1, not class 0.
    assert exp.base_value == pytest.approx(0.3)
    # prediction = base1 + sum(phi1), a class-1 quantity, not a hybrid.
    assert exp.prediction == pytest.approx(0.3 - 0.65)


def test_modern_3d_helper_selects_class1_slice():
    values, base = _class_consistent(SV_3D, EV_2)
    assert values.shape == (1, N_FEATURES)
    assert np.allclose(values, SV_3D[:, :, 1])
    assert base == pytest.approx(0.3)


def test_modern_3d_global_does_not_crash_and_is_additive(monkeypatch):
    """explain_global used to raise TypeError on the 3-D array."""
    sv = np.concatenate([SV_3D, 2.0 * SV_3D])  # (2, 4, 2)
    _install_stub_shap(monkeypatch, sv, EV_2)
    exps = TreeShapExplainer().explain_global(
        model=object(),
        X=np.zeros((2, N_FEATURES)),
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
    )
    assert len(exps) == 2
    for i, e in enumerate(exps):
        assert [a.feature for a in e.attributions] == FEATURE_NAMES
        row1 = sv[i, :, 1]
        assert [a.contribution for a in e.attributions] == pytest.approx(list(row1))
        assert e.base_value == pytest.approx(0.3)
        assert e.prediction == pytest.approx(0.3 + row1.sum())


def test_modern_3d_kernel_local_selects_class1(monkeypatch):
    _install_stub_shap(monkeypatch, SV_3D, EV_2)
    exp = KernelShapExplainer().explain_local(
        model=lambda X: np.zeros((len(X), 2)),
        x=np.zeros(N_FEATURES),
        background=np.zeros((5, N_FEATURES)),
        instance_id="i0",
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
    )
    assert [a.contribution for a in exp.attributions] == pytest.approx(list(SV_3D[0, :, 1]))
    assert exp.base_value == pytest.approx(0.3)


# ── does-not-overcorrect: legacy and single-output conventions unchanged ────


def test_legacy_list_convention_still_selects_class1():
    class0 = np.array([[0.1, -0.2]])
    class1 = np.array([[-0.1, 0.2]])
    values, base = _class_consistent([class0, class1], np.array([0.7, 0.3]))
    assert np.allclose(values, class1)
    assert base == pytest.approx(0.3)


def test_single_output_2d_passthrough_unchanged():
    sv = np.array([[0.1, -0.2], [0.3, 0.4]])  # regression / margin output
    values, base = _class_consistent(sv, 0.5)
    assert values is sv
    assert base == 0.5
    # 1-element array expected_value also passes through as before.
    values2, base2 = _class_consistent(sv, np.array([0.5]))
    assert np.allclose(values2, sv) and base2 == 0.5


def test_modern_3d_single_output_channel_clamps_index():
    sv = SV_3D[:, :, :1]  # (1, 4, 1): only one output channel
    values, base = _class_consistent(sv, np.array([0.7]))
    assert np.allclose(values, sv[:, :, 0])
    assert base == pytest.approx(0.7)


# ── real shap: additivity against the installed version ─────────────────────


@pytest.fixture(scope="module")
def rf_setup():
    pytest.importorskip("shap")
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, N_FEATURES))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)
    model = sklearn_ensemble.RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    return model, X


def test_real_shap_tree_local_additivity_class1(rf_setup):
    model, X = rf_setup
    x0 = X[0]
    proba1 = model.predict_proba(x0.reshape(1, -1))[0, 1]
    exp = TreeShapExplainer().explain_local(
        model,
        x0,
        instance_id="i0",
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
    )
    assert len(exp.attributions) == N_FEATURES
    assert [a.feature for a in exp.attributions] == FEATURE_NAMES
    # Canonical local-accuracy property: base1 + sum(phi1) == P(class 1).
    assert exp.prediction == pytest.approx(proba1, abs=1e-6)
    assert exp.base_value + sum(a.contribution for a in exp.attributions) == pytest.approx(
        proba1, abs=1e-6
    )


def test_real_shap_tree_global_additivity_class1(rf_setup):
    model, X = rf_setup
    exps = TreeShapExplainer().explain_global(
        model,
        X[:3],
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
    )
    proba1 = model.predict_proba(X[:3])[:, 1]
    assert len(exps) == 3
    for e, p1 in zip(exps, proba1):
        assert e.prediction == pytest.approx(p1, abs=1e-6)


def test_real_shap_kernel_local_additivity_class1(rf_setup):
    model, X = rf_setup
    x0 = X[0]
    proba1 = model.predict_proba(x0.reshape(1, -1))[0, 1]
    exp = KernelShapExplainer().explain_local(
        model.predict_proba,
        x0,
        X[:25],
        instance_id="i0",
        subject_id="s0",
        model_hash="mh",
        data_hash="dh",
        feature_names=FEATURE_NAMES,
        nsamples=200,
    )
    # KernelSHAP enforces local accuracy; with 4 features and nsamples=200 the
    # subset space is fully enumerated, so this is exact up to float noise.
    assert exp.prediction == pytest.approx(proba1, abs=1e-6)
