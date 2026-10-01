"""
Integrated Gradients adapter tests (backlog #P2-03).

Acceptance: a PyTorch model + a single input row returns an IG
Explanation that satisfies Completeness + Sensitivity + Implementation
Invariance. For a linear model these hold exactly:

    IG_i(x) = w_i * (x_i - baseline_i)     (sensitivity / invariance)
    sum_i IG_i(x) = f(x) - f(baseline)     (completeness)

so the linear case pins the adapter to closed-form ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("captum")

from vfairness.xai.explainers import IntegratedGradientsExplainer, get_explainer


def _linear_model(weights: list[float], bias: float):
    import torch.nn as nn

    model = nn.Linear(len(weights), 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([weights], dtype=torch.float32))
        model.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    model.eval()
    return model


def test_ig_completeness_on_linear_model() -> None:
    w = [2.0, -3.0, 0.5]
    model = _linear_model(w, bias=1.0)
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)

    exp = IntegratedGradientsExplainer().explain_local(
        model,
        x,
        background=None,  # zeros baseline
        instance_id="s1#row-0",
        subject_id="s1",
        model_hash="m",
        data_hash="d",
        feature_names=["a", "b", "c"],
    )

    attrs = np.array([a.contribution for a in exp.attributions])
    # Completeness: attributions sum to f(x) - f(baseline).
    assert np.isclose(attrs.sum(), exp.prediction - exp.base_value, atol=1e-3)
    assert exp.params["completeness_residual"] < 1e-3
    # Sensitivity / implementation invariance: exact for a linear model
    # with a zeros baseline -> IG_i = w_i * x_i.
    expected = np.array(w) * x
    assert np.allclose(attrs, expected, atol=1e-3)
    assert exp.method == "ig"
    assert exp.params["baseline"] == "zeros"
    assert "captum" in exp.library_versions


def test_ig_uses_background_mean_as_baseline() -> None:
    model = _linear_model([1.0, 1.0], bias=0.0)
    x = np.array([3.0, 3.0], dtype=np.float32)
    background = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)  # mean = [2,2]

    exp = IntegratedGradientsExplainer().explain_local(
        model,
        x,
        background=background,
        instance_id="s1#row-0",
        subject_id="s1",
        model_hash="m",
        data_hash="d",
    )
    attrs = np.array([a.contribution for a in exp.attributions])
    # baseline = mean([1,3]) = 2 per feature -> IG_i = 1 * (3 - 2) = 1.
    assert exp.params["baseline"] == "background_mean"
    assert np.allclose(attrs, [1.0, 1.0], atol=1e-3)
    assert np.isclose(attrs.sum(), exp.prediction - exp.base_value, atol=1e-3)


def test_ig_resolves_through_registry() -> None:
    assert isinstance(get_explainer("ig"), IntegratedGradientsExplainer)
