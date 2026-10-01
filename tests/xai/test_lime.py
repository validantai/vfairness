"""
LIME adapter tests (backlog #P2-02).

Acceptance: a black-box / missing-rich-background path produces a LIME
Explanation with ``fidelity != None`` (local R^2) and ``stability !=
None`` (sigma over reruns).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lime")
from sklearn.linear_model import LogisticRegression

from vfairness.xai.explainers import LimeExplainer, get_explainer, route_explainer


def _toy_classifier():
    rng = np.random.RandomState(7)
    n = 300
    X = rng.normal(0, 1, (n, 4))
    y = (1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3 > 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return model, X


def test_lime_records_fidelity_and_stability() -> None:
    model, X = _toy_classifier()
    exp = LimeExplainer().explain_local(
        model,
        X[0],
        background=X,
        instance_id="s#0",
        subject_id="s",
        model_hash="m",
        data_hash="d",
        feature_names=["a", "b", "c", "d"],
        num_samples=500,
        stability_reruns=3,
    )
    assert exp.method == "lime"
    assert exp.fidelity is not None
    assert exp.stability is not None
    assert exp.stability >= 0.0
    assert exp.params["surrogate"] == "sparse_linear"
    assert len(exp.attributions) == 4
    assert "lime" in exp.library_versions


def test_lime_requires_background() -> None:
    model, _ = _toy_classifier()
    with pytest.raises(ValueError):
        LimeExplainer().explain_local(
            model,
            np.zeros(4),
            background=None,
            instance_id="s#0",
            subject_id="s",
            model_hash="m",
            data_hash="d",
        )


def test_lime_resolves_through_registry() -> None:
    assert isinstance(get_explainer("lime"), LimeExplainer)


def test_blackbox_without_background_routes_to_lime() -> None:
    # The router downgrades KernelSHAP -> LIME when background is scarce.
    decision = route_explainer(model_type="blackbox", background_available=False)
    assert decision.primary == "lime"
    assert isinstance(get_explainer(decision.primary), LimeExplainer)
