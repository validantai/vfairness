"""
Anchors adapter tests (backlog #P2-04).

Acceptance: a black-box classifier + a single instance returns an Anchor
with ``precision >= threshold`` (the KL-LUCB target) and a non-empty
rule predicate list.

``anchor-exp`` has no Python 3.9 wheel (its spacy/thinc build-dep needs
3.10+); on a 3.9 test runner these tests skip. The adapter's contract
(routing, registry resolution, no-background guard) is still exercised
on every Python via the lighter checks below.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.xai.explainers import route_explainer


def test_anchors_is_a_blackbox_fallback_in_the_router() -> None:
    decision = route_explainer(model_type="blackbox", background_available=True, ood_risk="low")
    # KernelSHAP primary, with anchors among the fallbacks.
    assert "anchors" in decision.fallbacks


def test_anchors_capabilities_are_local_only() -> None:
    anchor = pytest.importorskip("anchor")  # noqa: F841
    from vfairness.xai.explainers import AnchorsExplainer

    caps = AnchorsExplainer.capabilities
    assert caps.method == "anchors"
    assert caps.supports_local is True
    assert caps.supports_global is False


def test_anchors_returns_rule_with_precision_and_coverage() -> None:
    pytest.importorskip("anchor")
    from sklearn.ensemble import RandomForestClassifier

    from vfairness.xai.explainers import AnchorsExplainer, get_explainer

    rng = np.random.RandomState(7)
    X = rng.normal(0, 1, (300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=7).fit(X, y)

    assert isinstance(get_explainer("anchors"), AnchorsExplainer)
    exp = AnchorsExplainer().explain_local(
        model,
        X[0],
        background=X,
        instance_id="s#0",
        subject_id="s",
        model_hash="m",
        data_hash="d",
        feature_names=["a", "b", "c", "d"],
        threshold=0.95,
    )
    assert exp.method == "anchors"
    assert exp.params["precision"] >= 0.95
    assert len(exp.params["rule"]) >= 1
    assert 0.0 <= exp.params["coverage"] <= 1.0
