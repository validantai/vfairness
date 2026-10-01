"""Additive fairness-disparity decomposition: identity closes, proxy is flagged."""

import json

import numpy as np

from vfairness.evaluation.vfairness_metrics.fairness_decomposition import (
    fairness_decomposition,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _make_proxy_dataset(n=240, seed=0):
    """A 'proxy' feature correlates with the protected group and drives the
    score; a 'neutral' feature does not."""
    rng = np.random.default_rng(seed)
    protected = rng.choice(["A", "B"], size=n)
    proxy = np.where(protected == "A", rng.normal(1.0, 0.3, n), rng.normal(-1.0, 0.3, n))
    neutral = rng.normal(0.0, 1.0, n)
    X = np.column_stack([proxy, neutral])
    return X, protected


def _predict_uses_proxy(X):
    X = np.asarray(X, dtype=float)
    return _sigmoid(2.0 * X[:, 0])  # depends only on the proxy column


def test_decomposition_identity_closes_within_tolerance():
    X, protected = _make_proxy_dataset()
    res = fairness_decomposition(
        _predict_uses_proxy,
        X,
        protected,
        feature_names=["proxy", "neutral"],
        n_permutations=20,
    )
    # sum of per-feature contributions == total disparity (base value cancels).
    assert abs(res.residual) < 1e-6
    assert abs(sum(res.per_feature.values()) - res.total_disparity) < 1e-6


def test_decomposition_flags_the_proxy_feature():
    X, protected = _make_proxy_dataset()
    res = fairness_decomposition(
        _predict_uses_proxy,
        X,
        protected,
        feature_names=["proxy", "neutral"],
        n_permutations=20,
    )
    # Group A (higher proxy) is advantaged; the proxy carries the disparity.
    assert res.group_advantaged == "A"
    assert res.total_disparity > 0.05
    assert "proxy" in res.flagged_proxies
    assert "neutral" not in res.flagged_proxies
    assert abs(res.per_feature["proxy"]) > abs(res.per_feature["neutral"])
    assert res.proxy_scores["proxy"] > res.proxy_scores["neutral"]


def test_decomposition_no_disparity_when_model_ignores_group():
    rng = np.random.default_rng(7)
    n = 200
    protected = rng.choice(["A", "B"], size=n)
    X = rng.normal(0, 1, size=(n, 2))  # features independent of the group
    res = fairness_decomposition(
        lambda Z: _sigmoid(np.asarray(Z, float)[:, 0]),
        X,
        protected,
        feature_names=["f0", "f1"],
        n_permutations=15,
    )
    # No real group dependence, so the disparity is negligible (small finite
    # sample noise may register as 'low' rather than exactly 'info').
    assert abs(res.total_disparity) < 0.05
    assert res.severity in ("info", "low")
    assert res.flagged_proxies == []


def test_decomposition_json_serialisable():
    X, protected = _make_proxy_dataset(n=60)
    res = fairness_decomposition(
        _predict_uses_proxy,
        X,
        protected,
        feature_names=["proxy", "neutral"],
        n_permutations=8,
    )
    json.dumps(res.to_dict())
