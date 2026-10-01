"""Per-instance Shapley matrix: exact additivity (completeness axiom)."""

import numpy as np

from vfairness.evaluation.vfairness_metrics.attribution import (
    FeatureAttributionExplainer,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _linear_predict(X):
    X = np.asarray(X, dtype=float)
    w = np.array([2.0, -1.0, 0.5, 0.0])
    return _sigmoid(X @ w)


def test_shapley_matrix_is_exactly_additive():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(40, 4))
    ex = FeatureAttributionExplainer(_linear_predict, feature_names=["a", "b", "c", "d"])
    mat = ex.shapley_matrix(X, n_permutations=15, random_state=1)

    values = np.asarray(mat["values"], dtype=float)
    base = mat["base_value"]
    preds = np.asarray(mat["predictions"], dtype=float)

    # Completeness: base_value + sum(contributions) == prediction, exactly.
    assert np.allclose(base + values.sum(axis=1), preds, atol=1e-9)
    # And those predictions match the real model on every row.
    assert np.allclose(preds, _linear_predict(X), atol=1e-9)
    assert values.shape == (40, 4)
    assert mat["method"] == "permutation_shapley"


def test_shapley_matrix_zero_weight_feature_is_negligible():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(60, 4))
    ex = FeatureAttributionExplainer(_linear_predict, feature_names=["a", "b", "c", "d"])
    mat = ex.shapley_matrix(X, n_permutations=20, random_state=3)
    values = np.asarray(mat["values"], dtype=float)
    # Feature 'd' has weight 0; its mean |contribution| must be far below 'a'.
    mean_abs = np.abs(values).mean(axis=0)
    assert mean_abs[3] < 0.1 * mean_abs[0]


def test_shapley_matrix_respects_max_rows():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(500, 4))
    ex = FeatureAttributionExplainer(_linear_predict, feature_names=["a", "b", "c", "d"])
    mat = ex.shapley_matrix(X, n_permutations=5, max_rows=50)
    assert mat["n_rows"] == 50
    assert len(mat["values"]) == 50


def test_shapley_matrix_json_serialisable():
    import json

    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(10, 4))
    ex = FeatureAttributionExplainer(_linear_predict)
    json.dumps(ex.shapley_matrix(X, n_permutations=5))
