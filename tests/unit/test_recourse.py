"""Algorithmic recourse: counterfactuals flip the decision, respect constraints."""

import json

import numpy as np

from vfairness.evaluation.vfairness_metrics.recourse import generate_recourse


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _predict(X):
    # Decision rises with feature 0, falls with feature 1, ignores feature 2.
    X = np.asarray(X, dtype=float)
    return _sigmoid(1.5 * X[:, 0] - 1.5 * X[:, 1])


def _background(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3, 3, size=(n, 3))


def test_recourse_finds_decision_flip():
    bg = _background()
    x = np.array([-2.0, 2.0, 0.0])  # strongly below threshold
    assert _predict(x.reshape(1, -1))[0] < 0.5
    res = generate_recourse(
        _predict,
        x,
        bg,
        feature_names=["income", "debt", "noise"],
        threshold=0.5,
        random_state=1,
    )
    assert res.found
    assert res.desired_outcome == "above_threshold"
    for cf in res.counterfactuals:
        assert cf.prediction_after >= 0.5
        assert cf.n_changes >= 1
        assert cf.sentence.endswith("approved.")


def test_recourse_respects_immutable_features():
    bg = _background(seed=1)
    x = np.array([-2.0, 2.0, 0.0])
    res = generate_recourse(
        _predict,
        x,
        bg,
        feature_names=["income", "debt", "noise"],
        threshold=0.5,
        immutable_features=["income", "debt"],
        random_state=2,
    )
    # Only 'noise' may change, and it does not affect the score, so no recourse.
    for cf in res.counterfactuals:
        for ch in cf.changes:
            assert ch.feature == "noise"
    assert not res.found


def test_recourse_ranks_by_sparsity_then_proximity():
    bg = _background(seed=3)
    x = np.array([-1.0, 1.0, 0.0])
    res = generate_recourse(
        _predict,
        x,
        bg,
        feature_names=["income", "debt", "noise"],
        threshold=0.5,
        max_changes=3,
        random_state=4,
    )
    if len(res.counterfactuals) >= 2:
        a, b = res.counterfactuals[0], res.counterfactuals[1]
        assert (a.n_changes, a.proximity) <= (b.n_changes, b.proximity)


def test_recourse_json_serialisable():
    bg = _background(seed=5)
    x = np.array([-2.0, 2.0, 0.0])
    res = generate_recourse(_predict, x, bg, feature_names=["a", "b", "c"], random_state=6)
    json.dumps(res.to_dict())
