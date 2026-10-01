"""Feature attribution (per-decision XAI) test suite, Workstream D.

Covers vfairness.evaluation.vfairness_metrics.attribution.FeatureAttributionExplainer.
Uses a synthetic dataset with a KNOWN dominant feature so we can assert the
explainer ranks it first, for both global (permutation) and local (occlusion)
scopes, with and without sklearn / a fitted estimator.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.attribution import (
    AttributionResult,
    FeatureAttributionExplainer,
)


# y depends strongly on x0, weakly on x1, not at all on x2.
def _make_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    score = 3.0 * X[:, 0] + 0.5 * X[:, 1] + 0.0 * X[:, 2]
    y = (score > 0).astype(int)
    return X, y, score


def _linear_predict(X):
    """A transparent 'model': the exact score function above."""
    X = np.asarray(X)
    return 3.0 * X[:, 0] + 0.5 * X[:, 1] + 0.0 * X[:, 2]


def test_requires_callable():
    with pytest.raises(TypeError):
        FeatureAttributionExplainer(predict="not callable")


def test_global_numpy_permutation_ranks_dominant_feature_first():
    X, _y, _ = _make_xy()
    expl = FeatureAttributionExplainer(_linear_predict, feature_names=["x0", "x1", "x2"])
    res = expl.global_importance(X)  # no y -> numpy fallback
    assert isinstance(res, AttributionResult)
    assert res.scope == "global"
    assert res.method == "permutation_numpy"
    assert res.top(1)[0].feature == "x0", [c.feature for c in res.top(3)]
    least = min(res.contributions, key=lambda c: c.importance)
    assert least.feature == "x2"


def test_global_sklearn_permutation_when_y_provided():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    X, _y, score = _make_xy()
    model = LinearRegression().fit(X, score)
    expl = FeatureAttributionExplainer(model.predict, feature_names=["x0", "x1", "x2"])
    res = expl.global_importance(X, y=score, n_repeats=5)
    assert res.method == "permutation"
    assert res.top(1)[0].feature == "x0"


def test_local_occlusion_attributes_to_dominant_feature():
    X, _y, _ = _make_xy()
    expl = FeatureAttributionExplainer(_linear_predict, feature_names=["x0", "x1", "x2"])
    row = np.array([2.5, 0.1, 1.0])  # large positive x0
    res = expl.explain_decision(row, background=X)
    assert res.scope == "local"
    assert res.method == "occlusion"
    assert res.prediction is not None and res.base_value is not None
    top = res.top(1)[0]
    assert top.feature == "x0"
    assert top.direction == "increase"
    x2 = next(c for c in res.contributions if c.feature == "x2")
    assert x2.importance == pytest.approx(0.0, abs=1e-6)


def test_local_negative_feature_decreases_prediction():
    X, _y, _ = _make_xy()
    expl = FeatureAttributionExplainer(_linear_predict, feature_names=["x0", "x1", "x2"])
    res = expl.explain_decision(np.array([-3.0, 0.0, 0.0]), background=X)
    x0 = next(c for c in res.contributions if c.feature == "x0")
    assert x0.direction == "decrease"


def test_to_dict_is_json_friendly():
    import json

    X, _y, _ = _make_xy()
    expl = FeatureAttributionExplainer(_linear_predict, feature_names=["x0", "x1", "x2"])
    d = expl.explain_decision(np.array([1.0, 1.0, 1.0]), background=X).to_dict()
    assert set(d) == {"scope", "method", "base_value", "prediction", "notes", "contributions"}
    assert all(
        set(c) == {"feature", "importance", "direction", "signed_value"} for c in d["contributions"]
    )
    json.dumps(d)  # must not raise


def test_dataframe_feature_names_inferred():
    pd = pytest.importorskip("pandas")
    X, _y, _ = _make_xy()
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma"])
    expl = FeatureAttributionExplainer(_linear_predict)  # no names -> infer from df
    res = expl.global_importance(df)
    assert {c.feature for c in res.contributions} == {"alpha", "beta", "gamma"}
    assert res.top(1)[0].feature == "alpha"
