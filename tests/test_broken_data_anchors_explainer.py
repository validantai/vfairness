"""Check 2 pin: AnchorsExplainer refuses a model that made no decision.

Found by scripts/broken_data_check.py in the all_nan_scores world: a model
returning NaN for every row came back as rule=[], precision 1.0, coverage 1.0,
predicted_class '0', because ``_label_fn`` cast NaN to int and every perturbed
sample landed on the same invented label.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("anchor")

from vfairness.xai.explainers.anchors_adapter import AnchorsExplainer  # noqa: E402

IDS = dict(instance_id="r0", subject_id="s", model_hash="m", data_hash="d")


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


class _NoDecision:
    """A model whose every output is missing, or only some of them."""

    def __init__(self, fitted=None):
        self.fitted = fitted

    def predict(self, A):
        A = np.atleast_2d(A)
        if self.fitted is None:
            return np.full(len(A), np.nan)
        out = self.fitted.predict(A).astype(float)
        out[A[:, 1] > 0] = np.nan  # undecided on half the space
        return out


def test_healthy_model_is_still_anchored():
    from sklearn.tree import DecisionTreeClassifier

    X, y = _data()
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = AnchorsExplainer().explain_local(model, X[0], X, feature_names=["a", "b", "c"], **IDS)
    # Recomputed independently: the class the model actually predicts.
    want = int(model.predict(X[:1])[0])
    assert e.prediction == float(want)
    assert e.params["predicted_class"] == str(want)
    assert e.params["precision"] >= 0.95
    assert 0.0 < e.params["coverage"] < 1.0
    assert e.params["rule"], "a model keyed on feature a has a non-empty anchor"


def test_a_model_with_no_decision_is_refused_not_anchored():
    X, _ = _data()
    with pytest.raises(ValueError, match="COULD NOT MEASURE"):
        AnchorsExplainer().explain_local(_NoDecision(), X[0], X, **IDS)


def test_a_model_undecided_on_part_of_the_space_is_refused():
    from sklearn.tree import DecisionTreeClassifier

    X, y = _data()
    fitted = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    x = X[X[:, 1] <= 0][0]  # the instance itself HAS a decision
    with pytest.raises(ValueError, match="not finite"):
        AnchorsExplainer().explain_local(_NoDecision(fitted), x, X, **IDS)


def test_a_callable_model_with_no_decision_is_refused():
    X, _ = _data()
    with pytest.raises(ValueError, match="COULD NOT MEASURE"):
        AnchorsExplainer().explain_local(lambda A: np.full(len(A), np.nan), X[0], X, **IDS)
