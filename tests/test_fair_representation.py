"""
Tests for FairRepresentationTransformer (adversarial fair representation).

Verifies that the learned latent representation:
  - has the requested dimensionality and excludes protected attributes,
  - reduces the latent code's ability to predict the protected attribute
    versus the raw features (the invariance goal),
  - is deterministic given a seed, and round-trips through fit/transform.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from vfairness import FairRepresentationTransformer


def _biased_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, size=n)
    # Several features that leak group membership to varying degrees.
    f1 = rng.normal(loc=2.0 * grp, scale=1.0, size=n)
    f2 = rng.normal(loc=-1.5 * grp, scale=1.0, size=n)
    f3 = rng.normal(loc=0.8 * grp, scale=1.0, size=n)
    f4 = rng.normal(loc=0.0, scale=1.0, size=n)  # group-neutral
    f5 = rng.normal(loc=1.2 * grp, scale=1.0, size=n)
    y = (f1 - f2 + f3 + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"grp": grp, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}), y


def _group_predictability(Z, g):
    """AUC-like: train logistic reg to predict group from Z, return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return cross_val_score(LogisticRegression(max_iter=500), Z, g, cv=3).mean()


def test_output_shape_and_excludes_protected():
    X, y = _biased_df()
    t = FairRepresentationTransformer(protected_attributes=["grp"], representation_dim=4, epochs=80)
    Z = t.fit_transform(X, y)
    assert list(Z.columns) == ["rep_0", "rep_1", "rep_2", "rep_3"]
    assert "grp" not in Z.columns
    assert len(Z) == len(X)


def test_reduces_group_predictability():
    X, y = _biased_df()
    t = FairRepresentationTransformer(
        protected_attributes=["grp"],
        representation_dim=4,
        lambda_fairness=3.0,
        epochs=300,
    )
    Z = t.fit_transform(X, y)
    raw_pred = _group_predictability(X[["f1", "f2"]].values, X["grp"].values)
    rep_pred = _group_predictability(Z.values, X["grp"].values)
    # The representation should be less group-predictive than the raw features.
    assert rep_pred <= raw_pred + 0.02
    # And clearly better than the raw leak in the typical case.
    assert rep_pred < 0.95


def test_deterministic_with_seed():
    X, y = _biased_df()
    a = FairRepresentationTransformer(
        protected_attributes=["grp"], representation_dim=3, epochs=50, random_state=7
    )
    b = FairRepresentationTransformer(
        protected_attributes=["grp"], representation_dim=3, epochs=50, random_state=7
    )
    Za = a.fit_transform(X, y)
    Zb = b.fit_transform(X, y)
    np.testing.assert_allclose(Za.values, Zb.values, rtol=1e-4, atol=1e-4)


def test_fit_result_metadata():
    X, y = _biased_df()
    t = FairRepresentationTransformer(protected_attributes=["grp"], representation_dim=4, epochs=50)
    t.fit(X, y)
    assert t.is_fitted
    assert t.fit_result.method == "fair_representation"
    assert t.fit_result.fit_metrics["representation_dim"] == 4
    assert t.fit_result.n_features_transformed == 4
