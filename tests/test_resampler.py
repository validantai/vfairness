"""
Tests for Resampler (Kamiran & Calders, 2012 over-/under-sampling).

Verifies that resampling:
  - balances group (and group x label) cell sizes,
  - over-sampling grows to the largest cell, under-sampling shrinks to the smallest,
  - keeps all input columns and aligns y,
  - leaves transform(X) as a feature pass-through.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness import Resampler


def _make_imbalanced_df(seed=0):
    rng = np.random.default_rng(seed)
    # grp=0 minority (100), grp=1 majority (400); within-group label imbalance too.
    g0 = pd.DataFrame(
        {"grp": 0, "score": rng.normal(size=100), "y": (rng.random(100) < 0.2).astype(int)}
    )
    g1 = pd.DataFrame(
        {"grp": 1, "score": rng.normal(size=400), "y": (rng.random(400) < 0.6).astype(int)}
    )
    df = pd.concat([g0, g1], ignore_index=True)
    return df.drop(columns=["y"]), df["y"].values


def test_oversample_balances_groups():
    X, y = _make_imbalanced_df()
    r = Resampler(protected_attributes=["grp"], strategy="oversample", balance_by="group")
    r.fit(X, y)
    X_res, y_res = r.get_resampled_data(X, y)
    counts = X_res["grp"].value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)
    # Oversampling should not drop below the original majority size.
    assert counts[1] >= 400


def test_undersample_balances_groups():
    X, y = _make_imbalanced_df()
    r = Resampler(protected_attributes=["grp"], strategy="undersample", balance_by="group")
    r.fit(X, y)
    X_res, y_res = r.get_resampled_data(X, y)
    counts = X_res["grp"].value_counts()
    assert counts[0] == counts[1]
    # Undersampling shrinks both groups to the smaller (100).
    assert counts[0] == 100


def test_group_label_balances_cells():
    X, y = _make_imbalanced_df()
    r = Resampler(protected_attributes=["grp"], strategy="oversample", balance_by="group_label")
    r.fit(X, y)
    X_res, y_res = r.get_resampled_data(X, y)
    df = X_res.assign(y=y_res)
    cell_sizes = df.groupby(["grp", "y"]).size()
    # All four group x label cells equal after oversampling.
    assert cell_sizes.nunique() == 1


def test_transform_is_feature_passthrough():
    X, y = _make_imbalanced_df()
    r = Resampler(protected_attributes=["grp"])
    Xt = r.fit_transform(X, y)
    assert "grp" not in Xt.columns
    assert list(Xt.columns) == ["score"]
    assert len(Xt) == len(X)  # transform preserves rows


def test_reproducible_with_seed():
    X, y = _make_imbalanced_df()
    a = Resampler(protected_attributes=["grp"], random_state=7)
    b = Resampler(protected_attributes=["grp"], random_state=7)
    a.fit(X, y)
    b.fit(X, y)
    Xa, ya = a.get_resampled_data(X, y)
    Xb, yb = b.get_resampled_data(X, y)
    pd.testing.assert_frame_equal(Xa, Xb)
    np.testing.assert_array_equal(ya, yb)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        Resampler(protected_attributes=["grp"], strategy="smote")
    with pytest.raises(ValueError):
        Resampler(protected_attributes=["grp"], balance_by="label")


def test_missing_group_label_cell_does_not_crash():
    """A totally-absent group x label cell can't be synthesized by random
    oversampling (needs SMOTE); the existing cells are still balanced and no
    error is raised."""
    rng = np.random.default_rng(0)
    g0 = pd.DataFrame({"grp": 0, "score": rng.normal(size=50), "y": 1})  # only label 1
    g1 = pd.DataFrame(
        {"grp": 1, "score": rng.normal(size=200), "y": (rng.random(200) < 0.5).astype(int)}
    )
    df = pd.concat([g0, g1], ignore_index=True)
    r = Resampler(protected_attributes=["grp"], strategy="oversample", balance_by="group_label")
    r.fit(df.drop(columns="y"), df["y"].values)
    X_res, y_res = r.get_resampled_data(df.drop(columns="y"), df["y"].values)
    cells = pd.Series(y_res).groupby(X_res["grp"].values).value_counts()
    # The three present cells are equalized; the absent (0,0) cell stays absent.
    assert cells.nunique() == 1
    assert len(X_res) == len(y_res)


def test_intersectional_two_attributes():
    """Two protected attributes balance the intersectional (race x gender) cells."""
    rng = np.random.default_rng(2)
    rows = []
    # Deliberately imbalanced across the 4 intersections.
    for race, gender, k in [(0, 0, 40), (0, 1, 120), (1, 0, 80), (1, 1, 200)]:
        rows.append(pd.DataFrame({"race": race, "gender": gender, "score": rng.normal(size=k)}))
    df = pd.concat(rows, ignore_index=True)
    y = (rng.random(len(df)) < 0.5).astype(int)
    r = Resampler(
        protected_attributes=["race", "gender"], strategy="oversample", balance_by="group"
    )
    r.fit(df, y)
    X_res, y_res = r.get_resampled_data(df, y)
    cells = X_res.groupby(["race", "gender"]).size()
    assert cells.nunique() == 1  # all four intersections equal
    assert cells.iloc[0] == 200  # grown to the largest intersection
    assert len(X_res) == len(y_res)


def test_undersample_to_tiny_cell():
    rng = np.random.default_rng(1)
    Xi = pd.concat(
        [
            pd.DataFrame({"grp": 0, "score": rng.normal(size=20)}),
            pd.DataFrame({"grp": 1, "score": rng.normal(size=300)}),
        ],
        ignore_index=True,
    )
    yi = (rng.random(320) < 0.5).astype(int)
    r = Resampler(protected_attributes=["grp"], strategy="undersample", balance_by="group")
    r.fit(Xi, yi)
    X_res, _ = r.get_resampled_data(Xi, yi)
    counts = X_res["grp"].value_counts()
    assert counts[0] == counts[1] == 20
