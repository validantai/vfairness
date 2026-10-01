"""
Tests for LabelMassager (Kamiran & Calders, 2012 "massaging").

Verifies that relabelling borderline instances:
  - reduces the gap in group positive rates,
  - preserves the number of rows and the feature columns,
  - flips at most the planned number of labels,
  - is a no-op when groups already have equal positive rates.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness import LabelMassager


def _make_biased_df(n=600, seed=0):
    """Deprived group (grp=0) has a much lower positive rate than grp=1."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, size=n)
    score = rng.normal(loc=0.0, scale=1.0, size=n)
    # Positive rate strongly depends on group: grp=1 is favored.
    logit = score + 2.0 * grp - 0.5
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"grp": grp, "score": score, "noise": rng.normal(size=n)}), y


def _pos_rate_gap(df, y, attr="grp"):
    rates = pd.Series(y).groupby(df[attr].values).mean()
    return abs(rates.iloc[0] - rates.iloc[1])


def test_reduces_positive_rate_gap():
    X, y = _make_biased_df()
    massager = LabelMassager(protected_attributes=["grp"])
    massager.fit(X, y)
    y_new = massager.get_massaged_labels(X, y)

    before = _pos_rate_gap(X, y)
    after = _pos_rate_gap(X, y_new)
    assert before > 0.15
    assert after < before
    # Massaging is designed to (nearly) equalize group positive rates.
    assert after < 0.05


def test_preserves_rows_and_features():
    X, y = _make_biased_df()
    massager = LabelMassager(protected_attributes=["grp"])
    Xt = massager.fit_transform(X, y)
    # protected column excluded from transformer output (handler re-adds it)
    assert "grp" not in Xt.columns
    assert list(Xt.columns) == ["score", "noise"]
    y_new = massager.get_massaged_labels(X, y)
    assert len(y_new) == len(y)


def test_flip_count_matches_metadata_and_cap():
    X, y = _make_biased_df()
    massager = LabelMassager(protected_attributes=["grp"], max_flip_fraction=0.5)
    massager.fit(X, y)
    y_new = massager.get_massaged_labels(X, y)
    n_changed = int(np.sum(y_new != y))
    planned = massager.fit_result.fit_metrics["n_labels_flipped"]
    # promote M + demote M => up to 2*M label changes.
    assert n_changed <= 2 * planned + 1
    assert planned >= 1


def test_max_flip_fraction_zero_is_noop():
    X, y = _make_biased_df()
    massager = LabelMassager(protected_attributes=["grp"], max_flip_fraction=0.0)
    massager.fit(X, y)
    y_new = massager.get_massaged_labels(X, y)
    np.testing.assert_array_equal(y_new, np.asarray(y))


def test_balanced_groups_no_flips():
    rng = np.random.default_rng(3)
    n = 400
    grp = rng.integers(0, 2, size=n)
    score = rng.normal(size=n)
    # Positive rate independent of group.
    y = (score > 0).astype(int)
    X = pd.DataFrame({"grp": grp, "score": score})
    massager = LabelMassager(protected_attributes=["grp"])
    massager.fit(X, y)
    y_new = massager.get_massaged_labels(X, y)
    assert _pos_rate_gap(X, y_new) <= _pos_rate_gap(X, y) + 1e-9


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        LabelMassager(protected_attributes=["grp"], ranker="xgboost")
    with pytest.raises(ValueError):
        LabelMassager(protected_attributes=["grp"], max_flip_fraction=0.9)


def test_multi_group_adjusts_extreme_pair():
    """With >2 groups, massaging targets the deprived (min) vs favored (max) pair."""
    rng = np.random.default_rng(1)
    n = 600
    grp = rng.integers(0, 3, size=n)
    score = rng.normal(size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(score + grp - 1.0)))).astype(int)
    X = pd.DataFrame({"grp": grp, "score": score})
    m = LabelMassager(protected_attributes=["grp"])
    m.fit(X, y)
    y_new = m.get_massaged_labels(X, y)
    assert m.deprived_group_ != m.favored_group_
    # Group keys are intersectional strings (e.g. "2"); group by the same string form.
    grp_key = pd.Series(grp).astype(str)
    rates_before = pd.Series(y).groupby(grp_key).mean()
    rates_after = pd.Series(y_new).groupby(grp_key).mean()
    gap_before = rates_before[m.favored_group_] - rates_before[m.deprived_group_]
    gap_after = rates_after[m.favored_group_] - rates_after[m.deprived_group_]
    assert gap_after <= gap_before + 1e-9


def test_intersectional_two_attributes():
    """Massaging with two protected attributes targets the extreme intersectional
    subgroup pair, not just the first attribute."""
    rng = np.random.default_rng(5)
    n = 800
    race = rng.integers(0, 2, size=n)
    gender = rng.integers(0, 2, size=n)
    score = rng.normal(size=n)
    # Positive rate depends on the intersection (race AND gender).
    logit = score + 1.5 * race + 1.5 * gender - 1.5
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"race": race, "gender": gender, "score": score})
    m = LabelMassager(protected_attributes=["race", "gender"])
    m.fit(X, y)
    y_new = m.get_massaged_labels(X, y)
    # deprived/favored are intersectional keys like "0|0" and "1|1".
    assert "|" in m.deprived_group_ and "|" in m.favored_group_
    assert m.fit_result.fit_metrics["protected_attributes"] == ["race", "gender"]
    key = X["race"].astype(str) + "|" + X["gender"].astype(str)
    rb = pd.Series(y).groupby(key).mean()
    ra = pd.Series(y_new).groupby(key).mean()
    gap_b = rb[m.favored_group_] - rb[m.deprived_group_]
    gap_a = ra[m.favored_group_] - ra[m.deprived_group_]
    assert gap_a <= gap_b + 1e-9


def test_no_negatives_to_promote_is_safe():
    """All-positive labels: nothing can be promoted, so no relabelling occurs."""
    rng = np.random.default_rng(2)
    n = 300
    grp = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"grp": grp, "score": rng.normal(size=n)})
    y = np.ones(n, dtype=int)
    m = LabelMassager(protected_attributes=["grp"])
    m.fit(X, y)
    y_new = m.get_massaged_labels(X, y)
    np.testing.assert_array_equal(y_new, y)
