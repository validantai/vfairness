"""
Tests for DisparateImpactRemover (Feldman et al., 2015 geometric repair).

Verifies that repairing a group-correlated numeric feature:
  - reduces the between-group mean gap and feature/group correlation,
  - is a no-op at repair_level=0.0,
  - preserves within-group ordering (rank correlation == 1.0),
  - leaves non-numeric features and the protected column untouched.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness import DisparateImpactRemover


def _make_biased_df(n=600, seed=0):
    """Feature `score` is strongly shifted by group; `noise` is group-neutral."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, size=n)
    # group 0 ~ N(0,1), group 1 ~ N(4,1) -> large disparate impact in `score`
    score = rng.normal(loc=4.0 * grp, scale=1.0)
    noise = rng.normal(loc=0.0, scale=1.0, size=n)
    y = (score + noise > np.median(score)).astype(int)
    return pd.DataFrame({"grp": grp, "score": score, "noise": noise, "y": y})


def _group_mean_gap(df, feat, attr="grp"):
    means = df.groupby(attr)[feat].mean()
    return abs(means.iloc[0] - means.iloc[1])


def test_full_repair_reduces_group_gap_and_correlation():
    df = _make_biased_df()
    X = df.drop(columns=["y"])

    remover = DisparateImpactRemover(protected_attributes=["grp"], repair_level=1.0)
    Xt = remover.fit_transform(X)

    # protected column excluded from transformer output (handler re-adds it)
    assert "grp" not in Xt.columns
    assert "score" in Xt.columns

    before_gap = _group_mean_gap(X.assign(), "score")
    after = Xt.assign(grp=df["grp"].values)
    after_gap = _group_mean_gap(after, "score")

    # Full repair should collapse the ~4.0 group-mean gap to near zero.
    assert before_gap > 3.0
    assert after_gap < 0.5 * before_gap
    assert after_gap < 0.75

    # |corr(score, grp)| should drop substantially.
    corr_before = abs(np.corrcoef(X["score"], df["grp"])[0, 1])
    corr_after = abs(np.corrcoef(after["score"], df["grp"])[0, 1])
    assert corr_after < corr_before
    assert corr_after < 0.3


def test_repair_level_zero_is_noop():
    df = _make_biased_df()
    X = df.drop(columns=["y"])
    remover = DisparateImpactRemover(protected_attributes=["grp"], repair_level=0.0)
    Xt = remover.fit_transform(X)
    np.testing.assert_allclose(Xt["score"].values, X["score"].values, rtol=1e-9, atol=1e-9)


def test_within_group_ordering_preserved():
    df = _make_biased_df()
    X = df.drop(columns=["y"])
    remover = DisparateImpactRemover(protected_attributes=["grp"], repair_level=1.0)
    Xt = remover.fit_transform(X)
    after = Xt.assign(grp=df["grp"].values)
    # Geometric repair preserves rank within each group.
    for g in [0, 1]:
        orig = X.loc[df["grp"] == g, "score"].rank().values
        rep = after.loc[after["grp"] == g, "score"].rank().values
        assert np.corrcoef(orig, rep)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_fit_result_metadata():
    df = _make_biased_df()
    X = df.drop(columns=["y"])
    remover = DisparateImpactRemover(protected_attributes=["grp"], repair_level=1.0)
    remover.fit(X)
    assert remover.is_fitted
    assert remover.fit_result.method == "disparate_impact_removal"
    assert "score" in remover.fit_result.features_modified
    assert "noise" in remover.fit_result.features_modified
    assert remover.fit_result.fit_metrics["repair_level"] == 1.0
    assert remover.fit_result.fit_metrics["n_groups"] == 2


def test_invalid_repair_level_raises():
    with pytest.raises(ValueError):
        DisparateImpactRemover(protected_attributes=["grp"], repair_level=1.5)


def test_handles_discrete_feature_with_ties():
    """Tie-heavy integer feature must not break the quantile mapping."""
    rng = np.random.default_rng(1)
    grp = rng.integers(0, 2, size=400)
    # discrete feature with heavy ties, shifted by group
    bins = rng.integers(0, 3, size=400) + 3 * grp
    X = pd.DataFrame({"grp": grp, "bins": bins.astype(float)})
    remover = DisparateImpactRemover(protected_attributes=["grp"], repair_level=1.0)
    Xt = remover.fit_transform(X)
    assert np.isfinite(Xt["bins"].values).all()
    after = Xt.assign(grp=grp)
    assert _group_mean_gap(after, "bins") < _group_mean_gap(X, "bins")


def test_intersectional_repair_reduces_both_attributes():
    """With two protected attributes, full repair removes the score gap across the
    JOINT (race x gender) groups, not just the first attribute."""
    rng = np.random.default_rng(0)
    n = 800
    race = rng.integers(0, 2, size=n)
    gender = rng.integers(0, 2, size=n)
    # score depends on BOTH attributes.
    score = rng.normal(loc=2.0 * race + 2.0 * gender, scale=1.0)
    X = pd.DataFrame({"race": race, "gender": gender, "score": score})

    remover = DisparateImpactRemover(protected_attributes=["race", "gender"], repair_level=1.0)
    Xt = remover.fit_transform(X)
    after = Xt.assign(race=race, gender=gender)

    # Correlation of repaired score with EACH attribute should drop substantially.
    for attr_vals in (race, gender):
        corr_before = abs(np.corrcoef(X["score"], attr_vals)[0, 1])
        corr_after = abs(np.corrcoef(after["score"], attr_vals)[0, 1])
        assert corr_after < corr_before
        assert corr_after < 0.25
    assert remover.fit_result.fit_metrics["n_groups"] == 4
