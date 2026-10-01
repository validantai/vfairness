"""Behavioural tests for `preprocessing.feature_engineering.data_balancing`.

This module exports NINE names through `vfairness.__all__` (SyntheticResampler,
SMOTEResampler, ADASYNResampler, TomekResampler, PropensityScoreWeighter,
InversePropensityWeighter, CounterfactualAugmenter, and friends) and had ZERO
tests: no file under tests/ so much as mentioned it. It is 352 statements of
shipped, publicly exported behaviour that no environment had ever executed.

Writing these found a real defect, which is the point of writing them rather
than adding coverage for its own sake. See
`test_an_absent_group_label_cell_is_reported_not_hidden` below.

These are behavioural, not line-coverage: each one pins a property a caller
relies on, and each would fail if the maths quietly changed under it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.data_balancing import (
    ADASYNResampler,
    CounterfactualAugmenter,
    InversePropensityWeighter,
    PropensityScoreWeighter,
    SMOTEResampler,
    propensity_weights,
)


def _imbalanced(n_a: int = 250, n_b: int = 50, seed: int = 0):
    """Both groups carry both labels, so every (group, label) cell is populated."""
    rng = np.random.default_rng(seed)
    groups = np.array(["A"] * n_a + ["B"] * n_b)
    n = n_a + n_b
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "grp": groups})
    # Alternate labels within each group so no cell is empty.
    y = np.concatenate([np.tile([0, 1], n_a // 2), np.tile([0, 1], n_b // 2)])
    return X, y


# ---------------------------------------------------------------------------
# The defect these tests found
# ---------------------------------------------------------------------------


def test_an_absent_group_label_cell_is_reported_not_hidden():
    """A balancer that could not balance must say so.

    Resampling works on (group x label) CELLS. A cell with no rows cannot be
    synthesised from nothing, so when a group never appears with one of the
    labels, the group totals stay uneven however well every populated cell is
    filled. That is correct arithmetic, and indistinguishable from a failed
    balance unless something reports it.

    Measured 2026-09-07, before the warning existed: 200 rows in group A and 2
    in group B, B carrying label 1 only, returned A=218 B=109 with
    `fit_result.warnings == []`.
    """
    rng = np.random.default_rng(0)
    n = 202
    groups = np.array(["A"] * 200 + ["B"] * 2)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "grp": groups})
    y = np.concatenate([np.tile([0, 1], 100), np.array([1, 1])])  # B has no label 0

    r = SMOTEResampler(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, _ = r.get_resampled_data(X, y)

    counts = Counter(X_res["grp"])
    assert len(set(counts.values())) > 1, "fixture no longer produces an uneven result"

    warned = " ".join(r.fit_result.warnings)
    assert warned, (
        f"the resampler returned UNBALANCED groups ({dict(counts)}) and reported nothing at all"
    )
    assert "not group-balanced" in warned
    assert "label=0" in warned, "the warning does not name the cell that was empty"


def test_a_fully_populated_run_warns_about_nothing():
    """Over-correction control. A warning on every run is a warning on none."""
    X, y = _imbalanced()
    r = SMOTEResampler(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, _ = r.get_resampled_data(X, y)

    counts = Counter(X_res["grp"])
    assert len(set(counts.values())) == 1, f"expected balanced groups, got {dict(counts)}"
    assert r.fit_result.warnings == [], f"unexpected warning: {r.fit_result.warnings}"


# ---------------------------------------------------------------------------
# The properties a caller actually relies on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_resampling_balances_the_protected_groups(cls):
    X, y = _imbalanced()
    before = Counter(X["grp"])
    assert before["A"] != before["B"], "fixture is not imbalanced"

    r = cls(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, y_res = r.get_resampled_data(X, y)

    after = Counter(X_res["grp"])
    assert len(set(after.values())) == 1, f"{cls.__name__} left groups uneven: {dict(after)}"
    assert after["B"] > before["B"], "the minority group did not grow"


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_labels_stay_aligned_with_rows(cls):
    """A row/label length mismatch silently corrupts every downstream metric."""
    X, y = _imbalanced()
    r = cls(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, y_res = r.get_resampled_data(X, y)

    assert len(X_res) == len(y_res), "X and y came back different lengths"
    assert set(np.unique(y_res)).issubset(set(np.unique(y))), "a new label was invented"


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_no_group_is_dropped(cls):
    """Losing a group entirely would remove it from every later fairness number."""
    X, y = _imbalanced()
    r = cls(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, _ = r.get_resampled_data(X, y)

    assert set(X_res["grp"]) == set(X["grp"]), "a protected group disappeared"


def test_every_input_column_survives_resampling():
    X, y = _imbalanced()
    r = SMOTEResampler(protected_attributes=["grp"])
    r.fit(X, y)
    X_res, _ = r.get_resampled_data(X, y)
    assert list(X_res.columns) == list(X.columns)


def test_the_same_random_state_gives_the_same_data_twice():
    """Without this, a fairness result is not reproducible from its inputs."""
    X, y = _imbalanced()
    out = []
    for _ in range(2):
        r = SMOTEResampler(protected_attributes=["grp"], random_state=7)
        r.fit(X, y)
        X_res, y_res = r.get_resampled_data(X, y)
        out.append((X_res.reset_index(drop=True), np.asarray(y_res)))
    pd.testing.assert_frame_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


# ---------------------------------------------------------------------------
# Propensity weighting
# ---------------------------------------------------------------------------


def test_propensity_weights_are_finite_positive_and_not_all_identical():
    """All-equal weights are a no-op wearing the costume of a reweighting."""
    X, y = _imbalanced()
    result = propensity_weights(X, protected_attributes=["grp"])

    w = np.asarray(result["sample_weights"], dtype=float)
    assert len(w) == len(X), "one weight per row is the whole contract"
    assert np.all(np.isfinite(w)), "a non-finite weight poisons every weighted metric"
    assert np.all(w > 0), "a zero weight silently deletes a row"
    assert not np.allclose(w, w[0]), (
        "every weight is identical, so the reweighting did nothing while reporting that it had"
    )


@pytest.mark.parametrize("cls", [PropensityScoreWeighter, InversePropensityWeighter])
def test_the_weighter_classes_produce_one_weight_per_row(cls):
    X, y = _imbalanced()
    w = cls(protected_attributes=["grp"]).fit(X, y).get_sample_weights(X)
    w = np.asarray(w, dtype=float)
    assert len(w) == len(X)
    assert np.all(np.isfinite(w)) and np.all(w > 0)


def test_propensity_scores_stay_inside_the_clip_bounds():
    """Outside (0,1) the inverse blows up to infinity or flips sign."""
    X, y = _imbalanced()
    result = propensity_weights(X, protected_attributes=["grp"], clip=(0.05, 0.95))
    scores = np.asarray(result["propensity_scores"], dtype=float)
    assert scores.min() >= 0.05 - 1e-9 and scores.max() <= 0.95 + 1e-9


# ---------------------------------------------------------------------------
# Counterfactual augmentation
# ---------------------------------------------------------------------------


def test_counterfactual_augmentation_actually_flips_the_attribute():
    """If it never flips, it is duplication with a fairness-sounding name."""
    X, y = _imbalanced()
    before = Counter(X["grp"])

    aug = CounterfactualAugmenter(protected_attributes=["grp"])
    aug.fit(X, y)
    X_aug, y_aug = aug.get_resampled_data(X, y)

    after = Counter(X_aug["grp"])
    assert len(X_aug) > len(X), "no rows were added"
    assert len(X_aug) == len(y_aug), "X and y came back different lengths"
    assert after["B"] > before["B"], "the minority group gained no counterfactual rows"
    assert set(after) == set(before), "augmentation invented or dropped a group"
