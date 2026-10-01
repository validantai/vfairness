"""Regression tests pinning the Wave-6 reweighting fix.

Finding (HIGH): ReweightingTransformer ignored the label: weights were
constant per protected group, which is not Kamiran & Calders (2012)
reweighing and mathematically cannot move demographic parity (a
group-constant weight cancels in every group-conditional mean).

Fixed: method='target_parity' now computes the K&C label-conditional
weights W(g, c) = P(group=g) * P(y=c) / P(group=g, y=c) from the observed
joint distribution over the intersectional group key, and requires y in
both fit() and get_sample_weights().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from vfairness.preprocessing.feature_engineering import ReweightingTransformer


def _biased_2x2_frame():
    """Hand-derivable 2x2 example.

    Group A: 40 rows, 25 with y=1, 15 with y=0.
    Group B: 20 rows,  5 with y=1, 15 with y=0.
    n=60, n_A=40, n_B=20, n_{y=1}=30, n_{y=0}=30.

    Kamiran & Calders weights W(g, c) = n_g * n_c / (n * n_gc):
        W(A, 1) = 40*30 / (60*25) = 4/5
        W(A, 0) = 40*30 / (60*15) = 4/3
        W(B, 1) = 20*30 / (60* 5) = 2
        W(B, 0) = 20*30 / (60*15) = 2/3
    Check: 25*(4/5) + 15*(4/3) + 5*2 + 15*(2/3) = 20+20+10+10 = 60 = n.
    """
    grp = np.array(["A"] * 40 + ["B"] * 20)
    y = np.concatenate([np.repeat([1, 0], [25, 15]), np.repeat([1, 0], [5, 15])])
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"grp": grp, "x1": rng.normal(size=60)})
    return df, grp, y


def _biased_training_data(n_a: int = 600, n_b: int = 400):
    """Strongly biased synthetic data: P(y=1|A)=0.70, P(y=1|B)=0.20,
    with the group visible to the learner (the recorded reproduction)."""
    grp = np.array(["A"] * n_a + ["B"] * n_b)
    y = np.concatenate(
        [
            np.repeat([1, 0], [int(0.70 * n_a), n_a - int(0.70 * n_a)]),
            np.repeat([1, 0], [int(0.20 * n_b), n_b - int(0.20 * n_b)]),
        ]
    )
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n_a + n_b) + 0.5 * y
    df = pd.DataFrame({"grp": grp, "x1": x1})
    X_feat = np.column_stack([x1, (grp == "A").astype(float)])
    return df, grp, y, X_feat


def _dp_gap(X_feat, y, grp, weights=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_feat, y, sample_weight=weights)
    pred = clf.predict(X_feat)
    return abs(pred[grp == "A"].mean() - pred[grp == "B"].mean())


class TestKamiranCaldersWeights:
    """Negative case first: the exact scenario that used to produce
    group-constant weights must now produce the four K&C weights exactly."""

    def test_kc_weights_exact_on_2x2(self):
        df, grp, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)

        expected = {
            ("A", 1): 4 / 5,
            ("A", 0): 4 / 3,
            ("B", 1): 2.0,
            ("B", 0): 2 / 3,
        }
        for (g, c), w_gc in expected.items():
            cell = w[(grp == g) & (y == c)]
            assert np.allclose(cell, w_gc, atol=1e-12), (
                f"W({g},{c}) should be {w_gc}, got {sorted(set(cell))}"
            )

    def test_weights_are_label_conditional_not_group_constant(self):
        """The old bug: identical weights for y=0 and y=1 within a group."""
        df, grp, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)
        for g in ("A", "B"):
            w1 = w[(grp == g) & (y == 1)][0]
            w0 = w[(grp == g) & (y == 0)][0]
            assert w1 != pytest.approx(w0), f"group {g}: weight must depend on the label"

    def test_weighted_group_label_rates_equalize(self):
        """K&C's defining property: weighted P(y=1|g) is equal across groups
        (and equals the overall weighted base rate)."""
        df, grp, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)
        rates = [np.average(y[grp == g], weights=w[grp == g]) for g in ("A", "B")]
        assert rates[0] == pytest.approx(rates[1], abs=1e-12)

    def test_weights_sum_to_n_samples(self):
        df, grp, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)
        assert w.sum() == pytest.approx(len(df), abs=1e-9)

    def test_methods_no_longer_algebraically_identical(self):
        """Pre-fix, 'target_parity' and 'inverse_frequency' returned identical
        weights on any data with within-group label imbalance."""
        df, grp, y = _biased_2x2_frame()
        tp = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        tp.fit(df, y)
        inv = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        inv.fit(df, y)
        assert not np.allclose(tp.get_sample_weights(df, y), inv.get_sample_weights(df))


class TestEndToEndDemographicParity:
    def test_reduces_dp_gap_of_label_aware_learner(self):
        """The recorded reproduction: a LogisticRegression that sees the group.
        Pre-fix the transformer weights left the DP gap unchanged
        (0.9408 -> 0.9425); the K&C weights must reduce it substantially."""
        df, grp, y, X_feat = _biased_training_data()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)

        gap_unweighted = _dp_gap(X_feat, y, grp)
        gap_weighted = _dp_gap(X_feat, y, grp, weights=w)

        assert gap_unweighted > 0.8, "sanity: the unweighted model must be strongly disparate"
        assert gap_weighted < 0.5 * gap_unweighted, (
            f"K&C weights must at least halve the DP gap: {gap_unweighted:.4f} -> {gap_weighted:.4f}"
        )
        assert gap_weighted < 0.2, f"expected a near-parity model, got gap {gap_weighted:.4f}"


class TestIntersectionalGroups:
    def test_target_parity_equalizes_joint_cells(self):
        """With two protected attributes the K&C weights condition on the
        intersectional cell: weighted P(y=1|cell) equal across all cells."""
        rng = np.random.default_rng(3)
        # Build cells explicitly: (race, gender, n, n_pos)
        cells = [
            ("A", "M", 300, 240),
            ("A", "F", 200, 100),
            ("B", "M", 150, 45),
            ("B", "F", 350, 35),
        ]
        race = np.concatenate([np.repeat(r, n) for r, _, n, _ in cells])
        gender = np.concatenate([np.repeat(g, n) for _, g, n, _ in cells])
        y = np.concatenate([np.repeat([1, 0], [p, n - p]) for _, _, n, p in cells])
        df = pd.DataFrame({"race": race, "gender": gender, "x1": rng.normal(size=len(y))})

        t = ReweightingTransformer(protected_attributes=["race", "gender"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)

        rates = []
        for r, g, _, _ in cells:
            m = (race == r) & (gender == g)
            rates.append(np.average(y[m], weights=w[m]))
        assert max(rates) - min(rates) < 1e-12, f"joint-cell weighted rates differ: {rates}"


class TestTargetParityRequiresLabel:
    """target_parity without y must refuse loudly, never fit silently
    ineffective weights (the silence was the defect)."""

    def test_fit_without_y_raises(self):
        df, _, _ = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        with pytest.raises(ValueError, match="target_parity"):
            t.fit(df)

    def test_get_sample_weights_without_y_raises(self):
        df, _, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        with pytest.raises(ValueError, match="target_parity"):
            t.get_sample_weights(df)

    def test_mismatched_y_length_raises(self):
        df, _, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        with pytest.raises(ValueError, match="length"):
            t.fit(df, y[:-1])


class TestDoesNotOvercorrect:
    """The group-marginal methods keep their documented semantics."""

    def test_inverse_frequency_unchanged(self):
        """Exact pre-fix values: w_g = n / (n_groups * count_g), label-free,
        no y required anywhere."""
        df, grp, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        t.fit(df)  # no y
        w = t.get_sample_weights(df)  # no y
        # n=60, 2 groups: w_A = 60/(2*40) = 0.75, w_B = 60/(2*20) = 1.5
        assert np.allclose(w[grp == "A"], 0.75, atol=1e-12)
        assert np.allclose(w[grp == "B"], 1.5, atol=1e-12)
        assert w.sum() == pytest.approx(60, abs=1e-9)

    def test_custom_unchanged(self):
        """Complete target_distribution still produces the requested shares."""
        grp = np.array(["A"] * 8 + ["B"] * 2)
        df = pd.DataFrame({"grp": grp, "x1": np.arange(10.0)})
        t = ReweightingTransformer(
            protected_attributes=["grp"],
            method="custom",
            target_distribution={"A": 0.5, "B": 0.5},
        )
        t.fit(df)
        w = t.get_sample_weights(df)
        share_a = w[grp == "A"].sum() / w.sum()
        assert share_a == pytest.approx(0.5, abs=1e-9)

    def test_target_parity_on_balanced_data_is_near_uniform(self):
        """When the data already satisfies parity (equal label rates per
        group), the K&C weights are all exactly 1: no overcorrection."""
        grp = np.array(["A"] * 40 + ["B"] * 20)
        y = np.concatenate([np.repeat([1, 0], [20, 20]), np.repeat([1, 0], [10, 10])])
        df = pd.DataFrame({"grp": grp, "x1": np.zeros(60)})
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        w = t.get_sample_weights(df, y)
        assert np.allclose(w, 1.0, atol=1e-12)

    def test_transform_still_returns_features_unchanged(self):
        df, _, y = _biased_2x2_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        out = t.transform(df)
        assert list(out.columns) == ["x1"]
        assert np.allclose(out["x1"].to_numpy(), df["x1"].to_numpy())
