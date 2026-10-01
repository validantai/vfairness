"""Audit-3 (preproc-encoding): categorical protected attributes must be one-hot
encoded before residualization / suppression.

Defect (transformers.py:613): CorrelationReducer and FeatureSuppressor label-
encoded a categorical protected attribute into ONE ordinal column, so
residualizing/suppressing removed only the linear-in-that-ordinal component and
left most of a nominal attribute's group-mean structure intact (while falsely
reporting correlation_after ~ 0, because success was measured with Pearson on
the same arbitrary codes). Binary attributes were unaffected (the ordinal code
equals the dummy).

Root-cause fix: one-hot encode categorical protected attributes (drop-first)
before residualization/decorrelation, and measure feature/attribute association
with the correlation ratio eta (numeric feature vs categorical attribute) rather
than Pearson on integer codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.transformers import (
    CorrelationReducer,
    FeatureSuppressor,
)


def _eta_sq(feature: np.ndarray, groups: np.ndarray) -> float:
    """True correlation ratio eta^2 of a numeric feature vs a categorical attr."""
    x = np.asarray(feature, dtype=float)
    grand = x.mean()
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    g = np.asarray(groups)
    for level in pd.unique(g):
        xg = x[g == level]
        ss_between += len(xg) * (xg.mean() - grand) ** 2
    return ss_between / ss_total


def _nominal_frame(seed: int = 42, per_group: int = 300):
    """3-category nominal attribute with a NON-monotonic group-mean pattern
    (A and C high, B low), the case Pearson-on-codes cannot see."""
    rng = np.random.default_rng(seed)
    race = np.array((["A"] * per_group) + (["B"] * per_group) + (["C"] * per_group))
    base = np.where(race == "B", 0.0, 10.0)
    feat = base + rng.normal(0.0, 1.0, len(race))
    return pd.DataFrame({"race": race, "feat": feat}), race


# ========================================================================= #
# Defect case FIRST: the exact scenario that used to give the wrong answer.
# ========================================================================= #


def test_residualize_removes_nominal_3category_association():
    """The protected-attribute association is substantially removed after
    residualize (not just the linear-ordinal part). Old code left eta^2 = 0.96.
    """
    df, race = _nominal_frame()
    before = _eta_sq(df["feat"].to_numpy(), race)
    assert before > 0.9  # the feature nearly determines the group

    reducer = CorrelationReducer(
        protected_attributes=["race"], method="residualize", preserve_variance=False
    )
    out = reducer.fit_transform(df)
    after = _eta_sq(out["feat"].to_numpy(), race)
    assert after < 1e-6, f"residualize left eta^2={after} (old bug left ~0.96)"


def test_partial_is_refused_because_it_never_existed():
    """Audit-6 lane 2 (2026-09-09): this test used to run method='partial' and
    assert that it removed the association. It did, because 'partial' was
    byte-identical to 'residualize' (its only distinct step set every scale
    factor to 1.0), so the pin was asserting residualize's behaviour under a
    second name. The honest behaviour is a refusal at construction; the
    residualize assertion above already covers what 'partial' actually did."""
    with pytest.raises(NotImplementedError, match="partial"):
        CorrelationReducer(protected_attributes=["race"], method="partial", preserve_variance=False)


def test_decorrelate_removes_nominal_3category_association():
    df, race = _nominal_frame()
    reducer = CorrelationReducer(
        protected_attributes=["race"], method="decorrelate", preserve_variance=False
    )
    out = reducer.fit_transform(df)
    after = _eta_sq(out["feat"].to_numpy(), race)
    assert after < 1e-6, f"decorrelate left eta^2={after} (old bug made it worse, ~0.97)"


def test_correlation_report_is_honest_for_nominal_attribute():
    """correlation_before must SEE the nominal association (eta, not Pearson on
    codes), and correlation_after must reflect the real removal. Old code
    reported correlation_after ~ 0 on an UNTOUCHED association (false success).
    """
    df, race = _nominal_frame()
    reducer = CorrelationReducer(
        protected_attributes=["race"], method="residualize", preserve_variance=False
    )
    reducer.fit(df)
    res = reducer.fit_result
    assert res is not None
    assert res.correlation_before["race"] > 0.9  # eta sees the nominal association
    assert res.correlation_after["race"] < 0.05  # honestly ~0 after real removal


def test_feature_suppressor_removes_nominal_associated_feature():
    """A feature with eta^2 > 0.9 against a nominal attribute must be identified
    and removed. Old code (Pearson on codes) identified nothing."""
    df, race = _nominal_frame()
    suppressor = FeatureSuppressor(
        protected_attributes=["race"], strategy="remove", correlation_threshold=0.3
    )
    out = suppressor.fit_transform(df)
    assert "feat" in suppressor._identified_features
    assert "feat" not in list(out.columns)


# ========================================================================= #
# Does-not-overcorrect cases.
# ========================================================================= #


def test_feature_suppressor_keeps_independent_feature():
    """A feature unrelated to the protected attribute must NOT be suppressed."""
    df, race = _nominal_frame()
    rng = np.random.default_rng(7)
    df["indep"] = rng.normal(0.0, 1.0, len(df))  # independent of race

    suppressor = FeatureSuppressor(
        protected_attributes=["race"], strategy="remove", correlation_threshold=0.3
    )
    out = suppressor.fit_transform(df)
    assert "indep" not in suppressor._identified_features
    assert "indep" in list(out.columns)


def test_residualize_preserves_within_group_signal():
    """Residualization removes the group-mean structure but must keep the
    within-group predictive signal (it must not zero out useful variation)."""
    rng = np.random.default_rng(11)
    per_group = 400
    race = np.array((["A"] * per_group) + (["B"] * per_group) + (["C"] * per_group))
    group_component = np.where(race == "B", 0.0, 10.0)
    signal = rng.normal(0.0, 1.0, len(race))  # unrelated to race
    feat = group_component + 3.0 * signal + rng.normal(0.0, 0.2, len(race))
    df = pd.DataFrame({"race": race, "feat": feat})

    reducer = CorrelationReducer(
        protected_attributes=["race"], method="residualize", preserve_variance=False
    )
    out = reducer.fit_transform(df)
    residual = out["feat"].to_numpy()

    # group structure gone ...
    assert _eta_sq(residual, race) < 1e-6
    # ... but the within-group signal survives
    kept = abs(np.corrcoef(residual, signal)[0, 1])
    assert kept > 0.8, f"residualize destroyed the within-group signal (corr={kept})"


def test_binary_categorical_attribute_still_decorrelated():
    """Binary attributes were already handled correctly; confirm no regression."""
    rng = np.random.default_rng(3)
    per_group = 450
    race = np.array((["A"] * per_group) + (["B"] * per_group))
    feat = np.where(race == "B", 0.0, 10.0) + rng.normal(0.0, 1.0, len(race))
    df = pd.DataFrame({"race": race, "feat": feat})

    reducer = CorrelationReducer(
        protected_attributes=["race"], method="residualize", preserve_variance=False
    )
    out = reducer.fit_transform(df)
    assert _eta_sq(out["feat"].to_numpy(), race) < 1e-6


def test_numeric_protected_attribute_still_uses_pearson():
    """A numeric protected attribute must still be linearly residualized (the
    one-hot path is only for categorical attributes)."""
    rng = np.random.default_rng(5)
    n = 900
    age = rng.normal(40.0, 10.0, n)
    feat = 2.5 * age + rng.normal(0.0, 1.0, n)  # strongly linear in age
    df = pd.DataFrame({"age": age, "feat": feat})

    before = abs(np.corrcoef(df["feat"], df["age"])[0, 1])
    assert before > 0.9

    reducer = CorrelationReducer(
        protected_attributes=["age"], method="residualize", preserve_variance=False
    )
    out = reducer.fit_transform(df)
    after = abs(np.corrcoef(out["feat"].to_numpy(), age)[0, 1])
    assert after < 0.05, f"numeric residualize left Pearson |r|={after}"
