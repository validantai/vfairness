"""Tests for the spec-v2 classification-family metrics that back the corrected
profile primaries: disparate_impact_ratio (US hiring/lending independence),
conditional_demographic_disparity (EU-hiring), negative_predictive_value_difference
(CJ sufficiency diagnostic), auroc_parity (EU-healthcare validity gate)."""

import numpy as np

from vfairness import (
    auroc_parity,
    conditional_demographic_disparity,
    disparate_impact_ratio,
    negative_predictive_value_difference,
)


def test_disparate_impact_ratio_flags_four_fifths_violation():
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.choice(["M", "F"], n)
    y_pred = np.where(a == "M", rng.uniform(0, 1, n) < 0.6, rng.uniform(0, 1, n) < 0.4).astype(int)
    y_true = (rng.uniform(0, 1, n) < 0.5).astype(int)
    ratio = disparate_impact_ratio(y_true, y_pred, a)
    assert ratio < 0.8  # 40/60 ~ 0.67, below the four-fifths floor
    # parity => ratio near 1
    y_bal = (rng.uniform(0, 1, n) < 0.5).astype(int)
    assert disparate_impact_ratio(y_true, y_bal, a) > 0.85


def test_conditional_demographic_disparity_is_zero_when_explained_by_stratum():
    rng = np.random.default_rng(2)
    n = 5000
    dept = rng.choice(["X", "Y"], n)
    # selection depends ONLY on dept; group is correlated with dept (marginal gap exists)
    y_pred = np.where(dept == "X", rng.uniform(0, 1, n) < 0.7, rng.uniform(0, 1, n) < 0.3).astype(
        int
    )
    a = np.where(
        dept == "X",
        np.where(rng.uniform(0, 1, n) < 0.8, "M", "F"),
        np.where(rng.uniform(0, 1, n) < 0.8, "F", "M"),
    )
    cdd_explained = conditional_demographic_disparity(y_pred, a, dept)
    assert cdd_explained < 0.06  # the gap is legitimately explained by department

    # within each department, favour M -> unexplained conditional disparity
    y_biased = np.where(a == "M", rng.uniform(0, 1, n) < 0.6, rng.uniform(0, 1, n) < 0.3).astype(
        int
    )
    cdd_unexplained = conditional_demographic_disparity(y_biased, a, dept)
    assert cdd_unexplained > 0.2


def test_negative_predictive_value_difference_detects_gap():
    rng = np.random.default_rng(3)
    n = 3000
    a = rng.choice(["A", "B"], n)
    y_true = (rng.uniform(0, 1, n) < 0.5).astype(int)
    # group B: negative predictions are much less trustworthy (more false negatives)
    y_pred = y_true.copy()
    flip = (a == "B") & (y_true == 1) & (rng.uniform(0, 1, n) < 0.6)
    y_pred[flip] = 0
    gap = negative_predictive_value_difference(y_true, y_pred, a)
    assert gap > 0.1


def test_auroc_parity_flags_weaker_group_discrimination():
    rng = np.random.default_rng(4)
    n = 4000
    a = rng.choice(["M", "F"], n)
    y_true = (rng.uniform(0, 1, n) < 0.5).astype(int)
    # M: informative score; F: pure noise -> AUROC gap
    score = np.where(a == "M", y_true * 0.6 + rng.uniform(0, 0.4, n), rng.uniform(0, 1, n))
    assert auroc_parity(y_true, score, a) > 0.2
    # equally informative for both -> small gap
    good = y_true * 0.6 + rng.uniform(0, 0.4, n)
    assert auroc_parity(y_true, good, a) < 0.1


def test_metrics_degrade_gracefully_on_single_group():
    # A single protected group cannot express a BETWEEN-group disparity, so the difference
    # metrics return NaN (insufficient evidence), NOT 0.0 ("perfectly fair"). Collapsing a
    # no-evidence case to 0.0 would let the seal certify fairness it never measured; NaN routes
    # it to insufficient_evidence. The RATIO metrics are held to the same standard: this
    # asserted disparate_impact_ratio == 1.0 as a "documented graceful" value, but 1.0 is
    # the four-fifths-rule reading for NO ADVERSE IMPACT, i.e. a legal all-clear issued for
    # an adverse-impact test that never ran. It is NaN now, like every other unmeasurable
    # comparison here.
    a = np.array(["A"] * 200)
    y_true = np.array([0, 1] * 100)
    y_pred = np.array([0, 1] * 100)
    score = np.linspace(0, 1, 200)
    assert np.isnan(disparate_impact_ratio(y_true, y_pred, a))
    assert np.isnan(negative_predictive_value_difference(y_true, y_pred, a))
    assert np.isnan(conditional_demographic_disparity(y_pred, a, np.array(["X", "Y"] * 100)))
    assert np.isnan(auroc_parity(y_true, score, a))
