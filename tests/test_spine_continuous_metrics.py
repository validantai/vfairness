"""Tests for the spec-v2 continuous / decision-utility metrics: pricing_disparity
(lending price), net_benefit_parity (EU-healthcare clinical utility), and
conditional_adverse_impact (hiring-US regression-controlled diagnostic)."""

import numpy as np

from vfairness import (
    conditional_adverse_impact,
    net_benefit_parity,
    pricing_disparity,
)


def test_pricing_disparity_isolates_residual_premium_after_controls():
    rng = np.random.default_rng(5)
    n = 4000
    a = rng.choice(["M", "F"], n)
    risk = rng.uniform(0, 1, n)  # legitimate risk factor
    # price = 5 + 3*risk + 2 (only for F); after controlling for risk, F pays +2
    price = 5 + 3 * risk + np.where(a == "F", 2.0, 0.0) + rng.normal(0, 0.3, n)
    controlled = pricing_disparity(price, a, risk)
    assert 1.8 < controlled < 2.2  # recovers the prohibited-basis premium
    # a price driven ONLY by the legitimate factor leaves ~no residual disparity
    fair_price = 5 + 3 * risk + rng.normal(0, 0.3, n)
    assert pricing_disparity(fair_price, a, risk) < 0.2


def test_net_benefit_parity_flags_clinically_weaker_group():
    rng = np.random.default_rng(6)
    n = 4000
    a = rng.choice(["M", "F"], n)
    y = (rng.uniform(0, 1, n) < 0.4).astype(int)
    # M: informative probability; F: noise -> lower clinical net benefit
    prob = np.where(a == "M", np.clip(y * 0.5 + rng.uniform(0, 0.5, n), 0, 1), rng.uniform(0, 1, n))
    assert net_benefit_parity(y, prob, a, threshold=0.4) > 0.1


def test_conditional_adverse_impact_zero_when_explained_by_covariate():
    rng = np.random.default_rng(7)
    n = 4000
    a = rng.choice(["M", "F"], n)
    risk = rng.uniform(0, 1, n)
    sel_explained = (risk > 0.5).astype(int)  # depends only on the legitimate covariate
    assert conditional_adverse_impact(sel_explained, a, risk) < 0.05
    # favour M beyond what risk explains -> positive residual adverse impact
    sel_biased = ((risk > 0.5) | ((a == "M") & (rng.uniform(0, 1, n) < 0.3))).astype(int)
    assert conditional_adverse_impact(sel_biased, a, risk) > 0.05


def test_continuous_metrics_degrade_on_single_group():
    # A single protected group cannot express a BETWEEN-group disparity, so these return NaN
    # (insufficient evidence), NOT 0.0 ("perfect parity"). A 0.0 here would seal false parity
    # on a metric that could not be measured. Mirrors the classification-metric convention.
    a = np.array(["A"] * 200)
    price = np.linspace(1, 10, 200)
    y = np.array([0, 1] * 100)
    prob = np.linspace(0, 1, 200)
    assert np.isnan(pricing_disparity(price, a))
    assert np.isnan(net_benefit_parity(y, prob, a, threshold=0.5))
    assert np.isnan(conditional_adverse_impact(y, a, prob))
