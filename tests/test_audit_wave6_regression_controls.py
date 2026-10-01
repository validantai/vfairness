"""Wave-6 audit pin: pricing_disparity omitted-group-regressor attenuation.

Finding (HIGH, second-iteration-audit-2026-08-22): regressing the price on the
controls WITHOUT the group indicators and reading the between-group residual-mean
gap under-measures the adjusted disparity by the factor (1 - R^2 of group on
controls) whenever the controls correlate with the protected attribute, i.e. in
the realistic case, biasing toward PASS on the sealed lending price band. The fix
estimates the group effect jointly (price ~ controls + group indicators, the
Frisch-Waugh-Lovell partialling-out estimand, Ross & Yinger 2002) and reports the
max pairwise group-coefficient difference.

The companion half of the finding (conditional_adverse_impact in
classification.py) fell through the wave-6 ownership split and was fixed in the
release pass (2026-08-22); its pins are the TestConditionalAdverseImpact class
at the end of this file (joint-fit average-marginal-effect estimand, the binary
counterpart of the pricing_disparity fix).
"""

import math

import numpy as np
import pytest

from vfairness import pricing_disparity, pricing_disparity_with_ci
from vfairness.evaluation.vfairness_metrics.classification import (
    conditional_adverse_impact,
)


def _correlated_controls_scenario(n=40000, seed=42):
    """The recorded reproduction: control correlated with the group, planted
    adjusted gap of exactly 5.0."""
    rng = np.random.default_rng(seed)
    a01 = rng.integers(0, 2, n)
    controls = 0.8 * a01 + rng.normal(0, 0.5, n)
    price = 5.0 * a01 + 2.0 * controls + rng.normal(0, 1.0, n)
    groups = np.where(a01 == 1, "G1", "G0")
    return price, groups, controls


def test_correlated_controls_recover_planted_gap():
    """NEGATIVE CASE (the exact recorded defect): with corr(group, control) ~ 0.62
    the old residual-gap estimator returned ~3.08 for a true adjusted gap of 5.0
    (analytic attenuation gamma*(1-R^2) = 3.06). The joint estimator must recover
    the planted gap."""
    price, groups, controls = _correlated_controls_scenario()
    gap = pricing_disparity(price, groups, controls)
    assert gap == pytest.approx(5.0, abs=0.15)
    # Explicitly refuse the old attenuated regime.
    assert gap > 4.0


def test_orthogonal_controls_stay_correct():
    """Does-not-overcorrect: with a control independent of the group (zero
    attenuation, the one case the old estimator got right) the joint estimator
    must return the same, correct answer."""
    rng = np.random.default_rng(7)
    n = 40000
    a01 = rng.integers(0, 2, n)
    controls = rng.normal(0, 0.5, n)
    price = 5.0 * a01 + 2.0 * controls + rng.normal(0, 1.0, n)
    groups = np.where(a01 == 1, "G1", "G0")
    assert pricing_disparity(price, groups, controls) == pytest.approx(5.0, abs=0.15)


def test_fair_pricing_with_correlated_controls_reads_near_zero():
    """Does-not-overcorrect: a price driven ONLY by a group-correlated legitimate
    control has zero prohibited-basis effect; the joint estimator must not invent
    one."""
    rng = np.random.default_rng(11)
    n = 40000
    a01 = rng.integers(0, 2, n)
    controls = 0.8 * a01 + rng.normal(0, 0.5, n)
    price = 2.0 * controls + rng.normal(0, 1.0, n)
    groups = np.where(a01 == 1, "G1", "G0")
    assert abs(pricing_disparity(price, groups, controls)) < 0.15


def test_no_controls_reduces_to_raw_mean_difference():
    """Contract: with controls=None the metric is the raw mean-price difference."""
    rng = np.random.default_rng(3)
    n = 2000
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    price = np.where(groups == "A", 6.0, 4.0) + rng.normal(0, 0.5, n)
    raw = abs(price[groups == "A"].mean() - price[groups == "B"].mean())
    assert pricing_disparity(price, groups) == pytest.approx(raw, abs=1e-9)


def test_single_group_still_nan():
    """Contract preserved: fewer than two assessable groups -> NaN, never 0.0."""
    price = np.linspace(1, 10, 200)
    groups = np.array(["A"] * 200)
    assert np.isnan(pricing_disparity(price, groups))
    assert np.isnan(pricing_disparity(price, groups, np.linspace(0, 1, 200)))


def test_perfect_proxy_control_is_unidentified_nan():
    """A control that IS the protected attribute makes the adjusted group effect
    unidentified. The old estimator silently returned ~0.0 (false PASS); the joint
    estimator must refuse with NaN and warn, never emit an arbitrary split."""
    rng = np.random.default_rng(19)
    n = 4000
    a01 = rng.integers(0, 2, n)
    price = 5.0 * a01 + rng.normal(0, 1.0, n)
    groups = np.where(a01 == 1, "G1", "G0")
    with pytest.warns(UserWarning, match="not identified"):
        result = pricing_disparity(price, groups, a01.astype(float))
    assert np.isnan(result)


def test_small_group_rows_do_not_contaminate_and_are_not_reported():
    """A group below min_group_size gets its own indicator (so its direct effect
    cannot leak into the control coefficients) but is excluded from the reported
    max pairwise difference."""
    rng = np.random.default_rng(23)
    n = 20000
    a01 = rng.integers(0, 2, n)
    controls = 0.8 * a01 + rng.normal(0, 0.5, n)
    price = 5.0 * a01 + 2.0 * controls + rng.normal(0, 1.0, n)
    groups = np.where(a01 == 1, "G1", "G0").astype(object)
    # 10 rows of a tiny group with a huge price effect; must not appear in the gap.
    groups[:10] = "TINY"
    price[:10] += 100.0
    gap = pricing_disparity(price, groups, controls, min_group_size=30)
    assert gap == pytest.approx(5.0, abs=0.2)


def test_three_groups_max_pairwise_effect():
    """Max pairwise adjusted-effect difference across 3+ groups: planted effects
    0 / 2 / 5 with a group-correlated control -> gap 5.0."""
    rng = np.random.default_rng(29)
    n = 30000
    idx = rng.integers(0, 3, n)
    effect = np.choose(idx, [0.0, 2.0, 5.0])
    controls = 0.6 * idx + rng.normal(0, 0.5, n)
    price = effect + 2.0 * controls + rng.normal(0, 1.0, n)
    groups = np.array(["G0", "G1", "G2"])[idx]
    assert pricing_disparity(price, groups, controls) == pytest.approx(5.0, abs=0.2)


def test_ci_covers_planted_gap_with_correlated_controls():
    """The bootstrap CI refits the joint model per resample; on the recorded
    scenario it must bracket the planted 5.0 gap (the old estimator's CI sat
    around 3.1 and could never cover it)."""
    price, groups, controls = _correlated_controls_scenario(n=4000, seed=5)
    result = pricing_disparity_with_ci(price, groups, controls, n_bootstrap=300, random_state=0)
    assert result.lower_bound < 5.0 < result.upper_bound
    assert result.point_estimate == pytest.approx(5.0, abs=0.25)
    # The whole interval must sit above the old attenuated regime (~3.1).
    assert result.lower_bound > 4.0


class TestConditionalAdverseImpact:
    """conditional_adverse_impact: the binary counterpart of the pricing_disparity
    finding. The old code regressed selection on the covariates ONLY and read the
    per-group mean residual, which is attenuated when the covariates correlate with
    the group (audit measured 0.21 vs a 0.33 joint reference). The fix fits selection
    ~ covariates + group indicators and reports the max pairwise average marginal
    effect, so the adjusted effect no longer depends on covariate-group correlation.
    """

    def _make(self, n=20000, seed=0, direct=1.2, corr=True):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 2, n)
        cov = (0.8 * g if corr else 0.0) + rng.normal(0, 1, n)
        lin = -0.5 + 1.5 * cov + direct * g
        yp = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        attr = np.where(g == 1, "A", "B")
        return yp, attr, cov

    def test_correlated_and_orthogonal_covariates_agree(self):
        # The adjusted effect must not depend on whether the covariate happens to
        # correlate with the group; the old residual estimator failed exactly here.
        yp_c, a_c, cov_c = self._make(seed=0, corr=True)
        yp_o, a_o, cov_o = self._make(seed=1, corr=False)
        cai_c = conditional_adverse_impact(yp_c, a_c, cov_c)
        cai_o = conditional_adverse_impact(yp_o, a_o, cov_o)
        assert cai_c == pytest.approx(cai_o, abs=0.03)
        # Both must be a real, non-attenuated effect (well above the ~0 a fair model gives).
        assert cai_c > 0.15

    def test_no_direct_effect_with_correlated_covariate_reads_near_zero(self):
        # Disparity fully explained by the legitimate covariate: adjusted AI ~ 0,
        # even though the covariate is correlated with the group.
        yp, attr, cov = self._make(seed=2, direct=0.0, corr=True)
        assert conditional_adverse_impact(yp, attr, cov) == pytest.approx(0.0, abs=0.03)

    def test_single_group_is_nan(self):
        yp, _, cov = self._make(seed=3)
        attr = np.array(["A"] * len(yp))
        assert math.isnan(conditional_adverse_impact(yp, attr, cov))

    def test_perfect_proxy_covariate_is_unidentified_nan(self):
        # A covariate that IS the group indicator makes the effect unidentified.
        yp, attr, _ = self._make(seed=4)
        proxy = (attr == "A").astype(float)
        assert math.isnan(conditional_adverse_impact(yp, attr, proxy))

    def test_no_decision_variation_returns_zero(self):
        n = 200
        yp = np.ones(n, dtype=int)
        attr = np.array(["A"] * 100 + ["B"] * 100)
        cov = np.random.default_rng(0).normal(0, 1, n)
        assert conditional_adverse_impact(yp, attr, cov) == 0.0
