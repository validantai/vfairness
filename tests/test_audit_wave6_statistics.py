"""Wave-6 audit pins for the statistics unit (2026-08-22).

Findings fixed and pinned here:

1. Percentile-bootstrap CI of max-min spread metrics was biased: 0% coverage
   at the parity null (the resample distribution of a folded nonnegative
   statistic sits strictly above 0, so the percentile interval could never
   contain the true value 0). Fixed with a fold-debiased construction in
   compute_metric_with_ci: permutation-gated lower bound (exact test
   inversion at the boundary) + the wider of the percentile/basic upper.

2. StatisticalResult.is_significant was True in ~100% of runs under perfect
   fairness for spread metrics (CI-excludes-zero on an interval whose lower
   bound could never reach 0). With the fixed interval the flag now fires at
   its nominal level at the null.

3. sequential_fairness_test (and FairnessPowerAnalyzer.sequential_test) used
   an LLR exactly 2x the Wald (1945) two-sample LLR, crossing the SPRT
   boundaries with half the required evidence (type I ~2.5x alpha under
   monitoring). Fixed: llr = (d*obs - d^2/2) / (1/n_a + 1/n_b).

Negative cases (the exact scenarios that used to produce the wrong result)
come first; does-not-overcorrect cases follow.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    compute_metric_with_ci,
    sequential_fairness_test,
    stratified_bootstrap_ci,
)
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
)
from vfairness.exceptions import ConfigurationError


def _parity_data(rng, n_groups, n_per, rates):
    y_pred = np.concatenate([rng.binomial(1, r, n_per) for r in rates])
    y_true = np.zeros_like(y_pred)
    attr = np.concatenate([np.full(n_per, chr(65 + g)) for g in range(n_groups)])
    return y_true, y_pred, attr


# ── Finding 1+2, negative case: the exact old failure ───────────────────────


def test_spread_ci_covers_zero_at_the_parity_null():
    """4 equal-rate groups (true max-min spread = 0): the old percentile CI
    contained 0 in 0% of runs and is_significant fired in 100%. The fixed
    interval must cover 0 at roughly its nominal rate and the significance
    flag must fire at roughly the test's level, not always."""
    rng = np.random.default_rng(42)
    n_sims = 25
    covered = 0
    significant = 0
    for _ in range(n_sims):
        y_true, y_pred, attr = _parity_data(rng, 4, 60, [0.5] * 4)
        r = compute_metric_with_ci(
            demographic_parity_difference,
            y_true,
            y_pred,
            attr,
            n_bootstrap=250,
            method="bootstrap",
            random_state=int(rng.integers(0, 2**31)),
        )
        if r.lower_bound <= 0.0 <= r.upper_bound:
            covered += 1
        if r.is_significant:
            significant += 1
    # Old behavior: covered == 0, significant == n_sims. Nominal is ~95% and
    # ~2.5%; the thresholds leave Monte Carlo slack at these sim counts.
    assert covered >= int(0.8 * n_sims), f"coverage collapsed again: {covered}/{n_sims}"
    assert significant <= int(0.2 * n_sims), (
        f"is_significant fired {significant}/{n_sims} under perfect fairness"
    )


def test_is_significant_false_at_exact_parity_single_run():
    """Single seeded parity dataset: the serialized payload must not claim a
    statistically significant disparity for a perfectly fair model."""
    rng = np.random.default_rng(7)
    y_true, y_pred, attr = _parity_data(rng, 4, 80, [0.5] * 4)
    r = compute_metric_with_ci(
        demographic_parity_difference,
        y_true,
        y_pred,
        attr,
        n_bootstrap=300,
        method="bootstrap",
        random_state=0,
    )
    assert r.is_significant is False
    assert r.lower_bound == 0.0
    assert r.to_dict()["is_significant"] is False
    # The exact boundary test is surfaced, and does not reject at parity.
    assert r.metadata.get("permutation_p_value") is not None
    assert r.metadata["permutation_p_value"] >= 0.025
    assert r.metadata.get("nonnegative_statistic") is True
    assert r.method == "stratified_bootstrap_fold_debiased"


# ── Finding 1+2, does-not-overcorrect ───────────────────────────────────────


def test_clear_disparity_still_flagged_and_covered():
    """A blatant disparity (rates 0.3 vs 0.7) must still be significant, and
    the CI must cover the true spread: the fix must not neuter detection."""
    rng = np.random.default_rng(11)
    n_sims = 20
    covered = 0
    significant = 0
    for _ in range(n_sims):
        y_true, y_pred, attr = _parity_data(rng, 2, 200, [0.3, 0.7])
        r = compute_metric_with_ci(
            demographic_parity_difference,
            y_true,
            y_pred,
            attr,
            n_bootstrap=250,
            method="bootstrap",
            random_state=int(rng.integers(0, 2**31)),
        )
        if r.lower_bound <= 0.4 <= r.upper_bound:
            covered += 1
        if r.is_significant:
            significant += 1
    assert significant == n_sims, f"real disparity missed: {significant}/{n_sims}"
    assert covered >= int(0.8 * n_sims), f"coverage lost at true 0.4: {covered}/{n_sims}"


def test_signed_statistic_keeps_percentile_interval():
    """A signed statistic (resamples straddle 0) is not fold-pathological;
    it must keep the plain percentile interval and straddle 0 at its null."""

    def signed_gap(y_true, y_pred, attr):
        return float(y_pred[attr == "A"].mean() - y_pred[attr == "B"].mean())

    rng = np.random.default_rng(3)
    y_true, y_pred, attr = _parity_data(rng, 2, 300, [0.5, 0.5])
    r = compute_metric_with_ci(
        signed_gap,
        y_true,
        y_pred,
        attr,
        n_bootstrap=300,
        method="bootstrap",
        random_state=5,
    )
    assert r.method == "stratified_bootstrap_percentile"
    assert r.lower_bound < 0.0 < r.upper_bound
    assert r.is_significant is False


def test_ratio_metric_lower_bound_is_not_clamped_to_zero():
    """disparate_impact_ratio is nonnegative but its parity value is 1, and
    its LOWER bound is the four-fifths decision surface. The fold-at-zero
    debiasing must not touch it: the permutation-null of a ratio is anchored
    near 1, not 0, so it must keep the percentile interval."""
    from vfairness.evaluation.vfairness_metrics.classification import (
        disparate_impact_ratio_with_ci,
    )

    rng = np.random.default_rng(5)
    n = 400
    y_pred = np.concatenate([rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n)])
    y_true = np.zeros_like(y_pred)
    attr = np.array(["A"] * n + ["B"] * n)
    fair = disparate_impact_ratio_with_ci(y_true, y_pred, attr, n_bootstrap=400, random_state=1)
    assert fair.method == "stratified_bootstrap_percentile"
    assert fair.lower_bound > 0.5, "fair model's DI lower bound was clamped toward 0"

    y_bad = np.concatenate([rng.binomial(1, 0.2, n), rng.binomial(1, 0.6, n)])
    unfair = disparate_impact_ratio_with_ci(y_true, y_bad, attr, n_bootstrap=400, random_state=2)
    assert unfair.upper_bound < 0.8, "blatant adverse impact must stay below four-fifths"


def test_stratified_bootstrap_ci_honors_method_and_refuses_bca():
    """The method parameter used to be accepted and silently ignored
    (percentile regardless). 'basic' must now differ from 'percentile' in
    construction, and the unimplemented 'bca' must be refused, not faked."""
    rng = np.random.default_rng(9)
    data = rng.normal(0.5, 1.0, 200)
    groups = np.repeat(["A", "B"], 100)

    def stat(d, g):
        return float(abs(d[g == "A"].mean() - d[g == "B"].mean()))

    perc = stratified_bootstrap_ci(
        data, groups, stat, n_bootstrap=300, method="percentile", random_state=1
    )
    basic = stratified_bootstrap_ci(
        data, groups, stat, n_bootstrap=300, method="basic", random_state=1
    )
    assert perc.metadata["interval_construction"] == "percentile"
    assert basic.metadata["interval_construction"] == "basic_reverse_percentile"
    assert (perc.lower_bound, perc.upper_bound) != (basic.lower_bound, basic.upper_bound)
    with pytest.raises(ConfigurationError):
        stratified_bootstrap_ci(data, groups, stat, n_bootstrap=50, method="bca")


# ── Finding 3, negative case: the doubled Wald LLR ──────────────────────────


def test_sprt_llr_matches_wald_formula_including_unequal_arms():
    """The LLR must be the TWO-SIDED Wald mixture at the correct scale.

    Two separate things are pinned here and both were once wrong:

    * THE SCALE, this test's original subject (audit 2026-08-22). The variance of
      the standardised difference is ``1/n_a + 1/n_b``, so the LLR divides by it.
      The earlier ``n_eff*(...)`` form doubled the evidence for equal arms, so the
      Wald boundaries were crossed with half the required data.
    * THE SIDEDNESS (2026-09-25). The function asks "do groups A and B differ",
      which is two-sided, and the one-sided LLR this test used to assert sent a
      difference in the unfavourable direction past the LOWER boundary and
      returned "no meaningful difference detected". The expected value is now the
      equal-weight mixture of +d and -d, ``log(cosh(t)) - c``, which is symmetric
      in the arms. The old one-sided value is asserted against, so the previous
      formula cannot come back.
    """
    rng = np.random.default_rng(21)
    for n_a, n_b in [(200, 200), (150, 250)]:
        a = rng.normal(0, 1, n_a)
        b = rng.normal(0.1, 1, n_b)
        r = sequential_fairness_test(a, b, effect_size=0.2)
        pooled = np.sqrt((a.var(ddof=1) * (n_a - 1) + b.var(ddof=1) * (n_b - 1)) / (n_a + n_b - 2))
        obs = (b.mean() - a.mean()) / pooled
        var_obs = 1.0 / n_a + 1.0 / n_b
        t = 0.2 * obs / var_obs
        correct = float(np.logaddexp(t, -t) - np.log(2.0) - 0.2**2 / (2.0 * var_obs))
        assert r["logLikelihoodRatio"] == pytest.approx(correct, abs=1e-4)

        one_sided = (0.2 * obs - 0.2**2 / 2.0) / var_obs
        assert r["logLikelihoodRatio"] != pytest.approx(one_sided, abs=1e-4)
        old_doubled = min(n_a, n_b) * 0.2 * obs - min(n_a, n_b) * 0.2**2 / 2.0
        assert r["logLikelihoodRatio"] != pytest.approx(old_doubled, abs=1e-4)

        # The whole point of the mixture: the arms may be swapped.
        swapped = sequential_fairness_test(b, a, effect_size=0.2)
        assert swapped["logLikelihoodRatio"] == pytest.approx(r["logLikelihoodRatio"], abs=1e-9)
        assert swapped["decision"] == r["decision"]


def test_sprt_anytime_type_i_error_within_wald_bound():
    """Under H0 with repeated looks, ever-reject must respect the Wald
    anytime bound alpha/(1-beta) = 0.0625. The doubled LLR produced ~0.13;
    the corrected LLR measures ~0.03 on this seeded ensemble."""
    rng = np.random.default_rng(123)
    n_streams, n_max, step = 300, 500, 25
    rejects = 0
    for _ in range(n_streams):
        a = rng.normal(0, 1, n_max)
        b = rng.normal(0, 1, n_max)
        for n in range(step, n_max + 1, step):
            r = sequential_fairness_test(a[:n], b[:n], alpha=0.05, power=0.8, effect_size=0.2)
            if r["decision"] == "reject_h0_groups_differ":
                rejects += 1
                break
            if r["decision"] == "accept_h0_no_difference":
                break
    assert rejects / n_streams <= 0.07, f"anytime type-I inflated again: {rejects}/{n_streams}"


# ── Finding 3, does-not-overcorrect ─────────────────────────────────────────


def test_sprt_still_detects_a_real_effect():
    """A true effect at the designed size must still be found: the corrected
    (halved) LLR must not make the test blind."""
    rng = np.random.default_rng(456)
    n_streams, n_max, step = 100, 400, 25
    rejects = 0
    for _ in range(n_streams):
        a = rng.normal(0, 1, n_max)
        b = rng.normal(0.5, 1, n_max)
        for n in range(step, n_max + 1, step):
            r = sequential_fairness_test(a[:n], b[:n], alpha=0.05, power=0.8, effect_size=0.5)
            if r["decision"] == "reject_h0_groups_differ":
                rejects += 1
                break
            if r["decision"] == "accept_h0_no_difference":
                break
    assert rejects / n_streams >= 0.8, f"SPRT lost its power: {rejects}/{n_streams}"


def test_power_analyzer_sequential_test_uses_corrected_llr():
    """FairnessPowerAnalyzer.sequential_test shared the doubled formula; pin
    that it now computes the Wald LLR (exactly half the old value)."""
    import pandas as pd

    from vfairness.operations.experimentation.experiment import FairnessExperiment
    from vfairness.operations.experimentation.power import FairnessPowerAnalyzer

    rng = np.random.default_rng(11)
    n = 120
    ctrl = pd.DataFrame({"g": ["A"] * n, "y": rng.normal(0, 1, n)})
    treat = pd.DataFrame({"g": ["A"] * n, "y": rng.normal(0.3, 1, n)})
    analyzer = FairnessPowerAnalyzer(FairnessExperiment(ctrl, treat, ["g"], "y"))
    result = analyzer.sequential_test(effect_size=0.2)[("A",)]

    c, t = ctrl["y"].values, treat["y"].values
    pooled = np.sqrt((c.var(ddof=1) * (n - 1) + t.var(ddof=1) * (n - 1)) / (2 * n - 2))
    obs = (t.mean() - c.mean()) / pooled
    correct = (0.2 * obs - 0.2**2 / 2.0) / (2.0 / n)
    old_doubled = n * 0.2 * obs - n * 0.2**2 / 2.0
    assert result.log_likelihood_ratio == pytest.approx(correct, abs=1e-3)
    assert old_doubled == pytest.approx(2.0 * result.log_likelihood_ratio, abs=1e-3)
