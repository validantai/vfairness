"""Determinism / reproducibility tests for randomized metrics (VB-TEST-5).

Any metric that draws random numbers (bootstrap confidence intervals,
permutation tests) must be reproducible: the same ``random_state`` must
produce byte-identical results, and repeated calls with that seed must be
stable. Where the API exposes a seed we use it; a different seed is expected
to (usually) move the result, which we sanity-check as well.

Signatures were verified against the source before writing:

    bootstrap_ci(data, statistic, n_bootstrap=..., random_state=...)
    demographic_parity_difference_with_ci(..., n_bootstrap=..., random_state=...)
    equalized_odds_difference_with_ci(..., random_state=...)
    permutation_test_demographic_parity(y_pred, sensitive_attr, ...,
        random_state=...)
    comprehensive_fairness_test(..., random_state=...)
"""

import numpy as np

import vfairness as v


def _two_group_dataset(seed=0):
    rng = np.random.default_rng(seed)
    n = 150
    groups = np.array(["A"] * n + ["B"] * n)
    y_pred = np.concatenate(
        [
            (rng.random(n) < 0.6).astype(int),
            (rng.random(n) < 0.3).astype(int),
        ]
    )
    y_true = np.concatenate(
        [
            (rng.random(n) < 0.5).astype(int),
            (rng.random(n) < 0.4).astype(int),
        ]
    )
    return y_true, y_pred, groups


# ---------------------------------------------------------------------------
# bootstrap_ci: same seed -> identical bounds; stable across repeats.
# ---------------------------------------------------------------------------
def test_bootstrap_ci_same_seed_is_identical():
    data = np.concatenate([np.zeros(60), np.ones(40)])
    r1 = v.bootstrap_ci(data, np.mean, n_bootstrap=500, random_state=42)
    r2 = v.bootstrap_ci(data, np.mean, n_bootstrap=500, random_state=42)
    assert r1.lower_bound == r2.lower_bound
    assert r1.upper_bound == r2.upper_bound
    assert r1.point_estimate == r2.point_estimate


def test_bootstrap_ci_stable_across_repeated_calls():
    data = np.concatenate([np.zeros(70), np.ones(30)])
    results = [
        (r.lower_bound, r.upper_bound)
        for r in (v.bootstrap_ci(data, np.mean, n_bootstrap=400, random_state=7) for _ in range(5))
    ]
    assert all(r == results[0] for r in results)


def test_bootstrap_ci_different_seed_can_differ():
    """Different seeds should move at least one bound. This guards against the
    seed being silently ignored (which would make every seed identical).

    Uses continuous data on purpose: with discrete 0/1 data the bootstrap-mean
    percentiles land on a coarse grid and two seeds collide ~10% of the time,
    which made this negative control flaky.
    """
    data = np.random.RandomState(0).normal(0.0, 1.0, 200)
    r_a = v.bootstrap_ci(data, np.mean, n_bootstrap=500, random_state=1)
    r_b = v.bootstrap_ci(data, np.mean, n_bootstrap=500, random_state=999)
    assert (r_a.lower_bound, r_a.upper_bound) != (r_b.lower_bound, r_b.upper_bound)


# ---------------------------------------------------------------------------
# demographic_parity_difference_with_ci: seeded bootstrap CI.
# ---------------------------------------------------------------------------
def test_demographic_parity_ci_same_seed_is_identical():
    y_true, y_pred, groups = _two_group_dataset(seed=3)
    s1 = v.demographic_parity_difference_with_ci(
        y_true, y_pred, groups, n_bootstrap=400, random_state=11
    )
    s2 = v.demographic_parity_difference_with_ci(
        y_true, y_pred, groups, n_bootstrap=400, random_state=11
    )
    assert s1.point_estimate == s2.point_estimate
    assert s1.lower_bound == s2.lower_bound
    assert s1.upper_bound == s2.upper_bound


def test_equalized_odds_ci_same_seed_is_identical():
    y_true, y_pred, groups = _two_group_dataset(seed=5)
    s1 = v.equalized_odds_difference_with_ci(
        y_true, y_pred, groups, n_bootstrap=400, random_state=13
    )
    s2 = v.equalized_odds_difference_with_ci(
        y_true, y_pred, groups, n_bootstrap=400, random_state=13
    )
    assert s1.point_estimate == s2.point_estimate
    assert s1.lower_bound == s2.lower_bound
    assert s1.upper_bound == s2.upper_bound


# ---------------------------------------------------------------------------
# permutation_test_demographic_parity: seeded null distribution.
# ---------------------------------------------------------------------------
def test_permutation_test_same_seed_is_identical():
    _, y_pred, groups = _two_group_dataset(seed=2)
    p1 = v.permutation_test_demographic_parity(y_pred, groups, n_permutations=300, random_state=21)
    p2 = v.permutation_test_demographic_parity(y_pred, groups, n_permutations=300, random_state=21)
    assert p1.observed_statistic == p2.observed_statistic
    assert p1.p_value == p2.p_value
    assert np.array_equal(np.asarray(p1.null_distribution), np.asarray(p2.null_distribution))


def test_permutation_test_different_seed_null_distribution_differs():
    _, y_pred, groups = _two_group_dataset(seed=2)
    p1 = v.permutation_test_demographic_parity(y_pred, groups, n_permutations=300, random_state=1)
    p2 = v.permutation_test_demographic_parity(y_pred, groups, n_permutations=300, random_state=2)
    # Observed statistic is deterministic (no randomness), so it must match;
    # the permuted null distribution is random, so it should differ.
    assert p1.observed_statistic == p2.observed_statistic
    assert not np.array_equal(np.asarray(p1.null_distribution), np.asarray(p2.null_distribution))


# ---------------------------------------------------------------------------
# comprehensive_fairness_test: end-to-end seeded pipeline reproducibility.
# ---------------------------------------------------------------------------
def test_comprehensive_fairness_test_same_seed_is_reproducible():
    y_true, y_pred, groups = _two_group_dataset(seed=9)
    r1 = v.comprehensive_fairness_test(y_true, y_pred, groups, n_permutations=300, random_state=31)
    r2 = v.comprehensive_fairness_test(y_true, y_pred, groups, n_permutations=300, random_state=31)

    # Top-level permutation p-values and the multiple-testing adjusted values
    # must be identical for the same seed.
    assert r1["p_values"]  # at least one test ran
    assert r1["p_values"] == r2["p_values"]
    assert r1["adjusted_p_values"] == r2["adjusted_p_values"]

    # Per-metric permutation results must match too.
    assert set(r1["metric_tests"]) == set(r2["metric_tests"])
    for name, res1 in r1["metric_tests"].items():
        res2 = r2["metric_tests"][name]
        assert res1["observed_statistic"] == res2["observed_statistic"]
        assert res1["p_value"] == res2["p_value"]
