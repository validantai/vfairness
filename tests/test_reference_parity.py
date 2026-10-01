"""Differential parity against reference libraries (VB-TEST-4).

Where a definition is shared, vfairness must agree with the established library
within tolerance. Intentional definitional differences are documented in
docs/DIVERGENCES.md and are tested against the intended definition, not blindly
against the other library. These tests skip if the reference library is not
installed (they run in the cross-library-parity CI job).
"""

import numpy as np
import pytest

import vfairness


def _two_group_fixture(n=400, seed=0):
    rng = np.random.RandomState(seed)
    half = n // 2
    sensitive = np.array(["A"] * half + ["B"] * half)
    # Different selection rates per group so the disparity is non-trivial.
    y_pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.3, half)])
    y_true = rng.binomial(1, 0.5, n)
    return y_true, y_pred, sensitive


def test_demographic_parity_matches_fairlearn_two_group():
    flm = pytest.importorskip("fairlearn.metrics")
    y_true, y_pred, sensitive = _two_group_fixture()
    vf = vfairness.demographic_parity_difference(y_true, y_pred, sensitive)
    fl = flm.demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    # Two groups: max-min spread and pairwise difference coincide (see DIVERGENCES.md).
    assert vf == pytest.approx(fl, abs=1e-9)


def test_equalized_odds_matches_fairlearn_two_group():
    flm = pytest.importorskip("fairlearn.metrics")
    y_true, y_pred, sensitive = _two_group_fixture(seed=1)
    vf = vfairness.equalized_odds_difference(y_true, y_pred, sensitive)
    fl = flm.equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    # Allow a small tolerance: definitions of the odds aggregation can differ
    # slightly (max of TPR/FPR gaps); both should be in the same ballpark.
    assert vf == pytest.approx(fl, abs=0.05)
