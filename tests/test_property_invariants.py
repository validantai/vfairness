"""Property-based invariants for core fairness metrics (VB-TEST-2).

Uses Hypothesis to assert properties that must hold for every valid input,
not just for a handful of hand-picked cases:

    * Range invariants: difference metrics live in [0, 1]; the disparate
      impact ratio lives in [0, 1].
    * Label-permutation symmetry: relabeling the two groups (A <-> B) does
      not change a symmetric max-minus-min disparity.
    * Degenerate inputs return the documented verdict rather than crashing.

Hypothesis is an optional test-time dependency. If it is not installed the
whole module skips cleanly (``pytest.importorskip``).
"""

import warnings

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import vfairness as v  # noqa: E402

MIN_GROUP = 30

# A group's predictions are described by (n_samples, n_positive). This keeps
# every group at or above the default minimum size so nothing is dropped.
_group_pred = st.integers(min_value=MIN_GROUP, max_value=120).flatmap(
    lambda n: st.tuples(st.just(n), st.integers(min_value=0, max_value=n))
)


def _build_two_group_pred(a, b):
    """Build (y_true, y_pred, groups) for two groups from (n, n_pos) specs."""
    n_a, pos_a = a
    n_b, pos_b = b
    y_pred = np.array([1] * pos_a + [0] * (n_a - pos_a) + [1] * pos_b + [0] * (n_b - pos_b))
    y_true = np.zeros(n_a + n_b, dtype=int)
    groups = np.array(["A"] * n_a + ["B"] * n_b)
    return y_true, y_pred, groups


_std_settings = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Range invariants.
# ---------------------------------------------------------------------------
@_std_settings
@given(a=_group_pred, b=_group_pred)
@example(a=(30, 30), b=(30, 0))  # maximal disparity
@example(a=(30, 15), b=(40, 20))  # equal selection rates
def test_demographic_parity_difference_in_unit_range(a, b):
    y_true, y_pred, groups = _build_two_group_pred(a, b)
    diff = v.demographic_parity_difference(y_true, y_pred, groups)
    assert not np.isnan(diff)
    assert 0.0 <= diff <= 1.0 + 1e-12


@_std_settings
@given(a=_group_pred, b=_group_pred)
@example(a=(30, 30), b=(30, 0))  # one group has zero selection -> ratio 0
@example(a=(30, 0), b=(30, 0))  # NOBODY selected -> 0/0, unmeasurable -> NaN
def test_disparate_impact_ratio_in_unit_range(a, b):
    """REWRITTEN 2026-08-28: the old version PINNED THE DEFECT.

    Its second ``@example`` was commented "both zero selection -> documented
    ratio 1.0" and it asserted ``not np.isnan(ratio)``, i.e. it required the
    all-reject model to answer PERFECT PARITY from a 0/0 quotient. fairlearn
    and aif360 both return NaN there; so does this library now. The range
    invariant itself is real and is kept, applied to the values that ARE
    measurable, which is the only place a range means anything.
    """
    y_true, y_pred, groups = _build_two_group_pred(a, b)
    ratio = v.demographic_parity_ratio(y_true, y_pred, groups)
    nobody_selected = a[1] == 0 and b[1] == 0
    if nobody_selected:
        # Not a range failure: there is no ratio to put in a range.
        assert np.isnan(ratio), "0/0 must be unmeasurable, never 1.0 (perfect parity)"
        return
    assert not np.isnan(ratio)
    assert 0.0 <= ratio <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Label-permutation (group-relabel) symmetry.
# The max-minus-min disparity is symmetric in the group labels, so swapping
# which block is "A" and which is "B" must leave the metric unchanged.
# ---------------------------------------------------------------------------
@_std_settings
@given(a=_group_pred, b=_group_pred)
@example(a=(30, 30), b=(30, 0))
@example(a=(50, 10), b=(30, 25))
def test_demographic_parity_difference_group_relabel_symmetry(a, b):
    y_true1, y_pred1, groups1 = _build_two_group_pred(a, b)
    y_true2, y_pred2, groups2 = _build_two_group_pred(b, a)
    d1 = v.demographic_parity_difference(y_true1, y_pred1, groups1)
    d2 = v.demographic_parity_difference(y_true2, y_pred2, groups2)
    assert d1 == pytest.approx(d2, abs=1e-12)


@_std_settings
@given(a=_group_pred, b=_group_pred)
@example(a=(30, 30), b=(40, 12))
@example(a=(30, 0), b=(40, 0))  # NOBODY selected: unmeasurable from either side
def test_disparate_impact_ratio_group_relabel_symmetry(a, b):
    """Relabelling the groups cannot change the answer, INCLUDING when the
    answer is "could not check".

    ``pytest.approx`` compares NaN as unequal, so the unmeasurable case is
    asserted as a pair of NaNs rather than smuggled through a numeric compare.
    Symmetry has to hold in the third state too: an input that is unmeasurable
    one way round and a number the other way round would be a real defect.
    """
    y_true1, y_pred1, groups1 = _build_two_group_pred(a, b)
    y_true2, y_pred2, groups2 = _build_two_group_pred(b, a)
    r1 = v.demographic_parity_ratio(y_true1, y_pred1, groups1)
    r2 = v.demographic_parity_ratio(y_true2, y_pred2, groups2)
    assert np.isnan(r1) == np.isnan(r2), "measurability must not depend on group order"
    if np.isnan(r1):
        return
    assert r1 == pytest.approx(r2, abs=1e-12)


# ---------------------------------------------------------------------------
# ECE range invariant. ECE is a weighted average of absolute gaps between
# per-bin confidence and accuracy, so it must land in [0, 1].
# ---------------------------------------------------------------------------
@_std_settings
@given(
    labels=st.lists(st.integers(min_value=0, max_value=1), min_size=MIN_GROUP, max_size=150),
    seed=st.integers(min_value=0, max_value=10_000),
)
@example(labels=[1] * MIN_GROUP, seed=0)  # all-positive labels
@example(labels=[0] * MIN_GROUP, seed=0)  # all-negative labels
def test_expected_calibration_error_in_unit_range(labels, seed):
    y_true = np.array(labels, dtype=int)
    rng = np.random.default_rng(seed)
    y_prob = rng.random(len(labels))
    result = v.expected_calibration_error(y_true, y_prob)
    assert 0.0 <= result.overall_value <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Degenerate inputs: documented verdicts (not raw crashes).
# ---------------------------------------------------------------------------
def test_single_group_is_not_assessable():
    """One group only: there is nothing to compare it against, so DP difference
    and DP ratio are NaN (insufficient evidence).

    These previously asserted 0.0 and 1.0, the values of PERFECT parity, which
    is the same misleading-sentinel failure that
    test_zero_positive_labels_equal_opportunity_returns_nan_verdict (below)
    already refuses for an undefined TPR. A metric that never ran must not
    return the number that means 'flawless'.
    """
    n = 60
    groups = np.array(["A"] * n)
    y_pred = np.array([1] * 30 + [0] * 30)
    y_true = np.zeros(n, dtype=int)
    assert np.isnan(v.demographic_parity_difference(y_true, y_pred, groups))
    assert np.isnan(v.demographic_parity_ratio(y_true, y_pred, groups))


def test_tiny_second_group_dropped_is_not_assessable():
    """A group below min_group_size is dropped, leaving one valid group, so the
    comparison cannot happen: NaN, not the 0.0 that reads as perfect parity."""
    groups = np.array(["A"] * 60 + ["B"] * 5)
    y_pred = np.array([1] * 30 + [0] * 30 + [1] * 3 + [0] * 2)
    y_true = np.zeros(65, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(v.demographic_parity_difference(y_true, y_pred, groups))


def test_all_positive_labels_equal_opportunity_is_defined():
    """Every label positive: TPR is well defined, so EO diff is a real number."""
    n = 60
    groups = np.array(["A"] * n + ["B"] * n)
    y_true = np.ones(2 * n, dtype=int)
    y_pred = np.array([1] * 40 + [0] * 20 + [1] * 30 + [0] * 30)
    got = v.equal_opportunity_difference(y_true, y_pred, groups)
    assert not np.isnan(got)
    assert 0.0 <= got <= 1.0 + 1e-12


def test_zero_positive_labels_equal_opportunity_returns_nan_verdict():
    """No positive labels anywhere: TPR is undefined. The library documents
    this as returning NaN with a UserWarning rather than a misleading 0.0."""
    n = 60
    groups = np.array(["A"] * n + ["B"] * n)
    y_true = np.zeros(2 * n, dtype=int)
    y_pred = np.array([1] * 20 + [0] * 40 + [1] * 10 + [0] * 50)
    with pytest.warns(UserWarning):
        got = v.equal_opportunity_difference(y_true, y_pred, groups)
    assert np.isnan(got)


def test_constant_probability_calibration_is_finite():
    """A single constant probability collapses into one bin; ECE stays finite
    (no zero-denominator NaN)."""
    n = 100
    y_true = np.array([1] * 50 + [0] * 50)
    y_prob = np.full(n, 0.5)
    result = v.expected_calibration_error(y_true, y_prob)
    assert not np.isnan(result.overall_value)
    assert 0.0 <= result.overall_value <= 1.0 + 1e-12
