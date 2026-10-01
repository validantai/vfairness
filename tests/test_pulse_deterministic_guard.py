"""The deterministic-separation guard, over the values and sizes it used to miss.

``_pair_stats`` treats two internally-constant arms as a deterministic
comparison: equal outputs are no evidence, and different outputs are separation
whose EFFECT SIZE is the actual magnitude of the gap. Two audit fixes live in
that branch:

  _MIN_DETERMINISTIC_DELTA   a gap below the scorer's quantization floor is not
                             a demographic disparity and must not fabricate a
                             finding.
  the 2026-07-11 effect-size fix
                             a 0.01 sentiment delta and a full favourable /
                             unfavourable flip must NOT be scored identically.

The branch was guarded by ``a.var() == 0.0``, an exact float comparison on an
ACCUMULATED statistic. A constant array of a value that is not exactly
representable in binary does not have exactly zero variance, and whether it
does depends on n as well as the value. Every (value, n) the guard missed fell
through to ``mannwhitneyu``, which separates two constant arms PERFECTLY
whatever the gap, so BOTH fixes were bypassed at once.

Measured before the fix, two deterministic arms differing by 0.0001, a
hundredth of the quantization floor:

    n=5   p=0.00398   effect=1.0
    n=20  p=4.68e-10  effect=1.0
    n=25  p=2.77e-12  effect=1.0

and the identical p-values for a gap of 0.7. Non-monotonic in n, which is the
tell that floating point and not the data is deciding.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.operations.pulse.llm_probe import _MIN_DETERMINISTIC_DELTA, _pair_stats

# Values chosen to include ones that are NOT exactly representable in binary,
# and sizes chosen because whether var() is exactly zero depends on both.
_VALUES = [0.5, 0.25, 0.9, 0.6, 0.2, 0.1, 1.0 / 3.0, 0.51]
_SIZES = [2, 5, 20, 25, 51]


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("size", _SIZES)
def test_a_sub_quantization_gap_never_fabricates_a_finding(value: float, size: int) -> None:
    """The defect, at every value and size. A gap a hundredth of the floor."""
    gap = _MIN_DETERMINISTIC_DELTA / 100.0
    p, effect = _pair_stats([value] * size, [value + gap] * size)
    assert p == 1.0, f"value={value} size={size} reported separation from a {gap} gap"
    assert effect == 0.0


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("size", _SIZES)
def test_identical_constant_arms_are_compared_not_refused(value: float, size: int) -> None:
    """At temperature 0 this is the commonest genuinely-fair shape there is."""
    p, effect = _pair_stats([value] * size, [value] * size)
    assert p == 1.0, f"value={value} size={size} did not read as compared-no-difference"
    assert effect == 0.0


@pytest.mark.parametrize("size", _SIZES)
def test_a_real_separation_reports_its_actual_magnitude(size: int) -> None:
    """The 2026-07-11 fix. A large gap and a small one must not score alike.

    Falling through to the rank-sum gave effect=1.0 for both, which is a
    maximal effect size reported from a quantization difference.
    """
    small_p, small_e = _pair_stats([0.9] * size, [0.9 + 0.06] * size)
    large_p, large_e = _pair_stats([0.9] * size, [0.9 + 0.7] * size)
    assert small_p == 0.0 and large_p == 0.0
    assert small_e == pytest.approx(0.06, abs=1e-9)
    assert large_e == pytest.approx(0.7, abs=1e-9)
    assert small_e < large_e, "a small gap and a large one must not score identically"


@pytest.mark.parametrize("size", _SIZES)
def test_the_guard_does_not_swallow_data_that_really_varies(size: int) -> None:
    """Over-correction control.

    ptp is exact, so one differing observation is variance and must take the
    ordinary rank-sum path. A guard that captured near-constant data would
    replace a real test with a deterministic reading.
    """
    if size < 3:
        pytest.skip("needs room for one differing observation")
    a = [0.9] * size
    a[0] = 0.1
    p, effect = _pair_stats(a, [0.9] * size)
    assert p is not None
    # Not the deterministic branch: that would answer exactly 1.0 or exactly 0.0.
    assert p not in (0.0,), "a varying arm must not take the deterministic path"


def test_the_variance_shortcut_really_was_value_and_size_dependent() -> None:
    """The arithmetic the fix rests on, asserted rather than described.

    If numpy ever makes var() exact for these, this test says so loudly rather
    than leaving a comment that has quietly become false.
    """
    assert float(np.var(np.full(20, 0.9))) != 0.0
    assert float(np.var(np.full(25, 0.9))) == 0.0
    assert float(np.var(np.full(20, 0.5))) == 0.0
    # ptp is exact for all of them, which is why it is the guard.
    for value in _VALUES:
        for size in _SIZES:
            assert float(np.ptp(np.full(size, value))) == 0.0
