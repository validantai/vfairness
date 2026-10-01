"""A test that CANNOT fire is not a test, and "not significant" must not be
printed for one.

READINESS-6, 2026-09-10. `_statistics._testable` already removes a hypothesis
that did not run, and that is only half the problem. A test that DID run can
still be incapable of ever producing a significant answer, because a discrete
test's p-value has a FLOOR set by its design. When that floor sits above the
threshold the family applies, the detector cannot fire for any data at all, and
the reader is shown "not significant".

Measured on this library the same day, before the fix:

- A pulse probe against a stub refusing 100 percent of requests for one
  demographic name and 0 percent for every other produced NO finding, because
  its Fisher floor of 0.0021645 sat above the rank-1 Benjamini-Hochberg bar of
  0.05/32 = 0.0015625.
- A calibration sufficiency test returned passes=True for a score band where one
  group's realised positive rate was 94.3 percent against another's 55.6 percent
  at the same predicted score, because its permutation floor of 1/201 across 12
  bins exceeded alpha.

Nine detectors in the library run a discrete test and one carried this check.
These pins cover the shared version of it.
"""

from __future__ import annotations

import math

import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_fisher,
    min_attainable_p_mannwhitney,
    min_attainable_p_permutation,
)


class TestTheFloorsAreTheRealArithmetic:
    """The numbers, not a plausible-looking approximation of them."""

    def test_fisher_matches_the_incident(self):
        # 2 / C(12, 6) = 2 / 924. This exact number suppressed a categorical
        # denial of service earlier today.
        assert min_attainable_p_fisher(6, 6) == pytest.approx(2 / math.comb(12, 6))
        assert min_attainable_p_fisher(6, 6) == pytest.approx(0.0021645, rel=1e-4)
        # And it moves with the design, which is the whole point.
        assert min_attainable_p_fisher(3, 3) == pytest.approx(0.10, rel=1e-6)
        assert min_attainable_p_fisher(8, 8) < min_attainable_p_fisher(6, 6)

    def test_permutation_is_the_plus_one_estimator(self):
        assert min_attainable_p_permutation(200) == pytest.approx(1 / 201)
        assert min_attainable_p_permutation(1000) == pytest.approx(1 / 1001)
        # Zero resamples cannot produce evidence of anything, and the floor
        # says so rather than reading as a clean p of 1.0.
        assert min_attainable_p_permutation(0) == 1.0

    def test_mann_whitney_takes_the_smallest_across_methods(self):
        """FAIL-SAFE DIRECTION, and it is not a detail.

        scipy resolves to the exact test on distinct values and to the
        tie-corrected asymptotic one when there are ties, and the tie-corrected
        floor is LOWER. Assuming the exact floor would call a detectable design
        dead, which suppresses a real finding: the exact harm this check exists
        to prevent, inflicted by the check itself.
        """
        from scipy import stats

        n = 5
        _, exact = stats.mannwhitneyu(
            list(range(1, n + 1)),
            list(range(n + 1, 2 * n + 1)),
            alternative="two-sided",
            method="exact",
        )
        _, tied = stats.mannwhitneyu(
            [0.0] * n, [1.0] * n, alternative="two-sided", method="asymptotic"
        )
        floor = min_attainable_p_mannwhitney(n, n)
        assert floor == pytest.approx(min(float(exact), float(tied)))
        assert floor < float(exact), (
            "the helper took the exact floor, which would declare a design "
            "undetectable that ties could in fact have made detectable"
        )

    def test_an_uncomputable_design_is_none_not_a_number(self):
        assert min_attainable_p_fisher(0, 6) is None
        assert min_attainable_p_fisher(-1, 6) is None
        assert min_attainable_p_mannwhitney(0, 5) is None
        assert min_attainable_p_permutation("not a number") is None


class TestDetectabilityHasThreeStates:
    def test_a_design_with_no_power_says_what_the_reading_does_not_mean(self):
        floor = min_attainable_p_fisher(6, 6)
        detectable, note = detectability(floor, n_family=32)
        assert detectable is False
        assert "NOT DETECTABLE" in note
        assert "absence of statistical power" in note, (
            "the note must say what a 'not significant' reading here does NOT "
            "mean, or it is just another number the reader has to interpret"
        )
        # The bar it names is the one it actually used.
        assert "0.0015625" in note or "0.001563" in note or "0.001562" in note

    def test_the_same_test_in_a_smaller_family_is_detectable(self):
        """OVER-CORRECTION CONTROL. A check that reports every design dead is
        as useless as no check. Splitting the family restored this exact test's
        power earlier today, and the helper has to see that."""
        floor = min_attainable_p_fisher(6, 6)
        assert detectability(floor, n_family=5)[0] is True
        assert detectability(floor, n_family=5)[1] == ""
        assert detectability(floor, n_family=1)[0] is True

    def test_the_calibration_case(self):
        floor = min_attainable_p_permutation(200)
        assert detectability(floor, n_family=12)[0] is False
        # And with enough bins removed, or enough resamples, it works again.
        assert detectability(floor, n_family=10)[0] is True
        assert detectability(min_attainable_p_permutation(1000), n_family=12)[0] is True

    def test_could_not_check_is_its_own_state(self):
        detectable, note = detectability(None, n_family=5)
        assert detectable is None, "an uncomputable floor is not a failed design"
        assert "COULD NOT CHECK" in note
        assert detectability(float("nan"), n_family=5)[0] is None

    def test_a_generous_design_is_never_called_dead(self):
        """The other over-correction control: real, well-powered designs."""
        assert detectability(min_attainable_p_fisher(50, 50), n_family=100)[0] is True
        assert detectability(min_attainable_p_mannwhitney(30, 30), n_family=50)[0] is True
        assert detectability(min_attainable_p_permutation(10000), n_family=200)[0] is True
