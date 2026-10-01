"""LF-20: a counterfactual run is a matched-pair design, and is tested as one.

The library tested it with ``mannwhitneyu``, which is unpaired. That throws
away the thing the design is built on, the prompt being held constant, and
charges prompt-to-prompt variance to the group difference. It is both weaker
than the design deserves and answering a different null than the design poses.

Two things every test here is really about:

  1. A test that could not run returns NaN, never 1.0. ``_testable`` excludes a
     NaN p from a correction family and is blind to a finite 1.0 sentinel, so a
     sentinel there is a silent full-power family member with no power at all.
  2. A "not significant" that the design could never have contradicted is
     refused, not reported. McNemar's evidence is only its discordant pairs.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness._not_assessed import NOT_ASSESSED
from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_mcnemar,
    min_attainable_p_sign_flip,
)
from vfairness.llm.paired import (
    mcnemar_paired_test,
    sign_flip_paired_test,
    usable_binary_pairs,
    usable_numeric_pairs,
)

# ---------------------------------------------------------------------------
# The design floors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("discordant", "expected"),
    [(0, 1.0), (1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)],
)
def test_the_mcnemar_floor_matches_the_closed_form(discordant: int, expected: float) -> None:
    """2 * 0.5 ** (b + c), capped at 1. Computed through scipy, asserted here."""
    assert min_attainable_p_mcnemar(discordant) == pytest.approx(expected)


def test_five_discordant_pairs_cannot_reach_alpha_and_the_note_says_what_that_means() -> None:
    """The line that makes this function worth existing.

    Five discordant pairs is an entirely ordinary counterfactual result, and no
    split of them reaches 0.05. A "not significant" there is an absence of
    power, and the note has to say so where a reader sees it.
    """
    detectable, note = detectability(min_attainable_p_mcnemar(5), n_family=1, alpha=0.05)
    assert detectable is False
    assert "No data, however extreme, could make this test significant" in note
    assert "not evidence that nothing is wrong" in note
    assert detectability(min_attainable_p_mcnemar(6), n_family=1, alpha=0.05)[0] is True


def test_the_family_makes_the_bar_harder_and_that_is_visible() -> None:
    # Six discordant pairs clears 0.05 alone and cannot clear its share of a
    # family of two.
    assert detectability(min_attainable_p_mcnemar(6), n_family=1)[0] is True
    assert detectability(min_attainable_p_mcnemar(6), n_family=2)[0] is False


def test_the_sign_flip_floor_takes_the_larger_of_the_two_limits() -> None:
    # Exact enumeration over 10 pairs floors at 2 / 2**10.
    assert min_attainable_p_sign_flip(10) == pytest.approx(2.0 / 1024.0)
    # With only 99 resamples the sampling floor 1/100 binds instead.
    assert min_attainable_p_sign_flip(10, 99) == pytest.approx(0.01)
    # With plenty of resamples the enumeration floor binds again.
    assert min_attainable_p_sign_flip(10, 9999) == pytest.approx(2.0 / 1024.0)


def test_a_floor_that_cannot_be_computed_is_none_not_a_number() -> None:
    assert min_attainable_p_mcnemar("five") is None  # type: ignore[arg-type]
    assert min_attainable_p_mcnemar(-1) is None
    assert min_attainable_p_sign_flip(-1) is None
    # Zero is different: the floor is genuinely known and it is the largest.
    assert min_attainable_p_sign_flip(0) == 1.0
    assert min_attainable_p_mcnemar(0) == 1.0


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_a_lopsided_refusal_disparity_is_found() -> None:
    # The variant arm is refused on 8 prompts the reference arm is not.
    ref = [False] * 20
    var = [True] * 8 + [False] * 12
    r = mcnemar_paired_test(ref, var, outcome="refusal")
    assert r["tested"] is True
    assert r["n_discordant"] == 8
    assert r["p_value"] < 0.05
    assert r["reason"] == ""


def test_too_few_discordant_pairs_is_refused_with_nan_never_one() -> None:
    """The defect this whole module guards.

    A 1.0 here would join a BH family as a full member with no power, and
    nothing downstream could tell it from a measured 1.0.
    """
    ref = [False] * 200
    var = [True] * 5 + [False] * 195
    r = mcnemar_paired_test(ref, var, outcome="refusal")
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
    assert r["p_value"] != 1.0
    assert r["state"] == NOT_ASSESSED
    # 200 pairs does not rescue 5 discordant ones, and the reason says why.
    assert r["n_pairs_usable"] == 200
    assert "evidence is ONLY its discordant pairs" in r["reason"]


def test_perfect_agreement_is_refused_rather_than_reported_as_a_clean_result() -> None:
    """Zero discordant pairs is no evidence, not evidence of fairness."""
    r = mcnemar_paired_test([True] * 30, [True] * 30, outcome="refusal")
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
    assert r["n_discordant"] == 0
    assert r["n_concordant"] == 30


def test_unusable_outcomes_are_dropped_and_counted_never_coerced() -> None:
    """``bool(x)`` would turn NaN into True and "" into False.

    Both are verdicts invented from unusable input, and on the refusal metric
    "True" is the finding itself.
    """
    ref = [True, False, float("nan"), "refused", None, 1, 0.0]
    var = [False, False, True, True, True, 0, 1.0]
    usable_ref, usable_var, supplied = usable_binary_pairs(ref, var)
    assert supplied == 7
    # nan, the string and None are dropped; 1 and 0.0 are usable.
    assert usable_ref == [True, False, True, False]
    assert usable_var == [False, False, False, True]

    r = mcnemar_paired_test(ref, var, outcome="refusal")
    assert r["n_pairs_supplied"] == 7
    assert r["n_pairs_usable"] == 4
    assert r["n_pairs_dropped"] == 3


def test_a_length_mismatch_counts_the_remainder_as_dropped() -> None:
    """Zipping to the shorter arm silently would hide an uneven producer."""
    r = mcnemar_paired_test([True] * 10, [False] * 6, outcome="refusal")
    assert r["n_pairs_supplied"] == 10
    assert r["n_pairs_usable"] == 6
    assert r["n_pairs_dropped"] == 4


def test_a_measured_result_on_a_subset_says_it_is_a_subset() -> None:
    ref = [False] * 20 + [None] * 5
    var = [True] * 8 + [False] * 12 + [None] * 5
    r = mcnemar_paired_test(ref, var, outcome="refusal")
    assert r["tested"] is True
    assert "measured on 20 of 25 pair(s)" in r["reason"]
    assert "not a random subset" in r["reason"]


def test_no_usable_pair_at_all_is_refused_and_says_how_many_were_supplied() -> None:
    r = mcnemar_paired_test([None] * 12, [None] * 12, outcome="refusal")
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
    assert "12 pair(s) supplied" in r["reason"]


# ---------------------------------------------------------------------------
# Paired sign-flip permutation
# ---------------------------------------------------------------------------


def test_a_consistent_shift_in_every_pair_is_found() -> None:
    ref = list(np.zeros(12))
    var = list(np.full(12, 0.4))
    r = sign_flip_paired_test(ref, var, metric="sentiment")
    assert r["tested"] is True
    assert r["p_value"] < 0.05
    assert r["statistic"] == pytest.approx(0.4)


def test_the_paired_test_sees_what_the_unpaired_one_misses() -> None:
    """The reason LF-20 exists, measured rather than argued.

    Twelve prompts with wildly different baselines, each shifted by the same
    small amount when the attribute changes. The pairing removes the baseline
    spread entirely. An unpaired rank-sum drowns in it.
    """
    from scipy import stats as _st

    rng = np.random.default_rng(11)
    baseline = rng.normal(0.0, 1.0, size=12)
    ref = baseline
    var = baseline + 0.25

    paired = sign_flip_paired_test(list(ref), list(var), metric="sentiment")
    _, unpaired_p = _st.mannwhitneyu(ref, var, alternative="two-sided")

    assert paired["tested"] is True
    assert paired["p_value"] < 0.05
    assert float(unpaired_p) > 0.05
    # And the paired answer is the one the design actually poses.
    assert paired["statistic"] == pytest.approx(0.25)


def test_too_few_pairs_is_refused_with_nan_never_one() -> None:
    r = sign_flip_paired_test(list(np.zeros(4)), list(np.full(4, 5.0)), metric="toxicity")
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
    assert r["min_attainable_p"] == pytest.approx(0.125)
    assert "could not reach 0.05 for any values" in r["reason"]


def test_identical_pairs_are_measured_at_one_not_refused() -> None:
    """The distinction the whole audit turns on.

    Every pair identical is a real observation whose answer is the largest p
    there is. That 1.0 is a measurement. The 1.0 this module refuses to emit is
    the one standing in for a test that never ran.
    """
    r = sign_flip_paired_test(list(np.ones(10)), list(np.ones(10)), metric="sentiment")
    assert r["tested"] is True
    assert r["p_value"] == 1.0
    assert "identical" in r["reason"]


def test_a_sampled_run_is_reproducible_and_uses_the_bias_corrected_estimator() -> None:
    ref = list(np.zeros(30))
    var = list(np.full(30, 0.3))
    a = sign_flip_paired_test(ref, var, n_resamples=999, seed=7)
    b = sign_flip_paired_test(ref, var, n_resamples=999, seed=7)
    assert a["p_value"] == b["p_value"]
    # (count + 1) / (B + 1) never returns 0: an estimated p of exactly zero
    # would claim more certainty than the resampling can support.
    assert a["p_value"] >= 1.0 / 1000.0


def test_enumeration_refuses_a_pair_count_it_cannot_enumerate() -> None:
    r = sign_flip_paired_test(list(np.zeros(21)), list(np.arange(21, dtype=float)))
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
    assert "2**21 sign vectors" in r["reason"]


def test_non_finite_values_are_dropped_pairwise_and_counted() -> None:
    ref = [0.0, 0.0, float("nan"), 0.0, float("inf")]
    var = [1.0, 1.0, 1.0, float("nan"), 1.0]
    usable_ref, usable_var, supplied = usable_numeric_pairs(ref, var)
    assert supplied == 5
    assert list(usable_ref) == [0.0, 0.0]
    assert list(usable_var) == [1.0, 1.0]

    r = sign_flip_paired_test(ref, var, metric="sentiment")
    assert r["n_pairs_supplied"] == 5
    assert r["n_pairs_usable"] == 2
    assert r["n_pairs_dropped"] == 3
    # Two pairs cannot reach 0.05, so this is refused rather than reported.
    assert r["tested"] is False
    assert np.isnan(r["p_value"])
