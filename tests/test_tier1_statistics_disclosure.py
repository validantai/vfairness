"""Four Tier-1 statistics defects, each pinned with its own control.

All four were found by executing the functions across input classes on 2026-09-25,
and all four share one shape: a confident answer produced where the statistic it
reports does not exist. They differ in what the honest answer is, and that
difference is the point. Two of them must REFUSE, and two of them must keep their
value and disclose what it does not establish. Turning either kind into the other
would be a defect in its own right.

  sequential_fairness_test   REFUSE the wrong verdict: the LLR was one-sided while
                             the question is two-sided, so a real disparity in the
                             unfavourable direction was reported as
                             "no meaningful difference detected".
  minimum_detectable_effect  REFUSE: it returned 1.0 for arms too small to compare,
                             under a comment reading "Cannot detect anything". 1.0
                             is finite, so every downstream isfinite() guard passed.
  power_warning              REFUSE: a power verdict from a subgroup larger than the
                             dataset, or a negative one. Returning None here means
                             ADEQUATE POWER.
  proportion_z_test          KEEP the value, DISCLOSE the degeneracy: with a pooled
                             proportion of 0 or 1 the two rates really are identical,
                             so 1.0 is a genuine observation; what was missing is
                             that the variance is zero, so it is not evidence a
                             difference would have been detected.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    minimum_detectable_effect,
    power_warning,
    proportion_z_test,
    sequential_fairness_test,
)


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


# ── sequential_fairness_test: the question is two-sided ───────────────────────


@pytest.mark.parametrize("n_per_arm", [60, 100, 400])
def test_the_sprt_verdict_does_not_depend_on_which_arm_is_passed_first(n_per_arm):
    """Before the fix, A=50%/B=0% gave accept_h0_no_difference (LLR -15.07) and the
    same data swapped gave reject_h0_groups_differ (LLR +13.07)."""
    a = np.asarray([1.0] * (n_per_arm // 2) + [0.0] * (n_per_arm - n_per_arm // 2))
    b = np.zeros(n_per_arm)
    forward, _ = _caught(lambda: sequential_fairness_test(a, b))
    backward, _ = _caught(lambda: sequential_fairness_test(b, a))
    assert forward["decision"] == backward["decision"]
    assert forward["logLikelihoodRatio"] == pytest.approx(backward["logLikelihoodRatio"], abs=1e-9)
    assert forward["decision"] == "reject_h0_groups_differ", (
        f"a {abs(a.mean() - b.mean()):.0%} gap was not detected: {forward}"
    )


def test_control_the_sprt_does_not_cry_wolf_on_a_true_null():
    """The other direction. A two-sided LLR is easier to push upward, so the null
    must still not be rejected."""
    rng = np.random.default_rng(11)
    rejects = 0
    for _ in range(40):
        out = sequential_fairness_test(rng.normal(0, 1, 300), rng.normal(0, 1, 300))
        rejects += out["decision"] == "reject_h0_groups_differ"
    # Wald's anytime bound at alpha=0.05, power=0.8 is alpha/(1-beta) = 0.0625.
    assert rejects <= 6, f"{rejects}/40 false rejects, above the Wald anytime bound"


# ── minimum_detectable_effect: no floor exists below two per arm ──────────────


@pytest.mark.parametrize("n1,n2", [(0, 100), (1, 100), (0, 0), (1, 1)])
def test_no_minimum_detectable_effect_is_invented_for_arms_too_small(n1, n2):
    value, msgs = _caught(lambda: minimum_detectable_effect(n1, n2))
    assert math.isnan(value), f"({n1}, {n2}) reported a detectable floor of {value}"
    assert any("could not check" in m for m in msgs), f"({n1}, {n2}) refused in silence"


def test_control_a_real_design_still_gets_its_floor():
    value, msgs = _caught(lambda: minimum_detectable_effect(200, 400))
    assert 0.0 < value < 1.0 and not msgs
    # Monotone in sample size, which a clamped constant would not be.
    bigger, _ = _caught(lambda: minimum_detectable_effect(800, 1600))
    assert bigger < value


# ── power_warning: None means adequate, so an impossible pair may not reach it ─


@pytest.mark.parametrize(
    "group,total",
    [(-5, 100), (250, 100), (300, 100), (30, 0), (30, -1)],
)
def test_no_power_verdict_from_a_pair_that_cannot_describe_a_dataset(group, total):
    verdict = power_warning(group, total)
    assert verdict is not None, (
        f"a subgroup of {group} in a dataset of {total} was cleared as adequate power"
    )
    assert "COULD NOT BE ASSESSED" in verdict


@pytest.mark.parametrize("group", [0, 1])
def test_a_subgroup_with_too_few_rows_is_could_not_check_not_low_power(group):
    """It used to read "Very low statistical power (n=0). This test can only detect
    disparities larger than 100%", a floor computed from the clamped 1.0."""
    verdict = power_warning(group, 100)
    assert verdict is not None and "COULD NOT BE ASSESSED" in verdict


def test_control_power_warning_still_grades_the_pairs_it_can():
    assert power_warning(250, 400) is None, "a genuinely powered subgroup was withheld"
    low = power_warning(50, 100)
    assert low is not None and "COULD NOT BE ASSESSED" not in low
    assert "statistical power" in low.lower()


# ── proportion_z_test: keep the number, disclose the zero variance ────────────


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_a_zero_variance_comparison_still_returns_its_value_and_says_so(p):
    value, msgs = _caught(lambda: proportion_z_test(p, 100, p, 100))
    assert value == 1.0, "the observation that the rates are identical was thrown away"
    assert any("no power at all" in m or "no z statistic is defined" in m for m in msgs), (
        "a comparison with zero variance returned 1.0 in silence"
    )


def test_control_a_measurable_comparison_discloses_nothing():
    """Identical rates with real variance have real power: nothing to disclose."""
    value, msgs = _caught(lambda: proportion_z_test(0.5, 100, 0.5, 100))
    assert value == pytest.approx(1.0, abs=1e-8)
    assert not msgs, f"a powered comparison emitted {msgs}"
    real, real_msgs = _caught(lambda: proportion_z_test(0.8, 100, 0.4, 100))
    assert real < 1e-6 and not real_msgs
