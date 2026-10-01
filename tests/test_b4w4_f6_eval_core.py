"""B4 wave 4, package F6-eval-core: the core fairness metrics every other surface quotes.

Each section below pins ONE overturned grade from the tier-1 audit package, and each
pin is paired with a control that asserts the healthy case's REAL number, because a
guard that refuses everything passes every refusal test.

A NOTE ON THE FIXTURES, because two of them are easy to get wrong.

ROUNDING NOISE NEEDS AN ARRAY THAT SPANS BINADES. ``np.spacing`` is constant inside
one binade, so ``(base + c) - base`` rounds identically for every element of
``base = 1e6 + np.arange(n)`` and the spread comes out EXACTLY 0.0, which the old
exact tests already caught. ``np.linspace(1e6, 1e8, n)`` crosses several powers of
two and the noise appears. Every such fixture here asserts its own ``np.ptp`` is
non-zero before it proves anything.

EVERY AXIS VARIES, INCLUDING n. A defect can hide behind a sample size nobody chose
deliberately, so the degeneracy fixtures run at n = 31, 40, 57 and 100 rather than at
one comfortable 40.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics import _statistics as S
from vfairness.evaluation.vfairness_metrics import regression as R
from vfairness.evaluation.vfairness_metrics._statistics import cohens_d
from vfairness.exceptions import InvalidDataError

DEGENERACY_SIZES = [31, 40, 57, 100]


def _messages(caught):
    return [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# regression.compute_regression_effect_sizes
#
# The operand-scaled degeneracy tolerance reached the RESIDUAL arm only. The
# `_DEGENERATE_DENOMINATOR_D = 1e3` band added on 2026-09-30 catches the part of the
# prediction arm's fabrication that is absurd on its face (|d| of tens of millions),
# and a SMALLER between-group gap over the same degenerate denominator lands inside
# the conventional bands, where it was graded "large" in silence.
# ---------------------------------------------------------------------------


def _degenerate_prediction_pair(n: int):
    """Two prediction arrays each constant to within the resolution of 1e8 arithmetic.

    The model's internal terms run at 1e6..1e8, where one ulp is 1.49e-8, so each
    group's prediction is 0.1 and 0.1000001 constant to about 7.5e-9. The gap between
    the groups is 1e-7, small enough that the fabricated Cohen's d is about 41 and the
    1e3 band cannot reach it. y_true is recorded at the magnitude the arithmetic ran
    at, which is what the operand scale is derived from.
    """
    base = np.linspace(1e6, 1e8, n)
    p1 = (base + 0.1) - base
    p2 = (base + 0.1000001) - base
    assert np.ptp(p1) > 0.0, "the fixture is exactly constant, so it proves nothing"
    assert np.ptp(p2) > 0.0, "the fixture is exactly constant, so it proves nothing"
    y_pred = np.concatenate([p1, p2])
    y_true = np.concatenate([base, base])
    return y_true, y_pred, np.array(["A"] * n + ["B"] * n)


@pytest.mark.parametrize("n", DEGENERACY_SIZES)
def test_the_prediction_arm_refuses_a_denominator_that_is_rounding_noise(n):
    y_true, y_pred, attr = _degenerate_prediction_pair(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = R.compute_regression_effect_sizes(y_true, y_pred, attr)
    row = out["A_vs_B"]
    assert not np.isfinite(row["cohens_d_predictions"]), (
        "before this fix the prediction arm published a finite -40.9 here, from two "
        "arrays each constant to 7.5e-9, and graded it 'large'"
    )
    assert "not interpretable" in row["interpretation"], row["interpretation"]
    assert row["interpretation"] != "large"
    assert any("standardising denominator is zero" in m for m in _messages(caught)), _messages(
        caught
    )


@pytest.mark.parametrize("n", DEGENERACY_SIZES)
def test_the_residual_arm_still_refuses_what_it_was_fixed_for(n):
    """SIBLING CONTROL. Moving the tolerance above the arm selection must not cost the
    residual arm the refusal it was given on 2026-09-29."""
    rng = np.random.default_rng(0)
    pred = rng.normal(1e6, 1e4, n)
    y_pred = np.concatenate([pred, pred])
    y_true = np.concatenate([pred + 0.0, pred + 50.0])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = R.compute_regression_effect_sizes(y_true, y_pred, np.array(["A"] * n + ["B"] * n))
    assert not np.isfinite(out["A_vs_B"]["cohens_d_residuals"])


def test_control_a_real_regression_effect_is_still_measured_and_silent():
    """OVER-CORRECTION CONTROL. The REAL number, not merely "it did not raise"."""
    n, rng = 40, np.random.default_rng(3)
    pred = np.concatenate([rng.normal(100, 10, n), rng.normal(120, 10, n)])
    y_true = pred + rng.normal(0.0, 2.0, 2 * n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = R.compute_regression_effect_sizes(y_true, pred, np.array(["A"] * n + ["B"] * n))
    row = out["A_vs_B"]
    assert row["cohens_d_predictions"] == pytest.approx(-1.7184, abs=1e-4)
    assert row["interpretation"] == "large"
    assert not caught, _messages(caught)


def test_a_caller_tolerance_is_a_floor_and_never_narrows_the_refusal():
    """The MECHANISM the fix above rests on, pinned in its own right.

    ``constant_atol`` used to REPLACE cohens_d's relative bound. Four ulps at a
    magnitude is about 9e-16 of it and the default is 1e-12 of it, so a caller trying
    to LOOSEN the test made it a thousand times stricter whenever its own arrays sat
    at the operand magnitude, and quietly removed coverage the function already had.
    """
    n = 40
    # Two arrays at 1e8 magnitude, constant to a spread the DEFAULT relative bound
    # (1e-12 * 1e8 = 1e-4) calls noise and a 4-ulp operand bound (5.96e-8) does not,
    # separated by 1.0, which is far outside both. MEASURED: ptp 9.9987e-06, default
    # atol 1.0000e-04, operand bound 5.9605e-08.
    g1 = 1e8 + (np.arange(n) % 2) * 1e-5
    g2 = g1 + 1.0
    assert 5.96e-8 < np.ptp(g1) < 1e-4, np.ptp(g1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        without = cohens_d(g1, g2)
        with_operand_bound = cohens_d(g1, g2, constant_atol=4.0 * float(np.spacing(1e8)))
    assert not np.isfinite(without), without
    assert not np.isfinite(with_operand_bound), (
        "a caller-supplied tolerance narrowed the refusal instead of widening it, so "
        f"passing one turned a refusal into the finite {with_operand_bound}"
    )


# ---------------------------------------------------------------------------
# regression: R² at both of its sites. FINITENESS IS NOT MEASURABILITY.
#
# `ss_tot == 0` (r2_parity_difference) and `ss_tot > 0` (get_group_metrics) are exact
# comparisons of an ACCUMULATED sum of squares against zero. A target constant only
# up to the resolution of the arithmetic that produced it walks past both.
# ---------------------------------------------------------------------------


def _near_constant_target(n: int):
    """A target of 100.1 carrying 7.5e-9 of rounding inherited from 1e8 arithmetic."""
    base = np.linspace(1e6, 1e8, n)
    yt_a = (base + 100.1) - base
    assert np.ptp(yt_a) > 0.0, "the fixture is exactly constant, so the old test caught it"
    assert np.sum((yt_a - yt_a.mean()) ** 2) > 0.0, "ss_tot is exactly zero, proves nothing"
    rng = np.random.default_rng(1)
    yt_b = 100.1 + rng.normal(0, 10, n)
    y_true = np.concatenate([yt_a, yt_b])
    y_pred = np.concatenate([np.full(n, 130.0), yt_b + rng.normal(0, 3, n)])
    return y_true, y_pred, np.array(["a"] * n + ["b"] * n)


@pytest.mark.parametrize("n", DEGENERACY_SIZES)
def test_get_group_metrics_refuses_an_r2_whose_denominator_is_resolution(n):
    y_true, y_pred, attr = _near_constant_target(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.get_group_metrics(y_true, y_pred, attr)
    assert not np.isfinite(got["a"]["r2"]), (
        "before this fix this published a finite -8.02e+19 with no warning naming the "
        "group, and every isfinite guard downstream passed it"
    )
    assert np.isfinite(got["b"]["r2"]), "the measured group must keep its score"
    named = [m for m in _messages(caught) if "R² is undefined for group(s)" in m]
    assert named, _messages(caught)
    assert "'a'" in named[0], named[0]
    # The could-not-check states WHICH quantity is degenerate, where a reader looks.
    assert "resolution of the numbers" in named[0], named[0]
    # Every other metric in that row is a real measurement and is not swept away.
    assert got["a"]["mae"] == pytest.approx(29.9, abs=0.2), got["a"]["mae"]


@pytest.mark.parametrize("n", DEGENERACY_SIZES)
def test_r2_parity_difference_refuses_the_same_target(n):
    y_true, y_pred, attr = _near_constant_target(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.r2_parity_difference(y_true, y_pred, attr)
    assert not np.isfinite(got), (
        "before this fix the parity gap was published as 8.02e+19 with zero warnings"
    )
    named = [m for m in _messages(caught) if "R² is undefined for group(s)" in m]
    assert named, _messages(caught)
    assert "NOTHING WAS COMPARED" in named[0], named[0]


@pytest.mark.parametrize("n", DEGENERACY_SIZES)
def test_a_perfect_r2_is_refused_when_there_was_no_variance_to_explain(n):
    """THE AXIS THE MAGNITUDE BAND CANNOT SEE, and the reason the spread tolerance is
    not redundant with it.

    A target constant to within its OWN last bits, predicted EXACTLY, gives
    ss_res = 0 and an ss_tot that is minute and not zero, so the old `ss_tot > 0`
    test passed and R² came out as 1.0: a PERFECT fit published for a group with no
    variance to explain. |R²| = 1.0 is nowhere near the 1e6 band, so only the spread
    tolerance can refuse it.

    Measured with the tolerance reverted to the old exact `spread <= 0.0`:
    r2[a] = 1.0 and r2_parity_difference = 0.0853 at n = 40, in silence.
    """
    yt_a = 100.1 + (np.arange(n) % 2) * np.spacing(100.1)
    assert np.ptp(yt_a) > 0.0, "the fixture is exactly constant, so it proves nothing"
    assert np.sum((yt_a - yt_a.mean()) ** 2) > 0.0, "ss_tot is exactly zero, proves nothing"
    rng = np.random.default_rng(2)
    yt_b = 100.1 + rng.normal(0, 10, n)
    y_true = np.concatenate([yt_a, yt_b])
    y_pred = np.concatenate([yt_a.copy(), yt_b + rng.normal(0, 3, n)])
    attr = np.array(["a"] * n + ["b"] * n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.get_group_metrics(y_true, y_pred, attr)
        parity = R.r2_parity_difference(y_true, y_pred, attr)
    assert not np.isfinite(got["a"]["r2"]), (
        f"a perfect R² of {got['a']['r2']} was published for a target with no variance"
    )
    assert not np.isfinite(parity), parity
    assert np.isfinite(got["b"]["r2"]), "the measured group must keep its score"
    # The MAE of that group really is 0.0, and that IS a measurement: the refusal is
    # about R² only and must not swallow the rest of the row.
    assert got["a"]["mae"] == pytest.approx(0.0, abs=1e-12), got["a"]["mae"]
    named = [m for m in _messages(caught) if "R² is undefined for group(s)" in m]
    assert named and "no variance to explain" in named[0], _messages(caught)


def test_control_two_healthy_groups_still_get_their_real_r2_and_gap():
    """OVER-CORRECTION CONTROL for both R² sites, asserting the numbers."""
    n, rng = 60, np.random.default_rng(7)
    yt_a = 100.0 + rng.normal(0, 10, n)
    yt_b = 100.0 + rng.normal(0, 10, n)
    y_true = np.concatenate([yt_a, yt_b])
    y_pred = np.concatenate([yt_a + rng.normal(0, 3, n), yt_b + rng.normal(0, 6, n)])
    attr = np.array(["a"] * n + ["b"] * n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.get_group_metrics(y_true, y_pred, attr)
        gap = R.r2_parity_difference(y_true, y_pred, attr)
    # MEASURED, not assumed.
    assert got["a"]["r2"] == pytest.approx(0.899813, abs=1e-4), got["a"]["r2"]
    assert got["b"]["r2"] == pytest.approx(0.528796, abs=1e-4), got["b"]["r2"]
    assert gap == pytest.approx(0.371017, abs=1e-4), gap
    assert not caught, _messages(caught)


def test_control_an_exactly_constant_target_keeps_its_own_refusal():
    """The 2026-09-27 fix must survive: an exactly constant target is still NaN and
    still named, and the reason it is given is the constant target rather than the
    magnitude band."""
    n, rng = 40, np.random.default_rng(5)
    yt_b = 100.0 + rng.normal(0, 10, n)
    y_true = np.concatenate([np.full(n, 100.0), yt_b])
    y_pred = np.concatenate([np.full(n, 130.0), yt_b + rng.normal(0, 3, n)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.get_group_metrics(y_true, y_pred, np.array(["A"] * n + ["B"] * n))
    assert not np.isfinite(got["A"]["r2"])
    assert got["A"]["mae"] == pytest.approx(30.0)
    named = [m for m in _messages(caught) if "R² is undefined for group(s)" in m]
    assert named and "no variance to explain" in named[0], _messages(caught)


def test_control_a_genuinely_bad_model_keeps_its_negative_r2():
    """A real model can score below zero, and that is a MEASUREMENT. The band must not
    swallow it."""
    n, rng = 50, np.random.default_rng(11)
    yt = 100.0 + rng.normal(0, 10, n)
    y_true = np.concatenate([yt, yt])
    # Group A predicted with an error three times the target's own spread: a genuinely
    # awful but entirely measurable fit.
    y_pred = np.concatenate([yt + rng.normal(0, 30, n), yt + rng.normal(0, 3, n)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = R.get_group_metrics(y_true, y_pred, np.array(["A"] * n + ["B"] * n))
    assert np.isfinite(got["A"]["r2"]) and got["A"]["r2"] < 0.0, got["A"]["r2"]
    assert got["A"]["r2"] == pytest.approx(-10.707276, abs=1e-4), got["A"]["r2"]
    assert got["B"]["r2"] == pytest.approx(0.884018, abs=1e-4), got["B"]["r2"]
    assert not caught, _messages(caught)


# ---------------------------------------------------------------------------
# _statistics.minimum_detectable_effect
#
# The unit's only input gate was the bare `n1 < 2 or n2 < 2`. A bare comparison
# lets an unmeasurable arm through, and because 1/inf is 0 an infinite arm makes
# the floor SMALLER, so the non-finite branch below it can never rescue it and the
# design reads as BETTER POWERED than a real one. Nothing range-checked alpha,
# power or baseline_rate.
# ---------------------------------------------------------------------------

REAL_FLOOR_100_1000 = 0.14691636828297927


@pytest.mark.parametrize(
    "n1,n2",
    [
        (float("inf"), 100),
        (100, float("inf")),
        (float("nan"), 100),
        (2.5, 100),
        (100, 2.5),
        (30.5, 500),
    ],
    ids=["inf_first_arm", "inf_second_arm", "nan_arm", "half_a_row", "half_a_row_2nd", "30_5"],
)
def test_no_floor_is_published_for_an_arm_that_is_not_a_whole_finite_count(n1, n2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = S.minimum_detectable_effect(n1, n2)
    assert np.isnan(got), (
        f"minimum_detectable_effect({n1}, {n2}) published {got}. An infinite arm used "
        f"to publish 0.14007926090564843, BELOW the real 0.14691636828297927 of "
        f"(100, 1000), so an unassessable design read as better powered than a real one"
    )
    assert any("COULD NOT CHECK" in m for m in _messages(caught)), _messages(caught)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 1.0},
        {"alpha": 0.0},
        {"alpha": -0.1},
        {"power": 1.0},
        {"power": 0.0},
        {"baseline_rate": 1.5},
        {"baseline_rate": 0.0},
        {"baseline_rate": 1.0},
        {"alpha": float("nan")},
    ],
)
def test_no_floor_is_published_for_a_parameter_that_is_not_a_probability(kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = S.minimum_detectable_effect(100, 1000, **kwargs)
    assert np.isnan(got), (
        f"minimum_detectable_effect(100, 1000, {kwargs}) published {got}; alpha=1.0 "
        f"used to publish 0.04413498982895752 against the real "
        f"{REAL_FLOOR_100_1000}, a threefold understatement, in silence"
    )
    assert any("COULD NOT CHECK" in m for m in _messages(caught)), _messages(caught)


def test_an_out_of_range_baseline_never_leaks_a_numpy_sqrt_warning():
    """baseline_rate=1.5 was refused only BY ACCIDENT, through np.sqrt of a negative,
    and numpy's own RuntimeWarning reached the caller before this unit said anything.
    The gate is above the arithmetic now, so the sqrt is never reached."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = S.minimum_detectable_effect(100, 1000, baseline_rate=1.5)
    assert np.isnan(got)
    assert not any("sqrt" in m for m in _messages(caught)), _messages(caught)
    assert not any(w.category is RuntimeWarning for w in caught), [
        w.category.__name__ for w in caught
    ]


def test_control_real_designs_keep_their_exact_floors_and_stay_silent():
    """OVER-CORRECTION CONTROL. The exact floors, and the ordering a clamp could not
    have, so a gate that refused everything cannot pass this."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = S.minimum_detectable_effect(3, 100)
        b = S.minimum_detectable_effect(30, 100)
        c = S.minimum_detectable_effect(100, 1000)
        d = S.minimum_detectable_effect(2, 1000)
        e = S.minimum_detectable_effect(100, 100, alpha=0.01, power=0.9)
    assert a == pytest.approx(0.8207895653160011)
    assert b == pytest.approx(0.2915982346576338)
    assert c == pytest.approx(REAL_FLOOR_100_1000)
    assert d == pytest.approx(0.9914999680923966)
    assert e == pytest.approx(0.27275801701552727)
    assert a > b > c, "the floor must fall as the arms grow"
    assert not caught, _messages(caught)


def test_control_a_numpy_integer_arm_is_a_measurement_not_a_refusal():
    """np.int64 is not a Python int subclass, so an isinstance(v, (int, float)) gate
    here would have discarded real evidence while reading as caution. is_measured is
    the repo-wide predicate and it names numpy's scalars."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = S.minimum_detectable_effect(np.int64(100), np.int32(1000))
        also = S.minimum_detectable_effect(np.float64(100.0), 1000)
    assert got == pytest.approx(REAL_FLOOR_100_1000), got
    assert also == pytest.approx(REAL_FLOOR_100_1000), also
    assert not caught, _messages(caught)


# ---------------------------------------------------------------------------
# _statistics.power_warning
#
# None from this function means ADEQUATE POWER. Its second arm was the WHOLE
# DATASET rather than the complement, so the subgroup was counted on both sides,
# and the `group_size >= 200` shortcut sat ABOVE the computation, so every guard
# the earlier waves added sat below it and could not fire.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group,total,complement",
    [(200, 200, 0), (250, 250, 0), (500, 500, 0), (199, 200, 1), (100, 101, 1), (2000, 2001, 1)],
)
def test_no_adequate_power_verdict_when_there_is_nothing_to_compare_against(
    group, total, complement
):
    """(200, 200) returned None, i.e. ADEQUATE POWER, for a dataset whose comparison
    arm has ZERO rows, because the >= 200 shortcut ran before anything was computed.
    """
    verdict = S.power_warning(group, total)
    assert verdict is not None, (
        f"power_warning({group}, {total}) cleared a subgroup whose comparison arm has "
        f"{complement} row(s) as adequately powered"
    )
    assert "COULD NOT BE ASSESSED" in verdict, verdict
    assert f"other {complement} row(s)" in verdict, verdict


def test_the_published_floor_is_computed_over_the_complement_not_the_whole():
    """The sharpest case. (199, 201) claimed it could detect a 14% disparity while the
    honest floor over the real comparison arm of TWO rows is 99.5%."""
    honest = S.minimum_detectable_effect(199, 2)
    assert honest == pytest.approx(0.9954749461789353), honest
    verdict = S.power_warning(199, 201)
    assert verdict is not None
    assert "larger than 14%" not in verdict, verdict
    assert "larger than 99.5%" in verdict, verdict


def test_a_non_finite_total_is_a_could_not_check_and_not_a_graded_band():
    """The coherence test is three BARE comparisons and inf answers all three the safe
    way, so power_warning(100, inf) printed a 14% floor in silence."""
    verdict = S.power_warning(100, float("inf"))
    assert verdict is not None and "COULD NOT BE ASSESSED" in verdict, verdict
    assert "larger than" not in verdict, verdict
    assert S.power_warning(float("nan"), 100) is not None


@pytest.mark.parametrize(
    "group,total",
    [(None, 100), (100, None), ("100", 1000), (True, 100), (100, True)],
    ids=["none_group", "none_total", "string_group", "bool_group", "bool_total"],
)
def test_an_unmeasured_pair_is_refused_in_words_and_never_raises(group, total):
    """THE CASE THE GATE IS LOAD-BEARING FOR, and the reason it is not redundant with
    the arm gate in minimum_detectable_effect.

    A non-finite total IS caught downstream, because the producer refuses it and this
    function turns that nan into the could-not-check sentence. A NON-NUMERIC one is
    not: `None < 0` raises TypeError out of the coherence test before any producer is
    reached, and an exception out of a power advisory is not one of the three states.
    A bool is the other door: `True < 0` is a perfectly good comparison and True
    would have been read as a subgroup of one row.
    """
    verdict = S.power_warning(group, total)
    assert verdict is not None, (group, total)
    assert "COULD NOT BE ASSESSED" in verdict, verdict
    assert "not a measured pair of row counts" in verdict, verdict


def test_control_power_warning_still_grades_and_still_clears_what_it_should():
    """OVER-CORRECTION CONTROL, with the sentences a reader gets.

    A fix that refused everything would pass every test above and destroy the one
    verdict this function exists to give.
    """
    assert S.power_warning(300, 1000) is None, "a genuinely powered subgroup was withheld"
    assert S.power_warning(250, 400) is None
    assert S.power_warning(199, 100000) is None
    three = S.power_warning(3, 100)
    assert "Very low statistical power (n=3)" in three and "larger than 82%" in three
    assert "COULD NOT BE ASSESSED" not in three
    hundred = S.power_warning(100, 1000)
    assert "Limited statistical power (n=100)" in hundred
    assert "larger than 15%" in hundred
    thirty = S.power_warning(30, 100)
    assert "larger than 31%" in thirty, thirty
    # The impossible pairs the earlier waves fixed still read the same way.
    for group, total in ((-5, 100), (250, 100), (30, 0), (30, -1)):
        assert "COULD NOT BE ASSESSED" in S.power_warning(group, total)
    for group in (0, 1, 2):
        assert "COULD NOT BE ASSESSED" in S.power_warning(group, 100)


# ---------------------------------------------------------------------------
# _statistics.bootstrap_over_index / stratified_bootstrap_ci
#
# A LENGTH CHECK IS NOT A COVERAGE CHECK. A groups array of exactly n entries whose
# strata leave rows out walked past both entry points' length checks, and a missing
# group label is exactly that: np.unique LISTS nan as a level while `groups == nan`
# matches nothing, so those rows are deleted from every replicate while the point
# estimate still uses all of them.
# ---------------------------------------------------------------------------


def _mean_of_rows(n: int):
    data = np.arange(float(n))
    return data, (lambda idx: float(np.mean(data[idx])))


@pytest.mark.parametrize("n", [20, 31, 40, 57, 100])
def test_strata_that_do_not_cover_every_row_are_refused(n):
    """Before this fix, at every one of these n, the published interval did NOT contain
    its own published point estimate and nothing said so.

    Measured: n=20 point 9.5 interval [5.6000, 8.2017]; n=31 15.0 [9.8313, 12.8750];
    n=40 19.5 [12.7983, 16.1350]; n=57 28.0 [19.0000, 23.1180]; n=100 49.5
    [34.1050, 39.4020]. sample_size still reported the full n and there were zero
    UserWarnings and no not-measured field of any kind.
    """
    _, stat = _mean_of_rows(n)
    half = n // 2
    tail = half // 2
    groups = np.array([0.0] * half + [1.0] * (n - half - tail) + [np.nan] * tail)
    assert len(groups) == n, "the fixture must pass the LENGTH check to test COVERAGE"
    assert len(np.unique(groups)) == 3, "nan must be LISTED as a level for this to bite"
    with pytest.raises(InvalidDataError) as excinfo:
        S.bootstrap_over_index(n, stat, groups, n_bootstrap=200, random_state=3)
    message = str(excinfo.value)
    assert "belong to no stratum" in message, message
    assert f"{tail} of {n} row(s)" in message, message
    assert "bootstrap_over_index" in message, message


def test_the_stratified_entry_point_refuses_the_same_coverage_gap_itself():
    """The guard sits where the strata are FORMED as well as at the outer entry point,
    so a caller that reaches stratified_bootstrap_ci directly is not left out."""
    data = np.arange(20.0)
    groups = np.array([0.0] * 15 + [np.nan] * 5)
    with pytest.raises(InvalidDataError) as excinfo:
        S.stratified_bootstrap_ci(
            data, groups, lambda d, _g: float(np.mean(d)), n_bootstrap=200, random_state=3
        )
    assert "stratified_bootstrap_ci" in str(excinfo.value)
    assert "belong to no stratum" in str(excinfo.value)


@pytest.mark.parametrize(
    "groups,label",
    [
        (np.array(["a"] * 15 + [None] * 5, dtype=object), "None in an object array"),
        (np.array(["a"] * 15 + [np.nan] * 5, dtype=object), "nan in an object array"),
    ],
    ids=["object_none", "object_nan"],
)
def test_an_unsortable_label_array_is_refused_and_not_a_bare_type_error(groups, label):
    """These two raised `TypeError: '<' not supported between instances of 'NoneType'
    and 'str'` out of numpy's own sort. An exception from the sort is not one of the
    three states either, and it names nothing a caller can act on."""
    _, stat = _mean_of_rows(20)
    with pytest.raises(InvalidDataError) as excinfo:
        S.bootstrap_over_index(20, stat, groups, n_bootstrap=200, random_state=3)
    assert "cannot be sorted into strata at all" in str(excinfo.value), label


def test_a_pandas_na_label_is_refused_rather_than_raising_on_its_own_truthiness():
    """pd.NA raised `TypeError: boolean value of NA is ambiguous`, and `x != x` cannot
    be used to find it, which is why the absence test goes through pandas.isna."""
    pd = pytest.importorskip("pandas")
    _, stat = _mean_of_rows(20)
    groups = pd.array(["a"] * 15 + [pd.NA] * 5, dtype="string")
    with pytest.raises(InvalidDataError) as excinfo:
        S.bootstrap_over_index(20, stat, groups, n_bootstrap=200, random_state=3)
    assert "cannot be sorted into strata at all" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("n", [20, 40, 57])
def test_control_clean_strata_still_get_their_real_interval(n):
    """OVER-CORRECTION CONTROL, with the real numbers. A guard that refused every
    groups array would pass every test above."""
    _, stat = _mean_of_rows(n)
    groups = np.array([0.0] * (n // 2) + [1.0] * (n - n // 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = S.bootstrap_over_index(n, stat, groups, n_bootstrap=200, random_state=3)
    assert got.point_estimate == pytest.approx((n - 1) / 2.0)
    assert got.lower_bound <= got.point_estimate <= got.upper_bound, (
        got.lower_bound,
        got.point_estimate,
        got.upper_bound,
    )
    assert got.sample_size == n
    assert not caught, _messages(caught)


def test_control_the_exact_interval_of_the_twenty_row_clean_case():
    """The measured numbers, so a guard that quietly widened or narrowed the interval
    would show up here rather than passing as 'it did not raise'."""
    _, stat = _mean_of_rows(20)
    groups = np.array([0.0] * 10 + [1.0] * 10)
    got = S.bootstrap_over_index(20, stat, groups, n_bootstrap=200, random_state=3)
    assert got.point_estimate == pytest.approx(9.5)
    assert got.lower_bound == pytest.approx(8.25, abs=1e-6)
    assert got.upper_bound == pytest.approx(10.6013, abs=1e-3)
    assert got.standard_error == pytest.approx(0.646944, abs=1e-5)


def test_control_a_blank_or_literal_none_label_is_a_stratum_and_keeps_its_rows():
    """DELIBERATE, and recorded so it is not mistaken for an oversight.

    A blank string and the literal string 'None' are ugly category names, not absent
    values: `groups == ''` matches its rows, so every row still belongs to a stratum
    and the interval still covers all of them. Refusing them would delete a
    legitimate, if badly named, category. The defect this guard closes is rows being
    DROPPED, and these are not dropped.
    """
    _, stat = _mean_of_rows(20)
    for tail in ("", "None"):
        groups = np.array(["a"] * 15 + [tail] * 5, dtype=object)
        got = S.bootstrap_over_index(20, stat, groups, n_bootstrap=200, random_state=3)
        assert got.lower_bound <= got.point_estimate <= got.upper_bound, tail
        assert got.point_estimate == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# analyzer.FairnessAnalyzer.compare_with_aequitas / compare_with_fairlearn
#
# Both guards covered len(valid) < 2 only. With TWO OR MORE comparable groups the
# reference library was handed ALL of self.sensitive_attr, including the groups the
# analyzer itself REFUSES to compare, and the result was published verbatim with no
# not_comparable key, no warning and nothing naming the excluded group or the
# threshold that excluded it.
# ---------------------------------------------------------------------------

_POPULATION_SHAPES = [(100, 100, 3, 50), (60, 60, 5, 50), (200, 150, 10, 50), (40, 40, 7, 30)]


class _StubGroup:
    """Records the frame it is handed; the assertions about the POPULATION do not
    depend on aequitas' own arithmetic and hold for any faithful stub."""

    seen: dict = {}

    def get_crosstabs(self, df, attr_cols=None):
        import pandas as pd

        type(self).seen = {
            "groups": sorted(df["sensitive_attr"].unique()),
            "counts": df["sensitive_attr"].value_counts().to_dict(),
        }
        return pd.DataFrame({"attribute_value": sorted(df["sensitive_attr"].unique())}), None


class _StubBias:
    def get_disparity_predefined_groups(
        self, xtab, original_df=None, ref_groups_dict=None, alpha=0.05
    ):
        import pandas as pd

        ref = ref_groups_dict["sensitive_attr"]
        rates = original_df.groupby("sensitive_attr")["score"].mean()
        values = sorted(original_df["sensitive_attr"].unique())
        return pd.DataFrame(
            {
                "attribute_value": values,
                "ppr_disparity": [
                    (rates[v] / rates[ref] if rates[ref] else float("nan")) for v in values
                ],
            }
        )


@pytest.fixture
def stub_aequitas(monkeypatch):
    import sys
    import types

    aequitas = types.ModuleType("aequitas")
    group = types.ModuleType("aequitas.group")
    bias = types.ModuleType("aequitas.bias")
    group.Group = _StubGroup
    bias.Bias = _StubBias
    aequitas.group, aequitas.bias = group, bias
    monkeypatch.setitem(sys.modules, "aequitas", aequitas)
    monkeypatch.setitem(sys.modules, "aequitas.group", group)
    monkeypatch.setitem(sys.modules, "aequitas.bias", bias)


def _three_group_analyzer(n_a, n_b, n_c, floor):
    from vfairness import FairnessAnalyzer

    rng = np.random.default_rng(4)
    pred = np.concatenate(
        [rng.integers(0, 2, n_a), rng.integers(0, 2, n_b), np.ones(n_c, dtype=int)]
    )
    attr = np.array(["a"] * n_a + ["b"] * n_b + ["c"] * n_c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(pred.copy(), pred, attr, min_group_size=floor)


@pytest.mark.parametrize("n_a,n_b,n_c,floor", _POPULATION_SHAPES)
def test_aequitas_is_not_asked_about_a_group_the_analyzer_refuses(
    stub_aequitas, n_a, n_b, n_c, floor
):
    analyzer = _three_group_analyzer(n_a, n_b, n_c, floor)
    valid = analyzer._group_manager.get_valid_groups(warn_if_empty=False)
    assert sorted(valid) == ["a", "b"], valid
    assert len(valid) >= 2, "the fixture must take the >= 2 branch, not the refusal branch"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyzer.compare_with_aequitas()
    assert "error" not in out, out
    assert _StubGroup.seen["groups"] == ["a", "b"], (
        f"the frame handed to aequitas still held {_StubGroup.seen['groups']}, so the "
        f"table carries a disparity for a group the analyzer will not compare"
    )
    assert "c" not in list(out["attribute_value"].values()), out["attribute_value"]
    assert out["groups_not_measured"] == {"c": n_c}, out.get("groups_not_measured")
    assert str(floor) in out["population_note"], out["population_note"]
    assert any("below min_group_size" in m for m in _messages(caught)), _messages(caught)


@pytest.mark.parametrize("n_a,n_b,n_c,floor", _POPULATION_SHAPES)
def test_the_fairlearn_twin_is_computed_over_the_assessed_population(n_a, n_b, n_c, floor):
    """THE SIBLING DOOR, and the worse of the two: fairlearn's
    demographic_parity_difference is a max-min SPREAD, so the refused group can set
    the whole number on its own. Measured at 100/100/3 before this change:
    0.5, entirely from the 3-row group, against 0.04 over the assessed population,
    with zero warnings."""
    pytest.importorskip("fairlearn")
    analyzer = _three_group_analyzer(n_a, n_b, n_c, floor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyzer.compare_with_fairlearn()
    assert "error" not in out, out
    assert out["groups_not_measured"] == {"c": n_c}, out.get("groups_not_measured")
    assessed = out["fairlearn_demographic_parity_difference"]
    all_groups = out["fairlearn_reported_all_groups_demographic_parity_difference"]
    assert assessed < all_groups, (assessed, all_groups)
    # The evidence is kept, not dropped: the all-groups figure is still there.
    assert all_groups > 0.4, all_groups
    # ... and it agrees with the analyzer's OWN metric over the same population.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = analyzer.compute_all_metrics()["demographic_parity_difference"]
    assert assessed == pytest.approx(ours, abs=1e-9), (ours, assessed)
    assert any("below min_group_size" in m for m in _messages(caught)), _messages(caught)


def test_control_a_frame_with_nothing_excluded_is_untouched_and_silent(stub_aequitas):
    """OVER-CORRECTION CONTROL for both methods, with the real ratio.

    Group a is selected 4/5 of the time and group b 2/5, so the ratio for b is 0.5
    exactly, nothing is withheld, no population note is added and nothing warns.
    """
    from vfairness import FairnessAnalyzer

    pred = np.array([1, 1, 1, 1, 0] * 6 + [1, 1, 0, 0, 0] * 6)
    attr = np.array(["a"] * 30 + ["b"] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = FairnessAnalyzer(pred.copy(), pred, attr)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyzer.compare_with_aequitas()
        fl = analyzer.compare_with_fairlearn()
    assert out["ppr_disparity"][1] == pytest.approx(0.5)
    assert out["ppr_disparity"][0] == pytest.approx(1.0)
    assert "groups_not_measured" not in out
    assert "population_note" not in out
    assert "groups_not_measured" not in fl
    assert fl["fairlearn_demographic_parity_difference"] == pytest.approx(0.4)
    assert not _messages(caught), _messages(caught)


# ---------------------------------------------------------------------------
# analyzer.FairnessAnalyzer.get_explanations
#
# get_explanations() returns report['explanations'] and NOTHING ELSE. The
# data-provenance clause was computed correctly and restated on
# assessment['summary'], one key over, so the surface this method returns carried a
# clean bill of health for a frame a third of whose rows had no protected
# attribute. And data_info['coverage'] was the constant 1.0.
# ---------------------------------------------------------------------------


def _analyzer_with_unlabelled_rows(n_a, n_b, n_missing, seed=0):
    from vfairness import FairnessAnalyzer

    rng = np.random.default_rng(seed)
    n = n_a + n_b + n_missing
    y = rng.integers(0, 2, n)
    attr = np.array(["a"] * n_a + ["b"] * n_b + [np.nan] * n_missing, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y, y.copy(), attr)


@pytest.mark.parametrize(
    "n_a,n_b,n_missing",
    [(40, 40, 40), (40, 35, 5), (40, 40, 120), (60, 55, 25)],
    ids=["a_third_unlabelled", "6_percent", "60_percent", "a_fifth"],
)
def test_get_explanations_states_the_provenance_its_sibling_already_computed(n_a, n_b, n_missing):
    analyzer = _analyzer_with_unlabelled_rows(n_a, n_b, n_missing)
    total = n_a + n_b + n_missing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = analyzer.get_explanations()["summary"]
    assert "data provenance" in summary, (
        "the only surface get_explanations returns said 'All assessable metrics are "
        f"within acceptable thresholds. Continue regular monitoring' while {n_missing} "
        f"of {total} rows had no protected attribute: {summary}"
    )
    assert f"{n_a + n_b} of {total} rows assessed" in summary, summary
    assert f"{n_missing} excluded for missing values" in summary, summary


@pytest.mark.parametrize(
    "n_a,n_b,n_missing",
    [(40, 40, 40), (40, 35, 5), (40, 40, 120), (60, 55, 25), (60, 60, 0)],
)
def test_data_info_coverage_is_the_measured_fraction_and_not_a_constant(n_a, n_b, n_missing):
    """It was 1.0 for every one of these, so it could not disagree with anything."""
    analyzer = _analyzer_with_unlabelled_rows(n_a, n_b, n_missing)
    total = n_a + n_b + n_missing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = analyzer.get_report()["data_info"]
    assert info["coverage"] == pytest.approx((n_a + n_b) / total), info["coverage"]
    assert info["original_size"] == total
    assert info["n_excluded"] == n_missing
    assert info["final_size"] == n_a + n_b


def test_a_group_that_lost_every_row_is_named_in_the_published_report():
    """The same copy-constructor omission, one field over, and louder: the artifact
    asserted that no group was dropped while a whole protected group had lost every
    row. Measured before: n_groups_before 2 and groups_dropped [] against the truth
    of 3 and ['c']."""
    from vfairness import FairnessAnalyzer

    n, rng = 40, np.random.default_rng(0)
    y = rng.integers(0, 2, 3 * n).astype(float)
    pred = y.copy()
    pred[2 * n :] = np.nan
    attr = np.array(["a"] * n + ["b"] * n + ["c"] * n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = FairnessAnalyzer(y, pred, attr).get_report()["data_info"]
    assert info["groups_dropped"] == ["c"], info["groups_dropped"]
    assert info["n_groups_before"] == 3, info["n_groups_before"]
    assert info["n_groups_after"] == 2, info["n_groups_after"]
    assert info["coverage"] == pytest.approx(2 / 3)


def test_control_the_explanations_verdict_itself_is_unchanged_and_still_earned():
    """OVER-CORRECTION CONTROL. The clause is ADDED; the verdict, the score and the
    NOT GRADED machinery the previous wave built must all survive intact."""
    analyzer = _analyzer_with_unlabelled_rows(40, 40, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = analyzer.get_explanations()["summary"]
    assert "Overall Fairness Score: 100.0% (5 passed, 0 failed)" in summary, summary
    assert "All assessable metrics are within acceptable thresholds" in summary, summary
    assert "80 of 80 rows assessed, 0 excluded for missing values" in summary, summary


def test_control_an_earned_critical_finding_is_not_displaced_by_the_clause():
    """A real disparity must still reach the summary as CRITICAL, with the clause
    beside it rather than instead of it."""
    from vfairness import FairnessAnalyzer

    n = 40
    pred = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    attr = np.array(["a"] * n + ["b"] * n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = FairnessAnalyzer(pred.copy(), pred, attr).get_explanations()["summary"]
    assert "CRITICAL" in summary, summary
    assert "80 of 80 rows assessed" in summary, summary
