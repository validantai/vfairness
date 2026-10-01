"""Surface grading, batch g001: vfairness_metrics/_statistics.py.

Every test here pairs a REFUSAL pin with a healthy-data CONTROL, because a
detector that refuses everything passes every degenerate-input test while
finding nothing real. The control values are computed independently in the test
(from the closed form, or from scipy), never copied out of the implementation.

Defects these pin, each proved by execution at the public entry before the fix:

  cohens_h_interpretation(nan)                 -> 'large'   (the top band)
  bayesian_difference_ci(0, 0, 2, 100)         -> 0.4688 gap, p_greater 0.9703
  wilson_score_interval(11, 4)                 -> "proportion" 2.75
  bayesian_mean_ci(constant arm, n=20)         -> 95% CI of width 2.2e-16
  proportion_z_test(1.0, 1, 0.0, 50)           -> 1.0       (100% vs 0%)
  min_attainable_p_sign_flip(2000)             -> OverflowError
  interpret_effect_size(0.0, 'odds_ratio')     -> ZeroDivisionError
  simultaneous_disparity_bounds({'A':(30,100)})-> ratio bound 0.5482 from ONE group
  minimum_detectable_effect(100,1000,0.5,0.1)  -> -0.0318, cleared by power_warning
  empirical_likelihood_ci(60, 50)              -> rate 1.0, silently clamped
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    IntervalType,
    MultipleTestingResult,
    StatisticalResult,
    apply_multiple_testing_correction,
    bayesian_difference_ci,
    bayesian_mean_ci,
    bootstrap_over_index,
    cohens_d,
    cohens_h,
    cohens_h_interpretation,
    compute_metric_with_ci,
    detectability,
    empirical_likelihood_ci,
    fisher_exact_test,
    group_reliability,
    interpret_effect_size,
    min_attainable_p_fisher,
    min_attainable_p_mcnemar,
    min_attainable_p_permutation,
    min_attainable_p_sign_flip,
    minimum_detectable_effect,
    odds_ratio,
    power_warning,
    proportion_z_test,
    reliability_tier,
    risk_ratio,
    select_method,
    sequential_fairness_test,
    simultaneous_disparity_bounds,
    stratified_bootstrap_ci,
    validate_confidence_level,
    wilson_score_interval,
)
from vfairness.exceptions import ConfigurationError

_BANDS = {"negligible", "small", "medium", "large"}


def _caught(fn):
    """Run fn with warnings ON and return (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in rec]


# cohens_h_interpretation: a magnitude band for a proportion nobody measured


def test_cohens_h_interpretation_refuses_an_unmeasured_effect():
    for bad in (float("nan"), float("inf"), float("-inf")):
        verdict, msgs = _caught(lambda b=bad: cohens_h_interpretation(b))
        assert verdict not in _BANDS, f"{bad} was graded {verdict!r}"
        assert verdict == "not interpretable"
        assert any("not a finite number" in m for m in msgs), bad


def test_cohens_h_interpretation_does_not_regrade_its_own_producers_nan():
    """cohens_h returns NaN for an unmeasured proportion (BGL-S2b). The band
    function must not turn that back into a finding."""
    h, _ = _caught(lambda: cohens_h(float("nan"), 0.5))
    assert math.isnan(h)
    assert cohens_h_interpretation(h) not in _BANDS


@pytest.mark.parametrize(
    "p1,p2,band",
    [
        (0.52, 0.50, "negligible"),  # h = 0.0400
        (0.60, 0.50, "small"),  # h = 0.2014
        (0.70, 0.50, "small"),  # h = 0.4115
        (0.80, 0.50, "medium"),  # h = 0.6435
        (0.80, 0.20, "large"),  # h = 1.2870
    ],
)
def test_control_cohens_h_and_its_band_are_still_measured(p1, p2, band):
    expected_h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    h, msgs = _caught(lambda: cohens_h(p1, p2))
    assert h == pytest.approx(expected_h, rel=1e-12)
    assert not msgs
    assert cohens_h_interpretation(h) == band


# interpret_effect_size: a measured ratio of zero must not crash


@pytest.mark.parametrize("kind", ["risk_ratio", "odds_ratio"])
def test_interpret_effect_size_reads_a_measured_zero_ratio(kind):
    """odds_ratio returns exactly 0.0 by design when only the numerator
    vanishes, and the reading of 1 / 0.0 used to be ZeroDivisionError."""
    measured, _ = _caught(lambda: odds_ratio(0, 30, 5, 30))
    assert measured[0] == 0.0
    text = interpret_effect_size(0.0, kind)
    assert "total separation" in text
    assert "negligible" not in text


def test_control_interpret_effect_size_still_reads_a_real_ratio():
    assert interpret_effect_size(2.0, "risk_ratio") == "group 1 has 2.00x higher risk"
    assert interpret_effect_size(1.0, "risk_ratio") == "negligible difference"
    verdict, msgs = _caught(lambda: interpret_effect_size(float("nan")))
    assert "not interpretable" in verdict
    assert msgs


# bayesian_difference_ci: a gap between a measured group and a prior


def test_bayesian_difference_ci_refuses_a_group_with_no_observations():
    result, msgs = _caught(lambda: bayesian_difference_ci(0, 0, 2, 100, random_state=7))
    assert math.isnan(result.point_estimate), "a gap was reported against the prior mean"
    assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
    assert result.metadata.get("could_not_check") is True
    assert math.isnan(result.metadata["p_greater"])
    assert result.is_significant is False
    assert any("no observations" in m for m in msgs)


def test_bayesian_difference_ci_refuses_an_impossible_count_pair():
    with pytest.raises(ConfigurationError, match="valid count pair"):
        bayesian_difference_ci(11, 4, 2, 10)


def test_control_bayesian_difference_ci_still_measures_a_real_gap():
    result, msgs = _caught(
        lambda: bayesian_difference_ci(30, 100, 10, 100, n_samples=40000, random_state=3)
    )
    # Posterior means of Beta(31, 71) and Beta(11, 91): 31/102 - 11/102.
    expected = 31.0 / 102.0 - 11.0 / 102.0
    assert result.point_estimate == pytest.approx(expected, abs=0.01)
    assert result.lower_bound > 0.0, "a 20-point gap must exclude parity"
    assert result.is_significant is True
    assert not msgs


# wilson_score_interval


def test_wilson_score_interval_refuses_an_impossible_count_pair():
    with pytest.raises(ConfigurationError, match="valid count pair"):
        wilson_score_interval(11, 4)


def test_control_wilson_score_interval_matches_the_closed_form():
    result, msgs = _caught(lambda: wilson_score_interval(30, 100))
    p, n = 0.3, 100
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    assert result.point_estimate == pytest.approx(p)
    assert result.lower_bound == pytest.approx(centre - half, rel=1e-3)
    assert result.upper_bound == pytest.approx(centre + half, rel=1e-3)
    assert not msgs
    empty, _ = _caught(lambda: wilson_score_interval(0, 0))
    assert math.isnan(empty.point_estimate)


# bayesian_mean_ci: a zero-width credible interval from a constant arm


@pytest.mark.parametrize("n", [20, 25])
def test_bayesian_mean_ci_refuses_a_constant_sample(n):
    """np.var of a constant array is an accumulated statistic: exactly 0.0 at
    n=25 and 5.19e-32 at n=20, so the two lengths took different branches and
    neither was honest. Both must refuse."""
    result, msgs = _caught(lambda: bayesian_mean_ci(np.full(n, 0.9), random_state=1))
    assert math.isnan(result.point_estimate)
    assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
    assert result.interval_width != 0.0 or math.isnan(result.interval_width)
    assert result.metadata.get("could_not_check") is True
    assert any("no variance" in m for m in msgs), msgs


def test_bayesian_mean_ci_refuses_non_finite_data():
    result, msgs = _caught(lambda: bayesian_mean_ci(np.full(20, np.nan), random_state=1))
    assert math.isnan(result.point_estimate)
    assert result.metadata.get("could_not_check") is True
    assert any("not finite" in m for m in msgs)


def test_control_bayesian_mean_ci_still_measures_a_real_mean():
    data = np.random.default_rng(1).normal(5.0, 2.0, 400)
    result, msgs = _caught(lambda: bayesian_mean_ci(data, random_state=1))
    assert result.point_estimate == pytest.approx(float(np.mean(data)), abs=0.05)
    assert result.lower_bound < result.point_estimate < result.upper_bound
    assert result.interval_width > 0.05, "a real sample has a real interval width"
    assert not msgs


# proportion_z_test / fisher_exact_test: 1.0 is not a sentinel


def test_proportion_z_test_refuses_a_group_too_small_to_test():
    p, msgs = _caught(lambda: proportion_z_test(1.0, 1, 0.0, 50))
    assert math.isnan(p), "100% against 0% was reported as p = 1.0"
    assert any("too small" in m for m in msgs)


def test_proportion_z_test_refuses_an_unmeasured_proportion():
    p, msgs = _caught(lambda: proportion_z_test(float("nan"), 100, 0.5, 100))
    assert math.isnan(p)
    assert any("not measured" in m for m in msgs)


def test_control_proportion_z_test_still_returns_a_real_p_value():
    p, msgs = _caught(lambda: proportion_z_test(0.8, 100, 0.4, 100))
    # Independent: pooled p = 0.6, se = sqrt(.6*.4*(1/100+1/100)), two-sided.
    se = math.sqrt(0.6 * 0.4 * (1 / 100 + 1 / 100))
    z = abs(0.8 - 0.4) / se
    expected = math.erfc(z / math.sqrt(2.0))
    assert p == pytest.approx(expected, rel=0.05)
    assert p < 1e-6
    assert not msgs
    # And a genuine no-difference observation keeps its measured 1.0.
    same, _ = _caught(lambda: proportion_z_test(0.0, 100, 0.0, 100))
    assert same == 1.0


@pytest.mark.parametrize(
    "table",
    [
        [0, 0, 0, 0],  # nothing counted at all
        [5, 5, 0, 0],  # empty second row
        [0, 0, 5, 5],  # empty first row
        [5, 0, 5, 0],  # empty second column
        [0, 5, 0, 5],  # empty first column
    ],
)
def test_fisher_exact_test_refuses_every_degenerate_margin(table):
    """BGL-G001, RESOLVED 2026-09-25. All four shapes at once, as the record asked.

    With any empty margin the odds ratio is not estimable: conditional on those
    margins the observed table is the only one possible. scipy answers 1.0 with a
    nan STATISTIC; this function returns only the p, so passing 1.0 on would keep
    the number and drop the disclosure, and 1.0 reads as "no association found".
    """
    a, b, c, d = table
    ours, msgs = _caught(lambda: fisher_exact_test(a, b, c, d))
    assert math.isnan(ours), f"{table} was reported as a measured p-value"
    assert any("could not check" in m for m in msgs), f"{table} refused in silence"


def test_fisher_exact_test_still_never_drifts_from_scipy_where_a_p_exists():
    """The half of the old pin that must survive: on every table with real
    margins this module's shortcut still agrees with the reference exactly."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for table in ([10, 90, 30, 70], [1, 0, 100, 149], [50, 50, 50, 50], [3, 7, 8, 2]):
        a, b, c, d = table
        ours, msgs = _caught(lambda a=a, b=b, c=c, d=d: fisher_exact_test(a, b, c, d))
        _, theirs = scipy_stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        assert ours == pytest.approx(float(theirs), rel=1e-12), table
        assert not msgs, f"{table} is measurable and should refuse nothing"


def test_control_fisher_exact_test_does_not_over_refuse_a_real_one():
    """A table with NO association must answer a measured 1.0, not nan. Removing a
    fabricated value is not licence to refuse the cases that are measurable: that
    direction throws away evidence, and it is the failure this campaign hit twice."""
    p_same, msgs = _caught(lambda: fisher_exact_test(50, 50, 50, 50))
    assert p_same == 1.0 and not math.isnan(p_same)
    assert not msgs


def test_control_fisher_exact_test_still_finds_a_real_association():
    p, msgs = _caught(lambda: fisher_exact_test(10, 90, 50, 50))
    scipy_stats = pytest.importorskip("scipy.stats")
    _, expected = scipy_stats.fisher_exact([[10, 90], [50, 50]], alternative="two-sided")
    assert p == pytest.approx(float(expected), rel=1e-9)
    assert p < 0.001
    assert not msgs


def test_an_untested_comparison_does_not_dilute_the_family():
    """The refusals above return NaN, and _testable must drop those rather than
    letting them inflate the Benjamini-Hochberg family size."""
    res, msgs = _caught(
        lambda: apply_multiple_testing_correction(np.array([0.001, np.nan, 0.9]), "fdr")
    )
    assert res.n_not_tested == 1
    assert list(res.tested_mask) == [True, False, True]
    assert res.adjusted_p_values[0] == pytest.approx(0.002)  # family of 2, not 3
    assert math.isnan(res.adjusted_p_values[1])
    assert any("NOT tested" in m for m in msgs)
    d = res.to_dict()
    assert d["tested_mask"] == [True, False, True] and d["n_not_tested"] == 1


def test_control_a_full_family_is_corrected_against_its_real_size():
    res, msgs = _caught(
        lambda: apply_multiple_testing_correction(np.array([0.001, 0.02, 0.9]), "fdr")
    )
    assert res.adjusted_p_values[0] == pytest.approx(0.003)  # 0.001 * 3 / 1
    assert res.n_not_tested == 0 and res.n_rejected == 2
    assert not msgs


# min_attainable_p_* and detectability


def test_min_attainable_p_sign_flip_survives_an_ordinary_pair_count():
    """2.0 ** 1024 raises OverflowError, and 1024 paired observations is an
    ordinary counterfactual run."""
    for n in (1024, 2000, 5000):
        floor = min_attainable_p_sign_flip(n)
        assert floor is not None and 0.0 <= floor <= 1.0, n


def test_control_min_attainable_p_floors_match_their_closed_forms():
    assert min_attainable_p_sign_flip(4) == pytest.approx(0.125)
    assert min_attainable_p_sign_flip(10) == pytest.approx(2.0 / 1024.0)
    assert min_attainable_p_sign_flip(10, 99) == pytest.approx(1.0 / 100.0)
    assert min_attainable_p_sign_flip(0) == 1.0
    assert min_attainable_p_sign_flip(-1) is None
    assert min_attainable_p_mcnemar(6) == pytest.approx(2.0 * 0.5**6)
    assert min_attainable_p_mcnemar(5) == pytest.approx(0.0625)
    assert min_attainable_p_mcnemar(0) == 1.0
    assert min_attainable_p_mcnemar("x") is None
    assert min_attainable_p_permutation(199) == pytest.approx(1.0 / 200.0)
    assert min_attainable_p_permutation(0) == 1.0
    assert min_attainable_p_permutation(-1) is None
    assert min_attainable_p_fisher(0, 6) is None
    assert min_attainable_p_fisher(6, 6) == pytest.approx(0.0021645021645, rel=1e-6)


def test_detectability_keeps_could_not_check_apart_from_not_detectable():
    assert detectability(None, 1)[0] is None
    assert detectability(float("nan"), 1)[0] is None
    assert detectability(0.5, "x")[0] is None
    assert detectability(0.0021645, 32)[0] is False
    assert detectability(0.0021645, 1)[0] is True  # control: this design can fire
    assert detectability(0.0021645, 1)[1] == ""


# minimum_detectable_effect / power_warning


def test_minimum_detectable_effect_refuses_a_design_with_no_positive_mde():
    """Each of these designs is refused with nan and a warning that says why.

    The REASON moved for two of the three on 2026-09-30 (B4 tier-1 audit) and the
    subject did not. alpha=0.0 and power=1.0 used to be caught downstream, by the
    arithmetic they produced ("not a positive size"); they are now range-checked
    above the arithmetic, because the same absence of a range check let alpha=1.0
    through to publish a floor of 0.0441 against the real 0.1469. So the assertion
    is on the refusal and on a stated reason, not on which of the two reasons.
    (100, 1000, 0.5, 0.1) still reaches the original branch: its parameters are all
    in range and it is the ARITHMETIC that comes out non-positive.
    """
    for args in ((100, 1000, 0.5, 0.1), (100, 1000, 0.0, 0.8), (100, 1000, 0.05, 1.0)):
        mde, msgs = _caught(lambda a=args: minimum_detectable_effect(*a))
        assert math.isnan(mde), args
        assert any(
            "not a positive size" in m or "not a probability strictly between 0 and 1" in m
            for m in msgs
        ), (args, msgs)
    # The original branch is still reachable and still worded the same way.
    _, msgs = _caught(lambda: minimum_detectable_effect(100, 1000, 0.5, 0.1))
    assert any("not a positive size" in m for m in msgs), msgs


def test_power_warning_does_not_clear_a_subgroup_it_could_not_assess():
    """None from power_warning means ADEQUATE POWER, and nan > 0.20 is False."""
    text, _ = _caught(lambda: power_warning(50, 1000, alpha=0.0))
    assert text is not None, "an uncomputable MDE cleared the subgroup"
    assert "COULD NOT BE ASSESSED" in text


def test_control_minimum_detectable_effect_and_power_warning_still_measure():
    mde, msgs = _caught(lambda: minimum_detectable_effect(100, 1000))
    z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    expected = (z_alpha + z_power) * math.sqrt(0.25 * (1 / 100 + 1 / 1000))
    assert mde == pytest.approx(expected, rel=0.01)
    assert not msgs
    assert power_warning(300, 1000) is None  # large enough, no warning
    small = power_warning(50, 1000)
    assert small is not None and "Very low statistical power" in small


# simultaneous_disparity_bounds: one group is not a group-to-group bound


def test_simultaneous_bounds_refuse_a_single_group():
    out, msgs = _caught(lambda: simultaneous_disparity_bounds({"A": (30, 100)}))
    assert out["available"] is False, "one group was compared with itself"
    assert math.isnan(out["worstCaseRatioBound"])
    assert math.isnan(out["worstCaseDifferenceBound"])
    assert "COULD NOT CHECK" in out["interpretation"]
    assert any("at least 2" in m for m in msgs)


def test_simultaneous_bounds_refuse_an_empty_family():
    out, msgs = _caught(lambda: simultaneous_disparity_bounds({}))
    assert out["available"] is False
    assert math.isnan(out["worstCaseRatioBound"])
    assert msgs


def test_control_simultaneous_bounds_still_bound_two_real_groups():
    out, msgs = _caught(lambda: simultaneous_disparity_bounds({"a": (70, 100), "b": (30, 100)}))
    assert out["available"] is True
    lows = [g["ci_low"] for g in out["groups"].values()]
    highs = [g["ci_high"] for g in out["groups"].values()]
    assert out["worstCaseRatioBound"] == pytest.approx(min(lows) / max(highs), abs=1e-4)
    assert out["worstCaseDifferenceBound"] == pytest.approx(max(highs) - min(lows), abs=1e-4)
    assert 0.0 < out["worstCaseRatioBound"] < 1.0
    assert not msgs


# empirical_likelihood_ci


def test_empirical_likelihood_ci_refuses_an_impossible_count_pair():
    with pytest.raises(ConfigurationError, match="valid count pair"):
        empirical_likelihood_ci(60, 50)


def test_empirical_likelihood_ci_refuses_no_observations():
    out, msgs = _caught(lambda: empirical_likelihood_ci(0, 0))
    assert math.isnan(out.point_estimate)
    assert math.isnan(out.lower_bound) and math.isnan(out.upper_bound)
    assert out.interval_width != 0.0 or math.isnan(out.interval_width)
    assert any("no observations" in m for m in msgs)


def test_control_empirical_likelihood_ci_still_brackets_a_real_rate():
    out, msgs = _caught(lambda: empirical_likelihood_ci(30, 100))
    assert out.point_estimate == pytest.approx(0.3)
    assert out.lower_bound < 0.3 < out.upper_bound
    assert 0.1 < out.interval_width < 0.25
    assert not msgs


# cohens_d, risk_ratio, odds_ratio


def test_cohens_d_refuses_two_constant_arms_at_every_length():
    for n in (5, 20, 25):
        d, msgs = _caught(lambda n=n: cohens_d(np.full(n, 0.9), np.full(n, 0.2)))
        assert math.isnan(d), f"n={n} returned {d}"
        assert any("undefined" in m for m in msgs), n
    # Genuinely identical arms are an OBSERVED zero effect, not a refusal.
    same, msgs = _caught(lambda: cohens_d(np.full(20, 0.9), np.full(20, 0.9)))
    assert same == 0.0 and not msgs
    assert math.isnan(cohens_d(np.array([1.0]), np.array([2.0])))


def test_control_cohens_d_measures_a_real_standardised_gap():
    g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    pooled = math.sqrt((4 * 2.5 + 4 * 2.5) / 8)  # both variances are 2.5
    expected = (3.0 - 5.0) / pooled
    d, msgs = _caught(lambda: cohens_d(g1, g2))
    assert d == pytest.approx(expected, rel=1e-12)
    assert not msgs


def test_ratios_refuse_an_empty_or_undefined_table():
    for fn in (risk_ratio, odds_ratio):
        out, msgs = _caught(lambda f=fn: f(0, 0, 0, 0))
        assert all(math.isnan(v) for v in out), fn.__name__
        assert msgs, fn.__name__
        with pytest.raises(ConfigurationError):
            fn(11, 4, 2, 10)
    undefined, msgs = _caught(lambda: odds_ratio(30, 30, 30, 30))
    assert math.isnan(undefined[0]), "inf/inf was reported as an infinite ratio"
    saturated, msgs = _caught(lambda: risk_ratio(30, 30, 30, 30))
    assert saturated[0] == 1.0 and math.isnan(saturated[1]) and math.isnan(saturated[2])
    assert any("no interval is estimable" in m for m in msgs)


def test_control_ratios_still_measure_a_real_disparity():
    rr, msgs = _caught(lambda: risk_ratio(30, 100, 10, 100))
    assert rr[0] == pytest.approx(0.3 / 0.1)
    assert rr[1] < rr[0] < rr[2] and rr[1] > 1.0
    assert not msgs
    orr, msgs = _caught(lambda: odds_ratio(30, 100, 10, 100))
    assert orr[0] == pytest.approx((30 / 70) / (10 / 90))
    assert orr[1] < orr[0] < orr[2] and orr[1] > 1.0
    assert not msgs


# Tiering, method recommendation, confidence-level validation


def test_reliability_tier_boundaries_and_group_reliability():
    assert [reliability_tier(n) for n in (100, 99, 30, 29, 10, 9, 0)] == [
        "reliable",
        "caution",
        "caution",
        "underpowered",
        "underpowered",
        "invalid",
        "invalid",
    ]
    out = group_reliability({"A": 150, "B": 5})
    assert out["A"]["interpretable"] is True and out["A"]["note"] == ""
    assert out["B"]["interpretable"] is False and "not interpretable" in out["B"]["note"]


def test_select_method_is_a_recommendation_and_refuses_prefer_bayesian():
    overall, per_group = select_method({"A": 20, "B": 200})
    assert (overall, per_group) == ("mixed", {"A": "bayesian", "B": "bootstrap"})
    with pytest.raises(ConfigurationError):
        select_method({"A": 200}, prefer_bayesian=True)


@pytest.mark.parametrize("bad", [1.0, 0.0, -0.2, 95, True, "0.95", float("nan")])
def test_validate_confidence_level_refuses_everything_outside_the_unit_interval(bad):
    with pytest.raises(ConfigurationError):
        validate_confidence_level(bad)


def test_control_validate_confidence_level_returns_a_real_level():
    assert validate_confidence_level(0.95) == 0.95
    assert validate_confidence_level(np.float32(0.9)) == pytest.approx(0.9, abs=1e-6)


# StatisticalResult.to_dict: the third state must survive serialisation


def test_statistical_result_to_dict_carries_the_could_not_check_state():
    unmeasured = StatisticalResult(
        float("nan"),
        float("nan"),
        float("nan"),
        IntervalType.CONFIDENCE,
        method="unavailable_no_data",
        metadata={"warning": "No data after exclusions"},
    )
    d = unmeasured.to_dict()
    assert math.isnan(d["point_estimate"]) and math.isnan(d["lower_bound"])
    assert d["is_significant"] is False, "a NaN interval cannot exclude a null"
    assert d["metadata"]["warning"]
    # A ratio's parity null must travel with the flag that tested against it.
    ratio = StatisticalResult(1.0, 0.9, 1.1, IntervalType.CONFIDENCE, null_value=1.0)
    rd = ratio.to_dict()
    assert rd["null_value"] == 1.0 and rd["is_significant"] is False


def test_control_statistical_result_to_dict_reports_a_real_finding():
    measured = StatisticalResult(
        0.12, 0.05, 0.19, IntervalType.CONFIDENCE, sample_size=200, n_bootstrap=5000
    )
    d = measured.to_dict()
    assert d["is_significant"] is True
    assert d["margin_of_error"] == pytest.approx(0.07)
    assert d["null_value"] == 0.0 and d["interval_type"] == "confidence"


def test_multiple_testing_result_to_dict_is_json_shaped():
    res = MultipleTestingResult(
        original_p_values=np.array([0.01, np.nan]),
        adjusted_p_values=np.array([0.01, np.nan]),
        rejection_mask=np.array([True, False]),
        method="none",
        significance_level=0.05,
        n_rejected=1,
        tested_mask=np.array([True, False]),
        n_not_tested=1,
    )
    d = res.to_dict()
    assert isinstance(d["rejection_mask"], list) and d["rejection_mask"] == [True, False]
    assert d["tested_mask"] == [True, False] and d["n_not_tested"] == 1


# Interval builders on data that carries nothing


def test_interval_builders_refuse_when_nothing_could_be_resampled():
    from vfairness import demographic_parity_difference

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 40)
    y_pred = rng.integers(0, 2, 40)
    empty, msgs = _caught(
        lambda: compute_metric_with_ci(
            demographic_parity_difference,
            y_true[:0],
            y_pred[:0],
            np.array([], dtype=object),
            n_bootstrap=20,
        )
    )
    assert math.isnan(empty.point_estimate) and math.isnan(empty.lower_bound)
    assert empty.method == "unavailable_no_data" and empty.is_significant is False
    assert any("No data remains" in m for m in msgs)

    one_group, msgs = _caught(
        lambda: compute_metric_with_ci(
            demographic_parity_difference,
            y_true,
            y_pred,
            np.array(["A"] * 40),
            n_bootstrap=50,
            random_state=1,
        )
    )
    assert math.isnan(one_group.lower_bound) and math.isnan(one_group.upper_bound)
    assert one_group.is_significant is False

    tiny = stratified_bootstrap_ci(
        np.array([1.0]), np.array(["A"]), lambda d, g: float(np.mean(d)), n_bootstrap=10
    )
    assert math.isnan(tiny.lower_bound) and math.isnan(tiny.upper_bound)

    nothing, msgs = _caught(
        lambda: bootstrap_over_index(60, lambda idx: float("nan"), n_bootstrap=60, random_state=1)
    )
    assert math.isnan(nothing.lower_bound) and math.isnan(nothing.upper_bound)
    assert any("NaN" in m for m in msgs)
    with pytest.raises(ConfigurationError):
        bootstrap_over_index(50, lambda idx: 1.0, groups=np.array(["A"] * 50), method="bca")


def test_control_interval_builders_still_bracket_a_real_disparity():
    from vfairness import demographic_parity_difference

    rng = np.random.default_rng(11)
    sens = np.array(["A"] * 300 + ["B"] * 300)
    y_true = rng.integers(0, 2, 600)
    y_pred = np.where(sens == "A", rng.random(600) < 0.7, rng.random(600) < 0.3).astype(int)
    result = compute_metric_with_ci(
        demographic_parity_difference, y_true, y_pred, sens, n_bootstrap=300, random_state=2
    )
    observed = float(np.mean(y_pred[sens == "A"])) - float(np.mean(y_pred[sens == "B"]))
    assert result.point_estimate == pytest.approx(abs(observed), abs=0.02)
    assert result.lower_bound > 0.2, "a 40-point gap must not include parity"
    assert result.is_significant is True

    idx_ci = bootstrap_over_index(
        400, lambda idx: float(np.mean(idx)), n_bootstrap=300, random_state=3
    )
    assert idx_ci.point_estimate == pytest.approx(199.5)
    assert idx_ci.lower_bound < 199.5 < idx_ci.upper_bound


# sequential_fairness_test: three states


def test_sequential_fairness_test_keeps_its_three_states():
    constant_gap = sequential_fairness_test(np.full(30, 0.9), np.full(30, 0.2))
    assert constant_gap["available"] is False
    assert constant_gap["decision"] == "could_not_check"
    assert constant_gap["standardisedDifference"] is None
    assert constant_gap["meanDifference"] == pytest.approx(-0.7)

    identical = sequential_fairness_test(np.full(30, 0.9), np.full(30, 0.9))
    assert identical["available"] is True
    assert identical["decision"] == "accept_h0_no_difference"
    assert identical["standardisedDifference"] == 0.0

    assert sequential_fairness_test([1.0], [2.0])["available"] is False


def test_control_sequential_fairness_test_still_finds_a_real_difference():
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(0.8, 1.0, 400)
    out = sequential_fairness_test(a, b)
    assert out["available"] is True
    assert out["decision"] == "reject_h0_groups_differ"
    assert out["standardisedDifference"] == pytest.approx(0.8, abs=0.2)
    null = sequential_fairness_test(rng.normal(0, 1, 400), rng.normal(0, 1, 400))
    assert null["decision"] != "reject_h0_groups_differ"
