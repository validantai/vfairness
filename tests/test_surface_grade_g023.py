"""Surface grading batch G-023: ``operations.experimentation.power``.

The defect class: a value nobody measured, returned in the slot a caller reads
as a measurement. This module had already been through three repair waves
(R-1 2026-09-09, READINESS-6 2026-09-10, BGL-S2/S2b 2026-09-16/17), each of
which fixed one site of the shape and left a sibling standing. The four sites
pinned below were all found by execution on 2026-09-17:

1. ``sequential_test`` below the ``min_group_size`` floor returned
   ``log_likelihood_ratio=0.0``, the exact midpoint of the two Wald boundaries,
   for a test it never ran, and warned nothing. The branch immediately above it
   (an arm with no observation at all) had been fixed for that reason the day
   before.
2. ``sequential_test`` substituted ``pooled_std = 1.0`` for an arm holding fewer
   than two observations. Measured: one control observation against 60
   treatment observations of mean 6.0 and sd 10.8, a gap of 0.55 sd, came back
   ``reject_null`` with ``stopped_early=True``. The pooled variance is in fact
   MEASURABLE there (a singleton arm contributes no degrees of freedom, so the
   other arm's variance is the pooled one), so the repair measures it rather
   than refusing: refusing would have thrown away a real number.
3. ``power_for_sample_size`` handed an arm with NO observation a power of 0.025
   whenever ``min_group_size`` was 0, and 0.025 is alpha/2, the false-positive
   rate, printed in the achieved-power column. ``minimum_detectable_effect``
   and ``get_power_summary`` died with ZeroDivisionError on the same input.
4. ``effect_size`` was only ever checked with ``<= 0``, so ``inf`` returned a
   required sample size of 0 ("you need no data"), ``power_for_sample_size``
   returned 1.0 for it, and NaN slipped through to die inside ``int()``.

Every test here carries a control asserting a real measurement is still
produced exactly, because the cheapest way to pass a refusal test is to refuse
everything.
"""

from __future__ import annotations

import math
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from vfairness.exceptions import ConfigurationError
from vfairness.operations.experimentation import FairnessExperiment
from vfairness.operations.experimentation.power import (
    FairnessPowerAnalyzer,
    PowerConfig,
    SPRTDecision,
)

# Independent of vfairness._statistics: stdlib normal quantiles, so the
# expected values below are computed here rather than copied from the code.
Z_ALPHA = NormalDist().inv_cdf(0.975)
Z_BETA = NormalDist().inv_cdf(0.80)


def _messages(caught) -> list:
    return [str(x.message) for x in caught]


def _analyzer(control: pd.DataFrame, treatment: pd.DataFrame, **cfg) -> FairnessPowerAnalyzer:
    return FairnessPowerAnalyzer(
        FairnessExperiment(control, treatment, ["gender"], "y"),
        PowerConfig(**cfg),
    )


def _expected_llr(ctrl: np.ndarray, treat: np.ndarray, effect_size: float) -> float:
    """Wald two-sample SPRT log-likelihood ratio, written out independently.

    Pooled variance weights each arm by its own degrees of freedom, so an arm
    of one observation contributes nothing and the other arm carries it.
    """
    ss = 0.0
    dof = 0
    for arm in (ctrl, treat):
        if len(arm) > 1:
            ss += float(np.var(arm, ddof=1)) * (len(arm) - 1)
            dof += len(arm) - 1
    pooled = math.sqrt(ss / dof)
    obs_diff = (float(np.mean(treat)) - float(np.mean(ctrl))) / pooled
    var_obs = 1.0 / len(ctrl) + 1.0 / len(treat)
    return (effect_size * obs_diff - effect_size**2 / 2) / var_obs


# ---------------------------------------------------------------------------
# 1. sequential_test below the size floor: a ratio for a test that never ran
# ---------------------------------------------------------------------------


def _small_arms(n: int = 10, gap: float = 2.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = rng.normal(gap, 1.0, n)
    return (
        pd.DataFrame({"gender": ["f"] * n, "y": c}),
        pd.DataFrame({"gender": ["f"] * n, "y": t}),
        c,
        t,
    )


class TestSPRTBelowTheFloorHasNoLikelihoodRatio:
    """Measured 2026-09-17 at 10 observations per arm against a floor of 30,
    carrying a real 2-sigma treatment gap: decision continue,
    log_likelihood_ratio 0.0, no warning. 0.0 sits exactly between the Wald
    boundaries (-1.5581 and 2.7726) and reads as balanced evidence. Had the
    SPRT run on those observations it would have crossed the upper boundary."""

    def test_the_ratio_is_not_zero_and_says_why(self):
        ctrl_df, treat_df, _, _ = _small_arms()
        analyzer = _analyzer(ctrl_df, treat_df, min_group_size=30)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.sequential_test(effect_size=0.5)[("f",)]

        assert math.isnan(result.log_likelihood_ratio), (
            f"a ratio was reported for a test that never ran: {result.log_likelihood_ratio!r}"
        )
        assert math.isnan(result.to_dict()["log_likelihood_ratio"])
        # The instruction is kept: below the floor, more data WILL change this.
        assert result.decision is SPRTDecision.CONTINUE
        assert result.stopped_early is False
        named = [m for m in _messages(caught) if "below the min_group_size floor of 30" in m]
        assert named, _messages(caught)
        assert "NOT 0.0" in named[0]

    def test_control_the_same_data_above_the_floor_is_still_measured(self):
        """OVER-CORRECTION CONTROL. Drop the floor to 10 and the identical
        observations produce a real ratio, matching an independently written
        Wald LLR, and a real decision."""
        ctrl_df, treat_df, c, t = _small_arms()
        analyzer = _analyzer(ctrl_df, treat_df, min_group_size=10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.sequential_test(effect_size=0.5)[("f",)]

        expected = _expected_llr(c, t, 0.5)
        assert result.log_likelihood_ratio == pytest.approx(round(expected, 4), abs=1e-4)
        assert expected > result.upper_boundary
        assert result.decision is SPRTDecision.REJECT_NULL
        assert result.stopped_early is True
        assert not [m for m in _messages(caught) if "min_group_size floor" in m]


# ---------------------------------------------------------------------------
# 2. sequential_test: a pooled standard deviation of 1.0 by decree
# ---------------------------------------------------------------------------


def _singleton_control_arms(seed: int = 3):
    """One control observation at 0.0; 60 treatment observations with mean
    exactly 6.0 and sd about 10.8, i.e. a gap of roughly 0.55 sd."""
    rng = np.random.default_rng(seed)
    t = rng.normal(0.0, 10.0, 60)
    t = t - t.mean() + 6.0
    return (
        pd.DataFrame({"gender": ["f"], "y": [0.0]}),
        pd.DataFrame({"gender": ["f"] * 60, "y": t}),
        np.array([0.0]),
        t,
    )


class TestSPRTDoesNotInventAPooledStandardDeviation:
    """Measured 2026-09-17 with min_group_size=1: reject_null,
    stopped_early=True, llr 2.8279 against an upper boundary of 2.7726, from a
    difference of 0.55 treatment standard deviations. The divisor was 1.0
    because ``len(ctrl) > 1`` was false, not because anything measured it."""

    def test_a_one_observation_arm_does_not_buy_a_stop_decision(self):
        ctrl_df, treat_df, c, t = _singleton_control_arms()
        analyzer = _analyzer(ctrl_df, treat_df, min_group_size=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.sequential_test(effect_size=0.5)[("f",)]

        assert result.decision is not SPRTDecision.REJECT_NULL, (
            "a 0.55-sigma gap crossed the Wald boundary on an invented unit"
        )
        assert result.stopped_early is False
        # The pooled variance IS measurable here (the singleton arm carries no
        # degrees of freedom), so the ratio is a measurement, not a refusal:
        # over-refusing would throw the treatment arm's variance away.
        expected = _expected_llr(c, t, 0.5)
        assert result.log_likelihood_ratio == pytest.approx(round(expected, 4), abs=1e-4)
        assert abs(expected) < abs(result.upper_boundary)
        assert not _messages(caught)

    def test_the_divisor_is_the_measured_spread_not_one(self):
        """The same gap read against two different real spreads must give two
        different ratios. With a divisor of 1.0 by decree they would be equal
        up to the arm-size term, which is what let 0.55 sigma look decisive."""
        ctrl_df, treat_df, c, t = _singleton_control_arms()
        wide = _analyzer(ctrl_df, treat_df, min_group_size=1)
        narrow_t = (t - 6.0) / 10.0 + 6.0  # same mean, one tenth the spread
        narrow = _analyzer(
            ctrl_df,
            pd.DataFrame({"gender": ["f"] * 60, "y": narrow_t}),
            min_group_size=1,
        )
        llr_wide = wide.sequential_test(effect_size=0.5)[("f",)].log_likelihood_ratio
        llr_narrow = narrow.sequential_test(effect_size=0.5)[("f",)].log_likelihood_ratio

        assert llr_narrow > llr_wide * 5, (llr_wide, llr_narrow)
        assert llr_wide == pytest.approx(round(_expected_llr(c, t, 0.5), 4), abs=1e-4)
        assert llr_narrow == pytest.approx(round(_expected_llr(c, narrow_t, 0.5), 4), abs=1e-4)

    def test_control_a_real_effect_at_full_size_still_stops(self):
        """OVER-CORRECTION CONTROL: 60 per arm, a true 0.8-sigma effect, still
        reject_null with a finite ratio and no warning."""
        rng = np.random.default_rng(11)
        c = rng.normal(0.0, 1.0, 60)
        t = rng.normal(0.8, 1.0, 60)
        analyzer = _analyzer(
            pd.DataFrame({"gender": ["f"] * 60, "y": c}),
            pd.DataFrame({"gender": ["f"] * 60, "y": t}),
            min_group_size=30,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.sequential_test(effect_size=0.5)[("f",)]

        assert result.decision is SPRTDecision.REJECT_NULL
        assert result.stopped_early is True
        assert result.log_likelihood_ratio == pytest.approx(
            round(_expected_llr(c, t, 0.5), 4), abs=1e-4
        )
        assert not _messages(caught)


# ---------------------------------------------------------------------------
# 3. No observations, no floor: alpha/2 printed as achieved power
# ---------------------------------------------------------------------------


def _no_observation_arms(n: int = 60):
    frame = pd.DataFrame({"gender": ["f"] * n, "y": [np.nan] * n})
    return frame, frame.copy()


class TestZeroObservationsIsNotAPowerOfAlphaOverTwo:
    """Measured 2026-09-17 with min_group_size=0 on 60 rows per arm whose
    outcome was NaN in every one: power_for_sample_size 0.025 (= alpha/2),
    minimum_detectable_effect and get_power_summary both ZeroDivisionError,
    while the warning raised two frames up promised "They report NaN and
    is_powered=None (could not check)". The surface contradicted its warning."""

    def test_power_and_mde_refuse_whatever_the_floor_is(self):
        control, treatment = _no_observation_arms()
        analyzer = _analyzer(control, treatment, min_group_size=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            power = analyzer.power_for_sample_size(effect_size=0.5)
            mde = analyzer.minimum_detectable_effect()

        assert math.isnan(power[("f",)]), f"alpha/2 reported as achieved power: {power}"
        assert math.isnan(mde[("f",)])
        assert [m for m in _messages(caught) if "no outcome observation" in m]

    def test_the_summary_renders_the_third_state_instead_of_crashing(self):
        control, treatment = _no_observation_arms()
        analyzer = _analyzer(control, treatment, min_group_size=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = analyzer.get_power_summary(effect_size=0.5).iloc[0]
            detailed = analyzer.get_detailed_results(effect_size=0.5)[0]

        assert math.isnan(row["power"])
        assert row["is_powered"] is None
        assert math.isnan(row["mde"])
        assert row["n_control"] == 0 and row["n_rows_control"] == 60
        assert math.isnan(detailed.power)
        assert detailed.is_powered is None
        assert math.isnan(detailed.to_dict()["power"])
        assert detailed.to_dict()["n_rows_control"] == 60
        assert [m for m in _messages(caught) if "no outcome observation" in m]

    def test_control_the_same_shape_fully_observed_is_measured_exactly(self):
        """OVER-CORRECTION CONTROL, with min_group_size still 0: 60 real
        observations per arm produce the textbook power and MDE, computed here
        from stdlib normal quantiles rather than copied from the code."""
        rng = np.random.default_rng(5)
        control = pd.DataFrame({"gender": ["f"] * 60, "y": rng.normal(0.0, 1.0, 60)})
        treatment = pd.DataFrame({"gender": ["f"] * 60, "y": rng.normal(0.5, 1.0, 60)})
        analyzer = _analyzer(control, treatment, min_group_size=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            power = analyzer.power_for_sample_size(effect_size=0.5)[("f",)]
            mde = analyzer.minimum_detectable_effect()[("f",)]
            required = analyzer.required_sample_size(effect_size=0.5)[("f",)]
            row = analyzer.get_power_summary(effect_size=0.5).iloc[0]

        expected_power = NormalDist().cdf(0.5 * math.sqrt(60 / 2) - Z_ALPHA)
        expected_mde = (Z_ALPHA + Z_BETA) * math.sqrt(2 / 60)
        expected_required = math.ceil(2 * ((Z_ALPHA + Z_BETA) / 0.5) ** 2)

        # 1e-3, not 1e-4: the library's own `_norm_ppf` is the Abramowitz and
        # Stegun rational approximation, off by about 4.3e-4 at the 0.975
        # quantile (1.9603949 against 1.9599640), which moves the power figure
        # by about 1.1e-4. That is an approximation, not a fabrication, and the
        # tolerance still separates a real measurement from alpha/2 or 1.0.
        assert power == pytest.approx(expected_power, abs=1e-3)
        assert mde == pytest.approx(expected_mde, abs=1e-4)
        assert required == expected_required
        # A verdict, not the third state: pandas hands this back as np.False_.
        assert row["is_powered"] is not None and bool(row["is_powered"]) is False
        assert not _messages(caught)


# ---------------------------------------------------------------------------
# 4. effect_size: a design parameter that is not a number
# ---------------------------------------------------------------------------


def _healthy_analyzer(seed: int = 2) -> FairnessPowerAnalyzer:
    rng = np.random.default_rng(seed)
    return _analyzer(
        pd.DataFrame({"gender": ["f"] * 60, "y": rng.normal(0.0, 1.0, 60)}),
        pd.DataFrame({"gender": ["f"] * 60, "y": rng.normal(0.5, 1.0, 60)}),
    )


class TestAnEffectSizeThatIsNotANumberIsRefused:
    """Measured 2026-09-17 on a 60-per-arm experiment:
    required_sample_size(inf) -> 0 ("you need no samples"),
    required_sample_size(nan) -> ValueError inside int(),
    power_for_sample_size(inf) -> 1.0, power_for_sample_size(0.0) -> 0.025,
    power_for_sample_size(nan) -> nan with no warning at all."""

    @pytest.mark.parametrize("effect_size", [float("nan"), float("inf"), 0.0, -0.5])
    @pytest.mark.parametrize(
        "method",
        ["required_sample_size", "power_for_sample_size", "sequential_test"],
    )
    def test_every_entry_point_refuses(self, method, effect_size):
        analyzer = _healthy_analyzer()
        with pytest.raises(ConfigurationError):
            getattr(analyzer, method)(effect_size=effect_size)

    def test_the_plan_refuses_too_rather_than_planning_around_it(self):
        analyzer = _healthy_analyzer()
        with pytest.raises(ConfigurationError):
            analyzer.adaptive_sampling_plan(budget=100, effect_size=float("inf"))

    def test_control_an_ordinary_effect_size_is_answered(self):
        """OVER-CORRECTION CONTROL: a refusal that refused everything would
        pass the parametrised test above and find nothing."""
        analyzer = _healthy_analyzer()
        assert analyzer.required_sample_size(effect_size=0.5)[("f",)] == math.ceil(
            2 * ((Z_ALPHA + Z_BETA) / 0.5) ** 2
        )
        assert analyzer.power_for_sample_size(effect_size=0.5)[("f",)] == pytest.approx(
            NormalDist().cdf(0.5 * math.sqrt(30) - Z_ALPHA), abs=1e-3
        )
        assert analyzer.sequential_test(effect_size=0.5)[("f",)].decision in {
            SPRTDecision.CONTINUE,
            SPRTDecision.REJECT_NULL,
            SPRTDecision.ACCEPT_NULL,
        }
        # A very small effect size is not a degenerate one: it has an answer,
        # and refusing it would be the reverse defect.
        assert analyzer.required_sample_size(effect_size=1e-3)[("f",)] > 10**6


# ---------------------------------------------------------------------------
# 5. SamplingPlan.to_dataframe: the third state a reader could not see
# ---------------------------------------------------------------------------


def _mixed_plan_analyzer():
    """'f' holds 60 rows per arm and no outcome value in any of them; 'm' holds
    60 real observations per arm."""
    rng = np.random.default_rng(13)
    control = pd.DataFrame(
        {
            "gender": ["f"] * 60 + ["m"] * 60,
            "y": [np.nan] * 60 + list(rng.normal(0.0, 1.0, 60)),
        }
    )
    treatment = pd.DataFrame(
        {
            "gender": ["f"] * 60 + ["m"] * 60,
            "y": [np.nan] * 60 + list(rng.normal(0.5, 1.0, 60)),
        }
    )
    return _analyzer(control, treatment)


class TestTheSamplingPlanFrameShowsWhatWasNotAssessed:
    """Measured 2026-09-17: ``to_dict`` carried ``n_not_assessed: 1`` while
    ``to_dataframe``, the surface a caller renders as "the plan", returned one
    row for 'm' and no trace that 'f' existed. A reader of the frame alone
    would conclude 'f' needs no action."""

    def test_the_unassessed_intersection_has_a_row_that_says_so(self):
        analyzer = _mixed_plan_analyzer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=100, effect_size=0.5)
            frame = plan.to_dataframe()

        assert plan.not_assessed == [("f",)]
        assert len(frame) == plan.n_underpowered + len(plan.not_assessed)
        by_ix = dict(zip(frame["intersection"], frame["additional_samples"]))
        assert "('f',)" in by_ix, f"the not-assessed intersection is invisible: {by_ix}"
        assert pd.isna(by_ix["('f',)"]), "a not-assessed intersection got a sample count"
        rationale = dict(zip(frame["intersection"], frame["rationale"]))
        assert rationale["('f',)"].startswith("NOT ASSESSED:")
        # The measured half is untouched.
        assert by_ix["('m',)"] == plan.allocations[("m",)] > 0
        assert "already powered" not in plan.rationale.values()
        assert [m for m in _messages(caught) if "not_assessed" in m]

    def test_control_a_fully_observed_plan_is_unchanged(self):
        """OVER-CORRECTION CONTROL: with nothing unassessed the frame is still
        exactly one row per underpowered intersection and carries no NaN."""
        rng = np.random.default_rng(17)
        control = pd.DataFrame(
            {
                "gender": ["f"] * 60 + ["m"] * 60,
                "y": list(rng.normal(0.0, 1.0, 120)),
            }
        )
        treatment = pd.DataFrame(
            {
                "gender": ["f"] * 60 + ["m"] * 60,
                "y": list(rng.normal(0.5, 1.0, 120)),
            }
        )
        analyzer = _analyzer(control, treatment)
        plan = analyzer.adaptive_sampling_plan(budget=100, effect_size=0.5)
        frame = plan.to_dataframe()

        assert plan.not_assessed == []
        assert len(frame) == plan.n_underpowered == 2
        assert not frame["additional_samples"].isna().any()
        assert (frame["additional_samples"] > 0).all()
