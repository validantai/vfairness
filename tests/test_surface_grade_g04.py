"""G04 pins: vfairness.operations.experimentation, graded 2026-09-30.

One class per defect found by EXECUTION on undefined input, each with the
refusal (or disclosure) pin AND a control asserting the healthy case's real
number. The controls are the load-bearing half: a guard that refuses everything
passes every refusal test and makes a power analysis useless, so every number
asserted here is derived in the test from the textbook formula, never quoted
from a previous run.

The defects, in the order they are pinned below:

1. ``FairnessExperiment.calculate_intersectional_power`` answered every
   impossible design with a number on the power scale: d=0 -> 0.025 (alpha/2,
   the false-positive rate), d=inf -> 1.0, alpha=1.0 -> 0.9998 ("adequately
   powered" from a test that always rejects), and with min_group_size=0 an arm
   holding NO observation -> 0.025 while the method's own warning said it
   reported NaN.
2. ``ExperimentConfig`` validated nothing, so min_group_size=0 counted an
   intersection with an empty arm as ANALYSED.
3. ``get_summary()['most_affected']`` returned the FIRST significant
   intersection in alphabetical order, not the most affected one.
4. ``ExperimentAnalysis.decision_recommendation`` returned deploy_treatment at
   0.7225 with "with no harmed groups" from a run in which NOT ONE intersection
   was compared.
5. ``heterogeneous_treatment_effects`` returned a (0, 0) frame with no columns,
   no warning and no coverage for that same run, and
   ``to_report_sections`` then dropped the whole intersectional section.
6. ``compute_pareto_frontier(maximize=[])`` MAXIMISED every metric, inverting
   the frontier for a caller whose metrics are all costs.
7. ``mediation_analysis`` published proportion_mediated=1.0 for an experiment
   with no total effect to apportion, and a complete decomposition of zeros
   from zero fittable rows.
8. ``temporal_stability_check`` let an unchecked alpha decide the verdict:
   alpha=nan returned is_stable=False, a trend FINDING, for p=0.6003.
9. Absence in a protected attribute went through six doors and behaved
   differently in each: '' and 'None' became demographic GROUPS with published
   effects, pd.NA crashed the whole analysis, np.nan vanished into "excluded
   for insufficient sample size".
10. ``spillover_detection`` dropped rows with no cluster id in silence and
    returned a verdict byte-identical to the run without them.
11. Bootstrap CIs were published below the resample count at which their bounds
    can be order statistics of the draws; at n_bootstrap=1 the bounds were
    equal, a zero-width 95 percent interval.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.exceptions import ConfigurationError
from vfairness.operations.experimentation.analysis import (
    _CATE_COLUMNS,
    ExperimentAnalysis,
)
from vfairness.operations.experimentation.experiment import (
    DesignType,
    ExperimentConfig,
    ExperimentResult,
    FairnessExperiment,
    IntersectionEffect,
)

Z_TWO_SIDED_05 = 1.959963984540054  # Phi^-1(0.975), written out, not imported


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _arms(n_per_group, shift, seed=0, groups=("f", "m"), outcome="real"):
    """Two arms, n_per_group rows per group per arm, treatment shifted by
    `shift`. `outcome='none'` blanks every outcome value, keeping the rows."""
    rng = np.random.default_rng(seed)
    frames = []
    for arm_shift in (0.0, shift):
        g = [v for v in groups for _ in range(n_per_group)]
        y = rng.normal(arm_shift, 1.0, len(g))
        if outcome == "none":
            y = np.full(len(g), np.nan)
        frames.append(pd.DataFrame({"g": g, "y": y}))
    return frames[0], frames[1]


def _experiment(control, treatment, **cfg):
    cfg.setdefault("n_bootstrap", 200)
    cfg.setdefault("min_group_size", 30)
    cfg.setdefault("random_state", 1)
    return FairnessExperiment(control, treatment, ["g"], "y", config=ExperimentConfig(**cfg))


def _expected_power(n_per_arm, effect_size, alpha=0.05):
    """The two-sample z-test power this method documents, ``Phi(d*sqrt(n/2) -
    z_{alpha/2})``, written out here with the exact normal CDF so the control
    asserts a number derived independently of the implementation.

    The library computes it through its own ``_norm_cdf``/``_norm_ppf`` rational
    approximations, which agree with the exact value to about 1.3e-4 over this
    range (measured: 4.7e-5 at n=1, 1.3e-4 at n=60, 5.0e-5 at n=100), so the
    controls below assert the derived number to 1e-3. That is a tolerance on
    the CDF approximation, not on the power formula: at 1e-3 the 0.025 the
    defect published for d=0 is 30 standard tolerances away from the 0.7818 a
    real design gets, so the assertion still discriminates.
    """
    z_alpha = Z_TWO_SIDED_05 if alpha == 0.05 else None
    assert z_alpha is not None, "only the documented alpha is derived here"
    x = effect_size * math.sqrt(n_per_arm / 2) - z_alpha
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 1. calculate_intersectional_power: a power figure for an impossible design
# ---------------------------------------------------------------------------


class TestPowerRefusesAnImpossibleDesign:
    @pytest.mark.parametrize(
        "effect_size, was",
        [(0.0, 0.025), (float("inf"), 1.0), (float("nan"), float("nan")), (-0.5, 0.0)],
        ids=["zero", "infinite", "nan", "negative"],
    )
    def test_refusal_an_effect_size_no_design_can_be_built_around(self, effect_size, was):
        """`was` records what this returned before the fix. 0.025 is alpha/2:
        the false-positive rate, printed in the power column."""
        exp = _experiment(*_arms(60, 0.5))
        with pytest.raises(ConfigurationError) as excinfo:
            exp.calculate_intersectional_power(effect_size=effect_size)
        assert "effect_size" in str(excinfo.value)

    @pytest.mark.parametrize(
        "alpha, was",
        [(0.0, 0.0), (1.0, 0.9998), (-0.2, 0.0), (float("nan"), float("nan"))],
        ids=["zero", "one", "negative", "nan"],
    )
    def test_refusal_an_alpha_that_decides_the_answer_by_itself(self, alpha, was):
        """alpha=1.0 is the dangerous direction: a test that always rejects was
        reported as 99.98 percent powered, and `powered = pwr >= 0.8` turns
        that into "adequately powered" on every consumer surface."""
        exp = _experiment(*_arms(60, 0.5))
        with pytest.raises(ConfigurationError) as excinfo:
            exp.calculate_intersectional_power(effect_size=0.5, alpha=alpha)
        assert "alpha" in str(excinfo.value)

    def test_refusal_a_floor_below_one_cannot_manufacture_power_from_nothing(self):
        """min_group_size=0 made `n_c < min_group_size` False for an arm of
        zero observations, so the arithmetic ran at n_eff=0 and published
        `Phi(-z_alpha)` = alpha/2 = 0.025. The method's own warning said, of
        that same intersection, that it reported NaN. The guard and the
        arithmetic disagreed and the guard was the honest one."""
        control, treatment = _arms(20, 0.5, outcome="none")
        exp = _experiment(control, treatment, min_group_size=1)
        power, messages = _caught(
            lambda: exp.calculate_intersectional_power(effect_size=0.5, min_group_size=0)
        )
        assert power, "the keys must still be present, carrying their refusal"
        assert all(math.isnan(v) for v in power.values()), power
        assert any("NO outcome observation" in m for m in messages)

    def test_control_a_real_design_still_reports_its_real_power(self):
        """OVER-CORRECTION CONTROL. 60 observations per arm at d=0.5 has a
        power this test derives from the formula the docstring states, and the
        healthy path must raise nothing at all."""
        exp = _experiment(*_arms(60, 0.5))
        power, messages = _caught(lambda: exp.calculate_intersectional_power(effect_size=0.5))
        expected = round(_expected_power(60, 0.5), 4)
        assert set(power) == {("f",), ("m",)}
        assert all(v == pytest.approx(expected, abs=1e-3) for v in power.values()), power
        assert not messages

    def test_control_one_observation_per_arm_is_terrible_not_impossible(self):
        """A design this library must still compute: the small-sample answer is
        a real number on the power scale, and refusing it would delete the very
        finding a reader needs."""
        control, treatment = _arms(1, 0.5)
        exp = _experiment(control, treatment, min_group_size=1)
        power, _ = _caught(
            lambda: exp.calculate_intersectional_power(effect_size=0.5, min_group_size=1)
        )
        expected = round(_expected_power(1, 0.5), 4)
        assert all(v == pytest.approx(expected, abs=1e-3) for v in power.values()), power
        assert 0.0 < expected < 0.1, "a one-observation design is nearly powerless, by design"


# ---------------------------------------------------------------------------
# 2. ExperimentConfig: an impossible configuration accepted in silence
# ---------------------------------------------------------------------------


class TestExperimentConfigRefusesWhatCannotBeHonoured:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"alpha": 5.0},
            {"alpha": float("nan")},
            {"n_bootstrap": 0},
            {"n_bootstrap": -1},
            {"min_group_size": 0},
            {"min_group_size": -5},
            {"design_type": "nope"},
            {"correction_method": "banana"},
        ],
    )
    def test_refusal_every_one_of_these_used_to_be_accepted(self, kwargs):
        with pytest.raises(ConfigurationError):
            ExperimentConfig(**kwargs)

    def test_refusal_a_floor_of_zero_counted_an_empty_arm_as_analysed(self):
        """The reachable consequence. Before the refusal, an intersection with
        rows in one arm and NONE in the other was reported as analysed:
        n_intersections_analysed=3, n_intersections_excluded=0, and its whole
        row of the CATE frame was NaN."""
        with pytest.raises(ConfigurationError) as excinfo:
            ExperimentConfig(min_group_size=0)
        assert "at least 1" in str(excinfo.value)

    def test_control_a_legal_configuration_is_untouched(self):
        cfg = ExperimentConfig()
        assert (cfg.alpha, cfg.n_bootstrap, cfg.min_group_size) == (0.05, 2000, 30)
        assert cfg.design_type is DesignType.SIMPLE_AB
        tight = ExperimentConfig(
            alpha=0.10,
            n_bootstrap=1,
            min_group_size=1,
            design_type=DesignType.CLUSTER,
            correction_method="none",
        )
        assert tight.min_group_size == 1, "a one-row floor is a legal small design"
        assert tight.alpha == 0.10 and tight.correction_method == "none", (
            "validation must not rewrite what it accepts"
        )


# ---------------------------------------------------------------------------
# 3. get_summary: 'most_affected' named a superlative and returned the first
# ---------------------------------------------------------------------------


class TestMostAffectedIsTheMostAffected:
    def _run(self):
        rng = np.random.default_rng(5)
        # 'f' is moved by ~0.3 and 'm' by ~0.9, so the most affected group is
        # NOT the alphabetically first one.
        control = pd.DataFrame({"g": ["f"] * 200 + ["m"] * 200, "y": rng.normal(0.0, 1.0, 400)})
        treatment = pd.DataFrame(
            {
                "g": ["f"] * 200 + ["m"] * 200,
                "y": np.concatenate([rng.normal(0.3, 1.0, 200), rng.normal(0.9, 1.0, 200)]),
            }
        )
        exp = _experiment(control, treatment)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = exp.run_full_analysis()
        return exp, result

    def test_refusal_the_largest_effect_is_the_one_reported(self):
        exp, result = self._run()
        summary = exp.get_summary()
        significant = [e for e in result.intersection_effects if e.significant is True]
        assert len(significant) == 2, "the fixture needs two graded effects to rank"
        # Derived from the result, not quoted: whichever intersection carries
        # the largest |effect| is the one the key must name.
        expected = max(significant, key=lambda e: abs(e.effect)).intersection
        assert summary["most_affected"] == expected
        first_in_order = significant[0].intersection
        assert expected != first_in_order, (
            "the fixture no longer distinguishes 'largest' from 'first'; the pin "
            "cannot fail as written and must be rebuilt"
        )

    def test_the_summary_discloses_intersections_nobody_graded(self):
        """Three states at the summary surface: significant / not significant /
        never tested. Two constant arms give a Welch t of nan, so the
        intersection carries no verdict."""
        n = 40
        control = pd.DataFrame({"g": ["f"] * n, "y": [1.0] * n})
        treatment = pd.DataFrame({"g": ["f"] * n, "y": [1.0] * n})
        exp = _experiment(control, treatment, min_group_size=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp.run_full_analysis()
        summary = exp.get_summary()
        assert summary["n_intersections_analysed"] == 1
        assert summary["n_significant_intersections"] == 0
        assert summary["n_intersections_not_graded"] == 1
        assert summary["most_affected"] is None

    def test_control_a_single_significant_group_is_still_named(self):
        exp, result = self._run()
        summary = exp.get_summary()
        assert summary["n_significant_intersections"] == 2
        assert summary["n_intersections_not_graded"] == 0
        assert summary["most_affected"] in {("f",), ("m",)}


# ---------------------------------------------------------------------------
# 4 + 5 + 6. A ship recommendation over a comparison nobody made
# ---------------------------------------------------------------------------


class TestARecommendationNeedsAComparisonUnderIt:
    def _all_excluded(self):
        """1600 rows per arm, min_group_size=2000, so BOTH intersections are
        excluded before anything is compared while the power figures are still
        computed and published (0.98 each), which is what let the deploy
        through."""
        control, treatment = _arms(800, 0.5, seed=3)
        exp = _experiment(control, treatment, min_group_size=2000, n_bootstrap=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = exp.run_full_analysis()
        assert result.n_intersections == 0 and result.n_excluded == 2
        assert all(v >= 0.8 for v in result.power_results.values()), (
            "the fixture needs well-powered intersections, or the power modifier "
            "downgrades the deploy for an unrelated reason and the pin proves nothing"
        )
        return exp, result

    def test_refusal_no_harmed_groups_may_not_be_said_when_none_was_graded(self):
        exp, result = self._all_excluded()
        rec, _ = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).decision_recommendation()
        )
        joined = " ".join(rec.reasoning)
        assert "with no harmed groups" not in joined
        assert "NOT ASSESSED" in joined
        assert "COULD NOT BE DETERMINED" in joined
        assert any("not as safety" in c for c in rec.caveats)

    def test_refusal_the_deploy_is_downgraded(self):
        exp, result = self._all_excluded()
        rec, _ = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).decision_recommendation()
        )
        assert rec.decision.value == "extend_experiment"
        assert rec.confidence < 0.7225, rec.confidence

    def test_refusal_the_cate_frame_carries_its_schema_and_its_coverage(self):
        exp, result = self._all_excluded()
        frame, messages = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).heterogeneous_treatment_effects()
        )
        assert frame.empty
        assert tuple(frame.columns) == _CATE_COLUMNS
        assert frame.attrs["coverage"] == {
            "n_intersections": 0,
            "n_excluded": 2,
            "measured": False,
            "notes": frame.attrs["coverage"]["notes"],
        }
        assert "EXCLUDED" in frame.attrs["coverage"]["notes"]
        assert any("EMPTY frame" in m and "could-not-check" in m for m in messages)

    def test_refusal_the_report_still_has_an_intersectional_section(self):
        """The surface a reader reads. The section used to be dropped entirely,
        so the report carried a deploy recommendation and no mention of any
        demographic group."""
        exp, result = self._all_excluded()
        sections, _ = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).to_report_sections()
        )
        titles = [s["title"] for s in sections]
        assert "Intersectional Treatment Effects" in titles
        content = next(
            s["content"] for s in sections if s["title"] == "Intersectional Treatment Effects"
        )
        assert "COULD NOT CHECK" in content
        assert "absence of evidence" in content

    def test_refusal_a_nan_effect_is_not_a_group_that_came_to_no_harm(self):
        """`e.effect < 0` is False for nan, so an intersection with no
        comparison at all passed the harm filter as though it had been graded
        and found safe. Hand-built, because the config floor now stops the
        producer from creating this shape."""
        effects = [
            IntersectionEffect(
                intersection=("winners",),
                control_mean=0.0,
                treatment_mean=2.0,
                effect=2.0,
                ci_lower=1.0,
                ci_upper=3.0,
                p_value=0.001,
                effect_size_d=0.5,
                n_control=50,
                n_treatment=50,
                significant=True,
                powered=True,
            ),
            IntersectionEffect(
                intersection=("nobody_compared",),
                control_mean=float("nan"),
                treatment_mean=float("nan"),
                effect=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                p_value=float("nan"),
                effect_size_d=float("nan"),
                n_control=5,
                n_treatment=0,
                significant=None,
                powered=None,
            ),
        ]
        result = ExperimentResult(
            overall_effect=1.0,
            overall_ci=(0.5, 1.5),
            overall_p_value=0.001,
            intersection_effects=effects,
            heterogeneity_detected=False,
            heterogeneity_p_value=0.4,
            power_results={str(e.intersection): 0.9 for e in effects},
            n_intersections=2,
        )
        rec, _ = _caught(lambda: ExperimentAnalysis(result).decision_recommendation())
        assert "with no harmed groups" not in " ".join(rec.reasoning)
        assert "no measurable effect" in " ".join(rec.reasoning)
        assert rec.decision.value == "extend_experiment"

    def test_control_a_fully_graded_experiment_still_deploys_at_full_confidence(self):
        """THE OVER-CORRECTION CONTROL for all of section 4. Every intersection
        compared, every one graded, all positive, all adequately powered: the
        phrase is true and must be said, the decision must be deploy, and the
        confidence must be the undiminished 0.85."""
        control, treatment = _arms(450, 0.5, seed=4)
        exp = _experiment(control, treatment, n_bootstrap=300)
        result = exp.run_full_analysis()
        assert result.n_excluded == 0 and result.n_intersections == 2
        assert all(e.significant is True for e in result.intersection_effects)
        assert all(e.powered is True for e in result.intersection_effects)
        rec, messages = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).decision_recommendation()
        )
        assert rec.decision.value == "deploy_treatment"
        assert rec.confidence == pytest.approx(0.85)
        assert "with no harmed groups" in " ".join(rec.reasoning)
        assert not any("NOT ASSESSED" in line for line in rec.reasoning)
        assert not messages

    def test_control_a_populated_cate_frame_is_untouched(self):
        control, treatment = _arms(450, 0.5, seed=4)
        exp = _experiment(control, treatment, n_bootstrap=300)
        result = exp.run_full_analysis()
        frame, messages = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).heterogeneous_treatment_effects()
        )
        assert tuple(frame.columns) == _CATE_COLUMNS, (
            "the empty frame's schema is pinned against the populated one, so the "
            "two cannot drift apart"
        )
        assert len(frame) == 2
        assert frame.attrs["coverage"]["measured"] is True
        assert not messages
        # The numbers in the frame are the result's own, not recomputed.
        by_ix = {str(e.intersection): e for e in result.intersection_effects}
        for _, row in frame.iterrows():
            assert row["cate"] == pytest.approx(by_ix[row["intersection"]].effect, abs=5e-7)

    def test_control_the_technical_section_prints_a_measured_effect(self):
        control, treatment = _arms(450, 0.5, seed=4)
        exp = _experiment(control, treatment, n_bootstrap=300)
        result = exp.run_full_analysis()
        sections = ExperimentAnalysis(result, experiment=exp).to_report_sections()
        technical = next(s for s in sections if s["tier"] == "technical")
        assert f"{result.overall_effect:.6f}" in technical["content"]
        assert "NOT MEASURED" not in technical["content"]

    def test_the_technical_section_says_not_measured_instead_of_nan(self):
        """`f"{nan:.6f}"` renders "nan", which in a line of numbers reads as a
        measurement that went wrong rather than one nobody made."""
        result = ExperimentResult(
            overall_effect=float("nan"),
            overall_ci=(float("nan"), float("nan")),
            overall_p_value=float("nan"),
            intersection_effects=[],
            heterogeneity_detected=None,
            heterogeneity_p_value=float("nan"),
            n_intersections=0,
            n_excluded=3,
        )
        sections, _ = _caught(lambda: ExperimentAnalysis(result).to_report_sections())
        technical = next(s for s in sections if s["tier"] == "technical")
        assert "nan" not in technical["content"]
        assert "NOT MEASURED" in technical["content"]
        assert "no confidence interval reported" in technical["content"]


# ---------------------------------------------------------------------------
# 6. compute_pareto_frontier: an empty maximize list inverted the frontier
# ---------------------------------------------------------------------------


class TestParetoDirectionIsTheCallersNotTheDefaults:
    COSTS = {"expensive": {"cost": 10.0}, "cheap": {"cost": 1.0}}

    def test_refusal_maximize_nothing_minimises_everything(self):
        points = ExperimentAnalysis(None).compute_pareto_frontier(self.COSTS, maximize=[])
        optimal = {p.variant for p in points if p.is_pareto_optimal}
        assert optimal == {"cheap"}, (
            "an empty maximize list is 'every metric is a cost'; it was answered "
            "with 'maximise all of them' and the dearest variant came out optimal"
        )

    def test_control_the_default_still_maximises(self):
        points = ExperimentAnalysis(None).compute_pareto_frontier(self.COSTS)
        optimal = {p.variant for p in points if p.is_pareto_optimal}
        assert optimal == {"expensive"}

    def test_control_an_explicit_list_is_honoured_in_both_directions(self):
        analysis = ExperimentAnalysis(None)
        both = {
            "a": {"revenue": 1.0, "disparity": 0.3},
            "b": {"revenue": 0.9, "disparity": 0.1},
        }
        points = analysis.compute_pareto_frontier(both, maximize=["revenue"])
        assert {p.variant for p in points if p.is_pareto_optimal} == {"a", "b"}
        # With disparity maximised too, 'a' dominates on both axes.
        points = analysis.compute_pareto_frontier(both, maximize=["revenue", "disparity"])
        assert {p.variant for p in points if p.is_pareto_optimal} == {"a"}


# ---------------------------------------------------------------------------
# 7. mediation_analysis: a proportion of an effect that was never established
# ---------------------------------------------------------------------------


class _Arms:
    """The minimal object mediation_analysis needs (it reads three attributes)."""

    def __init__(self, control, treatment, outcome="y"):
        self.control = control
        self.treatment = treatment
        self.outcome = outcome


class TestProportionMediatedNeedsAnEffectToApportion:
    def test_refusal_no_total_effect_means_no_proportion(self):
        """Measured before the fix: total_effect -0.010805, indirect_effect
        +2.167258 (200 times the total), proportion_mediated 1.0. The clamp
        alone produced that 1.0."""
        rng = np.random.default_rng(11)
        n = 300
        base = rng.normal(0, 1, n)
        exp = _Arms(
            pd.DataFrame({"y": base, "med": base}),
            pd.DataFrame({"y": rng.normal(0, 1, n), "med": rng.normal(0, 1, n) + 5}),
        )
        decomposition, messages = _caught(
            lambda: ExperimentAnalysis(None, experiment=exp).mediation_analysis("med")
        )
        assert decomposition.steps_satisfied["step1_treatment_to_outcome"] is False
        assert math.isnan(decomposition.proportion_mediated)
        assert math.isnan(decomposition.to_dict()["proportion_mediated"])
        assert any("step 1" in m for m in messages), messages
        # The effects themselves ARE measured and are still reported.
        assert math.isfinite(decomposition.total_effect)
        assert math.isfinite(decomposition.indirect_effect)

    def test_refusal_no_fittable_row_means_no_decomposition(self):
        """Measured before the fix, with the mediator column entirely missing
        so ZERO rows survived: total 0.0, direct 0.0, indirect 0.0, proportion
        0.0 and step 3 SATISFIED, announced by nothing but a numpy 'Mean of
        empty slice'."""
        n = 50
        rng = np.random.default_rng(2)
        exp = _Arms(
            pd.DataFrame({"y": rng.normal(0, 1, n), "med": [np.nan] * n}),
            pd.DataFrame({"y": rng.normal(1, 1, n), "med": [np.nan] * n}),
        )
        decomposition, messages = _caught(
            lambda: ExperimentAnalysis(None, experiment=exp).mediation_analysis("med")
        )
        assert math.isnan(decomposition.total_effect)
        assert math.isnan(decomposition.direct_effect)
        assert math.isnan(decomposition.indirect_effect)
        assert math.isnan(decomposition.proportion_mediated)
        assert set(decomposition.steps_satisfied.values()) == {None}
        assert any("mediation fit" in m for m in messages), messages

    def test_refusal_one_arm_only_is_no_contrast_to_decompose(self):
        rng = np.random.default_rng(3)
        n = 40
        exp = _Arms(
            pd.DataFrame({"y": rng.normal(0, 1, n), "med": rng.normal(0, 1, n)}),
            pd.DataFrame({"y": [], "med": []}),
        )
        decomposition, messages = _caught(
            lambda: ExperimentAnalysis(None, experiment=exp).mediation_analysis("med")
        )
        assert math.isnan(decomposition.proportion_mediated)
        assert set(decomposition.steps_satisfied.values()) == {None}
        assert any("same arm" in m for m in messages), messages

    def test_control_a_genuine_chain_still_reports_its_measured_share(self):
        """OVER-CORRECTION CONTROL, with the share derived here from the paths
        the method reports, so the assertion is arithmetic and not a quote."""
        rng = np.random.default_rng(7)
        n = 400
        m_c = rng.normal(0, 1, n)
        m_t = rng.normal(2, 1, n)
        exp = _Arms(
            pd.DataFrame({"med": m_c, "y": 1.5 * m_c + rng.normal(0, 1, n)}),
            pd.DataFrame({"med": m_t, "y": 1.5 * m_t + rng.normal(0, 1, n)}),
        )
        decomposition, messages = _caught(
            lambda: ExperimentAnalysis(None, experiment=exp).mediation_analysis("med")
        )
        assert decomposition.steps_satisfied["step1_treatment_to_outcome"] is True
        assert decomposition.steps_satisfied["step4_mediator_significant"] is True
        raw = abs(decomposition.indirect_effect / decomposition.total_effect)
        assert decomposition.proportion_mediated_raw == pytest.approx(raw)
        assert decomposition.proportion_mediated == pytest.approx(min(raw, 1.0))
        assert decomposition.proportion_mediated > 0.5, "a full mediation must read as one"
        assert not any("step 1" in m for m in messages)

    def test_control_a_partial_mediation_reports_a_share_strictly_inside_the_range(self):
        rng = np.random.default_rng(13)
        n = 500
        m_c = rng.normal(0, 1, n)
        m_t = rng.normal(1.0, 1, n)
        # Half the treatment effect travels through the mediator, half direct.
        exp = _Arms(
            pd.DataFrame({"med": m_c, "y": 1.0 * m_c + rng.normal(0, 1, n)}),
            pd.DataFrame({"med": m_t, "y": 1.0 * m_t + 1.0 + rng.normal(0, 1, n)}),
        )
        decomposition, _ = _caught(
            lambda: ExperimentAnalysis(None, experiment=exp).mediation_analysis("med")
        )
        share = decomposition.proportion_mediated
        assert 0.1 < share < 0.9, share
        assert share == pytest.approx(
            abs(decomposition.indirect_effect / decomposition.total_effect)
        )


# ---------------------------------------------------------------------------
# 8. temporal_stability_check: the threshold decided the verdict
# ---------------------------------------------------------------------------


class TestTemporalStabilityRefusesAThresholdThatDecidesAlone:
    def _analysis(self, n=300, effect=1.0):
        rng = np.random.default_rng(20260930)
        t = np.arange(n)
        control = pd.DataFrame({"t": t, "g": ["A"] * n, "y": rng.normal(0, 1.0, n)})
        treatment = pd.DataFrame({"t": t, "g": ["A"] * n, "y": rng.normal(0, 1.0, n) + effect})
        exp = _experiment(control, treatment, min_group_size=10, n_bootstrap=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = exp.run_full_analysis()
        return ExperimentAnalysis(result, experiment=exp)

    @pytest.mark.parametrize(
        "alpha, was",
        [(0.0, True), (1.0, False), (-1.0, True), (float("nan"), False)],
        ids=["zero", "one", "negative", "nan"],
    )
    def test_refusal_an_alpha_outside_the_open_unit_interval(self, alpha, was):
        """`was` is the is_stable this returned for a trend p-value of 0.6003.
        The nan case is the worst: a definite "NOT stable over time" out of a
        threshold nobody supplied, from data with no trend in it."""
        with pytest.raises(ConfigurationError) as excinfo:
            self._analysis().temporal_stability_check("t", n_periods=5, alpha=alpha)
        assert "alpha" in str(excinfo.value)

    @pytest.mark.parametrize("n_periods", [1, 0, -3, 2.5])
    def test_refusal_a_trend_needs_at_least_two_periods(self, n_periods):
        with pytest.raises(ConfigurationError) as excinfo:
            self._analysis().temporal_stability_check("t", n_periods=n_periods)
        assert "n_periods" in str(excinfo.value)

    def test_control_a_steady_effect_still_reads_as_stable(self):
        result, messages = _caught(
            lambda: self._analysis().temporal_stability_check("t", n_periods=5, alpha=0.05)
        )
        assert result.is_stable is True
        assert result.n_periods_measured == 5
        assert result.trend_p_value >= 0.05
        assert abs(result.trend_slope) < 0.2
        assert not messages

    def test_control_a_drifting_effect_still_reads_as_unstable(self):
        rng = np.random.default_rng(20260930)
        n = 300
        t = np.arange(n)
        period = np.minimum(t // (n // 5), 4)
        control = pd.DataFrame({"t": t, "g": ["A"] * n, "y": rng.normal(0, 1.0, n)})
        treatment = pd.DataFrame(
            {"t": t, "g": ["A"] * n, "y": rng.normal(0, 1.0, n) + period * 0.8}
        )
        exp = _experiment(control, treatment, min_group_size=10, n_bootstrap=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = exp.run_full_analysis()
        out, _ = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).temporal_stability_check(
                "t", n_periods=5
            )
        )
        assert out.is_stable is False
        assert out.trend_slope == pytest.approx(0.8, abs=0.15)


# ---------------------------------------------------------------------------
# 9. Absence in a protected attribute: six doors, one answer
# ---------------------------------------------------------------------------


class TestAnAbsentDemographicValueIsNotADemographicGroup:
    ABSENT = [np.nan, pd.NA, pd.NaT, "", "   ", "None", "nan", "<NA>"]

    def _frames(self, absent):
        rng = np.random.default_rng(31)
        g = ["f"] * 40 + ["m"] * 40 + [absent] * 20
        return (
            pd.DataFrame({"g": g, "y": rng.normal(0, 1, 100)}),
            pd.DataFrame({"g": g, "y": rng.normal(0.6, 1, 100)}),
        )

    @pytest.mark.parametrize("absent", ABSENT, ids=[repr(v) for v in ABSENT])
    def test_refusal_no_group_is_minted_and_nothing_crashes(self, absent):
        """Before the fix: '' and 'None' became groups with published effects
        (+0.448 and +0.108 over 20 rows per arm), counted among "3 analysed",
        while pd.NA raised TypeError('boolean value of NA is ambiguous') out of
        _get_intersection_data and killed the whole analysis."""
        control, treatment = self._frames(absent)
        exp, messages = _caught(lambda: _experiment(control, treatment, min_group_size=10))
        assert exp.intersections == [("f",), ("m",)]
        assert exp.rows_without_intersection == 40
        assert any("ABSENT value" in m and "40 row(s)" in m for m in messages), messages

    @pytest.mark.parametrize("absent", ABSENT, ids=[repr(v) for v in ABSENT])
    def test_refusal_the_rows_reach_the_recommendation(self, absent):
        control, treatment = self._frames(absent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = _experiment(control, treatment, min_group_size=10)
            result = exp.run_full_analysis()
        assert result.metadata["rows_without_intersection"] == 40
        rec, _ = _caught(
            lambda: ExperimentAnalysis(result, experiment=exp).decision_recommendation()
        )
        assert "with no harmed groups" not in " ".join(rec.reasoning)
        assert "protected attribute is absent" in " ".join(rec.reasoning)

    @pytest.mark.parametrize("labels", [("f", "m"), (0, 1), ("NA", "EU")])
    def test_control_a_real_label_is_still_a_group_and_raises_nothing(self, labels):
        """OVER-CORRECTION CONTROL. 0 is a perfectly good binary group label and
        is falsy; 'NA' is a region. Refusing either would delete real groups."""
        rng = np.random.default_rng(17)
        g = [labels[0]] * 50 + [labels[1]] * 50
        control = pd.DataFrame({"g": g, "y": rng.normal(0, 1, 100)})
        treatment = pd.DataFrame({"g": g, "y": rng.normal(0.6, 1, 100)})
        exp, messages = _caught(lambda: _experiment(control, treatment, min_group_size=10))
        assert len(exp.intersections) == 2
        assert exp.rows_without_intersection == 0
        assert not messages

    def test_control_no_protected_attribute_still_means_overall(self):
        control, treatment = _arms(50, 0.5)
        exp = FairnessExperiment(
            control, treatment, [], "y", config=ExperimentConfig(n_bootstrap=50, random_state=1)
        )
        assert exp.intersections == [("overall",)]
        power, _ = _caught(lambda: exp.calculate_intersectional_power(effect_size=0.5))
        assert list(power) == [("overall",)]
        assert power[("overall",)] == pytest.approx(round(_expected_power(100, 0.5), 4), abs=1e-3)


# ---------------------------------------------------------------------------
# 10. spillover_detection: rows with no cluster id left the result in silence
# ---------------------------------------------------------------------------


class TestSpilloverDisclosesRowsItCouldNotPlace:
    def _analysis(self, with_missing):
        rng = np.random.default_rng(77)
        cl = ["P"] * 30 + ["M"] * 30
        y = list(rng.normal(0, 1, 60))
        if with_missing:
            cl = cl + [np.nan] * 20
            y = y + list(rng.normal(5, 1, 20))  # a mean five sigma from the rest
        control = pd.DataFrame({"g": ["f"] * len(cl), "cl": cl, "y": y})
        treatment = pd.DataFrame({"g": ["f"] * 30, "cl": ["M"] * 30, "y": rng.normal(0.5, 1, 30)})
        exp = _experiment(control, treatment, min_group_size=10, n_bootstrap=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp.run_full_analysis()
        return ExperimentAnalysis(exp._result, experiment=exp)

    def test_refusal_the_dropped_rows_are_counted_and_named(self):
        out, messages = _caught(lambda: self._analysis(True).spillover_detection("cl"))
        assert out["n_rows_missing_cluster_id"] == 20
        assert any("carry NO 'cl' value" in m for m in messages), messages

    def test_control_a_complete_design_says_nothing_and_measures_the_same(self):
        clean, clean_messages = _caught(lambda: self._analysis(False).spillover_detection("cl"))
        dirty, _ = _caught(lambda: self._analysis(True).spillover_detection("cl"))
        assert clean["n_rows_missing_cluster_id"] == 0
        assert not [m for m in clean_messages if "cluster id" in m]
        assert clean["spillover_detected"] is False
        assert math.isfinite(clean["p_value"])
        # The verdict is unchanged by the disclosure: only the disclosure is new.
        assert dirty["p_value"] == pytest.approx(clean["p_value"])

    def test_control_a_design_with_no_pure_cluster_still_refuses(self):
        rng = np.random.default_rng(5)
        control = pd.DataFrame({"g": ["f"] * 60, "cl": ["M"] * 60, "y": rng.normal(0, 1, 60)})
        treatment = pd.DataFrame({"g": ["f"] * 30, "cl": ["M"] * 30, "y": rng.normal(0.5, 1, 30)})
        exp = _experiment(control, treatment, min_group_size=10, n_bootstrap=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp.run_full_analysis()
        out, messages = _caught(
            lambda: ExperimentAnalysis(exp._result, experiment=exp).spillover_detection("cl")
        )
        assert out["spillover_detected"] is None
        assert math.isnan(out["p_value"])
        assert any("could not check" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# 11. A confidence interval whose bounds no resample supports
# ---------------------------------------------------------------------------


class TestBootstrapBoundsNeedEnoughDraws:
    """The floor is `ceil(2/alpha) - 1` (39 at alpha=0.05), which is the rule
    ece_confidence_intervals derives and is pinned at its own 38/39 boundary.
    Asserted as the boundary and not as a round number."""

    def _run(self, n_bootstrap):
        control, treatment = _arms(60, 0.6, seed=9, groups=("f",))
        exp = _experiment(control, treatment, min_group_size=10, n_bootstrap=n_bootstrap)
        result, messages = _caught(exp.run_full_analysis)
        return result.intersection_effects[0], result, messages

    def test_refusal_one_resample_is_not_a_ninety_five_percent_interval(self):
        effect, result, messages = self._run(1)
        assert math.isnan(effect.ci_lower) and math.isnan(effect.ci_upper)
        assert all(math.isnan(b) for b in result.overall_ci)
        assert any("order statistics" in m for m in messages), messages

    def test_the_floor_is_the_order_statistic_count_not_a_round_number(self):
        below, _, below_messages = self._run(38)
        at, _, at_messages = self._run(39)
        assert math.isnan(below.ci_lower), "0.025 * (38 + 1) < 1: no order statistic"
        assert any("order statistics" in m for m in below_messages)
        assert math.isfinite(at.ci_lower) and math.isfinite(at.ci_upper)
        assert not [m for m in at_messages if "order statistics" in m]
        assert at.ci_lower < at.effect < at.ci_upper

    def test_control_the_effect_and_the_p_value_are_untouched_by_the_floor(self):
        below, _, _ = self._run(38)
        at, _, _ = self._run(39)
        assert below.effect == pytest.approx(at.effect)
        assert below.p_value == pytest.approx(at.p_value)
        assert math.isfinite(below.effect) and math.isfinite(below.p_value)

    def test_control_a_normal_resample_count_is_silent_and_bracketing(self):
        effect, result, messages = self._run(400)
        assert not [m for m in messages if "order statistics" in m]
        assert effect.ci_lower < effect.effect < effect.ci_upper
        assert result.overall_ci[0] < result.overall_effect < result.overall_ci[1]


# ---------------------------------------------------------------------------
# 12. The power dataclasses: executed, and checked for minting
# ---------------------------------------------------------------------------


class TestThePowerDataclassesDoNotMintValues:
    def test_a_config_field_nothing_reads_says_so(self):
        """`PowerConfig.effect_sizes` is documented as "Cohen's d values to
        evaluate" and is read NOWHERE in src/: every analyzer method takes its
        own `effect_size` parameter. Setting it was a silent no-op, so a caller
        who set it believed the analysis used it. Whether it should drive those
        defaults or leave the public config is a product decision; until then
        the no-op is audible."""
        from vfairness.operations.experimentation.power import PowerConfig

        _, messages = _caught(lambda: PowerConfig(effect_sizes=[0.3]))
        assert any("NOT READ" in m for m in messages), messages

    def test_control_the_default_config_is_silent_and_keeps_its_defaults(self):
        from vfairness.operations.experimentation.power import PowerConfig

        config, messages = _caught(PowerConfig)
        assert config.effect_sizes == [0.2, 0.5, 0.8]
        assert not messages
        explicit, messages = _caught(lambda: PowerConfig(effect_sizes=[0.2, 0.5, 0.8]))
        assert not messages, "asking for the default is not asking for a no-op"

    @pytest.mark.parametrize(
        "kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"alpha": float("nan")}, {"target_power": 1.0}]
    )
    def test_control_the_existing_config_refusals_still_fire(self, kwargs):
        from vfairness.operations.experimentation.power import PowerConfig

        with pytest.raises(ConfigurationError):
            PowerConfig(**kwargs)

    def test_an_empty_sampling_plan_still_carries_its_columns(self):
        from vfairness.operations.experimentation.power import _PLAN_COLUMNS, SamplingPlan

        frame = SamplingPlan({}, {}, []).to_dataframe()
        assert frame.empty
        assert tuple(frame.columns) == _PLAN_COLUMNS

    def test_control_a_populated_plan_keeps_the_same_columns_and_its_none(self):
        from vfairness.operations.experimentation.power import _PLAN_COLUMNS, SamplingPlan

        plan = SamplingPlan(
            allocations={("f",): 63},
            rationale={("f",): "power 0.31", ("m",): "NOT ALLOCATED: budget exhausted"},
            priority_order=[("f",), ("m",)],
            total_budget=63,
            n_underpowered=2,
            not_allocated=[("m",)],
        )
        frame = plan.to_dataframe()
        assert tuple(frame.columns) == _PLAN_COLUMNS
        assert frame.loc[0, "additional_samples"] == 63
        # pandas stores the None as NaN in a float column, which is the
        # could-not-check value this library uses; what matters is that it is
        # NOT 0, the number a powered intersection gets.
        assert pd.isna(frame.loc[1, "additional_samples"]), (
            "a budget the plan never reached is not an allocation of zero"
        )

    def test_the_result_dataclasses_pass_a_non_finite_value_through(self):
        """None of these may round a could-not-check into a number. Executed
        rather than read: `round(nan, 4)` is nan, and that is what must land in
        the dict a reader sees."""
        from vfairness.operations.experimentation.power import (
            PowerResult,
            SequentialTestResult,
            SPRTDecision,
        )

        power = PowerResult(("f",), float("nan"), 63, None).to_dict()
        assert math.isnan(power["power"]) and power["is_powered"] is None
        sprt = SequentialTestResult(
            ("f",), SPRTDecision.COULD_NOT_CHECK, float("nan"), 2.77, -2.25, False
        ).to_dict()
        assert sprt["decision"] == "could_not_check"
        assert math.isnan(sprt["log_likelihood_ratio"])
        # The boundaries ARE measured and must not be blanked with the ratio.
        assert sprt["upper_boundary"] == 2.77 and sprt["lower_boundary"] == -2.25

    def test_control_a_measured_power_result_is_reported_as_measured(self):
        from vfairness.operations.experimentation.power import PowerResult

        measured = PowerResult(("f",), 0.7818, 63, True, n_control=60, n_treatment=60).to_dict()
        assert measured["power"] == 0.7818 and measured["is_powered"] is True
