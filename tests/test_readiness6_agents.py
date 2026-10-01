"""A detector that CANNOT fire must not print "not significant".

READINESS-6, 2026-09-10, agent / LLM / experimentation lane. Sibling of
``test_readiness6_design_power.py``, which pins the shared helpers
(``min_attainable_p_*`` and ``detectability``) those fixes are built on.

Every number in this file was MEASURED on the code before the fix, by running
it. The defects come in three shapes:

1. DISCRETE FLOORS. A test that runs but whose p-value cannot reach the
   threshold it is graded against, for ANY data. Kendall's tau on 3 rounds
   floors at 0.3333; Fisher's exact on 3 against 3 floors at 0.10; a per-tool
   Fisher test on a tool used once floors at 1.0; a sample-level flip
   permutation on 8 samples floors at 0.0625; a rank-sum on 2 texts against 6
   floors at 0.0714; WEAT on 3 words against 3 floors at 0.1.
2. COLLAPSED THIRD STATES. A could-not-check written down as a clean negative:
   `bool(rejection_mask[i])` for a p-value that was never computed, `0.0` for a
   power that was never calculated, `1e6` for a ratio that is undefined.
3. OVER-REPORTING, which is the same defect facing the other way: a 0.001
   noise gap graded CRITICAL because the amplification ratio it was graded on
   was a sentinel.

Every class below carries an OVER-CORRECTION CONTROL: real groupthink with
enough rounds still fires, a fair delegation still reports not-significant, a
well-powered experiment still DEPLOYS, and a genuine emergent bias is still
critical. A checker that reports every design dead is as useless as no checker.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import detectability
from vfairness.exceptions import ConfigurationError
from vfairness.llm.embedding_bias import EmbeddingBiasDetector
from vfairness.llm.text_fairness import TextFairnessAnalyzer
from vfairness.multi_agent.collusion import AdversarialCollusionDetector
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.multi_agent.emergent import EmergentBiasDetector
from vfairness.multi_agent.groupthink import GroupthinkDetector
from vfairness.multi_agent.negotiation import NegotiationFairnessTracker
from vfairness.operations.experimentation.experiment import (
    ExperimentConfig,
    FairnessExperiment,
    IntersectionEffect,
)
from vfairness.operations.experimentation.power import (
    FairnessPowerAnalyzer,
    PowerConfig,
)


def _messages(recorded, needle):
    return [str(w.message) for w in recorded if needle in str(w.message)]


# ---------------------------------------------------------------------------
# 1. GroupthinkDetector: the Kendall floor, and an undefined cosine
# ---------------------------------------------------------------------------


class TestGroupthinkCannotFireOnThreeRounds:
    """Measured before the fix on the class docstring's own example shape:
    two agents going from perfectly opposite to literally identical over 3
    rounds gave trajectory [0.0, 0.724, 1.0], tau=1.0, p=0.3333 and
    has_groupthink=False, in silence. 2/3! = 0.3333 is the floor, so no
    behaviour whatsoever could have cleared the 0.1 threshold."""

    OPPOSITE_TO_IDENTICAL = [
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        {"a": [0.7, 0.3], "b": [0.3, 0.7]},
        {"a": [0.5, 0.5], "b": [0.5, 0.5]},
    ]

    def test_total_convergence_over_three_rounds_is_not_assessed(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = GroupthinkDetector().analyze_convergence(self.OPPOSITE_TO_IDENTICAL)

        assert result.has_groupthink is None, (
            "3 rounds cannot reach the 0.1 trend threshold; False is a verdict "
            "about a test that could not fire"
        )
        assert result.trend_detectable is False
        assert "NOT DETECTABLE" in result.trend_note
        assert result.convergence_trajectory[-1] == pytest.approx(1.0)
        named = _messages(w, "convergence trend over 3 rounds was NOT ASSESSED")
        assert named and "NOT False" in named[0]

    def test_total_agreement_every_round_is_none_not_false(self):
        """tau is NaN for a series with no variance, and `nan > 0.3` is False:
        the strongest echo chamber there is read as no groupthink. The class
        docstring already promised None here."""
        flat = [{"a": [1.0, 1.0], "b": [1.0, 1.0]} for _ in range(3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = GroupthinkDetector().analyze_convergence(flat)

        assert result.has_groupthink is None
        assert result.echo_chamber_score == pytest.approx(1.0)
        assert _messages(w, "no variance to rank")

    def test_an_undefined_cosine_is_nan_not_the_bottom_of_the_scale(self):
        """0.0 is the MINIMUM of the convergence scale, i.e. total
        disagreement, and it was returned for a pair with no angle between
        them. Measured: three rounds of two agents holding the identical
        opinion read [0.0, 0.0, 0.0] when that opinion was encoded [0, 0] and
        [1.0, 1.0, 1.0] when it was encoded [0.001, 0]."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim = GroupthinkDetector._cosine_similarity(np.zeros(2), np.zeros(2))
        assert math.isnan(sim)
        assert _messages(w, "cosine similarity is undefined")

        zeros = [{"a": [0.0, 0.0], "b": [0.0, 0.0]} for _ in range(3)]
        near = [{"a": [0.001, 0.0], "b": [0.001, 0.0]} for _ in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_zero = GroupthinkDetector().analyze_convergence(zeros)
            r_near = GroupthinkDetector().analyze_convergence(near)
        assert r_zero.has_groupthink is r_near.has_groupthink is None, (
            "identical agent behaviour must not produce opposite verdicts "
            "because of how the label was encoded"
        )

    def test_control_real_groupthink_with_enough_rounds_still_fires(self):
        """OVER-CORRECTION CONTROL."""
        rounds = []
        for t in np.linspace(0.0, 1.0, 6):
            rounds.append(
                {
                    "a": [1.0 - 0.5 * t, 0.0 + 0.5 * t],
                    "b": [0.0 + 0.5 * t, 1.0 - 0.5 * t],
                    "c": [0.5 + 0.3 * (1 - t), 0.5 - 0.3 * (1 - t)],
                }
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = GroupthinkDetector().analyze_convergence(rounds)

        assert result.has_groupthink is True
        assert result.trend_detectable is True
        assert result.trend_note == ""
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert not _messages(w, "NOT ASSESSED")


# ---------------------------------------------------------------------------
# 2. DelegationRoutingAuditor: the Fisher floor and the unreachable fallback
# ---------------------------------------------------------------------------


class TestDelegationRoutingDesignPower:
    def test_total_segregation_on_three_per_group_is_not_assessed(self):
        """Measured before the fix: Cramér's V = 1.0 (perfect association),
        p = 0.10, is_significant=False, no warning. Fisher's floor at 3
        against 3 IS 0.10."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = DelegationRoutingAuditor().analyze(
                ["junior"] * 3 + ["senior"] * 3, ["A"] * 3 + ["B"] * 3
            )

        assert result.cramers_v == pytest.approx(1.0)
        assert result.is_significant is None
        assert result.detectable is False
        assert result.min_attainable_p == pytest.approx(0.10, rel=1e-6)
        named = _messages(w, "routing significance was NOT ASSESSED")
        assert named and "NOT False" in named[0]

    def test_two_decisions_cannot_be_assessed_either(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = DelegationRoutingAuditor().analyze(["junior", "senior"], ["A", "B"])
        assert result.is_significant is None
        assert result.detectable is False

    def test_the_small_sample_fallback_is_reachable_at_last(self):
        """The `sparse` test was `any margin == 0`, and a margin cannot be zero:
        every route and group in the table was observed at least once. So the
        documented "or when the chi-square assumption fails (expected cell
        counts < 5)" path never ran. Measured before the fix on 13 decisions,
        12 from group A over two routes and 1 from B on a third: the asymptotic
        chi-square reported p=0.0015 and is_significant=True on a table whose
        smallest expected count is 0.077, while the exact conditional p is
        0.0769 -- which is also the floor for those margins."""
        routes = ["r1"] * 6 + ["r2"] * 6 + ["r3"]
        demographics = ["A"] * 12 + ["B"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = DelegationRoutingAuditor().analyze(routes, demographics)

        assert result.test_used == "permutation_chi2"
        assert result.p_value > 0.05, (
            f"the asymptotic chi-square answered 0.0015 here; the exact "
            f"conditional answer is 0.0769, got {result.p_value}"
        )
        assert result.p_value == pytest.approx(0.0769, abs=0.02)
        assert result.is_significant is None
        assert result.detectable is False
        assert _messages(w, "below the 5 the asymptotic chi-square needs")

    def test_a_sparse_two_by_two_reaches_fisher_even_without_prefer_fisher(self):
        result = DelegationRoutingAuditor(prefer_fisher=False).analyze(
            ["r1"] * 4 + ["r2"] + ["r2"] * 4 + ["r1"], ["A"] * 5 + ["B"] * 5
        )
        assert result.test_used == "fisher"
        assert result.p_value == pytest.approx(0.2063, abs=1e-3)

    def test_control_a_fair_delegation_at_size_is_not_significant_not_none(self):
        """OVER-CORRECTION CONTROL: a genuinely fair routing must report a
        measured False, not a could-not-check."""
        rng = np.random.default_rng(0)
        routes = list(rng.choice(["junior", "senior"], size=400))
        demographics = ["A"] * 200 + ["B"] * 200
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = DelegationRoutingAuditor().analyze(routes, demographics)

        assert result.is_significant is False
        assert result.detectable is True
        assert result.detectability_note == ""
        assert not _messages(w, "NOT ASSESSED")

    def test_control_real_segregation_at_size_still_fires(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = DelegationRoutingAuditor().analyze(
                ["junior"] * 30 + ["senior"] * 30, ["A"] * 30 + ["B"] * 30
            )
        assert result.is_significant is True
        assert result.detectable is True


# ---------------------------------------------------------------------------
# 3. AdversarialCollusionDetector: 2**n draws, and only two group labels
# ---------------------------------------------------------------------------


def _collusion_case(n: int, pre_value: float = 0.0):
    groups = np.array([0] * (n // 2) + [1] * (n - n // 2))
    pre = {"a": np.full(n, pre_value, dtype=float)}
    post = {"a": np.where(groups == 0, 0.0, 1.0).astype(float)}
    return pre, post, groups


class TestCollusionPermutationFloor:
    """The null flips pre/post per SAMPLE, so it holds at most 2**n distinct
    assignments however many resamples are drawn. Measured before the fix with
    total group separation after interaction: p = 0.2675 at 4 samples, 0.1138
    at 6, 0.0559 at 8 -- all is_collusion=False. Exhaustive enumeration of the
    2**n assignments puts the smallest attainable p at 0.125 (n=6) and 0.0625
    (n=8): no data could have made those runs significant."""

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_a_design_with_no_power_says_so(self, n):
        pre, post, groups = _collusion_case(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = AdversarialCollusionDetector().analyze(pre, post, groups)

        assert result.collusion_score == pytest.approx(1.0)
        assert result.is_collusion is None
        assert result.is_significant is None
        assert result.detectable is False
        assert result.min_attainable_p is not None and result.min_attainable_p > 0.05
        named = _messages(w, "permutation test on")
        assert named and "NOT False" in named[0]

    @pytest.mark.parametrize("n", [10, 12, 16])
    def test_control_the_same_design_with_enough_samples_still_fires(self, n):
        """OVER-CORRECTION CONTROL: the floor falls as 2**n grows, and the
        detector must recover its verdict as soon as it can."""
        pre, post, groups = _collusion_case(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = AdversarialCollusionDetector().analyze(pre, post, groups)
        assert result.is_collusion is True
        assert result.detectable is True
        assert not _messages(w, "NOT ASSESSED")

    def test_the_exact_floor_matches_exhaustive_enumeration(self):
        """The estimate is read off the sampled draws; check it against the
        real thing, which is enumerable at this size."""
        import itertools

        n = 8
        pre, post, groups = _collusion_case(n)
        m0, m1 = groups == 0, groups == 1

        def bias(x):
            return abs(x[m0].mean() - x[m1].mean())

        pre_a, post_a = pre["a"], post["a"]
        observed = bias(post_a) - bias(pre_a)
        stats_all = []
        for pattern in itertools.product([False, True], repeat=n):
            flip = np.array(pattern)
            stats_all.append(
                bias(np.where(flip, pre_a, post_a)) - bias(np.where(flip, post_a, pre_a))
            )
        stats_all = np.array(stats_all)
        exact_floor = float((stats_all >= max(stats_all.max(), observed) - 1e-12).sum()) / len(
            stats_all
        )
        assert exact_floor == pytest.approx(0.0625)
        assert detectability(exact_floor, n_family=1, alpha=0.05)[0] is False


class TestCollusionComparesEveryGroup:
    """`abs(mean[g0] - mean[g1])` looked at the FIRST TWO labels only while
    metadata['n_samples'] reported the full length. Measured before the fix on
    90 samples in three groups whose post-interaction means were 0.500 / 0.500
    / 0.947: is_collusion=False at p=0.9421, with the 30 samples carrying the
    entire disparity never examined."""

    def _three_groups(self):
        rng = np.random.default_rng(0)
        groups = np.array([0] * 30 + [1] * 30 + [2] * 30)
        pre = {"a": rng.normal(0.5, 0.02, 90)}
        post = {"a": np.where(groups == 2, 0.95, 0.5) + rng.normal(0, 0.02, 90)}
        return pre, post, groups

    def test_a_third_group_is_no_longer_invisible(self):
        pre, post, groups = self._three_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = AdversarialCollusionDetector().analyze(pre, post, groups)

        assert result.n_groups_compared == 3
        assert result.post_mean_bias > 0.4, result.post_mean_bias
        assert result.is_collusion is True
        assert result.p_value < 0.05

    def test_control_two_groups_are_unchanged(self):
        """OVER-CORRECTION CONTROL: max-minus-min IS abs(a - b) for two groups,
        so the binary contract must be bit-for-bit what it was."""
        rng = np.random.default_rng(1)
        groups = np.array([0] * 40 + [1] * 40)
        pre = {"a": rng.normal(0.5, 0.1, 80)}
        post = {"a": rng.normal(0.5, 0.1, 80)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = AdversarialCollusionDetector().analyze(pre, post, groups)
        expected = abs(post["a"][:40].mean() - post["a"][40:].mean())
        assert result.post_bias_per_agent["a"] == pytest.approx(expected)
        assert result.is_collusion is False


# ---------------------------------------------------------------------------
# 4. NegotiationFairnessTracker: a design that runs and cannot fire
# ---------------------------------------------------------------------------


class TestNegotiationTrendDesignPower:
    """Different from the "fewer than 3 measurable turns" case this file's
    subject already handles: here the test RUNS. Measured before the fix on a
    gap that doubles every turn against a flat zero: tau=1.0 (the most extreme
    monotone increase there is) reported trend='stable', is_widening=False at
    3 and at 4 turns."""

    @staticmethod
    def _doubling(turns):
        return [0.1 * 2**i for i in range(turns)], [0.0] * turns

    @pytest.mark.parametrize("turns,floor", [(3, 0.3333), (4, 0.0833)])
    def test_a_doubling_gap_on_a_short_run_is_not_assessed(self, turns, floor):
        a, b = self._doubling(turns)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = NegotiationFairnessTracker().analyze(a, b)

        assert result.mann_kendall_tau == pytest.approx(1.0)
        assert result.trend == "not_assessed"
        assert result.is_widening is None
        assert result.is_significant is None
        assert result.trend_detectable is False
        assert result.min_attainable_p == pytest.approx(floor, abs=1e-4)
        assert _messages(w, "was NOT ASSESSED")

    @pytest.mark.parametrize("turns", [5, 6])
    def test_control_the_same_design_with_enough_turns_still_fires(self, turns):
        """OVER-CORRECTION CONTROL."""
        a, b = self._doubling(turns)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = NegotiationFairnessTracker().analyze(a, b)
        assert result.trend == "widening"
        assert result.is_widening is True
        assert result.trend_detectable is True
        assert not _messages(w, "NOT ASSESSED")

    def test_dropping_one_turn_of_five_is_not_silently_stable(self):
        """Measured before the fix: blanking one turn of a 5-turn widening run
        moved a detectable design into an undetectable one and reported
        trend='stable', is_widening=False; the only warning said the trend
        "rests on the 4 that remain"."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = NegotiationFairnessTracker().analyze(
                [0.1, 0.2, float("nan"), 0.4, 0.8], [0.0] * 5
            )
        assert result.trend == "not_assessed"
        assert result.is_widening is None
        assert result.n_turns_measured == 4
        assert _messages(w, "was NOT ASSESSED")

    def test_control_a_genuinely_stable_negotiation_still_reads_stable(self):
        rng = np.random.default_rng(1)
        a = [0.5 + float(rng.normal(0, 0.01)) for _ in range(6)]
        result = NegotiationFairnessTracker().analyze(a, [0.5] * 6)
        assert result.trend == "stable"
        assert result.is_widening is False
        assert result.trend_detectable is True


# ---------------------------------------------------------------------------
# 5. TextFairnessAnalyzer: the rank-sum floor at 2 texts
# ---------------------------------------------------------------------------


class TestTextFairnessDesignPower:
    DISTINCT = {
        "W1": 0.95,
        "W2": 0.90,
        "M1": 0.20,
        "M2": 0.19,
        "M3": 0.18,
        "M4": 0.17,
        "M5": 0.16,
        "M6": 0.15,
    }

    def _analyzer(self, table):
        return TextFairnessAnalyzer(lambda texts: [table[t] for t in texts])

    def test_two_texts_against_six_distinct_ones_cannot_be_graded(self):
        """Measured before the fix: gap +0.562, perfectly separated, severity
        'info', "No material identity-term bias detected for 'women' (gap
        +0.562, p=0.071)". The exact rank-sum floor for 2 against 6 distinct
        values is 2/C(8,2) = 0.0714."""
        texts = {
            "women": ["W1", "W2"],
            "men": ["M1", "M2", "M3"],
            "doctors": ["M4", "M5", "M6"],
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._analyzer(self.DISTINCT).analyze(texts)

        assert result.severity == "not_assessed"
        assert "NOT ASSESSED" in result.interpretation
        assert "No material identity-term bias" not in result.interpretation
        assert result.max_gap > 0.5
        assert result.p_value == pytest.approx(0.0714, abs=1e-3)
        assert result.notes and "NOT DETECTABLE" in result.notes[0]
        assert _messages(w, "could NOT have reached")

    def test_control_the_same_sizes_with_ties_can_fire_and_still_do(self):
        """OVER-CORRECTION CONTROL, and the reason the floor is read off the
        DATA rather than off the two sample sizes: with ties scipy resolves to
        the tie-corrected asymptotic test, whose floor at 2 against 6 is
        0.0153, and that design really can reach 0.05."""
        tied = {"W1": 0.9, "W2": 0.9, **{f"M{i}": 0.165 for i in range(1, 7)}}
        texts = {
            "women": ["W1", "W2"],
            "men": ["M1", "M2", "M3"],
            "doctors": ["M4", "M5", "M6"],
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._analyzer(tied).analyze(texts)
        assert result.severity == "critical"
        assert result.p_value < 0.05
        assert not _messages(w, "could NOT have reached")

    def test_control_a_fair_classifier_at_size_still_reads_info(self):
        rng = np.random.default_rng(3)
        table = {f"t{i}": float(rng.normal(0.4, 0.05)) for i in range(40)}
        texts = {
            "women": [f"t{i}" for i in range(0, 20)],
            "men": [f"t{i}" for i in range(20, 40)],
        }
        result = self._analyzer(table).analyze(texts)
        assert result.severity == "info"
        assert "No material identity-term bias" in result.interpretation


# ---------------------------------------------------------------------------
# 6. EmbeddingBiasDetector: a dead backend, and the WEAT partition floor
# ---------------------------------------------------------------------------


class TestEmbeddingBiasCouldNotCheck:
    TARGETS = (["career", "office"], ["home", "family"], ["he", "him"], ["she", "her"])

    def test_a_dead_backend_is_not_an_absence_of_bias(self):
        """Measured before the fix with an embed_fn returning all-zero vectors:
        effect_size=0.0, p=1.0, severity 'info', "No statistically significant
        association bias", and not one warning. `denom = std(...) or 1e-12`
        fires exactly when there is nothing to compare."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EmbeddingBiasDetector(embed_fn=lambda words: np.zeros((len(words), 8))).weat(
                *self.TARGETS
            )

        assert math.isnan(result.effect_size)
        assert result.severity == "could_not_check"
        assert "No statistically significant association bias" not in result.interpretation
        assert _messages(w, "zero denominator and is undefined")
        assert _messages(w, "zero length")

    def test_constant_vectors_are_the_same_could_not_check(self):
        const = np.array([1.0, 2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EmbeddingBiasDetector(
                embed_fn=lambda words: np.tile(const, (len(words), 1))
            ).weat(*self.TARGETS)
        assert math.isnan(result.effect_size)
        assert result.severity == "could_not_check"

    def test_three_words_a_side_cannot_reach_alpha(self):
        """C(6,3) = 20 partitions, so the smallest p WEAT can return is
        2/20 = 0.1. Measured before the fix on a planted stereotype: effect
        size +1.83 with severity 'info' and "No statistically significant
        association bias"."""
        rng = np.random.default_rng(0)
        male, female = rng.normal(0, 1, 16), rng.normal(0, 1, 16)
        emb = {}
        for word in ["executive", "salary", "professional", "he", "him", "his"]:
            emb[word] = male + rng.normal(0, 0.15, 16)
        for word in ["home", "children", "wedding", "she", "her", "hers"]:
            emb[word] = female + rng.normal(0, 0.15, 16)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EmbeddingBiasDetector(embeddings=emb).weat(
                ["executive", "salary", "professional"],
                ["home", "children", "wedding"],
                ["he", "him", "his"],
                ["she", "her", "hers"],
            )
        assert result.effect_size > 1.0
        assert result.severity == "could_not_check"
        assert _messages(w, "could NOT have reached")

    def test_control_caliskan_sized_sets_still_grade_normally(self):
        """OVER-CORRECTION CONTROL at the word-set sizes WEAT is defined for
        (Caliskan et al. use 8 to 25 per side): a stereotype must be critical
        and an unbiased embedding must be a measured 'info'."""
        rng = np.random.default_rng(0)
        male, female = rng.normal(0, 1, 16), rng.normal(0, 1, 16)
        career = [f"career{i}" for i in range(8)]
        family = [f"family{i}" for i in range(8)]
        he = [f"he{i}" for i in range(8)]
        she = [f"she{i}" for i in range(8)]
        biased = {w: male + rng.normal(0, 0.2, 16) for w in career + he}
        biased.update({w: female + rng.normal(0, 0.2, 16) for w in family + she})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hit = EmbeddingBiasDetector(embeddings=biased).weat(career, family, he, she)
        assert hit.severity == "critical"
        assert hit.p_value < 0.05
        assert not _messages(w, "could NOT have reached")

        neutral = {w: rng.normal(0, 1, 16) for w in biased}
        miss = EmbeddingBiasDetector(embeddings=neutral).weat(career, family, he, she)
        assert miss.severity == "info"
        assert "No statistically significant association bias" in miss.interpretation


# ---------------------------------------------------------------------------
# 7. EmergentBiasDetector: over-reporting is a defect too
# ---------------------------------------------------------------------------


class TestEmergentAmplificationIsNotASentinel:
    """`amplification_factor = 1e6 if system_bias > 0 else 1.0` put a
    MEASUREMENT on the amplification scale where a ratio is undefined, and the
    pulse report grades severity straight off that field. Measured before the
    fix on 200 samples, components held constant (bias exactly 0) and a 0.001
    system gap: amplificationFactor 1000000.0, isEmergent true, severity
    "critical", one critical BIA finding."""

    @staticmethod
    def _case(gap, noise, n=200, seed=11):
        rng = np.random.default_rng(seed)
        groups = np.array([0] * (n // 2) + [1] * (n // 2))
        components = {"a": np.full(n, 0.5), "b": np.full(n, 0.5)}
        system = (
            0.5
            + np.where(groups == 0, -gap / 2, gap / 2)
            + (rng.normal(0, noise, n) if noise else 0.0)
        )
        return components, system, groups

    def test_an_undefined_ratio_is_nan_not_a_million(self):
        components, system, groups = self._case(0.001, 0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EmergentBiasDetector().analyze(components, system, groups)
        assert math.isnan(result.amplification_factor)
        assert result.amplification_factor != 1e6
        assert _messages(w, "amplification RATIO")

    def test_a_noise_gap_over_a_zero_baseline_is_not_emergent(self):
        components, system, groups = self._case(0.001, 0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EmergentBiasDetector().analyze(components, system, groups)
        assert result.is_emergent is False
        assert result.is_significant is False
        assert result.p_value > 0.05

    def test_the_shipped_pulse_report_no_longer_calls_it_critical(self):
        from vfairness.operations.pulse import run_pulse

        rows = []
        for i in range(200):
            group = "A" if i < 100 else "B"
            rows.append(
                {
                    "group": group,
                    "tool": "approve",
                    "agent_a_score": 0.5,
                    "agent_b_score": 0.5,
                    "system_score": 0.4995 if group == "A" else 0.5005,
                }
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = run_pulse(
                pd.DataFrame(rows),
                {"source_kind": "agent", "protected_attributes": ["group"]},
            )["data"]

        amplification = out["agent"]["amplification"]
        assert amplification["available"] is True, (
            "the comparison WAS made: an undefined ratio must not collapse the "
            "whole section to could-not-check"
        )
        comparison = amplification["perComparison"][0]
        assert comparison["amplificationFactor"] is None
        assert comparison["amplificationDefined"] is False
        assert comparison["severity"] == "info", comparison
        assert not [
            f for f in out["assurance"].get("findings", []) if f.get("severity") == "critical"
        ]

    def test_control_a_real_gap_over_a_zero_baseline_is_still_critical(self):
        """OVER-CORRECTION CONTROL, at the detector and in the report."""
        components, system, groups = self._case(0.9, 0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EmergentBiasDetector().analyze(components, system, groups)
        assert result.is_emergent is True
        assert result.is_significant is True

        from vfairness.operations.pulse import run_pulse

        rng = np.random.default_rng(3)
        rows = []
        for i in range(200):
            group = "A" if i < 100 else "B"
            rows.append(
                {
                    "group": group,
                    "tool": "approve",
                    "agent_a_score": 0.5,
                    "agent_b_score": 0.5,
                    "system_score": (0.2 if group == "A" else 0.8) + float(rng.normal(0, 0.05)),
                }
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = run_pulse(
                pd.DataFrame(rows),
                {"source_kind": "agent", "protected_attributes": ["group"]},
            )["data"]
        comparison = out["agent"]["amplification"]["perComparison"][0]
        assert comparison["isEmergent"] is True
        assert comparison["severity"] == "critical"

    def test_control_a_defined_ratio_is_untouched(self):
        rng = np.random.RandomState(42)
        groups = np.array([0] * 50 + [1] * 50)
        components = {"a": rng.normal(0.5, 0.1, 100), "b": rng.normal(0.5, 0.1, 100)}
        system = np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])
        result = EmergentBiasDetector().analyze(components, system, groups)
        assert result.is_emergent is True
        assert result.amplification_factor > 1.5
        assert result.is_significant is True


# ---------------------------------------------------------------------------
# 8. Pulse agent probe: a rare tool has no power and must not take a slot
# ---------------------------------------------------------------------------


def _tool_frame(rare: int) -> pd.DataFrame:
    """60 episodes per group. Group A escalates 7 times (11.7 percent), group
    B never. `rare` tools are used exactly once each, so their Fisher floor is
    1.0: they cannot produce a finding under any demographics."""
    rows = []
    a = (
        ["approve"] * 30
        + ["search"] * (23 - rare)
        + ["escalate"] * 7
        + [f"rare{i}" for i in range(rare)]
    )
    b = ["approve"] * 30 + ["search"] * 30
    rows.extend({"group": "A", "tool": t} for t in a)
    rows.extend({"group": "B", "tool": t} for t in b)
    return pd.DataFrame(rows)


def _probe(df):
    from vfairness.operations.pulse import run_pulse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})["data"]


def _tool_row(out, name):
    for comparison in out["agent"]["perComparison"]:
        for tool in comparison["tools"]:
            if tool["tool_name"] == name:
                return tool
    raise AssertionError(f"no row for {name}")


class TestRareToolsDoNotSpendTheFamilyBudget:
    """Measured before the fix on a real 11.7-percent versus 0 escalation
    disparity at raw p=0.0130: BH-adjusted 0.0390 and REPORTED with no rare
    tools in the family, 0.0519 and SUPPRESSED with one, 0.0649 with two. The
    finding turned on tools whose own p-floor is 1.0."""

    @pytest.mark.parametrize("rare", [0, 1, 2])
    def test_the_real_disparity_survives_the_long_tail(self, rare):
        out = _probe(_tool_frame(rare))
        escalate = _tool_row(out, "escalate")
        assert escalate["p_value"] == pytest.approx(0.0130, abs=1e-3)
        assert escalate["assessed"] is True
        assert escalate["family_significant"] is True, (
            f"a real disparity was suppressed by {rare} untestable tool(s)"
        )
        assert escalate["p_adjusted"] < 0.05

    def test_a_tool_used_once_is_reported_as_not_assessed(self):
        out = _probe(_tool_frame(2))
        for name in ("rare0", "rare1"):
            row = _tool_row(out, name)
            assert row["assessed"] is False
            assert row["family_significant"] is None, "None, never a clean False"
            assert row["minAttainableP"] == pytest.approx(1.0)
            assert "NOT DETECTABLE" in row["notAssessedReason"]
        probe = out["assurance"]["auditTrail"]["probe"]
        assert probe["toolTestsNotAssessed"] == 2
        assert probe["toolFamilySize"] == 3

    def test_no_significant_bias_detected_names_what_could_not_fire(self):
        """Measured before the fix: this sentence was printed while a deny_loan
        row showed rate_a=0.0667 against rate_b=0.0 at p=0.1187 -- which IS the
        floor for a tool used 4 times across 60 + 60 episodes."""
        rows = [{"group": "A", "tool": t} for t in ["approve"] * 56 + ["deny_loan"] * 4]
        rows += [{"group": "B", "tool": "approve"} for _ in range(60)]
        out = _probe(pd.DataFrame(rows))

        deny = _tool_row(out, "deny_loan")
        assert deny["p_value"] == pytest.approx(0.1187, abs=1e-3)
        assert deny["minAttainableP"] == pytest.approx(0.1187, abs=1e-3)
        assert deny["assessed"] is False
        assert deny["family_significant"] is None
        summary = out["agent"]["summary"]
        assert "No significant tool/action selection bias detected" in summary
        assert "could not fire at any data" in summary
        assert "not as absence of bias" in summary

    def test_control_a_null_trace_set_still_reports_a_clean_no_finding(self):
        """OVER-CORRECTION CONTROL: when every tool is used often enough, the
        family is untouched and nothing is marked not-assessed."""
        rng = np.random.default_rng(7)
        rows = []
        for group in ("A", "B"):
            picks = rng.choice(["search", "escalate", "respond"], size=200, p=[0.6, 0.3, 0.1])
            rows.extend({"group": group, "tool": t} for t in picks)
        out = _probe(pd.DataFrame(rows))
        probe = out["assurance"]["auditTrail"]["probe"]
        assert probe["toolTestsNotAssessed"] == 0
        assert probe["toolFamilySize"] == 3
        for comparison in out["agent"]["perComparison"]:
            for tool in comparison["tools"]:
                assert tool["assessed"] is True
                assert tool["family_significant"] is False
        assert "could not fire" not in out["agent"]["summary"]


# ---------------------------------------------------------------------------
# 9. FairnessExperiment: the untested intersection, and falsy zeros
# ---------------------------------------------------------------------------


def _experiment_frames(seed=3):
    """One intersection with a single observation per arm: Welch's t-test
    returns NaN there, and the effect is -5.0."""
    rng = np.random.default_rng(seed)
    rows = [
        {"arm": "control", "g": "X", "y": 10.0},
        {"arm": "treatment", "g": "X", "y": 5.0},
    ]
    for g in ("Y", "Z"):
        for _ in range(200):
            rows.append({"arm": "control", "g": g, "y": float(rng.normal(0, 1))})
            rows.append({"arm": "treatment", "g": g, "y": float(rng.normal(0.5, 1))})
    df = pd.DataFrame(rows)
    return df[df.arm == "control"], df[df.arm == "treatment"]


class TestUntestedIntersectionIsNotEvidenceOfSafety:
    """CRITICAL. The None-for-unmeasured logic existed ONLY in the
    `correction_method == "none"` branch, and the default is "fdr". Measured
    2026-09-10 on identical data:
      fdr  -> ('X',) effect=-5.0000 p=nan significant=False, and the
              recommendation read "Overall positive effect ... with no harmed
              groups."
      none -> ('X',) effect=-5.0000 p=nan significant=None, and the
              recommendation said the harm COULD NOT BE DETERMINED.
    """

    @pytest.mark.parametrize("method", ["fdr", "bonferroni", "none"])
    def test_every_correction_method_reports_the_same_third_state(self, method):
        control, treatment = _experiment_frames()
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(
                correction_method=method, min_group_size=1, n_bootstrap=200, random_state=1
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = experiment.detect_heterogeneous_effects()

        by_ix = {e.intersection: e for e in result.intersection_effects}
        untested = by_ix[("X",)]
        assert untested.effect == pytest.approx(-5.0)
        assert math.isnan(untested.p_value)
        assert untested.significant is None, (
            f"{method}: a hypothesis that was never tested is not a hypothesis that passed"
        )
        assert by_ix[("Y",)].significant is True
        assert by_ix[("Z",)].significant is True

    def test_the_recommendation_no_longer_claims_no_harmed_groups(self):
        from vfairness.operations.experimentation.analysis import ExperimentAnalysis

        control, treatment = _experiment_frames()
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(min_group_size=1, n_bootstrap=200, random_state=1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = experiment.detect_heterogeneous_effects()
        recommendation = ExperimentAnalysis(result).decision_recommendation()
        joined = " ".join(recommendation.reasoning)
        assert "no harmed groups" not in joined
        assert "COULD NOT BE DETERMINED" in joined

    def test_control_a_well_powered_experiment_still_deploys(self):
        """OVER-CORRECTION CONTROL: every intersection tested, all positive,
        adequately powered -> DEPLOY at full confidence, and "no harmed
        groups" is allowed to be said because it is true."""
        from vfairness.operations.experimentation.analysis import ExperimentAnalysis

        rng = np.random.default_rng(4)
        rows = []
        for g in ("Y", "Z"):
            for _ in range(450):
                rows.append({"arm": "control", "g": g, "y": float(rng.normal(0, 1))})
                rows.append({"arm": "treatment", "g": g, "y": float(rng.normal(0.5, 1))})
        df = pd.DataFrame(rows)
        experiment = FairnessExperiment(
            df[df.arm == "control"],
            df[df.arm == "treatment"],
            ["g"],
            "y",
            config=ExperimentConfig(n_bootstrap=300, random_state=1),
        )
        result = experiment.detect_heterogeneous_effects()
        assert all(e.significant is True for e in result.intersection_effects)
        assert all(e.powered is True for e in result.intersection_effects)
        recommendation = ExperimentAnalysis(result).decision_recommendation()
        assert recommendation.decision.value == "deploy_treatment"
        assert recommendation.confidence == pytest.approx(0.85)
        assert "no harmed groups" in " ".join(recommendation.reasoning)


class TestSuppliedZeroIsNotAMissingArgument:
    """`n_bootstrap or self.config.n_bootstrap` and `alpha or self.config.alpha`
    treat a legal zero as "not supplied". Measured before the fix:
    detect_heterogeneous_effects(alpha=0.0) ran at 0.05 and recorded
    metadata['alpha'] = 0.05, so the provenance record agreed with the
    substitution rather than with the request."""

    def _experiment(self):
        rng = np.random.default_rng(3)
        rows = []
        for g in ("Y", "Z"):
            for _ in range(60):
                rows.append({"arm": "control", "g": g, "y": float(rng.normal(0, 1))})
                rows.append({"arm": "treatment", "g": g, "y": float(rng.normal(0.5, 1))})
        df = pd.DataFrame(rows)
        return FairnessExperiment(
            df[df.arm == "control"],
            df[df.arm == "treatment"],
            ["g"],
            "y",
            config=ExperimentConfig(min_group_size=10, n_bootstrap=200, random_state=1),
        )

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"n_bootstrap": 0}])
    def test_an_impossible_request_is_refused_not_replaced(self, kwargs):
        with pytest.raises(ConfigurationError):
            self._experiment().detect_heterogeneous_effects(**kwargs)

    def test_control_a_legal_override_is_honoured_and_recorded(self):
        result = self._experiment().detect_heterogeneous_effects(alpha=0.10, n_bootstrap=150)
        assert result.metadata["alpha"] == 0.10

    def test_an_effect_object_does_not_assert_power_it_never_had(self):
        """`powered: bool = True` meant any IntersectionEffect built without a
        power calculation claimed adequate power."""
        effect = IntersectionEffect(
            intersection=("Q",),
            control_mean=0.0,
            treatment_mean=0.0,
            effect=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=float("nan"),
            effect_size_d=0.0,
            n_control=0,
            n_treatment=0,
        )
        assert effect.powered is None


# ---------------------------------------------------------------------------
# 10. FairnessPowerAnalyzer: SPRT boundaries and the power sentinel
# ---------------------------------------------------------------------------


def _power_experiment(n_per_arm: int, effect: float, seed: int = 5):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_per_arm):
        rows.append({"arm": "control", "g": "A", "y": float(rng.normal(0, 1))})
        rows.append({"arm": "treatment", "g": "A", "y": float(rng.normal(effect, 1))})
    df = pd.DataFrame(rows)
    return FairnessExperiment(
        df[df.arm == "control"],
        df[df.arm == "treatment"],
        ["g"],
        "y",
        config=ExperimentConfig(min_group_size=10),
    )


class TestSprtBoundariesAreMonotoneInAlpha:
    """Measured before the fix on a 2.1-sigma effect over 60 per arm:
    alpha=0.0 gave upper_boundary=10 and so decision=reject_null with
    stopped_early=True, while alpha=1e-12 -- a strictly weaker demand -- gave
    27.41 and correctly refused to stop. A request for zero type-I error was
    answered by the LOWEST bar on the scale."""

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, "0.05"])
    def test_an_impossible_alpha_is_refused(self, alpha):
        with pytest.raises(ConfigurationError):
            PowerConfig(alpha=alpha)

    @pytest.mark.parametrize("target_power", [0.0, 1.0])
    def test_an_impossible_target_power_is_refused(self, target_power):
        with pytest.raises(ConfigurationError):
            PowerConfig(target_power=target_power)

    def test_control_a_stricter_alpha_always_raises_the_bar(self):
        experiment = _power_experiment(60, 2.1)
        boundaries = []
        for alpha in (1e-12, 1e-6, 0.001, 0.05):
            analyzer = FairnessPowerAnalyzer(
                experiment, PowerConfig(alpha=alpha, min_group_size=10)
            )
            result = analyzer.sequential_test(effect_size=0.2)[("A",)]
            boundaries.append(result.upper_boundary)
        assert boundaries == sorted(boundaries, reverse=True), boundaries
        # and the SPRT still stops on a large real effect at a normal alpha
        analyzer = FairnessPowerAnalyzer(experiment, PowerConfig(alpha=0.05, min_group_size=10))
        result = analyzer.sequential_test(effect_size=0.2)[("A",)]
        assert result.decision.value == "reject_null"
        assert result.stopped_early is True


class TestPowerNotComputedIsNotPowerZero:
    """Measured before the fix at 28 per arm against a min_group_size floor of
    30 and a requirement of 25: power 0.0, mde inf, is_powered False, and the
    sampling plan's rationale "already powered" with n_underpowered=0, all in
    the same run and all silent."""

    def _analyzer(self):
        experiment = _power_experiment(28, 0.05)
        return FairnessPowerAnalyzer(experiment, PowerConfig(min_group_size=30))

    def test_the_sentinels_are_not_measurements_any_more(self):
        analyzer = self._analyzer()
        assert math.isnan(analyzer.power_for_sample_size(0.8)[("A",)])
        assert math.isnan(analyzer.minimum_detectable_effect()[("A",)])

    def test_the_plan_does_not_call_it_already_powered(self):
        analyzer = self._analyzer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=1000, effect_size=0.8)

        assert plan.not_assessed == [("A",)]
        assert plan.rationale[("A",)].startswith("NOT ASSESSED")
        assert "already powered" not in plan.rationale.values()
        assert plan.to_dict()["n_not_assessed"] == 1
        assert _messages(w, "power was not computed for 1 intersection(s)")

    def test_the_summary_and_detail_carry_the_third_state(self):
        analyzer = self._analyzer()
        row = analyzer.get_power_summary(0.8).iloc[0]
        assert math.isnan(row["power"])
        assert row["is_powered"] is None
        detail = analyzer.get_detailed_results(0.8)[0]
        assert detail.is_powered is None
        assert math.isnan(detail.power)

    def test_control_a_powered_experiment_still_reads_already_powered(self):
        """OVER-CORRECTION CONTROL."""
        experiment = _power_experiment(2000, 0.05)
        analyzer = FairnessPowerAnalyzer(experiment, PowerConfig(min_group_size=30))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=100, effect_size=0.2)
        assert plan.n_underpowered == 0
        assert plan.not_assessed == []
        assert set(plan.rationale.values()) == {"already powered"}
        assert not _messages(w, "not computed")
        detail = analyzer.get_detailed_results(0.2)[0]
        assert detail.is_powered is True


# ---------------------------------------------------------------------------
# 11. CounterfactualTester: a test that RETURNED is not a test that RAN
# ---------------------------------------------------------------------------


class _EchoProxy:
    """Every variant gets the same words, so length / toxicity / refusal are
    tied and sentiment is the only channel that could separate the arms."""

    def send_prompt(self, prompt, system_prompt=None, temperature=0.0, **kwargs):
        return {"text": "the applicant is qualified for the role"}


class _LengthProxy:
    """A real, large disparity: one demographic gets ten times the words."""

    def send_prompt(self, prompt, system_prompt=None, temperature=0.0, **kwargs):
        return {"text": " ".join(["word"] * (40 if "James" in prompt else 4))}


class _DeadScorer:
    """A scorer backend that is down: NaN for every text. This is not
    hypothetical -- KeywordSentimentScorer returns NaN by design rather than
    fabricate a neutral 0.0 for text it cannot read."""

    def score(self, text):
        return float("nan")

    def score_batch(self, texts):
        return np.full(len(texts), np.nan)


class _FlatScorer:
    def score(self, text):
        return 0.0

    def score_batch(self, texts):
        return np.zeros(len(texts))


class TestCounterfactualUntestedMetricIsNotANegative:
    """`any_tested = True` was set the moment mannwhitneyu RETURNED, whatever
    it returned. Measured 2026-09-10 with a dead sentiment scorer and every
    other metric tied: is_significant=False alongside `sentiment_delta: nan`
    and `data_quality: 'good'` in the same dict, and no warning."""

    def _tester(self, proxy, sentiment):
        from vfairness.llm.counterfactual import CounterfactualTester

        return CounterfactualTester(
            proxy,
            n_runs=6,
            sentiment_scorer=sentiment,
            toxicity_scorer=_FlatScorer(),
            refusal_scorer=_FlatScorer(),
        )

    def test_a_dead_scorer_is_named_not_averaged_into_a_clean_dict(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._tester(_EchoProxy(), _DeadScorer()).run_test(
                "Write a letter for {name}.", {"name": ["James", "Jamal"]}
            )

        metrics = result.disparity_metrics
        assert metrics["sentiment_delta"] is None, "None, never nan-inside-a-good-dict"
        assert metrics["data_quality"] == "scores_unavailable"
        assert metrics["unscored_metrics"] == ["sentiment_delta"]
        params = result.metadata.parameters
        assert params["n_tests_not_run"] == 1
        assert params["n_tests_run"] == 3
        sentiment = [t for t in params["significance_tests"] if t["metric"] == "sentiment"][0]
        assert sentiment["tested"] is False
        assert "no test could run" in sentiment["reason"]
        assert _messages(w, "did NOT run")
        assert _messages(w, "scores_unavailable")

    def test_the_four_metrics_are_corrected_as_one_family(self):
        """There was no correction: any_significant fired on the first metric
        under alpha and `break`-ed, which is four uncorrected looks at the same
        pair of response sets."""
        result = self._tester(_LengthProxy(), _FlatScorer()).run_test(
            "Write a letter for {name}.", {"name": ["James", "Jamal"]}
        )
        params = result.metadata.parameters
        assert params["correction"] == "benjamini_hochberg"
        length = [t for t in params["significance_tests"] if t["metric"] == "length"][0]
        assert length["p_adjusted"] > length["p_value"], "an uncorrected p is not a family p"
        assert result.is_significant is True, (
            "OVER-CORRECTION CONTROL: a real disparity must survive the correction"
        )

    def test_control_identical_arms_are_a_measured_false(self):
        result = self._tester(_EchoProxy(), _FlatScorer()).run_test(
            "Write a letter for {name}.", {"name": ["James", "Jamal"]}
        )
        assert result.is_significant is False
        assert result.metadata.parameters["n_tests_run"] == 4
        assert result.metadata.parameters["n_tests_not_run"] == 0
        assert result.disparity_metrics["data_quality"] == "good"

    def test_two_responses_a_side_cannot_reach_alpha(self):
        """DISCRETE FLOOR: the rank-sum on 2 against 2 cannot reach 0.05, so the
        one metric carrying a 10x length difference is excluded from the family
        and named, rather than counted as a test that found nothing."""
        tester = self._tester(_LengthProxy(), _FlatScorer())
        tester._n_runs = 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = tester.run_test("Write a letter for {name}.", {"name": ["James", "Jamal"]})

        length = [
            t for t in out.metadata.parameters["significance_tests"] if t["metric"] == "length"
        ][0]
        assert length["tested"] is False
        assert "could not reach 0.05" in length["reason"]
        assert "NOT DETECTABLE" in length["reason"]
        assert "p_adjusted" not in length, "an untestable metric takes no family slot"
        assert _messages(w, "did NOT run")
        # THE OTHER THREE METRICS FACE THE SAME FLOOR, and until 2026-09-27 this test
        # asserted otherwise: n_tests_not_run == 1, n_tests_run == 3, and
        # is_significant is False, under the comment "The three metrics that ARE tied
        # were genuinely measured as identical, so False is a finding about them".
        #
        # That reasoning is the defect. The floor at 2 against 2 comes from the sample
        # SIZES, not from the observed values: the rank-sum cannot go below 0.194 for
        # any data of that shape, and being tied does not exempt a metric from it. The
        # three escaped only because the identical-distributions shortcut returned
        # before the floor was computed, so the design got a verdict exactly when there
        # was nothing to find and was refused whenever there was something. Measured
        # now, all four report "could not reach 0.05", three of them adding "the values
        # were identical across 2 and 2 responses".
        #
        # The subject of this test is unchanged and still asserted above: the metric
        # carrying a tenfold difference is excluded from the family and named. What
        # changes is the claim about its three neighbours.
        assert out.metadata.parameters["n_tests_not_run"] == 4
        assert out.metadata.parameters["n_tests_run"] == 0
        assert out.is_significant is None, (
            "no metric could be tested at this sample size, so the overall verdict is "
            "unknown; False would say the comparison ran and found nothing"
        )
