"""
Adversarial collusion detection in multi-agent systems.

Detects bias that is *amplified by agent interaction itself* rather than
already present in individual agents at the outset. Distinguishes this
"coordination dynamics" failure mode from general convergence
(GroupthinkDetector) and from compositional amplification
(EmergentBiasDetector).

Methodology follows the multi-agent debate / deliberation literature:
- Khan et al. (2023) "Debating with More Persuasive LLMs Leads to More
  Truthful Answers"
- Du et al. (2023) "Improving Factuality and Reasoning in Language
  Models through Multiagent Debate"
- Bianchi et al. (2024) "Cooperation, Competition, and Maliciousness:
  LLM-Stakeholders Interactive Negotiation"

Each agent's group-conditional output is measured before any
inter-agent interaction and again after; collusion is detected when the
post-interaction bias systematically exceeds the pre-interaction bias.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_permutation,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class CollusionResult(SerializableMixin):
    """Result of an adversarial-collusion analysis.

    Attributes:
        pre_bias_per_agent: Per-agent group-conditional bias measured
            before inter-agent interaction.
        post_bias_per_agent: Same, measured after the interaction round.
        pre_mean_bias: Mean of pre-interaction biases across agents.
        post_mean_bias: Mean of post-interaction biases across agents.
        collusion_score: post_mean_bias - pre_mean_bias. Positive values
            indicate the interaction amplified bias.
        is_collusion: Whether the amplification is both positive and
            statistically significant under the permutation null. None means
            no test was run, NOT that no collusion was found.
        p_value: Permutation-test p-value (one-sided: post > pre), or NaN when
            the test could not be run.
        is_significant: Convenience alias for `p_value < alpha`; None when no
            test was run, or when the permutation null on this many samples
            could never have reached alpha.
        n_groups_compared: How many demographic groups the bias statistic
            spans. Every group is compared (max mean minus min mean), so this
            equals the number of distinct labels in ``groups``.
        min_attainable_p: Smallest p the permutation test could report for
            this data, whatever the outputs had been. None when not computable.
        detectable: Whether ``min_attainable_p`` clears alpha. None is
            could-not-check.
        detectability_note: What a "no collusion" reading does NOT mean when
            ``detectable`` is not True. Empty when the design has power.
    """

    pre_bias_per_agent: Dict[str, float]
    post_bias_per_agent: Dict[str, float]
    pre_mean_bias: float
    post_mean_bias: float
    collusion_score: float
    is_collusion: Optional[bool]
    p_value: float = 1.0
    is_significant: Optional[bool] = False
    metadata: RunMetadata = field(default_factory=RunMetadata)
    n_groups_compared: int = 2
    min_attainable_p: Optional[float] = None
    detectable: Optional[bool] = None
    detectability_note: str = ""


class AdversarialCollusionDetector:
    """Detects bias amplification arising from agent-to-agent interaction.

    The detector compares each agent's group-conditional output disparity
    before any inter-agent communication takes place against the same
    disparity after one or more rounds of interaction. A systematic
    increase in disparity post-interaction is the signature of
    coordination-induced bias (adversarial collusion).

    Args:
        alpha: Significance threshold for the permutation test.
        n_permutations: Number of permutations for the null distribution.
        random_seed: Seed for reproducible permutations.

    Example:
        >>> detector = AdversarialCollusionDetector()
        >>> groups = np.array([0]*40 + [1]*40)
        >>> pre = {"agent_a": np.random.rand(80), "agent_b": np.random.rand(80)}
        >>> post = {"agent_a": np.random.rand(80), "agent_b": np.random.rand(80)}
        >>> result = detector.analyze(pre, post, groups)
        >>> result.is_collusion

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: adversarial_collusion_detector. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 500,
        random_seed: int = 42,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_permutations < 50:
            raise ValueError(
                f"n_permutations must be >= 50 for a meaningful null distribution, "
                f"got {n_permutations}."
            )
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.random_seed = random_seed

    def _bias(self, outputs: np.ndarray, masks: Sequence[np.ndarray]) -> float:
        """Group-conditional bias: the widest gap between any two group means.

        Was ``abs(mean[group_0] - mean[group_1])`` over the FIRST TWO labels
        only, while ``metadata['n_samples']`` reported the full input length.
        Measured 2026-09-10 on 90 samples in three groups whose post-interaction
        means were 0.500 / 0.500 / 0.947: is_collusion=False at p=0.9421, with
        the 30 samples carrying the entire disparity never examined and nothing
        in the result saying so. Max-minus-min is identical to the old
        expression when there are exactly two groups, so the binary contract in
        the docstring is unchanged and the third group stops being invisible.
        """
        means = [float(np.mean(outputs[m])) for m in masks]
        return float(max(means) - min(means))

    def _design_floor(self, masks: Sequence[np.ndarray], n_samples: int) -> Optional[float]:
        """Smallest p this permutation design could return, at these group sizes.

        Asked of the DESIGN, not of the observed outputs: it runs the identical
        flip-permutation on a surrogate in which the interaction produced the
        strongest signal the statistic can express (no pre-interaction bias, a
        perfectly separated post-interaction bias of 1.0). If even that cannot
        clear alpha, nothing this run could have measured would have.

        Reading the floor off the OBSERVED draws instead would conflate two
        different things: at 8 samples a maximally colluding system floors at
        0.0625 because the null has 2**8 assignments (a design limit), but a
        run where pre and post are literally identical also has every draw tied
        with the observation, and that is a measured absence of amplification,
        not an absence of power. The surrogate separates them.
        """
        if n_samples <= 0 or len(masks) < 2:
            return None
        pre = np.zeros(n_samples, dtype=float)
        post = np.zeros(n_samples, dtype=float)
        post[np.asarray(masks[-1], dtype=bool)] = 1.0
        observed = self._bias(post, masks) - self._bias(pre, masks)
        rng = np.random.default_rng(self.random_seed)
        draws = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            flip = rng.random(n_samples) < 0.5
            null_pre = np.where(flip, post, pre)
            null_post = np.where(flip, pre, post)
            draws[i] = self._bias(null_post, masks) - self._bias(null_pre, masks)
        floor = (1.0 + int(np.sum(draws >= observed - 1e-12))) / (1.0 + self.n_permutations)
        resample_floor = min_attainable_p_permutation(self.n_permutations)
        if resample_floor is not None:
            floor = max(floor, resample_floor)
        return float(floor)

    def analyze(
        self,
        pre_interaction_outputs: Dict[str, np.ndarray],
        post_interaction_outputs: Dict[str, np.ndarray],
        groups: np.ndarray,
    ) -> CollusionResult:
        """Run the collusion test.

        Args:
            pre_interaction_outputs: Per-agent outputs before any
                inter-agent communication. Same agent set and same sample
                ordering as ``post_interaction_outputs``.
            post_interaction_outputs: Per-agent outputs after the
                interaction round(s) of interest.
            groups: Binary group membership array (0 or 1) aligned with
                the per-agent output arrays.

        Returns:
            CollusionResult.

        Raises:
            ValueError: If agent sets differ between pre/post, if any
                array length is inconsistent with ``groups``, or if
                ``groups`` has fewer than 2 unique values.
        """
        logger.info(
            "analyze: %d agents, %d samples",
            len(pre_interaction_outputs),
            len(groups),
        )

        if set(pre_interaction_outputs) != set(post_interaction_outputs):
            raise ValueError(
                "pre_interaction_outputs and post_interaction_outputs must cover the same agents."
            )
        if not pre_interaction_outputs:
            raise ValueError("At least one agent's outputs are required.")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Groups array must contain at least 2 unique values.")

        n_samples = len(groups)
        # EVERY group, not the first two: see _bias above.
        masks: List[np.ndarray] = [groups == g for g in unique_groups]

        pre_bias: Dict[str, float] = {}
        post_bias: Dict[str, float] = {}
        for name in pre_interaction_outputs:
            pre_arr = np.asarray(pre_interaction_outputs[name], dtype=float)
            post_arr = np.asarray(post_interaction_outputs[name], dtype=float)
            if len(pre_arr) != n_samples or len(post_arr) != n_samples:
                raise ValueError(
                    f"Agent '{name}' output length must match groups length ({n_samples})."
                )
            pre_bias[name] = self._bias(pre_arr, masks)
            post_bias[name] = self._bias(post_arr, masks)

        pre_mean = float(np.mean(list(pre_bias.values())))
        post_mean = float(np.mean(list(post_bias.values())))
        collusion_score = post_mean - pre_mean

        # Paired permutation test at the SAMPLE level: under the null
        # (no collusion) pre and post values are exchangeable per
        # sample. We flip pre <-> post for a random subset of samples
        # (the same subset across all agents to preserve the paired
        # structure) and recompute the test statistic. This has high
        # power even with only a few agents.
        rng = np.random.default_rng(self.random_seed)
        agents = list(pre_interaction_outputs.keys())
        pre_arrays = {a: np.asarray(pre_interaction_outputs[a], dtype=float) for a in agents}
        post_arrays = {a: np.asarray(post_interaction_outputs[a], dtype=float) for a in agents}

        null_stats = []
        for _ in range(self.n_permutations):
            flip = rng.random(n_samples) < 0.5
            null_pre_means = []
            null_post_means = []
            for a in agents:
                # Build per-sample-permuted arrays.
                null_pre = np.where(flip, post_arrays[a], pre_arrays[a])
                null_post = np.where(flip, pre_arrays[a], post_arrays[a])
                null_pre_means.append(self._bias(null_pre, masks))
                null_post_means.append(self._bias(null_post, masks))
            null_stats.append(float(np.mean(null_post_means) - np.mean(null_pre_means)))

        null_arr = np.asarray(null_stats)
        # One-sided permutation p-value with the standard (1 + k) / (1 + n)
        # correction so a finite permutation count can never report an
        # impossible-to-defend p = 0 exactly (Phipson & Smyth 2010).
        # `nan >= x` is False for EVERY draw, so an unmeasurable observed
        # statistic scored k=0 and collapsed to the smallest p this correction
        # can produce. Measured 2026-09-08 with NaN post-interaction outputs:
        # p_value=0.0196 and is_significant=True, byte-identical to the genuine
        # collusion control. Only is_collusion stayed False, and by accident
        # (`nan > 0` is False), not by design; any consumer reading p_value or
        # is_significant saw a maximally significant result from data nobody
        # measured.
        p_value: float
        is_significant: Optional[bool]
        is_collusion: Optional[bool]
        min_attainable: Optional[float] = None
        detectable: Optional[bool] = None
        note = ""
        if not np.isfinite(collusion_score) or not np.all(np.isfinite(null_arr)):
            warnings.warn(
                "AdversarialCollusionDetector.analyze: the observed statistic or part of "
                "the permutation null is not finite, so no permutation test was run. "
                "Reporting p_value=nan, is_significant=None and is_collusion=None (could "
                "not check), not a significant result.",
                UserWarning,
                stacklevel=2,
            )
            p_value = float("nan")
            is_significant = None
            is_collusion = None
            note = (
                "COULD NOT CHECK: the observed statistic or part of the permutation "
                "null was not finite, so no test was run."
            )
        else:
            k = int(np.sum(null_arr >= collusion_score))
            p_value = (1.0 + k) / (1.0 + self.n_permutations)

            # DISCRETE FLOOR (readiness 6, 2026-09-10). This null flips pre and
            # post per SAMPLE, so it holds at most 2**n_samples distinct
            # assignments however many resamples are drawn, and the observed
            # statistic is itself one of them: it can never beat the largest the
            # null can produce. Measured 2026-09-10 with every pre-interaction
            # output identical and the post-interaction outputs perfectly
            # separated by group (total collusion) against 500 resamples:
            # p = 0.2675 at 4 samples, 0.1138 at 6 and 0.0559 at 8, all reported
            # as is_collusion=False. Exhaustive enumeration of the 2**n
            # assignments confirms the smallest attainable p there is 0.125 at
            # n=6 and 0.0625 at n=8: no data could have made those runs
            # significant.
            #
            # TWO floors, and the design is called undetectable only when BOTH
            # say so, which is the fail-safe direction this library's shared
            # helper states in its own comment block (never suppress a finding
            # a favourable arrangement of the data could have produced):
            #   * the DESIGN floor, from a surrogate carrying the strongest
            #     signal this statistic can express at these group sizes;
            #   * the DATA floor, the share of THIS run's draws that tie the
            #     most extreme value the statistic can take on these values.
            # Neither alone is right. The design floor alone declared a run that
            # actually reported p=0.018 undetectable, because its surrogate has
            # pre=0 and so ties on every flip inside the unbiased group. The
            # data floor alone declared a run whose pre and post outputs are
            # IDENTICAL undetectable, because every draw ties the observation,
            # but that is a measured absence of amplification, not an absence of
            # power.
            ceiling = max(float(np.max(null_arr)), float(collusion_score))
            data_floor = (1.0 + int(np.sum(null_arr >= ceiling))) / (1.0 + self.n_permutations)
            resample_floor = min_attainable_p_permutation(self.n_permutations)
            if resample_floor is not None:
                data_floor = max(data_floor, resample_floor)
            design_floor = self._design_floor(masks, n_samples)
            min_attainable = data_floor if design_floor is None else min(data_floor, design_floor)
            detectable, note = detectability(min_attainable, n_family=1, alpha=self.alpha)
            if detectable is True:
                is_significant = bool(p_value < self.alpha)
                is_collusion = bool(collusion_score > 0 and is_significant)
            else:
                warnings.warn(
                    f"AdversarialCollusionDetector.analyze: the permutation test on "
                    f"{n_samples} sample(s) was NOT ASSESSED. {note} Reporting "
                    f"is_collusion=None and is_significant=None (could not check), NOT "
                    f"False. The measured amplification still stands: "
                    f"collusion_score={collusion_score:.4f}.",
                    UserWarning,
                    stacklevel=2,
                )
                is_significant = None
                is_collusion = None

        logger.info(
            "analyze complete: collusion_score=%.4f, p_value=%.4f, is_collusion=%s",
            collusion_score,
            p_value,
            is_collusion,
        )
        return CollusionResult(
            pre_bias_per_agent=pre_bias,
            post_bias_per_agent=post_bias,
            pre_mean_bias=pre_mean,
            post_mean_bias=post_mean,
            collusion_score=collusion_score,
            is_collusion=is_collusion,
            p_value=p_value,
            is_significant=is_significant,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "n_permutations": self.n_permutations,
                    "n_agents": len(agents),
                    "n_samples": n_samples,
                    "n_groups_compared": len(unique_groups),
                    "min_attainable_p": min_attainable,
                    "detectable": detectable,
                },
                random_seed=self.random_seed,
            ),
            n_groups_compared=len(unique_groups),
            min_attainable_p=min_attainable,
            detectable=detectable,
            detectability_note=note,
        )
