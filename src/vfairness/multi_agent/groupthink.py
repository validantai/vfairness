"""
Groupthink and echo-chamber detection in multi-agent systems.

Detects whether agents in a multi-agent system converge to similar
outputs over rounds of interaction, forming echo chambers or
exhibiting groupthink behavior that may amplify biases.
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import detectability
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)

#: Trend-test threshold this detector calls significant, and the bar the
#: design must be able to clear for a verdict to mean anything.
TREND_ALPHA = 0.1
#: Minimum Kendall's tau treated as substantive convergence.
TREND_TAU_THRESHOLD = 0.3
#: Threshold for the permutation test on the final round's convergence.
PERMUTATION_ALPHA = 0.05


def _min_attainable_kendall_p(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Smallest p Kendall's tau can return for THIS convergence series.

    Computed by running the real test on the most extreme arrangement of the
    same values: sorted ascending against the round index. Doing it on the
    observed values rather than on n alone keeps the tie structure and so the
    method scipy resolves to, which is what decides the floor (see the
    fail-safe note on ``min_attainable_p_mannwhitney`` in
    ``evaluation/vfairness_metrics/_statistics.py``). Returns ``None`` when the
    floor is not computable, which is could-not-check and never a number.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p = stats.kendalltau(np.asarray(x, dtype=float), np.sort(np.asarray(y, dtype=float)))
        p = float(p)
    except Exception:  # pragma: no cover - defensive
        return None
    return p if math.isfinite(p) else None


@dataclass
class GroupthinkResult(SerializableMixin):
    """Result of a groupthink analysis.

    Attributes:
        convergence_trajectory: Per-round convergence scores (0=diverse, 1=identical).
        has_groupthink: Whether significant convergence was detected. None
            means convergence could not be measured OR the trend test could
            never have fired on a series this short, NOT that none was found.
        coalition_structure: List of agent coalitions (sets of agent names).
        echo_chamber_score: Overall echo chamber intensity in [0, 1].
        trend_detectable: Whether the Mann-Kendall trend test on this series
            could reach ``TREND_ALPHA`` for ANY data. False means the rounds
            are too few for the verdict to carry information; None means that
            question itself could not be answered.
        trend_note: Plain-language statement of what a "no groupthink" reading
            does NOT mean when ``trend_detectable`` is not True. Empty when the
            design has power.
        convergence_detectable: The same question for the permutation test on
            the final round's convergence.
        convergence_note: Its note.
        unplaced_agents: Agents left OUT of ``coalition_structure`` because
            not one of their pairwise agreements in the final round was
            measurable. They are could-not-check, never independents.
        coalitions_detectable: True when every pairwise agreement in the final
            round was measurable, so ``coalition_structure`` rests on a
            complete graph. False when at least one pair was undefined, so the
            grouping is incomplete. None when no pair was measurable at all
            and ``coalition_structure`` is empty.
        coalition_note: Plain-language statement of what ``coalition_structure``
            does NOT mean when ``coalitions_detectable`` is not True. Empty
            when the agreement graph is complete.
        n_permutations_measured: Permutation draws whose convergence came out
            finite. These are the only draws the p-value rests on.
        n_permutations_unmeasurable: Permutation draws whose convergence could
            not be computed at all. They are EXCLUDED from the p-value rather
            than counted as draws that failed to reach the observed value,
            which is what a nan draw silently became.
    """

    convergence_trajectory: list
    has_groupthink: Optional[bool]
    coalition_structure: list
    echo_chamber_score: float
    p_value: float = 1.0
    is_significant: Optional[bool] = False
    metadata: RunMetadata = field(default_factory=RunMetadata)
    trend_detectable: Optional[bool] = None
    trend_note: str = ""
    convergence_detectable: Optional[bool] = None
    convergence_note: str = ""
    unplaced_agents: list = field(default_factory=list)
    coalitions_detectable: Optional[bool] = None
    coalition_note: str = ""
    n_permutations_measured: int = 0
    n_permutations_unmeasurable: int = 0


class GroupthinkDetector:
    """Detects groupthink and echo-chamber dynamics in multi-agent systems.

    Analyzes agent outputs over multiple rounds of interaction to detect
    convergence patterns that indicate groupthink or echo-chamber formation.

    Example:
        >>> detector = GroupthinkDetector()
        >>> outputs = [
        ...     {"agent_a": [0.8, 0.2], "agent_b": [0.3, 0.7]},  # round 0
        ...     {"agent_a": [0.6, 0.4], "agent_b": [0.5, 0.5]},  # round 1
        ...     {"agent_a": [0.55, 0.45], "agent_b": [0.54, 0.46]},  # round 2
        ... ]
        >>> result = detector.analyze_convergence(outputs)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: groupthink_detector. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def analyze_convergence(
        self,
        agent_outputs_per_round: list[dict[str, list]],
    ) -> GroupthinkResult:
        """Analyze convergence of agent outputs over interaction rounds.

        Measures how similar agent outputs become over successive rounds.
        Uses average pairwise cosine similarity as the convergence metric.

        Args:
            agent_outputs_per_round: List of dictionaries, one per round.
                Each dict maps agent names to their output vectors (lists
                of floats). All agents must appear in every round.

        Returns:
            GroupthinkResult with convergence trajectory and detection.

        Raises:
            ValueError: If fewer than 2 rounds are provided, or if the rounds do
                not all carry the SAME agents (see the note on the guard below:
                the permutation null is built from the first round's agents, so a
                differing set silently tests one statistic against another's null
                distribution).
        """
        logger.info("analyze_convergence: %d rounds", len(agent_outputs_per_round))
        if len(agent_outputs_per_round) < 2:
            raise ValueError("At least 2 rounds are required.")

        # BGL-W4 (2026-09-30). "All agents must appear in every round" was stated
        # in the Args above and enforced nowhere, and the two directions of
        # breaking it behaved completely differently.
        #
        # An agent MISSING from a later round raised a bare `KeyError: 'c'` out of
        # the permutation loop eighty lines down, naming neither the round nor the
        # requirement.
        #
        # An agent APPEARING only after the first round was silently accepted and
        # produced a maximal false finding, because `all_agents` below is taken
        # from `agent_outputs_per_round[0]` alone: the OBSERVED convergence and the
        # agreement matrix are computed over every agent in their own round, while
        # every permutation DRAW is computed over the first round's agents only. The
        # observed statistic is then compared against the null distribution of a
        # DIFFERENT statistic. Measured that day on four rounds of a, b where c is
        # absent from round 0 only: has_groupthink True, echo 0.9992 over three
        # pairs, is_significant True, convergence_detectable True, p_value
        # 0.001996007984031936, which is 1/(500 + 1), the SMALLEST p this estimator
        # can produce, reached because not one two-agent draw could match a
        # three-agent observation. metadata reported n_agents=2 while
        # coalition_structure reported [['a', 'b', 'c']], and there were ZERO
        # warnings.
        #
        # Refused rather than repaired, at the entry point, because the method has
        # no way to choose which agent set the trajectory belongs to and no field in
        # which to say it chose. MultiAgentRunHarness.record_sample refuses an
        # inconsistent agent set at its own recording boundary for the same reason.
        first_agents = set(agent_outputs_per_round[0])
        for index, round_data in enumerate(agent_outputs_per_round[1:], start=1):
            round_agents = set(round_data)
            if round_agents != first_agents:
                missing = sorted(first_agents - round_agents)
                extra = sorted(round_agents - first_agents)
                raise ValueError(
                    f"Every round must carry the same agents: round 0 has "
                    f"{sorted(first_agents)} but round {index} has "
                    f"{sorted(round_agents)} (missing {missing}, extra {extra}). The "
                    f"permutation null is built from round 0's agents, so an agent "
                    f"present in only some rounds would be measured in the observed "
                    f"convergence and absent from every permutation draw, which made "
                    f"the p-value collapse to its floor and report significance. Pass "
                    f"one agent set for the whole interaction, using a round's own "
                    f"recorded output for an agent that stayed silent, or analyse the "
                    f"rounds that share an agent set and say which rounds each verdict "
                    f"covers."
                )

        convergence_trajectory = []
        for round_data in agent_outputs_per_round:
            score = self._compute_round_convergence(round_data)
            convergence_trajectory.append(float(score))

        # Detect convergence trend using Kendall's tau.
        rounds = np.arange(len(convergence_trajectory))
        values = np.array(convergence_trajectory)

        # DISCRETE FLOOR (readiness 6, 2026-09-10). `tau > 0.3 and p_value <
        # 0.1` is a verdict on a test whose p-value has a FLOOR set by the
        # number of rounds: Kendall's tau on 3 rounds cannot return anything
        # below 2/3! = 0.3333, whatever the agents did. Measured 2026-09-10 on
        # this class's own docstring example shape, two agents going from
        # perfectly opposite to literally identical over 3 rounds
        # ([0.0, 0.724, 1.0]): tau=1.0, p=0.3333, has_groupthink=False, in
        # silence. Total convergence read as no convergence. At 4 rounds the
        # floor is 0.0833 and the same design fires, so the verdict turned on
        # the length of the series and not on the agents.
        #
        # A non-finite tau is the OTHER could-not-check: kendalltau returns NaN
        # for a series with no variance at all (total agreement in every round,
        # echo_chamber_score=1.0), and `nan > 0.3` is False, so the strongest
        # echo chamber there is reported as no groupthink. The class docstring
        # already promised None there; now it is true.
        trend_detectable: Optional[bool] = None
        trend_note = ""
        has_groupthink: Optional[bool]
        if len(values) >= 3:
            tau, p_value = stats.kendalltau(rounds, values)
            trend_detectable, trend_note = detectability(
                _min_attainable_kendall_p(rounds, values), n_family=1, alpha=TREND_ALPHA
            )
            if not math.isfinite(float(tau)) or not math.isfinite(float(p_value)):
                warnings.warn(
                    f"GroupthinkDetector.analyze_convergence: the convergence series "
                    f"{[round(float(v), 4) for v in convergence_trajectory]} has no "
                    f"variance to rank (or is not finite), so Kendall's tau is undefined "
                    f"and no trend test was run. Reporting has_groupthink=None (could not "
                    f"check), NOT False.",
                    UserWarning,
                    stacklevel=2,
                )
                has_groupthink = None
            elif trend_detectable is not True:
                warnings.warn(
                    f"GroupthinkDetector.analyze_convergence: the convergence trend over "
                    f"{len(values)} rounds was NOT ASSESSED. {trend_note} Reporting "
                    f"has_groupthink=None, NOT False.",
                    UserWarning,
                    stacklevel=2,
                )
                has_groupthink = None
            else:
                has_groupthink = bool(tau > TREND_TAU_THRESHOLD and p_value < TREND_ALPHA)
        else:
            # With only 2 rounds there is no trend TEST to run at all, so the
            # first-to-last movement is DESCRIPTION and not a verdict (see the
            # BGL-5 note below, which stopped it deciding has_groupthink). An
            # endpoint that was not measured decides nothing either: `nan > x +
            # 0.1` is False, which reads as "no convergence" for a round nobody
            # could measure.
            first, last = float(convergence_trajectory[0]), float(convergence_trajectory[-1])
            trend_detectable, trend_note = (
                None,
                "NOT A TEST: with 2 rounds there is no trend to test, so no groupthink "
                "verdict is reached at all (has_groupthink is None). The first and last "
                "convergence scores are reported as description; no p-value is computed "
                "and no significance claim is made either way.",
            )
            if not math.isfinite(first) or not math.isfinite(last):
                warnings.warn(
                    "GroupthinkDetector.analyze_convergence: a round's convergence could "
                    "not be measured, so the 2-round comparison was not made. Reporting "
                    "has_groupthink=None (could not check), NOT False.",
                    UserWarning,
                    stacklevel=2,
                )
                has_groupthink = None
            else:
                # BGL-5 (2026-09-27). The three-state fix went into the
                # >= 3-round arm of this dispatch and NOT into this one, so at
                # 2 rounds, the documented minimum this method accepts, the
                # verdict came from `bool(last > first + 0.1)`: a two-state
                # answer from a comparison the result itself labels "NOT A
                # TEST" through trend_detectable=None. Measured that day:
                # analyze_convergence([{'a': [1, 0], 'b': [1, 0]}] * 2), two
                # rounds of TOTAL agreement, gave echo_chamber_score 1.0, the
                # strongest echo chamber this scale has, with has_groupthink
                # FALSE and no warning naming that field, and
                # [{'a': [1, 0], 'b': [0, 1]}] * 2, total diversity, gave
                # echo_chamber_score 0.0 with the same FALSE. The identical
                # agreement data at 3 rounds returns None plus "Reporting
                # has_groupthink=None (could not check), NOT False", and at 3
                # rounds a perfect rise is None too, because the Kendall floor
                # there is 0.3333 against a TREND_ALPHA of 0.1: no data can
                # clear it before 4 rounds. Publishing True or False from 2
                # rounds therefore made the shortest series the MOST confident
                # one. Now: has_groupthink=None with the endpoint rise reported
                # as description, and the 5-round control still measures True
                # (echo 1.0, tau 1.0) and a descending 5-round series still
                # measures False.
                warnings.warn(
                    f"GroupthinkDetector.analyze_convergence: with 2 rounds there is no "
                    f"trend test to run, so no groupthink verdict was reached. The "
                    f"convergence scores moved from {first:.4g} to {last:.4g} "
                    f"(a change of {last - first:+.4g}), which is reported as "
                    f"description only: the same series at 3 rounds is refused for "
                    f"want of power, and a flat series at 1.0 (the strongest echo "
                    f"chamber on this scale) and a flat series at 0.0 (total "
                    f"diversity) are indistinguishable to a first-to-last comparison. "
                    f"Reporting has_groupthink=None (could not check), NOT False and "
                    f"NOT True.",
                    UserWarning,
                    stacklevel=2,
                )
                has_groupthink = None

        # Build agreement matrix from the final round for coalition detection.
        final_round = agent_outputs_per_round[-1]
        agents = sorted(final_round.keys())
        n_agents = len(agents)

        agreement_matrix = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                vec_i = np.array(final_round[agents[i]], dtype=float)
                vec_j = np.array(final_round[agents[j]], dtype=float)
                agreement_matrix[i, j] = self._cosine_similarity(vec_i, vec_j)

        coalitions = self.detect_coalitions(agreement_matrix)
        # Map indices back to agent names.
        named_coalitions = [sorted(agents[idx] for idx in coalition) for coalition in coalitions]

        # Three-state coverage for the coalition structure itself. Every other
        # field on this result already refuses when it cannot measure;
        # coalition_structure was the one that did not, because `nan >= 0.9` is
        # False and so an UNMEASURABLE agent took the same branch as a measured
        # below-threshold one. Name the agents that could not be placed, and
        # say when the grouping of the ones that WERE placed rests on an
        # incomplete agreement graph.
        _finite_agreement = np.isfinite(agreement_matrix)
        _placed = {name for coalition in named_coalitions for name in coalition}
        unplaced_agents = [name for name in agents if name not in _placed]
        # UNORDERED, OFF-DIAGONAL pairs. This counted `n_agents * n_agents`,
        # i.e. every ordered pair plus every agent's comparison with itself, so
        # three agents were reported against a denominator of 9 where they have
        # three pairs, and a single unmeasurable agent read as "5 of 9". The
        # number sits inside a could-not-check sentence a human is meant to act
        # on, so it names the same thing the coalitions do.
        _off_diagonal = ~np.eye(n_agents, dtype=bool)
        n_undefined_pairs = int((_off_diagonal & ~_finite_agreement).sum() // 2)
        n_pairs = n_agents * (n_agents - 1) // 2
        coalitions_detectable: Optional[bool]
        coalition_note = ""
        if n_pairs == 0:
            # A coalition is a PAIRWISE property; a final round with fewer than
            # two agents has no pair, so there is nothing to detect. Without
            # this branch `n_undefined_pairs == 0` was vacuously true and the
            # empty coalition structure was certified detectable.
            coalitions_detectable = None
            coalition_note = (
                f"COULD NOT CHECK: the final round holds {n_agents} agent(s), so there "
                "is no pair to compare and no coalition structure was determined. An "
                "empty coalition_structure here is not a finding that the agents are "
                "independent."
            )
        elif n_undefined_pairs == 0:
            coalitions_detectable = True
        elif not named_coalitions:
            coalitions_detectable = None
            coalition_note = (
                "COULD NOT CHECK: not one pairwise agreement in the final round was "
                "measurable, so no coalition structure was determined. An empty "
                "coalition_structure here is not a finding that the agents are "
                "independent."
            )
        else:
            coalitions_detectable = False
            _who = f" ({', '.join(unplaced_agents)})" if unplaced_agents else ""
            coalition_note = (
                f"COULD NOT CHECK (partial): {n_undefined_pairs} of "
                f"{n_pairs} pairwise agreements in the final round were "
                f"undefined, and {len(unplaced_agents)} agent(s) could not be placed "
                f"at all{_who}. The coalitions below are measured over the remaining "
                f"agents only; the unplaced agents are not independents, and a "
                f"coalition could have been larger through an edge that was never "
                f"measured."
            )

        echo_chamber_score = float(convergence_trajectory[-1])

        # Permutation test: is observed convergence significant?
        observed_convergence = convergence_trajectory[-1]
        rng = np.random.default_rng(42)
        n_permutations = 500
        perm_count = 0

        # `perm >= nan` is False for EVERY draw, so an unmeasurable observed
        # convergence would leave perm_count at 0 and collapse the p-value to
        # 1/(n+1), the SMALLEST this estimator can produce, and report it as
        # significant. This guard was added in the same change that made
        # _compute_round_convergence refuse a single-agent round: without it,
        # that refusal turned a p of 1.0 into a p of 0.002, which is a
        # regression the refusal itself created.
        if not math.isfinite(float(observed_convergence)):
            warnings.warn(
                "GroupthinkDetector.analyze_convergence: observed convergence could not "
                "be measured, so no permutation test was run. Reporting p_value=nan and "
                "is_significant=None (could not check), not a significant result.",
                UserWarning,
                stacklevel=2,
            )
            return GroupthinkResult(
                convergence_trajectory=convergence_trajectory,
                has_groupthink=None,
                coalition_structure=named_coalitions,
                echo_chamber_score=float("nan"),
                p_value=float("nan"),
                is_significant=None,
                metadata=RunMetadata(
                    parameters={
                        "n_rounds": len(agent_outputs_per_round),
                        "n_agents": len(agent_outputs_per_round[0]),
                        "n_permutations": 0,
                    },
                ),
                trend_detectable=trend_detectable,
                trend_note=trend_note,
                convergence_detectable=None,
                convergence_note=(
                    "COULD NOT CHECK: the observed convergence was not measurable, so "
                    "no permutation test was run."
                ),
                unplaced_agents=unplaced_agents,
                coalitions_detectable=coalitions_detectable,
                coalition_note=coalition_note,
                n_permutations_measured=0,
                n_permutations_unmeasurable=0,
            )

        # BGL-W4 (2026-09-30). The guard above refuses an unmeasurable OBSERVED
        # convergence and this loop left the unmeasurable DRAWS voting. The count
        # below used to be `perm_draws.append(float(perm_convergence))` followed by
        # `if perm_convergence >= observed_convergence: perm_count += 1`, and a nan
        # draw is never >= anything, so every draw that could not be measured stayed
        # in the denominator (n_permutations + 1) as a draw that did NOT reach the
        # observed value, which is evidence FOR significance. `finite_draws` was
        # computed twenty lines below and used for min_attainable but NOT for the
        # p-value, so the two numbers in the same result answered different
        # questions.
        #
        # Measured that day on 15 rounds of 2 agents where agent a holds a
        # zero-length output vector in 5 of them (cosine undefined, so
        # _cosine_similarity correctly returns nan): PUBLISHED p_value 0.039920,
        # is_significant True, convergence_detectable True, while 173 of the 500
        # draws were unmeasurable and only 19 of the 327 measurable draws reached
        # the observed convergence. The p over the measurable draws alone is
        # 0.060976, ABOVE this method's PERMUTATION_ALPHA of 0.05, so the published
        # verdict flipped from significant to not significant once the unmeasurable
        # draws stopped voting. R=16/K=5 published 0.047904 against an honest
        # 0.070796 (162 unmeasurable) and R=18/K=6 published 0.045908 against
        # 0.074434 (192 unmeasurable). Nothing disclosed it: metadata still said
        # n_permutations=500, to_dict() had no field naming the loss, and no warning
        # mentioned the draws.
        #
        # The SIBLING detector in this package was fixed for exactly this and
        # carries the disclosure this one lacked: EmergentBiasDetector.analyze
        # counts n_boot_unmeasurable, `continue`s past those resamples instead of
        # counting them, warns with both numbers, and publishes
        # n_bootstrap_unmeasurable. This is the same three states in the same shape.
        perm_draws: list[float] = []
        n_perm_unmeasurable = 0
        for _ in range(n_permutations):
            # Shuffle agent outputs across rounds to break temporal structure.
            all_agents = sorted(agent_outputs_per_round[0].keys())
            # Collect all output vectors across all rounds.
            all_vectors = {
                agent: [round_data[agent] for round_data in agent_outputs_per_round]
                for agent in all_agents
            }
            # For each agent, randomly permute which round its output came from.
            perm_round_data: list[dict[str, list]] = [
                {} for _ in range(len(agent_outputs_per_round))
            ]
            for agent in all_agents:
                perm_indices = rng.permutation(len(agent_outputs_per_round))
                for r_idx, src_idx in enumerate(perm_indices):
                    perm_round_data[r_idx][agent] = all_vectors[agent][src_idx]
            # Compute convergence on the last permuted round.
            perm_convergence = float(self._compute_round_convergence(perm_round_data[-1]))
            if not math.isfinite(perm_convergence):
                # COUNTED AND SKIPPED, never silently counted as "did not reach the
                # observed value". A draw nobody could measure is not evidence in
                # either direction.
                n_perm_unmeasurable += 1
                continue
            perm_draws.append(perm_convergence)
            if perm_convergence >= observed_convergence:
                perm_count += 1

        n_perm_measured = len(perm_draws)
        if n_perm_measured == 0:
            # TOTAL loss, refused, for the same reason the observed-value guard
            # above refuses: with no measurable draw there is no null distribution
            # to compare against, and (0 + 1) / (0 + 1) = 1.0 would publish the
            # most reassuring p-value this estimator can produce out of a test that
            # never ran.
            warnings.warn(
                f"GroupthinkDetector.analyze_convergence: not one of {n_permutations} "
                f"permutation draws had a measurable convergence, so no permutation "
                f"test was run. Reporting p_value=nan and is_significant=None (could "
                f"not check), NOT a p-value of 1.0 and NOT a significant result.",
                UserWarning,
                stacklevel=2,
            )
            return GroupthinkResult(
                convergence_trajectory=convergence_trajectory,
                has_groupthink=has_groupthink,
                coalition_structure=named_coalitions,
                echo_chamber_score=echo_chamber_score,
                p_value=float("nan"),
                is_significant=None,
                metadata=RunMetadata(
                    parameters={
                        "n_rounds": len(agent_outputs_per_round),
                        "n_agents": len(agent_outputs_per_round[0]),
                        "n_permutations": n_permutations,
                        "trend_alpha": TREND_ALPHA,
                        "trend_detectable": trend_detectable,
                        "permutation_alpha": PERMUTATION_ALPHA,
                        "n_permutations_measured": 0,
                        "n_permutations_unmeasurable": n_perm_unmeasurable,
                    },
                ),
                trend_detectable=trend_detectable,
                trend_note=trend_note,
                convergence_detectable=None,
                convergence_note=(
                    f"COULD NOT CHECK: all {n_permutations} permutation draws were "
                    f"unmeasurable, so the observed convergence was never compared "
                    f"against a null distribution. No significance claim is made "
                    f"either way."
                ),
                unplaced_agents=unplaced_agents,
                coalitions_detectable=coalitions_detectable,
                coalition_note=coalition_note,
                n_permutations_measured=0,
                n_permutations_unmeasurable=n_perm_unmeasurable,
            )

        if n_perm_unmeasurable:
            warnings.warn(
                f"GroupthinkDetector.analyze_convergence: {n_perm_unmeasurable} of "
                f"{n_permutations} permutation draws had a convergence that could not be "
                f"measured and were EXCLUDED; the p-value rests on the "
                f"{n_perm_measured} that remain. Counting them as draws that failed to "
                f"reach the observed convergence would be evidence FOR significance that "
                f"nobody measured.",
                UserWarning,
                stacklevel=2,
            )

        p_value_perm = (perm_count + 1) / (n_perm_measured + 1)

        # DISCRETE FLOOR, second instance in this method. The permutation null
        # here shuffles each agent's outputs across rounds, so it holds at most
        # (n_rounds!)^n_agents distinct draws, not n_permutations of them: with
        # 2 agents over 3 rounds there are 36, and a third of them reproduce the
        # observed final round exactly. The floor is therefore not 1/(B+1) but
        # the share of draws that TIE the most extreme value the statistic can
        # take on this data, and the observed convergence is itself one of the
        # draws, so it can never beat that share. Measured 2026-09-10 on two
        # agents converging to identical outputs over 3 rounds: p=0.3353, and
        # no arrangement of that data could have produced less.
        # The denominator here MUST be the one the p-value above uses. It was
        # 1 + n_permutations while the p-value divided by 1 + n_permutations too, so
        # they agreed; now that the p-value rests on the measurable draws only, a
        # floor computed over 500 would UNDERSTATE the floor of an estimator that
        # can only reach 1 / (n_perm_measured + 1), and detectability() would then
        # certify a design whose smallest attainable p-value is above alpha.
        # perm_draws now holds only finite values by construction; the filter stays
        # as a belt so a future edit that appends a raw draw cannot reach the
        # ceiling arithmetic.
        finite_draws = [d for d in perm_draws if math.isfinite(d)]
        min_attainable: Optional[float] = None
        if finite_draws:
            ceiling = max(max(finite_draws), float(observed_convergence))
            ties = sum(1 for d in finite_draws if d >= ceiling)
            min_attainable = (1.0 + ties) / (1.0 + n_perm_measured)
        convergence_detectable, convergence_note = detectability(
            min_attainable, n_family=1, alpha=PERMUTATION_ALPHA
        )
        if n_perm_unmeasurable:
            # PARTIAL loss has to be visible in the field a reader reads, not only
            # in a warning that a library consumer may have filtered. detectability()
            # leaves the note EMPTY when the design has power, which is exactly the
            # case where nothing else on the result would mention the loss.
            _partial = (
                f"PARTIAL COVERAGE: {n_perm_unmeasurable} of {n_permutations} "
                f"permutation draws could not be measured and were excluded; the "
                f"p-value and the detectability floor both rest on the "
                f"{n_perm_measured} that remain."
            )
            convergence_note = f"{convergence_note} {_partial}".strip()
        is_significant: Optional[bool]
        if convergence_detectable is True:
            is_significant = bool(p_value_perm < PERMUTATION_ALPHA)
        else:
            warnings.warn(
                f"GroupthinkDetector.analyze_convergence: the permutation test on the "
                f"final round's convergence was NOT ASSESSED. {convergence_note} "
                f"Reporting is_significant=None (could not check), NOT False.",
                UserWarning,
                stacklevel=2,
            )
            is_significant = None

        logger.info(
            "analyze_convergence complete: has_groupthink=%s, echo_chamber_score=%.3f, p_value=%.4f",
            has_groupthink,
            echo_chamber_score,
            p_value_perm,
        )
        return GroupthinkResult(
            convergence_trajectory=convergence_trajectory,
            has_groupthink=has_groupthink,
            coalition_structure=named_coalitions,
            echo_chamber_score=echo_chamber_score,
            p_value=float(p_value_perm),
            is_significant=is_significant,
            metadata=RunMetadata(
                parameters={
                    "n_rounds": len(agent_outputs_per_round),
                    "n_agents": len(agent_outputs_per_round[0]),
                    "n_permutations": n_permutations,
                    "n_permutations_measured": n_perm_measured,
                    "n_permutations_unmeasurable": n_perm_unmeasurable,
                    "trend_alpha": TREND_ALPHA,
                    "trend_detectable": trend_detectable,
                    "permutation_alpha": PERMUTATION_ALPHA,
                    "min_attainable_p_permutation": min_attainable,
                    "convergence_detectable": convergence_detectable,
                },
            ),
            trend_detectable=trend_detectable,
            trend_note=trend_note,
            convergence_detectable=convergence_detectable,
            convergence_note=convergence_note,
            unplaced_agents=unplaced_agents,
            coalitions_detectable=coalitions_detectable,
            coalition_note=coalition_note,
            n_permutations_measured=n_perm_measured,
            n_permutations_unmeasurable=n_perm_unmeasurable,
        )

    def detect_coalitions(
        self,
        agreement_matrix: np.ndarray,
        threshold: float = 0.9,
    ) -> list[set]:
        """Detect coalitions of highly agreeing agents.

        Uses a simple threshold-based clustering: agents with pairwise
        agreement above the threshold are placed in the same coalition.

        Args:
            agreement_matrix: Square matrix of pairwise agreement scores.
                Shape (n_agents, n_agents). The scale is [-1, 1], not [0, 1]:
                this class's own :meth:`_cosine_similarity` returns -1.0 for
                opposing vectors, and a negative agreement is read here as what
                it is, a measured disagreement that clears no coalition
                threshold.
            threshold: Minimum agreement for agents to be in the same
                coalition, required in [0, 1] because a coalition means POSITIVE
                agreement. A threshold outside that range, or a non-finite one,
                raises ValueError rather than returning a grouping the data could
                not have changed, and the message says which of the two ways it
                could not have changed it: above the range nothing can clear the
                bar, below zero everything does.

        Returns:
            List of sets, each containing agent indices that form a coalition.
            Agents whose agreement could not be measured against ANY PEER are
            OMITTED, not returned as singletons: see below.

        Raises:
            ValueError: If ``threshold`` is not a finite value in [0, 1].

        An undefined agreement is could-not-check, and ``nan >= threshold`` is
        False, which routed it into the same branch as a measured
        below-threshold agreement. An agent whose output vector has zero length
        (cosine undefined against every peer, itself included) therefore came
        back as its own one-member coalition, indistinguishable from an agent
        that was measured and found to agree with nobody. That is the one
        two-state field on an otherwise three-state result, so such agents are
        excluded here and named in ``GroupthinkResult.unplaced_agents``.
        Agents with at least one finite agreement are still placed: dropping
        them too would delete the real finding that the others converged.

        BGL-S2b (2026-09-17). The row test used to include the DIAGONAL, and
        an agent's agreement with itself is not evidence about a coalition.
        ``detect_coalitions([[1.0, nan], [nan, 1.0]])`` therefore returned
        ``[{0}, {1}]`` -- two measured-looking independents -- with no warning,
        on a matrix in which the ONE pair that matters was undefined. The
        diagonal is excluded here, so the test now asks the question the result
        claims to answer: was this agent comparable with any OTHER agent. A
        1x1 matrix has no off-diagonal entry at all and is unplaceable for the
        same reason: one agent has no pair.

        Any off-diagonal pair that is undefined is warned about even when both
        of its agents are placed through other edges, because a coalition can
        only be smaller than the truth when an edge is missing.

        ASYMMETRY. Agreement is taken as symmetric (cosine, the only producer
        inside this class, is). A caller-supplied matrix that is NOT symmetric
        is read cell by cell: ``[[1.0, 0.95], [nan, nan]]`` places agent 0 and
        refuses agent 1, because agent 1 has no finite off-diagonal entry of
        its own. It is not silently symmetrised: inventing ``m[1][0] = 0.95``
        from ``m[0][1]`` would be a fabricated measurement of exactly the kind
        this guard exists to stop.

        An EMPTY (0 agent) matrix returns ``[]`` with a warning for the same
        reason: nothing was examined, and that must not read as a measured
        absence of coalitions.
        """
        # BGL-5 (2026-09-27). The MATRIX was guarded by np.isfinite and the
        # THRESHOLD by nothing, so a threshold no agreement on the documented
        # [0, 1] scale can clear left the membership test unable to fire for any
        # data at all. Measured that day: detect_coalitions(
        # np.array([[1.0, 0.95], [0.95, 1.0]])) returns [{0, 1}], and the SAME
        # fully measured matrix at threshold=1.5 returned [{0}, {1}] with no
        # warning, as did threshold=nan, because `0.95 >= 1.5` and
        # `0.95 >= nan` are both False. That output is byte for byte what a
        # measured absence of coalitions returns, which is the two-meanings-for-
        # one-output defect the empty-matrix branch below exists to close. Now
        # both raise ValueError, the way the sibling detectors reject an
        # out-of-range alpha in __init__, while the default 0.9, an in-scale 0.5
        # and the attainable 1.0 still group that matrix exactly as before
        # ([{0, 1}], [{0, 1}] and [{0}, {1}]).
        #
        # BGL6 AUDIT, 2026-09-29. The refusal is right and the REASON it gave was
        # true of only half the range it refuses. "every agent would come back as
        # its own singleton" holds above the scale and for a non-finite value
        # (`0.95 >= 1.5` and `0.95 >= nan` are both False), and it is FALSE below
        # zero: measured on this class's own example matrix
        # [[1.0, 0.95], [0.95, 1.0]], the pre-fix comparison at -0.5 and at -1.0
        # fires for EVERY pair and the BFS returns [{0, 1}], one coalition holding
        # everybody, the opposite outcome. A refusal that states an untrue reason
        # teaches the caller the wrong lesson about their own data, and it is the
        # only half of the refused range the recorded before-value did not
        # reproduce. One reason per half, each one true of the half it explains.
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            if math.isfinite(threshold) and threshold < 0.0:
                why = (
                    "Agreement never falls below -1.0, so a negative threshold is "
                    "cleared by EVERY pair that was measured at all, and the result "
                    "would be one coalition holding every agent: a grouping the data "
                    "could not have changed, and not a finding that the agents "
                    "converged. The sibling selector get_high_correlations does not "
                    "refuse a negative threshold because it RETURNS the measured "
                    "numbers with a warning that the list is not a selection; this "
                    "function returns a selection and nothing else, so it refuses."
                )
            else:
                why = (
                    "No pair on the agreement scale can clear a threshold above it, "
                    "and no pair can be compared against a non-finite one, so every "
                    "agent would come back as its own singleton: a measured-looking "
                    "absence of coalitions from a comparison that could not have fired."
                )
            raise ValueError(
                f"threshold must be a finite agreement score in [0, 1], got {threshold!r}. {why}"
            )
        matrix = np.asarray(agreement_matrix, dtype=float)
        n = matrix.shape[0]
        if n == 0:
            # An empty matrix EXAMINED NOTHING, and `[]` is also what a measured
            # absence of coalitions returns, so the two were indistinguishable.
            # Measured 2026-09-27: `detect_coalitions(np.zeros((0, 0)))` returned
            # `[]` with no warning at all, the same output as a matrix in which
            # every pair was compared and none agreed. `analyze_convergence`
            # already rules on this shape through `n_pairs == 0`; the helper is
            # public and reachable on its own, so it has to say it too.
            warnings.warn(
                "GroupthinkDetector.detect_coalitions: the agreement matrix holds 0 "
                "agents, so no pair was compared and no coalition structure was "
                "determined. The empty list returned is NOT a finding that the agents "
                "are independent (could not check).",
                UserWarning,
                stacklevel=2,
            )
            return []
        finite = np.isfinite(matrix)
        # OFF-DIAGONAL only: an agent compared with itself says nothing about
        # whether it could be compared with anybody else.
        off_diagonal = ~np.eye(n, dtype=bool)
        finite_pairs = finite & off_diagonal
        unplaceable = {i for i in range(n) if not bool(finite_pairs[i].any())}
        n_undefined_pairs = int((off_diagonal & ~finite).sum() // 2) if n > 1 else 0
        if unplaceable or n_undefined_pairs:
            _who = (
                f"agent index(es) {sorted(unplaceable)} have no measurable agreement "
                f"with any other agent, so their coalition membership COULD NOT BE "
                f"CHECKED; they are omitted from the returned coalitions rather than "
                f"reported as independent singletons, which would read as a "
                f"measurement. "
                if unplaceable
                else ""
            )
            _pairs = (
                f"{n_undefined_pairs} pair(s) of agents could not be compared at all, "
                f"so a coalition below could be SMALLER than the truth through an edge "
                f"that was never measured."
                if n_undefined_pairs
                else ""
            )
            warnings.warn(
                f"GroupthinkDetector.detect_coalitions: {_who}{_pairs}",
                UserWarning,
                stacklevel=2,
            )
        visited = set(unplaceable)
        coalitions = []

        for i in range(n):
            if i in visited:
                continue

            coalition = {i}
            visited.add(i)

            # BFS to find all connected agents above threshold.
            queue = [i]
            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if j not in visited and matrix[current, j] >= threshold:
                        coalition.add(j)
                        visited.add(j)
                        queue.append(j)

            coalitions.append(coalition)

        return coalitions

    @staticmethod
    def _compute_round_convergence(round_data: dict[str, list]) -> float:
        """Compute average pairwise cosine similarity for a single round."""
        agents = list(round_data.keys())
        if len(agents) < 2:
            # 1.0 is TOTAL agreement on this scale, the strongest groupthink
            # signal there is, and it was returned for a round with nobody to
            # compare against. Convergence is pairwise; one agent has no pair.
            warnings.warn(
                f"GroupthinkDetector: a round with {len(agents)} agent(s) has no pair "
                f"to compare, so convergence was not measured. Returning nan, not 1.0, "
                f"which reads as total agreement.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        similarities = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                vec_i = np.array(round_data[agents[i]], dtype=float)
                vec_j = np.array(round_data[agents[j]], dtype=float)
                sim = GroupthinkDetector._cosine_similarity(vec_i, vec_j)
                similarities.append(sim)

        return float(np.mean(similarities))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity, or NaN when the angle is undefined.

        A zero-length vector has no direction, so its cosine against anything
        is undefined. Returning 0.0 for it put the MINIMUM of the convergence
        scale on a pair nobody could compare, and 0 on that scale is a
        measurement: maximal disagreement. Measured 2026-09-10 on three rounds
        of two agents holding the identical opinion throughout: encoded as
        [0.0, 0.0] the trajectory read [0.0, 0.0, 0.0] (total disagreement),
        and encoded as [0.001, 0.0] the same behaviour read [1.0, 1.0, 1.0]
        (total agreement). The verdict turned on the encoding of a label.
        NaN travels into `_compute_round_convergence` and out to the
        could-not-check branch of `analyze_convergence`, which is where an
        unmeasurable round belongs.
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0.0 or norm_b == 0.0:
            warnings.warn(
                "GroupthinkDetector: cosine similarity is undefined against a "
                "zero-length output vector, so this pair's agreement was not "
                "measured. Returning nan, not 0.0, which reads as total "
                "disagreement on the convergence scale.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        return float(np.dot(a, b) / (norm_a * norm_b))
