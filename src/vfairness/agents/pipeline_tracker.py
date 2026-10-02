"""
Multi-stage pipeline bias tracking for AI agents.

Tracks how bias accumulates, amplifies, or attenuates across sequential
processing stages in an AI agent pipeline (e.g., retrieval -> reasoning ->
tool selection -> action). Identifies which stage contributes most to
overall system bias.
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from vfairness.evaluation.vfairness_metrics._statistics import _mannwhitney_two_sided_p
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class StageResult(SerializableMixin):
    """Bias measurement for a single pipeline stage.

    Attributes:
        stage_name: Name of the pipeline stage.
        bias_metrics: Dictionary of bias metric names to values.
        cumulative_bias: Cumulative bias up to and including this stage.
            NaN once any earlier stage was unmeasurable, because a running
            sum with a hole in it is not a total.
        stage_contribution: This stage's marginal contribution to bias
            (this stage's disparity minus the PREVIOUS stage's). NaN when
            either of those two is unmeasurable, which includes the case
            where this stage is perfectly measured and its PREDECESSOR is
            not: the marginal is a difference, and a difference against an
            unknown is unknown. It is never 0.0 and never a carry-forward.
    """

    stage_name: str
    bias_metrics: dict
    cumulative_bias: float
    stage_contribution: float
    metadata: RunMetadata = field(default_factory=RunMetadata)


class PipelineTracker:
    """Tracks bias accumulation across multi-stage AI agent pipelines.

    Records per-stage outcomes for two demographic groups and computes
    how bias evolves through the pipeline. This enables identification
    of which pipeline stage is the primary source of bias.

    Args:
        stages: Ordered list of stage names in the pipeline.

    Example:
        >>> tracker = PipelineTracker(
        ...     stages=["retrieval", "reasoning", "tool_selection", "action"]
        ... )
        >>> tracker.record_stage("retrieval", np.array([0.8, 0.7]), np.array([0.6, 0.5]))
        >>> tracker.record_stage("reasoning", np.array([0.9, 0.8]), np.array([0.5, 0.4]))
        >>> results = tracker.compute_cumulative()

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: pipeline_tracker. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, stages: List[str]) -> None:
        if len(stages) == 0:
            raise ValueError("At least one stage is required.")

        self.stages = stages
        self._stage_data: Dict[str, dict] = {}
        logger.info("PipelineTracker initialized: stages=%s", stages)

    def record_stage(
        self,
        stage: str,
        outcomes_a: np.ndarray,
        outcomes_b: np.ndarray,
    ) -> None:
        """Record outcomes for a pipeline stage.

        Args:
            stage: Name of the stage (must be in the configured stages).
            outcomes_a: Outcome values for demographic group A.
            outcomes_b: Outcome values for demographic group B.

        Raises:
            ValueError: If stage name is not in the configured stages.
        """
        if stage not in self.stages:
            raise ValueError(f"Unknown stage '{stage}'. Expected one of: {self.stages}")

        outcomes_a = np.asarray(outcomes_a, dtype=float)
        outcomes_b = np.asarray(outcomes_b, dtype=float)

        if len(outcomes_a) == 0 or len(outcomes_b) == 0:
            raise ValueError("outcomes_a and outcomes_b must be non-empty arrays.")

        logger.debug(
            "record_stage: stage=%s, n_a=%d, n_b=%d", stage, len(outcomes_a), len(outcomes_b)
        )
        self._stage_data[stage] = {
            "outcomes_a": outcomes_a,
            "outcomes_b": outcomes_b,
        }

    def compute_cumulative(self) -> List[StageResult]:
        """Compute per-stage and cumulative bias across the pipeline.

        Uses mean difference as the bias metric. Cumulative bias is
        the running sum of absolute stage contributions.

        Returns:
            List of StageResult in pipeline order, only for stages
            that have recorded data. A configured stage with no recorded data is
            absent from the list AND named in a warning, because an absent stage
            was not measured and must not read as a stage with no bias.
        """
        logger.info(
            "compute_cumulative: %d stages configured, %d recorded",
            len(self.stages),
            len(self._stage_data),
        )
        results = []
        cumulative = 0.0
        previous_bias = 0.0
        untestable: List[str] = []
        no_p_value: List[str] = []
        unmeasurable: List[str] = []
        unmeasurable_contribution: List[str] = []

        for stage in self.stages:
            if stage not in self._stage_data:
                continue

            data = self._stage_data[stage]
            # A single non-finite outcome makes np.mean NaN, and NaN then
            # travels through stage_contribution into EVERY downstream stage's
            # contribution and the cumulative total, in silence: `untestable`
            # below is populated only by the n<=1 branch, so nothing warned.
            # NaN is still the honest value for the mean (dropping the bad rows
            # would be imputation by omission, and it would change the
            # denominator without saying so), but it must be SAID, and the
            # counts go beside the disparity for the same reason n_a and n_b do.
            n_a_bad = int((~np.isfinite(data["outcomes_a"])).sum())
            n_b_bad = int((~np.isfinite(data["outcomes_b"])).sum())
            mean_a = float(np.mean(data["outcomes_a"]))
            mean_b = float(np.mean(data["outcomes_b"]))

            stage_bias = abs(mean_a - mean_b)
            if not math.isfinite(stage_bias):
                unmeasurable.append(f"{stage} (n_a_nonfinite={n_a_bad}, n_b_nonfinite={n_b_bad})")

            # Mann-Whitney test for significance.
            #
            # READINESS-6, 2026-09-10. The `else` branch returned p_value = 1.0
            # when a side held ONE observation, and 1.0 is the most reassuring
            # p-value there is: it reads as "measured, and there is no difference
            # at all". No test ran. Measured on this repo before the change, a
            # three-stage pipeline where one stage held a single observation per
            # arm reported that stage at p=1.0000 AND abs_disparity=1.000, and
            # `identify_bias_source` named it the PRIMARY BIAS SOURCE, outranking
            # a stage with forty observations and a genuinely significant
            # p=0.0221. The whole verdict rested on one pair of numbers.
            #
            # NaN is the honest value. Identical arms keep p = 1.0 because that
            # is a real reading: two arms with the same outcomes genuinely show
            # no difference, and refusing there would discard a confirmation.
            #
            # The `np.array_equal` shortcut is LOAD-BEARING. It was documented
            # here as defence in depth on 2026-09-10, on the evidence that scipy
            # 1.17 returns exactly 1.0 for identical arms so removing it changed
            # nothing. The assumption was pinned rather than described, and it
            # broke the same day: CI runs scipy 1.18.1, where `mannwhitneyu`
            # returns NaN for identical arms, and the pin said so in one line.
            #
            # So on 1.18 this shortcut is the ONLY reason two identical arms
            # report p = 1.0 instead of a could-not-check. At temperature 0 that
            # is the commonest genuinely fair shape there is, so without it every
            # fair comparison would be reported as unmeasurable on a modern
            # scipy. Do not remove it, and do not "simplify" it back into the
            # call below.
            n_a, n_b = len(data["outcomes_a"]), len(data["outcomes_b"])
            if n_a > 1 and n_b > 1:
                if np.array_equal(data["outcomes_a"], data["outcomes_b"]):
                    p_value = 1.0
                else:
                    # The shared helper, for the case the shortcut above cannot
                    # see: every value tied but the arms of DIFFERENT lengths, so
                    # array_equal is False. scipy 1.18 returned nan there and the
                    # stage left the ranking with no warning naming it; the exact
                    # p is 1.0. Any other missing p stays NaN and is named below.
                    measured = _mannwhitney_two_sided_p(data["outcomes_a"], data["outcomes_b"])
                    if measured is None:
                        p_value = float("nan")
                        # Non-finite outcomes are already named by `unmeasurable`.
                        if n_a_bad == 0 and n_b_bad == 0:
                            no_p_value.append(stage)
                    else:
                        p_value = measured
            else:
                p_value = float("nan")
                untestable.append(f"{stage} (n_a={n_a}, n_b={n_b})")

            # Stage contribution is the change from previous stage.
            stage_contribution = stage_bias - previous_bias
            if not math.isfinite(stage_contribution) and math.isfinite(stage_bias):
                # This stage WAS measured; only its predecessor was not, so
                # its marginal contribution is unknown. That is the case that
                # hid the strongest stage in the pipeline: its own p_value is
                # finite, so it passed identify_bias_source's `testable`
                # filter, and then lost every `max()` comparison in silence.
                unmeasurable_contribution.append(stage)
            cumulative += abs(stage_contribution)

            metrics = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_difference": mean_a - mean_b,
                "abs_disparity": stage_bias,
                "p_value": float(p_value),
                # The DENOMINATOR. A disparity of 1.000 over one observation per
                # arm and one over forty are different claims, and without these
                # they are rendered identically.
                "n_a": n_a,
                "n_b": n_b,
                # How many of those observations were not finite. A mean over
                # an arm holding a NaN is NaN, and these say why.
                "n_a_nonfinite": n_a_bad,
                "n_b_nonfinite": n_b_bad,
            }

            results.append(
                StageResult(
                    stage_name=stage,
                    bias_metrics=metrics,
                    cumulative_bias=cumulative,
                    stage_contribution=stage_contribution,
                )
            )

            previous_bias = stage_bias

        if unmeasurable:
            warnings.warn(
                f"compute_cumulative: {len(unmeasurable)} stage(s) hold non-finite "
                f"outcomes, so their group means and their abs_disparity COULD NOT BE "
                f"MEASURED and are NaN, not 0.0: {', '.join(unmeasurable)}. This is not "
                f"a finding that those stages are unbiased.",
                UserWarning,
                stacklevel=2,
            )
        if unmeasurable_contribution:
            warnings.warn(
                f"compute_cumulative: {len(unmeasurable_contribution)} stage(s) were "
                f"measured but follow an unmeasurable stage, so their "
                f"stage_contribution and every later cumulative_bias are NaN: "
                f"{', '.join(unmeasurable_contribution)}. Their own abs_disparity IS "
                f"measured and may be the largest in the pipeline; they simply cannot "
                f"be RANKED by marginal contribution.",
                UserWarning,
                stacklevel=2,
            )
        # BGL-3 (2026-09-27): a configured stage that was never recorded is
        # skipped by the loop above, which is the documented behaviour, but
        # nothing SAID so. Measured that day on
        # PipelineTracker(["retrieval", "reasoning", "tool_selection", "action"])
        # with only 'retrieval' recorded: compute_cumulative returned one row and
        # identify_bias_source returned 'retrieval' with ZERO warnings, so the
        # sentence an operator acts on, "retrieval is the stage contributing most
        # to overall bias", rested on one quarter of the declared pipeline and
        # said nothing about the three stages nobody measured. The two warnings
        # above already hold the standard for this file: a stage left out of the
        # ranking is disclosed and is explicitly not cleared. Never-recorded
        # stages are the same omission one step earlier, and the ranking is still
        # a real measurement of what WAS recorded, so this discloses rather than
        # refuses.
        never_recorded = [s for s in self.stages if s not in self._stage_data]
        if never_recorded:
            warnings.warn(
                f"compute_cumulative: {len(never_recorded)} of {len(self.stages)} "
                f"configured stage(s) were never recorded and are ABSENT from the "
                f"result: {', '.join(never_recorded)}. Everything computed here, "
                f"including any ranking by identify_bias_source, covers only the "
                f"{len(results)} recorded stage(s). The absent stages were not "
                f"measured, which is not a finding that they carry no bias.",
                UserWarning,
                stacklevel=2,
            )
        if no_p_value:
            warnings.warn(
                f"compute_cumulative: the Mann-Whitney test returned no p-value for "
                f"{len(no_p_value)} stage(s), so their p_value is NaN and they are "
                f"excluded from identify_bias_source: {', '.join(no_p_value)}. That is "
                f"a test that did not run, not evidence of an absence of difference.",
                UserWarning,
                stacklevel=2,
            )
        if untestable:
            warnings.warn(
                f"compute_cumulative: {len(untestable)} stage(s) hold fewer than two "
                f"observations on a side, so no significance test could run for them "
                f"and their p_value is NaN, NOT 1.0: {', '.join(untestable)}. Their "
                f"disparities are still reported and are computed from those few "
                f"observations; they are not evidence of an absence of difference.",
                UserWarning,
                stacklevel=2,
            )
        return results

    def identify_bias_source(self) -> Optional[str]:
        """Identify the pipeline stage with the highest bias contribution.

        Returns:
            Name of the stage contributing most to overall bias, or ``None``
            when the ranking COULD NOT BE COMPLETED because a stage's own
            ranking key was unmeasurable. ``None`` is not a finding that the
            pipeline has no bias source; it means the question was not
            answered, and the warning says which stages blocked it.

            The name returned is the largest contributor AMONG THE STAGES THAT
            WERE RECORDED AND TESTABLE. Configured stages with no recorded data
            take no part in the ranking and are named in a warning from
            ``compute_cumulative``; they are not cleared by this answer.

        Raises:
            RuntimeError: If no stages have been recorded, or if not one
                recorded stage could be tested at all.
        """
        results = self.compute_cumulative()
        if not results:
            raise RuntimeError("No stage data has been recorded.")

        # READINESS-6, 2026-09-10. This ranked EVERY recorded stage, including
        # stages whose disparity rests on a single observation per arm. Measured
        # before the change: a stage with one observation each way, disparity
        # 1.000 and no test possible, was returned as the primary bias source
        # ahead of a stage with forty observations and a real p of 0.0221. The
        # return value is a sentence an operator acts on, so it must not be
        # decided by a pair of numbers that could not be tested.
        #
        # Rank among the stages whose test could RUN. Excluding the rest is not
        # a claim that they are clean, and the warning says so.
        testable = [
            r for r in results if not math.isnan(r.bias_metrics.get("p_value", float("nan")))
        ]
        excluded = [r.stage_name for r in results if r not in testable]

        if not testable:
            raise RuntimeError(
                "No stage could be tested: every recorded stage holds fewer than two "
                "observations on a side, so no significance test could run and no "
                "stage can be named the bias source. This is NOT a finding that the "
                f"pipeline is unbiased. Stages recorded: {[r.stage_name for r in results]}."
            )

        # GUARD ABOVE THE SELECTION. The filter above tests the p-value; the
        # ranking below reads stage_contribution. A stage can pass the first
        # and have no value for the second: when its PREDECESSOR was
        # unmeasurable, its marginal contribution is NaN while its own p-value
        # is perfectly finite. `max()` keeps whichever candidate it saw first
        # because every comparison against NaN is False, so such a stage lost
        # the ranking silently AND was missing from the excluded list, which
        # made the warning affirmatively false about it. Measured before this
        # change: a three-stage pipeline whose 'action' stage carried
        # abs_disparity 0.405 at p=1.9e-14 over 40 observations a side was
        # beaten by 'retrieval' at abs_disparity 0.041, p=0.125, and the
        # warning named only 'reasoning'.
        #
        # An unmeasurable ranking key is not an exclusion, it is an
        # incomplete ranking: there is no bound on where that stage would have
        # placed. Refuse the whole answer rather than name a winner of a race
        # somebody did not finish. This is deliberately narrower than the
        # p-value exclusion above, which stays as it was: a stage that is
        # untestable but whose CONTRIBUTION is measured is still ranked out,
        # and the largest testable contributor is still named.
        unrankable = [r.stage_name for r in testable if not math.isfinite(r.stage_contribution)]
        if unrankable:
            warnings.warn(
                f"identify_bias_source: {len(unrankable)} stage(s) have a MEASURED "
                f"disparity but no computable marginal contribution, because the stage "
                f"before them could not be measured ({', '.join(unrankable)}). They "
                f"cannot be placed in the ranking and nothing bounds where they would "
                f"have placed, so NO stage is named: returning None (could not check). "
                f"This is not a finding that the pipeline has no bias source.",
                UserWarning,
                stacklevel=2,
            )
            return None

        if excluded:
            warnings.warn(
                f"identify_bias_source: {len(excluded)} stage(s) could not be tested "
                f"and were excluded from the ranking ({', '.join(excluded)}). The stage "
                f"named below is the largest contributor AMONG THE TESTABLE ONES; the "
                f"excluded stages are not cleared.",
                UserWarning,
                stacklevel=2,
            )

        # SIGNED contribution: the source of bias is the stage that ADDS the
        # most bias. Ranking by abs() named a strongly bias-REDUCING stage
        # (large negative contribution) as the primary bias source.
        max_result = max(testable, key=lambda r: r.stage_contribution)
        return max_result.stage_name
