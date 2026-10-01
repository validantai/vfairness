"""
Tool selection bias auditing for AI agents.

Measures whether an AI agent selects different tools or invokes them
at different rates depending on the demographic group of the user or
subject. For example, an agent might route certain demographics to
manual review tools more often than others.
"""

import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_fisher,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class ToolBiasResult(SerializableMixin):
    """Result of a tool bias analysis for a single tool.

    Attributes:
        tool_name: Name of the tool analyzed.
        invocation_rate_a: Invocation rate for demographic group A.
        invocation_rate_b: Invocation rate for demographic group B.
        disparity_ratio: Ratio of invocation rates (min / max), or NaN when
            neither group invoked the tool at all so no ratio exists.
        is_significant: Whether disparity is statistically significant, or None
            when the test could never have reached ``alpha`` at these group
            sizes, whatever the tool selection had been. None is could-not-check
            and is never "no disparity": see
            :meth:`ToolBiasAuditor.compute_selection_disparity`.
        p_value: P-value from chi-square or Fisher's exact test.
        confidence_interval: Percentile bootstrap interval for
            ``disparity_ratio``, or ``(nan, nan)`` when no bootstrap draw
            produced a defined ratio.
        n_bootstrap_draws: Bootstrap resamples attempted.
        n_bootstrap_undefined: Draws in which the tool appeared in NEITHER
            resample, so the ratio was 0/0. They are EXCLUDED from the
            percentile rather than scored.
        ci_coverage: Fraction of draws the interval actually rests on
            (``1 - n_bootstrap_undefined / n_bootstrap_draws``), or NaN when
            nothing was drawn. Read it before reading the interval.

    BGL-S2 (2026-09-16). The three bootstrap fields are new. A 0/0 draw used to
    be scored ``1.0``, and on this min/max scale 1.0 is PERFECT PARITY, the best
    attainable value. Measured on 198 approve + 2 escalate_to_human against 200
    approve, seed 42:

        disparity_ratio 0.0 (total exclusion of group B, a real measurement),
        confidence_interval (0.0, 1.0), no warning at all.

    133 of the 1000 draws had the tool in neither resample and each was scored
    1.0; the distinct ratio values across the whole distribution were exactly
    {0.0, 1.0}, so the 97.5th-percentile upper bound of 1.0 was made ENTIRELY of
    those undefined draws. Every 0/0 draw pushes the interval toward "no
    disparity" and never toward disparity, so a reader saw an interval spanning
    total exclusion to perfect parity for a tool group B never used once. With
    the 133 excluded the interval is (0.0, 0.0).
    """

    tool_name: str
    invocation_rate_a: float
    invocation_rate_b: float
    disparity_ratio: float
    is_significant: Optional[bool]
    p_value: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    n_bootstrap_draws: int = 0
    n_bootstrap_undefined: int = 0
    ci_coverage: float = float("nan")


class ToolBiasAuditor:
    """Auditor for tool selection bias in AI agent traces.

    Compares tool invocation patterns between demographic groups to
    detect whether agents use different tools depending on the subject's
    demographic attributes.

    Args:
        alpha: Significance level for hypothesis tests.

    Example:
        >>> auditor = ToolBiasAuditor(alpha=0.05)
        >>> traces_a = [
        ...     {"tool": "approve", "timestamp": 1},
        ...     {"tool": "approve", "timestamp": 2},
        ... ]
        >>> traces_b = [
        ...     {"tool": "manual_review", "timestamp": 1},
        ...     {"tool": "approve", "timestamp": 2},
        ... ]
        >>> results = auditor.analyze_tool_calls(traces_a, traces_b)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: tool_bias_auditor. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, alpha: float = 0.05, random_seed: int = 42) -> None:
        self.alpha = alpha
        self.random_seed = random_seed
        logger.info("ToolBiasAuditor initialized: alpha=%.3f", alpha)

    def analyze_tool_calls(
        self,
        traces_a: list[dict],
        traces_b: list[dict],
    ) -> list[ToolBiasResult]:
        """Compare tool invocation patterns between two demographic groups.

        Extracts tool names from agent traces and performs per-tool
        statistical tests for differential invocation rates.

        Args:
            traces_a: Agent execution traces for demographic group A.
                Each dict must contain a ``tool`` key with the tool name.
            traces_b: Agent execution traces for demographic group B.

        Returns:
            List of ToolBiasResult, one per unique tool observed.

        Raises:
            ValueError: If either trace list is empty.
        """
        logger.info(
            "analyze_tool_calls: n_traces_a=%d, n_traces_b=%d", len(traces_a), len(traces_b)
        )
        if len(traces_a) == 0 or len(traces_b) == 0:
            raise ValueError("Trace lists must not be empty.")

        MIN_RECOMMENDED_SAMPLES = 30
        if len(traces_a) < MIN_RECOMMENDED_SAMPLES or len(traces_b) < MIN_RECOMMENDED_SAMPLES:
            import warnings

            logger.warning(
                "Sample size (%d, %d) below recommended minimum of %d",
                len(traces_a),
                len(traces_b),
                MIN_RECOMMENDED_SAMPLES,
            )
            warnings.warn(
                f"Sample size ({len(traces_a)}, {len(traces_b)}) below recommended minimum "
                f"of {MIN_RECOMMENDED_SAMPLES}. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        tools_a = [t["tool"] for t in traces_a]
        tools_b = [t["tool"] for t in traces_b]

        return self._compare_tool_distributions(tools_a, tools_b)

    def compute_selection_disparity(
        self,
        tools_a: list[str],
        tools_b: list[str],
    ) -> dict:
        """Compute per-tool selection disparity with chi-square tests.

        Args:
            tools_a: Tool names invoked for demographic group A.
            tools_b: Tool names invoked for demographic group B.

        Returns:
            Dictionary mapping tool names to disparity dictionaries with keys
            ``rate_a``, ``rate_b``, ``p_value``, ``is_significant``
            (None when no selection could have been significant at these group
            sizes), ``min_attainable_p`` and ``detectable``.

        Raises:
            ValueError: If either tool list is empty.
        """
        if len(tools_a) == 0 or len(tools_b) == 0:
            raise ValueError("Tool lists must not be empty.")

        count_a = Counter(tools_a)
        count_b = Counter(tools_b)
        all_tools = sorted(set(count_a.keys()) | set(count_b.keys()))

        total_a = len(tools_a)
        total_b = len(tools_b)

        # DISCRETE FLOOR (2026-09-27). Fisher's exact conditions on both margins,
        # so the smallest p a per-tool 2x2 can return is fixed by the two group
        # sizes before any tool call is looked at. Measured 2026-09-27 on 3
        # episodes that all called "approve" for group A against 3 that all
        # called "escalate" for group B, which is TOTAL EXCLUSION of each group
        # from the other's tool:
        #
        #   approve   rate 1.0 vs 0.0, p = 0.1, is_significant = False
        #   escalate  rate 0.0 vs 1.0, p = 0.1, is_significant = False
        #   disparity_ratio 0.0 on both, and no warning anywhere
        #
        # 0.1 is the SMALLEST p 3 against 3 can produce, so the False was not a
        # reading about these tools; no data at those group sizes could have
        # produced anything else. The disparity_ratio beside it (0.0, the worst
        # value on its scale) was a true measurement all along, which is what
        # made the False credible. Same fix and same two helpers as
        # DelegationRoutingAuditor (2026-09-10) and
        # ActionBiasAnalyzer.analyze_delegation.
        family_floor = min_attainable_p_fisher(total_a, total_b)
        detectable, note = detectability(family_floor, n_family=1, alpha=self.alpha)

        results = {}
        for tool in all_tools:
            n_a = count_a.get(tool, 0)
            n_b = count_b.get(tool, 0)

            rate_a = n_a / total_a
            rate_b = n_b / total_b

            # 2x2 contingency: [tool_used, tool_not_used] x [group_a, group_b]
            table = np.array(
                [
                    [n_a, total_a - n_a],
                    [n_b, total_b - n_b],
                ]
            )

            total = table.sum()
            if total < 40 or np.any(table < 5):
                _, p_value = stats.fisher_exact(table)
            else:
                chi2_stat, p_value, _, _ = stats.chi2_contingency(table, correction=True)

            results[tool] = {
                "rate_a": float(rate_a),
                "rate_b": float(rate_b),
                "p_value": float(p_value),
                # None, never False: at these group sizes the verdict may be one
                # the test could not have reached whatever the agent selected.
                "is_significant": (bool(p_value < self.alpha) if detectable is True else None),
                "min_attainable_p": family_floor,
                "detectable": detectable,
            }

        if detectable is not True:
            warnings.warn(
                f"ToolBiasAuditor.compute_selection_disparity: per-tool selection "
                f"significance was NOT ASSESSED for {total_a} vs {total_b} tool calls. "
                f"{note} Reporting is_significant=None for all {len(all_tools)} tool(s) "
                f"(could not check), NOT False. The measured rates and disparity ratios "
                f"still stand.",
                UserWarning,
                stacklevel=2,
            )

        return results

    def _bootstrap_ci_ratio(self, tools_a, tools_b, tool_name, n_bootstrap=1000, confidence=0.95):
        """Bootstrap CI for the disparity ratio of a single tool.

        Returns ``(low, high, n_undefined)``. A draw in which the tool appears
        in NEITHER resample has no ratio: 0/0 is undefined, not parity. Such
        draws are counted and excluded from the percentile instead of being
        scored ``1.0``, which on this min/max scale is the BEST attainable
        value and therefore biases the interval in exactly one direction, the
        reassuring one. See :class:`ToolBiasResult` for the measured numbers.
        """
        rng = np.random.default_rng(self.random_seed)
        ratios = []
        n_undefined = 0
        for _ in range(n_bootstrap):
            boot_a = rng.choice(tools_a, size=len(tools_a), replace=True)
            boot_b = rng.choice(tools_b, size=len(tools_b), replace=True)
            r_a = np.mean(boot_a == tool_name)
            r_b = np.mean(boot_b == tool_name)
            higher = max(r_a, r_b)
            if higher <= 0:
                n_undefined += 1
                continue
            ratios.append(min(r_a, r_b) / higher)
        if not ratios:
            return float("nan"), float("nan"), n_undefined
        alpha = 1 - confidence
        return (
            float(np.percentile(ratios, 100 * alpha / 2)),
            float(np.percentile(ratios, 100 * (1 - alpha / 2))),
            n_undefined,
        )

    def _compare_tool_distributions(
        self,
        tools_a: list[str],
        tools_b: list[str],
    ) -> list[ToolBiasResult]:
        """Internal: compare tool distributions and return typed results."""
        disparity_dict = self.compute_selection_disparity(tools_a, tools_b)

        arr_a = np.array(tools_a)
        arr_b = np.array(tools_b)

        results = []
        for tool_name, info in disparity_dict.items():
            rate_a = info["rate_a"]
            rate_b = info["rate_b"]

            higher = max(rate_a, rate_b)
            if higher == 0.0:
                # 0/0 is undefined, NOT parity. Same fabrication as the one
                # removed from the bootstrap loop; fixed at both sites at once
                # because they share the precondition, and a ratio of 1.0 here
                # would be the strongest possible all-clear over no data.
                ratio = float("nan")
            else:
                ratio = min(rate_a, rate_b) / higher

            n_draws = 1000
            ci_low, ci_high, n_undefined = self._bootstrap_ci_ratio(
                arr_a, arr_b, tool_name, n_bootstrap=n_draws
            )
            coverage = (n_draws - n_undefined) / n_draws if n_draws else float("nan")

            if n_undefined:
                warnings.warn(
                    f"ToolBiasAuditor: the bootstrap interval for {tool_name!r} rests on "
                    f"{n_draws - n_undefined} of {n_draws} draws. In {n_undefined} draw(s) the "
                    f"tool appeared in neither resample, so the ratio was 0/0 and undefined; "
                    f"those draws are EXCLUDED, not scored as 1.0 (perfect parity). Read "
                    f"ci_coverage={coverage:.3f} alongside the interval.",
                    UserWarning,
                    stacklevel=3,
                )

            results.append(
                ToolBiasResult(
                    tool_name=tool_name,
                    invocation_rate_a=rate_a,
                    invocation_rate_b=rate_b,
                    disparity_ratio=float(ratio),
                    is_significant=info["is_significant"],
                    p_value=info["p_value"],
                    confidence_interval=(ci_low, ci_high),
                    n_bootstrap_draws=n_draws,
                    n_bootstrap_undefined=n_undefined,
                    ci_coverage=coverage,
                )
            )

        return results
