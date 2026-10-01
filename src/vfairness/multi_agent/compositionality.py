"""
Non-compositionality analysis for multi-agent systems.

Tests whether bias in a multi-agent system can be predicted from
the biases of its individual components. When system-level bias
diverges from component-level biases, this indicates emergent
non-compositional bias behavior.
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class CompositionalityResult(SerializableMixin):
    """Result of a compositionality analysis.

    Attributes:
        component_scores: Per-component bias scores.
        system_score: Measured system-level bias.
        divergence: Difference between system and aggregate component bias.
        scenario: Classification of the divergence pattern, or ``"not_assessed"``
            when the system or a component could not be measured.
    """

    component_scores: dict
    system_score: float
    divergence: float
    scenario: Literal["amplification", "reduction", "novel_emergence", "consistent", "not_assessed"]
    metadata: RunMetadata = field(default_factory=RunMetadata)


class CompositionalityAnalyzer:
    """Analyzes whether multi-agent system bias is compositional.

    Compares the aggregate of individual component biases against the
    measured system-level bias to classify the bias pattern.

    Example:
        >>> analyzer = CompositionalityAnalyzer()
        >>> result = analyzer.analyze(
        ...     component_biases={"agent_a": 0.05, "agent_b": 0.03},
        ...     system_bias=0.15,
        ... )
        >>> result.scenario
        'amplification'

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: compositionality_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def analyze(
        self,
        component_biases: dict[str, float],
        system_bias: float,
        threshold: float = 0.05,
        aggregation_method: str = "max",
    ) -> CompositionalityResult:
        """Analyze compositionality of bias in a multi-agent system.

        Compares measured system bias against aggregate component biases
        and classifies the result into one of four scenarios (or reports
        ``not_assessed`` when the comparison could not be made):

        - **amplification**: System bias exceeds component aggregate.
        - **reduction**: System bias is lower than component aggregate.
        - **novel_emergence**: Components show no bias but system does.
        - **consistent**: System and component biases are aligned.

        Args:
            component_biases: Dictionary mapping component names to their
                individual bias scores (absolute values expected).
            system_bias: Measured system-level bias (absolute value).
            threshold: Tolerance for classifying divergence. Differences
                within this threshold are considered consistent. A tolerance,
                so it must be finite and greater than 0; anything else is
                refused rather than compared against.
            aggregation_method: How to aggregate component biases for
                comparison against system bias.  ``'max'`` (default)
                uses the maximum component bias (conservative estimate).
                ``'sum'`` uses the sum of all component biases, which
                models cumulative bias accumulation across components.

        Returns:
            CompositionalityResult with classification and metrics.

        Raises:
            ValueError: If component_biases is empty, aggregation_method is
                unsupported, or threshold is not a finite tolerance > 0.
        """
        logger.info(
            "analyze: %d components, system_bias=%.4f, threshold=%.3f, method=%s",
            len(component_biases),
            system_bias,
            threshold,
            aggregation_method,
        )
        if not component_biases:
            raise ValueError("At least one component bias score is required.")
        if aggregation_method not in ("max", "sum"):
            raise ValueError(
                f"Unsupported aggregation_method '{aggregation_method}'. Must be 'max' or 'sum'."
            )
        # BGL-5 (2026-09-27). The guard below covers the component scores and
        # the system bias and NOT the tolerance they are compared against, so
        # the mechanism its own comment describes ("every `>` and `<` against
        # NaN is False, so control fell through to the final `else` and reported
        # scenario 'consistent'") still arrived, through `threshold` instead of
        # through the data. Measured that day, verbatim:
        #   analyze({'a': 0.0, 'b': 0.0}, 0.9, threshold=nan)
        #     -> scenario 'consistent', divergence 0.9, NO warning, where the
        #        same call at the default threshold returns 'novel_emergence'
        #   analyze({'a': 0.0, 'b': 0.0}, 1e9, threshold=inf)
        #     -> scenario 'consistent' on a system bias of 1e9, and
        #        'amplification', 'reduction' and 'novel_emergence' unreachable
        #        for ANY data, so the verdict could not have disagreed
        #   analyze({'a': 0.05, 'b': 0.05}, 0.05, threshold=-5.0)
        #     -> scenario 'amplification' on a divergence of EXACTLY 0.0, which
        #        is the definition of consistent
        # A threshold of 0 is refused for the same vacuity reason: `abs(v) < 0`
        # is False for every component, so 'novel_emergence' could never fire.
        # Now each of the four raises ValueError, three lines below the check
        # that already rejects an unsupported aggregation_method and in the same
        # spirit as NegotiationFairnessTracker(alpha=nan). A measured tolerance
        # is untouched: the same inputs at the default 0.05 still classify
        # 'novel_emergence', 'amplification' (divergence 0.1) and 'consistent'.
        if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
            raise ValueError(
                f"threshold must be a finite tolerance greater than 0, got "
                f"{threshold!r}. Every `>` and `<` comparison against a non-finite "
                f"threshold is False, which reports scenario 'consistent' (the system "
                f"behaves in line with its parts) for a divergence nobody compared; a "
                f"negative threshold reports 'amplification' for a divergence of 0; "
                f"and 0 makes 'novel_emergence' unreachable for any data."
            )

        # An unmeasurable component or system bias makes `divergence` NaN, and
        # every `>` and `<` against NaN is False, so control fell through to the
        # final `else` and reported scenario "consistent": the system behaves in
        # line with its parts. Measured 2026-09-08 with a single NaN component
        # against system_bias=0.9, and again with a NaN system bias: both
        # returned scenario="consistent" with divergence=nan sitting in the same
        # result object.
        unmeasurable = [k for k, v in component_biases.items() if not math.isfinite(float(v))]
        if unmeasurable or not math.isfinite(float(system_bias)):
            what = (
                f"{len(unmeasurable)} component(s) ({', '.join(sorted(unmeasurable))})"
                if unmeasurable
                else "the system bias"
            )
            warnings.warn(
                f"CompositionalityAnalyzer.analyze: {what} could not be measured, so "
                f"system and component bias cannot be compared. Reporting scenario="
                f"'not_assessed', NOT 'consistent'.",
                UserWarning,
                stacklevel=2,
            )
            return CompositionalityResult(
                component_scores=dict(component_biases),
                system_score=system_bias,
                divergence=float("nan"),
                scenario="not_assessed",
                metadata=RunMetadata(
                    parameters={
                        "threshold": threshold,
                        "aggregation_method": aggregation_method,
                        "n_components": len(component_biases),
                        "n_components_unmeasurable": len(unmeasurable),
                    },
                ),
            )

        abs_values = [abs(v) for v in component_biases.values()]

        # Aggregate component biases according to the chosen method.
        if aggregation_method == "sum":
            aggregate = sum(abs_values)
        else:
            aggregate = max(abs_values)

        abs_system = abs(system_bias)
        divergence = abs_system - aggregate

        # Classify the scenario.
        all_components_low = all(abs(v) < threshold for v in component_biases.values())

        scenario: Literal[
            "amplification", "reduction", "novel_emergence", "consistent", "not_assessed"
        ]
        if all_components_low and abs_system >= threshold:
            scenario = "novel_emergence"
        elif divergence > threshold:
            scenario = "amplification"
        elif divergence < -threshold:
            scenario = "reduction"
        else:
            scenario = "consistent"

        logger.info("analyze complete: scenario=%s, divergence=%.4f", scenario, divergence)
        return CompositionalityResult(
            component_scores=dict(component_biases),
            system_score=system_bias,
            divergence=float(divergence),
            scenario=scenario,
            metadata=RunMetadata(
                parameters={
                    "threshold": threshold,
                    "aggregation_method": aggregation_method,
                    "n_components": len(component_biases),
                },
            ),
        )
