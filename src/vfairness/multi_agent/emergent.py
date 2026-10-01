"""
Emergent bias detection in multi-agent systems.

Detects bias that emerges from the interaction of multiple agents
but is not present (or is less severe) in any individual component.
This is analogous to emergent properties in complex systems.
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class EmergentBiasResult(SerializableMixin):
    """Result of an emergent bias analysis.

    Attributes:
        system_bias: Measured system-level bias.
        max_component_bias: Maximum bias among individual components.
        amplification_factor: Ratio of system bias to max component bias.
        is_emergent: Whether the system bias significantly exceeds
            component biases, indicating emergent behavior. None means no
            comparison was possible, NOT that no emergence was found.
        n_components_measured: Components whose bias came out finite.
        n_components_unmeasurable: Components excluded from the maximum.
        n_bootstrap_unmeasurable: Resamples that lost a group entirely and
            so say nothing about the system bias.
        n_samples_unmeasurable: Samples whose system output was not finite.
            They carry no system measurement, so they are excluded from every
            series rather than counted as agreement.
    """

    system_bias: float
    max_component_bias: float
    amplification_factor: float
    is_emergent: Optional[bool]
    p_value: float = 1.0
    is_significant: Optional[bool] = False
    metadata: RunMetadata = field(default_factory=RunMetadata)
    n_components_measured: int = 0
    n_components_unmeasurable: int = 0
    n_bootstrap_unmeasurable: int = 0
    n_samples_unmeasurable: int = 0


class EmergentBiasDetector:
    """Detects emergent bias in multi-agent systems.

    Compares system-level bias against individual component biases
    to identify cases where agent interactions produce bias that
    exceeds what any single component exhibits.

    Example:
        >>> detector = EmergentBiasDetector()
        >>> component_outputs = {
        ...     "agent_a": np.array([0.8, 0.7, 0.9, 0.6]),
        ...     "agent_b": np.array([0.75, 0.65, 0.85, 0.55]),
        ... }
        >>> system_outputs = np.array([0.9, 0.5, 0.95, 0.3])
        >>> groups = np.array([0, 1, 0, 1])
        >>> result = detector.analyze(component_outputs, system_outputs, groups)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: emergent_bias_detector. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def analyze(
        self,
        component_outputs: dict[str, np.ndarray],
        system_outputs: np.ndarray,
        groups: np.ndarray,
    ) -> EmergentBiasResult:
        """Analyze whether system bias exceeds component biases.

        Computes per-group mean difference for both the system and each
        component, then checks whether the system exhibits amplified or
        novel bias relative to the components.

        Args:
            component_outputs: Dictionary mapping component names to
                their output arrays. Each array has one value per sample.
            system_outputs: System-level output array, one value per sample.
                Samples whose system output is not finite carry no system
                measurement: they are excluded from every series, counted on
                ``n_samples_unmeasurable`` and warned about.
            groups: Binary group membership array (0 or 1) for each sample.
                Exactly two distinct labels: a third label cannot be compared
                by this detector and is refused rather than dropped.

        Returns:
            EmergentBiasResult with amplification analysis. If the retained
            samples no longer hold two groups, ``is_emergent`` is None (could
            not check) rather than False.

        Raises:
            ValueError: If inputs have inconsistent lengths, or groups has
                fewer than 2 or more than 2 unique values.
        """
        logger.info("analyze: %d components, %d samples", len(component_outputs), len(groups))
        groups = np.asarray(groups)
        system_outputs = np.asarray(system_outputs, dtype=float)

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Groups array must contain at least 2 unique values.")
        # BGL-5 (2026-09-27). The group count was validated on ONE side only,
        # and the comparison below reads eval_unique[0] and eval_unique[1], so
        # every label after the second was silently dropped from both masks.
        # Measured that day on 180 samples in 3 groups with the WHOLE bias in
        # group 2 (its system mean 1.0 against group 0's 0.0, component gap
        # 0.40): system_bias 0.0, max_component_bias 0.01, amplification 0.0,
        # is_emergent FALSE, p_value 1.0, is_significant False,
        # n_samples_unmeasurable 0, metadata n_samples_measured 180 and NO
        # warning, while 60 samples entered neither mask. The count field
        # actively asserted that everything had been measured. The same shape
        # with NaN group labels (np.unique sorts NaN last) reported is_emergent
        # True over a claimed 200 measured samples, two of which matched no
        # mask. This detector is documented binary and has no vocabulary for a
        # third group, so it now refuses by name instead of publishing a census
        # of two thirds of the data. A 2-group run is untouched: the shipped
        # consumer (_amplification_section in operations/pulse/agent_probe.py)
        # only ever passes two labels, pairwise or one-vs-rest.
        if len(unique_groups) > 2:
            _labels = unique_groups.tolist()
            _shown = _labels[:5]
            raise ValueError(
                f"Groups array must contain exactly 2 unique values for this binary "
                f"detector, got {len(_labels)}: {_shown}"
                f"{' ...' if len(_labels) > len(_shown) else ''}. Only the first two "
                f"sorted labels would be compared and every sample in a later group "
                f"would be dropped from both group masks without being counted, which "
                f"reports system_bias=0.0 and is_emergent=False on data whose whole gap "
                f"lives in a third group. Pass one pair at a time, or one group against "
                f"the rest, and say which pair the verdict belongs to."
            )

        n_samples = len(groups)
        if len(system_outputs) != n_samples:
            raise ValueError(
                f"system_outputs length ({len(system_outputs)}) must match "
                f"groups length ({n_samples})."
            )

        # NAN-01 (2026-09-10): ONE unmeasurable system output used to decide
        # the verdict for all of them. A single NaN made system_bias NaN, and
        # `nan > 0` and `nan > 1.5` are both False, so the amplification fell
        # to the neutral 1.0 and is_emergent to False. Measured 2026-09-10 on
        # 200 samples, unbiased components and a perfectly separating system:
        # amplification 1000000.0 -> 1.0 and is_emergent True -> False, in
        # silence. A sample with no system measurement is now excluded from
        # EVERY series at once, so system, components and groups stay aligned,
        # and the exclusion is counted, warned about and reported.
        finite_mask = np.isfinite(system_outputs)
        n_samples_unmeasurable = int(n_samples - np.count_nonzero(finite_mask))
        if n_samples_unmeasurable:
            warnings.warn(
                f"EmergentBiasDetector.analyze: {n_samples_unmeasurable} of {n_samples} "
                f"system outputs are not finite and were excluded from every series; "
                f"the analysis rests on the {n_samples - n_samples_unmeasurable} samples "
                f"that carry a system measurement.",
                UserWarning,
                stacklevel=2,
            )

        # Component lengths are checked against the SUPPLIED length, then every
        # component is restricted to the same retained samples.
        component_series: dict[str, np.ndarray] = {}
        for name, outputs in component_outputs.items():
            outputs = np.asarray(outputs, dtype=float)
            if len(outputs) != n_samples:
                raise ValueError(
                    f"Component '{name}' output length ({len(outputs)}) must "
                    f"match groups length ({n_samples})."
                )
            component_series[name] = outputs[finite_mask]

        eval_groups = groups[finite_mask]
        eval_system = system_outputs[finite_mask]
        eval_unique = np.unique(eval_groups)

        # Compute system-level bias.
        if len(eval_unique) >= 2:
            mask_0 = eval_groups == eval_unique[0]
            mask_1 = eval_groups == eval_unique[1]
            system_bias = abs(
                float(np.mean(eval_system[mask_0])) - float(np.mean(eval_system[mask_1]))
            )
        else:
            empty = np.zeros(len(eval_groups), dtype=bool)
            mask_0, mask_1 = empty, empty
            system_bias = float("nan")

        if not math.isfinite(system_bias):
            warnings.warn(
                f"EmergentBiasDetector.analyze: system bias could not be measured "
                f"({len(eval_unique)} group(s) remain after excluding "
                f"{n_samples_unmeasurable} of {n_samples} non-finite system outputs), "
                f"so there is nothing to compare the components against. Reporting "
                f"is_emergent=None (could not check), not an absence of emergent bias.",
                UserWarning,
                stacklevel=2,
            )
            return EmergentBiasResult(
                system_bias=float("nan"),
                max_component_bias=float("nan"),
                amplification_factor=float("nan"),
                is_emergent=None,
                p_value=float("nan"),
                is_significant=None,
                n_components_measured=0,
                n_components_unmeasurable=len(component_series),
                n_samples_unmeasurable=n_samples_unmeasurable,
                metadata=RunMetadata(
                    parameters={
                        "n_components": len(component_outputs),
                        "n_samples": n_samples,
                        "n_samples_unmeasurable": n_samples_unmeasurable,
                        "n_bootstrap": 0,
                    },
                ),
            )

        # Compute per-component bias, on the same retained samples.
        component_biases = {
            name: abs(float(np.mean(series[mask_0])) - float(np.mean(series[mask_1])))
            for name, series in component_series.items()
        }

        # "Emergent" is a CLAIM ABOUT A COMPARISON: the system is more biased
        # than any of its parts. `max(...) if component_biases else 0.0` turned
        # an absent comparison into a maximum of zero, and zero is the strongest
        # possible evidence of emergence. Measured 2026-09-08, `analyze({})` on a
        # perfectly separating system returned amplification_factor=1e6,
        # is_emergent=True, p_value=0.0, is_significant=True, with no warning: a
        # maximal claim of emergent bias derived from nothing at all.
        #
        # A component whose own bias came out non-finite is excluded rather than
        # ranked. Python's max() is order-dependent in the presence of NaN, so
        # leaving one in makes the answer depend on dict insertion order.
        measurable_components = {
            name: bias for name, bias in component_biases.items() if math.isfinite(bias)
        }
        n_measured = len(measurable_components)
        n_unmeasurable = len(component_biases) - n_measured

        amplification_factor: float
        is_emergent: Optional[bool]
        if n_measured == 0:
            reason = (
                "no components were supplied"
                if not component_outputs
                else f"none of the {len(component_biases)} components had a finite bias"
            )
            warnings.warn(
                f"EmergentBiasDetector.analyze: {reason}, so system bias has nothing to "
                f"be compared against. Reporting is_emergent=None (could not check), "
                f"not an emergence finding.",
                UserWarning,
                stacklevel=2,
            )
            return EmergentBiasResult(
                system_bias=float(system_bias),
                max_component_bias=float("nan"),
                amplification_factor=float("nan"),
                is_emergent=None,
                p_value=float("nan"),
                is_significant=None,
                n_components_measured=0,
                n_components_unmeasurable=n_unmeasurable,
                n_samples_unmeasurable=n_samples_unmeasurable,
                metadata=RunMetadata(
                    parameters={
                        "n_components": len(component_outputs),
                        "n_samples": len(groups),
                        "n_samples_unmeasurable": n_samples_unmeasurable,
                        "n_bootstrap": 0,
                    },
                ),
            )

        if n_unmeasurable:
            warnings.warn(
                f"EmergentBiasDetector.analyze: {n_unmeasurable} of "
                f"{len(component_biases)} components had a non-finite bias and were "
                f"excluded from the maximum; the comparison rests on the {n_measured} "
                f"that remain.",
                UserWarning,
                stacklevel=2,
            )

        max_component_bias = max(measurable_components.values())

        # Amplification factor.
        #
        # OVER-REPORTING (readiness 6, 2026-09-10). A ratio with a zero
        # denominator is not a big number, it is an undefined one, and 1e6 is a
        # MEASUREMENT on the amplification scale: the shipped pulse report grades
        # severity straight off this field, so anything at all divided by an
        # exactly-zero component baseline came out "critical". Measured
        # 2026-09-10 on 200 samples, two components held at a constant 0.5 (bias
        # exactly 0) and a system gap of 0.001: amplificationFactor 1000000.0,
        # isEmergent true, severity "critical", one critical BIA finding in the
        # report, from a gap three orders of magnitude below anything anyone
        # would act on. NaN here follows this library's house rule for a ratio
        # nobody could compute (see risk_ratio / odds_ratio, H-04), and the
        # emergence verdict moves to the bootstrap comparison below, which is a
        # measurement rather than a sentinel.
        amplification_defined = max_component_bias > 0
        if amplification_defined:
            amplification_factor = system_bias / max_component_bias
            # Emergent bias: system bias exceeds max component bias by > 50%.
            is_emergent = bool(amplification_factor > 1.5)
        else:
            amplification_factor = float("nan")
            is_emergent = None  # decided from the bootstrap below
            warnings.warn(
                f"EmergentBiasDetector.analyze: every one of the {n_measured} measured "
                f"component(s) has a bias of exactly 0, so the amplification RATIO "
                f"(system {system_bias:.6g} / component 0) is undefined. Reporting "
                f"amplification_factor=nan, NOT 1e6, and deciding is_emergent from the "
                f"bootstrap comparison of the system gap against that zero baseline.",
                UserWarning,
                stacklevel=2,
            )

        # Bootstrap test: is system bias significantly greater than max component bias?
        rng = np.random.default_rng(42)
        n_bootstrap = 500
        n_boot_unmeasurable = 0
        boot_system_biases = []
        # Resampled from the RETAINED samples only: a resample that drew an
        # unmeasurable system output would carry the same NaN into the
        # percentile and the p-value, where `nan <= max_component_bias` is
        # False and so reads as evidence FOR emergence.
        n_eval = len(eval_groups)
        # A ONE-ROW GROUP HAS NO SAMPLING SPREAD TO RESAMPLE. Every resample that
        # keeps it repeats the same value, so the draws understate the variability
        # of its mean to zero and the interval comes out falsely narrow. Measured
        # 2026-10-01 (broken-data check): 399 rows against 1 gave
        # is_significant=True at p=0.0064 from one observation. Below two rows in
        # either group the bootstrap is not run and significance is could-not-check.
        _, _group_counts = np.unique(eval_groups, return_counts=True)
        smallest_group = int(_group_counts.min()) if len(_group_counts) else 0
        if smallest_group < 2:
            n_bootstrap = 0
        for _ in range(n_bootstrap):
            idx = rng.choice(n_eval, size=n_eval, replace=True)
            boot_groups = eval_groups[idx]
            boot_sys = eval_system[idx]
            boot_unique = np.unique(boot_groups)
            if len(boot_unique) < 2:
                # A resample that lost a group entirely measures NOTHING. It used
                # to be recorded as bias=0.0, and since the p-value below counts
                # draws with `bias <= max_component_bias`, every one of those
                # zeros counted as evidence AGAINST emergence. On a group so
                # small that most resamples lose it, the reported p-value was
                # simply the fraction of unmeasurable draws.
                n_boot_unmeasurable += 1
                continue
            bm0 = boot_groups == boot_unique[0]
            bm1 = boot_groups == boot_unique[1]
            boot_bias = abs(float(np.mean(boot_sys[bm0])) - float(np.mean(boot_sys[bm1])))
            boot_system_biases.append(boot_bias)

        is_significant: Optional[bool]
        if not boot_system_biases:
            if smallest_group < 2:
                why = (
                    f"the smallest group holds {smallest_group} row(s), and a group "
                    f"needs at least 2 for a resample to say anything about its spread"
                )
            else:
                why = f"not one of {n_bootstrap} bootstrap resamples retained both groups"
            warnings.warn(
                f"EmergentBiasDetector.analyze: {why}, so no significance test was run. "
                f"Reporting is_significant=None (could not check) and p_value=nan.",
                UserWarning,
                stacklevel=2,
            )
            is_significant = None
            p_value = float("nan")
        else:
            if n_boot_unmeasurable:
                warnings.warn(
                    f"EmergentBiasDetector.analyze: {n_boot_unmeasurable} of "
                    f"{n_bootstrap} bootstrap resamples lost a group and were excluded; "
                    f"the p-value rests on the {len(boot_system_biases)} that remain.",
                    UserWarning,
                    stacklevel=2,
                )
            draws = np.asarray(boot_system_biases, dtype=float)
            # BOUND-AWARE (basic / reverse-percentile) LOWER BOUND, clipped at 0.
            # System bias is |mean_a - mean_b|: a FOLDED, nonnegative statistic,
            # whose resample distribution sits strictly above 0 at the parity
            # null. A percentile lower bound therefore cannot reach 0 and the old
            # `percentile(2.5) > max_component_bias` test could only answer True
            # when the comparison value was 0, for any system gap at all,
            # including pure noise. That is the same 0-percent-coverage trap
            # `_interval_from_bootstrap` in evaluation/vfairness_metrics/
            # _statistics.py documents, and 'basic' is the construction that
            # library already prescribes for it: only a pivot can reach the
            # boundary. The p-value is the matching pivotal quantity, so the flag
            # and the number beside it now answer the same question.
            ci_lower = max(0.0, 2.0 * float(system_bias) - float(np.percentile(draws, 97.5)))
            is_significant = bool(ci_lower > max_component_bias)
            p_value = float(np.mean((2.0 * float(system_bias) - draws) <= max_component_bias))

        if not amplification_defined:
            # The ratio is undefined, so emergence is decided by the comparison
            # the ratio was standing in for: is the system gap bigger than the
            # (zero) component baseline by more than resampling noise?
            is_emergent = is_significant
            logger.info(
                "amplification ratio undefined (zero component baseline); "
                "is_emergent taken from the bootstrap comparison: %s",
                is_emergent,
            )

        logger.info(
            "analyze complete: is_emergent=%s, amplification=%.2f, p_value=%.4f",
            is_emergent,
            amplification_factor,
            p_value,
        )
        return EmergentBiasResult(
            system_bias=float(system_bias),
            max_component_bias=float(max_component_bias),
            amplification_factor=float(amplification_factor),
            is_emergent=is_emergent,
            p_value=p_value,
            is_significant=is_significant,
            n_components_measured=n_measured,
            n_components_unmeasurable=n_unmeasurable,
            n_bootstrap_unmeasurable=n_boot_unmeasurable,
            n_samples_unmeasurable=n_samples_unmeasurable,
            metadata=RunMetadata(
                parameters={
                    "n_components": len(component_outputs),
                    "n_samples": len(groups),
                    "n_samples_measured": n_eval,
                    "n_samples_unmeasurable": n_samples_unmeasurable,
                    "n_bootstrap": n_bootstrap,
                    "n_bootstrap_measured": len(boot_system_biases),
                },
            ),
        )
