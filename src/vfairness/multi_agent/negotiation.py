"""
Turn-by-turn negotiation fairness tracking in multi-agent dialogues.

Many multi-agent systems engage in multi-turn deliberation, debate, or
negotiation (e.g. one agent acts on behalf of a user against a
counterparty agent). Whole-dialogue fairness metrics can hide *drift*:
the per-turn disparity widens or narrows systematically as the
negotiation proceeds, which a single end-of-dialogue measurement cannot
distinguish from a stable but non-zero gap.

Methodology grounding:
- Davidson et al. (2024) "MultiAgentBench: A Comprehensive Benchmark
  for LLM-based Multi-Agent Systems"
- Bianchi et al. (2024) "Cooperation, Competition, and Maliciousness:
  LLM-Stakeholders Interactive Negotiation"

Statistical test:
- Per-turn group-conditional disparity series.
- Mann-Kendall trend test (non-parametric, monotonic-trend
  identification) on the disparity series. Significant positive tau
  with widening disparity = drift away from parity over turns.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import detectability
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


def _min_attainable_kendall_p(turns: np.ndarray, disparity: np.ndarray) -> Optional[float]:
    """Smallest p Kendall's tau can return for THIS disparity series.

    The real test, run on the most extreme arrangement of the same values:
    sorted ascending against the same turn indices. Using the observed values
    rather than the turn count alone preserves the tie structure, and so the
    method scipy resolves to, which is what sets the floor.

    A CONSTANT series is the one case where the observed values cannot answer
    the question: no rearrangement of them creates a trend, so the floor comes
    out undefined. That is not a design failure, it is the shape this module
    already rules on deliberately a few lines below ("a genuinely CONSTANT
    disparity series ... really is no trend"), and its verdict rests on the RUN
    LENGTH: at 5 turns a doubling gap would have been caught, at 3 it would
    not, so the same is true of "stable". The fallback therefore asks the
    design-level question on the same turn indices, against a strictly
    increasing surrogate. Returns ``None`` only when even that is not
    computable, which is could-not-check and never a number.
    """

    def _p(values) -> Optional[float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = stats.kendalltau(turns, values)
            p = float(p)
        except Exception:  # pragma: no cover - defensive
            return None
        return p if np.isfinite(p) else None

    values = np.asarray(disparity, dtype=float)
    observed_floor = _p(np.sort(values))
    if observed_floor is not None:
        return observed_floor
    return _p(np.arange(len(values), dtype=float))


@dataclass
class NegotiationResult(SerializableMixin):
    """Result of a turn-by-turn negotiation-fairness analysis.

    Attributes:
        per_turn_disparity: Disparity (group_a_mean - group_b_mean) for
            each turn, in order.
        mean_disparity: Mean of `per_turn_disparity`.
        max_disparity: Largest absolute per-turn disparity.
        final_minus_initial: Disparity at the last turn minus disparity
            at the first turn: a simple drift summary.
        mann_kendall_tau: Kendall's tau on the disparity series.
        p_value: Two-sided p-value for the Mann-Kendall trend test.
        trend: One of "widening", "narrowing", "stable" based on the
            absolute magnitude of disparity at the first vs last turn,
            gated on a significant Mann-Kendall trend. Sign-agnostic so
            that a gap can be flagged regardless of which group is
            disadvantaged at the start. A fourth value, "not_assessed",
            means too few turns were measurable to test a trend at all.
            It is NOT a synonym for "stable".
        is_significant: Convenience flag for `p_value < alpha`, or None
            when no test was run.
        is_widening: True iff `trend == "widening"` (already implies
            significance and a substantive Kendall's tau). None when the
            question was not answered.
        n_turns_measured: Turns whose disparity was finite.
        n_turns_unmeasurable: Turns dropped because one side was not
            finite. These carry no information about the trend either way.
        trend_detectable: Whether the Mann-Kendall test on this many turns
            could reach ``alpha`` for ANY disparity series. False means the
            run is too short for a verdict to carry information; None means
            that question could not be answered.
        min_attainable_p: The smallest p this series could produce.
        detectability_note: What a "stable" reading would NOT have meant.
            Empty when the design has power.
    """

    per_turn_disparity: List[float]
    mean_disparity: float
    max_disparity: float
    final_minus_initial: float
    mann_kendall_tau: float
    p_value: float
    trend: str
    is_significant: Optional[bool]
    is_widening: Optional[bool]
    metadata: RunMetadata = field(default_factory=RunMetadata)
    n_turns_measured: int = 0
    n_turns_unmeasurable: int = 0
    trend_detectable: Optional[bool] = None
    min_attainable_p: Optional[float] = None
    detectability_note: str = ""


class NegotiationFairnessTracker:
    """Per-turn fairness tracker for multi-turn agent dialogues.

    Args:
        alpha: Significance threshold for the Mann-Kendall trend test.
        widening_tau_threshold: Minimum |tau| at which an effect is
            considered substantive enough to label as widening /
            narrowing (rather than "stable") when also significant.

    Example:
        >>> tracker = NegotiationFairnessTracker()
        >>> # Per-turn group-conditional outcomes (e.g., offered price).
        >>> turns_group_a = [100, 95, 90, 85, 80]
        >>> turns_group_b = [100, 98, 96, 94, 92]
        >>> result = tracker.analyze(turns_group_a, turns_group_b)
        >>> result.is_widening

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: negotiation_fairness_tracker. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        widening_tau_threshold: float = 0.2,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not 0 < widening_tau_threshold < 1:
            raise ValueError(
                f"widening_tau_threshold must be in (0, 1), got {widening_tau_threshold}."
            )
        self.alpha = alpha
        self.widening_tau_threshold = widening_tau_threshold

    def analyze(
        self,
        group_a_outcomes_per_turn: Sequence[float],
        group_b_outcomes_per_turn: Sequence[float],
    ) -> NegotiationResult:
        """Compute the per-turn disparity series and its trend.

        Each input is the group-conditional outcome at each turn of a
        single negotiation (or the per-turn average across many
        negotiations between the same two roles). The two sequences must
        have the same length and at least 3 turns for Kendall's tau to
        be defined.

        Args:
            group_a_outcomes_per_turn: Per-turn outcome for group A.
            group_b_outcomes_per_turn: Per-turn outcome for group B.

        Returns:
            NegotiationResult.

        Raises:
            ValueError: If the inputs have different lengths or fewer
                than 3 turns.
        """
        a = np.asarray(group_a_outcomes_per_turn, dtype=float)
        b = np.asarray(group_b_outcomes_per_turn, dtype=float)

        if len(a) != len(b):
            raise ValueError(
                f"group_a ({len(a)}) and group_b ({len(b)}) must have the same number of turns."
            )
        if len(a) < 3:
            raise ValueError("At least 3 turns are required for the Mann-Kendall trend test.")

        disparity = (a - b).tolist()
        disparity_arr = np.asarray(disparity)

        # A turn whose disparity is not finite was not measured, and it used to
        # be handled by `tau = 0.0 if isnan(tau)` and `p_value = 1.0 if
        # isnan(p_value)` far below, which is the neutral-default form of the
        # defect: kendalltau returns NaN as soon as ONE element is NaN, and 0.0
        # with p=1.0 is exactly the signature of "no trend". Measured
        # 2026-09-08 on the strictly widening series [0.1 .. 0.6] against a flat
        # zero: tau=1.0, p=0.0028, trend="widening". Blank ONE turn of that same
        # series and it reported tau=0.0, p=1.0, trend="stable",
        # is_widening=False, with no warning. A gap that doubled every turn read
        # as a gap under control.
        measurable = np.isfinite(disparity_arr)
        n_measured = int(measurable.sum())
        n_unmeasurable = int(len(disparity_arr) - n_measured)

        if n_measured < 3:
            warnings.warn(
                f"NegotiationFairnessTracker.analyze: only {n_measured} of "
                f"{len(disparity_arr)} turns had a finite disparity, and Kendall's tau "
                f"needs at least 3. Reporting trend='not_assessed' and is_widening=None "
                f"(could not check), NOT 'stable'.",
                UserWarning,
                stacklevel=2,
            )
            return NegotiationResult(
                per_turn_disparity=[float(x) for x in disparity],
                mean_disparity=float("nan"),
                max_disparity=float("nan"),
                final_minus_initial=float("nan"),
                mann_kendall_tau=float("nan"),
                p_value=float("nan"),
                trend="not_assessed",
                is_significant=None,
                is_widening=None,
                n_turns_measured=n_measured,
                n_turns_unmeasurable=n_unmeasurable,
                metadata=RunMetadata(
                    parameters={
                        "alpha": self.alpha,
                        "widening_tau_threshold": self.widening_tau_threshold,
                        "n_turns": len(disparity),
                        "n_turns_measured": n_measured,
                    },
                ),
            )

        if n_unmeasurable:
            warnings.warn(
                f"NegotiationFairnessTracker.analyze: {n_unmeasurable} of "
                f"{len(disparity_arr)} turns had a non-finite disparity and were "
                f"excluded; the trend rests on the {n_measured} that remain.",
                UserWarning,
                stacklevel=2,
            )

        # Mann-Kendall (via Kendall's tau between turn-index and disparity).
        # Indices are the ORIGINAL turn numbers, so dropping a turn does not
        # pretend the negotiation was shorter than it was.
        turns = np.arange(len(disparity_arr))[measurable]
        disparity_arr = disparity_arr[measurable]
        tau, p_value = stats.kendalltau(turns, disparity_arr)
        # Still NaN here means a genuinely CONSTANT disparity series, where tau
        # is undefined because there is no variance to rank. That case really is
        # "no trend", so the neutral default is the right answer for it, and now
        # only for it.
        tau = 0.0 if np.isnan(tau) else float(tau)
        p_value = 1.0 if np.isnan(p_value) else float(p_value)

        # DISCRETE FLOOR (readiness 6, 2026-09-10). The branch above covers a
        # design that could not RUN; this is the different problem of a design
        # that runs and cannot FIRE. Kendall's tau on 3 measurable turns has a
        # p-value floor of 2/3! = 0.3333, and on 4 turns 0.0833, both above the
        # default alpha of 0.05. Measured 2026-09-10 on a gap that DOUBLES every
        # turn ([0.1, 0.2, 0.4] and [0.1, 0.2, 0.4, 0.8]) against a flat zero:
        # tau=1.0, the most extreme monotone increase there is, reported
        # trend='stable' and is_widening=False at both lengths, in silence. At 5
        # turns the floor is 0.0167 and the identical design reads 'widening',
        # so blanking ONE turn of a 5-turn run moved a detectable design into an
        # undetectable one and the only warning said the trend "rests on the 4
        # that remain".
        min_attainable = _min_attainable_kendall_p(turns, disparity_arr)
        trend_detectable, note = detectability(min_attainable, n_family=1, alpha=self.alpha)
        if trend_detectable is not True:
            warnings.warn(
                f"NegotiationFairnessTracker.analyze: the trend over "
                f"{n_measured} measurable turn(s) was NOT ASSESSED. {note} Reporting "
                f"trend='not_assessed' and is_widening=None (could not check), NOT "
                f"'stable'. The measured series still stands: Kendall's tau = {tau:.3f}, "
                f"p = {p_value:.4g}.",
                UserWarning,
                stacklevel=2,
            )
            return NegotiationResult(
                per_turn_disparity=[float(x) for x in disparity],
                mean_disparity=float(np.mean(disparity_arr)),
                max_disparity=float(np.max(np.abs(disparity_arr))),
                final_minus_initial=float(disparity_arr[-1] - disparity_arr[0]),
                mann_kendall_tau=tau,
                p_value=p_value,
                trend="not_assessed",
                is_significant=None,
                is_widening=None,
                n_turns_measured=n_measured,
                n_turns_unmeasurable=n_unmeasurable,
                metadata=RunMetadata(
                    parameters={
                        "alpha": self.alpha,
                        "widening_tau_threshold": self.widening_tau_threshold,
                        "n_turns": len(disparity),
                        "n_turns_measured": n_measured,
                        "min_attainable_p": min_attainable,
                        "trend_detectable": trend_detectable,
                    },
                ),
                trend_detectable=trend_detectable,
                min_attainable_p=min_attainable,
                detectability_note=note,
            )

        is_significant = p_value < self.alpha
        substantive = abs(tau) >= self.widening_tau_threshold
        # Trend is judged by absolute disparity (sign-agnostic): is the
        # gap growing or shrinking? The sign of tau only tells us which
        # direction the *signed* disparity moves; we want the magnitude.
        abs_first = abs(float(disparity_arr[0]))
        abs_last = abs(float(disparity_arr[-1]))
        if is_significant and substantive and abs_last > abs_first:
            trend = "widening"
        elif is_significant and substantive and abs_last < abs_first:
            trend = "narrowing"
        else:
            trend = "stable"

        is_widening: Optional[bool] = trend == "widening"
        mean_disparity = float(np.mean(disparity_arr))
        max_disparity = float(np.max(np.abs(disparity_arr)))
        drift = float(disparity_arr[-1] - disparity_arr[0])

        logger.info(
            "analyze complete: trend=%s, tau=%.4f, p_value=%.4f",
            trend,
            tau,
            p_value,
        )
        return NegotiationResult(
            per_turn_disparity=[float(x) for x in disparity],
            mean_disparity=mean_disparity,
            max_disparity=max_disparity,
            final_minus_initial=drift,
            mann_kendall_tau=tau,
            p_value=p_value,
            trend=trend,
            is_significant=is_significant,
            is_widening=is_widening,
            n_turns_measured=n_measured,
            n_turns_unmeasurable=n_unmeasurable,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "widening_tau_threshold": self.widening_tau_threshold,
                    "n_turns": len(disparity),
                    "n_turns_measured": n_measured,
                    "min_attainable_p": min_attainable,
                    "trend_detectable": trend_detectable,
                },
            ),
            trend_detectable=trend_detectable,
            min_attainable_p=min_attainable,
            detectability_note=note,
        )
