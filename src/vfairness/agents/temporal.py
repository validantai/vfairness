"""
Temporal trajectory analysis for AI agent fairness.

Tracks how fairness metrics evolve over the course of a multi-turn
agent interaction. Detects bias drift (gradual increase in disparity)
and feedback loops (self-reinforcing bias patterns).
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from vfairness._not_assessed import NOT_ASSESSED, warn_not_assessed
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)

# Kendall's tau, CUSUM and EWMA all need at least three ordered observations
# before they describe anything: two points define a line, never a trend.
_MIN_TURNS_FOR_TREND = 3


@dataclass
class TrajectoryResult(SerializableMixin):
    """Fairness metric at a single turn in a multi-turn interaction.

    Attributes:
        session_id: Identifier for the interaction session.
        turn_number: Turn index (0-based).
        metric_name: Name of the fairness metric computed.
        value: Metric value at this turn.
        cumulative_drift: Total variation up to this turn, i.e. the sum of
            absolute turn-to-turn changes since the first recorded turn
            (not the net change from the initial value).
    """

    session_id: str
    turn_number: int
    metric_name: str
    value: float
    cumulative_drift: float
    metadata: RunMetadata = field(default_factory=RunMetadata)


class TemporalTracker:
    """Tracks fairness metric trajectories over multi-turn interactions.

    Records per-turn outcomes for two demographic groups and computes
    how the fairness metric evolves, enabling detection of bias drift
    and feedback loops.

    Args:
        metric_fn: Optional custom metric function ``f(outcomes_a, outcomes_b) -> float``.
            Defaults to mean difference if not provided.

    Example:
        >>> tracker = TemporalTracker()
        >>> tracker.record_turn(0, [0.8, 0.7], [0.6, 0.5])
        >>> tracker.record_turn(1, [0.9, 0.8], [0.5, 0.4])
        >>> trajectory = tracker.compute_trajectory()
        >>> tracker.detect_drift(threshold=0.1)

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

    Ledger row: temporal_tracker. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, metric_fn: Optional[Callable] = None) -> None:
        self.metric_fn = metric_fn or self._default_metric
        self._turns: Dict[int, dict] = {}
        self._session_id = "session_default"
        logger.info("TemporalTracker initialized")

    def record_turn(
        self,
        turn: int,
        outcomes_a: Union[List[float], np.ndarray],
        outcomes_b: Union[List[float], np.ndarray],
    ) -> None:
        """Record per-turn outcomes for both demographic groups.

        Args:
            turn: Turn number (0-based).
            outcomes_a: Outcomes for demographic group A at this turn.
            outcomes_b: Outcomes for demographic group B at this turn.
        """
        if turn < 0:
            raise ValueError(f"turn must be >= 0, got {turn}")
        arr_a = np.asarray(outcomes_a, dtype=float)
        arr_b = np.asarray(outcomes_b, dtype=float)
        if len(arr_a) == 0 or len(arr_b) == 0:
            raise ValueError("outcomes_a and outcomes_b must be non-empty.")

        logger.debug("record_turn: turn=%d", turn)
        self._turns[turn] = {
            "outcomes_a": arr_a,
            "outcomes_b": arr_b,
        }

    def compute_trajectory(self) -> List[TrajectoryResult]:
        """Compute the fairness metric trajectory across all recorded turns.

        Returns:
            List of TrajectoryResult sorted by turn number. A turn whose metric
            could not be computed keeps its place with ``value`` nan, and
            ``cumulative_drift`` is nan from that turn on, because a running
            total with a hole in it is not a total. How many such turns there
            are is stated in a warning.
        """
        logger.info("compute_trajectory: %d turns recorded", len(self._turns))
        if not self._turns:
            return []

        sorted_turns = sorted(self._turns.keys())
        results = []
        initial_value = None
        cumulative_drift = 0.0

        for turn in sorted_turns:
            data = self._turns[turn]
            value = self.metric_fn(data["outcomes_a"], data["outcomes_b"])

            if initial_value is None:
                initial_value = value
                cumulative_drift = 0.0
            else:
                cumulative_drift += abs(value - results[-1].value)

            results.append(
                TrajectoryResult(
                    session_id=self._session_id,
                    turn_number=turn,
                    metric_name="disparity",
                    value=float(value),
                    cumulative_drift=float(cumulative_drift),
                )
            )

        # BGL-3 (2026-09-27): the VALUES here were already honest, a nan value
        # and a nan cumulative_drift from that turn on, but nothing said how much
        # of the interaction was unmeasurable, while every detector on this class
        # states its counts for the same input. Measured that day on a six-turn
        # ramp with turn 2 blanked: values [0.0, 0.1, nan, 0.3, 0.4, 0.5] and
        # cumulative_drift [0.0, 0.1, nan, nan, nan, nan] with ZERO warnings,
        # where detect_feedback_loop on the same tracker said "1 of 6 recorded
        # turns had a non-finite disparity and were excluded". The counts are the
        # part a reader cannot reconstruct from a single field.
        n_unmeasurable = sum(1 for r in results if not np.isfinite(r.value))
        if n_unmeasurable:
            warnings.warn(
                f"TemporalTracker.compute_trajectory: {n_unmeasurable} of "
                f"{len(results)} recorded turns have a metric value that could not "
                f"be computed, so their value is nan and every later "
                f"cumulative_drift is nan, NOT 0.0, which on this scale would read "
                f"as a turn that did not move. The trajectory is returned in full; "
                f"the unmeasurable turns are not a finding of no drift.",
                UserWarning,
                stacklevel=2,
            )

        return results

    def detect_drift(self, threshold: float = 0.1) -> Optional[bool]:
        """Detect if the fairness trajectory drifts beyond a threshold.

        Drift is defined as the absolute difference between the initial
        and final metric values exceeding the threshold.

        Args:
            threshold: Maximum acceptable drift magnitude. A magnitude, so it
                must be finite and >= 0; anything else is refused rather than
                compared against (see Returns).

        Returns:
            True if drift exceeds the threshold, False if it does not, and
            None when fewer than two turns carry a finite disparity, so there
            is nothing to measure drift between, or when ``threshold`` itself
            is not a drift magnitude the measured drift can be compared against
            (it must be finite and >= 0). None is NOT a synonym for False. The
            endpoints compared are the first and last MEASURABLE turns; when
            any recorded turn was excluded the caller is warned and told how
            many.
        """
        # BGL-5 (2026-09-27). The BGL-S2 fix below guarded the DATA side of
        # `abs(final - initial) > threshold` and left the THRESHOLD side
        # unguarded, so the very mechanism its comment describes still arrived,
        # through the caller's argument instead of through the trajectory.
        # Measured that day on six recorded turns whose measurable disparities
        # run 0.0 to 0.5: detect_drift(0.1) is True with no warning,
        # detect_drift(nan) returned False with ZERO warnings and
        # detect_drift(inf) the same, which publishes the verdict "no drift
        # exceeds the threshold" from a comparison that was never made; and
        # detect_drift(-1.0) returned True for that ramp AND for a flat
        # trajectory, because `abs(...) > a negative number` cannot be False,
        # so the check could not have disagreed. Now each of the three returns
        # None with a warning naming the threshold, while 0.1 on that ramp
        # still returns True and 0.3 on a flat tracker still returns False.
        # CorrespondenceTester.four_fifths_rule guards its caller-supplied
        # rates in the same place and for the same reason.
        if not np.isfinite(threshold) or float(threshold) < 0.0:
            warnings.warn(
                f"TemporalTracker.detect_drift: threshold={threshold!r} is not a drift "
                f"magnitude the measured drift can be compared against (it must be "
                f"finite and >= 0), so no comparison was made. Reporting None (could "
                f"not check), NOT False, which is the verdict 'no drift exceeds the "
                f"threshold', and NOT True, which every trajectory clears when the "
                f"threshold is negative.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # BGL-S2 (2026-09-16): this was the ONLY detector on this class that
        # read the trajectory RAW instead of through _finite_series(). It took
        # trajectory[0].value and trajectory[-1].value directly, and
        # abs(nan - 0.0) is nan while `nan > threshold` is False for Python
        # floats, so an unmeasurable ENDPOINT returned the VERDICT False, "no
        # drift beyond the threshold", with no numpy invalid-value warning to
        # give it away. Measured 2026-09-16: eight turns of a strictly
        # widening disparity (0.0 -> 0.7) followed by ONE all-NaN turn ->
        # detect_drift(0.1) == False and zero warnings, while on the same
        # tracker detect_feedback_loop reported has_feedback_loop=True,
        # tau=0.99999, p=4.96e-05, n_turns_unmeasurable=1 WITH a warning, and
        # detect_drift_cusum / detect_drift_ewma likewise. With every turn
        # unmeasurable it still said False while all three siblings said None.
        # Routing through _finite_series() also fixes the subtler case where
        # the true first or last turn is dropped, which an endpoint comparison
        # would otherwise mislabel.
        _turns, values, n_recorded = self._finite_series()
        n_measured = int(values.size)
        n_unmeasurable = n_recorded - n_measured

        if n_measured < 2:
            # False here is "no drift exceeds the threshold", a verdict. Drift is
            # a change BETWEEN observations and one point has none to measure.
            warn_not_assessed(
                "TemporalTracker.detect_drift",
                measured=n_measured,
                total=n_recorded,
                unit="recorded turns had a finite disparity",
                requirement="drift is a change between observations and needs at least 2",
                reporting="None",
                instead_of="False, the verdict 'no drift exceeds the threshold'",
            )
            return None

        if n_unmeasurable:
            warnings.warn(
                f"TemporalTracker.detect_drift: {n_unmeasurable} of {n_recorded} "
                f"recorded turns had a non-finite disparity and were excluded; the "
                f"verdict compares the first and last MEASURABLE turns and rests on "
                f"the {n_measured} that remain.",
                UserWarning,
                stacklevel=2,
            )

        initial = float(values[0])
        final = float(values[-1])
        return bool(abs(final - initial) > threshold)

    def _finite_series(self) -> tuple:
        """The recorded turns whose metric value is finite, and how many were not.

        Every detector below reads the trajectory through this, so a turn whose
        metric could not be computed (an empty group after filtering, a metric_fn
        that returned NaN) is EXCLUDED and COUNTED rather than fed to a test that
        answers NaN and then reads as "no trend". Turn numbers, not array indices,
        are returned as the x-axis: they are the ordered positions the caller
        recorded, so dropping a turn does not pretend the interaction was shorter
        than it was. Kendall's tau is rank-based, so a monotone x-axis of turn
        numbers gives the same tau as consecutive indices when nothing is dropped.

        Returns:
            (turn_numbers, values, n_recorded) where turn_numbers and values
            cover only the finite turns and n_recorded is every recorded turn.
        """
        sorted_turns = sorted(self._turns.keys())
        values = np.array(
            [
                float(self.metric_fn(self._turns[t]["outcomes_a"], self._turns[t]["outcomes_b"]))
                for t in sorted_turns
            ],
            dtype=float,
        )
        finite = np.isfinite(values) if values.size else np.zeros(0, dtype=bool)
        kept = np.asarray(sorted_turns, dtype=float)[finite] if values.size else np.zeros(0)
        return kept, values[finite], len(sorted_turns)

    def detect_feedback_loop(self, alpha: float = 0.05) -> dict:
        """Detect self-reinforcing bias patterns (feedback loops).

        A feedback loop is indicated by a monotonically increasing
        absolute disparity across turns. Uses the Mann-Kendall trend
        test for monotonic trend detection.

        Args:
            alpha: Significance level the trend test's p-value is compared
                against. A probability, so it must be finite and strictly
                between 0 and 1; anything else is refused rather than compared
                against (see Returns).

        Returns:
            Dictionary with:
                - ``has_feedback_loop``: ``True`` / ``False`` when the trend
                  test ran, and ``None`` when it could not (fewer than 3 turns
                  with a finite disparity, or an ``alpha`` the p-value cannot
                  be compared against). ``None`` is NOT a synonym for
                  ``False``.
                - ``trend_direction``: 'increasing', 'decreasing', 'stable', or
                  'not_assessed' when the test could not run.
                - ``trend_strength``: Kendall's tau correlation coefficient,
                  ``nan`` when not assessed.
                - ``p_value``: P-value from the trend test, ``nan`` when not
                  assessed.
                - ``n_turns_measured`` / ``n_turns_unmeasurable``: how much of
                  the recorded interaction the verdict rests on.
        """
        turns, values, n_recorded = self._finite_series()
        abs_values = np.abs(values)
        n_measured = int(abs_values.size)
        n_unmeasurable = n_recorded - n_measured

        # BGL-W4 (2026-09-30). The third instance of the operand defect fixed in
        # detect_drift (threshold) and in the two drift detectors below
        # (sigma_limit / span, threshold / drift_limit), on the same class, in
        # the same measurement. Measured that day on eight turns ramping 0.0 ->
        # 0.84, whose default call reports has_feedback_loop=True,
        # trend_direction='increasing', p=4.96e-05:
        #   alpha=nan -> has_feedback_loop False with trend_direction 'stable'
        #     and that same p of 4.96e-05 printed beside it, and ZERO warnings:
        #     `p_value < nan` is False, so the verdict "no monotone trend" came
        #     from a comparison that was never made, on a trajectory that ramps
        #     the whole way from parity to a 0.84 gap.
        #   alpha=0.0 and alpha=-1.0 -> the same 'stable' False, because no
        #     p-value can fall below them; the check could not have disagreed.
        #   alpha=1.0, alpha=2.0 and alpha=inf -> 'increasing' for ANY positive
        #     tau, so every upward wobble becomes a feedback loop; the check
        #     could not have disagreed the other way.
        # Above the dispatch, so the too-few-turns branch cannot answer for a
        # caller who supplied an alpha that decides nothing.
        if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
            warnings.warn(
                f"TemporalTracker.detect_feedback_loop: alpha={alpha!r} is not a "
                f"significance level the trend test's p-value can be compared against "
                f"(it must be finite and strictly between 0 and 1), so no comparison was "
                f"made. Reporting has_feedback_loop=None (could not check) with "
                f"trend_direction='{NOT_ASSESSED}' and a nan tau and p-value, NOT False "
                f"with trend_direction='stable', which is the verdict 'no monotone trend', "
                f"and NOT True, which every positive tau reaches when alpha is >= 1.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "has_feedback_loop": None,
                "trend_direction": NOT_ASSESSED,
                "trend_strength": float("nan"),
                "p_value": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        # Two turns of TOTAL disparity used to answer has_feedback_loop=False,
        # trend_direction='stable', trend_strength=0.0, p_value=1.0, with no
        # warning: the signature of a trend test that ran and found nothing,
        # from a trend test that never ran. Same for a single blanked turn in an
        # otherwise strictly widening ramp: kendalltau answers NaN as soon as one
        # element is NaN, `p_value < alpha` is False for NaN, and the ramp read
        # as 'stable'. Measured 2026-09-10 on [0.0, 0.1, nan, 0.3, 0.4] against a
        # flat zero: tau=nan, p=nan, direction='stable'. Both are could-not-check.
        if n_measured < _MIN_TURNS_FOR_TREND:
            warn_not_assessed(
                "TemporalTracker.detect_feedback_loop",
                measured=n_measured,
                total=n_recorded,
                unit="recorded turns had a finite disparity",
                requirement=f"Kendall's tau needs at least {_MIN_TURNS_FOR_TREND}",
                reporting=(
                    f"has_feedback_loop=None and trend_direction='{NOT_ASSESSED}' "
                    f"with a nan tau and p-value"
                ),
                instead_of="False with trend_direction='stable', tau 0.0 and p 1.0",
            )
            return {
                "has_feedback_loop": None,
                "trend_direction": NOT_ASSESSED,
                "trend_strength": float("nan"),
                "p_value": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        if n_unmeasurable:
            warnings.warn(
                f"TemporalTracker.detect_feedback_loop: {n_unmeasurable} of "
                f"{n_recorded} recorded turns had a non-finite disparity and were "
                f"excluded; the trend rests on the {n_measured} that remain.",
                UserWarning,
                stacklevel=2,
            )

        # Kendall's tau for monotonic trend.
        tau, p_value = stats.kendalltau(turns, abs_values)
        # Still NaN here means a genuinely CONSTANT series, where tau is
        # undefined because there is no variance to rank. That case really is
        # "no trend", so the neutral answer is correct for it, and now only for it.
        tau = 0.0 if np.isnan(tau) else float(tau)
        p_value = 1.0 if np.isnan(p_value) else float(p_value)

        if p_value < alpha and tau > 0:
            direction = "increasing"
            has_loop = True
        elif p_value < alpha and tau < 0:
            direction = "decreasing"
            has_loop = False
        else:
            direction = "stable"
            has_loop = False

        return {
            "has_feedback_loop": has_loop,
            "trend_direction": direction,
            "trend_strength": float(tau),
            "p_value": float(p_value),
            "n_turns_measured": n_measured,
            "n_turns_unmeasurable": n_unmeasurable,
        }

    def detect_drift_cusum(self, threshold: float = 0.5, drift_limit: float = 5.0) -> dict:
        """Cumulative Sum (CUSUM) change-point detection.

        Detects sustained shifts in fairness metrics, even small ones.
        More sensitive than point-threshold detection for gradual drift.

        Each deviation from the trajectory mean is standardized by the
        trajectory standard deviation, so ``threshold`` (slack k) and
        ``drift_limit`` (decision interval h) are in sigma units; the defaults
        k = 0.5, h = 5.0 are the canonical tabular-CUSUM values (Montgomery
        2013). Before this standardization the deviations were in raw metric
        units, where a disparity in [0, 1] deviates from the pooled mean by at
        most ~0.5, so the increment never exceeded the 0.5 slack and the
        detector could not flag any drift at the default parameters. A flat
        trajectory (sigma ~ 0) has no variation and reports no drift.

        Args:
            threshold: Allowable slack k (tolerance for noise), in units of the
                trajectory standard deviation. Default 0.5. A tolerance, so it
                must be finite and >= 0; anything else is refused rather than
                accumulated (see Returns).
            drift_limit: CUSUM statistic threshold h for signaling drift, in
                sigma units. Default 5.0. A decision interval, so it must be
                finite and > 0; anything else is refused rather than compared
                against (see Returns).

        Returns:
            dict with keys: has_drift (``True`` / ``False``, or ``None`` when
            fewer than 3 recorded turns had a finite metric value, so the
            statistic never ran, or when ``threshold`` / ``drift_limit`` is not
            a CUSUM parameter the statistic can be compared against; ``None``
            is NOT a synonym for ``False``),
            drift_point (the recorded turn number where drift was first
            detected, None if no drift), cusum_positive, cusum_negative,
            max_cusum (all in sigma units, ``nan`` when not assessed),
            n_turns_measured, n_turns_unmeasurable.

        Reference: Page (1954), "Continuous Inspection Schemes"
        """
        turn_numbers, values, n_recorded = self._finite_series()
        sorted_turns = [int(t) for t in turn_numbers]
        n_measured = len(sorted_turns)
        n_unmeasurable = n_recorded - n_measured

        # BGL-W4 (2026-09-30). The SIBLING of the operand guard in
        # detect_drift_ewma below, found in the same measurement and reported
        # here rather than left for the next audit. Measured that day on the
        # same eight-turn ramp 0.0 -> 0.84, whose default call reports
        # has_drift=False with max_cusum 1.77:
        #   threshold=nan -> has_drift False AND max_cusum 0.0, the reading of a
        #     control chart that accumulated nothing, with ZERO warnings:
        #     max(0, C + z - nan) is 0 at every turn.
        #   threshold=inf -> identical.
        #   threshold=-1.0 -> has_drift True at turn 2, because a NEGATIVE slack
        #     adds |k| to the statistic every turn whether or not the series
        #     deviates, so a long enough trajectory always signals.
        #   drift_limit=nan -> has_drift False with a max_cusum of 1.77 sitting
        #     right there in the same dict, and ZERO warnings: `cusum > nan` is
        #     False at every turn, so the verdict came from a comparison never
        #     made.
        #   drift_limit=inf -> the same verdict from a limit nothing can cross.
        #   drift_limit=0.0 / -1.0 -> has_drift True at turn 0 for that ramp,
        #     a decision interval no accumulation can stay inside.
        # Above the dispatch for the same reason as in detect_drift_ewma.
        operand_problems = []
        if not np.isfinite(threshold) or float(threshold) < 0.0:
            operand_problems.append(
                f"threshold={threshold!r} is not a CUSUM slack this statistic can "
                f"accumulate against (it must be finite and >= 0; a negative slack adds "
                f"to the statistic every turn regardless of the data)"
            )
        if not np.isfinite(drift_limit) or float(drift_limit) <= 0.0:
            operand_problems.append(
                f"drift_limit={drift_limit!r} is not a decision interval the CUSUM can be "
                f"compared against (it must be finite and > 0)"
            )
        if operand_problems:
            warnings.warn(
                f"TemporalTracker.detect_drift_cusum: {'; '.join(operand_problems)}, so no "
                f"CUSUM verdict was reached. Reporting has_drift=None (could not check) "
                f"with a nan max_cusum, NOT False, which is the verdict 'the statistic "
                f"stayed inside its decision interval', and NOT True, which every "
                f"trajectory reaches when the slack is negative or the interval is not "
                f"positive.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "has_drift": None,
                "drift_point": None,
                "cusum_positive": [],
                "cusum_negative": [],
                "max_cusum": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        # Two turns of TOTAL disparity used to answer has_drift=False,
        # max_cusum=0.0 with EMPTY cusum series and no warning: a clean control
        # chart from a chart that was never drawn. A CUSUM needs a mean and a
        # sigma to standardize against, and neither exists for fewer than three
        # observations. Measured 2026-09-10; see detect_feedback_loop above.
        if n_measured < _MIN_TURNS_FOR_TREND:
            warn_not_assessed(
                "TemporalTracker.detect_drift_cusum",
                measured=n_measured,
                total=n_recorded,
                unit="recorded turns had a finite metric value",
                requirement=(
                    f"a CUSUM needs at least {_MIN_TURNS_FOR_TREND} to have a mean "
                    f"and a sigma to standardize against"
                ),
                reporting="has_drift=None and a nan max_cusum",
                instead_of="False with max_cusum 0.0",
            )
            return {
                "has_drift": None,
                "drift_point": None,
                "cusum_positive": [],
                "cusum_negative": [],
                "max_cusum": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        if n_unmeasurable:
            warnings.warn(
                f"TemporalTracker.detect_drift_cusum: {n_unmeasurable} of "
                f"{n_recorded} recorded turns had a non-finite metric value and were "
                f"excluded; the chart rests on the {n_measured} that remain.",
                UserWarning,
                stacklevel=2,
            )

        mean = np.mean(values)
        sigma = float(np.std(values))

        if sigma <= 1e-12:
            # Constant trajectory: no variation to accumulate, and the
            # standardization below would divide by ~0. This one IS a
            # measurement: three or more finite turns that never moved really
            # do carry no drift, so the zero series is the answer, not a
            # stand-in for one.
            n = len(values)
            return {
                "has_drift": False,
                "drift_point": None,
                "cusum_positive": [0.0] * n,
                "cusum_negative": [0.0] * n,
                "max_cusum": 0.0,
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        cusum_pos = [0.0]
        cusum_neg = [0.0]
        drift_point = None

        for i, v in enumerate(values):
            z = (v - mean) / sigma
            cusum_pos.append(max(0, cusum_pos[-1] + z - threshold))
            cusum_neg.append(max(0, cusum_neg[-1] - z - threshold))
            if (cusum_pos[-1] > drift_limit or cusum_neg[-1] > drift_limit) and drift_point is None:
                # Report the recorded turn number, not the array index:
                # turns need not be contiguous or start at 0.
                drift_point = sorted_turns[i]

        max_cusum = max(max(cusum_pos), max(cusum_neg))

        return {
            "has_drift": drift_point is not None,
            "drift_point": drift_point,
            "cusum_positive": cusum_pos[1:],
            "cusum_negative": cusum_neg[1:],
            "max_cusum": max_cusum,
            "n_turns_measured": n_measured,
            "n_turns_unmeasurable": n_unmeasurable,
        }

    def detect_drift_ewma(self, span: int = 5, sigma_limit: float = 3.0) -> dict:
        """Exponentially Weighted Moving Average (EWMA) drift detection.

        Smooths fairness metric trajectory and detects when it deviates
        beyond control limits.

        A flat trajectory (sigma ~ 0) has no variation, so its control limits
        have zero width and sit on the centre line: it reports no drift, the
        same reading detect_drift_cusum gives the same input, and that reading
        is a measurement rather than a stand-in for one.

        Args:
            span: EWMA span (higher = more smoothing). Default 5. The smoothing
                weight is ``2 / (span + 1)``, so this must be finite and >= 1
                for a weight in (0, 1]; anything else is refused rather than
                charted (see Returns).
            sigma_limit: Number of standard deviations for control limits.
                Default 3.0. A half-width in sigma units, so it must be finite
                and > 0; anything else is refused rather than compared against
                (see Returns).

        Returns:
            dict with keys: has_drift (``True`` / ``False``, or ``None`` when
            fewer than 3 recorded turns had a finite metric value, so no chart
            was drawn, or when ``span`` / ``sigma_limit`` is not a chart
            parameter the EWMA can be compared against; ``None`` is NOT a
            synonym for ``False``), drift_points
            (list of recorded turn numbers where EWMA left the control limits),
            ewma_values, upper_limit, lower_limit, center_line (``nan`` when
            not assessed), n_turns_measured, n_turns_unmeasurable.
        """
        turn_numbers, values, n_recorded = self._finite_series()
        sorted_turns = [int(t) for t in turn_numbers]
        n_measured = len(sorted_turns)
        n_unmeasurable = n_recorded - n_measured

        # BGL-W4 (2026-09-30). The BGL-5 fix further down guarded the DATA side
        # of this chart (sigma <= 1e-12, a chart with no width) and left the two
        # CALLER-SUPPLIED operands open, which is the same defect detect_drift
        # was fixed for in the same batch, arriving through a different
        # argument. Measured that day on eight turns ramping 0.0 -> 0.84, whose
        # default call draws limits 0.0256 .. 0.8144 and reports no drift:
        #   sigma_limit=nan   -> has_drift False, upper=lower=nan, ZERO
        #     warnings. `ewma_val > nan` and `ewma_val < nan` are both False, so
        #     the published verdict "no drift left the control limits" came from
        #     a comparison that was never made.
        #   sigma_limit=inf   -> the same verdict from limits +/- inf that no
        #     value can leave.
        #   sigma_limit=-3.0  -> has_drift True at ALL EIGHT turns, with upper
        #     0.0256 BELOW lower 0.8144: the limits are inverted, so the check
        #     could not have disagreed.
        #   sigma_limit=0.0   -> has_drift True at all eight turns, zero-width
        #     limits on a series that does move.
        #   span=nan          -> every ewma value nan, has_drift False, ZERO
        #     warnings: the identical fabricated verdict through the other
        #     operand.
        #   span=0 / span=-1  -> ZeroDivisionError out of lam / (2 - lam).
        # The guard sits ABOVE the three-way dispatch below (too few turns /
        # zero-width chart / chart) because all three share the precondition
        # that the caller asked for a chart that exists: inside one branch it
        # would move the fabrication to whichever branch happens not to read the
        # operand.
        operand_problems = []
        if not np.isfinite(span) or float(span) < 1.0:
            operand_problems.append(
                f"span={span!r} is not an EWMA span this chart can be smoothed with "
                f"(it must be finite and >= 1, so the smoothing weight 2/(span+1) "
                f"lands in (0, 1])"
            )
        if not np.isfinite(sigma_limit) or float(sigma_limit) <= 0.0:
            operand_problems.append(
                f"sigma_limit={sigma_limit!r} is not a control-limit width the EWMA can "
                f"be compared against (it must be finite and > 0)"
            )
        if operand_problems:
            warnings.warn(
                f"TemporalTracker.detect_drift_ewma: {'; '.join(operand_problems)}, so no "
                f"control chart was drawn. Reporting has_drift=None (could not check) "
                f"with nan chart lines, NOT False, which is the verdict 'no drift left "
                f"the control limits', and NOT True, which every turn takes when the "
                f"limits are inverted or have zero width.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "has_drift": None,
                "drift_points": [],
                "ewma_values": [],
                "upper_limit": float("nan"),
                "lower_limit": float("nan"),
                "center_line": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        # Two turns of TOTAL disparity used to answer has_drift=False with
        # center_line=0 and no warning, which reads as a control chart centred
        # on zero disparity. The control limits below need sigma with ddof=1, so
        # they are undefined for fewer than two finite points and meaningless
        # for fewer than three. Measured 2026-09-10; see detect_feedback_loop.
        if n_measured < _MIN_TURNS_FOR_TREND:
            warn_not_assessed(
                "TemporalTracker.detect_drift_ewma",
                measured=n_measured,
                total=n_recorded,
                unit="recorded turns had a finite metric value",
                requirement=(
                    f"an EWMA control chart needs at least {_MIN_TURNS_FOR_TREND} "
                    f"for its centre line and limits to exist"
                ),
                reporting="has_drift=None and a nan centre line and limits",
                instead_of="False with a centre line of 0",
            )
            return {
                "has_drift": None,
                "drift_points": [],
                "ewma_values": [],
                "upper_limit": float("nan"),
                "lower_limit": float("nan"),
                "center_line": float("nan"),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        if n_unmeasurable:
            warnings.warn(
                f"TemporalTracker.detect_drift_ewma: {n_unmeasurable} of "
                f"{n_recorded} recorded turns had a non-finite metric value and were "
                f"excluded; the chart rests on the {n_measured} that remain.",
                UserWarning,
                stacklevel=2,
            )

        lam = 2.0 / (span + 1)
        center = np.mean(values)
        sigma = np.std(values, ddof=1)

        if sigma <= 1e-12:
            # BGL-5 (2026-09-27). A CONSTANT trajectory has zero standard
            # deviation, so every control limit below has ZERO WIDTH, and the
            # recursion `lam * v + (1 - lam) * ewma[-1]` lands one ulp off the
            # centre for most ordinary values, which made `ewma_val > upper`
            # true at EVERY turn. Measured that day on five turns of an
            # identical 0.005 disparity: has_drift True, drift_points
            # [0, 1, 2, 3, 4], ewma_values [0.005000000000000001] x 5,
            # center_line == upper_limit == lower_limit == 0.005,
            # n_turns_unmeasurable 0 and ZERO warnings, on a series whose own
            # cumulative_drift is [0.0] x 5. A plain 9 of 10 against 0 of 10
            # rate gap, constant across five windows, fired identically
            # (drift_points [0, 1, 2, 3, 4] at a centre of 0.9) and reached the
            # published pulse payload as driftDetected TRUE with
            # detectorsFailed empty; a sweep of 2999 constant values by 7
            # series lengths fired 5726 times. Now: has_drift False,
            # drift_points [], ewma_values [0.005] x 5 and all three chart
            # lines equal to the centre.
            #
            # This one IS a measurement, not a refusal, for the same reason
            # detect_drift_cusum's `sigma <= 1e-12` branch above says so:
            # three or more finite turns that never moved really do carry no
            # drift, and the EWMA of a constant series IS that constant, so it
            # sits exactly on the centre line rather than outside a limit.
            n = len(values)
            return {
                "has_drift": False,
                "drift_points": [],
                "ewma_values": [float(center)] * n,
                "upper_limit": float(center),
                "lower_limit": float(center),
                "center_line": float(center),
                "n_turns_measured": n_measured,
                "n_turns_unmeasurable": n_unmeasurable,
            }

        ewma = [center]
        drift_points = []

        for i, v in enumerate(values):
            ewma_val = lam * v + (1 - lam) * ewma[-1]
            ewma.append(ewma_val)

            # Control limits widen with each observation
            se = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i + 1))))
            upper = center + sigma_limit * se
            lower = center - sigma_limit * se

            if ewma_val > upper or ewma_val < lower:
                # Report recorded turn numbers, not array indices.
                drift_points.append(sorted_turns[i])

        return {
            "has_drift": len(drift_points) > 0,
            "drift_points": drift_points,
            "ewma_values": ewma[1:],
            "upper_limit": float(center + sigma_limit * sigma * np.sqrt(lam / (2 - lam))),
            "lower_limit": float(center - sigma_limit * sigma * np.sqrt(lam / (2 - lam))),
            "center_line": float(center),
            "n_turns_measured": n_measured,
            "n_turns_unmeasurable": n_unmeasurable,
        }

    @staticmethod
    def _default_metric(
        outcomes_a: np.ndarray,
        outcomes_b: np.ndarray,
    ) -> float:
        """Default metric: mean difference between groups."""
        return float(np.mean(outcomes_a) - np.mean(outcomes_b))
