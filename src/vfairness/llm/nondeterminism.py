"""
Non-determinism management for LLM fairness testing.

LLMs produce stochastic outputs even at temperature=0 due to floating-point
non-determinism, batching effects, and KV-cache state. This module provides
tools to characterize the noise floor and isolate systematic bias from
stochastic variance.

Approach:
    1. Run identical prompts N times to characterize the noise floor.
    2. Compute noise offset = observed disparity - noise contribution.
    3. Use bootstrap confidence intervals for robust estimation.

References:
    - Ouyang et al. (2023): Non-determinism in GPT-4
    - Efron & Tibshirani (1993): Bootstrap Methods
"""

import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


# Recommended minimum runs by system type
_REQUIRED_RUNS = {
    "llm": 25,
    "agent": 50,
}


def _require_open_unit_interval(name: str, value: float) -> float:
    """Refuse a probability level that is not strictly inside (0, 1).

    BGL5 (2026-09-27). ``equivalence_test`` validated ``sesoi`` and NOT
    ``alpha``, while its two sibling methods in the same class both refused an
    alpha outside (0, 1). Measured on one fixed pair of N(0.5, 0.1) samples,
    n=20 each, seed 7:

        alpha=0.05  -> verdict 'undetermined',        p_tost 0.487941
        alpha=0.0   -> ACCEPTED, 'undetermined',      p_tost 0.487941
        alpha=1.5   -> ACCEPTED, 'fairness_confirmed', p_tost 0.487941
        alpha=-1.0  -> ACCEPTED, 'undetermined',      p_tost 0.487941
        alpha=nan   -> ACCEPTED, 'undetermined',      p_tost 0.487941

    The same data, unchanged, read "fairness_confirmed" because ``p_tost <
    alpha`` is true for every p once alpha exceeds 1. A TOST p of 0.49
    establishes equivalence at no real alpha, and nothing warned. Measured
    after: 0.0, 1.5, -1.0, nan and inf all raise ValueError and 0.05 still
    answers 'undetermined' with p_tost 0.487941.

    The finiteness half is not decoration: `nan <= 0` and `nan >= 1` are both
    False, so the two sibling guards, written as a pair of comparisons, let nan
    through as well. A RANGE check has to say what it does about the value that
    compares false to everything.
    """
    try:
        is_finite = math.isfinite(value)
    except TypeError as exc:  # not a number at all
        raise ValueError(f"{name} must be in (0, 1), got {value!r}") from exc
    if not is_finite or value <= 0 or value >= 1:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return float(value)


@dataclass
class NoiseProfile(SerializableMixin):
    """
    Characterization of stochastic noise in LLM outputs.

    Attributes:
        mean: Mean of the observed metric values.
        variance: Variance of the observed values.
        std_dev: Standard deviation of the observed values.
        iqr: Interquartile range (Q3 - Q1).
        noise_floor: Estimated noise floor (2 * std_dev), representing
            the range below which variation is attributable to randomness.
        sample_size: Number of observations used, which is the number of
            FINITE observations, not the number supplied.
        metadata: Audit trail metadata for traceability.
        n_excluded_non_finite: Observations that carried no measurement and
            were excluded from every statistic above. A floor built on
            fewer runs than were sent says so here.
    """

    mean: float
    variance: float
    std_dev: float
    iqr: float
    noise_floor: float
    sample_size: int
    metadata: RunMetadata = field(default_factory=RunMetadata)
    n_excluded_non_finite: int = 0


class NonDeterminismAnalyzer:
    """
    Manages non-determinism in LLM fairness measurements.

    Provides noise characterization, offset computation, and significance
    testing that accounts for the inherent stochasticity of LLM outputs.

    Args:
        system_type: Type of system under test. One of 'llm' (default 25
            runs) or 'agent' (default 50 runs due to higher variance from
            tool use and multi-step reasoning).
        min_runs: Override of the recommended run count. A finite number of at
            least 2, because that is where a noise floor starts existing; a
            lower value, and a non-finite one, would only silence the
            disclosure that the floor rests on too few runs.

    Raises:
        ValueError: If ``system_type`` is unknown, or ``min_runs`` is below 2
            or not finite.

    Example:
        >>> analyzer = NonDeterminismAnalyzer(system_type="llm")
        >>> # Characterize noise from identical prompt runs
        >>> lengths = np.array([len(r.split()) for r in repeated_responses])
        >>> noise = analyzer.characterize_noise(lengths)
        >>> print(f"Noise floor: {noise.noise_floor:.3f}")
        >>>
        >>> # Test if observed disparity exceeds noise
        >>> is_real = analyzer.is_significant_after_offset(
        ...     disparity=0.15, noise_floor=noise.noise_floor
        ... )

    INDEPENDENTLY CHECKED ON 2026-09-27, and two of its methods were OVERTURNED on
    that check (audit batch A-llm-1). ``required_runs`` accepted a non-finite
    ``min_runs`` (the guard tested the range only for integers, and `nan < 2` is
    False), and ``equivalence_test`` validated ``sesoi`` and not ``alpha``, so
    alpha=1.5 turned a TOST p of 0.487941 into 'fairness_confirmed' on unchanged
    data. Both are closed above and in ``_require_open_unit_interval``, each with
    its measured before and after, and pinned in tests/test_bgl5_llm_1.py. The two
    attacks that found them were one attack: take the guard's own message and try a
    value the pin never tried.

    The block below is MACHINE-WRITTEN between its two markers, by
    ``scripts/stamp_proof_status.py`` from the grade ledger, and it is deliberately
    not hand-edited here: an edited stamp is reported stale by that script's check
    mode and overwritten by its next run, so a correction typed into it would not
    survive. It also cannot say everything this paragraph says, because it composes one
    sentence per capability from flags, while this audit overturned two of the
    seven units it answers for. Measured 2026-09-27: the stamper wants to write
    "An independent check has argued with that judgement and upheld it" for this
    class, which is true of five of those units and not of the two named above.
    This paragraph and tests/test_bgl5_llm_1.py are the record of what the audit
    found; the ledger row is where that has to be reconciled.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: non_determinism_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        system_type: str = "llm",
        min_runs: Optional[int] = None,
    ) -> None:
        if system_type not in _REQUIRED_RUNS:
            raise ValueError(
                f"Unsupported system_type '{system_type}'. "
                f"Must be one of {list(_REQUIRED_RUNS.keys())}"
            )
        # BGL3 (2026-09-27): a minimum below 2 is not a recommendation, it is an
        # off switch for the only disclosure that says a floor is thin. Every
        # "fewer runs than recommended" statement in this module is `n <
        # required_runs()`, so `min_runs=0` makes all of them unreachable while
        # the result still reads as fully measured. Measured through the public
        # entry point on two runs per variant, noise_floor_from_runs with
        # metrics=["response_length"]:
        #
        #   default min_runs -> variant state 'measured_with_limitation',
        #                       reason 'below_recommended_runs (2 < 25)', two
        #                       limitations naming it, two RuntimeWarnings
        #   min_runs=0       -> variant state 'measured', reason None,
        #                       limitations [], both warnings gone
        #   min_runs=-5      -> identical to min_runs=0
        #
        # A two-run floor then reads exactly like a twenty-five-run floor.
        # characterize_noise itself refuses below 2 finite values, so 2 is the
        # floor of what any minimum can mean.
        # BGL5 (2026-09-27), the same off switch reached by a float. The guard
        # below closes the INTEGER path and `nan < 2` is False, so a non-finite
        # minimum walked straight past it. Measured before this branch existed:
        #
        #   NonDeterminismAnalyzer('llm', min_runs=float('nan'))
        #       -> ACCEPTED, required_runs() -> nan
        #   NonDeterminismAnalyzer('llm', min_runs=float('inf'))
        #       -> ACCEPTED, required_runs() -> inf
        #   noise_floor_from_runs(two runs per variant, metrics=['response_length'],
        #                         min_runs=float('nan'))
        #       -> variant states ['measured', 'measured'], limitations 0,
        #          warnings [] : character for character the min_runs=0 outcome
        #          recorded below, which is what the guard below refuses
        #
        # Measured after: both raise ValueError('min_runs must be at least 2,
        # got nan / inf: ...') from the constructor, and that noise_floor_from_runs
        # call raises instead of returning a clean-looking result. The default
        # path is untouched: two runs per variant still reports
        # ['measured_with_limitation', 'measured_with_limitation'] with 2
        # limitations and the 'recommended minimum is 25' RuntimeWarnings.
        #
        # A RANGE test, not a type test: `isinstance(min_runs, (int, float))`
        # accepts nan and inf, and rejects np.int64, so neither half of it
        # asks the question the message asks.
        if min_runs is not None:
            try:
                min_runs_is_finite = math.isfinite(min_runs)
            except TypeError:
                # Not a number at all. The comparison below reports it, as it
                # always has, rather than this branch claiming it is non-finite.
                min_runs_is_finite = True
            if not min_runs_is_finite:
                raise ValueError(
                    f"min_runs must be at least 2, got {min_runs}: a noise floor needs two "
                    f"repeated runs before it exists, and a non-finite minimum is not a "
                    f"number of runs at all. `nan < 2` is False, so it silences every "
                    f"'fewer runs than recommended' disclosure in this module instead of "
                    f"raising, which is exactly what a minimum below 2 does."
                )
        if min_runs is not None and min_runs < 2:
            raise ValueError(
                f"min_runs must be at least 2, got {min_runs}: a noise floor needs two "
                f"repeated runs before it exists, and a lower minimum only silences the "
                f"disclosure that says how thin the floor is."
            )
        self._system_type = system_type
        self._min_runs_override = min_runs

    def required_runs(self) -> int:
        """
        Return the recommended minimum number of runs for this system type.

        If ``min_runs`` was provided to the constructor, that value is
        returned instead of the default for the system type. The constructor
        refuses a ``min_runs`` below 2 AND a non-finite one, so this can never
        answer a number that makes every "fewer runs than recommended"
        disclosure unreachable. Both halves matter: `nan < 2` is False, so
        until 2026-09-27 the second half was missing and nan answered here.

        Returns:
            25 for 'llm', 50 for 'agent', or the custom ``min_runs``
            value if provided.
        """
        if self._min_runs_override is not None:
            return self._min_runs_override
        return _REQUIRED_RUNS[self._system_type]

    def characterize_noise(self, values: np.ndarray) -> NoiseProfile:
        """
        Compute noise floor from repeated identical-prompt measurements.

        The noise floor is defined as 2 * std_dev, representing the
        approximate 95% range of stochastic variation.

        Args:
            values: 1-D array of metric values from repeated identical runs.
                Must have at least 2 FINITE elements. Non-finite elements are
                runs that produced no measurement; they are excluded, counted
                on the profile, and warned about.

        Returns:
            NoiseProfile with descriptive statistics and noise floor.

        Raises:
            ValueError: If fewer than 2 finite values remain.
        """
        logger.info(
            "characterize_noise: system_type=%s, n_values=%d", self._system_type, len(values)
        )
        values = np.asarray(values, dtype=np.float64).ravel()

        # NAN-01 (2026-09-10): a run that produced no measurement is not a
        # measurement of no variation. ONE unscored run out of 30 made every
        # statistic below NaN, and that NaN floor then walked into
        # compute_noise_offset, which graded a real 0.40 disparity as within
        # noise. Non-finite elements are excluded and counted here, the way
        # noise_floor_from_runs already excludes empty responses, so the floor
        # is measured on the runs that HAVE a measurement and the profile
        # states how many it dropped.
        n_supplied = len(values)
        finite_values = values[np.isfinite(values)]
        n_excluded = n_supplied - len(finite_values)
        if n_excluded:
            logger.warning(
                "characterize_noise: excluded %d of %d non-finite values",
                n_excluded,
                n_supplied,
            )
            warnings.warn(
                f"characterize_noise: {n_excluded} of {n_supplied} values are not finite "
                f"and were excluded; the noise floor rests on the {len(finite_values)} "
                f"that carry a measurement, and says nothing about the excluded runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        values = finite_values

        if len(values) < 2:
            detail = f" ({n_excluded} of {n_supplied} were not finite)" if n_excluded else ""
            raise ValueError(
                f"Need at least 2 finite values to characterize noise, got {len(values)}{detail}"
            )

        if len(values) < self.required_runs():
            logger.warning(
                "Only %d samples provided; recommended minimum is %d for system_type='%s'",
                len(values),
                self.required_runs(),
                self._system_type,
            )
            warnings.warn(
                f"Only {len(values)} samples provided; recommended minimum "
                f"is {self.required_runs()} for system_type='{self._system_type}'.",
                RuntimeWarning,
                stacklevel=2,
            )

        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1))
        std_dev = float(np.std(values, ddof=1))
        q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        iqr = q3 - q1
        noise_floor = 2.0 * std_dev

        logger.info(
            "characterize_noise complete: noise_floor=%.4f, std_dev=%.4f", noise_floor, std_dev
        )
        return NoiseProfile(
            mean=mean,
            variance=variance,
            std_dev=std_dev,
            iqr=iqr,
            noise_floor=noise_floor,
            sample_size=len(values),
            metadata=RunMetadata(
                system_type=self._system_type,
                parameters={
                    "sample_size": len(values),
                    "n_supplied": n_supplied,
                    "n_excluded_non_finite": n_excluded,
                },
            ),
            n_excluded_non_finite=n_excluded,
        )

    def compute_noise_offset(
        self,
        observed_disparity: float,
        noise_floor: float,
        confidence: float = 0.95,
    ) -> dict:
        """
        Isolate systematic bias from stochastic variance.

        Subtracts the noise contribution from the observed disparity to
        estimate the systematic component.

        Args:
            observed_disparity: The raw disparity measurement (absolute value).
            noise_floor: The noise floor from characterize_noise().
            confidence: Confidence level for the z-score threshold.

        Returns:
            Dict with keys, three states and never two:
                - state (str): 'measured' or 'could_not_check'.
                - reason (str or None): why nothing was graded.
                - observed (float): Original disparity.
                - noise_floor (float): The noise floor used.
                - z_threshold (float): z-score for the confidence level.
                - noise_contribution (float or None): Estimated stochastic
                  component; None when it could not be computed.
                - systematic_offset (float or None): Estimated systematic bias
                  (max(0, observed - noise_contribution)); None when the
                  comparison could not be made.
                - exceeds_noise (bool or None): Whether disparity exceeds the
                  noise floor; None when that could not be decided.
        """
        confidence = _require_open_unit_interval("confidence", confidence)

        z_threshold = float(stats.norm.ppf((1 + confidence) / 2))

        # NAN-01 (2026-09-10): an unmeasurable comparison is not a quiet one.
        # `max(0.0, nan)` is 0.0 and `nan > floor` is False, so a noise floor
        # carrying one unscored run out of 30 reported systematic_offset 0.0
        # and exceeds_noise False for a real 0.40 disparity, and
        # is_significant_after_offset then answered False. Measured
        # 2026-09-10: significant True -> False on a single NaN. The grade is
        # withheld now, and the caller is told which input was not finite.
        if not math.isfinite(observed_disparity) or not math.isfinite(noise_floor):
            warnings.warn(
                f"compute_noise_offset: observed_disparity={observed_disparity} / "
                f"noise_floor={noise_floor} is not finite, so the disparity was not "
                f"graded against the noise. Reporting 'could_not_check'; this is "
                f"neither within noise nor above it.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "state": "could_not_check",
                "reason": "observed_disparity_or_noise_floor_not_finite",
                "observed": observed_disparity,
                "noise_floor": noise_floor,
                "z_threshold": z_threshold,
                "noise_contribution": None,
                "systematic_offset": None,
                "exceeds_noise": None,
            }

        # Noise contribution at the given confidence level
        noise_contribution = noise_floor * z_threshold / 2.0
        systematic = max(0.0, abs(observed_disparity) - noise_contribution)

        return {
            "state": "measured",
            "reason": None,
            "observed": observed_disparity,
            "noise_floor": noise_floor,
            "z_threshold": z_threshold,
            "noise_contribution": noise_contribution,
            "systematic_offset": systematic,
            "exceeds_noise": abs(observed_disparity) > noise_floor,
        }

    def is_significant_after_offset(
        self,
        disparity: float,
        noise_floor: float,
        alpha: float = 0.05,
    ) -> Optional[bool]:
        """
        Determine if a disparity is significant after noise offset.

        A disparity is considered significant if its absolute value exceeds
        the noise floor scaled by the z-score for (1 - alpha).

        Args:
            disparity: The observed disparity value.
            noise_floor: The noise floor from characterize_noise().
            alpha: Significance level.

        Returns:
            True if the disparity is statistically significant, False if it
            is not, and None if it could not be checked (NAN-01: a non-finite
            disparity or noise floor is not an insignificant one).
        """
        alpha = _require_open_unit_interval("alpha", alpha)

        confidence = 1.0 - alpha
        offset = self.compute_noise_offset(abs(disparity), noise_floor, confidence)
        systematic = offset["systematic_offset"]
        result: Optional[bool] = None if systematic is None else bool(systematic > 0)
        logger.debug(
            "is_significant_after_offset: disparity=%.4f, noise_floor=%.4f, result=%s",
            disparity,
            noise_floor,
            result,
        )
        return result

    # C-21: TOST Equivalence Testing (Lakens, 2017)
    def equivalence_test(
        self,
        group_a_values: np.ndarray,
        group_b_values: np.ndarray,
        sesoi: float = 0.2,
        alpha: float = 0.05,
        sesoi_raw: Optional[float] = None,
    ) -> dict:
        """
        Two One-Sided Tests (TOST) for equivalence.

        Standard null-hypothesis testing asks "is there a difference?"
        TOST asks "can we confirm there is NO meaningful difference?"
        This is the more useful question for fairness evaluation.

        Args:
            group_a_values: Metric values for demographic group A.
            group_b_values: Metric values for demographic group B.
            sesoi: Smallest Effect Size of Interest in Cohen's d units.
                Default 0.2 ("small" effect). For high-stakes domains
                (healthcare, criminal justice), use 0.1.
            alpha: Significance level for both the standard test and TOST.
                Must be strictly inside (0, 1), like the alpha of
                ``is_significant_after_offset`` and the confidence of
                ``compute_noise_offset``. An alpha above 1 makes ``p_tost <
                alpha`` true for every p, so it manufactures
                'fairness_confirmed' from any data at all; see
                ``_require_open_unit_interval`` for the measurement.
            sesoi_raw: The smallest meaningful difference on the METRIC'S OWN
                scale. Used only when both arms are constant, where Cohen's d
                is undefined because there is no variance to standardise by.
                Without it such a difference is reported as real and ungraded
                rather than being given a verdict this method cannot support.

        Returns:
            Dict with:
                - verdict: One of 'bias_detected', 'fairness_confirmed',
                  'below_practical_significance', 'undetermined'.
                - effect_size: Observed Cohen's d.
                - sesoi: The SESOI threshold used.
                - p_diff: p-value from standard two-sample t-test.
                - p_tost: p-value from TOST (max of two one-sided tests).
                - mean_diff: Raw mean difference (A - B).
                - ci_lower: Lower bound of 90% CI on effect size.
                - ci_upper: Upper bound of 90% CI on effect size.
                - interpretation: Human-readable explanation of the verdict.

        References:
            Lakens, D. (2017). Equivalence Tests: A Practical Primer for
            t Tests, Correlations, and Meta-Analyses. Social Psychological
            and Personality Science, 8(4), 355-362.
        """
        a = np.asarray(group_a_values, dtype=np.float64).ravel()
        b = np.asarray(group_b_values, dtype=np.float64).ravel()

        if len(a) < 2 or len(b) < 2:
            raise ValueError(
                f"Need at least 2 values per group for equivalence test. Got {len(a)} and {len(b)}."
            )
        # FINITE AND POSITIVE, not merely `> 0`. BGL6 B4 audit, 2026-09-29.
        # `inf <= 0` and `nan <= 0` are both False, so both walked through this
        # guard. Measured on two groups drawn from N(0.20, 0.01) and N(0.80, 0.01),
        # n=40, seed 0, which the same method calls 'bias_detected' at a real sesoi
        # of 0.05: sesoi=inf answered **'fairness_confirmed'** on a TOST p of 0.0,
        # with no warning and no raise, because an equivalence bound of infinity
        # makes any observed difference "equivalent" by construction. That is a
        # fabricated clean bill of fairness on a difference of 0.6.
        #
        # The comment immediately below records the alpha parameter of THIS SAME
        # method being fixed for THIS SAME shape two days earlier. The guard above
        # it was left as a bare comparison, which is why the pin for that fix could
        # pass while this one fabricated: when two parameters share a precondition,
        # checking one of them is not checking the precondition.
        if not (math.isfinite(sesoi) and sesoi > 0):
            raise ValueError(f"SESOI must be a finite positive number, got {sesoi}")
        # BGL5 (2026-09-27): sesoi was validated here and alpha was not, while
        # BOTH sibling methods in this class refuse an alpha outside (0, 1).
        # alpha is read twice below, by `p_diff < alpha` and by `p_tost <
        # alpha`, and the second of those is the branch that returns
        # 'fairness_confirmed'. Measured on one fixed pair of N(0.5, 0.1)
        # samples, n=20, seed 7: alpha=1.5 was ACCEPTED and answered
        # 'fairness_confirmed' on a TOST p of 0.487941, unchanged data,
        # no warning; alpha=0.0 and alpha=-1.0 were accepted too. Measured
        # after: all three raise ValueError('alpha must be in (0, 1), got ...')
        # and alpha=0.05 still answers 'undetermined' with p_tost 0.487941.
        # The guard is ABOVE the constant-arms dispatch on purpose: that branch
        # returns before the t-test, so a guard below it could not fire for the
        # commonest genuinely-fair shape there is.
        alpha = _require_open_unit_interval("alpha", alpha)

        mean_diff = float(np.mean(a) - np.mean(b))
        # ZERO VARIANCE IN BOTH ARMS. Handled here, before the 1e-12 clamp,
        # because the clamp turns this case into arithmetic that looks valid and
        # is meaningless: it scales the SESOI down by the same 1e-12, so the
        # t-statistics come out around 0.245 whatever the data did, and TOST
        # reports "equivalence not established" for two arms that were
        # BYTE-IDENTICAL on every observation.
        #
        # This is not an edge case. The probe calls the endpoint at temperature
        # 0, where repeated runs return identical text, so two constant equal
        # arms is the commonest genuinely-fair shape there is.
        #
        # It is also where this method DISAGREED with
        # ``operations.pulse.llm_probe._pair_stats`` on the same data:
        # _pair_stats reported (p=1.0, effect=0.0), "compared, no evidence of a
        # difference", and this reported 'could_not_check', because
        # ``ttest_ind`` on two constant arrays answers NaN. Two answers to "are
        # these the same" for the case that occurs most. _pair_stats had it
        # right and this now agrees with it.
        # np.ptp, NOT np.var == 0.0. The variance is an ACCUMULATED statistic
        # and a constant array of a value that is not exactly representable in
        # binary does not have exactly zero variance:
        #
        #   np.var([0.9] * 20)  ->  4.93e-32     != 0.0
        #   np.var([0.1] * 20)  ->  1.93e-34     != 0.0
        #   np.var([0.5] * 20)  ->  0.0          == 0.0
        #
        # and it depends on n as well as the value: at n=5 every one of those is
        # exactly 0.0, at n=20 only the powers of two are. So the guard fired
        # for round numbers and small samples and silently missed everything
        # else, which fell through to the 1e-12 clamp below and produced
        # Cohen's d of -4e11 reported as "a real, meaningful bias".
        #
        # ptp asks the actual question, did every observation have the same
        # value, on the RAW data with nothing accumulated and no epsilon to
        # argue about. Found by a peer session running the unequal case; my own
        # fixtures used 0.5 and n=5 and passed for the wrong reason.
        if float(np.ptp(a)) == 0.0 and float(np.ptp(b)) == 0.0:
            return self._constant_arms_verdict(a, b, mean_diff, sesoi, sesoi_raw)

        # Standard two-sample t-test (difference test). It runs only AFTER the
        # constant-arms branch: on two constant arrays it answers NaN and emits
        # a catastrophic-cancellation warning, and running a test whose answer
        # is going to be discarded is how that NaN reached the verdict logic in
        # the first place.
        t_diff, p_diff = stats.ttest_ind(a, b, equal_var=False)

        # Pooled standard deviation for Cohen's d
        n_a, n_b = len(a), len(b)
        pooled_std = float(
            np.sqrt(
                ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
            )
        )
        mean_diff = float(np.mean(a) - np.mean(b))

        if pooled_std < 1e-12:
            pooled_std = 1e-12  # Avoid division by zero

        cohens_d = mean_diff / pooled_std

        # Standard error of the mean difference
        se = pooled_std * np.sqrt(1.0 / n_a + 1.0 / n_b)
        df = n_a + n_b - 2

        # TOST: two one-sided tests against SESOI bounds
        # Upper bound test: H0: effect >= SESOI*pooled_std
        t_upper = (mean_diff - sesoi * pooled_std) / se
        p_upper = float(stats.t.cdf(t_upper, df=df))

        # Lower bound test: H0: effect <= -SESOI*pooled_std
        t_lower = (mean_diff + sesoi * pooled_std) / se
        p_lower = 1.0 - float(stats.t.cdf(t_lower, df=df))

        p_tost = max(p_upper, p_lower)

        # 90% CI on mean difference (appropriate for equivalence testing)
        ci_margin = float(stats.t.ppf(1 - alpha, df=df)) * se
        ci_lower = (mean_diff - ci_margin) / pooled_std  # In Cohen's d units
        ci_upper = (mean_diff + ci_margin) / pooled_std

        # Four-way verdict, plus the fifth state every one of them assumed
        # away: `nan < alpha` is False, so an untestable comparison walked into
        # whichever branch means "nothing to worry about here".
        if not math.isfinite(p_diff) or not math.isfinite(cohens_d):
            warnings.warn(
                f"nondeterminism analysis: p={p_diff} / d={cohens_d} is not finite, so "
                f"no verdict was reached. Reporting 'could_not_check' rather than one "
                f"of the four graded outcomes.",
                UserWarning,
                stacklevel=2,
            )
            significant_diff = None
        else:
            significant_diff = p_diff < alpha
        abs_d = abs(cohens_d)

        if significant_diff is None:
            verdict = "could_not_check"
            interpretation = (
                f"COULD NOT CHECK: the comparison did not produce a usable p-value "
                f"(p={p_diff}) or effect size (d={cohens_d}), so no conclusion about "
                f"non-determinism bias was reached. This is neither a pass nor a fail."
            )
        elif significant_diff and abs_d >= sesoi:
            verdict = "bias_detected"
            interpretation = (
                f"Statistically significant difference detected (p={p_diff:.4f}) "
                f"with effect size d={cohens_d:.3f} exceeding the SESOI threshold "
                f"of {sesoi}. This is a real, meaningful bias."
            )
        elif p_tost < alpha:
            verdict = "fairness_confirmed"
            interpretation = (
                f"Equivalence confirmed (TOST p={p_tost:.4f}). The effect size "
                f"d={cohens_d:.3f} falls within the equivalence bounds "
                f"[{-sesoi}, {sesoi}]. Groups are treated equivalently."
            )
        elif significant_diff and abs_d < sesoi:
            verdict = "below_practical_significance"
            interpretation = (
                f"A statistically significant difference exists (p={p_diff:.4f}) "
                f"but the effect size d={cohens_d:.3f} is below the SESOI "
                f"threshold of {sesoi}. The difference is real but too small "
                f"to be practically meaningful."
            )
        else:
            verdict = "undetermined"
            interpretation = (
                f"Insufficient evidence to conclude either bias or equivalence "
                f"(diff p={p_diff:.4f}, TOST p={p_tost:.4f}). "
                f"Consider increasing sample size for more statistical power."
            )

        logger.info(
            "equivalence_test: verdict=%s, d=%.4f, sesoi=%.2f, p_diff=%.4f, p_tost=%.4f",
            verdict,
            cohens_d,
            sesoi,
            p_diff,
            p_tost,
        )

        return {
            "verdict": verdict,
            "effect_size": round(cohens_d, 4),
            "sesoi": sesoi,
            "p_diff": round(float(p_diff), 6),
            "p_tost": round(float(p_tost), 6),
            "mean_diff": round(mean_diff, 6),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "interpretation": interpretation,
        }

    @staticmethod
    def _constant_arms_verdict(
        a: np.ndarray,
        b: np.ndarray,
        mean_diff: float,
        sesoi: float,
        sesoi_raw: Optional[float],
    ) -> dict:
        """The verdict when neither arm varies at all.

        There is no sampling noise to test against, so no p-value exists and
        none is invented: ``p_diff`` and ``p_tost`` are None. What the data DO
        say is unambiguous, and it is more than a t-test could have told us.

        Equal constants
            Every observation in both arms was the same value. The system
            behaved identically, reproducibly. That is equivalence established
            by observation rather than inferred, and reporting it as
            could-not-check discards a real measurement.

        Different constants
            The two arms differ by exactly the same amount every time. The
            DIFFERENCE is certain; only its practical significance is ungraded,
            because Cohen's d needs a variance and there is none. With a
            ``sesoi_raw`` it is graded on the metric's own scale. Without one it
            is reported as real and ungraded, leading with the finding, because
            'undetermined' alone would read as "nothing to see".
        """
        value_a, value_b = float(a[0]), float(b[0])
        common = {
            "sesoi": sesoi,
            "p_diff": None,
            "p_tost": None,
            "mean_diff": round(mean_diff, 6),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "constant_arms": True,
        }
        if mean_diff == 0.0:
            return {
                **common,
                "verdict": "fairness_confirmed",
                "effect_size": 0.0,
                "interpretation": (
                    f"Both groups produced the identical constant value {value_a} on every "
                    f"observation ({len(a)} and {len(b)}), so the difference is exactly zero and "
                    f"there is no sampling noise to test against. Equivalence is established by "
                    f"observation rather than inferred, and no p-value exists for it. This says "
                    f"nothing about prompts that were not tested."
                ),
            }
        if sesoi_raw is not None and abs(mean_diff) >= float(sesoi_raw):
            return {
                **common,
                "verdict": "bias_detected",
                "effect_size": None,
                "interpretation": (
                    f"Both groups are constant and they differ: {value_a} against {value_b}, a gap "
                    f"of {abs(mean_diff)} on every one of {len(a)} and {len(b)} observations. That "
                    f"gap meets the smallest meaningful difference of {sesoi_raw} on this metric's "
                    f"own scale. Cohen's d is undefined here because there is no variance to "
                    f"standardise by, so no effect size is reported: the raw gap is the finding, "
                    f"and it is perfectly reproducible."
                ),
            }
        if sesoi_raw is not None:
            return {
                **common,
                "verdict": "below_practical_significance",
                "effect_size": None,
                "interpretation": (
                    f"Both groups are constant and they differ by {abs(mean_diff)} "
                    f"({value_a} against {value_b}), below the smallest meaningful difference of "
                    f"{sesoi_raw} on this metric's own scale. The difference is perfectly "
                    f"reproducible and too small to matter."
                ),
            }
        return {
            **common,
            "verdict": "undetermined",
            "effect_size": None,
            "interpretation": (
                f"Both groups are constant and they DIFFER: {value_a} against {value_b}, a gap of "
                f"{abs(mean_diff)} on every one of {len(a)} and {len(b)} observations. The "
                f"difference is certain and perfectly reproducible. Whether it MATTERS could not "
                f"be graded: Cohen's d is undefined without variance, so the SESOI of {sesoi} in "
                f"d units cannot be applied. Pass sesoi_raw to grade it on this metric's own "
                f"scale. This is an ungraded real difference, not an absence of one."
            ),
        }

    def bootstrap_ci(
        self,
        values: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        statistic: str = "mean",
        random_state: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a statistic.

        Args:
            values: 1-D array of observed values.
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            statistic: Statistic to bootstrap. One of 'mean', 'median',
                'std'.
            random_state: Seed for the random number generator for
                reproducibility. Default None (non-deterministic).

        Returns:
            Tuple of ``(lower_bound, upper_bound)``, or ``(nan, nan)`` when the
            interval could NOT be computed. Three states, never two: an
            interval measured on finite observations, a refusal carried as
            ``(nan, nan)`` with a ``RuntimeWarning`` naming the reason, and the
            hard input errors below. A zero-width interval is never returned as
            a stand-in for "no uncertainty was measurable": it comes back ONLY
            when every finite observation is identical, and then it always
            carries a ``RuntimeWarning`` saying the width is observed constancy
            rather than estimated certainty.

        Raises:
            ValueError: If values is empty or confidence is invalid.
        """
        values = np.asarray(values, dtype=np.float64).ravel()

        if len(values) == 0:
            raise ValueError("Cannot compute bootstrap CI on empty array")
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

        # BGL-S2b (2026-09-17): this validation used to sit BELOW the
        # degenerate-input early return, so bootstrap_ci(one_value,
        # statistic="maen") answered (nan, nan) - a refusal about the DATA -
        # for what is a typo in the call. A misspelled statistic is a
        # programming error whatever the data looks like, so it raises first.
        stat_funcs: dict[str, Callable[[np.ndarray], Any]] = {
            "mean": np.mean,
            "median": np.median,
            "std": lambda x: np.std(x, ddof=1),
        }
        if statistic not in stat_funcs:
            raise ValueError(
                f"Unsupported statistic '{statistic}'. Must be one of {list(stat_funcs.keys())}"
            )

        # BGL-S2 (2026-09-16): n == 0 used to be the ONLY guard here, so this
        # method had two states where its own sibling characterize_noise has
        # three. Both directions were wrong on the same inputs. FORWARD: a
        # single observation returned a zero-width 95% interval ((0.6, 0.6)),
        # a confident claim that there is no uncertainty, from one run.
        # REVERSE: there was no finiteness filter, so ONE non-finite value
        # among 59 real measurements made np.percentile return (nan, nan) with
        # no warning and no reason, discarding 59 measurements as a mute
        # could-not-check that is byte-identical to the all-NaN case.
        # Mirrors characterize_noise: exclude non-finite values, count them,
        # say so; below 2 finite values refuse with (nan, nan) plus the reason
        # rather than inventing an interval; below required_runs() the
        # interval IS measured and is said to be less certain.
        n_supplied = len(values)
        finite_values = values[np.isfinite(values)]
        n_excluded = n_supplied - len(finite_values)
        if n_excluded:
            logger.warning(
                "bootstrap_ci: excluded %d of %d non-finite values", n_excluded, n_supplied
            )
            warnings.warn(
                f"bootstrap_ci: {n_excluded} of {n_supplied} values are not finite and were "
                f"excluded; the interval rests on the {len(finite_values)} that carry a "
                f"measurement, and says nothing about the excluded runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        values = finite_values

        if len(values) < 2:
            detail = f" ({n_excluded} of {n_supplied} were not finite)" if n_excluded else ""
            logger.warning("bootstrap_ci: %d finite value(s); returning (nan, nan)", len(values))
            warnings.warn(
                f"bootstrap_ci: need at least 2 finite values to estimate an interval, got "
                f"{len(values)}{detail}. Returning (nan, nan): the uncertainty was NOT "
                f"measured. A zero-width interval would claim certainty no observation "
                f"supports.",
                RuntimeWarning,
                stacklevel=2,
            )
            return (float("nan"), float("nan"))

        if len(values) < self.required_runs():
            logger.warning(
                "bootstrap_ci: only %d finite samples; recommended minimum is %d for "
                "system_type='%s'",
                len(values),
                self.required_runs(),
                self._system_type,
            )
            warnings.warn(
                f"bootstrap_ci: only {len(values)} finite samples provided; recommended "
                f"minimum is {self.required_runs()} for system_type='{self._system_type}'. "
                f"The interval is measured, and less certain than that minimum supports.",
                RuntimeWarning,
                stacklevel=2,
            )

        # BGL-S2b (2026-09-17): the SAME SHAPE one branch below the fix above.
        # A degenerate sample makes the bootstrap silent rather than wrong: it
        # returned a zero-width 95% interval and, at or above required_runs(),
        # no warning at all. Measured before this change:
        # bootstrap_ci(np.full(30, 0.5)) -> (0.5, 0.5), width 0.0, warnings [];
        # np.zeros(30), i.e. thirty runs that all scored refusal_rate 0.0 ->
        # (0.0, 0.0), warnings []. A reader cannot tell that from an interval
        # that was estimated and came out tight, and thirty identical 0.0
        # observations are consistent with a true rate near 3/n (the rule of
        # three), not with zero.
        #
        # The interval is NOT refused: every resample of a constant sample IS
        # that constant, so (c, c) is what was observed, and answering
        # (nan, nan) would throw away a real measurement. What was missing is
        # the sentence saying what it means. equivalence_test already says it
        # for the same input ("there is no sampling noise to test against ...
        # Equivalence is established by observation rather than inferred"), so
        # this is that treatment applied here. np.ptp on the RAW values, never
        # np.var(...) == 0.0, which is exactly false at most n.
        if float(np.ptp(values)) == 0.0:
            constant = float(values[0])
            logger.warning(
                "bootstrap_ci: all %d values are %r; the interval has zero width by "
                "observation, not by estimation",
                len(values),
                constant,
            )
            warnings.warn(
                f"bootstrap_ci: all {len(values)} finite values are {constant}, so every "
                f"resample is identical and the interval is ({constant}, {constant}), of "
                f"zero width. Read that as OBSERVED CONSTANCY, not as measured certainty: "
                f"a bootstrap cannot express uncertainty for a degenerate sample, and "
                f"{len(values)} identical observations are consistent with a true value "
                f"that differs from {constant}. The zero width is a fact about this "
                f"sample, not a confidence claim about the population.",
                RuntimeWarning,
                stacklevel=2,
            )

        func = stat_funcs[statistic]
        rng = np.random.default_rng(random_state)

        boot_stats = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_stats[i] = func(sample)

        alpha = 1.0 - confidence
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return (lower, upper)


# LF-01 (2026-09-09): the noise floor, from the series it is defined on.
#
# ``characterize_noise`` has always documented its input as "metric values
# from repeated identical runs". The Navigator's Non-Determinism step fed it
# the absolute per-metric DISPARITIES across different templates and metrics
# instead, so the number it printed as a noise floor was the spread of
# disparities, not run-to-run stochasticity, and the step made no model call.
# The per-run responses that a real floor needs were already stored by the
# counterfactual step. This function consumes exactly those.

#: Metrics a per-run series can be built for without any external service.
RUN_METRICS: Tuple[str, ...] = ("sentiment", "toxicity", "refusal_rate", "response_length")

#: Scripts that are written WITHOUT spaces between words: CJK ideographs, the
#: Japanese kana, and the CJK compatibility block. Hangul is deliberately NOT
#: here, because Korean does space its words and counting its syllables would
#: overcount the same way ``str.split`` undercounts Chinese.
_UNSPACED_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)


def _word_count(text: str) -> int:
    """Length in word-like units, for scripts that space their words and those that do not.

    BGL-S2b (2026-09-17). ``len(text.split())`` is a whitespace-token count,
    and Chinese and Japanese do not put whitespace between words, so every CJK
    sentence of any length whatsoever measured as exactly 1 "word". Measured
    before this change, a 13-character Chinese refusal against its 9-word
    English translation reported response_length 1.0 against 9.0, an
    eight-unit "length disparity" that is entirely an artefact of the
    tokeniser: the two responses say the same thing at the same length.

    Characters of an unspaced script count as one unit each, which is the
    standard convention for CJK length; everything else is counted by
    whitespace tokens as before, so a pure-ASCII corpus is numerically
    unchanged.
    """
    unspaced = len(_UNSPACED_SCRIPT_RE.findall(text))
    rest = len(_UNSPACED_SCRIPT_RE.sub(" ", text).split())
    return unspaced + rest


def noise_floor_from_runs(
    responses_by_variant: Dict[str, Sequence[Optional[str]]],
    *,
    metrics: Optional[Sequence[str]] = None,
    system_type: str = "llm",
    min_runs: Optional[int] = None,
    sampling: Optional[Dict[str, Any]] = None,
    scorers: Optional[Dict[str, Any]] = None,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    sesoi: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Noise floor per metric and per variant from REPEATED IDENTICAL PROMPTS.

    When ``sesoi`` is given, every measured comparison also carries an
    ``equivalence`` block from :meth:`NonDeterminismAnalyzer.equivalence_test`
    (TOST, Lakens 2017) on the two per-run series, so "no meaningful
    difference" can be claimed on the series it must be claimed on.

    Args:
        responses_by_variant: ``{variant_label: [response, ...]}`` where every
            list holds the raw responses to ONE prompt sent repeatedly. The
            first key is the reference variant. Empty and ``None`` responses
            are counted and excluded, never scored.
        metrics: Subset of :data:`RUN_METRICS`. Default: all four.
        system_type: ``"llm"`` or ``"agent"`` (sets the recommended run count).
        min_runs: Override of the recommended run count.
        sampling: The ``{"temperature", "top_p", "seed"}`` the runs were made
            with. Recorded verbatim; a floor is only meaningful at the
            settings it was measured at.
        scorers: Optional ``{metric: TextScorer}`` overrides.
        confidence: Confidence level for the noise-offset z threshold.
        random_state: Seed for the bootstrap.

    Returns:
        A dict with three states at every level, never two:

        - ``state``: ``"measured"`` if at least one metric could be compared,
          else ``"could_not_check"``.
        - ``variants[label]``: ``n_runs``, ``n_valid``, ``empty_rate``, a
          ``state`` of ``measured`` / ``measured_with_limitation`` (fewer
          valid runs than recommended) / ``could_not_check`` (fewer than two
          valid responses), and per-metric ``NoiseProfile`` dicts with a
          bootstrap interval on the mean.
        - ``metrics[name]``: for each non-reference variant the observed
          disparity in means, the **disparity noise floor**
          ``2 * sqrt(var_ref / n_ref + var_v / n_v)`` (the 2-sigma band of a
          difference of two run means), ``exceeds_noise``, and the
          ``systematic_offset`` from :meth:`NonDeterminismAnalyzer.compute_noise_offset`.
          Scorer provenance is attached so a floor measured with a keyword
          placeholder cannot be mistaken for one measured with a transformer.

        Nothing in the result is ever ``0.0`` for something that was not
        measured. Unmeasured is ``None`` with a ``reason``.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: noise_floor_from_runs. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    from .scorers import (
        DEFAULT_REFUSAL_SCORER,
        DEFAULT_SENTIMENT_SCORER,
        DEFAULT_TOXICITY_SCORER,
        scorer_provenance,
    )

    metric_names = list(metrics) if metrics else list(RUN_METRICS)
    unknown = [m for m in metric_names if m not in RUN_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}. Must be a subset of {list(RUN_METRICS)}")
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    sampling_record = dict(sampling or {})
    if not responses_by_variant:
        return {
            "available": False,
            "state": "could_not_check",
            "reason": "no_variants",
            "series_kind": "per_run_responses",
            "sampling": sampling_record,
            "variants": {},
            "metrics": {},
            "limitations": ["No variants were supplied; nothing was measured."],
        }

    scorer_map: Dict[str, Any] = {
        "sentiment": DEFAULT_SENTIMENT_SCORER,
        "toxicity": DEFAULT_TOXICITY_SCORER,
        "refusal_rate": DEFAULT_REFUSAL_SCORER,
    }
    if scorers:
        scorer_map.update(scorers)

    analyzer = NonDeterminismAnalyzer(system_type=system_type, min_runs=min_runs)
    required = analyzer.required_runs()

    def _series(metric: str, texts: List[str]) -> np.ndarray:
        if metric == "response_length":
            return np.array([_word_count(t) for t in texts], dtype=np.float64)
        return np.asarray(scorer_map[metric].score_batch(list(texts)), dtype=np.float64)

    variants_out: Dict[str, Dict[str, Any]] = {}
    series_store: Dict[Tuple[str, str], np.ndarray] = {}
    limitations: List[str] = []

    for label, responses in responses_by_variant.items():
        rows = list(responses or [])
        valid = [r for r in rows if isinstance(r, str) and r.strip()]
        n_runs, n_valid = len(rows), len(valid)
        entry: Dict[str, Any] = {
            "n_runs": n_runs,
            "n_valid": n_valid,
            "empty_rate": (1.0 - n_valid / n_runs) if n_runs else float("nan"),
            "metrics": {},
        }
        if n_valid < 2:
            entry["state"] = "could_not_check"
            entry["reason"] = "fewer_than_2_valid_responses"
            limitations.append(
                f"Variant '{label}': {n_valid} valid response(s) of {n_runs}; a noise floor "
                f"needs at least two repeated responses to the same prompt."
            )
            variants_out[label] = entry
            continue
        if n_valid < required:
            entry["state"] = "measured_with_limitation"
            entry["reason"] = f"below_recommended_runs ({n_valid} < {required})"
            limitations.append(
                f"Variant '{label}': {n_valid} valid runs, below the recommended {required} "
                f"for system_type='{system_type}'. The floor is measured, and less certain."
            )
        else:
            entry["state"] = "measured"
            entry["reason"] = None

        for metric in metric_names:
            try:
                s = _series(metric, valid)
            except Exception as exc:  # a scorer backend is optional; say which failed
                entry["metrics"][metric] = {
                    "state": "could_not_check",
                    "reason": f"scorer_failed: {type(exc).__name__}: {exc}",
                }
                continue
            # BGL-S2b (2026-09-17), COVERAGE REGRESSION. ``not
            # np.all(np.isfinite(s))`` discarded the WHOLE variant on a single
            # unreadable run, with no count anywhere, which is this audit's own
            # defect running backwards: instead of fabricating a value it threw
            # away established signal and reported "could not check". Measured
            # at this entry with 26 runs, 25 genuinely abusive and one emoji:
            # toxicity went from observed=-0.48 / exceeds_noise=True to
            # {"state": "could_not_check", "reason":
            # "series_not_finite_or_too_short"} with ``limitations == []``, so
            # 25 real measurements vanished silently. characterize_noise and
            # bootstrap_ci already drop non-finite rows, count them and warn;
            # this is the same treatment in their caller, which has to do it
            # itself because the comparison below also computes np.var and
            # np.mean over the stored series.
            n_supplied_runs = int(s.size)
            s_finite = s[np.isfinite(s)]
            n_excluded_non_finite = n_supplied_runs - int(s_finite.size)
            if s_finite.size < 2:
                entry["metrics"][metric] = {
                    "state": "could_not_check",
                    "reason": "series_not_finite_or_too_short",
                    "n_supplied": n_supplied_runs,
                    "n_scored": int(s_finite.size),
                    "n_excluded_non_finite": n_excluded_non_finite,
                }
                limitations.append(
                    f"Variant '{label}', metric '{metric}': {int(s_finite.size)} of "
                    f"{n_supplied_runs} run(s) carry a measurement; a noise floor needs at "
                    f"least two, so this metric was NOT measured for this variant."
                )
                continue
            if n_excluded_non_finite:
                limitations.append(
                    f"Variant '{label}', metric '{metric}': {n_excluded_non_finite} of "
                    f"{n_supplied_runs} run(s) could not be scored and are EXCLUDED. The "
                    f"floor rests on the {int(s_finite.size)} that remain, which is a "
                    f"SUBSET, and says nothing about the excluded runs. Attrition that "
                    f"differs between variants makes the comparison below a comparison of "
                    f"two differently selected samples: read n_excluded_non_finite on each."
                )
            s = s_finite
            profile = analyzer.characterize_noise(s)
            lo, hi = analyzer.bootstrap_ci(s, confidence=confidence, random_state=random_state)
            series_store[(label, metric)] = s
            entry["metrics"][metric] = {
                "state": ("measured_with_limitation" if n_excluded_non_finite else entry["state"]),
                "profile": profile.to_dict(),
                "mean_ci": [float(lo), float(hi)],
                "n_supplied": n_supplied_runs,
                "n_scored": int(s.size),
                "n_excluded_non_finite": n_excluded_non_finite,
            }
        variants_out[label] = entry

    labels = list(responses_by_variant.keys())
    reference = labels[0]
    metrics_out: Dict[str, Dict[str, Any]] = {}
    for metric in metric_names:
        ref_series = series_store.get((reference, metric))
        provenance = scorer_provenance(
            metric, None if metric == "response_length" else scorer_map.get(metric)
        )
        comparisons: List[Dict[str, Any]] = []
        for other in labels[1:]:
            other_series = series_store.get((other, metric))
            if ref_series is None or other_series is None:
                comparisons.append(
                    {
                        "variant": other,
                        "state": "could_not_check",
                        "reason": "series_missing_for_reference_or_variant",
                        "observed": None,
                        "disparity_noise_floor": None,
                        "exceeds_noise": None,
                        "systematic_offset": None,
                    }
                )
                continue
            var_ref = float(np.var(ref_series, ddof=1))
            var_other = float(np.var(other_series, ddof=1))
            floor = 2.0 * math.sqrt(var_ref / ref_series.size + var_other / other_series.size)
            observed = float(np.mean(ref_series) - np.mean(other_series))
            offset = analyzer.compute_noise_offset(observed, floor, confidence)
            if offset["state"] != "measured":
                # NAN-01: never cast a withheld grade. `bool(None)` is False,
                # which is exactly the "within noise" answer the offset just
                # refused to give.
                comparisons.append(
                    {
                        "variant": other,
                        "state": "could_not_check",
                        "reason": offset["reason"],
                        "observed": observed,
                        "disparity_noise_floor": floor,
                        "exceeds_noise": None,
                        "systematic_offset": None,
                    }
                )
                continue
            comparison: Dict[str, Any] = {
                "variant": other,
                "state": "measured",
                "reason": None,
                "observed": observed,
                "disparity_noise_floor": floor,
                "exceeds_noise": bool(offset["exceeds_noise"]),
                "systematic_offset": float(offset["systematic_offset"]),
                "z_threshold": float(offset["z_threshold"]),
                # Coverage travels with the comparison (BGL-S2b, 2026-09-17).
                # Unreadable runs are now excluded and measured around rather
                # than discarding the variant, so the number of runs each side
                # actually rests on has to be readable beside the number: a run
                # is unscorable because of its CONTENT, so uneven attrition
                # means these two means come from differently selected samples.
                "n_scored_reference": int(ref_series.size),
                "n_scored_variant": int(other_series.size),
                "n_excluded_reference": int(
                    variants_out[reference]["metrics"]
                    .get(metric, {})
                    .get("n_excluded_non_finite", 0)
                ),
                "n_excluded_variant": int(
                    variants_out[other]["metrics"].get(metric, {}).get("n_excluded_non_finite", 0)
                ),
            }
            if sesoi is not None:
                try:
                    comparison["equivalence"] = analyzer.equivalence_test(
                        ref_series, other_series, sesoi=sesoi, alpha=alpha
                    )
                except ValueError as exc:
                    comparison["equivalence"] = {
                        "verdict": "could_not_check",
                        "interpretation": f"COULD NOT CHECK: {exc}",
                        "sesoi": sesoi,
                    }
            comparisons.append(comparison)
        measured_any = any(c["state"] == "measured" for c in comparisons)
        metrics_out[metric] = {
            "state": "measured" if measured_any else "could_not_check",
            "reference_variant": reference,
            "comparisons": comparisons,
            **provenance,
        }

    if len(labels) < 2:
        limitations.append(
            "Only one variant was supplied: per-variant noise is measured, but there is "
            "no second variant to compare it against."
        )

    overall = (
        "measured"
        if any(m["state"] == "measured" for m in metrics_out.values())
        else ("could_not_check")
    )
    return {
        "available": overall == "measured",
        "state": overall,
        "reason": None if overall == "measured" else "no_metric_could_be_compared",
        "series_kind": "per_run_responses",
        "sampling": sampling_record,
        "system_type": system_type,
        "required_runs": required,
        "variants": variants_out,
        "metrics": metrics_out,
        "limitations": limitations,
    }
