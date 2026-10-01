"""
Unit 4: A/B Testing for Fairness: FairnessPowerAnalyzer
========================================================

Power analysis, sequential testing (SPRT), and adaptive sampling for
fairness-aware experimentation.

Traditional power analysis asks *"Do we have enough data?"* for a single
test.  **FairnessPowerAnalyzer** asks the same question *per demographic
intersection* and provides tools to:

- Compute required sample sizes so that every intersection is adequately
  powered.
- Run sequential probability ratio tests (SPRT) for early stopping.
- Generate adaptive sampling plans that allocate budget to underpowered
  intersections first.
- Calculate the minimum detectable effect (MDE) at each intersection
  given current data.

References
----------
Wald, A. (1945).  Sequential tests of statistical hypotheses.
Deng, A. et al. (2016).  Continuous monitoring of A/B tests without pain.
Module 4, Part 4, Unit 4: A/B Testing for Fairness.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...evaluation.vfairness_metrics._statistics import (
    _norm_cdf,
    _norm_ppf,
)
from ...exceptions import ConfigurationError

try:
    from scipy import stats as scipy_stats  # noqa: F401  # availability probe

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Type alias for intersection tuples
Intersection = Tuple


# Config & Result Dataclasses


@dataclass
class PowerConfig:
    """Configuration for :class:`FairnessPowerAnalyzer`.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    target_power : float
        Desired statistical power (default 0.80).
    effect_sizes : list[float], optional
        Cohen's d values to evaluate. Defaults to [0.2, 0.5, 0.8].
    correction_method : str
        Multiple comparison correction method (``'fdr'``, ``'bonferroni'``).
    min_group_size : int
        Minimum per-arm group size to consider an intersection.
    """

    alpha: float = 0.05
    target_power: float = 0.80
    effect_sizes: Optional[List[float]] = None
    correction_method: str = "fdr"
    min_group_size: int = 30

    def __post_init__(self):
        # alpha=0 and target_power=1 are not conservative requests, they are
        # impossible ones, and the SPRT below used to answer them with an
        # invented boundary rather than refusing. See sequential_test.
        if not isinstance(self.alpha, (int, float)) or not 0.0 < float(self.alpha) < 1.0:
            raise ConfigurationError(
                f"alpha must be strictly between 0 and 1, got {self.alpha!r}. "
                f"alpha=0 asks for a test that can never reject."
            )
        if (
            not isinstance(self.target_power, (int, float))
            or not 0.0 < float(self.target_power) < 1.0
        ):
            raise ConfigurationError(
                f"target_power must be strictly between 0 and 1, got {self.target_power!r}."
            )
        if self.effect_sizes is None:
            self.effect_sizes = [0.2, 0.5, 0.8]
        elif list(self.effect_sizes) != [0.2, 0.5, 0.8]:
            # A DOCUMENTED KNOB THAT NOTHING READS (G04, 2026-09-30). Measured
            # that day: `config.effect_sizes` appears in this file only at its
            # declaration, its docstring and the default above, and nowhere
            # else in src/. Every method that needs a Cohen's d takes it as a
            # PARAMETER (`required_sample_size(effect_size=...)`,
            # `power_for_sample_size`, `sequential_test`, `adaptive_sampling_plan`,
            # `minimum_detectable_effect`), so a caller who sets this field and
            # then calls them gets the parameter defaults, silently. Whether the
            # field should drive those defaults or leave the public config is a
            # product decision, so this release makes the no-op AUDIBLE rather
            # than choosing; delete this branch when the field is wired up.
            warnings.warn(
                f"PowerConfig.effect_sizes={list(self.effect_sizes)!r} is NOT READ by any "
                f"FairnessPowerAnalyzer method in this release: each of them takes its own "
                f"`effect_size` parameter and falls back to that parameter's default, not "
                f"to this field. Setting it changes nothing in the analysis. Pass "
                f"effect_size= to the method you are calling.",
                UserWarning,
                stacklevel=3,
            )


@dataclass
class PowerResult:
    """Power analysis result for a single intersection.

    Attributes
    ----------
    intersection : tuple
        Demographic intersection values.
    power : float
        Achieved statistical power (0–1), or NaN when it was not computed
        (the intersection is below ``min_group_size``).
    required_n : int
        Per-arm sample size needed for ``target_power``.
    is_powered : bool or None
        ``True`` when ``power >= target_power``; ``None`` when power was not
        computed, which is not the same as underpowered.
    n_control : int
        Control arm OBSERVATIONS: rows that carry an outcome value. BGL-S2
        changed the noun this counts from rows to observations without saying
        so on the field, so a reader comparing two runs could not tell whether
        a drop meant lost rows or lost outcome values. Read it beside
        ``n_rows_control``.
    n_treatment : int
        Treatment arm OBSERVATIONS, same change of noun.
    n_rows_control : int
        Control arm ROWS, regardless of missingness. ``60 rows, 0
        observations`` names the problem; either number alone hides half of it.
    n_rows_treatment : int
        Treatment arm ROWS, regardless of missingness.
    effect_size : float
        Effect size (Cohen's d) used for the calculation.
    """

    intersection: Tuple
    power: float
    required_n: int
    is_powered: Optional[bool]
    n_control: int = 0
    n_treatment: int = 0
    effect_size: float = 0.2
    n_rows_control: int = 0
    n_rows_treatment: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intersection": list(self.intersection),
            "power": (round(self.power, 4) if np.isfinite(self.power) else float("nan")),
            "required_n": self.required_n,
            "is_powered": self.is_powered,
            # OBSERVATIONS, not rows: see the class docstring.
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "n_rows_control": self.n_rows_control,
            "n_rows_treatment": self.n_rows_treatment,
            "effect_size": self.effect_size,
        }


class SPRTDecision(Enum):
    """Decision from a sequential probability ratio test."""

    ACCEPT_NULL = "accept_null"
    REJECT_NULL = "reject_null"
    CONTINUE = "continue"
    #: The test could not run on this data at all. READINESS-6, 2026-09-10.
    #: Distinct from CONTINUE, which says "collect more and look again": when
    #: both arms are constant, more of the same data will never help, so
    #: reporting CONTINUE would send a reader back for observations that cannot
    #: change the answer.
    COULD_NOT_CHECK = "could_not_check"


@dataclass
class SequentialTestResult:
    """Result of a sequential probability ratio test (SPRT).

    Attributes
    ----------
    intersection : tuple
        Demographic intersection.
    decision : SPRTDecision
        Stop-or-continue decision.
    log_likelihood_ratio : float
        Current log-likelihood ratio.
    upper_boundary : float
        Upper stopping boundary (reject H₀).
    lower_boundary : float
        Lower stopping boundary (accept H₀).
    stopped_early : bool
        Whether the test stopped before consuming all data.
    n_observations : int
        Number of observations consumed.
    """

    intersection: Tuple
    decision: SPRTDecision
    log_likelihood_ratio: float
    upper_boundary: float
    lower_boundary: float
    stopped_early: bool
    n_observations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intersection": list(self.intersection),
            "decision": self.decision.value,
            "log_likelihood_ratio": round(self.log_likelihood_ratio, 4),
            "upper_boundary": round(self.upper_boundary, 4),
            "lower_boundary": round(self.lower_boundary, 4),
            "stopped_early": self.stopped_early,
            "n_observations": self.n_observations,
        }


@dataclass
class SamplingPlan:
    """Adaptive sampling allocation plan.

    Attributes
    ----------
    allocations : dict[tuple, int]
        Recommended additional samples per intersection.
    rationale : dict[tuple, str]
        Human-readable explanation per intersection.
    priority_order : list[tuple]
        Intersections ordered by sampling priority (most urgent first).
    total_budget : int
        Total budget across all allocations.
    n_underpowered : int
        Number of underpowered intersections.
    not_allocated : list[tuple]
        Underpowered intersections the budget was exhausted before reaching.
        They are still underpowered: they are NOT "already powered", they carry
        no entry in ``allocations``, and their ``additional_samples`` in
        :meth:`to_dataframe` is ``None``, never 0 (R-1, 2026-09-09).
    not_assessed : list[tuple]
        Intersections whose power was never computed, either because they sit
        below the analyzer's ``min_group_size`` floor (readiness 6,
        2026-09-10) or because an arm holds no OBSERVATION at all, i.e. rows
        are present but none carries an outcome value (BGL-S2, 2026-09-16).
        They are neither powered nor underpowered: nobody measured them, and
        their rationale says so instead of "already powered". They also carry a
        row in :meth:`to_dataframe`, with ``additional_samples`` of ``None``
        (G-023, 2026-09-17): the frame is the surface a caller renders, and
        until then an unmeasured intersection was simply absent from it.
    """

    allocations: Dict[Tuple, int]
    rationale: Dict[Tuple, str]
    priority_order: List[Tuple]
    total_budget: int = 0
    n_underpowered: int = 0
    not_allocated: List[Tuple] = field(default_factory=list)
    not_assessed: List[Tuple] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": {str(k): v for k, v in self.allocations.items()},
            "rationale": {str(k): v for k, v in self.rationale.items()},
            "priority_order": [list(t) for t in self.priority_order],
            "total_budget": self.total_budget,
            "n_underpowered": self.n_underpowered,
            "not_allocated": [list(t) for t in self.not_allocated],
            "n_not_allocated": len(self.not_allocated),
            "not_assessed": [list(t) for t in self.not_assessed],
            "n_not_assessed": len(self.not_assessed),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the plan to a DataFrame.

        One row per underpowered intersection (``priority_order``) followed by
        one per NOT ASSESSED intersection, so the row count is
        ``n_underpowered + len(not_assessed)``. ``additional_samples`` is
        ``None`` both for an intersection the budget did not reach (see
        ``not_allocated``) and for one nobody measured; the ``rationale``
        column says which, beginning "NOT ALLOCATED:" or "NOT ASSESSED:".

        G-023 (2026-09-17): the not-assessed rows are new here. The third state
        was computed correctly, reached ``not_assessed`` and ``to_dict``, and
        was then invisible at this surface, which is the one a caller renders
        as "the plan". Measured that day on an experiment whose 'f'
        intersection held 60 rows per arm and no outcome value in any of them:
        ``to_dict`` carried ``n_not_assessed: 1``, while ``to_dataframe``
        returned a single row for 'm' and no trace that 'f' existed. A reader
        of the frame alone would have concluded 'f' needed no action, when the
        action it needs (record the outcome) is the most urgent one in the
        plan. The docstring this replaces predates ``not_assessed``.
        """
        rows: List[Dict[str, Any]] = []
        for ix in self.priority_order:
            rows.append(
                {
                    "intersection": str(ix),
                    # R-1 (2026-09-09): `.get(ix, 0)` read a MISSING allocation as
                    # a measured zero, the number a powered intersection gets.
                    "additional_samples": self.allocations.get(ix),
                    "rationale": self.rationale.get(ix, ""),
                }
            )
        for ix in self.not_assessed:
            rows.append(
                {
                    "intersection": str(ix),
                    "additional_samples": None,
                    "rationale": self.rationale.get(ix, ""),
                }
            )
        if not rows:
            # G04 (2026-09-30): an empty plan returned pd.DataFrame(), shape
            # (0, 0) with no columns at all, so a caller rendering the frame got
            # no schema to render. The empty case is legitimate here (nothing
            # underpowered and nothing unassessed is a real "no action needed",
            # unlike the empty CATE frame in `analysis`, which hid an excluded
            # count), so it carries the columns and no warning.
            return pd.DataFrame(columns=list(_PLAN_COLUMNS))
        return pd.DataFrame(rows)


#: Column order of :meth:`SamplingPlan.to_dataframe`, so an EMPTY plan can carry
#: the same schema instead of a frame with no columns (G04, 2026-09-30).
_PLAN_COLUMNS: Tuple[str, ...] = ("intersection", "additional_samples", "rationale")


def _validate_effect_size(effect_size: float, method: str) -> float:
    """Refuse an effect size no design can be built around.

    G-023 (2026-09-17). ``effect_size <= 0`` was already refused by
    :meth:`FairnessPowerAnalyzer.required_sample_size`, but the comparison let
    two values through that are not effect sizes at all, and the siblings that
    take the same parameter refused nothing. Measured before this change on a
    60-per-arm experiment:

        required_sample_size(inf)      -> {('f',): 0}     "you need no samples"
        required_sample_size(nan)      -> ValueError: cannot convert float NaN
        power_for_sample_size(0.0)     -> {('f',): 0.025}
        power_for_sample_size(inf)     -> {('f',): 1.0}   "perfect power"
        power_for_sample_size(nan)     -> {('f',): nan}   silently, no warning

    0.025 is alpha/2, the false-positive rate, printed in the column a reader
    reads as achieved power; 0 required samples and a power of 1.0 are
    instructions a caller can act on. None of the three is a measurement of
    anything. The refusal message for a non-positive size is kept verbatim
    because it names the reason a caller needs.
    """
    if isinstance(effect_size, (int, float)) and math.isnan(float(effect_size)):
        raise ConfigurationError(
            f"{method}: effect_size must be a finite number > 0, got {effect_size!r}. "
            f"A missing effect size is not an effect size of zero, so there is no "
            f"design to report."
        )
    if not math.isfinite(float(effect_size)):
        raise ConfigurationError(
            f"{method}: effect_size must be a finite number > 0, got {effect_size!r}. "
            f"An infinite effect is detectable by any design, so the answer would be "
            f"a statement about the input, not about these data."
        )
    if effect_size <= 0:
        raise ConfigurationError(
            f"effect_size must be > 0, got {effect_size!r}. No finite sample size "
            f"can detect an effect of zero, so there is no required n to return."
        )
    return float(effect_size)


def _power_label(power: Optional[float]) -> str:
    """A power figure for a rationale string, or the words for one that was
    never computed. `f"{nan:.2f}"` prints "nan", which reads as a measurement
    that went wrong rather than one that was never made."""
    if power is None or not np.isfinite(power):
        # G-023: two reasons reach this line, so the label names both rather
        # than asserting the floor. With min_group_size=0 an arm holding no
        # observation is not "below the floor", and saying so would be a false
        # explanation of a true refusal.
        return "not computed (no observations, or below the min_group_size floor)"
    return f"{power:.2f}"


class FairnessPowerAnalyzer:
    """Power analysis and sequential testing for fairness experiments.

    Works in concert with :class:`FairnessExperiment` to answer:

    * *How many samples do we need per intersection?*
    * *Can we stop early if effects are clear?*
    * *Where should we focus additional data collection?*

    Parameters
    ----------
    experiment : FairnessExperiment
        A configured (but not necessarily analysed) experiment instance.
    config : PowerConfig, optional
        Power analysis configuration.

    Examples
    --------
    >>> analyzer = FairnessPowerAnalyzer(experiment)
    >>> summary = analyzer.get_power_summary()
    >>> plan = analyzer.adaptive_sampling_plan(budget=500)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: power_analysis. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        experiment,  # FairnessExperiment: no type annotation to avoid circular
        config: Optional[PowerConfig] = None,
    ) -> None:
        self.experiment = experiment
        self.config = config or PowerConfig()

    # Helper: get per-intersection sample sizes

    def _intersection_sizes(self) -> Dict[Intersection, Tuple[int, int]]:
        """Return (n_control, n_treatment) OBSERVATIONS for every intersection.

        An observation is a row that actually carries an outcome value. Every
        public method on this class derives its n from here, so the noun this
        function counts decides what the whole class means by "sample size".

        BGL-S2 (2026-09-16). It used to count ROWS: ``len(df_slice)``, never
        touching the outcome column. A power analysis is a statement about
        observations, so an intersection with 60 rows and no outcome value in
        any of them was reported as n=60 and graded. Measured before this
        change on three datasets identical in shape (60 rows per intersection
        per arm) and differing ONLY in how many outcome values existed, 240
        real, 5 real, and 0 real:

            power_for_sample_size(0.5)  0.7818   0.7818   0.7818
            required_sample_size(0.5)   63       63       63
            minimum_detectable_effect() 0.5115   0.5115   0.5115
            plan.n_underpowered         4        4        4
            plan.not_assessed           []       []       []
            get_power_summary n_control 60       60       60
            warnings                    none     none     none

        The answers were invariant to whether any observation existed at all,
        and the sampling plan's rationale string said "have 60" for an
        intersection holding nothing. The partial case (5 of 240) matters more
        than the all-empty one: ordinary missingness, not a contrived column,
        produced the same confident numbers.

        ``sequential_test`` was already honest here because it dropna()s the
        outcome itself before computing, which is why its arm counts and this
        function's disagreed by design; they now agree.
        """
        sizes: Dict[Intersection, Tuple[int, int]] = {}
        outcome = self.experiment.outcome
        empty: List[Intersection] = []
        for ix in self.experiment.intersections:
            n_c = self._count_observations(self.experiment.control, ix, outcome)
            n_t = self._count_observations(self.experiment.treatment, ix, outcome)
            sizes[ix] = (n_c, n_t)
            if min(n_c, n_t) == 0:
                empty.append(ix)

        if empty:
            row_counts = self._intersection_row_counts()
            detail = ", ".join(
                f"{ix}: {row_counts.get(ix, (0, 0))[0]}/{row_counts.get(ix, (0, 0))[1]} rows "
                f"carry {sizes[ix][0]}/{sizes[ix][1]} outcome values"
                for ix in empty[:5]
            )
            warnings.warn(
                f"FairnessPowerAnalyzer: {len(empty)} intersection(s) have an arm with no "
                f"outcome observation, so no power and no MDE was computed for "
                f"them ({detail}). They report NaN and is_powered=None (could not check), "
                f"NOT a power figure derived from the row count. required_sample_size IS "
                f"still returned for them: it is a function of the effect size, alpha and "
                f"target power only, and says nothing about these rows.",
                UserWarning,
                stacklevel=3,
            )
        return sizes

    def _count_observations(self, arm: pd.DataFrame, ix: Intersection, outcome: str) -> int:
        """Rows in one arm of one intersection that carry an outcome value."""
        slice_ = self.experiment._get_intersection_data(arm, ix)
        if outcome not in slice_.columns:
            # No outcome column at all is a could-not-check, not a full arm.
            return 0
        return int(slice_[outcome].notna().sum())

    def _intersection_row_counts(self) -> Dict[Intersection, Tuple[int, int]]:
        """Return (rows_control, rows_treatment) regardless of missingness.

        Kept beside :meth:`_intersection_sizes` so a reader can see BOTH
        numbers: "60 rows, 0 observations" names the problem, while either
        number alone hides half of it.
        """
        counts: Dict[Intersection, Tuple[int, int]] = {}
        for ix in self.experiment.intersections:
            counts[ix] = (
                len(self.experiment._get_intersection_data(self.experiment.control, ix)),
                len(self.experiment._get_intersection_data(self.experiment.treatment, ix)),
            )
        return counts

    # Required sample size

    def required_sample_size(
        self,
        effect_size: float = 0.2,
    ) -> Dict[Intersection, int]:
        """Per-arm sample size needed to achieve ``target_power``.

        Inverts the two-sample z-test power formula:
        ``n = 2 × ((z_{α/2} + z_{β}) / d)²``

        where *d* is the standardised effect size and ``1 − β`` is the
        target power.

        Parameters
        ----------
        effect_size : float
            Expected standardised effect size (Cohen's d).

        Returns
        -------
        dict[tuple, int]
            Required per-arm *n* for each intersection.

        Raises
        ------
        ConfigurationError
            If ``effect_size`` is not strictly positive. Asking how many
            samples are needed to detect an effect of zero has no answer, and
            this used to be answered with the integer ``999_999`` under a
            warning that said "returning inf": the surface was more confident
            than the code, ``math.isinf`` on the returned value was False, and
            999999 is a number a caller can budget against. The refusal matches
            :meth:`PowerConfig.__post_init__`, which already refuses an
            impossible alpha rather than inventing a boundary for it.

            NaN and infinity are refused for the same reason (G-023,
            2026-09-17): ``inf`` used to return a required n of 0, and NaN
            escaped the ``<= 0`` comparison and died in ``int()`` with
            "cannot convert float NaN to integer".
        """
        z_alpha = _norm_ppf(1 - self.config.alpha / 2)
        z_beta = _norm_ppf(self.config.target_power)

        effect_size = _validate_effect_size(effect_size, "required_sample_size")

        n_required = int(math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))

        # Same formula for every intersection (homogeneous variance
        # assumption); the dict shows which intersections currently meet it.
        sizes = self._intersection_sizes()
        results: Dict[Intersection, int] = {}
        for ix, (n_c, n_t) in sizes.items():
            results[ix] = n_required
        return results

    # Achieved power for current data

    def power_for_sample_size(
        self,
        effect_size: float = 0.2,
    ) -> Dict[Intersection, float]:
        """Achieved statistical power given current sample sizes.

        Uses the z-test approximation:
        ``power = Φ(d × √(n/2) − z_{α/2})``

        Parameters
        ----------
        effect_size : float
            Expected standardised effect size (Cohen's d).

        Returns
        -------
        dict[tuple, float]
            Power (0–1) per intersection, or NaN where no power was computed.

        Raises
        ------
        ConfigurationError
            If ``effect_size`` is not a finite number greater than zero. See
            :func:`_validate_effect_size` for the values this used to answer.
        """
        z_alpha = _norm_ppf(1 - self.config.alpha / 2)
        effect_size = _validate_effect_size(effect_size, "power_for_sample_size")
        sizes = self._intersection_sizes()
        results: Dict[Intersection, float] = {}

        for ix, (n_c, n_t) in sizes.items():
            n_eff = min(n_c, n_t)
            # G-023 (2026-09-17): `n_eff == 0` first, and independently of the
            # floor. `min_group_size` is not validated, and at 0 an arm holding
            # NO OBSERVATION cleared the comparison and was handed
            # Phi(d*sqrt(0/2) - z) = 0.025, which is alpha/2 (the false-positive
            # rate) presented as achieved power. Measured that day on 60 rows
            # whose outcome was NaN in every one: power 0.025 while the warning
            # raised two frames up promised "They report NaN and
            # is_powered=None (could not check)". The surface contradicted its
            # own warning.
            if n_eff == 0 or n_eff < self.config.min_group_size:
                # NOT a power of zero. Below the size floor this method does not
                # compute power at all, and 0.0 is a measurement on the power
                # scale: it read as "underpowered" in is_powered while
                # adaptive_sampling_plan, comparing the same intersection
                # against required_sample_size, called it "already powered" in
                # the same breath. Measured 2026-09-10 at 28 per arm against a
                # floor of 30 and a requirement of 25: power 0.0, mde inf,
                # is_powered False, rationale "already powered",
                # n_underpowered 0, and no warning anywhere.
                results[ix] = float("nan")
            else:
                results[ix] = round(float(_norm_cdf(effect_size * np.sqrt(n_eff / 2) - z_alpha)), 4)
        return results

    # Sequential probability ratio test (SPRT)

    def sequential_test(
        self,
        effect_size: float = 0.2,
    ) -> Dict[Intersection, SequentialTestResult]:
        """Run a sequential probability ratio test per intersection.

        Implements Wald's SPRT with boundaries derived from ``alpha`` and
        ``target_power``:

        * Upper boundary  ``B = log((1 − β) / α)``  → reject H₀
        * Lower boundary  ``A = log(β / (1 − α))``  → accept H₀

        For each intersection, the observed standardised mean difference x̄
        (variance 1/n_c + 1/n_t) is converted to the Wald log-likelihood
        ratio under H₀ (δ = 0) vs H₁ (δ = effect_size):
        ``Λ = (d·x̄ − d²/2) / (1/n_c + 1/n_t)``.

        Parameters
        ----------
        effect_size : float
            Effect size under the alternative hypothesis.

        Returns
        -------
        dict[tuple, SequentialTestResult]

        Raises
        ------
        ConfigurationError
            If ``effect_size`` is not a finite number greater than zero. An
            alternative hypothesis of zero effect is the null, so the ratio
            between them carries no information (G-023, 2026-09-17).
        """
        effect_size = _validate_effect_size(effect_size, "sequential_test")
        alpha = self.config.alpha
        beta = 1 - self.config.target_power

        # Wald boundaries (log scale). The `if alpha > 0 else 10` fallback that
        # used to stand here INVERTED the meaning of the parameter: measured
        # 2026-09-10 on a 2.1-sigma treatment effect over 60 per arm,
        # alpha=0.0 produced upper_boundary=10 and so decision=reject_null with
        # stopped_early=True, while alpha=1e-12 (a strictly weaker demand)
        # produced 27.41 and correctly refused to stop. A request for zero
        # type-I error was answered by the LOWEST bar of any alpha on the scale.
        # PowerConfig now refuses alpha outside (0, 1) at construction, so both
        # divisions here are safe and there is no fabricated boundary to reach.
        upper_boundary = math.log((1 - beta) / alpha)
        lower_boundary = math.log(beta / (1 - alpha))

        sizes = self._intersection_sizes()
        results: Dict[Intersection, SequentialTestResult] = {}

        for ix, (n_c, n_t) in sizes.items():
            n_eff = min(n_c, n_t)
            if n_eff == 0:
                # BGL-S2b (2026-09-17). An arm with NO OBSERVATION has no
                # likelihood, so it has no likelihood RATIO. This returned
                # decision=CONTINUE with log_likelihood_ratio=0.0, and 0.0 is a
                # value on the LLR scale, the exact midpoint between the two
                # Wald boundaries: "the evidence so far is perfectly balanced"
                # for a test that was never run. (Before the S2 change it
                # raised ValueError here, so the refusal was at least audible.)
                # COULD_NOT_CHECK with a NaN ratio is the state this same
                # method already uses for the constant-arm case twenty lines
                # down; it is used here too rather than inventing a fourth.
                #
                # A small but NON-EMPTY arm keeps CONTINUE with 0.0 below: there
                # the SPRT genuinely has observations and genuinely has not
                # accumulated enough evidence to stop, which is what CONTINUE
                # means. Refusing that too would be an over-correction.
                warnings.warn(
                    f"sequential_test{ix}: one arm carries no outcome observation "
                    f"({n_c} control, {n_t} treatment), so no likelihood ratio exists. "
                    f"Reporting decision=could_not_check and "
                    f"log_likelihood_ratio=nan, NOT 0.0, which is the midpoint of the "
                    f"Wald boundaries and reads as balanced evidence.",
                    UserWarning,
                    stacklevel=2,
                )
                results[ix] = SequentialTestResult(
                    intersection=ix,
                    decision=SPRTDecision.COULD_NOT_CHECK,
                    log_likelihood_ratio=float("nan"),
                    upper_boundary=round(upper_boundary, 4),
                    lower_boundary=round(lower_boundary, 4),
                    stopped_early=False,
                    n_observations=n_c + n_t,
                )
                continue
            if n_eff < self.config.min_group_size:
                # G-023 (2026-09-17). The DECISION here is right and is kept:
                # below the size floor the instruction genuinely is "collect
                # more and look again", which is what CONTINUE means. The
                # RATIO was not. This returned log_likelihood_ratio=0.0 with
                # no warning for a test that never ran, and 0.0 is the exact
                # midpoint of the two Wald boundaries, i.e. "the evidence so
                # far is perfectly balanced". Measured that day at 10
                # observations per arm against a floor of 30, carrying a real
                # 2-sigma treatment gap: decision continue, llr 0.0, silence.
                # Had the SPRT actually run on those observations it would have
                # crossed the upper boundary and rejected, so 0.0 was not a
                # harmless placeholder: it reported balance where the data, if
                # anyone had looked, leaned hard one way.
                #
                # This is the sibling of the n_eff == 0 branch above, which was
                # fixed for the identical reason one wave earlier and left this
                # copy standing.
                warnings.warn(
                    f"sequential_test{ix}: {n_eff} observation(s) in the smaller arm is "
                    f"below the min_group_size floor of {self.config.min_group_size}, so "
                    f"no likelihood ratio was computed. Reporting "
                    f"log_likelihood_ratio=nan, NOT 0.0, which is the midpoint of the "
                    f"Wald boundaries and reads as balanced evidence. The decision stays "
                    f"'continue': more observations here will change the answer.",
                    UserWarning,
                    stacklevel=2,
                )
                results[ix] = SequentialTestResult(
                    intersection=ix,
                    decision=SPRTDecision.CONTINUE,
                    log_likelihood_ratio=float("nan"),
                    upper_boundary=upper_boundary,
                    lower_boundary=lower_boundary,
                    stopped_early=False,
                    n_observations=n_c + n_t,
                )
                continue

            ctrl = (
                self.experiment._get_intersection_data(
                    self.experiment.control,
                    ix,
                )[self.experiment.outcome]
                .dropna()
                .values.astype(float)
            )
            treat = (
                self.experiment._get_intersection_data(
                    self.experiment.treatment,
                    ix,
                )[self.experiment.outcome]
                .dropna()
                .values.astype(float)
            )

            # Pooled std for variance normalisation. An arm holding fewer than
            # two observations carries NO degrees of freedom, so it drops out of
            # the pooled variance and the other arm's variance IS the pooled
            # one: with n_ctrl = 1 the textbook formula reduces exactly to
            # s_treat**2. That is a measurement and it is kept, which is why
            # this computes the pooled variance rather than refusing here.
            #
            # G-023 (2026-09-17): the `else 1.0` this replaces was not a
            # measurement. It read as a harmless fallback for a short arm and
            # was an INVENTED SCALE, the same shape READINESS-6 describes below,
            # reached through a different door. Measured that day with
            # min_group_size=1, one control observation at 0.0, and 60 treatment
            # observations of mean 6.0 and sd 10.8 (a gap of 0.55 sd):
            #
            #   reject_null, stopped_early=True, llr 2.8279 (boundary 2.7726)
            #
            # while the SAME treatment arm against a two-observation control,
            # where the pooled sd is measured, returns continue with llr 0.30.
            # A single-observation arm bought a "stop, the treatment works"
            # verdict out of half a standard deviation, because the unit the
            # difference was divided by was 1.0 by decree.
            pooled_ss = 0.0
            pooled_dof = 0
            for _arm in (ctrl, treat):
                if len(_arm) > 1:
                    pooled_ss += float(_arm.var(ddof=1)) * (len(_arm) - 1)
                    pooled_dof += len(_arm) - 1
            pooled_std = math.sqrt(pooled_ss / pooled_dof) if pooled_dof > 0 else float("nan")
            # READINESS-6, 2026-09-10. `max(pooled_std, 1e-10)` MANUFACTURES
            # FINDINGS, and this is the third site of the identical defect
            # (see _statistics.sequential_fairness_test and
            # monitoring.drift.run_sprt). It reads as a divide-by-zero guard and
            # is one; what it guards is the crash, by turning an undefined
            # standardised difference into an enormous finite one that this
            # method converts straight into a decision.
            #
            # Measured before this change, two CONSTANT arms at 30 per arm:
            #
            #   gap 0.5     -> reject_null, stopped_early, llr = 1.5e10
            #   gap 0.01    -> reject_null, stopped_early, llr = 3.0e8
            #   gap 0.001   -> reject_null, stopped_early, llr = 3.0e7
            #   gap 0.0001  -> reject_null, stopped_early, llr = 3.0e6
            #
            # while the same gaps with ordinary noise correctly returned
            # `continue` for everything below 0.5. An experiment whose outcome
            # column is constant within each arm (a rate that has not moved, a
            # deterministic scorer) declared a treatment effect at a gap ten
            # thousand times smaller than the one it will call on real data.
            #
            # `np.ptp` rather than `var(...) == 0`: variance is accumulated and a
            # constant array of a non-representable value has a tiny non-zero
            # variance at some lengths and exactly zero at others.
            if float(np.ptp(ctrl)) == 0.0 and float(np.ptp(treat)) == 0.0:
                gap = float(treat.mean() - ctrl.mean())
                # Identical constants in both arms is an OBSERVED absence of
                # difference, and refusing there would discard a confirmation.
                measured_no_difference = gap == 0.0
                if not measured_no_difference:
                    warnings.warn(
                        f"sequential_test{ix}: both arms are constant, so they differ by "
                        f"exactly {gap:.6g} and there is no variation to standardise it "
                        f"against. A sequential test needs a pooled standard deviation "
                        f"and none exists, so no decision was taken. The DIFFERENCE IS "
                        f"REAL AND CERTAIN; only its size relative to noise is unknown, "
                        f"because there is no noise. Judge {gap:.6g} on this metric's own "
                        f"scale. Collecting more data will not change this.",
                        UserWarning,
                        stacklevel=2,
                    )
                results[ix] = SequentialTestResult(
                    intersection=ix,
                    decision=(
                        SPRTDecision.ACCEPT_NULL
                        if measured_no_difference
                        else SPRTDecision.COULD_NOT_CHECK
                    ),
                    log_likelihood_ratio=0.0 if measured_no_difference else float("nan"),
                    upper_boundary=upper_boundary,
                    lower_boundary=lower_boundary,
                    stopped_early=measured_no_difference,
                    n_observations=n_c + n_t,
                )
                continue

            if not math.isfinite(pooled_std) or pooled_std == 0.0:
                # Neither arm supplied a variance, so there is no unit to divide
                # the difference by. Placed BELOW the constant-arms branch on
                # purpose: two constant arms are an observed absence (or an
                # observed certain gap) and keep the disposition that branch
                # already gives them. Reaching here instead means the pooled
                # variance itself is undefined, and no decision follows from it.
                gap = float(treat.mean() - ctrl.mean())
                warnings.warn(
                    f"sequential_test{ix}: no pooled standard deviation exists for these "
                    f"arms ({len(ctrl)} control and {len(treat)} treatment observation(s)), "
                    f"so the difference of {gap:.6g} cannot be standardised and no decision "
                    f"was taken. Reporting decision=could_not_check and "
                    f"log_likelihood_ratio=nan, NOT a ratio computed against an assumed "
                    f"unit of 1.0.",
                    UserWarning,
                    stacklevel=2,
                )
                results[ix] = SequentialTestResult(
                    intersection=ix,
                    decision=SPRTDecision.COULD_NOT_CHECK,
                    log_likelihood_ratio=float("nan"),
                    upper_boundary=upper_boundary,
                    lower_boundary=lower_boundary,
                    stopped_early=False,
                    n_observations=n_c + n_t,
                )
                continue

            pooled_std = float(pooled_std)

            # Normalised observed difference
            obs_diff = (treat.mean() - ctrl.mean()) / pooled_std

            # Log-likelihood ratio for the two-sample Wald SPRT (Wald 1945):
            # the standardised difference x̄ has variance v = 1/n_c + 1/n_t,
            # so Λ = (d·x̄ − d²/2) / v, i.e. (n/2)·(d·x̄ − d²/2) for equal
            # arms, where d = effect_size. The previous n·(d·x̄ − d²/2)
            # DOUBLED the evidence, crossing the Wald boundaries with half
            # the required data (~2.5x alpha type-I under monitoring;
            # audit 2026-08-22). Arm sizes are the post-dropna outcome
            # counts; an empty arm degrades to a NaN LLR (decision CONTINUE),
            # matching the previous NaN-mean behaviour without dividing by 0.
            if len(ctrl) > 0 and len(treat) > 0:
                var_obs = 1.0 / len(ctrl) + 1.0 / len(treat)
                llr = (effect_size * obs_diff - effect_size**2 / 2) / var_obs
            else:
                llr = float("nan")

            if llr >= upper_boundary:
                decision = SPRTDecision.REJECT_NULL
                stopped_early = True
            elif llr <= lower_boundary:
                decision = SPRTDecision.ACCEPT_NULL
                stopped_early = True
            else:
                decision = SPRTDecision.CONTINUE
                stopped_early = False

            results[ix] = SequentialTestResult(
                intersection=ix,
                decision=decision,
                log_likelihood_ratio=round(float(llr), 4),
                upper_boundary=round(upper_boundary, 4),
                lower_boundary=round(lower_boundary, 4),
                stopped_early=stopped_early,
                n_observations=len(ctrl) + len(treat),
            )

        return results

    # Adaptive sampling plan

    def adaptive_sampling_plan(
        self,
        budget: int = 1000,
        effect_size: float = 0.2,
    ) -> SamplingPlan:
        """Generate an adaptive sampling plan that focuses on underpowered
        intersections.

        Allocates additional samples proportionally to the *gap* between
        the current sample size and the required sample size.  Intersections
        that are already adequately powered receive no allocation.  When the
        budget runs out before every underpowered intersection is reached, the
        unreached ones are listed in ``SamplingPlan.not_allocated`` with a
        rationale saying so, and a warning is raised; they are never reported
        as powered.

        Parameters
        ----------
        budget : int
            Total additional samples available.
        effect_size : float
            Effect size for power calculation.

        Returns
        -------
        SamplingPlan
        """
        required = self.required_sample_size(effect_size)
        current_power = self.power_for_sample_size(effect_size)
        sizes = self._intersection_sizes()

        # Determine the gap for each intersection
        gaps: Dict[Intersection, int] = {}
        # An intersection whose power was NOT COMPUTED (below the analyzer's
        # size floor, so power_for_sample_size reports NaN) AND which is not
        # short of the required n is neither powered nor underpowered: it used
        # to fall into the "already powered" fill loop below. Measured
        # 2026-09-10 at 28 per arm with min_group_size=30 and a requirement of
        # 25: rationale "already powered", n_underpowered=0, while the same
        # analyzer's power_for_sample_size said 0.0 and is_powered False.
        #
        # An intersection BELOW the requirement is unambiguously underpowered
        # whether or not a power number was computed for it (R-1's fixture is
        # exactly that: 9 per arm against a requirement of 393), so it keeps its
        # allocation and its place in n_underpowered.
        unmeasured = {ix for ix, p in current_power.items() if p is None or not np.isfinite(p)}
        row_counts = self._intersection_row_counts()
        not_assessed: List[Intersection] = []
        for ix, (n_c, n_t) in sizes.items():
            current_n = min(n_c, n_t)
            needed = required[ix]
            gap = max(0, needed - current_n)
            rows_c, rows_t = row_counts.get(ix, (0, 0))
            if min(n_c, n_t) == 0 and min(rows_c, rows_t) > 0:
                # BGL-S2: an arm with rows but ZERO observations is not
                # underpowered, it is unmeasured. The gap arithmetic reads it as
                # "short by the whole requirement" and prescribes collecting N
                # more samples, which is the wrong instruction: rows are already
                # there and carry no outcome value, so the fix is the data, not
                # the size. Checked ABOVE the gap branch, because both branches
                # share the precondition that `current_n` is evidence at all.
                #
                # BGL-S2b (2026-09-17): `min(rows_c, rows_t) > 0` is the
                # correction. The condition used to be `min(n_c, n_t) == 0`
                # alone, so an arm with NO ROWS AT ALL took this branch too and
                # was handed the rationale "rows are present ... collecting more
                # rows will not help until the outcome is recorded", which is
                # false about an empty arm and is the opposite of the right
                # instruction. Measured at the public entry on an experiment
                # whose treatment arm has no 'f' row at all: before the S2
                # change the plan allocated +23 samples and counted it in
                # n_underpowered; after it, allocations {} and
                # n_underpowered 0, i.e. the refusal DELETED a correct finding.
                # Zero rows is a MEASURED zero, not a could-not-check: we know
                # there are none, and collecting them is exactly the fix.
                not_assessed.append(ix)
            elif gap > 0:
                gaps[ix] = gap
            elif ix in unmeasured:
                not_assessed.append(ix)

        def _fill_rationale(rationale: Dict[Intersection, str]) -> Dict[Intersection, str]:
            for ix in not_assessed:
                n_c, n_t = sizes[ix]
                rows_c, rows_t = row_counts.get(ix, (0, 0))
                if min(n_c, n_t) == 0 and min(rows_c, rows_t) > 0:
                    rationale[ix] = (
                        f"NOT ASSESSED: {rows_c} control and {rows_t} treatment row(s) are "
                        f"present but only {n_c} and {n_t} carry an outcome value, so no "
                        f"power was computed. This is not a finding of adequate power, and "
                        f"it is not a sample-size shortfall either: collecting more rows "
                        f"will not help until the outcome is recorded."
                    )
                else:
                    rationale[ix] = (
                        f"NOT ASSESSED: {min(n_c, n_t)} per arm is below the "
                        f"min_group_size floor of {self.config.min_group_size}, so power "
                        f"was not computed. This is not a finding of adequate power."
                    )
            return rationale

        if not_assessed:
            no_observations = [
                ix
                for ix in not_assessed
                if min(sizes[ix]) == 0 and min(row_counts.get(ix, (0, 0))) > 0
            ]
            detail = (
                f" {len(no_observations)} of them hold no outcome observation at all: "
                f"{no_observations}."
                if no_observations
                else ""
            )
            warnings.warn(
                f"adaptive_sampling_plan: power was not computed for "
                f"{len(not_assessed)} intersection(s) below the min_group_size floor of "
                f"{self.config.min_group_size} or holding no observations: {not_assessed}."
                f"{detail} They are listed in not_assessed and are NOT counted as powered "
                f"or as underpowered.",
                UserWarning,
                stacklevel=2,
            )

        if not gaps:
            return SamplingPlan(
                allocations={},
                rationale=_fill_rationale(
                    {
                        ix: "already powered"
                        for ix in self.experiment.intersections
                        if ix not in not_assessed
                    }
                ),
                priority_order=[],
                total_budget=0,
                n_underpowered=0,
                not_assessed=not_assessed,
            )

        # Priority: largest gap first
        priority = sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)
        total_gap = sum(gaps.values())

        allocations: Dict[Intersection, int] = {}
        rationale: Dict[Intersection, str] = {}
        not_allocated: List[Intersection] = []
        remaining = budget

        for ix in priority:
            pwr = current_power.get(ix, 0)
            n_c, n_t = sizes[ix]
            if remaining <= 0:
                # R-1 (audit 6, 2026-09-09). This used to `break`; the fill loop
                # below then labelled every intersection without a rationale
                # "already powered", and to_dataframe read the missing allocation
                # as 0. Measured: four intersections at power 0.00 needing 393
                # per arm, budget=2; the plan reported two of them as
                # additional_samples=0, "already powered", while its own
                # n_underpowered said 4. An intersection the budget did not
                # reach is still underpowered and must say so.
                not_allocated.append(ix)
                rationale[ix] = (
                    f"Power={_power_label(pwr)}, need {required[ix]} per arm "
                    f"(have {min(n_c, n_t)}). NOT ALLOCATED: the budget of {budget} "
                    f"was exhausted before this intersection was reached."
                )
                continue
            # Proportional allocation
            proportion = gaps[ix] / total_gap
            alloc = min(int(math.ceil(proportion * budget)), remaining, gaps[ix])
            allocations[ix] = alloc
            remaining -= alloc

            rationale[ix] = (
                f"Power={_power_label(pwr)}, need {required[ix]} per arm "
                f"(have {min(n_c, n_t)}). +{alloc} samples."
            )

        if not_allocated:
            warnings.warn(
                f"adaptive_sampling_plan: the budget of {budget} was exhausted before "
                f"{len(not_allocated)} of {len(gaps)} underpowered intersection(s) received "
                f"an allocation: {not_allocated}. They remain underpowered; their "
                f"additional_samples is None, not 0.",
                UserWarning,
                stacklevel=2,
            )

        # Fill rationale for powered intersections. "already powered" means a
        # gap of zero and nothing else (R-1): an intersection that merely has no
        # allocation yet is not powered, and one whose power was never computed
        # is not powered either (readiness 6).
        for ix in self.experiment.intersections:
            if ix not in gaps and ix not in not_assessed:
                rationale[ix] = "already powered"
        _fill_rationale(rationale)

        return SamplingPlan(
            allocations=allocations,
            rationale=rationale,
            priority_order=priority,
            total_budget=sum(allocations.values()),
            n_underpowered=len(gaps),
            not_allocated=not_allocated,
            not_assessed=not_assessed,
        )

    # Minimum detectable effect

    def minimum_detectable_effect(self) -> Dict[Intersection, float]:
        """MDE per intersection at ``target_power``.

        Inverts the power formula to find the smallest Cohen's d detectable
        at the current sample size:
        ``d = (z_{α/2} + z_{β}) × √(2/n)``

        Returns
        -------
        dict[tuple, float]
            MDE (Cohen's d) per intersection.
        """
        z_alpha = _norm_ppf(1 - self.config.alpha / 2)
        z_beta = _norm_ppf(self.config.target_power)
        sizes = self._intersection_sizes()
        results: Dict[Intersection, float] = {}

        for ix, (n_c, n_t) in sizes.items():
            n_eff = min(n_c, n_t)
            # `n_eff == 0` first, and independently of the floor: with
            # min_group_size=0 this line used to reach `2 / n_eff` and die with
            # ZeroDivisionError, which also took get_power_summary down with it
            # (G-023, 2026-09-17).
            if n_eff == 0 or n_eff < self.config.min_group_size:
                # Same sentinel problem as power_for_sample_size above: inf is
                # the statement "no effect, however large, is detectable here",
                # which is a measurement. Below the floor nothing was computed.
                results[ix] = float("nan")
            else:
                mde = (z_alpha + z_beta) * np.sqrt(2 / n_eff)
                results[ix] = round(float(mde), 4)
        return results

    # Summary table

    def get_power_summary(
        self,
        effect_size: float = 0.2,
    ) -> pd.DataFrame:
        """Comprehensive power summary table for all intersections.

        Parameters
        ----------
        effect_size : float
            Effect size for power calculations.

        Returns
        -------
        pd.DataFrame
            Columns: intersection, n_control, n_treatment, n_rows_control,
            n_rows_treatment, power, required_n_per_arm, is_powered, mde,
            effect_size.

            ``n_control``/``n_treatment`` count OBSERVATIONS (rows carrying an
            outcome value); ``n_rows_control``/``n_rows_treatment`` count rows.
            The two columns sit side by side because "60 rows, 0 observations"
            names the problem and either number alone hides half of it
            (BGL-S2, 2026-09-16).
        """
        sizes = self._intersection_sizes()
        row_counts = self._intersection_row_counts()
        power = self.power_for_sample_size(effect_size)
        required = self.required_sample_size(effect_size)
        mde = self.minimum_detectable_effect()

        rows = []
        for ix in self.experiment.intersections:
            n_c, n_t = sizes[ix]
            rows_c, rows_t = row_counts.get(ix, (0, 0))
            pwr = power.get(ix)
            # `power.get(ix, 0)` read a MISSING power as a measured zero, and
            # `pwr >= target` graded a NaN (not computed) as False, i.e. as
            # "underpowered". Both are verdicts about a number nobody produced.
            measured = pwr is not None and np.isfinite(pwr)
            pwr_value = float(pwr) if measured and pwr is not None else float("nan")
            rows.append(
                {
                    "intersection": str(ix),
                    "n_control": n_c,
                    "n_treatment": n_t,
                    "n_rows_control": rows_c,
                    "n_rows_treatment": rows_t,
                    "power": round(pwr_value, 4) if measured else float("nan"),
                    "required_n_per_arm": required.get(ix),
                    "is_powered": (
                        bool(pwr_value >= self.config.target_power) if measured else None
                    ),
                    "mde": mde.get(ix, float("nan")),
                    "effect_size": effect_size,
                }
            )

        return pd.DataFrame(rows)

    def get_detailed_results(
        self,
        effect_size: float = 0.2,
    ) -> List[PowerResult]:
        """Return :class:`PowerResult` objects for every intersection.

        Parameters
        ----------
        effect_size : float
            Effect size for power calculations.

        Returns
        -------
        list[PowerResult]
        """
        sizes = self._intersection_sizes()
        row_counts = self._intersection_row_counts()
        power = self.power_for_sample_size(effect_size)
        required = self.required_sample_size(effect_size)

        results: List[PowerResult] = []
        for ix in self.experiment.intersections:
            n_c, n_t = sizes[ix]
            rows_c, rows_t = row_counts.get(ix, (0, 0))
            pwr = power.get(ix)
            measured = pwr is not None and np.isfinite(pwr)
            pwr_value = float(pwr) if measured and pwr is not None else float("nan")
            results.append(
                PowerResult(
                    intersection=ix,
                    power=pwr_value,
                    required_n=required.get(ix, 0),
                    is_powered=(bool(pwr_value >= self.config.target_power) if measured else None),
                    n_control=n_c,
                    n_treatment=n_t,
                    effect_size=effect_size,
                    n_rows_control=rows_c,
                    n_rows_treatment=rows_t,
                )
            )
        return results
