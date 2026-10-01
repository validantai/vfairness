"""
Unit 4: A/B Testing for Fairness: FairnessExperiment
=====================================================

Core A/B testing framework for fairness interventions with multi-objective
optimisation and intersectional analysis capabilities.

The ``FairnessExperiment`` class implements the curriculum specification from
Part 4, Unit 4 of the Fairness Pipeline Development Toolkit.  It answers the
question: *"Does the treatment affect different demographic intersections
differently?"*

Key features:

- **Intersectional analysis**: Automatically enumerates all observed
  demographic intersections and computes per-intersection treatment effects.
- **Bootstrap confidence intervals**: Non-parametric CIs for every effect.
- **Heterogeneity testing**: Detects whether treatment effects vary
  significantly across intersections (ANOVA F-test / Kruskal–Wallis).
- **Multiple comparison correction**: Bonferroni or FDR (Benjamini–Hochberg)
  adjustment for per-intersection p-values.
- **Design utilities**: Cluster randomisation and factorial design helpers.

References
----------
Athey, S. & Imbens, G. W. (2017). The econometrics of randomized experiments.
Deng, A. et al. (2023). Multi-objective optimization in online controlled
  experiments with multiple metrics. KDD 2023.
Module 4, Part 4, Unit 4: A/B Testing for Fairness.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd

from ...evaluation.vfairness_metrics._statistics import (
    _norm_cdf,
    _norm_ppf,
    apply_multiple_testing_correction,
    cohens_d,
)

# Reuse existing vfairness statistical utilities
from ...exceptions import ConfigurationError

# The effect-size refusal is shared with the sibling power module rather than
# restated here, so the two public entries that take a Cohen's d cannot drift
# apart again (G04, 2026-09-30). `power` imports nothing from this module, so
# this is acyclic.
from .power import _validate_effect_size

# scipy for t-tests and ANOVA (available via robustness.py pattern)
try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# Enums & Config


#: Correction methods ``apply_multiple_testing_correction`` accepts. Kept here
#: so ``ExperimentConfig`` can refuse an unknown one at construction instead of
#: raising from inside the callee after every bootstrap has already run.
_CORRECTION_METHODS = frozenset({"bonferroni", "fdr", "benjamini_hochberg", "none"})


class DesignType(Enum):
    """Experimental design type."""

    SIMPLE_AB = "simple_ab"
    STRATIFIED = "stratified"
    CLUSTER = "cluster"
    FACTORIAL = "factorial"


@dataclass
class ExperimentConfig:
    """Configuration for :class:`FairnessExperiment`.

    Parameters
    ----------
    alpha : float
        Significance level for hypothesis tests (default 0.05).
    correction_method : str
        Multiple comparison correction: ``'fdr'``, ``'bonferroni'``, or
        ``'none'``.
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.
    min_group_size : int
        Minimum intersection size in *each* arm: groups below this are
        excluded from per-intersection analysis.
    design_type : DesignType
        Experimental design type.
    cluster_column : str, optional
        Column name for cluster IDs (required for CLUSTER design).
    random_state : int, optional
        Random seed for reproducibility.
    """

    alpha: float = 0.05
    correction_method: str = "fdr"
    n_bootstrap: int = 2000
    min_group_size: int = 30
    design_type: DesignType = DesignType.SIMPLE_AB
    cluster_column: Optional[str] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        # G04 (2026-09-30). This dataclass validated NOTHING, while its sibling
        # PowerConfig refuses an impossible alpha at construction. Measured
        # before this change, every one of these was accepted in silence and
        # three of them reached a published number:
        #
        #   min_group_size=0  -> an intersection with 5 control rows and ZERO
        #       treatment rows was counted as ANALYSED:
        #       get_summary() read n_intersections_analysed=3,
        #       n_intersections_excluded=0, and the third row of the CATE frame
        #       was NaN in every column. The coverage figure a reader uses to
        #       judge how much of the experiment was measured named one more
        #       intersection than was ever compared.
        #   design_type='nope' -> AttributeError('str' has no attribute
        #       'value') from ExperimentResult.to_dict() and again from
        #       ExperimentAnalysis.to_report_sections(), i.e. after the whole
        #       analysis had run.
        #   alpha=5.0 / alpha=0.0 / n_bootstrap=0 -> accepted here and refused
        #       later by detect_heterogeneous_effects, so the object could sit
        #       in a caller's hands looking valid.
        #   correction_method='banana' -> accepted here, raised from inside
        #       apply_multiple_testing_correction after every bootstrap had run.
        #
        # min_group_size=1 stays legal: one observation per arm is a terrible
        # design, not an impossible one, and this library reports it as such
        # (p=nan, significant=None) rather than refusing it. Only a floor below
        # one is refused, because it admits an arm with nothing in it.
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise ConfigurationError(f"alpha must be a number, got {self.alpha!r}.")
        if not np.isfinite(float(self.alpha)) or not 0.0 < float(self.alpha) < 1.0:
            raise ConfigurationError(
                f"alpha must be strictly between 0 and 1, got {self.alpha!r}. "
                f"alpha=0 asks for a test that can never reject and alpha=1 for one "
                f"that always does."
            )
        if not isinstance(self.n_bootstrap, int) or isinstance(self.n_bootstrap, bool):
            raise ConfigurationError(f"n_bootstrap must be an int, got {self.n_bootstrap!r}.")
        if self.n_bootstrap < 1:
            raise ConfigurationError(
                f"n_bootstrap must be at least 1, got {self.n_bootstrap!r}; there is no "
                f"bootstrap interval to build from zero resamples."
            )
        if not isinstance(self.min_group_size, int) or isinstance(self.min_group_size, bool):
            raise ConfigurationError(f"min_group_size must be an int, got {self.min_group_size!r}.")
        if self.min_group_size < 1:
            raise ConfigurationError(
                f"min_group_size must be at least 1, got {self.min_group_size!r}. A floor "
                f"below one admits an arm holding no observation at all, and an "
                f"intersection with an empty arm was then counted as ANALYSED while "
                f"every number computed for it was NaN."
            )
        if not isinstance(self.design_type, DesignType):
            raise ConfigurationError(
                f"design_type must be a DesignType, got {self.design_type!r}. A bare "
                f"string reaches ExperimentResult.to_dict() and raises there, after the "
                f"analysis has already run."
            )
        if self.correction_method not in _CORRECTION_METHODS:
            raise ConfigurationError(
                f"Unknown correction method: {self.correction_method!r}. Expected one of "
                f"{sorted(_CORRECTION_METHODS)}."
            )


# Result Dataclasses


def _yes_no_not_assessed(verdict: Optional[bool]) -> str:
    """Three states in a repr: None is 'not assessed', never 'no'."""
    if verdict is None:
        return "not assessed"
    return "YES" if verdict else "no"


@dataclass
class IntersectionEffect:
    """Treatment effect result for a single demographic intersection.

    Attributes
    ----------
    intersection : tuple
        Demographic intersection values (e.g. ``('Female', 'Black')``).
    control_mean : float
        Mean outcome in control for this intersection.
    treatment_mean : float
        Mean outcome in treatment for this intersection.
    effect : float
        Treatment effect (treatment_mean − control_mean).
    ci_lower : float
        Lower bound of the confidence interval for the effect.
    ci_upper : float
        Upper bound of the confidence interval for the effect.
    p_value : float
        Two-sample t-test p-value.
    effect_size_d : float
        Cohen's d effect size.
    n_control : int
        Control sample size for this intersection.
    n_treatment : int
        Treatment sample size for this intersection.
    significant : bool or None
        Whether p-value < alpha (after correction); ``None`` when the p-value is
        not finite, so no test could run. This is three-state under EVERY
        correction method, not only ``correction_method='none'``.
    powered : bool or None
        Whether the intersection has >= 0.8 statistical power. ``None`` means
        power was not computed for it (below the size floor, or absent from the
        power results), which is not the same as underpowered. The default is
        None, not True: an effect object built without a power calculation
        asserted adequate power for every consumer that read the field.
    """

    intersection: Tuple
    control_mean: float
    treatment_mean: float
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size_d: float
    n_control: int
    n_treatment: int
    significant: Optional[bool] = False
    powered: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intersection": list(self.intersection),
            "control_mean": round(self.control_mean, 6),
            "treatment_mean": round(self.treatment_mean, 6),
            "effect": round(self.effect, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "p_value": round(self.p_value, 6),
            "effect_size_d": round(self.effect_size_d, 4),
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "significant": self.significant,
            "powered": self.powered,
        }


#: Column order of the tidy per-intersection frame, kept beside
#: ``IntersectionEffect.to_dict`` so an EMPTY frame can carry the same schema
#: instead of the (0, 0) frame with no columns at all that
#: ``to_dataframe`` used to return. Pinned against ``to_dict`` by
#: tests/test_bgl5_operations_3.py so the two cannot drift apart.
INTERSECTION_COLUMNS: Tuple[str, ...] = (
    "intersection",
    "control_mean",
    "treatment_mean",
    "effect",
    "ci_lower",
    "ci_upper",
    "p_value",
    "effect_size_d",
    "n_control",
    "n_treatment",
    "significant",
    "powered",
)


@dataclass
class ExperimentResult:
    """Complete result from a :class:`FairnessExperiment` analysis.

    Attributes
    ----------
    overall_effect : float
        Average treatment effect across all data.
    overall_ci : tuple[float, float]
        95 % CI for the overall effect.
    overall_p_value : float
        P-value for the overall treatment effect.
    intersection_effects : list[IntersectionEffect]
        Per-intersection treatment effects.
    heterogeneity_detected : bool or None
        ``True`` if treatment effects differ significantly across groups; ``None``
        when the heterogeneity test could not run.
    heterogeneity_p_value : float
        P-value for the heterogeneity test (F-test or Kruskal–Wallis).
    power_results : dict
        Statistical power per intersection.
    n_intersections : int
        Number of intersections analysed.
    n_excluded : int
        Number of intersections excluded for small sample.
    design_type : DesignType
        Experimental design used.
    metadata : dict
        Additional metadata.
    """

    overall_effect: float
    overall_ci: Tuple[float, float]
    overall_p_value: float
    intersection_effects: List[IntersectionEffect]
    heterogeneity_detected: Optional[bool]
    heterogeneity_p_value: float
    power_results: Dict[str, float] = field(default_factory=dict)
    n_intersections: int = 0
    n_excluded: int = 0
    design_type: DesignType = DesignType.SIMPLE_AB
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_effect": round(self.overall_effect, 6),
            "overall_ci": [round(x, 6) for x in self.overall_ci],
            "overall_p_value": round(self.overall_p_value, 6),
            "intersection_effects": [e.to_dict() for e in self.intersection_effects],
            "heterogeneity_detected": self.heterogeneity_detected,
            "heterogeneity_p_value": round(self.heterogeneity_p_value, 6),
            "power_results": {str(k): round(v, 4) for k, v in self.power_results.items()},
            "n_intersections": self.n_intersections,
            "n_excluded": self.n_excluded,
            "design_type": self.design_type.value,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Write JSON representation to a file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-intersection effects to a tidy DataFrame.

        An empty result is DISCLOSED rather than returned in silence: the frame
        carries the column schema, a warning says why it is empty, and
        ``frame.attrs['coverage']`` records the analysed and excluded counts.
        """
        if not self.intersection_effects:
            # BGL5 A-operations-3, 2026-09-27. `return pd.DataFrame()` made "no
            # intersection showed an effect" and "every intersection was excluded
            # before anything was compared" the same output, at the tidy surface a
            # caller reads, while this very object knew n_excluded. Measured on an
            # experiment of two 10-row arms with min_group_size=30, so both
            # intersections were excluded:
            #   before  empty=True shape=(0, 0) columns=[] warnings=[] attrs={},
            #           identical to a run that analysed intersections and found
            #           nothing, although get_summary() reported
            #           n_intersections_excluded=2.
            #   after   shape=(0, 12) with the documented columns, one warning
            #           naming '2 intersection(s) were EXCLUDED', and
            #           attrs['coverage'] = {'n_intersections': 0, 'n_excluded': 2,
            #           'measured': False}.
            # FairnessExperiment.to_dataframe delegates here, so both surfaces the
            # audit overturned are covered by this one guard. A run WITH effects is
            # untouched: 2 intersections still give shape (2, 12) and no warning.
            if self.n_excluded:
                why = (
                    f"{self.n_excluded} intersection(s) were EXCLUDED before any "
                    "comparison (below the configured min_group_size), so nothing "
                    "was compared for them"
                )
            elif self.n_intersections == 0:
                why = "no intersection was analysed at all, so nothing was compared anywhere"
            else:
                why = (
                    f"{self.n_intersections} intersection(s) are recorded as "
                    "analysed but this result carries no per-intersection effect"
                )
            warnings.warn(
                f"to_dataframe returned an EMPTY frame because {why}. This is a "
                "could-not-check, not a finding that the treatment had no effect on "
                "any intersection: an absence of rows here is an absence of "
                "evidence. See get_summary()['n_intersections_excluded'] and "
                "frame.attrs['coverage'].",
                UserWarning,
                stacklevel=2,
            )
            empty = pd.DataFrame(columns=list(INTERSECTION_COLUMNS))
            empty.attrs["coverage"] = {
                "n_intersections": int(self.n_intersections),
                "n_excluded": int(self.n_excluded),
                "measured": False,
                "notes": why,
            }
            return empty
        frame = pd.DataFrame([e.to_dict() for e in self.intersection_effects])
        frame.attrs["coverage"] = {
            "n_intersections": int(self.n_intersections),
            "n_excluded": int(self.n_excluded),
            "measured": True,
            "notes": (
                f"{len(self.intersection_effects)} intersection(s) compared; "
                f"{self.n_excluded} excluded before comparison"
            ),
        }
        return frame

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(overall_effect={self.overall_effect:.4f}, "
            f"heterogeneity={_yes_no_not_assessed(self.heterogeneity_detected)}, "
            f"intersections={self.n_intersections})"
        )


class FairnessExperiment:
    """A/B testing framework with intersectional fairness analysis.

    Implements the core A/B testing workflow from Part 4, Unit 4 of the
    Fairness Pipeline Development Toolkit.

    Parameters
    ----------
    control_data : pd.DataFrame
        Data from the control arm.
    treatment_data : pd.DataFrame
        Data from the treatment arm.
    protected_attributes : list[str]
        Column names of protected attributes (e.g. ``['gender', 'race']``).
    outcome_column : str
        Column name for the primary outcome variable.
    business_metrics : list[str], optional
        Column names for business KPIs (e.g. ``['revenue', 'conversion']``).
    config : ExperimentConfig, optional
        Experiment configuration.

    Examples
    --------
    >>> exp = FairnessExperiment(
    ...     control_data=df_control,
    ...     treatment_data=df_treatment,
    ...     protected_attributes=['gender', 'race'],
    ...     outcome_column='approved',
    ...     business_metrics=['revenue'],
    ... )
    >>> result = exp.run_full_analysis()
    >>> print(result.heterogeneity_detected)

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

    Ledger row: fairness_experiment. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        protected_attributes: List[str],
        outcome_column: str,
        business_metrics: Optional[List[str]] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.control = control_data.copy()
        self.treatment = treatment_data.copy()
        self.protected_attrs = list(protected_attributes)
        self.outcome = outcome_column
        self.business_metrics = list(business_metrics or [])
        self.config = config or ExperimentConfig()
        self._rng = np.random.default_rng(self.config.random_state)

        # Pre-compute intersections
        self.intersections = self._generate_intersections()
        self._result: Optional[ExperimentResult] = None

    # Intersection machinery (curriculum-specified)

    def _generate_intersections(self) -> List[Tuple]:
        """Generate all unique demographic intersection tuples.

        Combines control + treatment data, extracts every observed
        combination of protected attribute values.

        Returns
        -------
        list[tuple]
            Each tuple is one intersection, e.g. ``('Female', 'Black')``. A
            combination in which any protected attribute value is ABSENT is not
            an intersection and is left out; the rows carrying it are counted
            and named in a warning, and in ``self.rows_without_intersection``.
        """
        self.rows_without_intersection = 0
        if not self.protected_attrs:
            return [("overall",)]

        combined = pd.concat(
            [self.control[self.protected_attrs], self.treatment[self.protected_attrs]],
            ignore_index=True,
        ).drop_duplicates()

        intersections = [tuple(row) for _, row in combined.iterrows()]

        # AN ABSENT DEMOGRAPHIC VALUE IS NOT A DEMOGRAPHIC GROUP (G04,
        # 2026-09-30). Every combination observed in the frames became an
        # intersection, absence sentinels included, and the doors behaved
        # differently from each other, which is what made this hard to see.
        # Measured that day on 20 rows per arm whose 'g' value was absent,
        # beside 40 'f' and 40 'm' rows:
        #
        #   np.nan -> intersection (nan,) generated, then `df[attr] == nan` is
        #       False for every row, so it matched NOTHING and was reported as
        #       EXCLUDED for "insufficient sample size". The 20 rows were in no
        #       intersection and nothing said so.
        #   pd.NA  -> TypeError('boolean value of NA is ambiguous') out of
        #       `intersection == ("overall",)` in _get_intersection_data, which
        #       killed run_full_analysis outright. pd.NA is the missing value of
        #       every pandas nullable dtype, and pandas is a hard dependency.
        #   ''     -> a group named '' with a published effect of +0.448 over
        #       20 rows per arm, counted among "3 analysed", and the
        #       recommendation said "with no harmed groups".
        #   'None' -> a group named 'None', effect +0.108, same treatment.
        #
        # The last two are the incident recorded in multi_agent.harness, where
        # pd.NA minted a group '<NA>' carrying a maximal finding of routing
        # discrimination. Its predicate is reused rather than restated so the
        # six doors stay closed in one place; the import is local because
        # `multi_agent.harness` pulls in the llm package and this module has no
        # other reason to.
        from ...multi_agent.harness import _label_not_recorded

        # Partitioned in ONE pass by position. `ix not in named` would compare
        # tuples with `==`, and that is the pd.NA door again: the membership test
        # itself raised "boolean value of NA is ambiguous". No value in these
        # tuples is ever compared here, only classified.
        named: List[Tuple] = []
        dropped: List[Tuple] = []
        for ix in intersections:
            (dropped if any(_label_not_recorded(v) for v in ix) else named).append(ix)
        if dropped:
            n_rows = 0
            for frame in (self.control, self.treatment):
                absent = pd.Series(False, index=frame.index)
                for attr in self.protected_attrs:
                    absent |= frame[attr].map(_label_not_recorded).astype(bool)
                n_rows += int(absent.sum())
            self.rows_without_intersection = n_rows
            warnings.warn(
                f"{len(dropped)} observed combination(s) of {self.protected_attrs} carry an "
                f"ABSENT value and are NOT demographic intersections: "
                f"{[str(ix) for ix in dropped[:5]]}. The {n_rows} row(s) behind them "
                f"(of {len(self.control) + len(self.treatment)}) take part in the OVERALL "
                f"effect and in no per-intersection comparison, so no intersectional "
                f"finding covers them. A blank label and the literal 'None' were "
                f"previously reported as groups of their own, with a treatment effect "
                f"attributed to them. Record the attribute for those rows, or drop them "
                f"deliberately, rather than reading the intersectional result as covering "
                f"the whole experiment.",
                UserWarning,
                stacklevel=3,
            )
        return sorted(named, key=str)

    def _get_intersection_data(
        self,
        df: pd.DataFrame,
        intersection: Tuple,
    ) -> pd.DataFrame:
        """Filter a DataFrame to rows matching a demographic intersection.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        intersection : tuple
            One value per protected attribute.

        Returns
        -------
        pd.DataFrame
        """
        # `intersection == ("overall",)` compares a tuple of CALLER DATA against
        # a string, element by element. With pd.NA in the tuple that comparison
        # answers pd.NA and `bool()` of it raises "boolean value of NA is
        # ambiguous", from a line that has nothing to do with the caller's
        # problem (G04, 2026-09-30). The ("overall",) sentinel is generated by
        # _generate_intersections in exactly one case, and that case is tested
        # directly here instead.
        if not self.protected_attrs:
            return df

        mask = pd.Series(True, index=df.index)
        for attr, val in zip(self.protected_attrs, intersection):
            mask &= df[attr] == val
        return df[mask]

    # Power analysis (curriculum-specified)

    def calculate_intersectional_power(
        self,
        effect_size: float = 0.2,
        alpha: float = 0.05,
        min_group_size: int = 30,
    ) -> Dict[Tuple, float]:
        """Calculate statistical power for each demographic intersection.

        Uses the two-sample z-test power formula:
        ``power = Φ(effect × √(n/2) − z_{α/2})``.

        Parameters
        ----------
        effect_size : float
            Expected standardised effect size (Cohen's d).
        alpha : float
            Significance level.
        min_group_size : int
            Minimum per-arm count of OBSERVATIONS (rows carrying an outcome
            value), not of rows. See the note below.

        Returns
        -------
        dict[tuple, float]
            Power (0 to 1) per intersection, or NaN where power was not
            computed. NaN is a could-not-check and is not a power of zero.

        Notes
        -----
        BGL-final d02 (2026-09-17). This counted ROWS: ``len(slice)``, never
        touching the outcome column, while the sibling
        :class:`~vfairness.operations.experimentation.power.FairnessPowerAnalyzer`
        was moved to observations. A power analysis is a statement about
        observations, so an intersection with 60 rows and no outcome value in
        any of them was graded exactly like a fully observed one. Measured at
        this public entry before the change, on three frames identical in shape
        (60 rows per intersection per arm) and differing ONLY in how many
        outcome values existed, 240 real, 5 real, and 0 real::

            calculate_intersectional_power(effect_size=0.5)
                240 real -> {('f',): 0.7818, ('m',): 0.7818}   no warning
                  5 real -> {('f',): 0.7818, ('m',): 0.7818}   no warning
                  0 real -> {('f',): 0.7818, ('m',): 0.7818}   no warning

        and ``run_full_analysis()`` wrote that number into
        ``ExperimentResult.power_results``. The answer was invariant to whether
        any observation existed at all. It now counts observations, so an arm
        with rows but nothing recorded falls below the floor and reports NaN,
        with a warning naming BOTH numbers, because "60 rows, 0 observations"
        is the sentence that identifies the problem and either half alone hides
        it. A fully observed intersection is unchanged.
        """
        # G04 (2026-09-30). THE INPUTS WERE NEVER CHECKED, and this method
        # answered every impossible design with a number on the power scale.
        # Measured that day on two intersections of ~100 observations per arm:
        #
        #   effect_size=0.0   -> {('f',): 0.025, ('m',): 0.025}   no warning
        #   effect_size=inf   -> {('f',): 1.0,   ('m',): 1.0}     no warning
        #   effect_size=nan   -> {('f',): nan,   ('m',): nan}     no warning
        #   effect_size=-0.5  -> {('f',): 0.0,   ('m',): 0.0}     no warning
        #   alpha=0.0         -> {('f',): 0.0,   ('m',): 0.0}     no warning
        #   alpha=1.0         -> {('f',): 0.9998,('m',): 0.9998}  no warning
        #   alpha=-0.2        -> {('f',): 0.0,   ('m',): 0.0}     no warning
        #
        # 0.025 is alpha/2, the false-positive rate, printed in the column a
        # reader reads as achieved power. alpha=1.0 is the dangerous direction:
        # a test that always rejects was reported as 99.98 percent POWERED, and
        # `powered = pwr >= 0.8` turns that into "adequately powered" in
        # ExperimentResult. The sibling FairnessPowerAnalyzer has refused
        # exactly these effect sizes since G-023, through
        # _validate_effect_size, which is reused here rather than restated so
        # the two entries cannot drift; alpha gets the same refusal the alpha
        # of detect_heterogeneous_effects already gets.
        effect_size = _validate_effect_size(effect_size, "calculate_intersectional_power")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ConfigurationError(
                f"calculate_intersectional_power: alpha must be a number, got {alpha!r}."
            )
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ConfigurationError(
                f"calculate_intersectional_power: alpha must be strictly between 0 and 1, "
                f"got {alpha!r}. alpha=0 asks for a test that can never reject (reported "
                f"as a power of 0.0) and alpha=1 for one that always does (reported as a "
                f"power of ~1.0); neither figure is a property of these data."
            )
        alpha = float(alpha)

        results: Dict[Tuple, float] = {}
        z_alpha = _norm_ppf(1 - alpha / 2)
        unobserved: List[Tuple[Tuple, int, int, int, int]] = []
        # A FLOOR BELOW ONE IS NOT A FLOOR. `n_c < min_group_size` is False for
        # n_c = 0 whenever min_group_size <= 0, so the arithmetic below ran with
        # n_eff = 0 and published `_norm_cdf(-z_alpha)`, which is alpha/2.
        # Measured 2026-09-30 with min_group_size=0 on an intersection whose
        # every outcome value was missing: {('f',): 0.025, ('m',): 0.3522} while
        # the warning at the bottom of this method stated, of that same
        # intersection, "They report NaN (could not check), NOT a power figure
        # derived from the row count". The guard and the arithmetic disagreed,
        # and the guard was the one telling the truth. One observation per arm
        # remains computable: it is a terrible design, not an absent one.
        floor = min_group_size if min_group_size >= 1 else 1

        for ix in self.intersections:
            ctrl_slice = self._get_intersection_data(self.control, ix)
            treat_slice = self._get_intersection_data(self.treatment, ix)
            rows_c, rows_t = len(ctrl_slice), len(treat_slice)
            n_c = self._count_outcome_observations(ctrl_slice)
            n_t = self._count_outcome_observations(treat_slice)

            if min(n_c, n_t) == 0 and min(rows_c, rows_t) > 0:
                # Rows are present and carry no outcome value. Recorded here
                # and warned about once below, rather than per intersection.
                unobserved.append((ix, rows_c, rows_t, n_c, n_t))

            if n_c < floor or n_t < floor:
                # NOT a power of zero: below the size floor this method does not
                # compute power at all. 0.0 is a MEASUREMENT on the power scale
                # and was read as one (`power >= 0.8` -> "underpowered", and the
                # sampling plan's "already powered" in the sibling module), so
                # the could-not-compute state gets the value this library uses
                # for it everywhere else.
                results[ix] = float("nan")
                continue

            n_eff = min(n_c, n_t)
            power = _norm_cdf(effect_size * np.sqrt(n_eff / 2) - z_alpha)
            results[ix] = round(float(power), 4)

        if unobserved:
            detail = ", ".join(
                f"{ix}: {rc}/{rt} rows carry {nc}/{nt} outcome values"
                for ix, rc, rt, nc, nt in unobserved[:5]
            )
            warnings.warn(
                f"calculate_intersectional_power: {len(unobserved)} intersection(s) have "
                f"an arm with rows but NO outcome observation, so no power was computed "
                f"for them ({detail}). They report NaN (could not check), NOT a power "
                f"figure derived from the row count. Power is a statement about "
                f"observations; collecting more rows will not help until "
                f"{self.outcome!r} is recorded.",
                UserWarning,
                stacklevel=2,
            )

        return results

    def _count_outcome_observations(self, slice_: pd.DataFrame) -> int:
        """Rows in one arm of one intersection that carry an outcome value.

        No outcome column at all is a could-not-check, not a full arm, so it
        counts zero rather than ``len(slice_)``.
        """
        if self.outcome not in slice_.columns:
            return 0
        return int(slice_[self.outcome].notna().sum())

    # Heterogeneous effects (curriculum-specified)

    def detect_heterogeneous_effects(
        self,
        n_bootstrap: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> ExperimentResult:
        """Detect whether treatment effects vary across intersections.

        For each intersection with sufficient samples:

        1. Compute the mean difference (treatment − control).
        2. Bootstrap a CI for that difference.
        3. Compute Cohen's d.
        4. Compute a two-sample t-test p-value.

        Then test for heterogeneity using a one-way ANOVA F-test on the
        per-intersection effects.  Applies the configured multiple
        comparison correction.

        Parameters
        ----------
        n_bootstrap : int, optional
            Override ``config.n_bootstrap``.
        alpha : float, optional
            Override ``config.alpha``.

        Returns
        -------
        ExperimentResult
        """
        # `x or default` treats a legal ZERO as "not supplied". Measured
        # 2026-09-10: detect_heterogeneous_effects(alpha=0.0) ran the whole
        # analysis at 0.05 AND recorded metadata['alpha'] = 0.05, so the
        # provenance record agreed with the substitution instead of with the
        # request; n_bootstrap=0 silently ran 2000 resamples. `is None` is the
        # only test that distinguishes "not supplied" from "supplied as zero",
        # and a zero that IS supplied is refused rather than quietly repaired:
        # alpha=0 is a test that can never reject, which is not a configuration
        # this class can honour honestly.
        n_boot = self.config.n_bootstrap if n_bootstrap is None else int(n_bootstrap)
        sig_level = self.config.alpha if alpha is None else float(alpha)
        if not 0.0 < sig_level < 1.0 or not np.isfinite(sig_level):
            raise ConfigurationError(
                f"alpha must be strictly between 0 and 1, got {sig_level!r}. "
                f"alpha=0 asks for a test that can never reject and alpha=1 for one "
                f"that always does; neither is substituted with the default."
            )
        if n_boot < 1:
            raise ConfigurationError(
                f"n_bootstrap must be at least 1, got {n_boot!r}; there is no "
                f"bootstrap interval to build from zero resamples."
            )
        min_n = self.config.min_group_size
        # The resample count a percentile interval needs for BOTH bounds to be
        # order statistics of the draws: `ceil(2/alpha) - 1`, which is 39 at
        # alpha=0.05. Taken from ece_confidence_intervals, which derives it and
        # is pinned at the 38/39 boundary, rather than restated as a round
        # number here.
        min_resamples = max(2, int(math.ceil(2.0 / sig_level)) - 1)
        thin_ci: List[Tuple] = []

        effects: List[IntersectionEffect] = []
        excluded = 0
        raw_p_values: List[float] = []
        group_effect_lists: List[np.ndarray] = []  # for heterogeneity test

        for ix in self.intersections:
            ctrl = self._get_intersection_data(self.control, ix)[self.outcome].dropna()
            treat = self._get_intersection_data(self.treatment, ix)[self.outcome].dropna()

            if len(ctrl) < min_n or len(treat) < min_n:
                excluded += 1
                continue

            ctrl_arr = ctrl.values.astype(float)
            treat_arr = treat.values.astype(float)

            effect = float(treat_arr.mean() - ctrl_arr.mean())

            # Bootstrap CI
            boot_effects_list = []
            for _ in range(n_boot):
                c_samp = self._rng.choice(ctrl_arr, size=len(ctrl_arr), replace=True)
                t_samp = self._rng.choice(treat_arr, size=len(treat_arr), replace=True)
                boot_effects_list.append(t_samp.mean() - c_samp.mean())
            boot_effects = np.array(boot_effects_list)
            if n_boot < min_resamples:
                # THE BOUNDS OF AN INTERVAL NOBODY ESTIMATED (G04, 2026-09-30).
                # Below `ceil(2/alpha) - 1` draws, `np.percentile` cannot land
                # on an order statistic of the resamples and interpolates into
                # the tails, so the published bound is an artifact of the
                # interpolation rather than a quantile of the bootstrap
                # distribution. Measured that day at n_bootstrap=1 on two
                # constant 40-row arms: ci_lower == ci_upper == 0.0, a
                # ZERO-WIDTH 95 percent interval, reported as one.
                # The floor and its derivation are the repo's existing rule
                # (post_processing.calibration.metrics.ece_confidence_intervals,
                # pinned at the 38/39 boundary), reused rather than reinvented.
                # Only the BOUNDS become NaN: the effect, the p-value and
                # Cohen's d are measured from the data and are untouched.
                ci_low = float("nan")
                ci_high = float("nan")
                thin_ci.append(ix)
            else:
                ci_low = float(np.percentile(boot_effects, 100 * sig_level / 2))
                ci_high = float(np.percentile(boot_effects, 100 * (1 - sig_level / 2)))

            # Cohen's d
            d = float(cohens_d(treat_arr, ctrl_arr))

            # Two-sample t-test
            if _HAS_SCIPY:
                _, p_val = scipy_stats.ttest_ind(treat_arr, ctrl_arr, equal_var=False)
                p_val = float(p_val)
            else:
                # Welch's t-test manual fallback
                n1, n2 = len(treat_arr), len(ctrl_arr)
                s1, s2 = treat_arr.std(ddof=1), ctrl_arr.std(ddof=1)
                se = np.sqrt(s1**2 / n1 + s2**2 / n2)
                t_stat = effect / se if se > 0 else 0.0
                p_val = float(2 * _norm_cdf(-abs(t_stat)))

            raw_p_values.append(p_val)
            group_effect_lists.append(boot_effects)

            effects.append(
                IntersectionEffect(
                    intersection=ix,
                    control_mean=float(ctrl_arr.mean()),
                    treatment_mean=float(treat_arr.mean()),
                    effect=effect,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    p_value=p_val,
                    effect_size_d=d,
                    n_control=len(ctrl_arr),
                    n_treatment=len(treat_arr),
                )
            )

        # Multiple comparison correction
        if raw_p_values and self.config.correction_method != "none":
            corrected = apply_multiple_testing_correction(
                np.asarray(raw_p_values),
                # config.correction_method is a free-form str on the public
                # config; the callee validates it at runtime (raising on
                # unknown values), so bridge to its Literal parameter type.
                method=cast(
                    Literal["bonferroni", "fdr", "benjamini_hochberg", "none"],
                    self.config.correction_method,
                ),
                alpha=sig_level,
            )
            # THREE STATES, and this branch had two. `rejection_mask` is a bool
            # array: False there means "not rejected", which for a p-value that
            # was never finite means the test DID NOT RUN. `tested_mask` and
            # `n_not_tested` were published on the correction result for exactly
            # this, with a warning, and were dropped on the floor here, while
            # the `correction_method == "none"` branch below did it correctly and
            # is not the default. Measured 2026-09-10 on identical data, one
            # intersection with a single observation per arm (Welch t = nan):
            #   fdr  -> ("X",) effect=-5.0000 p=nan significant=False, and the
            #           recommendation read "Overall positive effect (0.5695,
            #           p=0.0000) with no harmed groups."
            #   none -> ("X",) effect=-5.0000 p=nan significant=None, and the
            #           recommendation named it: "1 intersection(s) with a
            #           negative effect carry no significance verdict".
            # A demographic the treatment moved by -5.0 and nobody tested was
            # being counted as evidence of safety.
            tested_mask = corrected.tested_mask
            for i, eff in enumerate(effects):
                eff.p_value = float(corrected.adjusted_p_values[i])
                was_tested = True if tested_mask is None else bool(tested_mask[i])
                eff.significant = bool(corrected.rejection_mask[i]) if was_tested else None
        else:
            for eff in effects:
                # `nan < sig_level` is False, which is indistinguishable from a
                # test that ran and found nothing.
                eff.significant = (
                    None if not np.isfinite(eff.p_value) else bool(eff.p_value < sig_level)
                )

        # Heterogeneity test: are effects different across intersections?
        # Cochran's Q on the POINT effects and their standard errors (the SE
        # of each effect estimated from its bootstrap distribution). Feeding
        # the raw bootstrap replicate arrays into a one-way ANOVA: as done
        # previously: treats every one of the n_boot resamples as an
        # independent observation, inflating the between-group F statistic by
        # roughly n_boot and collapsing het_p toward 0 whenever the point
        # estimates differ at all.
        if len(effects) >= 2:
            point_effects = np.array([e.effect for e in effects])
            ses = np.array(
                [float(np.std(g, ddof=1)) if len(g) > 1 else 0.0 for g in group_effect_lists]
            )
            valid = ses > 0
            if int(valid.sum()) >= 2:
                w = 1.0 / ses[valid] ** 2
                theta = point_effects[valid]
                theta_bar = float(np.sum(w * theta) / np.sum(w))
                q_stat = float(np.sum(w * (theta - theta_bar) ** 2))
                dof = int(valid.sum()) - 1
                if _HAS_SCIPY:
                    het_p = float(scipy_stats.chi2.sf(q_stat, dof))
                else:
                    # Wilson-Hilferty chi-square tail approximation
                    z = ((q_stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(
                        2.0 / (9.0 * dof)
                    )
                    het_p = float(_norm_cdf(-z))
            else:
                # Fewer than two intersections carry a positive standard error,
                # so Cochran's Q has nothing to weight. NaN, not 1.0: see below.
                het_p = float("nan")
        else:
            # Fewer than two analysable intersections, so there is no
            # between-group comparison to make at all. NaN, not 1.0: see below.
            het_p = float("nan")

        # p = 1.0 is the MOST non-significant value there is, so substituting it
        # for a test that could not run reported "no heterogeneity across
        # intersections" as a finding. Both branches above mean the test COULD
        # NOT RUN, and both used to assign 1.0, which sails through the finite
        # check below and reaches het_detected=False. Measured 2026-09-10 on two
        # intersections whose effects were 1.0 and 4.0: heterogeneity=no,
        # p=1.0000, no warning, on repr, to_dict, get_summary and the rendered
        # chart alike. They set NaN so the three-state gate below is reachable.
        het_measured = np.isfinite(het_p)
        if not het_measured:
            warnings.warn(
                "intersectional experiment: the heterogeneity test could not be "
                "computed, so het_detected is None (could not check), not False.",
                UserWarning,
                stacklevel=2,
            )
        het_p = float(het_p) if het_measured else float("nan")
        het_detected = bool(het_p < sig_level) if het_measured else None

        # Overall effect
        ctrl_all = self.control[self.outcome].dropna().values.astype(float)
        treat_all = self.treatment[self.outcome].dropna().values.astype(float)
        overall_eff = float(treat_all.mean() - ctrl_all.mean())

        # Overall bootstrap CI
        n_boot_overall = min(n_boot, 1000)
        boot_overall = []
        for _ in range(n_boot_overall):
            c_s = self._rng.choice(ctrl_all, size=len(ctrl_all), replace=True)
            t_s = self._rng.choice(treat_all, size=len(treat_all), replace=True)
            boot_overall.append(t_s.mean() - c_s.mean())
        # Same order-statistic floor as the per-intersection interval above.
        if n_boot_overall < min_resamples:
            overall_ci = (float("nan"), float("nan"))
        else:
            overall_ci = (
                float(np.percentile(boot_overall, 100 * sig_level / 2)),
                float(np.percentile(boot_overall, 100 * (1 - sig_level / 2))),
            )
        if thin_ci or n_boot_overall < min_resamples:
            warnings.warn(
                f"detect_heterogeneous_effects: {n_boot} bootstrap resample(s) is below "
                f"the {min_resamples} a {1 - sig_level:.0%} percentile interval needs for "
                f"both bounds to be order statistics of the draws, so "
                f"{len(thin_ci) + (1 if n_boot_overall < min_resamples else 0)} confidence "
                f"interval(s) were NOT estimated and their bounds are NaN, not a "
                f"zero-width interval. At n_bootstrap=1 the bounds were equal, which reads "
                f"as an exact estimate. The effects, p-values and Cohen's d are measured "
                f"from the data and are unaffected.",
                UserWarning,
                stacklevel=2,
            )

        # Overall p-value
        if _HAS_SCIPY:
            _, overall_p = scipy_stats.ttest_ind(treat_all, ctrl_all, equal_var=False)
            overall_p = float(overall_p)
        else:
            overall_p = (
                float(
                    2
                    * _norm_cdf(
                        -abs(overall_eff)
                        / np.sqrt(
                            treat_all.var(ddof=1) / len(treat_all)
                            + ctrl_all.var(ddof=1) / len(ctrl_all)
                        )
                    )
                )
                if (treat_all.var() + ctrl_all.var()) > 0
                # Both arms constant: the standard error is zero and the
                # t statistic is undefined, so NO test was run. 1.0 is the most
                # non-significant p there is, and it was being compared against
                # alpha and printed as "not significant (p=1.0000)". It also
                # disagreed with the scipy branch above, which answers 0.0 on
                # the identical arrays. NaN is what the rest of this result
                # already uses for could-not-check, and the renderer reads it
                # through `_finite`.
                else float("nan")
            )

        # Power results. `power.get(ix, 0) >= 0.8` read a MISSING power as a
        # measured zero, and calculate_intersectional_power returns NaN for an
        # intersection below the size floor (see there). `nan >= 0.8` is False,
        # which is indistinguishable from a computed "underpowered".
        power = self.calculate_intersectional_power()
        for eff in effects:
            pwr = power.get(eff.intersection)
            eff.powered = None if pwr is None or not np.isfinite(pwr) else bool(pwr >= 0.8)

        self._result = ExperimentResult(
            overall_effect=overall_eff,
            overall_ci=overall_ci,
            overall_p_value=overall_p,
            intersection_effects=effects,
            heterogeneity_detected=het_detected,
            heterogeneity_p_value=het_p,
            power_results={str(k): v for k, v in power.items()},
            n_intersections=len(effects),
            n_excluded=excluded,
            design_type=self.config.design_type,
            metadata={
                "n_control": len(self.control),
                "n_treatment": len(self.treatment),
                "protected_attributes": self.protected_attrs,
                "outcome_column": self.outcome,
                "business_metrics": self.business_metrics,
                "alpha": sig_level,
                "correction_method": self.config.correction_method,
                # Rows whose protected attribute is ABSENT belong to no
                # intersection (see _generate_intersections). They are in this
                # overall effect and in no per-intersection comparison, so the
                # count travels with the result: ExperimentAnalysis.
                # decision_recommendation reads it and refuses to say "with no
                # harmed groups" while it is non-zero (G04, 2026-09-30).
                "rows_without_intersection": int(
                    getattr(self, "rows_without_intersection", 0) or 0
                ),
            },
        )
        return self._result

    # Convenience

    def run_full_analysis(self) -> ExperimentResult:
        """Run the complete fairness experiment analysis.

        Delegates to :meth:`detect_heterogeneous_effects`, which runs
        :meth:`calculate_intersectional_power` itself and writes its figures
        into ``ExperimentResult.power_results``. (It does not call the two in
        sequence, as this said before G04, 2026-09-30; calling the power
        analysis here as well would run it twice.)

        Returns
        -------
        ExperimentResult
            Carries the same warnings the callee raises, including the absent
            protected-attribute disclosure and the bootstrap-resample floor.
        """
        return self.detect_heterogeneous_effects()

    def get_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict of the experiment.

        ``most_affected`` is the SIGNIFICANT intersection with the largest
        ``|effect|``, or ``None`` when no intersection was both tested and
        significant. ``n_intersections_not_graded`` counts the analysed
        intersections whose significance test could not run at all
        (``significant is None``): they are neither significant nor
        non-significant, so ``n_significant_intersections`` alone does not say
        how much of the experiment carries a verdict.
        """
        result = self._result or self.run_full_analysis()
        sig_effects = [e for e in result.intersection_effects if e.significant is True]
        # G04 (2026-09-30). `sig_effects[0]` is the FIRST significant
        # intersection in iteration order, and `_generate_intersections` sorts
        # by `str`, so the key named a superlative and returned an
        # alphabetical accident. Measured that day on two significant
        # intersections, ('f',) at +0.4836 and ('m',) at +0.5045:
        # most_affected read ('f',), the SMALLER of the two effects. A reader
        # acting on this key investigates the wrong demographic.
        # An effect that is not a finite number cannot take part in a
        # comparison of magnitudes, so it is excluded rather than allowed to
        # win or lose a `max()` by argument order.
        ranked = [e for e in sig_effects if np.isfinite(e.effect)]
        most_affected = max(ranked, key=lambda e: abs(e.effect)).intersection if ranked else None
        n_not_graded = sum(1 for e in result.intersection_effects if e.significant is None)
        return {
            "n_control": len(self.control),
            "n_treatment": len(self.treatment),
            "n_intersections_analysed": result.n_intersections,
            "n_intersections_excluded": result.n_excluded,
            "overall_effect": round(result.overall_effect, 4),
            "overall_p_value": round(result.overall_p_value, 4),
            "heterogeneity_detected": result.heterogeneity_detected,
            "heterogeneity_p_value": round(result.heterogeneity_p_value, 4),
            "n_significant_intersections": len(sig_effects),
            # THREE STATES at the summary surface too: significant /
            # not significant / never tested. Without this key a reader cannot
            # tell "1 of 3 significant" from "1 significant, 1 not, 1 never
            # tested", and the second is the one that needs action.
            "n_intersections_not_graded": n_not_graded,
            "most_affected": most_affected,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-intersection results as a tidy DataFrame."""
        result = self._result or self.run_full_analysis()
        return result.to_dataframe()


# Design Utilities


def assign_clusters(
    df: pd.DataFrame,
    cluster_column: str,
    treatment_fraction: float = 0.5,
    stratify_by: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Assign treatment/control at the cluster level.

    Randomises clusters (not individuals) to avoid spillover effects.
    Optionally stratifies so each stratum has balanced treatment assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a cluster column.
    cluster_column : str
        Column identifying clusters (e.g. ``'city'``, ``'store_id'``).
    treatment_fraction : float
        Fraction of clusters assigned to treatment (default 0.5).
    stratify_by : list[str], optional
        Columns to stratify clusters by before randomising.
    random_state : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Copy of input with an added ``'treatment'`` column (0 or 1).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    clusters = df[cluster_column].unique()

    # A STRATUM WITH ONE CLUSTER CANNOT BE RANDOMISED (2026-09-25).
    #
    # ``n_treat = max(1, int(n * treatment_fraction))`` has no upper clamp, so a
    # stratum holding a single cluster ALWAYS went to treatment: max(1, int(0.5))
    # is 1. Two consequences, both silent.
    #
    #   (a) Over one cluster the function returned a "design" in which every row
    #       was treated, with no control arm, and said nothing. Measured
    #       2026-09-25: 400 rows, one cluster, treatment_fraction=0.5, result
    #       {1: 400}. A cluster-randomised trial with one cluster is not a trial,
    #       and the analyst downstream compares against an empty control.
    #   (b) Stratified, it biases assignment UPWARD wherever strata are small.
    #       Ten singleton strata at treatment_fraction=0.5 put all ten in
    #       treatment: the requested fraction was 0.5 and the delivered one was
    #       1.0, with nothing in the return value recording that.
    #
    # The fix keeps at least one treated cluster, adds the missing clamp so at
    # least one CONTROL cluster survives whenever the count allows it, and
    # discloses the strata it could not randomise. A caller who genuinely wants a
    # full rollout passes treatment_fraction >= 1.0, and that intent is honoured
    # rather than clamped, with a warning that the result has no control arm.
    #
    # BGL5 A-operations-3, 2026-09-27. THAT CLAMP WAS DEAD CODE and is removed.
    # ``max(1, int(n * treatment_fraction))`` already satisfies n_treat <= n - 1
    # for every n >= 2 and every fraction < 1.0 (n * f < n, so int(n * f) <= n - 1),
    # and the clamp was guarded off at fraction >= 1.0, the only case that could
    # have needed it. Measured by sweeping n in 1..2000 against fraction
    # 0.000..1.000 in steps of 0.001: the clamp changed the answer on 0 of
    # 2002000 pairs, and sabotaging it alone left
    # tests/test_bgl2_cicd_and_experimentation.py entirely green. A guard nobody
    # can reach is worse than no guard, because it reads as protection. The two
    # WARNINGS are the load-bearing half of the 2026-09-25 fix and they stay.
    #
    # What the clamp did NOT cover, and what is now disclosed, is the SAME
    # deviation on the lower side, plus rows that could not be assigned at all:
    #   before  20 clusters, treatment_fraction=0.01 -> 1 cluster treated, a
    #           delivered fraction of 0.050, warnings=[] and nothing in the
    #           returned frame: a 1 percent canary ramp silently shipped 5
    #           percent. treatment_fraction=0.0 delivered 0.050 the same way
    #           (max(1, 0) invented a treated cluster nobody asked for), and 0.0
    #           stratified over 10 strata delivered 0.100. 40 of 400 rows whose
    #           cluster id was NaN were written treatment=0, i.e. CONTROL, with
    #           no warning: an assignment nobody randomised.
    #   after   treatment_fraction <= 0.0 is honoured as a deliberate no-rollout
    #           (0 clusters treated; the single-arm warning below then says the
    #           frame is not an experiment), which is the mirror of the >= 1.0
    #           decision above; a delivered fraction more than half a cluster
    #           away from the requested one warns and names both numbers; and
    #           unmappable rows are counted, named and recorded. Every case is
    #           also recorded in out.attrs['assignment'], which is where the
    #           returned frame carries it. 20 clusters at 0.5 still deliver
    #           10/20 = 0.500 in silence, and two clusters at 0.5 still fill
    #           both arms in silence.
    unrandomisable: List[str] = []

    def _n_treated(n: int) -> int:
        if treatment_fraction <= 0.0:
            return 0
        return max(1, int(n * treatment_fraction))

    if stratify_by:
        # Get the modal stratum for each cluster
        cluster_strata = (
            df.groupby(cluster_column)[stratify_by]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
        )
        strata_groups = cluster_strata.groupby(stratify_by)[cluster_column].apply(list)
        assignment = {}
        for stratum, cluster_list in strata_groups.items():
            cl = list(cluster_list)
            rng.shuffle(cl)
            if len(cl) < 2:
                unrandomisable.append(f"{stratum!r} (1 cluster)")
            n_treat = _n_treated(len(cl))
            for i, c in enumerate(cl):
                assignment[c] = 1 if i < n_treat else 0
    else:
        cluster_list = list(clusters)
        rng.shuffle(cluster_list)
        if len(cluster_list) < 2:
            unrandomisable.append(f"the whole frame ({len(cluster_list)} cluster)")
        n_treat = _n_treated(len(cluster_list))
        assignment = {c: (1 if i < n_treat else 0) for i, c in enumerate(cluster_list)}

    mapped = out[cluster_column].map(assignment)
    # Rows whose cluster id is not in the randomisation (a missing/NaN id, or a
    # value that groupby dropped): fillna(0) writes them CONTROL, which is an
    # arm nobody assigned them to. The value is kept for backward compatibility
    # (``treatment`` stays an int 0/1 column) and the count is disclosed instead.
    n_unassignable = int(mapped.isna().sum())
    out["treatment"] = mapped.fillna(0).astype(int)

    if unrandomisable:
        # ``treatment_fraction <= 0.0`` is now honoured, so a cluster that could
        # not be randomised lands in CONTROL there, not treatment. The wording
        # follows the arm the code actually used.
        forced_arm = "control" if treatment_fraction <= 0.0 else "treatment"
        forced_fraction = "0.0" if treatment_fraction <= 0.0 else "1.0"
        warnings.warn(
            f"cluster-level randomisation COULD NOT BE PERFORMED for "
            f"{len(unrandomisable)} stratum/strata, because cluster randomisation "
            f"needs at least two clusters to fill both arms: "
            f"{'; '.join(unrandomisable[:5])}. Those clusters were assigned to "
            f"{forced_arm}, so the delivered treatment fraction there is "
            f"{forced_fraction} and not "
            f"the {treatment_fraction} requested, and those rows carry no "
            f"counterfactual.",
            UserWarning,
            stacklevel=2,
        )

    # What was actually delivered, against what was asked for. Recorded on the
    # frame (the precedent is operations.pulse.traces, which records its
    # ingestion coverage in ``attrs``) so the number travels with the data and
    # not only in a warning the caller may never see.
    n_clusters = len(assignment)
    n_treated_clusters = int(sum(1 for v in assignment.values() if v == 1))
    delivered = (n_treated_clusters / n_clusters) if n_clusters else float("nan")
    requested = min(1.0, max(0.0, float(treatment_fraction)))
    out.attrs["assignment"] = {
        "requested_treatment_fraction": float(treatment_fraction),
        "delivered_treatment_fraction": delivered,
        "n_clusters": n_clusters,
        "n_clusters_treated": n_treated_clusters,
        "n_strata_not_randomised": len(unrandomisable),
        "n_rows_not_assignable": n_unassignable,
        "n_rows_missing_cluster_id": int(out[cluster_column].isna().sum()),
    }

    # Half a cluster is the finest granularity cluster randomisation has, so a
    # deviation within it is rounding and is not worth a warning. Anything
    # larger is a design the caller did not ask for.
    if n_clusters and abs(delivered - requested) > (0.5 / n_clusters):
        warnings.warn(
            f"the DELIVERED treatment fraction is {delivered:.3f} "
            f"({n_treated_clusters} of {n_clusters} cluster(s)), not the "
            f"{treatment_fraction} requested: with {n_clusters} cluster(s) the "
            f"finest step available is {1.0 / n_clusters:.3f}. Read the delivered "
            f"figure, not the requested one, when you report exposure or compute "
            f"an effect; both are recorded in frame.attrs['assignment'].",
            UserWarning,
            stacklevel=2,
        )

    n_missing_id = int(out[cluster_column].isna().sum())
    if n_missing_id and not n_unassignable:
        # The unstratified path reaches these rows through
        # ``df[cluster_column].unique()``, which returns the missing value as a
        # value like any other, so "no cluster id" was randomised as if it were
        # one cluster and counted in n_clusters. Measured: 40 of 400 rows with a
        # NaN id, treatment_fraction=0.5 -> n_clusters 3 (two real ones plus
        # "missing"), the 40 rows all landed in the same arm, and nothing said so.
        warnings.warn(
            f"{n_missing_id} of {len(out)} row(s) carry NO cluster id. The missing "
            f"value was treated as a cluster of its own, so those rows were "
            f"randomised together as one unit and counted among the {n_clusters} "
            f"cluster(s) here. They share no real cluster, so that unit is not a "
            f"randomisation unit: exclude them, or give them an id, if that was "
            f"not intended.",
            UserWarning,
            stacklevel=2,
        )

    if n_unassignable:
        warnings.warn(
            f"{n_unassignable} of {len(out)} row(s) carry a cluster id that is not "
            f"in the randomisation (missing, or never seen when clusters were "
            f"drawn), so they were written as CONTROL (treatment=0) without being "
            f"randomised into it. They are not a measured control arm: drop them, "
            f"or give them a cluster id, before estimating any effect.",
            UserWarning,
            stacklevel=2,
        )

    arms = out["treatment"].nunique()
    if arms < 2:
        # ``iloc[0]`` on a zero-row frame raised IndexError, so an empty input
        # crashed here instead of being told it is not an experiment.
        if not len(out):
            only = "neither: the frame has no rows at all"
        else:
            only = "treatment" if out["treatment"].iloc[0] == 1 else "control"
        warnings.warn(
            f"this assignment produced a SINGLE ARM: every row is {only}. There is "
            f"nothing to compare, so any effect estimated from this frame is not an "
            f"experimental result. Supply at least two clusters, or a "
            f"treatment_fraction that leaves both arms non-empty.",
            UserWarning,
            stacklevel=2,
        )

    return out


def create_factorial_design(
    df: pd.DataFrame,
    factor_columns: List[str],
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Create a 2^k factorial design for multi-factor experiments.

    Randomly assigns each row to one cell of the factorial design space.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    factor_columns : list[str]
        Names for the experimental factors. Each will be assigned 0 or 1.
    random_state : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Copy of input with binary columns for each factor and a
        ``'treatment_cell'`` column encoding the cell index.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    n = len(out)

    for col in factor_columns:
        out[col] = rng.integers(0, 2, size=n)

    # Encode cell index
    out["treatment_cell"] = sum(out[col] * (2**i) for i, col in enumerate(factor_columns))
    return out
