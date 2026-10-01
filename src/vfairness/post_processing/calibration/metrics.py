"""
Calibration Metrics for vfairness.

This module provides comprehensive metrics for evaluating calibration quality,
both overall and across demographic groups. These metrics are essential for
detecting and quantifying calibration disparities that may create unfairness.

Metrics Implemented:
    1. Expected Calibration Error (ECE): Weighted average calibration error
    2. Maximum Calibration Error (MCE): Worst-case calibration error
    3. Brier Score: Proper scoring rule for probability predictions
    4. Brier Score Decomposition: Resolution, reliability, uncertainty
    5. Calibration Curve: Reliability diagram data

Library Comparisons:
    scikit-learn: calibration_curve, brier_score_loss
    netcal: ECE, MCE, and many variants
    uncertainty-calibration: reliability diagrams

References:
    - Naeini, M. P., et al. (2015). Obtaining Well Calibrated Probabilities. AAAI.
    - Kumar, A., et al. (2019). Verified Uncertainty Calibration. NeurIPS.
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
    - DeGroot, M. H. & Fienberg, S. E. (1983). The Comparison and Evaluation
      of Forecasters. The Statistician.
"""

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from vfairness._triage import is_measured as _is_measured
from vfairness._triage import partition_measured as _partition_measured
from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)
from vfairness.evaluation.vfairness_metrics._statistics import (
    StatisticalResult,
    benjamini_hochberg_correction,
    bootstrap_over_index,
    detectability,
    min_attainable_p_permutation,
)
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_binary_labels,
    validate_probabilities,
)


@dataclass
class CalibrationMetricResult:
    """
    Container for calibration metric results.

    Attributes:
        metric_name: Name of the metric
        overall_value: Overall metric value
        group_values: Per-group metric values (if computed)
        n_samples: Total number of samples
        n_bins: Number of bins used
        bin_details: Detailed per-bin statistics
        metadata: Additional information
    """

    metric_name: str
    overall_value: float
    group_values: Optional[Dict[str, float]] = None
    n_samples: int = 0
    n_bins: int = 10
    bin_details: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def n_groups_compared(self) -> int:
        """How many groups carry a MEASURED value, i.e. how many the disparity
        could span.

        0 or 1 means no between-group comparison was made, and
        :attr:`max_group_disparity` is NaN rather than 0.0. It is serialised by
        :meth:`to_dict` so a JSON consumer can tell the third state apart from a
        measured parity of 0.0 without reading this source.

        BGL-final-d00 (2026-09-17). This counted ``len(group_values)``, every
        key including the ones holding NaN, so it asserted a comparison that did
        not happen. Measured before the fix on 2000 healthy rows in group A
        beside 60 single-outcome-class rows in group B::

            calibration_slope(y, p, g).to_dict()
            -> {'group_values': {'A': 0.961, 'B': nan},
                'max_group_disparity': nan,
                'n_groups_compared': 2}      # 2 groups "compared", one of them
                                             # never measured

        and the same for ``calibration_in_the_large``. The disparity beside it
        was already NaN, so the object contradicted itself: a refusal in one
        field and a two-group comparison claim in the next. A NaN entry is a
        group that was NOT compared, so it is not counted. A consumer reading
        ``n_groups_compared`` against ``len(group_values)`` can now see exactly
        how many groups fell out, which is the disclosure the NaN alone does not
        carry.

        G03 (2026-09-30). The test was a LOCAL ``v is not None and
        math.isfinite(float(v))``, not the repository's canonical
        :func:`vfairness._triage.is_measured`, and the two disagree in both
        directions. Measured before the fix on
        ``CalibrationMetricResult("expected_calibration_error", 0.04,
        {"a": 0.02, "b": <value>})``:

        - ``<value> = True`` -> ``n_groups_compared 2`` and
          ``max_group_disparity 0.98``. ``bool`` is a subclass of ``int`` and
          ``float(True) == 1.0``, so a yes/no FLAG was counted as a measured
          group and became a near-maximal disparity on the ECE scale. Rule 2 of
          ``_triage``'s docstring exists for exactly this, and
          ``visualization.py`` in this same package already imports it.
        - ``<value> = "0.9"`` -> ``n_groups_compared 2``, and then
          ``max_group_disparity`` raised ``TypeError: '>' not supported between
          instances of 'str' and 'float'``: the COUNT coerced where the
          ARITHMETIC could not, so the two halves of one object disagreed about
          whether the same value was a measurement.
        - ``<value>`` in ``""``, ``"None"``, ``pd.NA``, ``pd.NaT`` -> ``to_dict()``
          raised ``ValueError``/``TypeError`` out of ``float()``. Absence reaches
          a metric through all six of those doors, and an opaque crash inside the
          serialiser is not a disclosure of it.

        ``is_measured`` answers all five the same way and accepts a finite
        ``np.float32``, which a local ``isinstance(v, (int, float))`` would have
        discarded.
        """
        # Written as one expression on purpose. An ``if group_values is None:
        # return 0`` guard is a neutral literal returned under an emptiness
        # test. That is the shape a "no problem" default usually arrives in,
        # and this repository's fabrication scanner flags it as a candidate for
        # exactly that reason. Here 0 really is the count, but the cheapest way
        # to keep that clear is not to write the shape at all.
        return sum(1 for v in (self.group_values or {}).values() if _is_measured(v))

    @property
    def max_group_disparity(self) -> float:
        """Maximum difference between any two groups. NaN when fewer than two
        groups carry a value: no comparison was made.

        THREE states, never two. A number is a measured gap, NaN is
        could-not-check, and there is no third reading of 0.0 that means
        "nothing was compared".

        BGL-S2G08 (2026-09-16). This returned a literal ``0.0`` whenever
        ``group_values`` was None or held fewer than two entries, which is
        PERFECT PARITY ON EVERY SCALE reported for a comparison that never
        happened, and :meth:`to_dict` serialised it into every JSON consumer
        while ``visualization.py`` drew it as the 'ECE Disparity' bar under a
        0.05 target line. Measured before the fix, group sizes a=100 and b=1
        with the default ``min_group_size=30`` silently dropping b:
        ``expected_calibration_error(...).to_dict()`` ->
        ``{'group_values': {'a': 0.1796}, 'max_group_disparity': 0.0}``, and on
        ZERO rows ``brier_score(...)`` -> ``overall_value nan`` beside
        ``max_group_disparity 0.0``. The sibling
        :attr:`CalibrationDisparityResult.ece_disparity` in this same file was
        already NaN for exactly this condition, with a warning that reads
        "NaN, not 0.0"; this property applied the opposite standard silently.
        """
        if self.group_values is None or len(self.group_values) < 2:
            return float("nan")
        values = list(self.group_values.values())
        # Python's max/min SILENTLY SKIP NaN, and which way they fall depends on
        # the order the values arrive in. Measured 2026-09-16 on [0.0226, nan]:
        # max - min is 0.0, i.e. PERFECT PARITY for a group nobody could measure;
        # with the same two values in the other order it is nan. The verdict
        # therefore depended on dict insertion order, which is the order the
        # groups happened to appear in the data.
        #
        # This became reachable when group_values gained NaN members: before the
        # could-not-check work every value was a float, so the hole was latent.
        # Three independent auditors found it in the same afternoon.
        #
        # The spread over the MEASURED subset is not a safe substitute: the
        # missing group could sit anywhere, so that spread is a lower bound, not
        # the disparity. NaN is the answer, and it is the rule the sibling
        # CalibrationDisparityResult.ece_disparity in this file already applies.
        # A caller that wants the measured subset still has group_values.
        #
        # G03 (2026-09-30): the test is the canonical ``_triage.is_measured``
        # rather than a local ``math.isfinite(float(v))``, which raised out of
        # ``float()`` on a blank string, the literal "None", ``pd.NA`` and
        # ``pd.NaT``, and accepted a ``bool`` as a measurement worth 1.0. See
        # :attr:`n_groups_compared` for the measurements.
        if any(not _is_measured(v) for v in values):
            return float("nan")
        return max(float(v) for v in values) - min(float(v) for v in values)

    @property
    def measured_subset_disparity(self) -> float:
        """The spread across the groups that WERE measured. A LOWER BOUND on
        :attr:`max_group_disparity` whenever some group could not be measured,
        and equal to it when all of them were.

        BGL-final-d00 (2026-09-17). This exists so that refusing to state the
        overall disparity does not also throw away the evidence there IS.
        :attr:`max_group_disparity` must stay NaN when any group is unmeasured,
        because a spread over a SUBSET is not the disparity and this class has
        no field on which to hang that caveat. But with three groups at
        ``{'A': 0.1, 'B': 0.9, 'C': nan}`` the 0.8 gap between A and B is a real,
        determinate finding, and a reader who sees only NaN has lost it. The
        threshold-optimization module already resolves the same tension the same
        way (``ConstraintViolation.violation_is_lower_bound``): keep the partial
        measurement, and label it as partial.

        Read it against :attr:`n_groups_compared`: equal to ``len(group_values)``
        means this IS the disparity, fewer means it is a floor under it. NaN
        when fewer than two groups were measured, because then there is no pair
        and therefore no bound either.
        """
        # G03 (2026-09-30): the canonical ``_triage.is_measured``, for the same
        # reason as :attr:`n_groups_compared`. A local ``float(v)`` inside the
        # comprehension raised on four of the six absence doors and let a bool
        # through as 1.0.
        finite = [float(v) for v in (self.group_values or {}).values() if _is_measured(v)]
        if len(finite) < 2:
            return float("nan")
        return max(finite) - min(finite)

    # Per-metric rule-of-thumb thresholds for is_well_calibrated. A single
    # ECE-tuned cutoff (0.05) mislabelled MCE and Brier results: MCE is a
    # worst-case bin error so a looser bound applies, and the Brier score
    # of an uninformative constant p=0.5 forecast is already 0.25.
    _WELL_CALIBRATED_THRESHOLDS = {
        "expected_calibration_error": 0.05,
        "maximum_calibration_error": 0.10,
        "brier_score": 0.25,
    }

    # Metrics whose overall_value is a TARGET-CENTERED statistic, not an error.
    # The calibration slope's ideal is 1.0 (seal band [0.8, 1.2]); the ECE-style
    # "value < threshold" fallback INVERTED its verdict: a signal-free slope of
    # 0.006 read as well calibrated while a near-perfect 0.96 read as bad, and
    # to_dict() serialized that into every JSON consumer.
    #
    # calibration_in_the_large joins this table rather than the threshold table
    # above, and it is not a demotion: its overall_value is ``abs(overall_signed)``
    # (see :func:`calibration_in_the_large`), a magnitude around a target of
    # exactly 0.0, which is what a two-sided band expresses. A band is graded
    # WITHOUT consulting a better-direction, and that matters here, because the
    # shared resolver returns UNKNOWN for this name and this file may not extend
    # its table. Measured 2026-09-10: this preserves the verdict the 0.05 fallback
    # produced for every value it ever saw (the only difference is at exactly
    # 0.05, which the strict ``<`` failed and the closed band passes).
    _WELL_CALIBRATED_BANDS = {
        "calibration_slope": (0.8, 1.2),
        "calibration_in_the_large": (0.0, 0.05),
    }

    @property
    def is_well_calibrated(self) -> Optional[bool]:
        """Metric-aware calibration check. THREE states, never two.

        ``True`` well calibrated, ``False`` not well calibrated, ``None``
        could-not-check: no curated bound exists for this metric name, the
        shared resolver cannot say which direction is better, or the value is
        not a number. ``None`` is not a synonym for either verdict, and a
        consumer that collapses it into one is asserting a check nobody ran.

        Band metrics test a target band (calibration_slope 0.8 <= value <= 1.2,
        calibration_in_the_large |value| <= 0.05). Error thresholds: ECE 0.05,
        MCE 0.10, Brier 0.25, each compared IN ITS OWN DIRECTION by
        :func:`..._metric_direction.check_threshold`. Note the Brier score mixes
        calibration with refinement, so treat its verdict as a coarse screen,
        not a pure calibration statement.

        The fallback this replaces was ``self.overall_value < threshold`` with a
        hardcoded ``threshold = ...get(self.metric_name, 0.05)``: a
        lower-is-better rule and an ECE-shaped bound, applied to ANY name, with
        no third state. Measured on this repo 2026-09-10, before the fix:

        - ``R("some_new_metric", 0.04).is_well_calibrated`` -> ``True`` and
          ``R("some_new_metric", 0.06)`` -> ``False``, a graded verdict for a
          metric ``metric_direction`` reports as UNKNOWN. Two guesses, no
          refusal.
        - Hand it a higher-is-better name and the verdict inverts outright:
          ``R("worst_group_accuracy", 0.04)`` -> ``True`` (a model whose worst
          group is right 4 percent of the time, "well calibrated") and
          ``R("worst_group_accuracy", 0.99)`` -> ``False``. Same for
          ``disparate_impact_ratio`` and ``exposure_parity_ratio``.
          ``to_dict()`` serialised each of those into every JSON consumer.
        - A NaN value returned ``False``, because ``nan < 0.05`` is False: a
          "not well calibrated" verdict for a metric that was never computed.
          ``analyzer.py`` had already routed the same NaN case to None locally
          and left the hole open here.

        The comment above :data:`_WELL_CALIBRATED_BANDS` records that this very
        fallback shipped an inverted calibration_slope once already. The fix is
        not a better hardcoded rule, it is asking the module that owns the
        direction table.
        """
        band = self._WELL_CALIBRATED_BANDS.get(self.metric_name)
        if band is not None:
            # A band is target-centred and needs no better-direction, but it
            # still needs a number: every comparison against NaN is False, so an
            # uncomputable value used to read as "outside the band".
            if not np.isfinite(self.overall_value):
                return None
            return bool(band[0] <= self.overall_value <= band[1])

        threshold = self._WELL_CALIBRATED_THRESHOLDS.get(self.metric_name)
        if threshold is None:
            # No curated bound for this metric. There is nothing to grade it
            # against, and borrowing the ECE bound is how an arbitrary name got
            # an ECE-shaped verdict.
            return None

        outcome, _ = check_threshold(self.metric_name, self.overall_value, threshold)
        if outcome is ThresholdOutcome.PASS:
            return True
        if outcome is ThresholdOutcome.FAIL:
            return False
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "overall_value": self.overall_value,
            "group_values": self.group_values,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "max_group_disparity": self.max_group_disparity,
            # The spread over the groups that WERE measured. Equal to the field
            # above when every group was measured; a LOWER BOUND under it when
            # the field above is NaN because one was not. Carried in the dict so
            # a JSON consumer keeps the partial evidence instead of only a
            # refusal.
            "measured_subset_disparity": self.measured_subset_disparity,
            # How many groups the disparity above spans. NaN disparity beside
            # 0 or 1 here is could-not-check, not measured parity.
            "n_groups_compared": self.n_groups_compared,
            "is_well_calibrated": self.is_well_calibrated,
            "metadata": self.metadata,
        }


@dataclass
class BrierDecomposition:
    """
    Brier Score decomposition into reliability, resolution, and uncertainty.

    The Brier score can be decomposed as:
        Brier = Reliability - Resolution + Uncertainty

    Where:
        - Reliability: How well predicted probabilities match observed frequencies
        - Resolution: How well predictions discriminate between outcomes
        - Uncertainty: Inherent uncertainty in the base rate

    Attributes:
        brier_score: Overall Brier score
        reliability: Reliability component (lower is better)
        resolution: Resolution component (higher is better)
        uncertainty: Uncertainty component (fixed for dataset)
        n_samples: Number of samples
        n_bins: Number of bins used for decomposition
    """

    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    n_samples: int
    n_bins: int = 10

    @property
    def skill_score(self) -> float:
        """Brier Skill Score relative to climatology, or NaN when undefined.

        BSS = 1 - Brier / Uncertainty. When every label is the same class the
        climatological variance ``base_rate * (1 - base_rate)`` is exactly 0 and
        the ratio is UNDEFINED: there is no climatology to be more skilful than.

        BGL-S2G08 (2026-09-16). This returned ``0.0`` there, which is not a
        missing value but a specific, plottable verdict on the very same scale:
        "exactly as skilful as climatology", a score genuinely attainable when
        uncertainty is positive. Measured before the fix on 200 rows with every
        label 1: ``{'brier_score': 0.2774, 'uncertainty': 0.0, 'skill_score':
        0.0}`` with ``np.isfinite(skill_score)`` True and zero warnings, and
        ``to_dict()`` shipped that 0.0 to every consumer. NaN is what the rest
        of this module uses for could-not-check.
        """
        if not np.isfinite(self.uncertainty) or self.uncertainty == 0:
            return float("nan")
        return 1 - self.brier_score / self.uncertainty

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "brier_score": self.brier_score,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "skill_score": self.skill_score,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
        }


@dataclass
class CalibrationCurveResult:
    """
    Calibration curve (reliability diagram) data.

    Attributes:
        prob_true: Empirical probabilities in each bin (fraction of positives)
        prob_pred: Mean predicted probability in each bin
        bin_counts: Number of samples in each bin
        bin_edges: Bin boundaries
        overall_ece: Expected Calibration Error
    """

    prob_true: np.ndarray
    prob_pred: np.ndarray
    bin_counts: np.ndarray
    bin_edges: np.ndarray
    overall_ece: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "prob_true": self.prob_true.tolist(),
            "prob_pred": self.prob_pred.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "bin_edges": self.bin_edges.tolist(),
            "overall_ece": self.overall_ece,
        }


@dataclass
class CalibrationDisparityResult:
    """
    Result of calibration disparity analysis across groups.

    Attributes:
        ece_disparity: Maximum ECE difference between groups. NaN when
            fewer than two groups reached min_group_size: no comparison
            was made, so there is no disparity to report (not 0.0)
        mce_disparity: Maximum MCE difference between groups (NaN as above)
        brier_disparity: Maximum Brier score difference (NaN as above)
        group_ece: ECE values per group
        group_mce: MCE values per group
        group_brier: Brier scores per group
        most_miscalibrated_group: Group with highest ECE
        least_miscalibrated_group: Group with lowest ECE
        n_groups: Number of groups analyzed
        recommendations: Suggested interventions
        excluded_groups: Groups dropped because they fell below
            min_group_size; their calibration is NOT covered by the
            disparity figures above
        groups_without_measured_calibration: derived, not stored. Groups that
            CLEARED min_group_size and whose own ECE still is not a number, so
            they are not covered by the disparity figures either. The sibling
            door to excluded_groups; see the property
    """

    ece_disparity: float
    mce_disparity: float
    brier_disparity: float
    group_ece: Dict[str, float]
    group_mce: Dict[str, float]
    group_brier: Dict[str, float]
    most_miscalibrated_group: str
    least_miscalibrated_group: str
    n_groups: int
    recommendations: List[str] = field(default_factory=list)
    excluded_groups: List[str] = field(default_factory=list)

    @property
    def groups_without_measured_calibration(self) -> List[str]:
        """Groups INSIDE the size gate whose own ECE is not a measurement.

        Derived from the VALUES of ``group_ece`` rather than stored, so it
        cannot go stale and so it answers for a hand-built result too. It is the
        sibling door to :attr:`excluded_groups`, which holds only what the
        row-COUNT gate removed.

        BGL5 wave 4 (2026-09-30) fixed this in ``plot_calibration_disparity``,
        which derives the same list from the same values; the number and the
        verdict beside the chart did not have it. See
        :attr:`has_significant_disparity`.
        """
        return [str(g) for g, v in (self.group_ece or {}).items() if not _is_measured(v)]

    @property
    def has_significant_disparity(self) -> Optional[bool]:
        """Whether the ECE disparity exceeds 0.05: True / False / None.

        None means the disparity could not be measured (fewer than two
        groups reached min_group_size, so ece_disparity is NaN). It is
        not False: `nan > 0.05` is False, which would have turned an
        unmeasured comparison into a "no disparity" verdict (CAL-DISP,
        2026-09-09).

        BGL5 (2026-09-27), the same collapse one step along: a CLEAN verdict
        over a SUBSET of the groups, published as the verdict. ``ece_disparity``
        is the max-min spread over the groups that cleared ``min_group_size``,
        so with a group excluded it is a LOWER BOUND and the excluded group can
        decide the comparison either way. MEASURED on 150 'a' + 150 'b' + 25 'c'
        with 'c' scored 0.97 against an outcome of 0, min_group_size=30:
        excluded_groups ['c'], ece_disparity 0.029657 over ['a', 'b'], and this
        property returned False, which ``CalibrationReport`` and
        ``explainer._explain_calibration`` published as "Calibration analysis
        for 3 groups. ECE = 0.1027. No significant calibration disparity." The
        SAME data at min_group_size=20, which lets 'c' in, gives ece_disparity
        0.937755 and "Significant calibration disparity across groups": the
        group left out is the one that flips the verdict. It now returns None
        there, and the summary reads "Calibration disparity COULD NOT CHECK: no
        significance verdict was measured, so this is not a finding of no
        disparity."

        A determinate BREACH on partial evidence KEEPS its finding, exactly as
        ``threshold_optimization.constraints.compute_constraint_violation``
        keeps one: a spread already over the threshold cannot be argued away by
        a group nobody measured, and withholding it would be the reverse
        fabrication. Measured on 100 'a' + 100 'b' + 25 'c' with 'b'
        miscalibrated, ece_disparity 0.822 with excluded_groups ['c'] returned
        True before and returns True now.

        G03 (2026-09-30), the SIBLING DOOR the BGL5 guard could not see. That
        guard keys on ``excluded_groups``, which is a row-COUNT gate: it holds
        only the groups ``min_group_size`` kept out. A group that CLEARS the
        size gate and whose own ECE is still not a number is not in that list,
        so the clean verdict came back as a plain ``False``. MEASURED before this
        fix, on 240 rows in two 120-row groups with group B carrying no ground
        truth at all, at the default gate of 30::

            group_ece        {'A': 0.58, 'B': nan}
            excluded_groups  []
            ece_disparity    0.0          # max()-min() skipping the NaN
            has_significant_disparity     False
            recommendations  ['LOW: Calibration disparity is acceptable (0.000).', ...]
            warnings         []

        Group A's own ECE of 0.58 is a real and appalling measurement, and the
        object reported PERFECT PARITY beside it with no warning of any kind.
        ``plot_calibration_disparity`` was fixed for this input on 2026-09-30 and
        already derives the same list from the same values; the verdict that sits
        beside the chart applied the count gate only.
        """
        if not np.isfinite(self.ece_disparity):
            return None
        breach = bool(self.ece_disparity > 0.05)
        if breach or not (self.excluded_groups or self.groups_without_measured_calibration):
            return breach
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "ece_disparity": self.ece_disparity,
            "mce_disparity": self.mce_disparity,
            "brier_disparity": self.brier_disparity,
            "group_ece": self.group_ece,
            "group_mce": self.group_mce,
            "group_brier": self.group_brier,
            "most_miscalibrated_group": self.most_miscalibrated_group,
            "least_miscalibrated_group": self.least_miscalibrated_group,
            "n_groups": self.n_groups,
            "has_significant_disparity": self.has_significant_disparity,
            "recommendations": self.recommendations,
            "excluded_groups": self.excluded_groups,
            # The sibling door to excluded_groups, serialised for the same
            # reason: the disclosure has to survive the boundary a dashboard or
            # a stored record reads, or it exists only for a caller who holds
            # the dataclass. G03, 2026-09-30.
            "groups_without_measured_calibration": self.groups_without_measured_calibration,
        }


def _disclose_excluded_groups(
    gm: GroupManager, min_group_size: int, metric_label: str
) -> List[str]:
    """Names of the groups dropped for falling below ``min_group_size``, warned
    about once and returned for ``metadata['excluded_groups']``.

    BGL-S2G08 (2026-09-16). ``expected_calibration_error`` /
    ``maximum_calibration_error`` / ``brier_score`` dropped undersized groups
    inside ``if np.sum(mask) >= min_group_size`` and said nothing at all:
    measured on a=100, b=1 the result carried ``group_values {'a': 0.1796}``
    with ZERO warnings, so the only trace of group b was its absence from a
    dict nobody compares against the input. ``calibration_disparity`` in this
    same file already discloses exactly this; the three metrics it is built
    from did not.
    """
    # THIS HELPER READS THE EMPTY LIST AS A DROP LIST, not as a clean bill, so a
    # threshold that cannot flag anything means "none were dropped", which is TRUE.
    # GroupManager.get_invalid_groups gained a disclosure on 2026-09-27 for a
    # min_group_size of 1 or less, because every group there is built from levels
    # that actually occur so no group can ever be too small, and a caller reading
    # that empty list as "every group is large enough" is reading a vacuous answer.
    #
    # Measured: group_calibrator.py:493 and :496 build their GroupManager with
    # min_group_size=1 DELIBERATELY, to include every present group in the
    # intervention view, and reach this helper. The disclosure then fired on a
    # fully covered run and tests/test_surface_grade_g015.py's control, whose
    # subject is that a complete analysis says nothing, went red on `caught == []`.
    #
    # The same early return was already added to the twin in
    # evaluation/vfairness_metrics/classification.py::_warn_dropped_groups for the
    # same reason. The disclosure itself stays where it is: a caller that reads the
    # empty list AS a finding still needs it, and report.py:705 / :1121 do exactly
    # that, which is recorded as open work rather than suppressed here.
    if getattr(gm, "min_group_size", min_group_size) <= 1:
        return []
    excluded = gm.get_invalid_groups()
    if excluded:
        sizes = gm.get_group_sizes()
        detail = {g: sizes.get(g, 0) for g in excluded}
        warnings.warn(
            f"Groups excluded from the {metric_label} group analysis (below "
            f"min_group_size={min_group_size}): {detail}. Their calibration was "
            f"NOT assessed, and the group values and disparity do not cover them."
        )
    return excluded


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
    power: float = 1.0,
) -> CalibrationMetricResult:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed
    frequencies, weighted by bin population. A well-calibrated model has
    ECE close to 0.

    Formula:
        ECE = sum_{b=1}^{B} (n_b / N) * |accuracy_b - confidence_b|^p

    Library Comparisons:
        scikit-learn: Not directly available, use calibration_curve
        netcal: ECE with various binning strategies
        tensorflow-probability: expected_calibration_error

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        protected_attr: Optional protected attribute for group analysis
        n_bins: Number of bins for discretization
        strategy: Binning strategy ('uniform' or 'quantile')
        min_group_size: Minimum samples per group for group analysis
        power: Power for ECE calculation (1 for ECE, 2 for MCE^2)

    Returns:
        CalibrationMetricResult with ECE values

    Example:
        >>> result = expected_calibration_error(y_true, y_prob, gender)
        >>> print(f"Overall ECE: {result.overall_value:.3f}")
        >>> for group, ece in result.group_values.items():
        ...     print(f"  {group}: {ece:.3f}")

    References:
        Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well
        Calibrated Probabilities Using Bayesian Binning. AAAI.

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

    Ledger row: expected_calibration_error. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    # Non-binary labels silently inflate per-bin 'accuracy' and can push
    # ECE far above 1, so reject them loudly.
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    n = len(y_true)

    def compute_ece_single(y_t: np.ndarray, y_p: np.ndarray) -> Tuple[float, Dict]:
        """Compute ECE for a single group."""
        n_local = len(y_t)
        if n_local < 2:
            return np.nan, {}

        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        else:  # quantile
            bin_edges = np.percentile(y_p, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(np.clip(bin_edges, 0, 1))
            if len(bin_edges) < 2:
                # Constant probabilities collapse every quantile edge to one
                # value, leaving zero bins and a vacuous ECE of 0.0 ('perfect
                # calibration'). Use a single full bin instead, so the metric
                # equals |mean(y) - mean(p)| as it should.
                bin_edges = np.array([0.0, 1.0])
            bin_edges[0] = 0
            bin_edges[-1] = 1

        bin_indices = np.digitize(y_p, bin_edges[1:-1])

        ece = 0.0
        bin_details: Dict[str, list] = {
            "accuracies": [],
            "confidences": [],
            "counts": [],
            "contributions": [],
        }

        actual_bins = len(bin_edges) - 1
        for i in range(actual_bins):
            mask = bin_indices == i
            n_bin = np.sum(mask)

            if n_bin > 0:
                accuracy = np.mean(y_t[mask])
                confidence = np.mean(y_p[mask])
                contribution = (n_bin / n_local) * np.abs(accuracy - confidence) ** power

                ece += contribution

                bin_details["accuracies"].append(accuracy)
                bin_details["confidences"].append(confidence)
                bin_details["counts"].append(n_bin)
                bin_details["contributions"].append(contribution)
            else:
                bin_details["accuracies"].append(np.nan)
                bin_details["confidences"].append(np.nan)
                bin_details["counts"].append(0)
                bin_details["contributions"].append(0)

        return ece, bin_details

    overall_ece, overall_details = compute_ece_single(y_true, y_prob)

    group_values = None
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)

        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = _disclose_excluded_groups(gm, min_group_size, "expected calibration error")
        group_values = {}

        for group_name in gm.get_valid_groups():
            mask = gm.get_mask(group_name)
            if np.sum(mask) >= min_group_size:
                group_ece, _ = compute_ece_single(y_true[mask], y_prob[mask])
                group_values[group_name] = group_ece

    return CalibrationMetricResult(
        metric_name="expected_calibration_error",
        overall_value=overall_ece,
        group_values=group_values,
        n_samples=n,
        n_bins=n_bins,
        bin_details={
            "accuracies": np.array(overall_details.get("accuracies", [])),
            "confidences": np.array(overall_details.get("confidences", [])),
            "counts": np.array(overall_details.get("counts", [])),
        },
        metadata={"strategy": strategy, "power": power, "excluded_groups": excluded},
    )


def maximum_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
) -> CalibrationMetricResult:
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the worst-case calibration error across all bins.
    Useful for applications where worst-case performance matters.

    Formula:
        MCE = max_{b=1}^{B} |accuracy_b - confidence_b|

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Optional protected attribute for group analysis
        n_bins: Number of bins
        strategy: Binning strategy
        min_group_size: Minimum samples per group

    Returns:
        CalibrationMetricResult with MCE values

    Example:
        >>> result = maximum_calibration_error(y_true, y_prob)
        >>> print(f"MCE: {result.overall_value:.3f}")

    References:
        Naeini, M. P., et al. (2015). AAAI.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: maximum_calibration_error. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    n = len(y_true)

    def compute_mce_single(y_t: np.ndarray, y_p: np.ndarray) -> float:
        """Compute MCE for a single group."""
        if len(y_t) < 2:
            return np.nan

        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        else:
            bin_edges = np.percentile(y_p, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(np.clip(bin_edges, 0, 1))
            if len(bin_edges) < 2:
                # Constant probabilities collapse every quantile edge to one
                # value, leaving zero bins and a vacuous MCE of 0.0. Use a
                # single full bin instead.
                bin_edges = np.array([0.0, 1.0])

        bin_indices = np.digitize(y_p, bin_edges[1:-1])

        max_error = 0.0
        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_t[mask])
                confidence = np.mean(y_p[mask])
                error = np.abs(accuracy - confidence)
                max_error = max(max_error, error)

        return max_error

    overall_mce = compute_mce_single(y_true, y_prob)

    group_values = None
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)

        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = _disclose_excluded_groups(gm, min_group_size, "maximum calibration error")
        group_values = {}

        for group_name in gm.get_valid_groups():
            mask = gm.get_mask(group_name)
            if np.sum(mask) >= min_group_size:
                group_values[group_name] = compute_mce_single(y_true[mask], y_prob[mask])

    return CalibrationMetricResult(
        metric_name="maximum_calibration_error",
        overall_value=overall_mce,
        group_values=group_values,
        n_samples=n,
        n_bins=n_bins,
        metadata={"strategy": strategy, "excluded_groups": excluded},
    )


def brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    min_group_size: int = 30,
) -> CalibrationMetricResult:
    """
    Compute Brier Score.

    The Brier score is the mean squared error of probability predictions.
    It is a proper scoring rule that measures both calibration and refinement.

    Formula:
        Brier = (1/N) * sum((y_prob - y_true)^2)

    Library Comparisons:
        scikit-learn: brier_score_loss
        netcal: BrierScore

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Optional protected attribute for group analysis
        min_group_size: Minimum samples per group

    Returns:
        CalibrationMetricResult with Brier score

    Example:
        >>> result = brier_score(y_true, y_prob)
        >>> print(f"Brier Score: {result.overall_value:.3f}")

    Notes:
        - Perfect predictions: Brier = 0
        - Random predictions (p=0.5): Brier = 0.25
        - Worst predictions (always wrong): Brier = 1

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

    Ledger row: brier_score. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    n = len(y_true)
    overall_brier = np.mean((y_prob - y_true) ** 2)

    group_values = None
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)

        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = _disclose_excluded_groups(gm, min_group_size, "Brier score")
        group_values = {}

        for group_name in gm.get_valid_groups():
            mask = gm.get_mask(group_name)
            if np.sum(mask) >= min_group_size:
                group_values[group_name] = np.mean((y_prob[mask] - y_true[mask]) ** 2)

    return CalibrationMetricResult(
        metric_name="brier_score",
        overall_value=overall_brier,
        group_values=group_values,
        n_samples=n,
        metadata={"interpretation": "lower is better", "excluded_groups": excluded},
    )


def brier_score_decomposition(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> BrierDecomposition:
    """
    Decompose Brier Score into reliability, resolution, and uncertainty.

    This decomposition provides insight into different aspects of
    probabilistic forecast quality:
        - Reliability: Calibration quality (should be low)
        - Resolution: Discriminative ability (should be high)
        - Uncertainty: Inherent outcome variability (fixed)

    Formula:
        Brier = Reliability - Resolution + Uncertainty

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for decomposition
        strategy: Binning strategy

    Returns:
        BrierDecomposition with all components

    Example:
        >>> decomp = brier_score_decomposition(y_true, y_prob)
        >>> print(f"Reliability: {decomp.reliability:.3f}")
        >>> print(f"Resolution: {decomp.resolution:.3f}")
        >>> print(f"Skill Score: {decomp.skill_score:.3f}")

    References:
        DeGroot, M. H. & Fienberg, S. E. (1983). The Comparison and
        Evaluation of Forecasters. The Statistician, 32(1-2), 12-22.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: brier_score_decomposition. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    n = len(y_true)
    # Zero rows: np.mean of an empty slice is NaN plus a RuntimeWarning, and the
    # NaN is the right answer. Say it directly so the refusal is deliberate and
    # the numpy warning does not drown the UserWarning below.
    base_rate = float(np.mean(y_true)) if n > 0 else float("nan")

    # Uncertainty: variance of the outcome
    uncertainty = base_rate * (1 - base_rate)

    # BGL-S2G08 (2026-09-16). Test the RAW labels, not the accumulated
    # uncertainty: a single-class outcome has NO climatology to be more skilful
    # than, so the Brier Skill Score is undefined (see
    # BrierDecomposition.skill_score, which used to answer 0.0 there).
    if n > 0 and np.unique(y_true).size < 2:
        warnings.warn(
            f"Every label is the same class (base rate {base_rate:.3f}), so the "
            f"climatological uncertainty is 0 and the Brier Skill Score is "
            f"undefined. skill_score is NaN, not 0.0: no skill comparison was made."
        )

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(np.clip(bin_edges, 0, 1))
        if len(bin_edges) < 2:
            # Constant probabilities collapse every quantile edge to one
            # value, leaving zero bins and a vacuous ECE of 0.0 ('perfect
            # calibration'). Use a single full bin instead, so the metric
            # equals |mean(y) - mean(p)| as it should.
            bin_edges = np.array([0.0, 1.0])

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    reliability = 0.0
    resolution = 0.0
    n_bins_populated = 0

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        n_bin = np.sum(mask)

        if n_bin > 0:
            n_bins_populated += 1
            # Observed frequency in this bin
            o_k = np.mean(y_true[mask])
            # Mean forecast in this bin
            f_k = np.mean(y_prob[mask])

            # Reliability contribution
            reliability += (n_bin / n) * (f_k - o_k) ** 2

            # Resolution contribution
            resolution += (n_bin / n) * (o_k - base_rate) ** 2

    if n_bins_populated == 0:
        # No bin received a sample, so neither component accumulated anything and
        # the 0.0 initialisers were about to be returned as measurements:
        # reliability 0.0 is PERFECT reliability and resolution 0.0 is the worst
        # attainable resolution, both on zero evidence. Measured before the fix
        # on zero rows: {'reliability': 0.0, 'resolution': 0.0}.
        reliability = float("nan")
        resolution = float("nan")
        warnings.warn(
            "No prediction bin received a sample, so the Brier decomposition "
            "measured nothing. reliability and resolution are NaN, not 0.0."
        )

    # Compute overall Brier score for verification
    brier = np.mean((y_prob - y_true) ** 2) if n > 0 else float("nan")

    return BrierDecomposition(
        brier_score=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=float(uncertainty),
        n_samples=n,
        n_bins=len(bin_edges) - 1,
    )


def calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    normalize: bool = False,
) -> CalibrationCurveResult:
    """
    Compute calibration curve data for reliability diagrams.

    Returns the empirical probability versus mean predicted probability
    for each bin, along with sample counts.

    Library Comparisons:
        scikit-learn: calibration_curve

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
        normalize: Whether to normalize predictions to [0, 1]

    Returns:
        CalibrationCurveResult with curve data

    Example:
        >>> curve = calibration_curve(y_true, y_prob, n_bins=10)
        >>> plt.plot(curve.prob_pred, curve.prob_true, 'o-')
        >>> plt.plot([0, 1], [0, 1], '--', color='gray')
        >>> plt.xlabel('Mean Predicted Probability')
        >>> plt.ylabel('Fraction of Positives')

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: calibration_curve. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")

    if normalize:
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)

    validate_probabilities(y_prob, "y_prob")

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(np.clip(bin_edges, 0, 1))
        if len(bin_edges) < 2:
            # Constant probabilities collapse every quantile edge to one
            # value, leaving zero bins and a vacuous ECE of 0.0 ('perfect
            # calibration'). Use a single full bin instead, so the metric
            # equals |mean(y) - mean(p)| as it should.
            bin_edges = np.array([0.0, 1.0])
        bin_edges[0] = 0
        bin_edges[-1] = 1

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    prob_true = []
    prob_pred = []
    bin_counts = []
    ece = 0.0
    n = len(y_true)

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        n_bin = np.sum(mask)

        if n_bin > 0:
            mean_true = np.mean(y_true[mask])
            mean_pred = np.mean(y_prob[mask])

            prob_true.append(mean_true)
            prob_pred.append(mean_pred)
            bin_counts.append(n_bin)

            ece += (n_bin / n) * np.abs(mean_true - mean_pred)

    if not bin_counts:
        # BGL-S2G08 (2026-09-16). The accumulator above is guarded by
        # ``if n_bin > 0``, so when no bin holds a sample the ``ece = 0.0``
        # INITIALISER was returned untouched, and 0.0 is the best attainable
        # value on this scale: PERFECT CALIBRATION reported for a dataset with
        # nothing in it. Measured before the fix on zero rows:
        # ``{'prob_true': [], 'prob_pred': [], 'bin_counts': [],
        # 'overall_ece': 0.0}``, and the public plot consumer
        # ``plot_reliability_diagram`` captioned the empty chart 'ECE = 0.000'.
        # The sibling ``expected_calibration_error`` already returns NaN on the
        # identical input; this function now follows it.
        ece = float("nan")
        warnings.warn(
            f"No prediction bin received a sample (n_samples={n}), so no "
            f"calibration error was measured. overall_ece is NaN, not 0.0: "
            f"0.0 would read as perfect calibration."
        )

    return CalibrationCurveResult(
        prob_true=np.array(prob_true),
        prob_pred=np.array(prob_pred),
        bin_counts=np.array(bin_counts),
        bin_edges=bin_edges,
        overall_ece=ece,
    )


def group_calibration_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
) -> Dict[str, CalibrationMetricResult]:
    """
    Compute comprehensive calibration metrics for all groups.

    Returns ECE, MCE, and Brier score for overall data and each
    demographic group, enabling detailed calibration fairness analysis.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_bins: Number of bins for calibration metrics
        strategy: Binning strategy
        min_group_size: Minimum samples per group

    Returns:
        Dictionary with 'ece', 'mce', and 'brier' CalibrationMetricResults

    Example:
        >>> metrics = group_calibration_metrics(y_true, y_prob, gender)
        >>> print(f"Overall ECE: {metrics['ece'].overall_value:.3f}")
        >>> print(f"ECE Disparity: {metrics['ece'].max_group_disparity:.3f}")

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: group_calibration_metrics. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    ece_result = expected_calibration_error(
        y_true, y_prob, protected_attr, n_bins, strategy, min_group_size
    )
    mce_result = maximum_calibration_error(
        y_true, y_prob, protected_attr, n_bins, strategy, min_group_size
    )
    brier_result = brier_score(y_true, y_prob, protected_attr, min_group_size)

    return {"ece": ece_result, "mce": mce_result, "brier": brier_result}


def calibration_disparity(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
) -> CalibrationDisparityResult:
    """
    Analyze calibration disparities across demographic groups.

    This function provides comprehensive analysis of calibration
    differences between groups, identifying which groups are most
    and least well-calibrated, and providing recommendations.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_bins: Number of bins
        strategy: Binning strategy
        min_group_size: Minimum samples per group

    Returns:
        CalibrationDisparityResult with comprehensive disparity analysis

    Example:
        >>> result = calibration_disparity(y_true, y_prob, race)
        >>> print(f"ECE Disparity: {result.ece_disparity:.3f}")
        >>> print(f"Most miscalibrated: {result.most_miscalibrated_group}")
        >>> for rec in result.recommendations:
        ...     print(f"  - {rec}")

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

    Ledger rows: calibration_difference, calibration_disparity. See
    docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """
    metrics = group_calibration_metrics(
        y_true, y_prob, protected_attr, n_bins, strategy, min_group_size
    )

    ece_result = metrics["ece"]
    mce_result = metrics["mce"]
    brier_result = metrics["brier"]

    group_ece = ece_result.group_values or {}
    group_mce = mce_result.group_values or {}
    group_brier = brier_result.group_values or {}

    # Disclose groups dropped for being below min_group_size. Without this
    # the report can read as a clean pass while the smallest (often most
    # vulnerable) group was never assessed.
    gm_all = GroupManager(
        coerce_to_array(protected_attr, "protected_attr"),
        min_group_size=min_group_size,
    )
    excluded_groups = [g for g in gm_all.groups if g not in group_ece]
    if excluded_groups:
        warnings.warn(
            f"Groups excluded from calibration disparity analysis (below "
            f"min_group_size={min_group_size}): {excluded_groups}. "
            f"Disparity figures do not cover these groups."
        )

    # THE GATE COUNTS GROUPS THAT CARRY A MEASUREMENT, not keys in a dict.
    #
    # G03 (2026-09-30). ``len(group_ece) < 2`` is a COUNT gate and cannot see a
    # key whose VALUE is not a number, so two groups of which one has no
    # computable calibration error walked straight past it into the arithmetic
    # below. There, builtin ``max``/``min`` do NOT propagate a NaN that is not
    # first in the iterable (every comparison against a NaN is False, so they
    # keep the running extreme), and the spread came back as a FINITE 0.0:
    # PERFECT PARITY ON EVERY SCALE, past every ``math.isfinite`` guard a
    # consumer could write, because the value those guards inspect is finite.
    #
    # MEASURED before this fix, on 240 rows in two 120-row groups with group B
    # carrying no ground truth at all, at the default gate of 30::
    #
    #     group_ece        {'A': 0.58, 'B': nan}
    #     group_mce        {'A': 0.9,  'B': 0.0}     # a fabricated 0.0, not a NaN
    #     excluded_groups  []
    #     ece_disparity    0.0
    #     has_significant_disparity  False
    #     recommendations  ['LOW: Calibration disparity is acceptable (0.000).', ...]
    #     warnings         []
    #
    # and with the two groups in the other order the same data gives NaN, so the
    # verdict depended on which group happened to appear first.
    #
    # This is the ROOT CAUSE of PP1-A5. ``recommend_calibration_strategy`` was
    # given a guard for it on 2026-09-27 whose own comment says "The root cause is
    # in calibration_disparity itself ... which this batch does not own"; every
    # OTHER consumer kept receiving the 0.0. The identical mechanism was fixed in
    # ``CalibrationMetricResult.max_group_disparity`` in this same file on
    # 2026-09-16, and this function, 900 lines further down, still did the raw
    # ``max(values) - min(values)``.
    #
    # The partition is the canonical ``_triage.partition_measured``, so a bool, a
    # blank string, ``pd.NA`` and a finite ``np.float32`` are all classified the
    # way the rest of the library classifies them.
    ece_measured, ece_unmeasured = _partition_measured(group_ece)
    unmeasured_names = [str(g) for g in ece_unmeasured]

    if len(ece_measured) < 2:
        # CAL-DISP (2026-09-09). This branch returned ece/mce/brier
        # disparity = 0.0, which is PERFECT PARITY on every scale, for a
        # comparison that never happened. Measured: one adequate group gave
        # 0.0/0.0/0.0 with has_significant_disparity False, while the same
        # data split into two groups gave 0.032/0.048/0.007. A downstream
        # aggregate reads the three numbers and never n_groups, so the
        # fabricated zeros passed as a measurement. A disparity needs two
        # groups; with fewer it is NaN, and the verdict property is None.
        if unmeasured_names:
            warnings.warn(
                f"Fewer than 2 groups carry a MEASURED calibration error, so no "
                f"between-group disparity was computed. Measured groups: "
                f"{sorted(str(g) for g in ece_measured)}; inside min_group_size="
                f"{min_group_size} but with no computable calibration error: "
                f"{sorted(unmeasured_names)}. Disparity figures are NaN, not 0.0: "
                f"builtin max/min skip a NaN and would have reported a finite 0.0, "
                f"which reads as perfect parity."
            )
        else:
            warnings.warn(
                f"Fewer than 2 groups with sufficient samples for disparity "
                f"analysis (measured groups: {sorted(group_ece)}). Disparity "
                f"figures are NaN, not 0.0: no between-group comparison was made."
            )
        return CalibrationDisparityResult(
            ece_disparity=float("nan"),
            mce_disparity=float("nan"),
            brier_disparity=float("nan"),
            group_ece=group_ece,
            group_mce=group_mce,
            group_brier=group_brier,
            most_miscalibrated_group="",
            least_miscalibrated_group="",
            n_groups=len(group_ece),
            recommendations=["Insufficient groups for disparity analysis"],
            excluded_groups=excluded_groups,
        )

    # At least two groups carry a measured ECE. The spread is taken over the
    # MEASURED values only, which is what the ``excluded_groups`` design already
    # made this field mean: a LOWER BOUND when some group is not covered, and the
    # disparity itself when all of them are.
    #
    # G03 (2026-09-30), see the gate above for the measurement. ``max``/``min``
    # over the raw dict values silently skipped a NaN and returned a finite 0.0.
    #
    # ECE IS THE GROUP-LEVEL EVIDENCE TEST, for all three metrics, which is the
    # rule ``plot_calibration_disparity`` adopted for the same input on
    # 2026-09-30: when a group's ECE is not a real finite number nothing about
    # that group's calibration was measured, so its MCE and Brier values do not
    # enter a spread either, WHATEVER THEY HAPPEN TO HOLD. That matters because
    # ``group_mce`` for such a group arrives as a fabricated 0.0 rather than a
    # NaN, so an MCE spread taken against it is a real number computed against
    # an invented one (measured: 0.9 against B's 0.0). Each metric is then
    # partitioned on its own values as well, because a group can have a measured
    # ECE and an unmeasurable Brier score.
    def _spread(values: Dict[str, float]) -> float:
        covered = {g: v for g, v in values.items() if g in ece_measured}
        measured, _ = _partition_measured(covered)
        if len(measured) < 2:
            return float("nan")
        return max(measured.values()) - min(measured.values())

    ece_disparity = _spread(group_ece)
    mce_disparity = _spread(group_mce)
    brier_disparity = _spread(group_brier)

    if unmeasured_names:
        warnings.warn(
            f"{len(unmeasured_names)} group(s) cleared min_group_size="
            f"{min_group_size} and have NO measured calibration error "
            f"({sorted(unmeasured_names)}), so they are NOT part of the disparity "
            f"figures: the spread is a LOWER BOUND over the "
            f"{len(ece_measured)} group(s) that carry a measurement, and the "
            f"group left out can decide the comparison either way."
        )

    # Named over the MEASURED subset. ``max(group_ece, key=...)`` compares raw
    # values, and every comparison against a NaN is False, so the extremes fell
    # to whichever group arrived first: on {'A': 0.58, 'B': nan} both
    # most_miscalibrated and least_miscalibrated came back 'A', i.e. the SAME
    # group named as the best and the worst calibrated, and with the groups in
    # the other order both came back 'B', the group nobody measured.
    most_miscalibrated = max(ece_measured, key=lambda g: ece_measured[g])
    least_miscalibrated = min(ece_measured, key=lambda g: ece_measured[g])

    recommendations = []

    # BGL5 (2026-09-27). The exclusion notice used to be APPENDED, at the end of
    # this list, and every consumer of the list truncates it:
    # ``explainer._explain_calibration`` keeps ``report.recommendations[:5]`` and
    # ``CalibrationReport.summary()`` prints ``[:5]``. MEASURED on 100 'a' + 100
    # 'b' + 25 'c' with 'b' badly miscalibrated, the notice landed at index 5 of
    # 9 recommendations, so the ExplanationReport a reader meets carried NO trace
    # that 'c' exists, beside a summary reading "Calibration analysis for 3
    # groups". Written FIRST now, which is the ordering
    # ``analyzer._identify_critical_issues`` adopted for exactly this reason: on
    # the same input it is index 0 of 9 and survives every [:4] and [:5] above
    # it. Only the ORDER changes; no recommendation is added or removed.
    if excluded_groups:
        recommendations.append(
            f"NOT ASSESSED: groups {excluded_groups} were excluded "
            f"(fewer than {min_group_size} samples). Collect more data "
            f"before treating this analysis as covering them."
        )

    # G03 (2026-09-30). The notice above fires on the row-COUNT gate only. A
    # group inside the gate with no computable calibration error is equally
    # uncovered and had no notice at all, so it is given its own, in the same
    # leading position and for the same truncation reason. Separate from the line
    # above rather than merged into it, because the REMEDY differs: one needs more
    # rows, the other needs ground truth or scores for rows it already has.
    if unmeasured_names:
        recommendations.append(
            f"NOT ASSESSED: groups {sorted(unmeasured_names)} cleared "
            f"min_group_size={min_group_size} but have NO measured calibration "
            f"error, so the disparity figures do not cover them. Check whether "
            f"those rows carry ground truth and scores."
        )

    if ece_disparity > 0.1:
        recommendations.append(
            f"CRITICAL: High calibration disparity ({ece_disparity:.3f}). "
            f"Consider group-specific calibration."
        )
    elif ece_disparity > 0.05:
        recommendations.append(
            f"MODERATE: Notable calibration disparity ({ece_disparity:.3f}). "
            f"Evaluate impact on decisions."
        )
    else:
        # The only ALL-CLEAR line in this list, so it is the one line that may
        # not be written over a comparison that left a group out. A breach above
        # is a finding and stands on the groups that were compared; "acceptable"
        # is a statement about the whole. MEASURED on 150 'a' + 150 'b' + 25 'c'
        # (gate 30, 'c' ECE 0.97): "LOW: Calibration disparity is acceptable
        # (0.030)." with no mention of 'c' anywhere in the line. It now reads
        # "... (0.030) across the 2 group(s) compared; groups ['c'] were
        # excluded and are not covered by it." A fully measured comparison keeps
        # the original sentence unchanged.
        #
        # G03 (2026-09-30): ``excluded_groups`` is the row-COUNT gate, so a group
        # inside the gate with no computable calibration error was not named
        # here either. It is added to the same clause rather than a second one,
        # so the sentence reads the same way whichever gate the group fell
        # through, and a fully measured comparison still gets the original
        # sentence verbatim (the exact-string control in
        # test_bgl5_post_processing_2_and_rendering asserts that literal).
        uncovered = list(excluded_groups) + [
            f"{g} (not measured)" for g in sorted(unmeasured_names)
        ]
        if uncovered:
            recommendations.append(
                f"LOW: Calibration disparity is acceptable ({ece_disparity:.3f}) across "
                f"the {len(ece_measured)} group(s) compared; groups {uncovered} were "
                f"excluded and are not covered by it."
            )
        else:
            recommendations.append(
                f"LOW: Calibration disparity is acceptable ({ece_disparity:.3f})."
            )

    if group_ece[most_miscalibrated] > 0.1:
        recommendations.append(
            f"Group '{most_miscalibrated}' has poor calibration (ECE={group_ece[most_miscalibrated]:.3f}). "
            f"Consider isotonic regression calibration."
        )

    if mce_disparity > 0.15:
        recommendations.append(
            f"Large worst-case disparity (MCE diff={mce_disparity:.3f}). "
            f"Some probability ranges may be particularly problematic."
        )

    return CalibrationDisparityResult(
        ece_disparity=ece_disparity,
        mce_disparity=mce_disparity,
        brier_disparity=brier_disparity,
        group_ece=group_ece,
        group_mce=group_mce,
        group_brier=group_brier,
        most_miscalibrated_group=most_miscalibrated,
        least_miscalibrated_group=least_miscalibrated,
        n_groups=len(group_ece),
        recommendations=recommendations,
        excluded_groups=excluded_groups,
    )


# ===========================================================================
# Advanced calibration diagnostics (per-group Brier decomposition, ECE
# confidence intervals, sufficiency test, cross-validated stability).
#
# Note: the consumer handler defines its own _adaptive_n_bootstrap and
# _COMPUTE_BUDGET_SECONDS. The library did not expose those symbols, so they
# are defined here (module local) to keep the resample work bounded, matching
# the spirit of the handler budget rather than reimporting from it.
# ===========================================================================

# Wall clock ceiling for the resampling loops below. A single diagnostics call
# should stay interactive, so bootstrap / permutation loops stop once this is
# exceeded and report on whatever resamples completed.
_COMPUTE_BUDGET_SECONDS = 90


def _adaptive_n_bootstrap(n_samples: int) -> int:
    """Cap resample iterations inversely with sample size.

    Larger samples yield a stable resampled estimate with fewer iterations and
    each iteration is more expensive, so the ceiling shrinks as n grows. The
    caller takes min(requested, this) so the requested count is never exceeded.
    """
    if n_samples > 50000:
        return 200
    if n_samples > 10000:
        return 500
    if n_samples > 2000:
        return 1000
    return 2000


def per_group_brier_decomposition(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
) -> Dict[str, dict]:
    """
    Brier score decomposition (reliability, resolution, uncertainty) per group.

    Runs the existing ``brier_score_decomposition`` on each demographic group's
    masked arrays and on the full dataset. Groups below ``min_group_size`` are
    omitted (not an error).

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        protected_attr: Protected attribute defining groups
        n_bins: Number of bins for the decomposition
        strategy: Binning strategy ('uniform' or 'quantile')
        min_group_size: Minimum samples required to analyse a group

    Returns:
        Dict mapping each group name (and '__overall__') to a dict with
        brier_score, reliability, resolution, uncertainty, skill_score,
        n_samples, n_bins and not_assessed. The identity
        brier_score = reliability - resolution + uncertainty holds when the
        forecasts are constant within each bin.

        A group below ``min_group_size`` is STILL a key of this dict, carrying
        None for every figure and a ``not_assessed`` reason. It used to be
        omitted in silence: measured on A=100, B=100 and a 2-row C, the result
        held only 'A', 'B' and '__overall__' with ZERO warnings, so the only
        trace of C was its absence from a dict nobody compares against the
        input, while C's two rows WERE inside the '__overall__' figures
        (n_samples 202). An assessed group carries ``not_assessed=None``, so
        the schema is the same either way (BGL3, 2026-09-27).
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob, protected_attr)
    validate_probabilities(y_prob, "y_prob")

    def _pack(decomp: BrierDecomposition) -> dict:
        d = decomp.to_dict()
        return {
            "brier_score": float(d["brier_score"]),
            "reliability": float(d["reliability"]),
            "resolution": float(d["resolution"]),
            "uncertainty": float(d["uncertainty"]),
            "skill_score": float(d["skill_score"]),
            "n_samples": int(d["n_samples"]),
            "n_bins": int(d["n_bins"]),
            # Fixed schema, so "was this group assessed?" is a field a caller
            # reads rather than a group's presence it has to notice.
            "not_assessed": None,
        }

    results: Dict[str, dict] = {}

    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    for group_name in gm.get_valid_groups():
        mask = gm.get_mask(group_name)
        if int(np.sum(mask)) < min_group_size:
            continue
        decomp = brier_score_decomposition(
            y_true[mask], y_prob[mask], n_bins=n_bins, strategy=strategy
        )
        results[group_name] = _pack(decomp)

    # An undersized group is named, not dropped. See the Returns note for what
    # this silence measured before the fix.
    sizes = gm.get_group_sizes()
    for group_name in _disclose_excluded_groups(
        gm, min_group_size, "per-group Brier decomposition"
    ):
        results[group_name] = {
            "brier_score": None,
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
            "skill_score": None,
            "n_samples": int(sizes.get(group_name, 0)),
            "n_bins": int(n_bins),
            "not_assessed": (
                f"group {group_name!r} holds {int(sizes.get(group_name, 0))} row(s), "
                f"fewer than min_group_size={min_group_size}, so its Brier "
                f"decomposition was not computed"
            ),
        }

    overall = brier_score_decomposition(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    results["__overall__"] = _pack(overall)

    return results


def ece_confidence_intervals(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    min_group_size: int = 30,
    random_state: int = 42,
) -> Dict[str, dict]:
    """
    Bootstrap confidence intervals for per-group Expected Calibration Error.

    For each group, indices are resampled with replacement and ECE is recomputed
    (reusing ``expected_calibration_error``). The requested ``n_bootstrap`` is
    capped by ``_adaptive_n_bootstrap`` for the group size, and the loop stops
    if it exceeds the compute budget.

    CAVEAT (Kumar, Liang & Ma 2019, Verified Uncertainty Calibration): the
    binned plug-in ECE is upward-biased by O(sqrt(1/n_bin)) near perfect
    calibration, so this percentile interval is a CI for the BIASED plug-in
    estimand and MUST NOT be read as a test of 'true ECE = 0': for a
    perfectly calibrated group the lower bound is essentially always above 0.
    Each result therefore carries ``noise_floor`` (the expected plug-in ECE
    of an exactly calibrated model on this data: sum_b w_b *
    sqrt(2*var_b/pi), with var_b the binomial variance of the bin-b gap) and
    ``near_null=True`` when the point estimate is within two null standard
    deviations above that floor, meaning the observed ECE is
    indistinguishable from pure estimation noise.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_bins: Number of bins for ECE
        strategy: Binning strategy
        n_bootstrap: Requested bootstrap resamples (capped adaptively). Must be
            a positive integer. It was never validated, and with
            ``n_bootstrap=0`` the resample loop never ran: measured on two
            groups of 100, every group came back ``ci_lower == ci_upper ==
            ece``, ``ci_width 0.0``, ``se 0.0`` and ``unstable False``, with no
            warning. A zero-width interval is the most precise answer this
            function can give, and nothing had been resampled at all.
        ci: Confidence level (e.g. 0.95 for a 95% interval)
        min_group_size: Minimum samples required to analyse a group
        random_state: Seed for the bootstrap resampling

    Returns:
        Dict mapping each group name to a dict with ece, ci_lower, ci_upper,
        ci_width, se, n, n_bootstrap_effective, unstable (ci_width > 0.05),
        noise_floor, near_null (ece <= noise_floor) and not_assessed.

        Three states, never two (BGL3, 2026-09-27). With one resample,
        ``np.percentile`` of a single value returns that value at EVERY
        percentile, so the 95% interval was reported as the zero-width point
        ``[0.2075, 0.2075]`` around a point estimate of 0.1574, which it does
        not even contain, and ``unstable`` read False.

        The floor is not two either (BGL5, 2026-09-27). A percentile bound is an
        order statistic of the draws, so a ``ci`` of 0.95 needs
        ``2/alpha - 1 = 39`` of them; at n_bootstrap=2 the SAME group came back
        ``[0.10708, 0.13726]`` around an ECE of 0.15737 the interval does not
        contain, with ``unstable`` False and no warning. Too few usable
        resamples (a small ``n_bootstrap``, the wall-clock budget cutting the
        loop, or every resampled ECE coming back NaN) now gives NaN bounds, NaN
        se and ``unstable=None``, with the reason and the required count in
        ``not_assessed``.

        A group below ``min_group_size`` is a key of this dict too, with None
        figures and a ``not_assessed`` reason: measured on A=100, B=100 and a
        2-row C, the result held only 'A' and 'B' and said nothing about C.
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob, protected_attr)
    validate_probabilities(y_prob, "y_prob")

    # A resample count of zero is not a cheap run, it is an interval nobody
    # estimated. Same refusal, same reason, as sufficiency_test's
    # n_permutations above: measured, n_bootstrap=0 returned ci_width 0.0, se
    # 0.0 and unstable False for every group, silently (BGL3, 2026-09-27).
    try:
        n_bootstrap_int = int(n_bootstrap)
    except (TypeError, ValueError):
        raise ValueError(
            f"ece_confidence_intervals: n_bootstrap must be a positive integer, got "
            f"{n_bootstrap!r}. With no resamples the interval collapses onto the point "
            f"estimate, which reads as a perfectly precise ECE rather than as an "
            f"interval that was never estimated."
        ) from None
    if n_bootstrap_int < 1:
        raise ValueError(
            f"ece_confidence_intervals: n_bootstrap must be a positive integer, got "
            f"{n_bootstrap_int}. With no resamples the interval collapses onto the point "
            f"estimate (ci_width 0.0, se 0.0, unstable False), which reads as a "
            f"perfectly precise ECE rather than as an interval that was never estimated."
        )

    rng = np.random.default_rng(random_state)

    alpha = 1.0 - ci
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    # BGL5 (2026-09-27). The interval floor below was ``len(boot) < 2``, and the
    # defect the n_bootstrap=1 fix records is ALIVE AT 2. A percentile bootstrap
    # bound is an ORDER STATISTIC of the resample draws, so the tail probability
    # alpha/2 needs at least one draw beyond it: alpha/2 >= 1/(B+1), i.e.
    # B >= 2/alpha - 1, which is 39 draws for a 95% interval (Hall 1986;
    # Davison & Hinkley 1997 recommend (B+1)*alpha integral). Below that
    # ``np.percentile`` interpolates INTO the tails between the two extreme
    # draws and returns a bound no resample supports.
    #
    # MEASURED on the file's own _two_groups() fixture, group A, random_state=0:
    #   n_bootstrap=2  -> ci [0.10708156858467521, 0.13726336220604668],
    #                     width 0.03018, se 0.02247, unstable False,
    #                     not_assessed None, 0 warnings, around an ECE of
    #                     0.1573694620635577 that the interval does NOT contain
    #   n_bootstrap=50 -> ci [0.11589460845323353, 0.2734240523892867],
    #                     width 0.15753, unstable True
    # which is verbatim the n_bootstrap=1 defect ("an interval that does not even
    # contain the point estimate"), one value to the right of the pinned refusal.
    # At n_bootstrap=2 the bounds, the width, the se and `unstable` are now NaN /
    # None with the reason in ``not_assessed`` and a UserWarning; n_bootstrap=50
    # and above is untouched (same bounds, same se, unstable True).
    if alpha <= 0.0:
        # ci >= 1.0 asks for the 0th and 100th percentiles, which are the min and
        # the max of the draws: an order statistic at any count, so the floor is
        # the two draws an interval needs to have a width at all.
        min_resamples = 2
    else:
        min_resamples = max(2, int(math.ceil(2.0 / alpha)) - 1)

    def _ece_single(y_t: np.ndarray, y_p: np.ndarray) -> float:
        res = expected_calibration_error(
            y_t, y_p, protected_attr=None, n_bins=n_bins, strategy=strategy
        )
        return float(res.overall_value)

    def _null_noise_floor(y_p: np.ndarray) -> Tuple[float, float]:
        """(mean, sd) of the plug-in binned ECE of an EXACTLY calibrated model
        on these scores: gap_b ~ approx N(0, var_b) under the null with var_b =
        sum_i p_i(1-p_i) / n_b^2, so E|gap_b| = sqrt(2*var_b/pi) and
        Var|gap_b| = var_b * (1 - 2/pi). This is the Kumar et al. 2019
        near-null bias the percentile CI cannot see; a point estimate within
        this noise is not evidence of miscalibration."""
        n_local = len(y_p)
        if n_local == 0:
            # A noise floor of 0.0 means NOTHING is within noise, so every point
            # estimate reads as real miscalibration. This is the
            # fabricated-BREACH direction of the same defect.
            warnings.warn(
                "ece_confidence_intervals: no observations, so the near-null noise "
                "floor could not be estimated. Returning nan rather than 0.0, which "
                "would make every deviation look significant.",
                UserWarning,
                stacklevel=3,
            )
            return float("nan"), float("nan")
        if strategy == "uniform":
            edges = np.linspace(0, 1, n_bins + 1)
        else:  # quantile (mirrors expected_calibration_error's edge handling)
            edges = np.percentile(y_p, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(np.clip(edges, 0, 1))
            if len(edges) < 2:
                edges = np.array([0.0, 1.0])
            edges[0] = 0
            edges[-1] = 1
        idx = np.digitize(y_p, edges[1:-1])
        floor = 0.0
        var_sum = 0.0
        for b in range(len(edges) - 1):
            in_bin = idx == b
            n_b = int(np.sum(in_bin))
            if n_b == 0:
                continue
            p_b = y_p[in_bin]
            w_b = n_b / n_local
            var_b = float(np.sum(p_b * (1.0 - p_b))) / (n_b**2)
            floor += w_b * float(np.sqrt(2.0 * var_b / np.pi))
            var_sum += (w_b**2) * var_b * (1.0 - 2.0 / np.pi)
        return float(floor), float(np.sqrt(var_sum))

    results: Dict[str, dict] = {}

    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    for group_name in gm.get_valid_groups():
        mask = gm.get_mask(group_name)
        n_g = int(np.sum(mask))
        if n_g < min_group_size:
            continue

        y_t = y_true[mask]
        y_p = y_prob[mask]
        point = _ece_single(y_t, y_p)

        n_boot = min(n_bootstrap_int, _adaptive_n_bootstrap(n_g))
        boot = np.full(n_boot, np.nan, dtype=float)
        idxs = np.arange(n_g)
        t0 = time.monotonic()
        completed = 0
        for b in range(n_boot):
            samp = rng.choice(idxs, size=n_g, replace=True)
            boot[b] = _ece_single(y_t[samp], y_p[samp])
            completed = b + 1
            if time.monotonic() - t0 > _COMPUTE_BUDGET_SECONDS:
                break

        boot = boot[:completed]
        boot = boot[~np.isnan(boot)]

        # An interval over too few usable resamples does not exist.
        # np.percentile of a single value returns it at every percentile, so
        # both bounds landed on that one draw and ci_width came out 0.0, the
        # most precise reading this function has; with none at all they landed
        # on the point estimate itself and se was hard-coded 0.0. Neither is a
        # measurement, and unstable=False on either is a stability claim from
        # no evidence (BGL3, 2026-09-27). The floor is ``min_resamples``, not 2,
        # for the order-statistic reason measured where it is computed above.
        not_assessed: Optional[str] = None
        unstable: Optional[bool] = None

        # THE POINT ESTIMATE IS THE SHARED PRECONDITION of everything below it, so
        # its own could-not-check is decided ABOVE the interval branch.
        #
        # BGL5 wave 4, the sibling door beside near_null. The tail of the message
        # below read "the point estimate stands on its own", unconditionally. On
        # the measured B row above, whose ECE is NaN because the group carries no
        # ground truth, that sentence is FALSE in the one field a reader of this
        # row reads: it told them a point estimate survived and only the interval
        # around it was lost, when in fact nothing on the row was measured. A
        # reader acting on that disclosure would quote the ECE.
        point_measured = bool(np.isfinite(point))
        point_clause = (
            "the point estimate stands on its own and"
            if point_measured
            else (
                "NO ECE could be computed for this group either, so its point estimate "
                "is NaN and nothing on this row measures that group's calibration:"
            )
        )

        if len(boot) < min_resamples:
            ci_lower = ci_upper = ci_width = se = float("nan")
            not_assessed = (
                f"{len(boot)} of the {n_boot} requested resample(s) produced a usable "
                f"ECE, fewer than the {min_resamples} a {ci:.0%} percentile interval "
                f"needs for both of its bounds to be order statistics of the draws "
                f"rather than an interpolation into the tails, so no percentile "
                f"interval could be formed for group {group_name!r}; {point_clause} "
                f"the bounds, the standard error and the "
                f"stability verdict are not measured"
            )
            warnings.warn(
                f"ece_confidence_intervals: {not_assessed}. The bounds are NaN and "
                f"'unstable' is None (could-not-check), not 0.0 and False, which would "
                f"report a zero-width interval nobody estimated.",
                UserWarning,
                stacklevel=2,
            )
        else:
            ci_lower = float(np.percentile(boot, lo_pct))
            ci_upper = float(np.percentile(boot, hi_pct))
            se = float(np.std(boot, ddof=1))
            ci_width = float(ci_upper - ci_lower)
            unstable = bool(ci_width > 0.05)

        # The interval branch is reached by COUNTING usable resamples, so enough of
        # them can form an interval around a point estimate that is itself NaN. On
        # that path the row would carry a finite interval, a measured stability
        # verdict and not_assessed=None beside an ECE of NaN, which is the same
        # could-not-check-as-measurement one door along. The point estimate is the
        # precondition, so it is stated wherever it fails.
        if not point_measured and not_assessed is None:
            unstable = None
            not_assessed = (
                f"no ECE could be computed for group {group_name!r}: its point estimate "
                f"is NaN, so the {len(boot)} usable resample(s) form an interval around "
                f"nothing and the stability verdict is not measured"
            )
            warnings.warn(
                f"ece_confidence_intervals: {not_assessed}. 'unstable' and 'near_null' "
                f"are None (could-not-check), not False, which would report this group "
                f"as stable and as showing real miscalibration.",
                UserWarning,
                stacklevel=2,
            )

        noise_floor, noise_sd = _null_noise_floor(y_p)

        # near_null IS A THREE-STATE, like every other verdict on this row.
        #
        # BGL5 wave 4 (2026-09-30). This was
        # ``bool(point <= noise_floor + 2.0 * noise_sd)``, and ``nan <= anything``
        # is False, so a group whose point ECE could NOT be computed was published
        # as near_null FALSE. The docstring defines near_null=True as "the observed
        # ECE is indistinguishable from pure estimation noise", so False asserts
        # the opposite: that the observed ECE IS distinguishable from noise, i.e.
        # REAL MISCALIBRATION. There is no observed ECE. That is the
        # fabricated-BREACH direction of exactly the defect _null_noise_floor names
        # twelve lines up ("A noise floor of 0.0 means NOTHING is within noise, so
        # every point estimate reads as real miscalibration").
        #
        # MEASURED before this fix, 200 rows in two 100-row groups, A labelled and
        # B carrying no ground truth, n_bootstrap=60:
        #   A: ece 0.11616969696969695, unstable True,  near_null True
        #   B: ece NAN, ci [nan, nan], se nan, unstable NONE,
        #      n_bootstrap_effective 0, not_assessed '0 of the 60 requested
        #      resample(s) produced a usable ECE ...', AND near_null FALSE
        # Every other field on B's row already said could-not-check; this one
        # contradicted them. The function's OTHER return path, the size-excluded
        # block below, already writes ``"near_null": None`` for a group it could
        # not assess, so the two exits of one function disagreed about one field.
        #
        # The noise floor is part of the comparison, so an unmeasurable FLOOR is
        # equally disqualifying: _null_noise_floor returns (nan, nan) rather than
        # (0.0, 0.0) precisely so that this branch cannot read as a breach.
        if not (np.isfinite(point) and np.isfinite(noise_floor) and np.isfinite(noise_sd)):
            near_null: Optional[bool] = None
        else:
            near_null = bool(point <= noise_floor + 2.0 * noise_sd)

        results[group_name] = {
            "ece": float(point),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            "se": se,
            "n": n_g,
            # The count that actually bounds the interval, not the requested
            # one: the wall-clock budget can cut the loop short.
            "n_bootstrap_effective": int(len(boot)),
            "unstable": unstable,
            "noise_floor": noise_floor,
            "near_null": near_null,
            "not_assessed": not_assessed,
        }

    # An undersized group is named here rather than left out of the dict, the
    # same disclosure expected_calibration_error makes through this helper.
    sizes = gm.get_group_sizes()
    for group_name in _disclose_excluded_groups(gm, min_group_size, "per-group ECE interval"):
        results[group_name] = {
            "ece": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_width": None,
            "se": None,
            "n": int(sizes.get(group_name, 0)),
            "n_bootstrap_effective": 0,
            "unstable": None,
            "noise_floor": None,
            "near_null": None,
            "not_assessed": (
                f"group {group_name!r} holds {int(sizes.get(group_name, 0))} row(s), "
                f"fewer than min_group_size={min_group_size}, so neither its ECE nor an "
                f"interval around it was computed"
            ),
        }

    return results


def _chi2_statistic(table: np.ndarray) -> Tuple[float, int]:
    """Return the (uncorrected) chi-square statistic and dof for a table."""
    from scipy import stats as scipy_stats

    stat, _p, dof, _expected = scipy_stats.chi2_contingency(table, correction=False)
    return float(stat), int(dof)


def sufficiency_test(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    method: Literal["chi2", "permutation"] = "chi2",
    n_permutations: int = 1000,
    min_group_size: int = 30,
    random_state: int = 42,
) -> dict:
    """
    Test sufficiency: P(Y=1 | probability-bin, group) equal across groups.

    Within each uniform probability bin a groups x {positive, negative}
    contingency table is built. ``method='chi2'`` uses
    ``scipy.stats.chi2_contingency``; ``method='permutation'`` shuffles the
    group labels within the bin and compares the chi-square statistic. Bins
    where any group has zero samples (or a degenerate outcome column) are
    skipped because the statistic is undefined there.

    The overall verdict is multiplicity-corrected: the per-bin p-values are
    Benjamini-Hochberg adjusted (Benjamini & Hochberg 1995) and ``passes`` /
    ``n_bins_violated`` are read from the ADJUSTED p-values at alpha 0.05.
    Comparing the raw minimum over up to ``n_bins`` tests against the
    single-test alpha inflated the family-wise false-FAIL rate to ~30% on
    perfectly sufficient data.

    Three states, never two: when ZERO bins could be tested (non-overlapping
    group score distributions, degenerate outcomes) the verdict is
    ``passes=None`` with ``status='not_assessable'``. It is never ``True``:
    zero evidence is could-not-check, not a pass.

    The SAME rule applies to a test that ran but could never have fired.
    ``method='permutation'`` estimates p as ``(count + 1) / (B + 1)``, so its
    smallest attainable value is ``1 / (B + 1)``, and ``B`` is capped by
    :func:`_adaptive_n_bootstrap` (200 above 50,000 samples) and can be cut
    further by the wall-clock budget. Benjamini-Hochberg then puts the rank-1
    bar at ``alpha / n_bins_tested``. When the floor sits above that bar, NO
    arrangement of the data can produce a finding, and a "not significant"
    reading is an absence of power rather than evidence of sufficiency.
    Measured on this repo 2026-09-10, 60,000 samples, one score band in which
    group B's realised positive rate is 94.3 percent against group A's 55.4
    percent at the SAME predicted score:

    ==========  =========  ===========  =============================
    ``n_bins``  raw p      adjusted p   verdict
    ==========  =========  ===========  =============================
    10          0.0049751  0.049751     ``passes=False``
    11          0.0049751  0.054726     ``passes=True``  (!)
    12          0.0049751  0.059701     ``passes=True``  (!)
    ==========  =========  ===========  =============================

    The raw p is bit-for-bit ``1/201``, the design floor, in every row: the test
    saturated and said nothing about it, and at ``n_bins >= 11`` this detector
    could not fire at all. Those two rows now report ``passes=None`` with
    ``status='not_assessable'`` and a ``detectability_note`` saying so.
    ``detectable`` is ``True`` only when EVERY tested bin could have fired, and
    a bin that DID breach still reports ``passes=False``: a measured finding is
    a finding whatever the rest of the design could not see.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_bins: Number of uniform probability bins
        method: 'chi2' or 'permutation'
        n_permutations: Permutation count (capped adaptively) for permutation.
            Must be a positive integer. It was never validated, and with
            ``n_permutations=0`` the resample loop never ran, every bin's
            ``(count + 1) / (0 + 1)`` came out as exactly 1.0, and the function
            reported ``passes=True, status='assessed', n_bins_tested=10`` with
            no warning on data that ``method='chi2'`` fails at an adjusted p of
            6.3e-276.
        min_group_size: Minimum group size to include a group
        random_state: Seed for the permutation shuffles

    Returns:
        Dict with 'bins' (per bin statistic, dof, raw p_value, BH-adjusted
        p_value_adjusted, per-group positive rates and counts, bin range) and
        'overall' (min_bin_p_value, min_bin_p_value_adjusted, n_bins_violated,
        n_bins_tested, n_bins_skipped, passes, status, correction, method,
        min_attainable_p_value, detectable, detectability_note). Each tested bin
        also carries its own n_permutations_effective, min_attainable_p_value
        and detectable.
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob, protected_attr)
    validate_probabilities(y_prob, "y_prob")

    # A resample count of zero is not a cheap run, it is a test that cannot
    # produce evidence: the loop never executes and every bin's
    # ``(count + 1) / (B + 1)`` is exactly 1.0. Measured on this repo, that
    # returned passes=True / status='assessed' / n_bins_tested=10 with zero
    # warnings, on data chi2 rejects at an adjusted p of 6.3e-276. The parameter
    # was never validated at all, so a typo'd 0 or a negative was accepted in
    # silence.
    try:
        n_permutations_int = int(n_permutations)
    except (TypeError, ValueError):
        raise ValueError(
            f"sufficiency_test: n_permutations must be a positive integer, got "
            f"{n_permutations!r}. A permutation test with no resamples returns "
            f"p = 1.0 for every bin, which is not a pass, it is no evidence."
        ) from None
    if n_permutations_int < 1:
        raise ValueError(
            f"sufficiency_test: n_permutations must be a positive integer, got "
            f"{n_permutations_int}. A permutation test with no resamples returns "
            f"p = 1.0 for every bin, which is not a pass, it is no evidence."
        )

    rng = np.random.default_rng(random_state)

    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    valid_groups = gm.get_valid_groups()
    # Sufficiency is a statement about the groups that were IN the contingency
    # tables. An undersized group left no trace at all: measured on A=100,
    # B=100 and a 2-row C, the verdict came back passes=True, status='assessed',
    # over A and B only, with no warning and nothing in the result naming C
    # (BGL3, 2026-09-27). A pass that does not cover a group must say so.
    excluded_groups = _disclose_excluded_groups(gm, min_group_size, "sufficiency")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    n_perm = min(n_permutations_int, _adaptive_n_bootstrap(len(y_true)))

    bins_out: List[dict] = []
    p_values: List[float] = []
    # The smallest p each tested bin's DESIGN could have produced. Read together
    # with the Benjamini-Hochberg rank-1 bar below, this is what separates
    # "sufficiency holds" from "this test never had the power to say".
    p_floors: List[Optional[float]] = []
    t0 = time.monotonic()

    for i in range(n_bins):
        bin_mask = bin_indices == i

        table_rows = []
        group_pos_rates: Dict[str, Optional[float]] = {}
        n_by_group: Dict[str, int] = {}
        bin_groups = []
        bin_labels = []
        any_empty = False

        for g in valid_groups:
            g_mask = gm.get_mask(g) & bin_mask
            n_g = int(np.sum(g_mask))
            n_by_group[g] = n_g
            if n_g == 0:
                any_empty = True
                group_pos_rates[g] = None
                table_rows.append([0, 0])
                continue
            labels_g = y_true[g_mask]
            pos = int(np.sum(labels_g == 1))
            neg = n_g - pos
            table_rows.append([pos, neg])
            group_pos_rates[g] = float(pos) / float(n_g)
            bin_groups.append(np.full(n_g, g, dtype=object))
            bin_labels.append(labels_g)

        # Skip bins where any group is empty (chi-square undefined) or where
        # fewer than two groups are present.
        if any_empty or len(valid_groups) < 2:
            continue

        table = np.array(table_rows, dtype=float)
        # Degenerate outcome column (all positive or all negative in the bin)
        # gives a zero expected frequency, which chi-square cannot handle.
        if np.any(table.sum(axis=0) == 0):
            continue

        obs_stat, dof = _chi2_statistic(table)

        if method == "permutation":
            groups_arr = np.concatenate(bin_groups)
            labels_arr = np.concatenate(bin_labels)
            ge = 1  # count observed itself for an unbiased p-value
            done = 0
            for _ in range(n_perm):
                shuffled = rng.permutation(groups_arr)
                perm_rows = []
                for g in valid_groups:
                    m = shuffled == g
                    pos = int(np.sum(labels_arr[m] == 1))
                    neg = int(np.sum(m)) - pos
                    perm_rows.append([pos, neg])
                perm_table = np.array(perm_rows, dtype=float)
                perm_stat, _dof = _chi2_statistic(perm_table)
                if perm_stat >= obs_stat:
                    ge += 1
                done += 1
                if time.monotonic() - t0 > _COMPUTE_BUDGET_SECONDS:
                    break
            p_value = float(ge) / float(done + 1)
            statistic = obs_stat
            # ``done``, not ``n_perm``: the wall-clock budget can cut the loop
            # short, and the estimator's denominator is what actually bounds the
            # p-value from below.
            n_perm_effective: Optional[int] = int(done)
            p_floor: Optional[float] = min_attainable_p_permutation(done)
        else:
            from scipy import stats as scipy_stats

            statistic, p_value, dof, _expected = scipy_stats.chi2_contingency(table)
            statistic = float(statistic)
            p_value = float(p_value)
            dof = int(dof)
            # The asymptotic chi-square p is continuous and has no discrete
            # floor, so this design can always reach any bar. 0.0 records that,
            # rather than None, which is the could-not-compute state.
            n_perm_effective = None
            p_floor = 0.0

        bins_out.append(
            {
                "bin_index": int(i),
                "bin_range": [float(bin_edges[i]), float(bin_edges[i + 1])],
                "group_positive_rates": group_pos_rates,
                "n_by_group": n_by_group,
                "statistic": float(statistic),
                "dof": int(dof),
                "p_value": float(p_value),
                "n_permutations_effective": n_perm_effective,
                "min_attainable_p_value": p_floor,
            }
        )
        p_values.append(float(p_value))
        p_floors.append(p_floor)

    n_tested = len(p_values)

    # DETECTABILITY, resolved once the family size is known.
    #
    # ``detectability`` is the shared helper in
    # ``evaluation.vfairness_metrics._statistics``; do not write a local copy of
    # this rule, and do not compute the floor from ``n_perm``: nine detectors in
    # this library run a discrete test and the reason this one shipped a
    # false-PASS is that the check lived in exactly one of them.
    #
    # The bar every bin must clear on its own evidence is the Benjamini-Hochberg
    # rank-1 threshold, ``alpha / n_bins_tested``, which is why the family size
    # has to be the TESTED count and not the requested ``n_bins``: skipped bins
    # are not in the correction.
    #
    # A bin is undetectable when even a perfect separation of the groups could
    # not push its adjusted p under alpha. Measured on this repo, 60,000 samples
    # and one score band at 94.3 percent versus 55.4 percent: at n_bins=12 the
    # floor 1/201 = 0.0049751 sits above the 0.0041667 bar, the raw p came back
    # bit-for-bit AT the floor, and the function answered passes=True,
    # status='assessed', with no warning.
    worst_floor: Optional[float] = None
    if p_floors:
        computed = [f for f in p_floors if f is not None]
        # None is could-not-compute for that bin, and it dominates: we cannot
        # claim the family was detectable while one member's power is unknown.
        worst_floor = max(computed) if len(computed) == len(p_floors) else None
    detectable, detectability_note = (
        detectability(worst_floor, n_tested, alpha=0.05) if n_tested else (None, "")
    )
    for entry in bins_out:
        entry["detectable"], entry["detectability_note"] = detectability(
            entry["min_attainable_p_value"], n_tested, alpha=0.05
        )

    if p_values:
        # The verdict is min-over-bins, so uncorrected per-bin tests inflate the
        # family-wise error to ~1-0.95^n_bins (~30% measured false-FAIL at the
        # defaults on perfectly sufficient data). Benjamini-Hochberg over the
        # tested bins holds the false-FAIL rate near alpha under the global null.
        correction = benjamini_hochberg_correction(np.array(p_values), alpha=0.05)
        adjusted = correction.adjusted_p_values
        for entry, adj in zip(bins_out, adjusted):
            entry["p_value_adjusted"] = float(adj)
        min_p = float(min(p_values))
        min_adj = float(np.min(adjusted))
        n_violated = int(np.sum(adjusted < 0.05))
        if n_violated:
            # A measured breach is a finding whatever the rest of the design
            # could not see, and it proves this bin HAD the power. FAIL wins.
            passes = False
            status = "assessed"
        elif detectable is True:
            passes = True
            status = "assessed"
        else:
            # Nothing breached, and the design could not have shown a breach (or
            # its power could not be established). That is could-not-check, and
            # collapsing it into PASS is the whole defect: it is a clean bill of
            # health issued by a test that was never able to fail.
            passes = None
            status = "not_assessable"
    else:
        # ZERO bins tested means there was no evidence at all (e.g. the group
        # score distributions do not overlap). passes=True here collapsed
        # could-not-check into PASS, the exact false-PASS class 0b8bf54
        # eliminated for the sealed metrics. Three states, never two:
        # passes=None / status='not_assessable' is the only honest verdict.
        min_p = None
        min_adj = None
        n_violated = 0
        passes = None
        status = "not_assessable"
        detectability_note = (
            "COULD NOT CHECK: no bin could be tested, so there was no design to "
            "have power. See n_bins_skipped."
        )

    return {
        "bins": bins_out,
        "overall": {
            "min_bin_p_value": min_p,
            "min_bin_p_value_adjusted": min_adj,
            "n_bins_violated": n_violated,
            "n_bins_tested": n_tested,
            "n_bins_skipped": int(n_bins - n_tested),
            "passes": passes,
            "status": status,
            # The groups this verdict covers, and the ones it does not. A pass
            # is a statement about the compared groups only.
            "groups_tested": list(valid_groups),
            "excluded_groups": list(excluded_groups),
            "correction": "benjamini_hochberg",
            "method": method,
            "min_attainable_p_value": worst_floor,
            "detectable": detectable,
            "detectability_note": detectability_note,
        },
    }


def cv_calibration_stability(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_folds: int = 5,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_group_size: int = 30,
    random_state: int = 42,
) -> Dict[str, dict]:
    """
    Cross-validated stability of per-group ECE across K folds.

    Uses ``sklearn.model_selection.KFold`` (shuffled, seeded). On each held-out
    fold the per-group ECE and the overall ECE are computed; the fold values are
    aggregated to mean / std / coefficient of variation per group.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute defining groups
        n_folds: Number of KFold splits
        n_bins: Number of bins for ECE
        strategy: Binning strategy
        min_group_size: Minimum group size (on the full data) to include a group
        random_state: Seed for KFold shuffling

    Returns:
        Dict mapping each group name (and '__overall__') to a dict with
        fold_eces, n_folds_measured, mean_ece, std_ece, cv_coefficient,
        degradation_flag (std_ece > 0.02) and not_assessed.

        ``degradation_flag`` is None, never False, when the spread across folds
        could not be measured (BGL3, 2026-09-27). Two ways in, both measured on
        this repo with 29 + 29 + 2 rows, n_folds=2, min_group_size=2 and zero
        warnings:

        * ZERO folds left the group with two scored rows, so nothing was
          computed, and the entry read ``{'fold_eces': [], 'mean_ece': None,
          'std_ece': None, 'cv_coefficient': None, 'degradation_flag': False}``
          (random_state=2). False is the reading "this group's calibration is
          stable across folds".
        * ONE fold did, and ``np.std`` of a single value is exactly 0.0, so the
          entry read ``std_ece 0.0, cv_coefficient 0.0, degradation_flag
          False`` (random_state=0): zero fold-to-fold variation, from one fold.
          The single fold's ECE is a real measurement and is still reported in
          mean_ece; the DISPERSION over it is not.

        A group below ``min_group_size`` is a key of this dict too, with None
        figures and a ``not_assessed`` reason, rather than being dropped in
        silence.
    """
    from sklearn.model_selection import KFold

    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob, protected_attr)
    validate_probabilities(y_prob, "y_prob")

    n = len(y_true)
    pa_str = np.array([str(v) for v in protected_attr], dtype=object)

    gm_full = GroupManager(protected_attr, min_group_size=min_group_size)
    valid_groups = gm_full.get_valid_groups()

    fold_eces: Dict[str, List[float]] = {g: [] for g in valid_groups}
    overall_fold_eces: List[float] = []

    def _ece(y_t: np.ndarray, y_p: np.ndarray) -> float:
        if len(y_t) < 2:
            return float("nan")
        res = expected_calibration_error(
            y_t, y_p, protected_attr=None, n_bins=n_bins, strategy=strategy
        )
        return float(res.overall_value)

    n_splits = max(2, min(int(n_folds), n)) if n >= 2 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx_all = np.arange(n)

    for _train_idx, test_idx in kf.split(idx_all):
        y_t = y_true[test_idx]
        y_p = y_prob[test_idx]
        pa_f = pa_str[test_idx]

        overall_fold_eces.append(_ece(y_t, y_p))

        for g in valid_groups:
            g_mask = pa_f == g
            if int(np.sum(g_mask)) >= 2:
                fold_eces[g].append(_ece(y_t[g_mask], y_p[g_mask]))

    def _aggregate(label: str, values: List[float]) -> dict:
        arr = np.array(
            [v for v in values if v is not None and not np.isnan(v)],
            dtype=float,
        )
        # A spread needs two folds. See the Returns note for what one fold, and
        # what none at all, used to report.
        if len(arr) < 2:
            reason = (
                f"{label}: {len(arr)} of the {n_splits} fold(s) left enough scored rows "
                f"to compute an ECE, so the fold-to-fold spread does not exist. "
                f"std_ece, cv_coefficient and degradation_flag are None "
                f"(could-not-check), not 0.0 and False, which would report perfect "
                f"stability from " + ("no folds" if len(arr) == 0 else "a single fold")
            )
            warnings.warn(
                f"cv_calibration_stability: {reason}.",
                UserWarning,
                stacklevel=3,
            )
            return {
                "fold_eces": [float(x) for x in arr],
                "n_folds_measured": int(len(arr)),
                # One fold IS a measured ECE, and discarding it would throw away
                # evidence that was really collected; only the dispersion is
                # refused.
                "mean_ece": float(arr[0]) if len(arr) == 1 else None,
                "std_ece": None,
                "cv_coefficient": None,
                "degradation_flag": None,
                "not_assessed": reason,
            }
        mean_e = float(np.mean(arr))
        std_e = float(np.std(arr, ddof=0))
        # A coefficient of variation is std / mean, and at a mean of zero it has
        # no value. Every fold ECE is non-negative, so a mean this small forces
        # every one of them to be ~0 and std_e with them: 0.0 is then the
        # measured "no variation" and not a stand-in for an undefined ratio.
        cv_coef = float(std_e / mean_e) if mean_e > 1e-12 else 0.0
        return {
            "fold_eces": [float(x) for x in arr],
            "n_folds_measured": int(len(arr)),
            "mean_ece": mean_e,
            "std_ece": std_e,
            "cv_coefficient": cv_coef,
            "degradation_flag": bool(std_e > 0.02),
            "not_assessed": None,
        }

    results: Dict[str, dict] = {}
    for g in valid_groups:
        results[g] = _aggregate(f"group {g!r}", fold_eces[g])

    # An undersized group is named rather than dropped, the same disclosure
    # expected_calibration_error makes through this helper.
    sizes = gm_full.get_group_sizes()
    for g in _disclose_excluded_groups(gm_full, min_group_size, "cross-validated ECE stability"):
        results[g] = {
            "fold_eces": [],
            "n_folds_measured": 0,
            "mean_ece": None,
            "std_ece": None,
            "cv_coefficient": None,
            "degradation_flag": None,
            "not_assessed": (
                f"group {g!r} holds {int(sizes.get(g, 0))} row(s), fewer than "
                f"min_group_size={min_group_size}, so its calibration was not measured "
                f"on any fold"
            ),
        }

    results["__overall__"] = _aggregate("the pooled sample", overall_fold_eces)

    return results


# ===========================================================================
# spec-v2 spine additions: recalibration diagnostics + smooth/subgroup calibration
#
# The EU-healthcare profile seals on GROUP multicalibration / the integrated
# calibration index (NOT bare bin-ECE, which depends on an arbitrary grid), with
# calibration-in-the-large and the calibration slope as the TRIPOD recalibration
# validity diagnostics. Dependency-light (numpy + a small IRLS + a Gaussian
# kernel), so they run wherever the rest of the engine runs.
#
# References:
#   - Van Calster, Nieboer, Vergouwe, De Cock, Pencina & Steyerberg (2016),
#     "A calibration hierarchy for risk models" (calibration-in-the-large, slope).
#   - Austin & Steyerberg (2019), "The Integrated Calibration Index (ICI)".
#   - Hebert-Johnson, Kim, Reingold & Rothblum (2018), "Multicalibration".
# ===========================================================================


@dataclass
class IntegratedCalibrationResult:
    """Integrated Calibration Index (ICI), overall and across groups.

    ICI is the mean absolute distance between a prediction and a smoothed
    (kernel-regressed) estimate of the observed event rate at that prediction,
    so unlike bin-ECE it does not depend on an arbitrary bin grid. The sealed
    healthcare disparity statistic is ``ici_disparity`` (max between-group gap).
    """

    overall_ici: float
    group_ici: Dict[str, float]
    ici_disparity: float
    most_miscalibrated_group: str
    least_miscalibrated_group: str
    n_groups: int
    bandwidth: float
    excluded_groups: List[str] = field(default_factory=list)

    @property
    def has_significant_disparity(self) -> Optional[bool]:
        """Whether the ICI gap exceeds 0.05: True / False / None.

        An ICI gap over 0.05 between groups is a material calibration
        disparity. ``None`` is could-not-check: fewer than two groups were
        assessable, so ``ici_disparity`` is NaN and no between-group comparison
        was ever made.

        BGL-S2G08 (2026-09-16). This was the bare comparison
        ``self.ici_disparity > 0.05``, and ``nan > 0.05`` is False, so an
        unmeasured comparison graded as a CLEAN BILL OF HEALTH on calibration
        parity. Measured before the fix with a single group of 200 rows:
        ``ici_disparity nan, has_significant_disparity False, n_groups 1,
        excluded_groups []`` and NO warning, so ``to_dict()`` serialised
        ``'has_significant_disparity': false`` beside ``'ici_disparity': NaN``
        and the only thing separating "no material disparity" from "nothing was
        compared" was a NaN in a sibling field. Mirrors
        :attr:`CalibrationDisparityResult.has_significant_disparity`.

        G03 (2026-09-30). It mirrored the FIRST half of that sibling and not the
        second. ``ici_disparity`` is the max-min spread over the groups that
        cleared ``min_group_size``, so with a group left out it is a LOWER BOUND
        and the excluded group can decide the comparison either way, exactly as
        for ECE. MEASURED before this fix on 180 'a' + 180 'b' (both essentially
        perfectly calibrated) beside 15 'c' scored 0.9 against an outcome of 0,
        at the default gate of 30::

            excluded_groups  ['c']
            group_ici        {'a': 0.00024, 'b': 0.00024}
            ici_disparity    0.0
            has_significant_disparity               False   # a clean bill of health
            to_dict()['has_significant_disparity']  False

        The SAME data at ``min_group_size=10``, which lets 'c' in, gives
        ``ici_disparity 0.89976`` and ``True``: the group left out is the one
        that flips the verdict. A determinate BREACH on partial evidence keeps
        its finding, for the reason the sibling records: a spread already over
        the threshold cannot be argued away by a group nobody measured.
        """
        if not np.isfinite(self.ici_disparity):
            return None
        if self.n_groups < 2:
            # Belt and braces: the no-protected-attr path documents a 0.0
            # disparity because no disparity question was asked. That is still
            # not a measured parity, so it is not a False either.
            return None
        breach = bool(self.ici_disparity > 0.05)
        if breach or not self.uncovered_groups:
            return breach
        return None

    @property
    def uncovered_groups(self) -> List[str]:
        """Every group the ICI comparison does NOT cover, from either gate.

        ``excluded_groups`` holds what the row-COUNT gate removed; ``group_ici``
        can also hold a key whose VALUE is not a number, and a count gate cannot
        see that one. Both are uncovered, so both belong in one list, and
        :attr:`has_significant_disparity` reads this rather than either half.
        G03, 2026-09-30.
        """
        unmeasured = [str(g) for g, v in (self.group_ici or {}).items() if not _is_measured(v)]
        return [str(g) for g in self.excluded_groups] + unmeasured

    def to_dict(self) -> Dict:
        return {
            "overall_ici": self.overall_ici,
            "group_ici": self.group_ici,
            "ici_disparity": self.ici_disparity,
            "most_miscalibrated_group": self.most_miscalibrated_group,
            "least_miscalibrated_group": self.least_miscalibrated_group,
            "n_groups": self.n_groups,
            "bandwidth": self.bandwidth,
            "has_significant_disparity": self.has_significant_disparity,
            "excluded_groups": self.excluded_groups,
            # Both gates, in one list, at the boundary a consumer reads. G03.
            "uncovered_groups": self.uncovered_groups,
        }


@dataclass
class MulticalibrationResult:
    """Multicalibration audit (Hebert-Johnson et al. 2018).

    The multicalibration error ``alpha`` is the worst calibration violation over
    all (group x prediction-bin) cells with adequate support: a truly
    multicalibrated model is well calibrated within EVERY subgroup, not just on
    average. ``weighted_mean`` is the support-weighted average violation over
    the evaluated cells (cells below ``min_cell`` carry no evidence and are
    excluded from both numerator and denominator). ``group_max`` is NaN for a
    group with zero evaluated cells.
    """

    alpha: float
    weighted_mean: float
    worst_group: str
    worst_bin: int
    group_max: Dict[str, float]
    n_groups: int
    n_bins: int
    min_cell: int
    excluded_groups: List[str] = field(default_factory=list)

    @property
    def is_multicalibrated(self) -> Optional[bool]:
        """Worst subgroup calibration violation under 0.05: True / False / None.

        ``None`` is could-not-check: ``alpha`` is NaN because no (group x bin)
        cell reached ``min_cell``, so the audit had ZERO calibration evidence
        and no violation could have been found.

        BGL-S2G08 (2026-09-16), found while fixing
        :attr:`IntegratedCalibrationResult.has_significant_disparity`, which is
        the same shape one class up in this file. This was the bare comparison
        ``self.alpha < 0.05``, and ``nan < 0.05`` is False, so an audit that
        evaluated nothing was graded "not multicalibrated" and ``to_dict()``
        serialised that False. Measured before the fix with 60 rows over 60
        bins at min_cell=10: ``alpha nan, is_multicalibrated False``. The
        direction happens to be conservative for a gate, but it is still a
        verdict nobody measured, and the reader cannot tell it from a real
        violation. The NaN itself was deliberate (see ``alpha_out``); only the
        grade had not been carried across.

        G03 (2026-09-30), the CLEAN direction of the same collapse, and the more
        dangerous one. ``alpha`` is the worst violation over the cells that were
        EVALUATED, so ``alpha < 0.05`` is a pass statement about a SUBSET, and
        the group the audit never reached is exactly the group a multicalibration
        audit exists to find. Two doors, both measured on 180 'a' + 180 'b'
        (essentially perfectly calibrated):

        - the row-COUNT gate. With 15 'c' rows scored 0.9 against an outcome of
          0, at ``min_group_size=30``: ``excluded_groups ['c']``,
          ``group_max {'a': 0.0, 'b': 0.0}``, ``alpha 3.33e-16``,
          ``is_multicalibrated True``. The same data at ``min_group_size=10``
          gives ``alpha 0.9`` and ``False``.
        - the EVIDENCE gate, which no count can see and which is the reachable
          one here. With 40 'd' rows spread thinly over 20 bins so no cell
          reached ``min_cell``: ``excluded_groups []`` (d cleared the size gate),
          ``group_max {'a': 0.0, 'b': 0.0, 'd': nan}``, ``alpha 3.33e-16``,
          ``is_multicalibrated True`` and NOT ONE WARNING. "This model is
          multicalibrated" was published for a model in which a whole group
          carried no calibration evidence at all, and a guard keyed on
          ``excluded_groups`` alone would have passed it.

        A measured VIOLATION on partial evidence keeps its ``False``: a cell
        already over the threshold cannot be argued away by a group nobody
        evaluated. Same rule as
        :attr:`CalibrationDisparityResult.has_significant_disparity`, with the
        polarity the other way up because here the clean verdict is the True.
        """
        if not np.isfinite(self.alpha):
            return None
        multicalibrated = bool(self.alpha < 0.05)
        if not multicalibrated or not self.uncovered_groups:
            return multicalibrated
        return None

    @property
    def uncovered_groups(self) -> List[str]:
        """Every group the audit does NOT cover, from either gate.

        ``excluded_groups`` holds what ``min_group_size`` removed. ``group_max``
        holds NaN for a group that cleared that gate and still had no cell
        reaching ``min_cell``, which is the door a count gate cannot see. Both
        are uncovered, so both belong in one list. G03, 2026-09-30.
        """
        no_cells = [str(g) for g, v in (self.group_max or {}).items() if not _is_measured(v)]
        return [str(g) for g in self.excluded_groups] + no_cells

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "weighted_mean": self.weighted_mean,
            "worst_group": self.worst_group,
            "worst_bin": self.worst_bin,
            "group_max": self.group_max,
            "n_groups": self.n_groups,
            "n_bins": self.n_bins,
            "min_cell": self.min_cell,
            "is_multicalibrated": self.is_multicalibrated,
            "excluded_groups": self.excluded_groups,
            # Both gates, in one list, at the boundary a consumer reads. G03.
            "uncovered_groups": self.uncovered_groups,
        }


def _cal_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -709.0, 709.0)))


def _cal_safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# THERE IS NO MAGNITUDE BOUND ON THE FITTED COEFFICIENTS, AND ADDING ONE BACK
# DELETES A REAL FINDING.
#
# A ``_CAL_SEPARATION_BOUND = 20.0`` used to sit here and both callers below
# refused any converged fit whose largest coefficient exceeded it, on the
# reasoning that "the predictors are clipped logits, whose own range is about
# +/- 13.8, so a larger coefficient means the likelihood is still climbing".
# That reasoning silently assumes the scores SPAN that range. They need not.
#
# BGL-final-d00 (2026-09-17), measured at the public entry. 4000 rows, scores
# only ever 0.499 or 0.501, outcome rate 5 percent on the low score and 95
# percent on the high one. The predictor gap is logit(0.501) - logit(0.499) =
# 0.008 and the outcome log-odds gap is log(.95/.05) - log(.05/.95) = 5.888, so
# the MLE slope is 5.888 / 0.008 = about 728: an enormous but entirely real
# slope, and IRLS returned it with ``converged=True``.
#
#   before:  calibration_slope(y, p).overall_value -> nan, metadata
#            unmeasured_reason "the logistic fit diverged (separation)",
#            is_well_calibrated None, plus a could-not-check warning
#   after:   overall_value 728.90, measured True, is_well_calibrated False
#
# (728.90 is the sampled version of that fixture. The pin in
# tests/test_bgl_final_d00.py uses the deterministic one, exactly 5 percent and
# 95 percent, whose slope has the closed form 736.1088 and is returned to twelve
# significant figures, which is how we know it is the MLE and not an iterate.)
#
# A slope of 728 against a target band of [0.8, 1.2] is one of the most extreme
# miscalibration findings this metric can produce (predictions carrying almost
# no spread while the outcomes carry all of it), and the bound reported it as
# "nothing could be measured here". That is the fabricated-verdict defect
# running backwards: refusing evidence you DO have is not caution, it throws the
# finding away. The MLE either exists or it does not, and the thing that answers
# that question is the convergence flag returned by :func:`_irls_logistic`,
# together with the empty / single-outcome-class / constant-score tests in
# :func:`_cox_recalibration`. Those catch every separable case, measured: 60
# rows with every label 1 give ``converged=False`` on both the Cox fit and the
# CITL fit, and a saturated CITL offset walks to 8.3e8 with ``converged=False``.
# A non-finite coefficient is still refused, because that is not a number.


def _irls_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """Newton-Raphson (IRLS) logistic fit. ``X`` (n, k) includes any intercept
    column; ``offset`` (n,) enters the linear predictor with a fixed coefficient
    of 1 (used for the slope-fixed calibration-in-the-large). A tiny ridge on the
    Hessian keeps it stable on separable cells. Dependency-free.

    Returns ``(beta, converged)``. ``converged`` is False when the loop
    exhausted ``max_iter`` with the step still above ``tol``, which is what a
    separable fit does: the coefficient walks off toward infinity and the
    ITERATE at iteration 100 is not an estimate of anything.

    BGL-S2G08 (2026-09-16). The flag is new. Callers used to take the returned
    array unconditionally, so ``calibration_in_the_large`` on 60 separable rows
    (every label 1) reported ``overall_value 25.87`` as a measurement, and on
    ZERO rows the loop computed an empty gradient, took a zero step, passed the
    convergence test on iteration 1 and handed back the untouched
    ``beta = np.zeros(k)`` initialiser: ``overall_value 0.0``, the exact centre
    of the metric's own well calibrated band, graded ``is_well_calibrated True``
    on no rows at all.

    ``converged=True`` IS NOT A PROOF THAT THE MLE EXISTS, so do not use this
    flag as the only separability test. BGL-final-d00 (2026-09-17): when ``mu``
    saturates, ``mu * (1 - mu)`` hits the ``1e-9`` clip, which FLOORS the
    Hessian while the gradient decays exponentially, so the Newton step can fall
    under ``tol`` at a point that is not the maximum. Measured on the
    intercept-only (CITL) fit, 4000 rows, every label 1, constant score
    sigmoid(-4): ``(55.598, True)``, with the gradient exactly 0.0 because
    ``sigmoid`` underflows to 1.0 in float64. Each caller therefore carries its
    own STRUCTURAL separability test as well: see :func:`_cox_recalibration` and
    :func:`_citl_value`. A magnitude bound is not an acceptable substitute for
    those, because an extreme coefficient is often a real measurement.
    """
    n, k = X.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    y = np.asarray(y, dtype=float)
    converged = False
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = _cal_sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        XtW = X.T * w
        H = XtW @ X + 1e-8 * np.eye(k)
        grad = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    return beta, converged


def _cox_recalibration(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, Optional[str]]:
    """Cox recalibration: fit logit(P(y=1)) = alpha + beta * logit(p). Returns
    ``(intercept alpha, slope beta, unmeasured_reason)``. Perfect calibration
    => (0, 1). Both coefficients are NaN, and the reason is a short string,
    whenever the fit is not a measurement.

    BGL-S2G08 (2026-09-16). Three unmeasurable inputs all used to hand back a
    confident finite slope, measured at the public entry:

    - ZERO rows -> ``0.0``: the ``beta = np.zeros(2)`` initialiser, never
      updated, graded against the [0.8, 1.2] band as "not well calibrated".
    - CONSTANT scores, 60 real rows at p=0.7 -> ``0.0``. logit(p) is then a
      constant column, perfectly collinear with the intercept, so the slope is
      NOT IDENTIFIABLE; the ``1e-8`` ridge on the Hessian makes the singular
      system solvable and returns the minimum-norm answer, which a reader reads
      as "catastrophically overfit predictions". This is the dangerous one: 60
      rows do not look degenerate to a caller.
    - A SINGLE outcome class, 60 rows all y=1 -> ``-5.4e-16`` beside an
      intercept of 25.0, a separable fit reported at its 100th iterate.

    The zero-variance test is on ``np.unique`` of the actual (clipped) predictor
    column, never ``np.var(...) == 0``: an accumulated variance is not exactly
    0.0 at every n.

    An EXTREME converged slope is a measurement, not a refusal. Scores confined
    to a narrow band (0.499 / 0.501) against outcomes that swing 5 percent to 95
    percent give a genuine MLE slope near 730, and it is returned. See the
    comment above :func:`_irls_logistic` for the bound that used to hide it.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan"), "no rows"
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), "a single outcome class (no contrast to fit)"
    lp = _cal_safe_logit(p)
    if np.unique(lp).size < 2:
        return (
            float("nan"),
            float("nan"),
            "a constant score (logit(p) is collinear with the intercept, so the "
            "slope is not identifiable)",
        )
    X = np.column_stack([np.ones_like(lp), lp])
    coef, converged = _irls_logistic(X, y)
    if not converged:
        return float("nan"), float("nan"), "the logistic fit did not converge (separation)"
    # No magnitude bound: a converged MLE is a measurement however extreme it
    # is. See the comment above _irls_logistic for the finding a bound deleted.
    if not np.all(np.isfinite(coef)):
        return float("nan"), float("nan"), "the logistic fit returned a non-finite coefficient"
    return float(coef[0]), float(coef[1]), None


def _citl_value(y: np.ndarray, p: np.ndarray) -> Tuple[float, Optional[str]]:
    """Calibration-in-the-large: the intercept of a logistic fit with logit(p) as
    a fixed-slope (=1) offset. 0 => predicted risks match the observed rate.
    Returns ``(value, unmeasured_reason)``; the value is NaN when the fit is not
    a measurement (see :func:`_irls_logistic` for the measured before-values).

    THE SINGLE-OUTCOME-CLASS TEST IS EXACT HERE, not a heuristic. This model has
    ONE free parameter, so its score equation is
    ``sum(y) - sum(sigmoid(a + lp_i)) = 0``. The right-hand sum runs from 0 to n
    as ``a`` goes from -inf to +inf, so an interior maximum exists for EVERY y
    with ``0 < sum(y) < n`` and for no other y. ``sum(y)`` at an endpoint is
    complete separation: the likelihood climbs forever and there is no finite
    MLE to report.

    BGL-final-d00 (2026-09-17). This guard is why removing the magnitude bound
    above is safe. The bound was masking a convergence flag that says True at a
    point that is not the maximum: with every label 1 and a CONSTANT score, mu
    saturates, ``mu * (1 - mu)`` hits the ``1e-9`` clip that floors the Hessian
    while the gradient decays exponentially, so the Newton step falls under
    ``tol`` and the loop reports success. Measured on 4000 rows, every label 1::

        p = sigmoid(-4)     -> 55.598     converged=True, gradient exactly 0.0
        p = sigmoid(-13.8)  -> 984607.69  converged=True, gradient exactly 0.0

    The gradient is exactly 0.0 because ``sigmoid`` underflows to 1.0 in float64,
    so no gradient-based convergence test can catch this either; the structural
    test above is what does. The 2-parameter Cox fit does not share the hole (its
    two coefficients grow at different rates, so the step stays above ``tol``),
    measured across perfect score separation at n = 60, 400 and 4000.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan"), "no rows"
    if np.unique(y).size < 2:
        return (
            float("nan"),
            "a single outcome class, which is complete separation for an "
            "intercept-only fit (the likelihood has no interior maximum, so no "
            "finite MLE exists)",
        )
    lp = _cal_safe_logit(p)
    X = np.ones((len(y), 1))
    coef, converged = _irls_logistic(X, y, offset=lp)
    if not converged:
        return float("nan"), "the logistic fit did not converge (separation)"
    # No magnitude bound, for the reason recorded above _irls_logistic.
    if not np.isfinite(coef[0]):
        return float("nan"), "the logistic fit returned a non-finite coefficient"
    return float(coef[0]), None


def calibration_in_the_large(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    min_group_size: int = 30,
) -> CalibrationMetricResult:
    """Calibration-in-the-large (CITL), overall and per group.

    CITL is the intercept of a logistic recalibration in which logit(p) is
    entered as a fixed-slope offset. 0 means predicted risks match the observed
    event rate on average; a positive value means the model under-predicts risk.
    The stored ``overall_value`` is |CITL| (the seal gates |CITL| <= 0.05); the
    signed value and per-group signed values are in ``metadata``.

    Reference: Van Calster et al. (2016), a calibration hierarchy for risk models.

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

    Ledger row: calibration_in_the_large. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    overall_signed, overall_reason = _citl_value(y_true, y_prob)
    if overall_reason is not None:
        warnings.warn(
            f"Calibration in the large could not be measured: {overall_reason}. "
            f"overall_value is NaN, not 0.0, and is_well_calibrated is None: "
            f"0.0 is the centre of the well calibrated band, so it would have "
            f"read as a perfect result."
        )
    group_values: Optional[Dict[str, float]] = None
    signed_by_group: Dict[str, float] = {}
    unmeasured_groups: Dict[str, str] = {}
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)
        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = _disclose_excluded_groups(gm, min_group_size, "calibration in the large")
        group_values = {}
        for g in gm.get_valid_groups():
            m = gm.get_mask(g)
            s, reason = _citl_value(y_true[m], y_prob[m])
            if reason is not None:
                unmeasured_groups[g] = reason
            signed_by_group[g] = s
            group_values[g] = abs(s)
        if unmeasured_groups:
            warnings.warn(
                f"Calibration in the large could not be measured for groups "
                f"{unmeasured_groups}. Their values are NaN, not 0.0."
            )

    return CalibrationMetricResult(
        metric_name="calibration_in_the_large",
        overall_value=abs(overall_signed),
        group_values=group_values,
        n_samples=len(y_true),
        metadata={
            "signed": overall_signed,
            "signed_by_group": signed_by_group,
            "excluded_groups": excluded,
            # THREE states. measured True, or measured False with a reason
            # naming why the fit was not an estimate. Never a neutral 0.0.
            "measured": overall_reason is None,
            "unmeasured_reason": overall_reason,
            "unmeasured_groups": unmeasured_groups,
            "interpretation": "0 = perfect; |CITL| <= 0.05 = well calibrated in the large",
        },
    )


def calibration_slope(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    min_group_size: int = 30,
) -> CalibrationMetricResult:
    """Calibration slope, overall and per group.

    The slope beta of the Cox recalibration logit(P(y=1)) = alpha + beta*logit(p).
    1.0 = predictions neither over- nor under-fit; < 1 = predictions too extreme
    (overfit); > 1 = too moderate. The seal gates the slope to [0.8, 1.2].
    ``overall_value`` is the slope itself (NOT an error), so
    ``is_well_calibrated`` tests the target band 0.8 <= slope <= 1.2, which is
    also recorded in ``metadata``.

    Reference: Van Calster et al. (2016), a calibration hierarchy for risk models.

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

    Ledger row: calibration_slope. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    _, overall_slope, overall_reason = _cox_recalibration(y_true, y_prob)
    if overall_reason is not None:
        warnings.warn(
            f"The calibration slope could not be measured: {overall_reason}. "
            f"overall_value is NaN, not 0.0, and is_well_calibrated is None. "
            f"A slope of 0.0 is a real and alarming reading (predictions carry "
            f"no signal), so returning it here invented a finding."
        )
    group_values: Optional[Dict[str, float]] = None
    unmeasured_groups: Dict[str, str] = {}
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)
        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = _disclose_excluded_groups(gm, min_group_size, "calibration slope")
        group_values = {}
        for g in gm.get_valid_groups():
            m = gm.get_mask(g)
            _, b, reason = _cox_recalibration(y_true[m], y_prob[m])
            if reason is not None:
                unmeasured_groups[g] = reason
            group_values[g] = b
        if unmeasured_groups:
            warnings.warn(
                f"The calibration slope could not be measured for groups "
                f"{unmeasured_groups}. Their values are NaN, not 0.0."
            )

    return CalibrationMetricResult(
        metric_name="calibration_slope",
        overall_value=overall_slope,
        group_values=group_values,
        n_samples=len(y_true),
        metadata={
            "target": 1.0,
            "well_calibrated_band": [0.8, 1.2],
            "excluded_groups": excluded,
            # THREE states: measured, or not measured with the reason named.
            "measured": overall_reason is None,
            "unmeasured_reason": overall_reason,
            "unmeasured_groups": unmeasured_groups,
            "interpretation": "1.0 = ideal; <1 overfit (too extreme); >1 underfit (too moderate)",
        },
    )


def _kernel_calibration_curve(y: np.ndarray, p: np.ndarray, bandwidth: float) -> np.ndarray:
    """Nadaraya-Watson (Gaussian-kernel) estimate of E[y|p] at each p_i, computed
    on a fixed grid then linearly interpolated back (O(n * grid) rather than
    O(n^2)). Returns the smoothed observed-rate g(p_i) for every i."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    grid = np.linspace(0.0, 1.0, 101)
    diffs = (grid[:, None] - p[None, :]) / bandwidth  # (grid, n)
    K = np.exp(-0.5 * diffs**2)
    denom = np.clip(K.sum(axis=1), 1e-12, None)
    g_grid = np.clip((K @ y) / denom, 0.0, 1.0)
    return np.interp(p, grid, g_grid)


def _default_bandwidth(p: np.ndarray) -> float:
    """Silverman's rule of thumb for the kernel bandwidth, floored at 0.02 so a
    near-constant score does not collapse the smoother."""
    n = len(p)
    sd = float(np.std(p)) if n > 1 else 0.1
    h = 1.06 * sd * (max(n, 1) ** (-1.0 / 5.0)) if sd > 0 else 0.1
    return max(h, 0.02)


def _ici_value(y: np.ndarray, p: np.ndarray, bandwidth: float) -> float:
    g = _kernel_calibration_curve(y, p, bandwidth)
    return float(np.mean(np.abs(p - g)))


def integrated_calibration_index(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: Optional[ArrayLike] = None,
    bandwidth: Optional[float] = None,
    min_group_size: int = 30,
) -> IntegratedCalibrationResult:
    """Integrated Calibration Index (ICI), overall and across groups.

    ICI = mean_i |p_i - g(p_i)|, where g is a Gaussian-kernel smoothed estimate of
    the observed event rate at each prediction. It is the grid-free calibration
    statistic the EU-healthcare cell seals on; the sealed disparity is
    ``ici_disparity`` (max between-group ICI gap). The same bandwidth is used for
    every group so the gaps are comparable.

    Reference: Austin & Steyerberg (2019), The Integrated Calibration Index.

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

    Ledger row: integrated_calibration_index. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    check_consistent_length(y_true, y_prob)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    h = float(bandwidth) if bandwidth is not None else _default_bandwidth(y_prob)
    overall_ici = _ici_value(y_true, y_prob, h)

    group_ici: Dict[str, float] = {}
    excluded: List[str] = []
    if protected_attr is not None:
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, protected_attr)
        gm = GroupManager(protected_attr, min_group_size=min_group_size)
        excluded = gm.get_invalid_groups()
        if excluded:
            warnings.warn(
                f"Groups excluded from ICI disparity (below min_group_size="
                f"{min_group_size}): {excluded}. Disparity does not cover them."
            )
        for g in gm.get_valid_groups():
            m = gm.get_mask(g)
            group_ici[g] = _ici_value(y_true[m], y_prob[m], h)

    if len(group_ici) >= 2:
        values = list(group_ici.values())
        disparity = max(values) - min(values)
        most = max(group_ici, key=lambda k: group_ici[k])
        least = min(group_ici, key=lambda k: group_ici[k])
    elif protected_attr is not None:
        # A group analysis was requested but fewer than two groups were
        # assessable (the rest fell below min_group_size): the between-group
        # disparity is UNMEASURABLE. 0.0 here sealed 'perfect parity' on no
        # evidence with a [0.0, 0.0] bootstrap CI that clears any gate, while
        # auroc_parity on the same data correctly said NOT_ASSESSABLE. NaN
        # routes it to insufficient evidence (mirror of fd9746b); overall_ici
        # stays computed, only the disparity is undefined.
        #
        # BGL-S2G08 (2026-09-16): the warning is new. With ONE valid group and
        # nothing excluded (a single-valued protected attribute) no warning fired
        # at all and excluded_groups was empty, so a NaN in one field was the
        # only trace that no comparison had happened, next to
        # has_significant_disparity reading a bare False.
        warnings.warn(
            f"Fewer than 2 groups with sufficient samples for ICI disparity "
            f"(measured groups: {sorted(group_ici)}). ici_disparity is NaN, not "
            f"0.0, and has_significant_disparity is None: no between-group "
            f"comparison was made."
        )
        disparity, most, least = float("nan"), "", ""
    else:
        # No group analysis requested: no disparity question was asked.
        disparity, most, least = 0.0, "", ""

    return IntegratedCalibrationResult(
        overall_ici=overall_ici,
        group_ici=group_ici,
        ici_disparity=disparity,
        most_miscalibrated_group=most,
        least_miscalibrated_group=least,
        n_groups=len(group_ici),
        bandwidth=h,
        excluded_groups=excluded,
    )


def multicalibration(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    n_bins: int = 10,
    min_group_size: int = 30,
    min_cell: int = 10,
) -> MulticalibrationResult:
    """Multicalibration audit across (group x prediction-bin) cells.

    For every valid group the predictions are uniformly binned; within each bin
    with at least ``min_cell`` samples the calibration violation |mean(y) -
    mean(p)| is measured. ``alpha`` (the multicalibration error) is the worst
    violation over all such cells: it is small only when the model is well
    calibrated within EVERY subgroup, which is the property the impossibility
    theorem shows a single overall-ECE gate can hide. ``weighted_mean`` is the
    support-weighted average violation over the evaluated cells only (cells
    below ``min_cell`` are excluded from numerator AND denominator).

    Reference: Hebert-Johnson, Kim, Reingold & Rothblum (2018), Multicalibration.

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

    Ledger row: multicalibration. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    y_true = coerce_to_array(y_true, "y_true")
    y_prob = coerce_to_array(y_prob, "y_prob")
    protected_attr = coerce_to_array(protected_attr, "protected_attr")
    check_consistent_length(y_true, y_prob)
    check_consistent_length(y_true, protected_attr)
    validate_binary_labels(y_true, "y_true")
    validate_probabilities(y_prob, "y_prob")

    gm = GroupManager(protected_attr, min_group_size=min_group_size)
    excluded = gm.get_invalid_groups()
    if excluded:
        warnings.warn(
            f"Groups excluded from multicalibration (below min_group_size="
            f"{min_group_size}): {excluded}. The audit does not cover them."
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    group_max: Dict[str, float] = {}
    alpha = 0.0
    worst_group, worst_bin = "", -1
    weighted_sum = 0.0
    evaluated_support = 0
    n_cells_evaluated = 0

    for g in gm.get_valid_groups():
        m = gm.get_mask(g)
        yt, yp = y_true[m], y_prob[m]
        idx = np.clip(np.digitize(yp, bin_edges[1:-1]), 0, n_bins - 1)
        gmax = 0.0
        n_cells_g = 0
        for b in range(n_bins):
            cell = idx == b
            n_cell = int(cell.sum())
            if n_cell < min_cell:
                continue
            n_cells_evaluated += 1
            n_cells_g += 1
            violation = abs(float(np.mean(yt[cell])) - float(np.mean(yp[cell])))
            weighted_sum += n_cell * violation
            evaluated_support += n_cell
            if violation > gmax:
                gmax = violation
            if violation > alpha:
                alpha, worst_group, worst_bin = violation, g, b
        # A group with ZERO evaluated cells contributed no calibration evidence;
        # 0.0 there reads as 'perfectly calibrated group' on no evidence.
        group_max[g] = gmax if n_cells_g > 0 else float("nan")

    # A group that cleared min_group_size and still had no cell reaching
    # min_cell is NOT covered by this audit, and it said so nowhere. G03
    # (2026-09-30): measured on 40 rows spread thinly over 20 bins beside two
    # well calibrated 180-row groups, the result was ``group_max
    # {'a': 0.0, 'b': 0.0, 'd': nan}``, ``excluded_groups []``, ``alpha
    # 3.33e-16``, ``is_multicalibrated True`` and ZERO warnings. The NaN in
    # group_max was the only trace, and it is the one field a gate reading
    # ``alpha`` never looks at. The exclusion warning above fires on the size
    # gate only, which cannot see this door.
    no_cell_groups = sorted(g for g, v in group_max.items() if not _is_measured(v))
    if no_cell_groups:
        warnings.warn(
            f"Groups with NO evaluated (group x bin) cell (none reached min_cell="
            f"{min_cell}): {no_cell_groups}. They cleared min_group_size="
            f"{min_group_size} but contributed no calibration evidence, so alpha and "
            f"is_multicalibrated do NOT cover them: group_max is NaN for each rather "
            f"than 0.0, which would read as a perfectly calibrated group."
        )

    # No (group x bin) cell met min_cell: the audit had ZERO calibration evidence to
    # evaluate. Reporting alpha = 0.0 there would seal "perfectly multicalibrated" on no
    # evidence; emit NaN so the report/verdict route it to insufficient_evidence instead.
    alpha_out = alpha if n_cells_evaluated > 0 else float("nan")

    # Normalize over the EVALUATED support only. Dividing by the full dataset
    # size while sub-min_cell cells contribute nothing to the numerator pulled
    # the 'support-weighted average violation' toward zero exactly when much of
    # the data sat in unevaluated cells (0.45-everywhere read as 0.237).
    weighted_mean_out = weighted_sum / evaluated_support if evaluated_support > 0 else float("nan")

    return MulticalibrationResult(
        alpha=alpha_out,
        weighted_mean=weighted_mean_out,
        worst_group=worst_group,
        worst_bin=worst_bin,
        group_max=group_max,
        n_groups=len(group_max),
        n_bins=n_bins,
        min_cell=min_cell,
        excluded_groups=excluded,
    )


def integrated_calibration_index_with_ci(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    *,
    bandwidth: Optional[float] = None,
    min_group_size: int = 30,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the between-group ICI disparity (the sealed EU-healthcare
    statistic). Bootstraps over row indices, stratified by group; the smoothing bandwidth is
    fixed across resamples so the interval is not confounded by a re-estimated bandwidth. The
    seal gates on the CI UPPER bound being below tolerance (one-sided, pass on affirmative
    evidence). Fewer bootstraps than the difference metrics because each resample refits the
    kernel smoother per group.

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

    Ledger row: integrated_calibration_index_with_ci. See docs/BETA_GO_LIVE_PLAN.md
    for the batch definitions.
    (end Beta Go-Live proof status)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    a = np.asarray(protected_attr)
    h = float(bandwidth) if bandwidth is not None else _default_bandwidth(yp)

    def stat(i: np.ndarray) -> float:
        return integrated_calibration_index(
            yt[i], yp[i], a[i], bandwidth=h, min_group_size=min_group_size
        ).ici_disparity

    return bootstrap_over_index(
        len(yt),
        stat,
        groups=a,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


def multicalibration_with_ci(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    protected_attr: ArrayLike,
    *,
    n_bins: int = 10,
    min_group_size: int = 30,
    min_cell: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """Confidence interval for the multicalibration error alpha (the worst-subgroup calibration
    violation). Bootstraps over row indices, stratified by group. Because alpha is a MAXIMUM over
    (group x bin) cells it is upward-biased in small samples, so the seal reads it against the CI
    (upper bound below tolerance = pass on affirmative evidence), never the point estimate.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: multicalibration_with_ci. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    a = np.asarray(protected_attr)

    def stat(i: np.ndarray) -> float:
        return multicalibration(
            yt[i], yp[i], a[i], n_bins=n_bins, min_group_size=min_group_size, min_cell=min_cell
        ).alpha

    return bootstrap_over_index(
        len(yt),
        stat,
        groups=a,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
