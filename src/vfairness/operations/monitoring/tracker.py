"""
Unit 1: Metric Tracking and Real-Time Bias Detection
=====================================================

This module implements the sliding-window fairness monitoring infrastructure
described in Part 4, Unit 1 of the Fairness Pipeline Development Toolkit.

Key components:

- ``mmd_gaussian``: Maximum Mean Discrepancy with a Gaussian kernel.
  Used to detect whether the feature distribution of a demographic group
  has shifted between two time periods.

- ``FairnessMonitor``: Sliding-window monitor that ingests prediction batches
  as pandas DataFrames and continuously re-computes fairness metrics.
  Supports disparate impact, demographic parity, equalized odds, and equal
  opportunity out of the box.  Custom metric functions are also supported.

- ``TemporalFairnessAnalyzer``: Tracks daily fairness metrics over a rolling
  lookback window and exposes methods for weekly-cycle detection, linear
  trend estimation, and simple forward forecasting.

References
----------
Rabanser et al. (2019). Failing loudly: An empirical study of methods for
  detecting dataset shift. NeurIPS.
Barocas, Hardt & Narayanan (2023). Fairness and machine learning. MIT Press.
Liu et al. (2023). Delayed impact of fair machine learning. CACM 66(5).
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    metric_direction,
)

#: Fewest rows ONE ARM of ONE GROUP may hold for the conditional rate computed
#: over it to be a measurement. It counts the rows the rate's DENOMINATOR is
#: actually made of: the positive-label rows a true-positive rate rests on, the
#: negative-label rows a false-positive rate rests on. That is a LABELLED SUBSET
#: of the group, so it can be a handful while the group itself is large.
#:
#: IT IS DELIBERATELY NOT ``min_samples``, and reusing that would be wrong twice
#: over (F13, 2026-09-30). ``min_samples`` is a WHOLE-GROUP floor, documented as
#: "minimum group size before a metric is reported", and it counts a different
#: thing: applied per arm, its default of 30 refuses a 23-row arm inside a
#: completely healthy 46-row group, and 20-row arms inside 40-row groups carrying
#: a real 0.6 true-positive-rate gap. Measured 2026-09-30: doing that reddens
#: three of this repo's own over-correction controls
#: (tests/test_bgl3_operations_2 twice and
#: tests/test_readiness_tracker::test_over_correction_control_a_fully_sampled_window_is_measured),
#: every one of which asserts a REAL number this monitor must keep reporting. A
#: guard that argues for breaking the product is the wrong guard.
#:
#: 5 is not a new number in this package: it is
#: ``monitoring/drift.py::_MIN_ARM_OBSERVATIONS``, the per-arm floor ``_run_ks``
#: and ``_permutation_calibrated_scales`` already enforce, applied to a different
#: denominator. It is also the classic expected-count condition under every
#: normal approximation to a binomial proportion. The product reason is sharper:
#: a rate over n rows moves in steps of 1/n, and the verdict this monitor draws
#: from it is ``abs(gap) > 1 - alert_threshold``, which is 0.2 at the library
#: default. Below n = 5 one single row moves the gap by MORE than the entire
#: decision band, so the reading cannot separate a breach from a pass whatever
#: the data does. At n = 1 there is no resolution at all: the rate is exactly 0.0
#: or exactly 1.0, and a gap of 0.0 between two such rates is arithmetic, not
#: evidence of equality.
_MIN_ARM_ROWS = 5


# Distribution shift: Maximum Mean Discrepancy


def mmd_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """Compute Maximum Mean Discrepancy (MMD) with a Gaussian kernel.

    MMD measures the distance between two distributions by comparing their
    mean embeddings in a reproducing kernel Hilbert space (RKHS).  A value
    near 0 indicates the distributions are similar; larger values indicate
    greater divergence.

    Parameters
    ----------
    x : np.ndarray
        Samples from the first distribution (reference, 1-D).
    y : np.ndarray
        Samples from the second distribution (current, 1-D).
    sigma : float, default 1.0
        Bandwidth of the Gaussian kernel.  Can be set heuristically to the
        median pairwise distance (the "median heuristic").

    Returns
    -------
    float
        MMD² estimate.  Non-negative; 0 iff distributions are equal.
        ``np.nan`` with a ``UserWarning`` when either sample is empty or holds
        a non-finite value, because the kernel mean embedding is then
        undefined. Never 0.0 for those, which would read as "identical".

    Examples
    --------
    >>> import numpy as np
    >>> from vfairness.operations.monitoring import mmd_gaussian
    >>> rng = np.random.default_rng(42)
    >>> ref = rng.normal(0, 1, 500)
    >>> cur = rng.normal(0.3, 1, 500)       # shifted mean
    >>> mmd_gaussian(ref, cur)
    0.08...
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays.")

    # BGL g004, 2026-09-17. The three inputs below already produced nan, which
    # is the right answer, but the ONLY notice of it was numpy's own
    # "Mean of empty slice" / "invalid value encountered" RuntimeWarnings, and
    # those are routinely filtered out of analysis code and say nothing about
    # distributions. A caller that then writes the nan into a drift score has
    # no domain-level record of why. The value is unchanged; the refusal is now
    # stated in the channel a reader of this library is looking at.
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    n_fin_x = int(np.isfinite(x).sum())
    n_fin_y = int(np.isfinite(y).sum())
    if n_fin_x < len(x) or n_fin_y < len(y) or n_fin_x == 0 or n_fin_y == 0:
        warnings.warn(
            f"mmd_gaussian: the reference sample has {n_fin_x} finite value(s) of "
            f"{len(x)} and the current sample {n_fin_y} of {len(y)}, so the kernel "
            f"mean embedding is undefined and no distance between the two "
            f"distributions was measured. Returning nan (could not measure), never "
            f"0.0, which would read as 'the two distributions are identical'.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    denom = 2.0 * sigma**2
    xx = np.exp(-(np.subtract.outer(x, x) ** 2) / denom).mean()
    yy = np.exp(-(np.subtract.outer(y, y) ** 2) / denom).mean()
    xy = np.exp(-(np.subtract.outer(x, y) ** 2) / denom).mean()
    return float(xx + yy - 2.0 * xy)


def _unscored_predictions(values: pd.Series) -> int:
    """How many rows of a prediction column carry NO prediction at all.

    BGL5 A-operations-2, 2026-09-27. Every rate in this module was built with
    ``Series.mean()`` or ``(Series == 1).mean()``, and pandas SKIPS NaN in the
    first and counts it as ``False`` in the second, so a row that was never
    scored was silently read as a row the model declined. No measurement
    predicate was ever applied to a PREDICTION here, only to the metric computed
    from one.

    Measured before this helper existed, on 50 rows of group A predicted 0.9 and
    50 rows of group B whose prediction is NaN, ``min_samples=30`` so no floor
    fires: ``compute_disparate_impact`` returned exactly ``1.0`` (a perfect
    four-fifths ratio, well above the 0.8 floor) and ``compute_demographic_parity``
    exactly ``0.0`` (perfect parity), both with ZERO warnings, because ``max``
    and ``min`` each skip the NaN so the privileged group equalled the minimum.
    Renaming the groups so the NaN sorted FIRST made the same data answer nan,
    so the honesty rested on dict ordering. On a frame where 20 of group B's 40
    positive-label rows carried NaN, ``compute_equal_opportunity`` and
    ``compute_equalized_odds`` both returned exactly ``0.0`` with zero warnings
    while the TPR over B's SCORED rows was 1.0 against A's 0.5, a real gap of
    0.5. After it: nan on every one of those calls, with a warning naming the
    group and counting its unscored rows.

    ``operations/cicd/monitor.py`` already refuses the identical input on both
    arms, and its own source comment names this order dependence as the defect it
    fixed on 2026-09-17. That guard was never mirrored here.

    A BOOLEAN column is NOT unscored, which is where this deliberately parts from
    :func:`vfairness._triage.is_measured`. A metric value of ``True`` is a
    category error (``float(True) == 1.0`` clamps a flag to a perfect score),
    but a PREDICTION of ``True`` is one of the two decisions a binary classifier
    makes, and this library's own ``y_pred`` arrays carry it. Rejecting it would
    refuse every boolean prediction column in the library, which is the
    over-correction: the refusal must be aimed at the absent prediction only.
    """
    if pd.api.types.is_bool_dtype(values):
        # ``isna`` and not a bare 0, because pandas' NULLABLE boolean dtype holds
        # pd.NA beside True and False, and that IS an unscored row. A numpy bool
        # column cannot hold one, so this counts 0 there.
        return int(values.isna().sum())
    try:
        as_float = np.asarray(values, dtype="float64")
    except (TypeError, ValueError):
        # A column float() cannot read at all (strings, objects). The canonical
        # rule answers per value, and it refuses every one of them, which is
        # what the arithmetic below would have done by raising.
        return int(sum(1 for v in values if not is_measured(v)))
    return int(np.count_nonzero(~np.isfinite(as_float)))


def _unthresholded_scores(values: pd.Series) -> int:
    """How many rows hold a SCORE rather than one of the two decisions 0 / 1.

    BGL7 monitor-3/4, 2026-09-30. The predicate was inline inside
    :func:`_selected_share` and therefore reached only the two label-dependent
    metrics; :meth:`FairnessMonitor._group_positive_rates`, which the other two
    built-ins use, had no equivalent and published ``Series.mean()`` of a score
    column as a POSITIVE-PREDICTION RATE. Naming it here is what lets both
    families ask the same question, which is the whole point: the defect was one
    family refusing an input the other family scored.

    A row that was never scored at all is NOT counted here. That is
    :func:`_unscored_predictions`' question, the two states are different (no
    prediction exists vs a prediction exists and is not a decision), and a caller
    that conflates them cannot say which one it hit.

    A BOOLEAN column is a decision column, not a score column, for the reason
    :func:`_unscored_predictions` gives at length: this library's own ``y_pred``
    arrays carry booleans and refusing them would be the over-correction.
    """
    if pd.api.types.is_bool_dtype(values):
        return 0
    try:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        # Nothing float() can read is not a score either; it is an unscored row,
        # and _unscored_predictions counts it.
        return 0
    if not arr.size:
        return 0
    return int(np.count_nonzero(np.isfinite(arr) & ~np.isin(arr, (0.0, 1.0))))


def _unlabelled_rows(values: pd.Series) -> int:
    """How many rows carry no ground-truth label a rate can be conditioned on.

    BGL7 monitor-1/2, 2026-09-30. The sibling of :func:`_unscored_predictions`,
    aimed at the OTHER column the two label-dependent metrics read, and the one
    nobody had written. ``pos = grp[grp[label] == 1]`` is False for a NaN label
    exactly as ``(pos[pred] == 1)`` was False for a NaN prediction: the row is
    not refused, it is quietly moved out of the arm it belongs to, so the
    true-positive rate is computed over the rows whose label happens to be known
    and published as the group's. ``_missing_column_nan`` only checks that the
    label COLUMN exists, and the ``len(valid) < 2`` refusal only fires when a
    group has NO defined rate at all, so TOTAL label loss was refused and PARTIAL
    label loss passed. That is the coverage-guard-fires-only-on-zero shape.

    Measured 2026-09-30, ``min_samples=30``, group A 40 positive-label rows with
    20 selected (TPR 0.500) and group B 40 genuinely positive rows of which 20
    carry ``label = NaN`` and are selected, so B's real TPR is 30/40 = 0.750:

        compute_equal_opportunity -> 0.0   warnings []
        compute_equalized_odds    -> 0.0   warnings []   (both arms holding two
                                                          DEFINED rates, which is
                                                          why counting defined
                                                          rates could never catch it)

    against a real gap of 0.250 on both. After: nan on both, with a warning
    naming the group and counting its unlabelled rows.

    Anything that is not a measured 0 or 1 counts, not only a NaN.
    ``FairnessMonitorConfig.label_col`` documents "binary 0/1", and a row
    labelled 2 or "yes" lands in neither ``grp[label] == 1`` nor
    ``grp[label] == 0``, so it is dropped from BOTH arms by the same mechanism.
    This deliberately parts from ``prediction_col``, which documents "binary 0/1
    or scores" and therefore needs the separate :func:`_unthresholded_scores`
    reading rather than this one.

    A BOOLEAN label column is a label column: ``True`` IS the positive class and
    ``grp[label] == 1`` matches it. Only a NULLABLE boolean's ``pd.NA`` is an
    absent label there.
    """
    if pd.api.types.is_bool_dtype(values):
        return int(values.isna().sum())
    try:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        # A column float() cannot read at all holds no binary label in any row.
        return int(len(values))
    if not arr.size:
        return 0
    return int(np.count_nonzero(~np.isin(arr, (0.0, 1.0))))


def _selected_share(values: pd.Series) -> float:
    """Share of rows predicted positive, or nan when any row was never scored.

    See :func:`_unscored_predictions` for the measured before and after. A rate
    over the scored subset of a group is not that group's rate: it is a rate for
    a population nobody chose, and it was being published as the group's.

    THE UNSCORED REFUSAL IS DEFENCE IN DEPTH, not load-bearing today, and that is
    stated here because an unsabotaged guard is indistinguishable from one that
    cannot fail. Measured 2026-09-27: replacing the nan branch below with
    ``if False`` left ``tests/test_bgl5_operations_2.py`` and
    ``tests/test_bgl4_operations_2.py`` fully GREEN, 53 passed, because both
    callers raise their own refusal ABOVE the dispatch that would use this rate,
    so the wrong number is never reached. It is kept because the rate itself must
    be honest for any future caller, and because the day someone moves that guard
    below the dispatch, this is what stops a 0.0 being published.

    THE SCORE-COLUMN REFUSAL BECAME LOAD-BEARING ON 2026-09-30 (BGL7 monitor-3/4)
    and the sentence above does not cover it.
    :meth:`FairnessMonitor._group_positive_rates` now routes through this
    function, so this line is the only thing that stops a mean SCORE being
    published as a positive-prediction rate by ``compute_disparate_impact`` and
    ``compute_demographic_parity`` as well. Sabotaging it turns tests red: see
    the sabotage record in ``tests/test_bgl7_monitor.py``.
    """
    if _unscored_predictions(values) > 0:
        return float("nan")
    # NOT BINARY IS NOT ZERO (BGL6 F03, 2026-09-28). `(values == 1).mean()` counts a
    # row as selected only when it holds exactly 1, and
    # FairnessMonitorConfig.prediction_col documents a SCORE column as allowed
    # ("binary 0/1 or scores"). A score column therefore contributed no selected row
    # at all and this returned 0.0, which is a share, not a refusal. Both
    # true-positive rates were then 0.0 and the spread between them exactly 0.0, so
    # compute_equal_opportunity and compute_equalized_odds published PERFECT PARITY
    # for a maximally discriminating model.
    #
    # Measured 2026-09-28 on 80 rows per group, every row scored, group A's
    # positive-label rows all scoring 0.9 and group B's all scoring 0.1, which is a
    # maximal TPR gap at any threshold in (0.1, 0.9]:
    #
    #     compute_equal_opportunity  -> 0.0    warnings []
    #     compute_equalized_odds     -> 0.0    warnings []
    #     compute_disparate_impact   -> 0.111  (same frame, no warnings)
    #     compute_demographic_parity -> 0.800  (same frame, no warnings)
    #
    # Four built-ins disagreeing about one frame, and the two answering 0.0 are the
    # ones that read like a clean bill of health.
    #
    # A share of SELECTED rows is undefined for a score column until someone names a
    # decision threshold, and this function is given none, so it refuses.
    #
    # THE CARVE-OUT THIS COMMENT USED TO CARRY WAS FALSE (BGL7 monitor-3/4,
    # 2026-09-30). It read "A caller that wants the mean of the column has
    # compute_disparate_impact, which documents reading it as a rate", and
    # compute_disparate_impact's docstring says no such thing. Its own words are
    # "min(group rate) / max(group rate)" and "any group's POSITIVE rate", and the
    # string "mean" and the string "score" appear nowhere in it. So the sentence
    # pointed a reader at a second method as the sanctioned home for a mean score
    # while that method documented the same positive rate that is undefined here,
    # and computed it with Series.mean() anyway. The reading was not sanctioned
    # anywhere; it was only unrefused in one of the two places. _group_positive_rates
    # now routes through this function, so neither family scores a bare score column.
    if _unthresholded_scores(values) > 0:
        return float("nan")
    # THE TWO PREDICATES ABOVE READ THIS COLUMN THROUGH A NUMERIC COERCION AND THE
    # RATE READ IT RAW (BGL8 monitor-1/2/3/4, 2026-09-30), so for an object column
    # the guard and the measurement disagreed about the same values.
    # ``_unscored_predictions`` asks ``np.asarray(values, dtype="float64")`` and
    # ``_unthresholded_scores`` asks ``pd.to_numeric``, and both parse the string
    # "1" to 1.0 and answer, correctly, "every row holds a 0/1 decision". Then
    # ``(values == 1).mean()`` compared the ORIGINAL objects, where ``"1" == 1`` is
    # False, so EVERY row of a text-typed decision column counted as a row the model
    # declined and the share came out exactly 0.0. A column read from a CSV or a
    # database without a dtype is the ordinary way to get here, and
    # ``FairnessMonitorConfig.prediction_col`` documents only "binary 0/1 or
    # scores", never a dtype.
    #
    # Measured 2026-09-30, min_samples=30, 80 rows per group, predictions stored as
    # the strings "1" and "0", group A selecting EVERY one of its rows and group B
    # selecting NONE, which is maximal disparity on every one of the four built-ins:
    #
    #     compute_equal_opportunity  -> 0.0   warnings []   (truth 1.0)
    #     compute_equalized_odds     -> 0.0   warnings []   (truth 1.0)
    #     compute_demographic_parity -> 0.0   warnings []   (truth 1.0)
    #     compute_disparate_impact   -> nan                 (truth 0.0), and its
    #         reason read "no group was selected at all" while group A had selected
    #         all 80 of its rows
    #     _group_positive_rates      -> {'A': 0.0, 'B': 0.0}   (truth {1.0, 0.0})
    #     update_and_check           -> demographic_parity 0.0, alert False,
    #                                   any_alert False: a measured clean bill
    #
    # Three perfect scores and a wrong refusal reason, from data that is a total
    # denial of selection to one group. Casting the identical strings to int gives
    # 1.0 / 1.0 / 1.0 / 0.0.
    #
    # The fix is to make the share come from the SAME reading the predicates use, so
    # a column they have accepted as a decision column cannot then be scored as
    # something else. This is the divergence lesson of BGL7 monitor-3/4 one level
    # down: there it was two callers using two helpers, here it is one helper whose
    # guard and whose arithmetic looked at the column through different eyes. No rate
    # that WAS a measurement changes: for a numeric or boolean column
    # ``pd.to_numeric`` is the identity and ``(arr == 1.0)`` is the same mask as
    # ``(values == 1)``. Anything the coercion cannot read has already been refused
    # above, so ``errors="coerce"`` cannot silently mint a 0 here.
    numeric = pd.to_numeric(values, errors="coerce")
    if not len(numeric):
        return float("nan")
    return float((numeric == 1.0).mean())


# Data classes


#: The exact metric names ``FairnessMonitor.update_and_check`` branches on. It
#: tests membership with these literal strings (grep
#: ``in self.config.metrics_to_track``), so a name that is not here is computed
#: for no window and compared to no threshold. Kept beside the config that
#: names them rather than inside the method, so the validation and the dispatch
#: read one list: a second copy is how "supported" in a docstring drifts away
#: from what the code actually branches on.
MONITORABLE_METRICS = frozenset(
    {
        "disparate_impact",
        "demographic_parity",
        "equalized_odds",
        "equal_opportunity",
    }
)


@dataclass
class FairnessMonitorConfig:
    """Configuration for :class:`FairnessMonitor`.

    Attributes
    ----------
    window_size : int
        Maximum number of rows retained in the sliding window.
    alert_threshold : float
        Minimum acceptable disparate impact ratio (below this → alert).
        The "four-fifths rule" sets this to 0.8.
    min_samples : int
        Minimum group size before a metric is reported; avoids unreliable
        estimates from tiny subgroups.
    prediction_col : str
        Column name containing model predictions (binary 0/1 or scores).
    label_col : str
        Column name containing ground-truth labels (binary 0/1).
    metrics_to_track : list[str]
        Metrics computed on each ``update_and_check`` call.
        Supported: ``"disparate_impact"``, ``"demographic_parity"``,
        ``"equalized_odds"``, ``"equal_opportunity"``.
    alert_cooldown_seconds : float
        Minimum seconds between repeat alerts for the same group × metric pair.
    protected_detection_confidence : float
        Minimum confidence from :func:`detect_protected_attributes` for a column
        that is NOT ``group_``-prefixed to be monitored as a protected attribute.
        At the library default of 0.3 a column earns 0.3 from low cardinality and
        a categorical dtype alone, with no evidence of protectedness at all, so
        the floor here is 0.5: something beyond shape has to point at the column.
    """

    window_size: int = 1000
    alert_threshold: float = 0.8
    min_samples: int = 30
    prediction_col: str = "prediction"
    label_col: str = "label"
    protected_detection_confidence: float = 0.5
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "disparate_impact",
            "demographic_parity",
        ]
    )
    alert_cooldown_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """Say so at INGEST when this configuration cannot monitor anything.

        G11, 2026-09-30. Nothing validated this dataclass, so the two ways an
        operator can silently switch the whole monitor off both produced ZERO
        warnings at ingest and zero identifiable ones at reporting.

        1. A METRIC NAME THAT IS NOT ONE OF THE FOUR. ``update_and_check``
           branches on the exact strings below, so ``metrics_to_track=
           ["demografic_parity"]`` computes nothing, writes nothing to
           ``alerts``, and every window then reports ``any_alert = None``. A
           one-letter typo is indistinguishable downstream from a metric the
           window genuinely could not compute; only the NAME separates them,
           and it was nowhere. (The reporting half is now
           ``rendering.explain._fr_alert_timeline``, which names the uncompared
           metrics rather than counting them.)

        2. A THRESHOLD THE GUARDRAIL CANNOT BREACH. This one number drives two
           different comparisons: ``disparate_impact`` alerts on
           ``val < alert_threshold``, so the bound is a required MINIMUM, and
           the three difference metrics alert on
           ``abs(val) > (1 - alert_threshold)``, so the bound is a MAXIMUM of
           ``1 - alert_threshold``. Measured with ``alert_threshold=-1.0``:
           accepted in silence, and no value either comparison can ever see
           breaches it, so the monitor reports a clean window forever. The
           mirror, ``alert_threshold=5.0``, makes the difference band negative
           so every window alerts whatever the data.

           The REASON TEXT comes from the shared
           ``_metric_direction.vacuous_bound_reason``, the same function the
           deployment gate uses, so the monitor and the gate cannot drift apart
           about which bounds grade nothing.

        Warnings, not exceptions: a configuration object is not a measurement
        and refusing to construct one would break callers that set an odd bound
        on purpose. What it may not do is stay quiet.
        """
        import warnings as _warnings

        from ...evaluation.vfairness_metrics._metric_direction import (
            BoundRole,
            vacuous_bound_reason,
        )

        tracked = list(self.metrics_to_track or [])
        unknown = [m for m in tracked if m not in MONITORABLE_METRICS]
        if unknown:
            _warnings.warn(
                f"FairnessMonitorConfig.metrics_to_track names {len(unknown)} metric(s) this "
                f"monitor cannot compute: {', '.join(map(repr, unknown))}. Supported: "
                f"{', '.join(sorted(MONITORABLE_METRICS))}. Each unknown name is computed "
                f"for no window and compared to no threshold, so WindowMetrics.any_alert "
                f"reports None (could not check) rather than a clean verdict for it. Check "
                f"the spelling.",
                UserWarning,
                stacklevel=2,
            )
        if not tracked:
            _warnings.warn(
                "FairnessMonitorConfig.metrics_to_track is empty, so no fairness metric will "
                "be computed and every window's any_alert will be None (could not check). "
                "This is NOT a monitor that found nothing wrong.",
                UserWarning,
                stacklevel=2,
            )

        # The same bound in its two roles. A ratio bound of 0.0 or below and a
        # difference band of 1.0 or above are the two vacuous ends, and
        # vacuous_bound_reason answers for each in the metric's own range.
        for metric, bound, role in (
            ("disparate_impact", self.alert_threshold, BoundRole.THRESHOLD),
            ("demographic_parity", 1.0 - self.alert_threshold, BoundRole.THRESHOLD),
        ):
            if metric not in MONITORABLE_METRICS:  # pragma: no cover - defensive
                continue
            try:
                reason = vacuous_bound_reason(metric, float(bound), role)
            except (TypeError, ValueError):
                reason = None
            if reason:
                _warnings.warn(
                    f"FairnessMonitorConfig(alert_threshold={self.alert_threshold!r}) gives "
                    f"{metric} {reason}. That guardrail can never fire, so a clean window is "
                    f"not evidence of fairness. The monitor compares disparate_impact against "
                    f"alert_threshold as a minimum and every difference metric against "
                    f"1 - alert_threshold as a maximum.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.alert_threshold > 1.0:
            _warnings.warn(
                f"FairnessMonitorConfig(alert_threshold={self.alert_threshold!r}) makes the "
                f"difference band 1 - alert_threshold = {1.0 - self.alert_threshold:.4f}, which "
                f"is negative, so abs(gap) > band is True for every window including a perfectly "
                f"equal one. That is an alert nobody measured, the mirror of a guardrail that "
                f"can never fire.",
                UserWarning,
                stacklevel=2,
            )
        for name, value in (
            ("window_size", self.window_size),
            ("min_samples", self.min_samples),
        ):
            if not isinstance(value, bool) and isinstance(value, int) and value < 1:
                _warnings.warn(
                    f"FairnessMonitorConfig({name}={value!r}) is not a usable count: a "
                    f"{'window' if name == 'window_size' else 'group'} of fewer than one row "
                    f"cannot produce a measurement, so no metric will be reported.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.alert_cooldown_seconds < 0:
            _warnings.warn(
                f"FairnessMonitorConfig(alert_cooldown_seconds="
                f"{self.alert_cooldown_seconds!r}) is negative. The cooldown is a NOTIFICATION "
                f"rate limiter and never touches the reading, so a negative value notifies "
                f"every breach rather than suppressing any; it is accepted, but it was almost "
                f"certainly not intended.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class WindowMetrics:
    """Snapshot of fairness metrics for a single monitoring window.

    Attributes
    ----------
    batch_id : str
        UUID identifying this monitoring event.
    timestamp : datetime
        When the window was evaluated.
    sample_count : int
        Number of rows in the current window.
    metrics : dict[str, float]
        Computed fairness metric values keyed by ``"metric_groupcol"``.
    group_rates : dict[str, dict[str, float]]
        Positive prediction rates per group per protected attribute.
    alerts : dict[str, bool]
        Whether each metric BREACHED its threshold in this window (``True`` =
        breached). This is the record of which guardrails were APPLIED: only
        the four built-in metrics are compared to a threshold. A metric present
        in ``metrics`` but absent here (every custom metric, and every metric
        that could not be measured) was never compared to anything, and that
        absence is could-not-check, not ``False``. Readers must not default it
        to ``False``. R-2, 2026-09-09. :attr:`uncompared_metrics` names exactly
        those keys so a reader does not have to do the set difference.
    suppressed_alerts : list[str]
        Metric keys that breached in this window but are inside their
        ``alert_cooldown_seconds`` window, so a notifier should stay quiet.
        Suppression is about NOTIFYING, never about the reading: these keys are
        ``True`` in ``alerts`` and count towards ``any_alert``. R-3, 2026-09-09.
    mmd_scores : dict[str, float]
        MMD distribution-shift scores per protected attribute column.
    excluded_groups : dict[str, dict[str, int]]
        Protected-attribute column → ``{group: row_count}`` for the groups the
        ``min_samples`` floor kept OUT of every aggregate over that column.
        Empty means every group present in the window was measured. A window
        that dropped a group cannot report an aggregate over that column as a
        plain measurement, so the affected metrics are NaN and carry no alert
        entry. R-3, 2026-09-09.
    """

    batch_id: str
    timestamp: datetime
    sample_count: int
    metrics: Dict[str, float]
    group_rates: Dict[str, Dict[str, float]]
    alerts: Dict[str, bool]
    mmd_scores: Dict[str, float] = field(default_factory=dict)
    excluded_groups: Dict[str, Dict[str, int]] = field(default_factory=dict)
    suppressed_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary.

        ``any_alert`` is carried here as well as on the object. BGL g004,
        2026-09-17: this is the JSON boundary, and the three-state verdict
        stopped at it. Everything a reader of the serialised form had left was
        ``alerts``, and the one-liner it invites, ``any(d["alerts"].values())``,
        is ``False`` for the empty dict, which is precisely the collapse
        :attr:`any_alert` exists to prevent. ``None`` here is could-not-check
        and must not be read as ``False``.

        ``n_compared`` and ``uncompared_metrics`` carry the SCOPE of that verdict
        (F13, 2026-09-30). ``any_alert`` fires only when nothing at all was
        compared, so a window with one metric refused and one clean published a
        bare ``False``, and a JSON reader had no key saying that half the
        guardrails were never applied. See :attr:`uncompared_metrics` for the
        measurement and for why the verdict itself stays two-valued.
        """
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "sample_count": self.sample_count,
            "metrics": self.metrics,
            "group_rates": self.group_rates,
            "alerts": self.alerts,
            "any_alert": self.any_alert,
            # THE SCOPE OF THE LINE ABOVE (F13, 2026-09-30). ``any_alert`` answers
            # only for the metrics that WERE compared, and its False said nothing
            # about how many of this window's metrics that was. See
            # :attr:`uncompared_metrics`.
            "n_compared": len(self._comparisons),
            "uncompared_metrics": self.uncompared_metrics,
            "mmd_scores": self.mmd_scores,
            "excluded_groups": self.excluded_groups,
            "suppressed_alerts": self.suppressed_alerts,
        }

    @property
    def uncompared_metrics(self) -> List[str]:
        """The metric keys in :attr:`metrics` that no threshold was applied to.

        Empty means every metric this window computed was also compared, so
        :attr:`any_alert` answers for all of them. A non-empty list is the
        could-not-check half of a window that ALSO has a verdict, and neither
        value can be derived from the other.

        F13, 2026-09-30. ``any_alert`` is three-state over the question "did a
        comparison that ran come back clean", and it is honest about that
        question, but it was published at the JSON boundary with no statement of
        WHAT it covered. Measured 2026-09-30 on one window of 50 group-A and 50
        group-B rows in which nobody is selected, ``min_samples=30``, tracking
        both selection metrics: ``disparate_impact`` was REFUSED (nan, no alert
        entry, since the four-fifths ratio has no denominator when the privileged
        rate is 0) while ``demographic_parity`` measured 0.0 and was compared and
        came back clean::

            metrics  {'disparate_impact_group_gender': nan,
                      'demographic_parity_group_gender': 0.0}
            alerts   {'demographic_parity_group_gender': False}
            any_alert                 False
            to_dict()['any_alert']    False

        so the field a reader meets first said False for a window in which the
        four-fifths rule had never been applied at all.

        ``any_alert`` deliberately STAYS two-valued-plus-None rather than turning
        partial coverage into ``None``. Every custom metric is written to
        ``metrics`` and never to ``alerts`` by design (see
        :meth:`FairnessMonitor.update_and_check`), and an MMD-only or
        partly-unmeasurable window is ordinary, so "None whenever coverage is
        partial" would delete the False state in normal use and turn every clean
        window with one custom metric into a could-not-check. A guard that argues
        for breaking the product is the wrong guard: what was missing is the
        SCOPE, not a fourth state.

        ``self.metrics`` may hold a NaN for a metric that was computed and came
        out unmeasurable; that metric has no ``alerts`` entry either, so it
        appears here, which is correct: nothing was compared for it.
        """
        compared = self._comparisons
        return [k for k in self.metrics if k not in compared]

    @property
    def _comparisons(self) -> Dict[str, Any]:
        """The ``alerts`` entries that record a comparison that ACTUALLY RAN.

        THE GUARD IS ABOVE THE THREE CONSUMERS, not inside each of them, because
        ``any_alert``, ``n_compared`` and :attr:`uncompared_metrics` share one
        precondition (was this metric compared at all?) and a per-property test
        is one that can be forgotten. The proof is that breaking this reddens
        every one of the three.

        ``.get(key, default)`` DOES NOT FIRE WHEN THE KEY IS PRESENT HOLDING
        None, and neither does ``k not in self.alerts``: both are membership
        questions and this is a value question. Measured 2026-09-30 with
        ``metrics={"a": 0.1}, alerts={"a": None}``::

            any_alert             False      (a clean verdict)
            n_compared            1
            uncompared_metrics    []         ("everything was compared")

        i.e. a full clean bill of health over a guardrail that was never
        applied, while the two SVG renderers drawn from this same object read
        the identical window as NOT MONITORED, because both of them test
        ``name in alerts and alerts[name] is not None``
        (``adapters_monitoring.monitoring_dashboard_to_svg`` and
        ``alert_timeline_to_svg``, whose comment says "WindowMetrics documents
        that absence as could-not-check and forbids defaulting it to False").
        Two surfaces disagreeing about one window, and the one a JSON reader
        meets first was the one answering "clean".

        ``FairnessMonitor`` itself never writes a None here, it OMITS the key
        (see ``_record`` and the ``np.isnan`` guards above it). The reachable
        door is a window that was BUILT rather than computed: a stored window
        read back, a consumer's own record, a caller assembling one by hand.
        That is the shape the renderers were already written against.
        """
        raw = self.alerts if isinstance(self.alerts, dict) else {}
        return {k: v for k, v in raw.items() if v is not None}

    @property
    def any_alert(self) -> Optional[bool]:
        """Three states, never two.

        ``True``  at least one metric was compared to its threshold and breached.
        ``False`` at least one metric was compared to its threshold and every
                  such comparison came back clean.
        ``None``  NOTHING was compared to a threshold in this window, so there
                  is no verdict to give.

        ``None`` is could-not-check and must not be read as ``False``. Before
        2026-09-09 this was ``any(self.alerts.values())``, which is ``False``
        for an empty dict, so a window in which no protected column was found
        and ZERO fairness metrics were computed reported the same clean
        headline as a fully measured, fully compliant window. R-3.

        THE VERDICT IS NOT THE COVERAGE. ``True`` and ``False`` answer for the
        metrics in :attr:`alerts` and for no others, so a ``False`` beside a
        non-empty :attr:`uncompared_metrics` is "every guardrail that RAN came
        back clean", never "every guardrail came back clean". Read the two
        together; both are carried by :meth:`to_dict`. F13, 2026-09-30.
        """
        compared = self._comparisons
        if not compared:
            return None
        return any(compared.values())


class FairnessMonitor:
    """Sliding-window, real-time fairness monitor.

    Ingests prediction batches as pandas DataFrames and maintains a rolling
    window of the most recent ``window_size`` rows.  On each call to
    :meth:`update_and_check`, the monitor:

    1. Appends the new batch to the internal buffer.
    2. Trims the buffer to ``window_size`` rows.
    3. Recomputes configured fairness metrics for every protected-attribute
       column in the batch: the ``group_``-prefixed ones, plus every column
       :func:`detect_protected_attributes` identifies above
       ``config.protected_detection_confidence``.
    4. Computes MMD scores if a reference window is set.
    5. Returns a :class:`WindowMetrics` snapshot with per-metric alerts.

    A window in which no protected column was found computes nothing, warns,
    and reports ``any_alert=None``. It never reports ``False``, which is the
    verdict of a monitor that looked and found nothing wrong.

    Tiered Monitoring Role
    ----------------------
    This class implements **Tier 1 / Tier 2** monitoring:

    - *Tier 1*: Call :meth:`update_and_check` with small micro-batches
      (e.g., every 100 predictions) using a lightweight metric set such as
      ``["disparate_impact"]``.
    - *Tier 2*: Use larger batches (e.g., every 15 minutes) with the full
      metric set including ``"equalized_odds"``.

    Parameters
    ----------
    window_size : int, default 1000
        Maximum rows in the sliding window.
    alert_threshold : float, default 0.8
        Disparate-impact ratio below which an alert fires.
    config : FairnessMonitorConfig, optional
        Full configuration object; overrides scalar parameters when supplied.
    custom_metrics : dict, optional
        Mapping ``{name: callable(window_df) -> float}`` for user-defined
        metrics computed alongside the built-in ones.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from vfairness.operations.monitoring import FairnessMonitor
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "prediction": rng.integers(0, 2, 200),
    ...     "label":      rng.integers(0, 2, 200),
    ...     "group_gender": rng.choice(["M", "F"], 200),
    ... })
    >>> monitor = FairnessMonitor(window_size=500)
    >>> result = monitor.update_and_check(df)
    >>> result.metrics
    {'disparate_impact_group_gender': ..., ...}

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

    Ledger row: fairness_monitor. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold: float = 0.8,
        config: Optional[FairnessMonitorConfig] = None,
        custom_metrics: Optional[Dict[str, Callable[[pd.DataFrame], float]]] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = FairnessMonitorConfig(
                window_size=window_size,
                alert_threshold=alert_threshold,
            )

        self._buffer: List[pd.DataFrame] = []
        self._history: List[WindowMetrics] = []
        self._reference_df: Optional[pd.DataFrame] = None
        self._last_alert_time: Dict[str, datetime] = {}
        self._custom_metrics = custom_metrics or {}

    # Public API

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store a reference window for MMD distribution-shift monitoring.

        Parameters
        ----------
        df : pd.DataFrame
            Historical "clean" data whose distribution defines the baseline.
        """
        self._reference_df = df.copy()

    def update_and_check(self, batch_df: pd.DataFrame) -> WindowMetrics:
        """Ingest a new prediction batch and evaluate fairness metrics.

        Parameters
        ----------
        batch_df : pd.DataFrame
            New predictions.  Must contain at least the column defined in
            ``config.prediction_col``; label-dependent metrics additionally
            require ``config.label_col``; protected-attribute columns are
            taken from the ``"group_"`` prefix or auto-detected by
            :meth:`_detect_protected_cols`.

        Returns
        -------
        WindowMetrics
            Metrics and alert flags for the current window. ``any_alert`` is
            ``None`` when nothing was compared to a threshold.

        Warns
        -----
        UserWarning
            When no protected-attribute column could be found, so no fairness
            metric was computed at all; when the configured ``prediction_col``
            is absent from the batch, which makes every metric NaN; and when
            the configured ``label_col`` is absent while a label-dependent
            metric is being tracked. All three are could-not-check states, and
            none of them is a clean window.
        """
        self._buffer.append(batch_df)
        window_df = pd.concat(self._buffer, ignore_index=True)
        if len(window_df) > self.config.window_size:
            window_df = window_df.iloc[-self.config.window_size :]
        self._buffer = [window_df]

        protected_cols = self._detect_protected_cols(window_df)
        metrics: Dict[str, float] = {}
        group_rates: Dict[str, Dict[str, float]] = {}
        alerts: Dict[str, bool] = {}
        mmd_scores: Dict[str, float] = {}
        excluded_groups: Dict[str, Dict[str, int]] = {}
        suppressed_alerts: List[str] = []

        pred_col = self.config.prediction_col
        label_col = self.config.label_col
        has_labels = label_col in window_df.columns

        if not protected_cols:
            shown = list(window_df.columns)[:12]
            warnings.warn(
                f"FairnessMonitor.update_and_check: no protected-attribute column was "
                f"found among the {len(window_df.columns)} column(s) {shown}, so NO "
                f"fairness metric was computed for this window. The snapshot reports "
                f"any_alert=None (could not check), never False.",
                UserWarning,
                stacklevel=2,
            )

        # READINESS-5, 2026-09-10. The sibling of the warning above, and the
        # one that was missing. `prediction_col` DEFAULTS to "prediction", a
        # name most frames do not use, and when the configured column is absent
        # every metric computes to NaN: no threshold comparison is recorded, so
        # the window is honestly could-not-check but says nothing about WHY.
        #
        # Found in the library's own shipped notebook, which feeds a frame of
        # y_true / y_pred / group_gender and never sets prediction_col. All 120
        # windows produced NaN for both tracked metrics, 240 of 240, in total
        # silence, and the reporting demo built on top of that store presented
        # a health score, an executive report and a threshold simulation over a
        # monitor that had measured nothing at all. A misconfigured column name
        # is trivially detectable, so it must never be a silent void.
        if pred_col not in window_df.columns:
            shown = list(window_df.columns)[:12]
            warnings.warn(
                f"FairnessMonitor.update_and_check: the configured prediction column "
                f"{pred_col!r} is NOT among the {len(window_df.columns)} column(s) "
                f"{shown}, so every fairness metric for this window is NaN and NO "
                f"threshold comparison was made. This is could-not-check, not a clean "
                f"window. Set FairnessMonitorConfig(prediction_col=...) to the column "
                f"that actually holds the predictions.",
                UserWarning,
                stacklevel=2,
            )
        elif not has_labels and any(
            m in self.config.metrics_to_track for m in ("equalized_odds", "equal_opportunity")
        ):
            shown = list(window_df.columns)[:12]
            warnings.warn(
                f"FairnessMonitor.update_and_check: the configured label column "
                f"{label_col!r} is NOT among the {len(window_df.columns)} column(s) "
                f"{shown}, so the label-dependent metric(s) you asked to track "
                f"(equalized_odds / equal_opportunity) were NOT computed for this "
                f"window. Their absence is could-not-check, not a pass. Set "
                f"FairnessMonitorConfig(label_col=...) to the column holding the "
                f"ground truth.",
                UserWarning,
                stacklevel=2,
            )

        for col in protected_cols:
            g_rates = self._group_positive_rates(window_df, col, pred_col)
            group_rates[col] = g_rates
            dropped = self._undersampled_groups(window_df, col)
            if dropped:
                excluded_groups[col] = dropped

            if "disparate_impact" in self.config.metrics_to_track:
                key = f"disparate_impact_{col}"
                val = self.compute_disparate_impact(window_df, col, pred_col)
                metrics[key] = val
                # A NaN metric was never compared to the threshold, and the
                # previous ``... and not np.isnan(val)`` recorded that as
                # ``alerts[key] = False``: the record of a guardrail that ran
                # and passed. WindowMetrics documents an ABSENT entry as the
                # could-not-check state, so an unmeasurable metric now leaves
                # ``alerts`` entirely. R-3, 2026-09-09.
                if not np.isnan(val):
                    breached = val < self.config.alert_threshold
                    self._record(alerts, suppressed_alerts, key, breached)

            if "demographic_parity" in self.config.metrics_to_track:
                key = f"demographic_parity_{col}"
                val = self.compute_demographic_parity(window_df, col, pred_col)
                metrics[key] = val
                # Same tri-state guard as disparate_impact above.
                if not np.isnan(val):
                    breached = abs(val) > (1 - self.config.alert_threshold)
                    self._record(alerts, suppressed_alerts, key, breached)

            if has_labels:
                if "equalized_odds" in self.config.metrics_to_track:
                    key = f"equalized_odds_{col}"
                    val = self.compute_equalized_odds(window_df, col, pred_col, label_col)
                    metrics[key] = val
                    # Same tri-state guard as disparate_impact above.
                    if not np.isnan(val):
                        breached = abs(val) > (1 - self.config.alert_threshold)
                        self._record(alerts, suppressed_alerts, key, breached)

                if "equal_opportunity" in self.config.metrics_to_track:
                    key = f"equal_opportunity_{col}"
                    val = self.compute_equal_opportunity(window_df, col, pred_col, label_col)
                    metrics[key] = val
                    # Same tri-state guard as disparate_impact above.
                    if not np.isnan(val):
                        breached = abs(val) > (1 - self.config.alert_threshold)
                        self._record(alerts, suppressed_alerts, key, breached)

            # MMD distribution-shift detection
            if self._reference_df is not None and pred_col in self._reference_df.columns:
                try:
                    ref_group = self._reference_df[self._reference_df[col].notna()][
                        pred_col
                    ].values.astype(float)
                    cur_group = window_df[window_df[col].notna()][pred_col].values.astype(float)
                    if len(ref_group) >= 10 and len(cur_group) >= 10:
                        mmd_scores[col] = mmd_gaussian(ref_group, cur_group)
                except Exception:
                    # BGL g004, 2026-09-17. The absent key IS the right
                    # could-not-check answer (no shift score is invented), but
                    # discarding the exception left a reader with no way to
                    # find out WHY it is absent. Same house pattern as the MMD
                    # site in drift.py: keep the honest value, log the cause.
                    logging.getLogger(__name__).debug(
                        "MMD diagnostic failed for protected column %r; no shift score "
                        "is recorded for it (could not check), and no drift verdict is "
                        "derived from its absence",
                        col,
                        exc_info=True,
                    )

        # Custom metrics. They carry no threshold and no direction, so no
        # alert determination is possible for them: they are deliberately
        # written into ``metrics`` and NOT into ``alerts``. An absent alerts
        # entry is the could-not-check state; readers that need a verdict for
        # a custom metric must not invent one from the absence. R-2.
        for name, fn in self._custom_metrics.items():
            try:
                val = fn(window_df)
                metrics[name] = float(val)
            except Exception as exc:
                warnings.warn(f"Custom metric '{name}' raised: {exc}")

        snapshot = WindowMetrics(
            batch_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sample_count=len(window_df),
            metrics=metrics,
            group_rates=group_rates,
            alerts=alerts,
            mmd_scores=mmd_scores,
            excluded_groups=excluded_groups,
            suppressed_alerts=suppressed_alerts,
        )

        # Update last-alert timestamps. Only a breach that was actually
        # NOTIFIED restarts the cooldown; a suppressed repeat must not keep
        # pushing the window out, or a persistent breach would never be
        # notified again.
        now = datetime.now()
        for key, fired in alerts.items():
            if fired and key not in suppressed_alerts:
                self._last_alert_time[key] = now

        self._history.append(snapshot)
        return snapshot

    # Built-in fairness metrics (stateless, operate on a DataFrame)

    def compute_disparate_impact(
        self,
        df: pd.DataFrame,
        group_col: str,
        prediction_col: Optional[str] = None,
    ) -> float:
        """Disparate impact ratio: min(group rate) / max(group rate).

        The "four-fifths rule" (80% rule) flags a value below 0.8 as
        potential adverse impact under EEOC guidelines.

        Parameters
        ----------
        df : pd.DataFrame
        group_col : str
            Protected-attribute column.
        prediction_col : str, optional
            Defaults to ``config.prediction_col``.

        Returns
        -------
        float
            Ratio in [0, 1], or ``np.nan`` if not enough groups/samples, if
            the ``min_samples`` floor dropped a group from the window, or if any
            group's positive rate could not be measured: rows in it carry no
            prediction, or its predictions are SCORES rather than 0/1 decisions
            and no decision threshold is configured, so no share of rows SELECTED
            exists to compare. A mean score is not a positive-prediction rate.

        Warns
        -----
        UserWarning
            When a group was dropped, naming and counting it; when a group's
            rate is undefined, naming which of the two reasons applies (see
            :meth:`_warn_undefined_rates`); when fewer than two groups have a
            rate at all (see :meth:`_warn_too_few_rates`); and when the
            privileged rate is 0, which leaves the ratio without a denominator.
        """
        pred = prediction_col or self.config.prediction_col
        if self._warn_undersampled("compute_disparate_impact", df, group_col, pred):
            return np.nan
        rates = self._group_positive_rates(df, group_col, pred)
        if self._warn_too_few_rates("compute_disparate_impact", df, group_col, pred, rates):
            return np.nan
        # ABOVE the max/min dispatch, because that is what fabricated the
        # verdict: measured 2026-09-27 on 50 group-A rows predicted 0.9 against
        # 50 group-B rows predicted NaN, this method returned exactly 1.0 with
        # zero warnings, and it now returns nan and says which group and how many
        # of its rows were never scored.
        if self._warn_undefined_rates("compute_disparate_impact", df, group_col, pred, rates):
            return np.nan
        privileged = max(rates.values())
        # A SILENT REFUSAL IS NOT A DISCLOSURE (BGL8 monitor-3, 2026-09-30). The
        # value here was already honest, nan and never 0.0, and this method's own
        # docstring has promised a UserWarning for its refusals since the BGL7
        # wave. There was none: measured 2026-09-30 on 50 group-A and 50 group-B
        # rows in which NOBODY is selected, min_samples=30,
        # ``compute_disparate_impact`` returned nan with ZERO warnings while
        # ``compute_demographic_parity`` on the SAME frame measured 0.0 and was
        # compared to its threshold. The independent audit of get_alert_summary
        # named this line as the silent refusal that reaches that surface: three
        # such windows made 3 of 3 look fully compared while the four-fifths rule
        # had never once been applied. Same finding, and the same fix, as the
        # BGL3 wave made for compute_equal_opportunity's ``len(valid) < 2``:
        # docstring promises are not disclosures.
        if privileged == 0:
            n_groups = len(rates)
            warnings.warn(
                f"compute_disparate_impact: the highest positive-prediction rate over the "
                f"{n_groups} group(s) of '{group_col}' is 0, so NO group was selected at "
                f"all and the four-fifths ratio has no denominator: it was NOT computed "
                f"and NOT compared to the {self.config.alert_threshold} threshold. "
                f"Returning nan (could not measure), never 1.0, which reads as perfect "
                f"parity of selection. A window in which nobody is selected is not a "
                f"window in which selection was equitable: read the per-group rates and "
                f"get_metric_history(), whose per-row 'alert' carries the three states.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan
        return min(rates.values()) / privileged

    def compute_demographic_parity(
        self,
        df: pd.DataFrame,
        group_col: str,
        prediction_col: Optional[str] = None,
    ) -> float:
        """Demographic parity difference: max(rate) - min(rate).

        Returns
        -------
        float
            Non-negative difference; 0 = perfect parity. ``np.nan`` when fewer
            than two groups survive, when the ``min_samples`` floor dropped
            a group (the spread over the survivors is a LOWER BOUND on the
            spread, not the spread), or when a group's rate is undefined because
            rows in it carry no prediction or because its predictions are SCORES
            rather than 0/1 decisions and no decision threshold is configured.

        Warns
        -----
        UserWarning
            On every one of those refusals, including the fewer-than-two-groups
            one (see :meth:`_warn_too_few_rates`).
        """
        pred = prediction_col or self.config.prediction_col
        if self._warn_undersampled("compute_demographic_parity", df, group_col, pred):
            return np.nan
        rates = self._group_positive_rates(df, group_col, pred)
        # The sibling of the same silent refusal in compute_disparate_impact, and
        # shared with it rather than repeated, because a divergence between these
        # two is exactly what the BGL7 wave had to reconcile on
        # _group_positive_rates. BGL8 monitor-4, 2026-09-30.
        if self._warn_too_few_rates("compute_demographic_parity", df, group_col, pred, rates):
            return np.nan
        # Same guard, same reason, above the same dispatch. Measured 2026-09-27
        # on the frame above: 0.0, PERFECT PARITY, with zero warnings; now nan.
        if self._warn_undefined_rates("compute_demographic_parity", df, group_col, pred, rates):
            return np.nan
        return max(rates.values()) - min(rates.values())

    def compute_equalized_odds(
        self,
        df: pd.DataFrame,
        group_col: str,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> float:
        """Equalized odds difference (max difference in TPR and FPR).

        Returns
        -------
        float
            Maximum of |TPR_diff| and |FPR_diff| across groups. ``np.nan``,
            with a ``UserWarning``, when the configured prediction or label
            column is absent, when the ``min_samples`` floor dropped a group,
            when a group holds rows that were never scored, when a group holds
            rows with no usable ground-truth label (a label that is not 0 or 1 is
            dropped from BOTH arms, so each rate would cover the LABELLED rows
            only), when a group's positive- or negative-label arm holds fewer
            than ``_MIN_ARM_ROWS`` rows, so the rate computed over it moves in
            steps wider than the whole alert band (see
            :meth:`_warn_underpowered_arms`), or when either arm has fewer than
            two defined rates.
        """
        pred = prediction_col or self.config.prediction_col
        label = label_col or self.config.label_col
        if self._missing_column_nan("compute_equalized_odds", df, pred, label):
            return np.nan
        # The ``len(grp) < min_samples`` skip below is the same silent drop as
        # in _group_positive_rates: the spread over the survivors is a lower
        # bound on the disparity, so a SMALLER excluded group would report a
        # BETTER number. R-3, 2026-09-09.
        if self._warn_undersampled("compute_equalized_odds", df, group_col, pred):
            return np.nan

        groups = df[group_col].dropna().unique()
        tprs, fprs = {}, {}
        # Which groups hold rows that were never scored, and on which arm. Kept
        # apart from "this group has no row with that label at all": the second
        # is a quantity that does not exist, the first is one that exists and was
        # not measured, and only the first is a measurement failure. BGL5
        # A-operations-2, 2026-09-27.
        unscored: Dict[str, str] = {}
        # The sibling accounting for the OTHER column, on both arms. See
        # :func:`_unlabelled_rows` and the guard below. BGL7 monitor-2, 2026-09-30.
        unlabelled: Dict[str, str] = {}
        # How many rows each published rate was DIVIDED BY, per group per arm.
        # See :meth:`_warn_underpowered_arms`. F13, 2026-09-30.
        arm_rows: List[Tuple[str, int]] = []
        for g in groups:
            mask = df[group_col] == g
            grp = df[mask]
            if len(grp) < self.config.min_samples:
                continue
            # Counted BEFORE the two arms are cut, because the cut is what loses
            # these rows: a row whose label is not 0 or 1 lands in neither.
            n_unlabelled = _unlabelled_rows(grp[label])
            if n_unlabelled:
                unlabelled[str(g)] = f"{n_unlabelled} of {len(grp)} row(s)"
            pos = grp[grp[label] == 1]
            neg = grp[grp[label] == 0]
            arm_rows.append(
                (
                    f"group {g!s}, the true-positive rate over its positive-label rows",
                    int(len(pos)),
                )
            )
            arm_rows.append(
                (
                    f"group {g!s}, the false-positive rate over its negative-label rows",
                    int(len(neg)),
                )
            )
            tprs[g] = _selected_share(pos[pred]) if len(pos) > 0 else np.nan
            fprs[g] = _selected_share(neg[pred]) if len(neg) > 0 else np.nan
            missing_pos = _unscored_predictions(pos[pred])
            missing_neg = _unscored_predictions(neg[pred])
            if missing_pos or missing_neg:
                unscored[str(g)] = (
                    f"{missing_pos} of {len(pos)} positive-label row(s) and "
                    f"{missing_neg} of {len(neg)} negative-label row(s) carry no prediction"
                )

        # ABOVE the two len(valid) dispatches below, because on the input that
        # exposed this BOTH arms held two DEFINED rates, so that refusal was out
        # of reach by construction. Measured 2026-09-27 on 80 rows per group,
        # group A TPR 0.5 and FPR 0.25, group B FPR 0.25 and 20 of its 40
        # positive-label rows carrying NaN: this method returned np.float64(0.0),
        # PERFECT equalized odds, with warnings [], while the TPR over B's SCORED
        # rows was 1.000 against A's 0.500. It now returns nan with this warning.
        # (pos[pred] == 1) is False for a NaN, so an unscored row did not remove a
        # rate, it silently moved one, which is why counting DEFINED rates could
        # never catch it.
        if unscored:
            details = "; ".join(f"{g}: {why}" for g, why in sorted(unscored.items()))
            warnings.warn(
                f"compute_equalized_odds: {len(unscored)} group(s) of '{group_col}' hold "
                f"rows that were never scored, so their true- and false-positive rates "
                f"are rates over the SCORED rows only, not the group's: {details}. A row "
                f"with no prediction is not a row predicted negative. Returning nan "
                f"(could not measure), never 0.0, which reads as perfect equalized odds.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        # THE SAME DOOR ON THE LABEL COLUMN, BOTH ARMS (BGL7 monitor-2, 2026-09-30),
        # and ABOVE the two len(valid) dispatches for the same reason the unscored
        # guard is: on the input that exposed this, BOTH arms held two DEFINED rates,
        # so that refusal was out of reach by construction.
        #
        # Measured 2026-09-30, group A 40 positive-label rows with TPR 0.500 and 40
        # negative-label rows with FPR 0.250, group B 40 genuinely positive rows of
        # which 20 carry label = NaN and are selected, plus 40 negative-label rows
        # with FPR 0.250:
        #
        #     compute_equalized_odds -> 0.0   warnings []    (PERFECT equalized odds)
        #
        # against a real TPR of 30/40 = 0.750 for B and a real gap of 0.250.
        # `grp[label] == 1` is False for a NaN label, so B's TPR was computed over
        # the 20 rows whose label is known and published as the group's, exactly as
        # `(pos[pred] == 1)` did for a NaN prediction. Counting DEFINED rates could
        # never catch it: nothing was removed, a rate was silently MOVED.
        if unlabelled:
            details = "; ".join(f"{g}: {why}" for g, why in sorted(unlabelled.items()))
            warnings.warn(
                f"compute_equalized_odds: {len(unlabelled)} group(s) of '{group_col}' hold "
                f"rows with no usable ground-truth label, so neither arm can be "
                f"conditioned on the outcome: {details}. A row whose label is not 0 or 1 "
                f"is dropped from BOTH the positive and the negative arm, which makes each "
                f"rate a rate over the LABELLED rows only, not the group's. Returning nan "
                f"(could not measure), never 0.0, which reads as perfect equalized odds.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        # A FLOOR COUNTING ROWS WHERE THE QUANTITY NEEDS LABELS (F13, 2026-09-30).
        # ABOVE the two len(valid) dispatches and ABOVE the max() of the two arms,
        # for the same reason the two guards over it are: on the input that exposed
        # this BOTH arms held two DEFINED rates and BOTH groups cleared
        # min_samples, so neither of those refusals was in reach, and either arm of
        # the max() can be the underpowered one.
        #
        # Measured 2026-09-30, min_samples=30. Group A 40 rows of which EXACTLY ONE
        # carries label = 1 and it is selected, plus 39 negative-label rows none of
        # which is; group B 20 positive-label rows all selected plus 20
        # negative-label rows none of which is:
        #
        #     group A: n=40  pos-arm n=1  TPR 1.0000  neg-arm n=39  FPR 0.0000
        #     group B: n=40  pos-arm n=20 TPR 1.0000  neg-arm n=20  FPR 0.0000
        #     compute_equalized_odds -> 0.0   warnings []
        #     update_and_check       -> equalized_odds 0.0, alert False,
        #                               any_alert False, excluded_groups {}
        #
        # PERFECT equalized odds, a guardrail recorded as applied and passed, and
        # the A side of it is ONE OBSERVATION. min_samples passed the group because
        # it counts the group's 40 ROWS; the true-positive rate was divided by 1.
        if self._warn_underpowered_arms("compute_equalized_odds", group_col, arm_rows):
            return np.nan

        valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
        valid_fprs = [v for v in fprs.values() if not np.isnan(v)]
        # Equalized odds is the MAX of the two arms, so an arm that could not be
        # measured contributed 0.0 and simply lost the max. That makes the
        # answer a LOWER BOUND reported as the disparity, which is the rule
        # constraints/base.py already states in prose: "the spread over the
        # groups that happen to be measurable is a LOWER BOUND on the real
        # disparity, not the disparity". One unmeasurable arm is enough.
        if len(valid_tprs) < 2 or len(valid_fprs) < 2:
            warnings.warn(
                f"compute_equalized_odds: the TPR arm has {len(valid_tprs)} defined "
                f"rate(s) and the FPR arm {len(valid_fprs)}, and equalized odds is the "
                f"MAX of the two, so an unmeasured arm would silently lose the max and "
                f"turn the answer into a lower bound. Returning nan.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan
        tpr_diff = max(valid_tprs) - min(valid_tprs)
        fpr_diff = max(valid_fprs) - min(valid_fprs)
        return max(tpr_diff, fpr_diff)

    def compute_equal_opportunity(
        self,
        df: pd.DataFrame,
        group_col: str,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> float:
        """Equal opportunity difference (max difference in TPR across groups).

        Returns
        -------
        float
            max(TPR) - min(TPR) across groups; 0 = perfect equal opportunity.
            ``np.nan``, with a ``UserWarning``, when the configured prediction
            or label column is absent, when the ``min_samples`` floor dropped a
            group, when a group holds positive-label rows that were never
            scored, when a group holds rows with no usable ground-truth label (a
            label that is not 0 or 1 is dropped from the positive arm, so the
            rate would cover the LABELLED rows only), when a group's
            positive-label arm holds fewer than ``_MIN_ARM_ROWS`` rows, so the
            true-positive rate computed over it moves in steps wider than the
            whole alert band (see :meth:`_warn_underpowered_arms`), or when
            fewer than two groups have a defined TPR.
        """
        pred = prediction_col or self.config.prediction_col
        label = label_col or self.config.label_col
        if self._missing_column_nan("compute_equal_opportunity", df, pred, label):
            return np.nan
        # Same silent drop as compute_equalized_odds above. R-3.
        if self._warn_undersampled("compute_equal_opportunity", df, group_col, pred):
            return np.nan

        groups = df[group_col].dropna().unique()
        tprs = {}
        # See compute_equalized_odds: kept apart from "no positive label at all",
        # which is the reason the warning below already names. BGL5, 2026-09-27.
        unscored: Dict[str, str] = {}
        # The sibling accounting for the OTHER column. See :func:`_unlabelled_rows`
        # and the guard below. BGL7 monitor-1, 2026-09-30.
        unlabelled: Dict[str, str] = {}
        # How many rows each published TPR was DIVIDED BY, per group. See
        # :meth:`_warn_underpowered_arms`. F13, 2026-09-30.
        arm_rows: List[Tuple[str, int]] = []
        for g in groups:
            mask = df[group_col] == g
            grp = df[mask]
            if len(grp) < self.config.min_samples:
                continue
            # Counted BEFORE the positive arm is cut, and before the
            # no-positive-label `continue` below, because either one loses the row.
            n_unlabelled = _unlabelled_rows(grp[label])
            if n_unlabelled:
                unlabelled[str(g)] = f"{n_unlabelled} of {len(grp)} row(s)"
            pos = grp[grp[label] == 1]
            # Recorded BEFORE the `continue` below, for the same reason: the
            # count is the thing this is about. A ZERO-row arm is left to the
            # len(valid) < 2 refusal, which owns that case and names it more
            # precisely ("no positive label in this window"); this guard is
            # about an arm that EXISTS and is too thin to divide by.
            arm_rows.append(
                (
                    f"group {g!s}, the true-positive rate over its positive-label rows",
                    int(len(pos)),
                )
            )
            if len(pos) == 0:
                tprs[g] = np.nan
                continue
            missing = _unscored_predictions(pos[pred])
            if missing:
                unscored[str(g)] = f"{missing} of {len(pos)} positive-label row(s)"
            tprs[g] = _selected_share(pos[pred])

        # ABOVE the len(valid) dispatch below. Measured 2026-09-27 on 40
        # positive-label rows per group, group A genuinely 20 of 40 selected and
        # group B 20 selected with its other 20 never scored: this method returned
        # np.float64(0.0), PERFECT equal opportunity, with warnings [], because
        # (pos[pred] == 1).mean() counts a row with no prediction as a row that
        # was not selected and dragged B to exactly A's 0.5. The TPR over B's
        # scored rows is 1.0, a real gap of 0.5. It now returns nan with this
        # warning. The no-positive-label refusal below is untouched: its own pin
        # asserts exactly one warning on a frame with no unscored row.
        if unscored:
            details = "; ".join(f"{g}: {why}" for g, why in sorted(unscored.items()))
            warnings.warn(
                f"compute_equal_opportunity: {len(unscored)} group(s) of '{group_col}' "
                f"hold positive-label rows that were never scored, so the true-positive "
                f"rate would be a rate over the SCORED rows only, not the group's: "
                f"{details}. A row with no prediction is not a row predicted negative. "
                f"Returning nan (could not measure), never 0.0, which reads as perfect "
                f"equal opportunity.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        # THE SAME DOOR ON THE LABEL COLUMN (BGL7 monitor-1, 2026-09-30). The fix
        # above guards the unmeasurable PREDICTION; the unmeasurable LABEL was
        # unguarded. `pos = grp[grp[label] == 1]` is False for a NaN label exactly as
        # `(pos[pred] == 1)` was False for a NaN prediction, and _missing_column_nan
        # only checks that the label COLUMN exists. TOTAL label loss WAS refused (the
        # len(valid) < 2 door below), PARTIAL label loss passed, which is the
        # coverage-guard-fires-only-on-zero shape.
        #
        # Measured 2026-09-30, min_samples=30, group A 40 positive-label rows with 20
        # selected (TPR 0.500) and group B 40 genuinely positive rows of which 20
        # carry label = NaN and ARE selected, so B's real TPR is 30/40 = 0.750:
        #
        #     compute_equal_opportunity -> 0.0   warnings []   (PERFECT parity)
        #
        # against a real gap of 0.250. It now returns nan with this warning. ABOVE the
        # len(valid) dispatch, because both groups had a DEFINED TPR on that frame.
        if unlabelled:
            details = "; ".join(f"{g}: {why}" for g, why in sorted(unlabelled.items()))
            warnings.warn(
                f"compute_equal_opportunity: {len(unlabelled)} group(s) of '{group_col}' "
                f"hold rows with no usable ground-truth label, so the set of rows the "
                f"true-positive rate is conditioned on is not the group's positive-label "
                f"set: {details}. A row whose label is not 0 or 1 is dropped from the "
                f"positive arm, so the rate is a rate over the LABELLED rows only. "
                f"Returning nan (could not measure), never 0.0, which reads as perfect "
                f"equal opportunity.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        # THE SAME DOOR ON THE DENOMINATOR (F13, 2026-09-30), and the sibling of
        # the one in compute_equalized_odds. ABOVE the len(valid) dispatch below,
        # because on the input that exposed this BOTH groups held a DEFINED TPR
        # and both cleared min_samples.
        #
        # Measured 2026-09-30, min_samples=30. Group A 40 rows of which EXACTLY
        # ONE carries label = 1 and it is selected; group B 20 positive-label rows
        # all selected:
        #
        #     group A: n=40 pos-arm n=1  TPR 1.0000
        #     group B: n=40 pos-arm n=20 TPR 1.0000
        #     compute_equal_opportunity -> 0.0   warnings []
        #     update_and_check          -> equal_opportunity 0.0, alert False,
        #                                  any_alert False, excluded_groups {}
        #
        # PERFECT equal opportunity from a true-positive rate divided by 1, with a
        # guardrail recorded as applied and passed. min_samples let the group
        # through because it counts the group's 40 ROWS; the rate this metric
        # compares rests on the LABELLED SUBSET, which was one person.
        if self._warn_underpowered_arms("compute_equal_opportunity", group_col, arm_rows):
            return np.nan

        valid = [v for v in tprs.values() if not np.isnan(v)]
        # BGL3 operations-2, 2026-09-27. The VALUE here was already honest (nan,
        # never 0.0), and the docstring above has promised "``np.nan``, with a
        # ``UserWarning``, when ... fewer than two groups have a defined TPR"
        # since the R-3 wave. There was no warning. Measured before the fix on a
        # frame of 40 M rows (all label=1) and 40 F rows (all label=0), both well
        # over min_samples=30: ``compute_equal_opportunity`` returned nan and
        # emitted ZERO warnings, while ``compute_equalized_odds`` on the SAME
        # frame warned that its TPR arm held 1 defined rate. A bare nan tells a
        # caller that something is undefined; it does not say that the reason is
        # a group with no positive labels, which is the reading a TPR comparison
        # most needs to disclose. Docstring promises are not disclosures.
        if len(valid) < 2:
            no_positives = sorted(str(g) for g, v in tprs.items() if np.isnan(v))
            warnings.warn(
                f"compute_equal_opportunity: only {len(valid)} of {len(tprs)} group(s) of "
                f"'{group_col}' has a defined true-positive rate, below the 2 a spread "
                f"needs, so no equal-opportunity difference was measured. Group(s) with "
                f"no positive label in this window: {no_positives or 'none'}. Returning "
                f"nan (could not measure), never 0.0, which reads as perfect equal "
                f"opportunity.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan
        return max(valid) - min(valid)

    # State accessors

    def get_metric_history(self) -> pd.DataFrame:
        """Return metric time series as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``batch_id``, ``metric``, ``value``,
            ``sample_count``, ``alert``. ``alert`` has three states: ``True``
            (compared to its threshold and breached), ``False`` (compared and
            clean) and ``None`` (never compared to a threshold, which is every
            custom metric). ``None`` is could-not-check and must not be read as
            ``False``.

            ``sample_count`` was added on 2026-09-10 (READINESS-5). The window
            knows how many rows each reading came from and this frame used to
            drop it, so ``MetricsStore.ingest_from_monitor`` had nothing to put
            in ``group_size`` and left it at 0. The k-anonymity gate then read
            0 as "not below the threshold" and released the value as EXACT: a
            privacy guarantee asserted over a group size nobody had recorded.
            The count is measured here, so it is carried here.
        """
        rows = []
        for snap in self._history:
            for metric, value in snap.metrics.items():
                rows.append(
                    {
                        "timestamp": snap.timestamp,
                        "batch_id": snap.batch_id,
                        "metric": metric,
                        "value": value,
                        "sample_count": snap.sample_count,
                        # R-2. This was `.get(metric, False)`, so a custom
                        # metric that was never compared to a threshold left
                        # here as a clean, compliant reading, and MetricsStore
                        # averaged it into the health score. Measured
                        # 2026-09-09: built-ins both breaching scored 50.0;
                        # adding three maximally unfair custom metrics RAISED
                        # it to 71.4. No determination is None, not False.
                        "alert": snap.alerts.get(metric),
                    }
                )
        return pd.DataFrame(rows)

    def get_current_metrics(self) -> Dict[str, float]:
        """Return metrics from the most recent monitoring window."""
        if not self._history:
            return {}
        return self._history[-1].metrics.copy()

    def get_window_df(self) -> pd.DataFrame:
        """Return the current sliding-window DataFrame."""
        if not self._buffer:
            return pd.DataFrame()
        return self._buffer[0].copy()

    def get_alert_summary(self) -> Dict[str, int]:
        """Return the number of windows in which each metric key breached.

        A breach that a cooldown kept quiet is still a breach and is counted
        here; ``WindowMetrics.suppressed_alerts`` says which ones were not
        worth notifying about again.

        An ABSENT key is not a verdict. A metric is missing from this mapping
        both when it was compared to its threshold in every window and stayed
        clean AND when it was never compared to a threshold at all, so an empty
        mapping does NOT mean "monitored and clean": a monitor that found no
        protected column, or whose metrics were all NaN, also returns ``{}``.
        Read :meth:`get_metric_history`, whose per-row ``alert`` carries the
        three states, before concluding anything from a missing key or an empty
        result. BGL g004, 2026-09-17.

        An empty result that is could-not-check now says so in a
        ``UserWarning``, on all three doors: no window monitored at all, and a
        monitored history in which no window compared anything to a threshold
        (no protected column found, or every metric nan). An empty result from a
        history that DID compare something is a real finding and stays silent.
        BGL5 A-operations-2, 2026-09-27.

        COVERAGE IS REPORTED PER METRIC KEY AS WELL AS PER WINDOW (BGL7 monitor-5,
        2026-09-30). The doors above all count WINDOWS while this mapping is keyed
        by METRIC, so a metric never compared to its threshold in any window was
        absent from the result for the could-not-check reason while every window
        looked covered. The warning now names each metric key with the number of
        windows it was computed in and the number in which it was compared. A
        history in which every computed metric was compared in every window stays
        silent, whether or not the result is empty.

        COVERAGE IS ALSO REPORTED OVER THE REQUESTED SET (BGL8 monitor-5,
        2026-09-30). The door above reads the metric keys the windows actually
        carry, so a metric named in ``config.metrics_to_track`` that produced no
        reading in ANY window had no key here to be missing from and was invisible
        to every door. The warning now names those too. A history in which every
        requested metric was computed and compared in every window stays silent.
        """
        # DISCLOSED AT THE CALL, not only in the docstring above (2026-09-25).
        # The docstring has said since BGL g004 that an empty mapping does not
        # mean "monitored and clean", and a caveat only a reader of the source
        # can see is not a disclosure: `if not monitor.get_alert_summary()` reads
        # as a clean bill either way. The two cases ARE distinguishable, which is
        # what makes this fixable rather than merely documentable: a monitor that
        # never ran holds no history at all.
        if not self._history:
            warnings.warn(
                "FairnessMonitor.get_alert_summary: no window has been monitored, so "
                "the empty result means nothing was compared to a threshold. It is "
                "NOT a finding that no metric breached.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        # DOOR TWO AND DOOR THREE (BGL5 A-operations-2, 2026-09-27). Only the
        # never-ran door was closed, and the two doors this method's own docstring
        # names above were still silent. A window carries a threshold comparison
        # exactly when ``snap.alerts`` is non-empty, which WindowMetrics.alerts
        # and .any_alert already document as the could-not-check boundary, so both
        # doors are as distinguishable as the one that was fixed.
        #
        # Measured before this guard: a FairnessMonitor fed 120 rows of
        # prediction/label with NO protected column gave len(_history) == 1,
        # snapshot.alerts {}, any_alert None, and then get_alert_summary() -> {}
        # with warnings []; a frame whose prediction column was all NaN gave
        # metrics {'disparate_impact_group_gender': nan,
        # 'demographic_parity_group_gender': nan}, alerts {} and the same silent
        # {}. ``if not monitor.get_alert_summary()`` reads both as a clean bill.
        # After: {} with this warning, on both. A window that DID compare (both
        # groups at exactly 0.5, disparate impact 1.0, alerts
        # {'disparate_impact_group_gender': False,
        # 'demographic_parity_group_gender': False}) still returns {} in silence,
        # because "compared and nothing breached" is a real finding.
        if not any(snap.alerts for snap in self._history):
            warnings.warn(
                f"FairnessMonitor.get_alert_summary: {len(self._history)} window(s) were "
                f"monitored and nothing was compared to a threshold in any of them, so "
                f"the empty result is could-not-check. No fairness metric in this history "
                f"carries an alert determination: the protected column may be missing, or "
                f"every metric may have come out nan. It is NOT a finding that no metric "
                f"breached. Read get_metric_history(), whose per-row 'alert' carries the "
                f"three states.",
                UserWarning,
                stacklevel=2,
            )
            return {}

        # DOOR FOUR: THE PARTIAL HISTORY (BGL6 F03, 2026-09-28). The guard above is
        # `if not any(...)`, so it fires only when NOT ONE window in the history
        # carries a threshold comparison. A history in which one window compared and
        # nine could not is the same could-not-check with a smaller denominator, and
        # it was silent. Measured: one window with both groups selected at exactly
        # 0.5 (disparate impact 1.0, both alerts False) followed by nine windows
        # whose prediction column is all NaN (metrics nan, alerts {}, any_alert
        # None) gave 10 windows in history, 1 carrying any comparison, and
        # get_alert_summary() -> {} with no warnings. A ten-window fully monitored
        # clean history returns the same {} with the same silence, so
        # `if not monitor.get_alert_summary()` read a period in which 90 per cent of
        # windows were never compared as a clean bill.
        #
        # Coverage, not total absence, is the question. The count is stated whether
        # or not the summary is empty, because a summary over one window of ten is a
        # lower bound on the period either way.
        compared = sum(1 for snap in self._history if snap.alerts)

        # DOOR FIVE: THE GUARD COUNTED WINDOWS AND THE RESULT IS KEYED BY METRIC
        # (BGL7 monitor-5, 2026-09-30). Every door above asks whether a WINDOW carries
        # any comparison (`if not any(snap.alerts)`, `sum(... if snap.alerts)`), and a
        # window carries one as soon as ONE of its metrics was compared. The mapping
        # this method returns is per METRIC, so a metric that was never compared in
        # any window is missing from it for the could-not-check reason while every
        # window looks fully covered, and none of the three doors above can see it.
        #
        # Measured 2026-09-30, three windows of 40 group-A and 40 group-B rows in
        # which NOBODY is selected (both rates 0.0), min_samples=30. The privileged
        # rate is 0, so compute_disparate_impact refuses (a ratio needs a non-zero
        # denominator) while compute_demographic_parity measures 0.0 - 0.0 = 0.0 and
        # is compared to its threshold:
        #
        #     per window: metrics {disparate_impact: nan, demographic_parity: 0.0}
        #                 alerts  {demographic_parity_group_gender: False}
        #     windows carrying a comparison: 3 of 3, so doors two to four stay silent
        #     get_alert_summary() -> {}   warnings []
        #
        # A three-window fully compared clean history returns {} with the same
        # silence, byte-identical, so `if not monitor.get_alert_summary()` read three
        # windows in which the four-fifths rule was NEVER APPLIED as a clean bill on
        # it. An empty dict from a monitor reads as "nothing wrong", which is why this
        # matters as much as a fabricated number.
        #
        # Coverage per metric key, then, not only per window. A custom metric is
        # listed too and that is correct rather than noise: WindowMetrics.alerts and
        # get_metric_history both document that a custom metric NEVER reaches an alert
        # determination, so this summary genuinely says nothing about it, and R-2's
        # own finding was a reader defaulting that absence to False.
        #
        # ONE warning, not two. The window half and the metric half are the same
        # disclosure at two granularities, and a second warning would have broken
        # test_control_one_compared_window_beside_a_refused_one_stays_silent, whose
        # subject is that the disclosure fires once and only about coverage.
        appeared: Dict[str, int] = {}
        judged: Dict[str, int] = {}
        for snap in self._history:
            for key in snap.metrics:
                appeared[key] = appeared.get(key, 0) + 1
            for key in snap.alerts:
                judged[key] = judged.get(key, 0) + 1
        uncompared = sorted(k for k in appeared if judged.get(k, 0) < appeared[k])

        # DOOR SEVEN: THE COVERAGE WAS MEASURED OVER WHAT WAS PRODUCED, NOT OVER WHAT
        # WAS ASKED FOR (BGL8 monitor-5, 2026-09-30). Door five reads ``appeared``,
        # which is built from the metric keys the windows actually carry, so a metric
        # the operator put in ``config.metrics_to_track`` and that NEVER produced a
        # key in a single window has no entry to be missing from: it is invisible to
        # every door above. That is the fires-on-zero-passes-on-partial shape one
        # level up, over the REQUESTED metric set rather than over the windows or the
        # produced keys.
        #
        # Measured 2026-09-30, three windows of 50 group-A and 50 group-B rows,
        # min_samples=30, metrics_to_track holding all four built-ins and the frame
        # carrying NO label column, so equalized_odds and equal_opportunity were
        # never computed in any window:
        #
        #     appeared: disparate_impact_group_gender 3, demographic_parity_... 3
        #     compared: 3 of 3 windows, uncompared: none
        #     get_alert_summary() -> {}   warnings []
        #
        # and the CONTROL, the same frame with labels so all four are computed and
        # compared and clean, returns {} with the same silence. Byte-identical, so
        # ``if not monitor.get_alert_summary()`` read a period in which HALF the
        # requested guardrails were never applied as a clean bill on all four.
        #
        # The sharpest form has no disclosure anywhere: with
        # metrics_to_track=["disparate_impact", "equal_oportunity"] (one letter
        # missing) three windows produced ZERO warnings at ingest and ZERO here, and
        # equal opportunity was never monitored at all. A misconfigured metric name is
        # trivially detectable, which is the READINESS-5 argument for prediction_col
        # applied to the metric list.
        requested = list(self.config.metrics_to_track)
        never_computed = [
            name
            for name in requested
            if not any(k == name or k.startswith(f"{name}_") for k in appeared)
        ]

        parts: List[str] = []
        if compared < len(self._history):
            parts.append(
                f"only {compared} of {len(self._history)} window(s) carry any threshold "
                f"comparison, so this summary covers {compared} window(s) and says "
                f"nothing about the other {len(self._history) - compared}"
            )
        if uncompared:
            per_metric = "; ".join(
                f"{key} was computed in {appeared[key]} window(s) and compared to a "
                f"threshold in {judged.get(key, 0)}"
                for key in uncompared
            )
            parts.append(
                f"{len(uncompared)} monitored metric key(s) were NOT compared to a "
                f"threshold in every window they were computed in, and this mapping is "
                f"keyed by metric, so their absence from it is could-not-check rather "
                f"than clean: {per_metric}"
            )
        if never_computed:
            parts.append(
                f"{len(never_computed)} metric(s) you asked to track were NEVER computed "
                f"in any of the {len(self._history)} window(s), so this summary says "
                f"nothing whatever about them and they were never compared to a "
                f"threshold once: {sorted(never_computed)} of the requested "
                f"{sorted(requested)}. A metric that produced no reading in any window "
                f"has no key here to be missing from, so its absence is not a clean "
                f"reading; check that the column it needs is in the batch (equalized_odds "
                f"and equal_opportunity need config.label_col) and that the name is one "
                f"this monitor computes"
            )
        if parts:
            warnings.warn(
                "FairnessMonitor.get_alert_summary: "
                + "; and ".join(parts)
                + ". Where nothing was compared, no fairness metric reached an alert "
                "determination: the protected column may be missing, the group may be "
                "below min_samples, or the metric may have come out nan. An empty or "
                "quiet summary is NOT a finding that no metric breached across the "
                "period. Read get_metric_history(), whose per-row 'alert' carries the "
                "three states.",
                UserWarning,
                stacklevel=2,
            )
        counts: Dict[str, int] = {}
        for snap in self._history:
            for key, fired in snap.alerts.items():
                if fired:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def get_explanation(self, snap=None):
        """Generate educational explanations for a monitoring snapshot.

        Parameters
        ----------
        snap : WindowMetrics, optional
            A snapshot from :meth:`update_and_check`. If *None*, the most
            recent snapshot is used.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if snap is None:
            if self._history:
                snap = self._history[-1]
            else:
                from vfairness.explainer import _UNKNOWN_SEV, ExplanationReport

                # BGL3 operations-2, 2026-09-27. ``severity`` is the
                # machine-readable field a dashboard ranks, filters and colours
                # on, and "info" is what a window that WAS measured and found
                # clean gets. A monitor that has never seen a batch got the same
                # "info". Measured before the fix: ``FairnessMonitor()`` with no
                # ``update_and_check`` call at all returned severity "info",
                # byte-identical to the severity of a fully measured, fully
                # compliant window, while a window that found no protected
                # column already answered _UNKNOWN_SEV ("medium") through
                # FairnessExplainer.explain below. The stricter half was
                # hardened and the emptier half was not, which is the worse of
                # the two: nothing at all was compared here.
                return ExplanationReport(
                    title="Fairness Monitor Explanation",
                    summary=(
                        "No data has been processed yet, so NOTHING was compared to a "
                        "threshold. This is could not check, not a clean monitor."
                    ),
                    explanations=[],
                    severity=_UNKNOWN_SEV,
                )
        return FairnessExplainer.explain(snap)

    def reset(self) -> None:
        """Clear buffer and history, keeping configuration intact."""
        self._buffer = []
        self._history = []
        self._last_alert_time = {}
        self._reference_df = None

    # Private helpers

    def _detect_protected_cols(self, df: pd.DataFrame) -> List[str]:
        """Return columns that look like protected-attribute columns.

        Columns carrying the ``group_`` prefix are always taken. Every other
        column is put to :func:`detect_protected_attributes`, the library's own
        detector, and kept when it scores at or above
        ``config.protected_detection_confidence``.

        Before 2026-09-09 this was the prefix test alone, while the class
        docstring promised "every protected-attribute column in the batch" and
        :meth:`update_and_check` promised auto-detection. Measured that day on
        one batch of 2946 rows in which one group was never approved: with the
        column named ``race``, ``gender``, ``sex``, ``group`` or
        ``protected_race`` the monitor computed 0 metrics, raised 0 warnings
        and reported ``any_alert=False``; only ``group_race`` computed
        anything. R-3.
        """
        prefixed = [c for c in df.columns if c.startswith("group_")]
        detected: List[str] = []
        try:
            from vfairness.evaluation.vfairness_metrics.discovery import (
                detect_protected_attributes,
            )

            candidates = detect_protected_attributes(
                df,
                min_confidence=self.config.protected_detection_confidence,
                exclude_columns=[
                    self.config.prediction_col,
                    self.config.label_col,
                    *prefixed,
                ],
            )
            detected = [c.column for c in candidates]
        except Exception as exc:
            # Silence here would put the monitor straight back into the defect:
            # a detector that fell over looks exactly like a batch with nothing
            # protected in it.
            warnings.warn(
                f"FairnessMonitor: protected-attribute auto-detection failed ({exc!r}), so "
                f"only the {len(prefixed)} 'group_'-prefixed column(s) were monitored. Any "
                f"other protected column in this batch went unexamined, and a clean result "
                f"here is not evidence that there was nothing to find.",
                UserWarning,
                stacklevel=3,
            )
        return prefixed + [c for c in detected if c not in prefixed]

    def _group_positive_rates(
        self, df: pd.DataFrame, group_col: str, pred_col: str
    ) -> Dict[str, float]:
        """Positive-prediction rate per group (respects min_samples).

        Groups below ``min_samples`` are absent from the result. That absence
        is invisible to a caller looking only at this dict, so every aggregate
        built from it first asks :meth:`_warn_undersampled` whether anything
        was dropped.

        A group whose rows are present but NOT ALL SCORED is a different state
        and carries ``nan`` here rather than being absent: the quantity exists,
        it just was not measured. BGL5 A-operations-2, 2026-09-27: this line was
        ``float(sub[pred_col].mean())``, and pandas skips NaN, so a group of 30
        rows of which 29 carried no prediction reported the rate of the single
        scored row as the group's rate, while the ``min_samples`` floor counted
        all 30 and let it through. Measured on 50 group-A rows predicted 0.9
        against 50 group-B rows predicted NaN: ``{'A': 0.9, 'B': 0.9}`` before
        (both reductions skip the NaN) and now ``{'A': 0.9, 'B': nan}``, which
        :meth:`_warn_undefined_rates` refuses above every aggregate.

        A SCORE COLUMN HAS NO POSITIVE-PREDICTION RATE EITHER (BGL7 monitor-3/4,
        2026-09-30), and this was the last place in the module that computed one.
        The line was ``float(sub[pred_col].mean())``, the mean of the column, and
        ``FairnessMonitorConfig.prediction_col`` documents "binary 0/1 or scores",
        so for a score column that mean is a mean SCORE returned under this
        method's own name. :func:`_selected_share` has refused exactly this input
        since 2026-09-28 for the two label-dependent metrics; the refusal was never
        mirrored here, so one frame made the four built-ins disagree.

        Measured 2026-09-30 on 50 group-A rows scoring 1.0 and 0.0 half and half
        against 50 group-B rows all scoring exactly 0.5, ``min_samples=30``:

            _group_positive_rates       -> {'A': 0.5, 'B': 0.5}
            compute_disparate_impact    -> 1.0   warnings []
            compute_demographic_parity  -> 0.0   warnings []
            update_and_check            -> alerts {both False}, any_alert False
            _selected_share(B's column) -> nan   (the identical column, refused)

        A PERFECT four-fifths ratio and PERFECT parity, published as measurements,
        while at any decision threshold t in [0.5, 1) group A selects 0.5 of its
        rows and group B selects 0.0: the real ratio is 0.0 and the real gap 0.5.
        Now ``{'A': nan, 'B': nan}``, which :meth:`_warn_undefined_rates` refuses
        above every aggregate, naming the score reading rather than the unscored
        one.

        Delegating to :func:`_selected_share` rather than repeating its two
        predicates is deliberate: the defect WAS the divergence, so the two metric
        families now share one function and cannot drift apart again. For a 0/1 or
        boolean column ``(values == 1).mean()`` and ``Series.mean()`` are the same
        number, so no rate that WAS a measurement changes.
        """
        rates: Dict[str, float] = {}
        if pred_col not in df.columns or group_col not in df.columns:
            return rates
        for g, sub in df.groupby(group_col):
            if len(sub) >= self.config.min_samples:
                rates[str(g)] = _selected_share(sub[pred_col])
        return rates

    def _warn_underpowered_arms(
        self, metric: str, group_col: str, arms: List[Tuple[str, int]]
    ) -> bool:
        """Name the conditional rates divided by too few rows; ``True`` to refuse.

        A FLOOR COUNTING ROWS WHERE THE QUANTITY NEEDS LABELS (F13, 2026-09-30).
        ``min_samples`` is the only floor :meth:`compute_equalized_odds` and
        :meth:`compute_equal_opportunity` had, and it is a WHOLE-GROUP floor:
        ``if len(grp) < self.config.min_samples: continue`` counts the group's
        rows. Neither metric is a rate over the group. A true-positive rate is
        divided by the group's POSITIVE-LABEL rows and a false-positive rate by
        its NEGATIVE-LABEL rows, and those are labelled subsets that can hold one
        row inside a group of forty. So the guard and the arithmetic were counting
        two different things, and the group floor passed an arm of one.

        Measured 2026-09-30 on ``min_samples=30``, 40 rows in each of two groups,
        group A holding EXACTLY ONE positive-label row, which is selected, and
        group B holding twenty, all selected::

            compute_equal_opportunity -> 0.0   warnings []
            compute_equalized_odds    -> 0.0   warnings []
            update_and_check          -> both 0.0, both alerts False,
                                         any_alert False, excluded_groups {}

        Two readings of PERFECT equality, two guardrails recorded as applied and
        passed, and one side of both is a single observation.

        REFUSAL, not a disclosed measurement, and the choice is the point of this
        helper. An arm of n rows gives a rate that moves only in steps of 1/n, and
        the verdict :meth:`update_and_check` draws from the spread is
        ``abs(gap) > 1 - alert_threshold``, which is 0.2 by default: below
        ``_MIN_ARM_ROWS`` one row moves the gap further than the entire decision
        band, and at n = 1 the rate is exactly 0.0 or exactly 1.0 with nothing in
        between, so a gap of 0.0 between two such rates is arithmetic rather than
        evidence. There is also no honest way to carry "this rests on one row"
        inside a float: the metric's consumers read ``metrics[key]`` and
        ``alerts[key]``, and publishing the number would write
        ``alerts[key] = False``, the record of a guardrail that RAN and PASSED,
        which is the one thing :class:`WindowMetrics` documents an absent entry to
        mean instead. nan plus no alert entry is this file's established
        could-not-check channel, so the refusal routes into a state readers
        already handle rather than a fourth one.

        ``0 < n`` on purpose, and the same choice ``_undersized_rate_denominators``
        made in ``operations/cicd/gate.py`` for the same reason: an arm of ZERO
        rows makes the rate undefined, not coarse, and the ``len(valid) < 2``
        refusal below already answers for it and names it better ("no positive
        label in this window"). Two refusals on one number would be two wordings
        for one fact.

        The floor is ``_MIN_ARM_ROWS`` and deliberately NOT ``min_samples``: see
        that constant for the three over-correction controls in this repo that
        reusing ``min_samples`` here turns red, each of which asserts a real
        number this monitor has to keep reporting.
        """
        underpowered = [(desc, n) for desc, n in arms if 0 < n < _MIN_ARM_ROWS]
        if not underpowered:
            return False
        details = "; ".join(f"{desc}: {n} row(s)" for desc, n in sorted(underpowered))
        warnings.warn(
            f"{metric}: {len(underpowered)} conditional rate(s) over '{group_col}' rest on "
            f"fewer than {_MIN_ARM_ROWS} row(s) each, so they were divided by a handful of "
            f"labelled rows rather than by the group: {details}. The min_samples floor of "
            f"{self.config.min_samples} counts a group's ROWS, and these rates are computed "
            f"over the LABELLED SUBSET of it, so clearing that floor says nothing about "
            f"them. A rate over n rows moves only in steps of 1/n, which at these counts is "
            f"wider than the {1 - self.config.alert_threshold:.2f} alert band, and an arm of "
            f"one is exactly 0.0 or exactly 1.0 with no resolution between. Returning nan "
            f"(could not measure), never 0.0, which reads as perfect parity. No threshold "
            f"comparison was made for this metric: read get_metric_history(), whose per-row "
            f"'alert' carries the three states.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _warn_too_few_rates(
        self, metric: str, df: pd.DataFrame, group_col: str, pred_col: str, rates: Dict[str, float]
    ) -> bool:
        """Say why a spread needs two group rates and got fewer; ``True`` to refuse.

        BGL8 monitor-3/4, 2026-09-30. ``if len(rates) < 2: return np.nan`` was the
        one refusal in :meth:`compute_disparate_impact` and
        :meth:`compute_demographic_parity` that said NOTHING, and it is the same
        finding the BGL3 wave already fixed one method over: the value was honest
        (nan, never 0.0 or 1.0) and a bare nan tells a caller that something is
        undefined without saying that the reason is a window with one group in it,
        or a protected column that is not in the frame at all.
        :meth:`compute_equal_opportunity` has warned on its own ``len(valid) < 2``
        since 2026-09-27 with the note "docstring promises are not disclosures",
        and these two, which are the metrics ``metrics_to_track`` DEFAULTS to, did
        not. Measured 2026-09-30 on 50 rows of one single group, min_samples=30:

            compute_disparate_impact   -> nan   warnings 0
            compute_demographic_parity -> nan   warnings 0

        against ``compute_equal_opportunity``, which warns on the same shape.

        The two reasons are separated because they call for different actions: a
        frame that does not hold the column is a configuration error the caller can
        fix, and a window with one group in it is a fact about the data. Both are
        could-not-check, and neither is a comparison that came back clean.
        """
        if len(rates) >= 2:
            return False
        missing = [c for c in (group_col, pred_col) if c not in df.columns]
        if missing:
            shown = list(df.columns)[:12]
            why = (
                f"the column(s) {missing} are NOT among the {len(df.columns)} "
                f"column(s) {shown} of this frame, so no group rate could be computed "
                f"at all"
            )
        else:
            found = sorted(rates)
            why = (
                f"only {len(rates)} group(s) of '{group_col}' have a positive-prediction "
                f"rate in this window ({found or 'none'}), and a disparity is a "
                f"comparison BETWEEN groups, so there is nothing to compare it with"
            )
        warnings.warn(
            f"{metric}: {why}. Returning nan (could not measure), and no threshold "
            f"comparison was made for this window: the absence of an alert here is "
            f"could-not-check, not a clean reading. Read get_metric_history(), whose "
            f"per-row 'alert' carries the three states.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _warn_undefined_rates(
        self, metric: str, df: pd.DataFrame, group_col: str, pred_col: str, rates: Dict[str, float]
    ) -> bool:
        """Name the groups whose positive rate is undefined; ``True`` to refuse.

        BGL5 A-operations-2, 2026-09-27. Python's ``max``/``min`` are ORDER
        dependent around NaN (``max([0.9, nan])`` is 0.9 because ``nan > 0.9`` is
        False, and ``min([0.9, nan])`` is 0.9 for the mirror reason), so an
        undefined group rate was DROPPED OUT of the pair instead of refused, and
        privileged equalled the minimum. Measured on 50 group-A rows predicted
        0.9 against 50 group-B rows predicted NaN, ``min_samples=30``:
        ``compute_disparate_impact`` 1.0 and ``compute_demographic_parity`` 0.0,
        warnings ``[]``, ``update_and_check`` publishing
        ``{'disparate_impact_group_gender': 1.0}`` with ``alerts`` False and
        ``any_alert`` False. Putting the NaN group FIRST already answered nan, so
        the honesty rested on dict ordering. Both now answer nan with this
        warning, and the order no longer matters.

        The sibling ``operations/cicd/monitor._compute_default_metrics`` has
        refused exactly this since 2026-09-17 (``if any(r != r for r in
        dp_rates)``), and its comment names the order dependence. This is that
        guard mirrored, which is why it refuses the whole aggregate rather than
        reporting a spread over the groups that happen to be measurable: the
        latter is the LOWER BOUND :meth:`_warn_undersampled` exists to refuse.

        THE REASON IS NOW NAMED PER GROUP (BGL7 monitor-3/4, 2026-09-30). There are
        two ways a rate can fail to exist and this said "N of M row(s) carry no
        prediction" for both, so once :meth:`_group_positive_rates` started refusing
        a SCORE column the message would have read "0 of 50 row(s) carry no
        prediction" for a fully scored group: a count of zero offered as the reason
        for a refusal, which is the kind of line a reader concludes is a bug in the
        monitor rather than a finding about the data. The unscored wording is
        unchanged wherever unscored rows are what happened.
        """
        undefined = sorted(g for g, v in rates.items() if not is_measured(v))
        if not undefined:
            return False
        parts = []
        for g in undefined:
            detail = g
            if group_col in df.columns and pred_col in df.columns:
                sub = df[df[group_col].astype(str) == g]
                n_unscored = _unscored_predictions(sub[pred_col])
                n_scores = _unthresholded_scores(sub[pred_col])
                if n_unscored:
                    detail += f" ({n_unscored} of {len(sub)} row(s) carry no prediction)"
                elif n_scores:
                    detail += (
                        f" ({n_scores} of {len(sub)} row(s) hold a SCORE rather than a "
                        f"0/1 decision, and no decision threshold is configured, so the "
                        f"share of rows SELECTED does not exist for this group)"
                    )
                else:
                    detail += f" (of {len(sub)} row(s))"
            parts.append(detail)
        warnings.warn(
            f"{metric}: the positive-prediction rate of {len(undefined)} group(s) of "
            f"'{group_col}' could NOT be measured: {'; '.join(parts)}. A rate that does "
            f"not exist is dropped out of max()/min() rather than compared, which "
            f"reported a perfect 1.0 ratio and a 0.0 parity gap. Returning nan (could "
            f"not measure), not a number, not the rate over the scored rows only, and "
            f"not the mean of a score column.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _undersampled_groups(self, df: pd.DataFrame, group_col: str) -> Dict[str, int]:
        """Groups present in *df* but below ``min_samples``, with their sizes.

        Returns ``{group: row_count}``; empty when every group was measured.
        """
        if group_col not in df.columns:
            return {}
        dropped: Dict[str, int] = {}
        for g, sub in df.groupby(group_col):
            if len(sub) < self.config.min_samples:
                dropped[str(g)] = int(len(sub))
        return dropped

    def _missing_column_nan(
        self, metric: str, df: pd.DataFrame, pred_col: str, label_col: str
    ) -> bool:
        """Name the configured column(s) absent from *df*; ``True`` to refuse.

        BGL g004, 2026-09-17. Two problems lived on the one line this replaces,
        ``if label not in df.columns: return np.nan``.

        The label half refused SILENTLY, so a caller who mistyped ``label_col``
        got the same bare nan as a caller whose data genuinely had no labels.

        The prediction half was not checked AT ALL, and the two label-dependent
        metrics reach ``grp[pred]`` directly rather than through
        ``_group_positive_rates``, so a missing prediction column raised
        ``KeyError`` from inside them. Measured before the fix on a frame of
        ``y_pred`` / ``label`` / ``group_gender`` left at the default
        ``prediction_col="prediction"``: ``update_and_check`` emitted its own
        READINESS-5 warning promising "every fairness metric for this window is
        NaN" and then died with ``KeyError: 'prediction'`` on the next line,
        losing the whole snapshot including the disparate-impact and
        demographic-parity readings it had already taken. ``compute_disparate_impact``
        and ``compute_demographic_parity`` answer nan for the same frame, so
        this makes the four built-ins agree.
        """
        missing = [c for c in (pred_col, label_col) if c not in df.columns]
        if not missing:
            return False
        shown = list(df.columns)[:12]
        warnings.warn(
            f"{metric}: the configured column(s) {missing} are NOT among the "
            f"{len(df.columns)} column(s) {shown} of this frame, so neither rate this "
            f"metric compares could be computed. Returning nan (could not measure), "
            f"not a number. Set prediction_col / label_col to the columns that hold "
            f"the predictions and the ground truth.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _warn_undersampled(
        self, metric: str, df: pd.DataFrame, group_col: str, pred_col: str
    ) -> bool:
        """Name and count the groups ``min_samples`` drops; ``True`` to refuse.

        A ratio or spread computed over the SURVIVING groups is not the ratio
        or spread over the window: dropping a group can only raise the minimum
        or lower the maximum, so the reported number is a lower bound on the
        disparity presented as the disparity. Worse, it is MONOTONE the wrong
        way. Measured 2026-09-09 on 485 white / 485 black at ~0.49 plus a
        ``native`` group approved 0 of N: with min_samples=30, native n=5, 10
        and 12 all reported disparate_impact=0.9781 and alert=False, while the
        ratio over the whole window is 0.0. The smaller the victim group, the
        better the number, and at n=31 the same data reports 0.0 and alerts.

        A group with ZERO positives is the strongest reading in the window and
        the one this floor is most likely to swallow, so it is called out by
        name rather than merely counted.
        """
        # ROWS WITH NO PROTECTED ATTRIBUTE ARE A DROPPED POPULATION TOO (BGL6 F03,
        # 2026-09-28), and they were invisible here. `df.groupby(group_col)` silently
        # omits a row whose group value is missing, and _undersampled_groups groups
        # the same way, so no predicate in this class ever looked at the GROUP
        # column. Measured on 150 rows, 50 group A all selected, 50 group B all
        # selected and 50 carrying NO group value and none selected:
        #
        #     compute_disparate_impact   -> 1.0  warnings 0
        #     compute_demographic_parity -> 0.0  warnings 0
        #
        # A PERFECT four-fifths ratio over a population nobody chose, a third of the
        # window missing. Were those 50 rows a third group the ratio would be 0.0.
        # The reasoning below applies verbatim: an aggregate over the surviving rows
        # only is a LOWER BOUND on the disparity, which is why this refuses rather
        # than reporting the number with a footnote.
        if group_col in df.columns:
            n_unattributed = int(df[group_col].isna().sum())
            if n_unattributed:
                selected = ""
                if pred_col in df.columns:
                    positives = int((df[df[group_col].isna()][pred_col] == 1).sum())
                    selected = f", {positives} of {n_unattributed} predicted positive"
                    if positives == 0:
                        selected += ", NEVER selected"
                warnings.warn(
                    f"{metric}: {n_unattributed} of {len(df)} row(s) carry no value for "
                    f"'{group_col}'{selected}, and a groupby drops them, so they are in "
                    f"no group's rate and in no count. An aggregate over the rows that "
                    f"DO carry the attribute is a lower bound on the disparity: if these "
                    f"rows were a group of their own the reading could only get worse. "
                    f"Returning nan (could not measure), not a number.",
                    UserWarning,
                    stacklevel=3,
                )
                return True

        dropped = self._undersampled_groups(df, group_col)
        if not dropped:
            return False
        parts = []
        for g, n in sorted(dropped.items()):
            detail = f"{g} (n={n}"
            if pred_col in df.columns:
                positives = int((df[df[group_col].astype(str) == g][pred_col] == 1).sum())
                detail += f", {positives} of {n} predicted positive"
                if positives == 0:
                    detail += ", NEVER selected"
            parts.append(detail + ")")
        warnings.warn(
            f"{metric}: the min_samples floor of {self.config.min_samples} excludes "
            f"{len(dropped)} group(s) of '{group_col}' from the aggregate: "
            f"{'; '.join(parts)}. An aggregate over the surviving groups only is a "
            f"LOWER BOUND on the disparity, and it improves as the excluded group "
            f"gets smaller. Returning nan (could not measure), not a number.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _record(
        self,
        alerts: Dict[str, bool],
        suppressed: List[str],
        key: str,
        breached: bool,
    ) -> None:
        """Record a threshold comparison, and whether to notify on it.

        The cooldown is a NOTIFICATION rate limiter and must not touch the
        reading. Until 2026-09-09 it did: ``alerts[key] = breached and
        self._should_alert(key)`` wrote ``False`` for a metric that had just
        breached, which is the record of a guardrail that ran and passed.
        Measured that day on one batch repeated three times, group A selected
        90 percent against group B at 10 percent::

            window 1: DI=0.1111 alert=True  any_alert=True
            window 2: DI=0.1111 alert=False any_alert=False
            window 3: DI=0.1111 alert=False any_alert=False

        The disparity never moved. Only the report of it did, and it read clean
        for the whole cooldown hour. R-3.
        """
        alerts[key] = breached
        if breached and not self._should_alert(key):
            suppressed.append(key)

    def _should_alert(self, key: str) -> bool:
        """Return ``True`` if the cooldown period has elapsed for *key*.

        This decides whether a breach is NOTIFIED, never whether it happened:
        see :meth:`_record`.
        """
        if key not in self._last_alert_time:
            return True
        elapsed = (datetime.now() - self._last_alert_time[key]).total_seconds()
        return elapsed >= self.config.alert_cooldown_seconds


def _statistic_text(value: Optional[float]) -> str:
    """Render a descriptive statistic, or say it was never computed.

    ``None`` (the key is absent because the metric was never tracked) and NaN
    (the column exists but holds no usable values) are both could-not-check,
    and neither is the number ``0.0000``. R-2, 2026-09-09.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not measured"
    return f"{value:.4f}"


class TemporalFairnessAnalyzer:
    """Tracks fairness metrics as daily time-series and surfaces temporal patterns.

    Stores one metric row per day and provides:

    - **Weekly cycle detection**: Identifies days of the week where a metric
      is systematically worse than average (higher for disparity metrics;
      e.g., payday effects in credit systems, as documented in the Part 4
      case studies).
    - **Linear trend detection**: Determines whether a metric is drifting
      upward or downward over the lookback window.
    - **Forward forecasting**: Extrapolates the linear trend to predict future
      metric values.
    - **DataFrame export**: Returns the stored history as a ``pd.DataFrame``
      for downstream plotting or reporting.

    Parameters
    ----------
    lookback_days : int, default 90
        Maximum number of daily records retained.  Data older than
        ``lookback_days`` is automatically pruned on each update. Must be at
        least 1: see :meth:`__init__` for what 0 did to the row it was given.

    Raises
    ------
    ValueError
        If ``lookback_days`` is below 1 day.

    Examples
    --------
    >>> import pandas as pd
    >>> from vfairness.operations.monitoring import TemporalFairnessAnalyzer
    >>> analyzer = TemporalFairnessAnalyzer(lookback_days=30)
    >>> for i in range(14):
    ...     date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
    ...     analyzer.update_daily_metrics(date, {"demographic_parity": 0.05 + i * 0.002})
    >>> degraded, val = analyzer.detect_weekly_degradation("demographic_parity")

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

    Ledger row: temporal_fairness_analysis. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, lookback_days: int = 90) -> None:
        # A LOOKBACK THAT SWALLOWS THE ROW IT WAS JUST GIVEN (F13, 2026-09-30).
        # :meth:`update_daily_metrics` prunes with
        # ``date > date - Timedelta(days=lookback_days)``, so at 0 the cutoff IS
        # the row's own date and ``date > date`` is False: the reading handed in
        # is discarded by the same call that recorded it. Measured 2026-09-30,
        # ``TemporalFairnessAnalyzer(lookback_days=0)`` then one
        # ``update_daily_metrics(2025-03-01, {"demographic_parity": 0.9})``::
        #
        #     rows in window 0     warnings []
        #     detect_trend            -> ('not_assessed', nan)
        #     detect_weekly_degradation -> (None, nan)
        #     detect_seasonal_pattern -> {}
        #     forecast_metric         -> nan for every day
        #     get_metric_summary      -> n_days 0, every statistic nan
        #
        # Every reader was honest; the supplied history was gone. That is a
        # MEASUREMENT reported as a could-not-check, this campaign's defect
        # running backwards, and it is silent. A negative lookback does the same.
        # Raising matches :meth:`detect_seasonal_pattern`, which already refuses
        # ``period < 1`` in this class, and it fails before any reading is lost
        # rather than after. lookback_days=1 is the smallest window that keeps the
        # row it is given, so 1 is the bound.
        if int(lookback_days) < 1:
            raise ValueError(
                f"lookback_days must be at least 1 day, got {lookback_days}. At 0 or "
                f"below, update_daily_metrics prunes the very row it is recording (the "
                f"cutoff is that row's own date), so the analyzer would silently hold "
                f"nothing and every reader would report could-not-check for a history "
                f"you had supplied in full."
            )
        self.lookback_days = lookback_days
        self._daily_metrics: pd.DataFrame = pd.DataFrame()

    # Data ingestion

    def update_daily_metrics(self, date: pd.Timestamp, metrics: Dict[str, float]) -> None:
        """Append one day's worth of fairness metrics.

        Parameters
        ----------
        date : pd.Timestamp
            The calendar date of the metrics.
        metrics : dict[str, float]
            Metric name → value mapping (e.g. ``{"demographic_parity": 0.04}``).
        """
        row = {**metrics, "date": pd.Timestamp(date).normalize()}
        new_row = pd.DataFrame([row])
        self._daily_metrics = pd.concat([self._daily_metrics, new_row], ignore_index=True)
        cutoff = pd.Timestamp(date).normalize() - pd.Timedelta(days=self.lookback_days)
        self._daily_metrics = self._daily_metrics[self._daily_metrics["date"] > cutoff].copy()
        self._daily_metrics.sort_values("date", inplace=True)
        self._daily_metrics.reset_index(drop=True, inplace=True)

    # Temporal pattern analysis

    def detect_weekly_degradation(
        self,
        metric_name: str,
        degradation_threshold: float = 0.05,
        higher_is_better: Optional[bool] = None,
    ) -> Tuple[Optional[bool], float]:
        """Detect systematic within-week degradation for a metric.

        Identifies whether any weekday has a mean metric value that is more
        than ``degradation_threshold * 100``% WORSE than the overall mean.
        "Worse" is resolved from the METRIC'S OWN direction by
        :mod:`vfairness.evaluation.vfairness_metrics._metric_direction`, not
        assumed.

        Parameters
        ----------
        metric_name : str
            Column name in the stored daily metrics.
        degradation_threshold : float, default 0.05
            Relative deviation (5 %) that constitutes a degradation.
        higher_is_better : bool or None, default None
            Direction OVERRIDE for a caller that knows something the name does
            not. ``None`` (the default) resolves the direction from the metric
            name and refuses to grade when it cannot. ``False``: a weekday mean
            above ``overall * (1 + threshold)`` degrades. ``True``: a weekday
            mean below ``overall * (1 - threshold)`` degrades.

        Returns
        -------
        (degraded, worst_day_mean) : tuple[bool | None, float]
            *degraded* is ``True`` if any weekday shows systematic degradation,
            ``False`` if none does, and ``None`` when it could not be measured
            at all. ``None`` is NOT a synonym for ``False``.
            *worst_day_mean* is the mean value for the worst weekday.

        Notes
        -----
        This parameter used to default to ``False``, i.e. "every tracked metric
        is a disparity magnitude", and the two library callers
        (:meth:`get_explanation` below and
        ``rendering.adapters_monitoring.temporal_analysis_to_svg``, which loops
        over EVERY tracked metric name) both took that default. Measured on this
        repo 2026-09-10, on a ``disparate_impact_ratio`` where HIGHER is better:

        - Fridays at 0.10 against 0.30 the rest of the week returned
          ``(True, 0.30)`` and the rendered SVG read "DEGRADED / worst day mean:
          0.300". 0.300 is the HEALTHIEST weekday; the collapse to 0.10 is what
          the check was for.
        - Fridays at 1.00 (PERFECT parity) against 0.85 returned ``(True, 1.00)``
          and rendered "DEGRADED / worst day mean: 1.000": a false alarm naming
          the best possible reading as the worst day.
        - Fridays at 0.10 against 0.15 returned ``(False, 0.15)`` and rendered
          "STABLE / worst day mean: 0.150", certifying no weekday degradation for
          a week in which every day fails the four-fifths rule and one day sits
          at a ratio of 0.10.
        """
        # `False` is the verdict "no weekday shows systematic degradation", and
        # it was returned for a run with too few days to look at, or for a metric
        # that is not even in the frame. Found by execution 2026-09-08 while
        # testing the NaN guard below: this earlier return is the one that
        # actually fires, and it fires silently.
        if metric_name not in self._daily_metrics.columns:
            warnings.warn(
                f"detect_weekly_degradation: '{metric_name}' is not a tracked metric, "
                f"so nothing was examined. Returning None (could not check), not False.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")
        if len(self._daily_metrics) < 14:
            warnings.warn(
                f"detect_weekly_degradation: {len(self._daily_metrics)} day(s) of data, "
                f"below the 14 needed to compare weekdays, so degradation was not "
                f"measured. Returning None (could not check), not False.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")

        # Direction, resolved once, by the module that owns the precedence
        # table. An explicit ``higher_is_better`` is a caller who knows; None
        # means ask the resolver, and a metric the resolver cannot place is
        # could-not-check, never a guessed direction. Grading a ratio as a
        # magnitude does not merely mislabel the day, it inverts which day is
        # examined: ``weekly_means.max()`` on a four-fifths ratio is the BEST
        # weekday of the week.
        if higher_is_better is None:
            direction = metric_direction(metric_name)
            if direction is MetricDirection.UNKNOWN:
                warnings.warn(
                    f"detect_weekly_degradation: '{metric_name}' has no known "
                    f"better-direction (is a lower weekday mean better or worse?), so "
                    f"'worse than the weekly mean' could not be defined and no weekday "
                    f"was graded. Returning None (could not check), not False. Pass "
                    f"higher_is_better=True/False if you know the direction.",
                    UserWarning,
                    stacklevel=2,
                )
                return None, float("nan")
            higher_is_better = direction is MetricDirection.HIGHER_IS_BETTER

        df = self._daily_metrics.copy()
        df["weekday"] = df["date"].dt.dayofweek
        overall_mean = df[metric_name].mean()
        weekly_means = df.groupby("weekday")[metric_name].mean()
        # BGL g004, 2026-09-17. The 14-row floor above counts ROWS, and this
        # check compares WEEKDAYS. With only one weekday present, weekly_means
        # holds a single entry over exactly the rows overall_mean averages, so
        # worst_mean == overall_mean by construction and the comparison below
        # can only answer the verdict "no weekday shows systematic
        # degradation" for a week nobody observed. (For a negative overall
        # mean it answers the opposite, True, for the same non-reason.)
        # Measured before the fix on a WEEKLY reporting cadence, the realistic
        # way in: fourteen consecutive Mondays of demographic_parity at 0.02
        # with one Monday at 0.50 returned (False, 0.0543), and the SVG drew
        # the green STABLE chip because worst_day was a finite number.
        # Nothing true is lost here: with one slot no comparison between
        # weekdays exists to find anything with.
        n_weekday_slots = int(weekly_means.notna().sum())
        if n_weekday_slots < 2:
            warnings.warn(
                f"detect_weekly_degradation: the {len(df)} row(s) for '{metric_name}' "
                f"cover {n_weekday_slots} distinct weekday(s), so no weekday can be "
                f"compared against another and 'worse than the weekly mean' has no "
                f"content. Returning None (could not check), not False. A weekly "
                f"reporting cadence reaches this: it lands every row on one weekday.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")
        if higher_is_better:
            # Degradation = a weekday dropping below the weekly mean.
            worst_mean = weekly_means.min()
            degraded = worst_mean < overall_mean * (1 - degradation_threshold)
        else:
            # Disparity metrics: higher is worse, so degradation = a weekday
            # running significantly ABOVE the weekly mean. The previous
            # implementation flagged drops, i.e. it fired on the GOOD days.
            worst_mean = weekly_means.max()
            degraded = worst_mean > overall_mean * (1 + degradation_threshold)
        # Every comparison above is False when either mean is NaN, so a metric
        # that could not be averaged reported "not degraded".
        if not (np.isfinite(worst_mean) and np.isfinite(overall_mean)):
            warnings.warn(
                f"weekday degradation: the weekly mean ({overall_mean}) or the worst "
                f"weekday mean ({worst_mean}) is not finite, so degradation was not "
                f"measured. Returning None (could not check), not False.",
                UserWarning,
                stacklevel=2,
            )
            return None, float(worst_mean)
        return bool(degraded), float(worst_mean)

    def detect_trend(self, metric_name: str) -> Tuple[str, float]:
        """Estimate the linear trend direction and magnitude.

        Parameters
        ----------
        metric_name : str

        Returns
        -------
        (direction, slope) : tuple[str, float]
            *direction* is ``"increasing"``, ``"decreasing"``, ``"stable"``, or
            ``"not_assessed"``.  *slope* is the OLS slope per day, and it is
            ``nan`` exactly when the direction is ``"not_assessed"``.
            ``"not_assessed"`` is could-not-check and must not be read as
            ``"stable"``: too few rows, too few non-null values, or every
            surviving row sharing one calendar date, which leaves the time axis
            with no variance and the slope undefined.
        """
        # "stable" is a VERDICT about the trend, and it was returned for a run
        # with too few days to fit a line through. Same shape as
        # detect_weekly_degradation above.
        if metric_name not in self._daily_metrics.columns or len(self._daily_metrics) < 5:
            warnings.warn(
                f"detect_trend: '{metric_name}' has too little data to fit a trend "
                f"({len(self._daily_metrics)} row(s), 5 needed), so none was measured. "
                f"Returning 'not_assessed', not 'stable'.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed", float("nan")

        df = self._daily_metrics.dropna(subset=[metric_name]).copy()
        if len(df) < 3:
            warnings.warn(
                f"detect_trend: only {len(df)} non-null value(s) for '{metric_name}', "
                f"below the 3 needed, so no trend was measured. Returning "
                f"'not_assessed', not 'stable'.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed", float("nan")

        # Convert dates to integer days from first date
        t = (df["date"] - df["date"].iloc[0]).dt.days.values.astype(float)
        y = df[metric_name].values.astype(float)
        # OLS slope via formula
        t_mean, y_mean = t.mean(), y.mean()
        denom = np.sum((t - t_mean) ** 2)
        # BGL g004, 2026-09-17. ``else 0.0`` was an OLS slope nobody could
        # compute, handed on as the measurement 0.0 and then graded "stable".
        # ``denom`` is the total variance of the TIME axis, so it is zero
        # exactly when every surviving row shares one calendar date, which is
        # what a caller gets by calling update_daily_metrics more than once a
        # day. Measured on this repo before the fix: fourteen rows all dated
        # 2025-03-03 carrying 0.01 rising to 0.14, a fourteen-fold spread,
        # answered ("stable", 0.0); get_metric_summary published
        # trend_slope 0.0, forecast_metric drew a FLAT three-day forecast at
        # 0.14, get_explanation reported severity "info" with "Continue
        # standard monitoring.", and temporal_analysis_to_svg counted the row
        # as a fitted trend check and painted the stable badge.
        if denom <= 0:
            n_dates = int(df["date"].nunique())
            warnings.warn(
                f"detect_trend: the {len(df)} value(s) for '{metric_name}' all share "
                f"{n_dates} calendar date(s), so the time axis has no variance and the "
                f"OLS slope is undefined (0/0). Returning 'not_assessed' and nan, not "
                f"'stable' and 0.0. Record one row per day, or aggregate the "
                f"same-day rows before asking for a trend.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed", float("nan")
        slope = float(np.sum((t - t_mean) * (y - y_mean)) / denom)

        # BGL3 operations-2, 2026-09-27. THE GUARD SITS ABOVE THE DIRECTION
        # BRANCH, because every comparison in it is False for a nan slope and the
        # final ``else`` then reports "decreasing" with total confidence. For
        # every lower-is-better metric, which is most of what this analyzer
        # tracks, "decreasing" is the RECOVERING reading: the all-clear.
        #
        # Measured on this repo before the fix, 14 consecutive days of
        # demographic_parity rising 0.01 -> 0.14 with day 8 set to inf (a
        # ratio-family metric divided by a zero rate, which this library produces
        # and its own CI suite already pins elsewhere): detect_trend answered
        # ("decreasing", nan), get_metric_summary published
        # trend_direction="decreasing" beside trend_slope=nan, and
        # get_explanation reported severity "info" with "trend=decreasing" and
        # "Trend: decreasing (+nan/day)". The only notice of any of it was
        # numpy's own "invalid value encountered in subtract" RuntimeWarning,
        # which says nothing about fairness and is routinely filtered out.
        # A NaT on the date axis produced the same ("decreasing", nan) with NO
        # warning at all, because ``denom`` is nan there and ``nan <= 0`` is
        # False, so the 0/0 guard above does not fire either.
        #
        # This also restores the contract the docstring states: the slope is nan
        # exactly when the direction is "not_assessed".
        if not np.isfinite(slope):
            n_nonfinite = int((~np.isfinite(y)).sum())
            n_bad_dates = int((~np.isfinite(t)).sum())
            warnings.warn(
                f"detect_trend: the OLS slope for '{metric_name}' came out non-finite "
                f"({slope}) over {len(df)} row(s): {n_nonfinite} value(s) and "
                f"{n_bad_dates} date(s) are not finite, so no line was fitted. Returning "
                f"'not_assessed' and nan, not a direction. A direction here would be "
                f"read as a verdict, and for a lower-is-better metric the one this "
                f"arithmetic falls through to, 'decreasing', is the all-clear.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed", float("nan")

        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return direction, slope

    def detect_seasonal_pattern(
        self,
        metric_name: str,
        period: int = 7,
    ) -> Dict[int, float]:
        """Compute mean metric value per period slot (default: weekday).

        Parameters
        ----------
        metric_name : str
        period : int, default 7
            Cycle length in DAYS: consecutive calendar days advance the
            slot by one, wrapping every ``period`` days (7 = weekly).

        Returns
        -------
        dict[int, float]
            Slot index → mean metric value. For ``period=7`` the slot is
            exactly the weekday (Monday = 0). An EMPTY mapping, with a
            ``UserWarning``, is could-not-check: the metric is not tracked, or
            every stored value of it is null. A tracked metric with at least
            one value always fills at least one slot, so an empty mapping never
            means "no cycle was found".

        Raises
        ------
        ValueError
            If *period* is below 1 day.
        """
        # BGL g004, 2026-09-17. Both refusals below returned a bare ``{}``, and
        # an empty mapping is indistinguishable from a measurement: a tracked
        # metric with data ALWAYS yields at least one slot, so ``{}`` can only
        # ever mean could-not-check, and it said so to nobody.
        # ``adapters_monitoring.temporal_analysis_to_svg`` tests it with
        # ``if pattern:`` and simply draws no weekly chart, so an untracked
        # metric name and a metric with no usable values produce the same
        # silent hole in the report.
        if period < 1:
            raise ValueError(f"period must be at least 1 day, got {period}.")
        if metric_name not in self._daily_metrics.columns:
            warnings.warn(
                f"detect_seasonal_pattern: '{metric_name}' is not a tracked metric, so "
                f"no period slot was averaged and nothing was examined. Returning an "
                f"EMPTY mapping (could not check), which is not the same as a metric "
                f"with no cycle in it. Tracked: {self.get_tracked_metrics()}.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        df = self._daily_metrics.dropna(subset=[metric_name]).copy()
        if df.empty:
            # Before the fix this fell through to an arithmetic op on an empty
            # datetime column and raised TypeError, which the rendering adapter
            # swallows in a bare ``except Exception: pass``.
            warnings.warn(
                f"detect_seasonal_pattern: every one of the "
                f"{len(self._daily_metrics)} stored row(s) for '{metric_name}' is "
                f"null, so no period slot could be averaged. Returning an EMPTY "
                f"mapping (could not check), not a cycle of zeros.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        # Slot by calendar-day ordinal so ANY period forms a true repeating
        # cycle (the previous dayofweek % period broke every non-7 period:
        # weekday indices are not consecutive across week boundaries).
        # The -1 anchor keeps period=7 byte-identical to the historical
        # weekday slotting: proleptic-Gregorian day ordinal 1 is a Monday
        # and pandas dayofweek is Monday=0.
        df["slot"] = (df["date"].map(pd.Timestamp.toordinal) - 1) % period
        return df.groupby("slot")[metric_name].mean().to_dict()

    def forecast_metric(
        self,
        metric_name: str,
        days_ahead: int = 7,
    ) -> List[Tuple[pd.Timestamp, float]]:
        """Extrapolate the linear trend ``days_ahead`` into the future.

        Parameters
        ----------
        metric_name : str
        days_ahead : int, default 7

        Returns
        -------
        list[tuple[pd.Timestamp, float]]
            (date, predicted_value) pairs for each of the next *days_ahead* days.
            The value is ``nan`` for every day when :meth:`detect_trend` could
            not fit a line, or when the metric has no last observed value to
            extrapolate from. The dates are still returned, so a caller can see
            WHICH days are unknown rather than an empty list it has to guess at.
        """
        direction, slope = self.detect_trend(metric_name)
        # BGL g004, 2026-09-17. ``last_val`` fell back to the number 0.0 for a
        # metric this analyzer holds no column for and for an empty analyzer,
        # i.e. an anchor nobody observed, and the extrapolation is built on it.
        # Today that 0.0 is masked because detect_trend answers nan for both of
        # those inputs and nan + 0.0 * i is nan, which is exactly the shape this
        # campaign keeps finding one layer down: the honest output survives only
        # by accident of a sibling. The anchor is now nan in its own right.
        #
        # The ``.iloc[-1]`` also raised IndexError when every stored value for
        # the metric was null (thirty rows of nan measured before the fix), so
        # the all-null column is taken explicitly.
        last_val: float = float("nan")
        if not self._daily_metrics.empty:
            last_date = self._daily_metrics["date"].max()
            if metric_name in self._daily_metrics.columns:
                observed = self._daily_metrics.dropna(subset=[metric_name])[metric_name]
                if len(observed) > 0:
                    last_val = float(observed.iloc[-1])
                else:
                    warnings.warn(
                        f"forecast_metric: every stored value of '{metric_name}' is "
                        f"null, so there is no last observation to extrapolate from. "
                        f"The forecast is nan for all {days_ahead} day(s), not 0.0.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"forecast_metric: '{metric_name}' is not a tracked metric, so "
                    f"there is no last observation to extrapolate from. The forecast "
                    f"is nan for all {days_ahead} day(s), not 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            last_date = pd.Timestamp.now().normalize()
            warnings.warn(
                f"forecast_metric: this analyzer holds no daily metrics at all, so "
                f"nothing was extrapolated. The forecast is nan for all "
                f"{days_ahead} day(s), not 0.0.",
                UserWarning,
                stacklevel=2,
            )

        forecast = []
        for i in range(1, days_ahead + 1):
            future_date = last_date + pd.Timedelta(days=i)
            predicted = last_val + slope * i
            forecast.append((future_date, float(predicted)))
        return forecast

    # Accessors

    def to_dataframe(self) -> pd.DataFrame:
        """Return stored daily metrics as a DataFrame."""
        return self._daily_metrics.copy()

    def get_tracked_metrics(self) -> List[str]:
        """Every metric column this analyzer holds, in the order first recorded.

        Exists so a caller can cover the whole analyzer without reaching into
        ``_daily_metrics``, and so :meth:`get_explanation` can NAME the metrics
        one report does not cover.
        """
        if self._daily_metrics.empty:
            return []
        return [c for c in self._daily_metrics.columns if c != "date"]

    def get_explanation(self, metric_name: str):
        """Generate educational explanations for the temporal analysis.

        Parameters
        ----------
        metric_name : str
            The metric to explain (must have been tracked via
            :meth:`update_daily_metrics`). REQUIRED: see the note below. To
            cover everything the analyzer holds, loop over
            :meth:`get_tracked_metrics`.

        Returns
        -------
        ExplanationReport
            Its ``summary`` names how many of the tracked metrics this report
            covers, and lists by name the ones it does not.

        Notes
        -----
        This parameter used to default to ``"demographic_parity"``. The analyzer
        knows exactly which metrics it tracks, so the default silently narrowed
        the whole report to one of them and said nothing about the rest.

        Measured on this repo 2026-09-10, one analyzer, 20 days, two metrics
        with OPPOSITE directions: ``demographic_parity`` flat at 0.02 (clean,
        lower is better) and ``disparate_impact`` falling 0.95 -> 0.19 (severe,
        higher is better, ending at under a quarter of the four-fifths floor).
        ``get_explanation()`` with no argument returned "demographic_parity over
        20 days: mean=0.0200, trend=stable." at severity ``info`` with ZERO
        warnings, and never mentioned ``disparate_impact``; asked for that
        metric by name the same analyzer answered severity ``high``.

        The tell that this is a defect and not a design choice: when the
        defaulted metric was not tracked AT ALL, the same call warned twice and
        reported ``not_assessed`` correctly. The absent case had been hardened
        and the partially-covered case had not, which is the worse of the two,
        because a report that covers nothing looks empty while a report that
        covers a clean subset looks like an all-clear.

        Defaulting the metric name on a helper that then resolves a DIRECTION is
        also its own hazard, and ``tests/test_verdict_defaults.py`` sweeps the
        package for it by AST precisely so the next one cannot hide.
        """
        from vfairness.evaluation.vfairness_metrics.explainer import MetricExplanation
        from vfairness.explainer import _UNKNOWN_SEV, ExplanationReport, Severity

        summary = self.get_metric_summary(metric_name)
        direction, slope = self.detect_trend(metric_name)
        degraded, worst_val = self.detect_weekly_degradation(metric_name)

        # R-2, same defect class, one call site further on. This read
        # ``summary.get("mean", 0)``, so a metric that was never tracked was
        # reported as "Mean = 0.0000, std = 0.0000": a fabricated statistic of
        # perfect equality, printed beside an ``n_days`` that honestly read
        # "?" and beside detect_trend's own "nothing was measured" warning. A
        # statistic nobody computed is named as unmeasured, never rendered as
        # a number. 2026-09-09.
        mean_txt = _statistic_text(summary.get("mean"))
        std_txt = _statistic_text(summary.get("std"))

        # Which way is WORSE for THIS metric, resolved by the module that owns
        # the precedence table. ``None`` is could-not-check: the trend was
        # measured, but whether it is a degradation or a recovery is unknown, so
        # neither severity nor the recommendation may claim one.
        #
        # Every grading line below read ``direction == "increasing"``, i.e. it
        # assumed rising is always worse. Measured on this repo 2026-09-10 on a
        # ``disparate_impact_ratio`` (a four-fifths floor, where HIGHER is
        # better), 30 daily points either side:
        #   falling 0.98 -> 0.40 (a collapsing four-fifths ratio)
        #     severity "medium", recommendation "Continue standard monitoring."
        #   rising  0.40 -> 0.98 (the SAME model recovering)
        #     severity "high",  recommendation "Investigate the upward trend;
        #                       consider retraining or recalibrating."
        # The recovery was escalated and the collapse was waved through.
        metric_dir = metric_direction(metric_name)
        if metric_dir is MetricDirection.LOWER_IS_BETTER:
            worsening: Optional[bool] = direction == "increasing"
        elif metric_dir is MetricDirection.HIGHER_IS_BETTER:
            worsening = direction == "decreasing"
        else:
            worsening = None
        # "not_assessed" is detect_trend declining to fit at all; there is no
        # trend to call worsening or improving.
        if direction == "not_assessed":
            worsening = None

        sev: Severity = "info"
        if degraded:
            sev = "medium"
        if worsening and abs(slope) > 0.002:
            sev = "high"
        # BGL g004, 2026-09-17. ``severity`` is the machine-readable field a
        # dashboard ranks, filters and colours on, and "info" is what a metric
        # that was measured and found clean gets. A report in which NOTHING was
        # measured got the same "info": no mean, no trend, no weekday verdict.
        # Measured before the fix on an analyzer holding 30 days of
        # demographic_parity, asked about a metric it does not track at all:
        # severity "info", byte-identical to the severity of a real flat and
        # healthy series, while the prose underneath honestly read "mean=not
        # measured, trend=not_assessed". A reader sorting by severity never saw
        # the prose. FairnessMonitor.get_explanation already answers
        # _UNKNOWN_SEV for its own nothing-was-checked window; this is the same
        # floor, applied here.
        #
        # It can only RAISE, never lower a real finding: reaching this branch
        # requires degraded to be None and the trend to be unfitted, and both
        # of the escalations above are then dead, so the severity being
        # replaced is always the "info" default.
        #
        # BGL3 operations-2, 2026-09-27. The condition used to carry a third
        # term, ``not mean_measured``, and that term is what let the defect
        # survive the first fix: THE MEAN IS A DESCRIPTIVE STATISTIC, NOT A
        # VERDICT. This report grades exactly two things, the trend and the
        # weekday cycle, and when neither could be graded it has reached no
        # verdict at all, however well the mean averaged.
        #
        # Measured before this change on a 4-day analyzer of demographic_parity
        # swinging 0.02 / 0.90 / 0.03 / 0.85 (below the 5-row trend floor and the
        # 14-row weekday floor, so detect_trend answered "not_assessed" and
        # detect_weekly_degradation answered None): severity "info", BYTE
        # IDENTICAL to the severity of a measured 20-day flat healthy series,
        # because mean=0.4500 counted as "something was measured". Both
        # escalations above are dead in this state, so the prose said
        # "trend=not_assessed" while the field a dashboard sorts on said clean.
        # Four days of data is the everyday way in, not an exotic fixture.
        no_verdict_reached = direction == "not_assessed" and degraded is None
        if no_verdict_reached:
            sev = _UNKNOWN_SEV

        if worsening is None:
            trend_guide = (
                f"NOT GRADED: '{metric_name}' has no known better-direction, so a "
                f"trend that is {direction} cannot be read as degradation or as "
                f"recovery. The slope below is a measurement, not a verdict."
            )
            trend_recommendation = (
                f"Establish whether a higher {metric_name} is better or worse before "
                f"acting on this trend; it was not graded here."
            )
        else:
            worse_way = (
                "increasing" if metric_dir is MetricDirection.LOWER_IS_BETTER else "decreasing"
            )
            trend_guide = (
                f"For {metric_name}, a trend that is {worse_way} signals gradual "
                f"fairness degradation. Seasonal patterns (e.g. weekday cycles) are "
                f"normal but should be monitored."
            )
            trend_recommendation = (
                f"Investigate the {direction} trend; consider retraining or recalibrating."
                if worsening and abs(slope) > 0.001
                else "Continue standard monitoring."
            )

        if degraded:
            degradation_txt = " Weekly degradation detected."
        elif degraded is None:
            degradation_txt = " Weekly degradation could not be checked."
        else:
            degradation_txt = ""

        explanations = [
            MetricExplanation(
                metric_name=f"Temporal Trend: {metric_name}",
                definition=(
                    f"Linear trend analysis of daily {metric_name} values "
                    f"over the {summary.get('n_days', '?')} day lookback window."
                ),
                interpretation_guide=trend_guide,
                value=f"slope={slope:+.6f}/day, direction={direction}",
                evaluation=(
                    f"Trend is {direction} with slope {slope:+.6f}/day. "
                    f"Mean = {mean_txt}, "
                    f"std = {std_txt}." + degradation_txt
                ),
                benchmark_context=(
                    "Liu et al. (2023) showed that delayed impacts of fair ML "
                    "can cause gradual fairness degradation over time."
                ),
                recommendation=trend_recommendation,
                severity=sev,
                related_metrics=["weekly_degradation", "seasonal_pattern"],
            )
        ]

        # COVERAGE. One report covers ONE metric, and this analyzer commonly
        # holds several with opposite directions, so the report says which
        # subset it is. Silence here read as an all-clear over the whole
        # analyzer: an info/stable headline was produced, with no warning, by an
        # analyzer that also held a metric this method grades as high/degrading.
        tracked = self.get_tracked_metrics()
        not_covered = [m for m in tracked if m != metric_name]
        coverage_txt = ""
        if not_covered:
            coverage_txt = (
                f" Covers 1 of {len(tracked)} tracked metric(s); NOT assessed here: "
                f"{', '.join(not_covered)}."
            )

        return ExplanationReport(
            title=f"Temporal Fairness Analysis: {metric_name}",
            summary=(
                f"{metric_name} over {summary.get('n_days', '?')} days: "
                f"mean={mean_txt}, trend={direction}." + coverage_txt
            ),
            explanations=explanations,
            severity=sev,
            recommendations=[
                f"Trend: {direction} ({slope:+.6f}/day).",
            ]
            + (
                [
                    f"This report covers {metric_name} only. Call get_explanation "
                    f"for each of get_tracked_metrics() to cover the rest: "
                    f"{', '.join(not_covered)}."
                ]
                if not_covered
                else []
            )
            + (["Weekly degradation detected: investigate weekday patterns."] if degraded else [])
            # Could-not-check says so; it does not fall silent, because silence
            # here reads as "checked and clean".
            + (
                [
                    "Weekly degradation was NOT checked for this metric "
                    "(see the warning on detect_weekly_degradation)."
                ]
                if degraded is None
                else []
            ),
        )

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Return descriptive statistics for *metric_name*.

        Returns
        -------
        dict with keys: ``mean``, ``std``, ``min``, ``max``,
        ``trend_direction``, ``trend_slope``, ``n_days``.

        An EMPTY dict is could-not-check: this analyzer holds no column for
        *metric_name*, so no statistic was computed. It is not a metric whose
        statistics are all zero, and a caller must not fill the missing keys
        from a ``.get(key, 0)`` default: that is the exact defect R-2 fixed in
        :meth:`get_explanation` on 2026-09-09, where an untracked metric was
        reported as "Mean = 0.0000, std = 0.0000". When the column exists but
        every value in it is null, the keys ARE present and hold ``nan`` with
        ``n_days`` 0, which is the same message said a different way.
        """
        if metric_name not in self._daily_metrics.columns:
            # Disclosed at the call as well as in the docstring (2026-09-25). An
            # empty dict here is could-not-check, and R-2 already fixed a caller
            # that filled the missing keys from a .get(key, 0) default and
            # reported "Mean = 0.0000, std = 0.0000" for an untracked metric.
            # Saying so only in the docstring leaves the next caller to make the
            # same mistake.
            warnings.warn(
                f"TemporalFairnessAnalyzer.get_metric_summary: no column for "
                f"{metric_name!r}, so no statistic was computed. The empty result is "
                "could not check, NOT a metric whose statistics are zero.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        vals = self._daily_metrics[metric_name].dropna()
        direction, slope = self.detect_trend(metric_name)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "trend_direction": direction,
            "trend_slope": slope,
            "n_days": int(len(vals)),
        }
