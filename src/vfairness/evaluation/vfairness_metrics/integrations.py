"""
MLOps and CI/CD integration tools for vfairness.

This module provides helper functions to integrate fairness checks into
MLOps pipelines and automated testing workflows.

Features:
    - MLflow logging helper for fairness metrics
    - Weights & Biases (W&B) logging helper for fairness metrics
    - Auto-logging decorator for transparent experiment tracking
    - pytest assertion function for CI/CD pipelines
    - Integration with common ML workflow tools

Example (MLflow):
    >>> import mlflow
    >>> from vfairness import FairnessAnalyzer, log_fairness_to_mlflow
    >>>
    >>> with mlflow.start_run():
    ...     analyzer = FairnessAnalyzer(y_true, y_pred, gender)
    ...     log_fairness_to_mlflow(analyzer)

Example (W&B):
    >>> import wandb
    >>> from vfairness import FairnessAnalyzer, log_fairness_to_wandb
    >>>
    >>> wandb.init(project="my-model")
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
    >>> log_fairness_to_wandb(analyzer)

Example (Auto-logging):
    >>> from vfairness import auto_log_fairness
    >>>
    >>> @auto_log_fairness(backend="mlflow")
    ... def train_model(X, y, sensitive_attr):
    ...     model.fit(X, y)
    ...     return model.predict(X), y, sensitive_attr

Example (pytest):
    >>> from vfairness import assert_fairness
    >>>
    >>> def test_model_fairness():
    ...     assert_fairness(
    ...         y_true, y_pred, gender,
    ...         metrics=['demographic_parity_difference'],
    ...         thresholds={'demographic_parity_difference': 0.1}
    ...     )
"""

import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional

import numpy as np

from ..._triage import is_measured
from ._metric_direction import ThresholdOutcome, check_threshold
from ._validation import ArrayLike


def _is_measured(value: Any) -> bool:
    """True when ``value`` is a real number that was actually measured.

    Rejects None, non-numeric values, NaN and infinity. Used before writing a
    number to an experiment tracker: once a value is on a dashboard nothing
    distinguishes a fabricated or unmeasurable one from a measurement, so an
    unmeasurable value is written as nothing at all rather than as a number.

    READINESS-6, 2026-09-10. Delegated to ``_triage.is_measured``, which is the
    module whose docstring says it exists so this question is answered once.
    Before this there were six implementations of it and they disagreed on FIVE
    of nine inputs, including the canonical one classifying a finite
    ``np.float32`` as NOT measured. The local reasoning above is why THIS call
    site cares; the rule itself lives in one place.
    """
    return is_measured(value)


class _SizeGateExclusions(NamedTuple):
    """Protected levels ``min_group_size`` removed from EVERY metric of a run.

    Both gates in this module, ``assert_fairness`` and the
    ``fail_on_violation`` arm of ``create_fairness_callback``, need the same
    facts about a size-gate drop and must phrase them identically. They are
    computed once here so the two cannot end up disagreeing about one run, which
    is exactly what this library's two gate surfaces already did once:
    ModelFairnessGate.evaluate blocked a run that assert_fairness passed.
    """

    groups: Dict[str, int]
    n_rows: int
    n_rows_total: int
    share_percent: float
    preview: str
    covered: List[str]


def _size_gate_exclusions(analyzer: Any) -> _SizeGateExclusions:
    """Which protected levels the size gate removed, and how much data they hold.

    ``analyzer.groups`` is the VALID list (size gate applied) and
    ``analyzer.group_sizes`` carries every level with its n, so the difference is
    exactly what no metric on this run covered.

    The share of rows travels with the group names on purpose. "38 group(s)
    excluded" reads like a rounding detail, while "1102 of 2302 rows (47.9
    percent)" does not, and the share is the fact that decides whether a verdict
    computed over the survivors means anything at all.
    """
    covered = sorted(analyzer.groups)
    covered_set = set(covered)
    excluded = {
        name: int(size) for name, size in analyzer.group_sizes.items() if name not in covered_set
    }
    n_rows = sum(excluded.values())
    n_rows_total = int(analyzer.n_samples)
    preview = ", ".join(f"{name}={size}" for name, size in sorted(excluded.items())[:8])
    if len(excluded) > 8:
        preview += f", and {len(excluded) - 8} more"
    return _SizeGateExclusions(
        groups=excluded,
        n_rows=n_rows,
        n_rows_total=n_rows_total,
        share_percent=(100.0 * n_rows / n_rows_total) if n_rows_total else 0.0,
        preview=preview,
        covered=covered,
    )


def _excluded_levels_note(data_info: Dict[str, Any]) -> Dict[str, Any]:
    """What a tracker must record about levels the size gate removed.

    ``data_info['valid_groups']`` is logged as the run's ``groups`` on both
    trackers, and it is the SURVIVORS. A dashboard therefore showed
    ``groups = ['A', 'B']`` beside ``fairness_score = 1.0`` for a run where 38
    other levels, 47.9 percent of the rows, sat outside every metric: a reader
    could not tell that from a run with two levels and nothing omitted.

    Emitted on EVERY run, never only when something was excluded, for the same
    reason the report's provenance clause is: if the field appeared only on
    exclusion, its absence would be ambiguous between "nothing was excluded" and
    "this version did not record it", which is the absence-of-evidence reading
    disclosure exists to remove. A count that is genuinely unknown is recorded
    as ``"unknown"``, never as 0.
    """
    excluded = data_info.get("invalid_groups")
    sizes = data_info.get("group_sizes") or {}
    if excluded is None:
        return {"n_groups_excluded": "unknown", "groups_excluded": "unknown"}
    names = list(excluded)
    n_rows = sum(int(sizes[name]) for name in names if name in sizes)
    return {
        "n_groups_excluded": len(names),
        "groups_excluded": str(sorted(names)),
        "n_rows_excluded_by_min_group_size": n_rows if sizes else "unknown",
    }


def _logged_as_metric(value: Any) -> bool:
    """Whether this value is written to a tracker as a metric series.

    ONE expression, read by the write AND by the disclosure beside it, so the two
    can never disagree about which of a run's metrics were actually measured. It
    is deliberately NOT ``_is_measured``: that one also rejects infinity, and an
    infinite ratio is a real, measured finding (one arm received no positive
    outcomes at all), which the explainer refuses to reclassify as
    could-not-check for the same reason.

    BGL-5 (2026-09-27). ``isinstance(value, (int, float))`` is False for
    ``np.float32`` and for ``np.int64``, which are the types a numpy-backed
    report carries, so a REAL measurement was dropped from the tracker AND named
    in the disclosure as unmeasured: a false could-not-check, the fabricated
    verdict running backwards. Measured on a report whose
    demographic_parity_difference was ``np.float32(0.42)`` and whose group size
    was ``np.int64(150)``:
      before: neither logged, and
              metrics_not_measured     = ['demographic_parity_difference']
              group_stats_not_measured = ['M.size']
      after:  both logged (0.42 and 150.0), and both lists empty.
    ``np.float64`` is a subclass of ``float`` and always passed, which is why a
    real report hid this. A bool is refused by name for the reason the explainer
    gives: a flag is not a measurement, and ``isinstance(True, int)`` is True, so
    without this clause ``True`` was logged to a panel as the number 1.0.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return not np.isnan(value)


# The interval fields the trackers write, paired with the key suffix each one is
# written under. One list, read by both loggers' disclosure, so the names in the
# note and the keys on the panel cannot drift apart.
_CI_SERIES = (
    ("point_estimate", "value"),
    ("lower_bound", "ci_lower"),
    ("upper_bound", "ci_upper"),
    ("standard_error", "std_error"),
)

# The effect-size series the trackers write, paired with the key suffix each one
# is written under and with the shape its producer returns. One list, read by
# BOTH writers' write loops AND by the disclosure, so a series named as
# unmeasured is exactly a series not written.
#
# THIS IS THE SIBLING OF _CI_SERIES AND IT WAS THE DOOR LEFT OPEN. Before this
# list existed, both writers and the note hardcoded the same two fields,
# cohens_d_positive_rate and risk_ratio, while the real producer
# ``classification.compute_effect_sizes`` returns FOUR effect sizes per pair.
# Measured 2026-09-30 by feeding that producer's own output for 200 rows straight
# into both writers with fake tracker modules:
#
#   odds_ratio            = (1.1748251748251748, 0.6733467628673641, 2.049782173935953)
#   cohens_h_positive_rate = 0.08027920804079902
#   MLflow effect keys written = ['...A_vs_B.cohens_d', '...A_vs_B.risk_ratio']
#   fairness.n_effect_sizes_not_measured = 0
#   fairness.effect_sizes_not_measured   = '[]'
#   warnings                             = []
#
# So two fully measured effect sizes, one of them with its confidence interval,
# reached the tracker as nothing while every disclosure field on the run read
# clean. Same on the W&B path. The write and the name now come off one list.
#
# "interval" means the producer returns ``(value, lower, upper)`` and only the
# point value is written, which is what the risk ratio already did. The
# descriptive companions in the same dict (group1_positive_rate,
# group2_positive_rate, group1_size, group2_size) are deliberately NOT here: they
# are per-group statistics, not effect sizes, and the group-stats loop writes
# them under ``group.<name>.<stat>`` already. ``interpretation`` is prose.
_EFFECT_SERIES = (
    ("cohens_d_positive_rate", "cohens_d", "scalar"),
    ("cohens_h_positive_rate", "cohens_h", "scalar"),
    ("risk_ratio", "risk_ratio", "interval"),
    ("odds_ratio", "odds_ratio", "interval"),
)


def _effect_point_value(raw: Any, shape: str) -> Any:
    """The single number one effect-size series contributes, or ``None``.

    Read by both writers and by the disclosure, so the three cannot disagree
    about which effect sizes arrived. An "interval" series that is not the
    ``(value, lower, upper)`` tuple its producer returns yields ``None``, which
    ``_logged_as_metric`` refuses, so an unreadable shape is written as nothing
    AND named as unmeasured rather than skipped in silence.
    """
    if shape == "interval":
        return raw[0] if isinstance(raw, tuple) and raw else None
    return raw


def _unmeasured_note(
    report: Dict[str, Any], sep: str, include_group_stats: bool = True
) -> Dict[str, Any]:
    """Which metric series and per-group statistics this run could NOT measure.

    An unmeasurable value is written to a tracker as NOTHING, which is right (a
    NaN on a panel is indistinguishable from a measurement) and is not enough on
    its own: the panel then shows the metrics that worked and says nothing about
    the ones that did not. Measured 2026-09-27 on a 300-row
    ``classification_fairness_report`` where one group has no positive labels, so
    its TPR is undefined while demographic parity is fine:

      logged: fairness/group/M/tpr = 0.7826
      absent: fairness/group/F/tpr
      and     fairness/fairness_score_status = "assessed",
              fairness/n_groups_excluded = 0

    A reader comparing TPR across groups saw ONE bar, with every disclosure field
    on the run saying nothing had been left out, and three of the nine metric
    series (equalized_odds_difference, equal_opportunity_difference,
    auroc_parity) were absent the same way. That is the size-gate disclosure
    problem of ``_excluded_levels_note`` one level down, so it gets the same
    remedy and the same rule: emitted on EVERY run, never only when something was
    unmeasurable, because a field that appears only on failure makes its absence
    ambiguous between "nothing was missed" and "this version did not record it".

    ``sep`` is the tracker's own key separator, so a named-but-absent series reads
    exactly as the key a reader went looking for.

    BGL-5 (2026-09-27). This note read ``report['metrics']`` and
    ``report['group_stats']`` ONLY, so the two OTHER series both trackers write,
    the confidence intervals and the effect sizes, kept the whole of the defect
    above: written as nothing, named nowhere, beside disclosure fields that read
    zero. Measured on the exact shape ``compute_metric_with_ci`` returns from its
    "Bootstrap failed" branch (a MEASURED point estimate 0.42, bounds NaN,
    standard error NaN) plus a NaN Cohen's d::

        logged  fairness.demographic_parity_difference.value = 0.42
        absent  ...ci_lower, ...ci_upper, ...std_error,
                fairness.effect_size.F_vs_M.cohens_d
        before  n_metrics_not_measured 0, n_group_stats_not_measured 0,
                and no param or tag naming the interval at all
        after   n_intervals_not_measured 3, intervals_not_measured
                "['demographic_parity_difference.ci_lower',
                  'demographic_parity_difference.ci_upper',
                  'demographic_parity_difference.std_error']",
                n_effect_sizes_not_measured 1, effect_sizes_not_measured
                "['F_vs_M.cohens_d']"

    The predicates below are the ones the WRITE loops use (``_is_measured`` for
    an interval, ``_logged_as_metric`` for an effect size), so the disclosure and
    the write can never disagree about which series arrived. Every one of the four
    interval series now reads exactly as the key a reader went looking for on
    BOTH trackers: until 2026-09-30 the W&B write loop had no ``standard_error``
    branch at all while this list named ``std_error``, so on that path a MEASURED
    standard error was dropped with the disclosure reading clean, and an
    unmeasured one was named under a key that could never appear. That writer
    writes it now.

    2026-09-29. The invariant two paragraphs up, "the note follows the write
    rather than the dict's key set", was NOT true in two ways, and both were
    measured at both writers:

    1. ``include_group_stats=False`` is a first-class parameter of both writers.
       With it, NOT ONE per-group series is written, and on a report with two
       FULLY MEASURED groups the note still read
       ``n_group_stats_not_measured = 0, group_stats_not_measured = "[]"``. A
       panel reader saw zero bars and every disclosure field saying nothing had
       been left out, which is worse than the one-bar case this note exists for.
       Suppressed is its OWN state, named as such: it is not "not measured",
       because the values exist and the caller chose not to log them, and saying
       "not measured" would be the mirror defect. ``group_stats_status`` plus
       ``n_group_stats_suppressed`` account for every absent series.
    2. ``if not isinstance(ci_data, dict): continue`` (and the same test for the
       effect sizes and the per-group stats) dropped the disclosure with the
       write. Measured with ``metrics_with_ci = {'demographic_parity_difference':
       (0.42, 0.31, 0.53)}`` and ``effect_sizes = {'F_vs_M': (-0.81,)}``, the
       shape ``compute_metric_with_ci``'s tuple return would give a caller who
       passes it straight through: four interval series and one effect-size
       series written as NOTHING, with all four disclosure pairs reading clean.
       An entry this module cannot read is now counted as unwritten AND named in
       ``unreadable_entries``, with one warning, because an unreadable shape is a
       bug in the caller's report and not a property of the data.
    """
    unreadable_entries: List[str] = []

    metrics = report.get("metrics") or {}
    if isinstance(metrics, dict):
        unmeasured_metrics = sorted(
            str(name) for name, value in metrics.items() if not _logged_as_metric(value)
        )
    else:
        unmeasured_metrics = []
        unreadable_entries.append(f"metrics ({type(metrics).__name__})")

    unmeasured_stats: List[str] = []
    suppressed_stats: List[str] = []
    group_stats = report.get("group_stats") or {}
    if isinstance(group_stats, dict):
        for group_name, stats in group_stats.items():
            if not isinstance(stats, dict):
                # The write loop skips this entry too, so every series it might
                # have carried is absent from the panel. Named, not silent.
                unreadable_entries.append(f"group_stats{sep}{group_name} ({type(stats).__name__})")
                unmeasured_stats.append(f"{group_name}{sep}<unreadable>")
                continue
            for stat_name, value in stats.items():
                if not _logged_as_metric(value):
                    unmeasured_stats.append(f"{group_name}{sep}{stat_name}")
                elif not include_group_stats:
                    # Measured, and deliberately not written. A different state
                    # from the line above, and it needs its own name.
                    suppressed_stats.append(f"{group_name}{sep}{stat_name}")
    else:
        unreadable_entries.append(f"group_stats ({type(group_stats).__name__})")
    unmeasured_stats.sort()
    suppressed_stats.sort()

    # The interval series. The writers attempt all four of these fields and skip
    # each one that is not a measurement, so an ABSENT field is unmeasured too
    # (`ci_data.get(...)` is None there, which `_is_measured` refuses): the note
    # follows the write rather than the dict's key set.
    unmeasured_intervals: List[str] = []
    metrics_with_ci = report.get("metrics_with_ci") or {}
    if isinstance(metrics_with_ci, dict):
        for metric_name, ci_data in metrics_with_ci.items():
            if not isinstance(ci_data, dict):
                # The write loop attempts nothing for this entry, so all four of
                # its series are absent. Counted as absent, and the shape named.
                unreadable_entries.append(
                    f"metrics_with_ci{sep}{metric_name} ({type(ci_data).__name__})"
                )
                unmeasured_intervals.extend(
                    f"{metric_name}{sep}{suffix}" for _field, suffix in _CI_SERIES
                )
                continue
            for field, suffix in _CI_SERIES:
                if not _is_measured(ci_data.get(field)):
                    unmeasured_intervals.append(f"{metric_name}{sep}{suffix}")
    else:
        unreadable_entries.append(f"metrics_with_ci ({type(metrics_with_ci).__name__})")
    unmeasured_intervals.sort()

    # The effect sizes, with the writers' own shape test: a risk ratio is written
    # only when it arrives as the (value, lower, upper) tuple its producer
    # returns, so that is the shape this reads.
    unmeasured_effects: List[str] = []
    effect_sizes = report.get("effect_sizes") or {}
    if isinstance(effect_sizes, dict):
        for pair_name, effects in effect_sizes.items():
            if not isinstance(effects, dict):
                # Nothing is written for this pair. The pair is named WITHOUT a
                # series suffix, because which effect size a bare tuple was meant
                # to carry is exactly what this module cannot tell.
                unreadable_entries.append(
                    f"effect_sizes{sep}{pair_name} ({type(effects).__name__})"
                )
                unmeasured_effects.append(str(pair_name))
                continue
            # Off _EFFECT_SERIES, the same list both write loops iterate. It used
            # to name only cohens_d and risk_ratio, so the producer's measured
            # cohens_h and odds_ratio were neither written nor named.
            for field, suffix, shape in _EFFECT_SERIES:
                if field not in effects:
                    continue
                if not _logged_as_metric(_effect_point_value(effects[field], shape)):
                    unmeasured_effects.append(f"{pair_name}{sep}{suffix}")
    else:
        unreadable_entries.append(f"effect_sizes ({type(effect_sizes).__name__})")
    unmeasured_effects.sort()
    unreadable_entries.sort()

    # One warning, from the shared root, so both writers report it identically.
    # An unreadable entry is not a property of the data: it means the report this
    # was handed does not have the shape the writers read, and the series it
    # carried reached the tracker as nothing.
    if unreadable_entries:
        warnings.warn(
            f"{len(unreadable_entries)} report entr(ies) are not in a shape these loggers can "
            f"read, so NOTHING was written for them: {', '.join(unreadable_entries)}. They are "
            "counted in the not-measured disclosure and named in unreadable_entries on the run. "
            "An absent series is not a measured zero.",
            UserWarning,
            stacklevel=3,
        )

    return {
        "n_metrics_not_measured": len(unmeasured_metrics),
        "metrics_not_measured": str(unmeasured_metrics),
        "n_group_stats_not_measured": len(unmeasured_stats),
        "group_stats_not_measured": str(unmeasured_stats),
        # Measured, and NOT written, because the caller passed
        # include_group_stats=False. Recorded on every run, like everything else
        # here, so an empty per-group panel always carries its own reason.
        "group_stats_status": ("logged" if include_group_stats else "suppressed_by_caller"),
        "n_group_stats_suppressed": len(suppressed_stats),
        "group_stats_suppressed": str(suppressed_stats),
        "n_unreadable_entries": len(unreadable_entries),
        "unreadable_entries": str(unreadable_entries),
        "n_intervals_not_measured": len(unmeasured_intervals),
        "intervals_not_measured": str(unmeasured_intervals),
        "n_effect_sizes_not_measured": len(unmeasured_effects),
        "effect_sizes_not_measured": str(unmeasured_effects),
    }


def _library_version() -> str:
    """The version of vfairness that produced this run, or ``"unknown"``.

    Every logged run used to be tagged ``library_version = "1.1.0"``, a literal
    that no release ever carried: the package is at ``vfairness.__version__``,
    0.1.0 at the time this was found. A run tagged with a version the code is not
    cannot be reproduced from its own audit trail, which is the one thing that
    tag is for. Read from the package rather than restated here, so it cannot go
    stale again, and reported as ``"unknown"`` rather than as a plausible number
    if it cannot be read at all.

    Imported inside the function: ``vfairness/__init__`` imports this module, so
    a module-level import would be circular.
    """
    try:
        import vfairness

        version = getattr(vfairness, "__version__", None)
    except Exception:  # pragma: no cover - defensive, the package is importable
        version = None
    return version if isinstance(version, str) and version else "unknown"


# MLflow Integration


def log_fairness_to_mlflow(
    analyzer_or_report: Any,
    prefix: str = "fairness",
    log_artifacts: bool = True,
    include_group_stats: bool = True,
) -> Dict[str, Any]:
    """
    Log fairness results to an active MLflow run.

    Logs all fairness metrics, confidence intervals (if computed), effect sizes,
    and group statistics to the currently active MLflow run.

    Args:
        analyzer_or_report: Either a FairnessAnalyzer instance or a report dict
            from classification_fairness_report/regression_fairness_report
        prefix: Prefix for all logged metrics (default: "fairness")
        log_artifacts: Whether to log report as JSON artifact
        include_group_stats: Whether to log per-group statistics

    Returns:
        Dict with all logged metric names and values. ``<prefix>.fairness_score``
        is present only when the run produced a score; on a not-assessable run it
        is absent and ``<prefix>.fairness_score_status`` reads
        ``"not_assessable"`` instead of a fabricated number. An unmeasurable
        metric, per-group statistic, confidence interval or effect size is
        likewise logged as nothing, and named in
        ``<prefix>.metrics_not_measured`` / ``<prefix>.group_stats_not_measured``
        / ``<prefix>.intervals_not_measured`` /
        ``<prefix>.effect_sizes_not_measured``, which are recorded on every run
        so their absence is never ambiguous. ``include_group_stats=False``
        suppresses every per-group series, and that is its own state:
        ``<prefix>.group_stats_status`` reads ``"suppressed_by_caller"`` and
        ``<prefix>.group_stats_suppressed`` names the measured series that were
        NOT written, so an empty per-group panel always carries its reason.
        A report entry whose shape these loggers cannot read is counted as
        unwritten and named in ``<prefix>.unreadable_entries``, with a warning.

    Raises:
        ImportError: If MLflow is not installed
        RuntimeError: If no active MLflow run exists

    Example:
        >>> import mlflow
        >>> from vfairness import FairnessAnalyzer, log_fairness_to_mlflow
        >>>
        >>> with mlflow.start_run():
        ...     analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        ...     logged = log_fairness_to_mlflow(analyzer, prefix="model_v1")
        ...     print(f"Logged {len(logged)} metrics")
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "MLflow is required for this function. Install it with: pip install mlflow"
        )

    # Check for active run
    if mlflow.active_run() is None:
        raise RuntimeError(
            "No active MLflow run. Use mlflow.start_run() first or pass the run context."
        )

    # Annotated Dict[str, Any]: this records what was logged, which includes the
    # non-numeric fairness_score_status entry for a not-assessable run.
    logged_metrics: Dict[str, Any] = {}

    # Handle FairnessAnalyzer instance
    if hasattr(analyzer_or_report, "compute_all_metrics"):
        # It's a FairnessAnalyzer
        analyzer = analyzer_or_report
        report = analyzer.get_report(include_ci=True)
    elif isinstance(analyzer_or_report, dict):
        # It's a report dict
        report = analyzer_or_report
    else:
        raise TypeError(f"Expected FairnessAnalyzer or report dict, got {type(analyzer_or_report)}")

    # Log basic metrics. The guard is `_logged_as_metric` so the disclosure below
    # names exactly the series this loop did not write.
    metrics = report.get("metrics", {})
    for metric_name, value in metrics.items():
        if _logged_as_metric(value):
            key = f"{prefix}.{metric_name}"
            mlflow.log_metric(key, float(value))
            logged_metrics[key] = value

    # Log confidence intervals if present
    metrics_with_ci = report.get("metrics_with_ci", {})
    for metric_name, ci_data in metrics_with_ci.items():
        if isinstance(ci_data, dict):
            # Point estimate.
            #
            # CRITICAL (fail closed): guarded exactly like the bounds below. An
            # unguarded float() logged a NaN point estimate into the tracker as
            # if it were a measurement (and raised TypeError on None), so a run
            # that measured nothing wrote a data point a dashboard cannot tell
            # from a real one. An unmeasurable estimate is logged as nothing.
            if _is_measured(ci_data.get("point_estimate")):
                key = f"{prefix}.{metric_name}.value"
                mlflow.log_metric(key, float(ci_data["point_estimate"]))
                logged_metrics[key] = ci_data["point_estimate"]

            # Lower bound, upper bound and standard error.
            #
            # CRITICAL (fail closed): the SAME guard as the point estimate above,
            # and for the same reason. These three used a bare ``np.isnan``,
            # which differs from _is_measured in two ways that both reached the
            # tracker: it RAISES TypeError on None or a non-numeric bound (the
            # exact crash the point estimate was fixed for), and it answers False
            # for INFINITY, so an unestimated interval was logged as
            # ci_lower=-inf / ci_upper=inf, a pair of numbers a panel plots like
            # any measured bound. An interval that was never estimated is logged
            # as nothing.
            if _is_measured(ci_data.get("lower_bound")):
                key = f"{prefix}.{metric_name}.ci_lower"
                mlflow.log_metric(key, float(ci_data["lower_bound"]))
                logged_metrics[key] = ci_data["lower_bound"]

            if _is_measured(ci_data.get("upper_bound")):
                key = f"{prefix}.{metric_name}.ci_upper"
                mlflow.log_metric(key, float(ci_data["upper_bound"]))
                logged_metrics[key] = ci_data["upper_bound"]

            if _is_measured(ci_data.get("standard_error")):
                key = f"{prefix}.{metric_name}.std_error"
                mlflow.log_metric(key, float(ci_data["standard_error"]))
                logged_metrics[key] = ci_data["standard_error"]

    # Log effect sizes. The guard is `_logged_as_metric`, the same expression the
    # metric loop above and the disclosure below use, so the three cannot
    # disagree about which effect sizes arrived. It used to be a bare
    # `not np.isnan(val)`, which RAISES TypeError on a None effect size
    # (measured: "ufunc 'isnan' not supported for the input types") and so took
    # the whole logging call down instead of recording that the pair was not
    # measured. An infinite ratio is still written: it is a measured total
    # exclusion, and `_logged_as_metric` keeps it for that reason.
    effect_sizes = report.get("effect_sizes", {})
    for pair_name, effects in effect_sizes.items():
        if isinstance(effects, dict):
            # Every series on _EFFECT_SERIES, which is the same list the
            # disclosure below reads. Hardcoding cohens_d and risk_ratio here
            # dropped the producer's measured cohens_h and odds_ratio, and the
            # disclosure hardcoded the same two so it could not report the loss.
            for field, suffix, shape in _EFFECT_SERIES:
                if field not in effects:
                    continue
                val = _effect_point_value(effects[field], shape)
                if _logged_as_metric(val):
                    key = f"{prefix}.effect_size.{pair_name}.{suffix}"
                    mlflow.log_metric(key, float(val))
                    logged_metrics[key] = val

    # Log group statistics
    if include_group_stats:
        group_stats = report.get("group_stats", {})
        for group_name, stats in group_stats.items():
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    if _logged_as_metric(value):
                        key = f"{prefix}.group.{group_name}.{stat_name}"
                        mlflow.log_metric(key, float(value))
                        logged_metrics[key] = value

    # Log assessment summary.
    #
    # CRITICAL (fail closed): ``fairness_score`` is Optional. It is None when the
    # run was NOT ASSESSABLE, i.e. no metric could be assessed, and it used to
    # crash here with TypeError on float(None). Do NOT substitute a number: on a
    # dashboard 1.0 is indistinguishable from measured perfect fairness and 0.0
    # from a measured total failure, so either one turns "we could not check"
    # into a verdict nobody made. Skip the metric entirely and record an explicit
    # status tag, so the missing score is visible instead of implied.
    assessment = report.get("assessment", {})
    if "fairness_score" in assessment:
        score = assessment["fairness_score"]
        key = f"{prefix}.fairness_score"
        score_is_assessed = isinstance(score, (int, float)) and not np.isnan(float(score))
        if score_is_assessed:
            mlflow.log_metric(key, float(score))
            logged_metrics[key] = score
        status_key = f"{prefix}.fairness_score_status"
        status = "assessed" if score_is_assessed else "not_assessable"
        mlflow.set_tag(status_key, status)
        logged_metrics[status_key] = status

    # Log parameters
    mlflow.log_param(f"{prefix}.task_type", report.get("task_type", "unknown"))

    data_info = report.get("data_info", {})
    if "n_samples" in data_info:
        mlflow.log_param(f"{prefix}.n_samples", data_info["n_samples"])
    if "n_groups" in data_info:
        mlflow.log_param(f"{prefix}.n_groups", data_info["n_groups"])
    if "valid_groups" in data_info:
        mlflow.log_param(f"{prefix}.groups", str(data_info["valid_groups"]))
    # ``groups`` above is the SURVIVORS of min_group_size. Record what it leaves
    # out beside it, always, so a run whose protected levels were partly
    # unmeasured cannot look identical on the dashboard to one where nothing was.
    for note_key, note_value in _excluded_levels_note(data_info).items():
        mlflow.log_param(f"{prefix}.{note_key}", note_value)
    # A metric or a per-group statistic that could not be measured is logged as
    # nothing, which leaves the panel showing only what worked. Name what is
    # missing, on every run. See _unmeasured_note.
    # include_group_stats travels WITH the report: with it False not one per-group
    # series is written, and the note used to read "0 / []" beside an empty panel.
    for note_key, note_value in _unmeasured_note(report, ".", include_group_stats).items():
        mlflow.log_param(f"{prefix}.{note_key}", note_value)
        logged_metrics[f"{prefix}.{note_key}"] = note_value

    # Log tags
    mlflow.set_tag(f"{prefix}.library", "vfairness")
    mlflow.set_tag(f"{prefix}.library_version", _library_version())

    # Log report as artifact
    if log_artifacts:
        import json
        import os
        import tempfile

        # Convert report to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, "value"):  # Enum
                return obj.value
            return obj

        serializable_report = make_serializable(report)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(serializable_report, f, indent=2)
            temp_path = f.name

        mlflow.log_artifact(temp_path, f"{prefix}_report")
        os.unlink(temp_path)

    return logged_metrics


# Weights & Biases Integration


def log_fairness_to_wandb(
    analyzer_or_report: Any,
    *,
    prefix: str = "fairness",
    log_artifacts: bool = True,
    include_group_stats: bool = True,
) -> Dict[str, Any]:
    """
    Log fairness results to an active Weights & Biases run.

    Mirrors the interface of :func:`log_fairness_to_mlflow` for W&B.
    Logs all fairness metrics, confidence intervals, effect sizes,
    and group statistics to the currently active W&B run.

    Args:
        analyzer_or_report: Either a FairnessAnalyzer instance or a report dict
            from classification_fairness_report/regression_fairness_report.
        prefix: Prefix for all logged metrics (default: "fairness").
        log_artifacts: Whether to log the full report as a JSON artifact.
        include_group_stats: Whether to log per-group statistics.

    Returns:
        Dict with all logged metric names and values. ``<prefix>/fairness_score``
        is present only when the run produced a score; on a not-assessable run it
        is absent and ``<prefix>/fairness_score_status`` reads
        ``"not_assessable"`` instead of a fabricated number. An unmeasurable
        metric, per-group statistic, confidence interval or effect size is
        likewise logged as nothing, and named in
        ``<prefix>/metrics_not_measured`` / ``<prefix>/group_stats_not_measured``
        / ``<prefix>/intervals_not_measured`` /
        ``<prefix>/effect_sizes_not_measured``, which are recorded on every run
        so their absence is never ambiguous. ``include_group_stats=False``
        suppresses every per-group series, and that is its own state:
        ``<prefix>/group_stats_status`` reads ``"suppressed_by_caller"`` and
        ``<prefix>/group_stats_suppressed`` names the measured series that were
        NOT written, so an empty per-group panel always carries its reason.
        A report entry whose shape these loggers cannot read is counted as
        unwritten and named in ``<prefix>/unreadable_entries``, with a warning.

    Raises:
        ImportError: If wandb is not installed.
        RuntimeError: If no active W&B run exists.

    Example:
        >>> import wandb
        >>> from vfairness import FairnessAnalyzer, log_fairness_to_wandb
        >>>
        >>> wandb.init(project="fairness-audit")
        >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        >>> logged = log_fairness_to_wandb(analyzer)
        >>> print(f"Logged {len(logged)} metrics")
        >>> wandb.finish()
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "Weights & Biases is required for this function. Install it with: pip install wandb"
        )

    if wandb.run is None:
        raise RuntimeError("No active W&B run. Use wandb.init() first.")

    # Annotated Dict[str, Any]: this records what was logged, which includes the
    # non-numeric fairness_score_status entry for a not-assessable run.
    logged_metrics: Dict[str, Any] = {}

    # Handle FairnessAnalyzer instance
    if hasattr(analyzer_or_report, "compute_all_metrics"):
        analyzer = analyzer_or_report
        report = analyzer.get_report(include_ci=True)
    elif isinstance(analyzer_or_report, dict):
        report = analyzer_or_report
    else:
        raise TypeError(f"Expected FairnessAnalyzer or report dict, got {type(analyzer_or_report)}")

    log_data: Dict[str, Any] = {}

    # Log basic metrics. Same guard as the MLflow path, and the same reason: the
    # disclosure on the run summary names exactly the series this loop skipped.
    metrics = report.get("metrics", {})
    for metric_name, value in metrics.items():
        if _logged_as_metric(value):
            key = f"{prefix}/{metric_name}"
            log_data[key] = float(value)
            logged_metrics[key] = value

    # Log confidence intervals
    metrics_with_ci = report.get("metrics_with_ci", {})
    for metric_name, ci_data in metrics_with_ci.items():
        if isinstance(ci_data, dict):
            # CRITICAL (fail closed): mirrors the MLflow path. A NaN point
            # estimate is not a measurement, and W&B cannot distinguish one from
            # a measured value once it is on a panel. Log nothing instead.
            if _is_measured(ci_data.get("point_estimate")):
                key = f"{prefix}/{metric_name}/value"
                log_data[key] = float(ci_data["point_estimate"])
                logged_metrics[key] = ci_data["point_estimate"]

            # The bounds get the point estimate's guard too, mirroring the MLflow
            # path: a bare np.isnan raised on a None bound and passed an INFINITE
            # one straight onto a W&B panel as though the interval had been
            # estimated.
            if _is_measured(ci_data.get("lower_bound")):
                key = f"{prefix}/{metric_name}/ci_lower"
                log_data[key] = float(ci_data["lower_bound"])
                logged_metrics[key] = ci_data["lower_bound"]

            if _is_measured(ci_data.get("upper_bound")):
                key = f"{prefix}/{metric_name}/ci_upper"
                log_data[key] = float(ci_data["upper_bound"])
                logged_metrics[key] = ci_data["upper_bound"]

            # THE ARM WITHOUT ITS COMPLEMENT. _CI_SERIES is shared with the
            # disclosure precisely so that "a series named as unmeasured is
            # exactly a series not written", and this writer had no
            # standard_error branch at all while that list names std_error.
            # Measured 2026-09-30 with standard_error = 0.06: the MLflow path
            # wrote fairness.demographic_parity_difference.std_error, W&B wrote no
            # such key, and fairness/n_intervals_not_measured read 0 with no
            # warning. A MEASURED standard error was absent from the panel with
            # every disclosure field reading clean, and in the other direction an
            # unmeasured one was named under a key that could never exist on a
            # W&B run. Writing it makes the shared list true on both paths.
            if _is_measured(ci_data.get("standard_error")):
                key = f"{prefix}/{metric_name}/std_error"
                log_data[key] = float(ci_data["standard_error"])
                logged_metrics[key] = ci_data["standard_error"]

    # Log effect sizes. Same guard as the MLflow path and the same reason: a bare
    # `not np.isnan(val)` raises TypeError on a None effect size and took the
    # logging call down with it, and the disclosure below reads this very
    # predicate, so a series named as unmeasured is exactly a series not written.
    effect_sizes = report.get("effect_sizes", {})
    for pair_name, effects in effect_sizes.items():
        if isinstance(effects, dict):
            # _EFFECT_SERIES, exactly as on the MLflow path and in the
            # disclosure. See the list's own comment for what the two hardcoded
            # fields were losing.
            for field, suffix, shape in _EFFECT_SERIES:
                if field not in effects:
                    continue
                val = _effect_point_value(effects[field], shape)
                if _logged_as_metric(val):
                    key = f"{prefix}/effect_size/{pair_name}/{suffix}"
                    log_data[key] = float(val)
                    logged_metrics[key] = val

    # Log group statistics
    if include_group_stats:
        group_stats = report.get("group_stats", {})
        for group_name, stats in group_stats.items():
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    if _logged_as_metric(value):
                        key = f"{prefix}/group/{group_name}/{stat_name}"
                        log_data[key] = float(value)
                        logged_metrics[key] = value

    # Log assessment summary.
    #
    # CRITICAL (fail closed): mirrors the MLflow path above. ``fairness_score``
    # is None on a NOT ASSESSABLE run and used to crash here on float(None).
    # Never substitute 0.0 or 1.0: a W&B panel cannot tell a fabricated score
    # from a measured one, so a filled-in number would certify (or condemn) a run
    # that measured nothing. Log no score, log an explicit status instead.
    assessment = report.get("assessment", {})
    score_status: Optional[str] = None
    if "fairness_score" in assessment:
        score = assessment["fairness_score"]
        key = f"{prefix}/fairness_score"
        score_is_assessed = isinstance(score, (int, float)) and not np.isnan(float(score))
        if score_is_assessed:
            log_data[key] = float(score)
            logged_metrics[key] = score
        status_key = f"{prefix}/fairness_score_status"
        score_status = "assessed" if score_is_assessed else "not_assessable"
        log_data[status_key] = score_status
        logged_metrics[status_key] = score_status

    # Batch log all metrics
    wandb.log(log_data)

    # Log config as summary
    wandb.run.summary[f"{prefix}/task_type"] = report.get("task_type", "unknown")
    wandb.run.summary[f"{prefix}/library"] = "vfairness"
    if score_status is not None:
        # Also on the run summary, which is what the runs table shows: a reader
        # scanning runs must see "not_assessable" rather than an empty score
        # column they might read as a pass.
        wandb.run.summary[f"{prefix}/fairness_score_status"] = score_status

    data_info = report.get("data_info", {})
    if "n_samples" in data_info:
        wandb.run.summary[f"{prefix}/n_samples"] = data_info["n_samples"]
    if "n_groups" in data_info:
        wandb.run.summary[f"{prefix}/n_groups"] = data_info["n_groups"]
    # Same disclosure as the MLflow path, on the surface a reader actually scans.
    # See _excluded_levels_note.
    for note_key, note_value in _excluded_levels_note(data_info).items():
        wandb.run.summary[f"{prefix}/{note_key}"] = note_value
    # The metrics and per-group statistics that could not be measured, on the
    # surface a reader actually scans, on every run. See _unmeasured_note.
    # include_group_stats travels WITH the report, exactly as on the MLflow path.
    for note_key, note_value in _unmeasured_note(report, "/", include_group_stats).items():
        wandb.run.summary[f"{prefix}/{note_key}"] = note_value
        logged_metrics[f"{prefix}/{note_key}"] = note_value

    # Log report as artifact
    if log_artifacts:
        import json
        import os
        import tempfile

        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, "value"):
                return obj.value
            return obj

        serializable_report = make_serializable(report)

        artifact = wandb.Artifact(f"{prefix}_report", type="fairness_report")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(serializable_report, f, indent=2)
            temp_path = f.name

        artifact.add_file(temp_path, name="fairness_report.json")
        wandb.log_artifact(artifact)
        os.unlink(temp_path)

    return logged_metrics


# Auto-Logging Decorator


def _select_logged_metrics(report: Dict[str, Any], metrics: Optional[List[str]]) -> Dict[str, Any]:
    """Narrow ``report`` to the metric series the caller asked to log.

    ``metrics=None`` means "log every metric the analyzer produced" and returns
    the report unchanged.

    A requested name the analyzer did NOT produce is logged as nothing and named
    in a warning. No stand-in is written for it: once a number is on a tracker
    panel nothing distinguishes an invented one from a measurement, so the honest
    record of a metric that does not exist here is its absence plus a warning
    saying which names were dropped and what was available instead.
    """
    if metrics is None:
        return report

    requested = list(dict.fromkeys(metrics))
    produced = report.get("metrics", {})
    missing = [name for name in requested if name not in produced]
    if missing:
        available = ", ".join(sorted(produced)) or "none"
        warnings.warn(
            f"auto_log_fairness: requested metric(s) {', '.join(missing)} were not produced "
            "by the analyzer for this run, so nothing is logged for them (no value is "
            f"substituted). Metrics available here: {available}",
            stacklevel=2,
        )

    selected = {name: produced[name] for name in requested if name in produced}
    narrowed: Dict[str, Any] = dict(report)
    narrowed["metrics"] = selected
    ci_data = report.get("metrics_with_ci")
    if isinstance(ci_data, dict):
        narrowed["metrics_with_ci"] = {
            name: value for name, value in ci_data.items() if name in selected
        }
    return narrowed


def _record_thresholds(
    tracker: Any,
    thresholds: Optional[Dict[str, float]],
    prefix: str,
    backend: Literal["mlflow", "wandb"],
) -> None:
    """Record the caller's thresholds on the active run WITHOUT enforcing them.

    The docstring of :func:`auto_log_fairness` promised these were logged, and
    nothing logged them: the one thing it claimed to do with ``thresholds`` did
    not happen. They are recorded as run PARAMETERS (mlflow) / summary fields
    (W&B), never as metric series, because a metric series is exactly what a
    dashboard plots as something this run measured, and a threshold is
    configuration the caller supplied.

    ``<prefix>.thresholds_enforced`` is written as ``"false"`` beside them. Three
    states, never two: a threshold sitting next to a value looks like a verdict,
    and no verdict was made here. The report's own assessment used
    ``report["thresholds_used"]``, not these, so without the marker a reader
    would credit this decorator with a gate that never ran. Use
    :func:`assert_fairness` for a gate that actually refuses.
    """
    if not thresholds:
        return

    if backend == "mlflow":
        for metric_name, threshold in thresholds.items():
            tracker.log_param(f"{prefix}.threshold.{metric_name}", threshold)
        tracker.set_tag(f"{prefix}.thresholds_enforced", "false")
    else:
        for metric_name, threshold in thresholds.items():
            tracker.run.summary[f"{prefix}/threshold/{metric_name}"] = threshold
        tracker.run.summary[f"{prefix}/thresholds_enforced"] = "false"


def auto_log_fairness(
    backend: Literal["mlflow", "wandb"] = "mlflow",
    *,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    prefix: str = "fairness",
) -> Callable:
    """
    Decorator that automatically logs fairness metrics after a function call.

    The decorated function must return a tuple of
    ``(y_pred, y_true, sensitive_attr)`` or a dict with those keys.
    The decorator computes fairness metrics and logs them to the specified
    experiment tracking backend.

    Args:
        backend: Tracking backend - 'mlflow' or 'wandb'.
        metrics: Names of the fairness metrics to log. If None, every metric the
            analyzer produced is logged. A requested name the analyzer did not
            produce is logged as nothing and named in a warning; no value is
            substituted for it. The per-run summary (fairness score, group
            statistics, run parameters, effect sizes) is logged in full either
            way, because it describes the run rather than one metric.
        thresholds: Thresholds to RECORD next to the metrics. Logged as run
            parameters under ``<prefix>.threshold.<metric>`` and NOT enforced:
            nothing here compares a metric against them and no verdict is derived
            from them, which is why ``<prefix>.thresholds_enforced`` is logged as
            ``"false"`` alongside. Use :func:`assert_fairness` to gate on them.
        prefix: Prefix for logged metric names.

    Returns:
        Decorator function.

    Note:
        ``protected_attr_column`` was accepted here until 2026-09-09 and never
        read. It was documented as "column name if returning a DataFrame", and
        the wrapper has no DataFrame return path to name a column in: it accepts
        a ``(y_pred, y_true, sensitive_attr)`` tuple or a dict with those keys,
        and nothing else. It is removed rather than kept as an ignored argument,
        so passing it now raises ``TypeError`` instead of quietly doing nothing.

    Example:
        >>> @auto_log_fairness(backend="mlflow")
        ... def evaluate_model(X_test, y_test, sensitive_attr):
        ...     y_pred = model.predict(X_test)
        ...     return y_pred, y_test, sensitive_attr
        >>>
        >>> # When called, fairness metrics are automatically logged
        >>> evaluate_model(X_test, y_test, gender)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Parse the result
            if isinstance(result, tuple) and len(result) == 3:
                y_pred, y_true, sensitive_attr = result
            elif isinstance(result, dict):
                y_pred = result.get("y_pred")
                y_true = result.get("y_true")
                sensitive_attr = result.get("sensitive_attr")
            else:
                warnings.warn(
                    "auto_log_fairness: function must return (y_pred, y_true, sensitive_attr) "
                    "tuple or dict with those keys. Skipping fairness logging.",
                    stacklevel=2,
                )
                return result

            if y_pred is None or y_true is None or sensitive_attr is None:
                warnings.warn(
                    "auto_log_fairness: could not extract y_pred, y_true, sensitive_attr "
                    "from function result. Skipping fairness logging.",
                    stacklevel=2,
                )
                return result

            # Compute fairness report
            try:
                from .analyzer import FairnessAnalyzer

                analyzer = FairnessAnalyzer(y_true, y_pred, sensitive_attr)
                report = analyzer.get_report(include_ci=False)
                report = _select_logged_metrics(report, metrics)

                if backend == "mlflow":
                    try:
                        import mlflow

                        if mlflow.active_run() is not None:
                            # Configuration first, measurements second. The
                            # metric logging can fail on its own (an artifact
                            # upload, a tracker outage), and this whole block
                            # degrades to a warning when it does; recording the
                            # thresholds first means the run still says which
                            # thresholds were quoted and that nothing enforced
                            # them, rather than losing that with the metrics.
                            _record_thresholds(mlflow, thresholds, prefix, "mlflow")
                            log_fairness_to_mlflow(report, prefix=prefix)
                        else:
                            warnings.warn(
                                "auto_log_fairness: no active MLflow run. "
                                "Skipping fairness logging.",
                                stacklevel=2,
                            )
                    except ImportError:
                        warnings.warn(
                            "auto_log_fairness: mlflow not installed. Skipping fairness logging.",
                            stacklevel=2,
                        )

                elif backend == "wandb":
                    try:
                        import wandb

                        if wandb.run is not None:
                            # Same order as the MLflow branch, same reason.
                            _record_thresholds(wandb, thresholds, prefix, "wandb")
                            log_fairness_to_wandb(report, prefix=prefix)
                        else:
                            warnings.warn(
                                "auto_log_fairness: no active W&B run. Skipping fairness logging.",
                                stacklevel=2,
                            )
                    except ImportError:
                        warnings.warn(
                            "auto_log_fairness: wandb not installed. Skipping fairness logging.",
                            stacklevel=2,
                        )
                else:
                    warnings.warn(
                        f"auto_log_fairness: unknown backend '{backend}'. Use 'mlflow' or 'wandb'.",
                        stacklevel=2,
                    )

            except Exception as e:
                warnings.warn(
                    f"auto_log_fairness: fairness logging failed: {e}",
                    stacklevel=2,
                )

            return result

        return wrapper

    return decorator


# pytest Integration


class FairnessAssertionError(AssertionError):
    """Custom assertion error for fairness checks."""

    def __init__(
        self, message: str, failed_metrics: Dict[str, Dict[str, Any]], all_metrics: Dict[str, float]
    ):
        super().__init__(message)
        self.failed_metrics = failed_metrics
        self.all_metrics = all_metrics


def assert_fairness(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    *,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    task_type: Optional[Literal["classification", "regression"]] = None,
    min_group_size: int = 30,
    include_ci: bool = False,
    ci_must_exclude_zero: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    message: Optional[str] = None,
) -> Dict[str, float]:
    """
    Assert that fairness metrics are within acceptable thresholds.

    Use this function in pytest tests to automatically validate model fairness
    in CI/CD pipelines.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels/values
        sensitive_attr: Protected attribute(s)
        metrics: List of metrics to check. If None, checks all relevant metrics.
        thresholds: Dict mapping metric names to the bound each is graded
            against, in that metric's own direction (a maximum for a
            difference, a required minimum for a ratio). If None, uses the
            default thresholds below. A metric whose threshold is ``None``
            has no bound and is reported as could-not-check, which FAILS: it
            is never rewritten to a default.
        task_type: 'classification' or 'regression' (auto-detected if None)
        min_group_size: Minimum samples per group
        include_ci: Whether to compute and check confidence intervals
        ci_must_exclude_zero: If True and include_ci=True, requires that the CI
            excludes zero (i.e., the difference is statistically significant)
        n_bootstrap: Number of bootstrap samples (fewer for faster CI/CD)
        confidence_level: Confidence level for CI
        random_state: Random seed for reproducibility
        message: Custom message to include in assertion error

    Returns:
        Dict of computed metric values (useful for logging)

    Raises:
        FairnessAssertionError: If any metric breaches its threshold in that
            metric's own direction, or if a requested check did not happen. A
            check does not happen when the metric could not be measured (a
            non-finite value), when the analyzer never produced the requested
            metric at all, when the metric has no known better-direction, so no
            threshold could be applied, when no threshold is configured for it
            at all (an explicit ``None`` used to be rewritten to 0.1 and graded
            against that invented bound), or when ``min_group_size`` excluded a
            protected group from the metric, so the value grades the surviving
            groups and leaves the excluded one unchecked. All five fail the
            gate: the raised error marks them ``not_measurable`` in
            ``failed_metrics`` so a caller can tell a measured violation from a
            could-not-check result. The last case means a run whose protected
            attribute has a tail of small levels cannot pass this gate until
            those levels are measured (lower ``min_group_size``) or removed from
            the data on purpose; a group nobody measured is not a group that
            passed.
        ValueError: If ``ci_must_exclude_zero=True`` is combined with
            ``include_ci=False``. That configuration asks for a significance
            check and switches off the intervals it needs, and it used to drop
            the check silently and report a pass. Also if ``metrics=[]``, which
            requests zero checks: the loop cannot fail on a list it never
            enters, so the gate used to return ``{}`` and no error for a model
            it had not examined. ``metrics=None`` still means "the defaults".

    Example:
        >>> # In a pytest test file
        >>> def test_model_is_fair():
        ...     y_true, y_pred, gender = get_test_data()
        ...     assert_fairness(
        ...         y_true, y_pred, gender,
        ...         metrics=['demographic_parity_difference', 'equalized_odds_difference'],
        ...         thresholds={'demographic_parity_difference': 0.1}
        ...     )

        >>> # With confidence interval check
        >>> def test_model_fairness_with_ci():
        ...     assert_fairness(
        ...         y_true, y_pred, gender,
        ...         include_ci=True,
        ...         ci_must_exclude_zero=True,  # Fail if disparity might be 0
        ...         n_bootstrap=1000  # Faster for CI/CD
        ...     )
    """
    from .analyzer import FairnessAnalyzer, MetricResult

    # CRITICAL (fail closed): ci_must_exclude_zero=True with include_ci=False is
    # a contradiction, and the gate used to resolve it by SILENTLY DROPPING the
    # requested significance check and then reporting success. A caller who asks
    # for a check, does not get it, and is told the run passed holds a false
    # certificate. Refuse the configuration instead: it is a caller mistake with
    # two obvious fixes, and neither of them is "pass anyway". Raised before any
    # metric is computed, so the contradiction cannot hide behind an unrelated
    # verdict.
    if ci_must_exclude_zero and not include_ci:
        raise ValueError(
            "assert_fairness: ci_must_exclude_zero=True requires include_ci=True. "
            "Without confidence intervals there is nothing to test for significance, "
            "and the requested check would be silently skipped while the gate "
            "reported a pass. Pass include_ci=True to run the check, or "
            "ci_must_exclude_zero=False to state that you are not asking for it."
        )

    # CRITICAL (fail closed): an EMPTY metric list is a gate that checks nothing,
    # and the loop below cannot fail on a list it never enters, so the gate
    # returned normally. Measured 2026-09-27 on 400 rows where men are approved
    # 85 percent of the time and women 20 percent (demographic_parity_difference
    # 0.695, which the same call with metrics=None fails):
    #     assert_fairness(y_true, y_pred, gender, metrics=[]) -> {} and NO raise.
    # Zero checks read as a pass, in a function whose entire purpose is to stop a
    # CI pipeline. This is the sibling of the C-05 hole fixed in
    # operations/cicd/gate.py, where `evaluate(metrics=[])` returned
    # approved=True with the summary "APPROVED - All fairness requirements met":
    # there the remedy was a blocking reason, here it is a refusal, because an
    # empty list is a caller mistake with two obvious fixes and neither of them
    # is "pass anyway". Raised before any metric is computed, so it cannot hide
    # behind an unrelated verdict. `metrics=None` still means "check the
    # defaults"; only an explicitly empty list is refused.
    if metrics is not None and len(metrics) == 0:
        raise ValueError(
            "assert_fairness: metrics=[] requests ZERO checks, so nothing would be "
            "compared against any threshold and this gate would report a pass for a "
            "model it never examined. Pass the metric names you want gated, or "
            "metrics=None to gate on the defaults for this task type."
        )

    # Create analyzer
    analyzer = FairnessAnalyzer(
        y_true, y_pred, sensitive_attr, task_type=task_type, min_group_size=min_group_size
    )

    # Default thresholds. The classification values MUST match the
    # default_thresholds in report.classification_fairness_report: the same
    # metric previously passed here at 0.1 while the report failed it at
    # 0.05. report.py owns the numbers; keep this dict in sync with it.
    default_thresholds = {
        # Classification
        "demographic_parity_difference": 0.1,
        "demographic_parity_ratio": 0.8,  # Minimum acceptable ratio
        "equalized_odds_difference": 0.1,
        "equal_opportunity_difference": 0.05,
        "predictive_parity_difference": 0.05,
        # Regression (relative to std)
        "mae_parity_difference": None,  # Will be computed based on data
        "rmse_parity_difference": None,
        "mean_prediction_difference": None,
    }

    # Merge with provided thresholds
    if thresholds:
        default_thresholds.update(thresholds)

    # For regression, compute relative thresholds
    if analyzer.task_type == "regression":
        y_std = np.std(y_true)
        for metric in [
            "mae_parity_difference",
            "rmse_parity_difference",
            "mean_prediction_difference",
        ]:
            if default_thresholds.get(metric) is None:
                # y_std is a numpy scalar; coerce to float to match the dict's
                # float|None value type without altering the computed threshold.
                default_thresholds[metric] = float(y_std * 0.1)

    # Determine which metrics to check
    if metrics is None:
        if analyzer.task_type == "classification":
            metrics = [
                "demographic_parity_difference",
                "equalized_odds_difference",
                "equal_opportunity_difference",
            ]
        else:
            metrics = ["mae_parity_difference"]

    # Compute metrics
    all_results = analyzer.compute_all_metrics(
        include_ci=include_ci,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    # CRITICAL (fail closed): ``min_group_size`` removes small groups from EVERY
    # metric computed above, and this gate then certified the run on whatever
    # survived. Reproduced 2026-09-10 on 2302 applicants whose protected
    # attribute has an ordinary long tail: 38 levels of 29 people (1102
    # applicants, 47.9 percent of the data) were DENIED OUTRIGHT, every one of
    # them sits below the default min_group_size=30, so the three default metrics
    # were computed over the two surviving levels only and read
    # demographic_parity_difference 0.0067. assert_fairness returned those values
    # and did not raise. Measured with the tail kept (min_group_size=1) the truth
    # is demographic_parity_difference 0.532 and disparate_impact_ratio 0.00, and
    # ModelFairnessGate.evaluate, which applies no size gate, BLOCKED the same
    # run: two surfaces of one library disagreed about one run and the stricter
    # one was right.
    #
    # A protected level that no metric covered was NEVER CHECKED, and a check
    # that did not happen is could-not-check, which fails this gate for exactly
    # the reason the NOT AVAILABLE and NOT MEASURABLE branches below already
    # fail: a warning is not a gate. It does NOT overwrite a measured breach; a
    # metric that already failed keeps its own, more specific verdict.
    excluded = _size_gate_exclusions(analyzer)

    # Check metrics against thresholds
    failed_metrics = {}
    computed_values = {}

    for metric_name in metrics:
        # CRITICAL (fail closed): a metric the caller ASKED for and the analyzer
        # did not produce was warned about and skipped, after which the gate
        # returned normally. Executed on a model whose disparate impact is 0.00,
        # assert_fairness(metrics=['disparate_impact']) returned {} and did not
        # raise: the user asked for a check, never got it, and was told the run
        # was fine. A requested check that did not happen is could-not-check, and
        # could-not-check fails. A warning is not a gate.
        if metric_name not in all_results:
            available = ", ".join(sorted(all_results)) or "none"
            failed_metrics[metric_name] = {
                "value": None,
                "threshold": default_thresholds.get(metric_name),
                "reason": (
                    f"NOT AVAILABLE: '{metric_name}' was requested but the analyzer did not "
                    f"produce it for this {analyzer.task_type} run, so it was never checked "
                    f"(fail closed). Metrics available here: {available}"
                ),
                "not_measurable": True,
                "ci_lower": None,
                "ci_upper": None,
            }
            continue

        result = all_results[metric_name]

        # Extract value
        if isinstance(result, MetricResult):
            value = result.value
            ci_lower = result.confidence_interval[0]
            ci_upper = result.confidence_interval[1]
        elif isinstance(result, (int, float)):
            value = result
            ci_lower = ci_upper = float("nan")
        else:
            # Same silent skip one step later: a result of an unusable type
            # cannot be compared to a threshold, so the check did not happen.
            failed_metrics[metric_name] = {
                "value": None,
                "threshold": default_thresholds.get(metric_name),
                "reason": (
                    f"NOT AVAILABLE: '{metric_name}' was produced as {type(result).__name__}, "
                    "which cannot be compared against a threshold, so it was never checked "
                    "(fail closed)"
                ),
                "not_measurable": True,
                "ci_lower": None,
                "ci_upper": None,
            }
            continue

        computed_values[metric_name] = value

        # Get threshold.
        #
        # CRITICAL (fail closed): this was
        #     threshold = default_thresholds.get(metric_name, 0.1)
        #     if threshold is None:
        #         threshold = 0.1
        # so a threshold the caller explicitly set to None was REWRITTEN to 0.1
        # and the metric was then graded against a bound nobody chose. Measured
        # 2026-09-10: thresholds={'demographic_parity_ratio': None} on a run
        # whose demographic_parity_ratio is 0.1067 returned normally and did NOT
        # raise, because 0.1067 clears a fabricated floor of 0.1. The four-fifths
        # floor for that metric is 0.80, and the same call with an explicit 0.0
        # correctly reported NOT MEASURABLE (check_threshold refuses an
        # unbreachable minimum) -- so the ONE input that never reached that guard
        # was the one that silently invented a bound.
        #
        # A metric with no bound was never compared to anything, which is
        # could-not-check, and could-not-check fails exactly like the NOT
        # AVAILABLE and NOT MEASURABLE branches around it. The positional 0.1
        # default is gone for the same reason: it is unreachable today (every
        # metric the analyzer emits, classification and regression alike, has an
        # entry in default_thresholds above, and anything else is caught by the
        # NOT AVAILABLE branch), but an unreachable branch that fails OPEN is
        # worse than no branch.
        threshold = default_thresholds.get(metric_name)

        # Check threshold
        passed = True
        failure_reason = None
        not_measurable = False

        if threshold is None:
            passed = False
            not_measurable = True
            failure_reason = (
                f"NOT MEASURABLE: no threshold is configured for '{metric_name}', so the "
                f"measured value {value} was never compared against anything (fail closed). "
                f"Pass thresholds={{'{metric_name}': <bound>}} to gate on it, or drop it "
                f"from `metrics` if it is not being gated on."
            )
        # CRITICAL (fail closed): a non-finite metric was NOT MEASURED, so it
        # cannot satisfy the gate. Every comparison below is False for NaN
        # (abs(nan) > t, nan < t), which is how an unmeasurable run used to
        # return normally and certify a release nobody had checked. Reproduced
        # on real data: 100 rows of one group plus 5 of another, with the
        # default min_group_size=30, leaves a single group above the gate, so
        # there is no pair to compare and the metric returns NaN.
        # Could-not-check is its own state and must FAIL, never pass.
        elif not isinstance(value, (int, float)) or not np.isfinite(value):
            passed = False
            not_measurable = True
            failure_reason = (
                f"NOT MEASURABLE: value is {value}, so the metric was never compared "
                f"against threshold {threshold:.4f} (fail closed)"
            )
        else:
            # Direction comes from the SHARED rule, never from a local reading of
            # the name. The private rule here was two-state (an exact "_ratio"
            # suffix, else lower-is-better) with no could-not-check, so every
            # higher-is-better metric whose name lacks that suffix was graded
            # inverted: executed, "disparate_impact" at 0.10 and
            # "worst_group_accuracy" at 0.10 against a 0.80 floor both PASSED
            # here while check_threshold called both FAIL. An unknown direction
            # is could-not-check and fails; guessing one is what certified a
            # maximal violation as a pass.
            outcome, reason = check_threshold(metric_name, value, threshold)
            if outcome is ThresholdOutcome.FAIL:
                passed = False
                failure_reason = reason
            elif outcome is ThresholdOutcome.COULD_NOT_CHECK:
                passed = False
                not_measurable = True
                failure_reason = f"NOT MEASURABLE: {reason}"

        # Additional CI check if requested
        if include_ci and ci_must_exclude_zero and passed:
            # CRITICAL (fail closed): the same hole in the CI path. A non-finite
            # bound means the interval was never estimated, so the requested
            # "CI must exclude zero" check DID NOT RUN. Skipping it silently
            # let the metric pass on the strength of a check that never
            # happened, so an unavailable interval is could-not-check and fails.
            ci_is_measured = all(
                isinstance(bound, (int, float)) and np.isfinite(bound)
                for bound in (ci_lower, ci_upper)
            )
            if not ci_is_measured:
                passed = False
                not_measurable = True
                failure_reason = (
                    f"NOT MEASURABLE: confidence interval is [{ci_lower}, {ci_upper}], so the "
                    "requested ci_must_exclude_zero check could not run (fail closed)"
                )
            # Check if CI contains zero
            elif ci_lower <= 0 <= ci_upper:
                passed = False
                failure_reason = (
                    f"CI [{ci_lower:.4f}, {ci_upper:.4f}] contains zero "
                    "(disparity not statistically significant)"
                )

        # The size gate removed protected levels from this metric, so the value
        # grades the survivors and nothing else. Could-not-check for part of the
        # protected attribute is not a pass for the attribute (see the block
        # before this loop). Applied last, and only to a metric that would
        # otherwise PASS, so a measured breach keeps its own reason.
        if passed and excluded.groups:
            passed = False
            not_measurable = True
            failure_reason = (
                f"NOT MEASURABLE: min_group_size={min_group_size} excluded "
                f"{len(excluded.groups)} of {len(analyzer.group_sizes)} protected "
                f"group(s) ({excluded.preview}), that is {excluded.n_rows} of "
                f"{excluded.n_rows_total} rows ({excluded.share_percent:.1f} percent), "
                f"from this metric. The value {value:.4f} compares only "
                f"{excluded.covered}, so the excluded level(s) were never checked and "
                f"this gate cannot certify them (fail closed). Lower min_group_size to "
                f"measure them, or drop those rows deliberately."
            )

        if not passed:
            failed_metrics[metric_name] = {
                "value": value,
                "threshold": threshold,
                "reason": failure_reason,
                # Three states, never two: a caller inspecting the raised error
                # must be able to tell a MEASURED violation from a metric that
                # could not be checked at all. Both fail the gate; they are not
                # the same finding.
                "not_measurable": not_measurable,
                "ci_lower": ci_lower if include_ci else None,
                "ci_upper": ci_upper if include_ci else None,
            }

    # Raise assertion if any metrics failed
    if failed_metrics:
        # Build error message
        lines = [
            "Fairness check failed!",
            f"Task type: {analyzer.task_type}",
            f"Groups: {analyzer.groups}",
            f"Group sizes: {analyzer.group_sizes}",
            "",
            "Failed metrics:",
        ]

        for metric_name, failure in failed_metrics.items():
            lines.append(f"  - {metric_name}: {failure['reason']}")
            if failure.get("ci_lower") is not None:
                lines.append(f"    CI: [{failure['ci_lower']:.4f}, {failure['ci_upper']:.4f}]")

        # Name the unmeasurable metrics separately and say what to do about
        # them: "could not check" is a different problem from "measured and too
        # large", and it has a different fix.
        unmeasurable = [name for name, f in failed_metrics.items() if f.get("not_measurable")]
        if unmeasurable:
            lines.extend(
                [
                    "",
                    f"Not measurable ({len(unmeasurable)}): {', '.join(unmeasurable)}",
                    # Says "did not check every protected group", not "were NOT
                    # computed": a metric stopped by the size gate WAS computed,
                    # over the survivors, and telling a reader it was not is the
                    # same defect one layer out. Either way it never checked the
                    # model over the whole protected attribute.
                    "  These metrics did not check the model over every protected group, so",
                    "  they cannot satisfy this gate. This is a could-not-check result, not",
                    "  a pass.",
                    f"  Group sizes: {analyzer.group_sizes} against min_group_size="
                    f"{min_group_size}.",
                    "  A comparison needs at least two groups at or above min_group_size:",
                    "  supply more data for the small groups, lower min_group_size if that is",
                    "  defensible, or drop the metric from the check list on purpose.",
                ]
            )

        # State the size-gate exclusion ONCE, in rows and as a share of the data.
        # "38 group(s) excluded" reads like a rounding detail; "1102 of 2302 rows
        # (47.9 percent)" is the fact that decides whether this verdict means
        # anything, and it is the fact the passing run never showed anyone.
        if excluded.groups:
            lines.extend(
                [
                    "",
                    f"Excluded by min_group_size={min_group_size} "
                    f"({len(excluded.groups)} group(s), {excluded.n_rows} of "
                    f"{excluded.n_rows_total} rows, {excluded.share_percent:.1f} percent): "
                    f"{excluded.preview}.",
                    f"  Every metric above was computed over {excluded.covered} only. The",
                    "  excluded level(s) were not measured, so nothing here certifies them.",
                ]
            )

        if message:
            lines.insert(0, message)
            lines.insert(1, "")

        error_message = "\n".join(lines)

        raise FairnessAssertionError(
            error_message, failed_metrics=failed_metrics, all_metrics=computed_values
        )

    return computed_values


# Helper Functions


def _check_sensitive_attr_column(sensitive_attr: Any, expected: str) -> None:
    """Check that the attribute handed to a callback is the column it names.

    ``create_fairness_callback`` takes the column name as its REQUIRED first
    argument and used to read it nowhere at all, so a callback built to watch
    ``"gender"`` and handed the ``"age_band"`` column reported age disparity
    under a gender label and nothing noticed. The name is the callback's claim
    about what it is monitoring, so it is checked against what actually arrives.

    Three states, never two:

    - the attribute carries a name and it MATCHES: verified, nothing happens;
    - it carries a name and it DIFFERS: refused loudly, because every result
      that follows would be filed under the wrong protected attribute;
    - it carries no name at all (a numpy array, a plain list, an unnamed
      Series): could not check. The call proceeds and NOTHING claims the column
      was verified. A bare array has no column name to compare, and inventing a
      match would be the fabricated-verification this file exists to avoid.

    A DataFrame is checked by membership, not by selection: ``FairnessAnalyzer``
    treats a multi-column frame as an INTERSECTIONAL attribute (groups come out
    as ``"F_young"``, ``"M_old"``), so picking one column here would silently
    narrow the analysis the caller asked for.
    """
    columns = getattr(sensitive_attr, "columns", None)
    if columns is not None:
        names = [str(column) for column in columns]
        if expected not in names:
            raise ValueError(
                f"create_fairness_callback: this callback monitors "
                f"'{expected}', but the DataFrame passed to it has columns "
                f"{names} and none of them is '{expected}'. Fail closed: results "
                "filed under the wrong protected attribute are worse than no "
                "results. Pass the frame holding that column, or build the "
                "callback with the column name you are actually monitoring."
            )
        return

    name = getattr(sensitive_attr, "name", None)
    if name is None:
        # COULD NOT CHECK. Deliberately silent and deliberately not a pass:
        # nothing below records this as verified.
        return

    if str(name) != expected:
        raise ValueError(
            f"create_fairness_callback: this callback monitors '{expected}', but "
            f"it was passed the column '{name}'. Fail closed: the metrics would "
            f"be computed over '{name}' and reported as '{expected}'. Pass the "
            f"'{expected}' column, or build the callback with '{name}'."
        )


def create_fairness_callback(
    sensitive_attr_column: str,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    fail_on_violation: bool = False,
):
    """
    Create a callback function for use with training frameworks.

    Returns a callable that can be used as an evaluation callback
    in frameworks like scikit-learn, XGBoost, or custom training loops.

    Args:
        sensitive_attr_column: Name of the sensitive attribute column this
            callback monitors. Checked at call time against the attribute you
            pass, whenever that attribute carries its own name: a pandas Series
            named something else, or a DataFrame without this column, is refused
            rather than reported under the wrong label. A numpy array or a list
            carries no column name, so there is nothing to check and no check is
            claimed. It never SELECTS a column: a multi-column DataFrame is an
            intersectional attribute here, and narrowing it would change the
            analysis the caller asked for.
        metrics: Names of the metrics this callback returns. If None, every
            metric the analyzer produced is returned. A requested name the
            analyzer did not produce raises: the caller asked to monitor
            something that does not exist for this run, and returning the other
            metrics under that request would be answering a question nobody
            asked. Naming a metric in ``thresholds`` that is not in this list is
            refused when the callback is created.
        thresholds: Thresholds for pass/fail evaluation. Direction comes from
            :func:`._metric_direction.check_threshold`: a higher-is-better metric
            (the ratio family, ``worst_group_accuracy``) reads its threshold as a
            required MINIMUM, a violation magnitude reads it as a maximum
            acceptable absolute value.
        fail_on_violation: If True, raise error on fairness violation. A metric
            that could not be measured (a non-finite value), or whose direction
            is unknown so no threshold could be applied, raises as well: it was
            never checked, so it cannot pass. Requires ``thresholds``: every
            refusal here is conditional on them, so True without them would be a
            gate that can never fire and is refused at construction.

    Returns:
        Callback function

    Example:
        >>> callback = create_fairness_callback('gender')
        >>> # Use with sklearn cross-validation or custom training loop
        >>> for epoch in range(n_epochs):
        ...     train(model)
        ...     metrics = callback(y_val, model.predict(X_val), X_val['gender'])

    Raises:
        ValueError: If ``sensitive_attr_column`` is not a non-empty name, if
            ``thresholds`` names a metric that ``metrics`` excludes, if
            ``fail_on_violation=True`` is passed without thresholds, or if
            ``metrics=[]``. The last three are contradictions: the gate would
            refuse a training run over a metric the same call said not to
            monitor, or it would present itself as a gate that can never fire,
            or it would monitor nothing. All are refused here rather than at the
            first epoch.
    """
    # The column name is this callback's claim about what it monitors, and it is
    # now read (see _check_sensitive_attr_column). An empty or non-string name
    # can never match anything, so it is refused at construction rather than
    # becoming a check that quietly never fires.
    if not isinstance(sensitive_attr_column, str) or not sensitive_attr_column.strip():
        raise ValueError(
            "create_fairness_callback: sensitive_attr_column must be a non-empty "
            f"column name, got {sensitive_attr_column!r}. It names the protected "
            "attribute this callback monitors and is checked against the column "
            "passed at call time."
        )

    # CRITICAL (fail closed): ``fail_on_violation=True`` with no thresholds is a
    # gate with nothing to gate on. The whole threshold block in the callback,
    # INCLUDING the size-gate refusal at its end, sits behind
    # ``if fail_on_violation and thresholds:``, so a falsy ``thresholds`` switched
    # off every refusal the flag was asked for. Measured 2026-09-27 on 400 rows
    # where men are approved 85 percent of the time and women 20 percent:
    #     create_fairness_callback("gender", fail_on_violation=True)
    #       -> returned demographic_parity_difference 0.6951 and did NOT raise,
    #          so training continued through a maximal violation.
    # The same contradiction with ``thresholds={}``. This is the construction-time
    # twin of the ci_must_exclude_zero refusal in assert_fairness: the caller
    # asked for a check, the configuration cannot run one, and reporting no
    # violation is the one answer that must not be given. Refused here rather
    # than at the first epoch.
    if fail_on_violation and not thresholds:
        raise ValueError(
            "create_fairness_callback: fail_on_violation=True needs thresholds to gate "
            "on, and none were given. Every refusal in this callback is conditional on "
            "thresholds, so the callback would monitor the run and never stop it, while "
            "reading as a gate. Pass thresholds={'<metric>': <bound>}, or "
            "fail_on_violation=False to state that this is monitoring only."
        )

    # An EMPTY metric list is the same shape one layer down: the callback would
    # narrow its results to {} and monitor nothing at all, which a caller reads
    # as "no metric reported a problem". metrics=None still means "every metric
    # the analyzer produced".
    if metrics is not None and len(metrics) == 0:
        raise ValueError(
            "create_fairness_callback: metrics=[] asks this callback to monitor nothing, "
            "so it would return an empty result every epoch. Name the metrics to "
            "monitor, or pass metrics=None for every metric the analyzer produces."
        )

    requested_metrics = None if metrics is None else list(dict.fromkeys(metrics))

    # CRITICAL (fail closed): thresholds naming a metric that ``metrics``
    # excludes is a contradiction, and resolving it either way is wrong. Gating
    # on it contradicts the metric list; dropping it silently would leave a
    # training loop believing it was guarded on a metric nobody checked, which
    # is the same defect the "NOT AVAILABLE" branch below exists for. Refuse the
    # configuration up front, before any epoch runs.
    if requested_metrics is not None and thresholds:
        ungated = [name for name in thresholds if name not in requested_metrics]
        if ungated:
            raise ValueError(
                f"create_fairness_callback: thresholds name {', '.join(sorted(ungated))}, "
                f"which metrics={requested_metrics} excludes. A threshold on a metric this "
                "callback does not monitor would either gate on something outside the "
                "requested list or be dropped silently. Add the metric to metrics=, or "
                "drop its threshold."
            )

    def fairness_callback(y_true, y_pred, sensitive_attr):
        from .analyzer import FairnessAnalyzer

        _check_sensitive_attr_column(sensitive_attr, sensitive_attr_column)

        analyzer = FairnessAnalyzer(y_true, y_pred, sensitive_attr)
        results = analyzer.compute_all_metrics()

        if requested_metrics is not None:
            # CRITICAL (fail closed): the caller named the metrics to monitor and
            # the list was read nowhere, so a callback asked for
            # 'disparate_impact' returned the five metrics it happened to compute
            # and the caller read whichever key it found. A requested metric that
            # does not exist for this run is could-not-check, and returning the
            # others in its place answers a question nobody asked.
            missing = [name for name in requested_metrics if name not in results]
            if missing:
                available = ", ".join(sorted(results)) or "none"
                raise ValueError(
                    f"create_fairness_callback: requested metric(s) {', '.join(missing)} "
                    f"were not produced by the analyzer for this {analyzer.task_type} run, "
                    "so they were never computed. Fail closed: an unmonitored metric "
                    f"cannot be reported as monitored. Metrics available here: {available}"
                )
            results = {name: results[name] for name in requested_metrics}

        if fail_on_violation and thresholds:
            for metric_name, threshold in thresholds.items():
                # CRITICAL (fail closed): mirrors the same hole in
                # assert_fairness. A threshold naming a metric the analyzer never
                # produced was skipped by the membership test below, so a
                # training loop asked to stop on a disparate-impact breach was
                # guarded on nothing at all and never said so. A requested check
                # that did not run cannot pass.
                if metric_name not in results:
                    available = ", ".join(sorted(results)) or "none"
                    raise ValueError(
                        f"Fairness violation: NOT AVAILABLE: '{metric_name}' has a threshold "
                        f"({threshold}) but the analyzer did not produce it, so it was never "
                        f"checked. Fail closed: an unchecked metric cannot satisfy a fairness "
                        f"gate. Metrics available here: {available}"
                    )
                if metric_name in results:
                    value = results[metric_name]
                    # A MetricResult carries the point estimate; unwrap it so a
                    # CI-bearing result is checked rather than silently skipped.
                    value = getattr(value, "value", value)

                    # CRITICAL (fail closed): mirrors assert_fairness. A metric
                    # that is non-finite (or not a number at all) was NOT
                    # MEASURED. abs(nan) > threshold is False, so the old guard
                    # let an unmeasurable metric pass a callback whose whole
                    # purpose is to stop training on a violation. Could not
                    # check is not a pass.
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError(
                            f"Fairness violation: {metric_name} could not be measured "
                            f"(value is {value}), so it was never checked against "
                            f"{threshold}. Fail closed: an unmeasurable metric cannot "
                            "satisfy a fairness gate. Check that at least two groups "
                            "meet min_group_size."
                        )
                    # Direction is decided by the SHARED rule, never assumed and
                    # never re-derived here. A ratio metric has a MINIMUM
                    # threshold: abs(ratio) > minimum is the comparison inverted,
                    # so it graded a maximal ratio violation (0.10 against a 0.80
                    # floor) as a pass and a healthy 0.95 as a violation. The
                    # exact "_ratio" suffix test that replaced it fixed only the
                    # names carrying that suffix: "disparate_impact" and
                    # "worst_group_accuracy" are higher-is-better without it, and
                    # both stayed inverted. check_threshold owns this decision
                    # for every surface, and answers COULD_NOT_CHECK rather than
                    # guessing a direction it cannot establish.
                    outcome, reason = check_threshold(metric_name, value, threshold)
                    if outcome is ThresholdOutcome.FAIL:
                        raise ValueError(f"Fairness violation: {reason}")
                    if outcome is ThresholdOutcome.COULD_NOT_CHECK:
                        raise ValueError(
                            f"Fairness violation: {metric_name}={value} could not be checked "
                            f"against {threshold}: {reason} Fail closed: an unchecked metric "
                            "cannot satisfy a fairness gate."
                        )

            # CRITICAL (fail closed): the SIBLING of the size-gate hole fixed in
            # assert_fairness, found by looking for it here. This callback builds
            # its analyzer with the DEFAULT min_group_size=30 and offers no way
            # to change it, so a protected level with fewer members is dropped
            # from every metric and the threshold loop above then graded the
            # survivors. Reproduced 2026-09-10 on 2302 applicants whose 38-level
            # tail (1102 people, 47.9 percent) was denied outright: with
            # thresholds={'demographic_parity_difference': 0.1} and
            # fail_on_violation=True the callback RETURNED 0.0067 and training
            # continued. A level nobody measured is could-not-check, and a gate
            # cannot pass on could-not-check.
            #
            # Checked AFTER the threshold loop, so a metric-specific verdict wins:
            # a measured breach, a NaN that was never measured and an unknown
            # direction each keep their own reason, exactly as assert_fairness
            # keeps a measured breach ahead of this one.
            #
            # Only the GATE arm refuses. With fail_on_violation=False this is a
            # monitoring callback, the metric functions already warn about the
            # drop, and raising there would stop a training run that asked for no
            # gate at all.
            excluded = _size_gate_exclusions(analyzer)
            if excluded.groups:
                raise ValueError(
                    f"Fairness violation: NOT MEASURABLE: min_group_size="
                    f"{analyzer.min_group_size} excluded {len(excluded.groups)} protected "
                    f"group(s) ({excluded.preview}), that is {excluded.n_rows} of "
                    f"{excluded.n_rows_total} rows ({excluded.share_percent:.1f} percent), "
                    f"from every metric here. The thresholds were checked against "
                    f"{excluded.covered} only, so the excluded level(s) were never "
                    f"checked. Fail closed: an unchecked group cannot satisfy a fairness "
                    f"gate. Give the small groups more data, or drop those rows "
                    f"deliberately."
                )

        return results

    return fairness_callback
