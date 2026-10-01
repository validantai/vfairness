"""
Typed structure of the fairness report.

These ``TypedDict`` definitions document the contract returned by
``FairnessAnalyzer.get_report`` and the module-level ``classification_fairness_report``
/ ``regression_fairness_report`` functions, so callers can rely on the key
structure and a type checker can catch a mistyped access (``report["assessement"]``
would now be flagged).

They describe the SAME runtime object the report has always been: a plain,
JSON-serialisable ``dict``. A ``TypedDict`` is a static-only annotation, it adds
no runtime class and changes no value, so every existing consumer keeps working
unchanged, including ``report["assessment"]["summary"]`` subscript access,
``json.dumps(report)``, and the platform task-consumer that reads the report as
JSON. Nothing about the wire format changes.

Stability: the required keys below are part of the public API and follow the
deprecation policy in ``docs/API_STABILITY.md``. The ``NotRequired`` keys appear
only when their feature is requested (confidence intervals, explanations) or for
a specific task type (regression), and are documented as such.
"""

# NB: no ``from __future__ import annotations`` here, so PEP 655 ``NotRequired``
# resolves at class-creation time on every supported interpreter.
from typing import Any, Dict, List, Optional

try:  # ``NotRequired`` is in ``typing`` from 3.11 (the project's Python floor).
    from typing import NotRequired, TypedDict
except ImportError:  # pragma: no cover - only on <3.11, which the package excludes
    from typing_extensions import NotRequired, TypedDict


class MetricStatusEntry(TypedDict):
    """One metric's threshold verdict, as listed in ``assessment``.

    ``status`` is ``"PASS"`` / ``"FAIL"`` for an assessed metric (``threshold`` a
    float), or ``"NOT_ASSESSABLE"`` for an undefined/NaN metric (``threshold`` is
    ``None``, and the entry appears under ``not_assessable_metrics``).
    """

    metric: str
    value: float
    threshold: Optional[float]
    status: str


class InsufficientEvidenceGroup(TypedDict):
    """A protected group dropped from every metric (below ``min_group_size``) and
    therefore excluded from the verdict rather than counted as a silent pass."""

    group: Any
    n: int
    tier: str
    verdict: str
    reason: str


class AssessmentReport(TypedDict):
    """The verdict block, ``report["assessment"]``.

    ``fairness_score`` is the fraction of ASSESSABLE metrics within threshold, or
    ``None`` when nothing was assessable. ``None`` is the third state, "could not
    check": it is NOT the same as 0.0 (measured, and every metric failed) and must
    never be rendered as a percentage. It appears when ``assessable`` is ``False``
    or when no metric produced a threshold verdict at all.

    ``assessable`` is ``False`` on degenerate data (fewer than two valid groups),
    where disparity metrics are vacuous and certify nothing. Every metric entry is
    then listed under ``not_assessable_metrics``, never ``passed_metrics``.

    ``summary`` always ENDS with a data-provenance clause of the form
    ``(data provenance: <final> of <original> rows assessed, <n> excluded,
    missing_strategy='<strategy>')``, so the verdict line itself discloses how the
    rows behind it were handled. It is present on every report, including a
    NOT ASSESSABLE one, because an absent clause would be ambiguous between "no
    rows excluded" and "not recorded". The machine-readable form of the same facts
    is in ``data_info`` (see :class:`DataInfo`); the clause is a disclosure on the
    verdict surface, not a second source of truth. Consumers must treat ``summary``
    as human-readable prose and read ``data_info`` for values.
    """

    fairness_score: Optional[float]
    assessable: bool
    passed_metrics: List[MetricStatusEntry]
    failed_metrics: List[MetricStatusEntry]
    not_assessable_metrics: List[MetricStatusEntry]
    insufficient_evidence_groups: List[InsufficientEvidenceGroup]
    summary: str


class DataInfo(TypedDict, total=False):
    """Dataset and grouping context, ``report["data_info"]``.

    Declared ``total=False`` because the exact key set depends on the task and on
    preprocessing (it also carries through the raw ``info`` fields); the keys
    below are the documented ones produced for every report. ``y_std`` is
    regression-only.

    ``original_size``, ``final_size``, ``n_excluded`` and ``missing_strategy`` are
    the authoritative MISSING-DATA PROVENANCE of the run: how many rows arrived,
    how many were assessed, how many were dropped, and under which strategy. They
    are what makes two runs over the same data with different strategies tellable
    apart, so an auditor can explain a difference in verdict. The same facts are
    restated in prose at the end of ``assessment["summary"]``; these fields are the
    values to read.
    """

    original_size: int
    final_size: int
    n_samples: int
    n_excluded: int
    n_groups: int
    valid_groups: List[Any]
    invalid_groups: List[Any]
    group_sizes: Dict[Any, int]
    is_intersectional: bool
    missing_strategy: str
    y_std: float


class ExplanationsReport(TypedDict, total=False):
    """``report["explanations"]``, present only when FairExplAIner is enabled."""

    metrics: Dict[str, Any]
    statistical: Dict[str, Any]
    summary: str


class FairnessReport(TypedDict):
    """The full fairness report.

    Returned by ``FairnessAnalyzer.get_report`` and by
    ``classification_fairness_report`` / ``regression_fairness_report``. The
    required keys are always present; each ``NotRequired`` key appears only for
    the condition noted beside it.
    """

    task_type: str
    methodology_version: str
    metrics: Dict[str, float]
    group_stats: Dict[str, Any]
    assessment: AssessmentReport
    data_info: DataInfo
    thresholds_used: Dict[str, float]
    # Regression task only:
    residual_bias: NotRequired[Dict[str, Any]]
    # Present when include_ci=True:
    metrics_with_ci: NotRequired[Dict[str, Dict[str, Any]]]
    effect_sizes: NotRequired[Dict[str, Any]]
    statistical_validation: NotRequired[Dict[str, Any]]
    # Present when explanations are enabled. Typed loosely here so the analyzer
    # can assign the FairExplAIner output without a cast; ``ExplanationsReport``
    # above documents the shape a consumer can rely on inside it.
    explanations: NotRequired[Dict[str, Any]]


__all__ = [
    "MetricStatusEntry",
    "InsufficientEvidenceGroup",
    "AssessmentReport",
    "DataInfo",
    "ExplanationsReport",
    "FairnessReport",
]
