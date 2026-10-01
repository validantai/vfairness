"""
Unified FairnessAnalyzer class for vfairness.

This module provides a single unified interface that abstracts the underlying
complexity of fairness calculations. Every number it reports is computed by
vfairness' own native implementations. Delegation to Fairlearn or Aequitas was
described here and typed in the constructor's ``backend`` Literal, but it was
never built (see ``_select_backend``), so the description is gone rather than
the promise being left standing.

FairExplAIner Mode:
    When enabled (fair_explainer=True), the analyzer provides comprehensive
    explanations for each metric, including:
    - What the metric measures
    - How to interpret the values
    - Evaluation based on common benchmarks
    - Actionable recommendations

Example:
    >>> from vfairness import FairnessAnalyzer
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, sensitive_attr)
    >>> results = analyzer.compute_all_metrics()
    >>> print(results['demographic_parity_difference'])

With FairExplAIner:
    >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
    >>> report = analyzer.get_report(include_ci=True)
    >>> # report now includes 'explanations' with detailed explanations
"""

import copy
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, TypeGuard, Union

import numpy as np
import pandas as pd

from ..._methodology import METHODOLOGY_VERSION
from ..._triage import is_measured, unmeasurable_reason
from ...exceptions import ConfigurationError
from ._grouping import GroupManager
from ._metric_direction import MetricDirection, metric_direction
from ._statistics import RECOMMENDED_BOOTSTRAP_SAMPLES, validate_confidence_level
from ._validation import ArrayLike, MissingStrategy, _check_option, validate_inputs
from .report_types import FairnessReport

# Any rendered provenance clause, matched by SHAPE so a wording change cannot
# silently leave two of them in one summary. See the substitution below.
_PROVENANCE_CLAUSE_RE = re.compile(r"\s*\(data provenance:[^)]*\)")


@dataclass
class MetricResult:
    """
    Structured output for a fairness metric.

    Contains the metric value, confidence interval, effect size, and group sizes
    as required by the specification.

    Attributes:
        metric_name: Name of the fairness metric
        value: Point estimate of the metric
        confidence_interval: Tuple of (lower, upper) bounds
        effect_size: Standardized effect size (Cohen's d or risk ratio)
        effect_interpretation: Human-readable interpretation
        group_sizes: Dict mapping group names to sample sizes
        is_fair: Whether the metric passes the threshold. Defaults to False:
            an unpopulated result has not been shown to be fair, and a True
            default certified fairness for a metric nobody ever computed.
        threshold: The threshold used for evaluation
        assessable: Read-only. Whether the metric could actually be measured at
            all (see the property below).
    """

    metric_name: str
    value: float
    confidence_interval: tuple = (float("nan"), float("nan"))
    effect_size: float = float("nan")
    effect_interpretation: str = "not computed"
    group_sizes: Dict[str, int] = field(default_factory=dict)
    # NEVER default this to True. It pairs with verdict="not_computed" below, and a
    # default-constructed MetricResult must not read as a passing fairness result.
    is_fair: bool = False
    threshold: float = 0.1
    # Three-state verdict from the CI (fair / unfair / insufficient_evidence). is_fair is
    # the back-compat boolean == (verdict == "fair"). See _fairness_verdict.
    verdict: str = "not_computed"

    @property
    def assessable(self) -> bool:
        """Whether the metric was actually MEASURED, not merely returned.

        A metric whose between-group comparison never happened (fewer than two
        groups above min_group_size, an undefined per-group rate) returns NaN by
        the convention in classification.py, so a non-finite value is exactly the
        "could not check" state. Deliberately a derived property rather than a
        stored field: a stored flag can be forgotten at a construction site and
        then silently claims the metric was assessed, which is the failure this
        whole convention exists to prevent.

        Note this is a different question from the verdict being conclusive:
        a measured metric with a wide CI is assessable but 'insufficient_evidence'.
        """
        try:
            return bool(np.isfinite(self.value))
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Coerces numpy scalars (np.float64, np.int64, np.bool_) to native
        Python types at this boundary: np.bool_/np.int64 are NOT json
        serializable, so json.dumps(result.to_dict()) crashed before.
        """

        def _float(x: Any) -> Optional[float]:
            return None if x is None else float(x)

        return {
            "metric_name": self.metric_name,
            "value": _float(self.value),
            "confidence_interval": {
                "lower": _float(self.confidence_interval[0]),
                "upper": _float(self.confidence_interval[1]),
            },
            "effect_size": _float(self.effect_size),
            "effect_interpretation": self.effect_interpretation,
            "group_sizes": {str(k): int(v) for k, v in self.group_sizes.items()},
            "is_fair": bool(self.is_fair),
            "threshold": _float(self.threshold),
            "verdict": self.verdict,
            # Carried explicitly so a consumer that never inspects the NaN value
            # (a dashboard, a JSON gate) can still tell "measured and fair" apart
            # from "could not be measured".
            "assessable": bool(self.assessable),
        }


def _fairness_verdict(
    value: float,
    ci_lower: float,
    ci_upper: float,
    threshold: float,
    metric_name: str,
) -> str:
    """Three-state fairness verdict from the confidence interval, not the point estimate.

    A credential must claim 'fair' only on affirmative evidence of practical equivalence
    (the WHOLE CI inside the fair band) and 'unfair' only on significant evidence of
    exceedance (the whole CI outside it); everything in between is 'insufficient_evidence',
    NOT a pass or a fail. This replaces the prior point-estimate rule abs(value) <=
    threshold, which flagged wide-CI small-sample gaps as unfair and tight large-sample
    gaps as fair (spine audit finding 1).

    WHERE THE FAIR BAND IS depends on the metric, and this function used to assume
    [0, threshold] for every metric: 'fair' when hi <= threshold, 'unfair' when
    lo > threshold. For the ratio family the fair band is the other side of the bound
    ([threshold, 1] under the four-fifths rule), so the whole three-state reading,
    the confidence-interval reasoning included, was INVERTED for that family: a
    disparate-impact ratio whose entire interval sat at 0.0 (the protected group is
    never selected) read as 'fair', and perfect parity read as 'unfair'.

    The direction is resolved by the shared
    :func:`._metric_direction.metric_direction`, never by a local name test here.

    THE TWO BRANCHES MUST AGREE ON SIGN. For a lower-is-better metric the fair
    band is the SYMMETRIC ``[-threshold, +threshold]``: these are violation
    MAGNITUDES, and the sign of a max-minus-min gap is an artefact of which
    group the implementation subtracted from which, never evidence of fairness.
    Until 2026-08-27 the point-estimate fallback read it that way
    (``abs(value) <= threshold``) while the interval branch compared the RAW
    bounds, so one function contradicted itself: a CI of ``[-0.50, -0.40]``
    against a 0.10 bound returned 'fair' while its mirror ``[+0.40, +0.50]``
    and the point estimate ``-0.45`` both returned 'unfair'. Every in-repo
    caller happens to feed a non-negative gap, so nothing was mis-graded in
    production, but that is the same accident that kept the defaulted direction
    latent, and it is the identical defect wave 3 fixed in
    ``rendering/adapters_reporting._is_breached``.

    A BOUND THAT CANNOT BE BREACHED GRADES NOTHING, here as in
    :func:`._metric_direction.check_threshold`; see the guard below.

    Args:
        value: Point estimate, used only when there is no usable interval.
        ci_lower, ci_upper: Interval bounds (either order).
        threshold: The bound. A maximum for a lower-is-better metric, a required
            minimum for a higher-is-better one.
        metric_name: Name of the metric, used to resolve the direction.
            REQUIRED. A metric whose direction is unknown yields
            'insufficient_evidence' (could not check), never 'fair'.

    Returns:
        One of 'fair', 'unfair', 'insufficient_evidence'.

    Raises:
        TypeError: when ``metric_name`` is omitted.

    WHY IT IS REQUIRED. Until 2026-08-27 ``metric_name`` defaulted to ``None``
    and that path assumed LOWER_IS_BETTER, i.e. the direction was GUESSED for
    every unnamed call. No in-repo caller was affected (all four name their
    metric), but the next caller to hand this function a ratio without naming
    it would have had its verdict inverted in silence: a disparate-impact ratio
    of 0.05 against the 0.80 four-fifths floor read 'fair'. A defaulted
    direction is the same defect as the substring direction test that has now
    been fixed three times (the gate, the shared helper, a rendering adapter);
    it merely hides in a parameter default, where no source scan for substrings
    can see it. Passing ``None`` explicitly is not a way back in either: it
    resolves to UNKNOWN below and fails closed. ``tests/test_verdict_defaults.py``
    pins both the signature and the fail-closed reading.
    """
    direction = metric_direction(metric_name)

    if direction is MetricDirection.UNKNOWN:
        # We cannot tell which side of the threshold is the fair one, so we cannot
        # place the interval. Could-not-check, never a pass.
        return "insufficient_evidence"

    higher_is_better = direction is MetricDirection.HIGHER_IS_BETTER

    # A bound that CANNOT be breached grades nothing, and 'fair' is the same
    # false certificate here that PASS was in check_threshold: a required
    # minimum of 0.0 on a non-negative ratio is met by every possible value,
    # including a protected group that is never selected at all. Mirrors
    # _metric_direction.check_threshold exactly, so the verdict on a report and
    # the deployment gate's answer for the same bound cannot disagree. The
    # mirror bound is NOT degenerate: 0.0 on a violation magnitude is a real
    # zero-tolerance policy that any non-zero magnitude breaches.
    if higher_is_better and threshold <= 0.0:
        return "insufficient_evidence"
    if not higher_is_better and threshold < 0.0:
        return "insufficient_evidence"

    if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
        # No usable interval. If the point estimate itself is undefined (NaN), the metric
        # could not be measured -> insufficient_evidence, never a spurious 'unfair' (abs(nan)
        # <= t is False) that would defeat the NaN-on-unmeasurable convention.
        if not np.isfinite(value):
            return "insufficient_evidence"
        # Otherwise fall back to the point estimate, best effort, in this metric's
        # own direction.
        if higher_is_better:
            return "fair" if value >= threshold else "unfair"
        return "fair" if abs(value) <= threshold else "unfair"
    lo, hi = (ci_lower, ci_upper) if ci_lower <= ci_upper else (ci_upper, ci_lower)
    if higher_is_better:
        # Fair band is [threshold, ...): the WHOLE interval must sit above the bound.
        if lo >= threshold:
            return "fair"
        if hi < threshold:
            return "unfair"
        return "insufficient_evidence"
    # Lower-is-better means a violation MAGNITUDE, so the fair band is the
    # SYMMETRIC [-threshold, +threshold], exactly as the point-estimate branch
    # above reads it (abs(value) <= threshold) and as check_threshold and
    # adapters_reporting._is_breached read it. Until 2026-08-27 this branch
    # compared the RAW bounds instead, so the two branches of this one function
    # contradicted each other on sign: a CI of [-0.50, -0.40] against a 0.10
    # bound returned 'fair' (hi <= threshold) while its mirror [+0.40, +0.50]
    # returned 'unfair' and the point estimate -0.45 also returned 'unfair'.
    # The sign of a max-minus-min gap is an artefact of which group the
    # implementation subtracted from which, never evidence of fairness; this is
    # the same defect wave 3 fixed in adapters_reporting._is_breached.
    if lo >= -threshold and hi <= threshold:
        # The whole interval sits inside the band: affirmative equivalence.
        return "fair"
    if lo > threshold or hi < -threshold:
        # The whole interval sits outside it: significant exceedance.
        return "unfair"
    return "insufficient_evidence"


# Above this many distinct whole-number predictions, a column reads as a
# measurement (counts, ages, scores) rather than a label set, so regression is
# no longer a guess worth flagging. Below it, {0, 1, 2} could equally be three
# classes, and the caller is told which reading was taken.
_MAX_AMBIGUOUS_INTEGER_LEVELS = 10


def _is_missing_prediction(value: Any) -> bool:
    """True if ``value`` is a MISSING marker (NaN / None / NA), not a prediction.

    A missing prediction is an absence, never a third predicted class. Treating
    it as a class is what let ONE NaN flip a binary classifier into regression
    mode: np.unique() returns NaN alongside 0 and 1, the subset test against
    {0, 1, True, False} then fails, and the analyzer silently reported
    mae_parity_difference / mean_prediction_difference for a 0/1 classifier
    with no warning anywhere (VF-10). DO NOT count missing markers as values.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Arrays and exotic objects: not a scalar missing marker.
        return False


def _defined_unique_predictions(y_pred: ArrayLike) -> List[Any]:
    """Return the unique DEFINED values of y_pred, with missing markers dropped.

    np.unique() is still what produces the candidate values, so any error it
    raises on a caller's input is raised exactly as before; only missing
    markers are dropped afterwards.
    """
    unique_values = np.unique(y_pred)
    return [value for value in unique_values if not _is_missing_prediction(value)]


def _looks_like_label_levels(unique_preds: List[Any]) -> bool:
    """True if the defined values read as a small set of whole-number labels.

    Neither cleanly binary nor cleanly continuous: {0, 1, 2} is as plausibly
    three classes as it is a regression target. The task type is reported
    either way, but the caller is told a judgement was made.
    """
    if not 2 < len(unique_preds) <= _MAX_AMBIGUOUS_INTEGER_LEVELS:
        return False
    try:
        values = np.asarray(unique_preds, dtype=object).astype(float)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)) and np.all(values == np.round(values)))


def _is_nan_number(value: Any) -> bool:
    """True only for a numeric NaN, the ONE unmeasurable value the explainer refuses.

    Deliberately the same test the explainer's own predicate uses (``np.isnan``
    with a non-number falling through to False), because this helper decides
    which values are already handled THERE and must be passed on untouched. If
    the two tests disagreed, a value would be collapsed onto NaN that the
    explainer would have refused on its own, or left alone that it would have
    graded.
    """
    if isinstance(value, (bool, np.bool_)):
        # np.isnan(True) is False, so a flag would be "not NaN" and fall into the
        # substituted branch, which is right: a bool is not a measurement.
        return False
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _for_grading(metric_name: str, value: Any) -> Tuple[Any, Optional[str]]:
    """``(value the explainer may grade, refusal clause)`` for one metric value.

    THE GUARD FOR BOTH EXPLANATION SURFACES, kept in one function because
    ``explain_metric`` and ``get_report`` are two call sites of the same
    ``_get_explainer()`` dispatch and a rule written at one of them leaves the
    other one grading. It sits ABOVE that dispatch, so it cannot be bypassed by
    whichever explainer instance the dispatch returns.

    ``FairExplAIner._could_not_check_reason`` tests ``np.isnan(value)`` alone,
    while the SAME class's ``_finite_or_none`` says in its docstring that
    "NaN/inf mean nothing was measured" and refuses inf for the interval and
    effect-size surfaces. So infinity was graded. Measured at the public entry
    points on 2026-09-27, before this guard:

      analyzer.explain_metric("demographic_parity_ratio", float("inf"))
        severity 'info', evaluation "Excellent! The ratio of inf indicates
        near-perfect parity between groups.", recommendation "Continue
        monitoring this metric ...", no warning
      analyzer.get_explanations() on a two-group regression frame holding one
        infinite prediction: "... | CRITICAL: 4 metric(s) show critical
        disparities requiring immediate attention. | ..." with 'NOT GRADED'
        ABSENT, for four metrics the same report's assessment block calls
        NOT_ASSESSABLE "(infinite: no comparison can grade it)"

    After this guard the same two calls return severity 'could_not_check' with
    "COULD NOT CHECK: demographic_parity_ratio could not be measured on this
    data (infinite: no comparison can grade it) ...", and the summary carries
    "NOT GRADED: 4 metric(s) were never compared to a threshold" with no
    CRITICAL clause. ``check_threshold`` already answered COULD_NOT_CHECK for
    inf, so this is the same module's two halves agreeing again.

    An already-NaN value is handed on UNCHANGED with no clause: the explainer's
    NaN branch is correct, is pinned, and re-authoring its prose from here would
    put two wordings on one state. Only the classes it cannot see (inf, a bool,
    a string, None, an object) are collapsed onto NaN so that its pinned refusal
    runs, and the clause returned beside them restores the reason the collapse
    would otherwise have lost.
    """
    if is_measured(value) or _is_nan_number(value):
        return value, None
    return float("nan"), (
        f"COULD NOT CHECK: {metric_name} could not be measured on this data "
        f"({unmeasurable_reason(value)}), so no threshold was applied to it. "
        f"This is neither a pass nor a fail, and nothing here says it is satisfactory."
    )


def _report_for_grading(
    report: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, str]]]:
    """``(report the explainer may grade, {metric: (true value, refusal clause)})``.

    The report itself is NOT modified: the substitution exists only for the
    grading pass, because the report's own assessment block already refuses these
    values with the right reason ("... could not be measured on this data
    (infinite: no comparison can grade it)") and overwriting the value it names
    would destroy the evidence for that sentence.

    Why the grading pass needs it at all: the summary is built from the SEVERITY
    of each per-metric card, so a card the explainer graded is what put
    "CRITICAL: 4 metric(s) show critical disparities requiring immediate
    attention." on a run whose four metrics were never compared to anything. Fix
    the cards and the summary follows, which is why there is no second rule for
    the summary here.
    """
    gradeable: Dict[str, Any] = dict(report)
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return gradeable, {}

    refusals: Dict[str, Tuple[Any, str]] = {}
    graded_metrics: Dict[str, Any] = {}
    for name, value in metrics.items():
        value_for_grading, clause = _for_grading(str(name), value)
        graded_metrics[name] = value_for_grading
        if clause is not None:
            refusals[str(name)] = (value, clause)
    if refusals:
        gradeable["metrics"] = graded_metrics
    return gradeable, refusals


def _restate_ungradeable_cards(
    explanations: Any, refusals: Dict[str, Tuple[Any, str]]
) -> List[str]:
    """Put the true value and the named reason back on each refused card.

    Returns the metric names restated, so the caller can say so out loud. The
    severity and the recommendation are left exactly as the explainer's pinned
    NaN branch wrote them; a reader of the card sees the third state there, and
    the value beside it is the one the report's metrics block carries, so the two
    halves of one artifact cannot disagree about what was computed.
    """
    restated: List[str] = []
    cards = explanations.get("metrics") if isinstance(explanations, dict) else None
    if not isinstance(cards, dict):
        return restated
    for name, (true_value, clause) in refusals.items():
        card = cards.get(name)
        if isinstance(card, dict):
            card["value"] = true_value
            card["evaluation"] = clause
            restated.append(name)
    return restated


class FairnessAnalyzer:
    """
    Unified fairness analysis interface.

    A single class that wraps vfairness native functions. Users interact only
    with this class, which intelligently handles calculations. Every metric it
    reports is computed natively; there is no delegation to another library.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels/values
        sensitive_attr: Protected attribute(s) for grouping
        y_prob: Optional probability predictions
        task_type: 'classification' or 'regression' (auto-detected if not specified)
        min_group_size: Minimum samples required per group (default: 30)
        backend: 'auto' or 'native'; both run the native implementations.
            'fairlearn' and 'aequitas' are REFUSED with ConfigurationError,
            because no delegation exists to honour them (see _select_backend).

    Example:
        >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        >>>
        >>> # Quick analysis (point estimates only)
        >>> metrics = analyzer.compute_all_metrics()
        >>>
        >>> # Full analysis with confidence intervals
        >>> full_results = analyzer.compute_all_metrics(include_ci=True)
        >>>
        >>> # Access specific metrics
        >>> dp = analyzer.demographic_parity_difference()
        >>> eo = analyzer.equalized_odds_difference()
        >>>
        >>> # Get structured report
        >>> report = analyzer.get_report(include_ci=True)

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

    Ledger row: fairness_analysis. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    # The backends that actually compute something. 'fairlearn' and 'aequitas'
    # were listed here and accepted by the constructor, but no code path ever
    # read the selection: _backend was written once and printed by __repr__,
    # nothing else. Do NOT re-add a name to this list before the delegation
    # that honours it exists; a name here is a claim about which implementation
    # produced the numbers, and that line lands in notebooks and audit logs.
    BACKENDS = ["native"]

    #: Requested-backend values that are typed and documented but unimplemented.
    #: Kept as an explicit set so the refusal can say WHY rather than only "no".
    UNIMPLEMENTED_BACKENDS = ("fairlearn", "aequitas")

    def __init__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sensitive_attr: ArrayLike,
        y_prob: Optional[ArrayLike] = None,
        *,
        task_type: Optional[Literal["classification", "regression"]] = None,
        min_group_size: int = 30,
        missing_strategy: MissingStrategy = "exclude",
        backend: Literal["auto", "native"] = "auto",
        fair_explainer: bool = False,
        cache: bool = False,
    ):
        # Auto-detect task type if not specified.
        #
        # Detection is a GUESS about what the caller meant, so it reads only
        # the DEFINED predictions (missing values are an absence, not a class:
        # see _is_missing_prediction) and it discloses itself whenever the data
        # does not force the answer. It stays SILENT when the answer is forced
        # (a clean 0/1 column, a clean continuous one), because a warning on
        # every ordinary run teaches callers to ignore warnings, and that would
        # cost more than it buys.
        if task_type is None:
            unique_preds = _defined_unique_predictions(y_pred)
            if len(unique_preds) == 0:
                # Nothing defined to detect FROM. Any task type here would be
                # an unsupported claim, so the fallback is named out loud
                # rather than applied silently.
                task_type = "regression"
                warnings.warn(
                    "task_type could not be auto-detected: every value in "
                    "y_pred is missing, so there is no defined prediction to "
                    "infer from. Falling back to task_type='regression'; the "
                    "metrics that follow rest on that fallback, not on the "
                    "data. Pass task_type='classification' or "
                    "task_type='regression' explicitly.",
                    UserWarning,
                )
            elif len(unique_preds) <= 2 and set(unique_preds).issubset({0, 1, True, False}):
                task_type = "classification"
            else:
                task_type = "regression"
                if _looks_like_label_levels(unique_preds):
                    warnings.warn(
                        "task_type auto-detection is ambiguous: y_pred has "
                        f"{len(unique_preds)} distinct whole-number values, "
                        "which read as either a regression target or a "
                        "multiclass label set. Proceeding as "
                        "task_type='regression', so the report will carry "
                        "regression parity metrics. If these are class "
                        "labels, set task_type='classification' explicitly "
                        "(vfairness fairness metrics cover the binary case).",
                        UserWarning,
                    )
                # Probabilities passed as y_pred look continuous, so auto
                # detection silently flipped classification runs into
                # regression mode (wrong metrics, no error). Disclose the
                # inference so the caller can correct it.
                pred_arr = np.asarray(unique_preds, dtype=object)
                try:
                    pred_float = pred_arr.astype(float)
                    looks_like_probabilities = bool(
                        np.all(np.isfinite(pred_float))
                        and pred_float.min() >= 0.0
                        and pred_float.max() <= 1.0
                    )
                except (TypeError, ValueError):
                    looks_like_probabilities = False
                if looks_like_probabilities:
                    warnings.warn(
                        "task_type auto-detected as 'regression' because "
                        f"y_pred has {len(unique_preds)} unique non-binary "
                        "values, but all values lie in [0, 1] and may be "
                        "classification PROBABILITIES. If so, pass "
                        "thresholded 0/1 labels as y_pred (probabilities "
                        "belong in y_prob) or set "
                        "task_type='classification' explicitly.",
                        UserWarning,
                    )

        self.task_type = task_type
        self.min_group_size = min_group_size
        self.missing_strategy = missing_strategy

        # FairExplAIner mode
        self._fair_explainer_enabled = fair_explainer
        self._explainer = None  # Lazy initialization

        # Validate and store data
        self.y_true, self.y_pred, self.sensitive_attr, self.y_prob, self.data_info = (
            validate_inputs(
                y_true,
                y_pred,
                sensitive_attr,
                y_prob,
                task_type=task_type,
                missing_strategy=missing_strategy,
            )
        )

        # Initialize group manager
        self._group_manager = GroupManager(self.sensitive_attr, min_group_size=min_group_size)

        # Determine backend
        self._backend = self._select_backend(backend)

        # Lazy initialization for optional dependencies
        # These will be checked only when actually needed
        self._fairlearn_checked = False
        self._fairlearn_available_cache: Optional[bool] = None
        self._aequitas_checked = False
        self._aequitas_available_cache: Optional[bool] = None

        # Cache for computed metrics, keyed by the argument tuple built in
        # compute_all_metrics()/get_report() (never a plain string key).
        self._cache: Dict[Tuple[Any, ...], Any] = {}
        # Opt-in per-instance memoization of compute_all_metrics() and
        # get_report(), keyed on their arguments (VB-PERF-3). Off by default so
        # behaviour is unchanged and non-deterministic bootstrap runs (random_state
        # None) are never cached implicitly. Cached objects are deep-copied on both
        # store and return, so a caller mutating a result cannot corrupt the cache.
        self._cache_enabled = cache

    def _select_backend(self, backend: str) -> str:
        """Resolve the requested backend to the one that will actually run.

        FAIL CLOSED, and return what EXECUTED, not what was asked for. Until
        2026-08-28 this returned the requested string unchanged, and the value
        was read by exactly one place in the package: ``__repr__``. So
        ``FairnessAnalyzer(..., backend='fairlearn')`` computed the native
        numbers and then printed ``backend='fairlearn'`` in the line that lands
        in a notebook or an audit log, and any string at all was accepted
        (``backend='totally-made-up'`` included) with no error and no warning.
        A parameter that misstates which implementation produced a fairness
        number is a provenance claim the library cannot support.

        There is no delegation code anywhere in the package: the
        ``_fairlearn_available`` / ``_aequitas_available`` probes below are
        never consulted for selection. So a request for one of those backends
        cannot be honoured, and is refused rather than silently downgraded --
        the same fail-closed treatment ``task_type`` and ``missing_strategy``
        already get (see ``_validation._check_option``).

        Raises:
            ConfigurationError: for an unknown backend, or for one that is
                named but unimplemented.
        """
        if isinstance(backend, str) and backend in self.UNIMPLEMENTED_BACKENDS:
            raise ConfigurationError(
                f"backend={backend!r} is not implemented: vfairness has no "
                f"delegation to {backend}, so passing it would have returned "
                "the native numbers under another library's name. Pass "
                "backend='native' (or omit it) to compute them natively, and "
                f"call {backend} directly if you want a second opinion."
            )
        _check_option(backend, "backend", ("auto", "native"))
        # Both accepted values run the same implementation, and the return
        # value is what __repr__ prints, so it names the backend that RAN.
        return "native"

    @property
    def _fairlearn_available(self) -> bool:
        """
        Lazily check if Fairlearn is available.

        Only performs the import check when this property is first accessed,
        not during FairnessAnalyzer initialization. This enables graceful
        degradation when optional dependencies have issues.
        """
        if not self._fairlearn_checked:
            self._fairlearn_checked = True
            try:
                import fairlearn  # noqa: F401  # availability probe

                self._fairlearn_available_cache = True
            except (ImportError, Exception):
                # Catch any exception during import (not just ImportError)
                # This handles cases where the library is installed but broken
                self._fairlearn_available_cache = False
        # The block above always assigns a concrete bool on first access, so
        # the cache is never None once _fairlearn_checked is True.
        assert self._fairlearn_available_cache is not None
        return self._fairlearn_available_cache

    @property
    def _aequitas_available(self) -> bool:
        """
        Lazily check if Aequitas is available.

        Only performs the import check when this property is first accessed,
        not during FairnessAnalyzer initialization. This enables graceful
        degradation when optional dependencies have issues.
        """
        if not self._aequitas_checked:
            self._aequitas_checked = True
            try:
                import aequitas  # noqa: F401  # availability probe

                self._aequitas_available_cache = True
            except (ImportError, Exception):
                # Catch any exception during import (not just ImportError)
                # This handles cases where the library is installed but has
                # dependency conflicts (e.g., fairgbm issues)
                self._aequitas_available_cache = False
        # The block above always assigns a concrete bool on first access, so
        # the cache is never None once _aequitas_checked is True.
        assert self._aequitas_available_cache is not None
        return self._aequitas_available_cache

    def _get_explainer(self):
        """Get or create the FairExplAIner instance (lazy initialization)."""
        if self._explainer is None:
            from .explainer import FairExplAIner

            y_std = np.std(self.y_true) if self.task_type == "regression" else None
            self._explainer = FairExplAIner(task_type=self.task_type, y_std=y_std)
        return self._explainer

    def enable_fair_explainer(self) -> None:
        """
        Enable FairExplAIner mode.

        When enabled, get_report() will include detailed explanations
        for all metrics and statistical measures.

        Example:
            >>> analyzer.enable_fair_explainer()
            >>> report = analyzer.get_report()
            >>> print(report['explanations']['summary'])
        """
        self._fair_explainer_enabled = True

    def disable_fair_explainer(self) -> None:
        """
        Disable FairExplAIner mode.

        Reports will no longer include explanations.
        """
        self._fair_explainer_enabled = False

    @property
    def fair_explainer_enabled(self) -> bool:
        """Check if FairExplAIner mode is enabled."""
        return self._fair_explainer_enabled

    @property
    def groups(self) -> List[str]:
        """Get list of valid groups (meeting min_group_size requirement).

        Note: Groups smaller than min_group_size are excluded. Use group_sizes
        to see all groups regardless of size. If this returns an empty list
        but group_sizes shows groups exist, consider lowering min_group_size.
        """
        return self._group_manager.get_valid_groups(warn_if_empty=False)

    @property
    def all_groups(self) -> List[str]:
        """Get list of all groups (regardless of min_group_size)."""
        return self._group_manager.groups

    @property
    def group_sizes(self) -> Dict[str, int]:
        """Get sample sizes per group."""
        return self._group_manager.get_group_sizes()

    @property
    def n_samples(self) -> int:
        """Get total number of samples."""
        return len(self.y_true)

    def _check_valid_groups(self) -> None:
        """Check if there are valid groups and warn if not."""
        import warnings

        valid_groups = self._group_manager.get_valid_groups(warn_if_empty=False)
        all_groups = self._group_manager.groups

        if len(valid_groups) == 0 and len(all_groups) > 0:
            sizes = self.group_sizes
            max_size = max(sizes.values()) if sizes else 0
            warnings.warn(
                f"No groups meet min_group_size={self.min_group_size}. "
                f"All groups: {list(sizes.keys())} with sizes {list(sizes.values())}. "
                f"No between-group comparison is possible, so metrics return NaN "
                f"(insufficient evidence) and the report marks them NOT_ASSESSABLE. "
                f"To fix: set min_group_size <= {max_size} when creating FairnessAnalyzer.",
                UserWarning,
            )

    def _error_rate_effect(self, metric: Literal["equal_opportunity", "equalized_odds"]):
        """Cohen's h effect size on the rate the metric actually measures.

        equal_opportunity/equalized_odds are TPR/FPR gaps, but the effect
        size previously reported was cohens_d_positive_rate (the SELECTION
        RATE effect), which can read 'negligible' while the metric itself is
        maximal. Here the effect is computed from the extreme pair of the
        same conditional rates the metric compares: the TPR gap for equal
        opportunity, and the larger of the TPR/FPR gaps for equalized odds.
        Cohen's h is used because these are proportions.

        Returns:
            Tuple of (effect_size, interpretation). (nan, 'not computed')
            when fewer than two groups have a defined rate.
        """
        from ._statistics import cohens_h, cohens_h_interpretation

        def _extreme_pair(numerator_mask, denominator_mask):
            rates = self._group_manager.compute_group_rate(
                numerator_mask=numerator_mask, denominator_mask=denominator_mask
            )
            vals = [v for v in rates.values() if not np.isnan(v)]
            if len(vals) < 2:
                return None
            return float(max(vals)), float(min(vals))

        tpr_pair = _extreme_pair((self.y_pred == 1) & (self.y_true == 1), self.y_true == 1)
        if metric == "equal_opportunity":
            pair, label = tpr_pair, "TPR"
        else:
            fpr_pair = _extreme_pair((self.y_pred == 1) & (self.y_true == 0), self.y_true == 0)
            candidates = [
                (p, lab) for p, lab in ((tpr_pair, "TPR"), (fpr_pair, "FPR")) if p is not None
            ]
            if not candidates:
                pair, label = None, ""
            else:
                # The metric is max(TPR gap, FPR gap): report the effect of
                # the gap that drives the metric value.
                pair, label = max(candidates, key=lambda c: c[0][0] - c[0][1])

        if pair is None:
            return float("nan"), "not computed"
        h = cohens_h(pair[0], pair[1])
        return float(h), (f"{cohens_h_interpretation(h)} effect (Cohen's h on {label} gap)")

    # Classification Metrics

    def demographic_parity_difference(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Union[float, MetricResult]:
        """
        Compute demographic parity difference.

        Measures whether positive predictions are made at equal rates across groups.

        Args:
            include_ci: Whether to include confidence interval
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
            random_state: Random seed for reproducibility

        Returns:
            float if include_ci=False, MetricResult if include_ci=True
        """
        from .classification import (
            compute_effect_sizes,
            demographic_parity_difference,
            demographic_parity_difference_with_ci,
        )

        if include_ci:
            result = demographic_parity_difference_with_ci(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                min_group_size=self.min_group_size,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

            # Get effect sizes
            effects = compute_effect_sizes(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

            # Get primary effect size (first pair)
            effect_val = float("nan")
            effect_interp = "not computed"
            if effects:
                first_pair = list(effects.values())[0]
                effect_val = first_pair.get("cohens_d_positive_rate", float("nan"))
                effect_interp = first_pair.get("interpretation", "unknown")

            verdict = _fairness_verdict(
                result.point_estimate,
                result.lower_bound,
                result.upper_bound,
                0.1,
                metric_name="demographic_parity_difference",
            )
            return MetricResult(
                metric_name="demographic_parity_difference",
                value=result.point_estimate,
                confidence_interval=(result.lower_bound, result.upper_bound),
                effect_size=effect_val,
                effect_interpretation=effect_interp,
                group_sizes=self.group_sizes,
                is_fair=(verdict == "fair"),
                threshold=0.1,
                verdict=verdict,
            )
        else:
            return demographic_parity_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

    def equalized_odds_difference(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Union[float, MetricResult]:
        """
        Compute equalized odds difference.

        Measures whether TPR and FPR are equal across groups.

        Args:
            include_ci: Whether to include confidence interval
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
            random_state: Random seed for reproducibility

        Returns:
            float if include_ci=False, MetricResult if include_ci=True
        """
        from .classification import (
            equalized_odds_difference,
            equalized_odds_difference_with_ci,
        )

        if include_ci:
            result = equalized_odds_difference_with_ci(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                min_group_size=self.min_group_size,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

            # Effect from the quantity the metric measures (TPR/FPR gap),
            # not the selection rate. See _error_rate_effect.
            effect_val, effect_interp = self._error_rate_effect("equalized_odds")

            verdict = _fairness_verdict(
                result.point_estimate,
                result.lower_bound,
                result.upper_bound,
                0.1,
                metric_name="equalized_odds_difference",
            )
            return MetricResult(
                metric_name="equalized_odds_difference",
                value=result.point_estimate,
                confidence_interval=(result.lower_bound, result.upper_bound),
                effect_size=effect_val,
                effect_interpretation=effect_interp,
                group_sizes=self.group_sizes,
                is_fair=(verdict == "fair"),
                threshold=0.1,
                verdict=verdict,
            )
        else:
            return equalized_odds_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

    def equal_opportunity_difference(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Union[float, MetricResult]:
        """
        Compute equal opportunity difference.

        Measures whether TPR is equal across groups.

        Args:
            include_ci: Whether to include confidence interval
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
            random_state: Random seed for reproducibility

        Returns:
            float if include_ci=False, MetricResult if include_ci=True
        """
        from .classification import (
            equal_opportunity_difference,
            equal_opportunity_difference_with_ci,
        )

        if include_ci:
            result = equal_opportunity_difference_with_ci(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                min_group_size=self.min_group_size,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

            # Effect from the quantity the metric measures (the TPR gap),
            # not the selection rate. See _error_rate_effect.
            effect_val, effect_interp = self._error_rate_effect("equal_opportunity")

            verdict = _fairness_verdict(
                result.point_estimate,
                result.lower_bound,
                result.upper_bound,
                0.1,
                metric_name="equal_opportunity_difference",
            )
            return MetricResult(
                metric_name="equal_opportunity_difference",
                value=result.point_estimate,
                confidence_interval=(result.lower_bound, result.upper_bound),
                effect_size=effect_val,
                effect_interpretation=effect_interp,
                group_sizes=self.group_sizes,
                is_fair=(verdict == "fair"),
                threshold=0.1,
                verdict=verdict,
            )
        else:
            return equal_opportunity_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

    # Regression Metrics

    def mae_parity_difference(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Union[float, MetricResult]:
        """
        Compute MAE parity difference.

        Measures whether Mean Absolute Error is equal across groups.
        """
        from .regression import (
            compute_regression_effect_sizes,
            mae_parity_difference,
            mae_parity_difference_with_ci,
        )

        if include_ci:
            result = mae_parity_difference_with_ci(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                min_group_size=self.min_group_size,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

            effects = compute_regression_effect_sizes(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

            effect_val = float("nan")
            effect_interp = "not computed"
            if effects:
                first_pair = list(effects.values())[0]
                effect_val = first_pair.get("cohens_d_predictions", float("nan"))
                effect_interp = first_pair.get("interpretation", "unknown")

            # Use relative threshold for regression
            y_std = np.std(self.y_true)
            threshold = y_std * 0.1 if y_std > 0 else 0.1

            verdict = _fairness_verdict(
                result.point_estimate,
                result.lower_bound,
                result.upper_bound,
                threshold,
                metric_name="mae_parity_difference",
            )
            return MetricResult(
                metric_name="mae_parity_difference",
                value=result.point_estimate,
                confidence_interval=(result.lower_bound, result.upper_bound),
                effect_size=effect_val,
                effect_interpretation=effect_interp,
                group_sizes=self.group_sizes,
                is_fair=(verdict == "fair"),
                threshold=threshold,
                verdict=verdict,
            )
        else:
            return mae_parity_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

    # Compute All Metrics

    def compute_all_metrics(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Dict[str, Union[float, MetricResult]]:
        """
        Compute all relevant fairness metrics.

        Automatically selects metrics based on task type.

        Args:
            include_ci: Whether to include confidence intervals
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
            random_state: Random seed

        Returns:
            Dict mapping metric names to values or MetricResult objects

        Warning:
            If fewer than two groups meet the min_group_size threshold, no
            between-group comparison is possible and every metric returns NaN
            (insufficient evidence), never a "perfect" 0.0 / 1.0. A warning is
            issued in this case, and the report marks those metrics
            NOT_ASSESSABLE rather than passed.

        Raises:
            ConfigurationError: If confidence_level is not strictly inside (0, 1).
        """
        # Refuse an out-of-range confidence level here, not only where the
        # interval is built. Same reason the correction method is checked in
        # get_report: OUTSIDE the cache branch, and unconditionally, so
        # include_ci=False cannot smuggle a bad value into a cache key that a
        # later include_ci=True run reads back.
        validate_confidence_level(confidence_level)

        # Check for valid groups and warn if none exist
        self._check_valid_groups()

        cache_key = None
        if self._cache_enabled:
            cache_key = (
                "compute_all_metrics",
                include_ci,
                n_bootstrap,
                confidence_level,
                random_state,
            )
            if cache_key in self._cache:
                return copy.deepcopy(self._cache[cache_key])

        results: Dict[str, Union[float, MetricResult]] = {}

        kwargs: Dict[str, Any] = {
            "include_ci": include_ci,
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "random_state": random_state,
        }

        if self.task_type == "classification":
            results["demographic_parity_difference"] = self.demographic_parity_difference(**kwargs)
            results["equalized_odds_difference"] = self.equalized_odds_difference(**kwargs)
            results["equal_opportunity_difference"] = self.equal_opportunity_difference(**kwargs)

            # Add ratio metrics (point estimates only for now)
            from .classification import demographic_parity_ratio, predictive_parity_difference

            results["demographic_parity_ratio"] = demographic_parity_ratio(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )
            results["predictive_parity_difference"] = predictive_parity_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )
        else:
            # Regression metrics
            results["mae_parity_difference"] = self.mae_parity_difference(**kwargs)

            from .regression import mean_prediction_difference, rmse_parity_difference

            results["rmse_parity_difference"] = rmse_parity_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )
            results["mean_prediction_difference"] = mean_prediction_difference(
                self.y_true, self.y_pred, self.sensitive_attr, min_group_size=self.min_group_size
            )

        # SAY WHICH METRICS CAME BACK UNMEASURABLE, at the surface that hands out
        # the numbers. The values themselves were already honest; the silence was
        # not. Measured 2026-09-27 on a two-group REGRESSION frame carrying one
        # infinite prediction: compute_all_metrics() returned
        # {'mae_parity_difference': inf, 'rmse_parity_difference': inf,
        # 'mean_prediction_difference': inf} with ZERO warnings, while the same
        # object's get_report called all three "could not be measured on this data
        # (infinite: no comparison can grade it)". is_measured is False for every
        # one of them, so the entry point of this class published three
        # unmeasurable numbers and said nothing, and inf is the value that WINS
        # every threshold comparison. It now warns naming each metric and its
        # reason. A single-group frame warned only about PRECISION, a different
        # metric from the five it returned as NaN; that warning is still issued by
        # _check_valid_groups and this one names the five.
        #
        # Emitted on the computing path only: a cache hit returns above and
        # Python's default filter shows a repeat once per site anyway.
        unmeasured: Dict[str, Any] = {}
        for metric_name, metric_value in results.items():
            point = metric_value.value if isinstance(metric_value, MetricResult) else metric_value
            if not is_measured(point):
                unmeasured[metric_name] = point
        if unmeasured:
            warnings.warn(
                f"compute_all_metrics: {len(unmeasured)} of {len(results)} metric(s) could not "
                f"be measured on this data and are NOT comparisons: "
                + ", ".join(
                    f"{name}={value!r} ({unmeasurable_reason(value)})"
                    for name, value in unmeasured.items()
                )
                + ". Do not read any of them as a passing result.",
                UserWarning,
                stacklevel=2,
            )

        if cache_key is not None:
            self._cache[cache_key] = copy.deepcopy(results)
        return results

    def get_report(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
        multiple_testing_correction: Literal["bonferroni", "fdr", "none"] = "none",
        random_state: Optional[int] = None,
        include_explanations: Optional[bool] = None,
    ) -> FairnessReport:
        """
        Generate a comprehensive fairness report.

        Args:
            include_ci: Whether to include confidence intervals
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            multiple_testing_correction: Correction method for multiple comparisons
            random_state: Random seed
            include_explanations: Whether to include FairExplAIner explanations.
                                  If None, uses the fair_explainer_enabled setting.

        Returns:
            Dict containing full report with metrics, group stats, and assessment.
            If FairExplAIner is enabled, also includes 'explanations' section with:
                - metrics: Detailed explanations for each fairness metric
                - statistical: Explanations for CIs and effect sizes
                - summary: Overall fairness assessment summary
        """
        # Same treatment for the confidence level, and for the same reason: an
        # out-of-range value used to reach numpy as a percentile (or, at 1.0,
        # a 4,000,000-permutation loop that never returned).
        validate_confidence_level(confidence_level)

        # Refuse an unrecognised correction method instead of letting it fall
        # through to the default. OUTSIDE the cache branch on purpose: caching is
        # off by default, so a check placed inside it would never run for most
        # callers, which is how this option stayed fail-open while missing_strategy
        # and task_type were already fail-closed. Validated before the cache key is
        # built, so a typo also cannot be cached under its own key.
        _check_option(
            multiple_testing_correction,
            "multiple_testing_correction",
            ("bonferroni", "fdr", "benjamini_hochberg", "none"),
        )

        cache_key = None
        if self._cache_enabled:
            resolved_explain = (
                include_explanations
                if include_explanations is not None
                else self._fair_explainer_enabled
            )
            cache_key = (
                "get_report",
                include_ci,
                n_bootstrap,
                confidence_level,
                multiple_testing_correction,
                random_state,
                resolved_explain,
            )
            if cache_key in self._cache:
                return copy.deepcopy(self._cache[cache_key])

        if self.task_type == "classification":
            from .report import classification_fairness_report

            report = classification_fairness_report(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                y_prob=self.y_prob,
                min_group_size=self.min_group_size,
                include_ci=include_ci,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                multiple_testing_correction=multiple_testing_correction,
                random_state=random_state,
            )
        else:
            from .report import regression_fairness_report

            report = regression_fairness_report(
                self.y_true,
                self.y_pred,
                self.sensitive_attr,
                min_group_size=self.min_group_size,
                include_ci=include_ci,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                multiple_testing_correction=multiple_testing_correction,
                random_state=random_state,
            )

        # The builders above were handed the arrays this analyzer ALREADY cleaned
        # in __init__, so the provenance they derived describes the post-cleaning
        # remainder, not the data the user handed in. Restate the truth (which
        # __init__ recorded in self.data_info) before anything downstream reads
        # it: the explanations below, the deepcopy stored in the cache, and the
        # report returned to the caller all see the corrected block, so the cache
        # can never serve the stale numbers. See _apply_true_provenance.
        report = self._apply_true_provenance(report)

        # Add explanations if FairExplAIner is enabled
        should_explain = (
            include_explanations
            if include_explanations is not None
            else self._fair_explainer_enabled
        )
        if should_explain:
            explainer = self._get_explainer()
            # THE SAME GUARD AS explain_metric, above the same _get_explainer
            # dispatch, so the two surfaces cannot disagree about which values may
            # be graded. See _for_grading and _report_for_grading. Measured on a
            # two-group regression frame holding one infinite prediction,
            # 2026-09-27: before, get_explanations()["summary"] read "Overall
            # Fairness Score: NOT AVAILABLE (could not check) (0 passed, 0 failed,
            # 4 not assessable) | CRITICAL: 4 metric(s) show critical disparities
            # requiring immediate attention. | COULD NOT CHECK: ..." with no NOT
            # GRADED clause at all; after, the CRITICAL clause is gone and
            # "NOT GRADED: 4 metric(s) were never compared to a threshold, so they
            # are neither passing nor failing." is present.
            gradeable_report, refusals = _report_for_grading(report)
            report["explanations"] = explainer.explain_report(
                gradeable_report, include_statistical=include_ci
            )
            # THE PROVENANCE CLAUSE BELONGS ON THIS SURFACE TOO, and this is the
            # only place it can be put: report['explanations'] does not exist when
            # _apply_true_provenance runs. get_explanations() returns THIS dict and
            # nothing else, so a clause on assessment['summary'] alone is invisible
            # to every caller of it. See _apply_true_provenance for the measured
            # before-state (100.0% and "Continue regular monitoring" on a frame
            # where a third of the rows had no protected attribute).
            from .report import _missing_data_clause as _clause_for_explanations

            _info = report.get("data_info")
            self._restate_provenance_in(
                report.get("explanations"),
                _clause_for_explanations(dict(_info)) if isinstance(_info, dict) else "",
            )
            restated = _restate_ungradeable_cards(report["explanations"], refusals)
            if restated:
                # The cards and the summary say it too, but a caller that reads
                # neither (a gate polling the report dict) gets one line naming
                # the metrics that were never graded. An ungraded metric is not a
                # passing one.
                warnings.warn(
                    f"get_report: {len(restated)} metric value(s) cannot be graded and were "
                    f"NOT compared to any threshold: "
                    + ", ".join(
                        f"{name}={refusals[name][0]!r} ({unmeasurable_reason(refusals[name][0])})"
                        for name in restated
                    )
                    + ". Their explanation cards read COULD NOT CHECK, which is neither a "
                    "pass nor a fail.",
                    UserWarning,
                    stacklevel=2,
                )

        # Stamp the methodology version so a rating records the methodology that
        # produced it, reproducible independently of the code version (VB-DOC-1).
        report["methodology_version"] = METHODOLOGY_VERSION

        if cache_key is not None:
            self._cache[cache_key] = copy.deepcopy(report)
        return report

    def _apply_true_provenance(self, report: FairnessReport) -> FairnessReport:
        """Overlay the analyzer's TRUE missing-data provenance onto a built report.

        ``__init__`` runs ``validate_inputs`` once and keeps BOTH the cleaned
        arrays and the true ``data_info`` for the data the user actually handed
        in. ``get_report`` then hands those already-cleaned arrays to the report
        builder, which validates again: there is nothing left to drop, so the
        builder recorded ``original_size`` as the POST-exclusion count and
        ``n_excluded`` as 0, and it never saw ``missing_strategy`` at all (it
        used its own ``'exclude'`` default). On 120 rows of which 15 carried a
        missing prediction, the report positively asserted "105 of 105 rows
        assessed, 0 excluded" on a run that dropped 15, and an analyzer built
        with ``as_group`` produced a report naming ``'exclude'``. That is not an
        absent fact, it is a false one: a report that misstates its own row
        accounting cannot be audited, and two runs over the same data under
        different strategies differ with nothing in the artifact explaining why.

        WHY AN OVERLAY AND NOT A RE-VALIDATION. The other repair available here
        is to keep the user's ORIGINAL arrays and pass those plus
        ``self.missing_strategy`` to the builder, letting it re-derive the
        provenance. It was rejected: it changes the builder's INPUT, so every
        field of the report (metrics, group stats, CIs) is recomputed from
        something other than what is computed today, and it re-reads arrays the
        caller still owns and may have mutated since construction, which would
        silently change WHICH ROWS ARE ANALYSED. The overlay leaves the builder
        input byte-identical, so no metric, group stat or other field can move;
        only the four provenance fields and their restatement in the summary
        change, which is exactly the scope of this repair.

        BOTH SURFACES ARE CORRECTED, not just the structured one. The provenance
        is restated in prose at the end of ``assessment["summary"]``, and that
        clause is what most readers and downstream consumers see. Fixing
        ``data_info`` alone would leave the false sentence standing on the
        verdict line, so the stale clause (regenerated verbatim with the same
        helper the builder used) is replaced by the true one.

        FAIL CLOSED ON AN ACCOUNTING THAT DOES NOT RECONCILE. ``final_size`` and
        ``n_samples`` are left exactly as the builder computed them, because they
        describe the rows the metrics were actually computed over. Every
        ``validate_inputs`` branch guarantees
        ``original_size - n_excluded == final_size``; if that does not hold
        against the builder's ``final_size``, the two accounts cannot both be
        true and we do not know which is, so the counts are recorded as ``None``
        (rendered "unknown" by ``_missing_data_clause`` and ``print_report``)
        rather than published as a number nobody can stand behind.
        """
        from .report import _missing_data_clause

        info = report.get("data_info")
        if not isinstance(info, dict):
            # Nothing to overlay onto. Never fabricate the block: an absent
            # data_info is a could-not-check, not a clean run.
            return report

        original_size = self.data_info.get("original_size")
        n_excluded = self.data_info.get("n_excluded")
        strategy = self.data_info.get("missing_strategy", self.missing_strategy)
        final_size = info.get("final_size")

        def _is_count(value: Any) -> TypeGuard[int]:
            # np.integer counts as a count: every producer here returns a plain
            # int today, and treating a numpy one as "not a number" would report
            # unknown for provenance we actually hold. bool is excluded because
            # True would otherwise arithmetic its way into a row count.
            return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

        reconciles = (
            _is_count(original_size)
            and _is_count(n_excluded)
            and _is_count(final_size)
            and original_size - n_excluded == final_size
        )
        if not reconciles:
            original_size = None
            n_excluded = None

        # dict(info) because a TypedDict is not a dict[str, Any] to a type
        # checker, though it is one at runtime. The clause only reads.
        stale_clause = _missing_data_clause(dict(info))
        # EVERY FIELD THE BUILDER DERIVED FROM THE CLEANED ARRAYS, not the three
        # this overlay first listed. B4 tier-1 audit, 2026-09-30, and it is the
        # copy-constructor shape: a field-by-field rebuild carries the fields that
        # existed the day it was written.
        #
        # ``coverage`` is the share of the input the answer rests on, and the
        # builder computed len(cleaned)/len(cleaned), so it published the CONSTANT
        # 1.0 in a field named coverage and could not disagree with anything.
        # Measured before this change: 5 of 80 rows excluded -> coverage 1.0;
        # 40 of 120 -> 1.0; 120 of 200 -> 1.0; 0 excluded -> 1.0, while
        # validate_inputs had computed the real 0.9375, 0.6667 and 0.4 and none of
        # them reached data_info. It is DERIVED from the two counts published right
        # here rather than copied from self.data_info, so it can never disagree with
        # the numbers beside it, and it is None (not 1.0) when they do not
        # reconcile or the input was empty.
        #
        # ``n_groups_before`` and ``groups_dropped`` are stale the same way and the
        # consequence is louder: measured on 40 'a' + 40 'b' + 40 'c' where every
        # one of c's predictions is missing, the published report said
        # n_groups_before 2 and groups_dropped [], i.e. a whole protected group lost
        # every row and the artifact asserted that none was dropped, while
        # self.data_info held n_groups_before 3 and groups_dropped ['c'].
        if reconciles and original_size:
            true_coverage: Optional[float] = final_size / original_size
        else:
            true_coverage = None
        true_info = {
            **info,
            "original_size": original_size,
            "n_excluded": n_excluded,
            "missing_strategy": strategy,
            "coverage": true_coverage,
            "n_groups_before": self.data_info.get("n_groups_before", info.get("n_groups_before")),
            "n_groups_after": self.data_info.get("n_groups_after", info.get("n_groups_after")),
            "groups_dropped": self.data_info.get("groups_dropped", info.get("groups_dropped")),
        }
        true_clause = _missing_data_clause(true_info)
        report["data_info"] = true_info  # type: ignore[typeddict-item]

        # BOTH PROSE SURFACES, not only the assessment one. B4 tier-1 audit,
        # 2026-09-30. get_explanations() returns report['explanations'] and nothing
        # else, and the provenance clause was restated only on
        # assessment['summary'], one key over. Measured on 120 rows of which 40
        # carried no protected attribute, both groups above the floor and y_true ==
        # y_pred so the labelled part is genuinely fair:
        #   get_explanations()['summary'] = 'Overall Fairness Score: 100.0% (5
        #     passed, 0 failed) | All assessable metrics are within acceptable
        #     thresholds. Continue regular monitoring to ensure ongoing fairness.'
        #   report['assessment']['summary'] = '5/5 metrics within thresholds (data
        #     provenance: 80 of 120 rows assessed, 40 excluded for missing values,
        #     ...)'
        # A third of the rows had no protected attribute and the surface a reader
        # of get_explanations gets said "Continue regular monitoring". A token scan
        # of the WHOLE explanations payload for 'excluded', 'coverage', 'missing',
        # 'nan', 'not_assessable' and 'omit' found none of them, at 6.25, 33 and 60
        # percent excluded alike.
        #
        # Only the assessment summary can be restated HERE: report['explanations']
        # does not exist yet when this runs, it is built a few lines further up in
        # get_report, so it carries its own call to the same helper right after it
        # is built. One helper, two call sites, never two copies of the logic.
        self._restate_provenance_in(report.get("assessment"), true_clause, stale_clause)
        return report

    @staticmethod
    def _restate_provenance_in(
        section: Any, true_clause: str, stale_clause: Optional[str] = None
    ) -> None:
        """Put the TRUE provenance clause on one prose summary, in place.

        ``stale_clause`` is an optional exact string to swap out when the
        shape-matching pattern misses. It is Optional and not "" because
        ``"" in summary`` is True for every summary and ``summary.replace("", ...)``
        PREPENDS, so an empty sentinel would have quietly put the clause at the
        front of the sentence.
        """
        if isinstance(section, dict):
            summary = section.get("summary")
            if isinstance(summary, str) and true_clause not in summary:
                # Replace ANY existing provenance clause, matched by SHAPE and
                # not by exact text. Exact matching made this brittle in the
                # worst direction: the clause the summary already carried was
                # built from a smaller info dict, so the moment the renderer
                # gained a field the two strings diverged, the replace missed,
                # and the report ended up carrying BOTH clauses, the stale one
                # first. A reader then sees "105 of 105 rows assessed" and
                # "105 of 120 rows assessed" in one sentence, which is worse
                # than either alone. Measured 2026-09-10 when the clause grew
                # its group-size-floor disclosure.
                replaced, n_subs = _PROVENANCE_CLAUSE_RE.subn(
                    # true_clause carries its own leading space and the
                    # pattern consumed the old one, so do NOT strip here:
                    # that ran the clause into the sentence before it.
                    lambda _m: true_clause,
                    summary,
                    count=1,
                )
                if n_subs:
                    section["summary"] = replaced
                elif stale_clause is not None and stale_clause in summary:
                    section["summary"] = summary.replace(stale_clause, true_clause, 1)
                else:
                    # The summary carried no provenance clause we recognise, so
                    # appending the true one cannot contradict anything already
                    # stated there.
                    section["summary"] = summary + true_clause

    def clear_cache(self) -> None:
        """Clear the per-instance result cache (see the ``cache`` constructor argument)."""
        self._cache.clear()

    def explain_metric(self, metric_name: str, value: Optional[float] = None) -> Any:
        """
        Get a detailed explanation for a specific metric.

        If value is not provided, computes the metric first.

        Args:
            metric_name: Name of the metric to explain
            value: Optional pre-computed value

        Returns:
            MetricExplanation object with full context

        Example:
            >>> explanation = analyzer.explain_metric('demographic_parity_difference')
            >>> print(explanation)
        """

        # Compute value if not provided. The resolved value may be a plain
        # float (metric methods with include_ci=False) or a MetricResult
        # (from compute_all_metrics), so it is held as Any rather than
        # narrowing the Optional[float] parameter.
        resolved_value: Any = value
        if value is None:
            if hasattr(self, metric_name):
                resolved_value = getattr(self, metric_name)()
            else:
                metrics = self.compute_all_metrics()
                resolved_value = metrics.get(metric_name, float("nan"))

        # A structured result carries its point estimate in `.value`, and the
        # explainer's could-not-check test cannot read one: measured 2026-09-27,
        # explain_metric(name, MetricResult(value=0.2, ...)) came back severity
        # 'could_not_check' with "this metric was never computed", for a metric
        # measured at 0.2000, and its to_dict() then carried a MetricResult in
        # the "value" field, which json.dumps cannot encode. That is a
        # MEASUREMENT reported as a could-not-check, the mirror of the defect
        # below and the direction that quietly discards evidence. Unwrap it here,
        # above the guard, so one value reaches one rule.
        if isinstance(resolved_value, MetricResult):
            resolved_value = resolved_value.value

        explainer = self._get_explainer()
        # The guard above the dispatch. See _for_grading: the explainer grades
        # everything that is not NaN, so inf was "Excellent! The ratio of inf
        # indicates near-perfect parity between groups." at severity 'info'.
        gradeable, refusal = _for_grading(metric_name, resolved_value)
        explanation = explainer.explain_metric(
            metric_name, gradeable, group_stats=self._group_manager.get_group_sizes()
        )
        if refusal is None:
            return explanation
        # dataclasses.replace, never a field-by-field rebuild: a rebuild lists
        # the fields that existed the day it was written and silently drops every
        # one added later (benchmark_context, related_metrics), which is how a
        # three-state disclosure gets lost one layer up. The severity and the
        # recommendation are the explainer's own, produced by its pinned NaN
        # branch; only the value and the REASON are restored here, because
        # collapsing onto NaN is what loses them.
        return replace(explanation, value=resolved_value, evaluation=refusal)

    def get_explanations(
        self,
        include_ci: bool = False,
        n_bootstrap: int = RECOMMENDED_BOOTSTRAP_SAMPLES,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Get explanations for all metrics without generating a full report.

        This is a convenience method for quickly getting explanations.

        Args:
            include_ci: Whether to include CI explanations
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            Dictionary with explanations for all metrics
        """
        report = self.get_report(
            include_ci=include_ci,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            include_explanations=True,
        )
        return report.get("explanations", {})

    # Library Comparison / Delegation

    def compare_with_fairlearn(self) -> Optional[Dict[str, Any]]:
        """
        Compare results with Fairlearn (if available).

        Returns:
            Dict with Fairlearn metric values, or None if not available.
            On failure the dict contains a single ``'error'`` string entry.

        THREE STATES, added 2026-09-25, and the defect it closes was measured at
        this entry point. With fewer than two comparable groups there is no
        disparity for either library to measure, and this method handed back
        fairlearn's conventional ``0.0`` with no warning at all:

            FairnessAnalyzer(y, y, np.array(['a'] * 60)).compare_with_fairlearn()
            -> {'fairlearn_demographic_parity_difference': 0.0, ...}

        A reader sees perfect parity. The same object's own
        ``compute_all_metrics()`` returns ``nan`` for exactly that input and warns,
        so the library refused and its cross-library comparison did not, which is
        the worse of the two to get wrong: the number carries another project's
        name and reads as independent corroboration.

        The metric keys are now ``nan`` when no comparison is defined, and
        fairlearn's own convention is kept under ``fairlearn_reported_*`` rather
        than dropped, because the point of this method is to show what the
        reference library says. The degeneracy test is the analyzer's OWN
        ``get_valid_groups``, not a fresh rule invented here, so the two can never
        disagree about what is comparable.
        """
        if not self._fairlearn_available:
            return None

        try:
            from fairlearn.metrics import demographic_parity_difference as fl_dp_diff
            from fairlearn.metrics import equalized_odds_difference as fl_eo_diff

            def _fairlearn_on(mask: np.ndarray) -> Dict[str, Any]:
                return {
                    "fairlearn_demographic_parity_difference": fl_dp_diff(
                        self.y_true[mask],
                        self.y_pred[mask],
                        sensitive_features=self.sensitive_attr[mask],
                    ),
                    "fairlearn_equalized_odds_difference": fl_eo_diff(
                        self.y_true[mask],
                        self.y_pred[mask],
                        sensitive_features=self.sensitive_attr[mask],
                    ),
                }

            everything = np.ones(len(self.sensitive_attr), dtype=bool)
            reported = _fairlearn_on(everything)
            valid = self._group_manager.get_valid_groups(warn_if_empty=False)
            excluded = {
                group: size
                for group, size in self._group_manager.get_group_sizes().items()
                if group not in set(valid)
            }
            if len(valid) >= 2:
                if not excluded:
                    return reported
                # THE SAME VERDICT THAT GATES THE REFUSAL MUST GATE THE POPULATION.
                # B4 tier-1 audit, 2026-09-30. The guard below covers len(valid) < 2
                # only. With two or more comparable groups this returned fairlearn's
                # number computed over ALL of self.sensitive_attr, including the
                # groups the analyzer itself REFUSES to compare, and fairlearn's
                # demographic_parity_difference is a max-min spread, so a group under
                # the floor can BE the max or the min and set the whole number.
                # Measured before this change, 100 'a' + 100 'b' + 3 'c' at
                # min_group_size=50, where 'c' is predicted 1 always and a and b sit
                # near 0.5: get_valid_groups() -> ['a', 'b'], i.e. 'c' is refused, and
                # fairlearn_demographic_parity_difference came back 0.5, entirely set
                # by 'c', against 0.0407 over the population the analyzer actually
                # assesses. Zero warnings, no not_comparable key and nothing naming
                # the excluded group: a cross-library number that reads as
                # independent corroboration of a 50-point gap that the library's own
                # metrics never claimed. Reproduced at four size combinations
                # (100/100/3, 60/60/5, 200/150/10, 40/40/7).
                # The comparison is now computed over the SAME rows the analyzer's own
                # metrics use, the all-groups numbers are kept as evidence rather than
                # dropped, and the excluded groups are named with their sizes.
                keep = np.isin(self.sensitive_attr, list(valid))
                out: Dict[str, Any] = _fairlearn_on(keep)
                out["groups_not_measured"] = dict(excluded)
                out["population_note"] = (
                    f"computed over the {int(keep.sum())} row(s) in the "
                    f"{len(valid)} group(s) meeting min_group_size="
                    f"{self.min_group_size}, which is the same population this "
                    f"analyzer's own metrics use. {len(excluded)} group(s) "
                    f"{sorted(map(str, excluded))} with sizes "
                    f"{ {str(k): v for k, v in excluded.items()} } are BELOW that "
                    f"floor and are not part of this comparison. The all-groups "
                    f"numbers fairlearn reports for the same frame are kept under the "
                    f"fairlearn_reported_all_groups_* keys and are NOT a comparison "
                    f"over the assessed population."
                )
                for key, value in reported.items():
                    out[key.replace("fairlearn_", "fairlearn_reported_all_groups_")] = value
                warnings.warn(
                    f"compare_with_fairlearn: {len(excluded)} group(s) "
                    f"{sorted(map(str, excluded))} (sizes "
                    f"{ {str(k): v for k, v in excluded.items()} }) are below "
                    f"min_group_size={self.min_group_size} and are NOT compared, so "
                    f"the cross-library numbers are computed over the "
                    f"{int(keep.sum())} row(s) this analyzer actually assesses. "
                    f"Fairlearn's all-groups figures are kept under "
                    f"fairlearn_reported_all_groups_* and are not a comparison over "
                    f"that population: a group under the floor can set a max-min "
                    f"spread on its own.",
                    UserWarning,
                    stacklevel=2,
                )
                return out

            warnings.warn(
                f"compare_with_fairlearn: {len(valid)} comparable group(s) "
                f"(min_group_size={self.min_group_size}), so no disparity is defined "
                "and no cross-library comparison is possible. The metric values are "
                "nan, which means could not check. Fairlearn's own convention for "
                "this input is kept under the fairlearn_reported_* keys and is NOT "
                "evidence of parity.",
                UserWarning,
                stacklevel=2,
            )
            out: Dict[str, Any] = {
                "fairlearn_demographic_parity_difference": float("nan"),
                "fairlearn_equalized_odds_difference": float("nan"),
                "not_comparable": (
                    f"{len(valid)} group(s) meet min_group_size={self.min_group_size}; "
                    "a disparity between groups needs at least two"
                ),
            }
            for key, value in reported.items():
                out[key.replace("fairlearn_", "fairlearn_reported_")] = value
            return out
        except Exception as e:
            return {"error": str(e)}

    def compare_with_aequitas(self) -> Optional[Dict[str, Any]]:
        """
        Compare results with Aequitas (if available).

        Returns:
            Dict with Aequitas audit results, or None if not available.
            On failure the dict contains a single ``'error'`` string entry.

        THREE STATES, carried across from ``compare_with_fairlearn`` on
        2026-09-29. The guard the twin gained on 2026-09-25 was never applied
        here, and the before-state was measured at THIS entry point with a
        stubbed aequitas, because the guard needs no aequitas installed:

            FairnessAnalyzer(y, y, np.array(['a'] * 60)).compare_with_aequitas()
            -> {'attribute_value': {0: 'a'}, 'ppr_disparity': {0: 1.0},
                'fpr_disparity': {0: 1.0}}

        with zero warnings and no ``not_comparable`` key, while that same
        object's own ``demographic_parity_difference()`` is ``nan`` and the twin
        returned ``nan`` plus ``not_comparable`` plus exactly one warning. This
        entry point's before-state was the worse of the two: ``ref_groups_dict``
        is ``self.groups[0]``, so on a one-group frame the only row in the table
        is a group compared with ITSELF, whose disparity is 1.0 BY CONSTRUCTION
        and reads as full four-fifths compliance under another project's name.

        So: the ``*_disparity`` columns are ``nan`` when no comparison is
        defined, aequitas' own output is kept whole under
        ``aequitas_reported_disparities`` rather than dropped (withholding the
        verdict must not destroy the evidence), and the reference group is named
        on EVERY return, measured or not, because its own row is a
        self-comparison in both states. The degeneracy test is the analyzer's
        OWN ``get_valid_groups``, the same call the twin makes, so the two can
        never disagree about what is comparable.

        One boundary further out, and found by the test written for the guard:
        with NO group meeting ``min_group_size`` there is no reference group at
        all, ``self.groups[0]`` raised IndexError, and the broad ``except``
        turned that into ``{'error': 'list index out of range'}`` where the twin
        says "0 group(s) meet min_group_size=1000". That state is now refused
        above the dispatch, before anything is asked of aequitas.
        """
        if not self._aequitas_available:
            return None

        # THE GUARD ABOVE THE DISPATCH. `self.groups` is the VALID groups, so
        # with no group at all meeting min_group_size `self.groups[0]` raised
        # IndexError, the broad except below swallowed it, and the caller got
        # `{'error': 'list index out of range'}`: an opaque failure where the
        # twin says "0 group(s) meet min_group_size=1000". There is also no
        # reference group to name in that state, so nothing is asked of aequitas.
        valid = self._group_manager.get_valid_groups(warn_if_empty=False)
        if not valid:
            sizes = self._group_manager.get_group_sizes()
            warnings.warn(
                f"compare_with_aequitas: no group meets min_group_size="
                f"{self.min_group_size} (group sizes: {sizes}), so there is no "
                "reference group to compare against and no audit was requested from "
                "aequitas. This is could not check, not parity.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "not_comparable": (
                    f"0 group(s) meet min_group_size={self.min_group_size}; "
                    "a disparity between groups needs at least two"
                ),
                "reference_group": None,
                "reference_group_note": (
                    "no group meets min_group_size, so no reference group exists and "
                    "aequitas was not called"
                ),
                "group_sizes": sizes,
            }

        try:
            import pandas as pd
            from aequitas.bias import Bias
            from aequitas.group import Group

            # THE SAME VERDICT THAT GATES THE REFUSAL MUST GATE THE ROWS.
            # B4 tier-1 audit, 2026-09-30. The guard below covers len(valid) < 2
            # only. With two or more comparable groups this built the frame from ALL
            # of self.sensitive_attr and returned aequitas' table verbatim, so it
            # carried a disparity number for every group the analyzer itself REFUSES
            # to compare, with no not_comparable key, no warning and nothing naming
            # the excluded group or the threshold that excluded it.
            # Measured (aequitas is not installed here, so with a stub that records
            # the frame it is handed and computes a real selection-rate ratio out of
            # it), 100 'a' + 100 'b' + 3 'c' at min_group_size=50:
            #   get_valid_groups() -> ['a', 'b'] and get_group_sizes() ->
            #   {'a': 100, 'b': 100, 'c': 3}, i.e. the analyzer has REFUSED 'c'
            #   the frame handed to aequitas held ['a', 'b', 'c']
            #   the returned dict published attribute_value {0:'a',1:'b',2:'c'} and a
            #   ppr_disparity for index 2, with 'not_comparable' in out FALSE and its
            #   own warnings []
            # Reproduced at four size combinations (100/100/3, 60/60/5, 200/150/10,
            # 40/40/7). The two load-bearing facts are stub-independent: the
            # population handed to the library included groups get_valid_groups
            # refused, and the >= 2 branch returned that table verbatim.
            # The excluded rows are dropped BEFORE aequitas sees them, so no row of
            # the table is a group the analyzer will not compare, and the exclusion
            # is named on the return rather than left for a reader to notice.
            keep = np.isin(self.sensitive_attr, list(valid))
            excluded = {
                group: size
                for group, size in self._group_manager.get_group_sizes().items()
                if group not in set(valid)
            }

            # Prepare data for Aequitas
            df = pd.DataFrame(
                {
                    "score": self.y_pred[keep],
                    "label_value": self.y_true[keep],
                    "sensitive_attr": self.sensitive_attr[keep],
                }
            )

            g = Group()
            xtab, _ = g.get_crosstabs(df, attr_cols=["sensitive_attr"])

            b = Bias()
            reference = valid[0]
            bdf = b.get_disparity_predefined_groups(
                xtab, original_df=df, ref_groups_dict={"sensitive_attr": reference}, alpha=0.05
            )

            reported: Dict[str, Any] = bdf.to_dict()
            # Named on BOTH returns, not only the refusal: the reference group's
            # own row holds 1.0 in every *_disparity column because a group is
            # compared with itself. That is a definition, not a measurement, and
            # a reader of the table cannot tell which row it is unless it is
            # stated here.
            disclosure: Dict[str, Any] = {
                "reference_group": str(reference),
                "reference_group_note": (
                    f"the row for {reference!r} compares that group with itself, so its "
                    "*_disparity values are 1.0 by construction and are not measurements"
                ),
            }

            if len(valid) >= 2:
                out: Dict[str, Any] = dict(reported)
                out.update(disclosure)
                if excluded:
                    out["groups_not_measured"] = dict(excluded)
                    out["population_note"] = (
                        f"aequitas was asked about the {int(keep.sum())} row(s) in the "
                        f"{len(valid)} group(s) meeting min_group_size="
                        f"{self.min_group_size}, which is the same population this "
                        f"analyzer's own metrics use. {len(excluded)} group(s) "
                        f"{sorted(map(str, excluded))} with sizes "
                        f"{ {str(k): v for k, v in excluded.items()} } are BELOW that "
                        f"floor, so they have no row in this table. A disparity number "
                        f"for a group the analyzer refuses to compare is not evidence "
                        f"of anything; the group and its size are reported instead."
                    )
                    warnings.warn(
                        f"compare_with_aequitas: {len(excluded)} group(s) "
                        f"{sorted(map(str, excluded))} (sizes "
                        f"{ {str(k): v for k, v in excluded.items()} }) are below "
                        f"min_group_size={self.min_group_size}, so their rows were NOT "
                        f"sent to aequitas and the table carries no disparity for them. "
                        f"They are named under groups_not_measured: that is could not "
                        f"check for those groups, not parity.",
                        UserWarning,
                        stacklevel=2,
                    )
                return out

            # Exact suffix, never a substring test: `"disparity" in column` would
            # also match a column that merely mentions it, and a mis-scoped
            # substring test is how a false pass has survived in a fairness gate
            # before (a test for "ratio" that also matched "calibration"). Only
            # `<metric>_disparity` columns carry a between-group comparison.
            withheld = [c for c in reported if str(c).endswith("_disparity")]
            warnings.warn(
                f"compare_with_aequitas: {len(valid)} comparable group(s) "
                f"(min_group_size={self.min_group_size}), so no disparity is defined "
                "and no cross-library comparison is possible. The "
                f"{len(withheld)} *_disparity column(s) are nan, which means could not "
                f"check. The reference group is {reference!r}, which on this frame is "
                "compared with itself, so aequitas' own 1.0 is a self-comparison and is "
                "NOT evidence of parity; it is kept under "
                "aequitas_reported_disparities.",
                UserWarning,
                stacklevel=2,
            )
            out = dict(reported)
            for column in withheld:
                values = out[column]
                # to_dict() gives {column: {index: value}}; stay defensive about
                # any other shape rather than raising a new error from a refusal.
                if isinstance(values, dict):
                    out[column] = {index: float("nan") for index in values}
                else:
                    out[column] = float("nan")
            out.update(disclosure)
            out["not_comparable"] = (
                f"{len(valid)} group(s) meet min_group_size={self.min_group_size}; "
                "a disparity between groups needs at least two"
            )
            out["disparity_columns_withheld"] = withheld
            out["aequitas_reported_disparities"] = reported
            return out
        except Exception as e:
            return {"error": str(e)}

    def report_to_svg(self, save_path: Optional[str] = None) -> str:
        """Render the fairness report as a polished SVG dashboard.

        Calls ``get_report()`` internally and renders the result.
        Requires ``jinja2`` (install with ``pip install jinja2``).

        Parameters
        ----------
        save_path : str, optional
            If provided, also write the SVG to this file.

        Returns
        -------
        str
            Complete SVG markup.
        """
        from vfairness.rendering import fairness_report_to_svg

        # ``fairness_report_to_svg`` consumes a plain report dict; hand it one
        # (a ``FairnessReport`` is that dict at runtime, this only sheds the
        # static TypedDict type at the render boundary).
        return fairness_report_to_svg(dict(self.get_report()), save_path=save_path)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FairnessAnalyzer("
            f"task_type='{self.task_type}', "
            f"n_samples={self.n_samples}, "
            f"groups={self.groups}, "
            f"backend='{self._backend}')"
        )
