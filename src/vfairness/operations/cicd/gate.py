"""
Model Fairness Gate for CI/CD Pipelines.

This module provides the ModelFairnessGate class for enforcing fairness
requirements during model deployment in CI/CD workflows.

Features:
    - Basic fairness gate evaluation against thresholds
    - Hierarchical intersectional checking (overall → single → intersections)
    - Per-intersection threshold configuration
    - Small-sample warnings for unreliable group metrics, on BOTH entry points:
      a comparison drawn across a group below ``min_group_size`` is reported
      with its measured numbers and refused approval, never certified
    - FairnessReportCard for automated PR comments
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from vfairness._triage import is_measured, unmeasurable_reason
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    ThresholdOutcome,
    check_threshold,
    improvement_amount,
    relax_threshold,
    vacuous_bound_reason,
)


class GateStatus(Enum):
    """Status of a fairness gate evaluation."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    CONDITIONAL = "conditional"


# The smallest group a between-group comparison can be CERTIFIED on.
#
# 30 is this library's house number and is reused here on purpose: it is the
# default ``min_group_size`` of every metric in
# ``vfairness.evaluation.vfairness_metrics.classification``, the default of
# :class:`HierarchicalGateConfig` below, and ``_statistics
# .SMALL_SAMPLE_THRESHOLD``. A gate certifying at a LOOSER number than the
# metric layer would approve exactly the comparisons the metric layer reports as
# NOT MEASURABLE.
DEFAULT_MIN_GROUP_SIZE = 30


@dataclass
class GateConfig:
    """Configuration for model fairness gate.

    Attributes:
        metrics: List of fairness metrics to evaluate.
        thresholds: Dictionary mapping metric names to threshold values.
        require_improvement: Whether to require improvement over baseline.
        improvement_margin: Minimum improvement required (as fraction).
        blocking_metrics: Metrics that block deployment if failed (others warn).
        allow_degradation_margin: Maximum degradation from baseline that is
            still allowed, in the metric's own direction. ``None`` (the
            default) means no degradation bound is applied at all; ``0.0``
            means zero tolerance, so ANY degradation blocks while an unchanged
            metric still passes; a positive number is that much slack.

            It was a plain ``float`` defaulting to ``0.0``, guarded by
            ``> 0``, which made the default AND the strictest-looking setting
            switch the check OFF. Measured 2026-09-10 on a current
            demographic_parity_difference of 0.09 against a baseline of 0.01,
            nine times worse: ``margin=0.0`` returned approved=True /
            status=approved / github=success, while ``margin=0.001`` returned
            approved=False / status=blocked / "degraded beyond allowed
            margin". The parameter was non-monotonic and this line's own
            wording ("Maximum allowed degradation") was false at its default.

            ``Optional[float]`` with an ``is not None`` guard was chosen over
            re-reading ``0.0`` as "enforce" with a ``>= 0`` guard, for two
            reasons. It keeps the numeric domain monotonic (0.0 strictest,
            larger looser) while still giving "do not apply this bound" a name
            of its own, which ``>= 0`` deletes. And it leaves the DEFAULT
            behaviour alone: making 0.0 both the default and zero-tolerance
            would block every existing gate that passes a baseline on
            floating-point noise, which is the over-correction failure mode. A
            gate that blocks every deployment is as useless as one that
            approved everything.
        min_group_size: Smallest group :meth:`ModelFairnessGate.evaluate` will
            certify a comparison on. Groups below it are NEVER dropped from the
            computation (see ``_compute_default_metrics``); the measured numbers
            are still reported, but the verdict on them is withheld as
            could-not-check instead of APPROVED.
    """

    metrics: List[str] = field(default_factory=lambda: ["demographic_parity_difference"])
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {"demographic_parity_difference": 0.1}
    )
    require_improvement: bool = False
    improvement_margin: float = 0.0
    blocking_metrics: Optional[List[str]] = None
    # None = no degradation bound applied; 0.0 = zero tolerance (enforced).
    # See the attribute docstring above for the measurement that made this
    # Optional rather than a float defaulting to 0.0.
    allow_degradation_margin: Optional[float] = None
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "require_improvement": self.require_improvement,
            "improvement_margin": self.improvement_margin,
            "blocking_metrics": self.blocking_metrics,
            "allow_degradation_margin": self.allow_degradation_margin,
            "min_group_size": self.min_group_size,
        }


@dataclass
class HierarchicalGateConfig:
    """Configuration for hierarchical intersectional fairness gate checks.

    Controls whether the gate evaluates fairness at multiple levels:
    overall → single-attribute → intersectional groups.

    Attributes:
        check_overall: Evaluate overall metrics across all groups.
        check_single_attributes: Evaluate each protected attribute individually.
        check_intersections: Evaluate intersectional group combinations.
        intersection_depth: Max number of attributes to combine (2 = pairwise).
        per_intersection_thresholds: Per-group threshold overrides.
            Keys are group names (e.g., "Female_Black"), values are
            {metric_name: threshold} dicts.
        default_intersection_threshold_multiplier: Multiplier applied to
            base thresholds for intersectional groups (default 1.2 = 20% relaxed).
        min_group_size: Minimum samples per group; smaller groups trigger warnings.
    """

    check_overall: bool = True
    check_single_attributes: bool = True
    check_intersections: bool = True
    intersection_depth: int = 2
    per_intersection_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default_intersection_threshold_multiplier: float = 1.2
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_overall": self.check_overall,
            "check_single_attributes": self.check_single_attributes,
            "check_intersections": self.check_intersections,
            "intersection_depth": self.intersection_depth,
            "per_intersection_thresholds": self.per_intersection_thresholds,
            "default_intersection_threshold_multiplier": self.default_intersection_threshold_multiplier,
            "min_group_size": self.min_group_size,
        }


@dataclass
class SmallSampleWarning:
    """Warning issued when a group has too few samples for reliable metrics.

    Attributes:
        group_name: Name of the group with small sample size.
        sample_size: Actual number of samples in the group.
        minimum_recommended: Recommended minimum sample size.
        metric_name: Metric being evaluated (or 'all' for general warning).
        message: Human-readable warning message.
    """

    group_name: str
    sample_size: int
    minimum_recommended: int
    metric_name: str = "all"
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = (
                f"Group '{self.group_name}' has only {self.sample_size} samples "
                f"(recommended minimum: {self.minimum_recommended}). "
                f"Metric '{self.metric_name}' may be unreliable."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "sample_size": self.sample_size,
            "minimum_recommended": self.minimum_recommended,
            "metric_name": self.metric_name,
            "message": self.message,
        }


def _undersized_groups(protected_attr: np.ndarray, min_group_size: int) -> List[SmallSampleWarning]:
    """Groups too small for a comparison drawn across them to be certified.

    One :class:`SmallSampleWarning` per group holding fewer than
    ``min_group_size`` rows, in the order numpy reports the labels. No group is
    removed from anything here: this only reports which groups make the
    comparison uncertifiable.
    """
    groups, counts = np.unique(protected_attr, return_counts=True)
    return [
        SmallSampleWarning(
            group_name=str(group),
            sample_size=int(count),
            minimum_recommended=min_group_size,
        )
        for group, count in zip(groups, counts)
        if int(count) < min_group_size
    ]


def _rate_denominator_legs(
    metric_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Optional[List[Tuple[str, np.ndarray]]]:
    """The rows each per-group rate of ``metric_name`` is actually computed over.

    Three answers, and the third is why this function exists rather than a dict
    lookup with a default:

    * ``[]`` when the denominator IS the group, so counting the group's rows counts
      exactly the right thing and ``_undersized_groups`` already does it. The two
      selection-rate metrics are the whole of this case.
    * a list of ``(what those rows are, mask)`` when the rate is conditional, so the
      rows it rests on are a LABELLED SUBSET of the group and can be a handful while
      the group is large.
    * ``None`` when this gate cannot say what the statistic rests on. That is never
      read as "nothing to check": the caller discloses it as an unsized scope, the
      same way ``evaluate_hierarchical`` discloses a cell nothing sized.

    The masks mirror ``_compute_default_metrics`` line for line, because they have to
    be the SAME denominators: this function makes a claim about numbers that method
    produced.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric_name in ("demographic_parity_difference", "disparate_impact_ratio"):
        # A selection rate is computed over every row of the group.
        return []
    if metric_name == "equalized_odds_difference":
        return [
            ("the true-positive rate leg, rows whose true label is 1", y_true == 1),
            ("the false-positive rate leg, rows whose true label is 0", y_true == 0),
        ]
    if metric_name == "false_positive_rate_difference":
        return [("rows whose true label is 0", y_true == 0)]
    if metric_name == "predictive_parity_difference":
        return [("rows the model predicted positive", y_pred == 1)]
    return None


def _undersized_rate_denominators(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    min_group_size: int,
    skip_groups: frozenset = frozenset(),
) -> Optional[List[SmallSampleWarning]]:
    """Rate denominators below the certification floor, or None if unknowable.

    A FLOOR COUNTING ROWS WHERE THE QUANTITY NEEDS LABELS (W4, 2026-09-30), and the
    sibling of ``_vacuous_min_group_size``: that one closed a minimum nothing could
    fall below, and this one closes a minimum applied to the wrong count. The gate's
    certification floor was compared against ``np.unique(protected_attr)`` row
    counts, while three of the five metrics it computes rest on a CONDITIONAL
    denominator: a true-positive rate lives on the rows whose true label is 1, a
    false-positive rate on the rows whose label is 0, predictive parity on the rows
    the model predicted positive. A group of 100 rows holding ONE positive label
    passed the floor and its true-positive rate was certified on that one person.

    Measured on HEAD, 2026-09-30, two groups of 100 rows against
    ``min_group_size=30``, group a with 50 positives and group b with only a few, all
    predictions correct so both true-positive rates are 1.0 and the gap is the
    textbook perfect-parity 0.0000::

        positives in group b   1  equalized_odds_difference 0.0000 passed=True
            approved=True  status=approved  github 'success'  card row 'Pass'
            small_sample_warnings []  small_sample_check_ran True
            card '- RAN, and every group was at or above the minimum. This is a
                  measurement, not an absence of one.'
        positives in group b   2  identical
        positives in group b   5  identical
        positives in group b  29  identical
        group b shrunk to 29 ROWS  approved=False, BLOCKED, warnings [('b', 29)]

    and the same shape on predictive parity: group b with ONE predicted positive,
    which the model got right, gave predictive_parity_difference 0.0000, approved,
    zero warnings. So the identical comparison was REFUSED at 29 rows and CERTIFIED
    on one label, and the surface that said so in the strongest available words
    ("This is a measurement, not an absence of one") was talking about a count that
    had nothing to do with the number it was published beside.

    AFTER: the metric is could-not-check with the leg and its real count named, on
    the audit trail and in the blocking reason, routed exactly like the row floor
    (blocking metric -> BLOCKED, non-blocking -> CONDITIONAL). The measured number is
    still reported and no group is dropped from anything.

    ``0 < n`` on purpose. A denominator of ZERO makes the rate undefined, and
    ``_compute_default_metrics`` already answers NaN for it and ``evaluate`` already
    fails closed on that NaN with the more precise reason. Reporting it here as well
    would put two different refusals on one number, which is the rule the margin and
    baseline guards in this module already follow.

    ``skip_groups`` holds the groups the ROW floor has already refused. Their legs
    are smaller still and the row-floor message already refuses every metric on
    them, so naming them twice would only be two wordings for one fact.
    """
    legs = _rate_denominator_legs(metric_name, y_true, y_pred)
    if legs is None:
        return None
    protected_attr = np.asarray(protected_attr)
    undersized: List[SmallSampleWarning] = []
    for leg_label, leg_mask in legs:
        for group in np.unique(protected_attr):
            if str(group) in skip_groups:
                continue
            n = int(np.sum((protected_attr == group) & leg_mask))
            if 0 < n < min_group_size:
                undersized.append(
                    SmallSampleWarning(
                        group_name=f"{group}, {leg_label}",
                        sample_size=n,
                        minimum_recommended=min_group_size,
                        metric_name=metric_name,
                    )
                )
    return undersized


def _unmatched_blocking_message(unmatched: List[str], configured: List[str]) -> str:
    """The could-not-check sentence for a blocking requirement on an ungated metric.

    W4, 2026-09-30. ``is_blocking = metric_name in self.config.blocking_metrics`` is a
    MEMBERSHIP TEST, so a name in that list which is not in ``metrics`` does not merely
    do nothing: while it sits there, every metric that IS gated on tests False and is
    treated as NON-BLOCKING. One mistyped character turns a blocking gate into a
    warn-only one, and the summary says the model is approved.

    Measured on HEAD, ``GateConfig(metrics=['demographic_parity_difference'],
    thresholds={'demographic_parity_difference': 0.1})`` on 200 real rows with a
    measured gap of 0.5000, five times the threshold, identically through
    :meth:`ModelFairnessGate.evaluate` and
    :meth:`ModelFairnessGate.evaluate_from_metrics`::

        blocking_metrics=None                                approved=False BLOCKED
            github 'failure'  card '**Result**: Deployment blocked'  row '**FAIL**'
        blocking_metrics=['demographic_parity_difference']    approved=False BLOCKED
        blocking_metrics=['demographic_parity_diference']     approved=True  CONDITIONAL
            github 'neutral'  card '**Result**: Approved for deployment'  row 'Warn'
            summary 'APPROVED with 1 warning(s)'

    one deleted letter between the second and third rows, and nothing on any surface
    said the named metric did not exist. The declared blocking requirement was never
    evaluated, and the breach it was declared for was published as approved.

    ``blocking_metrics=[]`` IS NOT THIS CASE and is deliberately left alone: an empty
    list names no requirement, and it is the only way to ask for a warn-only gate.
    Refusing it would delete a documented mode, which is the over-correction. The
    refusal here is for a name that was typed and never gated on.
    """
    named = ", ".join(repr(n) for n in unmatched)
    return (
        f"{len(unmatched)} metric(s) named in blocking_metrics are not in the gate's "
        f"`metrics` list ({named}), so the blocking requirement declared on them was "
        f"never evaluated at all, and while they sit there every metric that IS gated "
        f"on ({', '.join(repr(m) for m in configured)}) is treated as non-blocking, so "
        f"a measured breach downgrades to a warning and the deployment is approved. The "
        f"gate fails closed rather than reporting an unevaluated blocking requirement "
        f"as met. Correct the names, add them to `metrics`, or pass "
        f"blocking_metrics=[] if a warn-only gate is what was wanted."
    )


def _unsized_rate_scope_sentence() -> str:
    """The scope line for rates this gate did not compute and cannot account for.

    A ``compute_metrics_fn`` decides for itself what rows each of its numbers rests
    on, so the group-size scan cannot answer for them and must not be published as
    though it had. One wording, used by both entry points that accept arrays, for the
    reason ``_unsized_scope_lines`` states: two wordings for one fact is how two
    surfaces drift apart.
    """
    return (
        "the rows each metric's rate was computed over (a caller-supplied "
        "compute_metrics_fn produced the numbers, so the gate cannot tell which rows "
        "each rate rests on)"
    )


def _vacuous_min_group_size(min_group_size: int) -> Optional[int]:
    """The configured minimum, when NO group count can fall below it, else None.

    THE SAME RULE AS _vacuous_bound_message, ON THE GATE'S FOURTH BOUND (W3,
    2026-09-30): a bound is unusable when the comparison it controls cannot be False
    for any data. ``_undersized_groups`` tests ``count < min_group_size`` over groups
    that ``np.unique`` found, so every group holds at least one row and the test is
    False for every group of every dataset once the minimum is 1 or less. The check
    then cannot fire, and its empty answer is vacuous rather than clean.

    THE THRESHOLD OF VACUITY IS 1, NOT 0, and this is the identical off-by-one that
    ``_grouping.GroupManager.get_invalid_groups`` closed on 2026-09-27, whose comment
    names ``operations/cicd/gate`` among the production call sites that pass 1. That
    fix covered the METRIC layer's parameter; the gate's OWN certification minimum,
    the one ``config.min_group_size`` sets and this module's DEFAULT_MIN_GROUP_SIZE
    comment is about, was never tested for it.

    Measured on HEAD 2026-09-30, one group of 200 and one group of ONE person, a
    selection-rate gap of 0.5000 against a threshold of 0.9::

        min_group_size=30  approved=False  warnings [('b', 1)]  check_ran True
        min_group_size=2   approved=False  warnings [('b', 1)]  check_ran True
        min_group_size=1   approved=True   warnings []          check_ran True
        min_group_size=0   approved=True   warnings []          check_ran True
        min_group_size=-5  approved=True   warnings []          check_ran True

    and at 1, 0 and -5 the report card read "RAN, and every group was at or above the
    minimum. This is a measurement, not an absence of one.", which is the strongest
    form that claim can take, about a comparison that could not have failed. The
    verdict moved with it: the one-person comparison is refused at 2 and APPROVED at
    1.

    AFTER: the measurement and the verdict are UNCHANGED (a caller who sets 1 has
    asked not to have group sizes certified, and there is no other way to say that,
    so blocking would leave no way to switch the check off), and
    ``small_sample_check_ran`` is False with the number named on every surface: the
    empty list means NOT CHECKED, exactly as it does for a decision built from bare
    numbers. The smallest minimum that can flag anything is 2.

    AND A MINIMUM THAT IS NOT A NUMBER AT ALL (W4, 2026-09-30), which is the same
    hole reached through ``is_measured`` rather than through the off-by-one. ``nan <=
    1`` is False, so the numeric test above called a NaN minimum usable, while
    ``count < nan`` is False for every group of every dataset exactly as ``count <
    1`` is. Measured on HEAD, one group of 200 and one group of ONE person, a
    selection-rate gap of 0.5000 against a threshold of 0.95::

        min_group_size=2    approved=False  warnings [('b', 1)]  check_ran True
        min_group_size=1    approved=True   warnings []          check_ran False
        min_group_size=nan  approved=True   warnings []          check_ran True

    and the NaN row published the card's strongest sentence, "RAN, and every group
    was at or above the minimum. This is a measurement, not an absence of one.", for
    a one-person comparison that no bound could have refused. AFTER: NOT RUN with
    the bound named, on every surface, and the verdict again left alone.

    INFINITY IS NOT THIS CASE and must not be routed here. ``count < inf`` is True
    for every group, so the scan fires for ALL of them and the gate blocks
    everything: a bound nothing can satisfy, which is the opposite failure and is
    already loud. Measured in the same run: ``min_group_size=inf`` gives
    approved=False with both groups named. Returning NOT RUN for it would turn a
    refusal into an approval, so the test below is for NaN alone.
    """
    # A bool IS an int in Python (True is 1, False is 0), so it belongs in the
    # numeric case below where the sentence can name the number it really is.
    if isinstance(min_group_size, bool):
        min_group_size = int(min_group_size)
    if isinstance(min_group_size, (float, np.floating)) and np.isnan(min_group_size):
        # Carried as NaN rather than as the value itself: returning
        # ``min_group_size`` would hand back None for a minimum of None, and None is
        # this function's "the bound is usable".
        return float("nan")
    return int(min_group_size) if min_group_size <= 1 else None


def _unknown_direction_message(
    requirement: str, metric_name: str, value: float, baseline_value: float
) -> str:
    """The could-not-check sentence for a baseline requirement with no direction.

    BGL-5 deferral from batch A-evaluation-1, closed 2026-09-27.
    ``improvement_amount`` returns None when a metric's better-direction is not
    declared, and BOTH gate entry points SUBSTITUTED a magnitude comparison for
    that None and then used the substituted number to decide ``passed``. A release
    gate was making a pass or fail decision out of a direction it did not know.

    The substitution is not merely unmeasured, IT ASSUMES SMALLER IS BETTER, so for
    any metric where larger is better the sign is backwards and a real improvement
    reads as a degradation. Measured 2026-09-27 with ``improvement_amount``
    returning None, and no warning in any case:

        value  0.2, baseline 0.3 -> substituted +0.100, read as "improved"
        value  0.3, baseline 0.2 -> substituted -0.100, read as "did not improve"
        value -0.5, baseline 0.1 -> substituted -0.400

    The honest answer in every one of those is None: nothing about the direction of
    travel was established.

    FOUR SITES, ONE HELPER, on purpose. The substitution appeared twice in
    ``evaluate`` and twice in ``evaluate_from_metrics``, and the degradation half's
    own comment records that those two paths had already diverged once, when only
    ``evaluate`` enforced the margin at all. Four copies of a rule is four chances
    to fix three of them.

    The gate fails closed on it, which is the rule the baseline guard above applies
    and the rule this module states for an unassessed severity: not having been able
    to check an improvement is not evidence that one happened.
    """
    return (
        f"{metric_name}: the {requirement} COULD NOT BE EVALUATED. Its better "
        f"direction is not declared, so whether {value} is "
        f"{'an improvement on' if requirement == 'required improvement' else 'a degradation from'} "
        f"the baseline {baseline_value} is not established, and the gate does not "
        f"substitute a magnitude comparison for it. Declare the metric's direction, "
        f"or stop requiring this of it. This is a could-not-check counted as "
        f"blocking, not an observed failure."
    )


def _unusable_margin_message(requirement: str, metric_name: str, margin: float) -> str:
    """The could-not-check sentence for a requirement whose MARGIN is not a number.

    BGL6 F03, 2026-09-28. ``improvement < margin`` has TWO operands and only one
    was guarded. The baseline guard covers the left-hand side; the margin is the
    right-hand side of the same two comparisons and had no guard anywhere, in
    EITHER entry point, so the sentence in _unmeasurable_baseline_message stayed
    true word for word of the margin.

    Measured on HEAD with ``GateConfig(metrics=['demographic_parity_difference'],
    thresholds={'demographic_parity_difference': 0.95}, require_improvement=True,
    improvement_margin=float('nan'))`` and
    ``evaluate_from_metrics({'demographic_parity_difference': 0.9},
    baseline_metrics={'demographic_parity_difference': 0.01})``, which is a
    degradation of 0.89 in the metric's own declared direction, BEFORE:

        approved=True  status=approved  'APPROVED - All fairness requirements met'
        warnings=[]  blocking=[]  improvement=-0.89
        create_github_check conclusion 'success'

    The identical call with ``improvement_margin=0.01`` blocks with
    'demographic_parity_difference did not improve by required margin', so the
    margin IS the operand that decides it. ``check_threshold`` in this same library
    already refuses an unmeasurable THRESHOLD for exactly this reason ("An
    unmeasurable value, an unmeasurable threshold, an unknown direction or a
    DEGENERATE bound all yield COULD_NOT_CHECK"); the gate's two bounds never
    reached that rule.

    AFTER: approved=False with the requirement named as unchecked, rather than an
    unchecked requirement published as met.
    """
    return (
        f"COULD NOT CHECK the {requirement} requirement for {metric_name}: its margin "
        f"is {margin!r}, which is not a usable bound, so the comparison that enforces "
        f"the requirement is False whatever the data and the requirement was never "
        f"checked. The gate fails closed rather than publishing an unchecked "
        f"requirement as met. Set a finite margin, or switch the requirement off "
        f"explicitly."
    )


def _vacuous_bound_message(requirement: str, metric_name: str, reason: str) -> str:
    """The could-not-check sentence for a bound NO DATA CAN BREACH.

    THE SIBLING OF _unusable_margin_message, AND THE WIDER RULE IT WAS A SPECIAL
    CASE OF (2026-09-30). That guard is gated on ``is_measured(margin)``, which
    closes NaN and the two infinities and leaves every FINITE vacuous bound open.
    The rule is:

        A BOUND IS UNUSABLE WHEN THE COMPARISON IT CONTROLS CANNOT BE FALSE FOR
        ANY DATA IN THE METRIC'S RANGE.

    That is a property of the bound and the metric's range together, not of
    whether the bound is finite; infinity is the easiest case of it. The library's
    own refusal sentence already states the criterion ("the comparison that
    enforces the requirement is False whatever the data"), and a finite bound can
    meet it exactly.

    Measured on HEAD, 2026-09-30, BEFORE this existed. A model that rejects 100
    percent of one intersectional cell and nobody else, so the intersectional gap
    is 1.0000, the largest a rate disparity can be, against a base threshold of
    0.6 relaxed by ``default_intersection_threshold_multiplier``::

        mult=1.2    approved=False  'BLOCKED - 3/4 levels passed'   gap 1.0000 vs 0.72  passed=False
        mult=2.0    approved=True   'APPROVED - 4/4 levels passed'  gap 1.0000 vs 1.2   passed=True
        mult=100.0  approved=True   'APPROVED - 4/4 levels passed'  gap 1.0000 vs 60.0  passed=True
        mult=inf    approved=False  'BLOCKED - 3/4 levels passed'   gap 1.0000 vs inf   passed=False

    and at mult 2.0 and 100.0 the report card dropped its '### Blocking Issues'
    section entirely, so the page a reviewer merges from showed nothing at all
    about the cell that is never selected. ``relax_threshold`` guards
    ``multiplier <= 0`` and NaN, so the multiplier carried the bound clean outside
    the metric's own range: a demographic_parity_difference lives in [0, 1] and
    was being compared against 60.0.

    The same shape on the baseline bounds, same date, on a measured DEGRADATION of
    0.5 (0.7 against a baseline of 0.2) through ``evaluate_from_metrics``, and
    identically through ``evaluate``::

        improvement_margin        0.01   -> BLOCKED
        improvement_margin        0.0    -> BLOCKED
        improvement_margin       -1.0    -> APPROVED, 0 warnings
        improvement_margin       -1e9    -> APPROVED, 0 warnings
        improvement_margin        inf    -> BLOCKED (is_measured, the narrow case)
        allow_degradation_margin  1e9    -> APPROVED, 0 warnings

    AFTER: each of those is could-not-check with the bound and the range named,
    and the requirement is refused rather than published as met.

    ONE RULE, FOUR BOUNDS, on purpose: the gate threshold (which is also the
    relaxed intersectional threshold and any per-intersection override, because
    both reach ``evaluate`` as that level's threshold), ``improvement_margin`` and
    ``allow_degradation_margin``, at BOTH entry points. The predicate itself lives
    in ``_metric_direction.vacuous_bound_reason`` beside ``check_threshold``'s
    narrower degenerate-bound rule, so there is one implementation of the test and
    not four.
    """
    return (
        f"COULD NOT CHECK the {requirement} for {metric_name}: it is {reason}. The "
        f"comparison that enforces it is False for every value the metric can take, so "
        f"the requirement was never checked and nothing was graded. The gate fails "
        f"closed rather than publishing an unchecked requirement as met. Set a bound "
        f"inside the metric's range; if this one came from "
        f"default_intersection_threshold_multiplier, lower the multiplier, or switch "
        f"the requirement off explicitly."
    )


def _unmeasurable_baseline_message(
    metric_name: str, value: float, baseline_value: Optional[float]
) -> str:
    """The could-not-check sentence for a baseline requirement nobody could check.

    BGL5 A-operations-1, 2026-09-27. Both gate entry points guarded the metric
    VALUE against a non-finite number three separate ways and left the BASELINE
    it is compared against unguarded. Measured on HEAD with
    ``GateConfig(metrics=['demographic_parity_difference'],
    thresholds={'demographic_parity_difference': 0.5}, require_improvement=True,
    improvement_margin=0.01)`` and
    ``baseline_metrics={'demographic_parity_difference': float('nan')}`` on 100 +
    100 genuinely fair rows, BEFORE:

        approved=True  status=GateStatus.APPROVED
        summary 'APPROVED - All fairness requirements met'
        warnings=[]  blocking_reasons=[]
        row {'value': 0.0, 'threshold': 0.5, 'passed': True,
             'baseline_value': nan, 'improvement': nan, 'message': ''}
        markdown '| demographic_parity_difference | 0.0000 | 0.5000 | (tick) Pass |'
        create_github_check conclusion 'success'

    ``improvement < margin`` is False for NaN and ``degradation > margin`` is
    False for NaN, so ONE unmeasurable baseline silently switched off BOTH
    configured requirements and the gate published the result as every
    requirement met. AFTER, same input: approved=False, status=BLOCKED,
    summary 'BLOCKED - 1 fairness requirement(s) not met', the metric row
    carries passed=False with this sentence, improvement=nan, and
    create_github_check reports conclusion 'failure'. A NON-BLOCKING metric
    downgrades to CONDITIONAL with the same sentence in ``warnings`` and a
    'neutral' check conclusion. With no baseline requirement configured
    (``require_improvement=False`` and ``allow_degradation_margin=None``)
    nothing was required, so nothing was unchecked and a NaN baseline changes
    no verdict: that control is pinned, because a guard that refuses every
    baseline would be the over-correction.

    The message lives here, once, because the identical hole stood at BOTH call
    sites and a second copy of the wording is how two paths drift apart.
    """
    return (
        f"{metric_name} was measured at {value:.4f}, but the baseline it had to be "
        f"compared against could not be measured ({unmeasurable_reason(baseline_value)}), "
        f"so the configured baseline requirement (require_improvement / "
        f"allow_degradation_margin) was never checked; the gate fails closed rather "
        f"than reporting an unchecked requirement as met"
    )


def _absent_baseline_message(metric_name: str, value: float) -> str:
    """The could-not-check sentence for a required baseline that was never supplied.

    W2 A-operations-1, 2026-09-29. THE SIBLING OF THE NON-FINITE BASELINE, and the
    commoner half by far: the guard above refused a baseline holding NaN or inf and
    treated a baseline that is simply NOT THERE as measurable, because its
    precondition read ``baseline_value is None or is_measured(baseline_value)``.
    ``baseline_value = baseline_metrics.get(metric_name) if baseline_metrics else
    None`` answers None to FOUR distinct caller mistakes, and every one of them
    reached the comparison as "nothing was required". Measured on HEAD with
    ``GateConfig(metrics=[dp], thresholds={dp: 0.9}, require_improvement=True,
    improvement_margin=0.01)`` and ``evaluate_from_metrics({dp: 0.2}, ...)``,
    identically through :meth:`ModelFairnessGate.evaluate` on 100 + 100 real rows:

        baseline ABSENT (arg omitted)  approved=True  blocking=0 warns=0 gh 'success'
        baseline dict EMPTY            approved=True  blocking=0 warns=0 gh 'success'
        baseline key MISSING           approved=True  blocking=0 warns=0 gh 'success'
        baseline key present = None    approved=True  blocking=0 warns=0 gh 'success'
        baseline key present = NaN     approved=False blocking=1        gh 'failure'
        baseline MEASURED (0.3)        approved=True  improvement=0.0999

    all four of the first rows reporting "APPROVED - All fairness requirements met"
    with ``improvement=None``. A caller asked the gate to BLOCK unless fairness
    improved on a baseline, supplied no baseline, and was told every requirement was
    met. Neither comparison can run without a left operand: the
    ``baseline_value is not None`` clause on both of them skipped straight past.

    AFTER, same input: approved=False, BLOCKED, this sentence in
    ``blocking_reasons``, ``improvement=nan`` (the comparison was attempted and
    could not be made, which is not the same as ``None``, "no comparison was asked
    for"), and ``create_github_check`` reports 'failure'. A non-blocking metric
    downgrades to CONDITIONAL with the sentence in ``warnings``. With no baseline
    requirement configured nothing was required, so an absent baseline changes no
    verdict and ``improvement`` stays None: that control is pinned, because a guard
    that refused every gate configured without a baseline would be the
    over-correction.

    The message lives here, once, beside its sibling, because the identical hole
    stood at BOTH call sites and a second copy of the wording is how two paths
    drift apart.
    """
    return (
        f"{metric_name} was measured at {value:.4f}, but NO BASELINE VALUE for it was "
        f"supplied, so the configured baseline requirement (require_improvement / "
        f"allow_degradation_margin) was never checked; the gate fails closed rather "
        f"than reporting an unchecked requirement as met. Supply "
        f"baseline_metrics[{metric_name!r}], or switch the requirement off explicitly."
    )


def _vacuous_minimum_sentence(minimum: int) -> str:
    """The NOT RUN sentence for a group-size bound no count can fall below.

    One wording, used by the flat report, the hierarchical report and the card, so
    the three cannot answer the same question differently. It names the NUMBER,
    because the operator's next action is to raise it: the smallest minimum that can
    flag anything is 2. See _vacuous_min_group_size for the measured before-state.

    TWO CAUSES, TWO SENTENCES (W4, 2026-09-30), because "every group holds at least
    one row" is not true of a minimum that is not a number: there the comparison
    could not be made at all rather than coming out False on a technicality. One
    sentence for two different facts is the copy-stronger-than-the-code failure this
    section exists to stop.
    """
    if isinstance(minimum, (float, np.floating)) and np.isnan(minimum):
        return (
            "- NOT RUN: the configured minimum was not a number (nan), so the "
            "comparison `group size < minimum` was False for every group and the check "
            "could not fire at all. An empty warning list means the check did not "
            "happen, NOT that every group was large enough. Set a whole number of 2 or "
            "more."
        )
    return (
        f"- NOT RUN: the minimum was {minimum}, and every group holds at least one "
        f"row, so no group count could fall below it and the comparison was False "
        f"for every group. An empty warning list means the check did not happen, NOT "
        f"that every group was large enough. The smallest minimum that can flag "
        f"anything is 2."
    )


def _unsized_scope_lines(unsized: List[str]) -> List[str]:
    """The NOT SIZED line naming every group set whose size was never compared.

    W3, 2026-09-30. ``small_sample_check_ran`` answers for the supplied ATTRIBUTES,
    which is what its docstring says and what ``evaluate_hierarchical`` computes,
    and all four graded surfaces rendered a True as a claim about every group in the
    data. Measured with two attributes whose groups are all 100 rows and whose four
    intersection cells are 95/5/5/95, against a minimum of 30::

        check_intersections=False  small_sample_check_ran True, warnings [],
            approved True, card "RAN, and every group was at or above the minimum.
            This is a measurement, not an absence of one."
        check_intersections=True   the same data names ('f_y', 5) and ('m_x', 5)

    so one run asserted a measurement about the very groups the other run found to
    be five people. The cells are sized in the Level 3 loop, which never fed the
    flag, and a reader of either surface had no way to tell which groups the answer
    covered.

    EVERY unsized set is listed, including the attribute half that the NOT RUN
    sentence already covers in its own words. Two wordings for one fact is how two
    surfaces drift apart; a disclosure that repeats itself is merely long, and
    silence about a group set is the failure this whole section exists to stop.
    """
    if not unsized:
        return []
    return [
        f"- NOT SIZED: {', '.join(unsized)}. No group size there was compared to the "
        f"minimum, so nothing above is a statement about those groups.",
        "",
    ]


@dataclass
class MetricEvaluation:
    """Evaluation result for a single metric."""

    metric_name: str
    value: float
    threshold: Optional[float]
    passed: bool
    baseline_value: Optional[float] = None
    improvement: Optional[float] = None
    is_blocking: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "baseline_value": self.baseline_value,
            "improvement": self.improvement,
            "is_blocking": self.is_blocking,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# ONE metric row, because it is rendered in FIVE places.
# ---------------------------------------------------------------------------
#
# G05, 2026-09-30. Every renderer of a MetricEvaluation mapped the row's verdict
# straight off ``passed``:
#
#     "Pass" if ev.passed else ("Warn" if not ev.is_blocking else "Fail")
#
# ``passed`` answers "did the comparison come out inside the bound". It cannot
# say whether the comparison HAPPENED, and the gate sets it False for BOTH a
# measured breach and a could-not-check, so the two rendered identically.
# Measured on ``evaluate(y, y, ['a'] * 200)`` (one group, so no disparity is
# definable) against a threshold of 0.1, all five surfaces at once::
#
#     GateDecision.to_markdown_report        | ... | nan | 0.1000 | X Fail |
#     IntersectionalGateDecision....         | ... | nan | 0.1000 | Fail |
#     FairnessReportCard._format_simple      | ... | nan | 0.1000 | **FAIL** |
#     FairnessReportCard._format_hierarchical  #### overall: Failed Metrics
#                                              - **...**: nan (threshold: 0.1000)
#                                              | overall | BLOCKED | 1 |
#
# The page-level "Blocking Issues" sections added on 2026-09-27 carry the
# sentence, and this file's own comment on IntersectionalGateDecision
# .to_markdown_report states the reading they exist to prevent ("A metric nobody
# could compute was published as a measured breach"). That fix landed on the
# PAGE and not on the ROW, which is the part a reviewer scans and the only part
# keyed to a metric name: with five metrics, one real breach and four
# unmeasurable, the table showed five identical failures and the reasons list is
# not keyed to rows at all (the hierarchical report prints the same sentence once
# per level with nothing saying which).
#
# A value of ``None`` was worse than misleading: ``f"{ev.value:.4f}"`` raised
# TypeError, so ``create_github_check`` could not build its payload at all,
# while ``to_dict`` on the same object serialised it happily. The sibling cell
# one line above already had the ``threshold is not None`` guard.
_ROW_PASS = "pass"
_ROW_WARN = "warn"
_ROW_FAIL = "fail"
_ROW_UNMEASURED = "unmeasured"
# Measured, inside its bound, and refused by something that is not that bound.
_ROW_REFUSED = "refused"


def _metric_row_state(evaluation: MetricEvaluation) -> str:
    """Which of FOUR states the row for one metric evaluation is in.

    ``_ROW_UNMEASURED`` first and above the ``passed`` dispatch, because every
    branch under it shares the precondition that a comparison was made: a guard
    inside either one cannot cover the other, and sabotaging this block has to
    redden the pass row and the fail row together.

    TWO DOORS, not one, and both are properties of the ROW ALONE:

    * the VALUE is not a measurement. ``is_measured`` rather than
      ``math.isnan``: an INFINITE metric value wins every comparison instead of
      losing it (``disparate_impact_ratio`` reaches ``inf`` from a group with no
      selections at all), and it is the same could-not-check. It is the
      predicate the gate's own value guard uses to build these rows, so the
      renderer and the producer cannot disagree about which values are
      measurements.
    * the BOUND could not grade anything: absent, not a number, or vacuous.
      ``vacuous_bound_reason`` is the repo's rule for the last one and it is a
      property of the bound AND the metric's range together, never of
      finiteness, so an ``isinf`` test leaves every finite vacuous bound open.
      Measured 2026-09-30 on a GENUINELY FAIR model, a
      ``demographic_parity_difference`` of 0.0100 against a threshold of 2.0 on
      a metric whose range is [0, 1]::

          | demographic_parity_difference | 0.0100 | 2.0000 | X Fail |

      A measured 0.01 gap reported as a FAILURE against a bound no data can
      breach: the arithmetic is absurd and the row was the only place a
      reviewer looks per metric. The gate's own message for it says "the
      comparison that enforces it is False for every value the metric can take,
      so the requirement was never checked".

    AND A ROW CAN BE REFUSED BY SOMETHING THAT IS NOT ITS BOUND, which is the
    third door and the reason ``check_threshold`` is consulted here rather than
    ``passed`` alone. ``passed`` is one boolean carrying the outcome of up to
    six checks (threshold, vacuous threshold, baseline, improvement margin,
    degradation margin, unknown direction), so "not passed" does not mean "over
    its threshold". Measured 2026-09-30 with ``require_improvement=True`` and a
    NaN baseline, an ABSENT baseline, and a NaN ``improvement_margin``, on a
    model whose measured gap was INSIDE its threshold -- all three rows read::

        | demographic_parity_difference | 0.0600 | 0.1000 | X Fail |

    0.06 marked as a failure against a maximum of 0.10, a row that contradicts
    its own two numbers. Asking the canonical comparator what the BOUND says
    separates them, and it cannot disagree with the producer, because the gate
    reaches its own verdict through that same function.

    A ``COULD_NOT_CHECK`` from the comparator is placed ABOVE the ``passed``
    dispatch on purpose: a pass claimed over a bound that cannot be applied
    (an unknown better-direction, a minimum at or below zero) is not a pass, and
    that is the one rule this library never weakens.

    WHICH LINE CARRIES WHICH CASE, measured by sabotage, because two of these
    lines are DEFENCE IN DEPTH and not the carrier:

    * removing the whole block reddens 12 tests;
    * the NaN / infinity / None value case is closed TWICE, by the
      ``is_measured(evaluation.value)`` line and again by the comparator's own
      ``COULD_NOT_CHECK``. Removing either ALONE stays green; removing both
      reddens 5. The first line is kept because it states the rule where a
      reader looks for it and does not depend on the comparator keeping that
      behaviour;
    * the same is true of ``threshold is None``, which the comparator also
      refuses;
    * the vacuous-bound line and the ``ThresholdOutcome.PASS`` line are each the
      SOLE carrier of their case (1 and 5 tests respectively).
    """
    if not is_measured(evaluation.value):
        return _ROW_UNMEASURED
    if evaluation.threshold is None or not is_measured(evaluation.threshold):
        return _ROW_UNMEASURED
    if vacuous_bound_reason(evaluation.metric_name, evaluation.threshold, BoundRole.THRESHOLD):
        return _ROW_UNMEASURED
    bound_outcome, _ = check_threshold(
        evaluation.metric_name, evaluation.value, evaluation.threshold
    )
    if bound_outcome is ThresholdOutcome.COULD_NOT_CHECK:
        return _ROW_UNMEASURED
    if evaluation.passed:
        return _ROW_PASS
    if bound_outcome is ThresholdOutcome.PASS:
        # The value is inside its bound and the row is still a refusal, so the
        # refusal is not about this bound. Naming it a threshold failure is the
        # self-contradicting row above; the reason is in ``message``, which
        # _metric_note_lines now prints beside the row.
        return _ROW_REFUSED
    return _ROW_WARN if not evaluation.is_blocking else _ROW_FAIL


def _metric_note_lines(evaluations: List[MetricEvaluation]) -> List[str]:
    """The per-metric reason each row carries, or ``[]`` when none carries one.

    ``MetricEvaluation.message`` is written at every one of the gate's
    fail-closed branches and was rendered by NO surface. The page-level
    "Blocking Issues" and "Warnings" sections carry the same sentences as a flat
    list that is not keyed to a metric at all, so with several metrics a reader
    could not tell which row a refusal belonged to, and the hierarchical report
    printed the identical sentence once per level with nothing saying which.
    Empty on a clean decision: ``message`` is "" for a row that passed, so this
    block appears only where there is something to say.
    """
    noted = [e for e in evaluations if e.message]
    if not noted:
        return []
    lines = ["", "### Per-metric notes", ""]
    lines.extend(f"- **{e.metric_name}**: {e.message}" for e in noted)
    return lines


def _metric_value_cell(value: Any) -> str:
    """The Value cell for one metric row: the number, or WHY there is none.

    Never ``f"{value:.4f}"`` on an unchecked value. ``None`` raises TypeError
    there, and ``nan`` / ``inf`` print as bare ``nan`` / ``inf`` beside a
    threshold and a verdict, which reads as a number that was compared.
    """
    if is_measured(value):
        return f"{float(value):.4f}"
    return f"not measured ({unmeasurable_reason(value)})"


@dataclass
class GateDecision:
    """Decision from a fairness gate evaluation.

    Attributes:
        approved: Whether the model is approved for deployment.
        status: Detailed status (approved, blocked, conditional).
        metric_evaluations: Evaluation results for each metric.
        blocking_reasons: List of reasons why deployment was blocked.
        warnings: List of warning messages.
        timestamp: When the evaluation was performed.
        metadata: Additional metadata about the evaluation.
        small_sample_warnings: Groups whose size is below the configured
            ``min_group_size``, so any comparison drawn across them could not be
            certified. Populated by :meth:`ModelFairnessGate.evaluate`, which
            sees the group labels; :meth:`ModelFairnessGate.evaluate_from_metrics`
            is handed numbers with no sample counts and cannot fill it in.

            AND THE RATE DENOMINATORS BELOW IT (W4, 2026-09-30), named by the leg
            they belong to (``'b, the true-positive rate leg, rows whose true label
            is 1'``) with ``metric_name`` set to the metric they refuse. A
            conditional rate rests on a LABELLED SUBSET of the group, so this list
            answers about the count each published number was actually computed
            over rather than only about the group's row count. See
            :func:`_undersized_rate_denominators` for the measured before-state, a
            true-positive rate certified on ONE person inside a group of 100.
        small_sample_unsized: The group sets and rate denominators whose size was
            NEVER compared to ``min_group_size``, named, so
            ``small_sample_check_ran`` can be read for what it covers. Empty means
            nothing was left unsized. The same field, the same meaning and the same
            renderer as :attr:`IntersectionalGateDecision.small_sample_unsized`.
        small_sample_check_ran: Whether the small-sample check HAPPENED. Three
            states, because an empty ``small_sample_warnings`` list had two
            meanings and no way to tell them apart. ``True``: the check ran, so
            an empty list means no group was too small. ``False``: it could not
            run (no group sizes were supplied, which is every
            ``evaluate_from_metrics`` decision) OR IT COULD NOT HAVE FAILED (the
            configured ``min_group_size`` is 1 or less, which no group count can
            fall below, recorded on ``small_sample_vacuous_minimum``; W3,
            2026-09-30), so an empty list means NOT CHECKED. ``None``: nobody
            recorded either way, which is what a hand-built :class:`GateDecision`
            says about itself.

            BGL5 A-operations-1, measured 2026-09-27 before this field existed,
            on two clean groups of 100 rows against the same numbers supplied
            pre-computed::

                evaluate()            -> small_sample_warnings [] approved True
                evaluate_from_metrics -> small_sample_warnings [] approved True
                identical: True
                keys naming whether the check RAN: ['small_sample_warnings']

            evaluate_from_metrics' own docstring insists on the distinction ("an
            empty list means NOT CHECKED, not 'no small groups'") and no
            consumer of ``to_dict`` or of the markdown report could read it.
    """

    approved: bool
    status: GateStatus
    metric_evaluations: List[MetricEvaluation]
    blocking_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    small_sample_warnings: List[SmallSampleWarning] = field(default_factory=list)
    # Defaults to None, never False: a decision nobody annotated has not
    # established that the check was skipped, only that nothing said so.
    small_sample_check_ran: Optional[bool] = None
    # The configured minimum when no group count can fall below it, else None.
    # It is the CAUSE of a False above, and the sentence has to name the number:
    # see _vacuous_min_group_size for the five measured rows.
    small_sample_vacuous_minimum: Optional[int] = None
    # What the answer above does NOT cover (W4, 2026-09-30). The flat twin of the
    # hierarchical field: a True means every count the gate KNOWS about was
    # compared, and the numbers from a caller-supplied compute_metrics_fn rest on
    # rows the gate cannot see.
    small_sample_unsized: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Generate a human-readable summary."""
        if self.approved:
            if self.warnings:
                return f"APPROVED with {len(self.warnings)} warning(s)"
            return "APPROVED - All fairness requirements met"
        else:
            return f"BLOCKED - {len(self.blocking_reasons)} fairness requirement(s) not met"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "approved": self.approved,
            "status": self.status.value,
            "metric_evaluations": [m.to_dict() for m in self.metric_evaluations],
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "small_sample_warnings": [w.to_dict() for w in self.small_sample_warnings],
            # The companion key that makes the line above readable. Without it
            # a checked-and-clean decision and a never-checked one serialised
            # byte-identically as [] (measured 2026-09-27, see the field
            # docstring); a JSON consumer now gets true, false or null.
            "small_sample_check_ran": self.small_sample_check_ran,
            # And WHY it is false when the BOUND is the reason (W3, 2026-09-30): a
            # minimum of 1 or less cannot be breached by any group count.
            "small_sample_vacuous_minimum": self.small_sample_vacuous_minimum,
            # And the SCOPE of the answer (W4, 2026-09-30), as on the hierarchical
            # twin: a True above covers the counts this gate sized, never every
            # count behind every number it was handed.
            "small_sample_unsized": list(self.small_sample_unsized),
        }

    def to_markdown_report(self) -> str:
        """Generate a markdown report of the decision."""
        lines = [
            "# Fairness Gate Decision Report",
            "",
            f"**Status**: {self.status.value.upper()}",
            f"**Timestamp**: {self.timestamp.isoformat()}",
            "",
            "## Metric Evaluations",
            "",
            "| Metric | Value | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ]

        for eval in self.metric_evaluations:
            # Four states, not three: see _metric_row_state for the measurement.
            # A could-not-check rendered here as "❌ Fail", the measured-breach
            # verdict, on the surface create_github_check publishes as
            # output.text.
            status = {
                _ROW_PASS: "✅ Pass",
                _ROW_WARN: "⚠️ Warn",
                _ROW_FAIL: "❌ Fail",
                _ROW_UNMEASURED: "🚫 Could not check",
                _ROW_REFUSED: "⛔ Refused (not by this bound)",
            }[_metric_row_state(eval)]
            # 'is not None': a threshold of 0.0 is a real, enforced bound
            # (zero-tolerance gate) and must not display as N/A.
            threshold_str = f"{eval.threshold:.4f}" if eval.threshold is not None else "N/A"
            lines.append(
                f"| {eval.metric_name} | {_metric_value_cell(eval.value)} "
                f"| {threshold_str} | {status} |"
            )

        # The reason belonging to THIS row, which no surface rendered.
        lines.extend(_metric_note_lines(self.metric_evaluations))

        if self.blocking_reasons:
            lines.extend(
                [
                    "",
                    "## Blocking Issues",
                    "",
                ]
            )
            for reason in self.blocking_reasons:
                lines.append(f"- {reason}")

        if self.warnings:
            lines.extend(
                [
                    "",
                    "## Warnings",
                    "",
                ]
            )
            for warning in self.warnings:
                lines.append(f"- {warning}")

        # The audit trail has to show WHICH group was too small, not only that a
        # comparison was refused. Without this section the markdown kept as the
        # record of the decision reported "Pass" on a five-person group with
        # nothing on the page to say the comparison had five people in it.
        if self.small_sample_warnings:
            lines.extend(
                [
                    "",
                    "## Small-Sample Warnings",
                    "",
                ]
            )
            for small in self.small_sample_warnings:
                lines.append(
                    f"- **{small.group_name}**: {small.sample_size} samples "
                    f"(min {small.minimum_recommended})"
                )
            # The scope of the scan that produced those rows, on the same builder
            # the hierarchical report and the card use (W4, 2026-09-30). A list of
            # named small groups is still not a statement about counts nobody
            # sized.
            lines.extend(_unsized_scope_lines(self.small_sample_unsized))
        else:
            # The same three-state disclosure as the dict, on the surface
            # create_github_check publishes as output.text. An absent
            # Small-Sample Warnings section read as "checked, every group was
            # big enough" on a decision where the check could not run at all.
            # Measured 2026-09-27: the report from evaluate() on two clean
            # groups of 100 and the report from evaluate_from_metrics on the
            # same numbers were identical on this question, both silent.
            #
            # ALL THREE STATES, NOT TWO (W3, 2026-09-30). This was
            # ``elif self.small_sample_check_ran is not True``, so the TRUE state
            # rendered NOTHING while FairnessReportCard printed "RAN, and every
            # group was at or above the minimum. This is a measurement, not an
            # absence of one." for the same decision: two surfaces answering one
            # question differently, and the report's answer was silence. Silence
            # is the one answer that is never right, as
            # _format_small_sample_state's own docstring says, and a reader can
            # only decode it by knowing that the other two states print. Same
            # sentence as the card, so neither can drift.
            lines.extend(["", "## Small-Sample Check", ""])
            if self.small_sample_check_ran is True:
                # SCOPED when something was left unsized, in the card's own words
                # (W4, 2026-09-30): the flat renderer had only the unqualified
                # sentence, so a decision whose rate denominators nobody sized read
                # here as a measurement about every count behind it.
                if self.small_sample_unsized:
                    lines.append(
                        "- RAN over every group set it covered, and each was at or above "
                        "the minimum. This is a measurement, not an absence of one."
                    )
                else:
                    lines.append(
                        "- RAN, and every group was at or above the minimum. This is a "
                        "measurement, not an absence of one."
                    )
            elif self.small_sample_vacuous_minimum is not None:
                # THE BOUND, not the absence of labels: this decision HAS the group
                # sizes and compared them to a minimum nothing could fall below, so
                # the bare-numbers sentence below would be false of it (W3).
                lines.append(_vacuous_minimum_sentence(self.small_sample_vacuous_minimum))
            elif self.small_sample_check_ran is False:
                lines.append(
                    "- NOT RUN: this decision was made from numbers with no group sizes "
                    "attached, so no group size was compared to the minimum. The empty "
                    "warning list above means the check did not happen, NOT that every "
                    "group was large enough."
                )
            else:
                lines.append(
                    "- NOT RECORDED: nothing on this decision states whether group sizes "
                    "were checked, so the empty warning list above cannot be read either "
                    "way."
                )
            lines.extend(_unsized_scope_lines(self.small_sample_unsized))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GateDecision({self.status.value}, approved={self.approved})"


@dataclass
class IntersectionalGateDecision:
    """Decision from a hierarchical intersectional fairness gate evaluation.

    Contains results at each hierarchy level (overall, single-attribute,
    intersectional) plus small-sample warnings.

    Attributes:
        approved: Whether the model is approved at all levels.
        status: Overall status across all hierarchy levels.
        level_results: Gate decisions keyed by level name
            ('overall', 'attr:<name>', 'intersection:<name1>_x_<name2>').
        small_sample_warnings: Warnings for groups with insufficient samples.
        summary_decision: Aggregated GateDecision across all levels.
        hierarchical_config: Config used for this evaluation.
        blocking_reasons: WHY this decision is not approved, in the words of the
            level that refused, plus any reason the aggregation itself added
            (the no-level-ran refusal is the only one of those today).
        small_sample_check_ran: Whether the group-size check HAPPENED across the
            supplied attributes. The same three states, and the same meaning, as
            :attr:`GateDecision.small_sample_check_ran`, because the two decisions
            are read on the same surfaces and must not answer the same question
            differently. ``True``: every supplied attribute's groups were compared
            to ``min_group_size``, so an empty ``small_sample_warnings`` list is a
            measurement. ``False``: the scan did not run over them (which is every
            ``check_single_attributes=False`` hierarchy), or it ran against a bound
            nothing could fall below (``min_group_size`` 1 or less, recorded on
            ``small_sample_vacuous_minimum``; W3, 2026-09-30), so an empty list means
            NOT CHECKED. ``None``: nobody recorded either way, which is what a
            hand-built :class:`IntersectionalGateDecision` says about itself.

            W2 A-operations-1, 2026-09-29. This field did not exist, and
            :meth:`FairnessReportCard._format_small_sample_state` returned ``[]``
            for every hierarchical decision on the stated premise that "an
            IntersectionalGateDecision carries no such field, and saying nothing is
            correct for it rather than claiming a check that does not apply". The
            check DOES apply: ``evaluate_hierarchical`` runs it, but only inside
            ``if hconfig.check_single_attributes:``, so the opt-out path laundered
            the caveat. Measured on HEAD over two clean groups of 100, with the
            scan off and with it on::

                report 'Small-Sample Check' False   'NOT RUN' False
                card   'Small-Sample Check' False   'NOT RUN' False
                to_dict()['markdown'] any token     False
                to_github_comment_payload()['body'] False
                hasattr(decision, 'small_sample_check_ran')  False

            while the FLAT twin on the same question printed "### Small-Sample
            Check / NOT RUN" on both of its surfaces. A hierarchy whose group-size
            check never ran was byte-identical to one that ran clean, on all four
            graded surfaces at once.
        small_sample_unsized: The group sets whose sizes were NEVER compared to
            ``min_group_size``, named, so the answer above can be read for what it
            covers. Recorded as the levels run, at the point the sizing happens or
            fails to.

            W3, 2026-09-30. ``small_sample_check_ran`` is True when every supplied
            ATTRIBUTE's groups were compared, which is what its own docstring says
            and what ``evaluate_hierarchical`` computes. The INTERSECTION cells are
            sized in a different loop that never fed it, and the four graded
            surfaces rendered the True as a universal claim. Measured on HEAD with
            two attributes whose single-attribute groups are all 100 rows and whose
            four intersection cells are 95/5/5/95, against a minimum of 30::

                check_intersections=False -> small_sample_check_ran True,
                    warnings [], approved True, and the card printed
                    "RAN, and every group was at or above the minimum. This is a
                    measurement, not an absence of one."
                check_intersections=True  -> the same data names ('f_y', 5) and
                    ('m_x', 5) against the minimum of 30

            so the card asserted a measurement about groups nothing had measured,
            and the two runs disagreed about one question on the same data. The
            claim is now carried with its scope on every surface; an empty list
            means nothing was left unsized.
        warnings: The non-blocking could-not-checks collected from every level.

            BGL5 A-operations-1, 2026-09-27. These two fields did not exist, and
            :meth:`ModelFairnessGate.evaluate_hierarchical` built both lists and
            then dropped them at this constructor, which accepted neither.
            Measured on HEAD, ``evaluate_hierarchical(y, y, {})``::

                summary            'BLOCKED - 0/0 levels passed'
                to_dict()          no blocking_reasons key at all
                to_markdown_report '# Hierarchical Fairness Gate Report /
                                    **Status**: BLOCKED / **Levels evaluated**: 0'
                report card        '**Result**: Deployment blocked'
                'no level was evaluated' present on any surface: False
                'nothing was checked'   present on any surface: False

            So a could-not-check was published as BLOCKED, the FAIL state, with
            the sentence explaining it visible nowhere, while the flat
            :class:`GateDecision` carried exactly that sentence in its own
            Blocking Issues section. The grade for that method quoted the reason
            as observed evidence; no consumer could observe it.
    """

    approved: bool
    status: GateStatus
    level_results: Dict[str, GateDecision] = field(default_factory=dict)
    small_sample_warnings: List[SmallSampleWarning] = field(default_factory=list)
    summary_decision: Optional[GateDecision] = None
    hierarchical_config: Optional[HierarchicalGateConfig] = None
    blocking_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Defaults to None, never False, for the same reason GateDecision's copy
    # does: a decision nobody annotated has not established that the check was
    # skipped, only that nothing says so.
    small_sample_check_ran: Optional[bool] = None
    # Empty means "nothing was left unsized", which is a different statement from
    # small_sample_check_ran and cannot be derived from it: the flag answers for
    # the ATTRIBUTES, this answers for every group set the data defines.
    small_sample_unsized: List[str] = field(default_factory=list)
    # The hierarchy's own copy of the flat field, because hconfig.min_group_size
    # is the bound ITS scan applies and it can be just as unbreachable.
    small_sample_vacuous_minimum: Optional[int] = None

    @property
    def summary(self) -> str:
        n_levels = len(self.level_results)
        n_passed = sum(1 for d in self.level_results.values() if d.approved)
        n_warnings = len(self.small_sample_warnings)
        base = f"{n_passed}/{n_levels} levels passed"
        if n_warnings:
            base += f", {n_warnings} small-sample warning(s)"
        if self.approved:
            return f"APPROVED - {base}"
        return f"BLOCKED - {base}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "status": self.status.value,
            "level_results": {k: v.to_dict() for k, v in self.level_results.items()},
            "small_sample_warnings": [w.to_dict() for w in self.small_sample_warnings],
            # The dataclass declares six fields and this serialiser emitted five:
            # summary_decision was dropped silently, so a hierarchical decision
            # written to JSON lost its summary arm with nothing to indicate it had
            # one. Present as None when there is no summary, which is a stated
            # absence rather than a missing key.
            "summary_decision": self.summary_decision.to_dict() if self.summary_decision else None,
            "hierarchical_config": self.hierarchical_config.to_dict()
            if self.hierarchical_config
            else None,
            # The reason a JSON consumer could not read at all before 2026-09-27:
            # the dict said {"approved": false, "status": "blocked"} and nothing
            # about WHY, so a gate that could not evaluate one single level was
            # indistinguishable from one that measured a real breach.
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
            # The companion key that makes small_sample_warnings readable, the
            # same one GateDecision.to_dict has carried since 2026-09-27. Without
            # it a hierarchy whose group-size scan never ran serialised
            # byte-identically to one that scanned and found nothing.
            "small_sample_check_ran": self.small_sample_check_ran,
            # The bound-shaped cause of a False, as on the flat twin.
            "small_sample_vacuous_minimum": self.small_sample_vacuous_minimum,
            # And the SCOPE of that answer (W3, 2026-09-30): a True above means
            # every supplied attribute was compared, never every group in the
            # data, and the difference is exactly where the intersection cells of
            # five people live.
            "small_sample_unsized": list(self.small_sample_unsized),
        }

    def to_markdown_report(self) -> str:
        lines = [
            "# Hierarchical Fairness Gate Report",
            "",
            f"**Status**: {self.status.value.upper()}",
            f"**Levels evaluated**: {len(self.level_results)}",
            "",
        ]

        for level_name, decision in self.level_results.items():
            icon = {
                GateStatus.APPROVED: "pass",
                GateStatus.CONDITIONAL: "warn",
                GateStatus.BLOCKED: "fail",
            }
            status_icon = icon.get(decision.status, "?")
            lines.append(f"## {level_name} [{status_icon}]")
            lines.append("")
            lines.append("| Metric | Value | Threshold | Status |")
            lines.append("|--------|-------|-----------|--------|")
            for ev in decision.metric_evaluations:
                # The flat twin's mapping, from the one shared resolver, so the
                # two reports cannot answer this differently: this renderer's own
                # comment below records a nan row published as "Fail" and fixed
                # only at the page level.
                st = {
                    _ROW_PASS: "Pass",
                    _ROW_WARN: "Warn",
                    _ROW_FAIL: "Fail",
                    _ROW_UNMEASURED: "Could not check",
                    _ROW_REFUSED: "Refused (not by this bound)",
                }[_metric_row_state(ev)]
                thr = f"{ev.threshold:.4f}" if ev.threshold is not None else "N/A"
                lines.append(
                    f"| {ev.metric_name} | {_metric_value_cell(ev.value)} | {thr} | {st} |"
                )
            # Keyed to the level AND the metric, which the page-level list below
            # cannot be: it printed the same sentence once per level.
            lines.extend(_metric_note_lines(decision.metric_evaluations))
            lines.append("")

        # MIRRORS GateDecision.to_markdown_report, which has carried these two
        # sections all along. This renderer had neither, so the fail-closed
        # sentence had nowhere to land: measured 2026-09-27 on
        # evaluate_hierarchical(y, y, {'gender': ['a'] * 200}), one group so no
        # disparity is definable, the whole page was '## overall [fail] |
        # demographic_parity_difference | nan | 0.5000 | Fail |' and the strings
        # 'could not be computed', 'fails closed' and 'not measured' were all
        # absent. A metric nobody could compute was published as a measured
        # breach. After: the same page carries '## Blocking Issues' listing
        # 'demographic_parity_difference could not be computed on this data; the
        # gate fails closed rather than approving an unevaluated metric'.
        if self.blocking_reasons:
            lines.append("## Blocking Issues")
            lines.append("")
            for reason in self.blocking_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        if self.small_sample_warnings:
            lines.append("## Small-Sample Warnings")
            lines.append("")
            for w in self.small_sample_warnings:
                lines.append(
                    f"- **{w.group_name}**: {w.sample_size} samples (min {w.minimum_recommended})"
                )
            lines.append("")
            lines.extend(_unsized_scope_lines(self.small_sample_unsized))
        else:
            # THE FLAT TWIN'S THIRD SECTION, which this renderer did not have at
            # all (W2 A-operations-1, 2026-09-29). GateDecision.to_markdown_report
            # has carried this three-state block since 2026-09-27 and its own
            # comment states the reading it exists to prevent: "An absent
            # Small-Sample Warnings section read as 'checked, every group was big
            # enough' on a decision where the check could not run at all." This
            # renderer printed the section only when the list was non-empty and
            # said nothing otherwise, which is exactly that reading. Same three
            # states and the same wording, so the two reports cannot say
            # different things about the same question.
            lines.append("## Small-Sample Check")
            lines.append("")
            if self.small_sample_check_ran is True:
                # THE TRUE STATE, WHICH THIS RENDERER USED TO LEAVE BLANK (W3,
                # 2026-09-30), for the reason recorded on the flat twin above. The
                # sentence is SCOPED when something was left unsized, because "every
                # group" was a claim about groups nothing had sized: see
                # small_sample_unsized for the 95/5/5/95 measurement.
                if self.small_sample_unsized:
                    lines.append(
                        "- RAN over every group set it covered, and each was at or above "
                        "the minimum. This is a measurement, not an absence of one."
                    )
                else:
                    lines.append(
                        "- RAN, and every group was at or above the minimum. This is a "
                        "measurement, not an absence of one."
                    )
            elif self.small_sample_vacuous_minimum is not None:
                # Same precedence as the flat twin: the attributes WERE scanned, and
                # against a bound that could not fire, so naming the switch would be
                # the wrong cause.
                lines.append(_vacuous_minimum_sentence(self.small_sample_vacuous_minimum))
            elif self.small_sample_check_ran is False:
                lines.append(
                    "- NOT RUN: no supplied attribute's group sizes were compared to the "
                    "minimum (check_single_attributes is off), so the empty warning list "
                    "means the check did not happen, NOT that every group was large "
                    "enough."
                )
            else:
                lines.append(
                    "- NOT RECORDED: nothing on this decision states whether group sizes "
                    "were checked, so the empty warning list cannot be read either way."
                )
            lines.extend(_unsized_scope_lines(self.small_sample_unsized))
            lines.append("")

        return "\n".join(lines)


class ModelFairnessGate:
    """Deployment gate that enforces fairness requirements.

    This class acts as a gate in CI/CD pipelines, evaluating model fairness
    and deciding whether deployment should proceed.

    Metrics :meth:`evaluate` can compute from raw data:
        - ``demographic_parity_difference`` (worst-case selection-rate gap)
        - ``disparate_impact_ratio`` (the four-fifths rule: min/max selection-rate
          ratio, breaching BELOW its threshold, so 0.80 is a FLOOR)
        - ``equalized_odds_difference`` (TPR gap)
        - ``false_positive_rate_difference``
        - ``predictive_parity_difference`` (PPV gap)

    Metrics :meth:`evaluate` CANNOT compute:
        Every other metric name in the library, including the calibration,
        AUROC, equal-opportunity and multicalibration families. Configuring one
        of those as a threshold here does NOT silently pass: the gate reports it
        as could-not-check and refuses approval, which blocks every deployment
        until either the name is corrected or a ``compute_metrics_fn`` supplies
        it. To gate on them, pass ``compute_metrics_fn`` (or pre-compute the
        metrics and use :meth:`evaluate_from_metrics`); the full library set is
        available from ``vfairness.evaluation.vfairness_metrics``.

    EVERY metric in ``metrics`` needs its own entry in ``thresholds``:
        A configured metric with no threshold was never compared to anything,
        so it is reported as could-not-check and refuses approval, exactly like
        a metric that could not be computed. It is never a pass. The example
        below used to configure two metrics and one threshold, and measured on
        data with TPR 1.00 vs 0.10 and FPR 0.00 vs 0.90 it returned
        ``approved=True`` with ``equalized_odds_difference`` at 0.9000 marked
        "Pass" in the audit trail. A threshold of ``0.0`` is a real
        zero-tolerance policy and is still enforced; it is ABSENCE, not
        falsiness, that fails closed.

    Attributes:
        config: Gate configuration.
        compute_metrics_fn: Optional custom function to compute metrics.

    Example:
        >>> gate = ModelFairnessGate(
        ...     metrics=['demographic_parity_difference', 'equalized_odds_difference'],
        ...     thresholds={
        ...         'demographic_parity_difference': 0.1,
        ...         'equalized_odds_difference': 0.1,  # every configured metric needs one
        ...     },
        ...     require_improvement=True
        ... )
        >>> decision = gate.evaluate(y_true, y_pred, protected_attr, baseline_metrics)
        >>> if decision.approved:
        ...     deploy_model()
        ... else:
        ...     block_deployment(decision.blocking_reasons)

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

    Ledger row: fairness_gate. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        require_improvement: bool = False,
        improvement_margin: float = 0.0,
        blocking_metrics: Optional[List[str]] = None,
        config: Optional[GateConfig] = None,
        compute_metrics_fn: Optional[Callable] = None,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    ):
        """Initialize the ModelFairnessGate.

        Args:
            metrics: List of fairness metrics to evaluate.
            thresholds: Dictionary mapping metric names to threshold values.
            require_improvement: Whether to require improvement over baseline.
            improvement_margin: Minimum improvement required (as fraction).
            blocking_metrics: Metrics that block deployment if failed.
            config: Full configuration object (overrides other args).
            compute_metrics_fn: Custom function to compute fairness metrics.
                Signature: fn(y_true, y_pred, protected_attr) -> Dict[str, float]
            min_group_size: Smallest group :meth:`evaluate` will certify a
                comparison on (default 30, the library-wide minimum). Small
                groups are still measured and reported; the gate withholds
                APPROVED on them rather than dropping them.
        """
        if config is not None:
            self.config = config
        else:
            # `is None`, not falsiness. `metrics or [...]` and `thresholds or
            # {...}` cannot tell "the caller said nothing" from "the caller said
            # NOTHING is configured", and they answered both with an invented
            # default. Measured 2026-09-10:
            #   ModelFairnessGate(metrics=['demographic_parity_difference'],
            #                     thresholds={})
            #   -> config.thresholds == {'demographic_parity_difference': 0.1}
            #   -> a value of 0.05 APPROVED against a bound nobody set.
            #   ModelFairnessGate(metrics=[], ...)
            #   -> config.metrics == ['demographic_parity_difference'], so the
            #      "the gate has no metrics configured" fail-closed guard in
            #      evaluate() could never fire through this constructor.
            # An empty dict now reaches the loop and each metric is reported as
            # could-not-check; an empty list reaches the no-metrics guard.
            self.config = GateConfig(
                metrics=(["demographic_parity_difference"] if metrics is None else list(metrics)),
                thresholds=(
                    {"demographic_parity_difference": 0.1}
                    if thresholds is None
                    else dict(thresholds)
                ),
                require_improvement=require_improvement,
                improvement_margin=improvement_margin,
                blocking_metrics=blocking_metrics,
                min_group_size=min_group_size,
            )

        self.compute_metrics_fn = compute_metrics_fn

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        baseline_metrics: Optional[Dict[str, float]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """Evaluate model fairness and make deployment decision.

        Without a ``compute_metrics_fn`` this measures exactly the five metrics
        listed on the class docstring: ``demographic_parity_difference``,
        ``disparate_impact_ratio``, ``equalized_odds_difference``,
        ``false_positive_rate_difference`` and ``predictive_parity_difference``.
        Any other configured metric name is reported as could-not-check and
        refuses approval; it is never treated as a pass. Supply
        ``compute_metrics_fn`` to gate on anything else.

        Groups smaller than ``config.min_group_size`` are NOT excluded from the
        computation: the measured numbers are reported as they were computed,
        and are also listed on ``decision.small_sample_warnings``. What is
        withheld is the VERDICT. A metric whose comparison rests on such a group
        is routed to the same could-not-check outcome as a NaN: a blocking
        metric refuses approval (BLOCKED), a non-blocking one downgrades to
        CONDITIONAL with a warning. It is never APPROVED.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            protected_attr: Protected attribute values.
            baseline_metrics: Optional baseline metrics for comparison.
            model_metadata: Optional metadata about the model.

        Returns:
            GateDecision with approval status and details.
        """
        # Compute fairness metrics
        if self.compute_metrics_fn:
            computed_metrics = self.compute_metrics_fn(y_true, y_pred, protected_attr)
        else:
            computed_metrics = self._compute_default_metrics(y_true, y_pred, protected_attr)

        metric_evaluations = []
        blocking_reasons = []
        warnings = []

        # C-05, third call site. The loop below fails closed on a configured
        # metric that could not be computed, but an EMPTY config configures
        # nothing, so the loop never runs and the gate approves. Measured
        # 2026-09-10 on THIS entry point: GateConfig(metrics=[], thresholds={})
        # with evaluate(y_true, y_pred, protected_attr) returned approved=True,
        # "APPROVED - All fairness requirements met", and create_github_check
        # reported conclusion "success". The identical guard already stood in
        # evaluate_from_metrics and evaluate_hierarchical; the raw-data entry
        # point the class docstring advertises was the one still missing it.
        if not self.config.metrics:
            blocking_reasons.append(
                "the gate has no metrics configured, so nothing was checked; it "
                "fails closed rather than approving an unevaluated model"
            )

        # A BLOCKING REQUIREMENT ON A METRIC NOBODY GATES (W4, 2026-09-30). The
        # sibling of the empty-config guard above: that one catches a gate with NO
        # requirements, and this one a requirement that reaches no comparison.
        # `is not None` because None means "every metric blocks" and names nothing,
        # and an EMPTY list is the documented warn-only mode and names nothing
        # either. See _unmatched_blocking_message for the three measured rows.
        declared_blocking = self.config.blocking_metrics
        if declared_blocking is not None:
            unmatched_blocking = [n for n in declared_blocking if n not in self.config.metrics]
            if unmatched_blocking:
                blocking_reasons.append(
                    _unmatched_blocking_message(unmatched_blocking, list(self.config.metrics))
                )

        # Evaluate each configured metric
        for metric_name in self.config.metrics:
            if metric_name not in computed_metrics:
                # A configured metric that could not be computed (e.g. fewer
                # than two groups so _compute_default_metrics returned {}, or a
                # custom compute_metrics_fn omitted it) is a could-not-check,
                # not a pass. Previously this only warned and continued, so a
                # BLOCKING metric that was never evaluated still let the gate
                # APPROVE deployment. Fail closed: a blocking metric refuses
                # approval, a non-blocking one downgrades to CONDITIONAL, exactly
                # like the NaN insufficient-evidence case below.
                is_blocking = (
                    self.config.blocking_metrics is None
                    or metric_name in self.config.blocking_metrics
                )
                message = (
                    f"{metric_name} could not be computed on this data; the gate "
                    f"fails closed rather than approving an unevaluated metric"
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=float("nan"),
                        threshold=self.config.thresholds.get(metric_name),
                        passed=False,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            value = computed_metrics[metric_name]
            threshold = self.config.thresholds.get(metric_name)
            baseline_value = baseline_metrics.get(metric_name) if baseline_metrics else None

            # Determine if this metric is blocking
            is_blocking = True
            if self.config.blocking_metrics is not None:
                is_blocking = metric_name in self.config.blocking_metrics

            # A NaN metric could not be measured on this data (undefined group
            # rate, degenerate upstream data, or a custom compute_metrics_fn
            # reporting insufficient evidence). NaN fails every comparison, so
            # it used to fall through the threshold check with passed=True and
            # APPROVE deployment on an unmeasured metric. Fail closed instead:
            # route it to an explicit could-not-check outcome. It is neither a
            # PASS nor a numeric threshold FAIL; a blocking metric refuses
            # approval (BLOCKED), a non-blocking one downgrades to CONDITIONAL.
            # INFINITY, added READINESS-6 (2026-09-10). NaN was guarded here
            # and infinity was not, and they are opposite halves of one hole:
            # NaN loses every comparison, inf WINS them. With every layer
            # NaN-only, disparate_impact_ratio=inf against the four-fifths
            # threshold of 0.8 gave status=APPROVED, approved=True, passed=True,
            # an EMPTY message and a GitHub check conclusion of "success",
            # indistinguishable from the genuine pass at 0.95. inf reaches the
            # ratio family from a zero denominator, i.e. a group with NO
            # SELECTIONS AT ALL.
            #
            # THIS GUARD IS DEFENCE IN DEPTH, NOT THE CARRIER. Measured by
            # sabotage: reinstating the NaN-only test HERE changes nothing on
            # its own, because check_threshold now answers COULD_NOT_CHECK for
            # a non-finite value and this gate fails closed on that outcome. It
            # is kept so the two paths cannot drift apart, and so the operator
            # gets the fail-closed sentence below rather than a bare
            # could-not-measure. Both facts are pinned in
            # tests/test_readiness6_infinity.py. Saying which layer actually
            # stops the defect matters: an earlier version of this comment
            # claimed this line was what changed the outcome, and it was not.
            if not is_measured(value):
                message = (
                    f"{metric_name} could not be measured on this data "
                    f"({unmeasurable_reason(value)}); the gate fails closed "
                    f"rather than approving an unmeasured metric"
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=value,
                        threshold=threshold,
                        passed=False,
                        baseline_value=baseline_value,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            # A metric configured on the gate with NO threshold was never
            # COMPARED to anything, and `passed = True` was the neutral default
            # the skipped comparison left standing. Measured 2026-09-10 with the
            # configuration copied VERBATIM from this class's own docstring
            # example (two metrics, one threshold), on data with TPR 1.00 vs 0.10
            # and FPR 0.00 vs 0.90:
            #     equalized_odds_difference: value=0.9000 threshold=None passed=True
            #     approved: True | status: APPROVED | github check: success
            #     audit trail row: | equalized_odds_difference | 0.9000 | N/A | Pass |
            # It reproduced identically with thresholds={}, with the key PRESENT
            # holding None, and with a threshold naming a different metric.
            # FairnessTestSuite handles the same input honestly (SKIPPED, and
            # .passed is False); this gate, the surface that decides whether the
            # model ships, was the one calling it a pass.
            #
            # An unconfigured bound is a could-not-check, exactly like the NaN
            # and absent-metric cases above, and it is routed to the same
            # fail-closed outcome: blocking metric -> BLOCKED, non-blocking ->
            # CONDITIONAL with a warning. `is None`, never falsiness: a
            # threshold of 0.0 is a real zero-tolerance policy (check_threshold
            # enforces it for a lower-is-better metric and reports the
            # unbreachable higher-is-better mirror as could-not-check), and
            # `if not threshold` would silently switch it off.
            if threshold is None:
                message = (
                    f"{metric_name} has no threshold configured, so it was never "
                    f"compared against anything; the gate fails closed rather than "
                    f"approving an unchecked metric. Add it to `thresholds`, or "
                    f"remove it from `metrics` if it is not being gated on."
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=value,
                        threshold=None,
                        passed=False,
                        baseline_value=baseline_value,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            # Evaluate against threshold.
            #
            # The direction is resolved once, in the shared resolver: a ratio
            # metric (four-fifths rule) breaches BELOW its threshold, a
            # difference metric ABOVE it, and a metric whose direction cannot
            # be determined cannot be checked at all. This used to be a bare
            # `abs(value) > threshold` commented "for difference metrics,
            # lower is better", which treated EVERY metric as a difference and
            # so INVERTED the verdict for the whole ratio family:
            # disparate_impact_ratio 0.00 (the protected group is never
            # selected) was APPROVED and written into the markdown audit trail
            # as "Pass", while 1.00 (perfect parity) was BLOCKED.
            # COULD_NOT_CHECK is failed closed here, exactly like the NaN and
            # absent-metric cases above: it is never a pass.
            passed = True
            message = ""

            outcome, threshold_message = check_threshold(metric_name, value, threshold)
            if outcome is not ThresholdOutcome.PASS:
                passed = False
                message = threshold_message
            else:
                # VACUOUS-THRESHOLD GUARD, the wider rule check_threshold's own
                # degenerate-bound test is one case of: a bound is unusable when the
                # comparison it controls cannot be False for any data in the METRIC'S
                # RANGE. check_threshold refuses the two cases it can see without a
                # range (a required minimum at or below zero, a negative maximum) and
                # returned PASS for a maximum of 1.2, or 60.0, on a rate gap that
                # cannot exceed 1.0. See _vacuous_bound_message for the four measured
                # multiplier rows, including a gap of 1.0000, the largest a disparity
                # can be, APPROVED against 60.0.
                #
                # IN THE ELSE ARM, and that is not a shortcut: a bound nothing can
                # breach cannot have produced a FAIL, so PASS is the only outcome it
                # can reach here, and when check_threshold has already refused, its
                # reason is the more precise one and must stand (the same rule the
                # baseline and small-sample blocks below follow).
                #
                # THIS ONE GUARD COVERS THREE BOUNDS. The relaxed intersectional
                # threshold and any per_intersection_thresholds override both arrive
                # here as that level's own threshold, because every level of
                # evaluate_hierarchical runs through this method; the base threshold
                # is the third. A guard placed at the relaxation site would have
                # covered one of the three.
                vacuous_threshold = vacuous_bound_reason(
                    metric_name, threshold, BoundRole.THRESHOLD
                )
                if vacuous_threshold:
                    passed = False
                    message = _vacuous_bound_message("threshold", metric_name, vacuous_threshold)

            # BASELINE GUARD, ABOVE BOTH BASELINE CHECKS, because both share the
            # precondition and a guard inside either one cannot cover the other.
            # See _unmeasurable_baseline_message for the measured before and
            # after: a NaN baseline returned approved=True / "APPROVED - All
            # fairness requirements met" / improvement=nan / markdown "Pass" /
            # GitHub conclusion "success" with require_improvement=True, and it
            # now returns approved=False / BLOCKED with the reason named.
            # `improvement < margin` and `degradation > margin` are BOTH False
            # for NaN, so one unguarded baseline switched off two requirements.
            #
            # is_measured, not `math.isnan`: an INFINITE baseline is the same
            # hole running the other way (every comparison against it wins), and
            # that is the pairing READINESS-6 recorded on the value guard above.
            #
            # AND AN ABSENT BASELINE IS THE THIRD HALF OF THE SAME HOLE (W2
            # A-operations-1, 2026-09-29). The precondition used to read
            # `baseline_value is None or is_measured(baseline_value)`, which
            # called a baseline that is NOT THERE measurable, so a configured
            # requirement was skipped in silence. See
            # _absent_baseline_message for the six measured rows: all four
            # no-baseline shapes (arg omitted, empty dict, key missing, key
            # present holding None) returned approved=True / "APPROVED - All
            # fairness requirements met" / GitHub 'success' while the NaN row
            # beside them blocked. `is None` has to mean could-not-check here,
            # not "nothing was required": whether anything was required is
            # `baseline_required`, and it is a separate question.
            improvement = None
            baseline_absent = baseline_value is None
            baseline_measurable = not baseline_absent and is_measured(baseline_value)
            baseline_required = self.config.require_improvement or (
                self.config.allow_degradation_margin is not None
            )

            # MARGIN GUARD, ABOVE BOTH COMPARISONS, for the same reason the baseline
            # guard below is above both: they share the precondition, and a guard
            # inside either comparison cannot cover the other. The baseline guard
            # covers the LEFT operand of `improvement < margin`; this covers the
            # RIGHT one, which had no guard anywhere in either entry point. See
            # _unusable_margin_message for the measured before and after.
            #
            # is_measured, not math.isnan: an INFINITE improvement margin makes the
            # requirement impossible to SATISFY rather than impossible to check, and
            # an infinite degradation margin allows every degradation. Both are
            # unusable bounds and both are silent.
            #
            # AND THE FINITE HALF OF THE SAME HOLE (2026-09-30). is_measured closes
            # NaN and the two infinities; every FINITE bound that no data can breach
            # stayed open, and the refusal sentence above states the criterion those
            # bounds meet exactly. improvement_margin=-1e9 and
            # allow_degradation_margin=1e9 each APPROVED a measured degradation of
            # 0.5 with zero warnings. One rule for both halves, one predicate for
            # all four bounds: see _vacuous_bound_message.
            unusable_margins = []
            if self.config.require_improvement:
                if not is_measured(self.config.improvement_margin):
                    unusable_margins.append(
                        _unusable_margin_message(
                            "required improvement",
                            metric_name,
                            self.config.improvement_margin,
                        )
                    )
                else:
                    vacuous_margin = vacuous_bound_reason(
                        metric_name,
                        self.config.improvement_margin,
                        BoundRole.REQUIRED_IMPROVEMENT,
                    )
                    if vacuous_margin:
                        unusable_margins.append(
                            _vacuous_bound_message(
                                "required improvement", metric_name, vacuous_margin
                            )
                        )
            if self.config.allow_degradation_margin is not None:
                if not is_measured(self.config.allow_degradation_margin):
                    unusable_margins.append(
                        _unusable_margin_message(
                            "allowed degradation",
                            metric_name,
                            self.config.allow_degradation_margin,
                        )
                    )
                else:
                    vacuous_margin = vacuous_bound_reason(
                        metric_name,
                        self.config.allow_degradation_margin,
                        BoundRole.ALLOWED_DEGRADATION,
                    )
                    if vacuous_margin:
                        unusable_margins.append(
                            _vacuous_bound_message(
                                "allowed degradation", metric_name, vacuous_margin
                            )
                        )
            for margin_message in unusable_margins:
                # The same fail-closed decision point as the baseline guard, and the
                # same routing: a blocking metric refuses approval, one already
                # failing keeps the more precise reason and carries this as a warning
                # rather than dropping it.
                if passed:
                    passed = False
                    message = margin_message
                elif margin_message not in warnings:
                    warnings.append(margin_message)
            if not baseline_measurable and baseline_required:
                # Two shapes of the same could-not-check, named apart because
                # the operator's next action differs: a non-finite baseline came
                # from a broken measurement, an absent one was never handed over.
                baseline_message = (
                    _absent_baseline_message(metric_name, value)
                    if baseline_absent
                    else _unmeasurable_baseline_message(metric_name, value, baseline_value)
                )
                # The comparison was attempted and could not be made: NaN, not
                # None, which is this dataclass's "no baseline comparison asked
                # for" and would read as nothing having been required.
                improvement = float("nan")
                if passed:
                    # The fail-closed decision point. A blocking metric refuses
                    # approval, a non-blocking one downgrades to CONDITIONAL;
                    # the shared tail below routes it.
                    passed = False
                    message = baseline_message
                else:
                    # Already failing on the threshold, whose reason is the more
                    # precise one (the small-sample block below follows the same
                    # rule). The unchecked requirement is still reported rather
                    # than dropped, as a warning beside the blocking reason.
                    warnings.append(baseline_message)

            # Check improvement requirement. "Improved" is direction-aware: a
            # ratio improves when it RISES toward parity, a difference when its
            # magnitude falls. For a metric whose direction is unknown the only
            # thing left is the historical magnitude comparison; the fail-closed
            # decision point for an unrecognised metric is the threshold check
            # above.
            if (
                baseline_value is not None
                and baseline_measurable
                and self.config.require_improvement
            ):
                improvement = improvement_amount(metric_name, value, baseline_value)
                if improvement is None:
                    # See _unknown_direction_message: this substituted
                    # abs(baseline_value) - abs(value) and decided `passed` from it.
                    unknown = _unknown_direction_message(
                        "required improvement", metric_name, value, baseline_value
                    )
                    # NaN, not None, and for the same reason the baseline guard
                    # above gives: None is this dataclass's "no baseline comparison
                    # was asked for", and leaving it None here would make an
                    # unevaluable requirement indistinguishable from one nobody
                    # configured. A first version of this fix did exactly that, and
                    # the control test caught it: with require_improvement=False
                    # the field is also None, so the two states had collapsed.
                    improvement = float("nan")
                    if passed:
                        passed = False
                        message = unknown
                    elif unknown not in warnings:
                        warnings.append(unknown)
                elif improvement < self.config.improvement_margin:
                    if passed:  # Only fail if we haven't already failed on threshold
                        passed = False
                        message = f"{metric_name} did not improve by required margin"

            # Check degradation (the negative of the improvement, same direction
            # handling: a ratio degrades when it FALLS away from parity, which the
            # old abs(value) - abs(baseline_value) scored as an improvement).
            #
            # `is not None`, not `> 0`: under the old guard a margin of 0.0 (the
            # default, and the strictest-looking value a user can type) switched
            # this check off entirely. See GateConfig.allow_degradation_margin
            # for the measured 0.09-against-0.01 case. `degradation > margin`
            # stays STRICT, so a margin of 0.0 blocks any real degradation while
            # an unchanged metric (degradation 0.0) still passes.
            degradation_margin = self.config.allow_degradation_margin
            if (
                baseline_value is not None
                and baseline_measurable
                and degradation_margin is not None
            ):
                improved = improvement_amount(metric_name, value, baseline_value)
                if improved is None:
                    # The same substitution with the sign flipped, and the same
                    # closure. See _unknown_direction_message.
                    unknown = _unknown_direction_message(
                        "allowed degradation", metric_name, value, baseline_value
                    )
                    # Same reasoning, and it matters more here: this block does not
                    # otherwise write to `improvement` at all, so without this the
                    # row published None both for "no degradation margin set" and
                    # for "the margin could not be checked".
                    improvement = float("nan")
                    if passed:
                        passed = False
                        message = unknown
                    elif unknown not in warnings:
                        warnings.append(unknown)
                elif -improved > degradation_margin:
                    passed = False
                    message = f"{metric_name} degraded beyond allowed margin"

            evaluation = MetricEvaluation(
                metric_name=metric_name,
                value=value,
                threshold=threshold,
                passed=passed,
                baseline_value=baseline_value,
                improvement=improvement,
                is_blocking=is_blocking,
                message=message,
            )
            metric_evaluations.append(evaluation)

            # Record blocking reasons and warnings
            if not passed:
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)

        # Small-sample: the numbers were computed, the COMPARISON was not
        # certifiable.
        #
        # Measured 2026-09-10 from a clean-room wheel install, group 'a' n=100 at
        # selection rate 0.40 against group 'b' n=5 at 0.40 (2 of 5 people):
        # this method returned approved=True, status APPROVED, warnings=[],
        # demographic_parity_difference 0.0000 and disparate_impact_ratio 1.0000,
        # wrote "Pass" for both into the markdown audit trail, and
        # create_github_check reported conclusion "success". Every other surface
        # in the same library refused the same comparison: the metric functions
        # returned NaN, report()['assessment']['assessable'] was False, and
        # assert_fairness raised "NOT MEASURABLE: value is nan, so the metric
        # certifies nothing". The gate, the surface that decides whether the
        # model ships, was the one certifying it. A five-person group produced
        # the textbook perfect-fairness pair.
        #
        # The fix is NOT to start excluding small groups: `_compute_default_metrics`
        # deliberately compares every group present (min_group_size=1), because
        # silently dropping a minority group is the failure the metric layer's
        # own convention comment warns about. Every group is still measured and
        # every measured number is still reported. What is withheld is the
        # VERDICT, routed to exactly the could-not-check outcome the NaN and
        # absent-metric branches above already use: blocking metric -> BLOCKED,
        # non-blocking metric -> CONDITIONAL with a warning. Three states, and
        # APPROVED is not one of them here.
        #
        # Until this existed, the warning behaviour the module docstring and
        # HierarchicalGateConfig advertise lived only on evaluate_hierarchical,
        # and even there the SmallSampleWarning list never reached the verdict:
        # measured on the same data, evaluate_hierarchical returned approved=True
        # with the summary "APPROVED - 2/2 levels passed, 1 small-sample
        # warning(s)". It now inherits the refusal from the level evaluations.
        # THE BOUND THE SCAN BELOW APPLIES, tested for the same vacuity the three
        # other gate bounds are (W3, 2026-09-30). A minimum of 1 or less cannot be
        # breached by any group count, so the scan runs, finds nothing it could have
        # found, and the surfaces published that as "every group was at or above the
        # minimum. This is a measurement, not an absence of one." See
        # _vacuous_min_group_size for the five measured rows and for why the VERDICT
        # is deliberately left alone: 1 is the only way a caller can say "do not
        # certify group sizes", and refusing it would leave no way to say that.
        vacuous_minimum = _vacuous_min_group_size(self.config.min_group_size)
        small_sample_warnings = _undersized_groups(protected_attr, self.config.min_group_size)
        if small_sample_warnings:
            detail = ", ".join(f"'{w.group_name}' n={w.sample_size}" for w in small_sample_warnings)
            for evaluation in metric_evaluations:
                if not evaluation.passed:
                    # Already failed on its own terms (threshold, NaN, absent).
                    # Its own reason is the more precise one; the group sizes are
                    # carried on decision.small_sample_warnings either way.
                    continue
                message = (
                    f"{evaluation.metric_name} was computed across a group below the "
                    f"minimum group size of {self.config.min_group_size} ({detail}), so "
                    f"the comparison could not be certified; the gate reports "
                    f"could-not-check rather than approving an uncertifiable comparison"
                )
                evaluation.passed = False
                evaluation.message = message
                if evaluation.is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)

        # THE SAME FLOOR, ON THE COUNT THE RATE IS ACTUALLY COMPUTED OVER (W4,
        # 2026-09-30). The scan above counts the ROWS of each group, and three of the
        # five metrics this gate computes rest on a labelled SUBSET of those rows, so
        # a group of 100 rows holding one positive label had its true-positive rate
        # certified on one person while every surface said every group was at or
        # above the minimum. See _undersized_rate_denominators for the five measured
        # rows, including the row-count control that is REFUSED at 29 rows while the
        # one-label comparison beside it was approved.
        #
        # PER METRIC, not the blanket refusal above, and that is the difference
        # between a fix and an over-correction: a demographic_parity_difference is
        # computed over the whole group, so a group with few positives says nothing
        # about it and it keeps its verdict. Each metric is refused for its own
        # denominators and nothing else.
        #
        # AFTER the row scan, because a group that is itself too small has already
        # refused every metric with the more precise reason, and its legs are
        # smaller still. skip_groups carries those groups so one fact is not
        # reported in two wordings.
        rate_unsized: List[str] = []
        if self.compute_metrics_fn is None:
            row_floor_groups = frozenset(w.group_name for w in small_sample_warnings)
            for evaluation in metric_evaluations:
                legs = _undersized_rate_denominators(
                    evaluation.metric_name,
                    y_true,
                    y_pred,
                    protected_attr,
                    self.config.min_group_size,
                    skip_groups=row_floor_groups,
                )
                if legs is None:
                    # Not one of the five statistics this method computes, so it can
                    # only have reached here already refused as not computable (the
                    # absent-metric branch above, value NaN, passed False). Should
                    # that ever cease to be true, the claim above must not quietly
                    # cover it: say what was not sized.
                    if evaluation.passed:
                        rate_unsized.append(
                            f"the rows {evaluation.metric_name}'s rate was computed over "
                            f"(this gate does not know what that statistic rests on)"
                        )
                    continue
                if not legs:
                    continue
                small_sample_warnings.extend(legs)
                detail = ", ".join(f"{w.group_name} n={w.sample_size}" for w in legs)
                message = (
                    f"{evaluation.metric_name} rests on a rate measured over fewer rows "
                    f"than the minimum group size of {self.config.min_group_size} "
                    f"({detail}), so the comparison could not be certified; the gate "
                    f"reports could-not-check rather than approving a rate computed over "
                    f"a handful of labels. The GROUP is large enough: this floor counts "
                    f"the rows the rate itself was computed over, which for this metric "
                    f"is a labelled subset of the group."
                )
                if evaluation.passed:
                    evaluation.passed = False
                    evaluation.message = message
                    if evaluation.is_blocking:
                        blocking_reasons.append(message)
                    else:
                        warnings.append(message)
                elif message not in warnings:
                    # Already refused on its own terms, whose reason is the more
                    # precise one. The unsized denominator is still reported rather
                    # than dropped, the same routing the margin guard above uses.
                    warnings.append(message)
        else:
            # A compute_metrics_fn decides what rows its numbers rest on, and this
            # method cannot see inside it: y_true and y_pred may not even be binary
            # labels. Refusing on a denominator we cannot identify would be a
            # fabricated refusal, the mirror of the fabricated certification above,
            # so the scope is DISCLOSED instead and no verdict moves.
            rate_unsized.append(_unsized_rate_scope_sentence())

        # Determine final approval status
        approved = len(blocking_reasons) == 0

        if approved:
            status = GateStatus.APPROVED if not warnings else GateStatus.CONDITIONAL
        else:
            status = GateStatus.BLOCKED

        return GateDecision(
            approved=approved,
            status=status,
            metric_evaluations=metric_evaluations,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            metadata={
                "model_metadata": model_metadata,
                "computed_metrics": computed_metrics,
                "config": self.config.to_dict(),
            },
            small_sample_warnings=small_sample_warnings,
            # This path HAS the group labels, so the check ran. An empty list
            # from here genuinely means no group was below min_group_size.
            #
            # UNLESS THE BOUND COULD NOT BE BREACHED, added W3: at a minimum of 1 or
            # less the comparison is False for every group of every dataset, so an
            # empty list means NOT CHECKED here exactly as it does for a decision
            # built from bare numbers. The group labels being present is necessary
            # and not sufficient.
            small_sample_check_ran=vacuous_minimum is None,
            small_sample_vacuous_minimum=vacuous_minimum,
            # W4: what that True does NOT cover. Empty for the default path, which
            # sizes every denominator of every metric it computes.
            small_sample_unsized=rate_unsized,
        )

    def _compute_default_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """Compute default fairness metrics.

        This provides basic metric computation. For full metrics,
        use with vfairness.vfairness_metrics module.

        Emits exactly five keys, and nothing else:
        ``demographic_parity_difference``, ``disparate_impact_ratio``,
        ``equalized_odds_difference``, ``false_positive_rate_difference`` and
        ``predictive_parity_difference``. Any other metric configured on the gate
        is absent from this dict and evaluate() fails closed on it.

        A metric whose per-group rates cannot all be measured (a group with
        an empty denominator) is reported as NaN, never as a 0.0 gap;
        evaluate() then fails closed on it instead of approving.
        """
        metrics: Dict[str, float] = {}

        groups = np.unique(protected_attr)
        if len(groups) < 2:
            # A disparity needs at least two groups to compare.
            return metrics

        # Compute each rate for EVERY group and report the worst-case
        # (max - min) gap. The previous implementation truncated to the first
        # two groups (np.unique alphabetical order), so a model unfair to a
        # third or later group could sail through the gate unmeasured.
        def _rates(numerator_mask_fn) -> list:
            # A group with an empty denominator (no positives for TPR, no
            # negatives for FPR, no predicted positives for PPV) has an
            # UNDEFINED rate. Record it as NaN instead of silently dropping
            # the group: dropping made an unmeasurable group indistinguishable
            # from a perfectly fair one, so the gate reported a 0.0 gap and
            # approved deployment on rates that were never measured (mirror of
            # the fd9746b undefined-rate fix in the classification metrics).
            rates = []
            for g in groups:
                gmask = protected_attr == g
                sel, tot = numerator_mask_fn(gmask)
                if tot > 0:
                    rates.append(float(np.sum(sel)) / float(tot))
                else:
                    rates.append(float("nan"))
            return rates

        def _gap(rates: list) -> float:
            # Any undefined per-group rate makes the worst-case gap itself
            # unmeasurable: return NaN so evaluate() fails closed on it.
            # This is deliberately stricter than the report-side metrics
            # (which compare the surviving defined rates): this is the
            # deployment-blocking surface, and a gap computed while ignoring
            # an unmeasurable protected group would still wave the model
            # into production unmeasured.
            if any(r != r for r in rates):  # NaN check
                return float("nan")
            # `_compute_default_metrics` returns above when there are fewer
            # than two groups, and _rates emits one entry per group, so the
            # fewer-than-two case is unreachable from here. Re-verified by
            # execution 2026-09-10.
            #
            # The branch stays, and it answers NaN. READINESS-5: the comment
            # that used to sit here said no such branch had been written "on
            # purpose", while the line beneath it read `... else 0.0`. So the
            # code carried a guard the comment denied, and that guard handed a
            # deployment-BLOCKING surface a fabricated gap of 0.0, the cleanest
            # value on the scale, if reachability ever changed. Every sibling
            # path in this function answers NaN for a gap it could not measure
            # and lets evaluate() fail closed; this one now agrees with them.
            # An unreachable branch that fails OPEN is worse than no branch.
            return max(rates) - min(rates) if len(rates) >= 2 else float("nan")

        # Demographic parity: P(Y_hat=1 | A=g), worst-case gap across groups
        dp_rates = _rates(lambda m: (y_pred[m] == 1, np.sum(m)))
        metrics["demographic_parity_difference"] = _gap(dp_rates)

        # True positive rate gap: P(Y_hat=1 | A=g, Y=1)
        tpr_rates = _rates(lambda m: (y_pred[m & (y_true == 1)] == 1, np.sum(m & (y_true == 1))))
        tpr_gap = _gap(tpr_rates)

        # False positive rate gap: P(Y_hat=1 | A=g, Y=0)
        fpr_rates = _rates(lambda m: (y_pred[m & (y_true == 0)] == 1, np.sum(m & (y_true == 0))))
        fpr_gap = _gap(fpr_rates)
        metrics["false_positive_rate_difference"] = fpr_gap

        # Equalized odds is the MAX of the TPR and FPR gaps. That is what the
        # library's own equalized_odds_difference computes, and it is what this
        # key promises to every consumer that reads it by name. Until 2026-09-09
        # this line was "(simplified: TPR gap)": measured with TPR equal in both
        # groups and FPR 0.0 vs 0.9, the library reported 0.90, this gate
        # reported 0.00 under the same key, and APPROVED the model against a
        # 0.10 threshold. On a deployment-blocking surface, a key that means
        # less than its name is a false pass with a canonical label on it.
        #
        # NaN-aware on purpose: `max(nan, x)` in Python depends on argument
        # order, and an arm that could not be measured would make the max a
        # LOWER BOUND reported as the disparity (the rule constraints/base.py
        # already states). Either arm unmeasurable means the whole metric is,
        # and evaluate() then fails closed on the NaN as it does for the others.
        metrics["equalized_odds_difference"] = (
            float("nan") if (tpr_gap != tpr_gap or fpr_gap != fpr_gap) else max(tpr_gap, fpr_gap)
        )

        # Predictive parity gap: P(Y=1 | A=g, Y_hat=1)
        ppv_rates = _rates(lambda m: (y_true[m & (y_pred == 1)] == 1, np.sum(m & (y_pred == 1))))
        metrics["predictive_parity_difference"] = _gap(ppv_rates)

        # Disparate impact (the four-fifths rule): the min/max selection-rate
        # RATIO, which is the statistic a regulator actually names. Until this
        # was added the gate emitted only the four *_difference gaps above, so a
        # user who configured a four-fifths threshold against evaluate() named a
        # metric that was never in this dict: it came back could-not-check and
        # BLOCKED every deployment, fair or unfair alike, unless a custom
        # compute_metrics_fn supplied it. Safe (fail-closed) but unenforceable.
        #
        # The maths is NOT restated here. It delegates to the canonical library
        # implementation so this surface and the report surface can never drift
        # apart on the same statistic.
        #
        # min_group_size=1 keeps this method's convention that EVERY group
        # present in the data is compared and none is silently dropped for being
        # small (the four gaps above use np.unique with no size filter, and
        # min_group_size only WARNS, it never excludes). That is a statement
        # about COMPUTATION, and it stays true. It is not a statement about the
        # verdict: evaluate() reports every group it found to be below
        # config.min_group_size on decision.small_sample_warnings and withholds
        # APPROVED for a comparison drawn across one. Before that existed, this
        # comment described warning behaviour that only evaluate_hierarchical
        # had, and even there the warnings never touched the verdict.
        # The canonical function returns NaN when fewer than two groups have a
        # defined selection rate, which is the same could-not-check convention
        # _gap() uses above; evaluate() then fails closed on that NaN.
        from vfairness.evaluation.vfairness_metrics.classification import (
            disparate_impact_ratio,
        )

        try:
            metrics["disparate_impact_ratio"] = float(
                disparate_impact_ratio(
                    y_true,
                    y_pred,
                    protected_attr,
                    min_group_size=1,
                    missing_strategy="exclude",
                )
            )
        except Exception:
            # The canonical implementation validates its inputs and refuses data
            # the four raw-numpy gaps above tolerate (non-binary labels, for
            # example). A deployment gate must not crash on that, and it must not
            # omit the key either: record could-not-check (NaN) so evaluate()
            # routes it to its explicit fail-closed branch and BLOCKS, exactly as
            # it does for an unmeasurable gap. Never a pass.
            metrics["disparate_impact_ratio"] = float("nan")

        return metrics

    def evaluate_from_metrics(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """Evaluate pre-computed metrics and make deployment decision.

        Use this when you've already computed fairness metrics elsewhere.

        This entry point is handed numbers, not group labels, so it cannot see
        sample sizes and applies no small-sample check: ``small_sample_warnings``
        on the returned decision is always empty here, and an empty list means
        NOT CHECKED, not "no small groups". :meth:`evaluate`, which is given the
        protected attribute, is the path that checks it.

        That distinction is now READABLE rather than only documented: the
        returned decision carries ``small_sample_check_ran=False``, which
        ``to_dict`` emits and ``to_markdown_report`` renders as a "Small-Sample
        Check: NOT RUN" section. Until 2026-09-27 both serialisers emitted ``[]``
        for this path and for a checked-and-clean :meth:`evaluate` decision with
        no key to tell them apart, so the sentence above reached no consumer.

        Args:
            metrics: Pre-computed fairness metrics.
            baseline_metrics: Optional baseline metrics for comparison.
            model_metadata: Optional metadata about the model.

        Returns:
            GateDecision with approval status and details.
        """
        metric_evaluations = []
        blocking_reasons = []
        warnings = []

        # C-05, second half. The loop below fails closed on a configured metric
        # that was not supplied, but an EMPTY config configures nothing, so the
        # loop never runs and the gate approves. Measured 2026-09-07:
        # GateConfig(metrics=[], thresholds={}) with evaluate_from_metrics({})
        # returned approved=True, "APPROVED - All fairness requirements met", and
        # create_github_check reported conclusion "success".
        #
        # A gate that checks nothing must not approve. This is the difference
        # between "every requirement was met" and "there were no requirements".
        if not self.config.metrics:
            blocking_reasons.append(
                "the gate has no metrics configured, so nothing was checked; it "
                "fails closed rather than approving an unevaluated model"
            )

        # A BLOCKING REQUIREMENT ON A METRIC NOBODY GATES (W4, 2026-09-30). The
        # sibling of the empty-config guard above: that one catches a gate with NO
        # requirements, and this one a requirement that reaches no comparison.
        # `is not None` because None means "every metric blocks" and names nothing,
        # and an EMPTY list is the documented warn-only mode and names nothing
        # either. See _unmatched_blocking_message for the three measured rows.
        declared_blocking = self.config.blocking_metrics
        if declared_blocking is not None:
            unmatched_blocking = [n for n in declared_blocking if n not in self.config.metrics]
            if unmatched_blocking:
                blocking_reasons.append(
                    _unmatched_blocking_message(unmatched_blocking, list(self.config.metrics))
                )

        for metric_name in self.config.metrics:
            if metric_name not in metrics:
                # A configured metric that was not supplied is a could-not-check,
                # not a pass. This mirrors the identical block in evaluate();
                # the third-iteration audit recorded the fix but it landed at
                # that call site only, so evaluate_from_metrics({}) went on
                # returning approved=True (and create_github_check() reported
                # conclusion='success') for a REQUIRED metric that was never
                # measured. Fail closed: a blocking metric refuses approval, a
                # non-blocking one downgrades to CONDITIONAL.
                is_blocking = (
                    self.config.blocking_metrics is None
                    or metric_name in self.config.blocking_metrics
                )
                message = (
                    f"{metric_name} was not provided in the supplied metrics; the "
                    f"gate fails closed rather than approving an unevaluated metric"
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=float("nan"),
                        threshold=self.config.thresholds.get(metric_name),
                        passed=False,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            value = metrics[metric_name]
            threshold = self.config.thresholds.get(metric_name)
            baseline_value = baseline_metrics.get(metric_name) if baseline_metrics else None

            is_blocking = True
            if self.config.blocking_metrics is not None:
                is_blocking = metric_name in self.config.blocking_metrics

            # NaN routing, same semantics as evaluate(): a pre-computed NaN
            # metric is could-not-check, never a silent PASS through the
            # threshold comparison. Fail closed (blocking -> BLOCKED,
            # non-blocking -> CONDITIONAL with a warning).
            # INFINITY, added READINESS-6 (2026-09-10). NaN was guarded here
            # and infinity was not, and they are opposite halves of one hole:
            # NaN loses every comparison, inf WINS them. With every layer
            # NaN-only, disparate_impact_ratio=inf against the four-fifths
            # threshold of 0.8 gave status=APPROVED, approved=True, passed=True,
            # an EMPTY message and a GitHub check conclusion of "success",
            # indistinguishable from the genuine pass at 0.95. inf reaches the
            # ratio family from a zero denominator, i.e. a group with NO
            # SELECTIONS AT ALL.
            #
            # THIS GUARD IS DEFENCE IN DEPTH, NOT THE CARRIER. Measured by
            # sabotage: reinstating the NaN-only test HERE changes nothing on
            # its own, because check_threshold now answers COULD_NOT_CHECK for
            # a non-finite value and this gate fails closed on that outcome. It
            # is kept so the two paths cannot drift apart, and so the operator
            # gets the fail-closed sentence below rather than a bare
            # could-not-measure. Both facts are pinned in
            # tests/test_readiness6_infinity.py. Saying which layer actually
            # stops the defect matters: an earlier version of this comment
            # claimed this line was what changed the outcome, and it was not.
            if not is_measured(value):
                message = (
                    f"{metric_name} could not be measured on this data "
                    f"({unmeasurable_reason(value)}); the gate fails closed "
                    f"rather than approving an unmeasured metric"
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=value,
                        threshold=threshold,
                        passed=False,
                        baseline_value=baseline_value,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            # Unconfigured-threshold routing, same semantics as evaluate(); see
            # the measurement recorded there. Reproduced on THIS entry point
            # 2026-09-10: GateConfig(metrics=['equalized_odds_difference'],
            # thresholds={}) with evaluate_from_metrics({'equalized_odds_
            # difference': 0.9}) returned threshold=None passed=True
            # approved=True status=approved, and create_github_check reported
            # conclusion "success". `is None`, never falsiness, so a configured
            # zero-tolerance threshold of 0.0 keeps being enforced.
            if threshold is None:
                message = (
                    f"{metric_name} has no threshold configured, so it was never "
                    f"compared against anything; the gate fails closed rather than "
                    f"approving an unchecked metric. Add it to `thresholds`, or "
                    f"remove it from `metrics` if it is not being gated on."
                )
                metric_evaluations.append(
                    MetricEvaluation(
                        metric_name=metric_name,
                        value=value,
                        threshold=None,
                        passed=False,
                        baseline_value=baseline_value,
                        is_blocking=is_blocking,
                        message=message,
                    )
                )
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)
                continue

            passed = True
            message = ""

            # Same direction-aware comparison as evaluate(); see the comment
            # there. A ratio metric breaches BELOW its threshold, a
            # difference metric ABOVE it, and an unknown metric fails closed.
            outcome, threshold_message = check_threshold(metric_name, value, threshold)
            if outcome is not ThresholdOutcome.PASS:
                passed = False
                message = threshold_message
            else:
                # VACUOUS-THRESHOLD GUARD, the wider rule check_threshold's own
                # degenerate-bound test is one case of: a bound is unusable when the
                # comparison it controls cannot be False for any data in the METRIC'S
                # RANGE. check_threshold refuses the two cases it can see without a
                # range (a required minimum at or below zero, a negative maximum) and
                # returned PASS for a maximum of 1.2, or 60.0, on a rate gap that
                # cannot exceed 1.0. See _vacuous_bound_message for the four measured
                # multiplier rows, including a gap of 1.0000, the largest a disparity
                # can be, APPROVED against 60.0.
                #
                # IN THE ELSE ARM, and that is not a shortcut: a bound nothing can
                # breach cannot have produced a FAIL, so PASS is the only outcome it
                # can reach here, and when check_threshold has already refused, its
                # reason is the more precise one and must stand (the same rule the
                # baseline and small-sample blocks below follow).
                #
                # THIS ONE GUARD COVERS THREE BOUNDS. The relaxed intersectional
                # threshold and any per_intersection_thresholds override both arrive
                # here as that level's own threshold, because every level of
                # evaluate_hierarchical runs through this method; the base threshold
                # is the third. A guard placed at the relaxation site would have
                # covered one of the three.
                vacuous_threshold = vacuous_bound_reason(
                    metric_name, threshold, BoundRole.THRESHOLD
                )
                if vacuous_threshold:
                    passed = False
                    message = _vacuous_bound_message("threshold", metric_name, vacuous_threshold)

            # BASELINE GUARD, ABOVE BOTH BASELINE CHECKS, the same fix as in
            # evaluate() and for the same reason: both checks share the
            # precondition, and `improvement < margin` / `degradation > margin`
            # are both False for a non-finite baseline. Measured on THIS entry
            # point, evaluate_from_metrics({'demographic_parity_difference':
            # 0.02}, baseline_metrics={'demographic_parity_difference': nan})
            # with require_improvement=True, improvement_margin=0.01, BEFORE:
            # approved=True, status=APPROVED, 'APPROVED - All fairness
            # requirements met', warnings=[], row {'value': 0.02, 'passed':
            # True, 'baseline_value': nan, 'improvement': nan, 'message': ''},
            # github check 'success'. AFTER: approved=False, BLOCKED, the
            # sentence from _unmeasurable_baseline_message in blocking_reasons,
            # github check 'failure'.
            #
            # AND THE ABSENT BASELINE, W2 A-operations-1, 2026-09-29: the SAME
            # CALL SITE, the same one-line precondition, fixed in the same
            # change because fixing one of two identical copies is how this file
            # has been bitten before (the absent-metric fail-closed landed at
            # one entry point only and evaluate_from_metrics({}) went on
            # approving for months). See _absent_baseline_message: measured on
            # THIS entry point, evaluate_from_metrics({dp: 0.2}) with
            # require_improvement=True and margin=0.01 returned approved=True,
            # "APPROVED - All fairness requirements met", improvement=None,
            # blocking=[] warnings=[] and GitHub 'success' for all four
            # no-baseline shapes.
            improvement = None
            baseline_absent = baseline_value is None
            baseline_measurable = not baseline_absent and is_measured(baseline_value)
            baseline_required = self.config.require_improvement or (
                self.config.allow_degradation_margin is not None
            )

            # MARGIN GUARD, ABOVE BOTH COMPARISONS, for the same reason the baseline
            # guard below is above both: they share the precondition, and a guard
            # inside either comparison cannot cover the other. The baseline guard
            # covers the LEFT operand of `improvement < margin`; this covers the
            # RIGHT one, which had no guard anywhere in either entry point. See
            # _unusable_margin_message for the measured before and after.
            #
            # is_measured, not math.isnan: an INFINITE improvement margin makes the
            # requirement impossible to SATISFY rather than impossible to check, and
            # an infinite degradation margin allows every degradation. Both are
            # unusable bounds and both are silent.
            #
            # AND THE FINITE HALF OF THE SAME HOLE (2026-09-30). is_measured closes
            # NaN and the two infinities; every FINITE bound that no data can breach
            # stayed open, and the refusal sentence above states the criterion those
            # bounds meet exactly. improvement_margin=-1e9 and
            # allow_degradation_margin=1e9 each APPROVED a measured degradation of
            # 0.5 with zero warnings. One rule for both halves, one predicate for
            # all four bounds: see _vacuous_bound_message.
            unusable_margins = []
            if self.config.require_improvement:
                if not is_measured(self.config.improvement_margin):
                    unusable_margins.append(
                        _unusable_margin_message(
                            "required improvement",
                            metric_name,
                            self.config.improvement_margin,
                        )
                    )
                else:
                    vacuous_margin = vacuous_bound_reason(
                        metric_name,
                        self.config.improvement_margin,
                        BoundRole.REQUIRED_IMPROVEMENT,
                    )
                    if vacuous_margin:
                        unusable_margins.append(
                            _vacuous_bound_message(
                                "required improvement", metric_name, vacuous_margin
                            )
                        )
            if self.config.allow_degradation_margin is not None:
                if not is_measured(self.config.allow_degradation_margin):
                    unusable_margins.append(
                        _unusable_margin_message(
                            "allowed degradation",
                            metric_name,
                            self.config.allow_degradation_margin,
                        )
                    )
                else:
                    vacuous_margin = vacuous_bound_reason(
                        metric_name,
                        self.config.allow_degradation_margin,
                        BoundRole.ALLOWED_DEGRADATION,
                    )
                    if vacuous_margin:
                        unusable_margins.append(
                            _vacuous_bound_message(
                                "allowed degradation", metric_name, vacuous_margin
                            )
                        )
            for margin_message in unusable_margins:
                # The same fail-closed decision point as the baseline guard, and the
                # same routing: a blocking metric refuses approval, one already
                # failing keeps the more precise reason and carries this as a warning
                # rather than dropping it.
                if passed:
                    passed = False
                    message = margin_message
                elif margin_message not in warnings:
                    warnings.append(margin_message)
            if not baseline_measurable and baseline_required:
                # Same two shapes as evaluate(), routed through the same two
                # shared builders so the two entry points cannot word the same
                # refusal differently.
                baseline_message = (
                    _absent_baseline_message(metric_name, value)
                    if baseline_absent
                    else _unmeasurable_baseline_message(metric_name, value, baseline_value)
                )
                improvement = float("nan")
                if passed:
                    passed = False
                    message = baseline_message
                else:
                    warnings.append(baseline_message)

            if (
                baseline_value is not None
                and baseline_measurable
                and self.config.require_improvement
            ):
                improvement = improvement_amount(metric_name, value, baseline_value)
                if improvement is None:
                    # See _unknown_direction_message: this substituted
                    # abs(baseline_value) - abs(value) and decided `passed` from it.
                    unknown = _unknown_direction_message(
                        "required improvement", metric_name, value, baseline_value
                    )
                    # NaN, not None, and for the same reason the baseline guard
                    # above gives: None is this dataclass's "no baseline comparison
                    # was asked for", and leaving it None here would make an
                    # unevaluable requirement indistinguishable from one nobody
                    # configured. A first version of this fix did exactly that, and
                    # the control test caught it: with require_improvement=False
                    # the field is also None, so the two states had collapsed.
                    improvement = float("nan")
                    if passed:
                        passed = False
                        message = unknown
                    elif unknown not in warnings:
                        warnings.append(unknown)
                elif improvement < self.config.improvement_margin:
                    if passed:
                        passed = False
                        message = f"{metric_name} did not improve by required margin"

            # Check degradation (same semantics as evaluate(); previously
            # only evaluate() enforced allow_degradation_margin, so
            # pre-computed metrics could regress past the margin unnoticed).
            # `is not None`, not `> 0`, for the reason recorded on
            # GateConfig.allow_degradation_margin: the old guard let the
            # default and the strictest-looking setting disable the check.
            degradation_margin = self.config.allow_degradation_margin
            if (
                baseline_value is not None
                and baseline_measurable
                and degradation_margin is not None
            ):
                improved = improvement_amount(metric_name, value, baseline_value)
                if improved is None:
                    # The same substitution with the sign flipped, and the same
                    # closure. See _unknown_direction_message.
                    unknown = _unknown_direction_message(
                        "allowed degradation", metric_name, value, baseline_value
                    )
                    # Same reasoning, and it matters more here: this block does not
                    # otherwise write to `improvement` at all, so without this the
                    # row published None both for "no degradation margin set" and
                    # for "the margin could not be checked".
                    improvement = float("nan")
                    if passed:
                        passed = False
                        message = unknown
                    elif unknown not in warnings:
                        warnings.append(unknown)
                elif -improved > degradation_margin:
                    passed = False
                    message = f"{metric_name} degraded beyond allowed margin"

            evaluation = MetricEvaluation(
                metric_name=metric_name,
                value=value,
                threshold=threshold,
                passed=passed,
                baseline_value=baseline_value,
                improvement=improvement,
                is_blocking=is_blocking,
                message=message,
            )
            metric_evaluations.append(evaluation)

            if not passed:
                if is_blocking:
                    blocking_reasons.append(message)
                else:
                    warnings.append(message)

        approved = len(blocking_reasons) == 0
        status = (
            GateStatus.APPROVED
            if approved and not warnings
            else (GateStatus.CONDITIONAL if approved else GateStatus.BLOCKED)
        )

        return GateDecision(
            approved=approved,
            status=status,
            metric_evaluations=metric_evaluations,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            metadata={
                "model_metadata": model_metadata,
                "provided_metrics": metrics,
                "config": self.config.to_dict(),
            },
            # This entry point is handed numbers with no sample counts, so the
            # small-sample check did NOT run and the empty warning list above
            # must not read as "no group was too small". The docstring said so
            # and, until 2026-09-27, no consumer could see it.
            small_sample_check_ran=False,
        )

    def get_explanation(self, decision):
        """Generate educational explanations for a gate decision.

        Parameters
        ----------
        decision : GateDecision
            The result from :meth:`evaluate`.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        return FairnessExplainer.explain(decision)

    def create_github_check(self, decision: GateDecision) -> Dict[str, Any]:
        """Create a GitHub Check Run compatible payload.

        THREE CONCLUSIONS, because the decision has three states. ``failure``
        when the gate blocked, ``neutral`` when it approved but something could
        not be checked (:attr:`GateStatus.CONDITIONAL`), ``success`` only for a
        gate that measured every configured metric and passed all of them.

        Args:
            decision: Gate decision to convert.

        Returns:
            Dictionary compatible with GitHub Checks API.
        """
        # BGL-3, measured 2026-09-27. `"success" if decision.approved else
        # "failure"` collapsed CONDITIONAL onto success, and CONDITIONAL exists
        # for exactly one reason: a metric was NOT measured and the gate chose
        # not to block on it. Measured with metrics
        # ['demographic_parity_difference', 'auroc_parity'], a threshold for
        # both, blocking_metrics=['demographic_parity_difference'], on 100 + 100
        # genuinely fair rows (auroc_parity is not one of the five evaluate()
        # can compute, so it comes back could-not-check):
        #
        #   status CONDITIONAL, warnings ['auroc_parity could not be computed
        #   ...'], and this payload said conclusion 'success', title 'Model
        #   Fairness Evaluation'  -- byte-identical in both fields to the run
        #   where every metric WAS measured and passed.
        #
        # The check conclusion is the machine-readable field branch protection
        # and every bot key off, so the one surface that decides whether the
        # pull request is mergeable was the only one that had dropped the third
        # state: `decision.status`, `decision.warnings`, the metric row and the
        # markdown all still said could-not-check. 'neutral' is GitHub's own
        # documented conclusion for "ran, and the result is neither pass nor
        # fail", and it does not block a merge, so the non-blocking contract
        # pinned in tests/test_readiness5_gate.py is unchanged: what changes is
        # that a reviewer can no longer read a green tick over an unmeasured
        # metric.
        if not decision.approved:
            conclusion = "failure"
            title = "Model Fairness Evaluation"
        elif decision.status is GateStatus.CONDITIONAL or decision.warnings:
            conclusion = "neutral"
            title = "Model Fairness Evaluation: not every metric could be checked"
        else:
            conclusion = "success"
            title = "Model Fairness Evaluation"

        return {
            "name": "Fairness Gate",
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": title,
                "summary": decision.summary,
                "text": decision.to_markdown_report(),
            },
        }

    def evaluate_hierarchical(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attrs: Dict[str, np.ndarray],
        *,
        hierarchical_config: Optional[HierarchicalGateConfig] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> IntersectionalGateDecision:
        """Evaluate fairness hierarchically: overall -> single-attribute -> intersections.

        Iterates through each hierarchy level and applies the gate check.
        For intersectional groups, thresholds can be relaxed via the
        ``default_intersection_threshold_multiplier`` or overridden per-group
        via ``per_intersection_thresholds``.  Groups smaller than
        ``min_group_size`` emit :class:`SmallSampleWarning` and are never dropped
        from the computation, but the level they appear at does not return
        APPROVED on them: each level runs through :meth:`evaluate`, which routes
        an uncertifiable comparison to could-not-check (blocking metric ->
        BLOCKED, non-blocking -> CONDITIONAL). Until 2026-09-10 the warnings were
        collected into a field nothing read, so this method returned
        "APPROVED - 2/2 levels passed, 1 small-sample warning(s)".

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            protected_attrs: Dict mapping attribute names to arrays.
                e.g. {'gender': gender_arr, 'race': race_arr}
            hierarchical_config: Config controlling which levels to check.
            baseline_metrics: Optional baseline for improvement checks.
            model_metadata: Optional model metadata.

        Returns:
            IntersectionalGateDecision with per-level results and warnings.
        """
        hconfig = hierarchical_config or HierarchicalGateConfig()

        # Every level check runs at the minimum the HIERARCHY declares, not the
        # gate's own. The two could not disagree before evaluate() enforced a
        # minimum at all; now they can, and a hierarchy configured with
        # min_group_size=50 would otherwise list a 40-person group as a
        # small-sample warning while the level decision it came from had already
        # returned APPROVED on it.
        level_gate = self
        if hconfig.min_group_size != self.config.min_group_size:
            level_gate = ModelFairnessGate(
                config=replace(self.config, min_group_size=hconfig.min_group_size),
                compute_metrics_fn=self.compute_metrics_fn,
            )

        level_results: Dict[str, GateDecision] = {}
        small_sample_warnings: List[SmallSampleWarning] = []
        all_blocking: List[str] = []
        all_warnings: List[str] = []

        attr_names = list(protected_attrs.keys())

        # WHAT WAS ACTUALLY LOOKED AT, recorded as the levels run rather than
        # derived from the config afterwards. W2 A-operations-1, 2026-09-29: the
        # C-05 guard below fires only on ZERO levels, so PARTIAL coverage
        # published as a clean pass. These two lists are what the coverage guards
        # after Level 3 read; they are appended to at the point the work happens,
        # because a second reading of the config is a second chance to disagree
        # with it.
        evaluated_attrs: List[str] = []
        small_sample_scanned_attrs: List[str] = []
        # And the intersections whose CELLS were sized, which is a different loop
        # from the one above and never fed the flag it is read through (W3,
        # 2026-09-30). Recorded here, beside its sibling, at the point the sizing
        # happens.
        small_sample_sized_intersections: List[str] = []

        # Level 1: Overall (using first attribute as reference)
        if hconfig.check_overall and attr_names:
            first_attr = protected_attrs[attr_names[0]]
            decision = level_gate.evaluate(
                y_true,
                y_pred,
                first_attr,
                baseline_metrics=baseline_metrics,
                model_metadata=model_metadata,
            )
            level_results["overall"] = decision
            all_blocking.extend(decision.blocking_reasons)
            all_warnings.extend(decision.warnings)
            # This level measures attr_names[0] and ONLY attr_names[0]. Every
            # other supplied attribute is still unexamined after it.
            evaluated_attrs.append(attr_names[0])

        # Level 2: Single-attribute checks
        if hconfig.check_single_attributes:
            for attr_name, attr_values in protected_attrs.items():
                # Check group sizes and emit warnings
                groups, counts = np.unique(attr_values, return_counts=True)
                # The group-size scan happened for THIS attribute. It is the only
                # place the hierarchy's own small_sample_warnings list is filled
                # from single-attribute groups, so it is what makes the decision's
                # small_sample_check_ran True rather than "not run".
                small_sample_scanned_attrs.append(attr_name)
                for g, c in zip(groups, counts):
                    if c < hconfig.min_group_size:
                        small_sample_warnings.append(
                            SmallSampleWarning(
                                group_name=str(g),
                                sample_size=int(c),
                                minimum_recommended=hconfig.min_group_size,
                            )
                        )

                decision = level_gate.evaluate(
                    y_true,
                    y_pred,
                    attr_values,
                    baseline_metrics=baseline_metrics,
                    model_metadata=model_metadata,
                )
                level_results[f"attr:{attr_name}"] = decision
                all_blocking.extend(decision.blocking_reasons)
                all_warnings.extend(decision.warnings)
                evaluated_attrs.append(attr_name)

        # Level 3: Intersectional checks
        if hconfig.check_intersections and len(attr_names) >= 2:
            depth = min(hconfig.intersection_depth, len(attr_names))
            for r in range(2, depth + 1):
                for combo in combinations(attr_names, r):
                    # Build intersectional attribute
                    combo_values = None
                    for attr_name in combo:
                        vals = np.array([str(v) for v in protected_attrs[attr_name]])
                        if combo_values is None:
                            combo_values = vals
                        else:
                            combo_values = np.array(
                                [f"{a}_{b}" for a, b in zip(combo_values, vals)]
                            )

                    level_name = f"intersection:{'_x_'.join(combo)}"

                    # combo has r >= 2 attribute names, so the loop above always
                    # ran at least once and combo_values is a built ndarray.
                    assert combo_values is not None

                    # Check group sizes for intersectional groups
                    groups, counts = np.unique(combo_values, return_counts=True)
                    # THE SCAN THE FLAG NEVER SAW. This loop compares every cell of
                    # this combination to the minimum, and until 2026-09-30 nothing
                    # recorded that it had, so a hierarchy that skipped this level
                    # entirely still reported "every group was at or above the
                    # minimum" on all four surfaces. See
                    # IntersectionalGateDecision.small_sample_unsized.
                    small_sample_sized_intersections.append("_x_".join(combo))
                    for g, c in zip(groups, counts):
                        if c < hconfig.min_group_size:
                            small_sample_warnings.append(
                                SmallSampleWarning(
                                    group_name=str(g),
                                    sample_size=int(c),
                                    minimum_recommended=hconfig.min_group_size,
                                )
                            )

                    # Apply per-intersection or relaxed thresholds. The multiplier
                    # is a RELAXATION factor, so it has to move each threshold in
                    # that metric's permissive direction: a larger allowed gap for
                    # a difference metric (multiply), a smaller accepted ratio for
                    # a ratio metric (divide). Multiplying blindly TIGHTENED the
                    # four-fifths floor from 0.80 to 0.96, making the "relaxed"
                    # intersectional check stricter than the base check on exactly
                    # the small groups it was meant to be gentler on. An unknown
                    # direction is left untouched rather than moved the wrong way.
                    intersection_thresholds = {}
                    for metric_name, base_threshold in self.config.thresholds.items():
                        if base_threshold is None:
                            # There is nothing to relax. A None base threshold
                            # reached relax_threshold and raised outright:
                            # measured 2026-09-10, evaluate_hierarchical with
                            # thresholds={'demographic_parity_difference': None}
                            # raised TypeError("unsupported operand type(s) for
                            # *: 'NoneType' and 'float'"). Leave the key out, so
                            # the intersection gate's own lookup returns None and
                            # evaluate() reports the honest could-not-check the
                            # flat path now reports, instead of a traceback.
                            continue
                        intersection_thresholds[metric_name] = relax_threshold(
                            metric_name,
                            base_threshold,
                            hconfig.default_intersection_threshold_multiplier,
                        )

                    # Override with per-intersection thresholds where specified.
                    # Only apply an override when its key refers to THIS
                    # intersection: a group present in it (e.g. "Female_Black")
                    # or the intersection name itself ("gender_x_race").
                    # Previously every override leaked into every combination,
                    # so a threshold relaxed for one intersection silently
                    # relaxed all of them.
                    group_set = {str(g) for g in groups}
                    combo_key = "_x_".join(combo)
                    for group_name, group_thresholds in hconfig.per_intersection_thresholds.items():
                        if group_name in group_set or group_name in (combo_key, level_name):
                            # Per-intersection thresholds override the multiplied defaults
                            intersection_thresholds.update(group_thresholds)

                    # Create a temporary gate with intersection thresholds.
                    #
                    # dataclasses.replace, NOT a field-by-field rebuild through
                    # the constructor kwargs. W2 A-operations-1, 2026-09-29: the
                    # kwargs list named five of GateConfig's seven fields and
                    # silently dropped allow_degradation_margin, which the
                    # constructor has no parameter for at all, so a degradation
                    # bound configured on the parent gate was NEVER ENFORCED at
                    # any intersection level. Measured on a gap of 0.5 against a
                    # baseline of 0.0 with allow_degradation_margin=0.0, which
                    # is zero tolerance:
                    #   overall                    margin=0.0  attr:race blocked
                    #   attr:gender                margin=0.0
                    #   intersection:gender_x_race margin=None approved=True
                    # the intersectional level, the one this whole method exists
                    # for, being the level the requirement was switched off on.
                    # `replace` carries every field this dataclass has now and
                    # every field it gains later; the two fields that genuinely
                    # differ for an intersection are named here.
                    intersection_gate = ModelFairnessGate(
                        config=replace(
                            self.config,
                            thresholds=intersection_thresholds,
                            min_group_size=hconfig.min_group_size,
                        ),
                        compute_metrics_fn=self.compute_metrics_fn,
                    )

                    decision = intersection_gate.evaluate(
                        y_true,
                        y_pred,
                        combo_values,
                        baseline_metrics=baseline_metrics,
                        model_metadata=model_metadata,
                    )
                    level_results[level_name] = decision
                    all_blocking.extend(decision.blocking_reasons)
                    all_warnings.extend(decision.warnings)
                    evaluated_attrs.extend(combo)

        # COVERAGE OF WHAT WAS ASKED FOR. W2 A-operations-1, 2026-09-29, the
        # sibling of the zero-level guard below: that guard is the ALL-or-nothing
        # case, and PARTIAL coverage published as a clean pass with nothing said
        # about what was skipped. Both cases were measured on HEAD.
        #
        # (a) An attribute handed to the gate and never evaluated at any level.
        #     evaluate_hierarchical(y, y, {'gender': 100+100 clean, 'race': 195+5},
        #     HierarchicalGateConfig(check_single_attributes=False,
        #     check_intersections=False)) returned approved=True, "APPROVED - 1/1
        #     levels passed", blocking_reasons [], warnings [],
        #     small_sample_warnings [], and the whole markdown report was five
        #     content lines in which the strings 'race' and its five-row group
        #     appeared nowhere ('race' in card False, in report False). Level 1
        #     measures attr_names[0] only, so 'race' was supplied, never looked
        #     at, and approved. Fail closed, exactly as the zero-level case does:
        #     a protected attribute nobody evaluated is not a passing one.
        unevaluated_attrs = [a for a in attr_names if a not in evaluated_attrs]
        if unevaluated_attrs:
            named = ", ".join(repr(a) for a in unevaluated_attrs)
            all_blocking.append(
                f"{len(unevaluated_attrs)} of {len(attr_names)} supplied protected "
                f"attribute(s) were never evaluated at any level ({named}): the "
                f"configured levels do not reach them, so no fairness comparison was "
                f"drawn on them at all. The gate fails closed rather than approving a "
                f"deployment on attributes it did not look at. Enable "
                f"check_single_attributes, or stop supplying the attributes that are "
                f"not being gated on."
            )

        # (b) Intersectional checking asked for and ZERO intersections evaluated.
        #     `for r in range(2, depth + 1)` is EMPTY for any depth below 2, so
        #     the whole level silently does nothing. Measured on two clean
        #     attributes with HierarchicalGateConfig(check_intersections=True,
        #     intersection_depth=1): approved=True, "APPROVED - 3/3 levels
        #     passed", intersections evaluated [], blocking [] warnings [], and
        #     identically for depth 0 and depth -1. This is the unusable-bound
        #     shape _unusable_margin_message already fails closed on: a
        #     configuration that makes the check vacuous whatever the data.
        #
        #     The len(attr_names) < 2 arm is NOT the same thing and is NOT
        #     blocked. With one attribute an intersection is not DEFINABLE, so
        #     nothing was skipped that could have been done, and blocking the
        #     default config's own documented single-attribute call would break
        #     the product rather than the defect. It is disclosed as a warning,
        #     so the reader who configured intersectional checking still learns
        #     none ran: CONDITIONAL, not BLOCKED and not silence.
        if hconfig.check_intersections and not any(
            name.startswith("intersection:") for name in level_results
        ):
            if hconfig.intersection_depth < 2:
                all_blocking.append(
                    f"check_intersections is on but intersection_depth is "
                    f"{hconfig.intersection_depth}, which combines fewer than two "
                    f"attributes, so ZERO intersectional groups were evaluated and the "
                    f"intersectional level was never checked. The gate fails closed "
                    f"rather than reporting an unrun level as passed. Set "
                    f"intersection_depth to 2 or more, or switch check_intersections "
                    f"off explicitly."
                )
            elif len(attr_names) < 2:
                all_warnings.append(
                    f"check_intersections is on but only {len(attr_names)} protected "
                    f"attribute(s) were supplied, so no intersectional group is "
                    f"definable and zero were evaluated. Nothing on this decision is "
                    f"an intersectional result; supply a second attribute to get one."
                )

        # Aggregate final decision.
        #
        # C-05. `approved = len(all_blocking) == 0` is True when NOTHING was
        # evaluated, because zero levels produce zero blocking reasons. Measured
        # 2026-09-07: `evaluate_hierarchical(y_true, y_pred, {})` returned
        # approved=True, status=APPROVED, summary "APPROVED - 0/0 levels passed",
        # and create_github_check reported conclusion "success".
        #
        # An absence of failures is not a pass. `evaluate_from_metrics` was fixed
        # for exactly this shape (a configured metric that could not be computed
        # fails closed rather than approving an unevaluated metric); the
        # hierarchical path was not, and is the one a CI system calls.
        if not level_results:
            all_blocking.append(
                "no level was evaluated, so nothing was checked; the gate fails "
                "closed rather than approving an unevaluated deployment"
            )

        approved = len(all_blocking) == 0
        if approved:
            status = GateStatus.APPROVED if not all_warnings else GateStatus.CONDITIONAL
        else:
            status = GateStatus.BLOCKED

        # Read from the list the Level 2 loop appends to as it scans, not from
        # hconfig re-read here, and compared as SETS rather than by count: the
        # scan has to have covered EVERY supplied attribute for an empty warning
        # list to mean "no group was too small". False when
        # check_single_attributes is off, when only some attributes were reached,
        # and when no attribute was supplied at all, because none of those three
        # compared any group size to the minimum. See
        # IntersectionalGateDecision.small_sample_check_ran for the before-state
        # on all four surfaces.
        # The hierarchy's own bound, tested exactly as the flat path tests its own
        # (W3, 2026-09-30): a scan that covered every attribute against a minimum
        # nothing can fall below has still checked nothing.
        vacuous_minimum = _vacuous_min_group_size(hconfig.min_group_size)
        small_sample_check_ran = (
            bool(attr_names)
            and set(small_sample_scanned_attrs) == set(attr_names)
            and vacuous_minimum is None
        )

        # THE SCOPE OF THAT ANSWER (W3, 2026-09-30). The flag above answers for the
        # ATTRIBUTES and the card rendered it as a claim about every group in the
        # data. The two are different whenever a group set exists that no loop
        # sized, and the commonest such set is the one intersectional fairness is
        # about: cells of five people inside attributes of a hundred. Read from the
        # two lists the loops append to as they scan, never from hconfig re-read
        # here, for the reason the coverage guard above states: a second reading of
        # the config is a second chance to disagree with it.
        small_sample_unsized: List[str] = []
        for attr_name in attr_names:
            if attr_name not in small_sample_scanned_attrs:
                small_sample_unsized.append(
                    f"the groups of attribute {attr_name!r} (check_single_attributes is off)"
                )
        # Pairwise AT LEAST, because that is what an intersection is: a depth below
        # 2 defines none, and the cells still exist in the data whether or not the
        # configuration asked for them. Deeper combinations than the configured
        # depth are NOT listed: those were never part of this decision's scope, and
        # naming them would turn every default three-attribute hierarchy into a
        # caveat, which is the over-correction.
        if not hconfig.check_intersections:
            unsized_cause = "check_intersections is off"
        elif hconfig.intersection_depth < 2:
            unsized_cause = f"intersection_depth is {hconfig.intersection_depth}"
        else:
            unsized_cause = "the intersectional level did not reach them"
        definable_depth = min(max(2, hconfig.intersection_depth), len(attr_names))
        for r in range(2, definable_depth + 1):
            for combo in combinations(attr_names, r):
                combo_key = "_x_".join(combo)
                if combo_key not in small_sample_sized_intersections:
                    small_sample_unsized.append(
                        f"the intersection cells of {combo_key} ({unsized_cause})"
                    )
        # AND THE RATE DENOMINATORS, when the numbers are not this gate's own (W4,
        # 2026-09-30). Every level runs through evaluate(), which sizes the
        # denominator of each metric it computes itself and discloses the scope when
        # a compute_metrics_fn produced the numbers instead. The same sentence here,
        # so a hierarchical decision and a flat one cannot answer this question
        # differently.
        if self.compute_metrics_fn is not None:
            small_sample_unsized.append(_unsized_rate_scope_sentence())

        return IntersectionalGateDecision(
            approved=approved,
            status=status,
            level_results=level_results,
            small_sample_warnings=small_sample_warnings,
            hierarchical_config=hconfig,
            # BGL5 A-operations-1, 2026-09-27. These two lists were built above,
            # decided `approved` on the line before this call, and were then
            # DROPPED: the dataclass had no field for either, so the reason this
            # method appends for itself ("no level was evaluated, so nothing was
            # checked ...") reached no consumer. Measured before this change,
            # evaluate_hierarchical(y, y, {}): to_dict carried no such key,
            # to_markdown_report printed three lines ending at "**Levels
            # evaluated**: 0", the report card said "Deployment blocked", and
            # 'nothing was checked' was absent from all of them. After: the
            # sentence is on the dict, on the markdown report and on the card.
            blocking_reasons=all_blocking,
            warnings=all_warnings,
            # W2 A-operations-1, 2026-09-29. Computed just above; see the comment
            # there and the field docstring for the four surfaces this unblocks.
            small_sample_check_ran=small_sample_check_ran,
            # W3, 2026-09-30: what that answer does NOT cover, so no surface has to
            # infer the scope of a True from the configuration.
            small_sample_unsized=small_sample_unsized,
            small_sample_vacuous_minimum=vacuous_minimum,
        )


# Fairness Report Card (for automated PR comments)


class FairnessReportCard:
    """Generates structured PR-ready fairness report cards.

    Takes a ``GateDecision`` or ``IntersectionalGateDecision`` and formats
    it into a concise markdown report suitable for automated PR comments
    in GitHub/GitLab workflows.

    Example:
        >>> decision = gate.evaluate(y_true, y_pred, protected_attr)
        >>> card = FairnessReportCard(decision, model_name="loan-model-v3")
        >>> print(card.to_markdown())
        >>> # Or get a payload for the GitHub Comments API:
        >>> payload = card.to_github_comment_payload()

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: fairness_report_card. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        decision: Union[GateDecision, IntersectionalGateDecision],
        model_name: Optional[str] = None,
        include_intersectional: bool = True,
    ):
        self.decision = decision
        self.model_name = model_name or "model"
        self.include_intersectional = include_intersectional

    def to_markdown(self) -> str:
        """Generate a markdown report card for PR comments."""
        is_hierarchical = isinstance(self.decision, IntersectionalGateDecision)
        status = self.decision.status
        approved = self.decision.approved

        # Header with status badge
        lines = [
            f"## Fairness Report Card: {self.model_name}",
            "",
            f"**Status**: `{status.value.upper()}`",
            f"**Result**: {'Approved for deployment' if approved else 'Deployment blocked'}",
            "",
        ]

        if is_hierarchical:
            lines.extend(self._format_hierarchical())
        else:
            lines.extend(self._format_simple())

        # THE THREE-STATE FIELD HAD TO REACH THIS SURFACE (BGL6 F03, 2026-09-28).
        # GateDecision.to_dict emits small_sample_check_ran and
        # to_markdown_report renders a "Small-Sample Check: NOT RUN" section, both
        # correct. This card, which this module's own comment calls the surface
        # "where a reviewer decides to merge", printed NEITHER, so the card for a
        # decision whose small-sample check never ran was BYTE-IDENTICAL to the card
        # for a decision on data whose groups were all large enough.
        #
        # Measured 2026-09-28 on the same number 0.0 through both entry points:
        # to_markdown() identical True, and none of 'NOT RUN', 'Small-Sample Check',
        # 'did not happen' or 'not checked' present in the never-checked card. A
        # correct measurement no reader can see, on the page the merge decision is
        # made from.
        lines.extend(self._format_small_sample_state())

        lines.extend(
            [
                "",
                "---",
                "*Generated by vfairness*",
            ]
        )

        return "\n".join(lines)

    def _format_small_sample_state(self) -> List[str]:
        """The small-sample question, answered in three states on the card.

        Same three states and the same wording as
        :meth:`GateDecision.to_markdown_report`, so the card and the report cannot
        say different things about the same decision.

        BOTH DECISION TYPES, W2 A-operations-1, 2026-09-29. This returned ``[]``
        for every hierarchical decision, on a premise written into this docstring
        that was false: "An IntersectionalGateDecision carries no such field, and
        saying nothing is correct for it rather than claiming a check that does not
        apply." The check DOES apply. ``evaluate_hierarchical`` runs it, inside
        ``if hconfig.check_single_attributes:``, so the opt-out path laundered the
        caveat instead of stating it, and the hasattr early return switched the
        disclosure off for the whole type rather than for a decision that had not
        recorded it. Measured on HEAD over two clean groups of 100, with the scan
        off and with it on, 'Small-Sample Check' / 'NOT RUN' / 'NOT RECORDED' were
        absent from ``to_markdown``, ``to_dict()['markdown']`` and
        ``to_github_comment_payload()['body']`` in both cases, while the FLAT twin
        printed "### Small-Sample Check / NOT RUN" for the same question.

        :class:`IntersectionalGateDecision` now carries the field, so the branches
        below answer for it too. The early return is gone rather than widened: an
        object that genuinely has no such field has not established that the check
        was skipped, which is what the NOT RECORDED branch says, and silence is the
        one answer that is never right.
        """
        ran = getattr(self.decision, "small_sample_check_ran", None)
        small = list(getattr(self.decision, "small_sample_warnings", []) or [])
        # What the answer does NOT cover, in the decision's own words. Empty for a
        # flat GateDecision, which is sized over the one attribute it was handed.
        unsized = list(getattr(self.decision, "small_sample_unsized", []) or [])
        lines = ["", "### Small-Sample Check", ""]
        if small:
            lines.append(
                f"- RAN, and {len(small)} group(s) are below the minimum: "
                + ", ".join(
                    f"**{w.group_name}** ({w.sample_size} samples, min {w.minimum_recommended})"
                    for w in small
                )
            )
        elif ran is True and unsized:
            # SCOPED, because "every group" was false of the groups nothing sized.
            # W3, 2026-09-30: with single-attribute groups of 100 and intersection
            # cells of 95/5/5/95, check_intersections=False printed the unqualified
            # sentence below while the same data with the level on named two cells
            # of 5 against a minimum of 30. See _unsized_scope_lines.
            lines.append(
                "- RAN over every group set it covered, and each was at or above the "
                "minimum. This is a measurement, not an absence of one."
            )
        elif ran is True:
            lines.append(
                "- RAN, and every group was at or above the minimum. This is a "
                "measurement, not an absence of one."
            )
        elif getattr(self.decision, "small_sample_vacuous_minimum", None) is not None:
            # THE BOUND-SHAPED CAUSE, ahead of both type-shaped ones (W3,
            # 2026-09-30). A minimum of 1 or less was compared to every group and
            # could not fire, so neither "no group sizes attached" nor
            # "check_single_attributes is off" is true of this decision: the sizes
            # were there and the scan ran. Same sentence as both reports.
            lines.append(_vacuous_minimum_sentence(self.decision.small_sample_vacuous_minimum))
        elif ran is False:
            # The CAUSE differs by decision type and the sentence has to be true of
            # the decision in hand: a flat one was handed numbers with no sample
            # counts, a hierarchical one was handed the arrays and did not scan
            # them because check_single_attributes was off. Writing the flat cause
            # on a hierarchical card would be a claim stronger than the code
            # establishes, which is the failure this whole section exists to stop.
            # The invariant half, what an empty list does NOT mean, is shared.
            cause = (
                "no supplied attribute's group sizes were compared to the minimum "
                "(check_single_attributes is off)"
                if isinstance(self.decision, IntersectionalGateDecision)
                else "this decision was made from numbers with no group sizes attached, "
                "so no group size was compared to the minimum"
            )
            lines.append(
                f"- NOT RUN: {cause}. An empty warning list means the check did not "
                f"happen, NOT that every group was large enough."
            )
        else:
            lines.append(
                "- NOT RECORDED: nothing on this decision states whether group sizes "
                "were checked, so an empty warning list cannot be read either way."
            )
        # The same builder the markdown report uses, so the card and the report
        # cannot name different scopes for one decision.
        lines.extend(line for line in _unsized_scope_lines(unsized) if line)
        return lines

    def _format_simple(self) -> List[str]:
        """Format a simple (non-hierarchical) gate decision."""
        # Only reached from to_markdown() when the decision is NOT an
        # IntersectionalGateDecision, i.e. a plain GateDecision.
        decision = self.decision
        assert isinstance(decision, GateDecision)
        lines = [
            "### Metric Results",
            "",
            "| Metric | Value | Threshold | Result |",
            "|--------|-------|-----------|--------|",
        ]

        for ev in decision.metric_evaluations:
            # Same shared resolver as both markdown reports, for the same reason:
            # this is the row a reviewer reads on the pull request.
            result = {
                _ROW_PASS: "Pass",
                _ROW_WARN: "Warn",
                _ROW_FAIL: "**FAIL**",
                _ROW_UNMEASURED: "**COULD NOT CHECK**",
                _ROW_REFUSED: "**REFUSED** (not by this bound)",
            }[_metric_row_state(ev)]
            thr = f"{ev.threshold:.4f}" if ev.threshold is not None else "N/A"
            lines.append(
                f"| {ev.metric_name} | {_metric_value_cell(ev.value)} | {thr} | {result} |"
            )

        if decision.blocking_reasons:
            lines.extend(["", "### Blocking Issues", ""])
            for reason in decision.blocking_reasons:
                lines.append(f"- {reason}")

        if decision.warnings:
            lines.extend(["", "### Warnings", ""])
            for w in decision.warnings:
                lines.append(f"- {w}")

        # Same reason as the markdown report: the PR comment is where a reviewer
        # decides to merge, and "which group had five people in it" is the fact
        # that decides it.
        if decision.small_sample_warnings:
            lines.extend(["", "### Small-Sample Warnings", ""])
            for small in decision.small_sample_warnings:
                lines.append(
                    f"- **{small.group_name}**: {small.sample_size} samples "
                    f"(min {small.minimum_recommended})"
                )

        return lines

    def _format_hierarchical(self) -> List[str]:
        """Format a hierarchical intersectional gate decision."""
        # Only reached from to_markdown() when the decision IS an
        # IntersectionalGateDecision.
        decision = self.decision
        assert isinstance(decision, IntersectionalGateDecision)
        lines: List[str] = []

        # Summary table
        lines.extend(
            [
                "### Hierarchy Summary",
                "",
                "| Level | Status | Failures |",
                "|-------|--------|----------|",
            ]
        )

        for level_name, level_decision in decision.level_results.items():
            status = level_decision.status.value.upper()
            states = [_metric_row_state(ev) for ev in level_decision.metric_evaluations]
            # A column headed "Failures" counted every not-passed row, so a level
            # where nothing could be measured reported the count of a comparison
            # it never made. The could-not-checks are named beside the number
            # rather than folded into it or dropped from it.
            n_fail = sum(1 for s in states if s in (_ROW_FAIL, _ROW_WARN, _ROW_REFUSED))
            n_unmeasured = sum(1 for s in states if s == _ROW_UNMEASURED)
            cell = f"{n_fail}"
            if n_unmeasured:
                cell += f" (+{n_unmeasured} could not be checked)"
            lines.append(f"| {level_name} | {status} | {cell} |")

        # Detail per level
        if self.include_intersectional:
            for level_name, level_decision in decision.level_results.items():
                rows = [(ev, _metric_row_state(ev)) for ev in level_decision.metric_evaluations]
                failed = [ev for ev, st in rows if st in (_ROW_FAIL, _ROW_WARN, _ROW_REFUSED)]
                unmeasured = [ev for ev, st in rows if st == _ROW_UNMEASURED]
                if failed:
                    lines.extend(["", f"#### {level_name}: Failed Metrics", ""])
                    for ev in failed:
                        thr = f"{ev.threshold:.4f}" if ev.threshold is not None else "N/A"
                        lines.append(
                            f"- **{ev.metric_name}**: {_metric_value_cell(ev.value)} "
                            f"(threshold: {thr})"
                        )
                # Its own heading, not a row under "Failed Metrics". Measured
                # before this split, on one group so no disparity is definable:
                # '#### overall: Failed Metrics / - **demographic_parity_
                # difference**: nan (threshold: 0.5000)'.
                if unmeasured:
                    lines.extend(["", f"#### {level_name}: Could Not Be Checked", ""])
                    for ev in unmeasured:
                        thr = f"{ev.threshold:.4f}" if ev.threshold is not None else "N/A"
                        lines.append(
                            f"- **{ev.metric_name}**: {_metric_value_cell(ev.value)} "
                            f"(threshold: {thr}), so it was never compared against it"
                        )

        # MIRRORS _format_simple, which prints decision.blocking_reasons and
        # decision.warnings. This half printed neither, and it is the half a PR
        # comment gets for a hierarchical gate.
        #
        # BGL5 A-operations-1, 2026-09-27. Coverage of the test file named as
        # this unit's evidence over this method: 0 of 22 body lines. Measured on
        # evaluate_hierarchical(y, y, {'gender': ['a'] * 200}), where one group
        # makes every disparity undefinable, BEFORE:
        #
        #   #### overall: Failed Metrics
        #   - **demographic_parity_difference**: nan (threshold: 0.5000)
        #
        # and 'could not be computed', 'fails closed', 'not measur' and
        # 'could-not-check' were ALL absent from the whole card, which
        # to_dict()['markdown'] and to_github_comment_payload()['body'] then
        # published verbatim. AFTER: the same card carries '### Blocking Issues'
        # with 'demographic_parity_difference could not be computed on this
        # data; the gate fails closed rather than approving an unevaluated
        # metric', which is the property the named test already asserts for the
        # flat half.
        if decision.blocking_reasons:
            lines.extend(["", "### Blocking Issues", ""])
            for reason in decision.blocking_reasons:
                lines.append(f"- {reason}")

        if decision.warnings:
            lines.extend(["", "### Warnings", ""])
            for warning in decision.warnings:
                lines.append(f"- {warning}")

        # Small-sample warnings
        if decision.small_sample_warnings:
            lines.extend(["", "### Small-Sample Warnings", ""])
            for w in decision.small_sample_warnings:
                lines.append(
                    f"- **{w.group_name}**: {w.sample_size} samples (min {w.minimum_recommended})"
                )

        return lines

    def to_github_comment_payload(self) -> Dict[str, Any]:
        """Create a payload suitable for the GitHub Comments API."""
        return {
            "body": self.to_markdown(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert report card to dictionary."""
        return {
            "model_name": self.model_name,
            "approved": self.decision.approved,
            "status": self.decision.status.value,
            "markdown": self.to_markdown(),
        }
