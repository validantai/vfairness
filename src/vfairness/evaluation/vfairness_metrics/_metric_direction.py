"""Which direction is "better" for a fairness metric, and what to do when that is unknown.

Every surface that compares a metric to a threshold needs the same three facts:

1. Is this metric a violation magnitude (lower is better, e.g. a demographic
   parity DIFFERENCE) or a parity ratio (higher is better, e.g. the four-fifths
   rule on ``disparate_impact_ratio``)?
2. Given that direction, does the value breach the threshold?
3. What if we cannot tell?

Historically each call site answered (1) on its own, and several answered it by
assuming every metric is lower-is-better. That inverts the verdict for the whole
ratio family: a ``disparate_impact_ratio`` of 0.00 (the protected group is NEVER
selected) satisfied ``abs(value) > 0.80 -> fail`` as FALSE, so the deployment gate
APPROVED it and wrote "Pass" into the audit-trail report, while perfect parity
(1.00) was BLOCKED. This module is the single answer, so the direction is decided
once and every surface inherits the same answer.

Two properties are load-bearing and must not be weakened.

**Exact token matching, never a substring.** "ratio" is a substring of
"cali[bratio]n_difference". A naive ``"ratio" in name`` test therefore classified
the lower-is-better calibration family as higher-is-better ratios, which let a
large miscalibration read as a PASS in production for weeks (see CLAUDE.md, the
cali-BRATIO-n incident, reconciled in 47f1e09f8). A metric is a ratio only when
its name ends with the ``_ratio`` SUFFIX, or matches one of the names in
:data:`RATIO_METRICS` exactly after normalisation.

That rule was written here and then broken here, in the very fix that recorded
it. Until 2026-08-27 :func:`is_ratio_metric` ended with a substring test for the
disparate-impact token, evaluated before any lower-is-better rule, so
``disparate_impact_difference`` and ``disparate_impact_gap`` (violation
magnitudes) were graded as four-fifths ratios: a 0.45 gap against a 0.10 bound
was reported as "at or above the required minimum", i.e. a PASS. The bug class
is not the word "ratio", it is matching a family token ANYWHERE inside a name.
So there is no substring test in this module for any token, and precedence is
explicit: an exact name first, then a violation-magnitude TOKEN, then the ratio
shape. A name carrying both (a ``_ratio`` suffix and a violation token) is
contradictory and resolves to UNKNOWN rather than to a guess in either
direction. ``tests/test_metric_direction.py`` pins the whole table and fails if
a substring direction test is reintroduced into this file.

**Fail closed on an unknown metric.** If the direction cannot be determined, the
answer is :attr:`ThresholdOutcome.COULD_NOT_CHECK`, never PASS. Three states,
never two: verified / failed-verification / could-not-check. A caller that
collapses COULD_NOT_CHECK into a pass reintroduces exactly the class of defect
this module exists to remove.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from ..._triage import is_measured, unmeasurable_reason


class MetricDirection(str, Enum):
    """Which way a metric has to move to be better."""

    #: A violation magnitude: it passes at or below its threshold (gaps, errors).
    LOWER_IS_BETTER = "lower_is_better"
    #: A parity ratio: it passes at or above its threshold (the four-fifths rule).
    HIGHER_IS_BETTER = "higher_is_better"
    #: Not determinable from the name. Callers MUST fail closed on this.
    UNKNOWN = "unknown"


class ThresholdOutcome(str, Enum):
    """Result of comparing one metric value against one threshold."""

    PASS = "pass"
    FAIL = "fail"
    #: Neither a pass nor a numeric fail: the comparison could not be performed
    #: (unmeasurable value, or a metric whose better-direction is unknown).
    COULD_NOT_CHECK = "could_not_check"


# Higher-is-better metrics that do NOT end in "_ratio" and so cannot be caught by
# the suffix rule. Audited against src/vfairness/_registry.py (194 dispatch keys)
# on 2026-08-27: worst_group_accuracy is the only scalar metric in the registry
# that is a performance floor rather than a violation magnitude or a ratio
# (Sagawa et al. 2020, minimum per-group accuracy; higher is better).
# Deliberately NOT listed, because they are genuinely ambiguous and must fail
# closed rather than be guessed: calibration_slope (target is 1.0, so neither
# direction is "better"), calibration_in_the_large (same shape, target 0.0),
# proxy_feature_score, representation_skew, conditional_adverse_impact.
#
# calibration_in_the_large was added to this list on 2026-09-10 after a fix wave
# proposed registering it as LOWER_IS_BETTER, on the reasonable-looking grounds
# that CalibrationMetricResult stores |CITL| in overall_value. That is true of
# the RESULT and false of the METRIC: the canonical CITL is SIGNED, and the
# signed overall and per-group values sit in that same result's metadata, where
# a large negative is exactly as bad as a large positive. Registering the name
# here would have handed every consumer of the signed value a direction that is
# wrong for half its range. A metric whose target is an interior point has no
# better direction, and a caller that needs to grade one should use a BAND, as
# post_processing/calibration/metrics.py does.
HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "worst_group_accuracy",
    }
)

# Lower-is-better metrics whose name carries no difference/gap token, plus the
# bare family names this codebase uses for gap quantities. Each one is a
# violation magnitude that passes at or below its threshold.
LOWER_IS_BETTER_METRICS = frozenset(
    {
        # Calibration family: violation magnitudes (Hebert-Johnson et al. 2018,
        # Pleiss et al. 2017). These are the exact names the cali-BRATIO-n
        # substring bug misclassified as ratios.
        "calibration",
        "multicalibration",
        "integrated_calibration_index",
        # Max between-group gaps despite the "_parity" name (see the docstrings
        # of auroc_parity / net_benefit_parity in
        # evaluation/vfairness_metrics/classification.py: both return
        # max - min across groups).
        "auroc_parity",
        "net_benefit_parity",
        # Proper scoring rule: lower is better.
        "brier_score",
        # Bare family names used across this codebase for the corresponding gap
        # (the Pulse cards, report dicts and SVG adapters all label the
        # max - min difference this way).
        "demographic_parity",
        "statistical_parity",
        "equalized_odds",
        "equal_opportunity",
        "predictive_parity",
        # Ranking family (evaluation/vfairness_metrics/ranking.py), registered
        # 2026-08-27 when rendering/adapters_ranking stopped grading by a local
        # name-free ``val <= threshold`` rule. Each of these is a magnitude that
        # is best at 0: attention_weighted_rank_fairness returns a max between
        # -group difference, and NDKL is a Kullback-Leibler divergence (Geyik et
        # al. 2019), so both pass at or below their bound. Their SIBLING
        # ``exposure_parity_ratio`` is deliberately absent: it is a min/max
        # four-fifths floor caught by the ``_ratio`` suffix rule, where HIGHER
        # is better, and grading it as one of these is the inversion that put a
        # green PASS on a 0.62 exposure ratio against a 0.80 bound.
        "attention_weighted_rank_fairness",
        "ndkl",
        "representation_ndkl",
        "normalized_discounted_kl_divergence",
        # Bare ranking family names, as the SVG adapters and report dicts label
        # the corresponding gap. Same convention as the classification bare
        # names above; note that "exposure_parity_ratio" is NOT reachable from
        # "exposure_parity", because matching is exact, never by prefix.
        "exposure_parity",
        "ndcg_parity",
    }
)

# Name TOKENS (underscore-delimited words, never substrings) that identify a
# violation magnitude. Token matching, not substring matching, is what keeps
# "cali[bratio]n" out of the ratio family and would equally keep a hypothetical
# "...gapped..." out of this one.
LOWER_IS_BETTER_TOKENS = frozenset(
    {
        "difference",
        "diff",
        "gap",
        "disparity",
        "error",
        "loss",
        "deviation",
    }
)

# Ratio metrics (HIGHER is better) whose name does NOT end in the "_ratio"
# suffix, so the suffix rule cannot reach them. Matched EXACTLY against the
# normalised name, never as a substring: "disparate_impact" is a prefix of
# "disparate_impact_difference", which is a violation magnitude, and of the
# per-column tracker keys ("disparate_impact_gender"), which are not this
# metric. Human label forms ("Disparate Impact", "disparate-impact") normalise
# onto these same names, so they need no branch of their own.
RATIO_METRICS = frozenset(
    {
        "disparate_impact",
    }
)

# The closed interval a metric's values can fall in, for the metrics whose own
# STATISTIC fixes it. Read only by :func:`metric_value_range`, which feeds
# :func:`vacuous_bound_reason`: a bound can only be shown to be unbreachable
# against a range, so a metric that is NOT listed here has no declared range and
# no vacuity claim is ever made about its bounds.
#
# Deliberately small, exact-matched, and safe in one direction only. Registering a
# range that is too NARROW would refuse a bound the metric can genuinely exceed,
# which blocks a legitimate deployment: that is the over-correction, and it is
# worse than staying silent. Every name below is a rate, a probability, or a
# min/max rate RATIO, so none of them can leave [0, 1]:
#   - a max - min difference of per-group rates that each live in [0, 1] (the
#     whole *_difference / *_parity_difference family, and auroc_parity, which
#     is a max difference of per-group AUROCs);
#   - a min/max selection-rate ratio, the four-fifths family: 1.0 at parity, 0.0
#     when some group is never selected;
#   - worst_group_accuracy, the minimum per-group accuracy.
#
# THE SIBLING TABLE IS ``operations/cicd/precommit.RATE_DIFFERENCE_METRICS`` /
# ``MIN_MAX_RATIO_METRICS``, which applies this same rule to a fairness CONFIG FILE
# at commit time. It is a deliberate duplicate, for the reason recorded on
# ``precommit._normalize_metric_name``: the pre-commit hook must stay importable
# with nothing but the standard library, and this module imports ``_triage``. The
# two tables must hold the same names, so a bound the hook refuses cannot be
# approved by the gate at runtime and vice versa; the names below marked "also in
# precommit" came from there.
#
# NOT listed, on purpose, because their range is genuinely not [0, 1]:
# net_benefit_parity (net benefit is not confined to [0, 1], so neither is a
# max - min of it), the KL-divergence ranking metrics ndkl / representation_ndkl
# / normalized_discounted_kl_divergence (a divergence has no upper bound), the
# calibration family (a CITL-shaped statistic is SIGNED, as the
# HIGHER_IS_BETTER_METRICS comment above records, and a multicalibration
# implementation may report a sum rather than a max), brier_score (in [0, 1] for
# a binary probability and up to 2 for the multiclass sum), and every
# error / loss / deviation name, which can be an unbounded regression quantity.
UNIT_INTERVAL_METRICS = frozenset(
    {
        # Selection-rate and error-rate gaps (classification.py), plus the bare
        # family names this codebase uses for the same gap.
        "demographic_parity_difference",
        "demographic_parity",
        "statistical_parity_difference",
        "statistical_parity",
        "equal_opportunity_difference",
        "equal_opportunity",
        "equalized_odds_difference",
        "equalized_odds",
        "predictive_parity_difference",
        "predictive_parity",
        "false_positive_rate_difference",
        "false_negative_rate_difference",
        "true_positive_rate_difference",
        "fpr_parity_difference",
        "fnr_parity_difference",
        "accuracy_parity_difference",
        "negative_predictive_value_difference",
        # Also in precommit's table: the same rate differences, named there
        # because a config file can configure any of them.
        "predictive_equality_difference",
        "predictive_equality",
        "true_negative_rate_difference",
        "selection_rate_difference",
        "accuracy_difference",
        "error_rate_difference",
        # A difference of two ratios that each live in [0, 1]. The name carries a
        # violation token, so metric_direction reads it lower-is-better, which is
        # what it is: the SIZE of a disparate-impact gap, not the ratio itself.
        "disparate_impact_difference",
        # A max difference of per-group AUROCs, each in [0, 1].
        "auroc_parity",
        # A minimum per-group accuracy (Sagawa et al. 2020).
        "worst_group_accuracy",
        # The four-fifths family: min/max selection-rate ratios.
        "disparate_impact_ratio",
        "disparate_impact",
        "demographic_parity_ratio",
        "exposure_parity_ratio",
        # Also in precommit's MIN_MAX_RATIO_METRICS.
        "adverse_impact_ratio",
    }
)

# Confidence-interval variants share the direction of the base metric.
_CI_SUFFIX = "_with_ci"
_RATIO_SUFFIX = "_ratio"


def _normalize(metric_name: str) -> str:
    """Canonicalise a metric name or human label for EXACT table lookups.

    Lower-cases, folds spaces and hyphens onto the snake_case separator (so the
    human label "Disparate Impact" becomes the metric name "disparate_impact"
    and can be matched exactly rather than by a substring test), collapses
    repeated separators, and drops the '_with_ci' variant suffix.
    """
    name = str(metric_name).strip().lower()
    name = name.replace("-", " ").replace("_", " ")
    name = "_".join(name.split())
    if name.endswith(_CI_SUFFIX):
        name = name[: -len(_CI_SUFFIX)]
    return name


def _has_violation_token(name: str) -> bool:
    """True when a normalised name carries a violation-magnitude TOKEN.

    Tokens are whole underscore-delimited words. That is what keeps
    "cali[bratio]n" out of the ratio family and would equally keep a
    hypothetical "...gapped..." out of this one.
    """
    return bool(LOWER_IS_BETTER_TOKENS.intersection(name.split("_")))


def _has_ratio_shape(name: str) -> bool:
    """True when a normalised name is shaped like a parity ratio.

    Exact ``_ratio`` SUFFIX, or exact membership of :data:`RATIO_METRICS`.
    Deliberately no containment test of any kind: matching a family token
    anywhere inside a name is the cali-BRATIO-n bug class, and it has now been
    introduced here twice, once for "ratio" and once for "disparate_impact".
    """
    return name.endswith(_RATIO_SUFFIX) or name in RATIO_METRICS


def is_ratio_metric(metric_name: str) -> bool:
    """True for ratio-type fairness metrics, where HIGHER is better.

    A name that also carries a violation-magnitude token is NOT a ratio, even
    when it is shaped like one: ``disparate_impact_difference`` is the size of a
    violation, not a four-fifths ratio, and grading it as a ratio turns a large
    violation into a pass.
    """
    name = _normalize(metric_name)
    if not name:
        return False
    if _has_violation_token(name):
        return False
    return _has_ratio_shape(name)


def metric_direction(metric_name: str) -> MetricDirection:
    """Resolve which direction is better for ``metric_name``.

    Returns :attr:`MetricDirection.UNKNOWN` when the name carries no reliable
    signal. Callers MUST treat UNKNOWN as could-not-check and fail closed;
    guessing lower-is-better is what inverted the deployment gate for the whole
    ratio family.

    Precedence, in order, and pinned by ``tests/test_metric_direction.py``:

    1. an EXACT name in one of the two tables;
    2. a violation-magnitude TOKEN, which beats the ratio shape, so
       ``disparate_impact_difference`` is lower-is-better even though it names
       the disparate-impact family;
    3. the ratio shape (``_ratio`` suffix or an exact :data:`RATIO_METRICS`
       name);
    4. otherwise UNKNOWN. A name carrying BOTH a violation token and the ratio
       shape is contradictory, so it lands here too: we do not know which
       reading was meant, and either guess can certify a violation as a pass.
    """
    name = _normalize(metric_name)
    if not name:
        return MetricDirection.UNKNOWN
    # Explicit names win over every shape rule below.
    if name in HIGHER_IS_BETTER_METRICS:
        return MetricDirection.HIGHER_IS_BETTER
    if name in LOWER_IS_BETTER_METRICS:
        return MetricDirection.LOWER_IS_BETTER

    violation = _has_violation_token(name)
    ratio = _has_ratio_shape(name)
    if violation and ratio:
        return MetricDirection.UNKNOWN
    if violation:
        return MetricDirection.LOWER_IS_BETTER
    if ratio:
        return MetricDirection.HIGHER_IS_BETTER
    return MetricDirection.UNKNOWN


def check_threshold(
    metric_name: str,
    value: float,
    threshold: float,
) -> Tuple[ThresholdOutcome, str]:
    """Compare one metric value against one threshold, in that metric's direction.

    Args:
        metric_name: Name of the metric, used to resolve the direction.
        value: The measured value.
        threshold: The bound. For a lower-is-better metric this is a maximum
            allowed magnitude; for a higher-is-better metric it is a required
            minimum (e.g. 0.80 for the four-fifths rule).

    Returns:
        ``(outcome, message)``. The message is empty for
        :attr:`ThresholdOutcome.PASS` and states the reason otherwise.

    An unmeasurable value, an unmeasurable threshold, an unknown direction or a
    DEGENERATE bound all yield COULD_NOT_CHECK. None of the four is a PASS.
    "Unmeasurable" means anything :func:`vfairness._triage.is_measured` refuses:
    NaN, +/-infinity, None, a bool, a string. It said "(NaN)" here until
    READINESS-6, and the code matched the narrower word rather than the intent.

    **A bound that cannot be breached grades nothing.** Comparing a value
    against a threshold and reporting PASS when the comparison does not trip
    never asks whether the comparison COULD trip. On a higher-is-better metric
    the threshold is a required MINIMUM and the values are non-negative, so a
    minimum of 0.0 is met by every possible value including the worst one.
    Executed on this repo before 2026-08-27:
    ``check_threshold("disparate_impact_ratio", 0.00, 0.0)`` returned
    ``(PASS, "")``, and the deployment gate configured with that threshold
    returned ``approved=True`` with a GitHub check conclusion of ``success``
    for a protected group that is never selected at all. That is the
    substring-direction lie in a different costume: a metric nobody graded,
    reported as a metric that passed. ``rendering/adapters_fairness._metric_state``
    had built this guard locally and noted that the hole was still open here;
    it is closed here now, so every caller inherits the answer.

    The MIRROR case is deliberately NOT degenerate: 0.0 on a lower-is-better
    metric is a real zero-tolerance policy, and ``abs(value)`` can exceed it.
    """
    direction = metric_direction(metric_name)

    # NaN fails every comparison, so an unguarded ``value > threshold`` reports
    # "not breached" for a metric that was never measured. Route it out first.
    #
    # INFINITY is the OTHER HALF of the same hole, and it was open here until
    # READINESS-6 (2026-09-10). NaN loses every comparison; inf WINS them. On a
    # higher-is-better metric that is not a silent non-breach, it is an outright
    # PASS. Measured on this repo before this line changed:
    # ``check_threshold("disparate_impact_ratio", inf, 0.8)`` returned
    # ``(PASS, "")``, and ``ModelFairnessGate`` built on it answered
    # ``approved=True``, ``passed=True``, an empty message and a GitHub check
    # conclusion of ``success``, indistinguishable from the genuine pass at 0.95.
    # inf reaches the ratio family from a denominator of zero, which means a
    # group with NO SELECTIONS AT ALL: the most extreme unfairness the data can
    # express, arriving as the most reassuring verdict the gate can write.
    #
    # ``is_measured`` is the canonical rule and also refuses a bool, a string and
    # None, none of which is a measurement either. Fail closed on all of them.
    if not is_measured(value):
        return (
            ThresholdOutcome.COULD_NOT_CHECK,
            f"{metric_name} could not be measured on this data ({unmeasurable_reason(value)})",
        )
    if not is_measured(threshold):
        return (
            ThresholdOutcome.COULD_NOT_CHECK,
            f"{metric_name} has an unusable threshold "
            f"({unmeasurable_reason(threshold)}), so it could not be checked",
        )

    if direction is MetricDirection.HIGHER_IS_BETTER:
        # A required MINIMUM at or below zero cannot be breached: the ratio
        # family and the performance floors are non-negative, so EVERY possible
        # value satisfies it, the worst one included. See the degenerate-bound
        # note above; this used to fall straight through to PASS.
        if threshold <= 0.0:
            return (
                ThresholdOutcome.COULD_NOT_CHECK,
                f"{metric_name} has a required minimum of {threshold:.4f}, which no value "
                f"can fall below, so the bound was never applied and nothing was graded",
            )
        if value < threshold:
            return (
                ThresholdOutcome.FAIL,
                f"{metric_name} ({value:.4f}) is below the required minimum ({threshold:.4f})",
            )
        return (ThresholdOutcome.PASS, "")

    if direction is MetricDirection.LOWER_IS_BETTER:
        # The MIRROR degenerate case, and only the mirror. A maximum of 0.0 on a
        # violation magnitude is a real zero-tolerance policy that grades (any
        # non-zero magnitude exceeds it) and must keep working. A NEGATIVE
        # maximum is the degenerate one: no magnitude is below it, so the value
        # was never compared against a bound that could hold, and reporting a
        # measured FAIL would overstate what happened.
        if threshold < 0.0:
            return (
                ThresholdOutcome.COULD_NOT_CHECK,
                f"{metric_name} has a maximum of {threshold:.4f}, which no magnitude "
                f"can meet, so the bound was never applied and nothing was graded",
            )
        if abs(value) > threshold:
            return (
                ThresholdOutcome.FAIL,
                f"{metric_name} ({value:.4f}) exceeds threshold ({threshold:.4f})",
            )
        return (ThresholdOutcome.PASS, "")

    return (
        ThresholdOutcome.COULD_NOT_CHECK,
        f"{metric_name} has no known better-direction (is a lower value better "
        f"or worse?), so its threshold ({threshold:.4f}) could not be applied; "
        f"failing closed rather than reporting an unchecked metric as passing",
    )


def relax_threshold(
    metric_name: str,
    threshold: float,
    multiplier: float,
) -> float:
    """Move ``threshold`` in the PERMISSIVE direction for this metric.

    ``multiplier`` is a relaxation factor: 1.2 means "20 percent more permissive".
    For a lower-is-better metric that means allowing a larger gap (multiply); for
    a higher-is-better metric it means accepting a smaller ratio (divide).

    Multiplying blindly is what tightened the intersectional four-fifths floor
    from 0.80 to 0.96, i.e. it made the relaxed check STRICTER than the base
    check on exactly the groups it was meant to be gentler on.

    An unknown direction is returned unchanged: we cannot know which way is
    permissive, and moving it the wrong way could loosen a real bound.
    """
    if multiplier <= 0 or multiplier != multiplier:
        return threshold
    direction = metric_direction(metric_name)
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return threshold / multiplier
    if direction is MetricDirection.LOWER_IS_BETTER:
        return threshold * multiplier
    return threshold


def improvement_amount(
    metric_name: str,
    value: float,
    baseline_value: float,
) -> Optional[float]:
    """How much BETTER ``value`` is than ``baseline_value``, in this metric's direction.

    Positive means improved, negative means degraded. Returns ``None`` when the
    direction is unknown, so the caller can decide (a caller enforcing an
    improvement or degradation bound must not silently assume a direction), and
    NaN when either side is not a measurement.

    Three states, and the two non-numbers are distinct on purpose: ``None`` means
    "we do not know which way is better", NaN means "there was nothing to
    compare". A 0.0 for either would read as "unchanged", which is a
    measurement.
    """
    # THE SAME HOLE check_threshold CLOSED AT READINESS-6, and this was the one
    # function in this module with no ``is_measured`` gate at all. NaN already
    # propagated through the arithmetic below, so NaN looked handled; every OTHER
    # unmeasurable class came back as a plain number no caller can tell from a
    # measurement. Measured on this repo on 2026-09-27, all four with a KNOWN
    # direction so the ``return None`` path below is not involved:
    #   improvement_amount("disparate_impact_ratio", inf, 0.5)        -> inf
    #   improvement_amount("demographic_parity_difference", 0.5, inf) -> inf
    #   improvement_amount("disparate_impact_ratio", True, 0.5)       -> 0.5
    #   improvement_amount("disparate_impact_ratio", "0.9", 0.5)      -> 0.4
    # After this guard each of those four returns NaN. ``check_threshold`` in
    # this same module already answered COULD_NOT_CHECK for all four, so the two
    # halves of one module disagreed about the same value: the gate refused to
    # grade it and then published +inf as the improvement, which any caller
    # enforcing ``improvement >= margin`` reads as improved. inf reaches a ratio
    # from a denominator of zero, i.e. a group with no selections at all.
    #
    # ABOVE the direction dispatch, not inside either branch: the question "was
    # this measured at all" has the same answer whichever way better points, and
    # a guard below the dispatch cannot fire for the branch that was not taken.
    if not is_measured(value) or not is_measured(baseline_value):
        return float("nan")

    direction = metric_direction(metric_name)
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return float(value) - float(baseline_value)
    if direction is MetricDirection.LOWER_IS_BETTER:
        return abs(float(baseline_value)) - abs(float(value))
    return None


def metric_value_range(metric_name: str) -> Optional[Tuple[float, float]]:
    """The closed interval ``metric_name``'s values live in, or None when unknown.

    ``None`` means "this library does not declare a range for that name", never
    "the metric is unbounded". It is the could-not-check of this question, and
    every caller has to treat it as one: no claim about a bound can be made
    without it. See :data:`UNIT_INTERVAL_METRICS` for why the table is
    deliberately small and which families are left out on purpose.
    """
    name = _normalize(metric_name)
    if not name:
        return None
    if name in UNIT_INTERVAL_METRICS:
        return (0.0, 1.0)
    return None


class BoundRole(str, Enum):
    """Which comparison a configured bound controls, i.e. what makes a check FAIL.

    Named per ROLE rather than per parameter, because the same rule has to be
    applied to bounds that arrive under different names at different call sites
    (a gate threshold, a threshold relaxed by an intersectional multiplier, a
    per-intersection override, ``improvement_margin``,
    ``allow_degradation_margin``). One rule, one vocabulary.
    """

    #: The value breaches it: ``abs(value) > bound`` for a lower-is-better
    #: metric, ``value < bound`` for a higher-is-better one.
    THRESHOLD = "threshold"
    #: ``improvement < bound`` fails the requirement.
    REQUIRED_IMPROVEMENT = "required_improvement"
    #: ``degradation > bound`` fails the requirement.
    ALLOWED_DEGRADATION = "allowed_degradation"


def vacuous_bound_reason(
    metric_name: str,
    bound: float,
    role: BoundRole,
) -> Optional[str]:
    """Why ``bound`` cannot be breached by ANY value this metric can take, or None.

    **A BOUND IS UNUSABLE WHEN THE COMPARISON IT CONTROLS CANNOT BE FALSE FOR ANY
    DATA IN THE METRIC'S RANGE.** That is a property of the bound and the metric's
    range TOGETHER, not a property of the bound being finite. Infinity is merely
    the easiest case of it, and guarding only the infinities is what left every
    finite vacuous bound open: measured on this repo 2026-09-30, with a
    ``demographic_parity_difference`` of 1.0000, the largest a rate gap can be,
    and ``require_improvement`` against a baseline the model had degraded from::

        intersectional threshold  base 0.6 x multiplier 2.0   -> 1.2   APPROVED
        intersectional threshold  base 0.6 x multiplier 100.0 -> 60.0  APPROVED
        improvement_margin        -1.0                                 APPROVED
        improvement_margin        -1e9                                 APPROVED
        allow_degradation_margin  1e9                                  APPROVED

    while the infinity of each of those was already refused, with a message
    stating the criterion exactly ("the comparison that enforces the requirement
    is False whatever the data") that ``-1e9`` and ``60.0`` meet just as fully.

    Returns a reason fragment naming the range and the comparison, for the caller
    to put inside its own fail-closed sentence, or ``None`` when the bound CAN be
    breached and so grades something.

    ``None`` is also the answer, deliberately, in two cases where no claim can be
    made:

    * the metric has no declared range (:func:`metric_value_range`), so whether
      the bound is reachable is unknown. Refusing there would block bounds that
      are perfectly usable, which is the over-correction: a gate that blocks
      every deployment ships nothing and passes every refusal test.
    * the bound is not a measurement at all (NaN, an infinity, None, a bool, a
      string). Those are refused one layer up by the ``is_measured`` guards that
      already exist at each call site, and answering here as well would put two
      refusals on one bound.

    This function does NOT decide a verdict and has no side effect on
    :func:`check_threshold`, whose own degenerate-bound rule (a required minimum
    at or below zero, a negative maximum) is the subset of this rule that needed
    no range to detect. Every caller that wants the wider rule applies it
    explicitly; ``vfairness.operations.cicd.gate``, the surface that decides
    whether a model ships, applies it to all four of its bounds.
    """
    # Not a measurement: the caller's own is_measured guard owns that case, so
    # that one bound cannot collect two different refusals.
    if not is_measured(bound):
        return None
    value_range = metric_value_range(metric_name)
    if value_range is None:
        # No declared range, so nothing can be shown about reachability. Silence
        # here is the could-not-check, not an all-clear: it leaves the bound
        # exactly as usable as the caller configured it.
        return None

    low, high = value_range
    bound = float(bound)
    # The widest an improvement or a degradation can be is the full width of the
    # range, whichever way "better" points: improvement is ``value - baseline``
    # for a higher-is-better metric and ``abs(baseline) - abs(value)`` for a
    # lower-is-better one, and both are inside [-(high - low), high - low].
    span = high - low

    if role is BoundRole.REQUIRED_IMPROVEMENT:
        # ``improvement < bound`` is what fails the requirement. The worst
        # possible result improves by -span, so a bound at or below -span is met
        # by every result including the worst one.
        if bound <= -span:
            return (
                f"a required improvement of {bound:.4f} on a metric whose values lie in "
                f"[{low:.4f}, {high:.4f}], where the largest possible DEGRADATION is "
                f"{span:.4f}, so no result can fall short of it"
            )
        return None

    if role is BoundRole.ALLOWED_DEGRADATION:
        # ``degradation > bound`` is what fails the requirement, and degradation
        # cannot exceed span, so a bound at or above span allows everything.
        if bound >= span:
            return (
                f"an allowed degradation of {bound:.4f} on a metric whose values lie in "
                f"[{low:.4f}, {high:.4f}], where the largest possible degradation is "
                f"{span:.4f}, so no result can exceed it"
            )
        return None

    # THE THRESHOLD CASE IS EXPLICIT, and it was the FALL-THROUGH until 2026-09-30.
    # Every role this function does not handle was judged by the threshold rule, and
    # the threshold rule's vacuity direction is the OPPOSITE of the improvement
    # rule's: a threshold is unusable when `bound >= worst`, a required improvement
    # when `bound <= -span`. Measured on this tree, with values in [0, 1] and
    # bound=60: the string 'required_baseline_floor', the integer 0 and None all
    # returned the threshold reason, while the real REQUIRED_IMPROVEMENT role is
    # correctly silent at that bound. So a fourth member added later, or a caller
    # that loses the role, is judged by the wrong rule IN THE WRONG DIRECTION, which
    # is the exact substitution this function exists to prevent. "One rule, one
    # vocabulary" is the class docstring's promise and the fall-through broke it.
    #
    # An unrecognised role FAILS CLOSED, with a reason that says what is actually
    # wrong. It does not raise: the gate reads a non-None return as "refuse this
    # bound" and puts the string in front of a reader, so a refusal with an accurate
    # message is more useful here than an exception that stops the whole run, and it
    # needs no change at any of the five call sites.
    if role is not BoundRole.THRESHOLD:
        return (
            f"the bound {bound!r} was supplied under the role {role!r}, which this rule "
            "does not recognise, so whether it can be breached by any value this metric "
            "takes COULD NOT BE CHECKED. Refusing rather than judging it by the "
            "threshold rule, whose direction is the opposite of the improvement rule's"
        )

    direction = metric_direction(metric_name)
    if direction is MetricDirection.LOWER_IS_BETTER:
        # ``abs(value) > bound`` is the breach, so the bound has to be below the
        # largest magnitude the range holds. A maximum of 1.2 (or 60.0) on a rate
        # gap that cannot exceed 1.0 grades nothing at all.
        worst = max(abs(low), abs(high))
        if bound >= worst:
            return (
                f"a maximum of {bound:.4f} on a metric whose values lie in "
                f"[{low:.4f}, {high:.4f}], where the largest possible magnitude is "
                f"{worst:.4f}, so no result can exceed it"
            )
        return None

    if direction is MetricDirection.HIGHER_IS_BETTER:
        # ``value < bound`` is the breach, so a required minimum at or below the
        # bottom of the range is met by every value. check_threshold already
        # refuses the ``bound <= 0`` case with its own message and it is the same
        # case whenever low is 0.0; stated in range terms here so the rule reads
        # the same for both roles.
        if bound <= low:
            return (
                f"a required minimum of {bound:.4f} on a metric whose values lie in "
                f"[{low:.4f}, {high:.4f}], so no result can fall below it"
            )
        return None

    # An unknown direction is not gradable at all, which check_threshold already
    # fails closed on. Claiming vacuity here would need a direction we do not
    # have.
    return None
