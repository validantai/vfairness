"""Deployment policies: the coverage a deployment REQUIRES, compared with what ran.

vfairness issue #1 (opened 2026-10-07): "Define policy-required assessment coverage
before deployment approval". The gate only knows what it was configured to do, so
every way of configuring it to do less still ended in an approval. Measured
2026-10-24 on four cells of 100 rows, every single-attribute group at a 0.5
selection rate and the two-way cells at 0.8 / 0.2 / 0.2 / 0.8, metric
demographic_parity_difference with threshold 0.1 and min_group_size 30::

    change from the defaults                           approved  status
    none (intersections checked)                       False     BLOCKED
    check_intersections=False                          True      APPROVED
    second attribute not supplied                      True      CONDITIONAL
    threshold 0.7                                      True      APPROVED
    blocking_metrics=[]                                True      CONDITIONAL
    default_intersection_threshold_multiplier=6.5      True      APPROVED
    per_intersection_thresholds 0.7 for the cell       True      APPROVED

and on cells of 95 / 5 / 5 / 95, ``min_group_size=1`` gave APPROVED with
``small_sample_check_ran=False``. Every one of those runs stated its scope
somewhere; none of them could be refused, because nothing told the gate what a
deployment requires.

THE SPLIT. This module owns the COMPARISON, and only the comparison:

* a :class:`CoveragePolicy` names what must be covered: attributes, named
  intersections, metrics with the loosest bound the policy accepts, and a minimum
  group size of its own;
* :meth:`ModelFairnessGate.evaluate_against_policy` runs the ordinary hierarchical
  measurement, UNCHANGED, and returns a :class:`PolicyDecision` beside it in which
  every required item has exactly one of four coverage states and, separately, a
  measured outcome;
* ``permitted`` is True only when every required item is assessed and passes and
  the measurement itself was not blocked.

What it deliberately does NOT do: choose groups for a domain, infer an attribute
that was not supplied, run a level the configuration switched off, or accept an
exception. There is no parameter through which a caller can turn a coverage gap
into a permission. Which policy applies to which deployment, who may grant an
exception and how that person is verified, its scope, rationale, evidence, expiry
and revocation belong to the integration that enforces the policy, which also keeps
the historical assessment.

The fingerprints on a :class:`PolicyDecision` are SHA-256 digests of the policy and
of the effective configuration. They identify WHICH policy and WHICH settings were
used. They are not signatures and establish nothing about who chose either.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vfairness._triage import is_flag, is_measured, unmeasurable_reason
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
    relax_threshold,
    vacuous_bound_reason,
)
from vfairness.exceptions import ConfigurationError

from .gate import (
    DEFAULT_MIN_GROUP_SIZE,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    IntersectionalGateDecision,
    MetricEvaluation,
    _metric_value_cell,
    _undersized_groups,
    _undersized_rate_denominators,
)


class CoverageState(Enum):
    """How far the assessment covered one required item. Exactly one per item.

    The four are kept apart because each asks for a different next action: fix the
    data, fix the settings, or collect more evidence. Only ASSESSED says anything
    about fairness, and even then the verdict is the separate
    :class:`MeasuredOutcome`.
    """

    #: A level compared this group set on this metric, the configured gate enforces
    #: it at least as strictly as the policy, and every group held enough rows.
    ASSESSED = "assessed"
    #: It ran, and the evidence is too thin to certify: a group or a rate
    #: denominator below the policy's minimum, or a value that could not be
    #: measured. Never a measured fairness failure.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: The configuration did not assess it, or assessed it more loosely than the
    #: policy: a level switched off, the metric not configured or not blocking, or
    #: a threshold looser than the policy bound. The reason names the setting.
    OMITTED_BY_CONFIGURATION = "omitted_by_configuration"
    #: An attribute it needs was not handed to the gate. The required scope is
    #: never reduced to the attributes that were supplied.
    INPUT_ABSENT = "input_absent"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


class MeasuredOutcome(Enum):
    """The measured value against the POLICY bound, recorded apart from coverage.

    A value can be measured and still not count (a looser configured threshold, a
    five-person group), which is why this is not folded into
    :class:`CoverageState`.
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_MEASURED = "not_measured"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


def _require_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _names(value: Any, what: str) -> Tuple[str, ...]:
    # A bare string is a sequence of characters, and ("race") is the string
    # "race", so accepting it would silently require the attributes 'r', 'a', 'c'
    # and 'e'.
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{what} must be a list or tuple of names, got {value!r}")
    names: List[str] = []
    for item in value:
        _require_name(item, f"each entry of {what}")
        if item not in names:
            names.append(item)
    return tuple(names)


def _bound_problem(metric: str, bound: Any) -> Optional[str]:
    """Why ``bound`` cannot serve as a policy bound for ``metric``, or None."""
    if is_flag(bound) or not is_measured(bound):
        return f"it is not a finite number ({unmeasurable_reason(bound)})"
    direction = metric_direction(metric)
    if direction is MetricDirection.UNKNOWN:
        return (
            "the library does not know whether a lower or a higher value of this "
            "metric is better, so no configured threshold can be compared with it"
        )
    # The degenerate bounds check_threshold refuses, so a policy can never be
    # written that no data could breach.
    if direction is MetricDirection.HIGHER_IS_BETTER and bound <= 0.0:
        return f"a required minimum of {bound} is met by every value, so it grades nothing"
    if direction is MetricDirection.LOWER_IS_BETTER and bound < 0.0:
        return f"a maximum of {bound} can be met by no value"
    vacuous = vacuous_bound_reason(metric, bound, BoundRole.THRESHOLD)
    if vacuous:
        return vacuous
    return None


@dataclass(frozen=True)
class CoveragePolicy:
    """What a deployment requires an assessment to cover. Opt in, versioned.

    Attributes:
        policy_id: The integration's name for the policy.
        version: Its version. A changed requirement is a new version, so a
            decision can always be traced to the exact requirement it was held to.
        required_attributes: Every protected attribute that must be assessed on
            its own. Order is kept for display; it does not change the fingerprint.
        required_metrics: Metric name to the LOOSEST bound the policy accepts: a
            maximum for a lower-is-better metric, a minimum for a higher-is-better
            one (``{"disparate_impact_ratio": 0.8}``). A configured threshold
            looser than this, at any level, makes that item omitted by
            configuration.
        required_intersections: Named intersections that must be assessed, each
            two or more of the required attributes, for example
            ``[("gender", "race")]``. Order inside one does not matter.
        min_group_size: The policy's own evidence floor. Group rows (and, for the
            gate's own conditional rates, the rows each rate rests on) are counted
            by this module against THIS number, never against the configured one,
            so a configured minimum of 1 cannot hide a five-person group. 2 or more.

    Raises:
        ConfigurationError: for a policy that could not be enforced as written: no
            attributes or metrics, an intersection of fewer than two attributes or
            of an attribute that is not required, a bound that grades nothing or
            whose metric direction is unknown, or a minimum below 2.
    """

    policy_id: str
    version: str
    required_attributes: Tuple[str, ...]
    required_metrics: Mapping[str, float]
    required_intersections: Tuple[Tuple[str, ...], ...] = ()
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE

    def __post_init__(self) -> None:
        _require_name(self.policy_id, "policy_id")
        _require_name(self.version, "version")
        attributes = _names(self.required_attributes, "required_attributes")
        if not attributes:
            raise ConfigurationError(
                "required_attributes is empty, so the policy requires nothing; name at "
                "least one protected attribute"
            )

        if isinstance(self.required_intersections, str) or not isinstance(
            self.required_intersections, Sequence
        ):
            raise ConfigurationError(
                f"required_intersections must be a list of attribute lists, got "
                f"{self.required_intersections!r}"
            )
        intersections: List[Tuple[str, ...]] = []
        seen = set()
        for raw in self.required_intersections:
            names = _names(raw, "each required intersection")
            if len(names) < 2:
                raise ConfigurationError(
                    f"the intersection {raw!r} combines fewer than two attributes, so it "
                    f"is not an intersection; require the attribute on its own instead"
                )
            unknown = [n for n in names if n not in attributes]
            if unknown:
                raise ConfigurationError(
                    f"the intersection {raw!r} names {unknown!r}, which "
                    f"required_attributes does not list; add it there too"
                )
            # In required_attributes order, so ("race", "gender") and
            # ("gender", "race") are the same requirement and display alike.
            ordered = tuple(a for a in attributes if a in names)
            if frozenset(ordered) not in seen:
                seen.add(frozenset(ordered))
                intersections.append(ordered)

        if not isinstance(self.required_metrics, Mapping) or not self.required_metrics:
            raise ConfigurationError(
                "required_metrics must name at least one metric with its bound, for "
                'example {"demographic_parity_difference": 0.1}'
            )
        metrics: Dict[str, float] = {}
        for metric, bound in self.required_metrics.items():
            _require_name(metric, "each required metric name")
            problem = _bound_problem(metric, bound)
            if problem:
                raise ConfigurationError(f"the policy bound {bound!r} for {metric}: {problem}")
            metrics[metric] = float(bound)

        if (
            is_flag(self.min_group_size)
            or not isinstance(self.min_group_size, (int, np.integer))
            or self.min_group_size < 2
        ):
            raise ConfigurationError(
                f"min_group_size must be a whole number of 2 or more, got "
                f"{self.min_group_size!r}: every group holds at least one row, so a "
                f"minimum of 1 or less could never flag anything"
            )

        object.__setattr__(self, "required_attributes", attributes)
        object.__setattr__(self, "required_intersections", tuple(intersections))
        object.__setattr__(self, "required_metrics", metrics)
        object.__setattr__(self, "min_group_size", int(self.min_group_size))

    @property
    def group_sets(self) -> Tuple[Tuple[str, ...], ...]:
        """Every required group set: each attribute on its own, then each intersection."""
        return tuple((a,) for a in self.required_attributes) + tuple(self.required_intersections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "required_attributes": list(self.required_attributes),
            "required_intersections": [list(i) for i in self.required_intersections],
            "required_metrics": dict(self.required_metrics),
            "min_group_size": self.min_group_size,
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the requirement, independent of the order it was written in."""
        canonical = {
            **self.to_dict(),
            "required_attributes": sorted(self.required_attributes),
            "required_intersections": sorted(sorted(i) for i in self.required_intersections),
        }
        return _fingerprint(canonical)


@dataclass(frozen=True)
class CoverageItem:
    """One required group set on one required metric, and how far it was covered.

    Attributes:
        attributes: The attribute, or the attributes of the intersection.
        metric: The required metric.
        state: The coverage state, exactly one.
        outcome: The measured value against the policy bound, recorded separately.
        reason: Why the item is in that state, naming the setting where one is the
            cause.
        level: The level of the measurement that evaluated it, or None.
        value: The measured value there, or None when no level measured it.
        threshold_used: The threshold the configured gate applied there, or None.
        policy_bound: The bound the policy accepts.
        undersized: The groups and rate denominators below the policy's minimum, as
            ``(name, rows)`` pairs.
        note: What this item's evidence check could not see, when anything.
    """

    attributes: Tuple[str, ...]
    metric: str
    state: CoverageState
    outcome: MeasuredOutcome
    reason: str
    level: Optional[str]
    value: Optional[float]
    threshold_used: Optional[float]
    policy_bound: float
    undersized: Tuple[Tuple[str, int], ...] = ()
    note: str = ""

    @property
    def label(self) -> str:
        return " × ".join(self.attributes)

    @property
    def satisfied(self) -> bool:
        """Assessed AND passed: the only combination that counts toward a permission."""
        return self.state is CoverageState.ASSESSED and self.outcome is MeasuredOutcome.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": list(self.attributes),
            "metric": self.metric,
            "state": self.state.value,
            "outcome": self.outcome.value,
            "satisfied": self.satisfied,
            "reason": self.reason,
            "level": self.level,
            "value": self.value,
            "threshold_used": self.threshold_used,
            "policy_bound": self.policy_bound,
            "undersized": [[name, rows] for name, rows in self.undersized],
            "note": self.note,
        }


@dataclass
class PolicyDecision:
    """An assessment held to a :class:`CoveragePolicy`, beside the measurement.

    ``permitted`` is the ONLY field here that means "may be deployed under this
    policy". The measurement keeps its own meaning: its ``approved`` says nothing
    blocked on the levels that ran, which is a statement about those levels and
    never about the ones the policy requires and the configuration left out. There
    is deliberately no ``approved`` on this class, so the two cannot be confused.

    Attributes:
        policy: The policy the assessment was held to.
        permitted: True only when every required item is assessed and passes, and
            the measurement itself was not blocked.
        items: One per required group set and required metric.
        reasons: Why it is not permitted, one line per unsatisfied item plus one
            for a blocked measurement. Empty when permitted.
        measurement: The ordinary hierarchical decision, unchanged by the policy.
        policy_fingerprint: ``sha256:`` digest of the policy.
        configuration: The effective configuration the measurement ran with.
        configuration_fingerprint: ``sha256:`` digest of ``configuration``.
        timestamp: When the comparison was made.
    """

    policy: CoveragePolicy
    permitted: bool
    items: List[CoverageItem]
    reasons: List[str]
    measurement: IntersectionalGateDecision
    policy_fingerprint: str
    configuration: Dict[str, Any]
    configuration_fingerprint: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def gaps(self) -> List[CoverageItem]:
        """Every required item that does not count toward a permission."""
        return [item for item in self.items if not item.satisfied]

    @property
    def summary(self) -> str:
        satisfied = len(self.items) - len(self.gaps)
        head = "PERMITTED" if self.permitted else "NOT PERMITTED"
        return (
            f"{head} under policy {self.policy.policy_id} v{self.policy.version}: "
            f"{satisfied} of {len(self.items)} required item(s) assessed and passing"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "permitted": self.permitted,
            "items": [item.to_dict() for item in self.items],
            "reasons": list(self.reasons),
            "measurement": self.measurement.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "configuration": self.configuration,
            "configuration_fingerprint": self.configuration_fingerprint,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_markdown_report(self) -> str:
        lines = ["# Fairness Policy Report", ""]
        lines.extend(_policy_section_lines(self, heading="##"))
        lines.extend(["", "---", ""])
        lines.append(self.measurement.to_markdown_report())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """A JSON-safe copy whose serialisation is stable, for fingerprinting only."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        # json.dumps would write NaN / Infinity, which is not JSON; a configuration
        # carrying one still has to fingerprint, and differently from a number.
        return f"non-finite:{value!r}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot fingerprint a value of type {type(value).__name__}: {value!r}")


def _fingerprint(value: Any) -> str:
    text = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _effective_configuration(
    gate_config: GateConfig,
    hconfig: HierarchicalGateConfig,
    compute_metrics_fn: Optional[Callable],
) -> Dict[str, Any]:
    fn = None
    if compute_metrics_fn is not None:
        fn = (
            f"{getattr(compute_metrics_fn, '__module__', '?')}."
            f"{getattr(compute_metrics_fn, '__qualname__', type(compute_metrics_fn).__name__)}"
        )
    return {
        "gate": gate_config.to_dict(),
        "hierarchy": hconfig.to_dict(),
        "compute_metrics_fn": fn,
    }


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _fmt(number: Any) -> str:
    return f"{float(number):.4g}" if is_measured(number) else str(number)


def _group_values(protected_attrs: Mapping[str, Any], ordered: Tuple[str, ...]) -> np.ndarray:
    """The group labels of one group set, built exactly as evaluate_hierarchical builds them."""
    if len(ordered) == 1:
        return np.asarray(protected_attrs[ordered[0]])
    combo: Optional[np.ndarray] = None
    for name in ordered:
        vals = np.array([str(v) for v in protected_attrs[name]])
        combo = vals if combo is None else np.array([f"{a}_{b}" for a, b in zip(combo, vals)])
    assert combo is not None
    return combo


def _find_level(
    ordered: Tuple[str, ...],
    measurement: IntersectionalGateDecision,
    attr_names: List[str],
    hconfig: HierarchicalGateConfig,
) -> Tuple[Optional[str], str]:
    """The level that evaluated this group set, or None and the setting that left it out."""
    levels = measurement.level_results
    if len(ordered) == 1:
        attribute = ordered[0]
        if f"attr:{attribute}" in levels:
            return f"attr:{attribute}", ""
        # The overall level compares the FIRST supplied attribute and only that one.
        if attr_names and attribute == attr_names[0] and "overall" in levels:
            return "overall", ""
        return None, "check_single_attributes is off, so no level compared this attribute's groups"
    key = "intersection:" + "_x_".join(ordered)
    if key in levels:
        return key, ""
    if not hconfig.check_intersections:
        return None, "check_intersections is off, so no level compared this intersection"
    if hconfig.intersection_depth < len(ordered):
        return None, (
            f"intersection_depth is {hconfig.intersection_depth}, below the "
            f"{len(ordered)} attributes this intersection combines"
        )
    return None, "the intersectional level did not reach this intersection"


def _is_looser(metric: str, threshold: Any, bound: float) -> bool:
    if threshold is None or is_flag(threshold) or not is_measured(threshold):
        return True
    direction = metric_direction(metric)
    if direction is MetricDirection.LOWER_IS_BETTER:
        return float(threshold) > bound
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return float(threshold) < bound
    return True


def _looser_threshold_reason(
    metric: str,
    used: Any,
    bound: float,
    level: str,
    gate_config: GateConfig,
    hconfig: HierarchicalGateConfig,
) -> Optional[str]:
    """Which setting made the threshold applied at ``level`` looser than the policy, or None."""
    if not _is_looser(metric, used, bound):
        return None
    if used is None or not is_measured(used):
        return (
            f"no usable threshold was applied to {metric} at {level} "
            f"(GateConfig.thresholds has {gate_config.thresholds.get(metric)!r})"
        )
    tail = f"looser than the policy bound {_fmt(bound)}"
    base = gate_config.thresholds.get(metric)
    if not level.startswith("intersection:") or _is_looser(metric, base, bound):
        return f"GateConfig.thresholds sets {_fmt(base)} for {metric}, {tail}"
    multiplier = hconfig.default_intersection_threshold_multiplier
    if base is not None and relax_threshold(metric, base, multiplier) == used:
        return (
            f"default_intersection_threshold_multiplier {_fmt(multiplier)} relaxes "
            f"{metric} from {_fmt(base)} to {_fmt(used)} at {level}, {tail}"
        )
    return f"per_intersection_thresholds sets {_fmt(used)} for {metric} at {level}, {tail}"


def _evaluation(
    measurement: IntersectionalGateDecision, level: str, metric: str
) -> Optional[MetricEvaluation]:
    for evaluation in measurement.level_results[level].metric_evaluations:
        if evaluation.metric_name == metric:
            return evaluation
    return None


def _outcome(metric: str, value: Any, bound: float) -> MeasuredOutcome:
    if value is None or not is_measured(value):
        return MeasuredOutcome.NOT_MEASURED
    verdict, _ = check_threshold(metric, value, bound)
    if verdict is ThresholdOutcome.PASS:
        return MeasuredOutcome.PASS
    if verdict is ThresholdOutcome.FAIL:
        return MeasuredOutcome.FAIL
    return MeasuredOutcome.NOT_MEASURED


def assess_coverage(
    policy: CoveragePolicy,
    measurement: IntersectionalGateDecision,
    *,
    protected_attrs: Mapping[str, Any],
    y_true: Any,
    y_pred: Any,
    gate_config: GateConfig,
    hierarchical_config: HierarchicalGateConfig,
    compute_metrics_fn: Optional[Callable] = None,
) -> PolicyDecision:
    """Hold one finished measurement to ``policy``. Pure: it changes nothing it is handed.

    Called by :meth:`ModelFairnessGate.evaluate_against_policy`, which is the entry
    point to use; it is public so an integration can re-check a stored measurement
    against a newer policy version.

    Per required group set and metric, the state is decided in this order, and the
    first that applies wins: an attribute not supplied is INPUT ABSENT; a level
    switched off, the metric not configured or not blocking, or a threshold looser
    than the policy bound is OMITTED BY CONFIGURATION; a group or rate denominator
    below the policy's minimum, or a value that could not be measured, is
    INSUFFICIENT EVIDENCE; anything else is ASSESSED. The measured outcome is
    graded against the POLICY bound whatever the state.
    """
    attr_names = list(protected_attrs.keys())
    items: List[CoverageItem] = []
    for group_set in policy.group_sets:
        missing = [a for a in group_set if a not in protected_attrs]
        if missing:
            named = ", ".join(repr(a) for a in missing)
            verb = "was" if len(missing) == 1 else "were"
            for metric, bound in policy.required_metrics.items():
                items.append(
                    CoverageItem(
                        attributes=group_set,
                        metric=metric,
                        state=CoverageState.INPUT_ABSENT,
                        outcome=MeasuredOutcome.NOT_MEASURED,
                        reason=(
                            f"{named} {verb} not supplied to the gate, so nothing about "
                            f"this group set was measured"
                        ),
                        level=None,
                        value=None,
                        threshold_used=None,
                        policy_bound=bound,
                    )
                )
            continue

        # The gate's own attribute order, so the label matches its level name.
        ordered = tuple(a for a in attr_names if a in group_set)
        level, level_gap = _find_level(ordered, measurement, attr_names, hierarchical_config)
        values = _group_values(protected_attrs, ordered)
        # Counted HERE against the policy's floor, never read from the measurement:
        # the measurement sized its groups against the configured minimum, which is
        # exactly the setting a policy must not depend on.
        row_floor = _undersized_groups(values, policy.min_group_size)

        for metric, bound in policy.required_metrics.items():
            omitted: List[str] = []
            if metric not in gate_config.metrics:
                omitted.append(f"{metric} is not one of the gate's configured metrics")
            if level is None:
                omitted.append(level_gap)
            if (
                gate_config.blocking_metrics is not None
                and metric not in gate_config.blocking_metrics
            ):
                omitted.append(
                    f"GateConfig.blocking_metrics leaves {metric} out, so the configured "
                    f"gate would only warn on a breach"
                )
            evaluation = (
                _evaluation(measurement, level, metric)
                if level is not None and metric in gate_config.metrics
                else None
            )
            used = evaluation.threshold if evaluation is not None else None
            if evaluation is not None and level is not None:
                looser = _looser_threshold_reason(
                    metric, used, bound, level, gate_config, hierarchical_config
                )
                if looser:
                    omitted.append(looser)
            value = evaluation.value if evaluation is not None else None

            thin: List[str] = []
            undersized: List[Tuple[str, int]] = []
            note = ""
            if evaluation is not None:
                if row_floor:
                    detail = ", ".join(f"'{w.group_name}' n={w.sample_size}" for w in row_floor)
                    thin.append(
                        f"{len(row_floor)} group(s) hold fewer rows than the policy's "
                        f"minimum of {policy.min_group_size} ({detail})"
                    )
                    undersized.extend((w.group_name, w.sample_size) for w in row_floor)
                if compute_metrics_fn is None:
                    legs = _undersized_rate_denominators(
                        metric,
                        y_true,
                        y_pred,
                        values,
                        policy.min_group_size,
                        skip_groups=frozenset(w.group_name for w in row_floor),
                    )
                    if legs:
                        detail = ", ".join(f"{w.group_name} n={w.sample_size}" for w in legs)
                        thin.append(
                            f"{metric} rests on rates measured over fewer rows than the "
                            f"policy's minimum of {policy.min_group_size} ({detail})"
                        )
                        undersized.extend((w.group_name, w.sample_size) for w in legs)
                else:
                    note = (
                        "group rows were counted against the policy's minimum; the rows "
                        "each rate rests on were not, because the numbers came from a "
                        "compute_metrics_fn whose inputs this library cannot see"
                    )
                if not is_measured(value):
                    thin.append(f"{metric} could not be measured ({unmeasurable_reason(value)})")

            outcome = _outcome(metric, value, bound)
            if omitted:
                state, reason = CoverageState.OMITTED_BY_CONFIGURATION, "; ".join(omitted)
            elif thin:
                state, reason = CoverageState.INSUFFICIENT_EVIDENCE, "; ".join(thin)
            else:
                state = CoverageState.ASSESSED
                reason = (
                    f"{_fmt(value)} is inside the policy bound {_fmt(bound)}"
                    if outcome is MeasuredOutcome.PASS
                    else f"measured breach: {_fmt(value)} against the policy bound {_fmt(bound)}"
                )
            items.append(
                CoverageItem(
                    attributes=ordered,
                    metric=metric,
                    state=state,
                    outcome=outcome,
                    reason=reason,
                    level=level,
                    value=value,
                    threshold_used=used,
                    policy_bound=bound,
                    undersized=tuple(undersized),
                    note=note,
                )
            )

    reasons = [
        f"{item.label}, {item.metric}: {item.state.label}: {item.reason}"
        for item in items
        if not item.satisfied
    ]
    if not measurement.approved:
        reasons.append(
            f"the measurement itself is {measurement.status.value.upper()} with "
            f"{len(measurement.blocking_reasons)} blocking reason(s) on the levels that "
            f"ran, listed in the measurement report"
        )
    configuration = _effective_configuration(gate_config, hierarchical_config, compute_metrics_fn)
    return PolicyDecision(
        policy=policy,
        permitted=not reasons,
        items=items,
        reasons=reasons,
        measurement=measurement,
        policy_fingerprint=policy.fingerprint,
        configuration=configuration,
        configuration_fingerprint=_fingerprint(configuration),
    )


# ---------------------------------------------------------------------------
# One wording, for the policy report, the report card and the GitHub check
# ---------------------------------------------------------------------------

_SCOPE_SENTENCE = (
    "The measurement keeps its own meaning: APPROVED there covers only the levels that "
    "ran. Under this policy only the result line above means the model may be deployed."
)
_FINGERPRINT_SENTENCE = (
    "Fingerprints identify which policy and which settings were used. They are not "
    "signatures and establish nothing about who chose them."
)


def policy_result_line(decision: PolicyDecision) -> str:
    """The one-line result every surface leads with."""
    head = "PERMITTED" if decision.permitted else "NOT PERMITTED"
    return (
        f"**Deployment policy**: `{head}` under `{decision.policy.policy_id}` "
        f"v{decision.policy.version}"
    )


def _policy_section_lines(
    decision: PolicyDecision, heading: str = "##", include_result: bool = True
) -> List[str]:
    satisfied = len(decision.items) - len(decision.gaps)
    lines = [
        f"{heading} Deployment Policy: {decision.policy.policy_id} v{decision.policy.version}",
        "",
    ]
    if include_result:
        lines.append(policy_result_line(decision))
    lines.extend(
        [
            f"{satisfied} of {len(decision.items)} required item(s) are assessed and pass.",
            "",
            "| Required | Metric | Coverage | Measured | Value | Threshold used | Policy bound |",
            "|----------|--------|----------|----------|-------|----------------|--------------|",
        ]
    )
    for item in decision.items:
        value = _metric_value_cell(item.value) if item.value is not None else "not measured"
        used = _fmt(item.threshold_used) if item.threshold_used is not None else "none"
        lines.append(
            f"| {item.label} | {item.metric} | {item.state.label} | {item.outcome.label} "
            f"| {value} | {used} | {_fmt(item.policy_bound)} |"
        )
    if decision.reasons:
        lines.extend(["", f"{heading}# Why it is not permitted", ""])
        lines.extend(f"- {reason}" for reason in decision.reasons)
    notes = sorted({item.note for item in decision.items if item.note})
    if notes:
        lines.extend(["", f"{heading}# What the evidence check could not see", ""])
        lines.extend(f"- {note}" for note in notes)
    status = decision.measurement.status
    lines.extend(
        [
            "",
            f"- Measurement status: `{status.value.upper()}`. {_SCOPE_SENTENCE}",
            f"- Policy fingerprint `{decision.policy_fingerprint}`, configuration "
            f"fingerprint `{decision.configuration_fingerprint}`. {_FINGERPRINT_SENTENCE}",
        ]
    )
    return lines


def policy_check_conclusion(decision: PolicyDecision) -> str:
    """The GitHub check conclusion: failure unless permitted, neutral for a CONDITIONAL measurement."""
    if not decision.permitted:
        return "failure"
    if decision.measurement.status is GateStatus.CONDITIONAL or decision.measurement.warnings:
        return "neutral"
    return "success"
