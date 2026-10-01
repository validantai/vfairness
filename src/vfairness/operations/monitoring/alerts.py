"""
Unit 2: Drift Detection: Alert Mechanisms
==========================================

This module implements the alert infrastructure described in Part 4, Unit 2
of the Fairness Pipeline Development Toolkit.

Two classes complete the drift-detection pipeline:

:class:`AdaptiveThresholdManager`
    Maintains a dictionary of per-metric-group alert thresholds that evolve
    over time based on human feedback.  When engineers confirm that an alert
    was a false positive, the threshold is raised slightly (reducing
    sensitivity).  When alerts are consistently valid, the threshold is
    lowered (increasing sensitivity).  This mechanism prevents both alert
    fatigue (too many false positives) and blind spots (missed real drift).

    The learning rule directly implements the algorithm from the MegaShop
    e-commerce case study in Unit 2, which achieved an 80% reduction in
    false-positive alerts while tripling detection speed for real bias.

:class:`FairnessAlertPrioritizer`
    Translates a raw drift event dictionary into a multi-factor priority
    score and maps the resulting severity to a notification channel.

    Scoring factors and their default weights (configurable):

    - ``regulatory_risk`` (×3.5): Protected-attribute status and jurisdiction.
    - ``drift_velocity`` (×2.5): Speed of metric change.
    - ``historical_discrimination`` (×3.0): Whether the affected group has a
      documented history of discrimination in the domain.
    - ``population_impact`` (×2.0): Fraction of total users affected.

    Severity bands:

    - Score > 8 → CRITICAL → PagerDuty (on-call engineer)
    - Score > 5 → HIGH     → Slack #fairness-alerts
    - Score ≤ 5 → LOW      → Jira ticket creation

    An alert payload is produced for each drift event, suitable for logging
    to a monitoring platform or dispatching to a notification service.

References
----------
Chen, Johansson & Sontag (2024). Why is my model suddenly unfair? ICML.
Mitchell et al. (2024). Algorithmic fairness: Choices, assumptions, and
  definitions. ACM Computing Surveys.
Kleinberg et al. (2017). Inherent trade-offs in the fair determination of
  risk scores. ITCS.
"""

from __future__ import annotations

import math
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..._triage import is_measured


def _as_measured_number(value: Any) -> Optional[float]:
    """Read *value* as a measured finite number, or ``None`` if it is not one.

    ``None`` here means COULD NOT CHECK: the value is absent, non-numeric,
    non-finite, or a flag. A measured ``0.0`` comes back as ``0.0`` and never
    as ``None``.

    The question "did this actually get measured?" is answered once, in
    :func:`vfairness._triage.is_measured`, and this wrapper only turns that
    answer into the ``Optional[float]`` the two call sites below want. Do not
    re-derive the rule here: the same predicate was re-invented in five places
    across four audits, and the copies disagreed.

    Notes
    -----
    BGL g014, 2026-09-17. The gate this replaces was
    ``isinstance(value, (int, float))``, and ``np.float64`` is a subclass of
    ``float`` while ``np.float32``, ``np.float16``, ``np.int64`` and
    ``np.bool_`` are NOT. So a drift score that was genuinely measured, by any
    pipeline carrying single-precision data, was read as never measured.
    Measured on HEAD, one drift event whose only difference was the dtype of a
    0.82 drift score::

        drift_score=0.82              -> (11.17, 'CRITICAL')  pagerduty
        drift_score=np.float64(0.82)  -> (11.17, 'CRITICAL')  pagerduty
        drift_score=np.float32(0.82)  -> (nan,   'UNSCORED')  slack triage
                                         warning: "the drift this alert is
                                         ABOUT was never measured"

    while the SAME value reached the payload as ``drift_score=0.82`` and the
    message printed "Drift score: 0.820" in the sentence that said the severity
    could not be computed. That is a real measurement thrown away and a real
    CRITICAL de-escalated, which is the refusal defect running backwards.
    ``_triage``'s own docstring records the identical finding, one audit
    earlier, which is why the answer lives there.

    Two consequences of adopting the canonical rule, both deliberate. A
    ``bool`` drift score is now COULD NOT CHECK rather than the silent ``+1.0``
    it used to add: a flag is not a drift magnitude. A ``str`` is not parsed,
    and neither is an ``ndarray``, 0-d or otherwise; ``numbers.Real`` excludes
    both by design. The four weighted factors keep their own ``math.isfinite``
    reading, which already accepts every numpy scalar and was never part of
    this defect; whether a bool factor should be excluded there is a separate
    question this wave did not measure.
    """
    return float(value) if is_measured(value) else None


@dataclass
class AlertPayload:
    """Structured alert payload dispatched to a notification channel.

    Attributes
    ----------
    alert_id : str
        UUID for deduplication and audit-trail purposes.
    timestamp : datetime
        When the alert was generated.
    severity : str
        One of ``"CRITICAL"``, ``"HIGH"``, ``"LOW"``, or ``"UNSCORED"``. The
        last one means the severity could not be computed and the alert is
        neither confirmed critical nor cleared.
    priority_score : float
        Raw multi-factor score from :class:`FairnessAlertPrioritizer`.
        ``float("nan")`` when no severity factor could be measured.
    metric_name : str
        Fairness metric that triggered the alert.
    affected_groups : list[str]
        Demographic groups / intersections showing drift.
    drift_score : float
        Composite KS drift score. ``float("nan")`` when no drift score was
        supplied: that is COULD NOT CHECK, not a metric that stayed still.
    mean_shift : float
        Change in metric mean (current - reference). ``float("nan")`` when it
        was never measured.
    routing : dict
        Channel-routing instructions from :meth:`FairnessAlertPrioritizer.route_alert`.
    drift_event : dict
        Raw drift-event dictionary passed to the prioritizer.
    message : str
        Human-readable summary.
    acknowledged : bool
        Set to ``True`` once an engineer has reviewed the alert.
    resolution : str, optional
        Free-text resolution notes.
    """

    alert_id: str
    timestamp: datetime
    severity: str
    priority_score: float
    metric_name: str
    affected_groups: List[str]
    drift_score: float
    mean_shift: float
    routing: Dict[str, str]
    drift_event: Dict[str, Any]
    message: str
    acknowledged: bool = False
    resolution: Optional[str] = None

    def acknowledge(self, resolution: Optional[str] = None) -> None:
        """Mark this alert as acknowledged with an optional resolution note."""
        self.acknowledged = True
        if resolution:
            self.resolution = resolution

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "priority_score": round(self.priority_score, 4),
            "metric_name": self.metric_name,
            "affected_groups": self.affected_groups,
            "drift_score": round(self.drift_score, 6),
            "mean_shift": round(self.mean_shift, 6),
            "routing": self.routing,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "resolution": self.resolution,
        }

    def __repr__(self) -> str:
        return (
            f"AlertPayload(id={self.alert_id[:8]}..., severity={self.severity}, "
            f"metric={self.metric_name}, score={self.priority_score:.2f}, "
            f"acknowledged={self.acknowledged})"
        )


class AdaptiveThresholdManager:
    """Self-adjusting alert thresholds that learn from operator feedback.

    Each threshold is keyed by a string identifier of the form
    ``"{metric}_{group}"``, e.g. ``"demographic_parity_gender"``.  The
    threshold represents the minimum drift score required to fire an alert;
    values returned by :class:`~drift.FairnessDriftDetector` above this
    threshold will trigger the prioritizer.

    Learning rule (matching the Unit 2 case study):
    - False-positive rate > 30% of recent alerts → raise threshold slightly
      (reduce sensitivity, prevent fatigue).
    - False-positive rate <  5% of recent alerts → lower threshold slightly
      (increase sensitivity, catch more real drift).

    Parameters
    ----------
    initial_threshold : float, default 0.7
        Starting threshold for all new keys.
    learning_rate : float, default 0.01
        Fractional step applied on each feedback call.
    min_threshold : float, default 0.1
        Floor to prevent the threshold from becoming trivially small.
    max_threshold : float, default 0.95
        Ceiling to prevent the threshold from becoming unreachably high.
    history_window : int, default 50
        Number of recent feedback entries considered when updating a key.
    false_positive_upper : float, default 0.30
        If false-positive rate exceeds this, threshold is raised.
    false_positive_lower : float, default 0.05
        If false-positive rate is below this, threshold is lowered.

    Examples
    --------
    >>> from vfairness.operations.monitoring import AdaptiveThresholdManager
    >>> mgr = AdaptiveThresholdManager(initial_threshold=0.7)
    >>> mgr.get_threshold("demographic_parity_gender")
    0.7
    >>> mgr.update_from_feedback("demographic_parity_gender", alert_was_valid=False)
    >>> mgr.get_threshold("demographic_parity_gender") > 0.7
    True

    STAMP CORRECTION, 2026-09-17, read this BEFORE the generated block below.
    The defect that block describes is FIXED here: ``is_alert_warranted`` returns ``None`` for a drift score it could not
    compare, and ``set_threshold`` refuses a non-finite threshold instead of
    clamping it to the 0.95 ceiling. The block still reads
    DEFECT OPEN because it is GENERATED from ``src/vfairness/_proof_status.py``,
    which ``tests/test_beta_go_live_proof_ledger.py`` checks line by line
    against the frozen census ``docs/beta-go-live-census-2026-09-11.json``, and
    ``scripts/stamp_proof_status.py --check`` compares the block itself byte for
    byte. Hand-editing either is the exact move those gates exist to refuse, so
    the ledger row closes when the census is re-run, not before.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: adaptive_threshold_management. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        initial_threshold: float = 0.7,
        learning_rate: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 0.95,
        history_window: int = 50,
        false_positive_upper: float = 0.30,
        false_positive_lower: float = 0.05,
    ) -> None:
        self.initial_threshold = initial_threshold
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.history_window = history_window
        self.false_positive_upper = false_positive_upper
        self.false_positive_lower = false_positive_lower

        self._thresholds: Dict[str, float] = {}
        self._alert_history: List[Dict[str, Any]] = []

    # Core threshold API

    def get_threshold(self, key: str) -> float:
        """Return the current threshold for *key*.

        If the key has never been seen before, the initial threshold is
        returned (and NOT stored, to avoid memory bloat from one-off keys).

        Parameters
        ----------
        key : str
            Metric-group identifier, e.g. ``"equalized_odds_race"``.
        """
        return self._thresholds.get(key, self.initial_threshold)

    def set_threshold(self, key: str, value: float) -> None:
        """Manually override the threshold for *key*.

        Parameters
        ----------
        key : str
        value : float
            Must be a FINITE number in [min_threshold, max_threshold]. A
            non-finite value is refused with ``ValueError``; it is never
            stored, and the key keeps whatever threshold it already had.

        Raises
        ------
        ValueError
            If *value* is NaN or infinite.

        Notes
        -----
        BGL S2b, 2026-09-17. The clamp below is ``max(lo, min(hi, value))``,
        and every comparison against NaN is False, so ``min(0.95, nan)``
        returned 0.95 and ``max(0.1, 0.95)`` returned 0.95: a threshold nobody
        specified was stored as the CEILING, silently, and was thereafter
        indistinguishable from a deliberate 0.95. Measured before this change::

            mgr.set_threshold("k", float("nan"))   # 0 warnings
            mgr.get_threshold("k")                 -> 0.95
            mgr.is_alert_warranted("k", 0.94)      -> False, 0 warnings

        so the gate hardened in :meth:`is_alert_warranted` (which refuses an
        unmeasurable SCORE) then returned a confident False for a genuinely
        measured 0.94 drift, because the THRESHOLD it compared against was the
        fabrication. The guard belongs on both operands, and this one raises
        rather than warning: a threshold is a setting, so an unusable one is a
        configuration error at any input, and keeping the old value silently
        would leave the caller believing the override took.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"set_threshold({key!r}): a threshold of {value!r} is not a finite "
                f"number, so it cannot be compared against a drift score. It was NOT "
                f"stored: the key keeps its previous threshold of "
                f"{self.get_threshold(key)}. Clamping it would have stored "
                f"{self.max_threshold} (max/min return the bound against NaN), "
                f"indistinguishable from a deliberate ceiling."
            )
        clamped = max(self.min_threshold, min(self.max_threshold, numeric))
        self._thresholds[key] = clamped

    def update_from_feedback(self, key: str, alert_was_valid: Optional[bool]) -> float:
        """Update the threshold for *key* based on human feedback.

        Parameters
        ----------
        key : str
            Metric-group identifier.
        alert_was_valid : bool or None
            ``True`` if the alert reflected real fairness drift; ``False`` if it
            was a false positive; ``None`` if nobody has triaged it yet. Three
            states, and the third one moves nothing.

        Returns
        -------
        float
            The updated threshold value.

        Notes
        -----
        R-11, 2026-09-10. The false-positive rate was ``not e["valid"]``, and
        ``not None`` is ``True``, so every UNREVIEWED alert was counted as a
        CONFIRMED false positive. Measured from the 0.70 default::

            30 untriaged alerts -> false_positive_rate 1.0, threshold 0.9435
            a real 0.75 drift   -> is_alert_warranted() False

        An alert nobody has looked at is not evidence the detector is wrong; a
        backlog is not a verdict. Untriaged entries are recorded (so the backlog
        is visible in ``get_feedback_stats``) and excluded from both sides of
        the rate.
        """
        self._alert_history.append({"key": key, "valid": alert_was_valid})
        # Prune global history
        if len(self._alert_history) > 10_000:
            self._alert_history = self._alert_history[-10_000:]

        # Compute false-positive rate from recent history for this key
        recent = [e for e in self._alert_history if e["key"] == key][-self.history_window :]

        if not recent:
            return self.get_threshold(key)

        triaged = [e for e in recent if e["valid"] is not None]
        if not triaged:
            warnings.warn(
                f"update_from_feedback({key!r}): none of the {len(recent)} recent alert(s) "
                f"has been triaged, so no false-positive rate exists and the threshold is "
                f"UNCHANGED at {self.get_threshold(key)}. An untriaged alert is not a "
                f"false positive.",
                UserWarning,
                stacklevel=2,
            )
            return self.get_threshold(key)

        fp_rate = sum(1 for e in triaged if e["valid"] is False) / len(triaged)
        current = self._thresholds.get(key, self.initial_threshold)

        if fp_rate > self.false_positive_upper:
            # Too many false positives → raise threshold (less sensitive)
            new_val = min(current * (1 + self.learning_rate), self.max_threshold)
        elif fp_rate < self.false_positive_lower:
            # Very few false positives → lower threshold (more sensitive)
            new_val = max(current * (1 - self.learning_rate), self.min_threshold)
        else:
            new_val = current  # Within acceptable range; no adjustment

        self._thresholds[key] = new_val
        return new_val

    def reset(self, key: Optional[str] = None) -> None:
        """Reset thresholds to the initial value.

        Parameters
        ----------
        key : str, optional
            If given, reset only this key.  Otherwise, reset all.
        """
        if key is not None:
            self._thresholds.pop(key, None)
        else:
            self._thresholds.clear()

    def get_all_thresholds(self) -> Dict[str, float]:
        """Return a copy of all stored thresholds."""
        return dict(self._thresholds)

    def get_feedback_stats(self, key: str) -> Dict[str, Any]:
        """Return feedback statistics for *key*.

        Returns
        -------
        dict with keys: ``n_alerts``, ``n_valid``, ``n_false_positive``,
        ``n_untriaged``, ``false_positive_rate``, ``current_threshold``.

        ``n_false_positive`` counts only alerts a person CONFIRMED as false. It
        was ``n_alerts - n_valid``, which swept every untriaged alert into it;
        those are in ``n_untriaged`` now. ``false_positive_rate`` is over the
        triaged alerts only and is ``None`` when none has been triaged, never
        0.0, which is the reading for a detector with a perfect record.
        """
        history = [e for e in self._alert_history if e["key"] == key]
        n = len(history)
        n_valid = sum(1 for e in history if e["valid"] is True)
        n_fp = sum(1 for e in history if e["valid"] is False)
        n_untriaged = n - n_valid - n_fp
        n_triaged = n_valid + n_fp
        return {
            "n_alerts": n,
            "n_valid": n_valid,
            "n_false_positive": n_fp,
            "n_untriaged": n_untriaged,
            "false_positive_rate": (n_fp / n_triaged) if n_triaged else None,
            "current_threshold": self.get_threshold(key),
        }

    def is_alert_warranted(self, key: str, drift_score: Optional[float]) -> Optional[bool]:
        """Return whether *drift_score* exceeds the current threshold for *key*.

        Returns
        -------
        bool or None
            ``True`` if the score is at or above the threshold, ``False`` if it
            is genuinely below it, and ``None`` when the score could not be
            compared at all (``None`` or non-finite). ``None`` is COULD NOT
            CHECK: it is not an all-clear and it does not mean "no alert".

        Notes
        -----
        BGL S2, 2026-09-16. This gate returned a bare ``bool``. ``nan >= 0.7``
        is False, so a drift score that COULD NOT BE COMPUTED came back as
        ``False``, byte-identical to a measured calm reading, and an
        unmeasurable window was silently routed to "nobody is paged". Measured
        against the 0.70 default, with the NaN produced by
        ``FairnessDriftDetector.check_drift`` on an all-NaN current window::

            measured 0.95 -> True   (0 warnings)
            measured 0.10 -> False  (0 warnings)
            measured 0.00 -> False  (0 warnings)
            NaN           -> False  (0 warnings)   <- the defect

        This is verbatim the H-12 defect that
        :meth:`FairnessAlertPrioritizer._score_to_severity` in this same module
        was given a third state for; the gate in front of it was left with two.
        A caller that wants the old two-state reading must now say so, e.g.
        ``bool(result)`` after deciding what an unmeasurable score means to it.

        BGL g014, 2026-09-17. The guard above covered ONE operand. The
        THRESHOLD reaches here from ``initial_threshold``, which no code path
        validates: ``set_threshold`` refuses a non-finite value precisely
        because "the guard belongs on both operands", and the constructor
        bypasses ``set_threshold`` entirely. Measured on HEAD::

            mgr = AdaptiveThresholdManager(initial_threshold=float("nan"))
            mgr.is_alert_warranted("k", 0.94)  -> False, 0 warnings
            mgr.is_alert_warranted("k", 0.99)  -> False, 0 warnings

        because every comparison against NaN is False. So a genuinely measured
        0.99 drift, the loudest reading this gate can receive, was answered
        with the same confident False as a calm window, silently, for every key
        that manager ever grades. An unusable threshold is COULD NOT CHECK too.
        """
        threshold = _as_measured_number(self.get_threshold(key))
        if threshold is None:
            warnings.warn(
                f"is_alert_warranted({key!r}): the threshold for this key is "
                f"{self.get_threshold(key)!r}, which is not a finite number, so a "
                f"drift score of {drift_score!r} could not be compared against it. "
                f"Returning None (COULD NOT CHECK), NOT False: this is not an "
                f"all-clear. Every comparison against NaN is False, so the old answer "
                f"was 'no alert' for any score at all. Set a finite threshold with "
                f"set_threshold() or a finite initial_threshold.",
                UserWarning,
                stacklevel=2,
            )
            return None
        score = _as_measured_number(drift_score)
        if score is None:
            warnings.warn(
                f"is_alert_warranted({key!r}): drift_score is {drift_score!r}, so it "
                f"could not be compared against the threshold of "
                f"{threshold}. Returning None (COULD NOT CHECK), NOT "
                f"False: this is not an all-clear, and the alert has not been ruled "
                f"out. Grade the underlying window by hand.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return bool(score >= threshold)


class FairnessAlertPrioritizer:
    """Multi-factor alert prioritizer and routing engine.

    Converts a raw drift-event dictionary into a priority score, assigns a
    severity level, routes the alert to the appropriate notification channel,
    and produces a structured :class:`AlertPayload`.

    The default severity weights match the MegaShop case study in Unit 2.
    All weights are configurable at construction time to reflect
    domain-specific risk profiles.

    Parameters
    ----------
    severity_weights : dict, optional
        Override the default factor weights.  Supported keys:
        ``"regulatory_risk"``, ``"population_impact"``,
        ``"drift_velocity"``, ``"historical_discrimination"``.
    critical_score : float, default 8.0
        Priority score above which an alert is CRITICAL.
    high_score : float, default 5.0
        Priority score above which an alert is HIGH (below critical).
    routing_map : dict, optional
        Override the default channel map ``{severity: {channel, team}}``.
    log_alerts : bool, default True
        Whether to keep a log of all produced alerts.

    Examples
    --------
    >>> from vfairness.operations.monitoring import FairnessAlertPrioritizer
    >>> prioritizer = FairnessAlertPrioritizer()
    >>> event = {
    ...     "regulatory_risk": 1.0,
    ...     "population_impact": 0.8,
    ...     "drift_velocity": 0.9,
    ...     "historical_discrimination": 1.0,
    ...     "metric_name": "equalized_odds",
    ...     "affected_groups": ["Black", "Female"],
    ...     "drift_score": 0.82,
    ...     "mean_shift": 0.07,
    ... }
    >>> score, severity = prioritizer.calculate_priority(event)
    >>> severity
    'CRITICAL'
    >>> prioritizer.route_alert(severity)
    {'channel': 'pagerduty', 'team': '@on-call-ml-eng'}

    STAMP CORRECTION, 2026-09-17, read this BEFORE the generated block below.
    The defect that block describes is FIXED here: an unmeasurable severity factor is excluded and named, all four
    unmeasurable makes the score NaN and the severity UNSCORED,
    ``route_alert`` no longer sends an ungraded alert down the LOW route, and
    an unmeasurable drift-score boost is excluded rather than counted as zero,
    with the message saying the severity is a floor. The block still reads
    DEFECT OPEN because it is GENERATED from ``src/vfairness/_proof_status.py``,
    which ``tests/test_beta_go_live_proof_ledger.py`` checks line by line
    against the frozen census ``docs/beta-go-live-census-2026-09-11.json``, and
    ``scripts/stamp_proof_status.py --check`` compares the block itself byte for
    byte. Hand-editing either is the exact move those gates exist to refuse, so
    the ledger row closes when the census is re-run, not before.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: alert_prioritization. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "regulatory_risk": 3.5,
        "population_impact": 2.0,
        "drift_velocity": 2.5,
        "historical_discrimination": 3.0,
    }

    #: The severity label ``_score_to_severity`` returns when no score could be
    #: computed. It is a key of ``_DEFAULT_ROUTING`` for the reason below, and
    #: ``route_alert`` refuses rather than guessing when a custom routing map
    #: omits it.
    UNSCORED_SEVERITY = "UNSCORED"

    _DEFAULT_ROUTING: Dict[str, Dict[str, str]] = {
        "CRITICAL": {"channel": "pagerduty", "team": "@on-call-ml-eng"},
        "HIGH": {"channel": "slack", "team": "#fairness-alerts"},
        "LOW": {"channel": "jira", "project": "FAIR"},
        # R-11, 2026-09-10. The H-12 fix changed the LABEL and left the
        # ROUTING alone: `_score_to_severity` started returning "UNSCORED",
        # nothing here answered to that key, and `route_alert` fell through to
        # `routing_map["LOW"]`. Measured end to end on one drift event whose
        # only difference was an unmeasurable drift score::
        #
        #   drift_score=0.95 -> score 11.30, CRITICAL, {'channel': 'pagerduty'}
        #   drift_score=NaN  -> score nan,   UNSCORED, {'channel': 'jira'}
        #
        # which is verbatim the defect H-12's own docstring says it closed:
        # "one unmeasurable input silently downgraded a page-someone alert to a
        # ticket". Could-not-check gets its OWN destination here. Not jira,
        # which is where a MEASURED low-severity alert goes and is the calmest
        # thing this map can say; and not pagerduty either, which would claim
        # an urgency nobody measured. It goes where a person sees it the same
        # day, carrying what has to happen to it.
        UNSCORED_SEVERITY: {
            "channel": "slack",
            "team": "#fairness-alerts",
            "triage": (
                "severity COULD NOT be computed for this alert; score it by hand "
                "before closing it, and do not read it as low severity"
            ),
        },
    }

    def __init__(
        self,
        severity_weights: Optional[Dict[str, float]] = None,
        critical_score: float = 8.0,
        high_score: float = 5.0,
        routing_map: Optional[Dict[str, Dict[str, str]]] = None,
        log_alerts: bool = True,
    ) -> None:
        self.severity_weights = {**self._DEFAULT_WEIGHTS, **(severity_weights or {})}
        self.critical_score = critical_score
        self.high_score = high_score
        self.routing_map = routing_map or dict(self._DEFAULT_ROUTING)
        self.log_alerts = log_alerts
        self._alert_log: List[AlertPayload] = []

    # Priority scoring

    def calculate_priority(self, drift_event: Dict[str, Any]) -> Tuple[float, str]:
        """Compute a weighted priority score and assign a severity label.

        Expected drift_event keys (all values normalised to [0, 1]):

        - ``regulatory_risk``: 1.0 = legally protected class in jurisdiction.
        - ``population_impact``: Fraction of users affected.
        - ``drift_velocity``: Rate of change (rapid = 1.0, slow = 0.1).
        - ``historical_discrimination``: 1.0 = documented historical pattern.

        Additional factors that boost the score:

        - ``drift_score``: Raw KS drift score (added directly).
        - ``intersectional``: Boolean: adds 0.5 if the group is an
          intersection of multiple attributes.

        Parameters
        ----------
        drift_event : dict
            Key-value pairs describing the drift event context.

        Returns
        -------
        (priority_score, severity) : tuple[float, str]
        """
        # H-12. A factor that is PRESENT and unmeasurable (NaN) poisons the
        # whole sum, and the severity mapping below then returns the calmest
        # band for it, so NaN factors are excluded and warned about.
        #
        # R-8, 2026-09-09. `.get(factor, 0.0)` treated an ABSENT factor as a
        # zero contribution, silently. Measured: all four factors gave
        # (11.17, CRITICAL, pagerduty); regulatory_risk ABSENT gave
        # (7.67, HIGH, slack) with no warning, while regulatory_risk NaN gave
        # the same (7.67, HIGH) WITH a warning. An absent factor is the same
        # absence as a NaN one (nobody measured it), so it takes the same
        # exclusion-and-warning path. The four weighted factors are the
        # EXPECTED keys of a drift event; the optional boosts below
        # (drift_score, intersectional) keep their documented absent-means-
        # no-boost reading. A factor present as None is treated as absent.
        score = 0.0
        unmeasurable_factors = []
        for factor, weight in self.severity_weights.items():
            value = drift_event.get(factor)
            if value is None or not math.isfinite(value):
                unmeasurable_factors.append(factor)
                continue
            score += value * weight
        # BGL S2, 2026-09-16. R-8 above excludes an unmeasurable factor from the
        # sum, which is right while SOME factor was measured. When EVERY factor
        # is unmeasurable the exclusion leaves ``score`` at its 0.0 initialiser,
        # and 0.0 is finite, so ``_score_to_severity`` never reaches the
        # UNSCORED band this class built for exactly this case. Measured:
        #
        #   calculate_priority({})  -> (0.0, 'LOW'), routed {'channel': 'jira'}
        #
        # a sum over an empty set reported as the calmest measured band. The
        # empty sum is not a low priority; it is no priority at all, so the
        # score becomes NaN and takes the UNSCORED route. A drift event whose
        # four factors are all genuinely 0.0 still scores 0.0 and still grades
        # LOW: the guard counts EXCLUSIONS, never the value of the sum.
        if unmeasurable_factors:
            all_excluded = len(unmeasurable_factors) == len(self.severity_weights)
            if all_excluded:
                warnings.warn(
                    f"Alert factor(s) {unmeasurable_factors} could not be measured, "
                    "which is ALL of the weighted severity factors, so there is no "
                    "priority score to compute. The score is NaN and the severity is "
                    f"{self.UNSCORED_SEVERITY!r}: this alert has NOT been graded "
                    "harmless.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan"), self.UNSCORED_SEVERITY
            warnings.warn(
                f"Alert factor(s) {unmeasurable_factors} could not be measured and "
                "were excluded from the priority score; the score is computed from "
                "the remaining factors.",
                UserWarning,
                stacklevel=2,
            )

        # Additional boosts.
        #
        # BGL S2b, 2026-09-17. THE SAME ABSENT-IS-ZERO SHAPE, three lines below
        # the guard above, and it made one payload contradict itself: R-8
        # deliberately scoped the boosts out, but ``create_alert`` then decided
        # that an absent drift_score is COULD NOT CHECK and prints "Drift
        # score: NOT MEASURED" in the very sentence that carries this score.
        # Measured (regulatory_risk 1.0, drift_velocity 0.8,
        # historical_persistence 0.7, affected_population 0.0):
        #
        #   drift ABSENT       -> 7.60 HIGH slack, drift_score=nan,
        #                         "Drift score: NOT MEASURED ... score: 7.60"
        #   drift MEASURED 0.0 -> 7.60 HIGH slack   (byte-identical)
        #   drift MEASURED 0.9 -> 8.50 CRITICAL pagerduty
        #
        # so an unmeasured drift magnitude scored exactly as a measured calm
        # one and held the alert one band below CRITICAL, silently.
        #
        # The score does NOT become NaN: the four weighted factors above WERE
        # measured, and discarding them would be the reverse defect. The boost
        # is excluded, the exclusion is named, and the warning says the thing
        # that actually matters to a reader: a boost can only ADD, so a score
        # missing one is a FLOOR, and the severity with it.
        #
        # ``.get(key, default)`` does NOT fire when the key is PRESENT holding
        # None, which is why this reads the value first: create_alert(
        # {..., "drift_score": None}) used to die on
        # "unsupported operand type(s) for +=: 'float' and 'NoneType'" before
        # the three states in its own docstring could be applied.
        #
        # BGL g014, 2026-09-17. The reading below was
        # ``isinstance(raw_drift, (int, float)) and math.isfinite(raw_drift)``,
        # which is TRUE for a Python float and for ``np.float64`` (a float
        # subclass) and FALSE for ``np.float32``, ``np.float16``, ``np.int64``,
        # ``np.bool_`` and a 0-d array. A measured drift score arriving in any
        # of those dtypes fell into the branch below and was announced as never
        # measured. See ``_as_measured_number`` for the measured before/after.
        unmeasured_boosts = []
        raw_drift = drift_event.get("drift_score")
        measured_drift = _as_measured_number(raw_drift)
        if measured_drift is not None:
            score += measured_drift  # Raw drift magnitude
        elif "drift_score" in drift_event:
            # BGL S2 rework, 2026-09-17. A drift_score that is PRESENT and
            # unmeasurable makes the whole alert UNSCORED, not merely a boost
            # that is missing.
            #
            # The four weighted factors are CONTEXT: regulatory_risk,
            # population_impact, drift_velocity, historical_discrimination. They
            # say how bad a drift WOULD be, not that one happened. drift_score is
            # the only field carrying whether it did. Treating it as one boost
            # among several graded an alert CRITICAL out of context alone:
            # measured, drift_score=NaN with regulatory_risk=1.0 returned
            # (10.35, 'CRITICAL') and paged, for a metric whose drift was never
            # computed. Every unmeasurable metric would have paged as critical,
            # which is the alert fatigue that gets real criticals ignored.
            #
            # An ABSENT drift_score is different and keeps the old meaning: the
            # caller never offered one, so this is a context-only priority.
            warnings.warn(
                f"drift_score is {raw_drift!r}, so the drift this alert is ABOUT "
                f"was never measured. The severity is "
                f"{self.UNSCORED_SEVERITY!r}, not a band derived from the context "
                f"factors alone: those say how bad a drift would be, not that one "
                f"happened. This alert has NOT been graded harmless.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), self.UNSCORED_SEVERITY
        else:
            unmeasured_boosts.append("drift_score")

        # ``intersectional`` is a FLAG, and a flag absent from a schema that
        # only ever sets it to True genuinely reads as "not flagged", so an
        # absent key keeps its documented no-boost meaning. A key PRESENT
        # holding None is different: something tried to say and could not.
        if "intersectional" in drift_event and drift_event["intersectional"] is None:
            unmeasured_boosts.append("intersectional")
        elif drift_event.get("intersectional", False):
            score += 0.5  # Intersectional groups deserve extra attention

        if unmeasured_boosts:
            warnings.warn(
                f"Alert boost(s) {unmeasured_boosts} could not be measured, so they "
                f"were EXCLUDED from the priority score rather than counted as zero. "
                f"Both boosts can only ADD, so {score:.2f} is a FLOOR and the severity "
                f"that follows from it is the LOWEST this event can warrant, not a "
                f"grade: a measured drift_score would have raised it. An absent "
                f"drift_score is the same absence create_alert records as NaN and "
                f"prints as 'Drift score: NOT MEASURED'.",
                UserWarning,
                stacklevel=2,
            )

        severity = self._score_to_severity(score)
        return score, severity

    def route_alert(self, severity: str) -> Dict[str, str]:
        """Map a severity label to a notification-channel configuration.

        Parameters
        ----------
        severity : str
            ``"CRITICAL"``, ``"HIGH"``, ``"LOW"`` or ``"UNSCORED"``.

        Returns
        -------
        dict
            Routing instructions with at least a ``"channel"`` key.

        Notes
        -----
        A label this map does not know is NOT routed as ``"LOW"``. That was the
        fallback, and it is how ``"UNSCORED"``, a label this class itself
        returns, reached ``{'channel': 'jira'}``. Sending an alert nobody
        could grade to the destination reserved for alerts graded harmless is
        the calmest possible answer to a question that was never answered.
        """
        routing = self.routing_map.get(severity)
        if routing is not None:
            return dict(routing)

        unscored = self.routing_map.get(self.UNSCORED_SEVERITY)
        if unscored is not None:
            warnings.warn(
                f"route_alert: severity {severity!r} is not in this routing map, so it "
                f"was routed to the UNSCORED destination for hand triage, NOT to the "
                f"LOW one.",
                UserWarning,
                stacklevel=2,
            )
            return dict(unscored)

        # A caller-supplied routing_map with no UNSCORED entry. Refusing is the
        # only honest answer left: any key still in the map is a MEASURED
        # severity band and routing here to one of them asserts a grade.
        warnings.warn(
            f"route_alert: severity {severity!r} is not in this routing map and the map "
            f"has no {self.UNSCORED_SEVERITY!r} entry either, so this alert has NO "
            f"route. Returning channel='unrouted'; it is NOT being sent anywhere and it "
            f"has NOT been graded harmless. Add an "
            f"{self.UNSCORED_SEVERITY!r} entry to routing_map.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "channel": "unrouted",
            "triage": (
                f"no routing rule matched severity {severity!r}; this alert was "
                f"delivered nowhere and is not cleared"
            ),
        }

    # Full alert payload

    def create_alert(
        self,
        drift_event: Dict[str, Any],
        metric_name: Optional[str] = None,
        affected_groups: Optional[List[str]] = None,
        drift_score: Optional[float] = None,
        mean_shift: Optional[float] = None,
    ) -> AlertPayload:
        """Build a full :class:`AlertPayload` from a drift event.

        Parameters
        ----------
        drift_event : dict
            Raw drift context (same as passed to :meth:`calculate_priority`).
        metric_name : str, optional
            Override ``drift_event.get("metric_name")``.
        affected_groups : list[str], optional
            Override ``drift_event.get("affected_groups", [])``.
        drift_score : float, optional
            When None (default), read from ``drift_event["drift_score"]``.
            An explicit 0.0 is honored as a real zero, never discarded. When
            neither source supplies it, the payload records ``float("nan")``
            and a ``UserWarning`` is raised: absent is not zero.
        mean_shift : float, optional
            When None (default), read from ``drift_event["mean_shift"]``.
            Same three states as ``drift_score``.

        Returns
        -------
        AlertPayload
        """
        # The drift_score OVERRIDE has to reach the priority calculation, or the
        # two disagree about what was supplied. `create_alert(event,
        # drift_score=0.0)` on an event carrying no drift_score left
        # calculate_priority seeing an ABSENT score, so it warned that the boost
        # was excluded while the alert itself printed a measured "Drift score:
        # 0.000". A measured zero is a fact, and nothing should warn about it.
        #
        # BGL3 operations-4, 2026-09-27. `and "drift_score" not in drift_event`
        # let the EVENT's entry win over an explicit argument, and the argument
        # is documented as an override, so the payload disagreed with the score
        # in both directions. Measured on the full-context event (regulatory_risk
        # 1.0, population_impact 0.6, drift_velocity 0.8,
        # historical_discrimination 1.0):
        #
        #   event {'drift_score': None}, drift_score=0.9
        #     -> severity UNSCORED, priority nan, payload drift_score 0.9,
        #        message "Drift score: 0.900 ... severity COULD NOT be computed",
        #        warning "drift_score is None, so the drift this alert is ABOUT
        #        was never measured" -- about a value the caller had just
        #        supplied
        #   event {'drift_score': 0.7}, drift_score=0.0
        #     -> priority 10.60 CRITICAL pagerduty from the 0.7, printed as
        #        "Drift score: 0.000"
        #
        # A supplied override is a measurement offered by the caller, so it is
        # the one number both halves use. Omitting the argument still reads the
        # event, unchanged.
        priority_event = drift_event
        if drift_score is not None:
            priority_event = {**drift_event, "drift_score": drift_score}
        score, severity = self.calculate_priority(priority_event)
        routing = self.route_alert(severity)

        m_name = metric_name or drift_event.get("metric_name", "unknown_metric")
        groups = affected_groups or drift_event.get("affected_groups", [])
        # 'is None' checks, not truthiness: 0.0 is a legitimate drift score
        # / mean shift and an 'or' fallback would silently discard it.
        #
        # BGL S2, 2026-09-16. An ABSENT drift score became 0.0 and was stamped
        # into payload.drift_score, to_dict() and the message as
        # "Drift score: 0.000", with NO warning, on alerts up to and including
        # CRITICAL. Measured on a drift event with all four factors present and
        # no drift_score key::
        #
        #   severity='CRITICAL', priority_score=10.35, drift_score=0.0,
        #   message "... Drift score: 0.000, mean shift: +0.0000 ...", 0 warnings
        #
        # 0.000 is the reading for a metric that did not move at all, printed
        # for a metric nobody read. Absent becomes NaN and says so, in the
        # payload AND in the sentence a person reads off the channel. An
        # explicit 0.0, from either the argument or the event dict, is still a
        # real zero and is untouched.
        unmeasured_fields = []
        d_score = drift_score if drift_score is not None else drift_event.get("drift_score")
        if d_score is None:
            d_score = float("nan")
            unmeasured_fields.append("drift_score")
        m_shift = mean_shift if mean_shift is not None else drift_event.get("mean_shift")
        if m_shift is None:
            m_shift = float("nan")
            unmeasured_fields.append("mean_shift")
        if unmeasured_fields:
            warnings.warn(
                f"create_alert: {unmeasured_fields} were not supplied by the caller or "
                f"the drift event, so they are recorded as NaN (COULD NOT CHECK), not "
                f"as 0.0. A 0.0 here reads as a metric that did not move; this one was "
                f"never read.",
                UserWarning,
                stacklevel=2,
            )

        def _fmt(value: float, spec: str, label: str) -> str:
            return f"{label}: {value:{spec}}" if math.isfinite(value) else f"{label}: NOT MEASURED"

        group_str = ", ".join(str(g) for g in groups) if groups else "unknown groups"
        score_str = f"{score:.2f}" if math.isfinite(score) else "NOT MEASURED"
        message = (
            f"[{severity}] Fairness drift detected in '{m_name}' "
            f"for group(s): {group_str}. "
            f"{_fmt(float(d_score), '.3f', 'Drift score')}, "
            f"{_fmt(float(m_shift), '+.4f', 'mean shift')}. "
            f"Priority score: {score_str}. "
            f"Route: {routing.get('channel', 'unknown')} → {routing.get('team', routing.get('project', '?'))}."
        )
        # BGL S2b, 2026-09-17. calculate_priority now EXCLUDES an unmeasurable
        # drift boost instead of adding 0.0 for it, and a boost can only add,
        # so the severity that comes back is a FLOOR. Saying it here is the
        # point: this sentence is what a person reads off the channel, and
        # "[HIGH] ... Drift score: NOT MEASURED ... Priority score: 7.60" reads
        # as a graded HIGH unless the floor is spelled out. Byte-identical
        # scores for an ABSENT and a measured-0.0 drift score is what made this
        # worth a sentence rather than a field.
        if not math.isfinite(float(d_score)):
            message += (
                " The drift magnitude was NOT measured, so it contributed nothing to"
                " the priority score: the severity above is the LOWEST this event can"
                " warrant, not a grade. A measured drift score can only raise it."
            )
        if severity == self.UNSCORED_SEVERITY:
            # The message is what a person actually reads off the channel, so
            # the could-not-check has to be a sentence there and not only a
            # label and a nan.
            message += (
                " This alert's severity COULD NOT be computed, so it is neither"
                " confirmed critical nor cleared as low: score it by hand."
            )

        payload = AlertPayload(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            priority_score=score,
            metric_name=m_name,
            affected_groups=list(groups),
            drift_score=float(d_score),
            mean_shift=float(m_shift),
            routing=routing,
            drift_event=dict(drift_event),
            message=message,
        )

        if self.log_alerts:
            self._alert_log.append(payload)

        return payload

    # Alert log

    def get_alert_log(
        self,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
    ) -> List[AlertPayload]:
        """Return logged alerts with optional filters.

        Parameters
        ----------
        severity : str, optional
            Filter to ``"CRITICAL"``, ``"HIGH"``, ``"LOW"`` or ``"UNSCORED"``.
            An alert whose severity could not be computed is kept under
            ``"UNSCORED"`` and is therefore absent from all three measured
            bands: a caller sweeping only CRITICAL/HIGH/LOW will not see it.
        acknowledged : bool, optional
            Filter by acknowledgement status.

        Notes
        -----
        BGL g014, 2026-09-17. With ``log_alerts=False`` nothing is retained, so
        this answers ``[]`` for a prioritizer that has just paged a dozen
        CRITICALs, byte-identical to one that has produced nothing.
        ``ReportingStore.ingest_from_prioritizer`` reports that as "0 alerts
        ingested". Empty is now warned about when retention is the reason, so
        the two readings are distinguishable without reading this source.
        """
        self._warn_if_retention_is_off("get_alert_log")
        log = list(self._alert_log)
        if severity is not None:
            log = [a for a in log if a.severity == severity.upper()]
        if acknowledged is not None:
            log = [a for a in log if a.acknowledged == acknowledged]
        return log

    def _warn_if_retention_is_off(self, caller: str) -> None:
        """Warn when an empty log is empty because nothing is being kept.

        BGL g014, 2026-09-17. ``log_alerts=False`` is a retention switch, not a
        statement about the world, but both readers of ``_alert_log`` answered
        as if it were one: ``get_alert_log()`` -> ``[]`` and
        ``get_alert_summary()`` -> ``{'total_alerts': 0, 'by_severity': {},
        'acknowledged': 0, 'unacknowledged': 0}``, byte-identical to a
        prioritizer that graded every event and found nothing. The counts are
        true of the log either way, so they stay; what was missing is that the
        log is not a record of the alerts.
        """
        if not self.log_alerts and not self._alert_log:
            warnings.warn(
                f"{caller}: this prioritizer was built with log_alerts=False, so no "
                f"alert is retained and the empty result is NOT evidence that no "
                f"alert was raised. It is a count of what is kept, which is nothing. "
                f"Build the prioritizer with log_alerts=True to count alerts.",
                UserWarning,
                stacklevel=3,
            )

    def get_alert_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics for the alert log.

        Every severity the prioritizer can assign is counted under its own key
        in ``by_severity``, ``UNSCORED`` included, so an alert nobody could
        grade is never folded into a measured band. See
        :meth:`_warn_if_retention_is_off` for what an all-zero summary means
        when ``log_alerts=False``.
        """
        self._warn_if_retention_is_off("get_alert_summary")
        total = len(self._alert_log)
        by_severity: Dict[str, int] = {}
        n_acked = 0
        for alert in self._alert_log:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            if alert.acknowledged:
                n_acked += 1

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "acknowledged": n_acked,
            "unacknowledged": total - n_acked,
        }

    def clear_log(self) -> None:
        """Clear the alert log."""
        self._alert_log = []

    # Convenience: generate simulation drift events

    #: The context factors this helper can carry. A factor the caller does not
    #: supply is LEFT OUT of the built event, never stood in for: the values
    #: that used to be substituted here (regulatory_risk 0.5,
    #: population_impact 0.3, drift_velocity 0.3, historical_discrimination
    #: 0.5) carried an alert over the HIGH band on their own. See
    #: ``build_drift_event`` for the measurement.
    _CONTEXT_FACTORS: Tuple[str, ...] = (
        "regulatory_risk",
        "population_impact",
        "drift_velocity",
        "historical_discrimination",
    )

    @staticmethod
    def build_drift_event(
        metric_name: str,
        affected_groups: List[str],
        drift_score: float,
        mean_shift: float,
        regulatory_risk: Optional[float] = None,
        population_impact: Optional[float] = None,
        drift_velocity: Optional[float] = None,
        historical_discrimination: Optional[float] = None,
        intersectional: bool = False,
    ) -> Dict[str, Any]:
        """Helper: construct a standardised drift-event dictionary.

        All factor values should be normalised to [0, 1].

        Parameters
        ----------
        metric_name : str
        affected_groups : list[str]
        drift_score : float
            KS-based drift score.
        mean_shift : float
            Signed change in the metric mean.
        regulatory_risk : float, optional
            OMITTED from the returned event when not supplied, and named in a
            ``UserWarning``. See the note below: it used to be substituted with
            0.5, and a substitute is not a measurement of this event.
        population_impact : float, optional
            Omitted when not supplied (it was substituted with 0.3).
        drift_velocity : float, optional
            Omitted when not supplied (it was substituted with 0.3).
        historical_discrimination : float, optional
            Omitted when not supplied (it was substituted with 0.5).
        intersectional : bool, default False
            A flag, not a measurement: absent genuinely reads as "not flagged",
            the same reading :meth:`calculate_priority` gives it.

        Returns
        -------
        dict

        Warns
        -----
        UserWarning
            Naming every context factor that was not supplied and is therefore
            absent from the returned event.

        Notes
        -----
        BGL g014, 2026-09-17, disclosed only. BGL5 A-operations-1, 2026-09-27,
        DEFECT CLOSED. The four context factors are facts about the WORLD (is
        this attribute protected in this jurisdiction, does this group carry a
        documented history of discrimination), and a caller who does not know
        them leaves them out. :meth:`calculate_priority` was hardened twice, in
        R-8 and BGL S2, so that an ABSENT factor is excluded from the score and
        named rather than counted as zero, and so that an event with none of
        them measured scores NaN and grades ``UNSCORED``. This helper defeated
        both by making the absence PRESENT. Measured BEFORE, the same unmeasured
        context by two routes::

            calculate_priority({"drift_score": 0.5, "mean_shift": 0.1})
                -> (nan, 'UNSCORED')   routed to hand triage, 1 warning
            calculate_priority(build_drift_event("dp", ["B"], 0.5, 0.1))
                -> (5.1, 'HIGH')       routed to slack, 0 warnings

        and ``create_alert`` on that built event produced severity HIGH,
        priority 5.10, route ``{'channel': 'slack', 'team': '#fairness-alerts'}``
        and a message reading "[HIGH] ... Priority score: 5.10", with nothing in
        it about the four numbers nobody measured. 5.1 clears the 5.0 HIGH band
        on the substitutes alone: 0.5*3.5 + 0.3*2.0 + 0.3*2.5 + 0.5*3.0 = 4.6,
        plus the 0.5 drift boost.

        AFTER, same call: the event carries ``metric_name``,
        ``affected_groups``, ``drift_score``, ``mean_shift`` and
        ``intersectional`` only, the four unsupplied factors are ABSENT, and
        both routes agree::

            calculate_priority(build_drift_event("dp", ["B"], 0.5, 0.1))
                -> (nan, 'UNSCORED')   routed to hand triage
            create_alert(that event)
                -> severity 'UNSCORED', priority nan, message
                   "[UNSCORED] Fairness drift detected ...", routed to
                   {'channel': 'slack', 'team': '#fairness-alerts',
                    'triage': 'severity COULD NOT be computed for this alert;
                    score it by hand before closing it, and do not read it as
                    low severity'}

        A fully supplied event is untouched, which is the over-correction
        control: ``build_drift_event("equalized_odds", ["Black", "Female"],
        0.82, 0.07, regulatory_risk=1.0, population_impact=0.8,
        drift_velocity=0.9, historical_discrimination=1.0,
        intersectional=True)`` warns about nothing and scores (11.67,
        'CRITICAL'), and a PARTIALLY supplied one still scores on what was
        measured: the same call with regulatory_risk=1.0 and
        historical_discrimination=1.0 only gives (7.0, 'HIGH') and names the
        other two. A supplied 0.0 is a measurement and is kept, never read as
        "unsupplied": all four at 0.0 scores (0.5, 'LOW') with no warning.
        """
        supplied = {
            "regulatory_risk": regulatory_risk,
            "population_impact": population_impact,
            "drift_velocity": drift_velocity,
            "historical_discrimination": historical_discrimination,
        }
        # ``is None``, never falsiness: 0.0 is a real answer to "is this group
        # historically discriminated against" and must not be swept into the
        # unsupplied set (pinned in tests/test_surface_grade_g014.py).
        unsupplied = [name for name, value in supplied.items() if value is None]
        if unsupplied:
            warnings.warn(
                f"build_drift_event({metric_name!r}): context factor(s) {unsupplied} "
                f"were not supplied, so they are ABSENT from the returned event rather "
                f"than standing in at a substituted value. calculate_priority excludes "
                f"an absent factor from the score and names it, and an event with none "
                f"of the four measured scores NaN and grades 'UNSCORED', which is hand "
                f"triage rather than a severity band. Until 2026-09-27 this helper "
                f"substituted them (0.5, 0.3, 0.3, 0.5), which scored 5.1 and graded "
                f"HIGH on the substitutes alone. Supply the ones you know.",
                UserWarning,
                stacklevel=2,
            )
        event = {
            "metric_name": metric_name,
            "affected_groups": affected_groups,
            "drift_score": drift_score,
            "mean_shift": mean_shift,
            "intersectional": intersectional,
        }
        # Only the factors the caller actually supplied. An omitted key is the
        # absence calculate_priority already knows how to refuse; a substituted
        # value is one it cannot tell from a measurement.
        for name in FairnessAlertPrioritizer._CONTEXT_FACTORS:
            if supplied[name] is not None:
                event[name] = supplied[name]
        return event

    # Private helpers

    def _score_to_severity(self, score: float) -> str:
        """Map a priority score to a severity band.

        H-12. `>` is False for NaN, so both tests fell through and a NaN score
        returned "LOW", the calmest band. Measured 2026-09-07: with every factor
        known the score was 11.70 -> CRITICAL routed to PagerDuty; with a single
        factor NaN it became nan -> LOW routed to Jira. One unmeasurable input
        silently downgraded a page-someone alert to a ticket.

        An unscorable alert is not a low-severity alert. It gets its own band so
        a router can decide what to do with it rather than treating it as calm.

        R-11, 2026-09-10: the band alone was not enough. ``_DEFAULT_ROUTING``
        had no entry for this word for a day, and ``route_alert`` sent it to the
        LOW destination, so the downgrade H-12 describes carried on happening
        under a different label. The routing entry is now part of this fix, not
        a separate one.
        """
        if not math.isfinite(score):
            warnings.warn(
                f"Alert priority score is {score!r}, so no severity could be "
                f"computed. Returning {self.UNSCORED_SEVERITY!r} rather than 'LOW': an "
                f"alert that could not be scored is not a low-severity alert.",
                UserWarning,
                stacklevel=2,
            )
            return self.UNSCORED_SEVERITY
        if score > self.critical_score:
            return "CRITICAL"
        if score > self.high_score:
            return "HIGH"
        return "LOW"
