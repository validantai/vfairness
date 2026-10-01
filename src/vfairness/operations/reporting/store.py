"""
Unit 3: Performance Dashboards: MetricsStore
==============================================

Unified data-aggregation and query layer for the fairness monitoring pipeline.

The ``MetricsStore`` acts as the **single source of truth** that powers every
dashboard view, automated report, and interactive exploration tool in Unit 3.
It is fueled by the five monitoring components from Units 1 and 2:

- ``FairnessMonitor`` sliding-window snapshots (Unit 1)
- ``TemporalFairnessAnalyzer`` daily time-series (Unit 1)
- ``FairnessDriftDetector`` multiscale drift results (Unit 2)
- ``FairnessAlertPrioritizer`` alert payloads (Unit 2)
- ``AdaptiveThresholdManager`` threshold evolution (Unit 2)

Privacy is a first-class concern.  Query results automatically apply a
three-tier privacy scheme:

- **Suppressed** (k-anonymity): Groups smaller than ``k_anonymity_threshold``
  have their metric values replaced with ``NaN``.
- **Noisy** (differential privacy): Groups between *k* and
  ``noisy_threshold`` receive calibrated Laplace noise (ε-DP).
- **Exact**: Groups above ``noisy_threshold`` return exact values.

The noise is drawn ONCE per record and is stable across queries. This is what
makes the ε claim true: repeatedly reading the same record returns the identical
noisy value, so the answers are post-processing of a single release and nothing
is averaged away. Until 2026-08-27 the noise was redrawn from the unseeded
global RNG on EVERY query, and 3000 reads of one record recovered its true value
to within a standard error (0.4370 against a true 0.42), so the guarantee held
for exactly one query and the module said otherwise. The noisy value is also
clipped back into ``[0, 1]``, which is post-processing and so preserves the
guarantee; before, 1813 of 3000 draws fell outside the range the module
documents, which made the noisy tier trivially identifiable.

**Scope of the guarantee, precisely.** Stability comes from a per-store secret
seed. Within one ``MetricsStore`` the mapping from record to noise is fixed, so
ε holds for that record no matter how many times it is queried. A NEW store
instance (a process restart, a store rebuilt from persistence) draws a new seed
and therefore constitutes an INDEPENDENT release, whose privacy loss composes.
Set :attr:`MetricsStoreConfig.dp_seed` to a persisted value if the guarantee
must hold across processes.

**Utility warning.** The noisy tier is private but close to uninformative at
the default ε, because the sensitivity is deliberately left at the worst case.
See :attr:`MetricsStoreConfig.dp_epsilon` for the numbers and for why the
sensitivity is not tuned.

References
----------
Dwork et al. (2006). Calibrating noise to sensitivity in private data
  analysis. Theory of Cryptography Conference (TCC).
Sweeney (2002). k-Anonymity: A model for protecting privacy.
  International Journal of Uncertainty, Fuzziness and Knowledge-Based
  Systems 10(5).
"""

from __future__ import annotations

import hashlib
import secrets
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Enums & Config


class PrivacyLevel(Enum):
    """Privacy protection level applied to a group's metric values."""

    SUPPRESSED = "suppressed"  # group_size < k_threshold
    NOISY = "noisy"  # k_threshold <= group_size < noisy_threshold
    EXACT = "exact"  # group_size >= noisy_threshold
    #: The group size was never recorded, so k-anonymity COULD NOT BE CHECKED.
    #: The value is withheld exactly as SUPPRESSED is, and kept a separate
    #: level because the two mean different things to a reader: one says the
    #: group is too small to release, the other says nobody knows. Added
    #: 2026-09-10 (READINESS-5), when an unrecorded size was falling through to
    #: EXACT and releasing values under a guarantee nothing had verified.
    UNKNOWN_SIZE = "unknown_size"


@dataclass
class MetricsStoreConfig:
    """Configuration for :class:`MetricsStore`.

    Parameters
    ----------
    k_anonymity_threshold : int
        Groups with fewer members are suppressed entirely (default 10).
    noisy_threshold : int
        Groups between *k* and this value receive Laplace noise (default 50).
    dp_epsilon : float
        Privacy budget for Laplace noise (default 1.0).  Smaller ε = more
        noise = more privacy.

        **Read this before trusting a noisy value.** The noise is calibrated
        with ``sensitivity = 1.0``, the worst case for a metric bounded in
        [0, 1], so the Laplace scale is ``1.0 / ε``.  At the default ε = 1.0
        that scale is 1.0: roughly 60 percent of draws exceed 0.5 in magnitude,
        and a mid-range value clips to 0.0 or 1.0 more often than not.  **The
        noisy tier is private but close to uninformative at these defaults.**
        Treat a ``privacy_level == "noisy"`` row as withheld unless you have
        raised ε knowingly and accepted the weaker privacy that buys.

        The sensitivity is deliberately NOT tuned per statistic.  A tighter,
        correct per-statistic value would recover most of the utility, but a
        wrong one silently WEAKENS a formal guarantee in a way no test can
        catch, so it stays conservative until someone derives it properly.  An
        uninformative-but-true tier beats an informative-but-overclaimed one.
    dp_seed : int, optional
        Secret seed fixing the record-to-noise mapping.  Leave ``None`` and the
        store draws a fresh 64-bit secret at construction, which keeps answers
        stable for the life of that store.  Supply a persisted value when the
        guarantee must survive a process restart: without one, each new store
        instance is an independent release and the privacy loss composes across
        them.  It is a SECRET; anyone who knows it can subtract the noise.
    max_history_days : int
        Records older than this are pruned on :meth:`prune` (default 365).
    enable_privacy : bool
        Master switch: when ``False``, all queries return exact values.
    """

    k_anonymity_threshold: int = 10
    noisy_threshold: int = 50
    dp_epsilon: float = 1.0
    dp_seed: Optional[int] = None
    max_history_days: int = 365
    enable_privacy: bool = True


# Internal record types


@dataclass
class StoredMetricRecord:
    """A single metric observation.

    ``alert`` has three states. ``True``: the metric was compared to its
    threshold and breached it. ``False``: compared and clean. ``None``: the
    record carries NO alert determination, because nothing ever compared it
    to a threshold (a FairnessMonitor custom metric, a per-group positive
    rate, an MMD score). ``None`` is could-not-check: it is excluded from the
    health score's compliance mean and counted in ``n_not_assessable``, never
    averaged in as compliant. R-2, 2026-09-09.

    The DEFAULT is ``None``. A record built without an explicit determination
    has not got one. Until 2026-09-10 the default was ``False``, so every
    construction site that simply omitted the argument stamped "compared and
    clean" onto a comparison that never happened, and two of them shipped a
    whole monitoring history that way. READINESS-5.
    """

    timestamp: datetime
    source: str
    metric_name: str
    value: float
    group: str = "overall"
    group_size: int = 0
    batch_id: Optional[str] = None
    alert: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Whether ``group_size`` is the size of the population THIS record was
    #: computed over. ``False`` means a count was substituted from somewhere
    #: else (``ingest_window_metrics`` stores the WINDOW's sample_count beside a
    #: per-group rate, because WindowMetrics carries no per-group count), so the
    #: k-anonymity gate cannot check this record's own group and
    #: ``privacy_level`` must not claim it did. ``None`` is "not stated", which
    #: every pre-existing construction site means, and is treated exactly as
    #: before. BGL5 A-operations-3, 2026-09-27: the substitution existed and was
    #: disclosed only in an ingest-time warning, so the record a consumer reads
    #: carried a verified-looking 'exact' tier decided against a quantity nobody
    #: measured for that group.
    group_size_measured: Optional[bool] = None


# Per-group descriptive statistics that ``ingest_window_metrics`` stores beside
# the fairness metrics. The monitor never compares them to a threshold, so they
# carry no alert determination (``alert=None``) and are not compliance
# evidence: they enter neither the numerator nor the denominator of the health
# score's compliance mean, the same way ingested alert payloads are kept out of
# it. Before 2026-09-09 they entered it as ``alert=False``, i.e. as compliant
# metrics, and diluted every breach fraction. They are excluded without a
# warning because there is nothing an operator could do about a rate having no
# threshold; a NON-descriptive record with no determination (a custom metric
# the operator asked to track) is excluded WITH a warning naming it. R-2.
_DESCRIPTIVE_METRICS = frozenset({"positive_rate", "mmd_score"})


# Health Score


@dataclass
class HealthScore:
    """Composite Fairness Health Score (0–100).

    Aggregates the latest metric compliance, alert frequency, and drift
    stability into a single traffic-light indicator suitable for Tier-1
    executive dashboards.

    Attributes
    ----------
    score : float
        Overall score (0 = critical, 100 = fully healthy).
    status : str
        Traffic-light label: ``"green"`` (≥ 80), ``"yellow"`` (50–79),
        ``"red"`` (< 50).
    trend : str
        ``"improving"``, ``"stable"``, or ``"degrading"``.
    trend_slope : float
        Rate of change per day of the composite score.
    components : dict
        Breakdown by sub-score: *metric_compliance*, *alert_frequency*,
        *drift_stability*. A component with NO evidence behind it is ABSENT from
        this dict rather than present holding a number, and the composite is the
        documented 50/30/20 weights renormalised over the ones that are present:
        that is how a reader tells "measured and perfect" from "nobody measured
        it". Drift is absent when the window holds no drift result, alert
        frequency when the store has never received an alert record (a clean
        prioritizer run and an unconnected channel both leave zero of them).
    timestamp : datetime
        When the score was computed.
    explanation : str
        NLG sentence describing the score.
    n_metrics : int
        Number of metric records that contributed.
    n_alerts : int
        Number of alerts in the evaluation window.
    """

    score: Optional[float]
    status: str
    trend: str
    trend_slope: float
    components: Dict[str, float]
    timestamp: datetime
    explanation: str
    n_metrics: int = 0
    n_alerts: int = 0
    #: Metric records in the window whose value could not be computed (H-10) or
    #: which carry no alert determination at all (R-2: a custom metric, a group
    #: positive rate, an MMD score; nothing compared them to a threshold). They
    #: are excluded from metric compliance rather than counted as compliant.
    n_not_assessable: int = 0
    #: Metric records in the STORE whose timestamp cannot be read, so they have no
    #: position in any window and could not be placed inside or outside this one.
    #: They are NOT the same as ``n_not_assessable``: those were placed in the
    #: window and could not be graded, these could not be placed at all. A non-zero
    #: count here means ``metric_compliance`` had no establishable window fraction
    #: and the whole score is withheld. BGL7 B4-rep-pre-w2, 2026-09-29.
    n_undatable_metrics: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "status": self.status,
            "trend": self.trend,
            "trend_slope": self.trend_slope,
            "components": self.components,
            "timestamp": str(self.timestamp),
            "explanation": self.explanation,
            "n_metrics": self.n_metrics,
            "n_alerts": self.n_alerts,
            # R-2. A count that never left the object was not surfaced.
            "n_not_assessable": self.n_not_assessable,
            # BGL7 B4-rep-pre-w2. Same reason: a could-not-check that stays inside
            # the object is not visible where a reader looks.
            "n_undatable_metrics": self.n_undatable_metrics,
        }

    def __repr__(self) -> str:
        # C-04. A withheld score is None, and the old f-string formatted it with
        # a float spec, so merely REPRING a not-assessed HealthScore raised
        # TypeError. That is how a refusal turns into a crash in a debugger, a
        # log line or a test report.
        score = "None" if self.score is None else f"{self.score:.1f}"
        return f"HealthScore(score={score}, status='{self.status}', trend='{self.trend}')"


# Privacy helpers


def _group_size_or_unknown(raw: Any) -> int:
    """A recorded group size, or 0 meaning nobody recorded one.

    ``None`` and NaN are how an absent size reaches a frame, and both used to
    compare False against every threshold and fall through to EXACT.
    """
    try:
        if raw is None or pd.isna(raw):
            return 0
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _classify_privacy(
    group_size: int,
    k: int = 10,
    noisy: int = 50,
) -> PrivacyLevel:
    """The k-anonymity decision for ONE record's group size.

    READINESS-5, 2026-09-10. Two things were wrong here.

    It was DEAD. Zero call sites: ``_apply_privacy`` carried a second copy of
    the same rule as vectorised masks, so the repository held two statements of
    one privacy policy and only one of them ran. That is how the cali-BRATIO-n
    incident went (one call site fixed, the other left), and it is recorded in
    CLAUDE.md because a false PASS survived in production for weeks. This is
    now the single definition and ``_apply_privacy`` calls it.

    And it read an UNRECORDED size as a large one. ``group_size`` defaults to
    0 on ``StoredMetricRecord``, and most ingestion paths never set it, so
    ``group_size > 0 and group_size < k`` was False, the noisy band was False,
    and the record fell through to EXACT. The store therefore released exact
    values while asserting a k-anonymity guarantee it had never checked, which
    is the one thing a privacy control may not do. Measured that day:
    ``ingest_from_monitor`` produced 240 records, every one of them
    ``group_size=0`` and ``privacy_level='exact'``, for a named demographic
    group.

    Three states, never two: a size below k is SUPPRESSED because the group is
    too small, and a size nobody recorded is UNKNOWN_SIZE, which is withheld
    for a different reason and says so.
    """
    if group_size <= 0:
        return PrivacyLevel.UNKNOWN_SIZE
    if group_size < k:
        return PrivacyLevel.SUPPRESSED
    if k <= group_size < noisy:
        return PrivacyLevel.NOISY
    return PrivacyLevel.EXACT


# Metrics are documented as bounded in [0, 1]; the noisy value is clipped back
# into that range. Clipping is post-processing of a differentially private
# release, so it cannot weaken the guarantee (Dwork & Roth 2014, Prop. 2.1).
_METRIC_RANGE = (0.0, 1.0)


def _record_noise(
    *,
    dp_seed: int,
    timestamp: Any,
    source: Any,
    metric_name: Any,
    group: Any,
    group_size: Any,
    value: float,
    scale: float,
) -> float:
    """The Laplace draw for ONE record: same record, same seed, same noise.

    Derived from a keyed hash rather than a live RNG, so it is stable across
    queries, across store re-sorting, and across pruning of other records,
    with no cache to grow or invalidate. Redrawing per query is what let 3000
    reads average the noise away and recover the true value.

    The store's seed is the HMAC key, so the record-to-noise mapping is secret:
    an attacker who knows the record's identity still cannot compute its noise.
    """
    identity = "|".join(repr(x) for x in (timestamp, source, metric_name, group, group_size, value))
    digest = hashlib.blake2b(
        identity.encode("utf-8"),
        key=int(dp_seed).to_bytes(8, "big", signed=False),
        digest_size=16,
    ).digest()
    # A uniform in (0, 1) from the digest, then the inverse-CDF of Laplace(0,
    # scale). Deterministic given (seed, record), and distributionally the same
    # draw np.random.laplace would have made.
    raw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    u = min(max(raw, np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0)) - 0.5
    return float(-scale * np.sign(u) * np.log1p(-2.0 * abs(u)))


def _apply_privacy(
    df: pd.DataFrame,
    config: MetricsStoreConfig,
    dp_seed: int,
) -> pd.DataFrame:
    """Apply k-anonymity suppression and differential-privacy noise.

    The decision for one record lives in :func:`_classify_privacy` and this
    function applies it to a frame. READINESS-5, 2026-09-10: the two used to be
    separate statements of the same policy, and only this one ran, so the rule
    could be corrected in one place and stay wrong in production. Now there is
    one rule with one caller.
    """
    if not config.enable_privacy or "group_size" not in df.columns:
        return df

    out = df.copy()
    k = config.k_anonymity_threshold
    noisy_thresh = config.noisy_threshold
    eps = config.dp_epsilon

    levels = out["group_size"].apply(
        lambda size: _classify_privacy(_group_size_or_unknown(size), k, noisy_thresh)
    )

    # A group size nobody recorded cannot be checked against k, so the value is
    # withheld rather than released under a guarantee nothing verified. It is
    # kept distinct from SUPPRESSED so a reader can tell "too small to release"
    # from "nobody knows how many people this came from", and warned about,
    # because the fix is on the ingestion side and silence hides it.
    unknown_mask = levels == PrivacyLevel.UNKNOWN_SIZE
    if unknown_mask.any():
        affected = sorted({str(g) for g in out.loc[unknown_mask, "group"]})[:6]
        warnings.warn(
            f"{int(unknown_mask.sum())} of {len(out)} record(s) carry no recorded group "
            f"size, so k-anonymity could not be checked for them and their values are "
            f"WITHHELD (privacy_level='{PrivacyLevel.UNKNOWN_SIZE.value}'), not released. "
            f"This is could-not-check, not a suppression for smallness and not a clean "
            f"release. Groups affected include {affected}. Record group_size when "
            f"ingesting, or set enable_privacy=False if these values are not "
            f"person-level.",
            UserWarning,
            stacklevel=3,
        )
        out.loc[unknown_mask, "value"] = np.nan
        out.loc[unknown_mask, "privacy_level"] = PrivacyLevel.UNKNOWN_SIZE.value

    # Suppress small groups
    suppress_mask = levels == PrivacyLevel.SUPPRESSED
    out.loc[suppress_mask, "value"] = np.nan
    out.loc[suppress_mask, "privacy_level"] = PrivacyLevel.SUPPRESSED.value

    # Add Laplace noise to medium groups
    noisy_mask = (levels == PrivacyLevel.NOISY) & out["value"].notna()
    if noisy_mask.any():
        sensitivity = 1.0  # bounded metrics in [0, 1]
        scale = sensitivity / eps
        # One deterministic draw per record, NOT a fresh batch per query.
        noisy_values = []
        for _, row in out.loc[noisy_mask].iterrows():
            noise = _record_noise(
                dp_seed=dp_seed,
                timestamp=row["timestamp"],
                source=row["source"],
                metric_name=row["metric_name"],
                group=row["group"],
                group_size=row["group_size"],
                value=float(row["value"]),
                scale=scale,
            )
            noisy_values.append(float(np.clip(float(row["value"]) + noise, *_METRIC_RANGE)))
        out.loc[noisy_mask, "value"] = noisy_values
        out.loc[noisy_mask, "privacy_level"] = PrivacyLevel.NOISY.value

    # BGL5 A-operations-3, 2026-09-27. A SIZE THAT BELONGS TO ANOTHER
    # POPULATION cannot verify k-anonymity for this row's group.
    # ``ingest_window_metrics`` stores the WINDOW's sample_count beside a
    # per-group rate (WindowMetrics carries no per-group count) and marks the
    # record ``group_size_measured=False``. Measured on a 500-row window with
    # group_rates {'gender': {'F': 0.31, 'M': 0.33}} and privacy enabled:
    #   before  get_metrics() ->
    #             positive_rate gender_F 0.31 group_size 500 privacy_level exact
    #             positive_rate gender_M 0.33 group_size 500 privacy_level exact
    #           with NO warning at query time and nothing in the row saying the
    #           size was the window's, so a consumer filtering on
    #           privacy_level == 'exact' read them as verified releases, and
    #           ingest_dataframe re-ingested them as measurements (it refuses
    #           only non-exact rows), laundering the caveat away for good.
    #   after   the same two rows read privacy_level 'unknown_size' with
    #           group_size_measured False, one warning at QUERY time, and
    #           ingest_dataframe now refuses them on the round trip.
    # The VALUE is deliberately left as it was released before: withholding
    # every per-group rate of a default (privacy-enabled) store would remove a
    # figure the monitoring surfaces render, and the value itself is not what
    # was fabricated. What was fabricated is the verification claim, and that is
    # what this corrects. Only EXACT and NOISY are relabelled: a window under k
    # means the group is under k too, so a SUPPRESSED row is suppressed for a
    # reason that does hold.
    if "group_size_measured" in out.columns:
        substituted = (out["group_size_measured"] == False) & levels.isin(  # noqa: E712
            [PrivacyLevel.EXACT, PrivacyLevel.NOISY]
        )
        if substituted.any():
            affected = sorted({str(g) for g in out.loc[substituted, "group"]})[:6]
            warnings.warn(
                f"{int(substituted.sum())} of {len(out)} record(s) carry a group_size "
                f"that was measured for a DIFFERENT population (the ingesting path "
                f"substituted it), so k-anonymity could not be verified for their own "
                f"group and their privacy_level is reported as "
                f"'{PrivacyLevel.UNKNOWN_SIZE.value}' rather than "
                f"'{PrivacyLevel.EXACT.value}'. The values are released as before; what "
                f"is withheld is the GUARANTEE, which nothing checked. Groups affected "
                f"include {affected}. Ingest through ingest_dataframe with a real "
                f"group_size_col to have these classified.",
                UserWarning,
                stacklevel=3,
            )
            out.loc[substituted, "privacy_level"] = PrivacyLevel.UNKNOWN_SIZE.value

    return out


class MetricsStore:
    """Unified data layer for fairness monitoring dashboards and reports.

    The store collects metric records from all five monitoring components and
    provides a single, consistent query interface with built-in privacy.

    Parameters
    ----------
    config : MetricsStoreConfig, optional
        Privacy and retention settings.  Defaults are sensible for most use.

    Examples
    --------
    >>> from vfairness.operations.reporting import MetricsStore
    >>> store = MetricsStore()
    >>> store.ingest_from_monitor(monitor)
    42
    >>> df = store.get_metrics(metrics=["demographic_parity"])
    >>> hs = store.compute_health_score()
    >>> print(hs.status)
    'green'

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

    Ledger row: metrics_store. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, config: Optional[MetricsStoreConfig] = None) -> None:
        self.config = config or MetricsStoreConfig()
        # Secret key fixing the record-to-noise mapping for this store. Drawn
        # once here, never per query. `secrets`, not `np.random`: it is a
        # privacy key, and anyone who can guess it can subtract the noise.
        self._dp_seed: int = (
            self.config.dp_seed if self.config.dp_seed is not None else secrets.randbits(64)
        )
        self._records: List[StoredMetricRecord] = []
        self._alert_records: List[Dict[str, Any]] = []
        self._drift_records: List[Dict[str, Any]] = []
        self._threshold_records: List[Dict[str, Any]] = []
        self._models: Dict[str, Dict[str, Any]] = {}

    # Ingestion: single snapshots

    def ingest_window_metrics(self, snap) -> int:
        """Ingest a single ``WindowMetrics`` snapshot from FairnessMonitor.

        Returns the number of new records created.

        Notes
        -----
        BGL3 operations-4, 2026-09-27. Two things this method cannot measure,
        and now says so rather than leaving a reader to assume it did.

        A per-group ``positive_rate`` is computed over ONE group's rows, and the
        snapshot reports no per-group row count, so the count stored beside it is
        the window's ``sample_count``. The k-anonymity gate in
        :meth:`get_metrics` then classifies that row against the size of the
        whole window. Measured on a 500-row window with ``group_rates={"gender":
        {"F": 0.31, "M": 0.33}}``::

            positive_rate gender_F  0.31  group_size 500  privacy_level exact
            positive_rate gender_M  0.33  group_size 500  privacy_level exact

        ``exact`` asserts the group cleared ``noisy_threshold``; nothing here
        checked the group at all. The fix belongs on ``WindowMetrics``, which
        would have to carry the per-group counts it already computes (the
        ``min_samples`` floor reads them), and an existing pin in
        ``tests/test_surface_grade_g003.py`` asserts every record carries the
        window count, so the value cannot move from this side alone. Until it
        does, the substitution is warned about whenever privacy is enabled,
        which is when the classification is what a caller relies on.

        ``snap.excluded_groups`` names the groups the ``min_samples`` floor kept
        OUT of every aggregate over a column, and it was dropped here entirely.
        Measured on a window that excluded ``gender=F`` (4 rows): the store held
        a NaN ``demographic_parity`` with no determination (honest), one
        ``positive_rate`` row for ``gender_M``, and nothing at all recording that
        a group had been left out, so a dashboard reading this store sees one bar
        and no absence.
        """
        count = 0
        ts = snap.timestamp

        # Fairness metrics
        for key, value in snap.metrics.items():
            metric_name, group = _split_monitor_key(key)
            self._records.append(
                StoredMetricRecord(
                    timestamp=ts,
                    source="FairnessMonitor",
                    metric_name=metric_name,
                    value=float(value),
                    group=group,
                    group_size=snap.sample_count,
                    batch_id=str(snap.batch_id),
                    # R-2. This was `.get(key, False)`. FairnessMonitor only
                    # writes an alerts entry for its four built-in metrics; a
                    # custom metric has no threshold and never gets one, and
                    # the False default turned that absence into a compliant
                    # record in the health score's mean. Measured 2026-09-09:
                    # built-ins both breaching scored 50.0; adding three
                    # maximally unfair custom metrics RAISED it to 71.4. No
                    # determination is None (could not check), never False.
                    alert=snap.alerts.get(key),
                    # The window IS the population this metric was computed
                    # over, so its sample_count is this record's own size.
                    group_size_measured=True,
                )
            )
            count += 1

        # Group positive rates. A rate is not a metric with a threshold: it
        # carries no alert determination, so alert is None, not the dataclass
        # default of False that used to dilute the breach fraction. R-2.
        for attr, groups in snap.group_rates.items():
            for group_name, rate in groups.items():
                self._records.append(
                    StoredMetricRecord(
                        timestamp=ts,
                        source="FairnessMonitor",
                        metric_name="positive_rate",
                        value=float(rate),
                        group=f"{attr}_{group_name}",
                        group_size=snap.sample_count,
                        batch_id=str(snap.batch_id),
                        alert=None,
                        # The count above is the WINDOW's, not this group's, so
                        # the k-anonymity gate cannot check this row's own group.
                        # Recorded on the record, so the caveat travels with it
                        # into get_metrics instead of living only in the
                        # ingest-time warning below. BGL5 A-operations-3.
                        group_size_measured=False,
                    )
                )
                count += 1

        # See this method's Notes: the size stored beside a per-group rate is the
        # WINDOW's, because the snapshot carries no per-group count, so the
        # privacy tier those rows are released under was decided against a
        # quantity nobody measured for them. Warned only when privacy is on,
        # which is when the tier is load-bearing AT INGEST.
        #
        # BGL5 A-operations-3-b, 2026-09-29. This gate is deliberately LEFT as it
        # is, and the disclosure for a privacy-disabled store was put where the
        # defect actually was: on the OUTPUT. A caller ingesting into a store with
        # enable_privacy=False has asked for no privacy processing and is told so by
        # its own configuration, which is the subject of the over-correction control
        # in tests/test_bgl3_operations_4.py ("a clean ingest must warn about nothing
        # at all"). What was wrong was that the RECORD a consumer reads still carried
        # a `privacy_level` of 'exact', the tier meaning "k-anonymity verified", with
        # no caveat at query time either, and ingest_dataframe then accepted it back
        # as a measurement. All three are corrected in get_metrics and stay correct
        # under both settings; see the row builder there for the measured numbers.
        if snap.group_rates and self.config.enable_privacy:
            n_group_rows = sum(len(groups) for groups in snap.group_rates.values())
            warnings.warn(
                f"ingest_window_metrics: {n_group_rows} per-group rate record(s) carry "
                f"the WINDOW's sample_count ({snap.sample_count}) as their group_size, "
                f"because WindowMetrics reports no per-group row count. The k-anonymity "
                f"gate therefore classifies them against the size of the whole window, "
                f"not of the group, so a 'privacy_level' of 'exact' on those rows is NOT "
                f"a verified k-anonymity result for the group. Treat per-group rates as "
                f"unclassified, or ingest them through ingest_dataframe with a real "
                f"group_size_col.",
                UserWarning,
                stacklevel=2,
            )

        # A group the min_samples floor kept out of every aggregate is an
        # absence, and an absence that reaches no record reads as a group that
        # was never there. The snapshot knows the names and the counts; nothing
        # here could publish them, so they are named at ingest time.
        excluded = getattr(snap, "excluded_groups", None) or {}
        if excluded:
            warnings.warn(
                f"ingest_window_metrics: this window kept group(s) OUT of every "
                f"aggregate over the column(s) they belong to: {excluded}. Those groups "
                f"have no record in this store and no per-group rate, so the rows it "
                f"does hold describe the remaining groups only. Any aggregate over an "
                f"affected column is a measurement of part of the population.",
                UserWarning,
                stacklevel=2,
            )

        # MMD scores. Same as positive rates: the monitor applies no threshold
        # to them, so there is no determination to record. R-2.
        for attr, score in snap.mmd_scores.items():
            self._records.append(
                StoredMetricRecord(
                    timestamp=ts,
                    source="FairnessMonitor",
                    metric_name="mmd_score",
                    value=float(score),
                    group=attr,
                    # READINESS-6: both sibling loops in this method record the
                    # window's sample count and this one did not, so every MMD
                    # record was withheld by the k-anonymity gate as
                    # 'unknown_size' and warned about on every query. The
                    # window measured it; there is no reason to drop it here.
                    group_size=snap.sample_count,
                    batch_id=str(snap.batch_id),
                    alert=None,
                    # An MMD score compares the whole window against the
                    # reference for one attribute, so the window's count is the
                    # size of the population it was computed over. READINESS-6
                    # deliberately restored that count here; it is not a
                    # substitution and is marked as measured.
                    group_size_measured=True,
                )
            )
            count += 1

        return count

    def ingest_drift_result(self, result) -> int:
        """Ingest a ``MultiscaleDriftResult`` from FairnessDriftDetector."""
        raw_ts = getattr(result, "timestamp", datetime.now())
        ts = pd.Timestamp(raw_ts).to_pydatetime() if raw_ts is not None else datetime.now()
        rec = {
            "timestamp": ts,
            "metric": result.metric,
            "overall_drift_score": result.overall_drift_score,
            "drift_detected": result.drift_detected,
            # BGL g003, 2026-09-17. This default was 0.0, which reads as "the
            # two distributions are identical". ``MultiscaleDriftResult.mmd_score``
            # is ``Optional[float]`` and is None when the MMD was never computed,
            # so a result object that does not carry the attribute at all has
            # not measured it either. None is could-not-check; 0.0 would be a
            # divergence nobody computed, published in the drift history frame.
            "mmd_score": getattr(result, "mmd_score", None),
        }
        if result.worst_scale:
            rec["worst_scale"] = result.worst_scale.scale
            rec["worst_mean_shift"] = result.worst_scale.mean_shift
        self._drift_records.append(rec)

        # Also store as a metric record for unified queries
        self._records.append(
            StoredMetricRecord(
                timestamp=ts,
                source="FairnessDriftDetector",
                metric_name="drift_score",
                value=result.overall_drift_score,
                alert=result.drift_detected,
            )
        )
        return 1

    def ingest_alert(self, alert) -> int:
        """Ingest an ``AlertPayload`` from FairnessAlertPrioritizer.

        An alert whose ``priority_score`` was never computed is stored with a
        value of NaN, not 0.0, and an alert whose ``severity`` was never
        assigned is recorded as ``"UNSCORED"``, the spelling
        :class:`~vfairness.operations.monitoring.alerts.AlertPayload` already
        uses for that state. BGL g003, 2026-09-17: both used to carry a neutral
        default. Measured on ``ingest_alert({"metric_name":
        "demographic_parity", "message": "who knows"})``, a dict-shaped alert
        the public method accepts and the suite already exercises: the store
        published ``metric_name='alert_priority', value=0.0,
        alert_determined=True, privacy_level='exact'`` through
        :meth:`get_metrics`, i.e. the LOWEST priority on a 0-to-10 scale
        presented as a released measurement, with the record's severity stamped
        ``'LOW'``. Nothing had scored that alert at all.
        """
        d = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        # Normalize timestamp to datetime
        received_at = datetime.now()
        # BGL5 A-operations-3-b, 2026-09-29. This was
        # `ts = d.get("timestamp", datetime.now())`, and `.get(key, default)` does
        # NOT fire for a key that is PRESENT holding a null. An alert dict built
        # from a pandas row whose timestamp cell is missing
        # (`pd.DataFrame([{..., "timestamp": pd.NaT}]).to_dict("records")[0]`, the
        # ordinary shape a caller reading alerts out of a frame produces) was
        # therefore stored with `timestamp = NaT`, returned 1, and warned about
        # nothing. Every windowed reader then dropped it in silence, because
        # `pd.Timestamp(NaT) >= start` and `<= end` are both False: measured on a
        # store holding two metric records and that one HIGH-severity,
        # priority-9.0 alert,
        #   get_alerts()                              -> 1 record
        #   get_alerts(now - 7d, now + 1d)            -> 0, warnings []
        #   compute_health_score() -> score 37.5, components
        #       {'metric_compliance': 0.0, 'alert_frequency': 100.0}
        # i.e. a spotless 30-percent alert component over a store holding an
        # unresolved HIGH alert. Identical with an explicit `"timestamp": None`.
        #
        # An ABSENT key is left exactly as it was: nothing was declared, the
        # store's own receipt time is a real fact about the record, and that
        # behaviour is pinned (tests/test_surface_grade_g003.py asserts no warning
        # for a dict-shaped alert with no timestamp). A key that IS present and
        # cannot be read is the different state: the caller meant to declare a
        # time, so the receipt time is not a substitute for it. The record keeps
        # `timestamp = None`, carries its own caveat in `timestamp_measured`, and
        # is disclosed by get_alerts and compute_health_score rather than
        # vanishing from them.
        declared_missing = "timestamp" not in d
        ts: Optional[datetime]
        if declared_missing:
            ts = received_at
        else:
            parsed = _alert_timestamp(d)
            if parsed is None:
                warnings.warn(
                    "ingest_alert: this alert declares a timestamp "
                    f"({d.get('timestamp')!r}) that cannot be read as a time, so the "
                    "record is stored with NO time (timestamp None, "
                    "timestamp_measured False) rather than with the ingest time. It "
                    "CANNOT be placed in any queried window: get_alerts excludes it "
                    "from a windowed result and says so, and compute_health_score "
                    "withholds the alert-frequency component instead of scoring a "
                    "window whose alert count could not be established. This is not "
                    "an alert that did not fire.",
                    UserWarning,
                    stacklevel=2,
                )
                ts = None
            else:
                ts = parsed.to_pydatetime()
        d["timestamp"] = ts
        # The caveat travels in the record a consumer reads, not only in the
        # warning above, the same rule as `group_size_measured` on
        # StoredMetricRecord. Only stamped for the undatable case, so a record
        # that has a time is byte-identical to what this method produced before.
        if ts is None:
            d["timestamp_measured"] = False
        self._alert_records.append(d)

        self._records.append(
            StoredMetricRecord(
                # The MIRROR is timestamped with the receipt time even when the
                # alert itself could not be dated, because a record with no time
                # would silently leave every windowed get_metrics frame too, which
                # is the same defect one channel over. `alert_time_measured` in the
                # metadata says the mirror's time is the store's receipt, not the
                # alert's own.
                timestamp=ts if ts is not None else received_at,
                source="FairnessAlertPrioritizer",
                metric_name="alert_priority",
                value=_alert_priority_value(d),
                alert=True,
                # `.get(key, default)` does NOT fire for a key that is present
                # holding None, so the fallback is spelled `or`, and an alert
                # the prioritizer itself could not score already arrives as
                # "UNSCORED".
                metadata=(
                    {"severity": d.get("severity") or "UNSCORED"}
                    if ts is not None
                    else {
                        "severity": d.get("severity") or "UNSCORED",
                        "alert_time_measured": False,
                    }
                ),
            )
        )
        return 1

    def ingest_threshold_update(
        self,
        key: str,
        threshold: float,
        feedback_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an adaptive threshold change."""
        self._threshold_records.append(
            {
                "timestamp": datetime.now(),
                "key": key,
                "threshold": threshold,
                "stats": feedback_stats or {},
            }
        )

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        source: str = "custom",
        timestamp_col: str = "timestamp",
        metric_col: str = "metric",
        value_col: str = "value",
        group_col: Optional[str] = None,
        alert_col: Optional[str] = None,
        group_size_col: Optional[str] = None,
        alert_determined_col: Optional[str] = None,
    ) -> int:
        """Generic ingestion from a tidy DataFrame.

        A tidy frame carries values, not verdicts. Unless *alert_col* names a
        column holding a threshold comparison the caller actually made, every
        record is stored with ``alert=None`` (no determination), which keeps it
        out of the health score's compliance mean instead of averaging it in as
        compliant.

        *group_size_col* names the column holding how many rows each reading
        came from. Without it the size is unrecorded, and with
        ``enable_privacy=True`` the k-anonymity gate cannot check it, so those
        values are WITHHELD rather than released. That is deliberate and it is
        warned about: releasing them was asserting a guarantee nothing had
        verified (READINESS-5, 2026-09-10). Pass this column, or leave privacy
        off if the values are not person-level.

        *alert_determined_col* names the column saying whether the alert value
        beside it is a DETERMINATION or merely absent. It defaults to the
        column this store's own :meth:`get_metrics` emits, so re-ingesting an
        exported frame is lossless without the caller having to know that.

        READINESS-5, 2026-09-10. This path stamped ``alert=False`` on every
        record, and ``False`` is read all the way down the reporting stack as
        "compared to its threshold and found clean". Measured: twelve rows of
        ``demographic_parity = 0.95`` and ``disparate_impact = 0.05``, about as
        unfair as those two metrics get, scored 100.0/100 GREEN with
        metric_compliance 100.0, n_not_assessable 0 and not one warning.

        READINESS-6, LATER THE SAME DAY. That fix was defeated by this store's
        OWN export. ``get_metrics`` emits ``alert`` as a strict bool, with the
        third state carried beside it in ``alert_determined``, and this method
        had no parameter that could receive that second column. So one
        ``to_dataframe`` / ``ingest_dataframe`` cycle turned every "nobody
        compared this" into "compared and clean". Measured: six readings of
        ``demographic_parity = 0.95``, none of them ever compared to a
        threshold, scored ``None / not_assessed / n_not_assessable=6`` before
        the round trip and ``100.0 / green / 0`` after it, with no warning.
        That is verbatim the scenario the paragraph above says was fixed.

        The same cycle also launders privacy: a value the k-anonymity gate
        suppressed or noised is re-ingested as a measurement, so a rows are
        re-exported as exact. Rows whose ``privacy_level`` says they were not
        released are therefore REFUSED here, counted and warned about, rather
        than silently promoted back to measurements.
        """
        # Default to this store's own export column, so a round trip of our own
        # output cannot lose the third state. READINESS-6.
        determined_col = alert_determined_col
        if determined_col is None and _ALERT_DETERMINED_COLUMN in df.columns:
            determined_col = _ALERT_DETERMINED_COLUMN

        # A row the privacy gate did not release is not a measurement, and
        # re-ingesting it would republish a suppressed or noised value as an
        # exact one. READINESS-6.
        # BGL7 B4-rep-pre-w2, 2026-09-29. THE SIBLING of the get_metrics row-builder
        # fix, in the same change, because that fix broke this guard's premise.
        #
        # The rule was `released = privacy_level == 'exact'`, and it states its own
        # reason in the warning below: re-ingesting would "republish a suppressed or
        # noised number as a measurement". Once the row builder started classifying
        # an UNRECORDED group size honestly (0 is UNKNOWN_SIZE, not EXACT), that
        # equality began refusing rows whose values the gate had RELEASED IN FULL:
        # a privacy-disabled store's own export. Two existing tests were right to go
        # red, and one of them is the over-correction control for the READINESS-6
        # round trip ("a round trip that lost real verdicts would be as broken as one
        # that invented them"): six datable readings with a recorded clean
        # determination scored 100.0 green before the trip and None / not_assessed
        # after it, because every row had been thrown away.
        #
        # The property this guard actually protects is that a round trip must not
        # publish a STRONGER privacy claim than the row it read. So that is what it
        # now tests, one condition per reason:
        #   * SUPPRESSED / NOISY: the value itself was withheld or perturbed.
        #     Re-ingesting it republishes a number the gate changed. Refused, as
        #     before.
        #   * a non-exact row whose VALUE is not a readable number: the gate withheld
        #     it (privacy-enabled UNKNOWN_SIZE rows carry NaN). Refused, as before.
        #   * a non-exact row whose OWN RECORDED SIZE would support a different tier:
        #     its caveat comes from something other than the size, so nothing in the
        #     ingested columns carries it and the clone would publish a
        #     verified-looking tier. This is the substituted per-group size
        #     (group_size 500 marked group_size_measured=False, tier UNKNOWN_SIZE, and
        #     this method has no parameter that can carry the flag), and it stays
        #     refused for exactly the reason BGL5 A-operations-3-b gave.
        # An UNKNOWN_SIZE row with NO recorded size and an intact value is ACCEPTED,
        # because its caveat IS the absent size and the absent size travels: the clone
        # publishes UNKNOWN_SIZE again. Nothing is laundered, and the alert
        # determination survives, which is what the READINESS-6 control asserts.
        #
        # `group_size_col` is deliberately NOT defaulted to the exported column the
        # way `alert_determined_col` is. Carrying the size forward WITHOUT
        # `group_size_measured` (for which there is no parameter) is what would turn
        # a substituted 500 into an 'exact' claim in the clone. Dropping it lands on
        # the WEAKER tier, which is the safe direction.
        withheld = 0
        if _PRIVACY_LEVEL_COLUMN in df.columns:
            level = df[_PRIVACY_LEVEL_COLUMN].astype(str)
            sizes = (
                df["group_size"].map(_group_size_or_unknown)
                if "group_size" in df.columns
                else pd.Series(0, index=df.index)
            )
            size_would_support = sizes.map(
                lambda s: (
                    _classify_privacy(
                        int(s),
                        self.config.k_anonymity_threshold,
                        self.config.noisy_threshold,
                    ).value
                )
            )
            value_readable = pd.to_numeric(df[value_col], errors="coerce").notna()
            perturbed = level.isin({PrivacyLevel.SUPPRESSED.value, PrivacyLevel.NOISY.value})
            refused = (level != PrivacyLevel.EXACT.value) & (
                perturbed | ~value_readable | (level != size_would_support)
            )
            released = ~refused
            withheld = int(refused.sum())
            if withheld:
                levels = sorted(set(df.loc[refused, _PRIVACY_LEVEL_COLUMN].astype(str)))
                warnings.warn(
                    f"ingest_dataframe: {withheld} of {len(df)} row(s) carry a "
                    f"privacy_level of {levels} whose caveat would NOT survive this "
                    f"ingest: either the k-anonymity gate did not release their values "
                    f"(suppressed, noised, or withheld as NaN), or the columns being "
                    f"ingested would classify them into a STRONGER tier than they carry. "
                    f"They are NOT ingested: re-ingesting them would republish a "
                    f"suppressed, noised or unverified number as a measurement. Ingest "
                    f"the pre-privacy frame (apply_privacy=False) if you need them.",
                    UserWarning,
                    stacklevel=2,
                )
                df = df[released]

        count = 0
        unreadable_determinations = 0
        # BGL7 B4-rep-pre-w2, 2026-09-29. THE SIBLING of the determination test
        # below: `timestamp=row[timestamp_col]` was stored with no readability test
        # at all, so a missing timestamp cell (pd.NaT) entered the store as a record
        # with no position in any window, and `compute_health_score` then graded a
        # window over whatever remained. The six-record measurement is recorded in
        # compute_health_score. The record is still STORED, because its value and
        # its alert determination are real and `compute_health_score` needs to see
        # it in order to refuse; what was missing was anyone saying so at the door.
        unreadable_timestamps = 0
        for _, row in df.iterrows():
            # BGL3 operations-4, 2026-09-27. The determination FLAG gets read
            # by the same rule as the flag it gates, because a frame spells an
            # absent bool as NaN and ``bool(float("nan"))`` is True. Measured
            # before this change, six readings of demographic_parity = 0.95
            # (about as unfair as that metric gets) that nobody had compared to
            # any threshold:
            #   alert_determined=False -> alert None -> health None
            #        /not_assessed, n_not_assessable=6, 2 warnings
            #   alert_determined=NaN   -> alert False -> health 100.0/green,
            #        metric_compliance 100.0, n_not_assessable=0, 0 warnings
            # byte-identical to the control where the comparison really ran and
            # came back clean. That is verbatim the READINESS-6 scenario this
            # method's docstring says was fixed, reached through an unrecorded
            # flag instead of a missing column. ``_alert_determination`` is the
            # one place that rule lives: None/NaN is could-not-check, and only
            # a recorded True lets the alert value through.
            determination_flag = (
                _alert_determination(row[determined_col])
                if (determined_col and determined_col in row)
                else None
            )
            if determined_col and determined_col in row and determination_flag is None:
                unreadable_determinations += 1
            if _readable_time_value(row[timestamp_col]) is None:
                unreadable_timestamps += 1
            self._records.append(
                StoredMetricRecord(
                    timestamp=row[timestamp_col],
                    source=source,
                    metric_name=row[metric_col],
                    value=float(row[value_col]),
                    group=str(row[group_col]) if group_col and group_col in row else "overall",
                    # Read exactly as ingest_from_monitor reads its history: a
                    # recorded determination survives the round trip, an absent
                    # one stays absent. Never invented.
                    #
                    # READINESS-6: when a determination column is present it
                    # DECIDES. Our own export flattens `alert` to a strict bool
                    # for the mask readers and carries the third state beside
                    # it, so reading `alert` alone turns every could-not-check
                    # into a clean comparison.
                    alert=(
                        (_alert_determination(row[alert_col]) if determination_flag else None)
                        if (
                            determined_col
                            and determined_col in row
                            and alert_col
                            and alert_col in row
                        )
                        else (
                            _alert_determination(row[alert_col])
                            if alert_col and alert_col in row
                            else None
                        )
                    ),
                    # Same rule as the determination above: recorded if the
                    # caller recorded it, absent otherwise, never invented.
                    group_size=(
                        _group_size_or_unknown(row[group_size_col])
                        if group_size_col and group_size_col in row
                        else 0
                    ),
                )
            )
            count += 1
        if unreadable_determinations:
            warnings.warn(
                f"ingest_dataframe: {unreadable_determinations} of {count} row(s) carry "
                f"no readable value in {determined_col!r} (None or NaN), so it is not "
                f"recorded whether they were ever compared to a threshold. Their alert "
                f"determination is stored as None (could not check), NOT as a clean "
                f"comparison, and they are excluded from the health score's compliance "
                f"mean and counted in n_not_assessable.",
                UserWarning,
                stacklevel=2,
            )
        if unreadable_timestamps:
            warnings.warn(
                f"ingest_dataframe: {unreadable_timestamps} of {count} row(s) carry no "
                f"readable value in {timestamp_col!r} (None, NaT or unparseable), so those "
                f"records have NO POSITION IN TIME. They cannot be placed inside or "
                f"outside any window, so they leave every windowed query silently "
                f"(get_metrics, get_alerts, get_drift_history) and compute_health_score "
                f"WITHHOLDS the whole score rather than grade a window over an incomplete "
                f"subset. This is could-not-check, not a record dated now. Give them a "
                f"readable timestamp to have them counted.",
                UserWarning,
                stacklevel=2,
            )
        return count

    # Bulk ingestion from component instances

    def ingest_from_monitor(self, monitor) -> int:
        """Bulk-ingest all history from a ``FairnessMonitor``."""
        history_df = monitor.get_metric_history()
        count = 0
        for _, row in history_df.iterrows():
            # READINESS-5: the same split ingest_window_metrics applies, so the
            # bulk route and the per-snapshot route agree on the metric's name
            # and its group. See _split_monitor_key.
            metric_name, group = _split_monitor_key(row["metric"])
            self._records.append(
                StoredMetricRecord(
                    timestamp=row["timestamp"],
                    source="FairnessMonitor",
                    metric_name=metric_name,
                    value=float(row["value"]),
                    group=group,
                    # READINESS-5: the window measured this and the history
                    # carries it now, so the k-anonymity gate has a real size
                    # to check instead of the 0 it used to read as "big
                    # enough". Frames from an older monitor have no such
                    # column, and 0 there stays UNRECORDED rather than
                    # becoming a size.
                    group_size=int(row["sample_count"]) if "sample_count" in row else 0,
                    batch_id=str(row["batch_id"]),
                    # R-2. `bool(row["alert"])` collapsed the history's None
                    # (never compared to a threshold) into False (compared and
                    # clean). The third state must survive the round trip.
                    alert=_alert_determination(row["alert"]),
                )
            )
            count += 1
        return count

    def ingest_from_analyzer(self, analyzer) -> int:
        """Bulk-ingest daily data from a ``TemporalFairnessAnalyzer``."""
        df = analyzer.to_dataframe()
        count = 0
        for _, row in df.iterrows():
            for col in df.columns:
                if col == "date":
                    continue
                self._records.append(
                    StoredMetricRecord(
                        timestamp=row["date"],
                        source="TemporalFairnessAnalyzer",
                        metric_name=col,
                        value=float(row[col]),
                        # READINESS-5, 2026-09-10. The analyzer's daily frame
                        # holds values only: it has no alert column and applies
                        # no threshold, so there is no determination to record.
                        # This relied on the old alert=False default, and a
                        # maximally unfair daily history (demographic_parity
                        # 0.95, disparate_impact 0.05, six days of each) scored
                        # 100.0/100 GREEN with no warning.
                        alert=None,
                    )
                )
                count += 1
        return count

    def ingest_from_detector(self, detector) -> int:
        """Bulk-ingest drift results from a ``FairnessDriftDetector``."""
        results = detector.get_results_history()
        for r in results:
            self.ingest_drift_result(r)
        return len(results)

    def ingest_from_prioritizer(self, prioritizer) -> int:
        """Bulk-ingest alerts from a ``FairnessAlertPrioritizer``."""
        alerts = prioritizer.get_alert_log()
        for a in alerts:
            self.ingest_alert(a)
        return len(alerts)

    # Model version registry

    def register_model(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Register a model version for cross-model comparison."""
        self._models[model_id] = {"registered_at": datetime.now(), **metadata}

    # Query API

    def window_now(self) -> datetime:
        """``now``, in the tz base this store's own timestamps use.

        The window bound every reader in the reporting package builds is
        ``now - time_window``, and a tz-naive bound cannot be compared against a
        tz-aware timestamp column at all. BGL grade-1 G08, 2026-09-30: a store
        ingested with ``datetime.now(timezone.utc)`` timestamps (what a
        timestamptz column and any ISO-8601 'Z' string give) answered
        ``get_metrics``, ``get_summary`` and ``get_alerts`` correctly and then
        raised "TypeError: Invalid comparison between dtype=datetime64[ns, UTC]
        and Timestamp" from ``compute_health_score``, which took every reader
        surface above it down: ``create_executive_view``,
        ``create_operational_view``, ``create_technical_view``,
        ``FairnessDashboard.to_html``, ``get_explanation``,
        ``InteractiveDashboard.to_standalone_html`` and every
        ``ReportGenerator`` tier.

        Use this in place of ``datetime.now()`` wherever the result becomes a
        window BOUND. It attaches the records' own tzinfo to the current instant
        and performs no offset arithmetic, so the window is identical; a naive
        store keeps the naive local ``datetime.now()`` it has always had.

        A CALLER-supplied bound is deliberately not adjusted (see
        :func:`_window_bound`): reconciling a caller's naive bound with an aware
        column needs a guess about which zone the caller meant, and pandas'
        refusal is the honest answer there.
        """
        return _now_like_records(self._records)

    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        apply_privacy: bool = True,
    ) -> pd.DataFrame:
        """Query the store with optional filters.

        Returns a tidy DataFrame with columns: *timestamp*, *source*,
        *metric_name*, *value*, *group*, *group_size*, *alert*,
        *alert_determined*, *privacy_level*, *group_size_measured*.

        ``group_size_measured`` is three-state and says whether *group_size* is
        the size of the population THAT row was computed over: ``True`` yes,
        ``False`` a count substituted from elsewhere (so *privacy_level* carries
        no verified k-anonymity statement about that group and reads
        ``unknown_size`` when privacy is enabled), ``None`` the producing path
        did not say.

        ``alert`` is a strict bool and answers one question only: did an
        alert FIRE for this record. ``alert_determined`` says whether the
        record was compared to a threshold at all. A row with
        ``alert_determined == False`` has ``alert == False`` because no alert
        can fire for a metric nobody compared, and it must NOT be read as
        compliant; ``compute_health_score`` excludes such rows from the
        compliance mean. Two columns rather than a tri-state ``alert`` so the
        existing mask readers (``df[df["alert"]]``) keep working. R-2.

        An unreadable *start_time* or *end_time* raises :class:`ValueError`. See
        :func:`_window_bound`: such a bound made this method return an empty frame
        in silence, which reads as "nothing in this window".
        """
        # BEFORE ANY WORK, and before the empty-store shortcut, so the refusal does
        # not depend on what the store happens to hold (BGL7 B4-rep-pre-w2).
        start_bound = _window_bound(start_time, method="get_metrics", argument="start_time")
        end_bound = _window_bound(end_time, method="get_metrics", argument="end_time")
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "source",
                    "metric_name",
                    "value",
                    "group",
                    "group_size",
                    "alert",
                    "alert_determined",
                    "privacy_level",
                    "group_size_measured",
                ]
            )

        rows = []
        for r in self._records:
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "source": r.source,
                    "metric_name": r.metric_name,
                    "value": r.value,
                    "group": r.group,
                    "group_size": r.group_size,
                    "alert": bool(r.alert),
                    "alert_determined": r.alert is not None,
                    # BGL5 A-operations-3-b, 2026-09-29. This was
                    # `PrivacyLevel.EXACT.value` unconditionally, and the ONLY
                    # place that corrected it for a substituted group size was
                    # inside `_apply_privacy`, which returns at its first line when
                    # `config.enable_privacy` is False. So a store configured with
                    # privacy off published the substituted rows as 'exact', the
                    # tier PrivacyLevel defines as "group_size >= noisy_threshold",
                    # with no warning at ingest OR query time, and
                    # `ingest_dataframe` (which refuses only non-exact rows) then
                    # re-ingested them as verified measurements. Measured on the
                    # identical 500-row window with group_rates {'gender': {'F':
                    # 0.31, 'M': 0.33}} into MetricsStore(MetricsStoreConfig(
                    # enable_privacy=False)):
                    #   before  privacy_level {'exact'} measured {False}, ingest
                    #           warns [], query warns [], and the round trip
                    #           ingested 2 of 2 with warnings [], the clone's rows
                    #           reading privacy_level exact, group_size 0,
                    #           group_size_measured None: the caveat laundered off.
                    #   after   privacy_level {'unknown_size'}, one query warning,
                    #           and the round trip refuses both rows.
                    # 'exact' is a claim about a k-anonymity check. A row whose size
                    # belongs to another population has not had that check, under
                    # any configuration, so the claim is not the privacy switch's to
                    # make. The VALUE is untouched: enable_privacy=False still
                    # releases every value exactly, which is what that switch means.
                    #
                    # BGL7 B4-rep-pre-w2, 2026-09-29. That fix handled
                    # `group_size_measured is False` and sent BOTH remaining states
                    # (True, and the THIRD one, None = "the producing path did not
                    # say") to `else: EXACT`, a claim that this record's group_size
                    # passed a k-anonymity check at the noisy threshold. The record
                    # type documents None as "not stated", so the tier was decided by
                    # the absence of a statement. Measured before this change, two
                    # rows ingested through the public ingest_dataframe with NO
                    # group_size_col into
                    # MetricsStore(MetricsStoreConfig(enable_privacy=False)):
                    #   privacy_level 'exact', group_size 0, group_size_measured None,
                    #   ingest warnings 0, query warnings 0, and the round trip
                    #   through ingest_dataframe ACCEPTED 2 of 2 (it refuses only
                    #   non-'exact' rows), laundering the caveat off for good exactly
                    #   as the block above describes.
                    # THREE STATES, and the third one is the SIZE, not the flag. A row
                    # with NO RECORDED SIZE has had no k-anonymity check under any
                    # configuration, which is verbatim what PrivacyLevel.UNKNOWN_SIZE
                    # is documented to mean ("The group size was never recorded, so
                    # k-anonymity COULD NOT BE CHECKED"), and it is the condition
                    # `_classify_privacy` has tested first since READINESS-5 for
                    # exactly this reason: an unrecorded size compares False against
                    # every threshold and fell through to EXACT.
                    #
                    # The whole classifier is deliberately NOT called here. It would
                    # stamp 'noisy' / 'suppressed' on rows whose values were never
                    # noised or suppressed, because `_apply_privacy` returns at its
                    # first line when enable_privacy is False, and those two tiers are
                    # claims about what was DONE to the value. This row builder makes
                    # only the claim it can support: a size was recorded and released,
                    # or no size was recorded so nothing was checked. When privacy IS
                    # enabled, `_apply_privacy` remains the one place that applies the
                    # full rule and overwrites the tier.
                    "privacy_level": (
                        PrivacyLevel.UNKNOWN_SIZE.value
                        if getattr(r, "group_size_measured", None) is False
                        or _group_size_or_unknown(r.group_size) <= 0
                        else PrivacyLevel.EXACT.value
                    ),
                    # BGL5 A-operations-3. Whether `group_size` is this record's
                    # OWN population count. False = substituted from elsewhere,
                    # so no k-anonymity statement about this group is possible;
                    # None = the producing path did not say. The caveat now
                    # travels in the row a consumer reads, not only in a warning
                    # at ingest time that no downstream reader ever receives.
                    "group_size_measured": (
                        None
                        if getattr(r, "group_size_measured", None) is None
                        else bool(r.group_size_measured)
                    ),
                }
            )

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # `is not None` rather than truthiness: an unreadable bound never reaches
        # here any more (it was refused above), and a readable one must be applied
        # whatever it is.
        if start_bound is not None:
            df = df[df["timestamp"] >= start_bound]
        if end_bound is not None:
            df = df[df["timestamp"] <= end_bound]
        if metrics:
            df = df[df["metric_name"].isin(metrics)]
        if groups:
            df = df[df["group"].isin(groups)]
        if sources:
            df = df[df["source"].isin(sources)]

        if apply_privacy and self.config.enable_privacy:
            df = _apply_privacy(df, self.config, self._dp_seed)
        elif not self.config.enable_privacy and "group_size_measured" in df.columns:
            # BGL5 A-operations-3-b, 2026-09-29. The query-time disclosure lived
            # only inside `_apply_privacy`, which this store never reaches, so a
            # reader of a privacy-disabled store got the substituted rows with no
            # caveat anywhere: not at ingest (that warning was gated on the same
            # switch) and not here. The tier is corrected in the row builder above;
            # this says so out loud, because a consumer that filters on
            # `privacy_level` needs to know the tier CHANGED for those rows rather
            # than discovering them missing from an 'exact' filter.
            #
            # Gated on the store's CONFIG and not on the `apply_privacy` argument:
            # `apply_privacy=False` is a caller explicitly asking for the
            # pre-privacy frame (compute_health_score does it on every call) and it
            # already knows; a store configured without privacy has no such moment.
            substituted_rows = df["group_size_measured"] == False  # noqa: E712
            if bool(substituted_rows.any()):
                affected = sorted({str(g) for g in df.loc[substituted_rows, "group"]})[:6]
                warnings.warn(
                    f"{int(substituted_rows.sum())} of {len(df)} record(s) carry a "
                    f"group_size that was measured for a DIFFERENT population (the "
                    f"ingesting path substituted it), so no k-anonymity statement about "
                    f"their own group was ever checked and their privacy_level reads "
                    f"'{PrivacyLevel.UNKNOWN_SIZE.value}' rather than "
                    f"'{PrivacyLevel.EXACT.value}'. This store has enable_privacy=False, "
                    f"so the VALUES are released exactly as configured; what is withheld "
                    f"is the guarantee, which nothing verified under any setting. Groups "
                    f"affected include {affected}. Ingest through ingest_dataframe with a "
                    f"real group_size_col to have these classified.",
                    UserWarning,
                    stacklevel=2,
                )
            # BGL7 B4-rep-pre-w2, 2026-09-29. THE SIBLING of the warning above, for
            # the OTHER way a row reaches UNKNOWN_SIZE: its size was never recorded
            # at all (the third state, group_size_measured None with group_size 0).
            # `_apply_privacy` has warned about exactly that since READINESS-5, and
            # a privacy-disabled store never reaches it, so those rows changed tier
            # here with nothing said. A consumer that filters on `privacy_level`
            # needs the same telling as the substituted rows get.
            unrecorded_rows = (df["privacy_level"] == PrivacyLevel.UNKNOWN_SIZE.value) & (
                df["group_size_measured"] != False  # noqa: E712
            )
            if bool(unrecorded_rows.any()):
                affected = sorted({str(g) for g in df.loc[unrecorded_rows, "group"]})[:6]
                warnings.warn(
                    f"{int(unrecorded_rows.sum())} of {len(df)} record(s) carry no "
                    f"recorded group size, so k-anonymity could not be checked for them "
                    f"and their privacy_level reads "
                    f"'{PrivacyLevel.UNKNOWN_SIZE.value}' rather than "
                    f"'{PrivacyLevel.EXACT.value}'. This is COULD NOT CHECK, not a "
                    f"verified release: the k-anonymity gate compares a RECORDED size, "
                    f"and these rows carry none. This store has "
                    f"enable_privacy=False, so the VALUES are released exactly as "
                    f"configured; what is withheld is the guarantee. Groups affected "
                    f"include {affected}. Ingest through ingest_dataframe with a real "
                    f"group_size_col to have these classified.",
                    UserWarning,
                    stacklevel=2,
                )

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query alert records.

        An empty list is "no alert matched this query", and it used to be
        returned in silence for a store that holds nothing at all, where it reads
        as "monitored, no alerts" (2026-09-25). The two are distinguishable: a
        store with no metric records has never been given anything to alert on.

        BGL5 A-operations-3, 2026-09-27: and a store whose ALERT CHANNEL was
        never fed cannot tell them apart either, which the 2026-09-25 guard
        missed because it asked whether ANYTHING had been ingested.

        BGL5 A-operations-3-b, 2026-09-29: there is a THIRD route to a silent
        ``[]``, and both earlier guards ask a store-level question that cannot
        see it. An alert record whose timestamp cannot be read has no position in
        time, so it satisfies neither window bound and leaves a windowed result
        whatever the window is. Such records are excluded from a windowed result
        (they are not evidence either way) and NAMED in a warning, with their
        severities, so a short result is not read as a measured count.

        BGL7 B4-rep-pre-w2, 2026-09-29: there is a FOURTH route, and all three
        guards above ask about the STORE or the RECORDS, so none of them could see
        a defect in the QUESTION. ``get_alerts(start_time=pd.NaT)`` returned ``[]``
        with zero warnings over a store holding a datable HIGH alert. An unreadable
        CALLER bound now raises :class:`ValueError`; see :func:`_window_bound`.
        """
        start_bound = _window_bound(start_time, method="get_alerts", argument="start_time")
        end_bound = _window_bound(end_time, method="get_alerts", argument="end_time")
        if not self._records and not self._alert_records:
            warnings.warn(
                "MetricsStore.get_alerts: this store holds no metric records and no "
                "alert records, so the empty result means nothing has been ingested. "
                "It is NOT a finding that no alert fired.",
                UserWarning,
                stacklevel=2,
            )
            return []
        if not self._alert_records:
            # THE GUARD ABOVE ASKED THE WRONG QUESTION. It fires only for a store
            # with nothing in it at all, so the common case stayed silent: a store
            # fed by FairnessMonitor but never by FairnessAlertPrioritizer returns
            # [] with no caveat, and compute_health_score then scored that empty
            # list as a perfect alert component. Measured on a store holding two
            # metric records that BOTH carry alert=True and no alert record:
            #   before  get_alerts() -> [], warnings []; compute_health_score()
            #           -> 37.5 with components alert_frequency 100.0
            #   after   get_alerts() -> [], one warning naming '2 of 2 metric
            #           record(s) ... carry alert=True', and the health score
            #           excludes the component (0.0, see compute_health_score).
            # A clean drift run leaves a row (drift_detected False) but a clean
            # alert run leaves nothing at all, which is what makes this absence
            # ambiguous where the drift absence is not.
            # Control: a store fed through ingest_alert whose window simply holds
            # no matching alert stays SILENT, because that [] is a measurement.
            n_records = len(self._records)
            flagged = sum(1 for r in self._records if _record_alert_flag(r) is True)
            flagged_note = (
                f" {flagged} of {n_records} metric record(s) in this store carry "
                "alert=True, so breaches WERE recorded on the metric side."
                if flagged
                else ""
            )
            warnings.warn(
                "MetricsStore.get_alerts: this store has never received an alert "
                f"record (it holds {n_records} metric record(s)), so the empty result "
                "means the alert channel was never fed. It is NOT a finding that no "
                "alert fired: ingest_alert has not been called on this store." + flagged_note,
                UserWarning,
                stacklevel=2,
            )
            return []
        recs = self._alert_records
        # BGL5 A-operations-3-b, 2026-09-29. The 2026-09-27 guard above asks a
        # STORE-level question (was this channel ever fed), and this is a WINDOW
        # query, so a fed alert that cannot be PLACED in the window was still
        # returned as a silent []. `pd.Timestamp(NaT) >= st` is False and
        # `<= et` is False as well, so an undatable record fell out of both
        # bounds with nothing said. Measured on a store holding two metric
        # records and one ingested HIGH-severity, priority-9.0 alert whose
        # timestamp was NaT:
        #   get_alerts()                    -> 1 record
        #   get_alerts(now - 7d, now + 1d)  -> 0 records, warnings []
        # and every consumer (reports.py:935/977/1063, dashboard.py:1006/1595,
        # store.py compute_health_score) reads that [] as "no alert matched".
        #
        # An undatable record is COULD NOT CHECK for a window: it is not evidence
        # that the window was quiet, and it is not evidence that it was not, so it
        # is excluded from the windowed result AND named. Without a window it is
        # still returned, because no window position is needed to return it.
        undatable = [r for r in recs if _alert_timestamp(r) is None]
        if undatable and (start_time is not None or end_time is not None):
            severities = sorted({str(r.get("severity") or "UNSCORED") for r in undatable})
            warnings.warn(
                f"MetricsStore.get_alerts: {len(undatable)} of {len(recs)} alert "
                "record(s) in this store carry no readable timestamp, so they COULD "
                "NOT be placed in the queried window and are excluded from this "
                f"result. Severities affected: {', '.join(severities)}. An empty or "
                "short result here is therefore not a finding that no alert fired in "
                "the window: that many alerts have no window position at all. Ingest "
                "alerts with a readable timestamp to have them counted.",
                UserWarning,
                stacklevel=2,
            )
        if start_time is not None or end_time is not None:
            # The window is applied over the records that HAVE a position in time,
            # and the undatable ones leave here rather than through a comparison
            # that silently answers False in both directions. No unreachable
            # fallback timestamp is invented for them: they are simply not placed.
            dated = [(ts, r) for r in recs if (ts := _alert_timestamp(r)) is not None]
            if start_bound is not None:
                dated = [(ts, r) for ts, r in dated if ts >= start_bound]
            if end_bound is not None:
                dated = [(ts, r) for ts, r in dated if ts <= end_bound]
            recs = [r for _, r in dated]
        if severity:
            recs = [r for r in recs if r.get("severity") == severity]
        return recs

    def get_drift_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get drift detection results as a DataFrame.

        An unreadable *start_time* or *end_time* raises :class:`ValueError`. The
        SIBLING of the same hole in :meth:`get_metrics` and :meth:`get_alerts`,
        fixed in the same change: all three compared against
        ``pd.Timestamp(bound)`` and an unreadable bound emptied the result in
        silence. See :func:`_window_bound`.
        """
        start_bound = _window_bound(start_time, method="get_drift_history", argument="start_time")
        end_bound = _window_bound(end_time, method="get_drift_history", argument="end_time")
        if not self._drift_records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "metric",
                    "overall_drift_score",
                    "drift_detected",
                ]
            )
        df = pd.DataFrame(self._drift_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start_bound is not None:
            df = df[df["timestamp"] >= start_bound]
        if end_bound is not None:
            df = df[df["timestamp"] <= end_bound]
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_threshold_history(self) -> pd.DataFrame:
        """Get adaptive threshold evolution."""
        if not self._threshold_records:
            return pd.DataFrame(columns=["timestamp", "key", "threshold"])
        return pd.DataFrame(self._threshold_records)

    # Convenience accessors

    def get_metric_names(self) -> List[str]:
        return sorted(set(r.metric_name for r in self._records))

    def get_groups(self) -> List[str]:
        return sorted(set(r.group for r in self._records if r.group != "overall"))

    def get_sources(self) -> List[str]:
        return sorted(set(r.source for r in self._records))

    def get_time_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        if not self._records:
            return (None, None)
        timestamps = [pd.Timestamp(r.timestamp) for r in self._records]
        return (min(timestamps).to_pydatetime(), max(timestamps).to_pydatetime())

    def get_latest(self, metric_name: str) -> Optional[StoredMetricRecord]:
        """Return the most recent record for a given metric, or None.

        This hands back the STORED record, so its ``value`` is the raw one and
        the three-tier privacy scheme in :meth:`get_metrics` is NOT applied to
        it. Measured 2026-09-17 (BGL g003) on a record with ``group_size=3``:
        ``get_metrics`` returned NaN with ``privacy_level='suppressed'`` and
        this returned 0.42. Use :meth:`get_metrics` for anything a reader sees;
        this is an internal accessor and is documented as one rather than
        quietly releasing a value the k-anonymity gate withholds.
        """
        matches = [r for r in self._records if r.metric_name == metric_name]
        return max(matches, key=lambda r: r.timestamp) if matches else None

    # Composite Health Score

    def compute_health_score(
        self,
        time_window: timedelta = timedelta(days=7),
    ) -> HealthScore:
        """Compute the composite Fairness Health Score (0–100).

        Three equally-important components:
        - **Metric compliance** (50%): ``100 * (1 - 2 * alert_fraction)``,
          floored at 0. Each alerting record is penalized at double weight,
          so the component hits 0 once half of the recent metric records
          alert (not the plain fraction without alerts).
        - **Alert frequency** (30%): Penalty for each alert in the window.
        - **Drift stability** (20%): Fraction of windows without drift,
          over the drift results that carry a verdict. A result the detector
          REFUSED to produce (``drift_detected is None``) is could-not-check:
          it is excluded and warned about, and when no result in the window
          carries a verdict the whole score is withheld rather than graded off
          a drift component nobody measured. READINESS-5.

        A metric record whose timestamp cannot be read has no position in any
        window, so it can be placed neither inside this one nor outside it and the
        breaching fraction is not establishable. The score is then WITHHELD
        (``score=None``, ``status='not_assessed'``), the count travels in
        ``n_undatable_metrics`` and a warning names how many of them recorded a
        breach. BGL7 B4-rep-pre-w2, 2026-09-29.
        """
        # The window bound is built in the SAME time base the records use. A
        # bare `datetime.now()` is tz-naive local time, and pandas refuses to
        # compare that against a tz-aware timestamp column, so BGL grade-1 G08
        # measured on 2026-09-30: a store ingested with
        # `datetime.now(timezone.utc)` timestamps (what a timestamptz column and
        # any ISO-8601 'Z' string give) answered get_metrics, get_summary and
        # get_alerts correctly and then raised
        # "TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and
        # Timestamp" out of this method, taking every consumer of it down with
        # it: create_executive_view, create_operational_view,
        # create_technical_view, FairnessDashboard.to_html,
        # InteractiveDashboard.to_standalone_html, get_explanation and every
        # ReportGenerator tier. It is the same INSTANT in both bases, so aligning
        # is not a guess about what the caller meant; a naive store keeps the
        # naive local `now` it always had.
        now = self.window_now()
        start = now - time_window

        # Get data
        df = self.get_metrics(start_time=start, end_time=now, apply_privacy=False)
        alerts = self.get_alerts(start_time=start, end_time=now)
        drift_df = self.get_drift_history(start_time=start, end_time=now)

        # WHETHER THE ALERT COMPONENT HAS EVIDENCE AT ALL. Read once here because
        # both the withheld-composite branch below and the graded path have to
        # agree about it; they disagreed before and the withheld branch published
        # the component the graded path excluded.
        #
        # BGL5 A-operations-3-b, 2026-09-29. `bool(self._alert_records)` is a
        # STORE-level fact and `alert_score` is a WINDOW measurement, so the
        # 2026-09-27 guard could not see the other way this component loses its
        # evidence: an alert record whose timestamp cannot be read has no position
        # in time, leaves every window through get_alerts, and left `n_alerts` at 0
        # on a channel that HAD been fed. Measured on one breaching record in two
        # plus one ingested HIGH-severity, priority-9.0 alert whose timestamp was
        # NaT (`pd.DataFrame([{..., "timestamp": pd.NaT}]).to_dict("records")[0]`):
        #   before  score 37.5, components {'metric_compliance': 0.0,
        #           'alert_frequency': 100.0}, no alert-coverage warning at all,
        #           over a store holding an unresolved HIGH alert. Byte-identical
        #           to the numbers the 2026-09-27 fix claims to have eliminated.
        #   after   score 0.0 (red), components {'metric_compliance': 0.0}, a
        #           warning naming the 1 undatable record, and the explanation
        #           saying the component was not measured.
        # The channel test is kept as well: the two absences have different causes
        # and different remedies, so they get different warnings, and either one
        # alone withholds the component. A quiet week on a connected channel whose
        # records are all datable is still a measured zero, which the
        # over-correction control asserts at 90.0 and 100.0.
        n_undatable_alerts = sum(1 for r in self._alert_records if _alert_timestamp(r) is None)
        alert_channel_fed = bool(self._alert_records)
        alert_measured = alert_channel_fed and n_undatable_alerts == 0

        # THE SAME DEFECT, in the 50 PERCENT COMPONENT.
        #
        # BGL7 B4-rep-pre-w2, 2026-09-29. The block above taught `alert_measured`
        # the difference between a STORE fact and a WINDOW fact, and left the
        # identical hole on `metric_compliance`, which is worth 50 percent to the
        # 30 the alert component carries. `df` comes from `get_metrics(start, now)`,
        # and a metric record whose timestamp cannot be read satisfies neither
        # bound, so it leaves EVERY window in silence and the compliance fraction is
        # computed over whatever happens to be left. Measured before this change, six
        # metric records through the PUBLIC `ingest_dataframe` (2 datable clean rows
        # and 4 breaching demographic_parity 0.95 rows with alert=True), plus one
        # datable HIGH-severity alert, on MetricsStoreConfig(enable_privacy=False):
        #   breaches with timestamp pd.NaT -> score 96.2 GREEN, components
        #       {'metric_compliance': 100.0, 'alert_frequency': 90.0}, n_metrics 3,
        #       and NOTHING counted or named the 4 records with no window position;
        #       get_metrics(window) returned 3 of the 7 stored rows.
        #   the same six rows, datable  -> score 33.8 RED, components
        #       {'metric_compliance': 0.0, 'alert_frequency': 90.0}.
        # A 100.0 metric compliance over four hidden breaches, published GREEN.
        #
        # An undatable metric record is COULD NOT CHECK for a window, exactly as an
        # undatable alert is: it was not counted, and it was not shown to be OUTSIDE
        # the window either, so the fraction of breaching records in the window is
        # not establishable and there is no honest number for this component. Metric
        # compliance is the component this score is ABOUT (the `len(df) == 0` branch
        # below already withholds the whole composite for want of it), so the
        # composite is withheld rather than graded off a fraction over a subset of
        # unknown completeness.
        #
        # THIS GUARD SITS ABOVE the `len(df) == 0` branch on purpose. A store whose
        # breaching records are ALL undatable has an empty window and would
        # otherwise be refused with the reason "No metric records in the window",
        # which is a false statement about a store holding six of them: the records
        # exist, they could not be placed. The reason a reader gets has to be the
        # real one.
        undatable_metrics = [r for r in self._records if _record_timestamp(r) is None]
        n_undatable_metrics = len(undatable_metrics)
        if n_undatable_metrics:
            # How many of them recorded a breach is the readable half of this
            # refusal, the same way get_alerts names the flagged metric records.
            n_undatable_breaching = sum(1 for r in undatable_metrics if r.alert is True)
            breach_note = (
                f" {n_undatable_breaching} of them carry alert=True, so breaches WERE "
                "recorded on records this window cannot account for."
                if n_undatable_breaching
                else ""
            )
            warnings.warn(
                f"{n_undatable_metrics} of {len(self._records)} metric record(s) in this "
                f"store carry no readable timestamp, so they cannot be placed in the "
                f"evaluation window and the fraction of breaching records in it could "
                f"NOT be established. Metric compliance is the component this score is "
                f"about, so no health score was computed: score is None and status is "
                f"'not_assessed'. This is COULD NOT CHECK. It is NOT a metric compliance "
                f"of 100: the undatable records were not counted, and they were not shown "
                f"to be outside the window either.{breach_note} See "
                f"n_undatable_metrics.",
                UserWarning,
                stacklevel=2,
            )
            # Same rule as the other withheld branches: a component that HAS
            # evidence is reported and one that does not is absent, which is how the
            # two are told apart.
            undatable_partial: Dict[str, float] = {}
            if alert_measured:
                undatable_partial["alert_frequency"] = max(0.0, 100.0 - len(alerts) * 10)
            return HealthScore(
                score=None,
                status="not_assessed",
                trend="unknown",
                trend_slope=0.0,
                components=undatable_partial,
                timestamp=now,
                explanation=(
                    f"{n_undatable_metrics} of {len(self._records)} metric record(s) in "
                    "this store carry no readable timestamp, so they could not be placed "
                    "in the evaluation window and metric compliance could not be "
                    "measured for it. The health score was NOT computed. This is not a "
                    "score of 100 and not a metric compliance of 100."
                    + breach_note
                    + (
                        f" Alert frequency IS measured ({len(alerts)} alert(s) in the "
                        "window) and is reported on its own."
                        if alert_measured
                        else ""
                    )
                ),
                n_metrics=len(df),
                n_alerts=len(alerts),
                n_undatable_metrics=n_undatable_metrics,
            )

        # C-04. THREE STATES. Every component below independently defaults to
        # the PERFECT value when its evidence table is empty, so a store holding
        # nothing scored 100.0 / green / "All systems nominal". Measured
        # 2026-09-07 on a fresh MetricsStore: score=100.0, status='green',
        # n_metrics=0, and the executive report rendered
        # "## Health Score: 100/100 (GREEN)".
        #
        # The metric records are the evidence this score is ABOUT. With none of
        # them there is nothing to be healthy or unhealthy, so the score is
        # withheld rather than computed from three defaults. Alerts and drift
        # rows are legitimately absent on a healthy system and are NOT required.
        if len(df) == 0:
            warnings.warn(
                "No metric records in the window, so no health score was computed. "
                "score is None and status is 'not_assessed': this is COULD NOT "
                "CHECK, not a clean bill of health.",
                UserWarning,
                stacklevel=2,
            )
            return HealthScore(
                score=None,
                status="not_assessed",
                trend="unknown",
                trend_slope=0.0,
                components={},
                timestamp=now,
                explanation=(
                    "No metric records were found in the evaluation window, so the "
                    "health score could not be computed. This is not a score of 100 "
                    "and certifies nothing."
                ),
                n_metrics=0,
                n_alerts=len(alerts),
            )

        # Component 1: Metric compliance. Derived from metric breach rows
        # only: ingested alerts mirror into the record table with alert=True
        # (source FairnessAlertPrioritizer) and already drive Component 2,
        # so counting them here would penalize the same alert twice.
        breach_df = df
        if "source" in df.columns:
            breach_df = df[df["source"] != "FairnessAlertPrioritizer"]

        # H-10. A metric that could not be computed is stored with value NaN and
        # alert=False, because FairnessMonitor's alert test is
        # `... and not np.isnan(val)`. That False then entered this mean as a
        # COMPLIANT record, so an undefined metric improved the health score.
        # Measured 2026-09-07: a monitor window whose only metric was NaN gave
        # metric_compliance 100.0 and an overall 100/100 GREEN.
        #
        # An unmeasured metric is not a compliant one. It leaves the denominator
        # and is counted separately, so "all clear" and "nothing was measurable"
        # stop sharing a number.
        #
        # R-2, 2026-09-09. The second way a record reached this mean without a
        # measurement: a record whose VALUE is fine but which nobody ever
        # compared to a threshold (a custom metric, a group positive rate, an
        # MMD score) arrived with alert=False and was averaged in as compliant.
        # Measured: built-ins both breaching scored 50.0; adding three
        # maximally unfair custom metrics RAISED it to 71.4. Such a record has
        # no alert determination: it leaves the denominator, is counted here,
        # and is named in the warning.
        breach_df, n_unassessable, undetermined_names, n_descriptive = _compliance_rows(breach_df)
        if undetermined_names:
            warnings.warn(
                f"{len(undetermined_names)} metric record(s) in the window carry no "
                f"alert determination (never compared to a threshold: "
                f"{', '.join(undetermined_names)}) and were excluded from metric "
                "compliance. They are could-not-check, not compliant.",
                UserWarning,
                stacklevel=2,
            )

        if len(breach_df) == 0 or "alert" not in breach_df.columns:
            # Every metric record in the window was unmeasurable, undetermined
            # or descriptive. Excluding them is correct, but then the
            # compliance component falls back to its perfect default and the
            # score reads 100/100 GREEN again, one level down from where C-04
            # was fixed. Same answer as C-04: withhold it.
            #
            # BGL S2b, 2026-09-17. The guard used to read
            # `len(breach_df) == 0 and (n_unassessable or n_descriptive)`, so it
            # missed the case where every record was excluded for the OTHER
            # reason a few lines up: the mirrored FairnessAlertPrioritizer rows.
            # Such a store has no compliance evidence at all, and execution fell
            # through to `else: metric_score = 100.0`. Measured on a store whose
            # only content was one unresolved CRITICAL alert: "97/100 (GREEN)"
            # over "| Metric Compliance | 100.0 |", with no warning. Zero
            # records were ever compared with a threshold; 100.0 is the perfect
            # default for no evidence, which is the defect this whole branch
            # exists to refuse. The count now includes them so the message says
            # how many rows were set aside and why.
            n_alert_mirror = max(0, int(len(df) - len(breach_df) - n_unassessable - n_descriptive))
            n_excluded = n_unassessable + n_descriptive + n_alert_mirror
            warnings.warn(
                f"All {n_excluded} metric record(s) in the window were unmeasurable, "
                "carried no alert determination, or were mirrored alert rows already "
                "counted by the alert component, so metric compliance has NO evidence "
                "and no health score was computed. score is None and status is "
                "'not_assessed': this is COULD NOT CHECK, not a metric compliance "
                "of 100.",
                UserWarning,
                stacklevel=2,
            )
            # The composite is withheld, but ALERT FREQUENCY is not: the alerts
            # in the window are real evidence, counted directly, and dropping
            # them is the reverse of the defect this branch fixes. An empty
            # `components` told a reader nothing was known when the alert count
            # was known exactly. So the component that HAS evidence is reported
            # and the one that does not is simply absent, which is what makes
            # the two distinguishable.
            #
            # BGL5 A-operations-3, 2026-09-27. "the alerts in the window are real
            # evidence" is true only when an alert channel was ever connected to
            # this store. With none, `len(alerts)` is 0 because nothing was ever
            # fed, and this dict published alert_frequency 100.0 for a component
            # nobody measured, in the very branch that exists to refuse exactly
            # that. Measured on a store whose only record was one unmeasurable
            # metric and no alert record: components {'alert_frequency': 100.0}
            # before, {} after, with the explanation naming both absences.
            #
            # BGL5 A-operations-3-b, 2026-09-29. `if self._alert_records` was the
            # store-level channel test again, so this branch published
            # alert_frequency for a window whose alert count could not be
            # established (an undatable record leaves the window in silence). It
            # now reads the same `alert_measured` the graded path uses.
            partial: Dict[str, float] = {}
            if alert_measured:
                partial["alert_frequency"] = max(0.0, 100.0 - len(alerts) * 10)
            if alert_measured:
                alert_sentence = (
                    f"Alert frequency IS measured ({len(alerts)} alert(s) in the window) "
                    "and is reported on its own; metric compliance and drift stability "
                    "are absent because nothing was measured for them."
                )
            elif alert_channel_fed:
                alert_sentence = (
                    f"Alert frequency is absent too: {n_undatable_alerts} alert "
                    "record(s) in this store carry no readable timestamp, so they "
                    "could not be placed in the window and its alert count is not a "
                    "measured zero. No component of this score has evidence."
                )
            else:
                alert_sentence = (
                    "Alert frequency is absent too: this store has never received an "
                    "alert record, so its empty alert list is not a measured zero. "
                    "No component of this score has evidence."
                )
            return HealthScore(
                score=None,
                status="not_assessed",
                trend="unknown",
                trend_slope=0.0,
                components=partial,
                timestamp=now,
                explanation=(
                    f"All {n_excluded} metric record(s) in the evaluation window "
                    "were unmeasurable, carried no alert determination, or were "
                    "mirrored alert rows, so metric compliance had no evidence and "
                    "the health score could not be computed. This is not a score of "
                    "100 and not a metric compliance of 100. " + alert_sentence
                ),
                n_metrics=0,
                n_alerts=len(alerts),
                n_not_assessable=n_unassessable,
            )

        # Unconditional now: the guard above returns for an empty frame AND for
        # one carrying no `alert` column, which is the only other way this used
        # to reach `else: metric_score = 100.0` with nothing measured.
        alert_frac = breach_df["alert"].astype(bool).mean()
        metric_score = max(0.0, 100.0 * (1.0 - alert_frac * 2))

        # Component 2: Alert frequency (each alert costs 10 points)
        n_alerts = len(alerts)
        alert_score = max(0.0, 100.0 - n_alerts * 10)

        # Component 3: Drift stability, over the drift rows that carry a
        # verdict. READINESS-5: a REFUSED drift test (drift_detected is None)
        # averaged to NaN here and graded as a hard 0.0. See _drift_stability.
        maybe_drift_score, n_drift_undetermined, drift_undetermined = _drift_stability(drift_df)
        if n_drift_undetermined:
            named = f": {', '.join(drift_undetermined)}" if drift_undetermined else ""
            warnings.warn(
                f"{n_drift_undetermined} drift result(s) in the window carry no verdict "
                f"because the detector refused to run the test{named}. They are "
                "could-not-check and are excluded from drift stability, counted "
                "neither as stable nor as drifting.",
                UserWarning,
                stacklevel=2,
            )
        if maybe_drift_score is None and n_drift_undetermined:
            # Every drift result in the window was refused, so this component
            # has no evidence at all. Same answer as C-04 and H-10/R-2 one
            # level up: withhold the composite rather than let a 20 percent
            # weight carry a number nobody measured, in either direction.
            warnings.warn(
                f"All {n_drift_undetermined} drift result(s) in the window were refused by "
                "the detector, so drift stability could not be measured and no health "
                "score was computed. score is None and status is 'not_assessed': this is "
                "COULD NOT CHECK, and it is not a drift stability of 0 either.",
                UserWarning,
                stacklevel=2,
            )
            return HealthScore(
                score=None,
                status="not_assessed",
                trend="unknown",
                trend_slope=0.0,
                components={},
                timestamp=now,
                explanation=(
                    f"All {n_drift_undetermined} drift result(s) in the evaluation window "
                    "were refused by the detector, so the drift stability component could "
                    "not be measured and the health score could not be computed. This is "
                    "not a score of 100 and not a drift stability of 0."
                ),
                n_metrics=len(df),
                n_alerts=n_alerts,
                n_not_assessable=n_unassessable,
            )

        # BGL3 operations-4, 2026-09-27. NO drift row in the window is its own
        # state: no drift test was recorded here, as against one that ran and
        # refused (handled above). _drift_stability used to answer 100.0 for it,
        # so a fifth of the composite was a constant standing in for a component
        # nobody measured; see its docstring for the measured before-and-after.
        #
        # The composite is NOT withheld for this, because the two components that
        # do have evidence have real evidence: this is the partial-coverage rule
        # the validator states in its own words, "grade over the subset that ran,
        # never withhold the verdict entirely, and state the count that did NOT
        # run right beside the verdict". So the 50/30 weights are renormalised
        # over themselves (0.80) and drift stability is ABSENT from `components`
        # rather than present holding a number, which is how a reader and a test
        # can tell the two apart. Two components both at 100 still yield 100.0;
        # what changes is that an imperfect score is no longer lifted by 20
        # points of nothing.
        drift_measured = maybe_drift_score is not None
        # BGL5 A-operations-3, 2026-09-27. THE SAME DEFECT, in the 30 percent
        # component. `alert_score = max(0.0, 100.0 - n_alerts * 10)` and n_alerts
        # comes from get_alerts, which returns [] for a store whose alert channel
        # was never fed, so a component with NO evidence carried a perfect 100.0
        # and lifted the composite by up to 30 points. Measured on two metric
        # records both carrying alert=True and no alert record at all:
        #   before  score 37.5 (yellow-to-red boundary) with components
        #           {'metric_compliance': 0.0, 'alert_frequency': 100.0} and a
        #           warning naming only the drift exclusion: the store held two
        #           breaches beside a spotless alert component.
        #   after   score 0.0 (red), components {'metric_compliance': 0.0}, one
        #           extra warning naming the alert exclusion, and the explanation
        #           says the score covers metric compliance only.
        # A clean drift run leaves a row (drift_detected=False) while a clean
        # alert run leaves nothing, which is why the alert absence needs the
        # channel test and the drift absence does not. The test is store-level
        # (has ingest_alert EVER been called) and not per-window, so a quiet week
        # on a connected channel is still a measured zero, and both windows of the
        # trend below use the same component set.
        if alert_channel_fed and n_undatable_alerts:
            warnings.warn(
                f"{n_undatable_alerts} of {len(self._alert_records)} alert record(s) in "
                "this store carry no readable timestamp, so they cannot be placed in "
                "the evaluation window and the alert count for it could not be "
                "established. Alert frequency is EXCLUDED from the health score and is "
                "absent from `components`. It is NOT an alert frequency of 100: the "
                "undatable alerts were not counted, and they were not shown to be "
                "outside the window either.",
                UserWarning,
                stacklevel=2,
            )
        if not alert_channel_fed:
            warnings.warn(
                "This store has never received an alert record, so alert frequency "
                "could not be measured. It is EXCLUDED from the health score and is "
                "absent from `components`. It is NOT an alert frequency of 100: "
                "nothing was compared, because a clean prioritizer run and an "
                "unconnected alert channel both leave zero alert records. Call "
                "ingest_alert to have this component graded.",
                UserWarning,
                stacklevel=2,
            )
        if not drift_measured:
            warnings.warn(
                "No drift result in the window, so drift stability could not be "
                "measured. It is EXCLUDED from the health score (the metric-compliance "
                "and alert-frequency weights are renormalised over the components that "
                "have evidence) and is absent from `components`. It is NOT a drift "
                "stability of 100: nothing was compared. Ingest drift results to have "
                "this component graded.",
                UserWarning,
                stacklevel=2,
            )
        # The documented 50/30/20 weights, renormalised over the components that
        # have evidence. With all three present this is arithmetically identical
        # to the previous expression, and with drift alone missing it is identical
        # to the previous `/ 0.80`.
        weighted = 0.50 * metric_score
        weight_total = 0.50
        if alert_measured:
            weighted += 0.30 * alert_score
            weight_total += 0.30
        # `maybe_drift_score is not None` rather than the `drift_measured` bool:
        # mypy does not narrow an Optional through a boolean variable, so the bool
        # form reported "Unsupported operand types for * (float and None)" here.
        if maybe_drift_score is not None:
            weighted += 0.20 * maybe_drift_score
            weight_total += 0.20
        score = weighted / weight_total
        score = round(min(100.0, max(0.0, score)), 1)

        # Status
        if score >= 80:
            status = "green"
        elif score >= 50:
            status = "yellow"
        else:
            status = "red"

        # Trend: compare current window score with the previous window score,
        # built from the SAME three components. (Previously the alert and
        # drift components of the previous window were hardcoded to 100, so
        # any current alert/drift made the slope read 'degrading' even when
        # the previous window was just as bad or worse.)
        prev_start = start - time_window
        prev_df = self.get_metrics(
            start_time=prev_start,
            end_time=start,
            apply_privacy=False,
        )
        # Same alert-row exclusion as the current window so the trend
        # compares like with like.
        if "source" in prev_df.columns:
            prev_df = prev_df[prev_df["source"] != "FairnessAlertPrioritizer"]
        # And the same unmeasured / undetermined exclusion (H-10, R-2), for
        # the same reason: a previous window padded with compliant-looking
        # undetermined rows would make the current one read as degrading.
        prev_df, _, _, _ = _compliance_rows(prev_df)
        # READINESS-5. Set when the PREVIOUS window has no composite to compare
        # against, which is its own state and not a stable trend.
        prev_not_assessable = False
        if len(prev_df) > 0 and "alert" in prev_df.columns:
            prev_alert_frac = prev_df["alert"].astype(bool).mean()
            prev_metric_score = max(0.0, 100.0 * (1.0 - prev_alert_frac * 2))

            prev_alerts = self.get_alerts(start_time=prev_start, end_time=start)
            prev_alert_score = max(0.0, 100.0 - len(prev_alerts) * 10)

            prev_drift_df = self.get_drift_history(
                start_time=prev_start,
                end_time=start,
            )
            # READINESS-5. The trend inherited the same defect one window
            # back: a refused drift verdict in the PREVIOUS window graded 0.0
            # there, cut prev_score by 20 points and made the current window
            # read as 'improving' off a comparison nobody ran.
            maybe_prev_drift, n_prev_undetermined, _ = _drift_stability(prev_drift_df)
            prev_drift_measured = maybe_prev_drift is not None
            if maybe_prev_drift is None and n_prev_undetermined:
                prev_not_assessable = True
                prev_score = score
                slope = 0.0
            elif prev_drift_measured != drift_measured:
                # BGL3 operations-4. One window has drift evidence and the other
                # does not, so the two composites are computed over different
                # component sets and their difference is not a rate of change of
                # anything. Same reason as the alert-row and undetermined-row
                # exclusions above: the trend compares like with like or it says
                # it could not.
                prev_not_assessable = True
                prev_score = score
                slope = 0.0
            else:
                # Same component set as the current window, for the same
                # like-with-like reason: `alert_measured` is a store-level fact,
                # so it cannot differ between the two windows.
                prev_weighted = 0.50 * prev_metric_score
                prev_weight_total = 0.50
                if alert_measured:
                    prev_weighted += 0.30 * prev_alert_score
                    prev_weight_total += 0.30
                if maybe_prev_drift is not None:
                    prev_weighted += 0.20 * maybe_prev_drift
                    prev_weight_total += 0.20
                prev_score = prev_weighted / prev_weight_total
                prev_score = min(100.0, max(0.0, prev_score))
                # DIVIDE BY THE WINDOW'S REAL LENGTH, not by timedelta.days.
                #
                # This was `/ time_window.days if time_window.days > 0 else 0.0`, and
                # `.days` TRUNCATES: timedelta(hours=12).days and
                # timedelta(hours=23).days are both 0. So for any window shorter than a
                # day the slope was forced to exactly 0.0, which falls into the
                # `else: trend = "stable"` arm of the mapping below, and a stable trend
                # was asserted for two windows whose scores may differ by anything at
                # all. A twelve-hour window over a score that fell from 100 to 20
                # reported "stable".
                #
                # total_seconds() measures the window that was actually asked for.
                #
                # The guarded zero branch is kept and is NOT a could-not-check: a
                # window of no duration cannot reach this line at all, because an empty
                # window returns earlier with score=None and trend='unknown' ("No
                # metric records were found in the evaluation window ... This is not a
                # score of 100 and certifies nothing"). A first attempt at this fix set
                # prev_not_assessable here, and sabotaging that branch left the test
                # green, which is how the branch was shown to be unreachable. An
                # unreachable guard is worse than none, because the next reader trusts
                # it, so the zero stays as plain arithmetic protection on a division.
                window_days = time_window.total_seconds() / 86400.0
                slope = (score - prev_score) / window_days if window_days > 0 else 0.0
        else:
            # BGL3 operations-4, 2026-09-27. This branch used to set
            # `prev_score = score; slope = 0.0`, which fell through to the
            # slope-based mapping below and reported trend='stable' with
            # slope=0.0. Measured on four clean records whose previous window
            # held NO record at all: trend 'stable', explanation "Fairness health
            # score is 100/100 (healthy) with a stable trend". "There was no
            # previous window to compare against" and "the score did not move"
            # reached the caller as the same string, and 'stable' is the
            # reassuring one. A first window has no trend; it is not a flat one.
            prev_not_assessable = True
            prev_score = score
            slope = 0.0

        if prev_not_assessable:
            # Three states on the trend too. 'stable' would assert that the two
            # windows match, which is exactly what could not be established.
            trend = "unknown"
        elif slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "degrading"
        else:
            trend = "stable"

        # NLG explanation
        explanation = _health_explanation(
            score, status, trend, n_alerts, time_window.days, n_not_assessable=n_unassessable
        )
        if not drift_measured:
            # The sentence is what a reader reads off the dashboard, so the
            # missing fifth of the score has to be in it and not only in
            # `components`.
            covered = (
                "metric compliance and alert frequency" if alert_measured else "metric compliance"
            )
            explanation += (
                " Drift stability was NOT measured (no drift result in the window), so "
                f"this score covers {covered} only."
            )
        if not alert_measured:
            # Same reason as the drift sentence: a third of the documented
            # composite is missing and the reader has to be told at the surface.
            # BGL5 A-operations-3-b: and told WHICH absence it was, because the
            # remedy differs (connect the channel, or give the alerts a readable
            # timestamp).
            why = (
                f"{n_undatable_alerts} alert record(s) in this store carry no readable "
                "timestamp, so they could not be placed in the window"
                if alert_channel_fed
                else "this store has never received an alert record"
            )
            explanation += (
                f" Alert frequency was NOT measured ({why}), so it is excluded from the "
                "score rather than counted as a perfect 100."
            )
        if trend == "unknown":
            explanation += (
                " The trend could not be computed: there is no comparable previous "
                "window, so this is not a stable trend."
            )

        # A component with evidence is present holding its number; a component
        # with none is ABSENT, which is how a reader and a test tell the two
        # apart. Same rule for all three.
        components = {"metric_compliance": round(metric_score, 1)}
        if alert_measured:
            components["alert_frequency"] = round(alert_score, 1)
        if maybe_drift_score is not None:
            components["drift_stability"] = round(maybe_drift_score, 1)

        return HealthScore(
            score=score,
            status=status,
            trend=trend,
            trend_slope=round(slope, 4),
            components=components,
            timestamp=now,
            explanation=explanation,
            n_metrics=len(df),
            n_alerts=n_alerts,
            n_not_assessable=n_unassessable,
        )

    # Lifecycle

    def prune(self, before: Optional[datetime] = None) -> int:
        """Remove records older than *before* (or max_history_days)."""
        cutoff = before or (datetime.now() - timedelta(days=self.config.max_history_days))
        original = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        return original - len(self._records)

    def clear(self) -> None:
        """Remove all stored data, the model registry included."""
        self._records.clear()
        self._alert_records.clear()
        self._drift_records.clear()
        self._threshold_records.clear()
        # BGL g003, 2026-09-17. The registry survived "remove all stored data",
        # so a technical report generated after clear() still rendered a
        # "Registered Models" section and a Store Statistics block reading
        # n_models=1 over a store holding nothing else. Measured: get_summary()
        # went from n_metric_records 1 / n_models 1 to n_metric_records 0 /
        # n_models 1.
        self._models.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Export the entire store as a privacy-protected DataFrame."""
        return self.get_metrics()

    def get_summary(self) -> Dict[str, Any]:
        """Summary statistics of the store contents."""
        return {
            "n_metric_records": len(self._records),
            "n_alert_records": len(self._alert_records),
            "n_drift_records": len(self._drift_records),
            "n_threshold_records": len(self._threshold_records),
            "n_models": len(self._models),
            "metric_names": self.get_metric_names(),
            "groups": self.get_groups(),
            "sources": self.get_sources(),
            "time_range": self.get_time_range(),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"MetricsStore(metrics={len(self._records)}, "
            f"alerts={len(self._alert_records)}, "
            f"drift={len(self._drift_records)})"
        )


# NLG helpers (kept local to avoid circular imports)


#: The metric names FairnessMonitor builds its per-attribute keys from. It
#: writes ``f"{metric}_{col}"`` (tracker.py, update_and_check), so the metric is
#: recovered by matching these prefixes, NOT by looking for a separator.
_MONITOR_METRIC_PREFIXES = (
    "disparate_impact",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
)


#: Columns this store's own get_metrics() emits that ingest_dataframe must
#: understand, named once so the producer and the consumer cannot drift apart.
#: Words that end a METRIC name in this library rather than naming a column, so
#: `demographic_parity_difference` is not read as `demographic_parity` measured
#: on a column called `difference`. Explicit because it has to be: the direction
#: resolver matches these as tokens anywhere in a name and cannot separate the
#: two readings.
_METRIC_NAME_TAILS = frozenset(
    {
        "difference",
        "diff",
        "ratio",
        "gap",
        "score",
        "rate",
        "index",
        "error",
        "delta",
    }
)

_ALERT_DETERMINED_COLUMN = "alert_determined"
_PRIVACY_LEVEL_COLUMN = "privacy_level"


def _split_monitor_key(key: str) -> Tuple[str, str]:
    """Split a FairnessMonitor metric key into (metric_name, group).

    ``demographic_parity_gender`` is the metric ``demographic_parity`` measured
    on the protected column ``gender``. A key that matches no known metric is
    returned whole, on group ``"overall"``: a custom metric is not a per-column
    reading and must not be cut apart on a guess.

    READINESS-5, 2026-09-10: ``ingest_window_metrics`` split this key and
    ``ingest_from_monitor``, which ingests the SAME numbers by the bulk route,
    stored it verbatim, so one measurement had two names and the composite one
    is a name no other part of the library knows. One definition, used by both.

    READINESS-6, LATER THE SAME DAY, and this is the interesting half. That
    first fix split on the literal ``"_group_"``, copied from the older of the
    two call sites. The monitor does not emit that separator. It emits
    ``f"{metric}_{col}"``, so ``"_group_"`` appears ONLY when the protected
    COLUMN is itself named ``group_gender``:

        column 'gender'       -> key 'demographic_parity_gender'
                              -> old split gave ('demographic_parity_gender', 'overall')
        column 'group_gender' -> key 'demographic_parity_group_gender'
                              -> old split gave ('demographic_parity', 'gender')

    Until 2026-09-09 that was almost harmless, because ``_detect_protected_cols``
    only took columns carrying the ``group_`` prefix. R-3 taught it to
    auto-detect columns without one, and this parser was never updated, so every
    auto-detected column has produced an unparseable key since. Downstream,
    ``metric_direction('demographic_parity_gender')`` is UNKNOWN, so the
    threshold simulator refuses every real metric and the store records no
    groups at all.

    The READINESS-5 guard test did not catch it because its fixture column was
    named ``group_gender``, so the separator came from the COLUMN NAME rather
    than from the monitor. A test that supplies the thing under test passes for
    the wrong reason; the pins below now cover a plain column name too.
    """
    text = str(key)

    # 1. The documented ``group_`` column convention, which is explicit and
    #    unambiguous, and covers a CUSTOM metric measured per column as well as
    #    a built-in one. This is the split that always worked.
    parts = text.rsplit("_group_", 1)
    if len(parts) > 1 and parts[0] and parts[1]:
        return parts[0], parts[1]

    # 2. A built-in metric followed by an AUTO-DETECTED column, which carries no
    #    prefix and so has no separator to find. This is the case R-3 created on
    #    2026-09-09 and that nothing parsed until now.
    #
    #    The remainder has to be a column name and not the tail of a longer
    #    METRIC name, or "demographic_parity_difference" is cut into the metric
    #    "demographic_parity" on a column called "difference", taking the whole
    #    *_difference / *_ratio family with it. An existing test caught exactly
    #    that while this parser was being corrected, which is why the set below
    #    is explicit rather than inferred: `metric_direction` resolves these by
    #    TOKEN, so it answers LOWER_IS_BETTER for "custom_gap_group_sex" too and
    #    cannot be used to tell the two apart.
    for metric in _MONITOR_METRIC_PREFIXES:
        prefix = metric + "_"
        if text.startswith(prefix) and len(text) > len(prefix):
            column = text[len(prefix) :]
            if column in _METRIC_NAME_TAILS:
                break
            return metric, column

    return text, "overall"


def _alert_determination(raw: Any) -> Optional[bool]:
    """Read an alert flag off a history row without inventing a verdict.

    ``None`` and NaN (pandas' spelling of None in a mixed column) both mean
    the metric was never compared to a threshold, and stay ``None``. Anything
    else is a recorded determination and becomes a bool. R-2.
    """
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    return bool(raw)


def _record_alert_flag(record: Any) -> Optional[bool]:
    """The ``alert`` flag of a stored record, whatever shape it is in.

    ``_records`` normally holds :class:`StoredMetricRecord` instances, but plain
    dicts are appended by callers and by tests, so both are read here. Returns
    ``None`` when the record carries no readable determination, which is the same
    three-state rule as :func:`_alert_determination`. Added BGL5 A-operations-3
    so ``get_alerts`` can say how many breaches the metric side recorded while
    the alert channel held nothing.
    """
    raw = record.get("alert") if isinstance(record, dict) else getattr(record, "alert", None)
    return _alert_determination(raw)


def _readable_time_value(raw: Any) -> Optional[pd.Timestamp]:
    """A :class:`pd.Timestamp` for *raw*, or ``None`` when it is not a time.

    THE ONE PLACE this module decides whether a value can be read as an instant.
    Extracted from :func:`_alert_timestamp` in BGL7 B4-rep-pre-w2, 2026-09-29,
    because three other surfaces needed exactly the same reading and had none:
    a METRIC record's timestamp (``compute_health_score`` graded a window over
    records it could not place in it) and a CALLER-SUPPLIED window bound (every
    reader in this class compared against ``pd.Timestamp(bound)`` and an
    unreadable bound answers False in both directions, so the query returned an
    empty result in silence). A second copy of one rule is how the cali-BRATIO-n
    incident happened, so there is one copy and three callers.

    ``None`` is COULD NOT CHECK: no time at all (``None``), a null
    (``NaT`` / NaN, which is the shape a pandas row produces for a missing cell)
    or a value no parser can read as a time.
    """
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if ts is pd.NaT:
        return None
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        return None
    return ts


def _now_like_records(records: Sequence[Any]) -> datetime:
    """``now``, in the same tz-awareness as the records it will be compared to.

    BGL grade-1 G08, 2026-09-30. See the call site in
    :meth:`MetricsStore.compute_health_score` for the measurement. A tz-naive
    bound and a tz-aware timestamp column cannot be compared at all in pandas,
    and the window this returns is compared against whatever the store holds.

    "Now" is one instant, so returning it with the records' own tzinfo attached
    is a change of REPRESENTATION and never of the window: no offset is added,
    subtracted or assumed. Nothing is inferred about a naive store, which keeps
    the naive local ``datetime.now()`` it has always had, so an existing caller's
    window is byte-identical.

    A caller-supplied bound is deliberately NOT aligned this way (see
    :func:`_window_bound`): a naive bound means local time and an aware column
    means a fixed offset, so reconciling those two would require guessing which
    the caller meant, and pandas' refusal is the honest answer there.
    """
    for record in records:
        raw = getattr(record, "timestamp", None)
        tzinfo = getattr(raw, "tzinfo", None)
        if tzinfo is not None:
            return datetime.now(tz=tzinfo)
    return datetime.now()


def _window_bound(raw: Any, *, method: str, argument: str) -> Optional[pd.Timestamp]:
    """Read a CALLER-SUPPLIED window bound, or refuse the query outright.

    BGL7 B4-rep-pre-w2, 2026-09-29. Every windowed reader in :class:`MetricsStore`
    compared its rows against ``pd.Timestamp(start_time)``, and
    ``ts >= pd.Timestamp(pd.NaT)`` is False for every ts, as is ``ts <= NaT``, so
    a bound the CALLER could not express emptied the result with nothing said.
    Measured before this change on a store holding six metric records and one
    datable HIGH-severity alert::

        get_alerts(start_time=pd.NaT)        -> 0 alert(s), warnings=0
        get_metrics(start_time=pd.NaT)       -> 0 row(s),   warnings=0
        get_drift_history(start_time=pd.NaT) -> 0 row(s),   warnings=0

    The three guards already in :meth:`get_alerts` all ask about the STORE or the
    RECORDS; none of them could see a defect in the QUESTION. This is not missing
    data and it is not a measurement: it is a window that does not exist, so it is
    REFUSED rather than answered with an empty result that reads as "nothing
    here". A refusal cannot be misread, which is why it is a raise and not a
    warning beside a plausible-looking frame.

    ``None`` still means "no bound", exactly as before.
    """
    if raw is None:
        return None
    ts = _readable_time_value(raw)
    if ts is None:
        raise ValueError(
            f"MetricsStore.{method}: {argument}={raw!r} cannot be read as a time, so the "
            f"window it is supposed to bound does not exist and no result can be "
            f"computed for it. This is COULD NOT CHECK and it is refused rather than "
            f"answered: every comparison against an unreadable bound is False, so the "
            f"query would otherwise return an EMPTY result that reads as 'nothing in "
            f"this window'. Pass a datetime, a readable timestamp, or None for no bound."
        )
    return ts


def _alert_timestamp(record: Any) -> Optional[pd.Timestamp]:
    """When an alert record says it fired, or ``None`` when it cannot say.

    Three states, like every other reader in this module: a :class:`pd.Timestamp`
    when the record carries a time that can be read, and ``None`` when it carries
    no time at all (the key absent), a null (``None`` / ``NaT`` / NaN, which is
    the shape a pandas row produces for a missing cell) or a value that cannot be
    parsed as a time. ``None`` is COULD NOT CHECK: such a record cannot be placed
    inside or outside any window, so it is neither in a windowed result nor
    evidence that the window was empty.

    Added BGL5 A-operations-3-b, 2026-09-29, after ``get_alerts`` was measured
    returning ``[]`` with no warning for a window holding a HIGH-severity,
    priority-9.0 alert: ``pd.Timestamp(pd.NaT) >= start`` is False, and so is
    ``<= end``, so an undatable record fell out of BOTH bounds in silence. The
    comparison could not have said anything else; what was missing was a reader
    that separates "outside the window" from "no window position at all".
    """
    raw = (
        record.get("timestamp") if isinstance(record, dict) else getattr(record, "timestamp", None)
    )
    return _readable_time_value(raw)


# ONE reader for a RECORD's time, whichever kind of record it is. The body above
# reads ``record["timestamp"]`` or ``record.timestamp`` and nothing
# alert-specific, and a metric record needs the identical three-state reading, so
# it is aliased rather than copied (BGL7 B4-rep-pre-w2, 2026-09-29: the metric
# side of this store had no such reader at all, and compute_health_score graded a
# 50 percent component over records it could not place in its window).
_record_timestamp = _alert_timestamp


def _alert_priority_value(d: Dict[str, Any]) -> float:
    """The priority score an alert payload actually carries, or NaN.

    BGL g003, 2026-09-17. This was ``d.get("priority_score", 0.0)`` inline in
    :meth:`MetricsStore.ingest_alert`. Two ways it invented a number:

    - the key is ABSENT, which is how every dict-shaped alert that the public
      method accepts arrives when the caller did not score it. ``0.0`` is the
      bottom of the prioritizer's roughly 0-to-10 scale, so "nobody ranked
      this" was published as "ranked lowest".
    - the key is PRESENT holding ``None``. ``.get`` with a default does not
      fire for that, so the None went straight into a field annotated float
      and reached the frame as an object-dtype None.

    ``AlertPayload`` already spells the could-not-check priority ``float("nan")``
    (see its docstring), so NaN is the value this returns, and it warns, because
    the fix is on the producing side.
    """
    raw = d.get("priority_score")
    if raw is None:
        warnings.warn(
            "ingest_alert: this alert carries no priority_score, so the stored "
            "alert_priority value is NaN (could not check), not 0.0. A 0.0 would be "
            "the LOWEST priority on the prioritizer's scale and would be "
            "indistinguishable from an alert that was scored and ranked bottom.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")
    try:
        return float(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"ingest_alert: priority_score {raw!r} is not a number, so the stored "
            "alert_priority value is NaN (could not check), not 0.0.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")


def _compliance_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str], int]:
    """Keep only the rows that can enter the metric-compliance mean.

    A row qualifies when its value is a real number (H-10) AND it carries an
    alert determination (R-2, the ``alert_determined`` column; absent only on
    frames built before that column existed, in which case every row is taken
    as determined, exactly as before). A descriptive row
    (``_DESCRIPTIVE_METRICS``) with no determination is not compliance
    evidence at all and is set aside silently; any other row with no
    determination is set aside, counted and named.

    Returns ``(kept_rows, n_not_assessable, undetermined_names,
    n_descriptive)``: the rows that enter the mean; how many were dropped as
    unmeasurable or undetermined; the sorted metric names of the undetermined
    ones, for the warning; and how many descriptive rows were set aside.
    """
    if len(df) == 0:
        return df, 0, [], 0
    keep = pd.Series(True, index=df.index)
    if "value" in df.columns:
        keep &= df["value"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))
    undetermined_names: List[str] = []
    descriptive = pd.Series(False, index=df.index)
    if "alert_determined" in df.columns:
        determined = df["alert_determined"].astype(bool)
        if "metric_name" in df.columns:
            names = df["metric_name"].astype(str)
            descriptive = ~determined & names.isin(_DESCRIPTIVE_METRICS)
            undetermined_names = sorted(set(names[keep & ~determined & ~descriptive].tolist()))
        keep &= determined
    n_descriptive = int(descriptive.sum())
    n_not_assessable = int((~keep).sum()) - n_descriptive
    return df[keep], n_not_assessable, undetermined_names, n_descriptive


def _drift_stability(df: pd.DataFrame) -> Tuple[Optional[float], int, List[str]]:
    """Drift stability over the drift rows that actually carry a verdict.

    ``MultiscaleDriftResult.drift_detected`` is ``Optional[bool]``: the
    detector returns ``None`` when it REFUSED to run the test (a series too
    short to decompose, no scale with a usable score) and warns when it does.
    ``None`` is could-not-check, not stability, and not instability either.

    Returns ``(score, n_undetermined, undetermined_metrics)``. ``score`` is
    ``None`` whenever no row carries a verdict, and ``n_undetermined``
    distinguishes the two ways that happens: a positive count means rows exist
    and every one of them was refused, and zero means there is no drift row at
    all. The caller needs both, because it withholds the composite for the first
    and grades over the remaining components for the second.

    BGL3 operations-4, 2026-09-27. An empty table used to return ``100.0``,
    described here as "the documented 100.0, because a system that ran no drift
    test at all has no drift rows either". That is the defect this campaign is
    about: 100.0 is PERFECT stability, and it was carrying 20 percent of a
    colour-banded composite for a component nothing had measured. Measured on
    four clean, fully determined metric records and zero drift rows, which is the
    default state of every ``MetricsStore``::

        score=100.0  status='green'  trend='stable'
        components={'metric_compliance': 100.0, 'alert_frequency': 100.0,
                    'drift_stability': 100.0}
        explanation "Fairness health score is 100/100 (healthy) ..."
        0 warnings

    with the word drift appearing nowhere a reader would see it.

    READINESS-5, 2026-09-10. ``drift_df["drift_detected"].mean()`` over a
    single refused verdict is NaN, and ``max(0.0, 100.0 * (1.0 - nan))`` is
    0.0, because NaN loses every comparison. A refused drift test was
    therefore graded as the WORST possible drift stability and weighted 20
    percent into a colour-banded composite with no warning and no
    could-not-check state. Measured: two clean, fully determined metric
    records plus one refused drift test printed "Health Score 80.0 GREEN"
    with "drift_stability 0.0".
    """
    if len(df) == 0 or "drift_detected" not in df.columns:
        # No drift row at all, so nothing to grade. n_undetermined is 0, which is
        # what tells the caller this is the empty case rather than the
        # every-verdict-refused one.
        return None, 0, []
    verdicts = df["drift_detected"].apply(_alert_determination)
    determined = verdicts.notna()
    n_undetermined = int((~determined).sum())
    undetermined_metrics: List[str] = []
    if n_undetermined and "metric" in df.columns:
        undetermined_metrics = sorted(set(df.loc[~determined, "metric"].astype(str)))
    if not bool(determined.any()):
        return None, n_undetermined, undetermined_metrics
    drift_frac = float(verdicts[determined].astype(bool).mean())
    return max(0.0, 100.0 * (1.0 - drift_frac)), n_undetermined, undetermined_metrics


def _health_explanation(
    score: float,
    status: str,
    trend: str,
    n_alerts: int,
    window_days: int,
    n_not_assessable: int = 0,
) -> str:
    """Generate a plain-language sentence for the health score."""
    status_word = {
        "green": "healthy",
        "yellow": "showing moderate fairness concerns",
        "red": "in critical condition requiring immediate attention",
    }.get(status, "under review")

    trend_word = {
        "improving": "and improving",
        "degrading": "and deteriorating",
        "stable": "with a stable trend",
    }.get(trend, "")

    alert_clause = ""
    if n_alerts > 0:
        alert_clause = (
            f" {n_alerts} alert{'s' if n_alerts != 1 else ''} fired in the past {window_days} days."
        )

    # R-2. Say beside the score what it does NOT cover, rather than leaving
    # the reader to infer it from a count that stayed on the object.
    excluded_clause = ""
    if n_not_assessable > 0:
        excluded_clause = (
            f" {n_not_assessable} metric record{'s' if n_not_assessable != 1 else ''} "
            "could not be assessed (no measurement or no alert determination) and "
            "are excluded from this score."
        )

    # `trend_word` is empty for a trend that could not be computed, and the old
    # single f-string then left a space before the full stop. "unknown" used to be
    # rare; BGL3 operations-4 made it the honest answer for a first window, so the
    # sentence has to read properly without it. The three measured trends produce
    # exactly the same string as before.
    head = f"Fairness health score is {score:.0f}/100 ({status_word})"
    head = f"{head} {trend_word}." if trend_word else f"{head}."
    return f"{head}{alert_clause}{excluded_clause}".strip()
