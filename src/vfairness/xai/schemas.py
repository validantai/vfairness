"""
vfairness.xai.schemas
=====================

Python mirrors of TypeScript contracts the consuming platform keeps in its
own, unpublished frontend sources. These are the frozen interfaces shared
with that frontend; bumping either side requires updating both in the same
commit.

Use dataclasses so the worker can ``asdict()`` them straight into a
Supabase insert. The ``to_db_row()`` helpers on the contract classes
emit the snake_case keys the canonical assessment-layer tables expect.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Naming-convention bridge
# Top-level row columns use snake_case (matches the SQL column names).
# Nested JSONB payloads (attributions, counterfactuals.flippedFeatures,
# etc.) use camelCase so the React reader can consume them without an
# extra transformer layer. The frontend TS contracts define camelCase for
# everything that crosses the wire as JSON; the Postgres column names stay
# snake_case as is the platform convention.


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:] if p)


def _camel_dict(obj: Any) -> Any:
    """Recursively map snake_case dict keys to camelCase.

    Used in ``to_db_row()`` for nested JSONB payloads so the frontend
    contract (``baseValue``, ``flippedFeatures``, etc.) sees the right
    shape verbatim. Top-level row columns are NOT touched -- those map
    to snake_case Postgres column names.
    """
    if isinstance(obj, dict):
        # A dict comprehension keyed on the CONVERTED name silently drops a
        # field whenever two source keys converge on one camelCase spelling,
        # and 'orig_value' and 'origValue' converge (so do 'a_b' and 'a__b').
        # Measured on this repo before the check, BGL grade-1 G08 2026-09-30:
        # CounterfactualExplanation.to_db_row on a counterfactual whose
        # flipped_features held {"orig_value": 1, "origValue": 999} wrote
        # {"origValue": 999} into the JSONB payload. One of the two values
        # simply was not there, with no error and nothing in the row saying a
        # field had been lost, and flipped_features entries are caller-shaped
        # dicts (the dataclass types them list[dict[str, Any]]), so the
        # collision is reachable from outside. Refusing is the only honest
        # option here: a serialiser cannot choose which of the two the frontend
        # contract meant, and writing one of them is writing a row the producer
        # did not produce.
        out: dict[str, Any] = {}
        origin: dict[str, str] = {}
        for k, v in obj.items():
            ck = _snake_to_camel(k)
            if ck in out:
                raise ValueError(
                    "to_db_row cannot serialise this payload: the keys "
                    f"{origin[ck]!r} and {k!r} both become {ck!r} in camelCase, so one "
                    "of the two values would be dropped from the row without a trace. "
                    "Rename one of them before serialising."
                )
            origin[ck] = k
            out[ck] = _camel_dict(v)
        return out
    if isinstance(obj, list):
        return [_camel_dict(v) for v in obj]
    return obj


def _is_finite_number(value: Any) -> bool:
    """True only for a real, finite number.

    Used by the identity check to NAME which per_feature entries were not
    measured. A bare ``math.isfinite`` raises TypeError on None or a string,
    and an identity check that raises TypeError says nothing about the identity.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# Enumerations (kept as Literal for static-checker friendliness)

ExplainerMethod = Literal[
    "shap.TreeExplainer",
    "shap.LinearExplainer",
    "shap.KernelExplainer",
    "shap.DeepExplainer",
    "shap.GradientExplainer",
    "lime",
    "dice",
    "anchors",
    "ig",
    "sp-shap",
]

ExplainerScope = Literal["local", "global"]
AttributionUnits = Literal["log-odds", "probability", "raw"]
ModelType = Literal["tree", "linear", "deep", "blackbox"]
Audience = Literal["data_subject", "board", "audit", "engineer"]
PrivacyTier = Literal["k_anonymity", "differential_privacy", "exact"]
ProblemType = Literal["classification", "regression", "ranking"]
ExplainerGoal = Literal[
    "single_decision",
    "global_understanding",
    "proxy_diagnosis",
    "actionable_recourse",
    "rule_extraction",
]
FairnessMetric = Literal[
    "demographic_parity",
    "equalized_odds",
    "disparate_impact",
    "predictive_parity",
]
Verdict = Literal["Compliant", "Watchlist", "Non-compliant"]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# Explanation (TS Part 5.1)


@dataclass
class Attribution:
    feature: str
    contribution: float
    base_value: float | None = None


@dataclass
class Explanation:
    method: ExplainerMethod
    instance_id: str
    subject_id: str
    model_hash: str
    data_hash: str
    base_value: float
    prediction: float
    attributions: list[Attribution]
    units: AttributionUnits
    fidelity: float | None = None
    stability: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    view_provenance: list[str] = field(default_factory=lambda: ["xai"])
    cross_refs: list[str] = field(default_factory=list)
    audit_artifact_id: str | None = None
    # Explicit scope. When unset, to_db_row falls back to the legacy
    # instance_id heuristic; that heuristic mislabels global-batch rows
    # (explainers stamp instance_id='<subject>#row-i' on them), so
    # callers producing global explanations should set scope='global'.
    scope: ExplainerScope | None = None

    def _scope_for_row(self) -> str:
        """The scope column, and a warning when it is a GUESS rather than a record.

        BGL grade-1 G08, 2026-09-30. The column is NOT NULL on the frozen
        assessment-layer contract, so an unset ``scope`` has to be filled with
        something, and the fallback below is a guess about what the row means.
        On the shape the in-repo explainers actually produce it is the WRONG
        guess: ``TreeShapExplainer.explain_global`` and
        ``LinearShapExplainer.explain_global`` stamp
        ``instance_id='<subject>#row-<i>'`` on every row of a global batch and
        set no scope, so measured on this repo, each of those rows serialises
        with ``scope='local'``. A reader filtering the table for local
        explanations gets one row per training observation and reads them as
        individual decisions that were explained.

        The guess is kept (a NULL cannot be written and refusing would take the
        writer out with it) and it is no longer silent: a guessed scope now says
        so once per call site, naming the id it guessed from, which is the
        signal that was missing while the batch rows were being mislabelled. The
        durable fix is on the producers, which must pass ``scope='global'``.
        """
        if self.scope is not None:
            return self.scope
        guessed = "local" if self.instance_id else "global"
        warnings.warn(
            "Explanation.to_db_row: scope was not set, so the row is being written "
            f"with scope={guessed!r} GUESSED from instance_id={self.instance_id!r}. "
            "That is not a record of the explainer's scope. A global batch stamps a "
            "per-row instance_id and is guessed 'local' here; pass scope='global' "
            "(or scope='local') on the Explanation so the column states what was "
            "computed instead of what the id looks like.",
            RuntimeWarning,
            stacklevel=3,
        )
        return guessed

    def to_db_row(self, owner: str) -> dict[str, Any]:
        return {
            "owner": owner,
            "subject_id": self.subject_id,
            "method": self.method,
            "scope": self._scope_for_row(),
            "instance_id": self.instance_id,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "attributions": [_camel_dict(asdict(a)) for a in self.attributions],
            "units": self.units,
            "fidelity": self.fidelity,
            "stability": self.stability,
            "params": self.params,
            "library_versions": self.library_versions,
            "view_provenance": self.view_provenance,
            "audit_artifact_id": self.audit_artifact_id,
        }


@dataclass
class CounterfactualInstance:
    flipped_features: list[dict[str, Any]]
    predicted_outcome: str
    proximity: float
    feasibility: float


@dataclass
class CounterfactualExplanation(Explanation):
    counterfactuals: list[CounterfactualInstance] = field(default_factory=list)

    def to_db_row(self, owner: str) -> dict[str, Any]:
        row = super().to_db_row(owner)
        # CounterfactualInstance fields are snake_case (flipped_features,
        # predicted_outcome). Convert to camelCase for the wire contract.
        row["params"] = {
            **(row.get("params") or {}),
            "counterfactuals": [_camel_dict(asdict(c)) for c in self.counterfactuals],
        }
        return row


# XaiAssessment (TS Part 5.2)


@dataclass
class XaiDiagnostics:
    faithfulness: float | None
    stability: float | None
    # THREE states, never two: True the probe fired, False the probe ran and
    # found nothing, None the probe could not run. The default used to be
    # False, so an XaiDiagnostics built without it recorded the clean verdict
    # for a probe that may never have run: measured 2026-09-27,
    # XaiDiagnostics(faithfulness=0.8, stability=0.02) produced
    # {'faithfulness': 0.8, 'stability': 0.02, 'adversarial_flag': False, ...}
    # and SupabaseWriter.write_assessment asdict()s that straight into the row.
    # Both adversarial probes already return flag=None for the could-not-check
    # state (xai.diagnostics.adversarial and
    # evaluation.vfairness_metrics.explanation_diagnostics), so False as a
    # DEFAULT could only ever be an unrun probe wearing a clean verdict.
    adversarial_flag: bool | None = None
    adversarial_reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class XaiAssessment:
    id: str
    subject_id: str
    created_at: str
    explainer: ExplainerMethod
    scope: ExplainerScope
    explanations: list[Explanation]
    sp_shap_representatives: list[str] | None
    diagnostics: XaiDiagnostics
    audit_artifact_id: str
    view_provenance: list[str] = field(default_factory=lambda: ["xai"])


# FairnessDecomposition (TS Part 5.3)


@dataclass
class FairnessDecomposition:
    id: str
    subject_id: str
    metric: FairnessMetric
    protected_attribute: str
    total_disparity: float
    per_feature: dict[str, float]
    proxy_scores: dict[str, float]
    flagged_proxies: list[str]
    audit_artifact_id: str

    @property
    def identity_residual(self) -> float:
        """``|sum(per_feature) - total_disparity|``. Must be <= 1e-6."""
        return abs(sum(self.per_feature.values()) - self.total_disparity)

    def assert_identity(self, tolerance: float = 1e-6) -> None:
        """Lundberg identity check (TS Part 5.3 / Part 7.4).

        Fails loudly so the worker never writes a corrupt row.

        Three states, never two: the identity holds, the identity is broken, or
        it COULD NOT BE CHECKED. The third used to read as the first. Measured
        2026-09-27 with total_disparity=nan and
        per_feature={"income": nan, "zipcode": 11.0}: identity_residual was nan,
        ``nan > tolerance`` is False, so this returned None and to_db_row went
        on to emit {'total_disparity': nan, 'per_feature': {'income': nan,
        'zipcode': 11.0}, 'flagged_proxies': []}. The producer in
        xai.decomposition.shap_fairness refuses non-finite SHAP for exactly this
        reason and its comment names this hole ("assert_identity() could not
        catch it either"); this method is public and is the guard the worker
        relies on last, so it closes here too.
        """
        residual = self.identity_residual
        if not math.isfinite(residual):
            unmeasured = sorted(
                name for name, value in self.per_feature.items() if not _is_finite_number(value)
            )
            raise ValueError(
                "FairnessDecomposition identity COULD NOT BE CHECKED: "
                f"|sum(per_feature) - total_disparity| = {residual}, which is not a "
                "number, so neither a passing nor a failing identity was established. "
                f"Non-finite per_feature entries: {unmeasured or 'none'}; "
                f"total_disparity={self.total_disparity!r}. Refusing rather than "
                "returning silently, which would let the row be written with the "
                "unmeasured values still in it."
            )
        if residual > tolerance:
            raise ValueError(
                "FairnessDecomposition identity broken: "
                f"|sum(per_feature) - total_disparity| = {residual:.3e} "
                f"exceeds tolerance {tolerance:.0e}. "
                "Likely SHAP units do not match the metric units; convert before decomposing."
            )

    def to_db_row(self, owner: str) -> dict[str, Any]:
        self.assert_identity()
        return {
            "owner": owner,
            "subject_id": self.subject_id,
            "metric": self.metric,
            "protected_attribute": self.protected_attribute,
            "total_disparity": self.total_disparity,
            "per_feature": self.per_feature,
            "proxy_scores": self.proxy_scores,
            "flagged_proxies": self.flagged_proxies,
            "audit_artifact_id": self.audit_artifact_id,
        }


# TrustPosture (TS Part 5.4)


@dataclass
class ViewSection:
    view: str
    verdict: Verdict
    headline: str
    evidence_refs: list[str]


@dataclass
class ReconciledFinding:
    metric: FairnessMetric
    protected_attribute: str
    disparity: float
    top_driver: str
    top_driver_share: float
    narrative: str


@dataclass
class ConflictFlag:
    views: list[str]
    topic: str
    description: str


@dataclass
class TrustPosture:
    subject_id: str
    generated_at: str
    sections: dict[str, ViewSection]
    reconciled: list[ReconciledFinding]
    conflicts: list[ConflictFlag]
    global_trust_score: float
    global_verdict: Verdict
    evidence_refs: list[str]
    library_versions: dict[str, str]


# Recommendation (mirrors the TS Recommendation type)


@dataclass
class AdvisorWarning:
    severity: Literal["info", "warn", "block"]
    message: str
    citation: str | None = None


@dataclass
class AdvisorAlternative:
    explainer: ExplainerMethod
    rationale: str
    cost_note: str


@dataclass
class Recommendation:
    explainer: ExplainerMethod
    component_id: str
    surrogate: str | None
    config: dict[str, Any]
    alternatives: list[AdvisorAlternative]
    warnings: list[AdvisorWarning]
    confidence: float
    rationale: list[str]
    async_routing: bool
    kg_citations: list[str] = field(default_factory=list)
    kg_evidence_class: str | None = None
