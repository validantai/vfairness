"""
vfairness.xai
=============

Explainability subsystem for the vfairness library. Implements the XAI
subgraph specified by the consuming platform's internal XAI design note,
which is not published with this library: SHAP family adapters, SP-SHAP,
the Lundberg fairness decomposition,
DiCE counterfactuals, faithfulness/stability diagnostics, plus the
storage + worker glue that writes results back into the Supabase
assessment-layer tables (``xai_explanations``,
``xai_fairness_decompositions``, ``xai_audit_artifacts``,
``xai_trust_postures``, ``xai_jobs``).

The data contracts here (Explanation, XaiAssessment, FairnessDecomposition,
TrustPosture) mirror TypeScript contracts the consuming platform keeps in
its own, unpublished frontend sources. Treat them as frozen interfaces;
bumping either side requires updating both.

The 1e-6 Lundberg identity is asserted before any FairnessDecomposition
is persisted; runs that violate it fail loudly rather than write a
corrupt row.
"""

from __future__ import annotations

from .decomposition.shap_fairness import lundberg_fairness_decomposition
from .explainers.router import route_explainer
from .schemas import (
    Attribution,
    AttributionUnits,
    CounterfactualExplanation,
    CounterfactualInstance,
    ExplainerGoal,
    ExplainerMethod,
    ExplainerScope,
    Explanation,
    FairnessDecomposition,
    FairnessMetric,
    ModelType,
    Recommendation,
    TrustPosture,
    XaiAssessment,
    XaiDiagnostics,
)

__all__ = [
    "Attribution",
    "AttributionUnits",
    "CounterfactualExplanation",
    "CounterfactualInstance",
    "Explanation",
    "ExplainerGoal",
    "ExplainerMethod",
    "ExplainerScope",
    "FairnessDecomposition",
    "FairnessMetric",
    "ModelType",
    "Recommendation",
    "TrustPosture",
    "XaiAssessment",
    "XaiDiagnostics",
    "route_explainer",
    "lundberg_fairness_decomposition",
]

__version__ = "0.0.9-scaffold"
