"""
vfairness.xai.explainers.router
===============================

Mechanical model-type -> explainer routing.

This mirrors two implementations the consuming platform keeps in its own,
unpublished sources: a TypeScript reference engine, and the Postgres
``public.xai_recommend()`` function declared in its schema migrations. The
three must agree; changing one without the others is a contract drift bug.

Routing rules (Part 7.1 router table of the XAI implementation plan):

| Model type | Primary                | Fallbacks         |
|------------|------------------------|-------------------|
| tree       | shap.TreeExplainer     | KernelSHAP        |
| linear     | shap.LinearExplainer   | KernelSHAP        |
| deep       | Integrated Gradients   | KernelSHAP        |
| blackbox   | shap.KernelExplainer   | LIME, then Anchors|

The ``deep`` row said "DeepExplainer, fallback Integrated Gradients (P2)"
until 2026-09-09 and had it backwards: DeepExplainer and GradientExplainer
are Phase 2 and NOT implemented, so Integrated Gradients is the primary and
the only thing that runs. The router's own ``reason`` string said so while
this table said otherwise. ``test_the_routing_table_matches_the_router``
now reads this table and compares it against ``route_explainer`` for every
model type, so the two cannot drift apart again.

Goal-driven override: when goal == "actionable_recourse" or
counterfactual_needed is True, DiCE takes the slot.

Data-availability gate: when background is scarce or out-of-distribution,
KernelSHAP downgrades to LIME with a wider kernel.

The preference order above lives HERE and nowhere else: it cannot be
derived from ``ExplainerCapabilities`` (five adapters declare ``tree``
and the dataclass carries no priority; KernelSHAP and LIME both declare
``requires_background=True``, so the downgrade is not derivable either).
The one thing the router does read from an adapter is
``capabilities.component_id``, via ``_component_id_for``, so the
catalogue id on a RouteDecision comes from the adapter that will run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ExplainerGoal, ExplainerMethod, ModelType
from .registry import _REGISTRY


@dataclass(frozen=True)
class RouteDecision:
    """The router's decision for one (data, use_case) pair."""

    primary: ExplainerMethod
    component_id: str
    fallbacks: tuple[ExplainerMethod, ...]
    downgraded_from: ExplainerMethod | None = None
    reason: str = ""


def _component_id_for(primary: ExplainerMethod) -> str:
    """Return the catalogue component id the adapter for *primary* declares.

    The id used to be a literal repeated in every branch below, so it could
    drift from the adapter that actually runs the job. Reading it off
    ``ExplainerCapabilities.component_id`` keeps one source of truth: change
    the adapter and the RouteDecision follows.
    """
    cls = _REGISTRY.get(primary)
    if cls is None:
        raise KeyError(
            f"route_explainer selected {primary!r}, which has no adapter in the "
            f"explainer registry, so its catalogue component id is unknown."
        )
    caps = cls.capabilities
    if caps.method != primary:
        raise ValueError(
            f"Explainer registry is inconsistent: {primary!r} maps to "
            f"{cls.__name__}, which declares capabilities.method={caps.method!r}."
        )
    if not caps.component_id:
        raise ValueError(
            f"{cls.__name__} declares no capabilities.component_id, so the "
            f"routing decision for {primary!r} has no catalogue component."
        )
    return caps.component_id


def _decision(
    primary: ExplainerMethod,
    *,
    fallbacks: tuple[ExplainerMethod, ...],
    reason: str,
    downgraded_from: ExplainerMethod | None = None,
) -> RouteDecision:
    """Build a RouteDecision, taking its component id from *primary*'s adapter."""
    return RouteDecision(
        primary=primary,
        component_id=_component_id_for(primary),
        fallbacks=fallbacks,
        downgraded_from=downgraded_from,
        reason=reason,
    )


def route_explainer(
    *,
    model_type: ModelType,
    goal: ExplainerGoal = "global_understanding",
    counterfactual_needed: bool = False,
    background_available: bool = True,
    ood_risk: str = "low",
) -> RouteDecision:
    """Return the routing decision for one (model_type, goal) pair.

    Mirrors the Postgres ``xai_recommend()`` routing exactly; tests in
    ``tests/xai/test_router.py`` lock them together.

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

    Ledger row: xai_explainer_routing. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    # Goal-driven override comes first because actionable_recourse is the
    # only goal that overrides the model-type primary unconditionally.
    if counterfactual_needed or goal == "actionable_recourse":
        return _decision(
            "dice",
            fallbacks=("shap.TreeExplainer",),
            reason="Actionable recourse goal selects DiCE; SHAP runs as supporting evidence.",
        )

    # Model-type primary.
    if model_type == "tree":
        return _decision(
            "shap.TreeExplainer",
            fallbacks=("shap.KernelExplainer",),
            reason="TreeSHAP exact, synchronous on the tree family.",
        )
    if model_type == "linear":
        return _decision(
            "shap.LinearExplainer",
            fallbacks=("shap.KernelExplainer",),
            reason="Closed-form LinearExplainer; the cheapest path.",
        )
    if model_type == "deep":
        # shap.DeepExplainer / GradientExplainer are specified but NOT yet
        # implemented (registry._NOT_YET_IMPLEMENTED): routing to them made
        # every deep-model job fail at get_explainer with
        # NotImplementedError. Integrated Gradients is the implemented
        # gradient-based primary; KernelSHAP is the model-agnostic fallback.
        return _decision(
            "ig",
            fallbacks=("shap.KernelExplainer",),
            reason=(
                "Integrated Gradients for deep models (DeepExplainer/"
                "GradientExplainer are Phase 2 and not yet implemented)."
            ),
        )

    # blackbox: KernelSHAP unless background-availability gate fires.
    if not background_available or ood_risk == "high":
        return _decision(
            "lime",
            fallbacks=("shap.KernelExplainer",),
            downgraded_from="shap.KernelExplainer",
            reason=(
                "Background scarce / out-of-distribution. KernelSHAP "
                "downgraded to LIME with a wider kernel (Slack et al. 2020 caveat)."
            ),
        )
    return _decision(
        "shap.KernelExplainer",
        fallbacks=("lime", "anchors"),
        reason="Black-box; KernelSHAP runs with LIME as a parallel fallback.",
    )
