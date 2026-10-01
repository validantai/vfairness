"""
vfairness.xai.explainers.base
=============================

ABC for explainer adapters. Each concrete adapter wraps one underlying
library (shap, lime, dice) and normalises its native output into the
``Explanation`` contract from ``vfairness.xai.schemas``.

Concrete adapters MUST:
* implement ``supports(model, model_type)``. NOTE: no dispatch path calls
  it today (see ``Explainer.supports``); it is a pre-flight check for a
  caller holding the model object, not the router's selection step.
* implement ``explain_local()`` and ``explain_global()`` returning
  ``Explanation`` / ``list[Explanation]`` respectively.
* record their library version in ``library_versions`` on every
  returned Explanation, so the AuditArtifact can replay byte-stably.

The base class does NOT import shap, lime or dice; concrete adapters
declare those as optional extras. This keeps a base install lightweight
and lets the package fall back gracefully when an adapter is missing.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schemas import ExplainerMethod, Explanation, ModelType


def prediction_fn_available(
    model: Any,
    *,
    where: str,
    accept_predict_proba: bool = True,
) -> bool:
    """Whether *model* can be called as a prediction function RIGHT NOW.

    The shared precondition of every model-agnostic adapter here (KernelSHAP,
    LIME, DiCE, Anchors), decided in ONE place because each of the four
    answered it with its own copy of ``callable(model) or hasattr(model,
    "predict")`` and every copy was wrong in the same direction.

    ``hasattr(model, "predict")`` is TRUE FOR AN UNFITTED sklearn estimator:
    the method exists, and calling it raises. Measured 2026-09-30, before this
    helper existed::

        KernelShapExplainer().supports(RandomForestClassifier(), "blackbox")  -> True
        LimeExplainer().supports(RandomForestClassifier(), "blackbox")        -> True
        DiceCounterfactualExplainer().supports(...)                           -> True
        AnchorsExplainer().supports(...)                                      -> True

    and the ``explain_local`` that followed on the same object died inside shap
    with "Provided model function fails when applied to the provided data set"
    and ``TypeError: 'RandomForestClassifier' object is not callable``: the
    failure-far-from-the-cause that the sibling ``TreeShapExplainer.supports``
    fix in this package was written against. True is the DANGEROUS direction
    for this predicate, because the base class documents ``supports`` as the
    guard a caller runs BEFORE explaining. ``TreeShapExplainer`` and
    ``LinearShapExplainer`` already declined an unfitted model, as a side
    effect of needing ``estimators_`` / ``coef_``, so the four agnostic
    adapters disagreed with their own siblings in the same file.

    Fitted-ness is read off the sklearn convention that a fitted estimator
    carries at least one public attribute ending in a single underscore, which
    is the rule ``sklearn.utils.validation.check_is_fitted`` itself uses. No
    import is needed, so this stays as cheap as the base class asks, and ONLY
    an object that declares itself an estimator (both ``get_params`` and
    ``fit``) is judged that way: a plain callable, a bound ``predict_proba``,
    an xgboost booster or a custom predictor has no ``get_params`` and reaches
    the same answer as before.

    A False for an unfitted estimator is a determination, not a
    could-not-check, and it is still SAID OUT LOUD, because a bare False is
    indistinguishable from "your model is the wrong kind" and only one of those
    is something the caller can fix. Same reasoning as
    ``IntegratedGradientsExplainer.supports`` on a missing torch.
    """
    has_interface = callable(model) or hasattr(model, "predict")
    if accept_predict_proba:
        has_interface = has_interface or hasattr(model, "predict_proba")
    if not has_interface:
        return False
    if hasattr(model, "get_params") and hasattr(model, "fit"):
        state = getattr(model, "__dict__", None) or {}
        fitted = [k for k in state if k.endswith("_") and not k.startswith("__")]
        if not fitted:
            warnings.warn(
                f"{where}: {type(model).__name__} exposes a prediction method but carries "
                f"no fitted state (no public attribute ending in '_', the same test "
                f"sklearn's check_is_fitted uses), so it CANNOT be called as a prediction "
                f"function yet and this adapter cannot explain it. Returning False. This "
                f"is a not-fitted-yet, not a wrong kind of model: fit it and ask again.",
                UserWarning,
                stacklevel=3,
            )
            return False
    return True


@dataclass(frozen=True)
class ExplainerCapabilities:
    """Describes what an adapter can handle.

    Two fields are load-bearing. ``route_explainer`` resolves
    ``component_id`` through the registry (``router._component_id_for``),
    so the catalogue id on a RouteDecision comes from the adapter that
    will run the job, and ``method`` is checked there against the key the
    adapter is registered under.

    The rest are DESCRIPTIVE metadata: ``supported_model_types``,
    ``supports_local``, ``supports_global``, ``is_async_eligible`` and
    ``requires_background`` document the adapter and no dispatch path
    consults them. The router does NOT pick from them and cannot: five
    adapters declare ``tree``, the dataclass carries no priority, and both
    KernelSHAP and LIME declare ``requires_background=True``, so neither
    the primary nor the background downgrade is derivable here. The
    preference order lives in ``router.py``. Changing a descriptive field
    documents the adapter; it does not change which adapter runs.
    """

    method: ExplainerMethod
    supported_model_types: tuple[ModelType, ...]
    supports_local: bool = True
    supports_global: bool = True
    is_async_eligible: bool = False
    requires_background: bool = False
    component_id: str = ""  # natural id in catalog_algorithmic_components


class Explainer(ABC):
    """Abstract base for one explainer adapter."""

    capabilities: ExplainerCapabilities

    @abstractmethod
    def supports(self, model: Any, model_type: ModelType) -> bool:
        """Return True if this adapter can explain *model* on its own.

        Implementations should be cheap (e.g. ``isinstance``).

        NOT WIRED: nothing in ``vfairness`` calls this. ``route_explainer``
        selects on ``model_type`` alone and never sees the model object;
        ``worker.runner`` resolves the routed method through
        ``get_explainer`` without consulting this method (its execute step
        is still a stub). An adapter answering False is therefore never
        filtered out by the package: call it yourself before
        ``explain_local`` / ``explain_global`` if you need that guard.
        """

    @abstractmethod
    def explain_local(
        self,
        model: Any,
        x: np.ndarray,
        background: np.ndarray | None = None,
        *,
        instance_id: str,
        subject_id: str,
        model_hash: str,
        data_hash: str,
        **params: Any,
    ) -> Explanation:
        """Return a per-instance Explanation."""

    @abstractmethod
    def explain_global(
        self,
        model: Any,
        X: np.ndarray,
        background: np.ndarray | None = None,
        *,
        subject_id: str,
        model_hash: str,
        data_hash: str,
        **params: Any,
    ) -> list[Explanation]:
        """Return one Explanation per row in X."""
