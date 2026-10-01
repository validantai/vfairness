"""
vfairness.xai.explainers.registry
==================================

Maps an :data:`~vfairness.xai.schemas.ExplainerMethod` string (what
``route_explainer`` returns) to the concrete adapter class that
implements it. This is the seam the worker uses to turn a routing
decision into a runnable explainer.

Importing this module is cheap: adapter classes defer their heavy
optional imports (shap / dice-ml / captum / torch) to ``__init__``, so
the registry can be built without any of those extras installed. The
``ImportError`` only fires when you actually instantiate an adapter
whose library is missing.

Methods that are specified but not yet implemented (lime, anchors,
sp-shap, sage, ALE, permutation, the SHAP deep/gradient explainers)
raise a clear ``NotImplementedError`` naming the backlog item, rather
than a bare ``KeyError``, so a premature route surfaces as an
actionable message.
"""

from __future__ import annotations

from ..schemas import ExplainerMethod
from .anchors_adapter import AnchorsExplainer
from .base import Explainer
from .dice_adapter import DiceCounterfactualExplainer
from .ig_adapter import IntegratedGradientsExplainer
from .lime_adapter import LimeExplainer
from .shap_adapter import KernelShapExplainer, LinearShapExplainer, TreeShapExplainer

# method string -> adapter class (instantiated lazily by get_explainer).
_REGISTRY: dict[ExplainerMethod, type[Explainer]] = {
    "shap.TreeExplainer": TreeShapExplainer,
    "shap.LinearExplainer": LinearShapExplainer,
    "shap.KernelExplainer": KernelShapExplainer,
    "dice": DiceCounterfactualExplainer,
    "ig": IntegratedGradientsExplainer,
    "lime": LimeExplainer,
    "anchors": AnchorsExplainer,
}

# Specified in the contract but not yet implemented; map to the backlog
# id so a premature route is self-explaining.
_NOT_YET_IMPLEMENTED: dict[str, str] = {
    "sp-shap": "#P1-11 SP-SHAP submodular pick",
    "shap.DeepExplainer": "SHAP DeepExplainer (Phase 2)",
    "shap.GradientExplainer": "SHAP GradientExplainer (Phase 2)",
}


def get_explainer(method: ExplainerMethod) -> Explainer:
    """Instantiate the adapter for *method*.

    Raises ``NotImplementedError`` for a specified-but-unbuilt method and
    ``KeyError`` for an unknown one. The adapter's own ``__init__`` raises
    ``ImportError`` if its optional library is not installed.
    """
    cls = _REGISTRY.get(method)
    if cls is not None:
        return cls()
    if method in _NOT_YET_IMPLEMENTED:
        raise NotImplementedError(
            f"Explainer method {method!r} is specified but not yet implemented "
            f"(backlog {_NOT_YET_IMPLEMENTED[method]}). The router should not "
            f"select it as a primary until the adapter lands."
        )
    raise KeyError(f"Unknown explainer method: {method!r}")


def available_methods() -> tuple[ExplainerMethod, ...]:
    """Return the methods that have a runnable adapter."""
    return tuple(_REGISTRY.keys())
