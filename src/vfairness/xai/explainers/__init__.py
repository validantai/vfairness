"""
vfairness.xai.explainers
========================

Explainer adapters that normalise each library's native output into the
Explanation contract from ``vfairness.xai.schemas``.

The TS reference engine is mirrored here so the Postgres
``xai_recommend()`` function, the TS ``recommend()`` and this Python
router all agree on routing decisions per (model_type, goal, data
availability).
"""

from __future__ import annotations

from .anchors_adapter import AnchorsExplainer
from .base import Explainer, ExplainerCapabilities
from .dice_adapter import DiceCounterfactualExplainer
from .ig_adapter import IntegratedGradientsExplainer
from .lime_adapter import LimeExplainer
from .registry import available_methods, get_explainer
from .router import route_explainer
from .shap_adapter import KernelShapExplainer, LinearShapExplainer, TreeShapExplainer

__all__ = [
    "Explainer",
    "ExplainerCapabilities",
    "route_explainer",
    "TreeShapExplainer",
    "LinearShapExplainer",
    "KernelShapExplainer",
    "DiceCounterfactualExplainer",
    "IntegratedGradientsExplainer",
    "LimeExplainer",
    "AnchorsExplainer",
    "get_explainer",
    "available_methods",
]
