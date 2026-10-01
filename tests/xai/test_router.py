"""
Router golden tests. The Python router, the TS recommendEngine and the
Postgres xai_recommend() function must all agree. Changing the routing
policy without updating all three is a contract drift bug.
"""

from __future__ import annotations

from vfairness.xai.explainers import route_explainer


def test_tree_routes_to_treeshap() -> None:
    d = route_explainer(model_type="tree")
    assert d.primary == "shap.TreeExplainer"
    assert d.component_id == "tree_shap"


def test_linear_routes_to_linearshap() -> None:
    d = route_explainer(model_type="linear")
    assert d.primary == "shap.LinearExplainer"
    assert d.component_id == "linear_explainer"


def test_deep_routes_to_integrated_gradients() -> None:
    # DeepExplainer/GradientExplainer are Phase 2 (not implemented); routing
    # to them made every deep job fail at get_explainer. The router must pick
    # an IMPLEMENTED primary: Integrated Gradients.
    d = route_explainer(model_type="deep")
    assert d.primary == "ig"
    assert "shap.KernelExplainer" in d.fallbacks


def test_blackbox_with_background_routes_to_kernelshap() -> None:
    d = route_explainer(model_type="blackbox", background_available=True, ood_risk="low")
    assert d.primary == "shap.KernelExplainer"
    assert d.downgraded_from is None


def test_blackbox_without_background_downgrades_to_lime() -> None:
    d = route_explainer(model_type="blackbox", background_available=False)
    assert d.primary == "lime"
    assert d.downgraded_from == "shap.KernelExplainer"


def test_high_ood_risk_downgrades_to_lime() -> None:
    d = route_explainer(model_type="blackbox", background_available=True, ood_risk="high")
    assert d.primary == "lime"
    assert d.downgraded_from == "shap.KernelExplainer"


def test_actionable_recourse_picks_dice() -> None:
    d = route_explainer(model_type="tree", goal="actionable_recourse")
    assert d.primary == "dice"
    assert d.component_id == "dice_counterfactuals"


def test_counterfactual_needed_picks_dice() -> None:
    d = route_explainer(model_type="linear", counterfactual_needed=True)
    assert d.primary == "dice"
