"""Audit 6, wave 4, lane C. S-06: the XAI explainer capability registry.

``ExplainerCapabilities`` used to say it was "Used by ``route_explainer`` to
pick the right adapter", while ``route_explainer`` was an if-ladder that read
none of its fields: a 240-combination sweep with a recording proxy installed
over every registered adapter measured ZERO capability field reads.

The fix keeps the preference order in ``router.py`` (it is not derivable from
the dataclass: five adapters declare ``tree`` and there is no priority field;
KernelSHAP and LIME both declare ``requires_background=True``, so the
background downgrade is not derivable either) and wires the one field that
maps exactly: the catalogue ``component_id``, which the router used to repeat
as a literal in all six branches.

These tests pin, in order:
  * that every routing decision is UNCHANGED by the refactor (measured
    primary / component_id / fallbacks / downgraded_from for all 240 combos),
  * that ``component_id`` now comes from the adapter and not from a literal,
  * that the fields the docstring calls descriptive really are never read,
  * that an unresolvable primary is refused loudly instead of silently
    getting a blank or wrong catalogue id,
  * that ``Explainer.supports`` really has no call site, as its docstring
    now says.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import pathlib
from typing import Any

import pytest

from vfairness.xai.explainers import route_explainer
from vfairness.xai.explainers.base import Explainer, ExplainerCapabilities
from vfairness.xai.explainers.registry import _REGISTRY
from vfairness.xai.explainers.router import _component_id_for
from vfairness.xai.schemas import ExplainerGoal, ExplainerMethod, ModelType

MODEL_TYPES: tuple[ModelType, ...] = ("tree", "linear", "deep", "blackbox")
GOALS: tuple[ExplainerGoal, ...] = (
    "single_decision",
    "global_understanding",
    "proxy_diagnosis",
    "actionable_recourse",
    "rule_extraction",
)
OOD_RISKS = ("low", "medium", "high")
CROSS: list[tuple[ModelType, ExplainerGoal, bool, bool, str]] = list(
    itertools.product(MODEL_TYPES, GOALS, (False, True), (True, False), OOD_RISKS)
)

# The routing table from the router's own module docstring, written out here
# independently of the implementation. Values are the measured catalogue ids
# the Postgres xai_recommend() and the TS recommendEngine must also produce.
_TREE = ("shap.TreeExplainer", "tree_shap", ("shap.KernelExplainer",), None)
_LINEAR = ("shap.LinearExplainer", "linear_explainer", ("shap.KernelExplainer",), None)
_DEEP = ("ig", "integrated_gradients", ("shap.KernelExplainer",), None)
_KERNEL = ("shap.KernelExplainer", "kernel_shap", ("lime", "anchors"), None)
_LIME = ("lime", "lime", ("shap.KernelExplainer",), "shap.KernelExplainer")
_DICE = ("dice", "dice_counterfactuals", ("shap.TreeExplainer",), None)


def _expected(
    model_type: str, goal: str, counterfactual: bool, background: bool, ood: str
) -> tuple[str, str, tuple[str, ...], str | None]:
    if counterfactual or goal == "actionable_recourse":
        return _DICE
    if model_type == "tree":
        return _TREE
    if model_type == "linear":
        return _LINEAR
    if model_type == "deep":
        return _DEEP
    if not background or ood == "high":
        return _LIME
    return _KERNEL


class _Recorder:
    """Proxy over a frozen ExplainerCapabilities that logs every field read."""

    def __init__(self, real: ExplainerCapabilities, log: list[str]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, item: str) -> Any:
        if not item.startswith("__"):
            object.__getattribute__(self, "_log").append(item)
        return getattr(object.__getattribute__(self, "_real"), item)


def _sweep() -> dict[tuple[Any, ...], tuple[Any, ...]]:
    out = {}
    for model_type, goal, cf, bg, ood in CROSS:
        d = route_explainer(
            model_type=model_type,
            goal=goal,
            counterfactual_needed=cf,
            background_available=bg,
            ood_risk=ood,
        )
        out[(model_type, goal, cf, bg, ood)] = (
            d.primary,
            d.component_id,
            d.fallbacks,
            d.downgraded_from,
        )
    return out


# ── over-correction control: routing must be IDENTICAL after the refactor ──


def test_full_cross_product_routing_is_unchanged() -> None:
    """All 240 (model_type, goal, cf, background, ood) decisions, by value."""
    measured = _sweep()
    assert len(measured) == 240
    for key, got in measured.items():
        assert got == _expected(*key), f"routing changed for {key}: {got}"

    # Spot pins on the literal catalogue ids, so a "fix" that resolves the id
    # from the WRONG adapter cannot pass by being merely self-consistent.
    assert route_explainer(model_type="tree").component_id == "tree_shap"
    assert route_explainer(model_type="linear").component_id == "linear_explainer"
    assert route_explainer(model_type="deep").component_id == "integrated_gradients"
    assert route_explainer(model_type="blackbox").component_id == "kernel_shap"
    assert route_explainer(model_type="blackbox", background_available=False).component_id == "lime"
    assert (
        route_explainer(model_type="tree", goal="actionable_recourse").component_id
        == "dice_counterfactuals"
    )


# ── refusal pin: the component id must come from the adapter, not a literal ─

_PRIMARY_CASES: list[tuple[dict[str, Any], ExplainerMethod, str]] = [
    ({"model_type": "tree"}, "shap.TreeExplainer", "tree_shap"),
    ({"model_type": "linear"}, "shap.LinearExplainer", "linear_explainer"),
    ({"model_type": "deep"}, "ig", "integrated_gradients"),
    ({"model_type": "blackbox"}, "shap.KernelExplainer", "kernel_shap"),
    ({"model_type": "blackbox", "background_available": False}, "lime", "lime"),
    ({"model_type": "tree", "goal": "actionable_recourse"}, "dice", "dice_counterfactuals"),
]


@pytest.mark.parametrize("kwargs,method,default_id", _PRIMARY_CASES)
def test_component_id_follows_the_adapter_registry(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    method: ExplainerMethod,
    default_id: str,
) -> None:
    """Re-declare the adapter's component id; the decision must follow it.

    With the hardcoded literal reinstated the router keeps answering the old
    id, which is the S-06 defect: a RouteDecision naming a catalogue
    component the adapter that runs the job no longer claims.
    """
    cls = _REGISTRY[method]
    assert route_explainer(**kwargs).component_id == default_id  # control

    patched = dataclasses.replace(cls.capabilities, component_id=default_id + "_v2")
    monkeypatch.setattr(cls, "capabilities", patched)
    assert route_explainer(**kwargs).component_id == default_id + "_v2"


# ── claims-vs-code pin: the descriptive fields really are never read ────────

_LOAD_BEARING = {"method", "component_id"}
_DESCRIPTIVE = {
    "supported_model_types",
    "supports_local",
    "supports_global",
    "is_async_eligible",
    "requires_background",
}


def test_capability_fields_read_by_the_router_match_the_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure which fields the router touches over the whole cross product."""
    log: list[str] = []
    for cls in _REGISTRY.values():
        monkeypatch.setattr(cls, "capabilities", _Recorder(cls.capabilities, log))

    measured = _sweep()
    assert len(measured) == 240
    read = set(log)

    # The router reads exactly the two fields base.py calls load-bearing.
    assert read == _LOAD_BEARING, f"router read {sorted(read)}"
    assert len(log) > 0, "no capability field read at all: the registry is decorative again"

    # And every field base.py calls descriptive is untouched by routing.
    assert read & _DESCRIPTIVE == set()

    doc = ExplainerCapabilities.__doc__ or ""
    assert "DESCRIPTIVE metadata" in doc, "the docstring must not imply routing reads them"
    for field in _DESCRIPTIVE:
        assert field in doc, f"{field} is unread and must be named as descriptive"
    for field in sorted(_LOAD_BEARING | _DESCRIPTIVE):
        assert field in {f.name for f in dataclasses.fields(ExplainerCapabilities)}


# ── refusal pins: an unresolvable primary is refused, never silently blank ──


def test_primary_with_no_adapter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _component_id_for("shap.TreeExplainer") == "tree_shap"  # control
    monkeypatch.delitem(_REGISTRY, "shap.TreeExplainer")
    with pytest.raises(KeyError, match="no adapter in the explainer registry"):
        route_explainer(model_type="tree")


def test_registry_key_disagreeing_with_the_adapter_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _REGISTRY["lime"]
    assert _component_id_for("lime") == "lime"  # control
    monkeypatch.setattr(cls, "capabilities", dataclasses.replace(cls.capabilities, method="dice"))
    with pytest.raises(ValueError, match="registry is inconsistent"):
        route_explainer(model_type="blackbox", background_available=False)


def test_blank_component_id_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _REGISTRY["shap.KernelExplainer"]
    assert _component_id_for("shap.KernelExplainer") == "kernel_shap"  # control
    monkeypatch.setattr(cls, "capabilities", dataclasses.replace(cls.capabilities, component_id=""))
    with pytest.raises(ValueError, match="no capabilities.component_id"):
        route_explainer(model_type="blackbox")


def test_every_registered_adapter_resolves_its_own_component_id() -> None:
    """Over-correction control: the seven registered adapters all resolve."""
    resolved = {m: _component_id_for(m) for m in _REGISTRY}
    assert resolved == {
        "shap.TreeExplainer": "tree_shap",
        "shap.LinearExplainer": "linear_explainer",
        "shap.KernelExplainer": "kernel_shap",
        "dice": "dice_counterfactuals",
        "ig": "integrated_gradients",
        "lime": "lime",
        "anchors": "anchors",
    }


# ── sibling: Explainer.supports() says NOT WIRED, so it must stay unwired ───


def test_supports_has_no_call_site_in_vfairness() -> None:
    """base.py used to say "the router calls this on every candidate adapter".

    Nothing calls it. The docstring now says so; this pins the claim, so a
    future wiring has to update the docstring with it.
    """
    doc = Explainer.supports.__doc__ or ""
    assert "NOT WIRED" in doc

    call_sites = []
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "vfairness"
    files = list(src.rglob("*.py"))
    assert len(files) > 100, f"source tree not found at {src}"
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "supports"
            ):
                call_sites.append(f"{path}:{node.lineno}")
    assert call_sites == [], f"supports() is wired now; update its docstring: {call_sites}"


class TestTheRoutingTableIsNotProse:
    """The module docstring carries a routing table. It was WRONG for the deep
    row (it named DeepExplainer, which is Phase 2 and never runs, while the
    router routes to Integrated Gradients), and nothing checked it. A table a
    reader trusts is a claim like any other, so it is now derived from the
    router rather than maintained beside it."""

    @staticmethod
    def _table_rows():
        import re

        from vfairness.xai.explainers import router as router_mod

        rows = {}
        for line in (router_mod.__doc__ or "").splitlines():
            m = re.match(r"\|\s*(tree|linear|deep|blackbox)\s*\|([^|]+)\|([^|]+)\|", line.strip())
            if m:
                rows[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
        return rows

    def test_the_table_lists_every_model_type(self):
        assert set(self._table_rows()) == {"tree", "linear", "deep", "blackbox"}

    @pytest.mark.parametrize("model_type", ["tree", "linear", "deep", "blackbox"])
    def test_the_routing_table_matches_the_router(self, model_type):
        from vfairness.xai.explainers.router import route_explainer

        primary_text, fallback_text = self._table_rows()[model_type]
        decision = route_explainer(model_type=model_type)
        # The table is prose, so compare on the distinctive token rather than
        # on an exact string: the point is that it cannot name a DIFFERENT
        # explainer from the one that runs.
        head = decision.primary.split(".")[-1].replace("Explainer", "").lower()
        assert head in primary_text.lower().replace(" ", "") or (
            decision.component_id.split("_")[0] in primary_text.lower()
        ), f"table says {primary_text!r}, router returns {decision.primary!r}"
        for fb in decision.fallbacks:
            token = fb.split(".")[-1].replace("Explainer", "").lower()
            assert token in fallback_text.lower(), f"table fallbacks {fallback_text!r} omit {fb!r}"

    def test_the_deep_row_does_not_name_an_unimplemented_explainer(self):
        """The specific regression: DeepExplainer is Phase 2 and never runs."""
        primary_text, _ = self._table_rows()["deep"]
        assert "deepexplainer" not in primary_text.lower().replace(" ", "")
