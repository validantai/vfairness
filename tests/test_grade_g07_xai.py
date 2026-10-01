"""G07 grading pins: vfairness.xai.explainers.

Two defects were measured here on 2026-09-30 and both are pinned below with
their healthy control beside them, because a guard that refuses everything
passes every refusal test and destroys the unit.

1. A MATRIX PUBLISHED AS ONE INSTANCE. Every ``explain_local`` in
   ``shap_adapter`` reached its numbers through
   ``np.asarray(shap_values).flatten()``, and flatten does not know about rows.
   ``LinearShapExplainer.explain_local`` with ``x`` of shape (5, 3) against a
   3-feature LogisticRegression and no ``feature_names`` returned ONE
   Explanation carrying 15 attributions named ``f0 .. f14``, the row axis
   interleaved into the feature axis, so row 0's value on the first feature
   (8.75575) and row 1's value on the SAME feature (2.44833) were published as
   two different features of one instance, with
   ``params['attributions_complete']`` True. The same call with ``x`` of shape
   (0, 3) returned ``attributions=[]`` with ``attributions_complete`` True: a
   complete-looking explanation of an instance that was never examined, the
   BGL3 xai-1 shape that ``TreeShapExplainer`` already refused and the other
   two adapters did not.

2. A PRE-FLIGHT CHECK THAT SAYS YES TO AN UNFITTED MODEL. The four
   model-agnostic adapters answered ``supports`` with ``callable(model) or
   hasattr(model, "predict")``, and ``hasattr`` is True for an unfitted sklearn
   estimator: the method exists, calling it raises. All four returned True for
   ``RandomForestClassifier()``, while ``TreeShapExplainer`` and
   ``LinearShapExplainer`` in the same file declined it. The subsequent
   ``explain_local`` died inside shap with "Provided model function fails when
   applied to the provided data set", the failure-far-from-the-cause that the
   sibling supports() fix was written against. True is the dangerous direction
   for a predicate the base class documents as the guard a caller runs BEFORE
   explaining.

The remaining tests execute the thin units in this batch (the ABC, the two
frozen dataclasses, the registry, the refusals of the two local-only adapters)
to establish that none of them quietly mints a value.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("shap")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from vfairness.xai.explainers.anchors_adapter import AnchorsExplainer  # noqa: E402
from vfairness.xai.explainers.base import (  # noqa: E402
    Explainer,
    ExplainerCapabilities,
    prediction_fn_available,
)
from vfairness.xai.explainers.dice_adapter import DiceCounterfactualExplainer  # noqa: E402
from vfairness.xai.explainers.ig_adapter import IntegratedGradientsExplainer  # noqa: E402
from vfairness.xai.explainers.lime_adapter import LimeExplainer  # noqa: E402
from vfairness.xai.explainers.registry import (  # noqa: E402
    _NOT_YET_IMPLEMENTED,
    _REGISTRY,
    available_methods,
    get_explainer,
)
from vfairness.xai.explainers.router import RouteDecision  # noqa: E402
from vfairness.xai.explainers.shap_adapter import (  # noqa: E402
    LOCAL_ACCURACY_TOL,
    KernelShapExplainer,
    LinearShapExplainer,
    TreeShapExplainer,
)

KW = dict(subject_id="s", model_hash="m", data_hash="d")


@pytest.fixture(scope="module")
def linear_setup():
    """A logistic model with ONE dominant coefficient, so the control can be derived.

    Nothing below quotes a number from this fixture: the expected dominant
    feature is read off ``model.coef_`` at assert time.
    """
    rs = np.random.RandomState(7)
    X = rs.normal(size=(200, 3))
    y = (X[:, 0] * 3 + rs.normal(size=200) * 0.2 > 0).astype(int)
    return LogisticRegression().fit(X, y), X


@pytest.fixture(scope="module")
def tree_setup():
    rs = np.random.RandomState(3)
    X = rs.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    return RandomForestClassifier(n_estimators=6, random_state=0).fit(X, y), X


# === 1. a matrix is not one instance =======================================


def test_linear_shap_explain_local_refuses_a_matrix(linear_setup):
    """DEFECT CASE. x of shape (5, 3) returned 15 attributions f0..f14."""
    model, X = linear_setup
    with pytest.raises(ValueError, match="instance"):
        LinearShapExplainer().explain_local(
            model, X[:5], X[:50], instance_id="i", feature_names=None, **KW
        )


@pytest.mark.parametrize("n_rows", [2, 5])
def test_every_shap_local_adapter_refuses_a_matrix(linear_setup, tree_setup, n_rows):
    """The flatten is shared by all three adapters, so the guard must be too."""
    lin, X = linear_setup
    tree, Xt = tree_setup
    with pytest.raises(ValueError, match="explain_global"):
        LinearShapExplainer().explain_local(
            lin, X[:n_rows], X[:50], instance_id="i", feature_names=None, **KW
        )
    with pytest.raises(ValueError, match="explain_global"):
        TreeShapExplainer().explain_local(tree, Xt[:n_rows], instance_id="i", **KW)
    with pytest.raises(ValueError, match="explain_global"):
        KernelShapExplainer().explain_local(
            tree.predict_proba, Xt[:n_rows], Xt[:20], instance_id="i", **KW
        )


@pytest.mark.parametrize("shape", [(0, 3), (0,), (1, 0)])
def test_every_shap_local_adapter_refuses_an_empty_instance(linear_setup, shape):
    """DEFECT CASE for Linear and Kernel; a standing control for Tree.

    With ``feature_names`` omitted the name list was derived from the (zero)
    value count, so the count check in ``_build_attributions`` could not fire
    and an Explanation with an empty attribution list was returned.
    """
    model, X = linear_setup
    with pytest.raises(ValueError, match="nothing to attribute"):
        LinearShapExplainer().explain_local(
            model, np.zeros(shape), X[:50], instance_id="i", feature_names=None, **KW
        )
    with pytest.raises(ValueError, match="nothing to attribute"):
        KernelShapExplainer().explain_local(
            model.predict_proba, np.zeros(shape), X[:20], instance_id="i", **KW
        )


def test_one_real_row_still_gets_a_real_explanation(linear_setup):
    """OVER-CORRECTION CONTROL, and the healthy measurement.

    A guard that refused every shape would pass all three refusal tests above.
    The dominant attribution must name the feature the model's own largest
    absolute coefficient names, and SHAP local accuracy must hold against the
    model's separately measured margin.
    """
    model, X = linear_setup
    names = ["a", "b", "c"]
    dominant = names[int(np.argmax(np.abs(model.coef_.flatten())))]
    row = X[int(np.argmax(np.abs(X[:, 0])))]

    for x in (row, row.reshape(1, -1)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a healthy run warns about nothing
            e = LinearShapExplainer().explain_local(
                model, x, X[:50], instance_id="i", feature_names=names, **KW
            )
        assert len(e.attributions) == len(names)
        assert [a.feature for a in e.attributions] == names
        top = max(e.attributions, key=lambda a: abs(a.contribution))
        assert top.feature == dominant, (
            f"the strongest attribution was {top.feature}, but the model's largest "
            f"coefficient is on {dominant}"
        )
        assert e.params["attributions_complete"] is True
        assert e.params["prediction_measured"] is True
        assert e.params["local_accuracy_ok"] is True
        scale = max(abs(e.prediction - e.base_value), 1.0)
        assert e.params["local_accuracy_residual"] <= LOCAL_ACCURACY_TOL * scale


def test_explain_global_over_the_same_rows_still_returns_one_per_row(linear_setup):
    """OVER-CORRECTION CONTROL. The matrix path is the one that takes a matrix."""
    model, X = linear_setup
    out = LinearShapExplainer().explain_global(
        model, X[:4], X[:50], feature_names=["a", "b", "c"], **KW
    )
    assert len(out) == 4
    assert [e.instance_id for e in out] == [f"s#row-{i}" for i in range(4)]
    assert all(len(e.attributions) == 3 for e in out)


# === 2. supports() must not say yes to a model it cannot use ===============


AGNOSTIC = [
    ("KernelShapExplainer", lambda: KernelShapExplainer(), "blackbox"),
    ("LimeExplainer", lambda: LimeExplainer(), "blackbox"),
    ("DiceCounterfactualExplainer", lambda: DiceCounterfactualExplainer(), "blackbox"),
    ("AnchorsExplainer", lambda: AnchorsExplainer(), "blackbox"),
]


@pytest.mark.parametrize("name,make,label", AGNOSTIC)
def test_a_model_agnostic_adapter_declines_an_unfitted_estimator(name, make, label):
    """DEFECT CASE. All four returned True for RandomForestClassifier()."""
    try:
        adapter = make()
    except ImportError as exc:  # pragma: no cover -- optional extra absent
        pytest.skip(f"{name} needs an optional extra: {exc}")
    for unfitted in (RandomForestClassifier(), LogisticRegression()):
        with pytest.warns(UserWarning, match="no fitted state"):
            got = adapter.supports(unfitted, label)
        assert got is False, (
            f"{name}.supports({type(unfitted).__name__}(), {label!r}) returned {got}. "
            f"That object's predict method exists and raises when called."
        )


@pytest.mark.parametrize("name,make,label", AGNOSTIC)
def test_a_model_agnostic_adapter_still_accepts_what_it_is_for(
    name, make, label, linear_setup, tree_setup
):
    """OVER-CORRECTION CONTROL. A False for everything passes the test above."""
    try:
        adapter = make()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"{name} needs an optional extra: {exc}")
    lin, _ = linear_setup
    tree, _ = tree_setup
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert adapter.supports(lin, label) is True
        assert adapter.supports(tree, label) is True
        assert adapter.supports(tree.predict, label) is True
        assert adapter.supports(lambda arr: np.zeros(len(arr)), label) is True
        # And the negative direction still works.
        assert adapter.supports("not a model", label) is False
        assert adapter.supports(None, label) is False
        assert adapter.supports({}, label) is False


def test_the_fitted_ness_check_only_judges_objects_that_claim_to_be_estimators():
    """A custom predictor carries no ``get_params``, so it is unaffected."""

    class Predictor:
        def predict(self, arr):
            return np.zeros(len(arr))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert prediction_fn_available(Predictor(), where="t") is True
        assert prediction_fn_available(lambda a: a, where="t") is True
        assert prediction_fn_available(object(), where="t") is False
    with pytest.warns(UserWarning, match="no fitted state"):
        assert prediction_fn_available(LogisticRegression(), where="t") is False


def test_tree_and_linear_supports_weigh_the_model_and_the_label(linear_setup, tree_setup):
    """Standing control for the two label-bound adapters in this batch."""
    lin, _ = linear_setup
    tree, _ = tree_setup
    assert TreeShapExplainer().supports(tree, "tree") is True
    assert TreeShapExplainer().supports(lin, "tree") is False
    assert TreeShapExplainer().supports(tree, "linear") is False
    assert TreeShapExplainer().supports("not a model", "tree") is False
    assert LinearShapExplainer().supports(lin, "linear") is True
    assert LinearShapExplainer().supports(tree, "linear") is False
    assert LinearShapExplainer().supports(lin, "tree") is False
    assert LinearShapExplainer().supports("not a model", "linear") is False


def test_integrated_gradients_supports_weighs_the_model_and_the_label(tree_setup):
    torch = pytest.importorskip("torch")
    tree, _ = tree_setup
    ig = IntegratedGradientsExplainer()
    net = torch.nn.Linear(3, 2)
    assert ig.supports(net, "deep") is True
    assert ig.supports(net, "tree") is False, "the caller's label must be honoured"
    assert ig.supports(tree, "deep") is False, "a forest is not a torch Module"
    assert ig.supports("not a model", "deep") is False


# === 3. the local-only adapters refuse the global mode they do not have ====


@pytest.mark.parametrize(
    "name,make",
    [
        ("DiceCounterfactualExplainer", lambda: DiceCounterfactualExplainer()),
        ("AnchorsExplainer", lambda: AnchorsExplainer()),
    ],
)
def test_a_local_only_adapter_refuses_explain_global_and_says_so_in_capabilities(
    name, make, tree_setup
):
    """A stub that returned ``[]`` would read as "no instance was explainable"."""
    try:
        adapter = make()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"{name} needs an optional extra: {exc}")
    tree, X = tree_setup
    assert adapter.capabilities.supports_global is False
    with pytest.raises(NotImplementedError, match="supports_global is False"):
        adapter.explain_global(tree, X, None, **KW)
    # ... and it refuses on ANY argument shape, so no caller reaches a neutral
    # return by handing it something else.
    with pytest.raises(NotImplementedError):
        adapter.explain_global()


# === 4. the registry ========================================================


def test_available_methods_names_exactly_the_runnable_adapters():
    assert available_methods() == tuple(_REGISTRY.keys())
    assert len(set(available_methods())) == len(available_methods())
    # Every advertised method must instantiate, or the advertisement is false.
    for method in available_methods():
        try:
            adapter = get_explainer(method)
        except ImportError as exc:  # pragma: no cover -- optional extra absent
            pytest.skip(f"{method} needs an optional extra: {exc}")
        assert isinstance(adapter, Explainer)
        assert adapter.capabilities.method == method, (
            f"{type(adapter).__name__} is registered under {method!r} but declares "
            f"{adapter.capabilities.method!r}"
        )
        assert adapter.capabilities.component_id, f"{method} carries no component id"
    # Nothing advertised as runnable may also be listed as unbuilt.
    assert set(available_methods()).isdisjoint(_NOT_YET_IMPLEMENTED)


@pytest.mark.parametrize("method", sorted(_NOT_YET_IMPLEMENTED))
def test_get_explainer_refuses_a_specified_but_unbuilt_method(method):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        get_explainer(method)


@pytest.mark.parametrize("method", ["nope", "", None, 0, "SHAP.TreeExplainer"])
def test_get_explainer_refuses_an_unknown_method(method):
    """A ``None`` return here would be indistinguishable from a working adapter
    until the first attribute access, far from the routing decision."""
    with pytest.raises(KeyError, match="Unknown explainer method"):
        get_explainer(method)


# === 5. the thin units: nothing minted ======================================


def test_the_explainer_abc_cannot_be_instantiated_and_all_three_stubs_are_abstract():
    assert Explainer.__abstractmethods__ == frozenset(
        {"supports", "explain_local", "explain_global"}
    )
    with pytest.raises(TypeError, match="abstract"):
        Explainer()

    class Partial(Explainer):
        def supports(self, model, model_type):
            return True

    with pytest.raises(TypeError, match="explain_global"):
        Partial()


def test_explainer_capabilities_is_frozen_and_defaults_to_nothing_load_bearing():
    caps = ExplainerCapabilities(method="lime", supported_model_types=("blackbox",))
    with pytest.raises(Exception):
        caps.method = "anchors"
    assert hash(caps) == hash(
        ExplainerCapabilities(method="lime", supported_model_types=("blackbox",))
    )
    # component_id is load-bearing (router._component_id_for reads it) and must
    # NOT default to a plausible-looking id.
    assert caps.component_id == ""
    assert caps.is_async_eligible is False
    assert caps.requires_background is False


def test_route_decision_is_frozen_and_carries_no_invented_provenance():
    rd = RouteDecision(
        primary="shap.TreeExplainer",
        component_id="tree_shap",
        fallbacks=("shap.KernelExplainer",),
    )
    with pytest.raises(Exception):
        rd.primary = "lime"
    # A default of "" / None, never a downgrade that did not happen nor a
    # reason nobody wrote.
    assert rd.downgraded_from is None
    assert rd.reason == ""
    assert rd.fallbacks == ("shap.KernelExplainer",)
    assert isinstance(rd.fallbacks, tuple), "a mutable default would be shared state"
