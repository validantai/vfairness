"""A small function still has to earn its answer.

The units here are one-liners and accessors, the kind nobody reviews closely because
there is visibly nothing in them. That is exactly where an answer can be asserted
rather than determined, because the code is too short to look wrong:

  1. ``supports(model, model_type)`` READ ONLY THE LABEL. ``return model_type ==
     "tree"`` answers a question about a model object without ever touching it.
     Measured before the fix, ``TreeShapExplainer().supports("not a model", "tree")``
     returned True: the adapter claimed it could explain a str. The same for
     ``LinearShapExplainer().supports(RandomForestClassifier(), "linear")``. The
     label and the model are separate evidence and they disagree precisely when the
     answer matters, and a mislabelled model then failed later inside shap, where the
     cause was no longer visible. IntegratedGradientsExplainer.supports in the same
     package already checked both, which is what made the omission legible.
  2. ``get_capabilities`` HARDCODED A TRUE. ``"whatif_simulation": True`` beside
     ``"dash_server": False`` and ``"standalone_html": False`` tells a caller yes
     about a feature with no remaining route to it. The sweep behind it is pure
     Python, which is what made the True look true.
  3. A STUB MUST REFUSE. An abstract or unsupported method that returns a neutral
     value instead of raising is the same defect wearing the smallest possible body.

Every assertion below runs BOTH directions, because "does not support" is easy to be
right about by accident: a supports() that returned False for everything would pass
every negative case in this file and fail every positive one.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import numpy as np
import pytest

# ----------------------------------------------------------------- supports(model)

pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

_X = np.random.RandomState(0).rand(40, 3)
_Y = (_X[:, 0] > 0.5).astype(int)


@pytest.fixture(scope="module")
def models():
    return {
        "linear": LogisticRegression().fit(_X, _Y),
        "tree": RandomForestClassifier(n_estimators=3, random_state=0).fit(_X, _Y),
        "not a model": "not a model",
    }


@pytest.mark.parametrize(
    "adapter,label,model_key,expected",
    [
        # Matched label and model: must say yes, or the adapter is useless.
        ("TreeShapExplainer", "tree", "tree", True),
        ("LinearShapExplainer", "linear", "linear", True),
        # Right label, WRONG model. Every one of these returned True before the fix.
        ("TreeShapExplainer", "tree", "linear", False),
        ("TreeShapExplainer", "tree", "not a model", False),
        ("LinearShapExplainer", "linear", "tree", False),
        ("LinearShapExplainer", "linear", "not a model", False),
        # Wrong label, right model: the label still has to be honoured.
        ("TreeShapExplainer", "linear", "tree", False),
        ("LinearShapExplainer", "tree", "linear", False),
    ],
)
def test_supports_judges_the_model_and_not_only_the_caller_s_label(
    adapter, label, model_key, expected, models
):
    from vfairness.xai.explainers import shap_adapter

    got = getattr(shap_adapter, adapter)().supports(models[model_key], label)
    assert got is expected, (
        f"{adapter}.supports({model_key!r}, {label!r}) returned {got}, expected "
        f"{expected}. A yes here is a claim about an object; if it was reached "
        f"without inspecting that object, it is not a claim this adapter is "
        f"entitled to make."
    )


def test_a_model_agnostic_adapter_still_accepts_anything_callable(models):
    """Over-correction control. KernelSHAP and LIME really are model agnostic, and
    tightening them to a label would break the thing they are for."""
    from vfairness.xai.explainers.lime_adapter import LimeExplainer
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer

    for expl in (KernelShapExplainer(), LimeExplainer()):
        assert expl.supports(models["tree"], "tree") is True
        assert expl.supports(models["linear"], "linear") is True
        assert expl.supports("not a model", "anything") is False


def test_integrated_gradients_checks_both_and_says_so_when_it_cannot_tell():
    """It already checked both. The addition is that a MISSING torch is reported as
    could-not-determine rather than as an ordinary no."""
    import vfairness.xai.explainers.ig_adapter as ig

    src = inspect.getsource(ig.IntegratedGradientsExplainer.supports)
    assert "isinstance(model" in src, "the model itself must be inspected"
    assert "model_type" in src, "the label must be honoured too"
    assert "warnings.warn" in src, (
        "a bare False for a missing dependency is indistinguishable from 'your model "
        "is the wrong kind', and the caller can fix one of those"
    )


# ------------------------------------------------------------------- stubs refuse


@pytest.mark.parametrize(
    "label,call",
    [
        (
            "AnchorsExplainer.explain_global",
            lambda: (
                __import__(
                    "vfairness.xai.explainers.anchors_adapter", fromlist=["AnchorsExplainer"]
                )
                .AnchorsExplainer()
                .explain_global(
                    model=lambda x: x,
                    X=np.zeros((2, 2)),
                    background=None,
                    subject_id="s",
                    model_hash="h",
                    data_hash="d",
                )
            ),
        ),
        (
            "DiceCounterfactualExplainer.explain_global",
            lambda: (
                __import__(
                    "vfairness.xai.explainers.dice_adapter",
                    fromlist=["DiceCounterfactualExplainer"],
                )
                .DiceCounterfactualExplainer()
                .explain_global(
                    model=lambda x: x,
                    X=np.zeros((2, 2)),
                    background=None,
                    subject_id="s",
                    model_hash="h",
                    data_hash="d",
                )
            ),
        ),
    ],
)
def test_an_unsupported_mode_raises_rather_than_returning_an_empty_answer(label, call):
    """An empty explanation is indistinguishable from an explanation that found
    nothing, and only one of those is true here."""
    with pytest.raises(NotImplementedError) as exc:
        call()
    assert len(str(exc.value)) > 30, (
        f"{label} refuses but does not say why, so a caller cannot act on it"
    )


def test_the_explainer_base_cannot_be_instantiated():
    from vfairness.xai.explainers.base import Explainer

    with pytest.raises(TypeError):
        Explainer()


def test_an_unknown_explainer_name_is_refused_not_defaulted():
    from vfairness.xai.explainers.registry import get_explainer

    with pytest.raises((KeyError, ValueError)):
        get_explainer("no-such-explainer")


# ------------------------------------------------------------- reported capability


def test_no_capability_is_reported_available_with_no_way_to_deliver_it():
    from vfairness.operations.reporting import interactive
    from vfairness.operations.reporting.store import MetricsStore

    caps = interactive.InteractiveDashboard(MetricsStore()).get_capabilities()
    surfaces = [caps["dash_server"], caps["standalone_html"]]
    if not any(surfaces):
        assert not any(caps.values()), (
            f"neither delivery surface is available, so nothing can be delivered, "
            f"yet these are reported as available: "
            f"{[k for k, v in caps.items() if v]}"
        )
    # And the reverse: a flag must not be hardcoded, which is what a source read
    # settles and a value read cannot.
    src = inspect.getsource(interactive.InteractiveDashboard.get_capabilities)
    hardcoded = re.findall(r'"(\w+)":\s*True\s*,', src)
    assert not hardcoded, (
        f"these capabilities are reported as True unconditionally, so they claim to "
        f"work in an environment nobody checked: {hardcoded}"
    )


# ------------------------------------------------- the static sweep, as a standing gate

# Families of very short units, swept for the three shapes named in the module
# docstring. An entry in ADJUDICATED_DROPS is a unit whose unused parameter is
# correct, and the reason says why.
ADJUDICATED_DROPS = {
    "vfairness.xai.explainers.base.Explainer.supports": "abstract; the body is a docstring",
    "vfairness.xai.explainers.base.Explainer.explain_global": "abstract; the body is a docstring",
    "vfairness.xai.explainers.base.Explainer.explain_local": "abstract; the body is a docstring",
}

SWEPT = [
    (
        "vfairness.xai.explainers.shap_adapter",
        ["TreeShapExplainer", "LinearShapExplainer", "KernelShapExplainer"],
    ),
    ("vfairness.xai.explainers.lime_adapter", ["LimeExplainer"]),
    ("vfairness.xai.explainers.ig_adapter", ["IntegratedGradientsExplainer"]),
    ("vfairness.xai.explainers.dice_adapter", ["DiceCounterfactualExplainer"]),
    ("vfairness.xai.explainers.anchors_adapter", ["AnchorsExplainer"]),
]


def _supports_methods():
    found = []
    for mod_name, classes in SWEPT:
        mod = __import__(mod_name, fromlist=classes)
        for cls_name in classes:
            cls = getattr(mod, cls_name)
            fn = cls.__dict__.get("supports")
            if fn is not None:
                found.append((f"{mod_name}.{cls_name}.supports", fn))
    return found


def test_no_supports_answers_from_a_label_alone():
    """The shape of the original defect, caught structurally as well as behaviourally.

    A behavioural test needs a model of the right shape to compare against, and
    there will be adapters nobody has built a fixture for. This one asks a weaker
    question of every implementation at once: does the body reference the model
    parameter at all? A supports() that never mentions `model` cannot have inspected
    it, whatever it returns.
    """
    offenders = []
    for qual, fn in _supports_methods():
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        fd = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        body = [
            n
            for n in fd.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        ]
        names = {n.id for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
        if "model" not in names:
            offenders.append(qual)
    assert not offenders, (
        "these supports() implementations never reference the model they are asked "
        f"about, so a yes from them is about a string: {offenders}"
    )
    assert len(_supports_methods()) >= 7, (
        "fewer supports() implementations were found than exist, so this sweep has gone blind"
    )
