"""
vfairness.xai.explainers.shap_adapter
=====================================

Thin adapters around the ``shap`` library that normalise into the
``Explanation`` contract. We never propagate matplotlib figures or the
shap Explanation object upward; only arrays + dataclasses cross the
boundary. The frontend renders.

Phase 1 adapters (covered):

* :class:`TreeShapExplainer` -- TreeSHAP exact for XGBoost / LightGBM /
  CatBoost / sklearn trees.
* :class:`LinearShapExplainer` -- closed-form for linear / GLM models.
* :class:`KernelShapExplainer` -- model-agnostic, async-eligible.

Phase 2 (stubs in __init__): DeepExplainer, GradientExplainer.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..schemas import Attribution, Explanation
from .base import Explainer, ExplainerCapabilities, prediction_fn_available


def _build_attributions(values: np.ndarray, feature_names: list[str]) -> list[Attribution]:
    """Pair every attributed value with its name, or refuse to pair any.

    BGL3 xai-1, 2026-09-27. This was a bare ``zip``, and zip STOPS AT THE
    SHORTER SEQUENCE without a word. Measured on a 3-feature random forest with
    ``feature_names=["a", "b"]``, TreeShapExplainer.explain_local returned two
    attributions summing to 0.50906 beside ``base_value`` 0.41250 and
    ``prediction`` 1.00000: 0.07844 of the movement, 13 percent of it, sat on a
    feature that had been deleted from the returned explanation, which therefore
    did not add up to its own prediction and said nothing about why. The same
    mismatch through KernelShapExplainer returned 2 of 3 values with
    ``prediction`` 0.0 over ``base_value`` 0.39375.

    A name list of the wrong length is a caller-side mismatch, and raw column
    names against a one-hot encoded matrix is the usual way it arises. There is
    no honest repair here: once the counts disagree nothing in this function
    knows which value belongs to which name, and guessing would mislabel every
    attribution after the first gap. Fail loudly, naming both counts, rather
    than publish a decomposition that is missing a term.
    """
    n_values = int(np.asarray(values).size)
    if len(feature_names) != n_values:
        raise ValueError(
            f"feature_names carries {len(feature_names)} name(s) for {n_values} attributed "
            f"value(s). Pairing them would silently drop the surplus on the longer side and "
            f"return an explanation whose attributions do not sum to its own prediction. "
            f"Supply one name per feature, in the order the model was fitted on."
        )
    return [
        Attribution(feature=name, contribution=float(v)) for name, v in zip(feature_names, values)
    ]


def _one_instance(x: Any, where: str) -> np.ndarray:
    """The single instance *where* was asked to explain, as ``(1, n_features)``.

    G07 2026-09-30. Every ``explain_local`` in this module reaches its numbers
    through ``np.asarray(shap_values).flatten()``, and FLATTEN DOES NOT KNOW
    ABOUT ROWS. Measured on a 3-feature LogisticRegression with ``x`` of shape
    ``(5, 3)`` and no ``feature_names``: ``LinearShapExplainer.explain_local``
    returned ONE Explanation carrying 15 attributions named ``f0 .. f14``, the
    row axis interleaved into the feature axis, so row 0's value on the first
    feature (8.75575) and row 1's value on the SAME feature (2.44833) were
    published as two DIFFERENT features of one instance, with
    ``params['attributions_complete']`` True and no warning about the shape.
    Supplying real ``feature_names`` raised in ``_build_attributions`` on the
    count, so the defect was reachable exactly when the caller had no names to
    give and the fabricated ``f0 .. fN`` labels were used instead.

    The same call with ``x`` of shape ``(0, 3)`` returned an Explanation with
    ``attributions=[]`` and ``attributions_complete=True``: a complete-looking
    explanation of an instance that was never examined. That is the BGL3 xai-1
    shape, which ``TreeShapExplainer.explain_local`` refuses after its own
    ``shap_values`` call; the refusal is hoisted here so it covers the three
    adapters that share the flatten, and so it fires BEFORE the shap explainer
    is built rather than after.

    There is no honest repair for more than one row: silently taking row 0
    would drop the rest, and ``explain_global`` is the method that returns one
    Explanation per row. Refuse and name it.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        raise ValueError(
            f"{where}: x has shape {np.asarray(x).shape}, which is not a single tabular "
            f"instance. Pass one row of shape (n_features,) or (1, n_features)."
        )
    # Emptiness FIRST, and in these words. An empty x is the BGL3 xai-1 case
    # that tests/test_bgl3_xai_1.py pins by matching "nothing to attribute";
    # shape (0, 3) is also zero ROWS, so deciding the row count first would
    # have answered that call with a different sentence and reddened a control
    # whose subject is still exactly right.
    if arr.size == 0:
        raise ValueError(
            f"{where}: x has shape {np.asarray(x).shape}, so there is nothing to "
            f"attribute. Returning an Explanation would publish an empty attribution "
            f"list, which reads as a complete explanation of an instance whose features "
            f"all contributed nothing rather than as an instance that was never examined."
        )
    n_rows = int(arr.shape[0])
    if n_rows != 1:
        raise ValueError(
            f"{where}: x has shape {np.asarray(x).shape}, which is {n_rows} instance(s), "
            f"and this method attributes ONE. The SHAP values are flattened before they "
            f"are named, so {n_rows} rows would be published as a single explanation with "
            f"{arr.size} attributions, the row axis interleaved into the feature axis and "
            f"the same feature appearing once per row under a different name. Use "
            f"explain_global for a matrix; it returns one Explanation per row."
        )
    return arr


#: Relative tolerance the SHAP local accuracy axiom is CHECKED against.
#: LinearExplainer's closed form is exact, so a healthy run lands at machine
#: epsilon (measured 4.4e-16); anything above this is a real disagreement
#: between the attributions and the model.
LOCAL_ACCURACY_TOL = 1e-6


def _unattributed(values: np.ndarray, feature_names: list[str]) -> list[str]:
    """The features whose SHAP value is not a number.

    A non-finite SHAP value is not a small contribution; it is a feature this
    run could not attribute. Emitted as an ordinary entry it is worse than
    invisible: every ranker in this package orders by ``abs(contribution)``
    and ``abs(nan) > x`` is False for every x, so the unattributable feature
    is sorted LAST and read as the least influential driver. Measured
    2026-09-16 on a logistic model whose strongest coefficient was ``income``
    (-5.60, about 3.6 times the next): with ``income`` NaN the documented
    public diagnostic ``vfairness.xai.diagnostics.removal_curve_auc`` returned
    a finite faithfulness score of 0.3600 over the ranking
    ``['zip', 'age', 'income']``, with no warning anywhere.
    """
    return [str(n) for n, v in zip(feature_names, values) if not np.isfinite(v)]


def _model_margin(model: Any, x: np.ndarray) -> float:
    """The model's own output at *x*, in the space SHAP attributes in.

    NaN when the model exposes no usable scoring function or the call fails:
    that is a could-not-check, and the caller is told so by
    ``params['prediction_measured']`` rather than being handed the identity
    ``base + sum(values)``, which agrees with the attributions by construction
    and therefore can never contradict them.
    """
    row = np.asarray(x, dtype=float).reshape(1, -1)
    fn = getattr(model, "decision_function", None)
    if fn is None and not hasattr(model, "classes_"):
        # A regressor: predict IS the margin. A CLASSIFIER without
        # decision_function is not scored here, because its predict returns a
        # class LABEL, and comparing a 0/1 label with a log-odds sum would
        # manufacture a local-accuracy failure out of a unit mismatch.
        fn = getattr(model, "predict", None)
    if fn is None:
        return float("nan")
    try:
        return float(np.asarray(fn(row)).flatten()[0])
    except Exception:  # noqa: BLE001 -- a scoring failure is a refusal, not a crash
        return float("nan")


def _class_consistent(shap_values: Any, expected_value: Any, cls: int = 1) -> tuple:
    """Pick ONE class index for both the attributions and the base value.

    For classifiers shap returns per-class outputs. The previous code combined
    class-``1`` attributions with the class-``0`` expected value, making
    ``base_value`` describe a different class than the attributions and
    ``prediction`` (base + sum(values)) a meaningless hybrid. Here both are
    taken from the same class: the positive class (index 1) when per-class
    outputs exist.

    Two return conventions exist and are detected by inspecting the returned
    OBJECT, never the version string:

    * shap < 0.45: a Python list of per-class ``(n_samples, n_features)``
      arrays; ``expected_value`` is a per-class array.
    * shap >= 0.45 (0.45.0 changelog, "shap_values" unified return): one
      ``(n_samples, n_features, n_outputs)`` ndarray with the class axis
      LAST; ``expected_value`` is still a per-class array. Without the
      3-D branch the class axis stays interleaved into the feature axis
      downstream, which silently mislabels attributions and drops half
      the features.

    Single-output models (regression / margin output) return a 2-D array and
    a scalar expected value; those pass through unchanged.
    """
    ev = expected_value

    def _base_from(idx: int) -> float:
        if isinstance(ev, (list, np.ndarray)) and np.size(ev) > 1:
            return float(np.asarray(ev).flatten()[min(idx, np.size(ev) - 1)])
        if isinstance(ev, (list, np.ndarray)):
            return float(np.asarray(ev).flatten()[0])
        return float(ev)

    if isinstance(shap_values, list):
        # Legacy convention (shap < 0.45): list of per-class arrays.
        idx = min(cls, len(shap_values) - 1)
        return shap_values[idx], _base_from(idx)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Modern convention (shap >= 0.45): (n_samples, n_features, n_outputs)
        # with the class/output axis last. Select the requested class on that
        # axis so values stay (n_samples, n_features).
        idx = min(cls, arr.shape[2] - 1)
        return arr[:, :, idx], _base_from(idx)

    # Single-output (regression / margin): nothing to select.
    return shap_values, _base_from(0)


class TreeShapExplainer(Explainer):
    """TreeSHAP. Exact, polynomial-time, synchronous.

    THE GENERATED BLOCK BELOW IS OUT OF DATE for explain_global. BGL5 A-xai-1
    (2026-09-27) closed a grade whose named test executed none of that method's body and
    pinned the refusal and the row labelling on that path in tests/test_bgl5_xai_1.py,
    sabotage-checked, so "nothing in the suite holds it there" no longer describes it.
    That sentence is RENDERED by scripts/stamp_proof_status.py from
    src/vfairness/_proof_status.py and docs/capability-status.json, which this batch
    does not own: editing it here is reverted by the next stamp run and reported STALE
    by that script's own check pass (measured: 1 stale stamp at this line), which
    tests/test_beta_go_live_proof_ledger.py::test_no_docstring_stamp_is_stale asserts
    must be zero. The ledger regeneration is recorded as a hand-back.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: tree_shap. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="shap.TreeExplainer",
        supported_model_types=("tree",),
        supports_local=True,
        supports_global=True,
        is_async_eligible=False,
        requires_background=False,
        component_id="tree_shap",
    )

    def __init__(self) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:  # pragma: no cover -- declared as extra
            raise ImportError(
                "vfairness.xai.explainers.TreeShapExplainer requires the optional `shap` extra: "
                "`pip install vfairness[xai]`."
            ) from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """Whether this adapter can explain *model*, judged from the model too.

        This was ``return model_type == "tree"``, which answered a question about a
        model object by reading a STRING THE CALLER SUPPLIED and never looking at the
        object at all. Measured before the fix::

            TreeShapExplainer().supports("not a model", "tree")   -> True

        It claimed it could explain a str. The label and the model are two separate
        pieces of evidence and they can disagree, which is exactly when an answer
        matters: a caller who mislabels a linear model as a tree got True here and a
        failure later inside shap.TreeExplainer, at a point where the cause is no
        longer visible.

        The model side is duck-typed on purpose, so xgboost, lightgbm and catboost
        pass without this module importing any of them, and it stays as cheap as the
        base class asks for. IntegratedGradientsExplainer.supports in this same
        package already checked both, which is what made the omission here visible.
        """
        if model_type != "tree":
            return False
        return hasattr(model, "predict") and any(
            hasattr(model, attr)
            for attr in ("estimators_", "tree_", "get_booster", "booster_", "feature_importances_")
        )

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
        feature_names: list[str] | None = None,
        **params: Any,
    ) -> Explanation:
        import shap

        # One instance, checked before shap is built (see _one_instance).
        x = _one_instance(x, "TreeShapExplainer.explain_local")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        # Class-consistent selection: attributions AND base value from the
        # same class (positive class for classifiers).
        shap_values, base_value = _class_consistent(shap_values, explainer.expected_value)
        values = np.asarray(shap_values).flatten()
        # BGL3 xai-1, 2026-09-27. Nothing to attribute is not an explanation.
        # Measured on x of shape (0, 3), a real fitted random forest and no other
        # change: this returned an Explanation with ``prediction`` 0.41250,
        # ``base_value`` 0.41250, ``attributions`` [] and no warning at all, which
        # a reader cannot tell from a genuine explanation whose features all
        # happened to contribute nothing. There is no instance here to explain, so
        # there is no Explanation to return.
        if values.size == 0:
            raise ValueError(
                "TreeShapExplainer.explain_local: shap produced no values for this input "
                f"(x has shape {np.asarray(x).shape}), so there is nothing to attribute. "
                "Returning an Explanation would publish prediction == base_value with an "
                "empty attribution list, which reads as a complete explanation of an "
                "instance that was never examined."
            )
        names = feature_names or [f"f{i}" for i in range(values.shape[0])]
        pred = float(base_value + np.sum(values))
        return Explanation(
            method="shap.TreeExplainer",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=base_value,
            prediction=pred,
            attributions=_build_attributions(values, names),
            units=params.get("units", "log-odds"),
            params=params,
            library_versions={"shap": _shap_version()},
        )

    def explain_global(
        self,
        model: Any,
        X: np.ndarray,
        background: np.ndarray | None = None,
        *,
        subject_id: str,
        model_hash: str,
        data_hash: str,
        feature_names: list[str] | None = None,
        **params: Any,
    ) -> list[Explanation]:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Class-consistent selection (see _class_consistent).
        shap_values, base_value = _class_consistent(shap_values, explainer.expected_value)
        names = feature_names or [f"f{i}" for i in range(np.asarray(shap_values).shape[1])]
        explanations: list[Explanation] = []
        for i, row in enumerate(np.asarray(shap_values)):
            explanations.append(
                Explanation(
                    method="shap.TreeExplainer",
                    instance_id=f"{subject_id}#row-{i}",
                    subject_id=subject_id,
                    model_hash=model_hash,
                    data_hash=data_hash,
                    base_value=base_value,
                    prediction=base_value + float(row.sum()),
                    attributions=_build_attributions(row, names),
                    units=params.get("units", "log-odds"),
                    params=params,
                    library_versions={"shap": _shap_version()},
                )
            )
        return explanations


class LinearShapExplainer(Explainer):
    """LinearExplainer: closed-form SHAP for linear / GLM models.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: linear_explainer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="shap.LinearExplainer",
        supported_model_types=("linear",),
        is_async_eligible=False,
        requires_background=True,
        component_id="linear_explainer",
    )

    def __init__(self) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LinearShapExplainer requires `shap`.") from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """Whether this adapter can explain *model*, judged from the model too.

        This was ``return model_type == "linear"``. Measured before the fix,
        ``LinearShapExplainer().supports(RandomForestClassifier(), "linear")`` and
        ``.supports("not a model", "linear")`` both returned True. See the note on
        TreeShapExplainer.supports; shap.LinearExplainer needs the fitted
        coefficients, so their presence is the real precondition and a caller's label
        is not.
        """
        if model_type != "linear":
            return False
        return hasattr(model, "coef_")

    def explain_local(
        self,
        model,
        x,
        background=None,
        *,
        instance_id,
        subject_id,
        model_hash,
        data_hash,
        feature_names=None,
        **params,
    ):
        import shap

        if background is None:
            raise ValueError("LinearExplainer requires a background sample.")
        # One instance, checked before shap is built (see _one_instance).
        x = _one_instance(x, "LinearShapExplainer.explain_local")
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(x)
        values = np.asarray(shap_values).flatten()
        names = feature_names or [f"f{i}" for i in range(values.shape[0])]

        base_value = float(explainer.expected_value)
        unattributed = _unattributed(values, names)

        # The model's OWN output, not base + sum(values). The identity
        # base + sum(values) satisfies the SHAP local-accuracy axiom by
        # construction, so it agreed with the attributions no matter what they
        # were: an empty background produced base=nan and three nan values and
        # the "prediction" inherited the nan without a word, and a healthy run
        # could never have disagreed with its own attributions either, which is
        # what a local-accuracy check is FOR. Measuring it separately makes the
        # residual a real check.
        measured = _model_margin(model, x)
        prediction_measured = math.isfinite(measured)
        identity = base_value + float(values.sum())
        residual = abs(measured - identity)

        out_params: dict[str, Any] = dict(params)
        out_params["unattributed_features"] = unattributed
        out_params["attributions_complete"] = not unattributed
        out_params["prediction_measured"] = prediction_measured
        out_params["base_value_measured"] = math.isfinite(base_value)
        out_params["local_accuracy_identity"] = identity
        # None, not 0.0: an unmeasurable prediction leaves local accuracy
        # UNCHECKED, and 0.0 there would read as a perfect agreement.
        out_params["local_accuracy_residual"] = residual if math.isfinite(residual) else None
        # And the residual is GRADED, not merely recorded. A number in params
        # that is compared with nothing is the same defect one step later: the
        # sibling adapter recorded an IG completeness residual of 1.0, a
        # prediction 100 percent unexplained, and nothing read it. Relative to
        # what has to be explained, with an absolute floor of 1.0 so a model
        # that did not move does not divide by zero. Three states: True within
        # tolerance, False outside it, None when it could not be computed.
        scale = max(abs(measured - base_value), 1.0) if prediction_measured else float("nan")
        local_accuracy_ok: bool | None
        if not math.isfinite(residual) or not math.isfinite(scale):
            local_accuracy_ok = None
        else:
            local_accuracy_ok = residual <= LOCAL_ACCURACY_TOL * scale
        out_params["local_accuracy_tol"] = LOCAL_ACCURACY_TOL
        out_params["local_accuracy_ok"] = local_accuracy_ok

        # ``units`` describes the space the ATTRIBUTIONS are in, and
        # LinearExplainer attributes in the model's margin space. Echoing back
        # whatever the caller asked for labelled a log-odds explanation
        # "probability": measured 2026-09-16, prediction -3.3766 (log-odds)
        # against a real predict_proba of 0.0330, under units='probability'.
        # The label now states the space the numbers are actually in, and the
        # request is preserved in params rather than silently honoured.
        requested_units = params.get("units", "log-odds")
        units = "log-odds"
        if requested_units != units:
            out_params["units_requested"] = requested_units
            out_params["units_note"] = (
                f"units '{requested_units}' was requested but LinearExplainer attributes "
                f"in the model's margin space; the attributions, base_value and "
                f"prediction below are log-odds and are labelled as such."
            )
            warnings.warn(
                f"LinearShapExplainer: units '{requested_units}' was requested, but these "
                f"SHAP values are in the model's log-odds margin space. The explanation "
                f"is labelled 'log-odds'; converting it would not make the attributions "
                f"additive in probability space.",
                UserWarning,
                stacklevel=2,
            )

        if unattributed:
            warnings.warn(
                "LinearShapExplainer: no SHAP value could be computed for "
                f"{', '.join(unattributed)}. Those contributions are NaN, not small, so "
                "the explanation is INCOMPLETE and any ranking by magnitude sorts them "
                "LAST rather than omitting them. See params['unattributed_features'].",
                UserWarning,
                stacklevel=2,
            )
        if not prediction_measured:
            warnings.warn(
                "LinearShapExplainer: the model could not be scored at this instance, so "
                "prediction is NaN and local accuracy is UNCHECKED. "
                "params['local_accuracy_residual'] is None, which is not a pass.",
                UserWarning,
                stacklevel=2,
            )
        elif local_accuracy_ok is None:
            warnings.warn(
                "LinearShapExplainer: the local accuracy residual could not be computed, "
                "so it is UNCHECKED. params['local_accuracy_ok'] is None, which is not a "
                "pass.",
                UserWarning,
                stacklevel=2,
            )
        elif local_accuracy_ok is False:
            warnings.warn(
                "LinearShapExplainer: the SHAP local accuracy axiom does not hold for "
                f"this explanation. base_value plus the attributions gives {identity:.6g} "
                f"while the model scores {measured:.6g} (residual {residual:.4g}, "
                f"tolerance {LOCAL_ACCURACY_TOL:.4g} relative). The attributions are not "
                "a faithful decomposition of this prediction.",
                UserWarning,
                stacklevel=2,
            )

        return Explanation(
            method="shap.LinearExplainer",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=base_value,
            prediction=measured,
            attributions=_build_attributions(values, names),
            units=units,
            params=out_params,
            library_versions={"shap": _shap_version()},
        )

    def explain_global(
        self,
        model,
        X,
        background=None,
        *,
        subject_id,
        model_hash,
        data_hash,
        feature_names=None,
        **params,
    ):
        out: list[Explanation] = []
        for i, x in enumerate(X):
            out.append(
                self.explain_local(
                    model,
                    x,
                    background,
                    instance_id=f"{subject_id}#row-{i}",
                    subject_id=subject_id,
                    model_hash=model_hash,
                    data_hash=data_hash,
                    feature_names=feature_names,
                    **params,
                )
            )
        return out


class KernelShapExplainer(Explainer):
    """KernelSHAP: model-agnostic, async-eligible.

    THE GENERATED BLOCK BELOW IS OUT OF DATE for explain_global. BGL5 A-xai-1
    (2026-09-27) closed a grade whose named test never called that method at all and
    pinned the per-row mapping, the caller's feature names and the refusal in
    tests/test_bgl5_xai_1.py, sabotage-checked, so "nothing in the suite holds it there"
    no longer describes it. That sentence is RENDERED by scripts/stamp_proof_status.py
    from src/vfairness/_proof_status.py and docs/capability-status.json, which this
    batch does not own: editing it here is reverted by the next stamp run and reported
    STALE by that script's own check pass (measured: 1 stale stamp at this line), which
    tests/test_beta_go_live_proof_ledger.py::test_no_docstring_stamp_is_stale asserts
    must be zero. The ledger regeneration is recorded as a hand-back.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: kernel_shap. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="shap.KernelExplainer",
        supported_model_types=("blackbox", "tree", "linear", "deep"),
        is_async_eligible=True,
        requires_background=True,
        component_id="kernel_shap",
    )

    def __init__(self) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("KernelShapExplainer requires `shap`.") from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """Any model exposing a prediction function, which is what agnostic means.

        This was ``return callable(model)``, and NO sklearn estimator is callable:
        ``callable(RandomForestClassifier())`` is False. So KernelSHAP, the one
        explainer here that is model agnostic by construction, declined every sklearn
        model in the library, while LimeExplainer, DiceCounterfactualExplainer and
        AnchorsExplainer all accepted the same object. A wrong no is less dangerous
        than a wrong yes and it is still a wrong answer: it makes a capability that
        exists read as absent, and nothing in the package calls supports(), so
        nothing contradicted it.

        shap.KernelExplainer takes a prediction FUNCTION, so predict or
        predict_proba is the real precondition and being callable is only one way to
        satisfy it.

        G07 2026-09-30: that precondition now goes through
        ``prediction_fn_available``, because ``hasattr(model, "predict")`` is True
        for an UNFITTED estimator, and this returned True for
        ``RandomForestClassifier()`` while its own siblings TreeShapExplainer and
        LinearShapExplainer declined the same object. See that helper.
        """
        return prediction_fn_available(model, where="KernelShapExplainer.supports")

    def explain_local(
        self,
        model,
        x,
        background=None,
        *,
        instance_id,
        subject_id,
        model_hash,
        data_hash,
        feature_names=None,
        **params,
    ):
        import shap

        if background is None:
            raise ValueError(
                "KernelExplainer requires a background sample (use shap.sample if large)."
            )
        nsamples = params.get("nsamples", "auto")
        # One instance, checked before shap is built (see _one_instance).
        x = _one_instance(x, "KernelShapExplainer.explain_local")
        explainer = shap.KernelExplainer(model, background)
        shap_values = explainer.shap_values(x, nsamples=nsamples)
        # Class-consistent selection (see _class_consistent).
        shap_values, base_value = _class_consistent(shap_values, explainer.expected_value)
        values = np.asarray(shap_values).flatten()
        names = feature_names or [f"f{i}" for i in range(values.shape[0])]
        return Explanation(
            method="shap.KernelExplainer",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=base_value,
            prediction=float(base_value + values.sum()),
            attributions=_build_attributions(values, names),
            units=params.get("units", "raw"),
            params=params,
            library_versions={"shap": _shap_version()},
        )

    def explain_global(
        self,
        model,
        X,
        background=None,
        *,
        subject_id,
        model_hash,
        data_hash,
        feature_names=None,
        **params,
    ):
        out: list[Explanation] = []
        for i, x in enumerate(X):
            out.append(
                self.explain_local(
                    model,
                    x,
                    background,
                    instance_id=f"{subject_id}#row-{i}",
                    subject_id=subject_id,
                    model_hash=model_hash,
                    data_hash=data_hash,
                    feature_names=feature_names,
                    **params,
                )
            )
        return out


def _shap_version() -> str:
    try:
        import shap

        return shap.__version__
    except Exception:  # pragma: no cover
        return "unknown"
