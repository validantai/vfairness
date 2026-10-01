"""
vfairness.xai.explainers.dice_adapter
=====================================

Adapter around the ``dice-ml`` library (Diverse Counterfactual
Explanations, Mothilal et al. 2020) that normalises native DiCE output
into the :class:`~vfairness.xai.schemas.CounterfactualExplanation`
contract.

Phase 2 item #P2-01. DiCE answers the "actionable recourse" goal: for a
single declined instance it returns N diverse counterfactuals, the
smallest realistic changes that flip the decision. The router selects
this adapter whenever ``goal == "actionable_recourse"`` or
``counterfactual_needed`` is set (see ``router.py``).

Like the SHAP adapters this never propagates the native DiCE object
upward; only the normalised dataclasses cross the boundary. The frontend
``CounterfactualSentence`` chart renders the result.

proximity / feasibility are computed here (DiCE does not attach them per
counterfactual):

* **proximity**: the mean normalised absolute change across the flipped
  features, ``mean(|cf_i - x_i| / range_i)``. Smaller means a closer,
  cheaper counterfactual. Range is taken from the supplied training
  frame so the scale matches the data the model saw.
* **feasibility**: the fraction of flipped features whose new value
  lands inside the observed ``[min, max]`` of the training frame. 1.0
  means every proposed change is in-distribution; lower means DiCE
  proposed an out-of-range (less actionable) value.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..schemas import (
    Attribution,
    CounterfactualExplanation,
    CounterfactualInstance,
    Explanation,
)
from .base import Explainer, ExplainerCapabilities, prediction_fn_available


class DiceCounterfactualExplainer(Explainer):
    """DiCE diverse counterfactuals. Local-only, async-eligible.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: dice_counterfactuals. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="dice",
        supported_model_types=("tree", "linear", "deep", "blackbox"),
        supports_local=True,
        supports_global=False,
        is_async_eligible=True,
        requires_background=True,  # DiCE needs the training frame for ranges
        component_id="dice_counterfactuals",
    )

    def __init__(self) -> None:
        try:
            import dice_ml  # noqa: F401
        except ImportError as exc:  # pragma: no cover -- declared as extra
            raise ImportError(
                "vfairness.xai.explainers.DiceCounterfactualExplainer requires the "
                "optional `dice-ml` extra: `pip install vfairness[xai]`."
            ) from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """DiCE wraps any model exposing predict (sklearn, or a torch/tf backend).

        G07 2026-09-30. This was ``callable(model) or hasattr(model, "predict")``,
        and ``hasattr`` is True for an UNFITTED sklearn estimator, so it answered
        True for ``RandomForestClassifier()``. The fitted-ness half now lives in
        ``prediction_fn_available``; see the measurement there.

        ``accept_predict_proba`` is False because this adapter's
        ``explain_local`` builds ``dice_ml.Model(..., backend="sklearn")`` around
        the object and DiCE reaches it through ``predict``: an object carrying
        only ``predict_proba`` answered False before this change too, and that
        part of the predicate is unchanged.
        """
        return prediction_fn_available(
            model,
            where="DiceCounterfactualExplainer.supports",
            accept_predict_proba=False,
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
        dataframe: Any | None = None,
        outcome_name: str | None = None,
        continuous_features: list[str] | None = None,
        **params: Any,
    ) -> CounterfactualExplanation:
        """Generate diverse counterfactuals for one instance.

        Requires the training ``dataframe`` (a pandas DataFrame including
        the ``outcome_name`` column) so DiCE can learn feature ranges.
        ``continuous_features`` lists the numeric columns; the rest are
        treated as categorical.
        """
        import dice_ml
        import pandas as pd

        if dataframe is None or outcome_name is None:
            raise ValueError(
                "DiceCounterfactualExplainer.explain_local requires `dataframe` "
                "(pandas, including the outcome column) and `outcome_name`."
            )

        feat_cols = [c for c in dataframe.columns if c != outcome_name]
        names = feature_names or feat_cols
        cont = continuous_features if continuous_features is not None else feat_cols
        backend = params.get("backend", "sklearn")
        total_cfs = int(params.get("total_CFs", 4))
        desired_class = params.get("desired_class", "opposite")
        seed = int(params.get("seed", 7))

        data = dice_ml.Data(
            dataframe=dataframe,
            continuous_features=list(cont),
            outcome_name=outcome_name,
        )
        dice_model = dice_ml.Model(model=model, backend=backend)
        engine = dice_ml.Dice(data, dice_model, method=params.get("method", "random"))

        query = pd.DataFrame([dict(zip(names, np.asarray(x).flatten().tolist()))])
        result = engine.generate_counterfactuals(
            query,
            total_CFs=total_cfs,
            desired_class=desired_class,
            random_seed=seed,
        )

        cf_df = result.cf_examples_list[0].final_cfs_df
        ranges = self._feature_ranges(dataframe, names)
        query_vals = dict(zip(names, np.asarray(x).flatten().tolist()))

        counterfactuals: list[CounterfactualInstance] = []
        if cf_df is not None:
            for _, cf_row in cf_df.iterrows():
                counterfactuals.append(
                    self._to_instance(cf_row, query_vals, names, outcome_name, ranges)
                )

        # The supporting "what drove the original decision" attributions
        # are produced by SHAP as the router's fallback evidence; here we
        # record the query values as the explanation's feature vector so
        # the audit bundle is self-contained.
        attributions = [
            Attribution(feature=n, contribution=float(query_vals[n]))
            for n in names
            if isinstance(query_vals[n], (int, float))
        ]
        original_pred = self._predict_label(model, query, backend)
        out_params: dict[str, Any] = {
            "total_CFs": total_cfs,
            "desired_class": desired_class,
            "method": params.get("method", "random"),
            "seed": seed,
            # A counterfactual is a set of CHANGES, not a contribution against a
            # reference output, so DiCE computes no base value and none was ever
            # computed here. It used to be sent as the literal 0.0, a number on
            # the output scale, beside counterfactuals that carry real ones. Same
            # disposition as anchors_adapter: the float field carries NaN and
            # this key says why, so a machine reader branches on the key.
            "base_value": None,
            "base_value_reason": (
                "DiCE returns counterfactual instances, not attributions against a "
                "reference output; there is no base value for this method and none "
                "was measured"
            ),
        }
        if not _is_number(original_pred):
            # prediction 0.0 is the DECLINED class, the very label a recourse
            # explanation is built for, so it may not stand in for "the model was
            # never asked". Measured 2026-09-27 with a model exposing
            # predict_proba only (DiCE generated 2 counterfactuals happily, and
            # _predict_label found neither .predict nor __call__): the returned
            # explanation carried prediction=0.0 and to_db_row wrote 0.0, with NO
            # warning at all on that branch. NaN now, plus this key.
            out_params["prediction_reason"] = (
                "COULD NOT MEASURE the original prediction: the model exposed no usable "
                "predict / __call__, or the call failed (see the RuntimeWarning). "
                "prediction is NaN, which is not the label 0."
            )
            warnings.warn(
                "DiCE original prediction COULD NOT BE MEASURED for "
                f"{instance_id!r}; recording NaN rather than 0.0, which is the declined "
                "class. See params['prediction_reason'].",
                RuntimeWarning,
                stacklevel=2,
            )

        return CounterfactualExplanation(
            method="dice",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=float("nan"),
            prediction=float(original_pred) if _is_number(original_pred) else float("nan"),
            attributions=attributions,
            units=params.get("units", "raw"),
            params=out_params,
            library_versions={"dice-ml": _dice_version()},
            counterfactuals=counterfactuals,
        )

    def explain_global(self, *args: Any, **kwargs: Any) -> list[Explanation]:
        raise NotImplementedError(
            "DiCE is a local (per-instance) counterfactual method; it has no "
            "global mode. capabilities.supports_global is False."
        )

    # helpers
    @staticmethod
    def _feature_ranges(dataframe: Any, names: list[str]) -> dict[str, dict[str, float]]:
        """Observed bounds and span per feature from the training frame.

        ``span`` normalises proximity; ``min``/``max`` ground the
        feasibility bound check (a counterfactual value outside the
        observed range is out-of-distribution, hence less actionable).
        Categorical columns get span 0 and no bounds.
        """
        ranges: dict[str, dict[str, float]] = {}
        for n in names:
            col = dataframe[n]
            try:
                lo, hi = float(col.min()), float(col.max())
                span = hi - lo
            except (TypeError, ValueError):  # categorical column
                lo = hi = float("nan")
                span = 0.0
            ranges[n] = {"min": lo, "max": hi, "span": span}
        return ranges

    def _to_instance(
        self,
        cf_row: Any,
        query_vals: dict[str, Any],
        names: list[str],
        outcome_name: str,
        ranges: dict[str, dict[str, float]],
    ) -> CounterfactualInstance:
        flipped: list[dict[str, Any]] = []
        norm_changes: list[float] = []
        in_range_hits: list[bool] = []
        for n in names:
            new_val = cf_row[n]
            old_val = query_vals[n]
            if _changed(old_val, new_val):
                flipped.append({"feature": n, "from": _coerce(old_val), "to": _coerce(new_val)})
                if _is_number(old_val) and _is_number(new_val):
                    span = ranges.get(n, {}).get("span", 0.0)
                    norm_changes.append(
                        abs(float(new_val) - float(old_val)) / (span if span > 0 else 1.0)
                    )
                    in_range_hits.append(self._within_observed(n, float(new_val), ranges))
        # Both are NaN, not the neutral-best number, when there was nothing to
        # measure. proximity 0.0 is the CHEAPEST counterfactual there can be
        # ("no distance at all") and feasibility 1.0 is "every proposed change is
        # in-distribution": the two strongest recourse scores, returned when no
        # numeric change existed to normalise or bound-check. Measured
        # 2026-09-27 on an all-categorical frame (region, band), model keyed on
        # region: both counterfactuals came back proximity=0.0 feasibility=1.0
        # for a flip of region alone, with no warning, so a categorical-only
        # recourse scored better than every numeric one DiCE can propose.
        # A flip with no measurable numeric part is still a real flip, so the
        # instance is returned; only the two scores say they were not measured.
        proximity = float(np.mean(norm_changes)) if norm_changes else float("nan")
        feasibility = float(np.mean(in_range_hits)) if in_range_hits else float("nan")
        if not norm_changes:
            changed = [f["feature"] for f in flipped]
            warnings.warn(
                "DiCE counterfactual has no measurable numeric change "
                + (f"(it changes only {changed})" if changed else "(it changes nothing)")
                + ", so proximity and feasibility COULD NOT BE MEASURED and are NaN, "
                "not 0.0 / 1.0 which would read as the closest, most feasible recourse "
                "possible.",
                RuntimeWarning,
                stacklevel=2,
            )
        outcome = cf_row[outcome_name] if outcome_name in cf_row else ""
        return CounterfactualInstance(
            flipped_features=flipped,
            predicted_outcome=_coerce(outcome),
            proximity=round(proximity, 6),
            feasibility=round(feasibility, 6),
        )

    @staticmethod
    def _within_observed(name: str, value: float, ranges: dict[str, dict[str, float]]) -> bool:
        # The documented feasibility contract: the new value must land
        # inside the observed [min, max] of the training frame. A
        # degenerate (span 0) or categorical feature cannot host a
        # plausible numeric change.
        r = ranges.get(name)
        if not r or r["span"] <= 0 or not np.isfinite(r["min"]):
            return False
        return bool(np.isfinite(value) and r["min"] <= value <= r["max"])

    @staticmethod
    def _predict_label(model: Any, query: Any, backend: str) -> Any:
        """Predict the original label; None (with a warning) on failure.

        A prediction failure must not silently become the label 0.0: the
        caller records prediction NaN when this returns a non-number, and
        the warning makes the gap visible in logs. Until 2026-09-27 the
        caller wrote 0.0 there, which is the declined class, so this
        warning was true and the recorded value was not.
        """
        try:
            if hasattr(model, "predict"):
                return model.predict(query)[0]
            if callable(model):
                return np.asarray(model(query)).flatten()[0]
        except Exception as exc:
            warnings.warn(
                f"DiCE original-prediction failed ({type(exc).__name__}: {exc}); "
                f"recording prediction as unavailable (NaN).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        # Model exposes neither predict nor __call__. This branch was SILENT and
        # the caller turned its None into 0.0; the caller now warns and records
        # NaN, so both routes to "not measured" are visible.
        return None


def _changed(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return abs(float(a) - float(b)) > 1e-9
    return str(a) != str(b)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _coerce(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _dice_version() -> str:
    try:
        import dice_ml

        return getattr(dice_ml, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"
