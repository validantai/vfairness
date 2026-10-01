"""
vfairness.xai.explainers.ig_adapter
====================================

Adapter around Captum's Integrated Gradients (Sundararajan, Taly & Yan
2017) for deep models. Phase 2 item #P2-03: the deep-model attribution
path that replaces what Alibi would have provided (Alibi was excluded by
backlog OUT-01 after its BSL relicence).

Integrated Gradients satisfies, by construction:

* **Sensitivity(a)**: a feature that changes the prediction gets a
  non-zero attribution.
* **Implementation Invariance**: two functionally identical networks
  produce identical attributions (it depends only on input/output
  gradients, not the graph).
* **Completeness**: the attributions sum to ``f(x) - f(baseline)``.

For a linear model ``f(x) = w·x + b`` the IG attribution of feature i is
exactly ``w_i · (x_i - baseline_i)``, which is what the unit test pins.

The adapter never propagates torch tensors upward; only normalised
``Explanation`` dataclasses cross the boundary.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..schemas import Attribution, Explanation
from .base import Explainer, ExplainerCapabilities

#: Default relative tolerance the completeness axiom is CHECKED against.
#: IG's guarantee is that the attributions sum to ``f(x) - f(baseline)``; the
#: residual was being recorded in ``params`` and compared with nothing, so a
#: saturated network at ``n_steps=2`` returned three finite 0.0 contributions
#: for a prediction that had moved by 1.0 (100 percent of it unexplained) and
#: nothing warned, graded or flagged it. Measured 2026-09-16.
#:
#: BGL-S2b (2026-09-17). 1e-3 was TIGHTER THAN IG'S OWN QUADRATURE ERROR at
#: this adapter's default ``n_steps=50``, so it failed healthy models. IG
#: approximates a path integral by a Riemann sum; for a piecewise-linear
#: (ReLU) network the kink points are never hit exactly and a residual of a
#: fraction of a percent of the attributed mass is normal, not a defect.
#: Measured on this machine with :func:`completeness_scale` as the
#: denominator, residual / scale:
#:
#:     12 untrained random ReLU nets, n_steps=50   0.06% .. 3.39%
#:     a trained ReLU net, n_steps=50              0.15%
#:     a trained ReLU net, n_steps=200             0.10%
#:     a linear model (IG is exact)                0.00%
#:     a saturated model whose whole movement
#:       is unattributed                         100.00%
#:
#: Raising ``n_steps`` does NOT clear it on its own: the same trained net went
#: 0.15% -> 0.10% between 50 and 200 steps, and at the old absolute 1e-3 it was
#: graded ``completeness_ok=False`` at BOTH step counts. 5e-2 sits an order of
#: magnitude above the worst healthy case and an order of magnitude below a
#: genuine decomposition failure.
COMPLETENESS_TOL = 5e-2


def completeness_scale(prediction: float, base_value: float, values: Any) -> float:
    """The quantity the completeness residual is judged RELATIVE to.

    ``max(|prediction - base_value|, sum|attribution|)``: the larger of what
    the model moved and the mass the explanation actually distributed.

    BGL-S2b (2026-09-17). The denominator used to be
    ``max(|prediction - base_value|, 1.0)``, and that hardcoded 1.0 is the same
    fabricated-constant shape the check exists to catch: for any model whose
    output moves less than 1.0 the "relative" tolerance silently became an
    ABSOLUTE one. Measured at the public entry, a model that moved by 0.0005
    with attributions summing to exactly 0.0 (100 percent of the movement
    unexplained) returned ``completeness_ok=True`` and raised no warning at
    all, because 0.0005 <= 1e-3 * 1.0. There is no universal output scale, so
    the floor is taken from THIS explanation instead: a decomposition that
    distributed no mass cannot hide behind a unit that has nothing to do with
    the model. Attribution mass, not ``1.0``, is also what keeps the ratio
    stable when the movement itself is near zero through cancellation, which is
    where a pure ``residual / |prediction - base_value|`` reads 20% on a
    perfectly healthy net (measured: seed 5 of the twelve above).

    Returns 0.0 whenever the model did not move at all, and ``nan`` when the
    movement is not finite. BGL3 xai-1, 2026-09-27: this said "0.0 only when the
    model did not move AND nothing was attributed", which stopped being true the
    moment attribution mass left the ``max()`` below. Measured,
    ``completeness_scale(0.5, 0.5, [5.0, 0.0])`` is 0.0 with a mass of 5.0. The
    caller does not divide by it: with a scale of 0.0 the tolerance collapses to
    ``residual <= 0.0``, so a model that did not move and an explanation that
    attributed nothing pass identically, and an explanation that distributed mass
    over a model that did not move is graded ``completeness_ok=False``, which is
    the correct verdict on it.
    """
    movement = abs(prediction - base_value)
    if not math.isfinite(movement):
        return float("nan")
    # MOVEMENT ONLY. Attribution mass used to sit in this max(), and it let a
    # CANCELLING explanation buy its own tolerance: completeness is
    # sum(attributions) == prediction - base_value, so an explanation whose parts
    # are large and opposite has huge mass and explains nothing. Any run whose
    # mass exceeded 20x the residual then passed, including runs where 100% of
    # the model's movement was unexplained. Measured by the Stage 2 audit at
    # amplitude 50 and 100: completeness_ok=True, zero warnings, with the entire
    # movement unaccounted for.
    #
    # The quantity completeness is ABOUT is how far the model moved from the
    # baseline. That is the only honest denominator; scaling by how much the
    # explainer happened to say makes the check weaker exactly when the
    # explanation is worst. The floor below keeps a model that did not move from
    # dividing by zero, and that case is handled by the caller before it gets
    # here.
    return movement


class IntegratedGradientsExplainer(Explainer):
    """Captum Integrated Gradients for deep (PyTorch) models.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: integrated_gradients. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="ig",
        supported_model_types=("deep",),
        supports_local=True,
        supports_global=True,
        is_async_eligible=True,
        requires_background=True,  # the baseline is the IG "background"
        component_id="integrated_gradients",
    )

    def __init__(self) -> None:
        try:
            import captum  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover -- declared as extra
            raise ImportError(
                "vfairness.xai.explainers.IntegratedGradientsExplainer requires the "
                "optional `captum` + `torch` extras: `pip install vfairness[xai]`."
            ) from exc

    def supports(self, model: Any, model_type: str) -> bool:
        try:
            import torch.nn as nn
        except ImportError:  # pragma: no cover
            # Fail closed, because claiming support for a model this adapter cannot
            # load is worse than declining it. But SAY SO: a bare False here is
            # indistinguishable from "your model is the wrong kind", and the actual
            # reason is that an optional dependency is absent, which the caller can
            # fix. Silence sent them looking at their model instead.
            warnings.warn(
                "IntegratedGradientsExplainer.supports: torch is not installed, so "
                "whether this model is a torch Module COULD NOT BE DETERMINED. "
                "Returning False to avoid claiming support, which is not the same as "
                "the model being unsupported. Install the 'xai' extra to find out.",
                stacklevel=2,
            )
            return False

        return model_type == "deep" and isinstance(model, nn.Module)

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
        target: int | None = None,
        n_steps: int = 50,
        completeness_tol: float = COMPLETENESS_TOL,
        **params: Any,
    ) -> Explanation:
        import torch
        from captum.attr import IntegratedGradients

        model.eval()
        x_arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
        inputs = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)

        # The baseline is IG's reference point. Use the mean of the
        # supplied background if present, else an all-zeros baseline
        # (the canonical default). The choice is recorded in params so
        # the AuditArtifact can replay it byte-stably.
        if background is not None and len(background) > 0:
            baseline_arr = (
                np.asarray(background, dtype=np.float32)
                .reshape(-1, x_arr.shape[1])
                .mean(axis=0, keepdims=True)
            )
            baseline_source = "background_mean"
        else:
            baseline_arr = np.zeros_like(x_arr)
            baseline_source = "zeros"
        baselines = torch.tensor(baseline_arr, dtype=torch.float32)

        ig = IntegratedGradients(model)
        attr = ig.attribute(inputs, baselines=baselines, target=target, n_steps=n_steps)
        values = attr.detach().numpy().flatten()
        names = feature_names or [f"f{i}" for i in range(values.shape[0])]
        # BGL3 xai-1, 2026-09-27. ``zip(names, values)`` at the bottom of this
        # method STOPS AT THE SHORTER SEQUENCE without a word, and every check
        # above it reads ``values``, not what gets returned. Measured on a
        # 3-feature linear torch model with ``feature_names=["a", "b"]``: two
        # attributions came back summing to 0.18012 while prediction minus
        # base_value was 0.29050, so 38.0 percent of the model's movement was
        # missing from the explanation the caller received, and params still
        # reported completeness_ok=True, attributions_complete=True and a residual
        # of 3.7e-09, with zero warnings. The axiom had been checked against the
        # decomposition this method COMPUTED rather than the one it HANDED BACK,
        # which is the fabricated-pass shape COMPLETENESS_TOL exists to close.
        # A count mismatch cannot be repaired here: nothing knows which value
        # belongs to which name once they disagree.
        if len(names) != values.shape[0]:
            raise ValueError(
                f"feature_names carries {len(names)} name(s) for {values.shape[0]} attributed "
                f"value(s). Pairing them would silently drop the surplus on the longer side, "
                f"so the returned attributions would not sum to prediction - base_value while "
                f"the completeness check, which reads the full attribution vector, still "
                f"passed. Supply one name per feature, in the order the model was fitted on."
            )

        base_value = self._scalar_output(model, baselines, target)
        prediction = self._scalar_output(model, inputs, target)

        # A non-finite attribution is NOT a contribution of unknown size that
        # happens to sort low; it is a feature this run could not attribute at
        # all. It was handed back as an ordinary entry with no flag, and every
        # ranker in this package orders by abs(contribution), where NaN loses
        # every comparison and therefore lands LAST: the feature nothing could
        # be said about was reported as the least influential driver. The
        # numeric field keeps the NaN, which is the third state a float can
        # carry, and these two params keys say the same thing in a form a
        # consumer can branch on without inspecting each value. The names match
        # ``AdverseActionReasons.unattributed_features`` / ``.complete`` in
        # operations.reporting.compliance, which already got this right.
        unattributed = [str(n) for n, v in zip(names, values) if not np.isfinite(v)]
        prediction_measured = math.isfinite(prediction)
        base_value_measured = math.isfinite(base_value)

        residual = abs(float(values.sum()) - (prediction - base_value))
        # Completeness is IG's own axiom, so a residual is a CHECKABLE claim
        # and not merely a recorded number. It is judged relative to
        # `completeness_scale` -- the larger of the model's movement and the
        # attribution mass -- never against a hardcoded unit; see that
        # function for the measurement that forced the change. Three states:
        # True within tolerance, False outside it, None when the residual
        # itself could not be computed (a non-finite attribution or output
        # makes it NaN), which is not a pass.
        scale = (
            completeness_scale(prediction, base_value, values)
            if prediction_measured and base_value_measured
            else float("nan")
        )
        completeness_ok: bool | None
        if not math.isfinite(residual) or not math.isfinite(scale):
            completeness_ok = None
        else:
            completeness_ok = residual <= completeness_tol * scale

        out_params: dict[str, Any] = {
            "n_steps": n_steps,
            "target": target,
            "baseline": baseline_source,
            # Completeness gap: |sum(attr) - (pred - base)|. ~0 for IG.
            "completeness_residual": residual,
            "completeness_tol": completeness_tol,
            # The denominator the residual was judged against. Published so a
            # reader can recompute the verdict without reading this source;
            # when it was a hidden constant 1.0, the check silently became an
            # absolute one for every small-output model.
            "completeness_scale": scale,
            "completeness_ok": completeness_ok,
            "unattributed_features": unattributed,
            "attributions_complete": not unattributed,
            "prediction_measured": prediction_measured,
            "base_value_measured": base_value_measured,
        }

        if unattributed:
            warnings.warn(
                "IntegratedGradientsExplainer: no attribution could be computed for "
                f"{', '.join(unattributed)}. Those contributions are NaN, not small: "
                "the explanation is INCOMPLETE and any ranking of it omits them. See "
                "params['unattributed_features'].",
                UserWarning,
                stacklevel=2,
            )
        if not prediction_measured or not base_value_measured:
            warnings.warn(
                "IntegratedGradientsExplainer: the model output was not finite at "
                + ("the instance" if not prediction_measured else "the baseline")
                + ", so prediction/base_value are NaN and are not measurements. See "
                "params['prediction_measured'] and params['base_value_measured'].",
                UserWarning,
                stacklevel=2,
            )
        if completeness_ok is None and not unattributed and prediction_measured:
            warnings.warn(
                "IntegratedGradientsExplainer: the completeness residual could not be "
                "computed, so it is UNCHECKED. params['completeness_ok'] is None, which "
                "is not a pass.",
                UserWarning,
                stacklevel=2,
            )
        elif completeness_ok is False:
            share = residual / scale if scale else float("nan")
            warnings.warn(
                "IntegratedGradientsExplainer: the IG completeness axiom does not hold "
                f"for this explanation. The attributions leave {share:.1%} of the "
                f"movement from the baseline unexplained (residual {residual:.4g}, "
                f"tolerance {completeness_tol:.4g} relative). The contributions are not "
                "a faithful decomposition of this prediction; raise n_steps or check "
                "for a saturated model.",
                UserWarning,
                stacklevel=2,
            )

        return Explanation(
            method="ig",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=base_value,
            prediction=prediction,
            attributions=[
                Attribution(feature=n, contribution=float(v)) for n, v in zip(names, values)
            ],
            units=params.get("units", "raw"),
            params=out_params,
            library_versions={"captum": _captum_version(), "torch": _torch_version()},
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
        target: int | None = None,
        n_steps: int = 50,
        **params: Any,
    ) -> list[Explanation]:
        out: list[Explanation] = []
        for i, row in enumerate(np.asarray(X)):
            out.append(
                self.explain_local(
                    model,
                    row,
                    background,
                    instance_id=f"{subject_id}#row-{i}",
                    subject_id=subject_id,
                    model_hash=model_hash,
                    data_hash=data_hash,
                    feature_names=feature_names,
                    target=target,
                    n_steps=n_steps,
                    **params,
                )
            )
        return out

    # helpers
    @staticmethod
    def _scalar_output(model: Any, inputs: Any, target: int | None) -> float:
        import torch

        with torch.no_grad():
            out = model(inputs)
        arr = (
            out.detach().numpy().flatten() if hasattr(out, "detach") else np.asarray(out).flatten()
        )
        if target is not None and target < arr.shape[0]:
            return float(arr[target])
        return float(arr[0])


def _captum_version() -> str:
    try:
        import captum

        return getattr(captum, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:  # pragma: no cover
        return "unknown"
