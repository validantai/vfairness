"""
vfairness.xai.explainers.lime_adapter
=====================================

Adapter around the ``lime`` library (Ribeiro, Singh & Guestrin 2016)
that normalises a LIME tabular explanation into the ``Explanation``
contract. Phase 2 item #P2-02.

LIME fits a sparse linear surrogate to the model's behaviour in a local
neighbourhood of the instance. The router selects it for black-box
models, and as the downgrade target when the background is scarce or
out-of-distribution (KernelSHAP -> LIME with a wider kernel; see
``router.py``).

Every returned Explanation records:

* **fidelity**: the local surrogate R^2 (``lime``'s own ``exp.score``),
  i.e. how well the simple stand-in matched the real model locally.
* **stability**: the mean per-feature sigma of the attribution vector
  across repeated runs with different seeds
  (``diagnostics.attribution_stability``). Perturbation-based methods
  are seed-sensitive; this quantifies it.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np

from ..diagnostics.stability import attribution_stability
from ..schemas import Attribution, Explanation
from .base import Explainer, ExplainerCapabilities, prediction_fn_available

#: Spread a background column must carry, RELATIVE to that column's own
#: magnitude, before ``lime_tabular`` can be said to have varied it.
#:
#: Relative, and never an equality with zero. BGL5 A-xai-1, 2026-09-27: this test
#: was ``float(np.ptp(finite)) == 0.0``, and a background one part in 1e12 away
#: from constant is not exactly zero. Measured on 50 copies of one instance plus
#: a 1e-12 jitter, against a LogisticRegression whose prediction moves from
#: 0.9999996 to 6.3e-07 when feature 'c' flips sign:
#:     per-column peak-to-peak spread   4.0e-12 .. 5.3e-12
#:     per-column magnitude             1.32 .. 2.49
#:     spread relative to magnitude     1.6e-12 .. 3.3e-12
#: so the exact test answered "this background has spread", the refusal below did
#: not fire, every attribution came back at 1e-17 and
#: ``params['unperturbable_features']`` was ``[]``, which is the key a machine
#: reader is told to branch on POSITIVELY asserting that every feature was
#: measured. A healthy sample clears this floor by twelve orders of magnitude:
#: the same 300-row fixture measures 1.2 .. 2.4 relative spread.
_MIN_RELATIVE_SPREAD = 1e-9

#: Relative spread below which an attribution's MAGNITUDE is disclosed as
#: uninterpretable, even though the column was varied and the refusal above did
#: not fire.
#:
#: BGL6 F12, 2026-09-29. Raising :data:`_MIN_RELATIVE_SPREAD` from an exact
#: equality to 1e-9 moved the knife edge; it did not remove it, because the
#: DISCLOSURE had only two states. ``params['unperturbable_features']`` is a list:
#: a column is either in it (not measured) or absent from it, and absent is read
#: as measured. There was no third state for "varied, but by so little that the
#: number returned is not a statement about this feature". Measured on 50 copies of
#: one instance plus a 1e-8 jitter, three orders ABOVE the 1e-12 the BGL5 fix
#: pinned:
#:     per-column relative spread   1.8e-08 .. 4.5e-08   (clears 1e-9)
#: and against the LogisticRegression whose prediction moves 0.9999996 to 6.3e-07
#: when feature 'c' flips sign, every attribution came back at 1e-15 or below with
#: ``unperturbable_features == []`` and ZERO warnings; the global path over the same
#: background returned three explanations the same way. A reader was told every
#: feature was measured and handed four numbers that read as "no influence".
#:
#: What removes the knife edge is not a different number here: it is that the
#: measured spread is now PUBLISHED per column in
#: ``params['background_relative_spread']``, so no reader has to trust where this
#: floor sits. The floor only decides whether the warning fires, and it is placed
#: with four orders of margin on both sides: the failing background measures
#: 1.8e-08, and the healthy 300-row fixture measures 1.2 .. 2.4. A dataset that is
#: simply recorded in tiny units is NOT caught, because the test is relative: the
#: whole fixture scaled by 1e-12 still measures 2.4.
_MIN_INTERPRETABLE_RELATIVE_SPREAD = 1e-4


def _background_relative_spread(background: np.ndarray, names: list[str]) -> dict[str, float]:
    """Measured spread of each examined background column, relative to that
    column's own magnitude.

    The quantity both floors are compared against, returned so it can be published
    rather than only consulted. NaN for a column with no finite value at all: that
    is a could-not-check and must not be read as 0.0 spread, which is a measurement.

    Keyed on the same ``min(n_columns, n_names)`` columns
    :func:`_unperturbable_features` examines, so the two can never disagree about
    which columns were looked at.
    """
    if background.ndim != 2 or background.shape[0] == 0:
        return {}
    out: dict[str, float] = {}
    for j in range(min(background.shape[1], len(names))):
        col = background[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[names[j]] = float("nan")
            continue
        # np.ptp, matching _unperturbable_features: max minus min is a subtraction
        # of two values really in the data, where an accumulated statistic is
        # exactly 0.0 only for some n and some values.
        spread = float(np.ptp(finite))
        magnitude = float(np.max(np.abs(finite)))
        out[names[j]] = spread / max(magnitude, np.finfo(float).tiny)
    return out


def _unperturbable_features(background: np.ndarray, names: list[str]) -> list[str]:
    """Feature names whose background column carries NO USABLE spread.

    ``lime_tabular`` builds the neighbourhood by scaling a normal draw with each
    column's standard deviation taken from the background, so a column with zero
    spread is never varied and the surrogate weight it comes back with is not a
    measurement of the model at all. Non-finite columns count as unperturbable
    for the same reason: a NaN scale perturbs nothing usable.

    "No usable spread" is judged RELATIVE to the column's own magnitude, so the
    answer does not depend on the units the feature is recorded in: a column
    living at 1e-12 with a spread of 5e-13 IS perturbable, and a column living at
    2.5 with a spread of 5e-12 is not.
    """
    if background.ndim != 2 or background.shape[0] == 0:
        return []
    out: list[str] = []
    for j in range(min(background.shape[1], len(names))):
        col = background[:, j]
        finite = col[np.isfinite(col)]
        # np.ptp, not a variance: max - min is a subtraction of two values that
        # are really in the data, where an accumulated sum is exactly 0.0 only
        # for some n and some values.
        if finite.size == 0:
            out.append(names[j])
            continue
        spread = float(np.ptp(finite))
        magnitude = float(np.max(np.abs(finite)))
        # An all-zero column has magnitude 0.0, and then this is `spread <= 0.0`,
        # which is the exact test, correctly, for the one case where exact is
        # what the data says.
        if spread <= _MIN_RELATIVE_SPREAD * max(magnitude, np.finfo(float).tiny):
            out.append(names[j])
    return out


class LimeExplainer(Explainer):
    """LIME tabular: local sparse-linear surrogate. Async-eligible.

    THE GENERATED BLOCK BELOW IS OUT OF DATE for explain_local and explain_global. BGL5
    A-xai-1 (2026-09-27) closed an all-zero explanation reported as fully measured, for
    a background whose spread is only numerical noise and pinned both in
    tests/test_bgl5_xai_1.py, sabotage-checked, so "nothing in the suite holds it there"
    no longer describes them. That sentence is RENDERED by scripts/stamp_proof_status.py
    from src/vfairness/_proof_status.py and docs/capability-status.json, which this
    batch does not own: editing it here is reverted by the next stamp run and reported
    STALE by that script's own check pass (measured: 1 stale stamp at this line), which
    tests/test_beta_go_live_proof_ledger.py::test_no_docstring_stamp_is_stale asserts
    must be zero. The ledger regeneration is recorded as a hand-back.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: lime. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="lime",
        supported_model_types=("blackbox", "tree", "linear", "deep"),
        supports_local=True,
        supports_global=True,
        is_async_eligible=True,
        requires_background=True,  # needs training stats for the neighbourhood
        component_id="lime",
    )

    def __init__(self) -> None:
        try:
            import lime  # noqa: F401
            import lime.lime_tabular  # noqa: F401
        except ImportError as exc:  # pragma: no cover -- declared as extra
            raise ImportError(
                "vfairness.xai.explainers.LimeExplainer requires the optional `lime` "
                "extra: `pip install vfairness[xai]`."
            ) from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """LIME is model agnostic: a usable prediction function is the whole test.

        G07 2026-09-30. This was ``callable(model) or hasattr(model, "predict") or
        hasattr(model, "predict_proba")``, and ``hasattr`` is True for an UNFITTED
        sklearn estimator, so it answered True for ``RandomForestClassifier()``.
        The fitted-ness half now lives in ``prediction_fn_available``; see the
        measurement there.
        """
        return prediction_fn_available(model, where="LimeExplainer.supports")

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
        mode: str | None = None,
        num_samples: int = 1000,
        num_features: int | None = None,
        kernel_width: float | None = None,
        stability_reruns: int = 5,
        **params: Any,
    ) -> Explanation:
        if background is None or len(background) == 0:
            raise ValueError(
                "LimeExplainer requires a background sample for the local neighbourhood. "
                "LIME tolerates a scarcer / more out-of-distribution background than "
                "KernelSHAP, but it cannot run with none."
            )
        x = np.asarray(x, dtype=float).flatten()
        n_features = x.shape[0]
        names = feature_names or [f"f{i}" for i in range(n_features)]

        # A feature LIME cannot perturb gets a surrogate weight of exactly 0.0,
        # and 0.0 is the strongest all-clear an attribution can give ("this
        # feature did not influence the decision"). Measured 2026-09-27 against
        # a LogisticRegression whose only live coefficient is on feature "c"
        # (5.80 against <= 0.06 on the rest), explaining the row with the
        # largest c: with a background that varies in c, LIME reported
        # c = 0.5508, the dominant attribution; with the SAME model, the same
        # row and c frozen in the background, it reported c = 0.0000 exactly,
        # while the model's prediction still moves from 1.000000 to 0.000001
        # when c flips sign. The zero is not a finding about the model, it is
        # the absence of a measurement, so it is named rather than left to read
        # as one. This is NOT the SHAP case in
        # tests/test_xai_degenerate_attribution.py: SHAP attributes a DEVIATION
        # from the background, where zero is the correct answer for an instance
        # equal to its own background, while LIME regresses on a sampled
        # neighbourhood that here does not exist.
        bg_arr = np.asarray(background, dtype=float)
        if bg_arr.ndim == 1:
            bg_arr = bg_arr.reshape(1, -1)
        unperturbable = _unperturbable_features(bg_arr, names)
        # Count the columns that were actually EXAMINED, not the names supplied.
        # BGL5 A-xai-1, 2026-09-27: _unperturbable_features only looks at
        # min(n_columns, n_names) columns, so `len(unperturbable) == len(names)`
        # is 4 == 5 for a caller who passes one name too many. Measured on an
        # EXACTLY constant 50-row 4-column background with
        # feature_names=['a','b','c','d','e']: before, the refusal was skipped and
        # an Explanation was returned with a warning only; now the same input
        # raises, as it does with four names.
        n_examined = min(bg_arr.shape[1], len(names)) if bg_arr.ndim == 2 else 0
        if n_examined and len(unperturbable) == n_examined:
            raise ValueError(
                "LimeExplainer COULD NOT MEASURE any attribution for "
                f"{instance_id!r}: none of the {n_examined} background column(s) has "
                f"any usable spread ({bg_arr.shape[0]} row(s) supplied), and LIME samples "
                "its neighbourhood from the background's per-feature scale, so every draw "
                "equals the instance and the surrogate has nothing to regress on. "
                "Measured on a 50-row background of one repeated row: every "
                "attribution came back exactly 0.0 with fidelity 0.0 and no warning, "
                "which reads as 'no feature influenced this decision'. Measured again "
                "on that background plus a 1e-12 jitter, which an exact spread test "
                "reads as varying: every attribution came back at 1e-17 with "
                "unperturbable_features == [], which asserts the opposite. Supply a "
                "background whose columns vary by more than "
                f"{_MIN_RELATIVE_SPREAD:.0e} of their own magnitude, or use a "
                "deviation-based explainer, which can answer this input."
            )
        if unperturbable:
            warnings.warn(
                f"LimeExplainer: {len(unperturbable)} of {n_examined} background "
                f"column(s) have no spread {unperturbable}, so LIME never varied them "
                "and the 0.0 contribution reported for each was NOT measured. See "
                "params['unperturbable_features'].",
                UserWarning,
                stacklevel=2,
            )

        # THE THIRD STATE, BGL6 F12, 2026-09-29. Everything above has two states:
        # a column is in unperturbable_features or it is not, and "not" is read as
        # measured. So a background sitting just above the refusal floor published
        # four attributions of 1e-15, unperturbable_features == [] and no warning,
        # for a model whose prediction moves 0.9999996 to 6.3e-07 when one of those
        # features flips sign. Measured relative spread on that background:
        # 1.8e-08 .. 4.5e-08, four orders above _MIN_RELATIVE_SPREAD.
        #
        # These columns WERE varied, so calling them unperturbable would be false.
        # What could not be measured is the MAGNITUDE: the window lime explored is
        # 1.8e-08 of the feature's own size, so a contribution near zero is a
        # could-not-check about that feature's influence, not a finding of none.
        # Published as its own list plus the measured numbers, so the verdict does
        # not depend on where the floor sits.
        spreads = _background_relative_spread(bg_arr, names)
        barely = [
            name
            for name in list(spreads)
            if name not in unperturbable
            and not (spreads[name] >= _MIN_INTERPRETABLE_RELATIVE_SPREAD)
        ]
        if barely:
            shown = ", ".join(f"{name}={spreads[name]:.1e}" for name in barely)
            warnings.warn(
                f"LimeExplainer: {len(barely)} of {n_examined} background column(s) vary "
                f"by less than {_MIN_INTERPRETABLE_RELATIVE_SPREAD:.0e} of their own "
                f"magnitude ({shown}), so lime explored a window that small around the "
                "instance. They WERE perturbed, so they are not in "
                "unperturbable_features, but the MAGNITUDE of the attribution reported "
                "for each is not a statement about that feature's influence: a "
                "contribution near zero here is a COULD NOT CHECK, not a finding of no "
                "influence. See params['barely_perturbable_features'] and "
                "params['background_relative_spread'].",
                UserWarning,
                stacklevel=2,
            )

        k = num_features if num_features is not None else n_features
        run_mode = mode or ("classification" if hasattr(model, "predict_proba") else "regression")
        predict_fn = self._predict_fn(model, run_mode)
        seed = int(params.get("seed", 7))

        vector, score, intercept, local_pred = self._run_once(
            np.asarray(background, dtype=float),
            x,
            names,
            run_mode,
            predict_fn,
            k,
            num_samples,
            kernel_width,
            seed,
        )

        # Stability: rerun with perturbed seeds and measure the spread.
        reruns = [vector]
        for i in range(1, max(1, stability_reruns)):
            v, *_ = self._run_once(
                np.asarray(background, dtype=float),
                x,
                names,
                run_mode,
                predict_fn,
                k,
                num_samples,
                kernel_width,
                seed + i,
            )
            reruns.append(v)
        stability = attribution_stability(reruns)

        return Explanation(
            method="lime",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            base_value=float(intercept),
            prediction=float(local_pred),
            attributions=[
                Attribution(feature=n, contribution=float(v)) for n, v in zip(names, vector)
            ],
            units=params.get("units", "raw"),
            fidelity=None if score is None else float(score),
            stability=float(stability),
            params={
                "surrogate": "sparse_linear",
                "mode": run_mode,
                "num_samples": num_samples,
                "kernel_width": kernel_width,
                "seed": seed,
                "stability_reruns": stability_reruns,
                # Explicit not-assessed list, empty when every feature was
                # perturbable. A machine reader branches on this key rather
                # than trying to tell a measured 0.0 from an unmeasured one.
                #
                # It is not the whole answer on its own: a column can be varied by
                # so little that the attribution's magnitude still means nothing,
                # and that is the SIBLING key below, not this one. Read both.
                # BGL6 F12, 2026-09-29.
                "unperturbable_features": unperturbable,
                # The third state: varied, but across a window too small for the
                # attribution's magnitude to be a statement about the feature. A
                # near-zero contribution for a name in this list is a
                # could-not-check, not a finding of no influence.
                "barely_perturbable_features": barely,
                # The measured numbers both lists are derived from, per examined
                # column, so nothing here depends on trusting where a floor sits.
                # NaN for a column with no finite value, which is a could-not-check
                # and not a spread of 0.0.
                "background_relative_spread": spreads,
            },
            library_versions={"lime": _lime_version()},
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
                    **params,
                )
            )
        return out

    # helpers
    @staticmethod
    def _predict_fn(model: Any, run_mode: str) -> Callable[[np.ndarray], np.ndarray]:
        if run_mode == "classification":
            if hasattr(model, "predict_proba"):
                return model.predict_proba

            # Wrap a hard classifier / callable into a 2-col proba.
            def _proba(arr: np.ndarray) -> np.ndarray:
                raw = np.asarray(model(arr) if callable(model) else model.predict(arr)).reshape(-1)
                return np.column_stack([1 - raw, raw])

            return _proba
        if hasattr(model, "predict"):
            return lambda arr: np.asarray(model.predict(arr)).reshape(-1)
        return lambda arr: np.asarray(model(arr)).reshape(-1)

    def _run_once(
        self,
        background,
        x,
        names,
        run_mode,
        predict_fn,
        k,
        num_samples,
        kernel_width,
        seed,
        # score is Optional[float] BY DESIGN: a surrogate R^2 of 0.0 means the
        # stand-in explained none of the model, so it may never stand in for
        # "lime reported no score". The annotation said float while the body
        # returned None, which mypy caught only once the refusal landed.
    ) -> tuple[np.ndarray, Optional[float], float, float]:
        import lime.lime_tabular

        explainer = lime.lime_tabular.LimeTabularExplainer(
            background,
            feature_names=names,
            mode=run_mode,
            discretize_continuous=True,
            kernel_width=kernel_width,
            random_state=seed,
        )
        if run_mode == "classification":
            label = int(np.argmax(np.asarray(predict_fn(x.reshape(1, -1))).reshape(-1)))
            exp = explainer.explain_instance(
                x,
                predict_fn,
                labels=(label,),
                num_features=k,
                num_samples=num_samples,
            )
            key = label
        else:
            exp = explainer.explain_instance(
                x,
                predict_fn,
                num_features=k,
                num_samples=num_samples,
            )
            # lime keeps the regression weights under ``local_exp[dummy_label]``
            # (== 1) and OVERWRITES ``local_exp[0]`` with the NEGATED copy
            # (lime_tabular.py: `local_exp[0] = [(i, -1 * j) for i, j in
            # local_exp[1]]`). Reassigning an existing key preserves insertion
            # order, so `next(iter(exp.as_map()))` deterministically picked 0,
            # the negated copy, and EVERY attribution came back with its sign
            # inverted: a feature that drove the prediction up was reported as
            # driving it down, and base_value + sum(attributions) no longer
            # approached the reported prediction. lime's own as_list()/HTML use
            # dummy_label, so the adapter disagreed with lime about the sign of
            # every feature. Regression is the DEFAULT run_mode for any model
            # without predict_proba, including a classifier served as a bare
            # callable, so the inversion was total, not intermittent.
            key = getattr(exp, "dummy_label", 1)

        amap = exp.as_map()
        pairs = amap.get(key, next(iter(amap.values())))
        vector = np.zeros(x.shape[0])
        for idx, weight in pairs:
            vector[int(idx)] = float(weight)
        # fidelity/base_value are MEASUREMENTS or they are nothing. A surrogate
        # R^2 of 0.0 means "the stand-in explained none of the model", and an
        # intercept of 0.0 is a base value on the output scale: neither may
        # stand in for "lime reported none". Explanation.fidelity is Optional,
        # so None says could-not-check there; base_value is a float, so NaN does.
        _score = getattr(exp, "score", None)
        score = None if _score is None else float(_score)
        intercept = (
            float(exp.intercept.get(key, list(exp.intercept.values())[0]))
            if getattr(exp, "intercept", None)
            else float("nan")
        )
        local_pred = float(np.asarray(getattr(exp, "local_pred", [intercept])).reshape(-1)[0])
        return vector, score, intercept, local_pred


def _lime_version() -> str:
    try:
        import lime

        return getattr(lime, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"
