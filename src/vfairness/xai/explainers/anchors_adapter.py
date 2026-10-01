"""
vfairness.xai.explainers.anchors_adapter
========================================

Adapter around ``anchor-exp`` (Ribeiro, Singh & Guestrin 2018, AAAI) for
high-precision IF-THEN rule explanations. Phase 2 item #P2-04.

An anchor is a rule ``A`` such that, when ``A`` holds, the model
predicts the same class with high probability: ``precision(A) >= tau``
(the KL-LUCB target, default 0.95): over a stated ``coverage`` of the
input space. It answers "what minimal set of conditions is *sufficient*
for this decision?".

This is the deep equivalent of a sufficient-condition rule; it replaces
what Alibi's AnchorTabular would have provided (Alibi excluded by
OUT-01). The rule, precision and coverage are recorded in
``Explanation.params``; anchors do not produce per-feature numeric
attributions so ``attributions`` is left empty.

Note: ``anchor-exp`` pulls in ``spacy`` and currently has no wheel for
Python 3.9 (its ``thinc`` build-dep requires 3.10+). On a 3.10+ host it
installs cleanly; the adapter defers the import
to ``__init__`` so the rest of the package is unaffected when it is
absent.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from ..schemas import Explanation
from .base import Explainer, ExplainerCapabilities, prediction_fn_available

#: Smallest ``train_data`` sample an anchor's precision and coverage are read
#: from. Below it the KL-LUCB search has almost no perturbation space to sample
#: and degenerates: measured 2026-09-16 with a 2-row background it returned the
#: EMPTY rule with precision 1.0 and coverage 1.0, which reads as a perfect,
#: universally applicable explanation and is neither. The number is a stated
#: floor, not a guarantee; what matters is that the caller is told which side
#: of it this explanation was produced on.
MIN_BACKGROUND_ROWS = 30


def _unperturbable_features(background: np.ndarray, names: list[str]) -> list[str]:
    """Feature names whose ``train_data`` column carries NO spread.

    ``AnchorTabularExplainer`` samples its perturbation space by swapping values
    drawn from ``train_data``, so a column with one value is never varied: no
    candidate condition on that feature can be evaluated, and the KL-LUCB search
    cannot find one however much the model depends on it. The twin of this check
    in ``lime_adapter`` is about a surrogate weight; here it is about which
    conditions the search was able to consider at all.
    """
    if background.ndim != 2 or background.shape[0] == 0:
        return []
    out: list[str] = []
    for j in range(min(background.shape[1], len(names))):
        col = background[:, j]
        finite = col[np.isfinite(col)]
        # np.ptp, not a variance: max - min is not an accumulated sum, so an
        # exact comparison with zero is sound at every column length.
        if finite.size == 0 or float(np.ptp(finite)) == 0.0:
            out.append(names[j])
    return out


class AnchorsExplainer(Explainer):
    """anchor-exp tabular: high-precision sufficient-condition rules.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: anchors. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    capabilities = ExplainerCapabilities(
        method="anchors",
        supported_model_types=("blackbox", "tree", "linear"),
        supports_local=True,
        supports_global=False,
        is_async_eligible=True,
        requires_background=True,  # needs train_data for the perturbation space
        component_id="anchors",
    )

    def __init__(self) -> None:
        try:
            from anchor import anchor_tabular  # noqa: F401
        except ImportError as exc:  # pragma: no cover -- declared as extra
            raise ImportError(
                "vfairness.xai.explainers.AnchorsExplainer requires the optional "
                "`anchor-exp` extra: `pip install vfairness[xai]`. (anchor-exp needs "
                "Python 3.10+ for its spacy/thinc build dependency.)"
            ) from exc

    def supports(self, model: Any, model_type: str) -> bool:
        """Anchors is model agnostic: a usable label function is the whole test.

        G07 2026-09-30. This was ``callable(model) or hasattr(model, "predict")``,
        and ``hasattr`` is True for an UNFITTED sklearn estimator, so it answered
        True for ``RandomForestClassifier()``. The fitted-ness half now lives in
        ``prediction_fn_available``; see the measurement there.

        ``accept_predict_proba`` is False because ``_label_fn`` below reaches the
        model through ``predict`` or by calling it, never through
        ``predict_proba``: an object carrying only ``predict_proba`` answered
        False before this change too, and that part of the predicate is
        unchanged.
        """
        return prediction_fn_available(
            model,
            where="AnchorsExplainer.supports",
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
        class_names: list[str] | None = None,
        categorical_names: dict[int, list[str]] | None = None,
        threshold: float = 0.95,
        **params: Any,
    ) -> Explanation:
        from anchor import anchor_tabular

        if background is None or len(background) == 0:
            raise ValueError("AnchorsExplainer requires a background (train_data) sample.")
        x = np.asarray(x, dtype=float).flatten()
        n_features = x.shape[0]
        names = feature_names or [f"f{i}" for i in range(n_features)]
        predict_fn = self._label_fn(model)
        classes = class_names or ["0", "1"]

        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=classes,
            feature_names=names,
            train_data=np.asarray(background, dtype=float),
            categorical_names=categorical_names or {},
        )
        exp = explainer.explain_instance(x, predict_fn, threshold=threshold)
        predicted = int(np.asarray(predict_fn(x.reshape(1, -1))).reshape(-1)[0])

        n_background = int(len(background))
        rule = list(exp.names())
        precision = float(exp.precision())
        coverage = float(exp.coverage())
        bg_arr = np.asarray(background, dtype=float)
        if bg_arr.ndim == 1:
            bg_arr = bg_arr.reshape(1, -1)
        unperturbable = _unperturbable_features(bg_arr, names)
        out_params: dict[str, Any] = {
            "rule": rule,
            "precision": precision,
            "coverage": coverage,
            "threshold": threshold,
            "predicted_class": classes[predicted] if predicted < len(classes) else str(predicted),
            # An anchor is a sufficient-condition RULE. There is no reference
            # output to subtract a contribution from, so there is no base value
            # to report and none was ever computed here. It used to be sent as
            # the literal 0.0, which is a number, and it travelled into
            # Explanation.to_db_row and into the sidecar JSON payload beside
            # "fidelity": null and "stability": null -- the same record using
            # null for two unmeasured quantities and a measured-looking 0.0 for
            # a third. The float field now carries NaN and this key says why,
            # so a machine reader branches on the key rather than on isnan.
            "base_value": None,
            "base_value_reason": (
                "anchors produce a sufficient-condition rule, not a reference output; "
                "there is no base value for this method and none was measured"
            ),
            # Provenance for precision / coverage: the numbers above are read
            # off a perturbation space sampled from the background, so their
            # worth depends on how much background there was.
            "n_background": n_background,
            "background_sufficient": n_background >= MIN_BACKGROUND_ROWS,
            # Which features the search could not condition on, because their
            # background column holds a single value. Empty when all could.
            "unperturbable_features": unperturbable,
        }
        # An EMPTY rule states no condition, and its precision and coverage are
        # those of the whole perturbation space, so 1.0 / 1.0 reads as a perfect,
        # universally applicable explanation while nothing was found. Until now
        # that was disclosed only below MIN_BACKGROUND_ROWS. Measured 2026-09-27
        # with a 300-row background (sufficient by that floor) frozen in the one
        # feature the model uses: rule=[], precision=1.0, coverage=1.0,
        # background_sufficient=True and no warning of any kind, while the same
        # model and the same row against a varying background returned
        # rule=['c > 0.72'] with coverage 0.2459. The numbers are true of the
        # perturbation space and false of the model, so say which.
        if not rule:
            empty_note = (
                f"the anchor search returned the EMPTY rule, so precision {precision} "
                f"and coverage {coverage} are those of the whole perturbation space and "
                "state no sufficient condition for this decision. This is not a finding "
                "that the decision needs no conditions unless the model is constant"
                + (
                    f"; no rule could condition on {unperturbable}, which hold a single "
                    "value in the background, so a feature the model depends on may be "
                    "invisible to the search."
                    if unperturbable
                    else "."
                )
            )
            out_params["empty_rule_warning"] = empty_note
            warnings.warn(f"AnchorsExplainer: {empty_note}", UserWarning, stacklevel=2)
        elif unperturbable:
            unperturbable_note = (
                f"{len(unperturbable)} of {len(names)} background column(s) hold a "
                f"single value {unperturbable}, so the search could not evaluate any "
                "condition on them and their absence from the rule was NOT measured."
            )
            out_params["unperturbable_warning"] = unperturbable_note
            warnings.warn(f"AnchorsExplainer: {unperturbable_note}", UserWarning, stacklevel=2)
        if n_background < MIN_BACKGROUND_ROWS:
            note = (
                f"anchors ran against {n_background} background row(s), fewer than the "
                f"{MIN_BACKGROUND_ROWS} this adapter states as a floor. The precision "
                f"and coverage above are not reliable estimates"
                + (
                    "; the search returned the EMPTY rule, whose precision and coverage "
                    "are trivially 1.0 and describe no condition at all."
                    if not rule
                    else "."
                )
            )
            out_params["background_warning"] = note
            warnings.warn(f"AnchorsExplainer: {note}", UserWarning, stacklevel=2)

        warnings.warn(
            "AnchorsExplainer: base_value is NaN because an anchor is a rule, not a "
            "reference output. It is not a measured base value of zero. See "
            "params['base_value_reason'].",
            UserWarning,
            stacklevel=2,
        )

        return Explanation(
            method="anchors",
            instance_id=instance_id,
            subject_id=subject_id,
            model_hash=model_hash,
            data_hash=data_hash,
            # NaN, not 0.0. ``Explanation.base_value`` is annotated ``float``
            # in ``..schemas`` and that module belongs to another change, so
            # the numeric third state is NaN here rather than None; the
            # params keys above carry the same statement in a form a consumer
            # can branch on.
            base_value=float("nan"),
            prediction=float(predicted),
            attributions=[],  # anchors yield a rule, not per-feature weights
            units=params.get("units", "raw"),
            params=out_params,
            library_versions={"anchor-exp": _anchor_version()},
        )

    def explain_global(self, *args: Any, **kwargs: Any) -> list[Explanation]:
        raise NotImplementedError(
            "Anchors is a local (per-instance) rule method; capabilities.supports_global is False."
        )

    @staticmethod
    def _label_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
        # A label that is not a number is a decision the model DID NOT MAKE, and
        # ``astype(int)`` turned it into one. Measured 2026-10-01 (check 2, the
        # all_nan_scores world): a model returning NaN for every row was
        # explained as rule=[], precision 1.0, coverage 1.0, predicted_class '0',
        # prediction 0.0, i.e. "this applicant is declined, unconditionally, with
        # certainty", when no decision existed at all. NaN cast to int is
        # platform-defined (0 on arm64, INT64_MIN on x86), so every perturbed
        # sample landed on the same fake label and the anchor's precision was
        # perfect by construction. The only disclosure was the empty-rule
        # warning, which blames the search, not the model. Refuse instead: the
        # anchor search and the predicted class both read through here, so one
        # check covers both.
        raw_fn = model.predict if hasattr(model, "predict") else model

        def _labels(arr: np.ndarray) -> np.ndarray:
            raw = np.asarray(raw_fn(arr)).reshape(-1)
            if raw.dtype.kind in "fc" and not np.isfinite(raw).all():
                bad = int((~np.isfinite(raw)).sum())
                raise ValueError(
                    f"AnchorsExplainer COULD NOT MEASURE an anchor: the model returned "
                    f"{bad} of {raw.size} label(s) that are not finite numbers, so there "
                    f"is no decision to anchor. Casting them to int would invent a class "
                    f"and report a rule with perfect precision for it."
                )
            return raw.astype(int)

        return _labels


def _anchor_version() -> str:
    try:
        import anchor

        return getattr(anchor, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"
