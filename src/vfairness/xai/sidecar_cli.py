"""
vfairness.xai.sidecar_cli
=========================

CLI entrypoint for the XAI sidecar. Runs an exact SHAP-family explainer in a
Python <=3.13 venv that has `shap` installed, on behalf of the Python-3.14
consumer that cannot install it (numba supports only <3.14). Mirrors the causal
sidecar CLI (``vfairness.operations.causal.task_handlers``): reads ONE JSON
payload on stdin, prints ONE JSON envelope on stdout.

    echo '<payload>' | python -m vfairness.xai.sidecar_cli

Payload: { method?, model_base64?, model_format?, trust_input?, endpoint_url? +
auth_*, allow_loopback?, csv_data, feature_columns?, row_index?, goal?, counterfactual_needed?,
ood_risk?, subject_id? }. Loading a `model_base64` blob deserializes untrusted
bytes (joblib/pickle/skops run arbitrary code), so it is refused unless the
caller sets `trust_input: true` or the operator sets the env var
``VFAIRNESS_TRUST_MODEL_INPUT=1`` (see VB-SEC-6).
If `method` is omitted the router picks one (TreeSHAP for trees,
LinearSHAP for linear, KernelSHAP for black-box) and the CLI walks its
fallbacks. Output: { success, data: { routing, method_used, method_explicit,
fallback_used, model_type, explanation } } or { success: False, error }.

WHAT THE EXPLANATION OBJECT CARRIES, and why more than it used to. Beside the
numbers (base_value, prediction, units, fidelity, stability, attributions) it
carries the explainer's own three-state disclosure, because a number here is
worth nothing without it:

    params                     the adapter's params dict, verbatim
    prediction_measured        True / False / null (null = not reported, which
    base_value_measured        is a could-not-check and never a False)
    attributions_complete
    unattributed_features      the features that carry NO SHAP value
    local_accuracy_residual    null when the identity could not be checked
    local_accuracy_ok
    attributions_measured      True / False / null, with
    attributions_measured_reason when it is not True
    background_rows            how many rows the background holds
    background_available       whether len(csv) >= 10, on EVERY path
    background_is_the_explained_instance

Non-finite numbers are rendered as JSON null, never as the bare token NaN: the
envelope is valid JSON for any parser, and the ``*_measured`` flags say why a
number is missing. See :func:`_json_ready` for the measurement that forced it.

Standalone module (NOT under xai.worker) so importing it does not pull the
pgmq WorkerLoop runner.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
from typing import Any, cast

import numpy as np

from vfairness.xai.schemas import ExplainerMethod, ModelType

# VB-SEC-6: this CLI reads ONE JSON payload from stdin, which is untrusted
# input. joblib/pickle and skops(trusted=True) run arbitrary code at
# deserialization time, so decoding a `model_base64` blob is a remote-code-
# execution sink. Loading is therefore gated behind an explicit opt-in: the
# caller must set `payload["trust_input"] = true` OR the operator must set the
# env var below to a truthy value. Without opt-in `_load_model` raises a clear
# error naming the risk. This preserves backward-compatible behavior for
# callers who opt in while refusing silent deserialization of untrusted bytes.
TRUST_INPUT_ENV = "VFAIRNESS_TRUST_MODEL_INPUT"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_opt_in() -> bool:
    return os.environ.get(TRUST_INPUT_ENV, "").strip().lower() in _TRUTHY


def _load_model(model_base64: str, model_format: str, trust_input: bool = False):
    # VB-SEC-6: refuse to deserialize untrusted model bytes unless the caller
    # or operator has explicitly opted in. joblib.load / skops(trusted=True)
    # execute arbitrary code during unpickling (RCE), so a hostile
    # `model_base64` could take over the sidecar process.
    if not (trust_input or _env_opt_in()):
        raise ValueError(
            "sidecar: refusing to deserialize an untrusted model blob. "
            "joblib/pickle and skops(trusted=True) run arbitrary code while "
            "loading, so a hostile model_base64 is a remote-code-execution "
            "risk. Only load models from a source you fully trust, then opt in "
            "by setting payload['trust_input'] = true or the environment "
            f"variable {TRUST_INPUT_ENV}=1."
        )
    raw = base64.b64decode(model_base64)
    fmt = (model_format or "joblib").lower()
    if fmt in ("joblib", "pkl", "pickle"):
        import joblib

        # nosec B301 - only reachable after an explicit trust_input opt-in in
        # _load_model above (see the deserialization-safety test); bandit cannot
        # see that guard, so annotate rather than fail CI on a gated, tested path.
        return joblib.load(io.BytesIO(raw))  # nosec B301
    if fmt == "skops":
        import skops.io as sio

        # trusted=True disables skops' type allowlist entirely; only reachable
        # once the caller has already opted into trusting this input above.
        return sio.load(io.BytesIO(raw), trusted=True)
    raise ValueError(f"sidecar: unsupported model_format {model_format!r}")


def _model_callable(model):
    """Wrap a fitted estimator into f(X)->1D scores (positive class proba)."""

    def f(X):
        X = np.asarray(X, dtype=float)
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X), dtype=float)
            return p[:, -1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
        return np.asarray(model.predict(X), dtype=float).ravel()

    return f


def _endpoint_callable(payload, feature_columns):
    """Build a predict-fn that POSTs rows to a caller-supplied model endpoint.

    Egress guard (VF-2). ``endpoint_url`` comes from the stdin payload, which
    this module's own header already calls untrusted input, and every POST
    carries ``auth_token`` in an ``Authorization``/``x-api-key`` header. An
    unvalidated URL therefore both reaches internal services AND hands them the
    credential. Until 2026-08-27 this was a bare ``requests.post`` with
    redirects followed unchecked, so a 302 to 169.254.169.254 forwarded the
    bearer token to cloud metadata. It now runs through
    :func:`vfairness.net.egress.guarded_post`, which vets the URL, pins the
    connection to the vetted IP and re-vets every redirect hop.

    ``allow_loopback`` in the payload is the explicit opt-in for a model server
    on this machine; it unlocks loopback only, never RFC1918 / link-local /
    cloud-metadata.
    """
    from vfairness.net.egress import guarded_post, validate_endpoint

    url = payload["endpoint_url"]
    allow_loopback = bool(payload.get("allow_loopback", False))
    # Vet once, up front, so a forbidden endpoint fails with a clear SSRFError
    # before any row is serialised or any credential is attached. guarded_post
    # re-vets on every hop after this.
    validate_endpoint(url, allow_http=True, allow_loopback=allow_loopback)
    headers = {"Content-Type": "application/json"}
    at, tok = payload.get("auth_type", "none"), payload.get("auth_token")
    if at == "bearer" and tok:
        headers["Authorization"] = f"Bearer {tok}"
    elif at == "api_key" and tok:
        headers["x-api-key"] = tok
    mk = payload.get("response_mapping_key", "predictions")
    pk = payload.get("probability_key", "probabilities")
    # KernelSHAP and LIME call the model with large coalition batches (often
    # thousands of rows). A single POST of every row risks an upstream payload
    # limit or timeout, so split into bounded chunks and concatenate.
    chunk = int(payload.get("endpoint_batch_size", 100)) or 100

    def _post_chunk(rows):
        # guarded_post, not requests.post: these headers carry the caller's
        # model-endpoint credential.
        r = guarded_post(
            url,
            allow_http=True,
            allow_loopback=allow_loopback,
            json={"instances": rows},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        src = None
        if isinstance(d, dict):
            for k in (pk, "probabilities", "probs"):
                if d.get(k) is not None:
                    src = d[k]
                    break
            if src is None:
                for k in (mk, "predictions", "outputs", "scores"):
                    if d.get(k) is not None:
                        src = d[k]
                        break
        elif isinstance(d, list):
            src = d
        arr = np.asarray(src, dtype=float)
        return arr[:, -1] if arr.ndim == 2 else arr.ravel()

    def f(X):
        instances = np.asarray(X).tolist()
        if len(instances) <= chunk:
            return _post_chunk(instances)
        parts = [_post_chunk(instances[i : i + chunk]) for i in range(0, len(instances), chunk)]
        return np.concatenate([np.atleast_1d(p) for p in parts])

    return f


# The three-state disclosure the SHAP adapters compute on Explanation.params. The
# envelope used to hand-list nine fields and carry none of these six, so the whole
# disclosure died at this boundary while the producer-side pin
# (tests/test_bgl_stage2_s2g09.py, "local_accuracy_residual is None") stayed green.
# A field an adapter does not report reaches the reader as null, which is the
# could-not-check: it is NOT turned into False.
_DISCLOSURE_FIELDS = (
    "prediction_measured",
    "base_value_measured",
    "attributions_complete",
    "unattributed_features",
    "local_accuracy_residual",
    "local_accuracy_ok",
)


def _json_ready(value: Any) -> Any:
    """Render *value* so ``json.dumps(..., allow_nan=False)`` cannot refuse it.

    THE ENVELOPE WAS NOT VALID JSON (2026-09-27). ``json.dumps`` writes the bare
    token ``NaN``, which RFC 8259 forbids, so on an unmeasurable explanation this
    module's documented "ONE JSON envelope on stdout" was a malformed reply:
    measured with node, 'JSON.parse FAILED: Unexpected token N ... "base_value":
    NaN'. Python's ``json.loads`` accepts it, so a Python consumer carried the NaN
    onward in silence and any other consumer saw a parse error rather than a
    refusal. Either way the could-not-check was never rendered as one.

    A non-finite number is now ``null``, which is JSON for "no value", and the
    ``*_measured`` flags beside it say WHY there is none. numpy scalars are
    unwrapped here too: ``json.dumps`` raises TypeError on ``np.float64``, which
    inside the candidate loop would have been recorded as a method failure.
    """
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _attribution_state(params: dict, background_is_instance: bool, n_background: int) -> tuple:
    """(measured, reason): three states, and never two.

    ``True`` only when an adapter said every feature was attributed; ``False`` with
    a reason when something makes the contributions unreadable; ``None`` when
    nothing in this run establishes either, which is a could-not-check and not a
    pass.
    """
    if background_is_instance:
        # MEASURED 2026-09-27: a 1-row CSV makes background == the explained
        # instance (background = X[: min(50, len(X))]), so every SHAP value is
        # exactly 0.0 by construction and the envelope reported
        # "contribution": 0.0 three times under success=true, which reads as "no
        # feature mattered" for a model whose only real driver is feature a. The
        # same thing happens for 60 identical rows, which a row COUNT would miss,
        # so the test is whether the background differs from the instance at all.
        return False, (
            f"the background is the explained instance itself ({n_background} row(s), "
            "none of them different from it), so every SHAP value is 0.0 by "
            "construction: no contribution was measured, and 0.0 here does not mean "
            "the feature did not matter"
        )
    complete = params.get("attributions_complete")
    if complete is False:
        missing = params.get("unattributed_features") or []
        return False, (
            f"no SHAP value could be computed for {', '.join(str(f) for f in missing)}: "
            "those contributions are could-not-check, not small"
        )
    if complete is True:
        return True, None
    return None, (
        "this explainer reports no completeness flag, so whether every feature was "
        "attributed was not checked in this run"
    )


def _detect_model_type(model) -> ModelType:
    if model is None:
        return "blackbox"
    n = type(model).__name__.lower()
    if (
        hasattr(model, "estimators_")
        or hasattr(model, "tree_")
        or any(t in n for t in ("forest", "tree", "boost", "xgb", "lgbm", "catboost"))
    ):
        return "tree"
    if hasattr(model, "coef_") or "linear" in n or "logistic" in n:
        return "linear"
    return "blackbox"


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"sidecar: bad payload ({exc})"}))
        return
    try:
        import pandas as pd

        from vfairness.xai.explainers.registry import get_explainer
        from vfairness.xai.explainers.router import route_explainer

        df = pd.read_csv(io.StringIO(payload["csv_data"]))
        cols = payload.get("feature_columns") or [str(c) for c in df.columns]
        X = df[cols].to_numpy().astype(float)
        row_index = int(payload.get("row_index", 0))
        if not (0 <= row_index < len(X)):
            print(
                json.dumps(
                    {"success": False, "error": f"sidecar: row_index {row_index} out of range"}
                )
            )
            return
        x = X[row_index]
        background = X[: min(50, len(X))]  # cap background for KernelSHAP cost
        # COMPUTED ABOVE THE DISPATCH, DELIBERATELY (2026-09-27). `len(X) >= 10` was
        # worked out further down and fed ONLY to route_explainer, and an explicit
        # payload['method'] skips the router entirely, so on that path nothing
        # consulted background sufficiency at all: a guard below a dispatch that
        # already chose cannot fire. Both facts are now established once, here,
        # before any branch, and both reach the envelope on every path.
        n_background = int(len(background))
        background_available = bool(len(X) >= 10)
        # Exact comparison on purpose. It catches a 1-row background AND 60 identical
        # rows, which a row count would miss, and NaN != NaN keeps the all-NaN case
        # with its own disclosure instead of relabelling it as a degenerate
        # background.
        background_is_instance = bool(n_background > 0 and np.all(background == x))

        model = (
            _load_model(
                payload["model_base64"],
                payload.get("model_format", "joblib"),
                trust_input=bool(payload.get("trust_input", False)),
            )
            if payload.get("model_base64")
            else None
        )
        endpoint_fn = _endpoint_callable(payload, cols) if payload.get("endpoint_url") else None

        # payload.get is untrusted JSON (Any); the router treats any unknown
        # model_type as blackbox, so narrowing to ModelType here is safe and
        # changes no behaviour. Falls back to detection when unset.
        model_type: ModelType = (
            cast(ModelType, payload["model_type"])
            if payload.get("model_type")
            else _detect_model_type(model)
        )
        routing = None
        if payload.get("method"):
            candidates = [payload["method"]]
        else:
            dec = route_explainer(
                model_type=model_type,
                goal=payload.get("goal", "single_decision"),
                counterfactual_needed=bool(payload.get("counterfactual_needed", False)),
                background_available=len(X) >= 10,
                ood_risk=payload.get("ood_risk", "low"),
            )
            routing = {
                "primary": dec.primary,
                "fallbacks": list(dec.fallbacks),
                "downgraded_from": dec.downgraded_from,
                "reason": dec.reason,
                "component_id": dec.component_id,
            }
            candidates = [dec.primary, *dec.fallbacks]

        last_err = None
        for method in candidates:
            m = str(method)
            # Attribution methods only (shap family + LIME). DiCE recourse and
            # rule/IG methods have a different output shape and are served
            # elsewhere (vfairness_recourse); skip them here.
            if not (m.startswith("shap.") or m == "lime"):
                last_err = f"{m} not run in this engine (attribution only)"
                continue
            if m in ("shap.TreeExplainer", "shap.LinearExplainer"):
                model_for = model  # needs the estimator object
            elif m == "lime":
                # LIME accepts a fitted estimator (predict_proba) or a callable.
                model_for = model if model is not None else endpoint_fn
            else:  # KernelExplainer needs a callable
                model_for = endpoint_fn or (_model_callable(model) if model is not None else None)
            if model_for is None:
                last_err = f"{m} needs a model object or callable"
                continue
            try:
                # m has passed the shap.*/lime attribution filter above, so it
                # is an ExplainerMethod member; an unrecognised value would only
                # raise KeyError, which the surrounding except already handles.
                adapter = get_explainer(cast(ExplainerMethod, m))
                exp = adapter.explain_local(
                    model_for,
                    x,
                    background,
                    instance_id=f"row-{row_index}",
                    subject_id=str(payload.get("subject_id", "byo")),
                    model_hash="live",
                    data_hash="live",
                    feature_names=cols,
                )
                params = dict(exp.params or {})
                measured, measured_reason = _attribution_state(
                    params, background_is_instance, n_background
                )
                explanation = {
                    "method": exp.method,
                    "scope": "local",
                    "base_value": exp.base_value,
                    "prediction": exp.prediction,
                    "units": exp.units,
                    "fidelity": exp.fidelity,
                    "stability": exp.stability,
                    "attributions": [
                        {"feature": a.feature, "contribution": a.contribution}
                        for a in exp.attributions
                    ],
                    "library_versions": exp.library_versions,
                    # THE DISCLOSURE THE ADAPTER ALREADY COMPUTED, CARRIED INSTEAD OF
                    # DROPPED (2026-09-27). Measured before this change on a CSV whose
                    # column b is all NaN: the adapter set unattributed_features=['b'],
                    # attributions_complete=False, prediction_measured=False,
                    # base_value_measured=False, local_accuracy_residual=None and
                    # local_accuracy_ok=None, and warned twice; the envelope printed
                    # success=true with base_value NaN, prediction NaN and two finite
                    # contributions beside a NaN one, carrying NONE of those six
                    # fields, so the explanation read as usable. The full params dict
                    # travels as well, since to_db_row keeps it and a reader comparing
                    # the two surfaces must not find different amounts of it.
                    "params": params,
                    "background_rows": n_background,
                    "background_available": background_available,
                    "background_is_the_explained_instance": background_is_instance,
                    "attributions_measured": measured,
                }
                if measured_reason is not None:
                    explanation["attributions_measured_reason"] = measured_reason
                for field_name in _DISCLOSURE_FIELDS:
                    # .get with no default: an adapter that does not report a field
                    # reaches the reader as null, a could-not-check, never as False.
                    explanation[field_name] = params.get(field_name)
                envelope = {
                    "success": True,
                    "data": {
                        "routing": routing,
                        "method_used": exp.method,
                        "fallback_used": bool(routing and m != routing["primary"]),
                        "method_explicit": bool(payload.get("method")),
                        "model_type": model_type,
                        "explanation": explanation,
                    },
                }
                try:
                    # allow_nan=False is the proof, not the cleanup: _json_ready has
                    # already replaced every non-finite number with null, so this can
                    # only fire on something nobody anticipated.
                    rendered = json.dumps(_json_ready(envelope), allow_nan=False)
                except (TypeError, ValueError) as render_exc:
                    # A failure HERE must not fall through to the next candidate: the
                    # explanation ran, and reporting it as "no SHAP method ran" would
                    # turn a rendering failure into a claim about the explainer.
                    print(
                        json.dumps(
                            {
                                "success": False,
                                "error": (
                                    f"sidecar: {m} produced an explanation that could not be "
                                    f"rendered as valid JSON ({type(render_exc).__name__}: "
                                    f"{render_exc}). This is a could-not-report, not a finding "
                                    "about the model."
                                ),
                            }
                        )
                    )
                    return
                print(rendered)
                return
            except Exception as exc:
                last_err = f"{m}: {type(exc).__name__}: {exc}"
                continue
        print(json.dumps({"success": False, "error": f"sidecar: no SHAP method ran ({last_err})"}))
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"success": False, "error": f"sidecar: {type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
