"""BGL4 audit of batch A-xai-2: one grade, overturned, recorded as evidence.

``vfairness.xai.sidecar_cli.main`` was graded SEMI-PROVEN from probe evidence
alone ("measured on healthy input and refused on all 8 degenerate worlds"). The
probe never looked at what this unit actually reports: ``main`` returns ``None``
and writes its result to stdout, and ``scripts/surface_probe.py`` classifies a
``None`` return as ``("refused", "None")``, so every world including the healthy
one was recorded as a refusal and clause 1 of ``scripts/grade_from_probe.py``
never fired.

Swept by execution on 2026-09-27, one input class fabricates. Feed the CLI a CSV
whose features are NaN and the ``LinearShapExplainer`` adapter does the honest
thing: it warns twice and sets six disclosure fields on ``Explanation.params``
(``prediction_measured=False``, ``base_value_measured=False``,
``attributions_complete=False``, ``unattributed_features=['b']``,
``local_accuracy_residual=None``, ``local_accuracy_ok=None``, the last two being
the deliberate three-state "UNCHECKED, which is not a pass"). ``main`` then
prints a hand-listed subset of the Explanation that contains NONE of those six
fields, under ``"success": true``:

    {"success": true, ..., "explanation": {"method": "shap.LinearExplainer",
     "scope": "local", "base_value": NaN, "prediction": NaN,
     "units": "log-odds", "fidelity": null, "stability": null,
     "attributions": [{"feature": "a", "contribution": 1.3670970151450934},
                      {"feature": "b", "contribution": NaN},
                      {"feature": "c", "contribution": -0.010452743648849027}],
     "library_versions": {"shap": "0.52.0"}}}

The producer-side pin (``tests/test_bgl_stage2_s2g09.py`` asserts
``exp.params["local_accuracy_residual"] is None``) passes throughout. The
disclosure is correct where it is computed and gone where a reader sees it, so
the corrected grade is DEFECT OPEN at this unit, one layer up from the adapter.

Each test below asserted what an HONEST envelope would say and carried
``xfail(strict=True)`` while the defect was open. FIXED 2026-09-27 (BGL5): the
envelope now carries the adapter's ``params`` and the six disclosure fields, a
three-state ``attributions_measured`` with its reason, the background facts, and
non-finite numbers rendered as JSON ``null`` instead of the bare token ``NaN``.
All four markers had to go, because under ``xfail_strict`` an XPASS is a failure.
Each test keeps its subject and its docstring; the assertions now name the
corrected rendering, and the NaN checks became null checks for the same reason.
The last test PASSES and is the control: this same harness does produce
``success: false`` on the refusal paths, so the ``success: true`` above is a real
discrimination and not an artifact of how the harness calls the CLI.

AUDITOR NOTE, SUPERSEDED: this file recorded defects. They are now closed, and the
measured before and after live in src/vfairness/xai/sidecar_cli.py with the
sabotage record in tests/test_bgl5_toplevel_and_net.py.
"""

from __future__ import annotations

import base64
import io
import json
import math
from typing import Any

import numpy as np
import pytest

shap = pytest.importorskip("shap", reason="the sidecar's explainers need shap")
joblib = pytest.importorskip("joblib", reason="the sidecar loads models with joblib")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from vfairness.xai import sidecar_cli  # noqa: E402

_COLS = ["a", "b", "c"]

# Every disclosure the adapter computes and the envelope never carries.
_DISCLOSURE_KEYS = (
    "prediction_measured",
    "base_value_measured",
    "attributions_complete",
    "unattributed_features",
    "local_accuracy_residual",
    "local_accuracy_ok",
)


def _model_blob() -> str:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(X, y)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return base64.b64encode(buffer.getvalue()).decode()


def _csv(rows: np.ndarray, columns: list[str]) -> str:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(
            ",".join(
                "" if (isinstance(v, float) and math.isnan(v)) else repr(float(v)) for v in row
            )
        )
    return "\n".join(lines) + "\n"


def _run(monkeypatch, capsys, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Run the CLI exactly as `echo payload | python -m ...` does."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    sidecar_cli.main()
    raw = capsys.readouterr().out.strip().splitlines()[-1]
    return raw, json.loads(raw)


def _explanation(envelope: dict[str, Any]) -> dict[str, Any]:
    return envelope["data"]["explanation"]


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: "DEFECT OPEN. Every
# feature NaN: the adapter warns that the model could not be scored and records
# prediction_measured=False with local_accuracy_residual=None, and the envelope reports
# success=true with base_value=NaN, prediction=NaN and all three contributions NaN,
# carrying none of the six disclosure fields."
def test_a_wholly_unmeasurable_explanation_is_not_reported_as_a_success(
    monkeypatch, capsys
) -> None:
    payload = {
        "csv_data": _csv(np.full((60, 3), np.nan), _COLS),
        "feature_columns": _COLS,
        "model_base64": _model_blob(),
        "trust_input": True,
        "method": "shap.LinearExplainer",
    }
    _raw, envelope = _run(monkeypatch, capsys, payload)
    explanation = _explanation(envelope)
    numbers = [explanation["base_value"], explanation["prediction"]] + [
        a["contribution"] for a in explanation["attributions"]
    ]
    # Nothing at all was measured. Every number that was NaN is now JSON null, which
    # is what a reader gets for "no value", and the check is deliberately NOT guarded
    # by an `if` any more: a rendering change must not be able to make this vacuous.
    assert numbers == [None] * len(numbers), numbers
    # An honest envelope either refuses or names the reason. It now names it.
    assert all(key in explanation for key in _DISCLOSURE_KEYS), sorted(explanation)
    assert explanation["prediction_measured"] is False
    assert explanation["base_value_measured"] is False
    assert explanation["attributions_complete"] is False
    assert sorted(explanation["unattributed_features"]) == _COLS
    # The deliberate three-state: UNCHECKED, which is not a pass.
    assert explanation["local_accuracy_residual"] is None
    assert explanation["local_accuracy_ok"] is None
    assert explanation["attributions_measured"] is False
    assert "could-not-check" in explanation["attributions_measured_reason"]


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: "DEFECT OPEN, the
# partial case, which is worse because it looks usable: column b is unattributed
# (contribution NaN) beside two finite contributions, the adapter sets
# attributions_complete=False and warns that any ranking by magnitude sorts b LAST
# rather than omitting it, and the envelope drops that flag entirely under
# success=true."
def test_an_incomplete_explanation_says_so_where_a_reader_sees_it(monkeypatch, capsys) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    X[:, 1] = np.nan
    payload = {
        "csv_data": _csv(X, _COLS),
        "feature_columns": _COLS,
        "model_base64": _model_blob(),
        "trust_input": True,
        "method": "shap.LinearExplainer",
    }
    _raw, envelope = _run(monkeypatch, capsys, payload)
    explanation = _explanation(envelope)
    # The NaN contribution is rendered as null now, so the unattributed feature is
    # found by that instead. The subject is unchanged: the two finite contributions
    # beside it are what makes this case read as usable.
    unattributed = [a["feature"] for a in explanation["attributions"] if a["contribution"] is None]
    assert unattributed == ["b"], explanation["attributions"]
    finite = [
        a["contribution"] for a in explanation["attributions"] if a["contribution"] is not None
    ]
    assert len(finite) == 2, finite
    assert explanation["attributions_complete"] is False, sorted(explanation)
    assert explanation["unattributed_features"] == ["b"], explanation["unattributed_features"]
    assert explanation["attributions_measured"] is False
    assert "b" in explanation["attributions_measured_reason"]


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: "DEFECT OPEN, one
# layer further out: json.dumps writes bare NaN, which RFC 8259 does not allow, so the
# documented one-JSON-envelope contract is broken for any strict parser. Measured with
# node: JSON.parse FAILED: Unexpected token 'N', ...\"e_value\": NaN, \"pred\"... is
# not valid JSON. A non-Python consumer sees a malformed reply, not a refusal."
def test_the_envelope_is_parseable_json(monkeypatch, capsys) -> None:
    payload = {
        "csv_data": _csv(np.full((60, 3), np.nan), _COLS),
        "feature_columns": _COLS,
        "model_base64": _model_blob(),
        "trust_input": True,
        "method": "shap.LinearExplainer",
    }
    raw, _envelope = _run(monkeypatch, capsys, payload)

    def _reject(constant: str) -> float:
        raise ValueError(f"not valid JSON: bare {constant}")

    json.loads(raw, parse_constant=_reject)


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: "DEFECT OPEN, a
# second unmeasurable input class. main builds background = X[: min(50, len(X))], so a
# one-row CSV makes the background the explained instance itself and every SHAP value
# is exactly 0.0 by construction. Observed: success=true with contributions a=0.0,
# b=0.0, c=0.0 and base_value == prediction, which reads as 'no feature mattered'. main
# does compute len(X) >= 10, but only as the router's background_available, and an
# explicit payload['method'] skips the router altogether, so nothing consults it on
# that path."
def test_an_explanation_against_a_background_of_one_row_is_disclosed(monkeypatch, capsys) -> None:
    rng = np.random.default_rng(7)
    payload = {
        "csv_data": _csv(rng.normal(size=(1, 3)), _COLS),
        "feature_columns": _COLS,
        "model_base64": _model_blob(),
        "trust_input": True,
        "method": "shap.KernelExplainer",
    }
    _raw, envelope = _run(monkeypatch, capsys, payload)
    explanation = _explanation(envelope)
    contributions = [a["contribution"] for a in explanation["attributions"]]
    assert contributions == [0.0, 0.0, 0.0], contributions
    # The zeros are still reported, because the envelope is a faithful record of what
    # the explainer returned; what was missing is that they were 0.0 BY CONSTRUCTION.
    assert explanation["background_rows"] == 1
    assert explanation["background_is_the_explained_instance"] is True
    # Computed on this path too, which is the point: an explicit method skips the
    # router, so the sufficiency fact is now established above the dispatch.
    assert explanation["background_available"] is False
    assert envelope["data"]["method_explicit"] is True
    assert explanation["attributions_measured"] is False
    assert "0.0 by construction" in explanation["attributions_measured_reason"]


def test_the_refusal_paths_this_cli_does_get_right(monkeypatch, capsys) -> None:
    """CONTROL: the same harness does produce success=false, four ways.

    Without this, a single success=true observation could be an artifact of how
    the harness calls the CLI rather than a finding about the CLI.
    """
    blob = _model_blob()
    healthy = _csv(np.random.default_rng(1).normal(size=(60, 3)), _COLS)
    refusals = {
        "row_index out of range": {
            "csv_data": healthy,
            "feature_columns": _COLS,
            "model_base64": blob,
            "trust_input": True,
            "row_index": 9999,
        },
        "no csv_data at all": {"model_base64": blob, "trust_input": True},
        "an unparseable feature value": {
            "csv_data": "a,b,c\nzz,qq,ww\n",
            "feature_columns": _COLS,
            "model_base64": blob,
            "trust_input": True,
        },
        "a model blob with no trust opt-in": {
            "csv_data": healthy,
            "feature_columns": _COLS,
            "model_base64": blob,
        },
    }
    monkeypatch.delenv(sidecar_cli.TRUST_INPUT_ENV, raising=False)
    for name, payload in refusals.items():
        _raw, envelope = _run(monkeypatch, capsys, payload)
        assert envelope["success"] is False, name
        assert envelope["error"], name
